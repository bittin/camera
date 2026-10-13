// SPDX-License-Identifier: GPL-3.0-only

//! Ordered, coalescing preference writes, off the UI thread. Camera recovery
//! markers retain their synchronous durability barrier outside this module.

use crate::config::Config;
use cosmic::cosmic_config::{Config as Store, ConfigSet};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};

// Keep native field types: converting through JSON changes RON enum encoding.
// Recovery markers deliberately stay outside this worker; they must be durable
// before starting a camera and must never be overwritten by an older save.
macro_rules! preference_fields {
    ($apply:ident) => {
        $apply!(
            app_theme,
            overlay_effect,
            controls_position,
            default_mode,
            save_folder_name,
            video_settings,
            photo_settings,
            last_video_encoder_index,
            bug_report_url,
            mirror_preview,
            mirror_captures,
            bitrate_preset,
            virtual_camera_enabled,
            photo_output_format,
            demosaic_method,
            motion_photo,
            save_burst_raw,
            burst_mode_setting,
            record_audio,
            audio_encoder,
            composition_guide,
            timelapse_interval,
            haptic_feedback,
            photo_aspect_ratio,
            preview_fit_to_view,
            key_bindings
        );
    };
}

type Write = Box<dyn FnOnce(&Store) -> Result<(), cosmic::cosmic_config::Error> + Send>;

#[derive(Default)]
struct Pending {
    writes: HashMap<&'static str, Write>,
    writing: bool,
    stopping: bool,
}

type Shared = Arc<(Mutex<Pending>, Condvar)>;

fn wait_for_writes(shared: &Shared) {
    let (lock, ready) = &**shared;
    let mut pending = lock.lock().unwrap();
    while pending.writing || !pending.writes.is_empty() {
        pending = ready.wait(pending).unwrap();
    }
}

struct Writer {
    shared: Shared,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Writer {
    fn start(store: Store) -> std::io::Result<Self> {
        let shared = Shared::default();
        let worker = shared.clone();
        let thread = std::thread::Builder::new()
            .name("camera-config".into())
            .spawn(move || {
                let (lock, ready) = &*worker;
                loop {
                    let mut pending = lock.lock().unwrap();
                    while pending.writes.is_empty() && !pending.stopping {
                        pending = ready.wait(pending).unwrap();
                    }
                    if pending.writes.is_empty() && pending.stopping {
                        break;
                    }
                    let writes = std::mem::take(&mut pending.writes);
                    pending.writing = true;
                    drop(pending);
                    for (key, write) in writes {
                        if let Err(error) = write(&store) {
                            tracing::error!(key, ?error, "Failed to persist preference");
                        }
                    }
                    let mut pending = lock.lock().unwrap();
                    pending.writing = false;
                    ready.notify_all();
                }
            })?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    fn submit(&self, writes: HashMap<&'static str, Write>) {
        let (lock, ready) = &*self.shared;
        // Newer values replace queued values of the same key. In-flight writes
        // finish first, so rapid toggles cannot publish an older value last.
        lock.lock().unwrap().writes.extend(writes);
        // Reload waiters share this condition variable with the writer. A
        // single wake can select a reload and leave queued writes asleep.
        ready.notify_all();
    }

    #[cfg(test)]
    fn flush(&self) {
        wait_for_writes(&self.shared);
    }

    fn stop(&self) {
        let (lock, ready) = &*self.shared;
        lock.lock().unwrap().stopping = true;
        ready.notify_all();
    }

    fn finish(mut self) -> Result<(), String> {
        self.stop();
        self.thread
            .take()
            .unwrap()
            .join()
            .map_err(|_| "Preference writer panicked during shutdown".to_string())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // Dropping UI state must not block; the worker drains owned writes.
        self.stop();
    }
}

#[derive(Default)]
pub(crate) struct ConfigPersistence {
    writer: Option<Writer>,
    revision: u64,
    baseline: Option<Config>,
}

impl ConfigPersistence {
    pub(crate) fn invalidate_reload(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    pub(crate) fn prepare(&mut self, config: &Config) {
        if self.baseline.is_none() {
            self.baseline = Some(config.clone());
        }
    }

    pub(crate) fn save(&mut self, store: &Store, config: &Config) {
        let before = self.baseline.take().unwrap_or_default();
        self.enqueue(store, &before, config);
        self.baseline = Some(config.clone());
    }

    pub(crate) fn enqueue(&mut self, store: &Store, before: &Config, after: &Config) {
        let mut writes: HashMap<&'static str, Write> = HashMap::new();
        macro_rules! changed {
            ($($field:ident),* $(,)?) => { $(
                if before.$field != after.$field {
                    let value = after.$field.clone();
                    writes.insert(stringify!($field), Box::new(move |store| {
                        store.set(stringify!($field), &value)
                    }));
                }
            )* };
        }
        preference_fields!(changed);
        if writes.is_empty() {
            return;
        }
        if self.writer.is_none() {
            match Writer::start(store.clone()) {
                Ok(writer) => self.writer = Some(writer),
                Err(error) => {
                    tracing::error!(?error, "Could not start preference writer");
                    return;
                }
            }
        }
        self.revision = self.revision.wrapping_add(1);
        self.writer.as_ref().unwrap().submit(writes);
    }

    pub(crate) fn reload(
        &mut self,
        store: Store,
    ) -> cosmic::Task<cosmic::Action<crate::app::state::Message>> {
        use cosmic::cosmic_config::CosmicConfigEntry;
        self.revision = self.revision.wrapping_add(1);
        let revision = self.revision;
        let shared = self.writer.as_ref().map(|writer| writer.shared.clone());
        cosmic::Task::perform(
            async move {
                let result = tokio::task::spawn_blocking(move || {
                    // Watcher snapshots can predate local saves, including a
                    // coalesced return to the original value (no echo). Read
                    // authoritative disk state only after our writer drains.
                    if let Some(shared) = shared {
                        wait_for_writes(&shared);
                    }
                    Config::get_entry(&store)
                        .map(Box::new)
                        .map_err(|(errors, _)| format!("Could not reload preferences: {errors:?}"))
                })
                .await
                .map_err(|error| error.to_string())
                .and_then(|result| result);
                (revision, result)
            },
            |(revision, result)| {
                cosmic::Action::App(crate::app::state::Message::ConfigReloaded(revision, result))
            },
        )
    }

    pub(crate) fn accept_reload(&mut self, revision: u64, config: &Config) -> bool {
        // A newer selection or reload request makes this disk read stale.
        if revision != self.revision {
            return false;
        }
        self.baseline = Some(config.clone());
        true
    }

    pub(crate) async fn finish(mut self) -> Result<(), String> {
        if let Some(writer) = self.writer.take() {
            tokio::task::spawn_blocking(move || writer.finish())
                .await
                .map_err(|error| error.to_string())??;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn flush(&self) {
        if let Some(writer) = &self.writer {
            writer.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::state::{AppModel, Message};
    use crate::config::PhotoOutputFormat;
    use cosmic::cosmic_config::CosmicConfigEntry;
    use std::collections::HashSet;
    use std::sync::mpsc;
    use std::time::Duration;

    struct Fixture {
        root: std::path::PathBuf,
        store: Store,
    }

    impl Fixture {
        fn new() -> Self {
            let root =
                std::env::temp_dir().join(format!("camera-config-worker-{}", uuid::Uuid::new_v4()));
            let store =
                Store::with_custom_path("camera-test", Config::VERSION, root.clone()).unwrap();
            Config::default().write_entry(&store).unwrap();
            Self { root, store }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }

    // Stall the real writer, not the UI, with a bounded channel rendezvous.
    fn stall(writer: &Writer) -> mpsc::Sender<()> {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let write: Write = Box::new(move |_| {
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            Ok(())
        });
        writer.submit(HashMap::from([("test-gate", write)]));
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        release_tx
    }

    #[test]
    fn blocked_writer_does_not_block_handlers_and_coalesces_latest_values() {
        let fixture = Fixture::new();
        let writer = Writer::start(fixture.store.clone()).unwrap();
        let release = stall(&writer);
        let mut app = AppModel {
            config_handler: Some(fixture.store.clone()),
            config_persistence: ConfigPersistence {
                writer: Some(writer),
                ..Default::default()
            },
            ..Default::default()
        };
        for index in 0..40 {
            drop(app.update(Message::ToggleMotionPhoto(index % 2 == 0)));
            drop(app.update(Message::SelectPhotoOutputFormat(index % 2)));
        }
        // Reaching this assertion before releasing the writer proves no UI
        // handler waited for it. Only two latest-key values may be queued.
        let shared = &app.config_persistence.writer.as_ref().unwrap().shared;
        assert_eq!(shared.0.lock().unwrap().writes.len(), 2);
        fixture
            .store
            .set("pending_camera_path", Some("new-camera"))
            .unwrap();
        fixture
            .store
            .set("bug_report_url", "external-update")
            .unwrap();
        release.send(()).unwrap();
        app.config_persistence.flush();
        let saved = Config::get_entry(&fixture.store).unwrap();
        assert!(!saved.motion_photo);
        assert_eq!(saved.photo_output_format, PhotoOutputFormat::Png);
        assert_eq!(saved.pending_camera_path.as_deref(), Some("new-camera"));
        assert_eq!(saved.bug_report_url, "external-update");
    }

    async fn dispatch_reload(app: &mut AppModel, task: cosmic::Task<cosmic::Action<Message>>) {
        use futures::StreamExt;
        let mut stream = cosmic::iced::runtime::task::into_stream(task).unwrap();
        while let Some(action) = stream.next().await {
            if let cosmic::iced::runtime::Action::Output(cosmic::Action::App(message)) = action {
                drop(app.update(message));
            }
        }
    }

    #[tokio::test]
    async fn stale_config_echo_cannot_revert_a_newer_selection() {
        let fixture = Fixture::new();
        let mut app = AppModel {
            config_handler: Some(fixture.store.clone()),
            ..Default::default()
        };
        drop(app.update(Message::SelectPhotoOutputFormat(3)));
        // The old snapshot is a notification, never authoritative state.
        let task = app.update(Message::UpdateConfig(Config::default()));
        dispatch_reload(&mut app, task).await;
        assert_eq!(app.config.photo_output_format, PhotoOutputFormat::Avif);
        // A later genuine external edit is re-read and applied.
        fixture.store.set("motion_photo", true).unwrap();
        let task = app.update(Message::UpdateConfig(Config::default()));
        dispatch_reload(&mut app, task).await;
        assert_eq!(app.config.photo_output_format, PhotoOutputFormat::Avif);
        assert!(app.config.motion_photo);
        // Fence a result if a new user selection occurs before it is applied.
        let task = app.update(Message::UpdateConfig(Config::default()));
        drop(app.update(Message::SelectPhotoOutputFormat(1)));
        dispatch_reload(&mut app, task).await;
        assert_eq!(app.config.photo_output_format, PhotoOutputFormat::Png);
        app.config_persistence.flush();
    }

    #[tokio::test]
    async fn finish_drains_preferences_before_returning() {
        let fixture = Fixture::new();
        let mut persistence = ConfigPersistence::default();
        persistence.prepare(&Config::default());
        let config = Config {
            motion_photo: true,
            photo_output_format: PhotoOutputFormat::Avif,
            ..Default::default()
        };
        persistence.save(&fixture.store, &config);
        persistence.finish().await.unwrap();
        assert_eq!(Config::get_entry(&fixture.store).unwrap(), config);
    }

    #[tokio::test]
    async fn external_edit_after_coalesced_return_to_default_is_applied() {
        use futures::StreamExt;
        let fixture = Fixture::new();
        let writer = Writer::start(fixture.store.clone()).unwrap();
        let release = stall(&writer);
        let mut app = AppModel {
            config_handler: Some(fixture.store.clone()),
            config_persistence: ConfigPersistence {
                writer: Some(writer),
                ..Default::default()
            },
            ..Default::default()
        };
        drop(app.update(Message::ToggleMotionPhoto(true)));
        drop(app.update(Message::ToggleMotionPhoto(false)));
        release.send(()).unwrap();
        app.config_persistence.flush();
        fixture.store.set("motion_photo", true).unwrap();
        let external = Config::get_entry(&fixture.store).unwrap();
        let task = app.update(Message::UpdateConfig(external));
        if let Some(mut stream) = cosmic::iced::runtime::task::into_stream(task) {
            while let Some(cosmic::iced::runtime::Action::Output(cosmic::Action::App(message))) =
                stream.next().await
            {
                drop(app.update(message));
            }
        }
        assert!(
            app.config.motion_photo,
            "a coalesced no-op has no watcher echo and must not permanently hide external edits"
        );
    }

    #[tokio::test]
    async fn closing_waits_for_accepted_preferences_before_exit_completion() {
        use futures::{FutureExt, StreamExt};
        let fixture = Fixture::new();
        let writer = Writer::start(fixture.store.clone()).unwrap();
        let release = stall(&writer);
        let mut app = AppModel {
            config_handler: Some(fixture.store.clone()),
            config_persistence: ConfigPersistence {
                writer: Some(writer),
                ..Default::default()
            },
            ..Default::default()
        };
        drop(app.update(Message::ToggleMotionPhoto(true)));
        let mut stream =
            cosmic::iced::runtime::task::into_stream(app.update(Message::WindowClose)).unwrap();
        assert!(app.pending_close);
        assert!(
            stream.next().now_or_never().is_none(),
            "close completion must not race an accepted save"
        );
        release.send(()).unwrap();
        let action = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            action,
            cosmic::iced::runtime::Action::Output(cosmic::Action::App(
                Message::ShutdownRecorderStopped(Ok(()))
            ))
        ));
        assert!(Config::get_entry(&fixture.store).unwrap().motion_photo);
        // Do not deliver this terminal completion: production legitimately _exits.
    }

    #[test]
    fn every_config_field_has_an_explicit_persistence_policy() {
        let mut fields = HashSet::new();
        macro_rules! record {
            ($($field:ident),* $(,)?) => { $(fields.insert(stringify!($field).to_string());)* };
        }
        preference_fields!(record);
        fields.extend(
            [
                "last_camera_path",
                "pending_camera_path",
                "failed_camera_paths",
            ]
            .map(str::to_owned),
        );
        let serialized = serde_json::to_value(Config::default()).unwrap();
        let expected: HashSet<_> = serialized.as_object().unwrap().keys().cloned().collect();
        assert_eq!(fields, expected);
    }
}
