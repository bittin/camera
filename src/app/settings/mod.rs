// SPDX-License-Identifier: GPL-3.0-only

//! Settings UI module
//!
//! This module handles settings drawer UI:
//! - Main settings view

mod touch_toggle;
pub mod view;
#[cfg(test)]
pub(crate) mod widget_test_support;

// Preference handlers live here alongside their settings views.

impl crate::app::state::AppModel {
    pub(crate) fn handle_toggle_motion_photo(
        &mut self,
        enabled: bool,
    ) -> cosmic::Task<cosmic::Action<crate::app::state::Message>> {
        // Disabled controls cannot send this, but queued messages can. Ignore
        // both directions for PNG/DNG so neither can erase the saved preference.
        if !self.motion_photo_format_compatible() {
            tracing::info!(enabled, format = ?self.config.photo_output_format, "Rejected Motion Photo setting toggle: incompatible photo format");
            return cosmic::Task::none();
        }
        tracing::info!(enabled, previous = self.config.motion_photo, format = ?self.config.photo_output_format, "Accepted Motion Photo setting toggle");
        self.config.motion_photo = enabled;
        self.persist_config_async();
        cosmic::Task::none()
    }
}

#[cfg(test)]
mod tests {
    use crate::app::state::{AppModel, Message};
    use crate::config::{Config, PhotoOutputFormat};
    use cosmic::cosmic_config::{Config as Store, CosmicConfigEntry};

    #[test]
    fn settings_updates_do_not_wait_for_disk_writes() {
        let root =
            std::env::temp_dir().join(format!("camera-settings-latency-{}", uuid::Uuid::new_v4()));
        let store = Store::with_custom_path("camera-test", Config::VERSION, root.clone()).unwrap();
        let mut app = AppModel {
            config_handler: Some(store),
            ..AppModel::default()
        };
        let start = std::time::Instant::now();
        for index in 0..20 {
            drop(app.update(Message::ToggleMotionPhoto(index % 2 == 0)));
            drop(app.update(Message::SelectPhotoOutputFormat(index % 2)));
        }
        let elapsed = start.elapsed();
        eprintln!("40 settings updates: {elapsed:?}");
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "settings handlers blocked on filesystem I/O: {elapsed:?}"
        );
        app.config_persistence.flush();
        drop(app);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn motion_photo_survives_photo_format_changes() {
        let root = std::env::temp_dir().join(format!(
            "camera-motion-photo-formats-{}",
            uuid::Uuid::new_v4()
        ));
        let store = Store::with_custom_path("camera-test", Config::VERSION, root.clone()).unwrap();
        let mut app = AppModel {
            config: Config {
                motion_photo: true,
                ..Config::default()
            },
            config_handler: Some(store),
            ..AppModel::default()
        };
        app.config
            .write_entry(app.config_handler.as_ref().unwrap())
            .unwrap();
        for (index, format) in PhotoOutputFormat::ALL.into_iter().enumerate() {
            let _ = app.update(Message::SelectPhotoOutputFormat(index));
            assert_eq!(app.config.photo_output_format, format);
            assert!(app.config.motion_photo);
            app.config_persistence.flush();
            let persisted = Config::get_entry(app.config_handler.as_ref().unwrap()).unwrap();
            assert_eq!(persisted.photo_output_format, format);
            assert!(persisted.motion_photo);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reset_all_settings_persists_disabled_motion_photo() {
        let root = std::env::temp_dir().join(format!(
            "camera-motion-photo-reset-{}",
            uuid::Uuid::new_v4()
        ));
        let store = Store::with_custom_path("camera-test", Config::VERSION, root.clone()).unwrap();
        let mut app = AppModel {
            config: Config {
                motion_photo: true,
                ..Config::default()
            },
            config_handler: Some(store),
            ..AppModel::default()
        };
        app.config
            .write_entry(app.config_handler.as_ref().unwrap())
            .unwrap();
        let _ = app.update(Message::ResetAllSettings);
        app.config_persistence.flush();
        assert!(!app.config.motion_photo);
        assert!(
            !Config::get_entry(app.config_handler.as_ref().unwrap())
                .unwrap()
                .motion_photo
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incompatible_motion_photo_queued_messages_leave_memory_and_store_unchanged() {
        let root = std::env::temp_dir().join(format!(
            "camera-motion-photo-disabled-{}",
            uuid::Uuid::new_v4()
        ));
        let store = Store::with_custom_path("camera-test", Config::VERSION, root.clone()).unwrap();
        let mut app = AppModel {
            config_handler: Some(store),
            ..AppModel::default()
        };
        for format in [PhotoOutputFormat::Png, PhotoOutputFormat::Dng] {
            for saved in [true, false] {
                app.config.photo_output_format = format;
                app.config.motion_photo = saved;
                app.config
                    .write_entry(app.config_handler.as_ref().unwrap())
                    .unwrap();
                let before = serde_json::to_value(&app.config).unwrap();
                for queued in [true, false] {
                    let _ = app.update(Message::ToggleMotionPhoto(queued));
                    let _ = app.update(Message::ToggleMotionPhotoCapture);
                    app.config_persistence.flush();
                    assert_eq!(serde_json::to_value(&app.config).unwrap(), before);
                    let persisted =
                        Config::get_entry(app.config_handler.as_ref().unwrap()).unwrap();
                    assert_eq!(serde_json::to_value(persisted).unwrap(), before);
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn motion_photo_message_persists_requested_value() {
        let root =
            std::env::temp_dir().join(format!("camera-motion-photo-{}", uuid::Uuid::new_v4()));
        let store = Store::with_custom_path("camera-test", Config::VERSION, root.clone()).unwrap();
        let mut app = AppModel {
            config_handler: Some(store),
            ..AppModel::default()
        };
        for enabled in [true, true, false] {
            let _ = app.update(Message::ToggleMotionPhoto(enabled));
            assert_eq!(app.config.motion_photo, enabled);
            app.config_persistence.flush();
            assert_eq!(
                Config::get_entry(app.config_handler.as_ref().unwrap())
                    .unwrap()
                    .motion_photo,
                enabled
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
