// SPDX-License-Identifier: GPL-3.0-only

//! System handlers
//!
//! Handles gallery operations, filter selection, settings, recovery, bug reports,
//! and QR code detection.

use crate::app::state::{AppModel, FilterType, Message, RecordingState};
use cosmic::Task;
use tracing::{error, info};

fn file_manager_dbus_args(file_uri: &str) -> Vec<String> {
    [
        "--session".to_string(),
        "--print-reply".to_string(),
        "--reply-timeout=5000".to_string(),
        "--dest=org.freedesktop.FileManager1".to_string(),
        "--type=method_call".to_string(),
        "/org/freedesktop/FileManager1".to_string(),
        "org.freedesktop.FileManager1.ShowItems".to_string(),
        format!("array:string:{file_uri}"),
        "string:".to_string(),
    ]
    .into()
}

fn portal_gallery_item(running_in_flatpak: bool, last_media_path: Option<&str>) -> Option<&str> {
    running_in_flatpak.then_some(last_media_path).flatten()
}

async fn open_gallery_item_via_portal(path: &std::path::Path) -> Result<(), String> {
    use ashpd::desktop::open_uri::OpenDirectoryRequest;
    use std::os::fd::AsFd;

    let file = std::fs::File::open(path)
        .map_err(|err| format!("Failed to open media file for portal request: {err}"))?;
    let request = OpenDirectoryRequest::default()
        .send(&file.as_fd())
        .await
        .map_err(|err| format!("OpenDirectory portal request failed: {err}"))?;
    request
        .response()
        .map_err(|err| format!("OpenDirectory portal rejected request: {err}"))
}

impl AppModel {
    // =========================================================================
    // Gallery Handlers
    // =========================================================================

    pub(crate) fn handle_open_gallery(&self) -> Task<cosmic::Action<Message>> {
        let running_in_flatpak = std::env::var_os("FLATPAK_ID").is_some();

        // OpenDirectory accepts the latest media file itself, then opens its
        // containing directory and asks the file manager to select it. Unlike a
        // direct FileManager1 call, the portal reliably crosses the sandbox.
        if let Some(path) = portal_gallery_item(running_in_flatpak, self.last_media_path.as_deref())
        {
            let path = std::path::PathBuf::from(path);
            let task_path = path.clone();
            info!(path = %path.display(), "Opening gallery item through XDG portal");
            return Task::perform(
                async move {
                    let result = open_gallery_item_via_portal(&task_path).await;
                    (task_path, result)
                },
                |(path, result)| {
                    cosmic::Action::App(Message::GalleryPortalOpenCompleted(path, result))
                },
            );
        }

        // Native builds retain file pre-selection through FileManager1.
        if let Some(ref path) = self.last_media_path {
            info!(path = %path, "Opening gallery with file pre-selected");
            if Self::show_in_file_manager(path).is_ok() {
                return Task::none();
            }
            info!("show_in_file_manager failed, falling back to directory open");
        }

        let photo_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        info!(path = %photo_dir.display(), "Opening gallery directory");

        if let Err(e) = open::that(&photo_dir) {
            error!(error = %e, path = %photo_dir.display(), "Failed to open gallery directory");
        } else {
            info!("Gallery opened successfully");
        }
        Task::none()
    }

    pub(crate) fn handle_gallery_portal_open_completed(
        &self,
        path: std::path::PathBuf,
        result: Result<(), String>,
    ) -> Task<cosmic::Action<Message>> {
        if let Err(err) = result {
            error!(error = %err, path = %path.display(), "Failed to open gallery item through XDG portal");
            if let Some(parent) = path.parent()
                && let Err(fallback_err) = open::that(parent)
            {
                error!(
                    error = %fallback_err,
                    path = %parent.display(),
                    "Failed to open gallery item's parent directory"
                );
            }
        } else {
            info!(path = %path.display(), "Gallery item opened through XDG portal");
        }
        Task::none()
    }

    pub(crate) fn handle_refresh_gallery_thumbnail(&self) -> Task<cosmic::Action<Message>> {
        let photos_dir = crate::app::get_photo_directory(&self.config.save_folder_name);
        let videos_dir = crate::app::get_video_directory(&self.config.save_folder_name);
        Task::perform(
            async move { crate::storage::load_latest_thumbnail(photos_dir, videos_dir).await },
            |handle| cosmic::Action::App(Message::GalleryThumbnailLoaded(handle)),
        )
    }

    pub(crate) fn handle_gallery_thumbnail_loaded(
        &mut self,
        data: Option<crate::storage::GalleryThumbnailData>,
    ) -> Task<cosmic::Action<Message>> {
        if let Some((handle, rgba, width, height, path)) = data {
            self.gallery_thumbnail = Some(handle);
            self.gallery_thumbnail_rgba = Some((rgba, width, height));
            self.last_media_path = Some(path.display().to_string());
        } else {
            self.gallery_thumbnail = None;
            self.gallery_thumbnail_rgba = None;
        }
        Task::none()
    }

    // =========================================================================
    // Filter Handlers
    // =========================================================================

    pub(crate) fn handle_select_filter(
        &mut self,
        filter: FilterType,
    ) -> Task<cosmic::Action<Message>> {
        self.selected_filter = filter;
        // Update the shared atomic so the recording pusher picks up the change
        self.recording_filter_code.store(
            filter.gpu_filter_code(),
            std::sync::atomic::Ordering::Relaxed,
        );
        info!("Filter selected: {:?}", filter);

        // Update virtual camera filter if streaming
        if self.virtual_camera.is_streaming() {
            self.virtual_camera.set_filter(filter);
        }

        // Close the filter drawer after selection
        self.core.window.show_context = false;

        Task::none()
    }

    // =========================================================================
    // Settings Handlers
    // =========================================================================

    pub(crate) fn handle_update_config(
        &mut self,
        _config: crate::config::Config,
    ) -> Task<cosmic::Action<Message>> {
        // The subscription snapshot can be older than a queued local save.
        // Re-read off-thread after ordered writes and fence stale results.
        match self.config_handler.clone() {
            Some(store) => self.config_persistence.reload(store),
            None => Task::none(),
        }
    }

    pub(crate) fn handle_config_reloaded(
        &mut self,
        revision: u64,
        result: Result<Box<crate::config::Config>, String>,
    ) -> Task<cosmic::Action<Message>> {
        match result {
            Ok(config) if self.config_persistence.accept_reload(revision, &config) => {
                self.config = *config;
                crate::app::overlay_style::init_overlay_effect(self.config.overlay_effect);
            }
            Err(error) => error!(%error, "Failed to reload config"),
            _ => {}
        }
        Task::none()
    }

    pub(crate) fn handle_set_app_theme(&mut self, index: usize) -> Task<cosmic::Action<Message>> {
        use crate::config::AppTheme;

        let app_theme = match index {
            0 => AppTheme::System,
            1 => AppTheme::Dark,
            2 => AppTheme::Light,
            _ => return Task::none(),
        };

        info!(?app_theme, "Setting application theme");
        self.config.app_theme = app_theme;

        self.persist_config_async();

        cosmic::command::set_theme(app_theme.theme())
    }

    pub(crate) fn handle_set_overlay_effect(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::OverlayEffect;

        // Indexes `available()` — the same slice the dropdown was built from,
        // which drops `System` off-COSMIC. Indexing `ALL` here would silently
        // shift every choice by one on those desktops.
        let effect = match OverlayEffect::from_dropdown_index(index) {
            Some(e) => e,
            None => return Task::none(),
        };

        info!(?effect, "Setting overlay effect");
        self.config.overlay_effect = effect;

        // The draw-time colour roots read the global, not `self.config`.
        crate::app::overlay_style::init_overlay_effect(effect);

        self.persist_config_async();

        Task::none()
    }

    pub(crate) fn handle_set_controls_position(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        let Some(&position) = crate::config::ControlsPosition::ALL.get(index) else {
            return Task::none();
        };

        info!(?position, "Setting controls panel position");
        self.config.controls_position = position;

        self.persist_config_async();

        Task::none()
    }

    pub(crate) fn handle_select_default_mode(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::app::state::CameraMode;

        // Build the same visible modes list as the dropdown
        let mut visible_modes = vec![CameraMode::Photo, CameraMode::Video, CameraMode::Timelapse];
        if self.config.virtual_camera_enabled {
            visible_modes.push(CameraMode::Virtual);
        }

        let mode = match visible_modes.get(index) {
            Some(&m) => m,
            None => return Task::none(),
        };

        info!(?mode, "Setting default launch mode");
        self.config.default_mode = mode;

        self.persist_config_async();

        Task::none()
    }

    /// Re-apply the active theme after a COSMIC theme config edit (accent
    /// color, palette, light/dark mode switch from outside the app, etc.).
    ///
    /// `AppTheme::theme()` re-reads from `cosmic_config` each call, so
    /// running the same selector through `set_theme` again gives the app
    /// the just-edited palette. Previously the app only re-applied on
    /// light/dark *toggles* (via the portal subscription on non-COSMIC, or
    /// implicitly via libcosmic's own restart heuristic on COSMIC), which
    /// is why an accent change in the *current* mode looked stuck until
    /// the user toggled to the other mode and back (issue #290).
    pub(crate) fn handle_cosmic_theme_changed(&mut self) -> Task<cosmic::Action<Message>> {
        info!("COSMIC theme config changed — re-applying active theme");
        cosmic::command::set_theme(self.config.app_theme.theme())
    }

    pub(crate) fn handle_portal_color_scheme_changed(
        &mut self,
        is_dark: bool,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::AppTheme;

        // Only apply if user wants to follow system theme
        if self.config.app_theme != AppTheme::System {
            return Task::none();
        }

        info!(is_dark, "Portal color scheme changed, updating theme");
        let theme = if is_dark {
            cosmic::Theme::dark()
        } else {
            cosmic::Theme::light()
        };
        cosmic::command::set_theme(theme)
    }

    pub(crate) fn handle_select_audio_device(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        if index < self.available_audio_devices.len() {
            info!(index, "Selected audio device index");
            self.current_audio_device_index = index;
        }
        self.sync_audio_probe();
        Task::none()
    }

    pub(crate) fn handle_audio_level_tick(&mut self) -> Task<cosmic::Action<Message>> {
        // Recorder wins over probe — they never coexist by design.
        let source = if self.recording.is_recording() {
            self.recording.audio_levels().cloned()
        } else {
            self.probe_audio_levels.clone()
        };

        self.audio_levels_snapshot =
            source.and_then(|arc| arc.lock().ok().map(|guard| guard.clone()));
        Task::none()
    }

    pub(crate) fn handle_audio_list_changed(
        &mut self,
        new_devices: Vec<crate::backends::audio::AudioDevice>,
    ) -> Task<cosmic::Action<Message>> {
        info!(
            old_count = self.available_audio_devices.len(),
            new_count = new_devices.len(),
            "Audio device list changed (hotplug event)"
        );

        // If enumeration returned empty but we had devices before, treat it as
        // an enumeration failure (e.g. pw-dump failed) rather than all devices
        // being removed. Preserve the existing list to avoid killing recordings.
        if new_devices.is_empty() && !self.available_audio_devices.is_empty() {
            info!("Audio enumeration returned empty list, keeping existing devices");
            return Task::none();
        }

        // Try to keep the current device selected if it's still available
        let current_still_available = if let Some(current) = self
            .available_audio_devices
            .get(self.current_audio_device_index)
        {
            new_devices
                .iter()
                .position(|d| d.serial == current.serial && d.name == current.name)
        } else {
            None
        };

        // Stop recording if the audio input used for recording was disconnected
        // Only do this when we have a non-empty new device list (confirmed enumeration)
        // and the specific device is genuinely missing from it
        if current_still_available.is_none()
            && !new_devices.is_empty()
            && self.config.record_audio
            && self.recording.is_recording()
        {
            info!("Audio input disconnected during recording, stopping recording gracefully");
            if let Some(sender) = self.recording.take_stop_sender() {
                let _ = sender.send(());
            }
            self.recording = RecordingState::Idle;
        }

        self.available_audio_devices = new_devices;
        self.audio_dropdown_options = self
            .available_audio_devices
            .iter()
            .map(|dev| {
                if dev.is_default {
                    format!("{} (Default)", dev.name)
                } else {
                    dev.name.clone()
                }
            })
            .collect();

        if let Some(new_index) = current_still_available {
            self.current_audio_device_index = new_index;
        } else if !self.available_audio_devices.is_empty() {
            // Reset to first device (default is sorted first)
            self.current_audio_device_index = 0;
        }

        self.sync_audio_probe();
        Task::none()
    }

    pub(crate) fn handle_select_video_encoder(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        if index < self.available_video_encoders.len() {
            info!(index, encoder = %self.available_video_encoders[index].display_name, "Selected video encoder");
            self.current_video_encoder_index = index;

            self.config.last_video_encoder_index = Some(index);
            self.persist_config_async();
        }
        Task::none()
    }

    pub(crate) fn handle_select_photo_output_format(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::PhotoOutputFormat;

        if index < PhotoOutputFormat::ALL.len() {
            let format = PhotoOutputFormat::ALL[index];
            info!(?format, "Selected photo output format");
            self.config.photo_output_format = format;

            self.persist_config_async();
        }
        Task::none()
    }

    pub(crate) fn handle_toggle_record_audio(&mut self) -> Task<cosmic::Action<Message>> {
        if self.recording.is_recording() {
            return Task::none();
        }

        self.config.record_audio = !self.config.record_audio;
        info!(
            record_audio = self.config.record_audio,
            "Toggled record audio"
        );

        self.persist_config_async();
        self.sync_audio_probe();
        Task::none()
    }

    pub(crate) fn handle_select_audio_encoder(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::AudioEncoder;

        if index < AudioEncoder::ALL.len() {
            let encoder = AudioEncoder::ALL[index];
            info!(?encoder, "Selected audio encoder");
            self.config.audio_encoder = encoder;

            self.persist_config_async();
        }
        Task::none()
    }

    pub(crate) fn handle_toggle_save_burst_raw(&mut self) -> Task<cosmic::Action<Message>> {
        self.config.save_burst_raw = !self.config.save_burst_raw;
        info!(
            save_burst_raw = self.config.save_burst_raw,
            "Toggled save burst raw frames"
        );

        self.persist_config_async();
        Task::none()
    }

    pub(crate) fn handle_select_composition_guide(
        &mut self,
        index: usize,
    ) -> Task<cosmic::Action<Message>> {
        use crate::config::CompositionGuide;
        if let Some(&guide) = CompositionGuide::ALL.get(index) {
            self.config.composition_guide = guide;
            info!(?guide, "Selected composition guide");

            self.persist_config_async();
        }
        Task::none()
    }

    pub(crate) fn handle_set_demosaic_method(
        &mut self,
        method: crate::shaders::DemosaicMethod,
    ) -> Task<cosmic::Action<Message>> {
        self.config.demosaic_method = method;
        self.persist_config_async();
        Task::none()
    }

    pub(crate) fn handle_reset_all_settings(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Resetting all settings to defaults");
        // Snapshot the pre-reset animated values so the fit/fill change
        // animates rather than snaps when reset flips preview_fit_to_view
        // (or drops out of View mode).
        let from_fit = self.capture_fit_state();

        let previous = std::mem::take(&mut self.config);
        self.config_persistence.invalidate_reload();
        // Recovery markers are intentionally synchronous. Only clear markers
        // that were set; ordinary defaults go through the preference worker.
        if let Some(store) = self.config_handler.as_ref() {
            use cosmic::cosmic_config::ConfigSet;
            macro_rules! clear_recovery {
                ($($field:ident),* $(,)?) => { $(
                    if previous.$field != self.config.$field
                        && let Err(error) = store.set(stringify!($field), &self.config.$field)
                    {
                        error!(?error, "Failed to clear recovery marker");
                    }
                )* };
            }
            clear_recovery!(last_camera_path, pending_camera_path, failed_camera_paths);
        }

        self.persist_config_async();

        // Reset UI state that shadows config values
        self.current_video_encoder_index = 0;
        self.selected_filter = FilterType::default();
        self.recording_filter_code.store(
            self.selected_filter.gpu_filter_code(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.photo_aspect_ratio = self.config.photo_aspect_ratio;
        self.preview_fit_to_view = self.config.preview_fit_to_view;
        self.zoom_level = 1.0;
        // View is a passive UI mode (no capture controls). Reset shouldn't
        // strand the user there — drop back to the configured default mode
        // so the rest of the reset (capture controls, scrim, etc.) is
        // actually visible.
        if self.mode == crate::app::state::CameraMode::View {
            self.mode = self.config.default_mode;
        }

        // Reset camera exposure and color controls to defaults
        let exposure_task = self.handle_reset_exposure_settings();
        let color_task = self.handle_reset_color_settings();
        let fit_anim_task = self.start_fit_animation(from_fit);

        self.update_all_dropdowns();
        Task::batch([
            exposure_task,
            color_task,
            fit_anim_task,
            cosmic::command::set_theme(self.config.app_theme.theme()),
        ])
    }

    // =========================================================================
    // System & Recovery Handlers
    // =========================================================================

    pub(crate) fn handle_camera_recovery_started(
        &self,
        attempt: u32,
        max_attempts: u32,
    ) -> Task<cosmic::Action<Message>> {
        info!(attempt, max_attempts, "Camera backend recovery started");
        Task::none()
    }

    pub(crate) fn handle_camera_recovery_succeeded(&self) -> Task<cosmic::Action<Message>> {
        info!("Camera backend recovery succeeded");
        Task::none()
    }

    pub(crate) fn handle_camera_recovery_failed(
        &mut self,
        error: String,
    ) -> Task<cosmic::Action<Message>> {
        error!(error = %error, "Camera backend recovery failed");
        self.camera_cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
        self.fail_unfinished_viewfinder_hdr(&error);
        Task::none()
    }

    pub(crate) fn handle_audio_recovery_started(
        &self,
        attempt: u32,
        max_attempts: u32,
    ) -> Task<cosmic::Action<Message>> {
        info!(attempt, max_attempts, "Audio backend recovery started");
        Task::none()
    }

    pub(crate) fn handle_audio_recovery_succeeded(&self) -> Task<cosmic::Action<Message>> {
        info!("Audio backend recovery succeeded");
        Task::none()
    }

    pub(crate) fn handle_audio_recovery_failed(
        &self,
        error: String,
    ) -> Task<cosmic::Action<Message>> {
        error!(error = %error, "Audio backend recovery failed");
        Task::none()
    }

    pub(crate) fn handle_generate_bug_report(&mut self) -> Task<cosmic::Action<Message>> {
        info!("Generating bug report...");

        // Ensure insights are up-to-date (normally only refreshed when the
        // Insights drawer is open, so they may be stale/empty here).
        let _ = self.handle_update_insights_metrics();

        let video_devices = self.available_cameras.clone();
        let audio_devices = self.available_audio_devices.clone();
        let video_encoders = self.available_video_encoders.clone();
        let save_folder_name = self.config.save_folder_name.clone();
        let insights = self.insights.clone();
        let config = self.config.clone();
        let snapshot = crate::bug_report::AppStateSnapshot {
            current_camera_index: self.current_camera_index,
            current_audio_device_index: self.current_audio_device_index,
            current_video_encoder_index: self.current_video_encoder_index,
        };

        Task::perform(
            async move {
                crate::bug_report::BugReportGenerator::generate(
                    &video_devices,
                    &audio_devices,
                    &video_encoders,
                    &save_folder_name,
                    &insights,
                    &config,
                    &snapshot,
                )
                .await
                .map(|p| p.display().to_string())
            },
            |result| cosmic::Action::App(Message::BugReportGenerated(result)),
        )
    }

    pub(crate) fn handle_bug_report_generated(
        &mut self,
        result: Result<String, String>,
    ) -> Task<cosmic::Action<Message>> {
        match result {
            Ok(path) => {
                info!(path = %path, "Bug report generated successfully");
                self.last_bug_report_path = Some(path);

                let url = &self.config.bug_report_url;
                if let Err(e) = open::that(url) {
                    error!(error = %e, url = %url, "Failed to open bug report URL");
                } else {
                    info!(url = %url, "Opened bug report URL");
                }
            }
            Err(err) => {
                error!(error = %err, "Failed to generate bug report");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_show_bug_report(&self) -> Task<cosmic::Action<Message>> {
        if let Some(report_path) = &self.last_bug_report_path {
            info!(path = %report_path, "Showing bug report in file manager");
            if let Err(e) = Self::show_in_file_manager(report_path) {
                error!(error = %e, path = %report_path, "Failed to show bug report in file manager");
            }
        }
        Task::none()
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Show a file in the file manager with pre-selection
    pub(crate) fn show_in_file_manager(file_path: &str) -> Result<(), String> {
        use std::process::Command;

        let path = std::path::Path::new(file_path);
        let file_uri = format!("file://{}", path.display());

        // Method 1: Try D-Bus FileManager1.ShowItems
        let dbus_result = Command::new("dbus-send")
            .args(file_manager_dbus_args(&file_uri))
            .output();

        if let Ok(output) = dbus_result
            && output.status.success()
        {
            info!("Opened file manager via D-Bus");
            return Ok(());
        }

        // Method 2: Try file manager-specific commands
        let file_managers = [
            ("nautilus", vec!["--select", file_path]),
            ("dolphin", vec!["--select", file_path]),
            ("nemo", vec![file_path]),
            ("caja", vec![file_path]),
            ("thunar", vec![file_path]),
        ];

        for (fm_name, args) in &file_managers {
            if let Ok(output) = Command::new(fm_name).args(args).spawn() {
                info!(file_manager = fm_name, "Opened file manager");
                drop(output);
                return Ok(());
            }
        }

        // Method 3: Fallback to opening the parent directory
        if let Some(parent) = path.parent()
            && let Ok(child) = Command::new("xdg-open").arg(parent).spawn()
        {
            info!("Opened parent directory as fallback");
            drop(child);
            return Ok(());
        }

        Err("Failed to open file manager".to_string())
    }

    // =========================================================================
    // QR Code Detection Handlers
    // =========================================================================

    pub(crate) fn handle_toggle_qr_detection(&mut self) -> Task<cosmic::Action<Message>> {
        self.qr_detection_enabled = !self.qr_detection_enabled;
        self.invalidate_qr_detection();
        info!(enabled = self.qr_detection_enabled, "QR detection toggled");

        // Clear detections when disabling
        if !self.qr_detection_enabled {
            self.qr_detections.clear();
            self.qr_detections_shown_at = None;
        }

        if self.qr_detection_enabled
            && let Some(frame) = self.current_frame.clone()
        {
            return self.start_qr_detection(&frame);
        }

        Task::none()
    }

    pub(crate) fn handle_qr_detections_updated(
        &mut self,
        generation: u64,
        detections: Vec<crate::app::frame_processor::QrDetection>,
    ) -> Task<cosmic::Action<Message>> {
        let count = detections.len();
        let is_current = self
            .qr_detection_scheduler
            .finish(generation, std::time::Instant::now());

        if is_current && self.qr_detection_enabled && self.qr_detection_allowed() {
            self.pending_qr_frame = None;
            let now = std::time::Instant::now();
            if count > 0 {
                self.qr_detections = detections;
                self.qr_detections_shown_at = Some(now);
                info!(count, "QR detections updated");
            } else if self.qr_detections.is_empty()
                || crate::app::frame_processor::tasks::qr_detector::minimum_overlay_time_elapsed(
                    self.qr_detections_shown_at,
                    now,
                )
            {
                self.qr_detections.clear();
                self.qr_detections_shown_at = None;
            }
        } else if self.qr_detection_enabled
            && self.qr_detection_allowed()
            && let Some(frame) = self.pending_qr_frame.take()
        {
            return self.start_qr_detection(&frame);
        }

        Task::none()
    }

    pub(crate) fn handle_qr_open_url(&self, url: String) -> Task<cosmic::Action<Message>> {
        info!(url = %url, "Opening URL from QR code");
        match open::that_detached(&url) {
            Ok(()) => {
                info!("URL opened successfully");
            }
            Err(err) => {
                error!(url = %url, error = %err, "Failed to open URL");
            }
        }
        Task::none()
    }

    pub(crate) fn handle_qr_connect_wifi(
        &self,
        ssid: String,
        password: Option<String>,
        security: String,
        hidden: bool,
    ) -> Task<cosmic::Action<Message>> {
        // Use NetworkManager D-Bus API - works in both native and flatpak
        Task::perform(
            crate::network_manager::connect_wifi(ssid, password, security, hidden),
            |_| cosmic::Action::App(Message::Noop),
        )
    }

    pub(crate) fn handle_qr_copy_text(&self, text: String) -> Task<cosmic::Action<Message>> {
        info!(
            text_length = text.len(),
            "Copying text from QR code to clipboard"
        );

        // Use iced/cosmic clipboard API - works in both native and flatpak
        cosmic::iced::clipboard::write(text).map(|_: ()| cosmic::Action::App(Message::Noop))
    }

    // =========================================================================
    // Insights Handlers
    // =========================================================================

    pub(crate) fn handle_update_insights_metrics(&mut self) -> Task<cosmic::Action<Message>> {
        use crate::backends::camera::preview_metrics::PreviewMetrics;

        let path = self
            .available_cameras
            .get(self.current_camera_index)
            .map(|camera| camera.path.as_str())
            .unwrap_or("");
        let snapshot = PreviewMetrics::active_for(path)
            .map(|metrics| metrics.snapshot())
            .filter(|snapshot| snapshot.frame_count > 0);
        self.insights
            .refresh_preview(path, snapshot.as_ref(), std::time::Instant::now());
        self.update_insights_backend();
        self.update_insights_performance();
        self.update_insights_frame_metadata();
        self.update_insights_recording();
        Task::none()
    }

    fn update_insights_backend(&mut self) {
        if let Some(camera) = self.available_cameras.get(self.current_camera_index) {
            if self.insights.backend_type == "libcamera" {
                self.insights.pipeline_handler = camera.pipeline_handler.clone();
                self.insights.libcamera_version = camera.libcamera_version.clone();
                self.insights.sensor_model = camera.sensor_model.clone();
                self.insights.libcamera_multistream_capable = camera.supports_multistream;
                self.insights.format_chain.framerate = self
                    .active_format
                    .as_ref()
                    .and_then(|format| format.framerate)
                    .map(|fps| format!("{fps} fps (configured)"))
                    .unwrap_or_else(|| "N/A".into());
            }
            let current_v4l2_path = camera.v4l2_path().unwrap_or("").to_string();
            if self.insights.v4l2_formats_device != current_v4l2_path {
                self.insights.v4l2_formats_device = current_v4l2_path.clone();
                self.insights.v4l2_formats = if current_v4l2_path.is_empty() {
                    Vec::new()
                } else {
                    crate::backends::camera::v4l2_utils::enumerate_v4l2_formats(&current_v4l2_path)
                };
            }
            // The same V4L2 node can be routed through different backends or modes.
            // Do not use the kernel-device cache key for backend capabilities.
            self.insights.backend_formats = self.available_formats.clone();
        } else {
            self.insights.backend_formats.clear();
            self.insights.v4l2_formats.clear();
            self.insights.v4l2_formats_device.clear();
        }
    }

    fn update_insights_performance(&mut self) {
        use crate::app::video_primitive;
        if self.insights.preview_stream.is_none() {
            return;
        }
        self.insights.gpu_conversion_time_us = video_primitive::get_gpu_upload_time_us();
        let gpu_frame_size = video_primitive::get_gpu_frame_size() as usize;
        self.insights.copy_bandwidth_mbps =
            if gpu_frame_size > 0 && self.insights.gpu_conversion_time_us > 10 {
                let seconds = self.insights.gpu_conversion_time_us as f64 / 1_000_000.0;
                gpu_frame_size as f64 / seconds / 1_000_000.0
            } else {
                0.0
            };
    }

    fn update_insights_frame_metadata(&mut self) {
        self.insights.audio_levels = self
            .recording
            .audio_levels()
            .and_then(|levels| levels.lock().ok())
            .map(|l| l.clone());

        self.insights.clear_frame_metadata();
        if let Some(frame) = &self.current_frame
            && !self.insights.backend_type.is_empty()
        {
            self.insights.frame_latency_us = frame.captured_at.elapsed().as_micros() as u64;

            if self.insights.backend_type == "libcamera"
                && let Some(meta) = &frame.libcamera_metadata
            {
                self.insights.has_libcamera_metadata = true;
                self.insights.meta_exposure_us = meta.exposure_time;
                self.insights.meta_analogue_gain = meta.analogue_gain;
                self.insights.meta_digital_gain = meta.digital_gain;
                self.insights.meta_colour_temperature = meta.colour_temperature;
                self.insights.meta_sequence = meta.sequence;
                self.insights.meta_colour_gains = meta.colour_gains;
                self.insights.meta_black_level = meta.black_level;
                self.insights.meta_lens_position = meta.lens_position;
                self.insights.meta_lux = meta.lux;
                self.insights.meta_focus_fom = meta.focus_fom;
                self.insights.frame_metadata = Some(meta.clone());
            }
        }
    }

    fn update_insights_recording(&mut self) {
        self.insights.recording_diag = crate::pipelines::video::get_recording_diagnostics();
        self.insights.recording_stats = if self.insights.recording_diag.is_some() {
            Some(crate::pipelines::video::get_recording_stats())
        } else {
            None
        };
    }

    pub(crate) fn handle_copy_pipeline_string(&self) -> Task<cosmic::Action<Message>> {
        if let Some(pipeline) = &self.insights.full_pipeline_string {
            info!("Copying pipeline string to clipboard");
            cosmic::iced::clipboard::write(pipeline.clone())
                .map(|_: ()| cosmic::Action::App(Message::Noop))
        } else {
            Task::none()
        }
    }

    // =========================================================================
    // Insights Capture (raw frame dump)
    // =========================================================================

    /// Capture raw binary frames from all running streams plus metadata as JSON.
    ///
    /// `frame_count`: 1 for single capture, 6 for burst.
    /// Saves to `{photos_dir}/insights/capture_{timestamp}_*.{bin,json}`.
    pub(crate) fn handle_insights_capture(
        &mut self,
        frame_count: usize,
    ) -> Task<cosmic::Action<Message>> {
        // Snapshot the insights state as a JSON-serialisable value
        let insights_json = self.build_insights_json();

        // Grab the current preview frame (viewfinder stream) immediately
        let preview_frame = self.current_frame.clone();

        // For the raw/capture stream we use the same still-capture mechanism as burst mode
        let is_multistream = self.is_current_camera_multistream();
        let still_capture = std::sync::Arc::clone(&self.still_capture);

        let save_dir =
            crate::app::get_photo_directory(&self.config.save_folder_name).join("insights");

        info!(frame_count, is_multistream, "Starting insights capture");

        Task::perform(
            async move {
                // Ensure output directory exists
                tokio::fs::create_dir_all(&save_dir)
                    .await
                    .map_err(|e| format!("Failed to create insights dir: {e}"))?;

                let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S_%3f");
                let mut saved_paths = Vec::new();

                // Save insights state JSON (once per capture session)
                let insights_path = save_dir.join(format!("capture_{timestamp}_insights.json"));
                tokio::fs::write(&insights_path, &insights_json)
                    .await
                    .map_err(|e| format!("Failed to write insights JSON: {e}"))?;
                saved_paths.push(insights_path.display().to_string());

                for i in 0..frame_count {
                    // e.g. "_1_of_6" for burst, "" for single
                    let suffix = if frame_count > 1 {
                        format!("_{}_of_{}", i + 1, frame_count)
                    } else {
                        String::new()
                    };

                    // --- Preview frame ---
                    // Snapshot from when capture was triggered (can't poll live
                    // frames from async task, but the preview frame is useful
                    // as a reference for each burst iteration's metadata).
                    if let Some(ref frame) = preview_frame {
                        let bin_path =
                            save_dir.join(format!("capture_{timestamp}_preview{suffix}.bin"));
                        tokio::fs::write(&bin_path, frame.data.as_ref())
                            .await
                            .map_err(|e| format!("Failed to write preview bin: {e}"))?;
                        saved_paths.push(bin_path.display().to_string());

                        let meta_path =
                            save_dir.join(format!("capture_{timestamp}_preview{suffix}_meta.json"));
                        let meta_json = frame_metadata_to_json(frame, "preview");
                        tokio::fs::write(&meta_path, meta_json)
                            .await
                            .map_err(|e| format!("Failed to write preview meta: {e}"))?;
                        saved_paths.push(meta_path.display().to_string());
                    }

                    // --- Raw/capture stream frame ---
                    if is_multistream {
                        let still_request = still_capture.request();

                        let timeout = std::time::Duration::from_secs(2);
                        let raw_frame =
                            super::capture::wait_for_still_frame(still_request, timeout).await;
                        if raw_frame.is_none() {
                            info!(frame = i + 1, "Timeout waiting for raw frame, skipping");
                        }

                        if let Some(ref frame) = raw_frame {
                            let bin_path =
                                save_dir.join(format!("capture_{timestamp}_raw{suffix}.bin"));
                            tokio::fs::write(&bin_path, frame.data.as_ref())
                                .await
                                .map_err(|e| format!("Failed to write raw bin: {e}"))?;
                            saved_paths.push(bin_path.display().to_string());

                            let meta_path =
                                save_dir.join(format!("capture_{timestamp}_raw{suffix}_meta.json"));
                            let meta_json = frame_metadata_to_json(frame, "raw");
                            tokio::fs::write(&meta_path, meta_json)
                                .await
                                .map_err(|e| format!("Failed to write raw meta: {e}"))?;
                            saved_paths.push(meta_path.display().to_string());
                        }
                    }
                }

                info!(
                    count = saved_paths.len(),
                    dir = %save_dir.display(),
                    "Insights capture complete"
                );
                Ok(saved_paths)
            },
            |result| cosmic::Action::App(Message::InsightsCaptureComplete(result)),
        )
    }

    /// Build a JSON string capturing the entire current insights state.
    fn build_insights_json(&self) -> String {
        let ins = &self.insights;
        let mut map = serde_json::Map::new();

        // Pipeline
        if let Some(ref p) = ins.full_pipeline_string {
            map.insert("pipeline".into(), serde_json::Value::String(p.clone()));
        }
        map.insert(
            "decoder_chain".into(),
            serde_json::Value::Array(
                ins.decoder_chain
                    .iter()
                    .map(|d| {
                        serde_json::json!({
                            "name": d.name,
                            "description": d.description,
                            "state": format!("{:?}", d.state),
                        })
                    })
                    .collect(),
            ),
        );

        // Format chain
        map.insert(
            "format_chain".into(),
            serde_json::json!({
                "source": ins.format_chain.source,
                "resolution": ins.format_chain.resolution,
                "framerate": ins.format_chain.framerate,
                "native_format": ins.format_chain.native_format,
                "wgpu_processing": ins.format_chain.wgpu_processing,
            }),
        );

        // Performance
        map.insert(
            "performance".into(),
            serde_json::json!({
                "frame_latency_us": ins.frame_latency_us,
                "dropped_frames": ins.dropped_frames,
                "frame_size_decoded": ins.frame_size_decoded,
                "gpu_conversion_time_us": ins.gpu_conversion_time_us,
                "copy_time_us": ins.copy_time_us,
                "cpu_decode_time_us": ins.cpu_decode_time_us,
                "cpu_processing": ins.cpu_processing,
                "copy_bandwidth_mbps": ins.copy_bandwidth_mbps,
            }),
        );

        // Backend
        map.insert(
            "backend".into(),
            serde_json::json!({
                "type": ins.backend_type,
                "pipeline_handler": ins.pipeline_handler,
                "libcamera_version": ins.libcamera_version,
                "sensor_model": ins.sensor_model,
                "mjpeg_decoder": ins.mjpeg_decoder,
                "is_multistream": ins.is_multistream,
                "libcamera_multistream_capable": ins.libcamera_multistream_capable,
            }),
        );

        // Streams
        if let Some(ref s) = ins.preview_stream {
            map.insert(
                "preview_stream".into(),
                serde_json::json!({
                    "role": s.role,
                    "resolution": s.resolution,
                    "pixel_format": s.pixel_format,
                    "frame_count": s.frame_count,
                }),
            );
        }
        if let Some(ref s) = ins.capture_stream {
            map.insert(
                "capture_stream".into(),
                serde_json::json!({
                    "role": s.role,
                    "resolution": s.resolution,
                    "pixel_format": s.pixel_format,
                    "frame_count": s.frame_count,
                }),
            );
        }

        // Recording pipeline
        if let Some(ref rec) = ins.recording_diag {
            let mut rec_json = serde_json::json!({
                "mode": rec.mode,
                "encoder": rec.encoder,
                "resolution": rec.resolution,
                "framerate": rec.framerate,
                "pipeline_string": rec.pipeline_string,
            });
            if let Some(ref stats) = ins.recording_stats {
                rec_json["stats"] = serde_json::json!({
                    "capture_sent": stats.capture_sent,
                    "capture_dropped": stats.capture_dropped,
                    "pusher_pushed": stats.pusher_pushed,
                    "pusher_skipped": stats.pusher_skipped,
                    "channel_backlog": stats.channel_backlog,
                    "effective_fps": stats.effective_fps,
                    "last_pts_ms": stats.last_pts_ms,
                    "last_processing_delay_us": stats.last_processing_delay_us,
                    "last_convert_time_us": stats.last_convert_time_us,
                });
            }
            map.insert("recording_pipeline".into(), rec_json);
        }

        // Frame metadata
        map.insert(
            "frame_metadata".into(),
            serde_json::json!({
                "has_libcamera_metadata": ins.has_libcamera_metadata,
                "exposure_us": ins.meta_exposure_us,
                "analogue_gain": ins.meta_analogue_gain,
                "digital_gain": ins.meta_digital_gain,
                "colour_temperature": ins.meta_colour_temperature,
                "sequence": ins.meta_sequence,
                "colour_gains": ins.meta_colour_gains,
                "black_level": ins.meta_black_level,
                "lens_position": ins.meta_lens_position,
                "lux": ins.meta_lux,
                "focus_fom": ins.meta_focus_fom,
            }),
        );

        serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_default()
    }

    pub(crate) fn persist_config(&mut self) {
        self.persist_config_async();
    }
}

/// Serialize a single CameraFrame's metadata to JSON.
fn frame_metadata_to_json(
    frame: &crate::backends::camera::types::CameraFrame,
    stream_name: &str,
) -> String {
    let mut map = serde_json::Map::new();

    map.insert(
        "stream".into(),
        serde_json::Value::String(stream_name.to_string()),
    );
    map.insert("width".into(), serde_json::json!(frame.width));
    map.insert("height".into(), serde_json::json!(frame.height));
    map.insert(
        "format".into(),
        serde_json::Value::String(format!("{:?}", frame.format)),
    );
    map.insert("stride".into(), serde_json::json!(frame.stride));
    map.insert("data_size".into(), serde_json::json!(frame.data.len()));
    map.insert(
        "sensor_timestamp_ns".into(),
        serde_json::json!(frame.sensor_timestamp_ns),
    );

    if let Some(ref planes) = frame.yuv_planes {
        map.insert(
            "yuv_planes".into(),
            serde_json::json!({
                "y_offset": planes.y_offset,
                "y_size": planes.y_size,
                "uv_offset": planes.uv_offset,
                "uv_size": planes.uv_size,
                "uv_stride": planes.uv_stride,
                "v_offset": planes.v_offset,
                "v_size": planes.v_size,
                "v_stride": planes.v_stride,
                "uv_width": planes.uv_width,
                "uv_height": planes.uv_height,
            }),
        );
    }

    if let Some(ref meta) = frame.libcamera_metadata {
        map.insert(
            "libcamera_metadata".into(),
            serde_json::json!({
                "exposure_time_us": meta.exposure_time,
                "analogue_gain": meta.analogue_gain,
                "digital_gain": meta.digital_gain,
                "colour_temperature": meta.colour_temperature,
                "lens_position": meta.lens_position,
                "sensor_timestamp": meta.sensor_timestamp,
                "sequence": meta.sequence,
                "af_state": format!("{:?}", meta.af_state),
                "ae_state": format!("{:?}", meta.ae_state),
                "awb_state": format!("{:?}", meta.awb_state),
                "colour_gains": meta.colour_gains,
                "colour_correction_matrix": meta.colour_correction_matrix,
                "black_level": meta.black_level,
                "lux": meta.lux,
                "focus_fom": meta.focus_fom,
            }),
        );
    }

    serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{file_manager_dbus_args, portal_gallery_item};

    #[test]
    fn demosaic_handler_persists_each_method_and_reset_restores_bilinear() {
        use crate::app::state::{AppModel, Message, SettingsPage};
        use crate::config::Config;
        use crate::shaders::DemosaicMethod;
        use cosmic::cosmic_config::{Config as Store, ConfigGet, CosmicConfigEntry};
        gstreamer::init().unwrap();
        let root =
            std::env::temp_dir().join(format!("camera-demosaic-handler-{}", uuid::Uuid::new_v4()));
        let store =
            Store::with_custom_path("camera-demosaic-test", Config::VERSION, root.clone()).unwrap();
        let mut app = AppModel {
            config_handler: Some(store),
            settings_page: SettingsPage::Photo,
            ..AppModel::default()
        };
        for method in [
            DemosaicMethod::Lmmse,
            DemosaicMethod::Bilinear,
            DemosaicMethod::Rcd,
        ] {
            drop(app.update(Message::SetDemosaicMethod(method)));
            app.config_persistence.flush();
            assert_eq!(app.config.demosaic_method, method);
            let store = app.config_handler.as_ref().unwrap();
            assert_eq!(
                store.get::<DemosaicMethod>("demosaic_method").unwrap(),
                method
            );
            assert_eq!(Config::get_entry(store).unwrap().demosaic_method, method);
            drop(app.settings_view());
        }
        drop(app.update(Message::SetDemosaicMethod(DemosaicMethod::Rcd)));
        drop(app.update(Message::ResetAllSettings));
        app.config_persistence.flush();
        assert_eq!(app.config.demosaic_method, DemosaicMethod::Bilinear);
        assert_eq!(
            Config::get_entry(app.config_handler.as_ref().unwrap())
                .unwrap()
                .demosaic_method,
            DemosaicMethod::Bilinear
        );
        drop(app);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn flatpak_gallery_routes_the_latest_video_to_the_portal() {
        let video = "/home/test/Videos/Camera/latest.mp4";

        assert_eq!(portal_gallery_item(true, Some(video)), Some(video));
        assert_eq!(portal_gallery_item(true, None), None);
        assert_eq!(portal_gallery_item(false, Some(video)), None);
    }

    #[test]
    fn file_manager_dbus_call_waits_for_the_method_reply() {
        let args = file_manager_dbus_args("file:///tmp/photo.jpg");

        assert_eq!(
            args,
            [
                "--session",
                "--print-reply",
                "--reply-timeout=5000",
                "--dest=org.freedesktop.FileManager1",
                "--type=method_call",
                "/org/freedesktop/FileManager1",
                "org.freedesktop.FileManager1.ShowItems",
                "array:string:file:///tmp/photo.jpg",
                "string:",
            ]
        );
    }
}
