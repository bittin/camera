// SPDX-License-Identifier: GPL-3.0-only

//! Exposure control handlers
//!
//! Handles exposure mode, compensation, time, gain, ISO, metering, and backlight.

use crate::app::exposure_picker::{
    AvailableExposureControls, ColorSettings, ExposureMode, ExposureSettings, MeteringMode,
};
use crate::app::state::{AppModel, Message};
use crate::backends::camera::v4l2_controls;
use cosmic::Task;
use tracing::{debug, info};

/// All exposure/color writes share the same completion protocol.
pub(super) fn control_task(
    operation: impl std::future::Future<Output = Result<(), String>> + Send + 'static,
) -> Task<cosmic::Action<Message>> {
    Task::perform(operation, |result| {
        cosmic::Action::App(match result {
            Ok(()) => Message::ExposureControlApplied,
            Err(error) => Message::ExposureControlFailed(error),
        })
    })
}

fn set_control_task(
    device_path: String,
    control_id: u32,
    value: i32,
) -> Task<cosmic::Action<Message>> {
    control_task(async move { v4l2_controls::set_control(&device_path, control_id, value) })
}

impl AppModel {
    // =========================================================================
    // Exposure Control Handlers
    // =========================================================================

    pub(crate) fn handle_toggle_exposure_picker(&mut self) -> Task<cosmic::Action<Message>> {
        let opening = !self.exposure_picker_visible;
        self.close_all_pickers();
        self.exposure_picker_visible = opening;
        info!(
            visible = self.exposure_picker_visible,
            "Exposure picker toggled"
        );
        Task::none()
    }

    pub(crate) fn handle_close_exposure_picker(&mut self) -> Task<cosmic::Action<Message>> {
        self.exposure_picker_visible = false;
        Task::none()
    }

    pub(crate) fn handle_set_exposure_mode(
        &mut self,
        mode: ExposureMode,
    ) -> Task<cosmic::Action<Message>> {
        // Update local state
        if let Some(ref mut settings) = self.exposure_settings {
            settings.mode = mode;
        }

        // Sync segmented button model
        let position = if mode == ExposureMode::Manual { 1 } else { 0 };
        self.exposure_mode_model.activate_position(position);

        // Apply to camera via V4L2
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        let value = mode.to_v4l2_value();
        info!(mode = ?mode, value, "Setting exposure mode");

        set_control_task(device_path, v4l2_controls::V4L2_CID_EXPOSURE_AUTO, value)
    }

    pub(crate) fn handle_exposure_mode_selected(
        &mut self,
        entity: cosmic::widget::segmented_button::Entity,
    ) -> Task<cosmic::Action<Message>> {
        // Activate the selected button
        self.exposure_mode_model.activate(entity);

        // Get position to determine mode (0 = Auto, 1 = Manual)
        let position = self.exposure_mode_model.position(entity).unwrap_or(0);
        let mode = if position == 0 {
            ExposureMode::AperturePriority
        } else {
            ExposureMode::Manual
        };

        self.handle_set_exposure_mode(mode)
    }

    pub(crate) fn handle_set_exposure_compensation(
        &mut self,
        value: i32,
    ) -> Task<cosmic::Action<Message>> {
        // Update local state
        if let Some(ref mut settings) = self.exposure_settings {
            settings.exposure_compensation = value;
        }

        // Apply to camera via V4L2
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        debug!(
            ev_units = value,
            ev = value as f32 / 1000.0,
            "Setting exposure compensation"
        );

        // Set EV bias directly (only works on cameras that support it)
        set_control_task(
            device_path,
            v4l2_controls::V4L2_CID_AUTO_EXPOSURE_BIAS,
            value,
        )
    }

    pub(crate) fn handle_reset_exposure_compensation(&mut self) -> Task<cosmic::Action<Message>> {
        self.handle_set_exposure_compensation(0)
    }

    pub(crate) fn handle_set_exposure_time(&mut self, value: i32) -> Task<cosmic::Action<Message>> {
        // Update local state
        if let Some(ref mut settings) = self.exposure_settings {
            settings.exposure_time = Some(value);
        }

        // Apply to camera via V4L2
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        // Get current gain to re-apply (some cameras need both set together)
        let current_gain = self
            .exposure_settings
            .as_ref()
            .and_then(|s| s.gain)
            .unwrap_or(self.available_exposure_controls.gain.default);
        let has_gain = self.available_exposure_controls.gain.available;

        debug!(
            time_100us = value,
            gain = current_gain,
            "Setting exposure time"
        );

        // Exposure time only works in manual mode - ensure we're in manual mode first
        let manual_value = ExposureMode::Manual.to_v4l2_value();

        control_task(async move {
            // Ensure manual mode is set (some cameras need this before each exposure change)
            let _ = v4l2_controls::set_control(
                &device_path,
                v4l2_controls::V4L2_CID_EXPOSURE_AUTO,
                manual_value,
            );

            // Set the exposure time
            let result = v4l2_controls::set_control(
                &device_path,
                v4l2_controls::V4L2_CID_EXPOSURE_ABSOLUTE,
                value,
            );

            // Also re-apply gain (some cameras need both set together for changes to take effect)
            if has_gain {
                let _ = v4l2_controls::set_control(
                    &device_path,
                    v4l2_controls::V4L2_CID_GAIN,
                    current_gain,
                );
            }

            result
        })
    }

    pub(crate) fn handle_set_gain(&mut self, value: i32) -> Task<cosmic::Action<Message>> {
        // Update local state
        if let Some(ref mut settings) = self.exposure_settings {
            settings.gain = Some(value);
        }

        // Apply to camera via V4L2
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        // Get current exposure time to re-apply (some cameras need both set together)
        let current_exposure = self
            .exposure_settings
            .as_ref()
            .and_then(|s| s.exposure_time)
            .unwrap_or(self.available_exposure_controls.exposure_time.default);
        let has_exposure = self.available_exposure_controls.exposure_time.available;

        debug!(gain = value, exposure = current_exposure, "Setting gain");

        // Ensure manual mode for gain control
        let manual_value = ExposureMode::Manual.to_v4l2_value();

        control_task(async move {
            // Ensure manual mode is set
            let _ = v4l2_controls::set_control(
                &device_path,
                v4l2_controls::V4L2_CID_EXPOSURE_AUTO,
                manual_value,
            );

            // Try standard gain first, then analogue gain
            let result =
                v4l2_controls::set_control(&device_path, v4l2_controls::V4L2_CID_GAIN, value)
                    .or_else(|_| {
                        v4l2_controls::set_control(
                            &device_path,
                            v4l2_controls::V4L2_CID_ANALOGUE_GAIN,
                            value,
                        )
                    });

            // Also re-apply exposure time (some cameras need both set together)
            if has_exposure {
                let _ = v4l2_controls::set_control(
                    &device_path,
                    v4l2_controls::V4L2_CID_EXPOSURE_ABSOLUTE,
                    current_exposure,
                );
            }

            result
        })
    }

    pub(crate) fn handle_set_iso_sensitivity(
        &mut self,
        value: i32,
    ) -> Task<cosmic::Action<Message>> {
        // Update local state
        if let Some(ref mut settings) = self.exposure_settings {
            settings.iso = Some(value);
        }

        // Apply to camera via V4L2
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        debug!(iso = value, "Setting ISO sensitivity");

        set_control_task(device_path, v4l2_controls::V4L2_CID_ISO_SENSITIVITY, value)
    }

    pub(crate) fn handle_set_metering_mode(
        &mut self,
        mode: MeteringMode,
    ) -> Task<cosmic::Action<Message>> {
        // Update local state
        if let Some(ref mut settings) = self.exposure_settings {
            settings.metering_mode = Some(mode);
        }

        // Apply to camera via V4L2
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        let value = mode.to_v4l2_value();
        debug!(mode = ?mode, value, "Setting metering mode");

        set_control_task(
            device_path,
            v4l2_controls::V4L2_CID_EXPOSURE_METERING,
            value,
        )
    }

    pub(crate) fn handle_toggle_auto_exposure_priority(&mut self) -> Task<cosmic::Action<Message>> {
        // Update local state
        let new_value = if let Some(ref mut settings) = self.exposure_settings {
            let current = settings.auto_priority.unwrap_or(false);
            settings.auto_priority = Some(!current);
            !current
        } else {
            true
        };

        // Apply to camera via V4L2
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        let value = if new_value { 1 } else { 0 };
        debug!(enabled = new_value, "Setting auto exposure priority");

        set_control_task(
            device_path,
            v4l2_controls::V4L2_CID_EXPOSURE_AUTO_PRIORITY,
            value,
        )
    }

    pub(crate) fn handle_set_backlight_compensation(
        &mut self,
        value: i32,
    ) -> Task<cosmic::Action<Message>> {
        if let Some(ref mut settings) = self.exposure_settings {
            settings.backlight_compensation = Some(value);
        }
        debug!(backlight = value, "Setting backlight compensation");
        self.set_v4l2_control(v4l2_controls::V4L2_CID_BACKLIGHT_COMPENSATION, value)
    }

    pub(crate) fn handle_set_focus_absolute(
        &mut self,
        value: i32,
    ) -> Task<cosmic::Action<Message>> {
        if let Some(ref mut settings) = self.exposure_settings {
            settings.focus_absolute = Some(value);
        }

        let focus_path = self
            .get_focus_device_path()
            .or_else(|| self.get_v4l2_device_path());
        let Some(focus_path) = focus_path else {
            return Task::none();
        };

        debug!(focus = value, path = %focus_path, "Setting focus absolute");

        set_control_task(focus_path, v4l2_controls::V4L2_CID_FOCUS_ABSOLUTE, value)
    }

    pub(crate) fn handle_toggle_focus_auto(&mut self) -> Task<cosmic::Action<Message>> {
        let new_value = if let Some(ref mut settings) = self.exposure_settings {
            let current = settings.focus_auto.unwrap_or(false);
            settings.focus_auto = Some(!current);
            !current
        } else {
            true
        };

        let focus_path = self
            .get_focus_device_path()
            .or_else(|| self.get_v4l2_device_path());
        let Some(focus_path) = focus_path else {
            return Task::none();
        };

        let value = if new_value { 1 } else { 0 };
        debug!(enabled = new_value, path = %focus_path, "Setting focus auto");

        set_control_task(focus_path, v4l2_controls::V4L2_CID_FOCUS_AUTO, value)
    }

    /// Reset all exposure settings to camera defaults (preserving current mode)
    pub(crate) fn handle_reset_exposure_settings(&mut self) -> Task<cosmic::Action<Message>> {
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        let controls = self.available_exposure_controls.clone();
        let current_mode = self
            .exposure_settings
            .as_ref()
            .map(|s| s.mode)
            .unwrap_or_default();

        info!(?current_mode, "Resetting exposure settings to defaults");

        // Update local state to defaults
        if let Some(ref mut settings) = self.exposure_settings {
            settings.exposure_compensation = controls.exposure_bias.default;
            settings.exposure_time = Some(controls.exposure_time.default);
            settings.gain = Some(controls.gain.default);
            settings.iso = Some(controls.iso.default);
            settings.backlight_compensation = Some(controls.backlight_compensation.default);
            settings.auto_priority = Some(false);
            // Keep mode unchanged
        }

        // Apply defaults to camera
        Task::perform(
            async move {
                // Keep the camera's write order and best-effort reset semantics.
                for (available, control_id, value) in [
                    (
                        controls.exposure_time.available,
                        v4l2_controls::V4L2_CID_EXPOSURE_ABSOLUTE,
                        controls.exposure_time.default,
                    ),
                    (
                        controls.gain.available,
                        v4l2_controls::V4L2_CID_GAIN,
                        controls.gain.default,
                    ),
                    (
                        controls.iso.available,
                        v4l2_controls::V4L2_CID_ISO_SENSITIVITY,
                        controls.iso.default,
                    ),
                    (
                        controls.exposure_bias.available,
                        v4l2_controls::V4L2_CID_AUTO_EXPOSURE_BIAS,
                        controls.exposure_bias.default,
                    ),
                    (
                        controls.backlight_compensation.available,
                        v4l2_controls::V4L2_CID_BACKLIGHT_COMPENSATION,
                        controls.backlight_compensation.default,
                    ),
                    (
                        controls.has_auto_priority,
                        v4l2_controls::V4L2_CID_EXPOSURE_AUTO_PRIORITY,
                        0,
                    ),
                ] {
                    if available {
                        let _ = v4l2_controls::set_control(&device_path, control_id, value);
                    }
                }
            },
            |_| cosmic::Action::App(Message::ExposureControlApplied),
        )
    }

    pub(crate) fn handle_exposure_controls_queried(
        &mut self,
        controls: Box<AvailableExposureControls>,
        settings: ExposureSettings,
        color_settings: ColorSettings,
    ) -> Task<cosmic::Action<Message>> {
        info!(
            has_mode = controls.has_exposure_auto,
            has_ev = controls.exposure_bias.available,
            has_time = controls.exposure_time.available,
            has_gain = controls.gain.available,
            has_iso = controls.iso.available,
            "Exposure controls queried"
        );
        self.available_exposure_controls = *controls;
        self.exposure_settings = Some(settings);
        self.color_settings = Some(color_settings);
        Task::none()
    }

    // =========================================================================
    // V4L2 Helpers (used by exposure and color handlers)
    // =========================================================================

    /// Get the V4L2 device path for the current camera
    pub(crate) fn get_v4l2_device_path(&self) -> Option<String> {
        self.available_cameras
            .get(self.current_camera_index)
            .and_then(|cam| cam.v4l2_path())
            .map(str::to_owned)
    }

    /// Helper to set a V4L2 control value asynchronously
    ///
    /// This reduces duplication across exposure and color control handlers.
    pub(crate) fn set_v4l2_control(
        &self,
        control_id: u32,
        value: i32,
    ) -> Task<cosmic::Action<Message>> {
        let Some(device_path) = self.get_v4l2_device_path() else {
            return Task::none();
        };

        set_control_task(device_path, control_id, value)
    }

    /// Get the lens actuator device path for the current camera
    pub(crate) fn get_focus_device_path(&self) -> Option<String> {
        self.available_cameras
            .get(self.current_camera_index)
            .and_then(|cam| cam.lens_actuator_path.clone())
    }

    /// Create a task to query exposure controls for the current camera
    /// This resets exposure settings to defaults (aperture priority mode, default backlight)
    pub(crate) fn query_exposure_controls_task(&self) -> Task<cosmic::Action<Message>> {
        if let Some(path) = self.get_v4l2_device_path() {
            let focus_path = self.get_focus_device_path();
            Task::perform(
                async move {
                    let controls = crate::app::exposure_picker::query_exposure_controls(
                        &path,
                        focus_path.as_deref(),
                    );
                    let settings = crate::app::exposure_picker::get_exposure_settings(
                        &path,
                        &controls,
                        focus_path.as_deref(),
                    );
                    let color_settings =
                        crate::app::exposure_picker::get_color_settings(&path, &controls);
                    (controls, settings, color_settings)
                },
                |(controls, settings, color_settings)| {
                    cosmic::Action::App(Message::ExposureControlsQueried(
                        Box::new(controls),
                        settings,
                        color_settings,
                    ))
                },
            )
        } else {
            Task::none()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    async fn completion(task: Task<cosmic::Action<Message>>) -> Message {
        let mut stream = cosmic::iced::runtime::task::into_stream(task).unwrap();
        let Some(cosmic::iced::runtime::Action::Output(cosmic::Action::App(message))) =
            stream.next().await
        else {
            panic!("control task must emit an application message");
        };
        assert!(stream.next().await.is_none());
        message
    }

    #[tokio::test]
    async fn control_tasks_preserve_success_and_error_completions() {
        assert!(matches!(
            completion(control_task(async { Ok(()) })).await,
            Message::ExposureControlApplied
        ));
        assert!(matches!(
            completion(control_task(async { Err("control unavailable".into()) })).await,
            Message::ExposureControlFailed(error) if error == "control unavailable"
        ));
    }

    #[tokio::test]
    async fn camera_write_errors_reach_the_completion_message() {
        let mut app = AppModel {
            available_cameras: vec![crate::backends::camera::types::CameraDevice {
                lens_actuator_path: Some("/nonexistent-camera-control-device".into()),
                ..Default::default()
            }],
            exposure_settings: Some(ExposureSettings::default()),
            ..Default::default()
        };
        assert!(matches!(
            completion(app.handle_set_focus_absolute(42)).await,
            Message::ExposureControlFailed(error) if !error.is_empty()
        ));
        assert_eq!(
            app.exposure_settings.as_ref().unwrap().focus_absolute,
            Some(42)
        );
    }

    #[tokio::test]
    async fn resets_preserve_mode_unavailable_colors_and_best_effort_completion() {
        use crate::app::exposure_picker::ControlRange;
        use crate::backends::camera::types::{CameraDevice, DeviceInfo};
        let mut app = AppModel {
            available_cameras: vec![CameraDevice {
                device_info: Some(DeviceInfo {
                    path: "/nonexistent-camera-control-device".into(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            exposure_settings: Some(ExposureSettings {
                mode: ExposureMode::Manual,
                focus_absolute: Some(42),
                ..Default::default()
            }),
            color_settings: Some(ColorSettings {
                hue: Some(17),
                white_balance_auto: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        app.available_exposure_controls.exposure_bias = ControlRange::new(-1000, 1000, 1, 100);
        app.available_exposure_controls.contrast = ControlRange::new(0, 100, 1, 50);
        app.available_exposure_controls.white_balance_temperature =
            ControlRange::new(2000, 8000, 1, 4500);

        assert!(matches!(
            completion(app.handle_reset_exposure_settings()).await,
            Message::ExposureControlApplied
        ));
        let settings = app.exposure_settings.as_ref().unwrap();
        assert_eq!(settings.mode, ExposureMode::Manual);
        assert_eq!(settings.exposure_compensation, 100);
        assert_eq!(settings.exposure_time, Some(0));
        assert_eq!(settings.gain, Some(0));
        assert_eq!(settings.auto_priority, Some(false));
        assert_eq!(settings.focus_absolute, Some(42));

        assert!(matches!(
            completion(app.handle_reset_color_settings()).await,
            Message::ExposureControlApplied
        ));
        let settings = app.color_settings.as_ref().unwrap();
        assert_eq!(settings.contrast, Some(50));
        assert_eq!(settings.white_balance_temperature, Some(4500));
        assert_eq!(settings.hue, Some(17));
        assert_eq!(settings.white_balance_auto, Some(false));
    }

    #[test]
    fn unavailable_camera_controls_still_update_local_settings_without_tasks() {
        let mut app = AppModel {
            exposure_settings: Some(ExposureSettings::default()),
            ..Default::default()
        };
        for value in [1200, 0] {
            let task = if value == 0 {
                app.handle_reset_exposure_compensation()
            } else {
                app.handle_set_exposure_compensation(value)
            };
            assert!(cosmic::iced::runtime::task::into_stream(task).is_none());
            assert_eq!(
                app.exposure_settings
                    .as_ref()
                    .unwrap()
                    .exposure_compensation,
                value
            );
        }
    }
}
