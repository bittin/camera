// SPDX-License-Identifier: GPL-3.0-only

//! Default capture-mode dropdown options for settings.

use crate::app::state::{AppModel, CameraMode};

/// Launch choices are the prefix of CameraMode::ALL, not the carousel order.
/// View is intentionally not offered as a launch default.
pub(crate) fn default_modes(virtual_camera_enabled: bool) -> &'static [CameraMode] {
    &CameraMode::ALL[..if virtual_camera_enabled { 4 } else { 3 }]
}

pub(crate) fn default_mode_options(virtual_camera_enabled: bool) -> Vec<String> {
    default_modes(virtual_camera_enabled)
        .iter()
        .map(|mode| match mode {
            CameraMode::Photo => crate::fl!("mode-photo"),
            CameraMode::Video => crate::fl!("mode-video"),
            CameraMode::Timelapse => crate::fl!("mode-timelapse"),
            CameraMode::Virtual => crate::fl!("mode-virtual"),
            CameraMode::View => unreachable!("View is not a default-mode option"),
        })
        .collect()
}

impl AppModel {
    /// Update default mode dropdown options (conditionally includes Virtual)
    pub fn update_default_mode_dropdown(&mut self) {
        self.default_mode_dropdown_options =
            default_mode_options(self.config.virtual_camera_enabled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_choices_preserve_order_and_exclude_view() {
        use CameraMode::*;
        assert_eq!(default_modes(false), &[Photo, Video, Timelapse]);
        assert_eq!(default_modes(true), &[Photo, Video, Timelapse, Virtual]);
        for enabled in [false, true] {
            let mut model = AppModel::default();
            model.config.virtual_camera_enabled = enabled;
            model.update_default_mode_dropdown();
            assert_eq!(
                model.default_mode_dropdown_options,
                default_mode_options(enabled)
            );
            for (index, &mode) in default_modes(enabled).iter().enumerate() {
                let _ = model.handle_select_default_mode(index);
                assert_eq!(model.config.default_mode, mode);
            }
            let previous = model.config.default_mode;
            let _ = model.handle_select_default_mode(default_modes(enabled).len());
            assert_eq!(model.config.default_mode, previous);
        }
    }
}
