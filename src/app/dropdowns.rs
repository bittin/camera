// SPDX-License-Identifier: GPL-3.0-only

//! Default capture-mode dropdown options for settings.

use crate::app::state::AppModel;

impl AppModel {
    /// Update default mode dropdown options (conditionally includes Virtual)
    pub fn update_default_mode_dropdown(&mut self) {
        self.default_mode_dropdown_options = vec![
            crate::fl!("mode-photo"),
            crate::fl!("mode-video"),
            crate::fl!("mode-timelapse"),
        ];
        if self.config.virtual_camera_enabled {
            self.default_mode_dropdown_options
                .push(crate::fl!("mode-virtual"));
        }
    }
}
