// SPDX-License-Identifier: GPL-3.0-only

//! Mode switcher widget implementation (Photo/Video/Timelapse/Virtual toggle)

use crate::app::state::{AppModel, CameraMode, Message};
use crate::app::view::overlay_container_style;
use crate::fl;
use cosmic::Element;
use cosmic::iced::Color;
use cosmic::widget;

/// Helper to wrap a mode button with overlay styling
fn styled_mode_button<'a>(
    button: impl Into<Element<'a, Message>>,
    is_disabled: bool,
) -> Element<'a, Message> {
    widget::container(button)
        .style(move |theme| {
            let mut style = overlay_container_style(theme);
            if is_disabled {
                style.text_color = Some(Color::from_rgba(1.0, 1.0, 1.0, 0.3));
            }
            style
        })
        .into()
}

/// Build a mode button: highlighted when active, sends SetMode when clicked.
fn mode_button<'a>(
    label: String,
    mode: CameraMode,
    current: CameraMode,
    is_disabled: bool,
) -> Element<'a, Message> {
    let class = if current == mode {
        cosmic::theme::Button::Suggested
    } else {
        cosmic::theme::Button::Text
    };
    let mut btn = widget::button::text(label).class(class);
    if !is_disabled {
        btn = btn.on_press(Message::SetMode(mode));
    }
    btn.into()
}

impl AppModel {
    /// Build the mode switcher widget
    ///
    /// Shows buttons for Timelapse, Video, Photo, and optionally Virtual modes.
    /// The active mode is highlighted with a suggested button style.
    /// Disabled and grayed out during transitions, recording, or streaming.
    /// Virtual mode button is only shown when virtual_camera_enabled is true.
    pub fn build_mode_switcher(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        // Disable mode switching during transitions, recording, or streaming
        let is_disabled = self.transition_state.ui_disabled
            || self.recording.is_recording()
            || self.virtual_camera.is_streaming()
            || self.timelapse.is_active();

        let mut row = widget::row()
            .push(styled_mode_button(
                mode_button(
                    fl!("mode-timelapse"),
                    CameraMode::Timelapse,
                    self.mode,
                    is_disabled,
                ),
                is_disabled,
            ))
            .push(widget::space::horizontal().width(spacing.space_xs))
            .push(styled_mode_button(
                mode_button(fl!("mode-video"), CameraMode::Video, self.mode, is_disabled),
                is_disabled,
            ))
            .push(widget::space::horizontal().width(spacing.space_xs))
            .push(styled_mode_button(
                mode_button(fl!("mode-photo"), CameraMode::Photo, self.mode, is_disabled),
                is_disabled,
            ))
            .spacing(spacing.space_xxs);

        // Only show Virtual button when the feature is enabled
        if self.config.virtual_camera_enabled {
            row = row
                .push(widget::space::horizontal().width(spacing.space_xs))
                .push(styled_mode_button(
                    mode_button(
                        fl!("mode-virtual"),
                        CameraMode::Virtual,
                        self.mode,
                        is_disabled,
                    ),
                    is_disabled,
                ));
        }

        row.into()
    }
}
