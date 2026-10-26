// SPDX-License-Identifier: GPL-3.0-only

//! Camera switcher button widget implementation

use crate::app::state::{AppModel, Message};
use crate::constants::ui;
use cosmic::Element;
use cosmic::iced::Length;
use cosmic::widget;

/// Camera switch icon SVG (camera with circular arrows)
const CAMERA_SWITCH_ICON: &[u8] =
    include_bytes!("../../../resources/button_icons/camera-switch.svg");

/// Shared native styling for the camera switcher and capture-lane Toolbox.
pub(crate) fn capture_side_button(
    handle: widget::icon::Handle,
    message: Option<Message>,
    selected: bool,
) -> Element<'static, Message> {
    let icon = widget::icon(handle)
        .size(24)
        .class(cosmic::theme::Svg::custom(|theme| {
            cosmic::iced::widget::svg::Style {
                color: Some(cosmic::iced::Color::from(theme.cosmic().accent_color())),
            }
        }));
    let mut button = widget::button::custom(icon)
        .padding(10)
        .width(Length::Fixed(ui::PLACEHOLDER_BUTTON_WIDTH))
        .height(Length::Fixed(ui::PLACEHOLDER_BUTTON_WIDTH))
        .class(cosmic::theme::Button::Standard)
        .selected(selected);
    if let Some(message) = message {
        button = button.on_press(message);
    }
    button.into()
}

impl AppModel {
    /// Build the camera switcher button widget
    ///
    /// Shows a flip button if multiple cameras are available,
    /// otherwise shows an invisible placeholder to maintain consistent layout.
    /// Disabled and grayed out during transitions and recording.
    /// Hidden during virtual camera streaming (camera cannot be switched while streaming).
    pub fn build_camera_switcher(&self) -> Element<'_, Message> {
        let is_disabled = self.transition_state.ui_disabled
            || self.recording.is_recording()
            || self.quick_record.is_recording()
            || self.timelapse.is_active();

        // Hide camera switcher during virtual camera streaming
        if self.virtual_camera.is_streaming() {
            return widget::Space::new()
                .width(Length::Fixed(ui::PLACEHOLDER_BUTTON_WIDTH))
                .height(Length::Shrink)
                .into();
        }

        if self.available_cameras.len() > 1 {
            capture_side_button(
                widget::icon::from_svg_bytes(CAMERA_SWITCH_ICON).symbolic(true),
                (!is_disabled).then_some(Message::SwitchCamera),
                false,
            )
        } else {
            widget::Space::new()
                .width(Length::Fixed(ui::PLACEHOLDER_BUTTON_WIDTH))
                .height(Length::Shrink)
                .into()
        }
    }
}
