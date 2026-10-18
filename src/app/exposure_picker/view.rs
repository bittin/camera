// SPDX-License-Identifier: GPL-3.0-only

//! Exposure picker UI view
//!
//! Floating overlay for exposure controls with semi-transparent background.

use super::ControlRange;
use crate::app::overlay_style::PICKER_PANEL;
use crate::app::state::{AppModel, Message};
use crate::fl;
use cosmic::Element;
use cosmic::iced::{Alignment, Length};
use cosmic::widget;

// UI Constants
const PICKER_PANEL_WIDTH: f32 = 260.0;
const COLOR_PICKER_WIDTH: f32 = 280.0;
const LABEL_WIDTH: f32 = 70.0;
const SLIDER_WIDTH_EXPOSURE: f32 = 100.0;
const SLIDER_WIDTH_COLOR: f32 = 120.0;
const VALUE_WIDTH_EXPOSURE: f32 = 50.0;
const VALUE_WIDTH_COLOR: f32 = 40.0;
const CONTROL_SPACING: u16 = 8;
const EXPOSURE_WIDTHS: (f32, f32) = (SLIDER_WIDTH_EXPOSURE, VALUE_WIDTH_EXPOSURE);
const COLOR_WIDTHS: (f32, f32) = (SLIDER_WIDTH_COLOR, VALUE_WIDTH_COLOR);

fn integer_text(value: i32) -> String {
    value.to_string()
}

impl AppModel {
    /// Build the exposure picker overlay
    pub fn build_exposure_picker(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        let settings_data = self.exposure_settings.as_ref();
        let current_mode = settings_data.map(|s| s.mode).unwrap_or_default();
        let is_manual = current_mode == super::ExposureMode::Manual;

        let mut column = widget::Column::new()
            .spacing(spacing.space_xs)
            .padding(spacing.space_s)
            .width(Length::Shrink);

        // Mode toggle row (segmented button + reset icon)
        column = column.push(self.build_mode_toggle());

        // Add controls based on mode
        if is_manual {
            column = self.add_manual_controls(column, settings_data);
        } else {
            column = self.add_auto_controls(column, settings_data);
        }

        // Build picker panel with semi-transparent themed background
        let picker_content = widget::scrollable(column).height(
            crate::app::view::picker_panel_height(self.controls_are_sideways(), self.screen_height),
        );
        let picker_panel = widget::mouse_area(
            widget::container(self.frosted_panel(picker_content.into(), PICKER_PANEL))
                .width(Length::Fixed(PICKER_PANEL_WIDTH)),
        )
        .on_press(Message::Noop);

        // Anchor to the top bar's inner edge (below it in portrait, off the side
        // bar in landscape), shared with the tools menu and the color picker.
        self.anchor_bar_popup(picker_panel.into(), Message::CloseExposurePicker)
    }

    /// Build mode toggle row (Auto/Manual)
    fn build_mode_toggle(&self) -> Element<'_, Message> {
        let toggle = widget::segmented_button::horizontal(&self.exposure_mode_model)
            .button_alignment(Alignment::Center)
            .width(Length::Fill)
            .on_activate(Message::ExposureModeSelected);

        let reset_btn = widget::button::icon(widget::icon::from_name("edit-undo-symbolic"))
            .on_press(Message::ResetExposureSettings)
            .class(cosmic::theme::Button::Text)
            .padding(4);

        widget::Row::new()
            .push(toggle)
            .push(reset_btn)
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .into()
    }

    /// Add auto mode controls to column
    fn add_auto_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ExposureSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        if controls.exposure_bias.available {
            column = column.push(self.build_ev_row(settings_data));
        }
        if controls.backlight_compensation.available {
            column = column.push(Self::build_slider_row(
                fl!("exposure-backlight"),
                settings_data.and_then(|s| s.backlight_compensation),
                &controls.backlight_compensation,
                EXPOSURE_WIDTHS,
                integer_text,
                Message::SetBacklightCompensation,
            ));
        }

        // Focus controls (available in both auto and manual modes)
        column = self.add_focus_controls(column, settings_data);

        column
    }

    /// Add manual mode controls to column (shows all controls, disabled if unsupported)
    fn add_manual_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ExposureSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        // Unsupported manual controls stay visible, in the same order.
        for (label, current, range, format, message) in [
            (
                fl!("exposure-time"),
                settings_data.and_then(|s| s.exposure_time),
                &controls.exposure_time,
                format_exposure_time as fn(i32) -> String,
                Message::SetExposureTime as fn(i32) -> Message,
            ),
            (
                fl!("exposure-gain"),
                settings_data.and_then(|s| s.gain),
                &controls.gain,
                integer_text,
                Message::SetGain,
            ),
            (
                fl!("exposure-iso"),
                settings_data.and_then(|s| s.iso),
                &controls.iso,
                integer_text,
                Message::SetIsoSensitivity,
            ),
        ] {
            column = column.push(if range.available {
                Self::build_slider_row(label, current, range, EXPOSURE_WIDTHS, format, message)
            } else {
                Self::build_unsupported_row(label)
            });
        }

        // EV Compensation
        if controls.exposure_bias.available {
            column = column.push(self.build_ev_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-ev")));
        }

        // Metering Mode
        if controls.has_metering && !controls.metering_modes.is_empty() {
            column = column.push(self.build_metering_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-metering")));
        }

        // Auto Priority
        if controls.has_auto_priority {
            column = column.push(self.build_auto_priority_row(settings_data));
        } else {
            column = column.push(Self::build_unsupported_row(fl!("exposure-auto-priority")));
        }

        // Focus controls (available in both auto and manual modes)
        column = self.add_focus_controls(column, settings_data);

        column
    }

    // =========================================================================
    // Control row builders
    // =========================================================================

    /// Build a control row with label, slider, and value display
    fn build_control_row<'a>(
        label: String,
        value_text: String,
        value_width: f32,
        slider: impl Into<Element<'a, Message>>,
    ) -> Element<'a, Message> {
        widget::row::with_capacity(3)
            .align_y(Alignment::Center)
            .spacing(CONTROL_SPACING)
            .width(Length::Shrink)
            .push(
                widget::text(label)
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(slider.into())
            .push(
                widget::text::body(value_text)
                    .width(Length::Fixed(value_width))
                    .align_x(Alignment::End),
            )
            .into()
    }

    /// Build a generic slider row using ControlRange
    fn build_slider_row<'a, F>(
        label: String,
        current: Option<i32>,
        range: &ControlRange,
        widths: (f32, f32),
        format_value: impl Fn(i32) -> String,
        message_fn: F,
    ) -> Element<'a, Message>
    where
        F: 'a + Fn(i32) -> Message,
    {
        let current = current.unwrap_or(range.default);
        let min = range.min as f32;
        let max = range.max as f32;
        let clamped = (current as f32).clamp(min, max);

        let slider = widget::slider(min..=max, clamped, move |v| message_fn(v as i32))
            .width(Length::Fixed(widths.0));

        Self::build_control_row(label, format_value(current), widths.1, slider)
    }

    /// Build a row for unsupported controls
    fn build_unsupported_row(label: String) -> Element<'static, Message> {
        widget::row::with_capacity(2)
            .align_y(Alignment::Center)
            .spacing(CONTROL_SPACING)
            .width(Length::Shrink)
            .push(
                widget::text(label)
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(widget::text(fl!("exposure-not-supported")).size(13))
            .into()
    }

    /// Build EV compensation row
    fn build_ev_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let controls = &self.available_exposure_controls;
        let current = settings_data.map(|s| s.exposure_compensation).unwrap_or(0);
        let range = &controls.exposure_bias;

        let min = range.min as f32;
        let max = range.max as f32;
        let clamped = (current as f32).clamp(min, max);

        let slider = widget::slider(min..=max, clamped, |v| {
            Message::SetExposureCompensation(v as i32)
        })
        .width(Length::Fixed(SLIDER_WIDTH_EXPOSURE))
        .breakpoints(&[0.0]);

        Self::build_control_row(
            fl!("exposure-ev"),
            format!("{:+.1}", current as f32 / 1000.0),
            VALUE_WIDTH_EXPOSURE,
            slider,
        )
    }

    /// Build metering mode row
    fn build_metering_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let controls = &self.available_exposure_controls;
        let current_mode = settings_data
            .and_then(|s| s.metering_mode)
            .unwrap_or_default();

        let mut row = widget::Row::new()
            .push(
                widget::text(fl!("exposure-metering"))
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .spacing(4)
            .align_y(Alignment::Center)
            .width(Length::Shrink);

        for mode in &controls.metering_modes {
            let is_active = *mode == current_mode;
            let mode_copy = *mode;

            let btn = widget::button::text(mode.display_name())
                .on_press(Message::SetMeteringMode(mode_copy))
                .class(if is_active {
                    cosmic::theme::Button::Suggested
                } else {
                    cosmic::theme::Button::Text
                });

            row = row.push(btn);
        }

        row.into()
    }

    /// Build auto priority row (frame rate variation toggle)
    fn build_auto_priority_row(
        &self,
        settings_data: Option<&super::ExposureSettings>,
    ) -> Element<'_, Message> {
        let enabled = settings_data.and_then(|s| s.auto_priority).unwrap_or(false);

        widget::Row::new()
            .push(
                widget::text(fl!("exposure-auto-priority"))
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(widget::toggler(enabled).on_toggle(|_| Message::ToggleAutoExposurePriority))
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .width(Length::Shrink)
            .into()
    }

    // =========================================================================
    // Focus Controls
    // =========================================================================

    /// Add focus controls (auto toggle + manual slider)
    fn add_focus_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ExposureSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        if controls.focus.available {
            if controls.has_focus_auto {
                column = column.push(Self::build_auto_manual_row(
                    fl!("focus-auto"),
                    settings_data.and_then(|s| s.focus_auto).unwrap_or(false),
                    |_| Message::ToggleFocusAuto,
                ));
            }

            // Show manual slider when auto focus is off (or unavailable)
            let is_auto = settings_data.and_then(|s| s.focus_auto).unwrap_or(false);
            if !is_auto || !controls.has_focus_auto {
                column = column.push(Self::build_slider_row(
                    fl!("focus-position"),
                    settings_data.and_then(|s| s.focus_absolute),
                    &controls.focus,
                    EXPOSURE_WIDTHS,
                    integer_text,
                    Message::SetFocusAbsolute,
                ));
            }
        }

        column
    }

    fn build_auto_manual_row(
        label: String,
        enabled: bool,
        on_toggle: fn(bool) -> Message,
    ) -> Element<'static, Message> {
        widget::Row::new()
            .push(
                widget::text(label)
                    .size(13)
                    .width(Length::Fixed(LABEL_WIDTH)),
            )
            .push(widget::toggler(enabled).on_toggle(on_toggle))
            .push(
                widget::text(if enabled {
                    fl!("color-auto")
                } else {
                    fl!("color-manual")
                })
                .size(12),
            )
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .width(Length::Shrink)
            .into()
    }

    // =========================================================================
    // Color Picker
    // =========================================================================

    /// Build the color picker overlay
    pub fn build_color_picker(&self) -> Element<'_, Message> {
        let spacing = cosmic::theme::spacing();
        let color_data = self.color_settings.as_ref();

        let mut column = widget::Column::new()
            .spacing(spacing.space_xs)
            .padding(spacing.space_s)
            .width(Length::Shrink);

        column = column.push(self.build_color_header());
        column = self.add_image_controls(column, color_data);
        column = self.add_white_balance_controls(column, color_data);

        let picker_content = widget::scrollable(column).height(
            crate::app::view::picker_panel_height(self.controls_are_sideways(), self.screen_height),
        );
        let picker_panel = widget::mouse_area(
            widget::container(self.frosted_panel(picker_content.into(), PICKER_PANEL))
                .width(Length::Fixed(COLOR_PICKER_WIDTH)),
        )
        .on_press(Message::Noop);

        // Anchor to the top bar's inner edge (below it in portrait, off the side
        // bar in landscape), shared with the tools menu and the exposure picker.
        self.anchor_bar_popup(picker_panel.into(), Message::CloseColorPicker)
    }

    /// Build color picker header with title and reset button
    fn build_color_header(&self) -> Element<'_, Message> {
        let reset_btn = widget::button::icon(widget::icon::from_name("edit-undo-symbolic"))
            .on_press(Message::ResetColorSettings)
            .class(cosmic::theme::Button::Text)
            .padding(4);

        widget::Row::new()
            .push(
                widget::text(fl!("color-title"))
                    .size(14)
                    .width(Length::Fill),
            )
            .push(reset_btn)
            .spacing(CONTROL_SPACING)
            .align_y(Alignment::Center)
            .into()
    }

    /// Add image adjustment controls (contrast, saturation, sharpness, hue)
    fn add_image_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ColorSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        for (label, current, range, message) in [
            (
                fl!("color-contrast"),
                settings_data.and_then(|s| s.contrast),
                &controls.contrast,
                Message::SetContrast as fn(i32) -> Message,
            ),
            (
                fl!("color-saturation"),
                settings_data.and_then(|s| s.saturation),
                &controls.saturation,
                Message::SetSaturation,
            ),
            (
                fl!("color-sharpness"),
                settings_data.and_then(|s| s.sharpness),
                &controls.sharpness,
                Message::SetSharpness,
            ),
            (
                fl!("color-hue"),
                settings_data.and_then(|s| s.hue),
                &controls.hue,
                Message::SetHue,
            ),
        ] {
            if range.available {
                column = column.push(Self::build_slider_row(
                    label,
                    current,
                    range,
                    COLOR_WIDTHS,
                    integer_text,
                    message,
                ));
            }
        }

        column
    }

    /// Add white balance controls
    fn add_white_balance_controls<'a>(
        &'a self,
        mut column: widget::Column<'a, Message, cosmic::Theme>,
        settings_data: Option<&'a super::ColorSettings>,
    ) -> widget::Column<'a, Message, cosmic::Theme> {
        let controls = &self.available_exposure_controls;

        if controls.has_white_balance_auto {
            column = column.push(Self::build_auto_manual_row(
                fl!("color-white-balance"),
                settings_data
                    .and_then(|s| s.white_balance_auto)
                    .unwrap_or(true),
                |_| Message::ToggleAutoWhiteBalance,
            ));
        }

        if controls.white_balance_temperature.available {
            let is_auto = settings_data
                .and_then(|s| s.white_balance_auto)
                .unwrap_or(true);
            if !is_auto || !controls.has_white_balance_auto {
                column = column.push(Self::build_slider_row(
                    fl!("color-temperature"),
                    settings_data.and_then(|s| s.white_balance_temperature),
                    &controls.white_balance_temperature,
                    (SLIDER_WIDTH_COLOR, VALUE_WIDTH_EXPOSURE),
                    |v| format!("{v}K"),
                    Message::SetWhiteBalanceTemperature,
                ));
            }
        }

        column
    }
}

/// Format exposure time in 100µs units for display
fn format_exposure_time(time_100us: i32) -> String {
    let seconds = time_100us as f64 / 10000.0;
    if seconds < 0.5 {
        let denominator = (1.0 / seconds).round() as i32;
        format!("1/{}s", denominator)
    } else {
        format!("{:.1}s", seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::settings::widget_test_support as ui;
    use cosmic::iced::{Size, advanced::Layout};

    #[test]
    fn numeric_rows_keep_unclamped_text_default_widths_and_unsnapped_messages() {
        let renderer = ui::renderer();
        let range = ControlRange::new(0, 100, 7, 23);
        for current in [None, Some(-10), Some(110)] {
            for widths in [
                EXPOSURE_WIDTHS,
                COLOR_WIDTHS,
                (SLIDER_WIDTH_COLOR, VALUE_WIDTH_EXPOSURE),
            ] {
                let mut row = AppModel::build_slider_row(
                    "Gain".into(),
                    current,
                    &range,
                    widths,
                    integer_text,
                    Message::SetGain,
                );
                let (mut tree, node) = ui::layout(&mut row, &renderer, Size::new(400.0, 100.0));
                let texts = ui::texts(&mut row, &mut tree, &node, &renderer);
                assert_eq!(texts[1].0, current.unwrap_or(23).to_string());
                assert_eq!(node.children()[0].size().width, LABEL_WIDTH);
                assert_eq!(node.children()[1].size().width, widths.0);
                assert_eq!(node.children()[2].size().width, widths.1);
                let center = Layout::new(&node)
                    .children()
                    .nth(1)
                    .unwrap()
                    .bounds()
                    .center();
                let messages = ui::click(&mut row, &mut tree, &node, &renderer, center);
                assert!(
                    matches!(messages.as_slice(), [Message::SetGain(50)]),
                    "{messages:?}"
                );
            }
        }
        assert_eq!(format_exposure_time(100), "1/100s");
        assert_eq!(format_exposure_time(5000), "0.5s");
        assert_eq!(format_exposure_time(10000), "1.0s");
    }

    #[test]
    fn manual_controls_keep_unsupported_order_and_image_rows_keep_message_order() {
        let renderer = ui::renderer();
        let mut app = AppModel::default();
        let mut manual: Element<'_, Message> =
            app.add_manual_controls(widget::Column::new(), None).into();
        let (mut tree, node) = ui::layout(&mut manual, &renderer, Size::new(400.0, 1000.0));
        assert_eq!(node.children().len(), 6);
        let texts = ui::texts(&mut manual, &mut tree, &node, &renderer);
        for (pair, key) in texts.as_chunks::<2>().0.iter().zip([
            "exposure-time",
            "exposure-gain",
            "exposure-iso",
            "exposure-ev",
            "exposure-metering",
            "exposure-auto-priority",
        ]) {
            assert_eq!(pair[0].0, crate::i18n::LANGUAGE_LOADER.get(key));
            assert_eq!(pair[1].0, fl!("exposure-not-supported"));
        }
        drop(manual);
        app.available_exposure_controls.exposure_time = ControlRange::new(0, 100, 7, 23);
        app.available_exposure_controls.gain = ControlRange::new(0, 100, 7, 23);
        app.available_exposure_controls.iso = ControlRange::new(0, 100, 7, 23);
        let mut manual: Element<'_, Message> =
            app.add_manual_controls(widget::Column::new(), None).into();
        let (mut tree, node) = ui::layout(&mut manual, &renderer, Size::new(400.0, 1000.0));
        let mut messages = Vec::new();
        for row in Layout::new(&node).children().take(3) {
            let center = row.children().nth(1).unwrap().bounds().center();
            messages.extend(ui::click(&mut manual, &mut tree, &node, &renderer, center));
        }
        assert!(
            matches!(
                messages.as_slice(),
                [
                    Message::SetExposureTime(50),
                    Message::SetGain(50),
                    Message::SetIsoSensitivity(50)
                ]
            ),
            "{messages:?}"
        );
        drop(manual);
        app.available_exposure_controls.contrast = ControlRange::new(0, 100, 7, 23);
        app.available_exposure_controls.saturation = ControlRange::new(0, 100, 7, 23);
        app.available_exposure_controls.sharpness = ControlRange::new(0, 100, 7, 23);
        app.available_exposure_controls.hue = ControlRange::new(0, 100, 7, 23);
        let mut images: Element<'_, Message> =
            app.add_image_controls(widget::Column::new(), None).into();
        let (mut tree, node) = ui::layout(&mut images, &renderer, Size::new(400.0, 1000.0));
        assert_eq!(node.children().len(), 4);
        let mut messages = Vec::new();
        for row in Layout::new(&node).children() {
            let center = row.children().nth(1).unwrap().bounds().center();
            messages.extend(ui::click(&mut images, &mut tree, &node, &renderer, center));
        }
        assert!(
            matches!(
                messages.as_slice(),
                [
                    Message::SetContrast(50),
                    Message::SetSaturation(50),
                    Message::SetSharpness(50),
                    Message::SetHue(50)
                ]
            ),
            "{messages:?}"
        );
    }

    #[test]
    fn focus_and_white_balance_keep_distinct_missing_snapshot_defaults_and_gates() {
        let renderer = ui::renderer();
        let mut app = AppModel::default();
        for has_auto in [false, true] {
            for available in [false, true] {
                app.available_exposure_controls.has_focus_auto = has_auto;
                app.available_exposure_controls.has_white_balance_auto = has_auto;
                app.available_exposure_controls.focus = ControlRange {
                    available,
                    ..ControlRange::new(0, 100, 1, 25)
                };
                app.available_exposure_controls.white_balance_temperature = ControlRange {
                    available,
                    ..ControlRange::new(1000, 8000, 100, 4500)
                };
                let mut focus: Element<'_, Message> =
                    app.add_focus_controls(widget::Column::new(), None).into();
                let (mut tree, node) = ui::layout(&mut focus, &renderer, Size::new(400.0, 1000.0));
                assert_eq!(
                    node.children().len(),
                    if available {
                        1 + usize::from(has_auto)
                    } else {
                        0
                    }
                );
                if available && has_auto {
                    let texts = ui::texts(&mut focus, &mut tree, &node, &renderer);
                    assert!(texts.iter().any(|(text, _)| text == &fl!("color-manual")));
                    let toggle = Layout::new(&node)
                        .children()
                        .next()
                        .unwrap()
                        .children()
                        .nth(1)
                        .unwrap()
                        .bounds()
                        .center();
                    assert!(matches!(
                        ui::click(&mut focus, &mut tree, &node, &renderer, toggle).as_slice(),
                        [Message::ToggleFocusAuto]
                    ));
                }
                let mut wb: Element<'_, Message> = app
                    .add_white_balance_controls(widget::Column::new(), None)
                    .into();
                let (mut tree, node) = ui::layout(&mut wb, &renderer, Size::new(400.0, 1000.0));
                assert_eq!(
                    node.children().len(),
                    usize::from(has_auto) + usize::from(available && !has_auto)
                );
                if has_auto {
                    let texts = ui::texts(&mut wb, &mut tree, &node, &renderer);
                    assert!(texts.iter().any(|(text, _)| text == &fl!("color-auto")));
                    let toggle = Layout::new(&node)
                        .children()
                        .next()
                        .unwrap()
                        .children()
                        .nth(1)
                        .unwrap()
                        .bounds()
                        .center();
                    assert!(matches!(
                        ui::click(&mut wb, &mut tree, &node, &renderer, toggle).as_slice(),
                        [Message::ToggleAutoWhiteBalance]
                    ));
                } else if available {
                    let texts = ui::texts(&mut wb, &mut tree, &node, &renderer);
                    assert!(texts.iter().any(|(text, _)| text == "4500K"));
                }
            }
        }
    }
}
