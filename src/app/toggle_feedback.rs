// SPDX-License-Identifier: GPL-3.0-only

//! Short-lived, localized confirmation for title-bar capture toggles.

use std::time::Instant;

use cosmic::iced::advanced::text::Wrapping;
use cosmic::iced::advanced::{
    Layout, Widget, layout, renderer,
    widget::{Id, Operation, Tree},
};
use cosmic::iced::{Alignment, Length, Point, Rectangle, Size, mouse};
use cosmic::{Element, Task, widget};

use crate::app::preview_geometry::BarInsets;
use crate::app::state::{AppModel, Message};
use crate::fl;

const FEEDBACK_DURATION_MS: u64 = 1500;
pub(super) const CAPTURE_GROUP_ID: &str = "title-bar-capture-group";

#[derive(Default)]
struct CaptureGroupBounds(Option<Rectangle>);

impl Operation for CaptureGroupBounds {
    fn traverse(&mut self, visit: &mut dyn FnMut(&mut dyn Operation)) {
        visit(self);
    }

    fn container(&mut self, id: Option<&Id>, bounds: Rectangle) {
        if id == Some(&Id::new(CAPTURE_GROUP_ID)) {
            self.0 = Some(bounds);
        }
    }
}

/// Measure the real title bar without drawing or activating a second copy.
/// This follows format visibility, font metrics and window resizes in the same
/// layout pass, without hard-coded button heights or a previous-frame anchor.
struct AnchoredFeedback<'a> {
    panel: Element<'a, Message>,
    title_bar: Element<'a, Message>,
    insets: BarInsets,
    sideways: bool,
    right: bool,
}

impl Widget<Message, cosmic::Theme, cosmic::Renderer> for AnchoredFeedback<'_> {
    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.panel), Tree::new(&self.title_bar)]
    }

    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(&mut [&mut self.panel, &mut self.title_bar]);
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &cosmic::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.resolve(Length::Fill, Length::Fill, Size::ZERO);
        let bar = self.title_bar.as_widget_mut().layout(
            &mut tree.children[1],
            renderer,
            &layout::Limits::new(Size::ZERO, size),
        );
        let mut capture = CaptureGroupBounds::default();
        self.title_bar.as_widget_mut().operate(
            &mut tree.children[1],
            Layout::new(&bar),
            renderer,
            &mut capture,
        );
        let capture = capture.0.expect("visible title-bar capture group");
        let gap = f32::from(cosmic::theme::spacing().space_s);
        let left = self.insets.left + gap;
        let top = self.insets.top + gap;
        let available = Size::new(
            (size.width - left - self.insets.right - gap).max(0.0),
            (size.height - top - self.insets.bottom - gap).max(0.0),
        );
        let panel = self.panel.as_widget_mut().layout(
            &mut tree.children[0],
            renderer,
            &layout::Limits::new(Size::ZERO, available),
        );
        let panel_size = panel.size();
        let x = if self.sideways {
            if self.right {
                left + available.width - panel_size.width
            } else {
                left
            }
        } else {
            (capture.center_x() - panel_size.width / 2.0)
                .clamp(left, left + (available.width - panel_size.width).max(0.0))
        };
        let y = if self.sideways {
            // Keep the group center even for a single toggle near the end.
            // The bar's inset already provides clearance; only clamp against
            // the viewport when a tall translated message would be clipped.
            (capture.center_y() - panel_size.height / 2.0)
                .clamp(0.0, (size.height - panel_size.height).max(0.0))
        } else {
            top
        };
        layout::Node::with_children(size, vec![panel.move_to(Point::new(x, y))])
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut cosmic::Renderer,
        theme: &cosmic::Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.panel.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout.children().next().unwrap(),
            cursor,
            viewport,
        );
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &cosmic::Renderer,
        operation: &mut dyn Operation,
    ) {
        self.panel.as_widget_mut().operate(
            &mut tree.children[0],
            layout.children().next().unwrap(),
            renderer,
            operation,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmic::iced::advanced::{layout, renderer::Headless, widget::Tree};
    use cosmic::iced::{Font, Pixels, Size};

    fn renderer() -> cosmic::Renderer {
        pollster::block_on(cosmic::Renderer::new(
            Font::DEFAULT,
            Pixels(14.0),
            Some("tiny-skia"),
        ))
        .expect("headless text renderer")
    }

    fn layout_element(
        mut element: Element<'_, Message>,
        renderer: &cosmic::Renderer,
        size: Size,
    ) -> layout::Node {
        let mut tree = Tree::new(element.as_widget());
        element
            .as_widget_mut()
            .layout(&mut tree, renderer, &layout::Limits::new(Size::ZERO, size))
    }

    fn assert_fits(node: &layout::Node) {
        for child in node.children() {
            let bounds = child.bounds();
            assert!(bounds.x >= -0.01 && bounds.y >= -0.01);
            assert!(
                bounds.x + bounds.width <= node.size().width + 0.01,
                "{bounds:?} outside {:?}",
                node.size()
            );
            assert!(
                bounds.y + bounds.height <= node.size().height + 0.01,
                "{bounds:?} outside {:?}",
                node.size()
            );
            assert_fits(child);
        }
    }

    #[test]
    fn confirmation_fits_minimum_height_and_rotated_bars() {
        use crate::backends::display_orientation::DisplayOrientation;
        let renderer = renderer();
        for (orientation, size) in [
            (DisplayOrientation::Rotate0, Size::new(320.0, 180.0)),
            (DisplayOrientation::Rotate0, Size::new(360.0, 720.0)),
            (DisplayOrientation::Rotate90, Size::new(720.0, 360.0)),
            (DisplayOrientation::Rotate270, Size::new(720.0, 360.0)),
            (DisplayOrientation::Rotate90, Size::new(720.0, 180.0)),
            (DisplayOrientation::Rotate270, Size::new(720.0, 180.0)),
        ] {
            let mut model = AppModel {
                display_orientation: orientation,
                ..Default::default()
            };
            let _ = model.update(Message::ToggleFlash);
            let node = layout_element(model.build_toggle_feedback().unwrap(), &renderer, size);
            assert_fits(&node);
            let chip = &node.children()[0];
            assert!(
                chip.size().height >= 24.0,
                "confirmation squeezed out at {size:?}: {:?}",
                chip.size()
            );
            let insets = model.bar_insets();
            assert!(chip.bounds().y >= insets.top);
            assert!(chip.bounds().x >= insets.left);
            assert!(chip.bounds().x + chip.size().width <= size.width - insets.right);
            let bar = layout_element(model.build_top_bar(), &renderer, size);
            let capture = &bar.children()[0].children()[1];
            if model.controls_are_sideways() {
                assert!(
                    (chip.bounds().center_y() - capture.bounds().center_y()).abs() < 0.01,
                    "confirmation must track the capture group, not the window center: {:?} vs {:?}",
                    chip.bounds(),
                    capture.bounds()
                );
            } else {
                assert!((chip.bounds().center_x() - capture.bounds().center_x()).abs() < 0.01);
            }
        }
    }

    #[test]
    fn title_bar_capture_group_is_centered_upright_and_end_aligned_sideways() {
        use crate::app::window_controls::WindowControls;
        use crate::backends::display_orientation::DisplayOrientation;
        use crate::config::PhotoOutputFormat;

        let renderer = renderer();
        for (orientation, size) in [
            (DisplayOrientation::Rotate0, Size::new(320.0, 180.0)),
            (DisplayOrientation::Rotate0, Size::new(360.0, 720.0)),
            (DisplayOrientation::Rotate0, Size::new(1280.0, 720.0)),
            (DisplayOrientation::Rotate90, Size::new(720.0, 360.0)),
            (DisplayOrientation::Rotate270, Size::new(720.0, 360.0)),
            (DisplayOrientation::Rotate90, Size::new(720.0, 180.0)),
            (DisplayOrientation::Rotate270, Size::new(720.0, 180.0)),
        ] {
            for layout in ["appmenu:", ":close", ":minimize,maximize,close"] {
                for format in [
                    PhotoOutputFormat::Jpeg,
                    PhotoOutputFormat::Avif,
                    PhotoOutputFormat::Png,
                ] {
                    let mut model = AppModel {
                        display_orientation: orientation,
                        window_controls: WindowControls::from_button_layout(layout),
                        ..Default::default()
                    };
                    model.config.photo_output_format = format;
                    for picker_open in [false, true] {
                        model.format_picker_visible = picker_open;
                        let bar = layout_element(model.build_top_bar(), &renderer, size);
                        let groups = &bar.children()[0];
                        assert_eq!(groups.children().len(), 3);
                        let capture = &groups.children()[1];
                        assert_eq!(
                            capture.children()[0].children().len(),
                            if format == PhotoOutputFormat::Png {
                                1
                            } else {
                                3
                            }
                        );
                        assert!(capture.size().width > 0.0 && capture.size().height > 0.0);
                        let (actual, expected) = if model.controls_are_sideways() {
                            let format_slot = &groups.children()[2];
                            let format = &format_slot.children()[0];
                            assert!(
                                (format_slot.size().height - format.size().height).abs() < 0.01,
                                "format slot must not reserve flexible space below capture controls"
                            );
                            assert!(
                                (format_slot.bounds().y + format_slot.size().height
                                    - (size.height - 7.0))
                                    .abs()
                                    < 0.01,
                                "format group must stay at the end of the sideways bar"
                            );
                            (
                                capture.bounds().y + capture.size().height,
                                format_slot.bounds().y,
                            )
                        } else {
                            (capture.bounds().center_x(), size.width / 2.0)
                        };
                        assert!(
                            (actual - expected).abs() < 0.01,
                            "capture group shifted: {actual} != {expected}, {orientation:?}, {size:?}, {layout}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn confirmation_sizes_to_text_and_wraps_long_translations() {
        let renderer = renderer();
        let roomy = Size::new(600.0, f32::INFINITY);
        let short = layout_element(feedback_chip("FLASH ON".into(), true), &renderer, roomy);
        let longer = layout_element(
            feedback_chip("BEWEGUNGSAUFNAHME EINGESCHALTET".into(), true),
            &renderer,
            roomy,
        );
        assert!(longer.size().width > short.size().width);
        assert!(longer.size().width < roomy.width);
        for text in [
            "BEWEGUNGSAUFNAHME EINGESCHALTET",
            "BEWEGUNGSAUFNAHME_EINGESCHALTET_MIT_LANGER_UEBERSETZUNG",
            "تم تشغيل الصور المتحركة",
            "动态照片已开启",
        ] {
            let wide = layout_element(feedback_chip(text.into(), true), &renderer, roomy);
            let narrow = layout_element(
                feedback_chip(text.into(), true),
                &renderer,
                Size::new(132.0, f32::INFINITY),
            );
            assert_fits(&narrow);
            assert!(narrow.size().width <= 132.0);
            if text.starts_with("BEWEGUNG") {
                assert!(
                    narrow.size().height > wide.size().height,
                    "long translation did not wrap"
                );
            }
        }
    }

    #[test]
    fn confirmations_follow_group_size_and_resize_without_a_stale_anchor() {
        use crate::app::settings::widget_test_support;
        use crate::backends::display_orientation::DisplayOrientation;
        use crate::config::PhotoOutputFormat;
        let renderer = renderer();
        for orientation in [DisplayOrientation::Rotate90, DisplayOrientation::Rotate270] {
            for format in [PhotoOutputFormat::Jpeg, PhotoOutputFormat::Png] {
                for kind in [ToggleKind::Flash, ToggleKind::MotionPhoto] {
                    if format == PhotoOutputFormat::Png && kind == ToggleKind::MotionPhoto {
                        continue;
                    }
                    for enabled in [false, true] {
                        let mut model = AppModel {
                            display_orientation: orientation,
                            ..Default::default()
                        };
                        model.config.photo_output_format = format;
                        model.config.motion_photo = enabled;
                        model.flash.enabled = enabled;
                        let _ = model.show_toggle_feedback(kind);
                        let mut feedback = model.build_toggle_feedback().unwrap();
                        // Reuse the same widget/tree across resizes to catch stale anchors.
                        let mut tree = Tree::new(feedback.as_widget());
                        for height in [360.0, 180.0, 480.0, 360.0] {
                            let size = Size::new(720.0, height);
                            feedback.as_widget_mut().diff(&mut tree);
                            let node = feedback.as_widget_mut().layout(
                                &mut tree,
                                &renderer,
                                &layout::Limits::new(Size::ZERO, size),
                            );
                            assert_fits(&node);
                            let chip = &node.children()[0];
                            let bar = layout_element(model.build_top_bar(), &renderer, size);
                            let capture = &bar.children()[0].children()[1];
                            // A lone capture toggle can sit closer to the edge than
                            // half the label height. Keep the label inside the viewport.
                            let expected_top = (capture.bounds().center_y()
                                - chip.size().height / 2.0)
                                .clamp(0.0, (height - chip.size().height).max(0.0));
                            assert!(
                                (chip.bounds().y - expected_top).abs() < 0.01,
                                "{orientation:?}, {format:?}, {kind:?}: {:?} vs {:?}",
                                chip.bounds(),
                                capture.bounds()
                            );
                            let texts = widget_test_support::texts(
                                &mut feedback,
                                &mut tree,
                                &node,
                                &renderer,
                            );
                            assert_eq!(
                                texts.len(),
                                1,
                                "hidden reference buttons must not be exposed"
                            );
                            assert!(
                                (texts[0].1.center_y() - chip.bounds().center_y()).abs() < 0.01,
                                "confirmation text itself must be centered inside its chip"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn confirmation_uses_theme_color_pairs() {
        for theme in [
            cosmic::Theme::dark(),
            cosmic::Theme::light(),
            cosmic::Theme::dark_hc(),
            cosmic::Theme::light_hc(),
        ] {
            for enabled in [true, false] {
                let style = feedback_style(&theme, enabled);
                assert_eq!(
                    style.background, None,
                    "the shared overlay panel must own the surface"
                );
                assert_eq!(
                    style.text_color,
                    enabled.then(|| crate::app::overlay_style::chrome_accent_color(&theme))
                );
            }
        }
    }

    #[tokio::test]
    async fn confirmation_timeout_dismisses_without_preview_frames() {
        use futures::StreamExt;
        let mut model = AppModel::default();
        let mut stream =
            cosmic::iced::runtime::task::into_stream(model.handle_toggle_flash()).unwrap();
        let action = tokio::time::timeout(std::time::Duration::from_secs(3), stream.next())
            .await
            .unwrap()
            .unwrap();
        let cosmic::iced::runtime::Action::Output(cosmic::Action::App(message)) = action else {
            panic!("expected app timeout");
        };
        assert!(matches!(message, Message::DismissToggleFeedback(_)));
        let _ = model.update(message);
        assert!(model.toggle_feedback.is_none());
    }

    #[test]
    fn stale_timeout_does_not_dismiss_newer_confirmation() {
        let mut model = AppModel::default();
        let stale = Instant::now() - std::time::Duration::from_secs(1);
        let _ = model.update(Message::ToggleFlash);
        let current = model.toggle_feedback.unwrap().shown_at;
        let _ = model.update(Message::DismissToggleFeedback(stale));
        assert_eq!(model.toggle_feedback.unwrap().shown_at, current);
        let _ = model.update(Message::DismissToggleFeedback(current));
        assert!(model.toggle_feedback.is_none());
    }

    #[test]
    fn unavailable_motion_photo_does_not_show_confirmation() {
        let mut model = AppModel::default();
        for format in [
            crate::config::PhotoOutputFormat::Png,
            crate::config::PhotoOutputFormat::Dng,
        ] {
            model.config.photo_output_format = format;
            let _ = model.update(Message::ToggleMotionPhotoCapture);
            assert!(model.toggle_feedback.is_none());
        }
    }

    #[test]
    fn confirmation_disappears_when_capture_controls_leave_the_current_mode() {
        for kind in [ToggleKind::Flash, ToggleKind::MotionPhoto, ToggleKind::Hdr] {
            let mut model = AppModel::default();
            let _ = model.show_toggle_feedback(kind);
            assert!(model.build_toggle_feedback().is_some());
            for mode in [
                crate::app::CameraMode::Virtual,
                crate::app::CameraMode::Video,
                crate::app::CameraMode::View,
            ] {
                model.mode = mode;
                assert!(
                    model.build_toggle_feedback().is_none(),
                    "stale {kind:?} confirmation in {mode:?}"
                );
            }
        }
    }

    #[test]
    fn motion_photo_toggle_shows_confirmation_in_both_directions() {
        let mut model = AppModel::default();
        for enabled in [true, false] {
            model.toggle_feedback = None;
            let _ = model.update(Message::ToggleMotionPhotoCapture);
            assert_eq!(model.config.motion_photo, enabled);
            assert!(model.toggle_feedback.is_some());
            assert!(model.build_toggle_feedback().is_some());
        }
    }

    #[test]
    fn hdr_toggle_shows_confirmation_in_both_directions() {
        let mut model = AppModel::default();
        model.config.burst_mode_setting = crate::config::BurstModeSetting::Frames4;
        for enabled in [false, true] {
            model.toggle_feedback = None;
            let _ = model.update(Message::ToggleBurstMode);
            assert_eq!(model.would_use_burst_mode(), enabled);
            assert!(model.toggle_feedback.is_some());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToggleKind {
    Flash,
    MotionPhoto,
    Hdr,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToggleFeedback {
    pub kind: ToggleKind,
    pub shown_at: Instant,
}

fn feedback_style(theme: &cosmic::Theme, enabled: bool) -> widget::container::Style {
    widget::container::Style {
        // Leave the surface and off-state foreground to the shared overlay panel.
        text_color: enabled.then(|| crate::app::overlay_style::chrome_accent_color(theme)),
        ..Default::default()
    }
}

fn feedback_chip(text: String, enabled: bool) -> Element<'static, Message> {
    let spacing = cosmic::theme::spacing();
    widget::container(
        widget::text::body(text)
            .align_x(Alignment::Center)
            .wrapping(Wrapping::WordOrGlyph),
    )
    .padding([spacing.space_xs, spacing.space_s])
    .style(move |theme| feedback_style(theme, enabled))
    .into()
}

impl AppModel {
    pub(crate) fn show_toggle_feedback(
        &mut self,
        kind: ToggleKind,
    ) -> Task<cosmic::Action<Message>> {
        let shown_at = Instant::now();
        self.toggle_feedback = Some(ToggleFeedback { kind, shown_at });
        Self::delay_task(
            FEEDBACK_DURATION_MS,
            Message::DismissToggleFeedback(shown_at),
        )
    }

    pub(crate) fn dismiss_toggle_feedback(&mut self, shown_at: Instant) {
        if self
            .toggle_feedback
            .is_some_and(|feedback| feedback.shown_at == shown_at)
        {
            self.toggle_feedback = None;
        }
    }

    pub(crate) fn build_toggle_feedback(&self) -> Option<Element<'_, Message>> {
        if self.ui_hidden || self.mode.is_view_only() || self.pending_close {
            return None;
        }
        let feedback = self.toggle_feedback?;
        // Match the owning controls' mode gates, including hardware-only torch
        // modes. A short-lived label must not survive its controls disappearing.
        let available = match feedback.kind {
            ToggleKind::Flash | ToggleKind::Hdr => self.should_show_flash_button(),
            ToggleKind::MotionPhoto => self.should_show_motion_photo_button(),
        };
        if !available {
            return None;
        }
        let (text, enabled) = match feedback.kind {
            ToggleKind::Flash => (
                if self.flash.enabled {
                    fl!("toggle-flash-on")
                } else {
                    fl!("toggle-flash-off")
                },
                self.flash.enabled,
            ),
            ToggleKind::MotionPhoto => (
                if self.config.motion_photo {
                    fl!("toggle-motion-on")
                } else {
                    fl!("toggle-motion-off")
                },
                self.config.motion_photo,
            ),
            ToggleKind::Hdr => (
                if self.would_use_burst_mode() {
                    fl!("toggle-hdr-on")
                } else {
                    fl!("toggle-hdr-off")
                },
                self.would_use_burst_mode(),
            ),
        };
        let mut insets = Self::map_bar_insets(
            self.controls_bar_layout(),
            self.top_ui_height(),
            self.bottom_ui_height(),
        );
        // Confirmation floats below the title bar; the bottom controls must not
        // squeeze it out of a short window. Still avoid both side strips.
        insets.bottom = 0.0;
        Some(Element::new(AnchoredFeedback {
            panel: self.frosted_panel(
                feedback_chip(text, enabled),
                crate::app::overlay_style::POPUP_PANEL,
            ),
            title_bar: self.build_top_bar(),
            insets,
            sideways: self.controls_are_sideways(),
            right: self.controls_bar_layout().top_bar == crate::app::bar_layout::Edge::Right,
        }))
    }
}
