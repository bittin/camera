// SPDX-License-Identifier: GPL-3.0-only

//! Capture button widget implementation

use crate::app::state::{AppModel, CameraMode, Message, PhotoTimerSetting};
use crate::constants::ui;
use cosmic::Element;
use cosmic::iced::{Background, Color, Length};
use cosmic::widget;

/// Ring border thickness for the capture button.
const RING_WIDTH: f32 = 3.0;
/// Gap between the ring's inner edge and the filled circle.
const RING_GAP: f32 = 5.0;
/// Reference element size that the theme's `radius_xl` is designed for.
const RADIUS_XL_REFERENCE_SIZE: f32 = 48.0;
/// Size ratio of the photo-during-recording button relative to the main capture button.
const PHOTO_BTN_SIZE_RATIO: f32 = 0.55;

/// Outer size (width and height) of the capture button at scale 1.0.
/// Exposed so the bottom-bar scrim can match the actual capture-button area.
pub const CAPTURE_BUTTON_OUTER_SIZE: f32 = ui::CAPTURE_BUTTON_INNER + (RING_GAP + RING_WIDTH) * 2.0;

/// Get the theme-based corner radius for a given element size.
/// Scales the cosmic theme's radius_xl proportionally to the element size,
/// capped so it never exceeds a full circle.
fn theme_radius(corner_radius: f32, size: f32) -> f32 {
    // Scale theme radius proportionally (radius_xl is designed for ~48px elements)
    let scaled = corner_radius * (size / RADIUS_XL_REFERENCE_SIZE);
    scaled.min(size / 2.0)
}

/// Build a capture button with a static ring around an animated filled circle.
/// `button_size` scales the entire button uniformly (ring, gap, fill, border).
/// `anim_scale` only scales the inner filled circle (for press animation).
///
/// Layout (inside out):
///   fill circle (INNER * button_size * anim_scale)
///   → centered in ring inner area (INNER + GAP*2) * button_size
///   → ring border (RING_WIDTH * button_size)
fn build_ringed_circle(
    color: Color,
    ring_color: Color,
    button_size: f32,
    anim_scale: f32,
    loading: bool,
) -> Element<'static, Message> {
    let base_inner = ui::CAPTURE_BUTTON_INNER * button_size;
    let border_w = RING_WIDTH * button_size;

    // Fetch the theme once for corner radius and background luminance.
    let theme = cosmic::theme::active();
    let cosmic_theme = theme.cosmic();
    let corner_radius = cosmic_theme.corner_radii.radius_xl[0];

    let fill_size = base_inner * anim_scale;
    let fill_radius = theme_radius(corner_radius, fill_size);

    // Compensate for irradiation illusion: bright backgrounds make gaps between
    // dark elements appear wider. Reduce the gap on light themes so the button
    // looks perceptually identical across themes.
    let bg = cosmic_theme.bg_color();
    let bg_luminance = 0.299 * bg.red + 0.587 * bg.green + 0.114 * bg.blue;
    let gap = RING_GAP * button_size * (1.0 - 0.25 * bg_luminance);

    let ring_inner = base_inner + gap * 2.0;
    let ring_outer = ring_inner + border_w * 2.0;
    let ring_radius = theme_radius(corner_radius, ring_outer);

    // Filled circle
    let circle = widget::container(
        widget::Space::new()
            .width(Length::Fixed(fill_size))
            .height(Length::Fixed(fill_size)),
    )
    .style(move |_theme| widget::container::Style {
        background: Some(Background::Color(color)),
        border: cosmic::iced::Border {
            radius: [fill_radius; 4].into(),
            ..Default::default()
        },
        ..Default::default()
    });

    let circle: Element<'static, Message> = if loading {
        let spinner = widget::container(
            widget::progress_bar::indeterminate_circular()
                .size(base_inner * 0.55)
                .bar_color(Color::from_rgb(0.55, 0.55, 0.55))
                .track_color(Color::TRANSPARENT)
                .border_color(Color::TRANSPARENT),
        )
        .width(Length::Fixed(fill_size))
        .height(Length::Fixed(fill_size))
        .center_x(fill_size)
        .center_y(fill_size);
        cosmic::iced::widget::stack![circle, spinner].into()
    } else {
        circle.into()
    };

    // Ring with fill centered inside
    let ringed = widget::container(
        widget::container(circle)
            .width(Length::Fixed(ring_inner))
            .height(Length::Fixed(ring_inner))
            .center_x(ring_inner)
            .center_y(ring_inner)
            .style(|_| widget::container::Style::default()),
    )
    .width(Length::Fixed(ring_outer))
    .height(Length::Fixed(ring_outer))
    .center_x(ring_outer)
    .center_y(ring_outer)
    .style(move |_theme| widget::container::Style {
        border: cosmic::iced::Border {
            radius: [ring_radius; 4].into(),
            width: border_w,
            color: ring_color,
        },
        ..Default::default()
    });

    // Stable layout container (doesn't change size with animation)
    widget::container(ringed)
        .width(Length::Fixed(ring_outer))
        .height(Length::Fixed(ring_outer))
        .center_x(ring_outer)
        .center_y(ring_outer)
        .style(|_| widget::container::Style::default())
        .into()
}

impl AppModel {
    /// Count accepted photos, including the frame currently being acquired.
    fn photo_queue_at_capacity(&self) -> bool {
        self.photo_jobs
            .len()
            .saturating_add(usize::from(self.hardware_photo_id.is_some()))
            >= 8
    }

    /// Animate capture button scale toward a new target.
    /// Call this whenever the button's target scale changes.
    pub fn animate_capture_scale(&mut self, target: f32) {
        if (self.capture_scale_to - target).abs() < 0.001 {
            return; // already targeting this scale
        }
        // Start from the current animated scale
        self.capture_scale_from = self.current_capture_scale();
        self.capture_scale_to = target;
        self.capture_anim_start = Some(std::time::Instant::now());
    }

    /// Get the current animated capture button scale.
    pub fn current_capture_scale(&self) -> f32 {
        Self::ease_scale(
            self.capture_anim_start,
            self.capture_scale_from,
            self.capture_scale_to,
        )
    }

    /// Animate the photo-during-recording button scale toward a new target.
    pub fn animate_photo_btn_scale(&mut self, target: f32) {
        if (self.photo_btn_scale_to - target).abs() < 0.001 {
            return;
        }
        self.photo_btn_scale_from = self.current_photo_btn_scale();
        self.photo_btn_scale_to = target;
        self.photo_btn_anim_start = Some(std::time::Instant::now());
    }

    /// Get the current animated photo-during-recording button scale.
    pub fn current_photo_btn_scale(&self) -> f32 {
        Self::ease_scale(
            self.photo_btn_anim_start,
            self.photo_btn_scale_from,
            self.photo_btn_scale_to,
        )
    }

    /// Shared ease-out cubic interpolation for button scale animations.
    fn ease_scale(start: Option<std::time::Instant>, from: f32, to: f32) -> f32 {
        if let Some(start) = start {
            let duration_ms = if to < from { 120.0 } else { 200.0 };
            let t = (start.elapsed().as_secs_f32() * 1000.0 / duration_ms).min(1.0);
            let eased = 1.0 - (1.0 - t).powi(3); // ease-out cubic
            from + (to - from) * eased
        } else {
            to
        }
    }

    /// Build the interactive capture circle (Shrink width, no outer Fill container).
    /// Used by both the centered layout and the recording row layout.
    pub(crate) fn build_capture_circle(&self) -> Element<'_, Message> {
        // Acquisition alone does not grey the shutter. Handlers still silently
        // reject overlapping requests; only a full queue shows the spinner.
        let photo_action = self.mode == CameraMode::Photo
            && !self.quick_record.is_recording()
            && self.photo_timer_countdown.is_none();
        let loading = photo_action && self.photo_queue_at_capacity();
        let is_disabled = loading
            || ((self.pending_close
                || self.transition_state.ui_disabled
                || self.burst_mode.is_active())
                && !self.quick_record.is_recording()
                && !(self.mode == CameraMode::Video && self.recording.is_recording())
                && self.photo_timer_countdown.is_none());

        let theme = cosmic::theme::active();
        let cosmic_theme = theme.cosmic();
        let accent: Color = cosmic_theme.accent_color().into();
        let destructive: Color = cosmic_theme.destructive_color().into();

        let color = if is_disabled {
            Color::from_rgba(0.5, 0.5, 0.5, 0.3)
        } else {
            match self.mode {
                CameraMode::Photo => {
                    if self.quick_record.is_recording() {
                        destructive
                    } else {
                        accent
                    }
                }
                CameraMode::Video => destructive,
                CameraMode::Timelapse => destructive,
                // View hides the capture button entirely; this color is
                // only used if the build path is reached, which it isn't.
                CameraMode::View => accent,
                CameraMode::Virtual => {
                    if self.virtual_camera.is_streaming() {
                        Color::from_rgb(0.1, 0.7, 0.2)
                    } else {
                        Color::from_rgb(0.2, 0.5, 0.9)
                    }
                }
            }
        };

        let scale = self.current_capture_scale();
        let base_circle = build_ringed_circle(color, color, 1.0, scale, loading);

        // Overlay timer seconds on the capture button when timer is set in Photo mode
        let circle: Element<'_, Message> = if !loading
            && self.mode == CameraMode::Photo
            && self.photo_timer_setting != PhotoTimerSetting::Off
            && self.photo_timer_countdown.is_none()
        {
            let seconds = self.photo_timer_setting.seconds();
            let ring_outer = (ui::CAPTURE_BUTTON_INNER + RING_GAP * 2.0) + RING_WIDTH * 2.0;
            let timer_label = widget::text(format!("{seconds}"))
                .size(32.0)
                .font(cosmic::font::bold())
                .center()
                .width(Length::Fill)
                .height(Length::Fill);
            let timer_overlay = widget::container(timer_label)
                .width(Length::Fixed(ring_outer))
                .height(Length::Fixed(ring_outer))
                .center_x(ring_outer)
                .center_y(ring_outer);
            cosmic::iced::widget::stack![base_circle, timer_overlay].into()
        } else {
            base_circle
        };

        if is_disabled {
            circle
        } else {
            let press_message = match self.mode {
                CameraMode::Photo => Message::CaptureButtonPressed,
                CameraMode::Video => Message::ToggleRecording,
                CameraMode::Virtual => Message::ToggleVirtualCamera,
                CameraMode::Timelapse => Message::ToggleTimelapse,
                // View mode hides the capture button entirely (see
                // `view::capture_button_only`); this branch is unreachable
                // in practice but needs to compile.
                CameraMode::View => Message::Noop,
            };
            let mut area = widget::mouse_area(circle)
                .on_press(press_message)
                .interaction(cosmic::iced::mouse::Interaction::Pointer);
            if self.mode == CameraMode::Photo {
                area = area.on_release(Message::CaptureButtonReleased);
            }
            area.into()
        }
    }

    /// Build the capture lane. Toolbox shares the still-during-video slot,
    /// yielding that slot to photo capture while recording/streaming.
    pub fn build_capture_button(&self) -> Element<'_, Message> {
        let recording_layout = (self.recording.is_recording() && !self.quick_record.is_recording())
            || self.virtual_camera.is_streaming();
        let right = if recording_layout {
            self.build_photo_during_recording_button()
        } else {
            self.build_capture_tools_button()
        };
        self.build_capture_row(
            self.build_video_play_pause_button(),
            self.build_capture_circle(),
            right,
        )
    }

    /// Build the photo capture button shown during video recording (right side).
    pub fn build_photo_during_recording_button(&self) -> Element<'_, Message> {
        let scale = self.current_photo_btn_scale();
        let theme = cosmic::theme::active();
        let cosmic_theme = theme.cosmic();
        let accent: Color = cosmic_theme.accent_color().into();
        let loading = self.photo_queue_at_capacity();
        let is_disabled = loading
            || self.pending_close
            || self.transition_state.ui_disabled
            || self.burst_mode.is_active();
        let color = if is_disabled {
            Color::from_rgba(0.5, 0.5, 0.5, 0.3)
        } else {
            accent
        };
        let circle = build_ringed_circle(color, color, PHOTO_BTN_SIZE_RATIO, scale, loading);
        if is_disabled {
            return circle;
        }

        widget::mouse_area(circle)
            .on_press(Message::Capture)
            .interaction(cosmic::iced::mouse::Interaction::Pointer)
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmic::iced::advanced::{
        Layout, Shell, clipboard, layout, renderer::Headless, widget::Tree,
    };
    use cosmic::iced::{Event, Font, Pixels, Point, Rectangle, Size, mouse, window};

    fn renderer() -> cosmic::Renderer {
        pollster::block_on(cosmic::Renderer::new(
            Font::DEFAULT,
            Pixels(14.0),
            Some("tiny-skia"),
        ))
        .expect("headless software renderer")
    }

    fn find_tools_button(
        element: Element<'_, Message>,
        renderer: &cosmic::Renderer,
        size: Size,
    ) -> Option<Point> {
        find_action_button(element, renderer, size, |message| {
            matches!(message, Message::ToggleToolsMenu)
        })
    }

    fn find_action_button(
        mut element: Element<'_, Message>,
        renderer: &cosmic::Renderer,
        size: Size,
        matches_action: impl Fn(&Message) -> bool,
    ) -> Option<Point> {
        let mut tree = Tree::new(element.as_widget());
        let node = element.as_widget_mut().layout(
            &mut tree,
            renderer,
            &layout::Limits::new(Size::ZERO, size),
        );
        let viewport = Rectangle::with_size(size);
        for y in (4..size.height as usize).step_by(8) {
            for x in (4..size.width as usize).step_by(8) {
                let point = Point::new(x as f32, y as f32);
                let mut messages = Vec::new();
                for event in [
                    Event::Window(window::Event::RedrawRequested(std::time::Instant::now())),
                    Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
                    Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
                ] {
                    let mut shell = Shell::new(&mut messages);
                    element.as_widget_mut().update(
                        &mut tree,
                        &event,
                        Layout::new(&node),
                        mouse::Cursor::Available(point),
                        renderer,
                        &mut clipboard::Null,
                        &mut shell,
                        &viewport,
                    );
                }
                if messages.iter().any(&matches_action) {
                    return Some(point);
                }
            }
        }
        None
    }

    #[test]
    fn toolbox_moves_from_title_bar_to_beside_the_shutter() {
        let renderer = renderer();
        let size = Size::new(640.0, 180.0);
        for mode in [
            CameraMode::Photo,
            CameraMode::Video,
            CameraMode::Timelapse,
            CameraMode::Virtual,
        ] {
            let mut model = AppModel {
                mode,
                ..Default::default()
            };
            let tools = find_tools_button(model.build_capture_button(), &renderer, size)
                .expect("toolbox must be available beside the shutter");
            assert!(tools.x > size.width / 2.0);
            assert!(
                find_tools_button(model.build_top_bar(), &renderer, size).is_none(),
                "toolbox must not remain duplicated in the title bar"
            );
            model.transition_state.ui_disabled = true;
            assert!(find_tools_button(model.build_capture_button(), &renderer, size).is_none());
        }
    }

    #[test]
    fn toolbox_uses_the_camera_switch_button_footprint() {
        let renderer = renderer();
        let model = AppModel::default();
        let (messages, _, size) = exercise(model.build_capture_tools_button(), &renderer);
        assert_eq!(
            size,
            Size::new(ui::PLACEHOLDER_BUTTON_WIDTH, ui::PLACEHOLDER_BUTTON_WIDTH)
        );
        assert!(
            messages
                .iter()
                .any(|message| matches!(message, Message::ToggleToolsMenu))
        );
    }

    #[test]
    fn toolbox_and_recording_photo_share_the_same_slot_in_each_rotation() {
        use crate::app::state::RecordingState;
        use crate::backends::display_orientation::DisplayOrientation;
        let renderer = renderer();
        for (orientation, size) in [
            (DisplayOrientation::Rotate0, Size::new(320.0, 100.0)),
            (DisplayOrientation::Rotate0, Size::new(640.0, 100.0)),
            (DisplayOrientation::Rotate90, Size::new(100.0, 360.0)),
            (DisplayOrientation::Rotate270, Size::new(100.0, 360.0)),
            (DisplayOrientation::Rotate90, Size::new(100.0, 180.0)),
            (DisplayOrientation::Rotate270, Size::new(100.0, 180.0)),
        ] {
            let mut model = AppModel {
                mode: CameraMode::Video,
                display_orientation: orientation,
                ..Default::default()
            };
            let layout = |mut element: Element<'_, Message>| {
                let mut tree = Tree::new(element.as_widget());
                element.as_widget_mut().layout(
                    &mut tree,
                    &renderer,
                    &layout::Limits::new(Size::ZERO, size),
                )
            };
            let idle = layout(model.build_capture_button());
            let tools = find_tools_button(model.build_capture_button(), &renderer, size)
                .expect("toolbox clipped out of capture lane");
            model.recording = RecordingState::Recording {
                session: 1,
                start_time: std::time::Instant::now(),
                file_path: String::new(),
                stop_sender: None,
                audio_levels: None,
            };
            let recording = layout(model.build_capture_button());
            assert_eq!(
                idle.size(),
                recording.size(),
                "changing slot action must not move the shutter"
            );
            assert_eq!(idle.children().len(), recording.children().len());
            for (idle, recording) in idle.children().iter().zip(recording.children()) {
                assert_eq!(idle.bounds().center(), recording.bounds().center());
            }
            assert!(find_tools_button(model.build_capture_button(), &renderer, size).is_none());
            let photo =
                find_action_button(model.build_capture_button(), &renderer, size, |message| {
                    matches!(message, Message::Capture)
                })
                .expect("photo capture during recording must remain reachable");
            assert!((photo.x - tools.x).abs() < 32.0 && (photo.y - tools.y).abs() < 32.0);
            assert!(
                find_action_button(
                    model.build_capture_button(),
                    &renderer,
                    size,
                    |message| matches!(message, Message::ToggleRecording)
                )
                .is_some()
            );
        }
    }

    fn exercise(
        mut element: Element<'_, Message>,
        renderer: &cosmic::Renderer,
    ) -> (Vec<Message>, bool, Size) {
        let mut tree = Tree::new(element.as_widget());
        let node = element.as_widget_mut().layout(
            &mut tree,
            renderer,
            &layout::Limits::new(Size::ZERO, Size::new(200.0, 200.0)),
        );
        let viewport = Rectangle::with_size(Size::new(200.0, 200.0));
        let mut messages = Vec::new();
        let mut animates = false;
        for event in [
            Event::Window(window::Event::RedrawRequested(std::time::Instant::now())),
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)),
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)),
        ] {
            let mut shell = Shell::new(&mut messages);
            element.as_widget_mut().update(
                &mut tree,
                &event,
                Layout::new(&node),
                mouse::Cursor::Available(Point::new(
                    node.size().width / 2.0,
                    node.size().height / 2.0,
                )),
                renderer,
                &mut clipboard::Null,
                &mut shell,
                &viewport,
            );
            animates |= shell.redraw_request() != window::RedrawRequest::Wait;
        }
        (messages, animates, node.size())
    }

    #[test]
    fn photo_readiness_tracks_acquisition_capacity_and_recovers_after_failure() {
        let renderer = renderer();
        for motion in [false, true] {
            let mut model = AppModel::default();
            model.config.motion_photo = motion;
            model.mode = CameraMode::Photo;
            model.capture_scale_to = 1.0;
            model.photo_btn_scale_to = 1.0;
            for blocked in 0..6 {
                match blocked {
                    0 => model.is_capturing = true,
                    1 => model.hardware_photo_id = Some(7),
                    2 => model.burst_mode.stage = crate::app::state::BurstModeStage::Processing,
                    3 => {
                        for id in 1..=8 {
                            model.photo_jobs.insert(
                                id,
                                crate::app::handlers::motion_photo::PhotoJob {
                                    live: None,
                                    still_path: None,
                                },
                            );
                        }
                    }
                    4 => model.pending_close = true,
                    5 => model.transition_state.ui_disabled = true,
                    _ => unreachable!(),
                }
                assert!(!model.can_accept_photo(), "blocker {blocked}");
                for element in [
                    model.build_capture_circle(),
                    model.build_photo_during_recording_button(),
                ] {
                    let (messages, animates, _) = exercise(element, &renderer);
                    assert_eq!(messages.is_empty(), blocked >= 2, "blocker {blocked}");
                    assert_eq!(animates, blocked == 3, "spinner only for queue capacity");
                }
                let next_id = model.next_photo_id;
                for task in [
                    model.capture_photo(),
                    model.handle_capture_button_pressed(),
                    model.handle_capture(),
                ] {
                    assert!(cosmic::iced::runtime::task::into_stream(task).is_none());
                }
                assert_eq!(model.next_photo_id, next_id);
                assert!(!model.still_capture.has_request());
                assert!(model.motion_photo_error.is_none());
                assert!(!model.quick_record.is_pressed());
                model.is_capturing = false;
                model.hardware_photo_id = None;
                model.burst_mode.stage = crate::app::state::BurstModeStage::Idle;
                model.photo_jobs.clear();
                model.pending_close = false;
                model.transition_state.ui_disabled = false;
                assert!(model.can_accept_photo());
            }
            // Hardware acquisition releases while earlier encoding remains busy.
            for id in 1..=6 {
                model.photo_jobs.insert(
                    id,
                    crate::app::handlers::motion_photo::PhotoJob {
                        live: None,
                        still_path: None,
                    },
                );
            }
            model.is_capturing = true;
            model.hardware_photo_id = Some(7);
            let _ = model.update(Message::PhotoCaptureAcquired(7));
            assert_eq!(model.photo_jobs.len(), 7);
            assert!(model.can_accept_photo());
            assert!(
                !exercise(model.build_capture_circle(), &renderer)
                    .0
                    .is_empty()
            );
            assert!(
                !exercise(model.build_photo_during_recording_button(), &renderer)
                    .0
                    .is_empty()
            );
            model.hardware_photo_id = Some(8);
            model.is_capturing = true;
            let _ = model.update(Message::PhotoCaptureAcquired(8));
            assert!(!model.can_accept_photo());
            let _ = model.update(Message::PhotoCaptureFinished(
                1,
                Err("real encoding failure".into()),
            ));
            assert_eq!(model.photo_jobs.len(), 7);
            assert_eq!(
                model.motion_photo_error.as_deref(),
                Some("real encoding failure")
            );
            assert!(
                model.can_accept_photo(),
                "diagnostics must not permanently lock the shutter"
            );
            let (messages, animates, _) =
                exercise(model.build_photo_during_recording_button(), &renderer);
            assert!(
                messages
                    .iter()
                    .any(|message| matches!(message, Message::Capture))
            );
            assert!(!animates);
        }
    }

    #[test]
    fn photo_backpressure_never_disables_recording_stop_or_timer_cancel() {
        use crate::app::state::{QuickRecordState, RecordingState};
        let renderer = renderer();
        let mut model = AppModel {
            capture_scale_to: 1.0,
            mode: CameraMode::Video,
            ..Default::default()
        };
        model.burst_mode.stage = crate::app::state::BurstModeStage::Processing;
        model.transition_state.ui_disabled = true;
        model.recording = RecordingState::Recording {
            session: 1,
            start_time: std::time::Instant::now(),
            file_path: String::new(),
            stop_sender: None,
            audio_levels: None,
        };
        assert!(
            exercise(model.build_capture_circle(), &renderer)
                .0
                .iter()
                .any(|message| matches!(message, Message::ToggleRecording))
        );
        model.mode = CameraMode::Photo;
        model.quick_record = QuickRecordState::Recording;
        let (sender, mut receiver) = tokio::sync::oneshot::channel();
        if let RecordingState::Recording { stop_sender, .. } = &mut model.recording {
            *stop_sender = Some(sender);
        }
        for message in exercise(model.build_capture_circle(), &renderer).0 {
            let _ = model.update(message);
        }
        assert!(receiver.try_recv().is_ok());
        assert!(!model.quick_record.is_recording());
        assert!(!model.recording.is_recording());
        model.photo_timer_countdown = Some(3);
        for message in exercise(model.build_capture_circle(), &renderer).0 {
            let _ = model.update(message);
        }
        assert!(model.photo_timer_countdown.is_none());
    }

    #[test]
    fn capture_diagnostics_never_add_a_modal_to_the_view() {
        fn shape(tree: &Tree) -> Vec<Vec<usize>> {
            fn visit(tree: &Tree, path: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
                result.push(path.clone());
                for (index, child) in tree.children.iter().enumerate() {
                    path.push(index);
                    visit(child, path, result);
                    path.pop();
                }
            }
            let mut result = Vec::new();
            visit(tree, &mut Vec::new(), &mut result);
            result
        }
        for motion in [false, true] {
            let mut model = AppModel::default();
            model.config.motion_photo = motion;
            let baseline = shape(&Tree::new(model.view().as_widget()));
            model.motion_photo_error = Some("real capture failure retained in diagnostics".into());
            assert_eq!(shape(&Tree::new(model.view().as_widget())), baseline);
            assert!(model.motion_photo_error.is_some());
        }
    }

    #[test]
    fn raw_acquisition_keeps_shutters_normal_until_queue_limit() {
        let renderer = renderer();
        let mut model = AppModel {
            mode: CameraMode::Photo,
            capture_scale_to: 1.0,
            photo_btn_scale_to: 1.0,
            is_capturing: true,
            hardware_photo_id: Some(8),
            ..Default::default()
        };
        for count in 0..=7 {
            if count > 0 {
                model.photo_jobs.insert(
                    count,
                    crate::app::handlers::motion_photo::PhotoJob {
                        live: None,
                        still_path: None,
                    },
                );
            }
            for element in [
                model.build_capture_circle(),
                model.build_photo_during_recording_button(),
            ] {
                let (messages, animates, _) = exercise(element, &renderer);
                assert_eq!(
                    animates,
                    count == 7,
                    "spinner only at eight accepted photos"
                );
                assert_eq!(
                    messages.is_empty(),
                    count == 7,
                    "normal shutter below limit"
                );
            }
        }
        model.photo_jobs.remove(&1);
        for element in [
            model.build_capture_circle(),
            model.build_photo_during_recording_button(),
        ] {
            let (messages, animates, _) = exercise(element, &renderer);
            assert!(!messages.is_empty());
            assert!(!animates);
        }
    }

    #[test]
    fn busy_photo_shutters_ignore_clicks_and_animate_without_resizing() {
        let renderer = renderer();
        let mut model = AppModel {
            mode: CameraMode::Photo,
            capture_scale_to: 1.0,
            photo_btn_scale_to: 1.0,
            ..Default::default()
        };
        let normal_main = exercise(model.build_capture_circle(), &renderer).2;
        let normal_secondary = exercise(model.build_photo_during_recording_button(), &renderer).2;
        model.is_capturing = true;
        model.hardware_photo_id = Some(8);
        for id in 1..=7 {
            model.photo_jobs.insert(
                id,
                crate::app::handlers::motion_photo::PhotoJob {
                    live: None,
                    still_path: None,
                },
            );
        }
        model.photo_timer_setting = PhotoTimerSetting::Sec3;
        for (element, expected_size) in [
            (model.build_capture_circle(), normal_main),
            (
                model.build_photo_during_recording_button(),
                normal_secondary,
            ),
        ] {
            let (messages, animates, size) = exercise(element, &renderer);
            assert!(messages.is_empty(), "busy shutter emitted {messages:?}");
            assert!(
                animates,
                "busy shutter must request spinner animation redraws"
            );
            assert_eq!(size, expected_size);
        }
    }
}
