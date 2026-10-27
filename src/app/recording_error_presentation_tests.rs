// SPDX-License-Identifier: GPL-3.0-only

//! Exercise the actual window dispatch and overlay, not just Application::dialog.
use super::*;
use cosmic::app::cosmic::Cosmic;
use cosmic::iced::advanced::{
    Layout, Shell, clipboard, layout,
    renderer::Headless,
    widget::{Id, Operation, Tree},
};
use cosmic::iced::{Event, Font, Pixels, Rectangle, Size, Vector, mouse};

#[derive(Default)]
struct TextBounds(Vec<(String, Rectangle)>);
impl Operation for TextBounds {
    fn traverse(&mut self, visit: &mut dyn FnMut(&mut dyn Operation)) {
        visit(self);
    }
    fn text(&mut self, _: Option<&Id>, bounds: Rectangle, text: &str) {
        self.0.push((text.to_owned(), bounds));
    }
}

fn assert_window_warning(use_template: bool, direct_view_main: bool) {
    assert_window_warning_sizes(
        use_template,
        direct_view_main,
        &[
            Size::new(360.0, 720.0),
            Size::new(720.0, 360.0),
            Size::new(720.0, 180.0),
        ],
    );
}

fn assert_window_warning_sizes(use_template: bool, direct_view_main: bool, sizes: &[Size]) {
    use cosmic::ApplicationExt;
    let renderer = pollster::block_on(cosmic::Renderer::new(
        Font::DEFAULT,
        Pixels(14.0),
        Some("tiny-skia"),
    ))
    .unwrap();
    for &size in sizes {
        let mut model = AppModel {
            screen_width: size.width,
            screen_height: size.height,
            ..Default::default()
        };
        let id = cosmic::iced::window::Id::unique();
        model.core.set_main_window_id(Some(id));
        // Match init's custom chrome and non-COSMIC compositor policy.
        model.core.window.use_template = use_template;
        model.core.window.show_headerbar = false;
        model.core.window.content_container = false;
        model.core.window.context_is_overlay = true;
        let operation = EncoderOperation::Video(7);
        model.recording_jobs.insert(7);
        model.register_encoder_operation(operation, RecordingMonitor::new("v4l2h264enc".into()));
        let _ = model.update(Message::EncoderFailed {
            operation,
            failure: RecordingFailure {
                encoder: "v4l2h264enc".into(),
                detail: "Synthetic driver failure: ".to_owned() + &"x".repeat(2048),
            },
        });
        let window = Cosmic::new(model);
        let mut element = if direct_view_main {
            window.app.view_main()
        } else {
            window.view(id)
        };
        let mut tree = Tree::new(element.as_widget());
        let node = element.as_widget_mut().layout(
            &mut tree,
            &renderer,
            &layout::Limits::new(Size::ZERO, size),
        );
        let viewport = Rectangle::with_size(size);
        let mut overlay = element.as_widget_mut().overlay(&mut tree, Layout::new(&node), &renderer, &viewport, Vector::ZERO)
            .unwrap_or_else(|| panic!("encoder warning absent from production window tree: template={use_template}, view_main={direct_view_main}, size={size:?}"));
        let popup = overlay.as_overlay_mut().layout(&renderer, size);
        let mut texts = TextBounds::default();
        overlay
            .as_overlay_mut()
            .operate(Layout::new(&popup), &renderer, &mut texts);
        assert!(
            texts
                .0
                .iter()
                .any(|(text, _)| text == &fl!("recording-error-title")),
            "missing title in actual overlay: {:?}",
            texts.0
        );
        assert!(
            texts
                .0
                .iter()
                .any(|(text, _)| text.contains("v4l2h264enc")
                    && text.contains("Restart your device")),
            "missing actual encoder/restart guidance: {:?}",
            texts.0
        );
        let action = texts
            .0
            .iter()
            .find(|(text, _)| text == &fl!("recording-error-dismiss"))
            .expect("dismiss text in full-window overlay")
            .1;
        assert!(
            action.height >= 14.0
                && action.width > 0.0
                && action.x >= 0.0
                && action.y >= 0.0
                && action.x + action.width <= size.width
                && action.y + action.height <= size.height,
            "dismiss action must be visible at {size:?}: {action:?}"
        );
        let mut messages = Vec::new();
        for event in [
            mouse::Event::ButtonPressed(mouse::Button::Left),
            mouse::Event::ButtonReleased(mouse::Button::Left),
        ] {
            overlay.as_overlay_mut().update(
                &Event::Mouse(event),
                Layout::new(&popup),
                mouse::Cursor::Available(action.center()),
                &renderer,
                &mut clipboard::Null,
                &mut Shell::new(&mut messages),
            );
        }
        assert!(
            messages.iter().any(|message| matches!(
                message,
                cosmic::Action::App(Message::DismissEncoderFailure)
            )),
            "visible native dismiss action must emit its message"
        );
        drop(overlay);
        drop(element);
        let mut model = window.app;
        let _ = model.update(Message::DismissEncoderFailure);
        assert!(model.encoder_restart_required());
        assert!(model.recording_jobs.contains(&7));
        assert!(model.encoder_operations.contains_key(&operation));
        let window = Cosmic::new(model);
        let mut element = if direct_view_main {
            window.app.view_main()
        } else {
            window.view(id)
        };
        let mut tree = Tree::new(element.as_widget());
        let node = element.as_widget_mut().layout(
            &mut tree,
            &renderer,
            &layout::Limits::new(Size::ZERO, size),
        );
        assert!(
            element
                .as_widget_mut()
                .overlay(
                    &mut tree,
                    Layout::new(&node),
                    &renderer,
                    &viewport,
                    Vector::ZERO
                )
                .is_none(),
            "dismissal removes only popup, not restart quarantine"
        );
    }
}

#[test]
fn full_libcosmic_view_main_presents_encoder_warning() {
    assert_window_warning(true, true);
}

#[test]
fn production_non_cosmic_window_presents_encoder_warning() {
    assert_window_warning(false, false);
}

#[test]
fn production_non_cosmic_landscape_window_presents_encoder_warning() {
    assert_window_warning_sizes(
        false,
        false,
        &[Size::new(720.0, 360.0), Size::new(720.0, 180.0)],
    );
}

#[test]
fn encoder_failure_arms_active_video_stop_before_signalling_worker() {
    use crate::pipelines::video::RecordingPhase;
    let mut model = AppModel::default();
    let operation = EncoderOperation::Video(7);
    let monitor = RecordingMonitor::new("actual-encoder".into());
    monitor.set_phase(RecordingPhase::Recording);
    model.register_encoder_operation(operation, monitor.clone());
    model.recording_jobs.insert(7);
    let (stop, mut stopped) = tokio::sync::oneshot::channel();
    model.recording = RecordingState::start(7, "unfinished.mp4".into(), stop, None);
    let _ = model.update(Message::EncoderFailed {
        operation,
        failure: RecordingFailure {
            encoder: "actual-encoder".into(),
            detail: "driver failed".into(),
        },
    });
    assert_eq!(
        monitor.subscribe().borrow().phase,
        RecordingPhase::Finalizing,
        "failure-triggered stop must arm the GUI-side cleanup deadline too"
    );
    assert_eq!(stopped.try_recv(), Ok(()));
    assert!(model.recording_jobs.contains(&7));
    assert!(model.encoder_operations.contains_key(&operation));
}

#[test]
fn encoder_failure_returns_before_contacting_unresponsive_inhibit_bus() {
    use std::os::unix::net::UnixListener;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};
    const CHILD: &str = "CAMERA_ENCODER_FAILURE_INHIBIT_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let mut model = AppModel::default();
        let operation = EncoderOperation::Timelapse(7);
        model.register_encoder_operation(operation, RecordingMonitor::new("actual-encoder".into()));
        model.timelapse = TimelapseState::Finalising {
            start_time: Instant::now(),
            shots_taken: 1,
        };
        let _ = model.update(Message::EncoderFailed {
            operation,
            failure: RecordingFailure {
                encoder: "actual-encoder".into(),
                detail: "driver failed".into(),
            },
        });
        assert!(model.recording_failure.as_ref().unwrap().visible);
        return;
    }
    let root = std::env::temp_dir().join(format!("encoder-inhibit-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&root).unwrap();
    let socket = root.join("bus");
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "app::recording_error::presentation_tests::encoder_failure_returns_before_contacting_unresponsive_inhibit_bus", "--nocapture"])
        .env(CHILD, "1")
        .env("DBUS_SESSION_BUS_ADDRESS", format!("unix:path={}", socket.display()))
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut contacted_bus = false;
    let mut connection = None;
    let success = loop {
        if let Ok((stream, _)) = listener.accept() {
            contacted_bus = true;
            connection = Some(stream); // Intentionally withhold D-Bus authentication.
        }
        if let Some(status) = child.try_wait().unwrap() {
            break status.success();
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            break false;
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    drop(connection);
    drop(listener);
    std::fs::remove_dir_all(root).unwrap();
    assert!(
        success && !contacted_bus,
        "failure handler must return for redraw without synchronous portal I/O: success={success}, contacted_bus={contacted_bus}"
    );
}
