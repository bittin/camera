// SPDX-License-Identifier: GPL-3.0-only

//! User-visible video recording failures and restart guidance.

use super::{AppModel, Message, RecordingState, TimelapseState};
use crate::fl;
use crate::pipelines::video::{RecordingFailure, RecordingMonitor};
use cosmic::{Element, Task, widget};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EncoderOperation {
    Video(u64),
    MotionPhoto(u64),
    Timelapse(u64),
}

#[derive(Debug)]
pub(crate) struct RecordingFailureNotice {
    pub operation: EncoderOperation,
    pub encoder: String,
    pub detail: String,
    pub visible: bool,
}

impl AppModel {
    pub(crate) fn encoder_failure_subscriptions(&self) -> cosmic::iced::Subscription<Message> {
        cosmic::iced::Subscription::batch(self.encoder_operations.iter().map(|(&operation, monitor)| {
            let mut receiver = monitor.subscribe();
            let stream = async_stream::stream! {
                if let Some(failure) = crate::pipelines::video::wait_for_recording_failure(receiver.clone()).await {
                    yield Message::EncoderFailed { operation, failure };
                }
                // A worker result or warning does not prove native resource cleanup.
                // Keep every operation subscribed until its last resource is released.
                loop {
                    let finished = receiver.borrow_and_update().phase == crate::pipelines::video::RecordingPhase::Finished;
                    if finished { break; }
                    if receiver.changed().await.is_err() { return; }
                }
                yield Message::EncoderOperationFinished(operation);
            };
            super::subscription_with_id(("recording-encoder-monitor", operation), stream)
        }))
    }

    pub(crate) fn register_encoder_operation(
        &mut self,
        operation: EncoderOperation,
        monitor: RecordingMonitor,
    ) {
        self.encoder_operations.insert(operation, monitor);
    }

    /// Dismissing a warning or changing a preference cannot repair a wedged driver.
    pub(crate) fn encoder_restart_required(&self) -> bool {
        self.recording_failure.is_some()
            || self
                .encoder_operations
                .values()
                .any(|monitor| monitor.subscribe().borrow().error.is_some())
    }

    pub(crate) fn finish_encoder_operation(
        &mut self,
        operation: EncoderOperation,
        error: Option<String>,
    ) {
        let Some(monitor) = self.encoder_operations.get(&operation) else {
            return;
        };
        let status = monitor.subscribe().borrow().clone();
        if let Some(detail) = status.error.or(error) {
            let _ = self.handle_encoder_failure(
                operation,
                RecordingFailure {
                    encoder: status.encoder,
                    detail,
                },
            );
        }
        if status.phase != crate::pipelines::video::RecordingPhase::Finished {
            return;
        }
        self.encoder_operations.remove(&operation);
        if let EncoderOperation::Video(session) = operation {
            self.recording_jobs.remove(&session);
        }
    }

    pub(crate) fn handle_encoder_failure(
        &mut self,
        operation: EncoderOperation,
        failure: RecordingFailure,
    ) -> Task<cosmic::Action<Message>> {
        if !self.encoder_operations.contains_key(&operation) {
            return Task::none();
        }
        tracing::error!(?operation, encoder = %failure.encoder, error = %failure.detail,
            "Video encoder failed; device restart required before further recording");
        if self.recording_failure.is_some() {
            // Preserve the first cause; cleanup completion must not reopen a dismissed dialog.
            return Task::none();
        }
        self.recording_failure = Some(RecordingFailureNotice {
            operation,
            encoder: failure.encoder,
            detail: failure.detail.chars().take(2048).collect(),
            visible: true,
        });
        // Signal producers, but retain ownership of all jobs until real completion.
        self.request_recording_stop();
        self.recording = RecordingState::Idle;
        self.quick_record = super::state::QuickRecordState::Idle;
        if let Some(bridge) = &self.recording_bridge {
            bridge.set_recording_sender(None);
        }
        if let TimelapseState::Running {
            start_time,
            shots_taken,
            ..
        } = &self.timelapse
        {
            self.timelapse = TimelapseState::Finalising {
                start_time: *start_time,
                shots_taken: *shots_taken,
            };
        }
        self.sync_motion_photo_recorder();
        // Preview and accepted jobs remain alive, so retain their current
        // inhibitor. Portal acquire/Drop performs synchronous D-Bus I/O and
        // must not delay the first warning redraw; completion paths reconcile it.
        let flash_devices = self.flash.hardware.devices.clone();
        if !flash_devices.is_empty() {
            // Some callers only latch the notice and discard its returned Task.
            // Launch now, but never wait for a potentially blocked sysfs write.
            let _ = std::thread::Builder::new()
                .name("encoder-failure-flash".into())
                .spawn(move || crate::flash::all_off(&flash_devices));
        }
        Task::none()
    }

    pub(crate) fn show_encoder_restart_warning(&mut self) {
        if self.recording_failure.is_none() {
            let pending = self
                .encoder_operations
                .iter()
                .find_map(|(&operation, monitor)| {
                    let status = monitor.subscribe().borrow().clone();
                    status.error.map(|detail| {
                        (
                            operation,
                            RecordingFailure {
                                encoder: status.encoder,
                                detail,
                            },
                        )
                    })
                });
            if let Some((operation, failure)) = pending {
                let _ = self.handle_encoder_failure(operation, failure);
            }
        }
        if let Some(failure) = &mut self.recording_failure {
            failure.visible = true;
        }
    }

    pub(crate) fn dismiss_encoder_failure(&mut self) -> Task<cosmic::Action<Message>> {
        if let Some(failure) = &mut self.recording_failure {
            failure.visible = false;
        }
        Task::none()
    }

    pub(crate) fn recording_failure_body(&self) -> Option<String> {
        self.recording_failure.as_ref().map(|failure| {
            fl!(
                "recording-error-body",
                encoder = failure.encoder.clone(),
                error = failure.detail.clone()
            )
        })
    }

    /// libcosmic skips its template (and Application::dialog) on non-COSMIC
    /// compositors. Keep the native popup in that full-bleed window path too.
    pub(crate) fn with_recording_failure_dialog<'a>(
        &'a self,
        content: Element<'a, Message>,
    ) -> Element<'a, Message> {
        if self.core.window.use_template {
            return content;
        }
        let mut popover = widget::popover(content).modal(true);
        if let Some(dialog) = self.recording_failure_dialog() {
            popover = popover.popup(dialog);
        }
        popover.into()
    }

    pub(crate) fn recording_failure_dialog(&self) -> Option<Element<'_, Message>> {
        let failure = self
            .recording_failure
            .as_ref()
            .filter(|failure| failure.visible)?;
        tracing::trace!(operation = ?failure.operation, "Presenting recording failure");
        let spacing = cosmic::theme::active().cosmic().spacing;
        // Native Dialog lays out shrink content before its action row. Bound the
        // entire explanation (including its title) so even a 180px window keeps
        // the dismiss button available, and long driver identifiers can wrap.
        let content_height = (self.screen_height
            - 16.0
            - 2.0 * f32::from(spacing.space_m)
            - f32::from(spacing.space_l)
            - 64.0)
            .clamp(1.0, 300.0);
        let explanation = widget::column::with_children([
            widget::text::title3(fl!("recording-error-title")).into(),
            widget::text::body(self.recording_failure_body().unwrap_or_default())
                .wrapping(cosmic::iced::widget::text::Wrapping::WordOrGlyph)
                .into(),
        ])
        .spacing(spacing.space_xxs);
        Some(
            widget::dialog()
                .control(widget::scrollable(explanation).height(content_height))
                .width(cosmic::iced::Length::Fill)
                .max_width((self.screen_width - 16.0).clamp(1.0, 570.0))
                .max_height((self.screen_height - 16.0).max(100.0))
                .primary_action(
                    widget::button::suggested(fl!("recording-error-dismiss"))
                        .on_press(Message::DismissEncoderFailure),
                )
                .into(),
        )
    }
}

#[cfg(test)]
#[path = "recording_error_presentation_tests.rs"]
mod presentation_tests;

#[cfg(test)]
mod tests {
    use crate::app::{AppModel, Message};
    use cosmic::Application;

    #[tokio::test]
    async fn recording_stop_retains_jobs_and_observer_until_native_resource_release() {
        use super::*;
        use crate::pipelines::video::RecordingPhase;
        use futures::StreamExt;
        use std::time::{Duration, Instant};

        for result in [Ok("video.mp4".into()), Err("worker failure".into())] {
            let mut model = AppModel::default();
            let operation = EncoderOperation::Video(7);
            let monitor = RecordingMonitor::new("actual-encoder".into());
            let lease = monitor.resource_lease();
            monitor.set_phase(RecordingPhase::Cleanup);
            monitor.finish(); // Owner returned, but the pusher still owns native resources.
            assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Cleanup);
            model.register_encoder_operation(operation, monitor.clone());
            model.recording_jobs.insert(7);
            model.photo_jobs.insert(
                8,
                crate::app::handlers::motion_photo::PhotoJob {
                    live: None,
                    still_path: None,
                },
            );
            let started = Instant::now();
            let _ = model.update(Message::RecordingStopped { session: 7, result });
            assert!(started.elapsed() < Duration::from_secs(1));
            assert!(
                model.encoder_operations.contains_key(&operation),
                "worker result is not native cleanup"
            );
            assert!(model.recording_jobs.contains(&7));
            let _ = model.update(Message::ToggleRecording);
            assert!(!model.recording.is_recording());
            assert_eq!(model.recording_jobs.len(), 1);

            // Build the actual subscription AFTER the worker result was delivered.
            let mut recipes =
                iced_futures::subscription::into_recipes(model.encoder_failure_subscriptions());
            let mut stream = recipes
                .pop()
                .unwrap()
                .stream(Box::pin(futures::stream::pending()));
            monitor.report_error("cleanup failure after worker result".into());
            let failure = tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .unwrap();
            assert!(
                matches!(failure, Message::EncoderFailed { operation: actual, .. } if actual == operation)
            );
            let _ = model.update(failure);
            assert!(model.recording_failure.is_some());
            let _ = model.update(Message::DismissEncoderFailure);
            assert!(model.encoder_restart_required());
            assert!(model.photo_jobs.contains_key(&8));
            assert!(
                tokio::time::timeout(Duration::from_millis(10), stream.next())
                    .await
                    .is_err()
            );
            // A premature/stale completion must not retire live native resources.
            let _ = model.update(Message::EncoderOperationFinished(operation));
            assert!(model.encoder_operations.contains_key(&operation));
            assert!(model.recording_jobs.contains(&7));
            model.photo_jobs.clear();
            model.pending_close = true;
            assert!(model.shutdown_has_pending_media());
            assert!(!model.shutdown_waiting_for_recorder);

            drop(lease); // Real resource-drop notification publishes Finished.
            let finished = tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .unwrap();
            assert!(
                matches!(finished, Message::EncoderOperationFinished(actual) if actual == operation)
            );
            let _ = model.update(finished);
            assert!(!model.encoder_operations.contains_key(&operation));
            assert!(model.recording_jobs.is_empty());
            assert!(!model.shutdown_has_pending_media());
            assert!(model.shutdown_waiting_for_recorder);
            assert!(model.encoder_restart_required());
            assert!(
                model.dialog().is_none(),
                "completion must not reopen the dismissed warning"
            );
        }
    }

    // Post-GREEN characterization: use the production cleanup deadline, no test clock.
    #[tokio::test]
    async fn cleanup_deadline_survives_delivered_video_result() {
        use super::*;
        use crate::pipelines::video::RecordingPhase;
        use futures::StreamExt;
        use std::time::Duration;

        let mut model = AppModel::default();
        let operation = EncoderOperation::Video(7);
        let monitor = RecordingMonitor::new("actual-encoder".into());
        let lease = monitor.resource_lease();
        monitor.set_phase(RecordingPhase::Cleanup);
        monitor.finish();
        model.register_encoder_operation(operation, monitor.clone());
        model.recording_jobs.insert(7);
        let _ = model.update(Message::RecordingStopped {
            session: 7,
            result: Ok("video.mp4".into()),
        });
        let mut recipes =
            iced_futures::subscription::into_recipes(model.encoder_failure_subscriptions());
        let mut stream = recipes
            .pop()
            .unwrap()
            .stream(Box::pin(futures::stream::pending()));
        let failure = tokio::time::timeout(Duration::from_secs(20), stream.next())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(&failure, Message::EncoderFailed { operation: actual, failure }
            if *actual == operation && failure.detail.contains("Cleanup timed out"))
        );
        let _ = model.update(failure);
        assert!(model.encoder_restart_required());
        assert!(model.recording_jobs.contains(&7));
        assert!(model.dialog().is_some());
        drop(lease);
        let finished = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap();
        let _ = model.update(finished);
        assert!(!model.encoder_operations.contains_key(&operation));
        assert!(model.recording_jobs.is_empty());
        assert!(model.encoder_restart_required());
    }

    #[tokio::test]
    async fn successful_monitor_completion_retires_every_encoder_path_without_warning() {
        use super::*;
        use futures::StreamExt;
        for operation in [
            EncoderOperation::Video(1),
            EncoderOperation::MotionPhoto(2),
            EncoderOperation::Timelapse(3),
        ] {
            let mut model = AppModel::default();
            let monitor = RecordingMonitor::new("healthy-encoder".into());
            model.register_encoder_operation(operation, monitor.clone());
            if let EncoderOperation::Video(session) = operation {
                model.recording_jobs.insert(session);
            }
            let mut recipes =
                iced_futures::subscription::into_recipes(model.encoder_failure_subscriptions());
            let mut stream = recipes
                .pop()
                .unwrap()
                .stream(Box::pin(futures::stream::pending()));
            monitor.finish();
            let finished = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .unwrap();
            assert!(
                matches!(finished, Message::EncoderOperationFinished(actual) if actual == operation)
            );
            let _ = model.update(finished);
            assert!(!model.encoder_operations.contains_key(&operation));
            assert!(model.recording_jobs.is_empty());
            assert!(!model.encoder_restart_required());
        }
    }

    #[test]
    fn recording_failure_opens_restart_dialog() {
        let mut model = AppModel::default();
        model.recording_jobs.insert(7);
        let _ = model.update(Message::RecordingStopped {
            session: 7,
            result: Err("Synthetic encoder failure".into()),
        });
        assert!(
            model.dialog().is_some(),
            "recording errors must be shown, not only logged"
        );
    }

    #[tokio::test]
    async fn dismissed_failure_blocks_all_video_entry_points_and_reopens_warning() {
        use super::*;
        for message in [
            Message::ToggleRecording,
            Message::StartRecordingAfterDelay,
            Message::QuickRecordThreshold,
            Message::ToggleTimelapse,
        ] {
            let mut model = AppModel::default();
            let operation = EncoderOperation::Video(7);
            model
                .register_encoder_operation(operation, RecordingMonitor::new("v4l2h265enc".into()));
            let _ = model.handle_encoder_failure(
                operation,
                RecordingFailure {
                    encoder: "v4l2h265enc".into(),
                    detail: "Synthetic driver failure".into(),
                },
            );
            let _ = model.update(Message::DismissEncoderFailure);
            assert!(model.dialog().is_none());
            assert!(model.encoder_restart_required());
            let _ = model.update(message);
            assert!(
                model.dialog().is_some(),
                "a new video attempt must show restart guidance"
            );
            assert!(!model.recording.is_recording());
            assert!(!model.timelapse.is_running());
            assert!(model.recording_jobs.is_empty());
        }
    }

    #[tokio::test]
    async fn every_encoder_path_reports_without_waiting_for_cleanup_or_losing_stills() {
        use super::*;
        use futures::StreamExt;
        for operation in [
            EncoderOperation::Video(1),
            EncoderOperation::MotionPhoto(2),
            EncoderOperation::Timelapse(3),
        ] {
            let mut model = AppModel {
                last_media_path: Some("completed-still.jpg".into()),
                ..Default::default()
            };
            model.photo_jobs.insert(
                8,
                crate::app::handlers::motion_photo::PhotoJob {
                    live: None,
                    still_path: None,
                },
            );
            let monitor = RecordingMonitor::new("selected-encoder".into());
            monitor.set_encoder("actual-encoder".into());
            monitor.set_phase(crate::pipelines::video::RecordingPhase::Cleanup);
            model.register_encoder_operation(operation, monitor.clone());
            let mut recipes =
                iced_futures::subscription::into_recipes(model.encoder_failure_subscriptions());
            let mut stream = recipes
                .pop()
                .unwrap()
                .stream(Box::pin(futures::stream::pending()));
            monitor.report_error("Synthetic firmware failure".into());
            let message = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .unwrap();
            assert!(
                matches!(message.clone(), Message::EncoderFailed { operation: actual, failure }
                if actual == operation && failure.encoder == "actual-encoder")
            );
            let _ = model.update(message);
            assert!(model.dialog().is_some());
            assert!(
                model.encoder_operations.contains_key(&operation),
                "warning does not complete a worker"
            );
            assert!(model.photo_jobs.contains_key(&8));
            assert_eq!(
                model.last_media_path.as_deref(),
                Some("completed-still.jpg")
            );
            let body = model.recording_failure_body().unwrap();
            for required in [
                "actual-encoder",
                "Restart your device",
                "Settings > Video",
                "Linux kernel",
                "device maintainers",
                "Synthetic firmware failure",
            ] {
                assert!(
                    body.contains(required),
                    "missing guidance {required}: {body}"
                );
            }
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(10), stream.next())
                    .await
                    .is_err()
            );
            monitor.finish();
            let finished = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
                .await
                .unwrap()
                .unwrap();
            let _ = model.update(finished);
            assert!(!model.encoder_operations.contains_key(&operation));
            assert!(model.encoder_restart_required());
        }
    }

    #[test]
    fn stale_failures_and_late_completion_cannot_replace_the_first_warning() {
        use super::*;
        let mut model = AppModel::default();
        let operation = EncoderOperation::Video(7);
        model.register_encoder_operation(operation, RecordingMonitor::new("actual-encoder".into()));
        model.finish_encoder_operation(operation, Some("first cause".into()));
        let _ = model.update(Message::DismissEncoderFailure);
        let _ = model.update(Message::EncoderFailed {
            operation,
            failure: RecordingFailure {
                encoder: "stale-encoder".into(),
                detail: "stale cause".into(),
            },
        });
        assert!(model.dialog().is_none());
        let notice = model.recording_failure.as_ref().unwrap();
        assert_eq!(notice.encoder, "actual-encoder");
        assert_eq!(notice.detail, "first cause");
        assert!(model.encoder_restart_required());
    }

    #[test]
    fn restart_dialog_keeps_its_dismiss_action_inside_short_and_rotated_viewports() {
        use super::*;
        use cosmic::iced::advanced::{Layout, layout, renderer::Headless, widget::Tree};
        use cosmic::iced::{Font, Pixels, Size};
        let renderer = pollster::block_on(cosmic::Renderer::new(
            Font::DEFAULT,
            Pixels(14.0),
            Some("tiny-skia"),
        ))
        .unwrap();
        for size in [
            Size::new(320.0, 180.0),
            Size::new(720.0, 180.0),
            Size::new(360.0, 720.0),
        ] {
            let mut model = AppModel {
                screen_width: size.width,
                screen_height: size.height,
                ..Default::default()
            };
            let operation = EncoderOperation::Video(1);
            model
                .register_encoder_operation(operation, RecordingMonitor::new("v4l2h264enc".into()));
            model.finish_encoder_operation(
                operation,
                Some("Synthetic firmware failure: ".to_owned() + &"x".repeat(2048)),
            );
            let mut element = model.recording_failure_dialog().unwrap();
            let mut tree = Tree::new(element.as_widget());
            let node = element.as_widget_mut().layout(
                &mut tree,
                &renderer,
                &layout::Limits::new(Size::ZERO, size),
            );
            let layout = Layout::new(&node);
            let column = layout.children().next().unwrap();
            let actions = column.children().nth(1).unwrap().bounds();
            assert!(
                actions.height >= 24.0,
                "dismiss action squeezed out at {size:?}: {actions:?}"
            );
            assert!(
                actions.y + actions.height <= node.size().height + 0.1,
                "dismiss action outside warning at {size:?}: {actions:?} vs {:?}",
                node.size()
            );
            assert!(node.size().height <= size.height && node.size().width <= size.width);
        }
    }

    #[test]
    fn timelapse_does_not_start_an_encoder_deadline_before_the_first_frame() {
        let mut model = AppModel::default();
        let _ = model.update(Message::ToggleTimelapse);
        assert!(model.timelapse.is_running());
        let monitor = model.encoder_operations.values().next().unwrap();
        assert_eq!(
            monitor.subscribe().borrow().phase,
            crate::pipelines::video::RecordingPhase::WaitingForFrames
        );
    }
}
