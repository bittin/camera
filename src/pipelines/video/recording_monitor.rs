// SPDX-License-Identifier: GPL-3.0-only

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use tokio::sync::watch;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingPhase {
    WaitingForFrames,
    Starting,
    Recording,
    Finalizing,
    Cleanup,
    Finished,
}

#[derive(Clone, Debug)]
pub struct RecordingStatus {
    pub encoder: String,
    pub phase: RecordingPhase,
    pub phase_started: Instant,
    pub error: Option<String>,
    activity: Arc<ActivityWatchdog>,
}

/// Independent of the recorder's blocking worker, including during kernel hangs.
#[derive(Clone, Debug)]
pub struct RecordingMonitor {
    sender: watch::Sender<RecordingStatus>,
    activity: Arc<ActivityWatchdog>,
    completion: Arc<std::sync::Mutex<ResourceCompletion>>,
}

#[derive(Debug, Default)]
struct ResourceCompletion {
    outstanding: usize,
    finish_requested: bool,
}

/// Drop after independently owned native resources, on their owning worker.
#[derive(Debug)]
pub(crate) struct RecordingResourceLease(RecordingMonitor);

impl Drop for RecordingResourceLease {
    fn drop(&mut self) {
        let mut completion = self.0.completion.lock().unwrap();
        completion.outstanding -= 1;
        if completion.finish_requested && completion.outstanding == 0 {
            self.0.set_phase(RecordingPhase::Finished);
        }
    }
}

/// Declare last in an owning pipeline struct: Rust drops fields in declaration
/// order, after its Drop implementation. NULL returning is not final unref.
/// Completion is deferred while independent native-resource leases remain.
#[derive(Debug)]
pub(crate) struct FinishRecording(pub RecordingMonitor);

impl Drop for FinishRecording {
    fn drop(&mut self) {
        self.0.finish();
    }
}

impl RecordingMonitor {
    pub fn new(encoder: String) -> Self {
        Self::with_phase(encoder, RecordingPhase::Starting)
    }

    pub fn new_waiting(encoder: String) -> Self {
        Self::with_phase(encoder, RecordingPhase::WaitingForFrames)
    }

    fn with_phase(encoder: String, phase: RecordingPhase) -> Self {
        let activity = Arc::new(ActivityWatchdog::default());
        let (sender, _) = watch::channel(RecordingStatus {
            encoder,
            phase,
            phase_started: Instant::now(),
            error: None,
            activity: activity.clone(),
        });
        Self {
            sender,
            activity,
            completion: Arc::default(),
        }
    }

    /// Opt in only for encoders expected to emit promptly (not sparse timelapse).
    pub fn enable_activity_watchdog(&self) {
        self.activity.enabled.store(true, Ordering::Relaxed);
    }

    /// Record offered frames, including frames rejected by a full appsrc queue.
    pub fn note_input(&self) {
        self.activity.note_input_at(self.activity.now());
    }

    /// An encoded buffer is progress even when live queues dropped input frames.
    pub fn note_output(&self) {
        self.activity.pending_since.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn has_pending_input(&self) -> bool {
        self.activity.pending_since.load(Ordering::Relaxed) != 0
    }

    pub fn subscribe(&self) -> watch::Receiver<RecordingStatus> {
        self.sender.subscribe()
    }

    pub fn set_phase(&self, phase: RecordingPhase) {
        self.sender.send_if_modified(|status| {
            // Late state notifications / pusher EOS must not undo teardown.
            if status.phase as u8 >= phase as u8 {
                return false;
            }
            status.phase = phase;
            status.phase_started = Instant::now();
            true
        });
    }

    pub fn set_encoder(&self, encoder: String) {
        self.sender.send_modify(|status| status.encoder = encoder);
    }

    pub fn report_error(&self, detail: String) {
        self.sender.send_if_modified(|status| {
            if status.error.is_some() {
                return false;
            }
            status.error = Some(detail.chars().take(2048).collect());
            true
        });
    }

    /// Register before handing native resources to an independent worker.
    pub(crate) fn resource_lease(&self) -> RecordingResourceLease {
        let mut completion = self.completion.lock().unwrap();
        assert!(
            !completion.finish_requested,
            "resources registered after finish"
        );
        completion.outstanding += 1;
        RecordingResourceLease(self.clone())
    }

    /// Owner teardown is complete; independently owned resources may remain.
    pub fn finish(&self) {
        let mut completion = self.completion.lock().unwrap();
        completion.finish_requested = true;
        if completion.outstanding == 0 {
            self.set_phase(RecordingPhase::Finished);
        }
    }
}

/// Per-frame metadata never publishes a watch update or calls into GStreamer.
/// A single atomic makes input/output races linearizable. An encoded buffer
/// acknowledges progress, not a one-to-one input count (live queues drop frames).
#[derive(Debug)]
struct ActivityWatchdog {
    epoch: Instant,
    enabled: AtomicBool,
    pending_since: AtomicU64,
}

impl Default for ActivityWatchdog {
    fn default() -> Self {
        Self {
            epoch: Instant::now(),
            enabled: AtomicBool::new(false),
            pending_since: AtomicU64::new(0),
        }
    }
}

impl ActivityWatchdog {
    fn now(&self) -> u64 {
        // Zero is reserved for no pending input.
        (self
            .epoch
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX - 1)) as u64)
            + 1
    }

    fn note_input_at(&self, now: u64) {
        if self.enabled.load(Ordering::Relaxed) {
            // Continued input/backpressure cannot extend the no-progress window.
            let _ =
                self.pending_since
                    .compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
        }
    }

    fn stalled_at(&self, now: u64, timeout: Duration) -> bool {
        let pending = self.pending_since.load(Ordering::Relaxed);
        self.enabled.load(Ordering::Relaxed)
            && pending != 0
            && u128::from(now.saturating_sub(pending)) >= timeout.as_nanos()
    }
}

#[derive(Clone, Debug)]
pub struct RecordingFailure {
    pub encoder: String,
    pub detail: String,
}

/// Observe failure without waiting on (or trying to cancel) a blocked driver.
pub async fn wait_for_recording_failure(
    receiver: watch::Receiver<RecordingStatus>,
) -> Option<RecordingFailure> {
    use std::time::Duration;
    wait_with_deadlines(
        receiver,
        [
            Duration::from_secs(30),
            Duration::from_secs(75),
            Duration::from_secs(15),
        ],
    )
    .await
}

async fn wait_with_deadlines(
    receiver: watch::Receiver<RecordingStatus>,
    deadlines: [Duration; 3],
) -> Option<RecordingFailure> {
    wait_with_activity_deadlines(
        receiver,
        deadlines,
        Duration::from_secs(30),
        Duration::from_millis(250),
    )
    .await
}

async fn wait_with_activity_deadlines(
    mut receiver: watch::Receiver<RecordingStatus>,
    deadlines: [Duration; 3],
    activity_timeout: Duration,
    activity_interval: Duration,
) -> Option<RecordingFailure> {
    loop {
        let status = receiver.borrow_and_update().clone();
        let failure = |detail| {
            Some(RecordingFailure {
                encoder: status.encoder.clone(),
                detail,
            })
        };
        if let Some(detail) = status.error {
            return failure(detail);
        }
        let timeout = match status.phase {
            RecordingPhase::Starting => Some(deadlines[0]),
            RecordingPhase::Finalizing => Some(deadlines[1]),
            RecordingPhase::Cleanup => Some(deadlines[2]),
            RecordingPhase::WaitingForFrames | RecordingPhase::Recording => None,
            RecordingPhase::Finished => return None,
        };
        if let Some(timeout) = timeout {
            let deadline = tokio::time::Instant::from_std(status.phase_started + timeout);
            tokio::select! {
                biased;
                changed = receiver.changed() => {
                    if changed.is_err() {
                        return failure("Recording monitor closed before cleanup completed".into());
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    return failure(format!("Recording {:?} timed out after {} seconds", status.phase, timeout.as_secs()));
                }
            }
        } else if status.phase == RecordingPhase::Recording {
            if status
                .activity
                .stalled_at(status.activity.now(), activity_timeout)
            {
                return failure(format!(
                    "Recording encoder produced no encoded output for {} seconds despite offered input",
                    activity_timeout.as_secs()
                ));
            }
            // Activity is atomic-only: recheck independently of watch events.
            tokio::select! {
                biased;
                changed = receiver.changed() => {
                    if changed.is_err() {
                        return failure("Recording monitor closed before cleanup completed".into());
                    }
                }
                _ = tokio::time::sleep(activity_interval) => {}
            }
        } else if receiver.changed().await.is_err() {
            return failure("Recording monitor closed before cleanup completed".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Post-GREEN characterization of independent-resource completion ordering.
    #[test]
    fn finish_waits_for_all_resource_leases_in_either_owner_drop_order() {
        for owner_first in [false, true] {
            let monitor = RecordingMonitor::new("identity".into());
            let first = monitor.resource_lease();
            let second = monitor.resource_lease();
            let owner = FinishRecording(monitor.clone());
            monitor.set_phase(RecordingPhase::Cleanup);
            if owner_first {
                drop(owner);
                monitor.finish(); // Repeated completion requests are harmless.
                drop(first);
                assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Cleanup);
                drop(second);
            } else {
                drop(first);
                drop(second);
                assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Cleanup);
                drop(owner);
            }
            assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
        }
        // Existing Motion Photo / timelapse owners do not register leases.
        let monitor = RecordingMonitor::new("legacy owner".into());
        drop(FinishRecording(monitor.clone()));
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Finished);
    }

    #[test]
    fn activity_without_inputs_never_stalls_and_sparse_activity_is_disabled() {
        let monitor = RecordingMonitor::new("sparse".into());
        let far_future = 10_000_000_000_000;
        assert!(
            !monitor
                .activity
                .stalled_at(far_future, Duration::from_secs(30))
        );
        monitor.activity.note_input_at(1);
        assert!(
            !monitor
                .activity
                .stalled_at(far_future, Duration::from_secs(30))
        );
        monitor.enable_activity_watchdog();
        assert!(
            !monitor
                .activity
                .stalled_at(far_future, Duration::from_secs(30))
        );
    }

    #[test]
    fn continued_input_does_not_extend_oldest_no_progress_deadline() {
        let monitor = RecordingMonitor::new("blocked".into());
        monitor.enable_activity_watchdog();
        monitor.activity.note_input_at(1);
        for second in 1..30 {
            monitor.activity.note_input_at(second * 1_000_000_000 + 1);
        }
        assert!(
            !monitor
                .activity
                .stalled_at(30_000_000_000, Duration::from_secs(30))
        );
        assert!(
            monitor
                .activity
                .stalled_at(30_000_000_001, Duration::from_secs(30))
        );
    }

    #[tokio::test]
    async fn phase_deadlines_override_stale_encoding_activity() {
        for phase in [
            RecordingPhase::Starting,
            RecordingPhase::Finalizing,
            RecordingPhase::Cleanup,
        ] {
            let monitor = RecordingMonitor::new("blocked".into());
            monitor.enable_activity_watchdog();
            monitor.note_input();
            monitor.set_phase(phase);
            let failure = wait_with_activity_deadlines(
                monitor.subscribe(),
                [Duration::ZERO; 3],
                Duration::ZERO,
                Duration::from_millis(1),
            )
            .await
            .unwrap();
            assert!(failure.detail.contains(&format!("{phase:?}")));
        }
        let monitor = RecordingMonitor::new_waiting("waiting".into());
        monitor.enable_activity_watchdog();
        monitor.note_input();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(5),
                wait_with_activity_deadlines(
                    monitor.subscribe(),
                    [Duration::ZERO; 3],
                    Duration::ZERO,
                    Duration::from_millis(1)
                )
            )
            .await
            .is_err()
        );
        monitor.finish();
        assert!(
            wait_with_activity_deadlines(
                monitor.subscribe(),
                [Duration::ZERO; 3],
                Duration::ZERO,
                Duration::from_millis(1)
            )
            .await
            .is_none()
        );
    }

    #[test]
    fn output_progress_rearms_activity_after_idle_without_total_duration_limit() {
        let monitor = RecordingMonitor::new("healthy".into());
        monitor.enable_activity_watchdog();
        let timeout = Duration::from_secs(30);
        let tick = |seconds: u64| seconds * 1_000_000_000 + 1;
        for seconds in [1, 10, 1000, 1_000_000] {
            monitor.activity.note_input_at(tick(seconds));
            assert!(!monitor.activity.stalled_at(tick(seconds + 29), timeout));
            monitor.note_output();
            assert!(!monitor.activity.stalled_at(tick(seconds + 100), timeout));
        }
        monitor.activity.note_input_at(tick(2_000_000));
        assert!(!monitor.activity.stalled_at(tick(2_000_029), timeout));
        assert!(monitor.activity.stalled_at(tick(2_000_030), timeout));
    }

    #[tokio::test]
    async fn activity_observer_detects_stall_without_watch_events() {
        use std::time::Duration;
        let monitor = RecordingMonitor::new("blocked encoder".into());
        monitor.enable_activity_watchdog();
        monitor.set_phase(RecordingPhase::Recording);
        let rx = monitor.subscribe();
        let waiter = tokio::spawn(wait_with_activity_deadlines(
            rx.clone(),
            [Duration::from_secs(1); 3],
            Duration::from_millis(5),
            Duration::from_millis(1),
        ));
        tokio::task::yield_now().await;
        monitor.note_input();
        assert!(
            !rx.has_changed().unwrap(),
            "frames must not wake watch subscribers"
        );
        let failure = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(failure.encoder, "blocked encoder");
        assert!(failure.detail.contains("encoded output"));
    }

    #[tokio::test]
    async fn waiting_for_frames_has_no_startup_deadline() {
        use std::time::Duration;
        let monitor = RecordingMonitor::new_waiting("lazy".into());
        assert_eq!(
            monitor.subscribe().borrow().phase,
            RecordingPhase::WaitingForFrames
        );
        assert!(
            tokio::time::timeout(
                Duration::from_millis(5),
                wait_with_deadlines(monitor.subscribe(), [Duration::ZERO; 3]),
            )
            .await
            .is_err()
        );
        monitor.set_phase(RecordingPhase::Starting);
        monitor.set_phase(RecordingPhase::WaitingForFrames);
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Starting);
        assert!(
            wait_with_deadlines(monitor.subscribe(), [Duration::ZERO; 3])
                .await
                .unwrap()
                .detail
                .contains("Starting")
        );
    }

    #[test]
    fn late_start_notifications_cannot_undo_finalization_or_cleanup() {
        let monitor = RecordingMonitor::new("encoder".into());
        monitor.set_phase(RecordingPhase::Finalizing);
        let started = monitor.subscribe().borrow().phase_started;
        monitor.set_phase(RecordingPhase::Recording);
        assert_eq!(
            monitor.subscribe().borrow().phase,
            RecordingPhase::Finalizing
        );
        assert_eq!(monitor.subscribe().borrow().phase_started, started);
        monitor.set_phase(RecordingPhase::Cleanup);
        monitor.set_phase(RecordingPhase::Finalizing);
        assert_eq!(monitor.subscribe().borrow().phase, RecordingPhase::Cleanup);
    }

    #[tokio::test]
    async fn deadlines_cover_blocking_phases_but_not_recording() {
        use std::time::Duration;
        for phase in [
            RecordingPhase::Starting,
            RecordingPhase::Finalizing,
            RecordingPhase::Cleanup,
        ] {
            let monitor = RecordingMonitor::new("encoder".into());
            monitor.set_phase(phase);
            let failure = wait_with_deadlines(monitor.subscribe(), [Duration::ZERO; 3])
                .await
                .unwrap();
            assert_eq!(failure.encoder, "encoder");
            assert!(failure.detail.contains(&format!("{phase:?}")));
        }
        let monitor = RecordingMonitor::new("encoder".into());
        monitor.set_phase(RecordingPhase::Recording);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(5),
                wait_with_deadlines(monitor.subscribe(), [Duration::ZERO; 3])
            )
            .await
            .is_err()
        );
        monitor.finish();
        assert!(
            wait_for_recording_failure(monitor.subscribe())
                .await
                .is_none()
        );
        monitor.report_error("before cleanup".into());
        assert_eq!(
            wait_for_recording_failure(monitor.subscribe())
                .await
                .unwrap()
                .detail,
            "before cleanup"
        );
    }

    #[tokio::test]
    async fn phase_change_wakes_waiter_and_preserves_original_deadline() {
        use std::time::Duration;
        let monitor = RecordingMonitor::new("encoder".into());
        monitor.set_phase(RecordingPhase::Recording);
        let waiter = tokio::spawn(wait_with_deadlines(
            monitor.subscribe(),
            [Duration::ZERO; 3],
        ));
        tokio::task::yield_now().await;
        monitor.set_phase(RecordingPhase::Cleanup);
        assert!(waiter.await.unwrap().unwrap().detail.contains("Cleanup"));
    }

    #[test]
    fn first_error_and_phase_clock_survive_updates() {
        let monitor = RecordingMonitor::new("selected".into());
        let rx = monitor.subscribe();
        let started = rx.borrow().phase_started;
        monitor.set_phase(RecordingPhase::Starting);
        monitor.set_encoder("resolved".into());
        assert_eq!(rx.borrow().phase_started, started);
        assert_eq!(rx.borrow().encoder, "resolved");
        monitor.report_error("first".into());
        monitor.report_error("second".into());
        monitor.finish();
        assert_eq!(rx.borrow().error.as_deref(), Some("first"));
        assert_eq!(rx.borrow().phase, RecordingPhase::Finished);
    }
}
