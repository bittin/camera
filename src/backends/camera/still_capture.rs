// SPDX-License-Identifier: GPL-3.0-only
//! One bounded, generation-owned still request/result shared by all capture routes.
use super::types::CameraFrame;
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StillTicket(u64);

#[derive(Default)]
struct State {
    generation: u64,
    active: Option<StillTicket>,
    frame: Option<CameraFrame>,
}

#[derive(Default)]
pub(crate) struct StillCapture {
    state: Mutex<State>,
    notify: Notify,
}

/// Consumer ownership retires this exact generation, including on task cancellation.
pub(crate) struct StillRequest {
    capture: Arc<StillCapture>,
    ticket: StillTicket,
}

impl StillCapture {
    /// Atomically supersede the old request AND any unconsumed completed frame.
    pub(crate) fn request(self: &Arc<Self>) -> StillRequest {
        let mut state = self.state.lock().unwrap();
        state.generation = state
            .generation
            .checked_add(1)
            .expect("still generation exhausted");
        let ticket = StillTicket(state.generation);
        state.active = Some(ticket);
        state.frame = None;
        drop(state);
        self.notify.notify_waiters();
        StillRequest {
            capture: Arc::clone(self),
            ticket,
        }
    }

    /// Snapshot before processing/copying/async decoding, never at publication time.
    pub(crate) fn pending(&self) -> Option<StillTicket> {
        let state = self.state.lock().unwrap();
        state.active.filter(|_| state.frame.is_none())
    }

    /// A retired producer cannot publish bytes or clear a newer request.
    pub(crate) fn publish(&self, ticket: StillTicket, frame: CameraFrame) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.active != Some(ticket) || state.frame.is_some() {
            return false;
        }
        state.frame = Some(frame);
        drop(state);
        self.notify.notify_waiters();
        true
    }

    /// Includes completed-but-not-yet-acquired frames for shutdown draining.
    pub(crate) fn has_request(&self) -> bool {
        self.state.lock().unwrap().active.is_some()
    }

    pub(crate) fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.active = None;
        state.frame = None;
        drop(state);
        self.notify.notify_waiters();
    }
}

impl StillRequest {
    pub(crate) fn take(&self) -> Option<CameraFrame> {
        let mut state = self.capture.state.lock().unwrap();
        if state.active != Some(self.ticket) {
            return None;
        }
        let frame = state.frame.take()?;
        state.active = None;
        Some(frame)
    }

    pub(crate) async fn wait(self, timeout: std::time::Duration) -> Option<CameraFrame> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.capture.notify.notified();
            tokio::pin!(notified);
            // Register before checking under the mutex, closing check/wait races.
            notified.as_mut().enable();
            if let Some(frame) = self.take() {
                return Some(frame);
            }
            if self.capture.state.lock().unwrap().active != Some(self.ticket) {
                return None;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return None;
            }
        }
        // Drop retires the request before the caller can unlock the shutter.
    }
}

impl Drop for StillRequest {
    fn drop(&mut self) {
        let mut state = self.capture.state.lock().unwrap();
        if state.active == Some(self.ticket) {
            state.active = None;
            state.frame = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use std::time::{Duration, Instant};

    fn frame(value: u8) -> CameraFrame {
        CameraFrame {
            width: 1,
            height: 1,
            stride: 4,
            format: super::super::types::PixelFormat::RGBA,
            data: super::super::types::FrameData::from_owned_vec(vec![value; 4]),
            captured_at: Instant::now(),
            yuv_planes: None,
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        }
    }

    #[tokio::test]
    async fn cancelled_producer_cannot_clear_or_complete_new_generation() {
        let capture = Arc::new(StillCapture::default());
        let a = capture.request();
        let producer_capture = capture.clone();
        let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
        let producer = tokio::spawn(async move {
            let ticket_a = producer_capture.pending().unwrap();
            observed_tx.send(()).unwrap();
            finish_rx.await.unwrap();
            assert!(!producer_capture.publish(ticket_a, frame(10)));
        });
        observed_rx.await.unwrap();
        drop(a);
        let b = capture.request();
        let ticket_b = capture.pending().unwrap();
        finish_tx.send(()).unwrap();
        producer.await.unwrap();
        assert!(b.take().is_none());
        assert_eq!(capture.pending(), Some(ticket_b));
        assert!(capture.publish(ticket_b, frame(20)));
        assert!(
            !capture.publish(ticket_b, frame(30)),
            "one result per request"
        );
        assert_eq!(b.take().unwrap().data.as_ref(), &[20; 4]);
        assert!(!capture.has_request());
    }

    #[test]
    fn issue_and_reset_retire_completed_slots_and_old_consumer_drops() {
        let capture = Arc::new(StillCapture::default());
        let a = capture.request();
        let ticket_a = capture.pending().unwrap();
        assert!(capture.publish(ticket_a, frame(10)));
        assert!(
            capture.has_request(),
            "completed frame stays owned until acquired"
        );
        let b = capture.request();
        let ticket_b = capture.pending().unwrap();
        assert!(a.take().is_none());
        drop(a);
        assert!(
            b.take().is_none(),
            "issue discards old completed bytes atomically"
        );
        assert_eq!(capture.pending(), Some(ticket_b));
        capture.reset();
        let c = capture.request();
        let ticket_c = capture.pending().unwrap();
        drop(b);
        assert!(!capture.publish(ticket_b, frame(20)));
        assert!(capture.publish(ticket_c, frame(30)));
        assert_eq!(c.take().unwrap().data.as_ref(), &[30; 4]);
    }

    #[tokio::test]
    async fn wait_handles_both_notification_orders_and_cancellation() {
        let capture = Arc::new(StillCapture::default());
        let a = capture.request();
        assert!(capture.publish(capture.pending().unwrap(), frame(10)));
        assert_eq!(
            a.wait(Duration::from_secs(1)).await.unwrap().data.as_ref(),
            &[10; 4]
        );
        let b = capture.request();
        let ticket_b = capture.pending().unwrap();
        let mut waiting = Box::pin(b.wait(Duration::from_secs(1)));
        assert!(waiting.as_mut().now_or_never().is_none());
        assert!(capture.publish(ticket_b, frame(20)));
        assert_eq!(waiting.await.unwrap().data.as_ref(), &[20; 4]);
        let c = capture.request();
        let ticket_c = capture.pending().unwrap();
        let mut waiting = Box::pin(c.wait(Duration::from_secs(1)));
        assert!(waiting.as_mut().now_or_never().is_none());
        drop(waiting);
        assert!(!capture.has_request());
        assert!(!capture.publish(ticket_c, frame(30)));
    }
}
