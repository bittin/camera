// SPDX-License-Identifier: GPL-3.0-only

//! Shared recording state that survives camera pipeline restarts.

use super::types::RecordingFrame;
use std::sync::Arc;
use std::sync::Mutex;

/// Shared recording sender type.
///
/// This Arc lives outside individual pipelines so it survives
/// pipeline restarts and is accessible from both the subscription's capture
/// thread and the recording start/stop code.
pub type SharedRecordingSender = Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>>;

/// Shared recording channel for the active capture pipeline.
///
/// Camera lifecycle and acquisition belong to the pipeline, not this state.
#[derive(Clone, Debug, Default)]
pub struct RecordingBridge {
    /// Shared recording sender — written by recording start/stop,
    /// read by the capture thread (via Arc clone passed to the pipeline).
    recording_sender: SharedRecordingSender,
}

impl RecordingBridge {
    /// Set (or clear) the direct recording sender.
    ///
    /// This writes to a shared Arc that the capture thread reads from,
    /// independent of which pipeline instance is active.
    pub fn set_recording_sender(&self, sender: Option<tokio::sync::mpsc::Sender<RecordingFrame>>) {
        *self.recording_sender.lock().unwrap() = sender;
    }

    /// Get a clone of the shared recording sender Arc.
    ///
    /// Pass this to the pipeline so the capture thread can read from it.
    pub fn recording_sender(&self) -> SharedRecordingSender {
        Arc::clone(&self.recording_sender)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::FrameData;

    #[test]
    fn recording_state_survives_pipeline_handle_replacement() {
        use crate::backends::camera::types::{CameraFrame, PixelFormat};
        let bridge = RecordingBridge::default();
        let initial_sender = bridge.recording_sender();
        assert!(initial_sender.lock().unwrap().is_none());
        let recorder = bridge.clone();
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        recorder.set_recording_sender(Some(sender));
        assert!(initial_sender.lock().unwrap().is_some());
        drop(initial_sender);
        let replacement_sender = bridge.recording_sender();
        let frame = Arc::new(CameraFrame {
            width: 1,
            height: 1,
            stride: 4,
            format: PixelFormat::RGBA,
            data: FrameData::from_owned_vec(vec![1, 2, 3, 255]),
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: Some(123),
            libcamera_metadata: None,
        });
        replacement_sender
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .try_send(frame.clone())
            .unwrap();
        let received = receiver.try_recv().unwrap();
        assert!(Arc::ptr_eq(&frame, &received));
        recorder.set_recording_sender(None);
        assert!(replacement_sender.lock().unwrap().is_none());
        assert!(matches!(
            receiver.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
    }
}
