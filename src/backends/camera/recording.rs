// SPDX-License-Identifier: GPL-3.0-only

//! Shared recording state that survives camera pipeline restarts.

use super::types::RecordingFrame;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared recording sender type.
///
/// This Arc lives outside individual pipelines so it survives
/// pipeline restarts and is accessible from both the subscription's capture
/// thread and the recording start/stop code.
pub type SharedRecordingSender = Arc<Mutex<Option<tokio::sync::mpsc::Sender<RecordingFrame>>>>;

/// Shared recording channel and JPEG mode for the active capture pipeline.
///
/// Camera lifecycle and acquisition belong to the pipeline, not this state.
#[derive(Clone, Debug, Default)]
pub struct RecordingBridge {
    /// Shared recording sender — written by recording start/stop,
    /// read by the capture thread (via Arc clone passed to the pipeline).
    recording_sender: SharedRecordingSender,
    /// When true, the capture thread sends raw JPEG bytes (not decoded frames)
    /// to the recording channel for GPU-accelerated decode via VA-API.
    jpeg_recording_mode: Arc<AtomicBool>,
}

impl RecordingBridge {
    /// Set (or clear) the direct recording sender.
    ///
    /// This writes to a shared Arc that the capture thread reads from,
    /// independent of which pipeline instance is active.
    pub fn set_recording_sender(&self, sender: Option<tokio::sync::mpsc::Sender<RecordingFrame>>) {
        *self.recording_sender.lock().unwrap() = sender;
    }

    /// Enable or disable JPEG recording mode.
    ///
    /// When enabled, the capture thread sends raw JPEG bytes instead of
    /// CPU-decoded frames to the recording channel.
    pub fn set_jpeg_recording_mode(&self, enabled: bool) {
        self.jpeg_recording_mode.store(enabled, Ordering::Relaxed);
    }

    /// Get a clone of the jpeg_recording_mode flag Arc.
    pub fn jpeg_recording_mode(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.jpeg_recording_mode)
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
        let bridge = RecordingBridge::default();
        let initial_sender = bridge.recording_sender();
        let initial_mode = bridge.jpeg_recording_mode();
        assert!(initial_sender.lock().unwrap().is_none());
        assert!(!initial_mode.load(Ordering::Relaxed));

        let recorder = bridge.clone();
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        recorder.set_recording_sender(Some(sender));
        recorder.set_jpeg_recording_mode(true);
        assert!(initial_sender.lock().unwrap().is_some());
        assert!(initial_mode.load(Ordering::Relaxed));
        drop(initial_sender);
        drop(initial_mode);

        let replacement_sender = bridge.recording_sender();
        let replacement_mode = bridge.jpeg_recording_mode();
        assert!(replacement_mode.load(Ordering::Relaxed));
        assert!(
            replacement_sender
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .try_send(RecordingFrame::Jpeg {
                    data: FrameData::from_owned_vec(vec![1, 2, 3]),
                    width: 1,
                    height: 1,
                    sensor_timestamp_ns: None,
                    sequence: None,
                })
                .is_ok()
        );
        let RecordingFrame::Jpeg { data, .. } = receiver.try_recv().unwrap() else {
            panic!("recording frame changed type");
        };
        assert_eq!(&*data, &[1, 2, 3]);

        recorder.set_recording_sender(None);
        recorder.set_jpeg_recording_mode(false);
        assert!(replacement_sender.lock().unwrap().is_none());
        assert!(!replacement_mode.load(Ordering::Relaxed));
        assert!(matches!(
            receiver.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        ));
    }
}
