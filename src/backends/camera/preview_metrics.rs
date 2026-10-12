// SPDX-License-Identifier: GPL-3.0-only
//! Per-capture preview diagnostics; never infer zero-copy from a missing measurement.

use super::types::{CameraFrame, Framerate, PixelFormat};
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
static ACTIVE: Mutex<Weak<PreviewMetrics>> = Mutex::new(Weak::new());

#[derive(Debug, Clone)]
pub(crate) struct StreamDiagnostics {
    pub pipeline_string: String,
    pub source_format: String,
    pub framerate: Option<Framerate>,
    pub cpu_processing: Option<String>,
    pub decoder: Option<&'static str>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PreviewSnapshot {
    pub generation: u64,
    pub frame_count: u64,
    /// Application preview losses (decode, delivery backpressure, UI coalescing).
    /// Does not claim to count losses inside the sensor, driver or PipeWire server.
    pub dropped_frames: u64,
    pub copy_time_us: Option<u64>,
    pub frame_size_bytes: usize,
    pub width: u32,
    pub height: u32,
    pub output_format: Option<PixelFormat>,
    pub stream: Option<StreamDiagnostics>,
}

pub(crate) struct PreviewMetrics {
    path: String,
    state: Mutex<PreviewSnapshot>,
}

impl PreviewMetrics {
    pub(crate) fn start(path: &str) -> Arc<Self> {
        let metrics = Arc::new(Self {
            path: path.into(),
            state: Mutex::new(PreviewSnapshot {
                generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
                ..Default::default()
            }),
        });
        *ACTIVE.lock().unwrap_or_else(|e| e.into_inner()) = Arc::downgrade(&metrics);
        metrics
    }

    pub(crate) fn active_for(path: &str) -> Option<Arc<Self>> {
        ACTIVE.lock().ok()?.upgrade().filter(|m| m.path == path)
    }

    pub(crate) fn stop(&self) {
        let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        if std::ptr::eq(active.as_ptr(), self) {
            *active = Weak::new();
        }
    }

    pub(crate) fn snapshot(&self) -> PreviewSnapshot {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub(crate) fn record_copy(&self, duration: Duration) {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .copy_time_us = Some(duration.as_micros().min(u64::MAX as u128) as u64);
    }

    pub(crate) fn record_frame(&self, frame: &CameraFrame) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.frame_count += 1;
        state.frame_size_bytes = frame.data.len();
        state.width = frame.width;
        state.height = frame.height;
        state.output_format = Some(frame.format);
    }

    pub(crate) fn record_drops(&self, count: u64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.dropped_frames = state.dropped_frames.saturating_add(count);
    }

    pub(crate) fn set_stream(&self, stream: StreamDiagnostics) {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).stream = Some(stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn copy_measurement_is_unavailable_until_a_real_copy_is_recorded() {
        let metrics = PreviewMetrics::start("copy-measurement-test");
        assert_eq!(metrics.snapshot().copy_time_us, None);
        metrics.record_copy(Duration::from_micros(37));
        assert_eq!(metrics.snapshot().copy_time_us, Some(37));
        metrics.record_copy(Duration::ZERO);
        assert_eq!(metrics.snapshot().copy_time_us, Some(0));
    }
}
