// SPDX-License-Identifier: GPL-3.0-only
//! One capture entry point for GUI, backend lifecycle, CLI and terminal.
use super::{
    libcamera::{NativeLibcameraPipeline, PipelineSharedState},
    pipewire::PipeWirePipeline,
    types::*,
};

pub(crate) enum CapturePipeline {
    Native(NativeLibcameraPipeline),
    PipeWire(PipeWirePipeline),
}
impl CapturePipeline {
    pub(crate) fn new(
        path: &str,
        format: &CameraFormat,
        multistream: bool,
        shared: PipelineSharedState,
    ) -> BackendResult<Self> {
        // Shared GUI still storage survives subscription replacement. Clear it
        // for both routes so a PipeWire frame/request cannot cross into a native
        // camera (including failed initialization), or vice versa.
        if let Ok(mut frame) = shared.still_frame.lock() {
            *frame = None;
        }
        shared
            .still_requested
            .store(false, std::sync::atomic::Ordering::Release);
        if let Some(serial) = path.strip_prefix("pipewire:") {
            tracing::info!(
                backend = "pipewire-v4l2",
                target = serial,
                "Starting camera capture"
            );
            PipeWirePipeline::new(serial, format, shared).map(Self::PipeWire)
        } else {
            tracing::info!(
                backend = "libcamera",
                target = path,
                "Starting camera capture"
            );
            NativeLibcameraPipeline::new(path, format, multistream, shared).map(Self::Native)
        }
    }
    pub(crate) fn request_still_capture(&self) {
        match self {
            Self::Native(p) => p.request_still_capture(),
            Self::PipeWire(p) => p.request_still_capture(),
        }
    }
    pub(crate) fn get_still_frame(&self) -> Option<CameraFrame> {
        match self {
            Self::Native(p) => p.get_still_frame(),
            Self::PipeWire(p) => p.get_still_frame(),
        }
    }
    pub(crate) fn get_preview_frame(&self) -> Option<CameraFrame> {
        match self {
            Self::Native(p) => p.get_preview_frame(),
            Self::PipeWire(p) => p.get_preview_frame(),
        }
    }
    pub(crate) fn stop(&self) -> BackendResult<()> {
        match self {
            Self::Native(p) => p.stop(),
            Self::PipeWire(p) => p.stop(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    #[test]
    fn switching_to_native_clears_fallback_still_even_if_initialization_fails() {
        let (tx, _rx) = futures::channel::mpsc::channel(1);
        let frame = CameraFrame {
            width: 1,
            height: 1,
            data: FrameData::from_owned_vec(vec![0; 4]),
            format: PixelFormat::RGBA,
            stride: 4,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        };
        let still = Arc::new(Mutex::new(Some(frame)));
        let requested = Arc::new(AtomicBool::new(true));
        let shared = PipelineSharedState {
            frame_sender: tx,
            still_requested: requested.clone(),
            still_frame: still.clone(),
            still_frame_notify: Arc::new(tokio::sync::Notify::new()),
            recording_sender: Arc::new(Mutex::new(None)),
            jpeg_recording_mode: Arc::new(AtomicBool::new(false)),
            cancel_flag: Arc::new(AtomicBool::new(false)),
        };
        assert!(
            CapturePipeline::new(
                "nonexistent-native-camera-for-switch-regression",
                &CameraFormat {
                    width: 1,
                    height: 1,
                    framerate: None,
                    pixel_format: "RGBA".into(),
                    hardware_accelerated: false
                },
                false,
                shared
            )
            .is_err()
        );
        assert!(
            still.lock().unwrap().is_none(),
            "a fallback still must not cross into the native camera lifecycle"
        );
        assert!(!requested.load(Ordering::Acquire));
    }
}
