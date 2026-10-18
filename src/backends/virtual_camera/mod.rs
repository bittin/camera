// SPDX-License-Identifier: GPL-3.0-only

//! Virtual camera backend for streaming filtered video to PipeWire
//!
//! This module creates a virtual camera device that other applications (like
//! video conferencing software) can use as a camera source. The video output
//! has filters applied using the shared GPU filter pipeline.
//!
//! # Architecture
//!
//! ```text
//! Camera Frames (RGBA)
//!        │
//!        ▼
//! ┌──────────────────┐
//! │ GPU Filter       │  ← Uses shared filter shaders
//! │ (RGBA → RGBA)    │    Direct RGBA texture processing
//! └──────────────────┘
//!        │
//!        ▼
//! ┌──────────────────┐
//! │ GStreamer Sink   │  ← appsrc → videoconvert → pipewiresink
//! │ (PipeWire)       │    Format negotiation handled by GStreamer
//! └──────────────────┘
//!        │
//!        ▼
//!   Video Apps (Zoom, Teams, etc.)
//! ```

mod file_source;
mod gpu_filter;
mod pipeline;

pub use file_source::{
    VideoDecoder, get_video_duration, load_image_as_frame, load_preview_frame,
    load_video_frame_at_position,
};
pub use gpu_filter::GpuFilterRenderer;
pub use pipeline::VirtualCameraPipeline;

use crate::app::FilterType;
use crate::backends::camera::types::{BackendError, BackendResult, CameraFrame, FrameData};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// Shared GPU filter renderer instance for virtual camera
/// Uses OnceLock + Mutex for lazy initialization and thread-safe access
static GPU_FILTER_RENDERER: std::sync::OnceLock<Arc<Mutex<Option<GpuFilterRenderer>>>> =
    std::sync::OnceLock::new();

/// Virtual camera manager
///
/// Manages the lifecycle of the virtual camera pipeline and handles
/// frame processing with GPU-accelerated filters (with software rendering fallback).
pub struct VirtualCameraManager {
    /// The GStreamer pipeline for virtual camera output
    pipeline: Option<VirtualCameraPipeline>,
    /// Current filter being applied
    current_filter: FilterType,
    /// Whether currently streaming
    streaming: bool,
}

impl VirtualCameraManager {
    /// Create a new virtual camera manager
    pub fn new() -> Self {
        Self {
            pipeline: None,
            current_filter: FilterType::Standard,
            streaming: false,
        }
    }

    /// Start streaming to virtual camera
    ///
    /// Creates a PipeWire virtual camera node that will be visible to other applications.
    /// Uses GPU-accelerated filtering with software rendering fallback.
    pub fn start(&mut self, width: u32, height: u32) -> BackendResult<()> {
        if self.streaming {
            return Err(BackendError::Other(
                "Virtual camera already streaming".into(),
            ));
        }

        info!(width, height, "Starting virtual camera");

        // Create and start the pipeline
        let pipeline = VirtualCameraPipeline::new(width, height)?;
        pipeline.start()?;

        self.pipeline = Some(pipeline);

        self.streaming = true;

        info!("Virtual camera started successfully");
        Ok(())
    }

    /// Stop streaming to virtual camera
    pub fn stop(&mut self) -> BackendResult<()> {
        if !self.streaming {
            return Err(BackendError::Other("Virtual camera not streaming".into()));
        }

        info!("Stopping virtual camera");

        if let Some(pipeline) = self.pipeline.take() {
            pipeline.stop()?;
        }

        self.streaming = false;
        info!("Virtual camera stopped");
        Ok(())
    }

    /// Check if currently streaming
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Set the current filter to apply
    pub fn set_filter(&mut self, filter: FilterType) {
        self.current_filter = filter;
        debug!(?filter, "Virtual camera filter changed");
    }

    /// Push a frame to the virtual camera
    ///
    /// Applies the current filter using the shared GPU filter pipeline
    /// and sends the result to the virtual camera sink.
    ///
    /// This method is synchronous and can be called from a dedicated thread.
    /// GPU initialization happens lazily on first call.
    pub fn push_frame(&mut self, frame: &CameraFrame) -> BackendResult<()> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| BackendError::Other("Virtual camera not started".into()))?;

        // For standard filter, just pass through the frame data
        if self.current_filter == FilterType::Standard {
            return self.push_passthrough_frame(pipeline, frame);
        }

        // Try to use GPU filter renderer (initialize lazily if needed)
        let cell = GPU_FILTER_RENDERER.get_or_init(|| Arc::new(Mutex::new(None)));

        // Try to lock without blocking - if locked, skip filtering this frame
        match cell.try_lock() {
            Ok(mut guard) => {
                // Initialize renderer if needed
                if guard.is_none() {
                    // Create a runtime for initialization only
                    match tokio::runtime::Handle::try_current() {
                        Ok(handle) => match handle.block_on(GpuFilterRenderer::new()) {
                            Ok(renderer) => {
                                info!("GPU filter renderer initialized for virtual camera");
                                *guard = Some(renderer);
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to initialize GPU filter renderer");
                            }
                        },
                        Err(_) => {
                            // No tokio runtime, try to create one temporarily
                            match tokio::runtime::Runtime::new() {
                                Ok(rt) => match rt.block_on(GpuFilterRenderer::new()) {
                                    Ok(renderer) => {
                                        info!("GPU filter renderer initialized for virtual camera");
                                        *guard = Some(renderer);
                                    }
                                    Err(e) => {
                                        warn!(error = %e, "Failed to initialize GPU filter renderer");
                                    }
                                },
                                Err(e) => {
                                    warn!(error = %e, "Failed to create tokio runtime for GPU init");
                                }
                            }
                        }
                    }
                }

                // Apply filter if renderer is available
                if let Some(renderer) = guard.as_mut() {
                    match renderer.apply_filter(frame, self.current_filter) {
                        Ok(rgba_data) => {
                            // Pass owned Vec directly - zero-copy to GStreamer
                            return pipeline.push_frame_rgba(rgba_data, frame.width, frame.height);
                        }
                        Err(e) => {
                            warn!(error = ?e, "GPU filter failed, using passthrough");
                        }
                    }
                }
            }
            Err(_) => {
                // Mutex is locked, skip filtering this frame
                debug!("GPU renderer busy, using passthrough");
            }
        }

        // Fallback: passthrough without filter
        self.push_passthrough_frame(pipeline, frame)
    }

    /// Push a frame without filtering (passthrough)
    fn push_passthrough_frame(
        &self,
        pipeline: &VirtualCameraPipeline,
        frame: &CameraFrame,
    ) -> BackendResult<()> {
        pipeline.push_frame_rgba(Self::passthrough_data(frame), frame.width, frame.height)
    }

    /// Preserve pixel order and share tight frames; compact only padded rows.
    fn passthrough_data(frame: &CameraFrame) -> FrameData {
        let row_bytes = frame.width as usize * 4;
        let stride = frame.stride as usize;
        if stride == row_bytes {
            return frame.data.clone();
        }
        let mut data = Vec::with_capacity(row_bytes * frame.height as usize);
        for y in 0..frame.height as usize {
            let start = y * stride;
            data.extend_from_slice(&frame.data[start..start + row_bytes]);
        }
        FrameData::from_owned_vec(data)
    }
}

impl Default for VirtualCameraManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VirtualCameraManager {
    fn drop(&mut self) {
        if self.streaming
            && let Err(e) = self.stop()
        {
            error!(?e, "Failed to stop virtual camera on drop");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::PixelFormat;

    #[test]
    fn file_passthrough_preserves_asymmetric_pixels_and_tight_storage() {
        let expected = vec![1, 2, 3, 255, 4, 5, 6, 128, 7, 8, 9, 64, 10, 11, 12, 0];
        for stride in [8, 12] {
            let mut bytes = expected[..8].to_vec();
            if stride == 12 {
                bytes.extend_from_slice(&[99; 4]);
            }
            bytes.extend_from_slice(&expected[8..]);
            if stride == 12 {
                bytes.extend_from_slice(&[99; 4]);
            }
            let frame = CameraFrame {
                width: 2,
                height: 2,
                stride,
                data: FrameData::from_owned_vec(bytes),
                format: PixelFormat::RGBA,
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let output = VirtualCameraManager::passthrough_data(&frame);
            assert_eq!(output.as_ref(), expected.as_slice());
            if stride == 8 {
                assert_eq!(output.as_ptr(), frame.data.as_ptr());
            }
            drop(frame);
            assert_eq!(output.as_ref(), expected.as_slice());
        }
    }
}
