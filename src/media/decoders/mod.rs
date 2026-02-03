// SPDX-License-Identifier: GPL-3.0-only

//! Hardware and software decoder utilities
//!
//! This module provides utilities for detecting and managing video decoders,
//! particularly hardware-accelerated decoders for formats like MJPEG, H.264, etc.

mod definitions;
mod hardware;
mod pipeline;

pub use definitions::{DecoderDef, H264_DECODERS, H265_DECODERS, MJPEG_DECODERS};
pub use hardware::detect_hw_decoders;
pub use pipeline::{get_full_pipeline_string, try_create_pipeline};

/// Pipeline backend selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBackend {
    /// PipeWire backend (allows simultaneous preview + recording)
    PipeWire,
}

// Conversion from CameraBackendType (for backward compatibility)
impl From<crate::backends::camera::CameraBackendType> for PipelineBackend {
    fn from(_backend: crate::backends::camera::CameraBackendType) -> Self {
        // Only PipeWire is supported
        PipelineBackend::PipeWire
    }
}
