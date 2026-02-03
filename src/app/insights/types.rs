// SPDX-License-Identifier: GPL-3.0-only

//! Types for the Insights drawer diagnostic information.

use crate::media::decoders::{DecoderDef, H264_DECODERS, H265_DECODERS, MJPEG_DECODERS};
use std::sync::OnceLock;

/// Cached decoder availability (checked once at startup, per codec)
static MJPEG_AVAILABILITY: OnceLock<Vec<bool>> = OnceLock::new();
static H264_AVAILABILITY: OnceLock<Vec<bool>> = OnceLock::new();
static H265_AVAILABILITY: OnceLock<Vec<bool>> = OnceLock::new();

/// State for Insights drawer diagnostic information
#[derive(Debug, Clone, Default)]
pub struct InsightsState {
    // Pipeline info
    /// Full GStreamer pipeline string
    pub full_pipeline_string: Option<String>,
    /// Decoder fallback chain status
    pub decoder_chain: Vec<DecoderStatus>,

    // Current format chain
    /// Current format pipeline information
    pub format_chain: FormatChain,

    // Performance metrics
    /// Frame latency in microseconds
    pub frame_latency_us: u64,
    /// Total dropped frames count
    pub dropped_frames: u64,
    /// Frame size after decoding in bytes
    pub frame_size_decoded: usize,
    /// GStreamer decode/conversion time in microseconds
    pub gstreamer_decode_time_us: u64,
    /// GPU compute shader conversion time in microseconds
    pub gpu_conversion_time_us: u64,
    /// Copy time (source to GPU) in microseconds
    pub copy_time_us: u64,
    /// Copy bandwidth in MB/s
    pub copy_bandwidth_mbps: f64,
}

/// Status of a decoder in the fallback chain
#[derive(Debug, Clone)]
pub struct DecoderStatus {
    /// Decoder element name (e.g., "vaapijpegdec")
    pub name: &'static str,
    /// Human-readable description
    pub description: &'static str,
    /// Current state in the fallback chain
    pub state: FallbackState,
}

/// State of a decoder in the fallback chain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackState {
    /// Currently active
    Selected,
    /// Available but not selected
    Available,
    /// Not present on the system
    #[default]
    Unavailable,
}

/// Current format pipeline chain
#[derive(Debug, Clone, Default)]
pub struct FormatChain {
    /// Camera source type (e.g., "V4L2 via PipeWire", "libcamera via PipeWire")
    pub source: String,
    /// Current resolution
    pub resolution: String,
    /// Current framerate
    pub framerate: String,
    /// Native format from camera (e.g., "MJPG", "YUYV", "NV12")
    pub native_format: String,
    /// GStreamer output format after decoding (if applicable, e.g., "I420", "NV12")
    pub gstreamer_output: Option<String>,
    /// WGPU processing description (e.g., "I420 → RGBA", "Passthrough")
    pub wgpu_processing: String,
}

/// Get cached decoder availability for a decoder list
fn get_cached_availability(
    decoders: &[DecoderDef],
    cache: &'static OnceLock<Vec<bool>>,
) -> &'static Vec<bool> {
    cache.get_or_init(|| {
        decoders
            .iter()
            .map(|d| gstreamer::ElementFactory::find(d.name).is_some())
            .collect()
    })
}

/// Build a decoder status chain from decoder definitions
///
/// This is the generic builder that replaces the three format-specific methods.
fn build_chain_from_defs(
    decoders: &'static [DecoderDef],
    availability: &[bool],
    full_pipeline: Option<&str>,
) -> Vec<DecoderStatus> {
    // Find which decoder is actually used in the pipeline
    let active_decoder = full_pipeline.and_then(|pipeline| {
        decoders.iter().find_map(|d| {
            // Check for decoder name followed by space, '!', or end of string
            if pipeline.contains(&format!("{} ", d.name))
                || pipeline.contains(&format!("{}!", d.name))
                || pipeline.ends_with(d.name)
            {
                Some(d.name)
            } else {
                None
            }
        })
    });

    decoders
        .iter()
        .enumerate()
        .map(|(i, decoder)| {
            let state = if active_decoder == Some(decoder.name) {
                FallbackState::Selected
            } else if availability.get(i).copied().unwrap_or(false) {
                FallbackState::Available
            } else {
                FallbackState::Unavailable
            };
            DecoderStatus {
                name: decoder.name,
                description: decoder.description,
                state,
            }
        })
        .collect()
}

impl InsightsState {
    /// Build the decoder fallback chain based on pixel format
    ///
    /// `pixel_format` is the camera's native format (e.g., "MJPG", "H264", "YUYV")
    /// `full_pipeline` is the actual GStreamer pipeline string to parse for the active decoder.
    /// Decoder availability is cached on first call since it doesn't change at runtime.
    pub fn build_decoder_chain(
        pixel_format: Option<&str>,
        full_pipeline: Option<&str>,
    ) -> Vec<DecoderStatus> {
        match pixel_format {
            Some("MJPG") | Some("MJPEG") => {
                let availability = get_cached_availability(MJPEG_DECODERS, &MJPEG_AVAILABILITY);
                build_chain_from_defs(MJPEG_DECODERS, availability, full_pipeline)
            }
            Some("H264") => {
                let availability = get_cached_availability(H264_DECODERS, &H264_AVAILABILITY);
                build_chain_from_defs(H264_DECODERS, availability, full_pipeline)
            }
            Some("H265") | Some("HEVC") => {
                let availability = get_cached_availability(H265_DECODERS, &H265_AVAILABILITY);
                build_chain_from_defs(H265_DECODERS, availability, full_pipeline)
            }
            // Raw formats don't need decoders
            _ => Vec::new(),
        }
    }
}
