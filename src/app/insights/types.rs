// SPDX-License-Identifier: GPL-3.0-only

//! Types for the Insights drawer diagnostic information.

use crate::media::decoders::{DecoderDef, H264_DECODERS, H265_DECODERS, MJPEG_DECODERS};
use crate::pipelines::video::AudioLevels;
use std::sync::OnceLock;

/// Cached decoder availability (checked once at startup, per codec)
static MJPEG_AVAILABILITY: OnceLock<Vec<bool>> = OnceLock::new();
static H264_AVAILABILITY: OnceLock<Vec<bool>> = OnceLock::new();
static H265_AVAILABILITY: OnceLock<Vec<bool>> = OnceLock::new();

/// State for Insights drawer diagnostic information
#[derive(Debug, Clone, Default)]
pub struct InsightsState {
    // Pipeline info
    /// Full pipeline string
    pub full_pipeline_string: Option<String>,
    /// Decoder fallback chain status
    pub decoder_chain: Vec<DecoderStatus>,

    // Current format chain
    /// Current format pipeline information
    pub format_chain: FormatChain,

    // Measured framerate (computed from frame count deltas)
    /// Measured FPS from preview stream frame count
    pub measured_fps: f64,
    /// Previous frame count for FPS calculation
    pub prev_frame_count: u64,
    /// Timestamp of previous FPS measurement
    pub prev_fps_instant: Option<std::time::Instant>,
    /// Capture-session identity, including restarts of the same camera.
    pub(crate) preview_generation: Option<u64>,

    // Performance metrics
    /// Frame latency in microseconds
    pub frame_latency_us: u64,
    /// Measured application preview losses; excludes sensor/driver/server losses
    pub dropped_frames: Option<u64>,
    /// Frame size after decoding in bytes
    pub frame_size_decoded: usize,
    /// GPU compute shader conversion time in microseconds
    pub gpu_conversion_time_us: u64,
    /// CPU camera-buffer copy duration in microseconds (None before measurement)
    pub copy_time_us: Option<u64>,
    /// CPU decode time in microseconds (e.g., turbojpeg MJPEG→I420)
    pub cpu_decode_time_us: u64,
    /// CPU processing description (e.g., "MJPEG → I420 (turbojpeg)")
    pub cpu_processing: Option<String>,
    /// Copy bandwidth in MB/s
    pub copy_bandwidth_mbps: f64,

    // Backend info
    /// Active backend type
    pub backend_type: &'static str,
    /// libcamera pipeline handler (e.g., "RPiCFE", "simple")
    pub pipeline_handler: Option<String>,
    /// libcamera version string
    pub libcamera_version: Option<String>,
    /// Sensor model (e.g., "sony,imx371")
    pub sensor_model: Option<String>,
    /// MJPEG decoder name when native libcamera decodes MJPEG (e.g., "turbojpeg (libjpeg-turbo)")
    pub mjpeg_decoder: Option<String>,

    // Per-frame libcamera metadata (updated every frame)
    /// Actual exposure time applied (microseconds)
    pub meta_exposure_us: Option<u64>,
    /// Actual analogue gain applied
    pub meta_analogue_gain: Option<f32>,
    /// Actual digital gain applied
    pub meta_digital_gain: Option<f32>,
    /// Color temperature (Kelvin)
    pub meta_colour_temperature: Option<u32>,
    /// Frame sequence number
    pub meta_sequence: Option<u32>,
    /// ISP white balance gains [R, B]
    pub meta_colour_gains: Option<[f32; 2]>,
    /// Sensor black level (normalized 0..1)
    pub meta_black_level: Option<f32>,
    /// Lens position (dioptres)
    pub meta_lens_position: Option<f32>,
    /// Scene illuminance (lux)
    pub meta_lux: Option<f32>,
    /// Focus figure of merit
    pub meta_focus_fom: Option<i32>,
    /// Whether libcamera metadata is present at all
    pub has_libcamera_metadata: bool,
    /// Full libcamera FrameMetadata snapshot (includes fields not shown in UI:
    /// sensor_timestamp, colour_correction_matrix, af/ae/awb state)
    pub frame_metadata: Option<crate::backends::camera::types::FrameMetadata>,

    // Recording pipeline info (populated while recording)
    /// Recording pipeline diagnostics (None when not recording)
    pub recording_diag: Option<crate::pipelines::video::RecordingDiagnostics>,
    /// Live recording pipeline stats (None when not recording)
    pub recording_stats: Option<crate::pipelines::video::RecordingStatsSnapshot>,

    // Audio levels (snapshot from SharedAudioLevels)
    /// Last polled audio level data (updated by insights tick)
    pub audio_levels: Option<AudioLevels>,

    // V4L2 format enumeration
    /// All V4L2 formats reported by the kernel driver
    pub v4l2_formats: Vec<crate::backends::camera::v4l2_utils::V4l2FormatInfo>,
    /// Formats available through the selected capture backend (for comparison)
    pub backend_formats: Vec<crate::backends::camera::types::CameraFormat>,
    /// V4L2 device path that was used to populate v4l2_formats (for cache invalidation)
    pub v4l2_formats_device: String,

    // Multi-stream info
    /// Whether dual-stream mode is active
    pub is_multistream: bool,
    /// Whether libcamera itself supports dual-stream (ViewFinder + Raw)
    /// even if GStreamer's Software ISP can't handle it
    pub libcamera_multistream_capable: bool,
    /// Preview stream details
    pub preview_stream: Option<StreamInfo>,
    /// Capture stream details
    pub capture_stream: Option<StreamInfo>,
}

/// Information about an individual stream in a multi-stream pipeline
#[derive(Debug, Clone, Default)]
pub struct StreamInfo {
    /// Stream role (e.g., "View-finder", "Raw", "Still-capture")
    pub role: String,
    /// Negotiated resolution (e.g., "1920x1080")
    pub resolution: String,
    /// Negotiated pixel format (e.g., "NV12", "SRGGB10_CSI2P")
    pub pixel_format: String,
    /// Total frames received on this stream
    pub frame_count: u64,
    /// Source description (e.g., "libcamera (native)")
    pub source: String,
    /// GPU processing description (e.g., "Bayer unpack → demosaic")
    pub gpu_processing: String,
    /// Frame size in bytes (from stride * height or buffer size)
    pub frame_size_bytes: usize,
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
    /// Camera source type (e.g., "libcamera (native)")
    pub source: String,
    /// Current resolution
    pub resolution: String,
    /// Current framerate
    pub framerate: String,
    /// Native format from camera (e.g., "MJPG", "YUYV", "NV12")
    pub native_format: String,
    /// WGPU processing description (e.g., "I420 → RGBA", "Passthrough")
    pub wgpu_processing: String,
}

/// Human-readable frame-buffer copy measurement, shared by UI and bug reports.
pub(crate) fn frame_copy_time_text(time_us: Option<u64>) -> String {
    match time_us {
        None => "N/A".into(),
        Some(us) if us < 10 => "< 0.01 ms (measured copy)".into(),
        Some(us) => format!("{:.2} ms", us as f64 / 1000.0),
    }
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
    pub(crate) fn refresh_preview(
        &mut self,
        path: &str,
        snapshot: Option<&crate::backends::camera::preview_metrics::PreviewSnapshot>,
        now: std::time::Instant,
    ) {
        use crate::backends::camera::libcamera::native::diagnostics as diag;
        use crate::backends::camera::types::PixelFormat;

        if snapshot.map(|s| s.generation) != self.preview_generation || snapshot.is_none() {
            // Keep independently refreshed capabilities/recording/audio, but never
            // carry pipeline information or sensor metadata across capture sessions.
            *self = Self {
                v4l2_formats: std::mem::take(&mut self.v4l2_formats),
                backend_formats: std::mem::take(&mut self.backend_formats),
                v4l2_formats_device: std::mem::take(&mut self.v4l2_formats_device),
                recording_diag: self.recording_diag.take(),
                recording_stats: self.recording_stats.take(),
                audio_levels: self.audio_levels.take(),
                ..Default::default()
            };
        }
        let Some(snapshot) = snapshot else { return };
        self.preview_generation = Some(snapshot.generation);
        let is_pipewire = path.starts_with("pipewire:");
        self.backend_type = if is_pipewire {
            "PipeWire (V4L2)"
        } else {
            "libcamera"
        };
        self.format_chain.source = if is_pipewire {
            "PipeWire (V4L2)"
        } else {
            "libcamera (native)"
        }
        .into();
        self.format_chain.resolution = format!("{}x{}", snapshot.width, snapshot.height);
        self.preview_stream = (snapshot.frame_count > 0).then(|| StreamInfo {
            role: "Preview + Capture".into(),
            resolution: self.format_chain.resolution.clone(),
            pixel_format: snapshot
                .output_format
                .map(|f| f.to_gst_format_string().into())
                .unwrap_or_default(),
            frame_count: snapshot.frame_count,
            source: self.format_chain.source.clone(),
            frame_size_bytes: snapshot.frame_size_bytes,
            ..Default::default()
        });
        self.frame_size_decoded = snapshot.frame_size_bytes;
        self.copy_time_us = snapshot.copy_time_us;
        self.dropped_frames = Some(snapshot.dropped_frames);
        self.format_chain.wgpu_processing = match snapshot.output_format {
            Some(PixelFormat::RGBA) => "RGBA passthrough (texture upload)".into(),
            Some(format) => format!("{} → RGBA (compute shader)", format.to_gst_format_string()),
            None => "N/A".into(),
        };
        if is_pipewire {
            self.pipeline_handler = None;
            self.libcamera_version = None;
            self.libcamera_multistream_capable = false;
            self.is_multistream = false;
            self.capture_stream = None;
            self.mjpeg_decoder = None;
            self.cpu_decode_time_us = 0;
            self.sensor_model = None;
            self.clear_frame_metadata();
            if let Some(stream) = &snapshot.stream {
                self.full_pipeline_string = Some(stream.pipeline_string.clone());
                self.format_chain.native_format = stream.source_format.clone();
                self.format_chain.framerate = stream
                    .framerate
                    .map(|f| format!("{f} fps"))
                    .unwrap_or_else(|| "N/A".into());
                self.cpu_processing = stream.cpu_processing.clone();
                self.decoder_chain = stream
                    .decoder
                    .map(|name| {
                        vec![DecoderStatus {
                            name,
                            description: "GStreamer JPEG decoder",
                            state: FallbackState::Selected,
                        }]
                    })
                    .unwrap_or_default();
            }
        } else {
            self.full_pipeline_string = diag::get_pipeline_string();
            self.is_multistream = diag::get_is_multistream();
            self.mjpeg_decoder = diag::get_mjpeg_decoder();
            self.cpu_decode_time_us = diag::get_mjpeg_decode_time_us();
            if let Some((_, format, role, _)) = diag::get_preview_stream_info() {
                self.format_chain.native_format = format.clone();
                if let Some(stream) = &mut self.preview_stream {
                    // Native MJPEG labels include the actual decoded subsampling
                    // (e.g. I422), which PixelFormat::I420 alone cannot express.
                    stream.pixel_format = format;
                    stream.role = role;
                }
            }
            self.capture_stream = diag::get_capture_stream_info().map(
                |(resolution, pixel_format, role, frame_count, stride, height)| StreamInfo {
                    gpu_processing: if pixel_format.contains("CSI2P") {
                        format!("{pixel_format} → unpack (compute) → demosaic (compute) → RGBA")
                    } else if pixel_format.starts_with('S') && pixel_format.contains("GG") {
                        format!("{pixel_format} → demosaic (compute) → RGBA")
                    } else {
                        String::new()
                    },
                    resolution,
                    pixel_format,
                    role,
                    frame_count,
                    source: "libcamera (native)".into(),
                    frame_size_bytes: stride as usize * height as usize,
                },
            );
            self.decoder_chain = if self.mjpeg_decoder.is_some() {
                vec![DecoderStatus {
                    name: "turbojpeg",
                    description: "libjpeg-turbo MJPEG decoder",
                    state: FallbackState::Selected,
                }]
            } else {
                Vec::new()
            };
            if let Some(decoded) = diag::get_mjpeg_decoded_format() {
                let note = diag::get_mjpeg_entropy_note()
                    .map(|note| format!(" ({note})"))
                    .unwrap_or_default();
                self.cpu_processing =
                    Some(format!("MJPEG{note} → {decoded} (turbojpeg); buffer copy"));
                self.format_chain.wgpu_processing = format!("{decoded} → RGBA (compute shader)");
            } else {
                self.cpu_processing = Some("Camera buffer copy".into());
            }
        }
        if let Some(previous) = self.prev_fps_instant {
            let elapsed = now.saturating_duration_since(previous).as_secs_f64();
            if elapsed < 0.1 {
                return;
            }
            let delta = snapshot.frame_count.saturating_sub(self.prev_frame_count);
            let instantaneous = delta as f64 / elapsed;
            self.measured_fps = if delta == 0 || self.measured_fps == 0.0 {
                instantaneous
            } else {
                self.measured_fps * 0.7 + instantaneous * 0.3
            };
        }
        self.prev_frame_count = snapshot.frame_count;
        self.prev_fps_instant = Some(now);
    }

    pub(crate) fn clear_frame_metadata(&mut self) {
        self.has_libcamera_metadata = false;
        self.meta_exposure_us = None;
        self.meta_analogue_gain = None;
        self.meta_digital_gain = None;
        self.meta_colour_temperature = None;
        self.meta_sequence = None;
        self.meta_colour_gains = None;
        self.meta_black_level = None;
        self.meta_lens_position = None;
        self.meta_lux = None;
        self.meta_focus_fom = None;
        self.frame_metadata = None;
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::preview_metrics::{PreviewSnapshot, StreamDiagnostics};
    use crate::backends::camera::types::{Framerate, PixelFormat};

    #[test]
    fn stopped_preview_clears_pipeline_metadata_and_performance() {
        let mut state = InsightsState {
            backend_type: "libcamera",
            has_libcamera_metadata: true,
            meta_exposure_us: Some(1000),
            meta_analogue_gain: Some(2.0),
            full_pipeline_string: Some("old camera".into()),
            copy_time_us: Some(42),
            dropped_frames: Some(5),
            measured_fps: 30.0,
            ..Default::default()
        };
        state.refresh_preview("pipewire:123", None, std::time::Instant::now());
        assert!(state.backend_type.is_empty());
        assert!(state.full_pipeline_string.is_none());
        assert!(!state.has_libcamera_metadata);
        assert!(state.meta_exposure_us.is_none());
        assert!(state.meta_analogue_gain.is_none());
        assert!(state.copy_time_us.is_none());
        assert!(state.dropped_frames.is_none());
        assert_eq!(state.measured_fps, 0.0);
    }

    #[test]
    fn measured_fps_uses_fallback_frames_and_resets_on_stall_and_restart() {
        let start = std::time::Instant::now();
        let mut state = InsightsState::default();
        let mut snapshot = PreviewSnapshot {
            generation: 1,
            frame_count: 10,
            ..Default::default()
        };
        state.refresh_preview("pipewire:123", Some(&snapshot), start);
        snapshot.frame_count = 40;
        state.refresh_preview(
            "pipewire:123",
            Some(&snapshot),
            start + std::time::Duration::from_secs(1),
        );
        assert_eq!(state.measured_fps, 30.0);
        state.refresh_preview(
            "pipewire:123",
            Some(&snapshot),
            start + std::time::Duration::from_secs(2),
        );
        assert_eq!(state.measured_fps, 0.0);
        snapshot.generation = 2;
        snapshot.frame_count = 1;
        state.refresh_preview(
            "pipewire:123",
            Some(&snapshot),
            start + std::time::Duration::from_secs(3),
        );
        assert_eq!(state.measured_fps, 0.0);
        assert_eq!(state.prev_frame_count, 1);
    }

    #[test]
    fn same_session_pipewire_frames_cannot_retain_native_sensor_metadata() {
        let mut state = InsightsState {
            preview_generation: Some(1),
            has_libcamera_metadata: true,
            meta_exposure_us: Some(1000),
            meta_lens_position: Some(4.0),
            libcamera_version: Some("old".into()),
            pipeline_handler: Some("simple".into()),
            ..Default::default()
        };
        state.refresh_preview(
            "pipewire:123",
            Some(&PreviewSnapshot {
                generation: 1,
                ..Default::default()
            }),
            std::time::Instant::now(),
        );
        assert!(!state.has_libcamera_metadata);
        assert!(state.meta_exposure_us.is_none());
        assert!(state.meta_lens_position.is_none());
        assert!(state.libcamera_version.is_none());
        assert!(state.pipeline_handler.is_none());
    }

    #[test]
    fn unavailable_copy_time_is_not_reported_as_zero_copy() {
        assert_eq!(frame_copy_time_text(None), "N/A");
        assert_eq!(frame_copy_time_text(Some(0)), "< 0.01 ms (measured copy)");
        assert_eq!(frame_copy_time_text(Some(37)), "0.04 ms");
    }

    #[test]
    fn pipewire_insights_describe_actual_cpu_conversion_and_rgba_preview() {
        let mut insights = InsightsState::default();
        let snapshot = PreviewSnapshot {
            generation: 1,
            frame_count: 42,
            dropped_frames: 3,
            copy_time_us: Some(25),
            width: 1328,
            height: 1760,
            frame_size_bytes: 9_349_120,
            output_format: Some(PixelFormat::RGBA),
            stream: Some(StreamDiagnostics {
                pipeline_string: "pipewiresrc ! NV12 ! videoconvert ! RGBA ! appsink".into(),
                source_format: "NV12".into(),
                framerate: Some(Framerate::new(30, 1)),
                cpu_processing: Some("NV12 → RGBA (videoconvert); buffer copy".into()),
                decoder: None,
            }),
        };
        insights.refresh_preview("pipewire:123", Some(&snapshot), std::time::Instant::now());
        assert_eq!(insights.backend_type, "PipeWire (V4L2)");
        assert_eq!(insights.format_chain.source, "PipeWire (V4L2)");
        assert_eq!(insights.format_chain.native_format, "NV12");
        assert_eq!(
            insights.format_chain.wgpu_processing,
            "RGBA passthrough (texture upload)"
        );
        assert!(
            insights
                .cpu_processing
                .as_ref()
                .unwrap()
                .contains("videoconvert")
        );
        assert_eq!(insights.preview_stream.as_ref().unwrap().frame_count, 42);
        assert_eq!(insights.frame_size_decoded, snapshot.frame_size_bytes);
        assert!(
            insights
                .full_pipeline_string
                .as_ref()
                .unwrap()
                .contains("pipewiresrc")
        );
        assert!(insights.decoder_chain.is_empty());
    }
}
