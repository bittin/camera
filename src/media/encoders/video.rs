// SPDX-License-Identifier: GPL-3.0-only

//! Video encoder selection with hardware acceleration priority
//!
//! This module implements intelligent video encoder selection based on:
//! - Hardware encoder availability (AV1 > HEVC > H.264)
//! - Software fallbacks for maximum compatibility
//! - Configurable quality presets

use gstreamer as gst;
use gstreamer::prelude::*;
use tracing::{debug, info, warn};

/// Blacklisted software AV1 encoders that cause issues in Flatpak environments
/// See: https://github.com/cosmic-utils/cosmic-ext-camera/issues/171
/// - svtav1enc (SVT-AV1): No file is created when recording
/// - av1enc (AOM AV1): Recording terminates immediately with unplayable output
const BLACKLISTED_ENCODERS: &[&str] = &["svtav1enc", "av1enc"];

/// Video codec types in priority order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// AV1 codec (best compression, modern)
    AV1,
    /// HEVC/H.265 codec (good compression)
    HEVC,
    /// H.264 codec (best compatibility)
    H264,
}

impl VideoCodec {
    /// Get the container format for this codec
    pub fn container_format(&self) -> ContainerFormat {
        match self {
            VideoCodec::AV1 => ContainerFormat::WebM,
            VideoCodec::HEVC => ContainerFormat::MP4,
            VideoCodec::H264 => ContainerFormat::MP4,
        }
    }

    /// Get the file extension for this codec's container
    pub fn file_extension(&self) -> &'static str {
        self.container_format().extension()
    }

    /// Get the parser element name (if needed)
    pub fn parser_name(&self) -> Option<&'static str> {
        match self {
            VideoCodec::AV1 => Some("av1parse"),
            VideoCodec::HEVC => Some("h265parse"),
            VideoCodec::H264 => Some("h264parse"),
        }
    }
}

/// Container formats for video
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// MP4 container (good compatibility)
    MP4,
    /// WebM container (open format)
    WebM,
}

impl ContainerFormat {
    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            ContainerFormat::MP4 => "mp4",
            ContainerFormat::WebM => "webm",
        }
    }

    /// Get muxer element name
    pub fn muxer_name(&self) -> &'static str {
        match self {
            ContainerFormat::MP4 => "mp4mux",
            ContainerFormat::WebM => "webmmux",
        }
    }
}

/// Video quality presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoQuality {
    /// Low quality (high compression, smaller files)
    Low,
    /// Medium quality (balanced)
    Medium,
    /// High quality (low compression, larger files)
    High,
    /// Maximum quality (minimal compression)
    Maximum,
}

impl VideoQuality {
    /// Get bitrate in kbps for given quality
    ///
    /// Bitrate scales with resolution using realistic encoding factors:
    /// - 720p: ~2-8 Mbps depending on quality
    /// - 1080p: ~4-15 Mbps depending on quality
    /// - 4K: ~15-40 Mbps depending on quality
    pub fn bitrate_kbps(&self, width: u32, height: u32) -> u32 {
        let pixels = width * height;
        // Factors chosen to give sensible bitrates:
        // 1080p (2M pixels): Low ~4Mbps, Med ~8Mbps, High ~12Mbps, Max ~20Mbps
        let base_bitrate = match self {
            VideoQuality::Low => (pixels as f64 * 0.002) as u32,
            VideoQuality::Medium => (pixels as f64 * 0.004) as u32,
            VideoQuality::High => (pixels as f64 * 0.006) as u32,
            VideoQuality::Maximum => (pixels as f64 * 0.010) as u32,
        };
        // Ensure minimum 500 kbps, maximum 50000 kbps (50 Mbps)
        base_bitrate.clamp(500, 50000)
    }

    /// Get x264/x265 preset name
    ///
    /// Even the Low preset uses `veryfast` rather than `ultrafast` because
    /// `ultrafast` disables CABAC and most motion estimation, producing
    /// very poor quality even at high bitrates. `veryfast` is still
    /// real-time capable on ARM devices at 1080p.
    pub fn x264_preset(&self) -> &'static str {
        match self {
            VideoQuality::Low => "veryfast",
            VideoQuality::Medium => "faster",
            VideoQuality::High => "medium",
            VideoQuality::Maximum => "slow",
        }
    }
}

/// Information about an available encoder
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderInfo {
    /// GStreamer element name
    pub element_name: String,
    /// Display name for UI
    pub display_name: String,
    /// Codec type
    pub codec: VideoCodec,
    /// Whether this is hardware accelerated
    pub is_hardware: bool,
    /// Priority (lower = higher priority)
    pub priority: u32,
}

/// Selected video encoder with configuration
pub struct SelectedVideoEncoder {
    /// The encoder element
    pub encoder: gst::Element,
    /// Optional parser element
    pub parser: Option<gst::Element>,
    /// Muxer element
    pub muxer: gst::Element,
    /// Codec being used
    pub codec: VideoCodec,
    /// Container format
    pub container: ContainerFormat,
    /// File extension
    pub extension: &'static str,
}

/// Select a CPU-backed raw format accepted by both `videoconvert` and an encoder.
///
/// Hardware encoder sink caps are device-specific. Prefer common 4:2:0 formats,
/// then consider every raw format known to GStreamer so V4L2 encoders are not
/// tied to a hard-coded device format.
pub fn select_raw_encoder_input_format(
    encoder_caps: &gst::Caps,
    converter_caps: &gst::Caps,
    width: u32,
    height: u32,
    framerate: u32,
) -> Option<String> {
    let preferred = ["NV12", "I420", "YV12", "NV21", "P010_10LE", "YUY2", "UYVY"];
    let mut formats: Vec<String> = preferred
        .iter()
        .map(|format| (*format).to_string())
        .collect();

    for format in gstreamer_video::VideoFormat::iter_raw() {
        let format = format.to_str().to_string();
        if !formats.contains(&format) {
            formats.push(format);
        }
    }

    formats.into_iter().find(|format| {
        let candidate = gst::Caps::builder("video/x-raw")
            .field("format", format.as_str())
            .field("width", width as i32)
            .field("height", height as i32)
            .field("framerate", gst::Fraction::new(framerate as i32, 1))
            .build();
        encoder_caps.can_intersect(&candidate) && converter_caps.can_intersect(&candidate)
    })
}

/// Query an encoder's actual sink caps and choose a compatible CPU-memory format.
pub fn negotiate_raw_encoder_input_format(
    encoder: &gst::Element,
    width: u32,
    height: u32,
    framerate: u32,
) -> Option<String> {
    let encoder_caps = encoder.static_pad("sink")?.query_caps(None);
    let converter = gst::ElementFactory::make("videoconvert").build().ok()?;
    let converter_caps = converter.static_pad("src")?.query_caps(None);

    debug!(
        encoder = encoder.name().as_str(),
        caps = %encoder_caps,
        width,
        height,
        framerate,
        "Negotiating raw encoder input format"
    );

    select_raw_encoder_input_format(&encoder_caps, &converter_caps, width, height, framerate)
}

/// Immutable facts shared by UI discovery and automatic selection.
/// UI priorities and auto ranks are independent policy: in particular x264 is
/// preferred over x265 automatically, and QSV/AMF are explicit/UI-only.
struct EncoderSpec {
    name: &'static str,
    display_name: &'static str,
    codec: VideoCodec,
    hardware: bool,
    ui_priority: u32,
    auto_rank: Option<u8>,
}

const ENCODER_SPECS: &[EncoderSpec] = &[
    EncoderSpec {
        name: "vaav1enc",
        display_name: "VA-API AV1 (HW)",
        codec: VideoCodec::AV1,
        hardware: true,
        ui_priority: 1,
        auto_rank: Some(0),
    },
    EncoderSpec {
        name: "nvav1enc",
        display_name: "NVIDIA AV1 (HW)",
        codec: VideoCodec::AV1,
        hardware: true,
        ui_priority: 2,
        auto_rank: Some(1),
    },
    EncoderSpec {
        name: "qsvav1enc",
        display_name: "Intel QSV AV1 (HW)",
        codec: VideoCodec::AV1,
        hardware: true,
        ui_priority: 3,
        auto_rank: None,
    },
    EncoderSpec {
        name: "amfav1enc",
        display_name: "AMD AMF AV1 (HW)",
        codec: VideoCodec::AV1,
        hardware: true,
        ui_priority: 4,
        auto_rank: None,
    },
    EncoderSpec {
        name: "svtav1enc",
        display_name: "SVT-AV1 (SW)",
        codec: VideoCodec::AV1,
        hardware: false,
        ui_priority: 10,
        auto_rank: None,
    },
    EncoderSpec {
        name: "av1enc",
        display_name: "AOM AV1 (SW)",
        codec: VideoCodec::AV1,
        hardware: false,
        ui_priority: 11,
        auto_rank: None,
    },
    EncoderSpec {
        name: "vah265enc",
        display_name: "VA-API H.265 (HW)",
        codec: VideoCodec::HEVC,
        hardware: true,
        ui_priority: 21,
        auto_rank: Some(2),
    },
    EncoderSpec {
        name: "nvh265enc",
        display_name: "NVIDIA H.265 (HW)",
        codec: VideoCodec::HEVC,
        hardware: true,
        ui_priority: 22,
        auto_rank: Some(3),
    },
    EncoderSpec {
        name: "qsvh265enc",
        display_name: "Intel QSV H.265 (HW)",
        codec: VideoCodec::HEVC,
        hardware: true,
        ui_priority: 23,
        auto_rank: None,
    },
    EncoderSpec {
        name: "amfh265enc",
        display_name: "AMD AMF H.265 (HW)",
        codec: VideoCodec::HEVC,
        hardware: true,
        ui_priority: 24,
        auto_rank: None,
    },
    EncoderSpec {
        name: "v4l2h265enc",
        display_name: "V4L2 H.265 (HW)",
        codec: VideoCodec::HEVC,
        hardware: true,
        ui_priority: 25,
        auto_rank: Some(4),
    },
    EncoderSpec {
        name: "x265enc",
        display_name: "x265 H.265 (SW)",
        codec: VideoCodec::HEVC,
        hardware: false,
        ui_priority: 31,
        auto_rank: Some(9),
    },
    EncoderSpec {
        name: "vah264enc",
        display_name: "VA-API H.264 (HW)",
        codec: VideoCodec::H264,
        hardware: true,
        ui_priority: 41,
        auto_rank: Some(5),
    },
    EncoderSpec {
        name: "nvh264enc",
        display_name: "NVIDIA H.264 (HW)",
        codec: VideoCodec::H264,
        hardware: true,
        ui_priority: 42,
        auto_rank: Some(6),
    },
    EncoderSpec {
        name: "qsvh264enc",
        display_name: "Intel QSV H.264 (HW)",
        codec: VideoCodec::H264,
        hardware: true,
        ui_priority: 43,
        auto_rank: None,
    },
    EncoderSpec {
        name: "amfh264enc",
        display_name: "AMD AMF H.264 (HW)",
        codec: VideoCodec::H264,
        hardware: true,
        ui_priority: 44,
        auto_rank: None,
    },
    EncoderSpec {
        name: "v4l2h264enc",
        display_name: "V4L2 H.264 (HW)",
        codec: VideoCodec::H264,
        hardware: true,
        ui_priority: 45,
        auto_rank: Some(7),
    },
    EncoderSpec {
        name: "x264enc",
        display_name: "x264 H.264 (SW)",
        codec: VideoCodec::H264,
        hardware: false,
        ui_priority: 30,
        auto_rank: Some(8),
    },
    EncoderSpec {
        name: "openh264enc",
        display_name: "OpenH264 H.264 (SW)",
        codec: VideoCodec::H264,
        hardware: false,
        ui_priority: 32,
        auto_rank: Some(10),
    },
];

fn automatic_encoder_specs() -> Vec<&'static EncoderSpec> {
    let mut specs: Vec<_> = ENCODER_SPECS
        .iter()
        .filter(|s| s.auto_rank.is_some())
        .collect();
    specs.sort_by_key(|s| s.auto_rank);
    specs
}

/// Enumerate available, non-blacklisted encoders in UI priority order.
pub fn enumerate_video_encoders() -> Vec<EncoderInfo> {
    let _ = gst::init();

    let mut available_encoders = Vec::new();

    for spec in ENCODER_SPECS {
        let element_name = spec.name;
        // Skip blacklisted encoders
        if BLACKLISTED_ENCODERS.contains(&element_name) {
            continue;
        }

        if gst::ElementFactory::find(element_name).is_some() {
            available_encoders.push(EncoderInfo {
                element_name: element_name.to_string(),
                display_name: spec.display_name.to_string(),
                codec: spec.codec,
                is_hardware: spec.hardware,
                priority: spec.ui_priority,
            });
        }
    }

    // Sort by priority (lower number = higher priority)
    available_encoders.sort_by_key(|e| e.priority);

    available_encoders
}

/// Create encoder from encoder info with optional bitrate override
pub fn create_encoder_from_info_with_bitrate(
    info: &EncoderInfo,
    quality: VideoQuality,
    width: u32,
    height: u32,
    bitrate_override_kbps: Option<u32>,
) -> Result<SelectedVideoEncoder, String> {
    let encoder = gst::ElementFactory::make(&info.element_name)
        .build()
        .map_err(|e| format!("Failed to create encoder {}: {}", info.element_name, e))?;

    configure_and_assemble(
        encoder,
        &info.element_name,
        info.codec,
        quality,
        width,
        height,
        bitrate_override_kbps,
    )
}

fn configure_and_assemble(
    encoder: gst::Element,
    name: &str,
    codec: VideoCodec,
    quality: VideoQuality,
    width: u32,
    height: u32,
    bitrate_override_kbps: Option<u32>,
) -> Result<SelectedVideoEncoder, String> {
    configure_video_encoder(
        &encoder,
        name,
        quality,
        width,
        height,
        bitrate_override_kbps,
    );
    assemble_video_encoder(encoder, codec)
}

fn assemble_video_encoder(
    encoder: gst::Element,
    codec: VideoCodec,
) -> Result<SelectedVideoEncoder, String> {
    // Create parser if needed
    let parser = if let Some(parser_name) = codec.parser_name() {
        match gst::ElementFactory::make(parser_name).build() {
            Ok(p) => {
                debug!("Created parser: {}", parser_name);
                Some(p)
            }
            Err(e) => {
                warn!("Failed to create parser {}: {}", parser_name, e);
                None
            }
        }
    } else {
        None
    };

    // Create muxer
    let container = codec.container_format();
    let muxer = gst::ElementFactory::make(container.muxer_name())
        .build()
        .map_err(|e| format!("Failed to create muxer {}: {}", container.muxer_name(), e))?;

    Ok(SelectedVideoEncoder {
        encoder,
        parser,
        muxer,
        codec,
        container,
        extension: codec.file_extension(),
    })
}

/// Select the best available video encoder with optional bitrate override
pub fn select_video_encoder_with_bitrate(
    quality: VideoQuality,
    width: u32,
    height: u32,
    bitrate_override_kbps: Option<u32>,
) -> Result<SelectedVideoEncoder, String> {
    gst::init().map_err(|e| format!("Failed to initialize GStreamer: {}", e))?;

    // Try encoders in priority order
    for spec in automatic_encoder_specs() {
        let encoder_name = spec.name;
        if let Ok(encoder) = gst::ElementFactory::make(encoder_name).build() {
            info!(
                encoder = %encoder_name,
                codec = ?spec.codec,
                hardware = spec.hardware,
                "Selected video encoder"
            );

            return configure_and_assemble(
                encoder,
                encoder_name,
                spec.codec,
                quality,
                width,
                height,
                bitrate_override_kbps,
            );
        }
    }

    Err("No video encoder available. Please install gstreamer1-plugins-ugly (x264enc) or gstreamer1-plugin-openh264".to_string())
}

/// Configure encoder based on type and quality
pub fn configure_video_encoder(
    encoder: &gst::Element,
    encoder_name: &str,
    quality: VideoQuality,
    width: u32,
    height: u32,
    bitrate_override_kbps: Option<u32>,
) {
    // Use bitrate override if provided, otherwise calculate from quality preset
    let bitrate = bitrate_override_kbps.unwrap_or_else(|| quality.bitrate_kbps(width, height));

    match encoder_name {
        // x264 software encoder
        "x264enc" => {
            encoder.set_property_from_str("speed-preset", quality.x264_preset());
            encoder.set_property("bitrate", bitrate);
            debug!(
                "Configured x264enc: preset={}, bitrate={} kbps",
                quality.x264_preset(),
                bitrate
            );
        }

        // x265 software encoder
        "x265enc" => {
            encoder.set_property_from_str("speed-preset", quality.x264_preset());
            encoder.set_property("bitrate", bitrate);
            debug!(
                "Configured x265enc: preset={}, bitrate={} kbps",
                quality.x264_preset(),
                bitrate
            );
        }

        // NVIDIA encoders (NVENC)
        // Preset p1-p7 scale from fastest (p1) to best quality (p7).
        "nvh264enc" | "nvh265enc" | "nvav1enc" => {
            encoder.set_property("bitrate", bitrate);
            encoder.set_property_from_str("rc-mode", "vbr");
            let preset = match quality {
                VideoQuality::Low => "p1",
                VideoQuality::Medium => "p3",
                VideoQuality::High => "p5",
                VideoQuality::Maximum => "p7",
            };
            encoder.set_property_from_str("preset", preset);
            debug!(
                "Configured NVIDIA encoder: preset={}, bitrate={} kbps",
                preset, bitrate
            );
        }

        // V4L2 hardware encoders
        "v4l2h264enc" | "v4l2h265enc" => {
            // V4L2 encoders typically have limited configuration
            debug!("Using V4L2 encoder with default configuration");
        }

        // OpenH264 (software H.264 encoder)
        "openh264enc" => {
            encoder.set_property_from_str("rate-control", "bitrate");
            encoder.set_property("bitrate", bitrate * 1000); // Bits per second
            debug!(
                "Configured openh264enc: rate-control=bitrate, bitrate={} bps",
                bitrate * 1000
            );
        }

        // SVT-AV1 encoder
        "svtav1enc" => {
            encoder.set_property("target-bitrate", bitrate);
            let preset = match quality {
                VideoQuality::Low => 8,
                VideoQuality::Medium => 6,
                VideoQuality::High => 4,
                VideoQuality::Maximum => 2,
            };
            encoder.set_property("speed", preset);
            debug!(
                "Configured svtav1enc: preset={}, bitrate={} kbps",
                preset, bitrate
            );
        }

        // AOM AV1 encoder
        "av1enc" => {
            encoder.set_property("target-bitrate", bitrate); // kilobits per second
            let cpu_used = match quality {
                VideoQuality::Low => 8,
                VideoQuality::Medium => 5,
                VideoQuality::High => 3,
                VideoQuality::Maximum => 1,
            };
            encoder.set_property("cpu-used", cpu_used);
            debug!(
                "Configured av1enc: cpu-used={}, bitrate={} kbps",
                cpu_used, bitrate
            );
        }

        // VA-API encoders
        "vaav1enc" | "vah264enc" | "vah265enc" => {
            encoder.set_property_from_str("rate-control", "cbr");
            encoder.set_property("bitrate", bitrate);
            debug!("Configured VA-API encoder: bitrate={} kbps", bitrate);
        }

        // AMD AMF encoders
        "amfh264enc" | "amfh265enc" | "amfav1enc" => {
            encoder.set_property("bitrate", bitrate);
            encoder.set_property_from_str("rate-control", "cbr");
            debug!("Configured AMD AMF encoder: bitrate={} kbps", bitrate);
        }

        // Intel QSV encoders
        "qsvh264enc" | "qsvh265enc" | "qsvav1enc" => {
            encoder.set_property("bitrate", bitrate);
            debug!("Configured Intel QSV encoder: bitrate={} kbps", bitrate);
        }

        _ => {
            debug!("Unknown encoder type, using default configuration");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn registry_keeps_distinct_automatic_and_ui_orders() {
        let automatic: Vec<_> = automatic_encoder_specs().iter().map(|s| s.name).collect();
        assert_eq!(
            automatic,
            [
                "vaav1enc",
                "nvav1enc",
                "vah265enc",
                "nvh265enc",
                "v4l2h265enc",
                "vah264enc",
                "nvh264enc",
                "v4l2h264enc",
                "x264enc",
                "x265enc",
                "openh264enc",
            ]
        );
        let mut ui: Vec<_> = ENCODER_SPECS
            .iter()
            .filter(|s| !BLACKLISTED_ENCODERS.contains(&s.name))
            .collect();
        ui.sort_by_key(|s| s.ui_priority);
        assert_eq!(
            ui.iter().map(|s| s.name).collect::<Vec<_>>(),
            [
                "vaav1enc",
                "nvav1enc",
                "qsvav1enc",
                "amfav1enc",
                "vah265enc",
                "nvh265enc",
                "qsvh265enc",
                "amfh265enc",
                "v4l2h265enc",
                "x264enc",
                "x265enc",
                "openh264enc",
                "vah264enc",
                "nvh264enc",
                "qsvh264enc",
                "amfh264enc",
                "v4l2h264enc",
            ]
        );
        let names: std::collections::HashSet<_> = ENCODER_SPECS.iter().map(|s| s.name).collect();
        assert_eq!(names.len(), ENCODER_SPECS.len());
        let ranks: std::collections::HashSet<_> =
            ENCODER_SPECS.iter().filter_map(|s| s.auto_rank).collect();
        assert_eq!(ranks.len(), automatic.len());
        for spec in ENCODER_SPECS {
            let expected_codec = if spec.name.contains("av1") {
                VideoCodec::AV1
            } else if spec.name.contains("265") {
                VideoCodec::HEVC
            } else {
                VideoCodec::H264
            };
            assert_eq!(spec.codec, expected_codec, "{}", spec.name);
            assert_eq!(
                spec.hardware,
                !["svtav1enc", "av1enc", "x264enc", "x265enc", "openh264enc"].contains(&spec.name)
            );
        }
    }

    #[test]
    fn aom_av1_bitrate_uses_kilobits_per_second() {
        gst::init().unwrap();
        let encoder = gst::ElementFactory::make("av1enc").build().unwrap();
        configure_video_encoder(&encoder, "av1enc", VideoQuality::Low, 64, 48, Some(800));
        assert_eq!(encoder.property::<u32>("target-bitrate"), 800);
    }

    #[test]
    fn test_codec_extensions() {
        assert_eq!(VideoCodec::H264.file_extension(), "mp4");
        assert_eq!(VideoCodec::HEVC.file_extension(), "mp4");
        assert_eq!(VideoCodec::AV1.file_extension(), "webm");
    }

    #[test]
    fn test_quality_bitrates() {
        // 1920x1080 (Full HD)
        let low = VideoQuality::Low.bitrate_kbps(1920, 1080);
        let high = VideoQuality::High.bitrate_kbps(1920, 1080);
        assert!(low < high);
        assert!(low >= 500); // Minimum
        assert!(high <= 50000); // Maximum
    }

    #[test]
    fn test_container_formats() {
        assert_eq!(ContainerFormat::MP4.extension(), "mp4");
        assert_eq!(ContainerFormat::WebM.extension(), "webm");
        assert_eq!(ContainerFormat::MP4.muxer_name(), "mp4mux");
        assert_eq!(ContainerFormat::WebM.muxer_name(), "webmmux");
    }

    #[test]
    fn hardware_input_format_prefers_nv12_when_supported() {
        gst::init().unwrap();
        let encoder_caps = gst::Caps::from_str(
            "video/x-raw,format=(string){ I420, NV12 },width=[ 16, 4096 ],height=[ 16, 4096 ]",
        )
        .unwrap();
        let converter_caps =
            gst::Caps::from_str("video/x-raw,format=(string){ NV12, I420 }").unwrap();

        assert_eq!(
            select_raw_encoder_input_format(&encoder_caps, &converter_caps, 1436, 1080, 30),
            Some("NV12".to_string())
        );
    }

    #[test]
    fn hardware_input_format_uses_other_system_memory_formats_generically() {
        gst::init().unwrap();
        let encoder_caps = gst::Caps::from_str("video/x-raw,format=YUY2").unwrap();
        let converter_caps =
            gst::Caps::from_str("video/x-raw,format=(string){ NV12, YUY2 }").unwrap();

        assert_eq!(
            select_raw_encoder_input_format(&encoder_caps, &converter_caps, 1280, 720, 30),
            Some("YUY2".to_string())
        );
    }

    #[test]
    fn hardware_input_format_rejects_unsupported_dimensions() {
        gst::init().unwrap();
        let encoder_caps =
            gst::Caps::from_str("video/x-raw,format=NV12,width=[ 16, 1280 ],height=[ 16, 720 ]")
                .unwrap();
        let converter_caps = gst::Caps::from_str("video/x-raw,format=NV12").unwrap();

        assert_eq!(
            select_raw_encoder_input_format(&encoder_caps, &converter_caps, 1436, 1080, 30),
            None
        );
    }

    #[test]
    fn hardware_input_format_rejects_dmabuf_only_caps_for_cpu_frames() {
        gst::init().unwrap();
        let encoder_caps = gst::Caps::from_str("video/x-raw(memory:DMABuf),format=NV12").unwrap();
        let converter_caps = gst::Caps::from_str("video/x-raw,format=NV12").unwrap();

        assert_eq!(
            select_raw_encoder_input_format(&encoder_caps, &converter_caps, 1280, 720, 30),
            None
        );
    }
}
