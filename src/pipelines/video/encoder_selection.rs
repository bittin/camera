// SPDX-License-Identifier: GPL-3.0-only

//! Encoder selection for video recording pipeline
//!
//! This module provides a simple interface to select video and audio encoders
//! for the recording pipeline.

use crate::media::encoders::{
    audio::{AudioChannels, AudioQuality, SelectedAudioEncoder, select_audio_encoder},
    video::{
        EncoderInfo, SelectedVideoEncoder, VideoQuality, create_encoder_from_info_with_bitrate,
        negotiate_raw_encoder_input_format, select_video_encoder_with_bitrate,
    },
};
use gstreamer as gst;

const OPENH264_MAX_PIXELS: u32 = 9_437_184;
const V4L2_MACROBLOCK_ALIGNMENT: u32 = 16;

/// Raw-video layout accepted by a selected encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoEncoderInput {
    pub width: u32,
    pub height: u32,
    pub format: Option<String>,
}

impl VideoEncoderInput {
    pub fn caps(&self, framerate: u32) -> gst::Caps {
        let mut caps = gst::Caps::builder("video/x-raw")
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", gst::Fraction::new(framerate as i32, 1));
        if let Some(format) = self.format.as_deref() {
            caps = caps.field("format", format);
        }
        caps.build()
    }
}

fn compatible_software_input_format(encoder_name: &str) -> Option<&'static str> {
    match encoder_name {
        "x264enc" | "x265enc" | "openh264enc" => Some("I420"),
        _ => None,
    }
}

fn compatible_output_dimensions(width: u32, height: u32, encoder_name: &str) -> (u32, u32) {
    if encoder_name == "openh264enc" && width.saturating_mul(height) > OPENH264_MAX_PIXELS {
        let aspect_ratio = width as f64 / height as f64;
        let target_width = 1920u32;
        let target_height = ((target_width as f64 / aspect_ratio) as u32 & !1).max(2);
        return (target_width, target_height);
    }

    if !encoder_name.starts_with("v4l2") {
        return (width, height);
    }

    let align_up = |value: u32| {
        (value.saturating_add(V4L2_MACROBLOCK_ALIGNMENT - 1) / V4L2_MACROBLOCK_ALIGNMENT
            * V4L2_MACROBLOCK_ALIGNMENT)
            .max(V4L2_MACROBLOCK_ALIGNMENT)
    };
    (align_up(width), align_up(height))
}

/// Resolve dimensions and raw format constraints shared by recording and timelapse.
pub fn prepare_video_encoder_input(
    encoder: &gst::Element,
    encoder_name: &str,
    width: u32,
    height: u32,
    framerate: u32,
) -> Result<VideoEncoderInput, String> {
    let (width, height) = compatible_output_dimensions(width, height, encoder_name);
    let format = if encoder_name.starts_with("v4l2") {
        Some(
            negotiate_raw_encoder_input_format(encoder, width, height, framerate).ok_or_else(
                || {
                    format!(
                        "{encoder_name} has no compatible CPU-memory input for {width}x{height}@{framerate}"
                    )
                },
            )?,
        )
    } else {
        compatible_software_input_format(encoder_name).map(str::to_string)
    };

    Ok(VideoEncoderInput {
        width,
        height,
        format,
    })
}

/// Configuration for encoder selection
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Video quality preset
    pub video_quality: VideoQuality,
    /// Audio quality preset
    pub audio_quality: AudioQuality,
    /// Audio channel configuration
    pub audio_channels: AudioChannels,
    /// Video width (for bitrate calculation)
    pub width: u32,
    /// Video height (for bitrate calculation)
    pub height: u32,
    /// Optional bitrate override in kbps (takes precedence over quality preset)
    pub bitrate_override_kbps: Option<u32>,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            video_quality: VideoQuality::High,
            audio_quality: AudioQuality::High,
            audio_channels: AudioChannels::Mono,
            width: 1920,
            height: 1080,
            bitrate_override_kbps: None,
        }
    }
}

/// Selected encoders for recording
pub struct SelectedEncoders {
    /// Video encoder configuration
    pub video: SelectedVideoEncoder,
    /// Audio encoder configuration (optional if no audio)
    pub audio: Option<SelectedAudioEncoder>,
}

/// Select an explicit video encoder or the best available one, plus optional audio.
pub fn select_encoders(
    config: &EncoderConfig,
    encoder_info: Option<&EncoderInfo>,
    enable_audio: bool,
) -> Result<SelectedEncoders, String> {
    let video = match encoder_info {
        Some(info) => create_encoder_from_info_with_bitrate(
            info,
            config.video_quality,
            config.width,
            config.height,
            config.bitrate_override_kbps,
        ),
        None => select_video_encoder_with_bitrate(
            config.video_quality,
            config.width,
            config.height,
            config.bitrate_override_kbps,
        ),
    }?;

    // Select audio encoder if enabled
    let audio = if enable_audio {
        match select_audio_encoder(config.audio_quality, config.audio_channels) {
            Ok(encoder) => Some(encoder),
            Err(e) => {
                tracing::warn!(
                    "Failed to select audio encoder: {}. Recording without audio.",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    Ok(SelectedEncoders { video, audio })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::encoders::video::VideoCodec;
    use gstreamer::prelude::*;
    use std::str::FromStr;

    #[test]
    fn explicit_encoder_preserves_bitrate_parser_muxer_and_audio_policy() {
        gst::init().unwrap();
        let info = EncoderInfo {
            element_name: "openh264enc".into(),
            display_name: "OpenH264".into(),
            codec: VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let config = EncoderConfig {
            bitrate_override_kbps: Some(800),
            ..Default::default()
        };
        for enable_audio in [false, true] {
            let selected = select_encoders(&config, Some(&info), enable_audio).unwrap();
            assert_eq!(
                selected.video.encoder.factory().unwrap().name(),
                "openh264enc"
            );
            assert_eq!(selected.video.encoder.property::<u32>("bitrate"), 800_000);
            assert_eq!(
                selected.video.parser.unwrap().factory().unwrap().name(),
                "h264parse"
            );
            assert_eq!(selected.video.muxer.factory().unwrap().name(), "mp4mux");
            assert_eq!(selected.video.extension, "mp4");
            assert_eq!(selected.audio.is_some(), enable_audio);
        }
    }

    #[test]
    fn unavailable_explicit_encoder_is_not_silently_substituted() {
        gst::init().unwrap();
        let info = EncoderInfo {
            element_name: "camera-nonexistent-encoder".into(),
            display_name: "Missing".into(),
            codec: VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        };
        let error = select_encoders(&EncoderConfig::default(), Some(&info), false)
            .err()
            .unwrap();
        assert!(error.contains(&info.element_name), "{error}");
    }

    #[test]
    fn automatic_selection_assembles_a_video_without_audio() {
        gst::init().unwrap();
        let selected = select_encoders(&EncoderConfig::default(), None, false).unwrap();
        assert!(selected.audio.is_none());
        assert_eq!(
            selected.video.extension,
            selected.video.codec.file_extension()
        );
        assert_eq!(
            selected.video.muxer.factory().unwrap().name(),
            selected.video.codec.container_format().muxer_name()
        );
    }

    #[test]
    fn v4l2_input_preparation_aligns_dimensions_and_negotiates_format() {
        gstreamer::init().unwrap();
        let caps = gstreamer::Caps::from_str(
            "video/x-raw,format=NV12,width=[ 16, 4096 ],height=[ 16, 4096 ]",
        )
        .unwrap();
        let encoder = gstreamer::ElementFactory::make("capsfilter")
            .property("caps", caps)
            .build()
            .unwrap();

        let input = prepare_video_encoder_input(&encoder, "v4l2h265enc", 1436, 1080, 30).unwrap();

        assert_eq!((input.width, input.height), (1440, 1088));
        assert_eq!(input.format.as_deref(), Some("NV12"));
    }
}
