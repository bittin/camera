// SPDX-License-Identifier: GPL-3.0-only

//! GStreamer encoder detection
//!
//! This module provides functionality to detect available video and audio
//! encoders in the GStreamer installation.

use gstreamer as gst;
use gstreamer::prelude::*;
use tracing::{debug, info, warn};

/// Check if a specific GStreamer element is available
pub fn is_element_available(element_name: &str) -> bool {
    gst::init().ok();
    gst::ElementFactory::find(element_name).is_some()
}

/// Detect all available audio encoders
///
/// Returns a list of available encoder names in no particular order.
pub fn detect_audio_encoders() -> Vec<String> {
    gst::init().ok();

    let encoders = [
        // Opus (preferred)
        "opusenc",
        // AAC (fallback)
        "avenc_aac",
        "faac",
        "voaacenc",
    ];

    let mut available = Vec::new();

    for encoder in &encoders {
        if is_element_available(encoder) {
            debug!("Audio encoder available: {}", encoder);
            available.push(encoder.to_string());
        }
    }

    info!("Detected {} audio encoders", available.len());
    available
}

/// Timeout for GStreamer probe pipelines (EOS or Error).
const PROBE_TIMEOUT_SECS: u64 = 5;

/// Run a GStreamer pipeline description to completion, returning `true` if it
/// reaches EOS within [`PROBE_TIMEOUT_SECS`] and `false` on error or timeout.
fn probe_gst_pipeline(pipeline_desc: &str) -> bool {
    let _ = gst::init();

    let pipeline = match gst::parse::launch(pipeline_desc) {
        Ok(p) => p,
        Err(_) => return false,
    };

    let bus = match pipeline.bus() {
        Some(b) => b,
        None => return false,
    };

    if pipeline.set_state(gst::State::Playing).is_err() {
        let _ = pipeline.set_state(gst::State::Null);
        return false;
    }

    let ok = loop {
        match bus.timed_pop(gst::ClockTime::from_seconds(PROBE_TIMEOUT_SECS)) {
            Some(msg) => match msg.view() {
                gst::MessageView::Eos(_) => break true,
                gst::MessageView::Error(_) => break false,
                _ => continue,
            },
            None => break false,
        }
    };

    let _ = pipeline.set_state(gst::State::Null);
    ok
}

/// Test whether a video encoder actually works by building a short pipeline
/// with `videotestsrc` and checking for negotiation or encoding errors.
///
/// Returns `true` if the encoder produced valid output, `false` if it failed.
pub fn probe_single_encoder(encoder_name: &str) -> bool {
    // Determine the codec's parser from the encoder name
    let parser = if encoder_name.contains("h265") || encoder_name.contains("hevc") {
        "h265parse"
    } else if encoder_name.contains("h264")
        || encoder_name.contains("openh264")
        || encoder_name.contains("x264")
    {
        "h264parse"
    } else if encoder_name.contains("av1") {
        "av1parse"
    } else {
        // Unknown codec, skip probing
        return true;
    };

    let pipeline_desc = format!(
        "videotestsrc num-buffers=3 ! video/x-raw,format=NV12,width=320,height=240,framerate=10/1 \
         ! videoconvert ! {encoder} ! {parser} ! fakesink",
        encoder = encoder_name,
        parser = parser,
    );
    probe_gst_pipeline(&pipeline_desc)
}

fn defer_probe_until_recording(encoder_name: &str) -> bool {
    encoder_name.starts_with("v4l2")
}

/// Probe all given video encoder names and return the names of the broken ones.
///
/// Hardware encoders are tested sequentially (they may share a single V4L2 or
/// VA-API device that doesn't allow concurrent access). Software encoders are
/// tested in parallel for speed.
pub fn probe_broken_encoders(encoder_names: &[String]) -> Vec<String> {
    let _ = gst::init();

    let mut broken = Vec::new();

    // V4L2 mem2mem encoders can share resources with an active camera. Keep
    // them available here and negotiate their exact sink caps only when the
    // user starts recording instead of running a competing test pipeline.
    for name in encoder_names
        .iter()
        .filter(|name| defer_probe_until_recording(name))
    {
        info!(encoder = %name, "Deferring V4L2 encoder probe until recording");
    }

    let (hw, sw): (Vec<&String>, Vec<&String>) = encoder_names
        .iter()
        .filter(|name| !defer_probe_until_recording(name))
        .partition(|name| {
            name.starts_with("va")
                || name.starts_with("nv")
                || name.starts_with("qsv")
                || name.starts_with("amf")
        });

    // Test hardware encoders sequentially
    for name in &hw {
        if !probe_single_encoder(name) {
            warn!(encoder = %name, "Hardware encoder failed probe — marking as broken");
            broken.push(name.to_string());
        } else {
            info!(encoder = %name, "Hardware encoder probe OK");
        }
    }

    // Test software encoders in parallel
    let sw_results: Vec<(String, bool)> = std::thread::scope(|s| {
        let handles: Vec<_> = sw
            .iter()
            .map(|name| {
                let name = name.to_string();
                s.spawn(move || {
                    let ok = probe_single_encoder(&name);
                    (name, ok)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for (name, ok) in sw_results {
        if !ok {
            warn!(encoder = %name, "Software encoder failed probe — marking as broken");
            broken.push(name);
        } else {
            info!(encoder = %name, "Software encoder probe OK");
        }
    }

    broken
}

/// Log all available encoders (for debugging)
pub fn log_available_encoders() {
    info!("=== GStreamer Encoder Detection ===");

    info!("Video encoders:");
    for encoder in super::video::enumerate_video_encoders() {
        info!("  ✓ {}", encoder.element_name);
    }

    info!("Audio encoders:");
    for encoder in detect_audio_encoders() {
        info!("  ✓ {}", encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4l2_encoder_probe_is_deferred_without_marking_it_broken() {
        assert!(defer_probe_until_recording("v4l2h264enc"));
        assert!(defer_probe_until_recording("v4l2h265enc"));
        assert!(!defer_probe_until_recording("x264enc"));
        assert!(!defer_probe_until_recording("vah264enc"));
    }
}
