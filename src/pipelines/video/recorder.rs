// SPDX-License-Identifier: GPL-3.0-only

//! Video recording pipeline with intelligent encoder selection
//!
//! This module implements video recording with:
//! - Automatic hardware encoder detection and selection
//! - Preview continues during recording (tee-based pipeline)
//! - Audio integration
//! - Quality presets

use super::encoder_selection::{EncoderConfig, select_encoders};
use super::muxer::link_audio_to_muxer;
use super::stats::{
    RECORDING_STATS, RecordingDiagnostics, clear_recording_diagnostics,
    publish_recording_diagnostics,
};
use crate::backends::camera::types::{CameraFrame, PixelFormat, RecordingFrame};
use crate::media::encoders::video::SelectedVideoEncoder;
use crate::pipelines::audio_level::install_level_sync_handler as install_shared_level_sync_handler;
use crate::pipelines::audio_level::{PULSESRC_BUFFER_TIME_US, PULSESRC_SLAVE_METHOD};
use chrono::{Datelike, Timelike, Utc};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// How often to emit periodic progress log messages (every Nth frame).
const LOG_EVERY_N_FRAMES: u64 = 60;

/// Maximum number of frames retained by appsrc while the encoder is slower
/// than capture. Live recording must prefer a recent frame over unbounded
/// latency and memory growth.
const APPSRC_MAX_BUFFERS: u64 = 3;

pub use crate::pipelines::audio_level::{AudioLevels, SharedAudioLevels};

/// Common recording configuration.
pub struct RecorderConfig<'a> {
    /// Video width
    pub width: u32,
    /// Video height
    pub height: u32,
    /// Video framerate
    pub framerate: u32,
    /// Output file path
    pub output_path: PathBuf,
    /// Encoder configuration
    pub encoder_config: EncoderConfig,
    /// Whether to record audio
    pub enable_audio: bool,
    /// Optional audio device path
    pub audio_device: Option<&'a str>,
    /// Native sample rate (Hz) of the selected audio source. Used to pin the
    /// audio capsfilter so PulseAudio passes through unchanged when Opus
    /// accepts the rate, and avoids a GStreamer `audioresample` element.
    /// `0` means "unknown" and falls back to 48 kHz.
    pub audio_source_rate_hz: u32,
    /// Specific encoder info (if None, auto-select)
    pub encoder_info: Option<&'a crate::media::encoders::video::EncoderInfo>,
    /// Immutable metadata captured at recording start.
    pub capture_metadata: crate::pipelines::capture_metadata::CaptureMetadata,
    /// Pre-created shared audio levels handle (UI reads this for live meters)
    pub audio_levels: SharedAudioLevels,
}

/// Appsrc-specific recording configuration (libcamera backend).
///
/// Frames are pushed from the application via a `tokio::sync::mpsc` channel
/// instead of using `pipewiresrc`. This avoids camera contention when the
/// native libcamera pipeline already holds the device.
pub struct AppsrcRecorderConfig<'a> {
    /// Common recording settings
    pub base: RecorderConfig<'a>,
    /// Pixel format of incoming frames
    pub pixel_format: crate::backends::camera::types::PixelFormat,
    /// Live filter code (read each frame via AtomicU32, updated by UI thread).
    /// Value is `FilterType::gpu_filter_code()`. 0 = Standard (no filter).
    pub live_filter_code: Arc<std::sync::atomic::AtomicU32>,
}

/// Video recorder using the new pipeline architecture
#[derive(Debug)]
pub struct VideoRecorder {
    pipeline: gst::Pipeline,
    file_path: PathBuf,
    /// Lifetime-tied PA source-volume restore. Constructed when a non-default
    /// audio device is configured; dropping (here or when `VideoRecorder`
    /// drops) restores the user's prior PA volume. `None` for default-device
    /// recordings or when the PulseAudio socket is unreachable.
    _pulse_volume_guard: Option<crate::backends::audio::PulseSourceVolumeGuard>,
    /// Handle to the appsrc pusher task. Stored so `stop()` / `Drop` can abort
    /// it before transitioning the pipeline to NULL, avoiding races where the
    /// detached task pushes into a finalising pipeline.
    pusher_handle: Option<tokio::task::JoinHandle<()>>,
}

pub(super) fn compatible_software_input_format(encoder_name: &str) -> Option<&'static str> {
    match encoder_name {
        "x264enc" | "x265enc" | "openh264enc" => Some("I420"),
        _ => None,
    }
}

fn encoder_input_caps_filter(encoder_name: &str) -> String {
    compatible_software_input_format(encoder_name)
        .map(|format| format!("! capsfilter caps=video/x-raw,format={format}"))
        .unwrap_or_default()
}

fn decoded_video_processing_chain_for_encoder(
    width: u32,
    height: u32,
    encoder_name: &str,
    needs_scaling: bool,
) -> String {
    let mut chain = if needs_scaling {
        format!(
            "! videoconvert ! videoscale \
             ! capsfilter caps=video/x-raw,format=I420,width={width},height={height} \
             ! videoconvert"
        )
    } else {
        "! videoconvert".to_string()
    };

    let compatibility_caps = encoder_input_caps_filter(encoder_name);
    if !compatibility_caps.is_empty() {
        chain.push(' ');
        chain.push_str(&compatibility_caps);
    }
    chain
}

fn video_tags(metadata: &crate::pipelines::capture_metadata::CaptureMetadata) -> gst::TagList {
    let mut tags = gst::TagList::new();
    let tags = tags.get_mut().expect("new tag list must be writable");
    let replace = gst::TagMergeMode::ReplaceAll;
    let application_name = crate::pipelines::capture_metadata::CaptureMetadata::application_name();

    tags.add::<gst::tags::ApplicationName>(&application_name.as_str(), replace);
    if let Some(version) = metadata.libcamera_version.as_deref() {
        let encoder = format!("libcamera {version}");
        tags.add::<gst::tags::Encoder>(&encoder.as_str(), replace);
    }
    tags.add::<gst::tags::ImageOrientation>(&metadata.orientation.gstreamer_tag(), replace);

    if let Some(make) = metadata.device_make.as_deref() {
        tags.add::<gst::tags::DeviceManufacturer>(&make, replace);
    }
    if let Some(model) = metadata.device_model.as_deref() {
        tags.add::<gst::tags::DeviceModel>(&model, replace);
    }
    if let Some(description) = metadata.description() {
        tags.add::<gst::tags::Description>(&description.as_str(), replace);
    }
    if let Some(captured_at) = metadata.captured_at.as_ref() {
        let captured_at = captured_at.with_timezone(&Utc);
        let seconds =
            captured_at.second() as f64 + f64::from(captured_at.nanosecond()) / 1_000_000_000.0;
        if let Ok(date_time) = gst::DateTime::new(
            0.0f32,
            captured_at.year(),
            captured_at.month() as i32,
            captured_at.day() as i32,
            captured_at.hour() as i32,
            captured_at.minute() as i32,
            seconds,
        ) {
            tags.add::<gst::tags::DateTime>(&date_time, replace);
        }
    }

    tags.to_owned()
}

fn apply_video_tags(
    pipeline: &gst::Pipeline,
    metadata: &crate::pipelines::capture_metadata::CaptureMetadata,
) {
    let Some(muxer) = pipeline.by_name("recording-muxer") else {
        warn!("Recording muxer not found; video metadata was not applied");
        return;
    };
    apply_video_tags_to_muxer(&muxer, metadata);
}

pub(super) fn apply_video_tags_to_muxer(
    muxer: &gst::Element,
    metadata: &crate::pipelines::capture_metadata::CaptureMetadata,
) {
    let Ok(tag_setter) = muxer.clone().dynamic_cast::<gst::TagSetter>() else {
        warn!("Recording muxer does not implement TagSetter; video metadata was not applied");
        return;
    };

    tag_setter.merge_tags(&video_tags(metadata), gst::TagMergeMode::ReplaceAll);
}

/// OpenH264 maximum pixel count (roughly 3072x3072).
const OPENH264_MAX_PIXELS: u32 = 9_437_184;

/// Build the optional PA-source-volume guard for a recording. Returns `None`
/// when audio is disabled, no device was picked (PA default-source path), or
/// PA can't be reached. Centralised so both recorder entry points
/// (`new_from_appsrc`, `new_from_appsrc_jpeg`) keep their guard semantics in
/// sync.
fn build_pulse_volume_guard(
    enable_audio: bool,
    audio_device: Option<&str>,
) -> Option<crate::backends::audio::PulseSourceVolumeGuard> {
    if !enable_audio {
        return None;
    }
    audio_device.and_then(crate::backends::audio::PulseSourceVolumeGuard::boost_to_full)
}

/// Downscale dimensions if they exceed OpenH264's pixel limit.
/// Returns the original dimensions if the encoder is not OpenH264 or the limit is not exceeded.
fn openh264_downscale(base_width: u32, base_height: u32, encoder_name: &str) -> (u32, u32) {
    let pixels = base_width * base_height;
    if encoder_name == "openh264enc" && pixels > OPENH264_MAX_PIXELS {
        let aspect_ratio = base_width as f64 / base_height as f64;
        let target_width = 1920u32;
        // Even height, and at least 2 pixels — `& !1` on small values can
        // round down to 0, which would cause GStreamer caps negotiation to fail.
        let target_height = ((target_width as f64 / aspect_ratio) as u32 & !1).max(2);
        warn!(
            "OpenH264 resolution limit exceeded ({}x{} = {} pixels > {} max), downscaling to {}x{}",
            base_width, base_height, pixels, OPENH264_MAX_PIXELS, target_width, target_height,
        );
        (target_width, target_height)
    } else {
        (base_width, base_height)
    }
}

/// Select encoder set: use a specific encoder if provided, otherwise auto-select.
fn select_encoder_set(
    encoder_info: Option<&crate::media::encoders::video::EncoderInfo>,
    encoder_config: &EncoderConfig,
    enable_audio: bool,
) -> Result<super::encoder_selection::SelectedEncoders, String> {
    if let Some(enc_info) = encoder_info {
        super::encoder_selection::select_encoders_with_video(encoder_config, enc_info, enable_audio)
    } else {
        select_encoders(encoder_config, enable_audio)
    }
}

/// Shared state from the common recorder preparation phase.
///
/// Both `new_from_appsrc` and `new_from_appsrc_jpeg` begin with the same
/// sequence: encoder selection, V4L2 fallback, audio branch creation, and
/// output path resolution. This struct captures the results so each
/// constructor only handles its format-specific pipeline description and
/// pusher spawn.
struct RecorderSetup {
    audio_elements: Option<AudioBranch>,
    encoder_name: String,
    parser_str: String,
    muxer_name: String,
    output_path: PathBuf,
    frame_duration_ns: i64,
}

/// Common setup for both appsrc recorder constructors.
///
/// Handles encoder selection, V4L2 fallback, audio branch creation,
/// and output path resolution. Format-specific encoder overrides
/// (e.g. NVIDIA domain matching) should be applied to the returned
/// [`RecorderSetup`] before building the pipeline description.
fn prepare_recorder(
    encoder_info: Option<&crate::media::encoders::video::EncoderInfo>,
    encoder_config: &EncoderConfig,
    enable_audio: bool,
    audio_device: Option<&str>,
    audio_source_rate_hz: u32,
    output_path: PathBuf,
    framerate: u32,
) -> Result<RecorderSetup, String> {
    let encoders = select_encoder_set(encoder_info, encoder_config, enable_audio)?;

    let audio_elements = if let Some(audio_encoder_config) = encoders.audio {
        VideoRecorder::create_audio_branch(
            audio_device,
            audio_source_rate_hz,
            audio_encoder_config,
        )?
    } else {
        None
    };

    info!(
        video_codec = ?encoders.video.codec,
        audio = audio_elements.is_some(),
        container = ?encoders.video.container,
        "Selected encoders"
    );

    let output_path = output_path.with_extension(encoders.video.extension);
    let frame_duration_ns = 1_000_000_000i64 / framerate as i64;

    let selected_encoder = encoders
        .video
        .encoder
        .factory()
        .map(|f| f.name().to_string())
        .unwrap_or_else(|| "openh264enc".to_string());

    let (encoder_name, parser_str, muxer_name) = if selected_encoder.starts_with("v4l2") {
        warn!(
            selected = %selected_encoder,
            "V4L2 encoder not compatible with appsrc pipeline, falling back to openh264enc"
        );
        (
            "openh264enc".to_string(),
            "! h264parse".to_string(),
            "mp4mux".to_string(),
        )
    } else {
        let (parser, muxer) = parser_and_muxer_names(&encoders.video);
        // Probe hardware encoders to catch cases where the element exists in
        // the registry but can't actually encode (e.g. VA-API backed by NVENC
        // in a flatpak sandbox that lacks libnvidia-encode.so).
        let is_software = selected_encoder == "openh264enc"
            || selected_encoder == "x264enc"
            || selected_encoder == "x265enc";
        if !is_software
            && !crate::media::encoders::detection::probe_single_encoder(&selected_encoder)
        {
            warn!(
                selected = %selected_encoder,
                "Hardware encoder probe failed, falling back to openh264enc"
            );
            (
                "openh264enc".to_string(),
                "! h264parse".to_string(),
                "mp4mux".to_string(),
            )
        } else {
            (selected_encoder, parser, muxer)
        }
    };

    Ok(RecorderSetup {
        audio_elements,
        encoder_name,
        parser_str,
        muxer_name,
        output_path,
        frame_duration_ns,
    })
}

/// Extract parser name (with `! ` prefix) and muxer name from a selected video encoder.
fn parser_and_muxer_names(video: &SelectedVideoEncoder) -> (String, String) {
    let parser = video
        .parser
        .as_ref()
        .and_then(|p| p.factory().map(|f| format!("! {}", f.name())))
        .unwrap_or_default();
    let muxer = video
        .muxer
        .factory()
        .map(|f| f.name().to_string())
        .unwrap_or_else(|| "mp4mux".to_string());
    (parser, muxer)
}

/// Read `CLOCK_BOOTTIME` in nanoseconds (same clock domain as libcamera
/// sensor timestamps).
fn read_clock_boottime_ns() -> u64 {
    use std::mem::MaybeUninit;
    unsafe {
        let mut ts = MaybeUninit::<libc::timespec>::uninit();
        if libc::clock_gettime(libc::CLOCK_BOOTTIME, ts.as_mut_ptr()) != 0 {
            return 0;
        }
        let ts = ts.assume_init();
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
}

/// Add audio branch elements to the pipeline, link the chain, connect to
/// the muxer, and install the level sync handler.
fn add_audio_branch_to_pipeline(
    pipeline: &gst::Pipeline,
    audio_branch: &AudioBranch,
    audio_levels: &SharedAudioLevels,
) -> Result<(), String> {
    pipeline
        .add_many([
            &audio_branch.source,
            &audio_branch.queue,
            &audio_branch.convert,
            &audio_branch.resample,
            &audio_branch.capsfilter,
            &audio_branch.compressor,
            &audio_branch.makeup_gain,
            &audio_branch.limiter,
            &audio_branch.level,
            &audio_branch.encoder,
        ])
        .map_err(|e| format!("Failed to add audio elements to pipeline: {}", e))?;

    VideoRecorder::link_audio_chain(audio_branch)?;

    let muxer = pipeline
        .by_name("recording-muxer")
        .ok_or("Failed to find recording-muxer for audio linking")?;
    link_audio_to_muxer(&audio_branch.encoder, &muxer)?;

    install_shared_level_sync_handler(pipeline, audio_levels);

    Ok(())
}

/// Bound the appsrc queue used by live recording.
///
/// `push_buffer` does not automatically stop accepting data after appsrc emits
/// `enough-data`. Without an explicit leaky limit, a slow encoder can therefore
/// retain every full-resolution frame until the process is killed by the OOM
/// killer. Dropping the oldest queued frame keeps latency and memory bounded.
fn configure_live_appsrc(appsrc: &gst_app::AppSrc) {
    appsrc.set_max_buffers(APPSRC_MAX_BUFFERS);
    appsrc.set_max_bytes(0);
    appsrc.set_block(false);
    appsrc.set_leaky_type(gst_app::AppLeakyType::Downstream);
}

/// Return whether appsrc has reached its configured live-frame limit.
///
/// Checking before format conversion avoids spending CPU/GPU time on a frame
/// that cannot enter the encoding pipeline yet. The next captured frame gets
/// another chance as soon as downstream frees a slot.
fn appsrc_queue_is_full(appsrc: &gst_app::AppSrc) -> bool {
    queue_at_capacity(appsrc.current_level_buffers(), appsrc.max_buffers())
}

fn queue_at_capacity(current_buffers: u64, max_buffers: u64) -> bool {
    max_buffers > 0 && current_buffers >= max_buffers
}

fn fallback_pts(capture_index: u64, frame_duration_ns: u64) -> u64 {
    capture_index * frame_duration_ns
}

/// Parse a GStreamer pipeline description and perform common configuration.
///
/// Returns the pipeline and appsrc element after:
/// - Parsing the pipeline description
/// - Extracting the `camera-appsrc` element
/// - Configuring the video encoder (bitrate / quality)
/// - Adding the audio branch (if present)
/// - Installing muxer fixup probes
fn build_recorder_pipeline(
    pipeline_desc: &str,
    encoder_name: &str,
    encoder_config: &EncoderConfig,
    encode_width: u32,
    encode_height: u32,
    audio_elements: Option<&AudioBranch>,
    audio_levels: &SharedAudioLevels,
) -> Result<(gst::Pipeline, gst_app::AppSrc), String> {
    let pipeline = gst::parse::launch(pipeline_desc)
        .map_err(|e| format!("Failed to parse pipeline: {}", e))?
        .dynamic_cast::<gst::Pipeline>()
        .map_err(|_| "Failed to cast to Pipeline")?;

    let appsrc = pipeline
        .by_name("camera-appsrc")
        .ok_or("Failed to find camera-appsrc in pipeline")?
        .dynamic_cast::<gst_app::AppSrc>()
        .map_err(|_| "Failed to cast to AppSrc")?;
    configure_live_appsrc(&appsrc);

    if let Some(enc_element) = pipeline.by_name("recording-encoder") {
        crate::media::encoders::video::configure_video_encoder(
            &enc_element,
            encoder_name,
            encoder_config.video_quality,
            encode_width,
            encode_height,
            encoder_config.bitrate_override_kbps,
        );
    }

    if let Some(audio_branch) = audio_elements {
        add_audio_branch_to_pipeline(&pipeline, audio_branch, audio_levels)?;
        info!("Audio branch added to recording pipeline");
    }

    install_muxer_fixup_probes(&pipeline);

    Ok((pipeline, appsrc))
}

/// Result of PTS computation for a single frame.
enum PtsResult {
    /// Computed PTS in nanoseconds — push this buffer.
    Pts(u64),
    /// Frame should be skipped (pipeline not yet PLAYING).
    Skip,
}

/// Compute PTS for a recording frame using sensor timestamps and pipeline
/// running-time for A/V sync. Falls back to frame-count-based PTS if
/// sensor timestamps are unavailable.
fn compute_pts(
    appsrc: &gst_app::AppSrc,
    sensor_ts: Option<u64>,
    capture_index: u64,
    frame_duration_ns: u64,
    pipeline_playing: &mut bool,
    ts_offset: &mut Option<(u64, u64)>,
) -> PtsResult {
    let Some(ts) = sensor_ts else {
        return PtsResult::Pts(fallback_pts(capture_index, frame_duration_ns));
    };

    // Skip frames until pipeline is PLAYING.
    if !*pipeline_playing {
        if appsrc.current_running_time().is_none() {
            RECORDING_STATS
                .pusher_skipped
                .fetch_add(1, Ordering::Relaxed);
            return PtsResult::Skip;
        }
        *pipeline_playing = true;
        info!("Pipeline is PLAYING, starting video capture");
    }
    let rt = match appsrc.current_running_time() {
        Some(t) => t.nseconds(),
        None => {
            RECORDING_STATS
                .pusher_skipped
                .fetch_add(1, Ordering::Relaxed);
            return PtsResult::Skip;
        }
    };

    // Record processing delay for diagnostics
    let now_boot = read_clock_boottime_ns();
    let processing_delay = now_boot.saturating_sub(ts);
    RECORDING_STATS
        .last_processing_delay_us
        .store(processing_delay / 1_000, Ordering::Relaxed);

    // On first frame, establish PTS base accounting for processing
    // delay so video timestamps reflect actual capture time.
    let is_first = ts_offset.is_none();
    let (first_ts, pts_base) = *ts_offset.get_or_insert((ts, rt.saturating_sub(processing_delay)));
    let pts = pts_base + ts.saturating_sub(first_ts);
    if is_first {
        info!(
            running_time_ms = rt / 1_000_000,
            processing_delay_ms = processing_delay / 1_000_000,
            pts_base_ms = pts_base / 1_000_000,
            pts_ms = pts / 1_000_000,
            sensor_ts_ms = ts / 1_000_000,
            "First video frame A/V sync: pts_base = running_time - processing_delay"
        );
    }
    PtsResult::Pts(pts)
}

/// Install a read-only PTS/DTS trace probe on a named element's src pad.
///
/// Logs timestamps at `debug!()` level for the first 5 frames and every
/// [`LOG_EVERY_N_FRAMES`] frames thereafter.
fn install_pts_trace_probe(element: &gst::Element, stage: &'static str) {
    let Some(src_pad) = element.static_pad("src") else {
        return;
    };
    let frame_count = std::sync::Arc::new(AtomicU64::new(0));
    let fc = frame_count.clone();
    src_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        let n = fc.fetch_add(1, Ordering::Relaxed);
        if (n < 5 || n.is_multiple_of(LOG_EVERY_N_FRAMES))
            && let Some(buffer) = info.buffer()
        {
            debug!(
                frame = n,
                pts_ms = buffer.pts().map(|p| p.mseconds()),
                dts_ms = buffer.dts().map(|d| d.mseconds()),
                stage,
                "PTS trace"
            );
        }
        gst::PadProbeReturn::Ok
    });
}

/// Install muxer sink pad probes that fix PTS=NONE (copies DTS → PTS).
///
/// NVENC encoders (nvh265enc/nvh264enc) add a 3 600 000 s offset to PTS/DTS
/// **and** to the segment event.  The aggregator-based mp4mux in GStreamer 1.28
/// converts PTS to running-time via `PTS − segment.start`, so the offset
/// cancels out.  Stripping the offset from buffers without also adjusting the
/// segment causes the muxer to clip every video buffer as "outside segment",
/// resulting in 0 video samples in the output file.
fn install_muxer_fixup_probes(pipeline: &gst::Pipeline) {
    let Some(muxer) = pipeline.by_name("recording-muxer") else {
        return;
    };
    for pad in muxer.sink_pads() {
        let pad_name = pad.name().to_string();
        let mux_probe_count = std::sync::Arc::new(AtomicU64::new(0));
        let mpc = mux_probe_count.clone();
        pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
            if let Some(buffer) = info.buffer_mut() {
                let buf = buffer.make_mut();
                // Fix PTS=NONE (some encoders set only DTS)
                if buf.pts().is_none()
                    && let Some(dts) = buf.dts()
                {
                    buf.set_pts(dts);
                }
            }
            let n = mpc.fetch_add(1, Ordering::Relaxed);
            if n < 3
                && let Some(buffer) = info.buffer()
            {
                debug!(
                    pad = pad_name.as_str(),
                    frame = n,
                    pts_ms = buffer.pts().map(|p| p.mseconds()),
                    dts_ms = buffer.dts().map(|d| d.mseconds()),
                    "Muxer sink pad buffer"
                );
            }
            gst::PadProbeReturn::Ok
        });
    }
}

/// Wrap tightly-packed RGBA frame storage directly for Standard-filter recording.
///
/// The GStreamer buffer owns a clone of [`FrameData`], so the underlying shared
/// allocation stays alive until downstream releases the buffer. Filtered frames,
/// padded RGBA, and non-RGBA formats return `None` and use the processing path.
fn try_shared_rgba_buffer(
    frame: &CameraFrame,
    filter_type: crate::app::FilterType,
) -> Option<gst::Buffer> {
    let row_bytes = frame.width.checked_mul(4)?;
    let expected_len = row_bytes.checked_mul(frame.height)? as usize;

    (filter_type == crate::app::FilterType::Standard
        && frame.format == PixelFormat::RGBA
        && frame.stride == row_bytes
        && frame.data.len() == expected_len)
        .then(|| gst::Buffer::from_slice(frame.data.clone()))
}

/// Convert a camera frame to tightly-packed RGBA using the GPU compute shader.
///
/// For frames already in RGBA format, strips stride padding.
/// For YUV and other formats, uses the GPU compute pipeline.
pub(crate) async fn convert_frame_to_rgba(frame: &CameraFrame) -> Result<Vec<u8>, String> {
    if frame.format == PixelFormat::RGBA {
        let row_bytes = (frame.width * 4) as usize;
        let stride = frame.stride as usize;
        if stride <= row_bytes {
            return Ok(frame.data.to_vec());
        }
        let mut out = Vec::with_capacity(row_bytes * frame.height as usize);
        for y in 0..frame.height as usize {
            out.extend_from_slice(&frame.data[y * stride..y * stride + row_bytes]);
        }
        return Ok(out);
    }

    let input = crate::shaders::GpuFrameInput::from_camera_frame(frame)?;

    let mut pipeline_guard = crate::shaders::get_gpu_convert_pipeline()
        .await
        .map_err(|e| format!("Failed to get GPU convert pipeline: {}", e))?;

    let pipeline = pipeline_guard
        .as_mut()
        .ok_or("GPU convert pipeline not initialized")?;

    pipeline
        .convert(&input)
        .map_err(|e| format!("GPU conversion failed: {}", e))?;

    pipeline
        .read_rgba_to_cpu(frame.width, frame.height)
        .await
        .map_err(|e| format!("Failed to read RGBA from GPU: {}", e))
}

/// Frame data prepared by a format-specific closure for the common pusher loop.
struct PusherFrame {
    buffer: gst::Buffer,
    sensor_ts: Option<u64>,
    sequence: Option<u32>,
}

/// Spawn a tokio task that reads `RecordingFrame`s from a channel, prepares
/// them via `prepare_frame`, and pushes them into the GStreamer `appsrc`.
///
/// The `prepare_frame` closure extracts format-specific data from each
/// `RecordingFrame` and creates a `gst::Buffer`. Return `None` to skip a
/// frame (e.g. wrong variant). The common loop handles PTS computation,
/// buffer timestamping, stats updates, periodic logging, and EOS teardown.
fn spawn_pusher<F>(
    appsrc: gst_app::AppSrc,
    mut frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
    framerate: u32,
    label: &'static str,
    mut prepare_frame: F,
) -> tokio::task::JoinHandle<()>
where
    F: FnMut(RecordingFrame, &gst_app::AppSrc) -> Option<PusherFrame> + Send + 'static,
{
    tokio::spawn(async move {
        info!(label, "Appsrc pusher task started");

        let start_epoch_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        RECORDING_STATS
            .pusher_start_epoch_ns
            .store(start_epoch_ns, Ordering::Relaxed);

        let mut frame_count: u64 = 0;
        let mut capture_frame_count: u64 = 0;
        let start_time = std::time::Instant::now();
        let frame_duration_ns = 1_000_000_000u64 / framerate as u64;
        let mut pipeline_playing = false;
        let mut ts_offset: Option<(u64, u64)> = None;

        while let Some(rec_frame) = frame_rx.recv().await {
            let Some(PusherFrame {
                mut buffer,
                sensor_ts,
                sequence,
            }) = prepare_frame(rec_frame, &appsrc)
            else {
                continue;
            };

            let capture_index = capture_frame_count;
            capture_frame_count += 1;
            if appsrc_queue_is_full(&appsrc) {
                RECORDING_STATS
                    .pusher_skipped
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let pts_ns = match compute_pts(
                &appsrc,
                sensor_ts,
                capture_index,
                frame_duration_ns,
                &mut pipeline_playing,
                &mut ts_offset,
            ) {
                PtsResult::Pts(pts) => pts,
                PtsResult::Skip => continue,
            };

            {
                let buf_ref = buffer.get_mut().unwrap();
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                buf_ref.set_duration(gst::ClockTime::from_nseconds(frame_duration_ns));
            }

            RECORDING_STATS.last_pts_ns.store(pts_ns, Ordering::Relaxed);

            if appsrc.push_buffer(buffer).is_err() {
                warn!(label, "Failed to push buffer to appsrc, stopping pusher");
                break;
            }

            RECORDING_STATS
                .pusher_pushed
                .fetch_add(1, Ordering::Relaxed);
            frame_count += 1;
            if frame_count.is_multiple_of(LOG_EVERY_N_FRAMES) {
                let elapsed = start_time.elapsed().as_secs_f64();
                debug!(
                    label,
                    frames = frame_count,
                    seq = ?sequence,
                    sensor_ts_ms = ?sensor_ts.map(|t| t / 1_000_000),
                    pts_ms = pts_ns / 1_000_000,
                    elapsed_secs = format!("{:.1}", elapsed),
                    effective_fps = format!("{:.1}", frame_count as f64 / elapsed),
                    "Pusher progress"
                );
            }
        }

        info!(
            label,
            total_frames = frame_count,
            "Frame channel closed, sending EOS to appsrc"
        );
        let _ = appsrc.end_of_stream();
    })
}

impl VideoRecorder {
    /// Create an appsrc-based video recorder for the libcamera backend.
    ///
    /// Frames from the native capture pipeline are received via `frame_rx` and
    /// pushed into a GStreamer encoding pipeline through `appsrc`. The preview
    /// continues uninterrupted because the same frames are displayed in the UI
    /// and forwarded here.
    ///
    /// The returned recorder must be started with `.start()`. When the `frame_rx`
    /// channel closes (sender dropped), the pusher task sends EOS and the
    /// pipeline finalizes gracefully.
    pub fn new_from_appsrc(
        config: AppsrcRecorderConfig<'_>,
        frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
    ) -> Result<Self, String> {
        let AppsrcRecorderConfig {
            base:
                RecorderConfig {
                    width,
                    height,
                    framerate,
                    output_path,
                    encoder_config,
                    enable_audio,
                    audio_device,
                    audio_source_rate_hz,
                    encoder_info,
                    capture_metadata,
                    audio_levels,
                },
            pixel_format,
            live_filter_code,
        } = config;

        // Keep stable RGBA appsrc caps for the whole recording so live filter
        // changes never renegotiate the GStreamer pipeline. Standard-filter
        // frames may still share tightly-packed RGBA storage directly.
        let initial_filter_code = live_filter_code.load(std::sync::atomic::Ordering::Relaxed);

        info!(
            width,
            height,
            framerate,
            format = ?pixel_format,
            initial_filter = initial_filter_code,
            output = %output_path.display(),
            audio = enable_audio,
            audio_device = ?audio_device,
            orientation = capture_metadata.orientation.gstreamer_tag(),
            "Creating appsrc-based video recorder (libcamera backend)"
        );

        // Boost the PA source to 100% before pulsesrc opens — see
        // `PulseSourceVolumeGuard`. Created here so it lives at least as long
        // as the recorder pipeline; dropped automatically when the
        // `VideoRecorder` itself drops (i.e. when recording stops).
        let pulse_volume_guard = build_pulse_volume_guard(enable_audio, audio_device);

        let setup = prepare_recorder(
            encoder_info,
            &encoder_config,
            enable_audio,
            audio_device,
            audio_source_rate_hz,
            output_path,
            framerate,
        )?;

        let (base_width, base_height) = (width, height);

        // OpenH264 has a maximum resolution limit — downscale if exceeded
        let (final_width, final_height) =
            openh264_downscale(base_width, base_height, &setup.encoder_name);

        // Only insert videoscale/capsfilter when downscaling is needed.
        // Capture orientation is stored as container metadata, so recording never
        // rotates or mirrors full video frames in software.
        let needs_scaling = final_width != base_width || final_height != base_height;

        // Always use RGBA input: the filtered pusher converts each frame to RGBA
        // (via GPU compute shader), applies the current filter, and pushes RGBA.
        // This lets the user toggle filters mid-recording.
        let initial_gst_format = "RGBA";

        let processing_chain = decoded_video_processing_chain_for_encoder(
            final_width,
            final_height,
            &setup.encoder_name,
            needs_scaling,
        );

        let pipeline_desc = format!(
            "appsrc name=camera-appsrc \
               caps=video/x-raw,format={fmt},width={w},height={h},framerate={fps}/1 \
               is-live=true do-timestamp=false format=time \
               min-latency={lat} max-latency={lat} \
             ! queue max-size-buffers=5 max-size-time=1000000000 \
             {processing} \
             ! {encoder} name=recording-encoder \
             {parser} \
             ! {muxer} name=recording-muxer \
             ! filesink location={loc}",
            fmt = initial_gst_format,
            w = width,
            h = height,
            fps = framerate,
            lat = setup.frame_duration_ns,
            processing = processing_chain,
            encoder = setup.encoder_name,
            parser = setup.parser_str,
            muxer = setup.muxer_name,
            loc = setup.output_path.display(),
        );

        info!(desc = %pipeline_desc, "Launching appsrc pipeline");

        let (pipeline, appsrc) = build_recorder_pipeline(
            &pipeline_desc,
            &setup.encoder_name,
            &encoder_config,
            final_width,
            final_height,
            setup.audio_elements.as_ref(),
            &audio_levels,
        )?;
        apply_video_tags(&pipeline, &capture_metadata);

        info!(
            initial_filter = initial_filter_code,
            "Pusher will apply live GPU filter (RGBA output)"
        );
        let pusher_handle =
            Self::spawn_filtered_pusher(appsrc, frame_rx, framerate, live_filter_code);

        // Publish diagnostics for the insights drawer
        let mode = if needs_scaling {
            "Filtered RGBA (videoconvert + scale)"
        } else {
            "Filtered RGBA (videoconvert)"
        };
        publish_recording_diagnostics(RecordingDiagnostics {
            mode: mode.to_string(),
            pipeline_string: pipeline_desc.clone(),
            encoder: setup.encoder_name.clone(),
            resolution: format!("{}x{}", final_width, final_height),
            framerate,
        });

        let recorder = VideoRecorder {
            pipeline,
            file_path: setup.output_path,
            _pulse_volume_guard: pulse_volume_guard,
            pusher_handle: Some(pusher_handle),
        };

        // Eagerly start: if a hardware encoder fails (e.g. VA-API backed by
        // NVENC in a flatpak sandbox), return Err so the caller can retry.
        recorder.start()?;

        Ok(recorder)
    }

    /// Spawn a pusher task that shares Standard-filter RGBA frames directly or
    /// converts and filters frames before pushing them to the same RGBA appsrc.
    ///
    /// Reads the current filter code from `live_filter_code` each frame so
    /// filter changes during recording are reflected in the output file.
    /// When filter code is 0 (Standard), the RGBA data is pushed without
    /// running the filter shader.
    fn spawn_filtered_pusher(
        appsrc: gst_app::AppSrc,
        mut frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
        framerate: u32,
        live_filter_code: Arc<std::sync::atomic::AtomicU32>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let initial = live_filter_code.load(std::sync::atomic::Ordering::Relaxed);
            info!(
                initial_filter_code = initial,
                "Filtered appsrc pusher task started"
            );

            let start_epoch_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            RECORDING_STATS
                .pusher_start_epoch_ns
                .store(start_epoch_ns, Ordering::Relaxed);

            let mut frame_count: u64 = 0;
            let mut capture_frame_count: u64 = 0;
            let start_time = std::time::Instant::now();
            let frame_duration_ns = 1_000_000_000u64 / framerate as u64;
            let mut pipeline_playing = false;
            let mut ts_offset: Option<(u64, u64)> = None;

            while let Some(rec_frame) = frame_rx.recv().await {
                let frame = match rec_frame {
                    RecordingFrame::Decoded(f) => f,
                    RecordingFrame::Jpeg { .. } => continue,
                };

                let capture_index = capture_frame_count;
                capture_frame_count += 1;
                if appsrc_queue_is_full(&appsrc) {
                    RECORDING_STATS
                        .pusher_skipped
                        .fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                let sensor_ts = frame.sensor_timestamp_ns;
                let sequence = frame.libcamera_metadata.as_ref().and_then(|m| m.sequence);

                // Read the live filter for every frame so Standard ↔ filtered
                // switching remains seamless within one fixed-caps pipeline.
                let filter_code = live_filter_code.load(std::sync::atomic::Ordering::Relaxed);
                let filter_type = crate::app::FilterType::from_gpu_filter_code(filter_code);

                let t0 = std::time::Instant::now();
                let mut buffer = if let Some(buffer) = try_shared_rgba_buffer(&frame, filter_type) {
                    buffer
                } else {
                    let rgba = match convert_frame_to_rgba(&frame).await {
                        Ok(data) => data,
                        Err(e) => {
                            warn!(error = %e, "Failed to convert frame to RGBA, skipping");
                            continue;
                        }
                    };

                    let output = if filter_type == crate::app::FilterType::Standard {
                        rgba
                    } else {
                        match crate::shaders::apply_filter_gpu_rgba(
                            &rgba,
                            frame.width,
                            frame.height,
                            filter_type,
                        )
                        .await
                        {
                            Ok(data) => data,
                            Err(e) => {
                                warn!(error = %e, "Failed to apply filter, using unfiltered RGBA");
                                rgba
                            }
                        }
                    };

                    gst::Buffer::from_mut_slice(output)
                };

                RECORDING_STATS
                    .last_convert_time_us
                    .store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);

                let pts_ns = match compute_pts(
                    &appsrc,
                    sensor_ts,
                    capture_index,
                    frame_duration_ns,
                    &mut pipeline_playing,
                    &mut ts_offset,
                ) {
                    PtsResult::Pts(pts) => pts,
                    PtsResult::Skip => continue,
                };

                {
                    let buf_ref = buffer.get_mut().unwrap();
                    buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                    buf_ref.set_duration(gst::ClockTime::from_nseconds(frame_duration_ns));
                }

                RECORDING_STATS.last_pts_ns.store(pts_ns, Ordering::Relaxed);

                if appsrc.push_buffer(buffer).is_err() {
                    warn!("Filtered pusher: failed to push buffer, stopping");
                    break;
                }

                RECORDING_STATS
                    .pusher_pushed
                    .fetch_add(1, Ordering::Relaxed);
                frame_count += 1;
                if frame_count.is_multiple_of(LOG_EVERY_N_FRAMES) {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    debug!(
                        frames = frame_count,
                        seq = ?sequence,
                        sensor_ts_ms = ?sensor_ts.map(|t| t / 1_000_000),
                        pts_ms = pts_ns / 1_000_000,
                        elapsed_secs = format!("{:.1}", elapsed),
                        effective_fps = format!("{:.1}", frame_count as f64 / elapsed),
                        filter_time_us = t0.elapsed().as_micros(),
                        "Filtered pusher progress"
                    );
                }
            }

            info!(
                total_frames = frame_count,
                "Frame channel closed, sending EOS to filtered appsrc"
            );
            let _ = appsrc.end_of_stream();
        })
    }

    /// Create a VA-API JPEG zero-copy recording pipeline.
    ///
    /// Instead of CPU-decoding MJPEG and converting to NV12, this pipeline sends
    /// raw JPEG bytes through `appsrc` → `vajpegdec` (GPU decode) → `vah265enc`.
    /// The GPU decoder outputs NV12 in VA-API memory that the encoder consumes
    /// zero-copy, eliminating both the turbojpeg CPU decode and the I420→NV12
    /// software conversion from the recording path.
    ///
    /// Falls back to `None` if the pipeline cannot be constructed (caller should
    /// retry with the legacy `new_from_appsrc` path).
    pub fn new_from_appsrc_jpeg(
        config: AppsrcRecorderConfig<'_>,
        va_jpeg_dec: &str,
        frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
    ) -> Result<Self, String> {
        let AppsrcRecorderConfig {
            base:
                RecorderConfig {
                    width,
                    height,
                    framerate,
                    output_path,
                    encoder_config,
                    enable_audio,
                    audio_device,
                    audio_source_rate_hz,
                    encoder_info,
                    capture_metadata,
                    audio_levels,
                },
            pixel_format: _,
            live_filter_code,
        } = config;

        if live_filter_code.load(std::sync::atomic::Ordering::Relaxed) != 0 {
            return Err(
                "VA-API JPEG pipeline does not support filters; falling back to legacy".to_string(),
            );
        }

        info!(
            width,
            height,
            framerate,
            va_jpeg_dec,
            output = %output_path.display(),
            audio = enable_audio,
            "Creating VA-API JPEG zero-copy recording pipeline"
        );

        // Boost PA source volume to 100% before pulsesrc opens — see
        // `PulseSourceVolumeGuard` and the matching block in `new_from_appsrc`.
        let pulse_volume_guard = build_pulse_volume_guard(enable_audio, audio_device);

        let mut setup = prepare_recorder(
            encoder_info,
            &encoder_config,
            enable_audio,
            audio_device,
            audio_source_rate_hz,
            output_path,
            framerate,
        )?;

        // Match encoder to decoder memory domain to avoid implicit GPU memory
        // transfers that cause frame stalls:
        //   nvjpegdec (CUDA memory)   → nvh265enc/nvh264enc (CUDA memory)
        //   vajpegdec (VA-API memory) → vah265enc/vah264enc (VA-API memory)
        let is_nvidia_decoder = va_jpeg_dec.starts_with("nv");
        if is_nvidia_decoder && !setup.encoder_name.starts_with("nv") {
            use crate::media::encoders::detection::probe_single_encoder;
            if probe_single_encoder("nvh265enc") {
                warn!(
                    decoder = va_jpeg_dec,
                    selected = %setup.encoder_name,
                    override_to = "nvh265enc",
                    "Overriding encoder to match NVIDIA decoder memory domain"
                );
                setup.encoder_name = "nvh265enc".to_string();
                setup.parser_str = "! h265parse".to_string();
                setup.muxer_name = "mp4mux".to_string();
            } else if probe_single_encoder("nvh264enc") {
                warn!(
                    decoder = va_jpeg_dec,
                    selected = %setup.encoder_name,
                    override_to = "nvh264enc",
                    "Overriding encoder to match NVIDIA decoder memory domain"
                );
                setup.encoder_name = "nvh264enc".to_string();
                setup.parser_str = "! h264parse".to_string();
                setup.muxer_name = "mp4mux".to_string();
            } else {
                warn!(
                    decoder = va_jpeg_dec,
                    "NVIDIA encoders not functional, using selected encoder with potential memory transfer"
                );
            }
        }

        let encoder_caps = encoder_input_caps_filter(&setup.encoder_name);
        let pipeline_desc = format!(
            "appsrc name=camera-appsrc \
               caps=image/jpeg,width={w},height={h},framerate={fps}/1 \
               is-live=true do-timestamp=false format=time \
               min-latency={lat} max-latency={lat} \
             ! queue max-size-buffers=60 max-size-time=3000000000 \
             ! {decoder} name=jpeg-decoder \
             ! videoconvert \
             {encoder_caps} \
             ! {encoder} name=recording-encoder \
             {parser} \
             ! {muxer} name=recording-muxer \
             ! filesink location={loc}",
            w = width,
            h = height,
            fps = framerate,
            lat = setup.frame_duration_ns,
            decoder = va_jpeg_dec,
            encoder_caps = encoder_caps,
            encoder = setup.encoder_name,
            parser = setup.parser_str,
            muxer = setup.muxer_name,
            loc = setup.output_path.display(),
        );

        info!(desc = %pipeline_desc, "Launching JPEG zero-copy pipeline");

        if setup.audio_elements.is_some() {
            info!("A/V sync: audio branch active, video PTS compensated in compute_pts");
        }

        let (pipeline, appsrc) = build_recorder_pipeline(
            &pipeline_desc,
            &setup.encoder_name,
            &encoder_config,
            width,
            height,
            setup.audio_elements.as_ref(),
            &audio_levels,
        )?;
        apply_video_tags(&pipeline, &capture_metadata);

        // JPEG-specific PTS verification probes
        if let Some(decoder) = pipeline.by_name("jpeg-decoder") {
            install_pts_trace_probe(&decoder, "decoder-out");
        }
        if let Some(enc_element) = pipeline.by_name("recording-encoder") {
            install_pts_trace_probe(&enc_element, "encoder-out");
        }

        let pusher_handle = Self::spawn_appsrc_jpeg_pusher(appsrc, frame_rx, framerate);

        publish_recording_diagnostics(RecordingDiagnostics {
            mode: format!("JPEG zero-copy ({} → {})", va_jpeg_dec, setup.encoder_name),
            pipeline_string: pipeline_desc.clone(),
            encoder: setup.encoder_name.clone(),
            resolution: format!("{}x{}", width, height),
            framerate,
        });

        let recorder = VideoRecorder {
            pipeline,
            file_path: setup.output_path,
            _pulse_volume_guard: pulse_volume_guard,
            pusher_handle: Some(pusher_handle),
        };

        // Eagerly start the pipeline so failures (e.g. NVIDIA encoder not
        // functional in a flatpak sandbox) are caught here and the caller
        // can fall back to the legacy appsrc path.
        recorder.start()?;

        Ok(recorder)
    }

    /// Spawn the JPEG (zero-copy) pusher task.
    ///
    /// Passes raw JPEG bytes straight through via [`spawn_pusher`].
    fn spawn_appsrc_jpeg_pusher(
        appsrc: gst_app::AppSrc,
        frame_rx: tokio::sync::mpsc::Receiver<RecordingFrame>,
        framerate: u32,
    ) -> tokio::task::JoinHandle<()> {
        spawn_pusher(
            appsrc,
            frame_rx,
            framerate,
            "JPEG recorder",
            |rec_frame, _appsrc| match rec_frame {
                RecordingFrame::Jpeg {
                    data,
                    sensor_timestamp_ns,
                    sequence,
                    ..
                } => Some(PusherFrame {
                    buffer: gst::Buffer::from_slice(data),
                    sensor_ts: sensor_timestamp_ns,
                    sequence,
                }),
                RecordingFrame::Decoded(_) => None,
            },
        )
    }

    /// Create audio branch elements
    ///
    /// Uses `pulsesrc` (PipeWire's PulseAudio compatibility layer) for reliable
    /// audio capture from all device types including pro-audio (multi-channel)
    /// and standard stereo/mono sources.
    ///
    /// All input channels are mixed down to mono via a capsfilter. The same
    /// capsfilter pins the sample rate to `opus_target_rate(audio_source_rate_hz)`
    /// — the source's native rate when Opus can encode it, else 48 kHz — so
    /// no `audioresample` element is needed in the GStreamer graph; PulseAudio
    /// handles any conversion internally and `opusenc` always sees a rate it
    /// accepts.
    fn create_audio_branch(
        audio_device: Option<&str>,
        audio_source_rate_hz: u32,
        audio_encoder_config: crate::media::encoders::audio::SelectedAudioEncoder,
    ) -> Result<Option<AudioBranch>, String> {
        let mut source_builder = gst::ElementFactory::make("pulsesrc")
            // Resample against the pipeline clock so audio stays synchronized
            // without `skew` advancing the capture pointer and dropping mic
            // samples when video processing temporarily starves the process.
            //
            // The value is sourced from `PULSESRC_SLAVE_METHOD` so the
            // pre-recording probe in `audio_probe.rs` configures pulsesrc
            // identically.
            .property_from_str("slave-method", PULSESRC_SLAVE_METHOD)
            .property("buffer-time", PULSESRC_BUFFER_TIME_US)
            .property("provide-clock", false);

        // pulsesrc `device` property takes the PipeWire/PulseAudio node name
        // (e.g. "alsa_input.usb-Focusrite_Scarlett_4i4_4th_Gen_...-00.pro-input-0")
        if let Some(device) = audio_device {
            if !device.is_empty() {
                info!(device = %device, "Using audio source device");
                source_builder = source_builder.property("device", device);
            }
        } else {
            info!("Using default audio source");
        }

        let source = source_builder
            .build()
            .map_err(|e| format!("Failed to create audio source: {}", e))?;

        let queue = gst::ElementFactory::make("queue")
            .property("max-size-buffers", 200u32)
            .property("max-size-time", 2_000_000_000u64)
            .build()
            .map_err(|e| format!("Failed to create audio queue: {}", e))?;

        let convert = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|e| format!("Failed to create audioconvert: {}", e))?;

        // Defensive `audioresample`: in the steady-state case the source
        // already emits at the rate `opus_target_rate` requested (PA negotiates
        // it server-side, see the capsfilter below), and this element acts as a
        // zero-cost pass-through. It exists for the default-source edge case
        // where `audio_source_rate_hz` is 0 (unknown), `opus_target_rate`
        // defaults to 48 kHz, and PA happens to serve a card at a rate it
        // won't or can't resample (some PA configs disable resampling). In
        // that case GStreamer needs an in-pipeline resampler or the audio
        // branch fails to negotiate.
        let resample = gst::ElementFactory::make("audioresample")
            .build()
            .map_err(|e| format!("Failed to create audioresample: {}", e))?;

        // Force mono output + an Opus-compatible sample rate. `opus_target_rate`
        // returns the source's native rate when Opus accepts it (no resampling
        // anywhere); otherwise 48 kHz, in which case either PulseAudio resamples
        // internally or the `audioresample` element above picks up the slack.
        let target_rate =
            crate::media::encoders::audio::opus_target_rate(audio_source_rate_hz) as i32;
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("audio/x-raw")
                    .field("channels", 1i32)
                    .field("rate", target_rate)
                    .build(),
            )
            .build()
            .map_err(|e| format!("Failed to create audio capsfilter: {}", e))?;
        info!(
            source_rate_hz = audio_source_rate_hz,
            target_rate_hz = target_rate,
            "Audio capsfilter rate negotiated"
        );

        // Soft-knee downward compressor — squashes loud peaks so the following
        // makeup-gain stage can lift quiet content without clipping. Without
        // this, on-camera mics with low sensitivity (USB webcams) or PA
        // sources that get attenuated by their ALSA UCM profile (e.g. the
        // Pixel 3a's VoiceCall mic clamped at -30 dB) record audio at
        // -40 LUFS or worse.
        //
        // gstreamer1.0-plugins-good `audiodynamic`:
        // - mode=compressor, characteristics=soft-knee
        // - threshold is normalized [0..1] amplitude; 0.2 ≈ -14 dBFS.
        // - ratio is the *output* slope above threshold: 0.3 ≈ 3:1 ratio
        //   (signals 12 dB over threshold land 4 dB over).
        // NB: `audiodynamic.threshold` / `.ratio` are GLib `gfloat` (f32),
        // not `gdouble` — passing f64 triggers a runtime "can't be set from
        // the given type" GObject type-mismatch panic. Parameter values are
        // shared with the settings probe via `pipelines::audio_level::dynamics`.
        use crate::pipelines::audio_level::dynamics;
        let compressor = gst::ElementFactory::make("audiodynamic")
            .property_from_str("mode", "compressor")
            .property_from_str("characteristics", "soft-knee")
            .property("threshold", dynamics::COMPRESSOR_THRESHOLD)
            .property("ratio", dynamics::COMPRESSOR_RATIO)
            .build()
            .map_err(|e| format!("Failed to create audio compressor: {}", e))?;

        // Makeup gain — lifts the compressed signal back to a sane recording
        // level. Linear scale: 2.0 ≈ +6 dB. With the upstream
        // `PulseSourceVolumeGuard` already boosting PA to 100%, +6 dB sits
        // safely below the brick-wall limiter that follows; on platforms
        // without `pactl` (so PA is wherever the user left it) +6 dB still
        // provides a noticeable lift.
        let makeup_gain = gst::ElementFactory::make("volume")
            .property("volume", dynamics::MAKEUP_GAIN)
            .build()
            .map_err(|e| format!("Failed to create makeup-gain element: {}", e))?;

        // Brick-wall limiter after makeup gain. Catches transients that the
        // 3:1 compressor lets slip through, capping output around -0.45 dBFS
        // (threshold 0.95 → 20·log10(0.95)). Ratio 0.05 ≈ 20:1, effectively
        // a limiter; characteristics=hard-knee for the sharpest cutoff.
        // Without this the boosted Pixel 3a recordings clip at +4 dBFS.
        let limiter = gst::ElementFactory::make("audiodynamic")
            .property_from_str("mode", "compressor")
            .property_from_str("characteristics", "hard-knee")
            .property("threshold", dynamics::LIMITER_THRESHOLD)
            .property("ratio", dynamics::LIMITER_RATIO)
            .build()
            .map_err(|e| format!("Failed to create audio limiter: {}", e))?;

        // Single level meter AFTER compressor + makeup gain — the UI then
        // reads the actual recorded signal level, not the raw mic level.
        let level = gst::ElementFactory::make("level")
            .name("audio-level-output")
            .property("post-messages", true)
            .property("interval", 100_000_000u64) // 100ms
            .build()
            .map_err(|e| format!("Failed to create level meter: {}", e))?;

        let encoder = audio_encoder_config.encoder;

        Ok(Some(AudioBranch {
            source,
            queue,
            convert,
            resample,
            capsfilter,
            compressor,
            makeup_gain,
            limiter,
            level,
            encoder,
        }))
    }

    /// Link audio chain:
    /// source → queue → convert → resample → capsfilter(mono) → compressor → makeup_gain → limiter → level → encoder
    fn link_audio_chain(audio_branch: &AudioBranch) -> Result<(), String> {
        gst::Element::link_many([
            &audio_branch.source,
            &audio_branch.queue,
            &audio_branch.convert,
            &audio_branch.resample,
            &audio_branch.capsfilter,
            &audio_branch.compressor,
            &audio_branch.makeup_gain,
            &audio_branch.limiter,
            &audio_branch.level,
            &audio_branch.encoder,
        ])
        .map_err(|_| "Failed to link audio chain")?;

        Ok(())
    }

    /// Start recording (idempotent — no-op if already playing)
    pub fn start(&self) -> Result<(), String> {
        // Skip if already playing (e.g. JPEG zero-copy path starts eagerly)
        if self.pipeline.current_state() == gst::State::Playing {
            info!("Pipeline already playing, skipping start");
            return Ok(());
        }

        info!("Starting video recording pipeline");

        // Log pipeline element names for diagnostics
        let mut element_names = Vec::new();
        for e in self.pipeline.iterate_elements().into_iter().flatten() {
            element_names.push(e.name().to_string());
        }
        info!(elements = ?element_names, "Pipeline elements");

        let result = self
            .pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| format!("Failed to start recording: {}", e))?;
        info!(state_change = ?result, "Pipeline set to Playing");
        Ok(())
    }

    /// Stop recording and finalize the file
    pub fn stop(mut self) -> Result<PathBuf, String> {
        info!("Stopping video recording");
        clear_recording_diagnostics();

        // Send EOS directly to every source element's src pad.
        // pipeline.send_event(EOS) doesn't reliably reach live sources like pulsesrc,
        // so the muxer (aggregator) never sees EOS on the audio pad and hangs.
        // By pushing EOS on each source's src pad, both appsrc and pulsesrc branches
        // propagate EOS through to the muxer, allowing it to finalize.
        info!("Sending EOS to all source elements");
        let iter = self.pipeline.iterate_sources();
        let mut eos_sent = 0u32;
        for src in iter {
            let Ok(src) = src else { continue };
            let name = src.name().to_string();
            // Use element-level send_event (not pad-level) — for source elements
            // this routes downstream events via gst_pad_push_event on the src pad.
            // pad.send_event() would send upstream, which is wrong for EOS.
            debug!(element = %name, "Sending EOS to source element");
            src.send_event(gst::event::Eos::new());
            eos_sent += 1;
        }
        if eos_sent == 0 {
            // Fallback: send EOS to pipeline
            warn!("No source pads found, sending EOS to pipeline");
            if !self.pipeline.send_event(gst::event::Eos::new()) {
                warn!("Failed to send EOS event to pipeline");
            }
        } else {
            info!(eos_sent, "EOS sent to source elements");
        }

        let mut eos_timeout = false;

        // Wait for EOS to propagate through the entire pipeline.
        // The bus posts an EOS message only after ALL sink elements have received
        // EOS, which means the muxer has finalized (written moov atom for MP4,
        // duration for WebM, etc.) and the filesink has flushed.
        if let Some(bus) = self.pipeline.bus() {
            info!("Waiting for pipeline EOS on bus...");
            match bus.timed_pop_filtered(
                gst::ClockTime::from_seconds(60),
                &[gst::MessageType::Eos, gst::MessageType::Error],
            ) {
                Some(msg) => match msg.view() {
                    gst::MessageView::Eos(_) => {
                        info!("Pipeline EOS received — file finalized");
                    }
                    gst::MessageView::Error(err) => {
                        error!(
                            error = %err.error(),
                            debug = ?err.debug(),
                            source = ?err.src().map(|s| s.name()),
                            "GStreamer error while waiting for EOS"
                        );
                        eos_timeout = true;
                    }
                    _ => {}
                },
                None => {
                    warn!("Timeout (60s) waiting for pipeline EOS, forcing shutdown");
                    eos_timeout = true;
                }
            }
        } else {
            // Fallback: no bus available, use fixed sleep
            warn!("No pipeline bus available, using fixed sleep fallback");
            std::thread::sleep(std::time::Duration::from_millis(1000));
            eos_timeout = true;
        }

        // Abort the pusher task before tearing down the pipeline so it can't
        // race with the NULL transition by pushing into a finalising appsrc.
        if let Some(handle) = self.pusher_handle.take() {
            handle.abort();
        }

        // Set pipeline to NULL state - this will trigger final cleanup
        info!("Setting pipeline to NULL state");
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| format!("Failed to stop pipeline: {}", e))?;

        let file_path = std::mem::take(&mut self.file_path);
        if eos_timeout {
            warn!(path = %file_path.display(), "Recording may be incomplete (EOS timeout)");
            Err(format!(
                "Recording saved but may be incomplete: {}",
                file_path.display()
            ))
        } else {
            info!(path = %file_path.display(), "Recording saved");
            Ok(file_path)
        }
    }
}

impl Drop for VideoRecorder {
    fn drop(&mut self) {
        // Abort the pusher first so it cannot keep pushing buffers into the
        // pipeline while we transition it to NULL.
        if let Some(handle) = self.pusher_handle.take() {
            handle.abort();
        }
        // Remove the bus sync handler to release its captured references
        if let Some(bus) = self.pipeline.bus() {
            bus.unset_sync_handler();
        }
        // Ensure pipeline is properly stopped — this disconnects pulsesrc from
        // PulseAudio and releases all GStreamer resources.
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

/// Audio branch elements
struct AudioBranch {
    source: gst::Element,
    queue: gst::Element,
    convert: gst::Element,
    /// Zero-cost pass-through when the source already produces the target rate;
    /// only does work when PA can't (or won't) serve the requested rate.
    resample: gst::Element,
    capsfilter: gst::Element,
    /// Soft-knee downward compressor. Squashes loud peaks so the following
    /// makeup-gain stage can lift quiet content without clipping. Same chain
    /// is used by the settings audio probe so the meter reflects what the
    /// recording captures.
    compressor: gst::Element,
    /// Makeup gain applied after the compressor. Linear scale; the user-facing
    /// "audio gain" setting will eventually drive this property.
    makeup_gain: gst::Element,
    /// Brick-wall limiter after makeup gain — caps output around -0.4 dBFS
    /// so a strong source plus +6 dB makeup never clips the encoder.
    limiter: gst::Element,
    /// Level meter after compressor + makeup gain so the UI reads the actual
    /// recorded signal level. The pre-mix per-channel meter was removed to
    /// cut CPU pressure on weak ARM hardware — same reason `audioresample`
    /// is gone (source rate flows through unchanged).
    level: gst::Element,
    encoder: gst::Element,
}

/// Check which video encoders are available (backward compatibility)
pub fn check_available_encoders() {
    crate::media::encoders::log_available_encoders();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rgba_frame(width: u32, height: u32, stride: u32) -> CameraFrame {
        let len = stride as usize * height as usize;
        CameraFrame {
            width,
            height,
            data: crate::backends::camera::types::FrameData::from_owned_vec(
                (0..len).map(|value| value as u8).collect(),
            ),
            format: PixelFormat::RGBA,
            stride,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        }
    }

    #[test]
    fn decoded_recording_orientation_does_not_transform_pixels() {
        let processing = decoded_video_processing_chain_for_encoder(1080, 1920, "nvh264enc", true);

        assert!(!processing.contains("videoflip"));
        assert!(processing.contains("width=1080,height=1920"));
    }

    #[test]
    fn software_encoder_rgba_processing_constrains_chroma_to_i420() {
        let x264 = decoded_video_processing_chain_for_encoder(4000, 3000, "x264enc", false);
        let x265 = decoded_video_processing_chain_for_encoder(4000, 3000, "x265enc", false);

        assert!(x264.contains("format=I420"));
        assert!(x265.contains("format=I420"));
    }

    #[test]
    fn standard_filter_wraps_tightly_packed_rgba_without_copying() {
        gst::init().unwrap();
        let frame = test_rgba_frame(4, 2, 16);
        let source_ptr = frame.data.as_ref().as_ptr();

        let buffer = try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard)
            .expect("tightly packed Standard RGBA should use shared storage");
        let mapped = buffer.map_readable().expect("buffer should be readable");

        assert_eq!(mapped.as_slice(), frame.data.as_ref());
        assert_eq!(mapped.as_slice().as_ptr(), source_ptr);
    }

    #[test]
    fn live_filter_switching_only_uses_shared_storage_for_standard_frames() {
        gst::init().unwrap();
        let frame = test_rgba_frame(4, 2, 16);

        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard).is_some());
        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Sepia).is_none());
        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard).is_some());
    }

    #[test]
    fn padded_standard_rgba_still_uses_the_repacking_path() {
        gst::init().unwrap();
        let frame = test_rgba_frame(4, 2, 20);

        assert!(try_shared_rgba_buffer(&frame, crate::app::FilterType::Standard).is_none());
    }

    #[test]
    fn video_tags_include_standard_orientation_and_device_fields() {
        gst::init().unwrap();
        let metadata = crate::pipelines::capture_metadata::CaptureMetadata {
            device_make: Some("Example".to_string()),
            device_model: Some("Phone".to_string()),
            captured_at: Some(
                chrono::DateTime::parse_from_rfc3339("2026-09-28T12:34:56.789+02:00").unwrap(),
            ),
            orientation: crate::pipelines::capture_metadata::CaptureOrientation::FlipRotate90,
            ..Default::default()
        };

        let tags = video_tags(&metadata);

        assert_eq!(
            tags.get::<gst::tags::ImageOrientation>().unwrap().get(),
            "flip-rotate-90"
        );
        assert_eq!(
            tags.get::<gst::tags::DeviceManufacturer>().unwrap().get(),
            "Example"
        );
        assert_eq!(tags.get::<gst::tags::DeviceModel>().unwrap().get(), "Phone");
        assert_eq!(
            tags.get::<gst::tags::ApplicationName>().unwrap().get(),
            crate::pipelines::capture_metadata::CaptureMetadata::application_name()
        );
        assert!(tags.get::<gst::tags::DateTime>().is_some());
    }

    #[test]
    fn live_appsrc_queue_stays_bounded_when_downstream_is_slow() {
        gst::init().unwrap();
        let pipeline = gst::parse::launch(
            "appsrc name=test-appsrc is-live=true format=time \
             ! identity sleep-time=250000 \
             ! fakesink sync=false",
        )
        .unwrap()
        .downcast::<gst::Pipeline>()
        .unwrap();
        let appsrc = pipeline
            .by_name("test-appsrc")
            .unwrap()
            .downcast::<gst_app::AppSrc>()
            .unwrap();

        configure_live_appsrc(&appsrc);
        pipeline.set_state(gst::State::Playing).unwrap();

        for _ in 0..32 {
            appsrc
                .push_buffer(gst::Buffer::with_size(1024).unwrap())
                .unwrap();
        }

        assert_eq!(appsrc.max_buffers(), APPSRC_MAX_BUFFERS);
        assert_eq!(appsrc.max_bytes(), 0);
        assert!(!appsrc.is_block());
        assert_eq!(appsrc.leaky_type(), gst_app::AppLeakyType::Downstream);
        assert!(appsrc.current_level_buffers() <= APPSRC_MAX_BUFFERS);
        pipeline.set_state(gst::State::Null).unwrap();
    }

    #[test]
    fn queue_capacity_treats_zero_as_unlimited() {
        assert!(!queue_at_capacity(100, 0));
        assert!(!queue_at_capacity(
            APPSRC_MAX_BUFFERS - 1,
            APPSRC_MAX_BUFFERS
        ));
        assert!(queue_at_capacity(APPSRC_MAX_BUFFERS, APPSRC_MAX_BUFFERS));
    }

    #[test]
    fn fallback_timestamps_preserve_time_across_dropped_frames() {
        let frame_duration_ns = 40_000_000;

        assert_eq!(fallback_pts(0, frame_duration_ns), 0);
        // Capture index 1 represents a frame discarded under backpressure.
        assert_eq!(fallback_pts(2, frame_duration_ns), 80_000_000);
    }
}
