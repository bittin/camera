// SPDX-License-Identifier: GPL-3.0-only
//! Selected-codec configuration shared by the continuous memory recorder.
//! The file encoder below is a test-only mux/codec compatibility fixture.
use crate::media::encoders::video::{EncoderInfo, VideoQuality, configure_video_encoder};
#[cfg(test)]
use crate::pipelines::video::encoder_selection::prepare_video_encoder_input;
use gst::prelude::*;
use gstreamer as gst;
#[cfg(test)]
use gstreamer_app as gst_app;
#[cfg(test)]
use std::path::Path;

pub(crate) fn validate_encoder(info: &EncoderInfo) -> Result<(), String> {
    if info.element_name.contains("vp8") || info.element_name.contains("vp9") {
        return Err(
            "Motion Photo requires AVC, HEVC or AV1 in MP4; select a compatible video encoder"
                .into(),
        );
    }
    Ok(())
}

pub(super) fn configure_clip_video_encoder(
    encoder: &gst::Element,
    info: &EncoderInfo,
    width: u32,
    height: u32,
    bitrate: u32,
) -> Result<(), String> {
    configure_video_encoder(
        encoder,
        &info.element_name,
        VideoQuality::Low,
        width,
        height,
        Some(bitrate),
    );
    if info.element_name.starts_with("v4l2") {
        if encoder.find_property("extra-controls").is_none() {
            return Err(
                "Selected V4L2 encoder cannot apply the Motion Photo bitrate preset".into(),
            );
        }
        let controls = gst::Structure::builder("controls")
            .field(
                "video_bitrate",
                bitrate.saturating_mul(1000).min(i32::MAX as u32) as i32,
            )
            .build();
        encoder.set_property("extra-controls", controls);
    }
    Ok(())
}

pub(super) fn add_aac_branch(
    pipeline: &gst::Pipeline,
    muxer: &gst::Element,
    audio: &gst::Element,
) -> Result<(), String> {
    let make = |name: &str| {
        gst::ElementFactory::make(name)
            .build()
            .map_err(|e| e.to_string())
    };
    let convert = make("audioconvert")?;
    let resample = make("audioresample")?;
    // Motion Photo permits mono/stereo AAC at 44.1, 48 or 96 kHz.
    // Pin a compliant layout instead of inheriting arbitrary device caps.
    let caps = gst::ElementFactory::make("capsfilter")
        .property(
            "caps",
            gst::Caps::builder("audio/x-raw")
                .field("rate", 48_000i32)
                .field("channels", 1i32)
                .build(),
        )
        .build()
        .map_err(|e| e.to_string())?;
    let encoder = ["fdkaacenc", "voaacenc", "avenc_aac"]
        .into_iter()
        .find_map(|name| make(name).ok())
        .ok_or("Motion Photo audio requires an AAC encoder")?;
    let parse = make("aacparse")?;
    let queue = make("queue")?;
    let chain = [audio, &convert, &resample, &caps, &encoder, &parse, &queue];
    pipeline.add_many(chain).map_err(|e| e.to_string())?;
    gst::Element::link_many(chain).map_err(|e| e.to_string())?;
    queue.link(muxer).map_err(|e| e.to_string())?;
    for element in chain {
        element
            .sync_state_with_parent()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) struct ClipEncoder {
    pipeline: gst::Pipeline,
    source: gst_app::AppSrc,
    audio_source: Option<gst::Element>,
    framerate: u32,
}
#[cfg(test)]
impl ClipEncoder {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        path: &Path,
        info: &EncoderInfo,
        width: u32,
        height: u32,
        framerate: u32,
        bitrate: u32,
        audio_device: Option<&str>,
        origin: std::time::Instant,
    ) -> Result<Self, String> {
        validate_encoder(info)?;
        gst::init().map_err(|e| e.to_string())?;
        let pipeline = gst::Pipeline::new();
        let result = Self::build(
            &pipeline,
            path,
            info,
            width,
            height,
            framerate,
            bitrate,
            audio_device,
            origin,
        );
        if result.is_err() {
            let _ = pipeline.set_state(gst::State::Null);
        }
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn build(
        pipeline: &gst::Pipeline,
        path: &Path,
        info: &EncoderInfo,
        width: u32,
        height: u32,
        framerate: u32,
        bitrate: u32,
        audio_device: Option<&str>,
        origin: std::time::Instant,
    ) -> Result<Self, String> {
        let make = |name: &str| {
            gst::ElementFactory::make(name)
                .build()
                .map_err(|e| e.to_string())
        };
        let encoder = make(&info.element_name)?;
        configure_clip_video_encoder(&encoder, info, width, height, bitrate)?;
        let input = prepare_video_encoder_input(
            &encoder,
            &info.element_name,
            width & !1,
            height & !1,
            framerate,
        )?;
        let source = gst_app::AppSrc::builder()
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", width as i32)
                    .field("height", height as i32)
                    .field("framerate", gst::Fraction::new(framerate as i32, 1))
                    .build(),
            )
            .format(gst::Format::Time)
            .is_live(true)
            .block(false)
            .max_buffers(3)
            .build();
        source.set_property_from_str("leaky-type", "downstream");
        let convert = make("videoconvert")?;
        let scale = make("videoscale")?;
        let caps = gst::ElementFactory::make("capsfilter")
            .property("caps", input.caps(framerate))
            .build()
            .map_err(|e| e.to_string())?;
        let parser = make(
            info.codec
                .parser_name()
                .ok_or("Selected codec has no MP4 parser")?,
        )?;
        let muxer = make("mp4mux")?;
        let sink = gst::ElementFactory::make("filesink")
            .property(
                "location",
                path.to_str().ok_or("Non-UTF8 Motion Photo path")?,
            )
            .build()
            .map_err(|e| e.to_string())?;
        let chain = [
            source.upcast_ref(),
            &convert,
            &scale,
            &caps,
            &encoder,
            &parser,
            &muxer,
            &sink,
        ];
        pipeline.add_many(chain).map_err(|e| e.to_string())?;
        gst::Element::link_many(chain).map_err(|e| e.to_string())?;
        let audio_source = if let Some(device) = audio_device {
            let audio = gst::ElementFactory::make("pulsesrc")
                .property("device", device)
                .build()
                .map_err(|e| e.to_string())?;
            add_aac_branch(pipeline, &muxer, &audio)?;
            Some(audio)
        } else {
            None
        };
        // Audio sources timestamp against pipeline running time. Anchor that
        // clock to the shutter too, including processing/build startup latency;
        // buffered video retains its actual capture PTS rather than being rebased.
        let clock = gst::SystemClock::obtain();
        pipeline.use_clock(Some(&clock));
        pipeline.set_start_time(gst::ClockTime::NONE);
        pipeline.set_base_time(clock.time().saturating_sub(gst::ClockTime::from_nseconds(
            origin.elapsed().as_nanos() as u64,
        )));
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            pipeline: pipeline.clone(),
            source,
            audio_source,
            framerate,
        })
    }
    pub(crate) fn push(&mut self, image: image::RgbImage, pts: u64) -> Result<(), String> {
        // GStreamer raw RGB rows are aligned to four bytes.
        let row = image.width() as usize * 3;
        let stride = (row + 3) & !3;
        let mut bytes = vec![0; stride * image.height() as usize];
        for (src, dst) in image
            .as_raw()
            .chunks_exact(row)
            .zip(bytes.chunks_exact_mut(stride))
        {
            dst[..row].copy_from_slice(src);
        }
        let mut buffer = gst::Buffer::from_mut_slice(bytes);
        let b = buffer
            .get_mut()
            .ok_or("Motion Photo buffer is not writable")?;
        b.set_pts(gst::ClockTime::from_nseconds(pts));
        b.set_duration(gst::ClockTime::from_nseconds(
            1_000_000_000 / self.framerate as u64,
        ));
        self.source
            .push_buffer(buffer)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    pub(crate) fn finish(self) -> Result<(), String> {
        self.source.end_of_stream().map_err(|e| e.to_string())?;
        if let Some(audio) = &self.audio_source {
            audio.send_event(gst::event::Eos::new());
        }
        let bus = self
            .pipeline
            .bus()
            .ok_or("Motion Photo pipeline has no bus")?;
        match bus.timed_pop_filtered(
            gst::ClockTime::from_seconds(5),
            &[gst::MessageType::Eos, gst::MessageType::Error],
        ) {
            Some(message) => match message.view() {
                gst::MessageView::Eos(_) => Ok(()),
                gst::MessageView::Error(e) => Err(format!(
                    "Motion Photo encoding failed: {} ({:?})",
                    e.error(),
                    e.debug()
                )),
                _ => unreachable!(),
            },
            None => Err("Motion Photo encoder did not finalize within five seconds".into()),
        }
    }
}
#[cfg(test)]
impl Drop for ClipEncoder {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::encoders::video::{EncoderInfo, VideoCodec};

    /// Recover the appended MP4 using the published directory, not the input size.
    fn check_packaged_clip_decodes_in_both_still_formats(clip: &Path) {
        let root =
            std::env::temp_dir().join(format!("motion-photo-package-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let pixels = image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90]));
        for ext in ["jpg", "avif"] {
            let still = root.join(format!("still.{ext}"));
            pixels.save(&still).unwrap();
            let packaged =
                crate::pipelines::motion_photo::embed_motion_photo(&still, clip, -1).unwrap();
            let bytes = std::fs::read(&packaged).unwrap();
            let start = bytes
                .windows(b"<x:xmpmeta".len())
                .position(|w| w == b"<x:xmpmeta")
                .unwrap();
            let end = bytes[start..]
                .windows(b"</x:xmpmeta>".len())
                .position(|w| w == b"</x:xmpmeta>")
                .unwrap()
                + start
                + b"</x:xmpmeta>".len();
            let xmp = std::str::from_utf8(&bytes[start..end]).unwrap();
            // Parse only the MotionPhoto item's directory attributes, independently
            // of the source clip and of JPEG/AVIF container offsets.
            let motion = xmp
                .split("<Container:Item ")
                .map(|item| item.split("/>").next().unwrap())
                .find(|item| item.contains("Item:Semantic=\"MotionPhoto\""))
                .unwrap();
            let length: usize = motion
                .split("Item:Length=\"")
                .nth(1)
                .unwrap()
                .split('"')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert!(length > 0 && length < bytes.len());
            let extracted = root.join(format!("recovered-{ext}.mp4"));
            let recovered = &bytes[bytes.len() - length..];
            assert_eq!(&recovered[4..8], b"ftyp");
            std::fs::write(&extracted, recovered).unwrap();
            let decode = std::process::Command::new("timeout")
                .args(["15s", "ffmpeg", "-v", "error", "-xerror", "-i"])
                .arg(&extracted)
                .args(["-map", "0", "-f", "null", "-"])
                .output()
                .unwrap();
            assert!(
                decode.status.success(),
                "{} + {ext}: {}",
                clip.display(),
                String::from_utf8_lossy(&decode.stderr)
            );
            image::open(&packaged).unwrap();
            eprintln!(
                "packaged {} + {ext}: recovered {length} bytes via MotionPhoto Item:Length; ffmpeg decoded all streams",
                clip.display()
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    fn software_encoder() -> EncoderInfo {
        EncoderInfo {
            element_name: "x264enc".into(),
            display_name: "H.264".into(),
            codec: VideoCodec::H264,
            is_hardware: false,
            priority: 0,
        }
    }

    #[test]
    fn live_clip_appsrc_drops_backlog_instead_of_retaining_unbounded_frames() {
        let path = std::env::temp_dir().join(format!(
            "motion-photo-buffer-test-{}.mp4",
            uuid::Uuid::new_v4()
        ));
        let encoder = ClipEncoder::new(
            &path,
            &software_encoder(),
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
        )
        .unwrap();
        assert_ne!(
            encoder
                .source
                .property_value("leaky-type")
                .serialize()
                .unwrap()
                .as_str(),
            "none"
        );
        assert_eq!(encoder.source.property::<u64>("max-buffers"), 3);
        drop(encoder);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn delayed_start_keeps_audio_on_the_video_shutter_timeline() {
        check_delayed_start_timestamps(false);
    }

    #[test]
    fn startup_buffered_frame_keeps_its_lead_over_later_audio() {
        check_delayed_start_timestamps(true);
    }

    fn check_delayed_start_timestamps(buffered: bool) {
        let origin = std::time::Instant::now();
        let buffered_pts = origin.elapsed().as_nanos() as u64;
        std::thread::sleep(std::time::Duration::from_millis(600));
        let path =
            std::env::temp_dir().join(format!("motion-photo-clock-{}.mp4", uuid::Uuid::new_v4()));
        let mut encoder =
            ClipEncoder::new(&path, &software_encoder(), 64, 48, 10, 800, None, origin).unwrap();
        let muxer = encoder
            .pipeline
            .iterate_elements()
            .into_iter()
            .flatten()
            .find(|e| e.factory().is_some_and(|f| f.name() == "mp4mux"))
            .unwrap();
        let audio = gst::ElementFactory::make("audiotestsrc")
            .property("is-live", true)
            .build()
            .unwrap();
        let audio_started = origin.elapsed().as_secs_f64();
        add_aac_branch(&encoder.pipeline, &muxer, &audio).unwrap();
        encoder.audio_source = Some(audio);
        let mut captured = Vec::new();
        if buffered {
            captured.push(buffered_pts as f64 / 1e9);
            encoder
                .push(
                    image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90])),
                    buffered_pts,
                )
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        for _ in 0..6 {
            let pts = origin.elapsed().as_nanos() as u64;
            captured.push(pts as f64 / 1e9);
            encoder
                .push(
                    image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90])),
                    pts,
                )
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        encoder.finish().unwrap();
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_packets",
                "-show_entries",
                "packet=stream_index,pts_time",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(probe.status.success());
        let data: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        let packets = data["packets"].as_array().unwrap();
        let times = |stream| {
            packets
                .iter()
                .filter(|p| p["stream_index"] == stream)
                .map(|p| p["pts_time"].as_str().unwrap().parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        };
        let mut video = times(0);
        video.sort_by(f64::total_cmp);
        let audio = times(1);
        std::fs::remove_file(path).unwrap();
        eprintln!(
            "delayed-start packet PTS: video={video:?}, audio first={:?}",
            audio.first()
        );
        let expected_audio_lead = audio_started - captured[0];
        assert!(
            (audio[0] - video[0] - expected_audio_lead).abs() < 0.08,
            "audio and video must share the shutter epoch, preserving buffering offset: expected={expected_audio_lead}"
        );
        // mp4mux may normalize the common epoch to the earliest stream PTS.
        let first_capture = captured[0];
        assert_eq!(video.len(), captured.len());
        for (actual, expected) in video.iter().zip(captured) {
            assert!(
                (actual - (expected - first_capture)).abs() < 0.03,
                "actual capture timing must survive startup"
            );
        }
    }

    #[test]
    fn synthetic_audio_is_aac_in_mp4_without_opening_a_microphone() {
        let path = std::env::temp_dir().join(format!(
            "motion-photo-aac-test-{}.mp4",
            uuid::Uuid::new_v4()
        ));
        let mut encoder = ClipEncoder::new(
            &path,
            &software_encoder(),
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
        )
        .unwrap();
        let muxer = encoder
            .pipeline
            .iterate_elements()
            .into_iter()
            .flatten()
            .find(|e| e.factory().is_some_and(|f| f.name() == "mp4mux"))
            .unwrap();
        let audio = gst::ElementFactory::make("audiotestsrc")
            .property("is-live", true)
            .build()
            .unwrap();
        add_aac_branch(&encoder.pipeline, &muxer, &audio).unwrap();
        audio.sync_state_with_parent().unwrap();
        encoder.audio_source = Some(audio);
        for i in 0..6 {
            encoder
                .push(
                    image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90])),
                    i * 100_000_000,
                )
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        encoder.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.windows(4).any(|x| x == b"mp4a"));
        assert!(!bytes.windows(4).any(|x| x == b"Opus"));
        let probe = std::process::Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(probe.status.success());
        let streams: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        assert_eq!(streams["streams"][0]["sample_rate"], "48000");
        assert_eq!(streams["streams"][0]["channels"], 1);
        check_packaged_clip_decodes_in_both_still_formats(&path);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn selected_hevc_and_av1_remain_in_mp4_containers() {
        for (name, codec, sample) in [
            ("x265enc", VideoCodec::HEVC, b"hvc1"),
            ("av1enc", VideoCodec::AV1, b"av01"),
        ] {
            let path = std::env::temp_dir().join(format!(
                "motion-photo-codec-test-{}.mp4",
                uuid::Uuid::new_v4()
            ));
            let info = EncoderInfo {
                element_name: name.into(),
                codec,
                ..software_encoder()
            };
            let mut encoder = ClipEncoder::new(
                &path,
                &info,
                64,
                48,
                10,
                800,
                None,
                std::time::Instant::now(),
            )
            .unwrap();
            for i in 0..3 {
                encoder
                    .push(
                        image::RgbImage::from_pixel(64, 48, image::Rgb([60, 40, 90])),
                        i * 100_000_000,
                    )
                    .unwrap();
            }
            encoder.finish().unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(&bytes[4..8], b"ftyp");
            assert!(bytes.windows(4).any(|x| x == sample));
            check_packaged_clip_decodes_in_both_still_formats(&path);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn v4l2_without_bitrate_controls_fails_instead_of_silently_ignoring_preset() {
        gst::init().unwrap();
        let element = gst::ElementFactory::make("identity").build().unwrap();
        let info = EncoderInfo {
            element_name: "v4l2h264enc".into(),
            ..software_encoder()
        };
        let result = configure_clip_video_encoder(&element, &info, 64, 48, 800);
        assert!(matches!(result, Err(ref e) if e.contains("bitrate")));
    }

    #[test]
    fn incompatible_explicit_encoder_is_rejected_without_substitution() {
        let mut info = software_encoder();
        info.element_name = "vp9enc".into();
        let path = std::env::temp_dir().join("must-not-create-motion-photo.mp4");
        let result = ClipEncoder::new(
            &path,
            &info,
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
        );
        assert!(matches!(result, Err(ref e) if e.contains("AVC, HEVC or AV1")));
    }

    #[test]
    fn synthetic_frames_produce_a_real_mp4_with_selected_encoder() {
        gstreamer::init().unwrap();
        let path =
            std::env::temp_dir().join(format!("motion-photo-test-{}.mp4", uuid::Uuid::new_v4()));
        let mut encoder = ClipEncoder::new(
            &path,
            &software_encoder(),
            64,
            48,
            10,
            800,
            None,
            std::time::Instant::now(),
        )
        .unwrap();
        for i in 0..6 {
            let image = image::RgbImage::from_pixel(64, 48, image::Rgb([i * 30, 40, 90]));
            encoder.push(image, u64::from(i) * 100_000_000).unwrap();
        }
        encoder.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
        assert!(bytes.windows(4).any(|x| x == b"moov"));
        assert!(bytes.windows(4).any(|x| x == b"avc1"));
        std::fs::remove_file(path).unwrap();
    }
}
