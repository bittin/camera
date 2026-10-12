// SPDX-License-Identifier: GPL-3.0-only
//! Temporary compatibility for PipeWire's pre-1.7.0 contiguous-plane metadata bug.

use gst::prelude::*;
use gstreamer as gst;
use gstreamer_video as video;
use std::sync::atomic::{AtomicBool, Ordering};

fn affected_version(version: &str) -> bool {
    let core = version.split(['-', '+']).next().unwrap_or_default();
    let parts: Option<Vec<u32>> = core
        .split('.')
        .map(|p| {
            (!p.is_empty() && p.bytes().all(|c| c.is_ascii_digit()))
                .then(|| p.parse().ok())
                .flatten()
        })
        .collect();
    matches!(parts.as_deref(), Some([major, minor, patch]) if (*major, *minor, *patch) < (1, 7, 0))
}

struct PlaneLayout {
    planes: usize,
    strides: [i32; 4],
    offsets: [usize; 4],
}

fn corrected_layout(buffer: &gst::BufferRef) -> Option<PlaneLayout> {
    // The producer bug concerns one contiguous memory block, not separate planes.
    if buffer.n_memory() != 1 || buffer.iter_meta::<video::VideoMeta>().count() != 1 {
        return None;
    }
    let meta = buffer.meta::<video::VideoMeta>()?;
    if !matches!(
        meta.format(),
        video::VideoFormat::Nv12
            | video::VideoFormat::Nv21
            | video::VideoFormat::I420
            | video::VideoFormat::Yv12
    ) || meta.id() != 0
    {
        return None;
    }
    let info = video::VideoInfo::builder(meta.format(), meta.width(), meta.height())
        .build()
        .ok()?;
    let planes = info.n_planes() as usize;
    if meta.n_planes() != info.n_planes()
        || meta.offset() != info.offset()
        || meta.stride()[1..] != info.stride()[1..]
        || meta.stride()[0] <= info.stride()[0]
        || meta.stride()[0] % 2 != 0
    {
        return None;
    }
    // Restricted to the four linear 8-bit 4:2:0 formats above. This matches
    // gst_video_format_info_extrapolate_stride without raising our GStreamer
    // minimum from 1.20 to 1.22. Chroma height rounds up for odd heights.
    let mut strides = [0i32; 4];
    let mut offsets = [0usize; 4];
    let mut size = 0usize;
    for plane in 0..planes {
        strides[plane] = if plane == 0 || planes == 2 {
            meta.stride()[0]
        } else {
            meta.stride()[0] / 2
        };
        offsets[plane] = size;
        let height = info.format_info().scale_height(plane as u8, info.height()) as usize;
        let end = (strides[plane] as usize)
            .checked_mul(height)
            .and_then(|bytes| size.checked_add(bytes))?;
        size = end;
    }
    // Reject truncation, trailing allocations and ambiguous nonstandard layouts.
    if size != buffer.size() {
        return None;
    }
    Some(PlaneLayout {
        planes,
        strides,
        offsets,
    })
}

fn apply_layout(buffer: &mut gst::BufferRef, layout: PlaneLayout) -> bool {
    let Some(mut meta) = buffer.meta_mut::<video::VideoMeta>() else {
        return false;
    };
    // SAFETY: the writable buffer exclusively owns this meta. Both arrays have
    // four entries and planes is validated against one of the supported formats.
    // Preserve all other meta fields (including mapping callbacks and flags).
    unsafe {
        let raw = &mut *meta.as_mut_ptr();
        raw.stride[..layout.planes].copy_from_slice(&layout.strides[..layout.planes]);
        raw.offset[..layout.planes].copy_from_slice(&layout.offsets[..layout.planes]);
    }
    true
}

pub(super) fn install(source: &gst::Element) -> Result<(), String> {
    let Some(factory) = source.factory().filter(|f| f.name() == "pipewiresrc") else {
        return Ok(());
    };
    let Some(plugin) = factory.plugin() else {
        return Ok(());
    };
    install_for_version(source, plugin.version().as_str()).map(|_| ())
}

fn repair_buffer(buffer: &mut gst::Buffer) -> bool {
    let Some(layout) = corrected_layout(buffer) else {
        return false;
    };
    // Leave correct or ambiguous shared buffers completely untouched.
    apply_layout(buffer.make_mut(), layout)
}

fn install_for_version(source: &gst::Element, version: &str) -> Result<bool, String> {
    if !affected_version(version) {
        return Ok(false);
    }
    let pad = source
        .static_pad("src")
        .ok_or("PipeWire source has no src pad")?;
    let version = version.to_owned();
    let logged = AtomicBool::new(false);
    pad.add_probe(gst::PadProbeType::BUFFER, move |_, probe| {
        if let Some(buffer) = probe.buffer_mut()
            && repair_buffer(buffer)
            && !logged.swap(true, Ordering::Relaxed)
        {
            tracing::warn!(plugin_version = %version,
                "Applied temporary PipeWire contiguous-plane metadata repair (upstream 95e89f786)");
        }
        gst::PadProbeReturn::Ok
    })
    .ok_or("Could not install PipeWire metadata probe")?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repair(buffer: &mut gst::BufferRef) -> bool {
        corrected_layout(buffer).is_some_and(|layout| apply_layout(buffer, layout))
    }

    fn padded_buffer(format: video::VideoFormat, height: u32) -> gst::Buffer {
        gst::init().unwrap();
        let info = video::VideoInfo::builder(format, 8, height)
            .build()
            .unwrap();
        let finfo = info.format_info();
        let strides: Vec<_> = (0..info.n_planes())
            .map(|p| {
                if p == 0 || info.n_planes() == 2 {
                    16
                } else {
                    8
                }
            })
            .collect();
        let size: usize = strides
            .iter()
            .enumerate()
            .map(|(p, s)| *s as usize * finfo.scale_height(p as u8, height) as usize)
            .sum();
        // Neutral image with poisoned row padding: stale chroma metadata must
        // visibly fail, rather than accidentally reading more neutral bytes.
        let mut pixels = vec![0u8; size];
        let mut offset = 0;
        for (plane, stride) in strides.iter().enumerate() {
            let rows = finfo.scale_height(plane as u8, height) as usize;
            let width = if plane == 0 || info.n_planes() == 2 {
                8
            } else {
                4
            };
            for row in 0..rows {
                let start = offset + row * *stride as usize;
                pixels[start..start + width].fill(128);
            }
            offset += *stride as usize * rows;
        }
        let mut buffer = gst::Buffer::from_mut_slice(pixels);
        let mut broken_strides = info.stride().to_vec();
        broken_strides[0] = 16;
        video::VideoMeta::add_full(
            buffer.get_mut().unwrap(),
            video::VideoFrameFlags::empty(),
            format,
            8,
            height,
            info.offset(),
            &broken_strides,
        )
        .unwrap();
        buffer
    }

    #[test]
    fn old_pipewire_repairs_padded_contiguous_planes() {
        for format in [
            video::VideoFormat::Nv12,
            video::VideoFormat::Nv21,
            video::VideoFormat::I420,
            video::VideoFormat::Yv12,
        ] {
            for height in [4, 5] {
                let mut buffer = padded_buffer(format, height);
                assert!(
                    repair(buffer.get_mut().unwrap()),
                    "{format:?}, height={height}"
                );
                let meta = buffer.meta::<video::VideoMeta>().unwrap();
                let info = video::VideoInfo::builder(format, 8, height)
                    .build()
                    .unwrap();
                let mut offset = 0;
                for plane in 0..info.n_planes() as usize {
                    let stride = if plane == 0 || info.n_planes() == 2 {
                        16
                    } else {
                        8
                    };
                    assert_eq!(meta.stride()[plane], stride);
                    assert_eq!(meta.offset()[plane], offset);
                    offset += stride as usize
                        * info.format_info().scale_height(plane as u8, height) as usize;
                }
                assert_eq!(offset, buffer.size());
                assert!(
                    !repair(buffer.get_mut().unwrap()),
                    "a second repair must be a no-op"
                );
            }
        }
    }

    #[test]
    fn version_gate_uses_numeric_components_and_fails_closed() {
        for version in ["0.3.85", "1.4.11", "1.6.9", "1.6.10", "1.6.9-distro1"] {
            assert!(affected_version(version), "{version}");
        }
        for version in [
            "1.7.0",
            "1.7.0-dev",
            "1.7.1",
            "1.10.0",
            "2.0.0",
            "",
            "unknown",
            "1.6",
            "1.6.x",
        ] {
            assert!(!affected_version(version), "{version}");
        }
    }

    #[test]
    fn ambiguous_layouts_and_unpadded_buffers_are_unchanged() {
        gst::init().unwrap();
        let mut missing = gst::Buffer::with_size(96).unwrap();
        assert!(!repair(missing.get_mut().unwrap()));
        let mut unpadded = gst::Buffer::with_size(48).unwrap();
        video::VideoMeta::add(
            unpadded.get_mut().unwrap(),
            video::VideoFrameFlags::empty(),
            video::VideoFormat::Nv12,
            8,
            4,
        )
        .unwrap();
        assert!(!repair(unpadded.get_mut().unwrap()));

        for extra in [-1isize, 1] {
            let mut buffer = padded_buffer(video::VideoFormat::Nv12, 4);
            let new_size = buffer.size().checked_add_signed(extra).unwrap();
            buffer.get_mut().unwrap().remove_all_memory();
            buffer
                .get_mut()
                .unwrap()
                .append_memory(gst::Memory::from_mut_slice(vec![0u8; new_size]));
            assert!(!repair(buffer.get_mut().unwrap()), "size mismatch {extra}");
        }
        let mut separate = padded_buffer(video::VideoFormat::Nv12, 4);
        separate.get_mut().unwrap().remove_all_memory();
        for size in [64, 32] {
            separate
                .get_mut()
                .unwrap()
                .append_memory(gst::Memory::from_mut_slice(vec![0u8; size]));
        }
        assert!(!repair(separate.get_mut().unwrap()));
        let mut custom = padded_buffer(video::VideoFormat::Nv12, 4);
        // Explicit plane layout, not the exact stale GstVideoInfo signature.
        unsafe {
            (*custom
                .get_mut()
                .unwrap()
                .meta_mut::<video::VideoMeta>()
                .unwrap()
                .as_mut_ptr())
            .offset[1] = 40;
        }
        assert!(!repair(custom.get_mut().unwrap()));
    }

    #[test]
    fn backported_correct_metadata_does_not_trigger_copy_on_write() {
        gst::init().unwrap();
        // Construct upstream-correct metadata independently of the repair.
        let mut buffer = gst::Buffer::with_size(96).unwrap();
        video::VideoMeta::add_full(
            buffer.get_mut().unwrap(),
            video::VideoFrameFlags::empty(),
            video::VideoFormat::Nv12,
            8,
            4,
            &[0, 64],
            &[16, 16],
        )
        .unwrap();
        let shared = buffer.clone();
        let original = buffer.as_ptr();
        assert!(!repair_buffer(&mut buffer));
        assert_eq!(
            buffer.as_ptr(),
            original,
            "correct shared buffers must not be cloned"
        );
        assert_eq!(
            shared.meta::<video::VideoMeta>().unwrap().offset(),
            &[0, 64]
        );
    }

    fn convert_fixture(version: &str, format: video::VideoFormat) -> Vec<u8> {
        gst::init().unwrap();
        let source = gstreamer_app::AppSrc::builder()
            .format(gst::Format::Time)
            .caps(
                &video::VideoInfo::builder(format, 8, 4)
                    .fps(gst::Fraction::new(30, 1))
                    .build()
                    .unwrap()
                    .to_caps()
                    .unwrap(),
            )
            .build();
        install_for_version(source.upcast_ref(), version).unwrap();
        let config = super::super::CameraFormat {
            width: 8,
            height: 4,
            pixel_format: format.to_string(),
            framerate: Some(super::super::Framerate::new(30, 1)),
            hardware_accelerated: false,
        };
        // Exercise the same capsfilter/videoconvert/appsink graph as hardware.
        let (pipeline, sink) =
            super::super::build_stream(source.clone().upcast(), &config).unwrap();
        pipeline.set_state(gst::State::Playing).unwrap();
        source.push_buffer(padded_buffer(format, 4)).unwrap();
        source.end_of_stream().unwrap();
        let sample = sink.try_pull_sample(gst::ClockTime::from_seconds(3));
        pipeline.set_state(gst::State::Null).unwrap();
        let frame =
            super::super::sample_to_frame(&sample.expect("converted sample"), None).unwrap();
        frame.data.to_vec()
    }

    #[test]
    fn repair_prevents_chroma_corruption_in_the_real_conversion_graph() {
        for format in [
            video::VideoFormat::Nv12,
            video::VideoFormat::Nv21,
            video::VideoFormat::I420,
            video::VideoFormat::Yv12,
        ] {
            let baseline = convert_fixture("1.7.0", format);
            assert!(
                baseline
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|p| p[0].abs_diff(p[1]) > 20 || p[1].abs_diff(p[2]) > 20),
                "fixture must catch unrepaired metadata: {format:?}"
            );
            let repaired = convert_fixture("1.6.9", format);
            assert!(
                repaired
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|p| p[0].abs_diff(p[1]) <= 2 && p[1].abs_diff(p[2]) <= 2),
                "neutral chroma must stay neutral: {format:?}"
            );
        }
    }
}
