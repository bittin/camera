// SPDX-License-Identifier: GPL-3.0-only
use super::*;
use crate::backends::camera::types::{FrameData, FrameMetadata};

fn frame(value: u8) -> CameraFrame {
    CameraFrame {
        width: 21,
        height: 19,
        stride: 21,
        format: PixelFormat::BayerRGGB,
        data: FrameData::from_owned_vec(vec![value; 21 * 19]),
        yuv_planes: None,
        captured_at: std::time::Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: Some(FrameMetadata {
            colour_gains: Some([1.0, 1.0]),
            raw_layout: Some(RawLayout::U8),
            ..Default::default()
        }),
    }
}

#[test]
fn prepared_photo_session_reuses_allocations_and_changed_pixels() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "prepared_photo_session_reuses_allocations_and_changed_pixels",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        let session = PhotoSession::new(21, 19, PixelFormat::BayerRGGB, RawLayout::U8, 21).unwrap();
        pipeline.ensure_resources(8, 6, PixelFormat::NV12, None);
        let preview_y = pipeline.tex_y.clone();
        let preview_output = pipeline.output_texture.clone();
        session.prepare(&mut pipeline).unwrap();
        assert_eq!(pipeline.tex_y, preview_y);
        assert_eq!(pipeline.output_texture, preview_output);
        assert_eq!((pipeline.cached_width, pipeline.cached_height), (8, 6));
        pipeline.release_frame_workspace();
        let mut identities = None;
        let mut previous = None;
        for value in [32, 192] {
            let frame = frame(value);
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let actual = {
                let mut workspace = pipeline.photo_workspace_for_session(
                    Some(Arc::clone(&session)),
                    &input,
                    super::super::DemosaicMethod::Bilinear,
                );
                let current = (
                    workspace.tex_y.clone().unwrap(),
                    workspace.output_texture.clone().unwrap(),
                    workspace.readback_staging_buffer.as_ref().unwrap().buffer(),
                );
                if let Some(expected) = &identities {
                    assert_eq!(&current, expected);
                }
                identities = Some(current);
                workspace
                    .convert_photo_bayer_region_with_method(
                        &input,
                        super::super::DemosaicMethod::Bilinear,
                        (1, 1, 7, 9),
                    )
                    .unwrap();
                let rgba = workspace
                    .read_rgba_region_to_cpu((1, 1, 7, 9))
                    .await
                    .unwrap();
                assert_eq!(rgba.len(), 7 * 9 * 4);
                workspace.complete();
                rgba
            };
            let mut reference = GpuConvertPipeline::new().await.unwrap();
            reference
                .convert_bayer_with_method(&input, super::super::DemosaicMethod::Bilinear)
                .unwrap();
            assert_eq!(
                actual,
                reference
                    .read_rgba_region_to_cpu((1, 1, 7, 9))
                    .await
                    .unwrap()
            );
            if let Some(previous) = previous.replace(actual.clone()) {
                assert_ne!(actual, previous);
            }
            assert!(pipeline.tex_y.is_none());
            assert!(session.has_workspace());
        }
        session.end();
        assert!(!session.has_workspace());
    });
}

#[test]
fn photo_session_errors_cancellation_and_end_do_not_retain_allocations() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "photo_session_errors_cancellation_and_end_do_not_retain_allocations",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        let frame = frame(64);
        let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
        let session = PhotoSession::new(21, 19, PixelFormat::BayerRGGB, RawLayout::U8, 21).unwrap();
        session.prepare(&mut pipeline).unwrap();
        drop(pipeline.photo_workspace_for_session(
            Some(Arc::clone(&session)),
            &input,
            super::super::DemosaicMethod::Bilinear,
        ));
        assert!(!session.has_workspace());
        assert!(pipeline.tex_y.is_none());
        session.prepare(&mut pipeline).unwrap();
        {
            let mut workspace = pipeline.photo_workspace_for_session(
                Some(Arc::clone(&session)),
                &input,
                super::super::DemosaicMethod::Bilinear,
            );
            session.end();
            workspace.complete();
        }
        assert!(!session.has_workspace());
        assert!(pipeline.tex_y.is_none());
        session.prepare(&mut pipeline).unwrap();
        assert!(
            !session.has_workspace(),
            "late background preparation must not revive a closed session"
        );
    });
}

#[test]
fn photo_session_rejects_invalid_and_unbounded_reservations() {
    assert!(PhotoSession::new(0, 19, PixelFormat::BayerRGGB, RawLayout::U8, 21).is_none());
    assert!(PhotoSession::new(21, 19, PixelFormat::BayerRGGB, RawLayout::U8, 1).is_none());
    assert!(
        PhotoSession::new(
            u32::MAX,
            u32::MAX,
            PixelFormat::BayerRGGB,
            RawLayout::U8,
            u32::MAX
        )
        .is_none()
    );
    assert!(
        PhotoSession::new(16384, 16384, PixelFormat::BayerRGGB, RawLayout::U8, 16384).is_none()
    );
}

#[test]
fn photo_session_packed_and_unpacked_reservations_reuse_exact_buffers() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "photo_session_packed_and_unpacked_reservations_reuse_exact_buffers",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        for layout in [
            RawLayout::Csi2 { bits: 10 },
            RawLayout::Csi2 { bits: 12 },
            RawLayout::Csi2 { bits: 14 },
            RawLayout::Unpacked16 { bits: 10 },
            RawLayout::Unpacked16 { bits: 16 },
        ] {
            let (_, stride) = super::packed_photo_tests::fixture(layout, 65, 19, |_, _| 0);
            let session =
                PhotoSession::new(65, 19, PixelFormat::BayerRGGB, layout, stride).unwrap();
            session.prepare(&mut pipeline).unwrap();
            let mut previous_buffers = None;
            for seed in [71, 193] {
                let (data, stride) = super::packed_photo_tests::fixture(layout, 65, 19, |x, y| {
                    (x * 173 + y * 79 + seed) * 17
                });
                let input = super::packed_photo_tests::input(&data, layout, 65, 19, stride);
                let actual = {
                    let mut workspace = pipeline.photo_workspace_for_session(
                        Some(Arc::clone(&session)),
                        &input,
                        super::super::DemosaicMethod::Bilinear,
                    );
                    let buffers = (
                        workspace.packed_input_buffer.clone(),
                        workspace.unpack_output_buffer.clone(),
                        workspace.readback_staging_buffer.as_ref().unwrap().buffer(),
                    );
                    if let Some(previous) = &previous_buffers {
                        assert_eq!(previous, &buffers);
                    }
                    previous_buffers = Some(buffers);
                    workspace
                        .convert_photo_bayer_region_with_method(
                            &input,
                            super::super::DemosaicMethod::Bilinear,
                            (3, 2, 9, 7),
                        )
                        .unwrap();
                    let result = workspace
                        .read_rgba_region_to_cpu((3, 2, 9, 7))
                        .await
                        .unwrap();
                    assert_eq!(result.len(), 9 * 7 * 4);
                    workspace.complete();
                    result
                };
                let mut reference = GpuConvertPipeline::new().await.unwrap();
                reference
                    .convert_bayer_with_method(&input, super::super::DemosaicMethod::Bilinear)
                    .unwrap();
                assert_eq!(
                    actual,
                    reference
                        .read_rgba_region_to_cpu((3, 2, 9, 7))
                        .await
                        .unwrap(),
                    "{layout:?}"
                );
                assert!(session.has_workspace());
            }
        }
    });
}

#[test]
fn photo_session_mismatch_and_processing_error_retire_parked_resources() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "photo_session_mismatch_and_processing_error_retire_parked_resources",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        let frame = frame(64);
        let mut input = GpuFrameInput::from_camera_frame(&frame).unwrap();
        let session = PhotoSession::new(21, 19, PixelFormat::BayerRGGB, RawLayout::U8, 21).unwrap();
        session.prepare(&mut pipeline).unwrap();
        input.colour_gains = Some([f32::NAN, 1.0]);
        {
            let mut workspace = pipeline.photo_workspace_for_session(
                Some(Arc::clone(&session)),
                &input,
                super::super::DemosaicMethod::Bilinear,
            );
            assert!(
                workspace
                    .convert_photo_bayer_with_method(&input, super::super::DemosaicMethod::Bilinear)
                    .is_err()
            );
        }
        assert!(!session.has_workspace());
        for kind in 0..4 {
            session.prepare(&mut pipeline).unwrap();
            let mut mismatched = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut method = super::super::DemosaicMethod::Bilinear;
            match kind {
                0 => mismatched.width -= 1,
                1 => mismatched.y_stride += 1,
                2 => mismatched.raw_layout = Some(RawLayout::Unpacked16 { bits: 10 }),
                _ => method = super::super::DemosaicMethod::Rcd,
            }
            drop(pipeline.photo_workspace_for_session(
                Some(Arc::clone(&session)),
                &mismatched,
                method,
            ));
            assert!(!session.has_workspace());
        }
    });
}

#[test]
fn photo_session_old_guard_and_job_cannot_invalidate_replacement_registration() {
    let old_guard = PhotoSession::start(21, 19, PixelFormat::BayerRGGB, RawLayout::U8, 21).unwrap();
    let old_job = current_photo_session().unwrap();
    let new_guard = PhotoSession::start(21, 19, PixelFormat::BayerRGGB, RawLayout::U8, 21).unwrap();
    let new_session = current_photo_session().unwrap();
    assert!(!Arc::ptr_eq(&old_job, &new_session));
    drop(old_guard);
    assert!(Arc::ptr_eq(&current_photo_session().unwrap(), &new_session));
    old_job.end();
    drop(old_job);
    assert!(Arc::ptr_eq(&current_photo_session().unwrap(), &new_session));
    drop(new_guard);
    assert!(current_photo_session().is_none());
    assert!(!new_session.has_workspace());
}
