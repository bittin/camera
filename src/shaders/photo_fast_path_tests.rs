// SPDX-License-Identifier: GPL-3.0-only
use super::*;
use crate::backends::camera::types::{FrameData, FrameMetadata};
use crate::shaders::DemosaicMethod;

fn frame(format: PixelFormat) -> CameraFrame {
    CameraFrame {
        width: 21,
        height: 19,
        stride: 21,
        format,
        data: FrameData::from_owned_vec(
            (0..21 * 19).map(|i| ((i * 37 + 19) % 256) as u8).collect(),
        ),
        yuv_planes: None,
        captured_at: std::time::Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: Some(FrameMetadata {
            colour_gains: Some([1.7, 1.3]),
            colour_correction_matrix: Some([
                [1.7, -0.4, -0.3],
                [-0.2, 1.4, -0.2],
                [-0.1, -0.3, 1.4],
            ]),
            ..Default::default()
        }),
    }
}

#[test]
fn startup_warmup_compiles_photo_without_retaining_frame_allocations() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "startup_warmup_compiles_photo_without_retaining_frame_allocations",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        pipeline.warmup_pipelines();
        let photo = pipeline
            .photo_bilinear_pipeline
            .as_ref()
            .expect("startup must compile the fused photo shader")
            .pipeline
            .clone();
        assert!(pipeline.tex_y.is_none());
        assert!(pipeline.output_texture.is_none());
        assert!(pipeline.readback_staging_buffer.is_none());
        assert!(pipeline.packed_input_buffer.is_none());
        assert!(pipeline.unpack_output_buffer.is_none());
        let frame = frame(PixelFormat::BayerRGGB);
        let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
        {
            let mut workspace = pipeline.photo_workspace();
            workspace
                .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
                .unwrap();
            workspace.read_rgba_to_cpu(21, 19).await.unwrap();
        }
        pipeline.warmup_pipelines();
        assert_eq!(
            pipeline.photo_bilinear_pipeline.as_ref().unwrap().pipeline,
            photo
        );
        assert!(pipeline.tex_y.is_none());
        assert!(pipeline.output_texture.is_none());
        assert!(pipeline.readback_staging_buffer.is_none());
    });
}

#[test]
fn photo_programs_compile_lazily_and_survive_workspace_retirement() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "photo_programs_compile_lazily_and_survive_workspace_retirement",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        assert!(pipeline.photo_bilinear_pipeline.is_none());
        assert!(pipeline.debayer_pipeline.is_none());
        assert!(pipeline.debayer_finish_pipeline.is_none());
        let frame = frame(PixelFormat::BayerRGGB);
        let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
        pipeline
            .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
            .unwrap();
        assert!(pipeline.debayer_pipeline.is_none());
        assert!(pipeline.debayer_finish_pipeline.is_none());
        let texture = pipeline
            .photo_bilinear_pipeline
            .as_ref()
            .unwrap()
            .pipeline
            .clone();
        let mut generic = GpuConvertPipeline::new().await.unwrap();
        generic
            .convert_bayer_with_method(&input, DemosaicMethod::Bilinear)
            .unwrap();
        assert!(generic.photo_bilinear_pipeline.is_none());
        assert!(generic.debayer_pipeline.is_some());
        assert!(generic.debayer_finish_pipeline.is_some());
        let mut previous = None;
        for _ in 0..2 {
            let rgba = {
                let mut workspace = pipeline.photo_workspace();
                workspace
                    .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
                    .unwrap();
                workspace
                    .read_rgba_to_cpu(frame.width, frame.height)
                    .await
                    .unwrap()
            };
            if let Some(expected) = previous.replace(rgba.clone()) {
                assert_eq!(rgba, expected);
            }
            assert!(pipeline.output_texture.is_none());
            assert!(pipeline.tex_y.is_none());
            assert!(pipeline.bayer_linear_texture.is_none());
            assert!(pipeline.bayer_demosaic_texture.is_none());
            assert!(pipeline.readback_staging_buffer.is_none());
            assert_eq!((pipeline.cached_width, pipeline.cached_height), (0, 0));
            assert_eq!(
                pipeline.photo_bilinear_pipeline.as_ref().unwrap().pipeline,
                texture
            );
        }
    });
}

#[test]
fn photo_bilinear_avoids_full_resolution_float_workspace() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("photo_bilinear_avoids_full_resolution_float_workspace");
        return;
    };
    pollster::block_on(async {
        let frame = frame(PixelFormat::BayerRGGB);
        let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        pipeline
            .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
            .unwrap();
        assert!(
            pipeline.bayer_linear_texture.is_none(),
            "Bilinear photo retained a full-frame float colour surface"
        );
        assert!(pipeline.demosaic_engine.is_none());
        assert_eq!(
            pipeline.read_rgba_to_cpu(21, 19).await.unwrap().len(),
            21 * 19 * 4
        );
    });
}

#[test]
fn photo_bilinear_matches_unfused_output_and_releases_old_surfaces() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "photo_bilinear_matches_unfused_output_and_releases_old_surfaces",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        for format in [
            PixelFormat::BayerRGGB,
            PixelFormat::BayerBGGR,
            PixelFormat::BayerGRBG,
            PixelFormat::BayerGBRG,
        ] {
            for (width, height) in [(2, 2), (3, 5), (21, 19), (65, 33)] {
                for supplied_gains in [false, true] {
                    let mut frame = frame(format);
                    frame.width = width;
                    frame.height = height;
                    frame.stride = width + 7;
                    let mut data = vec![0xff; (frame.stride * height) as usize];
                    for y in 0..height {
                        for x in 0..width {
                            data[(y * frame.stride + x) as usize] =
                                ((x * 37 + y * 53 + 19) % 256) as u8;
                        }
                    }
                    frame.data = FrameData::from_owned_vec(data);
                    let mut input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                    input.raw_layout = Some(RawLayout::U8);
                    input.sensor_black_levels = Some([0.02, 0.03, 0.04, 0.01]);
                    input.sensor_white_levels = Some([0.97, 0.98, 0.99, 0.96]);
                    if !supplied_gains {
                        input.colour_gains = None;
                    }
                    pipeline
                        .convert_bayer_with_method(&input, DemosaicMethod::Bilinear)
                        .unwrap();
                    assert!(pipeline.bayer_linear_texture.is_some());
                    let reference = pipeline.read_rgba_to_cpu(width, height).await.unwrap();
                    pipeline
                        .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
                        .unwrap();
                    let actual = pipeline.read_rgba_to_cpu(width, height).await.unwrap();
                    assert_eq!(
                        actual, reference,
                        "{format:?} {width}x{height} gains={supplied_gains}"
                    );
                    assert!(pipeline.bayer_linear_texture.is_none());
                    assert!(pipeline.demosaic_engine.is_none());
                    let placeholder = pipeline.bayer_demosaic_texture.as_ref().unwrap();
                    assert_eq!((placeholder.width(), placeholder.height()), (1, 1));
                }
            }
        }
    });
}
