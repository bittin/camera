// SPDX-License-Identifier: GPL-3.0-only
//! GPU pixel comparisons between regional and full-sensor production processing.
use super::*;

#[test]
fn bilinear_roi_matches_full_sensor_production() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("bilinear_roi_matches_full_sensor_production");
        return;
    };
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        use crate::backends::camera::types::{FrameData, FrameMetadata};
        use crate::shaders::{DemosaicMethod, GpuConvertPipeline};
        let (width, height) = (67u32, 65u32);
        for format in [
            PixelFormat::BayerRGGB,
            PixelFormat::BayerBGGR,
            PixelFormat::BayerGRBG,
            PixelFormat::BayerGBRG,
        ] {
            for gains in [Some([1.3, 0.9]), None] {
                // Nonuniform data outside the crop makes cropped-AWB errors visible.
                let frame = CameraFrame {
                    width,
                    height,
                    stride: width,
                    format,
                    data: FrameData::from_owned_vec(
                        (0..width * height)
                            .map(|i| {
                                ((i * 37 + (i / width) * 13 + (i % width) * (i / width)) % 251 + 1)
                                    as u8
                            })
                            .collect(),
                    ),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        colour_gains: gains,
                        ..Default::default()
                    }),
                };
                let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                let mut converter = GpuConvertPipeline::new().await.unwrap();
                let mut reference_converter = GpuConvertPipeline::new().await.unwrap();

                // This API always passes None to the resolver: no crop can leak
                // into the reference, unlike comparing two process() calls.
                let full = {
                    let mut workspace = reference_converter.photo_workspace();
                    workspace
                        .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
                        .unwrap();
                    workspace.read_rgba_to_cpu(width, height).await.unwrap()
                };
                for region in [
                    (1, 3, 9, 7),
                    (0, 0, 3, 3),
                    (64, 62, 3, 3),
                    (66, 64, 1, 1),
                    (1, 1, 66, 64),
                    (67, 65, 0, 0),
                ] {
                    let mut workspace = converter.photo_workspace();
                    workspace
                        .convert_photo_bayer_region_with_method(
                            &input,
                            DemosaicMethod::Bilinear,
                            region,
                        )
                        .unwrap();
                    let actual = workspace.read_rgba_region_to_cpu(region).await.unwrap();
                    let expected = PostProcessor::crop_rgba(
                        &full, width, height, region.0, region.1, region.2, region.3,
                    )
                    .unwrap();
                    assert_eq!(
                        actual, expected,
                        "{format:?} gains={gains:?} region={region:?}"
                    );
                }
                for (crop, zoom) in [
                    ((1, 3, 19, 17), 1.7),
                    ((51, 49, 16, 16), 1.5),
                    ((1, 1, 66, 64), 1.01),
                ] {
                    let config = PostProcessingConfig {
                        demosaic_method: DemosaicMethod::Bilinear,
                        filter_type: FilterType::Standard,
                        crop_rect: Some(crop),
                        zoom_level: zoom,
                        brightness: 0.04,
                        contrast: 1.1,
                        saturation: 0.9,
                        sharpening: true,
                    };
                    // Compare the production legacy paths. Ordinary SDR ROI
                    // coverage lives in still_sdr_tests for all methods.
                    let processor = PostProcessor::new(config).for_continuous_frames();
                    let reference = processor
                        .process(Arc::new(CameraFrame {
                            data: FrameData::from_owned_vec(full.clone()),
                            format: PixelFormat::RGBA,
                            stride: width * 4,
                            libcamera_metadata: None,
                            ..frame.clone()
                        }))
                        .await
                        .unwrap();
                    let actual = processor.process(Arc::new(frame.clone())).await.unwrap();
                    assert_eq!(
                        (actual.width, actual.height),
                        (reference.width, reference.height)
                    );
                    assert_eq!(
                        actual.image, reference.image,
                        "{format:?} gains={gains:?} crop={crop:?} zoom={zoom}"
                    );
                }
            }
        }
    });
}
