// SPDX-License-Identifier: GPL-3.0-only
//! Native scheduling scope regression through the recorder's actual constructor.
use super::*;
use crate::app::FilterType;
use crate::backends::camera::types::{FrameData, FrameMetadata, PixelFormat, RawLayout};
use crate::shaders::DemosaicMethod;
use crate::test_gpu::PhotoTrace as Completions;
use tracing_subscriber::layer::SubscriberExt;

fn frame(format: PixelFormat, packed: bool) -> Arc<CameraFrame> {
    let (width, height) = (513u32, 1057u32);
    let stride = if packed {
        width.div_ceil(4) * 5 + 7
    } else if format.is_bayer() {
        width + 7
    } else {
        width.div_ceil(2) * 4 + 8
    };
    let mut data = vec![0xef; (stride * height) as usize];
    for y in 0..height {
        if packed {
            for group in 0..width.div_ceil(4) {
                let mut low = 0;
                for i in 0..4 {
                    let value = (group * 4 * 977 + i * 977 + y * 577 + 257) % 1024;
                    data[(y * stride + group * 5 + i) as usize] = (value >> 2) as u8;
                    low |= ((value & 3) as u8) << (i * 2);
                }
                data[(y * stride + group * 5 + 4) as usize] = low;
            }
        } else {
            for x in 0..if format.is_bayer() {
                width
            } else {
                width.div_ceil(2) * 4
            } {
                data[(y * stride + x) as usize] = ((x * 37 + y * 13 + 19) % 256) as u8;
            }
        }
    }
    Arc::new(CameraFrame {
        width,
        height,
        stride,
        format,
        data: FrameData::from_owned_vec(data),
        yuv_planes: None,
        captured_at: Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: format.is_bayer().then_some(FrameMetadata {
            raw_layout: Some(if packed {
                RawLayout::Csi2 { bits: 10 }
            } else {
                RawLayout::U8
            }),
            // Also exercise full-sensor GPU AWB (no supplied gains).
            ..Default::default()
        }),
    })
}

fn check_continuous(format: PixelFormat, filter: FilterType) {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("continuous_motion_photo_gpu_pacing");
        return;
    };
    let trace = Completions::default();
    let subscriber = tracing_subscriber::registry().with(trace.clone());
    let _subscriber = tracing::subscriber::set_default(subscriber);
    pollster::block_on(async {
        let mut leaked = Vec::new();
        for packed in if format.is_bayer() {
            vec![false, true]
        } else {
            vec![false]
        } {
            let frame = frame(format, packed);
            for crop_rect in [None, Some((3, 489, 501, 541))] {
                let config = PostProcessingConfig {
                    demosaic_method: DemosaicMethod::Bilinear,
                    filter_type: filter,
                    crop_rect,
                    ..Default::default()
                };
                let continuous = continuous_processor(config.clone());
                let still = PostProcessor::new(config);
                trace.0.lock().unwrap().clear();
                let rgba = continuous.gpu_rgba_for_test(&frame).await.unwrap();
                let rgb = continuous.process(frame.clone()).await.unwrap();
                let completions = trace.completed_stages();
                if !completions.is_empty() {
                    leaked.push((packed, crop_rect, completions));
                }
                trace.0.lock().unwrap().clear();
                let expected_rgba = still.gpu_rgba_for_test(&frame).await.unwrap();
                let expected_rgb = still.process(frame.clone()).await.unwrap();
                assert!(
                    rgba == expected_rgba,
                    "exact RGBA: {format:?} {filter:?} packed={packed} crop={crop_rect:?}"
                );
                assert_eq!(rgb.image, expected_rgb.image, "exact RGB");
                assert_eq!(
                    (rgb.width, rgb.height),
                    (expected_rgb.width, expected_rgb.height)
                );
                let completions = trace.completed_stages();
                assert!(
                    completions
                        .iter()
                        .all(|stage| !stage.starts_with("filter-")),
                    "neither still nor continuous filters/preblur may be paced: {completions:?}"
                );
                if format.is_bayer() {
                    // Both RGBA and RGB calls complete three full-sensor bands,
                    // or two cropped bands for the unfiltered ROI.
                    let expected = if filter == FilterType::Standard && crop_rect.is_some() {
                        4
                    } else {
                        6
                    };
                    assert_eq!(
                        completions.iter().filter(|s| *s == "bilinear").count(),
                        expected,
                        "still Bayer must retain demosaic bands: {completions:?}"
                    );
                } else {
                    assert!(
                        completions.is_empty(),
                        "still YUV must be unpaced: {completions:?}"
                    );
                }
                assert_eq!(
                    completions.iter().filter(|s| *s == "unpack").count(),
                    if packed { 6 } else { 0 },
                    "RAW unpack uses full-sensor bands in both calls: {completions:?}"
                );

                // Reuse the SAME continuous processor after a paced call on the
                // shared converters: no mutable global policy may leak into it.
                trace.0.lock().unwrap().clear();
                assert!(continuous.gpu_rgba_for_test(&frame).await.unwrap() == rgba);
                let completions = trace.completed_stages();
                if !completions.is_empty() {
                    leaked.push((packed, crop_rect, completions));
                }
            }
        }
        assert!(
            leaked.is_empty(),
            "continuous Motion Photo must emit zero Photo GPU batch completion events; leaked={leaked:?}"
        );
    });
}

#[test]
fn continuous_motion_photo_bilinear_is_unpaced_and_exact() {
    check_continuous(PixelFormat::BayerRGGB, FilterType::Standard);
}

#[test]
fn continuous_factory_retains_synthetic_legacy_range_after_an_ordinary_still() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "continuous_factory_retains_synthetic_legacy_range_after_an_ordinary_still",
        );
        return;
    };
    let f = crate::test_gpu::synthetic_highlight_frame();
    pollster::block_on(async {
        let cfg = PostProcessingConfig {
            demosaic_method: DemosaicMethod::Bilinear,
            ..Default::default()
        };
        let continuous = continuous_processor(cfg.clone());
        let before = continuous.process(f.clone()).await.unwrap().image;
        assert_eq!(before.get_pixel(13, 16).0, [246, 117, 249]);
        let still = PostProcessor::new(cfg)
            .process(f.clone())
            .await
            .unwrap()
            .image;
        assert_eq!(still.get_pixel(13, 16).0, [228, 137, 249]);
        assert_eq!(before, continuous.process(f).await.unwrap().image);
    });
}

#[test]
fn continuous_motion_photo_filtered_bayer_is_unpaced_and_exact() {
    check_continuous(PixelFormat::BayerBGGR, FilterType::from_gpu_filter_code(14));
}

#[test]
fn continuous_motion_photo_filtered_yuv_is_unpaced_and_exact() {
    check_continuous(PixelFormat::YUYV, FilterType::from_gpu_filter_code(14));
}

#[test]
fn memory_recorder_uses_tested_continuous_constructor() {
    // Supplemental wiring check: the native tests above execute this factory.
    // Keep the live recorder from silently reverting to the still constructor.
    let source = include_str!("motion_photo.rs");
    let recorder = source.split("fn run_memory_recorder(").nth(1).unwrap();
    assert!(recorder.contains("let processor = continuous_processor(processing);"));
    assert!(!recorder.contains("let processor = PostProcessor::new(processing)"));
}
