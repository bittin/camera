// SPDX-License-Identifier: GPL-3.0-only
//! Ordinary-still display range, independent of GPU scheduling.
use super::DemosaicMethod;
use crate::backends::camera::types::{
    CameraFrame, FrameData, FrameMetadata, PixelFormat, RawLayout,
};
use crate::pipelines::photo::{PostProcessingConfig, PostProcessor};
use std::{sync::Arc, time::Instant};

#[allow(clippy::too_many_arguments)]
fn frame(
    data: Vec<u8>,
    width: u32,
    height: u32,
    stride: u32,
    format: PixelFormat,
    gains: [f32; 2],
    ccm: [[f32; 3]; 3],
    black: f32,
) -> Arc<CameraFrame> {
    Arc::new(CameraFrame {
        data: FrameData::from_owned_vec(data),
        width,
        height,
        stride,
        format,
        yuv_planes: None,
        captured_at: Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: Some(FrameMetadata {
            raw_layout: Some(RawLayout::Csi2 { bits: 10 }),
            colour_gains: Some(gains),
            colour_correction_matrix: Some(ccm),
            black_level: Some(black),
            ..Default::default()
        }),
    })
}
fn pack(width: usize, height: usize, stride: usize, code: impl Fn(usize, usize) -> u16) -> Vec<u8> {
    let mut data = vec![0xa5; stride * height];
    for y in 0..height {
        for group in 0..width.div_ceil(4) {
            let base = y * stride + group * 5;
            data[base + 4] = 0;
            for k in 0..4 {
                let v = code((group * 4 + k).min(width - 1), y);
                data[base + k] = (v >> 2) as u8;
                data[base + 4] |= ((v & 3) as u8) << (k * 2);
            }
        }
    }
    data
}
fn processor(method: DemosaicMethod) -> PostProcessor {
    PostProcessor::new(PostProcessingConfig {
        demosaic_method: method,
        ..Default::default()
    })
}
#[test]
fn synthetic_ordinary_still_restores_sdr_range() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("synthetic_ordinary_still_restores_sdr_range");
        return;
    };
    let f = crate::test_gpu::synthetic_highlight_frame();
    let result = pollster::block_on(processor(DemosaicMethod::Bilinear).process(f)).unwrap();
    let actual = result.image.get_pixel(13, 16).0;
    assert!(
        actual
            .into_iter()
            .zip([228, 137, 249])
            .all(|(a, b)| a.abs_diff(b) <= 1),
        "synthetic SDR witness: {actual:?}; signed legacy output is [246,117,249]"
    );
}
#[test]
fn ordinary_still_clamps_each_signed_ccm_candidate_before_blending() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "ordinary_still_clamps_each_signed_ccm_candidate_before_blending",
        );
        return;
    };
    let raw = pack(17, 15, 28, |x, y| match (x % 2, y % 2) {
        (0, 0) => 409,
        (1, 1) => 511,
        _ => 972,
    });
    let f = frame(
        raw,
        17,
        15,
        28,
        PixelFormat::BayerRGGB,
        [3., 3.],
        [[0., -1., 1.5], [3., 0., -2.5], [0., 1., 0.]],
        0.,
    );
    let rgb = [409.0_f32 / 1023., 972.0 / 1023., 511.0 / 1023.];
    let t = (rgb[1] - 0.9) / 0.1;
    let w = t * t * (3. - 2. * t);
    let a = [
        -rgb[1] + 1.5 * rgb[2] * 3.,
        9. * rgb[0] - 7.5 * rgb[2],
        rgb[1],
    ];
    let b = [-rgb[1] + 1.5, 3. - 2.5, rgb[1]];
    let encode = |v: f32| {
        let v = v.clamp(0., 1.);
        ((if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1. / 2.4) - 0.055
        }) * 255.)
            .round() as u8
    };
    let expected = std::array::from_fn::<_, 3, _>(|k| {
        encode((1. - w) * a[k].clamp(0., 1.) + w * b[k].clamp(0., 1.))
    });
    let wrong = std::array::from_fn::<_, 3, _>(|k| encode((1. - w) * a[k] + w * b[k]));
    assert!(
        expected.iter().zip(wrong).any(|(a, b)| a.abs_diff(b) > 8),
        "fixture must distinguish clamp placement"
    );
    let result = pollster::block_on(processor(DemosaicMethod::Bilinear).process(f)).unwrap();
    let actual = result.image.get_pixel(8, 6).0;
    assert!(
        actual
            .into_iter()
            .zip(expected)
            .all(|(a, b)| a.abs_diff(b) <= 1),
        "clamp-before-blend {actual:?}, expected {expected:?}, blend-before-clamp {wrong:?}"
    );
}

#[test]
fn still_intent_is_independent_of_pacing_cfa_roi_and_shared_jobs() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "still_intent_is_independent_of_pacing_cfa_roi_and_shared_jobs",
        );
        return;
    };
    pollster::block_on(async {
        for method in [
            DemosaicMethod::Bilinear,
            DemosaicMethod::Rcd,
            DemosaicMethod::Lmmse,
        ] {
            for format in [
                PixelFormat::BayerRGGB,
                PixelFormat::BayerBGGR,
                PixelFormat::BayerGRBG,
                PixelFormat::BayerGBRG,
            ] {
                for supplied_gains in [true, false] {
                    let raw = pack(67, 65, 92, |x, y| {
                        ((x * 977 + y * 577 + x * y * 13 + 257) % 1024) as u16
                    });
                    let mut f = frame(
                        raw,
                        67,
                        65,
                        92,
                        format,
                        [2., 1.5],
                        [[2., -0.75, -0.25], [-0.5, 2., -0.5], [-0.25, -0.5, 1.75]],
                        0.0625,
                    );
                    if !supplied_gains {
                        Arc::make_mut(&mut f)
                            .libcamera_metadata
                            .as_mut()
                            .unwrap()
                            .colour_gains = None;
                    }
                    let input = super::GpuFrameInput::from_camera_frame(&f).unwrap();
                    let before = {
                        let mut guard = super::get_gpu_convert_pipeline().await.unwrap();
                        let p = guard.as_mut().unwrap();
                        p.convert_bayer_with_method(&input, method).unwrap();
                        p.read_rgba_to_cpu(67, 65).await.unwrap()
                    };
                    let full = processor(method).gpu_rgba_for_test(&f).await.unwrap();
                    assert_eq!(
                        full,
                        processor(method)
                            .without_gpu_pacing()
                            .gpu_rgba_for_test(&f)
                            .await
                            .unwrap(),
                        "pacing changed SDR {method:?}/{format:?}/{supplied_gains}"
                    );
                    let legacy = processor(method)
                        .for_continuous_frames()
                        .without_gpu_pacing();
                    assert_eq!(
                        before,
                        legacy.gpu_rgba_for_test(&f).await.unwrap(),
                        "legacy conversion changed"
                    );
                    assert_ne!(
                        full, before,
                        "fixture must expose quality intent, not only scheduling"
                    );
                    for roi in [(1, 3, 19, 17), (64, 62, 3, 3), (66, 64, 1, 1)] {
                        let cfg = PostProcessingConfig {
                            demosaic_method: method,
                            crop_rect: Some(roi),
                            ..Default::default()
                        };
                        let expected: Vec<u8> = (roi.1..roi.1 + roi.3)
                            .flat_map(|y| {
                                let k = ((y * 67 + roi.0) * 4) as usize;
                                full[k..k + roi.2 as usize * 4].iter().copied()
                            })
                            .collect();
                        assert_eq!(
                            expected,
                            PostProcessor::new(cfg.clone())
                                .gpu_rgba_for_test(&f)
                                .await
                                .unwrap(),
                            "full-sensor ROI {method:?}/{format:?}/{supplied_gains}/{roi:?}"
                        );
                        assert_eq!(
                            expected,
                            PostProcessor::new(cfg)
                                .without_gpu_pacing()
                                .gpu_rgba_for_test(&f)
                                .await
                                .unwrap()
                        );
                    }
                    // Same shared converter: reset former AWB padding every job.
                    assert_eq!(before, legacy.gpu_rgba_for_test(&f).await.unwrap());
                    let mut guard = super::get_gpu_convert_pipeline().await.unwrap();
                    let p = guard.as_mut().unwrap();
                    p.convert_bayer_with_method(&input, method).unwrap();
                    assert_eq!(before, p.read_rgba_to_cpu(67, 65).await.unwrap());
                }
            }
        }
    });
}
