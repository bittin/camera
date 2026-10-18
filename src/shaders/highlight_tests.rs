// SPDX-License-Identifier: GPL-3.0-only
//! Synthetic sensor-clipping regressions with negative CCM coefficients.
//! No pixel samples or calibration values come from a camera capture.
use super::{DemosaicMethod, GpuConvertPipeline, GpuFrameInput};
use crate::backends::camera::types::{PixelFormat, RawLayout};

const GAINS: [f32; 2] = [2., 1.5];
const CCM: [[f32; 3]; 3] = [[2., -0.75, -0.25], [-0.5, 2., -0.5], [-0.25, -0.5, 1.75]];
fn input(data: &[u8], format: PixelFormat) -> GpuFrameInput<'_> {
    GpuFrameInput {
        width: 17,
        height: 15,
        format,
        y_data: data,
        y_stride: 40,
        uv_data: None,
        uv_stride: 0,
        uv_dimensions: None,
        yuv_color: Default::default(),
        v_data: None,
        v_stride: 0,
        colour_gains: Some(GAINS),
        colour_correction_matrix: Some(CCM),
        black_level: Some(0.),
        sensor_black_levels: None,
        sensor_white_levels: None,
        raw_layout: Some(RawLayout::Unpacked16 { bits: 10 }),
    }
}
fn bytes(code: impl Fn(usize, usize) -> u16) -> Vec<u8> {
    let mut data = vec![0xa5; 40 * 15];
    for y in 0..15 {
        for x in 0..17 {
            data[y * 40 + x * 2..y * 40 + x * 2 + 2].copy_from_slice(&code(x, y).to_le_bytes());
        }
    }
    data
}

#[test]
fn synthetic_highlight_has_no_cfa_recovery_island() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("synthetic_highlight_has_no_cfa_recovery_island");
        return;
    };
    pollster::block_on(async {
        let mut full = GpuConvertPipeline::new().await.unwrap();
        let mut regional = GpuConvertPipeline::new().await.unwrap();
        for method in [
            DemosaicMethod::Bilinear,
            DemosaicMethod::Rcd,
            DemosaicMethod::Lmmse,
        ] {
            let mut phase_pair = None;
            for (format, dx, dy) in [
                (PixelFormat::BayerRGGB, 0, 0),
                (PixelFormat::BayerBGGR, 1, 1),
                (PixelFormat::BayerGRBG, 1, 0),
                (PixelFormat::BayerGBRG, 0, 1),
            ] {
                let data = bytes(|x, y| {
                    let (x, y) = (x.saturating_sub(dx), y.saturating_sub(dy));
                    match (x % 2, y % 2) {
                        (0, 0) => 390,
                        (1, 1) => 950,
                        _ if (x, y) == (8, 5) => 1023,
                        _ => 1000,
                    }
                });
                let i = input(&data, format);
                full.convert_bayer_with_method(&i, method).unwrap();
                let roi = (3, 3, 11, 9);
                regional
                    .convert_photo_bayer_region_with_method(&i, method, roi)
                    .unwrap();
                let rgba = full.read_rgba_region_to_cpu(roi).await.unwrap();
                assert_eq!(
                    rgba,
                    regional.read_rgba_region_to_cpu(roi).await.unwrap(),
                    "ROI {method:?}/{format:?}"
                );
                let pixel = |x, y| {
                    let k = ((y - 3) * 11 + x - 3) * 4;
                    <[u8; 3]>::try_from(&rgba[k..k + 3]).unwrap()
                };
                let a = pixel(8 + dx, 6 + dy);
                let b = pixel(9 + dx, 6 + dy);
                // A single clipped green sample must not create a CFA island.
                // Advanced demosaicers may differ in red detail; Bilinear's
                // nearly constant reconstruction supports an all-channel bound.
                let jump = if method == DemosaicMethod::Bilinear {
                    a.iter().zip(b).map(|(a, b)| a.abs_diff(b)).max().unwrap()
                } else {
                    a[1].abs_diff(b[1])
                };
                assert!(
                    jump <= 12,
                    "CFA recovery island {method:?}/{format:?}: {a:?} -> {b:?} ({jump} codes)"
                );
                if method == DemosaicMethod::Bilinear {
                    if let Some(expected) = phase_pair {
                        assert_eq!((a, b), expected, "phase changed recovery");
                    }
                    phase_pair = Some((a, b));
                }
            }
        }
    });
}

#[test]
fn highlight_sensor_code_sweep_has_bounded_chroma_steps() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("highlight_sensor_code_sweep_has_bounded_chroma_steps");
        return;
    };
    pollster::block_on(async {
        let mut p = GpuConvertPipeline::new().await.unwrap();
        for (format, dx, dy) in [
            (PixelFormat::BayerRGGB, 0, 0),
            (PixelFormat::BayerBGGR, 1, 1),
            (PixelFormat::BayerGRBG, 1, 0),
            (PixelFormat::BayerGBRG, 0, 1),
        ] {
            let mut previous: Option<[u8; 3]> = None;
            for green in 900..=1023 {
                let data = bytes(|x, y| match ((x + dx) % 2, (y + dy) % 2) {
                    (0, 0) => 390,
                    (1, 1) => 950,
                    _ => green,
                });
                p.convert_photo_bayer_with_method(&input(&data, format), DemosaicMethod::Bilinear)
                    .unwrap();
                let rgba = p.read_rgba_region_to_cpu((8, 6, 1, 1)).await.unwrap();
                let rgb: [u8; 3] = rgba[..3].try_into().unwrap();
                if let Some(prev) = previous {
                    let jump = prev
                        .iter()
                        .zip(rgb)
                        .map(|(a, b)| a.abs_diff(b))
                        .max()
                        .unwrap();
                    assert!(
                        jump <= 4,
                        "one RAW10 code produced {jump} RGB8 codes at {green}: {prev:?} -> {rgb:?}"
                    );
                }
                previous = Some(rgb);
            }
        }
    });
}

#[test]
fn highlight_shoulder_has_bounded_recovery_slope() {
    let rgb = [0.375, 0.975, 0.925];
    let mut previous = super::apply_bayer_colour_reference(rgb, 0.9, GAINS, CCM);
    for code in 901..=1000 {
        let current = super::apply_bayer_colour_reference(rgb, code as f32 / 1000.0, GAINS, CCM);
        for channel in 0..3 {
            assert!(
                (current[channel] - previous[channel]).abs() < 0.02,
                ".001 sensor headroom step changed linear colour abruptly at {code}"
            );
        }
        previous = current;
    }
}
