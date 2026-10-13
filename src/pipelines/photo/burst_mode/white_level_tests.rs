// SPDX-License-Identifier: GPL-3.0-only
//! Post-GREEN production characterization of explicit sensor endpoints.
use super::*;
use crate::backends::camera::types::{FrameData, FrameMetadata, RawLayout};

const SITES: [[usize; 4]; 4] = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]];
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::BayerRGGB,
    PixelFormat::BayerBGGR,
    PixelFormat::BayerGRBG,
    PixelFormat::BayerGBRG,
];

// Independent CSI-2 encoder: upper eight bits followed by contiguous little-endian low bits.
fn pack_row(codes: &[u16], layout: RawLayout) -> Vec<u8> {
    match layout {
        RawLayout::U8 => codes.iter().map(|v| *v as u8).collect(),
        RawLayout::Unpacked16 { .. } => codes.iter().flat_map(|v| v.to_le_bytes()).collect(),
        RawLayout::Csi2 { bits } => {
            let group = if bits == 12 { 2 } else { 4 };
            let low = bits - 8;
            let mut row = Vec::new();
            for chunk in codes.chunks(group) {
                let mut tails = 0u32;
                for (i, code) in chunk.iter().enumerate() {
                    row.push((code >> low) as u8);
                    tails |= (u32::from(*code) & ((1 << low) - 1)) << (i as u32 * low);
                }
                for i in 0..group * low as usize / 8 {
                    row.push((tails >> (i * 8)) as u8);
                }
            }
            row
        }
    }
}

fn fixture(pattern: usize, layout: RawLayout, targets: [f32; 4]) -> (CameraFrame, [f32; 4]) {
    let bits = match layout {
        RawLayout::U8 => 8,
        RawLayout::Unpacked16 { bits } | RawLayout::Csi2 { bits } => bits,
    };
    let max = ((1u32 << bits) - 1) as f32;
    let black = [0.04_f32, 0.08, 0.12, 0.16];
    let white = [0.52_f32, 0.63, 0.74, 0.85];
    let codes: [u16; 4] = std::array::from_fn(|s| {
        ((black[s] + targets[s] * (white[s] - black[s])).clamp(0.0, 1.0) * max).round() as u16
    });
    let expected = std::array::from_fn(|s| {
        ((codes[s] as f32 / max - black[s]) / (white[s] - black[s])).clamp(-0.14798465, 1.0)
    });
    let mut bytes = Vec::new();
    let mut stride = 0;
    for y in 0..4 {
        let row: Vec<u16> = (0..4)
            .map(|x| codes[SITES[pattern][(y % 2) * 2 + x % 2]])
            .collect();
        let mut packed = pack_row(&row, layout);
        packed.extend_from_slice(&[0xa5; 3]);
        stride = packed.len() as u32;
        bytes.extend(packed);
    }
    (
        CameraFrame {
            width: 4,
            height: 4,
            stride,
            format: FORMATS[pattern],
            data: FrameData::from_owned_vec(bytes),
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: Some(FrameMetadata {
                raw_layout: Some(layout),
                sensor_black_levels: Some(black),
                sensor_white_levels: Some(white),
                colour_gains: Some([1.7, 1.3]),
                colour_correction_matrix: Some([
                    [1.1, -0.1, 0.0],
                    [-0.1, 1.2, -0.1],
                    [0.0, -0.2, 1.2],
                ]),
                ..Default::default()
            }),
        },
        expected,
    )
}

fn layouts() -> Vec<RawLayout> {
    std::iter::once(RawLayout::U8)
        .chain((8..=16).map(|bits| RawLayout::Unpacked16 { bits }))
        .chain([10, 12, 14].map(|bits| RawLayout::Csi2 { bits }))
        .collect()
}

#[test]
fn explicit_white_levels_all_packing_cfa_gpu_hdr_reference() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("explicit_white_levels_all_packing_cfa_gpu_hdr_reference");
        return;
    };
    pollster::block_on(async {
        let gpu = BurstModeGpuPipeline::new().await.unwrap();
        let mut converter = crate::shaders::GpuConvertPipeline::new().await.unwrap();
        for layout in layouts() {
            for (pattern, sites) in SITES.iter().enumerate() {
                // Signed shadow, unequal green values, knee interior and sensor clipping
                // substantially below nominal full scale. No calibration is inferred.
                for targets in [
                    [-0.5; 4],
                    [-0.1, 0.2, 0.3, 0.4],
                    [0.4, 0.5, 0.6, 0.7],
                    [0.7, 0.9875, 0.9875, 0.9],
                    [0.8, 1.1, 1.1, 1.05],
                ] {
                    let (frame, normalized) = fixture(pattern, layout, targets);
                    let planes = extract_bayer_planes(&frame).unwrap();
                    for quad in planes.data.as_chunks::<4>().0 {
                        for (actual, s) in quad.iter().zip([0, 1, 3, 2]) {
                            assert!(
                                (*actual - normalized[s]).abs() < 2e-6,
                                "{layout:?} {pattern} site={s}: {actual} vs {}",
                                normalized[s]
                            );
                        }
                    }
                    let buffer = gpu.create_storage_buffer(
                        "explicit_white_reference",
                        (planes.data.len() * 4) as u64,
                    );
                    gpu.queue
                        .write_buffer(&buffer, 0, bytemuck::cast_slice(&planes.data));
                    let hdr = gpu
                        .demosaic_bayer_planes_with_method(
                            &buffer,
                            2,
                            2,
                            pattern as u32,
                            planes.colour_gains,
                            planes.colour_correction_matrix,
                            crate::shaders::DemosaicMethod::Bilinear,
                        )
                        .await
                        .unwrap();
                    let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                    assert_eq!(
                        input.sensor_white_levels,
                        frame
                            .libcamera_metadata
                            .as_ref()
                            .unwrap()
                            .sensor_white_levels
                    );
                    converter
                        .convert_bayer_with_method(&input, crate::shaders::DemosaicMethod::Bilinear)
                        .unwrap();
                    let linear = converter.read_bayer_linear_for_test().await.unwrap();
                    let encoded = converter.read_rgba_to_cpu(4, 4).await.unwrap();
                    for y in 0..4 {
                        for x in 0..4 {
                            let s = sites[(y % 2) * 2 + x % 2];
                            let green = if s == 1 || s == 2 {
                                normalized[s]
                            } else {
                                (normalized[1] + normalized[2]) * 0.5
                            };
                            let peak = if s == 1 || s == 2 {
                                normalized[0].max(green).max(normalized[3])
                            } else {
                                normalized.into_iter().fold(f32::NEG_INFINITY, f32::max)
                            };
                            let rgb = [normalized[0] * 1.7, green, normalized[3] * 1.3];
                            let clipped = rgb.map(|v| v.clamp(0.0, 1.0));
                            let t = ((peak - 0.980) / 0.015).clamp(0.0, 1.0);
                            let blend = t * t * (3.0 - 2.0 * t);
                            let ccm = frame
                                .libcamera_metadata
                                .as_ref()
                                .unwrap()
                                .colour_correction_matrix
                                .unwrap();
                            let expected: [f32; 3] = std::array::from_fn(|row| {
                                let original: f32 =
                                    ccm[row].iter().zip(rgb).map(|(m, v)| m * v).sum();
                                let capped: f32 =
                                    ccm[row].iter().zip(clipped).map(|(m, v)| m * v).sum();
                                original + blend * (capped - original)
                            });
                            let display_peak = expected.into_iter().fold(1.0_f32, f32::max);
                            for (c, expected_channel) in expected.iter().enumerate() {
                                let index = (y * 4 + x) * 4 + c;
                                for actual in [linear[index], hdr.data[index]] {
                                    assert!(
                                        (actual - expected_channel).abs() < 4e-5,
                                        "{layout:?} pattern={pattern} targets={targets:?} ({x},{y}) ch={c}: {actual} vs {}",
                                        expected[c]
                                    );
                                }
                                let v = (expected_channel / display_peak).clamp(0.0, 1.0);
                                let srgb = if v <= 0.0031308 {
                                    v * 12.92
                                } else {
                                    1.055 * v.powf(1.0 / 2.4) - 0.055
                                };
                                assert!(encoded[index].abs_diff((srgb * 255.0).round() as u8) <= 1);
                            }
                        }
                    }
                }
                let (mut frame, _) = fixture(pattern, layout, [0.3, 0.4, 0.5, 0.6]);
                frame
                    .libcamera_metadata
                    .as_mut()
                    .unwrap()
                    .sensor_white_levels = None;
                let nominal_planes = extract_bayer_planes(&frame).unwrap().data;
                converter
                    .convert(&GpuFrameInput::from_camera_frame(&frame).unwrap())
                    .unwrap();
                let nominal_linear = converter.read_bayer_linear_for_test().await.unwrap();
                frame
                    .libcamera_metadata
                    .as_mut()
                    .unwrap()
                    .sensor_white_levels = Some([1.0; 4]);
                assert_eq!(extract_bayer_planes(&frame).unwrap().data, nominal_planes);
                converter
                    .convert(&GpuFrameInput::from_camera_frame(&frame).unwrap())
                    .unwrap();
                assert_eq!(
                    converter.read_bayer_linear_for_test().await.unwrap(),
                    nominal_linear
                );
            }
        }
    });
}

#[test]
fn explicit_white_levels_invalid_metadata_fail_closed() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("explicit_white_levels_invalid_metadata_fail_closed");
        return;
    };
    pollster::block_on(async {
        let mut converter = crate::shaders::GpuConvertPipeline::new().await.unwrap();
        for site in 0..4 {
            for invalid in [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -0.1,
                1.01,
                0.0,
                [0.04, 0.08, 0.12, 0.16][site],
                [0.03, 0.07, 0.11, 0.15][site],
            ] {
                let (mut frame, _) = fixture(0, RawLayout::U8, [0.5; 4]);
                frame
                    .libcamera_metadata
                    .as_mut()
                    .unwrap()
                    .sensor_white_levels
                    .as_mut()
                    .unwrap()[site] = invalid;
                assert!(extract_bayer_planes(&frame).is_err());
                let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                assert_eq!(
                    converter.convert_bayer(&input).unwrap_err(),
                    crate::shaders::BayerError::InvalidMetadata
                );
                assert!(converter.convert(&input).is_err());
            }
        }
        let (mut frame, _) = fixture(0, RawLayout::U8, [0.5; 4]);
        let meta = frame.libcamera_metadata.as_mut().unwrap();
        meta.sensor_black_levels = Some([0.0; 4]);
        meta.sensor_white_levels = Some([f32::from_bits(1); 4]);
        assert!(extract_bayer_planes(&frame).is_err());
        assert_eq!(
            converter
                .convert_bayer(&GpuFrameInput::from_camera_frame(&frame).unwrap())
                .unwrap_err(),
            crate::shaders::BayerError::UnsafeArithmetic
        );
    });
}
