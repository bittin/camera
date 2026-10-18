// SPDX-License-Identifier: GPL-3.0-only
//! Packed inputs use the ordinary unpack-to-texture photo route.
use super::*;
use crate::shaders::DemosaicMethod;

pub(super) fn fixture(
    layout: RawLayout,
    width: u32,
    height: u32,
    code: impl Fn(u32, u32) -> u32,
) -> (Vec<u8>, u32) {
    let (bits, group, bytes) = match layout {
        RawLayout::Csi2 { bits } => {
            let group = if bits == 12 { 2 } else { 4 };
            (bits, group, group * bits / 8)
        }
        RawLayout::Unpacked16 { bits } => (bits, 1, 2),
        _ => unreachable!(),
    };
    let stride = width.div_ceil(group) * bytes + 7;
    let mut data = vec![0xff; (stride * height) as usize];
    for y in 0..height {
        for start in (0..width).step_by(group as usize) {
            let base = (y * stride + start / group * bytes) as usize;
            let mut lows = 0u32;
            for pos in 0..group {
                let value = code(start + pos, y) & ((1 << bits) - 1);
                if matches!(layout, RawLayout::Unpacked16 { .. }) {
                    data[base..base + 2].copy_from_slice(&(value as u16).to_le_bytes());
                } else {
                    data[base + pos as usize] = (value >> (bits - 8)) as u8;
                    lows |= (value & ((1 << (bits - 8)) - 1)) << (pos * (bits - 8));
                }
            }
            if matches!(layout, RawLayout::Csi2 { .. }) {
                for pos in 0..bytes - group {
                    data[base + (group + pos) as usize] = (lows >> (8 * pos)) as u8;
                }
            }
        }
    }
    (data, stride)
}

pub(super) fn input(
    data: &[u8],
    layout: RawLayout,
    width: u32,
    height: u32,
    stride: u32,
) -> GpuFrameInput<'_> {
    GpuFrameInput {
        width,
        height,
        format: PixelFormat::BayerRGGB,
        y_data: data,
        y_stride: stride,
        uv_data: None,
        uv_stride: 0,
        uv_dimensions: None,
        yuv_color: Default::default(),
        v_data: None,
        v_stride: 0,
        colour_gains: Some([1.7, 1.3]),
        colour_correction_matrix: Some([[1.7, -0.4, -0.3], [-0.2, 1.4, -0.2], [-0.1, -0.3, 1.4]]),
        black_level: Some(0.02),
        sensor_black_levels: Some([0.02, 0.03, 0.04, 0.01]),
        sensor_white_levels: Some([0.97, 0.98, 0.99, 0.96]),
        raw_layout: Some(layout),
    }
}

async fn compare_routes(
    photo: &mut GpuConvertPipeline,
    generic: &mut GpuConvertPipeline,
    frozen: &mut GpuConvertPipeline,
    input: &GpuFrameInput<'_>,
) {
    let (w, h) = (input.width, input.height);
    generic
        .convert_bayer_with_method(input, DemosaicMethod::Bilinear)
        .unwrap();
    frozen
        .convert_photo_bayer_with_method(input, DemosaicMethod::Bilinear)
        .unwrap();
    // Full frame, odd sensor-origin ROI, last pixel, and empty ROI.
    for roi in [
        (0, 0, w, h),
        (1, 1, w - 1, h - 1),
        (w - 1, h - 1, 1, 1),
        (w, h, 0, 0),
    ] {
        photo
            .convert_photo_bayer_region_with_method(input, DemosaicMethod::Bilinear, roi)
            .unwrap();
        assert!(photo.tex_y.is_some());
        assert!(photo.bayer_linear_texture.is_none());
        assert!(photo.demosaic_engine.is_none());
        if matches!(input.raw_layout, Some(RawLayout::Csi2 { .. })) {
            assert!(photo.unpack_output_buffer.is_some());
        }
        let actual = photo.read_rgba_region_to_cpu(roi).await.unwrap();
        assert_eq!(actual.len(), (roi.2 * roi.3 * 4) as usize);
        assert!(
            actual
                .as_chunks::<4>()
                .0
                .iter()
                .all(|pixel| pixel[3] == 255)
        );
        assert_eq!(
            actual,
            generic.read_rgba_region_to_cpu(roi).await.unwrap(),
            "unfused {:?} {:?} gains={:?} roi={roi:?}",
            input.raw_layout,
            input.format,
            input.colour_gains
        );
        assert_eq!(
            actual,
            frozen.read_rgba_region_to_cpu(roi).await.unwrap(),
            "frozen {:?} {:?} gains={:?} roi={roi:?}",
            input.raw_layout,
            input.format,
            input.colour_gains
        );
    }
}

#[test]
fn packed_and_unpacked_photo_match_unfused_and_frozen() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("packed_and_unpacked_photo_match_unfused_and_frozen");
        return;
    };
    pollster::block_on(async {
        let mut photo = GpuConvertPipeline::new().await.unwrap();
        let mut generic = GpuConvertPipeline::new().await.unwrap();
        let mut frozen = GpuConvertPipeline::new().await.unwrap();
        frozen.install_roi_frozen_reference();
        for layout in [
            RawLayout::Csi2 { bits: 10 },
            RawLayout::Csi2 { bits: 12 },
            RawLayout::Csi2 { bits: 14 },
            RawLayout::Unpacked16 { bits: 16 },
        ] {
            for (w, h) in [(3, 5), (65, 19)] {
                let (data, stride) = fixture(layout, w, h, |x, y| x * 977 + y * 577);
                for format in [
                    PixelFormat::BayerRGGB,
                    PixelFormat::BayerBGGR,
                    PixelFormat::BayerGRBG,
                    PixelFormat::BayerGBRG,
                ] {
                    for gains in [None, Some([1.7, 1.3])] {
                        let mut i = input(&data, layout, w, h, stride);
                        i.format = format;
                        i.colour_gains = gains;
                        compare_routes(&mut photo, &mut generic, &mut frozen, &i).await;
                    }
                }
            }
        }
        let (data, stride) = fixture(RawLayout::Csi2 { bits: 10 }, 3, 5, |x, y| x + y);
        let mut i = input(&data, RawLayout::Csi2 { bits: 10 }, 3, 5, stride);
        i.colour_gains = Some([f32::NAN, 1.0]);
        assert!(matches!(
            photo.convert_photo_bayer_with_method(&i, DemosaicMethod::Bilinear),
            Err(BayerError::InvalidMetadata)
        ));
        i.colour_gains = Some([1.7, 1.3]);
        i.y_data = &data[..data.len() - 8]; // truncate the final active packed group, not just padding
        assert!(
            photo
                .convert_photo_bayer_with_method(&i, DemosaicMethod::Bilinear)
                .is_err()
        );
    });
}

fn raw10_stress_data(width: u32, height: u32, mut state: u32) -> (u32, Vec<u8>) {
    let stride = width.div_ceil(4) * 5 + 7;
    let mut data = vec![0xff; (stride * height) as usize];
    for y in 0..height {
        for group in 0..width.div_ceil(4) {
            let base = (y * stride + group * 5) as usize;
            let mut low = 0u8;
            // Advance through inactive tail pixels too: this is the historical stress sequence.
            for pos in 0..4 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let code = if state & 7 == 0 {
                    state & 1023
                } else {
                    state % 384
                };
                data[base + pos] = (code >> 2) as u8;
                low |= ((code & 3) as u8) << (pos * 2);
            }
            data[base + 4] = low;
        }
    }
    (stride, data)
}

#[test]
fn nonconstant_raw10_stress_preserves_padding_and_inactive_tail_sequence() {
    let (stride, data) = raw10_stress_data(5, 2, 0xDEADBEEF);
    assert_eq!((stride, data.len()), (17, 34));
    let decode = |y: usize, x: usize| {
        let base = y * stride as usize + x / 4 * 5;
        (u16::from(data[base + x % 4]) << 2) | u16::from((data[base + 4] >> ((x % 4) * 2)) & 3)
    };
    assert_eq!(
        (0..8).map(|x| decode(0, x)).collect::<Vec<_>>(),
        vec![183, 322, 88, 59, 840, 27, 220, 179]
    );
    assert_ne!(decode(0, 7), decode(1, 0));
    for y in 0..2 {
        assert!(data[y * 17 + 10..y * 17 + 17].iter().all(|&b| b == 0xff));
    }
}

#[test]
fn raw10_code257_and_nonconstant_regressions_match_unfused_and_frozen() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "raw10_code257_and_nonconstant_regressions_match_unfused_and_frozen",
        );
        return;
    };
    pollster::block_on(async {
        let mut photo = GpuConvertPipeline::new().await.unwrap();
        let mut generic = GpuConvertPipeline::new().await.unwrap();
        let mut frozen = GpuConvertPipeline::new().await.unwrap();
        frozen.install_roi_frozen_reference();
        for cohort in 0..4 {
            let layout = RawLayout::Csi2 { bits: 10 };
            let (width, height) = if cohort == 3 { (1025, 769) } else { (65, 33) };
            let (data, stride) = if cohort == 3 {
                let (stride, data) = raw10_stress_data(width, height, 0xDEADBEEF);
                (data, stride)
            } else {
                fixture(layout, 65, 33, |x, y| {
                    if x >= 65 {
                        0
                    } else if cohort == 0 {
                        (x * 37 + y * 53 + 19) % 1024
                    } else {
                        257
                    }
                })
            };
            let mut i = input(&data, layout, width, height, stride);
            if cohort >= 2 {
                // Synthetic signed CCM and unequal gains stress threshold rounding.
                i.colour_gains = Some([1.25, 2.75]);
                i.colour_correction_matrix = Some([
                    [1.75, -0.375, -0.375],
                    [-0.625, 2.25, -0.625],
                    [-0.125, -0.875, 2.],
                ]);
                i.black_level = Some(0.0625);
                i.sensor_black_levels = None;
                i.sensor_white_levels = None;
            }
            compare_routes(&mut photo, &mut generic, &mut frozen, &i).await;
            if cohort == 3 {
                i.colour_gains = None;
                compare_routes(&mut photo, &mut generic, &mut frozen, &i).await;
            }
        }
    });
}
