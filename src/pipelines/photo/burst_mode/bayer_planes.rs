// SPDX-License-Identifier: GPL-3.0-only

//! Bayer plane extraction from raw sensor frames.
//!
//! HDR+ paper §5: "We merge each Bayer color channel independently." This
//! module turns a `CameraFrame` carrying raw sensor data (8/10/12/14/16 bit,
//! optionally CSI-2 packed) into four half-resolution colour planes packed as
//! RGBA f32 for the GPU compute pipeline.
//!
//! Pure CPU code — no GPU dependencies — so it is independently testable.

use crate::backends::camera::types::{
    BayerLevels, CameraFrame, RawLayout, checked_plane, validate_bayer_levels,
};
use tracing::debug;

/// Bayer pattern sub-pixel offsets within a 2×2 quad.
///
/// Each field is `(dx, dy)` giving the column and row offset of that colour
/// channel inside the repeating 2×2 Bayer tile.
struct BayerOffsets {
    r: (usize, usize),
    gr: (usize, usize),
    gb: (usize, usize),
    b: (usize, usize),
}

/// Result of extracting Bayer color planes from raw sensor data.
///
/// Per the HDR+ paper (Section 5), the merge operates on each Bayer color plane
/// independently. We extract R, Gr, B, Gb planes at half resolution and pack them
/// as RGBA f32 for efficient GPU processing with existing vec4 shader infrastructure.
pub(crate) struct BayerPlanes {
    /// Black-to-white normalized RGBA f32: R=red, G=green_r, B=blue, A=green_b.
    /// Source endpoints have already been applied exactly once and are deliberately
    /// not retained for finishing: merging/finishing must not normalize again.
    /// Dimensions: (width/2) × (height/2) × 4 floats
    pub data: Vec<f32>,
    /// Width of each plane (half of full Bayer width)
    pub width: u32,
    /// Height of each plane (half of full Bayer height)
    pub height: u32,
    /// ISP white balance gains [R, B] from camera metadata
    pub colour_gains: Option<[f32; 2]>,
    /// 3x3 colour correction matrix from camera metadata
    pub colour_correction_matrix: Option<[[f32; 3]; 3]>,
    /// Negotiated source bit depth (8..=16 unpacked, 10/12/14 CSI-2).
    pub bit_depth: u32,
    /// Source CFA identity: 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG.
    pub pattern: u32,
}

/// Extract Bayer color planes from a raw camera frame.
///
/// HDR+ paper Section 5: "We merge each Bayer color channel independently."
/// This function extracts the 4 Bayer color planes (R, Gr, B, Gb) from raw sensor
/// data and packs them as RGBA at half resolution for GPU processing.
///
/// Handles:
/// - CSI-2 packed formats (10/12/14-bit)
/// - Standard 8-bit Bayer
/// - Little-endian unpacked Bayer in 16-bit containers (8..=16 meaningful bits)
///
/// Packing requires a negotiated descriptor except for tight U8 rows. Dimensions
/// must be even and >=2; HDR merging consumes rectangular complete CFA quads.
/// Per-site [R, Gr, Gb, B] black is subtracted exactly once, preserving signed
/// shadows down to the canonical float floor and capping source white at 1.
/// Downstream finishing must not subtract black again or normalize a second time.
pub(crate) fn extract_bayer_planes(frame: &CameraFrame) -> Result<BayerPlanes, String> {
    if !frame.format.is_bayer() {
        return Err(format!(
            "extract_bayer_planes: expected Bayer format, got {:?}",
            frame.format
        ));
    }

    let width = frame.width;
    let height = frame.height;
    let stride = frame.stride as usize;
    let data = frame.data.as_ref();
    let meta = frame.libcamera_metadata.as_ref();

    let levels = validate_bayer_levels(
        meta.and_then(|m| m.black_level),
        meta.and_then(|m| m.sensor_black_levels),
        meta.and_then(|m| m.sensor_white_levels),
    )
    .map_err(str::to_owned)?;

    // Validate before normalization/allocation. Keep WB/CCM unapplied here:
    // extraction subtracts black exactly once; GPU finishing applies colour.
    if let Some(meta) = meta {
        // This precheck historically uses a unit sample bound, not the actual
        // signed extraction range. Finishing validates its reconstructed bound.
        crate::shaders::validate_bayer_colour(
            meta.colour_gains,
            meta.colour_correction_matrix,
            1.0,
        )
        .map_err(|error| {
            if error == "Invalid Bayer colour metadata or samples" {
                "Invalid Bayer colour metadata".to_owned()
            } else {
                error.to_owned()
            }
        })?;
    }

    // HDR planes represent rectangular complete 2x2 quads: do not silently crop
    // a trailing odd row/column or change the source CFA origin.
    if width < 2 || height < 2 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err("HDR Bayer extraction requires even dimensions of at least 2x2".into());
    }
    let layout = meta
        .and_then(|m| m.raw_layout)
        .or_else(|| (frame.stride == width).then_some(RawLayout::U8))
        .ok_or("Unknown RAW packing; row stride is not a packing descriptor")?;
    let (bit_depth, is_packed, row_bytes) = match layout {
        RawLayout::U8 => (8, false, width),
        RawLayout::Unpacked16 {
            bits: bits @ 8..=16,
        } => (bits, false, width.checked_mul(2).ok_or("RAW row overflow")?),
        RawLayout::Csi2 { bits } if [10, 12, 14].contains(&bits) => {
            let group = if bits == 12 { 2 } else { 4 };
            (
                bits,
                true,
                width
                    .div_ceil(group)
                    .checked_mul(group * bits / 8)
                    .ok_or("RAW row overflow")?,
            )
        }
        _ => return Err("Unsupported RAW bit depth".into()),
    };
    let data = checked_plane(
        data,
        0,
        data.len(),
        stride,
        row_bytes as usize,
        height as usize,
    )
    .map_err(|e| e.to_string())?;

    let half_w = width / 2;
    let half_h = height / 2;
    let plane_len = (half_w as usize)
        .checked_mul(half_h as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or("HDR plane size overflow")?;
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(plane_len)
        .map_err(|e| format!("HDR plane allocation: {e}"))?;
    planes.resize(plane_len, 0.0f32);

    // Metadata endpoints are normalized, not raw sample counts.
    let max_val = ((1u32 << bit_depth) - 1) as f32;
    let scale = 1.0 / max_val;

    // Get Bayer pattern offsets: (r_dx, r_dy) tells where R is in the 2×2 quad
    let pattern = frame.format.bayer_pattern_code().unwrap_or(0);
    // Pattern layout (dx, dy offsets within 2×2 quad):
    // RGGB (0): R=(0,0) Gr=(1,0) Gb=(0,1) B=(1,1)
    // BGGR (1): B=(0,0) Gb=(1,0) Gr=(0,1) R=(1,1)
    // GRBG (2): Gr=(0,0) R=(1,0) B=(0,1) Gb=(1,1)
    // GBRG (3): Gb=(0,0) B=(1,0) R=(0,1) Gr=(1,1)
    let offsets = match pattern {
        0 => BayerOffsets {
            r: (0, 0),
            gr: (1, 0),
            gb: (0, 1),
            b: (1, 1),
        }, // RGGB
        1 => BayerOffsets {
            r: (1, 1),
            gr: (0, 1),
            gb: (1, 0),
            b: (0, 0),
        }, // BGGR
        2 => BayerOffsets {
            r: (1, 0),
            gr: (0, 0),
            gb: (1, 1),
            b: (0, 1),
        }, // GRBG
        3 => BayerOffsets {
            r: (0, 1),
            gr: (1, 1),
            gb: (0, 0),
            b: (1, 0),
        }, // GBRG
        _ => BayerOffsets {
            r: (0, 0),
            gr: (1, 0),
            gb: (0, 1),
            b: (1, 1),
        }, // default RGGB
    };

    if is_packed {
        extract_planes_csi2_packed(
            data,
            stride,
            bit_depth,
            half_w,
            half_h,
            levels,
            scale,
            &offsets,
            &mut planes,
        );
    } else if layout == RawLayout::U8 {
        extract_planes_8bit(
            data,
            stride,
            half_w,
            half_h,
            levels,
            scale,
            &offsets,
            &mut planes,
        );
    } else {
        extract_planes_16bit(
            data,
            stride,
            half_w,
            half_h,
            levels,
            scale,
            &offsets,
            &mut planes,
        );
    }

    debug!(
        half_w,
        half_h,
        bit_depth,
        is_packed,
        pattern,
        black_level = ?levels,
        "Extracted Bayer planes"
    );

    Ok(BayerPlanes {
        data: planes,
        width: half_w,
        height: half_h,
        colour_gains: meta.and_then(|m| m.colour_gains),
        colour_correction_matrix: meta.and_then(|m| m.colour_correction_matrix),
        bit_depth,
        pattern,
    })
}

/// Extract Bayer planes using a generic pixel-reading closure.
///
/// All Bayer extraction (8-bit, 16-bit, CSI-2 packed) shares the same loop structure
/// and normalization logic. Only the pixel-reading differs, provided by the closure.
fn extract_planes_generic(
    half_w: u32,
    half_h: u32,
    levels: BayerLevels,
    scale: f32,
    off: &BayerOffsets,
    planes: &mut [f32],
    read_pixel: impl Fn(usize, usize) -> f32,
) {
    let (r_dx, r_dy) = off.r;
    let (gr_dx, gr_dy) = off.gr;
    let (gb_dx, gb_dy) = off.gb;
    let (b_dx, b_dy) = off.b;
    for y in 0..half_h as usize {
        let by = y * 2;
        for x in 0..half_w as usize {
            let bx = x * 2;
            let r_val = read_pixel(by + r_dy, bx + r_dx);
            let gr_val = read_pixel(by + gr_dy, bx + gr_dx);
            let gb_val = read_pixel(by + gb_dy, bx + gb_dx);
            let b_val = read_pixel(by + b_dy, bx + b_dx);

            let idx = (y * half_w as usize + x) * 4;
            let normalize = |value: f32, site: usize| {
                ((value * scale - levels.black[site]) / (levels.white[site] - levels.black[site]))
                    .clamp(-0.147_984_65, 1.0)
            };
            planes[idx] = normalize(r_val, 0);
            planes[idx + 1] = normalize(gr_val, 1);
            planes[idx + 2] = normalize(b_val, 3);
            planes[idx + 3] = normalize(gb_val, 2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Extract Bayer planes from 8-bit data
fn extract_planes_8bit(
    data: &[u8],
    stride: usize,
    half_w: u32,
    half_h: u32,
    levels: BayerLevels,
    scale: f32,
    off: &BayerOffsets,
    planes: &mut [f32],
) {
    extract_planes_generic(half_w, half_h, levels, scale, off, planes, |row, col| {
        data[row * stride + col] as f32
    });
}

#[allow(clippy::too_many_arguments)]
/// Extract Bayer planes from 16-bit data (little-endian u16)
fn extract_planes_16bit(
    data: &[u8],
    stride: usize,
    half_w: u32,
    half_h: u32,
    levels: BayerLevels,
    scale: f32,
    off: &BayerOffsets,
    planes: &mut [f32],
) {
    extract_planes_generic(half_w, half_h, levels, scale, off, planes, |row, col| {
        let offset = row * stride + col * 2;
        u16::from_le_bytes([data[offset], data[offset + 1]]) as f32
    });
}

#[allow(clippy::too_many_arguments)]
/// Extract Bayer planes from CSI-2 packed data (10/12/14-bit)
fn extract_planes_csi2_packed(
    data: &[u8],
    stride: usize,
    bit_depth: u32,
    half_w: u32,
    half_h: u32,
    levels: BayerLevels,
    scale: f32,
    off: &BayerOffsets,
    planes: &mut [f32],
) {
    // Read a pixel value from CSI-2 packed row data
    let read_packed_pixel = |row_data: &[u8], pixel_x: usize| -> f32 {
        match bit_depth {
            10 => {
                // 10-bit: 4 pixels packed in 5 bytes
                // Bytes 0-3: MSBs of pixels 0-3 (bits 9..2)
                // Byte 4: LSBs of pixels 0-3 (bits 1..0), 2 bits each
                let group = pixel_x / 4;
                let pos = pixel_x % 4;
                let base = group * 5;
                let msb = row_data[base + pos] as u16;
                let lsb = ((row_data[base + 4] >> (pos * 2)) & 0x03) as u16;
                ((msb << 2) | lsb) as f32
            }
            12 => {
                // 12-bit: 2 pixels packed in 3 bytes
                let group = pixel_x / 2;
                let pos = pixel_x % 2;
                let base = group * 3;
                let msb = row_data[base + pos] as u16;
                let lsb = if pos == 0 {
                    (row_data[base + 2] & 0x0F) as u16
                } else {
                    ((row_data[base + 2] >> 4) & 0x0F) as u16
                };
                ((msb << 4) | lsb) as f32
            }
            14 => {
                // 14-bit: 4 pixels packed in 7 bytes (MIPI CSI-2 RAW14).
                //   Byte 0..3: MSBs (bits 13..6) of pixels 0..3
                //   Byte 4   : P1[1:0] | P0[5:0]
                //   Byte 5   : P2[3:0] | P1[5:2]
                //   Byte 6   : P3[5:0] | P2[5:4]
                let group = pixel_x / 4;
                let pos = pixel_x % 4;
                let base = group * 7;
                let msb = row_data[base + pos] as u16;
                let b4 = row_data[base + 4] as u16;
                let b5 = row_data[base + 5] as u16;
                let b6 = row_data[base + 6] as u16;
                let lsb: u16 = match pos {
                    0 => b4 & 0x3F,
                    1 => ((b5 & 0x0F) << 2) | (b4 >> 6),
                    2 => ((b6 & 0x03) << 4) | (b5 >> 4),
                    3 => b6 >> 2,
                    _ => 0,
                };
                ((msb << 6) | lsb) as f32
            }
            _ => unreachable!("validated CSI-2 bit depth"),
        }
    };

    extract_planes_generic(half_w, half_h, levels, scale, off, planes, |row, col| {
        let group = if bit_depth == 12 { 2 } else { 4 };
        let row_bytes = (half_w as usize * 2).div_ceil(group) * (group * bit_depth as usize / 8);
        let row_data = &data[row * stride..row * stride + row_bytes];
        read_packed_pixel(row_data, col)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::{FrameData, FrameMetadata, PixelFormat, RawLayout};

    fn frame(
        data: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
        layout: Option<RawLayout>,
    ) -> CameraFrame {
        CameraFrame {
            width,
            height,
            stride,
            data: FrameData::from_owned_vec(data),
            format: PixelFormat::BayerRGGB,
            yuv_planes: None,
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: Some(FrameMetadata {
                raw_layout: layout,
                ..Default::default()
            }),
        }
    }

    #[test]
    fn explicit_white_levels_normalize_hdr_sites() {
        let mut input = frame(vec![80, 100, 120, 140], 2, 2, 2, Some(RawLayout::U8));
        let meta = input.libcamera_metadata.as_mut().unwrap();
        meta.sensor_black_levels = Some([20.0, 30.0, 40.0, 50.0].map(|v| v / 255.0));
        meta.sensor_white_levels = Some([140.0, 170.0, 200.0, 230.0].map(|v| v / 255.0));
        let planes = extract_bayer_planes(&input).unwrap();
        for (actual, expected) in planes.data.iter().zip([0.5, 0.5, 0.5, 0.5]) {
            close(*actual, expected);
        }
    }

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    // Post-GREEN contract characterization (not additional RED evidence).
    #[test]
    fn tight_metadata_free_u8_and_colour_passthrough() {
        let mut input = frame(vec![0, 64, 128, 255], 2, 2, 2, None);
        input.libcamera_metadata = None;
        let planes = extract_bayer_planes(&input).unwrap();
        assert_eq!(planes.bit_depth, 8);
        for (actual, code) in planes.data.iter().zip([0, 64, 255, 128]) {
            close(*actual, code as f32 / 255.0);
        }
        let gains = [2.0, 3.0];
        let ccm = [[1.1, -0.2, 0.1], [0.1, 1.0, -0.1], [-0.1, 0.2, 0.9]];
        input.libcamera_metadata = Some(FrameMetadata {
            black_level: Some(0.1),
            colour_gains: Some(gains),
            colour_correction_matrix: Some(ccm),
            ..Default::default()
        });
        let planes = extract_bayer_planes(&input).unwrap();
        assert_eq!(planes.colour_gains, Some(gains));
        assert_eq!(planes.colour_correction_matrix, Some(ccm));
        close(planes.data[0], -0.1 / 0.9);
        close(planes.data[1], (64.0 / 255.0 - 0.1) / 0.9);
    }

    #[test]
    fn source_white_cap_and_partial_packed_groups() {
        let planes = extract_bayer_planes(&frame(
            vec![255; 8],
            2,
            2,
            4,
            Some(RawLayout::Unpacked16 { bits: 10 }),
        ))
        .unwrap();
        assert_eq!(planes.data, vec![1.0; 4]);
        // Complete CSI-2 groups are needed even when only two pixels are active.
        for (bits, row) in [(10, vec![255; 5]), (14, vec![255; 7])] {
            let stride = row.len() as u32;
            let planes = extract_bayer_planes(&frame(
                [row.clone(), row].concat(),
                2,
                2,
                stride,
                Some(RawLayout::Csi2 { bits }),
            ))
            .unwrap();
            assert_eq!(planes.data, vec![1.0; 4]);
        }
    }

    #[test]
    fn signed_per_site_black_is_subtracted_once() {
        let mut input = frame(vec![0, 64, 128, 255], 2, 2, 2, None);
        let meta = input.libcamera_metadata.as_mut().unwrap();
        meta.black_level = Some(0.4);
        meta.sensor_black_levels = Some([0.2, 0.3, 0.4, 0.5]);
        let planes = extract_bayer_planes(&input).unwrap();
        close(planes.data[0], -0.147_984_65_f32);
        close(planes.data[1], (64.0 / 255.0 - 0.3) / 0.7);
        close(planes.data[2], 1.0);
        close(planes.data[3], (128.0 / 255.0 - 0.4) / 0.6);
    }

    #[test]
    fn negotiated_unpack_depth_and_padding_are_not_guessed() {
        for bits in [8, 10, 12, 14, 16] {
            let max = (1u32 << bits) - 1;
            let codes = [max / 4, max / 2, max / 3, max];
            let mut bytes = Vec::new();
            for row in codes.chunks(2) {
                for &code in row {
                    bytes.extend_from_slice(&(code as u16).to_le_bytes());
                }
                if bytes.len() == 4 {
                    bytes.extend_from_slice(&[0xee; 6]);
                }
            }
            let planes = extract_bayer_planes(&frame(
                bytes,
                2,
                2,
                10,
                Some(RawLayout::Unpacked16 { bits }),
            ))
            .unwrap();
            assert_eq!(planes.bit_depth, bits);
            for (actual, code) in planes
                .data
                .iter()
                .zip([codes[0], codes[1], codes[3], codes[2]])
            {
                close(*actual, code as f32 / max as f32);
            }
        }
        let planes = extract_bayer_planes(&frame(
            vec![10, 20, 99, 99, 30, 40],
            2,
            2,
            4,
            Some(RawLayout::U8),
        ))
        .unwrap();
        close(planes.data[2], 40.0 / 255.0);
        assert!(extract_bayer_planes(&frame(vec![0; 8], 2, 2, 4, None)).is_err());
    }

    #[test]
    fn truncated_planes_and_invalid_layouts_fail_closed() {
        for (layout, stride, len) in [
            (Some(RawLayout::U8), 2, 3),
            (Some(RawLayout::Unpacked16 { bits: 10 }), 4, 7),
            (Some(RawLayout::Csi2 { bits: 10 }), 5, 9),
            (Some(RawLayout::Csi2 { bits: 12 }), 3, 5),
            (Some(RawLayout::Csi2 { bits: 14 }), 7, 13),
            (Some(RawLayout::U8), 1, 4),
            (Some(RawLayout::Unpacked16 { bits: 7 }), 4, 8),
            (Some(RawLayout::Unpacked16 { bits: 17 }), 4, 8),
            (Some(RawLayout::Csi2 { bits: 16 }), 4, 8),
        ] {
            let input = frame(vec![0; len], 2, 2, stride, layout);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                extract_bayer_planes(&input)
            }));
            assert!(
                matches!(result, Ok(Err(_))),
                "layout={layout:?}, stride={stride}, len={len}"
            );
        }
    }

    #[test]
    fn packed_rows_require_descriptor_and_complete_groups() {
        for (bits, row) in [
            (10, vec![0x40, 0x80, 0xff, 0x20, 0b11001001]),
            (12, vec![0x40, 0x80, 0xff, 0x20, 0x21, 0x43]),
            (
                14,
                vec![0x40, 0x80, 0xff, 0x20, 0b10000001, 0b00110001, 0b00010010],
            ),
        ] {
            // RAW12 has two groups: rearrange into its three-byte per-pair order.
            let row = if bits == 12 {
                vec![0x40, 0x80, 0x21, 0xff, 0x20, 0x43]
            } else {
                row
            };
            let expected = match bits {
                10 => [257, 514, 1020, 131],
                12 => [1025, 2050, 4083, 516],
                _ => [4097, 8198, 16355, 2052],
            };
            let stride = row.len() + 3;
            let bytes = [row.clone(), vec![0xff; 3], row].concat();
            let planes = extract_bayer_planes(&frame(
                bytes.clone(),
                4,
                2,
                stride as u32,
                Some(RawLayout::Csi2 { bits }),
            ))
            .unwrap();
            assert_eq!(planes.bit_depth, bits);
            for (actual, code) in planes.data.iter().zip([
                expected[0],
                expected[1],
                expected[1],
                expected[0],
                expected[2],
                expected[3],
                expected[3],
                expected[2],
            ]) {
                close(*actual, code as f32 / ((1u32 << bits) - 1) as f32);
            }
            assert!(extract_bayer_planes(&frame(bytes, 4, 2, stride as u32, None)).is_err());
        }
    }

    #[test]
    fn incomplete_cfa_quads_are_rejected() {
        for (width, height) in [(0, 2), (1, 2), (2, 1), (3, 2), (2, 3), (3, 3)] {
            assert!(
                extract_bayer_planes(&frame(
                    vec![0; (width * height) as usize],
                    width,
                    height,
                    width,
                    Some(RawLayout::U8)
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn cfa_identity_and_channel_order_survive_extraction() {
        for (pattern, format, bytes) in [
            (0, PixelFormat::BayerRGGB, vec![10, 20, 30, 40]),
            (1, PixelFormat::BayerBGGR, vec![40, 30, 20, 10]),
            (2, PixelFormat::BayerGRBG, vec![20, 10, 40, 30]),
            (3, PixelFormat::BayerGBRG, vec![30, 40, 10, 20]),
        ] {
            let mut input = frame(bytes, 2, 2, 2, None);
            input.format = format;
            let planes = extract_bayer_planes(&input).unwrap();
            assert_eq!(planes.pattern, pattern);
            assert_eq!((planes.width, planes.height, planes.data.len()), (1, 1, 4));
            for (actual, code) in planes.data.iter().zip([10, 20, 40, 30]) {
                close(*actual, code as f32 / 255.0);
            }
        }
    }

    #[test]
    fn invalid_and_unsafe_colour_metadata_is_rejected() {
        let base = frame(vec![0; 4], 2, 2, 2, None);
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.0] {
            let mut input = base.clone();
            input.libcamera_metadata.as_mut().unwrap().black_level = Some(value);
            assert!(extract_bayer_planes(&input).is_err());
            input.libcamera_metadata.as_mut().unwrap().black_level = Some(0.0);
            input
                .libcamera_metadata
                .as_mut()
                .unwrap()
                .sensor_black_levels = Some([0.0, value, 0.0, 0.0]);
            assert!(extract_bayer_planes(&input).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, 0.0, -1.0, f32::MAX] {
            let mut input = base.clone();
            input.libcamera_metadata.as_mut().unwrap().colour_gains = Some([value, 1.0]);
            assert_eq!(
                extract_bayer_planes(&input).err().unwrap(),
                if value == f32::MAX {
                    "Unsafe Bayer colour arithmetic"
                } else {
                    "Invalid Bayer colour metadata"
                }
            );
            // Metadata errors precede even malformed CFA geometry.
            input.width = 1;
            assert_eq!(
                extract_bayer_planes(&input).err().unwrap(),
                if value == f32::MAX {
                    "Unsafe Bayer colour arithmetic"
                } else {
                    "Invalid Bayer colour metadata"
                }
            );
        }
        for value in [f32::NAN, f32::INFINITY, f32::MAX] {
            let mut input = base.clone();
            input
                .libcamera_metadata
                .as_mut()
                .unwrap()
                .colour_correction_matrix = Some([[value; 3]; 3]);
            assert!(extract_bayer_planes(&input).is_err());
        }
        let mut input = base;
        let meta = input.libcamera_metadata.as_mut().unwrap();
        meta.colour_gains = Some([1e20, 1.0]);
        meta.colour_correction_matrix = Some([[1e20, -1e20, 0.0]; 3]);
        assert!(extract_bayer_planes(&input).is_err());
    }
}
