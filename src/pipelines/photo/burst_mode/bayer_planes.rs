// SPDX-License-Identifier: GPL-3.0-only

//! Bayer plane extraction from raw sensor frames.
//!
//! HDR+ paper §5: "We merge each Bayer color channel independently." This
//! module turns a `CameraFrame` carrying raw sensor data (8/10/12/14/16 bit,
//! optionally CSI-2 packed) into four half-resolution colour planes packed as
//! RGBA f32 for the GPU compute pipeline.
//!
//! Pure CPU code — no GPU dependencies — so it is independently testable.

use crate::backends::camera::types::CameraFrame;
use crate::backends::camera::v4l2_utils::detect_csi2_bit_depth;
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
/// independently. We extract R, Gr, Gb, B planes at half resolution and pack them
/// as RGBA f32 for efficient GPU processing with existing vec4 shader infrastructure.
pub(crate) struct BayerPlanes {
    /// RGBA f32 data where R=red, G=green_r, B=blue, A=green_b
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
    /// Bit depth of the raw data (8, 10, 12, or 14)
    pub bit_depth: u32,
}

/// Extract Bayer color planes from a raw camera frame.
///
/// HDR+ paper Section 5: "We merge each Bayer color channel independently."
/// This function extracts the 4 Bayer color planes (R, Gr, Gb, B) from raw sensor
/// data and packs them as RGBA at half resolution for GPU processing.
///
/// Handles:
/// - CSI-2 packed formats (10/12/14-bit)
/// - Standard 8-bit Bayer
/// - 16-bit Bayer
///
/// Black level is subtracted during extraction so all downstream processing
/// operates on linear data above the noise floor (HDR+ Section 6 Step 1).
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

    // Determine bit depth and whether data is CSI-2 packed
    let bytes_per_pixel_u16 = width as usize * 2;
    let (bit_depth, is_packed) = if stride > bytes_per_pixel_u16 {
        // stride > width*2: check for CSI-2 packed, otherwise padded 16-bit
        match detect_csi2_bit_depth(width, stride as u32) {
            Some(bd) => (bd, true),
            None => (16, false),
        }
    } else if stride == width as usize {
        (8, false)
    } else if stride >= bytes_per_pixel_u16 {
        (16, false)
    } else {
        // stride < width: CSI-2 packed
        match detect_csi2_bit_depth(width, stride as u32) {
            Some(bd) => (bd, true),
            None => (8, false), // fallback
        }
    };

    let half_w = width / 2;
    let half_h = height / 2;
    let plane_pixels = (half_w * half_h) as usize;
    let mut planes = vec![0.0f32; plane_pixels * 4]; // RGBA packed

    // Get black level from metadata (normalized 0..1 for the bit depth range)
    let black_level_raw = meta.and_then(|m| m.black_level);
    let max_val = ((1u32 << bit_depth) - 1) as f32;
    // black_level from metadata is normalized 0..1, convert to raw counts
    let black_counts = black_level_raw.unwrap_or(0.0) * max_val;
    let scale = 1.0 / (max_val - black_counts).max(1.0);

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
            black_counts,
            scale,
            &offsets,
            &mut planes,
        );
    } else if bit_depth == 8 {
        extract_planes_8bit(
            data,
            stride,
            half_w,
            half_h,
            black_counts,
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
            black_counts,
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
        black_level = ?black_counts,
        "Extracted Bayer planes"
    );

    Ok(BayerPlanes {
        data: planes,
        width: half_w,
        height: half_h,
        colour_gains: meta.and_then(|m| m.colour_gains),
        colour_correction_matrix: meta.and_then(|m| m.colour_correction_matrix),
        bit_depth,
    })
}

/// Extract Bayer planes using a generic pixel-reading closure.
///
/// All Bayer extraction (8-bit, 16-bit, CSI-2 packed) shares the same loop structure
/// and normalization logic. Only the pixel-reading differs, provided by the closure.
fn extract_planes_generic(
    half_w: u32,
    half_h: u32,
    black_counts: f32,
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
            planes[idx] = ((r_val - black_counts).max(0.0)) * scale;
            planes[idx + 1] = ((gr_val - black_counts).max(0.0)) * scale;
            planes[idx + 2] = ((b_val - black_counts).max(0.0)) * scale;
            planes[idx + 3] = ((gb_val - black_counts).max(0.0)) * scale;
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
    black_counts: f32,
    scale: f32,
    off: &BayerOffsets,
    planes: &mut [f32],
) {
    extract_planes_generic(
        half_w,
        half_h,
        black_counts,
        scale,
        off,
        planes,
        |row, col| data[row * stride + col] as f32,
    );
}

#[allow(clippy::too_many_arguments)]
/// Extract Bayer planes from 16-bit data (little-endian u16)
fn extract_planes_16bit(
    data: &[u8],
    stride: usize,
    half_w: u32,
    half_h: u32,
    black_counts: f32,
    scale: f32,
    off: &BayerOffsets,
    planes: &mut [f32],
) {
    extract_planes_generic(
        half_w,
        half_h,
        black_counts,
        scale,
        off,
        planes,
        |row, col| {
            let offset = row * stride + col * 2;
            if offset + 1 < data.len() {
                u16::from_le_bytes([data[offset], data[offset + 1]]) as f32
            } else {
                0.0
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
/// Extract Bayer planes from CSI-2 packed data (10/12/14-bit)
fn extract_planes_csi2_packed(
    data: &[u8],
    stride: usize,
    bit_depth: u32,
    half_w: u32,
    half_h: u32,
    black_counts: f32,
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
                if base + 4 >= row_data.len() {
                    return 0.0;
                }
                let msb = row_data[base + pos] as u16;
                let lsb = ((row_data[base + 4] >> (pos * 2)) & 0x03) as u16;
                ((msb << 2) | lsb) as f32
            }
            12 => {
                // 12-bit: 2 pixels packed in 3 bytes
                let group = pixel_x / 2;
                let pos = pixel_x % 2;
                let base = group * 3;
                if base + 2 >= row_data.len() {
                    return 0.0;
                }
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
                if base + 6 >= row_data.len() {
                    return 0.0;
                }
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
            _ => 0.0,
        }
    };

    extract_planes_generic(
        half_w,
        half_h,
        black_counts,
        scale,
        off,
        planes,
        |row, col| {
            let row_data = &data[row * stride..];
            read_packed_pixel(row_data, col)
        },
    );
}
