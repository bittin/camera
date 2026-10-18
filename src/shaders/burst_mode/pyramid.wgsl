// SPDX-License-Identifier: GPL-3.0-only
//
// Gaussian pyramid builder for night mode alignment
//
// Builds a 4-level Gaussian pyramid for hierarchical alignment.
// Each level is 2x downsampled with Gaussian blur to avoid aliasing.
//
// Based on HDR+ alignment pyramid.

struct PyramidParams {
    src_width: u32,      // Source level width
    src_height: u32,     // Source level height
    dst_width: u32,      // Destination level width (src/2)
    dst_height: u32,     // Destination level height (src/2)
}

// Source level (grayscale luminance or RGBA)
@group(0) @binding(0)
var<storage, read> src_level: array<f32>;

// Destination level (2x downsampled)
@group(0) @binding(1)
var<storage, read_write> dst_level: array<f32>;

@group(0) @binding(2)
var<uniform> params: PyramidParams;

//=============================================================================
// Gaussian kernel (5x5, sigma ~= 1.4)
// Used for anti-aliasing before 2x downsampling
//=============================================================================

// Separable Gaussian weights as individual constants
// (WGSL doesn't allow runtime indexing of const arrays)
const G0: f32 = 0.0625;
const G1: f32 = 0.25;
const G2: f32 = 0.375;
const G3: f32 = 0.25;
const G4: f32 = 0.0625;

// Helper function to get Gaussian weight by index
fn gauss_weight(i: i32) -> f32 {
    // Map -2..2 to weights
    switch (i) {
        case -2: { return G0; }
        case -1: { return G1; }
        case 0:  { return G2; }
        case 1:  { return G3; }
        case 2:  { return G4; }
        default: { return 0.0; }
    }
}

//=============================================================================
// Utility functions
//=============================================================================

fn get_src_pixel(x: i32, y: i32) -> f32 {
    let cx = clamp(x, 0, i32(params.src_width) - 1);
    let cy = clamp(y, 0, i32(params.src_height) - 1);
    let idx = u32(cy) * params.src_width + u32(cx);
    return src_level[idx];
}

//=============================================================================
// Downsample grayscale with 5x5 Gaussian blur
//=============================================================================

@compute @workgroup_size(16, 16)
fn downsample_gray(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_x = gid.x;
    let dst_y = gid.y;

    if (dst_x >= params.dst_width || dst_y >= params.dst_height) {
        return;
    }

    // Corresponding source position (center of 2x2 block)
    let src_x = i32(dst_x * 2u);
    let src_y = i32(dst_y * 2u);

    // Apply 5x5 Gaussian (using helper function for weight lookup)
    var sum = 0.0;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let weight = gauss_weight(dx) * gauss_weight(dy);
            sum += get_src_pixel(src_x + dx, src_y + dy) * weight;
        }
    }

    let dst_idx = dst_y * params.dst_width + dst_x;
    dst_level[dst_idx] = sum;
}
