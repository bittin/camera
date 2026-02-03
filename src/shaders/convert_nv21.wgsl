// SPDX-License-Identifier: GPL-3.0-only
// GPU compute shader for NV21 to RGBA conversion
//
// NV21: Semi-planar 4:2:0 (Y plane + interleaved VU plane)
// Same as NV12 but V and U channels are swapped
// Uses BT.601 color matrix (standard for webcams and JPEG)

struct ConvertParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var tex_y: texture_2d<f32>;
@group(0) @binding(1) var tex_uv: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: ConvertParams;

fn yuv_to_rgb_bt601(y: f32, u: f32, v: f32) -> vec3<f32> {
    let y_scaled = (y - 16.0 / 255.0) * (255.0 / 219.0);
    let u_shifted = u - 0.5;
    let v_shifted = v - 0.5;
    let r = y_scaled + 1.402 * v_shifted;
    let g = y_scaled - 0.344136 * u_shifted - 0.714136 * v_shifted;
    let b = y_scaled + 1.772 * u_shifted;
    return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let pos = vec2(x, y);

    // Sample Y at full resolution
    let luma = textureLoad(tex_y, pos, 0).r;

    // Sample VU at half resolution (V in R channel, U in G channel)
    let uv_pos = pos / 2u;
    let vu = textureLoad(tex_uv, uv_pos, 0);

    // VU layout: R=V, G=U (swapped from NV12's UV)
    let rgb = yuv_to_rgb_bt601(luma, vu.g, vu.r);
    textureStore(output, pos, vec4(rgb, 1.0));
}
