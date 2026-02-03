// SPDX-License-Identifier: GPL-3.0-only
// GPU compute shader for UYVY to RGBA conversion
//
// UYVY: Packed 4:2:2 - each 4 bytes encode 2 pixels: [U Y0 V Y1]
// Texture is uploaded as RGBA8 where: R=U, G=Y0, B=V, A=Y1
// Uses BT.601 color matrix (standard for webcams and JPEG)

struct ConvertParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var tex_packed: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ConvertParams;

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

    // Each RGBA texel contains 2 pixels worth of data
    let packed_x = x / 2u;
    let packed = textureLoad(tex_packed, vec2(packed_x, y), 0);

    // Select Y0 (G channel) for even pixels, Y1 (A channel) for odd pixels
    let is_odd = (x & 1u) == 1u;
    let luma = select(packed.g, packed.a, is_odd);

    // U in R channel, V in B channel
    let rgb = yuv_to_rgb_bt601(luma, packed.r, packed.b);
    textureStore(output, vec2(x, y), vec4(rgb, 1.0));
}
