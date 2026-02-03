// SPDX-License-Identifier: GPL-3.0-only
// GPU compute shader for Gray8 to RGBA conversion
//
// Gray8: Single channel luminance, output as grayscale RGB

struct ConvertParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var tex_gray: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ConvertParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let pos = vec2(x, y);
    let gray = textureLoad(tex_gray, pos, 0).r;
    textureStore(output, pos, vec4(gray, gray, gray, 1.0));
}
