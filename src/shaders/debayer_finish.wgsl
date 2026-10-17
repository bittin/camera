// SPDX-License-Identifier: GPL-3.0-only
// Bayer finishing. Scene-linear -> bounded encoded sRGB.
// Peak-ratio application recovered from CameraColorProcessing-018.ll, SoftLTM
// ltmCompressHighlight. Float domain and neutral T=min(m,1) curve are independent
// substitutions, not recovered Apple curve generation or 16-bit/half execution.
// Match the 96-byte DebayerParams prefix; only the dispatch rectangle is read.
struct FinishParams {
    prefix: array<vec4<u32>, 6>,
    region: vec4<u32>,
}
@group(0) @binding(0) var linear: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: FinishParams;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= params.region.zw) { return; }
    let q = id.xy + params.region.xy;
    let rgb = textureLoad(linear, q, 0).rgb;
    textureStore(output, q, vec4(finish_bayer_colour(rgb), 1.0));
}
