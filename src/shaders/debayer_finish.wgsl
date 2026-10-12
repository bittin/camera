// SPDX-License-Identifier: GPL-3.0-only
// Bayer finishing. Scene-linear -> bounded encoded sRGB.
// Peak-ratio application recovered from CameraColorProcessing-018.ll, SoftLTM
// ltmCompressHighlight. Float domain and neutral T=min(m,1) curve are independent
// substitutions, not recovered Apple curve generation or 16-bit/half execution.
@group(0) @binding(0) var linear: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
fn srgb(v: f32) -> f32 {
    if v <= 0.0031308 { return 12.92 * v; }
    return 1.055 * pow(v, 1.0/2.4) - 0.055;
}
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(linear);
    if id.x >= dims.x || id.y >= dims.y { return; }
    let rgb = textureLoad(linear, id.xy, 0).rgb;
    let m = max(rgb.r, max(rgb.g, rgb.b));
    // Neutral below white, hue-preserving cap above it. Protect the recovered
    // ratio's zero-domain independently, without attenuating sub-code signals.
    let t = clamp(m, 0.0, 1.0);
    let ratio = select(1.0, t / max(m, 0.000015259021893143654), m > 0.0);
    let bounded = clamp(rgb * ratio, vec3(0.0), vec3(1.0));
    textureStore(output, id.xy, vec4(srgb(bounded.r), srgb(bounded.g), srgb(bounded.b), 1.0));
}
