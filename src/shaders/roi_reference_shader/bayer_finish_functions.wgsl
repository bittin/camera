// SPDX-License-Identifier: GPL-3.0-only
// Shared float-domain highlight cap and sRGB encoding; no intermediate narrowing.
fn srgb(v: f32) -> f32 {
    if v <= 0.0031308 { return 12.92 * v; }
    return 1.055 * pow(v, 1.0/2.4) - 0.055;
}
fn finish_bayer_colour(rgb: vec3<f32>) -> vec3<f32> {
    let m = max(rgb.r, max(rgb.g, rgb.b));
    let t = clamp(m, 0.0, 1.0);
    let ratio = select(1.0, t / max(m, 0.000015259021893143654), m > 0.0);
    let bounded = clamp(rgb * ratio, vec3(0.0), vec3(1.0));
    return vec3(srgb(bounded.r), srgb(bounded.g), srgb(bounded.b));
}
