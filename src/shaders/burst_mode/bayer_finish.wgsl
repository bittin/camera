// SPDX-License-Identifier: GPL-3.0-only
// Merged-plane adapter composed with the exact shared debayer.wgsl core.
// Storage order R, Gr, B, Gb; CFA spatial phase is retained separately.
@group(0) @binding(0) var<storage, read> bayer_planes: array<vec4<f32>>;
@group(0) @binding(1) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: FinishParams;
@group(0) @binding(3) var camera_rgb:texture_2d<f32>;
struct FinishParams {
    half_width: u32, half_height: u32,
    full_width: u32, full_height: u32,
    gain_r: f32, gain_b: f32,
    ccm_row0: vec4<f32>, ccm_row1: vec4<f32>, ccm_row2: vec4<f32>,
    _pad0: u32, pattern: u32, _pad1: u32, _pad2: u32,
}
fn sample_bayer(x: i32, y: i32) -> f32 {
    let cx = u32(reflect_bayer_coordinate(x, params.full_width));
    let cy = u32(reflect_bayer_coordinate(y, params.full_height));
    let site = get_pixel_type(cx, cy, params.pattern);
    // Core site order R, Gr, Gb, B differs from merged plane storage.
    let channels = vec4<u32>(0u, 1u, 3u, 2u);
    return bayer_planes[(cy / 2u) * params.half_width + cx / 2u][channels[site]];
}
@compute @workgroup_size(16,16)
fn demosaic_and_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.full_width || gid.y >= params.full_height { return; }
    var raw: DemosaicResult;
    if params._pad0 != 0u {
        let rgb_peak = textureLoad(camera_rgb, gid.xy, 0);
        raw = DemosaicResult(rgb_peak.xyz, rgb_peak.w);
    } else {
        raw = demosaic(gid.x, gid.y, params.pattern);
    }
    let rgb = apply_bayer_colour(raw.rgb, raw.sensor_peak, vec2(params.gain_r, params.gain_b),
        params.ccm_row0.xyz, params.ccm_row1.xyz, params.ccm_row2.xyz);
    // Keep signed / over-white linear values until the tone-map boundary.
    textureStore(output, gid.xy, vec4(rgb, 1.0));
}
