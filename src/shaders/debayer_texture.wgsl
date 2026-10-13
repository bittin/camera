// SPDX-License-Identifier: GPL-3.0-only
// Texture adapter, composed with debayer.wgsl.
struct DebayerParams {
    width: u32,
    height: u32,
    pattern: u32,           // 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG
    _padding: u32,          // Shared AWB uniform prefix alignment
    black_level: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    ccm_row0: vec4<f32>,
    ccm_row1: vec4<f32>,
    ccm_row2: vec4<f32>,
    white_levels: vec4<f32>, // [R, Gr, Gb, B], normalized sensor codes
}

struct AwbGains {
    gain_r: f32,
    gain_b: f32,
}

@group(0) @binding(0) var tex_bayer: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: DebayerParams;
@group(0) @binding(3) var<storage, read> awb_gains: AwbGains;
// GPU RCD/LMMSE camera RGB with raw sensor peak in alpha, already normalized.
@group(0) @binding(4) var demosaiced: texture_2d<f32>;

// Get a normalized sensor sample with CFA-parity-preserving edge reflection.
fn sample_bayer(x: i32, y: i32) -> f32 {
    // Mirror the one-pixel neighborhood at edges to preserve CFA parity.
    let cx = reflect_bayer_coordinate(x, params.width);
    let cy = reflect_bayer_coordinate(y, params.height);
    var code = textureLoad(tex_bayer, vec2(u32(cx), u32(cy)), 0).r;
    // Normalize the negotiated sensor precision after upload to R16Unorm.
    code *= bitcast<f32>(params._pad0);
    let site = get_pixel_type(u32(cx),u32(cy),params.pattern);
    let blacks = vec4(params.black_level,params.ccm_row0.w,params.ccm_row1.w,params.ccm_row2.w);
    let bl = blacks[site];
    return clamp((code - bl) / (params.white_levels[site] - bl), -0.14798465371131897, 1.0);
}

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }
    var raw: DemosaicResult;
    if params._pad1 != 0u {
        let rgb_peak = textureLoad(demosaiced, gid.xy, 0);
        raw = DemosaicResult(rgb_peak.xyz, rgb_peak.w);
    } else {
        raw = demosaic(gid.x, gid.y, params.pattern);
    }
    let corrected = apply_bayer_colour(raw.rgb, raw.sensor_peak,
        vec2(awb_gains.gain_r, awb_gains.gain_b),
        params.ccm_row0.xyz, params.ccm_row1.xyz, params.ccm_row2.xyz);
    textureStore(output, gid.xy, vec4(corrected, 1.0));
}
