// SPDX-License-Identifier: GPL-3.0-only
// Lightweight Gaussian blur for filter pre-processing (multi-pass filters)
//
// This is a simple 13-tap blur used as a pre-pass before filters that need
// smooth input (e.g., Pencil edge detection). Much lighter than the transition
// blur shader — just enough to suppress sensor noise for clean spatial operations.
//
// The common prefix is composed from video_geometry.wgsl. Crop/zoom ordering
// below deliberately remains local: this preblur zooms BEFORE its crop remap.

@group(0) @binding(0)
var texture_source: texture_2d<f32>;

@group(0) @binding(1)
var sampler_source: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Pass through UV transforms (mirror, rotation, crop) so the intermediate
    // texture contains the correctly oriented, cropped image.
    let geometry = preview_geometry(in.tex_coords, textureDimensions(texture_source));
    var uv = geometry.uv;
    let effective_crop_min = geometry.crop_min;
    let effective_crop_max = geometry.crop_max;

    // Discard letterbox before the crop remap — see video_shader.wgsl for the
    // rationale (post-remap check can falsely keep fragments inside the crop).
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Apply digital zoom
    if (viewport.zoom_level > 1.0) {
        let inv_zoom = 1.0 / viewport.zoom_level;
        uv = (uv - vec2<f32>(0.5, 0.5)) * inv_zoom + vec2<f32>(0.5, 0.5);
    }

    // Apply the blended crop remap
    uv = mix(effective_crop_min, effective_crop_max, uv);

    // 13-tap Gaussian blur: center + 4 axis + 4 diagonal + 4 far axis
    // Weights approximate a Gaussian with sigma ~1.4, enough to smooth sensor noise.
    // center(dist=0)=4, axis(dist=1)=2, diagonal(dist=√2)=1, far(dist=2)=0.5
    // Total = 4 + 4*2 + 4*1 + 4*0.5 = 18
    let tex_dims = vec2<f32>(textureDimensions(texture_source));
    let px = 1.0 / tex_dims;

    let c  = textureSample(texture_source, sampler_source, uv).rgb;

    // 4 axis neighbors at distance 1 (weight 2)
    let n  = textureSample(texture_source, sampler_source, uv + vec2<f32>( 0.0, -px.y)).rgb;
    let s  = textureSample(texture_source, sampler_source, uv + vec2<f32>( 0.0,  px.y)).rgb;
    let e  = textureSample(texture_source, sampler_source, uv + vec2<f32>( px.x,  0.0)).rgb;
    let w  = textureSample(texture_source, sampler_source, uv + vec2<f32>(-px.x,  0.0)).rgb;

    // 4 diagonal neighbors at distance √2 (weight 1)
    let ne = textureSample(texture_source, sampler_source, uv + vec2<f32>( px.x, -px.y)).rgb;
    let nw = textureSample(texture_source, sampler_source, uv + vec2<f32>(-px.x, -px.y)).rgb;
    let se = textureSample(texture_source, sampler_source, uv + vec2<f32>( px.x,  px.y)).rgb;
    let sw = textureSample(texture_source, sampler_source, uv + vec2<f32>(-px.x,  px.y)).rgb;

    // 4 far axis at distance 2 (weight 0.5)
    let n2 = textureSample(texture_source, sampler_source, uv + vec2<f32>( 0.0, -2.0 * px.y)).rgb;
    let s2 = textureSample(texture_source, sampler_source, uv + vec2<f32>( 0.0,  2.0 * px.y)).rgb;
    let e2 = textureSample(texture_source, sampler_source, uv + vec2<f32>( 2.0 * px.x,  0.0)).rgb;
    let w2 = textureSample(texture_source, sampler_source, uv + vec2<f32>(-2.0 * px.x,  0.0)).rgb;

    let blurred = (c * 4.0
        + (n + s + e + w) * 2.0
        + (ne + nw + se + sw)
        + (n2 + s2 + e2 + w2) * 0.5
    ) / 18.0;

    return vec4<f32>(blurred, 1.0);
}
