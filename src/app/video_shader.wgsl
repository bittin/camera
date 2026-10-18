// SPDX-License-Identifier: GPL-3.0-only
// GPU shader for direct RGBA texture rendering with object-fit: cover support
// Filter functions are prepended by the Rust code from shaders/filters.wgsl

@group(0) @binding(0)
var texture_rgba: texture_2d<f32>;

@group(0) @binding(1)
var sampler_video: sampler;

// Fragment shader - RGBA passthrough with Cover mode support
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply scroll clipping UV transformation
    // This maps the visible portion's UV (0-1) to the correct portion of the full widget
    let geometry = preview_geometry(in.tex_coords, textureDimensions(texture_rgba));
    var tex_coords = geometry.uv;
    let effective_crop_min = geometry.crop_min;
    let effective_crop_max = geometry.crop_max;

    // Discard letterbox *before* the crop remap.  The intermediate range [0,1]
    // is the rendered image region; outside that range the crop remap can still
    // produce an in-range texture UV (when `crop_min > 0`), so a post-remap check
    // would silently stretch the crop across the letterbox.
    if (tex_coords.x < 0.0 || tex_coords.x > 1.0 || tex_coords.y < 0.0 || tex_coords.y > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Apply the blended crop remap.  Uses the same `effective_crop_*` as the zoom
    // above so the [0,1] intermediate maps onto the same region the zoom sized for.
    tex_coords = mix(effective_crop_min, effective_crop_max, tex_coords);

    // Apply digital zoom (center crop)
    // At zoom_level 2.0, show only center 50% of the image
    if (viewport.zoom_level > 1.0) {
        let inv_zoom = 1.0 / viewport.zoom_level;
        tex_coords = (tex_coords - vec2<f32>(0.5, 0.5)) * inv_zoom + vec2<f32>(0.5, 0.5);
    }

    // Sample RGBA texture
    var pixel = textureSample(texture_rgba, sampler_video, tex_coords);
    var color = pixel.rgb;

    // Apply the filter (0-14) using the shared preludes.
    color = apply_texture_filter(
        color,
        viewport.filter_mode,
        tex_coords,
        texture_rgba,
        sampler_video,
    );

    // Round the corners off the widget's own rect, exactly as the frosted
    // composite does: `panel_rect` and `corner_radius` in physical px, against
    // `@builtin(position)`. NOT off `viewport_size` — that is the box the fit
    // math works in, and passes that sample an intermediate match it to the
    // intermediate rather than the widget (see the pre-blur's second pass), so a
    // silhouette cut from it lands somewhere other than the widget's edge.
    var alpha = pixel.a;
    if (viewport.corner_radius > 0.0 && viewport.panel_rect.z > 0.0) {
        let half_size = viewport.panel_rect.zw * 0.5;
        let center = viewport.panel_rect.xy + half_size;
        let dist = rounded_box_sdf(in.position.xy - center, half_size, viewport.corner_radius);
        alpha = pixel.a * (1.0 - smoothstep(-1.0, 1.0, dist));
    }

    return vec4<f32>(color, alpha);
}
