// SPDX-License-Identifier: GPL-3.0-only
// Equal source-transform prefix only. Callers retain letterbox and crop/zoom order.
struct PreviewGeometry {
    uv: vec2<f32>,
    crop_min: vec2<f32>,
    crop_max: vec2<f32>,
}

fn preview_geometry(input_uv: vec2<f32>, source_size: vec2<u32>) -> PreviewGeometry {
    var tex_coords = viewport.uv_offset + input_uv * viewport.uv_scale;

    // Apply horizontal mirror if enabled (selfie mode)
    // This happens BEFORE rotation so the mirror is in screen space
    if (viewport.mirror_horizontal == 1u) {
        tex_coords.x = 1.0 - tex_coords.x;
    }

    // Apply rotation correction for sensor orientation
    // Transforms UV coordinates to correct for physical sensor rotation
    // For a sensor mounted N degrees CW, we rotate the UV coords N degrees CW
    // to sample from the correct position in the rotated texture
    if (viewport.rotation == 1u) {
        // 90 CW sensor -> sample rotated 90 CW: (u,v) -> (1-v, u)
        tex_coords = vec2<f32>(1.0 - tex_coords.y, tex_coords.x);
    } else if (viewport.rotation == 2u) {
        // 180 sensor -> rotate 180: (u,v) -> (1-u, 1-v)
        tex_coords = vec2<f32>(1.0 - tex_coords.x, 1.0 - tex_coords.y);
    } else if (viewport.rotation == 3u) {
        // 270 CW sensor -> sample rotated 270 CW: (u,v) -> (v, 1-u)
        tex_coords = vec2<f32>(tex_coords.y, 1.0 - tex_coords.x);
    }

    // Apply Cover/Contain blend (0.0 = Contain, 1.0 = Cover, intermediate = animating).
    //
    // The effective crop region is itself blended: at 1.0 it degenerates to the
    // full texture (no crop), at 0.0 it is the full aspect-ratio crop.  This keeps
    // the animated transition continuous — no discrete snap at the endpoints even
    // when Cover and Contain would nominally show different regions.
    let blend = viewport.content_fit_mode;
    let effective_crop_min = mix(viewport.crop_uv_min, vec2<f32>(0.0, 0.0), blend);
    let effective_crop_max = mix(viewport.crop_uv_max, vec2<f32>(1.0, 1.0), blend);
    {
        // Get texture dimensions, accounting for rotation
        let raw_tex_size = vec2<f32>(source_size);
        var tex_size = raw_tex_size;
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            tex_size = vec2<f32>(raw_tex_size.y, raw_tex_size.x);
        }
        // Effective dimensions after the (blended) crop.
        // `crop_range` is in texture-orientation (crop UVs are sensor-space, see
        // PhotoAspectRatio::crop_uv); `tex_size` is in display-orientation (swapped
        // above). Swap `crop_range` to display-orientation before multiplying so
        // `effective_tex` has the right aspect on rotated sensors. Without this,
        // an aspect-ratio crop in Contain mode produces a wrong-shape letterbox
        // and distorts the sampled image on the phone (rotation 1 / 3).
        var crop_range = effective_crop_max - effective_crop_min;
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            crop_range = vec2<f32>(crop_range.y, crop_range.x);
        }
        let effective_tex = tex_size * crop_range;

        // Content area between the effective UI edges (for contain centering)
        let content_width = viewport.viewport_size.x - viewport.bar_left_width - viewport.bar_right_width;
        let content_height = viewport.viewport_size.y - viewport.bar_top_height - viewport.bar_bottom_height;
        let content_center_x = (viewport.bar_left_width + content_width * 0.5) / viewport.viewport_size.x;
        let content_center_y = (viewport.bar_top_height + content_height * 0.5) / viewport.viewport_size.y;

        // Zoom levels using the blended effective texture dimensions
        let contain_zoom = min(content_width / effective_tex.x, content_height / effective_tex.y);
        let cover_zoom = max(viewport.viewport_size.x / effective_tex.x, viewport.viewport_size.y / effective_tex.y);

        let zoom = mix(contain_zoom, cover_zoom, blend);
        let center_x = mix(content_center_x, 0.5, blend);
        let center_y = mix(content_center_y, 0.5, blend);

        var scale = vec2<f32>(
            viewport.viewport_size.x / (effective_tex.x * zoom),
            viewport.viewport_size.y / (effective_tex.y * zoom),
        );
        // For 90/270 rotations, swap scale factors since we're in rotated UV space
        if (viewport.rotation == 1u || viewport.rotation == 3u) {
            scale = vec2<f32>(scale.y, scale.x);
        }

        // The pivot below is in screen-space; rotate it into texture-UV space so it
        // matches `tex_coords` (already rotated above). Without this, the asymmetric
        // `center_y` lands on the wrong axis for 90/270 rotations and is inverted for
        // 180 — Contain centering and aspect-ratio crops drift off-axis on the phone.
        // `tex_coords` was mirrored around the window above. Mirror the pivot
        // too, so the fitted image stays on its physical side of an asymmetric
        // left/right content area while only its pixels are reflected.
        let pivot_x = select(center_x, 1.0 - center_x, viewport.mirror_horizontal == 1u);
        var pivot = vec2<f32>(pivot_x, center_y);
        if (viewport.rotation == 1u) {
            pivot = vec2<f32>(1.0 - pivot.y, pivot.x);
        } else if (viewport.rotation == 2u) {
            pivot = vec2<f32>(1.0 - pivot.x, 1.0 - pivot.y);
        } else if (viewport.rotation == 3u) {
            pivot = vec2<f32>(pivot.y, 1.0 - pivot.x);
        }
        tex_coords = (tex_coords - pivot) * scale + vec2<f32>(0.5, 0.5);
    }
    return PreviewGeometry(tex_coords, effective_crop_min, effective_crop_max);
}
