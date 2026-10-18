// SPDX-License-Identifier: GPL-3.0-only
// PASS 0 of the blur chain: the TRANSFORM pass.
//
// It applies every preview transform — scroll clip, mirror, sensor rotation,
// cover/contain fit, letterbox fill, crop remap, digital zoom, colour filter —
// and renders the result into a PHYSICAL SCREEN RESOLUTION target. It runs NO
// blur kernel at all.
//
// # Why the kernel is gone
//
// This pass used to double as a blur pass: it sampled the sharp, full-resolution
// sensor frame through a 37-tap ring rosette. At high frost thickness the ring
// spacing reached ~18 sensor texels between taps across un-prefiltered data, and
// the 12-fold ring structure ghosted into the output as visible BANDING (found
// on device). A sparse lattice sampling sharp data is a structurally bad idea and
// no radius tuning fixes it.
//
// The chain now separates the two jobs. This pass only RESAMPLES (one bilinear
// tap), and `video_shader_kawase.wgsl` — a verbatim port of cosmic-comp's own
// dual-Kawase — does all the blurring, over progressively halved copies of THIS
// pass's output. Every Kawase kernel is dense relative to its own level and only
// ever samples data the previous pass band-limited, so it cannot band.
//
// This pass is also what makes the Kawase offsets literally cosmic-comp's: they
// are authored in physical screen px, and by rendering into a screen-resolution
// screen-space target here, the rest of the chain works in exactly that unit. No
// texel-to-screen conversion exists any more, because there is nothing to
// convert.

@group(0) @binding(0)
var texture_blur: texture_2d<f32>;

@group(0) @binding(1)
var sampler_blur: sampler;

// Fragment shader — transform only, one bilinear tap.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply scroll clipping UV transformation
    let geometry = preview_geometry(in.tex_coords, textureDimensions(texture_blur));
    var tex_coords = geometry.uv;
    let effective_crop_min = geometry.crop_min;
    let effective_crop_max = geometry.crop_max;

    // For the blur backdrop, paint letterbox with the theme's background
    // color (opaque) instead of discarding to transparent — the previous
    // discard let the COSMIC window background show through in Contain /
    // Fit mode. Return *before* the crop remap so we can short-circuit;
    // the regular post-remap letterbox check is unnecessary here because
    // anything out of [0,1] is letterbox by definition.
    if (tex_coords.x < 0.0 || tex_coords.x > 1.0 || tex_coords.y < 0.0 || tex_coords.y > 1.0) {
        return vec4<f32>(viewport.letterbox_color.rgb, 1.0);
    }

    // Apply the blended crop remap
    tex_coords = mix(effective_crop_min, effective_crop_max, tex_coords);

    // Apply digital zoom (center crop), AFTER the crop remap and AFTER the
    // letterbox early-out — the same place video_shader.wgsl applies it, so the
    // frosted backdrop samples exactly the region the sharp preview shows. The
    // order matters and is not cosmetic:
    //
    // * after the crop remap, so the pivot is the centre of the FULL texture in
    //   texture UV space, as it is in video_shader.wgsl. Zooming before the
    //   remap would pivot on the centre of the *crop* instead, which only
    //   coincides while the crop is centred.
    // * after the letterbox early-out, so the letterbox extent stays a property
    //   of the fit, not of the zoom — again matching video_shader.wgsl, whose
    //   early-out also precedes its zoom.
    //
    // THIS IS THE ONLY PASS THAT ZOOMS, because it is the only pass that ever
    // samples the source frame. Everything downstream — the Kawase ping-pong and
    // the final composite — works on this pass's screen-space output, in which
    // the zoom is already baked, and leaves `zoom_level` at its 1.0 default.
    if (viewport.zoom_level > 1.0) {
        let inv_zoom = 1.0 / viewport.zoom_level;
        tex_coords = (tex_coords - vec2<f32>(0.5, 0.5)) * inv_zoom + vec2<f32>(0.5, 0.5);
    }

    // ONE bilinear tap. This pass resamples; it does not blur. See the header
    // for why the 37-tap ring rosette that used to live here was the cause of
    // the banding, and `video_shader_kawase.wgsl` for what replaced it.
    var rgb_val = textureSample(texture_blur, sampler_blur, tex_coords).rgb;

    // Apply the filter here — the one pass that sees the source frame, so the
    // filter is visible in the blur exactly as it is in the sharp preview. The
    // whole range, including the two filters that re-sample: the backdrop that
    // stops at 12 is the backdrop that blurs a colour scene behind a pencil
    // sketch (found on device).
    //
    // Pencil arrives here WITHOUT the preblur the sharp preview gets — `render`
    // routes VIDEO_ID_FROSTED down the Kawase branch, which has no preblur pass.
    // That is the right trade and not an omission: the preblur exists to keep
    // Sobel off sensor noise, and every edge this pass emits is about to be
    // dual-Kawase'd across the screen anyway, so the extra noise is destroyed by
    // the next pass rather than shown. Buying it back would cost a full-res
    // pass on top of a chain already at ~2.6x screen area per frame.
    rgb_val = apply_texture_filter(
        rgb_val,
        viewport.filter_mode,
        tex_coords,
        texture_blur,
        sampler_blur,
    );

    // Opaque, always. The Kawase passes normalize by `sum.a` and treat a = 0 as
    // "outside the region" (see `video_shader_kawase.wgsl`), so this pass MUST
    // emit alpha = 1 across the whole target — letterbox included, which is why
    // the early-out above fills rather than discards. The rounded silhouette and
    // the dim belong to the final composite, not here.
    return vec4<f32>(rgb_val, 1.0);
}
