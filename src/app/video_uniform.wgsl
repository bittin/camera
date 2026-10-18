// SPDX-License-Identifier: GPL-3.0-only
// Canonical 160-byte video ABI; Rust offset tests pin every member.
struct ViewportUniform {
    viewport_size: vec2<f32>,   // Full widget size
    content_fit_mode: f32,      // 0.0 = Contain, 1.0 = Cover (interpolated during animation)
    filter_mode: u32,           // Filter index (0-15)
    corner_radius: f32,         // Corner radius in pixels (0 = no rounding)
    mirror_horizontal: u32,     // 0 = normal, 1 = mirrored horizontally
    uv_offset: vec2<f32>,       // UV offset for scroll clipping (0-1)
    uv_scale: vec2<f32>,        // UV scale for scroll clipping (0-1)
    crop_uv_min: vec2<f32>,     // Crop UV min (u_min, v_min) - normalized 0-1
    crop_uv_max: vec2<f32>,     // Crop UV max (u_max, v_max) - normalized 0-1
    zoom_level: f32,            // Zoom level (1.0 = no zoom, 2.0 = 2x zoom)
    rotation: u32,              // Sensor rotation: 0=None, 1=90CW, 2=180, 3=270CW
    bar_top_height: f32,        // Top bar height in pixels (for contain centering)
    bar_bottom_height: f32,     // Bottom bar height in pixels
    kawase_offset: f32,        // Unused here — read by the Kawase passes
    dim_factor: f32,           // Unused here — applied by the frosted composite
    letterbox_color: vec4<f32>, // RGBA — only used by the blur pass; declared here so the struct matches
    // The rect (x, y, w, h) the corners are cut from, in PHYSICAL px of the
    // render target, i.e. the same space as `@builtin(position)`.
    panel_rect: vec4<f32>,
    noise: f32,
    content_rect: vec4<f32>,
    bar_left_width: f32,
    bar_right_width: f32,
}

@group(0) @binding(2)
var<uniform> viewport: ViewportUniform;
