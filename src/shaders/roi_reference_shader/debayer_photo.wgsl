// SPDX-License-Identifier: GPL-3.0-only
// Bilinear photo development and finishing in one float-precision invocation.
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height { return; }
    textureStore(output, gid.xy, vec4(finish_bayer_colour(bayer_camera_rgb(gid.xy)), 1.0));
}
