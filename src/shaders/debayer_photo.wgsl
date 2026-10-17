// SPDX-License-Identifier: GPL-3.0-only
// Bilinear photo development and finishing in one float-precision invocation.
@group(0) @binding(1) var output: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(64,4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.region.z || gid.y >= params.region.w { return; }
    let position = gid.xy + params.region.xy;
    textureStore(output, position, vec4(finish_bayer_colour(bayer_camera_rgb(position)), 1.0));
}
