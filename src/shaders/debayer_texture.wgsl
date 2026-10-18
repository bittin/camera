// SPDX-License-Identifier: GPL-3.0-only
// Linear adapter composed with the shared Bayer texture functions.
@group(0) @binding(1) var output: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= params.region.zw) { return; }
    let q = gid.xy + params.region.xy;
    textureStore(output, q, vec4(bayer_camera_rgb(q), 1.0));
}
