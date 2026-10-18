// SPDX-License-Identifier: GPL-3.0-only
use super::{PHOTO_WORKGROUP_SIZE, TILE_WORKGROUP_SIZE, bayer_workgroups};

fn shader_workgroup(source: &str) -> [u32; 2] {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(module.entry_points.len(), 1);
    let entry = &module.entry_points[0];
    assert_eq!(entry.stage, naga::ShaderStage::Compute);
    assert_eq!(entry.workgroup_size[2], 1);
    [entry.workgroup_size[0], entry.workgroup_size[1]]
}

#[test]
fn ordinary_photo_workgroup_matches_retained_performance_shape() {
    assert_eq!(PHOTO_WORKGROUP_SIZE, [64, 4]);
    assert_eq!(
        shader_workgroup(&super::super::compose_bayer_texture(include_str!(
            "debayer_photo.wgsl"
        ))),
        [64, 4]
    );
}

#[test]
fn photo_dispatch_covers_tiny_odd_and_partial_workgroups_once() {
    for (width, height, expected) in [
        (1, 1, [1, 1]),
        (2, 3, [1, 1]),
        (63, 3, [1, 1]),
        (64, 4, [1, 1]),
        (65, 5, [2, 2]),
        (127, 7, [2, 2]),
        (128, 8, [2, 2]),
        (129, 9, [3, 3]),
        (21, 19, [1, 5]),
    ] {
        let groups = bayer_workgroups(width, height, true, PHOTO_WORKGROUP_SIZE);
        assert_eq!(groups, expected, "{width}x{height}");
        let mut visits = vec![0; (width * height) as usize];
        for y in 0..groups[1] * 4 {
            for x in 0..groups[0] * 64 {
                // Same output guard as the photo shader, including partial groups.
                if x < width && y < height {
                    visits[(y * width + x) as usize] += 1;
                }
            }
        }
        assert!(visits.iter().all(|&count| count == 1), "{width}x{height}");
    }
}

#[test]
fn non_fused_dispatch_keeps_generic_video_and_high_quality_tiles() {
    assert_eq!(TILE_WORKGROUP_SIZE, 16);
    for (width, height, expected) in [
        (1, 1, [1, 1]),
        (16, 16, [1, 1]),
        (17, 17, [2, 2]),
        (65, 5, [5, 1]),
        (129, 19, [9, 2]),
    ] {
        assert_eq!(
            bayer_workgroups(width, height, false, PHOTO_WORKGROUP_SIZE),
            expected
        );
    }
    assert_eq!(
        shader_workgroup(&super::super::compose_bayer_texture(include_str!(
            "debayer_texture.wgsl"
        ))),
        [16, 16]
    );
    assert_eq!(
        shader_workgroup(&format!(
            "{}\n{}",
            super::super::BAYER_FINISH_FUNCTIONS,
            include_str!("debayer_finish.wgsl")
        )),
        [16, 16]
    );
}

#[test]
fn frozen_reference_dispatch_can_explicitly_keep_sixteen_by_sixteen() {
    assert_eq!(bayer_workgroups(65, 19, true, [16, 16]), [5, 2]);
    assert_eq!(bayer_workgroups(65, 19, true, PHOTO_WORKGROUP_SIZE), [2, 5]);
}

#[test]
fn dispatch_ceiling_does_not_overflow_large_dimensions() {
    assert_eq!(
        bayer_workgroups(u32::MAX, u32::MAX, true, [64, 4]),
        [67_108_864, 1_073_741_824]
    );
    assert_eq!(
        bayer_workgroups(u32::MAX, u32::MAX, false, [64, 4]),
        [268_435_456, 268_435_456]
    );
}
