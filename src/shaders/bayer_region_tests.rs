// SPDX-License-Identifier: GPL-3.0-only
use super::*;

#[test]
fn roi_retained_dispatch_uses_final_extent() {
    let region = bayer_conversion_region(67, 65, true, Some((3, 5, 9, 7))).unwrap();
    assert_eq!(region, (3, 5, 9, 7));
    assert_eq!(bayer_workgroups(region.2, region.3, true), [1, 2]);
    assert_eq!(bayer_workgroups(67, 65, true), [2, 17]);
}

#[test]
fn roi_non_fused_and_full_call_preserve_full_sensor() {
    for (fused, requested) in [(false, Some((3, 5, 9, 7))), (true, None), (false, None)] {
        assert_eq!(
            bayer_conversion_region(67, 65, fused, requested).unwrap(),
            (0, 0, 67, 65)
        );
    }
}

#[test]
fn roi_bounds_are_overflow_safe_and_accept_empty_edge() {
    for region in [(0, 0, 1, 1), (66, 64, 1, 1), (1, 1, 66, 64), (67, 65, 0, 0)] {
        assert_eq!(
            bayer_conversion_region(67, 65, true, Some(region)).unwrap(),
            region
        );
    }
    for region in [(u32::MAX, 0, 2, 1), (66, 64, 2, 1), (0, 66, 1, 1)] {
        for fused in [false, true] {
            assert!(bayer_conversion_region(67, 65, fused, Some(region)).is_err());
        }
    }
}

#[test]
fn roi_uniform_preserves_original_96_byte_prefix() {
    assert_eq!(std::mem::size_of::<DebayerParams>(), 112);
    assert_eq!(std::mem::offset_of!(DebayerParams, width), 0);
    assert_eq!(std::mem::offset_of!(DebayerParams, height), 4);
    assert_eq!(std::mem::offset_of!(DebayerParams, region), 96);
}
