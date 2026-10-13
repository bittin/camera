// SPDX-License-Identifier: GPL-3.0-only
//! Lossless packing shared by mapped readback and CPU photo geometry.

/// Copy active RGBA pixels to an exactly sized RGB destination, ignoring alpha.
/// Both slices may be unaligned. Never read source padding or write past RGB.
pub(crate) fn copy_rgba_pixels_to_rgb(source: &[u8], dest: &mut [u8]) {
    assert_eq!(source.len() % 4, 0);
    assert_eq!(dest.len() / 3, source.len() / 4);
    assert_eq!(dest.len() % 3, 0);
    let mut pixels = 0;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use std::arch::aarch64::{uint8x16x3_t, vld4q_u8, vst3q_u8};
        while source.len() / 4 - pixels >= 16 {
            // SAFETY: NEON is enabled for this target. The checked loop leaves
            // 64 readable bytes and 48 writable bytes for each unaligned block;
            // safe disjoint slices prohibit overlap, and no padding is accessed.
            unsafe {
                let channels = vld4q_u8(source.as_ptr().add(pixels * 4));
                vst3q_u8(
                    dest.as_mut_ptr().add(pixels * 3),
                    uint8x16x3_t(channels.0, channels.1, channels.2),
                );
            }
            pixels += 16;
        }
    }
    for (rgb, rgba) in dest[pixels * 3..]
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(source[pixels * 4..].as_chunks::<4>().0.iter())
    {
        rgb.copy_from_slice(&rgba[..3]);
    }
}
