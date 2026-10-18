// SPDX-License-Identifier: GPL-3.0-only

//! Integration tests for constants module

use camera::constants::BitratePreset;

#[test]
fn test_bitrate_preset_ordering() {
    // Test that presets are ordered from lowest to highest quality
    let mut prev_bitrate = 0u32;
    for preset in BitratePreset::ALL {
        let bitrate = preset.bitrate_kbps(1920, 1080);
        assert!(
            bitrate >= prev_bitrate,
            "Presets should be ordered from lowest to highest"
        );
        prev_bitrate = bitrate;
    }
}

#[test]
fn test_bitrate_scales_with_resolution() {
    // Higher resolution should have higher bitrate at same preset
    let hd_bitrate = BitratePreset::Medium.bitrate_kbps(1280, 720);
    let fhd_bitrate = BitratePreset::Medium.bitrate_kbps(1920, 1080);
    let uhd_bitrate = BitratePreset::Medium.bitrate_kbps(3840, 2160);

    assert!(hd_bitrate < fhd_bitrate);
    assert!(fhd_bitrate < uhd_bitrate);
}

#[test]
fn test_bitrate_values_at_every_width_boundary() {
    let cases = [
        (0, [1_000, 2_000, 4_000]),
        (640, [1_000, 2_000, 4_000]),
        (1279, [1_000, 2_000, 4_000]),
        (1280, [2_500, 5_000, 10_000]),
        (1281, [2_500, 5_000, 10_000]),
        (1919, [2_500, 5_000, 10_000]),
        (1920, [4_000, 8_000, 16_000]),
        (1921, [4_000, 8_000, 16_000]),
        (2559, [4_000, 8_000, 16_000]),
        (2560, [8_000, 16_000, 32_000]),
        (2561, [8_000, 16_000, 32_000]),
        (3839, [8_000, 16_000, 32_000]),
        (3840, [15_000, 30_000, 50_000]),
        (3841, [15_000, 30_000, 50_000]),
        (7680, [15_000, 30_000, 50_000]),
        (u32::MAX, [15_000, 30_000, 50_000]),
    ];
    for (width, expected) in cases {
        for height in [0, 1, 480, 1080, 2160, u32::MAX] {
            for (preset, expected) in BitratePreset::ALL.into_iter().zip(expected) {
                assert_eq!(
                    preset.bitrate_kbps(width, height),
                    expected,
                    "{preset:?} at {width}x{height}"
                );
            }
        }
    }
}
