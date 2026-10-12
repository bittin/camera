// SPDX-License-Identifier: GPL-3.0-only
//! Shared shader definitions and GPU pipelines
//!
//! This module provides the single source of truth for shader implementations.
//! All components (preview, photo capture, virtual camera) use these shared shaders.
//!
//! ## Pipelines
//!
//! - **YUV Convert**: Converts YUV frames (NV12, I420, YUYV) to RGBA on GPU
//! - **GPU Filter**: Applies visual filters (sepia, mono, etc.) to RGBA frames
//! - **Histogram**: Analyzes brightness distribution for exposure metering
//!
//! All pipelines operate on RGBA textures for uniform downstream processing.

pub(crate) mod demosaic;
pub(crate) mod demosaic_gpu;
pub use demosaic::DemosaicMethod;
mod gpu_convert;
mod gpu_filter;
mod histogram_pipeline;

pub use gpu_convert::{BayerError, GpuConvertPipeline, GpuFrameInput, get_gpu_convert_pipeline};
pub use gpu_filter::{GpuFilterPipeline, apply_filter_gpu_rgba, get_gpu_filter_pipeline};
pub use histogram_pipeline::{BrightnessMetrics, analyze_brightness_gpu};

/// Precompile all GPU shader pipelines so the first capture doesn't pay compilation cost.
///
/// Triggers device creation and pipeline compilation for both the convert and filter
/// pipelines. Safe to call from an async task at startup.
pub async fn warmup_gpu_pipelines() -> Result<(), String> {
    // 1. Warm up the convert pipeline (debayer, finish, yuv, awb, unpack)
    {
        let mut guard = get_gpu_convert_pipeline().await?;
        if let Some(pipeline) = guard.as_mut() {
            pipeline.warmup_pipelines();
        }
    } // Drop convert lock before acquiring filter lock

    // 2. Warm up the standalone filter pipeline (triggers device + pipeline creation)
    {
        let _guard = get_gpu_filter_pipeline().await?;
    }

    Ok(())
}

/// Shared filter functions (WGSL)
/// Contains: luminance(), hash(), apply_filter()
/// Used by: preview shaders, photo capture, virtual camera
pub const FILTER_FUNCTIONS: &str = include_str!("filters.wgsl");

/// Shared production demosaic and signed WB/CCM/highlight math.
pub const DEBAYER_CORE: &str = include_str!("debayer.wgsl");
pub fn compose_debayer(adapter: &str) -> String {
    format!("{DEBAYER_CORE}\n{adapter}")
}

pub(crate) const BAYER_IDENTITY_CCM: [[f32; 3]; 3] =
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Upper bound for the existing integer GPU gray-world producer.
/// `bayer_awb.wgsl` accumulates u32 sums (including any wrapping); finalize
/// converts each to f32, at most 2^32 after rounding. A nonzero channel sum
/// is at least 1, so `sg / (sr * 2)` and `sg / (sb * 2)` are at most 2^31.
/// A zero denominator selects unity; zero green can produce zero gains.
/// This bound does not change AWB, require a readback, or assume image size.
/// Re-audit this contract if either AWB shader changes its sum type/ratio.
pub(crate) const BAYER_AWB_GAIN_BOUND: f32 = 2_147_483_648.0;

/// Reject invalid/unsafe supplied colour metadata before GPU submission.
/// `sample_bound` includes merged-plane excursions, not only sensor white.
pub(crate) fn validate_bayer_colour(
    gains: Option<[f32; 2]>,
    ccm: Option<[[f32; 3]; 3]>,
    sample_bound: f64,
) -> Result<(), &'static str> {
    if !sample_bound.is_finite()
        || sample_bound < 0.0
        || gains.is_some_and(|g| g.iter().any(|v| !v.is_finite() || *v <= 0.0))
        || ccm.is_some_and(|m| m.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err("Invalid Bayer colour metadata or samples");
    }
    // Reserve headroom for bilinear sums and signed WB/CCM arithmetic.
    // RCD/LMMSE additionally validate their actual reconstructed RGB bound.
    if sample_bound > f64::from(f32::MAX) / 16.0 {
        return Err("Unsafe Bayer demosaic arithmetic");
    }
    // Missing gains mean unity here (HDR). The single-frame caller passes
    // its generated-AWB upper bound instead; CCM never depends on presence.
    let g = gains.unwrap_or([1.0, 1.0]);
    let weights = [f64::from(g[0]), 1.0, f64::from(g[1])];
    let limit = f64::from(f32::MAX) / 16.0;
    let m = ccm.unwrap_or(BAYER_IDENTITY_CCM);
    if weights.iter().any(|w| *w * sample_bound.max(1.0) > limit)
        || m.iter().any(|row| {
            row.iter()
                .zip(weights)
                .map(|(c, w)| f64::from(*c).abs() * w)
                .sum::<f64>()
                * sample_bound.max(1.0)
                > limit
        })
    {
        return Err("Unsafe Bayer colour arithmetic");
    }
    Ok(())
}

/// Shared texture-sampling filter functions (WGSL)
/// Contains: `apply_texture_filter()` — the filters that must re-sample the
/// source texture (Chromatic Aberration, Pencil), which is why they are not in
/// [`FILTER_FUNCTIONS`]: that prelude also feeds compute modules, and
/// `textureSample` is fragment-only.
/// Requires [`FILTER_FUNCTIONS`] ahead of it.
/// Used by: the sharp preview shader, pass 0 of the frosted blur chain
pub const TEXTURE_FILTER_FUNCTIONS: &str = include_str!("texture_filters.wgsl");

/// Shared UI-geometry functions (WGSL)
/// Contains: rounded_box_sdf()
/// Used by: the video shader, the frosted composite, the gallery thumbnail shader
pub const GEOMETRY_FUNCTIONS: &str = include_str!("geometry.wgsl");

#[cfg(test)]
pub(crate) fn apply_bayer_colour_reference(
    sensor_rgb: [f32; 3],
    sensor_peak: f32,
    gains: [f32; 2],
    ccm: [[f32; 3]; 3],
) -> [f32; 3] {
    let balanced = [
        sensor_rgb[0] * gains[0].max(0.0),
        sensor_rgb[1],
        sensor_rgb[2] * gains[1].max(0.0),
    ];
    let corrected: [f32; 3] = std::array::from_fn(|row| {
        (0..3)
            .map(|column| balanced[column] * ccm[row][column])
            .sum::<f32>()
    });
    let clipped_balanced = balanced.map(|channel| channel.clamp(0.0, 1.0));
    let clipped_corrected: [f32; 3] = std::array::from_fn(|row| {
        (0..3)
            .map(|column| clipped_balanced[column] * ccm[row][column])
            .sum::<f32>()
    });
    let peak = if sensor_peak.is_finite() {
        sensor_peak
    } else {
        0.0
    }
    .clamp(0.0, 1.0);
    let transition = ((peak - 0.980) / (0.995 - 0.980)).clamp(0.0, 1.0);
    let blend = transition * transition * (3.0 - 2.0 * transition);

    std::array::from_fn(|channel| {
        let finite_or_zero = |value: f32| if value.is_finite() { value } else { 0.0 };
        let original = finite_or_zero(corrected[channel]);
        let clipped = finite_or_zero(clipped_corrected[channel]);
        original + blend * (clipped - original)
    })
}

#[cfg(test)]
mod tests {
    use super::{apply_bayer_colour_reference, compose_debayer};

    const IDENTITY_CCM: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn assert_near(actual: [f32; 3], expected: [f32; 3]) {
        for channel in 0..3 {
            assert!(
                (actual[channel] - expected[channel]).abs() < 1.0e-6,
                "channel {channel}: expected {}, got {}",
                expected[channel],
                actual[channel]
            );
        }
    }

    fn validate_shader(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("WGSL should validate");
    }

    #[test]
    fn clipped_pixel_front_camera_highlight_is_neutral() {
        let sensor_rgb = [0.7296, 0.9954, 0.9771];
        let gains = [2.1832347, 1.6135153];
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];

        let output = apply_bayer_colour_reference(sensor_rgb, 0.99509333, gains, ccm);

        assert!(output.iter().all(|channel| channel.is_finite()));
        assert!(
            output.iter().all(|channel| *channel >= 0.80),
            "clipped white became non-white: {output:?}"
        );
        let spread = output.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - output.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            spread < 0.18,
            "clipped white retained a colour cast: {output:?}"
        );
    }

    #[test]
    fn clipped_pixel_matches_preview_colour() {
        let sensor_rgb = [0.7296, 0.9954, 0.9771];
        let gains = [2.1832347, 1.6135153];
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];

        let output = apply_bayer_colour_reference(sensor_rgb, 0.99509333, gains, ccm);

        assert_near(output, [1.0036639, 0.9909827, 1.0026946]);
    }

    #[test]
    fn in_gamut_raw_colour_is_unchanged() {
        let output = apply_bayer_colour_reference([0.2, 0.3, 0.1], 0.3, [2.0, 1.5], IDENTITY_CCM);

        assert_near(output, [0.4, 0.3, 0.15]);
    }

    #[test]
    fn unclipped_saturated_colour_preserves_overwhite() {
        let output = apply_bayer_colour_reference([0.9, 0.1, 0.1], 0.9, [2.0, 1.5], IDENTITY_CCM);

        assert_near(output, [1.8, 0.1, 0.15]);
    }

    #[test]
    fn unclipped_out_of_gamut_colour_preserves_signed_float() {
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];
        let output =
            apply_bayer_colour_reference([0.6, 0.6, 0.55], 0.6, [2.1832347, 1.6135153], ccm);

        assert_near(output, [2.0036466, 0.12574552, 0.97167516]);
    }

    #[test]
    fn highlight_knee_has_exact_original_and_recovered_endpoints() {
        let ccm = [
            [2.100_02, -0.7964949, -0.30352515],
            [-0.469_05, 1.9604946, -0.49144363],
            [-0.1991288, -0.585788, 1.7849169],
        ];
        let original =
            apply_bayer_colour_reference([0.6, 0.6, 0.55], 0.980, [2.1832347, 1.6135153], ccm);
        let recovered =
            apply_bayer_colour_reference([0.6, 0.6, 0.55], 0.995, [2.1832347, 1.6135153], ccm);

        assert_near(original, [2.0036466, 0.12574552, 0.97167516]);
        assert_near(recovered, [1.3527647, 0.27112326, 1.0333933]);
    }

    #[test]
    fn clipped_in_gamut_colour_is_unchanged() {
        let output = apply_bayer_colour_reference([0.9, 0.1, 0.1], 1.0, [1.0, 1.0], IDENTITY_CCM);

        assert_near(output, [0.9, 0.1, 0.1]);
    }

    #[test]
    fn non_finite_highlight_input_is_rejected_before_dispatch() {
        // Non-finite data is now rejected before dispatch, not repaired by a
        // second HDR-only colour shader.
        assert!(super::validate_bayer_colour(Some([f32::NAN, 1.0]), None, 1.0).is_err());
        assert!(super::validate_bayer_colour(None, Some([[f32::INFINITY; 3]; 3]), 1.0).is_err());
        assert!(super::validate_bayer_colour(None, None, f64::INFINITY).is_err());
    }

    #[test]
    fn shared_core_has_one_colour_and_lattice_implementation() {
        let core = super::DEBAYER_CORE;
        assert!(!core.contains("@binding"));
        assert!(core.contains("var corrected = linear;"));
        for adapter in [
            include_str!("debayer_texture.wgsl"),
            include_str!("burst_mode/bayer_finish.wgsl"),
        ] {
            assert!(adapter.contains("demosaic("));
            assert!(adapter.contains("apply_bayer_colour("));
            assert!(!adapter.contains("fn demosaic("));
            assert!(!adapter.contains("dot("));
            validate_shader(&compose_debayer(adapter));
        }
    }

    #[test]
    fn merged_samples_without_gains_must_not_overflow_demosaic() {
        assert!(super::validate_bayer_colour(None, None, f64::from(f32::MAX)).is_err());
    }

    #[test]
    fn ccm_without_gains_rejects_unsafe_arithmetic() {
        for matrix in [[[f32::MAX; 3]; 3], [[1e38, -1e38, 0.0]; 3]] {
            assert!(super::validate_bayer_colour(None, Some(matrix), 1.0).is_err());
        }
        assert!(super::validate_bayer_colour(None, Some(IDENTITY_CCM), 1.0).is_ok());
        assert!(super::validate_bayer_colour(None, Some([[1e20, -1e20, 0.0]; 3]), 1.0).is_ok());
    }

    #[test]
    fn optional_ccm_and_gains_are_independent_validation_policies() {
        // Identity is a missing-matrix fallback, never a missing-gains fallback.
        let safe = [[2.0, -1.0, 0.0], [-2.0, 1.0, 0.0], [0.0, 0.0, 1.5]];
        for gains in [
            None,
            Some([1.2, 0.8]),
            Some([super::BAYER_AWB_GAIN_BOUND; 2]),
        ] {
            for ccm in [None, Some(safe)] {
                assert!(super::validate_bayer_colour(gains, ccm, 1.0).is_ok());
            }
        }
        // Unity is safe for this matrix, but cannot stand in for generated AWB.
        let awb_unsafe = Some([[1e30, -1e30, 0.0]; 3]);
        assert!(super::validate_bayer_colour(None, awb_unsafe, 1.0).is_ok());
        assert!(
            super::validate_bayer_colour(Some([super::BAYER_AWB_GAIN_BOUND; 2]), awb_unsafe, 1.0)
                .is_err()
        );
        assert!(
            super::validate_bayer_colour(
                Some([super::BAYER_AWB_GAIN_BOUND; 2]),
                Some([[1e20, -1e20, 0.0]; 3]),
                1.0
            )
            .is_ok()
        );
    }

    #[test]
    fn generated_awb_bound_matches_integer_shader_contract() {
        // Tripwire: changing these shader contracts requires re-auditing the
        // proof above, not silently keeping a host bound for a different AWB.
        let accumulation = include_str!("bayer_awb.wgsl");
        let finalize = include_str!("bayer_awb_finalize.wgsl");
        for channel in ["r", "g", "b"] {
            assert!(accumulation.contains(&format!("sum_{channel}: atomic<u32>")));
            assert!(finalize.contains(&format!("sum_{channel}: u32")));
            assert!(finalize.contains(&format!("let s{channel} = f32(sums.sum_{channel});")));
        }
        assert!(finalize.contains("gains.gain_r = select(1.0, sg / (sr * 2.0), sr > 0.0);"));
        assert!(finalize.contains("gains.gain_b = select(1.0, sg / (sb * 2.0), sb > 0.0);"));
        // Cover zero fallback/zero gains and extrema, including u32->f32 rounding.
        for green in [0, 1, u32::MAX] {
            for channel in [0, 1, 2, u32::MAX] {
                let gain = if channel == 0 {
                    1.0
                } else {
                    green as f32 / (channel as f32 * 2.0)
                };
                assert!(gain.is_finite() && (0.0..=super::BAYER_AWB_GAIN_BOUND).contains(&gain));
            }
        }
        assert_eq!(u32::MAX as f32 / 2.0, super::BAYER_AWB_GAIN_BOUND);
    }

    #[test]
    fn composed_raw_shaders_validate() {
        validate_shader(&compose_debayer(include_str!("debayer_texture.wgsl")));
        validate_shader(include_str!("debayer_finish.wgsl"));
        validate_shader(&compose_debayer(include_str!(
            "burst_mode/bayer_finish.wgsl"
        )));
    }
}
