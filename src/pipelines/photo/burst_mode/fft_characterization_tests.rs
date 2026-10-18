// SPDX-License-Identifier: GPL-3.0-only
//! Nonconstant native FFT characterization. No camera, codec or display required.
//!
//! For cross-revision evidence, run this exact test first on the baseline with
//! FFT_CHARACTERIZATION_RECORD=/absolute/empty/directory, then on the candidate
//! with FFT_CHARACTERIZATION_COMPARE pointing at that directory. Recording never
//! overwrites a fixture; comparison never regenerates expected pixels. Keep the
//! same adapter/driver and serialize both executions. Without these variables the
//! test still checks native bit-exact self-repeat, not cross-revision exactness.
use super::*;
use wgpu::util::DeviceExt;

fn planes(width: u32, height: u32, variant: u32) -> Vec<f32> {
    (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                std::array::from_fn::<_, 4, _>(|c| {
                    // Exact binary fractions, channel-specific gradients, impulses,
                    // checker texture and moving patches crossing tile boundaries.
                    let c = c as u32;
                    let code = (x * 37 + y * 53 + c * 71 + (x * y) % 101) % 1024;
                    let texture = if (x / 3 + y / 5 + c).is_multiple_of(2) {
                        0.125
                    } else {
                        -0.125
                    };
                    let patch = if (x + variant * 3) % 23 < 5 && (y + variant) % 19 < 4 {
                        0.5
                    } else {
                        0.0
                    };
                    code as f32 / 512.0 - 0.375 + texture + patch + variant as f32 / 128.0
                })
            })
        })
        .collect()
}

fn fixture(name: &str, bytes: &[u8]) {
    let record = std::env::var_os("FFT_CHARACTERIZATION_RECORD");
    let compare = std::env::var_os("FFT_CHARACTERIZATION_COMPARE");
    assert!(
        record.is_none() || compare.is_none(),
        "record and compare are exclusive"
    );
    if let Some(dir) = record {
        use std::io::Write;
        let dir = std::path::PathBuf::from(dir);
        assert!(dir.is_absolute(), "use an absolute fixture directory");
        std::fs::create_dir_all(&dir).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join(name))
            .unwrap();
        file.write_all(bytes).unwrap();
    } else if let Some(dir) = compare {
        let dir = std::path::PathBuf::from(dir);
        assert!(dir.is_absolute(), "use an absolute fixture directory");
        let expected = std::fs::read(dir.join(name)).unwrap();
        assert_eq!(bytes.len(), expected.len(), "{name}: fixture length");
        assert!(
            bytes == expected,
            "{name}: first differing byte {:?}",
            bytes.iter().zip(&expected).position(|(a, b)| a != b)
        );
    }
}

#[test]
fn nonconstant_fft_native_exact_characterization() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("nonconstant_fft_native_exact_characterization");
        return;
    };
    pollster::block_on(async {
        let gpu = super::super::BurstModeGpuPipeline::new().await.unwrap();
        // Tiny/partial tiles, exact tile extent, odd edges and >4 tile rows:
        // the last case exercises the chunked host row-offset boundary.
        for (width, height) in [(3, 5), (16, 16), (35, 83)] {
            let reference = planes(width, height, 0);
            fixture(
                &format!("{width}x{height}-reference.f32le"),
                &float_bytes(&reference),
            );
            let encoded: Vec<u8> = reference
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                .collect();
            let aligned: Vec<_> = (1..=2)
                .map(|variant| {
                    let data = planes(width, height, variant);
                    fixture(
                        &format!("{width}x{height}-aligned-{variant}.f32le"),
                        &float_bytes(&data),
                    );
                    GpuAlignedFrame {
                        buffer: gpu
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("nonconstant FFT aligned fixture"),
                                contents: bytemuck::cast_slice(&data),
                                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            }),
                        width,
                        height,
                    }
                })
                .collect();
            for count in 0..=2 {
                let frames = &aligned[..count];
                let mut previous = None;
                for repeat in 0..2 {
                    let buffer = gpu
                        .fft_pipeline
                        .merge_gpu_to_buffer(&reference, frames, width, height, 0.025, 1.0)
                        .await
                        .unwrap();
                    let linear = gpu
                        .read_buffer::<f32>(&buffer, reference.len())
                        .await
                        .unwrap();
                    assert!(
                        linear.iter().all(|v| v.is_finite()),
                        "{width}x{height}/{count}"
                    );
                    assert!(
                        linear.windows(2).any(|p| p[0].to_bits() != p[1].to_bits()),
                        "fixture collapsed to a constant"
                    );
                    let encoded_result = gpu
                        .fft_pipeline
                        .merge_gpu(&encoded, frames, width, height, 0.025, 1.0)
                        .await
                        .unwrap();
                    let current = (float_bytes(&linear), encoded_result);
                    if repeat == 0 {
                        fixture(
                            &format!("{width}x{height}-{count}-linear.f32le"),
                            &current.0,
                        );
                        fixture(
                            &format!("{width}x{height}-{count}-encoded.rgba8"),
                            &current.1,
                        );
                    }
                    if let Some(previous) = previous {
                        assert_eq!(
                            current, previous,
                            "{width}x{height}/{count}: native self-repeat"
                        );
                    }
                    previous = Some(current);
                }
            }
        }
    });
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}
