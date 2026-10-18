// SPDX-License-Identifier: GPL-3.0-only
// GPU demosaic resource and arithmetic validation.
use super::*;
#[allow(clippy::too_many_arguments)]
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    engine: &GpuDemosaic,
    input: &[f32],
    w: u32,
    h: u32,
    pattern: u32,
    method: DemosaicMethod,
) -> Result<Vec<[f32; 4]>, String> {
    run_mode(device, queue, engine, input, w, h, pattern, method, false)
}

#[allow(clippy::too_many_arguments)]
fn run_mode(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    engine: &GpuDemosaic,
    input: &[f32],
    w: u32,
    h: u32,
    pattern: u32,
    method: DemosaicMethod,
    paced: bool,
) -> Result<Vec<[f32; 4]>, String> {
    queue.write_texture(
        engine.raw.as_image_copy(),
        bytemuck::cast_slice(input),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    let output = texture(device, w, h, "test camera RGB");
    if paced {
        engine.encode_photo(device, queue, &output, pattern, method)?;
    } else {
        let mut encoder = device.create_command_encoder(&Default::default());
        engine.encode(device, queue, &mut encoder, &output, pattern, method);
        queue.submit([encoder.finish()]);
    }
    engine.validate()?;
    let row = (w * 16).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test decoded float RGB"),
        size: u64::from(row) * u64::from(h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |v| {
        tx.send(v).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let data = buffer.slice(..).get_mapped_range();
    let pixels: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let out = pixels
        .chunks_exact(row as usize / 16)
        .flat_map(|r| r[..w as usize].iter().copied())
        .collect();
    drop(data);
    buffer.unmap();
    Ok(out)
}
#[test]
fn normalized_cfa_uses_one_float_component_without_precision_loss() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "normalized_cfa_uses_one_float_component_without_precision_loss",
        );
        return;
    };
    let engine = GpuDemosaic::new(&device, 2, 2).unwrap();
    assert_eq!(
        engine.raw.format().block_copy_size(None),
        Some(4),
        "scalar CFA wastes unused float components"
    );
    // No FP16/normalized narrowing: a one-ULP distinction must survive upload
    // and the unchanged GPU sensor-peak output channel.
    let value = f32::from_bits(0.5f32.to_bits() + 1);
    let out = run(
        &device,
        &queue,
        &engine,
        &[value; 4],
        2,
        2,
        0,
        DemosaicMethod::Bilinear,
    )
    .unwrap();
    assert!(out.iter().all(|p| p[3].to_bits() == value.to_bits()));
}

#[test]
fn photo_scratch_is_bounded_independently_of_sensor_geometry() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("photo_scratch_is_bounded_independently_of_sensor_geometry");
        return;
    };
    let (w, h) = (1057, 1031);
    let engine = GpuDemosaic::new(&device, w, h).unwrap();
    let bytes: u64 = engine
        .scratch
        .iter()
        .map(|t| u64::from(t.width()) * u64::from(t.height()) * 16)
        .sum();
    assert!(
        bytes <= 4 * 544 * 544 * 16,
        "photo scratch must be tile-bounded, allocated {bytes} bytes"
    );
    let raw = vec![0.2; (w * h) as usize];
    for method in [
        DemosaicMethod::Rcd,
        DemosaicMethod::Lmmse,
        DemosaicMethod::Bilinear,
    ] {
        let out = run(&device, &queue, &engine, &raw, w, h, 0, method).unwrap();
        assert_eq!(out.len(), raw.len());
        assert!(out.iter().all(|v| v.iter().all(|c| c.is_finite())));
    }
}

#[test]
fn gpu_reduction_rejects_nan_infinity_and_overflow_without_clamping() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "gpu_reduction_rejects_nan_infinity_and_overflow_without_clamping",
        );
        return;
    };
    let engine = GpuDemosaic::new(&device, 2, 2).unwrap();
    for method in [
        DemosaicMethod::Rcd,
        DemosaicMethod::Lmmse,
        DemosaicMethod::Bilinear,
    ] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(run(&device, &queue, &engine, &[value; 4], 2, 2, 0, method).is_err());
        }
    }
    for method in [DemosaicMethod::Rcd, DemosaicMethod::Lmmse] {
        assert!(
            run(
                &device,
                &queue,
                &engine,
                &[f32::MAX, -f32::MAX, f32::MAX, -f32::MAX],
                2,
                2,
                0,
                method
            )
            .is_err()
        );
    }
    // Reusing the same reduction buffers after rejection must clear the status.
    assert!(
        run(
            &device,
            &queue,
            &engine,
            &[0.2; 4],
            2,
            2,
            0,
            DemosaicMethod::Rcd
        )
        .is_ok()
    );
}
#[test]
fn photo_global_bounds_halos_and_finite_status_match_unpaced_bits() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device() else {
        crate::test_gpu::skip_no_gpu(
            "photo_global_bounds_halos_and_finite_status_match_unpaced_bits",
        );
        return;
    };
    let (w, h) = (515, 517);
    let engine = GpuDemosaic::new(&device, w, h).unwrap();
    // Neither extreme appears in the first bounds band or output tile. Most
    // pixels are negative: a per-band reset would change LMMSE's transform.
    let mut input: Vec<f32> = (0..w * h)
        .map(|i| ((i * 977 % 1021) as f32 / 1021.0 - 0.9) * 0.2)
        .collect();
    input[(w * h - 2) as usize] = -0.14798465;
    input[(w * h - 1) as usize] = 1.0;
    for method in [DemosaicMethod::Rcd, DemosaicMethod::Lmmse] {
        for pattern in 0..4 {
            let expected = run(&device, &queue, &engine, &input, w, h, pattern, method).unwrap();
            let actual = run_mode(
                &device, &queue, &engine, &input, w, h, pattern, method, true,
            )
            .unwrap();
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&actual),
                bytemuck::cast_slice::<_, u8>(&expected),
                "{method:?} CFA {pattern}: float bits changed"
            );
        }
        for index in [0, (w * h - 1) as usize] {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let mut bad = input.clone();
                bad[index] = invalid;
                assert!(
                    run_mode(&device, &queue, &engine, &bad, w, h, 0, method, true).is_err(),
                    "{method:?} index={index} value={invalid}"
                );
            }
        }
        assert!(
            run_mode(&device, &queue, &engine, &input, w, h, 0, method, true).is_ok(),
            "status must reset between photos"
        );
    }
}

#[test]
fn halo_dimensions_are_rejected_before_gpu_allocation() {
    let Some((_guard, device, _queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("halo_dimensions_are_rejected_before_gpu_allocation");
        return;
    };
    assert!(GpuDemosaic::new(&device, device.limits().max_texture_dimension_2d, 2).is_err());
    assert!(GpuDemosaic::new(&device, u32::MAX, 2).is_err());
}

#[test]
fn safe_extreme_constant_sensor_planes_do_not_overflow_gpu_ratios() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "safe_extreme_constant_sensor_planes_do_not_overflow_gpu_ratios",
        );
        return;
    };
    let engine = GpuDemosaic::new(&device, 2, 2).unwrap();
    for method in [DemosaicMethod::Rcd, DemosaicMethod::Lmmse] {
        for value in [f32::MAX / 32.0, -f32::MAX / 32.0] {
            let out = run(&device, &queue, &engine, &[value; 4], 2, 2, 0, method).unwrap();
            for pixel in out {
                for c in 0..3 {
                    assert!(
                        (pixel[c] / value - 1.0).abs() < 2e-5,
                        "{method:?} {value} {pixel:?}"
                    );
                }
            }
        }
    }
}
