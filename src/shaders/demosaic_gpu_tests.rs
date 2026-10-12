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
    let rgba: Vec<[f32; 4]> = input.iter().map(|&v| [v, 0.0, 0.0, 0.0]).collect();
    queue.write_texture(
        engine.raw.as_image_copy(),
        bytemuck::cast_slice(&rgba),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 16),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    let output = texture(device, w, h, "test camera RGB");
    let mut encoder = device.create_command_encoder(&Default::default());
    engine.encode(device, queue, &mut encoder, &output, pattern, method);
    queue.submit([encoder.finish()]);
    engine.validate(device)?;
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
