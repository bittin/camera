// SPDX-License-Identifier: GPL-3.0-only
use super::*;

#[test]
fn failed_native_map_then_cached_demosaic_reuse_preserves_first_error() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "failed_native_map_then_cached_demosaic_reuse_preserves_first_error",
        );
        return;
    };
    let engine = GpuDemosaic::new(&device, 2, 2).unwrap();
    engine.staging_for_test().destroy();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let first = engine
        .validate(&device)
        .expect_err("destroyed buffer must fail mapping");
    eprintln!(
        "FIRST BACKEND DIAGNOSTIC: {:?}",
        pollster::block_on(scope.pop())
    );
    engine
        .validate(&device)
        .expect_err("unprepared retry must fail without already-mapped panic");
    eprintln!("FIRST MAP FAILURE: {first}");
    assert!(
        first.contains("map") || first.contains("Mapping"),
        "{first}"
    );
    // Obtain the next cached buffer, and put a nonzero status value into it.
    let staging = engine.staging_for_test();
    queue.write_buffer(
        &staging,
        0,
        bytemuck::cast_slice(&[0_u32, 0, 0.75_f32.to_bits(), 0]),
    );
    queue.submit([]);
    assert_eq!(engine.validate(&device).unwrap(), 0.75);
    assert_eq!(engine.validate(&device).unwrap(), 0.75);
}
