// SPDX-License-Identifier: GPL-3.0-only
use super::*;
use std::future::Future;
use std::task::{Context, Poll};

fn fixture(device: &wgpu::Device) -> ReadbackBuffer {
    ReadbackBuffer::from_buffer(
        device,
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback lifecycle fixture"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        "readback lifecycle fixture",
    )
}
fn write_sentinel(owner: &ReadbackBuffer, queue: &wgpu::Queue) -> wgpu::Buffer {
    let buffer = owner.buffer();
    queue.write_buffer(&buffer, 0, &[37; 16]);
    queue.submit([]);
    buffer
}

#[test]
fn native_pending_poll_failure_preserves_diagnostic_and_recreates_before_reuse() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "native_pending_poll_failure_preserves_diagnostic_and_recreates_before_reuse",
        );
        return;
    };
    let owner = fixture(&device);
    let old = write_sentinel(&owner, &queue);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result =
        pollster::block_on(owner.read_with_poll(|| Err("injected first poll failure".into())));
    assert_eq!(result.err().unwrap(), "injected first poll failure");
    assert!(owner.read().is_err());
    assert_ne!(write_sentinel(&owner, &queue), old);
    assert_eq!(&*owner.read().unwrap(), &[37; 16]);
    assert_eq!(&*owner.read().unwrap(), &[37; 16]);
    assert!(
        pollster::block_on(scope.pop()).is_none(),
        "cleanup must not cause NotMapped validation"
    );
}

#[test]
fn native_pending_future_cancellation_discards_map_before_cached_reuse() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "native_pending_future_cancellation_discards_map_before_cached_reuse",
        );
        return;
    };
    let owner = fixture(&device);
    let old = write_sentinel(&owner, &queue);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut future = Box::pin(owner.read_with_poll(|| Ok(())));
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    drop(future);
    assert!(owner.read().is_err());
    assert_ne!(write_sentinel(&owner, &queue), old);
    assert_eq!(&*owner.read().unwrap(), &[37; 16]);
    assert!(pollster::block_on(scope.pop()).is_none());
}

#[test]
fn native_completed_callback_future_cancellation_discards_without_double_unmap() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "native_completed_callback_future_cancellation_discards_without_double_unmap",
        );
        return;
    };
    let owner = fixture(&device);
    let old = write_sentinel(&owner, &queue);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut future = Box::pin(owner.read_with_poll(|| Ok(())));
    let waker = futures::task::noop_waker();
    assert!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    drop(future); // callback completed, consumer has not observed it
    assert_ne!(write_sentinel(&owner, &queue), old);
    assert_eq!(&*owner.read().unwrap(), &[37; 16]);
    assert!(pollster::block_on(scope.pop()).is_none());
}

#[test]
fn cached_production_readbacks_use_owned_lifecycle() {
    for (name, source) in [
        ("convert", include_str!("../shaders/gpu_convert.rs")),
        ("filter", include_str!("../shaders/gpu_filter.rs")),
        (
            "histogram",
            include_str!("../shaders/histogram_pipeline.rs"),
        ),
        (
            "virtual filter",
            include_str!("../backends/virtual_camera/gpu_filter.rs"),
        ),
    ] {
        assert!(
            !source.contains("Option<wgpu::Buffer>,\n") || source.contains("ReadbackBuffer"),
            "{name} cached staging has no lifecycle owner"
        );
        assert!(
            !source.contains("map_async(wgpu::MapMode::Read"),
            "{name} still maps outside owned lifecycle"
        );
    }
}
