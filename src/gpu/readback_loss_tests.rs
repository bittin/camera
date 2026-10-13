// SPDX-License-Identifier: GPL-3.0-only
use super::*;

#[test]
fn native_device_loss_then_repeated_readback_returns_errors_not_mapped_panics() {
    let Some((_guard, _shared_device, _shared_queue)) =
        crate::test_gpu::headless_device_with_compute()
    else {
        crate::test_gpu::skip_no_gpu(
            "native_device_loss_then_repeated_readback_returns_errors_not_mapped_panics",
        );
        return;
    };
    // Separate device: do not destroy the shared fixture used by other tests.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    eprintln!("Device-loss adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("isolated readback loss device"),
        ..Default::default()
    }))
    .unwrap();
    device.set_device_lost_callback(|reason, message| {
        eprintln!("FIRST DEVICE LOSS: {reason:?}: {message}")
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("loss staging"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let owner = ReadbackBuffer::from_buffer(&device, buffer, "loss staging");
    queue.write_buffer(&owner.buffer(), 0, &[29; 16]);
    queue.submit([]);
    device.destroy();
    let first = owner
        .read()
        .err()
        .expect("lost device must not produce successful pixels");
    eprintln!("FIRST LOST-DEVICE READBACK ERROR: {first}");
    assert!(owner.read().is_err());
    // Also exercise fresh preparation on the lost device. Recovery requires a
    // new Device; reinitializing just a staging buffer must still fail safely.
    owner.buffer();
    assert!(owner.read().is_err());
}

#[test]
fn poll_error_after_native_map_completed_preserves_error_without_not_mapped_cleanup() {
    let Some((_guard, device, queue)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "poll_error_after_native_map_completed_preserves_error_without_not_mapped_cleanup",
        );
        return;
    };
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("completed poll error staging"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let owner = ReadbackBuffer::from_buffer(&device, buffer, "completed poll error staging");
    queue.write_buffer(&owner.buffer(), 0, &[41; 16]);
    queue.submit([]);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let expected = wgpu::PollError::Timeout.to_string();
    let result = pollster::block_on(owner.read_with_poll(|| {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        Err(wgpu::PollError::Timeout.to_string())
    }));
    assert_eq!(result.err().unwrap(), expected);
    queue.write_buffer(&owner.buffer(), 0, &[41; 16]);
    queue.submit([]);
    assert_eq!(&*owner.read().unwrap(), &[41; 16]);
    assert!(pollster::block_on(scope.pop()).is_none());
}
