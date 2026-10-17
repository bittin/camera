// SPDX-License-Identifier: GPL-3.0-only

//! Shared GPU test harness.
//!
//! Exists so that every GPU test in the crate goes through ONE `wgpu::Instance`.
//! Building an instance per test is what made `cargo test --lib` SIGSEGV inside
//! lavapipe at default parallelism, so the device lives here rather than in any
//! one test module — and any new GPU test should reach for `headless_device()`
//! instead of standing up its own.

use iced_wgpu::wgpu;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

pub(crate) type PhotoEvents = Arc<Mutex<Vec<BTreeMap<String, String>>>>;

/// Generated RAW10 highlight with deliberately signed, out-of-gamut CCM candidates.
pub(crate) fn synthetic_highlight_frame() -> Arc<crate::backends::camera::types::CameraFrame> {
    use crate::backends::camera::types::{
        CameraFrame, FrameData, FrameMetadata, PixelFormat, RawLayout,
    };
    let mut data = vec![0xa5; 44 * 32];
    for y in 0..32 {
        for group in 0..8 {
            let base = y * 44 + group * 5;
            data[base + 4] = 0;
            for k in 0..4 {
                let code: u16 = match (k % 2, y % 2) {
                    (0, 0) => 409,
                    (1, 1) => 511,
                    _ => 972,
                };
                data[base + k] = (code >> 2) as u8;
                data[base + 4] |= ((code & 3) as u8) << (k * 2);
            }
        }
    }
    Arc::new(CameraFrame {
        width: 32,
        height: 32,
        stride: 44,
        format: PixelFormat::BayerRGGB,
        data: FrameData::from_owned_vec(data),
        yuv_planes: None,
        captured_at: std::time::Instant::now(),
        sensor_timestamp_ns: None,
        libcamera_metadata: Some(FrameMetadata {
            raw_layout: Some(RawLayout::Csi2 { bits: 10 }),
            colour_gains: Some([3., 3.]),
            colour_correction_matrix: Some([[0., -1., 1.5], [3., 0., -2.5], [0., 1., 0.]]),
            black_level: Some(0.),
            ..Default::default()
        }),
    })
}

/// Thread-local subscribers use this to observe completed production GPU work.
#[derive(Clone, Default)]
pub(crate) struct PhotoTrace(pub(crate) PhotoEvents);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for PhotoTrace {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        struct Fields(BTreeMap<String, String>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name().into(), format!("{value:?}"));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut fields = Fields(BTreeMap::new());
        event.record(&mut fields);
        if fields
            .0
            .get("message")
            .is_some_and(|m| m.starts_with("Photo "))
        {
            self.0.lock().unwrap().push(fields.0);
        }
    }
}

impl PhotoTrace {
    pub(crate) fn completed_stages(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|e| {
                e.get("message")
                    .is_some_and(|m| m == "Photo GPU batch completed")
            })
            .map(|e| e["stage"].trim_matches('"').to_owned())
            .collect()
    }
}

/// The ONE headless wgpu device the GPU tests share, or `None` when the
/// machine has no usable adapter.
///
/// Shared, and it has to be. Each `wgpu::Instance` loads and initialises the
/// Vulkan ICD; building ten of them concurrently — which is exactly what
/// `cargo test`'s default thread pool did, one per GPU test — SIGSEGV'd
/// inside lavapipe about half the time. The device is only ever read from
/// here, and wgpu's `Device`/`Queue` are internally synchronised Arc handles,
/// so one instance for the whole binary is both correct and enough.
static HEADLESS_DEVICE: LazyLock<Option<(wgpu::Device, wgpu::Queue)>> = LazyLock::new(|| {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    eprintln!(
        "Headless fixture adapter: {:?}; limits: {:?}",
        adapter.get_info(),
        adapter.limits()
    );
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("frosted corner test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::default(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        trace: wgpu::Trace::Off,
    }))
    .ok()
});

/// Serializes use of [`HEADLESS_DEVICE`].
///
/// wgpu's GLES backend protects one adapter context with a non-reentrant lock.
/// Concurrent commands from cloned `Device` handles can hold that context long
/// enough for another test to hit wgpu-hal's deadlock timeout. Vulkan permits
/// this concurrency, but Alpine's riscv64 runner only exposes GLES.
static HEADLESS_DEVICE_LOCK: Mutex<()> = Mutex::new(());

/// Exclusive access to the shared headless device, or `None` when the machine
/// has no usable adapter (CI) — in which case the caller skips via
/// [`skip_no_gpu`] rather than fails.
///
/// Keep the returned guard alive until every GPU resource created by the test
/// has been dropped. The tuple shape makes the guard explicit at each call site.
pub(crate) fn headless_device() -> Option<(MutexGuard<'static, ()>, wgpu::Device, wgpu::Queue)> {
    let guard = HEADLESS_DEVICE_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    HEADLESS_DEVICE
        .clone()
        .map(|(device, queue)| (guard, device, queue))
}

/// Exclusive preview device access for tests that also use production compute.
///
/// The preview fixture can use GLES on machines without Vulkan, but production
/// compute requires Vulkan. Check both before entering a mixed-path test so it
/// follows the same no-adapter skip policy as preview-only tests. Pipeline
/// creation and conversion errors remain assertions in the caller.
pub(crate) fn headless_device_with_compute()
-> Option<(MutexGuard<'static, ()>, wgpu::Device, wgpu::Queue)> {
    let fixture = headless_device()?;
    if let Err(error) = pollster::block_on(crate::gpu::get_shared_gpu()) {
        eprintln!("Production compute device unavailable: {error}");
        return None;
    }
    Some(fixture)
}

/// Called wherever a GPU test degrades to a skip because there is no adapter.
///
/// A `println!` and a `return` is a PASS, so without this every GPU test in the
/// crate reports green on an adapter-less machine while asserting nothing — and
/// these tests are the only thing between this branch and the black-bars
/// regression they were written for. Set `CI_REQUIRE_GPU=1` to turn the skip
/// into a hard failure, so CI can demand real coverage while local dev on a
/// headless box still skips gracefully.
pub(crate) fn skip_no_gpu(what: &str) {
    assert!(
        std::env::var("CI_REQUIRE_GPU").as_deref() != Ok("1"),
        "{what} needs a GPU adapter and CI_REQUIRE_GPU=1 forbids skipping, but no \
         usable wgpu adapter was found"
    );
    println!("Skipping {what} (no GPU adapter)");
}
