// SPDX-License-Identifier: GPL-3.0-only
//! One bounded, negotiated RAW workspace, parked outside the shared converter.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Weak};

const MAX_PHOTO_WORKSPACE_BYTES: u64 = 192 * 1024 * 1024;
static ACTIVE_SESSION: Mutex<Weak<PhotoSession>> = Mutex::new(Weak::new());

// Keep only the fused RAW route's frame resources, not high-quality float scratch.
struct FrameResources {
    tex_y: Option<wgpu::Texture>,
    tex_uv: Option<wgpu::Texture>,
    tex_v: Option<wgpu::Texture>,
    output_texture: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    packed_input_buffer: Option<wgpu::Buffer>,
    packed_input_buffer_size: u64,
    unpack_output_buffer: Option<wgpu::Buffer>,
    unpack_output_buffer_size: u64,
    readback_staging_buffer: Option<crate::gpu::readback::ReadbackBuffer>,
    readback_staging_buffer_size: u64,
    cached_width: u32,
    cached_height: u32,
    cached_format: PixelFormat,
    cached_uv_dims: Option<(u32, u32)>,
}

impl Default for FrameResources {
    fn default() -> Self {
        Self {
            tex_y: None,
            tex_uv: None,
            tex_v: None,
            output_texture: None,
            output_view: None,
            packed_input_buffer: None,
            packed_input_buffer_size: 0,
            unpack_output_buffer: None,
            unpack_output_buffer_size: 0,
            readback_staging_buffer: None,
            readback_staging_buffer_size: 0,
            cached_width: 0,
            cached_height: 0,
            cached_format: PixelFormat::RGBA,
            cached_uv_dims: None,
        }
    }
}

impl FrameResources {
    fn swap(&mut self, pipeline: &mut GpuConvertPipeline) {
        macro_rules! swap {
            ($($field:ident),+ $(,)?) => {$(std::mem::swap(&mut self.$field, &mut pipeline.$field);)+};
        }
        swap!(
            tex_y,
            tex_uv,
            tex_v,
            output_texture,
            output_view,
            packed_input_buffer,
            packed_input_buffer_size,
            unpack_output_buffer,
            unpack_output_buffer_size,
            readback_staging_buffer,
            readback_staging_buffer_size,
            cached_width,
            cached_height,
            cached_format,
            cached_uv_dims
        );
    }
}

struct SessionState {
    resources: FrameResources,
}

pub(crate) struct PhotoSession {
    width: u32,
    height: u32,
    format: PixelFormat,
    layout: RawLayout,
    stride: u32,
    packed_size: u64,
    unpack_size: u64,
    staging_size: u64,
    active: AtomicBool,
    state: Mutex<SessionState>,
}

impl std::fmt::Debug for PhotoSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotoSession")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// The capture thread owns this guard; queued photo jobs only clone the session.
/// Ending a session invalidates parked resources even while old jobs still exist.
pub(crate) struct PhotoSessionGuard(Arc<PhotoSession>);
impl Drop for PhotoSessionGuard {
    fn drop(&mut self) {
        self.0.end();
    }
}

pub(crate) fn current_photo_session() -> Option<Arc<PhotoSession>> {
    ACTIVE_SESSION
        .lock()
        .unwrap()
        .upgrade()
        .filter(|s| s.active.load(Ordering::Acquire))
}

impl PhotoSession {
    pub(super) fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        layout: RawLayout,
        stride: u32,
    ) -> Option<Arc<Self>> {
        if width < 2 || height < 2 || !format.is_bayer() {
            return None;
        }
        let row = match layout {
            RawLayout::U8 => u64::from(width),
            RawLayout::Unpacked16 {
                bits: 10 | 12 | 14 | 16,
            } => u64::from(width) * 2,
            RawLayout::Csi2 { bits: 10 } => u64::from(width).div_ceil(4) * 5,
            RawLayout::Csi2 { bits: 12 } => u64::from(width).div_ceil(2) * 3,
            RawLayout::Csi2 { bits: 14 } => u64::from(width).div_ceil(4) * 7,
            _ => return None,
        };
        if u64::from(stride) < row {
            return None;
        }
        let pixels = u64::from(width).checked_mul(u64::from(height))?;
        let staging_size = (u64::from(width) * 4)
            .next_multiple_of(256)
            .checked_mul(u64::from(height))?;
        let (packed_size, unpack_size) = if matches!(layout, RawLayout::Csi2 { .. }) {
            (
                (u64::from(stride) * u64::from(height)).next_multiple_of(4),
                (u64::from(width) * 2).next_multiple_of(256) * u64::from(height),
            )
        } else {
            (0, 0)
        };
        let bytes = pixels
            .checked_mul(6)?
            .checked_add(staging_size)?
            .checked_add(packed_size)?
            .checked_add(unpack_size)?
            .checked_add(1024 * 1024)?;
        if bytes > MAX_PHOTO_WORKSPACE_BYTES {
            return None;
        }
        Some(Arc::new(Self {
            width,
            height,
            format,
            layout,
            stride,
            packed_size,
            unpack_size,
            staging_size,
            active: AtomicBool::new(true),
            state: Mutex::new(SessionState {
                resources: FrameResources::default(),
            }),
        }))
    }

    pub(crate) fn start(
        width: u32,
        height: u32,
        format: PixelFormat,
        layout: RawLayout,
        stride: u32,
    ) -> Option<PhotoSessionGuard> {
        let session = Self::new(width, height, format, layout, stride)?;
        let old = {
            let mut active = ACTIVE_SESSION.lock().unwrap();
            let old = active.upgrade();
            *active = Arc::downgrade(&session);
            old
        };
        if let Some(old) = old {
            old.end();
        }
        let background = Arc::clone(&session);
        if let Err(error) = std::thread::Builder::new().name("photo-preallocate".into()).spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(runtime) => runtime,
                Err(error) => {
                    warn!(%error, "Could not create photo preallocation runtime");
                    return;
                }
            };
            runtime.block_on(async {
                crate::gpu::wait_for_renderer_seed_or_timeout(std::time::Duration::from_secs(2)).await;
                if !background.active.load(Ordering::Acquire) { return; }
                let result = async {
                    let mut guard = get_gpu_convert_pipeline().await?;
                    background.prepare(guard.as_mut().ok_or("Convert pipeline not initialized")?)
                }.await;
                if let Err(error) = result { warn!(%error, "Photo preallocation unavailable; using capture-time allocation"); }
            });
        }) {
            warn!(%error, "Could not start photo preallocation worker");
        }
        Some(PhotoSessionGuard(session))
    }

    #[cfg(test)]
    pub(super) fn has_workspace(&self) -> bool {
        self.state.lock().unwrap().resources.tex_y.is_some()
    }

    pub(super) fn end(&self) {
        self.active.store(false, Ordering::Release);
        self.discard();
    }

    pub(super) fn discard(&self) {
        let mut state = self.state.lock().unwrap();
        state.resources = FrameResources::default();
    }

    pub(super) fn prepare(&self, pipeline: &mut GpuConvertPipeline) -> Result<(), String> {
        let limits = pipeline.device.limits();
        if !pipeline.r16_input_storage_capable
            || self.width > limits.max_texture_dimension_2d
            || self.height > limits.max_texture_dimension_2d
            || self.staging_size > limits.max_buffer_size
            || self.packed_size > u64::from(limits.max_storage_buffer_binding_size)
            || self.packed_size > limits.max_buffer_size
            || self.unpack_size > limits.max_buffer_size
        {
            return Err("Photo reservation exceeds GPU capabilities".into());
        }
        if !self.active.load(Ordering::Acquire)
            || self.state.lock().unwrap().resources.tex_y.is_some()
        {
            return Ok(());
        }
        // Temporarily borrow the converter's allocator without evicting preview/recording state.
        let mut previous = FrameResources::default();
        previous.swap(pipeline);
        pipeline.ensure_resources(self.width, self.height, self.format, None);
        if self.packed_size != 0 {
            pipeline.ensure_unpack_buffers(self.packed_size, self.width, self.height);
        }
        pipeline.ensure_photo_readback(self.staging_size);
        let mut prepared = FrameResources::default();
        prepared.swap(pipeline);
        previous.swap(pipeline);
        let mut state = self.state.lock().unwrap();
        if self.active.load(Ordering::Acquire) {
            state.resources = prepared;
        }
        info!(
            width = self.width,
            height = self.height,
            "Photo workspace preallocated"
        );
        Ok(())
    }

    pub(super) fn matches(
        &self,
        input: &GpuFrameInput,
        method: super::super::DemosaicMethod,
    ) -> bool {
        method == super::super::DemosaicMethod::Bilinear
            && input.width == self.width
            && input.height == self.height
            && input.format == self.format
            && input.raw_layout == Some(self.layout)
            && input.y_stride == self.stride
            && (self.packed_size == 0 || input.y_data.len() as u64 <= self.packed_size)
    }

    pub(super) fn take(&self, pipeline: &mut GpuConvertPipeline) -> bool {
        let mut state = self.state.lock().unwrap();
        if !self.active.load(Ordering::Acquire) {
            return false;
        }
        state.resources.swap(pipeline);
        true
    }

    pub(super) fn store(&self, pipeline: &mut GpuConvertPipeline) {
        let mut state = self.state.lock().unwrap();
        if self.active.load(Ordering::Acquire)
            && pipeline.cached_width == self.width
            && pipeline.cached_height == self.height
            && pipeline.cached_format == self.format
            && pipeline.bayer_linear_texture.is_none()
            && pipeline.demosaic_engine.is_none()
            && pipeline.readback_staging_buffer_size <= self.staging_size
            && pipeline.packed_input_buffer_size <= self.packed_size
            && pipeline.unpack_output_buffer_size <= self.unpack_size
        {
            state.resources.swap(pipeline);
        }
    }
}
