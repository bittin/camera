// SPDX-License-Identifier: GPL-3.0-only
//! GPU-accelerated filter pipeline for images
//!
//! This module provides a unified GPU filter pipeline used by both photo capture
//! and virtual camera for consistent filter application.
//! It uses wgpu with software rendering fallback for systems without GPU support.

use crate::app::FilterType;
use crate::gpu::{self, wgpu};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Filter parameters uniform
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FilterParams {
    width: u32,
    height: u32,
    filter_mode: u32,
    _padding: u32,
}

/// Blur parameters uniform for the pre-blur compute shader
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    width: u32,
    height: u32,
    _padding: [u32; 2],
}

/// GPU filter pipeline for images
pub struct GpuFilterPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    // Pre-blur compute pipeline for multi-pass filters
    preblur_pipeline: wgpu::ComputePipeline,
    preblur_bind_group_layout: wgpu::BindGroupLayout,
    preblur_uniform_buffer: wgpu::Buffer,
    // Cached resources for current dimensions
    cached_width: u32,
    cached_height: u32,
    input_texture: Option<wgpu::Texture>,
    // Intermediate texture for pre-blur output (storage texture, same dimensions as input)
    preblur_texture: Option<wgpu::Texture>,
    output_buffer: Option<wgpu::Buffer>,
    staging_buffer: Option<crate::gpu::readback::ReadbackBuffer>,
}

/// Keep only compiled programs/uniforms after a photo, including cancellation.
struct PhotoFilterWorkspace<'a> {
    pipeline: &'a mut GpuFilterPipeline,
}

impl Drop for PhotoFilterWorkspace<'_> {
    fn drop(&mut self) {
        self.pipeline.release_frame_resources();
    }
}

impl GpuFilterPipeline {
    fn release_frame_resources(&mut self) {
        self.input_texture = None;
        self.preblur_texture = None;
        self.output_buffer = None;
        self.staging_buffer = None;
        self.cached_width = 0;
        self.cached_height = 0;
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            warn!(%error, "Photo filter workspace retirement poll failed");
        }
    }
    /// Create a new GPU filter pipeline
    ///
    /// This will attempt to use hardware GPU acceleration with low-priority queue
    /// to avoid starving UI rendering. Falls back to software rendering if no GPU.
    pub async fn new() -> Result<Self, String> {
        info!("Initializing GPU filter pipeline");

        // Get shared GPU device to avoid creating multiple wgpu instances
        let gpu = gpu::get_shared_gpu().await?;
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        info!(
            adapter_name = %gpu.info.adapter_name,
            adapter_backend = ?gpu.info.backend,
            low_priority = gpu.info.low_priority_enabled,
            "Using shared GPU device for filter pipeline"
        );

        // Create shader with shared filter functions
        let shader_source = format!(
            "{}\n{}",
            super::FILTER_FUNCTIONS,
            include_str!("filter_compute.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("filter_bind_group_layout"),
            entries: &[
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("filter_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("filter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("filter_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("filter_uniform_buffer"),
            size: std::mem::size_of::<FilterParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ===== Pre-blur compute pipeline =====
        let preblur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter_preblur_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("filter_preblur_compute.wgsl").into()),
        });

        let preblur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("filter_preblur_bind_group_layout"),
                entries: &[
                    // Input texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Output storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Uniform buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let preblur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("filter_preblur_pipeline_layout"),
                bind_group_layouts: &[&preblur_bind_group_layout],
                immediate_size: 0,
            });

        let preblur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("filter_preblur_pipeline"),
            layout: Some(&preblur_pipeline_layout),
            module: &preblur_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let preblur_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("filter_preblur_uniform_buffer"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            preblur_pipeline,
            preblur_bind_group_layout,
            preblur_uniform_buffer,
            cached_width: 0,
            cached_height: 0,
            input_texture: None,
            preblur_texture: None,
            output_buffer: None,
            staging_buffer: None,
        })
    }

    /// Ensure resources are allocated for the given dimensions
    fn ensure_resources(&mut self, width: u32, height: u32, upload: bool) {
        if self.cached_width == width && self.cached_height == height {
            return;
        }

        debug!(width, height, "Allocating filter pipeline resources");

        let buffer_size = (width * height * 4) as u64;

        // Create input texture
        self.input_texture = upload.then(|| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("filter_input_texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        });

        self.preblur_texture = None;

        // Create output storage buffer
        self.output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("filter_output_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.cached_width = width;
        self.cached_height = height;
    }

    fn ensure_preblur_texture(&mut self, width: u32, height: u32) {
        if self.preblur_texture.is_some() {
            return;
        }
        // Pre-blur intermediate texture (storage + texture binding for two-pass filters)
        self.preblur_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("filter_preblur_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        }));
    }

    /// Apply a filter to RGBA data
    ///
    /// Takes RGBA pixel data (width * height * 4 bytes) and returns filtered RGBA data.
    /// This runs on the GPU with software rendering fallback.
    pub async fn apply_filter_rgba(
        &mut self,
        rgba_data: &[u8],
        width: u32,
        height: u32,
        filter: FilterType,
    ) -> Result<Vec<u8>, String> {
        self.apply_filter_rgba_with_stride(rgba_data, width, height, width * 4, filter)
            .await
    }

    /// Apply a filter to RGBA rows with optional trailing padding.
    ///
    /// Filtered output is tightly packed. Standard copies tight input verbatim
    /// and strips padding from strided input without allocating GPU resources.
    /// Unlike photo texture filtering, uploads retain their workspace between
    /// frames and never participate in photo GPU pacing or retirement.
    pub async fn apply_filter_rgba_with_stride(
        &mut self,
        rgba_data: &[u8],
        width: u32,
        height: u32,
        stride: u32,
        filter: FilterType,
    ) -> Result<Vec<u8>, String> {
        if filter == FilterType::Standard {
            let row_bytes = (width * 4) as usize;
            if stride as usize == row_bytes {
                return Ok(rgba_data.to_vec());
            }
            let mut output = Vec::with_capacity(row_bytes * height as usize);
            for row in 0..height as usize {
                let start = row * stride as usize;
                output.extend_from_slice(&rgba_data[start..start + row_bytes]);
            }
            return Ok(output);
        }

        self.ensure_resources(width, height, true);

        let input_texture = self
            .input_texture
            .as_ref()
            .ok_or("Input texture not allocated")?;
        // Upload RGBA data to input texture
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: input_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.apply_filter_view_region(input_view, width, height, filter, (0, 0, width, height))
            .await
    }

    /// Chain the canonical RGBA8 converter texture without CPU readback/upload.
    /// Filter the FULL source before extracting the region: UV-dependent,
    /// neighbor-sampling and preblur filters must never see cropped dimensions.
    pub(crate) async fn apply_filter_texture_region(
        &mut self,
        texture: &wgpu::Texture,
        filter: FilterType,
        region: (u32, u32, u32, u32),
    ) -> Result<Vec<u8>, String> {
        self.release_frame_resources();
        let workspace = PhotoFilterWorkspace { pipeline: self };
        if texture.format() != wgpu::TextureFormat::Rgba8Unorm
            || !texture
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.depth_or_array_layers() != 1
            || texture.sample_count() != 1
            || filter == FilterType::Standard
        {
            return Err("Incompatible photo filter texture".into());
        }
        let (width, height) = (texture.width(), texture.height());
        workspace.pipeline.ensure_resources(width, height, false);
        workspace
            .pipeline
            .apply_filter_view_region(
                texture.create_view(&Default::default()),
                width,
                height,
                filter,
                region,
            )
            .await
    }

    async fn apply_filter_view_region(
        &mut self,
        input_view: wgpu::TextureView,
        width: u32,
        height: u32,
        filter: FilterType,
        (x, y, region_width, region_height): (u32, u32, u32, u32),
    ) -> Result<Vec<u8>, String> {
        if x > width || y > height || region_width > width - x || region_height > height - y {
            return Err("Filter readback region exceeds source dimensions".into());
        }
        if region_width == 0 || region_height == 0 {
            return Ok(Vec::new());
        }
        if filter.needs_preblur() {
            self.ensure_preblur_texture(width, height);
        }
        let buffer_size = u64::from(region_width) * u64::from(region_height) * 4;
        // Staging is sized to the final region, not the sensor dimensions.
        if self
            .staging_buffer
            .as_ref()
            .is_none_or(|b| b.buffer().size() != buffer_size)
        {
            self.staging_buffer = Some(crate::gpu::readback::ReadbackBuffer::from_buffer(
                &self.device,
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("filter_staging_buffer"),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                "filter_staging_buffer",
            ));
        }
        let readback = self.staging_buffer.as_ref().unwrap();
        let staging_buffer = readback.buffer();
        let output_buffer = self
            .output_buffer
            .as_ref()
            .ok_or("Output buffer not allocated")?;

        // Update uniform buffer
        let params = FilterParams {
            width,
            height,
            filter_mode: filter as u32,
            _padding: 0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        // Create and submit command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("filter_encoder"),
            });

        let workgroups_x = width.div_ceil(16);
        let workgroups_y = height.div_ceil(16);

        // Determine which texture the filter pass reads from.
        // For multi-pass filters, run a pre-blur pass first and read from that.
        let filter_input_view = if filter.needs_preblur() {
            let preblur_texture = self
                .preblur_texture
                .as_ref()
                .ok_or("Preblur texture not allocated")?;
            let preblur_view = preblur_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Update preblur uniform
            let blur_params = BlurParams {
                width,
                height,
                _padding: [0; 2],
            };
            self.queue.write_buffer(
                &self.preblur_uniform_buffer,
                0,
                bytemuck::bytes_of(&blur_params),
            );

            // Pre-blur pass: input_texture → preblur_texture
            let preblur_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("filter_preblur_bind_group"),
                layout: &self.preblur_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&preblur_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.preblur_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("filter_preblur_compute_pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.preblur_pipeline);
                compute_pass.set_bind_group(0, Some(&preblur_bind_group), &[]);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Filter pass reads from pre-blurred texture
            preblur_view
        } else {
            input_view
        };

        // Main filter pass
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("filter_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&filter_input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("filter_compute_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Row copies compact the final crop/zoom into one mapped allocation.
        if x == 0 && region_width == width {
            encoder.copy_buffer_to_buffer(
                output_buffer,
                u64::from(y) * u64::from(width) * 4,
                &staging_buffer,
                0,
                buffer_size,
            );
        } else {
            let row_bytes = u64::from(region_width) * 4;
            for row in 0..region_height {
                let source = (u64::from(y + row) * u64::from(width) + u64::from(x)) * 4;
                encoder.copy_buffer_to_buffer(
                    output_buffer,
                    source,
                    &staging_buffer,
                    u64::from(row) * row_bytes,
                    row_bytes,
                );
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map staging buffer and read back result
        let data = readback.read_async().await?;
        let output = data.to_vec();

        drop(data);

        Ok(output)
    }
}

/// Cached GPU filter pipeline instance
static GPU_FILTER_PIPELINE: std::sync::OnceLock<tokio::sync::Mutex<Option<GpuFilterPipeline>>> =
    std::sync::OnceLock::new();

/// Get or create the shared GPU filter pipeline instance
pub async fn get_gpu_filter_pipeline()
-> Result<tokio::sync::MutexGuard<'static, Option<GpuFilterPipeline>>, String> {
    let lock = GPU_FILTER_PIPELINE.get_or_init(|| tokio::sync::Mutex::new(None));
    let mut guard = lock.lock().await;

    if guard.is_none() {
        match GpuFilterPipeline::new().await {
            Ok(pipeline) => {
                *guard = Some(pipeline);
            }
            Err(e) => {
                warn!("Failed to initialize GPU filter pipeline: {}", e);
                return Err(e);
            }
        }
    }

    Ok(guard)
}

/// Apply a filter to RGBA data using the shared GPU pipeline
///
/// This is the main entry point for applying filters. It uses GPU acceleration
/// with software rendering fallback. Takes RGBA input and returns RGBA output.
pub async fn apply_filter_gpu_rgba(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    filter: FilterType,
) -> Result<Vec<u8>, String> {
    let mut guard = get_gpu_filter_pipeline().await?;
    let pipeline = guard
        .as_mut()
        .ok_or("GPU filter pipeline not initialized")?;

    pipeline
        .apply_filter_rgba(rgba_data, width, height, filter)
        .await
}

#[cfg(test)]
#[path = "gpu_filter_photo_tests.rs"]
mod photo_scheduling_tests;

#[cfg(test)]
mod texture_region_tests {
    use super::*;

    #[test]
    fn photo_filter_allocates_only_required_surfaces_and_cleans_cancelled_lease() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "photo_filter_allocates_only_required_surfaces_and_cleans_cancelled_lease",
            );
            return;
        };
        pollster::block_on(async {
            let mut filter = GpuFilterPipeline::new().await.unwrap();
            filter.ensure_resources(17, 19, false);
            assert!(filter.input_texture.is_none());
            assert!(
                filter.preblur_texture.is_none(),
                "single-pass photo allocated unused preblur"
            );
            filter.release_frame_resources();
            let mut future = Box::pin(async {
                let lease = PhotoFilterWorkspace {
                    pipeline: &mut filter,
                };
                lease.pipeline.ensure_resources(17, 19, false);
                std::future::pending::<()>().await;
                drop(lease);
            });
            let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
            assert!(std::future::Future::poll(future.as_mut(), &mut cx).is_pending());
            drop(future);
            assert!(filter.output_buffer.is_none());
            assert!(filter.preblur_texture.is_none());
            assert!(filter.staging_buffer.is_none());
            assert_eq!((filter.cached_width, filter.cached_height), (0, 0));
            // Recreate via the ordinary upload API after cancellation.
            assert_eq!(
                filter
                    .apply_filter_rgba(&[80, 100, 120, 255].repeat(4), 2, 2, FilterType::Mono)
                    .await
                    .unwrap()
                    .len(),
                16
            );
            let texture = filter.input_texture.as_ref().unwrap().clone();
            assert!(
                filter
                    .apply_filter_texture_region(&texture, FilterType::Mono, (u32::MAX, 0, 1, 1))
                    .await
                    .is_err()
            );
            assert!(filter.output_buffer.is_none());
            assert!(filter.input_texture.is_none());
            assert_eq!((filter.cached_width, filter.cached_height), (0, 0));
        });
    }
}

#[cfg(test)]
mod continuous_upload_tests {
    use super::*;
    use crate::test_gpu::{PhotoEvents, PhotoTrace};
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn padded_uploads_match_tight_pixels_and_reuse_unpaced_workspace_for_every_filter() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "padded_uploads_match_tight_pixels_and_reuse_unpaced_workspace_for_every_filter",
            );
            return;
        };
        let events = PhotoEvents::default();
        let subscriber = tracing_subscriber::registry().with(PhotoTrace(events.clone()));
        tracing::subscriber::with_default(subscriber, || {
            pollster::block_on(async {
                let mut pipeline = GpuFilterPipeline::new().await.unwrap();
                let mut reference = GpuFilterPipeline::new().await.unwrap();
                for (width, height) in [(17, 19), (3, 2)] {
                    let stride = width * 4 + 12;
                    let pixels: Vec<u8> = (0..width * height)
                        .flat_map(|i| {
                            [
                                (i * 37 % 256) as u8,
                                (i * 53 % 256) as u8,
                                (i * 97 % 256) as u8,
                                (i * 11 % 256) as u8,
                            ]
                        })
                        .collect();
                    let mut padded = vec![0xa5; (stride * height) as usize];
                    for (source, destination) in pixels
                        .chunks_exact((width * 4) as usize)
                        .zip(padded.chunks_exact_mut(stride as usize))
                    {
                        destination[..source.len()].copy_from_slice(source);
                    }
                    // Start with preblur so its allocation must survive every later mode.
                    pipeline
                        .apply_filter_rgba_with_stride(
                            &padded,
                            width,
                            height,
                            stride,
                            FilterType::Pencil,
                        )
                        .await
                        .unwrap();
                    let input = pipeline.input_texture.clone();
                    let preblur = pipeline.preblur_texture.clone();
                    let output = pipeline.output_buffer.clone();
                    let staging = pipeline.staging_buffer.as_ref().unwrap().buffer();
                    for code in (0..=14).rev() {
                        let mode = FilterType::from_gpu_filter_code(code);
                        for offset in 0..3u8 {
                            let changed: Vec<u8> = pixels
                                .iter()
                                .map(|byte| byte.wrapping_add(offset * 31))
                                .collect();
                            for (source, destination) in changed
                                .chunks_exact((width * 4) as usize)
                                .zip(padded.chunks_exact_mut(stride as usize))
                            {
                                destination[..source.len()].copy_from_slice(source);
                            }
                            let expected = reference
                                .apply_filter_rgba(&changed, width, height, mode)
                                .await
                                .unwrap();
                            let actual = pipeline
                                .apply_filter_rgba_with_stride(&padded, width, height, stride, mode)
                                .await
                                .unwrap();
                            assert_eq!(actual, expected, "{mode:?}, {width}x{height}");
                            assert_eq!(pipeline.input_texture, input);
                            assert_eq!(pipeline.preblur_texture, preblur);
                            assert_eq!(pipeline.output_buffer, output);
                            assert_eq!(pipeline.staging_buffer.as_ref().unwrap().buffer(), staging);
                            assert_eq!(
                                (pipeline.cached_width, pipeline.cached_height),
                                (width, height)
                            );
                        }
                    }
                }
                assert!(
                    events.lock().unwrap().is_empty(),
                    "continuous filters were photo-paced"
                );
            });
        });
    }

    #[test]
    fn standard_upload_does_not_allocate_or_dispatch() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("standard_upload_does_not_allocate_or_dispatch");
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuFilterPipeline::new().await.unwrap();
            let pixels = vec![10, 20, 30, 40, 50, 60, 70, 80];
            // Tight passthrough preserves all supplied bytes, as before.
            assert_eq!(
                pipeline
                    .apply_filter_rgba(&pixels, 1, 1, FilterType::Standard)
                    .await
                    .unwrap(),
                pixels
            );
            assert_eq!(
                pipeline
                    .apply_filter_rgba_with_stride(
                        &[10, 20, 30, 40, 0xa5, 0xa5, 0xa5, 0xa5, 50, 60, 70, 80],
                        1,
                        2,
                        8,
                        FilterType::Standard,
                    )
                    .await
                    .unwrap(),
                pixels
            );
            assert!(pipeline.input_texture.is_none());
            assert!(pipeline.preblur_texture.is_none());
            assert!(pipeline.output_buffer.is_none());
            assert!(pipeline.staging_buffer.is_none());
            assert_eq!((pipeline.cached_width, pipeline.cached_height), (0, 0));
        });
    }
}

#[cfg(test)]
mod readback_owner_tests {
    use super::*;
    #[test]
    fn filter_failed_cached_map_then_public_filter_recovers_same_pixels() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "filter_failed_cached_map_then_public_filter_recovers_same_pixels",
            );
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuFilterPipeline::new().await.unwrap();
            let pixels: Vec<u8> = (0..16).flat_map(|i| [i * 7, i * 11, i * 13, 255]).collect();
            let padded: Vec<u8> = pixels
                .as_chunks::<16>()
                .0
                .iter()
                .flat_map(|row| row.iter().copied().chain([0xa5; 12]))
                .collect();
            for mode in [FilterType::Mono, FilterType::Pencil] {
                let expected = pipeline
                    .apply_filter_rgba(&pixels, 4, 4, mode)
                    .await
                    .unwrap();
                let input = pipeline.input_texture.clone();
                let output = pipeline.output_buffer.clone();
                let preblur = pipeline.preblur_texture.clone();
                let cache = pipeline.staging_buffer.as_ref().unwrap();
                let failed_buffer = cache.buffer();
                failed_buffer.destroy();
                let error = cache.read_async().await.err().unwrap();
                assert!(error.contains("destroyed"), "{error}");
                // A failed map must never manufacture an empty successful read.
                assert!(
                    cache
                        .read_async()
                        .await
                        .err()
                        .unwrap()
                        .contains("unavailable")
                );
                for _ in 0..3 {
                    assert_eq!(
                        pipeline
                            .apply_filter_rgba_with_stride(&padded, 4, 4, 28, mode)
                            .await
                            .unwrap(),
                        expected
                    );
                    assert_eq!(pipeline.input_texture, input);
                    assert_eq!(pipeline.output_buffer, output);
                    assert_eq!(pipeline.preblur_texture, preblur);
                    assert_ne!(
                        pipeline.staging_buffer.as_ref().unwrap().buffer(),
                        failed_buffer
                    );
                }
            }
        });
    }
}
