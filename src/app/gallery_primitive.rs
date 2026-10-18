// SPDX-License-Identifier: GPL-3.0-only

//! Custom primitive for rendering gallery thumbnails with rounded corners

use super::video_primitive::rendering::{
    encode_screen_triangle, fullscreen_pipeline, sampled_texture_layout,
};
use crate::storage::GalleryThumbnailData;

use cosmic::iced::Rectangle;
use iced_wgpu::graphics::Viewport;
use iced_wgpu::primitive::{Pipeline as PipelineTrait, Primitive as PrimitiveTrait};
use iced_wgpu::wgpu;

/// Custom primitive for gallery thumbnail with rounded corners
#[derive(Debug, Clone)]
pub struct GalleryPrimitive {
    pub thumbnail: GalleryThumbnailData,
    pub corner_radius: f32,
    /// Accent border color (RGBA) shown when hovered
    pub accent_color: [f32; 4],
}

impl GalleryPrimitive {
    pub fn new(thumbnail: GalleryThumbnailData, corner_radius: f32) -> Self {
        Self {
            thumbnail,
            corner_radius,
            accent_color: [0.0; 4],
        }
    }
}

/// Pipeline for rendering gallery thumbnails
pub struct GalleryPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    texture_format: wgpu::TextureFormat,
    // Cache for the single uploaded texture (at most one gallery thumbnail at a time)
    texture_cache: Option<(uuid::Uuid, GalleryTexture)>,
}

struct GalleryTexture {
    bind_group: wgpu::BindGroup,
    viewport_buffer: wgpu::Buffer,
    /// Physical widget bounds (x, y, width, height) clamped to render target
    /// Stored during prepare() and used in render() for valid viewport rect
    physical_bounds: Option<(f32, f32, f32, f32)>,
}

/// The unused fourth scalar remains padding: WGSL's accent color starts at 16.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GalleryUniform {
    size: [f32; 2],
    corner_radius: f32,
    _pad: f32,
    accent_color: [f32; 4],
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
}

impl PipelineTrait for GalleryPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        GalleryPipeline::new(device, format)
    }

    fn trim(&mut self) {
        // No-op: we manage texture cache lifecycle ourselves.
        // Clearing here would destroy live textures and cause flickering.
    }
}

impl PrimitiveTrait for GalleryPrimitive {
    type Pipeline = GalleryPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        // Upload image texture if needed and update viewport buffer
        pipeline.upload_image(_device, queue, &self.thumbnail);

        // Calculate physical bounds from logical bounds using scale factor
        // Then clamp to render target to ensure valid viewport rect
        let scale = viewport.scale_factor() as f32;
        let render_target = viewport.physical_size();

        let raw = (
            bounds.x * scale,
            bounds.y * scale,
            bounds.width * scale,
            bounds.height * scale,
        );

        let clamped_x = raw.0.max(0.0);
        let clamped_y = raw.1.max(0.0);
        let clamped_w = ((raw.0 + raw.2).min(render_target.width as f32) - clamped_x).max(0.0);
        let clamped_h = ((raw.1 + raw.3).min(render_target.height as f32) - clamped_y).max(0.0);

        // Calculate UV offset/scale to compensate for clamping
        let (uv_offset, uv_scale) = if raw.2 > 0.0 && raw.3 > 0.0 {
            (
                ((clamped_x - raw.0) / raw.2, (clamped_y - raw.1) / raw.3),
                (clamped_w / raw.2, clamped_h / raw.3),
            )
        } else {
            ((0.0, 0.0), (1.0, 1.0))
        };

        // Store physical bounds for render()
        pipeline.store_physical_bounds(
            self.thumbnail.id(),
            (clamped_x, clamped_y, clamped_w, clamped_h),
        );

        // Update viewport size, corner radius, and UV adjustment
        pipeline.update_viewport(
            queue,
            self.thumbnail.id(),
            bounds.width,
            bounds.height,
            self.corner_radius,
            self.accent_color,
            uv_offset,
            uv_scale,
        );
    }

    fn render(
        &self,
        _pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        _pipeline.render(encoder, target, clip_bounds, self.thumbnail.id());
    }
}

fn gallery_shader() -> String {
    format!(
        "{}\n{}\n{}",
        super::video_primitive::rendering::FULLSCREEN_VERTEX,
        crate::shaders::GEOMETRY_FUNCTIONS,
        include_str!("gallery_rounded_shader.wgsl")
    )
}

impl GalleryPipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader_source = gallery_shader();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gallery rounded shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = sampled_texture_layout(device, "gallery bind group layout");

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gallery pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = fullscreen_pipeline(
            device,
            "gallery pipeline",
            &pipeline_layout,
            &shader,
            "fs_main",
            format,
            wgpu::BlendState::ALPHA_BLENDING,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gallery sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            // Source channel order is independent of the attachment; retain
            // its transfer function while sampling the RGBA bytes directly.
            texture_format: match format {
                wgpu::TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
                _ => format,
            },
            texture_cache: None,
        }
    }

    fn upload_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        thumbnail: &GalleryThumbnailData,
    ) {
        let image_id = thumbnail.id();

        // Check if already uploaded
        if self
            .texture_cache
            .as_ref()
            .is_some_and(|(id, _)| *id == image_id)
        {
            return;
        }

        let rgba_data = thumbnail.rgba();
        let (width, height) = thumbnail.dimensions();
        if rgba_data.is_empty() {
            tracing::warn!("Empty RGBA data for gallery thumbnail");
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gallery texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.texture_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload the image data
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create viewport params buffer (will be updated in update_viewport before render)
        // The unused shader hover scalar is padding; accent/UV offsets stay fixed.
        let viewport_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gallery viewport buffer"),
            size: std::mem::size_of::<GalleryUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gallery bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: viewport_buffer.as_entire_binding(),
                },
            ],
        });

        self.texture_cache = Some((
            image_id,
            GalleryTexture {
                bind_group,
                viewport_buffer,
                physical_bounds: None,
            },
        ));
    }

    fn store_physical_bounds(
        &mut self,
        image_id: uuid::Uuid,
        physical_bounds: (f32, f32, f32, f32),
    ) {
        if let Some((id, gallery_texture)) = &mut self.texture_cache
            && *id == image_id
        {
            gallery_texture.physical_bounds = Some(physical_bounds);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_viewport(
        &mut self,
        queue: &wgpu::Queue,
        image_id: uuid::Uuid,
        width: f32,
        height: f32,
        corner_radius: f32,
        accent_color: [f32; 4],
        uv_offset: (f32, f32),
        uv_scale: (f32, f32),
    ) {
        if let Some((id, gallery_texture)) = &self.texture_cache {
            if *id != image_id {
                return;
            }
            let viewport_data = GalleryUniform {
                size: [width, height],
                corner_radius,
                _pad: 0.0,
                accent_color,
                uv_offset: [uv_offset.0, uv_offset.1],
                uv_scale: [uv_scale.0, uv_scale.1],
            };
            queue.write_buffer(
                &gallery_texture.viewport_buffer,
                0,
                bytemuck::bytes_of(&viewport_data),
            );
        }
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
        image_id: uuid::Uuid,
    ) {
        if let Some((id, gallery_texture)) = &self.texture_cache {
            if *id != image_id || clip_bounds.width == 0 || clip_bounds.height == 0 {
                return;
            }

            // Use full widget bounds for viewport (prevents distortion when partially clipped)
            // Fall back to clip_bounds if physical_bounds not available
            let widget_bounds = gallery_texture.physical_bounds.unwrap_or((
                clip_bounds.x as f32,
                clip_bounds.y as f32,
                clip_bounds.width as f32,
                clip_bounds.height as f32,
            ));

            encode_screen_triangle(
                encoder,
                target,
                clip_bounds,
                widget_bounds,
                &self.pipeline,
                &gallery_texture.bind_group,
                "gallery render pass",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_gpu::{headless_device, skip_no_gpu};

    const SIDE: u32 = 64; // Exactly one aligned 256-byte readback row.
    const BOUNDS: Rectangle = Rectangle {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
    };

    fn thumbnail(color: [u8; 4]) -> GalleryThumbnailData {
        GalleryThumbnailData::new("same.png".into(), color.repeat(8 * 4), 8, 4)
    }

    fn render_cached(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &GalleryPipeline,
        primitive: &GalleryPrimitive,
        format: wgpu::TextureFormat,
    ) -> Vec<u8> {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gallery test target"),
            size: wgpu::Extent3d {
                width: SIDE,
                height: SIDE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gallery test readback"),
            size: u64::from(SIDE * SIDE * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gallery test clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        primitive.render(
            pipeline,
            &mut encoder,
            &view,
            &Rectangle {
                x: 0,
                y: 0,
                width: SIDE,
                height: SIDE,
            },
        );
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIDE * 4),
                    rows_per_image: Some(SIDE),
                },
            },
            target.size(),
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mut pixels = readback.slice(..).get_mapped_range().to_vec();
        readback.unmap();
        if matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        pixels
    }

    fn render_thumbnail(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &mut GalleryPipeline,
        primitive: &GalleryPrimitive,
        format: wgpu::TextureFormat,
        bounds: Rectangle,
    ) -> Vec<u8> {
        let viewport = Viewport::with_physical_size(cosmic::iced::Size::new(SIDE, SIDE), 1.0);
        primitive.prepare(pipeline, device, queue, &bounds, &viewport);
        render_cached(device, queue, pipeline, primitive, format)
    }

    fn pixel(pixels: &[u8], x: u32, y: u32) -> &[u8] {
        let offset = ((y * SIDE + x) * 4) as usize;
        &pixels[offset..offset + 4]
    }

    #[test]
    fn gallery_uniform_keeps_the_shader_abi_without_hover_state() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<GalleryUniform>(), 48);
        assert_eq!(offset_of!(GalleryUniform, size), 0);
        assert_eq!(offset_of!(GalleryUniform, corner_radius), 8);
        assert_eq!(offset_of!(GalleryUniform, _pad), 12);
        assert_eq!(offset_of!(GalleryUniform, accent_color), 16);
        assert_eq!(offset_of!(GalleryUniform, uv_offset), 32);
        assert_eq!(offset_of!(GalleryUniform, uv_scale), 40);
        let module = crate::test_gpu::validate_shader("gallery", &gallery_shader());
        let (_, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("ViewportParams"))
            .unwrap();
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("gallery uniform is not a struct")
        };
        assert_eq!(*span, 48);
        assert_eq!(
            members
                .iter()
                .map(|member| member.offset)
                .collect::<Vec<_>>(),
            vec![0, 8, 12, 16, 32, 40]
        );
    }

    #[test]
    fn gallery_cache_reuses_clones_and_replaces_same_size_payloads() {
        let Some((_gpu_test, device, queue)) = headless_device() else {
            skip_no_gpu("gallery_cache_reuses_clones_and_replaces_same_size_payloads");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = GalleryPipeline::new(&device, format);
        let first = GalleryPrimitive::new(thumbnail([30, 160, 220, 255]), 10.0);
        let original = render_thumbnail(&device, &queue, &mut pipeline, &first, format, BOUNDS);
        let (id, cached) = pipeline.texture_cache.as_ref().unwrap();
        assert_eq!(*id, first.thumbnail.id());
        let binding = cached.bind_group.clone();
        let uniform = cached.viewport_buffer.clone();
        pipeline.trim();
        for _ in 0..3 {
            let clone = first.clone();
            assert_eq!(
                clone.thumbnail.rgba().as_ptr(),
                first.thumbnail.rgba().as_ptr()
            );
            assert_eq!(
                render_thumbnail(&device, &queue, &mut pipeline, &clone, format, BOUNDS),
                original
            );
            let cached = &pipeline.texture_cache.as_ref().unwrap().1;
            assert_eq!(cached.bind_group, binding);
            assert_eq!(cached.viewport_buffer, uniform);
        }
        let replacement = GalleryPrimitive::new(thumbnail([200, 80, 20, 255]), 10.0);
        let replaced =
            render_thumbnail(&device, &queue, &mut pipeline, &replacement, format, BOUNDS);
        let (id, cached) = pipeline.texture_cache.as_ref().unwrap();
        assert_eq!(*id, replacement.thumbnail.id());
        assert_ne!(*id, first.thumbnail.id());
        assert_ne!(cached.bind_group, binding);
        assert_ne!(cached.viewport_buffer, uniform);
        assert_eq!(pixel(&replaced, 32, 32), &[200, 80, 20, 255]);
        assert_eq!(pixel(&original, 32, 32), &[30, 160, 220, 255]);
        let bounds = cached.physical_bounds;
        pipeline.store_physical_bounds(first.thumbnail.id(), (9.0, 9.0, 9.0, 9.0));
        assert_eq!(
            pipeline.texture_cache.as_ref().unwrap().1.physical_bounds,
            bounds
        );
        pipeline.update_viewport(
            &queue,
            first.thumbnail.id(),
            8.0,
            8.0,
            0.0,
            [1.0, 0.0, 0.0, 1.0],
            (0.0, 0.0),
            (1.0, 1.0),
        );
        assert_eq!(
            render_cached(&device, &queue, &pipeline, &replacement, format),
            replaced,
            "stale identity cannot overwrite the replacement's uniform"
        );
        assert!(
            render_cached(&device, &queue, &pipeline, &first, format)
                .iter()
                .all(|&v| v == 0),
            "stale identity cannot draw the replacement"
        );
        // Re-selecting an older retained clone is a real replacement, not a hit
        // based on equal dimensions or a dangling pixel pointer.
        assert_eq!(
            render_thumbnail(
                &device,
                &queue,
                &mut pipeline,
                &first.clone(),
                format,
                BOUNDS
            ),
            original
        );
    }

    #[test]
    fn gallery_sampling_preserves_color_hover_and_rounding_on_all_surface_orders() {
        let Some((_gpu_test, device, queue)) = headless_device() else {
            skip_no_gpu(
                "gallery_sampling_preserves_color_hover_and_rounding_on_all_surface_orders",
            );
            return;
        };
        for format in [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        ] {
            let mut pipeline = GalleryPipeline::new(&device, format);
            let primitive = GalleryPrimitive::new(thumbnail([30, 160, 220, 255]), 10.0);
            let plain =
                render_thumbnail(&device, &queue, &mut pipeline, &primitive, format, BOUNDS);
            for (&actual, expected) in pixel(&plain, 32, 32).iter().zip([30u8, 160, 220, 255]) {
                assert!(
                    actual.abs_diff(expected) <= 2,
                    "{format:?}: {actual} != {expected}"
                );
            }
            assert_eq!(pixel(&plain, 0, 0), &[0, 0, 0, 0]);
            let binding = pipeline
                .texture_cache
                .as_ref()
                .unwrap()
                .1
                .bind_group
                .clone();
            let mut hovered = primitive.clone();
            hovered.accent_color = [1.0, 0.0, 0.0, 1.0];
            let hover = render_thumbnail(&device, &queue, &mut pipeline, &hovered, format, BOUNDS);
            assert_eq!(
                pipeline.texture_cache.as_ref().unwrap().1.bind_group,
                binding,
                "hover must not reupload pixels"
            );
            assert_eq!(pixel(&hover, 32, 32), pixel(&plain, 32, 32));
            assert_eq!(pixel(&hover, 1, 32), &[255, 0, 0, 255]);
            assert_ne!(pixel(&plain, 1, 32), pixel(&hover, 1, 32));
            assert_eq!(pixel(&hover, 0, 0), &[0, 0, 0, 0]);
            assert_eq!(
                render_thumbnail(&device, &queue, &mut pipeline, &primitive, format, BOUNDS),
                plain,
                "leaving hover restores the unmodified image"
            );
        }
    }

    #[test]
    fn gallery_cover_fit_and_clipped_uvs_preserve_aspect_and_corner_positions() {
        let Some((_gpu_test, device, queue)) = headless_device() else {
            skip_no_gpu("gallery_cover_fit_and_clipped_uvs_preserve_aspect_and_corner_positions");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut pipeline = GalleryPipeline::new(&device, format);
        for (width, height) in [(128, 64), (64, 128)] {
            let image = image::RgbaImage::from_fn(width, height, |x, y| {
                let long_axis = if width > height { x } else { y };
                image::Rgba(if long_axis < 32 {
                    [255, 0, 0, 255]
                } else if long_axis < 96 {
                    [0, 255, 0, 255]
                } else {
                    [0, 0, 255, 255]
                })
            });
            let primitive = GalleryPrimitive::new(
                GalleryThumbnailData::new("aspect.png".into(), image.into_raw(), width, height),
                10.0,
            );
            let full = render_thumbnail(&device, &queue, &mut pipeline, &primitive, format, BOUNDS);
            for (x, y) in [(8, 32), (55, 32), (32, 8), (32, 55)] {
                assert_eq!(
                    pixel(&full, x, y),
                    &[0, 255, 0, 255],
                    "cover must crop the outer stripes, not stretch {width}x{height}"
                );
            }
            let clipped = render_thumbnail(
                &device,
                &queue,
                &mut pipeline,
                &primitive,
                format,
                Rectangle { x: -16.0, ..BOUNDS },
            );
            for y in 0..SIDE {
                for x in 0..48 {
                    for (&actual, &expected) in
                        pixel(&clipped, x, y).iter().zip(pixel(&full, x + 16, y))
                    {
                        assert!(
                            actual.abs_diff(expected) <= 2,
                            "clipping moved the corner/UV at {x},{y}"
                        );
                    }
                }
                assert_eq!(pixel(&clipped, 56, y), &[0, 0, 0, 0]);
            }
        }
    }

    /// `GalleryPipeline::new` must actually compile its shader.
    ///
    /// # Why this exists
    ///
    /// The gallery shader's `rounded_box_sdf` was deleted in favour of the shared
    /// `shaders/geometry.wgsl` prelude, which `new` prepends by hand:
    ///
    /// ```ignore
    /// format!("{}\n{}", crate::shaders::GEOMETRY_FUNCTIONS, include_str!("gallery_rounded_shader.wgsl"))
    /// ```
    ///
    /// Nothing else in the crate constructs a `GalleryPipeline`. WGSL is compiled
    /// at `create_shader_module`, i.e. at runtime, so a prelude that stopped being
    /// prepended — or a `rounded_box_sdf` that changed signature — is not a build
    /// error here. It is a panic on the device, the first time a thumbnail is
    /// drawn, in a shipped binary.
    ///
    /// `VideoPipeline::new` compiles all six of its modules in one function, so
    /// the existing GPU tests already cover those. This closes the one path where
    /// a prelude-wiring mistake reached the device unchallenged. It is a smoke
    /// test on purpose: it asserts construction, because construction is exactly
    /// the step that validates the shader.
    #[test]
    fn gallery_pipeline_compiles_its_shader() {
        let Some((_gpu_test, device, _queue)) = headless_device() else {
            skip_no_gpu("gallery_pipeline_compiles_its_shader");
            return;
        };
        // wgpu reports a shader compile failure on the device's error scope
        // rather than by returning, so capture it rather than trusting a
        // non-panicking return.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _pipeline = GalleryPipeline::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let err = pollster::block_on(scope.pop());
        assert!(
            err.is_none(),
            "GalleryPipeline::new failed to build: {err:?} — the gallery shader is \
             only compiled at runtime, so this is a device-side panic on the first \
             thumbnail, not a build error"
        );
    }

    /// The prelude really is what supplies the gallery's SDF.
    ///
    /// The smoke test above passes either way if the shader ever grows its own
    /// copy of `rounded_box_sdf` back — and a second copy silently forking from
    /// `geometry.wgsl` is the exact thing deleting it was meant to prevent. So
    /// pin the split: the shader calls the function and does not define it, and
    /// the prelude defines it.
    #[test]
    fn the_gallery_shader_takes_its_sdf_from_the_prelude() {
        crate::test_gpu::validate_shader("gallery", &gallery_shader());
        let shader = include_str!("gallery_rounded_shader.wgsl");
        assert!(
            shader.contains("rounded_box_sdf("),
            "the gallery shader should still USE the shared SDF"
        );
        assert!(
            !shader.contains("fn rounded_box_sdf"),
            "the gallery shader defines its own `rounded_box_sdf` again — that is \
             the duplicate `geometry.wgsl` exists to replace, and a second copy \
             will drift from the video shader's silhouette"
        );
        assert!(
            crate::shaders::GEOMETRY_FUNCTIONS.contains("fn rounded_box_sdf"),
            "the shared prelude must define `rounded_box_sdf`, or every shader \
             that stopped defining it fails to compile on device"
        );
    }
}
