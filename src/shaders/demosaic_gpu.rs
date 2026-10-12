// SPDX-License-Identifier: GPL-3.0-only
//! Shared texture-based photo/HDR demosaic. No FP64 or frame-sized storage
//! bindings: scratch lives in four halo textures; only 16 status bytes map.
use super::DemosaicMethod;
use crate::gpu::wgpu;

pub(crate) struct GpuDemosaic {
    pub raw: wgpu::Texture,
    scratch: [wgpu::Texture; 4],
    pipelines: Vec<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    status: wgpu::Buffer,
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
}

pub(crate) fn texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
impl GpuDemosaic {
    pub fn matches(&self, width: u32, height: u32) -> bool {
        self.width == width && self.height == height
    }
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Result<Self, String> {
        let limits = device.limits();
        let w = width
            .checked_add(32)
            .ok_or("Demosaic dimensions overflow")?;
        let h = height
            .checked_add(32)
            .ok_or("Demosaic dimensions overflow")?;
        if width < 2
            || height < 2
            || w > limits.max_texture_dimension_2d
            || h > limits.max_texture_dimension_2d
            || w.div_ceil(16) > limits.max_compute_workgroups_per_dimension
            || h.div_ceil(16) > limits.max_compute_workgroups_per_dimension
            || limits.max_sampled_textures_per_shader_stage < 3
            || limits.max_storage_textures_per_shader_stage < 1
            || limits.max_storage_buffer_binding_size < 16
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_storage_size < 2048
            || limits.max_compute_workgroup_size_x < 16
            || limits.max_compute_workgroup_size_y < 16
        {
            return Err("GPU limits cannot admit demosaic halo textures or dispatch".into());
        }
        let entries: Vec<_> = (0..6)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: match binding {
                    0..=2 => wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    3 => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    _ => wgpu::BindingType::Buffer {
                        ty: if binding == 4 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("demosaic textures"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("demosaic layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RCD / linear LMMSE"),
            source: wgpu::ShaderSource::Wgsl(include_str!("demosaic_gpu.wgsl").into()),
        });
        let pipelines = ["bounds", "first", "second", "third", "fourth", "final_pass"]
            .iter()
            .map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            raw: texture(device, width, height, "normalized CFA"),
            scratch: std::array::from_fn(|_| texture(device, w, h, "demosaic halo")),
            pipelines,
            layout,
            uniform: buffer(
                "demosaic params",
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            status: buffer(
                "demosaic status",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            staging: buffer(
                "demosaic compact readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            width,
            height,
        })
    }
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Texture,
        pattern: u32,
        method: DemosaicMethod,
    ) {
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[
                self.width,
                self.height,
                pattern,
                match method {
                    DemosaicMethod::Rcd => 0u32,
                    DemosaicMethod::Lmmse => 1,
                    DemosaicMethod::Bilinear => 2,
                },
            ]),
        );
        queue.write_buffer(
            &self.status,
            0,
            bytemuck::cast_slice(&[0u32, 1f32.to_bits(), 0, 0]),
        );
        let raw = self.raw.create_view(&Default::default());
        let views: Vec<_> = self
            .scratch
            .iter()
            .map(|t| t.create_view(&Default::default()))
            .collect();
        let output = output.create_view(&Default::default());
        for (stage, ai, bi, oi) in [
            (0, 0, 1, 2),
            (1, 1, 2, 0),
            (2, 0, 2, 1),
            (3, 0, 1, 2),
            (4, 2, if method == DemosaicMethod::Rcd { 1 } else { 2 }, 3),
            (5, 2, 3, 4),
        ] {
            if method == DemosaicMethod::Bilinear && stage > 0 && stage < 5 {
                continue;
            }
            let out = if oi == 4 { &output } else { &views[oi] };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("demosaic stage"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&raw),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[ai]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&views[bi]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(out),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.status.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU demosaic stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines[stage]);
            pass.set_bind_group(0, &bg, &[]);
            let halo = if stage == 0 || stage == 5 { 0 } else { 32 };
            pass.dispatch_workgroups(
                (self.width + halo).div_ceil(16),
                (self.height + halo).div_ceil(16),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&self.status, 0, &self.staging, 0, 16);
    }
    pub fn encode_merged(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        planes: &wgpu::Buffer,
        pattern: u32,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("merged CFA adapter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bayer_planes.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("merged CFA adapter"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[self.width, self.height, pattern, 0u32]),
        );
        let view = self.raw.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("merged CFA adapter"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: planes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("merged CFA adapter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(16), self.height.div_ceil(16), 1);
    }
    /// Submit first; validate only a compact GPU reduction before colour dispatch.
    pub fn validate(&self, device: &wgpu::Device) -> Result<f64, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| e.to_string())?;
        rx.recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let data = self.staging.slice(..).get_mapped_range();
        let s: &[u32] = bytemuck::cast_slice(&data);
        let result = if s[3] != 0 {
            Err("Unsafe GPU demosaic arithmetic".into())
        } else {
            Ok(f64::from(f32::from_bits(s[2])))
        };
        drop(data);
        self.staging.unmap();
        result
    }
}

#[cfg(test)]
#[path = "demosaic_gpu_tests.rs"]
mod characterization_tests;

#[cfg(test)]
mod mobile_output_contract_tests {
    #[test]
    fn hdr_finishing_does_not_bind_a_full_resolution_storage_buffer() {
        let shader = include_str!("burst_mode/bayer_finish.wgsl");
        assert!(
            shader.contains("var output: texture_storage_2d<rgba32float"),
            "HDR output exceeds mobile storage binding limits at photo resolution"
        );
    }
}

#[cfg(test)]
mod arithmetic_contract_tests {
    #[test]
    fn intermediate_arithmetic_is_checked_before_lmmse_clamps() {
        let source = include_str!("demosaic_gpu.wgsl");
        assert!(
            source.contains("if !finite(est) || !finite(v0)"),
            "a clamp must not conceal intermediate overflow/NaN"
        );
        assert_eq!(
            source
                .matches("for(var component=0u;component<4u;component++)")
                .count(),
            4,
            "all four intermediate stage textures need finite validation"
        );
    }
}
