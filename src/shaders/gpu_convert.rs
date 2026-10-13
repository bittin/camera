// SPDX-License-Identifier: GPL-3.0-only
//! GPU-accelerated format conversion pipelines
//!
//! Uses a unified compute shader (`yuv_convert.wgsl`) for all YUV→RGBA conversions,
//! with format selection via a uniform parameter. Debayer uses a separate shader.
//!
//! **Supported formats:**
//! - NV12/NV21: Semi-planar 4:2:0
//! - I420: Planar 4:2:0 (also handles I422/I444 from MJPEG)
//! - YUYV/UYVY/YVYU/VYUY: Packed 4:2:2
//! - Gray8/RGB24/ABGR/BGRA: Single-plane conversions
//! - Bayer: Raw sensor data (separate debayer shader)

use crate::backends::camera::types::{CameraFrame, PixelFormat, RawLayout, checked_plane};
use crate::gpu::{self, wgpu};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Workgroup size for the CSI-2 unpack shader (one thread per pixel pair).
const UNPACK_WORKGROUP_SIZE: u32 = 256;

/// Workgroup size for the debayer, AWB, and YUV conversion shaders (16×16 tile).
const TILE_WORKGROUP_SIZE: u32 = 16;

/// Required byte alignment for `copy_buffer_to_texture` / `copy_texture_to_buffer` row stride.
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Round `value` up to the next multiple of [`COPY_BYTES_PER_ROW_ALIGNMENT`].
fn align_to_copy_row(value: u32) -> u32 {
    (value + COPY_BYTES_PER_ROW_ALIGNMENT - 1) & !(COPY_BYTES_PER_ROW_ALIGNMENT - 1)
}

/// YUV conversion parameters (32 bytes, matches yuv_convert.wgsl)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvertParams {
    width: u32,
    height: u32,
    format: u32,
    y_stride: u32,
    uv_stride: u32,
    v_stride: u32,
    color: u32,
    _pad1: u32,
}

/// Debayer conversion parameters (96 bytes, std140-compatible)
/// White balance gains are now in a separate storage buffer (AwbGains).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DebayerParams {
    width: u32,
    height: u32,
    pattern: u32,  // 0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG
    _padding: u32, // Keeps the shared AWB uniform prefix aligned.
    black_level: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    ccm_row0: [f32; 4],     // xyz used, w=pad
    ccm_row1: [f32; 4],     // xyz used, w=pad
    ccm_row2: [f32; 4],     // xyz used, w=pad
    white_levels: [f32; 4], // [R, Gr, Gb, B], normalized sensor codes
}

/// White balance gains stored in a GPU storage buffer.
/// Written by CPU (ISP gains) or by the GPU AWB finalize shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AwbGains {
    gain_r: f32,
    gain_b: f32,
}

/// AWB accumulation sums (used to clear the buffer before each AWB pass)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AwbSums {
    sum_r: u32,
    sum_g: u32,
    sum_b: u32,
    _pad: u32,
}

/// Fail-closed outcomes for the signed float Bayer conversion path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BayerError {
    InvalidSource(String),
    InvalidMetadata,
    UnsafeArithmetic,
}
impl std::fmt::Display for BayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSource(message) => f.write_str(message),
            Self::InvalidMetadata => f.write_str("Invalid Bayer colour metadata"),
            Self::UnsafeArithmetic => f.write_str("Unsafe Bayer colour arithmetic"),
        }
    }
}
impl std::error::Error for BayerError {}
impl From<String> for BayerError {
    fn from(message: String) -> Self {
        if message.starts_with("Unsafe ") {
            Self::UnsafeArithmetic
        } else {
            Self::InvalidSource(message)
        }
    }
}
impl From<&str> for BayerError {
    fn from(message: &str) -> Self {
        Self::from(message.to_owned())
    }
}

/// Input frame data for conversion
pub struct GpuFrameInput<'a> {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    /// Y plane data (or packed data for YUYV variants)
    pub y_data: &'a [u8],
    pub y_stride: u32,
    /// UV plane data (NV12: interleaved UV, I420: U plane)
    pub uv_data: Option<&'a [u8]>,
    pub uv_stride: u32,
    /// Active chroma extent, never row padding.
    pub uv_dimensions: Option<(u32, u32)>,
    pub yuv_color: crate::backends::camera::types::YuvColor,
    /// V plane data (I420 only)
    pub v_data: Option<&'a [u8]>,
    pub v_stride: u32,
    /// ISP white balance gains [R, B] (Bayer only)
    pub colour_gains: Option<[f32; 2]>,
    /// 3x3 colour correction matrix (row-major, Bayer only)
    pub colour_correction_matrix: Option<[[f32; 3]; 3]>,
    /// Sensor black level normalized to 0..1 (Bayer only)
    pub black_level: Option<f32>,
    pub sensor_black_levels: Option<[f32; 4]>,
    /// [R, Gr, Gb, B] linear sensor code / negotiated RAW maximum; None is unity.
    pub sensor_white_levels: Option<[f32; 4]>,
    pub raw_layout: Option<crate::backends::camera::types::RawLayout>,
}

impl<'a> GpuFrameInput<'a> {
    /// Build a `GpuFrameInput` from a `CameraFrame`.
    ///
    /// Handles all YUV, packed, single-plane, and Bayer formats.
    /// Callers must handle `PixelFormat::RGBA` separately (it needs no conversion).
    pub fn from_camera_frame(
        frame: &'a CameraFrame,
    ) -> Result<Self, crate::backends::camera::types::FrameLayoutError> {
        if frame.format == PixelFormat::RGBA {
            return Err(crate::backends::camera::types::FrameLayoutError::UnsupportedFormat);
        }

        let buffer_data = frame.data.as_ref();
        crate::backends::camera::types::validate_frame_layout(
            buffer_data,
            frame.format,
            frame.width,
            frame.height,
            frame.stride,
            frame.yuv_planes.as_ref(),
        )?;

        let mut input = GpuFrameInput {
            format: frame.format,
            width: frame.width,
            height: frame.height,
            y_data: buffer_data,
            y_stride: frame.stride,
            uv_data: None,
            uv_stride: 0,
            uv_dimensions: frame.yuv_planes.map(|p| (p.uv_width, p.uv_height)),
            yuv_color: frame.yuv_planes.map(|p| p.color).unwrap_or_default(),
            v_data: None,
            v_stride: 0,
            colour_gains: None,
            colour_correction_matrix: None,
            black_level: None,
            sensor_black_levels: frame
                .libcamera_metadata
                .as_ref()
                .and_then(|m| m.sensor_black_levels),
            sensor_white_levels: frame
                .libcamera_metadata
                .as_ref()
                .and_then(|m| m.sensor_white_levels),
            raw_layout: frame.libcamera_metadata.as_ref().and_then(|m| m.raw_layout),
        };

        match frame.format {
            PixelFormat::NV12 | PixelFormat::NV21 => {
                let planes = frame
                    .yuv_planes
                    .as_ref()
                    .ok_or(crate::backends::camera::types::FrameLayoutError::MissingPlanes)?;
                input.y_data = crate::backends::camera::types::checked_plane(
                    buffer_data,
                    planes.y_offset,
                    planes.y_size,
                    frame.stride as usize,
                    frame.width as usize,
                    frame.height as usize,
                )?;
                input.uv_data = Some(crate::backends::camera::types::checked_plane(
                    buffer_data,
                    planes.uv_offset,
                    planes.uv_size,
                    planes.uv_stride as usize,
                    planes.uv_width as usize
                        * if frame.format == PixelFormat::I420 {
                            1
                        } else {
                            2
                        },
                    planes.uv_height as usize,
                )?);
                input.uv_stride = planes.uv_stride;
            }
            PixelFormat::I420 => {
                let planes = frame
                    .yuv_planes
                    .as_ref()
                    .ok_or(crate::backends::camera::types::FrameLayoutError::MissingPlanes)?;
                input.y_data = crate::backends::camera::types::checked_plane(
                    buffer_data,
                    planes.y_offset,
                    planes.y_size,
                    frame.stride as usize,
                    frame.width as usize,
                    frame.height as usize,
                )?;
                input.uv_data = Some(crate::backends::camera::types::checked_plane(
                    buffer_data,
                    planes.uv_offset,
                    planes.uv_size,
                    planes.uv_stride as usize,
                    planes.uv_width as usize
                        * if frame.format == PixelFormat::I420 {
                            1
                        } else {
                            2
                        },
                    planes.uv_height as usize,
                )?);
                input.uv_stride = planes.uv_stride;
                if planes.v_size > 0 {
                    input.v_data = Some(crate::backends::camera::types::checked_plane(
                        buffer_data,
                        planes.v_offset,
                        planes.v_size,
                        planes.v_stride as usize,
                        planes.uv_width as usize,
                        planes.uv_height as usize,
                    )?);
                }
                input.v_stride = planes.v_stride;
            }
            PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => {
                let meta = frame.libcamera_metadata.as_ref();
                input.colour_gains = meta.and_then(|m| m.colour_gains);
                input.colour_correction_matrix = meta.and_then(|m| m.colour_correction_matrix);
                input.black_level = meta.and_then(|m| m.black_level);
            }
            _ => {} // YUYV, UYVY, YVYU, VYUY, Gray8, RGB24, ABGR, BGRA: defaults are correct
        }

        Ok(input)
    }
}

#[cfg(test)]
mod bayer_input_tests {
    use super::*;
    use crate::backends::camera::types::{FrameData, YuvPlanes};

    #[test]
    fn completed_photo_releases_global_frame_workspace() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("completed_photo_releases_global_frame_workspace");
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 21,
                height: 19,
                stride: 21,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![64; 21 * 19]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            for method in [
                super::super::DemosaicMethod::Rcd,
                super::super::DemosaicMethod::Lmmse,
                super::super::DemosaicMethod::Bilinear,
            ] {
                let output =
                    crate::pipelines::photo::PostProcessor::memory_photo_for_test(&frame, method)
                        .await
                        .unwrap();
                assert_eq!(output.len(), 21 * 19 * 4);
                let guard = get_gpu_convert_pipeline().await.unwrap();
                let p = guard.as_ref().unwrap();
                assert!(
                    p.demosaic_engine.is_none(),
                    "completed photo retained demosaic workspace"
                );
                assert!(p.bayer_linear_texture.is_none() && p.bayer_demosaic_texture.is_none());
                assert!(p.tex_y.is_none() && p.output_texture.is_none() && p.output_view.is_none());
                assert!(
                    p.readback_staging_buffer.is_none()
                        && p.packed_input_buffer.is_none()
                        && p.unpack_output_buffer.is_none()
                );
                assert!(
                    p.debayer_pipeline.is_some(),
                    "lightweight programs should remain reusable"
                );
            }
        });
    }

    #[test]
    fn bilinear_switch_releases_inactive_demosaic_workspace() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("bilinear_switch_releases_inactive_demosaic_workspace");
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 21,
                height: 19,
                stride: 21,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![64; 21 * 19]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut p = GpuConvertPipeline::new().await.unwrap();
            p.convert_bayer_with_method(&input, super::super::DemosaicMethod::Rcd)
                .unwrap();
            let programs = p.demosaic_programs.clone().unwrap();
            let before = p.read_rgba_to_cpu(21, 19).await.unwrap();
            p.convert_bayer_with_method(&input, super::super::DemosaicMethod::Bilinear)
                .unwrap();
            assert!(
                p.demosaic_engine.is_none(),
                "Bilinear retained inactive RAW/scratch engine"
            );
            p.convert_bayer_with_method(&input, super::super::DemosaicMethod::Rcd)
                .unwrap();
            assert!(Arc::ptr_eq(
                &programs,
                p.demosaic_programs.as_ref().unwrap()
            ));
            assert_eq!(before, p.read_rgba_to_cpu(21, 19).await.unwrap());
        });
    }

    #[test]
    fn photo_scope_error_and_cancel_drop_frame_handles_before_unlock() {
        use std::future::Future;
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "photo_scope_error_and_cancel_drop_frame_handles_before_unlock",
            );
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 513,
                height: 517,
                stride: 513,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![64; 513 * 517]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut p = GpuConvertPipeline::new().await.unwrap();
            // Real early Err after allocating and completing GPU conversion.
            let result: Result<(), BayerError> = (|| {
                let mut photo = p.photo_workspace();
                photo.convert_bayer(&input)?;
                let mut bad = GpuFrameInput::from_camera_frame(&frame).unwrap();
                bad.colour_gains = Some([f32::NAN, 1.0]);
                photo.convert_bayer(&bad)?;
                Ok(())
            })();
            assert_eq!(result, Err(BayerError::InvalidMetadata));
            assert!(p.demosaic_engine.is_none() && p.output_texture.is_none() && p.tex_y.is_none());
            let programs = p.demosaic_programs.clone().unwrap();
            // A genuinely polled future owns the same production photo lease.
            // Cancellation is dropping that future while its scope is live.
            let mut future = Box::pin(async {
                let mut photo = p.photo_workspace();
                photo.convert_bayer(&input).unwrap();
                let _output = photo.read_rgba_to_cpu(513, 517).await.unwrap();
                std::future::pending::<()>().await;
            });
            let waker = futures::task::noop_waker();
            assert!(
                future
                    .as_mut()
                    .poll(&mut std::task::Context::from_waker(&waker))
                    .is_pending()
            );
            drop(future);
            assert!(
                p.demosaic_engine.is_none()
                    && p.bayer_linear_texture.is_none()
                    && p.bayer_demosaic_texture.is_none()
            );
            assert!(p.output_texture.is_none() && p.output_view.is_none() && p.tex_y.is_none());
            assert!(
                p.readback_staging_buffer.is_none()
                    && p.packed_input_buffer.is_none()
                    && p.unpack_output_buffer.is_none()
            );
            assert_eq!(
                (
                    p.packed_input_buffer_size,
                    p.unpack_output_buffer_size,
                    p.readback_staging_buffer_size,
                    p.cached_width,
                    p.cached_height
                ),
                (0, 0, 0, 0, 0)
            );
            assert!(Arc::ptr_eq(
                &programs,
                p.demosaic_programs.as_ref().unwrap()
            ));
            // Resource recreation must be valid immediately after cancellation.
            let mut photo = p.photo_workspace();
            photo.convert_bayer(&input).unwrap();
            assert_eq!(
                photo.read_rgba_to_cpu(513, 517).await.unwrap().len(),
                513 * 517 * 4
            );
        });
    }

    #[test]
    fn photo_scope_retires_previous_frame_before_allocating_another() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "photo_scope_retires_previous_frame_before_allocating_another",
            );
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 21,
                height: 19,
                stride: 21,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![64; 21 * 19]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut p = GpuConvertPipeline::new().await.unwrap();
            p.convert_bayer(&input).unwrap();
            let programs = p.demosaic_programs.clone().unwrap();
            let mut photo = p.photo_workspace();
            assert!(
                photo.demosaic_engine.is_none(),
                "old frame overlaps the new photo workspace"
            );
            assert!(
                photo.bayer_linear_texture.is_none()
                    && photo.output_texture.is_none()
                    && photo.tex_y.is_none()
            );
            photo.convert_bayer(&input).unwrap();
            assert!(Arc::ptr_eq(
                &programs,
                photo.demosaic_programs.as_ref().unwrap()
            ));
        });
    }

    #[test]
    fn captured_geometry_memory_census_releases_all_frame_resources() {
        use crate::backends::camera::types::{FrameMetadata, RawLayout};
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "captured_geometry_memory_census_releases_all_frame_resources",
            );
            return;
        };
        fn bytes(p: &GpuConvertPipeline) -> u64 {
            [
                &p.tex_y,
                &p.tex_uv,
                &p.tex_v,
                &p.output_texture,
                &p.bayer_linear_texture,
                &p.bayer_demosaic_texture,
            ]
            .into_iter()
            .flatten()
            .map(|t| {
                u64::from(t.width())
                    * u64::from(t.height())
                    * u64::from(t.format().block_copy_size(None).unwrap())
            })
            .sum::<u64>()
                + [&p.packed_input_buffer, &p.unpack_output_buffer]
                    .into_iter()
                    .flatten()
                    .map(wgpu::Buffer::size)
                    .sum::<u64>()
                + p.readback_staging_buffer
                    .as_ref()
                    .map_or(0, |buffer| buffer.size())
                + p.demosaic_engine
                    .as_ref()
                    .map_or(0, |e| e.resource_bytes_for_test())
        }
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 3280,
                height: 2464,
                stride: 4112,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![255; 4112 * 2464]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(FrameMetadata {
                    raw_layout: Some(RawLayout::Csi2 { bits: 10 }),
                    colour_gains: Some([1.0, 1.0]),
                    ..Default::default()
                }),
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut p = GpuConvertPipeline::new().await.unwrap();
            for method in [
                super::super::DemosaicMethod::Rcd,
                super::super::DemosaicMethod::Lmmse,
                super::super::DemosaicMethod::Bilinear,
            ] {
                let mut photo = p.photo_workspace();
                photo.convert_bayer_with_method(&input, method).unwrap();
                drop(photo.read_rgba_to_cpu(3280, 2464).await.unwrap());
                let active = bytes(&photo);
                let engine = photo
                    .demosaic_engine
                    .as_ref()
                    .map_or(0, |e| e.resource_bytes_for_test());
                let fixed = photo.uniform_buffer.size()
                    + photo.unpack_uniform_buffer.size()
                    + photo.awb_gains_buffer.size()
                    + photo.awb_sums_buffer.size();
                println!(
                    "MEMORY_CENSUS method={method:?} active_frame_bytes={active} engine_bytes={engine} fixed_buffer_bytes={fixed}"
                );
                assert!(engine <= 16 * 3280 * 2464 + 4 * 544 * 544 * 16 + 64);
                drop(photo);
                assert_eq!(bytes(&p), 0, "completed photo retained frame allocations");
                println!(
                    "MEMORY_CENSUS method={method:?} retained_frame_bytes={} fixed_buffer_bytes={fixed}",
                    bytes(&p)
                );
            }
        });
    }

    #[test]
    fn bilinear_resources_do_not_allocate_full_cpu_rgb_texture() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("bilinear_resources_do_not_allocate_full_cpu_rgb_texture");
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 4,
                height: 4,
                stride: 4,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![64; 16]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            for method in [
                super::super::DemosaicMethod::Bilinear,
                super::super::DemosaicMethod::Rcd,
                super::super::DemosaicMethod::Bilinear,
            ] {
                pipeline.convert_bayer_with_method(&input, method).unwrap();
                let texture = pipeline.bayer_demosaic_texture.as_ref().unwrap();
                let expected = if method == super::super::DemosaicMethod::Bilinear {
                    1
                } else {
                    4
                };
                assert_eq!((texture.width(), texture.height()), (expected, expected));
            }
        });
    }

    #[test]
    fn normal_still_gpu_ignores_planar_chroma_padding() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu("normal_still_gpu_ignores_planar_chroma_padding");
            return;
        };
        pollster::block_on(async {
            let gpu = gpu::get_shared_gpu().await.unwrap();
            println!(
                "Production still adapter: {:?}; features: {:?}; limits: {:?}",
                gpu.info,
                gpu.device.features(),
                gpu.device.limits()
            );
            let frame = CameraFrame {
                width: 3,
                height: 3,
                stride: 4,
                format: PixelFormat::I420,
                data: FrameData::from_owned_vec(vec![
                    100, 100, 100, 1, 100, 100, 100, 1, 100, 100, 100, 1, 128, 128, 255, 255, 128,
                    128, 255, 255, 128, 128, 0, 0, 128, 128, 0, 0,
                ]),
                yuv_planes: Some(YuvPlanes {
                    color: crate::backends::camera::types::YuvColor::Legacy,
                    y_offset: 0,
                    y_size: 12,
                    uv_offset: 12,
                    uv_size: 8,
                    uv_stride: 4,
                    v_offset: 20,
                    v_size: 8,
                    v_stride: 4,
                    uv_width: 2,
                    uv_height: 2,
                }),
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            pipeline.convert(&input).unwrap();
            let output = pipeline.read_rgba_to_cpu(3, 3).await.unwrap();
            for pixel in output.as_chunks::<4>().0 {
                assert_eq!(pixel, &output[0..4]);
            }
        });
    }

    #[test]
    fn normal_yuv_rejects_missing_v_and_invalid_active_geometry() {
        let base = CameraFrame {
            width: 3,
            height: 3,
            stride: 4,
            format: PixelFormat::I420,
            data: FrameData::from_owned_vec(vec![0; 28]),
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
            yuv_planes: Some(YuvPlanes {
                color: Default::default(),
                y_offset: 0,
                y_size: 12,
                uv_offset: 12,
                uv_size: 8,
                uv_stride: 4,
                v_offset: 20,
                v_size: 0,
                v_stride: 4,
                uv_width: 2,
                uv_height: 2,
            }),
        };
        assert!(
            GpuFrameInput::from_camera_frame(&base).is_err(),
            "Missing V must fail before touching GPU state"
        );
        let mut invalid = base.clone();
        let p = invalid.yuv_planes.as_mut().unwrap();
        p.v_size = 8;
        p.uv_width = 4;
        assert!(
            GpuFrameInput::from_camera_frame(&invalid).is_err(),
            "Active chroma cannot include row padding"
        );
    }

    #[test]
    fn normal_yuv_rejects_malformed_planes_without_panicking() {
        let frame = CameraFrame {
            width: 3,
            height: 3,
            stride: 4,
            format: PixelFormat::NV12,
            data: FrameData::from_owned_vec(vec![0; 24]),
            yuv_planes: Some(YuvPlanes {
                color: crate::backends::camera::types::YuvColor::Legacy,
                y_offset: usize::MAX,
                y_size: 12,
                uv_offset: 12,
                uv_size: 8,
                uv_stride: 4,
                v_offset: 0,
                v_size: 0,
                v_stride: 0,
                uv_width: 2,
                uv_height: 2,
            }),
            captured_at: std::time::Instant::now(),
            sensor_timestamp_ns: None,
            libcamera_metadata: None,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GpuFrameInput::from_camera_frame(&frame).map(|_| ())
        }));
        assert!(result.is_ok(), "Malformed producer planes must not panic");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn bayer_unknown_packing_is_an_error_not_stale_pixels() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu("bayer_unknown_packing_is_an_error_not_stale_pixels");
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 8,
                height: 8,
                stride: 17,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![50; 136]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            assert!(pipeline.convert(&input).is_err());
        });
    }

    #[test]
    fn generic_bayer_honors_ccm_without_supplied_gains() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("generic_bayer_honors_ccm_without_supplied_gains");
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            let sites = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]];
            let formats = [
                PixelFormat::BayerRGGB,
                PixelFormat::BayerBGGR,
                PixelFormat::BayerGRBG,
                PixelFormat::BayerGBRG,
            ];
            let ccm = [[2.0, -1.0, 0.0], [-2.0, 1.0, 0.0], [0.0, 0.0, 1.5]];
            for pattern in 0..4 {
                let mut data = Vec::new();
                for y in 0..5 {
                    for x in 0..5 {
                        data.push([32, 64, 64, 128][sites[pattern][(y % 2) * 2 + x % 2]]);
                    }
                    data.extend([255, 255]); // padding must not enter AWB/demosaic
                }
                let frame = CameraFrame {
                    width: 5,
                    height: 5,
                    stride: 7,
                    format: formats[pattern],
                    data: FrameData::from_owned_vec(data),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(crate::backends::camera::types::FrameMetadata {
                        raw_layout: Some(RawLayout::U8),
                        colour_correction_matrix: Some(ccm),
                        ..Default::default()
                    }),
                };
                let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                pipeline
                    .convert_bayer_with_method(&input, super::super::DemosaicMethod::Bilinear)
                    .unwrap(); // actual generic production route
                let linear = pipeline.read_bayer_linear_for_test().await.unwrap();
                // AWB sees integer per-quad sums (8,32,32): gains (2,.5).
                // It remains gray-world, then the independently supplied CCM runs.
                let expected = [64.0 / 255.0, -64.0 / 255.0, 96.0 / 255.0];
                for pixel in linear.as_chunks::<4>().0 {
                    for c in 0..3 {
                        assert!(
                            (pixel[c] - expected[c]).abs() < 2e-6,
                            "pattern={pattern}, pixel={pixel:?}, expected={expected:?}"
                        );
                    }
                }
                let encoded = pipeline.read_rgba_to_cpu(5, 5).await.unwrap();
                let expected_encoded = expected.map(|v: f32| {
                    let v = v.clamp(0.0, 1.0);
                    let v = if v <= 0.0031308 {
                        12.92 * v
                    } else {
                        1.055 * v.powf(1.0 / 2.4) - 0.055
                    };
                    (v * 255.0).round() as u8
                });
                for pixel in encoded.as_chunks::<4>().0 {
                    for c in 0..3 {
                        assert!(pixel[c].abs_diff(expected_encoded[c]) <= 1);
                    }
                }
            }
        });
    }

    #[test]
    fn generic_ccm_without_gains_rejects_unsafe_arithmetic() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu("generic_ccm_without_gains_rejects_unsafe_arithmetic");
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            let frame = CameraFrame {
                width: 2,
                height: 2,
                stride: 2,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec(vec![32, 64, 64, 128]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let mut input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            for ccm in [
                [[f32::MAX; 3]; 3],
                [[1e38, -1e38, 0.0]; 3],
                [[1e30, -1e30, 0.0]; 3],
            ] {
                input.colour_correction_matrix = Some(ccm);
                assert!(matches!(
                    pipeline.convert_bayer(&input),
                    Err(BayerError::UnsafeArithmetic)
                ));
            }
            for ccm in [[[f32::NAN; 3]; 3], [[f32::INFINITY; 3]; 3]] {
                input.colour_correction_matrix = Some(ccm);
                assert!(matches!(
                    pipeline.convert_bayer(&input),
                    Err(BayerError::InvalidMetadata)
                ));
            }
            // Extreme but safe for the worst generated AWB bound, including signed CCM.
            input.colour_correction_matrix = Some([[1e20, -1e20, 0.0]; 3]);
            pipeline.convert(&input).unwrap();
            assert!(
                pipeline
                    .read_bayer_linear_for_test()
                    .await
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite())
            );
        });
    }

    #[test]
    fn public_convert_bayer_preserves_negative_headroom_through_ccm() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "public_convert_bayer_preserves_negative_headroom_through_ccm",
            );
            return;
        };
        pollster::block_on(async {
            let frame = CameraFrame {
                width: 4,
                height: 4,
                stride: 4,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec([16, 64, 16, 64, 64, 96, 64, 96].repeat(2)),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(crate::backends::camera::types::FrameMetadata {
                    colour_gains: Some([1.0, 1.0]),
                    black_level: Some(32.0 / 255.0),
                    colour_correction_matrix: Some([
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    ..Default::default()
                }),
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            pipeline.convert(&input).unwrap();
            let rgba = pipeline.read_rgba_to_cpu(4, 4).await.unwrap();
            let encode = |v: f32| {
                let encoded = if v <= 0.0031308 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                };
                (encoded * 255.0).round() as u8
            };
            let expected = [
                encode(16.0 / 223.0),
                encode(32.0 / 223.0),
                encode(64.0 / 223.0),
                255,
            ];
            for actual in rgba.as_chunks::<4>().0 {
                for c in 0..4 {
                    assert!(
                        actual[c].abs_diff(expected[c]) <= 1,
                        "{actual:?} != {expected:?}"
                    );
                }
            }
        });
    }

    #[test]
    fn public_convert_and_typed_bayer_convert_share_float_result() {
        let Some((_guard, _device, _queue)) = crate::test_gpu::headless_device_with_compute()
        else {
            crate::test_gpu::skip_no_gpu(
                "public_convert_and_typed_bayer_convert_share_float_result",
            );
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            pipeline.warmup_pipelines();
            assert!(pipeline.debayer_pipeline.is_some());
            assert!(pipeline.debayer_finish_pipeline.is_some());
            assert!(pipeline.bayer_linear_texture.is_none());
            assert!(pipeline.tex_y.is_none());
            assert!(pipeline.output_texture.is_none());
            let frame = CameraFrame {
                width: 4,
                height: 4,
                stride: 4,
                format: PixelFormat::BayerRGGB,
                data: FrameData::from_owned_vec([64, 128, 64, 128, 128, 192, 128, 192].repeat(2)),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: Some(crate::backends::camera::types::FrameMetadata {
                    colour_gains: Some([1.0, 1.0]),
                    ..Default::default()
                }),
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            pipeline.convert(&input).unwrap();
            let before = pipeline.read_rgba_to_cpu(4, 4).await.unwrap();
            pipeline.convert_bayer(&input).unwrap();
            let typed = pipeline.read_rgba_to_cpu(4, 4).await.unwrap();
            assert_eq!(
                typed, before,
                "All Bayer callers must use the same float route"
            );
            pipeline.convert(&input).unwrap();
            assert_eq!(before, pipeline.read_rgba_to_cpu(4, 4).await.unwrap());

            // Tight U8 rows remain explicitly supported without RAW metadata,
            // including the existing GPU AWB producer.
            let mut no_metadata = frame.clone();
            no_metadata.libcamera_metadata = None;
            let input = GpuFrameInput::from_camera_frame(&no_metadata).unwrap();
            pipeline.convert(&input).unwrap();
            let automatic = pipeline.read_rgba_to_cpu(4, 4).await.unwrap();
            pipeline.convert_bayer(&input).unwrap();
            assert_eq!(automatic, pipeline.read_rgba_to_cpu(4, 4).await.unwrap());
            assert!(automatic.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
        });
    }
}

/// CSI-2 unpack parameters (32 bytes, matches unpack_csi2.wgsl)
///
/// WGSL uniform structs are rounded up to 16-byte alignment (roundUp(16, maxFieldAlign)),
/// so 5 × u32 = 20 bytes becomes 32 bytes with 12 bytes of trailing padding.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UnpackParams {
    width: u32,
    height: u32,
    packed_stride: u32,
    bit_depth: u32,         // 10, 12, or 14
    output_stride_u32: u32, // padded row stride in u32 units
    _pad: [u32; 3],         // align to 32 bytes for WGSL uniform struct layout
}

/// Binding type specification for pipeline creation
#[derive(Clone, Copy)]
enum BindingSpec {
    Texture,
    StorageTexture,
    FloatStorageTexture,
    ScalarFloatStorageTexture,
    StorageBuffer,
    ReadOnlyStorageBuffer,
    Uniform,
}

impl BindingSpec {
    fn to_layout_entry(self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: match self {
                BindingSpec::Texture => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                BindingSpec::ScalarFloatStorageTexture => wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                BindingSpec::FloatStorageTexture => wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                BindingSpec::StorageTexture => wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                BindingSpec::StorageBuffer => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingSpec::ReadOnlyStorageBuffer => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                BindingSpec::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            },
            count: None,
        }
    }
}

/// Pipeline resources (shader + bind group layout)
struct FormatPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

/// GPU pipeline for format conversion
///
/// Uses a single unified shader (`yuv_convert.wgsl`) for all YUV formats,
/// with a separate debayer shader for Bayer sensor data.
pub struct GpuConvertPipeline {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Unified YUV→RGBA pipeline (shared across all YUV formats)
    yuv_pipeline: Option<FormatPipeline>,
    /// Signed float demosaic and colour correction pipeline.
    debayer_pipeline: Option<FormatPipeline>,
    debayer_finish_pipeline: Option<FormatPipeline>,
    bayer_linear_texture: Option<wgpu::Texture>,
    bayer_demosaic_texture: Option<wgpu::Texture>,
    demosaic_engine: Option<super::demosaic_gpu::GpuDemosaic>,
    demosaic_programs: Option<Arc<super::demosaic_gpu::DemosaicPrograms>>,
    normalize_pipeline: Option<FormatPipeline>,
    /// GPU AWB accumulation pipeline
    awb_pipeline: Option<FormatPipeline>,
    /// GPU AWB finalize pipeline (1 thread: sums → gains)
    awb_finalize_pipeline: Option<FormatPipeline>,
    /// Storage buffer for AWB channel sums (16 bytes, atomics in AWB shader)
    awb_sums_buffer: wgpu::Buffer,
    /// Storage buffer for AWB gains (8 bytes, read by debayer shader)
    awb_gains_buffer: wgpu::Buffer,
    /// CSI-2 unpack compute pipeline
    unpack_pipeline: Option<FormatPipeline>,
    /// Storage buffer for packed CSI-2 input data
    packed_input_buffer: Option<wgpu::Buffer>,
    packed_input_buffer_size: u64,
    /// Storage buffer for unpacked output data (u16 pairs as u32)
    unpack_output_buffer: Option<wgpu::Buffer>,
    unpack_output_buffer_size: u64,
    /// Uniform buffer for unpack parameters
    unpack_uniform_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    // Cached resources for current dimensions/format
    cached_width: u32,
    cached_height: u32,
    cached_format: PixelFormat,
    cached_uv_dims: Option<(u32, u32)>,
    tex_y: Option<wgpu::Texture>,
    tex_uv: Option<wgpu::Texture>,
    tex_v: Option<wgpu::Texture>,
    output_texture: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    /// Cached staging buffer for `read_rgba_to_cpu` readback
    readback_staging_buffer: Option<crate::gpu::readback::ReadbackBuffer>,
    readback_staging_buffer_size: u64,
}

/// A still-photo lease over the reusable converter. Dropping the lease, including
/// a cancelled future or an early error, releases frame-sized handles while the
/// caller still owns the converter mutex. Pipelines and small uniforms survive.
#[must_use]
pub struct PhotoWorkspace<'a> {
    pipeline: &'a mut GpuConvertPipeline,
}
impl std::ops::Deref for PhotoWorkspace<'_> {
    type Target = GpuConvertPipeline;
    fn deref(&self) -> &Self::Target {
        self.pipeline
    }
}
impl std::ops::DerefMut for PhotoWorkspace<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.pipeline
    }
}
impl Drop for PhotoWorkspace<'_> {
    fn drop(&mut self) {
        self.pipeline.release_frame_workspace();
    }
}

impl GpuConvertPipeline {
    /// Use around conversion AND readback; do not return a borrowed GPU texture
    /// from the photo scope. Preview users keep the ordinary reusable API.
    pub fn photo_workspace(&mut self) -> PhotoWorkspace<'_> {
        self.release_frame_workspace();
        PhotoWorkspace { pipeline: self }
    }

    fn release_frame_workspace(&mut self) {
        self.demosaic_engine = None;
        self.bayer_linear_texture = None;
        self.bayer_demosaic_texture = None;
        self.output_view = None;
        self.output_texture = None;
        self.tex_y = None;
        self.tex_uv = None;
        self.tex_v = None;
        self.packed_input_buffer = None;
        self.unpack_output_buffer = None;
        self.readback_staging_buffer = None;
        self.packed_input_buffer_size = 0;
        self.unpack_output_buffer_size = 0;
        self.readback_staging_buffer_size = 0;
        self.cached_width = 0;
        self.cached_height = 0;
        self.cached_uv_dims = None;
        // Submitted commands may retain resources until completion. Poll drives
        // retirement without blocking cancellation; allocator slack is NOT an
        // application-owned frame cache and is not guaranteed to return to OS.
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            warn!(%error, "Photo workspace retirement poll failed");
        }
    }

    /// Create a new conversion pipeline
    pub async fn new() -> Result<Self, String> {
        info!("Initializing format conversion pipelines");

        let gpu = gpu::get_shared_gpu().await?;
        let device = gpu.device;
        let queue = gpu.queue;

        info!(
            adapter_name = %gpu.info.adapter_name,
            adapter_backend = ?gpu.info.backend,
            low_priority = gpu.info.low_priority_enabled,
            "Using shared GPU device for format conversion"
        );

        // Create uniform buffer (shared across all pipelines, sized for largest params struct)
        let uniform_size =
            std::mem::size_of::<ConvertParams>().max(std::mem::size_of::<DebayerParams>());
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert_uniform_buffer"),
            size: uniform_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // AWB storage buffers
        let awb_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("awb_sums_buffer"),
            size: std::mem::size_of::<AwbSums>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let awb_gains_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("awb_gains_buffer"),
            size: std::mem::size_of::<AwbGains>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Unpack pipeline uniform buffer (created eagerly, small fixed size)
        let unpack_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unpack_uniform_buffer"),
            size: std::mem::size_of::<UnpackParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            yuv_pipeline: None,
            debayer_pipeline: None,
            debayer_finish_pipeline: None,
            bayer_linear_texture: None,
            bayer_demosaic_texture: None,
            demosaic_engine: None,
            demosaic_programs: None,
            normalize_pipeline: None,
            awb_pipeline: None,
            awb_finalize_pipeline: None,
            awb_sums_buffer,
            awb_gains_buffer,
            unpack_pipeline: None,
            packed_input_buffer: None,
            packed_input_buffer_size: 0,
            unpack_output_buffer: None,
            unpack_output_buffer_size: 0,
            unpack_uniform_buffer,
            uniform_buffer,
            cached_width: 0,
            cached_height: 0,
            cached_format: PixelFormat::RGBA,
            cached_uv_dims: None,
            readback_staging_buffer: None,
            readback_staging_buffer_size: 0,
            tex_y: None,
            tex_uv: None,
            tex_v: None,
            output_texture: None,
            output_view: None,
        })
    }

    // Bind group layout: 3 textures + output + params (unified YUV shader)
    const BIND_LAYOUT_YUV: [(u32, BindingSpec); 5] = [
        (0, BindingSpec::Texture),        // tex_y
        (1, BindingSpec::Texture),        // tex_uv
        (2, BindingSpec::Texture),        // tex_v
        (3, BindingSpec::StorageTexture), // output
        (4, BindingSpec::Uniform),        // params
    ];

    // Bind group layout: tex_bayer + params + sums (AWB accumulate)
    const BIND_LAYOUT_AWB: [(u32, BindingSpec); 3] = [
        (0, BindingSpec::Texture),       // tex_bayer
        (1, BindingSpec::Uniform),       // AwbParams (reuses uniform_buffer)
        (2, BindingSpec::StorageBuffer), // AwbSums (read_write for atomics)
    ];

    // Bind group layout: sums + gains (AWB finalize)
    const BIND_LAYOUT_AWB_FINALIZE: [(u32, BindingSpec); 2] = [
        (0, BindingSpec::ReadOnlyStorageBuffer), // AwbSums (read-only)
        (1, BindingSpec::StorageBuffer),         // AwbGains (read_write)
    ];

    // Bind group layout for CSI-2 unpack: packed input, unpacked output, params
    const BIND_LAYOUT_UNPACK: [(u32, BindingSpec); 3] = [
        (0, BindingSpec::ReadOnlyStorageBuffer), // packed_data
        (1, BindingSpec::StorageBuffer),         // unpacked_data
        (2, BindingSpec::Uniform),               // UnpackParams
    ];

    /// Ensure the unified YUV pipeline exists
    fn ensure_yuv_pipeline(&mut self) {
        if self.yuv_pipeline.is_none() {
            debug!("Creating unified YUV conversion pipeline");
            self.yuv_pipeline = Some(self.create_pipeline(
                include_str!("yuv_convert.wgsl"),
                "yuv_convert",
                &Self::BIND_LAYOUT_YUV,
            ));
        }
    }

    /// Ensure the AWB accumulation pipeline exists
    fn ensure_awb_pipeline(&mut self) {
        if self.awb_pipeline.is_none() {
            debug!("Creating AWB accumulation pipeline");
            self.awb_pipeline = Some(self.create_pipeline(
                include_str!("bayer_awb.wgsl"),
                "bayer_awb",
                &Self::BIND_LAYOUT_AWB,
            ));
        }
    }

    /// Ensure the AWB finalize pipeline exists
    fn ensure_awb_finalize_pipeline(&mut self) {
        if self.awb_finalize_pipeline.is_none() {
            debug!("Creating AWB finalize pipeline");
            self.awb_finalize_pipeline = Some(self.create_pipeline(
                include_str!("bayer_awb_finalize.wgsl"),
                "bayer_awb_finalize",
                &Self::BIND_LAYOUT_AWB_FINALIZE,
            ));
        }
    }

    /// Precompile all shader pipelines so the first frame doesn't pay compilation cost.
    pub fn warmup_pipelines(&mut self) {
        let start = std::time::Instant::now();
        self.ensure_yuv_pipeline();
        self.ensure_bayer_pipelines();
        self.ensure_awb_pipeline();
        self.ensure_awb_finalize_pipeline();
        self.ensure_unpack_pipeline();
        info!(
            elapsed_ms = format!("{:.1}", start.elapsed().as_millis()),
            "GPU convert pipelines warmed up"
        );
    }

    /// Ensure the CSI-2 unpack pipeline exists
    fn ensure_unpack_pipeline(&mut self) {
        if self.unpack_pipeline.is_none() {
            debug!("Creating CSI-2 unpack pipeline");
            self.unpack_pipeline = Some(self.create_pipeline(
                include_str!("unpack_csi2.wgsl"),
                "unpack_csi2",
                &Self::BIND_LAYOUT_UNPACK,
            ));
        }
    }

    /// Ensure unpack input/output buffers are large enough.
    /// Returns `output_stride_u32` (padded row stride in u32 units for 256-byte alignment).
    fn ensure_unpack_buffers(&mut self, packed_size: u64, width: u32, height: u32) -> u32 {
        // Output stride: width pixels × 2 bytes each, padded to 256 bytes (required by copy_buffer_to_texture)
        let bytes_per_row = align_to_copy_row(width * 2);
        let output_stride_u32 = bytes_per_row / 4;
        let output_size = bytes_per_row as u64 * height as u64;

        // Pad packed input to 4-byte alignment for array<u32> access
        let padded_packed_size = (packed_size + 3) & !3;

        if padded_packed_size > self.packed_input_buffer_size {
            self.packed_input_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("unpack_packed_input"),
                size: padded_packed_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.packed_input_buffer_size = padded_packed_size;
        }

        if output_size > self.unpack_output_buffer_size {
            self.unpack_output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("unpack_output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.unpack_output_buffer_size = output_size;
        }

        output_stride_u32
    }

    /// Encode the CSI-2 unpack compute pass and buffer-to-texture copy.
    fn encode_unpack_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tex_y: &wgpu::Texture,
        width: u32,
        height: u32,
        output_stride_u32: u32,
    ) -> Result<(), String> {
        let unpack_pipeline = self
            .unpack_pipeline
            .as_ref()
            .ok_or("Unpack pipeline not created")?;
        let packed_buf = self
            .packed_input_buffer
            .as_ref()
            .ok_or("Packed input buffer not allocated")?;
        let output_buf = self
            .unpack_output_buffer
            .as_ref()
            .ok_or("Unpack output buffer not allocated")?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("unpack_bind_group"),
            layout: &unpack_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: packed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.unpack_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch: each thread handles one pixel pair
        let workgroups_x = width.div_ceil(2).div_ceil(UNPACK_WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unpack_csi2_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&unpack_pipeline.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups_x, height, 1);
        }

        // Copy unpacked buffer to R16Unorm texture
        let bytes_per_row = output_stride_u32 * 4;
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: output_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: tex_y,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    /// Create a compute pipeline with the given shader and bind group layout spec
    fn create_pipeline<const N: usize>(
        &self,
        shader_source: &str,
        name: &str,
        layout_spec: &[(u32, BindingSpec); N],
    ) -> FormatPipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{}_shader", name)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = layout_spec
            .iter()
            .map(|(binding, spec)| spec.to_layout_entry(*binding))
            .collect();

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", name)),
                    entries: &entries,
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_pipeline_layout", name)),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{}_pipeline", name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        FormatPipeline {
            pipeline,
            bind_group_layout,
        }
    }

    /// Ensure textures are allocated for the given dimensions and format.
    /// `uv_dims` overrides the default UV texture dimensions when provided
    /// (e.g. for MJPEG-decoded I422 reported as I420).
    fn ensure_resources(
        &mut self,
        width: u32,
        height: u32,
        format: PixelFormat,
        uv_dims: Option<(u32, u32)>,
    ) {
        if self.cached_width == width
            && self.cached_height == height
            && self.cached_format == format
            && self.cached_uv_dims == uv_dims
        {
            return;
        }

        debug!(width, height, ?format, "Allocating conversion resources");

        // Calculate texture dimensions based on format, with override for actual UV dims
        let (uv_width, uv_height) = uv_dims.unwrap_or(match format {
            PixelFormat::NV12 | PixelFormat::NV21 | PixelFormat::I420 => (width / 2, height / 2),
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                (width / 2, height)
            }
            PixelFormat::Gray8
            | PixelFormat::RGBA
            | PixelFormat::RGB24
            | PixelFormat::ABGR
            | PixelFormat::BGRA
            | PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => (1, 1),
        });

        // Y plane texture format and dimensions
        let (y_format, y_width) = match format {
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                (wgpu::TextureFormat::Rgba8Unorm, width.div_ceil(2))
            }
            PixelFormat::RGBA | PixelFormat::ABGR | PixelFormat::BGRA => {
                (wgpu::TextureFormat::Rgba8Unorm, width)
            }
            PixelFormat::RGB24 => (wgpu::TextureFormat::R8Unorm, width * 3),
            PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => (wgpu::TextureFormat::R16Unorm, width),
            _ => (wgpu::TextureFormat::R8Unorm, width),
        };

        // UV plane texture format
        let uv_format = match format {
            PixelFormat::NV12 | PixelFormat::NV21 => wgpu::TextureFormat::Rg8Unorm,
            _ => wgpu::TextureFormat::R8Unorm,
        };

        // Create Y texture
        self.tex_y = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_tex_y"),
            size: wgpu::Extent3d {
                width: y_width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: y_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));

        // Create UV texture
        self.tex_uv = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_tex_uv"),
            size: wgpu::Extent3d {
                width: uv_width.max(1),
                height: uv_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: uv_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));

        // Create V texture (I420 only)
        self.tex_v = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_tex_v"),
            size: wgpu::Extent3d {
                width: uv_width.max(1),
                height: uv_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));

        // Create output RGBA texture
        let output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("convert_output_rgba"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        self.output_view = Some(output.create_view(&wgpu::TextureViewDescriptor::default()));
        self.output_texture = Some(output);

        self.cached_width = width;
        self.cached_height = height;
        self.cached_format = format;
        self.cached_uv_dims = uv_dims;
    }

    /// Write debayer params to the uniform buffer and AWB gains to the gains buffer.
    /// Returns whether GPU AWB passes are needed (no ISP gains available).
    fn prepare_debayer_params(
        &self,
        input: &GpuFrameInput,
        raw_layout: RawLayout,
        gpu_demosaic: bool,
    ) -> bool {
        // CCM and WB are independently supplied libcamera metadata. Missing
        // gains still use GPU gray-world AWB, not an identity-CCM override.
        let bl = input.black_level.unwrap_or(0.0);
        let ccm = input
            .colour_correction_matrix
            .unwrap_or(super::BAYER_IDENTITY_CCM);

        let blacks = input.sensor_black_levels.unwrap_or([bl; 4]);
        let params = DebayerParams {
            width: input.width,
            height: input.height,
            pattern: input.format.bayer_pattern_code().unwrap_or(0),
            _padding: 0,
            black_level: blacks[0],
            _pad0: match raw_layout {
                RawLayout::U8 => (65535.0f32 / 65280.0).to_bits(),
                RawLayout::Unpacked16 { bits } => {
                    (65535.0f32 / (((1u32 << bits) - 1) << (16 - bits)) as f32).to_bits()
                }
                RawLayout::Csi2 { bits } => {
                    (65535.0f32 / (((1u32 << bits) - 1) << (16 - bits)) as f32).to_bits()
                }
            },
            _pad1: u32::from(gpu_demosaic),
            _pad2: 0,
            ccm_row0: [ccm[0][0], ccm[0][1], ccm[0][2], blacks[1]],
            ccm_row1: [ccm[1][0], ccm[1][1], ccm[1][2], blacks[2]],
            ccm_row2: [ccm[2][0], ccm[2][1], ccm[2][2], blacks[3]],
            white_levels: input.sensor_white_levels.unwrap_or([1.0; 4]),
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        if let Some(gains) = input.colour_gains {
            // ISP gains available: write directly to the gains buffer
            let awb = AwbGains {
                gain_r: gains[0],
                gain_b: gains[1],
            };
            self.queue
                .write_buffer(&self.awb_gains_buffer, 0, bytemuck::bytes_of(&awb));
            false // no GPU AWB needed
        } else {
            // Clear sums buffer before GPU AWB accumulation
            let zeros = AwbSums {
                sum_r: 0,
                sum_g: 0,
                sum_b: 0,
                _pad: 0,
            };
            self.queue
                .write_buffer(&self.awb_sums_buffer, 0, bytemuck::bytes_of(&zeros));
            true // GPU AWB needed
        }
    }

    /// Encode AWB accumulate + finalize passes into the command encoder.
    fn encode_awb_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        y_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let awb_pipeline = self
            .awb_pipeline
            .as_ref()
            .ok_or("AWB pipeline not created")?;
        let awb_finalize_pipeline = self
            .awb_finalize_pipeline
            .as_ref()
            .ok_or("AWB finalize pipeline not created")?;

        // AWB accumulate bind group
        let awb_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("awb_bind_group"),
            layout: &awb_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.awb_sums_buffer.as_entire_binding(),
                },
            ],
        });

        // AWB finalize bind group
        let awb_finalize_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("awb_finalize_bind_group"),
            layout: &awb_finalize_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.awb_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.awb_gains_buffer.as_entire_binding(),
                },
            ],
        });

        // AWB accumulate pass: each thread handles one 2x2 superpixel
        let awb_wg_x = (width / 2).div_ceil(TILE_WORKGROUP_SIZE);
        let awb_wg_y = (height / 2).div_ceil(TILE_WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awb_accumulate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&awb_pipeline.pipeline);
            pass.set_bind_group(0, Some(&awb_bind_group), &[]);
            pass.dispatch_workgroups(awb_wg_x, awb_wg_y, 1);
        }

        // AWB finalize pass: 1 thread computes gains from sums
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("awb_finalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&awb_finalize_pipeline.pipeline);
            pass.set_bind_group(0, Some(&awb_finalize_bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        Ok(())
    }

    /// Set up Bayer resources and encode unpack + AWB + debayer passes onto a command encoder.
    ///
    /// The canonical signed float Bayer path. Callers append the finishing
    /// pass to the returned encoder before submitting.
    ///
    /// Returns `(encoder, needs_gpu_awb, had_gpu_unpack)` for logging and chaining.
    fn prepare_and_encode_bayer(
        &mut self,
        input: &GpuFrameInput,
        method: super::DemosaicMethod,
    ) -> Result<(wgpu::CommandEncoder, bool, bool), String> {
        // All &mut self calls first (before immutable texture borrows)
        self.ensure_resources(input.width, input.height, input.format, None);
        if method == super::DemosaicMethod::Bilinear {
            self.demosaic_engine = None;
        }
        if method != super::DemosaicMethod::Bilinear
            && self
                .demosaic_engine
                .as_ref()
                .is_none_or(|e| !e.matches(input.width, input.height))
        {
            self.demosaic_engine = Some(super::demosaic_gpu::GpuDemosaic::with_programs(
                &self.device,
                input.width,
                input.height,
                self.demosaic_programs.clone(),
            )?);
            self.demosaic_programs = self.demosaic_engine.as_ref().map(|e| e.programs.clone());
        }
        if method != super::DemosaicMethod::Bilinear && self.normalize_pipeline.is_none() {
            self.normalize_pipeline = Some(self.create_pipeline(
                &super::compose_debayer(include_str!("bayer_normalize.wgsl")),
                "bayer_normalize",
                &[
                    (0, BindingSpec::Texture),
                    (1, BindingSpec::ScalarFloatStorageTexture),
                    (2, BindingSpec::Uniform),
                ],
            ));
        }
        self.ensure_bayer_resources(input.width, input.height, method);
        if input.colour_gains.is_none() {
            self.ensure_awb_pipeline();
            self.ensure_awb_finalize_pipeline();
        }

        let raw_layout = input
            .raw_layout
            .or_else(|| (input.y_stride == input.width).then_some(RawLayout::U8))
            .ok_or("Unknown RAW packing; row stride is not a packing descriptor")?;
        let row_bytes = match raw_layout {
            RawLayout::U8 => input.width,
            RawLayout::Unpacked16 { bits: 8..=16 } => {
                input.width.checked_mul(2).ok_or("RAW row overflow")?
            }
            RawLayout::Csi2 { bits } if [10, 12, 14].contains(&bits) => {
                let group = if bits == 12 { 2 } else { 4 };
                input
                    .width
                    .div_ceil(group)
                    .checked_mul(group * bits / 8)
                    .ok_or("RAW row overflow")?
            }
            _ => return Err("Unsupported RAW bit depth".into()),
        };
        checked_plane(
            input.y_data,
            0,
            input.y_data.len(),
            input.y_stride as usize,
            row_bytes as usize,
            input.height as usize,
        )
        .map_err(|e| e.to_string())?;
        let unpack_info = if let RawLayout::Csi2 { bits: bit_depth } = raw_layout {
            if [10, 12, 14].contains(&bit_depth) {
                self.ensure_unpack_pipeline();
                let packed_size = input.y_data.len() as u64;
                let output_stride_u32 =
                    self.ensure_unpack_buffers(packed_size, input.width, input.height);
                Some((bit_depth, output_stride_u32))
            } else {
                return Err(
                    "Unsupported RAW packing: producer did not describe a recognized layout".into(),
                );
            }
        } else {
            None
        };

        // Immutable references
        let tex_y = self.tex_y.as_ref().ok_or("Y texture not allocated")?;
        let linear_view = self
            .bayer_linear_texture
            .as_ref()
            .ok_or("Bayer linear texture missing")?
            .create_view(&Default::default());

        // Upload data
        if let Some((bit_depth, output_stride_u32)) = unpack_info {
            let packed_buf = self
                .packed_input_buffer
                .as_ref()
                .ok_or("Packed input buffer not allocated")?;
            if input.y_data.len().is_multiple_of(4) {
                self.queue.write_buffer(packed_buf, 0, input.y_data);
            } else {
                let mut aligned = input.y_data.to_vec();
                aligned.resize(aligned.len().next_multiple_of(4), 0);
                self.queue.write_buffer(packed_buf, 0, &aligned);
            }

            let unpack_params = UnpackParams {
                width: input.width,
                height: input.height,
                packed_stride: input.y_stride,
                bit_depth,
                output_stride_u32,
                _pad: [0; 3],
            };
            self.queue.write_buffer(
                &self.unpack_uniform_buffer,
                0,
                bytemuck::bytes_of(&unpack_params),
            );
        } else {
            match raw_layout {
                RawLayout::Unpacked16 { bits } => {
                    // Match CSI-2's left-justified sensor domain BEFORE the
                    // unchanged integer AWB producer sees the texture.
                    let mut promoted =
                        Vec::with_capacity(input.width as usize * input.height as usize * 2);
                    for row in 0..input.height as usize {
                        for code in input.y_data[row * input.y_stride as usize
                            ..row * input.y_stride as usize + input.width as usize * 2]
                            .as_chunks::<2>()
                            .0
                        {
                            let code = u16::from_le_bytes([code[0], code[1]]);
                            promoted.extend_from_slice(&(code << (16 - bits)).to_le_bytes());
                        }
                    }
                    self.write_plane(tex_y, &promoted, input.width * 2, input.width, input.height);
                }
                RawLayout::U8 => {
                    let mut promoted =
                        Vec::with_capacity(input.width as usize * input.height as usize * 2);
                    for row in 0..input.height as usize {
                        for &code in &input.y_data[row * input.y_stride as usize
                            ..row * input.y_stride as usize + input.width as usize]
                        {
                            promoted.extend_from_slice(&((code as u16) << 8).to_le_bytes());
                        }
                    }
                    self.write_plane(tex_y, &promoted, input.width * 2, input.width, input.height);
                }
                RawLayout::Csi2 { .. } => unreachable!("CSI2 uses GPU unpack"),
            }
        }

        let gpu_demosaic = method != super::DemosaicMethod::Bilinear;
        let needs_gpu_awb = self.prepare_debayer_params(input, raw_layout, gpu_demosaic);

        let debayer_pipeline = self
            .debayer_pipeline
            .as_ref()
            .ok_or("Debayer pipeline not created")?;

        let y_view = tex_y.create_view(&wgpu::TextureViewDescriptor::default());

        let demosaic_view = self
            .bayer_demosaic_texture
            .as_ref()
            .unwrap()
            .create_view(&Default::default());
        let debayer_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debayer_bind_group"),
            layout: &debayer_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&linear_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.awb_gains_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&demosaic_view),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bayer_encoder"),
            });

        if let Some((_, output_stride_u32)) = unpack_info {
            self.encode_unpack_pass(
                &mut encoder,
                tex_y,
                input.width,
                input.height,
                output_stride_u32,
            )?;
        }

        if needs_gpu_awb {
            self.encode_awb_passes(&mut encoder, &y_view, input.width, input.height)?;
        }

        if gpu_demosaic {
            let engine = self.demosaic_engine.as_ref().unwrap();
            let normalized = engine.raw.create_view(&Default::default());
            let pipeline = self.normalize_pipeline.as_ref().unwrap();
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("normalize CFA"),
                layout: &pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&y_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&normalized),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("normalize CFA once"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(input.width.div_ceil(16), input.height.div_ceil(16), 1);
            }
            engine.encode(
                &self.device,
                &self.queue,
                &mut encoder,
                self.bayer_demosaic_texture.as_ref().unwrap(),
                input.format.bayer_pattern_code().unwrap(),
                method,
            );
            self.queue.submit([encoder.finish()]);
            let bound = engine.validate(&self.device)?;
            super::validate_bayer_colour(
                Some(
                    input
                        .colour_gains
                        .unwrap_or([super::BAYER_AWB_GAIN_BOUND; 2]),
                ),
                input.colour_correction_matrix,
                bound,
            )
            .map_err(str::to_owned)?;
            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("validated Bayer colour"),
                });
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("debayer_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&debayer_pipeline.pipeline);
            pass.set_bind_group(0, Some(&debayer_bind_group), &[]);
            let workgroups_x = input.width.div_ceil(TILE_WORKGROUP_SIZE);
            let workgroups_y = input.height.div_ceil(TILE_WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        Ok((encoder, needs_gpu_awb, unpack_info.is_some()))
    }

    /// Compile canonical Bayer pipelines without allocating frame surfaces.
    fn ensure_bayer_pipelines(&mut self) {
        if self.debayer_pipeline.is_none() {
            self.debayer_pipeline = Some(self.create_pipeline(
                &super::compose_debayer(include_str!("debayer_texture.wgsl")),
                "debayer",
                &[
                    (0, BindingSpec::Texture),
                    (1, BindingSpec::FloatStorageTexture),
                    (2, BindingSpec::Uniform),
                    (3, BindingSpec::ReadOnlyStorageBuffer),
                    (4, BindingSpec::Texture),
                ],
            ));
        }
        if self.debayer_finish_pipeline.is_none() {
            self.debayer_finish_pipeline = Some(self.create_pipeline(
                include_str!("debayer_finish.wgsl"),
                "debayer_finish",
                &[(0, BindingSpec::Texture), (1, BindingSpec::StorageTexture)],
            ));
        }
    }

    fn ensure_bayer_resources(&mut self, width: u32, height: u32, method: super::DemosaicMethod) {
        self.ensure_bayer_pipelines();
        // Bilinear never samples the engine RGB binding; use a tiny placeholder
        // rather than allocating another full-frame float surface for video.
        let (rgb_width, rgb_height) = if method == super::DemosaicMethod::Bilinear {
            (1, 1)
        } else {
            (width, height)
        };
        if self
            .bayer_demosaic_texture
            .as_ref()
            .is_none_or(|t| t.width() != rgb_width || t.height() != rgb_height)
        {
            self.bayer_demosaic_texture =
                Some(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gpu_demosaic_linear_camera_rgb"),
                    size: wgpu::Extent3d {
                        width: rgb_width,
                        height: rgb_height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                }));
        }
        if self
            .bayer_linear_texture
            .as_ref()
            .is_none_or(|t| t.width() != width || t.height() != height)
        {
            self.bayer_linear_texture =
                Some(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("bayer_signed_linear"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                }));
        }
    }

    /// Canonical signed float Bayer conversion with typed fail-closed errors.
    /// The working surface is separate from HDR+; callers receive encoded RGBA.
    pub fn convert_bayer(&mut self, input: &GpuFrameInput) -> Result<&wgpu::Texture, BayerError> {
        self.convert_bayer_with_method(input, super::DemosaicMethod::Rcd)
    }

    /// Explicit photo method; shared GPU RCD/LMMSE followed by common colour finishing.
    pub fn convert_bayer_with_method(
        &mut self,
        input: &GpuFrameInput,
        method: super::DemosaicMethod,
    ) -> Result<&wgpu::Texture, BayerError> {
        if !input.format.is_bayer() {
            return Err("Bayer conversion requires a Bayer source".into());
        }
        if input.width < 2 || input.height < 2 {
            return Err("Bayer conversion needs at least a 2x2 CFA lattice".into());
        }
        if input
            .colour_gains
            .is_some_and(|g| g.iter().any(|v| !v.is_finite() || *v <= 0.0))
            || input
                .colour_correction_matrix
                .is_some_and(|m| m.iter().flatten().any(|v| !v.is_finite()))
        {
            return Err(BayerError::InvalidMetadata);
        }
        crate::backends::camera::types::validate_bayer_levels(
            input.black_level,
            input.sensor_black_levels,
            input.sensor_white_levels,
        )
        .map_err(|e| {
            if e.starts_with("Unsafe") {
                BayerError::UnsafeArithmetic
            } else {
                BayerError::InvalidMetadata
            }
        })?;
        // Texture adapter clamps sensor samples to [-0.14798465, 1]. Validate
        // against supplied gains or the proven integer GPU-AWB bound BEFORE
        // prepare/encoding any WB/CCM dispatch; unity is not an AWB bound.
        let gain_bounds = input
            .colour_gains
            .unwrap_or([super::BAYER_AWB_GAIN_BOUND; 2]);
        super::validate_bayer_colour(Some(gain_bounds), input.colour_correction_matrix, 1.0)
            .map_err(|_| BayerError::UnsafeArithmetic)?;
        let (mut encoder, _, _) = self.prepare_and_encode_bayer(input, method)?;
        let linear_view = self
            .bayer_linear_texture
            .as_ref()
            .ok_or("Missing Bayer linear texture")?
            .create_view(&Default::default());
        let output_view = self.output_view.as_ref().ok_or("Missing Bayer output")?;
        let pipeline = self
            .debayer_finish_pipeline
            .as_ref()
            .ok_or("Missing Bayer finishing pipeline")?;
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debayer_finish_bindings"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&linear_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bayer_scene_to_encoder"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, Some(&bindings), &[]);
            pass.dispatch_workgroups(input.width.div_ceil(16), input.height.div_ceil(16), 1);
        }
        self.queue.submit([encoder.finish()]);
        self.output_texture
            .as_ref()
            .ok_or("Missing Bayer output".into())
    }

    /// Convert frame to RGBA using unified shader
    pub fn convert(&mut self, input: &GpuFrameInput) -> Result<&wgpu::Texture, String> {
        let start = std::time::Instant::now();

        // Every Bayer caller uses the same signed float correction and finish.
        if input.format.is_bayer() {
            return self.convert_bayer(input).map_err(|error| error.to_string());
        }

        // YUV/packed-YUV path
        let uv_dims = input.uv_dimensions;

        self.ensure_resources(input.width, input.height, input.format, uv_dims);
        self.ensure_yuv_pipeline();

        let tex_y = self.tex_y.as_ref().ok_or("Y texture not allocated")?;
        let tex_uv = self.tex_uv.as_ref().ok_or("UV texture not allocated")?;
        let tex_v = self.tex_v.as_ref().ok_or("V texture not allocated")?;
        let output_view = self
            .output_view
            .as_ref()
            .ok_or("Output view not allocated")?;

        self.upload_textures(input, tex_y, tex_uv, tex_v)?;

        let params = ConvertParams {
            width: input.width,
            height: input.height,
            format: input.format.gpu_format_code(),
            y_stride: input.y_stride,
            uv_stride: input.uv_stride,
            v_stride: input.v_stride,
            color: input.yuv_color as u32,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        let format_pipeline = self
            .yuv_pipeline
            .as_ref()
            .ok_or("YUV pipeline not created")?;

        let y_view = tex_y.create_view(&wgpu::TextureViewDescriptor::default());
        let uv_view = tex_uv.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = tex_v.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv_convert_bind_group"),
            layout: &format_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&uv_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("convert_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("convert_compute_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&format_pipeline.pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups_x = input.width.div_ceil(TILE_WORKGROUP_SIZE);
            let workgroups_y = input.height.div_ceil(TILE_WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let elapsed = start.elapsed();
        if elapsed.as_millis() > 2 {
            debug!(
                elapsed_ms = format!("{:.2}", elapsed.as_micros() as f64 / 1000.0),
                width = input.width,
                height = input.height,
                format = ?input.format,
                "Format conversion"
            );
        }

        self.output_texture
            .as_ref()
            .ok_or("Output texture not allocated".to_string())
    }

    /// Write a single plane to a GPU texture.
    fn write_plane(
        &self,
        texture: &wgpu::Texture,
        data: &[u8],
        bytes_per_row: u32,
        width: u32,
        height: u32,
    ) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload textures based on format
    fn upload_textures(
        &self,
        input: &GpuFrameInput,
        tex_y: &wgpu::Texture,
        tex_uv: &wgpu::Texture,
        tex_v: &wgpu::Texture,
    ) -> Result<(), String> {
        match input.format {
            // Packed 4:2:2 formats
            PixelFormat::YUYV | PixelFormat::UYVY | PixelFormat::YVYU | PixelFormat::VYUY => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width.div_ceil(2),
                    input.height,
                );
            }

            // NV12/NV21: Y plane + UV plane
            PixelFormat::NV12 | PixelFormat::NV21 => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
                if let Some(uv_data) = input.uv_data {
                    let (w, h) = input
                        .uv_dimensions
                        .ok_or("Missing active chroma geometry")?;
                    self.write_plane(tex_uv, uv_data, input.uv_stride, w, h);
                }
            }

            // I420: Y + U + V planes
            // UV dimensions derived from actual data (supports I422/I444 tagged as I420)
            PixelFormat::I420 => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
                if let Some(uv_data) = input.uv_data {
                    let (uv_w, uv_h) = input
                        .uv_dimensions
                        .ok_or("Missing active chroma geometry")?;
                    self.write_plane(tex_uv, uv_data, input.uv_stride, uv_w, uv_h);
                }
                if let Some(v_data) = input.v_data {
                    let (v_w, v_h) = input
                        .uv_dimensions
                        .ok_or("Missing active chroma geometry")?;
                    self.write_plane(tex_v, v_data, input.v_stride, v_w, v_h);
                }
            }

            // Gray8: single channel
            PixelFormat::Gray8 => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
            }

            // RGBA-family: 4 bytes per pixel, uploaded as Rgba8Unorm
            PixelFormat::RGB24 => self.write_plane(
                tex_y,
                input.y_data,
                input.y_stride,
                input.width * 3,
                input.height,
            ),
            PixelFormat::RGBA | PixelFormat::ABGR | PixelFormat::BGRA => {
                self.write_plane(
                    tex_y,
                    input.y_data,
                    input.y_stride,
                    input.width,
                    input.height,
                );
            }

            // Bayer uploads belong exclusively to the validated float path.
            PixelFormat::BayerRGGB
            | PixelFormat::BayerBGGR
            | PixelFormat::BayerGRBG
            | PixelFormat::BayerGBRG => {
                return Err("Bayer upload requires the canonical Bayer conversion path".into());
            }
        }

        Ok(())
    }

    #[cfg(test)]
    pub(crate) async fn read_bayer_linear_for_test(&self) -> Result<Vec<f32>, String> {
        let texture = self
            .bayer_linear_texture
            .as_ref()
            .ok_or("Missing linear Bayer texture")?;
        let (width, height) = (texture.width(), texture.height());
        let row = align_to_copy_row(width * 16);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bayer_linear_test_readback"),
            size: u64::from(row) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let readback = crate::gpu::readback::ReadbackBuffer::from_buffer(
            &self.device,
            staging,
            "bayer linear test readback",
        );
        let data = readback.read_async().await?;
        let mut result = Vec::with_capacity((width * height * 4) as usize);
        for bytes in data.chunks_exact(row as usize) {
            result.extend_from_slice(bytemuck::cast_slice(&bytes[..(width * 16) as usize]));
        }
        drop(data);
        Ok(result)
    }

    /// Read back the converted RGBA data to CPU memory
    pub async fn read_rgba_to_cpu(&mut self, width: u32, height: u32) -> Result<Vec<u8>, String> {
        let output = self
            .output_texture
            .as_ref()
            .ok_or("Output texture not allocated")?;

        let padded_bytes_per_row = align_to_copy_row(width * 4);
        let required_size = (padded_bytes_per_row * height) as u64;

        // Reuse cached staging buffer if it matches, otherwise allocate and cache
        if self.readback_staging_buffer_size != required_size {
            self.readback_staging_buffer = Some(crate::gpu::readback::ReadbackBuffer::from_buffer(
                &self.device,
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("convert_staging"),
                    size: required_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                "convert_staging",
            ));
            self.readback_staging_buffer_size = required_size;
        }

        let readback = self.readback_staging_buffer.as_ref().unwrap();
        let staging_buffer = readback.buffer();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let data = readback.read_async().await?;
        let mut output_data = Vec::with_capacity((width * height * 4) as usize);

        if padded_bytes_per_row == width * 4 {
            output_data.extend_from_slice(&data[..(width * height * 4) as usize]);
        } else {
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + (width * 4) as usize;
                output_data.extend_from_slice(&data[start..end]);
            }
        }

        drop(data);

        Ok(output_data)
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output_texture.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn review_chroma_texture_sizes(&self) -> [(u32, u32); 2] {
        [self.tex_uv.as_ref().unwrap(), self.tex_v.as_ref().unwrap()]
            .map(|texture| (texture.width(), texture.height()))
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

/// Cached global pipeline instance
static GPU_CONVERT_PIPELINE: std::sync::OnceLock<tokio::sync::Mutex<Option<GpuConvertPipeline>>> =
    std::sync::OnceLock::new();

/// Get or create the shared pipeline instance
pub async fn get_gpu_convert_pipeline()
-> Result<tokio::sync::MutexGuard<'static, Option<GpuConvertPipeline>>, String> {
    let lock = GPU_CONVERT_PIPELINE.get_or_init(|| tokio::sync::Mutex::new(None));
    let mut guard = lock.lock().await;

    if guard.is_none() {
        match GpuConvertPipeline::new().await {
            Ok(pipeline) => {
                *guard = Some(pipeline);
            }
            Err(e) => {
                warn!("Failed to initialize convert pipeline: {}", e);
                return Err(e);
            }
        }
    }

    Ok(guard)
}

#[cfg(test)]
mod method_routing_tests {
    use super::*;
    use crate::backends::camera::types::FrameData;
    use crate::shaders::DemosaicMethod;
    #[test]
    fn production_gpu_routes_three_methods_and_generic_defaults_rcd() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "production_gpu_routes_three_methods_and_generic_defaults_rcd",
            );
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            for format in [
                PixelFormat::BayerRGGB,
                PixelFormat::BayerBGGR,
                PixelFormat::BayerGRBG,
                PixelFormat::BayerGBRG,
            ] {
                let data: Vec<u8> = (0..19)
                    .flat_map(|y| {
                        (0..21).map(move |x| {
                            (25 + (x * 37 + y * 53 + x * y * 7) % 97 + if x > 10 { 40 } else { 0 })
                                as u8
                        })
                    })
                    .collect();
                let frame = CameraFrame {
                    width: 21,
                    height: 19,
                    stride: 21,
                    format,
                    data: FrameData::from_owned_vec(data.clone()),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(crate::backends::camera::types::FrameMetadata {
                        colour_gains: Some([1.0, 1.0]),
                        ..Default::default()
                    }),
                };
                let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                let mut outputs = Vec::new();
                for method in [
                    DemosaicMethod::Rcd,
                    DemosaicMethod::Lmmse,
                    DemosaicMethod::Bilinear,
                ] {
                    pipeline.convert_bayer_with_method(&input, method).unwrap();
                    let actual = pipeline.read_bayer_linear_for_test().await.unwrap();
                    outputs.push(actual);
                }
                assert_ne!(outputs[0], outputs[1]);
                assert_ne!(outputs[0], outputs[2]);
                assert_ne!(outputs[1], outputs[2]);
                pipeline.convert(&input).unwrap();
                assert_eq!(
                    outputs[0],
                    pipeline.read_bayer_linear_for_test().await.unwrap()
                );
            }
        });
    }
}

#[cfg(test)]
mod method_colour_tests {
    use super::*;
    use crate::backends::camera::types::{FrameData, FrameMetadata};
    use crate::shaders::DemosaicMethod;
    #[test]
    fn all_methods_preserve_optional_ccm_and_missing_gain_policy() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "all_methods_preserve_optional_ccm_and_missing_gain_policy",
            );
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            let ccm = [[1.7, -0.4, -0.3], [-0.2, 1.4, -0.2], [-0.1, -0.3, 1.4]];
            let gains = [1.7, 1.3];
            let black = [0.12, 0.08, 0.1, 0.15];
            let white = [0.83, 0.92, 0.85, 0.95];
            for format in [
                PixelFormat::BayerRGGB,
                PixelFormat::BayerBGGR,
                PixelFormat::BayerGRBG,
                PixelFormat::BayerGBRG,
            ] {
                let data: Vec<u8> = (0..9)
                    .flat_map(|y| (0..11).map(move |x| ((x * 37 + y * 53 + x * y * 7) % 256) as u8))
                    .collect();
                let frame = CameraFrame {
                    width: 11,
                    height: 9,
                    stride: 11,
                    format,
                    data: FrameData::from_owned_vec(data.clone()),
                    yuv_planes: None,
                    captured_at: std::time::Instant::now(),
                    sensor_timestamp_ns: None,
                    libcamera_metadata: Some(FrameMetadata {
                        colour_gains: Some(gains),
                        colour_correction_matrix: Some(ccm),
                        sensor_black_levels: Some(black),
                        sensor_white_levels: Some(white),
                        ..Default::default()
                    }),
                };
                let mut input = GpuFrameInput::from_camera_frame(&frame).unwrap();
                // Independent optional CCM remains honored with generated gray-world gains.
                input.colour_gains = None;
                input.sensor_black_levels = None;
                input.sensor_white_levels = None;
                for method in [
                    DemosaicMethod::Rcd,
                    DemosaicMethod::Lmmse,
                    DemosaicMethod::Bilinear,
                ] {
                    input.colour_correction_matrix = None;
                    pipeline.convert_bayer_with_method(&input, method).unwrap();
                    let identity = pipeline.read_bayer_linear_for_test().await.unwrap();
                    input.colour_correction_matrix =
                        Some([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]);
                    pipeline.convert_bayer_with_method(&input, method).unwrap();
                    let permuted = pipeline.read_bayer_linear_for_test().await.unwrap();
                    for (a, b) in identity
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(permuted.as_chunks::<4>().0)
                    {
                        assert_eq!([a[1], a[2], a[0]], [b[0], b[1], b[2]]);
                    }
                    input.colour_correction_matrix = Some([[f32::MAX; 3]; 3]);
                    assert_eq!(
                        pipeline
                            .convert_bayer_with_method(&input, method)
                            .unwrap_err(),
                        BayerError::UnsafeArithmetic
                    );
                }
            }
        });
    }
}

#[cfg(test)]
mod engine_error_tests {
    use super::*;
    #[test]
    fn gpu_engine_arithmetic_rejection_keeps_typed_unsafe_error() {
        assert_eq!(
            BayerError::from("Unsafe Bayer colour arithmetic".to_owned()),
            BayerError::UnsafeArithmetic
        );
        assert_eq!(
            BayerError::from("Unsafe demosaic arithmetic".to_owned()),
            BayerError::UnsafeArithmetic
        );
        assert!(matches!(
            BayerError::from("Unknown RAW packing".to_owned()),
            BayerError::InvalidSource(_)
        ));
    }
}

#[cfg(test)]
mod fully_gpu_contract_tests {
    #[test]
    fn production_demosaic_never_reconstructs_on_cpu() {
        let single = include_str!("gpu_convert.rs");
        let prepare = single
            .split("fn prepare_and_encode_bayer(")
            .nth(1)
            .unwrap()
            .split("fn ensure_bayer_pipelines")
            .next()
            .unwrap();
        assert!(
            !prepare.contains("normalized_bayer_lattice("),
            "single photo CPU normalization"
        );
        assert!(
            !prepare.contains("demosaic::demosaic("),
            "single photo CPU demosaic"
        );
        let hdr = include_str!("../pipelines/photo/burst_mode/mod.rs");
        let finish = hdr
            .split("pub async fn demosaic")
            .nth(1)
            .unwrap_or(hdr)
            .split("/// Compute sharpness from pre-extracted")
            .next()
            .unwrap();
        assert!(!finish.contains("demosaic::demosaic("), "HDR CPU demosaic");
        assert!(
            !finish.contains("read_buffer::<f32>(merged_planes_buffer"),
            "HDR full-plane readback"
        );
    }
}

#[cfg(test)]
mod readback_owner_tests {
    use super::*;
    #[test]
    fn converter_failed_cached_map_then_public_readback_recovers_same_pixels() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "converter_failed_cached_map_then_public_readback_recovers_same_pixels",
            );
            return;
        };
        pollster::block_on(async {
            let mut pipeline = GpuConvertPipeline::new().await.unwrap();
            let frame = crate::backends::camera::types::CameraFrame {
                width: 4,
                height: 4,
                stride: 4,
                format: PixelFormat::Gray8,
                data: crate::backends::camera::types::FrameData::from_owned_vec(vec![96; 16]),
                yuv_planes: None,
                captured_at: std::time::Instant::now(),
                sensor_timestamp_ns: None,
                libcamera_metadata: None,
            };
            let input = GpuFrameInput::from_camera_frame(&frame).unwrap();
            pipeline.convert(&input).unwrap();
            let expected = pipeline.read_rgba_to_cpu(4, 4).await.unwrap();
            let cache = pipeline.readback_staging_buffer.as_ref().unwrap();
            cache.buffer().destroy();
            let error = cache.read_async().await.err().unwrap();
            assert!(error.contains("destroyed"), "{error}");
            assert_eq!(pipeline.read_rgba_to_cpu(4, 4).await.unwrap(), expected);
            assert_eq!(pipeline.read_rgba_to_cpu(4, 4).await.unwrap(), expected);
        });
    }
}
