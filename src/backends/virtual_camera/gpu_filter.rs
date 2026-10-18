// SPDX-License-Identifier: GPL-3.0-only

//! Synchronous virtual-camera adapter for the shared RGBA filter implementation.

use crate::app::FilterType;
use crate::backends::camera::types::{BackendError, BackendResult, CameraFrame, PixelFormat};
use crate::shaders::GpuFilterPipeline;

/// Owns a continuous filter workspace, independent of the global photo pipeline.
/// Same-sized frames reuse textures and readback storage without photo pacing.
pub struct GpuFilterRenderer {
    pipeline: GpuFilterPipeline,
}

impl GpuFilterRenderer {
    pub async fn new() -> BackendResult<Self> {
        Ok(Self {
            pipeline: GpuFilterPipeline::new()
                .await
                .map_err(BackendError::InitializationFailed)?,
        })
    }

    pub fn apply_filter(
        &mut self,
        frame: &CameraFrame,
        filter: FilterType,
    ) -> BackendResult<Vec<u8>> {
        if frame.format != PixelFormat::RGBA {
            return Err(BackendError::FormatNotSupported(
                "Only RGBA input is supported".into(),
            ));
        }

        pollster::block_on(self.pipeline.apply_filter_rgba_with_stride(
            &frame.data,
            frame.width,
            frame.height,
            frame.stride,
            filter,
        ))
        .map_err(BackendError::Other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::FrameData;
    use crate::test_gpu::{PhotoEvents, PhotoTrace};
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn virtual_filters_match_tight_uploads_including_preblur_and_standard() {
        let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
            crate::test_gpu::skip_no_gpu(
                "virtual_filters_match_tight_uploads_including_preblur_and_standard",
            );
            return;
        };
        let events = PhotoEvents::default();
        let subscriber = tracing_subscriber::registry().with(PhotoTrace(events.clone()));
        tracing::subscriber::with_default(subscriber, || {
            let mut renderer = pollster::block_on(GpuFilterRenderer::new()).unwrap();
            let mut reference = pollster::block_on(GpuFilterPipeline::new()).unwrap();
            for (width, height) in [(17, 19), (3, 2)] {
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
                for stride in [width * 4, width * 4 + 12] {
                    let mut data = vec![0xa5; (stride * height) as usize];
                    for (source, destination) in pixels
                        .chunks_exact((width * 4) as usize)
                        .zip(data.chunks_exact_mut(stride as usize))
                    {
                        destination[..source.len()].copy_from_slice(source);
                    }
                    let mut frame = CameraFrame {
                        width,
                        height,
                        stride,
                        format: PixelFormat::RGBA,
                        data: FrameData::from_owned_vec(data),
                        yuv_planes: None,
                        captured_at: std::time::Instant::now(),
                        sensor_timestamp_ns: None,
                        libcamera_metadata: None,
                    };
                    for code in 0..=14 {
                        let mode = FilterType::from_gpu_filter_code(code);
                        let expected = pollster::block_on(
                            reference.apply_filter_rgba(&pixels, width, height, mode),
                        )
                        .unwrap();
                        for _ in 0..3 {
                            assert_eq!(
                                renderer.apply_filter(&frame, mode).unwrap(),
                                expected,
                                "{mode:?}, {width}x{height}, stride {stride}"
                            );
                        }
                    }
                    frame.format = PixelFormat::NV12;
                    for mode in [FilterType::Standard, FilterType::Mono] {
                        assert!(matches!(renderer.apply_filter(&frame, mode),
                            Err(BackendError::FormatNotSupported(message))
                            if message == "Only RGBA input is supported"));
                    }
                }
            }
            assert!(
                events.lock().unwrap().is_empty(),
                "virtual filters were photo-paced"
            );
        });
    }
}
