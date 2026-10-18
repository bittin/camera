use super::*;

#[test]
fn r16_input_cpu_usage_and_existing_feature_contract() {
    let base = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    for qualified in [false, true] {
        for format in [
            wgpu::TextureFormat::R16Unorm,
            wgpu::TextureFormat::R16Snorm,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba32Float,
        ] {
            let expected = if qualified && format == wgpu::TextureFormat::R16Unorm {
                base | wgpu::TextureUsages::STORAGE_BINDING
            } else {
                base
            };
            assert_eq!(r16_input_usage(qualified, format), expected);
        }
    }
    let features = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    assert_eq!(wgpu::TextureFormat::R16Unorm.required_features(), features);
    assert!(
        wgpu::TextureFormat::R16Unorm
            .guaranteed_format_features(features)
            .allowed_usages
            .contains(r16_input_usage(true, wgpu::TextureFormat::R16Unorm))
    );
}

#[test]
fn qualified_r16_resources_gpu_first_allocation_and_retirement() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("qualified_r16_resources_gpu_first_allocation_and_retirement");
        return;
    };
    pollster::block_on(async {
        let gpu = gpu::get_shared_gpu().await.unwrap();
        let expected = gpu
            .device
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
        let mut p = GpuConvertPipeline::new().await.unwrap();
        assert_eq!(p.r16_input_storage_capable, expected);
        for format in [
            PixelFormat::BayerRGGB,
            PixelFormat::BayerBGGR,
            PixelFormat::BayerGRBG,
            PixelFormat::BayerGBRG,
            PixelFormat::NV12,
            PixelFormat::I420,
            PixelFormat::YUYV,
            PixelFormat::Gray8,
            PixelFormat::RGB24,
            PixelFormat::RGBA,
        ] {
            for (width, height) in [(17, 19), (65, 33), (17, 19)] {
                if !expected && format.is_bayer() {
                    // Never allocate R16Unorm without its required feature.
                    continue;
                }
                {
                    let mut w = p.photo_workspace();
                    w.ensure_resources(width, height, format, None);
                    let y = w.tex_y.as_ref().unwrap();
                    let base = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
                    let should_add = expected
                        && matches!(
                            format,
                            PixelFormat::BayerRGGB
                                | PixelFormat::BayerBGGR
                                | PixelFormat::BayerGRBG
                                | PixelFormat::BayerGBRG
                        );
                    assert_eq!(
                        y.usage(),
                        if should_add {
                            base | wgpu::TextureUsages::STORAGE_BINDING
                        } else {
                            base
                        }
                    );
                    assert_eq!(w.tex_uv.as_ref().unwrap().usage(), base);
                    assert_eq!(w.tex_v.as_ref().unwrap().usage(), base);
                    assert_eq!(
                        w.output_texture.as_ref().unwrap().usage(),
                        wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC
                    );
                    let y_width = match format {
                        PixelFormat::YUYV => width.div_ceil(2),
                        PixelFormat::RGB24 => width * 3,
                        _ => width,
                    };
                    assert_eq!((y.width(), y.height()), (y_width, height));
                    let output = w.output_texture.as_ref().unwrap();
                    assert_eq!((output.width(), output.height()), (width, height));
                    if format.is_bayer() {
                        assert_eq!(y.format(), wgpu::TextureFormat::R16Unorm);
                    }
                }
                assert!(
                    p.tex_y.is_none()
                        && p.output_texture.is_none()
                        && p.tex_uv.is_none()
                        && p.tex_v.is_none()
                );
                assert_eq!((p.cached_width, p.cached_height), (0, 0));
                assert_eq!(p.r16_input_storage_capable, expected);
            }
        }
    });
}
