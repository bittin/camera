// SPDX-License-Identifier: GPL-3.0-only
//! Exercise real, completed photo submissions rather than CPU-only scheduling.
use super::*;
use crate::shaders::DemosaicMethod;
use crate::test_gpu::{PhotoEvents as Events, PhotoTrace as Capture};
use tracing_subscriber::layer::SubscriberExt;

fn check_batches(events: &Events, stage: &str, width: u32, height: u32, alignment: u32) {
    let expected = if width == 0 || height == 0 {
        0
    } else {
        let rows = (262144 / width / alignment).max(1) * alignment;
        height.div_ceil(rows) as usize
    };
    let events = events.lock().unwrap();
    let completed = events
        .iter()
        .filter(|e| {
            e.get("message")
                .is_some_and(|m| m == "Photo GPU batch completed")
                && e.get("stage").is_some_and(|s| s.trim_matches('"') == stage)
        })
        .count();
    assert_eq!(
        completed, expected,
        "{stage}: missing completed bounded GPU submissions"
    );
    let bands: Vec<_> = events
        .iter()
        .filter(|e| {
            e.get("message")
                .is_some_and(|m| m == "Photo conversion band completed")
                && e.get("stage").is_some_and(|s| s.trim_matches('"') == stage)
        })
        .collect();
    assert_eq!(bands.len(), expected);
    let mut next = 0;
    for band in bands {
        let y: u32 = band["row"].parse().unwrap();
        let h: u32 = band["height"].parse().unwrap();
        assert_eq!(y, next);
        assert_eq!(band["width"].parse::<u32>().unwrap(), width);
        assert!(width * h <= 262144 || h == 1);
        next += h;
    }
    assert_eq!(next, height);
}

#[test]
fn photo_batches_complete_and_match_unpaced_bytes() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("photo_batches_complete_and_match_unpaced_bytes");
        return;
    };
    let events = Events::default();
    let subscriber = tracing_subscriber::registry().with(Capture(events.clone()));
    let _subscriber = tracing::subscriber::set_default(subscriber);
    pollster::block_on(async {
        let mut photo = GpuConvertPipeline::new().await.unwrap();
        let mut generic = GpuConvertPipeline::new().await.unwrap();

        for layout in [
            RawLayout::Csi2 { bits: 10 },
            RawLayout::Csi2 { bits: 12 },
            RawLayout::Csi2 { bits: 14 },
            RawLayout::U8,
            RawLayout::Unpacked16 { bits: 16 },
        ] {
            for (w, h) in [(513, 1101), (65, 8193), (3, 5)] {
                for format in [
                    PixelFormat::BayerRGGB,
                    PixelFormat::BayerBGGR,
                    PixelFormat::BayerGRBG,
                    PixelFormat::BayerGBRG,
                ] {
                    for gains in [None, Some([1.3, 1.7])] {
                        // Change active samples between successive conversions
                        // on the retained workspace, including padded odd rows.
                        let code =
                            |x: u32, y: u32| x * 977 + y * 577 + u32::from(gains.is_some()) * 37;
                        let (data, stride) = if layout == RawLayout::U8 {
                            let stride = w + 7;
                            let mut data = vec![0xff; (stride * h) as usize];
                            for y in 0..h {
                                for x in 0..w {
                                    data[(y * stride + x) as usize] = code(x, y) as u8;
                                }
                            }
                            (data, stride)
                        } else {
                            packed_photo_tests::fixture(layout, w, h, code)
                        };
                        let mut input = packed_photo_tests::input(&data, layout, w, h, stride);
                        input.format = format;
                        input.colour_gains = gains;
                        events.lock().unwrap().clear();
                        generic
                            .convert_bayer_with_method(&input, DemosaicMethod::Bilinear)
                            .unwrap();

                        assert!(
                            !events.lock().unwrap().iter().any(|e| e
                                .get("message")
                                .is_some_and(|m| m == "Photo GPU batch completed")),
                            "generic conversion must remain unpaced"
                        );
                        for roi in [
                            (0, 0, w, h),
                            (1, 1, w - 1, h - 1),
                            (w - 1, h - 1, 1, 1),
                            (w, h, 0, 0),
                        ] {
                            events.lock().unwrap().clear();
                            if roi == (0, 0, w, h) {
                                // This is also the development extent used before full-frame filters.
                                photo
                                    .convert_photo_bayer_with_method(
                                        &input,
                                        DemosaicMethod::Bilinear,
                                    )
                                    .unwrap();
                            } else {
                                photo
                                    .convert_photo_bayer_region_with_method(
                                        &input,
                                        DemosaicMethod::Bilinear,
                                        roi,
                                    )
                                    .unwrap();
                            }
                            let actual = photo.read_rgba_region_to_cpu(roi).await.unwrap();
                            let expected = generic.read_rgba_region_to_cpu(roi).await.unwrap();
                            assert!(
                                actual == expected,
                                "generic {layout:?} {format:?} {w}x{h} gains={gains:?} {roi:?}"
                            );

                            if matches!(layout, RawLayout::Csi2 { .. }) {
                                check_batches(&events, "unpack", w, h, 1);
                            }
                            check_batches(
                                &events,
                                "bilinear",
                                roi.2,
                                roi.3,
                                PHOTO_WORKGROUP_SIZE[1],
                            );
                            let preparation = events
                                .lock()
                                .unwrap()
                                .iter()
                                .filter(|e| {
                                    e.get("message")
                                        .is_some_and(|m| m == "Photo GPU batch completed")
                                        && e.get("stage")
                                            .is_some_and(|s| s.trim_matches('"') == "prepare")
                                })
                                .count();
                            assert_eq!(preparation, 1, "prepare/AWB must happen only once");
                        }
                    }
                }
            }
        }
    });
}

/// RCD/LMMSE retain full-sensor development even for regional readback.
#[test]
fn advanced_photo_rcd_batches_are_fenced_and_byte_exact() {
    advanced_photo_batches_are_fenced_and_byte_exact(DemosaicMethod::Rcd);
}

#[test]
fn advanced_photo_lmmse_batches_are_fenced_and_byte_exact() {
    advanced_photo_batches_are_fenced_and_byte_exact(DemosaicMethod::Lmmse);
}

fn advanced_photo_batches_are_fenced_and_byte_exact(method: DemosaicMethod) {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("advanced_photo_batches_are_fenced_and_byte_exact");
        return;
    };
    let events = Events::default();
    let subscriber = tracing_subscriber::registry().with(Capture(events.clone()));
    let _subscriber = tracing::subscriber::set_default(subscriber);
    pollster::block_on(async {
        let mut photo = GpuConvertPipeline::new().await.unwrap();
        let mut generic = GpuConvertPipeline::new().await.unwrap();
        {
            for layout in [
                RawLayout::Csi2 { bits: 10 },
                RawLayout::Csi2 { bits: 12 },
                RawLayout::Csi2 { bits: 14 },
                RawLayout::Unpacked16 { bits: 16 },
            ] {
                let (w, h) = (515, 517);
                let (data, stride) =
                    packed_photo_tests::fixture(layout, w, h, |x, y| x * 971 + y * 173);
                for format in [
                    PixelFormat::BayerRGGB,
                    PixelFormat::BayerBGGR,
                    PixelFormat::BayerGRBG,
                    PixelFormat::BayerGBRG,
                ] {
                    for gains in [None, Some([1.3, 1.7])] {
                        let mut input = packed_photo_tests::input(&data, layout, w, h, stride);
                        input.format = format;
                        input.colour_gains = gains;
                        // Negative headroom and distant bright/dark extrema exercise
                        // the global LMMSE reduction across bands and tile halos.
                        input.sensor_black_levels = Some([0.1, 0.08, 0.12, 0.09]);
                        events.lock().unwrap().clear();
                        generic.convert_bayer_with_method(&input, method).unwrap();
                        photo
                            .convert_photo_bayer_region_unpaced_with_method(
                                &input,
                                method,
                                (1, 1, w - 2, h - 2),
                            )
                            .unwrap();
                        assert!(!events.lock().unwrap().iter().any(|e| {
                            e.get("message")
                                .is_some_and(|m| m == "Photo GPU batch completed")
                        }));
                        let expected = generic.read_rgba_to_cpu(w, h).await.unwrap();
                        assert_eq!(photo.read_rgba_to_cpu(w, h).await.unwrap(), expected);
                        events.lock().unwrap().clear();
                        if gains.is_none() {
                            photo
                                .convert_photo_bayer_with_method(&input, method)
                                .unwrap();
                        } else {
                            photo
                                .convert_photo_bayer_region_with_method(
                                    &input,
                                    method,
                                    (1, 1, w - 2, h - 2),
                                )
                                .unwrap();
                        }
                        let count = |stage: &str| {
                            events
                                .lock()
                                .unwrap()
                                .iter()
                                .filter(|e| {
                                    e.get("message")
                                        .is_some_and(|m| m == "Photo GPU batch completed")
                                        && e.get("stage")
                                            .is_some_and(|s| s.trim_matches('"') == stage)
                                })
                                .count()
                        };
                        // Test all missing stages together so RED shows both methods.
                        assert_eq!(
                            (
                                count("normalize"),
                                count("bounds"),
                                count("demosaic-first"),
                                count("demosaic-second"),
                                count("demosaic-third"),
                                count("demosaic-fourth"),
                                count("demosaic-final"),
                                count("colour"),
                                count("finish")
                            ),
                            (2, 2, 4, 4, 4, 4, 4, 2, 2),
                            "{method:?}: missing fenced bounded stages"
                        );
                        if matches!(layout, RawLayout::Csi2 { .. }) {
                            check_batches(&events, "unpack", w, h, 1);
                        }
                        assert_eq!(count("prepare"), 1);
                        assert_eq!(count("demosaic-status"), 1);
                        for stage in ["normalize", "colour", "finish"] {
                            check_batches(&events, stage, w, h, 16);
                        }
                        let completed: Vec<String> = events
                            .lock()
                            .unwrap()
                            .iter()
                            .filter(|e| {
                                e.get("message")
                                    .is_some_and(|m| m == "Photo GPU batch completed")
                            })
                            .map(|e| e["stage"].trim_matches('"').to_owned())
                            .collect();
                        let unpack_count = if matches!(layout, RawLayout::Csi2 { .. }) {
                            2
                        } else {
                            0
                        };
                        let mut order = vec!["unpack"; unpack_count];
                        order.extend(["prepare", "normalize", "normalize", "bounds", "bounds"]);
                        for _ in 0..4 {
                            order.extend([
                                "demosaic-first",
                                "demosaic-second",
                                "demosaic-third",
                                "demosaic-fourth",
                                "demosaic-final",
                            ]);
                        }
                        order.extend(["demosaic-status", "colour", "colour", "finish", "finish"]);
                        assert_eq!(completed, order, "fences must precede the next stage/band");
                        {
                            let events_guard = events.lock().unwrap();
                            let stages: Vec<_> = events_guard
                                .iter()
                                .filter(|e| {
                                    e.get("message")
                                        .is_some_and(|m| m == "Photo demosaic stage completed")
                                })
                                .collect();
                            assert_eq!(stages.len(), 22);
                            for stage in stages {
                                let width: u32 = stage["width"].parse().unwrap();
                                let height: u32 = stage["height"].parse().unwrap();
                                if stage["stage"].trim_matches('"') == "bounds" {
                                    assert!(width * height <= 262144);
                                } else {
                                    assert!(
                                        width <= 544 && height <= 544,
                                        "one fixed tile stage per fence"
                                    );
                                }
                            }
                        }
                        assert_eq!(
                            photo.read_rgba_to_cpu(w, h).await.unwrap(),
                            expected,
                            "{method:?} {layout:?} {format:?} {gains:?}"
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn bounded_photo_workspace_error_and_cancel_retire_packed_resources() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu(
            "bounded_photo_workspace_error_and_cancel_retire_packed_resources",
        );
        return;
    };
    pollster::block_on(async {
        let mut pipeline = GpuConvertPipeline::new().await.unwrap();
        let layout = RawLayout::Csi2 { bits: 14 };
        let (w, h) = (513, 1101);
        let (data, stride) = packed_photo_tests::fixture(layout, w, h, |x, y| x * 971 + y * 173);
        let input = packed_photo_tests::input(&data, layout, w, h, stride);
        let session = PhotoSession::new(w, h, input.format, layout, stride).unwrap();
        session.prepare(&mut pipeline).unwrap();
        {
            let mut workspace = pipeline.photo_workspace_for_session(
                Some(session.clone()),
                &input,
                DemosaicMethod::Bilinear,
            );
            workspace
                .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
                .unwrap();
            workspace.complete();
        }
        assert!(session.has_workspace());
        let result: Result<(), BayerError> = (|| {
            let mut workspace = pipeline.photo_workspace_for_session(
                Some(session.clone()),
                &input,
                DemosaicMethod::Bilinear,
            );
            workspace.convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)?;
            let mut bad = packed_photo_tests::input(&data, layout, w, h, stride);
            bad.colour_gains = Some([f32::NAN, 1.0]);
            workspace.convert_photo_bayer_with_method(&bad, DemosaicMethod::Bilinear)?;
            workspace.complete();
            Ok(())
        })();
        assert_eq!(result, Err(BayerError::InvalidMetadata));
        assert!(!session.has_workspace());
        assert!(pipeline.tex_y.is_none() && pipeline.output_texture.is_none());
        assert!(pipeline.packed_input_buffer.is_none() && pipeline.unpack_output_buffer.is_none());

        session.prepare(&mut pipeline).unwrap();
        let mut pending = Box::pin(async {
            let mut workspace = pipeline.photo_workspace_for_session(
                Some(session.clone()),
                &input,
                DemosaicMethod::Bilinear,
            );
            workspace
                .convert_photo_bayer_with_method(&input, DemosaicMethod::Bilinear)
                .unwrap();
            std::future::pending::<()>().await;
            workspace.complete();
        });
        // Poll real conversion to completion, then cancel the enclosing photo
        // future before it can mark the retained workspace successful.
        let mut context = std::task::Context::from_waker(futures::task::noop_waker_ref());
        assert!(std::future::Future::poll(pending.as_mut(), &mut context).is_pending());
        drop(pending);
        assert!(!session.has_workspace());
        assert!(pipeline.tex_y.is_none() && pipeline.output_texture.is_none());
        assert!(pipeline.packed_input_buffer.is_none() && pipeline.unpack_output_buffer.is_none());
        assert_eq!(
            (
                pipeline.packed_input_buffer_size,
                pipeline.unpack_output_buffer_size
            ),
            (0, 0)
        );
    });
}
