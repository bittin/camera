// SPDX-License-Identifier: GPL-3.0-only
//! Filters and preblur never participate in RAW/demosaic GPU pacing.
use super::*;
use crate::test_gpu::{PhotoEvents, PhotoTrace as CompletionTrace};
use tracing_subscriber::layer::SubscriberExt;

#[test]
fn photo_filters_and_preblur_are_unpaced_and_exact() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("photo_filters_and_preblur_are_unpaced_and_exact");
        return;
    };
    let events = PhotoEvents::default();
    let subscriber = tracing_subscriber::registry().with(CompletionTrace(events.clone()));
    tracing::subscriber::with_default(subscriber, || {
        pollster::block_on(async {
            let mut filter = GpuFilterPipeline::new().await.unwrap();
            let mut comparisons = 0;
            for (width, height) in [(67, 19), (513, 1057), (67, 4001)] {
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
                assert_eq!(
                    filter
                        .apply_filter_rgba(&pixels, width, height, FilterType::Standard)
                        .await
                        .unwrap(),
                    pixels
                );
                for code in 1..=14 {
                    let mode = FilterType::from_gpu_filter_code(code);
                    events.lock().unwrap().clear();
                    let full = filter
                        .apply_filter_rgba(&pixels, width, height, mode)
                        .await
                        .unwrap();
                    assert!(
                        events.lock().unwrap().is_empty(),
                        "ordinary filter was paced"
                    );
                    let texture = filter.input_texture.as_ref().unwrap().clone();
                    for region in [
                        (0, 0, width, height),
                        if height == 19 {
                            (3, 5, 17, 9)
                        } else {
                            (3, 5, width - 7, height - 11)
                        },
                        (width - 1, height - 1, 1, 1),
                    ] {
                        events.lock().unwrap().clear();
                        let actual = filter
                            .apply_filter_texture_region(&texture, mode, region)
                            .await
                            .unwrap();
                        let (x, y, w, h) = region;
                        let expected: Vec<u8> = (y..y + h)
                            .flat_map(|row| {
                                full[((row * width + x) * 4) as usize
                                    ..((row * width + x + w) * 4) as usize]
                                    .iter()
                                    .copied()
                            })
                            .collect();
                        assert!(
                            actual == expected,
                            "exact RGBA mismatch for {mode:?}, {width}x{height}, {region:?}"
                        );
                        assert!(
                            events.lock().unwrap().is_empty(),
                            "filters and preblur must emit zero photo completions: {:?}",
                            events.lock().unwrap()
                        );
                        assert!(filter.input_texture.is_none());
                        assert!(filter.preblur_texture.is_none());
                        assert!(filter.output_buffer.is_none());
                        assert!(filter.staging_buffer.is_none());
                        assert_eq!((filter.cached_width, filter.cached_height), (0, 0));
                        comparisons += 1;
                    }
                    // Photo lease cleanup must not alter subsequent upload-path pixels.
                    events.lock().unwrap().clear();
                    assert!(
                        filter
                            .apply_filter_rgba(&pixels, width, height, mode)
                            .await
                            .unwrap()
                            == full
                    );
                    assert!(events.lock().unwrap().is_empty());
                }
            }
            assert_eq!(comparisons, 126);
        })
    });
}
