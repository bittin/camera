// SPDX-License-Identifier: GPL-3.0-only
//! Photo-only queue pacing. Call on a blocking worker, never the UI executor.
use crate::gpu::wgpu;

const PHOTO_BATCH_PIXELS: u32 = 256 * 1024;

pub(crate) fn photo_bands(
    width: u32,
    height: u32,
    alignment: u32,
) -> impl Iterator<Item = (u32, u32)> {
    let available_rows = (PHOTO_BATCH_PIXELS / width.max(1)).max(1);
    let alignment = alignment.max(1);
    // An indivisible workgroup-row strip is the minimum batch even if a
    // future device supports a texture wider than the nominal pixel budget.
    let rows = (available_rows / alignment * alignment).max(alignment);
    let height = if width == 0 { 0 } else { height };
    (0..height)
        .step_by(rows as usize)
        .map(move |y| (y, rows.min(height - y)))
}

pub(crate) fn submit_photo_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    stage: &str,
) -> Result<(), String> {
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .map_err(|error| format!("Photo GPU {stage} batch failed: {error}"))?;
    // Do not queue the remaining frame behind rendering: only this batch was
    // submitted, and its completion is confirmed before the next is encoded.
    std::thread::yield_now();
    tracing::debug!(stage, "Photo GPU batch completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn photo_bands_bound_work_and_cover_partial_tail() {
        assert_eq!(
            photo_bands(1024, 513, 16).collect::<Vec<_>>(),
            vec![(0, 256), (256, 256), (512, 1)]
        );
        for (width, height, alignment) in [
            (4032, 3024, 4),
            (65, 8193, 4),
            (1, 1, 16),
            (16384, 33, 16),
            (65536, 33, 16),
        ] {
            let mut covered = 0;
            for (origin, rows) in photo_bands(width, height, alignment) {
                assert_eq!(origin, covered);
                assert!(rows > 0);
                let budget =
                    u64::from(PHOTO_BATCH_PIXELS).max(u64::from(width) * u64::from(alignment));
                assert!(u64::from(rows) * u64::from(width) <= budget);
                if origin + rows < height {
                    assert!(rows.is_multiple_of(alignment));
                }
                covered += rows;
            }
            assert_eq!(covered, height);
        }
        assert_eq!(photo_bands(0, 42, 16).count(), 0);
        assert_eq!(photo_bands(42, 0, 16).count(), 0);
    }
}
