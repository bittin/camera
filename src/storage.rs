// SPDX-License-Identifier: GPL-3.0-only

//! Storage utilities for managing photo and video files

use crate::constants::file_formats;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, warn};

/// Gallery thumbnail data: image handle, RGBA bytes, width, height, and file path
pub type GalleryThumbnailData = (
    cosmic::widget::image::Handle,
    Arc<Vec<u8>>,
    u32,
    u32,
    PathBuf,
);

/// Load latest thumbnail for gallery button
///
/// Scans both photo and video directories for files, finds the most recent one,
/// and loads it as both an image handle and RGBA data for custom rendering.
/// For videos, extracts the first frame as a thumbnail.
/// Returns gallery thumbnail data for the most recent file
pub async fn load_latest_thumbnail(
    photos_dir: PathBuf,
    videos_dir: PathBuf,
) -> Option<GalleryThumbnailData> {
    // Get list of photo and video files from both directories (using blocking std::fs)
    let mut entries = tokio::task::spawn_blocking(move || {
        let mut files: Vec<(PathBuf, std::time::SystemTime)> = Vec::new();

        // Scan photos directory for image files
        if let Ok(entries) = std::fs::read_dir(&photos_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if file_formats::is_image_extension(&ext_str)
                        && let Ok(metadata) = entry.metadata()
                        && let Ok(modified) = metadata.modified()
                    {
                        files.push((path, modified));
                    }
                }
            }
        }

        // Scan videos directory for video files
        if let Ok(entries) = std::fs::read_dir(&videos_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if file_formats::is_video_extension(&ext_str)
                        && let Ok(metadata) = entry.metadata()
                        && let Ok(modified) = metadata.modified()
                    {
                        files.push((path, modified));
                    }
                }
            }
        }

        files
    })
    .await
    .ok()?;

    if entries.is_empty() {
        return None;
    }

    // Sort by modification time (newest first)
    entries.sort_by_key(|b| std::cmp::Reverse(b.1));

    let latest_path = entries.first()?.0.clone();
    let extension = latest_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    debug!(path = ?latest_path, "Loading latest thumbnail");

    // Check if it's a video file
    if file_formats::is_video_extension(&extension) {
        let (handle, rgba, w, h) = load_video_thumbnail(latest_path.clone()).await?;
        return Some((handle, rgba, w, h, latest_path));
    }

    // Load image bytes
    let bytes = tokio::fs::read(&latest_path).await.ok()?;
    let bytes_clone = bytes.clone();

    // Decode image to RGBA in blocking task
    let (rgba_data, width, height) = tokio::task::spawn_blocking(move || {
        use image::GenericImageView;

        // JPEG: decode straight to a thumbnail-sized image with libjpeg-turbo's
        // scaled DCT decode. Decoding a 12 MP capture at full resolution to feed
        // a 44 pt button costs ~68 ms and ~48 MB of RGBA (which the gallery
        // primitive then copies again for the BGRA swizzle); the 1/8 scale decode
        // is ~26 ms and under 1 MB.
        if let Some(scaled) = decode_jpeg_thumbnail(&bytes_clone) {
            return Some(scaled);
        }

        let img = image::load_from_memory(&bytes_clone).ok()?;
        let rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

        Some((rgba.into_raw(), width, height))
    })
    .await
    .ok()??;

    let handle = cosmic::widget::image::Handle::from_bytes(bytes);

    Some((handle, Arc::new(rgba_data), width, height, latest_path))
}

/// Smallest thumbnail edge we decode to, in pixels.
///
/// The gallery button is 44 pt; 256 px keeps it crisp on HiDPI and while the
/// hover overlay scales it up.
const THUMBNAIL_MIN_EDGE: usize = 256;

/// Maximum number of progressive scans accepted when decoding a thumbnail.
/// Real progressive JPEGs use on the order of ten; the limit only bounds the
/// time a crafted file can spend on a blocking thread.
const THUMBNAIL_SCAN_LIMIT: u32 = 500;

/// Decode a JPEG straight to thumbnail size using libjpeg-turbo's scaled DCT decode.
///
/// Returns `None` for anything that is not a lossy JPEG (PNG, DNG, lossless JPEG),
/// so the caller can fall back to the general `image` decoder.
fn decode_jpeg_thumbnail(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut decompressor = turbojpeg::Decompressor::new().ok()?;

    // Guard the decode against a pathological progressive JPEG: a file with
    // thousands of scans would otherwise pin a blocking thread for a long time.
    let _ = decompressor.set_scan_limit(THUMBNAIL_SCAN_LIMIT);

    let header = decompressor.read_header(bytes).ok()?;

    // Scaled decode is only available for lossy JPEG.
    if header.is_lossless {
        return None;
    }

    // Pick the most aggressive scale that still leaves the short edge above the
    // target, so small images are never upscaled or blurred.
    let factor = [
        turbojpeg::ScalingFactor::ONE_EIGHTH,
        turbojpeg::ScalingFactor::ONE_QUARTER,
        turbojpeg::ScalingFactor::ONE_HALF,
    ]
    .into_iter()
    .find(|f| {
        let scaled = header.scaled(*f);
        scaled.width.min(scaled.height) >= THUMBNAIL_MIN_EDGE
    })
    .unwrap_or(turbojpeg::ScalingFactor::ONE);

    decompressor.set_scaling_factor(factor).ok()?;

    let scaled = header.scaled(factor);
    if scaled.width == 0 || scaled.height == 0 {
        return None;
    }

    let mut pixels = vec![0u8; scaled.width * scaled.height * 4];
    let output = turbojpeg::Image {
        pixels: &mut pixels[..],
        width: scaled.width,
        pitch: scaled.width * 4,
        height: scaled.height,
        format: turbojpeg::PixelFormat::RGBA,
    };

    if let Err(e) = decompressor.decompress(bytes, output) {
        debug!(error = %e, "Scaled JPEG thumbnail decode failed, falling back to image crate");
        return None;
    }

    debug!(
        source = format!("{}x{}", header.width, header.height),
        thumbnail = format!("{}x{}", scaled.width, scaled.height),
        scale = %factor,
        "Decoded gallery thumbnail with scaled JPEG decode"
    );

    Some((pixels, scaled.width as u32, scaled.height as u32))
}

/// Load a thumbnail from a video file by extracting the first frame
async fn load_video_thumbnail(
    video_path: PathBuf,
) -> Option<(cosmic::widget::image::Handle, Arc<Vec<u8>>, u32, u32)> {
    debug!(path = ?video_path, "Extracting thumbnail from video");

    // Extract first frame from video in blocking task (uses GStreamer)
    let result = tokio::task::spawn_blocking(move || {
        use crate::backends::virtual_camera::load_preview_frame;

        match load_preview_frame(&video_path) {
            Ok(frame) => {
                let width = frame.width;
                let height = frame.height;
                let rgba_data: Vec<u8> = frame.data.to_vec();

                // Encode as PNG for the Handle
                let png_bytes = encode_rgba_to_png(&rgba_data, width, height)?;

                Some((png_bytes, rgba_data, width, height))
            }
            Err(e) => {
                warn!(error = ?e, "Failed to extract video thumbnail");
                None
            }
        }
    })
    .await
    .ok()??;

    let (png_bytes, rgba_data, width, height) = result;
    let handle = cosmic::widget::image::Handle::from_bytes(png_bytes);

    Some((handle, Arc::new(rgba_data), width, height))
}

/// Encode RGBA data to PNG bytes
fn encode_rgba_to_png(rgba_data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    use image::{ImageBuffer, Rgba};

    let img: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, rgba_data.to_vec())?;

    let mut png_bytes = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut png_bytes);

    img.write_to(&mut cursor, image::ImageFormat::Png).ok()?;

    Some(png_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buffer = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buffer);
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, 90)
            .encode(img.as_raw(), width, height, image::ExtendedColorType::Rgb8)
            .unwrap();
        buffer
    }

    #[test]
    fn large_jpeg_is_decoded_at_a_reduced_scale() {
        let (rgba, w, h) = decode_jpeg_thumbnail(&jpeg(4000, 3000)).expect("decode failed");

        // 1/8 is the most aggressive scale that keeps the short edge above the
        // 256 px target (3000/8 = 375).
        assert_eq!((w, h), (500, 375));
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    #[test]
    fn thumbnail_scale_keeps_the_short_edge_above_the_target() {
        // 1/8 and 1/4 leave the short edge at 100 and 200 px, below the target,
        // so the wider 1/2 scale must win.
        let (_, w, h) = decode_jpeg_thumbnail(&jpeg(1600, 800)).expect("decode failed");
        assert_eq!((w, h), (800, 400));
        assert!(h as usize >= THUMBNAIL_MIN_EDGE);
    }

    #[test]
    fn small_jpeg_is_decoded_at_full_size() {
        let (rgba, w, h) = decode_jpeg_thumbnail(&jpeg(320, 240)).expect("decode failed");
        assert_eq!((w, h), (320, 240));
        assert_eq!(rgba.len(), 320 * 240 * 4);
    }

    #[test]
    fn non_jpeg_input_falls_back_to_the_general_decoder() {
        let png = encode_rgba_to_png(&vec![255u8; 64 * 64 * 4], 64, 64).unwrap();
        assert!(decode_jpeg_thumbnail(&png).is_none());
    }

    #[test]
    fn decoded_thumbnail_keeps_the_source_colours() {
        let source = image::RgbImage::from_pixel(1024, 1024, image::Rgb([30, 160, 220]));
        let mut buffer = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buffer);
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, 95)
            .encode(source.as_raw(), 1024, 1024, image::ExtendedColorType::Rgb8)
            .unwrap();

        let (rgba, w, h) = decode_jpeg_thumbnail(&buffer).expect("decode failed");
        assert_eq!((w, h), (256, 256));

        let centre = (((h / 2) * w + w / 2) * 4) as usize;
        assert!(rgba[centre].abs_diff(30) < 10);
        assert!(rgba[centre + 1].abs_diff(160) < 10);
        assert!(rgba[centre + 2].abs_diff(220) < 10);
        assert_eq!(rgba[centre + 3], 255, "alpha channel must be opaque");
    }
}
