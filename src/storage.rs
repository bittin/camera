// SPDX-License-Identifier: GPL-3.0-only

//! Storage utilities for managing photo and video files

use crate::constants::file_formats;
use crate::pipelines::capture_metadata::embedded_image_orientation;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, warn};

/// Immutable decoded gallery thumbnail. Clones share pixels, metadata and identity;
/// loading a replacement gets a fresh identity, even for the same path and size.
#[derive(Debug, Clone)]
pub struct GalleryThumbnailData(Arc<GalleryThumbnail>);

#[derive(Debug)]
struct GalleryThumbnail {
    id: uuid::Uuid,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    path: PathBuf,
}

impl GalleryThumbnailData {
    pub(crate) fn new(path: PathBuf, rgba: Vec<u8>, width: u32, height: u32) -> Self {
        Self(Arc::new(GalleryThumbnail {
            id: uuid::Uuid::new_v4(),
            rgba,
            width,
            height,
            path,
        }))
    }

    pub(crate) fn id(&self) -> uuid::Uuid {
        self.0.id
    }

    pub(crate) fn rgba(&self) -> &[u8] {
        &self.0.rgba
    }

    pub(crate) fn dimensions(&self) -> (u32, u32) {
        (self.0.width, self.0.height)
    }

    pub(crate) fn path(&self) -> &std::path::Path {
        &self.0.path
    }
}

/// Load latest thumbnail for gallery button
///
/// Scans both photo and video directories for files, finds the most recent one,
/// and loads its oriented RGBA pixels for custom rendering.
/// For videos, extracts the first frame as a thumbnail.
/// Returns gallery thumbnail data for the most recent file
pub async fn load_latest_thumbnail(
    photos_dir: PathBuf,
    videos_dir: PathBuf,
) -> Option<GalleryThumbnailData> {
    let latest_path = tokio::task::spawn_blocking(move || {
        let directories = [
            (
                photos_dir,
                file_formats::is_image_extension as fn(&str) -> bool,
            ),
            (videos_dir, file_formats::is_video_extension),
        ];
        directories
            .into_iter()
            .flat_map(|(directory, accepts)| {
                std::fs::read_dir(directory)
                    .into_iter()
                    .flatten()
                    .filter_map(Result::ok)
                    .filter_map(move |entry| {
                        let path = entry.path();
                        if !accepts(&path.extension()?.to_string_lossy()) {
                            return None;
                        }
                        Some((path, entry.metadata().ok()?.modified().ok()?))
                    })
            })
            // min keeps the first equal timestamp, like the previous stable sort.
            .min_by_key(|(_, modified)| std::cmp::Reverse(*modified))
            .map(|(path, _)| path)
    })
    .await
    .ok()??;
    let extension = latest_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    debug!(path = ?latest_path, "Loading latest thumbnail");

    // Check if it's a video file
    if file_formats::is_video_extension(&extension) {
        return load_video_thumbnail(latest_path).await;
    }

    // Load image bytes
    let bytes = tokio::fs::read(&latest_path).await.ok()?;

    // Decode image to RGBA in blocking task
    let (rgba_data, width, height) = tokio::task::spawn_blocking(move || {
        use image::GenericImageView;

        // JPEG: decode straight to a thumbnail-sized image with libjpeg-turbo's
        // scaled DCT decode. Decoding a 12 MP capture at full resolution to feed
        // a 44 pt button costs ~68 ms and ~48 MB of RGBA; the 1/8 scale decode
        // is ~26 ms and under 1 MB.
        if let Some(scaled) = decode_jpeg_thumbnail(&bytes) {
            return Some(scaled);
        }

        let mut img = image::load_from_memory(&bytes).ok()?;
        img.apply_orientation(embedded_image_orientation(&bytes));
        let rgba = img.to_rgba8();
        let (width, height) = img.dimensions();

        Some((rgba.into_raw(), width, height))
    })
    .await
    .ok()??;

    Some(GalleryThumbnailData::new(
        latest_path,
        rgba_data,
        width,
        height,
    ))
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

    orient_thumbnail(
        pixels,
        scaled.width as u32,
        scaled.height as u32,
        embedded_image_orientation(bytes),
    )
}

fn orient_thumbnail(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    orientation: image::metadata::Orientation,
) -> Option<(Vec<u8>, u32, u32)> {
    let image = image::RgbaImage::from_raw(width, height, pixels)?;
    let mut image = image::DynamicImage::ImageRgba8(image);
    image.apply_orientation(orientation);
    let width = image.width();
    let height = image.height();
    Some((image.into_rgba8().into_raw(), width, height))
}

/// Load a thumbnail from a video file by extracting the first frame
async fn load_video_thumbnail(video_path: PathBuf) -> Option<GalleryThumbnailData> {
    debug!(path = ?video_path, "Extracting thumbnail from video");

    // Extract first frame from video in blocking task (uses GStreamer)
    tokio::task::spawn_blocking(move || {
        use crate::backends::virtual_camera::load_preview_frame;

        match load_preview_frame(&video_path) {
            Ok(frame) => {
                let width = frame.width;
                let height = frame.height;
                let rgba_data: Vec<u8> = frame.data.to_vec();

                Some(GalleryThumbnailData::new(
                    video_path, rgba_data, width, height,
                ))
            }
            Err(e) => {
                warn!(error = ?e, "Failed to extract video thumbnail");
                None
            }
        }
    })
    .await
    .ok()?
}

/// Encode RGBA data to PNG bytes
#[cfg(test)]
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

    #[test]
    fn thumbnail_clones_share_and_release_the_only_retained_pixel_buffer() {
        let pixels = vec![20, 40, 60, 255, 80, 100, 120, 255];
        let pixel_ptr = pixels.as_ptr();
        let thumbnail = GalleryThumbnailData::new("same.png".into(), pixels, 2, 1);
        let weak = Arc::downgrade(&thumbnail.0);
        let cloned = thumbnail.clone();
        assert!(Arc::ptr_eq(&thumbnail.0, &cloned.0));
        assert_eq!(thumbnail.id(), cloned.id());
        assert_eq!(cloned.rgba().as_ptr(), pixel_ptr);
        assert_eq!(cloned.rgba().len(), 8);
        assert_eq!(cloned.dimensions(), (2, 1));
        assert_eq!(Arc::strong_count(&thumbnail.0), 2);

        let replacement = GalleryThumbnailData::new("same.png".into(), vec![0; 8], 2, 1);
        assert_ne!(thumbnail.id(), replacement.id());
        drop(thumbnail);
        assert_eq!(cloned.rgba(), &[20, 40, 60, 255, 80, 100, 120, 255]);
        drop(cloned);
        assert!(weak.upgrade().is_none(), "no obsolete pixel owner survives");
        assert_eq!(replacement.rgba(), &[0; 8]);
    }

    #[tokio::test]
    async fn reloading_the_same_path_replaces_identity_and_pixels_without_mutating_clones() {
        let root = std::env::temp_dir().join(format!("camera-gallery-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("same.png");
        std::fs::write(&path, encode_rgba_to_png(&[20, 40, 60, 255], 1, 1).unwrap()).unwrap();
        let first = load_latest_thumbnail(root.clone(), root.join("missing"))
            .await
            .unwrap();
        let retained = first.clone();
        std::fs::write(
            &path,
            encode_rgba_to_png(&[180, 90, 30, 255], 1, 1).unwrap(),
        )
        .unwrap();
        let second = load_latest_thumbnail(root.clone(), root.join("missing"))
            .await
            .unwrap();
        std::fs::remove_dir_all(root).unwrap();
        assert_eq!(first.id(), retained.id());
        assert_ne!(first.id(), second.id());
        assert_eq!(second.path(), path);
        assert_eq!(second.dimensions(), first.dimensions());
        assert_eq!(retained.rgba(), &[20, 40, 60, 255]);
        assert_eq!(second.rgba(), &[180, 90, 30, 255]);
    }

    #[test]
    fn oriented_thumbnail_preserves_pixel_positions_and_aspect() {
        let pixels: Vec<u8> = (1..=6).flat_map(|v| [v, 20, 40, 255]).collect();
        let (oriented, width, height) =
            orient_thumbnail(pixels, 3, 2, image::metadata::Orientation::Rotate90).unwrap();
        assert_eq!((width, height), (2, 3));
        let expected: Vec<u8> = [4, 1, 5, 2, 6, 3]
            .into_iter()
            .flat_map(|v| [v, 20, 40, 255])
            .collect();
        assert_eq!(oriented, expected);
    }

    #[tokio::test]
    async fn latest_thumbnail_ignores_wrong_directory_formats_and_missing_directories() {
        let root = std::env::temp_dir().join(format!("camera-gallery-{}", uuid::Uuid::new_v4()));
        let photos = root.join("photos");
        let videos = root.join("videos");
        std::fs::create_dir_all(&photos).unwrap();
        std::fs::create_dir_all(&videos).unwrap();
        let png = encode_rgba_to_png(&[20, 40, 60, 255], 1, 1).unwrap();
        let expected = photos.join("new.PNG");
        let older = photos.join("old.png");
        std::fs::write(&expected, &png).unwrap();
        std::fs::write(&older, &png).unwrap();
        let old_time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1);
        std::fs::File::open(&older)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old_time))
            .unwrap();
        std::fs::write(videos.join("not-a-video.png"), &png).unwrap();
        std::fs::write(photos.join("not-an-image.txt"), b"ignored").unwrap();

        // Equal timestamps retain the photo-first traversal order. An invalid
        // video would fail decoding if it incorrectly displaced the photo.
        let tied_video = videos.join("same-time.mp4");
        std::fs::write(&tied_video, b"not selected").unwrap();
        std::fs::File::open(&tied_video)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(std::fs::metadata(&expected).unwrap().modified().unwrap()),
            )
            .unwrap();
        for video_dir in [videos, root.join("missing")] {
            let thumbnail = load_latest_thumbnail(photos.clone(), video_dir)
                .await
                .unwrap();
            assert_eq!(thumbnail.path(), expected);
            assert_eq!(thumbnail.dimensions(), (1, 1));
            assert_eq!(thumbnail.rgba(), &[20, 40, 60, 255]);
        }
        assert!(
            load_latest_thumbnail(root.join("absent"), root.join("missing"))
                .await
                .is_none()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn latest_avif_thumbnail_applies_capture_orientation() {
        use crate::pipelines::capture_metadata::{CaptureMetadata, CaptureOrientation};
        use crate::pipelines::photo::processing::ProcessedImage;
        use crate::pipelines::photo::{EncodingFormat, PhotoEncoder};

        let root = std::env::temp_dir().join(format!("camera-avif-test-{}", uuid::Uuid::new_v4()));
        let photos = root.join("photos");
        let mut encoder = PhotoEncoder::new();
        encoder.set_format(EncodingFormat::Avif);
        encoder.set_camera_metadata(CaptureMetadata {
            orientation: CaptureOrientation::Rotate270,
            ..Default::default()
        });
        let encoded = encoder
            .encode(ProcessedImage {
                image: image::RgbImage::from_pixel(65, 33, image::Rgb([200, 90, 40])),
                width: 65,
                height: 33,
            })
            .await
            .unwrap();
        let path = encoder.save(encoded, photos.clone()).await.unwrap();
        assert_eq!(path.extension().unwrap(), "avif");
        let result = load_latest_thumbnail(photos, root.join("videos")).await;
        std::fs::remove_dir_all(&root).unwrap();
        let thumbnail = result.expect("AVIF thumbnail missing");
        let (width, height) = thumbnail.dimensions();
        let rgba = thumbnail.rgba();
        assert_eq!(thumbnail.path(), path);
        assert_eq!((width, height), (33, 65));
        assert_eq!(rgba.len(), (width * height * 4) as usize);
        assert!(rgba[0].abs_diff(200) < 12);
        assert!(rgba[1].abs_diff(90) < 12);
        assert!(rgba[2].abs_diff(40) < 12);
        assert_eq!(rgba[3], 255);
    }

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

    #[test]
    fn jpeg_thumbnail_applies_embedded_orientation() {
        use little_exif::exif_tag::ExifTag;
        use little_exif::filetype::FileExtension;
        use little_exif::metadata::Metadata;

        let mut encoded = jpeg(320, 240);
        let mut metadata = Metadata::new();
        metadata.set_tag(ExifTag::Orientation(vec![6]));
        metadata
            .write_to_vec(&mut encoded, FileExtension::JPEG)
            .unwrap();

        let (_, width, height) = decode_jpeg_thumbnail(&encoded).expect("decode failed");
        assert_eq!((width, height), (240, 320));
    }
}
