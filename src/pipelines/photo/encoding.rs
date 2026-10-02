// SPDX-License-Identifier: GPL-3.0-only

//! Async photo encoding pipeline
//!
//! This module handles encoding processed images to various formats:
//! - JPEG (with quality control)
//! - PNG (lossless)
//! - AVIF (with quality control)
//!
//! All encoding operations run asynchronously to avoid blocking.

use super::processing::ProcessedImage;
use crate::backends::camera::types::PixelFormat;
use crate::pipelines::capture_metadata::CaptureMetadata;
use chrono::{DateTime, FixedOffset};
use image::RgbImage;
use little_exif::exif_tag::ExifTag;
use little_exif::filetype::FileExtension;
use little_exif::metadata::Metadata;
use std::path::PathBuf;
use tracing::{debug, error, info, warn};

/// Supported encoding formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingFormat {
    /// JPEG format (lossy compression)
    Jpeg,
    /// PNG format (lossless compression)
    Png,
    /// DNG format (raw image data)
    Dng,
    /// AVIF format (lossy AV1 compression)
    Avif,
}

impl EncodingFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            EncodingFormat::Jpeg => "jpg",
            EncodingFormat::Png => "png",
            EncodingFormat::Dng => "dng",
            EncodingFormat::Avif => "avif",
        }
    }
}

impl From<crate::config::PhotoOutputFormat> for EncodingFormat {
    fn from(format: crate::config::PhotoOutputFormat) -> Self {
        match format {
            crate::config::PhotoOutputFormat::Jpeg => EncodingFormat::Jpeg,
            crate::config::PhotoOutputFormat::Png => EncodingFormat::Png,
            crate::config::PhotoOutputFormat::Dng => EncodingFormat::Dng,
            crate::config::PhotoOutputFormat::Avif => EncodingFormat::Avif,
        }
    }
}

/// Encoding quality settings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingQuality {
    /// Low quality (high compression)
    Low,
    /// Medium quality (balanced)
    Medium,
    /// High quality (low compression)
    High,
    /// Maximum quality (minimal compression)
    Maximum,
}

impl EncodingQuality {
    /// Get JPEG quality value (0-100)
    pub fn jpeg_quality(&self) -> u8 {
        match self {
            EncodingQuality::Low => 60,
            EncodingQuality::Medium => 80,
            EncodingQuality::High => 92,
            EncodingQuality::Maximum => 98,
        }
    }

    /// Chroma subsampling used when encoding JPEG.
    ///
    /// 4:2:0 is what camera JPEGs conventionally use — half the chroma
    /// resolution is invisible at normal viewing sizes and saves about a third
    /// of the file. The Maximum preset keeps full chroma for anyone who plans
    /// to edit or crop hard.
    pub fn jpeg_subsamp(&self) -> turbojpeg::Subsamp {
        match self {
            EncodingQuality::Low | EncodingQuality::Medium | EncodingQuality::High => {
                turbojpeg::Subsamp::Sub2x2
            }
            EncodingQuality::Maximum => turbojpeg::Subsamp::None,
        }
    }
}

/// Encoded image data ready for saving
pub struct EncodedImage {
    pub data: Vec<u8>,
    pub format: EncodingFormat,
    pub width: u32,
    pub height: u32,
    pub captured_at: Option<DateTime<FixedOffset>>,
}

/// Raw Bayer data for DNG encoding (bypasses post-processing)
pub struct RawBayerData {
    /// Raw packed sensor data (e.g., CSI2P 10-bit)
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Row stride in bytes (may include alignment padding)
    pub stride: u32,
    /// Bayer pixel format
    pub format: PixelFormat,
}

/// Backwards-compatible name used by the photo pipeline API.
pub type CameraMetadata = CaptureMetadata;

/// Photo encoder
pub struct PhotoEncoder {
    format: EncodingFormat,
    quality: EncodingQuality,
    camera_metadata: CameraMetadata,
}

struct MetadataFallback<'a> {
    image: &'a RgbImage,
    format: EncodingFormat,
    quality: EncodingQuality,
}

impl PhotoEncoder {
    /// Create a new encoder with JPEG format and high quality
    pub fn new() -> Self {
        Self {
            format: EncodingFormat::Jpeg,
            quality: EncodingQuality::High,
            camera_metadata: CameraMetadata::default(),
        }
    }

    /// Get the current encoding format
    pub fn format(&self) -> EncodingFormat {
        self.format
    }

    /// Set encoding format
    pub fn set_format(&mut self, format: EncodingFormat) {
        self.format = format;
    }

    /// Set encoding quality (affects JPEG and AVIF)
    pub fn set_quality(&mut self, quality: EncodingQuality) {
        self.quality = quality;
    }

    /// Set camera metadata for DNG encoding
    pub fn set_camera_metadata(&mut self, metadata: CameraMetadata) {
        self.camera_metadata = metadata;
    }

    /// Encode raw Bayer data directly as DNG (bypasses post-processing)
    ///
    /// This writes the raw sensor data into a CFA-pattern DNG file with proper
    /// metadata tags. The data is unpacked from CSI2P 10-bit to 16-bit values.
    pub async fn encode_raw(&self, raw: RawBayerData) -> Result<EncodedImage, String> {
        info!(
            width = raw.width,
            height = raw.height,
            stride = raw.stride,
            format = ?raw.format,
            "Encoding raw Bayer DNG"
        );

        let camera_metadata = self.camera_metadata.clone();
        let width = raw.width;
        let height = raw.height;

        tokio::task::spawn_blocking(move || {
            let data = Self::encode_dng_raw(&raw, &camera_metadata)?;
            debug!(size = data.len(), "Raw DNG encoding complete");
            Ok(EncodedImage {
                data,
                format: EncodingFormat::Dng,
                width,
                height,
                captured_at: camera_metadata.captured_at,
            })
        })
        .await
        .map_err(|e| format!("Raw DNG encoding task error: {}", e))?
    }

    /// Encode a processed image asynchronously
    ///
    /// This runs the encoding in a background task to avoid blocking.
    ///
    /// # Arguments
    /// * `processed` - Processed RGB image
    ///
    /// # Returns
    /// * `Ok(EncodedImage)` - Encoded image data
    /// * `Err(String)` - Error message
    pub async fn encode(&self, processed: ProcessedImage) -> Result<EncodedImage, String> {
        info!(
            width = processed.width,
            height = processed.height,
            format = ?self.format,
            "Starting encoding"
        );

        let format = self.format;
        let quality = self.quality;
        let camera_metadata = self.camera_metadata.clone();

        // Run encoding in background task (CPU-bound)
        tokio::task::spawn_blocking(move || {
            let image = processed.image;
            let data = match format {
                EncodingFormat::Jpeg => {
                    let data = Self::encode_jpeg(&image, quality)?;
                    Self::embed_standard_metadata(
                        data,
                        FileExtension::JPEG,
                        processed.width,
                        processed.height,
                        &camera_metadata,
                        MetadataFallback {
                            image: &image,
                            format,
                            quality,
                        },
                    )
                }
                EncodingFormat::Png => {
                    let data = Self::encode_png(&image)?;
                    Self::embed_standard_metadata(
                        data,
                        FileExtension::PNG {
                            as_zTXt_chunk: false,
                        },
                        processed.width,
                        processed.height,
                        &camera_metadata,
                        MetadataFallback {
                            image: &image,
                            format,
                            quality,
                        },
                    )
                }
                EncodingFormat::Dng => {
                    Self::encode_dng(&image, processed.width, processed.height, &camera_metadata)?
                }
                EncodingFormat::Avif => {
                    let metadata = build_standard_exif(
                        processed.width,
                        processed.height,
                        &camera_metadata,
                    );
                    match metadata.as_u8_vec(FileExtension::HEIF) {
                        Ok(exif) => Self::encode_avif(&image, quality, Some(exif))?,
                        Err(error) => {
                            warn!(%error, "Failed to serialize AVIF metadata; baking orientation into pixels");
                            Self::encode_oriented_fallback(
                                &image,
                                format,
                                quality,
                                camera_metadata.orientation,
                            )?
                        }
                    }
                }
            };

            debug!(size = data.len(), "Encoding complete");

            Ok(EncodedImage {
                data,
                format,
                width: processed.width,
                height: processed.height,
                captured_at: camera_metadata.captured_at,
            })
        })
        .await
        .map_err(|e| format!("Encoding task error: {}", e))?
    }

    /// Save encoded image to disk asynchronously
    ///
    /// Generates a timestamped filename and saves to the specified directory.
    ///
    /// # Arguments
    /// * `encoded` - Encoded image data
    /// * `output_dir` - Directory to save the photo
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to saved file
    /// * `Err(String)` - Error message
    pub async fn save(
        &self,
        encoded: EncodedImage,
        output_dir: PathBuf,
    ) -> Result<PathBuf, String> {
        debug!(
            output_dir = %output_dir.display(),
            format = ?encoded.format,
            size_bytes = encoded.data.len(),
            "Preparing to save photo"
        );

        // Ensure output directory exists
        if let Err(e) = tokio::fs::create_dir_all(&output_dir).await {
            error!(
                output_dir = %output_dir.display(),
                error = %e,
                "Failed to create output directory - check filesystem permissions and path validity"
            );
            return Err(format!(
                "Failed to create output directory '{}': {}",
                output_dir.display(),
                e
            ));
        }

        // Generate filename with timestamp (millisecond precision so two rapid captures don't collide).
        let timestamp = encoded
            .captured_at
            .unwrap_or_else(|| chrono::Local::now().fixed_offset())
            .format("%Y%m%d_%H%M%S_%3f");
        let filename = format!("IMG_{}.{}", timestamp, encoded.format.extension());
        let filepath = output_dir.join(&filename);

        info!(path = %filepath.display(), "Saving photo");

        // Write to disk in background task (I/O-bound)
        let filepath_clone = filepath.clone();
        let filepath_for_error = filepath.clone();
        let write_result =
            tokio::task::spawn_blocking(move || std::fs::write(&filepath_clone, &encoded.data))
                .await;

        match write_result {
            Ok(Ok(())) => {
                info!(path = %filepath.display(), "Photo saved successfully");
                Ok(filepath)
            }
            Ok(Err(io_err)) => {
                error!(
                    path = %filepath_for_error.display(),
                    error = %io_err,
                    "Failed to write photo to disk - check disk space and permissions"
                );
                Err(format!(
                    "Failed to save photo to '{}': {}",
                    filepath_for_error.display(),
                    io_err
                ))
            }
            Err(join_err) => {
                error!(
                    path = %filepath_for_error.display(),
                    error = %join_err,
                    "Save task panicked or was cancelled"
                );
                Err(format!("Save task error: {}", join_err))
            }
        }
    }

    /// Encode image as JPEG
    ///
    /// Uses libjpeg-turbo (already linked for MJPEG viewfinder decoding) rather
    /// than the pure-Rust encoder in `image`. On a 12 MP capture at quality 92
    /// that is ~117 ms instead of ~269 ms, and the 4:2:0 chroma subsampling used
    /// for everything below the Maximum preset also cuts the file roughly a third
    /// (3.4 MB vs 5.2 MB) — `image` always writes 4:4:4.
    fn encode_jpeg(image: &RgbImage, quality: EncodingQuality) -> Result<Vec<u8>, String> {
        match Self::encode_jpeg_turbo(image, quality) {
            Ok(data) => Ok(data),
            Err(e) => {
                // Keep the pure-Rust encoder as a safety net so a turbojpeg
                // failure never costs the user a capture.
                warn!(error = %e, "libjpeg-turbo encoding failed, falling back to image crate");
                Self::encode_jpeg_image_crate(image, quality)
            }
        }
    }

    fn encode_jpeg_turbo(image: &RgbImage, quality: EncodingQuality) -> Result<Vec<u8>, String> {
        let mut compressor =
            turbojpeg::Compressor::new().map_err(|e| format!("turbojpeg init failed: {e}"))?;

        compressor
            .set_quality(quality.jpeg_quality() as i32)
            .map_err(|e| format!("turbojpeg quality: {e}"))?;
        compressor
            .set_subsamp(quality.jpeg_subsamp())
            .map_err(|e| format!("turbojpeg subsampling: {e}"))?;

        let src = turbojpeg::Image {
            pixels: image.as_raw().as_slice(),
            width: image.width() as usize,
            pitch: (image.width() * 3) as usize,
            height: image.height() as usize,
            format: turbojpeg::PixelFormat::RGB,
        };

        compressor
            .compress_to_vec(src)
            .map_err(|e| format!("JPEG encoding failed: {e}"))
    }

    fn encode_jpeg_image_crate(
        image: &RgbImage,
        quality: EncodingQuality,
    ) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buffer);

        // Create JPEG encoder with quality setting
        let mut encoder =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality.jpeg_quality());

        encoder
            .encode(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|e| format!("JPEG encoding failed: {}", e))?;

        Ok(buffer)
    }

    /// Encode image as PNG
    fn encode_png(image: &RgbImage) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        image
            .write_to(
                &mut std::io::Cursor::new(&mut buffer),
                image::ImageFormat::Png,
            )
            .map_err(|e| format!("PNG encoding failed: {}", e))?;

        Ok(buffer)
    }

    /// Encode AVIF with a bounded worker pool, leaving two CPUs for the UI.
    fn encode_avif(
        image: &RgbImage,
        quality: EncodingQuality,
        exif: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, String> {
        use image::ImageEncoder;

        let available = std::thread::available_parallelism().map_or(1, usize::from);
        let threads = avif_thread_count(available);
        let quality = match quality {
            EncodingQuality::Low => 60,
            EncodingQuality::Medium => 70,
            EncodingQuality::High => 80,
            EncodingQuality::Maximum => 90,
        };
        let mut buffer = Vec::new();
        let mut encoder =
            image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut buffer, 8, quality)
                .with_num_threads(Some(threads));
        if let Some(exif) = exif {
            // little_exif's HEIF serialization includes the TIFF offset and
            // Exif header; the AVIF encoder embeds this item unchanged.
            encoder
                .set_exif_metadata(exif)
                .map_err(|error| format!("AVIF metadata failed: {error}"))?;
        }
        encoder
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|error| format!("AVIF encoding failed: {error}"))?;
        Ok(buffer)
    }

    /// Embed standard EXIF fields in an encoded JPEG or PNG.
    ///
    /// Metadata is written into a copy so a metadata-writer failure can never
    /// corrupt or discard the successfully encoded image.
    fn embed_standard_metadata(
        data: Vec<u8>,
        file_type: FileExtension,
        width: u32,
        height: u32,
        camera_metadata: &CameraMetadata,
        fallback: MetadataFallback<'_>,
    ) -> Vec<u8> {
        let metadata = build_standard_exif(width, height, camera_metadata);
        let mut tagged = data.clone();
        let write_result = match file_type {
            FileExtension::PNG { .. } => Self::write_png_exif_chunk(&mut tagged, &metadata),
            other => metadata.write_to_vec(&mut tagged, other),
        };
        match write_result {
            Ok(()) => tagged,
            Err(error) => {
                warn!(%error, "Failed to embed photo metadata; baking orientation into pixels");
                match Self::encode_oriented_fallback(
                    fallback.image,
                    fallback.format,
                    fallback.quality,
                    camera_metadata.orientation,
                ) {
                    Ok(oriented) => oriented,
                    Err(fallback_error) => {
                        warn!(
                            error = %fallback_error,
                            "Failed to bake fallback orientation; keeping encoded image"
                        );
                        data
                    }
                }
            }
        }
    }

    fn write_png_exif_chunk(data: &mut Vec<u8>, metadata: &Metadata) -> std::io::Result<()> {
        use std::io::{Error, ErrorKind};

        const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
        const EXIF_HEADER_LEN: usize = 10;

        if data.len() < 24 || !data.starts_with(PNG_SIGNATURE) || &data[12..16] != b"IHDR" {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid PNG structure"));
        }

        let app1 = metadata.as_u8_vec(FileExtension::JPEG)?;
        if app1.len() <= EXIF_HEADER_LEN
            || !app1.starts_with(&[0xff, 0xe1])
            || &app1[4..10] != b"Exif\0\0"
        {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Invalid encoded EXIF payload",
            ));
        }
        let tiff = &app1[EXIF_HEADER_LEN..];
        let length = u32::try_from(tiff.len())
            .map_err(|_| Error::new(ErrorKind::InvalidData, "EXIF payload too large"))?;

        let mut chunk = Vec::with_capacity(12 + tiff.len());
        chunk.extend_from_slice(&length.to_be_bytes());
        chunk.extend_from_slice(b"eXIf");
        chunk.extend_from_slice(tiff);
        let mut crc = crc32fast::Hasher::new();
        crc.update(b"eXIf");
        crc.update(tiff);
        chunk.extend_from_slice(&crc.finalize().to_be_bytes());

        let ihdr_len = u32::from_be_bytes(data[8..12].try_into().unwrap()) as usize;
        let insert_at = 8usize
            .checked_add(12)
            .and_then(|value| value.checked_add(ihdr_len))
            .filter(|&value| value <= data.len())
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "Invalid PNG IHDR length"))?;
        data.splice(insert_at..insert_at, chunk);
        Ok(())
    }

    fn encode_oriented_fallback(
        image: &RgbImage,
        format: EncodingFormat,
        quality: EncodingQuality,
        orientation: crate::pipelines::capture_metadata::CaptureOrientation,
    ) -> Result<Vec<u8>, String> {
        let mut oriented = image::DynamicImage::ImageRgb8(image.clone());
        let orientation = image::metadata::Orientation::from_exif(orientation.exif_value() as u8)
            .ok_or_else(|| "Invalid capture orientation".to_string())?;
        oriented.apply_orientation(orientation);
        let oriented = oriented.into_rgb8();

        match format {
            EncodingFormat::Jpeg => Self::encode_jpeg(&oriented, quality),
            EncodingFormat::Png => Self::encode_png(&oriented),
            EncodingFormat::Avif => Self::encode_avif(&oriented, quality, None),
            EncodingFormat::Dng => Err("DNG metadata is written during encoding".into()),
        }
    }

    /// Encode image as DNG (Digital Negative raw format)
    ///
    /// Creates a simple linear DNG file with RGB data stored as strips.
    /// This preserves the image data in a raw-compatible format for later processing.
    fn encode_dng(
        image: &RgbImage,
        width: u32,
        height: u32,
        camera_metadata: &CameraMetadata,
    ) -> Result<Vec<u8>, String> {
        use dng::ifd::{IfdValue, Offsets};
        use dng::tags::ifd as tiff_tags;
        use dng::{DngWriter, FileType};
        use std::io::Cursor;
        use std::sync::Arc;

        let raw_data = image.as_raw().clone();
        let raw_data_len = raw_data.len() as u32;

        let version = env!("CARGO_PKG_VERSION");
        let mut ifd = set_common_dng_tags(width, height, camera_metadata, version);

        // RGB-specific tags
        ifd.insert(
            tiff_tags::BitsPerSample,
            IfdValue::List(vec![
                IfdValue::Short(8),
                IfdValue::Short(8),
                IfdValue::Short(8),
            ]),
        );
        ifd.insert(tiff_tags::PhotometricInterpretation, IfdValue::Short(2)); // RGB
        ifd.insert(tiff_tags::SamplesPerPixel, IfdValue::Short(3)); // RGB = 3 samples

        // Strip data
        let offsets: Arc<dyn Offsets + Send + Sync> = Arc::new(DngOffsets { data: raw_data });
        ifd.insert(tiff_tags::StripOffsets, IfdValue::Offsets(offsets));
        ifd.insert(tiff_tags::StripByteCounts, IfdValue::Long(raw_data_len));

        // Write the DNG file to a buffer
        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);

        DngWriter::write_dng(cursor, true, FileType::Dng, vec![ifd])
            .map_err(|e| format!("DNG encoding failed: {:?}", e))?;

        Ok(buffer)
    }

    /// Encode raw Bayer sensor data as a CFA-pattern DNG
    ///
    /// Unpacks CSI2P 10-bit packed data to 16-bit values and writes a proper
    /// DNG file with CFA metadata. This preserves the original sensor data
    /// for later raw processing in tools like RawTherapee or darktable.
    fn encode_dng_raw(
        raw: &RawBayerData,
        camera_metadata: &CameraMetadata,
    ) -> Result<Vec<u8>, String> {
        use dng::ifd::{IfdValue, Offsets};
        use dng::tags::ifd as tiff_tags;
        use dng::{DngWriter, FileType};
        use std::io::Cursor;
        use std::sync::Arc;

        let width = raw.width;
        let height = raw.height;
        let stride = raw.stride;

        // Determine bit depth from stride vs width ratio
        let min_stride_10 = (width * 5).div_ceil(4);
        let min_stride_12 = (width * 3).div_ceil(2);
        let is_packed = stride > width;
        let bit_depth: u32 = if is_packed {
            if stride >= min_stride_10 && stride < min_stride_12 {
                10
            } else if stride >= min_stride_12 {
                12
            } else {
                8
            }
        } else {
            8
        };

        info!(
            width,
            height, stride, bit_depth, is_packed, "Unpacking raw Bayer for DNG"
        );

        // Unpack to 16-bit values
        let pixel_data_16 = if bit_depth == 10 && is_packed {
            unpack_csi2p_10bit_to_16bit(&raw.data, width, height, stride)
        } else {
            // 8-bit: promote each byte to 16-bit and bit-replicate so the
            // value occupies the full 16-bit range (0..=255 → 0..=65535).
            // Without the shift, readers that map sample/65535 (instead of
            // sample/WhiteLevel) render the image extremely dark.
            let mut out = Vec::with_capacity((width * height * 2) as usize);
            for row in 0..height {
                let row_start = (row * stride) as usize;
                for col in 0..width {
                    let byte = raw.data[row_start + col as usize] as u16;
                    let val = (byte << 8) | byte;
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
            out
        };

        let raw_data_len = pixel_data_16.len() as u32;
        let white_level: u32 = if bit_depth == 10 {
            1023
        } else if bit_depth == 12 {
            4095
        } else {
            65535
        };

        // Get CFA pattern bytes: 0=R, 1=G, 2=B
        let cfa_pattern: Vec<u8> = match raw.format {
            PixelFormat::BayerRGGB => vec![0, 1, 1, 2], // R G / G B
            PixelFormat::BayerBGGR => vec![2, 1, 1, 0], // B G / G R
            PixelFormat::BayerGRBG => vec![1, 0, 2, 1], // G R / B G
            PixelFormat::BayerGBRG => vec![1, 2, 0, 1], // G B / R G
            _ => return Err(format!("Not a Bayer format: {:?}", raw.format)),
        };

        let version = env!("CARGO_PKG_VERSION");
        let mut ifd = set_common_dng_tags(width, height, camera_metadata, version);

        // DNG version 1.4.0.0
        ifd.insert(
            tiff_tags::DNGVersion,
            IfdValue::List(vec![
                IfdValue::Byte(1),
                IfdValue::Byte(4),
                IfdValue::Byte(0),
                IfdValue::Byte(0),
            ]),
        );
        ifd.insert(
            tiff_tags::DNGBackwardVersion,
            IfdValue::List(vec![
                IfdValue::Byte(1),
                IfdValue::Byte(1),
                IfdValue::Byte(0),
                IfdValue::Byte(0),
            ]),
        );

        // CFA-specific tags
        ifd.insert(tiff_tags::BitsPerSample, IfdValue::Short(16));
        ifd.insert(tiff_tags::PhotometricInterpretation, IfdValue::Short(32803)); // CFA
        ifd.insert(tiff_tags::SamplesPerPixel, IfdValue::Short(1));

        // CFA pattern
        ifd.insert(
            tiff_tags::CFARepeatPatternDim,
            IfdValue::List(vec![IfdValue::Short(2), IfdValue::Short(2)]),
        );
        ifd.insert(
            tiff_tags::CFAPattern,
            IfdValue::List(cfa_pattern.iter().map(|&b| IfdValue::Byte(b)).collect()),
        );
        ifd.insert(
            tiff_tags::CFAPlaneColor,
            IfdValue::List(vec![
                IfdValue::Byte(0),
                IfdValue::Byte(1),
                IfdValue::Byte(2),
            ]),
        );
        ifd.insert(tiff_tags::CFALayout, IfdValue::Short(1)); // Rectangular

        // Black/White levels
        ifd.insert(tiff_tags::BlackLevel, IfdValue::Long(0));
        ifd.insert(tiff_tags::WhiteLevel, IfdValue::Long(white_level));

        // Strip data
        let offsets: Arc<dyn Offsets + Send + Sync> = Arc::new(DngOffsets {
            data: pixel_data_16,
        });
        ifd.insert(tiff_tags::StripOffsets, IfdValue::Offsets(offsets));
        ifd.insert(tiff_tags::StripByteCounts, IfdValue::Long(raw_data_len));

        let mut buffer = Vec::new();
        let cursor = Cursor::new(&mut buffer);
        DngWriter::write_dng(cursor, true, FileType::Dng, vec![ifd])
            .map_err(|e| format!("Raw DNG encoding failed: {:?}", e))?;

        info!(size = buffer.len(), "Raw CFA DNG written");
        Ok(buffer)
    }
}

impl Default for PhotoEncoder {
    fn default() -> Self {
        Self::new()
    }
}

fn avif_thread_count(available: usize) -> usize {
    available.saturating_sub(2).max(1)
}

fn build_standard_exif(width: u32, height: u32, camera_metadata: &CameraMetadata) -> Metadata {
    let mut metadata = Metadata::new();

    metadata.set_tag(ExifTag::ExifImageWidth(vec![width]));
    metadata.set_tag(ExifTag::ExifImageHeight(vec![height]));
    metadata.set_tag(ExifTag::Orientation(vec![
        camera_metadata.orientation.exif_value(),
    ]));

    metadata.set_tag(ExifTag::Software(camera_metadata.software()));

    if let Some(captured_at) = camera_metadata.captured_at {
        let timestamp = captured_at.format("%Y:%m:%d %H:%M:%S").to_string();
        let offset = captured_at.format("%:z").to_string();
        let subseconds = captured_at.format("%3f").to_string();

        metadata.set_tag(ExifTag::ModifyDate(timestamp.clone()));
        metadata.set_tag(ExifTag::DateTimeOriginal(timestamp.clone()));
        metadata.set_tag(ExifTag::CreateDate(timestamp));
        metadata.set_tag(ExifTag::OffsetTime(offset.clone()));
        metadata.set_tag(ExifTag::OffsetTimeOriginal(offset.clone()));
        metadata.set_tag(ExifTag::OffsetTimeDigitized(offset));
        metadata.set_tag(ExifTag::SubSecTime(subseconds.clone()));
        metadata.set_tag(ExifTag::SubSecTimeOriginal(subseconds.clone()));
        metadata.set_tag(ExifTag::SubSecTimeDigitized(subseconds));
    }

    if let Some(make) = &camera_metadata.device_make {
        metadata.set_tag(ExifTag::Make(make.clone()));
    }
    if let Some(model) = &camera_metadata.device_model {
        metadata.set_tag(ExifTag::Model(model.clone()));
    }

    if let Some(exposure_time) = camera_metadata.exposure_time
        && exposure_time.is_finite()
        && exposure_time > 0.0
    {
        metadata.set_tag(ExifTag::ExposureTime(vec![exposure_time.into()]));
    }
    if let Some(iso) = camera_metadata.iso {
        metadata.set_tag(ExifTag::ISO(vec![iso.min(u16::MAX as u32) as u16]));
    }

    if let Some(description) = camera_metadata.description() {
        metadata.set_tag(ExifTag::ImageDescription(description));
    }

    metadata
}

/// Strip data offsets for DNG encoding (used for both RGB and raw CFA data)
struct DngOffsets {
    data: Vec<u8>,
}

impl dng::ifd::Offsets for DngOffsets {
    fn size(&self) -> u32 {
        self.data.len() as u32
    }

    fn write(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        writer.write_all(&self.data)
    }
}

/// Set TIFF/EXIF tags common to both RGB and raw CFA DNG files
///
/// Sets: ImageWidth, ImageLength, Compression, RowsPerStrip, PlanarConfiguration,
/// Software (with optional gain info), Make/Model, ExposureTime, and ISOSpeedRatings.
fn set_common_dng_tags(
    width: u32,
    height: u32,
    camera_metadata: &CameraMetadata,
    _version: &str,
) -> dng::ifd::Ifd {
    use dng::ifd::{Ifd, IfdValue};
    use dng::tags::ifd as tiff_tags;

    let mut ifd = Ifd::default();

    // Required TIFF tags
    ifd.insert(tiff_tags::ImageWidth, IfdValue::Long(width));
    ifd.insert(tiff_tags::ImageLength, IfdValue::Long(height));
    ifd.insert(tiff_tags::Compression, IfdValue::Short(1)); // No compression
    ifd.insert(tiff_tags::RowsPerStrip, IfdValue::Long(height)); // One strip
    ifd.insert(tiff_tags::PlanarConfiguration, IfdValue::Short(1)); // Chunky
    ifd.insert(
        tiff_tags::Orientation,
        IfdValue::Short(camera_metadata.orientation.exif_value()),
    );

    ifd.insert(
        tiff_tags::Software,
        IfdValue::Ascii(camera_metadata.software()),
    );

    if let Some(make) = &camera_metadata.device_make {
        ifd.insert(tiff_tags::Make, IfdValue::Ascii(make.clone()));
    }
    if let Some(model) = &camera_metadata.device_model {
        ifd.insert(tiff_tags::Model, IfdValue::Ascii(model.clone()));
    }
    if let Some(sensor) = camera_metadata.sensor_identity() {
        ifd.insert(
            tiff_tags::UniqueCameraModel,
            IfdValue::Ascii(sensor.to_string()),
        );
    }
    if let Some(description) = camera_metadata.description() {
        ifd.insert(tiff_tags::ImageDescription, IfdValue::Ascii(description));
    }
    if let Some(captured_at) = camera_metadata.captured_at.as_ref() {
        let timestamp = captured_at.format("%Y:%m:%d %H:%M:%S").to_string();
        ifd.insert(tiff_tags::DateTime, IfdValue::Ascii(timestamp.clone()));
        ifd.insert(tiff_tags::DateTimeOriginal, IfdValue::Ascii(timestamp));
    }

    // Exposure metadata (EXIF tags)
    if let Some(exposure_time) = camera_metadata.exposure_time {
        // Convert to rational: e.g., 0.033333 -> 1/30
        // Use microsecond precision for the rational representation
        let numerator = (exposure_time * 1_000_000.0).round() as u32;
        let denominator = 1_000_000u32;
        // Simplify the fraction by finding GCD
        let g = gcd(numerator, denominator);
        ifd.insert(
            tiff_tags::ExposureTime,
            IfdValue::Rational(numerator / g, denominator / g),
        );
    }

    if let Some(iso) = camera_metadata.iso {
        ifd.insert(
            tiff_tags::ISOSpeedRatings,
            IfdValue::Short(iso.min(65535) as u16),
        );
    }

    ifd
}

/// Unpack CSI-2 10-bit packed Bayer data to 16-bit little-endian values
///
/// CSI2P packing: every 5 bytes contain 4 pixels.
/// Bytes 0-3 are high 8 bits of pixels 0-3.
/// Byte 4 contains low 2 bits: [p0_lo:2 | p1_lo:2 | p2_lo:2 | p3_lo:2]
fn unpack_csi2p_10bit_to_16bit(packed: &[u8], width: u32, height: u32, stride: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 2) as usize);
    let groups_per_row = width / 4;

    for row in 0..height {
        let row_start = (row * stride) as usize;
        for group in 0..groups_per_row {
            let base = row_start + (group * 5) as usize;
            if base + 4 >= packed.len() {
                // Pad remaining pixels with zeros
                for _ in 0..(width - group * 4) {
                    out.extend_from_slice(&0u16.to_le_bytes());
                }
                break;
            }
            let lo = packed[base + 4];
            for i in 0..4u8 {
                let hi8 = packed[base + i as usize] as u16;
                let lo2 = ((lo >> (i * 2)) & 0x03) as u16;
                let val = (hi8 << 2) | lo2;
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
        // Handle remaining pixels if width is not divisible by 4
        let remaining = width % 4;
        if remaining > 0 {
            let base = row_start + (groups_per_row * 5) as usize;
            if base + remaining as usize <= packed.len() {
                let lo = if base + 4 < packed.len() {
                    packed[base + 4]
                } else {
                    0
                };
                for i in 0..remaining {
                    let hi8 = packed[base + i as usize] as u16;
                    let lo2 = ((lo >> (i as u8 * 2)) & 0x03) as u16;
                    let val = (hi8 << 2) | lo2;
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
    }

    out
}

/// Calculate greatest common divisor using Euclidean algorithm
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1) // Avoid division by zero
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Timelike};

    #[test]
    fn avif_workers_leave_two_cpus_with_at_least_one_worker() {
        for (available, expected) in [(0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (16, 14)] {
            assert_eq!(avif_thread_count(available), expected);
        }
    }

    #[test]
    fn avif_metadata_fallback_bakes_orientation_into_pixels() {
        let source = test_image(65, 33);
        let encoded = PhotoEncoder::encode_oriented_fallback(
            &source,
            EncodingFormat::Avif,
            EncodingQuality::High,
            crate::pipelines::capture_metadata::CaptureOrientation::Rotate270,
        )
        .expect("fallback encoding failed");
        let decoded = image::load_from_memory(&encoded).expect("fallback must decode");
        assert_eq!((decoded.width(), decoded.height()), (33, 65));
    }

    #[test]
    fn avif_quality_presets_round_trip_odd_sized_images() {
        for quality in [
            EncodingQuality::Low,
            EncodingQuality::Medium,
            EncodingQuality::High,
            EncodingQuality::Maximum,
        ] {
            let encoded = PhotoEncoder::encode_avif(&test_image(65, 33), quality, None)
                .expect("encoding failed");
            let decoded = image::load_from_memory(&encoded)
                .expect("decoding failed")
                .to_rgb8();
            assert_eq!(decoded.dimensions(), (65, 33));
            let image::Rgb([r, g, b]) = *decoded.get_pixel(32, 16);
            assert!(r.abs_diff(200) < 12 && g.abs_diff(90) < 12 && b.abs_diff(40) < 12);
        }
    }

    #[tokio::test]
    async fn avif_export_round_trips_with_capture_metadata() {
        let format: crate::config::PhotoOutputFormat =
            serde_json::from_str("\"Avif\"").expect("AVIF must be a selectable format");
        assert!(crate::config::PhotoOutputFormat::ALL.contains(&format));
        assert_eq!(format.extension(), "avif");
        let mut encoder = PhotoEncoder::new();
        encoder.set_format(format.into());
        let captured_at = FixedOffset::east_opt(7200)
            .unwrap()
            .with_ymd_and_hms(2026, 9, 28, 14, 5, 6)
            .unwrap();
        encoder.set_camera_metadata(CameraMetadata {
            captured_at: Some(captured_at),
            device_make: Some("Google".into()),
            device_model: Some("Pixel 3a".into()),
            orientation: crate::pipelines::capture_metadata::CaptureOrientation::Rotate270,
            ..Default::default()
        });
        let encoded = encoder
            .encode(ProcessedImage {
                image: test_image(65, 33),
                width: 65,
                height: 33,
            })
            .await
            .expect("AVIF encoding failed");
        assert_eq!(encoded.format.extension(), "avif");
        assert_eq!(encoded.captured_at, Some(captured_at));
        let decoded = image::load_from_memory(&encoded.data)
            .expect("gallery must be able to decode AVIF")
            .to_rgb8();
        assert_eq!(decoded.dimensions(), (65, 33));
        let image::Rgb([r, g, b]) = *decoded.get_pixel(32, 16);
        assert!(r.abs_diff(200) < 12 && g.abs_diff(90) < 12 && b.abs_diff(40) < 12);
        let metadata = Metadata::new_from_vec(&encoded.data, FileExtension::HEIF)
            .expect("AVIF EXIF must be readable");
        assert_eq!(
            tag_string(&metadata, ExifTag::Make(String::new())),
            "Google"
        );
        assert_eq!(
            tag_string(&metadata, ExifTag::Model(String::new())),
            "Pixel 3a"
        );
        assert_eq!(
            tag_string(&metadata, ExifTag::DateTimeOriginal(String::new())),
            "2026:09:28 14:05:06"
        );
        assert!(matches!(
            metadata.get_tag(&ExifTag::Orientation(Vec::new())).next(),
            Some(ExifTag::Orientation(value)) if value == &vec![8]
        ));
    }

    #[test]
    fn test_format_extensions() {
        assert_eq!(EncodingFormat::Jpeg.extension(), "jpg");
        assert_eq!(EncodingFormat::Png.extension(), "png");
        assert_eq!(EncodingFormat::Dng.extension(), "dng");
    }

    #[test]
    fn test_jpeg_quality_values() {
        assert_eq!(EncodingQuality::Low.jpeg_quality(), 60);
        assert_eq!(EncodingQuality::Medium.jpeg_quality(), 80);
        assert_eq!(EncodingQuality::High.jpeg_quality(), 92);
        assert_eq!(EncodingQuality::Maximum.jpeg_quality(), 98);
    }

    /// A flat-colour test image, sized so the 4:2:0 MCU grid does not divide it
    /// evenly (odd width and height exercise the chroma padding path).
    fn test_image(w: u32, h: u32) -> RgbImage {
        RgbImage::from_pixel(w, h, image::Rgb([200, 90, 40]))
    }

    #[test]
    fn jpeg_encoding_round_trips_through_turbojpeg() {
        for quality in [
            EncodingQuality::Low,
            EncodingQuality::Medium,
            EncodingQuality::High,
            EncodingQuality::Maximum,
        ] {
            let source = test_image(65, 33);
            let data = PhotoEncoder::encode_jpeg(&source, quality).expect("encoding failed");

            assert_eq!(&data[..2], &[0xFF, 0xD8], "missing JPEG SOI marker");

            let decoded = image::load_from_memory(&data)
                .expect("re-decoding failed")
                .to_rgb8();
            assert_eq!(decoded.dimensions(), (65, 33));

            // Flat colour survives any quality level within a small margin.
            let image::Rgb([r, g, b]) = *decoded.get_pixel(32, 16);
            assert!(
                r.abs_diff(200) < 12 && g.abs_diff(90) < 12 && b.abs_diff(40) < 12,
                "colour drifted at {quality:?}: {r},{g},{b}"
            );
        }
    }

    #[test]
    fn maximum_quality_keeps_full_chroma_resolution() {
        assert_eq!(
            EncodingQuality::Maximum.jpeg_subsamp(),
            turbojpeg::Subsamp::None
        );
        assert_eq!(
            EncodingQuality::High.jpeg_subsamp(),
            turbojpeg::Subsamp::Sub2x2
        );

        let source = test_image(64, 64);
        let data = PhotoEncoder::encode_jpeg(&source, EncodingQuality::Maximum).unwrap();
        let header = turbojpeg::read_header(&data).unwrap();
        assert_eq!(header.subsamp, turbojpeg::Subsamp::None);
    }

    #[test]
    fn subsampled_presets_produce_smaller_files_than_full_chroma() {
        // Gradient content so the chroma planes actually carry information.
        let source = RgbImage::from_fn(512, 512, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });

        let high = PhotoEncoder::encode_jpeg(&source, EncodingQuality::High).unwrap();
        let maximum = PhotoEncoder::encode_jpeg(&source, EncodingQuality::Maximum).unwrap();

        assert!(
            high.len() < maximum.len(),
            "expected 4:2:0 High ({} bytes) to be smaller than 4:4:4 Maximum ({} bytes)",
            high.len(),
            maximum.len()
        );
    }

    fn tag_string(metadata: &Metadata, prototype: ExifTag) -> String {
        let tag = metadata
            .get_tag(&prototype)
            .next()
            .expect("expected EXIF tag");
        match tag {
            ExifTag::DateTimeOriginal(value)
            | ExifTag::OffsetTimeOriginal(value)
            | ExifTag::SubSecTimeOriginal(value)
            | ExifTag::Software(value)
            | ExifTag::Make(value)
            | ExifTag::Model(value)
            | ExifTag::LensModel(value)
            | ExifTag::ImageDescription(value) => value.clone(),
            other => panic!("unexpected EXIF tag: {other:?}"),
        }
    }

    #[test]
    fn jpeg_metadata_uses_standard_capture_fields() {
        let captured_at = FixedOffset::east_opt(2 * 60 * 60)
            .unwrap()
            .with_ymd_and_hms(2026, 9, 28, 14, 5, 6)
            .unwrap()
            .with_nanosecond(123_000_000)
            .unwrap();
        let camera_metadata = CameraMetadata {
            camera_name: Some("Back Camera".into()),
            camera_driver: Some("libcamera".into()),
            sensor_model: Some("sony,imx363".into()),
            camera_location: Some("back".into()),
            pipeline_handler: Some("simple".into()),
            libcamera_version: Some("0.5.2".into()),
            device_make: Some("Google".into()),
            device_model: Some("Pixel 3a".into()),
            captured_at: Some(captured_at),
            orientation: crate::pipelines::capture_metadata::CaptureOrientation::Rotate270,
            exposure_time: Some(1.0 / 30.0),
            iso: Some(200),
            gain: Some(4),
        };

        let source = test_image(65, 33);
        let encoded =
            PhotoEncoder::encode_jpeg(&source, EncodingQuality::High).expect("encoding failed");
        let encoded = PhotoEncoder::embed_standard_metadata(
            encoded,
            FileExtension::JPEG,
            65,
            33,
            &camera_metadata,
            MetadataFallback {
                image: &source,
                format: EncodingFormat::Jpeg,
                quality: EncodingQuality::High,
            },
        );
        let metadata = Metadata::new_from_vec(&encoded, FileExtension::JPEG)
            .expect("metadata should be readable");

        assert_eq!(
            tag_string(&metadata, ExifTag::DateTimeOriginal(String::new())),
            "2026:09:28 14:05:06"
        );
        assert_eq!(
            tag_string(&metadata, ExifTag::OffsetTimeOriginal(String::new())),
            "+02:00"
        );
        assert_eq!(
            tag_string(&metadata, ExifTag::SubSecTimeOriginal(String::new())),
            "123"
        );
        assert_eq!(
            tag_string(&metadata, ExifTag::Make(String::new())),
            "Google"
        );
        assert_eq!(
            tag_string(&metadata, ExifTag::Model(String::new())),
            "Pixel 3a"
        );
        assert!(
            metadata
                .get_tag(&ExifTag::LensModel(String::new()))
                .next()
                .is_none(),
            "sensor identity must not be mislabeled as a lens"
        );
        assert_eq!(
            tag_string(&metadata, ExifTag::Software(String::new())),
            format!("Camera {}; libcamera 0.5.2", env!("CARGO_PKG_VERSION"))
        );
        assert!(matches!(
            metadata
                .get_tag(&ExifTag::Orientation(Vec::new()))
                .next(),
            Some(ExifTag::Orientation(value)) if value == &vec![8]
        ));
        assert!(
            tag_string(&metadata, ExifTag::ImageDescription(String::new()))
                .contains("libcamera pipeline: simple")
        );
    }

    #[test]
    fn metadata_failure_fallback_bakes_orientation_into_pixels() {
        let source = test_image(65, 33);
        let encoded = PhotoEncoder::encode_oriented_fallback(
            &source,
            EncodingFormat::Jpeg,
            EncodingQuality::High,
            crate::pipelines::capture_metadata::CaptureOrientation::Rotate270,
        )
        .expect("fallback encoding failed");
        let decoded = image::load_from_memory(&encoded).expect("fallback image should decode");
        assert_eq!((decoded.width(), decoded.height()), (33, 65));
    }

    #[test]
    fn png_metadata_uses_standard_exif_chunk() {
        let source = test_image(65, 33);
        let camera_metadata = CameraMetadata {
            orientation: crate::pipelines::capture_metadata::CaptureOrientation::Rotate270,
            ..Default::default()
        };
        let encoded = PhotoEncoder::encode_png(&source).expect("encoding failed");
        let encoded = PhotoEncoder::embed_standard_metadata(
            encoded,
            FileExtension::PNG {
                as_zTXt_chunk: true,
            },
            65,
            33,
            &camera_metadata,
            MetadataFallback {
                image: &source,
                format: EncodingFormat::Png,
                quality: EncodingQuality::High,
            },
        );

        assert!(encoded.windows(4).any(|chunk| chunk == b"eXIf"));
        assert!(!encoded.windows(4).any(|chunk| chunk == b"zTXt"));
    }
}
