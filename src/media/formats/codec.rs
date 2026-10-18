// SPDX-License-Identifier: GPL-3.0-only

//! Codec metadata and utilities for video pixel formats

/// Supported video codec/pixel format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    // ===== Compressed formats =====
    /// Motion JPEG - Frame-by-frame JPEG compression
    MJPEG,
    /// H.264/AVC - Interframe compression
    H264,
    /// H.265/HEVC - High efficiency interframe compression
    H265,

    // ===== Packed YUV 4:2:2 formats =====
    /// YUYV 4:2:2 - Packed YUV (Y0 U Y1 V byte order)
    YUYV,
    /// UYVY 4:2:2 - Packed YUV (U Y0 V Y1 byte order)
    UYVY,
    /// YUY2 4:2:2 - Same as YUYV (Microsoft naming)
    YUY2,
    /// YVYU 4:2:2 - Packed YUV (Y0 V Y1 U byte order)
    YVYU,
    /// VYUY 4:2:2 - Packed YUV (V Y0 U Y1 byte order)
    VYUY,

    // ===== Planar/Semi-planar YUV 4:2:0 formats =====
    /// NV12 4:2:0 - Semi-planar YUV (Y plane + interleaved UV)
    NV12,
    /// NV21 4:2:0 - Semi-planar YUV (Y plane + interleaved VU)
    NV21,
    /// YV12 4:2:0 - Planar YUV (Y + V + U planes)
    YV12,
    /// I420 4:2:0 - Planar YUV (Y + U + V planes)
    I420,

    // ===== RGB formats =====
    /// RGB 24-bit - Uncompressed RGB (3 bytes per pixel)
    RGB24,
    /// RGB 32-bit - Uncompressed RGBA (4 bytes per pixel)
    RGB32,
    /// BGR 24-bit - Uncompressed BGR (3 bytes per pixel)
    BGR24,
    /// BGR 32-bit - Uncompressed BGRA (4 bytes per pixel)
    BGR32,

    // ===== Bayer patterns (raw sensor data) =====
    /// Bayer GRBG - Green-Red / Blue-Green pattern (Kinect uses this)
    BayerGRBG,
    /// Bayer RGGB - Red-Green / Green-Blue pattern
    BayerRGGB,
    /// Bayer BGGR - Blue-Green / Green-Red pattern
    BayerBGGR,
    /// Bayer GBRG - Green-Blue / Red-Green pattern
    BayerGBRG,

    // ===== Depth/IR formats =====
    /// Y10B - 10-bit packed grayscale (depth sensor)
    Y10B,
    /// IR10 - 10-bit packed infrared (Kinect IR sensor)
    IR10,
    /// Y16 - 16-bit grayscale (depth in mm)
    Y16,
    /// GREY/Y8 - 8-bit grayscale
    GREY,

    // ===== Special =====
    /// Unknown/unsupported codec
    Unknown,
}

impl Codec {
    /// Parse codec from FourCC string
    pub fn from_fourcc(fourcc: &str) -> Self {
        match fourcc.to_uppercase().as_str() {
            // Compressed
            "MJPG" | "JPEG" => Self::MJPEG,
            "H264" | "AVC1" => Self::H264,
            "H265" | "HEVC" => Self::H265,

            // Packed YUV 4:2:2
            "YUYV" | "YUY2" => Self::YUYV,
            "UYVY" => Self::UYVY,
            "YVYU" => Self::YVYU,
            "VYUY" => Self::VYUY,

            // Planar/Semi-planar YUV 4:2:0
            "NV12" => Self::NV12,
            "NV21" => Self::NV21,
            "YV12" => Self::YV12,
            "I420" | "IYUV" => Self::I420,

            // RGB - various naming conventions (V4L2, GStreamer, etc.)
            "RGB" | "RGB3" | "RGB24" => Self::RGB24,
            "RGBA" | "RGBX" | "RGB4" | "RGB32" => Self::RGB32,
            "BGR" | "BGR3" | "BGR24" => Self::BGR24,
            "BGRA" | "BGRX" | "BGR4" | "BGR32" => Self::BGR32,
            // GStreamer ARGB/XRGB variants (alpha/padding in different positions)
            "ARGB" | "XRGB" | "ARGB32" | "ARGB8888" => Self::RGB32,
            "ABGR" | "XBGR" | "ABGR32" | "ABGR8888" => Self::BGR32,

            // Bayer patterns (V4L2 FourCC codes + libcamera naming)
            "GRBG" | "BA81" | "SGRBG8" => Self::BayerGRBG,
            "RGGB" | "SRGGB8" => Self::BayerRGGB,
            "BGGR" | "BA82" | "SBGGR8" => Self::BayerBGGR,
            "GBRG" | "SGBRG8" => Self::BayerGBRG,
            // Generic "BAYER" defaults to GRBG (most common, Kinect uses this)
            "BAYER" => Self::BayerGRBG,
            // libcamera Bayer format names (e.g., "BayerRGGB10LE" from caps enumeration)
            s if s.starts_with("BAYER") => {
                if s.contains("GRBG") {
                    Self::BayerGRBG
                } else if s.contains("RGGB") {
                    Self::BayerRGGB
                } else if s.contains("BGGR") {
                    Self::BayerBGGR
                } else if s.contains("GBRG") {
                    Self::BayerGBRG
                } else {
                    Self::BayerGRBG
                }
            }

            // Depth/IR/Grayscale
            "Y10B" => Self::Y10B,
            "IR10" => Self::IR10,
            "Y16" | "Y16 " => Self::Y16,
            "GREY" | "GRAY8" | "Y8" | "Y800" => Self::GREY,

            _ => Self::Unknown,
        }
    }

    /// Check if this is a raw/uncompressed format
    pub fn is_raw(&self) -> bool {
        matches!(
            self,
            Self::YUYV
                | Self::UYVY
                | Self::YUY2
                | Self::YVYU
                | Self::VYUY
                | Self::NV12
                | Self::NV21
                | Self::YV12
                | Self::I420
                | Self::RGB24
                | Self::RGB32
                | Self::BGR24
                | Self::BGR32
                | Self::BayerGRBG
                | Self::BayerRGGB
                | Self::BayerBGGR
                | Self::BayerGBRG
                | Self::Y10B
                | Self::IR10
                | Self::Y16
                | Self::GREY
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_parsing() {
        assert_eq!(Codec::from_fourcc("MJPG"), Codec::MJPEG);
        assert_eq!(Codec::from_fourcc("H264"), Codec::H264);
        assert_eq!(Codec::from_fourcc("YUYV"), Codec::YUYV);
        assert_eq!(Codec::from_fourcc("UYVY"), Codec::UYVY);
        assert_eq!(Codec::from_fourcc("GRBG"), Codec::BayerGRBG);
        assert_eq!(Codec::from_fourcc("BAYER"), Codec::BayerGRBG);
        assert_eq!(Codec::from_fourcc("UNKN"), Codec::Unknown);
    }

    #[test]
    fn test_gstreamer_format_names() {
        // GStreamer-style RGB variants
        assert_eq!(Codec::from_fourcc("RGB"), Codec::RGB24);
        assert_eq!(Codec::from_fourcc("RGBA"), Codec::RGB32);
        assert_eq!(Codec::from_fourcc("RGBX"), Codec::RGB32);
        assert_eq!(Codec::from_fourcc("BGR"), Codec::BGR24);
        assert_eq!(Codec::from_fourcc("BGRA"), Codec::BGR32);
        assert_eq!(Codec::from_fourcc("BGRX"), Codec::BGR32);
        assert_eq!(Codec::from_fourcc("ARGB"), Codec::RGB32);
        assert_eq!(Codec::from_fourcc("XRGB"), Codec::RGB32);
        assert_eq!(Codec::from_fourcc("ABGR"), Codec::BGR32);
        assert_eq!(Codec::from_fourcc("XBGR"), Codec::BGR32);
        // GStreamer grayscale
        assert_eq!(Codec::from_fourcc("GRAY8"), Codec::GREY);
    }

    #[test]
    fn test_raw_detection() {
        assert!(Codec::YUYV.is_raw());
        assert!(Codec::NV12.is_raw());
        assert!(Codec::BayerGRBG.is_raw());
        assert!(!Codec::MJPEG.is_raw());
        assert!(!Codec::H264.is_raw());
    }
}
