// SPDX-License-Identifier: GPL-3.0-only

//! Shared V4L2 utility functions
//!
//! This module provides common V4L2 operations used by the camera backend,
//! including device enumeration, driver queries, and sensor detection.

use super::types::DeviceInfo;
use std::os::unix::io::{AsRawFd, RawFd};
use tracing::debug;

/// VIDIOC_QUERYCAP ioctl number
const VIDIOC_QUERYCAP: libc::c_ulong = 0x80685600;

/// V4L2 capability flag for single-planar video capture
const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x00000001;

/// V4L2 capability structure for VIDIOC_QUERYCAP ioctl
#[repr(C)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

/// Query V4L2 capabilities for an open file descriptor.
///
/// Issues the `VIDIOC_QUERYCAP` ioctl and returns the capability struct,
/// or `None` if the ioctl fails.
fn query_v4l2_cap(fd: RawFd) -> Option<V4l2Capability> {
    let mut cap: V4l2Capability = unsafe { std::mem::zeroed() };
    let result = unsafe { libc::ioctl(fd, VIDIOC_QUERYCAP as _, &mut cap as *mut V4l2Capability) };
    if result < 0 { None } else { Some(cap) }
}

/// Get V4L2 driver name using ioctl
///
/// Opens the device and queries its capabilities to get the driver name.
/// Returns None if the device cannot be opened or the ioctl fails.
pub fn get_v4l2_driver(device_path: &str) -> Option<String> {
    let file = std::fs::File::open(device_path).ok()?;
    let cap = query_v4l2_cap(file.as_raw_fd())?;

    // Find null terminator or use full length
    let len = cap.driver.iter().position(|&c| c == 0).unwrap_or(16);
    let driver = String::from_utf8_lossy(&cap.driver[..len]).to_string();

    debug!(device_path, driver = %driver, "Got V4L2 driver name");
    Some(driver)
}

/// Build DeviceInfo from V4L2 device path and optional card name
///
/// Resolves symlinks to get the real device path and queries the driver name.
pub fn build_device_info(v4l2_path: &str, card: Option<&str>) -> DeviceInfo {
    // Get real path by resolving symlinks
    let real_path = std::fs::canonicalize(v4l2_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| v4l2_path.to_string());

    // Get driver name using V4L2 ioctl
    let driver = get_v4l2_driver(v4l2_path).unwrap_or_default();

    DeviceInfo {
        card: card.unwrap_or_default().to_string(),
        driver,
        path: v4l2_path.to_string(),
        real_path,
    }
}

/// Discover V4L2 subdevices that support focus control (lens actuators)
///
/// Scans `/dev/v4l-subdev*` for devices that support `V4L2_CID_FOCUS_ABSOLUTE`.
/// Returns a list of (device_path, name) tuples for discovered actuators.
pub fn discover_lens_actuators() -> Vec<(String, String)> {
    use super::v4l2_controls;

    let mut actuators = Vec::new();

    // Scan /dev/v4l-subdev* devices
    let entries = match std::fs::read_dir("/dev") {
        Ok(entries) => entries,
        Err(_) => return actuators,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("v4l-subdev") {
            continue;
        }

        let path = format!("/dev/{}", name_str);

        // Check if this subdevice supports V4L2_CID_FOCUS_ABSOLUTE
        if let Some(info) =
            v4l2_controls::query_control(&path, v4l2_controls::V4L2_CID_FOCUS_ABSOLUTE)
            && !info.is_disabled()
        {
            // Read name from sysfs for logging
            let sysfs_name =
                std::fs::read_to_string(format!("/sys/class/video4linux/{}/name", name_str))
                    .unwrap_or_default()
                    .trim()
                    .to_string();

            let display_name = if sysfs_name.is_empty() {
                name_str.to_string()
            } else {
                sysfs_name
            };

            debug!(
                path = %path,
                name = %display_name,
                range = format!("{}-{}", info.minimum, info.maximum),
                "Discovered lens actuator with focus control"
            );
            actuators.push((path, display_name));
        }
    }

    actuators
}

/// Find the V4L2 video capture device path for a libcamera camera ID.
///
/// libcamera camera IDs for UVC cameras contain the USB VID:PID as the last
/// segment (e.g., `\_SB_.PCI0.GP17.XHC1.RHUB.PRT4-2.3:1.0-3564:fef8`).
/// This function extracts the VID:PID, then scans `/sys/class/video4linux/`
/// to find a matching `/dev/videoX` device.
pub fn find_v4l2_device_for_libcamera(camera_id: &str) -> Option<String> {
    // Extract VID:PID from camera ID.
    // UVC IDs end with "-VVVV:PPPP" (4-hex-digit vendor : 4-hex-digit product).
    let vid_pid = camera_id.rsplit('-').next()?;
    let (vid_str, pid_str) = vid_pid.split_once(':')?;
    if vid_str.len() != 4 || pid_str.len() != 4 {
        return None;
    }
    // Verify they're valid hex
    u16::from_str_radix(vid_str, 16).ok()?;
    u16::from_str_radix(pid_str, 16).ok()?;

    debug!(
        camera_id,
        vid = vid_str,
        pid = pid_str,
        "Looking for V4L2 device"
    );

    let entries = std::fs::read_dir("/sys/class/video4linux").ok()?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with("video") {
            continue;
        }

        // Read the device symlink to find the USB device hierarchy
        let device_link = format!("/sys/class/video4linux/{}/device", name_str);
        let resolved = match std::fs::canonicalize(&device_link) {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        // Walk up the sysfs tree to find idVendor/idProduct files
        let mut path = std::path::PathBuf::from(&resolved);
        let mut matched = false;
        for _ in 0..5 {
            let vendor_file = path.join("idVendor");
            let product_file = path.join("idProduct");
            if vendor_file.exists() && product_file.exists() {
                let vendor = std::fs::read_to_string(&vendor_file)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                let product = std::fs::read_to_string(&product_file)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if vendor == vid_str && product == pid_str {
                    matched = true;
                }
                break;
            }
            if !path.pop() {
                break;
            }
        }

        if !matched {
            continue;
        }

        // Verify this is a video capture device (not metadata)
        let dev_path = format!("/dev/{}", name_str);
        let file = match std::fs::File::open(&dev_path) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let cap = match query_v4l2_cap(file.as_raw_fd()) {
            Some(c) => c,
            None => continue,
        };

        // Use device_caps if available, otherwise capabilities
        let caps = if cap.device_caps != 0 {
            cap.device_caps
        } else {
            cap.capabilities
        };

        if caps & V4L2_CAP_VIDEO_CAPTURE == 0 {
            continue;
        }

        // Verify this device actually has enumerable formats.
        // Some UVC cameras expose multiple /dev/videoX nodes where one is
        // metadata-only and reports no formats despite having VIDEO_CAPTURE.
        let mut fmtdesc: V4l2Fmtdesc = unsafe { std::mem::zeroed() };
        fmtdesc.index = 0;
        fmtdesc.buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        let has_formats = unsafe {
            libc::ioctl(
                file.as_raw_fd(),
                VIDIOC_ENUM_FMT as _,
                &mut fmtdesc as *mut _,
            )
        } >= 0;
        if !has_formats {
            debug!(device = %dev_path, "V4L2 device has no formats, skipping");
            continue;
        }

        debug!(camera_id, device = %dev_path, "Found V4L2 device for libcamera camera");
        return Some(dev_path);
    }

    debug!(camera_id, "No V4L2 device found for libcamera camera");
    None
}

/// Discover new V4L2 video capture devices from a set of device node names.
///
/// For each node name (e.g. `"video2"`), opens `/dev/<name>`, checks it
/// supports single-planar video capture, and returns `(dev_path, card_name)`
/// for each match. Metadata-only nodes are filtered out.
pub fn discover_v4l2_capture_devices(
    node_names: &std::collections::BTreeSet<std::ffi::OsString>,
) -> Vec<(String, String)> {
    let mut results = Vec::new();
    for name in node_names {
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let dev_path = format!("/dev/{}", name_str);
        let file = match std::fs::File::open(&dev_path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let cap = match query_v4l2_cap(file.as_raw_fd()) {
            Some(c) => c,
            None => continue,
        };
        let caps = if cap.device_caps != 0 {
            cap.device_caps
        } else {
            cap.capabilities
        };
        if caps & V4L2_CAP_VIDEO_CAPTURE == 0 {
            continue;
        }
        let card_len = cap.card.iter().position(|&c| c == 0).unwrap_or(32);
        let card = String::from_utf8_lossy(&cap.card[..card_len]).to_string();
        debug!(dev_path, card, "Discovered V4L2 capture device");
        results.push((dev_path, card));
    }
    results
}

/// Scan `/dev/` for `video*` device node names.
///
/// Returns a sorted set of filenames (e.g. `{"video0", "video1"}`).
/// This works regardless of whether a capture pipeline is active because
/// it doesn't touch libcamera at all.
pub fn scan_video_device_nodes() -> std::collections::BTreeSet<std::ffi::OsString> {
    let Ok(entries) = std::fs::read_dir("/dev") else {
        return std::collections::BTreeSet::new();
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name())
        .filter(|name| {
            name.to_str()
                .map(|s| s.starts_with("video"))
                .unwrap_or(false)
        })
        .collect()
}

/// Detect CSI-2 bit depth from packed stride relative to image width.
///
/// Returns `Some(10)`, `Some(12)`, or `Some(14)` for recognized CSI-2 packed formats.
/// Returns `None` if the stride doesn't match any known packing ratio.
pub fn detect_csi2_bit_depth(width: u32, packed_stride: u32) -> Option<u32> {
    let min_stride_10 = (width * 5).div_ceil(4);
    let min_stride_12 = (width * 3).div_ceil(2);
    let min_stride_14 = (width * 7).div_ceil(4);

    if packed_stride >= min_stride_10 && packed_stride < min_stride_12 {
        Some(10)
    } else if packed_stride >= min_stride_12 && packed_stride < min_stride_14 {
        Some(12)
    } else if packed_stride >= min_stride_14 && packed_stride < width * 2 {
        Some(14)
    } else {
        None
    }
}

/// Check if a libcamera pipeline handler supports multi-stream capture
///
/// With native libcamera-rs bindings, all known pipeline handlers support
/// ViewFinder + Raw dual-stream configuration:
/// - "simple" handler: ViewFinder (Software ISP) + Raw (bypass ISP)
/// - Hardware ISP handlers ("vc4", "ipu3", "rkisp1"): native multi-stream
pub fn supports_multistream(pipeline_handler: Option<&str>) -> bool {
    // All known pipeline handlers (both "simple" Software ISP and hardware ISP
    // handlers like "vc4", "ipu3", "rkisp1") support ViewFinder + Raw dual-stream.
    // If no handler is known, assume single-stream to be safe.
    pipeline_handler.is_some()
}

// ===== V4L2 Format Enumeration =====

/// VIDIOC_ENUM_FMT ioctl number (v4l2_fmtdesc: 64 bytes, nr=2)
const VIDIOC_ENUM_FMT: libc::c_ulong = 0xC0405602;

/// VIDIOC_ENUM_FRAMESIZES ioctl number (v4l2_frmsizeenum: 44 bytes, nr=74)
const VIDIOC_ENUM_FRAMESIZES: libc::c_ulong = 0xC02C564A;

/// VIDIOC_ENUM_FRAMEINTERVALS ioctl number (v4l2_frmivalenum: 52 bytes, nr=75)
const VIDIOC_ENUM_FRAMEINTERVALS: libc::c_ulong = 0xC034564B;

/// V4L2 buffer type for video capture
const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;

/// Frame size type: discrete
const V4L2_FRMSIZE_TYPE_DISCRETE: u32 = 1;

/// Frame interval type: discrete
const V4L2_FRMIVAL_TYPE_DISCRETE: u32 = 1;

/// V4L2 format description structure
#[repr(C)]
struct V4l2Fmtdesc {
    index: u32,
    buf_type: u32,
    flags: u32,
    description: [u8; 32],
    pixelformat: u32,
    mbus_code: u32,
    reserved: [u32; 3],
}

/// V4L2 discrete frame size within frmsizeenum union
#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2FrmsizeDiscrete {
    width: u32,
    height: u32,
}

/// V4L2 frame size enumeration structure
#[repr(C)]
struct V4l2Frmsizeenum {
    index: u32,
    pixel_format: u32,
    size_type: u32,
    discrete: V4l2FrmsizeDiscrete,
    _reserved: [u32; 6],
}

/// V4L2 fraction (for frame intervals)
#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2Fract {
    numerator: u32,
    denominator: u32,
}

/// V4L2 frame interval enumeration structure
#[repr(C)]
struct V4l2Frmivalenum {
    index: u32,
    pixel_format: u32,
    width: u32,
    height: u32,
    interval_type: u32,
    discrete: V4l2Fract,
    _reserved: [u32; 6],
}

/// A single V4L2 format with resolution and frame rates
#[derive(Debug, Clone)]
pub struct V4l2FormatInfo {
    /// FourCC string (e.g., "MJPG", "YUYV", "H264")
    pub fourcc: String,
    /// Human-readable description from the driver
    pub description: String,
    /// Available resolutions with their frame rates
    pub sizes: Vec<V4l2SizeInfo>,
}

/// A resolution with its available frame rates
#[derive(Debug, Clone)]
pub struct V4l2SizeInfo {
    pub width: u32,
    pub height: u32,
    /// Frame rates as (numerator, denominator) fractions
    pub framerates: Vec<(u32, u32)>,
}

/// Convert a V4L2 pixelformat u32 to a FourCC string
fn fourcc_to_string(fourcc: u32) -> String {
    let bytes = fourcc.to_le_bytes();
    bytes.iter().map(|&b| b as char).collect()
}

/// Enumerate all V4L2 formats, resolutions, and frame rates for a device.
///
/// Returns an empty Vec if the device cannot be opened or doesn't support
/// format enumeration.
pub fn enumerate_v4l2_formats(device_path: &str) -> Vec<V4l2FormatInfo> {
    let file = match std::fs::File::open(device_path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let fd = file.as_raw_fd();
    let mut formats = Vec::new();

    // Enumerate pixel formats
    for fmt_idx in 0u32..64 {
        let mut fmtdesc: V4l2Fmtdesc = unsafe { std::mem::zeroed() };
        fmtdesc.index = fmt_idx;
        fmtdesc.buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        let ret = unsafe { libc::ioctl(fd, VIDIOC_ENUM_FMT as _, &mut fmtdesc as *mut _) };
        if ret < 0 {
            break;
        }

        let fourcc = fourcc_to_string(fmtdesc.pixelformat);
        let desc_len = fmtdesc
            .description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(32);
        let description = String::from_utf8_lossy(&fmtdesc.description[..desc_len]).to_string();

        // Enumerate frame sizes for this format
        let mut sizes = Vec::new();
        for size_idx in 0u32..256 {
            let mut frmsize: V4l2Frmsizeenum = unsafe { std::mem::zeroed() };
            frmsize.index = size_idx;
            frmsize.pixel_format = fmtdesc.pixelformat;

            let ret =
                unsafe { libc::ioctl(fd, VIDIOC_ENUM_FRAMESIZES as _, &mut frmsize as *mut _) };
            if ret < 0 {
                break;
            }

            if frmsize.size_type != V4L2_FRMSIZE_TYPE_DISCRETE {
                break; // Only handle discrete sizes
            }

            let width = frmsize.discrete.width;
            let height = frmsize.discrete.height;

            // Enumerate frame intervals for this format+size
            let mut framerates = Vec::new();
            for ival_idx in 0u32..64 {
                let mut frmival: V4l2Frmivalenum = unsafe { std::mem::zeroed() };
                frmival.index = ival_idx;
                frmival.pixel_format = fmtdesc.pixelformat;
                frmival.width = width;
                frmival.height = height;

                let ret = unsafe {
                    libc::ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS as _, &mut frmival as *mut _)
                };
                if ret < 0 {
                    break;
                }

                if frmival.interval_type != V4L2_FRMIVAL_TYPE_DISCRETE {
                    break;
                }

                // Frame interval is numerator/denominator (e.g., 1/30 = 30fps)
                framerates.push((frmival.discrete.numerator, frmival.discrete.denominator));
            }

            sizes.push(V4l2SizeInfo {
                width,
                height,
                framerates,
            });
        }

        formats.push(V4l2FormatInfo {
            fourcc,
            description,
            sizes,
        });
    }

    formats
}
