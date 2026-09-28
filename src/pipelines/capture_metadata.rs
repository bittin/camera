// SPDX-License-Identifier: GPL-3.0-only

//! Shared, privacy-conscious metadata for captured media.

use std::sync::OnceLock;

use chrono::{DateTime, FixedOffset};

use crate::backends::camera::types::SensorRotation;

/// Standard display transform stored in EXIF/QuickTime/Matroska metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CaptureOrientation {
    #[default]
    Rotate0,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipRotate0,
    FlipRotate90,
    FlipRotate180,
    FlipRotate270,
}

impl CaptureOrientation {
    /// Preserve the former pixel operation: inverse sensor rotation followed
    /// by a horizontal mirror in upright output coordinates.
    pub fn from_capture(rotation: SensorRotation, mirror_horizontal: bool) -> Self {
        match (rotation, mirror_horizontal) {
            (SensorRotation::None, false) => Self::Rotate0,
            (SensorRotation::Rotate90, false) => Self::Rotate270,
            (SensorRotation::Rotate180, false) => Self::Rotate180,
            (SensorRotation::Rotate270, false) => Self::Rotate90,
            (SensorRotation::None, true) => Self::FlipRotate0,
            (SensorRotation::Rotate90, true) => Self::FlipRotate90,
            (SensorRotation::Rotate180, true) => Self::FlipRotate180,
            (SensorRotation::Rotate270, true) => Self::FlipRotate270,
        }
    }

    pub fn exif_value(self) -> u16 {
        match self {
            Self::Rotate0 => 1,
            Self::FlipRotate0 => 2,
            Self::Rotate180 => 3,
            Self::FlipRotate180 => 4,
            Self::FlipRotate270 => 5,
            Self::Rotate90 => 6,
            Self::FlipRotate90 => 7,
            Self::Rotate270 => 8,
        }
    }

    pub fn gstreamer_tag(self) -> &'static str {
        match self {
            Self::Rotate0 => "rotate-0",
            Self::Rotate90 => "rotate-90",
            Self::Rotate180 => "rotate-180",
            Self::Rotate270 => "rotate-270",
            Self::FlipRotate0 => "flip-rotate-0",
            Self::FlipRotate90 => "flip-rotate-90",
            Self::FlipRotate180 => "flip-rotate-180",
            Self::FlipRotate270 => "flip-rotate-270",
        }
    }

    pub fn swaps_dimensions(self) -> bool {
        matches!(
            self,
            Self::Rotate90 | Self::Rotate270 | Self::FlipRotate90 | Self::FlipRotate270
        )
    }
}

/// Metadata snapshot shared by still-image and video capture pipelines.
#[derive(Debug, Clone, Default)]
pub struct CaptureMetadata {
    /// Human-readable camera name (for example, `Back Camera`).
    pub camera_name: Option<String>,
    /// Camera backend/driver name (for example, `uvcvideo`).
    pub camera_driver: Option<String>,
    /// Camera sensor model (for example, `sony,imx363`).
    pub sensor_model: Option<String>,
    /// Camera position (`front`, `back`, or `external`).
    pub camera_location: Option<String>,
    /// libcamera pipeline handler name.
    pub pipeline_handler: Option<String>,
    /// Runtime libcamera version reported by the active backend.
    pub libcamera_version: Option<String>,
    /// Human-readable host device manufacturer; never a serial or hostname.
    pub device_make: Option<String>,
    /// Human-readable host device model; never a serial or hostname.
    pub device_model: Option<String>,
    /// Wall-clock capture/start time including the local UTC offset.
    pub captured_at: Option<DateTime<FixedOffset>>,
    /// Rotation/mirroring to apply when displaying the encoded pixels.
    pub orientation: CaptureOrientation,
    /// Exposure time in seconds.
    pub exposure_time: Option<f64>,
    /// ISO sensitivity.
    pub iso: Option<u32>,
    /// Gain value in camera-specific units.
    pub gain: Option<i32>,
}

impl CaptureMetadata {
    pub fn application_name() -> String {
        format!("Camera {}", env!("CARGO_PKG_VERSION"))
    }

    pub fn software(&self) -> String {
        let app = Self::application_name();
        match self.libcamera_version.as_deref() {
            Some(version) => format!("{app}; libcamera {version}"),
            None => app,
        }
    }

    pub fn sensor_identity(&self) -> Option<&str> {
        self.sensor_model.as_deref()
    }

    pub fn description(&self) -> Option<String> {
        let mut fields = Vec::new();
        match (self.device_make.as_deref(), self.device_model.as_deref()) {
            (Some(make), Some(model)) => fields.push(format!("Device: {make} {model}")),
            (Some(make), None) => fields.push(format!("Device: {make}")),
            (None, Some(model)) => fields.push(format!("Device: {model}")),
            (None, None) => {}
        }
        if let Some(sensor) = &self.sensor_model {
            fields.push(format!("Sensor: {sensor}"));
        }
        if let Some(location) = &self.camera_location {
            fields.push(format!("Position: {location}"));
        }
        if let Some(driver) = &self.camera_driver {
            fields.push(format!("Driver: {driver}"));
        }
        if let Some(handler) = &self.pipeline_handler {
            fields.push(format!("libcamera pipeline: {handler}"));
        }
        if let Some(gain) = self.gain {
            fields.push(format!("Gain: {gain}"));
        }
        (!fields.is_empty()).then(|| fields.join("; "))
    }
}

/// Human-readable host device identity suitable for standard media metadata.
///
/// This deliberately excludes serial numbers, hostnames, device paths, and
/// other identifiers that could uniquely identify the user's device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostDeviceMetadata {
    pub make: Option<String>,
    pub model: Option<String>,
}

impl HostDeviceMetadata {
    fn from_sources(
        device_tree_model: Option<&str>,
        device_tree_compatible: Option<&str>,
        dmi_vendor: Option<&str>,
        dmi_product: Option<&str>,
    ) -> Self {
        if let Some(model) = clean_value(device_tree_model) {
            let make = device_tree_compatible
                .and_then(first_compatible_vendor)
                .or_else(|| first_model_word(&model));
            return Self {
                model: Some(strip_make_prefix(&model, make.as_deref())),
                make,
            };
        }

        let make = clean_value(dmi_vendor);
        let model =
            clean_value(dmi_product).map(|model| strip_make_prefix(&model, make.as_deref()));
        Self { make, model }
    }
}

/// Detect the host make/model once for reuse by photo and video captures.
pub fn host_device_metadata() -> &'static HostDeviceMetadata {
    static METADATA: OnceLock<HostDeviceMetadata> = OnceLock::new();
    METADATA.get_or_init(|| {
        let dt_model = read_text("/sys/firmware/devicetree/base/model");
        let dt_compatible = read_text("/sys/firmware/devicetree/base/compatible");
        let dmi_vendor = read_text("/sys/class/dmi/id/sys_vendor");
        let dmi_product = read_text("/sys/class/dmi/id/product_name");

        HostDeviceMetadata::from_sources(
            dt_model.as_deref(),
            dt_compatible.as_deref(),
            dmi_vendor.as_deref(),
            dmi_product.as_deref(),
        )
    })
}

fn read_text(path: &str) -> Option<String> {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|value| clean_value(Some(&value)))
}

fn clean_value(value: Option<&str>) -> Option<String> {
    let value = value?.trim_matches(['\0', ' ', '\n', '\r', '\t']);
    (!value.is_empty()).then(|| value.to_string())
}

fn first_compatible_vendor(value: &str) -> Option<String> {
    value
        .split('\0')
        .find_map(|entry| entry.split_once(',').map(|(vendor, _)| vendor))
        .and_then(|vendor| clean_value(Some(vendor)))
        .map(|vendor| title_case_ascii(&vendor))
}

fn first_model_word(model: &str) -> Option<String> {
    model.split_whitespace().next().map(title_case_ascii)
}

fn title_case_ascii(value: &str) -> String {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    first.to_uppercase().collect::<String>() + chars.as_str()
}

fn strip_make_prefix(model: &str, make: Option<&str>) -> String {
    let Some(make) = make else {
        return model.to_string();
    };
    if model.len() > make.len()
        && model[..make.len()].eq_ignore_ascii_case(make)
        && model.as_bytes().get(make.len()) == Some(&b' ')
    {
        model[make.len() + 1..].to_string()
    } else {
        model.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::SensorRotation;

    #[test]
    fn parses_device_tree_identity_without_unique_identifiers() {
        let metadata = HostDeviceMetadata::from_sources(
            Some("Google Pixel 3a\0"),
            Some("google,sargo\0qcom,sdm670\0"),
            None,
            None,
        );

        assert_eq!(metadata.make.as_deref(), Some("Google"));
        assert_eq!(metadata.model.as_deref(), Some("Pixel 3a"));
    }

    #[test]
    fn falls_back_to_dmi_identity_and_avoids_duplicate_make() {
        let metadata = HostDeviceMetadata::from_sources(
            None,
            None,
            Some("LENOVO\n"),
            Some("LENOVO ThinkPad X1 Carbon Gen 12\n"),
        );

        assert_eq!(metadata.make.as_deref(), Some("LENOVO"));
        assert_eq!(metadata.model.as_deref(), Some("ThinkPad X1 Carbon Gen 12"));
    }

    #[test]
    fn orientation_metadata_applies_inverse_sensor_rotation() {
        let orientation = CaptureOrientation::from_capture(SensorRotation::Rotate90, false);
        assert_eq!(orientation.exif_value(), 8);
        assert_eq!(orientation.gstreamer_tag(), "rotate-270");
    }

    #[test]
    fn mirrored_orientation_preserves_rotate_then_output_mirror() {
        let orientation = CaptureOrientation::from_capture(SensorRotation::Rotate90, true);
        assert_eq!(orientation.exif_value(), 7);
        assert_eq!(orientation.gstreamer_tag(), "flip-rotate-90");
    }

    #[test]
    fn description_excludes_backend_camera_ids_and_includes_safe_device_identity() {
        let metadata = CaptureMetadata {
            camera_name: Some("/base/soc/camera@1a".into()),
            sensor_model: Some("sony,imx363".into()),
            camera_location: Some("back".into()),
            device_make: Some("Google".into()),
            device_model: Some("Pixel 3a".into()),
            ..Default::default()
        };

        let description = metadata.description().expect("description");
        assert!(!description.contains("/base/soc"));
        assert!(description.contains("Device: Google Pixel 3a"));
        assert!(description.contains("Sensor: sony,imx363"));
    }

    #[test]
    fn sensor_identity_never_falls_back_to_backend_camera_id() {
        let metadata = CaptureMetadata {
            camera_name: Some("/base/soc/camera@1a".into()),
            ..Default::default()
        };
        assert_eq!(metadata.sensor_identity(), None);
    }
}
