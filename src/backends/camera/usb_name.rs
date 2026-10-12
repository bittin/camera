// SPDX-License-Identifier: GPL-3.0-only

//! Display-only USB camera name enrichment. Never use these names as identity.

use std::os::unix::fs::{FileTypeExt, MetadataExt};

pub(super) fn display_name(original: &str, device_path: &str) -> String {
    // V4L2 card names can repeat the model before a colon (and truncate
    // the second copy). Keep the more specific suffix, not both copies.
    let original = original
        .split_once(": ")
        .filter(|(prefix, model)| !prefix.is_empty() && model.starts_with(prefix))
        .map_or(original, |(_, model)| model);
    if !is_generic_name(original) {
        return original.to_string();
    }
    usb_parent(device_path)
        .map(|parent| {
            let name = select_name(
                original,
                parent
                    .property_value("ID_MODEL_FROM_DATABASE")
                    .and_then(|v| v.to_str()),
            );
            if name != original {
                return name;
            }
            select_name(
                original,
                parent.attribute_value("product").and_then(|v| v.to_str()),
            )
        })
        .unwrap_or_else(|| original.to_string())
}

/// Resolve the physical USB parent without opening or acquiring the camera.
fn usb_parent(device_path: &str) -> Option<udev::Device> {
    let metadata = std::fs::metadata(device_path).ok()?;
    if !metadata.file_type().is_char_device() {
        return None;
    }
    let device = udev::Device::from_devnum(udev::DeviceType::Character, metadata.rdev()).ok()?;
    if device.subsystem()?.to_str()? != "video4linux" {
        return None;
    }
    device
        .parent_with_subsystem_devtype("usb", "usb_device")
        .ok()?
}

fn select_name(original: &str, model: Option<&str>) -> String {
    let model = model.map(str::trim).filter(|model| {
        !model.is_empty()
            && !is_generic_name(model)
            && !(model.len() == 4 && model.bytes().all(|b| b.is_ascii_hexdigit()))
    });
    if !is_generic_name(original) {
        return original.to_string();
    }
    let Some(model) = model else {
        return original.to_string();
    };
    model.to_string()
}

fn is_generic_name(name: &str) -> bool {
    // Location labels are placeholders too: USB cameras can report "front".
    // display_name only replaces them after resolving a physical USB parent,
    // so built-in phone cameras retain their positional labels.
    if matches!(
        name,
        "UVC Camera" | "USB Camera" | "USB2.0 Camera" | "Front Camera" | "Back Camera" | "Camera"
    ) {
        return true;
    }
    name.strip_prefix("UVC Camera (")
        .and_then(|name| name.strip_suffix(')'))
        .and_then(|ids| ids.split_once(':'))
        .is_some_and(|(vendor, product)| {
            vendor.len() == 4
                && product.len() == 4
                && vendor
                    .bytes()
                    .chain(product.bytes())
                    .all(|b| b.is_ascii_hexdigit())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a connected Logitech C270 and its udev database entry; does not acquire the camera"]
    fn live_usb_parent_supplies_camera_model() {
        let mut enumerator = udev::Enumerator::new().unwrap();
        enumerator.match_subsystem("video4linux").unwrap();
        let device = enumerator
            .scan_devices()
            .unwrap()
            .find(|device| {
                device
                    .parent_with_subsystem_devtype("usb", "usb_device")
                    .unwrap()
                    .is_some_and(|parent| {
                        parent.attribute_value("idVendor").and_then(|v| v.to_str()) == Some("046d")
                            && parent.attribute_value("idProduct").and_then(|v| v.to_str())
                                == Some("0825")
                    })
            })
            .expect("connected Logitech C270");
        let path = device.devnode().unwrap();
        let name = display_name("UVC Camera (046d:0825)", path.to_str().unwrap());
        assert_eq!(name, "Webcam C270");
        println!("{}: {name}", path.display());
    }

    #[test]
    fn informative_camera_names_are_preserved() {
        for original in ["Logitech HD Pro Webcam C920", "Live Streamer CAM 313"] {
            assert_eq!(select_name(original, Some("Other model")), original);
        }
    }

    #[test]
    fn positional_usb_names_use_model() {
        for original in ["Front Camera", "Back Camera", "Camera"] {
            assert_eq!(
                select_name(original, Some("Live Streamer CAM 313")),
                "Live Streamer CAM 313"
            );
        }
    }

    #[test]
    fn non_usb_positional_names_are_preserved() {
        for original in ["Front Camera", "Back Camera", "Camera"] {
            assert_eq!(display_name(original, "/dev/null"), original);
        }
    }

    #[test]
    #[ignore = "requires a connected Live Streamer CAM 313; does not acquire the camera"]
    fn live_front_usb_camera_uses_model() {
        let mut enumerator = udev::Enumerator::new().unwrap();
        enumerator.match_subsystem("video4linux").unwrap();
        let device = enumerator
            .scan_devices()
            .unwrap()
            .find(|device| {
                device
                    .parent_with_subsystem_devtype("usb", "usb_device")
                    .unwrap()
                    .is_some_and(|parent| {
                        parent.attribute_value("idVendor").and_then(|v| v.to_str()) == Some("07ca")
                            && parent.attribute_value("idProduct").and_then(|v| v.to_str())
                                == Some("313a")
                    })
            })
            .expect("connected Live Streamer CAM 313");
        let path = device.devnode().unwrap();
        let name = display_name("Front Camera", path.to_str().unwrap());
        assert_eq!(name, "Live Streamer CAM 313");
        println!("{}: {name}", path.display());
    }

    #[test]
    fn missing_or_unhelpful_database_models_keep_original_name() {
        for model in [None, Some(""), Some("  "), Some("0825"), Some("UVC Camera")] {
            assert_eq!(
                select_name("UVC Camera (046d:0825)", model),
                "UVC Camera (046d:0825)"
            );
        }
    }

    #[test]
    fn model_names_are_trimmed_and_preserved() {
        assert_eq!(
            select_name("USB Camera", Some("Webcam C270")),
            "Webcam C270"
        );
        assert_eq!(
            select_name("USB Camera", Some("Logitech Webcam C270")),
            "Logitech Webcam C270"
        );
        assert_eq!(
            select_name("USB Camera", Some(" Webcam C270 ")),
            "Webcam C270"
        );
    }

    #[test]
    fn generic_uvc_name_uses_hardware_database_model() {
        assert_eq!(
            select_name("UVC Camera (046d:0825)", Some("Webcam C270")),
            "Webcam C270"
        );
    }

    #[test]
    fn repeated_v4l2_card_prefix_is_removed() {
        assert_eq!(
            display_name("OBSBOT Tiny 2: OBSBOT Tiny 2 St", "/dev/null"),
            "OBSBOT Tiny 2 St"
        );
        assert_eq!(
            display_name("Camera: Special model", "/dev/null"),
            "Camera: Special model"
        );
    }
}
