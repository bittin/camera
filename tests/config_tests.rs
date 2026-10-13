// SPDX-License-Identifier: GPL-3.0-only

//! Integration tests for configuration module

use camera::Config;

#[test]
fn demosaic_default_is_rcd() {
    let value = serde_json::to_value(Config::default()).unwrap();
    assert_eq!(
        value.get("demosaic_method"),
        Some(&serde_json::json!("rcd"))
    );
}

#[test]
fn demosaic_preferences_round_trip_through_cosmic_config() {
    use camera::shaders::DemosaicMethod;
    use cosmic::cosmic_config::{Config as Store, ConfigGet, CosmicConfigEntry};
    let root =
        std::env::temp_dir().join(format!("camera-demosaic-config-{}", uuid::Uuid::new_v4()));
    let store =
        Store::with_custom_path("camera-demosaic-test", Config::VERSION, root.clone()).unwrap();
    for (method, spelling) in [
        (DemosaicMethod::Rcd, "rcd"),
        (DemosaicMethod::Lmmse, "lmmse"),
        (DemosaicMethod::Bilinear, "bilinear"),
    ] {
        let config = Config {
            demosaic_method: method,
            ..Default::default()
        };
        config.write_entry(&store).unwrap();
        assert_eq!(Config::get_entry(&store).unwrap().demosaic_method, method);
        assert_eq!(
            store.get::<DemosaicMethod>("demosaic_method").unwrap(),
            method
        );
        assert_eq!(
            serde_json::to_value(method).unwrap(),
            serde_json::json!(spelling)
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn old_config_without_demosaic_method_defaults_to_rcd() {
    use camera::shaders::DemosaicMethod;
    use cosmic::cosmic_config::{Config as Store, CosmicConfigEntry};
    let mut json = serde_json::to_value(Config::default()).unwrap();
    json.as_object_mut().unwrap().remove("demosaic_method");
    assert_eq!(
        serde_json::from_value::<Config>(json)
            .unwrap()
            .demosaic_method,
        DemosaicMethod::Rcd
    );

    let root = std::env::temp_dir().join(format!(
        "camera-old-demosaic-config-{}",
        uuid::Uuid::new_v4()
    ));
    let store =
        Store::with_custom_path("camera-demosaic-test", Config::VERSION, root.clone()).unwrap();
    Config::default().write_entry(&store).unwrap();
    let mut pending = vec![root.clone()];
    let mut files = Vec::new();
    while let Some(path) = pending.pop() {
        for entry in std::fs::read_dir(path).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                pending.push(entry.path());
            } else if entry.file_name() == "demosaic_method" {
                files.push(entry.path());
            }
        }
    }
    assert_eq!(files.len(), 1);
    std::fs::remove_file(&files[0]).unwrap();
    let restored = Config::get_entry(&store).unwrap_or_else(|(_, recovered)| recovered);
    assert_eq!(restored.demosaic_method, DemosaicMethod::Rcd);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn avif_preference_round_trips_through_cosmic_config() {
    use camera::config::PhotoOutputFormat;
    use cosmic::cosmic_config::{Config as Store, CosmicConfigEntry};

    let root = std::env::temp_dir().join(format!("camera-config-test-{}", uuid::Uuid::new_v4()));
    let store = Store::with_custom_path("camera-test", Config::VERSION, root.clone()).unwrap();
    let config = Config {
        photo_output_format: PhotoOutputFormat::Avif,
        ..Default::default()
    };
    config.write_entry(&store).unwrap();
    let restored = Config::get_entry(&store);
    std::fs::remove_dir_all(root).unwrap();
    let restored = restored.expect("AVIF preference must survive persistence");
    assert_eq!(restored.photo_output_format, PhotoOutputFormat::Avif);
    assert_eq!(PhotoOutputFormat::default(), PhotoOutputFormat::Jpeg);
    assert_eq!(PhotoOutputFormat::Avif.display_name(), "AVIF");
    assert_eq!(
        &PhotoOutputFormat::ALL[..3],
        &[
            PhotoOutputFormat::Jpeg,
            PhotoOutputFormat::Png,
            PhotoOutputFormat::Dng
        ]
    );
}

fn assigned_number(source: &str, prefix: &str) -> u64 {
    source
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix(prefix)
                .and_then(|value| value.trim_end_matches(';').parse().ok())
        })
        .unwrap_or_else(|| panic!("missing numeric assignment for {prefix}"))
}

fn config_schema_version(source: &str) -> u64 {
    source
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("#[version = ")
                .and_then(|value| value.strip_suffix(']'))
                .and_then(|value| value.parse().ok())
        })
        .expect("Config is missing its cosmic-config schema version")
}

#[test]
fn obsolete_normal_raw_preference_does_not_affect_config_loading() {
    use cosmic::cosmic_config::{Config as Store, ConfigSet, CosmicConfigEntry};
    let root = std::env::temp_dir().join(format!(
        "camera-normal-config-test-{}",
        uuid::Uuid::new_v4()
    ));
    let store =
        Store::with_custom_path("camera-normal-test", Config::VERSION, root.clone()).unwrap();
    let config = Config::default();
    config.write_entry(&store).unwrap();
    // Existing installations may retain either value of the retired setting.
    for enabled in [false, true] {
        store.set("normal_raw", enabled).unwrap();
        let restored = Config::get_entry(&store).unwrap();
        assert_eq!(restored.photo_output_format, config.photo_output_format);
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_config_default() {
    // Test that default config can be created
    let config = Config::default();

    // Check sensible defaults
    assert!(
        config.mirror_preview,
        "Mirror preview should be enabled by default"
    );
}

#[test]
fn test_config_bug_report_url() {
    // Test that bug report URL is set
    let config = Config::default();
    assert!(
        !config.bug_report_url.is_empty(),
        "Bug report URL should not be empty"
    );
}

#[test]
fn preview_harness_uses_current_config_schema() {
    let config_source = include_str!("../src/config.rs");
    let harness_source = include_str!("../preview/capture-previews.sh");

    assert_eq!(
        assigned_number(harness_source, "CONFIG_VERSION="),
        config_schema_version(config_source),
        "preview/capture-previews.sh writes forced appearance and fit/fill settings to a stale cosmic-config directory"
    );
}
