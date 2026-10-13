// SPDX-License-Identifier: GPL-3.0-only
//! Native discovery snapshots; PipeWire registry results must never be cached here.

use super::CameraDevice;
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::os::unix::fs::MetadataExt;

// Two subsequent hotplug polls allow newly exposed nodes to become ready.
// Never retry a settled PipeWire-only topology indefinitely.
const READINESS_RETRIES: usize = 2;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct VideoNodeIdentity {
    inode: u64,
    device_number: u64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

pub(super) type VideoNodeTopology = BTreeMap<OsString, VideoNodeIdentity>;

pub(super) fn scan_video_topology() -> std::io::Result<VideoNodeTopology> {
    scan_video_topology_at(std::path::Path::new("/dev"))
}

fn scan_video_topology_at(directory: &std::path::Path) -> std::io::Result<VideoNodeTopology> {
    let mut topology = BTreeMap::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        if !name.to_str().is_some_and(|name| name.starts_with("video")) {
            continue;
        }
        // Names and device numbers can be reused between polls. Include the
        // devtmpfs inode and change time to detect removal/recreation as well.
        let metadata = entry.metadata()?;
        topology.insert(
            name,
            VideoNodeIdentity {
                inode: metadata.ino(),
                device_number: metadata.rdev(),
                changed_seconds: metadata.ctime(),
                changed_nanoseconds: metadata.ctime_nsec(),
            },
        );
    }
    Ok(topology)
}

#[derive(Default)]
pub(super) struct NativeDiscoveryCache {
    pub(super) devices: Vec<CameraDevice>,
    video_nodes: Option<VideoNodeTopology>,
    readiness_retries: usize,
}

impl NativeDiscoveryCache {
    pub(super) fn matches(&self, video_nodes: &VideoNodeTopology) -> bool {
        self.readiness_retries == 0 && self.video_nodes.as_ref() == Some(video_nodes)
    }

    pub(super) fn update(&mut self, video_nodes: VideoNodeTopology, devices: Vec<CameraDevice>) {
        self.readiness_retries = if self.video_nodes.as_ref() == Some(&video_nodes) {
            self.readiness_retries.saturating_sub(1)
        } else {
            READINESS_RETRIES
        };
        self.video_nodes = Some(video_nodes);
        self.devices = devices;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(names: &[&str]) -> VideoNodeTopology {
        names
            .iter()
            .map(|name| (OsString::from(name), VideoNodeIdentity::default()))
            .collect()
    }

    fn settle(
        cache: &mut NativeDiscoveryCache,
        topology: VideoNodeTopology,
        devices: Vec<CameraDevice>,
    ) {
        for _ in 0..=READINESS_RETRIES {
            cache.update(topology.clone(), devices.clone());
        }
    }

    #[test]
    fn unchanged_topology_reuses_empty_native_discovery() {
        // A PipeWire-only laptop legitimately has no native cameras. An empty
        // successful result is still a snapshot, not a reason to reprobe.
        let mut cache = NativeDiscoveryCache::default();
        let topology = nodes(&["video0", "video1"]);
        assert!(!cache.matches(&topology));
        settle(&mut cache, topology.clone(), Vec::new());
        for _ in 0..3 {
            assert!(
                cache.matches(&topology),
                "unchanged topology must reuse discovery"
            );
        }
    }

    #[test]
    fn video_node_addition_invalidates_native_discovery() {
        let mut cache = NativeDiscoveryCache::default();
        settle(&mut cache, nodes(&["video0"]), Vec::new());
        assert!(!cache.matches(&nodes(&["video0", "video1"])));
    }

    #[test]
    fn delayed_native_camera_is_retried_without_node_changes() {
        let mut cache = NativeDiscoveryCache::default();
        let topology = nodes(&["video0"]);
        cache.update(topology.clone(), Vec::new());
        assert!(
            !cache.matches(&topology),
            "new topology needs readiness retries"
        );
        let device = CameraDevice {
            path: "platform/late-sensor".into(),
            ..Default::default()
        };
        cache.update(topology.clone(), vec![device.clone()]);
        assert!(!cache.matches(&topology));
        cache.update(topology.clone(), vec![device]);
        assert!(cache.matches(&topology));
        assert_eq!(cache.devices[0].path, "platform/late-sensor");
    }

    #[test]
    fn video_node_removal_invalidates_native_discovery() {
        let mut cache = NativeDiscoveryCache::default();
        settle(&mut cache, nodes(&["video0", "video1"]), Vec::new());
        assert!(!cache.matches(&nodes(&["video0"])));
    }

    #[test]
    fn same_name_video_node_replacement_invalidates_native_discovery() {
        let directory =
            std::env::temp_dir().join(format!("camera-topology-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("video0"), []).unwrap();
        let first = scan_video_topology_at(&directory).unwrap();
        // Atomic replacement preserves the filename but changes its inode.
        std::fs::write(directory.join("replacement"), []).unwrap();
        std::fs::rename(directory.join("replacement"), directory.join("video0")).unwrap();
        let second = scan_video_topology_at(&directory).unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        let mut cache = NativeDiscoveryCache::default();
        settle(&mut cache, first, Vec::new());
        assert!(
            !cache.matches(&second),
            "replaced video node must invalidate discovery"
        );
    }

    #[test]
    fn empty_topology_requires_initial_discovery() {
        let mut cache = NativeDiscoveryCache::default();
        assert!(!cache.matches(&nodes(&[])));
        settle(&mut cache, nodes(&[]), Vec::new());
        assert!(cache.matches(&nodes(&[])));
    }

    #[test]
    fn native_descriptors_survive_unchanged_topology() {
        let mut cache = NativeDiscoveryCache::default();
        settle(
            &mut cache,
            nodes(&["video0"]),
            vec![CameraDevice {
                path: "platform/sensor-1".into(),
                supports_multistream: true,
                ..Default::default()
            }],
        );
        assert!(cache.matches(&nodes(&["video0"])));
        assert_eq!(cache.devices[0].path, "platform/sensor-1");
        assert!(cache.devices[0].supports_multistream);
        cache.update(nodes(&[]), Vec::new());
        assert!(cache.devices.is_empty());
        assert!(!cache.matches(&nodes(&["video0"])));
    }
}
