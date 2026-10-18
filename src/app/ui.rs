// SPDX-License-Identifier: GPL-3.0-only

//! UI helper functions
//!
//! This module contains shared UI utilities that are used by multiple view modules.

use crate::app::state::AppModel;
use crate::backends::camera::types::CameraFormat;
use crate::constants;
use std::collections::HashMap;

/// One winning resolution per label, retaining original format indices.
pub(crate) struct ResolutionGroup<'a> {
    pub label: &'static str,
    pub width: u32,
    pub formats: Vec<(usize, &'a CameraFormat)>,
}
/// (best_width, best_score, formats) for label grouping
type BestFormatEntry<'a> = (u32, u32, Vec<(usize, &'a CameraFormat)>);

impl AppModel {
    /// Group formats by resolution label and return sorted list with best resolution for each label
    ///
    /// This helper is used by the format picker to organize formats by resolution categories
    /// (SD, HD, 720p, 4K, etc.).
    ///
    /// Groups are ordered by width; format ties retain their input order.
    pub(crate) fn group_formats_by_label(&self) -> Vec<ResolutionGroup<'_>> {
        // Group formats by their label (SD, HD, 4K, etc.) and pick the highest resolution for each
        let mut label_to_best_format: HashMap<&'static str, BestFormatEntry<'_>> = HashMap::new();

        for (idx, fmt) in self.available_formats.iter().enumerate() {
            if let Some(label) = constants::get_resolution_label(fmt.width) {
                let resolution_score = fmt.width * fmt.height;

                label_to_best_format
                    .entry(label)
                    .and_modify(|(best_width, best_score, formats)| {
                        if resolution_score > *best_score {
                            *best_width = fmt.width;
                            *best_score = resolution_score;
                            formats.clear();
                            formats.push((idx, fmt));
                        } else if resolution_score == *best_score && fmt.width == *best_width {
                            formats.push((idx, fmt));
                        }
                    })
                    .or_insert_with(|| (fmt.width, resolution_score, vec![(idx, fmt)]));
            }
        }

        let mut groups: Vec<_> = label_to_best_format
            .into_iter()
            .map(|(label, (width, _, formats))| ResolutionGroup {
                label,
                width,
                formats,
            })
            .collect();
        groups.sort_by_key(|group| group.width);
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::camera::types::Framerate;

    #[test]
    fn resolution_groups_keep_winning_width_ties_and_source_indices() {
        let format = |width, height, fps, pixel_format: &str| CameraFormat {
            width,
            height,
            framerate: Some(Framerate::from_int(fps)),
            hardware_accelerated: false,
            pixel_format: pixel_format.into(),
        };
        let model = AppModel {
            available_formats: vec![
                format(3840, 2160, 30, "MJPG"), // 0: 4K
                format(1920, 1080, 60, "MJPG"), // 1: replaced HD
                format(320, 240, 30, "YUYV"),   // 2: unknown label
                format(1280, 720, 30, "MJPG"),  // 3: replaced SD
                format(2000, 1200, 30, "MJPG"), // 4: winning HD
                format(2400, 1000, 60, "MJPG"), // 5: equal area, different width: not retained
                format(2000, 1200, 60, "YUYV"), // 6: retained same-width tie
                format(2000, 1200, 30, "YUYV"), // 7: retained codec tie
                format(1600, 900, 30, "MJPG"),  // 8: winning SD
                format(1920, 1080, 30, "MJPG"), // 9: smaller HD: not retained
            ],
            ..Default::default()
        };
        let groups = model.group_formats_by_label();
        assert_eq!(
            groups
                .iter()
                .map(|g| (g.label, g.width))
                .collect::<Vec<_>>(),
            vec![("SD", 1600), ("HD", 2000), ("4K", 3840)]
        );
        let indices = groups
            .iter()
            .map(|g| g.formats.iter().map(|&(i, _)| i).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(indices, vec![vec![8], vec![4, 6, 7], vec![0]]);
        for group in &groups {
            for &(index, format) in &group.formats {
                assert!(std::ptr::eq(format, &model.available_formats[index]));
            }
        }
        assert!(!groups.iter().any(|g| g.width == 1920));
        assert!(AppModel::default().group_formats_by_label().is_empty());
    }
}
