// SPDX-License-Identifier: GPL-3.0-only
/// Photo demosaic selection; realtime recording explicitly retains GPU Bilinear.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DemosaicMethod {
    #[default]
    Rcd,
    Lmmse,
    Bilinear,
}

#[cfg(test)]
pub(crate) fn site(x: usize, y: usize, pattern: u32) -> usize {
    let sites = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 0, 3, 2], [2, 3, 0, 1]];
    sites[pattern as usize][(y & 1) * 2 + (x & 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rcd_is_default_and_all_methods_roundtrip() {
        assert_eq!(DemosaicMethod::default(), DemosaicMethod::Rcd);
        for method in [
            DemosaicMethod::Rcd,
            DemosaicMethod::Lmmse,
            DemosaicMethod::Bilinear,
        ] {
            assert_eq!(
                serde_json::from_str::<DemosaicMethod>(&serde_json::to_string(&method).unwrap())
                    .unwrap(),
                method
            );
        }
    }
}
