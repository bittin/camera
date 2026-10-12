// SPDX-License-Identifier: GPL-3.0-only

/// Negotiate stream configuration only. Acquisition, allocation and start errors
/// belong to the caller and must not trigger a second initialization attempt.
pub(super) fn negotiate<T>(
    mut attempt: impl FnMut(bool) -> Result<T, String>,
) -> Result<(T, bool), String> {
    match attempt(true) {
        Ok(config) => Ok((config, true)),
        Err(dual_error) => {
            tracing::debug!(%dual_error, "Dual-stream configuration rejected; trying viewfinder only");
            attempt(false)
                .map(|config| (config, false))
                .map_err(|single_error| {
                    format!("Dual-stream: {dual_error}; single-stream: {single_error}")
                })
        }
    }
}

pub(super) fn layout_usable(invalid: bool, streams: usize, raw: bool, dual: bool) -> bool {
    !invalid
        && if dual {
            streams == 2 && raw
        } else {
            streams == 1
        }
}

pub(super) fn preview_size(width: u32, height: u32, dual: bool) -> (u32, u32) {
    if !dual || (width <= 1920 && height <= 1080) {
        return (width, height);
    }
    let scale = (1920.0 / width as f64).min(1080.0 / height as f64);
    (
        ((width as f64 * scale) as u32) & !1,
        ((height as f64 * scale) as u32) & !1,
    )
}

/// Probe roles without acquiring the camera, or prepare a capture-sized layout.
pub(super) fn prepare(
    cam: &libcamera::camera::Camera<'_>,
    dual: bool,
    size: Option<(u32, u32)>,
) -> Result<libcamera::camera::CameraConfiguration, String> {
    use libcamera::stream::StreamRole;
    let roles = if dual {
        vec![StreamRole::ViewFinder, StreamRole::Raw]
    } else {
        vec![StreamRole::ViewFinder]
    };
    let mut config = cam
        .generate_configuration(&roles)
        .ok_or_else(|| "Failed to generate camera configuration".to_string())?;
    if let Some((width, height)) = size
        && let Some(mut vf) = config.get_mut(0)
    {
        vf.set_size(libcamera::geometry::Size::new(width, height));
    }
    let status = config.validate();
    let raw = config
        .get(1)
        .is_some_and(|c| super::is_bayer_format(c.get_pixel_format()));
    if !layout_usable(status.is_invalid(), config.len(), raw, dual) {
        return Err(format!(
            "Unsupported stream layout ({status:?}): {}",
            config.to_string_repr()
        ));
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::negotiate;

    #[test]
    fn single_stream_fallback_keeps_capture_resolution() {
        assert_eq!(super::preview_size(3840, 2160, false), (3840, 2160));
        assert_eq!(super::preview_size(3840, 2160, true), (1920, 1080));
        assert_eq!(super::preview_size(640, 480, true), (640, 480));
    }

    #[test]
    fn rejects_invalid_and_adjusted_away_raw_streams() {
        assert!(!super::layout_usable(true, 2, true, true));
        assert!(!super::layout_usable(false, 1, false, true));
        assert!(!super::layout_usable(false, 2, false, true));
        assert!(super::layout_usable(false, 2, true, true));
        assert!(super::layout_usable(false, 1, false, false));
        assert!(!super::layout_usable(false, 0, false, false));
    }

    #[test]
    fn accepts_dual_stream_without_retry() {
        let mut attempts = Vec::new();
        let result = negotiate(|dual| {
            attempts.push(dual);
            Ok(42)
        });
        assert_eq!(result, Ok((42, true)));
        assert_eq!(attempts, [true]);
    }

    #[test]
    fn rejected_dual_stream_falls_back_once() {
        let mut attempts = Vec::new();
        let result = negotiate(|dual| {
            attempts.push(dual);
            if dual {
                Err("unsupported raw stream".into())
            } else {
                Ok(42)
            }
        });
        assert_eq!(result, Ok((42, false)));
        assert_eq!(attempts, [true, false]);
    }

    #[test]
    fn preserves_both_errors_when_no_configuration_works() {
        let mut attempts = Vec::new();
        let result = negotiate::<()>(|dual| {
            attempts.push(dual);
            Err(if dual {
                "unsupported raw stream"
            } else {
                "invalid viewfinder"
            }
            .into())
        });
        assert_eq!(
            result.unwrap_err(),
            "Dual-stream: unsupported raw stream; single-stream: invalid viewfinder"
        );
        assert_eq!(attempts, [true, false]);
    }
}
