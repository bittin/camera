// SPDX-License-Identifier: GPL-3.0-only

//! Lightweight audio-level probe used by the Settings drawer to show a
//! microphone meter before recording starts.
//!
//! Mirrors the recorder's processing chain: pulsesrc → audioconvert →
//! audioresample → mono/rate caps → compressor → makeup gain → limiter →
//! output level → fakesink. The defensive resampler handles sources that
//! cannot negotiate the requested Opus-compatible rate. The shared
//! [`install_level_sync_handler`] writes peak/RMS values into a
//! [`SharedAudioLevels`] mutex that the UI then snapshots on a 100 ms tick.
//!
//! The probe is owned by `AppModel` and torn down whenever the settings
//! drawer closes, `record_audio` is toggled off, the selected device
//! changes, or a real recording starts.

use gstreamer as gst;
use gstreamer::prelude::*;
use tracing::{info, warn};

use crate::pipelines::audio_level::{
    PULSESRC_BUFFER_TIME_US, PULSESRC_SLAVE_METHOD, SharedAudioLevels, dynamics,
    install_level_sync_handler,
};

/// Running probe pipeline. Drop or call [`AudioLevelProbe::stop`] to tear down.
pub struct AudioLevelProbe {
    pipeline: gst::Pipeline,
    levels: SharedAudioLevels,
    device: Option<String>,
}

impl AudioLevelProbe {
    /// Build and start the probe. `device` is the PipeWire/PulseAudio node
    /// name (e.g. `"alsa_input.usb-…"`). Pass `None` to capture from the
    /// system default. `source_rate_hz` is the source's native sample rate
    /// (0 for "unknown") — the probe pins the capsfilter to the same Opus-
    /// compatible rate the recorder will use, so the meter reflects what the
    /// recording actually captures.
    pub fn start(device: Option<&str>, source_rate_hz: u32) -> Result<Self, String> {
        let pipeline = Self::build_pipeline(device, source_rate_hz)?;

        let levels: SharedAudioLevels = Default::default();
        install_level_sync_handler(&pipeline, &levels);

        let probe = Self {
            pipeline,
            levels,
            device: device.map(|d| d.to_string()),
        };
        probe
            .pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| format!("Failed to start probe pipeline: {e}"))?;
        Ok(probe)
    }

    /// Build in NULL state; keep probe timestamp policy distinct from recording.
    fn build_pipeline(device: Option<&str>, source_rate_hz: u32) -> Result<gst::Pipeline, String> {
        let device_str = device
            .map(|d| format!("device=\"{}\" ", d.replace('"', "\\\"")))
            .unwrap_or_default();

        let target_rate = crate::media::encoders::audio::opus_target_rate(source_rate_hz);

        // Same compressor + makeup-gain chain the recorder uses, so the meter
        // shows what the recording captures, not the raw mic. See
        // `VideoRecorder::create_audio_branch` for the threshold/ratio/gain
        // rationale.
        // `audioresample` is a defensive pass-through; see the comment in
        // `VideoRecorder::create_audio_branch` for the rationale. Dynamics
        // parameters come from `dynamics::*` so the meter reflects exactly
        // what the recording will capture.
        let desc = format!(
            "pulsesrc name=probe-src {device_str}slave-method={slave} buffer-time={buffer_time} do-timestamp=true provide-clock=false \
             ! audioconvert \
             ! audioresample \
             ! capsfilter caps=audio/x-raw,channels=1,rate={rate} \
             ! audiodynamic mode=compressor characteristics=soft-knee threshold={ct} ratio={cr} \
             ! volume volume={mg} \
             ! audiodynamic mode=compressor characteristics=hard-knee threshold={lt} ratio={lr} \
             ! level name=audio-level-output post-messages=true interval=100000000 \
             ! fakesink sync=false",
            slave = PULSESRC_SLAVE_METHOD,
            buffer_time = PULSESRC_BUFFER_TIME_US,
            rate = target_rate,
            ct = dynamics::COMPRESSOR_THRESHOLD,
            cr = dynamics::COMPRESSOR_RATIO,
            mg = dynamics::MAKEUP_GAIN,
            lt = dynamics::LIMITER_THRESHOLD,
            lr = dynamics::LIMITER_RATIO,
        );

        info!(desc = %desc, "Starting audio-level probe pipeline");

        let pipeline = gst::parse::launch(&desc)
            .map_err(|e| format!("Failed to parse probe pipeline: {e}"))?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|_| "Failed to cast probe element to Pipeline".to_string())?;

        Ok(pipeline)
    }

    /// Cloned `Arc` handle to the live level data.
    pub fn levels(&self) -> SharedAudioLevels {
        self.levels.clone()
    }

    /// The PipeWire node name the probe was started with, or `None` for
    /// the system default.
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    /// Stop the pipeline and release GStreamer resources.
    pub fn stop(self) {
        // Use the same teardown as ordinary Drop, exactly once.
        drop(self);
    }
}

impl Drop for AudioLevelProbe {
    fn drop(&mut self) {
        if let Some(bus) = self.pipeline.bus() {
            bus.unset_sync_handler();
        }
        if let Err(e) = self.pipeline.set_state(gst::State::Null) {
            warn!(error = %e, "Failed to stop probe pipeline cleanly");
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_graph_matches_recorded_processing_but_keeps_its_timestamp_policy() {
        gst::init().unwrap();
        let pipeline = AudioLevelProbe::build_pipeline(None, 44_100).unwrap();
        let mut element = pipeline.by_name("probe-src").unwrap();
        assert!(element.property::<bool>("do-timestamp"));
        assert!(!element.property::<bool>("provide-clock"));
        assert_eq!(
            element.property::<i64>("buffer-time"),
            PULSESRC_BUFFER_TIME_US
        );
        assert_eq!(
            element
                .property_value("slave-method")
                .serialize()
                .unwrap()
                .as_str(),
            PULSESRC_SLAVE_METHOD
        );
        for expected in [
            "pulsesrc",
            "audioconvert",
            "audioresample",
            "capsfilter",
            "audiodynamic",
            "volume",
            "audiodynamic",
            "level",
            "fakesink",
        ] {
            assert_eq!(element.factory().unwrap().name(), expected);
            if expected == "capsfilter" {
                let caps = element.property::<gst::Caps>("caps");
                let structure = caps.structure(0).unwrap();
                assert_eq!(structure.get::<i32>("channels").unwrap(), 1);
                assert_eq!(structure.get::<i32>("rate").unwrap(), 48_000);
            }
            if expected == "volume" {
                assert_eq!(element.property::<f64>("volume"), dynamics::MAKEUP_GAIN);
            }
            if expected == "audiodynamic" {
                let limiter = element
                    .static_pad("src")
                    .unwrap()
                    .peer()
                    .unwrap()
                    .parent_element()
                    .unwrap()
                    .factory()
                    .unwrap()
                    .name()
                    == "level";
                assert_eq!(
                    element.property::<f32>("threshold"),
                    if limiter {
                        dynamics::LIMITER_THRESHOLD
                    } else {
                        dynamics::COMPRESSOR_THRESHOLD
                    }
                );
                assert_eq!(
                    element.property::<f32>("ratio"),
                    if limiter {
                        dynamics::LIMITER_RATIO
                    } else {
                        dynamics::COMPRESSOR_RATIO
                    }
                );
            }
            if expected != "fakesink" {
                element = element
                    .static_pad("src")
                    .unwrap()
                    .peer()
                    .unwrap()
                    .parent_element()
                    .unwrap();
            }
        }
        assert_eq!(pipeline.current_state(), gst::State::Null);
        assert_eq!(pipeline.children().len(), 9);
    }

    #[test]
    fn explicit_stop_and_drop_release_bus_handler_ownership() {
        gst::init().unwrap();
        for explicit in [false, true] {
            let pipeline = gst::Pipeline::new();
            let levels: SharedAudioLevels = Default::default();
            let weak = std::sync::Arc::downgrade(&levels);
            install_level_sync_handler(&pipeline, &levels);
            let probe = AudioLevelProbe {
                pipeline: pipeline.clone(),
                levels,
                device: None,
            };
            if explicit {
                probe.stop();
            } else {
                drop(probe);
            }
            assert!(
                weak.upgrade().is_none(),
                "teardown must release the sync handler even with a retained pipeline"
            );
            assert_eq!(pipeline.current_state(), gst::State::Null);
        }
    }
}
