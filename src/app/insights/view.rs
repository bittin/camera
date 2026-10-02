// SPDX-License-Identifier: GPL-3.0-only

//! Insights drawer view for displaying diagnostic information

use crate::app::state::{AppModel, ContextPage, Message};
use crate::fl;
use cosmic::Element;
use cosmic::app::context_drawer;
use cosmic::iced::advanced::text::Wrapping;
use cosmic::iced::{Alignment, Length};
use cosmic::widget;

use super::types::FallbackState;

fn insight_item<'a>(label: impl Into<std::borrow::Cow<'a, str>>) -> InsightItem<'a> {
    InsightItem(label.into())
}

struct InsightItem<'a>(std::borrow::Cow<'a, str>);

impl<'a> InsightItem<'a> {
    fn heading(self) -> widget::Row<'a, Message, cosmic::Theme> {
        widget::settings::item_row(vec![diagnostic_text(self.0).width(Length::Fill).into()])
    }

    fn control(
        self,
        value: impl Into<Element<'a, Message>>,
    ) -> widget::Row<'a, Message, cosmic::Theme> {
        widget::settings::item_row(vec![
            diagnostic_text(self.0).width(Length::FillPortion(1)).into(),
            widget::container(value)
                .align_right(Length::FillPortion(2))
                .into(),
        ])
    }
}

/// Keep long identifiers readable without letting an unbroken word escape its bounds.
fn diagnostic_text<'a>(
    text: impl Into<std::borrow::Cow<'a, str>> + 'a,
) -> widget::text::Text<'a, cosmic::Theme, cosmic::Renderer> {
    widget::text::body(text).wrapping(Wrapping::WordOrGlyph)
}

/// Keep channel names and volume separate; live levels get their own lines.
fn audio_channel_row<'a>(
    position: &'a str,
    volume: String,
    live_rms: Option<f64>,
) -> widget::Row<'a, Message, cosmic::Theme> {
    let mut details = widget::Column::new()
        .push(diagnostic_text(volume).size(11))
        .spacing(4)
        .align_x(Alignment::End);
    if let Some(rms_db) = live_rms {
        details = details.push(audio_level(rms_db));
    }
    widget::settings::item_row(vec![
        diagnostic_text(position)
            .font(cosmic::font::mono())
            .size(12)
            .width(Length::FillPortion(1))
            .into(),
        widget::container(details)
            .align_right(Length::FillPortion(2))
            .into(),
    ])
}

fn audio_level<'a>(rms_db: f64) -> widget::Column<'a, Message, cosmic::Theme> {
    use crate::app::controls::audio_meter::{AudioMeterStyle, audio_meter};
    widget::Column::new()
        .push(audio_meter(
            rms_db,
            rms_db,
            AudioMeterStyle {
                width: 80.0,
                height: 8.0,
                show_peak: false,
            },
        ))
        .push(
            diagnostic_text(format!("{:.1} dB", rms_db))
                .size(10)
                .font(cosmic::font::mono()),
        )
        .spacing(4)
        .align_x(Alignment::End)
}

/// Theme-aware destructive/error text style.
fn error_text_style(theme: &cosmic::Theme) -> cosmic::iced::widget::text::Style {
    cosmic::iced::widget::text::Style {
        color: Some(cosmic::iced::Color::from(
            theme.cosmic().destructive_color(),
        )),
        ..Default::default()
    }
}

/// Create a status text widget, styled based on availability and active state.
fn v4l2_status_text(text: String, available: bool, is_active: bool) -> Element<'static, Message> {
    if is_active {
        diagnostic_text(text)
            .class(cosmic::theme::style::Text::Accent)
            .into()
    } else if available {
        diagnostic_text(text).into()
    } else {
        diagnostic_text(text)
            .class(cosmic::theme::style::iced::Text::Custom(error_text_style))
            .into()
    }
}

/// Check if a libcamera pixel format name matches a V4L2 FourCC string.
///
/// libcamera may use different names (e.g., "MJPEG" vs V4L2's "MJPG").
fn format_matches_fourcc(libcamera_fmt: &str, v4l2_fourcc: &str) -> bool {
    if libcamera_fmt.eq_ignore_ascii_case(v4l2_fourcc) {
        return true;
    }
    // Common aliases between libcamera and V4L2
    matches!(
        (libcamera_fmt, v4l2_fourcc),
        ("MJPEG", "MJPG") | ("MJPG", "MJPEG")
    )
}

impl AppModel {
    /// Create the insights view for the context drawer
    ///
    /// Shows pipeline information, performance metrics, and format capabilities.
    pub fn insights_view(&self) -> context_drawer::ContextDrawer<'_, Message> {
        // Capture buttons row at the top
        let capture_buttons: Element<'_, Message> = widget::Row::new()
            .push(
                widget::button::standard(fl!("insights-capture"))
                    .on_press(Message::InsightsCaptureFrames),
            )
            .push(widget::space::horizontal().width(Length::Fixed(8.0)))
            .push(
                widget::button::standard(fl!("insights-capture-burst"))
                    .on_press(Message::InsightsCaptureBurst),
            )
            .padding(8)
            .into();

        let mut sections = vec![capture_buttons, self.build_pipeline_section().into()];

        // Show backend/multistream sections when libcamera backend is active
        if !self.insights.backend_type.is_empty() {
            sections.push(self.build_backend_section().into());
        }

        if self.insights.is_multistream {
            // Dual-stream: separate Preview and Capture sections
            sections.push(self.build_preview_stream_section().into());
            if self.insights.capture_stream.is_some() {
                sections.push(self.build_capture_stream_section().into());
            }
        } else {
            // Single-stream: combined section
            sections.push(self.build_combined_stream_section().into());
        }

        // Recording section (shown when recording is active)
        if self.insights.recording_diag.is_some() {
            sections.push(self.build_recording_section().into());
        }

        // Audio section
        sections.push(self.build_audio_section().into());

        // Per-frame metadata section (libcamera only)
        if self.insights.has_libcamera_metadata {
            sections.push(self.build_metadata_section().into());
        }

        // V4L2 device formats sections (one per pixel format)
        for fmt in &self.insights.v4l2_formats {
            sections.push(self.build_v4l2_format_section(fmt).into());
        }

        let content: Element<'_, Message> = widget::settings::view_column(sections).into();

        context_drawer::context_drawer(content, Message::ToggleContextPage(ContextPage::Insights))
            .title(fl!("insights-title"))
            .actions(self.settings_back_button())
    }

    /// Build the Pipeline section
    fn build_pipeline_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-pipeline"));

        // Full GStreamer pipeline string with copy button
        let pipeline_text = self
            .insights
            .full_pipeline_string
            .as_deref()
            .unwrap_or("No pipeline active");

        let pipeline_content = widget::container(
            diagnostic_text(pipeline_text)
                .font(cosmic::font::mono())
                .size(10),
        )
        .padding(8)
        .class(cosmic::style::Container::Card)
        .width(Length::Fill);

        // Copy button
        let copy_button =
            widget::button::icon(widget::icon::from_name("edit-copy-symbolic").symbolic(true))
                .extra_small()
                .on_press(Message::CopyPipelineString);

        let pipeline_label = fl!("insights-pipeline-full-libcamera");
        section = section.add(insight_item(pipeline_label).control(copy_button));

        section = section.add(widget::settings::item_row(vec![pipeline_content.into()]));

        // Decoder fallback chain
        if !self.insights.decoder_chain.is_empty() {
            section = section.add(insight_item(fl!("insights-decoder-chain")).heading());

            for decoder in &self.insights.decoder_chain {
                let (icon_name, status_text) = match decoder.state {
                    FallbackState::Selected => ("emblem-ok-symbolic", fl!("insights-selected")),
                    FallbackState::Available => {
                        ("media-record-symbolic", fl!("insights-available"))
                    }
                    FallbackState::Unavailable => {
                        ("window-close-symbolic", fl!("insights-unavailable"))
                    }
                };

                let row = widget::Row::new()
                    .push(widget::icon::from_name(icon_name).symbolic(true).size(16))
                    .push(widget::space::horizontal().width(Length::Fixed(8.0)))
                    .push(
                        widget::Column::new()
                            .push(diagnostic_text(decoder.name).font(cosmic::font::mono()))
                            .push(
                                widget::text::caption(format!(
                                    "{} - {}",
                                    decoder.description, status_text
                                ))
                                .size(11),
                            ),
                    )
                    .align_y(Alignment::Center)
                    .padding(4);

                section = section.add(widget::settings::item_row(vec![row.into()]));
            }
        }

        section
    }

    /// Add performance metrics to a section
    fn add_performance_items<'a>(
        &self,
        mut section: widget::settings::Section<'a, Message>,
    ) -> widget::settings::Section<'a, Message> {
        // Frame latency
        let latency_ms = self.insights.frame_latency_us as f64 / 1000.0;
        section = section.add(
            insight_item(fl!("insights-frame-latency"))
                .control(diagnostic_text(format!("{:.2} ms", latency_ms))),
        );

        // Dropped frames
        section = section.add(
            insight_item(fl!("insights-dropped-frames"))
                .control(diagnostic_text(format!("{}", self.insights.dropped_frames))),
        );

        // Frame size
        let decoded_mb = self.insights.frame_size_decoded as f64 / (1024.0 * 1024.0);
        section = section.add(
            insight_item(fl!("insights-frame-size-decoded"))
                .control(diagnostic_text(format!("{:.2} MB", decoded_mb))),
        );

        // CPU decode time (turbojpeg MJPEG→I420)
        if self.insights.cpu_decode_time_us > 0 {
            let cpu_decode_ms = self.insights.cpu_decode_time_us as f64 / 1000.0;
            section = section.add(
                insight_item(fl!("insights-cpu-decode-time"))
                    .control(diagnostic_text(format!("{:.2} ms", cpu_decode_ms))),
            );
        }

        // Frame wrap time
        let copy_ms = self.insights.copy_time_us as f64 / 1000.0;
        let copy_text = if copy_ms < 0.01 {
            "< 0.01 ms (zero-copy)".to_string()
        } else {
            format!("{:.2} ms", copy_ms)
        };
        section = section
            .add(insight_item(fl!("insights-copy-time")).control(diagnostic_text(copy_text)));

        // GPU upload time
        let gpu_upload_ms = self.insights.gpu_conversion_time_us as f64 / 1000.0;
        section = section.add(
            insight_item(fl!("insights-gpu-upload-time"))
                .control(diagnostic_text(format!("{:.2} ms", gpu_upload_ms))),
        );

        // GPU upload bandwidth
        let bandwidth_text = if self.insights.copy_bandwidth_mbps > 0.0 {
            format!("{:.1} MB/s", self.insights.copy_bandwidth_mbps)
        } else {
            "N/A".to_string()
        };
        section = section.add(
            insight_item(fl!("insights-gpu-upload-bandwidth"))
                .control(diagnostic_text(bandwidth_text)),
        );

        section
    }

    /// Add format chain items to a section
    ///
    /// When `skip_resolution` is true, the Resolution and Framerate rows are
    /// omitted because they are already shown by the stream info items above.
    fn add_format_items<'a>(
        &'a self,
        mut section: widget::settings::Section<'a, Message>,
        skip_resolution: bool,
    ) -> widget::settings::Section<'a, Message> {
        let chain = &self.insights.format_chain;

        section = section.add(
            insight_item(fl!("insights-format-source")).control(diagnostic_text(&chain.source)),
        );
        if !skip_resolution {
            section = section.add(
                insight_item(fl!("insights-format-resolution"))
                    .control(diagnostic_text(&chain.resolution)),
            );
            section = section.add(
                insight_item(fl!("insights-format-framerate"))
                    .control(diagnostic_text(&chain.framerate)),
            );
        }
        section = section.add(
            insight_item(fl!("insights-format-native"))
                .control(diagnostic_text(&chain.native_format)),
        );
        if let Some(cpu_proc) = &self.insights.cpu_processing {
            section = section.add(
                insight_item(fl!("insights-cpu-processing")).control(diagnostic_text(cpu_proc)),
            );
        }
        section = section.add(
            insight_item(fl!("insights-format-wgpu"))
                .control(diagnostic_text(&chain.wgpu_processing)),
        );

        section
    }

    /// Add stream info items to a section
    fn add_stream_items<'a>(
        &'a self,
        mut section: widget::settings::Section<'a, Message>,
        stream: &'a super::types::StreamInfo,
    ) -> widget::settings::Section<'a, Message> {
        section = section
            .add(insight_item(fl!("insights-stream-role")).control(diagnostic_text(&stream.role)));
        section = section.add(
            insight_item(fl!("insights-stream-resolution"))
                .control(diagnostic_text(&stream.resolution)),
        );
        // Show configured framerate, or measured FPS from frame count
        let framerate_text = if self.insights.format_chain.framerate != "N/A"
            && !self.insights.format_chain.framerate.is_empty()
        {
            self.insights.format_chain.framerate.clone()
        } else if self.insights.measured_fps > 0.0 {
            format!("{:.1} fps (measured)", self.insights.measured_fps)
        } else {
            String::new()
        };
        if !framerate_text.is_empty() {
            section = section.add(
                insight_item(fl!("insights-format-framerate"))
                    .control(diagnostic_text(framerate_text)),
            );
        }
        section = section.add(
            insight_item(fl!("insights-stream-pixel-format"))
                .control(diagnostic_text(&stream.pixel_format)),
        );
        section = section.add(
            insight_item(fl!("insights-stream-frame-count"))
                .control(diagnostic_text(format!("{}", stream.frame_count))),
        );
        section
    }

    /// Build the combined single-stream section (Preview + Capture)
    fn build_combined_stream_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-stream-combined"));

        // Stream info if available
        let has_stream = self.insights.preview_stream.is_some();
        if let Some(stream) = &self.insights.preview_stream {
            section = self.add_stream_items(section, stream);
        }

        // Format chain (skip resolution/framerate if stream info already shows them)
        section = self.add_format_items(section, has_stream);

        // Performance metrics
        section = self.add_performance_items(section);

        section
    }

    /// Build the Preview Stream section (dual-stream mode)
    fn build_preview_stream_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-stream-preview"));

        // Stream info
        let has_stream = self.insights.preview_stream.is_some();
        if let Some(stream) = &self.insights.preview_stream {
            section = self.add_stream_items(section, stream);
        }

        // Format chain (skip resolution/framerate if stream info already shows them)
        section = self.add_format_items(section, has_stream);

        // Performance metrics (apply to preview rendering)
        section = self.add_performance_items(section);

        section
    }

    /// Build the Capture Stream section (dual-stream mode)
    fn build_capture_stream_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-stream-capture"));

        if let Some(stream) = &self.insights.capture_stream {
            section = self.add_stream_items(section, stream);

            // Source
            if !stream.source.is_empty() {
                section = section.add(
                    insight_item(fl!("insights-format-source"))
                        .control(diagnostic_text(&stream.source)),
                );
            }

            // GPU processing
            if !stream.gpu_processing.is_empty() {
                section = section.add(
                    insight_item(fl!("insights-format-wgpu"))
                        .control(diagnostic_text(&stream.gpu_processing)),
                );
            }

            // Frame size
            if stream.frame_size_bytes > 0 {
                let mb = stream.frame_size_bytes as f64 / (1024.0 * 1024.0);
                section = section.add(
                    insight_item(fl!("insights-frame-size-decoded"))
                        .control(diagnostic_text(format!("{:.2} MB", mb))),
                );
            }
        }

        section
    }

    /// Build the Recording section (active recording pipeline info + live stats)
    fn build_recording_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-recording"));

        let diag = self.insights.recording_diag.as_ref().unwrap();

        // Recording mode
        section = section
            .add(insight_item(fl!("insights-recording-mode")).control(diagnostic_text(&diag.mode)));

        // Encoder
        section = section.add(
            insight_item(fl!("insights-recording-encoder"))
                .control(diagnostic_text(&diag.encoder).font(cosmic::font::mono())),
        );

        // Resolution + Framerate on one line
        section = section.add(insight_item(fl!("insights-recording-resolution")).control(
            diagnostic_text(format!("{} @ {} fps", diag.resolution, diag.framerate)),
        ));

        // Live stats (if available)
        if let Some(stats) = &self.insights.recording_stats {
            // Capture → Channel
            section = section.add(insight_item(fl!("insights-recording-capture")).control(
                diagnostic_text(format!(
                    "{} sent, {} dropped",
                    stats.capture_sent, stats.capture_dropped
                )),
            ));

            // Channel backlog
            section = section.add(
                insight_item(fl!("insights-recording-channel"))
                    .control(diagnostic_text(format!("{} queued", stats.channel_backlog))),
            );

            // Pusher → Appsrc
            section = section.add(insight_item(fl!("insights-recording-pusher")).control(
                diagnostic_text(format!(
                    "{} pushed, {} skipped",
                    stats.pusher_pushed, stats.pusher_skipped
                )),
            ));

            // Effective FPS
            section = section.add(
                insight_item(fl!("insights-recording-fps"))
                    .control(diagnostic_text(format!("{:.1} fps", stats.effective_fps))),
            );

            // Processing delay
            if stats.last_processing_delay_us > 0 {
                let delay_ms = stats.last_processing_delay_us as f64 / 1000.0;
                section = section.add(
                    insight_item(fl!("insights-recording-delay"))
                        .control(diagnostic_text(format!("{:.1} ms", delay_ms))),
                );
            }

            // NV12 conversion time (only shown for pusher NV12 path)
            if stats.last_convert_time_us > 0 {
                let convert_ms = stats.last_convert_time_us as f64 / 1000.0;
                section = section.add(
                    insight_item(fl!("insights-recording-convert"))
                        .control(diagnostic_text(format!("{:.2} ms", convert_ms))),
                );
            }

            // Current PTS
            section = section.add(insight_item(fl!("insights-recording-pts")).control(
                diagnostic_text(format!("{:.1} s", stats.last_pts_ms as f64 / 1000.0)),
            ));
        }

        // Full pipeline string
        let pipeline_content = widget::container(
            diagnostic_text(&diag.pipeline_string)
                .font(cosmic::font::mono())
                .size(10),
        )
        .padding(8)
        .class(cosmic::style::Container::Card)
        .width(Length::Fill);

        section = section.add(insight_item(fl!("insights-recording-pipeline")).heading());
        section = section.add(widget::settings::item_row(vec![pipeline_content.into()]));

        section
    }

    /// Build the per-frame metadata section (libcamera only)
    ///
    /// Shows all metadata fields with "N/A" when a value is not reported by the ISP.
    fn build_metadata_section(&self) -> widget::settings::Section<'_, Message> {
        let na = fl!("insights-meta-na");
        let mut section = widget::settings::section().title(fl!("insights-metadata"));

        // Exposure
        let text = match self.insights.meta_exposure_us {
            Some(us) if us >= 1_000_000 => format!("{:.2} s", us as f64 / 1_000_000.0),
            Some(us) if us >= 1_000 => format!("{:.2} ms", us as f64 / 1_000.0),
            Some(us) => format!("{} \u{00b5}s", us),
            None => na.clone(),
        };
        section =
            section.add(insight_item(fl!("insights-meta-exposure")).control(diagnostic_text(text)));

        // Analogue Gain
        section = section.add(
            insight_item(fl!("insights-meta-analogue-gain")).control(diagnostic_text(
                self.insights
                    .meta_analogue_gain
                    .map_or_else(|| na.clone(), |g| format!("{:.2}x", g)),
            )),
        );

        // Digital Gain
        section = section.add(
            insight_item(fl!("insights-meta-digital-gain")).control(diagnostic_text(
                self.insights
                    .meta_digital_gain
                    .map_or_else(|| na.clone(), |g| format!("{:.2}x", g)),
            )),
        );

        // Colour Temperature
        section = section.add(
            insight_item(fl!("insights-meta-colour-temp")).control(diagnostic_text(
                self.insights
                    .meta_colour_temperature
                    .map_or_else(|| na.clone(), |t| format!("{} K", t)),
            )),
        );

        // WB Gains
        section = section.add(
            insight_item(fl!("insights-meta-colour-gains")).control(diagnostic_text(
                self.insights
                    .meta_colour_gains
                    .map_or_else(|| na.clone(), |g| format!("{:.2}, {:.2}", g[0], g[1])),
            )),
        );

        // Black Level
        section = section.add(
            insight_item(fl!("insights-meta-black-level")).control(diagnostic_text(
                self.insights
                    .meta_black_level
                    .map_or_else(|| na.clone(), |bl| format!("{:.4}", bl)),
            )),
        );

        // Illuminance (Lux)
        section = section.add(
            insight_item(fl!("insights-meta-lux")).control(diagnostic_text(
                self.insights
                    .meta_lux
                    .map_or_else(|| na.clone(), |l| format!("{:.0} lux", l)),
            )),
        );

        // Lens Position
        section = section.add(
            insight_item(fl!("insights-meta-lens-position")).control(diagnostic_text(
                self.insights
                    .meta_lens_position
                    .map_or_else(|| na.clone(), |p| format!("{:.2} dioptres", p)),
            )),
        );

        // Focus FoM
        section = section.add(
            insight_item(fl!("insights-meta-focus-fom")).control(diagnostic_text(
                self.insights
                    .meta_focus_fom
                    .map_or_else(|| na.clone(), |f| format!("{}", f)),
            )),
        );

        // Sequence
        section = section.add(
            insight_item(fl!("insights-meta-sequence")).control(diagnostic_text(
                self.insights
                    .meta_sequence
                    .map_or_else(|| na.clone(), |s| format!("{}", s)),
            )),
        );

        section
    }

    /// Build the Audio section showing audio device, pipeline, per-channel details, and live levels
    fn build_audio_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-audio"));

        // Recording enabled/disabled
        let status = if self.config.record_audio {
            fl!("insights-audio-enabled")
        } else {
            fl!("insights-audio-disabled")
        };
        section = section
            .add(insight_item(fl!("insights-audio-recording")).control(diagnostic_text(status)));

        // Selected audio device details
        let dev = self
            .available_audio_devices
            .get(self.current_audio_device_index);

        if let Some(dev) = dev {
            let name = if dev.is_default {
                format!("{} {}", dev.name, fl!("insights-audio-default"))
            } else {
                dev.name.clone()
            };
            section = section
                .add(insight_item(fl!("insights-audio-device")).control(diagnostic_text(name)));

            // Audio device node name (monospace)
            let node_content = widget::container(
                diagnostic_text(&dev.node_name)
                    .font(cosmic::font::mono())
                    .size(11),
            )
            .padding(4);
            section = section.add(insight_item(fl!("insights-audio-node")).control(node_content));

            // Native format info
            if !dev.sample_format.is_empty() {
                let format_text = format!(
                    "{} / {} Hz / {}ch",
                    dev.sample_format,
                    dev.sample_rate,
                    dev.channels.len()
                );
                section = section.add(
                    insight_item(fl!("insights-audio-format"))
                        .control(diagnostic_text(format_text)),
                );
            }
        }

        // Audio codec
        let codec = if gstreamer::ElementFactory::find("opusenc").is_some() {
            "Opus"
        } else if gstreamer::ElementFactory::find("avenc_aac").is_some()
            || gstreamer::ElementFactory::find("faac").is_some()
            || gstreamer::ElementFactory::find("voaacenc").is_some()
        {
            "AAC"
        } else {
            "None"
        };
        section =
            section.add(insight_item(fl!("insights-audio-codec")).control(diagnostic_text(codec)));

        // Output channels (always mono)
        section = section.add(
            insight_item(fl!("insights-audio-channels"))
                .control(diagnostic_text(fl!("insights-audio-mono"))),
        );

        // Pipeline chain description
        let pipeline_desc = if self.config.record_audio {
            format!(
                "pulsesrc \u{2192} queue \u{2192} audioconvert \u{2192} audioresample \u{2192} level \u{2192} capsfilter(mono) \u{2192} level \u{2192} {}",
                if codec == "Opus" {
                    "opusenc"
                } else if codec == "AAC" {
                    "avenc_aac"
                } else {
                    "none"
                }
            )
        } else {
            fl!("insights-audio-disabled")
        };
        let pipeline_content = widget::container(
            diagnostic_text(pipeline_desc)
                .font(cosmic::font::mono())
                .size(10),
        )
        .padding(8)
        .class(cosmic::style::Container::Card)
        .width(Length::Fill);
        section = section.add(insight_item(fl!("insights-audio-pipeline")).heading());
        section = section.add(widget::settings::item_row(vec![pipeline_content.into()]));

        // Per-channel input details from audio device info
        if let Some(dev) = dev
            && !dev.channels.is_empty()
        {
            section = section.add(insight_item(fl!("insights-audio-inputs")).heading());

            let levels = &self.insights.audio_levels;

            for (i, ch) in dev.channels.iter().enumerate() {
                // Get live level for this channel (if recording)
                let live_rms = levels.as_ref().and_then(|l| l.input_rms_db.get(i).copied());

                let vol_text = format!("{:.1} dB", ch.volume_db,);

                section = section.add(widget::settings::item_row(vec![
                    audio_channel_row(&ch.position, vol_text, live_rms).into(),
                ]));
            }
        }

        // Mono output level (after mix)
        if let Some(levels) = &self.insights.audio_levels {
            section = section.add(insight_item(fl!("insights-audio-output-level")).heading());
            section = section.add(
                insight_item(fl!("insights-audio-mono")).control(audio_level(levels.output_rms_db)),
            );
        } else if self.recording.is_recording() && self.config.record_audio {
            // Recording but no levels yet
            section = section.add(
                insight_item(fl!("insights-audio-output-level")).control(diagnostic_text("...")),
            );
        }

        section
    }

    /// Build a V4L2 format section for a single pixel format (e.g., MJPG, YUYV, H264).
    /// Each resolution+framerate combination gets its own row.
    fn build_v4l2_format_section(
        &self,
        fmt: &crate::backends::camera::v4l2_utils::V4l2FormatInfo,
    ) -> widget::settings::Section<'_, Message> {
        let title = format!("{} ({})", fmt.fourcc.trim(), fmt.description);
        let mut section = widget::settings::section().title(title);
        let fourcc_trimmed = fmt.fourcc.trim();

        // Get the actual running stream info for highlighting.
        // Use preview_stream (the actual negotiated format) rather than active_format
        // (user-selected format) since libcamera may negotiate differently
        // (e.g., user selects YUYV but libcamera uses MJPEG for better framerate).
        let stream = self.insights.preview_stream.as_ref();

        let mut sorted_sizes = fmt.sizes.clone();
        sorted_sizes.sort_by_key(|b| std::cmp::Reverse(b.width * b.height));

        for size in &sorted_sizes {
            let in_libcamera = self.insights.libcamera_formats.iter().any(|lf| {
                lf.width == size.width
                    && lf.height == size.height
                    && format_matches_fourcc(&lf.pixel_format, fourcc_trimmed)
            });

            // Check if this resolution+format matches the actual running stream.
            // The stream pixel_format may be e.g. "I422 (MJPEG)" or "NV12",
            // so check if it contains the V4L2 fourcc or its alias.
            let is_active_resolution = stream.is_some_and(|s| {
                let res_parts: Vec<&str> = s.resolution.split('x').collect();
                let matches_res = res_parts.len() == 2
                    && res_parts[0].parse::<u32>().ok() == Some(size.width)
                    && res_parts[1].parse::<u32>().ok() == Some(size.height);
                let stream_fmt = &s.pixel_format;
                let matches_fmt = stream_fmt.eq_ignore_ascii_case(fourcc_trimmed)
                    || stream_fmt.contains(fourcc_trimmed)
                    || (fourcc_trimmed == "MJPG"
                        && (stream_fmt.contains("MJPEG") || stream_fmt.contains("MJPG")))
                    || (fourcc_trimmed == "YUYV" && stream_fmt.contains("YUYV"));
                matches_res && matches_fmt
            });

            let status_text = if is_active_resolution {
                format!("\u{2713} {}", fl!("insights-v4l2-active-in-libcamera"))
            } else if in_libcamera {
                format!("\u{2713} {}", fl!("insights-v4l2-in-libcamera"))
            } else {
                format!("\u{2717} {}", fl!("insights-v4l2-not-in-libcamera"))
            };

            let resolution_label = format!("{}x{}", size.width, size.height);

            if size.framerates.is_empty() {
                let status_widget =
                    v4l2_status_text(status_text, in_libcamera, is_active_resolution);
                section = section.add(insight_item(resolution_label).control(status_widget));
            } else {
                for &(num, denom) in &size.framerates {
                    let fps = if num > 0 {
                        let fps = denom as f64 / num as f64;
                        if fps == fps.round() {
                            format!("{} fps", fps as u32)
                        } else {
                            format!("{:.2} fps", fps)
                        }
                    } else {
                        "? fps".to_string()
                    };

                    let row_status =
                        v4l2_status_text(status_text.clone(), in_libcamera, is_active_resolution);
                    section = section.add(
                        insight_item(format!("{} @ {}", resolution_label, fps)).control(row_status),
                    );
                }
            }
        }

        section
    }

    /// Build the Backend section (libcamera-specific info)
    fn build_backend_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-backend"));

        // Backend type
        section = section.add(
            insight_item(fl!("insights-backend-type"))
                .control(diagnostic_text(self.insights.backend_type)),
        );

        // Pipeline handler
        if let Some(handler) = &self.insights.pipeline_handler {
            section = section.add(
                insight_item(fl!("insights-pipeline-handler")).control(diagnostic_text(handler)),
            );
        }

        // Sensor model
        if let Some(sensor) = &self.insights.sensor_model {
            section = section
                .add(insight_item(fl!("insights-sensor-model")).control(diagnostic_text(sensor)));
        }

        // libcamera version
        if let Some(version) = &self.insights.libcamera_version {
            section = section.add(
                insight_item(fl!("insights-libcamera-version")).control(diagnostic_text(version)),
            );
        }

        // MJPEG decoder (shown when native libcamera decodes MJPEG)
        if let Some(decoder) = &self.insights.mjpeg_decoder {
            section = section
                .add(insight_item(fl!("insights-mjpeg-decoder")).control(diagnostic_text(decoder)));
        }

        // Stream mode: label is Single/Dual-stream, control shows source assignment
        let (mode_label, source_text) = if self.insights.is_multistream {
            (
                fl!("insights-multistream-dual"),
                fl!("insights-multistream-source-separate"),
            )
        } else {
            (
                fl!("insights-multistream-single"),
                fl!("insights-multistream-source-shared"),
            )
        };
        section = section.add(insight_item(mode_label).control(diagnostic_text(source_text)));

        section
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosmic::iced::advanced::{layout, renderer::Headless, widget::Tree};
    use cosmic::iced::{Font, Pixels, Size};

    fn renderer() -> cosmic::Renderer {
        pollster::block_on(cosmic::Renderer::new(
            Font::DEFAULT,
            Pixels(14.0),
            Some("tiny-skia"),
        ))
        .expect("headless software renderer for text layout")
    }

    fn assert_content_fits(node: &layout::Node) {
        for child in node.children() {
            let bounds = child.bounds();
            assert!(bounds.x >= -0.01);
            assert!(bounds.y >= -0.01);
            assert!(
                bounds.x + bounds.width <= node.size().width + 0.01,
                "child {bounds:?} exceeds parent {:?}",
                node.size()
            );
            assert!(bounds.y + bounds.height <= node.size().height + 0.01);
            assert_content_fits(child);
        }
    }

    #[test]
    fn unbroken_device_identifiers_wrap_inside_their_value_column() {
        let renderer = renderer();
        for width in [132.0, 180.0, 264.0, 368.0] {
            let mut element: Element<'_, Message> = insight_item("Node")
                .control(
                    widget::container(
                        diagnostic_text(
                            "alsa_input_usb_Remo_Tech_Co__Ltd_OBSBOT_Tiny_2_02_analog_stereo",
                        )
                        .font(cosmic::font::mono())
                        .size(11),
                    )
                    .padding(4),
                )
                .into();
            let mut tree = Tree::new(element.as_widget());
            let node = element.as_widget_mut().layout(
                &mut tree,
                &renderer,
                &layout::Limits::new(Size::ZERO, Size::new(width, f32::INFINITY)),
            );
            assert_content_fits(&node);
            assert!(
                node.size().height > 30.0,
                "identifier did not wrap at {width}"
            );
        }
    }

    #[test]
    fn audio_channels_reserve_space_for_full_names() {
        let renderer = renderer();
        for width in [132.0, 180.0, 264.0, 368.0, 600.0] {
            for level in [None, Some(-12.0)] {
                let mut element: Element<'_, Message> =
                    audio_channel_row("FrontRight", "0.0 dB".into(), level).into();
                let mut tree = Tree::new(element.as_widget());
                let node = element.as_widget_mut().layout(
                    &mut tree,
                    &renderer,
                    &layout::Limits::new(Size::ZERO, Size::new(width, f32::INFINITY)),
                );
                assert!(
                    node.children()[0].bounds().width >= width / 4.0,
                    "channel name squeezed at {width}"
                );
                assert_content_fits(&node);
            }
        }
    }

    #[test]
    fn diagnostic_rows_reserve_label_space_at_narrow_widths() {
        let renderer = renderer();
        for width in [132.0, 180.0, 264.0, 368.0, 600.0] {
            for (label, value) in [
                ("Device", "OBSBOT Tiny 2 Analog Stereo (Default)"),
                ("MJPEG Decoder", "turbojpeg (libjpeg-turbo, 2 workers)"),
                (
                    "GPU Processing",
                    "I420 (YUV 4:2:0) -> RGBA (compute shader)",
                ),
            ] {
                let mut element: Element<'_, Message> =
                    insight_item(label).control(diagnostic_text(value)).into();
                let mut tree = Tree::new(element.as_widget());
                let node = element.as_widget_mut().layout(
                    &mut tree,
                    &renderer,
                    &layout::Limits::new(Size::ZERO, Size::new(width, f32::INFINITY)),
                );
                let label_bounds = node.children()[0].bounds();
                let value_bounds = node.children()[1].bounds();
                assert!(
                    label_bounds.width >= width / 4.0,
                    "{label} squeezed at {width}: {label_bounds:?}"
                );
                assert!(label_bounds.x + label_bounds.width <= value_bounds.x);
                assert!(value_bounds.x + value_bounds.width <= width + 0.01);
                assert_content_fits(&node);
            }
        }
    }
}
