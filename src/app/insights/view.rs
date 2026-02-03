// SPDX-License-Identifier: GPL-3.0-only

//! Insights drawer view for displaying diagnostic information

use crate::app::state::{AppModel, ContextPage, Message};
use crate::fl;
use cosmic::Element;
use cosmic::app::context_drawer;
use cosmic::iced::{Alignment, Length};
use cosmic::widget;

use super::types::FallbackState;

impl AppModel {
    /// Create the insights view for the context drawer
    ///
    /// Shows pipeline information, performance metrics, and format capabilities.
    pub fn insights_view(&self) -> context_drawer::ContextDrawer<'_, Message> {
        let sections = vec![
            self.build_pipeline_section().into(),
            self.build_performance_section().into(),
            self.build_formats_section().into(),
        ];

        let content: Element<'_, Message> = widget::settings::view_column(sections).into();

        context_drawer::context_drawer(content, Message::ToggleContextPage(ContextPage::Insights))
            .title(fl!("insights-title"))
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
            widget::text::body(pipeline_text)
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

        section = section.add(
            widget::settings::item::builder(fl!("insights-pipeline-full")).control(copy_button),
        );

        section = section.add(widget::settings::item_row(vec![pipeline_content.into()]));

        // Decoder fallback chain
        if !self.insights.decoder_chain.is_empty() {
            section = section.add(
                widget::settings::item::builder(fl!("insights-decoder-chain"))
                    .control(widget::Space::new(0, 0)),
            );

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

                let row = widget::row()
                    .push(widget::icon::from_name(icon_name).symbolic(true).size(16))
                    .push(widget::horizontal_space().width(Length::Fixed(8.0)))
                    .push(
                        widget::column()
                            .push(widget::text::body(decoder.name).font(cosmic::font::mono()))
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

    /// Build the Performance section
    fn build_performance_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-performance"));

        // Frame latency
        let latency_ms = self.insights.frame_latency_us as f64 / 1000.0;
        section = section.add(
            widget::settings::item::builder(fl!("insights-frame-latency"))
                .control(widget::text::body(format!("{:.2} ms", latency_ms))),
        );

        // Dropped frames
        section = section.add(
            widget::settings::item::builder(fl!("insights-dropped-frames")).control(
                widget::text::body(format!("{}", self.insights.dropped_frames)),
            ),
        );

        // Frame size
        let decoded_mb = self.insights.frame_size_decoded as f64 / (1024.0 * 1024.0);
        section = section.add(
            widget::settings::item::builder(fl!("insights-frame-size-decoded"))
                .control(widget::text::body(format!("{:.2} MB", decoded_mb))),
        );

        // Buffer processing time (time to pull sample and map buffer)
        let gst_decode_ms = self.insights.gstreamer_decode_time_us as f64 / 1000.0;
        section = section.add(
            widget::settings::item::builder(fl!("insights-decode-time-gst"))
                .control(widget::text::body(format!("{:.2} ms", gst_decode_ms))),
        );

        // Frame wrap time (zero-copy: just offset extraction)
        let copy_ms = self.insights.copy_time_us as f64 / 1000.0;
        let copy_text = if copy_ms < 0.01 {
            "< 0.01 ms (zero-copy)".to_string()
        } else {
            format!("{:.2} ms", copy_ms)
        };
        section = section.add(
            widget::settings::item::builder(fl!("insights-copy-time"))
                .control(widget::text::body(copy_text)),
        );

        // GPU upload time
        let gpu_upload_ms = self.insights.gpu_conversion_time_us as f64 / 1000.0;
        section = section.add(
            widget::settings::item::builder(fl!("insights-gpu-upload-time"))
                .control(widget::text::body(format!("{:.2} ms", gpu_upload_ms))),
        );

        // GPU upload bandwidth (based on GPU upload time)
        let bandwidth_text = if self.insights.copy_bandwidth_mbps > 0.0 {
            format!("{:.1} MB/s", self.insights.copy_bandwidth_mbps)
        } else {
            "N/A".to_string()
        };
        section = section.add(
            widget::settings::item::builder(fl!("insights-gpu-upload-bandwidth"))
                .control(widget::text::body(bandwidth_text)),
        );

        section
    }

    /// Build the Format section (current format chain)
    fn build_formats_section(&self) -> widget::settings::Section<'_, Message> {
        let mut section = widget::settings::section().title(fl!("insights-format"));

        let chain = &self.insights.format_chain;

        // Source
        section = section.add(
            widget::settings::item::builder(fl!("insights-format-source"))
                .control(widget::text::body(&chain.source)),
        );

        // Resolution
        section = section.add(
            widget::settings::item::builder(fl!("insights-format-resolution"))
                .control(widget::text::body(&chain.resolution)),
        );

        // Framerate
        section = section.add(
            widget::settings::item::builder(fl!("insights-format-framerate"))
                .control(widget::text::body(&chain.framerate)),
        );

        // Native format (what the camera sends)
        section = section.add(
            widget::settings::item::builder(fl!("insights-format-native"))
                .control(widget::text::body(&chain.native_format)),
        );

        // GStreamer output (if decoding is involved)
        if let Some(gst_output) = &chain.gstreamer_output {
            section = section.add(
                widget::settings::item::builder(fl!("insights-format-gstreamer"))
                    .control(widget::text::body(gst_output)),
            );
        }

        // WGPU processing
        section = section.add(
            widget::settings::item::builder(fl!("insights-format-wgpu"))
                .control(widget::text::body(&chain.wgpu_processing)),
        );

        section
    }
}
