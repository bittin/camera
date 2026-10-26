// SPDX-License-Identifier: GPL-3.0-only

//! Application logging through tracing-appender's bounded, lossy worker.
//! Keep the returned guard alive until application shutdown so accepted logs
//! can drain using the appender's standard shutdown behavior.

use std::io::Write;
use tracing_appender::non_blocking::{NonBlockingBuilder, WorkerGuard};

pub(crate) fn subscriber<W: Write + Send + 'static>(
    writer: W,
    env_filter: tracing_subscriber::EnvFilter,
) -> (impl tracing::Subscriber + Send + Sync, WorkerGuard) {
    let (writer, guard) = NonBlockingBuilder::default()
        .buffered_lines_limit(128)
        .lossy(true)
        .thread_name("camera-log-writer")
        .finish(writer);
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_level(true)
        .with_writer(writer)
        .log_internal_errors(false)
        .finish();
    (subscriber, guard)
}

#[cfg(test)]
#[path = "logging_tests.rs"]
mod tests;
