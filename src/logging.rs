// SPDX-License-Identifier: GPL-3.0-only

//! Best-effort logging isolated from stdout/journal backpressure.
//!
//! Retain at most 128 queued records of 8192 bytes each, plus one in flight.
//! Cap formatter output and retained fields per live span too, not just the
//! queue: the stock fmt layer otherwise builds an unlimited thread-local String.
//! Full/disconnected queues and sink errors are silently discarded. No worker
//! guard, drain, join, or sink flush runs on producers or at shutdown; even
//! accepted records can be lost on exit. A permanently stuck sink strands only
//! the one detached writer thread and its bounded backlog.
//!
//! This is not a hard realtime guarantee: std channels/allocators have internal
//! synchronization, and caller-supplied Display/Debug implementations still
//! execute on the caller and can themselves block, allocate, or panic.

use std::io::{self, Write};
use std::sync::mpsc;

const QUEUE_CAPACITY: usize = 128;
const MAX_MESSAGE_BYTES: usize = 8192;

#[derive(Clone)]
struct NonblockingWriter(mpsc::SyncSender<Vec<u8>>);

impl Write for NonblockingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() <= MAX_MESSAGE_BYTES {
            let _ = self.0.try_send(bytes.to_vec());
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for NonblockingWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn nonblocking<W: Write + Send + 'static>(mut writer: W) -> NonblockingWriter {
    let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(QUEUE_CAPACITY);
    // Intentionally detached: neither subscriber drop nor shutdown joins I/O.
    std::thread::Builder::new()
        .name("camera-log-writer".into())
        .spawn(move || {
            for bytes in receiver {
                let _ = writer.write_all(&bytes).and_then(|()| writer.flush());
            }
        })
        .expect("could not start logging thread");
    NonblockingWriter(sender)
}

struct CappedWriter<W> {
    inner: W,
    remaining: usize,
}

impl<W: std::fmt::Write> std::fmt::Write for CappedWriter<W> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        if value.len() > self.remaining {
            return Err(std::fmt::Error);
        }
        self.inner.write_str(value)?;
        self.remaining -= value.len();
        Ok(())
    }
}

struct CappedFormat;

impl<S, N> tracing_subscriber::fmt::FormatEvent<S, N> for CappedFormat
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> tracing_subscriber::fmt::FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        writer: tracing_subscriber::fmt::format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let ansi = writer.has_ansi_escapes();
        let mut capped = CappedWriter {
            inner: writer,
            remaining: MAX_MESSAGE_BYTES,
        };
        tracing_subscriber::fmt::format()
            .with_ansi(ansi)
            .format_event(
                ctx,
                tracing_subscriber::fmt::format::Writer::new(&mut capped),
                event,
            )
    }
}

struct CappedFields;

impl<'writer> tracing_subscriber::fmt::FormatFields<'writer> for CappedFields {
    fn format_fields<R: tracing_subscriber::field::RecordFields>(
        &self,
        writer: tracing_subscriber::fmt::format::Writer<'writer>,
        fields: R,
    ) -> std::fmt::Result {
        let mut capped = CappedWriter {
            inner: writer,
            remaining: MAX_MESSAGE_BYTES,
        };
        // The fmt layer prints directly to stderr on span-field errors, even
        // with log_internal_errors(false). Keep the bounded prefix instead.
        let _ = tracing_subscriber::fmt::format::DefaultFields::new().format_fields(
            tracing_subscriber::fmt::format::Writer::new(&mut capped),
            fields,
        );
        Ok(())
    }

    fn add_fields(
        &self,
        current: &'writer mut tracing_subscriber::fmt::FormattedFields<Self>,
        fields: &tracing::span::Record<'_>,
    ) -> std::fmt::Result {
        let remaining = MAX_MESSAGE_BYTES.saturating_sub(current.fields.len());
        if remaining == 0 {
            return Ok(());
        }
        let mut capped = CappedWriter {
            inner: current.as_writer(),
            remaining,
        };
        let _ = std::fmt::Write::write_str(&mut capped, " ");
        let _ = tracing_subscriber::fmt::format::DefaultFields::new().format_fields(
            tracing_subscriber::fmt::format::Writer::new(&mut capped),
            fields,
        );
        Ok(())
    }
}

pub(crate) fn subscriber<W: Write + Send + 'static>(
    writer: W,
    env_filter: tracing_subscriber::EnvFilter,
) -> impl tracing::Subscriber + Send + Sync {
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_level(true)
        .with_writer(nonblocking(writer))
        .log_internal_errors(false)
        .fmt_fields(CappedFields)
        .event_format(CappedFormat)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    const DEADLINE: Duration = Duration::from_secs(2);

    struct GateWriter {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl Write for GateWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.entered
                .send(())
                .map_err(|_| io::ErrorKind::BrokenPipe)?;
            self.release.recv().map_err(|_| io::ErrorKind::BrokenPipe)?;
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    struct CaptureWriter {
        gate: Option<GateWriter>,
        bytes: Vec<u8>,
        finished: mpsc::Sender<Vec<u8>>,
    }

    impl Write for CaptureWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if let Some(mut gate) = self.gate.take() {
                gate.write_all(bytes)?;
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Drop for CaptureWriter {
        fn drop(&mut self) {
            let _ = self.finished.send(std::mem::take(&mut self.bytes));
        }
    }

    struct LargeDisplay(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl std::fmt::Display for LargeDisplay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for _ in 0..10000 {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                f.write_str(&"x".repeat(1024))?;
            }
            Ok(())
        }
    }

    #[test]
    fn event_formatting_stops_at_byte_budget_before_enqueue() {
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (finished_tx, finished_rx) = mpsc::channel();
        tracing::subscriber::with_default(
            subscriber(
                CaptureWriter {
                    gate: None,
                    bytes: Vec::new(),
                    finished: finished_tx,
                },
                tracing_subscriber::EnvFilter::new("warn"),
            ),
            || tracing::error!(value = %LargeDisplay(count.clone()), "large event"),
        );
        let bytes = finished_rx.recv_timeout(DEADLINE).unwrap();
        assert!(bytes.len() <= 8192);
        assert!(
            count.load(std::sync::atomic::Ordering::Relaxed) <= 9,
            "formatter built an unbounded intermediate event"
        );
    }

    #[test]
    fn span_fields_and_repeated_records_have_a_fixed_byte_budget() {
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (finished_tx, finished_rx) = mpsc::channel();
        tracing::subscriber::with_default(
            subscriber(
                CaptureWriter {
                    gate: None,
                    bytes: Vec::new(),
                    finished: finished_tx,
                },
                tracing_subscriber::EnvFilter::new("warn"),
            ),
            || {
                let span = tracing::error_span!("large", value = %LargeDisplay(count.clone()));
                for _ in 0..16 {
                    span.record(
                        "value",
                        tracing::field::display(LargeDisplay(count.clone())),
                    );
                }
            },
        );
        finished_rx.recv_timeout(DEADLINE).unwrap();
        assert!(
            count.load(std::sync::atomic::Ordering::Relaxed) <= 25,
            "span fields or updates grew without bound"
        );
    }

    #[test]
    fn oversized_messages_are_dropped_without_losing_following_messages() {
        let (finished_tx, finished_rx) = mpsc::channel();
        let mut writer = nonblocking(CaptureWriter {
            gate: None,
            bytes: Vec::new(),
            finished: finished_tx,
        });
        writer.write_all(&vec![b'x'; 8193]).unwrap();
        writer.write_all(b"healthy\n").unwrap();
        drop(writer);
        assert_eq!(finished_rx.recv_timeout(DEADLINE).unwrap(), b"healthy\n");
    }

    #[test]
    fn saturation_drops_after_128_queued_messages_without_waiting() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let mut writer = nonblocking(CaptureWriter {
            gate: Some(GateWriter {
                entered: entered_tx,
                release: release_rx,
            }),
            bytes: Vec::new(),
            finished: finished_tx,
        });
        writer.write_all(b"first\n").unwrap();
        let entered = entered_rx.recv_timeout(DEADLINE);
        let producer = thread::spawn(move || {
            for _ in 0..1024 {
                writer.write_all(b"queued\n").unwrap();
            }
            writer.flush().unwrap();
            drop(writer);
            done_tx.send(()).unwrap();
        });
        let done = done_rx.recv_timeout(DEADLINE);
        let _ = release_tx.send(());
        producer.join().unwrap();
        let bytes = finished_rx.recv_timeout(DEADLINE).unwrap();
        assert!(entered.is_ok());
        assert!(done.is_ok(), "saturated enqueue, flush, or drop blocked");
        assert_eq!(
            bytes,
            [b"first\n".as_slice(), b"queued\n".repeat(128).as_slice()].concat()
        );
    }

    struct FlushGateWriter(Option<GateWriter>);

    impl Write for FlushGateWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            if let Some(mut gate) = self.0.take() {
                gate.write_all(b"flush")?;
            }
            Ok(())
        }
    }

    #[test]
    fn sink_flush_runs_only_on_detached_writer_thread() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let producer = thread::spawn(move || {
            tracing::subscriber::with_default(
                subscriber(
                    FlushGateWriter(Some(GateWriter {
                        entered: entered_tx,
                        release: release_rx,
                    })),
                    tracing_subscriber::EnvFilter::new("warn"),
                ),
                || tracing::error!("flush this event"),
            );
            done_tx.send(()).unwrap();
        });
        let entered = entered_rx.recv_timeout(DEADLINE);
        let done = done_rx.recv_timeout(DEADLINE);
        let _ = release_tx.send(());
        producer.join().unwrap();
        assert!(entered.is_ok(), "writer did not flush accepted events");
        assert!(done.is_ok(), "producer waited for sink flush");
    }

    #[test]
    fn healthy_output_preserves_order_targets_levels_and_env_filter() {
        let (finished_tx, finished_rx) = mpsc::channel();
        tracing::subscriber::with_default(
            subscriber(
                CaptureWriter {
                    gate: None,
                    bytes: Vec::new(),
                    finished: finished_tx,
                },
                tracing_subscriber::EnvFilter::new("warn,logging_test=info"),
            ),
            || {
                tracing::debug!(target: "logging_test", "filtered debug");
                tracing::info!(target: "other_target", "filtered info");
                tracing::info!(target: "logging_test", "first accepted");
                tracing::warn!(target: "logging_test", "second accepted");
                tracing::error!(target: "logging_test", "third accepted");
            },
        );
        let output = String::from_utf8(finished_rx.recv_timeout(DEADLINE).unwrap()).unwrap();
        let lines: Vec<_> = output.lines().collect();
        assert_eq!(lines.len(), 3);
        for (line, (level, message)) in lines.iter().zip([
            ("INFO", "first accepted"),
            ("WARN", "second accepted"),
            ("ERROR", "third accepted"),
        ]) {
            assert!(
                line.contains(level) && line.contains("logging_test") && line.contains(message),
                "{line}"
            );
        }
        assert!(!output.contains("filtered"));
    }

    struct FailingWriter {
        attempts: mpsc::Sender<()>,
        fail_flush: bool,
    }

    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if !self.fail_flush {
                self.attempts.send(()).unwrap();
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "fixture write failure",
                ));
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.attempts.send(()).unwrap();
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "fixture flush failure",
            ))
        }
    }

    #[test]
    fn writer_io_errors_do_not_panic_or_disable_later_enqueues() {
        for fail_flush in [false, true] {
            let (attempts_tx, attempts_rx) = mpsc::channel();
            let mut writer = nonblocking(FailingWriter {
                attempts: attempts_tx,
                fail_flush,
            });
            writer.write_all(b"first\n").unwrap();
            attempts_rx.recv_timeout(DEADLINE).unwrap();
            writer.write_all(b"second\n").unwrap();
            attempts_rx.recv_timeout(DEADLINE).unwrap();
            drop(writer);
            assert!(matches!(
                attempts_rx.recv_timeout(DEADLINE),
                Err(mpsc::RecvTimeoutError::Disconnected)
            ));
        }
    }

    #[derive(Default)]
    struct BrokenDisplay(std::sync::atomic::AtomicBool);

    impl std::fmt::Display for BrokenDisplay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if !self.0.swap(true, std::sync::atomic::Ordering::Relaxed) {
                return Err(std::fmt::Error);
            }
            // A fallback that retries formatting can fill the unread pipe.
            f.write_str(&"stderr fallback ".repeat(1024))
        }
    }

    #[test]
    fn formatting_errors_never_fall_back_to_blocking_stderr() {
        const CHILD: &str = "CAMERA_LOGGING_FORMAT_ERROR_CHILD";
        if std::env::var_os(CHILD).is_none() {
            // An unread pipe makes any synchronous stderr fallback observable.
            // Run separately so the test harness's own stderr stays untouched.
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "logging::tests::formatting_errors_never_fall_back_to_blocking_stderr",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(8);
            loop {
                if let Some(status) = child.try_wait().unwrap() {
                    assert!(status.success(), "format-error child failed: {status}");
                    return;
                }
                if std::time::Instant::now() >= deadline {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("format-error logging blocked with an unread stderr pipe");
                }
                thread::sleep(Duration::from_millis(10));
            }
        }

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let producer = thread::spawn(move || {
            tracing::subscriber::with_default(
                subscriber(
                    GateWriter {
                        entered: entered_tx,
                        release: release_rx,
                    },
                    tracing_subscriber::EnvFilter::new("warn"),
                ),
                || {
                    tracing::error!("stall the sink");
                    continue_rx.recv_timeout(DEADLINE).unwrap();
                    for _ in 0..1024 {
                        tracing::error!(error = %BrokenDisplay::default(), "formatting failure");
                        let span =
                            tracing::error_span!("broken span", error = %BrokenDisplay::default());
                        span.record("error", tracing::field::display(BrokenDisplay::default()));
                    }
                },
            );
            done_tx.send(()).unwrap();
        });
        let entered = entered_rx.recv_timeout(DEADLINE);
        let _ = continue_tx.send(());
        let done = done_rx.recv_timeout(DEADLINE);
        let _ = release_tx.send(());
        // There can be queued events after releasing the first gated write.
        // Dropping the receiver makes those writes fail instead of blocking.
        drop(release_tx);
        producer.join().unwrap();
        assert!(entered.is_ok(), "sink was not gated");
        assert!(
            done.is_ok(),
            "format-error handling waited for sink or stderr"
        );
    }

    #[test]
    fn error_handler_logging_and_subscriber_drop_do_not_wait_for_sink() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let producer = thread::spawn(move || {
            let subscriber = subscriber(
                GateWriter {
                    entered: entered_tx,
                    release: release_rx,
                },
                tracing_subscriber::EnvFilter::new("warn"),
            );
            tracing::subscriber::with_default(subscriber, || {
                // Same synchronous error-reporting path used by UI handlers.
                tracing::error!(error = "native encoder failed", "recording failed");
            });
            done_tx.send(()).unwrap();
        });
        let entered = entered_rx.recv_timeout(DEADLINE);
        let done = done_rx.recv_timeout(DEADLINE);
        // Always unblock the real writer before asserting or joining.
        let _ = release_tx.send(());
        producer.join().unwrap();
        assert!(entered.is_ok(), "sink was never exercised");
        assert!(
            done.is_ok(),
            "error handler or subscriber drop waited for the sink"
        );
    }
}
