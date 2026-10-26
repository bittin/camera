// SPDX-License-Identifier: GPL-3.0-only

use super::*;
use std::io;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(2);

#[test]
fn global_subscriber_routes_tracing_and_log_from_workers() {
    const CHILD: &str = "CAMERA_LOGGING_GLOBAL_WIRING_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Global subscribers/loggers cannot be reset between parallel tests.
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "logging::tests::global_subscriber_routes_tracing_and_log_from_workers",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(8);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "global-logging child failed: {status}");
                return;
            }
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("global application logging did not complete");
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    struct RecordWriter(mpsc::Sender<(Vec<u8>, String)>);

    impl Write for RecordWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .send((
                    bytes.to_vec(),
                    thread::current().name().unwrap_or_default().to_owned(),
                ))
                .map_err(|_| io::ErrorKind::BrokenPipe)?;
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    use tracing_subscriber::util::SubscriberInitExt;
    let (records_tx, records_rx) = mpsc::channel();
    let (subscriber, _guard) = subscriber(
        RecordWriter(records_tx),
        tracing_subscriber::EnvFilter::new("off,camera_logging_wiring=info"),
    );
    subscriber.init();

    tracing::debug!(target: "camera_logging_wiring", "filtered tracing event");
    log::debug!(target: "camera_logging_wiring", "filtered log record");
    tracing::info!(target: "camera_logging_wiring", "main emitter");
    thread::spawn(|| {
        tracing::warn!(target: "camera_logging_wiring", "worker emitter");
        log::error!(target: "camera_logging_wiring", "worker log facade");
    })
    .join()
    .unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        tokio::spawn(async {
            tracing::info!(target: "camera_logging_wiring", "async emitter");
        })
        .await
        .unwrap();
    });

    for (level, message) in [
        ("INFO", "main emitter"),
        ("WARN", "worker emitter"),
        ("ERROR", "worker log facade"),
        ("INFO", "async emitter"),
    ] {
        let (bytes, writer_thread) = records_rx.recv_timeout(DEADLINE).unwrap();
        let output = String::from_utf8(bytes).unwrap();
        assert_eq!(writer_thread, "camera-log-writer");
        assert!(output.contains("camera_logging_wiring"), "{output}");
        assert!(
            output.contains(level) && output.contains(message),
            "{output}"
        );
    }
    assert!(matches!(
        records_rx.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
}

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

#[test]
fn standard_formatting_preserves_large_records() {
    let (finished_tx, finished_rx) = mpsc::channel();
    let value = "x".repeat(10000);
    let (subscriber, guard) = subscriber(
        CaptureWriter {
            gate: None,
            bytes: Vec::new(),
            finished: finished_tx,
        },
        tracing_subscriber::EnvFilter::new("warn"),
    );
    tracing::subscriber::with_default(subscriber, || {
        tracing::error!(value = %value, "large record");
    });
    drop(guard);
    let output = String::from_utf8(finished_rx.recv_timeout(DEADLINE).unwrap()).unwrap();
    assert!(output.contains(&value), "record was dropped or truncated");
    assert!(output.contains("large record"));
}

#[test]
fn healthy_output_preserves_order_targets_levels_and_env_filter() {
    let (finished_tx, finished_rx) = mpsc::channel();
    let (subscriber, guard) = subscriber(
        CaptureWriter {
            gate: None,
            bytes: Vec::new(),
            finished: finished_tx,
        },
        tracing_subscriber::EnvFilter::new("warn,logging_test=info"),
    );
    tracing::subscriber::with_default(subscriber, || {
        tracing::debug!(target: "logging_test", "filtered debug");
        tracing::info!(target: "other_target", "filtered info");
        tracing::info!(target: "logging_test", "first accepted");
        tracing::warn!(target: "logging_test", "second accepted");
        tracing::error!(target: "logging_test", "third accepted");
    });
    // The standard guard drains accepted records before completing shutdown.
    drop(guard);
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

#[test]
fn saturation_drops_after_128_queued_messages_without_waiting() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (continue_tx, continue_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let (subscriber, guard) = subscriber(
        CaptureWriter {
            gate: Some(GateWriter {
                entered: entered_tx,
                release: release_rx,
            }),
            bytes: Vec::new(),
            finished: finished_tx,
        },
        tracing_subscriber::EnvFilter::new("warn"),
    );
    let producer = thread::spawn(move || {
        tracing::subscriber::with_default(subscriber, || {
            tracing::error!("first record");
            continue_rx.recv_timeout(DEADLINE).unwrap();
            for _ in 0..1024 {
                tracing::error!("queued record");
            }
        });
        done_tx.send(()).unwrap();
    });
    let entered = entered_rx.recv_timeout(DEADLINE);
    let _ = continue_tx.send(());
    let done = done_rx.recv_timeout(DEADLINE);
    // Unblock the sink before draining the appender at shutdown.
    let _ = release_tx.send(());
    producer.join().unwrap();
    drop(guard);
    let output = String::from_utf8(finished_rx.recv_timeout(DEADLINE).unwrap()).unwrap();
    assert!(entered.is_ok(), "sink was not gated");
    assert!(done.is_ok(), "producer waited for the saturated sink");
    assert_eq!(output.lines().count(), 129);
    assert_eq!(output.matches("first record").count(), 1);
    assert_eq!(output.matches("queued record").count(), 128);
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
fn sink_flush_runs_only_on_appender_worker_thread() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let (subscriber, guard) = subscriber(
        FlushGateWriter(Some(GateWriter {
            entered: entered_tx,
            release: release_rx,
        })),
        tracing_subscriber::EnvFilter::new("warn"),
    );
    let producer = thread::spawn(move || {
        tracing::subscriber::with_default(subscriber, || tracing::error!("flush this event"));
        done_tx.send(()).unwrap();
    });
    let entered = entered_rx.recv_timeout(DEADLINE);
    let done = done_rx.recv_timeout(DEADLINE);
    let _ = release_tx.send(());
    producer.join().unwrap();
    drop(guard);
    assert!(entered.is_ok(), "writer did not flush accepted events");
    assert!(done.is_ok(), "producer waited for sink flush");
}
