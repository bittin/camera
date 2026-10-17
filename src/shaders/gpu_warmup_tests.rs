// SPDX-License-Identifier: GPL-3.0-only

use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};

struct CompilationThreads(Arc<Mutex<Vec<ThreadId>>>);

impl<S: tracing::Subscriber> Layer<S> for CompilationThreads {
    fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
        if event.metadata().target().ends_with("shaders::gpu_convert") {
            self.0.lock().unwrap().push(std::thread::current().id());
        }
    }
}

#[test]
fn startup_warmup_does_not_compile_on_the_async_worker() {
    let Some((_guard, _, _)) = crate::test_gpu::headless_device_with_compute() else {
        crate::test_gpu::skip_no_gpu("startup_warmup_does_not_compile_on_the_async_worker");
        return;
    };
    let worker = std::thread::current().id();
    let threads = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(CompilationThreads(Arc::clone(&threads)));
    let _subscriber = tracing::subscriber::set_default(subscriber);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(super::warmup_gpu_pipelines())
        .unwrap();
    let threads = threads.lock().unwrap();
    assert!(
        !threads.is_empty(),
        "warmup must execute the real converter"
    );
    assert!(
        threads.iter().all(|thread| *thread != worker),
        "shader compilation ran inline on the async worker"
    );
}
