// SPDX-License-Identifier: GPL-3.0-only
use super::wgpu;
use std::sync::Mutex;

/// A staging allocation is reusable only after a confirmed successful map and
/// unmap. wgpu 28 records the API range before core starts mapping; core may
/// already be Idle after an error, so unconditionally unmapping is not valid.
/// Failed or abandoned attempts retire their allocation instead.
pub(crate) struct ReadbackBuffer {
    device: wgpu::Device,
    label: String,
    size: u64,
    usage: wgpu::BufferUsages,
    available: Mutex<Option<wgpu::Buffer>>,
}

impl ReadbackBuffer {
    #[cfg(test)]
    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    pub(crate) fn from_buffer(device: &wgpu::Device, buffer: wgpu::Buffer, label: &str) -> Self {
        Self {
            device: device.clone(),
            label: label.into(),
            size: buffer.size(),
            usage: buffer.usage(),
            available: Mutex::new(Some(buffer)),
        }
    }

    /// Prepare a destination BEFORE encoding a new copy. Never recreate an
    /// empty staging allocation in read(), which would report bogus zero data.
    pub(crate) fn buffer(&self) -> wgpu::Buffer {
        let mut slot = self.available.lock().unwrap();
        slot.get_or_insert_with(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&self.label),
                size: self.size,
                usage: self.usage,
                mapped_at_creation: false,
            })
        })
        .clone()
    }

    pub(crate) fn read(&self) -> Result<MappedReadback<'_>, String> {
        pollster::block_on(self.read_async())
    }

    pub(crate) async fn read_async(&self) -> Result<MappedReadback<'_>, String> {
        self.read_with_poll(|| {
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .map(|_| ())
                .map_err(|e| format!("GPU readback poll failed: {e}"))
        })
        .await
    }

    // This seam changes only polling, not the native buffer/map operation.
    async fn read_with_poll(
        &self,
        poll: impl FnOnce() -> Result<(), String>,
    ) -> Result<MappedReadback<'_>, String> {
        let buffer = self.available.lock().unwrap().take().ok_or(
            "GPU readback unavailable: prepare a new copy after a failed or pending mapping",
        )?;
        let mut mapping = Mapping {
            owner: self,
            buffer: Some(buffer),
            mapped: false,
        };
        let (tx, rx) = futures::channel::oneshot::channel();
        // A lost device can create an invalid replacement handle. Native
        // map_async reports that through the validation sink as well as the
        // callback. Convert it to an explicit diagnostic, not an uncaptured
        // panic; never install a process-wide error handler or mask unmap errors.
        let validation = {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            mapping.buffer.as_ref().unwrap().slice(..).map_async(
                wgpu::MapMode::Read,
                move |result| {
                    let _ = tx.send(result);
                },
            );
            scope.pop()
        };
        if let Some(error) = validation.await {
            return Err(format!("GPU readback map failed: {error}"));
        }
        poll()?;
        rx.await
            .map_err(|e| format!("GPU readback map channel failed: {e}"))?
            .map_err(|e| format!("GPU readback map failed: {e}"))?;
        mapping.mapped = true;
        let view = mapping
            .buffer
            .as_ref()
            .unwrap()
            .slice(..)
            .get_mapped_range();
        Ok(MappedReadback {
            view: Some(view),
            mapping,
        })
    }
}

struct Mapping<'a> {
    owner: &'a ReadbackBuffer,
    buffer: Option<wgpu::Buffer>,
    mapped: bool,
}

impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        let buffer = self.buffer.take().unwrap();
        if self.mapped {
            buffer.unmap();
            let mut slot = self.owner.available.lock().unwrap();
            // Do not overwrite a separately prepared destination.
            if slot.is_none() {
                *slot = Some(buffer);
            }
        } else {
            // destroy is idempotent, legal in pending/Idle/Active states and
            // retires backend resources without an invalid NotMapped unmap.
            // The late callback owns only the sender, never the cache.
            buffer.destroy();
        }
    }
}

pub(crate) struct MappedReadback<'a> {
    view: Option<wgpu::BufferView>,
    mapping: Mapping<'a>,
}

impl std::ops::Deref for MappedReadback<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.view.as_ref().unwrap()
    }
}

impl Drop for MappedReadback<'_> {
    fn drop(&mut self) {
        // A BufferView must die before the Mapping guard unmaps its buffer.
        drop(self.view.take());
        let _ = &self.mapping;
    }
}

#[cfg(test)]
#[path = "readback_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "readback_loss_tests.rs"]
mod loss_tests;
