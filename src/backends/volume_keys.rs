// SPDX-License-Identifier: GPL-3.0-only

//! Volume-key shutter trigger via Linux evdev.
//!
//! Phone hardware exposes the volume up/down keys as input devices that
//! the Wayland compositor typically consumes for system audio, never
//! delivering them as keyboard events to the focused app. This backend
//! opens the matching `/dev/input/event*` nodes directly and forwards
//! key-down events on a channel, so the app can wire them up to capture.
//!
//! Focus-aware grab: while the camera window has keyboard focus we open
//! each volume-key device and call `EVIOCGRAB(1)` so the compositor
//! stops seeing the presses. On focus loss we close the fd, which
//! always releases the grab at the kernel level — we don't rely on
//! `EVIOCGRAB(0)` for release because some downstream kernels
//! (postmarketOS / Pixel 3a sdm670 6.19 observed) refuse the
//! ungrab ioctl with `EBUSY`, leaving the device stuck. Close-on-blur
//! is portable: `close(2)` always tears down the grab.
//!
//! Works in Flatpak as long as the manifest grants `--device=all` (or
//! `--device=input`) so `/dev/input/event*` is visible inside the
//! sandbox. Detection uses the `EVIOCGBIT` ioctl against each device fd
//! — no `/sys` access needed, which keeps the path identical between
//! native and Flatpak runs.

use std::fs::File;
use std::io::ErrorKind;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{debug, info, warn};

/// Linux evdev key codes (`linux/input-event-codes.h`).
const KEY_VOLUMEDOWN: u16 = 114;
const KEY_VOLUMEUP: u16 = 115;

/// `EV_KEY` event type.
const EV_KEY: u16 = 0x01;

/// `struct input_event` size on 64-bit Linux: `struct timeval` (2× 8-byte
/// `time_t`) + `u16 type` + `u16 code` + `s32 value` = 24 bytes. Layout is
/// kernel ABI and identical across glibc and musl.
const INPUT_EVENT_SIZE: usize = 24;

/// Buffer length covering all KEY codes we care about (volume keys are
/// 114/115; 32 bytes covers bits 0..255 inclusive).
const KEY_BITMAP_BYTES: usize = 32;

/// `EVIOCGRAB(int)` request number. `_IOW('E', 0x90, int)` packs to:
/// `(WRITE=1)<<30 | sizeof(int)<<16 | 'E'<<8 | 0x90` = `0x40044590`.
const EVIOCGRAB: libc::c_ulong = 0x40044590;

/// Poll interval for the reader threads. While the camera is unfocused
/// the threads sleep at this cadence; while focused they `poll(2)` on
/// their device fd with the same timeout so a focus-loss transition is
/// noticed within ~50 ms.
const POLL_INTERVAL_MS: libc::c_int = 50;

#[derive(Debug, Clone, Copy)]
pub enum VolumeKey {
    Up,
    Down,
}

struct Handle {
    /// Mirrors the iced window-focus state. Reader threads observe this
    /// to decide whether to keep the device open and grabbed.
    focused: Arc<AtomicBool>,
}

static HANDLE: OnceLock<Handle> = OnceLock::new();

/// Build the `EVIOCGBIT(ev, len)` ioctl request number.
const fn eviocgbit(ev: u32, len: u32) -> libc::c_ulong {
    // _IOC packing: dir<<30 | size<<16 | type<<8 | nr (64-bit Linux).
    const IOC_READ: u32 = 2;
    let ty = b'E' as u32;
    let nr = 0x20 + ev;
    ((IOC_READ << 30) | (len << 16) | (ty << 8) | nr) as libc::c_ulong
}

/// Query whether a given input-device fd reports either of the volume
/// key codes among its `EV_KEY` capabilities.
fn device_has_volume_key(fd: i32) -> bool {
    let mut buf = [0u8; KEY_BITMAP_BYTES];
    let req = eviocgbit(EV_KEY as u32, buf.len() as u32);
    // SAFETY: ioctl with a writable byte buffer of the size encoded in the
    // request number. EVIOCGBIT writes the supported-codes bitmap.
    let ret = unsafe { libc::ioctl(fd, req as _, buf.as_mut_ptr()) };
    if ret < 0 {
        return false;
    }
    bit_set(&buf, KEY_VOLUMEUP as usize) || bit_set(&buf, KEY_VOLUMEDOWN as usize)
}

fn bit_set(buf: &[u8], bit: usize) -> bool {
    buf.get(bit / 8).is_some_and(|b| (b >> (bit % 8)) & 1 == 1)
}

/// Scan `/dev/input/event*`, returning the paths of devices whose
/// capabilities include a volume key. Devices the process cannot open
/// (permission, EBUSY) are silently skipped.
fn detect_volume_devices() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let entries = match std::fs::read_dir("/dev/input") {
        Ok(entries) => entries,
        Err(e) => {
            debug!(error = %e, "Cannot scan /dev/input — volume-key shutter unavailable");
            return out;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("event") {
            continue;
        }
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                debug!(device = %path.display(), error = %e, "Could not open input device for capability probe");
                continue;
            }
        };
        if device_has_volume_key(file.as_raw_fd()) {
            info!(device = %path.display(), "Volume-key device detected");
            out.push(path);
        }
    }
    out
}

/// Start the volume-key listener. Spawns one OS thread per detected
/// device; each thread opens its device and calls `EVIOCGRAB(1)` while
/// the window has focus, and closes the fd on focus loss so the kernel
/// releases the grab.
///
/// No-op (returns an empty receiver) if no volume devices are found, or
/// if called more than once.
pub fn start() -> UnboundedReceiver<VolumeKey> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    if HANDLE.get().is_some() {
        warn!("volume_keys::start called more than once — ignoring");
        return rx;
    }
    let focused = Arc::new(AtomicBool::new(false));
    let _ = HANDLE.set(Handle {
        focused: Arc::clone(&focused),
    });
    let devices = detect_volume_devices();
    if devices.is_empty() {
        debug!("No volume-key devices found");
        return rx;
    }
    for dev_path in devices {
        let tx = tx.clone();
        let focused = Arc::clone(&focused);
        let dev_str = dev_path.to_string_lossy().into_owned();
        if let Err(e) = std::thread::Builder::new()
            .name(format!("volume-keys-{dev_str}"))
            .spawn(move || reader_thread(dev_path, tx, focused))
        {
            warn!(device = %dev_str, error = %e, "Failed to spawn volume-key reader thread");
        }
    }
    rx
}

/// Update the focus state. Reader threads observe this atomic and
/// (re)open + grab the device on `true → false → true` transitions.
///
/// No-op before [`start`] runs, or if no devices were detected.
pub fn set_focused(focused: bool) {
    let Some(handle) = HANDLE.get() else {
        info!(focused, "volume_keys::set_focused called before start()");
        return;
    };
    let prev = handle.focused.swap(focused, Ordering::AcqRel);
    if prev != focused {
        info!(focused, "volume_keys focus updated");
    }
}

/// Per-device reader thread. Lives for the process lifetime; transitions
/// between "open + grab + read events" and "closed, sleeping" driven by
/// the shared `focused` atomic.
fn reader_thread(dev_path: PathBuf, tx: UnboundedSender<VolumeKey>, focused: Arc<AtomicBool>) {
    loop {
        // Wait for the window to gain focus.
        while !focused.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS as u64));
        }
        // Open the device and grab. If either step fails (transient
        // EBUSY from another grabber, permission change, hot-unplug),
        // log and back off; the loop will retry on the next focus
        // transition.
        let file = match open_and_grab(&dev_path) {
            Ok(f) => f,
            Err(e) => {
                warn!(device = %dev_path.display(), error = %e, "Failed to open/grab volume-key device");
                // Avoid busy-looping if the failure persists across
                // focus transitions: wait a beat before re-checking.
                std::thread::sleep(Duration::from_millis(250));
                continue;
            }
        };
        debug!(device = %dev_path.display(), "Volume-key device opened and grabbed");
        if let Err(e) = run_read_loop(&file, &tx, &focused) {
            warn!(device = %dev_path.display(), error = %e, "Volume-key read loop ended with error");
        }
        // Drop `file` → close(2) → kernel releases the grab.
        drop(file);
        debug!(device = %dev_path.display(), "Volume-key device closed (focus lost)");
    }
}

fn open_and_grab(path: &Path) -> std::io::Result<File> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    let val: libc::c_int = 1;
    // SAFETY: file is owned and open; EVIOCGRAB takes an int pointer.
    let ret = unsafe { libc::ioctl(file.as_raw_fd(), EVIOCGRAB as _, &val as *const libc::c_int) };
    if ret < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(file)
}

/// Read evdev events from `file` and emit volume-key presses on `tx`
/// until the focus flag goes false or an I/O error occurs. Uses `poll(2)`
/// with a short timeout so the focus flag is observed promptly without
/// needing a signal/eventfd-based wakeup.
fn run_read_loop(
    file: &File,
    tx: &UnboundedSender<VolumeKey>,
    focused: &AtomicBool,
) -> std::io::Result<()> {
    let fd = file.as_raw_fd();
    let mut buf = [0u8; INPUT_EVENT_SIZE];
    loop {
        if !focused.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` is a single valid pollfd; nfds = 1.
        let r = unsafe { libc::poll(&mut pfd, 1, POLL_INTERVAL_MS) };
        if r < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        if r == 0 {
            // Timeout — loop and re-check focus.
            continue;
        }
        if pfd.revents & libc::POLLIN == 0 {
            // POLLERR / POLLHUP without data — bail so we re-open next focus.
            return Err(std::io::Error::other(format!(
                "poll revents 0x{:x}",
                pfd.revents
            )));
        }
        // SAFETY: blocking read after poll readiness; `fd` is open for
        // the lifetime of `file`.
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut _, buf.len()) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            match e.kind() {
                ErrorKind::Interrupted | ErrorKind::WouldBlock => continue,
                _ => return Err(e),
            }
        }
        if n as usize != INPUT_EVENT_SIZE {
            continue;
        }
        let type_ = u16::from_ne_bytes([buf[16], buf[17]]);
        let code = u16::from_ne_bytes([buf[18], buf[19]]);
        let value = i32::from_ne_bytes([buf[20], buf[21], buf[22], buf[23]]);
        if type_ != EV_KEY || value != 1 {
            continue;
        }
        let key = match code {
            KEY_VOLUMEUP => VolumeKey::Up,
            KEY_VOLUMEDOWN => VolumeKey::Down,
            _ => continue,
        };
        if tx.send(key).is_err() {
            // Receiver dropped — runtime tearing down.
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_set_handles_volume_codes() {
        let mut buf = [0u8; KEY_BITMAP_BYTES];
        buf[115 / 8] |= 1 << (115 % 8);
        assert!(bit_set(&buf, KEY_VOLUMEUP as usize));
        assert!(!bit_set(&buf, KEY_VOLUMEDOWN as usize));
        let mut buf2 = [0u8; KEY_BITMAP_BYTES];
        buf2[114 / 8] |= 1 << (114 % 8);
        assert!(bit_set(&buf2, KEY_VOLUMEDOWN as usize));
        assert!(!bit_set(&buf2, KEY_VOLUMEUP as usize));
    }

    #[test]
    fn bit_set_handles_oob() {
        let buf = [0u8; KEY_BITMAP_BYTES];
        assert!(!bit_set(&buf, 9999));
    }

    #[test]
    fn eviocgbit_matches_kernel_macro() {
        // EVIOCGBIT(EV_KEY=0x01, 32) = _IOR('E', 0x21, char[32])
        //   = (READ=2)<<30 | (size=32)<<16 | ('E'=0x45)<<8 | 0x21 = 0x80204521
        assert_eq!(eviocgbit(EV_KEY as u32, 32), 0x80204521);
    }

    #[test]
    fn eviocgrab_matches_kernel_macro() {
        // _IOW('E', 0x90, int)
        //   = (WRITE=1)<<30 | sizeof(int)<<16 | 'E'<<8 | 0x90 = 0x40044590
        assert_eq!(EVIOCGRAB, 0x40044590);
    }
}
