//! Gate inference-time tracing (`println!`) so CLI `--infer` can stay quiet by default.
//! Training and other entrypoints leave trace **on** until `set_infer_trace_quiet(true)`.

use std::sync::atomic::{AtomicU8, Ordering};

static INFER_TRACE: AtomicU8 = AtomicU8::new(1);

const QUIET: u8 = 0;

/// `true` → `infer_trace!` and related diagnostics print; `false` → suppressed.
#[inline]
pub fn infer_trace_enabled() -> bool {
    INFER_TRACE.load(Ordering::Relaxed) != QUIET
}

/// Call from CLI before loading the topic graph / brain when running quiet inference.
#[inline]
pub fn set_infer_trace_quiet(quiet: bool) {
    INFER_TRACE.store(if quiet { QUIET } else { 1 }, Ordering::Relaxed);
}
