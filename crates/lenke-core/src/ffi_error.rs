//! Out-of-band error channel for the C ABI — an "errno-style last-error".
//!
//! The fallible `lnk_*` functions keep their existing `null`/`-1` return
//! contract; on failure they record a structured error here, and the caller
//! retrieves it via [`lnk_last_error_json`] and rethrows it as a `LenkeError`
//! carrying the shared [`ErrorCode`](crate::error_codes::ErrorCode). The *data*
//! return is never overloaded with an error union, so the binary, zero-copy
//! Arrow carrier (`lnk_query_arrow`) is unaffected — the error rides its own
//! channel.
//!
//! **Why a thread-local is safe here:** bun:ffi calls run synchronously on the
//! single JS thread, so the slot is effectively a guarded global. We make
//! mis-attribution impossible regardless: [`begin`] *clears on entry* and
//! [`lnk_last_error_json`] *takes on read*, so a stale report from an earlier
//! call can never be paired with a later call's `null` return.
//!
//! This module carries no `serde_json` (the `gql` feature deliberately omits it
//! to keep the binary small), so the JSON report is hand-rolled.

// C-ABI boundary module: re-permit `unsafe` (denied crate-wide) but keep every raw-pointer
// op an explicit block via `unsafe_op_in_unsafe_fn`. See the note atop `ffi.rs`.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "_fallible-ffi")]
#[cfg(feature = "_fallible-ffi")]
use crate::error_codes::ErrorCode;
use std::cell::RefCell;

thread_local! {
    /// The calling thread's most recent failure, pre-rendered as a JSON report
    /// (`{"code","message","details"}`) ready to hand across the boundary.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Clear any prior error at the start of a fallible FFI call, so a `null`/`-1`
/// return can never be paired with a stale report from an earlier call.
///
/// Gated to the features whose `lnk_*` surfaces actually record errors; a
/// minimal build still exports [`lnk_last_error_json`] (it just always reports
/// "no error"), keeping the ABI stable across feature combos.
#[cfg(feature = "_fallible-ffi")]
pub(crate) fn begin() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Record a failure with structured `details` (a pre-rendered JSON object such
/// as `{"pos":12}`, or `"null"`). Call on every error path that returns a
/// failure sentinel.
#[cfg(feature = "_fallible-ffi")]
pub(crate) fn set(code: ErrorCode, message: &str, details_json: &str) {
    let mut report = String::with_capacity(message.len() + 48);
    report.push_str("{\"code\":\"");
    report.push_str(code.as_str());
    report.push_str("\",\"message\":");
    crate::jsonfmt::push_json_str(&mut report, message);
    report.push_str(",\"details\":");
    report.push_str(details_json);
    report.push('}');
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(report));
}

/// Record a failure with no structured details (`details` = `null`).
#[cfg(feature = "_fallible-ffi")]
pub(crate) fn set_code(code: ErrorCode, message: &str) {
    set(code, message, "null");
}

/// Retrieve **and clear** the calling thread's last error as a JSON document
/// (`{"code","message","details"}`). Writes the byte length to `out_len` and
/// returns a heap buffer the caller frees via `lnk_free_buf`; returns `null`
/// (and leaves `out_len` untouched) when no error is pending.
///
/// Call this immediately after a `lnk_*` function returns its `null`/`-1`
/// failure sentinel. "Take on read" resets the slot, so one failure is never
/// reported twice.
///
/// # Safety
/// `out_len` must be a valid, writable `*mut usize` (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_last_error_json(out_len: *mut usize) -> *mut u8 {
    let taken = LAST_ERROR.with(|slot| slot.borrow_mut().take());
    match taken {
        Some(json) => {
            let bytes = json.into_bytes().into_boxed_slice();
            if !out_len.is_null() {
                // SAFETY: out_len is non-null (checked) and the caller's # Safety contract requires it be writable.
                unsafe { *out_len = bytes.len() };
            }
            Box::into_raw(bytes) as *mut u8
        }
        None => std::ptr::null_mut(),
    }
}
