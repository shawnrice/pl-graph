//! Out-of-band error channel for the engine's C ABI — an "errno-style last-error".
//!
//! The fallible `lnk_*` functions keep their `null` / `-1` return contract; on
//! failure they record a structured report here, and the host retrieves it via
//! [`lnk_last_error_json`]. The data return is never overloaded with an error
//! union, so a binary carrier (Arrow) rides its own channel unaffected.
//!
//! **Why a thread-local is safe:** bun:ffi and wasm both call synchronously on a
//! single thread, so the slot is a guarded global. We make mis-attribution
//! impossible regardless: [`begin`] clears on entry and [`lnk_last_error_json`]
//! takes on read, so a stale report can never be paired with a later `null`.
//!
//! The engine deliberately carries no `serde_json`, so the JSON report is
//! hand-rolled. Engine errors are plain `String` (no shared `ErrorCode` enum yet),
//! so the `code` is a short caller-supplied slug like `E_FFI` / `E_UNIMPLEMENTED`.

// C-ABI boundary module: re-permit `unsafe` and keep every raw-pointer op explicit.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;

thread_local! {
    /// The calling thread's most recent failure, pre-rendered as a JSON report
    /// (`{"code","message","details"}`) ready to hand across the boundary.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Clear any prior error at the start of a fallible call, so a `null` / `-1`
/// return can never be paired with a stale report from an earlier call.
pub(crate) fn begin() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Record a failure. `code` is a short stable slug (`E_FFI`, `E_UNIMPLEMENTED`,
/// `E_CONSTRAINT`, …); `message` is human-readable and is JSON-escaped here.
pub(crate) fn set(code: &str, message: &str) {
    let mut report = String::with_capacity(message.len() + 48);
    report.push_str("{\"code\":\"");
    report.push_str(code);
    report.push_str("\",\"message\":\"");
    push_escaped(&mut report, message);
    report.push_str("\",\"details\":null}");
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(report));
}

/// Append `s` to `out` with the minimal JSON string escaping (RFC 8259).
fn push_escaped(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str("\\u");
                for shift in [12, 8, 4, 0] {
                    let nib = ((c as u32) >> shift) & 0xf;
                    out.push(char::from_digit(nib, 16).unwrap());
                }
            }
            c => out.push(c),
        }
    }
}

/// Retrieve **and clear** the calling thread's last error as a JSON document
/// (`{"code","message","details"}`). Writes the byte length to `out_len` and
/// returns a heap buffer freed via [`crate::ffi::lnk_free`]; returns `null` (and
/// leaves `out_len` untouched) when no error is pending.
///
/// Call immediately after a `lnk_*` function returns its `null` / `-1` sentinel.
/// "Take on read" resets the slot, so one failure is never reported twice.
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
                // SAFETY: out_len is non-null (checked) and the caller's contract requires it writable.
                unsafe { *out_len = bytes.len() };
            }
            Box::into_raw(bytes) as *mut u8
        }
        None => std::ptr::null_mut(),
    }
}
