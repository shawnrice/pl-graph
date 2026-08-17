//! The engine's C ABI — a deliberately small, flat exported surface (16 symbols)
//! that `packages/native` can load in place of `lenke-core`.
//!
//! Design (see `docs/abi.md`):
//!   * **Pare down external symbols.** Variant families are folded behind an enum
//!     argument (`lnk_query(lang, format)`, `lnk_tx(action)`, `lnk_stat(which)`,
//!     `lnk_open`/`lnk_encode(format)`) rather than a symbol per variant.
//!   * **Exotic tiers, simple ABI.** Prepared statements, Arrow, CDC scope,
//!     binary snapshots, fork/merge all ride one generic [`lnk_command`] — adding
//!     a feature fills a match arm, never a new export.
//!
//! The handle is an engine [`Store`]. Buffers the engine returns are freed with
//! [`lnk_free`]; host-owned input memory (wasm) is [`lnk_alloc`]/[`lnk_dealloc`].
//! Errors ride the out-of-band channel in [`crate::ffi_error`].
//!
//! Failures ride the error channel with a canonical `E_*` code from the shared
//! `@lenke/errors` vocabulary (`E_SYNTAX`, `E_CONSTRAINT_VIOLATION`,
//! `E_INVALID_VALUE`, `E_UNSUPPORTED`, …) so the host maps them to the same
//! `LenkeError` the core backend produces.

// C-ABI boundary module: keep every raw-pointer op an explicit `unsafe {}` block.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use crate::store::Store;

/// The ABI version the host asserts in lockstep (see `packages/native/src/abi.ts`).
pub const ABI_VERSION: u32 = 18;

// ------------------------------------------------------------------ helpers ---

/// Borrow a `Store` immutably (null → `None`).
///
/// # Safety
/// If non-null, `s` must point to a live `Store` not mutably aliased for `'a`.
unsafe fn store_ref<'a>(s: *const Store) -> Option<&'a Store> {
    // SAFETY: as_ref() yields None for null; else the caller's contract requires a live, non-aliased Store.
    unsafe { s.as_ref() }
}

/// Borrow a `Store` mutably (null → `None`).
///
/// # Safety
/// If non-null, `s` must point to a live `Store` not otherwise aliased for `'a`.
unsafe fn store_mut<'a>(s: *mut Store) -> Option<&'a mut Store> {
    // SAFETY: as_mut() yields None for null; else the caller's contract requires exclusive access.
    unsafe { s.as_mut() }
}

/// Borrow a caller-owned byte buffer as UTF-8 `&str` (null or non-UTF-8 → `None`).
///
/// # Safety
/// If non-null, `ptr`/`len` must describe a readable, initialized range valid for `'a`.
unsafe fn in_str<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr non-null (checked); the caller's contract requires a readable, initialized range for 'a.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).ok()
}

/// Borrow a caller-owned byte buffer (arbitrary bytes, not UTF-8; null → `None`).
///
/// # Safety
/// If non-null, `ptr`/`len` must describe a readable, initialized range valid for `'a`.
unsafe fn in_bytes<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr non-null (checked); the caller's contract requires a readable range for 'a.
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Hand a heap `String` back as a `(ptr, len)` buffer freed with [`lnk_free`].
///
/// # Safety
/// `out_len` must be a valid, writable pointer.
unsafe fn out_string(s: String, out_len: *mut usize) -> *mut u8 {
    let bytes = s.into_bytes().into_boxed_slice();
    // SAFETY: the caller's contract requires out_len writable.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// Run a query closure behind a panic backstop: a fault fails this ONE call with
/// an error, never unwinds across the `extern "C"` boundary (which is UB).
fn guarded<T, F: FnOnce() -> Result<T, String>>(f: F) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => Err("query evaluation panicked".to_string()),
    }
}

/// Hand a heap `Vec<u8>` back as a `(ptr, len)` buffer freed with [`lnk_free`].
///
/// # Safety
/// `out_len` must be a valid, writable pointer.
unsafe fn out_bytes(bytes: Vec<u8>, out_len: *mut usize) -> *mut u8 {
    let boxed = bytes.into_boxed_slice();
    // SAFETY: the caller's contract requires out_len writable.
    unsafe { *out_len = boxed.len() };
    Box::into_raw(boxed) as *mut u8
}

/// Parse a prepared handle out of a `{"handle":"<h>"}` field (a decimal string,
/// since a generational handle can exceed a JS f64).
fn parse_handle_field(fields: &[(String, crate::ndjson::Json)]) -> Result<u64, String> {
    let j = crate::ndjson::req(fields, "handle")?;
    let s = crate::ndjson::json_string(j)?;
    s.parse::<u64>()
        .map_err(|_| format!("invalid prepared handle `{s}`"))
}

/// A decoded `prepared_run` payload: the handle, its params, and the output format.
type PreparedRun = (u64, Vec<(String, crate::value::Value)>, String);

/// Parse a `prepared_run` payload `{"handle":"<h>", "params":{…}, "format":"…"}`.
/// `format` is `json` (default), `arrow`, or `arrow_ipc`.
fn prepared_payload(input: Option<&str>) -> Result<PreparedRun, String> {
    let text = input.ok_or("prepared_run requires a {handle, params} JSON payload")?;
    let crate::ndjson::Json::Obj(fields) = crate::ndjson::parse_json(text)? else {
        return Err("prepared_run payload must be a JSON object".into());
    };
    let handle = parse_handle_field(&fields)?;
    let params = match crate::ndjson::field(&fields, "params") {
        Some(p) => crate::ndjson::params_from_obj(p)?,
        None => Vec::new(),
    };
    let format = match crate::ndjson::field(&fields, "format") {
        Some(f) => crate::ndjson::json_string(f)?,
        None => "json".to_string(),
    };
    Ok((handle, params, format))
}

/// Fold a camelCase identifier to snake_case: each uppercase letter becomes `_` +
/// its lowercase. Idempotent on a snake_case input (no uppercase to fold), so it
/// maps `connectedComponents` → `connected_components` while leaving
/// `connected_components` unchanged. Used to accept both algo-name spellings.
fn camel_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Run the `algo` command: validate the config, run the procedure, optionally
/// write each result to `writeProperty`, and render the `{columns, rows}` JSON.
fn run_algo_command(
    store: &mut Store,
    fields: &[(String, crate::ndjson::Json)],
) -> Result<String, String> {
    // The shared Backend / direct-algo methods send the camelCase procedure name
    // (`connectedComponents`, `labelPropagation`, …); GQL `CALL` uses the engine's
    // snake_case (`connected_components`). Normalize camelCase → snake_case here so
    // BOTH spellings dispatch — the conversion is idempotent on a snake_case name
    // (no uppercase to fold), so CALL is unaffected.
    let raw_name = crate::ndjson::json_string(crate::ndjson::req(fields, "name")?)?;
    let name = camel_to_snake(&raw_name);
    let config = match crate::ndjson::field(fields, "config") {
        Some(o) => crate::ndjson::params_from_obj(o)?,
        None => Vec::new(),
    };
    crate::algo::validate_config(&config)?;
    let column = crate::algo::procedure_result_col(&name)
        .ok_or_else(|| format!("unknown algorithm `{raw_name}`"))?;
    let results = crate::algo::run_procedure(store, &name, &config)
        .ok_or_else(|| format!("algorithm `{name}` rejected its config"))?;

    // A `writeProperty` writes each result back to that node property (as core does).
    let write_prop = config.iter().find_map(|(k, v)| {
        if k == "writeProperty" {
            if let crate::value::Value::Str(s) = v {
                return Some(s.to_string());
            }
        }
        None
    });
    if let Some(prop) = &write_prop {
        for (node, val) in &results {
            store.set_prop(*node, prop, val.clone());
        }
    }

    let mut out = String::with_capacity(results.len() * 24 + 32);
    out.push_str("{\"columns\":[\"node\",");
    crate::ndjson::encode_string(&mut out, column);
    out.push_str("],\"rows\":[");
    for (i, (node, val)) in results.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        crate::ndjson::encode_string(&mut out, &store.node_ext_id(*node).unwrap_or_default());
        out.push(',');
        crate::ndjson::encode_value(&mut out, val);
        out.push(']');
    }
    out.push_str("]}");
    Ok(out)
}

/// Parse a `prepared_free` payload `{"handle":"<h>"}`.
fn prepared_handle(input: Option<&str>) -> Result<u64, String> {
    let text = input.ok_or("prepared_free requires a {handle} JSON payload")?;
    let crate::ndjson::Json::Obj(fields) = crate::ndjson::parse_json(text)? else {
        return Err("prepared_free payload must be a JSON object".into());
    };
    parse_handle_field(&fields)
}

/// Report a write/eval error on the error channel with its canonical wire code.
///
/// The exec layer signals over `Result<_, String>`; a constraint violation carries
/// an `E_*:`-prefixed message. Map the known prefixes to the shared `@lenke/errors`
/// codes (so a constraint failure surfaces as `E_CONSTRAINT_VIOLATION`, matching
/// lenke-core), stripping the prefix from the human message; anything else is a
/// generic `E_INVALID_VALUE` evaluation error.
fn set_exec_error(e: &str) {
    // Every constraint kind funnels to the one canonical violation code.
    const CONSTRAINT: &[&str] = &[
        "E_UNIQUE",
        "E_REQUIRED",
        "E_TYPE",
        "E_CARDINALITY",
        "E_VALIDATOR",
        "E_INVARIANT",
    ];
    if let Some((prefix, rest)) = e.split_once(": ") {
        if CONSTRAINT.contains(&prefix) {
            crate::ffi_error::set("E_CONSTRAINT_VIOLATION", rest);
            return;
        }
        match prefix {
            "E_INVALID_GRAPH_OP" => {
                crate::ffi_error::set("E_INVALID_GRAPH_OP", rest);
                return;
            }
            "E_MISSING_PARAMETER" => {
                crate::ffi_error::set("E_MISSING_PARAMETER", rest);
                return;
            }
            _ => {}
        }
    }
    crate::ffi_error::set("E_INVALID_VALUE", e);
}

// ----------------------------------------------------------------- plumbing ---

/// The ABI version this artifact implements.
#[no_mangle]
pub extern "C" fn lnk_abi_version() -> u32 {
    ABI_VERSION
}

/// Allocate `len` bytes of engine-owned scratch for the host to fill (wasm input
/// path). Free with [`lnk_dealloc`]. The bytes are uninitialized.
#[no_mangle]
pub extern "C" fn lnk_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Release a buffer from [`lnk_alloc`].
///
/// # Safety
/// `ptr`/`len` must be a prior [`lnk_alloc`] result, freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn lnk_dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        // SAFETY: ptr came from lnk_alloc with this capacity; reconstruct with len 0 so drop frees the capacity.
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }
}

/// Free any buffer the engine *returned* (query results, schema dump, encode,
/// error JSON, command output) — the single release path for engine allocations.
///
/// # Safety
/// `ptr`/`len` must be a `(ptr, out_len)` the engine returned, freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn lnk_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        // SAFETY: the buffer was produced by out_string / Box<[u8]>::into_raw with this exact length.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}

// re-export the error-channel reader so it lands in this crate's cdylib surface.
pub use crate::ffi_error::lnk_last_error_json;

// ---------------------------------------------------------------- lifecycle ---

/// Build a `Store`. `format`: 0 = NDJSON (a null `ptr` yields an empty graph),
/// 1 = binary snapshot, 2 = pg-json, 3 = pg-text, 4 = graphson, 5 = csv. Null on
/// error (detail on the error channel).
///
/// # Safety
/// If non-null, `ptr`/`len` must describe a readable byte range.
#[no_mangle]
pub unsafe extern "C" fn lnk_open(ptr: *const u8, len: usize, format: u8) -> *mut Store {
    crate::ffi_error::begin();
    match format {
        0 => {
            if ptr.is_null() {
                return Box::into_raw(Box::new(Store::default()));
            }
            // SAFETY: forwards this fn's contract to in_str.
            let Some(text) = (unsafe { in_str(ptr, len) }) else {
                crate::ffi_error::set("E_FFI", "NDJSON input is not valid UTF-8");
                return std::ptr::null_mut();
            };
            match crate::ndjson::from_ndjson(text) {
                Ok(store) => Box::into_raw(Box::new(store)),
                Err(e) => {
                    crate::ffi_error::set("E_INVALID_JSON", &e.to_string());
                    std::ptr::null_mut()
                }
            }
        }
        1 => {
            // SAFETY: forwards this fn's contract to the byte-slice shim.
            let Some(bytes) = (unsafe { in_bytes(ptr, len) }) else {
                crate::ffi_error::set("E_FFI", "null binary snapshot pointer");
                return std::ptr::null_mut();
            };
            match crate::binary::from_binary(bytes) {
                Ok(store) => Box::into_raw(Box::new(store)),
                Err(e) => {
                    crate::ffi_error::set("E_FFI", &e);
                    std::ptr::null_mut()
                }
            }
        }
        // The shared textual codecs (pg-json/pg-text/graphson/csv), via the neutral
        // GraphData bridge — one byte-identical implementation with lenke-core. Gated
        // behind the `codecs` feature; without it, an unknown format.
        // SAFETY: forwards this fn's contract to the codec helper.
        _ => unsafe { open_via_codec(ptr, len, format) },
    }
}

/// The textual-codec name for a `lnk_open`/`lnk_encode` format byte >= 2, or
/// `None` for an unknown byte. NDJSON (0) and binary (1) are handled inline.
#[cfg(feature = "codecs")]
fn codec_format_name(format: u8) -> Option<&'static str> {
    match format {
        2 => Some("pg-json"),
        3 => Some("pg-text"),
        4 => Some("graphson"),
        5 => Some("csv"),
        _ => None,
    }
}

/// Open a store from a textual-codec format (>= 2), via the shared crate.
///
/// # Safety
/// If non-null, `ptr`/`len` must describe a readable byte range.
#[cfg(feature = "codecs")]
unsafe fn open_via_codec(ptr: *const u8, len: usize, format: u8) -> *mut Store {
    let Some(name) = codec_format_name(format) else {
        crate::ffi_error::set("E_UNKNOWN_FORMAT", "unknown open format");
        return std::ptr::null_mut();
    };
    // SAFETY: forwards this fn's contract to in_str.
    let Some(text) = (unsafe { in_str(ptr, len) }) else {
        crate::ffi_error::set("E_FFI", "codec input is not valid UTF-8");
        return std::ptr::null_mut();
    };
    match crate::codec::deserialize(text, name) {
        Ok(store) => Box::into_raw(Box::new(store)),
        Err(e) => {
            crate::ffi_error::set(e.code, &e.message);
            std::ptr::null_mut()
        }
    }
}

/// The no-codecs build: every textual format is unknown.
///
/// # Safety
/// Trivially safe (touches no pointer); the signature matches the gated version.
#[cfg(not(feature = "codecs"))]
unsafe fn open_via_codec(_ptr: *const u8, _len: usize, _format: u8) -> *mut Store {
    crate::ffi_error::set(
        "E_UNKNOWN_FORMAT",
        "this build has no textual codecs (compiled without the `codecs` feature)",
    );
    std::ptr::null_mut()
}

/// Free a `Store` from [`lnk_open`] / [`lnk_clone`].
///
/// # Safety
/// `s` must be a handle from this module (or null), freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn lnk_close(s: *mut Store) {
    if !s.is_null() {
        // SAFETY: the pointer came from Box::into_raw in this module and is freed exactly once.
        drop(unsafe { Box::from_raw(s) });
    }
}

/// Deep-copy a `Store`. Null on error.
///
/// # Safety
/// `s` must be a valid handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_clone(s: *const Store) -> *mut Store {
    crate::ffi_error::begin();
    // SAFETY: forwards this fn's contract to store_ref.
    let Some(store) = (unsafe { store_ref(s) }) else {
        crate::ffi_error::set("E_FFI", "null store handle");
        return std::ptr::null_mut();
    };
    Box::into_raw(Box::new(store.clone()))
}

/// Set a resource limit by its stable [`ConfigId`](crate::store::ConfigId). Returns
/// 1 when applied, 0 for a null handle / unknown id / zero value.
///
/// # Safety
/// `s` must be a valid handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_config(s: *mut Store, id: u32, value: u64) -> u32 {
    // SAFETY: forwards this fn's contract to store_mut.
    let Some(store) = (unsafe { store_mut(s) }) else {
        return 0;
    };
    let Some(id) = crate::store::ConfigId::from_u32(id) else {
        return 0;
    };
    if value == 0 {
        return 0; // reject a zero ceiling (matches core)
    }
    store.set_limit(id, value);
    1
}

/// Read a scalar statistic. `which`: 0 = vertex count, 1 = edge count,
/// 2 = version (monotonic mutation counter). 0 for a null handle / unknown
/// selector. (Per-token epoch is NOT here — it takes a name, so it rides
/// `lnk_command` "epoch", not a bare selector.)
///
/// # Safety
/// `s` must be a valid handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_stat(s: *const Store, which: u32) -> u64 {
    // SAFETY: forwards this fn's contract to store_ref.
    let Some(store) = (unsafe { store_ref(s) }) else {
        return 0;
    };
    match which {
        0 => store.node_count() as u64,
        1 => store.edge_count() as u64,
        2 => store.version(),
        _ => 0,
    }
}

// -------------------------------------------------------------- query / exec ---

/// Run a query. `lang`: 0 = GQL, 1 = Gremlin. `p_ptr`/`p_len`: JSON params (null =
/// none). `format`: 0 = JSON rows, 1 = Arrow, 2 = Arrow IPC. Returns the carrier
/// for `(lang, format)`; null on error.
///
/// # Safety
/// `s` valid; `q_ptr`/`q_len` a valid UTF-8 range; `p_ptr` null or a valid range;
/// `out_len` writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn lnk_query(
    s: *mut Store,
    lang: u8,
    q_ptr: *const u8,
    q_len: usize,
    p_ptr: *const u8,
    p_len: usize,
    format: u8,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    // Mutable borrow: a GQL write query (SET/INSERT/_MERGE/…) needs it; reads and
    // Gremlin reborrow it immutably.
    let (Some(store), Some(q)) = (unsafe { store_mut(s) }, unsafe { in_str(q_ptr, q_len) }) else {
        crate::ffi_error::set("E_FFI", "null store handle or non-UTF-8 query");
        return std::ptr::null_mut();
    };
    // Decode the JSON params object (`{"name": value, …}`); null / empty = none.
    let params = if p_ptr.is_null() || p_len == 0 {
        Vec::new()
    } else {
        let Some(p) = (unsafe { in_str(p_ptr, p_len) }) else {
            crate::ffi_error::set("E_FFI", "params are not valid UTF-8");
            return std::ptr::null_mut();
        };
        match crate::ndjson::parse_params(p) {
            Ok(params) => params,
            Err(e) => {
                crate::ffi_error::set("E_INVALID_JSON", &e);
                return std::ptr::null_mut();
            }
        }
    };
    // Arrow output (GQL only): run to a Rows batch, then frame it. Format 1 is the raw
    // columnar ARW1 blob (column buffers at 8-aligned offsets within it — the global
    // allocator returns >=8-aligned memory on native and wasm, so typed-array views over
    // it are valid without a dedicated aligned allocator); format 2 is the portable Arrow
    // IPC / Feather byte stream for the DuckDB/Polars/pandas handoff. Both are plain byte
    // buffers freed by lnk_free.
    if format != 0 {
        if lang != 0 {
            crate::ffi_error::set("E_FFI", "Arrow output is only available for GQL queries");
            return std::ptr::null_mut();
        }
        let plan = match crate::gql::parse_with_params(q, &params) {
            Ok(plan) => crate::opt::optimize_indexed(plan, store),
            Err(e) => {
                crate::ffi_error::set("E_SYNTAX", &e);
                return std::ptr::null_mut();
            }
        };
        let rows = match guarded(|| crate::exec::try_run(&plan, store)) {
            Ok(r) => r,
            Err(e) => {
                crate::ffi_error::set("E_INVALID_VALUE", &e);
                return std::ptr::null_mut();
            }
        };
        return match format {
            // SAFETY: out_len is writable per this fn's contract.
            1 => unsafe { out_bytes(crate::arrow::to_arrow(&rows), out_len) },
            // Arrow IPC: format 2 = FILE (Feather, ARROW1 magic), 3 = STREAM. The host
            // picks per `queryArrowIpc`'s `format` option; a stream request that lands on
            // 2 (file) used to emit the wrong framing (the `_file` flag was dropped).
            2 => unsafe { out_bytes(crate::arrow::to_arrow_ipc(&rows, true), out_len) },
            3 => unsafe { out_bytes(crate::arrow::to_arrow_ipc(&rows, false), out_len) },
            _ => {
                crate::ffi_error::set("E_FFI", "unknown query format");
                std::ptr::null_mut()
            }
        };
    }
    // Parse (E_SYNTAX on failure) → optimize → run (E_INVALID_VALUE on failure).
    let result = match lang {
        0 => {
            let plan = match crate::gql::parse_with_params(q, &params) {
                Ok(plan) => plan,
                Err(e) => {
                    crate::ffi_error::set("E_SYNTAX", &e);
                    return std::ptr::null_mut();
                }
            };
            // An ISO transaction-control command (`START TRANSACTION`/`COMMIT`/
            // `ROLLBACK`) drives the session's transaction frame and yields no rows —
            // it is neither optimized nor run as a query plan.
            if let crate::ir::Plan::TxControl { kind, read_only } = plan {
                guarded(|| {
                    crate::exec::run_tx_control(store, kind, read_only)
                        .map(|rows| crate::json::gql_rows_json(&rows))
                })
            } else {
                let plan = crate::opt::optimize_indexed(plan, store);
                // A write query (SET/INSERT/_MERGE/…) runs through the mutable executor
                // and renders its result rows — but is rejected first if the active
                // transaction is READ ONLY; a read takes the streaming JSON path.
                if crate::exec::is_write(&plan) {
                    guarded(|| {
                        crate::exec::enforce_read_only(store)
                            .and_then(|()| crate::exec::execute(&plan, store))
                            .map(|rows| crate::json::gql_rows_json(&rows))
                    })
                } else {
                    guarded(|| crate::exec::try_run_gql_json(&plan, store))
                }
            }
        }
        1 => {
            // Gremlin bindings are a distinct mechanism (bytecode bindings); the engine
            // does not accept parameters on the Gremlin path yet.
            if !params.is_empty() {
                crate::ffi_error::set(
                    "E_UNSUPPORTED",
                    "Gremlin query parameters are not supported",
                );
                return std::ptr::null_mut();
            }
            let plan = match crate::gremlin::parse(q) {
                Ok(plan) => crate::opt::optimize_indexed(plan, store),
                Err(e) => {
                    crate::ffi_error::set("E_SYNTAX", &e);
                    return std::ptr::null_mut();
                }
            };
            guarded(|| crate::exec::try_run_gremlin_json(&plan, store))
        }
        _ => {
            crate::ffi_error::set("E_FFI", "unknown query language");
            return std::ptr::null_mut();
        }
    };
    match result {
        Ok(json) => {
            // SAFETY: out_len is writable per this fn's contract.
            unsafe { out_string(json, out_len) }
        }
        Err(e) => {
            set_exec_error(&e);
            std::ptr::null_mut()
        }
    }
}

// ------------------------------------------------------------- transactions ---

/// Transaction control. `action`: 0 = begin, 1 = commit, 2 = rollback. 0 on
/// success, -1 on an unknown action / null handle.
///
/// # Safety
/// `s` must be a valid handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_tx(s: *mut Store, action: u8) -> i32 {
    crate::ffi_error::begin();
    // SAFETY: forwards this fn's contract to store_mut.
    let Some(store) = (unsafe { store_mut(s) }) else {
        crate::ffi_error::set("E_FFI", "null store handle");
        return -1;
    };
    match action {
        0 => store.begin(),
        // Commit runs the DEFERRED declared-constraint checks against the fully-staged
        // graph first; a violation rolls the whole transaction back and fails the commit
        // with its coded error (so the host `transaction()` throws, matching core).
        1 => {
            if let Err(e) = crate::exec::commit_with_deferred_checks(store) {
                set_exec_error(&e);
                return -1;
            }
        }
        2 => store.rollback(),
        _ => {
            crate::ffi_error::set("E_FFI", "unknown tx action");
            return -1;
        }
    }
    0
}

// -------------------------------------------------------------------- schema ---

/// Apply one schema op described by JSON (see `docs/abi.md`). 0 ok · -1 arg/parse
/// error · -2 rejected by current data. Detail on the error channel.
///
/// # Safety
/// `s` valid; `json_ptr`/`json_len` a valid UTF-8 range (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_schema_apply(
    s: *mut Store,
    json_ptr: *const u8,
    json_len: usize,
) -> i32 {
    crate::ffi_error::begin();
    // SAFETY: forwards this fn's contract to the shims.
    let (Some(store), Some(json)) = (unsafe { store_mut(s) }, unsafe {
        in_str(json_ptr, json_len)
    }) else {
        crate::ffi_error::set("E_FFI", "null store handle or non-UTF-8 payload");
        return -1;
    };
    use crate::schema_op::SchemaError;
    // The exec layer is the single schema entry point: it handles validator /
    // invariant ops (which need the query evaluator) and delegates the rest.
    match crate::exec::apply_schema_op(store, json) {
        Ok(()) => 0,
        Err(SchemaError::BadRequest(msg)) => {
            crate::ffi_error::set("E_FFI", &msg);
            -1
        }
        Err(SchemaError::Invalid(msg)) => {
            crate::ffi_error::set("E_INVALID_VALUE", &msg);
            -1
        }
        Err(SchemaError::Syntax(msg)) => {
            crate::ffi_error::set("E_SYNTAX", &msg);
            -1
        }
        Err(SchemaError::GraphOp(msg)) => {
            crate::ffi_error::set("E_INVALID_GRAPH_OP", &msg);
            -1
        }
        Err(SchemaError::Rejected(msg)) => {
            crate::ffi_error::set("E_CONSTRAINT_VIOLATION", &msg);
            -2
        }
    }
}

/// Dump the full schema as a JSON op-list (the single schema read; subsumes the
/// per-element index lists). Null on error.
///
/// # Safety
/// `s` valid; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_schema_dump(s: *const Store, out_len: *mut usize) -> *mut u8 {
    crate::ffi_error::begin();
    // SAFETY: forwards this fn's contract to store_ref.
    let Some(store) = (unsafe { store_ref(s) }) else {
        crate::ffi_error::set("E_FFI", "null store handle");
        return std::ptr::null_mut();
    };
    // Indexes and constraints, in the same `{"op":…}` vocabulary lnk_schema_apply
    // consumes, so dump -> apply round-trips (see schema_op::dump).
    let ops = crate::schema_op::dump(store);
    // SAFETY: out_len is writable per this fn's contract.
    unsafe { out_string(ops, out_len) }
}

// ------------------------------------------------------------------ snapshot ---

/// Encode the graph's data channel. `format`: 0 = NDJSON, 1 = binary, 2 = pg-json,
/// 3 = pg-text, 4 = graphson, 5 = csv. Pairs with [`lnk_schema_dump`] for a full
/// snapshot. Null on error.
///
/// # Safety
/// `s` valid; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_encode(s: *const Store, format: u8, out_len: *mut usize) -> *mut u8 {
    crate::ffi_error::begin();
    // SAFETY: forwards this fn's contract to store_ref.
    let Some(store) = (unsafe { store_ref(s) }) else {
        crate::ffi_error::set("E_FFI", "null store handle");
        return std::ptr::null_mut();
    };
    match format {
        0 => {
            let text = crate::ndjson::to_ndjson(store);
            // SAFETY: out_len is writable per this fn's contract.
            unsafe { out_string(text, out_len) }
        }
        1 => {
            // SAFETY: out_len is writable per this fn's contract.
            unsafe { out_bytes(crate::binary::to_binary(store), out_len) }
        }
        // The shared textual codecs, via the neutral GraphData bridge (the `codecs`
        // feature; an unknown format without it).
        // SAFETY: out_len is writable per this fn's contract.
        _ => unsafe { encode_via_codec(store, format, out_len) },
    }
}

/// Encode a store in a textual-codec format (>= 2), via the shared crate.
///
/// # Safety
/// `out_len` must be a valid, writable pointer.
#[cfg(feature = "codecs")]
unsafe fn encode_via_codec(store: &Store, format: u8, out_len: *mut usize) -> *mut u8 {
    let Some(name) = codec_format_name(format) else {
        crate::ffi_error::set("E_UNKNOWN_FORMAT", "unknown encode format");
        return std::ptr::null_mut();
    };
    match crate::codec::serialize(store, name) {
        // SAFETY: out_len is writable per this fn's contract.
        Ok(text) => unsafe { out_string(text, out_len) },
        Err(e) => {
            crate::ffi_error::set(e.code, &e.message);
            std::ptr::null_mut()
        }
    }
}

/// The no-codecs build: every textual format is unknown.
///
/// # Safety
/// Trivially safe (touches no pointer); the signature matches the gated version.
#[cfg(not(feature = "codecs"))]
unsafe fn encode_via_codec(_store: &Store, _format: u8, _out_len: *mut usize) -> *mut u8 {
    crate::ffi_error::set(
        "E_UNKNOWN_FORMAT",
        "this build has no textual codecs (compiled without the `codecs` feature)",
    );
    std::ptr::null_mut()
}

// -------------------------------------------------------------- escape hatch ---

/// Run a named command with a JSON/bytes input, returning a JSON/bytes buffer.
/// This is the single home for the exotic tiers (algo, prepared statements, CDC
/// scope, fork/merge, …): adding one fills a match arm here, never a new symbol.
/// Null on error / unknown name.
///
/// # Safety
/// `s` valid; `name_ptr`/`name_len` a valid UTF-8 range; `in_ptr` null or a valid
/// range; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_command(
    s: *mut Store,
    name_ptr: *const u8,
    name_len: usize,
    in_ptr: *const u8,
    in_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    // SAFETY: forwards this fn's contract to the shims.
    let (Some(store), Some(name)) = (unsafe { store_mut(s) }, unsafe {
        in_str(name_ptr, name_len)
    }) else {
        crate::ffi_error::set("E_FFI", "null store handle or non-UTF-8 command name");
        return std::ptr::null_mut();
    };
    // The per-command JSON/bytes payload (interpretation is command-specific).
    let input = unsafe { in_str(in_ptr, in_len) };
    match name {
        // CDC: the content-derived scopes the last commit touched, plus a fail-open
        // flag. Input is the scope-key property name; output `{"scopes":[…],"open":b}`.
        "last_write_scope" => {
            let Some(scope_key) = input else {
                crate::ffi_error::set(
                    "E_FFI",
                    "last_write_scope requires the scope-key name as the input payload",
                );
                return std::ptr::null_mut();
            };
            // SAFETY: out_len is writable per this fn's contract.
            unsafe { out_string(store.last_write_scope_json(scope_key), out_len) }
        }
        // First-wins bulk merge of an NDJSON document (matching core). Input is the
        // NDJSON text; output the `MergeReport`
        // `{"nodesAdded","edgesAdded","nodesSkipped":[…],"edgesSkipped":[…],"phantomVertices":[…]}`.
        "merge" => {
            let Some(text) = input else {
                crate::ffi_error::set("E_FFI", "merge requires an NDJSON payload as input");
                return std::ptr::null_mut();
            };
            match crate::ndjson::merge_ndjson(store, text) {
                Ok(report) => {
                    let mut json = String::new();
                    json.push_str(&format!(
                        "{{\"nodesAdded\":{},\"edgesAdded\":{},\"nodesSkipped\":",
                        report.nodes_added, report.edges_added
                    ));
                    crate::ndjson::encode_str_array(&mut json, &report.nodes_skipped);
                    json.push_str(",\"edgesSkipped\":");
                    crate::ndjson::encode_str_array(&mut json, &report.edges_skipped);
                    json.push_str(",\"phantomVertices\":");
                    crate::ndjson::encode_str_array(&mut json, &report.phantom_vertices);
                    json.push('}');
                    // SAFETY: out_len is writable per this fn's contract.
                    unsafe { out_string(json, out_len) }
                }
                Err(e) => {
                    crate::ffi_error::set("E_INVALID_JSON", &e);
                    std::ptr::null_mut()
                }
            }
        }
        // Per-token change epoch (finer invalidation than the global version).
        // Input is the token name (label / edge-type / property-key); output `{"epoch":N}`.
        "epoch" => {
            let Some(token) = input else {
                crate::ffi_error::set(
                    "E_FFI",
                    "epoch requires the token name as the input payload",
                );
                return std::ptr::null_mut();
            };
            let json = format!("{{\"epoch\":{}}}", store.epoch(token));
            // SAFETY: out_len is writable per this fn's contract.
            unsafe { out_string(json, out_len) }
        }
        // Prepared statements (Design A: parse once, bind + optimize + run per call —
        // amortizes parse cost across a loop). Handles live in a generational slab
        // (`crate::prepared`), so a stale/double handle is a clean error, never a bad
        // dereference. The handle is a decimal string (it can exceed a JS f64). The
        // caller owns lifetime: prepare then N prepared_run then prepared_free — and a
        // host `using` / FinalizationRegistry reclaims a forgotten one.
        "prepare" => {
            let Some(query) = input else {
                crate::ffi_error::set("E_FFI", "prepare requires the query text as input");
                return std::ptr::null_mut();
            };
            match crate::gql::parse_prepared(query) {
                Ok(plan) => {
                    let handle = crate::prepared::insert(plan);
                    // SAFETY: out_len is writable per this fn's contract.
                    unsafe { out_string(format!("{{\"handle\":\"{handle}\"}}"), out_len) }
                }
                Err(e) => {
                    crate::ffi_error::set("E_SYNTAX", &e);
                    std::ptr::null_mut()
                }
            }
        }
        "prepared_run" => {
            // Input: {"handle":"<ptr>", "params":{...}}. Clone the cached plan, bind
            // the params, optimize + run against this store (the plan is graph-agnostic,
            // so one prepared statement serves any store).
            let (handle, params, format) = match prepared_payload(input) {
                Ok(hp) => hp,
                Err(e) => {
                    crate::ffi_error::set("E_FFI", &e);
                    return std::ptr::null_mut();
                }
            };
            // A stale/freed handle resolves to None here — a clean error, not a
            // dangling dereference (the whole point of the generational slab).
            let Some(mut plan) = crate::prepared::get_clone(handle) else {
                crate::ffi_error::set("E_FFI", "invalid or freed prepared handle");
                return std::ptr::null_mut();
            };
            if let Err(e) = crate::bind::bind_params(&mut plan, &params) {
                crate::ffi_error::set("E_MISSING_PARAMETER", &e);
                return std::ptr::null_mut();
            }
            let plan = crate::opt::optimize_indexed(plan, store);
            // JSON rows (default) or an Arrow carrier (raw ARW1 / IPC), so a prepared
            // statement has the same output surface as `lnk_query`.
            match format.as_str() {
                "arrow" | "arrow_ipc" => match guarded(|| crate::exec::try_run(&plan, store)) {
                    Ok(rows) => {
                        let bytes = if format == "arrow_ipc" {
                            crate::arrow::to_arrow_ipc(&rows, true)
                        } else {
                            crate::arrow::to_arrow(&rows)
                        };
                        // SAFETY: out_len is writable per this fn's contract.
                        unsafe { out_bytes(bytes, out_len) }
                    }
                    Err(e) => {
                        set_exec_error(&e);
                        std::ptr::null_mut()
                    }
                },
                _ => match guarded(|| crate::exec::try_run_gql_json(&plan, store)) {
                    // SAFETY: out_len is writable per this fn's contract.
                    Ok(json) => unsafe { out_string(json, out_len) },
                    Err(e) => {
                        set_exec_error(&e);
                        std::ptr::null_mut()
                    }
                },
            }
        }
        "prepared_free" => {
            let handle = match prepared_handle(input) {
                Ok(h) => h,
                Err(e) => {
                    crate::ffi_error::set("E_FFI", &e);
                    return std::ptr::null_mut();
                }
            };
            // A double-free / unknown handle returns false — a clean error, not a
            // second drop of freed memory.
            if !crate::prepared::free(handle) {
                crate::ffi_error::set("E_FFI", "invalid or already-freed prepared handle");
                return std::ptr::null_mut();
            }
            // SAFETY: out_len is writable per this fn's contract.
            unsafe { out_string("{}".to_string(), out_len) }
        }
        // Run a native graph algorithm directly (also reachable via GQL `CALL`).
        // Input `{"name": "<algo>", "config": {…}}`; output a
        // `{"columns":["node","<result>"],"rows":[["<ext id>", value], …]}` row set
        // (same shape as core's `lnk_algo`). A `writeProperty` in the config writes
        // each result back to that node property.
        "algo" => {
            let Some(text) = input else {
                crate::ffi_error::set("E_FFI", "algo requires a {name, config} JSON payload");
                return std::ptr::null_mut();
            };
            let fields = match crate::ndjson::parse_json(text) {
                Ok(crate::ndjson::Json::Obj(f)) => f,
                Ok(_) => {
                    crate::ffi_error::set("E_FFI", "algo payload must be a JSON object");
                    return std::ptr::null_mut();
                }
                Err(e) => {
                    crate::ffi_error::set("E_INVALID_JSON", &e);
                    return std::ptr::null_mut();
                }
            };
            match run_algo_command(store, &fields) {
                Ok(json) => unsafe { out_string(json, out_len) },
                Err(e) => {
                    crate::ffi_error::set("E_INVALID_VALUE", &e);
                    std::ptr::null_mut()
                }
            }
        }
        // Remaining exotic tiers (prepared statements, epoch, merge, binary snapshot)
        // are not yet built in the engine — each fills an arm here, never a new symbol.
        other => {
            crate::ffi_error::set(
                "E_UNSUPPORTED",
                &format!("command '{other}' is not yet implemented"),
            );
            std::ptr::null_mut()
        }
    }
}
