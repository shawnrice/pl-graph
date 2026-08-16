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
//! This is a SCAFFOLD: symbols wired to methods that exist run today; every gap
//! returns `E_UNIMPLEMENTED` with a specific message. That stub list is the
//! work-queue to finish before the engine can be the shipped backend.

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
/// 1 = binary snapshot. Null on error (detail on the error channel).
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
                    crate::ffi_error::set("E_NDJSON", &e.to_string());
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
                    crate::ffi_error::set("E_BINARY", &e);
                    std::ptr::null_mut()
                }
            }
        }
        _ => {
            crate::ffi_error::set("E_FFI", "unknown open format");
            std::ptr::null_mut()
        }
    }
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
    // SAFETY: forwards this fn's contract to the shims. Read-only exec today → &Store.
    let (Some(store), Some(q)) = (unsafe { store_ref(s) }, unsafe { in_str(q_ptr, q_len) }) else {
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
                crate::ffi_error::set("E_FFI", &e);
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
        let rows = match crate::gql::parse_with_params(q, &params) {
            Ok(plan) => {
                let plan = crate::opt::optimize_indexed(plan, store);
                guarded(|| crate::exec::try_run(&plan, store))
            }
            Err(e) => Err(e),
        };
        let rows = match rows {
            Ok(r) => r,
            Err(e) => {
                crate::ffi_error::set("E_QUERY", &e);
                return std::ptr::null_mut();
            }
        };
        return match format {
            // SAFETY: out_len is writable per this fn's contract.
            1 => unsafe { out_bytes(crate::arrow::to_arrow(&rows), out_len) },
            2 => unsafe { out_bytes(crate::arrow::to_arrow_ipc(&rows, true), out_len) },
            _ => {
                crate::ffi_error::set("E_FFI", "unknown query format");
                std::ptr::null_mut()
            }
        };
    }
    let result = match lang {
        0 => match crate::gql::parse_with_params(q, &params) {
            Ok(plan) => {
                let plan = crate::opt::optimize_indexed(plan, store);
                guarded(|| crate::exec::try_run_gql_json(&plan, store))
            }
            Err(e) => Err(e),
        },
        1 => {
            // Gremlin bindings are a distinct mechanism (bytecode bindings); the engine
            // does not accept parameters on the Gremlin path yet.
            if !params.is_empty() {
                crate::ffi_error::set(
                    "E_UNIMPLEMENTED",
                    "Gremlin query parameters are not yet supported",
                );
                return std::ptr::null_mut();
            }
            match crate::gremlin::parse(q) {
                Ok(plan) => {
                    let plan = crate::opt::optimize_indexed(plan, store);
                    guarded(|| crate::exec::try_run_gremlin_json(&plan, store))
                }
                Err(e) => Err(e),
            }
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
            crate::ffi_error::set("E_QUERY", &e);
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
        1 => store.commit(),
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
    match crate::schema_op::apply(store, json) {
        Ok(()) => 0,
        Err(crate::schema_op::SchemaError::BadRequest(msg)) => {
            crate::ffi_error::set("E_FFI", &msg);
            -1
        }
        Err(crate::schema_op::SchemaError::Rejected(msg)) => {
            crate::ffi_error::set("E_CONSTRAINT", &msg);
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

/// Encode the graph's data channel. `format`: 0 = NDJSON, 1 = binary. Pairs with
/// [`lnk_schema_dump`] for a full snapshot. Null on error.
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
        _ => {
            crate::ffi_error::set("E_FFI", "unknown encode format");
            std::ptr::null_mut()
        }
    }
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
        // Merge an NDJSON document into this store, last-write-wins by external id.
        // Input is the NDJSON text; output the post-merge `{"nodes":N,"edges":M}`.
        "merge" => {
            let Some(text) = input else {
                crate::ffi_error::set("E_FFI", "merge requires an NDJSON payload as input");
                return std::ptr::null_mut();
            };
            match crate::ndjson::merge_ndjson(store, text) {
                Ok(()) => {
                    let json = format!(
                        "{{\"nodes\":{},\"edges\":{}}}",
                        store.node_count(),
                        store.edge_count()
                    );
                    // SAFETY: out_len is writable per this fn's contract.
                    unsafe { out_string(json, out_len) }
                }
                Err(e) => {
                    crate::ffi_error::set("E_MERGE", &e);
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
        // Algorithms are reachable through the conformant GQL path already, so the
        // direct command is a redundant fast-path we have not needed.
        "algo" => {
            crate::ffi_error::set(
                "E_UNIMPLEMENTED",
                "run algorithms via GQL: lnk_query(GQL, \"CALL <name>(...) YIELD ...\")",
            );
            std::ptr::null_mut()
        }
        // Remaining exotic tiers (prepared statements, epoch, merge, binary snapshot)
        // are not yet built in the engine — each fills an arm here, never a new symbol.
        other => {
            crate::ffi_error::set(
                "E_UNIMPLEMENTED",
                &format!("command '{other}' is not yet implemented"),
            );
            std::ptr::null_mut()
        }
    }
}
