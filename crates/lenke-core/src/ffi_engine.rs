//! Parallel C ABI for the STANDALONE columnar engine (`lenke-engine`), exposed only
//! under the `engine-compare` feature. Every export is prefixed `lnk_e_` and mirrors a
//! `lnk_*` core export so the SAME graph and query drive BOTH engines through one
//! comparison harness (`packages/native`), across every surface — because these are
//! `#[no_mangle]` in this crate they land in both the native cdylib (bun:ffi) and the
//! wasm build automatically, exactly like the core surface.
//!
//! This is a MEASUREMENT surface, not a shipped one: it holds an engine `Store` handle
//! (built from engine-dialect NDJSON), runs a read-only Gremlin or GQL query, and hands
//! back the result as JSON matching core's carrier shape (`gremlin_json` → a bare array;
//! `query_rows` → a `{columns, rows}` document), so the host can `JSON.parse` both sides
//! and compare values directly.
//!
//! Result buffers are freed with the existing [`crate::ffi::lnk_free_buf`]; there is no
//! engine-specific free for them.

#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use lenke_engine::store::Store;

/// Borrow an engine `Store` handle immutably (null → `None`).
///
/// # Safety
/// If non-null, `s` must point to a live `Store` not mutably aliased for `'a`.
unsafe fn store_ref<'a>(s: *const Store) -> Option<&'a Store> {
    // SAFETY: as_ref() yields None for null; otherwise the caller's # Safety contract requires s point to a live, aligned, non-mutably-aliased Store for 'a.
    unsafe { s.as_ref() }
}

/// Borrow an engine `Store` handle mutably (null → `None`).
///
/// # Safety
/// If non-null, `s` must point to a live `Store` not otherwise aliased for `'a`.
unsafe fn store_mut<'a>(s: *mut Store) -> Option<&'a mut Store> {
    // SAFETY: as_mut() yields None for null; otherwise the caller's # Safety contract requires exclusive access to a live, aligned Store for 'a.
    unsafe { s.as_mut() }
}

/// Set one engine resource limit by its stable `ConfigId` (mirrors core's
/// `lnk_graph_set_config`, same id space — so one host code path configures either
/// engine). Returns 1 when applied, 0 for a null handle / unknown id / zero value, so a
/// host on an older artifact can report an unsupported setting rather than silently
/// running with the default.
///
/// # Safety
/// `s` must be a valid `Store` handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_e_graph_set_config(s: *mut Store, id: u32, value: u64) -> u32 {
    // SAFETY: forwards this fn's # Safety contract to store_mut (null -> None).
    let Some(store) = (unsafe { store_mut(s) }) else {
        return 0;
    };
    let Some(id) = lenke_engine::store::ConfigId::from_u32(id) else {
        return 0;
    };
    if value == 0 {
        return 0; // reject a zero ceiling (matches core), which would break every query
    }
    store.set_limit(id, value);
    1
}

/// Borrow a caller-owned byte buffer as UTF-8 `&str` (null or non-UTF-8 → `None`).
///
/// # Safety
/// If non-null, `ptr`/`len` must describe a readable, initialized byte range valid for `'a`.
unsafe fn in_str<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr is non-null (checked); the caller's # Safety contract requires ptr/len describe a readable, initialized UTF-8 range valid for 'a.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).ok()
}

/// Hand a heap `String` back to the caller as a `(ptr, len)` buffer, freed with
/// [`crate::ffi::lnk_free_buf`]. `out_len` receives the byte length.
///
/// # Safety
/// `out_len` must be a valid, writable pointer.
unsafe fn out_string(s: String, out_len: *mut usize) -> *mut u8 {
    let bytes = s.into_bytes().into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

// ---------- engine store handle ----------

/// Decode ENGINE-dialect NDJSON (`{id,labels,props}` / `{from,to,type|labels,props}`)
/// into an engine `Store` and return an owning handle. Null on bad UTF-8 or a decode
/// error. Free with [`lnk_e_graph_free`].
///
/// # Safety
/// `ptr`/`len` must describe a valid UTF-8 range (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_e_graph_from_ndjson(ptr: *const u8, len: usize) -> *mut Store {
    // SAFETY: forwards this fn's # Safety contract to in_str (null / non-UTF-8 -> None).
    let Some(text) = (unsafe { in_str(ptr, len) }) else {
        return std::ptr::null_mut();
    };
    match lenke_engine::ndjson::from_ndjson(text) {
        Ok(store) => Box::into_raw(Box::new(store)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a `Store` handle from [`lnk_e_graph_from_ndjson`].
///
/// # Safety
/// `s` must be a handle from this module (or null), freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn lnk_e_graph_free(s: *mut Store) {
    if !s.is_null() {
        // SAFETY: the pointer came from this module's Box::into_raw, per the fn's # Safety contract, and is freed exactly once.
        drop(unsafe { Box::from_raw(s) });
    }
}

/// The store's live node count.
///
/// # Safety
/// `s` must be a valid `Store` handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_e_graph_vertex_count(s: *const Store) -> u64 {
    // SAFETY: forwards this fn's # Safety contract to store_ref (null -> None).
    match unsafe { store_ref(s) } {
        Some(s) => s.node_count() as u64,
        None => 0,
    }
}

/// The store's live edge count.
///
/// # Safety
/// `s` must be a valid `Store` handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_e_graph_edge_count(s: *const Store) -> u64 {
    // SAFETY: forwards this fn's # Safety contract to store_ref (null -> None).
    match unsafe { store_ref(s) } {
        Some(s) => s.edge_count() as u64,
        None => 0,
    }
}

/// Run a Gremlin query against the engine store and return the results as a bare JSON
/// array (core's `lnk_gremlin_json` carrier shape). Null on a null handle / bad UTF-8 /
/// parse failure. `out_len` receives the byte length; free with `lnk_free_buf`.
///
/// # Safety
/// `s` valid; `q_ptr`/`q_len` a valid UTF-8 range; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_e_gremlin_json(
    s: *const Store,
    q_ptr: *const u8,
    q_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    // SAFETY: forwards this fn's # Safety contract to the shims.
    let (Some(store), Some(q)) = (unsafe { store_ref(s) }, unsafe { in_str(q_ptr, q_len) }) else {
        return std::ptr::null_mut();
    };
    let plan = match lenke_engine::gremlin::parse(q) {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };
    let plan = lenke_engine::opt::optimize_indexed(plan, store);
    // Run behind a panic backstop: an evaluation fault returns `Err` (→ null), and even a
    // genuine panic (e.g. an internal index bug) is caught rather than unwinding across
    // this `extern "C"` boundary — which is undefined behavior / a process abort. A
    // faulting query must fail this ONE call, never take the host down.
    let json = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lenke_engine::exec::try_run_gremlin_json(&plan, store)
    })) {
        Ok(Ok(json)) => json,
        Ok(Err(_)) | Err(_) => return std::ptr::null_mut(),
    };
    // SAFETY: out_len is writable per this fn's # Safety contract.
    unsafe { out_string(json, out_len) }
}

/// Run a GQL query against the engine store and return a `{columns, rows}` JSON document
/// (core's `lnk_query_rows` carrier shape). No params — the comparison harness runs
/// read-only, param-free queries. Null on a null handle / bad UTF-8 / parse failure.
/// `out_len` receives the byte length; free with `lnk_free_buf`.
///
/// # Safety
/// `s` valid; `q_ptr`/`q_len` a valid UTF-8 range; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_e_query_rows(
    s: *const Store,
    q_ptr: *const u8,
    q_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    // SAFETY: forwards this fn's # Safety contract to the shims.
    let (Some(store), Some(q)) = (unsafe { store_ref(s) }, unsafe { in_str(q_ptr, q_len) }) else {
        return std::ptr::null_mut();
    };
    let plan = match lenke_engine::gql::parse(q) {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };
    let plan = lenke_engine::opt::optimize_indexed(plan, store);
    // Panic backstop (see `lnk_e_gremlin_json`): a faulting query fails this one call with
    // a null result, never unwinds across the C boundary.
    let json = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Streams a var-length endpoint projection straight to the {columns,rows}
        // document (no giant row batch), else materializes + serializes — byte-identical.
        lenke_engine::exec::try_run_gql_json(&plan, store)
    })) {
        Ok(Ok(json)) => json,
        Ok(Err(_)) | Err(_) => return std::ptr::null_mut(),
    };
    // SAFETY: out_len is writable per this fn's # Safety contract.
    unsafe { out_string(json, out_len) }
}
