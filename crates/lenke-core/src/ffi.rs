//! C ABI for bun:ffi (and later wasm-bindgen). Exposes a stateful graph handle
//! (`lnk_graph_*`) that owns a decoded columnar graph, so queries / encode run
//! without re-marshalling the whole graph on each call.
//!
//! Buffers passed in are caller-owned and read-only. Buffers handed back
//! (`lnk_encode_ndjson`) are heap-allocated here and must be returned via
//! `lnk_free_buf`. The graph handle must be returned via `lnk_graph_free`.

// This module IS the unsafe boundary: the crate root denies `unsafe_code`, so it is
// re-permitted here — and `unsafe_op_in_unsafe_fn` is denied so every raw-pointer op is
// an explicit `unsafe { … }` block, never an implicit whole-body context. The real
// unsafe surface is the three shims below; every export routes its pointers through them
// and is otherwise ordinary safe Rust.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "_fallible-ffi")]
use crate::error_codes::ErrorCode;
use crate::graph::Graph;
use crate::query;

// ---------- raw-pointer boundary shims ----------
//
// The only place caller-owned raw pointers become safe references/slices. `as_ref`/
// `as_mut` fold the null check into the deref (null → None); `in_str`/`in_bytes` borrow
// an inbound buffer. Reviewed once here, so every `lnk_*` export below is safe code apart
// from its `unsafe { … }` call to one of these.

/// Borrow a graph handle immutably (null → `None`).
///
/// # Safety
/// If non-null, `g` must point to a live `Graph` that is not mutably aliased for `'a`.
unsafe fn graph_ref<'a>(g: *const Graph) -> Option<&'a Graph> {
    // SAFETY: as_ref() yields None for a null pointer; otherwise the caller's # Safety contract requires g point to a live, aligned, non-mutably-aliased Graph for 'a.
    unsafe { g.as_ref() }
}

/// Borrow a graph handle mutably (null → `None`).
///
/// # Safety
/// If non-null, `g` must point to a live `Graph` uniquely borrowed (no other alias) for `'a`.
unsafe fn graph_mut<'a>(g: *mut Graph) -> Option<&'a mut Graph> {
    // SAFETY: as_mut() yields None for a null pointer; otherwise the caller's # Safety contract requires g point to a live, aligned, uniquely-borrowed Graph for 'a.
    unsafe { g.as_mut() }
}

/// Borrow a caller-owned byte buffer as bytes (null → `None`).
///
/// # Safety
/// If non-null, `ptr`/`len` must describe a readable, initialized byte range valid for `'a`.
unsafe fn in_bytes<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr is non-null (checked above); the caller's # Safety contract requires ptr/len describe a readable, initialized byte range valid for 'a.
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Borrow a caller-owned byte buffer as UTF-8 `&str` (null or non-UTF-8 → `None`).
///
/// # Safety
/// If non-null, `ptr`/`len` must describe a readable, initialized byte range valid for `'a`.
unsafe fn in_str<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    // SAFETY: forwards ptr/len to in_bytes (null -> None) under this fn's # Safety contract; the surrounding from_utf8 is safe.
    std::str::from_utf8(unsafe { in_bytes(ptr, len) }?).ok()
}

#[no_mangle]
pub extern "C" fn lnk_abi_version() -> u32 {
    17 // 17: lnk_create_index (single parametric index creator — replaces the three
       //     lnk_create_{vertex,edge,edge_interval}_index exports with element+kind);
       // 16: lnk_create_edge_interval_index (RI-tree interval index for as-of/overlap);
       // 15: lnk_dump_schema (read schema back as SchemaOps, for snapshot persistence);
       // 14: lnk_last_write_scope (content-derived CDC value-scope);
       // 13: lnk_query_arrow_ipc (native Arrow IPC egress);
       // 12: lnk_prepare takes a max-operator-chain arg; 11: per-graph operator-chain
       //    ceiling (lnk_graph_set_max_operator_chain);
       // 10: native graph algorithms (lnk_algo); 9: query params (lnk_query_rows/
       //    lnk_query_arrow take a params-JSON doc); 8: reactive change tracking
       //    (lnk_graph_version/lnk_graph_epoch); 7: codecs (lnk_serialize/
       //    lnk_deserialize); 6: inbound allocator; 5: Gremlin; 4: Arrow
}

// ---------- inbound allocator (wasm linear memory) ----------
//
// Over bun:ffi, JS hands us a pointer to a JS-owned buffer and we read it in
// place. In a browser there is only wasm linear memory: JS cannot point at its
// own heap, it must first copy bytes *into* the module's memory. These two
// exports are that inbound path — JS calls `lnk_alloc(len)`, writes the query /
// NDJSON bytes into the returned offset, then passes (ptr, len) to any `lnk_*`
// symbol exactly as the native binding does. They are the inverse of
// `lnk_free_buf` (which frees buffers we hand *out*); same one-ABI-two-backends
// design. Harmless on native, so unconditionally exported.

/// Allocate `len` bytes inside the module and return a pointer the caller fills.
/// Free it with `lnk_dealloc(ptr, len)` (or pass ownership to a `lnk_*` call
/// that consumes it). Returns null for `len == 0`.
#[no_mangle]
pub extern "C" fn lnk_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free a buffer obtained from `lnk_alloc`. `len` must be the same value passed
/// to `lnk_alloc`.
///
/// # Safety
/// `ptr`/`len` must come from a prior `lnk_alloc` call and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn lnk_dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len != 0 {
        // SAFETY: ptr/len came from a prior lnk_alloc, per the fn's # Safety contract, and is freed exactly once.
        drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
    }
}

// ---------- stateful columnar graph ----------

/// Decode NDJSON bytes into a graph and return an owning handle.
///
/// # Safety
/// `ptr`/`len` must describe valid UTF-8 NDJSON. Returns null on bad UTF-8.
#[cfg(feature = "ndjson")]
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_from_ndjson(
    ptr: *const u8,
    len: usize,
    parallel: u32,
) -> *mut Graph {
    crate::ffi_error::begin();
    if ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null NDJSON pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let text = match unsafe { in_str(ptr, len) } {
        Some(t) => t,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "NDJSON bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    let decoded = if parallel != 0 {
        crate::ndjson::decode(text)
    } else {
        crate::ndjson::decode_serial(text)
    };
    match decoded {
        Ok(g) => Box::into_raw(Box::new(g)),
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            std::ptr::null_mut()
        }
    }
}

/// Bulk-append NDJSON `bytes` into an existing graph — a `COPY FROM` for a live
/// store (the incremental twin of [`lnk_graph_from_ndjson`], at bulk speed, not
/// per-`INSERT` speed). Returns a JSON `MergeReport` buffer (free with
/// [`lnk_free_buf`]) describing what applied vs. skipped, its length in
/// `out_len`; null on a null / parse error (details in the last-error channel).
///
/// # Safety
/// `g` valid + uniquely borrowed; `ptr`/`len` a valid UTF-8 slice; `out_len`
/// writable.
#[cfg(feature = "ndjson")]
#[no_mangle]
pub unsafe extern "C" fn lnk_merge_ndjson(
    g: *mut Graph,
    ptr: *const u8,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if g.is_null() || ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null graph or NDJSON pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let text = match unsafe { in_str(ptr, len) } {
        Some(t) => t,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "NDJSON bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    match crate::ndjson::append(g, text) {
        Ok(report) => {
            let bytes = report.to_json().into_bytes().into_boxed_slice();
            // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
            unsafe { *out_len = bytes.len() };
            Box::into_raw(bytes) as *mut u8
        }
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            std::ptr::null_mut()
        }
    }
}

/// Deep-copy a graph into a fresh, fully independent handle — the fast substrate
/// for a fork/branch. An O(V+E) clone of the columnar store (see `impl Clone for
/// Graph`), not a serialize→parse NDJSON round-trip: no text, no re-validation, no
/// re-indexing, and the element ids are preserved exactly (a base-vs-copy diff by
/// id is exact). The returned handle must be freed with [`lnk_graph_free`] like any
/// other. Null in → null out.
///
/// # Safety
/// `g` must be a valid graph handle (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_clone(g: *const Graph) -> *mut Graph {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match unsafe { graph_ref(g) } {
        Some(g) => Box::into_raw(Box::new(g.clone())),
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `g` must be a handle from `lnk_graph_from_ndjson` (or null).
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_free(g: *mut Graph) {
    if !g.is_null() {
        // Reconstitute ownership to drop it — the one op the borrow shims can't cover.
        // SAFETY: the pointer came from this module's matching Box::into_raw, per the fn's # Safety contract, and is freed exactly once.
        drop(unsafe { Box::from_raw(g) });
    }
}

/// # Safety
/// `g` must be a valid graph handle.
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_vertex_count(g: *const Graph) -> u64 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match unsafe { graph_ref(g) } {
        Some(g) => g.vertex_count() as u64,
        None => 0,
    }
}

/// Set the graph's GQL operator-chain ceiling (the native `maxOperatorChain`
/// option); the parser rejects a longer chain with `E_SYNTAX`. A no-op on a null
/// handle.
///
/// # Safety
/// `g` must be a valid graph handle.
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_set_max_operator_chain(g: *mut Graph, n: u64) {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    if let Some(g) = unsafe { graph_mut(g) } {
        g.set_max_operator_chain(n as usize);
    }
}

/// # Safety
/// `g` must be a valid graph handle.
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_edge_count(g: *const Graph) -> u64 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match unsafe { graph_ref(g) } {
        Some(g) => g.edge_count() as u64,
        None => 0,
    }
}

/// The graph's monotonic mutation version — an O(1) "did anything change?" read
/// for `useSyncExternalStore`-style snapshots. Always available (mutation is
/// core), so a minimal frontend build still gets reactive snapshots.
///
/// # Safety
/// `g` must be a valid graph handle.
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_version(g: *const Graph) -> u64 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match unsafe { graph_ref(g) } {
        Some(g) => g.version(),
        None => 0,
    }
}

/// The per-token change epoch for a label / edge-type / property-key `name`
/// (0 if never touched) — used for finer invalidation than the global version.
///
/// # Safety
/// `g` valid; `name_ptr`/`name_len` valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn lnk_graph_epoch(
    g: *const Graph,
    name_ptr: *const u8,
    name_len: usize,
) -> u64 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match (unsafe { graph_ref(g) }, unsafe {
        in_str(name_ptr, name_len)
    }) {
        (Some(g), Some(name)) => g.epoch(name),
        _ => 0,
    }
}

/// Declare an opt-in secondary index — the single consolidated index-creation
/// entry point (backfills existing elements, then stays current; idempotent).
///
/// `element`: 0 = vertex, 1 = edge. `kind`: 0 = hash (equality seek, one key —
/// turns `WHERE x.k = …` into a seek), 1 = interval (RI-tree over an edge
/// `[k0, k1)` temporal pair — an as-of/overlap predicate seeds from it; two of
/// them, valid `[vf,vt)` + transaction `[tf,tt)`, cover a bitemporal as-of).
///
/// Keys are passed positionally: `k0` (required) and an optional `k1` (the hi key
/// of an interval index; pass a null `k1_ptr` for a one-key hash index).
///
/// Returns 0 on success, -1 on a null/bad-UTF-8 error or an unsupported
/// (element, kind) combination (e.g. a vertex interval index, or an interval
/// index missing its second key).
///
/// # Safety
/// `g` is a valid, uniquely-borrowed `*mut Graph`; `k0_ptr`/`k0_len` a valid UTF-8
/// slice; `k1_ptr`/`k1_len` a valid UTF-8 slice when `k1_ptr` is non-null.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_index(
    g: *mut Graph,
    element: u8,
    kind: u8,
    k0_ptr: *const u8,
    k0_len: usize,
    k1_ptr: *const u8,
    k1_len: usize,
) -> i32 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires valid (or null).
    let g = unsafe { graph_mut(g) };
    // SAFETY: k0_ptr/k0_len is the caller-supplied buffer this fn's # Safety contract requires valid.
    let k0 = unsafe { in_str(k0_ptr, k0_len) };
    let k1 = if k1_ptr.is_null() {
        None
    } else {
        // SAFETY: k1_ptr/k1_len is a valid UTF-8 slice when non-null, per this fn's # Safety contract.
        unsafe { in_str(k1_ptr, k1_len) }
    };
    let (Some(g), Some(k0)) = (g, k0) else {
        return -1;
    };
    match (element, kind) {
        (0, 0) => {
            g.create_vertex_index(k0);
            0
        }
        (1, 0) => {
            g.create_edge_index(k0);
            0
        }
        (1, 1) => match k1 {
            Some(k1) => {
                g.create_edge_interval_index(k0, k1);
                0
            }
            None => -1, // interval index requires a second key
        },
        _ => -1, // unsupported (e.g. vertex interval index)
    }
}

/// Declare a UNIQUE constraint on `(label, key)`: at most one live vertex with
/// `label` may hold a given non-null value for `key`. Creates the backing vertex
/// index. Returns 0 on success, -1 on a null / bad-UTF-8 error, and **-2** if the
/// current data already violates it (surfaced as `ConstraintViolation` by the
/// caller). See `docs/design/gql-extensions.md` §3.
///
/// # Safety
/// `g` is a valid, uniquely-borrowed `*mut Graph`; both ptr/len pairs are valid
/// UTF-8 slices.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_unique_constraint(
    g: *mut Graph,
    label_ptr: *const u8,
    label_len: usize,
    key_ptr: *const u8,
    key_len: usize,
) -> i32 {
    let (Some(g), Some(label), Some(key)) = (
        // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
        unsafe { graph_mut(g) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(label_ptr, label_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(key_ptr, key_len) },
    ) else {
        return -1;
    };
    match g.create_unique_constraint(label, key) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Declare a REQUIRED constraint on `(label, key)`: every live vertex with
/// `label` must hold a present, non-null value for `key`. Returns 0 on success,
/// -1 on a null / bad-UTF-8 error, and **-2** if the current data already
/// violates it (surfaced as `ConstraintViolation` by the caller).
///
/// # Safety
/// As [`lnk_create_unique_constraint`].
#[no_mangle]
pub unsafe extern "C" fn lnk_create_required_constraint(
    g: *mut Graph,
    label_ptr: *const u8,
    label_len: usize,
    key_ptr: *const u8,
    key_len: usize,
) -> i32 {
    let (Some(g), Some(label), Some(key)) = (
        // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
        unsafe { graph_mut(g) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(label_ptr, label_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(key_ptr, key_len) },
    ) else {
        return -1;
    };
    match g.create_required_constraint(label, key) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Declare a TYPE constraint on `(label, key)` requiring the scalar type named by
/// `type_ptr` (string/number/boolean/date/datetime/duration/list). Returns 0 on
/// success, -1 on a null / bad-UTF-8 error, **-2** if the current data already
/// violates it, and **-3** for an unknown type name.
///
/// # Safety
/// As [`lnk_create_unique_constraint`], plus `type_ptr`/`type_len` a valid slice.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_type_constraint(
    g: *mut Graph,
    label_ptr: *const u8,
    label_len: usize,
    key_ptr: *const u8,
    key_len: usize,
    type_ptr: *const u8,
    type_len: usize,
) -> i32 {
    let (Some(g), Some(label), Some(key), Some(ty)) = (
        // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
        unsafe { graph_mut(g) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(label_ptr, label_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(key_ptr, key_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(type_ptr, type_len) },
    ) else {
        return -1;
    };
    match g.create_type_constraint(label, key, ty) {
        Ok(()) => 0,
        Err(e) if e.code == crate::error_codes::ErrorCode::InvalidValue => -3,
        Err(_) => -2,
    }
}

/// Declare a UNIQUE constraint on `(edge_type, key)`: at most one live edge of
/// `edge_type` may hold a given non-null value for `key`. Creates the backing
/// edge index. Edge analogue of [`lnk_create_unique_constraint`]. Returns 0 on
/// success, -1 on a null / bad-UTF-8 error, and **-2** if the current data
/// already violates it.
///
/// # Safety
/// `g` is a valid, uniquely-borrowed `*mut Graph`; both ptr/len pairs are valid
/// UTF-8 slices.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_edge_unique_constraint(
    g: *mut Graph,
    etype_ptr: *const u8,
    etype_len: usize,
    key_ptr: *const u8,
    key_len: usize,
) -> i32 {
    let (Some(g), Some(etype), Some(key)) = (
        // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
        unsafe { graph_mut(g) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(etype_ptr, etype_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(key_ptr, key_len) },
    ) else {
        return -1;
    };
    match g.create_edge_unique_constraint(etype, key) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Declare a REQUIRED constraint on `(edge_type, key)`: every live edge of
/// `edge_type` must hold a present, non-null value for `key`. Edge analogue of
/// [`lnk_create_required_constraint`]. Returns 0 on success, -1 on error, **-2**
/// if the current data already violates it.
///
/// # Safety
/// As [`lnk_create_edge_unique_constraint`].
#[no_mangle]
pub unsafe extern "C" fn lnk_create_edge_required_constraint(
    g: *mut Graph,
    etype_ptr: *const u8,
    etype_len: usize,
    key_ptr: *const u8,
    key_len: usize,
) -> i32 {
    let (Some(g), Some(etype), Some(key)) = (
        // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
        unsafe { graph_mut(g) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(etype_ptr, etype_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(key_ptr, key_len) },
    ) else {
        return -1;
    };
    match g.create_edge_required_constraint(etype, key) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Declare a TYPE constraint on `(edge_type, key)` requiring the scalar type
/// named by `type_ptr`. Edge analogue of [`lnk_create_type_constraint`]. Returns
/// 0 on success, -1 on error, **-2** if the current data already violates it, and
/// **-3** for an unknown type name.
///
/// # Safety
/// As [`lnk_create_edge_unique_constraint`], plus `type_ptr`/`type_len` a valid slice.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_edge_type_constraint(
    g: *mut Graph,
    etype_ptr: *const u8,
    etype_len: usize,
    key_ptr: *const u8,
    key_len: usize,
    type_ptr: *const u8,
    type_len: usize,
) -> i32 {
    let (Some(g), Some(etype), Some(key), Some(ty)) = (
        // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
        unsafe { graph_mut(g) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(etype_ptr, etype_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(key_ptr, key_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(type_ptr, type_len) },
    ) else {
        return -1;
    };
    match g.create_edge_type_constraint(etype, key, ty) {
        Ok(()) => 0,
        Err(e) if e.code == crate::error_codes::ErrorCode::InvalidValue => -3,
        Err(_) => -2,
    }
}

/// Declare a CARDINALITY constraint bounding the degree of every vertex carrying
/// `label` over `etype` in `direction` (0 = out / the vertex is the edge source,
/// 1 = in / the target) to `min..=max`, where `max < 0` means unbounded. Existing
/// data is scanned at declare time. Returns **0** on success, **-1** if the
/// current data already violates it (surfaced as `ConstraintViolation` by the
/// caller), and **-2** on a null graph / bad-UTF-8 slice.
///
/// # Safety
/// `g` is a valid, uniquely-borrowed `*mut Graph`; both ptr/len pairs are valid
/// UTF-8 slices.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_cardinality_constraint(
    g: *mut Graph,
    label_ptr: *const u8,
    label_len: usize,
    etype_ptr: *const u8,
    etype_len: usize,
    direction: u8,
    min: u32,
    max: i64,
) -> i32 {
    let (Some(g), Some(label), Some(etype)) = (
        // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
        unsafe { graph_mut(g) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(label_ptr, label_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(etype_ptr, etype_len) },
    ) else {
        return -2;
    };
    let max_opt = if max < 0 { None } else { Some(max as u32) };
    match g.create_cardinality_constraint(label, etype, direction, min, max_opt) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// Declare a custom VALIDATOR on `label` (a vertex label OR an edge type): every
/// element carrying the label must satisfy the GQL boolean `predicate`, with the
/// element bound to `var`. SQL-`CHECK` semantics — rejected only on a definite
/// `false`, a null/unknown result passes.
///
/// Returns `0` on success, `-1` if existing data already violates the predicate
/// (`ConstraintViolation`), `-2` if the predicate can't be parsed (`Syntax`), or
/// `-3` for a null handle. (A bad-UTF-8 string arg — never sent by the TS encoder
/// — also maps to `-2`, a bad-predicate input.)
///
/// # Safety
/// `g` is a valid, uniquely-borrowed `*mut Graph`; all three ptr/len pairs are
/// valid UTF-8 slices.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_validator(
    g: *mut Graph,
    label_ptr: *const u8,
    label_len: usize,
    var_ptr: *const u8,
    var_len: usize,
    pred_ptr: *const u8,
    pred_len: usize,
) -> i32 {
    if g.is_null() || label_ptr.is_null() || var_ptr.is_null() || pred_ptr.is_null() {
        return -3;
    }
    let (Some(label), Some(var), Some(predicate)) = (
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(label_ptr, label_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(var_ptr, var_len) },
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        unsafe { in_str(pred_ptr, pred_len) },
    ) else {
        return -2;
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return -3;
    };
    match g.create_validator(label, var, predicate) {
        Ok(()) => 0,
        Err(e) if e.code == ErrorCode::Syntax => -2,
        Err(_) => -1,
    }
}

/// Declare a graph-level INVARIANT `name` = a whole-graph GQL `query` that must
/// hold after every write transaction. `false`-only-fails: VIOLATED iff any cell
/// in the query's result set is boolean `false` (`true`/`null`/non-boolean/empty
/// all hold). A declare-time run rejects if the current graph already violates it.
///
/// Returns `0` on success, `-1` if existing data already violates the invariant
/// (`ConstraintViolation`), `-2` if the query can't be parsed (`Syntax`), or `-3`
/// for a null handle. (A bad-UTF-8 string arg — never sent by the TS encoder —
/// also maps to `-2`, a bad-query input.)
///
/// # Safety
/// `g` is a valid, uniquely-borrowed `*mut Graph`; the two `(ptr, len)` pairs
/// describe live UTF-8 byte ranges.
#[no_mangle]
pub unsafe extern "C" fn lnk_create_invariant(
    g: *mut Graph,
    name_ptr: *const u8,
    name_len: usize,
    query_ptr: *const u8,
    query_len: usize,
) -> i32 {
    if g.is_null() || name_ptr.is_null() || query_ptr.is_null() {
        return -3;
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let (Some(name), Some(query)) = (unsafe { in_str(name_ptr, name_len) }, unsafe {
        in_str(query_ptr, query_len)
    }) else {
        return -2;
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return -3;
    };
    match g.create_invariant(name, query) {
        Ok(()) => 0,
        Err(e) if e.code == ErrorCode::Syntax => -2,
        Err(_) => -1,
    }
}

/// Open a transaction frame (R-TX): an atomic mutation boundary with rollback +
/// deferred constraint checks. Writes still apply eagerly (read-your-writes), but
/// record an inverse op; the outermost commit runs the built-in constraint checks
/// against the fully-staged graph. Nesting joins the outer frame. Returns 0 on
/// success, -1 on a null graph.
///
/// # Safety
/// `g` is a valid, uniquely-borrowed `*mut Graph`.
#[no_mangle]
pub unsafe extern "C" fn lnk_begin_tx(g: *mut Graph) -> i32 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match unsafe { graph_mut(g) } {
        Some(g) => {
            g.begin_tx();
            0
        }
        None => -1,
    }
}

/// Commit the current transaction frame. Returns **0** on success (or an inner
/// commit that the outermost frame will finalize), **-1** if a deferred
/// constraint check failed — the transaction has already been rolled back, and
/// the caller surfaces this as `ConstraintViolation` — **-2** if no transaction
/// is open, and **-3** on a null graph.
///
/// # Safety
/// As [`lnk_begin_tx`].
#[no_mangle]
pub unsafe extern "C" fn lnk_commit_tx(g: *mut Graph) -> i32 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match unsafe { graph_mut(g) } {
        Some(g) => match g.commit_tx() {
            Ok(()) => 0,
            Err(crate::graph::TxCommitError::NoTx) => -2,
            Err(_) => -1, // Required / Type / Unique — all surface as ConstraintViolation
        },
        None => -3,
    }
}

/// Roll the current transaction back: replay the undo log newest-first. A no-op
/// if no transaction is open. Returns 0 on success, -1 on a null graph.
///
/// # Safety
/// As [`lnk_begin_tx`].
#[no_mangle]
pub unsafe extern "C" fn lnk_rollback_tx(g: *mut Graph) -> i32 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    match unsafe { graph_mut(g) } {
        Some(g) => {
            g.rollback_tx();
            0
        }
        None => -1,
    }
}

/// Drop a vertex property index (no-op if absent). Returns 0 on success, -1 on
/// error.
///
/// # Safety
/// As [`lnk_create_vertex_index`].
#[no_mangle]
pub unsafe extern "C" fn lnk_drop_vertex_index(
    g: *mut Graph,
    name_ptr: *const u8,
    name_len: usize,
) -> i32 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let (Some(g), Some(name)) = (unsafe { graph_mut(g) }, unsafe {
        in_str(name_ptr, name_len)
    }) else {
        return -1;
    };
    match g.drop_vertex_index(name) {
        Ok(()) => 0,
        Err(_) => -2, // backs a unique constraint (InvalidGraphOp)
    }
}

/// Drop an edge property index. Edge analogue of [`lnk_drop_vertex_index`].
///
/// # Safety
/// As [`lnk_create_vertex_index`].
#[no_mangle]
pub unsafe extern "C" fn lnk_drop_edge_index(
    g: *mut Graph,
    name_ptr: *const u8,
    name_len: usize,
) -> i32 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let (Some(g), Some(name)) = (unsafe { graph_mut(g) }, unsafe {
        in_str(name_ptr, name_len)
    }) else {
        return -1;
    };
    match g.drop_edge_index(name) {
        Ok(()) => 0,
        Err(_) => -2, // backs a unique constraint (InvalidGraphOp)
    }
}

/// The currently-indexed vertex property keys as a JSON string array (e.g.
/// `["age","name"]`), returned as an owned buffer (free with [`lnk_free_buf`]);
/// its byte length is written to `out_len`.
///
/// # Safety
/// `g` valid; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_vertex_indexes(g: *const Graph, out_len: *mut usize) -> *mut u8 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let keys = match unsafe { graph_ref(g) } {
        Some(g) => g.vertex_indexes(),
        None => Vec::new(),
    };
    // SAFETY: index_keys_buf writes only through out_len, which this fn's # Safety contract requires be a valid, writable pointer.
    unsafe { index_keys_buf(&keys, out_len) }
}

/// The currently-indexed edge property keys as a JSON string array. Edge
/// analogue of [`lnk_vertex_indexes`].
///
/// # Safety
/// `g` valid; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_edge_indexes(g: *const Graph, out_len: *mut usize) -> *mut u8 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let keys = match unsafe { graph_ref(g) } {
        Some(g) => g.edge_indexes(),
        None => Vec::new(),
    };
    // SAFETY: index_keys_buf writes only through out_len, which this fn's # Safety contract requires be a valid, writable pointer.
    unsafe { index_keys_buf(&keys, out_len) }
}

/// The full active schema as a JSON array of replayable `SchemaOp` objects (see
/// [`Graph::dump_schema`]) — constraints, validators, invariants, and indexes.
/// The snapshot codec persists this alongside the graph NDJSON so a cold boot can
/// replay the schema the data alone can't reconstruct. Free with [`lnk_free_buf`].
///
/// # Safety
/// `g` valid; `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_dump_schema(g: *const Graph, out_len: *mut usize) -> *mut u8 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let json = match unsafe { graph_ref(g) } {
        Some(g) => g.dump_schema(),
        None => String::from("[]"),
    };
    let bytes = json.into_bytes().into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// Encode a key list as a JSON string array into an owned buffer for the two
/// `*_indexes` exports (free with [`lnk_free_buf`]).
unsafe fn index_keys_buf(keys: &[String], out_len: *mut usize) -> *mut u8 {
    let mut json = String::from("[");
    for (i, k) in keys.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        crate::jsonfmt::push_json_str(&mut json, k);
    }
    json.push(']');
    let bytes = json.into_bytes().into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// The distinct values of property `key` across the vertices touched by the most
/// recent committed write — that write's content-derived CDC value-scope — as a
/// JSON string array (free with [`lnk_free_buf`]). Empty (`[]`) when the last write
/// touched no vertex carrying `key`. Null graph / bad UTF-8 → an empty array.
///
/// # Safety
/// `g` valid + shared for this call; `key_ptr`/`key_len` valid UTF-8; `out_len`
/// writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_last_write_scope(
    g: *const Graph,
    key_ptr: *const u8,
    key_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    // SAFETY: g is the caller-supplied handle, and the ptr/len the caller-supplied buffer, that this fn's # Safety contract requires be valid (or null -> None).
    let scope = match (unsafe { graph_ref(g) }, unsafe { in_str(key_ptr, key_len) }) {
        (Some(g), Some(key)) => g.last_write_scope(key),
        _ => Vec::new(),
    };
    // SAFETY: index_keys_buf writes only through out_len, which this fn's # Safety contract requires be a valid, writable pointer.
    unsafe { index_keys_buf(&scope, out_len) }
}

/// Parse + run a GQL-subset query, writing the `(count, sum)` signature.
/// Returns 0 on success, -1 on a parse/null error.
///
/// # Safety
/// `g` valid; `q_ptr`/`q_len` valid UTF-8; out pointers writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_query(
    g: *const Graph,
    q_ptr: *const u8,
    q_len: usize,
    out_count: *mut u64,
    out_sum: *mut f64,
    out_checksum: *mut u64,
) -> i32 {
    // SAFETY: g is the caller-supplied handle, and the ptr/len the caller-supplied buffer, that this fn's # Safety contract requires be valid (or null -> None).
    let (Some(g), Some(q)) = (unsafe { graph_ref(g) }, unsafe { in_str(q_ptr, q_len) }) else {
        return -1;
    };
    let parsed = match query::parse(q) {
        Ok(p) => p,
        Err(_) => return -1,
    };
    let r = parsed.run(g);
    // SAFETY: the caller's # Safety contract requires out_count be a valid, writable pointer.
    unsafe { *out_count = r.count };
    // SAFETY: the caller's # Safety contract requires out_sum be a valid, writable pointer.
    unsafe { *out_sum = r.sum };
    // SAFETY: the caller's # Safety contract requires out_checksum be a valid, writable pointer.
    unsafe { *out_checksum = r.checksum };
    0
}

/// Run many queries (newline-joined) in ONE crossing — amortizes the per-call
/// FFI tax. Results are written into the caller's `count`/`sum`/`checksum`
/// arrays (each sized to the query count). Returns the number run, or -1.
///
/// # Safety
/// `g` valid; `q_ptr`/`q_len` valid UTF-8; out arrays sized to the number of
/// newline-separated queries.
#[no_mangle]
pub unsafe extern "C" fn lnk_query_batch(
    g: *const Graph,
    q_ptr: *const u8,
    q_len: usize,
    out_count: *mut u64,
    out_sum: *mut f64,
    out_checksum: *mut u64,
) -> i64 {
    // SAFETY: g is the caller-supplied handle, and the ptr/len the caller-supplied buffer, that this fn's # Safety contract requires be valid (or null -> None).
    let (Some(g), Some(text)) = (unsafe { graph_ref(g) }, unsafe { in_str(q_ptr, q_len) }) else {
        return -1;
    };
    let mut i = 0isize;
    for line in text.split('\n') {
        if line.trim().is_empty() {
            continue;
        }
        let r = match query::parse(line) {
            Ok(p) => p.run(g),
            Err(_) => return -1,
        };
        // SAFETY: index i < the run count, so out_count.offset(i) is in-bounds of the caller-provided output array (# Safety contract), and the slot is writable.
        unsafe { *out_count.offset(i) = r.count };
        // SAFETY: index i < the run count, so out_sum.offset(i) is in-bounds of the caller-provided output array (# Safety contract), and the slot is writable.
        unsafe { *out_sum.offset(i) = r.sum };
        // SAFETY: index i < the run count, so out_checksum.offset(i) is in-bounds of the caller-provided output array (# Safety contract), and the slot is writable.
        unsafe { *out_checksum.offset(i) = r.checksum };
        i += 1;
    }
    i as i64
}

/// Decode the optional params-JSON argument shared by the query entry points.
/// A null/empty pointer means "no params". On a decode failure the last-error
/// report is set and `Err(())` is returned.
///
/// # Safety
/// `p_ptr` is either null or valid for `p_len` bytes of UTF-8.
#[cfg(feature = "gql")]
unsafe fn decode_params(p_ptr: *const u8, p_len: usize) -> Result<crate::gql::eval::Params, ()> {
    if p_ptr.is_null() || p_len == 0 {
        return Ok(crate::gql::eval::Params::new());
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let text = match unsafe { in_str(p_ptr, p_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "params bytes are not valid UTF-8");
            return Err(());
        }
    };
    match crate::gql::params_from_json(text) {
        Ok(p) => Ok(p),
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            Err(())
        }
    }
}

/// Parse + run a query and return the *real* result rows as a JSON document
/// (`{"columns":[...],"rows":[[...]]}`). The buffer is heap-allocated here;
/// the caller must return it via `lnk_free_buf` with the written length. The
/// byte length is written to `out_len`. Returns null on a parse/null/UTF-8
/// error (and leaves `out_len` untouched).
///
/// `p_ptr`/`p_len` optionally carry a flat JSON object of `$name` bindings
/// (see [`crate::gql::params`]); pass null for none. Param values bind to
/// already-parsed `$name` slots at execute time — they never touch the parser,
/// which is the injection-safety contract of the whole params surface.
///
/// This is the row-returning counterpart to `lnk_query` (which only yields the
/// `(count, sum, checksum)` benchmark fingerprint). JSON is the carrier so the
/// same symbol serves both bun:ffi and a future wasm-bindgen binding with one
/// buffer crossing instead of per-cell marshalling. The graph handle is `*mut`
/// because a query may mutate it (`INSERT`/`SET`/`REMOVE`/`DELETE`).
///
/// # Safety
/// `g` valid and exclusively borrowed for this call; `q_ptr`/`q_len` valid
/// UTF-8; `p_ptr` null or valid for `p_len` bytes; `out_len` writable.
#[cfg(feature = "gql")]
#[no_mangle]
pub unsafe extern "C" fn lnk_query_rows(
    g: *mut Graph,
    q_ptr: *const u8,
    q_len: usize,
    p_ptr: *const u8,
    p_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if g.is_null() || q_ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null graph or query pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let q = match unsafe { in_str(q_ptr, q_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "query bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: p_ptr/p_len is the caller-supplied params buffer this fn's # Safety contract requires be null or a valid readable range.
    let Ok(params) = (unsafe { decode_params(p_ptr, p_len) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    // Route to the full GQL engine (the complete ISO-subset port). A parse
    // failure carries its source offset; an execution failure (an unsupported
    // clause in this partial engine) carries the engine's message.
    let parsed = match crate::gql::parse_with_max_chain(q, g.max_operator_chain()) {
        Ok(p) => p,
        Err(e) => {
            crate::ffi_error::set(
                ErrorCode::Syntax,
                &e.message,
                &format!("{{\"pos\":{}}}", e.pos),
            );
            return std::ptr::null_mut();
        }
    };
    let rowset = match parsed.execute(g, &params) {
        Ok(rs) => rs,
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            return std::ptr::null_mut();
        }
    };
    let bytes = rowset.to_json().into_bytes().into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// Run a native graph algorithm (`degree`, `pagerank`, `connectedComponents`,
/// `labelPropagation`, `shortestPath`) over the graph in a single call — the whole
/// computation stays in the engine, no per-iteration FFI. `name` selects the
/// algorithm; `cfg_ptr`/`cfg_len` carry its JSON config (`{}` / null = defaults). If
/// the config sets `writeProperty`, each vertex's result is written back to that
/// property (mutating the graph) before returning. Returns the `{columns,rows}` JSON
/// document (free with [`lnk_free_buf`]); the byte length lands in `out_len`. Returns
/// null on an unknown algorithm / bad config / UTF-8 error.
///
/// The row/JSON carrier mirrors [`lnk_query_rows`] so both bun:ffi and wasm-bindgen
/// share one buffer crossing. The handle is `*mut` because `writeProperty` mutates.
///
/// # Safety
/// `g` valid and exclusively borrowed for this call; `name_ptr`/`name_len` valid
/// UTF-8; `cfg_ptr` null or valid for `cfg_len` bytes; `out_len` writable.
#[cfg(feature = "ndjson")]
#[no_mangle]
pub unsafe extern "C" fn lnk_algo(
    g: *mut Graph,
    name_ptr: *const u8,
    name_len: usize,
    cfg_ptr: *const u8,
    cfg_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if g.is_null() || name_ptr.is_null() {
        crate::ffi_error::set_code(
            crate::error_codes::ErrorCode::Ffi,
            "null graph or algorithm-name pointer",
        );
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let name = match unsafe { in_str(name_ptr, name_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(
                crate::error_codes::ErrorCode::Ffi,
                "algorithm name bytes are not valid UTF-8",
            );
            return std::ptr::null_mut();
        }
    };
    let cfg = if cfg_ptr.is_null() {
        ""
    } else {
        // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
        match unsafe { in_str(cfg_ptr, cfg_len) } {
            Some(s) => s,
            None => {
                crate::ffi_error::set_code(
                    crate::error_codes::ErrorCode::Ffi,
                    "algorithm config bytes are not valid UTF-8",
                );
                return std::ptr::null_mut();
            }
        }
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    let rowset = match crate::algo::run(g, name, cfg) {
        Ok(rs) => rs,
        Err(msg) => {
            crate::ffi_error::set_code(crate::error_codes::ErrorCode::Ffi, &msg);
            return std::ptr::null_mut();
        }
    };
    let bytes = rowset.to_json().into_bytes().into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// Parse + run a query and return the result as an **Apache Arrow** columnar
/// blob (see [`crate::arrow`]) instead of JSON. The buffer is heap-allocated here
/// **8-byte aligned** (so the caller's `Float64Array`/`Int32Array` views over the
/// column buffers are valid) and must be returned via `lnk_free_arrow` with the
/// written length. The byte length is written to `out_len`. Returns null on a
/// parse / null / UTF-8 error (and leaves `out_len` untouched).
///
/// The columnar carrier is zero-copy on the caller side: a query result is
/// single-owner and consume-once, so the caller views the Arrow buffers in place
/// (browser: `apache-arrow` `makeData`; server: bun:ffi typed-array over the
/// pointer) — no serialize here, no parse there.
///
/// `p_ptr`/`p_len` optionally carry a flat JSON object of `$name` bindings,
/// exactly as on [`lnk_query_rows`]; pass null for none.
///
/// # Safety
/// `g` valid and exclusively borrowed for this call; `q_ptr`/`q_len` valid
/// UTF-8; `p_ptr` null or valid for `p_len` bytes; `out_len` writable.
#[cfg(feature = "arrow")]
#[no_mangle]
pub unsafe extern "C" fn lnk_query_arrow(
    g: *mut Graph,
    q_ptr: *const u8,
    q_len: usize,
    p_ptr: *const u8,
    p_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if g.is_null() || q_ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null graph or query pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let q = match unsafe { in_str(q_ptr, q_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "query bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: p_ptr/p_len is the caller-supplied params buffer this fn's # Safety contract requires be null or a valid readable range.
    let Ok(params) = (unsafe { decode_params(p_ptr, p_len) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    // The error rides the last-error channel, never this return pointer — so the
    // binary Arrow carrier below stays a pure column blob with no error union.
    let parsed = match crate::gql::parse_with_max_chain(q, g.max_operator_chain()) {
        Ok(p) => p,
        Err(e) => {
            crate::ffi_error::set(
                ErrorCode::Syntax,
                &e.message,
                &format!("{{\"pos\":{}}}", e.pos),
            );
            return std::ptr::null_mut();
        }
    };
    // execute_arrow keeps numeric/bool result columns typed end-to-end (no
    // Val/Value boxing) for the common single-MATCH … RETURN shape.
    let blob = match parsed.execute_arrow(g, &params) {
        Ok(b) => b,
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            return std::ptr::null_mut();
        }
    };
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = blob.len() };
    // 8-byte-aligned copy so the caller can view f64/i32 column buffers directly.
    let len = blob.len().max(1);
    let layout = std::alloc::Layout::from_size_align(len, 8).unwrap();
    // SAFETY: layout has non-zero size (len.max(1)) and valid 8-byte alignment; alloc returns null on failure, checked before use.
    let p = unsafe { std::alloc::alloc(layout) };
    if !p.is_null() {
        // SAFETY: the source blob is valid for blob.len() bytes; the destination is freshly allocated, non-null (checked), sized len >= blob.len(), and non-overlapping.
        unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr(), p, blob.len()) };
    }
    p
}

/// Run a GQL query and return standard **Apache Arrow IPC** bytes (`file != 0` →
/// the file / Feather-v2 layout, else the IPC stream layout) — the whole
/// query→IPC path runs natively, so a caller never re-encodes in JS. Same
/// contract as [`lnk_query_arrow`] (last-error channel, `lnk_free_arrow` with
/// `out_len`, null on error). Returns null (and leaves `out_len` untouched) on a
/// parse / null / UTF-8 error.
///
/// # Safety
/// As [`lnk_query_arrow`]: `g` valid + exclusively borrowed; `q_ptr`/`q_len` valid
/// UTF-8; `p_ptr` null or valid for `p_len` bytes; `out_len` writable.
#[cfg(feature = "arrow")]
#[no_mangle]
pub unsafe extern "C" fn lnk_query_arrow_ipc(
    g: *mut Graph,
    q_ptr: *const u8,
    q_len: usize,
    p_ptr: *const u8,
    p_len: usize,
    file: u32,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if g.is_null() || q_ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null graph or query pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let q = match unsafe { in_str(q_ptr, q_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "query bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: p_ptr/p_len is the caller-supplied params buffer this fn's # Safety contract requires be null or a valid readable range.
    let Ok(params) = (unsafe { decode_params(p_ptr, p_len) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    let parsed = match crate::gql::parse_with_max_chain(q, g.max_operator_chain()) {
        Ok(p) => p,
        Err(e) => {
            crate::ffi_error::set(
                ErrorCode::Syntax,
                &e.message,
                &format!("{{\"pos\":{}}}", e.pos),
            );
            return std::ptr::null_mut();
        }
    };
    // Reuse the typed Arrow column path, then frame it as IPC natively.
    let blob = match parsed.execute_arrow(g, &params) {
        Ok(b) => b,
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            return std::ptr::null_mut();
        }
    };
    let ipc = crate::arrow::arrow_ipc_from_blob(&blob, file != 0);
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = ipc.len() };
    let len = ipc.len().max(1);
    let layout = std::alloc::Layout::from_size_align(len, 8).unwrap();
    // SAFETY: layout has non-zero size (len.max(1)) and valid 8-byte alignment; alloc returns null on failure, checked before use.
    let p = unsafe { std::alloc::alloc(layout) };
    if !p.is_null() {
        // SAFETY: the source blob is valid for blob.len() bytes; the destination is freshly allocated, non-null (checked), sized len >= blob.len(), and non-overlapping.
        unsafe { std::ptr::copy_nonoverlapping(ipc.as_ptr(), p, ipc.len()) };
    }
    p
}

// --- prepared statements -----------------------------------------------------
//
// `lnk_prepare` lexes/parses/lowers a query ONCE into an owned `Prepared`
// (opaque handle); `lnk_prepared_query_rows`/`_arrow` execute it against a graph
// with fresh params, skipping the re-parse every `lnk_query_*` call pays. Free
// the handle with `lnk_prepared_free`. The `Prepared` is graph-independent — one
// compile can run against any graph.

/// Compile a GQL query into a reusable prepared statement — an owning
/// `*mut Prepared` (free with [`lnk_prepared_free`]), or null on a parse / bad-
/// UTF-8 error (details in the last-error channel).
///
/// # Safety
/// `q_ptr`/`q_len` a valid UTF-8 slice.
#[no_mangle]
pub unsafe extern "C" fn lnk_prepare(
    q_ptr: *const u8,
    q_len: usize,
    max_chain: u64,
) -> *mut crate::gql::Prepared {
    crate::ffi_error::begin();
    if q_ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null query pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let q = match unsafe { in_str(q_ptr, q_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "query bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    match crate::gql::prepare_with_max_chain(q, max_chain as usize) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(e) => {
            crate::ffi_error::set(
                ErrorCode::Syntax,
                &e.message,
                &format!("{{\"pos\":{}}}", e.pos),
            );
            std::ptr::null_mut()
        }
    }
}

/// Free a prepared statement from [`lnk_prepare`].
///
/// # Safety
/// `p` must come from [`lnk_prepare`] and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn lnk_prepared_free(p: *mut crate::gql::Prepared) {
    if !p.is_null() {
        // SAFETY: the pointer came from this module's matching Box::into_raw, per the fn's # Safety contract, and is freed exactly once.
        drop(unsafe { Box::from_raw(p) });
    }
}

/// Execute a prepared statement against `g` with `params`, returning the
/// `{columns, rows}` JSON buffer (free with [`lnk_free_buf`]) — the prepared
/// twin of [`lnk_query_rows`], minus the parse.
///
/// # Safety
/// `p` from [`lnk_prepare`]; `g` valid + uniquely borrowed; params a valid UTF-8
/// slice (or null); `out_len` writable.
#[no_mangle]
pub unsafe extern "C" fn lnk_prepared_query_rows(
    p: *const crate::gql::Prepared,
    g: *mut Graph,
    p_ptr: *const u8,
    p_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if p.is_null() || g.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null prepared or graph pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: p_ptr/p_len is the caller-supplied params buffer this fn's # Safety contract requires be null or a valid readable range.
    let Ok(params) = (unsafe { decode_params(p_ptr, p_len) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: p is the caller-supplied Prepared handle this fn's # Safety contract requires be valid.
    let prepared = unsafe { &*p };
    let rowset = match prepared.execute(g, &params) {
        Ok(rs) => rs,
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            return std::ptr::null_mut();
        }
    };
    let bytes = rowset.to_json().into_bytes().into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// Execute a prepared statement → Arrow ("ARW1") blob (free with
/// [`lnk_free_arrow`]). The prepared twin of [`lnk_query_arrow`].
///
/// # Safety
/// As [`lnk_prepared_query_rows`].
#[no_mangle]
pub unsafe extern "C" fn lnk_prepared_query_arrow(
    p: *const crate::gql::Prepared,
    g: *mut Graph,
    p_ptr: *const u8,
    p_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if p.is_null() || g.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null prepared or graph pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: p_ptr/p_len is the caller-supplied params buffer this fn's # Safety contract requires be null or a valid readable range.
    let Ok(params) = (unsafe { decode_params(p_ptr, p_len) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    // SAFETY: p is the caller-supplied Prepared handle this fn's # Safety contract requires be valid.
    let prepared = unsafe { &*p };
    let blob = match prepared.execute_arrow(g, &params) {
        Ok(b) => b,
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            return std::ptr::null_mut();
        }
    };
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = blob.len() };
    let len = blob.len().max(1);
    let layout = std::alloc::Layout::from_size_align(len, 8).unwrap();
    // SAFETY: layout has non-zero size (len.max(1)) and valid 8-byte alignment; alloc returns null on failure, checked before use.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if !ptr.is_null() {
        // SAFETY: the source blob is valid for blob.len() bytes; the destination is freshly allocated, non-null (checked), sized len >= blob.len(), and non-overlapping.
        unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr(), ptr, blob.len()) };
    }
    ptr
}

/// Free a buffer returned by `lnk_query_arrow`. Must use the same `len` that was
/// written to `out_len` (the allocation is 8-byte aligned).
///
/// # Safety
/// `ptr`/`len` must come from `lnk_query_arrow`.
#[cfg(feature = "arrow")]
#[no_mangle]
pub unsafe extern "C" fn lnk_free_arrow(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        let len = len.max(1);
        // SAFETY: ptr came from this module's std::alloc::alloc with the same 8-byte layout, per the fn's # Safety contract, and is freed exactly once.
        unsafe { std::alloc::dealloc(ptr, std::alloc::Layout::from_size_align(len, 8).unwrap()) };
    }
}

/// Parse + run a **textual Gremlin** query (the Groovy wire form, e.g.
/// `g.V().has('name','marko').out('knows').values('name')`) and return the
/// results as a JSON array. Heap-allocated; free with `lnk_free_buf`. Byte length
/// written to `out_len`. Null on a parse/UTF-8 error (and `out_len` untouched).
///
/// JSON (not Arrow) because Gremlin results are heterogeneous per row — a stream
/// of mixed scalars / elements / maps — which doesn't fit a columnar carrier.
///
/// # Safety
/// `g` valid and exclusively borrowed; `q_ptr`/`q_len` valid UTF-8; `out_len` writable.
#[cfg(feature = "gremlin")]
#[no_mangle]
pub unsafe extern "C" fn lnk_gremlin_json(
    g: *mut Graph,
    q_ptr: *const u8,
    q_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if g.is_null() || q_ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null graph or query pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let q = match unsafe { in_str(q_ptr, q_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "query bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_mut(g) }) else {
        return std::ptr::null_mut();
    };
    let plan = match crate::gremlin::parse(q) {
        Ok(p) => p,
        Err(e) => {
            crate::ffi_error::set_code(ErrorCode::Syntax, &e);
            return std::ptr::null_mut();
        }
    };
    let vals = match crate::gremlin::try_run(g, &plan) {
        Ok(v) => v,
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            return std::ptr::null_mut();
        }
    };
    let bytes = crate::gremlin::exec::results_to_json(g, &vals)
        .into_bytes()
        .into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// Serialize the graph in a named format — `pg-json | pg-text | graphson | csv |
/// ndjson` (see [`crate::codec`]). Returns a heap buffer (free with
/// `lnk_free_buf`) and writes its byte length to `out_len`. Null on an unknown
/// format / null / UTF-8 error.
///
/// # Safety
/// `g` valid; `fmt_ptr`/`fmt_len` valid UTF-8; `out_len` writable.
#[cfg(feature = "codecs")]
#[no_mangle]
pub unsafe extern "C" fn lnk_serialize(
    g: *const Graph,
    fmt_ptr: *const u8,
    fmt_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    crate::ffi_error::begin();
    if g.is_null() || fmt_ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null graph or format pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let fmt = match unsafe { in_str(fmt_ptr, fmt_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "format bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_ref(g) }) else {
        return std::ptr::null_mut();
    };
    match crate::codec::serialize(g, fmt) {
        Ok(s) => {
            let bytes = s.into_bytes().into_boxed_slice();
            // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
            unsafe { *out_len = bytes.len() };
            Box::into_raw(bytes) as *mut u8
        }
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            std::ptr::null_mut()
        }
    }
}

/// Deserialize bytes in a named format into a fresh graph handle (free with
/// `lnk_graph_free`). The format name is the same set as `lnk_serialize`. Null on
/// an unknown format, a parse error, or bad UTF-8.
///
/// # Safety
/// `ptr`/`len` valid UTF-8; `fmt_ptr`/`fmt_len` valid UTF-8.
#[cfg(feature = "codecs")]
#[no_mangle]
pub unsafe extern "C" fn lnk_deserialize(
    ptr: *const u8,
    len: usize,
    fmt_ptr: *const u8,
    fmt_len: usize,
) -> *mut Graph {
    crate::ffi_error::begin();
    if ptr.is_null() || fmt_ptr.is_null() {
        crate::ffi_error::set_code(ErrorCode::Ffi, "null input or format pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let text = match unsafe { in_str(ptr, len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "input bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: the ptr/len here is the caller-supplied buffer this fn's # Safety contract requires be a valid readable range (or null -> None).
    let fmt = match unsafe { in_str(fmt_ptr, fmt_len) } {
        Some(s) => s,
        None => {
            crate::ffi_error::set_code(ErrorCode::Ffi, "format bytes are not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    match crate::codec::deserialize(text, fmt) {
        Ok(g) => Box::into_raw(Box::new(g)),
        // The codec now carries a precise code (UnknownFormat / InvalidJson /
        // InvalidShape / …); surface it directly — no message-matching heuristic.
        Err(e) => {
            crate::ffi_error::set_code(e.code, &e.message);
            std::ptr::null_mut()
        }
    }
}

/// Encode the graph to NDJSON. Returns a heap pointer (free with `lnk_free_buf`)
/// and writes the byte length to `out_len`.
///
/// # Safety
/// `g` valid; `out_len` writable.
#[cfg(feature = "ndjson")]
#[no_mangle]
pub unsafe extern "C" fn lnk_encode_ndjson(g: *const Graph, out_len: *mut usize) -> *mut u8 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let Some(g) = (unsafe { graph_ref(g) }) else {
        return std::ptr::null_mut();
    };
    let bytes = crate::ndjson::encode(g).into_bytes().into_boxed_slice();
    // SAFETY: the caller's # Safety contract requires out_len be a valid, writable pointer.
    unsafe { *out_len = bytes.len() };
    Box::into_raw(bytes) as *mut u8
}

/// # Safety
/// `ptr`/`len` must come from `lnk_encode_ndjson`.
#[no_mangle]
pub unsafe extern "C" fn lnk_free_buf(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        // SAFETY: the pointer came from this module's matching Box::into_raw, per the fn's # Safety contract, and is freed exactly once.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}

/// Serialize the graph and write it straight to a file — the realistic
/// "serialize to disk" path: the bytes never cross back into JS as a string.
/// Returns the number of bytes written, or -1 on error.
///
/// # Safety
/// `g` valid; `path_ptr`/`path_len` valid UTF-8.
///
/// Native-only: there is no filesystem in the browser, so this symbol is absent
/// from the wasm build. Browser callers serialize via `lnk_encode_ndjson` and
/// persist through the host (download, IndexedDB, …) instead.
#[cfg(all(feature = "ndjson", not(target_arch = "wasm32")))]
#[no_mangle]
pub unsafe extern "C" fn lnk_write_ndjson(
    g: *const Graph,
    path_ptr: *const u8,
    path_len: usize,
) -> i64 {
    // SAFETY: g is the caller-supplied handle this fn's # Safety contract requires be a valid graph (or null -> None).
    let (Some(g), Some(path)) = (unsafe { graph_ref(g) }, unsafe {
        in_str(path_ptr, path_len)
    }) else {
        return -1;
    };
    let bytes = crate::ndjson::encode(g).into_bytes();
    match std::fs::write(path, &bytes) {
        Ok(()) => bytes.len() as i64,
        Err(_) => -1,
    }
}
