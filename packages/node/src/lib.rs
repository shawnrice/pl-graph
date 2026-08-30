//! `@lenke/node` — a native Node.js addon for the `lenke-engine` graph engine,
//! built with napi-rs (N-API).
//!
//! This is a THIN marshaling layer over the engine's C FFI (`lnk_*` in
//! `lenke-engine/src/ffi.rs`) — the exact same interface the bun:ffi and wasm
//! backends drive, so it inherits every guarantee the fuzzers establish for that
//! path. The high-level `Backend` (queries, algorithms, codecs, schema DDL,
//! prepared statements) is assembled in JS by `buildEngineBackend` over this thin
//! abi (see `backend.js`); this crate only bridges N-API values ↔ the `lnk_*`
//! boundary. Where the bun path crosses a C ABI and copies pointers by hand, this
//! addon holds the engine in-process (path dep on `lenke-engine`) and calls the
//! same functions directly — no dynamic library to locate at runtime.

use lenke_engine::ffi;
use lenke_engine::ndjson::{field, json_string, parse_json, Json};
use lenke_engine::store::Store;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// A pointer to a live engine `Store`. Wrapped so the off-thread `command_async`
/// task can be `Send`: soundness is the `@lenke/native` facade's single-flight
/// guard — no other native call touches the graph while an async command is
/// pending, so the shared read never races a mutation.
#[derive(Clone, Copy)]
struct StorePtr(*mut Store);
// SAFETY: see the type doc — the facade serializes access around the async task.
unsafe impl Send for StorePtr {}

/// Read (and clear) the calling thread's last engine error into a coded N-API
/// exception `lenke: <op>: <message> [<CODE>]`. The JS adapter (`backend.js`, via
/// `errorFromNapi`) parses the `[CODE]` tail back into a coded `LenkeError`, giving
/// the addon full error-code parity with the bun:ffi and wasm backends.
///
/// # Safety
/// Call only right after a `lnk_*` function returned its `null` / `-1` sentinel.
unsafe fn last_error(op: &str) -> Error {
    let mut len: usize = 0;
    let p = ffi::lnk_last_error_json(&mut len);
    if p.is_null() {
        return Error::new(
            Status::GenericFailure,
            format!("lenke: {op}: unknown error [E_FFI]"),
        );
    }
    let json = String::from_utf8_lossy(std::slice::from_raw_parts(p, len)).into_owned();
    ffi::lnk_free(p, len);
    let (code, message) = match parse_json(&json) {
        Ok(Json::Obj(fields)) => {
            let get = |k: &str| field(&fields, k).and_then(|j| json_string(j).ok());
            (
                get("code").unwrap_or_else(|| "E_FFI".into()),
                get("message").unwrap_or_default(),
            )
        }
        _ => ("E_FFI".into(), json),
    };
    Error::new(
        Status::GenericFailure,
        format!("lenke: {op}: {message} [{code}]"),
    )
}

/// Copy a crate-owned `(ptr, len)` result into an owned `Vec`, then hand the crate
/// buffer back to `lnk_free`. A null ptr is a failure — read the last error.
///
/// # Safety
/// `ptr`/`len` must be a `lnk_*` byte result (or `ptr` null on failure).
unsafe fn take(op: &str, ptr: *mut u8, len: usize) -> Result<Vec<u8>> {
    if ptr.is_null() {
        return Err(last_error(op));
    }
    let v = std::slice::from_raw_parts(ptr, len).to_vec();
    ffi::lnk_free(ptr, len);
    Ok(v)
}

/// A decoded, in-memory columnar graph — a live engine `Store`. napi frees it when
/// the JS object is garbage-collected (`Drop` → `lnk_close`).
#[napi]
pub struct Graph {
    store: StorePtr,
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: `store` was produced by `lnk_open`/`lnk_clone` and is closed once.
        unsafe { ffi::lnk_close(self.store.0) };
    }
}

/// A [`Graph::command_async`] unit of work: runs a `lnk_command` (e.g. an
/// algorithm) on a libuv threadpool thread, reading the graph through a `Send`
/// pointer, then resolves the crate-owned result bytes back on the main thread.
pub struct CommandTask {
    store: StorePtr,
    name: String,
    input: Option<Vec<u8>>,
}

impl Task for CommandTask {
    type Output = Vec<u8>;
    type JsValue = Buffer;

    fn compute(&mut self) -> Result<Self::Output> {
        let (ip, il) = match &self.input {
            Some(b) => (b.as_ptr(), b.len()),
            None => (std::ptr::null(), 0),
        };
        // SAFETY: the facade's single-flight guard keeps this off-thread call from
        // racing any main-thread access to the same store (see `StorePtr`).
        unsafe {
            let mut len = 0usize;
            let p = ffi::lnk_command(self.store.0, self.name.as_ptr(), self.name.len(), ip, il, &mut len);
            take("command", p, len)
        }
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Buffer> {
        Ok(output.into())
    }
}

#[napi]
impl Graph {
    /// Open a store from `bytes` in the given `format` (0 = NDJSON — null bytes ⇒
    /// empty graph, 1 = binary, 2 = pg-json, 3 = pg-text, 4 = graphson, 5 = csv).
    #[napi(factory)]
    pub fn open(bytes: Option<Buffer>, format: u8) -> Result<Self> {
        let (p, l) = match &bytes {
            Some(b) => (b.as_ptr(), b.len()),
            None => (std::ptr::null(), 0),
        };
        // SAFETY: `p`/`l` describe the JS Buffer; `lnk_open` copies out of it.
        // `threads = 1`: the napi addon decodes serially (parse parallelism is wired
        // through the bun:ffi / wasm backends' `graphFromNdjson` thread count).
        let store = unsafe { ffi::lnk_open(p, l, format, 1) };
        if store.is_null() {
            return Err(unsafe { last_error("open") });
        }
        Ok(Self {
            store: StorePtr(store),
        })
    }

    /// Deep-copy this graph into a fresh, independent handle (`lnk_clone`).
    #[napi]
    pub fn clone_graph(&self) -> Result<Graph> {
        // SAFETY: `store` is a live handle from `open`/`clone`.
        let c = unsafe { ffi::lnk_clone(self.store.0) };
        if c.is_null() {
            return Err(unsafe { last_error("clone") });
        }
        Ok(Self { store: StorePtr(c) })
    }

    /// A store statistic by id: 0 = vertex count, 1 = edge count, 2 = version.
    #[napi]
    pub fn stat(&self, which: u32) -> f64 {
        // SAFETY: infallible read over a live handle.
        unsafe { ffi::lnk_stat(self.store.0, which) as f64 }
    }

    /// Apply one graph setting by its stable id; returns the `lnk_config` status
    /// (1 = applied, 0 = unrecognized id / rejected value).
    #[napi]
    pub fn config(&mut self, id: u32, value: f64) -> u32 {
        // SAFETY: mutable op over a live handle.
        unsafe { ffi::lnk_config(self.store.0, id, value as u64) }
    }

    /// Run a query. `lang` 0 = GQL, 1 = Gremlin; `format` 0 = JSON, 1 = Arrow,
    /// 2 = Arrow-IPC file, 3 = Arrow-IPC stream. `&mut` because a query may mutate.
    #[napi]
    pub fn query(
        &mut self,
        lang: u8,
        query: String,
        params: Option<String>,
        format: u8,
    ) -> Result<Buffer> {
        let (pp, pl) = match &params {
            Some(s) => (s.as_ptr(), s.len()),
            None => (std::ptr::null(), 0),
        };
        // SAFETY: pointers describe the query/params strings; result is crate-owned.
        let bytes = unsafe {
            let mut len = 0usize;
            let p = ffi::lnk_query(
                self.store.0,
                lang,
                query.as_ptr(),
                query.len(),
                pp,
                pl,
                format,
                &mut len,
            );
            take("query", p, len)?
        };
        Ok(bytes.into())
    }

    /// A named "exotic-tier" command (`algo`, `merge`, `epoch`, `last_write_scope`,
    /// `arrow_ipc`, `prepare`, `prepared_run`, `prepared_free`) with an optional
    /// byte payload (`lnk_command`). Returns the crate-owned result bytes.
    #[napi]
    pub fn command(&mut self, name: String, input: Option<Buffer>) -> Result<Buffer> {
        let (ip, il) = match &input {
            Some(b) => (b.as_ptr(), b.len()),
            None => (std::ptr::null(), 0),
        };
        // SAFETY: pointers describe the name/input; result is crate-owned.
        let bytes = unsafe {
            let mut len = 0usize;
            let p = ffi::lnk_command(self.store.0, name.as_ptr(), name.len(), ip, il, &mut len);
            take("command", p, len)?
        };
        Ok(bytes.into())
    }

    /// Non-blocking [`Graph::command`]: runs the command on a libuv threadpool
    /// thread and resolves a `Promise` with the result bytes, keeping the JS event
    /// loop free. Used for `algoAsync`. The facade's single-flight guard forbids any
    /// other native call on this graph while the promise is pending.
    #[napi(ts_return_type = "Promise<Buffer>")]
    pub fn command_async(&mut self, name: String, input: Option<Buffer>) -> AsyncTask<CommandTask> {
        AsyncTask::new(CommandTask {
            store: self.store,
            name,
            input: input.map(|b| b.to_vec()),
        })
    }

    /// Apply a schema-DDL op (`{op:…}` JSON: createIndex / unique / required / …)
    /// via `lnk_schema_apply`. Throws on a non-zero status.
    #[napi]
    pub fn schema_apply(&mut self, json: String) -> Result<()> {
        // SAFETY: pointer describes the JSON string.
        let rc = unsafe { ffi::lnk_schema_apply(self.store.0, json.as_ptr(), json.len()) };
        if rc != 0 {
            return Err(unsafe { last_error("schemaApply") });
        }
        Ok(())
    }

    /// The full active schema as NDJSON `{op:…}` lines (`lnk_schema_dump`).
    #[napi]
    pub fn schema_dump(&self) -> Result<Buffer> {
        // SAFETY: read over a live handle; result is crate-owned.
        let bytes = unsafe {
            let mut len = 0usize;
            let p = ffi::lnk_schema_dump(self.store.0, &mut len);
            take("schemaDump", p, len)?
        };
        Ok(bytes.into())
    }

    /// Serialize the store in the given `format` byte (`lnk_encode`; see [`open`]).
    #[napi]
    pub fn encode(&self, format: u8) -> Result<Buffer> {
        // SAFETY: read over a live handle; result is crate-owned.
        let bytes = unsafe {
            let mut len = 0usize;
            let p = ffi::lnk_encode(self.store.0, format, &mut len);
            take("encode", p, len)?
        };
        Ok(bytes.into())
    }

    /// A transaction action: 0 = begin, 1 = commit, 2 = rollback (`lnk_tx`).
    /// Throws on a non-zero status.
    #[napi]
    pub fn tx(&mut self, action: u8) -> Result<()> {
        // SAFETY: mutable op over a live handle.
        let rc = unsafe { ffi::lnk_tx(self.store.0, action) };
        if rc != 0 {
            return Err(unsafe { last_error("tx") });
        }
        Ok(())
    }
}

/// The engine's ABI version — the value the C ABI reports via `lnk_abi_version`,
/// exposed so the JS adapter can satisfy the shared `Backend` contract's
/// `abiVersion` field.
#[napi]
pub fn abi_version() -> u32 {
    ffi::lnk_abi_version()
}
