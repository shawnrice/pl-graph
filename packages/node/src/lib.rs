//! `@lenke/node` — a native Node.js addon for the `lenke-core` graph engine,
//! built with napi-rs (N-API).
//!
//! This is the *fast* Node path. Where the bun:ffi / wasm backends cross a C ABI
//! and marshal pointers by hand, this addon speaks N-API directly: JS strings and
//! Buffers arrive as real Rust values, results go back as Buffers with no
//! serialize-then-reparse dance on the boundary. The engine is compiled straight
//! in (path dep on `lenke-core`), so there is no dynamic library to locate at
//! runtime either.
//!
//! The surface mirrors the C ABI's logical operations (see `lenke-core/src/ffi.rs`)
//! so a thin adapter can expose it as the shared `Backend` contract from
//! `@lenke/native` (see `backend.mjs`), which in turn lights up the whole
//! `RustGraph` / `createStore` / `liveQuery` facade on Node unchanged.

use lenke_core::error::CodeError;
use lenke_core::error_codes::ErrorCode;
use lenke_core::gql::eval::Params;
use lenke_core::graph::Graph as CoreGraph;
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Turn a coded engine error into a JS exception: a human `lenke: <op>: <message>`
/// with the stable wire code in a machine-parseable tail (`[E_SYNTAX]`). The TS
/// adapter (`backend.mjs`) reads the tail and rebuilds a `LenkeError` carrying the
/// code, giving the napi addon full error-code parity with the bun:ffi and wasm
/// backends — while a direct `Graph` user still reads a plain message.
fn coded(op: &str, e: CodeError) -> Error {
    Error::new(
        Status::GenericFailure,
        format!("lenke: {op}: {} [{}]", e.message, e.code.as_str()),
    )
}

/// Coded error for a fixed message + code (parse failures, UTF-8 faults) — the
/// same channel as [`coded`], so these carry a stable code too.
fn coded_msg(op: &str, code: ErrorCode, message: impl Into<String>) -> Error {
    coded(op, CodeError::new(code, message))
}

/// Decode the optional params-JSON argument (a flat `{"name": value}` object of
/// `$name` bindings) via the crate's strict hand-rolled decoder. Values bind to
/// already-parsed param slots at execute time — they never touch the GQL
/// parser, which is the injection-safety contract of the params surface.
fn decode_params(op: &str, params_json: Option<&str>) -> Result<Params> {
    match params_json {
        None => Ok(Params::new()),
        Some(text) => lenke_core::gql::params_from_json(text).map_err(|e| coded(op, e)),
    }
}

/// A decoded, in-memory columnar graph. Owns its `lenke-core` graph; napi frees
/// it when the JS object is garbage-collected, so there is no explicit free.
#[napi]
pub struct Graph {
    inner: CoreGraph,
}

/// A [`Graph::algo_async`] unit of work: computes the algorithm on a libuv
/// threadpool thread reading `&Graph` through `graph_ptr` (a `usize` so the task is
/// `Send`), then — back on the main thread in `resolve` — applies any `writeProperty`
/// writes and hands back the `{columns, rows}` JSON bytes.
///
/// The pointer is valid for the task's lifetime because the JS `Graph` object stays
/// referenced by the awaiting frame, and the `@lenke/native` facade refuses any other
/// native call on the graph while the promise is pending, so the off-thread read
/// never races a mutation.
pub struct AlgoTask {
    graph_ptr: usize,
    name: String,
    config: Option<String>,
}

impl Task for AlgoTask {
    // (rows JSON bytes, pending writeProperty writes to apply on the main thread)
    type Output = (Vec<u8>, Option<(String, Vec<(u32, lenke_core::graph::Value)>)>);
    type JsValue = Buffer;

    fn compute(&mut self) -> Result<Self::Output> {
        // SAFETY: `graph_ptr` points at a live `CoreGraph` (the JS object is pinned by
        // the pending promise) and the facade guarantees no concurrent mutation, so
        // this shared read is sound. Only `&Graph` is taken here.
        let graph = unsafe { &*(self.graph_ptr as *const CoreGraph) };
        let (rows, writes) =
            lenke_core::algo::compute_parts(graph, &self.name, self.config.as_deref().unwrap_or(""))
                .map_err(|msg| coded_msg("algoAsync", ErrorCode::Ffi, msg))?;
        Ok((rows.to_json().into_bytes(), writes))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Buffer> {
        let (bytes, writes) = output;
        if let Some((prop, results)) = writes {
            // SAFETY: `resolve` runs on the main thread with no other JS executing, so
            // this exclusive `&mut` cannot alias the (now-finished) off-thread read.
            let graph = unsafe { &mut *(self.graph_ptr as *mut CoreGraph) };
            for (v, val) in results {
                graph.set_vertex_prop(v, &prop, val);
            }
        }
        Ok(bytes.into())
    }
}

#[napi]
impl Graph {
    /// Decode NDJSON bytes into a graph. `parallel` (default true) uses the
    /// rayon bulk decoder; pass false for a serial parse.
    #[napi(factory)]
    pub fn from_ndjson(bytes: Buffer, parallel: Option<bool>) -> Result<Self> {
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            coded_msg(
                "fromNdjson",
                ErrorCode::Ffi,
                "NDJSON bytes are not valid UTF-8",
            )
        })?;
        let inner = if parallel.unwrap_or(true) {
            lenke_core::ndjson::decode(text)
        } else {
            lenke_core::ndjson::decode_serial(text)
        }
        .map_err(|e| coded("fromNdjson", e))?;

        Ok(Self { inner })
    }

    /// Deep-copy this graph into a fresh, fully independent handle — the fast
    /// fork/branch substrate. An O(V+E) clone of the columnar store (`Graph: Clone`
    /// in lenke-core), not a serialize→parse round-trip: element ids are preserved
    /// exactly. `#[napi(factory)]` so it returns a new JS `Graph` wrapper.
    #[napi(factory)]
    pub fn clone_graph(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Deserialize bytes in a named format (`pg-json | pg-text | graphson | csv |
    /// ndjson`) into a new graph.
    #[napi(factory)]
    pub fn deserialize(bytes: Buffer, format: String) -> Result<Self> {
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            coded_msg(
                "deserialize",
                ErrorCode::Ffi,
                "input bytes are not valid UTF-8",
            )
        })?;
        let inner =
            lenke_core::codec::deserialize(text, &format).map_err(|e| coded("deserialize", e))?;

        Ok(Self { inner })
    }

    #[napi(getter)]
    pub fn vertex_count(&self) -> f64 {
        self.inner.vertex_count() as f64
    }

    #[napi(getter)]
    pub fn edge_count(&self) -> f64 {
        self.inner.edge_count() as f64
    }

    /// Monotonic mutation counter — the O(1) "did anything change?" signal for
    /// reactive snapshots.
    #[napi]
    pub fn version(&self) -> f64 {
        self.inner.version() as f64
    }

    /// Per-token change epoch (label / edge-type / property-key) for finer
    /// invalidation than the global version.
    #[napi]
    pub fn epoch(&self, name: String) -> f64 {
        self.inner.epoch(&name) as f64
    }

    /// Declare an opt-in secondary index (backfills existing elements, then stays
    /// current; idempotent) — the single index-creation entry point. `on` is
    /// `"vertex"` or `"edge"`; `kind` is `"hash"` (equality seek, one key — turns
    /// `WHERE x.k = …` into a seek) or `"interval"` (RI-tree over an edge
    /// `[keys[0], keys[1])` temporal pair for as-of / overlap seeks; two cover a
    /// bitemporal as-of). Unsupported combinations are a no-op.
    #[napi]
    pub fn create_index(&mut self, on: String, kind: String, keys: Vec<String>) {
        match (on.as_str(), kind.as_str()) {
            ("edge", "interval") => {
                if keys.len() >= 2 {
                    self.inner.create_edge_interval_index(&keys[0], &keys[1]);
                }
            }
            ("edge", "hash") => {
                if let Some(k) = keys.first() {
                    self.inner.create_edge_index(k);
                }
            }
            ("vertex", "hash") => {
                if let Some(k) = keys.first() {
                    self.inner.create_vertex_index(k);
                }
            }
            _ => {}
        }
    }

    /// Drop a vertex / edge property index (no-op if absent). Rejected if the key
    /// backs a unique constraint — drop the constraint first.
    #[napi]
    pub fn drop_vertex_index(&mut self, key: String) -> Result<()> {
        self.inner
            .drop_vertex_index(&key)
            .map_err(|e| coded("dropVertexIndex", e))
    }

    #[napi]
    pub fn drop_edge_index(&mut self, key: String) -> Result<()> {
        self.inner
            .drop_edge_index(&key)
            .map_err(|e| coded("dropEdgeIndex", e))
    }

    /// The currently-indexed vertex / edge property keys (sorted).
    #[napi]
    pub fn vertex_indexes(&self) -> Vec<String> {
        self.inner.vertex_indexes()
    }

    /// The distinct values of property `key` across the vertices the most recent
    /// committed write touched — that write's content-derived CDC value-scope.
    #[napi]
    pub fn last_write_scope(&self, key: String) -> Vec<String> {
        self.inner.last_write_scope(&key)
    }

    #[napi]
    pub fn edge_indexes(&self) -> Vec<String> {
        self.inner.edge_indexes()
    }

    /// The full active schema as a JSON array of replayable `SchemaOp` objects
    /// (constraints / validators / invariants / indexes) — the read side of the
    /// `create_*` declarations. Returned as a JSON string the `Backend` adapter
    /// parses, mirroring the C ABI's `lnk_dump_schema`. Used to persist schema in
    /// a snapshot so a cold boot restores what data alone can't.
    #[napi]
    pub fn dump_schema(&self) -> String {
        self.inner.dump_schema()
    }

    /// Apply one graph setting by its stable id (see the core `ConfigId`).
    /// Returns false for an unrecognized id or a zero value, so the caller can
    /// report an unsupported setting rather than silently running with the
    /// default. ONE id-keyed method rather than one per knob — the operator-chain
    /// ceiling is now `ConfigId::LimitsOperatorChain`.
    #[napi]
    pub fn set_config(&mut self, id: u32, value: f64) -> bool {
        let Some(id) = lenke_core::graph::ConfigId::from_u32(id) else {
            return false;
        };
        if value < 1.0 {
            return false;
        }
        self.inner.set_config(id, value as u64)
    }

    /// Run a GQL query; returns the `{columns, rows}` JSON document as bytes.
    /// `params_json` optionally carries a flat JSON object of `$name` bindings.
    /// `&mut` because a query may mutate (`INSERT`/`SET`/`REMOVE`/`DELETE`).
    #[napi]
    pub fn query(&mut self, text: String, params_json: Option<String>) -> Result<Buffer> {
        let params = decode_params("query", params_json.as_deref())?;
        let parsed = lenke_core::gql::parse_with_max_chain(&text, self.inner.max_operator_chain())
            .map_err(|e| {
            coded_msg(
                "query",
                ErrorCode::Syntax,
                format!("{} (pos {})", e.message, e.pos),
            )
        })?;
        let rows = parsed
            .execute(&mut self.inner, &params)
            .map_err(|e| coded("query", e))?;

        Ok(rows.to_json().into_bytes().into())
    }

    /// Run a GQL query; returns the Apache Arrow ("ARW1") columnar blob.
    /// Takes the same optional `params_json` bindings as [`Graph::query`].
    #[napi]
    pub fn query_arrow(&mut self, text: String, params_json: Option<String>) -> Result<Buffer> {
        let params = decode_params("queryArrow", params_json.as_deref())?;
        let parsed = lenke_core::gql::parse_with_max_chain(&text, self.inner.max_operator_chain())
            .map_err(|e| {
            coded_msg(
                "queryArrow",
                ErrorCode::Syntax,
                format!("{} (pos {})", e.message, e.pos),
            )
        })?;
        let blob = parsed
            .execute_arrow(&mut self.inner, &params)
            .map_err(|e| coded("queryArrow", e))?;

        Ok(blob.into())
    }

    /// Run a GQL query and return standard Apache Arrow IPC bytes (`file` → the
    /// file / Feather-v2 layout, else the IPC stream layout) — the whole query→IPC
    /// path runs natively, no JS re-encode. Same params as [`Graph::query_arrow`].
    #[napi]
    pub fn query_arrow_ipc(
        &mut self,
        text: String,
        params_json: Option<String>,
        file: bool,
    ) -> Result<Buffer> {
        let params = decode_params("queryArrowIpc", params_json.as_deref())?;
        let parsed = lenke_core::gql::parse_with_max_chain(&text, self.inner.max_operator_chain())
            .map_err(|e| {
            coded_msg(
                "queryArrowIpc",
                ErrorCode::Syntax,
                format!("{} (pos {})", e.message, e.pos),
            )
        })?;
        let blob = parsed
            .execute_arrow(&mut self.inner, &params)
            .map_err(|e| coded("queryArrowIpc", e))?;

        Ok(lenke_core::arrow::arrow_ipc_from_blob(&blob, file).into())
    }

    /// Run a native graph algorithm (`degree`, `pagerank`, `connectedComponents`,
    /// `labelPropagation`, `shortestPath`) over the whole graph in one call; returns
    /// the `{columns, rows}` JSON document as bytes. `config` is the algorithm's JSON
    /// config object (`None`/`{}` = defaults). `&mut` because a `writeProperty` config
    /// mutates the graph.
    #[napi]
    pub fn algo(&mut self, name: String, config: Option<String>) -> Result<Buffer> {
        let rows = lenke_core::algo::run(&mut self.inner, &name, config.as_deref().unwrap_or(""))
            .map_err(|msg| coded_msg("algo", ErrorCode::Ffi, msg))?;
        Ok(rows.to_json().into_bytes().into())
    }

    /// Non-blocking [`Graph::algo`]: runs the whole algorithm on a libuv threadpool
    /// thread (keeping the engine's internal parallelism) and resolves a `Promise`
    /// with the `{columns, rows}` bytes, so the JS event loop stays free. The compute
    /// is read-only (`&Graph`); a `writeProperty` config's writes are applied back on
    /// the main thread in `resolve`.
    ///
    /// Safety contract (enforced by the `@lenke/native` facade's single-flight guard):
    /// the graph must not be touched by another native call while the returned promise
    /// is pending — the task holds a pointer into this graph and reads it off-thread,
    /// so a concurrent mutation would be a data race.
    #[napi(ts_return_type = "Promise<Buffer>")]
    pub fn algo_async(&mut self, name: String, config: Option<String>) -> AsyncTask<AlgoTask> {
        let graph_ptr = &mut self.inner as *mut CoreGraph as usize;
        AsyncTask::new(AlgoTask { graph_ptr, name, config })
    }

    /// Run a textual Gremlin query; returns the JSON-array result as bytes.
    #[napi]
    pub fn gremlin(&mut self, text: String) -> Result<Buffer> {
        let plan = lenke_core::gremlin::parse(&text)
            .map_err(|e| coded_msg("gremlin", ErrorCode::Syntax, e))?;
        let vals = lenke_core::gremlin::try_run(&mut self.inner, &plan)
            .map_err(|e| coded("gremlin", e))?;

        Ok(
            lenke_core::gremlin::exec::results_to_json(&self.inner, &vals)
                .into_bytes()
                .into(),
        )
    }

    /// Serialize the graph in a named format (`pg-json | pg-text | graphson |
    /// csv | ndjson`).
    #[napi]
    pub fn serialize(&self, format: String) -> Result<Buffer> {
        let out = lenke_core::codec::serialize(&self.inner, &format)
            .map_err(|e| coded("serialize", e))?;

        Ok(out.into_bytes().into())
    }

    /// Serialize the whole graph back to NDJSON bytes.
    #[napi]
    pub fn encode_ndjson(&self) -> Buffer {
        lenke_core::ndjson::encode(&self.inner).into_bytes().into()
    }

    /// Bulk-append NDJSON `bytes` into this graph — a `COPY FROM` for a live
    /// store, at bulk speed (no per-`INSERT` parse). Returns a JSON `MergeReport`
    /// (what applied vs. skipped). See `ndjson::append`.
    #[napi]
    pub fn merge_ndjson(&mut self, bytes: Buffer) -> Result<Buffer> {
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            coded_msg("mergeNdjson", ErrorCode::Ffi, "NDJSON bytes are not valid UTF-8")
        })?;
        let report =
            lenke_core::ndjson::append(&mut self.inner, text).map_err(|e| coded("mergeNdjson", e))?;

        Ok(report.to_json().into_bytes().into())
    }
}

/// A compiled, reusable GQL query — lex/parse/lower done once, then executed
/// against a graph with fresh params, skipping the per-call re-parse the
/// `Graph::query` path pays. Graph-independent: one prepare can run against any
/// graph. (`prepare` is a free function because a `Prepared` doesn't need a
/// graph to exist.)
#[napi]
pub struct PreparedQuery {
    inner: lenke_core::gql::Prepared,
}

#[napi]
impl PreparedQuery {
    /// Execute against `graph` with `params_json`; returns the `{columns, rows}`
    /// JSON document as bytes.
    #[napi]
    pub fn query(&self, graph: &mut Graph, params_json: Option<String>) -> Result<Buffer> {
        let params = decode_params("preparedQuery", params_json.as_deref())?;
        let rows = self
            .inner
            .execute(&mut graph.inner, &params)
            .map_err(|e| coded("preparedQuery", e))?;

        Ok(rows.to_json().into_bytes().into())
    }

    /// Execute against `graph` → the Apache Arrow ("ARW1") columnar blob.
    #[napi]
    pub fn query_arrow(&self, graph: &mut Graph, params_json: Option<String>) -> Result<Buffer> {
        let params = decode_params("preparedQueryArrow", params_json.as_deref())?;
        let blob = self
            .inner
            .execute_arrow(&mut graph.inner, &params)
            .map_err(|e| coded("preparedQueryArrow", e))?;

        Ok(blob.into())
    }
}

/// Compile a GQL query string into a reusable [`PreparedQuery`]. `max_operator_chain`
/// is the anti-resource-abuse operator-chain ceiling (default 10_000 when omitted).
#[napi]
pub fn prepare(text: String, max_operator_chain: Option<f64>) -> Result<PreparedQuery> {
    let max = max_operator_chain.map_or(10_000, |n| n as usize);
    let inner = lenke_core::gql::prepare_with_max_chain(&text, max).map_err(|e| {
        coded_msg(
            "prepare",
            ErrorCode::Syntax,
            format!("{} (pos {})", e.message, e.pos),
        )
    })?;

    Ok(PreparedQuery { inner })
}

/// The engine's ABI version — the same value the C ABI reports via
/// `lnk_abi_version`, exposed here so the `Backend` adapter can satisfy the
/// shared contract's `abiVersion` field.
#[napi]
pub fn abi_version() -> u32 {
    lenke_core::ffi::lnk_abi_version()
}
