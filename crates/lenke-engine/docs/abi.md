# The engine C-ABI (lean draft)

The goal: a drop-in backend that `packages/native` can load in place of
`lenke-core`, with a **deliberately small, flat exported surface**. Core grew one
C entry point per operation and reached 44; the engine holds the line at **16
symbols** and — crucially — that count does **not grow as features are added**.

Two guiding principles, both from how core's surface sprawled:

1. **Pare down external symbols in general.** Every exported `#[no_mangle]` is
   ABI surface we must keep stable forever, version in lockstep, and re-audit on
   every wasm/ffi change. Fewer symbols = a smaller contract to defend. Where a
   family of calls differs only by a variant (out/in, vertex/edge, gql/gremlin,
   json/arrow), **parameterize with an enum argument** instead of minting a
   symbol per variant.

2. **We want the exotic tiers — but the ABI stays simple.** Prepared statements,
   Arrow egress, CDC scope, binary snapshots, fork/merge, and whatever comes
   next are all real features we intend to ship. They do **not** each get their
   own symbols. They ride a single generic dispatcher, `lnk_command(name, in) →
out`. Adding a feature fills in a match arm, not a new export — so the ABI is
   the same 16 symbols whether the engine ships 5 exotic features or 50.

The handle is the engine's `Store`. The flat `lnk_*` surface described here is what
the engine ships (`src/ffi.rs`).

## The surface at a glance: 44 → 16 (flat)

| Group        | Symbols                                                                              | How the fold works                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Plumbing     | `lnk_abi_version`, `lnk_alloc`, `lnk_dealloc`, `lnk_free`, `lnk_last_error_json` (5) | one `lnk_free` for every engine-returned buffer (core had `free_buf` + `free_arrow`)                                                                       |
| Lifecycle    | `lnk_open`, `lnk_close`, `lnk_clone`, `lnk_config`, `lnk_stat` (5)                   | `lnk_stat(which)` folds the 4 count/version/epoch getters; `lnk_open(…, format)` folds ndjson + binary + pg-json/pg-text/graphson/csv + empty construction |
| Query        | `lnk_query` (1)                                                                      | `lnk_query(lang, …, format, …)` folds `query_rows` + `gremlin_json` + `query_arrow` + `query_arrow_ipc`                                                    |
| Transaction  | `lnk_tx` (1)                                                                         | `lnk_tx(action)` folds begin/commit/rollback                                                                                                               |
| Schema       | `lnk_schema_apply`, `lnk_schema_dump` (2)                                            | `schema_apply(json)` folds all 10 `create_*` + 2 `drop_*`; `schema_dump` folds the 2 index-introspection reads                                             |
| Snapshot     | `lnk_encode` (1)                                                                     | `lnk_encode(format)` folds ndjson + binary + pg-json/pg-text/graphson/csv out                                                                              |
| Escape hatch | `lnk_command` (1)                                                                    | every exotic tier — no new symbols, ever                                                                                                                   |

**16 symbols, fixed.** Compare to core's 44-and-counting.

## Conventions (unchanged from core)

- **Handle:** `*mut Store` (mutating) / `*const Store` (read-only).
- **Buffer-out:** `(…, out_len: *mut usize) -> *mut u8`. Returns a heap buffer
  (length in `*out_len`) the caller frees with `lnk_free`. Null on error, detail
  on the error channel.
- **Buffer-in:** `(ptr: *const u8, len: usize)` pairs, UTF-8 for strings; a null
  `ptr` means "absent" where the doc says so.
- **Error channel:** every fallible call runs `ffi_error::begin()` on entry and
  `ffi_error::set(code, msg)` on failure; the host reads it via
  `lnk_last_error_json`. Buffer-returning calls signal failure with null,
  status-returning calls with a negative `i32`.
- **Status codes** (`i32`): `0` ok · `-1` argument / parse / UTF-8 error ·
  `-2` operation rejected by current data (e.g. a constraint already violated).
- **wasm memory:** `lnk_alloc`/`lnk_dealloc` place host-owned input bytes in
  linear memory (bun:ffi passes JS buffers directly and skips them; the wasm
  backend needs them) — so they are part of the ABI. `lnk_free` is the separate
  release path for buffers the _engine_ allocates and returns.

## Enum parameters (the "parameterize, don't multiply" levers)

```
lnk_open   format:  0 = NDJSON   1 = binary snapshot   2 = pg-json  3 = pg-text  4 = graphson  5 = csv   (ptr null + format 0 = empty graph)
lnk_stat   which:   0 = vertex_count  1 = edge_count  2 = version   (per-token epoch takes a name → lnk_command "epoch", not a selector)
lnk_query  lang:    0 = GQL      1 = Gremlin
lnk_query  format:  0 = JSON rows  1 = Arrow  2 = Arrow IPC
lnk_tx     action:  0 = begin    1 = commit  2 = rollback
lnk_encode format:  0 = NDJSON   1 = binary snapshot   2 = pg-json  3 = pg-text  4 = graphson  5 = csv

Formats 2..5 (pg-json/pg-text/graphson/csv) route through the shared `lenke-codec`
crate via the `src/codec.rs` Store<->GraphData bridge — so the native and wasm builds
emit byte-identical bytes from identical data.
```

An unknown enum value is a `-1` / null error, never undefined behaviour — a host
on a newer artifact degrades cleanly against an older engine.

## Signatures

```rust
// ---- Plumbing (5) ----
pub extern "C" fn lnk_abi_version() -> u32;
pub extern "C" fn lnk_alloc(len: usize) -> *mut u8;
pub unsafe extern "C" fn lnk_dealloc(ptr: *mut u8, len: usize);
pub unsafe extern "C" fn lnk_free(ptr: *mut u8, len: usize);
pub unsafe extern "C" fn lnk_last_error_json(out_len: *mut usize) -> *mut u8;

// ---- Lifecycle (5) ----
/// format 0 = NDJSON (null ptr = empty graph), 1 = binary, 2..5 = pg-json/pg-text/graphson/csv. null on error.
pub unsafe extern "C" fn lnk_open(ptr: *const u8, len: usize, format: u8) -> *mut Store;
pub unsafe extern "C" fn lnk_close(s: *mut Store);
pub unsafe extern "C" fn lnk_clone(s: *const Store) -> *mut Store;
pub unsafe extern "C" fn lnk_config(s: *mut Store, id: u32, value: u64) -> u32;
pub unsafe extern "C" fn lnk_stat(s: *const Store, which: u32) -> u64;

// ---- Query / exec (1) ----
/// lang 0=GQL 1=Gremlin; format 0=JSON 1=Arrow 2=Arrow-IPC. p_ptr/p_len = JSON params
/// (null = none). Returns the carrier for (lang, format). null on error.
pub unsafe extern "C" fn lnk_query(
    s: *mut Store,
    lang: u8,
    q_ptr: *const u8, q_len: usize,
    p_ptr: *const u8, p_len: usize,
    format: u8,
    out_len: *mut usize,
) -> *mut u8;

// ---- Transaction (1) ----
/// action 0=begin 1=commit 2=rollback. 0 ok, -1 on a bad action / no active tx.
pub unsafe extern "C" fn lnk_tx(s: *mut Store, action: u8) -> i32;

// ---- Schema (2) ----
/// Apply one schema op (JSON, see below). 0 ok · -1 arg/parse · -2 rejected by data.
pub unsafe extern "C" fn lnk_schema_apply(s: *mut Store, json_ptr: *const u8, json_len: usize) -> i32;
/// Full schema as a JSON op-list — the single schema *read* (subsumes vertex/edge index lists).
pub unsafe extern "C" fn lnk_schema_dump(s: *const Store, out_len: *mut usize) -> *mut u8;

// ---- Snapshot (1) ----
/// format 0 = NDJSON, 1 = binary, 2..5 = pg-json/pg-text/graphson/csv. Pairs with lnk_schema_dump for a full snapshot.
pub unsafe extern "C" fn lnk_encode(s: *const Store, format: u8, out_len: *mut usize) -> *mut u8;

// ---- Escape hatch (1) — every exotic tier, no new symbols ----
/// Run a named command with a JSON/bytes input, returning a JSON/bytes buffer.
/// `name` selects the op (see the command registry). null on error / unknown name.
pub unsafe extern "C" fn lnk_command(
    s: *mut Store,
    name_ptr: *const u8, name_len: usize,
    in_ptr: *const u8, in_len: usize,
    out_len: *mut usize,
) -> *mut u8;
```

## `lnk_schema_apply` JSON schema

One tagged object per call; `op` selects the form. `on` is `"vertex"` or
`"edge"`; vertex forms key on `label`, edge forms on `etype`. This is the whole
of core's 12 schema-mutation symbols behind one call.

```jsonc
// indexes  (core: lnk_create_index / lnk_drop_*_index)
{ "op": "createIndex", "on": "vertex", "kind": "hash",     "keys": ["age"] }
{ "op": "createIndex", "on": "vertex", "kind": "range",    "keys": ["score"] }
{ "op": "createIndex", "on": "edge",   "kind": "interval", "keys": ["lo", "hi"] } // 2 keys
{ "op": "dropIndex",   "on": "vertex", "key": "age" }
{ "op": "dropIndex",   "on": "edge",   "key": "weight" }

// constraints  (core: the 8 lnk_create_*_constraint)
{ "op": "unique",   "on": "vertex", "label": "Person", "key": "email" }
{ "op": "unique",   "on": "edge",   "etype": "KNOWS",  "key": "id" }
{ "op": "required", "on": "vertex", "label": "Person", "key": "name" }
{ "op": "required", "on": "edge",   "etype": "KNOWS",  "key": "since" }
{ "op": "type",     "on": "vertex", "label": "Person", "key": "age",   "type": "number" }
{ "op": "type",     "on": "edge",   "etype": "KNOWS",  "key": "since", "type": "number" }

// cardinality: edge degree per (label, edgeType, direction) bounded to min..=max.
// direction "out"|"in"; "max" is a number or null (unbounded).
{ "op": "cardinality", "label": "Person", "edgeType": "OWNS", "direction": "out", "min": 0, "max": 1 }

// validator: SQL-CHECK — the `var` element carrying `label` (a vertex label OR an
// edge type) must satisfy `predicate` (only a definite-false fails; null passes).
{ "op": "validator", "label": "Person", "var": "p", "predicate": "p.age >= 0" }
// invariant: a named GQL query that holds unless its result has a boolean-false cell.
{ "op": "invariant", "name": "ages_nonneg", "query": "MATCH (p:Person) RETURN p.age >= 0" }
```

## `lnk_command` registry (the exotic tiers)

Each entry is a `name` + an input/output JSON (or bytes) shape. Adding a tier =
adding a match arm here, **not** a symbol. Handles for stateful features (a
prepared statement) are returned as an integer id into a `Store`-side slab and
passed back in later commands — so even handle-based features add no pointer-typed
exports.

| `name`             | input                       | output                                | status                     |
| ------------------ | --------------------------- | ------------------------------------- | -------------------------- |
| `last_write_scope` | scope-key name (raw str)    | `{scopes:[…], open:b}`                | **wired** (CDC)            |
| `epoch`            | token name (raw str)        | `{epoch:N}`                           | **wired** (per-token)      |
| `merge`            | NDJSON text (raw)           | `MergeReport` (added/skipped/phantom) | **wired** (first-wins)     |
| `prepare`          | query text (raw str)        | `{handle:"<ptr>"}`                    | **wired** (prepared stmts) |
| `prepared_run`     | `{handle, params, format?}` | `{columns, rows}` / Arrow             | **wired** (JSON + Arrow)   |
| `prepared_free`    | `{handle}`                  | `{}`                                  | **wired** (prepared stmts) |
| `algo`             | `{name, config}`            | `{columns, rows}`                     | **wired** (also via CALL)  |

`prepared_run` takes an optional `format` (`json` default, `arrow`, `arrow_ipc`),
so a prepared statement has the same output surface as `lnk_query`. `algo` runs a
native algorithm directly (honoring a `writeProperty` in the config) and is _also_
reachable through `lnk_query(lang=GQL, "CALL pagerank(...) YIELD ...")`, the
conformant home for the graph algorithms.

## Host-side impact

`backend-ffi.ts` shrinks its `SYMBOLS` table 44 → 16 and folds variant families
into enum args: `queryRows`/`gremlin`/`queryArrow` become `lnk_query` with a
`lang`/`format`; `createIndex`/`createUniqueConstraint`/… become
`lnk_schema_apply` payload builders; `vertexIndexes()`/`edgeIndexes()` read from
the parsed `lnk_schema_dump`; the exotic methods call `lnk_command`. This is a
transport reshape — byte-identity and the conformance suites are unaffected.

## Packaging notes

- `Cargo.toml`: `crate-type = ["lib", "cdylib"]`, so the engine emits its own
  `liblenke_engine.so` / `lenke_engine.wasm`.
- **The C ABI is behind a `capi` feature (off by default).** Gating
  `ffi`/`ffi_error` on `capi` keeps the engine's `#[no_mangle]` symbols
  (`lnk_abi_version`, `lnk_alloc`, `lnk_query`, …) out of the plain lib/test build;
  the standalone backend builds with
  `cargo build -p lenke-engine --release --features capi`.
- Symbols export as plain `lnk_*`.
- `lnk_abi_version()` returns the ABI the host asserts; the host loads the engine's
  artifact by these names.
- Arrow buffers must use the _same_ allocator as every other returned buffer, so
  one `lnk_free` releases them all.

## wasm build

`bun run engine:build:wasm` (native cdylib: `engine:build:rust`) — both in
`packages/native/package.json`, mirroring core's `build:wasm`/`build:rust` with
`--features capi`. The engine builds for `wasm32-unknown-unknown` with **no
external imports** — the module loads with `WebAssembly.instantiate(mod, {})`,
exports its `memory` and all 16 `lnk_*`, and `lnk_abi_version()` returns 18.
Verified end-to-end over wasm: NDJSON load → GQL query → result read, and a full
prepared-statement lifecycle including a use-after-free surfacing as a clean
`E_FFI` error.

What it took (the only wasm-specific code):

- **`on_big_stack` is cfg-gated.** On native a deep-recursion traversal runs on a
  spawned 1 GiB-stack thread; wasm has no threads, so it runs inline
  (`#[cfg(target_arch = "wasm32")]`). If an unbounded quantifier overflows the
  module stack, raise it at link time (`-C link-arg=-zstack-size=…`).
- Nothing else needed gating: the engine's only runtime dep is `regex` (pure
  Rust); the cost estimator's `/proc/meminfo` read already falls back gracefully
  (`.ok()…map_or(4 GiB)` — returns `Err` on wasm, no panic); `Instant` is
  `#[cfg(test)]`-only.

Known wasm limitation (shared with core): `wasm32-unknown-unknown` is
`panic = "abort"`, so the `catch_unwind` query backstop cannot recover a faulting
query — a genuine engine panic aborts the module instance rather than failing the
one call. Parse/exec errors still return cleanly (they are `Result`, not panics).

## Current state of the scaffold (`src/ffi.rs`)

The skeleton exports all 16 symbols today. Wired to existing `Store` methods:
`abi_version`, `alloc`/`dealloc`/`free`, `last_error_json`, `open` (NDJSON +
empty + **binary**), `close`, `clone` (deep copy), `config`, `stat` (counts **and
version**), `query` (GQL + Gremlin JSON; GQL params; **Arrow** raw + IPC, formats
1/2), `tx`, `encode` (NDJSON + **binary**), `schema_apply` + `schema_dump`, and
`lnk_command` for `last_write_scope` (CDC), `epoch` (per-token), and `merge`
(first-wins bulk fill, matching core). The binary snapshot ([`src/binary.rs`](../src/binary.rs)) is the
engine's own versioned format (`LNKB` magic + `u16` version header, so a future
bump is recognized not mis-decoded) for compact/fast browser-local persistence;
it funnels decode through the shared `build_store`, so fidelity matches NDJSON.
Prepared statements (`prepare`/`prepared_run`/`prepared_free`) are wired too — see
the pass below. **Every feature core exposes is now reachable** through the 16
symbols; the only intentional non-command is `algo` (reachable via GQL `CALL`).

### Prepared statements pass (Design A: parse once, bind + run many)

`prepare` parses in prepared mode ([`gql::parse_prepared`], which emits each
`$name` as an unbound [`Expr::Param`] instead of substituting it) and returns the
parsed plan's pointer as a decimal-string `handle` (a 64-bit pointer does not fit
a JSON `f64`). `prepared_run` clones that cached plan, binds the run's params via
[`bind::bind_params`] (an exhaustive `Param`→`Lit` walk over `Plan`/`Expr`), then
optimizes + executes — so the parse cost is paid once across a loop. `prepared_free`
drops the plan. The caller owns handle lifetime. A `Param` that survives binding is
a loud unbound-parameter error in `eval` (the safety net). Binding-then-optimizing
produces a plan byte-identical to a direct parameterized query (asserted via `Debug`).
Not supported in prepared mode: params in `LIMIT`/`SKIP` and literal-only positions
(INSERT / procedure config), which error at parse.

### Query params pass (GQL)

`lnk_query` accepts a JSON params object `{"name": value, …}` (`p_ptr`/`p_len`);
`ndjson::parse_params` decodes it with the stored-value rules (scalars, lists,
records, temporals). `gql::parse_with_params` substitutes each `$name` to its
typed `Value` **at parse time** — in `primary()` (expression positions),
`literal_value()` (the inline-prop seed fast-path `{k: $p}`), and `usize_lit()`
(`LIMIT`/`SKIP $n`). Because substitution happens before `opt`/`exec`, the value
is typed (never spliced into query text — the safety win) and the planner sees a
literal (so `WHERE k = $p` / `{k: $p}` still seed an index — the performance win).
For a scalar/inline-prop param the produced plan is **byte-identical** to inlining
the literal (asserted via derived `Debug`); behavioural equality across WHERE / IN
/ LIMIT is covered too. 4 unit tests, and the full cross-engine corpus +
conformance suites still pass unchanged. **Gremlin** params are a distinct
mechanism (bytecode bindings) and remain unsupported — the ffi errors on
non-empty params for `lang = Gremlin`.

### Lifecycle pass (clone + version)

`lnk_clone` deep-copies the `Store` (hand-written `Clone`: the two `RwLock` derived
caches are rebuilt; every data field is cloned, compiler-enforced complete).
`lnk_stat(which=2)` returns a monotonic **version** — a mutation counter bumped by
every data-mutation primitive (`add_node`, `add_edge`, `set_prop`, `remove_prop`,
`remove_edge_prop`, `delete_edge`, `delete_node`) via `Store::touch`. Version is
out-of-band metadata (not in any query result, codec, or ordering), so it does not
affect byte-identity — verified: the full cross-engine corpus + conformance suites
still pass. Per-token **epoch** takes a name, so it is _not_ a `stat` selector; it
rides `lnk_command "epoch"` (wired). Covered by 2 `Store` unit tests.

### Schema pass (`src/exec::apply_schema_op` + `src/schema_op.rs`)

`schema_apply` (the C ABI) routes through `exec::apply_schema_op`, the single
schema entry point: it handles the two ops that need the query evaluator
(`validator`, `invariant`) and delegates the rest to `schema_op::apply`, which
parses one `{"op":…}` object and dispatches to real `Store` methods. `schema_dump`
emits the **same** vocabulary so `dump → apply` round-trips. `SchemaError` carries
the wire code (`BadRequest`→`E_FFI`/-1, `Invalid`→`E_INVALID_VALUE`/-1,
`Syntax`→`E_SYNTAX`/-1, `GraphOp`→`E_INVALID_GRAPH_OP`/-1,
`Rejected`→`E_CONSTRAINT_VIOLATION`/-2).

The **whole vocabulary is implemented**: `createIndex` (vertex hash/range, edge
interval/type) and `dropIndex`; `unique` / `required` / `type` on both vertices
and edges; `cardinality`; `validator`; `invariant`. Write-time enforcement is
centralized in `Store::run_deferred_checks` (pure-store constraints, derived from
the transaction's CDC change set) plus `exec::enforce_expr_constraints`
(validators + invariants); a violation rolls the statement back with
`E_CONSTRAINT_VIOLATION`. Every kind also persists in the binary snapshot (v2) and
the schema dump. Covered by store + exec + schema_op unit tests.

### Testing note

The `capi` feature exports the engine's `lnk_*` `#[no_mangle]` symbols; the plain
build leaves them off. Run `cargo test -p lenke-engine` (no capi) for the
conformance/corpus suites, and `cargo test -p lenke-engine --lib --features capi`
for the ffi/schema unit tests.
