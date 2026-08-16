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

The handle is the engine's `Store` (core's is `Graph`). The existing
`ffi_engine.rs` compare shim (`lnk_e_*` over `*mut Store`) already proves the
core of the pattern; this completes it and drops the prefix.

## The surface at a glance: 44 → 16 (flat)

| Group        | Symbols                                                                              | How the fold works                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Plumbing     | `lnk_abi_version`, `lnk_alloc`, `lnk_dealloc`, `lnk_free`, `lnk_last_error_json` (5) | one `lnk_free` for every engine-returned buffer (core had `free_buf` + `free_arrow`)                                        |
| Lifecycle    | `lnk_open`, `lnk_close`, `lnk_clone`, `lnk_config`, `lnk_stat` (5)                   | `lnk_stat(which)` folds the 4 count/version/epoch getters; `lnk_open(…, format)` folds ndjson + binary + empty construction |
| Query        | `lnk_query` (1)                                                                      | `lnk_query(lang, …, format, …)` folds `query_rows` + `gremlin_json` + `query_arrow` + `query_arrow_ipc`                     |
| Transaction  | `lnk_tx` (1)                                                                         | `lnk_tx(action)` folds begin/commit/rollback                                                                                |
| Schema       | `lnk_schema_apply`, `lnk_schema_dump` (2)                                            | `schema_apply(json)` folds all 10 `create_*` + 2 `drop_*`; `schema_dump` folds the 2 index-introspection reads              |
| Snapshot     | `lnk_encode` (1)                                                                     | `lnk_encode(format)` folds ndjson + binary out                                                                              |
| Escape hatch | `lnk_command` (1)                                                                    | every exotic tier — no new symbols, ever                                                                                    |

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
lnk_open   format:  0 = NDJSON   1 = binary snapshot        (ptr null + format 0 = empty graph)
lnk_stat   which:   0 = vertex_count  1 = edge_count  2 = version  3 = epoch
lnk_query  lang:    0 = GQL      1 = Gremlin
lnk_query  format:  0 = JSON rows  1 = Arrow  2 = Arrow IPC
lnk_tx     action:  0 = begin    1 = commit  2 = rollback
lnk_encode format:  0 = NDJSON   1 = binary snapshot
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
/// format 0 = NDJSON (null ptr = empty graph), 1 = binary snapshot. null on error.
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
/// format 0 = NDJSON, 1 = binary. Pairs with lnk_schema_dump for a full snapshot.
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

// cardinality: edge degree per (label, etype, direction) bounded to min..=max.
// direction "out"|"in"; omit "max" (or null) for unbounded.
{ "op": "cardinality", "label": "Person", "etype": "OWNS", "direction": "out", "min": 0, "max": 1 }

// validator: SQL-CHECK — the bound `var` element with `label` must satisfy `predicate`.
{ "op": "validator", "label": "Person", "var": "p", "predicate": "p.age >= 0" }
// invariant: a named GQL query that must return zero rows to hold.
{ "op": "invariant", "name": "no_orphans", "query": "MATCH (n) WHERE ... RETURN n" }
```

## `lnk_command` registry (the exotic tiers)

Each entry is a `name` + an input/output JSON (or bytes) shape. Adding a tier =
adding a match arm here, **not** a symbol. Handles for stateful features (a
prepared statement) are returned as an integer id into a `Store`-side slab and
passed back in later commands — so even handle-based features add no pointer-typed
exports.

| `name`             | input                      | output            | tier                          |
| ------------------ | -------------------------- | ----------------- | ----------------------------- |
| `algo`             | `{name, config}`           | `{columns, rows}` | also reachable via GQL `CALL` |
| `prepare`          | `{lang, query}`            | `{handle}`        | prepared statements           |
| `prepared_run`     | `{handle, params, format}` | carrier           | prepared statements           |
| `prepared_free`    | `{handle}`                 | `{}`              | prepared statements           |
| `last_write_scope` | `{}`                       | `{scope}`         | CDC                           |
| `merge`            | `{format, bytes}`          | `{}`              | fork / merge                  |

`algo` is listed for completeness but is _also_ reachable through
`lnk_query(lang=GQL, "CALL pagerank(...) YIELD ...")`, which is the conformant
home for the graph algorithms — so most hosts never need the direct command.

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
- **The C ABI is behind a `capi` feature (off by default).** core links this
  crate as a _lib_ for `engine-compare`, and the engine's `#[no_mangle]` symbols
  (`lnk_abi_version`, `lnk_alloc`, `lnk_last_error_json`, …) are byte-identical
  names to core's own exports — linking both into core's cdylib is a duplicate-
  symbol error. Gating `ffi`/`ffi_error` on `capi` keeps them out of the
  compare-lib build; the standalone backend builds with
  `cargo build -p lenke-engine --release --features capi`.
- Symbols export as plain `lnk_*` (not the `lnk_e_*` compare prefix): the engine
  is its own cdylib, so the names don't collide with core the way they do inside
  the shared engine-compare build.
- `lnk_abi_version()` returns the ABI the host asserts; the host loads core's or
  the engine's artifact by these names.
- The engine is already single-threaded (no rayon) — exactly what the wasm
  target needs, so core's wasm-safety work is free here.
- Arrow buffers must use the _same_ allocator as every other returned buffer, so
  one `lnk_free` releases them all (core needed a separate `free_arrow`).

## Current state of the scaffold (`src/ffi.rs`)

The skeleton exports all 16 symbols today. Wired to existing `Store` methods:
`abi_version`, `alloc`/`dealloc`/`free`, `last_error_json`, `open` (NDJSON +
empty), `close`, `config`, `stat` (counts), `query` (GQL + Gremlin, JSON
format, param-free), `tx`, `encode` (NDJSON), and — as of the schema pass —
`schema_apply` + `schema_dump` (see [`src/schema_op.rs`](../src/schema_op.rs)).
Every remaining capability returns a specific error — that stub list **is** the
work-queue to finish before the flip:

- `lnk_query` params (`p_len > 0`), Arrow / Arrow-IPC formats
- `lnk_open` / `lnk_encode` binary-snapshot format
- `lnk_clone` (needs `Store: Clone`)
- `lnk_stat` version / epoch (2, 3)
- every `lnk_command` name

### Schema pass (`src/schema_op.rs`)

`schema_apply` parses one `{"op":…}` object (reusing the engine's JSON parser via
`ndjson::parse_json`) and dispatches to real `Store` methods; `schema_dump` emits
the **same** vocabulary so `dump → apply` round-trips. `SchemaError` splits
`BadRequest` (→ `-1`) from `Rejected` (→ `-2`). Covered by 10 unit tests.

Implemented ops: `createIndex` on `vertex/hash`, `vertex/range`, `edge/interval`,
`edge/type`; `unique` and `required` on vertices (single or composite `keys`).

Still bad-request (no backing `Store` method yet — the schema work still to do):
`dropIndex`; `type` / `cardinality` / `validator` / `invariant`; edge `unique` /
`required` / `type` constraints; and **index enumeration in `schema_dump`** (only
constraints are introspectable today, so indexes are omitted from the dump).
