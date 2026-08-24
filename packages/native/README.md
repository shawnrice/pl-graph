# @lenke/native

> JavaScript/TypeScript bindings to the Rust `lenke-engine` columnar graph engine, with a single facade over native (FFI) and WebAssembly backends.

Loads a labeled-property graph into the native columnar core and runs GQL or Gremlin queries against it from JS/TS. One C ABI is exposed through interchangeable backends behind a shared `Backend` contract: a native dynamic library loaded over `bun:ffi` (server/CLI, requires Bun), a WebAssembly module instantiated from bytes or a `fetch` response (browser), and — for plain Node — the prebuilt N-API addon in the sibling `@lenke/node` package (`createNodeBackend()`). Reach for this when you want the Rust engine's query performance from JS without reimplementing it. The backend modules are split behind subpath exports so importing the package in a browser never pulls in the Bun-only `bun:ffi` builtin.

## Install

```bash
bun add @lenke/native
```

## Usage

```ts
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';
import { graphFromNdjson } from '@lenke/native';

// Load the native library built from `crates/lenke-engine`
// (liblenke_engine.{dylib,so,dll}).
const backend = createFfiEngineBackend('/path/to/liblenke_engine.dylib');

// Decode an NDJSON document into a graph.
const g = graphFromNdjson(backend, await Bun.file('graph.ndjson').bytes());

console.log(g.vertexCount, g.edgeCount);

// GQL via tagged template (or a plain string) → decoded rows.
const rows = g.query`MATCH (p:Person) RETURN p.name AS name`;
for (const row of rows) {
  console.log(row.name);
}

// Gremlin against the same graph.
const result = g.gremlin`g.V().hasLabel('Person').count()`;

// The graph is heap-owned by the native module; release it when done.
g.free();
```

In the browser, swap the backend for the wasm one; the rest of the API is identical:

```ts
import { createWasmEngineBackend } from '@lenke/native/wasm-engine';
import { graphFromNdjson } from '@lenke/native';

const backend = await createWasmEngineBackend(fetch('/lenke_engine.wasm'));
const g = graphFromNdjson(backend, ndjsonBytes);
```

## Guides, recipes & query patterns

`g.query(...)` speaks **ISO GQL** (ISO/IEC 39075:2024 — not Cypher). For the query
language itself, worked recipes, and the graph-algorithm / graph-ML surface, see:

- **[`docs/guides/`](../../docs/guides/index.md)** — the canonical guides, including
  [`algorithms.md`](../../docs/guides/algorithms.md) (PageRank, components, centrality,
  and **`neighborAggregate`** message passing with edge weights + GCN normalization).
- **The `@lenke/mcp` how-to guides** ([`packages/mcp/src/guides.ts`](../../packages/mcp/src/guides.ts)) —
  a growing cookbook: writing GQL, temporal queries, Arrow egress, and a **Recipes**
  guide (per-hop path predicates over quantified paths, fan-in/structuring, cycles,
  subgraph explanation) plus a **graph-ml** guide. AI coding assistants get these
  automatically through the `@lenke/mcp` server; humans can read them as source or via
  the MCP `how_to` tool.

## Loading a backend

The entry point (`@lenke/native`) is environment-neutral: it exports the `RustGraph` facade, the graph constructors, and the reactive store. The backend itself comes from a subpath:

- `@lenke/native/ffi-engine` — `createFfiEngineBackend(libPath: string): Backend`. Requires **Bun** (uses `bun:ffi`). Pass the absolute path to the built `liblenke_engine.{dylib,so,dll}`.
- `@lenke/native/wasm-engine` — `createWasmEngineBackend(source): Promise<Backend>`. `source` is a `WebAssembly.Module`, `ArrayBuffer`, `ArrayBufferView`, or a (promise of a) `fetch` `Response`.
- **Node** — use the sibling `@lenke/node` package's `createNodeBackend()`, a prebuilt N-API addon. It's the intended production backend under plain Node (no Bun, no wasm overhead) and plugs into this same `Backend` contract.

All assert that the loaded artifact's ABI version matches the exported `ABI_VERSION`, throwing on mismatch. `isBun` is exported as a convenience flag (`true` when running under Bun, where the FFI backend is available).

## Graph API

`graphFromNdjson(backend, ndjson, { parallel? })` (string or bytes) and `graphFromFormat(backend, input, { format })` deserialize a document into a `RustGraph`; `createEmptyGraph(backend)` cold-boots a blank one to `INSERT` / `mergeNdjson` into; `attachGraph(backend, handle)` wraps an existing backend + handle.

Every factory takes the graph's settings, which are fixed for its life — `{ limits: { range, trail, intermediate, operatorChain } }`, each a ceiling whose breach is a loud `E_RESOURCE_EXHAUSTED` rather than a truncated result (`maxOperatorChain` is the shorthand for `limits.operatorChain`). A `RustGraph` exposes:

- `vertexCount` / `edgeCount` — counts (numbers).
- `version` — monotonic mutation counter for O(1) change detection.
- `epoch(name)` — per-token change epoch (by label / edge-type / property-key).
- `query(q, ...subs)` — run GQL (tagged template or string) → `Row[]`, where `Row` is `Record<string, unknown>`.
- `queryArrow(q, ...subs)` — run GQL → raw `ARW1` columnar blob as `Uint8Array` (a compact in-process framing; decode it with the exported `decodeArrow<R>(blob)`, no dependency). **Scalar columns only:** ARW1 carries float64/bool/utf8, so a list column (`collect_list`) or an element column (`RETURN n`) is flattened to a text cell and won't reconstruct as a structured array/object. The flattened text is **lossy and not JSON** — a list renders bare and comma-joined (`[1,2,3]` → the string `"[1,2,3]"`, `["a","b"]` → `"[a,b]"`), so it will **not** `JSON.parse` back. Use the JSON `query` for list/element projections; reserve Arrow for scalar analytical columns.
- `queryArrowIpc(q, { format?, params? })` — run GQL → **standard Apache Arrow IPC** bytes, framed natively (no JS re-encode), for hand-off to DuckDB / Polars / pandas. `format` picks the IPC `'stream'` (default) or `'file'` / Feather-v2 layout. To transcode an existing `ARW1` blob JS-side instead, `toArrowIPC(blob, format)` from the `@lenke/native/arrow` subpath produces byte-identical bytes with zero runtime deps.
- `gremlin(q, ...subs)` — run textual Gremlin → JSON-decoded `unknown[]`. **Use the tagged-template form** — it is Gremlin's parameter binding; see [Passing values into a query](#passing-values-into-a-query).
- `toNdjson()` — serialize back to NDJSON bytes.
- `serialize(format)` — serialize to a named format (`pg-json | pg-text | graphson | csv | ndjson`).
- `mergeNdjson(bytes)` — bulk-append an NDJSON batch into this live graph (a `COPY FROM`; no per-record round-trip, indexes stay current). Returns a `MergeReport` (`{ nodesAdded, edgesAdded, nodesSkipped, edgesSkipped, phantomVertices }`) so a conflicting/partial merge is auditable.
- `createIndex({ on, kind, keys })` (+ `drop*Index`, `vertexIndexes()` / `edgeIndexes()`) — opt-in property indexes; declaring one never changes results, only speed. Host-API only (no GQL `CREATE INDEX`). See the `indexes` guide for the full picture.
  - **`kind: 'hash'`** — the default for almost everything. An ordered map (`BTreeMap`) that serves equality, `IN`, **and** range from one structure, over any value type incl. temporals (so a single date column is a hash index: `WHERE at = $d` and `WHERE at >= $d1 AND at < $d2` both seek). Keys can be **dotted** to index a nested field: `{ on: 'vertex', kind: 'hash', keys: ['meta.errorId'] }`.
  - **`kind: 'interval'`** — an edge-only RI-tree over a half-open `[lo, hi)` pair (`keys: ['vf', 'vt']`, lo inclusive / hi exclusive) for containment (`vf <= $v AND vt > $v`) and overlap seeks that a hash index can't do in one pass. Reach for it for bitemporal/valid-time history, reservations/calendars, or numeric ranges (price/version bands) — not for a single-instant column.
- `free()` — release the underlying graph; the handle is invalid afterward. A `FinalizationRegistry` reclaims a leaked handle as a **best-effort backstop** (and warns once in dev), but GC timing is not guaranteed — prefer an explicit `free()` or a `using` binding for prompt, deterministic release.
- `prepare(text)` — parse/lower a GQL query once into a `PreparedQuery` (`.query(params)` / `.queryArrow(params)`). It has **no GC backstop** (unlike the graph handle): release it with `free()` or a `using` binding, or it leaks.
- `pagerank(config?)` / `personalizedPagerank(config?)` / `connectedComponents(config?)` / `stronglyConnectedComponents(config?)` / `onCycle(config?)` / `labelPropagation(config?)` / `peerPressure(config?)` / `degree(config?)` / `betweenness(config?)` / `closeness(config?)` / `shortestPath(config?)` — the in-engine graph algorithms, run on a **libuv threadpool thread** (genuinely off the JS thread, keeping the engine's rayon parallelism). Each returns a `Promise<Row[]>` (`{ node, score }`, `{ node, componentId }`, `{ node, centrality }`, `{ node, onCycle }`, …); a `writeProperty` config writes each result back onto its vertex. **Single-flight:** while the promise is pending the graph is locked — any other call throws `E_INVALID_GRAPH_OP` until it settles, so `await` before the next call. The `config` shape and results are identical to the `@lenke/core` free functions, and the algorithms are equally reachable from GQL (`CALL pagerank() YIELD node, score`) and Gremlin (`g.V().pageRank()`). `betweenness` (Brandes) and `closeness` are directed, unnormalized shortest-path centralities — **O(V·E)**, so keep them small-to-mid or pass `betweenness({ pivots: k })` for a byte-identical approximate run.

```ts
const scores = await g.pagerank({ iterations: 20, writeProperty: 'pr' });
```

## Passing values into a query

The two query languages bind values differently, and the difference is load-bearing.

**GQL has engine-side binding.** Values travel beside the text and are decoded by the crate, so a value never reaches the parser:

```ts
g.query('MATCH (p:Person) WHERE p.name = $name RETURN p.age AS age', { name: userInput });
g.query`MATCH (p:Person) WHERE p.name = ${userInput} RETURN p.age AS age`; // → $p0
```

**Gremlin has none.** There is no `$name` in the traversal language, so values are escaped _into_ the text at compose time. The tagged template is that seam — it is the binding mechanism, not a convenience:

```ts
const asOf = { '@date': '2021-06-01' };

g.gremlin`g.V().has('name', ${userInput}).values('age')`;
g.gremlin`g.V().has('id', eq(${id})).outE('R').has('vf', lte(${asOf})).inV().values('id')`;
```

A string interpolation becomes a single quoted literal with `\` and `'` escaped exactly as the lexer decodes them, so a hostile value collapses to inert data rather than closing the quote and injecting steps:

```ts
const evil = "marko'); g.V().drop(); //";
g.gremlin`g.V().has('name', ${evil}).values('name')`; // → [] — and the graph is intact
```

`escapeGremlin(value)` is the per-value rule, exported for building text yourself: strings quote-and-escape; finite non-exponential numbers, bigints and booleans pass through; temporals become their literal constructors (`date('2021-06-01')`, `duration('P1D')`), accepting either a stored instance or the tagged wire form `{"@date": …}`. Anything else — `null`, arrays, plain objects — has no Gremlin literal and throws. The `GremlinLiteral` type names exactly that set, so an unembeddable value is a **compile** error.

Two shapes to avoid, both rejected by the types:

```ts
g.gremlin('g.V().has($v)', { v: asOf }); // ✗ not a binding form — Gremlin has no $name
g.gremlin('g.V().has(' + userInput + ')'); // ✗ string concatenation is the injection
```

Pass a plain string only when it is a constant. To build traversal text for somewhere else — a sync client, a log, a stored query — use the exported `gremlin` tag directly; it returns the composed string instead of running it.

## Reactive store

`createStore(graph)` builds a framework-agnostic store designed for React's `useSyncExternalStore` (the package has no React dependency). `store.liveQuery(text, { deps, params? })` returns a `{ subscribe, getSnapshot }` pair whose snapshot reference is stable until a relevant mutation occurs; `store.mutate(fn)` runs a mutating callback and notifies subscribers only if the graph's `version` actually changed. `deps` is required — the label / edge-type / property-key tokens whose epochs re-run the query (`null` = recompute on any change); `inferDeps(text)` best-effort extracts them from a query string. `params` binds `$name` placeholders safely.

Release the store (and its underlying graph handle) with a `using` binding or an explicit `store[Symbol.dispose]()` — the store has no `free()` method (unlike the raw graph); disposing it frees the graph.

## License

Apache-2.0
