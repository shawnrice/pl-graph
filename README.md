# lenke

> A labeled-property-graph toolkit for JavaScript & TypeScript: an in-memory graph you query with **ISO-GQL** or **Gremlin** — as a pure-TypeScript library, or backed by a **Rust columnar engine** over FFI / WebAssembly.

lenke models data as a labeled property graph — vertices and edges that each carry a set of labels and a bag of properties — and lets you query it with **ISO-GQL** or **Apache Gremlin**, with no external database to run. The same data model and query surface come in two interchangeable engines: a **pure-TypeScript** one that runs anywhere with zero native dependencies, and a **Rust columnar core** for throughput, reachable over `bun:ffi`, N-API, or **WebAssembly**. React bindings expose either as a reactive store.

## What you can build

lenke is a toolkit you compose, not a single binary — you pull in the pieces a given job needs. A few things it's a good fit for:

- **Query connected data in-process, without a graph database.** When your data _is_ the relationships — a social graph, a dependency or org tree, a service/topology map, an access-control graph — model it directly and query it with GQL or Gremlin in your app's memory — no separate graph database to run, no recursive SQL to hand-write. Pure TypeScript, any JS runtime.
  → [pure-ts guide](docs/guides/pure-ts.md)

- **Back a React UI with a reactive graph.** `@lenke/react` renders components straight off a live graph and re-runs a selector only when a mutation touches the labels or keys it actually reads — so a large graph doesn't re-render the tree on every edit. Fits dashboards, topology/network views, and graph editors.
  → [frontend guide](docs/guides/frontend-main-thread.md)

- **Embed a fast in-memory graph cache on a server.** Bulk-load a snapshot (NDJSON, CSV, GraphSON) into the Rust columnar engine and serve GQL/Gremlin from a Node or Bun process — a materialized view or read cache beside your system of record, with Apache Arrow output for columnar hand-off. One process can hold many isolated graphs (multi-tenant).
  → [backend-embedded guide](docs/guides/backend-embedded.md)

- **Go local-first with live queries.** Keep the graph in a web worker (the WebAssembly engine) and subscribe to standing queries instead of re-fetching; a result pushes only when it actually changes. The sync host is transport-agnostic — the _identical_ host serves a Worker port in the browser or a WebSocket from a server — so the same live-query code runs offline-capable or client/server.
  → [frontend-worker guide](docs/guides/frontend-worker.md)

Two worked examples: [`examples/service-map`](examples/service-map) threads one feature through the entire stack — React → worker → sync engine → wasm store, with a Node server host — and [`examples/explorer`](examples/explorer) is a visual graph explorer (force-directed diagram + GQL highlighting) on the pure-TS engine.

## Quick start

### Query a graph with GQL (pure TypeScript)

```ts
import { Graph } from '@lenke/core';
import { query } from '@lenke/gql';

const g = new Graph();
const marko = g.addVertex({ labels: ['Person'], properties: { name: 'marko', age: 29 } });
const josh = g.addVertex({ labels: ['Person'], properties: { name: 'josh', age: 32 } });
g.addEdge({ from: marko, to: josh, labels: ['KNOWS'], properties: {} });

const rows = query(g, `MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name AS friend`);
// => [{ friend: 'josh' }]
```

The same graph can be traversed with Gremlin instead — see [`@lenke/gremlin`](packages/gremlin).

### Use it in React

```tsx
import { Graph } from '@lenke/core';
import { GraphProvider, useGraphSelector } from '@lenke/react';

const graph = new Graph();
graph.addVertex({ labels: ['Person'], properties: { name: 'marko' } });

function PeopleCount() {
  // `deps` scopes invalidation to the 'Person' label, so unrelated mutations
  // don't re-run the selector or re-render this component.
  const count = useGraphSelector((g) => g.getVerticesByLabel('Person').size, Object.is, ['Person']);
  return <p>{count} people</p>;
}

export function App() {
  return (
    <GraphProvider graph={graph}>
      <PeopleCount />
    </GraphProvider>
  );
}
```

For a native-backed store, drive components with `useLiveQuery` over a `@lenke/native` store instead — see [`@lenke/react`](packages/react).

### Run the Rust engine via WebAssembly (Node)

The WebAssembly backend needs no native addon, so it runs the Rust engine from plain Node (or Deno, or the browser):

```ts
import { readFile } from 'node:fs/promises';
import { createWasmEngineBackend } from '@lenke/native/wasm-engine';
import { graphFromNdjson } from '@lenke/native';

const backend = await createWasmEngineBackend(await readFile('lenke_engine.wasm'));

// The graph is heap-owned by the wasm module; `using` frees it at scope exit
// (or call `g.free()` explicitly). Same rule on the ffi and N-API backends.
using g = graphFromNdjson(backend, await readFile('graph.ndjson'));

const rows = g.query`MATCH (p:Person) RETURN p.name AS name`;
console.log(rows); // [{ name: 'marko' }, ...]
```

Under Bun, swap `@lenke/native/wasm-engine` for `@lenke/native/ffi-engine` (`createFfiEngineBackend(libPath)`) to load the native dynamic library directly — the rest of the API is identical.

## How it fits together

- **Graph & queries (pure TS).** [`@lenke/core`](packages/core) is the in-memory graph; [`@lenke/gql`](packages/gql) and [`@lenke/gremlin`](packages/gremlin) query it; [`@lenke/serialization`](packages/serialization) reads and writes it.
- **Native engine (Rust).** [`lenke-engine`](crates/lenke-engine) is a columnar reimplementation of the graph and both query engines; [`@lenke/native`](packages/native) binds it to JS via FFI or wasm. Same query languages, more throughput.
- **React.** [`@lenke/react`](packages/react) drives components from either the in-process graph or the native store, re-rendering only when a relevant mutation changes what a component reads.
- **Live queries & sync.** [`@lenke/sync`](packages/sync) turns a store into a declarative live-query service over any port-shaped channel — a Worker in the browser or a WebSocket on a server — pushing an update only when a standing query's result actually changes.
- **Building blocks.** Small standalone primitives the rest are built on.

## Packages

**Graph & queries**

| Package                                          | Description                                                              |
| ------------------------------------------------ | ------------------------------------------------------------------------ |
| [`@lenke/core`](packages/core)                   | In-memory labeled-property graph with label and opt-in property indexes. |
| [`@lenke/gql`](packages/gql)                     | ISO-GQL (ISO/IEC 39075) query engine over a core graph.                  |
| [`@lenke/gremlin`](packages/gremlin)             | Apache TinkerPop-style Gremlin traversal engine over a core graph.       |
| [`@lenke/serialization`](packages/serialization) | Graph codecs: pg-json, pg-text, ndjson, GraphSON, CSV.                   |

**Native engine (Rust)**

| Package                            | Description                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| [`lenke-engine`](crates/lenke-engine)  | Rust columnar graph + GQL/Gremlin engines + Apache Arrow output, behind a C ABI. |
| [`@lenke/native`](packages/native) | JS/TS bindings to the Rust core via `bun:ffi` or WebAssembly.                    |

**React & live queries**

| Package                          | Description                                                                             |
| -------------------------------- | --------------------------------------------------------------------------------------- |
| [`@lenke/react`](packages/react) | Hooks and providers exposing a graph (TS or native) as a reactive store.                |
| [`@lenke/sync`](packages/sync)   | Declarative live-query protocol + a transport-agnostic host (WebSocket or Worker port). |

**Building blocks**

| Package                              | Description                                                            |
| ------------------------------------ | ---------------------------------------------------------------------- |
| [`@lenke/emitter`](packages/emitter) | Typed, cancelable, error-isolated event emitter.                       |
| [`@lenke/errors`](packages/errors)   | Stable `E_*` error codes and a shared `LenkeError` type.               |
| [`@lenke/fp`](packages/fp)           | Lazy, curried iterable combinators composed with `pipe`.               |
| [`@lenke/list`](packages/list)       | Lazy, iterator-backed `List<T>`.                                       |
| [`@lenke/tree`](packages/tree)       | `TreeNode` and `Trie` data structures.                                 |
| [`@lenke/utils`](packages/utils)     | Small shared helpers.                                                  |
| [`@lenke/inspect`](packages/inspect) | Debug helpers: `describe()` a graph, tabulate results, print elements. |
| [`@lenke/cli`](packages/cli)         | `lenke` REPL/CLI: load a codec, query in GQL/Gremlin, serialize out.   |
| [`@lenke/lint`](packages/lint)       | oxlint/ESLint plugin flagging raw interpolation into query text.       |
| [`@lenke/dev`](packages/dev)         | Internal build & lint tooling (bundler, lint rules, shared config).    |

Each package has its own README with a full API walkthrough.

## Guides

The package READMEs are the API reference; the **[deployment guides](docs/guides/)** are task-oriented — how to wire lenke up for each way you can run it. Every deployment is a point in three orthogonal choices (engine, query frontend, reach-path); [choosing-your-build](docs/guides/choosing-your-build.md) is the matrix.

- [pure-ts](docs/guides/pure-ts.md) — the TypeScript graph, anywhere, no native artifacts
- [native](docs/guides/native.md) / [wasm](docs/guides/wasm.md) — the Rust engine via N-API / bun:ffi / WebAssembly
- [frontend-main-thread](docs/guides/frontend-main-thread.md) / [frontend-worker](docs/guides/frontend-worker.md) — React on the main thread, or a worker-resident sync engine
- [backend-embedded](docs/guides/backend-embedded.md) — an embedded cache / view machine, and multi-tenancy

## Develop

A Bun + nx monorepo (`packages/*`) plus Rust crates under `crates/` (`lenke-engine`, the foundational-types crate `lenke-engine-core`, and the shared codecs crate `lenke-codec`).

```bash
bun install

bun run check    # typecheck + lint + format check (the pre-commit gate)
bun run build    # build all packages
bun run test     # run all package tests

# Rust engine
cargo test --manifest-path crates/lenke-engine/Cargo.toml
cargo build --release --features capi --manifest-path crates/lenke-engine/Cargo.toml   # cdylib for bun:ffi
```

## License

[Apache-2.0](LICENSE)
