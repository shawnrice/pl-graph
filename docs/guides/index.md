# lenke deployment guides

lenke is a labeled-property-graph toolkit: an in-memory graph you query with **ISO-GQL** or **Gremlin**, with two interchangeable engines (pure-TypeScript or a Rust columnar core) reachable from JS over FFI, N-API, or WebAssembly. It runs the same data model and query surface in a browser tab, a web worker, a Node server, or a Bun CLI.

Because it composes rather than ships as one binary, "how do I use lenke?" has more than one answer. These guides walk each one. Start here to pick your build, then follow the guide for your topology.

## The three choices

Every deployment is a point in three **orthogonal** axes. Pick each independently.

| Axis                             | Options                                              | How you choose                                                                                                                                                                                        |
| -------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Engine** — the graph substrate | pure-TS (`@lenke/core`) · Rust core (`lenke-engine`) | Two complete, interchangeable implementations. TS runs anywhere with zero native deps; Rust is faster and columnar.                                                                                   |
| **Query frontend**               | GQL (`@lenke/gql`) · Gremlin (`@lenke/gremlin`)      | A shop standardizes on one. The graph is language-agnostic — you bolt on the frontend you use and leave the other out. On TS it's a package you install; on Rust it's a Cargo feature you compile in. |
| **Reach-path** — Rust only       | bun:ffi · N-API · WebAssembly                        | Which runtime you're on: Bun, Node, or the browser. All three expose one identical JS surface.                                                                                                        |

See **[choosing-your-build](./choosing-your-build.md)** for the full matrix, including build commands and the memory model.

## Which guide?

Pick by what you're building:

- **A graph you query in-process, anywhere** → [pure-ts](./pure-ts.md) — `@lenke/core` + a query frontend, no native artifacts.
- **A server-side cache or view machine** (Node/Bun) → [backend-embedded](./backend-embedded.md) — the Rust engine embedded in a process, bulk-loaded and queried; covers multi-tenancy.
- **The Rust engine on a server or CLI** → [native](./native.md) — N-API (`@lenke/node`, the fast Node path) and bun:ffi.
- **The Rust engine in a browser** → [wasm](./wasm.md) — `@lenke/native/wasm-engine`, async load, no native addon.
- **A React UI over a graph on the main thread** → [frontend-main-thread](./frontend-main-thread.md) — `@lenke/react`, either engine.
- **A local-first UI with the graph in a worker** → [frontend-worker](./frontend-worker.md) — `@lenke/sync` as a worker-resident cache against your own API.

The [`examples/service-map`](../../examples/service-map) app threads one feature through the whole stack (React → worker → sync engine → wasm store, and a Node server host) and is the worked reference for the frontend and backend guides. The [`examples/explorer`](../../examples/explorer) app is a smaller, pure-TS reference: a visual force-directed graph explorer that queries with GQL.

## Patterns & topics

Cross-cutting guides that apply on any build:

- **[algorithms](./algorithms.md)** — the in-engine graph algorithms (PageRank, connected/strongly-connected components, label propagation, betweenness/closeness, shortest path, …), their four call surfaces, and the byte-identity guarantee across engines.
- **[bitemporal](./bitemporal.md)** — modelling valid + transaction time from the shipped primitives (`DATE` columns, parameterized `WHERE`, the host clock, atomic corrections). Covers both the edge-period and version-node shapes, "as of" queries, and the supersession pitfalls. A documented recipe — lenke ships no bitemporal engine by design.

## Conventions in these guides

- Snippets use the real exported APIs. Where a capability is on the roadmap but not yet shipped, it's called out explicitly — the guides don't describe vaporware as if it exists.
- "The engines are the same graph" means the same substrate and data model; GQL and Gremlin stay faithful to their own ontologies, so query semantics differ only where the two query models genuinely differ.
