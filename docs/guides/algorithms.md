# In-engine graph algorithms

lenke runs whole-graph algorithms **inside** the engine — PageRank (and personalized PageRank), connected components, strongly-connected components, cycle membership, label propagation, degree, shortest path, peer pressure, and betweenness/closeness centrality. They are not a bolt-on library that pulls your graph out into JS arrays; they execute against the live store. By default they run **single-threaded** — the pure-TS driver time-slices so it never blocks the event loop, and on the native engine each run happens genuinely off the JS thread. The float-heavy ones can additionally use **opt-in multicore** on the native build (see [Parallelism](#parallelism) below).

The same computation is reachable from **four surfaces**. Pick the one that fits your call site — the `config` shape and the results are identical across all of them, and the numeric output is **byte-identical** across the pure-TS and Rust engines (a fixed summation order is the rule that guarantees it, so a score computed in the browser matches one computed on the server bit for bit).

## The shipped algorithms

| Name                          | Result rows             | What it computes                                                |
| ----------------------------- | ----------------------- | --------------------------------------------------------------- |
| `pagerank`                    | `{ node, score }`       | Influence / centrality by link structure.                       |
| `personalizedPagerank`        | `{ node, score }`       | PageRank restarted to a `sourceNodes` seed set (proximity/RWR). |
| `connectedComponents`         | `{ node, componentId }` | Weakly-connected component membership (edges undirected).       |
| `stronglyConnectedComponents` | `{ node, componentId }` | Strongly-connected component membership (directed).             |
| `onCycle`                     | `{ node, onCycle }`     | Whether a node lies on a directed cycle (SCC>1 or self-loop).   |
| `labelPropagation`            | `{ node, label }`       | Community detection by label spreading.                         |
| `peerPressure`                | `{ node, label }`       | Community detection by majority vote.                           |
| `degree`                      | `{ node, degree }`      | In/out/total edge count per node.                               |
| `shortestPath`                | `{ node, distance }`    | Shortest path from `source` (BFS / Dijkstra / A\*).             |
| `betweenness`                 | `{ node, centrality }`  | Brokerage — how often a node lies on shortest paths.            |
| `closeness`                   | `{ node, centrality }`  | Reciprocal of total distance to reachable nodes.                |
| `neighborAggregate`           | `{ node, vector }`      | GNN message passing: aggregate neighbors' feature vectors.      |

The shared `config` object (all fields optional) carries `edgeLabel`, `direction` (`'out' | 'in' | 'both'`), `weightProperty`, `dampingFactor`, `iterations`, `source`/`target`, `writeProperty`, and a few algorithm-specific knobs: `sourceNodes` (personalized-PageRank seed set), `pivots` (approximate betweenness — see below), `seedProperty` (label-propagation anchors — a vertex carrying a non-null value for that key keeps its own label, so communities form around seeds instead of collapsing on a hubby graph), and `feature`/`op`/`includeSelf`/`norm` (`neighborAggregate` — see below). It is portable _verbatim_ across the four surfaces below.

> **Reserved-word footgun on `writeProperty`.** The result is written back as a property, so pick a name that isn't a GQL reserved word — `closeness({ writeProperty: 'close' })` writes fine, but reading it back as `n.close` is `E_SYNTAX` (`close` is reserved). Use a safe name (`closeness_c`, `cc`) or quote it with backticks (`` n.`close` ``). Same for `size`, `count`, etc.

> **Centrality cost — exact vs approximate.** `betweenness` (Brandes' algorithm) and `closeness` are exact and byte-identical across engines, but **O(V·E)** — every node runs a full traversal. Fine for thousands of nodes; past ~100k, pass `betweenness({ pivots: k })` for an **approximate** run — Brandes from a deterministic evenly-spaced sample of `k` sources scaled by `|V|/k`, so it's O(pivots·E) and still byte-identical across engines (`pivots >= |V|` is exact). `betweenness`/`closeness` are directed and unnormalized (`closeness = 1 / Σ distance`, 0 when nothing is reachable).
>
> **Memory sizing.** A whole-graph algorithm allocates per-vertex working state on top of the resident graph, so peak RSS during a run is roughly **~2× the resident graph** (budget ≈760 B/element resident as a rough guide). A ~30M-element graph therefore wants ~40+ GB to run centrality/PageRank comfortably; size the host accordingly, or fan the work out (per-component, or the `pivots` sample for betweenness) when the graph doesn't fit ~2× in RAM.

### `neighborAggregate` — message passing / GNN feature engineering

For each vertex, aggregate its neighbors' **list-valued** `feature` vectors element-wise over the whole `D`-dim block in one native pass — the message-passing / graph-convolution primitive, instead of `D` separate GQL `SET`s. It returns `{ node, vector }` rows (and, with `writeProperty`, writes the aggregated list back onto each vertex, so a downstream query or a second layer can read it — vector properties egress as a real Arrow `FixedSizeList<Float64>`, see the Arrow guide).

Config (beyond `feature`, `edgeLabel`, `direction`, `writeProperty`):

| key              | values                                     | meaning                                                                                                                                                                   |
| ---------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `feature`        | property key (**required**)                | Each vertex's list-of-numbers feature vector (all vertices must share the dimension).                                                                                     |
| `op`             | `'mean'` (default) `'sum'` `'max'` `'min'` | Element-wise reduction over contributors.                                                                                                                                 |
| `direction`      | `'out'` `'in'` `'both'` (default)          | Which neighbors contribute.                                                                                                                                               |
| `includeSelf`    | `false` (default) / `true`                 | Fold the vertex's own vector in too (the GCN self-loop).                                                                                                                  |
| `weightProperty` | numeric edge property                      | **Weighted** aggregation: contributor `j` is scaled by its edge weight (weighted `mean` divides by Σ weights). `sum`/`mean` only.                                         |
| `norm`           | `'none'` (default) `'gcn'`                 | **GCN symmetric normalization**: contributor `j` of `i` scaled by `1/sqrt(deg_i·deg_j)`. Composes with `weightProperty` (coefficient = weight × norm). `sum`/`mean` only. |

A `weightProperty`/`norm` is rejected for `max`/`min` (scale-independent). Standard GCN = `{ op: 'sum', direction: 'both', includeSelf: true, norm: 'gcn' }`. Byte-identical across engines (fixed edge-index accumulation order; integer degrees make the `1/sqrt(deg_i·deg_j)` match bit for bit).

```ts
// One GCN-style message-passing layer, written back for the next layer / egress.
await neighborAggregate(
  {
    feature: 'h',
    op: 'sum',
    direction: 'both',
    includeSelf: true,
    norm: 'gcn',
    writeProperty: 'h1',
  },
  g,
);
```

## Surface 1 — `@lenke/core` async free functions

Data-last, always-async free functions. The `async` is deliberate: a long run never blocks the loop.

```ts
import { Graph, pagerank, connectedComponents, degree } from '@lenke/core';
import { query } from '@lenke/gql';

const g = new Graph();
query(g, `INSERT (:P {id:'a'}), (:P {id:'b'}), (:P {id:'c'})`);
query(g, `MATCH (a:P {id:'a'}),(b:P {id:'b'}) INSERT (a)-[:F]->(b)`);
query(g, `MATCH (b:P {id:'b'}),(c:P {id:'c'}) INSERT (b)-[:F]->(c)`);
query(g, `MATCH (a:P {id:'a'}),(c:P {id:'c'}) INSERT (a)-[:F]->(c)`);

const scores = await pagerank({ iterations: 20 }, g);
// → [{ node: '…', score: 0.197… }, { node: '…', score: 0.281… }, { node: '…', score: 0.520… }]
```

They compose data-last, so `pagerank(config)` partially applied also works under a `pipe`.

### Feature write-back with `writeProperty`

Give any algorithm a `writeProperty` and it writes each result back **onto its vertex** as a property — turning a computed score into queryable graph data:

```ts
await pagerank({ iterations: 20, writeProperty: 'pr' }, g);
const rows = query(g, `MATCH (p:P) RETURN p.id AS id, p.pr AS pr ORDER BY pr DESC`);
// → [{ id: 'c', pr: 0.520… }, { id: 'b', pr: 0.281… }, { id: 'a', pr: 0.197… }]
```

Now every downstream GQL/Gremlin query can filter, sort, and traverse on `pr` — the algorithm's output has become a first-class feature of the graph.

## Surface 2 — native `RustGraph` methods

The identical algorithms hang off the native graph handle, each returning a `Promise<Row[]>` and running on a libuv threadpool thread (off the JS thread; single-threaded by default — opt into multicore with `parallelism`/`threads`, see [Parallelism](#parallelism)):

```ts
const scores = await g.pagerank({ iterations: 20 });
await g.pagerank({ iterations: 20, writeProperty: 'pr' }); // same write-back
const comps = await g.connectedComponents();
```

**Single-flight:** while an algorithm promise is pending the graph is locked — any other engine call throws `E_INVALID_GRAPH_OP` until it settles. Always `await` one before issuing the next.

## Surface 3 — the ISO GQL `CALL` procedure

The conformant home for the algorithms inside a query: a named procedure with a config map, `YIELD`ing its result columns, which you then treat as an ordinary row source:

```ts
const top = query(
  g,
  `CALL pagerank({ iterations: 20 }) YIELD node, score
   RETURN score ORDER BY score DESC LIMIT 1`,
);
// → [{ score: 0.520… }]
```

`YIELD` names the columns the procedure produces (`node`, `score` for PageRank; `node`, `componentId` for components; and so on), and everything after it is normal GQL — `WHERE`, `ORDER BY`, `RETURN`, joins against other patterns.

> **Procedure names are `snake_case` in `CALL`.** The GQL catalog spells the algorithms `pagerank`, `personalized_pagerank`, `connected_components`, `strongly_connected_components`, `on_cycle`, `label_propagation`, `peer_pressure`, `degree`, `betweenness`, `closeness`, `shortest_path` — the `snake_case` form, **not** the camelCase of the JS free functions / `RustGraph` methods / Gremlin steps in the table above. A camelCase spelling faults with a hint: `CALL connectedComponents(...)` → `E_UNSUPPORTED: unknown procedure: connectedComponents (did you mean 'connected_components'?)` (both engines, byte-identical).

> **`node` is the whole vertex element, not just an id.** The yielded `node` binds a full graph element, so you can read its properties (`node.name`) and — crucially — **join it back into the graph** in the same query. This makes a two-line "rank, then expand its neighbourhood" query trivial:
>
> ```ts
> query(
>   g,
>   `CALL pagerank({ iterations: 20 }) YIELD node, score
>    MATCH (node)-[:KNOWS]->(friend)
>    RETURN node.name AS influencer, friend.name AS reaches, score
>    ORDER BY score DESC LIMIT 3`,
> );
> ```
>
> Because `node` is a real element, `MATCH (node)-[…]->()` resolves against the same vertex the procedure scored — no id round-trip, no re-lookup. The ranking (`ORDER BY score DESC LIMIT 3`) lives on the **final `RETURN`**: a standalone `ORDER BY`/`LIMIT` cannot sit as its own clause between `YIELD` and `MATCH` (that is `E_SYNTAX`) — order and limit the result rows, not the mid-query stream.

## Surface 4 — Gremlin steps

The Gremlin frontend exposes the same computations as traversal steps:

```ts
// g.V().pageRank()  — and the degree/component analogues
```

## Which surface to reach for

- **A one-shot analysis in application code** → the `@lenke/core` free function or the native method.
- **A score you want to keep and query** → any surface with `writeProperty`, then read the property back.
- **An algorithm as one stage of a larger query** → the GQL `CALL … YIELD` form (compose with `WHERE`/`ORDER BY`/`RETURN`).
- **A Gremlin shop** → the traversal steps.

All four are the same engine code path — the choice is purely about where the call lives, never about what it computes.

## Parallelism

The float-heavy algorithms — **betweenness**, **closeness**, **PageRank**, **personalized PageRank**, **label propagation**, **peer pressure** (and **degree**) — can run across multiple cores on the **native engine**. It is strictly **opt-in**: the default is one thread (serial), so single-core performance is unchanged and nothing you don't ask for competes with your process.

Set it per graph, or override per call:

```ts
// Graph-level default for every algorithm run on this graph:
const g = graphFromNdjson(backend, ndjson, { parallelism: 8 });
await g.betweenness(); // uses up to 8 workers

// Or per call (overrides the graph default):
await g.betweenness({ threads: 8 });
await g.closeness({ threads: 4 });
```

From GQL, pass `threads` in the `CALL` config: `CALL betweenness({ threads: 8 }) YIELD node, centrality`.

Two guarantees make this safe to turn on:

- **It will not starve your process.** Each run uses a **dedicated, bounded pool** of exactly the thread count you asked for — never a global pool sized to every core — so the host's event loop and other work keep running. A conservative number (say, half your cores) is usually the right call on a shared machine.
- **Results are byte-identical at any thread count.** A parallel run returns bit-for-bit the same scores as a serial one (and the same as the pure-TS engine): the parallelized work is either independent per node or folded back in a fixed order, so no float sum is ever reassociated. `threads: 1` and `threads: 16` produce identical output.

Not available on the **wasm** build (WebAssembly has no threads) or the **pure-TS** engine — both accept the setting for API symmetry and run serially. Betweenness and closeness (the `O(V·E)` centralities) benefit most; a typical 8-thread run is several times faster than serial on a mid-size graph.
