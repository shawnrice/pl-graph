# @lenke/core

> A mutable, in-memory labeled-property graph for JavaScript and TypeScript.

`Graph` stores vertices and edges that each carry a set of labels and a bag of properties, with built-in indexes by label and opt-in secondary indexes over property values. Reach for it when you need an in-process graph you can mutate, query, observe, and feed to a query engine — without standing up an external database.

## Install

```bash
bun add @lenke/core
```

## Usage

```ts
import { Graph } from '@lenke/core';

const graph = new Graph();

// Add vertices: each carries labels and a property bag.
const alice = graph.addVertex({ labels: ['Person'], properties: { name: 'Alice', age: 34 } });
const bob = graph.addVertex({ labels: ['Person'], properties: { name: 'Bob', age: 28 } });

// Connect them with a directed, labeled edge.
graph.addEdge({ from: alice, to: bob, labels: ['KNOWS'], properties: { since: 2020 } });

// Query by label.
graph.getVerticesByLabel('Person'); // Set { alice, bob }

// Declare a secondary index, then query by property value or range.
graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
graph.getVerticesByProperty('age', 34); // Set { alice }
graph.getVerticesByPropertyRange('age', { gte: 30 }); // Set { alice }

// Mutate properties through the element's methods (never write to `.properties`).
alice.setProperty('age', 35);
alice.getProperty('age'); // 35 (typed `unknown`)
alice.getProperty<number>('age'); // 35 (typed `number` — no cast)

// Traverse edges from a vertex by edge label.
alice.edgesFromByLabel('KNOWS'); // Set { the alice->bob edge }
```

## Vertices and edges

`Vertex` and `Edge` are concrete (non-generic) classes. Properties are typed as `Record<string, unknown>`; to read one without a cast, pass a type to `getProperty` — `vertex.getProperty<string>('name')` returns `string` (an opt-in, caller-side assertion — nothing is validated; use `<string | undefined>` if the key might be absent). Or cast the whole bag at the boundary (`vertex.properties as Person`).

The `.properties` getter returns the graph's live, frozen property bag — a top-level write throws. Mutate through the element instead:

```ts
vertex.setProperty(key, value);
vertex.setProperties({ a: 1, b: 2 });
vertex.removeProperty(key);
vertex.removeProperties(['a', 'b']);
vertex.addLabel(label);
vertex.removeLabel(label);
```

`Edge` exposes the same property and label methods, plus `from` / `to` vertex accessors. Every edge is directed and must carry at least one label; `addEdge` throws if an endpoint isn't in the graph or no label is given. Removing a vertex cascades to remove its incident edges.

## Property indexes

Secondary indexes are opt-in per key — declare one with `createIndex({ on, kind, keys })` (backfilled from existing elements and kept current on every mutation), then query it. Use **`kind: 'hash'`** for almost everything (`{ on: 'vertex', kind: 'hash', keys: ['age'] }`) — an ordered map that serves equality, `IN`, and range from one structure. A key can be **dotted** to index a nested field: `keys: ['meta.errorId']`. (The `'interval'` kind — an `[lo, hi)` containment/overlap index — is native-engine only; the pure-TS engine throws on it. See `@lenke/native`.)

- `getVerticesByProperty(key, value)` / `getEdgesByProperty(key, value)` — equality, an O(1) bucket lookup.
- `getVerticesByPropertyRange(key, bound)` / `getEdgesByPropertyRange(key, bound)` — range, where `bound` is a `RangeBound` (`{ gt?, gte?, lt?, lte? }`). Indexable values are `string | number | boolean | null`; ranges are clamped to the bound's value type.

Drop an index with `dropVertexIndex(key)` / `dropEdgeIndex(key)`, and list active indexes with `vertexIndexes()` / `edgeIndexes()`. Querying an unindexed key returns an empty set.

## Events and reactivity

Mutations emit typed graph events you can listen to with `graph.on(type, listener)` / `graph.once(...)`; **`graph.on` returns an unsubscribe function** (call it to detach). Events are **observation-only** — a notification that a write happened, for side effects and reactivity (there is no veto; enforce rules with constraints instead, below). Inside a `graph.transaction(...)` events are buffered and dispatched as one batch on commit (and discarded on rollback), so an emitted event always corresponds to a committed write.

**Read the payload off `event.value`, not off `event`.** Each listener receives an event whose `.value` is the payload object — a common trip-up is reaching for `event.previous` (which is `undefined`); it's `event.value.previous`:

```ts
const off = graph.on('@graph/VertexPropertyChanged', (event) => {
  const { vertex, key, value, previous } = event.value;
  //     ^element  ^key   ^new   ^old (undefined if the key was absent)
  console.log(`${vertex.id}.${key}: ${previous} → ${value}`);
});
off(); // detach
```

The singular property-change events (`VertexPropertyChanged` / `EdgePropertyChanged`) carry `previous` alongside `value` — enough to build an undo/redo stack purely from events. `graph.subscribe(callback)` registers a coalesced change callback (one deferred notification per tick) and returns an unsubscribe function. `graph.version` is a monotonic mutation counter, and `graph.epoch(name)` is a per-token (label / edge-type / property-key) change counter — both suited to `useSyncExternalStore`-style snapshots. Use `enableEvents()` / `disableEvents()` to toggle emission.

`graph.clone()` produces an independent deep copy; `graph.snapshot()` returns the live graph for reads; `graph.truncate()` empties the graph while keeping declared indexes.

## Transactions

Wrap a set of writes in `graph.transaction(fn)` and they apply atomically: the callback's writes commit together, or — if it throws — roll back as if none happened (buffered events are discarded, not dispatched). This is the engine-neutral transaction surface: the same mechanism backs both `@lenke/gql` and `@lenke/gremlin`, and any constraints (below) are checked once at the commit boundary.

```ts
const a = graph.addVertex({ labels: ['Acct'], properties: { name: 'a', balance: 100 } });
const b = graph.addVertex({ labels: ['Acct'], properties: { name: 'b', balance: 0 } });

try {
  graph.transaction(() => {
    a.setProperty('balance', a.getProperty<number>('balance') - 100);
    b.setProperty('balance', b.getProperty<number>('balance') + 100);
    throw new Error('abort'); // → both writes revert
  });
} catch {}
// a.balance === 100, b.balance === 0  (neither write survived)
```

For explicit control there's a TinkerPop-style handle — `const t = graph.tx(); …; t.commit()` (or `t.rollback()`). Transactions are flat (savepoint-less): a nested `transaction(...)` joins the outer frame, which owns the commit.

## Constraints

Declare integrity rules enforced **at the mutation boundary** — a violating write throws `ErrorCode.ConstraintViolation` (and, inside a transaction, rolls the whole thing back). They're host APIs (not query DDL), so both query engines honor them identically, and declaring one that existing data already violates throws immediately.

```ts
graph.createUniqueConstraint('User', 'email'); // ≤ one live :User per non-null email
graph.createRequiredConstraint('User', 'name'); // every :User must carry name
graph.createTypeConstraint('User', 'age', 'number'); // age, if present, is a number
```

- `createUniqueConstraint(label, key)` — index-backed; null values are exempt (SQL semantics). It's also what `_MERGE` upserts on (see [`@lenke/gql`](../gql)).
- `createRequiredConstraint(label, key)` — the key must be present.
- `createTypeConstraint(label, key, type)` — `type` is a `ScalarTypeName` (`'string' | 'number' | 'boolean' | 'date' | 'datetime' | 'duration' | 'list'`).
- `createCardinalityConstraint(label, edgeType, direction, min, max)` — bound a vertex's degree along an edge type (`direction` is `'out' | 'in'`; `max` may be `null` for unbounded). A **`min ≥ 1`** constraint makes node creation atomic with its required edges: under per-statement atomicity a bare `INSERT (:Airport)` faults immediately (0 edges `<` min), since each statement is its own transaction. Create the node and its required edge in **one** statement (`INSERT (:Airport)-[:LOCATED_IN]->(:City {…})`), or wrap both writes in a `graph.transaction(…)` — the min check runs once at the commit boundary, so the intermediate 0-edge state is allowed.

Edge analogues exist too (`createEdgeUniqueConstraint`, `createEdgeRequiredConstraint`, `createEdgeTypeConstraint`), each with `drop*` / introspection companions. `@lenke/gql` layers two GQL-expression constraints on top — `createValidator(graph, label, varName, predicate)` (a per-element SQL-`CHECK`) and `createInvariant(graph, name, query)` (a whole-graph assertion re-checked at each commit).

## Schema validation (`defineNode`)

`defineNode(label, schema)` binds a node label to any [Standard Schema](https://standardschema.dev) — a Zod (≥3.24), Valibot, or ArkType object schema, or anything exposing `~standard` — with **zero added dependency** and full type inference (lenke owns no schema DSL; you bring your validator):

```ts
import { z } from 'zod';
import { defineNode } from '@lenke/core';

const User = defineNode('User', z.object({ name: z.string().trim(), age: z.number().optional() }));

const v = await User.create(graph, { name: '  ada  ' }); // validates, then writes a :User vertex
User.parse({ name: 'ada' }); // validate only, no write
```

`create` validates HOST-side and stores the schema's **output** — a schema that trims/defaults/coerces persists the normalized value (`v.getProperty('name') === 'ada'` above). A failure throws `ConstraintViolation` listing every issue; both `create` and `parse` are async (a Standard Schema may validate asynchronously). Because it runs in JS before the write, `defineNode` guards `create` calls — **not** a raw GQL `INSERT`; for engine-level enforcement that also covers raw writes, use the constraints above. The two compose cleanly: a schema at the app boundary, constraints at the engine.

`defineEdge(edgeType, schema)` is the edge-property mirror. Its `create` takes the two endpoints — **`Vertex` objects _or_ bare vertex ids** (you rarely still hold the `Vertex`) — followed by the typed props:

```ts
import { defineEdge } from '@lenke/core';

const Follows = defineEdge(
  'FOLLOWS',
  z.object({ since: z.number(), tag: z.string().trim().optional() }),
);

const e = await Follows.create(graph, ada.id, lin.id, { since: 2020 }); // validates, then writes an :FOLLOWS edge
Follows.parse({ since: 2020 }); // validate only, no write
```

An unresolved endpoint id throws `MissingVertex`. Everything else matches `defineNode`: validate-before-write, stores the coerced output, async, host-side only. It composes with the engine edge constraints (`createEdgeRequiredConstraint` / `createEdgeTypeConstraint` / `createEdgeUniqueConstraint`) exactly as `defineNode` composes with the vertex constraints.

## Graph algorithms

Whole-graph computations ship as data-last, **async** free functions — `degree`, `connectedComponents`, `stronglyConnectedComponents`, `onCycle`, `labelPropagation`, `pagerank`, `personalizedPagerank`, `peerPressure`, `betweenness`, `closeness`, `shortestPath`:

```ts
import { pagerank } from '@lenke/core';

const scores = await pagerank({ iterations: 20 }, graph);
// → [{ node: '…', score: 0.2128 }, …]   (data-last: pagerank(config)(graph) also composes under pipe)
```

They're always async so a long run never blocks the event loop (the pure-TS driver checkpoints on a ~5 ms time budget). A `writeProperty` config writes each result back onto its vertex — `pagerank({ writeProperty: 'pr' }, graph)`, then read `p.pr`. The `config` shape (`edgeLabel`, `direction`, `weightProperty`, `dampingFactor`, `iterations`, `source`/`target`, …) is portable verbatim to the native engine. The same computations are reachable from the native `RustGraph` (`g.pagerank(config)`, off-thread), from GQL (`CALL pagerank() YIELD node, score`), and from Gremlin (`pageRank()`) — **byte-identical** across all four. See [`src/algorithms/README.md`](src/algorithms/README.md) for the worker-offload recipe and which form to reach for.

`betweenness` and `closeness` are shortest-path **centrality** measures over the directed graph (out-edges, optionally one `edgeLabel`, unweighted BFS or weighted via `weightProperty`), each yielding a `{ node, centrality }` row: `betweenness` is Brandes' algorithm (directed, **unnormalized** — no `1/((n-1)(n-2))` scaling), `closeness` is `1 / Σ d(s,t)` over reachable `t` (**unnormalized**; a vertex that reaches nothing scores 0). Both run one shortest-path pass per vertex — **O(V·E)** (unweighted) — so they're intended for small-to-mid graphs; past ~100k nodes pass `betweenness({ pivots: k })` for a deterministic **approximate** run (Brandes from an evenly-spaced k-source sample scaled by `|V|/k`, O(pivots·E), still byte-identical; `pivots >= |V|` is exact). Reachable as `betweenness(config, graph)`, `g.betweenness(config)`, and `CALL betweenness() YIELD node, centrality` (same for `closeness`) — byte-identical across engines.

## Temporal values & the host clock

lenke stores ISO temporal values as first-class property values — `LocalDate`, `LocalTime`, `LocalDateTime`, `ZonedTime`, `ZonedDateTime`, `Duration` (all exported). Construct one with a `parse*` helper and store it like any value:

```ts
import { parseDate } from '@lenke/core';

vertex.setProperty('hired', parseDate('2020-01-15'));
```

Reading one back gives the same class, which bridges to your date library via `.toISOString()` / `.toTemporal()` (a TC39 `Temporal.Plain*`). A native `Date` is deliberately **not** auto-coerced (it's a zoned instant; the local types are zone-less) — pass an ISO string, a `Temporal.Plain*`, or `LocalDateTime.fromJSDate(d, { zone })`. Note that `JSON.stringify` on a result row renders a temporal cell as its tagged wire form (`{"@date":"2020-01-01"}`); map cells through `.toISOString()` (or `String(cell)`) when you want a plain ISO string in JSON output.

> Snapshot/round-trip helpers like `graphContentEqual` live in **`@lenke/serialization`** (alongside `serialize`/`deserialize`), not `@lenke/core`.

**Current time is host-injected.** The GQL now-functions (`current_date` / `current_timestamp`) read a clock you wire, keeping results deterministic by default. Wire wall time with `setClock` (chainable, returns the graph):

```ts
import { LocalDateTime } from '@lenke/core';

graph.setClock(() => LocalDateTime.fromJSDate(new Date(), { zone: 'utc' }));
```

With no clock wired (and no explicit `$__now` param) the now-functions read as `null` — the engine never invents a time. See [`@lenke/gql`](../gql) for the full temporal query surface (literals, constructors, arithmetic, ordering).

## License

Apache-2.0
