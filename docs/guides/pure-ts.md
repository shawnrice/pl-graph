# The pure-TypeScript graph

**Engine:** `@lenke/core` · **Frontend:** `@lenke/gql` or `@lenke/gremlin` · **Runtime:** anywhere JS runs.

Use this when you want a graph with zero native artifacts — a browser tab, a Node/Deno/Bun process, an edge function — and your data is small-to-medium or your queries are light. It's the simplest deployment: `npm install`, `new Graph()`, done. For large data or query-heavy workloads, reach for the Rust engine ([native](./native.md) / [wasm](./wasm.md)) instead.

## The graph

`@lenke/core` is the substrate — a mutable labeled-property graph you drive with method calls. It has no query language of its own; that's the [query frontend](./choosing-your-build.md#axis-2--the-query-frontend) you add.

```ts
import { Graph } from '@lenke/core';

const g = new Graph();

// Mutate
const marko = g.addVertex({ labels: ['Person'], properties: { name: 'marko', age: 29 } });
const josh = g.addVertex({ labels: ['Person'], properties: { name: 'josh', age: 32 } });
g.addEdge({ from: marko, to: josh, labels: ['KNOWS'], properties: {} });

// Look up
g.getVertexById(marko.id);
g.getVerticesByLabel('Person'); // a Set

// Opt-in secondary index for property lookups / ranges
g.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
g.getVerticesByProperty('age', 29);
g.getVerticesByPropertyRange('age', { gte: 30 });
```

It's an ordinary GC-managed object — nothing to free. `g.truncate()` empties it (keeping declared indexes); `g.clone()` makes an independent deep copy.

## Querying it

Bring the frontend your shop standardizes on — install **one**, tree-shake the other.

### GQL — [`@lenke/gql`](../../packages/gql)

```ts
import { query } from '@lenke/gql';

const rows = query(g, `MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name AS friend`);
// => [{ friend: 'josh' }]

// Bind values as params — never splice user input into the text:
query(g, `MATCH (p:Person) WHERE p.name = $name RETURN p.age`, { name: 'marko' });
```

Use `prepare(text)` to compile a plan once and `execute` it repeatedly.

### Gremlin — [`@lenke/gremlin`](../../packages/gremlin)

```ts
import { traversal, V, out, values } from '@lenke/gremlin';

const friends = g.toArray(traversal(V(marko.id), out('KNOWS'), values('name')));
// => ['josh']
```

Both run over the _same_ `Graph`; pick the one language you use.

## Loading data

[`@lenke/serialization`](../../packages/serialization) reads and writes a `Graph` in five codecs — `pg-json`, `pg-text`, `ndjson`, `graphson`, `csv`.

```ts
import { deserialize, serialize, deserializeStream } from '@lenke/serialization';

const g = new Graph();
deserialize(text, 'ndjson', g); // parse a whole document into g (mutates in place)

// Stream a large source of truth in, bounded memory:
await deserializeStream(fileStream, 'ndjson', g); // fileStream: AsyncIterable<string | Uint8Array>

// Persist / round-trip:
const dump = serialize(g, 'ndjson');
```

A codec speaks the core `Graph` API directly and preserves element ids, so `deserialize(serialize(g)) ` reconstructs the same graph.

## Reacting to changes

The graph exposes the same reactive signal the Rust engine does, so the React bindings ([frontend-main-thread](./frontend-main-thread.md)) work over it:

- `g.subscribe(fn)` — coalesced change notifications (returns an unsubscribe).
- `g.version` — monotonic "did anything change?" counter.
- `g.epoch(name)` — per-label / edge-type / property-key counter for fine-grained invalidation.

## When to move off it

The TS engine has no columnar scans, no Arrow output, and no compiled query engine. If you hit throughput limits on large graphs or heavy queries, the same data model is available on the Rust core — switch the engine (and, in React, `GraphProvider` → `StoreProvider`) without changing your query language. See [choosing-your-build](./choosing-your-build.md).
