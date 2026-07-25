// How-to guides exposed as MCP resources. Each is plain Markdown an assistant can
// read to help you build with lenke — creating a graph, writing GQL or Gremlin,
// running algorithms, Arrow egress, multiplayer/sync, workers, and more.

import type { ResourceDef } from './protocol.js';

export type Guide = { id: string; title: string; description: string; text: string };

const overview: Guide = {
  id: 'overview',
  title: 'lenke overview',
  description: 'What lenke is, its engines and query frontends, and which to pick.',
  text: `# lenke

lenke is an embeddable labeled-property-graph database. You choose along three axes:

**Engine** — where the graph lives:
- \`@lenke/core\` — a pure-TypeScript in-memory graph. Zero native artifacts; runs anywhere JS runs. Great for tests, small/medium graphs, and the browser main thread.
- \`@lenke/native\` — bindings to the Rust columnar engine, over one of three backends: **FFI** (\`@lenke/native/ffi\`, Bun/server), **N-API** (\`@lenke/node\`, the fast Node path), or **WASM** (\`@lenke/native/wasm\`, browser). Use for large graphs and maximum throughput.

**Query frontend** — how you ask:
- **GQL** — ISO/IEC 39075 graph query language (\`@lenke/gql\`, or \`g.query(...)\` on a native graph).
- **Gremlin** — Apache TinkerPop-style traversals (\`@lenke/gremlin\`, or \`g.gremlin(...)\` on a native graph).

Both frontends run over either engine, and the results are the same. Pick one language per query; you don't mix them in a single query.

**Reach path** — main thread, a worker (local-first UI), or a server. See the \`workers\` and \`multiplayer-sync\` guides.

Everything is ESM under the \`@lenke/*\` scope. The docs/ folder has deeper guides (\`docs/guides/index.md\` is the map).`,
};

const gettingStarted: Guide = {
  id: 'getting-started',
  title: 'Getting started',
  description: 'Create a graph, load data, and run your first query.',
  text: `# Getting started

## Pure-TS graph (\`@lenke/core\`)

\`\`\`ts
import { Graph } from '@lenke/core';
import { query } from '@lenke/gql';

const g = new Graph();
const ada = g.addVertex({ labels: ['Person'], properties: { name: 'ada', age: 36 } });
const lin = g.addVertex({ labels: ['Person'], properties: { name: 'lin', age: 29 } });
g.addEdge({ from: ada, to: lin, labels: ['KNOWS'], properties: { since: 2020 } });

query(g, 'MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS knower, b.name AS known');
// [{ knower: 'ada', known: 'lin' }]
\`\`\`

Every edge needs at least one label, and both endpoints must already be in the graph.

## Load data from text

\`\`\`ts
import { deserialize } from '@lenke/serialization';
const g = deserialize(ndjsonText, 'ndjson');       // new graph from NDJSON
deserialize(moreText, 'ndjson', g);                // append into an existing graph
\`\`\`

Formats: \`ndjson\`, \`pg-json\`, \`pg-text\`, \`graphson\`, \`csv\`. NDJSON is one JSON object per line, e.g.
\`{"type":"node","id":"a","labels":["Person"],"properties":{"name":"ada"}}\` and
\`{"type":"edge","from":"a","to":"b","labels":["KNOWS"],"properties":{}}\`.

## Native graph (Rust engine)

\`\`\`ts
import { createFfiBackend } from '@lenke/native/ffi';   // Bun/server
import { graphFromNdjson } from '@lenke/native';

const backend = createFfiBackend(libPath);   // libPath = the compiled lenke native library
using g = graphFromNdjson(backend, await Bun.file('graph.ndjson').bytes());
g.query\`MATCH (a:Person) RETURN a.name\`;   // tagged-template form (safe binding)
\`\`\`

On Node, use \`createNodeBackend()\` from \`@lenke/node/backend\` (no path needed — the prebuilt addon ships with it); in the browser, \`createWasmBackend(fetch(wasmUrl))\`. A native graph owns memory — release it with \`using\`, \`g.free()\`, or let GC back you up. See \`docs/guides/native.md\` and \`docs/guides/wasm.md\` for the artifact paths.`,
};

const gqlGuide: Guide = {
  id: 'gql',
  title: 'Writing GQL',
  description:
    'ISO-GQL patterns: matching, variable-length paths, filters, projection, writes, temporal.',
  text: `# Writing GQL

lenke's GQL is ISO/IEC 39075. Run it with \`query(graph, text, params?)\` (\`@lenke/gql\`) or \`g.query(...)\` on a native graph.

## Match and return
\`\`\`
MATCH (a:Person)-[:KNOWS]->(b:Person)
WHERE a.age > 30
RETURN a.name AS name, b.name AS friend
ORDER BY name
LIMIT 10
\`\`\`
Labels are boolean expressions: \`(:A|B)\` (A or B), \`(:A&B)\`, \`%\` (any label). Undirected is \`~\`: \`(a)~[:KNOWS]~(b)\`. Comments start with \`--\` or \`//\`.

## Variable-length paths
Use a quantifier **after** the relationship:
\`\`\`
MATCH (a:Person)-[:KNOWS]->{1,3}(b)   RETURN b        -- 1 to 3 hops
MATCH (a)-[:KNOWS]->*(b)              RETURN b        -- 0 or more
MATCH (a)-[:KNOWS]->+(b)              RETURN b        -- 1 or more
\`\`\`
A **per-hop predicate** goes inside the bracket and filters every edge of the walk:
\`\`\`
MATCH (a:Acct)-[e:SENT WHERE e.amount >= 1000]->{1,5}(b) RETURN b
\`\`\`
**Bind the whole path** to a variable and read its pieces:
\`\`\`
MATCH p = (a:Acct)-[:SENT]->{2,6}(b)
RETURN path_length(p) AS hops, nodes(p) AS ns, relationships(p) AS rels
\`\`\`
Path element access is 0-based: \`relationships(p)[0].amount\`, \`nodes(p)[1].id\`.

**Path selectors and modes** pick which of many matching paths to keep:
\`\`\`
MATCH p = ANY SHORTEST (a)-[:SENT]->*(b) RETURN p       -- one shortest
MATCH p = ALL SHORTEST (a)-[:SENT]->*(b) RETURN p       -- every tied shortest
MATCH p = SHORTEST 3 (a)-[:SENT]->*(b) RETURN p         -- the 3 shortest
MATCH p = SIMPLE (a)-[:SENT]->{1,6}(a) RETURN p         -- no repeated node (cycles back to a)
\`\`\`
Modes: \`WALK\` / \`TRAIL\` (default) / \`SIMPLE\` / \`ACYCLIC\`.

## Aggregation
Grouping is implicit — the non-aggregated RETURN items are the group key:
\`\`\`
MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS name, count(*) AS friends
\`\`\`

## Parameters (injection-safe)
\`\`\`ts
query(g, 'MATCH (a:Person) WHERE a.name = $n RETURN a', { n: userInput });
gql(g)\`MATCH (a:Person) WHERE a.name = \${userInput} RETURN a\`;   // template = binding, not splicing
\`\`\`

## Temporal
Datetimes and durations are ISO-8601 strings:
\`\`\`
WHERE e2.ts < e1.ts + duration('PT24H')        -- within 24 hours
\`\`\`
\`datetime('2024-01-01T00:00:00')\` is zoneless; for a timestamp with an offset use \`zoned_datetime('2024-01-01T00:00:00Z')\`. \`date(...)\`, \`duration('P1D')\`, and \`duration_between(a, b)\` are available.

## Writes
\`\`\`
INSERT (:Person {name: 'ada', age: 36})
MATCH (a:Person {name: 'ada'}) SET a.age = 37
MATCH (a:Person {name: 'ada'}) REMOVE a.age            -- delete the property
MATCH (a:Person {name: 'ada'}) DETACH DELETE a
\`\`\`

## Coming from Cypher?
A few things differ: variable-length is \`-[:R]->{1,5}\` (not \`-[:R*1..5]\`); a per-hop condition goes *inside* the bracket (\`-[e:R WHERE …]->{1,5}\`); durations are ISO strings (\`duration('PT24H')\`, not a map); labels are boolean expressions. Use the \`gql_check\` tool to validate a query and \`gql_run\` to try it on sample data.`,
};

const gremlinGuide: Guide = {
  id: 'gremlin',
  title: 'Writing Gremlin',
  description: 'TinkerPop-style traversals: the TS fluent builder and the native string form.',
  text: `# Writing Gremlin

## TypeScript engine (fluent builder)
Compose a plan from step functions, then run it against a \`@lenke/core\` graph:
\`\`\`ts
import { V, out, has, gt, values, traversal, toArray } from '@lenke/gremlin';

const plan = traversal(V(), has('Person', 'age', gt(30)), out('KNOWS'), values('name'));
const names = toArray(plan, graph);          // eager; also toSet(plan, graph) / run(plan, graph)
\`\`\`
Sources: \`V(...ids)\`, \`E(...ids)\`, \`inject(...values)\`. Predicates: \`eq\`, \`gt\`, \`gte\`, \`lt\`, \`lte\`, \`neq\`, \`not\`, \`regex\`. Steps include \`out\`/\`in\`/\`both\`, \`outE\`/\`inE\`, \`has\`/\`hasLabel\`, \`where\`, \`select\`, \`project\`, \`order\`, \`dedupe\`, \`count\`, \`sum\`, \`union\`, \`branch\`, \`math\`, \`sack\`/\`withSack\`, and OLAP steps \`pageRank\`/\`connectedComponent\`/\`peerPressure\`. \`planToGremlin(plan)\` renders the Groovy text.

## Native engine (string form)
A native graph runs textual Gremlin and returns JSON-decoded results. Gremlin has no engine-side parameters, so the tagged-template form **is** the binding — each \`\${v}\` is escaped to a literal:
\`\`\`ts
g.gremlin\`g.V().has('name', \${userInput}).values('age')\`;
\`\`\`
For forwarding wrappers, compose safely with \`composeGremlin(query, ...subs)\` and \`escapeGremlin(value)\` from \`@lenke/native\`.

Pick GQL or Gremlin per query to taste — both read the same graph and return equivalent results.`,
};

const algorithmsGuide: Guide = {
  id: 'algorithms',
  title: 'Graph algorithms',
  description:
    'Run degree, PageRank, components, centrality, shortest path, and neighbor aggregation.',
  text: `# Graph algorithms

lenke ships in-engine algorithms with four call surfaces — all take the same config and give the same results.

**1. Free functions (\`@lenke/core\`)** — data-last, dual-form, async:
\`\`\`ts
import { pagerank, degree, shortestPath, neighborAggregate } from '@lenke/core';
const ranks = await pagerank({ iterations: 20, dampingFactor: 0.85 }, g);
const runner = pagerank({ iterations: 20 });   // curried: (graph) => Promise<Row[]>
\`\`\`

**2. Native methods** (\`RustGraph\`, run off-thread):
\`\`\`ts
const ranks = await g.pagerank({ iterations: 20, dampingFactor: 0.85 });
\`\`\`

**3. GQL \`CALL\`**:
\`\`\`
CALL pagerank() YIELD node, score RETURN node.name AS n ORDER BY score DESC LIMIT 10
CALL degree({ writeProperty: 'deg' }) YIELD node RETURN node       -- writes deg onto each vertex
CALL degree() RETURN node.name AS n, degree                        -- YIELD-less binds node + degree
\`\`\`

Algorithms: \`degree\`, \`pagerank\`, \`personalizedPagerank\`, \`connectedComponents\`, \`stronglyConnectedComponents\`, \`labelPropagation\`, \`peerPressure\`, \`betweenness\`, \`closeness\`, \`shortestPath\`, \`onCycle\`, \`neighborAggregate\`. Each yields \`{ node, <result> }\` rows in vertex order.

**Config** (all optional): \`direction\` ('out'|'in'|'both'), \`edgeLabel\`, \`weightProperty\`, \`dampingFactor\`, \`iterations\`, \`pivots\` (approximate betweenness), \`seedProperty\` (label propagation), \`source\`/\`target\` + \`algorithm\` ('dijkstra'|'astar') + \`heuristicProperty\` (shortest path), \`sourceNodes\` (personalized PageRank), \`writeProperty\` (write the result onto each vertex), and \`feature\` + \`op\` ('mean'|'sum'|'max'|'min') + \`includeSelf\` (neighborAggregate).

**neighborAggregate** is the message-passing / feature-propagation primitive: for each vertex it aggregates its neighbors' list-valued \`feature\` vector element-wise, in one pass — useful for GNN-style feature engineering.
\`\`\`
MATCH (n:Account) SET n.h = [n.r0, n.r1, n.r2]            -- pack scalar features into a vector
CALL neighbor_aggregate({ feature: 'h', op: 'mean', direction: 'both', writeProperty: 'h1' }) YIELD node RETURN node
\`\`\``,
};

const arrowGuide: Guide = {
  id: 'arrow',
  title: 'Arrow egress',
  description: 'Hand query results to DuckDB / Polars / pandas as Apache Arrow.',
  text: `# Arrow egress

A native graph can hand results out as Apache Arrow with no JSON round-trip.

\`\`\`ts
import { decodeArrow, toArrowIPC } from '@lenke/native/arrow';

// Lenke's compact ARW1 columnar blob (scalar columns: float64 / bool / utf8 / fixed-size-list):
const blob = g.queryArrow('MATCH (n:Person) RETURN n.name, n.age');

// Back to row objects in JS, no Arrow dependency:
const rows = decodeArrow<{ name: string; age: number }>(blob);

// Standard Apache Arrow IPC for other tools:
const stream = toArrowIPC(blob, 'stream');   // pyarrow.ipc.open_stream / polars.read_ipc_stream
const feather = toArrowIPC(blob, 'file');    // Feather v2: pandas.read_feather
\`\`\`

Or produce IPC directly in one native call:
\`\`\`ts
const ipc = g.queryArrowIpc('MATCH (n:Person) RETURN n.name, n.age', { format: 'stream' });
\`\`\`

Numeric columns come through as real numbers, and a fixed-length numeric-list column (e.g. a per-node feature vector) egresses as an Arrow \`FixedSizeList<Float64>\` — a genuine numeric matrix for downstream ML/analytics. Use \`query()\` for non-scalar projections you don't need in Arrow.`,
};

const syncGuide: Guide = {
  id: 'multiplayer-sync',
  title: 'Multiplayer / sync',
  description: 'Live queries, collaborative writes, and CDC with @lenke/sync.',
  text: `# Multiplayer / sync

\`@lenke/sync\` turns a graph into a live, collaborative store over any message channel (a Worker port, a WebSocket, …). The graph lives on one side (a host); UIs subscribe from the other (clients).

## Host side (holds the graph)
\`\`\`ts
import { createSyncEngine } from '@lenke/sync';
import { createStore, graphFromNdjson } from '@lenke/native';

const store = createStore(graphFromNdjson(backend, seedBytes));
const engine = createSyncEngine({ store, collections, upstream });   // demand-fill loaders + write queue
const host = engine.createHost({ send: (msg) => channel.post(msg) });
channel.onMessage((msg) => host.receive(msg));
\`\`\`

## Client side (a UI thread)
\`\`\`ts
import { createSyncClient } from '@lenke/sync';

const client = createSyncClient({ send: (msg) => channel.post(msg) });
channel.onMessage((msg) => client.receive(msg));

// A standing query that updates as data changes. \`deps\` names the labels/keys it
// depends on so the host knows when to recompute; it's useSyncExternalStore-ready.
const q = client.liveQuery('MATCH (p:Person) RETURN p.name AS name', { deps: ['Person', 'name'] });
q.subscribe(() => render(q.getSnapshot().rows));

// Writes replicate to everyone:
await client.mutate('INSERT (:Person {name: $n})', { n: 'ada' });
\`\`\`

## Change data capture (CDC)
See other clients' committed writes in order — the basis for presence, notifications, and derived state:
\`\`\`ts
const stop = client.subscribeWrites(
  (writes) => writes.forEach(applyLocally),
  { scopes: ['room-42'] },        // value-scope the stream (host must set a scopeKey)
);
\`\`\`
Writes carry the originating \`clientId\` for origin-skip and exactly-once dedupe; ordering is by a monotonic op-log cursor (a gap triggers a cold-boot resync). \`client.pushWrite(write)\` is a drop-in for forwarding a write upstream.

## Reconnection & persistence
\`createReconnectingClient(...)\` re-dials with backoff, re-subscribes, and parks writes while offline. Snapshot the store to OPFS/memory with \`createSnapshotStore\` + \`encodeSnapshot\`/\`graphFromSnapshot\`. What crosses the channel is query results and writes — not the whole graph.

A common shape: the graph + engine + host in a Web Worker, the client + \`liveQuery\` on the UI thread. See the \`workers\` guide.`,
};

const workersGuide: Guide = {
  id: 'workers',
  title: 'Running in a worker',
  description: 'Keep the graph off the UI thread for a responsive local-first app.',
  text: `# Running in a worker

For a local-first UI, put the graph in a Web Worker so queries never block rendering. The worker holds the store + sync host; the UI holds a sync client.

## Worker (\`graph.worker.ts\`)
\`\`\`ts
import { createWasmBackend, createStore, graphFromNdjson } from '@lenke/native';
import { createSyncEngine } from '@lenke/sync';

const backend = await createWasmBackend(fetch(wasmUrl));   // wasmUrl → the lenke wasm engine
const store = createStore(graphFromNdjson(backend, seedBytes));
const engine = createSyncEngine({ store, collections, upstream });
const host = engine.createHost({ send: (m) => self.postMessage(m) });
self.onmessage = (e) => host.receive(e.data);
\`\`\`

## UI thread
\`\`\`ts
import { createSyncClient } from '@lenke/sync';

const worker = new Worker(new URL('./graph.worker.ts', import.meta.url), { type: 'module' });
const client = createSyncClient({ send: (m) => worker.postMessage(m) });
worker.onmessage = (e) => client.receive(e.data);

const people = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
\`\`\`

Only query results (and NDJSON snapshots for persistence) cross the worker boundary — never the graph itself. \`docs/guides/frontend-worker.md\` has the full recipe, including OPFS persistence; \`docs/guides/wasm.md\` covers loading the wasm engine.`,
};

const transactionsGuide: Guide = {
  id: 'transactions',
  title: 'Transactions',
  description: 'Atomic multi-statement writes with deferred constraint checks.',
  text: `# Transactions

Group writes so they commit together or not at all. Reads inside the transaction see its own writes; a thrown error or a failed constraint rolls the whole batch back.

\`\`\`ts
g.transaction((tx) => {
  tx.query('INSERT (:Acct {id: 1, balance: 100})');
  tx.query('INSERT (:Acct {id: 2, balance: -100})');
  // any deferred invariant (e.g. sum(balance) = 0) is checked at commit;
  // on violation the whole transaction rolls back and throws.
});
\`\`\`

There's also a TinkerPop-style explicit handle:
\`\`\`ts
const t = g.tx();
try { /* … writes … */ t.commit(); } catch (e) { t.rollback(); throw e; }
\`\`\`

Nesting joins the outer transaction (flat, no savepoints); the outermost commit runs deferred required/type/unique/cardinality/validator checks. The same surface exists on both the core \`Graph\` and native \`RustGraph\`. See \`docs/design/r-tx.md\`.`,
};

const typedNodesGuide: Guide = {
  id: 'typed-nodes',
  title: 'Typed nodes & schema',
  description: 'Validate-before-write with any Standard Schema (Zod / Valibot / ArkType).',
  text: `# Typed nodes & schema

\`defineNode\` / \`defineEdge\` validate and coerce input against a [Standard Schema](https://standardschema.dev) (Zod ≥3.24, Valibot, ArkType, …) before writing. The validated **output** is what gets stored.

\`\`\`ts
import { defineNode, defineEdge } from '@lenke/core';
import { z } from 'zod';

const User = defineNode('User', z.object({ name: z.string(), age: z.number().int().optional() }));
const ada = await User.create(graph, { name: 'ada', age: 36 });   // validated, then stored

const Follows = defineEdge('FOLLOWS', z.object({ since: z.number() }));
await Follows.create(graph, ada.id, lin.id, { since: 2020 });     // from/to are vertex ids
\`\`\`

\`create\` and \`parse\` are async (schemas may validate asynchronously). This is a host-side guard on \`create\` — it doesn't police raw GQL writes or bulk NDJSON loads. For in-engine enforcement that guards *every* write (both engines), use the constraint surface instead: \`createTypeConstraint\`, \`createRequiredConstraint\`, \`createUniqueConstraint\`, \`createValidator\`.`,
};

const serializationGuide: Guide = {
  id: 'serialization',
  title: 'Serialization',
  description: 'Read and write graphs as NDJSON, pg-json, GraphSON, and CSV.',
  text: `# Serialization

\`@lenke/serialization\` converts a \`@lenke/core\` graph to and from text.

\`\`\`ts
import { serialize, deserialize } from '@lenke/serialization';

const text = serialize(g, 'ndjson');
const g2 = deserialize(text, 'graphson');          // fresh graph
deserialize(moreText, 'ndjson', g2);               // append into an existing graph
\`\`\`

Formats: \`ndjson\`, \`pg-json\`, \`pg-text\`, \`graphson\`, \`csv\`. For large inputs use \`serializeStream\` / \`deserializeStream\`, or the non-blocking \`serializeAsync\` / \`deserializeAsync\`.

On a native graph the equivalents are \`g.serialize(format)\`, \`g.toNdjson()\`, and \`g.mergeNdjson(bytes)\` (bulk-append, like \`deserialize(bytes, 'ndjson', existing)\`).`,
};

const GUIDES: readonly Guide[] = [
  overview,
  gettingStarted,
  gqlGuide,
  gremlinGuide,
  algorithmsGuide,
  arrowGuide,
  syncGuide,
  workersGuide,
  transactionsGuide,
  typedNodesGuide,
  serializationGuide,
];

/** The guides as MCP resources, addressed \`lenke://guide/<id>\`. */
export const RESOURCES: readonly ResourceDef[] = GUIDES.map((guide) => ({
  uri: `lenke://guide/${guide.id}`,
  name: guide.title,
  description: guide.description,
  mimeType: 'text/markdown',
  read: () => guide.text,
}));

export { GUIDES };
