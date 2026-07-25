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

Everything is ESM under the \`@lenke/*\` scope. The lenke documentation has deeper guides for each build path.`,
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

On Node, use \`createNodeBackend()\` from \`@lenke/node/backend\` (no path needed — the prebuilt addon ships with it); in the browser, \`createWasmBackend(fetch(wasmUrl))\`. A native graph owns memory — release it with \`using\`, \`g.free()\`, or let GC back you up. The native and wasm setup docs cover where the compiled artifact lives.`,
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
RETURN path_length(p) AS hops, nodes(p) AS ns, edges(p) AS rels
\`\`\`
Path element access is 0-based: \`edges(p)[0].amount\`, \`nodes(p)[1].id\`.

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

const temporalGuide: Guide = {
  id: 'temporal',
  title: 'Working with time',
  description: 'Dates, datetimes, durations, time windows, and the current-time clock.',
  text: `# Working with time

lenke has first-class temporal types, all ISO-8601: dates, datetimes (zoneless and zoned), and durations.

## Constructors
\`\`\`
RETURN date('2024-06-01')                          -- a calendar date
RETURN datetime('2024-06-01T12:00:00')             -- a zoneless (local) datetime
RETURN zoned_datetime('2024-06-01T12:00:00Z')      -- a datetime carrying an offset / Z
RETURN duration('PT24H')                           -- 24 hours; also 'P1D', 'P1DT2H30M', 'PT90M'
\`\`\`
\`datetime()\` is **zoneless** — a timestamp that carries an offset or a \`Z\` parses with \`zoned_datetime()\` (passing such a string to \`datetime()\` yields \`null\`). Store timestamps consistently as one kind.

## Arithmetic and time windows
Shift an instant by adding a duration:
\`\`\`
RETURN datetime('2024-06-01T12:00:00') + duration('PT24H')   -- 2024-06-02T12:00:00
RETURN date('2024-06-01') + duration('P7D')                  -- 2024-06-08
\`\`\`
Express a window by comparing **instants** (this is the idiom for "within N hours/days"):
\`\`\`
MATCH (a)-[e1:SENT]->(b)-[e2:SENT]->(c)
WHERE e2.ts > e1.ts AND e2.ts < e1.ts + duration('PT24H')     -- e2 within 24h after e1
RETURN a, c
\`\`\`
Measure the gap between two instants:
\`\`\`
RETURN duration_between(e1.ts, e2.ts) AS gap
\`\`\`
Instants order directly (\`datetime < datetime\`, \`date < date\`). Compare durations *through the instants they bound* — write \`a.ts < b.ts + duration(...)\` rather than comparing one duration to another.

## Calendar/clock components
Extract a component with the named functions \`year(x)\`, \`month(x)\`, \`day(x)\`, \`hour(x)\`, \`minute(x)\`, \`second(x)\` — the ISO GQL form (not SQL \`EXTRACT\`, not a \`.year\` accessor). The argument must be a temporal that carries that component (\`year\`/\`month\`/\`day\` need a date; \`hour\`/\`minute\`/\`second\` need a time); a zoned value is read in its own offset. Bucket or cohort by a period with \`GROUP BY\`:
\`\`\`
MATCH (h:Hire) RETURN year(h.hired) AS yr, count(*) AS n GROUP BY yr ORDER BY yr
\`\`\`
A **string is not coerced** — \`year('2024-01-01')\` throws; wrap it first: \`year(date(h.day))\`.

## Current time
\`current_date()\` and \`current_timestamp()\` read a clock you provide — \`graph.setClock(() => Date.now())\` on a native graph, or pass \`$__now\` in the query params. Without a clock they read as \`null\`, which keeps queries deterministic by default.

## Good to know
- **Store timestamps as temporals, not strings.** A string compared against a temporal silently matches nothing. If your data has string timestamps, wrap them: \`datetime(e.ts)\`, \`date(e.day)\`.
- **Build temporal values with the constructor functions** \`date(x)\` / \`datetime(x)\` / \`zoned_datetime(x)\` / \`duration(x)\` (rather than \`CAST(x AS DATE)\`). The bare literal prefix is \`DATETIME '…'\`.
- **Date-part extraction is \`year()\`/\`month()\`/\`day()\`/\`hour()\`/\`minute()\`/\`second()\`** (see above), NOT SQL \`EXTRACT\` or a \`.year\` accessor. Passing a string throws — wrap with \`date()\`/\`datetime()\` first.
- \`min\`/\`max\` over a duration column work; a duration in a numeric \`sum\`/\`avg\` is a loud \`E_DATA_EXCEPTION\`. Out-of-range date fields (month 13) are a syntax error, but a valid-looking overflow like \`date('2025-02-29')\` rolls to \`2025-03-01\`.`,
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

Only query results (and NDJSON snapshots for persistence) cross the worker boundary — never the graph itself. The lenke local-first and wasm guides have the full recipe, including OPFS persistence.`,
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

Nesting joins the outer transaction (flat, no savepoints); the outermost commit runs deferred required/type/unique/cardinality/validator checks. The same surface exists on both the core \`Graph\` and native \`RustGraph\`.

## Constraints and transactions work together
- **Constraint checks defer to commit inside a transaction**, so a transaction that *transiently* violates a unique constraint and resolves it before commit is accepted — the invariant is the end state. This is how you swap two unique values or rebuild an index-backed set.
- **A minimum-cardinality constraint forces atomic creation.** With, say, \`Airport LOCATED_IN out 1..1\`, a bare \`INSERT (:Airport)\` fails (0 < min). Create the node and its required edge in one statement, or inside \`transaction(...)\`.
- **Constraints are a layer above the raw store.** \`createUniqueConstraint\` / \`createRequiredConstraint\` / \`createValidator\` guard the GQL write path; the underlying index itself permits duplicates. Declare them up front. Whole-invariant rules (e.g. debits == credits) are transaction invariants (\`createInvariant\`), not per-edge checks.
- **A vetoed \`addVertex\`/\`addEdge\` still returns an element object** — check for the violation explicitly (or use the GQL write path, which throws \`E_CONSTRAINT_VIOLATION\`); don't infer success from the return value.`,
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

const performanceGuide: Guide = {
  id: 'performance',
  title: 'Performance & scale',
  description:
    'Anchoring on indexes, cheap vs expensive operations, bounds, and the memory envelope.',
  text: `# Performance & scale

## Anchor traversals on an indexed key
The single biggest lever: seed a traversal from a specific vertex with an **inline pattern map** on an indexed property, not a post-hoc \`WHERE\`.
\`\`\`
MATCH (a:Acct {id: $id})-[:SENT]->{1,4}(b) RETURN b       -- seeks the index, then walks
\`\`\`
Writing \`MATCH (a:Acct)-[:SENT]->{1,4}(b) WHERE a.id = $id\` instead scans every \`Acct\` and expands before filtering — orders of magnitude slower on a large graph. Create the index first: \`g.createVertexIndex('id')\` (native \`RustGraph\`) or the same method on a core \`Graph\`. An indexed point lookup is dramatically faster than a scan.

## Cheap vs expensive
- **Cheap:** indexed seeks, \`degree\`, \`pagerank\`, grouped-count aggregations, and neighbor aggregation (linear in nodes × dims).
- **Expensive:** exact \`betweenness\` and \`closeness\` are O(V·E) and dominate at scale. Use \`pivots\` for approximate betweenness (a sampled estimate — lower ranking fidelity). Unbounded \`->*\` over all pairs, and unrolled fixed-length chains, grow fast in hop count.

## Bound the search, and let consumers stop early
Prefer a bounded quantifier (\`->{1,5}\`) to an open \`->*\` when you can. Consumers that don't need every match — \`EXISTS { … }\`, \`LIMIT\` — short-circuit a variable-length walk instead of enumerating it. A pattern that would enumerate an intractable number of paths faults \`E_RESOURCE_EXHAUSTED\` — that's the guard protecting you; add a tighter bound, anchor an endpoint, or add a \`LIMIT\`.

## Writes
Prefer one aggregate or bulk operation over many wide per-node \`SET\`s. For per-node feature vectors, \`neighbor_aggregate\` writes the whole block in one pass (see the \`graph-ml\` guide).

## Memory envelope
An in-memory graph takes **several times its NDJSON text size** in memory; a whole-graph algorithm roughly doubles peak memory over the resident graph. \`graphFromNdjson\` decodes in parallel and loads quickly. \`new Graph({ maxOperatorChain })\` bounds \`AND\`/\`OR\`/arithmetic operator chains (default 10,000) as an anti-DoS guard.`,
};

const recipesGuide: Guide = {
  id: 'recipes',
  title: 'Recipes: common graph patterns',
  description: 'Multi-hop chains, cycles, fan-in/structuring, and subgraph extraction.',
  text: `# Recipes: common graph patterns

## Multi-hop chains with per-hop conditions
Every hop meets a condition — put the \`WHERE\` inside the relationship bracket so it prunes each hop:
\`\`\`
MATCH p = (a:Acct {id: $id})-[e:SENT WHERE e.amount >= 1000]->{4,6}(b) RETURN nodes(p)
\`\`\`
For conditions relating **consecutive** hops (e.g. each hop within 24h of the prior, amount within 10%), compare adjacent path elements — \`edges(p)[i]\` against \`[i-1]\`:
\`\`\`
MATCH p = (a:Acct)-[:SENT]->{2,6}(b)
WHERE datetime(edges(p)[1].ts) < datetime(edges(p)[0].ts) + duration('PT24H')
  -- add one clause per index up to the max hop; guard the optional tail with IS NULL
RETURN p
\`\`\`
A tractable "value-preserving relay" motif (two hops within a window, amount preserved), which you can stitch into longer chains in host code:
\`\`\`
MATCH (a)-[e1:SENT]->(b)-[e2:SENT]->(c)
WHERE datetime(e2.ts) > datetime(e1.ts)
  AND datetime(e2.ts) < datetime(e1.ts) + duration('PT24H')
  AND abs(e2.amount - e1.amount) <= 0.1 * e1.amount
RETURN a.id, b.id, c.id
\`\`\`

## Cycles
On a dense graph, strongly-connected components collapse into one giant component, so filter by component **size** rather than treating membership as the signal:
\`\`\`
CALL strongly_connected_components() YIELD node, componentId
WITH componentId, count(*) AS size WHERE size > 1
RETURN componentId, size ORDER BY size DESC
\`\`\`
\`CALL on_cycle()\` gives per-vertex cycle membership. For money-cycle detection, anchor on time and amount as well as structure.

## Fan-in / structuring
Many transfers just under a threshold flowing into one account:
\`\`\`
MATCH (s)-[e:SENT]->(h) WHERE e.amount >= 8000 AND e.amount < 10000
RETURN h.id AS account, count(*) AS n, sum(e.amount) AS total ORDER BY n DESC
\`\`\`

## Subgraph extraction for explanation
Given a suspect pair, pull the actual edges via an indexed seek — instant, and it reads as a narrative:
\`\`\`
MATCH (a {id: $from})-[e:SENT]->(b {id: $to}) RETURN e.ts, e.amount ORDER BY e.ts
\`\`\``,
};

const graphMlGuide: Guide = {
  id: 'graph-ml',
  title: 'Graph machine learning',
  description: 'Feature engineering: message passing, structural features, and matrix egress.',
  text: `# Graph machine learning

lenke is a comfortable substrate for GNN-style feature engineering: propagate features across the graph, add structural signals, and egress a numeric matrix.

## 1. Pack features into a vector
Raw scalar features become one list property (stored efficiently, and it egresses as a real numeric matrix):
\`\`\`
MATCH (n:Account) SET n.h = [n.r0, n.r1, n.r2, n.r3]
\`\`\`

## 2. Message passing
Aggregate each node's neighbors' feature vector element-wise, over the whole block in one pass:
\`\`\`
CALL neighbor_aggregate({ feature: 'h', op: 'mean', direction: 'both', includeSelf: true, writeProperty: 'h1' })
YIELD node RETURN count(*)
\`\`\`
Iterate layers by alternating two buffers (\`h\` → \`h1\` → \`h2\`) rather than adding a new column per layer. If you write a neighbor-mean in plain GQL, use \`OPTIONAL MATCH\` + \`coalesce\` so degree-0 nodes aren't dropped:
\`\`\`
MATCH (n) OPTIONAL MATCH (n)-[]-(m) WITH n, avg(m.f) AS a SET n.h = coalesce(a, n.f)
\`\`\`

## 3. Structural features
Write algorithm results onto nodes, then read them as features:
\`\`\`
CALL pagerank() YIELD node, score SET node.pr = score
CALL degree({ direction: 'in' }) YIELD node, degree SET node.indeg = degree
\`\`\`
Component and label-propagation outputs are id **strings**, not numbers — use them as categories (group/compare on the string), don't coerce to a number. All numeric columns egress as Float64.

## 4. Egress the feature matrix
Hand the matrix to your training code as Apache Arrow (see the \`arrow\` guide): numbers come through as numbers, and a fixed-length numeric-list column egresses as a \`FixedSizeList<Float64>\` — a genuine numeric matrix.

## Cost
\`degree\` and \`pagerank\` are cheap; exact \`betweenness\`/\`closeness\` are O(V·E) and dominate at scale (use \`pivots\` for approximate betweenness). See the \`performance\` guide.`,
};

const gotchasGuide: Guide = {
  id: 'gotchas',
  title: 'Gotchas & footguns',
  description: 'Reserved-word aliases, temporal typing, categorical outputs, and Cypher-isms.',
  text: `# Gotchas & footguns

A few behaviors worth knowing — mostly around reserved words and silently-empty results.

## Reserved words can't be bare aliases or labels
Words like \`from\`, \`to\`, \`date\`, \`datetime\`, \`value\`, \`order\`, \`group\`, \`count\`, \`sum\`, \`path\`, \`match\` can't be a bare \`AS\` alias or label. Quote with backticks or rename:
\`\`\`
RETURN a.name AS \`from\`, b.name AS \`to\`     -- backticks
RETURN a.name AS knower, b.name AS known    -- or just rename
\`\`\`

## Temporal comparisons need real temporal types
A timestamp stored as a **string** compared against a temporal silently matches nothing. Wrap it, or store timestamps as datetimes:
\`\`\`
WHERE datetime(e.ts) < datetime(other.ts) + duration('PT24H')
\`\`\`
\`datetime()\` is zoneless — use \`zoned_datetime('…Z')\` for offset/Zulu strings. Two durations don't compare relationally; compare the instants they bound (\`a.ts < b.ts + duration(...)\`). See the \`temporal\` guide.

## Categorical algorithm outputs are strings
\`connected_components\` and \`label_propagation\` write a component/label **id string**. It egresses as text — \`Number()\` on it silently yields 0. Use it as a category.

## Isolated nodes drop from a naive aggregation
\`MATCH (n)-[]-(m) …\` silently skips degree-0 nodes. Use \`OPTIONAL MATCH\` + \`coalesce\` (see \`graph-ml\`).

## Procedure names are snake_case
\`CALL pagerank()\`, \`degree\`, \`connected_components\`, \`strongly_connected_components\`, \`label_propagation\`, \`neighbor_aggregate\`, \`betweenness\`, \`closeness\`, \`on_cycle\` — not camelCase.

## Grouping is by the non-aggregated RETURN items
An aggregate that appears **only** in \`ORDER BY\` doesn't create a grouping — alias it in \`RETURN\` too if you want per-group rows.

## Coming from Cypher
lenke rejects Cypher-isms rather than silently mis-running them: use \`OFFSET\` (not \`SKIP\`), \`-[:R]->{n,m}\` (not \`-[:R*n..m]\`), \`power(x, y)\` (not \`^\`), and a per-hop \`WHERE\` **inside** the relationship bracket. Durations are ISO-8601 strings.`,
};

const GUIDES: readonly Guide[] = [
  overview,
  gettingStarted,
  gqlGuide,
  temporalGuide,
  gremlinGuide,
  algorithmsGuide,
  recipesGuide,
  graphMlGuide,
  arrowGuide,
  syncGuide,
  workersGuide,
  transactionsGuide,
  typedNodesGuide,
  serializationGuide,
  performanceGuide,
  gotchasGuide,
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
