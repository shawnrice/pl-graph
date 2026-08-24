# @lenke/sync

> The lenke sync engine: the v1 live-query wire protocol and the transport-agnostic host that serves it.

The frontend asks **declaratively** — the primitive is a standing query, not a fetch. All messages are tagged plain data, so the same protocol rides any port-shaped channel: a Worker `postMessage` port in the browser, a WebSocket to a server. That symmetry is the design's core claim: **a WebSocket is structurally a port**, so the server-embedded host and the browser worker host are one implementation.

## Protocol v1 (~6 messages)

```
client → host:  subscribe   { sub, query, deps, params?, key?, lang?, window? }
host → client:  rows        { sub, rows | (patch, remove, order) | values, version, complete }
client → host:  unsubscribe { sub }
client → host:  query       { req, query, params?, lang?, format? }   // one-shot (gql | gremlin)
host → client:  result      { req, rows | values | arrow | error }
client → host:  mutate      { req, text, lang?, params? }             // gql | gremlin
host → client:  ack         { req, ok, error? }                // UI effect arrives via rows pushes
host → client:  status      { connected, pendingWrites, protocol }
```

`params` is a flat object of `$name` bindings, on every GQL-carrying message. Values bind engine-side to already-parsed param slots and **never touch the GQL parser** — send values as params; never build query text from user input. `lang: 'gremlin'` runs the text through the Gremlin engine instead (results ride `values`); Gremlin has no param binding, so interpolate values with the `gremlin` tag / `escapeGremlin`.

Conformance is **structural**: consumers may write these shapes down independently, with no dependency in either direction. Two backward-compatible extensions ride the same protocol: **keyed row diffs** (declare `key` on subscribe → `patch`/`remove`/`order` pushes instead of full rows) and **Arrow one-shots** (`format: 'arrow'` over a binary transport). Resumable subscriptions and Arrow on the push path are out of v1's scope.

## The host

`createSyncHost(store, { send })` attaches one client connection to a `Store` (from `@lenke/native`). You hand it a `send` function and feed inbound messages to `receive` — which is the shape of every transport:

> **Bare host vs. engine host.** `createSyncHost(store, …)` alone serves a **complete, local-only** store — every subscription is trivially `complete: true`, no demand-fill. For per-collection completeness and demand-fill you want `engine.createHost(…)` (see [The sync engine](#the-sync-engine) below), which wires the same host into a `createSyncEngine`. Reach for the bare host only when the store already holds all the data.

```ts
// Server — WebSocket (Bun.serve shown; `ws` on Node is the same three lines):
Bun.serve({
  fetch: (req, srv) => (srv.upgrade(req) ? undefined : new Response(null, { status: 400 })),
  websocket: {
    open(ws) {
      hosts.set(ws, createSyncHost(store, { send: (m) => ws.send(JSON.stringify(m)) }));
    },
    message(ws, raw) {
      hosts.get(ws)?.receive(JSON.parse(String(raw)));
    },
    close(ws) {
      hosts.get(ws)?.close();
      hosts.delete(ws);
    },
  },
});

// Browser — Worker (the identical host, different port):
const host = createSyncHost(store, { send: (m) => self.postMessage(m) });
self.onmessage = (e) => host.receive(e.data);
```

Change routing is epoch-driven: any write through `store.mutate` — this connection's, another connection's on the same store, or a future CDC ingest — bumps the graph version; each subscription's epoch-gated snapshot recomputes only if its dependency tokens moved; a push goes out only when the snapshot reference actually changed. The write path never knows subscriptions exist.

## The client

`createSyncClient({ send })` is `liveQuery`'s port-crossing shadow — the registry the UI consumes. Same transport seam as the host:

```ts
const client = createSyncClient({ send: (m) => ws.send(JSON.stringify(m)) });
ws.onmessage = (e) => client.receive(JSON.parse(String(e.data)));

// A standing query. N consumers of the same (query, params, deps) share ONE
// wire subscription; the wire teardown is refcounted. The dedupe signature
// normalizes formatting (whitespace/comments; values untouched) and treats
// deps as a set — but never folds case (labels/properties are case-sensitive).
// `deps` is REQUIRED (the client does no inference): the label / edge-type /
// property-key tokens whose epochs re-run this query. Pass `null` to recompute
// on every change, or derive it with `inferDeps(query)` (re-exported here). It's
// also load-bearing for demand-fill — the collection is keyed off these tokens.
const live = client.liveQuery('MATCH (p:Person) WHERE p.age >= $min RETURN p.name', {
  deps: ['Person', 'name'],
  params: { min: 18 },
});

// useSyncExternalStore-ready (no React dependency here):
const { rows, complete, error } = useSyncExternalStore(live.subscribe, live.getSnapshot);
// `complete` is false until the host answers — render skeletons, not lies.

await client.mutate('INSERT (:Person {name: $n})', { n: 'zoe' }); // resolves on ack
await client.mutateGremlin`g.addV('Person').property('name', ${name})`; // Gremlin write, values escaped
const rows = await client.query('MATCH (p:Person) RETURN p.name'); // one-shot
const vals = await client.gremlin`g.V().has('name', ${name}).values('age')`; // one-shot Gremlin
```

Snapshots are referentially stable between pushes. A handle whose refcount hits zero tears down its wire subscription and retires into a bounded LRU (`maxInactiveQueries`, default 64) — a quick re-subscribe revives it warm with a fresh wire sub (React StrictMode's mount dance is safe), while an entry evicted past the cap simply re-subscribes cold on next use (a re-query against the host's store, not a refetch). Mutation effects arrive through subscription pushes, exactly as if another client had written.

## The sync loop

`createSyncEngine` is the worker-side machinery between the local store and the network — one producer and one mechanism per arrow:

```
frontend declares interest      →  host onSubscribe fires ensure(deps)
worker fills what that implies  →  collection loaders write into the graph
server pushes what changed      →  engine.ingest(writes); epochs route
local writes go back up         →  engine.mutate: optimistic + FIFO queue w/ backoff
```

```ts
const engine = createSyncEngine({
  store,
  collections: {
    // A collection = an app scope + the labels it covers + how to load it.
    // Demand-fill needs no protocol addition: a subscription's deps already
    // name the labels it reads, so intersecting collections load on demand.
    people: {
      labels: ['Person', 'REPORTS_TO'],
      load: async () => {
        const res = await fetch('/api/people').then((r) => r.json());
        return res.map((p) => ({
          text: 'INSERT (:Person {name: $n, age: $a})',
          params: { n: p.name, a: p.age },
        }));
      },
    },
    // A KEYED collection tracks completeness + demand-fills per distinct VALUE
    // of `key`, read straight off the subscription's params. A subscription for
    // `WHERE t.proj = $proj` with `params: { proj: 'apollo' }` fills only the
    // apollo scope; a different `$proj` is a separate fill. `key` (a value
    // partition) is distinct from `deps` (label intersection) — you need both.
    tasks: {
      labels: ['Task'],
      key: 'proj',
      load: async ({ proj }) => {
        const res = await fetch(`/api/tasks?proj=${proj}`).then((r) => r.json());
        return res.map((t) => ({
          text: 'INSERT (:Task {id: $id, proj: $p})',
          params: { id: t.id, p: proj },
        }));
      },
    },
  },
  initiallyComplete: ['people'], // when the boot snapshot already covers it
  upstream: { push: (w) => api.mutate(w.text, w.params, w.lang) }, // write-back target — forward the lang!
  retry: { attempts: 5, baseMs: 250 },
  onWriteError: (w, e) => report(w, e),
});

// One wired host per client connection:
const host = engine.createHost({ send: (m) => ws.send(JSON.stringify(m)) });
```

Semantics worth knowing:

- **Answer now, fill after** — a subscription over an unloaded collection gets its local (possibly stale) rows immediately with `complete: false`; the loader's writes land in one `store.mutate`, epochs route the push, and `complete` flips. An **empty scope still flips** `complete` (same rows, new truth).
- **A failed load surfaces as a retryable error, not a forever-skeleton** — if a loader throws, the subscribing client's snapshot gets `{ complete: false, error }` (the subscription stays alive; the next demand re-attempts and clears it), and the worker-side `onLoadError(collection, err)` also fires. The wire `error.code` is whatever the loader threw — throw a `LenkeError` (coded) from your loader if you want a meaningful `code`, else it's `'Unknown'` with the thrown message.
- **Loaders return writes** (`SyncWrite[]` — `{ text, params? }` for GQL, `{ text, lang: 'gremlin' }` for a Gremlin traversal), not graphs — values stay on the params path, and the engine stays ignorant of your fetch/decode shape.
- **Write-back is optimistic** — local readers see a mutation before upstream answers; the queue is FIFO, one in flight, exponential backoff; a write that exhausted its retries is dropped and reported (`onWriteError`) — rollback-and-correct is out of v1's scope (it needs server cursors). A mutation that changed nothing replicates nothing (version-gated).
- **`ingest` never echoes** — server-pushed writes apply locally and route by epoch, with no path back into the queue.

## Snapshots (warm boot)

The load-bearing rule: the local snapshot is **warmth, never truth** — every failure mode (eviction, tamper, wrong key, version/user mismatch, corruption) reads as _absent_ and the app cold-boots. The one exception is the **pending-write queue** (truth-on-client until acked), which rides inside the snapshot.

`createSnapshotStore` is the enforced path — the **key decides where the snapshot lives**, so plaintext can't reach disk:

```ts
const key = await importSnapshotKey(rawKeyFromAuth); // worker memory only, never persisted
// key present → AES-GCM-sealed + durable on OPFS; key omitted → memory-only,
// NEVER written to disk (warm within a SharedWorker's life; gone on its exit).
const snapshots = createSnapshotStore({ filename: 'lenke.snapshot', key });

// Save (on an interval, on visibilitychange — the app decides):
await snapshots.save(engine.store, {
  schemaVersion: 'v3',
  userId: session.userId,
  serverCursor: stream.cursor, // resume point for the app's sync stream
  collections: ['people'], // scopes this snapshot covers
  pendingWrites: engine.queuedWrites(), // unsynced changes survive the reload
});

// Boot:
const snap = await snapshots.load({ schemaVersion: 'v3', userId: session.userId });
const store = createStore(
  snap
    ? graphFromSnapshot(backend, snap) // warm: restores data AND schema (constraints/indexes)
    : createEmptyGraph(backend),
); // cold: demand-fill does the rest
const engine = createSyncEngine({
  store,
  collections,
  initiallyComplete: snap?.header.collections ?? [],
  initialWrites: snap?.pendingWrites ?? [], // stranded writes resume replication immediately
  upstream,
});
// resume the push stream from snap?.header.serverCursor ?? '' — a cursor-too-old
// answer means: snapshots.clear(), cold boot.
```

**Storage sink (browser vs. server).** `createSnapshotStore({ filename, key })` defaults its durable sink to `opfsStorage(filename)`, which throws off-browser. On Node/Bun pass a `durable` override — `SnapshotStorage` is just `{ read, write, delete }` over bytes:

```ts
import { readFile, writeFile, rm } from 'node:fs/promises';

const fileStorage = (path: string): SnapshotStorage => ({
  read: () => readFile(path).catch(() => null),
  write: (bytes) => writeFile(path, bytes),
  delete: () => rm(path, { force: true }),
});
const snapshots = createSnapshotStore({
  filename: 'ignored',
  key,
  durable: fileStorage('./state.lnks'),
});
```

Encryption is **secure-by-default and structural**: `createSnapshotStore` writes durably only when it holds a key; without one the snapshot stays in memory and never touches disk. And the durable sink enforces this on its own — `opfsStorage.write` **refuses an unencrypted snapshot** (a one-byte crypto flag in the format lets it check without a key), so plaintext can't reach disk even through the raw primitives. The lower-level `encodeSnapshot`/`decodeSnapshot`/`readSnapshot` each still take a required, explicit `{ key }` or `{ unencrypted: true }` — there's no "pass nothing" path, because an unencrypted snapshot carries no authentication at all.

Format: a **plaintext header** `{ formatVersion, schemaVersion, userId, serverCursor, collections }` — the invalidation tier, checked before any decryption (`peekHeader` reads it without the key) — then a gzip payload of three sections: the graph **NDJSON** (data), the **pending-write queue**, and the **schema** (constraints / validators / invariants / indexes, read straight from the engine via `dumpSchema` — data NDJSON can't carry them, and the engine is the one source of truth so they never drift). NDJSON stays pure data; schema rides its own section, which is why `graphFromSnapshot` (data **and** schema) is the complete warm-boot path where a bare `graphFromNdjson` would silently drop constraints. The payload is optionally AES-GCM-sealed (compress-then-encrypt; authenticated, so tamper — payload _or_ the header, bound in as AEAD `additionalData` — reads as absent; fresh IV per save; revocation = drop the key — crypto-shredding). `readSnapshot`/`load` **delete** a snapshot they can't decode on the way out — and this is intentional: a wrong key or a tampered/authentication-failing payload is treated as "invalid, drop it" (a security property — a snapshot you can't open is a snapshot you evict, not keep). So do NOT probe a key against your live snapshot to "test" it — a failed load destroys it. Use `peekHeader` (reads the plaintext header without the key) for a non-destructive check. Logout should still answer with `Clear-Site-Data: "storage"` — the browser wipes the origin without trusting app code to clean up.

## Multiplayer: the CDC write stream

Live queries push **rows** — the _result_ of a query, recomputed and re-sent when the store changes. That's enough for one client reading its own store, but for **multiplayer** — many clients, each with an optimistic local engine, that must see _each other's_ writes — you want the writes themselves, not re-derived rows. That's the CDC (change-data-capture) write stream.

**Topology.** One authoritative server: a shared `Store` and a shared `WriteLog`, with one `createSyncHost(store, { …, writeLog })` per connection. Each client runs a local `createSyncEngine` (its own optimistic store) and a `createSyncClient` over the socket.

```ts
// server: one store, one op log, a host per socket
const store = createStore(/* … */);
const writeLog = createWriteLog();
onConnection((socket) => {
  const host = createSyncHost(store, { send: (m) => socket.send(m), writeLog });
  socket.onMessage((m) => host.receive(m));
});

// client: pipe the write stream into the local optimistic engine
client.subscribeWrites((writes) => engine.ingest(writes), {
  onResync: () => coldBootFromSnapshot(),
});
```

**A complete in-process fan-out (N replicas, no sockets).** The transport is just a pair of `send` callbacks, so you can wire host↔client directly in one process — useful for tests and for seeing the whole loop at once. One authoritative store + one `WriteLog`; each client gets its own host and its own local store; a write on any client appears on every other. This example is runnable as-is:

```ts
import { createStore, graphFromNdjson, type Store } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';
import { createSyncClient, createSyncHost, createWriteLog, type SyncClient } from '@lenke/sync';

const backend = createFfiEngineBackend('/path/to/liblenke_engine.so'); // your built cdylib
const emptyStore = (): Store =>
  createStore(graphFromNdjson(backend, new TextEncoder().encode('\n'))); // one newline = empty

// Authoritative server: one shared Store + one shared WriteLog.
const server = emptyStore();
const writeLog = createWriteLog();

// Attach one client connection: its own host (on the shared store + log) and its
// own local optimistic store, fed by the CDC stream. No wire liveQuery here — a
// pure-CDC replica, so it receives ALL writes (interest routing doesn't narrow it).
function connect() {
  const local = emptyStore();
  const link: { c?: SyncClient } = {};
  const host = createSyncHost(server, { send: (m) => link.c?.receive(m), writeLog });
  const client = createSyncClient({ send: (m) => host.receive(m) });
  link.c = client;
  client.subscribeWrites((writes) => {
    for (const w of writes) local.mutate((g) => g.query(w.text, w.params));
  });
  return { local, client };
}

const a = connect();
const b = connect();
const c = connect();

await a.client.mutate('INSERT (:Widget {n: $n})', { n: 1 }); // A writes
await new Promise((r) => setTimeout(r, 10)); // let the fan-out flush

const count = (s: Store) =>
  s.mutate((g) => g.query<{ c: number }>('MATCH (n:Widget) RETURN count(*) AS c'))[0].c;
console.log(count(server), count(b.local), count(c.local)); // → 1 1 1
```

B and C never subscribed to a live query, yet both received A's `Widget` insert — the pure-CDC path. (Swap the direct `send` callbacks for `ws.send`/`ws.onmessage` and it's a real server, unchanged.)

**How it flows.** When any client mutates, its host commits to the shared store _and_ appends the write to the `WriteLog` — statement-based replication: the op is the `SyncWrite` (write text + resolved params), replayed through `runWrite`, deterministic because the two engines are byte-identical. Every _other_ stream subscriber receives it and `ingest`s it into its local store, so a change appears everywhere without re-querying. The writer never gets its own echo (origin-skip) — it already applied it optimistically.

**Ordering & resume.** Each op carries a monotonic `seq`. `subscribeWrites` sends `{ since: writeCursor }`; the client tracks the last-applied seq and resumes from it on reconnect (`replay()` re-subscribes automatically). If the client has fallen off the `WriteLog`'s bounded tail (a long disconnect), the host answers `resync` and the client cold-boots from a snapshot.

> **The live tail assumes in-order delivery.** While connected, each streamed op is `ingest`ed **blindly, in arrival order** — the live path does NOT check that `seq` is contiguous. This is sound over a real WebSocket/TCP transport, which is FIFO per connection, so ops arrive in the order the host appended them. But if you wire `subscribeWrites` over an **unordered or lossy** transport (UDP, a multiplexed channel that can reorder, a fan-out that drops), an out-of-order or gapped op is applied anyway and the replica can **silently diverge** — statement-based replication is order-sensitive (a later `SET` applied before the `INSERT` it depends on produces different state). Only the **reconnect** path gap-detects (via `since`/`writeCursor` → `resync`); the steady-state tail does not. Keep the stream on an in-order transport, or add your own seq-contiguity check before `ingest` if you can't guarantee one.

**v1 catch-up is full-replay — there is no snapshot+cursor bootstrap.** The host answers `since` with `writeLog.since(msg.since ?? 0)`, and `since(0)` means _from the very start of the retained tail_, not "from head". A **fresh** client boots with `writeCursor = 0`, so its first `subscribeWrites` replays the **entire retained backlog** (up to the `WriteLog`'s `capacity`, default 1024 ops) before it goes live. That is the intended v1 behavior: a new replica reconstructs state by replaying the op log, statement by statement — there is no "load a snapshot, then resume from its cursor" fast path yet (a snapshot's `serverCursor` belongs to the _app's own_ sync stream, not this op log). Two implications: (1) size `capacity` for your worst-case cold client, since anything older than the tail is unrecoverable via CDC (it answers `resync` → cold-boot from an app snapshot); (2) the `SubscribeWritesMessage` JSDoc phrase "omit (or `0`) to start from the current head" describes the _intended_ opt-out, but the shipped `since(0)` path replays from the start — pass an explicit non-zero cursor (`writeLog.head()`, surfaced by a prior session) if you truly want head-only. A returning client that carries a real `writeCursor` resumes from exactly there (exclusive), so it never re-applies or double-counts.

**Cost & interest routing.** A write fans out as **O(N)** shared-payload sends (each of N clients must hear about it), not the O(N²) of re-running every client's subscription query per write. **Interest routing** narrows it further: the host forwards a write to a client only if the write's tokens — labels / edge-types / property keys, via `inferDeps` — intersect that client's **wire** live-query `deps`, so a client watching only `:User` never receives a `:Resource` write. A client with no _wire_ subscriptions (or a `deps: null` "recompute on everything" one) gets all writes; a Gremlin write (no inferable tokens) is forwarded to all. Routing is by **label/token, not row value**: a homogeneous-label app (every node a `:Shape`) still fans a shape-write to every shape-watcher — row/viewport-level routing would reintroduce the per-subscription evaluation the op-stream exists to avoid. Skipped writes still tick the client's cursor (an empty batch), so resume never spuriously resyncs. Idempotent writes (`_MERGE`) make the at-least-once cases replay-safe.

**Two `liveQuery` surfaces — and only one drives routing.** They are named alike but sit on opposite sides of the wire:

- **`store.liveQuery`** (from `@lenke/native`'s `createStore`) — a standing query over the client's **local replica**. It's a purely in-process reactive read; it registers **no host interest** and sends nothing over the socket. This is what your UI binds to for the fast, optimistic local view.
- **`client.liveQuery`** (from `createSyncClient`) — a standing query over the **wire**. _This_ is what emits a `subscribe` to the host, and its `deps` are the only thing interest routing narrows on.

The consequence is easy to trip over: **a pure-CDC replica — one that ingests the write stream but registers no `client.liveQuery` — has no wire `deps`, so the host routes it ALL writes** (same as `deps: null`). That's usually what you want (a replica reconstructing the whole graph from the op stream must see every write), but it means interest routing does _not_ filter a CDC-only client. To narrow what a client receives, it must hold a wire `client.liveQuery` (or wire `subscribe`) whose `deps` name the tokens it cares about — a _local_ `store.liveQuery` will not narrow the stream, because the host never learns about it.

### Schema replication — constraints ride the same log

Constraints, validators, invariants, and indexes replicate over the **same op stream** as data. A schema declaration is set up through the programmatic API (`g.createUniqueConstraint(…)`, `g.createValidator(…)`, `g.createIndex({ on: 'vertex', kind: 'hash', keys: […] })`), not as write-language text, so it rides the log as a **structured `SchemaOp`** rather than a `text` statement — and `runWrite` replays it on every replica (`runWrite` → `applySchemaOp`), keeping schema in lock-step. This is what lets a replica derive a keyed `_MERGE` (which needs the target label's unique constraint to resolve the key) or reject the same writes as the source. Without it, a replica was silently missing the constraints and a replayed `_MERGE` threw.

Apply a schema change on the **authoritative** side with `applySchema` — it applies the op to the store _and_ appends it to the `WriteLog` in one call, so replicas replay it in order with the surrounding data:

```ts
import { applySchema } from '@lenke/sync';

// server side: `store` + `writeLog` are the shared authoritative pair.
applySchema(store, { op: 'createUniqueConstraint', label: 'User', key: 'email' }, { writeLog });
// every subscribed replica now enforces it — and a keyed `_MERGE (:User {email})` upserts everywhere.
```

**Server-authoritative — not a CRDT.** Matching statement replication, the store is the single writer of schema; replicas _receive_ it, they don't originate it. `applySchema` **applies before it logs**, so a constraint the source's own data already violates throws and is **never published** — a replica never sees a constraint the source itself couldn't take. A schema op is unscoped and un-tokened, so it forwards past interest/scope routing to **every** subscriber (schema is graph-global). All 13 declaration methods replicate: vertex/edge index (+ drop), unique/required/type (node & edge), cardinality, validator, invariant.

`defineNode`/`defineEdge` are the deliberate exception: they bind a label to a **host-side JS validator** (a Zod/Valibot/ArkType schema), which isn't engine state and can't cross the wire. Their _writes_ replicate; a replica that runs its own local `.create()` executes the same app code. If the engine itself must enforce a slice on every replica, pair them with an in-engine type/required constraint (which _does_ replicate here).

**Cold boot carries schema too.** A late replica catches up by replaying the log backlog, which includes the schema ops. And if the log has rolled past a constraint's creation (older than the bounded tail → `resync` → cold-boot from an app snapshot), the **snapshot carries schema** as well: `encodeSnapshot` reads the live schema out of the engine (`graph.dumpSchema()`) into its own section, and `graphFromSnapshot(backend, snap)` replays it on boot (see [Snapshots (warm boot)](#snapshots-warm-boot) above). So a warm-booted replica enforces the same constraints without re-declaring them — use `graphFromSnapshot` instead of a bare `graphFromNdjson` and there's nothing left to do by hand.

## v1 boundaries (deliberate)

- **`window` is carried but not interpreted** — re-subscribe with the same `sub` to replace a standing query (that is also how a windowed grid will scroll).
- **Rows are JSON** — Arrow-buffer negotiation is an extension.
- **No auth/scoping** — a host serves its store; "the server syncs only what this user may see" is a store-construction concern for the sync loop, not a protocol concern.
- **No resume for _live-query_ subscriptions** — a reconnect re-subscribes standing queries from scratch (the host re-answers rows; `applyDiff` preserves identity). The **CDC write stream is resumable** (a `since` cursor + bounded op log — see above). `createReconnectingClient` wraps the client with re-dial/backoff and replays every standing query, parked one-shot, and the write-stream cursor over the fresh transport. Delivery is at-least-once — exactly-once needs server-side request-id dedupe.
- **No built-in conflict resolution (LWW/HLC)** — concurrent writes to the same element are ordered by whatever your app puts in `_MERGE … WHERE`; there is no shipped clock. If you add last-write-wins, the ordering stamp **must be host-assigned and bounded** — a client-assigned HLC is poisonable (skew-to-win + persistent, spreading clock inflation). See the design note & threat model: [`docs/design/conflict-resolution.md`](../../docs/design/conflict-resolution.md).
