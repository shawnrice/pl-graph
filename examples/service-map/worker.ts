// The SharedWorker: one lenke store per origin, shared by every tab.
//
// Boot order: OPFS snapshot (warm) or empty graph (cold) → sync engine with
// per-cluster demand-fill collections → one protocol host per connecting tab
// (the SharedWorker `connect` event hands us a MessagePort per tab — exactly
// the protocol's one-host-per-connection shape). The server link is a
// reconnecting WebSocket speaking the same protocol: loaders are one-shot
// `query`s against it, write-back is `mutate` (its ack settles the queue).
//
// Offline is not a special mode: while the socket is down, upstream.push
// simply doesn't settle — the engine's write stays queued (and counted in
// every tab's status bar), OPFS snapshots keep it across reloads, and the
// queue drains on reconnect.

import { createStore, graphFromNdjson } from '@lenke/native';
import { createWasmBackend } from '@lenke/native/wasm';
import {
  createReconnectingClient,
  createSnapshotStore,
  createSyncEngine,
  serveSharedWorker,
  type CollectionDefinition,
  type SyncWrite,
} from '@lenke/sync';

// oxlint-disable-next-line boundaries/no-cross-package-relative-import -- Vite `?url` asset import of the compiled wasm build output; a build artifact has no package entry point.
import wasmUrl from '../../crates/lenke-core/target/wasm32-unknown-unknown/release/lenke_core.wasm?url';
import { CLUSTERS } from './datagen.ts';

const SERVER_URL = 'ws://localhost:8787';
const SCHEMA_VERSION = 'service-map-v1';
const USER_ID = 'demo'; // a real app: the authenticated user, + a key for AES-GCM

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
//
// The server link used to be a ~40-line hand-rolled reconnecting client right
// here (the tire-kick finding that motivated it). It now IS the library's
// `createReconnectingClient`: loaders are its `query`, write-back its `mutate`,
// both parking while offline and replaying on reconnect; `onConnectivity`
// nudges every tab's status line. The whole transport is the `connect`
// callback below — open a socket, wire its lifecycle, hand back send/close.

const boot = async () => {
  // Demo data, no login → no key, so snapshots stay in memory and never touch
  // OPFS. This is a SharedWorker, so that memory warms every tab across reloads
  // for as long as the worker lives (closing every tab drops it → cold boot). A
  // real app hands a per-user CryptoKey ({ key }) → sealed + durable on disk.
  const snapshots = createSnapshotStore({ filename: 'service-map.lnks' });
  const snap = await snapshots.load({ schemaVersion: SCHEMA_VERSION, userId: USER_ID });

  const backend = await createWasmBackend(fetch(wasmUrl));
  const store = createStore(
    graphFromNdjson(backend, snap ? snap.ndjson : new TextEncoder().encode('')),
  );
  // Index the service id up front. Demand-fill now bulk-COPYs each cluster via
  // `mergeNdjson` (see the collection's `load`), so the load itself no longer
  // MATCHes by `sid` — but the LIVE queries still do: the status write
  // (`WHERE s.sid = $sid`) and the blast radius (`WHERE x.sid = $sid`). The
  // planner uses the index automatically for those `{sid: $x}` / `WHERE .sid = $x`
  // seeks.
  store.graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['sid'] });
  const server = createReconnectingClient({
    connect: ({ opened, received, closed }) => {
      const ws = new WebSocket(SERVER_URL);
      ws.onopen = opened;
      ws.onmessage = (e) => received(JSON.parse(String(e.data)));
      ws.onclose = closed;
      ws.onerror = () => ws.close();

      return { send: (m) => ws.send(JSON.stringify(m)), close: () => ws.close() };
    },
    retry: { baseMs: 500, maxMs: 5000 },
  });

  // ONE demand-fill collection, sliced by the `cluster` param. Every cluster
  // shares the :Service/:CALLS labels, so labels alone can't tell prod-east
  // from prod-west — but the subscription already carries the value as a param
  // (`WHERE s.cluster = $cluster`), so the collection just declares `cluster`
  // its scope key and the engine tracks completeness / demand-fill per value.
  // No synthetic token, no magic string on the deps channel.
  const collections: Record<string, CollectionDefinition> = {
    services: {
      labels: ['Service'],
      key: 'cluster',
      load: async ({ cluster }): Promise<SyncWrite[]> => {
        const services = await server.query(
          'MATCH (s:Service) WHERE s.cluster = $cluster RETURN s.sid, s.name, s.cluster, s.tier, s.status',
          { cluster },
        );
        const calls = await server.query(
          'MATCH (a:Service)-[t:CALLS]->(b:Service) WHERE a.cluster = $cluster RETURN t.cid, a.sid, b.sid, t.protocol, t.p95',
          { cluster },
        );

        // COPY the whole cluster in as ONE NDJSON batch instead of ~150
        // individual INSERTs — `mergeNdjson` bulk-appends it (no per-record
        // parse, no per-element crossing), ~6x faster for a cluster's worth of
        // rows. The server RETURNs columns named `s.sid`/`t.cid`/…, remapped here
        // to NDJSON node/edge records. Cross-cluster calls (whose target isn't in
        // this batch) are dropped — the same as the old two-endpoint MATCH — so
        // the merge stays clean (no phantom endpoints).
        const lines = [
          ...services.map((r) =>
            JSON.stringify({
              type: 'node',
              id: r['s.sid'],
              labels: ['Service'],
              properties: {
                sid: r['s.sid'],
                name: r['s.name'],
                cluster: r['s.cluster'],
                tier: r['s.tier'],
                status: r['s.status'],
              },
            }),
          ),
          ...calls
            .filter((r) => String(r['b.sid']).startsWith(`${cluster}:`))
            .map((r) =>
              JSON.stringify({
                type: 'edge',
                id: r['t.cid'],
                from: r['a.sid'],
                to: r['b.sid'],
                labels: ['CALLS'],
                properties: { cid: r['t.cid'], protocol: r['t.protocol'], p95: r['t.p95'] },
              }),
            ),
        ];

        return [{ text: '', ndjson: new TextEncoder().encode(lines.join('\n')) }];
      },
    },
  };

  const engine = createSyncEngine({
    store,
    collections,
    // Snapshot header stores the cluster names it covered; restore each as a
    // scoped slice of the one `services` collection.
    initiallyComplete: (snap?.header.collections ?? []).map((cluster) => ({
      name: 'services',
      scope: { cluster },
    })),
    initialWrites: snap?.pendingWrites ?? [],
    // `pushWrite` forwards the whole SyncWrite — text, params, AND lang —
    // together, so a Gremlin write can't lose its language and get parsed as
    // GQL on the server (which would park it in the queue forever).
    upstream: { push: server.pushWrite },
    retry: { attempts: Number.MAX_SAFE_INTEGER, baseMs: 500, maxMs: 5000 }, // outage ≠ poison: park, don't drop
  });

  // Snapshot on a debounce whenever anything moved (version, queue, loads).
  // "Moved" = the version OR the queue length changed since the last save —
  // every enqueue bumps the version (writes are version-gated) and every drain
  // drops the count, so the pair captures all queue movement. Comparing against
  // the last SAVE (not `> 0`) matters: a stuck offline queue would otherwise
  // re-encode the entire graph every tick for the whole outage.
  let lastSaved = -1;
  let lastSavedPending = -1;
  const save = async (): Promise<void> => {
    const loaded = CLUSTERS.filter(
      (c) => engine.collectionState('services', { cluster: c }) === 'complete',
    );
    lastSaved = store.version;
    lastSavedPending = engine.pendingWrites();
    await snapshots.save(store, {
      schemaVersion: SCHEMA_VERSION,
      userId: USER_ID,
      collections: loaded,
      pendingWrites: engine.queuedWrites(),
    });
  };

  setInterval(() => {
    if (store.version !== lastSaved || engine.pendingWrites() !== lastSavedPending) {
      void save();
    }
  }, 3000);

  return { engine, server };
};

// One host per connecting tab — with the full bye/bfcache/`close` teardown so a
// dead tab's standing queries don't re-run forever — plus a status broadcast on
// every upstream connectivity flip. `serveSharedWorker` sets `self.onconnect`
// synchronously (no early connection missed) and serves each tab once the
// still-booting engine resolves. The whole ~45-line port dance is now three
// lines.
const ready = boot();
const service = serveSharedWorker(ready.then(({ engine }) => engine));
void ready.then(({ server }) => server.onConnectivity(() => service.broadcastStatus()));
