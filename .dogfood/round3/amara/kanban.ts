import { rm } from 'node:fs/promises';

import {
  createEmptyGraph,
  createStore,
  graphFromNdjson,
  type Store,
} from '@lenke/native';
/**
 * Collaborative kanban board — a REAL client<->server split over a WebSocket.
 *
 *   SERVER (authoritative graph)  --- ws --->  CLIENT (local-first worker)
 *
 * - Server: Bun.serve({ websocket }) hosts the graph; one `createSyncHost` per
 *   socket serves standing queries + one-shots + mutations.
 * - Client: `createReconnectingClient` dials the ws (re-dial/back-off/replay);
 *   `createSyncEngine` holds a LOCAL store, demand-fills from the server, and
 *   replicates local writes upstream through the reconnecting client. Its
 *   write-back queue (`pendingWrites`) is what survives an outage + a reload.
 * - UI reads the local store through `engine.createHost` + `createSyncClient`
 *   over an in-process port (the browser worker seam).
 * - Warm boot: `createSnapshotStore` (AES-GCM key + a Bun filesystem durable
 *   adapter) persists graph + pendingWrites; a simulated reload restores both.
 *
 * Run: `bun kanban.ts`
 */
import { createFfiBackend } from '@lenke/native/ffi';
import {
  createReconnectingClient,
  createSnapshotStore,
  createSyncClient,
  createSyncEngine,
  createSyncHost,
  importSnapshotKey,
  type ReconnectingClient,
  type SnapshotStorage,
  type SyncClient,
  type SyncEngine,
} from '@lenke/sync';

const LIB = '/home/shawn/projects/pl-graph/crates/lenke-core/target/release/liblenke_core.so';
const SNAP_PATH = '/home/shawn/projects/pl-graph/.dogfood/round3/amara/kanban.snapshot';
const backend = createFfiBackend(LIB);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const banner = (s: string) => console.log(`\n=== ${s} ===`);

async function waitUntil(fn: () => boolean, label: string, timeoutMs = 5000): Promise<void> {
  const start = Date.now();
  while (!fn()) {
    if (Date.now() - start > timeoutMs) throw new Error(`timeout waiting for: ${label}`);
    await sleep(15);
  }
}

// A view of a card set as "id:col" for legible logging.
const fmt = (rows: Array<Record<string, unknown>>) =>
  '[' +
  rows
    .map((r) => `${r.id}:${r.col}`)
    .sort()
    .join(', ') +
  ']';

// ---------------------------------------------------------------------------
// SERVER — authoritative graph behind a real WebSocket.
// ---------------------------------------------------------------------------
const serverStore = createStore(createEmptyGraph(backend));
// Seed two cards (the "database").
serverStore.mutate((g) =>
  g.query('INSERT (:Card {id: $id, title: $t, col: $c})', { id: 'a', t: 'Design API', c: 'todo' }),
);
serverStore.mutate((g) =>
  g.query('INSERT (:Card {id: $id, title: $t, col: $c})', { id: 'b', t: 'Write tests', c: 'todo' }),
);

// A gate to simulate the server being unreachable (a network outage).
let accepting = true;
const hosts = new Map<unknown, ReturnType<typeof createSyncHost>>();

const server = Bun.serve({
  port: 0,
  fetch(req, srv) {
    if (accepting && srv.upgrade(req)) return undefined;
    return new Response(null, { status: 400 });
  },
  websocket: {
    open(ws) {
      hosts.set(ws, createSyncHost(serverStore, { send: (m) => ws.send(JSON.stringify(m)) }));
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
const URL_WS = `ws://localhost:${server.port}`;
console.log(`server up on ${URL_WS}, seeded ${serverStore.graph.vertexCount} cards`);

// A read-only helper: what does the server's graph actually hold right now?
const serverTruth = () =>
  fmt(serverStore.graph.query('MATCH (c:Card) RETURN c.id AS id, c.col AS col'));

// ---------------------------------------------------------------------------
// CLIENT — assembled as a factory so a "reload" can rebuild it from a snapshot.
// ---------------------------------------------------------------------------
type ClientRig = {
  reconnecting: ReconnectingClient;
  engine: SyncEngine;
  localStore: Store;
  uiClient: SyncClient;
  serverView: ReturnType<SyncClient['liveQuery']>;
  localView: ReturnType<SyncClient['liveQuery']>;
  teardown: () => void;
};

let currentWs: WebSocket | null = null;

function buildClient(opts: {
  seedNdjson?: Uint8Array;
  initialWrites?: readonly any[];
  complete?: readonly string[];
}): ClientRig {
  const localStore = createStore(
    opts.seedNdjson ? graphFromNdjson(backend, opts.seedNdjson) : createEmptyGraph(backend),
  );

  // The reconnecting client: one durable handle over a flapping socket.
  const reconnecting = createReconnectingClient({
    retry: { baseMs: 100, maxMs: 400 },
    connect: ({ opened, received, closed }) => {
      const ws = new WebSocket(URL_WS);
      currentWs = ws;
      ws.onopen = () => opened();
      ws.onmessage = (e) => received(JSON.parse(String(e.data)));
      ws.onclose = () => closed();
      ws.onerror = () => ws.close();
      return { send: (m) => ws.send(JSON.stringify(m)), close: () => ws.close() };
    },
  });

  // The sync engine: local store + demand-fill from server + upstream replication.
  const engine = createSyncEngine({
    store: localStore,
    collections: {
      cards: {
        labels: ['Card'],
        load: async () => {
          const rows = await reconnecting.query<{ id: string; title: string; col: string }>(
            'MATCH (c:Card) RETURN c.id AS id, c.title AS title, c.col AS col',
          );
          return rows.map((r) => ({
            text: 'INSERT (:Card {id: $id, title: $title, col: $col})',
            params: { id: r.id, title: r.title, col: r.col },
          }));
        },
      },
    },
    initiallyComplete: opts.complete ?? [],
    initialWrites: opts.initialWrites ?? [],
    upstream: { push: (w) => reconnecting.pushWrite(w) },
    retry: { attempts: 8, baseMs: 100, maxMs: 500 },
    onWriteError: (w, e) => console.log('  !! write dropped after retries:', w.text, e),
    onLoadError: (c, e) => console.log(`  !! load failed for ${c}:`, String(e)),
  });

  // The UI seam: read the LOCAL store through the engine's host over an
  // in-process port (the browser worker's postMessage channel, in miniature).
  let engineHost: ReturnType<SyncEngine['createHost']>;
  const uiClient = createSyncClient({ send: (m) => engineHost.receive(m) });
  engineHost = engine.createHost({ send: (m) => uiClient.receive(m) });

  // Standing query straight at the SERVER over the ws (proves round-trip + replay).
  const serverView = reconnecting.liveQuery('MATCH (c:Card) RETURN c.id AS id, c.col AS col', {
    deps: ['Card', 'id', 'col'],
  });
  serverView.subscribe(() => {});
  // Standing query at the LOCAL store (optimistic reads, demand-filled).
  const localView = uiClient.liveQuery('MATCH (c:Card) RETURN c.id AS id, c.col AS col', {
    deps: ['Card', 'id', 'col'],
  });
  localView.subscribe(() => {});

  return {
    reconnecting,
    engine,
    localStore,
    uiClient,
    serverView,
    localView,
    teardown: () => {
      reconnecting.close();
      uiClient.close();
      localStore[Symbol.dispose]();
    },
  };
}

// A Bun filesystem durable adapter (createSnapshotStore defaults to OPFS, which
// doesn't exist off-browser — so a server/Bun app must supply `durable`).
const fileStorage = (path: string): SnapshotStorage => ({
  read: async () => {
    const f = Bun.file(path);
    return (await f.exists()) ? new Uint8Array(await f.arrayBuffer()) : null;
  },
  write: async (bytes) => {
    await Bun.write(path, bytes);
  },
  delete: async () => {
    await rm(path, { force: true });
  },
});

async function main() {
  await rm(SNAP_PATH, { force: true });

  // ----- 1. WS ROUND-TRIP: cold-boot, demand-fill, observe both views -----
  banner('1. WS round-trip + demand-fill (cold boot)');
  let rig = buildClient({});
  await waitUntil(() => rig.reconnecting.connected(), 'ws connected');
  console.log('connected:', rig.reconnecting.connected());
  await waitUntil(() => rig.serverView.getSnapshot().complete, 'server view answered');
  console.log(
    'server view :',
    fmt(rig.serverView.getSnapshot().rows as any),
    'complete=',
    rig.serverView.getSnapshot().complete,
  );
  // Local view demand-fills the "cards" collection off the server on first subscribe.
  await waitUntil(
    () => rig.localView.getSnapshot().complete && rig.localView.getSnapshot().rows.length === 2,
    'local demand-fill complete',
  );
  console.log(
    'local view  :',
    fmt(rig.localView.getSnapshot().rows as any),
    'complete=',
    rig.localView.getSnapshot().complete,
  );
  console.log('server truth:', serverTruth());

  // ----- 2. ONLINE WRITE: optimistic local + replicate upstream -----
  banner('2. Online write replicates to server');
  rig.engine.mutate('INSERT (:Card {id: $id, title: $t, col: $c})', {
    id: 'c',
    t: 'Ship it',
    c: 'todo',
  });
  console.log(
    'after local mutate -> local:',
    fmt(rig.localView.getSnapshot().rows as any),
    'pendingWrites=',
    rig.engine.pendingWrites(),
  );
  await waitUntil(() => rig.engine.pendingWrites() === 0, 'write drained upstream');
  await waitUntil(
    () => (rig.serverView.getSnapshot().rows as any).length === 3,
    'server view sees c',
  );
  console.log('server view :', fmt(rig.serverView.getSnapshot().rows as any));
  console.log('server truth:', serverTruth(), '(c reached the authoritative graph)');

  // ----- 3. FORCED RECONNECT: replay standing queries -----
  banner('3. Forced reconnect replays standing queries');
  console.log('killing socket (network blip)...');
  currentWs?.close();
  await waitUntil(() => !rig.reconnecting.connected(), 'went offline');
  console.log('offline. subscriptionCount held =', rig.reconnecting.subscriptionCount());
  // Change server truth WHILE the client is away (another user moves card a).
  serverStore.mutate((g) =>
    g.query('MATCH (c:Card {id: $id}) SET c.col = $col', { id: 'a', col: 'doing' }),
  );
  console.log('another user moved a -> doing while we were offline; server truth:', serverTruth());
  await waitUntil(() => rig.reconnecting.connected(), 'reconnected');
  console.log('reconnected. waiting for replay to re-answer the standing query...');
  await waitUntil(
    () =>
      (rig.serverView.getSnapshot().rows as any).some(
        (r: any) => r.id === 'a' && r.col === 'doing',
      ),
    'replay delivered a:doing',
  );
  console.log(
    'server view after replay:',
    fmt(rig.serverView.getSnapshot().rows as any),
    '(picked up a:doing without re-subscribing by hand)',
  );

  // ----- 4. OFFLINE WRITE QUEUE drains on reconnect -----
  banner('4. Offline write queue drains on reconnect');
  console.log('taking server DOWN (gate closed) ...');
  accepting = false;
  currentWs?.close();
  await waitUntil(() => !rig.reconnecting.connected(), 'offline');
  rig.engine.mutate('INSERT (:Card {id: $id, title: $t, col: $c})', {
    id: 'd',
    t: 'Offline task',
    c: 'todo',
  });
  rig.engine.mutate('MATCH (c:Card {id: $id}) SET c.col = $col', { id: 'b', col: 'doing' });
  await sleep(300); // stay offline a while; the queue must NOT drain or drop
  console.log('while offline -> local:', fmt(rig.localView.getSnapshot().rows as any));
  console.log(
    'while offline -> pendingWrites=',
    rig.engine.pendingWrites(),
    '| server truth still:',
    serverTruth(),
  );
  const stranded = rig.engine.queuedWrites();
  console.log(
    'queued (stranded) writes:',
    stranded.map((w) => w.text.slice(0, 24) + '…'),
  );

  // ----- 5. ENCRYPTED SNAPSHOT while writes are still stranded -----
  banner('5. Encrypted snapshot (graph + pendingWrites)');
  const rawKey = crypto.getRandomValues(new Uint8Array(32)); // delivered at auth in a real app
  const key = await importSnapshotKey(rawKey);
  const snapStore = createSnapshotStore({
    filename: 'kanban.snapshot',
    key,
    durable: fileStorage(SNAP_PATH),
  });
  console.log('snapshot store durable(encrypted)=', snapStore.durable);
  await snapStore.save(rig.localStore, {
    schemaVersion: 'v1',
    userId: 'amara',
    collections: ['cards'],
    pendingWrites: rig.engine.queuedWrites(),
  });
  const onDisk = await Bun.file(SNAP_PATH).bytes();
  console.log(
    'wrote',
    onDisk.length,
    'bytes; encrypted flag byte =',
    onDisk[/* magic4 + u32 len + header */ 0] === 0x4c ? 'LNKS magic ok' : 'BAD',
  );
  // Prove it is NOT plaintext: the card titles must not appear in the bytes.
  const asText = new TextDecoder('latin1').decode(onDisk);
  console.log(
    'plaintext leak check — "Offline task" present in file bytes?',
    asText.includes('Offline task'),
  );

  // A WRONG key must read as absent (cold boot), never a lie. NOTE: load() =
  // readSnapshot(), which DELETES the file on a failed decode — so this negative
  // test runs against a non-deleting frozen copy of the bytes, or it would nuke
  // the real snapshot the warm boot needs. (Gotcha, logged in the report.)
  const frozen: SnapshotStorage = {
    read: async () => onDisk,
    write: async () => {},
    delete: async () => {},
  };
  const wrongKey = await importSnapshotKey(crypto.getRandomValues(new Uint8Array(32)));
  const wrongStore = createSnapshotStore({
    filename: 'kanban.snapshot',
    key: wrongKey,
    durable: frozen,
  });
  console.log(
    'load with WRONG key ->',
    await wrongStore.load({ schemaVersion: 'v1', userId: 'amara' }),
    '(null = cold boot, correct)',
  );

  // ----- 6. WARM BOOT: simulate a reload, restore graph + resume stranded writes -----
  banner('6. Warm boot from encrypted snapshot');
  console.log('simulating reload: tearing down client (engine + sockets gone)...');
  rig.teardown();
  await sleep(50);
  console.log('server truth (unchanged, d/b:doing never made it up):', serverTruth());

  const snap = await snapStore.load({ schemaVersion: 'v1', userId: 'amara' });
  if (!snap) throw new Error('warm boot FAILED: snapshot read as null');
  console.log(
    'decrypted snapshot: ndjson',
    snap.ndjson.length,
    'bytes, pendingWrites=',
    snap.pendingWrites.length,
    ', collections=',
    snap.header.collections,
  );

  accepting = true; // server reachable again
  rig = buildClient({
    seedNdjson: snap.ndjson,
    initialWrites: snap.pendingWrites,
    complete: snap.header.collections,
  });
  // Warm: the local view answers immediately from the restored graph.
  await waitUntil(() => rig.localView.getSnapshot().complete, 'warm local view');
  console.log(
    'warm local view (from disk, no server needed):',
    fmt(rig.localView.getSnapshot().rows as any),
  );
  console.log('restored pendingWrites=', rig.engine.pendingWrites());

  await waitUntil(() => rig.reconnecting.connected(), 'reconnected after reload');
  console.log('reconnected; draining stranded writes to the server...');
  await waitUntil(() => rig.engine.pendingWrites() === 0, 'stranded writes drained');
  await waitUntil(() => {
    const t = serverTruth();
    return t.includes('d:todo') && t.includes('b:doing');
  }, 'server caught up with stranded writes');
  console.log(
    'server truth after warm-boot drain:',
    serverTruth(),
    '(d + b:doing resumed replication)',
  );
  console.log('final server view:', fmt(rig.serverView.getSnapshot().rows as any));

  // ----- done -----
  banner('DONE — tearing down');
  rig.teardown();
  await rm(SNAP_PATH, { force: true });
  server.stop(true);
  serverStore[Symbol.dispose]();
  await sleep(50);
  console.log('clean exit');
  process.exit(0);
}

main().catch((e) => {
  console.error('FAILED:', e);
  process.exit(1);
});
