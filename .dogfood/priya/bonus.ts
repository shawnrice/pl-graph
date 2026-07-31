import {
  createEmptyGraph,
  createStore,
  graphFromNdjson,
} from '@lenke/native';
/**
 * Bonus: the port lifecycle helper (`servePort`) and the snapshot store's
 * warm-boot path — an offline write survives a "reload" and drains after.
 *
 * Run: bun bonus.ts
 */
import { createFfiBackend } from '@lenke/native/ffi';
import {
  createSnapshotStore,
  createSyncClient,
  createSyncEngine,
  servePort,
  type PortLike,
  type SyncWrite,
} from '@lenke/sync';

const LIB = '/home/shawn/projects/pl-graph/crates/lenke-core/target/release/liblenke_core.so';
const log = (...a: unknown[]) => console.log(...a);
const hr = (t: string) => log(`\n──────── ${t} ────────`);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const backend = createFfiBackend(LIB);

// ── A minimal in-process MessagePort pair (what a real SharedWorker gives you).
function portPair(): [PortLike, PortLike] {
  const a: PortLike & { _peer?: PortLike } = { postMessage: () => {}, start: () => {} };
  const b: PortLike & { _peer?: PortLike } = { postMessage: () => {}, start: () => {} };
  a.postMessage = (m) => queueMicrotask(() => b.onmessage?.({ data: m } as MessageEvent));
  b.postMessage = (m) => queueMicrotask(() => a.onmessage?.({ data: m } as MessageEvent));
  return [a, b];
}

async function partOne() {
  hr('A. servePort helper: one host per connection over a bare port');
  const store = createStore(createEmptyGraph(backend));
  const engine = createSyncEngine({
    store,
    collections: {
      tasks: {
        labels: ['Task'],
        key: 'project',
        load: async (scope) => [
          {
            text: 'INSERT (:Task {id: $id, proj: $p})',
            params: { id: 'seed-1', p: scope.project },
          },
        ],
      },
    },
  });

  const [workerPort, uiPort] = portPair();
  // Worker side: servePort wires createHost + inbound pump + teardown-on-bye.
  const served = servePort(engine, workerPort);
  // UI side: a client over the other end.
  const client = createSyncClient({ send: (m) => uiPort.postMessage(m) });
  uiPort.onmessage = (e) => client.receive(e.data);

  const live = client.liveQuery('MATCH (t:Task) WHERE t.proj = $project RETURN t.id AS id', {
    deps: ['Task'],
    params: { project: 'proj-x' },
  });
  live.subscribe(() =>
    log(
      `  [ui] rows=${JSON.stringify(live.getSnapshot().rows.map((r) => r.id))} complete=${live.getSnapshot().complete}`,
    ),
  );
  await sleep(50);

  // A `bye` from the tab tears the host down; the returned `close` also does.
  workerPort.onmessage?.({ data: { type: 'bye' } } as MessageEvent);
  log('  sent bye; servePort tore its host down (no leak). sendStatus after bye is a no-op:');
  served.sendStatus(); // no throw
  served.close();
  client.close();
  store[Symbol.dispose]();
}

async function partTwo() {
  hr('B. Snapshot warm-boot: an offline write survives a reload and drains');
  // Memory-only snapshot store (no key → never touches disk). One instance so
  // save + load share the in-memory sink.
  const snapshots = createSnapshotStore({ filename: 'priya.snap' });
  const meta = { schemaVersion: 'v1', userId: 'priya' };

  // --- Session 1: apply an optimistic write while OFFLINE, then persist. ---
  const store1 = createStore(createEmptyGraph(backend));
  const engine1 = createSyncEngine({
    store: store1,
    upstream: {
      push: async () => {
        throw new Error('offline');
      },
    }, // never drains
    retry: { attempts: 1, baseMs: 5 },
    onWriteError: () => {}, // 1 attempt then drop — but we snapshot BEFORE it drops
  });
  engine1.mutate('INSERT (:Task {id: $id, title: $title})', { id: 'off-1', title: 'made offline' });
  const queued = engine1.queuedWrites();
  log(`  session1 queued ${queued.length} write(s) offline`);
  await snapshots.save(engine1.store, { ...meta, collections: [], pendingWrites: queued });
  log('  saved snapshot (graph + pending queue)');
  store1[Symbol.dispose]();

  // --- Session 2: "reload" — warm-boot from the snapshot, come back ONLINE. ---
  const snap = await snapshots.load(meta);
  if (!snap) throw new Error('snapshot missing');
  log(
    `  loaded snapshot: pendingWrites=${snap.pendingWrites.length}, ndjson=${snap.ndjson.byteLength}B`,
  );
  const store2 = createStore(graphFromNdjson(backend, snap.ndjson));
  const drained: SyncWrite[] = [];
  const engine2 = createSyncEngine({
    store: store2,
    initialWrites: snap.pendingWrites, // stranded write resumes replication
    upstream: {
      push: async (w) => {
        drained.push(w);
      },
    }, // online now
  });
  // The write's effect is already in the warm graph:
  const rows = store2.mutate((g) => g.query('MATCH (t:Task) RETURN t.id AS id, t.title AS title'));
  log(`  warm graph already contains: ${JSON.stringify(rows)}`);
  await sleep(50); // engine2 flushes initialWrites on construction
  log(
    `  after reconnect: engine2.pendingWrites()=${engine2.pendingWrites()}, upstream drained ${drained.length}`,
  );
  log(`  drained write params: ${JSON.stringify(drained.map((w) => w.params))}`);
  store2[Symbol.dispose]();
}

async function main() {
  await partOne();
  await partTwo();
  hr('DONE');
}

main().catch((e) => {
  console.error('FATAL', e);
  process.exit(1);
});
