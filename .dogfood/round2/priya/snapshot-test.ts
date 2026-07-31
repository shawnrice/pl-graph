// Exercises the README "Snapshots (warm boot)" example AS WRITTEN, in-process.
// Run: bun snapshot-test.ts
//
// OPFS/CompressionStream/WebCrypto: OPFS doesn't exist under Bun, so we use the
// documented `durable:` override (memorySnapshotStorage) to run the full
// encode->encrypt->store->load->decrypt path here. gzip + subtle.crypto are
// native in Bun.

import {
  createEmptyGraph,
  createStore,
  graphFromNdjson,
} from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';
import {
  createSyncEngine,
  createSnapshotStore,
  memorySnapshotStorage,
  importSnapshotKey,
  encodeSnapshot,
  decodeSnapshot,
  peekHeader,
  type SyncWrite,
} from '@lenke/sync';

const LIB = '/home/shawn/projects/pl-graph/crates/lenke-core/target/release/liblenke_core.so';
const backend = createFfiBackend(LIB);
const log = (...a: unknown[]) => console.log(...a);
const hr = (t: string) => log(`\n=== ${t} ===`);

async function main() {
  // --- Build a store + engine with some queued (un-acked) offline writes -----
  const store = createStore(createEmptyGraph(backend));
  store.mutate((g) => g.query("INSERT (:Task {id: 't1', title: 'seed task', done: false})"));

  // upstream that never settles → writes stay queued (offline), so
  // engine.queuedWrites() has something to persist.
  const engine = createSyncEngine({
    store,
    upstream: { push: () => new Promise<void>(() => {}) },
    retry: { attempts: 3, baseMs: 5_000 },
  });
  engine.mutate('INSERT (:Task {id: $id, title: $t, done: false})', { id: 't2', t: 'unsynced A' });
  engine.mutate('INSERT (:Task {id: $id, title: $t, done: false})', { id: 't3', t: 'unsynced B' });
  await new Promise((r) => setTimeout(r, 10));
  log('pendingWrites before save:', engine.pendingWrites());
  log('queuedWrites persisted:', engine.queuedWrites().length);

  // --- Save (README example: createSnapshotStore + save) ---------------------
  hr('SAVE (encrypted + durable via memory sink)');
  const key = await importSnapshotKey(crypto.getRandomValues(new Uint8Array(32)));
  const durable = memorySnapshotStorage();
  const snapshots = createSnapshotStore({ filename: 'lenke.snapshot', key, durable });
  log('snapshots.durable:', snapshots.durable);
  await snapshots.save(engine.store, {
    schemaVersion: 'v3',
    userId: 'priya',
    serverCursor: 'cursor-42',
    collections: ['tasks'],
    pendingWrites: engine.queuedWrites(),
  });
  const rawBytes = await durable.read();
  log('bytes on the durable sink:', rawBytes?.byteLength, 'bytes');
  log('peekHeader (no key):', peekHeader(rawBytes!));
  store[Symbol.dispose]();

  // --- Boot (README example: load + graphFromNdjson + initialWrites) ---------
  hr('BOOT (warm)');
  const snap = await snapshots.load({ schemaVersion: 'v3', userId: 'priya' });
  log('snap present:', !!snap, '| header.collections:', snap?.header.collections);
  log('restored pendingWrites:', snap?.pendingWrites.length);

  const replicated: SyncWrite[] = [];
  const store2 = createStore(
    snap ? graphFromNdjson(backend, snap.ndjson) : createEmptyGraph(backend),
  );
  const engine2 = createSyncEngine({
    store: store2,
    collections: {},
    initiallyComplete: snap?.header.collections ?? [],
    initialWrites: snap?.pendingWrites ?? [],
    upstream: { push: async (w) => void replicated.push(w) }, // now online
  });
  // Warm graph already has the optimistic effects:
  log(
    'warm rows:',
    store2.graph.query('MATCH (t:Task) RETURN t.id AS id, t.title AS title ORDER BY id'),
  );
  await new Promise((r) => setTimeout(r, 30));
  log('initialWrites replicated to upstream (no local re-apply):', replicated.length);
  log('engine2.pendingWrites() after drain:', engine2.pendingWrites());

  // --- Invalidation: warmth-never-truth --------------------------------------
  hr('INVALIDATION (wrong user / wrong schema → cold boot)');
  log(
    'load wrong userId  →',
    await snapshots.load({ schemaVersion: 'v3', userId: 'someone-else' }),
  );
  log('load wrong schema  →', await snapshots.load({ schemaVersion: 'v9', userId: 'priya' }));

  // --- Keyless memory store (docs: key omitted → memory-only, never disk) -----
  hr('KEYLESS memory store round-trip');
  const memStore = createSnapshotStore({ filename: 'mem.snap' });
  log('durable (should be false):', memStore.durable);
  await memStore.save(store2, { schemaVersion: 'v3', userId: 'priya', collections: ['tasks'] });
  const memSnap = await memStore.load({ schemaVersion: 'v3', userId: 'priya' });
  log('keyless load present:', !!memSnap, '| ndjson bytes:', memSnap?.ndjson.byteLength);

  // --- Low-level primitive: explicit { unencrypted: true } --------------------
  hr('PRIMITIVE encode/decode { unencrypted: true }');
  const plain = await encodeSnapshot(
    store2,
    { schemaVersion: 'v3', userId: 'priya', collections: ['tasks'] },
    { unencrypted: true },
  );
  const decoded = await decodeSnapshot(
    plain,
    { schemaVersion: 'v3', userId: 'priya' },
    { unencrypted: true },
  );
  log('unencrypted round-trip present:', !!decoded, '| header:', decoded?.header.schemaVersion);
  // encrypted bytes fed a keyless decode must read as absent:
  const mustBeNull = await decodeSnapshot(
    rawBytes!,
    { schemaVersion: 'v3', userId: 'priya' },
    {
      unencrypted: true,
    },
  );
  log('encrypted-bytes + unencrypted decode →', mustBeNull);

  store2[Symbol.dispose]();
  hr('SNAPSHOT DONE');
}

main().catch((e) => {
  console.error('FAILED:', e);
  process.exit(1);
});
