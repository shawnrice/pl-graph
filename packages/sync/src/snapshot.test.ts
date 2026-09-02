// Proves the warm-boot layer honors its load-bearing rule — the snapshot is
// warmth, never truth: every failure mode (tamper, truncation, wrong key,
// version/user mismatch, garbage bytes) reads as ABSENT and the caller cold
// boots. Plus the one exception: the pending-write queue rides the snapshot
// and resumes replication on boot. Run: bun test packages/sync/src/snapshot.test.ts
import { afterEach, describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson, type Store } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';

import { createSyncEngine, type SyncWrite } from './engine.js';
import {
  createSnapshotStore,
  decodeSnapshot,
  encodeSnapshot,
  graphFromSnapshot,
  importSnapshotKey,
  memorySnapshotStorage,
  opfsStorage,
  peekHeader,
  readSnapshot,
  type SnapshotStorage,
} from './snapshot.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[snapshot.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"b","labels":["Person"],"properties":{"name":"vadas","age":27}}',
].join('\n');

// Every store built here owns a native graph handle. These tests create them
// freely (often inline, as a throwaway arg to encodeSnapshot), so rather than
// dispose each by hand we track them and free the lot after each test — a leaked
// handle otherwise trips the GC-backstop warning when the finalizer runs. free()
// is idempotent, so a directly-tracked store is safe to free twice.
const created: Store[] = [];
const track = (store: Store): Store => {
  created.push(store);

  return store;
};

afterEach(() => {
  for (const store of created.splice(0)) {
    store.free();
  }
});

const newStore = (seed: string = NDJSON): Store =>
  track(createStore(graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(seed))));

const EXPECT = { schemaVersion: 'v1', userId: 'shawn' };
const KEY_BYTES = new Uint8Array(32).map((_, i) => i * 7 + 1);
// Persisting plaintext is an explicit, greppable choice (secure-by-default).
const PLAIN = { unencrypted: true } as const;

suite('@lenke/sync snapshot · codec', () => {
  test('plaintext round-trip: graph, header, and queue survive', async () => {
    const store = newStore();
    const writes: SyncWrite[] = [{ text: 'INSERT (:Person {name: $n})', params: { n: 'queued' } }];
    const bytes = await encodeSnapshot(
      store,
      {
        ...EXPECT,
        serverCursor: 'cursor-42',
        collections: ['people'],
        pendingWrites: writes,
      },
      PLAIN,
    );

    const snap = await decodeSnapshot(bytes, EXPECT, PLAIN);
    expect(snap).not.toBeNull();
    expect(snap!.header).toMatchObject({
      formatVersion: 3,
      schemaVersion: 'v1',
      userId: 'shawn',
      serverCursor: 'cursor-42',
      collections: ['people'],
    });
    expect(snap!.pendingWrites).toEqual(writes);

    // The NDJSON payload actually reconstructs the graph.
    const restored = newStore(new TextDecoder().decode(snap!.ndjson));
    expect(restored.graph.vertexCount).toBe(2);
  });

  test('ephemeral nodes (and their incident edges) are stripped from the snapshot', async () => {
    const seed = [
      '{"type":"node","id":"u1","labels":["User"],"properties":{"name":"a"}}',
      '{"type":"node","id":"p1","labels":["Presence"],"properties":{"sid":"x"}}',
      '{"type":"edge","id":"e1","from":"u1","to":"p1","labels":["HAS_PRESENCE"],"properties":{}}',
    ].join('\n');
    const store = newStore(seed);

    const bytes = await encodeSnapshot(store, { ...EXPECT, ephemeralLabels: ['Presence'] }, PLAIN);
    const snap = await decodeSnapshot(bytes, EXPECT, PLAIN);
    const text = new TextDecoder().decode(snap!.ndjson);

    expect(text).toContain('User'); // the durable node survives
    expect(text).not.toContain('Presence'); // the ephemeral node is stripped
    expect(text).not.toContain('HAS_PRESENCE'); // and the edge incident to it

    // The stripped snapshot still reconstructs cleanly — one durable vertex.
    const restored = newStore(text);
    expect(restored.graph.vertexCount).toBe(1);
  });

  test('crypto choice is required — no silent keyless plaintext', async () => {
    const store = newStore();
    // @ts-expect-error the union forbids omitting the crypto choice…
    expect(encodeSnapshot(store, EXPECT)).rejects.toThrow(/refusing to silently write/);
    // …and passing an empty object (the old accidental default) is refused too.
    expect(encodeSnapshot(store, EXPECT, {} as { unencrypted: true })).rejects.toThrow(
      /refusing to silently write/,
    );

    const bytes = await encodeSnapshot(store, EXPECT, PLAIN);
    // A misused crypto arg on decode throws loudly (programmer error), rather
    // than being swallowed as a corrupt-snapshot cold boot.
    expect(decodeSnapshot(bytes, EXPECT, {} as { unencrypted: true })).rejects.toThrow(
      /refusing to silently write/,
    );
  });

  test('legacy and malformed persisted writes normalize safely', async () => {
    const store = newStore();
    // Simulate a pre-`lang` snapshot ({gql}) and a corrupted write (neither
    // key) riding in the persisted queue — the cast models old/damaged bytes.
    const persisted = [
      { gql: 'INSERT (:Person {name: $n})', params: { n: 'legacy' } },
      { params: { n: 'garbage' } }, // no text, no gql → must be dropped
      { text: "g.addV('Person')", lang: 'gremlin' },
    ] as unknown as SyncWrite[];
    const bytes = await encodeSnapshot(store, { ...EXPECT, pendingWrites: persisted }, PLAIN);

    const snap = await decodeSnapshot(bytes, EXPECT, PLAIN);
    expect(snap!.pendingWrites).toEqual([
      { text: 'INSERT (:Person {name: $n})', params: { n: 'legacy' } }, // gql → text
      { text: "g.addV('Person')", lang: 'gremlin' }, // lang survives
    ]); // the text-less write is gone, not { text: undefined } poisoning the queue
  });

  test('encrypted round-trip; wrong key and missing key read as absent', async () => {
    const store = newStore();
    const key = await importSnapshotKey(KEY_BYTES);
    const bytes = await encodeSnapshot(store, EXPECT, { key });

    const good = await decodeSnapshot(bytes, EXPECT, { key });
    expect(good).not.toBeNull();
    expect(newStore(new TextDecoder().decode(good!.ndjson)).graph.vertexCount).toBe(2);

    const wrongKey = await importSnapshotKey(new Uint8Array(32).map((_, i) => i + 99));
    expect(await decodeSnapshot(bytes, EXPECT, { key: wrongKey })).toBeNull();
    // Decoding a sealed snapshot as plaintext (no key) gzip-fails → absent.
    expect(await decodeSnapshot(bytes, EXPECT, PLAIN)).toBeNull();
  });

  test('the header stays readable without the key (the invalidation tier)', async () => {
    const key = await importSnapshotKey(KEY_BYTES);
    const bytes = await encodeSnapshot(newStore(), { ...EXPECT, serverCursor: 'c9' }, { key });

    expect(peekHeader(bytes)).toMatchObject({ userId: 'shawn', serverCursor: 'c9' });
  });

  test('any expectation mismatch reads as absent (dump → cold boot)', async () => {
    const bytes = await encodeSnapshot(newStore(), EXPECT, PLAIN);

    expect(await decodeSnapshot(bytes, { schemaVersion: 'v2', userId: 'shawn' }, PLAIN)).toBeNull();
    expect(
      await decodeSnapshot(bytes, { schemaVersion: 'v1', userId: 'intruder' }, PLAIN),
    ).toBeNull();
  });

  test('tamper, truncation, and garbage all read as absent', async () => {
    const key = await importSnapshotKey(KEY_BYTES);
    const plain = await encodeSnapshot(newStore(), EXPECT, PLAIN);
    const sealed = await encodeSnapshot(newStore(), EXPECT, { key });

    const flipped = sealed.slice();
    flipped[flipped.byteLength - 1] ^= 0xff; // tamper with AES-GCM ciphertext
    expect(await decodeSnapshot(flipped, EXPECT, { key })).toBeNull();

    const corrupt = plain.slice();
    corrupt[corrupt.byteLength - 5] ^= 0xff; // corrupt the gzip payload
    expect(await decodeSnapshot(corrupt, EXPECT, PLAIN)).toBeNull();

    expect(await decodeSnapshot(plain.subarray(0, 40), EXPECT, PLAIN)).toBeNull(); // truncated
    expect(
      await decodeSnapshot(new TextEncoder().encode('not a snapshot'), EXPECT, PLAIN),
    ).toBeNull();
    expect(peekHeader(new Uint8Array(0))).toBeNull();
  });

  test('an edited (but still valid) header on an encrypted snapshot reads as absent', async () => {
    const key = await importSnapshotKey(KEY_BYTES);
    // A distinctive cursor we can swap for an equal-length one — same headerLen,
    // same JSON structure, so peekHeader still parses and expectations pass.
    const bytes = await encodeSnapshot(
      newStore(),
      { ...EXPECT, serverCursor: 'AAAAAAAA' },
      { key },
    );
    expect(await decodeSnapshot(bytes, EXPECT, { key })).not.toBeNull(); // untouched: fine

    const tampered = bytes.slice();
    const view = new DataView(tampered.buffer, tampered.byteOffset, tampered.byteLength);
    const headerLen = view.getUint32(5, true);
    const headerText = new TextDecoder().decode(tampered.subarray(9, 9 + headerLen));
    const at = headerText.indexOf('AAAAAAAA');
    new TextEncoder().encodeInto('BBBBBBBB', tampered.subarray(9 + at)); // edit in place

    // The header edit took (it's plaintext) — but it's bound to the ciphertext
    // as AEAD additionalData, so the decrypt fails the tag and we cold-boot,
    // instead of trusting a forged `collections`/`serverCursor`.
    expect(peekHeader(tampered)).toMatchObject({ serverCursor: 'BBBBBBBB' });
    expect(await decodeSnapshot(tampered, EXPECT, { key })).toBeNull();
  });

  test('readSnapshot deletes an invalid-forever snapshot on the way out', async () => {
    const storage = memorySnapshotStorage();
    await storage.write(await encodeSnapshot(newStore(), EXPECT, PLAIN));

    // Schema bumped: the stored snapshot can never become valid again.
    const miss = await readSnapshot(storage, { schemaVersion: 'v2', userId: 'shawn' }, PLAIN);
    expect(miss).toBeNull();
    expect(await storage.read()).toBeNull(); // reclaimed

    // Absent storage is a quiet cold boot.
    expect(await readSnapshot(storage, EXPECT, PLAIN)).toBeNull();
  });
});

suite('@lenke/sync snapshot · opfsStorage backstop (encrypted-only durable)', () => {
  test('refuses to write a plaintext snapshot to OPFS', async () => {
    const plain = await encodeSnapshot(newStore(), EXPECT, PLAIN);
    // The crypto guard runs before any navigator/OPFS access, so it throws even
    // in this OPFS-less runtime — plaintext can't reach disk via the primitive.
    expect(opfsStorage('x.lnks').write(plain)).rejects.toThrow(/refusing to write an unencrypted/);
  });

  test('lets a sealed snapshot through the guard (fails later, not on crypto)', async () => {
    const key = await importSnapshotKey(KEY_BYTES);
    const sealed = await encodeSnapshot(newStore(), EXPECT, { key });
    let msg = '';

    try {
      await opfsStorage('x.lnks').write(sealed);
    } catch (e) {
      msg = e instanceof Error ? e.message : String(e);
    }

    // No OPFS here, so the write still fails — but on navigator access, NOT the
    // crypto guard: proof the guard passed a sealed snapshot through.
    expect(msg).not.toMatch(/refusing to write an unencrypted/);
    expect(msg).not.toBe('');
  });
});

suite('@lenke/sync snapshot · createSnapshotStore (keyed routing)', () => {
  // A durable sink that records everything the store writes to "disk".
  const spySink = () => {
    const inner = memorySnapshotStorage();
    const writes: Uint8Array[] = [];
    let deletes = 0;
    const storage: SnapshotStorage = {
      read: inner.read,
      write: async (b) => {
        writes.push(b.slice());
        await inner.write(b);
      },
      delete: async () => {
        deletes += 1;
        await inner.delete();
      },
    };

    return { storage, writes, deletes: () => deletes };
  };

  test('keyless: round-trips in memory and never touches the durable sink', async () => {
    const sink = spySink();
    // Pass a durable override AND no key — the override must be ignored.
    const snaps = createSnapshotStore({ filename: 'x.lnks', durable: sink.storage });
    expect(snaps.durable).toBe(false);

    await snaps.save(newStore(), { ...EXPECT, collections: ['people'] });
    expect(sink.writes).toHaveLength(0); // nothing hit disk

    const snap = await snaps.load(EXPECT);
    expect(snap).not.toBeNull();
    expect(snap!.header.collections).toEqual(['people']);
    expect(newStore(new TextDecoder().decode(snap!.ndjson)).graph.vertexCount).toBe(2);
  });

  test('keyed: seals to the durable sink; plaintext never lands on disk', async () => {
    const sink = spySink();
    const key = await importSnapshotKey(KEY_BYTES);
    const snaps = createSnapshotStore({ filename: 'x.lnks', key, durable: sink.storage });
    expect(snaps.durable).toBe(true);

    await snaps.save(newStore(), EXPECT);
    expect(sink.writes).toHaveLength(1); // persisted durably

    // What reached "disk" is ciphertext: the header reads, but only the key decodes it.
    const [onDisk] = sink.writes;
    expect(peekHeader(onDisk)).toMatchObject({ userId: 'shawn' });
    expect(await decodeSnapshot(onDisk, EXPECT, PLAIN)).toBeNull(); // not plaintext
    expect(await snaps.load(EXPECT)).not.toBeNull(); // the store decrypts its own

    await snaps.clear();
    expect(sink.deletes()).toBe(1);
    expect(await snaps.load(EXPECT)).toBeNull(); // gone → cold boot
  });
});

suite('@lenke/sync snapshot · warm boot', () => {
  test('save → boot: graph, completeness, cursor, and queue all resume', async () => {
    // Session 1: a loaded engine with one write stuck in the queue.
    const held: SyncWrite[] = [];
    const session1 = createSyncEngine({
      store: newStore(),
      collections: { people: { labels: ['Person'], load: () => Promise.resolve([]) } },
      initiallyComplete: ['people'],
      upstream: { push: () => new Promise(() => {}) }, // upstream never answers
    });
    session1.mutate('INSERT (:Person {name: $n})', { n: 'offline-edit' });
    expect(session1.pendingWrites()).toBe(1);

    const storage = memorySnapshotStorage();
    await storage.write(
      await encodeSnapshot(
        session1.store,
        {
          ...EXPECT,
          serverCursor: 'resume-here',
          collections: ['people'],
          pendingWrites: session1.queuedWrites(),
        },
        PLAIN,
      ),
    );

    // Session 2 (the "reload"): boot from the snapshot.
    const snap = await readSnapshot(storage, EXPECT, PLAIN);
    expect(snap).not.toBeNull();
    expect(snap!.header.serverCursor).toBe('resume-here'); // the app resumes its stream here

    const store2 = newStore(new TextDecoder().decode(snap!.ndjson));
    const session2 = createSyncEngine({
      store: store2,
      collections: { people: { labels: ['Person'], load: () => Promise.resolve([]) } },
      initiallyComplete: snap!.header.collections,
      initialWrites: snap!.pendingWrites,
      upstream: {
        push: (w) => {
          held.push(w);

          return Promise.resolve();
        },
      },
    });

    // The optimistic edit is IN the graph (not replayed, not lost)…
    expect(store2.graph.vertexCount).toBe(3);
    // …the collection is warm (no demand-fill)…
    expect(session2.isComplete(['Person'])).toBe(true);
    // …and the stranded write resumes replication immediately.
    await new Promise((r) => {
      setTimeout(r, 10);
    });
    expect(held).toEqual([{ text: 'INSERT (:Person {name: $n})', params: { n: 'offline-edit' } }]);
    expect(session2.pendingWrites()).toBe(0);
  });
});

// A snapshot carries the graph's schema (constraints/validators/invariants/indexes)
// in its own section — data NDJSON can't — so a COLD BOOT restores a graph that
// still enforces them. Without this, a warm-booted replica would silently drop its
// constraints and a queued `_MERGE` would throw for want of the key it upserts on.
suite('@lenke/sync snapshot · schema restore', () => {
  test('constraints + validators survive encode → decode → graphFromSnapshot', async () => {
    const store = newStore('\n'); // empty graph; declare schema, then upsert into it
    store.mutate((g) => {
      g.createUniqueConstraint('User', 'email');
      g.createValidator('User', 'u', 'u.age >= 0');
      g.createIndex({ on: 'vertex', kind: 'hash', keys: ['email'] });
      g.query("_MERGE (u:User {email: 'a@b.com', age: 30})");
    });

    const bytes = await encodeSnapshot(store, { ...EXPECT, pendingWrites: [] }, PLAIN);
    const snap = await decodeSnapshot(bytes, EXPECT, PLAIN);
    expect(snap).not.toBeNull();
    // The schema section decoded as structured ops (not write-language text).
    expect(snap!.schema).toContainEqual({
      op: 'createUniqueConstraint',
      label: 'User',
      key: 'email',
    });
    expect(snap!.schema).toContainEqual({
      op: 'createValidator',
      label: 'User',
      varName: 'u',
      predicate: 'u.age >= 0',
    });

    // Rebuild the graph the complete way — data AND schema replayed.
    const restored = track(createStore(graphFromSnapshot(createFfiEngineBackend(LIB), snap!)));

    // The unique constraint is live: the keyed `_MERGE` upserts (doesn't duplicate)…
    restored.mutate((g) => g.query("_MERGE (u:User {email: 'a@b.com', age: 31})"));
    const users = restored.mutate((g) =>
      g.query<{ email: string; age: number }>(
        'MATCH (u:User) RETURN u.email AS email, u.age AS age',
      ),
    );
    expect(users).toEqual([{ email: 'a@b.com', age: 31 }]); // one row, payload updated

    // …and the validator is live: a forbidden write is rejected on the replica.
    expect(() =>
      restored.mutate((g) => g.query('INSERT (:User {email: $e, age: -5})', { e: 'x@y.z' })),
    ).toThrow();
  });

  test('a schema-less graph round-trips with an empty schema section', async () => {
    const store = newStore(); // two Persons, no constraints
    const bytes = await encodeSnapshot(store, { ...EXPECT, pendingWrites: [] }, PLAIN);
    const snap = await decodeSnapshot(bytes, EXPECT, PLAIN);
    expect(snap!.schema).toEqual([]);
    // Data still restores fine through the complete-boot helper.
    const restored = track(createStore(graphFromSnapshot(createFfiEngineBackend(LIB), snap!)));
    expect(restored.graph.vertexCount).toBe(2);
  });
});
