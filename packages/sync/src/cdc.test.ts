// The CDC write stream end-to-end: several clients share one authoritative
// Store + one WriteLog (the multiplayer server topology). One client mutates;
// the others' stream subscribers receive the raw write and ingest it into their
// OWN local store — cross-client optimism over the wire, which the rows-only
// protocol couldn't do. Also covers origin-skip, catch-up replay, and resync.
// Run: bun test packages/sync/src/cdc.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson, type Store } from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';

import { createSyncClient, type SyncClient } from './client.js';
import { createDedupRegistry } from './dedup.js';
import { createSyncHost } from './host.js';
import {
  runWrite,
  type ClientMessage,
  type HostMessage,
  type SyncWrite,
  type WritesMessage,
} from './protocol.js';
import { applySchema } from './schema.js';
import { createWriteLog, type WriteLog } from './writelog.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  // eslint-disable-next-line no-console
  console.warn(`[cdc.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const SEED = '{"type":"node","id":"seed","labels":["Seed"],"properties":{}}';
const newStore = (): Store =>
  createStore(graphFromNdjson(createFfiBackend(LIB), new TextEncoder().encode(SEED)));

/** Count of `Widget` nodes — a cheap way to read a store's state. */
const widgets = (s: Store): number =>
  s.mutate((g) => g.query<{ c: number }>('MATCH (n:Widget) RETURN count(*) AS c'))[0].c;

/** A server: one authoritative store + one shared op log; `client()` attaches a
 *  fresh connection (its own host) to it. */
const server = (writeLog: WriteLog = createWriteLog(), scopeKey?: string) => {
  const store = newStore();
  // A fresh connection (its own host) on the shared store+log. Pass a `clientId`
  // to pin a stable identity — the same value re-attached over a new host is a
  // reconnect of the SAME client (origin-skip must hold across it).
  const client = (clientId?: string): SyncClient => {
    // Circular wiring (host.send → client, client.send → host); a box lets both
    // stay `const` while the client is assigned after the host is built.
    const link: { c?: SyncClient } = {};
    const host = createSyncHost(store, { send: (m) => link.c?.receive(m), writeLog, scopeKey });
    link.c = createSyncClient({ send: (m) => host.receive(m), clientId });

    return link.c;
  };

  return { store, writeLog, client };
};

suite('CDC write stream (TS vs native store)', () => {
  test("a client ingests another client's write into its own local store", async () => {
    const s = server();
    const a = s.client();
    const b = s.client();

    // A pipes the CDC stream into a local store — exactly what engine.ingest does.
    const localA = newStore();
    a.subscribeWrites((writes) => {
      localA.mutate((g) => {
        for (const w of writes) {
          runWrite(g, w);
        }
      });
    });

    expect(widgets(localA)).toBe(0);
    await b.mutate('INSERT (:Widget {id: 1})');

    // Delivered over the write stream — A's local store now reflects B's write,
    // and so does the authoritative server.
    expect(widgets(localA)).toBe(1);
    expect(widgets(s.store)).toBe(1);
  });

  test('the writer does not receive its own write (origin-skip)', async () => {
    const s = server();
    const a = s.client();
    const b = s.client();
    const aSeen: SyncWrite[] = [];
    const bSeen: SyncWrite[] = [];
    a.subscribeWrites((w) => aSeen.push(...w));
    b.subscribeWrites((w) => bSeen.push(...w));

    await b.mutate('INSERT (:Widget {id: 1})');

    expect(aSeen.map((w) => w.text)).toEqual(['INSERT (:Widget {id: 1})']);
    expect(bSeen).toEqual([]); // B applied it optimistically; no self-echo
  });

  test('value-scope routing: a room-scoped subscriber gets only its room', async () => {
    // The server partitions the CDC stream by the `room` property.
    const s = server(createWriteLog(), 'room');
    const a = s.client(); // watches room 42 only
    const b = s.client(); // watches rooms 7 and 9
    const writer = s.client();
    const aSeen: string[] = [];
    const bSeen: string[] = [];
    a.subscribeWrites((w) => aSeen.push(...w.map((x) => x.text)), { scopes: ['42'] });
    b.subscribeWrites((w) => bSeen.push(...w.map((x) => x.text)), { scopes: ['7', '9'] });

    await writer.mutate('INSERT (:Msg {room: 42, body: $t})', { t: 'to-42' });
    await writer.mutate('INSERT (:Msg {room: 7, body: $t})', { t: 'to-7' });
    await writer.mutate('INSERT (:Msg {room: 99, body: $t})', { t: 'to-99' });

    // A (room 42) saw only the 42 write; B (rooms 7,9) saw only the 7 write; the
    // room-99 write reached neither — the graph is NOT replicated whole.
    expect(aSeen).toEqual(['INSERT (:Msg {room: 42, body: $t})']);
    expect(bSeen).toEqual(['INSERT (:Msg {room: 7, body: $t})']);
  });

  test('an unscoped subscriber still receives every write (opt-in only)', async () => {
    const s = server(createWriteLog(), 'room');
    const watcher = s.client(); // no scopes → fire hose (back-compat)
    const writer = s.client();
    const seen: string[] = [];
    watcher.subscribeWrites((w) => seen.push(...w.map((x) => x.text)));

    await writer.mutate('INSERT (:Msg {room: 1})');
    await writer.mutate('INSERT (:Msg {room: 2})');

    expect(seen).toEqual(['INSERT (:Msg {room: 1})', 'INSERT (:Msg {room: 2})']);
  });

  test('a write with no scope value reaches a scoped subscriber (fail-open)', async () => {
    // A write that touches no `room`-bearing element can't be scoped, so it is
    // NOT hidden from a scoped subscriber — scope only ever narrows, never drops
    // an unclassifiable write.
    const s = server(createWriteLog(), 'room');
    const a = s.client();
    const writer = s.client();
    const seen: string[] = [];
    a.subscribeWrites((w) => seen.push(...w.map((x) => x.text)), { scopes: ['42'] });

    await writer.mutate('INSERT (:Note {body: $b})', { b: 'unscoped' }); // no room
    await writer.mutate('INSERT (:Msg {room: 42})'); // in scope
    await writer.mutate('INSERT (:Msg {room: 7})'); // out of scope

    expect(seen).toEqual(['INSERT (:Note {body: $b})', 'INSERT (:Msg {room: 42})']);
  });

  test('catch-up: a late subscriber replays the retained tail', async () => {
    const s = server();
    const a = s.client();
    const b = s.client();

    await b.mutate('INSERT (:Widget {id: 1})');
    await b.mutate('INSERT (:Widget {id: 2})');

    // A subscribes AFTER both writes → the host replays them from the log.
    const seen: SyncWrite[] = [];
    a.subscribeWrites((w) => seen.push(...w));

    expect(seen.map((w) => w.text)).toEqual([
      'INSERT (:Widget {id: 1})',
      'INSERT (:Widget {id: 2})',
    ]);
  });

  test('resync: subscribing after the retained tail dropped triggers cold-boot', async () => {
    const s = server(createWriteLog({ capacity: 1 }));
    const a = s.client();
    const b = s.client();

    await b.mutate('INSERT (:Foo)'); // seq 1
    await b.mutate('INSERT (:Bar)'); // seq 2 — ring (cap 1) now holds only seq 2

    let resynced = false;
    const seen: SyncWrite[] = [];
    a.subscribeWrites((w) => seen.push(...w), { onResync: () => (resynced = true) });

    expect(resynced).toBe(true); // since=0 fell off the tail
    expect(seen).toEqual([]);
  });

  test('the live tail cold-boots on a gap or reordered batch', () => {
    const s = server();
    const a = s.client();

    let resynced = false;
    const seen: string[] = [];
    a.subscribeWrites((w) => seen.push(...w.map((x) => x.text)), {
      onResync: () => (resynced = true),
    });

    // Contiguous batches (each `from` chains off the client's cursor, from 0) apply.
    a.receive({ type: 'writes', writes: [{ text: 'INSERT (:A)' }], cursor: 1, from: 0 });
    a.receive({ type: 'writes', writes: [{ text: 'INSERT (:B)' }], cursor: 2, from: 1 });
    expect(seen).toEqual(['INSERT (:A)', 'INSERT (:B)']);
    expect(resynced).toBe(false);

    // A batch that does NOT follow our cursor (from=4, but we're at 2) means a
    // live-tail message was lost or reordered → cold-boot, and the out-of-order
    // writes are NOT silently applied.
    a.receive({ type: 'writes', writes: [{ text: 'INSERT (:D)' }], cursor: 5, from: 4 });
    expect(resynced).toBe(true);
    expect(seen).toEqual(['INSERT (:A)', 'INSERT (:B)']);
  });

  test('reconnect (replay) resumes from the cursor without double-applying', async () => {
    const s = server();
    const a = s.client();
    const b = s.client();
    const seen: string[] = [];
    a.subscribeWrites((w) => seen.push(...w.map((x) => x.text)));

    await b.mutate('INSERT (:Widget {id: 1})');
    expect(seen).toEqual(['INSERT (:Widget {id: 1})']);

    // Simulate reconnect: replay re-subscribes the write stream from the cursor.
    // The cursor is current, so nothing is re-delivered (no double-apply)…
    a.replay();
    expect(seen).toEqual(['INSERT (:Widget {id: 1})']);

    // …and live delivery continues afterward.
    await b.mutate('INSERT (:Widget {id: 2})');
    expect(seen).toEqual(['INSERT (:Widget {id: 1})', 'INSERT (:Widget {id: 2})']);
  });

  test('a live subscriber receives every write regardless of ring capacity', async () => {
    const s = server(createWriteLog({ capacity: 1 })); // a tiny ring
    const a = s.client();
    const b = s.client();
    const seen: string[] = [];
    a.subscribeWrites((w) => seen.push(...w.map((x) => x.text)));

    for (const id of [1, 2, 3, 4]) {
      await b.mutate(`INSERT (:Widget {id: ${id}})`);
    }

    // The ring bounds CATCH-UP, not live delivery — a live subscriber sees them all.
    expect(seen).toEqual([1, 2, 3, 4].map((id) => `INSERT (:Widget {id: ${id}})`));
  });

  test('interleaved writes from multiple writers arrive in seq order', async () => {
    const s = server();
    const watcher = s.client();
    const b = s.client();
    const c = s.client();
    const seen: string[] = [];
    watcher.subscribeWrites((w) => seen.push(...w.map((x) => x.text)));

    await b.mutate('INSERT (:Widget {id: 1})');
    await c.mutate('INSERT (:Widget {id: 2})');
    await b.mutate('INSERT (:Widget {id: 3})');
    await c.mutate('INSERT (:Widget {id: 4})');

    expect(seen).toEqual([1, 2, 3, 4].map((id) => `INSERT (:Widget {id: ${id}})`));
  });

  test('subscribeWrites against a host with no writeLog is a silent no-op', async () => {
    const store = newStore();
    const link: { c?: SyncClient } = {};
    const host = createSyncHost(store, { send: (m) => link.c?.receive(m) }); // NO writeLog
    link.c = createSyncClient({ send: (m) => host.receive(m) });
    const seen: SyncWrite[] = [];
    link.c.subscribeWrites((w) => seen.push(...w));

    await link.c.mutate('INSERT (:Widget {id: 1})'); // the mutate still works…

    expect(seen).toEqual([]); // …but no CDC is delivered
    expect(widgets(store)).toBe(1);
  });

  test("interest routing: only writes touching the client's subscription deps are forwarded", () => {
    const store = newStore();
    const writeLog = createWriteLog();
    const sent: HostMessage[] = [];
    const host = createSyncHost(store, { send: (m) => sent.push(m), writeLog });

    // The client declares a live query over :User (deps ['User']) + opts into CDC.
    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: 'MATCH (u:User) RETURN u.id',
      deps: ['User'],
    });
    host.receive({ type: 'subscribeWrites' });
    sent.length = 0; // drop the setup pushes

    // Another participant commits a User write and a Team write to the shared log.
    const other = 'other-client';
    writeLog.append(other, { text: 'INSERT (:User {id: 1})' }, ['User']);
    writeLog.append(other, { text: 'INSERT (:Team {id: 1})' }, ['Team']);

    const forwarded = sent
      .filter((m): m is WritesMessage => m.type === 'writes')
      .flatMap((m) => m.writes.map((w) => w.text));
    expect(forwarded).toEqual(['INSERT (:User {id: 1})']); // the Team write is filtered out
  });

  test('dedupe: a re-sent write (same req) applies once across a reconnect', () => {
    const store = newStore();
    const dedup = createDedupRegistry();
    const writeLog = createWriteLog();
    const sent1: HostMessage[] = [];
    const sent2: HostMessage[] = [];
    const h1 = createSyncHost(store, { send: (m) => sent1.push(m), dedup, writeLog });
    const h2 = createSyncHost(store, { send: (m) => sent2.push(m), dedup, writeLog });

    const req = 'm-client-1';
    h1.receive({ type: 'mutate', req, text: 'INSERT (:Widget {id: 1})' }); // lands on conn 1
    h2.receive({ type: 'mutate', req, text: 'INSERT (:Widget {id: 1})' }); // replayed on conn 2 (lost ack)

    expect(widgets(store)).toBe(1); // applied exactly once
    expect(writeLog.head()).toBe(1); // and broadcast exactly once (no double CDC)
    expect([...sent1, ...sent2].filter((m) => m.type === 'ack' && m.ok).length).toBe(2); // both acked ok
  });

  test('origin-skip survives reconnect: a client never re-ingests its OWN backlog write', () => {
    const store = newStore();
    const writeLog = createWriteLog();
    const sent1: HostMessage[] = [];
    const sent2: HostMessage[] = [];
    const sentB: HostMessage[] = [];
    const h1 = createSyncHost(store, { send: (m) => sent1.push(m), writeLog });

    const clientId = 'client-A';

    // Client A opts into CDC on connection 1 and commits a write. Its own write is
    // NOT echoed back to it (origin-skip, tagged with A's stable clientId).
    h1.receive({ type: 'subscribeWrites', clientId });
    h1.receive({
      type: 'mutate',
      req: `m-${clientId}-1`,
      text: 'INSERT (:Widget {id: 1})',
      clientId,
    });
    const echoedToA1 = sent1
      .filter((m): m is WritesMessage => m.type === 'writes')
      .flatMap((m) => m.writes.map((w) => w.text));
    expect(echoedToA1).toEqual([]); // own write skipped on its own connection

    // The connection drops BEFORE A advanced its cursor. A re-dials → a FRESH host
    // (which used to mint a new per-connection origin) and resumes from cursor 0.
    const h2 = createSyncHost(store, { send: (m) => sent2.push(m), writeLog });
    h2.receive({ type: 'subscribeWrites', since: 0, clientId });
    const backlogToA2 = sent2
      .filter((m): m is WritesMessage => m.type === 'writes')
      .flatMap((m) => m.writes.map((w) => w.text));
    // The fix: the backlog still skips A's own write, because origin is A's stable
    // clientId, not the (now different) connection. Before, A re-ingested it → the
    // optimistic/authoritative divergence.
    expect(backlogToA2).toEqual([]);

    // A DIFFERENT client B, resuming from 0, DOES receive A's write — the skip is
    // per-client, not "skip everything".
    const hB = createSyncHost(store, { send: (m) => sentB.push(m), writeLog });
    hB.receive({ type: 'subscribeWrites', since: 0, clientId: 'client-B' });
    const backlogToB = sentB
      .filter((m): m is WritesMessage => m.type === 'writes')
      .flatMap((m) => m.writes.map((w) => w.text));
    expect(backlogToB).toEqual(['INSERT (:Widget {id: 1})']);
  });

  test('createSyncClient({ clientId }): origin-skip survives a reconnect end-to-end', () => {
    const s = server();
    const clientId = 'device-42'; // a durable, caller-supplied identity

    // Client A (fixed clientId) opts into CDC and commits — its own write is not
    // echoed to it. The connection drops BEFORE A's cursor advanced.
    const a1 = s.client(clientId);
    expect(a1.clientId).toBe(clientId); // the option is exposed for readback/persistence
    const seenA1: string[] = [];
    a1.subscribeWrites((w) => seenA1.push(...w.map((x) => x.text)));
    // In-process wiring: the write applies + broadcasts synchronously during
    // `receive`, so the CDC assertions below hold without awaiting the ack.
    void a1.mutate('INSERT (:Widget {id: 1})');
    expect(seenA1).toEqual([]); // no self-echo

    // A re-dials: a FRESH host + client, but the SAME persisted clientId, resuming
    // from cursor 0. Its own backlog write must still be skipped (durable
    // origin-skip) — the whole point of a stable clientId across reconnects.
    const a2 = s.client(clientId);
    const seenA2: string[] = [];
    a2.subscribeWrites((w) => seenA2.push(...w.map((x) => x.text)));
    expect(seenA2).toEqual([]);

    // A DIFFERENT client, resuming from 0, DOES receive A's write — the skip is
    // per-identity, not "skip the backlog".
    const other = s.client('someone-else');
    const seenOther: string[] = [];
    other.subscribeWrites((w) => seenOther.push(...w.map((x) => x.text)));
    expect(seenOther).toEqual(['INSERT (:Widget {id: 1})']);
  });

  test('createSyncClient without clientId still mints a unique, readable id', () => {
    const s = server();
    const a = s.client();
    const b = s.client();

    expect(typeof a.clientId).toBe('string');
    expect(a.clientId.length).toBeGreaterThan(0);
    expect(a.clientId).not.toBe(b.clientId); // per-instance, so two clients differ
  });

  test('dedupe: distinct writes both apply; a failed write is not recorded', () => {
    const store = newStore();
    const dedup = createDedupRegistry();
    const sent: HostMessage[] = [];
    const h = createSyncHost(store, { send: (m) => sent.push(m), dedup });

    h.receive({ type: 'mutate', req: 'm-a-1', text: 'INSERT (:Widget {id: 1})' });
    h.receive({ type: 'mutate', req: 'm-a-2', text: 'INSERT (:Widget {id: 2})' });
    expect(widgets(store)).toBe(2); // distinct reqs → both apply

    h.receive({ type: 'mutate', req: 'm-a-3', text: 'NOT VALID GQL' });
    expect(sent.some((m) => m.type === 'ack' && !m.ok)).toBe(true); // failed
    expect(dedup.seen('m-a-3')).toBe(false); // not recorded → a retry can still apply
  });

  test('ephemeral: presence is torn down and broadcast on disconnect', async () => {
    const store = newStore();
    const writeLog = createWriteLog();
    const count = (s: Store, label: string): number =>
      s.mutate((g) => g.query<{ c: number }>(`MATCH (n:${label}) RETURN count(*) AS c`))[0].c;

    // A presenter connection (kept so the test can close it) + a watcher ingesting CDC.
    const linkP: { c?: SyncClient } = {};
    const hostP = createSyncHost(store, { send: (m) => linkP.c?.receive(m), writeLog });
    linkP.c = createSyncClient({ send: (m) => hostP.receive(m) });

    const linkW: { c?: SyncClient } = {};
    const hostW = createSyncHost(store, { send: (m) => linkW.c?.receive(m), writeLog });
    linkW.c = createSyncClient({ send: (m) => hostW.receive(m) });
    const localW = newStore();
    linkW.c.subscribeWrites((writes) => {
      localW.mutate((g) => {
        for (const w of writes) {
          runWrite(g, w);
        }
      });
    });

    // Register the ephemeral cleanup, then set presence (a normal write).
    linkP.c.onDisconnect([{ text: "MATCH (p:Presence {sid: 'x'}) DETACH DELETE p" }]);
    await linkP.c.mutate("INSERT (:Presence {sid: 'x'})");
    expect(count(store, 'Presence')).toBe(1);
    expect(count(localW, 'Presence')).toBe(1); // the watcher saw the presence via CDC

    // Presenter drops → host.close runs the cleanup and broadcasts the removal.
    hostP.close();
    expect(count(store, 'Presence')).toBe(0); // gone on the server
    expect(count(localW, 'Presence')).toBe(0); // and the watcher saw the DELETE via CDC
  });
});

// The ordering/idempotence guard is pure client logic — drive receive() directly,
// so these run without the native lib.
describe('CDC write stream — client ordering guard (transport-free)', () => {
  test('a duplicate or stale batch is ignored; the cursor never regresses', () => {
    const sent: ClientMessage[] = [];
    const client = createSyncClient({ send: (m) => sent.push(m) });
    const seen: string[] = [];
    client.subscribeWrites((w) => seen.push(...w.map((x) => x.text)));

    client.receive({ type: 'writes', writes: [{ text: 'a' }], cursor: 1 });
    client.receive({ type: 'writes', writes: [{ text: 'a-dup' }], cursor: 1 }); // duplicate seq
    client.receive({ type: 'writes', writes: [{ text: 'stale' }], cursor: 0 }); // stale (< cursor)
    client.receive({ type: 'writes', writes: [{ text: 'b' }], cursor: 2 });

    expect(seen).toEqual(['a', 'b']); // duplicate + stale dropped, no double-apply

    // The cursor held at 2 (never regressed) — a resume asks from there.
    client.replay();
    expect(sent.filter((m) => m.type === 'subscribeWrites').at(-1)).toMatchObject({
      type: 'subscribeWrites',
      since: 2,
    });
  });

  test('a resync message fires onResync and moves the resume cursor', () => {
    const sent: ClientMessage[] = [];
    const client = createSyncClient({ send: (m) => sent.push(m) });
    let resynced = false;
    client.subscribeWrites(() => {}, { onResync: () => (resynced = true) });

    client.receive({ type: 'writes', writes: [], cursor: 42, resync: true });

    expect(resynced).toBe(true);
    client.replay();
    expect(sent.filter((m) => m.type === 'subscribeWrites').at(-1)).toMatchObject({
      type: 'subscribeWrites',
      since: 42,
    });
  });

  test('a poison write is isolated: onIngestError fires, receive() never throws, the pump survives', () => {
    const client = createSyncClient({ send: () => {} });
    const applied: string[] = [];
    const errors: unknown[] = [];
    client.subscribeWrites(
      (writes) => {
        for (const w of writes) {
          if (w.text === 'POISON') {
            throw new Error('un-appliable write');
          }

          applied.push(w.text);
        }
      },
      { onIngestError: (e) => errors.push(e) },
    );

    // A good batch applies. A poison batch must NOT escape receive() and wedge the
    // transport — it's surfaced via onIngestError. A later good batch still applies,
    // proving the pump survived (before the fix, the throw killed the message loop).
    expect(() =>
      client.receive({ type: 'writes', writes: [{ text: 'a' }], cursor: 1 }),
    ).not.toThrow();
    expect(() =>
      client.receive({ type: 'writes', writes: [{ text: 'POISON' }], cursor: 2 }),
    ).not.toThrow();
    expect(() =>
      client.receive({ type: 'writes', writes: [{ text: 'b' }], cursor: 3 }),
    ).not.toThrow();

    expect(applied).toEqual(['a', 'b']);
    expect(errors.length).toBe(1);
  });

  test('mutate reqs are globally unique (stable per-client prefix, for dedupe)', () => {
    const sentA: ClientMessage[] = [];
    const sentB: ClientMessage[] = [];
    const a = createSyncClient({ send: (m) => sentA.push(m) });
    const b = createSyncClient({ send: (m) => sentB.push(m) });
    void a.mutate('INSERT (:W)');
    void b.mutate('INSERT (:W)');

    const reqOf = (sent: ClientMessage[]): string =>
      (sent.find((m) => m.type === 'mutate') as { req: string }).req;
    // Two clients issuing "the same" write get distinct reqs → no false dedupe.
    expect(reqOf(sentA)).not.toEqual(reqOf(sentB));
  });
});

// Schema (constraints / validators / invariants / indexes) rides the SAME CDC log
// as data — `applySchema` applies it to the authoritative store and appends a
// structured op, so a replica replaying the stream stays schema-in-lock-step. This
// is what lets a replica derive a keyed `_MERGE` or reject the same writes as the
// source — without it, the replica is missing the constraints. Server-authoritative
// (no CRDT): the store is the single writer; replicas receive, they don't originate.
suite('schema replication over the CDC log', () => {
  /** A replica: pipe the write stream into a fresh local store (what engine.ingest
   *  does). Subscribing with a default cursor replays the full backlog in seq order. */
  const replicaOf = (c: SyncClient): Store => {
    const local = newStore();
    c.subscribeWrites((writes) => {
      local.mutate((g) => {
        for (const w of writes) {
          runWrite(g, w);
        }
      });
    });

    return local;
  };

  const accts = (s: Store): Array<{ email: string; name: string | null }> =>
    s.mutate((g) =>
      g.query<{ email: string; name: string | null }>(
        'MATCH (u:Acct) RETURN u.email AS email, u.name AS name ORDER BY u.email',
      ),
    );

  test("a unique constraint replicates so a replica's keyed _MERGE derives its key", () => {
    const s = server();
    const local = replicaOf(s.client());

    // Before the constraint lands, a keyed upsert on the replica has no key to
    // upsert on → it errors, exactly as on any fresh engine.
    expect(() => local.mutate((g) => g.query("_MERGE (u:Acct {email: 'm@x.io'})"))).toThrow();

    // Publish the schema over the log; the replica ingests + applies it in order.
    applySchema(
      s.store,
      { op: 'createUniqueConstraint', label: 'Acct', key: 'email' },
      { writeLog: s.writeLog },
    );

    // Now the same upsert succeeds on the replica AND is idempotent on the key —
    // the second _MERGE clobbers the payload of the one row rather than inserting.
    local.mutate((g) => {
      g.query("_MERGE (u:Acct {email: 'm@x.io', name: 'A'})");
      g.query("_MERGE (u:Acct {email: 'm@x.io', name: 'B'})");
    });
    expect(accts(local)).toEqual([{ email: 'm@x.io', name: 'B' }]);

    // …and the authoritative store enforces the very same constraint.
    expect(() =>
      s.store.mutate((g) => {
        g.query("INSERT (:Acct {email: 'x@x.io'})");
        g.query("INSERT (:Acct {email: 'x@x.io'})");
      }),
    ).toThrow();
  });

  test('a validator replicates so the replica rejects the same forbidden write', () => {
    const s = server();
    const local = replicaOf(s.client());

    applySchema(
      s.store,
      { op: 'createValidator', label: 'User', varName: 'u', predicate: 'u.age >= 0' },
      { writeLog: s.writeLog },
    );

    // The replica now rejects a write the predicate forbids — same as the source.
    expect(() => local.mutate((g) => g.query('INSERT (:User {age: -1})'))).toThrow();
    // A conforming write still lands.
    local.mutate((g) => g.query('INSERT (:User {age: 5})'));
    expect(
      local.mutate((g) => g.query<{ c: number }>('MATCH (u:User) RETURN count(*) AS c'))[0].c,
    ).toBe(1);
  });

  test('a schema change rejected by existing data throws and never reaches the log', () => {
    const s = server();
    // Seed two Accts sharing an email directly on the store (does NOT log — only a
    // host mutate appends). A unique constraint can't hold over this data.
    s.store.mutate((g) => {
      g.query("INSERT (:Acct {email: 'dup@x.io'})");
      g.query("INSERT (:Acct {email: 'dup@x.io'})");
    });
    const headBefore = s.writeLog.head();

    // Apply-then-log: the store rejects the constraint, so nothing is published —
    // a replica never sees a constraint the source itself couldn't take.
    expect(() =>
      applySchema(
        s.store,
        { op: 'createUniqueConstraint', label: 'Acct', key: 'email' },
        { writeLog: s.writeLog },
      ),
    ).toThrow();
    expect(s.writeLog.head()).toBe(headBefore);
  });

  test('a late-joining replica replays schema THEN data from the backlog in order', async () => {
    const s = server();
    const writer = s.client();

    // Schema first, then a keyed upsert that depends on it — both land in the log.
    applySchema(
      s.store,
      { op: 'createUniqueConstraint', label: 'Acct', key: 'email' },
      { writeLog: s.writeLog },
    );
    await writer.mutate("_MERGE (u:Acct {email: 'k@x.io', name: 'K'})");

    // A replica connects only now and replays the whole backlog from seq 0. If the
    // constraint didn't replay BEFORE the _MERGE, the upsert would have thrown.
    const local = replicaOf(s.client());
    expect(accts(local)).toEqual([{ email: 'k@x.io', name: 'K' }]);
  });

  test('an index declaration replicates (structured op, not write-language text)', () => {
    const s = server();
    const local = replicaOf(s.client());

    applySchema(s.store, { op: 'createVertexIndex', key: 'email' }, { writeLog: s.writeLog });

    // The op reached the replica's engine — the index now exists there too. (A
    // seek-plan speedup isn't observable from results; presence is the contract.)
    expect(local.mutate((g) => g.vertexIndexes())).toContain('email');
    expect(s.store.mutate((g) => g.vertexIndexes())).toContain('email');
  });
});
