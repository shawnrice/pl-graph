// Proves the client registry over a direct in-memory loop to a real host:
// dedupe by (query, params, deps) signature, refcounted wire teardown,
// referentially-stable snapshots with honest complete/error state, and
// promise-shaped one-shots — the full client contract, transport-free.
// Run: bun test packages/sync/src/client.test.ts
import { afterEach, describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { hasErrorCode, ErrorCode } from '@lenke/errors';
import { createStore, graphFromNdjson } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';

import { createSyncClient } from './client.js';
import { createSyncHost } from './host.js';
import type { ClientMessage } from './protocol.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[client.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"b","labels":["Person"],"properties":{"name":"vadas","age":27}}',
].join('\n');

// Track every store so its native graph handle is released after each test; a
// leaked handle otherwise trips the GC-backstop warning when the finalizer runs.
const created: ReturnType<typeof createStore>[] = [];

afterEach(() => {
  for (const store of created.splice(0)) {
    store.free();
  }
});

/** Client ↔ host wired directly — the minimal port. `wire` records traffic. */
const connect = (clientOpts: { maxInactiveQueries?: number } = {}) => {
  const store = createStore(
    graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
  );
  created.push(store);
  const wire: ClientMessage[] = [];
  // Declared before the host exists; the host's status message on attach
  // arrives before the client is constructed, so buffer and replay.
  const buffered: unknown[] = [];
  let deliver: (msg: unknown) => void = (m) => buffered.push(m);
  const host = createSyncHost(store, { send: (m) => deliver(m) });
  const client = createSyncClient({
    send: (m) => {
      wire.push(m);
      host.receive(m);
    },
    ...clientOpts,
  });
  deliver = (m) => client.receive(m);
  buffered.forEach((m) => client.receive(m));

  return { client, host, store, wire };
};

suite('@lenke/sync client · registry semantics', () => {
  test('liveQuery answers with an honest lifecycle: skeleton → complete rows', () => {
    const { client } = connect();
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });

    // Synchronous loop means rows already arrived; but the INITIAL contract
    // is observable through a fresh signature before any push: complete=false.
    const snap = live.getSnapshot();
    expect(snap.complete).toBe(true);
    expect(snap.rows).toHaveLength(2);
    expect(typeof snap.version).toBe('number');
  });

  test('snapshots are referentially stable between pushes and replaced on change', () => {
    const { client } = connect();
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', {
      deps: ['Person', 'name'],
    });
    const stop = live.subscribe(() => {});

    const a = live.getSnapshot();
    expect(live.getSnapshot()).toBe(a);

    void client.mutate('INSERT (:Person {name: $n})', { n: 'zoe' });
    const b = live.getSnapshot();
    expect(b).not.toBe(a);
    expect(b.rows).toHaveLength(3);
    stop();
  });

  test('same signature dedupes to ONE wire subscription; different params do not', () => {
    const { client, wire } = connect();

    const h1 = client.liveQuery('MATCH (p:Person) WHERE p.age >= $min RETURN p.name', {
      deps: null,
      params: { min: 28 },
    });
    const h2 = client.liveQuery('MATCH (p:Person) WHERE p.age >= $min RETURN p.name', {
      deps: null,
      params: { min: 28 },
    });
    const h3 = client.liveQuery('MATCH (p:Person) WHERE p.age >= $min RETURN p.name', {
      deps: null,
      params: { min: 20 },
    });

    expect(h1).toBe(h2); // shared handle
    expect(h3).not.toBe(h1);
    expect(wire.filter((m) => m.type === 'subscribe')).toHaveLength(2);
    expect(client.subscriptionCount()).toBe(2);
    expect(h1.getSnapshot().rows).toHaveLength(1); // marko only
    expect(h3.getSnapshot().rows).toHaveLength(2);
  });

  test('refcounted teardown: wire unsubscribe only when the LAST local subscriber leaves', () => {
    const { client, wire } = connect();
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });

    const stopA = live.subscribe(() => {});
    const stopB = live.subscribe(() => {});

    stopA();
    expect(wire.filter((m) => m.type === 'unsubscribe')).toHaveLength(0);
    expect(client.subscriptionCount()).toBe(1);

    stopB();
    expect(wire.filter((m) => m.type === 'unsubscribe')).toHaveLength(1);
    expect(client.subscriptionCount()).toBe(0);
  });

  test('one-shot query resolves rows; failed mutate rejects with the coded error', async () => {
    const { client } = connect();

    const rows = await client.query('MATCH (p:Person) WHERE p.name = $n RETURN p.age', {
      n: 'vadas',
    });
    expect(rows).toEqual([{ 'p.age': 27 }]);

    await client.mutate('INSERT (:Person {name: $n})', { n: 'carol' });

    expect(client.mutate('NOT GQL AT ALL')).rejects.toThrow();

    try {
      await client.mutate('NOT GQL AT ALL');
    } catch (e) {
      expect(hasErrorCode(e, ErrorCode.Syntax)).toBe(true);
      // The origin's `lenke:` prefix isn't doubled crossing the wire.
      expect((e as Error).message).not.toContain('lenke: lenke:');
    }
  });

  test('pushWrite replicates a whole SyncWrite — text+params, and lang (no gremlin→gql degrade)', async () => {
    const { client, store } = connect();
    const before = store.graph.vertexCount;

    // GQL write: text + params carried through.
    await client.pushWrite({ text: 'INSERT (:Person {name: $n})', params: { n: 'gql-pushed' } });
    expect(
      await client.query('MATCH (p:Person) WHERE p.name = $n RETURN p.name', { n: 'gql-pushed' }),
    ).toEqual([{ 'p.name': 'gql-pushed' }]);

    // Gremlin write: lang carried through, so it runs as a traversal instead of
    // being parsed as GQL (which would park/reject on the wire).
    await client.pushWrite({
      text: "g.addV('Person').property('name', 'gremlin-pushed')",
      lang: 'gremlin',
    });
    expect(
      await client.query('MATCH (p:Person) WHERE p.name = $n RETURN p.name', {
        n: 'gremlin-pushed',
      }),
    ).toEqual([{ 'p.name': 'gremlin-pushed' }]);

    expect(store.graph.vertexCount).toBe(before + 2);
  });

  test('pushWrite refuses a bulk ndjson load — it is never replicated upstream', async () => {
    const { client } = connect();
    const batch = new TextEncoder().encode(
      '{"type":"node","id":"z","labels":["Person"],"properties":{"name":"zed"}}',
    );

    try {
      await client.pushWrite({ text: '', ndjson: batch });

      throw new Error('expected a rejection');
    } catch (e) {
      expect(hasErrorCode(e, ErrorCode.InvalidGraphOp)).toBe(true);
    }
  });

  test('an injection-shaped param stays inert through the whole loop', async () => {
    const { client, store } = connect();
    const before = store.graph.vertexCount;

    const rows = await client.query('MATCH (p:Person) WHERE p.name = $n RETURN p.name', {
      n: "' DETACH DELETE p RETURN 1 //",
    });
    expect(rows).toEqual([]);
    expect(store.graph.vertexCount).toBe(before);
  });

  test('a bad standing query surfaces error on the snapshot and detaches', () => {
    const { client } = connect();
    const live = client.liveQuery('THIS IS NOT GQL', { deps: null });

    const snap = live.getSnapshot();
    expect(snap.error?.code).toBeDefined();
    expect(snap.complete).toBe(false);
    expect(client.subscriptionCount()).toBe(0);
  });

  test('a torn-down handle revives on re-subscribe (StrictMode mount dance)', () => {
    const { client, wire } = connect();
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });

    const stop = live.subscribe(() => {});
    stop(); // refcount → 0 → wire unsubscribe
    expect(client.subscriptionCount()).toBe(0);

    // Re-subscribing the SAME handle re-establishes a fresh wire subscription
    // and keeps receiving pushes.
    const stop2 = live.subscribe(() => {});
    expect(client.subscriptionCount()).toBe(1);
    expect(wire.filter((m) => m.type === 'subscribe')).toHaveLength(2);

    void client.mutate('INSERT (:Person {name: $n})', { n: 'revive-check' });
    expect(live.getSnapshot().rows).toHaveLength(3);
    stop2();
  });

  test('inactive entries retire into a bounded LRU; past the cap they evict', () => {
    const { client } = connect({ maxInactiveQueries: 2 });
    const q = (min: number) =>
      client.liveQuery('MATCH (p:Person) WHERE p.age >= $min RETURN p.name', {
        deps: null,
        params: { min },
      });

    // Three distinct signatures, each subscribed then fully unsubscribed.
    const h1 = q(1);
    h1.subscribe(() => {})();
    const h2 = q(2);
    h2.subscribe(() => {})();
    const h3 = q(3);
    h3.subscribe(() => {})();

    // Cap 2 → the oldest inactive entry (h1's) was dropped, so its retained
    // rows are collectable and a fresh liveQuery mints a NEW canonical handle.
    expect(q(1)).not.toBe(h1);
    // h2/h3 stayed within the cap: still the same warm handles.
    expect(q(2)).toBe(h2);
    expect(q(3)).toBe(h3);
  });

  test('active subscriptions are never evicted, whatever the cap', () => {
    const { client } = connect({ maxInactiveQueries: 1 });
    const active = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });
    const stop = active.subscribe(() => {}); // stays subscribed throughout

    // Churn several inactive signatures well past the cap.
    for (const min of [1, 2, 3]) {
      client
        .liveQuery('MATCH (p:Person) WHERE p.age >= $min RETURN p.name', {
          deps: null,
          params: { min },
        })
        .subscribe(() => {})();
    }

    expect(client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null })).toBe(active);
    expect(client.subscriptionCount()).toBe(1);
    stop();
  });

  test('an evicted stale handle still revives and re-registers', () => {
    const { client } = connect({ maxInactiveQueries: 1 });
    const q = (min: number) =>
      client.liveQuery('MATCH (p:Person) WHERE p.age >= $min RETURN p.name', {
        deps: null,
        params: { min },
      });
    const h1 = q(1);
    h1.subscribe(() => {})(); // retire
    q(2).subscribe(() => {})(); // retires too; cap 1 evicts h1's entry

    // The app kept h1 around: re-subscribing still works (fresh wire sub, live
    // rows) and re-registers it, so dedupe finds it again.
    const stop = h1.subscribe(() => {});
    expect(h1.getSnapshot().rows).toHaveLength(2);
    expect(q(1)).toBe(h1);
    stop();
  });

  test('deps order does not defeat dedupe (deps are a set)', () => {
    const { client, wire } = connect();

    const a = client.liveQuery('MATCH (p:Person) RETURN p.name', {
      deps: ['Person', 'name'],
    });
    const b = client.liveQuery('MATCH (p:Person) RETURN p.name', {
      deps: ['name', 'Person'], // same tokens, different declaration order
    });

    expect(b).toBe(a); // one entry, one wire subscription
    expect(wire.filter((m) => m.type === 'subscribe')).toHaveLength(1);

    // But null (recompute-always) and [] (never) stay distinct signatures.
    const always = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });
    const never = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: [] });
    expect(always).not.toBe(never);
    expect(always).not.toBe(a);
  });

  test('formatting differences do not defeat dedupe; values and case do', () => {
    const { client, wire } = connect();

    const canonicalForm = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });
    // Same query, reformatted: extra spaces, newlines, a trailing comment.
    const reformatted = client.liveQuery('MATCH  (p:Person)\n\tRETURN p.name  // rows', {
      deps: null,
    });
    const blockComment = client.liveQuery('MATCH /* all people */ (p:Person) RETURN p.name', {
      deps: null,
    });

    expect(reformatted).toBe(canonicalForm);
    expect(blockComment).toBe(canonicalForm);
    expect(wire.filter((m) => m.type === 'subscribe')).toHaveLength(1);

    // Whitespace INSIDE a string literal is the value — never normalized.
    const spaced = client.liveQuery("MATCH (p:Person) WHERE p.name = 'a b' RETURN p", {
      deps: null,
    });
    const doubleSpaced = client.liveQuery("MATCH (p:Person) WHERE p.name = 'a  b' RETURN p", {
      deps: null,
    });
    expect(doubleSpaced).not.toBe(spaced);

    // Case is NOT folded: keywords are case-insensitive but labels are not,
    // so folding could merge different queries — the miss is the safe side.
    const lower = client.liveQuery('match (p:Person) RETURN p.name', { deps: null });
    expect(lower).not.toBe(canonicalForm);
  });

  test('gremlin text normalizes whitespace but never treats // as a comment', () => {
    const { client, wire } = connect();
    const before = wire.filter((m) => m.type === 'subscribe').length;

    // Runs collapse to ONE space (never zero — deleting whitespace could fuse
    // tokens), so single-space and multi-space/newline forms share an entry.
    const a = client.liveQuery("g.V() .hasLabel('Person') .values('name')", {
      deps: null,
      lang: 'gremlin',
    });
    const b = client.liveQuery("g.V()  .hasLabel('Person')\n   .values('name')", {
      deps: null,
      lang: 'gremlin',
    });
    expect(b).toBe(a);
    expect(wire.filter((m) => m.type === 'subscribe')).toHaveLength(before + 1);

    // '//' inside gremlin (e.g. in a path-ish string or future syntax) must
    // not be stripped as a GQL comment — these are different traversals.
    const withSlashes = client.liveQuery("g.V().has('url', 'https://x')", {
      deps: null,
      lang: 'gremlin',
    });
    const without = client.liveQuery("g.V().has('url', 'https:')", {
      deps: null,
      lang: 'gremlin',
    });
    expect(withSlashes).not.toBe(without);
  });

  test('a windowed liveQuery gets just the page; a different window is a distinct query', () => {
    const { client, wire } = connect();
    const q = 'MATCH (p:Person) RETURN p.name ORDER BY p.name';

    const page0 = client.liveQuery(q, { deps: ['Person'], window: { offset: 0, limit: 1 } });
    page0.subscribe(() => {});
    expect(page0.getSnapshot().rows).toHaveLength(1); // the page, not all 2 rows

    const page1 = client.liveQuery(q, { deps: ['Person'], window: { offset: 1, limit: 1 } });
    page1.subscribe(() => {});
    expect(page1).not.toBe(page0); // a different window → a distinct standing query
    expect(page1.getSnapshot().rows).toHaveLength(1);
    expect(page1.getSnapshot().rows[0]).not.toEqual(page0.getSnapshot().rows[0]); // scrolled

    // Two separate wire subscriptions (one per window).
    expect(wire.filter((m) => m.type === 'subscribe')).toHaveLength(2);

    // Same window de-dupes to the same handle (one wire sub).
    const again = client.liveQuery(q, { deps: ['Person'], window: { offset: 0, limit: 1 } });
    expect(again).toBe(page0);
  });

  test('status handshake is captured', async () => {
    const { client } = connect();
    await Promise.resolve(); // the host announces status on a microtask
    expect(client.getStatus()).toEqual({ pendingWrites: 0 });
  });

  test('onStatus wakes subscribers on each push; getStatus is a stable ref between', () => {
    const client = createSyncClient({ send: () => {} });
    let calls = 0;
    const stop = client.onStatus(() => {
      calls += 1;
    });

    client.receive({ type: 'status', pendingWrites: 0, protocol: 1 });
    const first = client.getStatus();
    expect(calls).toBe(1);
    expect(first).toEqual({ pendingWrites: 0 });
    expect(client.getStatus()).toBe(first); // no new object between pushes (useSyncExternalStore-safe)

    client.receive({ type: 'status', pendingWrites: 2, protocol: 1 });
    expect(calls).toBe(2);
    expect(client.getStatus()).toEqual({ pendingWrites: 2 });

    stop();
    client.receive({ type: 'status', pendingWrites: 5, protocol: 1 });
    expect(calls).toBe(2); // unsubscribed — no further wakes
  });

  test('keyed diffs apply as patch/remove/order and keep unchanged-row identity', () => {
    const wire: ClientMessage[] = [];
    const client = createSyncClient({ send: (m) => wire.push(m) });
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name, p.age', {
      deps: null,
      key: 'p.name',
    });
    live.subscribe(() => {});
    const { sub } = wire.find((m) => m.type === 'subscribe') as { sub: string };

    // Initial full diff: every row a patch, in order.
    client.receive({
      type: 'rows',
      sub,
      complete: true,
      version: 1,
      patch: [
        { key: 'marko', set: { 'p.name': 'marko', 'p.age': 29 } },
        { key: 'vadas', set: { 'p.name': 'vadas', 'p.age': 27 } },
      ],
      order: ['marko', 'vadas'],
    });
    expect(live.getSnapshot().rows).toEqual([
      { 'p.name': 'marko', 'p.age': 29 },
      { 'p.name': 'vadas', 'p.age': 27 },
    ]);
    const [marko] = live.getSnapshot().rows;

    // A lone cell change to vadas (no order): marko keeps its object identity.
    client.receive({
      type: 'rows',
      sub,
      complete: true,
      version: 2,
      patch: [{ key: 'vadas', set: { 'p.age': 28 } }],
    });
    expect(live.getSnapshot().rows).toEqual([
      { 'p.name': 'marko', 'p.age': 29 },
      { 'p.name': 'vadas', 'p.age': 28 },
    ]);
    expect(live.getSnapshot().rows[0]).toBe(marko);

    // Insert with a new order: marko is still the same object across the reorder.
    client.receive({
      type: 'rows',
      sub,
      complete: true,
      version: 3,
      patch: [{ key: 'aaron', set: { 'p.name': 'aaron', 'p.age': 40 } }],
      order: ['aaron', 'marko', 'vadas'],
    });
    expect(live.getSnapshot().rows.map((r) => r['p.name'])).toEqual(['aaron', 'marko', 'vadas']);
    expect(live.getSnapshot().rows[1]).toBe(marko);

    // Remove vadas.
    client.receive({
      type: 'rows',
      sub,
      complete: true,
      version: 4,
      remove: ['vadas'],
      order: ['aaron', 'marko'],
    });
    expect(live.getSnapshot().rows.map((r) => r['p.name'])).toEqual(['aaron', 'marko']);

    // A completeness-only push (no ops) keeps the same rows array reference.
    const rowsRef = live.getSnapshot().rows;
    client.receive({ type: 'rows', sub, complete: false, version: 5 });
    expect(live.getSnapshot().rows).toBe(rowsRef);
    expect(live.getSnapshot().complete).toBe(false);
  });

  test('reconnect resume: a re-push keeps unchanged-row identity, and updates/adds/drops', () => {
    const wire: ClientMessage[] = [];
    const client = createSyncClient({ send: (m) => wire.push(m) });
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name, p.age', {
      deps: null,
      key: 'p.name',
    });
    live.subscribe(() => {});
    const sub1 = (wire.find((m) => m.type === 'subscribe') as { sub: string }).sub;

    client.receive({
      type: 'rows',
      sub: sub1,
      complete: true,
      version: 1,
      patch: [
        { key: 'marko', set: { 'p.name': 'marko', 'p.age': 29 } },
        { key: 'vadas', set: { 'p.name': 'vadas', 'p.age': 27 } },
        { key: 'zoe', set: { 'p.name': 'zoe', 'p.age': 31 } },
      ],
      order: ['marko', 'vadas', 'zoe'],
    });
    const [marko, vadas] = live.getSnapshot().rows;

    // Reconnect: the client re-subscribes (same sub id) and KEEPS its base.
    client.replay();
    const sub2 = wire.filter((m) => m.type === 'subscribe').at(-1)?.sub;
    expect(sub2).toBe(sub1);

    // The fresh host re-pushes the current world as full patches + order (no
    // removes): marko unchanged, vadas now 28, zoe gone, carol new.
    client.receive({
      type: 'rows',
      sub: sub1,
      complete: true,
      version: 2,
      patch: [
        { key: 'marko', set: { 'p.name': 'marko', 'p.age': 29 } },
        { key: 'vadas', set: { 'p.name': 'vadas', 'p.age': 28 } },
        { key: 'carol', set: { 'p.name': 'carol', 'p.age': 40 } },
      ],
      order: ['carol', 'marko', 'vadas'],
    });

    const { rows } = live.getSnapshot();
    expect(rows.map((r) => r['p.name'])).toEqual(['carol', 'marko', 'vadas']); // zoe dropped
    expect(rows.find((r) => r['p.name'] === 'marko')).toBe(marko); // identity survived reconnect
    expect(rows.find((r) => r['p.name'] === 'vadas')).not.toBe(vadas); // changed → new object
    expect(rows.find((r) => r['p.name'] === 'vadas')).toEqual({ 'p.name': 'vadas', 'p.age': 28 });
  });

  test('reconnect to an empty result: an authoritative empty order clears stale rows', () => {
    const wire: ClientMessage[] = [];
    const client = createSyncClient({ send: (m) => wire.push(m) });
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null, key: 'p.name' });
    live.subscribe(() => {});
    const { sub } = wire.find((m) => m.type === 'subscribe') as { sub: string };

    client.receive({
      type: 'rows',
      sub,
      complete: true,
      version: 1,
      patch: [
        { key: 'a', set: { 'p.name': 'a' } },
        { key: 'b', set: { 'p.name': 'b' } },
      ],
      order: ['a', 'b'],
    });
    expect(live.getSnapshot().rows).toHaveLength(2);

    // Reconnect; the fresh host finds an empty result and (forceOrder) sends
    // order: [] with no patch/remove. The client must drop the stale rows.
    // (The host-side production of that order is covered in host.test.ts; here
    // we verify only that the client applies an empty order by pruning.)
    client.replay();
    client.receive({ type: 'rows', sub, complete: true, version: 2, order: [] });
    expect(live.getSnapshot().rows).toEqual([]);
  });

  test('reconnect while still loading keeps warm rows (incomplete first push, no order)', () => {
    const wire: ClientMessage[] = [];
    const client = createSyncClient({ send: (m) => wire.push(m) });
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null, key: 'p.name' });
    live.subscribe(() => {});
    const { sub } = wire.find((m) => m.type === 'subscribe') as { sub: string };

    client.receive({
      type: 'rows',
      sub,
      complete: true,
      version: 1,
      patch: [{ key: 'a', set: { 'p.name': 'a' } }],
      order: ['a'],
    });
    const warm = live.getSnapshot().rows;
    expect(warm).toHaveLength(1);

    // Reconnect to a host still loading: empty-for-now + incomplete + NO order
    // (the host does not force one while incomplete). The warm row must stay.
    client.replay();
    client.receive({ type: 'rows', sub, complete: false, version: 2 });
    expect(live.getSnapshot().rows).toBe(warm); // same reference — not blanked
    expect(live.getSnapshot().complete).toBe(false);
  });

  test('an arrow result that fails to decode rejects the query promise (no hang)', async () => {
    const wire: ClientMessage[] = [];
    const client = createSyncClient({ send: (m) => wire.push(m) });
    const pending = client.query('MATCH (p:Person) RETURN p.name', undefined, { format: 'arrow' });
    const { req } = wire.find((m) => m.type === 'query') as { req: string };

    // A JSON transport would deliver the Uint8Array as a plain object — decode
    // must reject the promise, not throw out of receive() and hang it.
    client.receive({ type: 'result', req, arrow: { 0: 65, 1: 66 } as unknown as Uint8Array });
    let error: unknown;
    await pending.catch((e: unknown) => {
      error = e;
    });
    expect(String(error)).toMatch(/arrow/i);
  });

  test('keyed round-trip over a real host: a cell edit updates rows in place', () => {
    const { client, store } = connect();
    const live = client.liveQuery('MATCH (p:Person) RETURN p.name, p.age ORDER BY p.name', {
      deps: null,
      key: 'p.name',
    });
    live.subscribe(() => {});
    expect(live.getSnapshot().rows).toEqual([
      { 'p.name': 'marko', 'p.age': 29 },
      { 'p.name': 'vadas', 'p.age': 27 },
    ]);
    const [marko] = live.getSnapshot().rows;

    // A write to the store fans out as a keyed diff; the client applies it.
    store.mutate((g) =>
      g.query('MATCH (p:Person) WHERE p.name = $n SET p.age = $a', { n: 'vadas', a: 28 }),
    );
    expect(live.getSnapshot().rows).toEqual([
      { 'p.name': 'marko', 'p.age': 29 },
      { 'p.name': 'vadas', 'p.age': 28 },
    ]);
    expect(live.getSnapshot().rows[0]).toBe(marko); // unchanged-row identity survives the round-trip
  });

  test('gremlin() round-trips a traversal over a real host and resolves its values', async () => {
    const { client } = connect();

    expect(await client.gremlin('g.V().count()')).toEqual([2]);

    const names = await client.gremlin("g.V().values('name')");
    expect([...(names as string[])].sort()).toEqual(['marko', 'vadas']);
  });

  test('query with format arrow crosses columnar and decodes to identical rows', async () => {
    const { client, wire } = connect();
    const q = 'MATCH (p:Person) RETURN p.name, p.age ORDER BY p.name';

    const arrowRows = await client.query(q, undefined, { format: 'arrow' });
    expect(arrowRows).toEqual(await client.query(q)); // byte-for-byte the JSON result

    const sent = wire.find((m) => m.type === 'query' && m.format === 'arrow');
    expect(sent).toBeDefined(); // the request really asked for arrow
  });

  test('client.gremlin as a tagged template escapes interpolations — injection stays inert', async () => {
    const { client, store } = connect();
    const before = store.graph.vertexCount;

    const evil = "marko'); g.V().drop(); //";
    const rows = await client.gremlin`g.V().has('name', ${evil}).values('name')`;

    expect(rows).toEqual([]); // one literal string — matches nothing
    expect(store.graph.vertexCount).toBe(before); // the graph was NOT dropped
    expect(await client.gremlin`g.V().has('name', ${'marko'}).count()`).toEqual([1]);
  });

  test('a Gremlin live query exposes values on its snapshot and updates on change', () => {
    const { client, store } = connect();
    const live = client.liveQuery('g.V().count()', { deps: ['Person'], lang: 'gremlin' });
    live.subscribe(() => {});

    expect(live.getSnapshot().values).toEqual([2]);
    expect(live.getSnapshot().rows).toEqual([]); // rows stays empty for a Gremlin query

    // A relevant write re-runs the standing traversal; values reflect it.
    store.mutate((g) => g.query("INSERT (:Person {name: 'carol'})"));
    expect(live.getSnapshot().values).toEqual([3]);
  });

  test('close unsubscribes everything and rejects pending requests', async () => {
    const wire: ClientMessage[] = [];
    // A black-hole transport: nothing ever answers, so requests stay pending.
    const client = createSyncClient({ send: (m) => wire.push(m) });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: null });
    live.subscribe(() => {});
    expect(live.getSnapshot().complete).toBe(false); // INITIAL — nothing answered

    const inflight = client.mutate('INSERT (:Person {name: $n})', { n: 'x' });
    client.close();

    expect(inflight).rejects.toThrow(/client closed/);
    expect(wire.filter((m) => m.type === 'unsubscribe')).toHaveLength(1);
    expect(client.subscriptionCount()).toBe(0);
  });
});
