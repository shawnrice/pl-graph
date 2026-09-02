// Proves the live-query host speaks protocol v1 correctly over a bare
// send/receive pair (transport-free): subscribe pushes now and on change,
// epoch gating suppresses irrelevant pushes, re-subscribe replaces, mutate
// acks and fans out to every host on the store, and errors ride the coded
// wire shape. Run: bun test packages/sync/src/host.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson, type Store } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';

import { createSyncHost } from './host.js';
import type { HostMessage, RowsMessage } from './protocol.js';

// Host-specific shared-library extension; `build:rust` emits this platform's.
const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;

// Built by `bun run build:rust`, not the test — skip cleanly when it's absent.
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[host.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON = [
  '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"marko","age":29}}',
  '{"type":"node","id":"b","labels":["Person"],"properties":{"name":"vadas","age":27}}',
  '{"type":"edge","id":"e0","from":"a","to":"b","labels":["KNOWS"],"properties":{}}',
].join('\n');

const newStore = (): Store => {
  const backend = createFfiEngineBackend(LIB);

  return createStore(graphFromNdjson(backend, new TextEncoder().encode(NDJSON)));
};

/** A host wired to a capture buffer — the minimal "connection". */
const attach = (store: Store) => {
  const sent: HostMessage[] = [];
  const host = createSyncHost(store, { send: (m) => sent.push(m) });

  return { host, sent, take: () => sent.splice(0, sent.length) };
};

const rowsOf = (m: HostMessage): RowsMessage => {
  expect(m.type).toBe('rows');

  return m as RowsMessage;
};

suite('@lenke/sync host · protocol v1', () => {
  test('announces status on attach', async () => {
    const { sent } = attach(newStore());

    // Status is announced on a microtask (not synchronously in the constructor)
    // so `const host = …(send: m => client.receive(m)); const client = …` — where
    // `client` is assigned after the host — can't call `send` before it exists.
    await Promise.resolve();
    expect(sent[0]).toEqual({ type: 'status', pendingWrites: 0, protocol: 1 });
  });

  test('safe when the host is constructed BEFORE its client (deferred status send)', async () => {
    // The natural wiring closes `send` over a `client` assigned after the host.
    // A synchronous status send in the constructor would call `client.receive`
    // before it exists (throw); deferring one microtask makes this order work.
    let client: { receive: (m: unknown) => void } | null = null;
    const seen: unknown[] = [];
    const host = createSyncHost(newStore(), { send: (m) => client!.receive(m) });
    client = { receive: (m) => seen.push(m) }; // assigned AFTER the host — no crash

    expect(seen).toHaveLength(0); // nothing sent synchronously
    await Promise.resolve();
    expect(seen[0]).toMatchObject({ type: 'status' }); // arrives next microtask
    host.close();
  });

  test('subscribe answers immediately with rows + version + complete', () => {
    const { host, take } = attach(newStore());
    take();

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: 'MATCH (p:Person) RETURN p.name ORDER BY p.name',
    });

    const [msg] = take();
    const rows = rowsOf(msg);
    expect(rows.sub).toBe('s1');
    expect(rows.complete).toBe(true);
    expect(typeof rows.version).toBe('number');
    expect(rows.rows?.map((r) => r['p.name'] ?? Object.values(r)[0])).toEqual(['marko', 'vadas']);
  });

  test('a relevant mutation pushes fresh rows; an irrelevant one stays silent', () => {
    const store = newStore();
    const { host, take } = attach(store);

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: 'MATCH (p:Person) RETURN p.name',
      deps: ['Person', 'name'],
    });
    take();

    // Irrelevant: touches `age`, not `name` — epoch gating keeps the wire quiet.
    host.receive({
      type: 'mutate',
      req: 'm1',
      text: "MATCH (p:Person) WHERE p.name = 'marko' SET p.age = 30",
    });
    let msgs = take();
    expect(msgs).toEqual([{ type: 'ack', req: 'm1', ok: true }]);

    // Relevant: a new Person name — the subscription hears it.
    host.receive({ type: 'mutate', req: 'm2', text: "INSERT (:Person {name: 'zoe'})" });
    msgs = take();
    expect(msgs.map((m) => m.type).sort()).toEqual(['ack', 'rows']);
    const rows = rowsOf(msgs.find((m) => m.type === 'rows')!);
    expect(rows.rows).toHaveLength(3);
  });

  test('a legacy pre-`lang` mutate ({gql}) still applies (wire-skew shim)', () => {
    // A stale tab against an upgraded SharedWorker (or an old app build against
    // a new server) still sends the write text under `gql` — honor it.
    const store = newStore();
    const { host, take } = attach(store);
    take();

    host.receive({ type: 'mutate', req: 'old', gql: "INSERT (:Person {name: 'legacy'})" });

    expect(take()).toEqual([{ type: 'ack', req: 'old', ok: true }]);
    expect(store.graph.vertexCount).toBe(3);
  });

  test('a mutate with no query text at all acks a coded error, not a crash', () => {
    const { host, take } = attach(newStore());
    take();

    host.receive({ type: 'mutate', req: 'bad' });

    const [ack] = take();
    expect(ack).toMatchObject({ type: 'ack', req: 'bad', ok: false });
    expect((ack as { error?: { code: string } }).error?.code).toBe('E_INVALID_SHAPE');
  });

  test('one connection’s write fans out to another host on the same store', () => {
    const store = newStore();
    const alice = attach(store);
    const bob = attach(store);

    alice.host.receive({ type: 'subscribe', sub: 'a', query: 'MATCH (p:Person) RETURN p.name' });
    alice.take();
    bob.take();

    bob.host.receive({ type: 'mutate', req: 'w', text: "INSERT (:Person {name: 'carol'})" });

    expect(bob.take()).toEqual([{ type: 'ack', req: 'w', ok: true }]);
    const pushed = rowsOf(alice.take()[0]);
    expect(pushed.rows).toHaveLength(3);
  });

  test('unsubscribe stops pushes; close drops everything', () => {
    const store = newStore();
    const { host, take } = attach(store);

    host.receive({ type: 'subscribe', sub: 's1', query: 'MATCH (p:Person) RETURN p.name' });
    take();
    expect(host.subscriptionCount()).toBe(1);

    host.receive({ type: 'unsubscribe', sub: 's1' });
    store.mutate((g) => g.query("INSERT (:Person {name: 'dan'})"));
    expect(take()).toEqual([]);
    expect(host.subscriptionCount()).toBe(0);

    host.receive({ type: 'subscribe', sub: 's2', query: 'MATCH (p:Person) RETURN p.name' });
    host.close();
    expect(host.subscriptionCount()).toBe(0);
  });

  test('re-subscribing the same sub replaces the standing query', () => {
    const { host, take } = attach(newStore());

    host.receive({ type: 'subscribe', sub: 's', query: 'MATCH (p:Person) RETURN p.name' });
    host.receive({ type: 'subscribe', sub: 's', query: 'MATCH (p:Person) RETURN p.age' });
    expect(host.subscriptionCount()).toBe(1);
    take();

    host.receive({
      type: 'mutate',
      req: 'm',
      text: "MATCH (p:Person) WHERE p.name = 'marko' SET p.age = 31",
    });
    const rows = rowsOf(take().find((m) => m.type === 'rows')!);
    expect(JSON.stringify(rows.rows)).toContain('31');
  });

  test('one-shot query answers result once and never subscribes', () => {
    const { host, take } = attach(newStore());
    take();

    host.receive({
      type: 'query',
      req: 'q1',
      query: 'MATCH (p:Person) RETURN p.name ORDER BY p.name',
    });

    const msgs = take();
    expect(msgs).toHaveLength(1);
    expect(msgs[0].type).toBe('result');
    expect(host.subscriptionCount()).toBe(0);
  });

  test('a bad subscription query closes with a coded rows error', () => {
    const { host, take } = attach(newStore());
    take();

    host.receive({ type: 'subscribe', sub: 'bad', query: 'THIS IS NOT GQL' });

    const rows = rowsOf(take()[0]);
    expect(rows.error?.code).toBeDefined();
    expect(rows.rows).toBeUndefined();
    expect(host.subscriptionCount()).toBe(0);
  });

  test('a failed mutation acks ok:false with the stable code', () => {
    const { host, take } = attach(newStore());
    take();

    host.receive({ type: 'mutate', req: 'm', text: 'ALSO NOT GQL' });

    const [ack] = take();
    expect(ack).toMatchObject({ type: 'ack', req: 'm', ok: false });
    expect((ack as { error?: { code: string } }).error?.code).toBeDefined();
  });

  test('params bind on subscribe, mutate, and one-shot query', () => {
    const { host, take } = attach(newStore());
    take();

    // A parameterized standing query: bindings are part of its identity.
    host.receive({
      type: 'subscribe',
      sub: 'adults',
      query: 'MATCH (p:Person) WHERE p.age >= $min RETURN p.name',
      deps: ['Person', 'age', 'name'],
      params: { min: 28 },
    });
    expect(rowsOf(take()[0]).rows).toHaveLength(1); // only marko (29)

    // A parameterized mutation: the value rides params, not spliced GQL.
    host.receive({
      type: 'mutate',
      req: 'm',
      text: 'INSERT (:Person {name: $n, age: $a})',
      params: { n: 'zoe', a: 31 },
    });
    const msgs = take();
    expect(msgs.find((m) => m.type === 'ack')).toMatchObject({ ok: true });
    expect(rowsOf(msgs.find((m) => m.type === 'rows')!).rows).toHaveLength(2);

    // One-shot with bindings.
    host.receive({
      type: 'query',
      req: 'q',
      query: 'MATCH (p:Person) WHERE p.name = $n RETURN p.age',
      params: { n: 'zoe' },
    });
    expect(take()[0]).toMatchObject({ type: 'result', req: 'q', rows: [{ 'p.age': 31 }] });
  });

  test('injection-shaped param values are inert over the wire', () => {
    const store = newStore();
    const { host, take } = attach(store);
    take();
    const before = store.graph.vertexCount;

    host.receive({
      type: 'query',
      req: 'q',
      query: 'MATCH (p:Person) WHERE p.name = $n RETURN p.name',
      params: { n: "' DETACH DELETE p RETURN 1 //" },
    });

    expect(take()[0]).toMatchObject({ type: 'result', req: 'q', rows: [] });
    expect(store.graph.vertexCount).toBe(before);
  });

  test('unknown message tags are ignored (forward-compat)', () => {
    const { host, take } = attach(newStore());
    take();

    host.receive({ type: 'hello-from-the-future', anything: true });
    host.receive(null);
    host.receive('nonsense');

    expect(take()).toEqual([]);
  });

  test('a keyed subscription pushes patch+order, then a lone patch on a cell change', () => {
    const store = newStore();
    const { host, take } = attach(store);
    take(); // drain the attach status

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: 'MATCH (p:Person) RETURN p.name, p.age ORDER BY p.name',
      key: 'p.name',
    });

    // Initial keyed push: every row as a full patch, the key order, no `rows`.
    const first = rowsOf(take()[0]);
    expect(first.rows).toBeUndefined();
    expect(first.order).toEqual(['marko', 'vadas']);
    expect(first.patch).toEqual([
      { key: 'marko', set: { 'p.name': 'marko', 'p.age': 29 } },
      { key: 'vadas', set: { 'p.name': 'vadas', 'p.age': 27 } },
    ]);
    expect(first.remove).toBeUndefined();

    // One cell moves: exactly one patch of the changed column — no order, no full rows.
    host.receive({
      type: 'mutate',
      req: 'w1',
      text: 'MATCH (p:Person) WHERE p.name = $n SET p.age = $a',
      params: { n: 'marko', a: 30 },
    });
    const push = rowsOf(take().find((m) => m.type === 'rows') as HostMessage);
    expect(push.patch).toEqual([{ key: 'marko', set: { 'p.age': 30 } }]);
    expect(push.order).toBeUndefined();
    expect(push.remove).toBeUndefined();
    expect(push.rows).toBeUndefined();
  });

  test('keyed diffs carry remove when a row leaves the result, and order when membership moves', () => {
    const store = newStore();
    const { host, take } = attach(store);
    take();

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: 'MATCH (p:Person) WHERE p.age >= $min RETURN p.name, p.age ORDER BY p.name',
      params: { min: 27 },
      key: 'p.name',
    });
    expect(rowsOf(take()[0]).order).toEqual(['marko', 'vadas']);

    // A qualifying insert that sorts first → its full patch + the new order.
    host.receive({
      type: 'mutate',
      req: 'w1',
      text: 'INSERT (:Person {name: $n, age: $a})',
      params: { n: 'aaron', a: 40 },
    });
    const ins = rowsOf(take().find((m) => m.type === 'rows') as HostMessage);
    expect(ins.patch).toEqual([{ key: 'aaron', set: { 'p.name': 'aaron', 'p.age': 40 } }]);
    expect(ins.order).toEqual(['aaron', 'marko', 'vadas']);
    expect(ins.remove).toBeUndefined();

    // Push vadas below the threshold → it leaves the result → remove + order, no patch.
    host.receive({
      type: 'mutate',
      req: 'w2',
      text: 'MATCH (p:Person) WHERE p.name = $n SET p.age = $a',
      params: { n: 'vadas', a: 10 },
    });
    const del = rowsOf(take().find((m) => m.type === 'rows') as HostMessage);
    expect(del.remove).toEqual(['vadas']);
    expect(del.order).toEqual(['aaron', 'marko']);
    expect(del.patch).toBeUndefined();
  });

  test('a one-shot Gremlin query answers with values, not rows', () => {
    const { host, take } = attach(newStore());
    take(); // drain status

    host.receive({ type: 'query', req: 'g1', query: 'g.V().count()', lang: 'gremlin' });
    expect(take()[0]).toEqual({ type: 'result', req: 'g1', values: [2] });

    host.receive({
      type: 'query',
      req: 'g2',
      query: "g.V().has('name','marko').out('KNOWS').values('name')",
      lang: 'gremlin',
    });
    expect(take()[0]).toEqual({ type: 'result', req: 'g2', values: ['vadas'] });
  });

  test('a keyed subscription over an empty result sends an authoritative empty order', () => {
    const { host, take } = attach(newStore());
    take(); // drain status

    // A query that matches nothing — the fresh subscription's first push must
    // still tell the client "the current set is empty" (order: []), so a client
    // holding stale rows from before a reconnect prunes them.
    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: "MATCH (p:Person) WHERE p.name = 'nobody' RETURN p.name",
      key: 'p.name',
      deps: ['Person', 'name'],
    });

    const first = rowsOf(take()[0]);
    expect(first.order).toEqual([]); // authoritative empty set (forced on first push)
    expect(first.patch).toBeUndefined();
    expect(first.rows).toBeUndefined();
  });

  test('an INCOMPLETE empty keyed first push does not force order (warm rows preserved)', () => {
    const sent: HostMessage[] = [];
    // isComplete: false → the reconnected host is still loading its scope; an
    // empty-for-now first push must NOT force order, or it would blank the
    // client's warm rows.
    const host = createSyncHost(newStore(), { send: (m) => sent.push(m), isComplete: () => false });
    sent.length = 0; // drain the attach status

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: "MATCH (p:Person) WHERE p.name = 'nobody' RETURN p.name",
      key: 'p.name',
      deps: ['Person', 'name'],
    });

    const first = rowsOf(sent.find((m) => m.type === 'rows') as HostMessage);
    expect(first.complete).toBe(false);
    expect(first.order).toBeUndefined(); // NOT forced while incomplete
    expect(first.patch).toBeUndefined();
  });

  test('an incomplete→complete keyed transition emits the authoritative empty order', () => {
    const sent: HostMessage[] = [];
    let loaded = false; // the scope is still filling on (re)subscribe
    const host = createSyncHost(newStore(), {
      send: (m) => sent.push(m),
      isComplete: () => loaded,
    });
    sent.length = 0; // drain the attach status

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: "MATCH (p:Person) WHERE p.name = 'nobody' RETURN p.name",
      key: 'p.name',
      deps: ['Person', 'name'],
    });

    // First push: still loading → incomplete, no order (client keeps warm rows).
    const first = rowsOf(sent.find((m) => m.type === 'rows') as HostMessage);
    expect(first.complete).toBe(false);
    expect(first.order).toBeUndefined();
    sent.length = 0;

    // The load lands on an EMPTY scope: `complete` flips true, rows ref
    // unchanged. The FIRST complete push must still ship the authoritative
    // (empty) order so a client holding warm rows across a reconnect prunes
    // them — the transition the old `s.last === null` gate missed entirely.
    loaded = true;
    host.refresh();

    const second = rowsOf(sent.find((m) => m.type === 'rows') as HostMessage);
    expect(second.complete).toBe(true);
    expect(second.order).toEqual([]); // authoritative empty set
    expect(second.patch).toBeUndefined();
  });

  test('a mutating Gremlin fans out to a subscriber (runs through store.mutate)', () => {
    const store = newStore();
    const { host, take } = attach(store);
    take();

    host.receive({ type: 'subscribe', sub: 's1', query: 'MATCH (p:Person) RETURN p.name' });
    expect(rowsOf(take()[0]).rows).toHaveLength(2);

    // A Gremlin write over the query path must still notify standing queries.
    host.receive({
      type: 'query',
      req: 'g1',
      query: "g.addV('Person').property('name', 'carol')",
      lang: 'gremlin',
    });
    const msgs = take();
    expect(msgs.find((m) => m.type === 'result')).toBeDefined(); // the gremlin answer
    expect(rowsOf(msgs.find((m) => m.type === 'rows') as HostMessage).rows).toHaveLength(3);
  });

  test('a Gremlin parse error rides the coded wire error shape', () => {
    const { host, take } = attach(newStore());
    take();

    host.receive({ type: 'query', req: 'g1', query: 'g.V().totallyNotAStep()', lang: 'gremlin' });
    const [res] = take();
    expect(res.type).toBe('result');
    expect((res as { error?: { code: string } }).error?.code).toBeTruthy();
  });

  test('a Gremlin standing subscription pushes values, and again on a relevant change', () => {
    const store = newStore();
    const { host, take } = attach(store);
    take(); // drain status

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: 'g.V().count()',
      deps: ['Person'],
      lang: 'gremlin',
    });

    // Initial push carries `values` (arbitrary JSON), never rows/diffs.
    const first = rowsOf(take()[0]);
    expect(first.values).toEqual([2]);
    expect(first.rows).toBeUndefined();
    expect(first.patch).toBeUndefined();

    // Adding a Person moves the Person epoch → the standing traversal re-pushes.
    host.receive({ type: 'mutate', req: 'w1', text: "INSERT (:Person {name: 'carol'})" });
    const push = rowsOf(take().find((m) => m.type === 'rows') as HostMessage);
    expect(push.values).toEqual([3]);
  });

  test('a windowed keyless subscription pushes just the page; re-subscribe scrolls', () => {
    const store = newStore();
    const { host, take } = attach(store);
    take();

    const q = 'MATCH (p:Person) RETURN p.name ORDER BY p.name';
    host.receive({
      type: 'subscribe',
      sub: 'g',
      query: q,
      deps: ['Person'],
      window: { offset: 0, limit: 1 },
    });
    const page0 = rowsOf(take().find((m) => m.type === 'rows') as HostMessage);
    expect(page0.rows).toHaveLength(1); // page of 1, not the full 2 rows

    // Re-subscribe the SAME sub with the next window — that's how a grid scrolls.
    host.receive({
      type: 'subscribe',
      sub: 'g',
      query: q,
      deps: ['Person'],
      window: { offset: 1, limit: 1 },
    });
    const page1 = rowsOf(take().find((m) => m.type === 'rows') as HostMessage);
    expect(page1.rows).toHaveLength(1);
    expect(page1.rows![0]).not.toEqual(page0.rows![0]); // a different row — scrolled

    // A window past the end is an empty page, and `complete` still reflects the scope.
    host.receive({
      type: 'subscribe',
      sub: 'g',
      query: q,
      deps: ['Person'],
      window: { offset: 9, limit: 5 },
    });
    const past = rowsOf(take().find((m) => m.type === 'rows') as HostMessage);
    expect(past.rows).toEqual([]);
    expect(past.complete).toBe(true);
  });
});
