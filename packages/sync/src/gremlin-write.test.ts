// The Gremlin write path — the twin of the GQL one. Reads were already
// bilingual (liveGremlin / lang:'gremlin'); these prove a mutation can travel as
// Gremlin end to end: host applies it, it notifies standing queries, the client
// escapes values safely through `mutateGremlin`, and the engine queues a
// `lang:'gremlin'` write upstream. Run: bun test packages/sync/src/gremlin-write.test.ts
import { afterEach, describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson, type Store } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';

import { createSyncClient, type SyncClient } from './client.js';
import { createSyncEngine, type SyncEngineOptions, type SyncWrite } from './engine.js';
import { createSyncHost } from './host.js';
import type { HostMessage } from './protocol.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(
    `[gremlin-write.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`,
  );
}

const suite = hasLib ? describe : describe.skip;

// One seed Person — the warm starting point.
const NDJSON = '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"local"}}';

const created: Store[] = [];

// Track every store so its native graph handle is released after each test; a
// leaked handle otherwise trips the GC-backstop warning when the finalizer runs.
afterEach(() => {
  for (const store of created.splice(0)) {
    store.free();
  }
});

const newStore = (): Store => {
  const store = createStore(
    graphFromNdjson(createFfiEngineBackend(LIB), new TextEncoder().encode(NDJSON)),
  );
  created.push(store);

  return store;
};

const until = async (check: () => boolean): Promise<void> => {
  for (let i = 0; i < 500; i += 1) {
    if (check()) {
      return;
    }

    await new Promise((r) => {
      setTimeout(r, 2);
    });
  }

  throw new Error('until(): condition never became true');
};

/** Engine + one client wired through engine.createHost (mirrors engine.test's helper). */
const connect = (opts: Omit<SyncEngineOptions, 'store'> = {}) => {
  const store = newStore();
  const engine = createSyncEngine({ store, ...opts });
  let deliver: (m: unknown) => void = () => {};
  const host = engine.createHost({ send: (m) => deliver(m) });
  const client: SyncClient = createSyncClient({ send: (m) => host.receive(m) });
  deliver = (m) => client.receive(m);
  host.sendStatus();

  return { store, engine, host, client };
};

const names = (rows: readonly Record<string, unknown>[]): unknown[] =>
  rows.map((r) => r['p.name']).sort();

suite('@lenke/sync · gremlin write path', () => {
  test('the host applies a lang:gremlin mutate and acks', () => {
    const store = newStore();
    const sent: HostMessage[] = [];
    const host = createSyncHost(store, { send: (m) => sent.push(m) });

    const before = store.graph.vertexCount;
    host.receive({
      type: 'mutate',
      req: 'm1',
      text: "g.addV('Person').property('name', 'zed')",
      lang: 'gremlin',
    });

    expect(store.graph.vertexCount).toBe(before + 1);
    expect(sent).toContainEqual({ type: 'ack', req: 'm1', ok: true });
  });

  test('a gremlin mutate notifies a standing GQL subscription', () => {
    const store = newStore();
    const sent: HostMessage[] = [];
    const host = createSyncHost(store, { send: (m) => sent.push(m) });

    host.receive({
      type: 'subscribe',
      sub: 's1',
      query: 'MATCH (p:Person) RETURN p.name',
      deps: ['Person'],
    });
    sent.length = 0;

    host.receive({
      type: 'mutate',
      req: 'm',
      text: "g.addV('Person').property('name', 'zed')",
      lang: 'gremlin',
    });

    const push = sent.find((m): m is Extract<HostMessage, { type: 'rows' }> => m.type === 'rows');
    expect(push).toBeDefined();
    expect(names(push!.rows ?? [])).toEqual(['local', 'zed']);
  });

  test('a bad gremlin mutate acks not-ok with a coded error', () => {
    const store = newStore();
    const sent: HostMessage[] = [];
    const host = createSyncHost(store, { send: (m) => sent.push(m) });

    host.receive({ type: 'mutate', req: 'm', text: 'g.NOT.a.traversal()', lang: 'gremlin' });

    const ack = sent.find((m): m is Extract<HostMessage, { type: 'ack' }> => m.type === 'ack');
    expect(ack?.ok).toBe(false);
    expect(ack?.error).toBeDefined();
  });

  test('client.mutateGremlin escapes values and drives the loop safely', async () => {
    const { client } = connect();

    // An apostrophe in the value would break out of a naively-built literal;
    // mutateGremlin escapes it, so it inserts as data, not syntax.
    await client.mutateGremlin`g.addV('Person').property('name', ${"o'brien"})`;

    const rows = await client.query('MATCH (p:Person) WHERE p.name = $n RETURN p.name', {
      n: "o'brien",
    });
    expect(rows).toHaveLength(1);
  });

  test('engine.mutate(lang:gremlin) applies locally and queues a gremlin write upstream', async () => {
    const store = newStore();
    const pushed: SyncWrite[] = [];
    const engine = createSyncEngine({
      store,
      upstream: {
        push: async (w) => {
          pushed.push(w);
        },
      },
    });

    const before = store.graph.vertexCount;
    engine.mutate("g.addV('Person').property('name', 'x')", undefined, 'gremlin');

    expect(store.graph.vertexCount).toBe(before + 1); // optimistic local apply
    await until(() => pushed.length === 1);
    expect(pushed[0]).toEqual({ text: "g.addV('Person').property('name', 'x')", lang: 'gremlin' });
  });

  test('a gremlin write replicates upstream AS gremlin (worker → wire → server)', async () => {
    // The full replication chain: a worker-side engine queues a Gremlin write,
    // upstream.push forwards it through a wire client (with its lang), and the
    // server-side host runs it through the Gremlin engine — not as GQL.
    const serverStore = newStore();
    let deliver: (m: unknown) => void = () => {};
    const serverHost = createSyncHost(serverStore, { send: (m) => deliver(m) });
    const wire: SyncClient = createSyncClient({ send: (m) => serverHost.receive(m) });
    deliver = (m) => wire.receive(m);

    const workerStore = newStore();
    const pushed: SyncWrite[] = [];
    const engine = createSyncEngine({
      store: workerStore,
      upstream: {
        push: async (w) => {
          pushed.push(w);
          await wire.mutate(w.text, w.params, w.lang); // the bridge forwards lang
        },
      },
    });

    engine.mutate("g.addV('Person').property('name', 'replicated')", undefined, 'gremlin');

    await until(() => serverStore.graph.vertexCount === 2); // seed + the replicated vertex
    expect(pushed[0]?.lang).toBe('gremlin');
    const rows = serverStore.graph.query('MATCH (p:Person) WHERE p.name = $n RETURN p.name', {
      n: 'replicated',
    });
    expect(rows).toHaveLength(1);
  });

  test('client.mutate rejects a gremlin write carrying params', async () => {
    const { client } = connect();

    // Gremlin has no param binding — this would run with an unbound $n literal.
    expect(
      client.mutate("g.addV('Person').property('name', $n)", { n: 'x' }, 'gremlin'),
    ).rejects.toThrow('no param binding');
  });

  test('a collection loader can materialize its scope with a gremlin write', async () => {
    const { client } = connect({
      collections: {
        people: {
          labels: ['Person'],
          load: async (): Promise<SyncWrite[]> => [
            { text: "g.addV('Person').property('name', 'loaded')", lang: 'gremlin' },
          ],
        },
      },
    });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person'] });
    live.subscribe(() => {});

    await until(() => live.getSnapshot().complete && live.getSnapshot().rows.length === 2);
    expect(names(live.getSnapshot().rows)).toEqual(['loaded', 'local']);
  });
});
