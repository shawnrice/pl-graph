// Proves the sync loop end to end: demand-fill fires off subscription deps
// and flips completeness honestly (including for empty scopes), local writes
// apply optimistically and replicate upstream with retry/backoff, server
// pushes ingest without re-replicating, and the whole thing drives real
// clients through engine.createHost. Run: bun test packages/sync/src/engine.test.ts
import { afterEach, describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson, type Store } from '@lenke/native';
import { createFfiEngineBackend } from '@lenke/native/ffi-engine';

import { createSyncClient, type SyncClient } from './client.js';
import { createSyncEngine, type SyncWrite, type SyncEngineOptions } from './engine.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-engine/target/release/liblenke_engine.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[engine.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
}

const suite = hasLib ? describe : describe.skip;

// Local seed: one Person already present (the "warm snapshot" starting point).
const NDJSON =
  '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"local","age":50}}';

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

/** A promise settled from the outside — deterministic async control. */
const deferred = <T>() => {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
};

/** Await until `check` passes (the pump/loaders hop the microtask queue). */
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

/** Engine + one client wired through engine.createHost — the full local loop. */
const connect = (opts: Omit<SyncEngineOptions, 'store'>) => {
  const store = newStore();
  const engine = createSyncEngine({ store, ...opts });
  let deliver: (m: unknown) => void = () => {};
  const host = engine.createHost({ send: (m) => deliver(m) });
  const client: SyncClient = createSyncClient({ send: (m) => host.receive(m) });
  deliver = (m) => client.receive(m);
  host.sendStatus(); // replay the handshake the buffered wiring missed

  return { store, engine, host, client };
};

suite('@lenke/sync engine · demand-fill', () => {
  test('a subscription over an unloaded collection answers incomplete, then fills', async () => {
    const gate = deferred<SyncWrite[]>();
    const { engine, client } = connect({
      collections: { people: { labels: ['Person'], load: () => gate.promise } },
    });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name ORDER BY p.name', {
      deps: ['Person', 'name'],
    });
    const stop = live.subscribe(() => {});

    // Immediate answer: the local (stale) row, honestly marked incomplete.
    expect(live.getSnapshot().complete).toBe(false);
    expect(live.getSnapshot().rows).toHaveLength(1);
    expect(engine.collectionState('people')).toBe('loading');

    // The loader lands → writes apply → epochs route → push flips complete.
    gate.resolve([
      { text: 'INSERT (:Person {name: $n, age: $a})', params: { n: 'remote-1', a: 30 } },
      { text: 'INSERT (:Person {name: $n, age: $a})', params: { n: 'remote-2', a: 31 } },
    ]);
    await until(() => live.getSnapshot().complete);

    expect(live.getSnapshot().rows).toHaveLength(3);
    expect(engine.collectionState('people')).toBe('complete');
    stop();
  });

  test('deps no collection covers are complete by definition (local-only data)', () => {
    const { client } = connect({
      collections: { people: { labels: ['Person'], load: () => Promise.resolve([]) } },
    });

    const live = client.liveQuery('MATCH (t:Team) RETURN t.name', { deps: ['Team', 'name'] });
    expect(live.getSnapshot().complete).toBe(true);
  });

  test('an EMPTY scope still flips complete (same rows, new truth)', async () => {
    const { client } = connect({
      collections: { teams: { labels: ['Team'], load: () => Promise.resolve([]) } },
    });

    const live = client.liveQuery('MATCH (t:Team) RETURN t.name', { deps: ['Team', 'name'] });
    const stop = live.subscribe(() => {});
    expect(live.getSnapshot()).toMatchObject({ complete: false, rows: [] });

    // No rows will ever change here — the completeness flip alone must push.
    await until(() => live.getSnapshot().complete);
    expect(live.getSnapshot().rows).toEqual([]);
    stop();
  });

  test('a failed load reports, surfaces a retryable error to the client, and the next demand retries', async () => {
    let calls = 0;
    const errors: string[] = [];
    const { engine, client } = connect({
      collections: {
        people: {
          labels: ['Person'],
          load: () => {
            calls += 1;

            return calls === 1
              ? Promise.reject(new Error('backend down'))
              : Promise.resolve([{ text: 'INSERT (:Person {name: $n})', params: { n: 'late' } }]);
          },
        },
      },
      onLoadError: (name) => {
        errors.push(name);
      },
    });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
    const stop = live.subscribe(() => {});

    await until(() => engine.collectionState('people') === 'error');
    expect(errors).toEqual(['people']);

    // The failure now reaches the CLIENT as a retryable error (not a silent
    // forever-skeleton), WITHOUT tearing the subscription down — the handle
    // stays wire-active so a later successful load just updates it.
    await until(() => live.getSnapshot().error !== undefined);
    expect(live.getSnapshot().complete).toBe(false);
    expect(client.subscriptionCount()).toBe(1);

    // A new demand re-triggers the load; on success the error CLEARS and rows fill.
    engine.ensure(['Person']);
    await until(() => live.getSnapshot().complete);
    expect(live.getSnapshot().error).toBeUndefined();
    expect(live.getSnapshot().rows.length).toBeGreaterThan(0);
    expect(calls).toBe(2);
    stop();
  });

  test('a keyed collection demand-fills and completes PER scope value', async () => {
    const loaded: string[] = [];
    const { engine, client } = connect({
      collections: {
        // One collection, sliced by the `city` param — no synthetic label.
        people: {
          labels: ['Person'],
          key: 'city',
          load: async ({ city }) => {
            loaded.push(city as string);

            return [
              { text: 'INSERT (:Person {name: $n, city: $c})', params: { n: city, c: city } },
            ];
          },
        },
      },
    });

    const oslo = client.liveQuery('MATCH (p:Person) WHERE p.city = $city RETURN p.name', {
      params: { city: 'oslo' },
      deps: ['Person', 'city'],
    });
    const stopOslo = oslo.subscribe(() => {});

    // Oslo fills; Bergen is a different, still-empty slice of the SAME collection.
    await until(() => oslo.getSnapshot().complete);
    expect(loaded).toEqual(['oslo']);
    expect(engine.collectionState('people', { city: 'oslo' })).toBe('complete');
    expect(engine.collectionState('people', { city: 'bergen' })).toBe('empty');

    const bergen = client.liveQuery('MATCH (p:Person) WHERE p.city = $city RETURN p.name', {
      params: { city: 'bergen' },
      deps: ['Person', 'city'],
    });
    const stopBergen = bergen.subscribe(() => {});

    await until(() => bergen.getSnapshot().complete);
    expect(loaded).toEqual(['oslo', 'bergen']); // each value loads exactly once
    expect(engine.collectionState('people', { city: 'bergen' })).toBe('complete');

    stopOslo();
    stopBergen();
  });

  test('initiallyComplete seeds one keyed slice; the others still fill', async () => {
    const loaded: string[] = [];
    const { client } = connect({
      collections: {
        people: {
          labels: ['Person'],
          key: 'city',
          load: async ({ city }) => {
            loaded.push(city as string);

            return [];
          },
        },
      },
      // The boot snapshot already covered Oslo.
      initiallyComplete: [{ name: 'people', scope: { city: 'oslo' } }],
    });

    const oslo = client.liveQuery('MATCH (p:Person) WHERE p.city = $city RETURN p.name', {
      params: { city: 'oslo' },
      deps: ['Person', 'city'],
    });
    const stopOslo = oslo.subscribe(() => {});

    // Oslo answers complete immediately without a load.
    await until(() => oslo.getSnapshot().complete);
    expect(loaded).toEqual([]);

    const bergen = client.liveQuery('MATCH (p:Person) WHERE p.city = $city RETURN p.name', {
      params: { city: 'bergen' },
      deps: ['Person', 'city'],
    });
    const stopBergen = bergen.subscribe(() => {});

    await until(() => bergen.getSnapshot().complete);
    expect(loaded).toEqual(['bergen']); // only the unseeded slice loaded
    stopOslo();
    stopBergen();
  });
});

suite('@lenke/sync engine · write-back', () => {
  test('mutate applies optimistically and replicates upstream', async () => {
    const pushed: SyncWrite[] = [];
    const { engine, client } = connect({
      upstream: {
        push: (w) => {
          pushed.push(w);

          return Promise.resolve();
        },
      },
    });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
    const stop = live.subscribe(() => {});

    engine.mutate('INSERT (:Person {name: $n})', { n: 'zoe' });

    // Optimistic: local readers see it before upstream ever answers.
    expect(live.getSnapshot().rows).toHaveLength(2);

    await until(() => engine.pendingWrites() === 0);
    expect(pushed).toEqual([{ text: 'INSERT (:Person {name: $n})', params: { n: 'zoe' } }]);
    stop();
  });

  test('a full write-back queue refuses new writes before the optimistic apply', async () => {
    const gate = deferred<void>();
    const { store, engine } = connect({
      maxPendingWrites: 2,
      upstream: { push: () => gate.promise }, // upstream never drains
    });

    const before = store.graph.vertexCount;
    engine.mutate('INSERT (:Person {name: $n})', { n: 'one' });
    engine.mutate('INSERT (:Person {name: $n})', { n: 'two' });

    // Queue is at the cap: the third write is REFUSED — coded throw, and the
    // local graph is untouched (no optimistic apply that would never replicate).
    expect(() => {
      engine.mutate('INSERT (:Person {name: $n})', { n: 'three' });
    }).toThrow('queue is full');
    expect(store.graph.vertexCount).toBe(before + 2);
    expect(engine.pendingWrites()).toBe(2);

    gate.resolve(); // upstream drains → capacity returns
    await until(() => engine.pendingWrites() === 0);
    engine.mutate('INSERT (:Person {name: $n})', { n: 'three' });
    expect(store.graph.vertexCount).toBe(before + 3);
  });

  test('a gremlin write with params is refused, not silently unbound', () => {
    const { engine } = connect({});

    expect(() => {
      engine.mutate("g.addV('Person').property('name', $n)", { n: 'zoe' }, 'gremlin');
    }).toThrow('no param binding');
  });

  test('a write that changed nothing replicates nothing (version gate)', async () => {
    const pushed: SyncWrite[] = [];
    const { engine } = connect({
      upstream: {
        push: (w) => {
          pushed.push(w);

          return Promise.resolve();
        },
      },
    });

    engine.mutate('MATCH (p:Person) RETURN p.name'); // a read smuggled into mutate
    await new Promise((r) => {
      setTimeout(r, 10);
    });
    expect(pushed).toEqual([]);
    expect(engine.pendingWrites()).toBe(0);
  });

  test('retry with backoff: transient failures deliver eventually', async () => {
    let attempts = 0;
    const { engine } = connect({
      retry: { attempts: 5, baseMs: 1 },
      upstream: {
        push: () => {
          attempts += 1;

          return attempts < 3 ? Promise.reject(new Error('flaky')) : Promise.resolve();
        },
      },
    });

    engine.mutate('INSERT (:Person {name: $n})', { n: 'retry-me' });
    await until(() => engine.pendingWrites() === 0);
    expect(attempts).toBe(3);
  });

  test('terminal failure drops the write and reports it', async () => {
    const failed: SyncWrite[] = [];
    const { engine } = connect({
      retry: { attempts: 2, baseMs: 1 },
      upstream: { push: () => Promise.reject(new Error('dead upstream')) },
      onWriteError: (w) => {
        failed.push(w);
      },
    });

    engine.mutate('INSERT (:Person {name: $n})', { n: 'doomed' });
    await until(() => engine.pendingWrites() === 0);
    expect(failed).toEqual([{ text: 'INSERT (:Person {name: $n})', params: { n: 'doomed' } }]);
  });

  test('a throwing onWriteError does not wedge the pump — later writes still drain', async () => {
    const seen: string[] = [];
    let failNext = true;
    let reported = false;
    const { engine } = connect({
      retry: { attempts: 1, baseMs: 1 },
      upstream: {
        push: (w) => {
          seen.push((w.params as { n: string }).n);

          return failNext ? Promise.reject(new Error('down')) : Promise.resolve();
        },
      },
      onWriteError: () => {
        reported = true;

        throw new Error('boom in user callback'); // the pump must survive this
      },
    });

    engine.mutate('INSERT (:Person {name: $n})', { n: 'one' }); // fails → onWriteError throws
    await until(() => reported);

    // Upstream recovers. Before the fix, a thrown onWriteError left `pumping`
    // stuck true, so this write would never drain (pendingWrites stays 1).
    failNext = false;
    engine.mutate('INSERT (:Person {name: $n})', { n: 'two' });

    await until(() => engine.pendingWrites() === 0);
    expect(seen).toContain('two');
  });

  test('client mutations flow through the queue; status reports the backlog', async () => {
    const gate = deferred<void>();
    const pushed: SyncWrite[] = [];
    const { client, engine } = connect({
      upstream: {
        push: (w) => {
          pushed.push(w);

          return gate.promise; // hold the write in flight
        },
      },
    });

    await client.mutate('INSERT (:Person {name: $n})', { n: 'via-wire' }); // ack is local+optimistic
    expect(engine.pendingWrites()).toBe(1);
    await until(() => client.getStatus()?.pendingWrites === 1); // status rode the queue change

    gate.resolve();
    await until(() => engine.pendingWrites() === 0);
    expect(pushed).toHaveLength(1);
    await until(() => client.getStatus()?.pendingWrites === 0);
  });
});

suite('@lenke/sync engine · ingest', () => {
  test('server pushes apply locally, route by epoch, and never re-replicate', async () => {
    const pushed: SyncWrite[] = [];
    const { engine, client } = connect({
      upstream: {
        push: (w) => {
          pushed.push(w);

          return Promise.resolve();
        },
      },
    });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
    const stop = live.subscribe(() => {});

    engine.ingest([
      { text: 'INSERT (:Person {name: $n})', params: { n: 'from-server-1' } },
      { text: 'INSERT (:Person {name: $n})', params: { n: 'from-server-2' } },
    ]);

    expect(live.getSnapshot().rows).toHaveLength(3); // the standing query heard it
    await new Promise((r) => {
      setTimeout(r, 10);
    });
    expect(pushed).toEqual([]); // and nothing echoed back upstream
    stop();
  });
});
