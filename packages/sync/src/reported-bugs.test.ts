// Reproductions for reported bugs in the sync host + engine. Each test either
// FAILS (bug confirmed) or PASSES (bug disproved). These are diagnostic — they
// are NOT fixes. Run: bun test packages/sync/src/reported-bugs.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson, type LiveQuery, type Row, type Store } from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';

import { createSyncClient, type SyncClient } from './client.js';
import { createSyncEngine, type SyncEngineOptions } from './engine.js';
import { createSyncHost } from './host.js';
import type { HostMessage } from './protocol.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[reported-bugs.test] skipping LIB suites: ${LIB} not found.`);
}

const suite = hasLib ? describe : describe.skip;

const NDJSON =
  '{"type":"node","id":"a","labels":["Person"],"properties":{"name":"local","age":50}}';

const newStore = (): Store =>
  createStore(graphFromNdjson(createFfiBackend(LIB), new TextEncoder().encode(NDJSON)));

const deferred = <T>() => {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve, reject };
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
};

const tick = (ms = 5): Promise<void> =>
  new Promise((r) => {
    setTimeout(r, ms);
  });

/** Engine + one client wired through engine.createHost — the full local loop. */
const connect = (opts: Omit<SyncEngineOptions, 'store'>, store: Store = newStore()) => {
  const engine = createSyncEngine({ store, ...opts });
  let deliver: (m: unknown) => void = () => {};
  const host = engine.createHost({ send: (m) => deliver(m) });
  const client: SyncClient = createSyncClient({ send: (m) => host.receive(m) });
  deliver = (m) => client.receive(m);
  host.sendStatus();

  return { store, engine, host, client };
};

// =====================================================================
// Bug 1: demand-fill never retries after a loader failure — a standing
// subscription driven ONLY by normal pushes/refreshes (never a manual
// ensure) stays stuck incomplete forever once its first load fails.
// =====================================================================
suite('reported-bug #1 · demand-fill never retries after a loader failure', () => {
  // CONFIRMED, deferred: the fix is a retry-POLICY decision. Naively re-ensuring
  // on every refresh conflicts with the intentional "stays incomplete, retries on
  // the next explicit demand" engine test and risks a retry storm (a failed load
  // → notifyChange → refresh → re-ensure → …). Needs a deliberate policy
  // (backoff / trigger conditions). Kept as a skipped, documented reproduction.
  test.skip('a failed load recovers under ordinary pushes/refreshes (no manual ensure)', async () => {
    let calls = 0;
    const { engine, host, client } = connect({
      collections: {
        people: {
          labels: ['Person'],
          load: () => {
            calls += 1;

            // First load fails; every later load would succeed.
            return calls === 1
              ? Promise.reject(new Error('backend down'))
              : Promise.resolve([{ gql: 'INSERT (:Person {name: $n})', params: { n: 'late' } }]);
          },
        },
      },
      onLoadError: () => {},
    });

    const live = client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] });
    const stop = live.subscribe(() => {});

    // The one-and-only ensure fired at subscribe; the load failed.
    await until(() => engine.collectionState('people') === 'error');
    expect(live.getSnapshot().complete).toBe(false);

    // Now drive the loop the way a live app would — WITHOUT calling ensure:
    // ordinary local writes (each a real store notification + host refresh) and
    // explicit host refreshes (what the engine does on every change).
    for (let i = 0; i < 5; i += 1) {
      engine.mutate('INSERT (:Other {tag: $t})', { t: i });
      host.refresh();
      await tick();
    }

    // Correct behavior: a subsequent push/refresh should re-trigger the errored
    // collection's load so the standing subscription recovers. It does not.
    expect(calls).toBe(2); // FAILS if confirmed: the loader never re-runs (stays 1)
    expect(live.getSnapshot().complete).toBe(true); // FAILS if confirmed: stuck incomplete
    stop();
  });
});

// =====================================================================
// Bug 2: cellChanged() (via diffRows) runs OUTSIDE push()'s try/catch, so an
// unstringifiable object-valued cell (e.g. one holding a BigInt) throws out of
// the store-notify callback and breaks the push loop instead of being handled.
//
// Note: the real graph query() path decodes via JSON.parse and can only ever
// produce JSON-safe cells, so it cannot itself surface a BigInt. We drive the
// exact host code path with a minimal fake Store whose keyed subscription's
// object cell mutates into a BigInt-bearing object between pushes — proving the
// unguarded JSON.stringify escapes push().
// =====================================================================
describe('reported-bug #2 · unstringifiable cell escapes push()', () => {
  test('a BigInt-bearing object cell throws out of the push loop', () => {
    // A controllable single-key result. First push: a plain object cell. Second
    // push: same key, a NEW object cell containing a BigInt — cellChanged then
    // compares two objects and JSON.stringify()s them.
    let rows: Row[] = [{ id: 'k1', data: { v: 1 } }];
    let notify: () => void = () => {};

    const fakeLive: LiveQuery<Row> = {
      subscribe: (cb) => {
        notify = cb;

        return () => {
          notify = () => {};
        };
      },
      getSnapshot: () => rows,
    };

    const fakeStore = {
      graph: {} as never,
      version: 1,
      mutate: <T>(fn: (g: never) => T) => fn({} as never),
      liveQuery: () => fakeLive,
      liveGremlin: () => fakeLive as LiveQuery<unknown>,
    } as unknown as Store;

    const sent: HostMessage[] = [];
    const host = createSyncHost(fakeStore, { send: (m) => sent.push(m) });

    // Keyed subscription → the diff path (cellChanged) is exercised.
    host.receive({ type: 'subscribe', sub: 's1', query: 'ignored', key: 'id' });

    // Now the same key's object cell changes into a BigInt-bearing object and a
    // push fires. cellChanged → JSON.stringify({ v: 1n }) throws.
    rows = [{ id: 'k1', data: { v: 1n } }];

    // Correct behavior: push() should not throw; the subscription should survive
    // (the failure handled like a snapshot failure, not propagated to the caller
    // of the store-notify callback).
    expect(() => notify()).not.toThrow(); // FAILS if confirmed: BigInt stringify escapes push
    expect(host.subscriptionCount()).toBe(1); // and the subscription should still stand
  });
});

// =====================================================================
// Bug 3: refresh re-walks EVERY subscription on a status-only change.
// A pending-write count moving (data unchanged) fires notifyChange →
// host.refresh(), which calls getSnapshot() on every subscription to
// ultimately emit nothing.
// =====================================================================
suite('reported-bug #3 · status-only change re-walks every subscription', () => {
  test('a queue movement (no data change) calls getSnapshot per sub but emits no rows', async () => {
    // Instrument the store: count getSnapshot() invocations across all live
    // queries the host builds, and record whether diffRows would have run (a
    // recompute of the underlying rows).
    const base = newStore();
    let snapshotCalls = 0;
    const instrumented: Store = {
      ...base,
      get graph() {
        return base.graph;
      },
      get version() {
        return base.version;
      },
      mutate: base.mutate,
      liveGremlin: base.liveGremlin,
      liveQuery: (text, o) => {
        const live: LiveQuery<Row> = base.liveQuery(text, o);

        return {
          subscribe: live.subscribe,
          getSnapshot: () => {
            snapshotCalls += 1;

            return live.getSnapshot();
          },
        };
      },
    };

    const gate = deferred<void>();
    const { engine, client } = connect(
      {
        upstream: {
          push: () => gate.promise, // hold the write in flight so the queue lingers
        },
      },
      instrumented,
    );

    // A handful of standing queries so "per-sub work" is measurable.
    const subs = [
      client.liveQuery('MATCH (p:Person) RETURN p.name', { deps: ['Person', 'name'] }),
      client.liveQuery('MATCH (p:Person) RETURN p.age', { deps: ['Person', 'age'] }),
      client.liveQuery('MATCH (t:Team) RETURN t.name', { deps: ['Team', 'name'] }),
    ];
    const stops = subs.map((s) => s.subscribe(() => {}));

    // Drive an optimistic write: applies locally + enqueues (pendingWrites 1).
    await client.mutate('INSERT (:Team {name: $n})', { n: 'blue' });
    await until(() => engine.pendingWrites() === 1);
    await tick();

    // Baseline: from here, no graph data will change. Capture each snapshot and
    // reset the getSnapshot counter.
    const snapAfterReset = subs.map((s) => JSON.stringify(s.getSnapshot()));
    snapshotCalls = 0;

    // The status-only change: the in-flight write settles → queue shifts →
    // pendingWrites 1→0 → notifyChange → host.refresh() over ALL subs. (Client
    // getSnapshot() does not touch this counter — only the host reads the store,
    // so this measures the refresh's work in isolation.)
    gate.resolve();
    await until(() => engine.pendingWrites() === 0);
    await tick();
    const callsFromStatusRefresh = snapshotCalls;

    // The redundant work: refresh() called getSnapshot() for every subscription
    // even though nothing about their rows moved.
    expect(callsFromStatusRefresh).toBeGreaterThanOrEqual(subs.length);

    // ...yet no subscription's snapshot actually changed (no patch was needed).
    const snapAfterStatus = subs.map((s) => JSON.stringify(s.getSnapshot()));
    expect(snapAfterStatus).toEqual(snapAfterReset);

    stops.forEach((s) => s());
  });
});
