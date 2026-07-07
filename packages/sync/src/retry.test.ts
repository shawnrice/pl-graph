// Demand-fill retry scheduler: the storm-safety and footgun contract. A failed
// load must retry a BOUNDED number of times (never a storm), self-heal when the
// backend returns, honor an app-supplied scheduler escape hatch, let an explicit
// demand jump the backoff, prioritize under a concurrency cap, and reset cleanly
// on retryAll. Run: bun test packages/sync/src/retry.test.ts
import { describe, expect, test } from 'bun:test';
import { existsSync } from 'node:fs';

import { createStore, graphFromNdjson, type Store } from '@lenke/native';
import { createFfiBackend } from '@lenke/native/ffi';

import {
  createSyncEngine,
  type SyncWrite,
  type LoadJob,
  type LoadScheduler,
  type SyncEngine,
  type SyncEngineOptions,
} from './engine.js';

const LIB_EXTENSIONS: Partial<Record<NodeJS.Platform, string>> = { darwin: 'dylib', win32: 'dll' };
const LIB_EXT = LIB_EXTENSIONS[process.platform] ?? 'so';
const LIB = new URL(
  `../../../crates/lenke-core/target/release/liblenke_core.${LIB_EXT}`,
  import.meta.url,
).pathname;
const hasLib = existsSync(LIB);

if (!hasLib) {
  console.warn(`[retry.test] skipping: ${LIB} not found — run \`bun run build:rust\` first.`);
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

const until = async (check: () => boolean, label = 'condition'): Promise<void> => {
  for (let i = 0; i < 500; i += 1) {
    if (check()) {
      return;
    }

    await new Promise((r) => {
      setTimeout(r, 2);
    });
  }

  throw new Error(`until(): ${label} never became true`);
};

const tick = (ms = 5): Promise<void> =>
  new Promise((r) => {
    setTimeout(r, ms);
  });

const mkEngine = (opts: Omit<SyncEngineOptions, 'store'>): SyncEngine =>
  createSyncEngine({ store: newStore(), ...opts });

/**
 * A hand-driven scheduler standing in for an app's escape hatch: it never runs a
 * job on its own — the test calls `run()` — so retry accounting is fully
 * deterministic (no timers, no jitter). It records every job it's handed and
 * every cancel the engine fires.
 */
const manualScheduler = () => {
  const jobs: LoadJob[] = [];
  const canceled: LoadJob[] = [];
  const scheduler: LoadScheduler = (job) => {
    jobs.push(job);

    return () => {
      canceled.push(job);
    };
  };

  return { scheduler, jobs, canceled };
};

const INSERT: SyncWrite = { text: 'INSERT (:Person {name: $n})', params: { n: 'late' } };

suite('demand-fill retry · storm safety', () => {
  test('a load that always fails retries a BOUNDED number of times, never a storm', async () => {
    const man = manualScheduler();
    let calls = 0;
    const engine = mkEngine({
      collections: {
        people: {
          labels: ['Person'],
          load: () => {
            calls += 1;

            return Promise.reject(new Error('backend down'));
          },
        },
      },
      loadRetry: { attempts: 5 },
      loadScheduler: man.scheduler,
      onLoadError: () => {},
    });

    // First demand → one job. Drain the whole retry chain: each failed run mints
    // exactly one next job, until the attempt budget is spent.
    engine.ensure(['Person']);

    for (let i = 0; i < 20 && man.jobs.length > calls; i += 1) {
      const job = man.jobs[man.jobs.length - 1];
      await job.run().catch(() => {});
    }

    // Exactly `attempts` loads — attempt 0..4 — then it gives up. Not infinite.
    expect(calls).toBe(5);
    expect(man.jobs.length).toBe(5);
    expect(man.jobs.map((j) => j.attempt)).toEqual([0, 1, 2, 3, 4]);
    expect(engine.collectionState('people')).toBe('error');
  });

  test('pushes and refreshes never re-trigger a failed load (the storm footgun)', async () => {
    const man = manualScheduler();
    let calls = 0;
    const engine = mkEngine({
      collections: {
        people: {
          labels: ['Person'],
          load: () => {
            calls += 1;

            return Promise.reject(new Error('down'));
          },
        },
      },
      loadRetry: { attempts: 3 },
      loadScheduler: man.scheduler,
      onLoadError: () => {},
    });

    engine.ensure(['Person']);
    await man.jobs[0].run().catch(() => {}); // attempt 0 fails → attempt 1 scheduled
    expect(calls).toBe(1);
    expect(man.jobs.length).toBe(2);

    // Simulate the app hammering the loop: local writes + change notifications,
    // the exact path that used to re-ensure and storm. None of it schedules a
    // new load — only the pending scheduled retry can advance the chain.
    for (let i = 0; i < 10; i += 1) {
      engine.mutate('INSERT (:Other {tag: $t})', { t: i });
    }

    expect(man.jobs.length).toBe(2); // still just [attempt0, attempt1]
    expect(calls).toBe(1); // the pending retry hasn't been run by the harness yet
  });
});

suite('demand-fill retry · self-heal', () => {
  test('a failed load recovers on a later scheduled retry — no manual ensure', async () => {
    let calls = 0;
    const engine = mkEngine({
      collections: {
        people: {
          labels: ['Person'],
          load: () => {
            calls += 1;

            // First attempt fails; the scheduled retry succeeds.
            return calls === 1
              ? Promise.reject(new Error('cold start'))
              : Promise.resolve([INSERT]);
          },
        },
      },
      // Tiny backoff so the real default scheduler's timer fires fast in-test.
      loadRetry: { attempts: 5, baseMs: 10, maxMs: 40 },
      onLoadError: () => {},
    });

    engine.ensure(['Person']);
    await until(() => engine.collectionState('people') === 'error', 'first load fails');

    // No further ensure() — the engine's own scheduled retry heals it.
    await until(() => engine.collectionState('people') === 'complete', 'scheduled retry heals');
    expect(calls).toBe(2);
  });

  test('retryAll resets every errored collection to a fresh immediate load', async () => {
    const man = manualScheduler();
    let peopleOk = false;
    let teamsOk = false;
    const engine = mkEngine({
      collections: {
        people: {
          labels: ['Person'],
          load: () => (peopleOk ? Promise.resolve([INSERT]) : Promise.reject(new Error('p down'))),
        },
        teams: {
          labels: ['Team'],
          load: () => (teamsOk ? Promise.resolve([]) : Promise.reject(new Error('t down'))),
        },
      },
      loadRetry: { attempts: 1 }, // one shot each → straight to 'error', no auto-retry
      loadScheduler: man.scheduler,
      onLoadError: () => {},
    });

    engine.ensure(['Person']);
    engine.ensure(['Team']);
    await man.jobs[0].run().catch(() => {});
    await man.jobs[1].run().catch(() => {});
    expect(engine.collectionState('people')).toBe('error');
    expect(engine.collectionState('teams')).toBe('error');

    // Backend returns; a reconnect handler calls retryAll → one fresh attempt-0
    // job per errored slice (no waiting out any backoff).
    peopleOk = true;
    teamsOk = true;
    const before = man.jobs.length;
    engine.retryAll();
    const fresh = man.jobs.slice(before);
    expect(fresh.length).toBe(2);
    expect(fresh.every((j) => j.attempt === 0)).toBe(true);

    await Promise.all(fresh.map((j) => j.run()));
    expect(engine.collectionState('people')).toBe('complete');
    expect(engine.collectionState('teams')).toBe('complete');
  });
});

suite('demand-fill retry · escape hatch + priority', () => {
  test('an app-supplied loadScheduler receives every job (default gate bypassed)', async () => {
    const man = manualScheduler();
    const engine = mkEngine({
      collections: { people: { labels: ['Person'], load: () => Promise.resolve([INSERT]) } },
      loadScheduler: man.scheduler,
    });

    engine.ensure(['Person']);
    expect(man.jobs.length).toBe(1);
    expect(man.jobs[0]).toMatchObject({ collection: 'people', attempt: 0, priority: 0 });

    // Nothing loads until the app's scheduler chooses to run it.
    expect(engine.collectionState('people')).toBe('empty');
    await man.jobs[0].run();
    expect(engine.collectionState('people')).toBe('complete');
  });

  test('an explicit ensure supersedes a pending backoff — cancels it, runs now', async () => {
    const man = manualScheduler();
    const engine = mkEngine({
      collections: {
        people: { labels: ['Person'], load: () => Promise.reject(new Error('down')) },
      },
      loadRetry: { attempts: 5 },
      loadScheduler: man.scheduler,
      onLoadError: () => {},
    });

    engine.ensure(['Person']);
    await man.jobs[0].run().catch(() => {}); // fails → attempt-1 retry now pending
    expect(man.jobs.length).toBe(2);
    expect(man.jobs[1].attempt).toBe(1);
    expect(man.canceled.length).toBe(0);

    // A user action re-demands the same slice: the pending backoff job is
    // canceled and a fresh attempt-0 job takes its place — no waiting.
    engine.ensure(['Person'], undefined, 10);
    expect(man.canceled).toContain(man.jobs[1]); // the backoff retry was dropped
    expect(man.jobs.length).toBe(3);
    expect(man.jobs[2]).toMatchObject({ attempt: 0, priority: 10 });
  });

  test('the default scheduler runs higher priority first when loads contend for a slot', async () => {
    const block = deferred<SyncWrite[]>();
    const order: string[] = [];
    const gateOf = (name: string, resolveWith: SyncWrite[]) => () => {
      order.push(name);

      return Promise.resolve(resolveWith);
    };

    const engine = mkEngine({
      // Concurrency 1: the blocker holds the single slot while low + high queue,
      // so when it frees the scheduler must CHOOSE — priority decides.
      maxConcurrentLoads: 1,
      collections: {
        blocker: { labels: ['Block'], load: () => block.promise },
        low: { labels: ['Low'], load: gateOf('low', []) },
        high: { labels: ['High'], load: gateOf('high', []) },
      },
    });

    engine.ensure(['Block'], undefined, 0); // occupies the only slot
    await until(() => engine.collectionState('blocker') === 'loading', 'blocker holds slot');

    engine.ensure(['Low'], undefined, 1); // both wait behind the blocker
    engine.ensure(['High'], undefined, 9);
    expect(engine.collectionState('low')).toBe('empty');
    expect(engine.collectionState('high')).toBe('empty');

    block.resolve([]); // slot frees → highest priority (high) runs before low
    await until(() => order.length === 2, 'both queued loads ran');
    expect(order).toEqual(['high', 'low']);
  });

  test('the default scheduler caps concurrency — a burst never all runs at once', async () => {
    const gates = [deferred<SyncWrite[]>(), deferred<SyncWrite[]>(), deferred<SyncWrite[]>()];
    let running = 0;
    let peakRunning = 0;
    const loadOf = (i: number) => () => {
      running += 1;
      peakRunning = Math.max(peakRunning, running);

      return gates[i].promise.finally(() => {
        running -= 1;
      });
    };

    const engine = mkEngine({
      maxConcurrentLoads: 2,
      collections: {
        a: { labels: ['A'], load: loadOf(0) },
        b: { labels: ['B'], load: loadOf(1) },
        c: { labels: ['C'], load: loadOf(2) },
      },
    });

    // Demand all three at once; the cap must hold the third back.
    engine.ensure(['A'], undefined, 0);
    engine.ensure(['B'], undefined, 0);
    engine.ensure(['C'], undefined, 0);
    await tick();

    expect(peakRunning).toBe(2); // never 3 in flight
    expect(running).toBe(2);

    gates[0].resolve([]);
    await until(() => running === 2 && peakRunning === 2, 'third starts after a slot frees');
    // The third only started once one of the first two settled — still ≤ 2.
    expect(peakRunning).toBe(2);

    gates[1].resolve([]);
    gates[2].resolve([]);
    await until(() => running === 0, 'all drain');
  });
});
