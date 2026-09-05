import { describe, expect, test } from 'bun:test';

import type { Backend, GraphHandle } from './backend.js';
import { __resetLeakWarnedForTests, attachGraph } from './graph.js';
import { createStore } from './store.js';

// A do-nothing backend that only records graphFree calls — enough to exercise
// the disposal wiring in attachGraph without a built native/wasm artifact.
const countingBackend = (): { backend: Backend; freed: GraphHandle[] } => {
  const freed: GraphHandle[] = [];
  const backend: Backend = {
    abiVersion: 10,
    graphFromNdjson: () => 1,
    graphClone: () => 1,
    graphFree: (handle) => {
      freed.push(handle);
    },
    vertexCount: () => 0,
    edgeCount: () => 0,
    version: () => 0,
    epoch: () => 0,
    createIndex: () => {},
    createUniqueConstraint: () => {},
    createRequiredConstraint: () => {},
    createTypeConstraint: () => {},
    createEdgeUniqueConstraint: () => {},
    createEdgeRequiredConstraint: () => {},
    createEdgeTypeConstraint: () => {},
    createCardinalityConstraint: () => {},
    createValidator: () => {},
    createInvariant: () => {},
    dropVertexIndex: () => {},
    dropEdgeIndex: () => {},
    beginTransaction: () => {},
    commitTransaction: () => {},
    rollbackTransaction: () => {},
    vertexIndexes: () => [],
    edgeIndexes: () => [],
    dumpSchema: () => [],
    mergeNdjson: () => ({
      nodesAdded: 0,
      edgesAdded: 0,
      nodesSkipped: [],
      edgesSkipped: [],
      phantomVertices: [],
    }),
    prepare: () => 1,
    preparedFree: () => {},
    preparedQueryRows: () => new TextEncoder().encode('{"columns":[],"rows":[]}'),
    preparedQueryArrow: () => new Uint8Array(),
    queryRows: () => new TextEncoder().encode('{"columns":[],"rows":[]}'),
    queryArrow: () => new Uint8Array(),
    queryArrowIpc: () => new Uint8Array(),
    gremlinJson: () => new TextEncoder().encode('[]'),
    algo: () => new TextEncoder().encode('{"columns":[],"rows":[]}'),
    encodeNdjson: () => new Uint8Array(),
    serialize: () => new Uint8Array(),
    deserialize: () => 1,
    setConfig: () => true,
    lastWriteScope: () => ({ scopes: [], open: false }),
  };

  return { backend, freed };
};

describe('RustGraph disposal', () => {
  test('free() releases the handle once', () => {
    const { backend, freed } = countingBackend();
    const g = attachGraph(backend, 7);

    g.free();

    expect(freed).toEqual([7]);
  });

  test('free() is idempotent — a second call does not double-free', () => {
    const { backend, freed } = countingBackend();
    const g = attachGraph(backend, 7);

    g.free();
    g.free();

    expect(freed).toEqual([7]);
  });

  test('[Symbol.dispose] is the same freed-once path as free()', () => {
    const { backend, freed } = countingBackend();
    const g = attachGraph(backend, 3);

    g[Symbol.dispose]();
    g.free(); // already disposed → no-op

    expect(freed).toEqual([3]);
  });

  test('`using` frees at scope exit', () => {
    const { backend, freed } = countingBackend();

    {
      using g = attachGraph(backend, 5);
      expect(g.vertexCount).toBe(0); // touch it so nothing tree-shakes the binding
      expect(freed).toEqual([]);
    }

    expect(freed).toEqual([5]);
  });

  test('disposing a store frees its underlying graph', () => {
    const { backend, freed } = countingBackend();
    const store = createStore(attachGraph(backend, 9));

    store[Symbol.dispose]();

    expect(freed).toEqual([9]);
  });

  test('reads on a freed graph throw a coded error, not a backend call', () => {
    const { backend, freed } = countingBackend();
    const g = attachGraph(backend, 7);

    g.free();

    expect(() => g.vertexCount).toThrow('after free');
    expect(() => g.query('MATCH (n) RETURN n')).toThrow('after free');
    expect(() => g.gremlin('g.V()')).toThrow('after free');
    expect(() => g.toNdjson()).toThrow('after free');
    expect(freed).toEqual([7]); // the reads never reached the backend
  });

  test('two wrappers over one handle share the freed-once guard', () => {
    const { backend, freed } = countingBackend();
    const a = attachGraph(backend, 7);
    const b = attachGraph(backend, 7);

    a.free();
    b.free(); // shared state → no second graphFree (an ffi double-free)

    expect(freed).toEqual([7]);
    expect(() => b.vertexCount).toThrow('after free'); // b knows it's dead too
  });

  test('a recycled handle value gets fresh state after free', () => {
    // free() removes the shared state, so a backend that recycles the same
    // handle value (ffi handles are pointers) hands the next attachment a
    // brand-new graph — it must NOT inherit the old wrapper's freed flag.
    const { backend, freed } = countingBackend();
    attachGraph(backend, 7).free();
    const reborn = attachGraph(backend, 7);

    expect(reborn.vertexCount).toBe(0); // alive, not poisoned by the old free
    reborn.free();
    expect(freed).toEqual([7, 7]); // two graphs' lifetimes, one free each
  });

  test('a graphFree that throws leaves the graph retryable, not marked freed', () => {
    const { backend, freed } = countingBackend();
    let failures = 1;
    const flaky: Backend = {
      ...backend,
      graphFree: (handle) => {
        if (failures > 0) {
          failures -= 1;

          throw new Error('transient backend fault');
        }

        backend.graphFree(handle);
      },
    };
    const g = attachGraph(flaky, 4);

    expect(() => g.free()).toThrow('transient');
    expect(g.vertexCount).toBe(0); // NOT marked freed — still usable
    g.free(); // retry succeeds

    expect(freed).toEqual([4]);
    expect(() => g.vertexCount).toThrow('after free');
  });

  test('the GC backstop reclaims a leaked graph and warns once (dev aid)', async () => {
    const { backend, freed } = countingBackend();
    const warnings: string[] = [];
    const realWarn = console.warn;
    console.warn = (...args: unknown[]) => {
      warnings.push(args.map(String).join(' '));
    };

    // A graph leaked by an earlier test may already have tripped the
    // once-per-process latch (its warning went to the real console.warn, before
    // this override). Reset so this test observes its OWN single warning.
    __resetLeakWarnedForTests();

    try {
      // Attach and drop the only reference — no free(), no `using`. The IIFE
      // keeps the wrapper from staying reachable via a lingering binding.
      (() => {
        attachGraph(backend, 42);
      })();

      // Finalizers are best-effort; a forced GC runs them within a tick or two
      // on Bun, so poll a few rounds until the reclaim lands.
      for (let i = 0; i < 20 && freed.length === 0; i++) {
        Bun.gc(true);

        await new Promise((r) => setTimeout(r, 5));
      }

      expect(freed).toEqual([42]); // reclaimed despite never being freed
      // Exactly one leak warning, and it names the deterministic-disposal fix.
      const leakWarns = warnings.filter((w) => w.includes('without free()'));
      expect(leakWarns).toHaveLength(1);
      expect(leakWarns[0]).toContain('using');
    } finally {
      console.warn = realWarn;
    }
  });

  test('a disposed store severs subscriptions and keeps snapshots stable', () => {
    const { backend } = countingBackend();
    const store = createStore(attachGraph(backend, 9));
    const live = store.liveQuery('MATCH (n) RETURN n', { deps: null });
    let woke = 0;
    live.subscribe(() => {
      woke += 1;
    });
    const before = live.getSnapshot(); // prime the cache while the graph is live

    store[Symbol.dispose]();

    expect(live.getSnapshot()).toBe(before); // cached, never touches the freed graph
    expect(live.subscribe(() => {})()).toBeUndefined(); // no-op subscription
    expect(() => store.mutate((g) => g.query('INSERT (:X)'))).toThrow('after dispose');
    expect(woke).toBe(0); // dispose itself never notifies
  });
});
