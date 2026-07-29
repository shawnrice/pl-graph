// The store's live-query pool: identical standing queries share ONE recompute
// and ONE cached array; distinct ones don't; a fully-unsubscribed cell is
// evicted (so the pool never accumulates). A counting fake backend makes the
// sharing observable without a native artifact.
import { describe, expect, test } from 'bun:test';

import type { Backend, GraphHandle } from './backend.js';
import { attachGraph } from './graph.js';
import { createStore } from './store.js';

const rows = (cols: string[], data: unknown[][]): Uint8Array =>
  new TextEncoder().encode(JSON.stringify({ columns: cols, rows: data }));

/** A fake graph whose `version`/`epoch` we drive, counting query + gremlin runs. */
const fakeStore = () => {
  let version = 0;
  const counts = { query: 0, gremlin: 0 };
  const backend: Backend = {
    abiVersion: 10,
    graphFromNdjson: () => 1,
    graphClone: () => 1,
    graphFree: () => {},
    vertexCount: () => 0,
    edgeCount: () => 0,
    version: () => version,
    epoch: () => version, // every dep's epoch tracks version → any dep fires on a bump
    createVertexIndex: () => {},
    createEdgeIndex: () => {},
    createEdgeIntervalIndex: () => {},
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
    preparedQueryArrow: () => new Uint8Array(),
    preparedQueryRows: () => rows(['n'], [[counts.query]]),
    queryRows: () => {
      counts.query += 1;

      return rows(['n'], [[counts.query]]);
    },
    queryArrow: () => new Uint8Array(),
    queryArrowIpc: () => new Uint8Array(),
    gremlinJson: () => {
      counts.gremlin += 1;

      return new TextEncoder().encode(`[${counts.gremlin}]`);
    },
    algo: () => rows(['node', 'degree'], []),
    encodeNdjson: () => new Uint8Array(),
    serialize: () => new Uint8Array(),
    deserialize: (): GraphHandle => 1,
    setMaxOperatorChain: () => {},
    lastWriteScope: () => [],
  };
  const store = createStore(attachGraph(backend, 1));

  return {
    store,
    counts,
    bump: () => {
      version += 1;
    },
  };
};

describe('store live-query pool', () => {
  test('identical live queries share one recompute and one cached array', () => {
    const { store, counts, bump } = fakeStore();
    const opts = { deps: ['Person'] };

    const a = store.liveQuery('MATCH (p:Person) RETURN p', opts);
    const b = store.liveQuery('MATCH (p:Person) RETURN p', opts);
    a.subscribe(() => {});
    b.subscribe(() => {});

    const ra = a.getSnapshot();
    const rb = b.getSnapshot();
    expect(rb).toBe(ra); // same array reference — one cached result, shared
    expect(counts.query).toBe(1); // computed once, not per handle

    bump();
    store.mutate(() => {}); // a version-moving mutation (notify happens via the store)
    const ra2 = a.getSnapshot();
    const rb2 = b.getSnapshot();
    expect(rb2).toBe(ra2);
    expect(counts.query).toBe(2); // recomputed once for both, not twice
  });

  test('distinct signatures do NOT share (text / params / kind / null-vs-empty deps)', () => {
    const { store, counts } = fakeStore();

    store.liveQuery('MATCH (a) RETURN a', { deps: null }).getSnapshot();
    store.liveQuery('MATCH (b) RETURN b', { deps: null }).getSnapshot(); // different text
    store.liveQuery('MATCH (a) RETURN a', { deps: null, params: { x: 1 } }).getSnapshot(); // params
    store.liveQuery('MATCH (a) RETURN a', { deps: [] }).getSnapshot(); // [] ≠ null
    expect(counts.query).toBe(4); // four distinct cells → four computes

    store.liveGremlin('g.V()', { deps: null }).getSnapshot(); // different kind, different runner
    expect(counts.gremlin).toBe(1);
  });

  test('deps are a set — declaration order does not defeat sharing', () => {
    const { store, counts } = fakeStore();

    const a = store.liveQuery('MATCH (p:Person) RETURN p', { deps: ['Person', 'name'] });
    const b = store.liveQuery('MATCH (p:Person) RETURN p', { deps: ['name', 'Person'] });
    a.subscribe(() => {});
    b.subscribe(() => {});

    expect(b.getSnapshot()).toBe(a.getSnapshot());
    expect(counts.query).toBe(1); // shared despite reversed deps order
  });

  test('a fully-unsubscribed cell is evicted (a fresh identical query re-primes)', () => {
    const { store, counts } = fakeStore();
    const q = () => store.liveQuery('MATCH (p:Person) RETURN p', { deps: ['Person'] });

    const a = q();
    const stopA = a.subscribe(() => {});
    const b = q();
    const stopB = b.subscribe(() => {});
    a.getSnapshot();
    b.getSnapshot();
    expect(counts.query).toBe(1); // shared

    stopA();
    stopB(); // refs → 0 → cell evicted from the pool

    // A new identical handle gets a FRESH cell — it primes (recomputes) even
    // though the version never moved. Proof the old cell didn't linger.
    q().getSnapshot();
    expect(counts.query).toBe(2);
  });

  test('the same callback on two live queries is two independent registrations', () => {
    const { store, bump } = fakeStore();
    let calls = 0;
    const cb = (): void => {
      calls += 1;
    };

    // One `onChange` reference wired to two DISTINCT live queries.
    const a = store.liveQuery('MATCH (p:Person) RETURN p', { deps: ['Person'] });
    const b = store.liveQuery('MATCH (t:Team) RETURN t', { deps: ['Team'] });
    const stopA = a.subscribe(cb);
    b.subscribe(cb);

    store.mutate(() => bump());
    expect(calls).toBe(2); // both registrations fire (was 1: identity-collapsed)

    // Dropping A must not delete B's registration of the same callback.
    calls = 0;
    stopA();
    store.mutate(() => bump());
    expect(calls).toBe(1); // B still fires (was 0: the survivor was silently killed)
  });
});
