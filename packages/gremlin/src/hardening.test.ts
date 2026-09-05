import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { ErrorCode, hasErrorCode } from '@lenke/errors';

import { run } from './executor.js';
import {
  V,
  addV,
  both,
  bothE,
  drop,
  eq,
  groupCount,
  gt,
  inject,
  is,
  max,
  min,
  order,
  regex,
  repeat,
  serialize,
  shortestPath,
  sum,
  traversal,
  valueMap,
} from './index.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

const thrown = (fn: () => unknown): unknown => {
  try {
    fn();
  } catch (e) {
    return e;
  }

  return undefined;
};

/** One vertex with a single self-loop edge. */
const selfLoopGraph = (): Graph => {
  const g = new Graph();
  g.disableEvents();
  const n = g.addVertex({ id: 'n', labels: ['N'], properties: { name: 'n' } });
  g.addEdge({ id: 'e', from: n, to: n, labels: ['LOOP'], properties: {} });
  g.enableEvents();

  return g;
};

/** A complete directed graph on `n` vertices (every ordered pair has an edge). */
const completeGraph = (n: number): Graph => {
  const g = new Graph();
  g.disableEvents();
  const vs = Array.from({ length: n }, (_, i) =>
    g.addVertex({ id: `${i}`, labels: ['N'], properties: {} }),
  );
  let e = 0;

  for (const from of vs) {
    for (const to of vs) {
      if (from !== to) {
        g.addEdge({ id: `e${e++}`, from, to, labels: ['R'], properties: {} });
      }
    }
  }

  g.enableEvents();

  return g;
};

// --- G1: self-loop adjacency matches TinkerPop (both = union(out, in)) -------
// NOT a bug — documented here so it isn't "fixed" to dedup (which would diverge
// from TinkerPop; gql's ISO `~` dedups, gremlin's `both` does not).
describe('hardening G1: self-loop both/bothE match TinkerPop union semantics', () => {
  test('both() yields a self-loop neighbor twice (once per direction)', () => {
    expect(arr(run(traversal(V('n'), both()), selfLoopGraph()))).toHaveLength(2);
  });

  test('bothE() yields a self-loop edge twice (once per direction)', () => {
    expect(arr(run(traversal(V('n'), bothE()), selfLoopGraph()))).toHaveLength(2);
  });
});

// --- G2: repeat() work budget -----------------------------------------------
describe('hardening G2: repeat() bounds runaway work instead of OOM', () => {
  test('repeat(both()) with no termination on a dense cyclic graph raises ResourceExhausted', () => {
    const g = completeGraph(6);
    const e = thrown(() => arr(run(traversal(V(), repeat(both())), g)));
    expect(hasErrorCode(e, ErrorCode.ResourceExhausted)).toBe(true);
  });

  test('a normally-bounded repeat still works', () => {
    const g = completeGraph(4);
    // repeat(both()).times(2) is bounded — must not trip the budget.
    expect(() => arr(run(traversal(V('0'), repeat(both()).times(2)), g))).not.toThrow();
  });
});

// --- G3: mutation steps snapshot their input --------------------------------
describe('hardening G3: mutation does not loop over a live source view', () => {
  test('V().addV() adds one vertex per pre-existing vertex and terminates', () => {
    const g = completeGraph(3); // 3 vertices
    arr(run(traversal(V(), addV('Q')), g));
    expect(g.vertexCount).toBe(6); // 3 original + 3 new, then stops
  });
});

// --- G4: dropping the same vertex twice is a no-op, not a crash --------------
describe('hardening G4: drop is idempotent on an already-removed vertex', () => {
  test('inject(v, v).drop() does not throw and removes the vertex once', () => {
    const g = completeGraph(3);
    const v = g.getVertexById('0')!;
    const before = g.vertexCount;
    expect(() => arr(run(traversal(inject(v, v), drop()), g))).not.toThrow();
    expect(g.vertexCount).toBe(before - 1);
  });
});

// --- G7/G8/G9: TinkerPop Comparable semantics (throw on incomparable) --------
describe('hardening G7-G9: comparison/order/aggregation reject incomparable types', () => {
  test('ordering predicate on a comparable type still works', () => {
    expect(arr(run(traversal(inject(5, 1, 9), is(gt(3))), new Graph()))).toEqual([5, 9]);
  });

  test('an ordering PREDICATE over incomparable types FILTERS, it does not throw', () => {
    // TinkerPop's has()/is()/where() filter a cross-type comparison (verified on
    // createModern(): `g.V().has('name', gte(0))` matches nothing) — they do NOT
    // throw. `max()`/`min()`/`sum()` still throw (below); order() total-orders.
    expect(arr(run(traversal(inject(5), is(gt('x'))), new Graph()))).toEqual([]);
  });

  // order()/min()/max() use a TOTAL order over mixed types — numbers before strings —
  // rather than throwing, matching the engine (deterministic + byte-identical) and
  // TinkerPop's Orderability. Only sum()/mean() (numeric) and the ordering PREDICATES
  // fault/filter on a cross-type value.
  test('order() over mixed types TOTAL-orders (numbers before strings), no throw', () => {
    expect(arr(run(traversal(inject(3, 'a', 1), order()), new Graph()))).toEqual([1, 3, 'a']);
  });

  test('min()/max() TOTAL-order any comparable (consistent with order(), like TinkerPop)', () => {
    // Not numeric-only: min/max use the same total order as order() above — numbers
    // before strings — so a mixed stream reduces deterministically instead of faulting.
    expect(arr(run(traversal(inject(3, 'a'), min()), new Graph()))).toEqual([3]);
    expect(arr(run(traversal(inject(3, 'a'), max()), new Graph()))).toEqual(['a']);
  });

  test('sum() of a non-numeric value throws', () => {
    const e = thrown(() => arr(run(traversal(inject(1, 'x'), sum()), new Graph())));
    expect(hasErrorCode(e, ErrorCode.InvalidValue)).toBe(true);
  });

  test('same-type aggregation still works', () => {
    expect(arr(run(traversal(inject(3, 1, 2), order()), new Graph()))).toEqual([1, 2, 3]);
    expect(arr(run(traversal(inject(3, 1, 2), sum()), new Graph()))).toEqual([6]);
  });
});

// --- G11: group/groupCount key plain objects structurally --------------------
describe('hardening G11: groupCount keys equal object projections together', () => {
  test('two vertices with the same valueMap form one group', () => {
    const g = new Graph();
    g.disableEvents();
    g.addVertex({ id: 'a', labels: ['N'], properties: { name: 'x' } });
    g.addVertex({ id: 'b', labels: ['N'], properties: { name: 'x' } });
    g.enableEvents();
    const r = arr(run(traversal(V(), groupCount().by(valueMap('name'))), g));
    const m = r[0] as Map<unknown, number>;
    // Both vertices share {name:'x'} → one group of count 2 (not two ref groups).
    expect(m.size).toBe(1);
    expect([...m.values()]).toEqual([2]);
  });
});

// --- lows: regex build-time validation + serialize non-serializable guard ----
describe('hardening lows: clean errors for bad regex / non-serializable plans', () => {
  test('an invalid regex pattern errors at build, not mid-stream', () => {
    expect(() => regex('[')).toThrow();
  });

  test('a valid regex still builds', () => {
    expect(() => regex('^ma')).not.toThrow();
  });

  test('serializing a plan with a BigInt literal is a clean error, not a raw TypeError', () => {
    const e = thrown(() => serialize(traversal(inject(1), is(eq(10n)))));
    expect(hasErrorCode(e, ErrorCode.InvalidValue)).toBe(true);
  });
});

// --- low: shortestPath() bounds exponential path reconstruction --------------
describe('hardening low: shortestPath() path-count budget', () => {
  // An N-diamond chain has 2^N equal-length shortest paths from m0 to mN.
  const diamondChain = (n: number): Graph => {
    const g = new Graph();
    g.disableEvents();
    g.addVertex({ id: 'm0', labels: ['N'], properties: {} });
    let e = 0;
    const edge = (from: string, to: string) =>
      g.addEdge({
        id: `e${e++}`,
        from: g.getVertexById(from)!,
        to: g.getVertexById(to)!,
        labels: ['R'],
        properties: {},
      });

    for (let i = 0; i < n; i++) {
      g.addVertex({ id: `m${i + 1}`, labels: ['N'], properties: {} });
      g.addVertex({ id: `a${i}`, labels: ['N'], properties: {} });
      g.addVertex({ id: `b${i}`, labels: ['N'], properties: {} });
      edge(`m${i}`, `a${i}`);
      edge(`a${i}`, `m${i + 1}`);
      edge(`m${i}`, `b${i}`);
      edge(`b${i}`, `m${i + 1}`);
    }

    g.enableEvents();

    return g;
  };

  test('an exponential shortest-path set raises ResourceExhausted', () => {
    const g = diamondChain(22); // 2^22 ≈ 4.2M shortest paths
    const err = thrown(() => arr(run(traversal(V('m0'), shortestPath()), g)));
    expect(hasErrorCode(err, ErrorCode.ResourceExhausted)).toBe(true);
  });

  test('a small shortest-path query still works', () => {
    const g = diamondChain(2); // 4 shortest paths m0→m2
    expect(() => arr(run(traversal(V('m0'), shortestPath()), g))).not.toThrow();
  });
});
