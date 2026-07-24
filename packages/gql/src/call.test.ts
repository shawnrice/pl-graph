import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';

import { query } from './index.js';

// The TinkerPop "modern" graph.
const modern = (): Graph => {
  const g = new Graph();
  const v = (id: string, label: string, name: string) =>
    g.addVertex({ id, labels: [label], properties: { name } });
  const marko = v('marko', 'Person', 'marko');
  const vadas = v('vadas', 'Person', 'vadas');
  const josh = v('josh', 'Person', 'josh');
  const peter = v('peter', 'Person', 'peter');
  const lop = v('lop', 'Software', 'lop');
  const ripple = v('ripple', 'Software', 'ripple');
  g.addEdge({ from: marko, to: vadas, labels: ['KNOWS'], properties: {} });
  g.addEdge({ from: marko, to: josh, labels: ['KNOWS'], properties: {} });
  g.addEdge({ from: marko, to: lop, labels: ['CREATED'], properties: {} });
  g.addEdge({ from: josh, to: ripple, labels: ['CREATED'], properties: {} });
  g.addEdge({ from: josh, to: lop, labels: ['CREATED'], properties: {} });
  g.addEdge({ from: peter, to: lop, labels: ['CREATED'], properties: {} });

  return g;
};

describe('named procedure CALL', () => {
  test('CALL pagerank YIELD — lop (most in-edges) is the top score', () => {
    // `node` is a live vertex handle, so `node.name` reads its property.
    const rows = query(
      modern(),
      'CALL pagerank() YIELD node, score RETURN node.name AS n ORDER BY score DESC, n LIMIT 1',
    );
    expect(rows).toEqual([{ n: 'lop' }]);
  });

  test('CALL degree — YIELD-less binds node + degree; one row per vertex', () => {
    const rows = query(modern(), 'CALL degree() RETURN node.name AS n, degree');
    expect(rows).toHaveLength(6);
  });

  test('YIELD aliasing + ISO WITH … WHERE filtering', () => {
    const rows = query(
      modern(),
      'CALL degree() YIELD node AS v, degree AS d WITH v, d WHERE d >= 3 RETURN v.name AS n ORDER BY n',
    );
    expect(rows).toEqual([{ n: 'marko' }]); // marko has out-degree 3
  });

  test('config writeProperty mutates the graph', () => {
    const g = modern();
    query(g, "CALL degree({writeProperty: 'deg'}) YIELD node RETURN node");
    const read = query(g, "MATCH (n) WHERE n.name = 'marko' RETURN n.deg AS d");
    expect(read).toEqual([{ d: 3 }]);
  });

  test('unknown procedure faults', () => {
    expect(() => query(modern(), 'CALL bogus() YIELD x RETURN x')).toThrow();
  });

  test('a camelCase procedure name suggests the snake_case one (native parity)', () => {
    // The GQL `CALL` catalog is snake_case; a camelCase spelling of a real
    // algorithm faults E_UNSUPPORTED with a "did you mean" hint. Messages are
    // asserted verbatim — the native engine emits the same bytes.
    const grab = (q: string): string => {
      try {
        query(modern(), q);
      } catch (e) {
        return (e as Error).message;
      }

      throw new Error('expected a fault');
    };

    expect(grab('CALL connectedComponents({}) YIELD node RETURN node')).toBe(
      "unknown procedure: connectedComponents (did you mean 'connected_components'?)",
    );
    expect(grab('CALL pageRank({}) YIELD node RETURN node')).toBe(
      "unknown procedure: pageRank (did you mean 'pagerank'?)",
    );
    expect(grab('CALL totallyBogus({}) YIELD node RETURN node')).toBe(
      'unknown procedure: totallyBogus',
    );
  });

  test('CALL betweenness({pivots: k}) actually samples (config reaches the algo)', () => {
    // The config-map path used to drop `pivots` (and `seedProperty`), silently
    // running exact O(V·E) betweenness. On a clean directed path an 8-of-16 pivot
    // sample scales differently from exact, so the two sums must differ. If pivots
    // is dropped, both are exact and equal.
    const g = new Graph();
    const n = 16;
    const ids = Array.from({ length: n }, (_, i) =>
      g.addVertex({ id: `n${i}`, labels: ['P'], properties: {} }),
    );

    for (let i = 0; i < n - 1; i++) {
      g.addEdge({ from: ids[i], to: ids[i + 1], labels: ['E'], properties: {} });
    }

    const sum = (cfg: string): number =>
      query(g, `CALL betweenness(${cfg}) YIELD node, centrality RETURN sum(centrality) AS s`)[0]
        .s as number;

    const exact = sum('{}');
    const sampled = sum('{pivots: 8}');

    expect(exact).toBeGreaterThan(0);
    expect(sampled).not.toBe(exact);
  });
});

describe('inline subquery CALL', () => {
  test('correlated subquery — lateral join, merges nested RETURN columns', () => {
    // For each person, count how many things they created.
    const rows = query(
      modern(),
      `MATCH (p:Person)
       CALL (p) {
         MATCH (p)-[:CREATED]->(w)
         RETURN count(w) AS created
       }
       RETURN p.name AS name, created ORDER BY name`,
    );
    expect(rows).toEqual([
      { name: 'josh', created: 2 },
      { name: 'marko', created: 1 },
      { name: 'peter', created: 1 },
      { name: 'vadas', created: 0 },
    ]);
  });

  test('row duplication — a subquery returning N rows fans the outer row out', () => {
    const rows = query(
      modern(),
      `MATCH (p:Person {name: 'marko'})
       CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS friend }
       RETURN friend ORDER BY friend`,
    );
    expect(rows).toEqual([{ friend: 'josh' }, { friend: 'vadas' }]);
  });

  test('non-OPTIONAL empty subquery drops the outer row; OPTIONAL keeps it', () => {
    // vadas created nothing → dropped without OPTIONAL.
    const dropped = query(
      modern(),
      `MATCH (p:Person)
       CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN w.name AS thing }
       RETURN p.name AS name ORDER BY name`,
    );
    expect(dropped.map((r) => r.name)).toEqual(['josh', 'josh', 'marko', 'peter']);

    // OPTIONAL keeps vadas, with the nested column null-filled.
    const kept = query(
      modern(),
      `MATCH (p:Person)
       OPTIONAL CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN w.name AS thing }
       RETURN p.name AS name, thing ORDER BY name, thing`,
    );
    expect(kept.some((r) => r.name === 'vadas' && r.thing === null)).toBe(true);
  });

  // Regression: an undeclared YIELD column used to bind silently to `undefined`,
  // so `YIELD nodeId` (or any typo) returned rows with the column simply missing —
  // a silent wrong answer, and a divergence from native, which raises
  // E_INVALID_VALUE. Found by the round-16 dogfood sim (`_p14_bug.ts`).
  test('YIELD rejects a column the procedure does not expose', () => {
    // `node` and the single result column are the whole contract.
    expect(() =>
      query(modern(), `CALL degree({direction: 'both'}) YIELD node, degree RETURN count(*) AS n`),
    ).not.toThrow();

    for (const bad of ['nodeId', 'totally_made_up_column']) {
      expect(() =>
        query(modern(), `CALL degree({direction: 'both'}) YIELD ${bad} RETURN count(*) AS n`),
      ).toThrow(new RegExp(`has no output column \`${bad}\``));
    }
  });

  test('scope isolation — an unscoped outer var is not visible to the subquery', () => {
    // `p` is not imported, so the inner MATCH (p) is a fresh unbound pattern
    // matching every vertex — not the outer marko.
    const rows = query(
      modern(),
      `MATCH (p:Person {name: 'marko'})
       CALL () { MATCH (n) RETURN count(n) AS total }
       RETURN total`,
    );
    expect(rows).toEqual([{ total: 6 }]); // all 6 vertices, not just marko's
  });
});
