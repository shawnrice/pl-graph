import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';

import { run } from './executor.js';
import {
  E,
  V,
  count,
  dedupe,
  eq,
  gt,
  has,
  hasLabel,
  max,
  order,
  otherV,
  out,
  outE,
  inV,
  repeat,
  simplePath,
  sum,
  take,
  toList,
  traversal,
  values,
} from './index.js';

const buildSocialGraph = () => {
  const g = new Graph();
  g.disableEvents();
  // Vertices: 1=alice(28), 2=bob(35), 3=charlie(40), 4=diane(22)
  const alice = g.addVertex({ id: '1', labels: ['user'], properties: { name: 'alice', age: 28 } });
  const bob = g.addVertex({ id: '2', labels: ['user'], properties: { name: 'bob', age: 35 } });
  const charlie = g.addVertex({
    id: '3',
    labels: ['user'],
    properties: { name: 'charlie', age: 40 },
  });
  const diane = g.addVertex({ id: '4', labels: ['user'], properties: { name: 'diane', age: 22 } });
  // Edges (knows): alice→bob, bob→charlie, charlie→alice (cycle), alice→diane
  g.addEdge({ id: 'e1', from: alice, to: bob, labels: ['knows'], properties: {} });
  g.addEdge({ id: 'e2', from: bob, to: charlie, labels: ['knows'], properties: {} });
  g.addEdge({ id: 'e3', from: charlie, to: alice, labels: ['knows'], properties: {} });
  g.addEdge({ id: 'e4', from: alice, to: diane, labels: ['knows'], properties: {} });
  g.enableEvents();

  return g;
};

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('executor', () => {
  test('V() yields all vertices', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V()), g);
    const ids = arr(result).map((v: any) => v.id);
    expect(ids.sort()).toEqual(['1', '2', '3', '4']);
  });

  test('V(id) yields a single vertex', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V('1')), g);
    const xs = arr(result) as any[];
    expect(xs).toHaveLength(1);
    expect(xs[0].properties.name).toBe('alice');
  });

  test('out() walks edges', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V('1'), out('knows'), values('name')), g);
    expect((arr(result) as string[]).sort()).toEqual(['bob', 'diane']);
  });

  test('has() filters by property', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V(), hasLabel('user'), has('age', gt(30)), values('name')), g);
    expect((arr(result) as string[]).sort()).toEqual(['bob', 'charlie']);
  });

  test('count() returns scalar', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V(), count()), g);
    expect(arr(result)).toEqual([4]);
  });

  test('toList() collects', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V(), values('name'), toList()), g);
    const xs = arr(result);
    expect(xs).toHaveLength(1);
    expect((xs[0] as string[]).sort()).toEqual(['alice', 'bob', 'charlie', 'diane']);
  });

  test('outE then inV equivalents to out', () => {
    const g = buildSocialGraph();
    const a = run(traversal(V('1'), out('knows'), values('name')), g);
    const b = run(traversal(V('1'), outE('knows'), inV(), values('name')), g);
    expect((arr(a) as string[]).sort()).toEqual((arr(b) as string[]).sort());
  });

  test('repeat(out).times(2) walks two hops', () => {
    const g = buildSocialGraph();
    // 1 → 2 → 3 (and 1 → 4 but 4 has no outgoing edges)
    const result = run(traversal(V('1'), repeat(out('knows')).times(2), values('name')), g);
    expect((arr(result) as string[]).sort()).toEqual(['charlie']);
  });

  test('cycle does NOT cause infinite loop with simplePath', () => {
    const g = buildSocialGraph();
    // 1 → 2 → 3 → 1 forms a cycle. Without simplePath, repeating times(5)
    // would happily walk the cycle. With simplePath, paths that revisit
    // are dropped.
    const result = run(
      traversal(V('1'), repeat(out('knows')).times(5), simplePath(), values('name')),
      g,
    );
    // After 5 hops with simplePath, no path can avoid revisiting on this graph.
    expect(arr(result)).toEqual([]);
  });

  test('cycle handled at 3 hops (alice → bob → charlie → alice would revisit)', () => {
    const g = buildSocialGraph();
    const result = run(
      traversal(V('1'), repeat(out('knows')).times(3), simplePath(), values('name')),
      g,
    );
    // 3 hops from alice without revisits: alice→bob→charlie→? — only revisits.
    // None survive simplePath on this small cyclic graph.
    expect(arr(result)).toEqual([]);
  });

  test('take limits results', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V(), take(2)), g);
    expect(arr(result)).toHaveLength(2);
  });

  test('dedupe removes duplicates', () => {
    const g = buildSocialGraph();
    // both() will yield neighbors via either direction, producing duplicates
    // when there are reciprocal-ish paths.
    const result = run(traversal(V('1'), out('knows'), out('knows'), dedupe(), values('name')), g);
    const names = (arr(result) as string[]).sort();
    // 1 → 2 → 3, and 1 → 4 has no out, so unique result = [charlie]
    expect(names).toEqual(['charlie']);
  });

  test('plan starting with E() yields edges', () => {
    const g = buildSocialGraph();
    const result = run(traversal(E()), g);
    expect(arr(result)).toHaveLength(4);
  });

  test('has() with eq predicate', () => {
    const g = buildSocialGraph();
    const result = run(traversal(V(), has('name', eq('bob'))), g);
    const xs = arr(result) as any[];
    expect(xs).toHaveLength(1);
    expect(xs[0].id).toBe('2');
  });
});

describe('static frontier-type faults (parity with the native engine)', () => {
  const g = buildSocialGraph();
  const throws = (plan: ReturnType<typeof traversal>) => expect(() => run(plan, g));
  const ok = (plan: ReturnType<typeof traversal>) => expect(() => arr(run(plan, g)));

  test('aggregate / order over a graph element faults — incl. an EMPTY frontier', () => {
    // A vertex/edge is not a number and has no natural order.
    throws(traversal(V(), sum())).toThrow(/graph elements/);
    throws(traversal(V(), max())).toThrow(/graph elements/);
    throws(traversal(V(), order())).toThrow(/graph elements/);
    throws(traversal(V(), has('age', gt(1)), sum())).toThrow(/graph elements/); // filter preserves
    // The empty-frontier case is why this is STATIC, not runtime: `out('NOPE')` yields no
    // vertex, so a runtime check would give null — but the frontier KIND is still an element.
    throws(traversal(V(), out('NOPE'), sum())).toThrow(/graph elements/);
  });

  test('inV/outV/bothV/otherV require an edge frontier — incl. an EMPTY frontier', () => {
    throws(traversal(V(), otherV())).toThrow(/requires an edge/);
    throws(traversal(V(), inV())).toThrow(/requires an edge/);
    throws(traversal(V(), out('NOPE'), otherV())).toThrow(/requires an edge/);
  });

  test('projected / scalar frontiers are fine — no false positives', () => {
    ok(traversal(V(), values('age'), sum())).not.toThrow();
    ok(traversal(V(), values('age'), order())).not.toThrow();
    ok(traversal(V(), count())).not.toThrow();
    ok(traversal(V(), outE('knows'), otherV(), count())).not.toThrow();
  });
});
