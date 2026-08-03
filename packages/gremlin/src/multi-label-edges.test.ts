import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';

import { run } from './executor.js';
import { V, bothE, count, inE, out, outE, toList, traversal } from './index.js';

/**
 * An edge carries a SET of labels and the directional index buckets it under
 * every one of them, so a step that walks a bucket per named label sees a
 * two-label edge once per matching name. Native walks one adjacency list and
 * tests "does this edge carry any of these types", emitting it once — so a
 * duplicate here is a cross-engine divergence.
 */
/** The executor yields a stream; every assertion here is over the drained list. */
const arr = (r: Iterable<unknown>): unknown[] => [...r];

const twoLabelChain = (): Graph => {
  const graph = new Graph();

  for (const id of ['a', 'b']) {
    graph.addVertex({ id, labels: ['V'], properties: {} });
  }

  graph.addEdge({
    id: 'e0',
    from: graph.getVertexById('a')!,
    to: graph.getVertexById('b')!,
    labels: ['R', 'S'],
  });

  return graph;
};

describe('an edge matching several named labels traverses once', () => {
  test('naming both of an edge’s labels yields one edge', () => {
    const graph = twoLabelChain();

    expect(arr(run(traversal(V('a'), outE('R', 'S'), count()), graph))).toEqual([1]);
    expect(arr(run(traversal(V('a'), out('R', 'S'), count()), graph))).toEqual([1]);
  });

  test('in and both agree with out', () => {
    const graph = twoLabelChain();

    expect(arr(run(traversal(V('b'), inE('R', 'S'), count()), graph))).toEqual([1]);
    expect(arr(run(traversal(V('a'), bothE('R', 'S'), count()), graph))).toEqual([1]);
    expect(arr(run(traversal(V('b'), bothE('R', 'S'), count()), graph))).toEqual([1]);
  });

  test('naming both labels matches naming either one', () => {
    const graph = twoLabelChain();
    const one = arr(run(traversal(V('a'), outE('R'), toList()), graph));

    expect(arr(run(traversal(V('a'), outE('S'), toList()), graph))).toEqual(one);
    expect(arr(run(traversal(V('a'), outE('R', 'S'), toList()), graph))).toEqual(one);
    // ...and naming no label at all, since every edge here carries both.
    expect(arr(run(traversal(V('a'), outE(), toList()), graph))).toEqual(one);
  });

  test('a named label that no edge carries adds nothing', () => {
    const graph = twoLabelChain();

    expect(arr(run(traversal(V('a'), outE('R', 'ABSENT'), count()), graph))).toEqual([1]);
    expect(arr(run(traversal(V('a'), outE('ABSENT'), count()), graph))).toEqual([0]);
  });
});
