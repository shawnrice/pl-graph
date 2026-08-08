import { describe, expect, test } from 'bun:test';

import { Graph } from '../core/Graph.js';
import { degree } from './degree.js';
import { labelPropagation } from './label-propagation.js';
import { shortestPath } from './shortest-path.js';

/**
 * Edges carry a SET of labels, and the directional indexes bucket an edge under
 * every one of them. Any reader that reconstructs "all of a vertex's edges" by
 * unioning those buckets therefore sees a two-label edge twice — which is the
 * shape of these tests. Native walks its adjacency once and counts each edge
 * once, so a duplicate here is a byte-identity divergence, not just a wrong
 * number.
 */
const twoLabelChain = (): Graph => {
  const g = new Graph();

  for (const id of ['a', 'b', 'c']) {
    g.addVertex({ id, labels: ['V'], properties: {} });
  }

  for (const [id, from, to] of [
    ['e0', 'a', 'b'],
    ['e1', 'b', 'c'],
  ] as const) {
    g.addEdge({
      id,
      from: g.getVertexById(from)!,
      to: g.getVertexById(to)!,
      labels: ['R', 'NOISE'],
    });
  }

  return g;
};

describe('an edge in two label buckets is still one edge', () => {
  test('degree counts a two-label edge once', async () => {
    const g = twoLabelChain();

    expect((await degree({ direction: 'out' }, g)).map((r) => r.degree)).toEqual([1, 1, 0]);
    expect((await degree({ direction: 'both' }, g)).map((r) => r.degree)).toEqual([1, 2, 1]);
  });

  test('degree agrees with itself whichever label is asked for', async () => {
    const g = twoLabelChain();
    const r = (await degree({ direction: 'both', edgeLabel: 'R' }, g)).map((x) => x.degree);

    expect(r).toEqual(
      (await degree({ direction: 'both', edgeLabel: 'NOISE' }, g)).map((x) => x.degree),
    );
    expect(r).toEqual([1, 2, 1]);
  });

  test('label propagation sees each neighbour once', async () => {
    const g = twoLabelChain();
    const all = (await labelPropagation({}, g)).map((r) => r.label);

    // The row property is `label` (`LabelRow = AlgorithmRow<'label', string>`).
    // This read `r.communityId` until 2026-08-08, which is not a property of
    // that type: every element was `undefined`, so the comparisons below held
    // trivially and the test asserted nothing. `bun test` could not see it —
    // only `tsc` could, which is why it survived until a CI run.
    expect(all.every((l) => typeof l === 'string')).toBe(true);

    // Every edge is both `R` and `NOISE`, so filtering on either must give the
    // same communities as not filtering at all. A doubled neighbour list would
    // still converge, but the tie-breaks it takes on the way differ.
    expect(all).toEqual((await labelPropagation({ edgeLabel: 'R' }, g)).map((r) => r.label));
    expect(all).toEqual((await labelPropagation({ edgeLabel: 'NOISE' }, g)).map((r) => r.label));
  });

  test('shortest path does not walk a two-label edge twice', async () => {
    const g = twoLabelChain();
    const unfiltered = await shortestPath({ source: 'a', target: 'c' }, g);

    expect(unfiltered).toEqual(await shortestPath({ source: 'a', target: 'c', edgeLabel: 'R' }, g));
    expect(unfiltered.map((r) => r.distance)).toEqual([2]);
  });
});
