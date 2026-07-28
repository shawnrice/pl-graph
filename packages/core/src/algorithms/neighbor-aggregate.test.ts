import { describe, expect, test } from 'bun:test';

import { Graph } from '../core/Graph.js';
import { neighborAggregate } from './neighbor-aggregate.js';

// a→b, b→c, a→c with list features h: a=[1,2], b=[3,4], c=[5,6]. Mirrors the Rust
// `neighbor_aggregate` known-answer test one-for-one.
const featured = (): Graph => {
  const g = new Graph();

  for (const [id, h] of [
    ['a', [1, 2]],
    ['b', [3, 4]],
    ['c', [5, 6]],
  ] as const) {
    g.addVertex({ id, labels: ['N'], properties: { h } });
  }

  for (const [from, to] of [
    ['a', 'b'],
    ['b', 'c'],
    ['a', 'c'],
  ] as const) {
    g.addEdge({ from: g.getVertexById(from)!, to: g.getVertexById(to)!, labels: ['R'] });
  }

  return g;
};

describe('neighborAggregate', () => {
  test('mean / sum over out-neighbours', async () => {
    expect(
      await neighborAggregate({ feature: 'h', op: 'mean', direction: 'out' }, featured()),
    ).toEqual([
      { node: 'a', vector: [4, 5] }, // mean(b, c)
      { node: 'b', vector: [5, 6] }, // c
      { node: 'c', vector: [0, 0] }, // no out-neighbours → zero vector
    ]);
    expect(
      await neighborAggregate({ feature: 'h', op: 'sum', direction: 'out' }, featured()),
    ).toEqual([
      { node: 'a', vector: [8, 10] },
      { node: 'b', vector: [5, 6] },
      { node: 'c', vector: [0, 0] },
    ]);
  });

  test('includeSelf + both makes every vertex see a,b,c → mean [3,4]', async () => {
    expect(
      await neighborAggregate(
        { feature: 'h', op: 'mean', direction: 'both', includeSelf: true },
        featured(),
      ),
    ).toEqual([
      { node: 'a', vector: [3, 4] },
      { node: 'b', vector: [3, 4] },
      { node: 'c', vector: [3, 4] },
    ]);
  });

  test('max over out-neighbours', async () => {
    expect(
      await neighborAggregate({ feature: 'h', op: 'max', direction: 'out' }, featured()),
    ).toEqual([
      { node: 'a', vector: [5, 6] },
      { node: 'b', vector: [5, 6] },
      { node: 'c', vector: [0, 0] },
    ]);
  });

  test('writeProperty stores the aggregate list', async () => {
    const g = featured();
    await neighborAggregate({ feature: 'h', op: 'sum', direction: 'out', writeProperty: 'agg' }, g);
    expect(g.getVertexById('a')!.getProperty<number[]>('agg')).toEqual([8, 10]);
  });

  const rejects = async (p: Promise<unknown>): Promise<boolean> => {
    try {
      await p;

      return false;
    } catch {
      return true;
    }
  };

  test('rejects a missing feature / bad op / bad direction', async () => {
    expect(await rejects(neighborAggregate({ op: 'mean' }, featured()))).toBe(true);
    expect(
      await rejects(neighborAggregate({ feature: 'h', op: 'nope' as never }, featured())),
    ).toBe(true);
    expect(
      await rejects(
        neighborAggregate({ feature: 'h', direction: 'sideways' as never }, featured()),
      ),
    ).toBe(true);
  });

  // a[1,2]→b[3,4] (w2), a→c[5,6] (w1), b→c (w3).
  const weightedFeatured = (): Graph => {
    const g = new Graph();

    for (const [id, h] of [
      ['a', [1, 2]],
      ['b', [3, 4]],
      ['c', [5, 6]],
    ] as const) {
      g.addVertex({ id, labels: ['N'], properties: { h } });
    }

    for (const [from, to, w] of [
      ['a', 'b', 2],
      ['a', 'c', 1],
      ['b', 'c', 3],
    ] as const) {
      g.addEdge({
        from: g.getVertexById(from)!,
        to: g.getVertexById(to)!,
        labels: ['R'],
        properties: { w },
      });
    }

    return g;
  };

  test('weighted sum and mean over out-neighbours', async () => {
    // sum: a = 2·[3,4]+1·[5,6] = [11,14]; b = 3·[5,6] = [15,18].
    expect(
      await neighborAggregate(
        { feature: 'h', op: 'sum', direction: 'out', weightProperty: 'w' },
        weightedFeatured(),
      ),
    ).toEqual([
      { node: 'a', vector: [11, 14] },
      { node: 'b', vector: [15, 18] },
      { node: 'c', vector: [0, 0] },
    ]);
    // mean divides by the WEIGHT sum: a = [11,14]/3.
    expect(
      await neighborAggregate(
        { feature: 'h', op: 'mean', direction: 'out', weightProperty: 'w' },
        weightedFeatured(),
      ),
    ).toEqual([
      { node: 'a', vector: [11 / 3, 14 / 3] },
      { node: 'b', vector: [5, 6] },
      { node: 'c', vector: [0, 0] },
    ]);
  });

  test('gcn norm on a two-node graph gives clean 1/2 coefficients', async () => {
    // a[1,2]→b[3,4], both + includeSelf → degree 2 each → every coef 1/2 → ½([1,2]+[3,4]) = [2,3].
    const g = new Graph();
    g.addVertex({ id: 'a', labels: ['N'], properties: { h: [1, 2] } });
    g.addVertex({ id: 'b', labels: ['N'], properties: { h: [3, 4] } });
    g.addEdge({ from: g.getVertexById('a')!, to: g.getVertexById('b')!, labels: ['R'] });
    expect(
      await neighborAggregate(
        { feature: 'h', op: 'sum', direction: 'both', includeSelf: true, norm: 'gcn' },
        g,
      ),
    ).toEqual([
      { node: 'a', vector: [2, 3] },
      { node: 'b', vector: [2, 3] },
    ]);
  });

  test('rejects a weight / norm with max|min, and a bad norm', async () => {
    const g = weightedFeatured();
    expect(
      await rejects(neighborAggregate({ feature: 'h', op: 'max', weightProperty: 'w' }, g)),
    ).toBe(true);
    expect(await rejects(neighborAggregate({ feature: 'h', op: 'min', norm: 'gcn' }, g))).toBe(
      true,
    );
    expect(
      await rejects(neighborAggregate({ feature: 'h', op: 'sum', norm: 'nope' as never }, g)),
    ).toBe(true);
  });
});
