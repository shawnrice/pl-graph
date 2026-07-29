import { describe, expect, test } from 'bun:test';

import { Graph } from './Graph.js';

const personGraph = (): Graph => {
  const graph = new Graph();
  graph.addVertex({ id: 'a', labels: ['Person'], properties: { name: 'marko', age: 29 } });
  graph.addVertex({ id: 'b', labels: ['Person'], properties: { name: 'vadas', age: 27 } });
  graph.addVertex({ id: 'c', labels: ['Person'], properties: { name: 'josh', age: 32 } });
  graph.addVertex({ id: 'd', labels: ['Person'], properties: { name: 'peter', age: 35 } });

  return graph;
};

const ids = (vs: Iterable<{ id: string }>): string[] => Array.from(vs, (v) => v.id).sort();

describe('PropertyIndex equality', () => {
  test('createIndex backfills existing vertices', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    expect(ids(graph.getVerticesByProperty('age', 29))).toEqual(['a']);
    expect(ids(graph.getVerticesByProperty('age', 27))).toEqual(['b']);
  });

  test('unindexed keys return an empty set', () => {
    const graph = personGraph();
    // name was never indexed -> no seed, falls back to scan elsewhere
    expect(ids(graph.getVerticesByProperty('name', 'marko'))).toEqual([]);
  });

  test('insert after index keeps it current', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.addVertex({ id: 'e', labels: ['Person'], properties: { age: 29 } });
    expect(ids(graph.getVerticesByProperty('age', 29))).toEqual(['a', 'e']);
  });

  test('type-tagged values do not collide', () => {
    const graph = new Graph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['v'] });
    graph.addVertex({ id: 'num', labels: [], properties: { v: 1 } });
    graph.addVertex({ id: 'str', labels: [], properties: { v: '1' } });
    graph.addVertex({ id: 'bool', labels: [], properties: { v: true } });
    expect(ids(graph.getVerticesByProperty('v', 1))).toEqual(['num']);
    expect(ids(graph.getVerticesByProperty('v', '1'))).toEqual(['str']);
    expect(ids(graph.getVerticesByProperty('v', true))).toEqual(['bool']);
  });

  test('setProperty moves a vertex between buckets', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.getVertexById('a')!.setProperty('age', 27);
    expect(ids(graph.getVerticesByProperty('age', 29))).toEqual([]);
    expect(ids(graph.getVerticesByProperty('age', 27))).toEqual(['a', 'b']);
  });

  test('removeProperty de-indexes the value', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.getVertexById('a')!.removeProperty('age');
    expect(ids(graph.getVerticesByProperty('age', 29))).toEqual([]);
  });

  test('removeVertex de-indexes its properties', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.removeVertex('c');
    expect(ids(graph.getVerticesByProperty('age', 32))).toEqual([]);
  });

  test('setProperties maintains every changed key', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    graph.getVertexById('a')!.setProperties({ age: 40, name: 'marco' });
    expect(ids(graph.getVerticesByProperty('age', 40))).toEqual(['a']);
    expect(ids(graph.getVerticesByProperty('name', 'marco'))).toEqual(['a']);
    expect(ids(graph.getVerticesByProperty('name', 'marko'))).toEqual([]);
  });

  test('an observing listener does not block the write — the index still updates', () => {
    // Events are observation-only: a listener reacts (here, just watches) but
    // cannot veto, so the write commits and the index reflects it.
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    let seen = 0;
    graph.on('@graph/VertexPropertyChanged', () => {
      seen += 1;
    });
    graph.getVertexById('a')!.setProperty('age', 99);
    expect(seen).toBe(1); // the listener saw the change…
    expect(ids(graph.getVerticesByProperty('age', 99))).toEqual(['a']); // …and it committed
    expect(ids(graph.getVerticesByProperty('age', 29))).toEqual([]);
  });
});

describe('PropertyIndex range', () => {
  test('open lower bound stays within the numeric type', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    expect(ids(graph.getVerticesByPropertyRange('age', { gt: 30 }))).toEqual(['c', 'd']);
    expect(ids(graph.getVerticesByPropertyRange('age', { gte: 32 }))).toEqual(['c', 'd']);
  });

  test('closed range is inclusive on both ends', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    expect(ids(graph.getVerticesByPropertyRange('age', { gte: 27, lte: 32 }))).toEqual([
      'a',
      'b',
      'c',
    ]);
    expect(ids(graph.getVerticesByPropertyRange('age', { gt: 27, lt: 35 }))).toEqual(['a', 'c']);
  });

  test('a numeric bound never bleeds into string values', () => {
    const graph = new Graph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['v'] });
    graph.addVertex({ id: 'n', labels: [], properties: { v: 5 } });
    graph.addVertex({ id: 's', labels: [], properties: { v: 'zzz' } });
    expect(ids(graph.getVerticesByPropertyRange('v', { gt: 0 }))).toEqual(['n']);
  });

  test('range reflects mutations', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.getVertexById('b')!.setProperty('age', 50);
    expect(ids(graph.getVerticesByPropertyRange('age', { gt: 30 }))).toEqual(['b', 'c', 'd']);
  });

  test('countEquals / countRange match the set sizes without building them', () => {
    const graph = personGraph();
    const index = graph.vertexPropertyIndex;
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    // countEquals: undefined for unindexed key, a count for an indexed one.
    expect(index.countEquals('name', 'marko')).toBeUndefined();
    expect(index.countEquals('age', 29)).toBe(1);
    expect(index.countEquals('age', 999)).toBe(0);
    // countRange agrees with range().size and stays type-clamped.
    expect(index.countRange('age', { gt: 30 })).toBe(2); // josh, peter
    expect(index.countRange('age', { gte: 27, lte: 32 })).toBe(3);
    expect(index.countRange('name', { gt: 0 })).toBeUndefined(); // unindexed
  });

  test('the ordered view is maintained after it is lazily built', () => {
    const graph = new Graph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.addVertex({ id: 'a', labels: [], properties: { age: 10 } });
    graph.addVertex({ id: 'b', labels: [], properties: { age: 30 } });
    // First range query materializes the ordered view from the buckets.
    expect(ids(graph.getVerticesByPropertyRange('age', { gte: 20 }))).toEqual(['b']);
    // A new distinct value inserted *after* the build must show up...
    graph.addVertex({ id: 'c', labels: [], properties: { age: 40 } });
    expect(ids(graph.getVerticesByPropertyRange('age', { gte: 20 }))).toEqual(['b', 'c']);
    // ...and a removal must drop out.
    graph.removeVertex('b');
    expect(ids(graph.getVerticesByPropertyRange('age', { gte: 20 }))).toEqual(['c']);
  });

  test('B+-tree range scans match brute force across a multi-level tree', () => {
    // > 64^2 distinct values forces internal splits and a 3-level tree.
    const COUNT = 5000;
    const graph = new Graph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['k'] });
    const present = new Set<number>();
    let seed = 12345;
    const rnd = (n: number): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;

      return seed % n;
    };
    // Insert COUNT distinct values in shuffled order (before any range query, so
    // the ordered view is later bulk-built; some after, exercising incremental).
    const values = Array.from({ length: COUNT }, (_, i) => i * 3 - COUNT);

    for (let i = values.length - 1; i > 0; i--) {
      const j = rnd(i + 1);
      [values[i], values[j]] = [values[j], values[i]];
    }

    const insert = (v: number): void => {
      graph.addVertex({ id: `v${v}`, labels: [], properties: { k: v } });
      present.add(v);
    };
    const remove = (v: number): void => {
      graph.removeVertex(`v${v}`);
      present.delete(v);
    };
    const rangeVals = (lo: number, hi: number): number[] =>
      Array.from(graph.getVerticesByPropertyRange('k', { gte: lo, lt: hi }), (x) =>
        Number((x.properties as { k: number }).k),
      ).sort((a, b) => a - b);
    const brute = (lo: number, hi: number): number[] =>
      [...present].filter((v) => v >= lo && v < hi).sort((a, b) => a - b);

    values.slice(0, COUNT - 50).forEach(insert); // bulk-built on first query
    let lo = -200;
    let hi = 2000;
    expect(rangeVals(lo, hi)).toEqual(brute(lo, hi)); // materializes the tree
    values.slice(COUNT - 50).forEach(insert); // incremental adds after build

    for (let q = 0; q < 20; q++) {
      lo = rnd(COUNT * 3) - COUNT;
      hi = lo + rnd(COUNT);
      expect(rangeVals(lo, hi)).toEqual(brute(lo, hi));
      expect(graph.getVerticesByPropertyRange('k', { gte: lo, lt: hi }).size).toBe(
        brute(lo, hi).length,
      );
    }

    // Delete a third at random, then re-check (snapshot — we mutate `present`).
    for (const v of Array.from(present)) {
      if (rnd(3) === 0) {
        remove(v);
      }
    }

    for (let q = 0; q < 20; q++) {
      lo = rnd(COUNT * 3) - COUNT;
      hi = lo + rnd(COUNT);
      expect(rangeVals(lo, hi)).toEqual(brute(lo, hi));
    }

    expect(graph.getVerticesByPropertyRange('k', { gt: -1e9 }).size).toBe(present.size);
  });

  test('a mixed-type column rebuilds with the full comparator', () => {
    const graph = new Graph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['v'] });
    graph.addVertex({ id: 'n1', labels: [], properties: { v: 10 } });
    graph.addVertex({ id: 'n2', labels: [], properties: { v: 20 } });
    // First range query builds a numeric (monomorphic) ordered view.
    expect(graph.getVerticesByPropertyRange('v', { gte: 15 }).size).toBe(1);
    // A string value breaks the numeric assumption → view invalidated + rebuilt.
    graph.addVertex({ id: 's1', labels: [], properties: { v: 'apple' } });
    graph.addVertex({ id: 's2', labels: [], properties: { v: 'pear' } });
    expect(ids(graph.getVerticesByPropertyRange('v', { gte: 15 }))).toEqual(['n2']); // numbers only
    expect(ids(graph.getVerticesByPropertyRange('v', { gte: 'a', lt: 'q' }))).toEqual(['s1', 's2']); // strings only
    expect(graph.getVerticesByProperty('v', 10).size).toBe(1); // equality unaffected
  });

  test('the ordered structure stays correct across heavy insert/delete churn', () => {
    const graph = new Graph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['k'] });
    // Insert 500 distinct values in shuffled order.
    const order = Array.from({ length: 500 }, (_, i) => i);

    for (let i = order.length - 1; i > 0; i--) {
      const j = (i * 2654435761) % (i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }

    for (const v of order) {
      graph.addVertex({ id: `v${v}`, labels: [], properties: { k: v } });
    }

    expect(graph.getVerticesByPropertyRange('k', { gte: 100, lt: 200 }).size).toBe(100);

    // Delete every even value, then re-check the range.
    for (const v of order) {
      if (v % 2 === 0) {
        graph.removeVertex(`v${v}`);
      }
    }

    expect(graph.getVerticesByPropertyRange('k', { gte: 100, lt: 200 }).size).toBe(50); // odds only
    expect(graph.getVerticesByProperty('k', 150).size).toBe(0); // deleted
    expect(graph.getVerticesByProperty('k', 151).size).toBe(1); // kept
    expect(graph.getVerticesByPropertyRange('k', { gt: -1 }).size).toBe(250); // all odds 1..499
  });
});

describe('PropertyIndex snapshots and edges', () => {
  test('clone does not alias the source buckets', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    const snapshot = graph.clone();

    // Mutating the original must not change the snapshot's view.
    graph.getVertexById('a')!.setProperty('age', 100);
    expect(ids(snapshot.getVerticesByProperty('age', 29))).toEqual(['a']);
    expect(ids(graph.getVerticesByProperty('age', 29))).toEqual([]);
  });

  test('edge property indexes work the same way', () => {
    const graph = new Graph();
    const a = graph.addVertex({ id: 'a', labels: ['P'], properties: {} });
    const b = graph.addVertex({ id: 'b', labels: ['P'], properties: {} });
    graph.createIndex({ on: 'edge', kind: 'hash', keys: ['weight'] });
    graph.addEdge({ id: 'e1', from: a, to: b, labels: ['knows'], properties: { weight: 5 } });
    graph.addEdge({ id: 'e2', from: a, to: b, labels: ['knows'], properties: { weight: 9 } });
    expect(Array.from(graph.getEdgesByProperty('weight', 5), (e) => e.id)).toEqual(['e1']);
    expect(
      Array.from(graph.getEdgesByPropertyRange('weight', { gte: 5 }), (e) => e.id).sort(),
    ).toEqual(['e1', 'e2']);
  });

  test('vertexIndexes / edgeIndexes report declared keys', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['name'] });
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.createIndex({ on: 'edge', kind: 'hash', keys: ['weight'] });
    expect(graph.vertexIndexes().sort()).toEqual(['age', 'name']);
    expect(graph.edgeIndexes()).toEqual(['weight']);
    graph.dropVertexIndex('age');
    expect(graph.vertexIndexes()).toEqual(['name']);
  });

  test('truncate empties buckets but keeps declarations', () => {
    const graph = personGraph();
    graph.createIndex({ on: 'vertex', kind: 'hash', keys: ['age'] });
    graph.truncate();
    expect(graph.vertexPropertyIndex.isIndexed('age')).toBe(true);
    graph.addVertex({ id: 'z', labels: [], properties: { age: 29 } });
    expect(ids(graph.getVerticesByProperty('age', 29))).toEqual(['z']);
  });
});
