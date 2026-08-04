import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';

import { query } from './index.js';

/**
 * An edge carries a SET of types and the label indexes bucket it under every
 * one, so anything that sums or concatenates buckets sees a two-type edge once
 * per bucket. Native walks one adjacency list and asks "does this edge carry
 * any of these types", so a duplicate here is a cross-engine divergence — the
 * same shape as the `[:R|S]` double-count already fixed on the native side.
 *
 * Every edge below is BOTH `R` and `S`, which makes the reference trivial: every
 * spelling that selects an edge at all must select exactly the same edges.
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
    g.addEdge({ id, from: g.getVertexById(from)!, to: g.getVertexById(to)!, labels: ['R', 'S'] });
  }

  return g;
};

const rows = (src: string): unknown[] => query(twoLabelChain(), src);

describe('a multi-type edge is matched once per pattern', () => {
  test('a type disjunction counts each edge once', () => {
    const one = rows('MATCH (a)-[:R]->(b) RETURN count(*) AS n');

    expect(one).toEqual([{ n: 2 }]);
    expect(rows('MATCH (a)-[:S]->(b) RETURN count(*) AS n')).toEqual(one);
    expect(rows('MATCH (a)-[]->(b) RETURN count(*) AS n')).toEqual(one);
    expect(rows('MATCH (a)-[:R|S]->(b) RETURN count(*) AS n')).toEqual(one);
  });

  test('the labelled-endpoint count shortcut agrees', () => {
    const one = rows('MATCH (a:V)-[:R]->(b:V) RETURN count(*) AS n');

    expect(one).toEqual([{ n: 2 }]);
    expect(rows('MATCH (a:V)-[:R|S]->(b:V) RETURN count(*) AS n')).toEqual(one);
  });

  test('the two-hop degree product agrees', () => {
    const one = rows('MATCH (a)-[:R]->(b)-[:R]->(c) RETURN count(*) AS n');

    expect(one).toEqual([{ n: 1 }]);
    expect(rows('MATCH (a)-[:R|S]->(b)-[:R|S]->(c) RETURN count(*) AS n')).toEqual(one);
  });

  test('enumeration agrees with the counts', () => {
    const one = rows('MATCH (a)-[:R]->(b) RETURN a.id AS x ORDER BY x');

    expect(rows('MATCH (a)-[:R|S]->(b) RETURN a.id AS x ORDER BY x')).toEqual(one);
  });

  test('a reachability walk visits each edge once', () => {
    const one = rows('MATCH (a)-[:R]->{1,2}(b) RETURN count(*) AS n');

    expect(rows('MATCH (a)-[:R|S]->{1,2}(b) RETURN count(*) AS n')).toEqual(one);
  });
});

/**
 * The O(1) `count(*)` shortcut sums per-type bucket sizes, which is only sound
 * while no edge sits in two buckets — `graph.multiTypeEdgeCount === 0` arms it.
 * A counter that drifts silently returns a wrong number with no error, so these
 * drive an edge across the 1↔2 type boundary in every way the API allows and
 * check the answer after each step.
 */
describe('the count shortcut tracks edges crossing the one-type boundary', () => {
  const chain = (): Graph => {
    const g = new Graph();

    for (const id of ['a', 'b']) {
      g.addVertex({ id, labels: ['V'], properties: {} });
    }

    g.addEdge({ id: 'e0', from: g.getVertexById('a')!, to: g.getVertexById('b')!, labels: ['R'] });

    return g;
  };

  const n = (g: Graph, q: string): number => (query(g, q)[0] as { n: number } | undefined)?.n ?? 0;
  const both = (g: Graph): number => n(g, 'MATCH ()-[:R|S]->() RETURN count(*) AS n');

  test('single → multi → single', () => {
    const g = chain();
    const e = g.getEdgeById('e0')!;

    expect(both(g)).toBe(1);
    expect(g.multiTypeEdgeCount).toBe(0);

    g.addLabelToEdge('S', e);
    expect(g.multiTypeEdgeCount).toBe(1);
    expect(both(g)).toBe(1); // one edge, now in both buckets

    g.removeLabelFromEdge('S', e);
    expect(g.multiTypeEdgeCount).toBe(0);
    expect(both(g)).toBe(1);
  });

  test('re-adding a type it already has changes nothing', () => {
    const g = chain();
    const e = g.getEdgeById('e0')!;

    g.addLabelToEdge('S', e);
    g.addLabelToEdge('S', e);

    expect(g.multiTypeEdgeCount).toBe(1);
    expect(both(g)).toBe(1);
  });

  test('removing a type it does not have changes nothing', () => {
    const g = chain();

    g.removeLabelFromEdge('S', g.getEdgeById('e0')!);

    expect(g.multiTypeEdgeCount).toBe(0);
    expect(both(g)).toBe(1);
  });

  test('a third type and back', () => {
    const g = chain();
    const e = g.getEdgeById('e0')!;

    g.addLabelToEdge('S', e);
    g.addLabelToEdge('T', e);
    expect(g.multiTypeEdgeCount).toBe(1); // still ONE multi-type edge, not two
    expect(both(g)).toBe(1);

    g.removeLabelFromEdge('T', e);
    expect(g.multiTypeEdgeCount).toBe(1);
    g.removeLabelFromEdge('S', e);
    expect(g.multiTypeEdgeCount).toBe(0);
    expect(both(g)).toBe(1);
  });

  test('removing a multi-type edge re-arms the shortcut', () => {
    const g = chain();
    const e = g.getEdgeById('e0')!;

    g.addLabelToEdge('S', e);
    g.removeEdge(e);

    expect(g.multiTypeEdgeCount).toBe(0);
    expect(both(g)).toBe(0);
  });

  test('inserting a multi-type edge disarms it', () => {
    const g = chain();

    g.addEdge({
      id: 'e1',
      from: g.getVertexById('b')!,
      to: g.getVertexById('a')!,
      labels: ['R', 'S'],
    });

    expect(g.multiTypeEdgeCount).toBe(1);
    expect(both(g)).toBe(2);
  });

  test('a rolled-back type change leaves the counter where it started', () => {
    const g = chain();

    expect(() =>
      g.transaction(() => {
        g.addLabelToEdge('S', g.getEdgeById('e0')!);
        expect(g.multiTypeEdgeCount).toBe(1);

        throw new Error('rollback');
      }),
    ).toThrow('rollback');

    expect(g.multiTypeEdgeCount).toBe(0);
    expect(both(g)).toBe(1);
  });

  test('truncate resets the counter', () => {
    const g = chain();

    g.addLabelToEdge('S', g.getEdgeById('e0')!);
    g.truncate();

    expect(g.multiTypeEdgeCount).toBe(0);
    expect(both(g)).toBe(0);
  });
});

/**
 * `labels()` takes an ELEMENT — a node or an edge — and returns its label set.
 *
 * ISO GQL does not define `labels()` at all: the standard interrogates labels
 * with the `IS LABELED` predicate, and its only element function is
 * `element_id`. `labels` is a Cypher inheritance vendors added, and the two that
 * ship it define it over an element rather than just a node — Spanner's
 * `LABELS(GRAPH_ELEMENT) -> ARRAY<STRING>` and Fabric's `labels(node_or_edge)`
 * "the labels of a node or edge as a list of strings". Both return a length-1
 * list for an edge because neither has multi-label edges; this engine does, so
 * it returns the whole set.
 */
describe('labels() of an edge is its type set', () => {
  const mixed = (): Graph => {
    const g = new Graph();

    g.addVertex({ id: 'a', labels: ['W', 'V'], properties: {} });
    g.addVertex({ id: 'b', labels: ['V'], properties: {} });
    g.addEdge({
      id: 'e0',
      from: g.getVertexById('a')!,
      to: g.getVertexById('b')!,
      labels: ['S', 'R'],
    });
    g.addEdge({ id: 'e1', from: g.getVertexById('b')!, to: g.getVertexById('a')!, labels: ['T'] });

    return g;
  };
  const ask = (src: string): unknown[] => query(mixed(), src);

  test('an edge reports every type, sorted', () => {
    expect(ask('MATCH ()-[e]->() RETURN labels(e) AS x ORDER BY x')).toEqual([
      { x: ['R', 'S'] },
      { x: ['T'] },
    ]);
  });

  test('a single-type edge still gets a one-element list', () => {
    expect(ask('MATCH ()-[e:T]->() RETURN labels(e) AS x')).toEqual([{ x: ['T'] }]);
  });

  test('the node arm is unchanged', () => {
    expect(ask('MATCH (n) RETURN labels(n) AS x ORDER BY x')).toEqual([
      { x: ['V'] },
      { x: ['V', 'W'] },
    ]);
  });

  test('type() stays singular', () => {
    // openCypher's `type(relationship) -> String` cannot express a set, so it
    // reports the first type. `labels(e)` is the accessor for all of them.
    expect(ask('MATCH ()-[e:R]->() RETURN type(e) AS x')).toEqual([{ x: 'S' }]);
  });

  test('neither function accepts a non-element', () => {
    expect(ask('MATCH (n) RETURN labels(n.missing) AS x LIMIT 1')).toEqual([{ x: null }]);
    expect(ask('MATCH (n) RETURN type(n) AS x LIMIT 1')).toEqual([{ x: null }]);
  });

  test('labels() agrees with the pattern matcher', () => {
    for (const t of ['R', 'S']) {
      expect(ask(`MATCH ()-[e:${t}]->() RETURN labels(e) AS x`)).toEqual([{ x: ['R', 'S'] }]);
    }

    expect(ask('MATCH ()-[e:S]->() RETURN size(labels(e)) AS x')).toEqual([{ x: 2 }]);
  });
});
