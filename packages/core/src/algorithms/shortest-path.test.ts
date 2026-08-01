import { describe, expect, test } from 'bun:test';

import { Graph } from '../core/Graph.js';
import { shortestPath } from './shortest-path.js';

// 1→2 (w1), 2→3 (w2), 1→3 (w5); node 4 isolated (unreachable from 1). Mirrors the
// Rust `weighted_chain` test.
const weightedChain = (): Graph => {
  const g = new Graph();

  for (const id of ['1', '2', '3', '4']) {
    g.addVertex({ id, labels: ['N'] });
  }

  for (const [from, to, w] of [
    ['1', '2', 1],
    ['2', '3', 2],
    ['1', '3', 5],
  ] as const) {
    g.addEdge({
      from: g.getVertexById(from)!,
      to: g.getVertexById(to)!,
      labels: ['E'],
      properties: { w },
    });
  }

  return g;
};

const map = (rows: { node: string; distance: number }[]): Record<string, number> =>
  Object.fromEntries(rows.map((r) => [r.node, r.distance]));

describe('shortest path', () => {
  test('unweighted BFS — direct 1→3 edge is one hop', async () => {
    expect(map(await shortestPath({ source: '1' }, weightedChain()))).toEqual({ 1: 0, 2: 1, 3: 1 });
  });

  test('weighted Dijkstra — 1→2→3 (3) beats direct 1→3 (5)', async () => {
    expect(map(await shortestPath({ source: '1', weightProperty: 'w' }, weightedChain()))).toEqual({
      1: 0,
      2: 1,
      3: 3,
    });
  });

  test('reachable set excludes upstream/disconnected vertices', async () => {
    // From 2: only 2 and 3; node 1 is upstream and node 4 disconnected.
    expect(map(await shortestPath({ source: '2', weightProperty: 'w' }, weightedChain()))).toEqual({
      2: 0,
      3: 2,
    });
  });

  test('unknown source → no rows', async () => {
    expect(await shortestPath({ source: '99' }, weightedChain())).toEqual([]);
    expect(await shortestPath({}, weightedChain())).toEqual([]);
  });

  test('unknown edge type → only the source at distance 0', async () => {
    expect(map(await shortestPath({ source: '1', edgeLabel: 'NOPE' }, weightedChain()))).toEqual({
      1: 0,
    });
  });

  test('writeProperty writes each distance back to the vertex', async () => {
    const g = weightedChain();
    await shortestPath({ source: '1', weightProperty: 'w', writeProperty: 'dist' }, g);
    expect(g.getVertexById('3')?.getProperty<number>('dist')).toBe(3);
    expect(g.getVertexById('1')?.getProperty<number>('dist')).toBe(0);
    // Unreachable node 4 gets no distance written.
    expect(g.getVertexById('4')?.getProperty('dist')).toBeUndefined();
  });

  test('dual-form: curried application equals direct', async () => {
    expect(await shortestPath({ source: '1' })(weightedChain())).toEqual(
      await shortestPath({ source: '1' }, weightedChain()),
    );
  });
});

// ---------------------------------------------------------------------------
// Negative weights are REJECTED, not run.
//
// Dijkstra's precondition was documented and never enforced, so a graph that
// violated it did not fail — it spun forever. A negative self-loop (ONE node,
// ONE edge) was enough to hang the engine. Found by the randomized algorithm
// differential, whose weight corpus includes negatives; the native engine hung
// on the identical input and is guarded the same way.
// ---------------------------------------------------------------------------

describe('shortestPath rejects negative weights', () => {
  const withEdges = (edges: [string, string, number][]): Graph => {
    const g = new Graph();

    for (const id of ['a', 'b', 'c']) {
      g.addVertex({ id, labels: ['N'] });
    }

    for (const [from, to, w] of edges) {
      g.addEdge({
        from: g.getVertexById(from)!,
        to: g.getVertexById(to)!,
        labels: ['E'],
        properties: { w },
      });
    }

    return g;
  };
  const code = async (g: Graph, weightProperty?: string): Promise<string> => {
    try {
      await shortestPath(
        { source: 'a', ...(weightProperty === undefined ? {} : { weightProperty }) },
        g,
      );

      return 'ok';
    } catch (e) {
      return (e as { code?: string }).code ?? 'UNCODED';
    }
  };

  test('a negative self-loop is rejected, not run forever', async () => {
    // The minimal hang: one node, one edge.
    expect(await code(withEdges([['a', 'a', -1]]), 'w')).toBe('E_INVALID_VALUE');
  });

  test('a negative cycle is rejected', async () => {
    expect(
      await code(
        withEdges([
          ['a', 'b', -1],
          ['b', 'a', 0],
        ]),
        'w',
      ),
    ).toBe('E_INVALID_VALUE');
  });

  test('a single negative edge is rejected even without a cycle', async () => {
    // This terminated before the fix and returned -1. Dijkstra can settle a
    // vertex before a cheaper negative path reaches it, so an acyclic negative
    // graph terminating is luck rather than a correct answer.
    expect(await code(withEdges([['a', 'b', -1]]), 'w')).toBe('E_INVALID_VALUE');
  });

  test('non-negative weights still run, zero included', async () => {
    expect(
      await code(
        withEdges([
          ['a', 'b', 1],
          ['b', 'c', 2.5],
        ]),
        'w',
      ),
    ).toBe('ok');
    expect(await code(withEdges([['a', 'b', 0]]), 'w')).toBe('ok');
  });

  test('an unweighted run ignores negative weights entirely', async () => {
    // BFS has no such precondition, so the guard must not fire without
    // `weightProperty`.
    expect(await code(withEdges([['a', 'b', -5]]))).toBe('ok');
  });
});
