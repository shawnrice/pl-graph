import { describe, expect, test } from 'bun:test';

import { Graph, Path } from '@lenke/core';

import { parseQuery, query } from './index.js';

// The TinkerPop "modern" graph (same topology as the native GQL tests).
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

const names = (rows: readonly Record<string, unknown>[]): string[] =>
  rows.map((r) => r.n as string).sort();

describe('ANY SHORTEST path patterns', () => {
  test('reachable set from marko over ->* (min 0 includes marko itself)', () => {
    const rows = query(
      modern(),
      "MATCH ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' RETURN b.name AS n",
    );
    expect(names(rows)).toEqual(['josh', 'lop', 'marko', 'ripple', 'vadas']);
  });

  test('binds a genuinely shortest path (1-hop marko->lop, not marko->josh->lop)', () => {
    const rows = query(
      modern(),
      "MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop' RETURN p",
    );
    expect(rows).toHaveLength(1);
    const p = rows[0].p as Path;
    expect(p).toBeInstanceOf(Path);
    expect(p.hops).toBe(1);
    expect(p.vertices.map((x) => x.id)).toEqual(['marko', 'lop']);
    expect(p.edges).toHaveLength(1);
  });

  test('-> + excludes the zero-length self path', () => {
    const rows = query(
      modern(),
      "MATCH ANY SHORTEST (a)-[]->+(b) WHERE a.name = 'marko' RETURN b.name AS n",
    );
    expect(names(rows)).toEqual(['josh', 'lop', 'ripple', 'vadas']);
  });

  test('-> + closes on the seed via the shortest cycle', () => {
    const g = new Graph();
    const a = g.addVertex({ id: 'a', labels: ['N'], properties: { id: 'a' } });
    const b = g.addVertex({ id: 'b', labels: ['N'], properties: { id: 'b' } });
    g.addEdge({ from: a, to: b, labels: ['R'], properties: {} });
    g.addEdge({ from: b, to: a, labels: ['R'], properties: {} });

    // ->+(a): the shortest cycle a→b→a (hops 2), closing on the seed variable.
    const plus = query(g, "MATCH p = ANY SHORTEST (a:N {id:'a'})-[:R]->+(a) RETURN p");
    expect(plus).toHaveLength(1);
    const p = plus[0].p as Path;
    expect(p.hops).toBe(2);
    expect(p.vertices.map((x) => x.id)).toEqual(['a', 'b', 'a']);

    // ->*(a): min 0 admits the zero-length self path (hops 0), unchanged.
    const star = query(g, "MATCH p = ANY SHORTEST (a:N {id:'a'})-[:R]->*(a) RETURN p");
    expect((star[0].p as Path).hops).toBe(0);
  });

  test('->{1,1} keeps only the direct out-neighbours', () => {
    const rows = query(
      modern(),
      "MATCH ANY SHORTEST (a)-[]->{1,1}(b) WHERE a.name = 'marko' RETURN b.name AS n",
    );
    expect(names(rows)).toEqual(['josh', 'lop', 'vadas']);
  });

  test('ISO path functions: path_length/cardinality, nodes, edges, elements', () => {
    const rows = query(
      modern(),
      "MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'ripple' " +
        'RETURN path_length(p) AS len, cardinality(p) AS card, ' +
        'nodes(p) AS ns, edges(p) AS es, elements(p) AS el',
    );
    expect(rows).toHaveLength(1);
    const [row] = rows;

    expect(row.len).toBe(2); // marko -> josh -> ripple: 2 hops (edges)
    // ISO cardinality = every element: 3 nodes + 2 edges = 5. (`length` is not an ISO
    // GQL function; PATH_LENGTH gives hops, CARDINALITY gives the element count.)
    expect(row.card).toBe(5);

    const ns = row.ns as Array<{ id: string }>;
    expect(ns.map((v) => v.id)).toEqual(['marko', 'josh', 'ripple']);

    expect((row.es as unknown[]).length).toBe(2);
    expect((row.el as unknown[]).length).toBe(5); // interleaved node,edge,node,edge,node
  });

  test('unsupported selector shapes are rejected at parse time', () => {
    expect(() => parseQuery('MATCH (a)-[]->*(b) RETURN b')).not.toThrow();
    expect(() => parseQuery('MATCH ALL SHORTEST (a)-[]->*(b) RETURN b')).not.toThrow(); // now supported
    expect(() => parseQuery('MATCH ANY (a)-[]->*(b) RETURN b')).not.toThrow(); // bare ANY now supported
    expect(() => parseQuery('MATCH SHORTEST 2 (a)-[]->*(b) RETURN b')).not.toThrow(); // SHORTEST k now supported
    expect(() => parseQuery('MATCH SHORTEST 2 GROUP (a)-[]->*(b) RETURN b')).not.toThrow();
    expect(() => parseQuery('MATCH SHORTEST (a)-[]->*(b) RETURN b')).toThrow(); // …but needs a count
    expect(() => parseQuery('MATCH SHORTEST 0 (a)-[]->*(b) RETURN b')).toThrow(); // k >= 1
    expect(() => parseQuery('MATCH ANY SHORTEST (a)-[]->(b) RETURN b')).toThrow();
    expect(() => parseQuery('MATCH ANY SHORTEST (a)-[]->{2,4}(b) RETURN b')).toThrow();
    expect(() => parseQuery('MATCH p = (a)-[]->(b) RETURN p')).toThrow();
  });
});
