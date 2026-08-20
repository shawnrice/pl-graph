import { describe, expect, test } from 'bun:test';

import { openBackend } from './engine.js';
import { emptyGraph, loadGraph, saveGraph } from './io.js';
import { classify, runQuery } from './query.js';

describe('classify', () => {
  test('a leading `g.` is Gremlin; everything else is GQL', () => {
    expect(classify('g.V().count()')).toBe('gremlin');
    expect(classify('  g . V()')).toBe('gremlin');
    expect(classify('MATCH (n) RETURN n')).toBe('gql');
    expect(classify('RETURN 1')).toBe('gql');
  });
});

// The CLI runs on the native/wasm engine — these load the real artifact.
const backend = await openBackend();

const withPeople = () => {
  const g = emptyGraph(backend);
  g.query("INSERT (:Person {name: 'marko', age: 29})");
  g.query("INSERT (:Person {name: 'josh', age: 32})");

  return g;
};

describe('runQuery (wasm engine)', () => {
  test('GQL → a table of rows', () => {
    const g = withPeople();
    const { lang, output } = runQuery(g, 'MATCH (p:Person) RETURN p.name, p.age', undefined, false);

    expect(lang).toBe('gql');
    expect(output).toContain('p.name');
    expect(output).toContain('marko');
    expect(output).toContain('josh');
    g.free();
  });

  test('Gremlin → a scalar wrapped in a value column', () => {
    const g = withPeople();
    const { lang, output } = runQuery(g, 'g.V().count()', undefined, false);

    expect(lang).toBe('gremlin');
    expect(output).toContain('value');
    expect(output).toContain('2');
    g.free();
  });

  test('a forced language overrides the auto-detect', () => {
    const g = withPeople();

    // `g.V().count()` auto-classifies as Gremlin (leading `g.`); forcing GQL routes
    // it to the GQL engine instead, which rejects the Gremlin syntax — proving the
    // explicit `lang` argument wins over `classify`.
    expect(classify('g.V().count()')).toBe('gremlin');
    expect(() => runQuery(g, 'g.V().count()', 'gql', false)).toThrow();
    g.free();
  });
});

describe('load / save round-trip', () => {
  test('ndjson', () => {
    const g = withPeople();
    const g2 = loadGraph(backend, saveGraph(g, 'ndjson'), 'ndjson');

    expect(g2.vertexCount).toBe(2);
    g.free();
    g2.free();
  });

  test('graphson', () => {
    const g = withPeople();
    const g2 = loadGraph(backend, saveGraph(g, 'graphson'), 'graphson');

    expect(g2.vertexCount).toBe(2);
    g.free();
    g2.free();
  });
});
