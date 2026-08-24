import { describe, expect, test } from 'bun:test';

import { openBackend } from './engine.js';
import { emptyGraph } from './io.js';
import { isBalanced, makeState, runMeta, runStatement, type State } from './shell.js';

const backend = await openBackend();

const stateWith = (setup?: (g: ReturnType<typeof emptyGraph>) => void): State => {
  const g = emptyGraph(backend);

  setup?.(g);

  return makeState({ graph: g, backend, color: false });
};

describe('isBalanced', () => {
  test('balanced brackets and quotes are complete', () => {
    expect(isBalanced('MATCH (p:Person) RETURN p.name')).toBe(true);
    expect(isBalanced('{a: [1, 2]}')).toBe(true);
    expect(isBalanced("RETURN 'a) b'")).toBe(true); // paren inside a string
  });

  test('an open bracket or quote is incomplete', () => {
    expect(isBalanced('MATCH (p:Person')).toBe(false);
    expect(isBalanced("RETURN 'unterminated")).toBe(false);
  });
});

describe('runStatement', () => {
  test('GQL mode runs the query', () => {
    const state = stateWith((g) => g.query("INSERT (:Person {name: 'marko'})"));

    expect(runStatement(state, 'MATCH (p:Person) RETURN p.name AS name')).toEqual([
      { name: 'marko' },
    ]);
  });

  test('Gremlin mode runs a traversal', () => {
    const state = stateWith((g) => g.query("INSERT (:Person {name: 'x'})"));

    state.mode = 'gremlin';
    expect(runStatement(state, "g.V().hasLabel('Person').count()")).toEqual([1]);
  });

  test('JS mode evaluates over the last result as `_`', () => {
    const state = stateWith();

    state.mode = 'js';
    state.last = [{ n: 1 }, { n: 2 }, { n: 3 }];
    expect(runStatement(state, '_.filter((r) => r.n > 1).map((r) => r.n)')).toEqual([2, 3]);
  });
});

describe('runMeta', () => {
  const capture = () => {
    const lines: string[] = [];

    return { out: (s: string) => lines.push(s), lines };
  };

  test('\\gremlin / \\gql switch the mode', () => {
    const state = stateWith();
    const { out } = capture();

    expect(runMeta(state, '\\gremlin', out)).toBe(false);
    expect(state.mode).toBe('gremlin');
    runMeta(state, '\\gql', out);
    expect(state.mode).toBe('gql');
  });

  test('\\q signals quit', () => {
    expect(runMeta(stateWith(), '\\q', () => {})).toBe(true);
  });

  test('\\format sets the output format; a bad value is rejected', () => {
    const state = stateWith();
    const { out, lines } = capture();

    runMeta(state, '\\format json', out);
    expect(state.format).toBe('json');
    runMeta(state, '\\format nonsense', out);
    expect(state.format).toBe('json'); // unchanged
    expect(lines.at(-1)).toContain('usage');
  });

  test('\\c loads a bundled sample', () => {
    const state = stateWith();

    runMeta(state, '\\c employment', () => {});
    expect(state.graph.vertexCount).toBe(5);
    expect(state.labels).toContain('Person');
  });

  test('\\clock sets a clock; a bad date is reported, not thrown', () => {
    const state = stateWith();
    const { out, lines } = capture();

    expect(() => runMeta(state, '\\clock 2020-06-01', out)).not.toThrow();
    runMeta(state, '\\clock not-a-date', out);
    expect(lines.at(-1)).toContain('not a date');
  });
});
