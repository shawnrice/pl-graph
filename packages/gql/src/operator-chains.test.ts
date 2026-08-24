import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';
import { deserialize } from '@lenke/serialization';

import { prepare, query } from './index.js';

// Behavioral golden tests for operator-CHAIN semantics, pinned BEFORE the n-ary
// AST flatten refactor so a regression in precedence,
// associativity, three-valued boolean folding, concat null propagation, error
// propagation (the evaluator does NOT short-circuit), the long-chain value, or
// the planner's AND-split index seed is caught. Mirror of
// crates/lenke-engine/src/gql/tests.rs (byte-identity).

const val = (expr: string): unknown => {
  const rows = query(new Graph(), `RETURN ${expr} AS r`) as Array<{ r: unknown }>;

  return rows[0]?.r;
};

const errs = (expr: string): boolean => {
  try {
    query(new Graph(), `RETURN ${expr} AS r`);

    return false;
  } catch {
    return true;
  }
};

describe('operator-chain semantics (n-ary refactor regression guard)', () => {
  test('three-valued AND/OR/XOR folding (null == UNKNOWN)', () => {
    const cases: Array<[string, unknown]> = [
      ['true AND true', true],
      ['true AND false', false],
      ['true AND null', null],
      ['false AND null', false], // false dominates
      ['null AND null', null],
      ['true OR false', true],
      ['false OR false', false],
      ['true OR null', true], // true dominates
      ['false OR null', null],
      ['null OR null', null],
      ['true XOR false', true],
      ['true XOR true', false],
      ['true XOR null', null],
      ['null XOR null', null],
    ];

    for (const [e, want] of cases) {
      expect(val(e)).toBe(want as never);
    }
  });

  test('boolean chains fold correctly', () => {
    expect(val('true AND true AND false')).toBe(false);
    expect(val('true OR false OR null')).toBe(true);
    expect(val('true XOR false XOR true')).toBe(false);
    expect(val('false AND false AND false')).toBe(false);
    expect(val('null OR null OR true')).toBe(true);
  });

  test('boolean precedence & associativity', () => {
    expect(val('true OR false AND false')).toBe(true); // AND binds tighter
    expect(val('NOT true AND false')).toBe(false); // (NOT true) AND false
    expect(val('NOT (true AND false)')).toBe(true);
    expect(val('true XOR true XOR true')).toBe(true); // left-assoc
  });

  test('arithmetic left-associativity (non-regroupable ops)', () => {
    expect(val('10 - 3 - 2')).toBe(5);
    expect(val('100 / 5 / 2')).toBe(10);
    expect(val('20 % 7 % 3')).toBe(0);
    expect(val('10 - 2 + 3')).toBe(11);
    expect(val('2 * 3 % 4')).toBe(2);
    expect(val('2 + 3 * 4')).toBe(14); // * binds tighter
    expect(val('(2 + 3) * 4')).toBe(20);
    expect(val('100 / 10 * 5')).toBe(50);
    expect(val('7 - 3 - 2 - 1')).toBe(1);
  });

  test('string concat chains + null propagation', () => {
    expect(val("'a' || 'b' || 'c'")).toBe('abc');
    expect(val("'x' || null")).toBe(null);
    expect(val("null || 'y'")).toBe(null);
  });

  test('the evaluator does NOT short-circuit: a fault in any operand propagates', () => {
    // `false AND …` / `true OR …` still evaluate the other operand, so a
    // division-by-zero in it raises rather than being skipped. The n-ary fold
    // must preserve this (evaluate every element).
    expect(errs('false AND (1.0 / 0.0)')).toBe(true);
    expect(errs('true OR (1.0 / 0.0)')).toBe(true);
  });

  test('long chains evaluate to the right value (not just "not a crash")', () => {
    expect(val(Array(100).fill('1').join(' + '))).toBe(100);
    expect(val(Array(50).fill('true').join(' AND '))).toBe(true);
    expect(val(Array(200).fill('false').join(' OR '))).toBe(false);
  });

  // WHERE chains over a scan (vectorized eval path) + planner AND-split seed.
  const nGraph = (index: boolean): Graph => {
    const nd = Array.from({ length: 12 }, (_, i) =>
      JSON.stringify({
        type: 'node',
        id: `n${i}`,
        labels: ['N'],
        properties: { id: `n${i}`, a: i, b: i % 3 },
      }),
    ).join('\n');
    const g = deserialize(nd, 'ndjson', new Graph());

    if (index) {
      g.createIndex({ on: 'vertex', kind: 'hash', keys: ['id'] });
    }

    return g;
  };
  const colA = (g: Graph, q: string, params?: Record<string, unknown>): number[] =>
    (query(g, q, params) as Array<{ a: number }>).map((r) => r.a);

  test('WHERE chains over a scan return the right rows', () => {
    const g = nGraph(false);
    expect(
      colA(g, 'MATCH (n:N) WHERE n.a > 1 AND n.a < 8 AND n.b <> 0 RETURN n.a AS a ORDER BY a'),
    ).toEqual([2, 4, 5, 7]);
    expect(
      colA(g, 'MATCH (n:N) WHERE n.a = 0 OR n.a = 5 OR n.a = 11 RETURN n.a AS a ORDER BY a'),
    ).toEqual([0, 5, 11]);
    expect(
      colA(g, 'MATCH (n:N) WHERE n.a < 3 OR n.a > 9 AND n.b = 0 RETURN n.a AS a ORDER BY a'),
    ).toEqual([0, 1, 2]);
    expect(
      colA(g, 'MATCH (n:N) WHERE NOT (n.a = 1 OR n.a = 2) AND n.a < 5 RETURN n.a AS a ORDER BY a'),
    ).toEqual([0, 3, 4]);
  });

  test('AND-split index seed returns the same rows with and without an index', () => {
    for (const indexed of [false, true]) {
      const g = nGraph(indexed);
      expect(
        colA(g, 'MATCH (n:N) WHERE n.id = $x AND n.b = $y RETURN n.a AS a', { x: 'n6', y: 0 }),
      ).toEqual([6]);
    }
  });

  test('a prepared statement honours the operator-chain ceiling too (default 10k, configurable)', () => {
    const over = `RETURN ${Array(10_002).fill('true').join(' AND ')} AS r`; // 10_001 ops

    expect(() => prepare(over)).toThrow(); // default 10k rejects
    const plan = prepare<{ r: boolean }>(
      `RETURN ${Array(50_000).fill('true').join(' AND ')} AS r`,
      {
        maxOperatorChain: 200_000,
      },
    );
    expect(plan(new Graph())).toEqual([{ r: true }]);
  });
});
