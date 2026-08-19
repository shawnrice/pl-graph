// The JS ground truth for three type/NaN questions the Rust engine matches
// (see crates/lenke-core gql/tests.rs bug1_*). The Rust core runs the same
// logical query down two drivers (vectorized single-pattern vs scalar
// multi-pattern) and both must agree with THIS engine — the JS library is the
// reference. No coercion of a bool into the numeric domain; arithmetic on a
// bool is a type error; a NaN never displaces a running MIN/MAX extreme.
import { describe, expect, test } from 'bun:test';

import { Graph } from '@lenke/core';

import { query } from './index.js';

const fixture = (): Graph => {
  const g = new Graph();
  g.disableEvents();
  // score −1 is scanned FIRST so sqrt(score) = NaN seeds the extreme.
  g.addVertex({ id: 'a', labels: ['T'], properties: { score: -1, age: 1 } });
  g.addVertex({ id: 'b', labels: ['T'], properties: { score: 4, age: 2 } });
  g.addVertex({ id: 'c', labels: ['T'], properties: { score: 9, age: 3 } });
  g.enableEvents();

  return g;
};

describe('GQL type/NaN semantics (the reference the Rust engine matches)', () => {
  test('a number never equals a boolean → no rows (no true→1 coercion)', () => {
    expect(query(fixture(), `MATCH (n:T) WHERE n.age = true RETURN n.age AS a`)).toEqual([]);
  });

  test('arithmetic on a boolean is a type error', () => {
    expect(() => query(fixture(), `MATCH (n:T) WHERE n.age = 1 RETURN true + 1 AS x`)).toThrow(
      /number/i,
    );
  });

  test('a null operand short-circuits arithmetic to null before the type-check', () => {
    // Null propagates BEFORE the numeric type-check (matching the engine): the result is
    // null regardless of the other operand's type. The type error only fires when NEITHER
    // side is null. So `true + 1` throws (above), but `null + true` / `null - 'abc'` are
    // null, NOT a type error.
    const v = (e: string): unknown =>
      query(fixture(), `MATCH (n:T) RETURN ${e} AS x LIMIT 1`)[0].x;
    expect(v(`null - 'abc'`)).toBe(null);
    expect(v(`null + true`)).toBe(null);
    expect(v(`null - [1, 2]`)).toBe(null);
    expect(v(`'abc' * null`)).toBe(null);
    // Null even wins over a division-by-zero (also matching the engine).
    expect(v(`null / 0`)).toBe(null);
  });

  test('min/max use the total order: max keeps NaN (largest), min skips it', () => {
    // sqrt(score) over [-1, 4, 9] → [NaN, 2, 3]. Total order (NaN last): max is
    // NaN, min is 2 — deterministic regardless of scan order.
    const max = query(fixture(), `MATCH (n:T) RETURN max(sqrt(n.score)) AS m`) as Array<{
      m: number;
    }>;
    const min = query(fixture(), `MATCH (n:T) RETURN min(sqrt(n.score)) AS m`) as Array<{
      m: number;
    }>;
    expect(Number.isNaN(max[0].m)).toBe(true);
    expect(min[0].m).toBe(2);
  });
});
