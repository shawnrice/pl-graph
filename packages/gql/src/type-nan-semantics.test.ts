// The JS ground truth for three type/NaN questions the Rust engine matches
// (see crates/lenke-engine/src/gql/tests.rs). The Rust engine runs the same
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
    const v = (e: string): unknown => query(fixture(), `MATCH (n:T) RETURN ${e} AS x LIMIT 1`)[0].x;
    expect(v(`null - 'abc'`)).toBe(null);
    expect(v(`null + true`)).toBe(null);
    expect(v(`null - [1, 2]`)).toBe(null);
    expect(v(`'abc' * null`)).toBe(null);
    // Null even wins over a division-by-zero (also matching the engine).
    expect(v(`null / 0`)).toBe(null);
  });

  test('a named numeric function validates non-null args even beside a null', () => {
    // Unlike an arithmetic OPERATOR (null-first), a named function faults on a non-numeric
    // argument even when another argument is null — `atan2(null, duration)` is a type error,
    // matching the engine's gate. A null beside a VALID-typed arg still propagates to null.
    const v = (e: string): unknown => query(fixture(), `MATCH (n:T) RETURN ${e} AS x LIMIT 1`)[0].x;
    expect(() => v(`atan2(null, duration('P1Y'))`)).toThrow(/number/i);
    expect(() => v(`power(null, 'abc')`)).toThrow(/number/i);
    expect(() => v(`round(null, 'abc')`)).toThrow(/number/i);
    // A null beside a valid number still short-circuits to null.
    expect(v(`atan2(null, 5)`)).toBe(null);
    expect(v(`round(5, null)`)).toBe(5);
  });

  test('stddev faults on a non-numeric value even when too few rows to compute', () => {
    // stddev_samp needs >= 2 rows, but a non-numeric VALUE is a type error regardless —
    // the type-check runs before the row-count short-circuit. `DISTINCT true` dedups to a
    // single boolean; it faults (not null), matching the engine and sum/avg/percentile.
    const q = (e: string): unknown[] => query(fixture(), `MATCH (n:T) RETURN ${e} AS x`);
    expect(() => q(`stddev_samp(DISTINCT true)`)).toThrow(/number/i);
    expect(() => q(`stddev_pop(DISTINCT (n.n IS NULL))`)).toThrow(/number/i);
    // A single numeric value is still null (too few rows), not a fault.
    expect(q(`stddev_samp(DISTINCT 5)`)).toEqual([{ x: null }]);
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
