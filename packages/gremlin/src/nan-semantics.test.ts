// NaN value-semantics: the one Tier-3 drift that lives below the Groovy-text
// boundary (a NaN literal can't be lexed as Groovy nor carried by JSON, so the
// TS↔native conformance runner can't reach it — see gremlin-conformance.test.ts
// in @lenke/native). These pin the JS-idiomatic behavior, which here coincides
// with TinkerPop: in JS every comparison with NaN is `false`, so NaN satisfies
// NO ordering predicate and is filtered out — matching Rust's `partial_cmp →
// None → filtered`.
import { describe, expect, test } from 'bun:test';

import type { Plan, Predicate } from './ast.js';
import { run } from './executor.js';
import { createTestTinkerGraph } from './fixtures/createTestTinkerGraph.js';
import { between, gt, gte, inside, lt, lte, outside } from './predicates.js';
import { dedupe, inject, is, max, min, order } from './steps.js';
import { traversal } from './traversal.js';

// inject() is a literal source and never reads the graph, but `run` wants one.
const g = createTestTinkerGraph();
const arr = (plan: Plan): unknown[] => [...run(plan, g)];
const over = (pred: Predicate): unknown[] => arr(traversal(inject(1, Number.NaN, 3), is(pred)));

describe('NaN is filtered by every ordering predicate (JS: NaN compares false)', () => {
  test('gte keeps only real matches, drops NaN', () => {
    expect(over(gte(2))).toEqual([3]);
  });

  test('lte keeps only real matches, drops NaN', () => {
    expect(over(lte(2))).toEqual([1]);
  });

  test('gt drops NaN', () => {
    expect(over(gt(2))).toEqual([3]);
  });

  test('lt drops NaN', () => {
    expect(over(lt(2))).toEqual([1]);
  });

  test('between drops NaN', () => {
    expect(over(between(0, 5))).toEqual([1, 3]);
  });

  test('inside drops NaN', () => {
    expect(over(inside(0, 5))).toEqual([1, 3]);
  });

  test('outside drops NaN', () => {
    // outside(2, 2): value < 2 || value > 2 — keeps 1 and 3, drops NaN.
    expect(over(outside(2, 2))).toEqual([1, 3]);
  });
});

describe('order()/min()/max() use a total order (NaN last, deterministic)', () => {
  // order/min/max use `compareOrder` (NaN sorts last, NaN === NaN) — a TOTAL
  // order, so the result is deterministic and identical to the Rust engine (a
  // total comparator makes the sort algorithm irrelevant). This is the language
  // rule (TinkerPop's `Double.compare` / SQL), distinct from the predicate path
  // above, where NaN stays JS-unordered and is filtered.

  test('min() never picks the NaN (largest) → smallest real value', () => {
    // NaN leading, to prove position-independence (no "sticky first" quirk).
    expect(arr(traversal(inject(Number.NaN, 3, 1), min()))).toEqual([1]);
  });

  test('max() keeps the NaN — it is the largest (like JS Math.max)', () => {
    const r = arr(traversal(inject(3, Number.NaN, 1), max()));
    expect(r).toHaveLength(1);
    expect(Number.isNaN(r[0])).toBe(true);
  });

  test('order() sorts the NaN last, deterministically', () => {
    expect(
      arr(traversal(inject(3, Number.NaN, 1), order())).map((x) => (Number.isNaN(x) ? 'NaN' : x)),
    ).toEqual([1, 3, 'NaN']);
  });
});

describe('NaN dedup already matches JS Set semantics (regression lock)', () => {
  test('two NaNs collapse to one, kept distinct from null', () => {
    // SameValueZero: NaN === NaN (dedup), NaN !== null (kept separate).
    const r = arr(traversal(inject(Number.NaN, Number.NaN, null), dedupe()));
    expect(r).toHaveLength(2);
    expect(Number.isNaN(r[0])).toBe(true);
    expect(r[1]).toBeNull();
  });
});
