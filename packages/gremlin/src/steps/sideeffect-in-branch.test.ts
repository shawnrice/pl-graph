import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import {
  V,
  aggregate,
  cap,
  coalesce,
  optional,
  out,
  pipe,
  sideEffect,
  store,
  union,
} from '../steps.js';
import { traversal } from '../traversal.js';

const g = createTestTinkerGraph();
const arr = (p: Parameters<typeof run>[0]): unknown[] => [...run(p, g)];
const capLen = (p: Parameters<typeof run>[0]): number => ((arr(p)[0] as unknown[]) ?? []).length;

// A side-effect (aggregate/store) nested inside a branching step must write to
// the SAME run context so a later cap() sees it. Previously the TS engine gave
// each sub-plan a fresh throwaway context, so cap() saw nothing — a real bug and
// a divergence from the Rust engine (which threads context correctly).
// marko's KNOWS edges → vadas, josh (2).
describe('side-effects inside branching steps reach cap (ctx threading)', () => {
  test('aggregate inside union', () => {
    expect(capLen(traversal(V(), union(pipe(out('KNOWS'), aggregate('y'))), cap('y')))).toBe(2);
  });
  test('aggregate inside coalesce', () => {
    expect(capLen(traversal(V(), coalesce(pipe(out('KNOWS'), aggregate('y'))), cap('y')))).toBe(2);
  });
  test('store inside optional', () => {
    expect(capLen(traversal(V(), optional(pipe(out('KNOWS'), store('y'))), cap('y')))).toBe(2);
  });
  test('aggregate inside sideEffect', () => {
    expect(capLen(traversal(V(), sideEffect(pipe(out('KNOWS'), aggregate('y'))), cap('y')))).toBe(
      2,
    );
  });
});
