import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { V, both, inject, max, repeat, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP max tests', () => {
    test('max works with numbers', () => {
      const r = run(traversal(V(), values('age'), max()), tinkerGraph);
      expect(arr(r)).toEqual([35]);
    });

    test('max over strings faults — min()/max() are numeric-only', () => {
      // TinkerPop orders any Comparable, but a mixed/string max is decided by an
      // arbitrary type rank, so we keep min()/max() to numbers (like sum()/mean()).
      expect(() => arr(run(traversal(V(), values('name'), max()), tinkerGraph))).toThrow(
        /numeric values/,
      );
    });

    // doc: g.V().repeat(both()).times(3).values('age').max() — 35
    test('max after repeat(both()).times(3)', () => {
      const r = run(traversal(V(), repeat(both()).times(3), values('age'), max()), tinkerGraph);
      expect(arr(r)).toEqual([35]);
    });

    test('max filters out null', () => {
      const r = run(traversal(inject(null, 10, 9, null), max()), tinkerGraph);
      expect(arr(r)).toEqual([10]);
    });

    test('max takes null if that is all it got', () => {
      const r = run(traversal(inject(null, null, null, null), max()), tinkerGraph);
      expect(arr(r)).toEqual([null]);
    });
  });
});
