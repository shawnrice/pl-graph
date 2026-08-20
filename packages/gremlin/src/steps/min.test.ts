import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { V, both, inject, min, repeat, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP min tests', () => {
    test('min works with numbers', () => {
      const r = run(traversal(V(), values('age'), min()), tinkerGraph);
      expect(arr(r)).toEqual([27]);
    });

    test('min over strings faults — min()/max() are numeric-only', () => {
      // TinkerPop orders any Comparable, but a mixed/string max is decided by an
      // arbitrary type rank, so we keep min()/max() to numbers (like sum()/mean()).
      expect(() => arr(run(traversal(V(), values('name'), min()), tinkerGraph))).toThrow(
        /numeric values/,
      );
    });

    // doc: g.V().repeat(both()).times(3).values('age').min() — 27
    test('min after repeat(both()).times(3)', () => {
      const r = run(traversal(V(), repeat(both()).times(3), values('age'), min()), tinkerGraph);
      expect(arr(r)).toEqual([27]);
    });

    test('min filters out null', () => {
      const r = run(traversal(inject(null, 10, 9, null), min()), tinkerGraph);
      expect(arr(r)).toEqual([9]);
    });

    test('min takes null if that is all it got', () => {
      const r = run(traversal(inject(null, null, null, null), min()), tinkerGraph);
      expect(arr(r)).toEqual([null]);
    });
  });
});
