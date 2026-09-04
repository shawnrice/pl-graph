import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { V, both, inject, max, min, repeat, values } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP min tests', () => {
    test('min works with numbers', () => {
      const r = run(traversal(V(), values('age'), min()), tinkerGraph);
      expect(arr(r)).toEqual([27]);
    });

    test('min over strings uses the total order (like the native engine + TinkerPop)', () => {
      // min()/max() order ANY comparable over the engine's total order — no longer
      // numeric-only (that faulted on `values('name').min()`, diverging from native).
      expect(arr(run(traversal(V(), values('name'), min()), tinkerGraph))).toEqual(['josh']);
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

    test('min/max reduce over the total order — strings too, not just numbers', () => {
      // Regression: an earlier numeric-only guard threw on `values('name').min()`,
      // diverging from the native engine and TinkerPop (both order any comparable).
      expect(arr(run(traversal(V(), values('name'), min()), tinkerGraph))).toEqual(['josh']);
      expect(arr(run(traversal(V(), values('name'), max()), tinkerGraph))).toEqual(['vadas']);
      expect(arr(run(traversal(V(), values('age'), min()), tinkerGraph))).toEqual([27]);
      expect(arr(run(traversal(V(), values('age'), max()), tinkerGraph))).toEqual([35]);
    });
  });
});
