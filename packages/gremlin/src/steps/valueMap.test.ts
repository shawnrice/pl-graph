import { describe, expect, test } from 'bun:test';

import { run } from '../executor.js';
import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph.js';
import { E, V, valueMap } from '../steps.js';
import { traversal } from '../traversal.js';

const arr = (r: Iterable<unknown>): unknown[] => [...r];

describe('Gremlin tests', () => {
  describe('STEP, valueMap tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it can get all properties', () => {
      const result = arr(run(traversal(V(), valueMap()), tinkerGraph));
      expect(result).toEqual([
        { name: 'marko', age: 29 },
        { name: 'vadas', age: 27 },
        { name: 'josh', age: 32 },
        { name: 'peter', age: 35 },
        { name: 'lop', lang: 'java' },
        { name: 'ripple', lang: 'java' },
      ]);
    });

    test('it can get a single property', () => {
      const result = arr(run(traversal(V(), valueMap('age')), tinkerGraph));
      expect(result).toEqual([{ age: 29 }, { age: 27 }, { age: 32 }, { age: 35 }, {}, {}]);
    });

    // doc: g.V().valueMap('age','blah') — same as valueMap('age'); 'blah' is silently skipped.
    // Drift: TinkerPop wraps single-cardinality values in a list ([29]); our v2 impl
    // returns bare scalars (29).
    test('valueMap silently skips missing keys', () => {
      const result = arr(run(traversal(V(), valueMap('age', 'blah')), tinkerGraph));
      expect(result).toEqual([{ age: 29 }, { age: 27 }, { age: 32 }, { age: 35 }, {}, {}]);
    });

    // doc: g.E().valueMap() — edge property maps; edges have single-cardinality values.
    test('valueMap on edges yields one entry per edge', () => {
      const result = arr(run(traversal(E(), valueMap()), tinkerGraph));
      expect(result).toEqual([
        { weight: 0.5 },
        { weight: 1.0 },
        { weight: 0.4 },
        { weight: 1.0 },
        { weight: 0.4 },
        { weight: 0.2 },
      ]);
    });

    // doc: g.V().valueMap(true) — the includeTokens overload prepends id + label.
    test('valueMap(true) prepends id + label on vertices', () => {
      const result = arr(run(traversal(V(), valueMap(true)), tinkerGraph));
      expect(result[0]).toEqual({ id: '1', label: 'PERSON', name: 'marko', age: 29 });
      // A software vertex carries lang, not age.
      expect(result[4]).toEqual({ id: '3', label: 'SOFTWARE', name: 'lop', lang: 'java' });
    });

    // doc: g.V().valueMap(true,'name') — tokens plus a filtered key set.
    test('valueMap(true, key) keeps id + label and filters properties', () => {
      const result = arr(run(traversal(V(), valueMap(true, 'name')), tinkerGraph));
      expect(result[0]).toEqual({ id: '1', label: 'PERSON', name: 'marko' });
    });

    // doc: g.E().valueMap(true) — edges get id + label, but no IN/OUT (unlike elementMap).
    test('valueMap(true) prepends id + label on edges, no IN/OUT', () => {
      const result = arr(run(traversal(E(), valueMap(true)), tinkerGraph));
      expect(result[0]).toEqual({ id: '7', label: 'KNOWS', weight: 0.5 });
    });

    // valueMap() without the boolean stays properties-only (no tokens).
    test('valueMap() omits id + label', () => {
      const result = arr(run(traversal(V(), valueMap()), tinkerGraph));
      expect(result[0]).toEqual({ name: 'marko', age: 29 });
    });
  });
});
