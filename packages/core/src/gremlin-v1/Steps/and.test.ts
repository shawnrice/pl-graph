import { describe, expect, test } from 'bun:test';

import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import { inE } from './inE';
import { outE } from './outE';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('and tests', () => {
    // Do some tests here

    test('and test 1', () => {
      const result = tinkerGraph
        .traverse(g => g.V().and(outE('KNOWS'), outE('CREATED')).values('name'))
        .toArray();

      expect(result).toEqual(['marko']);
    });

    test('and test can filter everything', () => {
      const result = tinkerGraph
        .traverse(g => g.V().hasLabel('SOFTWARE').and(outE('KNOWS')).values('name'))
        .toArray();

      expect(result).toEqual([]);
    });

    test('and test 2', () => {
      const result = tinkerGraph
        .traverse(g => g.V().and(inE('KNOWS'), outE('CREATED')).values('name'))
        .toArray();

      expect(result).toEqual(['josh']);
    });
  });
});
