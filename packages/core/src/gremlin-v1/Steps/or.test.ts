import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import { describe, test, expect } from 'bun:test';
import { inE } from './inE';
import { outE } from './outE';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('or tests', () => {
    // Do some tests here

    test('or test 1', () => {
      const result = tinkerGraph
        .traverse(g => g.V().or(outE('KNOWS'), outE('CREATED')).values('name'))
        .toArray();

      expect(result).toEqual(['marko', 'josh', 'peter']);
    });

    test('or test can filter everything', () => {
      const result = tinkerGraph
        .traverse(g => g.V().hasLabel('SOFTWARE').or(outE('KNOWS')).values('name'))
        .toArray();

      expect(result).toEqual([]);
    });

    test('or test 2', () => {
      const result = tinkerGraph
        .traverse(g => g.V().or(inE('KNOWS'), outE('CREATED')).values('name'))
        .toArray();

      expect(result).toEqual(['marko', 'vadas', 'josh', 'peter']);
    });
  });
});
