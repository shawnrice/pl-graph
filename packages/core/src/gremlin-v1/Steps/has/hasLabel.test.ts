import { describe, expect, test } from 'bun:test';

import { createTestTinkerGraph } from '../../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('hasLabel tests', () => {
    test('we can get all the person vertices', () => {
      const result = tinkerGraph.traverse(traversal => traversal.V().hasLabel('PERSON').toArray());

      expect(result).toHaveLength(4);
    });

    test('we have stable query order', () => {
      // Query order is based on insertion order
      const result = tinkerGraph.traverse(traversal => traversal.V().hasLabel('PERSON').toArray());

      expect(result.map(x => x.properties.name)).toEqual(['marko', 'vadas', 'josh', 'peter']);
    });

    test('we can get a vertex', () => {
      const result = tinkerGraph.traverse(traversal => {
        return traversal.V('1').hasLabel('PERSON').toArray();
      });

      expect(result[0].properties.name).toBe('marko');
    });
  });
});
