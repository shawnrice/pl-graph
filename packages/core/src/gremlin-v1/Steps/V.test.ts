import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('V tests', () => {
    test('we can get all the vertices', () => {
      const result = tinkerGraph.traverse(traversal => traversal.V()).toArray();

      expect(result).toHaveLength(6);
    });

    test('we have stable query order', () => {
      // Query order is based on insertion order
      const result = tinkerGraph.traverse(traversal => traversal.V()).toArray();

      expect(result.map(x => x.properties.name)).toEqual([
        'marko',
        'vadas',
        'josh',
        'peter',
        'lop',
        'ripple',
      ]);
    });

    test('we can get a vertex', () => {
      const result = tinkerGraph
        .traverse(traversal => {
          return traversal.V('1');
        })
        .toArray();

      expect(result[0].properties.name).toBe('marko');
    });
  });
});
