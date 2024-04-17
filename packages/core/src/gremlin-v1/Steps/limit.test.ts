import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, limit tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it limits to three', () => {
      const result = tinkerGraph.traverse(g => g.V().limit(3).values('name')).toArray();

      expect(result).toEqual(['marko', 'vadas', 'josh']);
    });

    test('it can skip and take', () => {
      const result = tinkerGraph.traverse(g => g.V().values('age').skip(2).limit(1)).toArray();

      expect(result).toEqual([32]);
    });

    test('it can have an open end', () => {
      const result = tinkerGraph
        .traverse(g => g.V().hasLabel('SOFTWARE').values('name').limit(90))
        .toArray();

      expect(result).toEqual(['lop', 'ripple']);
    });
  });
});
