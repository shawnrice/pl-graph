import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, skip tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it gets the first three', () => {
      const result = tinkerGraph.traverse(g => g.V().range(0, 3)).toArray();

      expect(result.map(x => x.properties.name)).toEqual(['marko', 'vadas', 'josh']);
    });

    test('it can skip on the low end', () => {
      const result = tinkerGraph.traverse(g => g.V().values('age').skip(2)).toArray();

      expect(result).toEqual([32, 35]);
    });

    test('it can have an open end', () => {
      const result = tinkerGraph.traverse(g => g.V().values('name').skip(3)).toArray();

      expect(result).toEqual(['peter', 'lop', 'ripple']);
    });
  });
});
