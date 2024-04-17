import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, values tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it can get all values', () => {
      const result = tinkerGraph.traverse(g => g.V('1').values()).toArray();

      expect(result).toEqual(['marko', 29]);
    });

    test('it filters out vertices without specified value', () => {
      const result = tinkerGraph.traverse(g => g.V().values('age')).toArray();

      expect(result).toEqual([29, 27, 32, 35]);
    });

    test('it can get multiple values', () => {
      const result = tinkerGraph.traverse(g => g.V().values('name', 'age')).toArray();

      expect(result).toEqual(['marko', 29, 'vadas', 27, 'josh', 32, 'peter', 35, 'lop', 'ripple']);
    });
  });
});
