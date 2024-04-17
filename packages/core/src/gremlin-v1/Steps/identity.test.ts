import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, id tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('identity works', () => {
      const result = tinkerGraph.traverse(g => g.V().identity()).toArray();

      expect(result).toEqual(['1', '2', '4', '6', '3', '5'].map(x => tinkerGraph.getVertexById(x)));

      expect(result.map(x => x.id)).toEqual(['1', '2', '4', '6', '3', '5']);
    });
  });
});
