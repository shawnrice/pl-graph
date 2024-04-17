import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, properties tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test.skip('it gets the specified property', () => {
      const result = tinkerGraph.traverse(g => g.V().properties('name')).toArray();

      expect(result).toEqual([{ name: 'marko' }]);
    });
  });
});
