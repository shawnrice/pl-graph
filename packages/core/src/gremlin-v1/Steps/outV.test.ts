import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('outV tests', () => {
    test('toy test', () => {
      const result = tinkerGraph.traverse(query => query.V('4').inE().outV().toArray());

      expect(result[0].properties.name).toBe('marko');
    });
  });
});
