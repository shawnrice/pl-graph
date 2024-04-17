import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('bothV tests', () => {
    test('toy test', () => {
      const result = tinkerGraph.traverse(query => query.V('4').inE().bothV()).toArray();

      expect(result[0].properties.name).toBe('marko');
      expect(result[1].properties.name).toBe('josh');
    });
  });
});
