import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('inV tests', () => {
    test('toy test', () => {
      const result = tinkerGraph.traverse(query => query.V('4').outE().inV()).toArray();

      expect(result[0].properties.name).toBe('ripple');
      expect(result[1].properties.name).toBe('lop');
    });
  });
});
