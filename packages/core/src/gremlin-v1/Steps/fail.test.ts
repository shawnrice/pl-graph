import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP fail tests', () => {
    test('fail works', () => {
      const result = tinkerGraph.traverse<string[]>(g =>
        g.V().has('person', 'name', 'peter').fold().fail('Test Fail'),
      );

      expect(() => result.toArray()).toThrow('Test Fail');
    });
  });
});
