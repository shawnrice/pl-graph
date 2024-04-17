import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP min tests', () => {
    test('min works with numbers', () => {
      const result = tinkerGraph.traverse<string>(g => g.V().values('age').min()).toArray();

      expect(result).toEqual([27]);
    });

    test('min works with strings, again', () => {
      const result = tinkerGraph.traverse<string>(g => g.V().values('name').min()).toArray();

      expect(result).toEqual(['josh']);
    });

    test('min filters out null', () => {
      const result = tinkerGraph.traverse<string>(g => g.inject(null, 10, 9, null).min()).toArray();

      expect(result).toEqual([9]);
    });

    test('min takes null if that is all it got', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.inject(null, null, null, null).min())
        .toArray();

      expect(result).toEqual([null]);
    });
  });
});
