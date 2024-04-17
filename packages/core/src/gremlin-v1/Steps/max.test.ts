import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP max tests', () => {
    test('max works with numbers', () => {
      const result = tinkerGraph.traverse<string>(g => g.V().values('age').max()).toArray();

      expect(result).toEqual([35]);
    });

    test('max works with strings, again', () => {
      const result = tinkerGraph.traverse<string>(g => g.V().values('name').max()).toArray();

      expect(result).toEqual(['vadas']);
    });

    test('max filters out null', () => {
      const result = tinkerGraph.traverse<string>(g => g.inject(null, 10, 9, null).max()).toArray();

      expect(result).toEqual([10]);
    });

    test('max takes null if that is all it got', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.inject(null, null, null, null).max())
        .toArray();

      expect(result).toEqual([null]);
    });
  });
});
