import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import { both } from './';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP mean tests', () => {
    test('mean works with numbers', () => {
      const result = tinkerGraph.traverse<string>(g => g.V().values('age').mean()).toArray();

      expect(result).toEqual([30.75]);
    });

    test.skip('mean works with repeat', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V().repeat(both()).times(3).values('age').mean())
        .toArray();

      expect(result).toEqual([30.75]);
    });

    test('mean filters out null', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.inject(null, 10, 9, null).mean())
        .toArray();

      expect(result).toEqual([9.5]);
    });

    test('mean takes null if that is all it got', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.inject(null, null, null, null).mean())
        .toArray();

      expect(result).toEqual([null]);
    });
  });
});
