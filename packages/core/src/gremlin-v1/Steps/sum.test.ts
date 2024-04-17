import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import { both } from './';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP sum tests', () => {
    test('sum works with numbers', () => {
      const result = tinkerGraph.traverse<string>(g => g.V().values('age').sum()).toArray();

      expect(result).toEqual([123]);
    });

    test.skip('sum works with repeat', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V().repeat(both()).times(3).values('age').sum())
        .toArray();

      expect(result).toEqual([1471]);
    });

    test('sum filters out null', () => {
      const result = tinkerGraph.traverse<string>(g => g.inject(null, 10, 9, null).sum()).toArray();

      expect(result).toEqual([19]);
    });

    test('sum takes null if that is all it got', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.inject(null, null, null, null).sum())
        .toArray();

      expect(result).toEqual([null]);
    });
  });
});
