import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP count tests', () => {
    test('count works', () => {
      const result1 = tinkerGraph.traverse<string>(g => g.V().count()).toArray();

      expect(result1).toEqual([6]);
    });

    test('count works, again', () => {
      const result1 = tinkerGraph.traverse<string>(g => g.V().hasLabel('PERSON').count()).toArray();

      expect(result1).toEqual([4]);
    });

    test('count works again and again', () => {
      const result1 = tinkerGraph
        .traverse<string>(g => g.V().hasLabel('PERSON').out().count())
        .toArray();
      const result2 = tinkerGraph
        .traverse<string>(g => g.V().hasLabel('PERSON').out().values('name'))
        .toArray();

      expect(result1).toEqual([6]);
      expect(result2).toEqual(['vadas', 'josh', 'lop', 'ripple', 'lop', 'lop']);
    });
  });
});
