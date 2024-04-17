import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP unfold tests', () => {
    test('fold setups up correctly', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V('1').out().fold().inject('gremlin', [1.23, 2.34]))
        .toArray();

      expect(result).toEqual([
        'gremlin',
        [1.23, 2.34],
        [
          tinkerGraph.getVertexById('3'),
          tinkerGraph.getVertexById('2'),
          tinkerGraph.getVertexById('4'),
        ],
      ]);
    });

    test('unfold, well, unfolds', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V('1').out().fold().inject('gremlin', [1.23, 2.34]).unfold())
        .toArray();

      expect(result).toEqual([
        'gremlin',
        1.23,
        2.34,
        tinkerGraph.getVertexById('3'),
        tinkerGraph.getVertexById('2'),
        tinkerGraph.getVertexById('4'),
      ]);
    });

    test('unfold is not deep', () => {
      const result1 = tinkerGraph.traverse<string>(g => g.inject(1, [2, 3, [4, 5, [6]]])).toArray();

      expect(result1).toEqual([1, [2, 3, [4, 5, [6]]]]);

      const result2 = tinkerGraph
        .traverse<string>(g => g.inject(1, [2, 3, [4, 5, [6]]]).unfold())
        .toArray();

      expect(result2).toEqual([1, 2, 3, [4, 5, [6]]]);
    });
  });
});
