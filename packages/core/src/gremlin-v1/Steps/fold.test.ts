import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP fold tests', () => {
    test('fold works', () => {
      const result1 = tinkerGraph
        .traverse<string>(g => g.V('1').out('KNOWS').values('name'))
        .toArray();

      expect(result1).toEqual(['vadas', 'josh']);

      const result2 = tinkerGraph
        .traverse<string[]>(g => g.V('1').out('KNOWS').values('name').fold())
        .toArray();
      expect(result2).toEqual([['vadas', 'josh']]);
    });

    test('fold works with a bifunctor', () => {
      const result1 = tinkerGraph
        .traverse<number>(g => g.V('1').out('KNOWS').values('age'))
        .toArray();
      const result2 = tinkerGraph
        .traverse<number>(g =>
          g
            .V('1')
            .out('KNOWS')
            .values('age')
            .fold(0, (a: number, b: number) => a + b),
        )
        .toArray();

      expect(result1).toEqual([27, 32]);
      expect(result2).toEqual([59]);
    });
  });
});
