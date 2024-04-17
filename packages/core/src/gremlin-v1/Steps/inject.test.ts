import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP inject tests', () => {
    // Do some tests here

    test('it can inject a string', () => {
      const result = tinkerGraph
        .traverse(g => g.V('4').out().values('name').inject('daniel'))
        .toArray();

      expect(result).toEqual(['daniel', 'ripple', 'lop']);
    });

    test('it injected objects work like others', () => {
      const result = tinkerGraph
        .traverse(g =>
          g
            .V('4')
            .out()
            .values('name', 'idonotexist')
            .inject('daniel')
            .map(x => x.length),
        )
        .toArray();

      expect(result).toEqual([6, 6, 3]);
    });

    test('it injected objects work like others with path', () => {
      const result = tinkerGraph
        .traverse(g =>
          g
            .V('4')
            .out()
            .values('name')
            .inject('daniel')
            .map(x => x.length)
            .path(),
        )
        .toArray();

      expect(result).toEqual([
        ['daniel', 6], // We can see that inject starts a new traverser
        [tinkerGraph.getVertexById('4'), tinkerGraph.getVertexById('5'), 'ripple', 6],
        [tinkerGraph.getVertexById('4'), tinkerGraph.getVertexById('3'), 'lop', 3],
      ]);
    });
  });
});
