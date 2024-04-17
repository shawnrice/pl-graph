import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP dedupe tests', () => {
    // Do some tests here

    test('it deduplicates strings', () => {
      const result = tinkerGraph.traverse(g => g.V().values('lang'));

      expect(result.toArray()).toEqual(['java', 'java']);

      expect(result.dedup().toArray()).toEqual(['java']);
    });

    // WE CURRENTLY DON'T SUPPORT DE-DUPING ON LABELS
    test.skip('it dedupes big things', () => {
      const result1 = tinkerGraph.traverse(g =>
        g.V().as('a').out('CREATED').as('b').in('CREATED').as('c').select('a', 'b', 'c'),
      );

      expect(result1.toArray()).toEqual([
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('1'),
        },
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('4'),
        },
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('6'),
        },
        {
          a: tinkerGraph.getVertexById('4'),
          b: tinkerGraph.getVertexById('5'),
          c: tinkerGraph.getVertexById('4'),
        },
        {
          a: tinkerGraph.getVertexById('4'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('1'),
        },
        {
          a: tinkerGraph.getVertexById('4'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('4'),
        },
        {
          a: tinkerGraph.getVertexById('4'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('6'),
        },
        {
          a: tinkerGraph.getVertexById('6'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('1'),
        },
        {
          a: tinkerGraph.getVertexById('6'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('4'),
        },
        {
          a: tinkerGraph.getVertexById('6'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('6'),
        },
      ]);

      const result2 = tinkerGraph.traverse(g =>
        g
          .V()
          .as('a')
          .out('CREATED')
          .as('b')
          .in('CREATED')
          .as('c')
          .dedup('a', 'b')
          .select('a', 'b', 'c'),
      );

      expect(result2.toArray()).toEqual([
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('1'),
        },
        {
          a: tinkerGraph.getVertexById('4'),
          b: tinkerGraph.getVertexById('5'),
          c: tinkerGraph.getVertexById('4'),
        },
        {
          a: tinkerGraph.getVertexById('4'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('1'),
        },
        {
          a: tinkerGraph.getVertexById('6'),
          b: tinkerGraph.getVertexById('3'),
          c: tinkerGraph.getVertexById('1'),
        },
      ]);
    });
  });
});
