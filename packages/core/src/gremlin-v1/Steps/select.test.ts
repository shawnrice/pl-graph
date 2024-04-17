/**
 * @jest-environment jsdom
 */

import { Vertex } from '../../core/Vertex';
import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('select tests', () => {
    test('we can select multiple things', () => {
      const result = tinkerGraph
        .traverse(g => g.V().as('a').out().as('b').out().as('c').select('a', 'b', 'c'))
        .toArray();

      expect(result).toEqual([
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('4'),
          c: tinkerGraph.getVertexById('5'),
        },
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('4'),
          c: tinkerGraph.getVertexById('3'),
        },
      ]);

      expect(
        (result as Vertex<any>[]).map(x =>
          Object.fromEntries(Object.entries(x).map(([key, value]) => [key, value.id])),
        ),
      ).toEqual([
        { a: '1', b: '4', c: '5' },
        { a: '1', b: '4', c: '3' },
      ]);
    });

    test('we need not select everything we labeled', () => {
      const result = tinkerGraph
        .traverse(g => g.V().as('a').out().as('b').out().as('c').select('a', 'b'))
        .toArray();

      expect(result).toEqual([
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('4'),
        },
        {
          a: tinkerGraph.getVertexById('1'),
          b: tinkerGraph.getVertexById('4'),
        },
      ]);

      expect(
        result.map(x =>
          Object.fromEntries(Object.entries(x).map(([key, value]) => [key, (value as Vertex).id])),
        ),
      ).toEqual([
        { a: '1', b: '4' },
        { a: '1', b: '4' },
      ]);
    });

    test('select works with by', () => {
      const result = tinkerGraph
        .traverse(g =>
          g.V().as('a').out().as('b').out().as('c').select('a', 'b').as('d').by('name'),
        )
        .toArray();

      expect(result).toEqual([
        {
          a: 'marko',
          b: 'josh',
        },
        {
          a: 'marko',
          b: 'josh',
        },
      ]);
    });

    test('if the selection is one step, then no object is returned', () => {
      const result = tinkerGraph
        .traverse(g => g.V().as('a').out().as('b').out().as('c').select('a'))
        .toArray();

      expect(result).toEqual([tinkerGraph.getVertexById('1'), tinkerGraph.getVertexById('1')]);
    });

    test('we can select with an implicit filter', () => {
      const result = tinkerGraph
        .traverse(g => g.V('1').as('a').both().as('b').select('a', 'b').by('age'))
        .toArray();
      expect(result).toHaveLength(2);
      expect(result).toEqual([
        { a: 29, b: 27 },
        { a: 29, b: 32 },
      ]);
    });

    test('we can find a longer path and get the first node', () => {
      const result = tinkerGraph.traverse(g => g.V().as('x').out().out().select('x')).toArray();
      expect(result).toHaveLength(2);
      expect(result).toEqual([tinkerGraph.getVertexById('1'), tinkerGraph.getVertexById('1')]);
    });

    test('we can select after labeling the middle', () => {
      const result = tinkerGraph.traverse(g => g.V().out().as('x').out().select('x')).toArray();
      expect(result).toHaveLength(2);
      expect(result).toEqual([tinkerGraph.getVertexById('4'), tinkerGraph.getVertexById('4')]);
      expect(result.map(x => x.id)).toEqual(['4', '4']);
    });

    test('we can pointlessly select things', () => {
      // note, while this works, it's also dumb, so don't do it
      const result = tinkerGraph.traverse(g => g.V().out().out().as('x').select('x')).toArray();
      expect(result).toHaveLength(2);
      expect(result.map(x => x.properties.name)).toEqual(['ripple', 'lop']);
      expect(result.map(x => x.id)).toEqual(['5', '3']);
    });
  });
});
