/* eslint-disable no-magic-numbers */
import {
  describe,
  expect,
  test,
} from 'bun:test';

import { createTestTinkerGraph } from '../../../fixtures/createTestTinkerGraph';
import * as P from '../../Predicates';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('has tests', () => {
    test('we can filter using has with a predicate', () => {
      const result = tinkerGraph
        .traverse(g => g.V().hasLabel('PERSON').out().has('name', P.within('vadas', 'josh')))
        .toArray();
      expect(result.map(x => x.id)).toEqual(['2', '4']);
    });

    test('we can chian filters using with predicates', () => {
      const result = tinkerGraph
        .traverse(g =>
          g
            .V()
            .hasLabel('PERSON')
            .out()
            .has('name', P.within('vadas', 'josh'))
            .outE()
            .hasLabel('CREATED'),
        )
        .toArray();
      expect(result.map(x => x.id)).toEqual(['10', '11']);
    });

    test.skip('we can get all the person vertices', () => {
      // Find all vertices whose ages are between 20 (exclusive) and 30 (exclusive). In other words, the age must be greater than 20 and less than 30.
      const result = tinkerGraph
        .traverse(g => g.V().has('age', P.inside(20, 30)).values('age'))
        .toArray();

      expect(result).toEqual([29, 27]);
    });

    test.skip('we can get all the person vertices 2', () => {
      // Find all vertices whose ages are not between 20 (inclusive) and 30 (inclusive). In other words, the age must be less than 20 or greater than 30.
      const result = tinkerGraph
        .traverse(g => g.V().has('age', P.outside(20, 30)).values('age'))
        .toArray();

      expect(result).toEqual([29, 27]);
    });

    test('we can do some other things', () => {
      const result = tinkerGraph
        .traverse(g => g.V().has('PERSON', 'name', P.startingWith('m')))
        .toArray();
      expect(result.map(x => x.id)).toEqual(['1']);
    });

    test('we can filter just by key', () => {
      const result = tinkerGraph.traverse(g => g.V().has('age')).toArray();

      expect(result.map(x => x.id)).toEqual(['1', '2', '4', '6']);
      expect(result.map(x => x.properties.name)).toEqual(['marko', 'vadas', 'josh', 'peter']);
    });

    test('calling with a first arg of null is empty', () => {
      const result = tinkerGraph.traverse(g => g.V().has(null, 'vadas')).toArray();
      expect(result).toEqual([]);
    });
  });
});
