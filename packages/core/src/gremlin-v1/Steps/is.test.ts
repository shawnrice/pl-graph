import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import * as P from '../Predicates';

describe('Gremlin tests', () => {
  describe('STEP is tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test.skip('it works with a simple number', () => {
      const result = tinkerGraph.traverse(g => g.V().values('age').is(32)).toArray();

      expect(result).toEqual([32]);
    });

    test.skip('it works with a comparator', () => {
      const result = tinkerGraph.traverse(g => g.V().values('age').is(P.lte(30))).toArray();

      expect(result).toEqual([29, 27]);
    });

    test.skip('it works with another comparator', () => {
      const result = tinkerGraph.traverse(g => g.V().values('age').is(P.inside(30, 40))).toArray();

      expect(result).toEqual([32, 35]);
    });

    test.skip('it works with WHERE', () => {
      // Find projects having exactly one contributor
      const result = tinkerGraph
        .traverse(g => g.V().where(__.in('created').count().is(1)).values('name'))
        .toArray();

      expect(result).toEqual(['ripple']);
    });

    test.skip('it works with WHERE 2', () => {
      // Find projects having two or more contributors.
      const result = tinkerGraph
        .traverse(g =>
          g
            .V()
            .where(__.in('created').count().is(P.gte(2)))
            .values('name'),
        )
        .toArray();

      expect(result).toEqual(['lop']);
    });

    test.skip('it works with WHERE 3', () => {
      // Find projects whose contributors average age is between 30 and 35.
      const result = tinkerGraph
        .traverse(g =>
          g
            .V()
            .where(__.in('created').values('age').mean().is(P.inside(30, 35)))
            .values('name'),
        )
        .toArray();

      expect(result).toEqual(['lop', 'ripple']);
    });
  });
});
