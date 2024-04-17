import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import * as P from '../Predicates';
import {
  hasLabel,
  identity,
  in_,
  option,
  out,
  values,
} from './';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP choose tests', () => {
    // We need to be able to chain in sub-traversals before we can unlock this one
    test.skip('choose can branch functionally', () => {
      // If the traversal yields an element, then do in, else do out (i.e. true/false-based option selection).
      const result = tinkerGraph.traverse(g =>
        g
          .V()
          .hasLabel('PERSON')
          .choose(values('age').is(P.lte(30)), in_(), out())
          .values('name'),
      );

      expect(result.toArray()).toEqual(['marko', 'ripple', 'lop', 'lop']);
    });

    test.skip('choose can branch with option', () => {
      // Use the result of the traversal as a key to the map of traversal options (i.e. value-based option selection).
      const result = tinkerGraph
        .traverse(g =>
          g
            .V()
            .hasLabel('PERSON')
            .choose(values('age'))
            .option(27, in_())
            .option(32, out())
            .values('name'),
        )
        .toArray();

      expect(result).toEqual(['marko', 'ripple', 'lop']);
    });

    test('choose with two args is if/then semantics', () => {
      // If the vertex is a person, emit the vertices they created, else emit the vertex.
      const result = tinkerGraph
        .traverse(g => g.V().choose(hasLabel('PERSON'), out('CREATED')).values('name'))
        .toArray();

      expect(result).toEqual(['lop', 'ripple', 'lop', 'lop', 'lop', 'ripple']);
    });

    test('choose can fallback to identity', () => {
      // If the vertex is a person, emit the vertices they created, else emit the vertex.
      const result = tinkerGraph
        .traverse(g => g.V().choose(hasLabel('PERSON'), out('CREATED'), identity()).values('name'))
        .toArray();

      expect(result).toEqual(['lop', 'ripple', 'lop', 'lop', 'lop', 'ripple']);
    });
  });
});
