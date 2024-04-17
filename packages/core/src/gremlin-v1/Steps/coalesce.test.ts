import { UnaryFn } from '../../../../fp/src/types';
import { Vertex } from '../../core/Vertex';
import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import * as P from '../Predicates';
import { Traverser } from '../Traverser';
import {
  hasLabel,
  identity,
  in_,
  option,
  out,
  outE,
  values,
} from './';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP coalesce tests', () => {
    // Do some tests here

    test('coalesce can get the first option', () => {
      // If the traversal yields an element, then do in, else do out (i.e. true/false-based option selection).
      const result = tinkerGraph.traverse(g =>
        g.V('1').coalesce(outE('CREATED'), outE('KNOWS')).inV().values('name'),
      );

      expect(result.toArray()).toEqual(['lop']);
    });

    test('coalesce can get the second option', () => {
      // If the traversal yields an element, then do in, else do out (i.e. true/false-based option selection).
      const result = tinkerGraph.traverse(g =>
        g.V('1').coalesce(outE('KNOWS'), outE('CREATED')).inV().values('name'),
      );

      expect(result.toArray()).toEqual(['vadas', 'josh']);
    });
  });
});
