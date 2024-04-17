import { describe, expect, test } from 'bun:test';

import { Vertex } from '../../core/Vertex';
import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import { Traverser } from '../Traverser';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('as tests', () => {
    // Do some tests here

    test('ensure it gets enqueued', () => {
      const result = tinkerGraph.traverse(query => query.V(tinkerGraph.getVertexById('1')).as('a'));

      // So, we know that we enqueued Marko, and that we just tag marko with `a`, so we should have marko
      expect(result.toArray()[0].properties.name).toBe('marko');
      expect(result.toArray()).toHaveLength(1);

      // This is checking some internals: we need to ensure that the vertex that represents marko
      // has been tagged as 'a', so we should check that. We can't call `toArray` here because
      // we'll lose the metadata
      const meta = Array.from(result.steps).reduce(
        (acc, step) => step.callback(acc),
        result.current,
      );

      const [first] = Array.from(meta as Generator<Traverser<Vertex<any>>>) as Traverser<
        Vertex<any>
      >[];
      expect(first!.sideEffect('a')).toBe(first!.get());
    });

    test('multiple as statements do not affect the return', () => {
      const result = tinkerGraph.traverse(g => g.V().as('a').out().as('b').out().as('c')).toArray();
      expect(result.map(x => x.properties.name)).toEqual(['ripple', 'lop']);
    });
  });
});
