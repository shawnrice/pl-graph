import { describe, expect, test } from 'bun:test';

import { Vertex } from '../../../core/Vertex';
/* eslint-disable max-lines-per-function */
/* eslint-disable no-magic-numbers */
import { createTestTinkerGraph } from '../../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('by tests', () => {
    test('by works with select', () => {
      // A standard select() that generates a Map<String,Object> of variables bindings in the path (i.e. a and b) for the sake of a running example.
      const result = tinkerGraph
        .traverse(g =>
          g.V().as('a').out('CREATED').in('CREATED').as('b').select('a', 'b').by('name'),
        )
        .toArray();

      expect(result).toEqual([
        { a: 'marko', b: 'marko' },
        { a: 'marko', b: 'josh' },
        { a: 'marko', b: 'peter' },
        { a: 'josh', b: 'josh' },
        { a: 'josh', b: 'marko' },
        { a: 'josh', b: 'josh' },
        { a: 'josh', b: 'peter' },
        { a: 'peter', b: 'marko' },
        { a: 'peter', b: 'josh' },
        { a: 'peter', b: 'peter' },
      ]);
    });

    test.skip('select with where works', () => {
      // The select().by('name') projects each binding vertex to their name property value and where() operates to ensure respective a and b strings are not the same.
      const result = tinkerGraph
        .traverse(g =>
          g
            .V()
            .as('a')
            .out('created')
            .in('created')
            .as('b')
            .select('a', 'b')
            .by('name')
            .where('a', neq('b')),
        )
        .toArray();

      expect(result).toEqual([
        { a: 'marko', b: 'josh' },
        { a: 'marko', b: 'peter' },
        { a: 'josh', b: 'marko' },
        { a: 'josh', b: 'peter' },
        { a: 'peter', b: 'marko' },
        { a: 'peter', b: 'josh' },
      ]);
    });

    test.skip('select with where works 2', () => {
      // The first select() projects a vertex binding set. A binding is filtered if a vertex equals b vertex. A binding is filtered if a doesn’t know b. The second and final select() projects the name of the vertices.
      const result = tinkerGraph
        .traverse(g =>
          g
            .V()
            .as('a')
            .out('created')
            .in('created')
            .as('b')
            .select('a', 'b') //// (3)
            .where('a', neq('b'))
            .where(__.as('a').out('knows').as('b'))
            .select('a', 'b')
            .by('name'),
        )
        .toArray();

      expect(result).toEqual([
        { a: 'marko', b: 'josh' },
        { a: 'marko', b: 'peter' },
        { a: 'josh', b: 'marko' },
        { a: 'josh', b: 'peter' },
        { a: 'peter', b: 'marko' },
        { a: 'peter', b: 'josh' },
      ]);
    });
  });
});
