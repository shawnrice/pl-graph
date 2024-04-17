/* eslint-disable new-cap */
/* eslint-disable no-magic-numbers */
import { describe, expect, test } from 'bun:test';

import { createTestTinkerGraph } from '../../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('hasId tests', () => {
    test('we can filter just by key', () => {
      const result = tinkerGraph.traverse(g => g.V().hasId('1')).toArray();

      expect(result.map(x => x.id)).toEqual(['1']);
      expect(result.map(x => x.properties.name)).toEqual(['marko']);
    });

    test('we can filter by ids out of order', () => {
      const result = tinkerGraph.traverse(g => g.V().hasId('6', '2', '1', '4')).toArray();

      expect(result.map(x => x.id)).toEqual(['1', '2', '4', '6']);
      expect(result.map(x => x.properties.name)).toEqual(['marko', 'vadas', 'josh', 'peter']);
    });

    test('we can call it on edges', () => {
      const result = tinkerGraph.traverse(g => g.E().hasId('7', '8')).toArray();

      expect(result.map(x => x.id)).toEqual(['7', '8']);
    });

    test('we can do something more complex', () => {
      const result = tinkerGraph
        .traverse(g => g.E().hasId('7', '8').outV().out().out().hasId('5'))
        .toArray();

      expect(result.map(x => x.id)).toEqual(['5', '5']);
    });
  });
});
