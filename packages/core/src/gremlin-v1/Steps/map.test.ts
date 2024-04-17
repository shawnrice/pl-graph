import { describe, expect, test } from 'bun:test';

import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, map tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it can map properties', () => {
      const result = tinkerGraph.traverse(g => g.V().map(v => v.properties.name)).toArray();

      expect(result).toEqual(['marko', 'vadas', 'josh', 'peter', 'lop', 'ripple']);
    });
  });
});
