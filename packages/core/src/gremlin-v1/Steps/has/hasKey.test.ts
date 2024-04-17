import { describe, expect, test } from 'bun:test';

/* eslint-disable no-magic-numbers */
import { createTestTinkerGraph } from '../../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('hasKey tests', () => {
    test.skip('we can filter just by key', () => {
      const result = tinkerGraph.traverse(g => g.V().properties().hasKey('age').value()).toArray();

      expect(result).toEqual([29, 27, 32, 35]);
    });
  });
});
