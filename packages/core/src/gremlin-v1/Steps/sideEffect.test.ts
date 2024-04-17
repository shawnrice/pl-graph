import { describe, expect, mock, test } from 'bun:test';

import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP sideEffect', () => {
    test('sideEffect calls effects', () => {
      const noop = mock(() => {});
      const result = tinkerGraph
        .traverse(g => g.V().hasLabel('SOFTWARE').sideEffect(noop).values('name'))
        .toArray();
      expect(result).toEqual(['lop', 'ripple']);
      expect(noop).toHaveBeenCalledTimes(2);
    });
  });
});
