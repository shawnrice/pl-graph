/* eslint-disable yoda */
import { describe, expect, test } from 'bun:test';

import { createTestTinkerGraph } from '../fixtures/createTestTinkerGraph';
import { graph2PGJSON } from './serialize';

describe('serialize tests', () => {
  /**
   * @see https://tinkerpop.apache.org/docs/current/images/tinkerpop-modern.png
   */
  const graph = createTestTinkerGraph();

  test('It serializes a graph with nodes and edges', () => {
    const result = graph2PGJSON(graph);
    expect(result).toHaveProperty('nodes');
    expect(result.nodes).toHaveLength(6);

    expect(result).toHaveProperty('edges');
    expect(result.edges).toHaveLength(6);
  });
});
