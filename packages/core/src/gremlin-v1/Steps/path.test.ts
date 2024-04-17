import { Vertex } from '../../core/Vertex';
import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('path tests', () => {
    test('simple tinker toy path', () => {
      const result = tinkerGraph.traverse(g => g.V().out().out().path()).toArray();
      expect(result).toEqual([
        [
          tinkerGraph.getVertexById('1'),
          tinkerGraph.getVertexById('4'),
          tinkerGraph.getVertexById('5'),
        ],
        [
          tinkerGraph.getVertexById('1'),
          tinkerGraph.getVertexById('4'),
          tinkerGraph.getVertexById('3'),
        ],
      ]);
    });

    test('complex paths work', () => {
      const result = tinkerGraph.traverse(g => g.V().outE().inV().outE().inV().path()).toArray();
      expect(result).toHaveLength(2);
      expect(result).toEqual([
        [
          tinkerGraph.getVertexById('1'),
          tinkerGraph.getEdgeById('8'),
          tinkerGraph.getVertexById('4'),
          tinkerGraph.getEdgeById('10'),
          tinkerGraph.getVertexById('5'),
        ],
        [
          tinkerGraph.getVertexById('1'),
          tinkerGraph.getEdgeById('8'),
          tinkerGraph.getVertexById('4'),
          tinkerGraph.getEdgeById('10'),
          tinkerGraph.getVertexById('3'),
        ],
      ]);
    });

    /**
     * @see https://tinkerpop.apache.org/docs/current/reference/#path-step
     */
    test('path works with by', () => {
      const res1 = tinkerGraph.traverse(query => query.V().both().path()).toArray();

      expect(res1.map(x0 => x0.map((x1: Vertex) => x1.properties.name))).toEqual([
        ['marko', 'vadas'],
        ['marko', 'josh'],
        ['marko', 'lop'],
        ['vadas', 'marko'],
        ['josh', 'ripple'],
        ['josh', 'lop'],
        ['josh', 'marko'],
        ['peter', 'lop'],
        ['lop', 'marko'],
        ['lop', 'josh'],
        ['lop', 'peter'],
        ['ripple', 'josh'],
      ]);

      const result = tinkerGraph.traverse(query => query.V().both().path().by('age')).toArray();

      expect(result).toEqual([
        [29, 27],
        [29, 32],
        [27, 29],
        [32, 29],
      ]);
    });
  });
});
