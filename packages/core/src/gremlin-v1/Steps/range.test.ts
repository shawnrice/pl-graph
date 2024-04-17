import { Vertex } from '../../core/Vertex';
import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, range tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it gets the first three', () => {
      const result = tinkerGraph.traverse(g => g.V().range(0, 3)).toArray();

      expect(result.map(x => x.properties.name)).toEqual(['marko', 'vadas', 'josh']);
    });

    test('it can skip on the low end', () => {
      const result = tinkerGraph.traverse(g => g.V().range(3, 5)).toArray();

      expect(result.map((x: Vertex<any>) => x.properties.name)).toEqual(['peter', 'lop']);
    });

    test('it can have an open end', () => {
      const result = tinkerGraph.traverse(g => g.V().range(3, -1)).toArray();

      expect(result.map(x => x.properties.name)).toEqual(['peter', 'lop', 'ripple']);
    });
  });
});
