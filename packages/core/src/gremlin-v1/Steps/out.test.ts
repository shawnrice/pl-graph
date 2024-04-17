import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('out tests', () => {
    // Do some tests here

    test('toy test', () => {
      const result = tinkerGraph.traverse(query => query.V('4').out()).toArray();
      expect(result.map(x => x.properties.name)).toEqual(['ripple', 'lop']);
    });

    test('double out', () => {
      const result = tinkerGraph.traverse(g => g.V().out().out()).toArray();
      expect(result.map(x => x.properties.name)).toEqual(['ripple', 'lop']);
    });

    test('get a specific label', () => {
      const result = tinkerGraph.traverse(query => query.V('1').out('KNOWS')).toArray();

      expect(result.map(x => x.properties.name)).toEqual(['vadas', 'josh']);
    });

    test('get multiple labels', () => {
      const result = tinkerGraph.traverse(query => query.V('1').out('KNOWS', 'CREATED')).toArray();

      expect(result.map(x => x.properties.name)).toEqual(['vadas', 'josh', 'lop']);
    });

    test('getting all the labels is like asking for none of the labels', () => {
      expect(tinkerGraph.traverse(query => query.V('1').out('KNOWS', 'CREATED')).toArray()).toEqual(
        tinkerGraph.traverse(query => query.V('1').out()).toArray(),
      );
    });

    test('querying labels in order matters', () => {
      expect(
        tinkerGraph
          .traverse(query => query.V('1').out('CREATED', 'KNOWS'))
          .toArray()
          .map(x => x.properties.name),
      ).toEqual(['lop', 'vadas', 'josh']);
    });
  });
});
