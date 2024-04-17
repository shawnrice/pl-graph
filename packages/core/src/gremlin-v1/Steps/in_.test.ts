import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('in_ tests', () => {
    // Do some tests here

    test('toy test', () => {
      const result = tinkerGraph.traverse(query => query.V('4').in()).toArray();
      expect(result.map(x => x.properties.name)).toEqual(['marko']);
    });

    test('get a specific label', () => {
      const result = tinkerGraph.traverse(query => query.V('1').in('KNOWS')).toArray();

      expect(result.map(x => x.properties.name)).toEqual([]);
    });

    test('get a specific label 2', () => {
      const result = tinkerGraph.traverse(query => query.V('3').in('CREATED')).toArray();

      expect(result.map(x => x.properties.name)).toEqual(['marko', 'josh', 'peter']);
    });

    test('getting all the labels is like asking for none of the labels', () => {
      expect(tinkerGraph.traverse(query => query.V('3').in('CREATED')).toArray()).toEqual(
        tinkerGraph.traverse(query => query.V('3').in()).toArray(),
      );
    });
  });
});
