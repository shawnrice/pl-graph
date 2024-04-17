import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('both tests', () => {
    // Do some tests here

    test('toy test', () => {
      const result = tinkerGraph.traverse(query =>
        query.V('4').both('KNOWS', 'CREATED', 'BLAH').toArray(),
      );
      expect(result.map(x => x.properties.name)).toEqual(['marko', 'ripple', 'lop']);
    });

    test('get a specific label', () => {
      const result = tinkerGraph.traverse(query => query.V('1').both('KNOWS').toArray());

      expect(result.map(x => x.properties.name)).toEqual(['vadas', 'josh']);
    });

    test('getting all the labels is like asking for none of the labels', () => {
      const result = tinkerGraph.traverse(query => query.V('4').both().toArray());
      expect(result.map(x => x.properties.name)).toEqual(['ripple', 'lop', 'marko']);
    });
  });
});
