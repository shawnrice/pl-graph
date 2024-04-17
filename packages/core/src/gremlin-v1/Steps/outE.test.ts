import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('outE tests', () => {
    // Do some tests here

    test('toy test', () => {
      const result = tinkerGraph.traverse(query => query.V('4').outE()).toArray();
      expect(result.map(x => x.properties.weight)).toEqual([1, 0.4]);
    });

    test('get a specific label', () => {
      const result = tinkerGraph.traverse(query => query.V('1').outE('KNOWS').toArray());

      expect(result[0].to.properties.name).toBe('vadas');
      expect(result[0].properties.weight).toBe(0.5);

      expect(result[1].to.properties.name).toBe('josh');
      expect(result[1].properties.weight).toBe(1);

      expect(result[2]).toBeUndefined();
    });

    test('get multiple labels', () => {
      const result = tinkerGraph.traverse(query => query.V('1').outE('KNOWS', 'CREATED').toArray());

      expect(result[0].to.properties.name).toBe('vadas');
      expect(result[0].properties.weight).toBe(0.5);

      expect(result[1].to.properties.name).toBe('josh');
      expect(result[1].properties.weight).toBe(1);

      expect(result[2].to.properties.name).toBe('lop');
      expect(result[2].properties.weight).toBe(0.4);

      expect(result[3]).toBeUndefined();
    });

    test('getting all the labels is like asking for none of the labels', () => {
      expect(
        tinkerGraph.traverse(query => query.V('1').outE('CREATED', 'KNOWS').toArray()),
      ).toEqual(tinkerGraph.traverse(query => query.V('1').outE().toArray()));
    });
  });
});
