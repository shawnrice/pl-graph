import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('inE tests', () => {
    // Do some tests here

    test('toy test', () => {
      const result = tinkerGraph.traverse(query => query.V('4').inE()).toArray();
      expect(result[0].from.properties.name).toBe('marko');
      expect(result[0].from.properties.age).toBe(29);
      expect(result[0].properties.weight).toBe(1.0);
    });

    test('get a specific label', () => {
      const result = tinkerGraph.traverse(query => query.V('1').inE('KNOWS')).toArray();

      expect(result).toHaveLength(0);
    });

    test('get a specific label 2', () => {
      const result = tinkerGraph.traverse(query => query.V('3').inE('CREATED')).toArray();

      expect(result).toHaveLength(3);
      expect(result.map(x => x.from.properties.name)).toEqual(['marko', 'josh', 'peter']);
      expect(result.map(x => x.properties.weight)).toEqual([0.4, 0.4, 0.2]);
    });
  });
});
