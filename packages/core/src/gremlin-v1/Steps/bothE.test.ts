import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('bothE tests', () => {
    // Do some tests here

    test('toy test', () => {
      const result = tinkerGraph.traverse(query =>
        query.V('4').bothE('KNOWS', 'CREATED', 'BLAH').toArray(),
      );

      expect(result[0].from.properties.name).toBe('marko');
      expect(result[0].labels.has('KNOWS')).toBe(true);
      expect(result[0].hasLabel('KNOWS')).toBe(true);
      expect(result[1].to.properties.name).toBe('ripple');
      expect(result[1].labels.has('CREATED')).toBe(true);
      expect(result[1].hasLabel('CREATED')).toBe(true);
      expect(result[2].to.properties.name).toBe('lop');
      expect(result[2].hasLabel('CREATED')).toBe(true);
      expect(result).toHaveLength(3);
    });

    test('get a specific label', () => {
      const result = tinkerGraph.traverse(query => query.V('1').bothE('KNOWS').toArray());

      expect(result.map(x => x.to.properties.name)).toEqual(['vadas', 'josh']);
    });

    test('getting all the labels is like asking for none of the labels', () => {
      const result = tinkerGraph.traverse(query => query.V('4').bothE().toArray());
      expect(result.map(x => x.to.properties.name)).toEqual(['ripple', 'lop', 'josh']);
    });
  });
});
