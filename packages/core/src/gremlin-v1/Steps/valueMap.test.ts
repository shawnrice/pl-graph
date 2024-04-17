import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, valueMap tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it can get all properties', () => {
      const result = tinkerGraph.traverse(g => g.V().valueMap()).toArray();

      expect(result).toEqual([
        { name: 'marko', age: 29 },
        { name: 'vadas', age: 27 },
        { name: 'josh', age: 32 },
        { name: 'peter', age: 35 },
        { name: 'lop', lang: 'java' },
        { name: 'ripple', lang: 'java' },
      ]);
    });

    test('it can get a single properties', () => {
      const result = tinkerGraph.traverse(g => g.V().valueMap('age')).toArray();

      expect(result).toEqual([{ age: 29 }, { age: 27 }, { age: 32 }, { age: 35 }, {}, {}]);
    });
  });
});
