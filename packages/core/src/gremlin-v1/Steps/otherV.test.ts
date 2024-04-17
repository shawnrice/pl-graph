import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('otherV tests', () => {
    // Do some tests here

    test('toy test', () => {
      const result = tinkerGraph
        .traverse(q => q.V('4').bothE('KNOWS', 'CREATED', 'blah').otherV())
        .toArray();

      expect(result.map(x => x.id)).toEqual(['1', '5', '3']);
      expect(result[0].properties.name).toBe('marko');
      expect(result[1].properties.name).toBe('ripple');
      expect(result[2].properties.name).toBe('lop');
    });
  });
});
