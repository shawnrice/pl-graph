import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP tail tests', () => {
    test('tail works', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V().hasLabel('PERSON').values('name').tail())
        .toArray();

      expect(result).toEqual(['peter']);
    });

    test('tail works with order', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V().hasLabel('PERSON').values('name').order().tail())
        .toArray();

      expect(result).toEqual(['vadas']);
    });

    test('tail works with order and an index', () => {
      const result1 = tinkerGraph
        .traverse<string>(g => g.V().hasLabel('PERSON').values('name').order().tail())
        .toArray();

      const result2 = tinkerGraph
        .traverse<string>(g => g.V().hasLabel('PERSON').values('name').order().tail(1))
        .toArray();

      expect(result1).toEqual(result2);
    });

    test('tail works with multiple items', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V().values('name').order().tail(3))
        .toArray();

      expect(result).toEqual(['peter', 'ripple', 'vadas']);
    });
  });
});
