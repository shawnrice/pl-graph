import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, id tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('id works', () => {
      const result = tinkerGraph.traverse(g => g.V().id()).toArray();

      expect(result).toEqual(['1', '2', '4', '6', '3', '5']);
    });

    test.skip('id with is can filter', () => {
      const result = tinkerGraph.traverse(g => g.V(1).out().id().is(2)).toArray();

      expect(result).toEqual(['2']);
    });

    test('we can get some from other vertices', () => {
      const result = tinkerGraph.traverse(g => g.V('1').outE().id()).toArray();
      expect(result).toEqual(['7', '8', '9']);
    });
  });
});
