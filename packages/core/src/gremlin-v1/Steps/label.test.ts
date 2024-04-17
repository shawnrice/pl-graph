import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';

describe('Gremlin tests', () => {
  describe('STEP, label tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test('it can get the label', () => {
      const result = tinkerGraph.traverse(g => g.V().label()).toArray();

      expect(result).toEqual(['PERSON', 'PERSON', 'PERSON', 'PERSON', 'SOFTWARE', 'SOFTWARE']);
    });

    test('it can get specific labels', () => {
      const result = tinkerGraph.traverse(g => g.V('1').outE().label()).toArray();

      expect(result).toEqual(['KNOWS', 'KNOWS', 'CREATED']);
    });

    test.skip('it acts as `keys` when used on a regular object', () => {
      const result = tinkerGraph.traverse(g => g.V('1').properties().label()).toArray();
      expect(result).toEqual(['name', 'age']);
    });
  });
});
