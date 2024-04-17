import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import { Traversal } from '../Traversal';
import { outE } from './';

describe('Gremlin tests', () => {
  describe('STEP, WHERE tests', () => {
    const tinkerGraph = createTestTinkerGraph();

    test.skip('it works', () => {
      const result = tinkerGraph.traverse(g => g.V().where(outE('created'))).toArray();

      expect(result).toEqual(['marko', 29]);
    });
  });
});
