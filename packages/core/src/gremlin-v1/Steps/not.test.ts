import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import { hasLabel } from './';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('or tests', () => {
    // Do some tests here

    test('or test 1', () => {
      const result = tinkerGraph
        .traverse(g => g.V().not(hasLabel('PERSON')).values('name'))
        .toArray();

      expect(result).toEqual(['lop', 'ripple']);
    });
  });
});
