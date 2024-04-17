import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import {
  in_,
  out,
} from './';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('union tests', () => {
    // Do some tests here

    test('it works with subtraversals', () => {
      const result = tinkerGraph
        .traverse(g => g.V('4').union(in_(), out()).values('age', 'lang'))
        .toArray();
      expect(result).toEqual([29, 'java', 'java']);
    });

    // TODO we cannot chain in_().values('age') because `in_()` just returns a function
    test.skip('it works with subtraversals', () => {
      const result = tinkerGraph
        .traverse(g => g.V('4').union(in_().values('age'), out().values('lang')))
        .toArray();
      expect(result).toEqual([29, 'java', 'java']);
    });
  });
});
