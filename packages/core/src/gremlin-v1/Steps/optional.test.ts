import { createTestTinkerGraph } from '../../fixtures/createTestTinkerGraph';
import {
  in_,
  out,
} from './';

describe('Gremlin tests', () => {
  const tinkerGraph = createTestTinkerGraph();

  describe('STEP optional tests', () => {
    test('optional can return the original', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V('2').optional(out('KNOWS')).values('name'))
        .toArray();

      expect(result).toEqual(['vadas']);
    });

    test('optional can return the newly found thing', () => {
      const result = tinkerGraph
        .traverse<string>(g => g.V('2').optional(in_('KNOWS')).values('name'))
        .toArray();

      expect(result).toEqual(['marko']);
    });
  });
});
