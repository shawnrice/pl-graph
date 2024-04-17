import { List } from '../List';
import { of } from './of';

describe('List functional tests', () => {
  test('of works', () => {
    const list = of(1, 2, 3, 4, 5);
    expect(List.isList(list)).toBe(true);
    expect(list.toArray()).toEqual([1, 2, 3, 4, 5]);
    expect(list).toHaveLength(5);
  });
});
