import { List } from '../List';
import { empty } from './empty';

describe('List functional tests', () => {
  test('empty works', () => {
    const list = empty();
    expect(List.isList(list)).toBe(true);
    expect(list).toHaveLength(0);
  });
});
