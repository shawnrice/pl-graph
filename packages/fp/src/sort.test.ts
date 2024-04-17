import { describe, expect, test } from 'bun:test';

import { sort } from './sort';

describe('functional iterator tests', () => {
  test('sort works', () => {
    const sorter = (a: number, b: number) => a - b;
    expect(sort(sorter, [1, 2, 3, 4, 5])).toEqual([1, 2, 3, 4, 5]);
  });

  test('sort works backwards', () => {
    const sorter = (a: number, b: number) => b - a;
    expect(sort(sorter, [1, 2, 3, 4, 5])).toEqual([5, 4, 3, 2, 1]);
  });

  test('it curries', () => {
    const sorter = (a: number, b: number) => b - a;
    const fn = sort(sorter);
    expect(fn([1, 2, 3, 4, 5])).toEqual([5, 4, 3, 2, 1]);
  });
});
