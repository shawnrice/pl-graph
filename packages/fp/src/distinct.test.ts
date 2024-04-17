import { describe, expect, mock, test } from 'bun:test';

import { distinct } from './distinct';
import { filter } from './filter';
import { take } from './take';

describe('functional iterator tests', () => {
  test('distinct works', () => {
    expect(Array.from(distinct([1, 2, 3, 3, 3, 4, 3, 2, 1]))).toEqual([1, 2, 3, 4]);
  });

  test('it can be lazy', () => {
    const arr = [1, 2, 3, 3, 4, 3, 2, 2, 1, 2, 3, 3, 6, 7, 8];
    const isOdd = mock((x: number) => !!(x % 2));
    expect(Array.from(take(3, filter(isOdd, distinct(arr))))).toEqual([1, 3, 7]);
    expect(isOdd).toHaveBeenCalledTimes(6);

    const isEven = mock((x: number) => !(x % 2));
    expect(Array.from(take(2, distinct(filter(isEven, arr))))).toEqual([2, 4]);
    expect(isEven).toHaveBeenCalledTimes(5);
  });
});
