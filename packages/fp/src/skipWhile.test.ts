import { describe, expect, mock, test } from 'bun:test';

import { skipWhile } from './skipWhile';

describe('functional iterator tests', () => {
  const arr = [1, 2, 3, 5, 6, 7];

  test('skipWhile works', () => {
    const isLessThan5 = mock((x: number) => x < 5);

    expect(Array.from(skipWhile(isLessThan5, arr))).toEqual([5, 6, 7]);
    expect(isLessThan5).toHaveBeenCalledTimes(4);
  });

  test('skipWhile is not 3', () => {
    const isNotThree = mock((x: number) => x !== 3);
    expect(Array.from(skipWhile(isNotThree, arr))).toEqual([3, 5, 6, 7]);
    expect(isNotThree).toHaveBeenCalledTimes(3);
  });

  test('skipWhile is not 4', () => {
    const isNotFour = mock((x: number) => x !== 4);
    expect(Array.from(skipWhile(isNotFour, arr))).toEqual([]);
    expect(isNotFour).toHaveBeenCalledTimes(6);
  });

  test('skipWhile isOdd', () => {
    const isOdd = mock((x: number) => !!(x % 2));
    expect(Array.from(skipWhile(isOdd, arr))).toEqual([2, 3, 5, 6, 7]);
    expect(isOdd).toHaveBeenCalledTimes(2);
  });
});
