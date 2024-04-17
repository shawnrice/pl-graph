import { describe, expect, mock, test } from 'bun:test';

import { some } from './some';

describe('functional iterator tests', () => {
  test('some works', () => {
    const isFive = mock((x: number) => x === 5);
    expect(some(isFive, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).toBe(true);
    expect(isFive).toHaveBeenCalledTimes(5);
  });

  test('some can return true', () => {
    const isFive = mock((x: number) => x === 5);
    expect(some(isFive, [5, 5, 5, 5, 5])).toBe(true);
    expect(isFive).toHaveBeenCalledTimes(1);
  });

  test('some can return false', () => {
    const isFour = mock((x: number) => x === 4);
    expect(some(isFour, [5, 5, 5, 5, 5])).toBe(false);
    expect(isFour).toHaveBeenCalledTimes(5);
  });
});
