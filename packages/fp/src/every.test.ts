import { describe, expect, mock, test } from 'bun:test';

import { every } from './every';

describe('functional iterator tests', () => {
  test('every works', () => {
    const isFive = mock((x: number) => x === 5);
    expect(every(isFive, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])).toBe(false);
    expect(isFive).toHaveBeenCalledTimes(1);
  });

  test('every can return true', () => {
    const isFive = mock((x: number) => x === 5);
    expect(every(isFive, [5, 5, 5, 5, 5])).toBe(true);
    expect(isFive).toHaveBeenCalledTimes(5);
  });
});
