import { describe, expect, mock, test } from 'bun:test';

import { map } from './map';
import { pipe } from './pipe';
import { take } from './take';

describe('functional iterator tests', () => {
  test('map works', () => {
    const isOdd = mock((x: number): boolean => !!(x % 2));
    expect(Array.from(map(isOdd, [1, 2, 3, 4]))).toEqual([true, false, true, false]);
    expect(isOdd).toHaveBeenCalledTimes(4);
  });

  test('map is curryable', () => {
    const isOdd = mock((x: number): boolean => !!(x % 2));
    const fn = map(isOdd);
    expect(Array.from(fn([1, 2, 3, 4]))).toEqual([true, false, true, false]);
    expect(isOdd).toHaveBeenCalledTimes(4);
  });

  test('curryable map is chainable', () => {
    const isOdd = mock((x: number): boolean => !!(x % 2));
    const doMyThing = pipe(map(isOdd), take(2));

    expect(Array.from(doMyThing([1, 2, 3, 4]))).toEqual([true, false]);
    expect(isOdd).toHaveBeenCalledTimes(2);
  });
});
