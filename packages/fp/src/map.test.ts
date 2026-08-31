import { describe, expect, mock, test } from 'bun:test';

import { map } from './map.js';
import { pipe } from './pipe.js';
import { range } from './range.js';
import { skip } from './skip.js';
import { take } from './take.js';

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

  test('map over range with skip also works', () => {
    const val = pipe<Iterable<number>>()(
      skip(100),
      map((x) => x * 2),
      take(5),
    )(range());

    const x1 = Array.from(val);

    expect(x1).toEqual([200, 202, 204, 206, 208]);
  });

  test('the empty string is data-first input, not a missing iterable', () => {
    // `''` is a valid (empty) indexed iterable but falsy — the dispatch must key on
    // `iterable === undefined`, not truthiness, or `map(fn, '')` returns the CURRIED
    // function instead of an empty result.
    expect([...map((c: string) => c.toUpperCase(), '')]).toEqual([]);
    expect([...map((c: string) => c.toUpperCase(), 'ab')]).toEqual(['A', 'B']);
    // the curried (data-last) form still returns a function
    expect(typeof map((c: string) => c)).toBe('function');
  });
});
