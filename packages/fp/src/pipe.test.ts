import { describe, expect, mock, test } from 'bun:test';

import { distinct } from './distinct';
import { map } from './map';
import { pipe } from './pipe';
import { take } from './take';

describe('functional iterator tests', () => {
  test('we can pipe', () => {
    const isOdd = mock((x: number): boolean => !!(x % 2));

    const fn = pipe(map(isOdd), distinct, take(3), Array.from);

    const iterable = Array.from({ length: 100 }).map((_, i) => i + 1);

    expect(fn(iterable)).toEqual([true, false]);
    expect(isOdd).toHaveBeenCalledTimes(100);
  });

  test('we can pipe a lot', () => {
    const plusplus = mock(function* plusplus(x0: Iterable<number>) {
      for (const x of x0) {
        yield x + 1;
      }
    });
    const toString = mock(function* tostring(x0: Iterable<number>) {
      for (const x of x0) {
        yield `${x}`;
      }
    });

    const fn = pipe(
      plusplus,
      plusplus,
      plusplus,
      plusplus,
      plusplus,
      plusplus,
      plusplus,
      plusplus,
      toString,
      take(5),
    );

    const iterable = Array.from({ length: 100 }).map((_, i) => i + 1);

    expect(Array.from(fn(iterable))).toEqual(['9', '10', '11', '12', '13']);
  });
});
