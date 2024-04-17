import { describe, expect, mock, test } from 'bun:test';

import { sideEffect } from './sideEffect';
import { slice } from './slice';

function* range(start: number, stop: number) {
  let x = start;
  while (x <= stop) {
    // eslint-disable-next-line no-plusplus
    yield x++;
  }
}

describe('functional iterator tests', () => {
  test('slice works', () => {
    const start = 2;
    const end = 6;
    expect(Array.from(slice(start, end, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))).toEqual([3, 4, 5, 6]);
  });

  test('slice works efficiently', () => {
    const beforeEffect = mock(() => {});
    const afterEffect = mock(() => {});
    const start = 2;
    const end = 6;

    const res = Array.from(
      sideEffect(afterEffect, slice(start, end, sideEffect(beforeEffect, range(1, 10)))),
    );
    expect(res).toEqual([3, 4, 5, 6]);
    expect(beforeEffect).toHaveBeenCalledTimes(6);
    expect(afterEffect).toHaveBeenCalledTimes(4);
  });
});
