import { describe, expect, mock, test } from 'bun:test';

/* eslint-disable max-lines-per-function */
/* eslint-disable no-magic-numbers */
import { List } from './List';

describe('LazyList tests', () => {
  test('It generates a list', () => {
    const list = List.from([1, 2, 3, 4, 5]);
    expect(list.filter(x => Boolean(x % 2)).toArray()).toEqual([1, 3, 5]);
  });

  test('it can turn into an array', () => {
    expect(Array.isArray(List.from([1]).toArray())).toBe(true);
  });

  test('map works', () => {
    expect(
      List.from([1, 2, 3, 4])
        .map(x => x + 1)
        .toArray(),
    ).toEqual([2, 3, 4, 5]);
  });

  test('take works', () => {
    expect(List.from([1, 2, 3, 4, 5]).take(3).toArray()).toEqual([1, 2, 3]);
  });

  test('filter works', () => {
    const isOdd = mock(x => Boolean(x % 2));
    const res = List.from([1, 2, 3, 4, 5, 6]).filter(isOdd).toArray();
    expect(res).toEqual([1, 3, 5]);
    expect(isOdd).toHaveBeenCalledTimes(6);
  });

  test('reject works', () => {
    const isOdd = mock(x => Boolean(x % 2));
    const res = List.from([1, 2, 3, 4, 5, 6]).reject(isOdd).toArray();
    expect(res).toEqual([2, 4, 6]);
    expect(isOdd).toHaveBeenCalledTimes(6);
  });

  test('some works', () => {
    const isOdd = mock(x => Boolean(x % 2));
    expect(List.from([1, 2, 3, 4, 5, 6]).some(isOdd)).toBe(true);
    expect(isOdd).toHaveBeenCalledTimes(1);
  });

  test('every works', () => {
    const isOdd = mock(x => Boolean(x % 2));
    expect(List.from([1, 2, 3, 4, 5, 6]).every(isOdd)).toBe(false);
    expect(isOdd).toHaveBeenCalledTimes(2);
  });

  test('it is lazy', () => {
    const isOdd = mock(x => Boolean(x % 2));

    const res = List.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).filter(isOdd).take(2).toArray();

    expect(res).toEqual([1, 3]);
    expect(isOdd).toHaveBeenCalledTimes(3);
  });

  test('skipWhile works', () => {
    const lengthIsLessThan3 = mock((x: string): boolean => x.length < 3);

    const list = List.from(['a', 'b', 'c', 'd', 'e', 'ab', 'def', 'fgx', 'a']);

    expect(list.skipWhile(lengthIsLessThan3).toArray()).toEqual(['def', 'fgx', 'a']);
    expect(lengthIsLessThan3).toHaveBeenCalledTimes(7);
  });

  test('it skips lazily', () => {
    const isOdd = mock(x => Boolean(x % 2));

    const res = List.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).skip(2).filter(isOdd).take(2).toArray();

    expect(res).toEqual([3, 5]);
    expect(isOdd).toHaveBeenCalledTimes(3);
  });

  test('it skips lazily 2', () => {
    const isOdd = mock(x => Boolean(x % 2));

    const res = List.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).filter(isOdd).skip(2).take(2).toArray();

    expect(res).toEqual([5, 7]);
    expect(isOdd).toHaveBeenCalledTimes(7);
  });

  test('after works', () => {
    const res = List.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
      .after(x => x >= 7)
      .toArray();
    expect(res).toEqual([8, 9, 10]);
  });

  test('after with equals works', () => {
    expect(
      List.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .after(x => x >= 7)
        .equals(List.of(8, 9, 10)),
    ).toBe(true);
  });

  test('before works', () => {
    const res = List.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
      .before(x => x > 7)
      .toArray();
    expect(res).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  test('before might return empty', () => {
    const res = List.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
      .before(x => x < 7)
      .toArray();
    expect(res).toEqual([]);
  });

  test('toList clones', () => {
    const res = List.of(1, 2, 3);
    const clone = res.toList();
    expect(res.equals(clone)).toBe(true);
    expect(res !== clone).toBe(true);
  });

  test('of works', () => {
    expect(List.of(1, 2, 3, 4, 5).toArray()).toEqual([1, 2, 3, 4, 5]);
  });

  test('takeWhile works', () => {
    expect(
      List.of(1, 2, 3, 4, 5)
        .takeWhile(x => x < 3)
        .toArray(),
    ).toEqual([1, 2]);
  });

  test('sort works', () => {
    expect(
      List.of(1, 3, 2)
        .sort((x, y) => x - y)
        .toArray(),
    ).toEqual([1, 2, 3]);
  });

  test('sideEffect works', () => {
    const noop = mock(() => {});
    const l = List.of(1, 2, 3, 4, 5).sideEffect(noop);
    expect(l.toArray()).toEqual([1, 2, 3, 4, 5]);
    expect(noop).toHaveBeenCalledTimes(5);
  });
});
