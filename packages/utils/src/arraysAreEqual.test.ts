import { describe, expect, test } from 'bun:test';

import { arraysAreEqual } from './arraysAreEqual';

describe('arraysAreEqual', () => {
  test('arrays of different lengths are not equal', () => {
    expect(arraysAreEqual([1], [1, 2])).toBe(false);
  });

  test('arrays can be equal', () => {
    expect(arraysAreEqual([1, 2, 3], [1, 2, 3])).toBe(true);
  });

  test('strings and numbers are not the same thing', () => {
    expect(arraysAreEqual([1, 2, 3], ['1', '2', '3'])).toBe(false);
  });
});
