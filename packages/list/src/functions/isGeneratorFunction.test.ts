import { describe, expect, test } from 'bun:test';

import { isGeneratorFunction } from './isGeneratorFunction';

type MapFn<T, R> = (value: T, index: number) => R;
function* internalMap<T, R = T>(mapper: MapFn<T, R>, x0: Iterable<T>): Iterable<R> {
  let index = 0;

  for (const iteration of x0) {
    yield mapper(iteration, index++);
  }
}

describe('isGeneratorFunction', () => {
  test('isGeneratorFunction returns true for generator functions', () => {
    expect(isGeneratorFunction(internalMap)).toBe(true);
  });

  test('isGeneratorFunction returns false for non-generator functions', () => {
    expect(isGeneratorFunction(Array.isArray)).toBe(false);
    expect(isGeneratorFunction(isGeneratorFunction)).toBe(false);
  });
});
