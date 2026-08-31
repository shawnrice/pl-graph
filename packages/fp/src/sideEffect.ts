import type { UnaryFn } from './types.js';

const internalSideEffect = function* <T>(
  effect: UnaryFn<T, unknown>,
  iterable: Iterable<T>,
): Iterable<T> {
  for (const iteration of iterable) {
    effect(iteration);

    yield iteration;
  }
};

export function sideEffect<T>(effect: UnaryFn<T, unknown>): UnaryFn<Iterable<T>>;
export function sideEffect<T>(effect: UnaryFn<T, unknown>, iterable: Iterable<T>): Iterable<T>;
export function sideEffect<T>(
  effect: UnaryFn<T, unknown>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable === undefined
    ? (x0) => internalSideEffect(effect, x0)
    : internalSideEffect(effect, iterable);
}
