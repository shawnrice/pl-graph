import type { Predicate, UnaryFn } from './types.js';

const internalAfter = function* <T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T> {
  let found = false;

  for (const iteration of iterable) {
    if (found) {
      yield iteration;
    } else {
      found ||= predicate(iteration);
    }
  }
};

export function after<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>>;
export function after<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T>;
export function after<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable === undefined
    ? (x0) => internalAfter(predicate, x0)
    : internalAfter(predicate, iterable);
}
