import { Predicate, UnaryFn } from './types';

function* internalAfter<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T> {
  let found = false;

  for (const iteration of iterable) {
    if (found) {
      yield iteration;
    } else {
      found ||= predicate(iteration);
    }
  }
}

export function after<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>>;
export function after<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T>;
export function after<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable ? internalAfter(predicate, iterable) : x0 => internalAfter(predicate, x0);
}
