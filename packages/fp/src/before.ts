import { Predicate, UnaryFn } from './types';

function* internalBefore<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T> {
  let found = false;

  for (const iteration of iterable) {
    found ||= predicate(iteration);

    if (!found) {
      yield iteration;
    }
  }
}

export function before<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>>;
export function before<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T>;
export function before<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable ? internalBefore(predicate, iterable) : x0 => internalBefore(predicate, x0);
}
