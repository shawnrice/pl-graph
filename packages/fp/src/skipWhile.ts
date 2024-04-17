import { Predicate, UnaryFn } from './types';

function* internalSkipWhile<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T> {
  let found = false;

  for (const iteration of iterable) {
    found ||= !predicate(iteration);

    if (found) {
      yield iteration;
    }
  }
}

export function skipWhile<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>>;
export function skipWhile<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T>;
export function skipWhile<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable ? internalSkipWhile(predicate, iterable) : x0 => internalSkipWhile(predicate, x0);
}
