import { Predicate, UnaryFn } from './types';

function* internalTakeWhile<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T> {
  for (const iteration of iterable) {
    if (!predicate(iteration)) {
      break;
    }

    yield iteration;
  }
}

export function takeWhile<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>>;
export function takeWhile<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T>;
export function takeWhile<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable ? internalTakeWhile(predicate, iterable) : x0 => internalTakeWhile(predicate, x0);
}
