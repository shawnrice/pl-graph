import { Predicate, UnaryFn } from './types';

function internalEvery<T>(predicate: Predicate<T>, iterable: Iterable<T>): boolean {
  for (const iteration of iterable) {
    if (!predicate(iteration)) {
      return false;
    }
  }

  return true;
}

export function every<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>, boolean>;
export function every<T>(predicate: Predicate<T>, iterable: Iterable<T>): boolean;
export function every<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>, boolean> | boolean {
  return iterable ? internalEvery(predicate, iterable) : x0 => internalEvery(predicate, x0);
}
