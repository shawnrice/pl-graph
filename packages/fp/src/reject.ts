import { filter } from './filter.js';
import type { Predicate, UnaryFn } from './types.js';

export function reject<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>>;
export function reject<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T>;
export function reject<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  const fn = (x: T) => !predicate(x);

  return iterable === undefined ? (x0) => filter(fn, x0) : filter(fn, iterable);
}
