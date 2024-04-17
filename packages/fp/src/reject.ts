import { filter } from './filter';
import { Predicate, UnaryFn } from './types';

export function reject<T>(predicate: Predicate<T>): UnaryFn<Iterable<T>>;
export function reject<T>(predicate: Predicate<T>, iterable: Iterable<T>): Iterable<T>;
export function reject<T>(
  predicate: Predicate<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  const fn = (x: T) => !predicate(x);
  return iterable ? filter(fn, iterable) : x0 => filter(fn, x0);
}
