/* eslint-disable no-plusplus, no-restricted-syntax, functional/no-let, func-style  */
import { MapFn, UnaryFn } from './types';

function* internalMap<T, R = T>(mapper: MapFn<T, R>, x0: Iterable<T>): Iterable<R> {
  let index = 0;

  for (const iteration of x0) {
    yield mapper(iteration, index++);
  }
}

export function map<T, R = T>(mapper: MapFn<T, R>): UnaryFn<Iterable<T>, Iterable<R>>;
export function map<T, R = T>(mapper: MapFn<T, R>, iterable: Iterable<T>): Iterable<R>;
export function map<T, R = T>(
  mapper: MapFn<T, R>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>, Iterable<R>> | Iterable<R> {
  return iterable ? internalMap(mapper, iterable) : (x0: Iterable<T>) => internalMap(mapper, x0);
}
