import type { UnaryFn } from './types.js';

const internalFlatten = function* <T>(iterable: Iterable<Iterable<T>>): Iterable<T> {
  for (const inner of iterable) {
    yield* inner;
  }
};

export function flatten<T>(): UnaryFn<Iterable<Iterable<T>>, Iterable<T>>;
export function flatten<T>(iterable: Iterable<Iterable<T>>): Iterable<T>;
export function flatten<T>(
  iterable?: Iterable<Iterable<T>>,
): UnaryFn<Iterable<Iterable<T>>, Iterable<T>> | Iterable<T> {
  return iterable === undefined ? (x0) => internalFlatten(x0) : internalFlatten(iterable);
}
