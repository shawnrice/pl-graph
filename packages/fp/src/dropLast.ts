import type { UnaryFn } from './types.js';

/**
 * Yields all but the last N elements from an iterable. Lazy with O(N) buffer.
 *
 * Example:
 *   Array.from(dropLast(2, [1,2,3,4])); // [1,2]
 *   const dl = dropLast<number>(2);
 *   Array.from(dl([1,2,3,4])); // [1,2]
 */
const internalDropLast = function* <T>(count: number, iterable: Iterable<T>): Iterable<T> {
  if (count <= 0) {
    yield* iterable;

    return;
  }

  const buf: T[] = [];

  for (const item of iterable) {
    buf.push(item);

    if (buf.length > count) {
      yield buf.shift() as T;
    }
  }
};

export function dropLast<T>(count: number): UnaryFn<Iterable<T>>;
export function dropLast<T>(count: number, iterable: Iterable<T>): Iterable<T>;
export function dropLast<T>(
  count: number,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable === undefined
    ? (x0) => internalDropLast(count, x0)
    : internalDropLast(count, iterable);
}
