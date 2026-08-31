import type { UnaryFn } from './types.js';

const internalTakeLast = function* <T>(count: number, iterable: Iterable<T>): Iterable<T> {
  if (count <= 0) {
    return;
  }

  const buf: T[] = [];

  for (const item of iterable) {
    buf.push(item);

    if (buf.length > count) {
      buf.shift();
    }
  }

  yield* buf;
};

/**
 * Yields the last N elements from an iterable. TERMINAL-ish (buffers N items).
 *
 * @example:
 * ```ts
 *   Array.from(takeLast(2, [1,2,3,4])); // [3,4]
 *   const tl = takeLast<number>(2);
 *   Array.from(tl([1,2,3,4])); // [3,4]
 * ```
 */
export function takeLast<T>(count: number): UnaryFn<Iterable<T>>;
export function takeLast<T>(count: number, iterable: Iterable<T>): Iterable<T>;
export function takeLast<T>(
  count: number,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable === undefined
    ? (x0) => internalTakeLast(count, x0)
    : internalTakeLast(count, iterable);
}
