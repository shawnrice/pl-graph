import type { UnaryFn } from './types.js';

const internalFirst = <T>(iterable: Iterable<T>): T | undefined => {
  for (const item of iterable) {
    return item;
  }

  return undefined;
};

/**
 * Returns the first element of an iterable, or undefined if empty. TERMINAL.
 *
 * @example
 * ```ts
 *   const xs = [1, 2, 3];
 *   first(xs); // 1
 *   const f = first<number>();
 *   f(xs); // 1
 * ```
 */
export function first<T>(): UnaryFn<Iterable<T>, T | undefined>;
export function first<T>(iterable: Iterable<T>): T | undefined;
export function first<T>(
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>, T | undefined> | T | undefined {
  return iterable === undefined ? internalFirst : internalFirst(iterable);
}
