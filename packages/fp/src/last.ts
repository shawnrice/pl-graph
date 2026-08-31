import { type Boundary, boundary } from './boundary.js';
import type { UnaryFn } from './types.js';

const internalLast = boundary(<T>(iterable: Iterable<T>): T | undefined => {
  let seen: T | undefined;

  for (const item of iterable) {
    seen = item;
  }

  return seen;
});

/**
 * Returns the last element of an iterable, or undefined if empty. Materializes
 * the iterable — pair with `take` (or wrap with `bounded`) if the source may
 * be infinite.
 */
export function last<T>(): Boundary<UnaryFn<Iterable<T>, T | undefined>>;
export function last<T>(iterable: Iterable<T>): T | undefined;
export function last<T>(iterable?: Iterable<T>) {
  return iterable === undefined ? internalLast : internalLast(iterable);
}
