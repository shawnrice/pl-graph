import { BinaryFn } from './types';

const defaultComparator = <T>(a: T, b: T): boolean => a === b;

const MAX_ITERATIONS = 1_000_000;

/**
 * This will only compare to 1m iterations. After that, it will fail
 */
export function equals<T>(
  x: Iterable<T>,
  y: Iterable<T>,
  comparator?: BinaryFn<T, T, boolean>,
): boolean {
  const compare = comparator ?? defaultComparator;

  const y0 = y[Symbol.iterator]();

  let count = 0;

  for (const x1 of x) {
    const y1 = y0.next();

    if (y1.done) {
      return false;
    }

    if (!compare(x1, y1.value)) {
      return false;
    }

    if (++count > MAX_ITERATIONS) {
      // Avoid accidental DDOS from infinite iterators
      return false;
    }
  }

  return y0.next().done ?? false;
}
