import type { UnaryFn } from './types.js';

const internalSlice = function* <T>(
  start: number,
  end: number,
  iterable: Iterable<T>,
): Iterable<T> {
  let index = 0;

  for (const iteration of iterable) {
    if (start <= index && index < end) {
      yield iteration;
    }

    if (++index === end) {
      break;
    }
  }
};

export function slice<T>(start: number, end: number): UnaryFn<Iterable<T>>;
export function slice<T>(start: number, end: number, iterable: Iterable<T>): Iterable<T>;
export function slice<T>(
  start: number,
  end: number,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable === undefined
    ? (x0) => internalSlice(start, end, x0)
    : internalSlice(start, end, iterable);
}
