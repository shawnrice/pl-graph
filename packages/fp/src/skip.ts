import { UnaryFn } from './types';

function* internalSkip<T>(x: number, iterable: Iterable<T>): Iterable<T> {
  let count = 0;

  for (const iteration of iterable) {
    if (count < x) {
      count++;
    } else {
      yield iteration;
    }
  }
}

export function skip<T>(x: number): UnaryFn<Iterable<T>>;
export function skip<T>(x: number, iterable: Iterable<T>): Iterable<T>;
export function skip<T>(x: number, iterable?: Iterable<T>): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable ? internalSkip(x, iterable) : x0 => internalSkip(x, x0);
}
