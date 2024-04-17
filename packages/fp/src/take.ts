import { UnaryFn } from './types';

export function take<T>(x: number): UnaryFn<Iterable<T>>;
export function take<T>(x: number, iterable: Iterable<T>): Iterable<T>;
export function take<T>(x: number, iterable?: Iterable<T>): UnaryFn<Iterable<T>> | Iterable<T> {
  function* internalTake(x0: Iterable<T>) {
    let count = 0;

    for (const iteration of x0) {
      if (count < x) {
        yield iteration;
      }

      if (++count >= x) {
        break;
      }
    }
  }

  return iterable ? internalTake(iterable) : internalTake;
}
