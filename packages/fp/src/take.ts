const internalTake = function* <T>(x: number, iterable: Iterable<T>): Iterable<T> {
  let count = 0;

  for (const iteration of iterable) {
    if (count < x) {
      yield iteration;
    }

    if (++count >= x) {
      break;
    }
  }
};

export function take(x: number): <T>(iterable: Iterable<T>) => Iterable<T>;
export function take<T>(x: number, iterable: Iterable<T>): Iterable<T>;
export function take<T>(x: number, iterable?: Iterable<T>) {
  return iterable === undefined
    ? <U>(x0: Iterable<U>) => internalTake(x, x0)
    : internalTake(x, iterable);
}
