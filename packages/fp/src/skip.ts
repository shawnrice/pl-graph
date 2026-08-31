const internalSkip = function* <T>(x: number, iterable: Iterable<T>): Iterable<T> {
  let count = 0;

  for (const iteration of iterable) {
    if (count < x) {
      count++;
    } else {
      yield iteration;
    }
  }
};

export function skip(x: number): <T>(iterable: Iterable<T>) => Iterable<T>;
export function skip<T>(x: number, iterable: Iterable<T>): Iterable<T>;
export function skip<T>(x: number, iterable?: Iterable<T>) {
  return iterable === undefined
    ? <U>(x0: Iterable<U>) => internalSkip(x, x0)
    : internalSkip(x, iterable);
}
