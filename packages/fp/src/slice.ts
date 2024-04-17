export function* slice<T>(start: number, end: number, iterable: Iterable<T>): Iterable<T> {
  let index = 0;

  for (const iteration of iterable) {
    if (start <= index && index < end) {
      yield iteration;
    }

    if (++index === end) {
      break;
    }
  }
}
