export function* range(start: number, end?: number): Iterable<number> {
  if (typeof end === 'undefined') {
    for (let i = 0; i <= start; i++) {
      yield i;
    }
    return;
  }

  if (start === end) {
    yield start;
    return;
  }

  if (start > end) {
    for (let i = start; i >= end; i--) {
      yield i;
    }
    return;
  }

  for (let i = start; i <= end; i++) {
    yield i;
  }
}
