export function* distinct<T>(iterable: Iterable<T>): Iterable<T> {
  const seen = new Set<T>();

  for (const iteration of iterable) {
    if (!seen.has(iteration)) {
      seen.add(iteration);
      yield iteration;
    }
  }
}
