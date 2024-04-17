type IterableWithLength<T> = Iterable<T> & { length: number };
type IterableWithSize<T> = Iterable<T> & { size: number };

/**
 * Counts an iterable
 *
 * Will short-circuit at 1 million
 */
export function count<T>(iterable: Iterable<T>): number {
  if (
    'length' in iterable &&
    typeof (iterable as Iterable<T> & { length: unknown }).length === 'number'
  ) {
    return (iterable as IterableWithLength<T>).length;
  }

  if (
    'size' in iterable &&
    typeof (iterable as Iterable<T> & { size: unknown }).size === 'number'
  ) {
    return (iterable as IterableWithSize<T>).size;
  }

  let i = 0;

  for (const _ of iterable) {
    i++; // eslint-disable-line no-plusplus

    // eslint-disable-next-line yoda, @typescript-eslint/no-magic-numbers
    if (i >= 1_000_000) {
      return i;
    }
  }

  return i;
}
