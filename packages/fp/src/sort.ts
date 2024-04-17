import { SortFn, UnaryFn } from './types';

/**
 * This is not lazy
 */
const internalSort = <T>(sorter: SortFn<T>, iterable: Iterable<T>): Iterable<T> =>
  [...iterable].sort(sorter);

export function sort<T>(sorter: SortFn<T>): UnaryFn<Iterable<T>>;
export function sort<T>(sorter: SortFn<T>, iterable: Iterable<T>): Iterable<T>;
export function sort<T>(
  sorter: SortFn<T>,
  iterable?: Iterable<T>,
): UnaryFn<Iterable<T>> | Iterable<T> {
  return iterable ? internalSort(sorter, iterable) : x0 => internalSort(sorter, x0);
}
