import { Predicate } from './types';

export function some<T>(predicate: Predicate<T>, iterable: Iterable<T>): boolean {
  for (const iteration of iterable) {
    if (predicate(iteration)) {
      return true;
    }
  }

  return false;
}
