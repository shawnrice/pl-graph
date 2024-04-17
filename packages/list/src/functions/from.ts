import { List } from '../List';
import { isGeneratorFunction } from './isGeneratorFunction';

export function from<T>(iterable: Iterable<T>): List<T> {
  if (isGeneratorFunction<T>(iterable)) {
    return new List<T>(iterable); // we're not actually testing for <T> here
  }

  const innerFrom = function* () {
    yield* iterable;
  };

  if ('length' in iterable && typeof iterable.length === 'number') {
    return new List<T>(innerFrom, iterable.length);
  }

  if ('size' in iterable && typeof iterable.size === 'number') {
    return new List<T>(innerFrom, iterable.size);
  }

  return new List<T>(innerFrom);
}
