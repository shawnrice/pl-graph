import { Traverser } from '../Traverser';

/**
 * Essentially, an iterable no-op
 */
export function* passThrough(prev: Iterable<Traverser<any>>): Iterable<Traverser<any>> {
  yield* prev;
}
