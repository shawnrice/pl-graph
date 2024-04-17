import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

const genus: StepGenus = 'flatmap';
const species = 'unfold';

/**
 * Unfold step
 *
 * Unrolls iterators, iterables, or any "map"
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#unfold-step
 */
export const unfold =
  <S, E = S>(): UnaryFn<Traversal<S, E>> =>
  traversal => {
    const callback = function* (prev: Iterable<Traverser<any>>) {
      for (const t of prev) {
        const value = t.get();

        // if this is an iterator, iterable, or "map," then unroll it into a linear form
        if (typeof value === 'string') {
          // strings are iterable, but we don't want to iterate on them
          yield t;
        } else if (value && typeof value[Symbol.iterator] === 'function') {
          for (const v of value) {
            yield t.clone({ value: v });
          }
        } else {
          yield t;
        }
      }
    };

    return traversal.addStep({ args: [], callback, genus, species });
  };
