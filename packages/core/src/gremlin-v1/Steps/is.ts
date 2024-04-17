import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

/**
 * Helps with filtering
 */
export const is = (x0: any): UnaryFn<Traversal<any, boolean>> => {
  return traversal => {
    const genus: StepGenus = 'map';
    const species = 'is';

    const callback = function* (prev: Iterable<Traverser<any>>): Generator<Traverser<any>, void> {
      for (const t of prev) {
        const p = t.get();

        switch (typeof p) {
          case 'string':
          case 'number':
            if (p === x0) {
              yield t.clone();
            }
            break;
          default:
          // noop
        }
      }
    };

    return traversal.addStep({ args: x0, callback, genus, species });
  };
};
