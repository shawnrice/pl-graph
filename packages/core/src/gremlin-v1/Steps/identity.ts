import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

export const identity = (): UnaryFn<Traversal<any, any>> => {
  return traversal => {
    const genus: StepGenus = 'map';
    const species = 'identity';

    const callback = function* (prev: Iterable<Traverser<any>>) {
      for (const t of prev) {
        yield t.clone();
      }
    };

    return traversal.addStep({ args: [], callback, genus, species });
  };
};
