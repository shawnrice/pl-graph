import { UnaryFn } from '../../../../fp/src/types';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

export function fold<S, E = S>(): UnaryFn<Traversal<S, E>>;
export function fold<S, E = S>(initial: E, bifunctor: (a: E, b: S) => E): UnaryFn<Traversal<S, E>>;
export function fold<S, E = S>(...args: any[]): UnaryFn<Traversal<S, E>> {
  return traversal => {
    const genus: StepGenus = 'map';
    const species = 'fold';

    // This is the barrier step
    // It resets runs all the steps and saves the value
    traversal.flush();
    const { graph, current: value } = traversal;

    const params = { args, genus, isBarrier: true, species };

    if (args.length === 0) {
      const callback = function* () {
        yield new Traverser({ value, graph });
      };

      return traversal.addStep({ ...params, callback });
    }

    if (args.length === 2) {
      const [initial, bifunction] = args;
      const next = value.reduce(bifunction, initial);

      const callback = function* () {
        yield new Traverser({ value: next, graph });
      };

      return traversal.addStep({ ...params, callback });
    }

    // What do we do if there is a mismatch?

    return traversal;
  };
}
