import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'sideEffect';
const species = 'sideEffect';

/**
 * The sideEffect() step performs some operation on the traverser and passes it to the next step in the process.
 */
export const sideEffect: GremlinStep<UnaryFn<Traverser<any>, any>> = (
  effect: UnaryFn<Traverser<any>, any>,
) => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      effect(t);
      yield t;
    }
  };

  const addStep = (traversal: Traversal) => {
    return traversal.addStep({ args: [], callback, genus, isBarrier: false, species });
  };

  return execute(addStep, callback);
};
