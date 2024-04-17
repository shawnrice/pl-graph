import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep } from './types';

const genus = 'branch';
const species = 'optional';

/**
 * The not()-step (filter) removes objects from the traversal stream when the traversal provided as
 *  an argument returns an object.
 *
 * __FILTER__
 */
export const optional: GremlinStep<UnaryFn<Traversal>> = step => {
  const callback = function* callback(prev: Iterable<Traverser<any>>) {
    for (const traverser of prev) {
      const trav = step(new Traversal({ graph: traverser.graph, value: [traverser] }));

      const iterator = trav.run().next();
      if (iterator.value !== void 0 && iterator.done !== true) {
        yield* trav.run();
      } else {
        yield traverser;
      }
    }
  };

  const addStep = (t: Traversal) =>
    t.addStep({
      args: [step],
      callback,
      genus,
      species,
    });

  return execute(addStep, callback);
};
