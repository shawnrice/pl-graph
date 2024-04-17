import type { UnaryFn } from '@pl-graph/fp/src';

import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep } from './types';

const genus = 'filter';
const species = 'or';

/**
 * Ensures that at least one of the provided traversals yield a result.
 *
 * __FILTER__
 *
 * Note: this does not support _infix_ notation. So, `outE('KNOWS').or().InE('HELPS')` will fail
 */
export const or: GremlinStep<UnaryFn<Traversal>[]> = (...traversals: UnaryFn<Traversal>[]) => {
  const callback = function* callback(prev: Iterable<Traverser<Vertex>>) {
    for (const traverser of prev) {
      if (
        traversals.some(
          createTraversal =>
            createTraversal(new Traversal({ graph: traverser.graph, value: [traverser] })).toArray()
              .length !== 0,
        )
      ) {
        yield traverser.clone();
      }
    }
  };

  const addStep = (t: Traversal) =>
    t.addStep({
      args: traversals,
      callback,
      genus,
      species,
    });

  return execute(addStep, callback);
};
