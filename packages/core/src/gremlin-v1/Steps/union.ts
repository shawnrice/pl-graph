import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep } from './types';

const genus = 'branch';
const species = 'union';

/**
 * The union()-step (branch) supports the merging of the results of an arbitrary number of traversals.
 * When a traverser reaches a union()-step, it is copied to each of its internal steps. The traversers
 * emitted from union() are the outputs of the respective internal traversals.
 */
export const union: GremlinStep<UnaryFn<Traversal>[]> = (...traversals) => {
  const callback = function* callback(prev: Iterable<Traverser<any>>) {
    for (const traversal of traversals) {
      yield* traversal(prev);
    }
  };

  const addStep = (t: Traversal) => {
    function* callbackWithTraversal(prev: Iterable<Traverser<any>>) {
      for (const traversal of traversals) {
        yield* traversal(new Traversal({ graph: t.graph, value: prev ?? [] })).run();
      }
    }

    return t.addStep({
      args: traversals,
      callback: callbackWithTraversal,
      genus,
      species,
    });
  };

  return execute(addStep, callback);
};
