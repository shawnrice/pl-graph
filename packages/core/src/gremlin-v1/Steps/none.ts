import type { UnaryFn } from '@pl-graph/fp/src';

import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep } from './types';

const genus = 'filter';
const species = 'none';

/**
 * Filter all traversers in the traversal.
 *
 * This step has narrow use cases and is primarily intended for use as a signal to remote servers
 * that iterate() was called. While it may be directly used, it is often a sign that a traversal
 * should be re-written in another form.
 *
 * __FILTER__
 *
 * __HERE BE DRAGONS__: Since we cannot pass on any traversers, if anything comes after this,
 * then we'll lose reference to the graph
 */
export const none: GremlinStep<UnaryFn<Traversal>[]> = (...traversals) => {
  const callback = function* callback(prev: Iterable<Traverser<Vertex>>) {
    yield* [];
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
