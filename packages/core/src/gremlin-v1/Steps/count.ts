import { Graph } from '../../core/Graph';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'count';

/**
 * Count the number of elements in the traversal
 * There's an unhandled edge case here where if the traversal is empty, any new
 * traversals won't have a graph, so any subsequent "Vertex steps", along with
 * V() and E() will fail.
 */
export const count: GremlinStep = () => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    let counter = 0;

    // This shouldn't be undefined, but we'll fix this later.
    // eslint-disable-next-line @typescript-eslint/init-declarations
    let graph;

    for (const t of prev) {
      counter += 1;
      graph ??= t.graph;
    }

    yield new Traverser({ value: counter, graph: graph as Graph<any, any> });
  };

  const addStep = (traversal: Traversal) => {
    return traversal.addStep({ args: [], callback, genus, isBarrier: true, species });
  };

  return execute(addStep, callback);
};
