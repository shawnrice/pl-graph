import { Graph } from '../../core/Graph';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'sum';

/**
 * The sum()-step (map) operates on a stream of numbers and sums the numbers together to yield a
 * result. Note that the current traverser number is multiplied by the traverser bulk to determine
 * how many such numbers are being represented.
 */
export const sum: GremlinStep = () => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    let total: null | number = null;

    // This shouldn't be undefined, but we'll fix this later.
    // eslint-disable-next-line @typescript-eslint/init-declarations
    let graph;

    for (const t of prev) {
      graph ??= t.graph;

      const value = t.get();

      if (typeof value === 'number') {
        if (typeof total === 'number') {
          total += value;
        } else {
          total = value;
        }
      }
    }

    yield new Traverser({ value: total, graph: graph as Graph<any, any> });
  };

  const addStep = (traversal: Traversal) => {
    return traversal.addStep({ args: [], callback, genus, isBarrier: true, species });
  };

  return execute(addStep, callback);
};
