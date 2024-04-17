import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'mean';

/**
 * The mean()-step (map) operates on a stream of numbers and determines the average of those numbers.
 *
 * __HERE BE DRAGONS__: We have not yet implemented scope
 * __HERE BE DRAGONS__: This only works with numbers right now
 */
export const mean: GremlinStep = () => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    let total: null | number = null;
    let count = 0;

    // This shouldn't be undefined, but we'll fix this later.
    // eslint-disable-next-line @typescript-eslint/init-declarations
    let graph;

    for (const t of prev) {
      graph ??= t.graph;

      const value = t.get();

      if (typeof value === 'number') {
        if (typeof total === 'number') {
          total += value;
          count++;
        } else {
          total = value;
          count++;
        }
      }
    }

    if (total !== null && count > 0) {
      yield new Traverser({ value: total / count, graph });
    } else {
      yield new Traverser({ value: total, graph });
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [], callback, genus, isBarrier: true, species });

  return execute(addStep, callback);
};
