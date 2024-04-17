import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'max';

/**
 * Determines the largest value in the stream
 *
 * __HERE BE DRAGONS__: We have not yet implemented scope
 * __HERE BE DRAGONS__: This only works with numbers right now
 *
 * @todo expand to strings
 */
export const max: GremlinStep = () => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    let current: number;

    // This shouldn't be undefined, but we'll fix this later.
    // eslint-disable-next-line @typescript-eslint/init-declarations
    let graph;

    for (const t of prev) {
      graph ??= t.graph;

      const value = t.get();

      current ??= value;

      if (value !== void 0 && value !== null && current < value) {
        current = value;
      }
    }

    yield new Traverser({ value: current, graph });
  };

  const addStep = (traversal: Traversal) => {
    return traversal.addStep({ args: [], callback, genus, isBarrier: true, species });
  };

  return execute(addStep, callback);
};
