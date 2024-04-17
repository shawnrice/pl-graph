import { UnaryFn } from '../../../../fp/src/types';
import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep } from './types';

const genus = 'branch'; // ???
const species = 'coalesce';

/**
 * The coalesce()-step evaluates the provided traversals in order and returns the first traversal
 * that emits at least one element
 */
export const coalesce: GremlinStep<UnaryFn<Traversal>[]> = (
  ...traversals: UnaryFn<Traversal>[]
) => {
  const callback = function* callback(prev: Iterable<Traverser<Vertex>>) {
    // We need to find a reference to the graph...
    yield* prev;
  };

  const addStep = (t: Traversal) => {
    const callback = function* callback(prev: Iterable<Traverser<Vertex>>) {
      // We need to find a reference to the graph...
      for (const traversal of traversals) {
        const generator = traversal(new Traversal({ graph: t.graph, value: prev })).run();
        if (!generator.next().done) {
          yield* traversal(new Traversal({ graph: t.graph, value: prev })).run();
          break;
        }
      }
    };

    return t.addStep({
      args: traversals,
      callback,
      genus,
      species,
    });
  };

  return execute(addStep, callback);
};
