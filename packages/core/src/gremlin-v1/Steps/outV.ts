import { Edge } from '../../core/Edge';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus, TraverserFunction } from './types';

const genus: StepGenus = 'flatmap';
const species = 'outV';

export const outV: GremlinStep = () => {
  const callback: TraverserFunction = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const edge = t.get() as Edge<any, any, any>;

      // This only works on vertices, so we'll skip any non-edge
      if (Edge.isEdge(edge)) {
        yield t.clone({ value: edge.from });
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [], callback, genus, species });

  return execute(addStep, callback);
};
