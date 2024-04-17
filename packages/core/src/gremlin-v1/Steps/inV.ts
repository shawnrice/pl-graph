import type { UnaryFn } from '@pl-graph/fp/src';

import { Edge } from '../../core/Edge';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'flatmap';
const species = 'inV';

export const inV = (): UnaryFn<Traversal<any, any>> => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const edge = t.get() as Edge<any, any, any>;

      // This only works on edges, so we'll skip any non-edge
      if (Edge.isEdge(edge)) {
        yield t.clone({ value: edge.to });
      }
    }
  };
  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [], callback, genus, species });

  return execute(addStep, callback);
};
