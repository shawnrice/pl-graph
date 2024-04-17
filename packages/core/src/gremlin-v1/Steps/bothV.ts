import { UnaryFn } from '../../../../fp/src/types';
import { Edge } from '../../core/Edge';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'flatmap';
const species = 'bothV';

export const bothV = (): UnaryFn<Traversal<any, any>> => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const edge = t.get() as Edge<any, any, any>;

      // This only works on vertices, so we'll skip any non-edge
      if (!Edge.isEdge(edge)) {
        // eslint-disable-next-line no-continue
        continue;
      }

      yield t.clone({ value: edge.from });
      yield t.clone({ value: edge.to });
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [], callback, genus, species });

  return execute(addStep, callback);
};
