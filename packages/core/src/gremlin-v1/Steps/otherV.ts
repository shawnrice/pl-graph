import type { UnaryFn } from '@pl-graph/fp/src';

import { Edge } from '../../core/Edge';
import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'flatmap';
const species = 'otherV';

export const otherV = (): UnaryFn<Traversal<any, any>> => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const edge = t.get() as Edge<any, any, any>;

      // This only works on vertices, so we'll skip any non-edge
      if (!Edge.isEdge(edge)) {
        // eslint-disable-next-line no-continue
        continue;
      }

      // eslint-disable-next-line @typescript-eslint/no-magic-numbers
      const [lastV, lastE] = t.pathMap.slice(-2).map(x => x?.get());

      if (!Vertex.isVertex(lastV) || !Edge.isEdge(lastE)) {
        // We didn't walk from a vertex to an edge, so this doesn't make sense
        // eslint-disable-next-line no-continue
        continue;
      }

      if (edge.to === lastV) {
        yield t.clone({ value: edge.from });
      } else if (edge.from === lastV) {
        yield t.clone({ value: edge.to });
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [], callback, genus, species });

  return execute(addStep, callback);
};
