import { Vertex } from '../../core/Vertex';
/* eslint-disable no-underscore-dangle */
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'flatmap';
const species = 'in';

export const in_: GremlinStep<string[]> = (...edgeLabels) => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const vertex = t.get() as Vertex<any>;

      // This only works on vertices, so we'll skip any non-vertex
      if (!Vertex.isVertex(vertex)) {
        // eslint-disable-next-line no-continue
        continue;
      }

      const edgesToByLabel = t.graph.edgesToByLabel.get(vertex.id) ?? new Map();

      // if the user provided labels, then they care about certain labels, otherwise, they want
      // all of the labels.
      const labelsToUse = edgeLabels.length ? edgeLabels : edgesToByLabel.keys();

      // Iterate over each of the labels
      for (const label of labelsToUse) {
        const edgesTo = edgesToByLabel.get(label) ?? new Set();

        for (const edge of edgesTo) {
          yield t.clone({ value: edge.from });
        }
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: edgeLabels, callback, genus, species });

  return execute(addStep, callback);
};
