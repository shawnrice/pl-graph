import { UnaryFn } from '../../../../fp/src/types';
import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'flatmap';
const species = 'both';

export const bothE = (...edgeLabels: string[]): UnaryFn<Traversal<any, any>> => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const vertex = t.get() as Vertex<any>;

      // This only works on vertices, so we'll skip any non-vertex
      if (!Vertex.isVertex(vertex)) {
        // eslint-disable-next-line no-continue
        continue;
      }

      const edgesFromByLabel = t.graph.edgesFromByLabel.get(vertex.id) ?? new Map();
      const edgesToByLabel = t.graph.edgesToByLabel.get(vertex.id) ?? new Map();

      // if the user provided labels, then they care about certain labels, otherwise, they want
      // all of the labels.
      const labelsToUse = edgeLabels.length
        ? edgeLabels
        : new Set([...edgesFromByLabel.keys(), ...edgesToByLabel.keys()]);

      // Iterate over each of the labels
      for (const label of labelsToUse) {
        for (const edge of edgesFromByLabel.get(label) ?? new Set()) {
          yield t.clone({ value: edge });
        }

        for (const edge of edgesToByLabel.get(label) ?? new Set()) {
          yield t.clone({ value: edge });
        }
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: edgeLabels, callback, genus, species });

  return execute(addStep, callback);
};
