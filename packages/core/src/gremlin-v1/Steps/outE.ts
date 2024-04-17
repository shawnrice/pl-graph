import { Edge } from '../../core/Edge';
import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, TraverserFunction } from './types';

export const outE: GremlinStep<string[]> = (...edgeLabels) => {
  const callback: TraverserFunction = function* outEdges(prev: Iterable<Traverser<Vertex>>) {
    for (const t of prev) {
      const vertex = t.get() as Vertex;

      // This only works on vertices, so we'll skip any non-vertex
      if (!Vertex.isVertex(vertex)) {
        continue; // eslint-disable-line no-continue
      }

      const edgesFromByLabel: Map<string, Set<Edge>> = t.graph.edgesFromByLabel.get(vertex.id) ??
      new Map();

      // if the user provided labels, then they care about certain labels, otherwise, they want
      // all of the labels.
      const labelsToUse = edgeLabels.length ? edgeLabels : edgesFromByLabel.keys();

      // Iterate over each of the labels
      for (const label of labelsToUse) {
        for (const value of edgesFromByLabel.get(label) ?? new Set()) {
          yield t.clone<Edge>({ value });
        }
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({
      args: edgeLabels,
      callback,
      genus: 'flatmap',
      species: 'outE',
    });

  return execute(addStep, callback);
};
