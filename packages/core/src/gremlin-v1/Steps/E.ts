import { Edge } from '../../core/Edge';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

const genus: StepGenus = 'start';
const species = 'E';

export const E =
  (x?: Edge<any, any, any> | string | null): UnaryFn<Traversal<any, any>> =>
  traversal => {
    const { graph } = traversal;

    const fallback = function* (prev: Generator<any, void>) {
      yield* prev;
    };

    const params = { args: x ?? [], callback: fallback, genus, graph, species };

    if (typeof x === 'undefined') {
      const callback = function* (prev: Iterator<any>) {
        yield* prev; // <-- Iterable<Traverser<T>>
        for (const edge of graph.edges) {
          // This is a start step, which means that we will create new Traversers
          yield new Traverser<Edge<any, any, any>>({ value: edge, graph });
        }
      };

      return traversal.addStep({ ...params, callback });
    }

    if (typeof x === 'object' && x === null) {
      // This turns out to be a no-op step
      return traversal;
    }

    if (x instanceof Edge) {
      if (graph.owns(x)) {
        const callback = function* (prev: Generator<any, void>) {
          yield* prev;
          yield new Traverser<Edge<any, any, any>>({ value: x, graph });
        };

        return traversal.addStep({ ...params, callback });
      }
      // There wasn't a valid thing to insert, so, uh...
      return traversal.addStep(params);
    }

    if (typeof x === 'string') {
      const edge = graph.getEdgeById(x);

      if (edge && graph.owns(edge)) {
        const callback = function* (prev: Generator<any, void>) {
          yield* prev;
          yield new Traverser<Edge<any, any, any>>({ value: edge, graph });
        };

        return traversal.addStep({ ...params, callback });
      }

      console.log('Cannot find Edge by id', x);
    }

    return traversal.addStep(params);
  };
