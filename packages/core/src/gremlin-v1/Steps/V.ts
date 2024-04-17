import { map, UnaryFn } from '../../../../fp/src';
import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { passThrough } from './passThrough';

import type { StepGenus } from './types';

const genus: StepGenus = 'start';
const species = 'V';

export const V = (x?: Vertex<any> | string | null): UnaryFn<Traversal<any>> => {
  const params = {
    args: typeof x === 'undefined' ? [] : [x],
    callback: passThrough,
    genus,
    species,
  };

  return traversal => {
    const { graph } = traversal;

    if (typeof x === 'undefined') {
      const callback = function* (prev: Iterable<Traverser<any>>) {
        yield* prev;
        yield* map((value: Vertex) => new Traverser<Vertex<any>>({ value, graph }), graph.vertices);
      };

      return traversal.addStep({ ...params, callback });
    }

    if (typeof x === 'object' && x === null) {
      // This turns out to be a no-op step
      return traversal;
    }

    if (x instanceof Vertex) {
      if (graph.owns(x)) {
        const callback = function* (prev: Generator<any, void>) {
          yield* prev;
          yield new Traverser<Vertex<any>>({ value: x, graph });
        };

        return traversal.addStep({ ...params, callback });
      }

      // There wasn't a valid thing to insert, so, uh...
      return traversal.addStep(params);
    }

    if (typeof x === 'string') {
      const vertex = graph.getVertexById(x);

      if (vertex && graph.owns(vertex)) {
        const callback = function* (prev: Generator<any, void>) {
          yield* prev;
          yield new Traverser<Vertex<any>>({ value: vertex, graph });
        };

        return traversal.addStep({ ...params, callback });
      }
    }

    return traversal.addStep(params);
  };
};
