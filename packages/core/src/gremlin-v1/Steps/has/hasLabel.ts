/* eslint-disable functional/immutable-data */
/* eslint-disable no-restricted-syntax */
import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../../Traversal';
import { Traverser } from '../../Traverser';

const fallback = function*(prev: Generator<any, void>) {
  yield* prev;
};

const getName = (labels: string[]): string => {
  if (labels.length === 1) {
    const [label] = labels;
    return `has[~label.eq(${label!})]`;
  }

  return `has[~label.includes(${labels.join(',')})]`;
};

export const hasLabel =
  <Start, End>(...labels: string[]): UnaryFn<Traversal<Start, End>, Traversal<Start, End>> =>
    traversal => {
      const { graph } = traversal;

      const species = getName(labels);
      const params = { args: labels, callback: fallback, genus: 'filter', graph, species };

      const count = traversal.steps.length;
      const lastStep = traversal.steps[count - 1];
      if (count === 1 && lastStep && lastStep.genus === 'start') {
        if (lastStep.species === 'V') {
          // So, if there is only one step queued, and if that step is a 'V' step, then we can replace
          // it. (We also should check to see if it was a `V(string)` or a `V(vertex)` call rather
          // than an empty `V` call)

          // We're going to pretend it was `g().V().hasLabel(x)`

          // So, we're effectively replacing the last step with a noop function
          // we'll take care of injecting those values in this one but more efficiently
          lastStep.callback = fallback;

          const callback = function* callback(prev: Iterable<Traverser<any>>) {
            yield* prev; // I think we're replacing this. Check it
            for (const label of labels) {
              const vertices = graph.verticesByLabel.get(label) ?? new Set();
              for (const vertex of vertices) {
                yield new Traverser({ value: vertex, graph });
              }
            }
          };

          return traversal.addStep({ ...params, callback });
        }

        if (lastStep.species === 'E') {
          // So, if there is only one step queued, and if that step is a 'E' step, then we can replace
          // it. (We also should check to see if it was a `E(string)` or a `E(edge)` call rather
          // than an empty `E` call)

          // We're going to pretend it was `g().E().hasLabel(x)`

          // So, we're effectively replacing the last step with a noop function
          // we'll take care of injecting those values in this one but more efficiently
          lastStep.callback = fallback;

          const callback = function* callback(prev: Iterable<Traverser<any>>) {
            yield* prev; // I think we're replacing this. Check it
            for (const label of labels) {
              const edges = graph.edgesByLabel.get(label) ?? new Set();
              for (const edge of edges) {
                yield new Traverser({ value: edge, graph });
              }
            }
          };

          return traversal.addStep({ ...params, callback });
        }
      }

      const callback = function* callback(prev: Iterable<Traverser<any>>) {
        for (const traverser of prev) {
          const value = traverser.get();
          if (value && typeof value.hasLabel === 'function' && value.hasLabel(...labels)) {
            yield traverser.clone();
          }
        }
      };

      return traversal.addStep({ ...params, callback });
    };
