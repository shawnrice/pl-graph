/* eslint-disable no-restricted-syntax */
import { UnaryFn } from '../../../../fp/src/types';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'branch';
const species = 'chose';

export const choose = (
  condition: Traversal<any> | UnaryFn<Traverser<any>, any>,
  branchA: Traversal<any> | UnaryFn<Traverser<any>, any>,
  branchB?: Traversal<any> | UnaryFn<Traverser<any>, any>,
): UnaryFn<Traversal<any, any>> => {
  const callback = function* callback(prev: Iterable<Traverser<any>>) {
    throw new Error('We need access to the graph?');
  };

  const addStep = (t: Traversal) => {
    const callback = function* callback(prev: Iterable<Traverser<Vertex>>) {
      for (const traverser of prev) {
        const generator = condition(
          new Traversal({ graph: t.graph, value: [traverser.clone()] }),
        ).run();
        if (!generator.next().done) {
          yield* branchA(new Traversal({ graph: t.graph, value: [traverser] })).run();
        } else if (branchB) {
          yield* branchB(new Traversal({ graph: t.graph, value: [traverser] })).run();
        } else {
          yield traverser;
        }
      }
    };

    return t.addStep({
      args: [condition, branchA, branchB],
      callback,
      genus,
      species,
    });
  };

  return execute(addStep, callback);
};
