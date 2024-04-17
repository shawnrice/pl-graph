import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

const genus = 'filter';
const species = 'range';

export const range = (low: number, high: number): UnaryFn<Traversal<any, any>> => {
  function* callback(prev: Iterable<Traverser<any>>) {
    let counter = 0;
    const highValue = high === -1 ? Number.MAX_SAFE_INTEGER : high;
    for (const traverser of prev) {
      if (highValue <= counter) {
        break;
      }

      if (low <= counter) {
        yield traverser;
      }

      counter++;
    }
  }

  const addStep = (traversal: Traversal) => {
    // what this needs to do is to get the last value and label it in the side effects
    return traversal.addStep({
      args: [low, high],
      callback,
      genus,
      species,
    });
  };

  return execute(addStep, callback);
};
