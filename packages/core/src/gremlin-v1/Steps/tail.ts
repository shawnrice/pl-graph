import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep } from './types';

const genus = 'filter';
const species = 'tail';

export const tail: GremlinStep<number[]> = (x?: number) => {
  const num = typeof x === 'number' ? x : 1;

  function* callback(prev: Iterable<Traverser<any>>) {
    const bucket = [];

    if (num < 1) {
      yield* [];
    } else {
      for (const traverser of prev) {
        bucket.push(traverser);
      }

      yield* bucket.slice(-num);
    }
  }

  const addStep = (traversal: Traversal) => {
    // what this needs to do is to get the last value and label it in the side effects
    return traversal.addStep({
      args: x ? [x] : [],
      callback,
      genus,
      species,
    });
  };

  return execute(addStep, callback);
};
