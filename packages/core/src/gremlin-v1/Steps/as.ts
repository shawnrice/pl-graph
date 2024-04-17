import { UnaryFn } from '../../../../fp/src/types';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

/**
 * Keeps a value of a traverser for later
 *
 * NOTE: This is modeled as a real step, but it's supposed to be just a step modulator
 */
export const as = (...keys: string[]): UnaryFn<Traversal<any, any>> => {
  const callback = function* (traversers: Iterable<Traverser<any>>) {
    // Below is basically the callback
    for (const traverser of traversers) {
      for (const key of keys) {
        traverser.sideEffects.set(key, traverser.get());
        yield traverser;
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({
      args: keys,
      callback,
      genus: 'modulator',
      species: 'as',
    });

  return execute(addStep, callback);
};
