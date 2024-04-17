import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep } from './types';

const genus = 'map';
const species = 'select';

export const select: GremlinStep<string[]> = (...keys: string[]) => {
  function* callback(prev: Iterable<Traverser<any>>) {
    for (const traverser of prev) {
      if (keys.length === 1) {
        const [key] = keys as [string];

        if (traverser.sideEffects.has(key)) {
          yield traverser.clone({ value: traverser.sideEffect(key) });
        }
      } else {
        const value = Object.fromEntries(
          keys
            .map(key => (traverser.sideEffects.has(key) ? [key, traverser.sideEffect(key)] : false))
            .filter(Boolean) as [string, any][],
        );

        yield traverser.clone({ value });
      }
    }
  }

  const addStep = (traversal: Traversal) =>
    traversal.addStep({
      args: keys,
      callback,
      genus,
      species,
    });

  return execute(addStep, callback);
};
