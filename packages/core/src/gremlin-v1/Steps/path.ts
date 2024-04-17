import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'path';

export const path: GremlinStep<never> = () => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      // We are yielding and not yield*ing to the array here
      yield t.clone({ value: t.path() });
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [], callback, genus, species });

  return execute(addStep, callback);
};
