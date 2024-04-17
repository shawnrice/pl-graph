import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

/**
 * This is a terminal step... in that sense, it's not really a step but a function that runs everything
 */
export const toArray = <S, E>(traversal: Traversal<S, E>): E[] => {
  // Run through all the steps
  const result: Traverser<E>[] | E[] = traversal.steps.reduce(
    (current, step) => Array.from(step.callback(current)),
    traversal.current,
  );

  // Unwrap the items from the Traversers
  return result.map(x => (x instanceof Traverser ? x.value : x));
};
