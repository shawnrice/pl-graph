import { List } from '../../../../list/src';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

/**
 * This is a terminal step... in that sense, it's not really a step but a function that runs everything
 */
export const toList = <S, E>(traversal: Traversal<S, E>): List<E> => {
  const callback = function* traversalToList() {
    let val = traversal.current;
    for (const step of traversal.steps) {
      val = Array.from(step.callback(val));
    }

    yield* List.from(val).map(x => (x instanceof Traverser ? x.value : x));
  };

  return new List(callback);
};
