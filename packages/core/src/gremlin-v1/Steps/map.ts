import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'map';

export const map = <S, E = S>(mapper: UnaryFn<S, E>): UnaryFn<Traversal<any, any>> => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const mapped = mapper(t.get());

      yield t.clone({ value: mapped });
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [mapper], callback, genus, species });

  return execute(addStep, callback);
};
