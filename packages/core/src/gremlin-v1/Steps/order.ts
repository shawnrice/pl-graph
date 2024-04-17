/* eslint-disable @typescript-eslint/no-magic-numbers */

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'order';

export type SortOrder = 'ASC' | 'DESC' | 0 | 1;

export const order: GremlinStep = (...args) => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    yield* Array.from(prev).sort((a, b) => {
      const valA = a.get();
      const valB = b.get();

      if (typeof valA === 'string' && typeof valB === 'string') {
        return valA.localeCompare(valB);
      }

      return valA - valB;
    });
  };

  const addStep = (traversal: Traversal) => traversal.addStep({ args, callback, genus, species });

  return execute(addStep, callback);
};
