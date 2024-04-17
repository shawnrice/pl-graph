import { filter as fpFilter } from '../../../../fp/src';
import { UnaryFn } from '../../../../fp/src/types';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'filter';
const species = 'filter';

/**
 * Helps with filtering
 */
export const filter = (predicate: UnaryFn<any, boolean>): UnaryFn<Traversal<any, boolean>> => {
  const fn = (t: Traverser<any>): boolean => predicate(t.get());
  const callback = fpFilter(fn);

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [predicate], callback, genus, species });

  return execute(addStep, callback);
};
