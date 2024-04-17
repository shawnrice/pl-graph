import { UnaryFn } from '../../../../fp/src/types';
import { Traversal } from '../Traversal';

import type { StepGenus } from './types';

export class FailStep extends Error {}

const genus: StepGenus = 'sideEffect';
const species = 'fail';

export function fail<S, E = S>(message?: string): UnaryFn<Traversal<S, E>> {
  return traversal => {
    const callback = () => {
      throw new FailStep(message ?? 'FailStep Triggered');
    };

    return traversal.addStep({ args: message ? [message] : [], callback, genus, species });
  };
}
