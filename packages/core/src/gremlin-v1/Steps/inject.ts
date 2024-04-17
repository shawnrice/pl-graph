import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

const genus: StepGenus = 'sideEffect';
const species = 'inject';

export function inject<S, E = S>(...args: any[]): UnaryFn<Traversal<S, E>> {
  return traversal => {
    const { graph } = traversal;

    const callback = function* (prev: Iterable<Traverser<any>>) {
      // The injection happens first before we yield to the other items in the stream
      for (const arg of args) {
        yield new Traverser({ value: arg, graph });
      }

      yield* prev;
    };

    return traversal.addStep({ args, callback, genus, species });
  };
}
