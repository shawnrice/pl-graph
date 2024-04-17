import { isElement } from '@pl-graph/core/src/core/Element';

import { Traversal } from '../../Traversal';
import { Traverser } from '../../Traverser';

/* eslint-disable @typescript-eslint/no-magic-numbers */
/* eslint-disable func-style */
/* eslint-disable functional/immutable-data */
/* eslint-disable no-restricted-syntax */
import type { UnaryFn } from '@pl-graph/fp/src';

// TODO add hasValue(predicate: StringPredicate)

const noop = function* <S>(prev: Iterable<Traverser<S>>) {
  yield* prev;
};

/**
 * According to the spec (as far as I can tell), the hasValue step should be chained only off
 * a properties() step
 */
export const hasValue = <S, E = S>(...values: any[]): UnaryFn<Traversal<S, E>> => {
  return traversal => {
    const lastStep = traversal.steps[traversal.steps.length - 1];
    if (!(traversal.steps.length > 0 && lastStep && lastStep.species === 'properties')) {
      // this is a no-op....
      // TODO figure out what to do here
      return traversal;
    }

    lastStep.callback = noop;
    const keys = lastStep.args;

    const callback = function* callback(prev: Iterable<Traverser<S>>) {
      for (const t of prev) {
        const element = t.get();

        if (!isElement(element)) {
          // eslint-disable-next-line no-continue
          continue;
        }

        if (keys.length === 0) {
          // What do we do here?
          yield t.clone({ value: element.properties });
        } else {
          const partial = keys.reduce<Record<string, any>>((acc, key) => {
            if (key in element.properties && values.includes(element.properties[key])) {
              // eslint-disable-next-line functional/immutable-data
              acc[key] = element.properties[key];
            }
            return acc;
          }, {});

          yield t.clone({ value: partial });
        }
      }
    };

    return traversal.addStep({
      args: keys,
      callback,
      genus: 'filter',
      species: 'hasValue',
    });
  };
};
