/* eslint-disable @typescript-eslint/no-magic-numbers, func-style, functional/immutable-data, no-restricted-syntax */
import { isElement } from '@pl-graph/core/src/core/Element';

import { Traversal } from '../../Traversal';
import { Traverser } from '../../Traverser';

import type { UnaryFn } from '@pl-graph/fp/src';

// TODO add hasKey(predicate: StringPredicate)

export const hasKey = <S, E = S>(...keys: string[]): UnaryFn<Traversal<S, E>> => {
  return traversal => {
    const callback = function* callback(prev: Iterable<Traverser<S>>) {
      for (const t of prev) {
        const val = t.get();
        if (isElement(val)) {
          if (keys.some(key => key in val.properties)) {
            yield t.clone();
          }
        } else if (typeof val === 'object' && val && keys.some(key => key in val)) {
          yield t.clone();
        }
      }
    };

    return traversal.addStep({ args: keys, genus: 'filter', callback, species: 'hasKey' });
  };
};
