import { isElement } from '@pl-graph/core/src/core/Element';

import { Traversal } from '../../Traversal';
import { Traverser } from '../../Traverser';

/* eslint-disable @typescript-eslint/no-magic-numbers */
/* eslint-disable func-style */
/* eslint-disable functional/immutable-data */
/* eslint-disable no-restricted-syntax */
import type { UnaryFn } from '@pl-graph/fp/src';

export const hasNot = <S, E = S>(key: string): UnaryFn<Traversal<S, E>> => {
  return traversal => {
    const callback = function* callback(prev: Iterable<Traverser<S>>) {
      for (const t of prev) {
        const val = t.get();
        if (isElement(val)) {
          if (!(key in val.properties)) {
            yield t.clone();
          }
        } else if (typeof val === 'object' && val && !(key in val)) {
          yield t.clone();
        }
      }
    };

    return traversal.addStep({
      args: [key],
      callback,
      genus: 'filter',
      species: 'hasNot',
    });
  };
};
