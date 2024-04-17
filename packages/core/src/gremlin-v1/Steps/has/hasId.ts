import type { UnaryFn } from '@pl-graph/fp/src';

/* eslint-disable functional/immutable-data */
/* eslint-disable no-restricted-syntax */
import { Traversal } from '../../Traversal';
import { Traverser } from '../../Traverser';

export const hasId =
  <S, E = S>(...ids: string[]): UnaryFn<Traversal<S, E>> =>
  traversal => {
    const callback = function* callback(prev: Iterable<Traverser<any>>) {
      for (const traverser of prev) {
        const value = traverser.get();
        if (value && 'id' in value && ids.includes(value.id)) {
          yield traverser.clone();
        }
      }
    };

    return traversal.addStep({ args: ids, callback, genus: 'filter', species: 'hasId' });
  };
