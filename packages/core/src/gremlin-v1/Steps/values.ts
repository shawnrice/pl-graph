import { filter, map as fpMap, pipe, UnaryFn } from '../../../../fp/src';
import { isElement } from '../../core/Element';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'values';

export const values = (...keys: string[]): UnaryFn<Traversal<any, any>> => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const traverser of prev) {
      const element = traverser.get();

      if (isElement(element)) {
        if (keys.length === 0) {
          const map = fpMap((value: any) => traverser.clone({ value }));
          yield* map(Object.values(element.properties));
        } else {
          const map = pipe(
            filter((key: any) => key in element.properties),
            fpMap((key: string) => traverser.clone({ value: element.properties[key]! })),
          );

          yield* map(keys);
        }
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: [], callback, genus, species });

  return execute(addStep, callback);
};
