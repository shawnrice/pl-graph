/* eslint-disable no-restricted-syntax */
import { isElement } from '@pl-graph/core/src/core/Element';

import { Traverser } from '../../Traverser';

export const pathElementPropertyProjection = (key: string) => {
  return function* elementPropertyProjection(
    prev: Iterable<Traverser<any>>,
  ): Iterable<Traverser<any>> {
    for (const t of prev) {
      const p = t.get();
      if (Array.isArray(p) && p.every(x0 => isElement(x0) && key in x0.properties)) {
        yield t.clone({ value: p.map(x0 => x0.properties[key]) });
      }
    }
  };
};
