import type { UnaryFn } from '@pl-graph/fp/src';

import { Edge } from '../../core/Edge';
import { isElement } from '../../core/Element';
import { Vertex } from '../../core/Vertex';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

export const id = (): UnaryFn<Traversal<any, any>> => traversal => {
  const genus: StepGenus = 'map';
  const species = 'id';

  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const element = t.get() as Vertex<any> | Edge<any, any, any>;

      // This only works on elements (vertices and edges), so we'll skip any non-elements
      if (isElement(element) && 'id' in element) {
        yield t.clone({ value: element.id });
      }
    }
  };

  return traversal.addStep({ args: [], callback, genus, species });
};
