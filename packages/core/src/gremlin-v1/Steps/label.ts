import type { UnaryFn } from '@pl-graph/fp/src';
import { map } from '@pl-graph/fp/src';
import { isElement } from '../../core/Element';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import type { StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'label';

export const label = (): UnaryFn<Traversal<any, any>> => {
  const callback = function*(prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const element = t.get();

      if (isElement(element)) {
        // For, Vertices and Edges, we'll yield the labels
        yield* map((value: string) => t.clone({ value }), element.labels);
      } else if (typeof element === 'object' && element !== null) {
        // If it's an actual object, then we'll yield to the keys
        yield* map((value: string) => t.clone({ value }), Object.keys(element));
      }
    }
  };

  const addStep = (t: Traversal) => t.addStep({ args: [], callback, genus, species });

  return execute(addStep, callback);
};
