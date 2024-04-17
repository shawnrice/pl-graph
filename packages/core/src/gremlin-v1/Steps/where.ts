import type { UnaryFn } from '@pl-graph/fp/src';

import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';

import type { StepGenus } from './types';

export const where =
  (x: UnaryFn<Traversal<any>>): UnaryFn<Traversal<any, any>> =>
  traversal => {
    const { graph } = traversal;

    const genus: StepGenus = 'filter';
    const species = 'where';

    const callback = function* (prev: Iterable<Traverser<any>>) {
      const inflectionPoint = new Traversal({ graph });

      inflectionPoint.current = prev;
      x(inflectionPoint);

      inflectionPoint.flush(); // we should avoid some eager evaluation

      for (const y of inflectionPoint.current) {
        console.log(y);
        yield y;
      }

      const upUntilNow = inflectionPoint.steps.reduce((g, f) => f.callback(g), prev);

      for (const t of upUntilNow) {
        console.log('TESTING', t.get());
        if (Array.isArray(t.get())) {
          if (t.get().length !== 0) {
            yield t;
          }
        } else if (t.get()) {
          yield t;
        }
      }
    };

    return traversal.addStep({ args: [x], callback, genus, graph, species });
  };
