import { isElement } from '../../core/Element';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'valueMap';

export const valueMap: GremlinStep<string[]> = (...keys) => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const element = t.get();
      if (isElement(element)) {
        if (keys.length === 0) {
          yield t.clone({ value: t.get().properties });
        } else {
          yield t.clone({
            value: keys.reduce<Record<string, any>>((acc, key) => {
              if (key in element.properties) {
                // eslint-disable-next-line functional/immutable-data
                acc[key] = element.properties[key];
              }
              return acc;
            }, {}),
          });
        }
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: keys, callback, genus, species });

  return execute(addStep, callback);
};
