import { pick } from '../../../../fp/src/pick';
import { isElement } from '../../core/Element';
import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'map';
const species = 'properties';

export const properties: GremlinStep<string[]> = (...keys) => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    for (const t of prev) {
      const element = t.get();

      if (!isElement(element)) {
        // eslint-disable-next-line no-continue
        continue;
      }

      if (keys.length === 0) {
        yield t.clone({ value: element.properties });
      } else {
        yield t.clone({ value: pick(keys, element.properties) });
      }
    }
  };

  const addStep = (traversal: Traversal) =>
    traversal.addStep({ args: keys, callback, genus, species });

  return execute(addStep, callback);
};
