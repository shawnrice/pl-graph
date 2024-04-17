import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';

import type { StepGenus } from './types';

const genus: StepGenus = 'filter';
const species = 'dedupe';

/**
 * @see https://tinkerpop.apache.org/docs/current/reference/#dedup-step
 */
export const dedupe = <S, E = S>(...args: string[]): UnaryFn<Traversal<S, E>> => {
  // TODO support de-duplicating based on passed in labels
  // @see https://tinkerpop.apache.org/javadocs/3.6.1/core/org/apache/tinkerpop/gremlin/process/traversal/dsl/graph/GraphTraversal.html#dedup-java.lang.String...-

  const callback = function* (prev: Iterable<Traverser<any>>) {
    const seen = new Set();
    for (const t of prev) {
      const val = t.get();

      if (!seen.has(val)) {
        seen.add(val);
        yield t;
      }
    }
  };

  const addStep = (traversal: Traversal) => traversal.addStep({ args, callback, genus, species });

  return execute(addStep, callback);
};
