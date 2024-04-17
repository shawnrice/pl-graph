import { Traversal } from '../Traversal';
import { Traverser } from '../Traverser';
import { execute } from './execute';
import { GremlinStep, StepGenus } from './types';

const genus: StepGenus = 'sideEffect';
const species = 'property';

/**
 * The `property()`-step is used to add properties to the elements of the graph (**sideEffect**).
 * Unlike `addV()` and `addE()`, `property()` is a full sideEffect step in that it does not return
 * the property it created, but the element that streamed into it. Moreover, if `property()`
 * follows an `addV()` or `addE()`, then it is "folded" into the previous step to enable vertex and
 * edge creation with all its properties in one creation operation.
 *
 * `property(value: Map<object,object>): GraphTraversal<S,E>`
 * When a `Map` is supplied then each of the key/value pairs in the map will be added as property.
 *
 * `property(key: object, value: object value, ...keyValues: object[]): GraphTraversal<S,E>`
 * Sets the key and value of a `Property`.
 *
 * `property(VertexProperty.Cardinality cardinality, Object key, Object value, Object... keyValues): GraphTraversal<S,E>`
 * Sets a `Property` value and related meta properties if supplied, if supported by the `Graph` and
 * if the `Element` is a `VertexProperty`.
 */
export const property: GremlinStep<string[]> = (...keys) => {
  const callback = function* (prev: Iterable<Traverser<any>>) {
    throw new Error('Not implemented');
  };

  const addStep = (traversal: Traversal) => {
    return traversal.addStep({ args: keys, callback, genus, species });
  };

  return execute(addStep, callback);
};
