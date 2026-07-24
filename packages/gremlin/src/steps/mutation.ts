import { ErrorCode, LenkeError } from '@lenke/errors';

import {
  appendStep,
  buildPlan,
  CARDINALITY_TO_KIND,
  type CardinalitySym,
  isSubPlan,
  type Plan,
  type StepFn,
  type SubPlan,
} from './framework.js';

/**
 * Insert a new vertex into the graph and emit it as the next traverser.
 *
 * Subsequent `property(key, value)` calls bind values to the new vertex.
 * The label is optional; without one, the vertex is created label-less.
 *
 * Example: `traversal(addV('PERSON'), property('name', 'marko'))`.
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#addvertex-step
 */
export const addV = (label?: string): StepFn => appendStep({ kind: 'addV', label });

type AddEEndpointArg = string | SubPlan;

const toAddEEndpoint = (arg: AddEEndpointArg) =>
  typeof arg === 'string'
    ? { kind: 'tag' as const, label: arg }
    : { kind: 'plan' as const, plan: buildPlan(arg) };

/**
 * Builder returned by `addE(label)`. Both `.from()` and `.to()` are optional;
 * if only one is set, the current traverser fills the other slot. If neither
 * is set, the executor throws (an edge needs both endpoints).
 *
 * Each accepts a tag string (recalled via prior `as(label)`) or a sub-plan
 * (`traversal(V('2'))`, `inject(someVertex)`, etc.).
 */
type AddEBuilder = StepFn & {
  from: (arg: AddEEndpointArg) => AddEBuilder;
  to: (arg: AddEEndpointArg) => AddEBuilder;
};

/**
 * Insert a new edge and emit it as the next traverser.
 *
 * Common shapes:
 *   - `traversal(V('1'), addE('KNOWS').to(V('2')))`           // input is FROM
 *   - `traversal(V('1'), as_('a'), V('2'), addE('KNOWS').from('a'))` // tag-form
 *   - `traversal(addE('KNOWS').from(V('1')).to(V('2')))`      // both explicit
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#addedge-step
 */
export const addE = (label: string): AddEBuilder => {
  const make = (config: {
    label: string;
    from?: { kind: 'tag'; label: string } | { kind: 'plan'; plan: Plan };
    to?: { kind: 'tag'; label: string } | { kind: 'plan'; plan: Plan };
  }): AddEBuilder => {
    const fn: StepFn = appendStep({ kind: 'addE', ...config });

    return Object.assign(fn, {
      from: (arg: AddEEndpointArg) => make({ ...config, from: toAddEEndpoint(arg) }),
      to: (arg: AddEEndpointArg) => make({ ...config, to: toAddEEndpoint(arg) }),
    });
  };

  return make({ label });
};

/**
 * Set a property on the current vertex/edge.
 *
 * Two forms:
 *   - `property(key, value)` — single-cardinality (overwrite). Default.
 *   - `property(Cardinality.X, key, value)` — explicit cardinality. v2's
 *     storage model is single-valued (`Record<string, unknown>`), so `list`
 *     and `set` currently behave identically to `single`. The cardinality is
 *     still threaded through the AST so a future multi-cardinality executor
 *     pass can pick it up without a DSL break.
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#addproperty-step
 */
// A traversal value (`property('deg', __.outE().count())`, standard TinkerPop
// "traversal-induced value") is normalized to a Plan; the executor evaluates it
// per element. A plain literal passes through unchanged.
const normPropValue = (v: unknown): unknown => (isSubPlan(v) ? buildPlan(v) : v);

export function property(key: string, value: unknown): StepFn;
export function property(cardinality: CardinalitySym, key: string, value: unknown): StepFn;
export function property(...args: [string, unknown] | [CardinalitySym, string, unknown]): StepFn {
  if (typeof args[0] === 'symbol') {
    const cardinality = CARDINALITY_TO_KIND.get(args[0]);

    if (cardinality === undefined) {
      throw new LenkeError(`property(): unrecognized cardinality symbol ${String(args[0])}`, {
        code: ErrorCode.Unsupported,
      });
    }

    return appendStep({
      kind: 'property',
      key: args[1] as string,
      value: normPropValue(args[2]),
      cardinality,
    });
  }

  return appendStep({ kind: 'property', key: args[0], value: normPropValue(args[1]) });
}

/**
 * Remove the current vertex or edge from the graph. The traverser is dropped
 * (no value is emitted for it). Dropping a vertex cascades — any edges
 * incident to it are also removed.
 *
 * @see https://tinkerpop.apache.org/docs/current/reference/#drop-step
 */
export const drop = (): StepFn => appendStep({ kind: 'drop' });
