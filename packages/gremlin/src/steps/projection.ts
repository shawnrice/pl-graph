import {
  appendStep,
  type ByableStep,
  makeByable,
  type Step,
  type StepFn,
  toBy,
} from './framework.js';

// Project to property values for the named keys.
export const values = (...keys: string[]): StepFn => appendStep({ kind: 'values', keys });

// Project to a single map of `{ [key]: value }`. With no keys, all properties.
// TinkerPop's `valueMap(includeTokens, ...keys)`: a leading boolean `true` also
// includes `id` and `label` in the map (like `elementMap()`, but no edge IN/OUT).
export const valueMap = (...args: (string | boolean)[]): StepFn => {
  const includeTokens = typeof args[0] === 'boolean' ? args[0] : false;
  const keys = (typeof args[0] === 'boolean' ? args.slice(1) : args) as string[];

  return appendStep({
    kind: 'valueMap',
    keys: keys.length ? keys : undefined,
    includeTokens: includeTokens || undefined,
  });
};

// Project property objects (`{key, value}`) for the named keys.
export const properties = (...keys: string[]): StepFn => appendStep({ kind: 'properties', keys });

// Project all (or selected) properties as a single map of arrays.
export const propertyMap = (...keys: string[]): StepFn =>
  appendStep({ kind: 'propertyMap', keys: keys.length ? keys : undefined });

// Project to id+label+(selected) properties. For edges, also includes IN/OUT
// submaps with the endpoint id+label.
export const elementMap = (...keys: string[]): StepFn =>
  appendStep({ kind: 'elementMap', keys: keys.length ? keys : undefined });

// Yield the value of the current property/edge. For `{key, value}` from
// `properties()`, unwraps to the value; otherwise identity.
export const value = (): StepFn => appendStep({ kind: 'value' });

// Project to the element id.
export const id = (): StepFn => appendStep({ kind: 'id' });

// Project to the element label (first label if multiple).
export const label = (): StepFn => appendStep({ kind: 'label' });

// `project(...)` emits a `{ [key]: value }` per traverser. Two call styles:
//   - TinkerPop varargs: `project('a', 'b').by(…).by(…)` (keys as varargs, one
//     `.by()` modulator per key) — matches the native string API and TinkerPop.
//   - Array form: `project(['a', 'b'], [byA, byB]?)` — keys array + optional
//     parallel `bys` array; `bys[i]` modulates `keys[i]`.
export function project(...keys: string[]): ByableStep<Extract<Step, { kind: 'project' }>>;
export function project(
  keys: readonly string[],
  bys?: readonly (string | StepFn | undefined)[],
): ByableStep<Extract<Step, { kind: 'project' }>>;
export function project(
  first: string | readonly string[],
  ...rest: unknown[]
): ByableStep<Extract<Step, { kind: 'project' }>> {
  const arrayForm = Array.isArray(first);
  const keys = arrayForm ? (first as readonly string[]) : [first as string, ...(rest as string[])];
  const bys = arrayForm
    ? (rest[0] as readonly (string | StepFn | undefined)[] | undefined)
    : undefined;
  const initial = bys?.map((b) => toBy(b));

  return makeByable<Extract<Step, { kind: 'project' }>>(
    (later) => ({ kind: 'project', keys, bys: later }),
    initial,
  );
}
