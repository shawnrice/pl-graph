import {
  appendStep,
  type ByableStep,
  makeByable,
  orderDirOf,
  scopeTokenOf,
  type Step,
  type StepFn,
} from './framework.js';

// Numeric/comparable aggregates (return a one-element stream).
//
// Each accepts an optional first `Scope` argument. With `Scope.local`, the
// aggregate is computed over each traverser's iterable VALUE rather than
// across the stream — typical use: `g.V().valueMap('age').fold().count(Scope.local)`.
// With `Scope.global` (default), the aggregate runs across the stream.
export function count(scope?: symbol): StepFn {
  return appendStep({ kind: 'count', scope: scope ? scopeTokenOf(scope) : undefined });
}

export function sum(scope?: symbol): StepFn {
  return appendStep({ kind: 'sum', scope: scope ? scopeTokenOf(scope) : undefined });
}

export function min(scope?: symbol): StepFn {
  return appendStep({ kind: 'min', scope: scope ? scopeTokenOf(scope) : undefined });
}

export function max(scope?: symbol): StepFn {
  return appendStep({ kind: 'max', scope: scope ? scopeTokenOf(scope) : undefined });
}

export function mean(scope?: symbol): StepFn {
  return appendStep({ kind: 'mean', scope: scope ? scopeTokenOf(scope) : undefined });
}

// Sort. With no args, natural order on the values themselves; with a `key`,
// sort by that property (vertex/edge); pass `desc: true` to flip. The
// modulator form `order().by(...)` overrides the legacy config-object args.
// `order(Scope.local)` sorts WITHIN each traverser's value (a group Map's
// entries by value, or a list's elements) instead of across the stream — e.g.
// `groupCount().by(x).order(Scope.local).by(Order.desc)` for a top-N ranking.
export function order(scopeOrDir: symbol): ByableStep<Extract<Step, { kind: 'order' }>>;
export function order(config?: {
  key?: string;
  desc?: boolean;
}): ByableStep<Extract<Step, { kind: 'order' }>>;
export function order(
  arg: symbol | { key?: string; desc?: boolean } = {},
): ByableStep<Extract<Step, { kind: 'order' }>> {
  // `order(desc)` is a superset — TinkerPop's `order()` takes only a Scope and
  // the direction belongs in the modulator — accepted because it is written, and
  // because the Rust engine accepts it. There it used to PARSE the direction and
  // then drop it, sorting ascending in silence; here it threw "Expected
  // Scope.local or Scope.global". Both now sort descending.
  const sym = typeof arg === 'symbol' ? arg : undefined;
  const dir = sym === undefined ? undefined : orderDirOf(sym);
  // Only a non-direction symbol is asked to be a scope, so an unknown one still
  // throws "Expected Scope.local or Scope.global".
  const scope = sym !== undefined && dir === undefined ? scopeTokenOf(sym) : undefined;
  let config: { key?: string; desc?: boolean };

  if (sym === undefined) {
    config = arg as { key?: string; desc?: boolean };
  } else {
    config = dir === undefined ? {} : { desc: dir === 'desc' };
  }

  return makeByable<Extract<Step, { kind: 'order' }>>((bys) => ({
    kind: 'order',
    ...config,
    bys,
    scope,
  }));
}

// `group()` collects the whole stream into a single `Map<key, value[]>`.
// The legacy config-object form (`group({ keyBy, valueBy })`) is still
// accepted; the modulator form is `group().by(keyBy).by(valueBy)`.
export const group = (
  config: { keyBy?: string; valueBy?: string } = {},
): ByableStep<Extract<Step, { kind: 'group' }>> =>
  makeByable<Extract<Step, { kind: 'group' }>>((bys) => ({
    kind: 'group',
    ...config,
    bys,
  }));

// `groupCount()` is `group` with values replaced by counts. Legacy
// config-object form `{ by }` still works; modulator form is
// `groupCount().by(...)`.
export const groupCount = (
  config: { by?: string } = {},
): ByableStep<Extract<Step, { kind: 'groupCount' }>> =>
  makeByable<Extract<Step, { kind: 'groupCount' }>>((bys) => ({
    kind: 'groupCount',
    ...config,
    bys,
  }));

// Eager terminal alias for `fold()`.
export const toList = (): StepFn => appendStep({ kind: 'toList' });
