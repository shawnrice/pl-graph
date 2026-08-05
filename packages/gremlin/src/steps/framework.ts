//
// Shared scaffolding for step constructors. Each `steps/{group}.ts` imports
// from here. Nothing in this file is a step; everything here is either
// a token table (`T`/`Order`/`Scope`/`Cardinality`/`Pop`), a small type
// (`StepFn`/`SubPlan`/`ByableStep`), or a helper (`makeByable`/`buildPlan`
// /`isPredicate`/etc.).

import { ErrorCode, LenkeError } from '@lenke/errors';

import {
  appendStep,
  type By,
  isStepFn,
  type Plan,
  type Predicate,
  type SackOp,
  type Step,
} from '../ast.js';

// ---------- Core types ----------

export type StepFn = (plan: Plan) => Plan;

/**
 * What every sub-plan combinator (`where`, `filter`, `map`, `repeat`,
 * `union`, `choose`, ...) accepts. In TinkerPop terms: anywhere a `Traversal`
 * is expected — anonymous (`__.out()`) or rooted (`g.V()`). We accept both:
 *
 *   - `StepFn`  — what `out()`, `pipe(...)`, etc. produce
 *   - `Plan`    — what `traversal(...)` produces
 *
 * Closure-bearing combinators (`map`, `filter`, `flatMap`, `sideEffect`)
 * additionally accept a raw closure; the dispatch in each constructor
 * routes by shape.
 */
export type SubPlan = StepFn | Plan;

// ---------- Token tables ----------

// A registry (`Symbol.for`) symbol tagged with its family, so the token tables
// (`T` / `Order` / `Column`) become DISTINCT nominal types rather than all
// collapsing to bare `symbol`. Without the brand, `Token`, `OrderSym`, and
// `ColumnSym` are structurally identical (every `Symbol.for(...)` is typed
// `symbol`), so `by()` couldn't reject a symbol from the wrong family and
// `ByModulator = string | Token | ColumnSym | …` had a duplicate constituent.
// The brand is phantom (type-level only); at runtime these are plain symbols.
type FamilySymbol<Family extends string> = symbol & { readonly __family: Family };
const familySymbol = <Family extends string>(key: string): FamilySymbol<Family> =>
  Symbol.for(key) as FamilySymbol<Family>;

// Tokens for `by()` — project to well-known facets of an element. Symbols
// (rather than string literals) so they don't collide with user-supplied
// property names.
export const T = {
  id: familySymbol<'Token'>('@lenke/gremlin/T.id'),
  label: familySymbol<'Token'>('@lenke/gremlin/T.label'),
  key: familySymbol<'Token'>('@lenke/gremlin/T.key'),
  value: familySymbol<'Token'>('@lenke/gremlin/T.value'),
} as const;

export type Token = (typeof T)[keyof typeof T];

const TOKEN_TO_KIND: ReadonlyMap<symbol, 'id' | 'label' | 'key' | 'value'> = new Map([
  [T.id, 'id'],
  [T.label, 'label'],
  [T.key, 'key'],
  [T.value, 'value'],
]);

// Comparator symbols for `order().by(...)`. Pass alone (`by(Order.desc)`) for
// natural-order direction, or as the second arg (`by('age', Order.desc)`) to
// pair with a projection.
export const Order = {
  asc: familySymbol<'Order'>('@lenke/gremlin/Order.asc'),
  desc: familySymbol<'Order'>('@lenke/gremlin/Order.desc'),
} as const;

export type OrderSym = (typeof Order)[keyof typeof Order];

const ORDER_TO_DIR: ReadonlyMap<symbol, 'asc' | 'desc'> = new Map([
  [Order.asc, 'asc'],
  [Order.desc, 'desc'],
]);

/** The direction an `Order` symbol names, or `undefined` if it is not one. */
export const orderDirOf = (s: symbol): 'asc' | 'desc' | undefined => ORDER_TO_DIR.get(s);

// TinkerPop `Operator` merge functions for `sack(op).by(...)`: `newSack =
// op(currentSack, projectedValue)`. `assign` replaces; the rest fold numerically.
export const Operator = {
  assign: familySymbol<'Operator'>('@lenke/gremlin/Operator.assign'),
  sum: familySymbol<'Operator'>('@lenke/gremlin/Operator.sum'),
  mult: familySymbol<'Operator'>('@lenke/gremlin/Operator.mult'),
  min: familySymbol<'Operator'>('@lenke/gremlin/Operator.min'),
  max: familySymbol<'Operator'>('@lenke/gremlin/Operator.max'),
} as const;

export type OperatorSym = (typeof Operator)[keyof typeof Operator];

export const OPERATOR_TO_SACKOP: ReadonlyMap<symbol, SackOp> = new Map([
  [Operator.assign, 'assign'],
  [Operator.sum, 'sum'],
  [Operator.mult, 'mult'],
  [Operator.min, 'min'],
  [Operator.max, 'max'],
]);

// Scope of a barrier-like step. `global` (default) operates over the whole
// stream; `local` confines the operation to each traverser's value (typically
// when that value is itself a list/map).
//
// Wired through `take`/`skip`/`limit`/`range`/`tail` today. Other consumers
// (`count`/`sum`/`min`/`max`/`mean`/etc.) accept the symbol at the type
// level but don't yet branch on it — those are tracked in `GAPS.md`.
export const Scope = {
  global: Symbol.for('@lenke/gremlin/Scope.global'),
  local: Symbol.for('@lenke/gremlin/Scope.local'),
} as const;

// Map column selectors: `Column.keys` / `Column.values`. In `order(local).by(...)`
// they choose whether to sort a Map's entries by their key or value; in
// `select(Column.x)` they extract that column as a list.
export const Column = {
  keys: familySymbol<'Column'>('@lenke/gremlin/Column.keys'),
  values: familySymbol<'Column'>('@lenke/gremlin/Column.values'),
} as const;

export type ColumnSym = (typeof Column)[keyof typeof Column];

export const COLUMN_TO_KIND: ReadonlyMap<symbol, 'keys' | 'values'> = new Map([
  [Column.keys, 'keys'],
  [Column.values, 'values'],
]);

// Property cardinality, used with `property(Cardinality.X, key, value)`.
// `single` overwrites; `list` appends; `set` appends only if not present.
// Our storage is single-valued today; the cardinality is still threaded
// through the AST so a future multi-cardinality executor can pick it up.
export const Cardinality = {
  single: Symbol.for('@lenke/gremlin/Cardinality.single'),
  list: Symbol.for('@lenke/gremlin/Cardinality.list'),
  set: Symbol.for('@lenke/gremlin/Cardinality.set'),
} as const;

export type CardinalitySym = (typeof Cardinality)[keyof typeof Cardinality];

export const CARDINALITY_TO_KIND: ReadonlyMap<symbol, 'single' | 'list' | 'set'> = new Map([
  [Cardinality.single, 'single'],
  [Cardinality.list, 'list'],
  [Cardinality.set, 'set'],
]);

// Pop modes for `select`. Use as the optional first argument:
//   `select('a')`               // default — last value
//   `select(Pop.first, 'a')`    // first value
//   `select(Pop.all, 'a')`      // all values as a list
export const Pop = {
  first: Symbol.for('@lenke/gremlin/Pop.first'),
  last: Symbol.for('@lenke/gremlin/Pop.last'),
  all: Symbol.for('@lenke/gremlin/Pop.all'),
} as const;

export const POP_TO_STR: ReadonlyMap<symbol, 'first' | 'last' | 'all'> = new Map([
  [Pop.first, 'first'],
  [Pop.last, 'last'],
  [Pop.all, 'all'],
]);

// ---------- by() modulators ----------

// `by()` accepts:
//   - undefined  → identity (use the value as-is)
//   - string     → project by property name
//   - Token (T.x) → project to id / label / key / value
//   - Order.asc/desc → identity projection with comparator direction (order only)
//   - StepFn     → run a single-step sub-traversal (e.g. `count()`)
//   - Plan       → run a multi-step sub-traversal built via `traversal(...)`
export type ByModulator = string | Token | ColumnSym | OrderSym | StepFn | Plan;

export type ByableStep<S extends Step> = StepFn & {
  readonly by: (modulator?: ByModulator, comparator?: OrderSym) => ByableStep<S>;
};

// ---------- Predicate / sub-plan introspection ----------

export const isPlan = (x: unknown): x is Plan =>
  typeof x === 'object' && x !== null && 'steps' in x;

/**
 * True if `x` is a sub-plan in either accepted shape (a `traversal(...)` or a
 * branded `StepFn`). Used by combinators that *also* accept a closure
 * (`map`, `filter`, `flatMap`, `sideEffect`) to route to the sub-plan branch.
 */
export const isSubPlan = (x: unknown): x is SubPlan => isPlan(x) || isStepFn(x);

/**
 * Coerce either form to a `Plan`. The runtime accepts both; this is the one
 * conversion point so call sites stay shape-agnostic.
 */
export const buildPlan = (sub: SubPlan): Plan => (isPlan(sub) ? sub : sub({ steps: [] }));

/**
 * Predicate-vs-value detection. Predicates are objects with an `op`
 * discriminant; raw values are anything else. Used by `has`, `not`, etc.
 * to dispatch on argument shape.
 */
export const isPredicate = (x: unknown): x is Predicate =>
  typeof x === 'object' && x !== null && 'op' in x;

// ---------- by() and ByableStep machinery ----------

export const toBy = (modulator: ByModulator | undefined, comparator?: OrderSym): By => {
  const direction = comparator !== undefined ? ORDER_TO_DIR.get(comparator) : undefined;

  // `by(Order.asc/desc)` alone — identity projection, comparator-only.
  if (typeof modulator === 'symbol' && ORDER_TO_DIR.has(modulator)) {
    return { kind: 'identity', direction: ORDER_TO_DIR.get(modulator) };
  }

  if (modulator === undefined) {
    return { kind: 'identity', direction };
  }

  if (typeof modulator === 'string') {
    return { kind: 'key', key: modulator, direction };
  }

  if (typeof modulator === 'symbol') {
    const tokenKind = TOKEN_TO_KIND.get(modulator);

    if (tokenKind) {
      return { kind: 'token', token: tokenKind, direction };
    }

    const columnKind = COLUMN_TO_KIND.get(modulator);

    if (columnKind) {
      return { kind: 'column', column: columnKind, direction };
    }

    throw new LenkeError(`Unrecognized symbol: ${String(modulator)}`, {
      code: ErrorCode.Unsupported,
    });
  }

  if (isPlan(modulator)) {
    return { kind: 'traversal', plan: modulator, direction };
  }

  return { kind: 'traversal', plan: buildPlan(modulator), direction };
};

// Build a step node with optional `bys` and a `.by(...)` method that appends
// to it. The factory `make` rebuilds the node from a new bys array so each
// call is pure.
export const makeByable = <S extends Step & { bys?: readonly By[] }>(
  make: (bys: readonly By[] | undefined) => S,
  bys?: readonly By[],
): ByableStep<S> => {
  const fn: StepFn = appendStep(make(bys));

  return Object.assign(fn, {
    by: (modulator?: ByModulator, comparator?: OrderSym) =>
      makeByable(make, [...(bys ?? []), toBy(modulator, comparator)]),
  });
};

// ---------- Scope token translation ----------

/**
 * Translate a `Scope.local` / `Scope.global` Symbol to its string token.
 * Used by cardinality steps that accept Scope as a first-arg overload.
 * Symbols don't survive `JSON.stringify`, so the AST stores the token
 * string — this is the conversion point.
 */
export const scopeTokenOf = (s: symbol): 'global' | 'local' => {
  if (s === Scope.local) {
    return 'local';
  }

  if (s === Scope.global) {
    return 'global';
  }

  throw new LenkeError('Expected Scope.local or Scope.global', { code: ErrorCode.Unsupported });
};

// ---------- Re-exports the step files need from ast ----------

export { appendStep, STEP_FN } from '../ast.js';
export type {
  By,
  FilterClosure,
  FlatMapClosure,
  ID,
  MapClosure,
  Plan,
  Predicate,
  ReducerClosure,
  SideEffectClosure,
  Step,
} from '../ast.js';
