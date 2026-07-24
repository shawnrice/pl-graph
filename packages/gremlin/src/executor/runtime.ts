// Shared scaffolding for the executor. Per-category step-impl files import
// from here for the runtime types (`Traverser`, `RunContext`), traverser
// primitives, type guards, modulator evaluation, and the small set of
// generators (`mapTraverser`/`filterStream`/etc.) that show up in many
// places.
//
// What's NOT here: per-step implementations (those live in their category
// file), and the dispatcher (`applyStep` + `applyPlanToStream`, in
// `./dispatch.ts`). Those depend on each other recursively, so they stay
// adjacent.

import type { Edge, Graph, Vertex } from '@lenke/core';

import type { By, Plan, Step } from '../ast.js';
// Cycle: `runtime.ts` ↔ `dispatch.ts`. ESM handles this safely because
// neither module dereferences the other at init time — `applyPlanToStream`
// is only called from inside `evalBy`, by which time both modules' exports
// are bound. Same shape as `applyStep`/`applyPlanToStream` mutual-recursion.
import { applyPlanToStream } from './dispatch.js';

// ---------- Traverser ----------

/**
 * Runtime traverser. Carries the current value and the path of values seen
 * to reach it. Path is immutable — branching produces fresh paths via
 * structural sharing.
 */
export type Traverser<T> = {
  readonly value: T;
  readonly path: readonly unknown[];
  /** Number of repeat-body iterations entered so far. 0 outside any repeat. */
  readonly loopCount: number;
  /**
   * Labeled positions tagged via `as(label)`, recalled by `select(label)`.
   * A label can accumulate multiple values inside iterative steps (e.g.
   * `repeat(out().as('a'))`); `select(Pop.first|last|all, 'a')` picks how
   * to read. The default `select('a')` returns the last tagged value.
   */
  readonly tags: ReadonlyMap<string, readonly unknown[]>;
  /**
   * The per-traverser sack (TinkerPop `sack()`). LAZY: `undefined` until a
   * `sack(op)` write sets it; a read before that returns the `withSack` default
   * (on {@link RunContext}) without storing. With no `withSack`, no sack ever
   * exists on any traverser.
   */
  readonly sack?: unknown;
};

export const emptyTags: ReadonlyMap<string, readonly unknown[]> = new Map();

export const isEmptyPlan = (plan: Plan): boolean => plan.steps.length === 0;

/**
 * Shared empty-path sentinel for traversers in a run that never reads the path.
 * `extend` checks identity against it to propagate "not tracking" down a chain
 * for free — see {@link planReadsPath}. Frozen so an accidental mutation throws
 * rather than corrupting every traverser that shares it.
 */
const NO_PATH: readonly unknown[] = Object.freeze([]);

/**
 * Step kinds that read a traverser's accumulated path. The first five touch
 * `t.path` directly; the closure-bearing kinds hand it to user code via
 * `closureView`, so we must conservatively assume they read it. This list is
 * exhaustive against every `.path` read in the executor — keep it in sync if a
 * new path-consuming step is added (a missed entry would silently drop paths).
 */
const PATH_DEPENDENT_KINDS: ReadonlySet<string> = new Set([
  'path',
  'tree',
  'simplePath',
  'cyclicPath',
  'otherV',
  'mapFn',
  'flatMapFn',
  'filterFn',
  'sideEffectFn',
  'foldFn',
]);

/** The sub-plans a step embeds, by their known plan-bearing field names. */
const subPlansOf = (step: Step): Plan[] => {
  const plans: Plan[] = [];
  const fields = step as Record<string, unknown>;

  for (const name of ['body', 'until', 'emit', 'plan', 'test', 'thenPlan', 'elsePlan']) {
    if (fields[name]) {
      plans.push(fields[name] as Plan);
    }
  }

  if (Array.isArray(fields.plans)) {
    plans.push(...(fields.plans as Plan[]));
  }

  if (Array.isArray(fields.bys)) {
    for (const by of fields.bys as By[]) {
      if (by.kind === 'traversal') {
        plans.push(by.plan);
      }
    }
  }

  return plans;
};

/**
 * Does any step anywhere in the plan tree read the traverser path? Recurses
 * only through plan-bearing fields (never `inject` values, predicate operands,
 * or closures), so it can't loop on or be fooled by user data. When this is
 * false the run skips path bookkeeping entirely; when in doubt it returns true,
 * so the optimization only ever removes provably-unobservable work.
 */
export const planReadsPath = (plan: Plan): boolean =>
  plan.steps.some(
    (step) => PATH_DEPENDENT_KINDS.has(step.kind) || subPlansOf(step).some(planReadsPath),
  );

/** A `{ key, value }` property element (from `.properties(k)`). */
export type PropertyElement = { key: string; value: unknown };

/**
 * A property element stays exactly `{ key, value }`; its owning vertex/edge is
 * tracked HERE (a side WeakMap keyed by the element's identity), not as a field
 * on the object — so `.drop()` can delete the property while nothing leaks into
 * projections, serialization, or structural equality. The WeakMap lets the
 * element and its owning-link be GC'd together once the traversal drops them.
 */
const propertyOwners = new WeakMap<object, Vertex | Edge>();

/** Mint a property element and remember its owner (called by `.properties`). */
export const ownedProperty = (
  owner: Vertex | Edge,
  key: string,
  value: unknown,
): PropertyElement => {
  const el: PropertyElement = { key, value };
  propertyOwners.set(el, owner);

  return el;
};

/** The owning vertex/edge of a property element, or `undefined` if `x` isn't one. */
export const propertyOwner = (x: unknown): Vertex | Edge | undefined =>
  typeof x === 'object' && x !== null ? propertyOwners.get(x) : undefined;

export const startTraverser = <T>(value: T, tracksPath = true): Traverser<T> => ({
  value,
  path: tracksPath ? [value] : NO_PATH,
  loopCount: 0,
  tags: emptyTags,
});

export const extend = <T>(prev: Traverser<unknown>, value: T): Traverser<T> => ({
  value,
  // Propagate the not-tracking sentinel by identity; otherwise grow the path.
  path: prev.path === NO_PATH ? NO_PATH : [...prev.path, value],
  loopCount: prev.loopCount,
  tags: prev.tags,
  // Propagate the sack (split-on-branch = clone). Only defined once a sack write
  // has run, so the common no-sack path carries `undefined` (no cost).
  ...(prev.sack === undefined ? {} : { sack: prev.sack }),
});

export const incLoops = <T>(t: Traverser<T>): Traverser<T> => ({
  ...t,
  loopCount: t.loopCount + 1,
});

// ---------- Type guards ----------

export const isVertex = (x: unknown): x is Vertex =>
  typeof x === 'object' && x !== null && 'id' in x && !('from' in x);

export const isEdge = (x: unknown): x is Edge =>
  typeof x === 'object' && x !== null && 'from' in x && 'to' in x;

export const firstLabel = (s: ReadonlySet<string>): string | undefined => {
  for (const l of s) {
    return l;
  }

  return undefined;
};

// ---------- Run context ----------

/**
 * Per-run context — currently just the side-effects bag for
 * `aggregate(key)` / `cap(key)`. Passed by reference so all steps in a
 * single `run()` share the same bag.
 */
export type RunContext = {
  readonly sideEffects: Map<string, unknown[]>;
  /**
   * Subgraph side-effects keyed by name: `subgraph(key)` accumulates matching
   * edges (and their endpoints) into a fresh `Graph` here, which `cap(key)`
   * returns. Kept separate from {@link sideEffects} because the value is a graph,
   * not an appended list.
   */
  readonly subgraphs: Map<string, Graph>;
  /**
   * Whether this run tracks traverser paths. Computed once from the plan via
   * {@link planReadsPath}; defaults to `true` so any context built without it
   * (e.g. sub-plan evaluation) is always correct, just unoptimized.
   */
  readonly tracksPath: boolean;
  /**
   * The `withSack(init)` default, boxed so `undefined` means "no sack
   * configured" — then `sack()` faults and no per-traverser sack is created.
   * Mutable: set once by the leading `withSack` step. This is the laziness gate.
   */
  sackInit?: { readonly value: unknown };
};

export const newContext = (tracksPath = true): RunContext => ({
  sideEffects: new Map(),
  subgraphs: new Map(),
  tracksPath,
});

/**
 * Project a runtime `Traverser` plus the run's side-effects into the
 * read-only view that user closures see. Built lazily per-call (rather than
 * stored on every traverser) because side-effects are per-run, not per-
 * traverser; threading the same Map reference avoids per-traverser overhead.
 */
export const closureView = (
  t: Traverser<unknown>,
  ctx: RunContext,
): {
  value: unknown;
  path: readonly unknown[];
  loopCount: number;
  tags: ReadonlyMap<string, readonly unknown[]>;
  sideEffects: ReadonlyMap<string, readonly unknown[]>;
} => ({
  value: t.value,
  path: t.path,
  loopCount: t.loopCount,
  tags: t.tags,
  sideEffects: ctx.sideEffects,
});

// ---------- Stream-of-traversers helpers (used by many steps) ----------

export const unwrap = function* (stream: Iterable<Traverser<unknown>>): Iterable<unknown> {
  for (const t of stream) {
    yield t.value;
  }
};

export const hasAny = (stream: Iterable<Traverser<unknown>>): boolean => {
  for (const _ of stream) {
    return true;
  }

  return false;
};

export const mapTraverser = function* (
  stream: Iterable<Traverser<unknown>>,
  fn: (v: unknown, t: Traverser<unknown>) => unknown,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    yield extend(t, fn(t.value, t));
  }
};

export const filterStream = function* (
  stream: Iterable<Traverser<unknown>>,
  predicate: (v: unknown) => boolean,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    if (predicate(t.value)) {
      yield t;
    }
  }
};

export const filterTraverser = function* (
  stream: Iterable<Traverser<unknown>>,
  predicate: (t: Traverser<unknown>) => boolean,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    if (predicate(t)) {
      yield t;
    }
  }
};

// ---------- Local-scope iteration (Scope.local consumers) ----------

/**
 * True for non-string objects that expose `Symbol.iterator`. Used by every
 * `Scope.local` consumer (slice family + aggregation family) to decide
 * whether to operate on a traverser's value as a sequence or pass it through
 * unchanged. Strings are deliberately excluded — TinkerPop's `Scope.local`
 * doesn't treat a string as a sequence of characters.
 */
export const isSliceable = (v: unknown): v is Iterable<unknown> => {
  if (v === null || v === undefined || typeof v === 'string') {
    return false;
  }

  return typeof (v as { [Symbol.iterator]?: unknown })[Symbol.iterator] === 'function';
};

// ---------- Tag recall (select / where-compare / addE) ----------

export type Pop = 'first' | 'last' | 'all';

export const recallTag = (
  tags: ReadonlyMap<string, readonly unknown[]>,
  label: string,
  pop: Pop,
): { ok: true; value: unknown } | { ok: false } => {
  const list = tags.get(label);

  if (!list || list.length === 0) {
    return { ok: false };
  }

  if (pop === 'first') {
    return { ok: true, value: list[0] };
  }

  if (pop === 'last') {
    return { ok: true, value: list[list.length - 1] };
  }

  return { ok: true, value: [...list] };
};

// ---------- Tuple key (dedupe / group across multiple labels) ----------

// Build a stable string key from a tuple of values. Vertices/edges are keyed
// by `id` (graph-unique); other values fall through to JSON. NUL separates
// fields so distinct tuples can't collide via concatenation.
export const tupleKey = (parts: readonly unknown[]): string =>
  parts
    .map((p) => {
      if (isVertex(p) || isEdge(p)) {
        return `@${p.id}`;
      }

      return JSON.stringify(p) ?? 'undefined';
    })
    .join('\x00');

// ---------- by() modulator evaluation ----------

// `by` legacy fallback: if a step has no `bys`, reconstruct one from a legacy
// property-name field (e.g. `order.key`, `group.keyBy`).
export const keyToBy = (key: string | undefined): By =>
  key === undefined ? { kind: 'identity' } : { kind: 'key', key };

// Normalize legacy `key` field into a one-element `bys` array if `bys` is unset.
export const normalizeBys = (
  bys: readonly By[] | undefined,
  legacyKey: string | undefined,
): readonly By[] => {
  if (bys && bys.length > 0) {
    return bys;
  }

  if (legacyKey !== undefined) {
    return [{ kind: 'key', key: legacyKey }];
  }

  return [{ kind: 'identity' }];
};

// `T.id` / `T.label` / `T.key` / `T.value` projection.
export const projectToken = (token: 'id' | 'label' | 'key' | 'value', value: unknown): unknown => {
  if (token === 'id') {
    return isVertex(value) || isEdge(value) ? value.id : undefined;
  }

  if (token === 'label') {
    return isVertex(value) || isEdge(value) ? (firstLabel(value.labels) ?? null) : undefined;
  }

  // `T.key` / `T.value` apply to property objects of shape `{ key, value }`
  // (as produced by `properties()`). For anything else, undefined.
  if (typeof value === 'object' && value !== null && 'key' in value && 'value' in value) {
    return (value as { key: unknown; value: unknown })[token];
  }

  return undefined;
};

// Evaluate a `By` modulator against a single value. The traversal form runs
// the sub-plan with `value` as the starting traverser and projects to its
// first emitted result (or `undefined` if empty).
export const evalBy = (by: By, value: unknown, graph: Graph, ctx: RunContext): unknown => {
  switch (by.kind) {
    case 'identity':
      return value;
    case 'key':
      if (isVertex(value) || isEdge(value)) {
        return value.properties[by.key];
      }

      // `by('k')` over a Map (`group`/`groupCount`) or a `project()` row object
      // projects the value at that key — e.g. `project('name','age').order().by('age')`.
      // Without this the whole container reached the comparator ("cannot order …").
      if (value instanceof Map) {
        return value.get(by.key);
      }

      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        return (value as Record<string, unknown>)[by.key];
      }

      return value;
    case 'traversal': {
      const out = applyPlanToStream(by.plan, [startTraverser(value)], graph, ctx);

      for (const t of out) {
        return t.value;
      }

      return undefined;
    }
    case 'token':
      return projectToken(by.token, value);
    case 'column':
      // `Column.keys` / `Column.values` over a Map yields the list of its keys /
      // values (TinkerPop `select(Column)`); `order(local)` special-cases it to sort
      // a Map's entries and never routes through here. Non-map → identity.
      if (value instanceof Map) {
        return by.column === 'keys' ? [...value.keys()] : [...value.values()];
      }

      return value;
  }
};
