import { type Graph, isElement } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { By } from '../ast.js';
import { compareTotal } from '../predicates.js';
import {
  evalBy,
  extend,
  isSliceable,
  type RunContext,
  startTraverser,
  type Traverser,
} from './runtime.js';

/** Coerce a numeric-aggregate element, throwing on a non-number (TinkerPop's
 * `sum`/`mean` require `Number`s and raise on anything else, rather than
 * silently coercing to `NaN`). `null` is filtered by the caller. */
const asNumber = (v: unknown): number => {
  if (typeof v === 'number') {
    return v;
  }

  throw new LenkeError(`numeric aggregation requires a number, got ${typeof v}`, {
    code: ErrorCode.InvalidValue,
  });
};

export const aggregateNumber = function* (
  stream: Iterable<Traverser<unknown>>,
  kind: 'sum' | 'mean',
): Iterable<Traverser<unknown>> {
  let sum = 0;
  let count = 0;
  let sawNonNull = false;

  for (const t of stream) {
    if (t.value == null) {
      continue;
    }

    sawNonNull = true;
    sum += asNumber(t.value);
    count++;
  }

  if (!sawNonNull) {
    yield startTraverser(null);

    return;
  }

  yield startTraverser(kind === 'sum' ? sum : sum / count);
};

// min()/max() reduce over the engine's TOTAL ORDER (the same one `order()` and the native
// engine use — see the numeric/NaN comparison policy), so they work on ANY value, not just
// numbers: strings compare lexicographically (`min` of names → the first alphabetically),
// NaN sorts greatest, and a mixed stream is decided by the deterministic type rank. This
// matches the native engine AND TinkerPop (which orders any Comparable); an earlier
// numeric-only guard faulted on `values('name').min()`, diverging from both.
export const aggregateComparable = function* (
  stream: Iterable<Traverser<unknown>>,
  kind: 'min' | 'max',
): Iterable<Traverser<unknown>> {
  let best: unknown;
  let sawNonNull = false;

  for (const t of stream) {
    if (t.value == null) {
      continue;
    }

    if (!sawNonNull) {
      best = t.value;
      sawNonNull = true;
    } else {
      const c = compareTotal(t.value, best);

      if (kind === 'min' ? c < 0 : c > 0) {
        best = t.value;
      }
    }
  }

  if (!sawNonNull) {
    yield startTraverser(null);

    return;
  }

  yield startTraverser(best);
};

export const countStep = function* (
  stream: Iterable<Traverser<unknown>>,
): Iterable<Traverser<unknown>> {
  let n = 0;

  for (const _ of stream) {
    n++;
  }

  yield startTraverser(n);
};

// ---------- Scope.local aggregations (per-traverser) ----------
//
// Each `*Local` helper computes the aggregate of a single traverser's
// iterable VALUE (post-`fold()` array, list-cardinality projection, etc.).
// For non-iterable values, the traverser is treated as a one-element
// sequence — `count` returns 1, `sum`/`min`/`max`/`mean` return the value
// itself when numeric. Empty iterables return `null` to mirror the
// global-scope behavior on empty streams.

const elementsOf = (v: unknown): unknown[] => (isSliceable(v) ? [...v] : [v]);

export const countLocal = (v: unknown): number => elementsOf(v).length;

export const sumLocal = (v: unknown): number | null => {
  const items = elementsOf(v).filter((x) => x != null);

  if (items.length === 0) {
    return null;
  }

  return items.reduce<number>((s, x) => s + asNumber(x), 0);
};

export const meanLocal = (v: unknown): number | null => {
  const items = elementsOf(v).filter((x) => x != null);

  if (items.length === 0) {
    return null;
  }

  return items.reduce<number>((s, x) => s + asNumber(x), 0) / items.length;
};

export const minLocal = (v: unknown): unknown => reduceComparable(elementsOf(v), 'min');

export const maxLocal = (v: unknown): unknown => reduceComparable(elementsOf(v), 'max');

const reduceComparable = (items: readonly unknown[], kind: 'min' | 'max'): unknown => {
  let best: unknown;
  let saw = false;

  for (const x of items) {
    if (x == null) {
      continue;
    }

    if (!saw) {
      best = x;
      saw = true;
      continue;
    }

    const c = compareTotal(x, best);

    if (kind === 'min' ? c < 0 : c > 0) {
      best = x;
    }
  }

  return saw ? best : null;
};

// Sort `items` IN PLACE by projecting each through the `bys` (first = primary
// key, rest = tie-breakers, in order). Per-by direction prefers the By's
// `direction`; falls back to the step-level `desc` (legacy `order({desc})` form).
const sortByBys = <T>(
  items: T[],
  project: (item: T) => unknown,
  bys: readonly By[],
  desc: boolean,
  graph: Graph,
  ctx: RunContext,
): void => {
  const dirs = bys.map((by) => by.direction ?? (desc ? 'desc' : 'asc'));
  const keyed = items.map((item) => ({
    item,
    keys: bys.map((by) => evalBy(by, project(item), graph, ctx)),
  }));

  // A raw graph element has no natural order. The parse-time check in `checkStep`
  // rejects a pure-element `order()`, but an element still reaches here in a MIXED
  // stream (`both(...).inject(1).order()`) whose frontier is no longer "element", or
  // via `by(<element-valued traversal>)`. Fault at runtime to match the native engine.
  if (keyed.some((k) => k.keys.some(isElement))) {
    throw new LenkeError(
      `order() over graph elements is not supported — elements have no natural order; ` +
        `use order().by('<key>')`,
      { code: ErrorCode.Syntax },
    );
  }

  keyed.sort((a, b) => {
    for (let i = 0; i < bys.length; i++) {
      const c = compareTotal(a.keys[i], b.keys[i]) * (dirs[i] === 'desc' ? -1 : 1);

      if (c !== 0) {
        return c;
      }
    }

    return 0;
  });

  for (let i = 0; i < items.length; i++) {
    items[i] = keyed[i].item;
  }
};

// `order` (global scope) materializes the stream, sorts the traversers by their
// value, then re-yields. Boundary step.
export const orderStep = function* (
  stream: Iterable<Traverser<unknown>>,
  bys: readonly By[],
  desc: boolean,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  const items = [...stream];
  sortByBys(items, (t) => t.value, bys, desc, graph, ctx);

  yield* items;
};

// `order(Scope.local)` sorts WITHIN each traverser's value instead of across the
// stream: a Map's entries — by their VALUE by default, or by their KEY when a
// `by(keys)` Column selector is given (the `groupCount().order(local)` top-N idiom;
// `by(values)` is the explicit default) — or a list's elements. A non-column
// by-modulator projects the sort key off each entry-value / element (identity → the
// value itself). A scalar value has nothing to sort → passes through unchanged.
export const orderLocalStep = function* (
  stream: Iterable<Traverser<unknown>>,
  bys: readonly By[],
  desc: boolean,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  // A Map entry's sort key for one `by`: `by(keys)` → the entry key, `by(values)` →
  // the entry value, any other by → projected off the value (matches native).
  const entryKey = (by: By, entry: [unknown, unknown]): unknown => {
    if (by.kind === 'column') {
      return by.column === 'keys' ? entry[0] : entry[1];
    }

    return evalBy(by, entry[1], graph, ctx);
  };

  for (const t of stream) {
    const v = t.value;

    if (v instanceof Map) {
      const dirs = bys.map((by) => by.direction ?? (desc ? 'desc' : 'asc'));
      const keyed = [...v.entries()].map((entry) => ({
        entry,
        keys: bys.map((by) => entryKey(by, entry)),
      }));

      keyed.sort((a, b) => {
        for (let i = 0; i < bys.length; i++) {
          const c = compareTotal(a.keys[i], b.keys[i]) * (dirs[i] === 'desc' ? -1 : 1);

          if (c !== 0) {
            return c;
          }
        }

        return 0;
      });

      yield extend(t, new Map(keyed.map((k) => k.entry)));
    } else if (isSliceable(v)) {
      const items = [...v];
      sortByBys(items, (x) => x, bys, desc, graph, ctx);

      yield extend(t, items);
    } else {
      yield t;
    }
  }
};
