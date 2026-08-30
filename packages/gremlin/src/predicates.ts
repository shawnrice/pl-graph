import { fromTaggedJson, isTemporal, temporalCmpTotal, temporalRelCmp } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Orderable, Predicate } from './ast.js';

/**
 * Accept a tagged temporal literal (`{"@date": "2020-01-01"}`) anywhere a stored
 * temporal could appear.
 *
 * The graph stores temporals as class instances, but they arrive from users and
 * from the text dialect in the tagged wire form — the same form `toJSON` emits
 * and GQL params take. Without lifting, the two never met: `eq` compared a plain
 * object to a `LocalDate` by `===` and **silently matched nothing**, and the
 * ordering predicates never reached the temporal branch of `compareValues`
 * (`isTemporal` only recognizes instances) so they threw `E_INVALID_VALUE`. A
 * silent empty result is the worse of the two — it ships.
 *
 * Lifted once at predicate construction rather than per candidate value.
 */
const lift = (v: unknown): unknown => fromTaggedJson(v) ?? v;

/**
 * `lift` for the ordering predicates, whose AST slot is `Orderable`. The cast is
 * the honest boundary: callers may pass anything, and a value that is not
 * actually comparable is rejected at match time by `compareValues`, which throws
 * `cannot order X with Y` rather than coercing to a misleading boolean. Checking
 * here instead would move that error from evaluation to construction and diverge
 * from the numeric/string behaviour.
 */
const liftOrd = (v: unknown): Orderable => lift(v) as Orderable;

/**
 * Equality matching the GQL engine: two temporals are equal by value — same kind
 * and same instant/components — not by reference. Everything else keeps `===`.
 */
const valueEq = (a: unknown, b: unknown): boolean =>
  isTemporal(a) && isTemporal(b) ? temporalCmpTotal(a, b) === 0 : a === b;

const typeName = (v: unknown): string => {
  if (v === null || v === undefined) {
    return 'null';
  }

  if (Array.isArray(v)) {
    return 'a list';
  }

  return typeof v === 'object' ? 'an element' : typeof v;
};

const cmpOrd = (x: number | string, y: number | string): number => {
  if (x < y) {
    return -1;
  }

  return x > y ? 1 : 0;
};

/**
 * Order two values the way TinkerPop's `Comparable` does: numbers with numbers,
 * strings with strings, booleans with booleans. Comparing genuinely
 * incomparable types — a number with a string, an element with a scalar —
 * throws (mirroring TinkerPop's `ClassCastException`) rather than coercing to a
 * misleading boolean. Returns a negative / zero / positive number.
 */
/**
 * The SORT/aggregate comparator — a genuine total order, where `compareValues`
 * is the PREDICATE one and is deliberately partial.
 *
 * The two must not be the same function. `compareValues` returns `NaN` for a
 * NaN operand so that `is(gt(0))` filters it, which is right for a predicate;
 * but `Array.prototype.sort` reads a `NaN` comparator result as 0, making the
 * order non-total. We own the comparator and not the sort ALGORITHM (V8's
 * `Array.sort` is not Rust's `slice::sort_by`), so that let the algorithm leak:
 * `values('m').math('sqrt _').order()` over ten values returned them scattered
 * here (`[NaN,2,NaN,3,4,NaN,5,NaN,1,NaN]`) while native returned every NaN
 * first — same input, two different answers.
 *
 * NaN is the GREATEST value and equals itself, either sign, so `max()` keeps it
 * and `min()` never picks one. Mirrors the Rust `gcmp_total`.
 */
export const compareTotal = (a: unknown, b: unknown): number => {
  // NULLS FIRST, and two nulls equal. `compareValues` has no null arm at all —
  // it falls through to `cannot order null with null` — so ordering a stream
  // with one missing property THREW here while the Rust engine sorted it, which is
  // a byte-identity break on an ordinary traversal (`values('k').order()` over a
  // vertex that lacks `k`, or anything after `path().id()`).
  //
  // First, not last: that is TinkerPop's, verbatim from
  // `GremlinValueComparator.ORDERABILITY` —
  //
  //     // nulls first
  //     if (f == null || s == null)
  //         return f == s ? 0 : f == null ? -1 : 1;
  //
  // and it is what `gval_type_rank` already does by ranking `Null` at 0. GQL
  // sorts nulls LAST, which is the ISO contract for a different language; the
  // two do not have to agree and here they must not.
  //
  // The PREDICATE comparator keeps throwing, which is also TinkerPop: it splits
  // Comparability ("throws type errors … for cross-type comparison (including
  // nulltype)") from Orderability. That split is exactly this pair of functions.
  if (a === null || a === undefined || b === null || b === undefined) {
    const aNull = a === null || a === undefined;
    const bNull = b === null || b === undefined;

    if (aNull && bNull) {
      return 0;
    }

    return aNull ? -1 : 1;
  }

  if (typeof a === 'number' && typeof b === 'number') {
    const aNaN = Number.isNaN(a);
    const bNaN = Number.isNaN(b);

    if (aNaN || bNaN) {
      if (aNaN && bNaN) {
        return 0;
      }

      return aNaN ? 1 : -1;
    }
  }

  // A TOTAL order across types — order()/min()/max() never throw on a mixed stream
  // (TinkerPop's Orderability, and the engine's deliberate total order for sort/min/max
  // so both stay deterministic + byte-identical). Different types sort by a fixed rank
  // (number < string < boolean < temporal < list < …, matching the Rust `type_rank`);
  // same-type pairs compare normally. Only `sum()`/`mean()` (numeric) and the ordering
  // PREDICATES fault/filter — never the sort.
  const ra = typeRank(a);
  const rb = typeRank(b);

  if (ra !== rb) {
    return ra - rb;
  }

  // Same rank but still incomparable (cross-kind temporals) → treat as equal for a
  // stable total order rather than throwing.
  return cmpSameType(a, b) ?? 0;
};

/**
 * The cross-type SORT rank, matching the Rust engine's `type_rank` (Num < Str < Bool <
 * Temporal < List < compound) so `order()`/min()/max() over a mixed column agree
 * byte-for-byte. NULL is handled separately (sorts first).
 */
const typeRank = (v: unknown): number => {
  if (typeof v === 'number') {
    return 0;
  }

  if (typeof v === 'string') {
    return 1;
  }

  if (typeof v === 'boolean') {
    return 2;
  }

  if (isTemporal(v)) {
    return 3;
  }

  return Array.isArray(v) ? 4 : 5;
};

/**
 * The SAME-TYPE ordering: a negative/zero/positive number, `NaN` for a NaN
 * operand (unordered), or `null` when the two are genuinely INCOMPARABLE (a
 * number with a string, an element with a scalar, cross-kind temporals). The
 * callers split on `null`: `compareValues` throws (for `max`/`min`), the
 * predicate comparator returns `NaN` so the ordering predicate simply does not
 * match (TinkerPop FILTERS a cross-type `has()`/`is()`/`where()`, it does not
 * throw).
 */
const cmpSameType = (a: unknown, b: unknown): number | null => {
  if (typeof a === 'number' && typeof b === 'number') {
    // NaN is unordered: in JS every comparison with NaN is false, so a NaN
    // operand must satisfy no ordering predicate (`> 0`, `>= 0`, `< 0`, `<= 0`
    // all become false). Returning NaN propagates that — matching Rust's
    // `partial_cmp → None → filtered`. `cmpOrd` would wrongly return 0 here,
    // leaking NaN through `gte`/`lte`.
    if (Number.isNaN(a) || Number.isNaN(b)) {
      return Number.NaN;
    }

    return cmpOrd(a, b);
  }

  if (typeof a === 'string' && typeof b === 'string') {
    return cmpOrd(a, b);
  }

  if (typeof a === 'boolean' && typeof b === 'boolean') {
    if (a === b) {
      return 0;
    }

    return a ? 1 : -1;
  }

  // Temporals of the same instant kind (date/datetime) order chronologically;
  // durations and cross-kind pairs are not orderable.
  if (isTemporal(a) && isTemporal(b)) {
    const c = temporalRelCmp(a, b);

    if (c !== null) {
      return c;
    }
  }

  return null; // genuinely incomparable
};

/**
 * Order two values the way TinkerPop's `Comparable` does. Genuinely incomparable
 * types THROW (mirroring the `ClassCastException` `max()`/`min()` raise). The
 * ordering PREDICATES do NOT use this — they filter instead (`predCmp`).
 */
export const compareValues = (a: unknown, b: unknown): number => {
  const c = cmpSameType(a, b);

  if (c !== null) {
    return c;
  }

  throw new LenkeError(`cannot order ${typeName(a)} with ${typeName(b)}`, {
    code: ErrorCode.InvalidValue,
  });
};

/**
 * The comparator for ORDERING PREDICATES (`gt`/`gte`/`lt`/`lte`/`between`/…): a
 * cross-type pair is `NaN`, so every comparison against it is false and the
 * predicate FILTERS the value out rather than throwing — `g.V().has('name',
 * gte(0))` returns nothing, exactly as TinkerPop (verified on createModern()).
 */
const predCmp = (a: unknown, b: unknown): number => cmpSameType(a, b) ?? Number.NaN;

// Compile each regex pattern once (the predicate is re-applied per value). The
// pattern is validated at build time in `regex()`, so this never throws here.
// NB: JS `RegExp` is backtracking, so a pathological pattern over a long input
// can still be slow (ReDoS) — the same exposure as TinkerPop's Java `Pattern`.
// There's no native regex timeout in JS; the mitigation is to not run untrusted
// patterns, so we don't reject patterns TinkerPop would accept.
const regexCache = new Map<string, RegExp>();
const compiledRegex = (pattern: string): RegExp => {
  let re = regexCache.get(pattern);

  if (!re) {
    if (regexCache.size >= 1000) {
      regexCache.clear(); // bound memory; patterns are typically few
    }

    re = new RegExp(pattern);
    regexCache.set(pattern, re);
  }

  return re;
};

export const eq = (value: unknown): Predicate => ({ op: 'eq', value: lift(value) });
export const neq = (value: unknown): Predicate => ({ op: 'neq', value: lift(value) });
export const gt = (value: unknown): Predicate => ({ op: 'gt', value: liftOrd(value) });
export const gte = (value: unknown): Predicate => ({ op: 'gte', value: liftOrd(value) });
export const lt = (value: unknown): Predicate => ({ op: 'lt', value: liftOrd(value) });
export const lte = (value: unknown): Predicate => ({ op: 'lte', value: liftOrd(value) });
// Half-open [min, max). Matches Gremlin's `P.between` semantics.
export const between = (min: unknown, max: unknown): Predicate => ({
  op: 'between',
  min: liftOrd(min),
  max: liftOrd(max),
});

// Strict open (min, max). Matches Gremlin's `P.inside`.
export const inside = (min: unknown, max: unknown): Predicate => ({
  op: 'inside',
  min: liftOrd(min),
  max: liftOrd(max),
});

// Strict complement: value < min OR value > max. Matches `P.outside`.
export const outside = (min: unknown, max: unknown): Predicate => ({
  op: 'outside',
  min: liftOrd(min),
  max: liftOrd(max),
});
export const within = (...values: readonly unknown[]): Predicate => ({
  op: 'within',
  values: values.map(lift),
});
export const without = (...values: readonly unknown[]): Predicate => ({
  op: 'without',
  values: values.map(lift),
});
export const startsWith = (value: string): Predicate => ({ op: 'startsWith', value });
// TextP-style string predicates.
export const endingWith = (value: string): Predicate => ({ op: 'endingWith', value });
export const containing = (value: string): Predicate => ({ op: 'containing', value });
export const notContaining = (value: string): Predicate => ({ op: 'notContaining', value });
export const regex = (value: string): Predicate => {
  // Validate the pattern up front so an invalid regex is a clean build-time
  // error rather than an unwrapped `SyntaxError` thrown mid-stream per value.
  try {
    void new RegExp(value);
  } catch (cause) {
    throw new LenkeError(`regex(): invalid pattern ${JSON.stringify(value)}`, {
      code: ErrorCode.Syntax,
      cause,
    });
  }

  return { op: 'regex', value };
};

/**
 * Evaluate a predicate against a value. Used by executors; not part of the
 * AST surface.
 */
export const matches = (pred: Predicate, value: unknown): boolean => {
  switch (pred.op) {
    case 'eq':
      return valueEq(value, pred.value);
    case 'neq':
      return !valueEq(value, pred.value);
    // Ordering predicates: a missing OR incomparable value is filtered out
    // (false), never an error — `predCmp` returns NaN for a cross-type pair, and
    // every comparison against NaN is false. TinkerPop's has()/is()/where() do the
    // same (a cross-type `has('name', gte(0))` matches nothing).
    case 'gt':
      return value != null && predCmp(value, pred.value) > 0;
    case 'gte':
      return value != null && predCmp(value, pred.value) >= 0;
    case 'lt':
      return value != null && predCmp(value, pred.value) < 0;
    case 'lte':
      return value != null && predCmp(value, pred.value) <= 0;
    case 'between':
      return value != null && predCmp(value, pred.min) >= 0 && predCmp(value, pred.max) < 0;
    case 'inside':
      return value != null && predCmp(value, pred.min) > 0 && predCmp(value, pred.max) < 0;
    case 'outside':
      return value != null && (predCmp(value, pred.min) < 0 || predCmp(value, pred.max) > 0);
    case 'within':
      return pred.values.some((x) => valueEq(value, x));
    case 'without':
      return !pred.values.some((x) => valueEq(value, x));
    case 'startsWith':
      return typeof value === 'string' && value.startsWith(pred.value);
    case 'endingWith':
      return typeof value === 'string' && value.endsWith(pred.value);
    case 'containing':
      return typeof value === 'string' && value.includes(pred.value);
    case 'notContaining':
      return typeof value === 'string' && !value.includes(pred.value);
    case 'regex':
      return typeof value === 'string' && compiledRegex(pred.value).test(value);
    case 'not':
      return !matches(pred.predicate, value);
  }
};
