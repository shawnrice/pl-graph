// ISO scalar functions + three-valued (Kleene) logic + value helpers, extracted
// from the executor. Pure value operations (no back-dependency on compilation or
// matching), so this is a leaf module.
import {
  civilFromDays,
  DEFAULT_CONFIG,
  Duration,
  durationBetween,
  isRecord,
  isTemporal,
  LenkeRecord,
  LocalDate,
  LocalDateTime,
  LocalTime,
  Path,
  temporalArith,
  temporalCmpTotal,
  temporalParse,
  ZonedDateTime,
  ZonedTime,
} from '@lenke/core';
import { mathSign } from '@lenke/core';
import type { GraphLimits, Temporal } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { ArithOp, CompareOp, Expr } from '../ast.js';
// Shared element guards + ordering comparator live in the executor trunk; used
// lazily inside scalar fns, so this back-edge is a safe function-level cycle.
import { compareSort, isEdge, isElement, isVertex } from '../executor.js';

// --- three-valued logic & scalar helpers -------------------------------------

// ISO three-valued (Kleene) logic: `null` is UNKNOWN. A row is kept only when a
// predicate evaluates to exactly `true` (see callers comparing `=== true`).
export type Truth = boolean | null;
export const isNullish = (v: unknown): boolean => v === null || v === undefined;
export const asTruth = (v: unknown): Truth => (isNullish(v) ? null : Boolean(v));
export const not3 = (t: Truth): Truth => (t === null ? null : !t);
export const and3 = (a: Truth, b: Truth): Truth => {
  if (a === false || b === false) {
    return false;
  }

  return a === null || b === null ? null : true;
};
export const or3 = (a: Truth, b: Truth): Truth => {
  if (a === true || b === true) {
    return true;
  }

  return a === null || b === null ? null : false;
};
export const xor3 = (a: Truth, b: Truth): Truth => (a === null || b === null ? null : a !== b);
/** The binary three-valued connectives, keyed by AST node kind. */
export const BOOL3: Record<'and' | 'or' | 'xor', (a: Truth, b: Truth) => Truth> = {
  and: and3,
  or: or3,
  xor: xor3,
};

/** Raise an ISO data exception (SQLSTATE class 22): a runtime value/type fault. */
export const dataException = (message: string): never => {
  throw new LenkeError(message, { code: ErrorCode.DataException });
};

/** Lexicographic compare of two field-name strings (matches the native key sort). */
export const cmpKey = (a: string, b: string): number => {
  if (a < b) {
    return -1;
  }

  return a > b ? 1 : 0;
};

/** Build a canonical record from field pairs (dup last-wins, keys sorted). The
 *  record value (`LenkeRecord`) lives in `@lenke/core`, shared with the
 *  serialization codecs and the Gremlin engine. */
export const makeRecord = (fields: readonly (readonly [string, unknown])[]): LenkeRecord =>
  LenkeRecord.from(fields);

/** Read field `key` from a record/map, or `null` if absent (three-valued). */
export const recordGet = (rec: LenkeRecord, key: string): unknown =>
  rec.has(key) ? rec.get(key) : null;

export const typeName = (v: unknown): string => {
  if (Array.isArray(v)) {
    return 'a list';
  }

  if (v instanceof LenkeRecord) {
    return 'a map';
  }

  if (v !== null && typeof v === 'object') {
    return 'a graph element';
  }

  return typeof v;
};

// ISO arithmetic operands must be numbers (or NULL, which propagates). A
// non-numeric value is a data exception, not a silent `Number()` coercion to
// NaN — `'abc' + 1` and `true * 2` raise rather than producing garbage.
export const numOf = (v: unknown): number | null => {
  if (isNullish(v)) {
    return null;
  }

  if (typeof v === 'number') {
    return v;
  }

  return dataException(`arithmetic requires a number, got ${typeName(v)}`);
};

// `v IN list` is a three-valued OR of equalities `v = e` over the elements,
// whose identity (empty list) is FALSE. So `null IN []` is FALSE — there is
// nothing to be uncertain about — while `null IN [1]` and `3 IN [1, null]` are
// UNKNOWN. A TRUE equality short-circuits past any UNKNOWN.
// Structural value equality — lists compare by length then element-wise, matching
// the Rust engine's `val_eq`. (TS previously used reference identity, so
// `[1,2] = [1,2]` disagreed between the two engines.) A null list element compares
// equal here, same as Rust; strict ISO three-valued list equality would be
// UNKNOWN when an element is null — a documented, engine-symmetric deviation.
export const structuralEq = (a: unknown, b: unknown): boolean => {
  const aList = Array.isArray(a);
  const bList = Array.isArray(b);

  if (aList && bList) {
    return a.length === b.length && a.every((x, i) => structuralEq(x, b[i]));
  }

  if (aList || bList) {
    return false;
  }

  // Records are equal iff they have the same fields (keys canonical/sorted) with
  // recursively-equal values — ISO records support `=`/`<>`.
  const aRec = a instanceof LenkeRecord;
  const bRec = b instanceof LenkeRecord;

  if (aRec && bRec) {
    if (a.size !== b.size) {
      return false;
    }

    const ae = [...a];
    const be = [...b];

    return ae.every(([k, v], i) => k === be[i][0] && structuralEq(v, be[i][1]));
  }

  if (aRec || bRec) {
    return false;
  }

  // Two temporal instances are equal by value (same kind + same instant/
  // components), not by reference — `DATE '2020-01-01' = DATE '2020-01-01'`.
  if (isTemporal(a) && isTemporal(b)) {
    return temporalCmpTotal(a, b) === 0;
  }

  return a === b;
};

export const inList = (v: unknown, list: unknown): Truth => {
  if (!Array.isArray(list)) {
    return null;
  }

  let sawUnknown = false;

  for (const e of list) {
    if (isNullish(v) || isNullish(e)) {
      sawUnknown = true;
      continue;
    }

    if (structuralEq(e, v)) {
      return true;
    }
  }

  return sawUnknown ? null : false;
};

/** Binary operators resolved to a function once, at compile time. */
export const ARITH: Record<ArithOp, (a: number, b: number) => number> = {
  '+': (a, b) => a + b,
  '-': (a, b) => a - b,
  '*': (a, b) => a * b,
  '/': (a, b) => a / b,
  '%': (a, b) => a % b,
};

/**
 * One step of a left-associative arithmetic fold `lval <op> rval`. Preserves the
 * binary semantics: temporal arithmetic when either side is temporal, null from a
 * non-numeric operand, and a division/modulo-by-zero data exception.
 */
export const arithStep = (
  op: ArithOp,
  fn: (a: number, b: number) => number,
  lval: unknown,
  rval: unknown,
): unknown => {
  if (isTemporal(lval) || isTemporal(rval)) {
    return temporalArith(op, lval, rval);
  }

  const lv = numOf(lval);
  const rv = numOf(rval);

  if (lv === null || rv === null) {
    return null;
  }

  if ((op === '/' || op === '%') && rv === 0) {
    return dataException('division by zero');
  }

  return fn(lv, rv);
};

/** One step of a left-associative `||` fold: null propagates; two lists concat. */
export const concatStep = (lv: unknown, rv: unknown): unknown => {
  if (isNullish(lv) || isNullish(rv)) {
    return null;
  }

  if (Array.isArray(lv) && Array.isArray(rv)) {
    return [...lv, ...rv];
  }

  return str(lv) + str(rv);
};
/**
 * Compare two strings by Unicode CODE POINT, matching Rust `str::cmp` (UTF-8 byte
 * order == code-point order). JS `<`/`>` order by UTF-16 code UNIT, which disagrees
 * when an astral char (surrogate pair, ≥ U+10000) meets a BMP char in U+E000..U+FFFF:
 * the high surrogate (0xD800..0xDBFF) sorts BELOW U+E000 by code unit, but the astral
 * code point sorts ABOVE it. Iterating code points (via `codePointAt` + a 1-or-2 unit
 * step, allocation-free) makes both engines agree. Fast path: with no surrogate in
 * either string every char is BMP, so UTF-16 order already equals code-point order and
 * the native compare (much faster than a per-char loop) is exact.
 */
export const compareCodePoints = (a: string, b: string): number => {
  if (!HAS_SURROGATE.test(a) && !HAS_SURROGATE.test(b)) {
    if (a < b) {
      return -1;
    }

    return a > b ? 1 : 0;
  }

  let i = 0;
  let j = 0;

  while (i < a.length && j < b.length) {
    const ca = a.codePointAt(i) as number;
    const cb = b.codePointAt(j) as number;

    if (ca !== cb) {
      return ca < cb ? -1 : 1;
    }

    i += ca > 0xffff ? 2 : 1;
    j += cb > 0xffff ? 2 : 1;
  }

  // Whichever string still has characters is the longer one, hence greater; a
  // prefix sorts before its extension.
  if (i < a.length) {
    return 1;
  }

  return j < b.length ? -1 : 0;
};
const HAS_SURROGATE = /[\uD800-\uDBFF]/;

const strOr =
  (op: (c: number) => boolean, raw: (a: number | string, b: number | string) => boolean) =>
  (a: number | string, b: number | string): boolean =>
    typeof a === 'string' && typeof b === 'string' ? op(compareCodePoints(a, b)) : raw(a, b);

export const COMPARE: Record<CompareOp, (a: number | string, b: number | string) => boolean> = {
  '=': (a, b) => a === b,
  '<>': (a, b) => a !== b,
  '<': strOr(
    (c) => c < 0,
    (a, b) => a < b,
  ),
  '>': strOr(
    (c) => c > 0,
    (a, b) => a > b,
  ),
  '<=': strOr(
    (c) => c <= 0,
    (a, b) => a <= b,
  ),
  '>=': strOr(
    (c) => c >= 0,
    (a, b) => a >= b,
  ),
};

export type FuncExpr = Extract<Expr, { kind: 'func' }>;

export const AGGREGATES = new Set([
  'count',
  'sum',
  'avg',
  'min',
  'max',
  'collect_list',
  'percentile_cont',
  'percentile_disc',
  'stddev_pop',
  'stddev_samp',
]);

/**
 * ISO ordered-set percentile over a group's numeric values. `cont`
 * (`percentile_cont`) interpolates linearly between the two ranks bracketing
 * `frac·(n−1)`; otherwise (`percentile_disc`) it returns the value at the smallest
 * 0-based rank `k` with `(k+1)/n ≥ frac`. Non-numeric / non-finite values are
 * dropped; `frac` is pre-clamped to `[0, 1]`. Empty input → `null`.
 */
export const percentileOf = (
  values: readonly unknown[],
  frac: number,
  cont: boolean,
): number | null => {
  // `numArg`, not `Number` — the engine's numeric coercion, which the native
  // side uses here too. Raw `Number` accepts spellings the engine rejects
  // everywhere else (`Number('0x10')` is 16, while `to_float`/`sum`/`avg` and
  // the native `percentile` all read '0x10' as non-numeric).
  const nums = values
    .map(numArg)
    .filter((x) => Number.isFinite(x))
    .sort((a, b) => a - b);
  const n = nums.length;

  if (n === 0) {
    return null;
  }

  if (cont) {
    const rn = frac * (n - 1);
    const lo = Math.floor(rn);
    const hi = Math.ceil(rn);

    return lo === hi ? nums[lo] : nums[lo] + (rn - lo) * (nums[hi] - nums[lo]);
  }

  return nums[Math.min(n - 1, Math.max(0, Math.ceil(frac * n) - 1))];
};

/** Does an expression contain an aggregate anywhere (→ implicit grouping)? */
export const hasAggregate = (expr: Expr): boolean => {
  switch (expr.kind) {
    case 'func':
      return AGGREGATES.has(expr.name) || expr.args.some(hasAggregate);
    case 'graphPred':
      return expr.args.some(hasAggregate);
    case 'neg':
    case 'not':
    case 'isNull':
    case 'isTruth':
    case 'isLabeled':
    case 'isTyped':
      return hasAggregate(expr.expr);
    case 'arith':
      return hasAggregate(expr.head) || expr.tail.some(([, e]) => hasAggregate(e));
    case 'concat':
    case 'and':
    case 'or':
    case 'xor':
      return expr.items.some(hasAggregate);
    case 'compare':
      return hasAggregate(expr.left) || hasAggregate(expr.right);
    case 'letIn':
      return expr.bindings.some((b) => hasAggregate(b.expr)) || hasAggregate(expr.body);
    case 'in':
      return hasAggregate(expr.expr) || hasAggregate(expr.list);
    case 'index':
      return hasAggregate(expr.base) || hasAggregate(expr.index);
    case 'field':
      return hasAggregate(expr.base);
    case 'list':
      return expr.items.some(hasAggregate);
    case 'record':
      return expr.fields.some((f) => hasAggregate(f.value));
    case 'case':
      return (
        (expr.subject ? hasAggregate(expr.subject) : false) ||
        expr.whens.some((w) => hasAggregate(w.when) || hasAggregate(w.then)) ||
        (expr.elseExpr ? hasAggregate(expr.elseExpr) : false)
      );
    default:
      return false;
  }
};

// ISO `<numeric value function>` unary forms, keyed by function name. Each takes
// a single number; null in → null out is handled by the caller.
export const UNARY_NUM: Record<string, (n: number) => number> = {
  abs: Math.abs,
  ceil: Math.ceil,
  ceiling: Math.ceil,
  floor: Math.floor,
  sqrt: Math.sqrt,
  exp: Math.exp,
  ln: Math.log,
  log10: Math.log10,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  cot: (n) => 1 / Math.tan(n),
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  degrees: (n) => (n * 180) / Math.PI,
  radians: (n) => (n * Math.PI) / 180,
  sign: (n) => mathSign(n),
};

// ISO GQL 0-arg numeric constants.
export const NUM_CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  e: Math.E,
};

/**
 * Render one element of a list/path: a null element joins as the EMPTY string
 * (`String([1,null,3])` === `"1,,3"`), unlike a top-level null, which renders as
 * `"null"`. The native engine mirrors this JS rule deliberately.
 */
const joinElement = (x: unknown): string => (isNullish(x) ? '' : str(x));

/**
 * Stringify a value the way the native engine does (`js_str` in `gql/eval.rs`) —
 * the single coercion behind `to_string`, `CAST(… AS STRING)`, `||`, and every
 * string function's arguments.
 *
 * Plain `String(v)` agrees only on the primitives. A record is a `Map` subclass,
 * so it stringifies as `"[object Map]"`; a vertex/edge/path carries a debug
 * `toString` (`"Vertex (1) {}"`); and `Array.prototype.join` re-enters `String`
 * rather than this function, so a record nested in a list would slip through.
 * Each of those rendered differently from the native engine.
 */
export const str = (v: unknown): string => {
  // A record renders as its canonical JSON object — the same form it serializes
  // to in a result row (keys sorted by `LenkeRecord.from`).
  if (isRecord(v)) {
    return JSON.stringify(v);
  }

  // An element renders as its id, matching `element_id` (and native's `js_str`).
  if (isElement(v)) {
    return String((v as { id: unknown }).id);
  }

  if (Array.isArray(v)) {
    return v.map(joinElement).join(',');
  }

  // A path renders as its interleaved vertex/edge id sequence.
  if (v instanceof Path) {
    return [...v].map(joinElement).join(',');
  }

  return String(v);
};

// Round half away from zero — Rust's `f64::round` semantics. JS `Math.round`
// rounds half toward +∞ (`Math.round(-2.5) === -2`), so we apply the sign
// around `Math.abs` to match the native engine bit-for-bit.
export const roundHalfAway = (v: number): number => Math.sign(v) * Math.round(Math.abs(v));

// `sign` lives in `@lenke/core`: the Gremlin engine needs the same one, and the
// whole reason it is hand-written rather than `Math.sign` is cross-engine
// agreement.
export { mathSign };

/** ISO unary string value functions: one string in, a value out. */
export const UNARY_STR: Record<string, (s: string) => unknown> = {
  upper: (s) => s.toUpperCase(),
  lower: (s) => s.toLowerCase(),
  char_length: (s) => s.length,
  character_length: (s) => s.length,
};

// `trim`/`btrim` (both ends), `ltrim` (leading), `rtrim` (trailing). With a 2nd
// arg — a SET of characters to strip — they trim those code points (mirrors the
// Rust `multi_trim`); without one they strip whitespace (the existing JS
// behavior, kept as-is for byte-identity on the whitespace path).
export const multiTrim = (
  s: string,
  chars: string,
  leading: boolean,
  trailing: boolean,
): string => {
  // Code-point iteration (matches the Rust `chars()`), so the trim is
  // byte-identical across engines for the character-set case.
  const set = new Set(Array.from(chars));
  const cps = Array.from(s);
  let lo = 0;
  let hi = cps.length;

  while (leading && lo < hi && set.has(cps[lo])) {
    lo++;
  }

  while (trailing && hi > lo && set.has(cps[hi - 1])) {
    hi--;
  }

  return cps.slice(lo, hi).join('');
};

export const TRIM_FNS = new Set(['trim', 'btrim', 'ltrim', 'rtrim']);

export const callTrim = (name: string, a: unknown, b: unknown): unknown => {
  if (isNullish(a)) {
    return null;
  }

  const s = str(a);

  if (isNullish(b)) {
    // whitespace default — unchanged JS behavior.
    if (name === 'ltrim') {
      return s.replace(/^\s+/, '');
    }

    return name === 'rtrim' ? s.replace(/\s+$/, '') : s.trim();
  }

  return multiTrim(s, str(b), name !== 'rtrim', name !== 'ltrim');
};

/** ISO binary numeric value functions: LOG takes (base, value). */
export const BINARY_NUM: Record<string, (x: number, y: number) => number> = {
  // KNOWN LIMITATION (won't-fix): V8's `**`/`Math.pow` differs from Rust's
  // `powf` (glibc `pow`, the native engine) by ≤1 ULP on some inputs — e.g.
  // power(0.7,10) → …4ae here vs …4af native; power(2,-0.5) → …bcc vs …bcd. So
  // `power`/`pow`/`^` are NOT byte-identical cross-engine on those inputs; a true
  // fix needs a shared deterministic pow kernel. See this package's README.md.
  power: (x, y) => x ** y,
  mod: (x, y) => x % y,
  log: (base, value) => Math.log(value) / Math.log(base),
  // atan2(y, x): the ISO GQL binary arctangent (quadrant-correct). Mirrors the
  // native `y.atan2(x)`.
  atan2: (y, x) => Math.atan2(y, x),
};

/** The loud data exception for `avg` over a temporal, or `sum` over a
 *  non-DURATION temporal — byte-identical message to native `FAULT_TEMPORAL_AGG`. */
export const unsupportedTemporalAgg = (): LenkeError =>
  new LenkeError(
    "unsupported temporal aggregate: sum() is defined only for DURATION (dates/times aren't " +
      'summable), and avg() over DURATION would need duration/count (often non-representable, ' +
      'e.g. avg(P1M,P2M)=P1.5M); use min()/max(), or sum() + host division',
    { code: ErrorCode.DataException },
  );

/** `sum`/`avg` over a list/map is not numeric — throw loud rather than element-sum
 *  (which is the Gremlin `Scope.local` behavior, not a global aggregate). */
export const nonNumericAgg = (): LenkeError =>
  new LenkeError(
    'sum()/avg() require numeric values; a list/map is not summable — reduce it first ' +
      '(Gremlin sum(local), or GQL UNWIND + sum)',
    { code: ErrorCode.DataException },
  );

/** `sum` over gathered temporal values: fold DURATIONs via the same `dur + dur`
 *  (`temporalArith('+')`, which throws on overflow); fault on any non-DURATION. */
export const temporalSum = (values: unknown[]): unknown => {
  let acc: unknown;

  for (const v of values) {
    if (!(v instanceof Duration)) {
      throw unsupportedTemporalAgg();
    }

    acc = acc === undefined ? v : temporalArith('+', acc, v);
  }

  return acc ?? null;
};

/**
 * `size`/`length`/`path_length`: a path's hop count (NOT `Path.length`, which is
 * the interleaved element count per the List contract), else a list/string length.
 */
export const lengthOf = (a: unknown): number | null => {
  if (isNullish(a)) {
    return null;
  }

  if (a instanceof Path) {
    return a.hops;
  }

  return Array.isArray(a) || typeof a === 'string' ? a.length : null;
};

/** Scalar (non-aggregate) functions: the ISO numeric/string value functions. */
// Coerce a numeric-function argument the way the Rust `num_of` does, NOT JS
// `Number()` — which additionally accepts hex/binary/octal strings ('0x10' → 16)
// and single-element arrays ([5] → 5). Numbers pass through; booleans → 0/1; a
// string parses under the strict decimal/scientific grammar (empty → 0); anything
// else → NaN. So abs('0x10') / abs([5]) are NaN in both engines.
export const numArg = (v: unknown): number => {
  if (typeof v === 'number') {
    return v;
  }

  if (typeof v === 'boolean') {
    return v ? 1 : 0;
  }

  if (typeof v === 'string') {
    const t = v.trim();

    if (t === '') {
      return 0;
    }

    if (!FINITE_NUMERIC.test(t)) {
      return Number.NaN;
    }

    // A syntactically-valid literal can still overflow ('1e1000' → Infinity).
    // Native filters non-finite parses to NaN, so this must too — otherwise
    // `sqrt('1e1000')` is Infinity here and null there.
    const parsed = Number(t);

    return Number.isFinite(parsed) ? parsed : Number.NaN;
  }

  return Number.NaN;
};

// Strict argument typing for the string / byte functions and the polymorphic
// string|list functions — mirrors the native engine's `call_scalar_checked`. A non-null
// argument of the wrong type is a data exception (never JS-coerced); a NULL argument
// still propagates to NULL. The mixed-arity fns (left/right/substring) type their first
// position as the string and the rest as numbers.
const STR_ARG_POSITIONS: Record<string, readonly number[]> = {
  upper: [0],
  lower: [0],
  char_length: [0],
  character_length: [0],
  byte_length: [0],
  octet_length: [0],
  trim: [0, 1],
  btrim: [0, 1],
  ltrim: [0, 1],
  rtrim: [0, 1],
  split: [0, 1],
  starts_with: [0, 1],
  ends_with: [0, 1],
  contains: [0, 1],
  regex_match: [0, 1],
  replace: [0, 1, 2],
  left: [0],
  right: [0],
  substring: [0],
};
const NUM_ARG_POSITIONS: Record<string, readonly number[]> = {
  left: [1],
  right: [1],
  substring: [1, 2],
};
const POLY_STR_LIST = new Set(['reverse', 'size', 'cardinality']);

/**
 * Raise a data exception if a string/byte/polymorphic function is given an argument of
 * the wrong type — the strict typing the native engine enforces in `call_scalar_checked`.
 * A NULL argument passes (it propagates to NULL downstream).
 */
const assertScalarArgTypes = (name: string, args: readonly unknown[]): void => {
  const strPos = STR_ARG_POSITIONS[name];

  if (strPos) {
    for (const i of strPos) {
      const v = args[i];

      if (v !== undefined && !isNullish(v) && typeof v !== 'string') {
        dataException(`${name}() requires a string (a number is not coerced)`);
      }
    }
  }

  const numPos = NUM_ARG_POSITIONS[name];

  if (numPos) {
    for (const i of numPos) {
      const v = args[i];

      if (v !== undefined && !isNullish(v) && typeof v !== 'number') {
        dataException(`${name}() requires a number (a string is not coerced)`);
      }
    }
  }

  if (POLY_STR_LIST.has(name)) {
    const [v] = args;

    if (v !== undefined && !isNullish(v) && typeof v !== 'string' && !Array.isArray(v)) {
      dataException(`${name}() requires a string or list`);
    }
  }
};

export const callScalar = (
  name: string,
  args: readonly unknown[],
  // Required, not defaulted: a default parameter counts against this function's
  // complexity budget, and every real call site has a graph in hand anyway.
  limits: GraphLimits,
): unknown => {
  assertScalarArgTypes(name, args);

  const [a, b] = args;
  const unaryNum = UNARY_NUM[name];

  if (unaryNum) {
    return isNullish(a) ? null : unaryNum(numArg(a));
  }

  const unaryStr = UNARY_STR[name];

  if (unaryStr) {
    return isNullish(a) ? null : unaryStr(str(a));
  }

  const binaryNum = BINARY_NUM[name];

  if (binaryNum) {
    return isNullish(a) || isNullish(b) ? null : binaryNum(numArg(a), numArg(b));
  }

  if (name in NUM_CONSTANTS) {
    return NUM_CONSTANTS[name];
  }

  if (TRIM_FNS.has(name)) {
    return callTrim(name, a, b);
  }

  switch (name) {
    case 'round': {
      // round(num, [digits]) — digits default 0; half away from zero.
      if (isNullish(a)) {
        return null;
      }

      const digits = isNullish(b) ? 0 : Math.trunc(numArg(b));
      const f = 10 ** digits;

      return roundHalfAway(numArg(a) * f) / f;
    }
    // `cardinality` is the ISO GQL / SQL name; `size` is the openCypher spelling.
    case 'cardinality':
    case 'size':
    case 'length':
    case 'path_length':
      return lengthOf(a);
    case 'left':
      return isNullish(a) || isNullish(b)
        ? null
        : sanitizeSurrogates(str(a).slice(0, Math.max(0, numArg(b))));
    case 'right': {
      if (isNullish(a) || isNullish(b)) {
        return null;
      }

      const s = str(a);
      // Truncate toward zero to mirror native's `as usize` cast: a fractional
      // length takes whole characters, so `right('abcdef', 2.9)` is 'ef' (2), not
      // the 'def' that `slice(6 - 2.9)` = `slice(3.1)` would give. `left` already
      // truncates, because its fraction lands in `slice`'s END argument.
      const n = Math.trunc(numArg(b));

      // `n > 0`, not `!(n <= 0)`, so a NaN length (`right(s, 'nan')`) takes the
      // empty branch. Falling through with NaN would `slice(NaN)` — i.e.
      // `slice(0)`, the WHOLE string — whereas native's `NaN as usize` saturates
      // to 0 and yields ''. `left` is already empty for NaN.
      return n > 0 ? sanitizeSurrogates(s.slice(Math.max(0, s.length - n))) : '';
    }
    case 'coalesce':
      return args.find((x) => !isNullish(x)) ?? null;
    case 'nullif':
      // ISO `<case abbreviation>`: NULLIF(a, b) = NULL when a = b, else a. The
      // equality is the engine's VALUE equality (`structuralEq`, what `=` uses),
      // not JS `===` — two temporals/lists/records holding the same value are
      // distinct objects, so `===` said "not equal" where the native `val_eq`
      // said equal.
      return !isNullish(a) && !isNullish(b) && structuralEq(a, b) ? null : (a ?? null);
    case 'element_id':
      // ISO `<element_id function>`: the identifier of a node or edge.
      return a && typeof a === 'object' && 'id' in a ? (a as { id: unknown }).id : null;
    default:
      // Graph/conversion/string/list functions live in a second dispatcher so
      // neither switch exceeds the complexity budget.
      return callExtendedScalar(name, args, limits);
  }
};

// --- ISO graph / conversion / string-list scalar functions -------------------
// Split out of `callScalar` (complexity budget). Semantics mirror the Rust
// engine (`gql/eval.rs`) byte-for-byte so both engines agree: labels/keys are
// sorted, slices are UTF-16-safe, `null` in → `null` out, and an unknown name is
// an `Unsupported` fault — never a silent `null`.

// Strict numeric-string parse matching Rust's `str::trim().parse::<f64>()`: the
// WHOLE trimmed string must be a finite decimal (optional sign, integer/fraction,
// exponent). `Number.parseFloat` is lenient — it would read `'12abc'` as `12`,
// diverging from the Rust engine — so we gate on the grammar first. Exotic forms
// (`inf`, `nan`, hex) are out of scope and yield null on both engines' common path.
export const FINITE_NUMERIC = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

export const numericStringToFloat = (s: string): number | null => {
  const t = s.trim();

  if (!FINITE_NUMERIC.test(t)) {
    return null;
  }

  // A syntactically-valid literal can still overflow to Infinity ('1e1000').
  // Native filters those out (`.filter(|n| n.is_finite())`), so a non-finite
  // result is NOT a value here either — it reads as null, which matters for
  // `IS NOT NULL` even though JSON renders both as null.
  const v = Number.parseFloat(t);

  return Number.isFinite(v) ? v : null;
};

export const toIntScalar = (a: unknown): number | null => {
  if (isNullish(a)) {
    return null;
  }

  if (typeof a === 'number') {
    return Math.trunc(a);
  }

  // Only a number or a STRING converts — mirroring the native arms. Falling back
  // to `str(a)` would convert by stringifying, so a one-element list (`[0]` → "0")
  // or a vertex (whose `str` is its id) would come back as a number instead of the
  // null the native engine returns.
  if (typeof a !== 'string') {
    return null;
  }

  const p = numericStringToFloat(a);

  return p === null ? null : Math.trunc(p);
};

export const toFloatScalar = (a: unknown): number | null => {
  if (isNullish(a)) {
    return null;
  }

  if (typeof a === 'number') {
    return a;
  }

  // Number or string only — see the note in `toIntScalar`.
  return typeof a === 'string' ? numericStringToFloat(a) : null;
};

export const substringScalar = (a: unknown, b: unknown, len: unknown): string | null => {
  if (isNullish(a) || isNullish(b)) {
    return null;
  }

  const s = str(a);
  // ISO GQL: the start index is 1-based (SQL `SUBSTRING` convention), so
  // `substring('crystal hawk river', 1, 7)` → 'crystal'. Convert to a 0-based
  // UTF-16 offset; a start <= 0 shrinks the window from the front (SQL
  // semantics), which the native engine mirrors exactly.
  const zeroStart = numArg(b) - 1;
  const from = Math.max(0, zeroStart);

  return sanitizeSurrogates(
    isNullish(len) ? s.slice(from) : s.slice(from, Math.max(0, zeroStart + numArg(len))),
  );
};

// Decode a UTF-16 code-unit sequence to a string exactly as Rust's
// `String::from_utf16_lossy` does: a valid high+low surrogate pair combines to
// its scalar; any UNPAIRED surrogate becomes U+FFFD. `split('')` and `reverse`
// operate on UTF-16 code units (JS `.length` model, so `size` and these agree),
// and this shared lossy decode keeps them byte-identical with the native
// engine — whose UTF-8 strings cannot carry a lone surrogate. NOTE: this
// diverges from JS `String.split('')` / naive reversal, which PRESERVE lone
// surrogates; splitting or reversing across an astral character is inherently
// lossy here (documented non-conformance, mirroring the native engine).
export const fromUtf16UnitsLossy = (units: readonly number[]): string => {
  let out = '';

  for (let i = 0; i < units.length; i++) {
    const u = units[i];

    if (u >= 0xd800 && u <= 0xdbff) {
      const lo = i + 1 < units.length ? units[i + 1] : -1;

      if (lo >= 0xdc00 && lo <= 0xdfff) {
        out += String.fromCharCode(u, lo);
        i++;
      } else {
        out += '�';
      }
    } else if (u >= 0xdc00 && u <= 0xdfff) {
      out += '�';
    } else {
      out += String.fromCharCode(u);
    }
  }

  return out;
};

// A UTF-16 slice (`substring`/`left`/`right`) can cut a surrogate pair, leaving a
// LONE surrogate. The native engine's UTF-8 strings can't carry one, so it
// renders U+FFFD; run every sliced result through the same lossy decode so the
// two engines stay byte-identical on astral-boundary slices.
export const sanitizeSurrogates = (s: string): string =>
  fromUtf16UnitsLossy(Array.from({ length: s.length }, (_, i) => s.charCodeAt(i)));

export const splitScalar = (a: unknown, b: unknown): string[] | null => {
  if (isNullish(a) || isNullish(b)) {
    return null;
  }

  const s = str(a);
  const delim = str(b);

  // Empty delimiter → one element per UTF-16 code unit (JS `.length` model);
  // lone surrogates render as U+FFFD for byte-identity with the native engine.
  return delim === ''
    ? Array.from({ length: s.length }, (_, i) => fromUtf16UnitsLossy([s.charCodeAt(i)]))
    : s.split(delim);
};

export const replaceScalar = (a: unknown, b: unknown, repl: unknown): string | null => {
  if (isNullish(a) || isNullish(b)) {
    return null;
  }

  const search = str(b);

  return search === ''
    ? str(a)
    : str(a)
        .split(search)
        .join(isNullish(repl) ? '' : str(repl));
};

// ISO GQL `to_boolean`: bool → itself; number → (x != 0), NaN → null; string →
// case-insensitive true/false variants ('true'/'yes'/'1' | 'false'/'no'/'0'),
// anything else → null. Mirrors the Rust `ToBoolean` arm.
export const toBooleanScalar = (a: unknown): boolean | null => {
  if (isNullish(a)) {
    return null;
  }

  if (typeof a === 'boolean') {
    return a;
  }

  if (typeof a === 'number') {
    return Number.isNaN(a) ? null : a !== 0;
  }

  // Boolean, number, or string only — see the note in `toIntScalar`.
  if (typeof a !== 'string') {
    return null;
  }

  const t = a.trim().toLowerCase();

  if (t === 'true' || t === 'yes' || t === '1') {
    return true;
  }

  return t === 'false' || t === 'no' || t === '0' ? false : null;
};

// ISO GQL `to_list`: a string → its UTF-16 code-unit characters (same unit
// model as `split('')`); a list → itself; any other value → a singleton list.
export const toListScalar = (a: unknown): unknown[] | null => {
  if (isNullish(a)) {
    return null;
  }

  if (Array.isArray(a)) {
    return a;
  }

  return typeof a === 'string' ? (splitScalar(a, '') as string[]) : [a];
};

export const UTF8 = new TextEncoder();

// UTF-8 byte length (ISO GQL `byte_length` / `octet_length`), matching Rust's
// `str::len()` (a UTF-8 byte count).
export const byteLen = (s: string): number => UTF8.encode(s).length;

// ISO GQL `range(start, end, [step])` → an inclusive list of integers. A zero
// step has no defined progression → null. Mirrors the Rust `Range` arm.
/**
 * `range()`, bounded by the graph's `limits.range` ceiling (see `GraphLimits`) —
 * the same bound the native engine applies, so both fault on exactly the same
 * inputs. A GQL list is a MATERIALIZED value in both engines (a JS array here,
 * `Vec<Val>` there), so it cannot be produced lazily without a lazy-list variant
 * threaded through both value models. Unbounded, `range(0, 1e21)` is not merely
 * slow: the counter stops advancing at 2^53 (`i += 1` is a no-op there), so the
 * loop never terminates while pushing, and the host dies on an OOM kill instead of
 * the query erroring.
 */
export const rangeScalar = (
  a: unknown,
  b: unknown,
  step: unknown,
  limit: number = DEFAULT_CONFIG.limits.range,
): number[] | null => {
  if (isNullish(a) || isNullish(b)) {
    return null;
  }

  const s = Math.trunc(numArg(a));
  const e = Math.trunc(numArg(b));
  const st = isNullish(step) ? 1 : Math.trunc(numArg(step));

  if (st === 0) {
    return null;
  }

  // The element count is computed UP FRONT: it bounds the allocation before a
  // single push, and it makes the loop COUNT-driven rather than comparison-driven,
  // so the 2^53 stall cannot spin forever (`range(9007199254740992,
  // 9007199254740994)` has a count of 3 but never terminates under `i <= e`).
  // The values still come from repeated addition, so the sequence is unchanged.
  const count = Math.floor((e - s) / st) + 1;

  // A backwards span (or a NaN bound) yields no elements.
  if (Number.isNaN(count) || count <= 0) {
    return [];
  }

  if (count > limit) {
    throw new LenkeError(
      `range() would materialize more than ${limit} elements; narrow the bounds or widen the step`,
      { code: ErrorCode.ResourceExhausted },
    );
  }

  const out: number[] = new Array<number>(count);
  let i = s;

  for (let k = 0; k < count; k++) {
    out[k] = i;
    i += st;
  }

  return out;
};

export const headScalar = (a: unknown): unknown => (Array.isArray(a) && a.length > 0 ? a[0] : null);

export const lastScalar = (a: unknown): unknown =>
  Array.isArray(a) && a.length > 0 ? a[a.length - 1] : null;

export const reverseScalar = (a: unknown): unknown => {
  if (isNullish(a)) {
    return null;
  }

  if (Array.isArray(a)) {
    return [...a].reverse();
  }

  if (typeof a !== 'string') {
    return null;
  }

  // Reverse by UTF-16 code unit (JS `.length` model), lossy-decoding the
  // reversed units the same way the native engine does (see fromUtf16UnitsLossy).
  const units: number[] = [];

  for (let i = 0; i < a.length; i++) {
    units.push(a.charCodeAt(i));
  }

  units.reverse();

  return fromUtf16UnitsLossy(units);
};

// A sentinel returned by a sub-dispatcher that doesn't handle `name`, so the
// callers can chain dispatchers (each kept under the complexity budget) and
// fall through to the unknown-function error.
export const UNHANDLED = Symbol('unhandled');

// Graph functions — label/key order sorted for cross-engine parity.
export const callGraphFn = (name: string, a: unknown): unknown => {
  switch (name) {
    // NOT VERIFIED CONFORMANT — see docs/conformance/gql-feature-checklist.md.
    // No free source consulted shows `labels` as an ISO function at all (the
    // standard appears to use the `IS LABELED` predicate), so this is a Cypher
    // inheritance rather than a conformance claim. It takes an ELEMENT because
    // the two vendors that ship it do: Spanner's
    // `LABELS(GRAPH_ELEMENT) -> ARRAY<STRING>` and Fabric's
    // `labels(node_or_edge)`. Both return a length-1 list for an edge since
    // neither has multi-label edges; this engine does, so it returns the set —
    // the natural generalization, and the only accessor for it.
    case 'labels':
      return isVertex(a) || isEdge(a) ? [...a.labels].sort() : null;
    // `type` stays SINGULAR: it is openCypher's `type(relationship) -> String`,
    // which cannot express a set. It reports an edge's first type, exactly as
    // Gremlin's `label()` reports a multi-label vertex's first label — both have
    // to return one. `labels(e)` is how you get all of them.
    case 'type':
      return isEdge(a) ? ([...a.labels][0] ?? '') : null;
    // `property_names` is the ISO GQL name; `keys` is the openCypher spelling.
    case 'property_names':
    case 'keys':
      return isElement(a) ? Object.keys(a.properties).sort() : null;
    // ISO GQL path functions. Vertices/edges stay live element handles (they
    // serialize richly, like `RETURN a`); `elements` is the path's own
    // interleaved iteration (vertex, edge, …, vertex).
    case 'nodes':
      return a instanceof Path ? [...a.vertices] : null;
    // `edges` is the ISO GQL name (Cypher's `relationships` is deliberately not
    // accepted — GQL's element vocabulary is node/edge).
    case 'edges':
      return a instanceof Path ? [...a.edges] : null;
    case 'elements':
      return a instanceof Path ? [...a] : null;
    default:
      return UNHANDLED;
  }
};

// Conversion functions (null in → null out).
export const callConversionFn = (name: string, a: unknown): unknown => {
  switch (name) {
    case 'tostring':
    case 'to_string':
      return isNullish(a) ? null : str(a);
    case 'tointeger':
    case 'to_integer':
      return toIntScalar(a);
    case 'tofloat':
    case 'to_float':
      return toFloatScalar(a);
    case 'toboolean':
    case 'to_boolean':
      return toBooleanScalar(a);
    case 'tolist':
    case 'to_list':
      return toListScalar(a);
    default:
      return UNHANDLED;
  }
};

// String predicates (ISO BOOL-returning) + byte-length measurement.
export const callStringPredicateFn = (name: string, a: unknown, b: unknown): unknown => {
  switch (name) {
    case 'contains':
      return isNullish(a) || isNullish(b) ? null : str(a).includes(str(b));
    case 'starts_with':
      return isNullish(a) || isNullish(b) ? null : str(a).startsWith(str(b));
    case 'ends_with':
      return isNullish(a) || isNullish(b) ? null : str(a).endsWith(str(b));
    case 'byte_length':
    case 'octet_length':
      return isNullish(a) ? null : byteLen(str(a));
    default:
      return UNHANDLED;
  }
};

// String / list functions.
export const callStringListFn = (
  name: string,
  args: readonly unknown[],
  limits: GraphLimits,
): unknown => {
  const [a, b] = args;

  switch (name) {
    case 'substring':
      return substringScalar(a, b, args[2]);
    case 'split':
      return splitScalar(a, b);
    case 'replace':
      return replaceScalar(a, b, args[2]);
    case 'head':
      return headScalar(a);
    case 'last':
      return lastScalar(a);
    case 'reverse':
      return reverseScalar(a);
    case 'tail':
      return Array.isArray(a) ? a.slice(1) : null;
    case 'append':
      // The element may be null (a first-class value); only a null LIST → null.
      return Array.isArray(a) ? [...a, args[1] ?? null] : null;
    case 'range':
      return rangeScalar(a, b, args[2], limits.range);
    default:
      return UNHANDLED;
  }
};

// Push `v` into `out` unless a structurally-equal element is already present
// (first occurrence wins) — the dedup building block for the set-style list
// functions, mirroring the Rust `push_unique`.
export const pushUnique = (out: unknown[], v: unknown): void => {
  if (!out.some((x) => structuralEq(x, v))) {
    out.push(v);
  }
};

// ISO GQL `list_sort` <nullOrder> arg → the `nullsFirst` flag; `undefined`
// (default / unrecognized) falls back to the ORDER BY default in `compareSort`.
export const nullOrderArg = (v: unknown): boolean | undefined => {
  if (typeof v !== 'string') {
    return undefined;
  }

  const s = v.toLowerCase();

  if (s === 'first') {
    return true;
  }

  return s === 'last' ? false : undefined;
};

// Set-style list functions (ISO GQL). All dedup by structural equality with the
// first occurrence winning; list_sort reuses the ORDER BY total order so it is
// byte-identical with `ORDER BY`.
export const callListSetFn = (
  name: string,
  a: unknown,
  b: unknown,
  args: readonly unknown[],
): unknown => {
  switch (name) {
    case 'list_union': {
      if (!Array.isArray(a) || !Array.isArray(b)) {
        return null;
      }

      const out: unknown[] = [];

      for (const v of [...a, ...b]) {
        pushUnique(out, v);
      }

      return out;
    }
    case 'intersection': {
      if (!Array.isArray(a) || !Array.isArray(b)) {
        return null;
      }

      const out: unknown[] = [];

      for (const v of a) {
        if (b.some((w) => structuralEq(w, v))) {
          pushUnique(out, v);
        }
      }

      return out;
    }
    case 'difference': {
      if (!Array.isArray(a) || !Array.isArray(b)) {
        return null;
      }

      const out: unknown[] = [];

      for (const v of a) {
        if (!b.some((w) => structuralEq(w, v))) {
          pushUnique(out, v);
        }
      }

      return out;
    }
    case 'list_contains':
      // ISO returns the numeric 1 / 0 (not a boolean); the value may be null.
      if (!Array.isArray(a)) {
        return null;
      }

      return a.some((w) => structuralEq(w, b)) ? 1 : 0;
    case 'list_sort':
      if (!Array.isArray(a)) {
        return null;
      }

      return [...a].sort((x, y) =>
        compareSort(
          x,
          y,
          typeof b === 'string' && b.toLowerCase() === 'desc',
          nullOrderArg(args[2]),
        ),
      );
    default:
      return UNHANDLED;
  }
};

// Temporal constructors: `date(x)` / `local_datetime(x)` / `duration(x)`. Mirror
// the Rust `temporal_ctor` — parse a string, convert a temporal by kind (date↔
// datetime), else null (lenient, like the to_* conversions).
export const TEMPORAL_CTOR: Record<
  string,
  'date' | 'localtime' | 'datetime' | 'zoned_time' | 'zoned_datetime' | 'duration'
> = {
  date: 'date',
  local_time: 'localtime',
  local_datetime: 'datetime',
  datetime: 'datetime',
  zoned_time: 'zoned_time',
  zoned_datetime: 'zoned_datetime',
  duration: 'duration',
};

export const temporalCtor = (
  kind: 'date' | 'localtime' | 'datetime' | 'zoned_time' | 'zoned_datetime' | 'duration',
  v: unknown,
): unknown => {
  if (isNullish(v)) {
    return null;
  }

  if (typeof v === 'string') {
    // A bare date-only `YYYY-MM-DD` (no time part) coerces to midnight for a
    // datetime target — consistent with date() and the DATE `$__now` → midnight
    // precedent. Mirrors the Rust `temporal_ctor`.
    if (kind === 'datetime' && !/[T ]/.test(v)) {
      try {
        const d = temporalParse('date', v) as LocalDate;

        return new LocalDateTime(d.days * 86_400, 0);
      } catch {
        return null;
      }
    }

    try {
      return temporalParse(kind, v);
    } catch {
      return null;
    }
  }

  if (isTemporal(v)) {
    if (v.kind === kind) {
      return v;
    }

    if (kind === 'date' && v instanceof LocalDateTime) {
      return new LocalDate(Math.floor(v.secs / 86_400));
    }

    // local_time(datetime) → the time-of-day part.
    if (kind === 'localtime' && v instanceof LocalDateTime) {
      return new LocalTime(((v.secs % 86_400) + 86_400) % 86_400, v.nanos);
    }

    if (kind === 'datetime' && v instanceof LocalDate) {
      return new LocalDateTime(v.days * 86_400, 0);
    }
  }

  return null;
};

export const callTemporalFn = (name: string, args: readonly unknown[]): unknown => {
  const kind = TEMPORAL_CTOR[name];

  if (kind !== undefined) {
    return temporalCtor(kind, args[0]);
  }

  if (name === 'duration_between') {
    const [x, y] = args;

    return isTemporal(x) && isTemporal(y) ? durationBetween(x, y) : null;
  }

  return UNHANDLED;
};

// Temporal component extraction (the `_year`/`_month`/`_day`/`_hour`/`_minute`/
// `_second` lenke extension — see docs/design/gql-extensions.md). Euclidean
// floor/mod so pre-epoch instants (negative seconds) decompose byte-identically
// to the Rust `div_euclid`/`rem_euclid`.
export const SECS_PER_DAY = 86_400;
export const floorDiv = (n: number, d: number): number => Math.floor(n / d);
export const euclidMod = (n: number, d: number): number => ((n % d) + d) % d;

/** The integer component, or `null` if the temporal kind lacks it (e.g. `year`
 * of a LOCAL TIME, `hour` of a DATE) — the caller turns `null` into a throw. */
export const datePart = (name: string, t: Temporal): number | null => {
  if (name === 'year' || name === 'month' || name === 'day') {
    let epochDays: number;

    if (t instanceof LocalDate) {
      epochDays = t.days;
    } else if (t instanceof LocalDateTime) {
      epochDays = floorDiv(t.secs, SECS_PER_DAY);
    } else if (t instanceof ZonedDateTime) {
      epochDays = floorDiv(t.secs + t.offset * 60, SECS_PER_DAY);
    } else {
      return null;
    }

    const [y, m, d] = civilFromDays(epochDays);

    if (name === 'year') {
      return y;
    }

    return name === 'month' ? m : d;
  }

  let tod: number;

  if (t instanceof LocalTime) {
    tod = t.secs;
  } else if (t instanceof LocalDateTime) {
    tod = euclidMod(t.secs, SECS_PER_DAY);
  } else if (t instanceof ZonedTime || t instanceof ZonedDateTime) {
    tod = euclidMod(t.secs + t.offset * 60, SECS_PER_DAY);
  } else {
    return null;
  }

  if (name === 'hour') {
    return floorDiv(tod, 3600);
  }

  return name === 'minute' ? floorDiv(tod, 60) % 60 : tod % 60;
};

// Sigil-prefixed because date-part extraction is a lenke EXTENSION, not in the
// ISO GQL function catalogue (see docs/design/gql-extensions.md). The bare names
// (`year`/…) stay unknown functions, flagging non-portability at the call site.
export const DATE_PART_FNS = new Set(['_year', '_month', '_day', '_hour', '_minute', '_second']);

export const callDatePartFn = (name: string, a: unknown): unknown => {
  if (!DATE_PART_FNS.has(name)) {
    return UNHANDLED;
  }

  if (isNullish(a)) {
    return null; // null in → null out
  }

  const component = name.slice(1); // strip the extension sigil → year/month/…

  if (isTemporal(a)) {
    const n = datePart(component, a);

    if (n !== null) {
      return n;
    }
  }

  // A string is NOT coerced, and a temporal lacking the component faults loudly.
  throw new LenkeError(
    `${name}() requires a temporal value that carries that component ` +
      `(a date carries year/month/day; a time carries hour/minute/second) — ` +
      `a string is NOT coerced; wrap it with date()/local_datetime()/local_time() first`,
    { code: ErrorCode.InvalidValue },
  );
};

// The extended-scalar family dispatchers, in priority order. Each returns UNHANDLED
// for a name that isn't one of its own, so `callExtendedScalar` returns the first
// non-UNHANDLED result. Built once at module scope. Order matters only for
// callDatePartFn, whose throw is keyed to its own `_year`/… names.
const EXTENDED_DISPATCHERS: readonly ((
  name: string,
  a: unknown,
  b: unknown,
  args: readonly unknown[],
  limits: GraphLimits,
) => unknown)[] = [
  (name, a) => callGraphFn(name, a),
  (name, a) => callConversionFn(name, a),
  (name, a) => callDatePartFn(name, a),
  (name, _a, _b, args) => callTemporalFn(name, args),
  (name, a, b) => callStringPredicateFn(name, a, b),
  (name, _a, _b, args, limits) => callStringListFn(name, args, limits),
  (name, a, b, args) => callListSetFn(name, a, b, args),
];

export const callExtendedScalar = (
  name: string,
  args: readonly unknown[],
  limits: GraphLimits = DEFAULT_CONFIG.limits,
): unknown => {
  const [a, b] = args;

  // Short-circuit on the first handler. (The previous array-literal form invoked
  // ALL seven before the `!== UNHANDLED` check, so the early-return was dead and
  // every fall-through call paid for all seven switch dispatches.)
  for (const dispatch of EXTENDED_DISPATCHERS) {
    const result = dispatch(name, a, b, args, limits);

    if (result !== UNHANDLED) {
      return result;
    }
  }

  throw new LenkeError(`call to an unknown or unimplemented function: ${name}()`, {
    code: ErrorCode.UnknownFunction,
  });
};
