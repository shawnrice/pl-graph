import type { Edge, Graph, IndexableValue, RangeBound, Vertex } from '@lenke/core';
import {
  type AlgorithmConfig,
  type AlgorithmName,
  civilFromDays,
  Duration,
  durationBetween,
  fromTaggedJson,
  isTemporal,
  LocalDate,
  LocalTime,
  LocalDateTime,
  Path,
  runAlgorithmSync,
  type Temporal,
  temporalArith,
  temporalCmpTotal,
  temporalParse,
  temporalRelCmp,
  ZonedDateTime,
  ZonedTime,
} from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';
import { filter, flatMap, map, skip, take, toArray } from '@lenke/fp';

import type {
  ArithOp,
  Clause,
  CompareOp,
  CountValue,
  Expr,
  LabelExpr,
  LinearQuery,
  NodePattern,
  PathPattern,
  PathMode,
  PathSelector,
  Projection,
  PropertyConstraint,
  Query,
  RelPattern,
  Segment,
  RemoveItem,
  SetItem,
  SetOp,
  Statement,
  TxControl,
} from './ast.js';
import { isTxControl } from './ast.js';
import {
  candidateCount,
  candidateVertices,
  expand,
  hasIncidentEdges,
  labelsMatch,
  matchesLabel,
} from './graph-queries.js';

/**
 * The executor turns a parsed `Query` into result rows by *pattern matching*:
 * a declarative MATCH is evaluated as a sequence of nested loops that grow a
 * partial binding (variable -> graph element) one segment at a time. This is
 * the declarative<->imperative bridge — the language says "find this shape",
 * the executor picks the walk order (here, naive left-to-right).
 *
 * Rather than interpret the AST on every run, we *compile* it once: `compile`
 * lowers a `Query` into a tree of closures (a `Plan`) that captures all the
 * graph/param-independent decisions — operator dispatch, aggregate detection,
 * alias resolution, label-seed selection. The returned `Plan` is reusable:
 * `(graph, params) => Row[]`. Running it again skips the lexer, the parser, and
 * the per-node `switch` dispatch entirely. Params flow *through* the closures
 * (no module-global state), so a plan is reentrant.
 */

/** A bound element: a matched vertex or edge for a pattern variable. */
type Bound = Vertex | Edge;

/**
 * One candidate solution: the variables bound so far. Values are graph elements
 * after MATCH, but `WITH` can project arbitrary scalars into scope, so the value
 * type is `unknown`.
 */
type Binding = ReadonlyMap<string, unknown>;

/** A projected result row: alias/derived-name -> value. */
export type Row = Record<string, unknown>;

/** Query parameters (`$name`), supplied at run time and threaded through plans. */
type Params = Record<string, unknown>;

/**
 * The environment a compiled expression evaluates against: the current binding,
 * the query params, the graph (only EXISTS/COUNT subqueries read it), and — for
 * aggregates — the `group` of bindings being folded. Passing this explicitly
 * keeps expression evaluation a pure function of its inputs (no run-state global).
 */
type EvalEnv = {
  binding: Binding;
  params: Params;
  graph: Graph;
  group?: readonly Binding[];
};

/**
 * A compiled expression. The structural `switch (expr.kind)` happens once, at
 * compile time; what's left is this closure, evaluated against an `EvalEnv`.
 */
type CompiledExpr = (env: EvalEnv) => unknown;

/**
 * A reusable execution plan: bind a graph and params, get rows. This is the
 * artifact `compile` produces — analyze once, run many.
 *
 * `R` is the row shape you expect back — an opt-in, caller-side assertion (rows
 * are `Record<string, unknown>` at runtime; nothing is validated), so
 * `query<{ name: string }>(...)` returns `{ name: string }[]` and drops the
 * per-field cast. Defaults to `Row`.
 */
export type Plan<R extends Row = Row> = (graph: Graph, params?: Params) => R[];

// --- binding helpers ---------------------------------------------------------

const withBinding = (binding: Binding, name: string | undefined, value: Bound | Path): Binding => {
  if (!name) {
    return binding;
  }

  const next = new Map(binding);
  next.set(name, value);

  return next;
};

/**
 * Is binding `value` to `name` consistent with what's already bound? An
 * unbound variable always binds; a bound one must refer to the same element
 * (this is what makes shared variables across patterns act as a join key).
 */
const consistent = (binding: Binding, name: string | undefined, value: Bound): boolean => {
  if (!name) {
    return true;
  }

  const existing = binding.get(name);

  return existing === undefined || existing === value;
};

// ISO GQL: accessing a property that is absent — or a property of a NULL element
// (e.g. an unmatched OPTIONAL variable) — yields NULL, not `undefined`. Coalesce
// here so the whole pipeline (output rows, IS NULL, arithmetic) sees ISO's NULL.
const propOf = (bound: unknown, key: string): unknown =>
  (bound as { properties?: Record<string, unknown> } | undefined)?.properties?.[key] ?? null;

// --- three-valued logic & scalar helpers -------------------------------------

// ISO three-valued (Kleene) logic: `null` is UNKNOWN. A row is kept only when a
// predicate evaluates to exactly `true` (see callers comparing `=== true`).
type Truth = boolean | null;
const isNullish = (v: unknown): boolean => v === null || v === undefined;
const asTruth = (v: unknown): Truth => (isNullish(v) ? null : Boolean(v));
const not3 = (t: Truth): Truth => (t === null ? null : !t);
const and3 = (a: Truth, b: Truth): Truth => {
  if (a === false || b === false) {
    return false;
  }

  return a === null || b === null ? null : true;
};
const or3 = (a: Truth, b: Truth): Truth => {
  if (a === true || b === true) {
    return true;
  }

  return a === null || b === null ? null : false;
};
const xor3 = (a: Truth, b: Truth): Truth => (a === null || b === null ? null : a !== b);
/** The binary three-valued connectives, keyed by AST node kind. */
const BOOL3: Record<'and' | 'or' | 'xor', (a: Truth, b: Truth) => Truth> = {
  and: and3,
  or: or3,
  xor: xor3,
};

/** Raise an ISO data exception (SQLSTATE class 22): a runtime value/type fault. */
const dataException = (message: string): never => {
  throw new LenkeError(message, { code: ErrorCode.DataException });
};

const typeName = (v: unknown): string => {
  if (Array.isArray(v)) {
    return 'a list';
  }

  if (v !== null && typeof v === 'object') {
    return 'a graph element';
  }

  return typeof v;
};

// ISO arithmetic operands must be numbers (or NULL, which propagates). A
// non-numeric value is a data exception, not a silent `Number()` coercion to
// NaN — `'abc' + 1` and `true * 2` raise rather than producing garbage.
const numOf = (v: unknown): number | null => {
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
const structuralEq = (a: unknown, b: unknown): boolean => {
  const aList = Array.isArray(a);
  const bList = Array.isArray(b);

  if (aList && bList) {
    return a.length === b.length && a.every((x, i) => structuralEq(x, b[i]));
  }

  if (aList || bList) {
    return false;
  }

  // Two temporal instances are equal by value (same kind + same instant/
  // components), not by reference — `DATE '2020-01-01' = DATE '2020-01-01'`.
  if (isTemporal(a) && isTemporal(b)) {
    return temporalCmpTotal(a, b) === 0;
  }

  return a === b;
};

const inList = (v: unknown, list: unknown): Truth => {
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
const ARITH: Record<ArithOp, (a: number, b: number) => number> = {
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
const arithStep = (
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
const concatStep = (lv: unknown, rv: unknown): unknown => {
  if (isNullish(lv) || isNullish(rv)) {
    return null;
  }

  if (Array.isArray(lv) && Array.isArray(rv)) {
    return [...lv, ...rv];
  }

  return String(lv) + String(rv);
};
const COMPARE: Record<CompareOp, (a: number | string, b: number | string) => boolean> = {
  '=': (a, b) => a === b,
  '<>': (a, b) => a !== b,
  '<': (a, b) => a < b,
  '>': (a, b) => a > b,
  '<=': (a, b) => a <= b,
  '>=': (a, b) => a >= b,
};

type FuncExpr = Extract<Expr, { kind: 'func' }>;

const AGGREGATES = new Set([
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
const percentileOf = (values: readonly unknown[], frac: number, cont: boolean): number | null => {
  const nums = values
    .map(Number)
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
const hasAggregate = (expr: Expr): boolean => {
  switch (expr.kind) {
    case 'func':
      return AGGREGATES.has(expr.name) || expr.args.some(hasAggregate);
    case 'neg':
    case 'not':
    case 'isNull':
    case 'isTruth':
    case 'isLabeled':
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
    case 'in':
      return hasAggregate(expr.expr) || hasAggregate(expr.list);
    case 'index':
      return hasAggregate(expr.base) || hasAggregate(expr.index);
    case 'field':
      return hasAggregate(expr.base);
    case 'list':
      return expr.items.some(hasAggregate);
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
const UNARY_NUM: Record<string, (n: number) => number> = {
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
const NUM_CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  e: Math.E,
};

const str = (v: unknown): string => String(v);

// Round half away from zero — Rust's `f64::round` semantics. JS `Math.round`
// rounds half toward +∞ (`Math.round(-2.5) === -2`), so we apply the sign
// around `Math.abs` to match the native engine bit-for-bit.
const roundHalfAway = (v: number): number => Math.sign(v) * Math.round(Math.abs(v));

// ISO GQL `sign` → -1 | 0 | 1 (NaN passes through). NOT `Math.sign`, whose
// signed-zero result (`Math.sign(-0) === -0`) and Rust's `f64::signum`
// (`+1` for `0.0`) both diverge; this explicit form matches across engines.
const mathSign = (x: number): number => {
  if (Number.isNaN(x)) {
    return Number.NaN;
  }

  if (x > 0) {
    return 1;
  }

  return x < 0 ? -1 : 0;
};

/** ISO unary string value functions: one string in, a value out. */
const UNARY_STR: Record<string, (s: string) => unknown> = {
  upper: (s) => s.toUpperCase(),
  lower: (s) => s.toLowerCase(),
  trim: (s) => s.trim(),
  btrim: (s) => s.trim(),
  ltrim: (s) => s.replace(/^\s+/, ''),
  rtrim: (s) => s.replace(/\s+$/, ''),
  char_length: (s) => s.length,
  character_length: (s) => s.length,
};

/** ISO binary numeric value functions: LOG takes (base, value). */
const BINARY_NUM: Record<string, (x: number, y: number) => number> = {
  // KNOWN LIMITATION (won't-fix): V8's `**`/`Math.pow` differs from Rust's
  // `powf` (glibc `pow`, the native engine) by ≤1 ULP on some inputs — e.g.
  // power(0.7,10) → …4ae here vs …4af native; power(2,-0.5) → …bcc vs …bcd. So
  // `power`/`pow`/`^` are NOT byte-identical cross-engine on those inputs; a true
  // fix needs a shared deterministic pow kernel. See docs/dogfood/findings/
  // round15.md and this package's README.md.
  power: (x, y) => x ** y,
  mod: (x, y) => x % y,
  log: (base, value) => Math.log(value) / Math.log(base),
  // atan2(y, x): the ISO GQL binary arctangent (quadrant-correct). Mirrors the
  // native `y.atan2(x)`.
  atan2: (y, x) => Math.atan2(y, x),
};

/** The loud data exception for `avg` over a temporal, or `sum` over a
 *  non-DURATION temporal — byte-identical message to native `FAULT_TEMPORAL_AGG`. */
const unsupportedTemporalAgg = (): LenkeError =>
  new LenkeError(
    "unsupported temporal aggregate: sum() is defined only for DURATION (dates/times aren't " +
      'summable), and avg() over DURATION would need duration/count (often non-representable, ' +
      'e.g. avg(P1M,P2M)=P1.5M); use min()/max(), or sum() + host division',
    { code: ErrorCode.DataException },
  );

/** `sum`/`avg` over a list/map is not numeric — throw loud rather than element-sum
 *  (which is the Gremlin `Scope.local` behavior, not a global aggregate). */
const nonNumericAgg = (): LenkeError =>
  new LenkeError(
    'sum()/avg() require numeric values; a list/map is not summable — reduce it first ' +
      '(Gremlin sum(local), or GQL UNWIND + sum)',
    { code: ErrorCode.DataException },
  );

/** `sum` over gathered temporal values: fold DURATIONs via the same `dur + dur`
 *  (`temporalArith('+')`, which throws on overflow); fault on any non-DURATION. */
const temporalSum = (values: unknown[]): unknown => {
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
const lengthOf = (a: unknown): number | null => {
  if (isNullish(a)) {
    return null;
  }

  if (a instanceof Path) {
    return a.hops;
  }

  return Array.isArray(a) || typeof a === 'string' ? a.length : null;
};

/** Scalar (non-aggregate) functions: the ISO numeric/string value functions. */
const callScalar = (name: string, args: readonly unknown[]): unknown => {
  const [a, b] = args;
  const unaryNum = UNARY_NUM[name];

  if (unaryNum) {
    return isNullish(a) ? null : unaryNum(Number(a));
  }

  const unaryStr = UNARY_STR[name];

  if (unaryStr) {
    return isNullish(a) ? null : unaryStr(str(a));
  }

  const binaryNum = BINARY_NUM[name];

  if (binaryNum) {
    return isNullish(a) || isNullish(b) ? null : binaryNum(Number(a), Number(b));
  }

  if (name in NUM_CONSTANTS) {
    return NUM_CONSTANTS[name];
  }

  switch (name) {
    case 'round': {
      // round(num, [digits]) — digits default 0; half away from zero.
      if (isNullish(a)) {
        return null;
      }

      const digits = isNullish(b) ? 0 : Math.trunc(Number(b));
      const f = 10 ** digits;

      return roundHalfAway(Number(a) * f) / f;
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
        : sanitizeSurrogates(str(a).slice(0, Math.max(0, Number(b))));
    case 'right': {
      if (isNullish(a) || isNullish(b)) {
        return null;
      }

      const s = str(a);
      const n = Number(b);

      return n <= 0 ? '' : sanitizeSurrogates(s.slice(Math.max(0, s.length - n)));
    }
    case 'coalesce':
      return args.find((x) => !isNullish(x)) ?? null;
    case 'nullif':
      // ISO `<case abbreviation>`: NULLIF(a, b) = NULL when a = b, else a.
      return !isNullish(a) && !isNullish(b) && a === b ? null : (a ?? null);
    case 'element_id':
      // ISO `<element_id function>`: the identifier of a node or edge.
      return a && typeof a === 'object' && 'id' in a ? (a as { id: unknown }).id : null;
    default:
      // Graph/conversion/string/list functions live in a second dispatcher so
      // neither switch exceeds the complexity budget.
      return callExtendedScalar(name, args);
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
const FINITE_NUMERIC = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;

const numericStringToFloat = (s: string): number | null => {
  const t = s.trim();

  return FINITE_NUMERIC.test(t) ? Number.parseFloat(t) : null;
};

const toIntScalar = (a: unknown): number | null => {
  if (isNullish(a)) {
    return null;
  }

  if (typeof a === 'number') {
    return Math.trunc(a);
  }

  const p = numericStringToFloat(str(a));

  return p === null ? null : Math.trunc(p);
};

const toFloatScalar = (a: unknown): number | null => {
  if (isNullish(a)) {
    return null;
  }

  if (typeof a === 'number') {
    return a;
  }

  return numericStringToFloat(str(a));
};

const substringScalar = (a: unknown, b: unknown, len: unknown): string | null => {
  if (isNullish(a) || isNullish(b)) {
    return null;
  }

  const s = str(a);
  // ISO GQL: the start index is 1-based (SQL `SUBSTRING` convention), so
  // `substring('crystal hawk river', 1, 7)` → 'crystal'. Convert to a 0-based
  // UTF-16 offset; a start <= 0 shrinks the window from the front (SQL
  // semantics), which the native engine mirrors exactly.
  const zeroStart = Number(b) - 1;
  const from = Math.max(0, zeroStart);

  return sanitizeSurrogates(
    isNullish(len) ? s.slice(from) : s.slice(from, Math.max(0, zeroStart + Number(len))),
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
const fromUtf16UnitsLossy = (units: readonly number[]): string => {
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
const sanitizeSurrogates = (s: string): string =>
  fromUtf16UnitsLossy(Array.from({ length: s.length }, (_, i) => s.charCodeAt(i)));

const splitScalar = (a: unknown, b: unknown): string[] | null => {
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

const replaceScalar = (a: unknown, b: unknown, repl: unknown): string | null => {
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
const toBooleanScalar = (a: unknown): boolean | null => {
  if (isNullish(a)) {
    return null;
  }

  if (typeof a === 'boolean') {
    return a;
  }

  if (typeof a === 'number') {
    return Number.isNaN(a) ? null : a !== 0;
  }

  const t = str(a).trim().toLowerCase();

  if (t === 'true' || t === 'yes' || t === '1') {
    return true;
  }

  return t === 'false' || t === 'no' || t === '0' ? false : null;
};

// ISO GQL `to_list`: a string → its UTF-16 code-unit characters (same unit
// model as `split('')`); a list → itself; any other value → a singleton list.
const toListScalar = (a: unknown): unknown[] | null => {
  if (isNullish(a)) {
    return null;
  }

  if (Array.isArray(a)) {
    return a;
  }

  return typeof a === 'string' ? (splitScalar(a, '') as string[]) : [a];
};

const UTF8 = new TextEncoder();

// UTF-8 byte length (ISO GQL `byte_length` / `octet_length`), matching Rust's
// `str::len()` (a UTF-8 byte count).
const byteLen = (s: string): number => UTF8.encode(s).length;

// ISO GQL `range(start, end, [step])` → an inclusive list of integers. A zero
// step has no defined progression → null. Mirrors the Rust `Range` arm.
const rangeScalar = (a: unknown, b: unknown, step: unknown): number[] | null => {
  if (isNullish(a) || isNullish(b)) {
    return null;
  }

  const s = Math.trunc(Number(a));
  const e = Math.trunc(Number(b));
  const st = isNullish(step) ? 1 : Math.trunc(Number(step));

  if (st === 0) {
    return null;
  }

  const out: number[] = [];

  if (st > 0) {
    for (let i = s; i <= e; i += st) {
      out.push(i);
    }
  } else {
    for (let i = s; i >= e; i += st) {
      out.push(i);
    }
  }

  return out;
};

const headScalar = (a: unknown): unknown => (Array.isArray(a) && a.length > 0 ? a[0] : null);

const lastScalar = (a: unknown): unknown =>
  Array.isArray(a) && a.length > 0 ? a[a.length - 1] : null;

const reverseScalar = (a: unknown): unknown => {
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
const UNHANDLED = Symbol('unhandled');

// Graph functions — label/key order sorted for cross-engine parity.
const callGraphFn = (name: string, a: unknown): unknown => {
  switch (name) {
    case 'labels':
      return isVertex(a) ? [...a.labels].sort() : null;
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
const callConversionFn = (name: string, a: unknown): unknown => {
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
const callStringPredicateFn = (name: string, a: unknown, b: unknown): unknown => {
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
const callStringListFn = (name: string, args: readonly unknown[]): unknown => {
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
      return rangeScalar(a, b, args[2]);
    default:
      return UNHANDLED;
  }
};

// Push `v` into `out` unless a structurally-equal element is already present
// (first occurrence wins) — the dedup building block for the set-style list
// functions, mirroring the Rust `push_unique`.
const pushUnique = (out: unknown[], v: unknown): void => {
  if (!out.some((x) => structuralEq(x, v))) {
    out.push(v);
  }
};

// ISO GQL `list_sort` <nullOrder> arg → the `nullsFirst` flag; `undefined`
// (default / unrecognized) falls back to the ORDER BY default in `compareSort`.
const nullOrderArg = (v: unknown): boolean | undefined => {
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
const callListSetFn = (name: string, a: unknown, b: unknown, args: readonly unknown[]): unknown => {
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
const TEMPORAL_CTOR: Record<
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

const temporalCtor = (
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

const callTemporalFn = (name: string, args: readonly unknown[]): unknown => {
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
const SECS_PER_DAY = 86_400;
const floorDiv = (n: number, d: number): number => Math.floor(n / d);
const euclidMod = (n: number, d: number): number => ((n % d) + d) % d;

/** The integer component, or `null` if the temporal kind lacks it (e.g. `year`
 * of a LOCAL TIME, `hour` of a DATE) — the caller turns `null` into a throw. */
const datePart = (name: string, t: Temporal): number | null => {
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
const DATE_PART_FNS = new Set(['_year', '_month', '_day', '_hour', '_minute', '_second']);

const callDatePartFn = (name: string, a: unknown): unknown => {
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

const callExtendedScalar = (name: string, args: readonly unknown[]): unknown => {
  const [a, b] = args;

  for (const result of [
    callGraphFn(name, a),
    callConversionFn(name, a),
    callDatePartFn(name, a),
    callTemporalFn(name, args),
    callStringPredicateFn(name, a, b),
    callStringListFn(name, args),
    callListSetFn(name, a, b, args),
  ]) {
    if (result !== UNHANDLED) {
      return result;
    }
  }

  throw new LenkeError(`call to an unknown or unimplemented function: ${name}()`, {
    code: ErrorCode.UnknownFunction,
  });
};

// --- expression compilation --------------------------------------------------

/**
 * Lower an expression to a closure. Every `case` resolves its sub-expressions
 * to closures *now* and captures them, so the run-time path is plain function
 * application — no AST re-traversal, no `kind`/`op` dispatch.
 */
// Compile-time side channel: while `compile` walks the tree, every `$name` it
// meets is recorded here so the plan can eager-validate all referenced params
// are bound before it runs (mirrors the Rust engine's `positional`). Set only
// for the duration of one synchronous `compile` call (JS is single-threaded, and
// sub-patterns compile in-line — no re-entrant `compile`), then cleared.
let paramCollector: Set<string> | null = null;

// Compile-time side channel: the names of unknown/unimplemented functions a
// *query* references, gathered while `compile` walks the tree so it can throw
// `UnknownFunction` eagerly (before running, and independent of row count /
// branch reachability). Set only for the duration of one query `compile`; left
// null while a validator predicate compiles, so a validator's unknown fn keeps
// its per-row fault (parity with the Rust `eval_predicate` path). Mirrors the
// Rust plan's `unknown_fns`, checked in `run_cquery_body`.
let unknownFnCollector: Set<string> | null = null;

// Compile-time side channel: the names of `$param`s used as a `LIMIT` / `OFFSET`
// bound. Their bound value must be a non-negative integer, so it is validated
// up-front in the plan closure (mirrors the Rust engine's `check_count_params`),
// making a bad bound fault before any row is produced — identically over zero
// rows or many. The name is also added to `paramCollector` so a missing bound
// param surfaces the usual `MissingParameter` error.
let countParamCollector: Set<string> | null = null;

// Resolve a `LIMIT` / `OFFSET` bound to a concrete count at execution: a literal
// passes through; a `$param` is read from the bound params (its value is already
// validated to be a non-negative integer up-front, in the plan closure).
const resolveCount = (v: CountValue | undefined, params: Params): number | undefined => {
  if (v === undefined || typeof v === 'number') {
    return v;
  }

  return Number(params[v.param]);
};

const compileExpr = (expr: Expr): CompiledExpr => {
  switch (expr.kind) {
    case 'lit': {
      const { value } = expr;

      return () => value;
    }
    case 'var': {
      const { name } = expr;

      return (env) => env.binding.get(name);
    }
    case 'param': {
      const { name } = expr;
      paramCollector?.add(name);

      // Own-property only: a query text referencing `$__proto__` / `$constructor`
      // must read undefined (an unbound param), never `Object.prototype`. The
      // Rust engine is immune (params live in a HashMap); this matches it. A
      // param the caller genuinely passed under that key is an own property and
      // still resolves.
      return (env) => {
        if (Object.hasOwn(env.params, name)) {
          return env.params[name];
        }

        // An unsupplied `$__now` (from a bare `current_*`) reads as null to match
        // the Rust engine; any other unbound param can't reach here (it fails the
        // eager validation above).
        return name === '__now' ? null : undefined;
      };
    }
    case 'prop': {
      const { variable, key } = expr;

      return (env) => propOf(env.binding.get(variable), key);
    }
    case 'list': {
      const items = expr.items.map(compileExpr);

      return (env) => items.map((f) => f(env));
    }
    case 'index': {
      // ISO GQL list subscript `base[index]`: 0-based, out of range → null, and
      // null-safe (non-list base, null / non-integer / negative index → null).
      // `numOf` mirrors the native `num_of` coercion for byte-identity.
      const baseF = compileExpr(expr.base);
      const idxF = compileExpr(expr.index);

      return (env) => {
        const base = baseF(env);
        const i = numOf(idxF(env));

        if (
          !Array.isArray(base) ||
          i === null ||
          !Number.isInteger(i) ||
          i < 0 ||
          i >= base.length
        ) {
          return null;
        }

        return base[i] ?? null;
      };
    }
    case 'field': {
      // Property access chained off any expression (`edges(p)[0].amount`).
      // `propOf` reads a node/edge's stored property; anything else → null, exactly
      // like the bare `prop` path.
      const baseF = compileExpr(expr.base);
      const { key } = expr;

      return (env) => propOf(baseF(env), key);
    }
    case 'func':
      return compileFunc(expr);
    case 'neg': {
      const fn = compileExpr(expr.expr);

      return (env) => {
        const v = numOf(fn(env));

        return v === null ? null : -v;
      };
    }
    case 'arith': {
      // n-ary left-associative fold: `head` then each `[op, operand]`. Every
      // operand is evaluated (no short-circuit — a fault propagates), matching
      // the old left-nested binary tree, but with no per-chain recursion depth.
      const head = compileExpr(expr.head);
      const steps = expr.tail.map(([op, e]) => ({ op, fn: ARITH[op], ce: compileExpr(e) }));

      return (env) => {
        let acc = head(env);

        for (const { op, fn, ce } of steps) {
          acc = arithStep(op, fn, acc, ce(env));
        }

        return acc;
      };
    }
    case 'concat': {
      const parts = expr.items.map(compileExpr);

      return (env) => {
        let acc = parts[0](env);

        for (let i = 1; i < parts.length; i++) {
          acc = concatStep(acc, parts[i](env));
        }

        return acc;
      };
    }
    case 'not': {
      const fn = compileExpr(expr.expr);

      return (env) => not3(asTruth(fn(env)));
    }
    case 'and':
    case 'or':
    case 'xor': {
      const fn = BOOL3[expr.kind];
      const parts = expr.items.map(compileExpr);

      return (env) => {
        let acc = asTruth(parts[0](env));

        for (let i = 1; i < parts.length; i++) {
          acc = fn(acc, asTruth(parts[i](env)));
        }

        return acc;
      };
    }
    case 'isNull': {
      const fn = compileExpr(expr.expr);
      const { negated } = expr;

      return (env) => {
        const isnull = isNullish(fn(env));

        return negated ? !isnull : isnull;
      };
    }
    case 'isTruth': {
      // `x IS [NOT] TRUE|FALSE|UNKNOWN` collapses three-valued logic to a
      // definite boolean: it tests whether x's truth value equals the target.
      const fn = compileExpr(expr.expr);
      const { truth, negated } = expr;

      return (env) => {
        const matches = asTruth(fn(env)) === truth;

        return negated ? !matches : matches;
      };
    }
    case 'isLabeled': {
      // `x IS [NOT] LABELED <label expr>` — does x's label set satisfy it?
      const fn = compileExpr(expr.expr);
      const { label, negated } = expr;

      return (env) => {
        const el = fn(env);
        const has = isElement(el) ? labelsMatch(el.labels, label) : false;

        return negated ? !has : has;
      };
    }
    case 'in': {
      const e = compileExpr(expr.expr);
      const list = compileExpr(expr.list);
      const { negated } = expr;

      return (env) => {
        const result = inList(e(env), list(env));

        return negated ? not3(result) : result;
      };
    }
    case 'compare': {
      const l = compileExpr(expr.left);
      const r = compileExpr(expr.right);
      const { op } = expr;
      const fn = COMPARE[op];

      return (env) => {
        const lv = l(env);
        const rv = r(env);

        if (isNullish(lv) || isNullish(rv)) {
          return null; // UNKNOWN
        }

        // Equality is structural and holds across any types (mismatched types are
        // simply unequal). Ordering is only defined *within* one orderable
        // primitive type (number, string, or boolean) — comparing a number to a
        // string, or two graph elements, is UNKNOWN per ISO, not a JS coercion.
        if (op === '=' || op === '<>') {
          const eq = structuralEq(lv, rv);

          return op === '=' ? eq : !eq;
        }

        // Temporals: date/datetime (same kind) order chronologically; durations
        // and cross-kind pairs are UNKNOWN. `fn(c, 0)` applies the operator to the
        // -1/0/1 comparison result (e.g. `<` becomes `c < 0`).
        if (isTemporal(lv) && isTemporal(rv)) {
          const c = temporalRelCmp(lv, rv);

          return c === null ? null : fn(c, 0);
        }

        // A temporal vs a non-temporal relational comparison (both-temporal was
        // handled above) is a type error — an untagged string param vs a stored
        // DATE is a mistake, not "no rows" — so fault instead of silently UNKNOWN.
        // Byte-identical to native's FAULT_CMP_TEMPORAL.
        if (isTemporal(lv) !== isTemporal(rv)) {
          throw new LenkeError(
            "cannot order-compare a temporal value with a non-temporal value; tag the literal (e.g. DATE '2024-01-01') or CAST it to the matching type",
            { code: ErrorCode.InvalidValue },
          );
        }

        const t = typeof lv;
        const orderable = t === typeof rv && (t === 'number' || t === 'string' || t === 'boolean');

        if (!orderable) {
          return null; // UNKNOWN
        }

        return fn(lv as number | string, rv as number | string);
      };
    }
    case 'case':
      return compileCase(expr);
    case 'exists':
      return compileExists(expr);
    case 'countSubquery':
      return compileCountSubquery(expr);
  }
};

/**
 * Compile a braced subquery body (`{ pattern, … [WHERE pred] }`) into a MATCH
 * clause. The sub-pattern is compiled once; at run time it is matched seeded with
 * the outer binding, so EXISTS/COUNT are correlated.
 */
const compileSubMatch = (sub: { patterns: readonly PathPattern[]; where?: Expr }): CMatch => ({
  kind: 'match',
  optional: false,
  patterns: sub.patterns.map(compilePath),
  where: sub.where ? compileExpr(sub.where) : undefined,
  nullVars: [],
});

/**
 * Reachability fast path for `EXISTS { (a)-[:T]->+/*(b …) }`: a single unbounded
 * var-length segment from an already-bound `a` is *reachability* — BFS the reached
 * set and stop at the first vertex satisfying the endpoint, instead of enumerating
 * trails (exponential; testing an *unreachable* target hits the trail budget and
 * faults). Mirrors the native `any_match_reachable`. Returns `undefined` when the
 * shape doesn't apply.
 */
const existsReachable = (
  graph: Graph,
  sub: CMatch,
  binding: Binding,
  params: Params,
): boolean | undefined => {
  if (sub.patterns.length !== 1 || sub.patterns[0].segments.length !== 1) {
    return undefined;
  }

  const [path] = sub.patterns;
  const [{ rel, node }] = path.segments;
  const { quantifier: q } = rel;

  if (!q) {
    return undefined;
  }

  const startVar = path.start.variable;
  const types = relTypeNames(rel.label);

  if (
    q.max !== null ||
    rel.variable !== undefined ||
    rel.direction === 'both' ||
    rel.pred.props.length > 0 ||
    rel.pred.where !== undefined ||
    types === null ||
    startVar === undefined ||
    !binding.has(startVar) ||
    path.start.pred.props.length > 0 ||
    path.start.pred.where !== undefined
  ) {
    return undefined;
  }

  const startV = binding.get(startVar) as Vertex;
  const out = rel.direction === 'out';
  const nbrs = (v: Vertex): Vertex[] => {
    const byType = (out ? graph.edgesFromByLabel : graph.edgesToByLabel).get(v.id);
    const acc: Vertex[] = [];

    for (const set of bucketsFor(byType, types ?? undefined)) {
      if (set) {
        for (const e of set) {
          acc.push(out ? e.to : e.from);
        }
      }
    }

    return acc;
  };
  // Is `v` a valid endpoint (label + inline pred + the EXISTS WHERE)?
  const hit = (v: Vertex): boolean => {
    const bound = matchNode(binding, node, v, params, graph);

    return (
      bound !== null &&
      (sub.where === undefined || asTruth(sub.where({ binding: bound, params, graph })) === true)
    );
  };

  // `->*` also admits the zero-length path — the start itself.
  if (q.min === 0 && hit(startV)) {
    return true;
  }

  const seen = new Set<string>();
  const stack: Vertex[] = [];
  const visit = (w: Vertex): boolean => {
    if (seen.has(w.id)) {
      return false;
    }

    seen.add(w.id);
    stack.push(w);

    return true;
  };

  for (const w of nbrs(startV)) {
    if (visit(w) && hit(w)) {
      return true;
    }
  }

  while (stack.length > 0) {
    for (const w of nbrs(stack.pop()!)) {
      if (visit(w) && hit(w)) {
        return true;
      }
    }
  }

  return false;
};

/** ISO EXISTS: TRUE iff the correlated sub-pattern has at least one match. */
const compileExists = (expr: Extract<Expr, { kind: 'exists' }>): CompiledExpr => {
  const sub = compileSubMatch(expr);

  return (env) => {
    const reach = existsReachable(env.graph, sub, env.binding, env.params);

    if (reach !== undefined) {
      return reach;
    }

    const matches = matchClauseBindings(env.graph, sub, env.binding, env.params)[Symbol.iterator]();

    return !matches.next().done;
  };
};

/** ISO count subquery: the number of matches of the correlated sub-pattern. */
const compileCountSubquery = (expr: Extract<Expr, { kind: 'countSubquery' }>): CompiledExpr => {
  const sub = compileSubMatch(expr);

  return (env) => [...matchClauseBindings(env.graph, sub, env.binding, env.params)].length;
};

/**
 * Compile an ISO CASE expression. A simple CASE (with `subject`) returns the
 * first branch whose value equals the subject; a searched CASE returns the first
 * branch whose condition is exactly TRUE. No match falls to ELSE (or NULL).
 */
const compileCase = (expr: Extract<Expr, { kind: 'case' }>): CompiledExpr => {
  const subject = expr.subject ? compileExpr(expr.subject) : undefined;
  // `then` is the ISO GQL CASE…WHEN…THEN branch, not a thenable; never awaited.
  // eslint-disable-next-line unicorn/no-thenable
  const whens = expr.whens.map((w) => ({ when: compileExpr(w.when), then: compileExpr(w.then) }));
  const elseFn = expr.elseExpr ? compileExpr(expr.elseExpr) : undefined;

  return (env) => {
    if (subject) {
      const s = subject(env);

      for (const w of whens) {
        const wv = w.when(env);

        // `subject = when` with SQL/ISO null semantics: NULL never matches.
        if (!isNullish(s) && !isNullish(wv) && s === wv) {
          return w.then(env);
        }
      }
    } else {
      for (const w of whens) {
        if (asTruth(w.when(env)) === true) {
          return w.then(env);
        }
      }
    }

    return elseFn ? elseFn(env) : null;
  };
};

const compileFunc = (expr: FuncExpr): CompiledExpr => {
  if (AGGREGATES.has(expr.name)) {
    return compileAggregate(expr);
  }

  const { name } = expr;
  const args = expr.args.map(compileExpr);

  // Resolve the function NAME eagerly while a *query* compiles — before any row
  // runs. An unknown function is never valid regardless of row count or branch
  // reachability, so `bogus_fn(x)` must fault identically over zero rows, one
  // row, or inside a never-taken `CASE` branch. Previously the `UnknownFunction`
  // fault fired only from the per-row `callScalar`, so an empty result set (or a
  // dead branch) silently returned `[]`. The name is recorded into the
  // query-scoped `unknownFnCollector`; `compile` throws once it finishes walking
  // the tree (mirrors the Rust plan's `unknown_fns`, checked in `run_cquery_body`
  // before the first row). A validator predicate compiles with the collector
  // unset, so its unknown-fn timing stays per-row — matching the Rust
  // `eval_predicate` path (both engines surface a validator's unknown fn at write
  // time, not declare time).
  if (unknownFnCollector && !isKnownScalarFn(name)) {
    unknownFnCollector.add(name);
  }

  return (env) =>
    callScalar(
      name,
      args.map((f) => f(env)),
    );
};

// A shared empty probe arg list — `callScalar` treats absent args as null, so a
// KNOWN scalar function resolves to null (never throwing `UnknownFunction`);
// only a genuinely unknown name reaches the `UnknownFunction` throw.
const FN_PROBE_ARGS: readonly unknown[] = [];

/**
 * Compile-time name resolution for a scalar function. Probes the shared scalar
 * dispatch with placeholder args: the ONLY source of `ErrorCode.UnknownFunction`
 * is an unresolved name (every known function returns — or throws some *other*
 * error — on null args), so a thrown `UnknownFunction` means the name is bogus.
 * Any other throw means the name DID resolve (its real per-row error, e.g. bad
 * arity, still stands). Reachability-independent by construction — the name is
 * resolved whether or not the call ever executes.
 */
const isKnownScalarFn = (name: string): boolean => {
  try {
    callScalar(name, FN_PROBE_ARGS);

    return true;
  } catch (error) {
    return !(error instanceof LenkeError && error.code === ErrorCode.UnknownFunction);
  }
};

/**
 * Compile an aggregate. The argument expression is lowered once; at run time we
 * fold it over the `group` of bindings (or `[binding]` when called outside an
 * aggregating projection).
 */
const compileAggregate = (expr: FuncExpr): CompiledExpr => {
  const { name, star, distinct } = expr;

  // ISO forbids an aggregate whose argument contains another aggregate.
  if (expr.args[0] && hasAggregate(expr.args[0])) {
    throw new LenkeError(`aggregate function ${name}() cannot contain another aggregate`, {
      code: ErrorCode.Unsupported,
    });
  }

  // `count(*)` is the only aggregate with no argument expression; anything else
  // argless (`sum()`, bare `count()`) is malformed — reject it cleanly rather
  // than dereferencing an absent argument at fold time.
  if (!expr.args[0] && !(name === 'count' && star)) {
    throw new LenkeError(`aggregate function ${name}() requires an argument`, {
      code: ErrorCode.Unsupported,
    });
  }

  // Percentile aggregates take `(value, literal fraction)`. A malformed call
  // (wrong arity / non-literal fraction) is rejected, mirroring the native engine.
  const isPercentile = name === 'percentile_cont' || name === 'percentile_disc';
  let frac = 0;

  if (isPercentile) {
    const [, f] = expr.args;

    if (expr.args.length !== 2 || f?.kind !== 'lit' || typeof f.value !== 'number') {
      throw new LenkeError(`${name}() requires a numeric literal fraction`, {
        code: ErrorCode.Unsupported,
      });
    }

    frac = Math.min(1, Math.max(0, f.value));
  }

  const argFn = expr.args[0] ? compileExpr(expr.args[0]) : undefined;

  return (env) => {
    const group = env.group ?? [env.binding];

    if (name === 'count' && star) {
      return group.length;
    }

    const raw = group.map((b) => argFn!({ ...env, binding: b, group }));
    const nonNull = raw.filter((v) => !isNullish(v));
    const values = distinct ? [...new Set(nonNull)] : nonNull;

    switch (name) {
      case 'count':
        return values.length;
      // `sum` over DURATIONs folds via the same `dur + dur` (overflow throws);
      // `avg` over any temporal, and `sum` over a non-DURATION, are loud data
      // exceptions rather than a silent NaN → null. Mirrors the native engine.
      // A temporal ANYWHERE in the group (not just the first row) makes the
      // aggregate unrepresentable — a heterogeneous numeric+temporal column must
      // throw, not silently coerce the temporal to `NaN`→`null`. Native faults
      // per-value, so we scan for any temporal to stay byte-identical.
      case 'sum':
        if (values.some(isTemporal)) {
          return temporalSum(values);
        }

        if (values.some(Array.isArray)) {
          throw nonNumericAgg();
        }

        return values.reduce<number>((s, v) => s + Number(v), 0);
      case 'avg':
        if (values.some(isTemporal)) {
          throw unsupportedTemporalAgg();
        }

        if (values.some(Array.isArray)) {
          throw nonNumericAgg();
        }

        return values.length === 0
          ? null
          : values.reduce<number>((s, v) => s + Number(v), 0) / values.length;
      case 'min':
        return values.length === 0
          ? null
          : values.reduce((m, v) => (compareValues(v, m) < 0 ? v : m));
      case 'max':
        return values.length === 0
          ? null
          : values.reduce((m, v) => (compareValues(v, m) > 0 ? v : m));
      case 'collect_list':
        return values;
      case 'percentile_cont':
        return percentileOf(values, frac, true);
      case 'percentile_disc':
        return percentileOf(values, frac, false);
      // ISO population / sample standard deviation, one-pass over the group's
      // numeric values — the SAME formula the native engine uses, so the f64
      // result is byte-identical. `stddev_pop` null over 0 rows; `stddev_samp`
      // null over fewer than 2; the summed squared deviation is clamped at 0 so
      // float cancellation can't slip a tiny negative into sqrt.
      case 'stddev_pop':
      case 'stddev_samp': {
        const sample = name === 'stddev_samp';
        const n = values.length;

        if (sample ? n < 2 : n === 0) {
          return null;
        }

        let s = 0;
        let sq = 0;

        for (const v of values) {
          const x = Number(v);
          s += x;
          sq += x * x;
        }

        const variance = (sq - (s * s) / n) / (sample ? n - 1 : n);

        return Math.sqrt(Math.max(0, variance));
      }
      default:
        return null;
    }
  };
};

/**
 * Compile a standalone boolean predicate (a declarative VALIDATOR constraint,
 * see `@lenke/gql`'s `createValidator`) into a closure that evaluates it against
 * a single graph element bound to `varName`, with empty params. Returns the
 * three-valued result — `true` / `false` / `null` (UNKNOWN) — computed by the
 * *same* expression evaluator a `WHERE` clause uses, so a validator and a `WHERE`
 * agree bit-for-bit. SQL-`CHECK` callers reject only on a definite `false`; a
 * `null` passes. `graph` is read only by an EXISTS/COUNT subquery in the
 * predicate (rare in a validator, but supported for parity with `WHERE`).
 */
export const compileValidator = (
  expr: Expr,
  varName: string,
): ((element: Vertex | Edge, graph: Graph) => boolean | null) => {
  const fn = compileExpr(expr);

  return (element, graph) => {
    const binding: Binding = new Map([[varName, element]]);

    return asTruth(fn({ binding, params: {}, graph }));
  };
};

/** Collect the variables a sub-pattern introduces (start node, each hop's rel + node). */
const patternBoundVars = (p: PathPattern, into: Set<string>): void => {
  const addNode = (n: NodePattern): void => {
    if (n.variable !== undefined) {
      into.add(n.variable);
    }
  };

  if (p.pathVar !== undefined) {
    into.add(p.pathVar);
  }

  addNode(p.start);

  for (const seg of p.segments) {
    if (seg.rel.variable !== undefined) {
      into.add(seg.rel.variable);
    }

    addNode(seg.node);
  }
};

/**
 * Collect every FREE variable a predicate references — a `var`/`prop` name that
 * is NOT bound by an enclosing `EXISTS`/`COUNT` sub-pattern. A VALIDATOR
 * predicate has exactly one legitimate free variable, the declared `varName`
 * (the element under test); a reference to any *other* free name (a typo like
 * `x.age` when the binding is `u`, or a bare `age`) is unbound, so the predicate
 * silently evaluates to UNKNOWN and the SQL-`CHECK` never fires. `createValidator`
 * walks this set and rejects such a predicate at declare time. Sub-query pattern
 * variables are bound *within* the sub-query, so they are correctly NOT free and
 * must not be flagged. Mirrors the Rust `free_predicate_vars` (`plan.rs`).
 */
export const freePredicateVars = (expr: Expr): Set<string> => {
  const free = new Set<string>();

  const walkPattern = (p: PathPattern, bound: ReadonlySet<string>): void => {
    const walkNode = (n: NodePattern): void => {
      for (const c of n.properties ?? []) {
        walk(c.value, bound);
      }

      if (n.where) {
        walk(n.where, bound);
      }
    };

    walkNode(p.start);

    for (const seg of p.segments) {
      for (const c of seg.rel.properties ?? []) {
        walk(c.value, bound);
      }

      if (seg.rel.where) {
        walk(seg.rel.where, bound);
      }

      walkNode(seg.node);
    }
  };

  const walk = (e: Expr, bound: ReadonlySet<string>): void => {
    switch (e.kind) {
      case 'var':
        if (!bound.has(e.name)) {
          free.add(e.name);
        }

        return;
      case 'prop':
        if (!bound.has(e.variable)) {
          free.add(e.variable);
        }

        return;
      case 'lit':
      case 'param':
        return;
      case 'index':
        walk(e.base, bound);
        walk(e.index, bound);

        return;
      case 'field':
        walk(e.base, bound);

        return;
      case 'neg':
      case 'not':
      case 'isNull':
      case 'isTruth':
      case 'isLabeled':
        walk(e.expr, bound);

        return;
      case 'compare':
        walk(e.left, bound);
        walk(e.right, bound);

        return;
      case 'arith':
        walk(e.head, bound);

        for (const [, el] of e.tail) {
          walk(el, bound);
        }

        return;
      case 'list':
      case 'concat':
      case 'and':
      case 'or':
      case 'xor':
        for (const el of e.items) {
          walk(el, bound);
        }

        return;
      case 'in':
        walk(e.expr, bound);
        walk(e.list, bound);

        return;
      case 'case':
        if (e.subject) {
          walk(e.subject, bound);
        }

        for (const w of e.whens) {
          walk(w.when, bound);
          walk(w.then, bound);
        }

        if (e.elseExpr) {
          walk(e.elseExpr, bound);
        }

        return;
      case 'func':
        for (const a of e.args) {
          walk(a, bound);
        }

        return;
      case 'exists':
      case 'countSubquery': {
        // The sub-pattern binds its own variables; extend the bound set before
        // descending into its inline predicates and WHERE so those bindings are
        // not mistaken for free references. Outer names still read as free.
        const inner = new Set(bound);

        for (const p of e.patterns) {
          patternBoundVars(p, inner);
        }

        for (const p of e.patterns) {
          walkPattern(p, inner);
        }

        if (e.where) {
          walk(e.where, inner);
        }

        return;
      }
    }
  };

  walk(expr, new Set());

  return free;
};

// --- value / ordering helpers ------------------------------------------------

/** Derive a column name for a RETURN item that has no explicit `AS` alias. */
const columnName = (expr: Expr): string => {
  switch (expr.kind) {
    case 'var':
      return expr.name;
    case 'prop':
      return `${expr.variable}.${expr.key}`;
    default:
      return 'expr';
  }
};

/** A coarse type ordering so ORDER BY/min/max have a *total* order across types. */
const typeRank = (v: unknown): number => {
  switch (typeof v) {
    case 'number':
      return 0;
    case 'string':
      return 1;
    case 'boolean':
      return 2;
    default:
      return isTemporal(v) ? 3 : 4; // temporal, then graph elements/lists/objects
  }
};

/**
 * Compare two values for ORDER BY; nulls sort last. Values of different types
 * are ordered by a fixed type rank first (number < string < boolean < other),
 * so a column mixing types has a deterministic total order rather than the
 * unstable result of raw JS `<` across types.
 */
const compareValues = (a: unknown, b: unknown): number => {
  if (isNullish(a) && isNullish(b)) {
    return 0;
  }

  if (isNullish(a)) {
    return 1;
  }

  if (isNullish(b)) {
    return -1;
  }

  const ra = typeRank(a);
  const rb = typeRank(b);

  if (ra !== rb) {
    return ra < rb ? -1 : 1;
  }

  // Temporals (same rank) compare by the deterministic total order (date/datetime
  // chronological, duration lexicographic) — mirrors the Rust `cmp_total`.
  if (isTemporal(a) && isTemporal(b)) {
    return temporalCmpTotal(a, b);
  }

  // Lists compare element-wise (lexicographic, shorter-is-less on a prefix),
  // recursing through the same order — mirrors the Rust `cmp_total`. Without this
  // they'd fall to `x < y`, which coerces arrays to strings (`[10] < [9]`), so
  // `min`/`max` and `ORDER BY` over lists would diverge from the native engine.
  if (Array.isArray(a) && Array.isArray(b)) {
    const n = Math.min(a.length, b.length);

    for (let i = 0; i < n; i++) {
      const c = compareValues(a[i], b[i]);

      if (c !== 0) {
        return c;
      }
    }

    if (a.length < b.length) {
      return -1;
    }

    return a.length > b.length ? 1 : 0;
  }

  const x = a as number | string;
  const y = b as number | string;

  if (x < y) {
    return -1;
  }

  return x > y ? 1 : 0;
};

/**
 * Compare two ORDER BY keys, honoring direction and ISO `NULLS FIRST/LAST`. Null
 * placement is absolute (first or last in the final order), independent of the
 * direction applied to non-null values. With no explicit null ordering, nulls
 * sort LAST (ISO GQL leaves the default unspecified, so we pin one for
 * cross-engine determinism — the Rust `compare_sort` matches).
 */
const compareSort = (
  a: unknown,
  b: unknown,
  descending: boolean,
  nullsFirst: boolean | undefined,
): number => {
  const aNull = isNullish(a);
  const bNull = isNullish(b);

  if (aNull && bNull) {
    return 0;
  }

  if (aNull || bNull) {
    const first = nullsFirst ?? false;

    return aNull === first ? -1 : 1;
  }

  return compareValues(a, b) * (descending ? -1 : 1);
};

/** Stable distinct key for a projected binding; graph elements key by id. */
const valueKey = (v: unknown): string => {
  if (v && typeof v === 'object' && 'id' in v) {
    return `@${String((v as { id: unknown }).id)}`;
  }

  // `JSON.stringify` maps NaN and ±Infinity all to `"null"`, collapsing them
  // into the real-null group (and each other). Tag non-finite numbers distinctly.
  if (typeof v === 'number' && !Number.isFinite(v)) {
    return `#${String(v)}`;
  }

  return JSON.stringify(v) ?? 'undefined';
};
const rowKey = (b: Binding): string => [...b].map(([k, v]) => `${k}=${valueKey(v)}`).join('');

// --- projection compilation --------------------------------------------------

/** A projected output column: its name and the closure producing its value. */
type CReturnItem = { name: string; fn: CompiledExpr; isAgg: boolean };
type CSortItem = { fn: CompiledExpr; descending: boolean; nullsFirst?: boolean };

/**
 * A compiled projection body (shared by `RETURN` and `WITH`). All the structural
 * analysis — alias resolution, aggregate detection, picking the GROUP BY keys —
 * is done here, once.
 */
type CProjection = {
  star: boolean;
  distinct: boolean;
  items: readonly CReturnItem[];
  /** True when any non-`*` item aggregates → implicit grouping kicks in. */
  aggregating: boolean;
  /** The non-aggregate item closures, used to build each group's key. */
  groupKeys: readonly CompiledExpr[];
  orderBy: readonly CSortItem[];
  skip?: CountValue;
  limit?: CountValue;
};

// Register a LIMIT/OFFSET bound param so it is both bound-checked (MissingParameter)
// and value-checked (non-negative integer) up-front, exactly like the Rust plan.
const noteCountParam = (v: CountValue | undefined): void => {
  if (v !== undefined && typeof v === 'object') {
    paramCollector?.add(v.param);
    countParamCollector?.add(v.param);
  }
};

const compileProjection = (projection: Projection): CProjection => {
  const items: CReturnItem[] = projection.items.map((i) => ({
    name: i.alias ?? columnName(i.expr),
    fn: compileExpr(i.expr),
    isAgg: hasAggregate(i.expr),
  }));
  const aggregating = !projection.star && items.some((i) => i.isAgg);
  noteCountParam(projection.skip);
  noteCountParam(projection.limit);
  const groupKeys = items.filter((i) => !i.isAgg).map((i) => i.fn);
  // ORDER BY keys are evaluated against the projected output overlaid on the
  // input binding (see `applyProjection`), so output aliases resolve even inside
  // an expression — `ORDER BY n + 2` uses the column `n`, not the input variable.
  const orderBy: CSortItem[] = (projection.orderBy ?? []).map((s) => ({
    fn: compileExpr(s.expr),
    descending: s.descending,
    nullsFirst: s.nullsFirst,
  }));

  return {
    star: projection.star,
    distinct: projection.distinct,
    items,
    aggregating,
    groupKeys,
    orderBy,
    skip: projection.skip,
    limit: projection.limit,
  };
};

/** Build the output binding for one input binding (or aggregate group). */
const projectBinding = (
  proj: CProjection,
  binding: Binding,
  params: Params,
  graph: Graph,
  group?: readonly Binding[],
): Binding => {
  if (proj.star) {
    return new Map(binding);
  }

  const env: EvalEnv = { binding, params, graph, group };
  const out = new Map<string, unknown>();

  for (const item of proj.items) {
    out.set(item.name, item.fn(env));
  }

  return out;
};

/**
 * The `cap` rows that sort first under `cmp`, in sorted order — an O(n log cap)
 * bounded selection instead of sorting all n rows to keep a small prefix. Streams
 * its input (only `cap` rows are ever held). Ties break by original stream
 * position, so the result is byte-identical to a *stable* full sort + `slice(0,
 * cap)`. Used for `ORDER BY … LIMIT k`.
 */
const boundedTopK = <T>(rows: Iterable<T>, cap: number, cmp: (a: T, b: T) => number): T[] => {
  if (cap <= 0) {
    return [];
  }

  type E = { v: T; i: number };
  const heap: E[] = []; // max-heap by `less`: the root is the worst (largest) kept
  const less = (a: E, b: E): boolean => {
    const c = cmp(a.v, b.v);

    return c !== 0 ? c < 0 : a.i < b.i; // index tiebreak reproduces stable order
  };
  const swap = (x: number, y: number) => {
    const t = heap[x];
    heap[x] = heap[y];
    heap[y] = t;
  };
  const up = (start: number) => {
    let i = start;

    while (i > 0) {
      const p = (i - 1) >> 1;

      if (!less(heap[p], heap[i])) {
        break; // parent already >= child
      }

      swap(i, p);
      i = p;
    }
  };
  const down = (start: number) => {
    let i = start;

    for (;;) {
      const l = i * 2 + 1;
      const r = l + 1;
      let m = i;

      if (l < heap.length && less(heap[m], heap[l])) {
        m = l;
      }

      if (r < heap.length && less(heap[m], heap[r])) {
        m = r;
      }

      if (m === i) {
        break;
      }

      swap(i, m);
      i = m;
    }
  };

  let idx = 0;

  for (const v of rows) {
    const e = { v, i: idx };
    idx += 1;

    if (heap.length < cap) {
      heap.push(e);
      up(heap.length - 1);
    } else if (less(e, heap[0])) {
      heap[0] = e; // better than the worst kept — evict the root
      down(0);
    }
  }

  heap.sort((a, b) => {
    if (less(a, b)) {
      return -1;
    }

    return less(b, a) ? 1 : 0;
  });

  return heap.map((e) => e.v);
};

/**
 * Apply a projection (`RETURN` or `WITH` body) to a set of bindings: implicit
 * grouping/aggregation, then DISTINCT, ORDER BY, SKIP, LIMIT. Returns the
 * projected bindings — `RETURN` turns these into rows, `WITH` feeds them on.
 */
const applyProjection = (
  proj: CProjection,
  bindings: Iterable<Binding>,
  params: Params,
  graph: Graph,
): Iterable<Binding> => {
  const { orderBy } = proj;
  // A `$param` bound resolves here (validated up-front); a literal passes through.
  const skipBound = resolveCount(proj.skip, params);
  const limitBound = resolveCount(proj.limit, params);
  type Keyed = { b: Binding; keys: readonly unknown[] };
  let keyed: Iterable<Keyed>;

  if (proj.aggregating) {
    // Grouping is a barrier — it must see every binding before it can emit. We
    // still fold into per-group buckets with single `push`es (never a spread).
    const groups = new Map<string, Binding[]>();

    for (const b of bindings) {
      const key = JSON.stringify(
        proj.groupKeys.map((fn) => valueKey(fn({ binding: b, params, graph }))),
      );
      const existing = groups.get(key);

      if (existing) {
        existing.push(b);
      } else {
        groups.set(key, [b]);
      }
    }

    if (groups.size === 0 && proj.groupKeys.length === 0) {
      groups.set('[]', []);
    }

    keyed = map((group: Binding[]) => {
      const rep: Binding = group[0] ?? new Map();
      const projected = projectBinding(proj, rep, params, graph, group);
      // ORDER BY sees the output columns overlaid on the input variables.
      const sortBinding = orderBy.length > 0 ? new Map([...rep, ...projected]) : rep;

      return {
        b: projected,
        keys: orderBy.map((s) => s.fn({ binding: sortBinding, params, graph, group })),
      };
    }, groups.values());
  } else {
    // Non-aggregating: a lazy map — rows are projected on demand.
    keyed = map((b: Binding) => {
      const projected = projectBinding(proj, b, params, graph);
      const sortBinding = orderBy.length > 0 ? new Map([...b, ...projected]) : b;

      return {
        b: projected,
        keys: orderBy.map((s) => s.fn({ binding: sortBinding, params, graph })),
      };
    }, bindings);
  }

  if (proj.distinct) {
    const seen = new Set<string>();
    keyed = filter((r: Keyed) => {
      const k = rowKey(r.b);

      if (seen.has(k)) {
        return false;
      }

      seen.add(k);

      return true;
    }, keyed);
  }

  // ORDER BY is the other barrier. With a LIMIT we only need skip+limit rows, so a
  // bounded top-k (O(n log k), never materializing the rest) beats sorting all n;
  // without a LIMIT, sort the whole owned array.
  const cmp = (a: Keyed, b: Keyed): number => {
    for (let i = 0; i < orderBy.length; i += 1) {
      const c = compareSort(a.keys[i], b.keys[i], orderBy[i].descending, orderBy[i].nullsFirst);

      if (c !== 0) {
        return c;
      }
    }

    return 0;
  };
  let ordered: Iterable<Keyed> = keyed;

  if (orderBy.length > 0 && limitBound !== undefined) {
    ordered = boundedTopK(keyed, (skipBound ?? 0) + limitBound, cmp);
  } else if (orderBy.length > 0) {
    const arr = toArray(keyed);
    arr.sort(cmp);
    ordered = arr;
  }

  // SKIP/LIMIT stay lazy — `take` short-circuits, so `LIMIT n` over a huge
  // unordered stream stops after n rows instead of computing them all.
  const start = skipBound ?? 0;
  let sliced: Iterable<Keyed> = start > 0 ? skip(start, ordered) : ordered;

  if (limitBound !== undefined) {
    sliced = take(limitBound, sliced);
  }

  return map((r: Keyed) => r.b, sliced);
};

// --- pattern compilation -----------------------------------------------------

/** A compiled property map + inline WHERE (the ISO element-pattern predicate). */
type CProp = { key: string; value: CompiledExpr };
type CPredicate = { props: readonly CProp[]; where?: CompiledExpr };

/** Range bounds whose endpoints are compiled value closures (resolved per seed). */
type CRangeBound = { gt?: CompiledExpr; gte?: CompiledExpr; lt?: CompiledExpr; lte?: CompiledExpr };

/**
 * A seedable predicate lifted out of a WHERE / inline-pattern conjunction: a
 * necessary condition on a node's property that an index can seek. Sound only
 * because it comes from an AND-chain (every conjunct must hold), so the seed is
 * always a superset of the node's true matches — `matchNode` re-validates.
 */
type CSeedHint =
  | { kind: 'eq'; key: string; value: CompiledExpr }
  | { kind: 'within'; key: string; values: CompiledExpr }
  | { kind: 'range'; key: string; bound: CRangeBound };

type CNode = {
  variable?: string;
  label?: LabelExpr;
  pred: CPredicate;
  seedHints?: readonly CSeedHint[];
};
type CRel = {
  variable?: string;
  label?: LabelExpr;
  direction: RelPattern['direction'];
  pred: CPredicate;
  quantifier?: RelPattern['quantifier'];
};
type CSegment = { rel: CRel; node: CNode };
type CPath = {
  start: CNode;
  segments: readonly CSegment[];
  /** Whole path bound to this variable (`p = …`), or unnamed. */
  pathVar?: string;
  /** Which matching paths to keep; defaults to `walk`. */
  selector: PathSelector;
  /** The repeated-element restrictor on a var-length walk; defaults to `trail`. */
  mode: PathMode;
};

const compileProps = (props: readonly PropertyConstraint[] | undefined): CProp[] =>
  (props ?? []).map(({ key, value }) => ({ key, value: compileExpr(value) }));

const compilePredicate = (
  properties: readonly PropertyConstraint[] | undefined,
  where: Expr | undefined,
): CPredicate => ({
  props: compileProps(properties),
  where: where ? compileExpr(where) : undefined,
});

// --- seed-hint extraction ----------------------------------------------------

/** A value usable as a seek key without binding the node's own variable. */
const isConstExpr = (e: Expr): boolean => e.kind === 'lit' || e.kind === 'param';

/** Mirror a comparison operator when its operands are swapped (`30 < a.age`). */
const FLIP: Record<CompareOp, CompareOp> = {
  '=': '=',
  '<>': '<>',
  '<': '>',
  '>': '<',
  '<=': '>=',
  '>=': '<=',
};

type HintMap = Map<string, CSeedHint[]>;

const pushHint = (into: HintMap, variable: string, hint: CSeedHint): void => {
  const list = into.get(variable);

  if (list) {
    list.push(hint);
  } else {
    into.set(variable, [hint]);
  }
};

/** A `prop <op> const` comparison, normalized so the property is on the left. */
const asPropCompare = (
  expr: Extract<Expr, { kind: 'compare' }>,
): { variable: string; key: string; op: CompareOp; value: Expr } | null => {
  if (expr.left.kind === 'prop' && isConstExpr(expr.right)) {
    return { variable: expr.left.variable, key: expr.left.key, op: expr.op, value: expr.right };
  }

  if (expr.right.kind === 'prop' && isConstExpr(expr.left)) {
    return {
      variable: expr.right.variable,
      key: expr.right.key,
      op: FLIP[expr.op],
      value: expr.left,
    };
  }

  return null;
};

const BOUND_OF: Partial<Record<CompareOp, keyof CRangeBound>> = {
  '>': 'gt',
  '>=': 'gte',
  '<': 'lt',
  '<=': 'lte',
};

/**
 * Walk a predicate's AND-chain, collecting per-variable seed hints from the
 * conjuncts an index can seek: `prop = const`, range comparisons, and `prop IN
 * [consts]`. Only `and` is descended — an `or`/`not` branch could admit rows a
 * single-conjunct seed would miss, so those (and every non-seekable shape) are
 * left entirely to the residual WHERE.
 */
const collectHints = (where: Expr, into: HintMap): void => {
  switch (where.kind) {
    case 'and':
      for (const conjunct of where.items) {
        collectHints(conjunct, into);
      }

      return;
    case 'compare': {
      const pc = asPropCompare(where);

      if (!pc) {
        return;
      }

      if (pc.op === '=') {
        pushHint(into, pc.variable, { kind: 'eq', key: pc.key, value: compileExpr(pc.value) });

        return;
      }

      const boundKey = BOUND_OF[pc.op];

      if (boundKey) {
        pushHint(into, pc.variable, {
          kind: 'range',
          key: pc.key,
          bound: { [boundKey]: compileExpr(pc.value) },
        });
      }

      return;
    }
    case 'in':
      if (
        !where.negated &&
        where.expr.kind === 'prop' &&
        where.list.kind === 'list' &&
        where.list.items.every(isConstExpr)
      ) {
        pushHint(into, where.expr.variable, {
          kind: 'within',
          key: where.expr.key,
          values: compileExpr(where.list),
        });
      }

      return;
    default:
  }
};

/** Hints a predicate contributes to one variable (used for inline node WHERE). */
const hintsForVariable = (where: Expr | undefined, variable: string | undefined): CSeedHint[] => {
  if (!where || !variable) {
    return [];
  }

  const into: HintMap = new Map();
  collectHints(where, into);

  return coalesceRangeHints(into.get(variable) ?? []);
};

/**
 * Fold a variable's range hints on the same key into one bound, so
 * `n.age >= 29 AND n.age < 35` seeks the tight `[29, 35)` slice rather than
 * just the more selective single side. First-wins on a repeated side (e.g. two
 * lower bounds) — dropping a redundant tightening only widens the seed, which
 * stays a sound superset for `matchNode` to re-validate.
 */
const coalesceRangeHints = (hints: readonly CSeedHint[]): CSeedHint[] => {
  const bounds = new Map<string, CRangeBound>();
  const out: CSeedHint[] = [];

  for (const hint of hints) {
    if (hint.kind !== 'range') {
      out.push(hint);
      continue;
    }

    const existing = bounds.get(hint.key);

    if (!existing) {
      const bound: CRangeBound = { ...hint.bound };
      bounds.set(hint.key, bound);
      out.push({ kind: 'range', key: hint.key, bound });
      continue;
    }

    for (const side of ['gt', 'gte', 'lt', 'lte'] as const) {
      if (existing[side] === undefined && hint.bound[side] !== undefined) {
        existing[side] = hint.bound[side];
      }
    }
  }

  return out;
};

const compileNode = (node: NodePattern): CNode => {
  const seedHints = hintsForVariable(node.where, node.variable);

  return {
    variable: node.variable,
    label: node.label,
    pred: compilePredicate(node.properties, node.where),
    seedHints: seedHints.length > 0 ? seedHints : undefined,
  };
};

// A quantified segment may carry a per-hop predicate (inline props / WHERE),
// applied to every edge of the walk; the optional edge variable names each hop's
// edge in turn for that predicate (it is not yet a group/list variable exposed to
// the outer query). `trailEnds` binds and filters each hop.
const compileRel = (rel: RelPattern): CRel => ({
  variable: rel.variable,
  label: rel.label,
  direction: rel.direction,
  pred: compilePredicate(rel.properties, rel.where),
  quantifier: rel.quantifier,
});

const relHasPredicate = (rel: RelPattern): boolean =>
  (rel.properties?.length ?? 0) > 0 || rel.where !== undefined;

const compilePath = (pattern: PathPattern): CPath => {
  const selector = pattern.selector ?? 'walk';

  // The shortest drivers are pure BFS over labels — they do not evaluate a per-hop
  // edge predicate. Reject rather than silently ignore the filter (native rejects
  // the same shape at parse time; both fault E_SYNTAX).
  if (selector === 'anyShortest' || selector === 'allShortest') {
    const seg = pattern.segments[0]?.rel;

    if (seg?.quantifier && relHasPredicate(seg)) {
      throw new LenkeError(
        'a per-hop edge predicate on a variable-length segment is not yet supported together with a path selector (ANY/ALL SHORTEST)',
        { code: ErrorCode.Syntax },
      );
    }
  }

  return {
    start: compileNode(pattern.start),
    segments: pattern.segments.map(({ rel, node }) => ({
      rel: compileRel(rel),
      node: compileNode(node),
    })),
    ...(pattern.pathVar !== undefined ? { pathVar: pattern.pathVar } : {}),
    selector,
    mode: pattern.mode ?? 'trail',
  };
};

/**
 * The ISO element-pattern predicate: every property-map entry must equal the
 * element's stored value, and any inline `WHERE` must hold. Both are evaluated
 * against `binding`, which already includes this element's own variable, so
 * `(n WHERE n.age > 30)` can reference `n`.
 */
const satisfies = (
  element: Bound,
  pred: CPredicate,
  binding: Binding,
  params: Params,
  graph: Graph,
): boolean => {
  const env: EvalEnv = { binding, params, graph };

  for (const { key, value } of pred.props) {
    if (propOf(element, key) !== value(env)) {
      return false;
    }
  }

  return pred.where === undefined || pred.where(env) === true;
};

// --- matching ----------------------------------------------------------------

const matchNode = (
  binding: Binding,
  node: CNode,
  vertex: Vertex,
  params: Params,
  graph: Graph,
): Binding | null => {
  if (!matchesLabel(vertex, node.label)) {
    return null;
  }

  if (!consistent(binding, node.variable, vertex)) {
    return null;
  }

  const bound = withBinding(binding, node.variable, vertex);

  if (!satisfies(vertex, node.pred, bound, params, graph)) {
    return null;
  }

  return bound;
};

/** A scalar the property index can seek on (mirrors PropertyIndex's IndexableValue). */
const isScalar = (v: unknown): v is IndexableValue =>
  v === null ||
  typeof v === 'string' ||
  typeof v === 'boolean' ||
  (typeof v === 'number' && !Number.isNaN(v));

const EMPTY: ReadonlySet<Vertex> = new Set<Vertex>();

/** Resolve a compiled range bound to concrete scalar endpoints, or null. */
const evalBound = (bound: CRangeBound, env: EvalEnv): RangeBound | null => {
  const out: RangeBound = {};
  let any = false;

  for (const key of ['gt', 'gte', 'lt', 'lte'] as const) {
    const fn = bound[key];

    if (!fn) {
      continue;
    }

    const v = fn(env);

    if (!isScalar(v)) {
      return null; // a non-scalar endpoint makes the seek meaningless
    }

    out[key] = v;
    any = true;
  }

  return any ? out : null;
};

/** A seekable predicate: its estimated cardinality and a thunk for the set. */
type SeedCandidate = { count: number; build: () => ReadonlySet<Vertex> };

/**
 * Every index seek a node pattern offers: its ISO equality constraints
 * (`(n {k: v})`) plus the seed hints lifted from WHERE / inline predicates.
 * Each candidate carries a cardinality estimate (computed without touching a
 * set) and a thunk that builds the set only if it's chosen.
 */
const indexCandidates = function* (
  graph: Graph,
  node: CNode,
  env: EvalEnv,
): Iterable<SeedCandidate> {
  const idx = graph.vertexPropertyIndex;
  const eqCandidate = (key: string, v: unknown): SeedCandidate => ({
    count: idx.countEquals(key, v) ?? 0,
    build: () => idx.equals(key, v) ?? EMPTY,
  });

  for (const { key, value } of node.pred.props) {
    if (idx.isIndexed(key)) {
      const v = value(env);

      if (isScalar(v)) {
        yield eqCandidate(key, v);
      }
    }
  }

  for (const hint of node.seedHints ?? []) {
    if (!idx.isIndexed(hint.key)) {
      continue;
    }

    if (hint.kind === 'eq') {
      const v = hint.value(env);

      if (isScalar(v)) {
        yield eqCandidate(hint.key, v);
      }
    } else if (hint.kind === 'within') {
      const list = hint.values(env);

      if (Array.isArray(list) && list.every(isScalar)) {
        let count = 0;

        for (const item of list) {
          count += idx.countEquals(hint.key, item) ?? 0;
        }

        yield {
          count,
          build: () => {
            const out = new Set<Vertex>();

            for (const item of list) {
              for (const vertex of idx.equals(hint.key, item) ?? EMPTY) {
                out.add(vertex);
              }
            }

            return out;
          },
        };
      }
    } else {
      const bound = evalBound(hint.bound, env);

      if (bound) {
        yield {
          count: idx.countRange(hint.key, bound) ?? 0,
          build: () => idx.range(hint.key, bound) ?? EMPTY,
        };
      }
    }
  }
};

/**
 * Seed candidates for a node pattern. An indexed equality / range / `IN` — from
 * an element-pattern map (`(n:Person {name: 'marko'})`) or a seekable WHERE
 * conjunct (`WHERE n.age > 30`) — seeks the index instead of scanning every
 * vertex. The most selective seek (smallest estimated cardinality) is chosen
 * and materialized; `matchNode` and the residual WHERE re-validate the rest, so
 * the seed only has to be a superset. Falls back to the label-narrowed scan
 * when nothing is indexed.
 */
const seedVertices = function* (
  graph: Graph,
  node: CNode,
  binding: Binding,
  params: Params,
): Iterable<Vertex> {
  const env: EvalEnv = { binding, params, graph };
  let best: SeedCandidate | undefined;

  for (const candidate of indexCandidates(graph, node, env)) {
    if (!best || candidate.count < best.count) {
      best = candidate;
    }
  }

  if (best) {
    yield* best.build();

    return;
  }

  yield* candidateVertices(graph, node.label);
};

/** The estimated number of seed vertices for starting a pattern at `node`. */
const estimateSeed = (graph: Graph, node: CNode, binding: Binding, params: Params): number => {
  // An already-bound variable seeds from exactly one vertex.
  if (node.variable && binding.has(node.variable)) {
    return 1;
  }

  const env: EvalEnv = { binding, params, graph };
  let best = Infinity;

  for (const candidate of indexCandidates(graph, node, env)) {
    best = Math.min(best, candidate.count);
  }

  return best === Infinity ? candidateCount(graph, node.label) : best;
};

const FLIP_DIRECTION: Record<RelPattern['direction'], RelPattern['direction']> = {
  out: 'in',
  in: 'out',
  both: 'both',
};

/**
 * Walk a fixed-length path from its other end: reverse the segment order and
 * flip each relationship's direction. The matched bindings are identical (same
 * edges, same nodes); only the seed side — and thus enumeration order — changes.
 */
const reversePath = (path: CPath): CPath => {
  const nodes = [path.start, ...path.segments.map((s) => s.node)];
  const segments: CSegment[] = [];

  for (let i = path.segments.length - 1; i >= 0; i--) {
    const seg = path.segments[i];
    segments.push({
      rel: { ...seg.rel, direction: FLIP_DIRECTION[seg.rel.direction] },
      node: nodes[i],
    });
  }

  // Reversing swaps the endpoints but not what the path binds to.
  return {
    start: nodes[nodes.length - 1],
    segments,
    ...(path.pathVar !== undefined ? { pathVar: path.pathVar } : {}),
    selector: path.selector,
    mode: path.mode,
  };
};

/**
 * Pick which end of a fixed-length path to seed from: the side with the smaller
 * estimated seed, so the join starts from the more selective anchor. Patterns
 * with a variable-length segment keep their written orientation (reversing a
 * quantified walk is not handled here).
 */
const orient = (graph: Graph, pattern: CPath, binding: Binding, params: Params): CPath => {
  if (pattern.segments.length === 0 || pattern.segments.some((s) => s.rel.quantifier)) {
    return pattern;
  }

  const endNode = pattern.segments[pattern.segments.length - 1].node;
  const startEst = estimateSeed(graph, pattern.start, binding, params);
  const endEst = estimateSeed(graph, endNode, binding, params);

  return endEst < startEst ? reversePath(pattern) : pattern;
};

/**
 * `ANY SHORTEST` over a single quantified segment `(start)-[rel q]->(end)`: from
 * the already-matched `seed`, BFS out to one fewest-hop path per reachable
 * endpoint (a vertex is discovered once, keeping its first/shortest predecessor),
 * bind that {@link Path} to the path variable (if named), and yield.
 *
 * Determinism (so native == TS): endpoints are emitted in graph insertion order
 * — the mirror of native's ascending dense-vertex-id order. `q.max` bounds the
 * BFS depth; `q.min ≤ 1` is guaranteed by the parser.
 */
const shortestWalk = function* (
  graph: Graph,
  pattern: CPath,
  seed: Vertex,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  const [{ rel, node: endNode }] = pattern.segments;
  const { min, max } = rel.quantifier!;

  // BFS: shortest hop distance + predecessor (vertex, edge) for each vertex.
  const dist = new Map<string, number>([[seed.id, 0]]);
  const pred = new Map<string, { prev: Vertex; edge: Edge }>();
  // A live array iterator: vertices pushed during the walk are visited in turn,
  // giving a FIFO breadth-first order.
  const queue: Vertex[] = [seed];
  // The shortest cycle back to the seed (its first BFS re-arrival). The seed is
  // marked at distance 0 and never re-discovered, so a `+`/`{1,n}` path that
  // closes on it (`(a)-[]->+(a)`, or any endpoint reached via a cycle) would
  // otherwise be missed.
  let seedCycle: { dist: number; prev: Vertex; edge: Edge } | null = null;

  for (const v of queue) {
    const d = dist.get(v.id)!;

    if (max !== null && d >= max) {
      continue; // don't expand past the hop ceiling
    }

    for (const { edge, node: nbr } of expand(graph, v, rel)) {
      if (nbr.id === seed.id && seedCycle === null) {
        seedCycle = { dist: d + 1, prev: v, edge };
      }

      if (!dist.has(nbr.id)) {
        dist.set(nbr.id, d + 1);
        pred.set(nbr.id, { prev: v, edge });
        queue.push(nbr);
      }
    }
  }

  // When `min ≥ 1` excludes the seed's zero-hop path but a cycle back to it fits
  // the hop ceiling, the seed is an endpoint at the shortest-cycle distance.
  // `min ≤ 1` is guaranteed, so this never double-emits a seed already at dist 0.
  const seedCycleEnd = min >= 1 && seedCycle !== null && (max === null || seedCycle.dist <= max);

  // Endpoints in insertion order (= native's dense-id order).
  for (const end of graph.vertices) {
    const isSeedCycle = end.id === seed.id && seedCycleEnd;
    const d = dist.get(end.id);

    if (!isSeedCycle && (d === undefined || d < min)) {
      continue;
    }

    const matched = matchNode(binding, endNode, end, params, graph);

    if (!matched) {
      continue;
    }

    if (pattern.pathVar === undefined) {
      yield matched;

      continue;
    }

    // Reconstruct the path from the predecessor tree — the shortest path seed…end,
    // or (for the seed-cycle endpoint) the path seed…prev closed by the cycle edge.
    const steps: { edge: Edge; vertex: Vertex }[] = [];
    let cur = isSeedCycle ? seedCycle!.prev : end;

    while (cur.id !== seed.id) {
      const step = pred.get(cur.id)!;
      steps.push({ edge: step.edge, vertex: cur });
      cur = step.prev;
    }

    steps.reverse();

    if (isSeedCycle) {
      steps.push({ edge: seedCycle!.edge, vertex: seed });
    }

    yield withBinding(matched, pattern.pathVar, Path.fromSteps(seed, steps));
  }
};

// `ALL SHORTEST`: every fewest-hop path to each reachable end-matching vertex.
// Like `shortestWalk`, but records ALL shortest predecessors per vertex and
// enumerates the resulting shortest-path DAG (one row per path — ISO per-path
// multiplicity, even without a path variable). Determinism identical to
// `shortestWalk` plus per-endpoint paths in predecessor-recording order, so it
// stays byte-identical with the native `all_shortest_walk`.
const allShortestWalk = function* (
  graph: Graph,
  pattern: CPath,
  seed: Vertex,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  const [{ rel, node: endNode }] = pattern.segments;
  const { min, max } = rel.quantifier!;

  const dist = new Map<string, number>([[seed.id, 0]]);
  const preds = new Map<string, { prev: Vertex; edge: Edge }[]>();
  const queue: Vertex[] = [seed];
  let seedCycleDist: number | null = null;
  const seedCycles: { prev: Vertex; edge: Edge }[] = [];

  for (const v of queue) {
    const d = dist.get(v.id)!;

    if (max !== null && d >= max) {
      continue;
    }

    for (const { edge, node: nbr } of expand(graph, v, rel)) {
      if (nbr.id === seed.id) {
        if (seedCycleDist === null) {
          seedCycleDist = d + 1;
          seedCycles.push({ prev: v, edge });
        } else if (seedCycleDist === d + 1) {
          seedCycles.push({ prev: v, edge });
        }
      }

      const dn = dist.get(nbr.id);

      if (dn === undefined) {
        dist.set(nbr.id, d + 1);
        preds.set(nbr.id, [{ prev: v, edge }]);
        queue.push(nbr);
      } else if (dn === d + 1) {
        preds.get(nbr.id)!.push({ prev: v, edge });
      }
    }
  }

  const seedCycleEnd = min >= 1 && seedCycleDist !== null && (max === null || seedCycleDist <= max);

  // Every shortest path seed…v as a forward `steps` array, via the preds DAG.
  const enumerate = (v: Vertex): { edge: Edge; vertex: Vertex }[][] => {
    if (v.id === seed.id) {
      return [[]];
    }

    const ps = preds.get(v.id);

    if (!ps) {
      return [];
    }

    const out: { edge: Edge; vertex: Vertex }[][] = [];

    for (const { prev, edge } of ps) {
      for (const sub of enumerate(prev)) {
        out.push([...sub, { edge, vertex: v }]);
      }
    }

    return out;
  };

  for (const end of graph.vertices) {
    const isSeedCycle = end.id === seed.id && seedCycleEnd;
    const d = dist.get(end.id);

    if (!isSeedCycle && (d === undefined || d < min)) {
      continue;
    }

    const matched = matchNode(binding, endNode, end, params, graph);

    if (!matched) {
      continue;
    }

    let paths: { edge: Edge; vertex: Vertex }[][];

    if (isSeedCycle) {
      paths = [];

      for (const { prev, edge } of seedCycles) {
        for (const sub of enumerate(prev)) {
          paths.push([...sub, { edge, vertex: seed }]);
        }
      }
    } else {
      paths = enumerate(end);
    }

    for (const steps of paths) {
      yield pattern.pathVar === undefined
        ? matched
        : withBinding(matched, pattern.pathVar, Path.fromSteps(seed, steps));
    }
  }
};

/** A start-seeded path driver: yields each binding extending `binding` from `seed`. */
type SeedDriver = (
  graph: Graph,
  pattern: CPath,
  seed: Vertex,
  binding: Binding,
  params: Params,
) => Iterable<Binding>;

/** Can a selector pattern reduce to a BFS driver? True when the single var-length
 * segment is a `*`/`+` (min ≤ 1) with no per-hop predicate — the shape the BFS
 * drivers are correct for. Mirrors native `bfs_reducible`. */
const bfsReducible = (pattern: CPath): boolean => {
  const seg = pattern.segments.length === 1 ? pattern.segments[0] : undefined;
  const q = seg?.rel.quantifier;

  return (
    q !== undefined &&
    q.min <= 1 &&
    seg!.rel.pred.props.length === 0 &&
    seg!.rel.pred.where === undefined
  );
};

/** Pick the start-seeded driver for a selector, or null for the walk / path-var
 * matcher (handled below). `ANY` and `SHORTEST 1 [GROUP]` reduce to the O(V+E)
 * BFS drivers when `bfsReducible` (a shortest path is a valid arbitrary /
 * 1-shortest path); otherwise they enumerate trails. Both engines route
 * identically, so the result stays byte-identical. */
const pickSeedDriver = (pattern: CPath, selector: PathSelector): SeedDriver | null => {
  if (selector === 'anyShortest') {
    return shortestWalk;
  }

  if (selector === 'allShortest') {
    return allShortestWalk;
  }

  if (selector === 'any') {
    return bfsReducible(pattern) ? shortestWalk : anyWalk;
  }

  if (typeof selector === 'object' && selector.kind === 'shortestK') {
    if (selector.k === 1 && bfsReducible(pattern)) {
      return selector.group ? allShortestWalk : shortestWalk;
    }

    const sel = selector;

    return (graph, pat, seed, binding, params) =>
      shortestKWalk(graph, pat, seed, binding, params, sel);
  }

  return null;
};

/** Yield every binding that extends `binding` by matching `pattern`. */
const matchPattern = function* (
  graph: Graph,
  pattern: CPath,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  const selector = pattern.selector ?? 'walk';

  // Selectors that seed from the START and yield via a dedicated driver (a BFS
  // one or the trail enumerator). `ANY`/`SHORTEST 1 [GROUP]` reduce to the BFS
  // drivers when the segment is shortest-shaped — see `pickSeedDriver`.
  const seedDriver = pickSeedDriver(pattern, selector);

  if (seedDriver) {
    const seeds: Iterable<Vertex> =
      pattern.start.variable && binding.has(pattern.start.variable)
        ? [binding.get(pattern.start.variable) as Vertex]
        : seedVertices(graph, pattern.start, binding, params);

    for (const seed of seeds) {
      const seeded = matchNode(binding, pattern.start, seed, params, graph);

      if (seeded) {
        yield* seedDriver(graph, pattern, seed, seeded, params);
      }
    }

    return;
  }

  // Seed from whichever end is more selective, then walk from there.
  const path = orient(graph, pattern, binding, params);

  // Reuse an already-bound vertex if the start variable is known, otherwise
  // seed from an indexed constraint or a label-narrowed scan.
  const seeds: Iterable<Vertex> =
    path.start.variable && binding.has(path.start.variable)
      ? [binding.get(path.start.variable) as Vertex]
      : seedVertices(graph, path.start, binding, params);

  for (const seed of seeds) {
    const seeded = matchNode(binding, path.start, seed, params, graph);

    if (seeded) {
      // A bound path variable over a single quantified segment binds each walk as
      // a Path; otherwise the plain endpoint walk.
      yield* path.pathVar !== undefined
        ? allWalk(graph, path, seed, seeded, params)
        : walkSegments(graph, path, 0, seed, seeded, params);
    }
  }
};

// Bare path binding over a single quantified segment (`p = (a)-[:R]->{m,n}(b)`):
// enumerate every walk under the pattern's mode and bind each as a Path value.
// Mirrors native `all_walk`.
const allWalk = function* (
  graph: Graph,
  pattern: CPath,
  seed: Vertex,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  const [{ rel, node: endNode }] = pattern.segments;

  for (const { end, verts, edges } of trailEnds(graph, seed, rel, rel.quantifier!, {
    mode: pattern.mode ?? 'trail',
    binding,
    params,
    wantPath: true,
  })) {
    const matched = matchNode(binding, endNode, end, params, graph);

    if (!matched) {
      continue;
    }

    if (pattern.pathVar === undefined) {
      yield matched;

      continue;
    }

    const steps = edges.map((edge, i) => ({ edge, vertex: verts[i + 1] }));

    yield withBinding(matched, pattern.pathVar, Path.fromSteps(verts[0], steps));
  }
};

/** Bind the endpoint node and (if named) the walk as a Path, yielding the row.
 * Shared by the trail-enumerating selectors (`ANY`, `SHORTEST k`). */
const bindEndAndPath = (
  graph: Graph,
  pattern: CPath,
  walk: TrailEnd,
  binding: Binding,
  params: Params,
): Binding | null => {
  const [{ node: endNode }] = pattern.segments;
  const matched = matchNode(binding, endNode, walk.end, params, graph);

  if (!matched) {
    return null;
  }

  if (pattern.pathVar === undefined) {
    return matched;
  }

  const steps = walk.edges.map((edge, i) => ({ edge, vertex: walk.verts[i + 1] }));

  return withBinding(matched, pattern.pathVar, Path.fromSteps(walk.verts[0], steps));
};

// Bare `ANY`: one arbitrary path per endpoint — the first walk that reaches each
// distinct endpoint in trail-discovery order. Byte-identical because that order
// is. Mirrors native `any_walk`.
const anyWalk = function* (
  graph: Graph,
  pattern: CPath,
  seed: Vertex,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  const [{ rel }] = pattern.segments;
  const seen = new Set<Vertex>();

  for (const walk of trailEnds(graph, seed, rel, rel.quantifier!, {
    mode: pattern.mode ?? 'trail',
    binding,
    params,
    wantPath: pattern.pathVar !== undefined,
  })) {
    // First witness per endpoint only (the endpoint match is per-vertex).
    if (seen.has(walk.end)) {
      continue;
    }

    seen.add(walk.end);

    const row = bindEndAndPath(graph, pattern, walk, binding, params);

    if (row) {
      yield row;
    }
  }
};

// `SHORTEST k [GROUP]`: enumerate every trail, group by endpoint, order each
// endpoint's paths by (length, discovery), then keep the first `k` (plain) or
// every path in the `k` smallest distinct-length groups (`group`). Mirrors native
// `shortest_k_walk`.
const shortestKWalk = function* (
  graph: Graph,
  pattern: CPath,
  seed: Vertex,
  binding: Binding,
  params: Params,
  sel: { k: number; group: boolean },
): Iterable<Binding> {
  const { k, group } = sel;
  const [{ rel }] = pattern.segments;

  // endpoint -> its trails as { walk, len }, in discovery order.
  const perEnd = new Map<Vertex, { walk: TrailEnd; len: number }[]>();

  for (const walk of trailEnds(graph, seed, rel, rel.quantifier!, {
    mode: pattern.mode ?? 'trail',
    binding,
    params,
    wantPath: true,
  })) {
    const bucket = perEnd.get(walk.end);
    const entry = { walk, len: walk.edges.length };

    if (bucket) {
      bucket.push(entry);
    } else {
      perEnd.set(walk.end, [entry]);
    }
  }

  // Endpoints in graph (insertion/index) order — mirrors native `ends.sort_unstable()`
  // (native ids are the insertion index), the same ordering `allShortestWalk` uses.
  for (const end of graph.vertices) {
    const paths = perEnd.get(end);

    if (!paths) {
      continue;
    }

    // Stable sort by length → shortest first, discovery order within a length.
    paths.sort((a, b) => a.len - b.len);

    let selected: typeof paths;

    if (group) {
      // The k smallest distinct lengths (paths are length-sorted, so equal lengths
      // are contiguous); keep every path at or below the kth.
      const distinct: number[] = [];

      for (const p of paths) {
        if (distinct[distinct.length - 1] !== p.len) {
          distinct.push(p.len);
        }
      }

      const cutoff = distinct.slice(0, k).at(-1);
      selected = cutoff === undefined ? [] : paths.filter((p) => p.len <= cutoff);
    } else {
      selected = paths.slice(0, k);
    }

    for (const { walk } of selected) {
      const row = bindEndAndPath(graph, pattern, walk, binding, params);

      if (row) {
        yield row;
      }
    }
  }
};

/** Recursively extend a binding across the remaining segments of a pattern. */
const walkSegments = function* (
  graph: Graph,
  pattern: CPath,
  index: number,
  from: Vertex,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  if (index >= pattern.segments.length) {
    yield binding;

    return;
  }

  const { rel, node } = pattern.segments[index];

  // Variable-length: enumerate the endpoint of every trail within [min, max]
  // hops (one per trail → ISO per-path multiplicity), then continue from each.
  // `trailEnds` binds and filters each hop's edge for a per-hop predicate.
  if (rel.quantifier) {
    for (const { end } of trailEnds(graph, from, rel, rel.quantifier, {
      mode: pattern.mode ?? 'trail',
      binding,
      params,
    })) {
      const matched = matchNode(binding, node, end, params, graph);

      if (matched) {
        yield* walkSegments(graph, pattern, index + 1, end, matched, params);
      }
    }

    return;
  }

  for (const { edge, node: nextVertex } of expand(graph, from, rel)) {
    if (!consistent(binding, rel.variable, edge)) {
      continue;
    }

    const withEdge = withBinding(binding, rel.variable, edge);

    if (!satisfies(edge, rel.pred, withEdge, params, graph)) {
      continue;
    }

    const matched = matchNode(withEdge, node, nextVertex, params, graph);

    if (matched) {
      yield* walkSegments(graph, pattern, index + 1, nextVertex, matched, params);
    }
  }
};

/** Per-expansion cap on trail-traversal steps; a guard against exponential blowup. */
const TRAIL_BUDGET = 1_000_000;

/**
 * Endpoints of every *trail* — a path that traverses each relationship at most
 * once (ISO/IEC 39075 default for a quantified path) — from `from` within
 * [min, max] hops of `rel`. Yielded one per trail, so an endpoint reachable by
 * `k` distinct trails is yielded `k` times (ISO per-path multiplicity); a `min`
 * of 0 includes the zero-length trail (the start node itself).
 *
 * Edge-uniqueness bounds a trail's length to the edge count, so this always
 * terminates even on cycles — but the *number* of trails can be exponential, so
 * a per-expansion step budget throws rather than letting a pathological `*`
 * exhaust memory/time.
 */
type TrailEnd = { end: Vertex; verts: readonly Vertex[]; edges: readonly Edge[] };

type TrailOpts = {
  mode: PathMode;
  // The binding at the segment's start (outer bound vars stay visible to a per-hop
  // predicate) and the query params. Each hop's edge is bound onto `binding` for
  // the predicate; the binding itself is not mutated.
  binding: Binding;
  params: Params;
  // When true, reconstruct each trail's vertices/edges from the frame stack (a
  // path variable needs the whole walk); otherwise `verts`/`edges` are empty.
  wantPath?: boolean;
};

const trailEnds = function* (
  graph: Graph,
  from: Vertex,
  rel: CRel,
  q: NonNullable<CRel['quantifier']>,
  opts: TrailOpts,
): Iterable<TrailEnd> {
  const { mode, binding, params, wantPath = false } = opts;

  // A per-hop predicate (inline props / WHERE) filters every edge of the walk;
  // the optional edge variable names each hop's edge for it.
  const hasPred = rel.pred.props.length > 0 || rel.pred.where !== undefined;

  if (q.min === 0) {
    yield { end: from, verts: wantPath ? [from] : [], edges: [] };
  }

  // The repeated-element marks on the CURRENT path. TRAIL (default) marks EDGES;
  // SIMPLE/ACYCLIC mark NODES (seed pre-marked); WALK marks nothing (bounded only
  // by the quantifier / trail budget). Mirrors native `reachable_each`.
  const vertexMode = mode === 'simple' || mode === 'acyclic';
  const marks = new Set<Edge | Vertex>();

  if (vertexMode) {
    marks.add(from);
  }

  let steps = 0;

  // Explicit DFS stack — a trail can be as long as the edge count, so recursion
  // would overflow on a long chain. Each frame walks one vertex's outgoing
  // steps; `entry` is the mark taken to reach it, unmarked when the frame pops.
  // A generator suspends on consumer-stop, so no explicit stop cleanup is needed.
  const stack: {
    iter: Iterator<{ edge: Edge; node: Vertex }>;
    entry: Edge | Vertex | null;
    depth: number;
    // For path reconstruction (wantPath): the vertex this frame explores from,
    // and the edge taken to reach it (`null` at the seed).
    vertex: Vertex;
    entryEdge: Edge | null;
  }[] = [
    {
      iter: expand(graph, from, rel)[Symbol.iterator](),
      entry: null,
      depth: 0,
      vertex: from,
      entryEdge: null,
    },
  ];

  while (stack.length > 0) {
    const top = stack[stack.length - 1];

    // Past the max hop → this trail can't extend; backtrack.
    if (q.max !== null && top.depth >= q.max) {
      if (top.entry) {
        marks.delete(top.entry);
      }

      stack.pop();
      continue;
    }

    // Each visit advances this frame's iterator by one step (the recursion-free
    // equivalent of the for-loop); when exhausted, backtrack.
    const res = top.iter.next();

    if (res.done) {
      if (top.entry) {
        marks.delete(top.entry);
      }

      stack.pop();
      continue;
    }

    const { edge, node } = res.value;

    // Per-hop predicate: skip edges that fail it before any mark/step accounting
    // (a failing edge never enters the trail — mirrors native `expand_filtered`).
    if (
      hasPred &&
      !satisfies(edge, rel.pred, withBinding(binding, rel.variable, edge), params, graph)
    ) {
      continue;
    }

    // Whether this step is allowed, what it marks, and whether it's a
    // non-extending SIMPLE close back on the seed.
    let markTarget: Edge | Vertex | null = null;
    let isClose = false;

    if (mode === 'trail') {
      if (marks.has(edge)) {
        continue; // each relationship at most once
      }

      markTarget = edge;
    } else if (mode === 'acyclic') {
      if (marks.has(node)) {
        continue; // no repeated node (not even the seed)
      }

      markTarget = node;
    } else if (mode === 'simple') {
      if (node === from) {
        isClose = true; // close the cycle on the seed: emit, don't extend
      } else if (marks.has(node)) {
        continue; // no repeated node except that close
      } else {
        markTarget = node;
      }
    }
    // mode === 'walk': no marks, always allowed.

    steps += 1;

    if (steps > TRAIL_BUDGET) {
      throw new LenkeError(
        'Variable-length pattern exceeded the trail budget; add a tighter bound',
        { code: ErrorCode.ResourceExhausted },
      );
    }

    if (markTarget !== null) {
      marks.add(markTarget);
    }

    const d = top.depth + 1;

    if (d >= q.min) {
      // Rebuild the walk `seed … node` from the live stack (each frame's vertex,
      // then `node`; edges are each frame's entry edge, then this `edge`). A
      // SIMPLE close reconstructs the closing cycle naturally.
      if (wantPath) {
        const verts = stack.map((f) => f.vertex);
        verts.push(node);

        const edges = stack.map((f) => f.entryEdge).filter((e): e is Edge => e !== null);
        edges.push(edge);

        yield { end: node, verts, edges };
      } else {
        yield { end: node, verts: [], edges: [] };
      }
    }

    if (!isClose) {
      stack.push({
        iter: expand(graph, node, rel)[Symbol.iterator](),
        entry: markTarget,
        depth: d,
        vertex: node,
        entryEdge: edge,
      });
    }
  }
};

// --- clause compilation ------------------------------------------------------

/** Every variable a pattern introduces (for OPTIONAL MATCH null-binding). */
const patternVars = (patterns: readonly PathPattern[]): string[] => {
  const vars: string[] = [];

  for (const p of patterns) {
    if (p.pathVar) {
      vars.push(p.pathVar);
    }

    if (p.start.variable) {
      vars.push(p.start.variable);
    }

    for (const { rel, node } of p.segments) {
      if (rel.variable) {
        vars.push(rel.variable);
      }

      if (node.variable) {
        vars.push(node.variable);
      }
    }
  }

  return vars;
};

/** A compiled SET assignment: a label add, or a property set with a value closure. */
type CSetItem =
  | { variable: string; label: string }
  | { variable: string; key: string; value: CompiledExpr };

/** A compiled INSERT node/rel: labels are fixed, property values are closures. */
type CInsertNode = { variable?: string; labels: readonly string[]; props: readonly CProp[] };
type CInsertRel = {
  variable?: string;
  labels: readonly string[];
  direction: RelPattern['direction'];
  props: readonly CProp[];
};
type CInsertPath = {
  start: CInsertNode;
  segments: readonly { rel: CInsertRel; node: CInsertNode }[];
};

type CMatch = {
  kind: 'match';
  optional: boolean;
  patterns: readonly CPath[];
  where?: CompiledExpr;
  nullVars: readonly string[];
};
type CWith = { kind: 'with'; projection: CProjection; where?: CompiledExpr };
type CFilter = { kind: 'filter'; where: CompiledExpr };
type CLet = { kind: 'let'; items: readonly { var: string; expr: CompiledExpr }[] };
type CFor = {
  kind: 'for';
  list: CompiledExpr;
  alias: string;
  ordinality?: { kind: 'ordinality' | 'offset'; var: string };
};
type CReturn = { kind: 'return'; projection: CProjection };
type CInsert = { kind: 'insert'; patterns: readonly CInsertPath[] };
type CMergeUpdate =
  | { kind: 'set'; items: readonly CSetItem[]; where?: CompiledExpr }
  | { kind: 'nothing' };
type CMerge = {
  kind: 'merge';
  pattern: CInsertPath;
  onCreate?: readonly CSetItem[];
  onUpdate?: CMergeUpdate;
};
type CSet = { kind: 'set'; items: readonly CSetItem[] };
type CRemove = { kind: 'remove'; items: readonly RemoveItem[] };
type CDelete = { kind: 'delete'; detach: boolean; targets: readonly CompiledExpr[] };
type CFinish = { kind: 'finish' };
type CCallNamed = {
  kind: 'callNamed';
  optional: boolean;
  procName: string;
  /** Resolved algorithm dispatch name; `null` = unknown procedure (faults). */
  algo: AlgorithmName | null;
  config: readonly { key: string; value: CompiledExpr }[];
  /** Procedure output column → the variable it yields into. */
  binds: readonly { column: string; var: string }[];
};
type CCallInline = {
  kind: 'callInline';
  optional: boolean;
  scope: readonly string[];
  body: CLinear;
  /**
   * Additional set-op parts (`… UNION/EXCEPT/INTERSECT …`) after the first. Empty
   * for a plain single-part body; each part shares the same imported scope and
   * yields the same columns, folded with `combineRows`.
   */
  bodyMore: readonly { op: SetOp; part: CLinear }[];
  /** Output columns of the nested RETURN (for OPTIONAL null-fill). */
  returnColumns: readonly string[];
};
type CClause =
  | CMatch
  | CWith
  | CFilter
  | CLet
  | CFor
  | CReturn
  | CInsert
  | CMerge
  | CSet
  | CRemove
  | CCallNamed
  | CCallInline
  | CDelete
  | CFinish;

// Labels to CREATE for an INSERT element. A non-conjunction label expression
// (`A|B`, `!A`, `%`) is ambiguous — reject it rather than silently create an
// unlabelled node (an unlabelled node — no expression — stays legitimate).
// Mirrors the Rust `creatable_labels`. (A typeless edge's empty `[]` is caught
// downstream by `Graph.addEdge`, which requires ≥1 label.)
const creatableLabels = (expr: LabelExpr | undefined): string[] => {
  if (!expr) {
    return [];
  }

  if (expr.kind === 'label') {
    return [expr.name];
  }

  if (expr.kind === 'and') {
    return [...creatableLabels(expr.left), ...creatableLabels(expr.right)];
  }

  throw new LenkeError(
    "INSERT: a node's label expression must be a plain conjunction (`A` or `A&B`) and an edge must carry exactly one type — a disjunction/negation/wildcard is not creatable",
    { code: ErrorCode.InvalidGraphOp },
  );
};

const compileInsertNode = (node: NodePattern): CInsertNode => ({
  variable: node.variable,
  labels: creatableLabels(node.label),
  props: compileProps(node.properties),
});

const compileInsertPath = (pattern: PathPattern): CInsertPath => ({
  start: compileInsertNode(pattern.start),
  segments: pattern.segments.map(({ rel, node }) => ({
    rel: {
      variable: rel.variable,
      labels: creatableLabels(rel.label),
      direction: rel.direction,
      props: compileProps(rel.properties),
    },
    node: compileInsertNode(node),
  })),
});

const compileSetItem = (item: SetItem): CSetItem =>
  'label' in item
    ? { variable: item.variable, label: item.label }
    : { variable: item.variable, key: item.key, value: compileExpr(item.value) };

const compileClause = (clause: Clause): CClause => {
  switch (clause.kind) {
    case 'match': {
      const patterns = clause.patterns.map(compilePath);

      // Lift seekable conjuncts of the clause WHERE onto every pattern node by
      // variable — not just the start — so either end of a pattern can be the
      // seed side. `MATCH (a:Person) WHERE a.name = 'marko'` then seeds like the
      // inline `(a:Person {name: 'marko'})` form, and a constraint on the far
      // end lets `orient` start the walk from there.
      if (clause.where) {
        const hints: HintMap = new Map();
        collectHints(clause.where, hints);
        const attach = (node: CNode): CNode => {
          const extra = node.variable ? hints.get(node.variable) : undefined;

          return extra
            ? { ...node, seedHints: coalesceRangeHints([...(node.seedHints ?? []), ...extra]) }
            : node;
        };

        for (let i = 0; i < patterns.length; i++) {
          patterns[i] = {
            ...patterns[i],
            start: attach(patterns[i].start),
            segments: patterns[i].segments.map((s) => ({ rel: s.rel, node: attach(s.node) })),
          };
        }
      }

      return {
        kind: 'match',
        optional: clause.optional,
        patterns,
        where: clause.where ? compileExpr(clause.where) : undefined,
        nullVars: clause.optional ? patternVars(clause.patterns) : [],
      };
    }
    case 'with':
      return {
        kind: 'with',
        projection: compileProjection(clause.projection),
        where: clause.where ? compileExpr(clause.where) : undefined,
      };
    case 'filter':
      return { kind: 'filter', where: compileExpr(clause.where) };
    case 'let':
      return {
        kind: 'let',
        items: clause.items.map((it) => ({ var: it.var, expr: compileExpr(it.expr) })),
      };
    case 'for':
      return {
        kind: 'for',
        list: compileExpr(clause.list),
        alias: clause.alias,
        ordinality: clause.ordinality,
      };
    case 'return':
      return { kind: 'return', projection: compileProjection(clause.projection) };
    case 'callNamed': {
      const spec = procedureSpec(clause.name);
      const columns = spec ? ['node', spec.resultColumn] : [];
      const binds = clause.yields
        ? clause.yields.map((y) => ({ column: y.name, var: y.alias ?? y.name }))
        : columns.map((c) => ({ column: c, var: c }));

      return {
        kind: 'callNamed',
        optional: clause.optional,
        procName: clause.name,
        algo: spec?.algo ?? null,
        config: clause.config.map((p) => ({ key: p.key, value: compileExpr(p.value) })),
        binds,
      };
    }
    case 'callInline': {
      // All set-op parts share the same output columns, so the first is
      // authoritative for the OPTIONAL null-fill column names.
      const ret = clause.body.parts[0].clauses.find((c) => c.kind === 'return');
      const returnColumns =
        ret && !ret.projection.star
          ? ret.projection.items.map((i) => i.alias ?? columnName(i.expr))
          : [];

      return {
        kind: 'callInline',
        optional: clause.optional,
        scope: clause.scope,
        body: compileLinear(clause.body.parts[0]),
        bodyMore: clause.body.ops.map((op, i) => ({
          op,
          part: compileLinear(clause.body.parts[i + 1]),
        })),
        returnColumns,
      };
    }
    case 'insert':
      return { kind: 'insert', patterns: clause.patterns.map(compileInsertPath) };
    case 'merge': {
      const { onUpdate } = clause;
      const compileUpdate = (): CMergeUpdate | undefined => {
        if (onUpdate === undefined || onUpdate.kind === 'nothing') {
          return onUpdate;
        }

        return {
          kind: 'set',
          items: onUpdate.items.map(compileSetItem),
          where: onUpdate.where ? compileExpr(onUpdate.where) : undefined,
        };
      };

      return {
        kind: 'merge',
        pattern: compileInsertPath(clause.pattern),
        onCreate: clause.onCreate?.map(compileSetItem),
        onUpdate: compileUpdate(),
      };
    }
    case 'set':
      return { kind: 'set', items: clause.items.map(compileSetItem) };
    case 'remove':
      return { kind: 'remove', items: clause.items };
    case 'delete':
      return { kind: 'delete', detach: clause.detach, targets: clause.targets.map(compileExpr) };
    case 'finish':
      return { kind: 'finish' };
  }
};

// --- count(*) shortcuts (edge-anchored + two-hop degree product) -------------
// Detected on the raw AST at compile time; they compute the count directly from
// the type-bucket sizes / a degree product instead of enumerating every match.
// Mirror the native engine's try_count_edges / try_count_two_hop, and are
// provably identical for the homomorphic count these shapes produce.

/** Edge types a rel label admits: `null` = bail (and/not/wildcard); `undefined`
 * = "any type" (no `:T`); `string[]` = exactly those types. */
const relTypeNames = (label: LabelExpr | undefined): string[] | null | undefined => {
  if (!label) {
    return undefined;
  }

  if (label.kind === 'label') {
    return [label.name];
  }

  if (label.kind === 'or') {
    const l = relTypeNames(label.left);
    const r = relTypeNames(label.right);

    return l && r ? [...l, ...r] : null;
  }

  return null;
};

const plainNode = (n: NodePattern): boolean =>
  (n.properties?.length ?? 0) === 0 && n.where === undefined;
const plainRel = (r: RelPattern): boolean =>
  (r.properties?.length ?? 0) === 0 && r.where === undefined && r.quantifier === undefined;

/** The per-type `Set<Edge>` buckets for `types` (undefined = every type). */
const bucketsFor = (
  byType: Map<string, Set<Edge>> | undefined,
  types: string[] | undefined,
): (Set<Edge> | undefined)[] => {
  if (!byType) {
    return [];
  }

  return types ? types.map((t) => byType.get(t)) : [...byType.values()];
};

/** Count edges across `buckets` that pass `keep`. */
const countEdges = (buckets: (Set<Edge> | undefined)[], keep: (e: Edge) => boolean): number => {
  let n = 0;

  for (const set of buckets) {
    if (!set) {
      continue;
    }

    for (const e of set) {
      if (keep(e)) {
        n += 1;
      }
    }
  }

  return n;
};

type CountFn = (graph: Graph, params: Params) => Row;

/** 1-hop `(a)-[:T]->(b)` count: bucket sizes (unlabeled) or a filtered bucket
 * scan. `null` if the segment can't be bucket-counted (both/And/Not/wildcard). */
const buildOneHopCount = (
  seg: Segment,
  start: NodePattern,
  rowOf: (n: number) => Row,
): CountFn | null => {
  const { rel, node } = seg;

  if (!plainRel(rel) || !plainNode(node) || rel.direction === 'both') {
    return null;
  }

  const types = relTypeNames(rel.label);

  if (types === null) {
    return null;
  }

  const aLabel = start.label;
  const bLabel = node.label;
  const out = rel.direction === 'out';

  return (graph) => {
    if (aLabel === undefined && bLabel === undefined && types) {
      // Unlabeled endpoints → the bucket sizes. O(1) per type.
      return rowOf(types.reduce((n, t) => n + (graph.edgesByLabel.get(t)?.size ?? 0), 0));
    }

    return rowOf(
      countEdges(
        bucketsFor(graph.edgesByLabel, types),
        (edge) =>
          matchesLabel(out ? edge.from : edge.to, aLabel) &&
          matchesLabel(out ? edge.to : edge.from, bLabel),
      ),
    );
  };
};

/** 2-hop `(a)-[:T1]->(b)-[:T2]->(c)` count via the degree product
 * `Σ_b (edges reaching a valid a) × (edges reaching a valid c)`. `null` unless
 * both rels are anonymous + directed and the node variables are distinct. */
const buildTwoHopCount = (
  s1: Segment,
  s2: Segment,
  start: NodePattern,
  rowOf: (n: number) => Row,
): CountFn | null => {
  if (
    !plainRel(s1.rel) ||
    !plainRel(s2.rel) ||
    s1.rel.variable !== undefined ||
    s2.rel.variable !== undefined ||
    s1.rel.direction === 'both' ||
    s2.rel.direction === 'both' ||
    !plainNode(s1.node) ||
    !plainNode(s2.node)
  ) {
    return null;
  }

  const vars = [start.variable, s1.node.variable, s2.node.variable].filter(
    (v): v is string => v !== undefined,
  );

  if (new Set(vars).size !== vars.length) {
    return null; // a shared node variable is a self-join the product can't express
  }

  const t1 = relTypeNames(s1.rel.label);
  const t2 = relTypeNames(s2.rel.label);

  if (t1 === null || t2 === null) {
    return null;
  }

  const aLabel = start.label;
  const midLabel = s1.node.label;
  const cLabel = s2.node.label;
  // seg1 reaches `a` from b's reverse side; seg2 reaches `c` from b's forward side.
  const toAOut = s1.rel.direction === 'in';
  const fromCOut = s2.rel.direction === 'out';
  const side = (
    graph: Graph,
    bId: string,
    out: boolean,
    types: string[] | undefined,
    far: LabelExpr | undefined,
  ): number => {
    const byType = (out ? graph.edgesFromByLabel : graph.edgesToByLabel).get(bId);

    return countEdges(bucketsFor(byType, types), (edge) =>
      matchesLabel(out ? edge.to : edge.from, far),
    );
  };

  return (graph) => {
    const mids =
      midLabel?.kind === 'label'
        ? (graph.verticesByLabel.get(midLabel.name) ?? new Set<Vertex>())
        : graph.verticesById.values();
    let count = 0;

    for (const b of mids) {
      if (!matchesLabel(b, midLabel)) {
        continue;
      }

      const ways = side(graph, b.id, toAOut, t1, aLabel);

      if (ways === 0) {
        continue;
      }

      count += ways * side(graph, b.id, fromCOut, t2, cLabel);
    }

    return rowOf(count);
  };
};

/**
 * If a linear query is exactly `MATCH <1- or 2-segment path> RETURN count(*)`,
 * return a closure computing the count directly (O(1)/O(E)) instead of
 * enumerating every match; `null` if the shape doesn't qualify. The conditions
 * match the native engine: 1-hop directed, no props/WHERE; 2-hop additionally
 * needs anonymous rels and pairwise-distinct node variables (so the homomorphic
 * degree product is exact).
 */
const detectCountShortcut = (clauses: readonly Clause[]): CountFn | null => {
  if (clauses.length !== 2) {
    return null;
  }

  const [m, ret] = clauses;

  if (m.kind !== 'match' || m.optional || m.where !== undefined || m.patterns.length !== 1) {
    return null;
  }

  if (ret.kind !== 'return') {
    return null;
  }

  const proj = ret.projection;

  if (
    proj.star ||
    proj.distinct ||
    (proj.orderBy?.length ?? 0) > 0 ||
    proj.skip !== undefined ||
    proj.limit !== undefined ||
    proj.items.length !== 1
  ) {
    return null;
  }

  const [item] = proj.items;
  const e = item.expr;

  if (e.kind !== 'func' || e.name !== 'count' || !e.star || e.distinct) {
    return null;
  }

  const column = item.alias ?? columnName(e);
  const rowOf = (count: number): Row => ({ [column]: count });
  const [{ start, segments }] = m.patterns;

  if (!plainNode(start)) {
    return null;
  }

  if (segments.length === 1) {
    const [seg] = segments;

    return buildOneHopCount(seg, start, rowOf);
  }

  if (segments.length === 2) {
    const [s1, s2] = segments;

    return buildTwoHopCount(s1, s2, start, rowOf);
  }

  return null;
};

type ReachFn = (graph: Graph, params: Params) => Row[];

/** Whether `e` reads only variable `v` (a bare `v`, `v.key`, or a constant). */
const refsOnlyVar = (e: Expr, v: string): boolean => {
  switch (e.kind) {
    case 'var':
      return e.name === v;
    case 'prop':
      return e.variable === v;
    case 'lit':
    case 'param':
      return true;
    default:
      return false;
  }
};

/**
 * Reachability shortcut for **unbounded var-length with DISTINCT**:
 * `MATCH (a{..})-[:T]->+(b) RETURN DISTINCT <b…>` (and `->*`, `count(DISTINCT b)`).
 * Trail enumeration is exponential on a connected graph and hits `TRAIL_BUDGET`
 * (a fault), but a DISTINCT result only wants the reachable *set* — multiplicity
 * collapses — which a plain O(V+E) BFS answers. `->+` = reachable via ≥1 hop; `->*`
 * also includes the seed(s). Mirrors the native engine's `try_reachable_distinct`
 * so both engines behave identically. Seeds via the compiled start node.
 */
type ReachSpec = {
  cstart: CNode;
  items: readonly CReturnItem[];
  bVar: string;
  bLabel: LabelExpr | undefined;
  out: boolean;
  types: string[] | undefined;
  minZero: boolean;
  isCount: boolean;
  /** For `count(DISTINCT <expr>)` with a non-bare arg (e.g. `b.k`): the compiled
   *  arg to evaluate + dedup per reached vertex. Undefined = bare `count(DISTINCT
   *  b)`, whose distinct count is just the reached-set size. */
  countArg?: CompiledExpr;
  skip: CountValue;
  limit?: CountValue;
};

/** BFS the reachable set, then project the endpoint + DISTINCT (or count it). */
const runReach = (spec: ReachSpec, graph: Graph, params: Params): Row[] => {
  const { cstart, items, bVar, bLabel, out, types, minZero, isCount } = spec;
  // A `$param` bound resolves here (validated up-front); a literal passes through.
  const skipN = resolveCount(spec.skip, params) ?? 0;
  const limit = resolveCount(spec.limit, params);
  // Seeds matching the start's label + inline props/WHERE. `seedVertices` only
  // narrows by label/index, so a no-index inline predicate (`{k:0}`) still needs a
  // per-seed check — otherwise we'd seed from the whole label and overcount the
  // reachable set. Mirrors the native `reach_seed_vertices`.
  const seeds = [...seedVertices(graph, cstart, new Map(), params)].filter(
    (v) => matchNode(new Map(), cstart, v, params, graph) !== null,
  );
  const nbrs = (v: Vertex): Vertex[] => {
    const byType = (out ? graph.edgesFromByLabel : graph.edgesToByLabel).get(v.id);
    const acc: Vertex[] = [];

    for (const set of bucketsFor(byType, types)) {
      if (set) {
        for (const e of set) {
          acc.push(out ? e.to : e.from);
        }
      }
    }

    return acc;
  };

  // Forward reachability (≥1 hop) as a DFS closure — each vertex expands once.
  const seen = new Set<string>();
  const reached: Vertex[] = [];
  const stack: Vertex[] = [];
  const push = (w: Vertex): void => {
    if (!seen.has(w.id)) {
      seen.add(w.id);
      reached.push(w);
      stack.push(w);
    }
  };

  for (const s of seeds) {
    for (const w of nbrs(s)) {
      push(w);
    }
  }

  while (stack.length > 0) {
    for (const w of nbrs(stack.pop()!)) {
      push(w);
    }
  }

  // `->*` also admits the zero-length path — the seeds themselves.
  if (minZero) {
    for (const s of seeds) {
      if (!seen.has(s.id)) {
        seen.add(s.id);
        reached.push(s);
      }
    }
  }

  const kept = reached.filter((v) => matchesLabel(v, bLabel));

  if (isCount) {
    // Bare `count(DISTINCT b)`: distinct endpoints = the reached set.
    if (spec.countArg === undefined) {
      return [{ [items[0].name]: kept.length }];
    }

    // `count(DISTINCT <expr>)` (e.g. `b.k`): evaluate per reached vertex, skip
    // nulls, dedup values — mirrors the native `try_reachable_distinct` count mode.
    const seenVals = new Set<string>();
    let n = 0;

    for (const v of kept) {
      const cell = spec.countArg({ binding: new Map([[bVar, v]]), params, graph });

      if (isNullish(cell)) {
        continue;
      }

      const k = valueKey(cell);

      if (!seenVals.has(k)) {
        seenVals.add(k);
        n += 1;
      }
    }

    return [{ [items[0].name]: n }];
  }

  // DISTINCT rows: project the endpoint per reached vertex, dedup the tuples.
  const seenRows = new Set<string>();
  const rows: Row[] = [];

  for (const v of kept) {
    const env: EvalEnv = { binding: new Map([[bVar, v]]), params, graph };
    const cells = items.map((it) => it.fn(env));
    const key = cells.map(valueKey).join('');

    if (!seenRows.has(key)) {
      seenRows.add(key);
      rows.push(Object.fromEntries(items.map((it, i) => [it.name, cells[i]])));
    }
  }

  if (skipN === 0 && limit === undefined) {
    return rows;
  }

  return rows.slice(skipN, limit === undefined ? undefined : skipN + limit);
};

/**
 * If the projection is exactly `count(DISTINCT <expr over only b>)`, return the
 * arg AST — `'bare'` when it is exactly `b` (so distinct endpoints = the reached
 * set), else the sub-expression (e.g. `b.k`) to evaluate + dedup per reached
 * vertex. `null` when it is not a count-distinct over the endpoint. Uses the same
 * `refsOnlyVar` gate as the native `refs_only_endpoint`, so both engines take the
 * shortcut on the same query (previously TS only accepted a bare `b`, so
 * `count(DISTINCT b.k)` fell through to trail enumeration and faulted where native
 * answered via BFS).
 */
const reachCount = (proj: Projection, bVar: string): { countArg?: CompiledExpr } | null => {
  const first = proj.items[0]?.expr;

  if (
    proj.items.length !== 1 ||
    first?.kind !== 'func' ||
    first.name !== 'count' ||
    !first.distinct ||
    first.star ||
    first.args.length !== 1 ||
    !refsOnlyVar(first.args[0], bVar)
  ) {
    return null;
  }

  const [arg] = first.args;

  // Bare `count(DISTINCT b)` → no arg (distinct endpoints = reached set); an
  // expression (`b.k`) → compile it to evaluate + dedup per reached vertex.
  return arg.kind === 'var' && arg.name === bVar ? {} : { countArg: compileExpr(arg) };
};

/** A pattern that only the general scalar matcher handles — a path selector
 * (`ANY`/`ALL SHORTEST`), a non-default mode (`SIMPLE`/`ACYCLIC`/`WALK`), or a
 * per-hop edge predicate on a quantified segment. The count/vectorized shortcuts
 * enumerate or count trails without the predicate, which is wrong for any. */
const needsGeneralMatcher = (p: PathPattern): boolean =>
  (p.selector ?? 'walk') !== 'walk' ||
  (p.mode ?? 'trail') !== 'trail' ||
  p.segments.some((s) => s.rel.quantifier !== undefined && relHasPredicate(s.rel));

const detectReachableShortcut = (
  clauses: readonly Clause[],
  compiled: readonly CClause[],
): ReachFn | null => {
  if (clauses.length !== 2) {
    return null;
  }

  const [m, ret] = clauses;
  const [cm, cret] = compiled;

  if (
    m.kind !== 'match' ||
    m.optional ||
    m.where !== undefined ||
    m.patterns.length !== 1 ||
    ret.kind !== 'return' ||
    cm.kind !== 'match' ||
    cret.kind !== 'return'
  ) {
    return null;
  }

  if (m.patterns[0].segments.length !== 1) {
    return null;
  }

  // A path selector (`ANY`/`ALL SHORTEST`) or non-default mode (`SIMPLE`/
  // `ACYCLIC`/`WALK`) is handled only by the general matcher — this shortcut
  // counts trails (edge-uniqueness), wrong for either.
  if (needsGeneralMatcher(m.patterns[0])) {
    return null;
  }

  const [{ rel, node }] = m.patterns[0].segments;
  const { quantifier: q } = rel;
  const bVar = node.variable;
  const types = relTypeNames(rel.label);

  // Unbounded (`->+` / `->*`) directed segment, no edge var / props / WHERE, a bare
  // labelled endpoint bound to a variable, a buildable rel type, no ORDER BY.
  if (
    q?.max !== null ||
    rel.variable !== undefined ||
    rel.direction === 'both' ||
    (rel.properties?.length ?? 0) > 0 ||
    rel.where !== undefined ||
    bVar === undefined ||
    (node.properties?.length ?? 0) > 0 ||
    node.where !== undefined ||
    types === null ||
    (ret.projection.orderBy?.length ?? 0) > 0
  ) {
    return null;
  }

  const { projection } = ret;
  const count = reachCount(projection, bVar);
  const isRows = projection.distinct && projection.items.every((it) => refsOnlyVar(it.expr, bVar));

  if (count === null && !isRows) {
    return null;
  }

  const spec: ReachSpec = {
    cstart: cm.patterns[0].start,
    items: cret.projection.items,
    bVar,
    bLabel: node.label,
    out: rel.direction === 'out',
    types: types ?? undefined,
    minZero: q.min === 0,
    isCount: count !== null,
    countArg: count?.countArg,
    skip: projection.skip ?? 0,
    limit: projection.limit,
  };

  return (graph, params) => runReach(spec, graph, params);
};

type CLinear = {
  clauses: readonly CClause[];
  /** Precomputed direct-count closure for `MATCH … RETURN count(*)`; else null. */
  countShortcut: ((graph: Graph, params: Params) => Row) | null;
  /** BFS closure for unbounded var-length + DISTINCT; else null. */
  reachShortcut: ReachFn | null;
};
const compileLinear = (linear: LinearQuery): CLinear => {
  const clauses = linear.clauses.map(compileClause);

  return {
    clauses,
    countShortcut: detectCountShortcut(linear.clauses),
    reachShortcut: detectReachableShortcut(linear.clauses, clauses),
  };
};

// --- write clauses -----------------------------------------------------------

const isEdge = (v: unknown): v is Edge =>
  typeof v === 'object' && v !== null && 'from' in v && 'to' in v;
const isElement = (v: unknown): v is Vertex | Edge =>
  typeof v === 'object' && v !== null && 'id' in v;
const isVertex = (v: unknown): v is Vertex => isElement(v) && !isEdge(v);

const evalProps = (
  props: readonly CProp[],
  b: Binding,
  params: Params,
  graph: Graph,
): Record<string, unknown> => {
  const env: EvalEnv = { binding: b, params, graph };
  const out: Record<string, unknown> = {};

  for (const { key, value } of props) {
    out[key] = value(env);
  }

  return out;
};

/**
 * Insert a vertex, using a string `id` property as the element's identity — so
 * `element_id(n)` equals it and serialization round-trips by domain identity
 * instead of a random synthetic id, while `id` is still stored as an ordinary
 * property. A non-string or absent id mints a synthetic one; a duplicate string id
 * throws (ids are unique). Mirrors the native `insert_vertex_with_id`.
 */
const insertVertexWithId = (
  graph: Graph,
  labels: readonly string[],
  properties: Record<string, unknown>,
): Vertex => {
  const { id } = properties;

  if (typeof id === 'string') {
    if (graph.getVertexById(id) !== null) {
      throw new LenkeError(
        `an element with id '${id}' already exists — a string \`id\` property is the ` +
          `element's unique identity; use _MERGE to upsert, or a fresh id`,
        { code: ErrorCode.ConstraintViolation },
      );
    }

    return graph.addVertex({ id, labels: [...labels], properties });
  }

  return graph.addVertex({ labels: [...labels], properties });
};

/**
 * Insert an edge, using a string `id` property as its external identity — the edge
 * analogue of {@link insertVertexWithId}. Edge ids are unique among edges; a
 * duplicate string id throws. Mirrors native `insert_edge_with_id`.
 */
const insertEdgeWithId = (
  graph: Graph,
  from: Vertex,
  to: Vertex,
  labels: readonly string[],
  properties: Record<string, unknown>,
): Edge => {
  const { id } = properties;

  if (typeof id === 'string') {
    if (graph.getEdgeById(id) !== null) {
      throw new LenkeError(
        `an element with id '${id}' already exists — a string \`id\` property is the ` +
          `element's unique identity; use _MERGE to upsert, or a fresh id`,
        { code: ErrorCode.ConstraintViolation },
      );
    }

    return graph.addEdge({ id, from, to, labels: [...labels], properties });
  }

  return graph.addEdge({ from, to, labels: [...labels], properties });
};

/** Create a node from a pattern, reusing an already-bound variable. */
const ensureNode = (
  graph: Graph,
  binding: Map<string, unknown>,
  node: CInsertNode,
  params: Params,
): Vertex => {
  if (node.variable && binding.has(node.variable)) {
    return binding.get(node.variable) as Vertex;
  }

  const properties = evalProps(node.props, binding, params, graph);

  // A plain INSERT that breaks a unique constraint is rejected — but the check is
  // deferred to commit (via `addVertex`'s constraint chokepoint + `runDeferredChecks`),
  // not eager, so a transient duplicate resolved before commit is allowed. This
  // matches the native engine's R-TX deferred-check semantics; an eager check here
  // wrongly rejected `INSERT` + later `DELETE` of the dup within one transaction
  // (round-12 F3). `_MERGE` still reconciles instead (docs/design/gql-extensions.md §3).
  const vertex = insertVertexWithId(graph, node.labels, properties);

  if (node.variable) {
    binding.set(node.variable, vertex);
  }

  return vertex;
};

const runInsert = (graph: Graph, clause: CInsert, binding: Binding, params: Params): Binding => {
  const out = new Map(binding);

  for (const pattern of clause.patterns) {
    let prev = ensureNode(graph, out, pattern.start, params);

    for (const { rel, node } of pattern.segments) {
      const next = ensureNode(graph, out, node, params);
      const [from, to] = rel.direction === 'in' ? [next, prev] : [prev, next];
      const edge = insertEdgeWithId(
        graph,
        from,
        to,
        rel.labels,
        evalProps(rel.props, out, params, graph),
      );

      if (rel.variable) {
        out.set(rel.variable, edge);
      }

      prev = next;
    }
  }

  return out;
};

/**
 * Infer the conflict key for `_MERGE`: the single unique-constrained key present
 * in the pattern's properties. No applicable constraint → error (can't define
 * "the key"); more than one → ambiguous. See docs/design/gql-extensions.md §2.2.
 */
const inferMergeKey = (
  graph: Graph,
  labels: readonly string[],
  properties: Record<string, unknown>,
): { label: string; key: string; value: unknown } => {
  const candidates: { label: string; key: string; value: unknown }[] = [];

  for (const label of labels) {
    for (const key of graph.uniqueKeys(label)) {
      if (key in properties) {
        candidates.push({ label, key, value: properties[key] });
      }
    }
  }

  if (candidates.length === 0) {
    throw new LenkeError(
      `_MERGE needs a unique constraint on the pattern's label(s) [${labels.join(', ')}] to define the key — declare one with createUniqueConstraint`,
      { code: ErrorCode.InvalidGraphOp },
    );
  }

  if (candidates.length > 1) {
    throw new LenkeError(
      `_MERGE key is ambiguous: the pattern touches multiple unique constraints (${candidates
        .map((c) => `${c.label}.${c.key}`)
        .join(', ')}) — narrow it to one`,
      { code: ErrorCode.InvalidGraphOp },
    );
  }

  return candidates[0];
};

/** Apply `_ON_CREATE` / `_ON_UPDATE` SET items to the merged `vertex`. */
// Apply `_ON_CREATE` / `_ON_UPDATE` SET items to the node or edge each item's
// variable resolves to in `binding` (mirrors `runSet`).
const applyMergeSets = (
  graph: Graph,
  items: readonly CSetItem[],
  binding: Binding,
  params: Params,
): void => {
  for (const item of items) {
    const el = binding.get(item.variable);

    if (!isElement(el)) {
      continue;
    }

    if ('label' in item) {
      if (isEdge(el)) {
        graph.addLabelToEdge(item.label, el);
      } else {
        graph.addLabelToVertex(item.label, el);
      }
    } else {
      el.setProperty(item.key, item.value({ binding, params, graph }));
    }
  }
};

// Resolve a `_MERGE` edge endpoint: the vertex matched by its unique-constraint
// key. Throws (InvalidGraphOp) if no key can be inferred or no vertex matches.
const resolveMergeEndpoint = (
  graph: Graph,
  node: CInsertNode,
  binding: Binding,
  params: Params,
): Vertex => {
  // An endpoint bound by a preceding clause — `MATCH (a), (b) _MERGE (a)-[:R]->(b)`,
  // the natural way to merge an edge between two known vertices — is already a
  // resolved vertex. Use it directly rather than re-inferring a unique key from the
  // (empty) node pattern, which would throw and made the bound-variable form of
  // edge `_MERGE` unusable. Mirrors the native engine's `resolve_merge_endpoint`.
  if (node.variable !== undefined) {
    const bound = binding.get(node.variable);

    if (isVertex(bound)) {
      return bound;
    }
  }

  const properties = evalProps(node.props, binding, params, graph);
  const { label, key, value } = inferMergeKey(graph, node.labels, properties);
  const found = graph.uniqueLookup(label, key, value);

  if (found === undefined) {
    throw new LenkeError(
      `_MERGE: endpoint (:${node.labels.join('&')} {${key}: …}) not found — its key must match an existing vertex`,
      { code: ErrorCode.InvalidGraphOp },
    );
  }

  return found;
};

// `_MERGE` edge form (v1): match both endpoints by key, then upsert the single
// edge between them keyed structurally by (from, to, type). Dispositions apply to
// the edge (which has no key prop, so the default clobbers all its props).
const runMergeEdge = (graph: Graph, clause: CMerge, binding: Binding, params: Params): Binding => {
  const out = new Map(binding);
  const [seg] = clause.pattern.segments;
  const startV = resolveMergeEndpoint(graph, clause.pattern.start, out, params);
  const endV = resolveMergeEndpoint(graph, seg.node, out, params);
  const [from, to] = seg.rel.direction === 'in' ? [endV, startV] : [startV, endV];
  const [relType] = seg.rel.labels;

  if (relType === undefined) {
    throw new LenkeError('_MERGE: an edge must carry exactly one type', {
      code: ErrorCode.InvalidGraphOp,
    });
  }

  const edgeProps = evalProps(seg.rel.props, out, params, graph);

  // Bind the resolved endpoints so the dispositions can read them.
  if (clause.pattern.start.variable) {
    out.set(clause.pattern.start.variable, startV);
  }

  if (seg.node.variable) {
    out.set(seg.node.variable, endV);
  }

  const existing = graph.findEdge(from, to, relType);
  let edge: Edge;

  if (existing === undefined) {
    edge = graph.addEdge({ from, to, labels: [relType], properties: edgeProps });

    if (seg.rel.variable) {
      out.set(seg.rel.variable, edge);
    }

    if (clause.onCreate) {
      applyMergeSets(graph, clause.onCreate, out, params);
    }
  } else {
    edge = existing;

    if (seg.rel.variable) {
      out.set(seg.rel.variable, edge);
    }

    const disp = clause.onUpdate;

    if (disp === undefined) {
      // An edge has no key prop → the default clobbers all its props.
      for (const [k, v] of Object.entries(edgeProps)) {
        edge.setProperty(k, v);
      }
    } else if (disp.kind === 'set') {
      const passes =
        disp.where === undefined || disp.where({ binding: out, params, graph }) === true;

      if (passes) {
        applyMergeSets(graph, disp.items, out, params);
      }
    }
    // disp.kind === 'nothing' → leave the edge untouched.
  }

  if (seg.rel.variable) {
    out.set(seg.rel.variable, edge);
  }

  return out;
};

// `_MERGE` keyed upsert. Node form: match by the constraint key; on miss insert
// the pattern (key + payload) then `_ON_CREATE`; on hit apply the update
// disposition — default clobbers the non-key payload, `_ON_UPDATE SET … [WHERE]`
// replaces it, `_ON_UPDATE_NOTHING` leaves it. One segment → the edge form above;
// multi-hop compound patterns are deferred (v2).
const runMerge = (graph: Graph, clause: CMerge, binding: Binding, params: Params): Binding => {
  if (clause.pattern.segments.length === 1) {
    return runMergeEdge(graph, clause, binding, params);
  }

  if (clause.pattern.segments.length > 1) {
    throw new LenkeError('_MERGE multi-hop compound patterns are not yet supported (v2)', {
      code: ErrorCode.NotImplemented,
    });
  }

  const out = new Map(binding);
  const node = clause.pattern.start;
  const properties = evalProps(node.props, out, params, graph);
  const { label, key, value } = inferMergeKey(graph, node.labels, properties);
  const existing = graph.uniqueLookup(label, key, value);

  let vertex: Vertex;

  if (existing === undefined) {
    vertex = insertVertexWithId(graph, node.labels, properties);

    if (node.variable) {
      out.set(node.variable, vertex);
    }

    if (clause.onCreate) {
      applyMergeSets(graph, clause.onCreate, out, params);
    }
  } else {
    vertex = existing;

    if (node.variable) {
      out.set(node.variable, vertex);
    }

    const disp = clause.onUpdate;

    if (disp === undefined) {
      // Default clobber: write every non-key payload prop to the pattern's value.
      for (const [k, v] of Object.entries(properties)) {
        if (k !== key) {
          vertex.setProperty(k, v);
        }
      }
    } else if (disp.kind === 'set') {
      // An explicit update replaces the default, gated by WHERE if present.
      const passes =
        disp.where === undefined || disp.where({ binding: out, params, graph }) === true;

      if (passes) {
        applyMergeSets(graph, disp.items, out, params);
      }
    }
    // disp.kind === 'nothing' → leave the existing element untouched.
  }

  if (node.variable) {
    out.set(node.variable, vertex);
  }

  return out;
};

// Both labels and properties go through the element's index-maintaining
// mutators (`addLabelTo*` / `setProperty`) so the graph's label and property
// value indexes stay consistent — a later MATCH seeds from `vertexPropertyIndex`,
// so a direct `el.properties =` write would leave that index stale (and skip
// mutation events).
const runSet = (graph: Graph, clause: CSet, binding: Binding, params: Params): void => {
  for (const item of clause.items) {
    const el = binding.get(item.variable);

    if (!isElement(el)) {
      continue;
    }

    if ('label' in item) {
      if (isEdge(el)) {
        graph.addLabelToEdge(item.label, el);
      } else {
        graph.addLabelToVertex(item.label, el);
      }
    } else if (item.key === 'id' && el.id === el.properties.id) {
      // An element keyed by a string `id` has that id as its identity (external id
      // === the `id` property), fixed at creation — re-keying it would break
      // `element_id` / round-trip stability, so reject the SET. A numeric/absent
      // `id` is an ordinary (possibly unique-constrained) property and stays
      // SET-able. Mirrors native `vertex_id_is_identity` / `edge_id_is_identity`.
      throw new LenkeError(
        "cannot SET `id`: a string `id` is the element's identity and is fixed at " +
          'creation — insert a new element with the new id instead',
        { code: ErrorCode.InvalidGraphOp },
      );
    } else {
      const value = item.value({ binding, params, graph });

      // A SET that collides under a unique constraint is rejected via the core
      // property-write chokepoint (`assertUniqueOnSet`), which defers the check to
      // commit inside a transaction. Deferring (rather than an eager check here)
      // lets a transient collision that is reverted before commit succeed, matching
      // the native engine's R-TX semantics (round-12 F3). Constraints are vertex-only.
      el.setProperty(item.key, value);
    }
  }
};

const runRemove = (graph: Graph, clause: CRemove, binding: Binding): void => {
  for (const item of clause.items) {
    const el = binding.get(item.variable);

    if (!isElement(el)) {
      continue;
    }

    if ('label' in item) {
      if (isEdge(el)) {
        graph.removeLabelFromEdge(item.label, el);
      } else {
        graph.removeLabelFromVertex(item.label, el);
      }
    } else {
      el.removeProperty(item.key);
    }
  }
};

const runDelete = (graph: Graph, clause: CDelete, binding: Binding, params: Params): void => {
  for (const target of clause.targets) {
    const el = target({ binding, params, graph });

    if (isEdge(el)) {
      graph.removeEdge(el);
    } else if (isElement(el)) {
      const vertex = el as Vertex;

      // Plain DELETE must not orphan relationships: deleting a still-connected
      // node is a graph violation unless the user opted into DETACH (which
      // cascades the incident edges).
      if (!clause.detach && hasIncidentEdges(graph, vertex)) {
        throw new LenkeError(
          'Cannot delete a node that still has relationships; use DETACH DELETE',
          { code: ErrorCode.InvalidGraphOp },
        );
      }

      graph.removeVertex(vertex);
    }
  }
};

// --- clause processing -------------------------------------------------------

/**
 * Extend a binding through every pattern of a MATCH clause, then filter WHERE.
 * Fully lazy: each pattern is a `flatMap` over the prior stream, so a clause
 * that expands to millions of bindings never materializes them — they flow one
 * at a time to whatever consumes the result.
 */
const matchClauseBindings = (
  graph: Graph,
  clause: CMatch,
  binding: Binding,
  params: Params,
): Iterable<Binding> => {
  let stream: Iterable<Binding> = [binding];

  for (const pattern of clause.patterns) {
    stream = flatMap((b: Binding) => matchPattern(graph, pattern, b, params), stream);
  }

  return clause.where === undefined
    ? stream
    : filter((b: Binding) => clause.where!({ binding: b, params, graph }) === true, stream);
};

/** Per-incoming-binding: stream its matches, or (for OPTIONAL) one null-filled row. */
const matchOrOptional = function* (
  graph: Graph,
  clause: CMatch,
  binding: Binding,
  params: Params,
): Iterable<Binding> {
  let matched = false;

  for (const m of matchClauseBindings(graph, clause, binding, params)) {
    matched = true;

    yield m;
  }

  if (!matched && clause.optional) {
    // No match: keep the row with the pattern's new variables set to null.
    const filled = new Map(binding);

    for (const v of clause.nullVars) {
      if (!filled.has(v)) {
        filled.set(v, null);
      }
    }

    yield filled;
  }
};

/** Lazily expand a binding stream through a MATCH — no intermediate array. */
const runMatch = (
  graph: Graph,
  clause: CMatch,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> =>
  flatMap((binding: Binding) => matchOrOptional(graph, clause, binding, params), bindings);

/**
 * Lazily unwind a list per incoming binding — one row per element (ISO GQL's
 * FOR / UNWIND). A list unwinds its elements; null/undefined yields zero rows;
 * any other scalar unwinds as a one-element list. Matches the Rust engine
 * byte-for-byte. ORDINALITY counts from 1, OFFSET from 0.
 */
/**
 * The built-in procedure catalog: procedure name → its algorithm and non-`node`
 * result column. Output columns are always `[node, <result>]`. Mirrors native
 * `procedure_spec` in plan.rs.
 */
const PROCEDURES: Record<string, { algo: AlgorithmName; resultColumn: string }> = {
  pagerank: { algo: 'pagerank', resultColumn: 'score' },
  personalized_pagerank: { algo: 'personalizedPagerank', resultColumn: 'score' },
  connected_components: { algo: 'connectedComponents', resultColumn: 'componentId' },
  strongly_connected_components: {
    algo: 'stronglyConnectedComponents',
    resultColumn: 'componentId',
  },
  on_cycle: { algo: 'onCycle', resultColumn: 'onCycle' },
  label_propagation: { algo: 'labelPropagation', resultColumn: 'label' },
  peer_pressure: { algo: 'peerPressure', resultColumn: 'cluster' },
  degree: { algo: 'degree', resultColumn: 'degree' },
  betweenness: { algo: 'betweenness', resultColumn: 'centrality' },
  closeness: { algo: 'closeness', resultColumn: 'centrality' },
  shortest_path: { algo: 'shortestPath', resultColumn: 'distance' },
  neighbor_aggregate: { algo: 'neighborAggregate', resultColumn: 'vector' },
};

const procedureSpec = (name: string): { algo: AlgorithmName; resultColumn: string } | null =>
  PROCEDURES[name] ?? null;

/**
 * For an unknown procedure name, the canonical snake_case name it most likely
 * meant — matched by ignoring case and `_` separators, so a camelCase spelling
 * (`pageRank`, `connectedComponents`) resolves to its surface name. `null` when
 * nothing plausibly matches. Mirrors the native `suggest_procedure` so both
 * engines' "did you mean" faults read identically.
 */
const normProcName = (s: string): string => s.replaceAll('_', '').toLowerCase();

const suggestProcedure = (name: string): string | null => {
  const target = normProcName(name);

  return Object.keys(PROCEDURES).find((n) => normProcName(n) === target) ?? null;
};

/** The accepted `CALL <algo>({...})` config keys and the value type each expects.
 * Order is fixed so the "did you mean" tie-break matches the native engine. */
const ALGO_CONFIG_TYPES: ReadonlyArray<
  readonly [string, 'string' | 'number' | 'stringList' | 'boolean']
> = [
  ['edgeLabel', 'string'],
  ['direction', 'string'],
  ['weightProperty', 'string'],
  ['dampingFactor', 'number'],
  ['iterations', 'number'],
  ['pivots', 'number'],
  ['seedProperty', 'string'],
  ['source', 'string'],
  ['sourceNodes', 'stringList'],
  ['target', 'string'],
  ['writeProperty', 'string'],
  ['algorithm', 'string'],
  ['heuristicProperty', 'string'],
  ['feature', 'string'],
  ['op', 'string'],
  ['includeSelf', 'boolean'],
];

/** Case-insensitive Levenshtein edit distance — a plain DP over code points,
 * matching the native `edit_distance` so both engines suggest identically. */
const editDistance = (a: string, b: string): number => {
  const x = Array.from(a.toLowerCase());
  const y = Array.from(b.toLowerCase());
  let prev = Array.from({ length: y.length + 1 }, (_, i) => i);
  const cur = new Array<number>(y.length + 1);

  for (let i = 0; i < x.length; i++) {
    cur[0] = i + 1;

    for (let j = 0; j < y.length; j++) {
      const cost = x[i] === y[j] ? 0 : 1;
      cur[j + 1] = Math.min(prev[j + 1] + 1, cur[j] + 1, prev[j] + cost);
    }

    prev = cur.slice();
  }

  return prev[y.length];
};

/** For an unknown config key, the closest known key within edit distance 2 (else
 * null). Scans in fixed order so ties resolve to the earliest — identical to native. */
const suggestConfigKey = (name: string): string | null => {
  let best: string | null = null;
  let bestDist = 3;

  for (const [key] of ALGO_CONFIG_TYPES) {
    const d = editDistance(name, key);

    if (d <= 2 && d < bestDist) {
      best = key;
      bestDist = d;
    }
  }

  return best;
};

/** Set one algorithm-config field from a CALL config-map entry. An unknown key
 * (with a "did you mean" hint) or a wrong-typed value is an error — a silently
 * dropped key once hid the `pivots` bug. Mirrors native `apply_algo_config`. */
const applyAlgoConfig = (cfg: AlgorithmConfig, key: string, v: unknown): void => {
  const spec = ALGO_CONFIG_TYPES.find(([k]) => k === key);

  if (spec === undefined) {
    const s = suggestConfigKey(key);

    throw new LenkeError(
      s ? `unknown config key '${key}' (did you mean '${s}'?)` : `unknown config key '${key}'`,
      { code: ErrorCode.InvalidValue },
    );
  }

  const [, expected] = spec;
  const okByType: Record<typeof expected, boolean> = {
    string: typeof v === 'string',
    number: typeof v === 'number',
    stringList: Array.isArray(v),
    boolean: typeof v === 'boolean',
  };

  if (!okByType[expected]) {
    const label = expected === 'stringList' ? 'a list' : `a ${expected}`;

    throw new LenkeError(`config key '${key}' expects ${label}`, { code: ErrorCode.InvalidValue });
  }

  (cfg as Record<string, unknown>)[key] =
    expected === 'stringList' ? (v as unknown[]).filter((x) => typeof x === 'string') : v;
};

/**
 * `[OPTIONAL] CALL name(config) YIELD …`: run the algorithm once (uncorrelated),
 * then cross-join its rows into the binding stream, binding each yielded column.
 * OPTIONAL keeps the outer row (null-filled) when the procedure yields nothing.
 */
const runCall = (
  graph: Graph,
  clause: CCallNamed,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> => {
  if (!clause.algo) {
    const suggestion = suggestProcedure(clause.procName);
    const msg = suggestion
      ? `unknown procedure: ${clause.procName} (did you mean '${suggestion}'?)`
      : `unknown procedure: ${clause.procName}`;

    throw new LenkeError(msg, { code: ErrorCode.Unsupported });
  }

  const config: AlgorithmConfig = {};
  const scratch: Binding = new Map();

  for (const c of clause.config) {
    applyAlgoConfig(config, c.key, c.value({ binding: scratch, params, graph }));
  }

  const rows = runAlgorithmSync(clause.algo, config, graph) as Array<Record<string, unknown>>;

  // Validate the YIELD columns against what the procedure actually exposes: a
  // vertex handle as `node`, plus its single result column. Anything else is an
  // error — previously an undeclared column (`YIELD nodeId`, or any typo) bound
  // silently to `undefined`, so the query returned rows with that column simply
  // missing. A silent wrong answer, and a divergence: native raises
  // E_INVALID_VALUE here. Checked after the algorithm runs, mirroring native, so a
  // `writeProperty` side effect lands identically on both engines.
  const resultColumn = procedureSpec(clause.procName)?.resultColumn;

  for (const bind of clause.binds) {
    if (bind.column !== 'node' && bind.column !== resultColumn) {
      throw new LenkeError(
        `procedure \`${clause.procName}\` has no output column \`${bind.column}\``,
        { code: ErrorCode.InvalidValue },
      );
    }
  }

  // Materialize the outer bindings first: the call may write a property, and the
  // outer stream must be read against the pre-write graph.
  const outer = toArray(bindings);

  return flatMap((binding: Binding) => {
    if (rows.length === 0 && clause.optional) {
      const b = new Map(binding);

      for (const bind of clause.binds) {
        b.set(bind.var, null);
      }

      return [b];
    }

    return rows.map((r) => {
      const b = new Map(binding);

      for (const bind of clause.binds) {
        // `node` binds as the live Vertex handle (so it hydrates only when
        // actually returned whole, and `node.name` / further MATCH work);
        // mirrors native's `Val::Node`. Other columns are the raw value.
        const value =
          bind.column === 'node' ? graph.verticesById.get(r.node as string) : r[bind.column];
        b.set(bind.var, value);
      }

      return b;
    });
  }, outer);
};

/**
 * `[OPTIONAL] CALL (scope) { … }`: run the nested query once per outer row
 * (correlated / lateral), seeding it with only the imported scope variables, and
 * merge the nested RETURN columns back — duplicating the outer row per nested row.
 * OPTIONAL keeps the outer row (nested columns null-filled) when the subquery is
 * empty; a non-OPTIONAL empty subquery drops the outer row.
 */
const runCallInline = (
  graph: Graph,
  clause: CCallInline,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> =>
  flatMap((outer: Binding) => {
    // Import only the scoped variables into the subquery's initial binding.
    const seed = new Map<string, unknown>();

    for (const v of clause.scope) {
      if (outer.has(v)) {
        seed.set(v, outer.get(v));
      }
    }

    let nested = runLinearClauses(clause.body, graph, params, new Map(seed));

    // Fold in any set-op parts (`… UNION/EXCEPT/INTERSECT …`), each run against
    // the same seed, matching the top-level set-op semantics.
    for (const { op, part } of clause.bodyMore) {
      nested = combineRows(op, nested, runLinearClauses(part, graph, params, new Map(seed)));
    }

    if (nested.length === 0 && clause.optional) {
      const b = new Map(outer);

      for (const col of clause.returnColumns) {
        b.set(col, null);
      }

      return [b];
    }

    return nested.map((row) => {
      const b = new Map(outer);

      for (const [k, val] of Object.entries(row)) {
        b.set(k, val);
      }

      return b;
    });
  }, bindings);

const runFor = (
  graph: Graph,
  clause: CFor,
  bindings: Iterable<Binding>,
  params: Params,
): Iterable<Binding> =>
  flatMap((binding: Binding) => {
    const listv = clause.list({ binding, params, graph });
    let elems: unknown[];

    if (listv === null || listv === undefined) {
      elems = [];
    } else if (Array.isArray(listv)) {
      elems = listv;
    } else {
      elems = [listv];
    }

    return elems.map((elem, i) => {
      const b = new Map(binding);
      b.set(clause.alias, elem);

      if (clause.ordinality) {
        b.set(clause.ordinality.var, clause.ordinality.kind === 'ordinality' ? i + 1 : i);
      }

      return b;
    });
  }, bindings);

const mapToRow = (b: Binding): Row => {
  const row: Row = {};

  for (const [k, v] of b) {
    row[k] = v;
  }

  return row;
};

const WRITE_CLAUSES = new Set(['insert', 'merge', 'set', 'remove', 'delete']);

/** Does a query mutate the graph (contain any INSERT/MERGE/SET/REMOVE/DELETE)? */
const queryHasWrite = (query: Query): boolean =>
  query.parts.some((part) => part.clauses.some((clause) => WRITE_CLAUSES.has(clause.kind)));

/**
 * Execute an ISO GQL transaction-control command (`START TRANSACTION`/`COMMIT`/
 * `ROLLBACK`) by driving the session's transaction frame on the graph. Returns
 * nothing (a write-only shape — no rows/columns). ISO semantics are enforced
 * here, not in the core primitives:
 *  - `START TRANSACTION` while one is already active → `E_INVALID_GRAPH_OP`
 *    (ISO forbids nesting). The graph's tx depth reflects only explicit
 *    transactions here, since a TxControl is not a write and so is never wrapped
 *    in a per-statement auto-commit frame.
 *  - `COMMIT`/`ROLLBACK` with no active transaction → `E_INVALID_GRAPH_OP`. The
 *    depth is checked in the executor so ROLLBACK is symmetric with COMMIT
 *    *without* changing the core `rollbackTransaction`'s idempotent contract.
 *  - The READ ONLY access mode is recorded on the graph (and cleared on
 *    commit/rollback); a subsequent write statement consults it (see `execute`).
 */
const runTxControl = (tx: TxControl, graph: Graph): void => {
  switch (tx.kind) {
    case 'start':
      if (graph.isTransacting()) {
        throw new LenkeError('START TRANSACTION: a transaction is already active', {
          code: ErrorCode.InvalidGraphOp,
        });
      }

      graph.beginTransaction();
      graph.setTransactionReadOnly(tx.accessMode === 'read only');
      break;
    case 'commit':
      if (!graph.isTransacting()) {
        throw new LenkeError('COMMIT: no active transaction', { code: ErrorCode.InvalidGraphOp });
      }

      try {
        graph.commitTransaction(); // may throw (deferred checks) after rolling back
      } finally {
        graph.setTransactionReadOnly(false);
      }

      break;
    case 'rollback':
      if (!graph.isTransacting()) {
        throw new LenkeError('ROLLBACK: no active transaction', { code: ErrorCode.InvalidGraphOp });
      }

      graph.rollbackTransaction();
      graph.setTransactionReadOnly(false);
      break;
  }
};

/**
 * Run one compiled linear query (clause sequence) to result rows. A statement
 * that writes runs inside one transaction, so a mid-statement fault (e.g. a
 * later row of a multi-row INSERT violating a constraint) rolls the earlier rows
 * back instead of leaving the write half-applied — per-statement atomicity,
 * byte-identical to the native engine's auto-commit frame. Read-only statements
 * skip the frame (no undo/commit overhead).
 */
const runLinear = (linear: CLinear, graph: Graph, params: Params): Row[] => {
  const writes = linear.clauses.some((clause) => WRITE_CLAUSES.has(clause.kind));

  if (!writes) {
    return runLinearClauses(linear, graph, params);
  }

  return graph.transaction(() => runLinearClauses(linear, graph, params));
};

const runLinearClauses = (
  linear: CLinear,
  graph: Graph,
  params: Params,
  initial?: Binding,
): Row[] => {
  // The fast paths assume an empty start; a seeded (inline-subquery) run skips
  // them and takes the general clause loop.
  if (initial === undefined) {
    // Direct `count(*)` shortcut (edge-bucket size / degree product) — skips
    // enumerating every match. Only fires for the exact `MATCH … RETURN count(*)`
    // shapes `detectCountShortcut` accepts.
    if (linear.countShortcut) {
      return [linear.countShortcut(graph, params)];
    }

    // Unbounded var-length + DISTINCT → BFS the reachable set instead of enumerating
    // trails (exponential, hits the trail budget). See `detectReachableShortcut`.
    if (linear.reachShortcut) {
      return linear.reachShortcut(graph, params);
    }
  }

  // Bindings flow as a lazy stream; only barriers (mutations, aggregation,
  // ORDER BY) force materialization — so a streaming read never holds the whole
  // result set in memory.
  let bindings: Iterable<Binding> = [initial ?? new Map()];

  for (const clause of linear.clauses) {
    switch (clause.kind) {
      case 'match':
        bindings = runMatch(graph, clause, bindings, params);
        break;
      case 'for':
        bindings = runFor(graph, clause, bindings, params);
        break;
      case 'callNamed':
        bindings = runCall(graph, clause, bindings, params);
        break;
      case 'callInline':
        bindings = runCallInline(graph, clause, bindings, params);
        break;
      case 'with': {
        const projected = applyProjection(clause.projection, bindings, params, graph);
        bindings =
          clause.where === undefined
            ? projected
            : filter(
                (b: Binding) => clause.where!({ binding: b, params, graph }) === true,
                projected,
              );
        break;
      }
      case 'filter':
        // ISO §14.6: drop rows where the condition is not TRUE (three-valued).
        bindings = filter(
          (b: Binding) => clause.where({ binding: b, params, graph }) === true,
          bindings,
        );
        break;
      case 'let':
        // ISO §14.7: bind new vars additively, left-to-right (a later item sees
        // an earlier one via the in-progress binding copy).
        bindings = map((b: Binding) => {
          const nb = new Map(b);

          for (const it of clause.items) {
            nb.set(it.var, it.expr({ binding: nb, params, graph }));
          }

          return nb;
        }, bindings);
        break;
      case 'insert':
        // Mutations must run eagerly and exactly once — force evaluation.
        bindings = toArray(map((b: Binding) => runInsert(graph, clause, b, params), bindings));
        break;
      case 'merge':
        bindings = toArray(map((b: Binding) => runMerge(graph, clause, b, params), bindings));
        break;
      case 'set': {
        const arr = toArray(bindings);

        for (const b of arr) {
          runSet(graph, clause, b, params);
        }

        bindings = arr;
        break;
      }
      case 'remove': {
        const arr = toArray(bindings);

        for (const b of arr) {
          runRemove(graph, clause, b);
        }

        bindings = arr;
        break;
      }
      case 'delete': {
        const arr = toArray(bindings);

        for (const b of arr) {
          runDelete(graph, clause, b, params);
        }

        bindings = arr;
        break;
      }
      case 'finish':
        return [];
      case 'return':
        return toArray(map(mapToRow, applyProjection(clause.projection, bindings, params, graph)));
    }
  }

  return []; // a write-only query produces no rows
};

// --- set operations ----------------------------------------------------------

/** Stable key for a result row; graph-element columns key by id. */
const rowKeyOf = (row: Row): string =>
  Object.entries(row)
    .map(([k, v]) => `${k}=${valueKey(v)}`)
    .join('');

const distinctRows = (rows: readonly Row[]): Row[] => {
  const seen = new Set<string>();

  return rows.filter((r) => {
    const k = rowKeyOf(r);

    return seen.has(k) ? false : (seen.add(k), true);
  });
};

/** Combine two row sets per a set operator. */
const combineRows = (op: SetOp, left: readonly Row[], right: readonly Row[]): Row[] => {
  const rightKeys = new Set(right.map(rowKeyOf));

  switch (op.op) {
    case 'union':
      return op.all ? [...left, ...right] : distinctRows([...left, ...right]);
    case 'except': {
      const kept = left.filter((r) => !rightKeys.has(rowKeyOf(r)));

      return op.all ? kept : distinctRows(kept);
    }
    case 'intersect': {
      const kept = left.filter((r) => rightKeys.has(rowKeyOf(r)));

      return op.all ? kept : distinctRows(kept);
    }
  }
};

// --- compile & execute -------------------------------------------------------

/** A whole compiled query: its linear parts and the set operators joining them. */
type CQuery = { parts: readonly CLinear[]; ops: readonly SetOp[] };

/**
 * Revive a single param value: a single-key tagged-temporal object
 * (`{'@date':'…'}`) becomes its `Temporal`, a list has its elements revived
 * (mirroring the Rust param parser, which revives tagged temporals inside a list
 * too), anything else passes through unchanged.
 */
const reviveParamValue = (v: unknown): unknown => {
  if (Array.isArray(v)) {
    return v.map(reviveParamValue);
  }

  return fromTaggedJson(v) ?? v;
};

/**
 * A param value of `undefined`, a function, or a symbol is dropped by the native
 * FFI's `JSON.stringify` param marshalling, so its binding reads as MISSING there
 * (→ `E_MISSING_PARAMETER`). The TS engine must agree instead of silently
 * evaluating `$name` to `undefined` (which returns `[]` for `WHERE n.x = $name`
 * with no error). (D2)
 */
const isEffectivelyMissing = (v: unknown): boolean =>
  v === undefined || typeof v === 'function' || typeof v === 'symbol';

/**
 * Validate one already-revived param value against the LPG param model, matching
 * the native FFI param decoder (`gql/params.rs`) so both engines accept and reject
 * exactly the same inputs (D3). Accepts a scalar (`string | number | boolean |
 * null`), a revived tagged-temporal instance, or a FLAT list of those. Rejects:
 *   - a `bigint` → `E_INVALID_VALUE` (float64 model; native rejects it JS-side in
 *     `stringifyParams` before the FFI crossing)
 *   - a nested list, or a plain (non-temporal) object → `E_INVALID_JSON` (native:
 *     "nested arrays are not valid param values" / "the only valid object param
 *     value is a tagged temporal")
 * A tagged-temporal object is already a `Temporal` instance by this point
 * (`reviveParams` ran first), so it passes as a scalar — never mistaken for a
 * rejected plain object.
 */
/**
 * Does `s` contain a lone (unpaired) UTF-16 surrogate? Mirrors
 * `@lenke/serialization`'s `hasLoneSurrogate` (duplicated to avoid a package
 * dependency): the native store is UTF-8 and rejects a lone surrogate as it
 * JSON-decodes a param crossing the FFI boundary, so the TS param path must
 * reject it too for byte-identity (a JS string can carry one; native cannot).
 */
const hasLoneSurrogate = (s: string): boolean => {
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);

    if (c >= 0xd800 && c <= 0xdbff) {
      // `charCodeAt` past the end is NaN; the positive-form test treats an
      // end-of-string high surrogate (no following low) as lone.
      const next = s.charCodeAt(i + 1);

      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        return true;
      }

      i++;
    } else if (c >= 0xdc00 && c <= 0xdfff) {
      return true;
    }
  }

  return false;
};

const validateParamScalar = (name: string, v: unknown): void => {
  if (typeof v === 'string') {
    if (hasLoneSurrogate(v)) {
      throw new LenkeError(
        `parameter $${name} is a string containing a lone (unpaired) UTF-16 surrogate, ` +
          `which is not a valid Unicode scalar in the LPG string model`,
        { code: ErrorCode.InvalidJson, details: { param: name } },
      );
    }

    return;
  }

  if (v === null || typeof v === 'boolean' || typeof v === 'number' || isTemporal(v)) {
    return;
  }

  if (typeof v === 'bigint') {
    throw new LenkeError(
      `a bigint parameter ($${name}) is not supported: the numeric model is float64 — ` +
        `pass Number(x) or a string`,
      { code: ErrorCode.InvalidValue, details: { param: name } },
    );
  }

  throw new LenkeError(
    `parameter $${name} is outside the LPG param model: only a scalar, a flat list ` +
      `of scalars, or a tagged-temporal object is a valid param value`,
    { code: ErrorCode.InvalidJson, details: { param: name } },
  );
};

const validateParamValue = (name: string, v: unknown): void => {
  if (Array.isArray(v)) {
    for (const el of v) {
      if (Array.isArray(el)) {
        throw new LenkeError(`parameter $${name}: nested arrays are not valid param values`, {
          code: ErrorCode.InvalidJson,
          details: { param: name },
        });
      }

      validateParamScalar(name, el);
    }

    return;
  }

  validateParamScalar(name, v);
};

/** Revive every param value, sharing the input object when nothing changed. */
const reviveParams = (params: Params): Params => {
  let out: Params | null = null;

  for (const key of Object.keys(params)) {
    const revived = reviveParamValue(params[key]);

    if (revived !== params[key]) {
      out ??= { ...params };
      out[key] = revived;
    }
  }

  return out ?? params;
};

/**
 * Compile a parsed query into a reusable `Plan`. All graph/param-independent
 * work — operator dispatch, aggregate detection, alias resolution, label-seed
 * selection — happens here, once. Run the returned plan against any graph and
 * params; it never re-parses or re-analyzes.
 */
export const compile = <R extends Row = Row>(query: Query): Plan<R> => {
  const referenced = new Set<string>();
  const unknownFns = new Set<string>();
  const countParams = new Set<string>();
  const prevParam = paramCollector;
  const prevUnknown = unknownFnCollector;
  const prevCount = countParamCollector;
  paramCollector = referenced;
  unknownFnCollector = unknownFns;
  countParamCollector = countParams;

  let compiled: CQuery;

  try {
    compiled = { parts: query.parts.map(compileLinear), ops: query.ops };
  } finally {
    paramCollector = prevParam;
    unknownFnCollector = prevUnknown;
    countParamCollector = prevCount;
  }

  // Eager unknown-function rejection: a name the query references that resolves
  // to no scalar (or aggregate) function is never valid — throw NOW, at compile
  // time, before the plan runs, so `compile(parse(q))` / `query(...)` faults
  // identically over zero rows, one row, or a never-taken branch. Matches the
  // Rust engine, which raises the same coded error from `run_cquery_body` off the
  // plan's `unknown_fns`. The message names the offending function(s) verbatim.
  if (unknownFns.size > 0) {
    const named = [...unknownFns].map((n) => `${n}()`).join(', ');

    throw new LenkeError(`call to an unknown or unimplemented function: ${named}`, {
      code: ErrorCode.UnknownFunction,
    });
  }

  const names = [...referenced];
  const countNames = [...countParams];

  // Rows are `Row` at runtime; `R` is the caller's asserted shape (see `Plan`).
  const plan: Plan = (graph, rawParams = {}) => {
    // Revive any single-key tagged-temporal object param (`{'@date':'…'}`,
    // `@datetime`, `@localtime`, `@zoned_time`, `@zoned_datetime`, `@duration`)
    // into its temporal value, so the engine's OWN tagged output round-trips as
    // an input param. The Rust engine already does this while parsing its param
    // string (`temporal_object`); this closes the byte-identity gap.
    const params = reviveParams(rawParams);

    // Eager param validation: a `$name` the query references but the caller
    // didn't bind is a programming error — throw before running, not a silent
    // empty result. (The Rust engine does the same in `positional`.)
    for (const name of names) {
      const present = Object.hasOwn(params, name);
      const value = present ? params[name] : undefined;

      // The reserved `$__now` (from a bare `current_*` function) is optional: an
      // unsupplied `now` reads as NULL (so `current_date` → null), not an error.
      // A bound value of `undefined`/function/symbol counts as MISSING too (D2),
      // because native's `JSON.stringify` param marshalling drops such keys.
      if (name !== '__now' && (!present || isEffectivelyMissing(value))) {
        throw new LenkeError(`missing parameter: $${name}`, {
          code: ErrorCode.MissingParameter,
          details: { param: name },
        });
      }

      // Validate the value against the LPG param model — the same rules the native
      // FFI decoder enforces (D3): a bigint is rejected (float64 model), and a
      // nested list / plain (non-temporal) object is rejected rather than reaching
      // the engine as a silent no-op. Skips `$__now` when it wasn't supplied.
      if (present && !isEffectivelyMissing(value)) {
        validateParamValue(name, value);
      }
    }

    // Eager LIMIT/OFFSET bound-value validation: a `$param` used as a bound must
    // resolve to a non-negative integer. Checked here, before any row is produced,
    // so a bad bound faults identically over zero rows or many — mirroring the Rust
    // engine's `check_count_params`. (Missing/bigint binds are already caught above.)
    for (const name of countNames) {
      const v = params[name];

      if (typeof v !== 'number' || !Number.isInteger(v) || v < 0) {
        throw new LenkeError('a LIMIT/OFFSET parameter must resolve to a non-negative integer', {
          code: ErrorCode.InvalidValue,
          details: { param: name },
        });
      }
    }

    let rows = runLinear(compiled.parts[0], graph, params);
    compiled.ops.forEach((op, i) => {
      rows = combineRows(op, rows, runLinear(compiled.parts[i + 1], graph, params));
    });

    return rows;
  };

  return plan as Plan<R>;
};

/**
 * Compile and run a parsed statement in one call (no plan reuse). A
 * transaction-control command (`START TRANSACTION`/`COMMIT`/`ROLLBACK`) drives the
 * session's transaction frame and returns no rows; a linear query compiles and
 * runs as usual. READ ONLY enforcement lives here, at the statement level: a write
 * statement (any INSERT/MERGE/SET/REMOVE/DELETE) run while the active transaction
 * is READ ONLY is rejected *before* it applies — no mutator is touched.
 */
export const execute = <R extends Row = Row>(
  stmt: Statement,
  graph: Graph,
  params: Params = {},
): R[] => {
  if (isTxControl(stmt)) {
    runTxControl(stmt, graph);

    return [];
  }

  if (graph.isReadOnlyTransaction() && queryHasWrite(stmt)) {
    throw new LenkeError('write statement rejected: the active transaction is READ ONLY', {
      code: ErrorCode.InvalidGraphOp,
    });
  }

  return compile<R>(stmt)(graph, params);
};
