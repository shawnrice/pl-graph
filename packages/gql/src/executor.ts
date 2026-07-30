import type { Edge, Graph, Path, Vertex } from '@lenke/core';
import {
  type AlgorithmName,
  Duration,
  fromTaggedJson,
  isTemporal,
  LocalDate,
  LocalTime,
  LocalDateTime,
  LenkeRecord,
  temporalCmpTotal,
  temporalRelCmp,
  ZonedDateTime,
  ZonedTime,
} from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';
import { filter, map, skip, take, toArray } from '@lenke/fp';

import type {
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
  Quantifier,
  Query,
  RelPattern,
  Segment,
  RemoveItem,
  SetItem,
  SetOp,
  Statement,
  TypeTest,
} from './ast.js';
import { isTxControl } from './ast.js';
import { labelsMatch } from './graph-queries.js';

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
export type Binding = ReadonlyMap<string, unknown>;

/** A projected result row: alias/derived-name -> value. */
export type Row = Record<string, unknown>;

/** Query parameters (`$name`), supplied at run time and threaded through plans. */
export type Params = Record<string, unknown>;

/**
 * The environment a compiled expression evaluates against: the current binding,
 * the query params, the graph (only EXISTS/COUNT subqueries read it), and — for
 * aggregates — the `group` of bindings being folded. Passing this explicitly
 * keeps expression evaluation a pure function of its inputs (no run-state global).
 */
export type EvalEnv = {
  binding: Binding;
  params: Params;
  graph: Graph;
  group?: readonly Binding[];
};

/**
 * A compiled expression. The structural `switch (expr.kind)` happens once, at
 * compile time; what's left is this closure, evaluated against an `EvalEnv`.
 */
export type CompiledExpr = (env: EvalEnv) => unknown;

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

export const withBinding = (
  binding: Binding,
  name: string | undefined,
  value: Bound | Path,
): Binding => {
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
export const consistent = (binding: Binding, name: string | undefined, value: Bound): boolean => {
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

// Element identity, mirroring the Rust `val_eq`: nodes/edges are equal iff same
// kind + same id; a non-element falls back to structural (value) equality.
const sameElement = (a: unknown, b: unknown): boolean => {
  if (isVertex(a) && isVertex(b)) {
    return a.id === b.id;
  }

  if (isEdge(a) && isEdge(b)) {
    return a.id === b.id;
  }

  if (isElement(a) || isElement(b)) {
    return false; // element vs non-element (or node vs edge)
  }

  return structuralEq(a, b);
};

// The graph-element predicates (`IS DIRECTED` / `IS SOURCE|DESTINATION OF` /
// `ALL_DIFFERENT` / `SAME`). Three-valued: `null` on a null operand or a type
// mismatch. Mirrors the Rust `eval_graph_pred`.
const graphPredResult = (predKind: string, vals: readonly unknown[]): boolean | null => {
  switch (predKind) {
    case 'directed':
      return isEdge(vals[0]) ? true : null; // every edge is directed; else unknown
    case 'sourceOf':
    case 'destOf': {
      const [node, edge] = vals;

      if (!isVertex(node) || !isEdge(edge)) {
        return null;
      }

      return (predKind === 'sourceOf' ? edge.from : edge.to).id === node.id;
    }
    case 'allDifferent':
    case 'same': {
      if (vals.some(isNullish)) {
        return null;
      }

      if (predKind === 'same') {
        return vals.every((v) => sameElement(v, vals[0]));
      }

      for (let i = 0; i < vals.length; i++) {
        for (let j = i + 1; j < vals.length; j++) {
          if (sameElement(vals[i], vals[j])) {
            return false;
          }
        }
      }

      return true;
    }
    default:
      return null;
  }
};

// Does a NON-null value match a scalar type category? Numeric split: `integer` =
// a whole-valued number, `float` = any number (one f64 numeric type; boundary
// inference). The open record (`ANY RECORD`) is a `TypeTest`, not a category —
// handled in `valueIsTypedTy`. Mirrors the Rust `category_matches`.
const categoryMatches = (v: unknown, category: string): boolean => {
  switch (category) {
    case 'any':
      return true;
    case 'null':
      return false;
    case 'bool':
      return typeof v === 'boolean';
    case 'string':
      return typeof v === 'string';
    case 'integer':
      return typeof v === 'number' && Number.isInteger(v);
    case 'float':
      return typeof v === 'number';
    case 'list':
      return Array.isArray(v);
    case 'date':
      return v instanceof LocalDate;
    case 'local_time':
      return v instanceof LocalTime;
    case 'local_datetime':
      return v instanceof LocalDateTime;
    case 'zoned_time':
      return v instanceof ZonedTime;
    case 'zoned_datetime':
      return v instanceof ZonedDateTime;
    case 'duration':
      return v instanceof Duration;
    default:
      return false;
  }
};

// The ISO value-type predicate `x IS TYPED <value type> [NOT NULL]`. Null conforms
// to any nullable type, so a null value is `!notNull`. A closed `RECORD {…}` is
// CLOSED on extras and matches each present field's value against its type (a field
// null/absent is OK unless the field is NOT NULL). Mirrors the Rust
// `value_is_typed_ty`.
const valueIsTypedTy = (v: unknown, ty: TypeTest, notNull: boolean): boolean => {
  if (isNullish(v)) {
    return !notNull;
  }

  if (ty.kind === 'scalar') {
    return categoryMatches(v, ty.category);
  }

  if (ty.kind === 'anyRecord') {
    return v instanceof LenkeRecord;
  }

  if (!(v instanceof LenkeRecord)) {
    return false;
  }

  // Closed: every present key must be a declared field.
  const declared = new Set(ty.fields.map(([k]) => k));

  for (const k of v.keys()) {
    if (!declared.has(k)) {
      return false;
    }
  }

  return ty.fields.every(([k, ft, fieldNotNull]) =>
    v.has(k) ? valueIsTypedTy(v.get(k), ft, fieldNotNull) : !fieldNotNull,
  );
};

// `PROPERTY_EXISTS(n, key)`: is `key` a *present* property of element `n`? A
// boolean for an element (distinguishing an absent key from a stored null — null
// is first-class), and `null` for a non-element / NULL (three-valued). `key in
// props` would walk the prototype chain, so use `hasOwnProperty`.
const propPresent = (bound: unknown, key: string): boolean | null => {
  const props = (bound as { properties?: Record<string, unknown> } | undefined)?.properties;

  return props == null ? null : Object.hasOwn(props, key);
};

// Scalar functions, three-valued logic & value helpers (leaf module).
import {
  AGGREGATES,
  ARITH,
  arithStep,
  asTruth,
  BOOL3,
  callScalar,
  cmpKey,
  COMPARE,
  concatStep,
  dataException,
  hasAggregate,
  inList,
  isNullish,
  makeRecord,
  nonNumericAgg,
  compareCodePoints,
  not3,
  numOf,
  percentileOf,
  recordGet,
  structuralEq,
  temporalSum,
  unsupportedTemporalAgg,
} from './executor/scalars.js';
import type { FuncExpr } from './executor/scalars.js';

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
export const resolveCount = (v: CountValue | undefined, params: Params): number | undefined => {
  if (v === undefined || typeof v === 'number') {
    return v;
  }

  return Number(params[v.param]);
};

export const compileExpr = (expr: Expr): CompiledExpr => {
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
    case 'property_exists': {
      const { variable, key } = expr;

      return (env) => propPresent(env.binding.get(variable), key);
    }
    case 'list': {
      const items = expr.items.map(compileExpr);

      return (env) => items.map((f) => f(env));
    }
    case 'record': {
      // ISO record constructor → a canonical (sorted, dup-last-wins) map value.
      const fields = expr.fields.map((f) => ({ key: f.key, fn: compileExpr(f.value) }));

      return (env) => makeRecord(fields.map((f) => [f.key, f.fn(env)] as const));
    }
    case 'index': {
      // ISO GQL list subscript `base[index]`: 0-based, out of range → null, and
      // null-safe. A STRING index on a record/map is field access; a non-string
      // index / non-integer list index → null. `numOf` mirrors native `num_of`.
      const baseF = compileExpr(expr.base);
      const idxF = compileExpr(expr.index);

      return (env) => {
        const base = baseF(env);
        const idx = idxF(env);

        if (base instanceof LenkeRecord) {
          return typeof idx === 'string' ? recordGet(base, idx) : null;
        }

        const i = numOf(idx);

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
      // `.field` chained off any expression. A record/map base reads the field by
      // name; a node/edge base reads its stored property via `propOf`; anything
      // else → null, exactly like the bare `prop` path.
      const baseF = compileExpr(expr.base);
      const { key } = expr;

      return (env) => {
        const base = baseF(env);

        return base instanceof LenkeRecord ? recordGet(base, key) : propOf(base, key);
      };
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
    case 'isTyped': {
      // `x IS [NOT] TYPED <value type> [NOT NULL]` — the ISO value-type predicate.
      const fn = compileExpr(expr.expr);
      const { ty, notNull, negated } = expr;

      return (env) => {
        const m = valueIsTypedTy(fn(env), ty, notNull);

        return negated ? !m : m;
      };
    }
    case 'graphPred': {
      // IS DIRECTED / IS SOURCE|DESTINATION OF / ALL_DIFFERENT / SAME.
      const argFns = expr.args.map(compileExpr);
      const { predKind, negated } = expr;

      return (env) => {
        const r = graphPredResult(
          predKind,
          argFns.map((f) => f(env)),
        );

        if (r === null) {
          return null;
        }

        return negated ? !r : r;
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
    case 'valueSubquery':
      return compileValueSubquery(expr);
    case 'letIn':
      return compileLetIn(expr);
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
  const nbrs = (v: Vertex): Vertex[] => outNeighbors(graph, v, out, types ?? undefined);
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
 * ISO `VALUE { … RETURN e }`: a scalar (single-value) correlated subquery.
 * Collect every correlated match, then: an aggregate RETURN folds the whole
 * group to one value (0 rows → the aggregate's empty answer); a non-aggregate
 * RETURN yields NULL for 0 rows, the value for exactly one, and a **cardinality
 * error** for more than one (ISO: a scalar subquery must not deliver >1 row) —
 * loud, never a silent first-of-many. Mirrors the native `value_subquery`.
 */
const compileValueSubquery = (expr: Extract<Expr, { kind: 'valueSubquery' }>): CompiledExpr => {
  const sub = compileSubMatch(expr);
  const retFn = compileExpr(expr.ret);
  const isAgg = hasAggregate(expr.ret);

  return (env) => {
    const matches = [...matchClauseBindings(env.graph, sub, env.binding, env.params)];

    if (isAgg) {
      // Fold over the group: `count(*)` reads its length, other aggregates read
      // `env.group`; a plain sub-expression reads the first match (or the outer
      // binding when there were none).
      const base = matches[0] ?? env.binding;

      return retFn({ ...env, binding: base, group: matches });
    }

    if (matches.length === 0) {
      return null;
    }

    if (matches.length > 1) {
      dataException(
        'a VALUE scalar subquery returned more than one row; add an aggregate ' +
          '(e.g. count/collect), a LIMIT-like bound, or a more selective pattern',
      );
    }

    return retFn({ ...env, binding: matches[0] });
  };
};

/**
 * ISO `<let value expression>`: `LET x = e, … IN body END`. Binds each local into
 * a fresh extended binding (left-to-right, so a later binding sees earlier ones),
 * then evaluates the body against it. The group / aggregate context is preserved
 * so an aggregate binding folds over the same group and the body reads the
 * resulting scalar. Mirrors the native `CExpr::LetIn`.
 */
const compileLetIn = (expr: Extract<Expr, { kind: 'letIn' }>): CompiledExpr => {
  const bindings = expr.bindings.map((b) => ({ name: b.var, fn: compileExpr(b.expr) }));
  const bodyFn = compileExpr(expr.body);

  return (env) => {
    const local = new Map(env.binding);
    const scoped: EvalEnv = { ...env, binding: local };

    for (const { name, fn } of bindings) {
      local.set(name, fn(scoped));
    }

    return bodyFn(scoped);
  };
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
    // DISTINCT dedups STRUCTURALLY by valueKey (mirroring Rust val_key / dense id),
    // not by reference-identity `new Set` — which would keep value-equal but distinct
    // object instances (temporals, lists, records) separate. e.g. count(DISTINCT
    // DATE'2020-01-01') over two rows must be 1, and count(DISTINCT [x]) dedups too.
    const values = distinct ? distinctByValueKey(nonNull) : nonNull;

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
    // The inner source `(x)` and target `(y)` group variables of a subpath, plus any
    // intermediate nodes / edges of a MULTI-element repetition unit.
    if (seg.hopFrom !== undefined) {
      addNode(seg.hopFrom);
    }

    if (seg.hopTo !== undefined) {
      addNode(seg.hopTo);
    }

    for (const extra of seg.unitRest ?? []) {
      if (extra.rel.variable !== undefined) {
        into.add(extra.rel.variable);
      }

      addNode(extra.node);
    }

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

  // `EXISTS { … }` / `COUNT { … }`: the sub-pattern binds its own variables, so
  // extend the bound set before descending into its inline predicates/WHERE —
  // those bindings aren't free references, but outer names still are. Extracted
  // from `walk` to keep that switch under the complexity budget.
  const walkSubquery = (
    e: Extract<Expr, { kind: 'exists' | 'countSubquery' | 'valueSubquery' }>,
    bound: ReadonlySet<string>,
  ): void => {
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

    // A VALUE subquery's RETURN expression also reads the subquery's own bindings.
    if (e.kind === 'valueSubquery') {
      walk(e.ret, inner);
    }
  };

  // A CASE expression's operands: subject (simple CASE), every WHEN/THEN, and
  // ELSE. Extracted from `walk` to keep that switch under the complexity budget.
  const walkCase = (e: Extract<Expr, { kind: 'case' }>, bound: ReadonlySet<string>): void => {
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
  };

  // `LET x = e, … IN body END`: each binding may reference outer vars and prior
  // LET locals; the local it introduces then shadows for later bindings and the
  // body. Extracted from `walk` to keep that switch under the complexity budget.
  const walkLetIn = (e: Extract<Expr, { kind: 'letIn' }>, bound: ReadonlySet<string>): void => {
    const inner = new Set(bound);

    for (const b of e.bindings) {
      walk(b.expr, inner);
      inner.add(b.var);
    }

    walk(e.body, inner);
  };

  const walkAll = (exprs: Iterable<Expr>, bound: ReadonlySet<string>): void => {
    for (const x of exprs) {
      walk(x, bound);
    }
  };

  const walk = (e: Expr, bound: ReadonlySet<string>): void => {
    switch (e.kind) {
      case 'var':
        if (!bound.has(e.name)) {
          free.add(e.name);
        }

        return;
      case 'property_exists':
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
      case 'isTyped':
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
        walkAll(e.items, bound);

        return;
      case 'in':
        walk(e.expr, bound);
        walk(e.list, bound);

        return;
      case 'record':
        walkAll(
          e.fields.map((f) => f.value),
          bound,
        );

        return;
      case 'case':
        walkCase(e, bound);

        return;
      case 'func':
      case 'graphPred':
        walkAll(e.args, bound);

        return;
      case 'exists':
      case 'countSubquery':
      case 'valueSubquery':
        walkSubquery(e, bound);

        return;
      case 'letIn':
        walkLetIn(e, bound);

        return;
    }
  };

  walk(expr, new Set());

  return free;
};

// --- value / ordering helpers ------------------------------------------------

/** Derive a column name for a RETURN item that has no explicit `AS` alias. */
export const columnName = (expr: Expr): string => {
  switch (expr.kind) {
    case 'var':
      return expr.name;
    case 'prop':
      return `${expr.variable}.${expr.key}`;
    case 'property_exists':
      return `property_exists(${expr.variable}, ${expr.key})`;
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

  // Records: keys are canonical (sorted), so compare field-by-field (key then
  // value), shorter-is-less — a total order for ORDER BY/DISTINCT even though ISO
  // defines no relational `<`/`>` on records. Mirrors the Rust `cmp_total`.
  if (a instanceof LenkeRecord && b instanceof LenkeRecord) {
    const ae = [...a];
    const be = [...b];
    const n = Math.min(ae.length, be.length);

    for (let i = 0; i < n; i++) {
      const kc = cmpKey(ae[i][0], be[i][0]);

      if (kc !== 0) {
        return kc;
      }

      const vc = compareValues(ae[i][1], be[i][1]);

      if (vc !== 0) {
        return vc;
      }
    }

    if (ae.length < be.length) {
      return -1;
    }

    return ae.length > be.length ? 1 : 0;
  }

  const x = a as number | string;
  const y = b as number | string;

  // Strings compare by Unicode code point (matching Rust str::cmp), not JS's
  // UTF-16 code-unit order — see compareCodePoints.
  if (typeof x === 'string' && typeof y === 'string') {
    return compareCodePoints(x, y);
  }

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
export const compareSort = (
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

// The replacer used only for nested containers (below): `JSON.stringify` maps
// NaN/±Infinity all to `null`, collapsing them with each other AND with real null —
// a byte-identity break vs the Rust `val_key`, which keys numbers by bit pattern
// (so `[NaN]`, `[Infinity]`, `[null]` are three groups). This tags non-finite
// numbers as bare `#…` tokens and prefixes every string with `$`, keeping the two
// key-spaces disjoint at any depth — a real string `"#Infinity"` (→ `"$#Infinity"`)
// can never collide with the number `Infinity` (→ `"#Infinity"`).
const keyReplacer = (_k: string, val: unknown): unknown => {
  if (typeof val === 'number') {
    return Number.isFinite(val) ? val : `#${val}`;
  }

  return typeof val === 'string' ? `$${val}` : val;
};

/**
 * Stable distinct/group key for a projected value, partitioning exactly as the
 * Rust `val_key`. Primitives and graph elements are keyed inline (the common
 * DISTINCT/GROUP BY case — no `JSON.stringify`, so it's off the hot path); only a
 * nested container (list/record/temporal/path — rare as a group key) pays the
 * `JSON.stringify` + `keyReplacer` cost, which is what makes nested non-finite safe.
 * Type-tag prefixes (`n`/`s`/`b`/`@`/`#`/`N`) keep the primitive key-spaces disjoint
 * from each other and from container JSON (which starts with `{`/`[`).
 */
export const valueKey = (v: unknown): string => {
  switch (typeof v) {
    case 'number':
      return Number.isFinite(v) ? `n${v}` : `#${v}`;
    case 'string':
      return `s${v}`;
    case 'boolean':
      return v ? 'bT' : 'bF';
    case 'object':
      if (v === null) {
        return 'N';
      }

      if ('id' in v) {
        return `@${String((v as { id: unknown }).id)}`;
      }

      // Lists are native arrays (the common container key): recurse through the
      // inline keyer — no JSON.stringify, and nested non-finite is handled by the
      // element's own `#…` tag. `[` can't collide with a record's `{` (below) or
      // any primitive prefix. A manual loop (no `.map` intermediate) keeps this on
      // the hot path for large DISTINCT/GROUP BY over projected tuples.
      if (Array.isArray(v)) {
        let key = '[';

        for (const item of v) {
          key += `${valueKey(item)},`;
        }

        return `${key}]`;
      }

      // Record / temporal / path — rarer as a group key; JSON.stringify + the
      // replacer keeps their nested non-finite safe too.
      return JSON.stringify(v, keyReplacer) ?? 'undefined';
    default:
      // undefined / bigint / symbol are outside the value model; a stable fallback.
      return `j${String(v)}`;
  }
};
const rowKey = (b: Binding): string => [...b].map(([k, v]) => `${k}=${valueKey(v)}`).join('');

/** Keep the first occurrence of each distinct value, keyed structurally by
 *  `valueKey` (so value-equal temporals/lists/records collapse) — the DISTINCT
 *  aggregate dedup, matching the Rust engine's `val_key` / dense-id partition. */
const distinctByValueKey = (values: readonly unknown[]): unknown[] => {
  const seen = new Set<string>();
  const out: unknown[] = [];

  for (const v of values) {
    const k = valueKey(v);

    if (!seen.has(k)) {
      seen.add(k);
      out.push(v);
    }
  }

  return out;
};

// --- projection compilation --------------------------------------------------

/** A projected output column: its name and the closure producing its value. */
export type CReturnItem = { name: string; fn: CompiledExpr; isAgg: boolean };
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
  /** ISO HAVING — a post-aggregation predicate on each group (SELECT only). */
  having?: CompiledExpr;
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
  noteCountParam(projection.skip);
  noteCountParam(projection.limit);
  // Explicit GROUP BY keys DRIVE grouping (and force it on, even without an
  // aggregate); absent → implicit grouping by the non-aggregate items.
  const groupByExprs = projection.groupBy ?? [];
  const having = projection.having ? compileExpr(projection.having) : undefined;
  const aggregating =
    !projection.star &&
    (items.some((i) => i.isAgg) || groupByExprs.length > 0 || having !== undefined);
  const groupKeys =
    groupByExprs.length > 0
      ? groupByExprs.map((e) => compileExpr(e))
      : items.filter((i) => !i.isAgg).map((i) => i.fn);
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
    ...(having ? { having } : {}),
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
export const applyProjection = (
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

    // ISO HAVING: drop a group whose post-aggregation predicate is not exactly
    // TRUE (three-valued — NULL/false both drop). The aggregates fold over the
    // group array; group keys read the representative binding.
    const { having } = proj;
    const groupList = having
      ? [...groups.values()].filter((group) => {
          const rep = group[0] ?? new Map();

          return asTruth(having({ binding: rep, params, graph, group })) === true;
        })
      : groups.values();

    keyed = map((group: Binding[]) => {
      const rep: Binding = group[0] ?? new Map();
      const projected = projectBinding(proj, rep, params, graph, group);
      // ORDER BY sees the output columns overlaid on the input variables.
      const sortBinding = orderBy.length > 0 ? new Map([...rep, ...projected]) : rep;

      return {
        b: projected,
        keys: orderBy.map((s) => s.fn({ binding: sortBinding, params, graph, group })),
      };
    }, groupList);
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
export type CProp = { key: string; value: CompiledExpr };
type CPredicate = { props: readonly CProp[]; where?: CompiledExpr };

/** Range bounds whose endpoints are compiled value closures (resolved per seed). */
export type CRangeBound = {
  gt?: CompiledExpr;
  gte?: CompiledExpr;
  lt?: CompiledExpr;
  lte?: CompiledExpr;
};

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

export type CNode = {
  variable?: string;
  label?: LabelExpr;
  pred: CPredicate;
  seedHints?: readonly CSeedHint[];
};
export type CRel = {
  variable?: string;
  label?: LabelExpr;
  direction: RelPattern['direction'];
  pred: CPredicate;
  quantifier?: RelPattern['quantifier'];
};
/** One hop of a repetition unit: traverse `rel`, land on `targetVar` (a group var). */
type CHop = { rel: CRel; targetVar?: string };
/** A nested quantified sub-unit — the general `( … ){a,b}` nesting. `max` is `null`
 *  for an unbounded `*`/`+`/`{n,}`. `targetVar` is the sub-unit's LANDING group var
 *  (`y` in `-[]->{a,b}(y)`), bound once per enclosing rep. Mirrors native `CSub`. */
type CSub = { unit: CUnit; min: number; max: number | null; targetVar?: string };
/** One element of a unit: a single hop, or a nested quantified sub-unit. Mirrors
 *  native `CElem`. */
type CElem = { hop: CHop } | { sub: CSub };
/**
 * A quantified parenthesized subpath compiled to a repetition UNIT: a linear element
 * sequence repeated `[min, max]` times. `startVar` is the source `(x)` group var; a
 * hop's `targetVar` is an intermediate/target group var; `where` is the per-repetition
 * predicate. Elements may be nested quantified sub-units. Mirrors native `CUnit`.
 */
export type CUnit = { elems: readonly CElem[]; startVar?: string; where?: CompiledExpr };
export type CSegment = { rel: CRel; node: CNode; unit?: CUnit };

/** Whether a unit binds any GROUP variable (source, an edge, a target, a `Sub`'s
 *  landing, or anything a nested sub-unit binds). Mirrors native `CUnit::exposes`. */
export const unitExposes = (u: CUnit): boolean =>
  u.startVar !== undefined ||
  u.elems.some((e) =>
    'hop' in e
      ? e.hop.targetVar !== undefined || e.hop.rel.variable !== undefined
      : e.sub.targetVar !== undefined || unitExposes(e.sub.unit),
  );

/** Whether every element is a plain hop (no nested `Sub`) — a flat unit binds group
 *  variables by the cheap `k`-stride over the flat walk, never per-hop steps. Mirrors
 *  native `CUnit::is_flat`. */
export const unitIsFlat = (u: CUnit): boolean => u.elems.every((e) => 'hop' in e);
export type CPath = {
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

export const relHasPredicate = (rel: RelPattern): boolean =>
  (rel.properties?.length ?? 0) > 0 || rel.where !== undefined;

/** Build one repetition unit from a quantified-subpath segment. A NESTED parenthesized
 *  subpath (`( ((x)-[e]->(y)){a,b} ){n,m}`) recurses: the outer unit's sole element is a
 *  `Sub` wrapping the inner subpath's unit, so its variables nest one list level deeper.
 *  Otherwise it's the inner hop chain. Mirrors native `plan::Lowerer::subpath_unit`. */
const compileSubpathUnit = (seg: Segment): CUnit => {
  if (seg.nested !== undefined) {
    const inner = seg.nested;
    const q = inner.rel.quantifier!; // a nested subpath is always quantified

    // The nested subpath's LANDING is the outer segment's endpoint node, matched
    // separately — not a group variable of this unit (so no `targetVar` on the `Sub`). The
    // outer subpath-level WHERE (per outer rep, inner vars bound as lists) is the unit WHERE.
    return {
      elems: [{ sub: { unit: compileSubpathUnit(inner), min: q.min, max: q.max ?? null } }],
      ...(seg.subpathWhere !== undefined ? { where: compileExpr(seg.subpathWhere) } : {}),
    };
  }

  const { rel, hopFrom, hopTo, unitRest, innerQ } = seg;
  // Each inner hop is a plain hop, OR a nested single-edge `Sub` when it carries its own
  // quantifier (`-[e]->{a,b}`). The first hop's nested quantifier is `innerQ`; later hops
  // carry theirs on `rel.quantifier`.
  const astElems: { rel: RelPattern; targetVar?: string; q?: Quantifier }[] = [
    {
      rel,
      ...(hopTo?.variable !== undefined ? { targetVar: hopTo.variable } : {}),
      ...(innerQ !== undefined ? { q: innerQ } : {}),
    },
    ...(unitRest ?? []).map((extra) => ({
      rel: extra.rel,
      ...(extra.node.variable !== undefined ? { targetVar: extra.node.variable } : {}),
      ...(extra.rel.quantifier !== undefined ? { q: extra.rel.quantifier } : {}),
    })),
  ];

  // The subpath-level WHERE is the per-repetition predicate; a PLAIN hop's inline WHERE is
  // also lifted to the unit level; a NESTED hop's WHERE (`-[e WHERE …]->{a,b}`) is a
  // per-inner-edge predicate that STAYS on the `Sub`'s inner hop (not lifted).
  const whereExprs = [
    seg.subpathWhere,
    ...astElems.filter((h) => h.q === undefined).map((h) => h.rel.where),
  ].filter((w): w is Expr => w !== undefined);
  let unitWhere: Expr | undefined;

  if (whereExprs.length === 1) {
    [unitWhere] = whereExprs;
  } else if (whereExprs.length > 1) {
    unitWhere = { kind: 'and', items: whereExprs };
  }

  const hopOrSub = (h: { rel: RelPattern; targetVar?: string; q?: Quantifier }): CElem => {
    if (h.q === undefined) {
      // Plain hop: its WHERE was lifted, so strip it from the hop predicate.
      const chop: CHop = {
        rel: compileRel({ ...h.rel, where: undefined, quantifier: undefined }),
        ...(h.targetVar !== undefined ? { targetVar: h.targetVar } : {}),
      };

      return { hop: chop };
    }

    // Nested `-[]->{a,b}(y)` hop: the landing `y` is the WHOLE sub-unit's target (bound
    // once per enclosing rep — `CSub.targetVar`); the inner hop's own target is anonymous.
    // The inner hop keeps its WHERE (per-inner-edge predicate).
    const innerHop: CHop = { rel: compileRel({ ...h.rel, quantifier: undefined }) };

    return {
      sub: {
        unit: { elems: [{ hop: innerHop }] },
        min: h.q.min,
        max: h.q.max ?? null,
        ...(h.targetVar !== undefined ? { targetVar: h.targetVar } : {}),
      },
    };
  };

  return {
    elems: astElems.map(hopOrSub),
    ...(hopFrom?.variable !== undefined ? { startVar: hopFrom.variable } : {}),
    ...(unitWhere !== undefined ? { where: compileExpr(unitWhere) } : {}),
  };
};

const compilePath = (pattern: PathPattern): CPath => {
  const selector = pattern.selector ?? 'walk';

  return {
    start: compileNode(pattern.start),
    segments: pattern.segments.map((seg) => {
      const crel = compileRel(seg.rel);

      // A plain / abbreviated hop — no repetition unit.
      if (seg.hopFrom === undefined && seg.nested === undefined) {
        return { rel: crel, node: compileNode(seg.node) };
      }

      // A quantified parenthesized subpath compiles to a UNIT; `node` is the separate
      // outer endpoint. Mirrors native `plan::segment`.
      return { rel: crel, node: compileNode(seg.node), unit: compileSubpathUnit(seg) };
    }),
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
export const satisfies = (
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

// Pattern matching (see executor/matching.ts).
import { matchNode } from './executor/matching.js';

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
export type CSetItem =
  | { variable: string; label: string }
  | { variable: string; key: string; value: CompiledExpr };

/** A compiled INSERT node/rel: labels are fixed, property values are closures. */
export type CInsertNode = { variable?: string; labels: readonly string[]; props: readonly CProp[] };
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

export type CMatch = {
  kind: 'match';
  optional: boolean;
  patterns: readonly CPath[];
  where?: CompiledExpr;
  nullVars: readonly string[];
};
type CWith = { kind: 'with'; projection: CProjection; where?: CompiledExpr };
type CFilter = { kind: 'filter'; where: CompiledExpr };
type CLet = { kind: 'let'; items: readonly { var: string; expr: CompiledExpr }[] };
export type CFor = {
  kind: 'for';
  list: CompiledExpr;
  alias: string;
  ordinality?: { kind: 'ordinality' | 'offset'; var: string };
};
type CReturn = { kind: 'return'; projection: CProjection };
export type CInsert = { kind: 'insert'; patterns: readonly CInsertPath[] };
type CMergeUpdate =
  | { kind: 'set'; items: readonly CSetItem[]; where?: CompiledExpr }
  | { kind: 'nothing' };
export type CMerge = {
  kind: 'merge';
  pattern: CInsertPath;
  onCreate?: readonly CSetItem[];
  onUpdate?: CMergeUpdate;
};
export type CSet = { kind: 'set'; items: readonly CSetItem[] };
export type CRemove = { kind: 'remove'; items: readonly RemoveItem[] };
export type CDelete = { kind: 'delete'; detach: boolean; targets: readonly CompiledExpr[] };
type CFinish = { kind: 'finish' };
export type CCallNamed = {
  kind: 'callNamed';
  optional: boolean;
  procName: string;
  /** Resolved algorithm dispatch name; `null` = unknown procedure (faults). */
  algo: AlgorithmName | null;
  config: readonly { key: string; value: CompiledExpr }[];
  /** Procedure output column → the variable it yields into. */
  binds: readonly { column: string; var: string }[];
};
export type CCallInline = {
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
export type CClause =
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

// --- shared bucket/adjacency primitives --------------------------------------
// Type-bucket sizing + neighbour expansion used by the general matcher and by the
// count/reachability fast-paths in ./executor/shortcuts.ts. Exported so the
// shortcut detectors can't drift from the trunk's expansion of the same edges.

/** Edge types a rel label admits: `null` = bail (and/not/wildcard); `undefined`
 * = "any type" (no `:T`); `string[]` = exactly those types. */
export const relTypeNames = (label: LabelExpr | undefined): string[] | null | undefined => {
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

/** The per-type `Set<Edge>` buckets for `types` (undefined = every type). */
export const bucketsFor = (
  byType: Map<string, Set<Edge>> | undefined,
  types: string[] | undefined,
): (Set<Edge> | undefined)[] => {
  if (!byType) {
    return [];
  }

  return types ? types.map((t) => byType.get(t)) : [...byType.values()];
};

/** One-hop neighbours of `v` along `out` (or in) edges of the given `types` (all
 * types when undefined). Shared by the EXISTS-reachable and RETURN-reachable
 * traversals so their neighbour expansion can't drift from each other or native. */
export const outNeighbors = (
  graph: Graph,
  v: Vertex,
  out: boolean,
  types: string[] | undefined,
): Vertex[] => {
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

/** Count edges across `buckets` that pass `keep`. */
export const countEdges = (
  buckets: (Set<Edge> | undefined)[],
  keep: (e: Edge) => boolean,
): number => {
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

export type CLinear = {
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

export const isEdge = (v: unknown): v is Edge =>
  typeof v === 'object' && v !== null && 'from' in v && 'to' in v;
export const isElement = (v: unknown): v is Vertex | Edge =>
  typeof v === 'object' && v !== null && 'id' in v;
export const isVertex = (v: unknown): v is Vertex => isElement(v) && !isEdge(v);

// Statement execution: writes, clause processing, set ops (see executor/clauses.ts).
import {
  combineRows,
  matchClauseBindings,
  procedureSpec,
  queryHasWrite,
  runLinear,
  runTxControl,
} from './executor/clauses.js';
// Count / reachability fast-paths (see executor/shortcuts.ts) — this back-edge and
// the shortcuts' import of the trunk's bucket primitives form a safe function-level
// cycle, matching the other executor submodules.
import { detectCountShortcut, detectReachableShortcut } from './executor/shortcuts.js';
import type { ReachFn } from './executor/shortcuts.js';

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
