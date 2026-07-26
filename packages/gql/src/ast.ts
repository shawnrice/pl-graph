/**
 * GQL query AST. Plain data only (no closures), mirroring the gremlin package's
 * discipline: a parsed query is a serializable description of *what* to match,
 * never *how* to walk it. The executor decides traversal order.
 *
 * The surface is the ISO GQL (ISO/IEC 39075) core: `MATCH` graph patterns, an
 * optional `WHERE` filter, and a `RETURN` projection. Pattern syntax is the ISO
 * ASCII-art form: `(a:Person)-[r:KNOWS]->(b)`. Note this follows ISO, not
 * Cypher — `--` is a line comment, and undirected edges use `~`.
 */

/**
 * A whole query: one or more linear queries combined by set operators
 * (`p0 UNION p1 EXCEPT p2 …`, left-associative). `ops[i]` joins `parts[i]` to
 * `parts[i + 1]`, so `ops.length === parts.length - 1`.
 */
export type Query = {
  parts: readonly LinearQuery[];
  ops: readonly SetOp[];
};

/**
 * ISO/IEC 39075 transaction-control command — a complete top-level statement that
 * is NOT a linear query: `START TRANSACTION [READ ONLY | READ WRITE]`,
 * `COMMIT [WORK]`, `ROLLBACK [WORK]`. It drives the session's transaction frame
 * (`Graph.beginTransaction`/`commitTransaction`/`rollbackTransaction`) rather than
 * matching a pattern, so it carries no clauses. `accessMode` is only meaningful
 * for `start` (default `read write` when omitted).
 */
export type TxControl = {
  kind: 'start' | 'commit' | 'rollback';
  accessMode?: 'read only' | 'read write';
};

/**
 * A parsed top-level statement: either a linear (pattern) query or a
 * transaction-control command. `parse()` returns this union; a `TxControl` is
 * discriminated by its `kind` field (a `Query` has none).
 */
export type Statement = Query | TxControl;

/** Narrow a parsed statement to a {@link TxControl} (vs a {@link Query}). */
export const isTxControl = (stmt: Statement): stmt is TxControl => 'kind' in stmt;

/** `UNION` / `EXCEPT` / `INTERSECT`, optionally `ALL` (keep duplicates). */
export type SetOp = {
  op: 'union' | 'except' | 'intersect';
  all: boolean;
};

/**
 * A linear query: a sequence of clauses ending in `RETURN`. `WITH` projects and
 * chains intermediate results; `OPTIONAL MATCH` keeps rows that don't match.
 */
export type LinearQuery = {
  clauses: readonly Clause[];
};

/** Parse dialect: `lenke` permits sigil extensions (`_MERGE`); `iso-strict` rejects them. */
export type Dialect = 'lenke' | 'iso-strict';

export type Clause =
  | MatchClause
  | WithClause
  | FilterClause
  | LetClause
  | ForClause
  | InsertClause
  | MergeClause
  | SetClause
  | RemoveClause
  | DeleteClause
  | FinishClause
  | CallNamedClause
  | CallInlineClause
  | ReturnClause;

/**
 * `[OPTIONAL] CALL (scope) { … }` — an ISO GQL inline procedure call
 * (`callProcedureStatement` → `inlineProcedureCall`). Runs the nested query once
 * per incoming row (correlated / lateral), importing only the `scope` variables,
 * and merges the nested `RETURN` columns back into the row. OPTIONAL keeps the
 * outer row (nested columns null-filled) when the subquery yields nothing.
 */
export type CallInlineClause = {
  kind: 'callInline';
  optional: boolean;
  /** Outer variables the subquery imports (`(a, b)`); empty = none / `()`. */
  scope: readonly string[];
  /**
   * The nested query: one or more linear parts joined by set operators
   * (`UNION`/`EXCEPT`/`INTERSECT`). A plain body is a single part, no ops.
   */
  body: Query;
};

/**
 * `[OPTIONAL] CALL name(config) [YIELD col [AS alias], …]` — an ISO GQL named
 * procedure call (`callProcedureStatement` → `namedProcedureCall`). Invokes a
 * catalog procedure (here: the built-in graph algorithms); `yields` picks and
 * renames its output columns (absent ⇒ every column, under its own name).
 */
export type CallNamedClause = {
  kind: 'callNamed';
  /** `OPTIONAL CALL` — keep the outer row (null-filled) if the call is empty. */
  optional: boolean;
  /** Procedure name (a dotted `parent.name` is joined with `.`). */
  name: string;
  /** The procedure's config, written as a `{key: value}` map argument. */
  config: readonly PropertyConstraint[];
  yields?: readonly YieldItem[];
};

/** One `YIELD` output item: an output column, optionally renamed with `AS`. */
export type YieldItem = { name: string; alias?: string };

/**
 * `_MERGE pattern [_ON_CREATE SET …] [_ON_UPDATE SET … [WHERE p] | _ON_UPDATE_NOTHING]`
 * — the lenke keyed-upsert **extension** (NOT ISO GQL; sigil-marked, recognized
 * only under the `lenke` dialect). v1: `pattern` upserts a single element — a
 * node, or an edge whose endpoints are matched by key. The conflict key is a
 * declared unique constraint. Absent `onUpdate` = clobber the pattern's payload.
 * See docs/design/gql-extensions.md §2.
 */
export type MergeClause = {
  kind: 'merge';
  pattern: PathPattern;
  /** `_ON_CREATE SET …` — birth-only extras. */
  onCreate?: readonly SetItem[];
  /** The update-path disposition; absent ⇒ default clobber of the payload. */
  onUpdate?: MergeUpdate;
};

export type MergeUpdate =
  /** `_ON_UPDATE SET … [WHERE p]` — replaces the default; runs only if `where` holds. */
  | { kind: 'set'; items: readonly SetItem[]; where?: Expr }
  /** `_ON_UPDATE_NOTHING` — leave the existing element untouched. */
  | { kind: 'nothing' };

/** `INSERT pattern, …` — create the pattern's nodes and edges. */
export type InsertClause = {
  kind: 'insert';
  patterns: readonly PathPattern[];
};

/** `SET n.key = value` (property) or `SET n:Label` (label). */
export type SetClause = {
  kind: 'set';
  items: readonly SetItem[];
};
export type SetItem =
  | { variable: string; key: string; value: Expr }
  | { variable: string; label: string };

/** `REMOVE n.key` (property) or `REMOVE n:Label` (label). */
export type RemoveClause = {
  kind: 'remove';
  items: readonly RemoveItem[];
};
export type RemoveItem = { variable: string; key: string } | { variable: string; label: string };

/** `[DETACH] DELETE n, …`. */
export type DeleteClause = {
  kind: 'delete';
  detach: boolean;
  targets: readonly Expr[];
};

/** `FINISH` — run for side effects, return nothing. */
export type FinishClause = { kind: 'finish' };

/** `[OPTIONAL] MATCH p1, p2, … [WHERE pred]`. */
export type MatchClause = {
  kind: 'match';
  optional: boolean;
  patterns: readonly PathPattern[];
  where?: Expr;
};

/**
 * `FOR x IN <list> [WITH ORDINALITY|OFFSET n]` — ISO/IEC 39075 list unwind (the
 * standard's equivalent of Cypher `UNWIND`). Multiplies the row table: one row
 * per element of the evaluated list. Core ISO GQL (no sigil), every dialect.
 * `list` is evaluated in the scope before `alias` is bound. `ordinality`, when
 * present, binds a counter — 1-based for `ordinality`, 0-based for `offset`.
 */
export type ForClause = {
  kind: 'for';
  alias: string;
  list: Expr;
  ordinality?: { kind: 'ordinality' | 'offset'; var: string };
};

/** `WITH … [WHERE pred]` — a projection that flows into the next clause. */
export type WithClause = {
  kind: 'with';
  projection: Projection;
  where?: Expr;
};

/**
 * `FILTER [WHERE] <condition>` — ISO GQL §14.6. Drops rows where the condition
 * is not TRUE (three-valued, like `WHERE`). The `WHERE` keyword is optional.
 */
export type FilterClause = {
  kind: 'filter';
  where: Expr;
};

/**
 * `LET x = e, y = e, …` — ISO GQL §14.7. Binds new value variables into scope
 * (additive). Items evaluate left-to-right, so a later one may reference an
 * earlier (`LET x = 1, y = x + 1`).
 */
export type LetClause = {
  kind: 'let';
  items: readonly { var: string; expr: Expr }[];
};

/** `RETURN …` — the final projection producing result rows. */
export type ReturnClause = {
  kind: 'return';
  projection: Projection;
};

/**
 * A linear path pattern: a starting node followed by zero or more
 * `(relationship)(node)` segments. `(a)-[:KNOWS]->(b)-[:KNOWS]->(c)` is one
 * PathPattern with two segments.
 */
export type PathPattern = {
  start: NodePattern;
  segments: readonly Segment[];
  /** The whole path bound to this variable (`p = …`), or unnamed. */
  pathVar?: string;
  /** Which matching paths to keep; defaults to `walk` (every match). */
  selector?: PathSelector;
  /** The repeated-element restrictor on a var-length walk; defaults to `trail`. */
  mode?: PathMode;
};

/**
 * How many of the paths matching a pattern to keep. `walk` (the ISO default, no
 * selector, and the target of bare `ALL`) keeps every match; `any` (bare `ANY`)
 * keeps one arbitrary path per endpoint; `anyShortest` (`ANY SHORTEST`) keeps one
 * fewest-hop path per endpoint pair; `allShortest` (`ALL SHORTEST`) keeps every
 * path tied for that fewest-hop length; `{ kind: 'shortestK', k, group }`
 * (`SHORTEST k [GROUP]`) keeps the k shortest paths per endpoint — or, with
 * `group`, every path in the k smallest length-groups.
 */
export type PathSelector =
  | 'walk'
  | 'any'
  | 'anyShortest'
  | 'allShortest'
  | { readonly kind: 'shortestK'; readonly k: number; readonly group: boolean };

/**
 * A path mode (restrictor): which element repeats a matched path may contain.
 * `trail` (no repeated EDGES) is lenke's default — the spec's nominal `walk`
 * default is unusable unbounded, so, like Neo4j/Fabric, we default to `trail`.
 * `walk` allows repeats (bounded only by the quantifier); `acyclic` forbids any
 * repeated NODE; `simple` forbids repeated nodes except a closing `start == end`.
 */
export type PathMode = 'walk' | 'trail' | 'simple' | 'acyclic';

/** One hop: traverse `rel`, land on `node`. */
export type Segment = {
  rel: RelPattern;
  node: NodePattern;
};

/**
 * `(variable:LabelExpr {props} WHERE pred)` — all parts optional. A missing
 * `label` matches any node, including unlabelled ones. `properties` and `where`
 * are the ISO element-pattern predicate (§16.5): a property map and/or an
 * inline filter that the matched element must satisfy.
 */
export type NodePattern = {
  variable?: string;
  label?: LabelExpr;
  properties?: readonly PropertyConstraint[];
  where?: Expr;
};

/** One `key: valueExpression` entry of a pattern property map. */
export type PropertyConstraint = {
  key: string;
  value: Expr;
};

/**
 * An ISO label expression (ISO/IEC 39075 §16.7). This is the boolean-algebra
 * form — `A&B` (conjunction), `A|B` (disjunction), `!A` (negation), `%`
 * (wildcard) — *not* Cypher's colon-chained `:A:B`.
 */
export type LabelExpr =
  | { kind: 'label'; name: string }
  | { kind: 'wildcard' }
  | { kind: 'not'; expr: LabelExpr }
  | { kind: 'and'; left: LabelExpr; right: LabelExpr }
  | { kind: 'or'; left: LabelExpr; right: LabelExpr };

/**
 * A relationship pattern with a direction.
 *  - `out`:  `-[...]->`  or abbreviated `->`
 *  - `in`:   `<-[...]-`  or abbreviated `<-`
 *  - `both`: `~[...]~`, `-[...]-`, `<-[...]->`  or abbreviated `~` / `<->`
 *
 * `label` is an ISO label expression over edge types (`KNOWS|CREATED`,
 * `!IGNORED`, …); a missing `label` matches any type.
 */
export type RelPattern = {
  variable?: string;
  label?: LabelExpr;
  direction: 'out' | 'in' | 'both';
  properties?: readonly PropertyConstraint[];
  where?: Expr;
  /** Variable-length quantifier: `*`={0,∞}, `+`={1,∞}, `{n}`, `{n,m}`. */
  quantifier?: { min: number; max: number | null };
};

/**
 * A projection body, shared by `RETURN` and `WITH`:
 * `DISTINCT (* | item, …) ORDER BY … SKIP n LIMIT n`. `star` carries all bound
 * variables forward; otherwise `items` are projected.
 */
export type Projection = {
  star: boolean;
  items: readonly ReturnItem[];
  distinct: boolean;
  orderBy?: readonly SortItem[];
  skip?: CountValue;
  limit?: CountValue;
};

/**
 * A `LIMIT` / `OFFSET` bound. ISO GQL's `nonNegativeIntegerSpecification`
 * (opengql:2268) is `unsignedInteger | dynamicParameterSpecification`, so the
 * bound is either an integer literal or a `$param` resolved (and validated to be
 * a non-negative integer) at execution. `SKIP` is the Cypher spelling of
 * `OFFSET` and accepts only a literal — a `$param` after `SKIP` is rejected.
 */
export type CountValue = number | { param: string };

/** One `ORDER BY` key, with optional ISO `NULLS FIRST` / `NULLS LAST`. */
export type SortItem = {
  expr: Expr;
  descending: boolean;
  /** `true` = NULLS FIRST, `false` = NULLS LAST, `undefined` = engine default. */
  nullsFirst?: boolean;
};

/** A single RETURN expression with an optional `AS` alias. */
export type ReturnItem = {
  expr: Expr;
  alias?: string;
};

/** Comparison operators usable in WHERE and RETURN expressions. */
export type CompareOp = '=' | '<>' | '<' | '>' | '<=' | '>=';

/**
 * Expression tree. Unlike gremlin's `Predicate` (which compares one value
 * against a constant), a GQL expression compares two arbitrary sub-expressions,
 * so `a.age > b.age` is expressible.
 */
export type ArithOp = '+' | '-' | '*' | '/' | '%';

export type Expr =
  | { kind: 'var'; name: string }
  | { kind: 'param'; name: string }
  | { kind: 'prop'; variable: string; key: string }
  // `PROPERTY_EXISTS(n, key)` — true iff `key` is *present* on element `n`,
  // regardless of value (distinguishes an absent key from a stored null). The
  // second argument is a bare property name, not an expression.
  | { kind: 'property_exists'; variable: string; key: string }
  | { kind: 'lit'; value: unknown }
  | { kind: 'list'; items: readonly Expr[] }
  // ISO GQL list element access `base[index]` — 0-based; out of range → null.
  | { kind: 'index'; base: Expr; index: Expr }
  // Property access chained off any expression (`edges(p)[0].amount`).
  // A bare `variable.key` stays `prop` (the hot path); this is the postfix `.key`.
  | { kind: 'field'; base: Expr; key: string }
  | { kind: 'compare'; op: CompareOp; left: Expr; right: Expr }
  // n-ary left-associative operator runs: a long chain is a flat array, not a
  // chain-deep binary tree that would overflow the stack when walked (C1).
  // `arith` folds `head` then each `[op, operand]` left to right (non-regroupable
  // `- / %` keep per-element ops); `concat`/`and`/`or`/`xor` fold a same-operator
  // run (mixed OR/XOR nests on the operator switch).
  | { kind: 'arith'; head: Expr; tail: readonly (readonly [ArithOp, Expr])[] }
  | { kind: 'concat'; items: readonly Expr[] }
  | { kind: 'neg'; expr: Expr }
  | { kind: 'and'; items: readonly Expr[] }
  | { kind: 'or'; items: readonly Expr[] }
  | { kind: 'xor'; items: readonly Expr[] }
  | { kind: 'not'; expr: Expr }
  | { kind: 'isNull'; expr: Expr; negated: boolean }
  // ISO `<boolean test>`: `x IS [NOT] TRUE|FALSE|UNKNOWN`. `truth` is the target
  // truth value (`null` = UNKNOWN); the predicate is always TRUE or FALSE.
  | { kind: 'isTruth'; expr: Expr; truth: boolean | null; negated: boolean }
  // ISO `<labeled predicate>`: `x IS [NOT] LABELED <label expression>`.
  | { kind: 'isLabeled'; expr: Expr; label: LabelExpr; negated: boolean }
  // `x IS [NOT] TYPED <type> [NOT NULL]` — the ISO value-type predicate. `category`
  // is a normalized type family; null conforms to any nullable type.
  | { kind: 'isTyped'; expr: Expr; category: string; notNull: boolean; negated: boolean }
  | { kind: 'in'; expr: Expr; list: Expr; negated: boolean }
  // ISO `<exists predicate>`: `EXISTS { p1, p2, … [WHERE pred] }` — TRUE when the
  // (correlated) sub-pattern has at least one match. Carries its own patterns and
  // optional WHERE, like a MATCH clause.
  | { kind: 'exists'; patterns: readonly PathPattern[]; where?: Expr }
  // ISO count subquery: `COUNT { p1, … [WHERE pred] }` — the number of matches of
  // the (correlated) sub-pattern. A scalar per outer row, distinct from the
  // `count(...)` grouping aggregate.
  | { kind: 'countSubquery'; patterns: readonly PathPattern[]; where?: Expr }
  // ISO `<case expression>`. With `subject`: a simple CASE (`subject = when`);
  // without: a searched CASE (`when` is a boolean condition). `elseExpr` is the
  // ELSE result, defaulting to NULL.
  | {
      kind: 'case';
      subject?: Expr;
      whens: readonly { when: Expr; then: Expr }[];
      elseExpr?: Expr;
    }
  | { kind: 'func'; name: string; args: readonly Expr[]; distinct: boolean; star: boolean };
