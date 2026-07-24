import type { Temporal } from '@lenke/core';

/**
 * The traversal AST. A `Plan` is a sequence of `Step`s. The DSL constructors
 * build these declaratively; strategies rewrite them; an executor runs them
 * against a backend graph.
 *
 * Two design rules:
 *   1. Every node is plain data — no closures, no generators. This makes the
 *      AST serializable and reorderable.
 *   2. Predicates are also data, not functions. Same reason.
 */

export type ID = string | number;

export type Direction = 'out' | 'in' | 'both';

/** Runtime `Direction` enum (TinkerPop `Direction.OUT`/`IN`/`BOTH`), e.g.
 *  `shortestPath().with(ShortestPath.edges, Direction.OUT)`. */
export const Direction = { OUT: 'out', IN: 'in', BOTH: 'both' } as const;

/** What an ordering predicate can be compared against. Temporals are admitted
 *  alongside numbers and strings: a stored DATE/DATETIME orders chronologically,
 *  and the predicate constructors lift a tagged literal (`{"@date": …}`) to an
 *  instance so `lt(D('2021-01-01'))` reaches the temporal comparator instead of
 *  falling through to the incomparable-types throw. */
export type Orderable = number | string | Temporal;

export type Predicate =
  | { op: 'eq'; value: unknown }
  | { op: 'neq'; value: unknown }
  | { op: 'gt'; value: Orderable }
  | { op: 'gte'; value: Orderable }
  | { op: 'lt'; value: Orderable }
  | { op: 'lte'; value: Orderable }
  // Gremlin semantics: half-open [first, second). To match, our `between`
  // is inclusive on `min` and exclusive on `max`.
  | { op: 'between'; min: Orderable; max: Orderable }
  // Strict open (first, second) — exclusive on both ends.
  | { op: 'inside'; min: Orderable; max: Orderable }
  // Strict complement: value < min || value > max.
  | { op: 'outside'; min: Orderable; max: Orderable }
  | { op: 'within'; values: readonly unknown[] }
  | { op: 'without'; values: readonly unknown[] }
  | { op: 'startsWith'; value: string }
  // TextP-style string predicates.
  | { op: 'endingWith'; value: string }
  | { op: 'containing'; value: string }
  | { op: 'notContaining'; value: string }
  | { op: 'regex'; value: string }
  // Negation: `has('name', not(within('a','b')))` — built via the
  // polymorphic `not(...)` step constructor when called with a predicate.
  | { op: 'not'; predicate: Predicate };

/**
 * The `by()` modulator. Attached to a parent step (path, order, dedupe,
 * group, groupCount, project, select, tree) to specify how to project each
 * value before the parent step uses it.
 *
 * Forms:
 *   - `identity`: `by()` — use the value as-is
 *   - `key`:      `by('name')` — read a property by name (vertex/edge only;
 *                  primitives pass through unchanged)
 *   - `traversal`: `by(outE().count())` — run a sub-plan with the value as
 *                  the starting traverser; project to the first result
 *
 * Comparator forms (`by(Order.asc)`, `by('age', Order.desc)`) carry an optional
 * `direction`; token forms (`by(T.id)`, `by(T.label)`) use the `token` kind.
 */
// A single `by()` call produces one of these. `direction` is meaningful only
// for `order()` (Gremlin's `by(asc)` / `by(key, desc)` / etc.); other steps
// ignore it. We attach it to every variant so the AST stays a single union.
// A `sack(op)` merge operator: `newSack = op(currentSack, projectedValue)`.
export type SackOp = 'assign' | 'sum' | 'mult' | 'min' | 'max';

export type By =
  | { kind: 'identity'; direction?: 'asc' | 'desc' }
  | { kind: 'key'; key: string; direction?: 'asc' | 'desc' }
  | { kind: 'traversal'; plan: Plan; direction?: 'asc' | 'desc' }
  // Tokens project to well-known facets of an element rather than a property.
  // Mirrors Gremlin's `T.id`, `T.label`, `T.key`, `T.value`.
  | { kind: 'token'; token: 'id' | 'label' | 'key' | 'value'; direction?: 'asc' | 'desc' }
  // `Column.keys` / `Column.values` — in `order(local)` selects whether a Map's
  // entries sort by their key or value.
  | { kind: 'column'; column: 'keys' | 'values'; direction?: 'asc' | 'desc' };

// --- Closure types passed to user-supplied steps ---
// These functions are opaque to the optimizer and break plan serialization.
type CTraverser = {
  readonly value: unknown;
  readonly path: readonly unknown[];
  readonly loopCount: number;
  /**
   * Per-traverser tags assigned via `as(label)`. A label can carry multiple
   * values inside iterative steps (e.g. `repeat(out().as('a'))`).
   */
  readonly tags: ReadonlyMap<string, unknown>;
  /**
   * Per-run side-effect bags (the same Map that backs `aggregate(key)` /
   * `store(key)` / `cap(key)`). Closures can read but should not mutate;
   * the runtime hands out a read-only view of the live map.
   */
  readonly sideEffects: ReadonlyMap<string, readonly unknown[]>;
};
export type MapClosure = (value: unknown, t: CTraverser) => unknown;
export type FlatMapClosure = (value: unknown, t: CTraverser) => Iterable<unknown>;
export type FilterClosure = (value: unknown, t: CTraverser) => boolean;
export type SideEffectClosure = (value: unknown, t: CTraverser) => void;
export type ReducerClosure = (acc: unknown, value: unknown, t: CTraverser) => unknown;

/**
 * Steps that produce, transform, or consume traversers. Each variant is a
 * pure data description — no execution semantics live here.
 */
export type Step =
  // Sources
  | { kind: 'V'; ids?: readonly ID[] }
  | { kind: 'E'; ids?: readonly ID[] }
  // Movement (vertex → vertex via edges)
  | { kind: 'out'; labels: readonly string[] }
  | { kind: 'in'; labels: readonly string[] }
  | { kind: 'both'; labels: readonly string[] }
  // Movement (vertex → edge)
  | { kind: 'outE'; labels: readonly string[] }
  | { kind: 'inE'; labels: readonly string[] }
  | { kind: 'bothE'; labels: readonly string[] }
  // Movement (edge → vertex)
  | { kind: 'outV' }
  | { kind: 'inV' }
  | { kind: 'bothV' }
  | { kind: 'otherV' }
  // Filters
  | { kind: 'has'; key: string; pred: Predicate }
  | { kind: 'hasLabel'; labels: readonly string[] }
  | { kind: 'hasId'; ids: readonly ID[] }
  | { kind: 'hasKey'; keys: readonly string[] }
  | { kind: 'simplePath' }
  | { kind: 'cyclicPath' }
  // `labels` dedupes on the tuple of values tagged at those `as_` labels;
  // `bys` is the projection modulator (e.g. `dedupe().by('name')` dedupes on the
  // `name` property). With neither, dedupes on the value itself.
  | { kind: 'dedupe'; labels?: readonly string[]; bys?: readonly By[] }
  // Cardinality. With `scope: 'local'`, the operation slices each
  // traverser's iterable value in place (typical use: after `fold()` or on
  // list-shaped values from `valueMap`/`select(...all)`). With `scope:
  // 'global'` (default, omitted), the operation slices the stream of
  // traversers itself.
  | { kind: 'take'; n: number; scope?: 'global' | 'local' }
  | { kind: 'skip'; n: number; scope?: 'global' | 'local' }
  | { kind: 'range'; start: number; end: number; scope?: 'global' | 'local' } // end < 0 means open-ended
  | { kind: 'tail'; n: number; scope?: 'global' | 'local' }
  // Iteration
  | {
      kind: 'repeat';
      body: Plan;
      until?: Plan;
      emit?: Plan;
      // TinkerPop placement: `repeat(body).emit(...)` (post-form) emits AFTER
      // each body application; `emit(...).repeat(body)` (pre-form, our
      // `.emitBefore()`) emits BEFORE each body application, including the
      // input traverser at level 0.
      emitBefore?: boolean;
      // TinkerPop `until` placement: `repeat(body).until(cond)` (post-form, the
      // default `.until()`) checks the condition AFTER the body — do-while, the
      // body runs at least once. `until(cond).repeat(body)` (pre-form, our
      // `.untilBefore()`) checks BEFORE the body — while-do, a satisfier never
      // enters the body.
      untilBefore?: boolean;
      times?: number;
    }
  // Predicates on the current value
  | { kind: 'is'; pred: Predicate }
  // Identity / no-op
  | { kind: 'identity' }
  // Filter every traverser out, ending the stream with no results.
  //
  // This step has narrow direct uses: in classic TinkerPop it's primarily a
  // signal to a remote server that `iterate()` was called and downstream
  // results are not wanted. Locally it's mostly an explicit "drop everything"
  // marker. If you find yourself reaching for it elsewhere, the traversal
  // probably wants to be rewritten — e.g. an upstream `filter`/`where` that
  // simply yields no traversers.
  //
  // Anything chained after `none()` will see an empty stream.
  //
  // With a `pred` (TinkerPop 3.8): list-predicate filter — keep the traverser
  // iff no element of its iterable value satisfies the predicate.
  | { kind: 'none'; pred?: Predicate }
  // Inject literals into the stream (source or mid-stream)
  | { kind: 'inject'; values: readonly unknown[] }
  // Unfold one level of an iterable (excluding strings)
  | { kind: 'unfold' }
  // Projection
  | { kind: 'values'; keys: readonly string[] }
  | { kind: 'valueMap'; keys?: readonly string[] }
  | { kind: 'properties'; keys: readonly string[] }
  | { kind: 'id' }
  | { kind: 'label' }
  // `bys` cycles across path elements: `path().by('name').by('age')` projects
  // the 1st, 3rd, 5th... element via 'name' and the 2nd, 4th... via 'age'.
  | { kind: 'path'; bys?: readonly By[] }
  // Terminals (executor produces a scalar). With `scope: 'local'` (Gremlin's
  // `Scope.local`), the aggregate is computed over each traverser's iterable
  // VALUE rather than across the stream — useful after `fold()` or on
  // list-shaped projections. Default is `'global'` (across the stream).
  | { kind: 'count'; scope?: 'global' | 'local' }
  | { kind: 'fold' }
  | { kind: 'toList' }
  // Numeric/comparable aggregates — return a one-element stream (Gremlin semantics)
  | { kind: 'sum'; scope?: 'global' | 'local' }
  | { kind: 'min'; scope?: 'global' | 'local' }
  | { kind: 'max'; scope?: 'global' | 'local' }
  | { kind: 'mean'; scope?: 'global' | 'local' }
  // Sort. Legacy `key` projects each element to a comparable property; `bys`
  // is the modulator form (`order().by('age')`). The first `by` projects;
  // additional `by`s are tie-breakers (Gremlin semantics). `desc` flips order
  // (applies to all `by`s for now — comparator-per-by would need closures).
  | { kind: 'order'; key?: string; desc?: boolean; bys?: readonly By[]; scope?: 'local' | 'global' }
  // Stop the stream with an error.
  | { kind: 'fail'; message?: string }
  // Sub-traversal filters: keep traverser if the sub-plan produces ≥1 result.
  // `where` has two shapes that share the same `kind`. TS narrows on the
  // presence of `plan` vs `startKey` so the executor can dispatch without
  // nullable fields.
  //
  //  - `where(subPlan)`: filter traversers whose sub-plan emits anything.
  //  - `where(startKey, predicate)`: compare the value tagged at `startKey`
  //    to the value tagged at `pred.value` (treated as another `as_` label
  //    name), via the predicate's op. Optional `bys` apply round-robin to
  //    the start- and end-tag values.
  | { kind: 'where'; plan: Plan }
  | { kind: 'where'; startKey: string; pred: Predicate; bys?: readonly By[] }
  //  - `where(predicate)`: compare the CURRENT traverser value to the value tagged
  //    at `pred.value` (a step label), via the predicate's op — e.g.
  //    `where(neq('me'))`. Narrowed by having `pred` but no `startKey`/`plan`.
  | { kind: 'where'; pred: Predicate }
  // Logical combinators over sub-plans (each plan starts from the current traverser).
  | { kind: 'and'; plans: readonly Plan[] }
  | { kind: 'or'; plans: readonly Plan[] }
  | { kind: 'not'; plan: Plan }
  // Run each sub-plan starting from the current traverser; merge outputs in order.
  | { kind: 'union'; plans: readonly Plan[] }
  // Try plans in order; yield the first non-empty plan's results per traverser.
  | { kind: 'coalesce'; plans: readonly Plan[] }
  // Run sub-plan; if it yields nothing, fall back to yielding the input traverser.
  | { kind: 'optional'; plan: Plan }
  // If the test plan yields ≥1 result, run thenPlan; else elsePlan (if present).
  | { kind: 'choose'; test: Plan; thenPlan: Plan; elsePlan?: Plan }
  // Filter: same shape as `where`. Distinct name for self-documenting traversals.
  | { kind: 'filter'; plan: Plan }
  // Inverse of `has`: keep elements WITHOUT any of the given property keys.
  // Accepts variadic keys: matches if NONE of the listed keys exist.
  | { kind: 'hasNot'; keys: readonly string[] }
  // 3-arg `has`: filter by both label and key/pred. Common shorthand.
  | { kind: 'hasLabelAnd'; label: string; key: string; pred: Predicate }
  // Like `valueMap` but also includes id and label as `T.id` / `T.label` keys.
  | { kind: 'elementMap'; keys?: readonly string[] }
  // Like `properties` but yields a single map of {key: [values]} rather than per-property.
  | { kind: 'propertyMap'; keys?: readonly string[] }
  // Replace each traverser's value with a constant.
  | { kind: 'constant'; value: unknown }
  // Inside a `repeat` body, the current loop count (number of iterations so far).
  | { kind: 'loops' }
  // Tag the current traverser's value with `label` so a downstream `select`
  // can recall it. Stored on the traverser itself; not a side effect.
  | { kind: 'as'; label: string }
  // Pull labeled positions back out. With one label, replaces the traverser
  // value with the tagged value. With multiple labels, yields an object
  // `{[label]: value}`. Traversers missing any selected label are dropped.
  // `pop` selects which tag to recall when a label was tagged multiple times
  // (e.g. inside `repeat(out().as('a'))`). Defaults to `'last'`.
  | {
      kind: 'select';
      labels: readonly string[];
      pop?: 'first' | 'last' | 'all';
      // Modulator: project each labeled position via `bys[i]`. If `bys` has
      // fewer entries than labels, the extra labels project as identity.
      bys?: readonly By[];
    }
  // `select(Column.keys)` / `select(Column.values)`: extract a Map's keys or values
  // as a list, preserving entry order (the observable reader for `order(local)`).
  | { kind: 'selectColumn'; column: 'keys' | 'values' }
  // Aggregation: collect the entire stream into one Map<key, value[]>.
  // `keyBy` / `valueBy` are property names for now; full sub-traversal `by()`
  // lands later. Without `keyBy`, group by the value itself; without
  // `valueBy`, the lists hold the elements themselves.
  // Legacy `keyBy`/`valueBy` are property names. `bys[0]` overrides keyBy and
  // `bys[1]` overrides valueBy via the modulator form (`group().by(k).by(v)`).
  | { kind: 'group'; keyBy?: string; valueBy?: string; bys?: readonly By[] }
  // Aggregation: like `group`, but values are counts. Legacy `by` is a
  // property name; `bys[0]` is the modulator form.
  | { kind: 'groupCount'; by?: string; bys?: readonly By[] }
  // Per-traverser: yield a `{ [key]: value }` object. `bys[i]` corresponds to
  // `keys[i]` and is a property name; an undefined `by` projects the traverser
  // value itself for that key.
  // `bys[i]` projects the value for `keys[i]`. A missing entry projects the
  // traverser value itself.
  | { kind: 'project'; keys: readonly string[]; bys?: readonly By[] }
  // --- Closure-bearing variants ---
  // These are NOT JSON-serializable. Use `serialize(plan)` to detect this
  // at the boundary if you need to ship a plan over the wire.
  // Each has a sibling sub-plan form (above); the DSL's `map`/`filter`/etc.
  // dispatch on whether the user passed a `StepFn` or a raw closure.
  | { kind: 'mapFn'; fn: MapClosure }
  | { kind: 'flatMapFn'; fn: FlatMapClosure }
  | { kind: 'filterFn'; fn: FilterClosure }
  | { kind: 'sideEffectFn'; fn: SideEffectClosure }
  | { kind: 'foldFn'; seed: unknown; fn: ReducerClosure }
  // Run a sub-plan for its effect and pass the original traverser through unchanged.
  | { kind: 'sideEffect'; plan: Plan }
  // Run a sub-plan as a "barrier": the body sees only the current traverser,
  // not the whole stream. Useful for `local(out().count())` semantics.
  | { kind: 'local'; plan: Plan }
  // Side-effect: collect each traverser into a named bag in the run's
  // sideEffects map. Pass the traverser through unchanged.
  | { kind: 'aggregate'; key: string }
  // Like `aggregate`, but lazy/eager-agnostic alias name. In TinkerPop,
  // `store` is the lazy form (no barrier); `aggregate` collects with a
  // barrier. In v2 our `aggregate` is already lazy (per-traverser yield), so
  // `store` is a semantic alias today. Kept distinct in the AST so a future
  // optimizer pass can introduce the barrier on `aggregate` without breaking
  // `store` semantics.
  | { kind: 'store'; key: string }
  // Read back from the sideEffects map. Replaces the stream with the bag.
  | { kind: 'cap'; key: string }
  // Materialization point: drain the upstream into a list, then re-emit. With
  // bulk traversers this would also collapse duplicates; v2 has no bulk, so
  // `barrier()` is a forced eager-evaluation marker. Useful before steps that
  // depend on full upstream availability (e.g. before a sub-plan that closes
  // over side-effects populated upstream).
  | { kind: 'barrier' }
  // Terminal: collect all traversers' paths into a nested map. Each path
  // becomes a chain of map keys (path[0] -> path[1] -> ... -> {}). `bys` project
  // each path element before it becomes a key (cycling, as in `path().by(...)`).
  | { kind: 'tree'; bys?: readonly By[] }
  // Switch over a sub-plan's first result. Per traverser, run `test`; route
  // to the first option whose `match` equals that result, else `default`.
  | {
      kind: 'branch';
      test: Plan;
      options: readonly { match: unknown; plan: Plan }[];
      default?: Plan;
    }
  // Replace each traverser's value with all outputs of the sub-plan
  // (zero or more per traverser).
  | { kind: 'flatMap'; plan: Plan }
  // Replace each traverser's value with the first output of the sub-plan.
  // Drops the traverser if the sub-plan yields nothing.
  | { kind: 'map'; plan: Plan }
  // Random subset of N traversers. Materializes the stream.
  | { kind: 'sample'; n: number }
  // Yield the value of a property/edge as if already projected. For Vertex/Edge
  // this is the value as-is; for `{key, value}` property objects (from
  // `properties()`), unwrap to the value field.
  | { kind: 'value' }
  // Annotate stream with positional indexes: each traverser becomes
  // `[value, index]`.
  | { kind: 'index' }
  // Evaluate an arithmetic expression on traverser values. `_` references the
  // current traverser value; other identifiers reference `as_`-bound labels.
  // `bys` project each operand (by first-appearance order, cycling) — e.g.
  // `math('a + b').by('age')` sums the `age` of the values tagged `a` and `b`.
  | { kind: 'math'; expr: string; bys?: readonly By[] }
  // Filter property objects (`{key, value}`) by their `value` field.
  | { kind: 'hasValue'; values: readonly unknown[] }
  // Declarative pattern match across labeled positions.
  | { kind: 'match'; patterns: readonly Plan[] }
  // Side-effect: accumulate matching edges into a named subgraph.
  | { kind: 'subgraph'; key: string }
  // Emit the shortest vertex path(s) from each source vertex. `target` (set via
  // `.with(ShortestPath.target, …)`) filters which vertices are destinations;
  // absent ⇒ every reachable vertex. `direction` (set via `.with(ShortestPath.edges,
  // Direction.OUT|IN|BOTH)`) picks which incident edges the BFS follows — `'both'`
  // (default, TinkerPop-conformant undirected), `'out'`, or `'in'`.
  | { kind: 'shortestPath'; target?: Plan; direction?: Direction }
  // --- OLAP graph algorithms (computed locally) -------------------------
  //
  // Each step runs a whole-graph algorithm and writes a per-vertex result to a
  // property, then passes the incoming traversers through so a downstream step
  // reads it (`g.V().pageRank().order().by('gremlin.pageRankVertexProgram.pageRank')`).
  // `property` overrides the default TinkerPop column name (via
  // `.with(<Algo>.propertyName, …)`); `times` sets the iteration count
  // (`.with(<Algo>.times, …)`); PageRank's `alpha` is the damping factor.
  // `withComputer()` is accepted upstream as a no-op — lenke always computes
  // in-process — so it leaves no step.
  | { kind: 'pageRank'; property?: string; times?: number; alpha?: number }
  | { kind: 'connectedComponent'; property?: string }
  | { kind: 'peerPressure'; property?: string; times?: number }
  // --- Mutation (graph-write) -------------------------------------------
  //
  // `addV(label?)` inserts a fresh vertex into the graph. The output stream
  // contains the new vertex. Subsequent `property(...)` calls bind values to
  // it. With no label, the vertex is created label-less.
  | { kind: 'addV'; label?: string }
  // `addE(label).from(X).to(Y)` inserts an edge between two vertex endpoints.
  // Each endpoint is one of:
  //   - undefined   → fall back to the current traverser value
  //   - tag string  → recall a previously `as(label)`-tagged vertex
  //   - sub-plan    → run a sub-plan and use its first emitted vertex
  // If both `from` and `to` are undefined the current traverser is treated as
  // the OUT (from) endpoint and the executor throws — IN must be specified.
  // The output stream contains the new edge.
  | {
      kind: 'addE';
      label: string;
      from?: AddEEndpoint;
      to?: AddEEndpoint;
    }
  // `property(key, value)` overwrites a single property on the current
  // vertex/edge. Output stream contains the (mutated) element.
  // `cardinality` is recorded for spec parity but v2 stores properties as
  // `Record<string, unknown>` (single-valued); `'list'` and `'set'` currently
  // behave the same as `'single'`. Once multi-cardinality lands, this is the
  // hook the executor will read.
  | {
      kind: 'property';
      key: string;
      value: unknown;
      cardinality?: 'single' | 'list' | 'set';
    }
  // `drop()` removes the current vertex/edge from the graph and emits nothing
  // for that traverser. Dropping a vertex cascades through any edges
  // attached to it (per `Graph.removeVertex`).
  | { kind: 'drop' }
  // `withSack(init)` enables the per-traverser sack (default `init`).
  | { kind: 'withSack'; init: unknown }
  // `sack()` emits the current sack; `sack(op).by(proj)` merges the projected
  // value into the sack via `op` and passes the traverser through.
  | { kind: 'sack'; op?: SackOp; bys?: readonly By[] };

/**
 * Endpoint specification for `addE(...).from(X)` / `.to(X)`.
 *  - `'tag'`  — recall a previously `as(label)`-tagged vertex by name
 *  - `'plan'` — run a sub-plan and use its first emitted vertex
 */
export type AddEEndpoint = { kind: 'tag'; label: string } | { kind: 'plan'; plan: Plan };

export type Plan = {
  readonly steps: readonly Step[];
};

export const emptyPlan: Plan = { steps: [] };

/**
 * Brand applied to every DSL-produced `(plan: Plan) => Plan` so the runtime
 * can distinguish a sub-traversal from a user-supplied closure of the same
 * arity. See the dispatch in `steps.ts`'s overloaded constructors (`map`,
 * `filter`, etc.).
 */
export const STEP_FN: unique symbol = Symbol.for('@lenke/gremlin/StepFn');

export type StepFn = ((plan: Plan) => Plan) & { readonly [STEP_FN]: true };

export const appendStep = (step: Step): StepFn => {
  const fn = (plan: Plan): Plan => ({ steps: [...plan.steps, step] });
  Object.defineProperty(fn, STEP_FN, { value: true, enumerable: false });

  return fn as StepFn;
};

export const isStepFn = (x: unknown): x is StepFn =>
  typeof x === 'function' && (x as { [STEP_FN]?: boolean })[STEP_FN] === true;

/**
 * Type-only view of a runtime traverser, exposed to user closures. Closures
 * receive the current value AND a read-only view of the carrying traverser
 * so they can read tags, path, loop count.
 */
export type TraverserView = {
  readonly value: unknown;
  readonly path: readonly unknown[];
  readonly loopCount: number;
  readonly tags: ReadonlyMap<string, unknown>;
  readonly sideEffects: ReadonlyMap<string, readonly unknown[]>;
};
