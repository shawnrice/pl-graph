// Public API of the gremlin executor.
//
// `run(plan, graph)` is the primary entry point — runs a plan and yields the
// final traverser values. `toArray` and `toSet` are eager terminals.
//
// Implementation lives in sibling files: `runtime.ts` for shared types
// and helpers, `dispatch.ts` for the kind-routing switch + recursive
// `applyPlanToStream`, and per-category files (`movement.ts`, `filters.ts`,
// `aggregation.ts`, etc.) for the step-impl generators.

import type { Graph } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Plan, Step } from '../ast.js';
import { applyStep } from './dispatch.js';
import { seedFromIndex } from './index-seed.js';
import { newContext, planReadsPath, type Traverser, unwrap } from './runtime.js';
import { applySource } from './sources.js';

/**
 * Fault on a plan that is invalid whatever the data.
 *
 * `path().id()` cannot succeed on any graph: a path is not an Element, so it has
 * no id and no label. TinkerPop types `IdStep<S extends Element>` and does
 * `traverser.get().id()`, which on a path is a bare `ClassCastException` once
 * the generic is erased ("ImmutablePath cannot be cast to ...Element"). Both
 * engines used to return null instead — agreeing with each other, and with
 * nothing else.
 *
 * Checked on the STEP LIST rather than per traverser, so it costs one scan and
 * no walk. That is also what makes it expressible here at all: at RUNTIME a path
 * is a plain array, indistinguishable from what `fold()` produces, but the
 * `path` STEP is not.
 *
 * `DataException` — an ISO data exception covers "a type mismatch in an
 * operation", which is what this is. Not a syntax error: the traversal parses
 * fine. Not `InvalidValue`: that means a value outside the LPG property-value
 * model, and a path is a perfectly good value that simply has no id. The Rust
 * core raises the same code from the same check.
 *
 * Only through steps that pass the traverser's value on UNCHANGED — `unfold()`
 * turns a path into its elements, and `id()` on those is fine, so a scan that
 * merely looked for a later `id()` would reject working traversals.
 */
/**
 * Fault, statically, on an aggregate/sort/vertex-move applied to the WRONG frontier
 * KIND — mirroring the native engine, whose parser rejects these at parse time.
 *
 * The engine is a static planner (it faults at parse); pure-TS is a streaming executor
 * that would otherwise fault only at RUNTIME when a bad value flows through — so the two
 * diverge on an EMPTY frontier (`g.V().out('NOPE').sum()` never sees a vertex, so a
 * runtime check gives null while the engine still faults). This pass restores symmetry by
 * classifying the frontier kind from the step chain (exactly what the engine's parser
 * does) and faulting up front, regardless of data:
 *
 *  - `sum`/`min`/`max`/`mean` (global) and a bare `order()` / direction-only `order().by(desc)`
 *    over a graph ELEMENT (vertex/edge): a Vertex/Edge is not a number and has no natural
 *    order — TinkerPop faults; project with `values('<key>')` / `order().by('<key>')`.
 *  - `inV`/`outV`/`bothV`/`otherV` on a NON-edge frontier: these move to an edge's endpoint,
 *    so they require an edge traverser (`g.V().otherV()` is invalid; use `outE().otherV()`).
 *
 * Conservative: the frontier is only 'vertex'/'edge'/'scalar' when KNOWN; branch/collection/
 * map producers reset it to 'unknown', where nothing faults — a missed fault is safe, a false
 * one would break a valid query. Code is `E_SYNTAX`, matching the engine's parse-time fault.
 */
type Frontier = 'vertex' | 'edge' | 'scalar' | 'unknown';

const VERTEX_STEPS = new Set(['V', 'out', 'in', 'both', 'inV', 'outV', 'bothV', 'otherV', 'addV']);
const EDGE_STEPS = new Set(['E', 'outE', 'inE', 'bothE', 'addE']);
const SCALAR_STEPS = new Set([
  'values',
  'value',
  'id',
  'label',
  'count',
  'sum',
  'min',
  'max',
  'mean',
  'math',
  'loops',
  // inject() prepends literal values (strings/numbers) to the stream, so the frontier is
  // no longer purely elements — a following navigation/projection faults, exactly as native
  // rejects `inject('x').outV()` and TinkerPop throws (the literal is not an element).
  'inject',
]);

// Element-type algebra (rejected statically, matching native's parse-time rejection and
// TinkerPop's runtime ClassCastException — verified against gremlin-console):
//  - out/in/both (adjacency) + outE/inE/bothE (edge hops) navigate FROM a vertex.
//  - inV/outV/bothV/otherV (endpoints) move to an edge's endpoint, so need an edge.
//  - values/value/id/label/properties/… read off an ELEMENT.
// count/fold/sum/… are stream reducers, valid on ANY frontier (`V().count().count()` → [1]).
const ADJ_OR_HOP = new Set(['out', 'in', 'both', 'outE', 'inE', 'bothE']);
const PROJECTION = new Set([
  'values',
  'value',
  'id',
  'label',
  'properties',
  'propertyMap',
  'valueMap',
  'elementMap',
  'key',
]);
// Frontier-PRESERVING filters / barriers / side-effects / writes-that-pass-through.
const PRESERVE_STEPS = new Set([
  'has',
  'hasLabel',
  'hasId',
  'hasKey',
  'hasNot',
  'hasValue',
  'hasLabelAnd',
  'where',
  'and',
  'or',
  'not',
  'is',
  'dedupe',
  'take',
  'skip',
  'range',
  'tail',
  'order',
  'as',
  'simplePath',
  'cyclicPath',
  'aggregate',
  'store',
  'barrier',
  'sample',
  'sideEffect',
  'sideEffectFn',
  'filter',
  'filterFn',
  'identity',
  'none',
  'withSack',
  'drop',
  'property',
  'repeat',
]);

// Everything not classified (maps/collections/branches/OLAP/ambiguous) resets to 'unknown',
// where nothing faults — a missed fault is safe; a false positive would break a valid query.
const nextFrontier = (kind: Step['kind'], prev: Frontier): Frontier => {
  if (VERTEX_STEPS.has(kind)) {
    return 'vertex';
  }

  if (EDGE_STEPS.has(kind)) {
    return 'edge';
  }

  if (SCALAR_STEPS.has(kind)) {
    return 'scalar';
  }

  return PRESERVE_STEPS.has(kind) ? prev : 'unknown';
};

const isElement = (f: Frontier): boolean => f === 'vertex' || f === 'edge';

const EDGE_HOPS = new Set(['outE', 'inE', 'bothE']);

// Whether an EDGE is in scope for `inV`/`outV`/`otherV`, which move to an edge's endpoint
// and so require one. `true` = an edge is reachable, `false` = definitely none, `undefined`
// = unknown (a branch/collection producer — never fault). An edge step establishes it; a
// vertex step or vertex-move (out/inV/…) clears it (the edge, if any, was consumed reaching
// the vertex); a scalar projection (values/label/count) also CLEARS it — a scalar is not an
// edge, so `E().count().inV()` and `E().label().inV()` fault (TinkerPop throws), matching the
// frontier-based checks below. Preserving filters/barriers keep it.
type HasEdge = boolean | undefined;

const EDGE_SOURCE = new Set(['E', 'outE', 'inE', 'bothE', 'addE']);
const VERTEX_MOVE = new Set(['V', 'out', 'in', 'both', 'addV', 'inV', 'outV', 'bothV', 'otherV']);

const nextHasEdge = (kind: Step['kind'], prev: HasEdge): HasEdge => {
  if (EDGE_SOURCE.has(kind)) {
    return true;
  }

  if (VERTEX_MOVE.has(kind)) {
    return false;
  }

  // A scalar producer clears the edge scope (a number/string/id is not an edge).
  if (SCALAR_STEPS.has(kind)) {
    return false;
  }

  return PRESERVE_STEPS.has(kind) ? prev : undefined;
};

// Combine the ending edge-scope of a branch's arms: any arm that DEFINITELY yields a
// non-edge (`false`) makes a following `inV`/`outV` faultable; all-edge arms keep it in
// scope; anything unknown stays unknown (conservative — never fault).
const combineHasEdge = (arms: readonly HasEdge[]): HasEdge => {
  if (arms.some((a) => a === false)) {
    return false;
  }

  if (arms.length > 0 && arms.every((a) => a === true)) {
    return true;
  }

  return undefined;
};

const checkStep = (step: Step, f: Frontier, hasEdge: HasEdge, edgeHasOrigin: boolean): void => {
  const k = step.kind;

  // Adjacency (out/in/both) and edge hops (outE/inE/bothE) navigate FROM a vertex. On a
  // KNOWN edge or scalar frontier the value is not a vertex, so TinkerPop throws and native
  // rejects — fault to match. `unknown` (a branch output that MIGHT be a vertex) never faults.
  if (ADJ_OR_HOP.has(k) && (f === 'edge' || f === 'scalar')) {
    throw new LenkeError(
      `${k}() moves from a vertex, but the frontier is ${f === 'edge' ? 'an edge' : 'a scalar'} — ` +
        `${f === 'edge' ? 'use an endpoint step (inV()/outV()/otherV())' : 'project to a vertex'} ` +
        `before ${k}()`,
      { code: ErrorCode.Syntax },
    );
  }

  // Projections (values/id/label/properties/…) read off an ELEMENT. On a scalar frontier
  // (`V().id().values('x')`, `E().count().label()`) there is no element — TinkerPop throws.
  if (PROJECTION.has(k) && f === 'scalar') {
    throw new LenkeError(
      `${k}() reads from a graph element, but the frontier is a projected scalar ` +
        `(values()/id()/label()/count()/inject()); it has no ${k === 'id' || k === 'label' ? k : 'properties'}`,
      { code: ErrorCode.Syntax },
    );
  }

  if (
    (k === 'sum' || k === 'min' || k === 'max' || k === 'mean') &&
    (step as { scope?: string }).scope !== 'local' &&
    isElement(f)
  ) {
    throw new LenkeError(
      `${k}() over graph elements is not supported — a vertex/edge is not a number; ` +
        `project with values('<key>') first`,
      { code: ErrorCode.Syntax },
    );
  }

  if (k === 'order' && isElement(f)) {
    const o = step as { key?: string; bys?: readonly { kind: string }[]; scope?: string };
    // Sorts the RAW element when there is no key projection and either no `by` or a
    // direction-only `by(desc)` (an `identity` By). A `by('<key>')`/`by(<traversal>)`/
    // `by(T.id)` projects a comparable value, so it is fine.
    const sortsRawElement =
      o.scope !== 'local' &&
      !o.key &&
      (!o.bys || o.bys.length === 0 || o.bys.some((b) => b.kind === 'identity'));

    if (sortsRawElement) {
      throw new LenkeError(
        `order() over graph elements is not supported — elements have no natural order; ` +
          `use order().by('<key>')`,
        { code: ErrorCode.Syntax },
      );
    }
  }

  // `inV`/`outV`/`bothV`/`otherV` move to an edge's endpoint, so they require an edge in
  // scope. Fault iff there is DEFINITELY none (`g.V().otherV()`, `g.V().values('x').inV()`);
  // an edge frontier or an edge-derived scalar (`E().label().inV()`) is valid, and an
  // unknown frontier (a branch that might yield edges) never faults. Matches the native
  // engine's parse-time rejection.
  if ((k === 'inV' || k === 'outV' || k === 'bothV' || k === 'otherV') && hasEdge === false) {
    throw new LenkeError(
      `${k}() requires an edge — a vertex has no incident edge to move across; ` +
        `use an edge step (outE()/inE()/bothE()) before ${k}()`,
      { code: ErrorCode.Syntax },
    );
  }

  // `otherV()` is "the endpoint I did NOT arrive from" — it reads the path's reference
  // vertex. Off a BARE edge frontier (`g.E().otherV()`) there is no prior vertex, so it
  // is undefined; TinkerPop throws and the native engine rejects it. `inV`/`outV`/`bothV`
  // name a specific endpoint and stay valid off a bare edge.
  if (k === 'otherV' && f === 'edge' && !edgeHasOrigin) {
    throw new LenkeError(
      `otherV() has no reference vertex off a bare edge frontier — it returns the endpoint ` +
        `not arrived from, but E() provides no origin; reach the edge via a vertex ` +
        `(outE()/inE()/bothE()) first`,
      { code: ErrorCode.Syntax },
    );
  }
};

// Recurse into a step list, tracking the frontier kind. A branch step's sub-traversals
// (union/coalesce arms, an optional/choose body, a choose condition) each START from the
// frontier at the branch — so `V().union(inV(), …)` faults on the vertex-move exactly as
// the native engine's parser does, not just at the top level.
const checkSteps = (
  steps: readonly Step[],
  start: Frontier,
  startHasEdge: HasEdge,
  startEdgeHasOrigin: boolean,
): { f: Frontier; hasEdge: HasEdge } => {
  let f = start;
  let hasEdge = startHasEdge;
  // Whether the current edge frontier was reached THROUGH a vertex (so the path holds a
  // reference vertex for `otherV()`). An edge hop (`outE`/…) sets it; a bare `E()` source
  // clears it. A branch arm inherits it from the frontier at the branch.
  let edgeHasOrigin = startEdgeHasOrigin;

  for (const step of steps) {
    checkStep(step, f, hasEdge, edgeHasOrigin);

    const k = step.kind;

    // A branch's arms each START from the frontier at the branch; the branch's OUTPUT
    // edge-scope is the combination of the arms' endings (so `V().union(outE(), out()).inV()`
    // faults — one arm yields a non-edge — exactly as the native parser rejects it).
    if (k === 'union' || k === 'coalesce') {
      const ends = (step as { plans: readonly Plan[] }).plans.map((p) =>
        checkSteps(p.steps, f, hasEdge, edgeHasOrigin),
      );

      f = 'unknown';
      hasEdge = combineHasEdge(ends.map((e) => e.hasEdge));

      continue;
    }

    if (k === 'optional') {
      // The output is the matched (body) traversers AND the unmatched (pre-body) ones,
      // so combine the body's ending edge-scope with the pre-body scope.
      const body = checkSteps((step as { plan: Plan }).plan.steps, f, hasEdge, edgeHasOrigin);

      f = 'unknown';
      hasEdge = combineHasEdge([hasEdge, body.hasEdge]);

      continue;
    }

    if (k === 'choose') {
      const c = step as { test: Plan; thenPlan: Plan; elsePlan?: Plan };
      checkSteps(c.test.steps, f, hasEdge, edgeHasOrigin);
      const thenEnd = checkSteps(c.thenPlan.steps, f, hasEdge, edgeHasOrigin);
      const elseEnd = c.elsePlan
        ? checkSteps(c.elsePlan.steps, f, hasEdge, edgeHasOrigin)
        : { f, hasEdge };

      f = 'unknown';
      hasEdge = combineHasEdge([thenEnd.hasEdge, elseEnd.hasEdge]);

      continue;
    }

    if (EDGE_HOPS.has(k)) {
      edgeHasOrigin = true;
    } else if (k === 'E') {
      edgeHasOrigin = false;
    }

    f = nextFrontier(k, f);
    hasEdge = nextHasEdge(k, hasEdge);
  }

  return { f, hasEdge };
};

const assertFrontierTypes = (plan: Plan): void => {
  checkSteps(plan.steps, 'unknown', false, false);
};

// Steps that PASS A PATH THROUGH unchanged — the scan keeps looking past them for an
// element step. (KINDS, not names — `limit()` builds a `take`, `dedup()` a `dedupe`.)
const PATH_PASSTHROUGH = new Set([
  'take',
  'skip',
  'range',
  'tail',
  'sample',
  'dedupe',
  'order',
  'barrier',
  'as',
  'identity',
  'simplePath',
  'cyclicPath',
]);

// Steps that require an ELEMENT (a vertex/edge). Applied straight to a PATH they throw in
// TinkerPop (`ImmutablePath cannot be cast to Element/Edge/Vertex`) — a path is not an
// element. Verified on gremlin-console: `path().values(...)`, `.hasLabel(...)`, `.inV()`,
// `.out(...)`, `.id()`/`.label()` all ClassCastException. `unfold()` (turns a path into
// its elements), `count`/`fold`/etc CONSUME the path and end the scan.
const ELEMENT_STEP_ON_PATH = new Set([
  'out',
  'in',
  'both',
  'outE',
  'inE',
  'bothE',
  'inV',
  'outV',
  'bothV',
  'otherV',
  'values',
  'value',
  'valueMap',
  'propertyMap',
  'properties',
  'property',
  'key',
  'id',
  'label',
  'has',
  'hasNot',
  'hasLabel',
  'hasId',
  'hasKey',
  'hasValue',
]);

const assertPlanIsSatisfiable = (plan: Plan): void => {
  for (let i = 0; i < plan.steps.length; i++) {
    if (plan.steps[i].kind !== 'path') {
      continue;
    }

    for (const later of plan.steps.slice(i + 1)) {
      if (ELEMENT_STEP_ON_PATH.has(later.kind)) {
        throw new LenkeError(
          `${later.kind}() is not defined on a path: a path is not an element ` +
            `(unfold() it into its elements first)`,
          { code: ErrorCode.DataException },
        );
      }

      if (!PATH_PASSTHROUGH.has(later.kind)) {
        break;
      }
    }
  }
};

/**
 * Run a plan against a graph. Always returns an `Iterable<unknown>` —
 * terminal steps (`count`, `fold`, `toList`) yield exactly one value; other
 * steps yield zero or more. This matches Gremlin's "every step is a stream"
 * model and keeps `pipe(count(), is(gt(5)))` composable.
 */
export const run = (plan: Plan, graph: Graph): Iterable<unknown> => {
  // Before anything runs: static faults the engine raises at parse time — a plan that
  // cannot succeed on any graph, and an aggregate/sort/vertex-move on the wrong frontier.
  assertPlanIsSatisfiable(plan);
  assertFrontierTypes(plan);

  // Decide once whether any step observes the path; if not, traversers skip
  // path bookkeeping for the whole run (see planReadsPath / startTraverser).
  const ctx = newContext(planReadsPath(plan));

  // Peel leading source-configuration steps (`withSack`) — like TinkerPop's
  // GraphTraversalSource config they precede the actual source (V()/E()/…), so
  // they set up the context rather than seed a stream.
  let head = 0;

  while (head < plan.steps.length && plan.steps[head].kind === 'withSack') {
    ctx.sackInit = { value: (plan.steps[head] as Extract<Step, { kind: 'withSack' }>).init };
    head++;
  }

  const effective = head === 0 ? plan : { ...plan, steps: plan.steps.slice(head) };

  // If the plan opens `V()`/`E()` + a seedable `has` on an indexed key, seed
  // from the index and apply only the residual steps; otherwise scan as usual.
  const seeded = seedFromIndex(effective, graph, ctx.tracksPath);
  let stream: Iterable<Traverser<unknown>> | null = seeded?.stream ?? null;
  const steps = seeded?.steps ?? effective.steps;

  for (const step of steps) {
    if (stream === null) {
      stream = applySource(step, graph, ctx.tracksPath);
      continue;
    }

    stream = applyStep(step, stream, graph, ctx);
  }

  return unwrap(stream ?? []);
};

/**
 * Eager terminal: run the plan and collect every emitted value into an array.
 *
 * Equivalent to `[...run(plan, graph)]`. Provided for parity with legacy and
 * because the intent ("I want the answer as an array, not a lazy iterable")
 * is common enough to deserve a name.
 */
export const toArray = (plan: Plan, graph: Graph): unknown[] => [...run(plan, graph)];

/**
 * Eager terminal: run the plan and collect emitted values into a Set, dropping
 * duplicates by JS reference/primitive equality.
 *
 * Equivalent to `new Set(run(plan, graph))`. For value-based de-duplication
 * over vertices/edges/objects, prefer the `dedupe()` step inside the plan —
 * a `Set` only de-dupes by `===`, so two distinct vertex objects with the same
 * `id` would both be retained.
 */
export const toSet = (plan: Plan, graph: Graph): Set<unknown> => new Set(run(plan, graph));
