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
const assertPlanIsSatisfiable = (plan: Plan): void => {
  // The step KINDS, which are not always the step names — `limit()` builds a
  // `take` and `dedup()` builds a `dedupe`.
  const passesValueThrough = new Set([
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
  ]);

  for (let i = 0; i < plan.steps.length; i++) {
    if (plan.steps[i].kind !== 'path') {
      continue;
    }

    for (const later of plan.steps.slice(i + 1)) {
      if (later.kind === 'id' || later.kind === 'label') {
        throw new LenkeError(`${later.kind}() is not defined on a path: a path is not an element`, {
          code: ErrorCode.DataException,
        });
      }

      if (!passesValueThrough.has(later.kind)) {
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
  // Before anything runs: a plan that cannot succeed on any graph.
  assertPlanIsSatisfiable(plan);

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
