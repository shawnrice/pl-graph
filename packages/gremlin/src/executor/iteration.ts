import type { Graph } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Plan, Step } from '../ast.js';
import { applyPlanToStream } from './dispatch.js';
import { hasAny, incLoops, isEmptyPlan, type RunContext, type Traverser } from './runtime.js';

/**
 * Per-`repeat()` cap on the total traversers its body produces. A
 * `repeat(both())` with no `until`/`times` on a cyclic or dense graph grows the
 * frontier multiplicatively each level (bounded only by the 100-iteration cap,
 * which bounds depth, not work), so it can exhaust memory long before it stops.
 * Past this budget we raise `ResourceExhausted` rather than hang/OOM.
 */
const REPEAT_BUDGET = 1_000_000;

export const unionStep = function* (
  stream: Iterable<Traverser<unknown>>,
  plans: readonly Plan[],
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  // union() runs each branch over the WHOLE incoming stream and concatenates the results
  // arm-by-arm — NOT per element. A reducing barrier in an arm (`count()`, `limit(1)`,
  // `fold()`) therefore reduces the whole stream: `V().union(limit(1), count())` yields one
  // vertex then the total count, matching TinkerPop and the native branch. (coalesce/choose/
  // optional below stay PER-ELEMENT — they route each traverser individually.)
  const all = [...stream];

  for (const plan of plans) {
    yield* applyPlanToStream(plan, all, graph, ctx);
  }
};

export const coalesceStep = function* (
  stream: Iterable<Traverser<unknown>>,
  plans: readonly Plan[],
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    for (const plan of plans) {
      const out = [...applyPlanToStream(plan, [t], graph, ctx)];

      if (out.length > 0) {
        yield* out;
        break;
      }
    }
  }
};

export const optionalStep = function* (
  stream: Iterable<Traverser<unknown>>,
  plan: Plan,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    const out = [...applyPlanToStream(plan, [t], graph, ctx)];

    if (out.length > 0) {
      yield* out;
    } else {
      yield t;
    }
  }
};

export const chooseStep = function* (
  stream: Iterable<Traverser<unknown>>,
  test: Plan,
  thenPlan: Plan,
  elsePlan: Plan | undefined,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    const branch = hasAny(applyPlanToStream(test, [t], graph, ctx)) ? thenPlan : elsePlan;

    if (branch) {
      yield* applyPlanToStream(branch, [t], graph, ctx);
    } else {
      // Per TinkerPop spec: if test fails and no elsePlan, traverser passes
      // through unchanged (identity behavior).
      yield t;
    }
  }
};

// --- Repeat -------------------------------------------------------------

export const repeatStep = function* (
  stream: Iterable<Traverser<unknown>>,
  step: Extract<Step, { kind: 'repeat' }>,
  graph: Graph,
  // The enclosing run's context — threaded into the body (and until/emit)
  // sub-plans so a side-effect step inside the loop (`subgraph`/`store`/
  // `aggregate`) writes to the OUTER side-effect scope a downstream `cap(key)`
  // reads. Without it the body ran in a fresh context and its side-effects
  // silently vanished (a transitive-`subgraph` blast radius came back empty).
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  // Cap iterations to `times` if given; else 100 to avoid runaway.
  const maxIterations = step.times ?? 100;
  // `until(plan)` empty means "no until" — let `times` be the only stopper.
  // `emit(plan)` empty means "emit every traverser at every level".
  const hasUntil = step.until !== undefined && !isEmptyPlan(step.until);
  const hasEmit = step.emit !== undefined;
  const emitAll = hasEmit && step.emit !== undefined && isEmptyPlan(step.emit);
  const emitBefore = step.emitBefore === true;

  const matchesEmit = (t: Traverser<unknown>): boolean => {
    if (emitAll) {
      return true;
    }

    return hasAny(applyPlanToStream(step.emit!, [t], graph, ctx));
  };

  // `until` placement (TinkerPop): pre-form `until(cond).repeat(body)` checks
  // BEFORE the body (while-do — a satisfier never enters the body); the default
  // post-form `repeat(body).until(cond)` checks AFTER the body (do-while — the
  // body runs at least once). Distinguished by the AST `untilBefore` flag.
  const untilBefore = hasUntil && step.untilBefore === true;
  const untilAfter = hasUntil && !untilBefore;
  const untilMatches = (t: Traverser<unknown>): boolean =>
    hasAny(applyPlanToStream(step.until!, [t], graph, ctx));

  let frontier: Traverser<unknown>[] = [...stream].map(incLoops);
  let work = 0;

  for (let i = 0; i < maxIterations && frontier.length > 0; i++) {
    // Pre-form emit (TinkerPop's `emit(...).repeat(body)`): emit before each
    // body application, including the input traverser at level 0.
    if (hasEmit && emitBefore) {
      for (const t of frontier) {
        if (matchesEmit(t)) {
          yield t;
        }
      }
    }

    // while-do: check `until` BEFORE the body — a satisfier exits without ever
    // running the body.
    let advancing = frontier;

    if (untilBefore) {
      advancing = [];

      for (const t of frontier) {
        if (untilMatches(t)) {
          yield t;
        } else {
          advancing.push(t);
        }
      }
    }

    // Advance the frontier through the body (in the ENCLOSING ctx, so
    // side-effects inside the body reach the outer scope).
    const stepped: Traverser<unknown>[] = [];

    for (const t of applyPlanToStream(step.body, advancing, graph, ctx)) {
      work += 1;

      if (work > REPEAT_BUDGET) {
        throw new LenkeError(
          'repeat() exceeded the traversal budget; add a tighter until()/times()',
          { code: ErrorCode.ResourceExhausted },
        );
      }

      stepped.push(incLoops(t));
    }

    // Post-form emit (TinkerPop's default `repeat(body).emit(...)`): emit every
    // body output. The final iteration's output is emitted here, so with an
    // until stopper no additional post-loop yield is needed.
    if (hasEmit && !emitBefore) {
      for (const t of stepped) {
        if (matchesEmit(t)) {
          yield t;
        }
      }
    }

    // do-while: check `until` AFTER the body — a satisfier exits; the rest loop.
    if (untilAfter) {
      const cont: Traverser<unknown>[] = [];

      for (const t of stepped) {
        if (untilMatches(t)) {
          yield t;
        } else {
          cont.push(t);
        }
      }

      frontier = cont;
    } else {
      frontier = stepped;
    }
  }

  // Post-loop yield rules:
  //   - With `until()` (either placement): traversers exit via the until-yield
  //     above; nothing more.
  //   - With post-form emit: every body output was already emitted; nothing more.
  //   - With pre-form emit: pre-emit caught input + intermediates, but the
  //     final body output never had a "next iteration" to be pre-emitted, so
  //     yield it here.
  //   - With no emit: yield the final frontier (the natural repeat result).
  if (!hasUntil && (!hasEmit || emitBefore)) {
    yield* frontier;
  }
};

// `local` runs the sub-plan against each traverser independently, so steps
// like `count()` or `fold()` operate per-traverser instead of over the whole
// stream.
export const localStep = function* (
  stream: Iterable<Traverser<unknown>>,
  plan: Plan,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    yield* applyPlanToStream(plan, [t], graph, ctx);
  }
};

// `branch(test).option(v, plan)...none(plan)` — per traverser, run the test
// plan, take its first result, and route to the matching option's plan
// (deep-equality on `match`), else `default` if present.
export const branchStep = function* (
  stream: Iterable<Traverser<unknown>>,
  test: Plan,
  options: readonly { match: unknown; plan: Plan }[],
  defaultPlan: Plan | undefined,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    let testResult: unknown = undefined;
    let sawResult = false;

    for (const r of applyPlanToStream(test, [t], graph, ctx)) {
      testResult = r.value;
      sawResult = true;
      break;
    }

    let matched: Plan | undefined;

    if (sawResult) {
      for (const opt of options) {
        if (Object.is(opt.match, testResult) || opt.match === testResult) {
          matched = opt.plan;
          break;
        }
      }
    }

    const target = matched ?? defaultPlan;

    if (target) {
      yield* applyPlanToStream(target, [t], graph, ctx);
    }
  }
};
