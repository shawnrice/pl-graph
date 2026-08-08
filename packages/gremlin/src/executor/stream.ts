import type { Graph } from '@lenke/core';

import type { FlatMapClosure, Plan, ReducerClosure, SideEffectClosure } from '../ast.js';
import { applyPlanToStream } from './dispatch.js';
import { closureView, extend, type RunContext, startTraverser, type Traverser } from './runtime.js';

// `map(plan)` — first output of the sub-plan replaces the traverser value.
// Drops traversers where the sub-plan is empty.
export const mapStep = function* (
  stream: Iterable<Traverser<unknown>>,
  plan: Plan,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    for (const r of applyPlanToStream(plan, [t], graph, ctx)) {
      yield r;
      break;
    }
  }
};

// `flatMap(plan)` — sub-plan's outputs replace the traverser value (0+).
export const flatMapStep = function* (
  stream: Iterable<Traverser<unknown>>,
  plan: Plan,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    yield* applyPlanToStream(plan, [t], graph, ctx);
  }
};

export const flatMapFnStep = function* (
  stream: Iterable<Traverser<unknown>>,
  fn: FlatMapClosure,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    for (const v of fn(t.value, closureView(t, ctx))) {
      yield extend(t, v);
    }
  }
};

export const foldStep = function* (
  stream: Iterable<Traverser<unknown>>,
): Iterable<Traverser<unknown>> {
  const list: unknown[] = [];

  for (const t of stream) {
    list.push(t.value);
  }

  yield startTraverser(list);
};

// `foldFn(seed, reducer)` — barrier step. Reduces the entire stream into a
// single accumulator and yields exactly one traverser carrying it.
export const foldFnStep = function* (
  stream: Iterable<Traverser<unknown>>,
  seed: unknown,
  fn: ReducerClosure,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  let acc = seed;

  for (const t of stream) {
    acc = fn(acc, t.value, closureView(t, ctx));
  }

  yield startTraverser(acc);
};

export const unfoldStream = function* (
  stream: Iterable<Traverser<unknown>>,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    const v = t.value;

    if (v !== null && typeof v !== 'string' && typeof v === 'object' && Symbol.iterator in v) {
      for (const item of v as Iterable<unknown>) {
        yield extend(t, item);
      }
    } else {
      yield t;
    }
  }
};

export const injectMidStream = function* (
  stream: Iterable<Traverser<unknown>>,
  values: readonly unknown[],
): Iterable<Traverser<unknown>> {
  for (const v of values) {
    yield startTraverser(v);
  }

  yield* stream;
};

export const sideEffectStep = function* (
  stream: Iterable<Traverser<unknown>>,
  plan: Plan,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    // Drain the sub-plan for its effect; discard outputs.
    for (const _ of applyPlanToStream(plan, [t], graph, ctx)) {
      // intentionally consume
    }

    yield t;
  }
};

export const sideEffectFnStep = function* (
  stream: Iterable<Traverser<unknown>>,
  fn: SideEffectClosure,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    fn(t.value, closureView(t, ctx));

    yield t;
  }
};

// `index()` — pair each value with its position in the stream.
export const indexStep = function* (
  stream: Iterable<Traverser<unknown>>,
): Iterable<Traverser<unknown>> {
  let i = 0;

  for (const t of stream) {
    yield extend(t, [t.value, i]);
    i++;
  }
};
