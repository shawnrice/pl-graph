// --- Mutation helpers --------------------------------------------------
//
// `addV` / `addE` / `property` / `drop` mutate the graph in place. The
// underlying `Graph.addVertex` / `addEdge` / `removeVertex` / `removeEdge`
// methods emit events, so subscribers see changes as they happen during
// traversal. Callers who need a transactional "all or nothing" semantic
// should clone the graph first (`graph.clone()`).

import type { Graph, Vertex } from '@lenke/core';
import { ErrorCode, LenkeError } from '@lenke/errors';

import type { AddEEndpoint, Plan } from '../ast.js';
import { isPlan } from '../steps/framework.js';
import { applyPlanToStream, applyStep } from './dispatch.js';
import {
  extend,
  isEdge,
  isVertex,
  type PropertyElement,
  propertyOwner,
  type RunContext,
  type Traverser,
} from './runtime.js';
import { applySource } from './sources.js';

export const addVStep = function* (
  stream: Iterable<Traverser<unknown>>,
  graph: Graph,
  label: string | undefined,
): Iterable<Traverser<unknown>> {
  // Snapshot the input before mutating: `g.V()` yields from a *live* view over
  // the vertex map, so adding vertices mid-iteration would feed the new
  // vertices back in and loop forever. Buffering matches TinkerPop's semantics
  // (one addition per pre-existing input).
  // oxlint-disable-next-line unicorn/no-useless-spread -- snapshot the live source before mutating
  for (const t of [...stream]) {
    const v = graph.addVertex({
      labels: label ? [label] : [],
      properties: {},
    });

    yield extend(t, v);
  }
};

// Run an AddE endpoint sub-plan. The sub-plan may start with a source step
// (`V('2')`, `inject(...)`) or may be rooted at the current traverser. We
// detect the source case and route through `applySource` accordingly so
// that `addE('X').to(V('2'))` works alongside `addE('X').to(out('knows'))`.
export const runEndpointPlan = (
  plan: Plan,
  graph: Graph,
  ctx: RunContext,
  rooted: Traverser<unknown>,
): Iterable<Traverser<unknown>> => {
  if (plan.steps.length === 0) {
    return [rooted];
  }

  const [first] = plan.steps;

  if (first.kind === 'V' || first.kind === 'E' || first.kind === 'inject') {
    let stream: Iterable<Traverser<unknown>> = applySource(first, graph);

    for (let i = 1; i < plan.steps.length; i++) {
      stream = applyStep(plan.steps[i], stream, graph, ctx);
    }

    return stream;
  }

  return applyPlanToStream(plan, [rooted], graph, ctx);
};

export const resolveAddEEndpoint = (
  endpoint: AddEEndpoint | undefined,
  t: Traverser<unknown>,
  graph: Graph,
  ctx: RunContext,
): Vertex | null => {
  if (endpoint === undefined) {
    return isVertex(t.value) ? t.value : null;
  }

  if (endpoint.kind === 'tag') {
    // Pop.last semantics — most recent tagged value wins.
    const list = t.tags.get(endpoint.label);

    if (!list || list.length === 0) {
      return null;
    }

    const v = list[list.length - 1];

    return isVertex(v) ? v : null;
  }

  for (const result of runEndpointPlan(endpoint.plan, graph, ctx, t)) {
    return isVertex(result.value) ? result.value : null;
  }

  return null;
};

export const addEStep = function* (
  stream: Iterable<Traverser<unknown>>,
  graph: Graph,
  step: { label: string; from?: AddEEndpoint; to?: AddEEndpoint },
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  // Snapshot before mutating — an `E()` source iterates a live edge view.
  // oxlint-disable-next-line unicorn/no-useless-spread -- snapshot the live source before mutating
  for (const t of [...stream]) {
    if (step.from === undefined && step.to === undefined) {
      throw new LenkeError(
        `addE('${step.label}'): at least one of .from() or .to() must be specified`,
        {
          code: ErrorCode.Syntax,
        },
      );
    }

    const from = resolveAddEEndpoint(step.from, t, graph, ctx);
    const to = resolveAddEEndpoint(step.to, t, graph, ctx);

    if (!from || !to) {
      throw new LenkeError(
        `addE('${step.label}'): could not resolve endpoint vertices (from=${!!from}, to=${!!to})`,
        { code: ErrorCode.MissingVertex },
      );
    }

    const e = graph.addEdge({
      from,
      to,
      labels: [step.label],
      properties: {},
    });

    yield extend(t, e);
  }
};

export const propertyStep = function* (
  stream: Iterable<Traverser<unknown>>,
  key: string,
  value: unknown,
  graph: Graph,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  const asPlan = isPlan(value) ? value : undefined;

  for (const t of stream) {
    const el = t.value;

    // `property` only makes sense on a vertex/edge — non-elements are dropped.
    if (!isVertex(el) && !isEdge(el)) {
      continue;
    }

    let v = value;

    if (asPlan) {
      // A traversal value re-evaluates per element, rooted at the CURRENT
      // traverser (preserving tags, so it can `select(...)` an outer label —
      // matching native `sub_vals(&t)`). Its first output is the value; no
      // output leaves the property unset and drops the traverser.
      const first = applyPlanToStream(asPlan, [t], graph, ctx)[Symbol.iterator]().next();

      if (first.done) {
        continue;
      }

      v = first.value.value;
    }

    el.setProperty(key, v);

    yield t;
  }
};

// eslint-disable-next-line require-yield -- drop is a sink: drains the stream and emits nothing.
export const dropStep = function* (
  stream: Iterable<Traverser<unknown>>,
  graph: Graph,
): Iterable<Traverser<unknown>> {
  // Snapshot before mutating — `V()`/`E()` sources iterate live views, and a
  // drop mid-iteration both invalidates the iterator and (via edge cascade)
  // can hide elements the traversal still needs to reach.
  // oxlint-disable-next-line unicorn/no-useless-spread -- snapshot the live source before mutating
  for (const t of [...stream]) {
    const v = t.value;

    if (isVertex(v)) {
      graph.removeVertex(v);
    } else if (isEdge(v)) {
      graph.removeEdge(v);
    } else {
      // `.properties(k).drop()` — delete the property from its owner. This is the
      // ONLY Gremlin-native way to delete a property, because `property(k, null)`
      // STORES a present null (a deliberate divergence from TinkerPop, which has
      // no null property values). The owner comes from the side WeakMap.
      const owner = propertyOwner(v);

      if (owner) {
        owner.removeProperty((v as PropertyElement).key);
      }
    }
    // `drop` is a sink — emit nothing for any traverser regardless of type.
  }
};
