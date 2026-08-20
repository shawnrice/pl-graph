import { Graph, type Vertex } from '@lenke/core';

import { isEdge, type RunContext, startTraverser, type Traverser } from './runtime.js';

export const aggregateStep = (
  stream: Iterable<Traverser<unknown>>,
  key: string,
  ctx: RunContext,
  eager: boolean,
): Iterable<Traverser<unknown>> => {
  if (!ctx.sideEffects.has(key)) {
    ctx.sideEffects.set(key, []);
  }

  const bucket = ctx.sideEffects.get(key)!;

  if (eager) {
    // `aggregate('x')` is `aggregate(Scope.global, 'x')` — a COLLECTING BARRIER in
    // TinkerPop: it drains the ENTIRE upstream into the side-effect at THIS point in the
    // pipeline (a plain function that materializes `stream` NOW, not a lazy generator),
    // so a downstream `limit(0)` — or an `inject`/branch step that yields early — cannot
    // cancel the collection. `V().aggregate('x').inject(1).limit(0).cap('x')` is therefore
    // the 6 vertices, not `[]`. A lazy generator here would never run: `limit(0)` never
    // pulls it. `store` stays LAZY (the non-barrier form).
    const buffered = [...stream];

    for (const t of buffered) {
      bucket.push(t.value);
    }

    return buffered;
  }

  // `store` — lazy, per-traverser, no barrier.
  return (function* () {
    for (const t of stream) {
      bucket.push(t.value);

      yield t;
    }
  })();
};

// Force eager materialization of the upstream stream. With v2's lack of
// bulk traversers there's nothing to collapse — this just guarantees the
// upstream side-effects have been driven to completion before downstream
// reads them.
export const barrierStep = function* (
  stream: Iterable<Traverser<unknown>>,
): Iterable<Traverser<unknown>> {
  const buffered = [...stream];

  yield* buffered;
};

// `subgraph(key)` is a side-effect over EDGE traversers: each edge (with its two
// endpoints) is added to a named `Graph` accumulator, then `cap(key)` returns it.
// Traversers pass through unchanged, so it composes mid-stream
// (`outE().subgraph(k).inV()…`). addVertex/addEdge dedupe by id, so repeated or
// chained accumulation into the same key is idempotent per element.
export const subgraphStep = function* (
  stream: Iterable<Traverser<unknown>>,
  key: string,
  ctx: RunContext,
): Iterable<Traverser<unknown>> {
  let sg = ctx.subgraphs.get(key);

  if (!sg) {
    sg = new Graph();
    ctx.subgraphs.set(key, sg);
  }

  const endpoint = (v: Vertex): Vertex =>
    sg.getVertexById(v.id) ??
    sg.addVertex({ id: v.id, labels: [...v.labels], properties: { ...v.properties } });

  for (const t of stream) {
    const e = t.value;

    if (isEdge(e)) {
      const from = endpoint(e.from);
      const to = endpoint(e.to);
      sg.addEdge({ id: e.id, from, to, labels: [...e.labels], properties: { ...e.properties } });
    }

    yield t;
  }
};

// `cap(key)` is a barrier: drain the upstream stream first (which is what
// populates side-effect bags via `aggregate`/`subgraph`), then yield the
// captured value — the accumulated `Graph` for a subgraph key, else the bag.
export const capStep = function* (
  stream: Iterable<Traverser<unknown>>,
  ctx: RunContext,
  key: string,
): Iterable<Traverser<unknown>> {
  for (const _ of stream) {
    // intentionally drain
  }

  if (ctx.subgraphs.has(key)) {
    yield startTraverser(ctx.subgraphs.get(key)!);

    return;
  }

  yield startTraverser(ctx.sideEffects.get(key) ?? []);
};
