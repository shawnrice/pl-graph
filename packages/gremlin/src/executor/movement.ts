import type { Edge, Graph, Vertex } from '@lenke/core';

import { bothEdgesOf, inEdgesOf, outEdgesOf } from '../graph-queries.js';
import { extend, isEdge, isVertex, type Traverser } from './runtime.js';

export type AdjacencyKind = 'out' | 'in' | 'both' | 'outE' | 'inE' | 'bothE';

export const adjacentEdges = (
  kind: AdjacencyKind,
  graph: Graph,
  v: Vertex,
  labels: readonly string[],
): Iterable<Edge> => {
  switch (kind) {
    case 'out':
    case 'outE':
      return outEdgesOf(graph, v, labels);
    case 'in':
    case 'inE':
      return inEdgesOf(graph, v, labels);
    case 'both':
    case 'bothE':
      return bothEdgesOf(graph, v, labels);
  }
};

export const otherEndpoint = (kind: 'out' | 'in' | 'both', edge: Edge, v: Vertex): Vertex => {
  switch (kind) {
    case 'out':
      return edge.to;
    case 'in':
      return edge.from;
    case 'both':
      return edge.from.id === v.id ? edge.to : edge.from;
  }
};

export const traverseToVertex = function* (
  stream: Iterable<Traverser<unknown>>,
  graph: Graph,
  step: { kind: 'out' | 'in' | 'both'; labels: readonly string[] },
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    // A vertex step on a NON-VERTEX yields nothing — matching the native engine, which
    // is lenient here (unlike the edge→vertex steps below, which fault on a non-edge).
    if (!isVertex(t.value)) {
      continue;
    }

    const v = t.value;

    for (const e of adjacentEdges(step.kind, graph, v, step.labels)) {
      yield extend(t, otherEndpoint(step.kind, e, v));
    }
  }
};

export const traverseToEdge = function* (
  stream: Iterable<Traverser<unknown>>,
  graph: Graph,
  step: { kind: 'outE' | 'inE' | 'bothE'; labels: readonly string[] },
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    if (!isVertex(t.value)) {
      continue;
    }

    const v = t.value;

    for (const e of adjacentEdges(step.kind, graph, v, step.labels)) {
      yield extend(t, e);
    }
  }
};

export const edgeToVertex = function* (
  stream: Iterable<Traverser<unknown>>,
  step: { kind: 'outV' | 'inV' | 'bothV' | 'otherV' },
): Iterable<Traverser<unknown>> {
  for (const t of stream) {
    if (!isEdge(t.value)) {
      continue;
    }

    const e = t.value;

    if (step.kind === 'outV') {
      yield extend(t, e.from);
    } else if (step.kind === 'inV') {
      yield extend(t, e.to);
    } else if (step.kind === 'bothV') {
      yield extend(t, e.from);

      yield extend(t, e.to);
    } else {
      // otherV — find the previous vertex in the path and emit the other endpoint.
      const prev = [...t.path].reverse().find((p): p is Vertex => isVertex(p));
      const other = prev?.id === e.from.id ? e.to : e.from;

      yield extend(t, other);
    }
  }
};
