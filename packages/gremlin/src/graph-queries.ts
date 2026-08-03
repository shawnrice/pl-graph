import type { Edge, Graph, Vertex } from '@lenke/core';

/**
 * Vertex-centric adjacency queries used by the gremlin executor. These belong
 * here, not on `core/Graph`, because "out/in/both" is traversal vocabulary —
 * a different query language (Cypher, GQL) would express adjacency in its own
 * terms over the same underlying `from`/`to` edges.
 *
 * All three read from core's existing indexes
 * (`edgesFromByLabel` / `edgesToByLabel`) for O(1) per-label lookup.
 */

export const outEdgesOf = (
  graph: Graph,
  v: Vertex,
  labels: readonly string[] = [],
): Iterable<Edge> => iterByLabel(graph.edgesFromByLabel.get(v.id), labels);

export const inEdgesOf = (
  graph: Graph,
  v: Vertex,
  labels: readonly string[] = [],
): Iterable<Edge> => iterByLabel(graph.edgesToByLabel.get(v.id), labels);

export const bothEdgesOf = (
  graph: Graph,
  v: Vertex,
  labels: readonly string[] = [],
): Iterable<Edge> =>
  iterBoth(graph.edgesFromByLabel.get(v.id), graph.edgesToByLabel.get(v.id), labels);

// With labels, yield edges per label (in label-arg order). With no labels,
// yield every edge once across all label-buckets — an edge that's indexed
// under multiple labels still only emits once.
const iterByLabel = function* (
  byLabel: Map<string, Set<Edge>> | undefined,
  labels: readonly string[],
): Iterable<Edge> {
  if (!byLabel) {
    return;
  }

  if (labels.length > 0) {
    yield* iterLabeled(byLabel, labels);

    return;
  }

  yield* iterAllDeduped(byLabel);
};

// Named labels are a disjunction over ONE edge, not a bucket-per-name concat:
// an edge carrying both `R` and `S` is in both buckets, so `outE('R','S')` has
// to dedupe or it emits that edge twice. Native walks a single adjacency list
// and asks "does this edge carry any of these types", which yields it once.
// A single name can't collide with itself, so the common case stays allocation-free.
const iterLabeled = function* (
  byLabel: Map<string, Set<Edge>>,
  labels: readonly string[],
): Iterable<Edge> {
  if (labels.length === 1) {
    yield* byLabel.get(labels[0]) ?? [];

    return;
  }

  const seen = new Set<Edge>();

  for (const label of labels) {
    for (const e of byLabel.get(label) ?? []) {
      if (seen.has(e)) {
        continue;
      }

      seen.add(e);

      yield e;
    }
  }
};

const iterAllDeduped = function* (byLabel: Map<string, Set<Edge>>): Iterable<Edge> {
  const seen = new Set<Edge>();

  for (const set of byLabel.values()) {
    for (const e of set) {
      if (seen.has(e)) {
        continue;
      }

      seen.add(e);

      yield e;
    }
  }
};

// Out-edges first, then in-edges, to match TinkerPop's `both`/`bothE` ordering.
// Within each direction, iteration follows label-arg order (or insertion order
// when no labels are given).
const iterBoth = function* (
  fromByLabel: Map<string, Set<Edge>> | undefined,
  toByLabel: Map<string, Set<Edge>> | undefined,
  labels: readonly string[],
): Iterable<Edge> {
  yield* iterByLabel(fromByLabel, labels);

  yield* iterByLabel(toByLabel, labels);
};
