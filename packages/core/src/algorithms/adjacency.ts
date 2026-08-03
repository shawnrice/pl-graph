import type { Edge } from '../core/Edge.js';

/**
 * One direction of a vertex's edges, optionally restricted to a single edge type
 * — a `graph.edgesFromByLabel` / `edgesToByLabel` entry, which is a
 * `Map<edgeType, Set<Edge>>`.
 *
 * Edges are multi-label and the index buckets an edge under EVERY label it
 * carries, so the unrestricted case has to dedupe: unioning the buckets naively
 * yields a two-label edge twice, and every algorithm below counts one push per
 * edge (parallel edges and self-loops keep their multiplicity, matching the
 * native tally). Native walks a single adjacency list, so a duplicate here is a
 * cross-engine divergence, not merely a wrong number.
 *
 * The restricted case needs no dedupe — an edge is in a given label's bucket at
 * most once — and stays a single map lookup, which is the hot path.
 */
export const incidentEdges = function* (
  byLabel: Map<string, Set<Edge>> | undefined,
  edgeLabel: string | undefined,
): Iterable<Edge> {
  if (byLabel === undefined) {
    return;
  }

  if (edgeLabel !== undefined) {
    yield* byLabel.get(edgeLabel) ?? [];

    return;
  }

  // Only pay for the dedupe when more than one bucket can hold the same edge.
  if (byLabel.size === 1) {
    for (const set of byLabel.values()) {
      yield* set;
    }

    return;
  }

  const seen = new Set<Edge>();

  for (const set of byLabel.values()) {
    for (const edge of set) {
      if (!seen.has(edge)) {
        seen.add(edge);

        yield edge;
      }
    }
  }
};

/** How many distinct edges `incidentEdges` would yield. */
export const countIncidentEdges = (
  byLabel: Map<string, Set<Edge>> | undefined,
  edgeLabel: string | undefined,
): number => {
  if (byLabel === undefined) {
    return 0;
  }

  if (edgeLabel !== undefined) {
    return byLabel.get(edgeLabel)?.size ?? 0;
  }

  if (byLabel.size === 1) {
    let n = 0;

    for (const set of byLabel.values()) {
      n += set.size;
    }

    return n;
  }

  const seen = new Set<Edge>();

  for (const set of byLabel.values()) {
    for (const edge of set) {
      seen.add(edge);
    }
  }

  return seen.size;
};
