import type { Graph } from '../core/Graph.js';
import { countIncidentEdges } from './adjacency.js';
import { type AlgorithmGen, defineAlgorithm, YIELD_EVERY } from './async.js';
import type { AlgorithmConfig, AlgorithmRow } from './types.js';

/** A degree-centrality result row: `{ node, degree }`. */
export type DegreeRow = AlgorithmRow<'degree', number>;

export const computeGen = function* (
  config: AlgorithmConfig,
  graph: Graph,
): AlgorithmGen<DegreeRow> {
  const { edgeLabel, direction = 'out', writeProperty } = config;
  const rows: DegreeRow[] = [];
  let sinceYield = 0;

  // Insertion order (= native dense-vertex-id order), so results are row-identical.
  for (const vertex of graph.vertices) {
    const out = () => countIncidentEdges(graph.edgesFromByLabel.get(vertex.id), edgeLabel);
    const inc = () => countIncidentEdges(graph.edgesToByLabel.get(vertex.id), edgeLabel);
    // "both" sums out + in — a self-loop counts once each way, matching native.
    let degree = out();

    if (direction === 'in') {
      degree = inc();
    } else if (direction === 'both') {
      degree = out() + inc();
    }

    if (writeProperty !== undefined) {
      vertex.setProperty(writeProperty, degree);
    }

    rows.push({ node: vertex.id, degree });

    if (++sinceYield >= YIELD_EVERY) {
      sinceYield = 0;

      yield;
    }
  }

  return rows;
};

/**
 * Degree centrality — per-vertex count of incident edges (out by default, in, or
 * both), optionally over a single `edgeLabel`. O(V + E), in insertion order.
 * Resolves `Promise<DegreeRow[]>` without blocking the event loop (yields every
 * {@link YIELD_EVERY} vertices). Data-last dual-form: `degree(config, graph)` or
 * `degree(config)(graph)`.
 */
export const degree = defineAlgorithm(computeGen);
