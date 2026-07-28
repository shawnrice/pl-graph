import type { Edge } from '../core/Edge.js';
import type { Graph } from '../core/Graph.js';
import type { Vertex } from '../core/Vertex.js';
import { type AlgorithmGen, defineAlgorithm, YIELD_EVERY } from './async.js';
import type { AlgorithmConfig, AlgorithmRow } from './types.js';

/** A neighbour-aggregation result row: `{ node, vector }` (the aggregated vector).
 * The column is `vector` (not `aggregate`, which is a reserved GQL word). */
export type NeighborAggregateRow = AlgorithmRow<'vector', number[]>;

type Op = 'mean' | 'sum' | 'max' | 'min';

/** Read a vertex's feature as a numeric vector, or `null` if absent / not a list of
 * numbers (such a vertex contributes nothing). Mirrors native `read_vec`. */
const readVec = (vertex: Vertex, key: string): number[] | null => {
  const raw = vertex.getProperty(key);

  if (!Array.isArray(raw)) {
    return null;
  }

  const out: number[] = [];

  for (const x of raw) {
    if (typeof x !== 'number') {
      return null; // a non-numeric element → not a feature vector
    }

    out.push(x);
  }

  return out;
};

/** Fold one contributor vector into `acc` under `op`, scaled by `coef` (edge weight ×
 * normalization; 1 unweighted). `started` marks whether `acc` holds a real value yet (the
 * identity for max/min is the first contributor). `max`/`min` ignore `coef` (scale-
 * independent; a weight/norm is rejected at the call site). Mirrors native `fold`. */
const fold = (op: Op, acc: number[], vec: number[], started: boolean, coef: number): void => {
  for (let i = 0; i < acc.length; i++) {
    if (op === 'sum' || op === 'mean') {
      acc[i] += coef * vec[i];
    } else if (!started) {
      acc[i] = vec[i];
    } else if (op === 'max') {
      acc[i] = Math.max(acc[i], vec[i]);
    } else {
      acc[i] = Math.min(acc[i], vec[i]);
    }
  }
};

type Contributor = { eidx: number; nbr: string };

/** Validate `op` (raw so the message shows the offending value, not `never`). */
const resolveOp = (raw: string): Op => {
  if (raw !== 'mean' && raw !== 'sum' && raw !== 'max' && raw !== 'min') {
    throw new Error(`neighborAggregate \`op\` must be one of mean|sum|max|min, got '${raw}'`);
  }

  return raw;
};

/** Validate `direction` into `[wantOut, wantIn]`. */
const resolveDirs = (raw: string): [boolean, boolean] => {
  if (raw !== 'out' && raw !== 'in' && raw !== 'both') {
    throw new Error(`neighborAggregate \`direction\` must be one of out|in|both, got '${raw}'`);
  }

  return [raw === 'out' || raw === 'both', raw === 'in' || raw === 'both'];
};

/** Validate `norm` into a GCN flag. */
const resolveGcn = (raw: string): boolean => {
  if (raw !== 'none' && raw !== 'gcn') {
    throw new Error(`neighborAggregate \`norm\` must be one of none|gcn, got '${raw}'`);
  }

  return raw === 'gcn';
};

/** Per-edge weights by edge-insertion index (= native eidx); a missing / non-numeric
 * value is 0, mirroring native `edge_weights`. */
const buildEdgeWeights = (graph: Graph, key: string): number[] => {
  const out: number[] = [];

  for (const edge of graph.edges) {
    const w = edge.getProperty(key);
    out.push(typeof w === 'number' ? w : 0);
  }

  return out;
};

/** Every vertex's feature vector (by id), plus the shared dimension `d`; a length
 * mismatch faults. Iterated in insertion (= dense-id) order. */
const buildFeatures = (
  graph: Graph,
  feature: string,
): { feats: Map<string, number[] | null>; d: number } => {
  const feats = new Map<string, number[] | null>();
  let dim: number | undefined;

  for (const vertex of graph.vertices) {
    const vec = readVec(vertex, feature);

    feats.set(vertex.id, vec);

    if (vec !== null && dim === undefined) {
      dim = vec.length;
    } else if (vec !== null && dim !== vec.length) {
      throw new Error(
        `neighborAggregate feature vectors must all have the same length; found ${dim} and ${vec.length}`,
      );
    }
  }

  return { feats, d: dim ?? 0 };
};

/** Per-vertex out/in adjacency in edge-insertion (= native eidx) order. */
const buildAdjacency = (
  graph: Graph,
  edgeLabel: string | undefined,
): { outAdj: Map<string, Contributor[]>; inAdj: Map<string, Contributor[]> } => {
  const outAdj = new Map<string, Contributor[]>();
  const inAdj = new Map<string, Contributor[]>();
  const typeOk = (edge: Edge): boolean => edgeLabel === undefined || edge.labels.has(edgeLabel);
  let eidx = 0;

  for (const edge of graph.edges) {
    if (typeOk(edge)) {
      const from = edge.from.id;
      const to = edge.to.id;
      (outAdj.get(from) ?? outAdj.set(from, []).get(from)!).push({ eidx, nbr: to });
      (inAdj.get(to) ?? inAdj.set(to, []).get(to)!).push({ eidx, nbr: from });
    }

    eidx++;
  }

  return { outAdj, inAdj };
};

/** Contributors of `v` by direction, sorted by edge index (the canonical,
 * engine-independent accumulation order). A `both`-direction self-loop counts once. */
const gatherContributors = (
  v: string,
  outAdj: Map<string, Contributor[]>,
  inAdj: Map<string, Contributor[]>,
  wantOut: boolean,
  wantIn: boolean,
): Contributor[] => {
  const contrib: Contributor[] = [];

  if (wantOut) {
    contrib.push(...(outAdj.get(v) ?? []));
  }

  if (wantIn) {
    for (const a of inAdj.get(v) ?? []) {
      if (!(wantOut && a.nbr === v)) {
        contrib.push(a);
      }
    }
  }

  return contrib.sort((x, y) => x.eidx - y.eidx);
};

export const computeGen = function* (
  config: AlgorithmConfig,
  graph: Graph,
): AlgorithmGen<NeighborAggregateRow> {
  const { feature, edgeLabel, direction = 'both', writeProperty, weightProperty } = config;

  if (feature === undefined) {
    throw new Error('neighborAggregate requires a `feature` property');
  }

  const op = resolveOp(config.op ?? 'mean');
  const [wantOut, wantIn] = resolveDirs(direction);
  const includeSelf = config.includeSelf ?? false;
  const gcn = resolveGcn(config.norm ?? 'none');
  // A weight or `gcn` norm SCALES contributors — meaningless for order/scale-independent
  // max/min. Reject loudly rather than silently ignore. Mirrors native.
  const weighted = weightProperty !== undefined;

  if ((weighted || gcn) && (op === 'max' || op === 'min')) {
    throw new Error(
      'neighborAggregate `weightProperty`/`norm` apply only to op=sum|mean (max/min are scale-independent)',
    );
  }

  const { feats, d } = buildFeatures(graph, feature);
  const { outAdj, inAdj } = buildAdjacency(graph, edgeLabel);
  const edgeWeights = weighted ? buildEdgeWeights(graph, weightProperty) : [];

  // GCN degree per vertex: contributor count under the configured direction/etype (+ the
  // self-loop when includeSelf), floored at 1. Mirrors native `deg`.
  const deg = new Map<string, number>();

  if (gcn) {
    for (const vertex of graph.vertices) {
      const c = gatherContributors(vertex.id, outAdj, inAdj, wantOut, wantIn).length;
      deg.set(vertex.id, Math.max(c + (includeSelf ? 1 : 0), 1));
    }
  }

  const rows: NeighborAggregateRow[] = [];
  let sinceYield = 0;

  for (const vertex of graph.vertices) {
    const v = vertex.id;
    const acc = new Array<number>(d).fill(0);
    let coefSum = 0;
    let started = false;

    // Self first (when included), with GCN factor 1/deg_i (= 1/sqrt(deg_i·deg_i)).
    if (includeSelf) {
      const sv = feats.get(v) ?? null;

      if (sv !== null) {
        const coef = gcn ? 1 / deg.get(v)! : 1;
        fold(op, acc, sv, started, coef);
        started = true;
        coefSum += coef;
      }
    }

    // Then neighbours in edge-index order, each scaled by edge weight × GCN factor.
    for (const { eidx, nbr } of gatherContributors(v, outAdj, inAdj, wantOut, wantIn)) {
      const nv = feats.get(nbr) ?? null;

      if (nv !== null) {
        const w = weighted ? edgeWeights[eidx] : 1;
        const nf = gcn ? 1 / Math.sqrt(deg.get(v)! * deg.get(nbr)!) : 1;
        const coef = w * nf;
        fold(op, acc, nv, started, coef);
        started = true;
        coefSum += coef;
      }
    }

    if (op === 'mean' && coefSum !== 0) {
      for (let i = 0; i < d; i++) {
        acc[i] /= coefSum;
      }
    }

    if (writeProperty !== undefined) {
      vertex.setProperty(writeProperty, acc);
    }

    rows.push({ node: vertex.id, vector: acc });

    if (++sinceYield >= YIELD_EVERY) {
      sinceYield = 0;

      yield;
    }
  }

  return rows;
};

/**
 * Vectorized neighbour aggregation — for each vertex, aggregate its neighbours'
 * list-valued `feature` vectors element-wise over the whole block (`op` = mean /
 * sum / max / min), by `direction`, optionally including the vertex's own vector.
 * Byte-identical to native `neighborAggregate`: contributors fold in ascending
 * edge-index order (self first when included). Data-last dual-form.
 */
export const neighborAggregate = defineAlgorithm(computeGen);
