import { ErrorCode, LenkeError } from '@lenke/errors';

import type { Edge } from '../core/Edge.js';
import type { Graph } from '../core/Graph.js';
import { type AlgorithmGen, defineAlgorithm, materializeVertices, YIELD_EVERY } from './async.js';
import type { AlgorithmConfig, AlgorithmRow } from './types.js';

/** A betweenness / closeness result row: `{ node, centrality }`. */
export type CentralityRow = AlgorithmRow<'centrality', number>;

/**
 * A compressed-sparse-row out-adjacency built by scanning ALL edges once in global
 * edge-insertion order (matching native `build_csr`), so both engines traverse a
 * vertex's out-edges in the exact same sequence — the key to byte-identical BFS /
 * Dijkstra exploration and therefore identical Brandes dependency accumulation.
 */
type OutCsr = { off: Int32Array; nbr: Int32Array; w: Float64Array; weighted: boolean };

/** Build the out-adjacency CSR, yielding at checkpoints so the O(E) setup is frame-safe. */
const buildCsr = function* (
  graph: Graph,
  order: readonly { id: string }[],
  index: Map<string, number>,
  config: AlgorithmConfig,
): Generator<void, OutCsr, void> {
  const { edgeLabel, weightProperty } = config;
  const n = order.length;
  const weighted = weightProperty !== undefined;
  const typeOk = (edge: Edge): boolean => edgeLabel === undefined || edge.labels.has(edgeLabel);
  const weightOf = (edge: Edge): number => {
    if (weightProperty === undefined) {
      return 1;
    }

    const x = edge.getProperty(weightProperty);

    return typeof x === 'number' ? x : 0;
  };

  // Weighted closeness/betweenness run Dijkstra, which requires NON-NEGATIVE weights —
  // a negative edge never settles and a NaN strands the search, leaving the distances
  // undefined. Reject any negative/NaN weight up front (matching shortestPath + the
  // native engine), whether or not it lies on a matched-label reachable path.
  if (weighted) {
    for (const edge of graph.edges) {
      const x = edge.getProperty(weightProperty);

      if (typeof x === 'number' && (Number.isNaN(x) || x < 0)) {
        throw new LenkeError(
          `\`weightProperty\` (${weightProperty}) must hold non-negative numbers — ` +
            `Dijkstra does not admit negative weights`,
          { code: ErrorCode.InvalidValue },
        );
      }
    }
  }

  let sinceYield = 0;
  const off = new Int32Array(n + 1);

  for (const edge of graph.edges) {
    if (typeOk(edge)) {
      off[index.get(edge.from.id)! + 1]++;
    }

    if (++sinceYield >= YIELD_EVERY) {
      sinceYield = 0;

      yield;
    }
  }

  for (let v = 0; v < n; v++) {
    off[v + 1] += off[v];
  }

  const nbr = new Int32Array(off[n]);
  const w = new Float64Array(off[n]);
  const cursor = off.slice(0, n);

  for (const edge of graph.edges) {
    if (!typeOk(edge)) {
      continue;
    }

    const src = index.get(edge.from.id)!;
    const pos = cursor[src]++;
    nbr[pos] = index.get(edge.to.id)!;
    w[pos] = weightOf(edge);

    if (++sinceYield >= YIELD_EVERY) {
      sinceYield = 0;

      yield;
    }
  }

  return { off, nbr, w, weighted };
};

/** A binary min-heap entry `[distance, vertexIndex]`, ordered by distance then index. */
type HeapItem = [number, number];

const less = (a: HeapItem, b: HeapItem): boolean => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);

const heapPush = (heap: HeapItem[], item: HeapItem): void => {
  heap.push(item);
  let i = heap.length - 1;

  while (i > 0) {
    const parent = (i - 1) >> 1;

    if (!less(heap[i], heap[parent])) {
      break;
    }

    [heap[i], heap[parent]] = [heap[parent], heap[i]];
    i = parent;
  }
};

const heapPop = (heap: HeapItem[]): HeapItem => {
  const [top] = heap;
  const last = heap.pop()!;

  if (heap.length > 0) {
    heap[0] = last;
    let i = 0;

    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let smallest = i;

      if (l < heap.length && less(heap[l], heap[smallest])) {
        smallest = l;
      }

      if (r < heap.length && less(heap[r], heap[smallest])) {
        smallest = r;
      }

      if (smallest === i) {
        break;
      }

      [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
      i = smallest;
    }
  }

  return top;
};

/** The per-source shortest-path DAG (Brandes bookkeeping): visit stack, path counts,
 *  predecessor lists, and distances — mirroring native `Sssp`. */
type Sssp = { stack: number[]; sigma: Float64Array; pred: number[][]; dist: Float64Array };

/**
 * Single-source shortest paths from `s` over the CSR — unweighted BFS layers or
 * weighted Dijkstra — recording the stack (settle order), `sigma` (path counts) and
 * predecessor lists (in CSR / edge-insertion order). The Dijkstra frontier ties on
 * `(dist, vertex index)`, matching native, so the stack + predecessors are identical.
 */
const sssp = (csr: OutCsr, n: number, s: number): Sssp => {
  const sigma = new Float64Array(n);
  const pred: number[][] = Array.from({ length: n }, () => []);
  const stack: number[] = [];
  const dist = new Float64Array(n).fill(Infinity);
  sigma[s] = 1;
  dist[s] = 0;

  if (csr.weighted) {
    const settled = new Uint8Array(n);
    const heap: HeapItem[] = [[0, s]];

    while (heap.length > 0) {
      const [, v] = heapPop(heap);

      if (settled[v]) {
        continue;
      }

      settled[v] = 1;
      stack.push(v);
      const dv = dist[v];

      for (let j = csr.off[v]; j < csr.off[v + 1]; j++) {
        const to = csr.nbr[j];
        const nd = dv + csr.w[j];

        if (nd < dist[to]) {
          dist[to] = nd;
          sigma[to] = sigma[v];
          pred[to] = [v];
          heapPush(heap, [nd, to]);
        } else if (nd === dist[to]) {
          sigma[to] += sigma[v];
          pred[to].push(v);
        }
      }
    }
  } else {
    const queue: number[] = [s];
    let head = 0;

    while (head < queue.length) {
      const v = queue[head++];
      stack.push(v);
      const dv = dist[v];

      for (let j = csr.off[v]; j < csr.off[v + 1]; j++) {
        const to = csr.nbr[j];

        if (dist[to] === Infinity) {
          dist[to] = dv + 1;
          queue.push(to);
        }

        if (dist[to] === dv + 1) {
          sigma[to] += sigma[v];
          pred[to].push(v);
        }
      }
    }
  }

  return { stack, sigma, pred, dist };
};

export const betweennessGen = function* (
  config: AlgorithmConfig,
  graph: Graph,
): AlgorithmGen<CentralityRow> {
  const order = yield* materializeVertices(graph);
  const n = order.length;

  if (n === 0) {
    return [];
  }

  const index = new Map<string, number>();
  order.forEach((v, i) => index.set(v.id, i));

  const csr = yield* buildCsr(graph, order, index, config);
  const cb = new Float64Array(n);
  let sinceYield = 0;

  // Source set: exact = every vertex; approximate (config.pivots) = a deterministic
  // evenly-spaced sample of `pivots` sources (indices `i*n/k`), identical to native,
  // scaled below by n/k (the Brandes–Pich estimator).
  const { pivots } = config;
  const sources: number[] =
    pivots !== undefined && pivots < n && pivots > 0
      ? Array.from({ length: pivots }, (_, i) => Math.floor((i * n) / pivots))
      : Array.from({ length: n }, (_, i) => i);

  for (const s of sources) {
    const sp = sssp(csr, n, s);
    const delta = new Float64Array(n);

    // Pop in reverse visit order (non-increasing distance): each w's dependency is
    // final before it flows back to its predecessors. Same order as native, so the
    // per-vertex f64 accumulation is byte-identical.
    for (let k = sp.stack.length - 1; k >= 0; k--) {
      const w = sp.stack[k];
      const coeff = 1 + delta[w];

      for (const v of sp.pred[w]) {
        delta[v] += (sp.sigma[v] / sp.sigma[w]) * coeff;
      }

      if (w !== s) {
        cb[w] += delta[w];
      }
    }

    if (++sinceYield >= YIELD_EVERY) {
      sinceYield = 0;

      yield;
    }
  }

  // Scale a sampled run up to a full-graph estimate (exact runs use every source).
  if (sources.length < n) {
    const scale = n / sources.length;

    for (let i = 0; i < n; i++) {
      cb[i] *= scale;
    }
  }

  const rows: CentralityRow[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const v = order[i];
    const score = cb[i];

    if (config.writeProperty !== undefined) {
      v.setProperty(config.writeProperty, score);
    }

    rows[i] = { node: v.id, centrality: score };
  }

  return rows;
};

export const closenessGen = function* (
  config: AlgorithmConfig,
  graph: Graph,
): AlgorithmGen<CentralityRow> {
  const order = yield* materializeVertices(graph);
  const n = order.length;

  if (n === 0) {
    return [];
  }

  const index = new Map<string, number>();
  order.forEach((v, i) => index.set(v.id, i));

  const csr = yield* buildCsr(graph, order, index, config);
  const rows: CentralityRow[] = new Array(n);
  let sinceYield = 0;

  for (let s = 0; s < n; s++) {
    const sp = sssp(csr, n, s);

    // Sum finite distances in vertex-insertion order (source's own 0 adds nothing;
    // unreachable Infinity excluded) — same order as native, so identical.
    let sum = 0;

    for (let v = 0; v < n; v++) {
      const d = sp.dist[v];

      if (Number.isFinite(d)) {
        sum += d;
      }
    }

    const centrality = sum === 0 ? 0 : 1 / sum;
    const vtx = order[s];

    if (config.writeProperty !== undefined) {
      vtx.setProperty(config.writeProperty, centrality);
    }

    rows[s] = { node: vtx.id, centrality };

    if (++sinceYield >= YIELD_EVERY) {
      sinceYield = 0;

      yield;
    }
  }

  return rows;
};

/**
 * Betweenness centrality (Brandes' algorithm) over the directed graph (out-edges,
 * optionally one `edgeLabel`, optionally weighted by `weightProperty`). Each
 * vertex's score is the sum over all ordered pairs `(s,t)` of the fraction of
 * shortest `s→t` paths through it — directed and UNNORMALIZED. Dependencies
 * accumulate in a fixed reverse-visit order, so scores are byte-identical to the
 * native engine. **O(V·E)** (unweighted) — does not scale to very large graphs.
 * Data-last dual-form: `betweenness(config, graph)` / `betweenness(config)(graph)`.
 */
export const betweenness = defineAlgorithm(betweennessGen);

/**
 * Closeness centrality — `1 / Σ_t d(s,t)` over every reachable `t ≠ s`
 * (UNNORMALIZED; a vertex reaching nothing scores 0). Follows out-edges, optionally
 * one `edgeLabel`, unweighted BFS or (with `weightProperty`) Dijkstra. The distance
 * sum is taken in vertex-insertion order, so scores are byte-identical to native.
 * **O(V·E)** — does not scale to very large graphs. Data-last dual-form:
 * `closeness(config, graph)` / `closeness(config)(graph)`.
 */
export const closeness = defineAlgorithm(closenessGen);
