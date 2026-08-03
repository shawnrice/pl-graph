//! Betweenness and closeness centrality — two shortest-path-based measures over the
//! directed graph (out-edges, optionally one `edgeLabel`, optionally weighted by a
//! `weightProperty`). Both are O(V·E) unweighted / O(V·(E + V log V)) weighted: one
//! single-source shortest-path pass per vertex, so they do NOT scale to very large
//! graphs — see the README perf note.
//!
//! - **Betweenness** (Brandes' algorithm): for each vertex `v`, the sum over all
//!   ordered pairs `(s,t)` of the fraction of shortest `s→t` paths that pass through
//!   `v`. Directed and UNNORMALIZED (no `1/((n-1)(n-2))` scaling). Endpoints score 0
//!   for their own pairs.
//! - **Closeness**: `1 / Σ_t d(s,t)` over every `t` reachable from `s`
//!   (t ≠ s). UNNORMALIZED (not `(reachable-1) / Σ d`). A vertex that reaches nothing
//!   scores 0.
//!
//! Cross-engine bit-identity: both engines build each vertex's out-adjacency by
//! scanning **all edges once in global edge-insertion order** (a CSR, exactly like
//! `pagerank`), so the BFS queue / Dijkstra settle order — and therefore Brandes'
//! stack + predecessor lists — are identical, making every f64 dependency
//! accumulation `δ[v] += (σ[v]/σ[w])·(1+δ[w])` add its terms in the same sequence as
//! the TS mirror. `σ` (shortest-path counts) are integers summed exactly; the
//! closeness distance sum is taken in vertex-insertion order. The Dijkstra frontier
//! breaks ties by `(dist, vertex index)`, matching the TS heap.

use std::collections::{BinaryHeap, VecDeque};

use super::AlgoConfig;
use crate::graph::{Graph, Value};

/// A compressed-sparse-row out-adjacency: `nbr[off[v]..off[v+1]]` are v's
/// out-neighbours (with parallel edge weights in `w`), built in global
/// edge-insertion order so both engines traverse a vertex's edges identically.
struct OutCsr {
    off: Vec<usize>,
    nbr: Vec<u32>,
    w: Vec<f64>,
    weighted: bool,
}

/// Build the out-adjacency CSR, scanning all live edges once in insertion order.
fn build_csr(graph: &Graph, cfg: &AlgoConfig) -> OutCsr {
    let slots = graph.n;
    // Some(None) = every type; Some(Some(t)) = one type; None = named-but-unknown
    // type → no edges match (the graph is treated as edgeless).
    let etype = cfg.etype(graph);
    // ANY of the edge's labels — `e_type` alone is only its FIRST, which made
    // every algorithm here see an edgeless graph when the filtered label was
    // stored second.
    let type_ok = |ei: usize| match etype {
        Some(Some(t)) => graph.edge_has_label(ei as u32, t),
        Some(None) => true,
        None => false,
    };
    let weights: Option<Vec<f64>> = cfg
        .weight_property
        .as_deref()
        .map(|k| super::edge_weights(graph, k));
    let weight_of = |ei: usize| -> f64 { weights.as_ref().map_or(1.0, |w| w[ei]) };

    let mut off = vec![0usize; slots + 1];
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) {
            off[graph.e_src[ei] as usize + 1] += 1;
        }
    }
    for v in 0..slots {
        off[v + 1] += off[v];
    }
    let mut nbr = vec![0u32; off[slots]];
    let mut w = vec![0f64; off[slots]];
    let mut cursor = off[..slots].to_vec();
    for ei in 0..graph.edge_slots() {
        if !graph.is_edge_live(ei as u32) || !type_ok(ei) {
            continue;
        }
        let src = graph.e_src[ei] as usize;
        let pos = cursor[src];
        cursor[src] += 1;
        nbr[pos] = graph.e_dst[ei];
        w[pos] = weight_of(ei);
    }
    OutCsr {
        off,
        nbr,
        w,
        weighted: weights.is_some(),
    }
}

/// A Dijkstra frontier entry ordered as a min-heap on `(dist, idx)` (see
/// `shortest_path::State` — same tie-break so both engines settle identically).
struct State {
    dist: f64,
    idx: u32,
}
impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.idx == other.idx
    }
}
impl Eq for State {}
impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The per-source shortest-path DAG Brandes needs: the settle/visit `stack` (in
/// non-decreasing distance order), shortest-path counts `sigma`, and predecessor
/// lists `pred` (each in global edge-insertion order).
struct Sssp {
    stack: Vec<u32>,
    sigma: Vec<f64>,
    pred: Vec<Vec<u32>>,
    dist: Vec<f64>,
}

/// Single-source shortest paths from `s` over the CSR — unweighted BFS layers or
/// weighted Dijkstra — recording the Brandes bookkeeping. Explores neighbours in
/// CSR (edge-insertion) order, so the stack and predecessor lists are deterministic.
fn sssp(csr: &OutCsr, slots: usize, s: u32) -> Sssp {
    let mut sigma = vec![0f64; slots];
    let mut pred: Vec<Vec<u32>> = vec![Vec::new(); slots];
    let mut stack: Vec<u32> = Vec::new();
    let mut dist = vec![f64::INFINITY; slots];
    sigma[s as usize] = 1.0;
    dist[s as usize] = 0.0;

    if csr.weighted {
        let mut settled = vec![false; slots];
        let mut heap = BinaryHeap::new();
        heap.push(State { dist: 0.0, idx: s });
        while let Some(State { idx: v, .. }) = heap.pop() {
            if settled[v as usize] {
                continue;
            }
            settled[v as usize] = true;
            stack.push(v);
            let dv = dist[v as usize];
            for j in csr.off[v as usize]..csr.off[v as usize + 1] {
                let to = csr.nbr[j];
                let nd = dv + csr.w[j];
                if nd < dist[to as usize] {
                    dist[to as usize] = nd;
                    sigma[to as usize] = sigma[v as usize];
                    pred[to as usize] = vec![v];
                    heap.push(State { dist: nd, idx: to });
                } else if nd == dist[to as usize] {
                    sigma[to as usize] += sigma[v as usize];
                    pred[to as usize].push(v);
                }
            }
        }
    } else {
        let mut queue = VecDeque::new();
        queue.push_back(s);
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let dv = dist[v as usize];
            for j in csr.off[v as usize]..csr.off[v as usize + 1] {
                let to = csr.nbr[j];
                if dist[to as usize].is_infinite() {
                    dist[to as usize] = dv + 1.0;
                    queue.push_back(to);
                }
                if dist[to as usize] == dv + 1.0 {
                    sigma[to as usize] += sigma[v as usize];
                    pred[to as usize].push(v);
                }
            }
        }
    }

    Sssp {
        stack,
        sigma,
        pred,
        dist,
    }
}

/// Betweenness centrality (Brandes). Accumulates each source's dependencies in
/// reverse-stack (non-increasing distance) order — a fixed, deterministic order —
/// so the per-vertex f64 sum is byte-identical to the TS mirror.
pub fn betweenness(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let slots = graph.n;
    let csr = build_csr(graph, cfg);
    let verts: Vec<u32> = graph.vertex_indices().collect();
    let n = verts.len();

    // Source set: exact = every vertex; approximate = a deterministic evenly-spaced
    // sample of `pivots` vertices (indices `i*n/k`), so both engines pick the SAME
    // sources → byte-identical estimate. A sampled run scales the summed
    // dependencies by n/k (the Brandes–Pich estimator).
    let sources: Vec<u32> = match cfg.pivots {
        Some(k) if (k as usize) < n && k > 0 => {
            let k = k as usize;
            (0..k).map(|i| verts[i * n / k]).collect()
        }
        _ => verts.clone(),
    };

    let mut cb = vec![0f64; slots];
    for &s in &sources {
        let sp = sssp(&csr, slots, s);
        let mut delta = vec![0f64; slots];
        // Pop in reverse visit order (non-increasing distance): each w's dependency
        // is final before it flows back to its predecessors.
        for &w in sp.stack.iter().rev() {
            let coeff = 1.0 + delta[w as usize];
            for &v in &sp.pred[w as usize] {
                delta[v as usize] += (sp.sigma[v as usize] / sp.sigma[w as usize]) * coeff;
            }
            if w != s {
                cb[w as usize] += delta[w as usize];
            }
        }
    }

    // Scale a sampled run up to a full-graph estimate. (Exact runs use every
    // source, so `sources.len() == n` and the scale is 1.)
    if sources.len() < n {
        let scale = n as f64 / sources.len() as f64;
        for v in &mut cb {
            *v *= scale;
        }
    }

    graph
        .vertex_indices()
        .map(|v| (v, Value::Num(cb[v as usize])))
        .collect()
}

/// Closeness centrality: `1 / Σ_t d(s,t)` over reachable `t ≠ s`, summed in
/// vertex-insertion order (0 when nothing is reachable). Unnormalized.
pub fn closeness(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let slots = graph.n;
    let csr = build_csr(graph, cfg);

    graph
        .vertex_indices()
        .map(|s| {
            let sp = sssp(&csr, slots, s);
            // Sum finite distances in vertex-insertion order (the source's own 0
            // contributes nothing). INF (unreachable) is excluded.
            let mut sum = 0.0f64;
            for v in graph.vertex_indices() {
                let d = sp.dist[v as usize];
                if d.is_finite() {
                    sum += d;
                }
            }
            let c = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            (s, Value::Num(c))
        })
        .collect()
}
