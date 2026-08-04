//! Single-source shortest path from a `source` external id, following edges in the
//! configured `direction` (`out` default / `in` / `both`, mirroring `degree`;
//! `both` = undirected), optionally of one type. Unweighted → BFS integer hop
//! distance; weighted (a
//! `weightProperty` is set) → Dijkstra f64 distance. Returns `{node, distance}` for
//! every reachable vertex (including the source at 0), in vertex-insertion order.
//!
//! Cross-engine identity: a vertex's shortest distance is the canonical minimum
//! over all path costs — BFS layer distances are unique integers, and Dijkstra's
//! settled distance is the minimum path float-sum — so both engines produce the
//! same distances. The priority queue breaks ties by (distance, then vertex index)
//! so the *exploration* order is identical too, keeping even float-pathological
//! graphs byte-identical. Unknown/absent source → no rows.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use super::AlgoConfig;
use crate::graph::{Adj, Graph, Value};

/// A Dijkstra frontier entry ordered as a min-heap on `(dist, idx)`: `BinaryHeap`
/// is a max-heap, so `Ord` is reversed — the smallest distance (then smallest
/// vertex index) compares greatest and pops first.
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
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Which incident edges the traversal follows, from `AlgorithmConfig.direction`
/// (default `out`, mirroring `degree`); `both` treats the graph as undirected.
#[derive(Clone, Copy)]
enum Dir {
    Out,
    In,
    Both,
}

impl Dir {
    fn of(cfg: &AlgoConfig) -> Self {
        match cfg.direction.as_deref() {
            Some("in") => Self::In,
            Some("both") => Self::Both,
            _ => Self::Out,
        }
    }
}

/// Visit each incident edge of `u` in the configured direction. `a.nbr` is the far
/// endpoint in every case (target of an out-edge, source of an in-edge), so callers
/// relax toward `a.nbr` uniformly. `both` visits out-edges then in-edges — both
/// engines share this order so Dijkstra's `(dist, idx)` tie-break stays identical.
fn visit_adj(graph: &Graph, u: u32, dir: Dir, f: &mut impl FnMut(&Adj)) {
    match dir {
        Dir::Out => {
            for a in graph.out_adj(u) {
                f(&a);
            }
        }
        Dir::In => {
            for a in graph.in_adj(u) {
                f(&a);
            }
        }
        Dir::Both => {
            for a in graph.out_adj(u) {
                f(&a);
            }
            for a in graph.in_adj(u) {
                f(&a);
            }
        }
    }
}

pub fn shortest_path(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    // Resolve the source external id → dense id; unknown/absent → no reachable set.
    let Some(src) = cfg.source.as_deref().and_then(|s| graph.vid.get(s)) else {
        return Vec::new();
    };
    // A named-but-unknown edge type → only the source is reachable (no edges).
    let etype = cfg.etype(graph);
    // Whether any edge carries the wanted type as a NON-first label, resolved
    // once. When it does not — which is every graph with single-type edges — the
    // adjacency entry alone decides and `graph` is never dereferenced in the
    // walk. Consulting it per edge instead cost 1.08x on a 1M/8M BFS.
    let need_extra = matches!(etype, Some(Some(t)) if graph.etypes_need_extra_lookup(&[t]));
    let passes = |a: &Adj| match etype {
        Some(Some(t)) => a.etype == t || (need_extra && graph.edge_has_label(a.eidx, t)),
        Some(None) => true,
        None => false,
    };

    let slots = graph.n;
    let dir = Dir::of(cfg);
    // Precompute per-edge weights once (weighted runs read them per relaxation, so a
    // hashed property lookup there would dominate). `None` = unweighted.
    let weights: Option<Vec<f64>> = cfg
        .weight_property
        .as_deref()
        .map(|k| super::edge_weights(graph, k));

    // A* is a goal-directed backend: given a `target`, it returns just the source→
    // target distance (identical to Dijkstra's, so interchangeable), exploring far
    // fewer vertices via the admissible `heuristicProperty`.
    if cfg.algorithm.as_deref() == Some("astar") {
        let Some(tgt) = cfg.target.as_deref().and_then(|t| graph.vid.get(t)) else {
            return Vec::new();
        };
        return match astar(graph, src, tgt, slots, cfg, weights.as_deref(), &passes) {
            Some(d) => vec![(tgt, Value::Num(d))],
            None => Vec::new(),
        };
    }

    // Full SSSP (the default): unweighted BFS layers, or weighted Dijkstra.
    let dist = match weights.as_deref() {
        None => bfs(graph, src, slots, dir, &passes),
        Some(w) => dijkstra(graph, src, slots, dir, w, &passes),
    };

    // A `target` restricts the result to that one vertex's distance (like A*),
    // instead of a row per reachable vertex. Unknown/unreachable target → no rows.
    if let Some(t) = cfg.target.as_deref() {
        let Some(tgt) = graph.vid.get(t).filter(|&v| dist[v as usize].is_finite()) else {
            return Vec::new();
        };
        return vec![(tgt, Value::Num(dist[tgt as usize]))];
    }

    graph
        .vertex_indices()
        .filter(|&v| dist[v as usize].is_finite())
        .map(|v| (v, Value::Num(dist[v as usize])))
        .collect()
}

/// Goal-directed A\*: explore by `f = g + h`, where `h` is the admissible estimate
/// to `tgt` read from each vertex's `heuristicProperty` (absent → 0, degrading to
/// Dijkstra). Returns `Some(distance)` when `tgt` is settled (its `g` is then
/// optimal — identical to Dijkstra), `None` if unreachable. Edge weights come from
/// `weightProperty` (absent → unit weights). Same `(priority, idx)` tie-break as
/// Dijkstra, so native and TS explore identically.
fn astar(
    graph: &Graph,
    src: u32,
    tgt: u32,
    slots: usize,
    cfg: &AlgoConfig,
    weights: Option<&[f64]>,
    passes: &impl Fn(&Adj) -> bool,
) -> Option<f64> {
    let dir = Dir::of(cfg);
    let hkey = cfg.heuristic_property.as_deref();
    let h = |v: u32| -> f64 {
        match hkey {
            None => 0.0,
            Some(k) => match graph.props.value(v as usize, k, &graph.strs) {
                Value::Num(x) => x,
                _ => 0.0,
            },
        }
    };
    let weight = |a: &Adj| -> f64 { weights.map_or(1.0, |w| w[a.eidx as usize]) };

    let mut g = vec![f64::INFINITY; slots];
    let mut closed = vec![false; slots];
    g[src as usize] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        dist: h(src),
        idx: src,
    });
    while let Some(State { idx: u, .. }) = heap.pop() {
        if closed[u as usize] {
            continue;
        }
        closed[u as usize] = true;
        if u == tgt {
            return Some(g[u as usize]);
        }
        visit_adj(graph, u, dir, &mut |a| {
            if !passes(a) || closed[a.nbr as usize] {
                return;
            }
            let ng = g[u as usize] + weight(a);
            if ng < g[a.nbr as usize] {
                g[a.nbr as usize] = ng;
                heap.push(State {
                    dist: ng + h(a.nbr),
                    idx: a.nbr,
                });
            }
        });
    }
    None
}

/// Unweighted BFS hop distance (as f64), `INFINITY` for unreached.
fn bfs(
    graph: &Graph,
    src: u32,
    slots: usize,
    dir: Dir,
    passes: &impl Fn(&Adj) -> bool,
) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; slots];
    dist[src as usize] = 0.0;
    let mut queue = VecDeque::new();
    queue.push_back(src);
    while let Some(u) = queue.pop_front() {
        let du = dist[u as usize];
        visit_adj(graph, u, dir, &mut |a| {
            if passes(a) && dist[a.nbr as usize].is_infinite() {
                dist[a.nbr as usize] = du + 1.0;
                queue.push_back(a.nbr);
            }
        });
    }
    dist
}

/// Weighted Dijkstra f64 distance, `INFINITY` for unreached. `weights` is the
/// precomputed per-edge weight (indexed by edge id); negative weights are out of
/// contract (Dijkstra).
fn dijkstra(
    graph: &Graph,
    src: u32,
    slots: usize,
    dir: Dir,
    weights: &[f64],
    passes: &impl Fn(&Adj) -> bool,
) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; slots];
    dist[src as usize] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(State {
        dist: 0.0,
        idx: src,
    });
    while let Some(State { dist: du, idx: u }) = heap.pop() {
        // Skip a stale entry (a shorter distance was already settled).
        if du > dist[u as usize] {
            continue;
        }
        visit_adj(graph, u, dir, &mut |a| {
            if !passes(a) {
                return;
            }
            let nd = du + weights[a.eidx as usize];
            if nd < dist[a.nbr as usize] {
                dist[a.nbr as usize] = nd;
                heap.push(State {
                    dist: nd,
                    idx: a.nbr,
                });
            }
        });
    }
    dist
}
