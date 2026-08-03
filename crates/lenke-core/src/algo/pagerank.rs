//! PageRank (pull model, f64). `pr'[v] = (1−d)/N + d·Σ_{u→v} pr[u]·w(u→v)/S[u] +
//! d·dangling/N`, where `S[u]` is u's out-strength (out-degree, or Σ edge weight
//! when `weightProperty` is set), `dangling = Σ pr[u]` over out-strength-0 vertices,
//! `d` = damping (default 0.85), for a fixed `iterations` (default 20).
//!
//! Cross-engine bit-identity: the summation order of every f64 accumulation is
//! pinned to **global edge-insertion order** (dense eidx == NDJSON order == the TS
//! `edgesById` insertion order). Both engines build the per-target contribution
//! lists by scanning all edges once in that order, so each `pr'[v]` sum adds its
//! terms in the same sequence — making the TS mirror byte-for-byte equal, parallel
//! edges and weights included. The dangling sum is taken in vertex-insertion order,
//! also identical.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::AlgoConfig;
use crate::graph::{Graph, Value};

const DEFAULT_DAMPING: f64 = 0.85;
const DEFAULT_ITERATIONS: u32 = 20;

/// The per-target contribution CSR shared by every PageRank variant. `inc_off[v]..
/// inc_off[v+1]` indexes `(inc_src, inc_fac)` for target v, filled in global
/// edge-insertion order so each target's pull sum is order-canonical (the f64
/// byte-identity contract). `out_strength[u]` is u's out-strength (its dangling
/// test). All arrays are `slots`-sized (dead slots see an empty CSR range).
struct PullGraph {
    out_strength: Vec<f64>,
    inc_off: Vec<usize>,
    inc_src: Vec<u32>,
    inc_fac: Vec<f64>,
}

/// Build the pull CSR once. Two edge sweeps in insertion order (out-strength /
/// slot counts, then the transposed contribution lists) — the exact sequence the
/// TS mirror scans, so factors land in identical order.
fn build_pull_graph(graph: &Graph, cfg: &AlgoConfig) -> PullGraph {
    // Some(None) = every type; Some(Some(t)) = one type; None = unknown type → no
    // edges (every vertex dangling → uniform 1/N).
    let etype = cfg.etype(graph);
    // ANY of the edge's labels — see `algo::edge_type_ok`.
    let type_ok = |ei: usize| match etype {
        Some(Some(t)) => graph.edge_has_label(ei as u32, t),
        Some(None) => true,
        None => false,
    };
    // Precompute per-edge weights ONCE so both build passes index a flat Vec instead
    // of re-resolving the property twice per edge. `None` = unweighted (1.0).
    let weights: Option<Vec<f64>> = cfg
        .weight_property
        .as_deref()
        .map(|k| super::edge_weights(graph, k));
    let weight_of = |ei: usize| -> f64 { weights.as_ref().map_or(1.0, |w| w[ei]) };

    let slots = graph.n;

    // Pass 1: out-strength per source, accumulated in edge-insertion order (serial —
    // the per-source weighted sum's f64 order is part of the byte-identity contract).
    let mut out_strength = vec![0.0f64; slots];
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) {
            out_strength[graph.e_src[ei] as usize] += weight_of(ei);
        }
    }

    // Pass 2: per-target contribution lists as a flat CSR — filled in edge-insertion
    // order so each target's later sum is order-canonical. Flat arrays (vs Vec<Vec>)
    // give the hot pull loop contiguous, cache-friendly reads and one allocation.
    let mut inc_off = vec![0usize; slots + 1];
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) {
            inc_off[graph.e_dst[ei] as usize + 1] += 1;
        }
    }
    for v in 0..slots {
        inc_off[v + 1] += inc_off[v];
    }
    let mut inc_src = vec![0u32; inc_off[slots]];
    let mut inc_fac = vec![0.0f64; inc_off[slots]];
    let mut cursor = inc_off[..slots].to_vec();
    for ei in 0..graph.edge_slots() {
        if !graph.is_edge_live(ei as u32) || !type_ok(ei) {
            continue;
        }
        let src = graph.e_src[ei];
        // A node whose total out-weight is 0 (only reachable in the weighted path,
        // when every out-edge has weight 0) is a DANGLING node: its rank mass is
        // redistributed uniformly via the `dangling` sum below — it must NOT be
        // divided by zero (`weight_of / 0 == 0/0 == NaN`, which would poison every
        // score). Emit a 0 factor so this edge carries no directed mass. The CSR
        // slot is still filled (Pass 1 reserved it), so the summation order stays
        // byte-identical to the unweighted path and to the TS engine. Unweighted
        // never hits this branch: an existing edge implies out_strength > 0.
        let factor = if out_strength[src as usize] == 0.0 {
            0.0
        } else {
            weight_of(ei) / out_strength[src as usize]
        };
        let pos = cursor[graph.e_dst[ei] as usize];
        cursor[graph.e_dst[ei] as usize] += 1;
        inc_src[pos] = src;
        inc_fac[pos] = factor;
    }

    PullGraph {
        out_strength,
        inc_off,
        inc_src,
        inc_fac,
    }
}

pub fn pagerank(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let d = cfg.damping_factor.unwrap_or(DEFAULT_DAMPING);
    let iterations = cfg.iterations.unwrap_or(DEFAULT_ITERATIONS);
    let n = graph.vertex_count();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let slots = graph.n;

    let PullGraph {
        out_strength,
        inc_off,
        inc_src,
        inc_fac,
    } = build_pull_graph(graph, cfg);

    let mut pr = vec![0.0f64; slots];
    for v in graph.vertex_indices() {
        pr[v as usize] = 1.0 / nf;
    }

    for _ in 0..iterations {
        // Dangling mass: Σ pr[u] over out-strength-0 vertices, in vertex order (serial
        // reduction — a parallel one would reorder the f64 sum).
        let mut dangling = 0.0;
        for u in graph.vertex_indices() {
            if out_strength[u as usize] == 0.0 {
                dangling += pr[u as usize];
            }
        }
        let base = (1.0 - d) / nf + d * dangling / nf;

        // Pull: each target's sum is independent and taken in its own fixed CSR
        // order, so parallelizing ACROSS targets keeps every accumulation
        // bit-identical. Dead slots see an empty range → `base` (never read/output).
        let mut next = vec![0.0f64; slots];
        let pull = |v: usize, nv: &mut f64| {
            let mut sum = 0.0;
            for j in inc_off[v]..inc_off[v + 1] {
                sum += pr[inc_src[j] as usize] * inc_fac[j];
            }
            *nv = base + d * sum;
        };
        #[cfg(feature = "parallel")]
        next.par_iter_mut()
            .enumerate()
            .for_each(|(v, nv)| pull(v, nv));
        #[cfg(not(feature = "parallel"))]
        for (v, nv) in next.iter_mut().enumerate() {
            pull(v, nv);
        }
        pr = next;
    }

    graph
        .vertex_indices()
        .map(|v| (v, Value::Num(pr[v as usize])))
        .collect()
}

/// Personalized PageRank / random-walk-with-restart: identical to global PageRank
/// except the random surfer restarts (and dangling mass redistributes) to a **seed
/// set** `cfg.source_nodes` instead of uniformly. The personalization vector `p` is
/// uniform `1/k` over the k distinct, resolvable seeds and 0 elsewhere; the initial
/// rank is `p` too. Unknown seed ids are dropped; if no seed resolves, `p` falls
/// back to uniform `1/N` — i.e. it degenerates to global PageRank. Byte-identical
/// across engines: `p`, the teleport scalar (dangling summed in vertex order), and
/// every pull sum (fixed CSR order) match the TS mirror bit-for-bit.
pub fn personalized_pagerank(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let d = cfg.damping_factor.unwrap_or(DEFAULT_DAMPING);
    let iterations = cfg.iterations.unwrap_or(DEFAULT_ITERATIONS);
    let n = graph.vertex_count();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let slots = graph.n;

    let PullGraph {
        out_strength,
        inc_off,
        inc_src,
        inc_fac,
    } = build_pull_graph(graph, cfg);

    // Resolve seed external ids → dense slots, dedup, drop unknowns (a seed pointing
    // at no live vertex contributes nothing). `seen` keeps the set distinct so a
    // repeated id never double-weights.
    let mut seed_slots: Vec<usize> = Vec::new();
    let mut seen = vec![false; slots];
    if let Some(ids) = &cfg.source_nodes {
        for id in ids {
            if let Some(slot) = graph.vid.get(id) {
                let s = slot as usize;
                if !seen[s] {
                    seen[s] = true;
                    seed_slots.push(s);
                }
            }
        }
    }

    // The personalization vector `p`: uniform `1/k` over the k distinct seeds, or —
    // when no seed resolves — uniform `1/N` (degenerate to global PageRank), keeping
    // mass conservation well-defined.
    let mut p = vec![0.0f64; slots];
    if seed_slots.is_empty() {
        for v in graph.vertex_indices() {
            p[v as usize] = 1.0 / nf;
        }
    } else {
        let share = 1.0 / seed_slots.len() as f64;
        for &s in &seed_slots {
            p[s] = share;
        }
    }

    // Initial rank = the personalization vector (a proper distribution summing to 1).
    let mut pr = p.clone();

    for _ in 0..iterations {
        // Dangling mass: Σ pr[u] over out-strength-0 vertices, in vertex order (serial
        // reduction — a parallel one would reorder the f64 sum).
        let mut dangling = 0.0;
        for u in graph.vertex_indices() {
            if out_strength[u as usize] == 0.0 {
                dangling += pr[u as usize];
            }
        }
        // The restart mass (damping complement + dangling) is redistributed per the
        // personalization vector `p` rather than uniformly, so `teleport * p[v]` is
        // v's share. With `p[v] = 1/N` everywhere this equals global PageRank's base.
        let teleport = (1.0 - d) + d * dangling;

        // Pull: each target's sum is independent and taken in its own fixed CSR order,
        // so parallelizing ACROSS targets keeps every accumulation bit-identical.
        let mut next = vec![0.0f64; slots];
        let pull = |v: usize, nv: &mut f64| {
            let mut sum = 0.0;
            for j in inc_off[v]..inc_off[v + 1] {
                sum += pr[inc_src[j] as usize] * inc_fac[j];
            }
            *nv = teleport * p[v] + d * sum;
        };
        #[cfg(feature = "parallel")]
        next.par_iter_mut()
            .enumerate()
            .for_each(|(v, nv)| pull(v, nv));
        #[cfg(not(feature = "parallel"))]
        for (v, nv) in next.iter_mut().enumerate() {
            pull(v, nv);
        }
        pr = next;
    }

    graph
        .vertex_indices()
        .map(|v| (v, Value::Num(pr[v as usize])))
        .collect()
}
