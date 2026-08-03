//! Peer Pressure community detection (TinkerPop's `PeerPressureVertexProgram`). Every
//! vertex starts in its own cluster (its external id). Each vertex casts a vote for
//! its cluster along its **out-edges** with strength `1/out-degree`, and each
//! synchronous round every vertex adopts the cluster carrying the highest total
//! **incoming** vote energy — ties broken by the smallest cluster external-id string.
//! Iterates until a round changes nothing (or a fixed `iterations` cap, default 30).
//!
//! Unlike plain label propagation (equal-weight neighbour counts) the vote is
//! weighted by `1/out-degree`, so a high-degree hub's endorsement is diluted. Vote
//! energies are f64; to stay byte-identical to the TS mirror each vertex's per-cluster
//! sums are accumulated over its in-edges in **global edge-insertion (eidx) order**
//! (the same canonical order PageRank uses), and the winner is chosen by (energy, then
//! smallest cluster string) — order-independent — so parallel/serial/TS all agree.

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::AlgoConfig;
use crate::graph::{Graph, Value};

const DEFAULT_ITERATIONS: u32 = 30;

pub fn peer_pressure(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let iterations = cfg.iterations.unwrap_or(DEFAULT_ITERATIONS);
    let slots = graph.n;

    // cluster[v] = the dense id of the vertex whose external id is v's current cluster;
    // starts as v itself. A named-but-unknown edge type → no votes → every vertex stays
    // its own cluster.
    let mut cluster: Vec<u32> = (0..slots as u32).collect();
    let Some(etype) = cfg.etype(graph) else {
        return graph
            .vertex_indices()
            .map(|v| (v, Value::Str(graph.vid.arc(v))))
            .collect();
    };
    // ANY of the edge's labels — see `algo::edge_type_ok`.
    let type_ok = |ei: usize| etype.is_none_or(|t| graph.edge_has_label(ei as u32, t));

    // Out-degree → each vertex's per-vote strength; in-degree → sizes the in-CSR.
    let mut out_degree = vec![0u32; slots];
    let mut in_degree = vec![0usize; slots];
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) {
            out_degree[graph.e_src[ei] as usize] += 1;
            in_degree[graph.e_dst[ei] as usize] += 1;
        }
    }
    let vote: Vec<f64> = out_degree
        .iter()
        .map(|&d| if d > 0 { 1.0 / d as f64 } else { 0.0 })
        .collect();

    // In-CSR: `in_src[in_off[u]..in_off[u+1]]` are u's in-neighbour source ids, filled
    // in edge-insertion order so each vertex's later energy sum is order-canonical.
    let mut in_off = vec![0usize; slots + 1];
    for v in 0..slots {
        in_off[v + 1] = in_off[v] + in_degree[v];
    }
    let mut in_src = vec![0u32; in_off[slots]];
    let mut cursor = in_off[..slots].to_vec();
    for ei in 0..graph.edge_slots() {
        if graph.is_edge_live(ei as u32) && type_ok(ei) {
            let dst = graph.e_dst[ei] as usize;
            in_src[cursor[dst]] = graph.e_src[ei];
            cursor[dst] += 1;
        }
    }

    for _ in 0..iterations {
        let next = round(graph, &cluster, &vote, &in_off, &in_src);
        if next == cluster {
            break; // converged
        }
        cluster = next;
    }

    graph
        .vertex_indices()
        .map(|v| (v, Value::Str(graph.vid.arc(cluster[v as usize]))))
        .collect()
}

/// One synchronous round: every vertex adopts the highest-energy incoming cluster from
/// the frozen `cluster` snapshot. Parallel across vertices; a per-thread scratch map is
/// reused across the vertices that thread handles.
fn round(
    graph: &Graph,
    cluster: &[u32],
    vote: &[f64],
    in_off: &[usize],
    in_src: &[u32],
) -> Vec<u32> {
    let pick = |u: u32, energy: &mut HashMap<u32, f64>| -> u32 {
        let range = in_off[u as usize]..in_off[u as usize + 1];
        if range.is_empty() {
            return cluster[u as usize]; // no incoming votes → keep own cluster
        }
        energy.clear();
        for &s in &in_src[range] {
            *energy.entry(cluster[s as usize]).or_insert(0.0) += vote[s as usize];
        }
        // Adopt the max-energy cluster; tie → lexicographically smallest external id.
        let mut best: Option<(u32, f64)> = None;
        for (&c, &e) in energy.iter() {
            let better = match best {
                None => true,
                Some((bc, be)) => e > be || (e == be && graph.vid.text(c) < graph.vid.text(bc)),
            };
            if better {
                best = Some((c, e));
            }
        }
        best.map_or(cluster[u as usize], |(c, _)| c)
    };

    let n = graph.n as u32;
    #[cfg(feature = "parallel")]
    {
        (0..n)
            .into_par_iter()
            .map_init(HashMap::new, |energy, u| pick(u, energy))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut energy = HashMap::new();
        (0..n).map(|u| pick(u, &mut energy)).collect()
    }
}
