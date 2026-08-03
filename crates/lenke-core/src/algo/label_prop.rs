//! Synchronous label propagation (a community-detection heuristic). Every vertex
//! starts labelled with its own external id; each round it adopts the label most
//! common among its neighbours (edges undirected — both in- and out-neighbours
//! count), ties broken by the **smallest label string**. Rounds are synchronous
//! (all vertices read the previous round's labels, then commit together) for a
//! fixed `iterations` count (default 10), with an early stop once a round changes
//! nothing (the remaining rounds are no-ops, so the result is unchanged).
//!
//! A label is carried internally as the **dense id of the vertex whose external id
//! is that label** (all labels are vertex external ids, so this is exact) — so a
//! round tallies `u32`s in a reused scratch map instead of hashing strings, and the
//! per-round work parallelizes across vertices (each vertex reads only the frozen
//! snapshot). The winner is chosen by (count, then smallest external-id string),
//! independent of neighbour-enumeration order, so the TS mirror is byte-identical.
//!
//! Two structural changes that help the *TS* mirror were tried here and reverted,
//! because native's data structures are already tighter: a precomputed neighbour CSR
//! (native `out_adj`/`in_adj` are already flat CSR slices) and a count-array +
//! dirty-list tally (a full-size count array thrashes cache on a spread-out random
//! graph, where the small `HashMap<u32,u32>` working set does not). Both live in the
//! TS version, which pays a high per-op cost for nested maps / the JS `Map`.

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::AlgoConfig;
use crate::graph::{Graph, Value};

const DEFAULT_ITERATIONS: u32 = 10;

pub fn label_propagation(graph: &Graph, cfg: &AlgoConfig) -> Vec<(u32, Value)> {
    let iterations = cfg.iterations.unwrap_or(DEFAULT_ITERATIONS);
    // Some(None) = every edge type; Some(Some(t)) = one type; None = a named-but-
    // unknown type → no edges, so every vertex keeps its own label forever.
    let etype = cfg.etype(graph);

    // Seed/anchor vertices (carrying a non-null `seedProperty`) never adopt a
    // neighbour's label — they stay pinned to their own id, so communities nucleate
    // around them instead of collapsing to one label. Empty (all false) → the prior
    // unsupervised behaviour.
    let is_seed: Vec<bool> = cfg.seed_property.as_deref().map_or_else(
        || vec![false; graph.n],
        |key| {
            (0..graph.n)
                .map(|v| !matches!(graph.props.value(v, key, &graph.strs), Value::Null))
                .collect()
        },
    );

    // labels[v] = the dense id of the vertex whose external id is v's current label;
    // starts as v itself. Tombstoned slots are self-labelled and never read.
    let mut labels: Vec<u32> = (0..graph.n as u32).collect();

    if let Some(etype) = etype {
        for _ in 0..iterations {
            let next = round(graph, &labels, etype, &is_seed);
            if next == labels {
                break; // converged — later rounds would be no-ops
            }
            labels = next;
        }
    }

    graph
        .vertex_indices()
        .map(|v| (v, Value::Str(graph.vid.arc(labels[v as usize]))))
        .collect()
}

/// One synchronous round: every vertex adopts the winning neighbour label from the
/// frozen `labels` snapshot. Parallel across vertices (each is independent); a
/// per-thread scratch map is reused across the vertices that thread handles.
fn round(graph: &Graph, labels: &[u32], etype: Option<u32>, is_seed: &[bool]) -> Vec<u32> {
    let pick = |v: u32, counts: &mut HashMap<u32, u32>| -> u32 {
        // A seed anchor (and a dead slot) keeps its current label — never adopts.
        if is_seed[v as usize] || !graph.is_vertex_live(v) {
            return labels[v as usize];
        }
        counts.clear();
        for a in graph.out_adj(v).chain(graph.in_adj(v)) {
            let ok = match etype {
                Some(t) => super::adj_type_ok(graph, Some(t), &a),
                None => true,
            };
            if ok {
                *counts.entry(labels[a.nbr as usize]).or_insert(0) += 1;
            }
        }
        // Adopt the most-frequent label; tie → lexicographically smallest external
        // id. No neighbours → keep the current label.
        let mut best: Option<(u32, u32)> = None;
        for (&lbl, &c) in counts.iter() {
            let better = match best {
                None => true,
                Some((bl, bc)) => c > bc || (c == bc && graph.vid.text(lbl) < graph.vid.text(bl)),
            };
            if better {
                best = Some((lbl, c));
            }
        }
        best.map_or(labels[v as usize], |(lbl, _)| lbl)
    };

    let n = graph.n as u32;
    #[cfg(feature = "parallel")]
    {
        (0..n)
            .into_par_iter()
            .map_init(HashMap::new, |counts, v| pick(v, counts))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut counts = HashMap::new();
        (0..n).map(|v| pick(v, &mut counts)).collect()
    }
}
