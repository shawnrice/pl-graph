//! Vectorized neighbour aggregation (`neighborAggregate` / message passing).
//!
//! For each vertex, aggregate the list-valued `feature` vectors of its neighbours
//! **element-wise over the whole D-dim block in one native pass** — the primitive
//! host-driven GCN / GraphSAGE message passing wants (instead of D separate GQL
//! `SET`s). `op` chooses `mean` (default) / `sum` / `max` / `min`; `direction`
//! picks out- / in- / both-neighbours; `includeSelf` adds the vertex's own vector.
//! The result is a list value per vertex, optionally written to `writeProperty`.
//!
//! **Byte-identity with the TS mirror** rests on a fixed accumulation order: the
//! vertex's own vector first (when included), then each neighbour in ascending
//! **edge-index** order — so the f64 `sum`/`mean` adds land in the same order in
//! both engines (`mean` divides by an integer contributor count; `max`/`min` are
//! order-independent). A `both`-direction self-loop is counted once (the in-side
//! copy is dropped, mirroring `expand`).

use crate::algo::AlgoConfig;
use crate::graph::{Graph, Value};

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Mean,
    Sum,
    Max,
    Min,
}

/// Read vertex `v`'s `key` property as a numeric vector, or `None` if it is absent
/// or not a list of numbers (such a vertex contributes nothing to an aggregate).
/// A typed `Column::Vec` feature is a zero-copy slice (no boxing) → the fast path;
/// a list still boxed in a `Mixed` column falls back to unboxing via `value`.
fn read_vec(graph: &Graph, v: u32, key: &str) -> Option<Vec<f64>> {
    if let Some(slice) = graph.props.vector(v as usize, key) {
        return Some(slice.to_vec()); // contiguous copy — no per-element unboxing
    }
    match graph.props.value(v as usize, key, &graph.strs) {
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for it in &items {
                match it {
                    Value::Num(n) => out.push(*n),
                    _ => return None, // a non-numeric element → not a feature vector
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Fold one contributor vector into `acc` under `op`; `count`/`started` track how
/// many have folded (for `mean`) and whether `acc` holds a real value yet (for
/// `max`/`min`, whose identity is the first contributor, not zero).
fn fold(op: Op, acc: &mut [f64], count: &mut usize, started: &mut bool, vec: &[f64]) {
    match op {
        Op::Sum | Op::Mean => {
            for (a, x) in acc.iter_mut().zip(vec) {
                *a += x;
            }
        }
        Op::Max => {
            if *started {
                for (a, x) in acc.iter_mut().zip(vec) {
                    *a = a.max(*x);
                }
            } else {
                acc.copy_from_slice(vec);
            }
        }
        Op::Min => {
            if *started {
                for (a, x) in acc.iter_mut().zip(vec) {
                    *a = a.min(*x);
                }
            } else {
                acc.copy_from_slice(vec);
            }
        }
    }
    *started = true;
    *count += 1;
}

pub fn neighbor_aggregate(graph: &Graph, cfg: &AlgoConfig) -> Result<Vec<(u32, Value)>, String> {
    let feature = cfg
        .feature
        .as_deref()
        .ok_or_else(|| "neighborAggregate requires a `feature` property".to_string())?;
    let op = match cfg.op.as_deref().unwrap_or("mean") {
        "mean" => Op::Mean,
        "sum" => Op::Sum,
        "max" => Op::Max,
        "min" => Op::Min,
        other => {
            return Err(format!(
                "neighborAggregate `op` must be one of mean|sum|max|min, got '{other}'"
            ));
        }
    };
    let (want_out, want_in) = match cfg.direction.as_deref().unwrap_or("both") {
        "out" => (true, false),
        "in" => (false, true),
        "both" => (true, true),
        other => {
            return Err(format!(
                "neighborAggregate `direction` must be one of out|in|both, got '{other}'"
            ));
        }
    };
    let include_self = cfg.include_self.unwrap_or(false);
    // `Some(None)` = every type, `Some(Some(id))` = one, `None` = named-but-unknown
    // (no edges match, so every aggregate is over an empty neighbourhood).
    let etype = cfg.etype(graph);

    // Precompute every vertex's feature vector once (avoids re-reading/re-allocating
    // per neighbour) and infer the shared dimension `d`; a length mismatch faults.
    let feats: Vec<Option<Vec<f64>>> = graph
        .vertex_indices()
        .map(|v| read_vec(graph, v, feature))
        .collect();
    let mut dim: Option<usize> = None;
    for f in feats.iter().flatten() {
        match dim {
            None => dim = Some(f.len()),
            Some(d) if d != f.len() => {
                return Err(format!(
                    "neighborAggregate feature vectors must all have the same length; found {} and {}",
                    d,
                    f.len()
                ));
            }
            _ => {}
        }
    }
    let d = dim.unwrap_or(0);

    let etype_ok = |t: u32| match etype {
        Some(None) => true,
        Some(Some(id)) => t == id,
        None => false,
    };

    let mut out: Vec<(u32, Value)> = Vec::with_capacity(graph.vertex_count());
    for v in graph.vertex_indices() {
        // Gather contributor `(eidx, nbr)` pairs by direction, then sort by edge
        // index for a canonical, engine-independent accumulation order.
        let mut contrib: Vec<(u32, u32)> = Vec::new();
        if want_out {
            for a in graph.out_adj(v) {
                if etype_ok(a.etype) {
                    contrib.push((a.eidx, a.nbr));
                }
            }
        }
        if want_in {
            for a in graph.in_adj(v) {
                // A `both`-direction self-loop is already counted on the out-side.
                if want_out && a.nbr == v {
                    continue;
                }
                if etype_ok(a.etype) {
                    contrib.push((a.eidx, a.nbr));
                }
            }
        }
        contrib.sort_unstable_by_key(|&(eidx, _)| eidx);

        let mut acc = vec![0.0f64; d];
        let mut count = 0usize;
        let mut started = false;
        if include_self {
            if let Some(sv) = &feats[v as usize] {
                fold(op, &mut acc, &mut count, &mut started, sv);
            }
        }
        for &(_, nbr) in &contrib {
            if let Some(nv) = &feats[nbr as usize] {
                fold(op, &mut acc, &mut count, &mut started, nv);
            }
        }
        if op == Op::Mean && count > 0 {
            for a in &mut acc {
                *a /= count as f64;
            }
        }
        // No contributors → the zero vector (acc is already zeros; `max`/`min` also
        // leave zeros since `started` stayed false).
        out.push((v, Value::List(acc.into_iter().map(Value::Num).collect())));
    }
    Ok(out)
}
