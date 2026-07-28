//! Vectorized neighbour aggregation (`neighborAggregate` / message passing).
//!
//! For each vertex, aggregate the list-valued `feature` vectors of its neighbours
//! **element-wise over the whole D-dim block in one native pass** — the primitive
//! host-driven GCN / GraphSAGE message passing wants (instead of D separate GQL
//! `SET`s). `op` chooses `mean` (default) / `sum` / `max` / `min`; `direction`
//! picks out- / in- / both-neighbours; `includeSelf` adds the vertex's own vector.
//! Each contributor is scaled by a COEFFICIENT = edge weight (`weightProperty`, 1.0
//! unweighted) × normalization (`norm:"gcn"` → `1/sqrt(deg_i·deg_j)`, else 1.0), so
//! `sum` = Σ coefⱼ·hⱼ and `mean` = that ÷ Σ coefⱼ (a WEIGHTED mean). A weight/norm is
//! rejected for `max`/`min` (scale-independent). The result is a list value per vertex,
//! optionally written to `writeProperty`.
//!
//! **Byte-identity with the TS mirror** rests on a fixed accumulation order: the
//! vertex's own vector first (when included), then each neighbour in ascending
//! **edge-index** order — so the f64 `sum`/`mean` adds land in the same order in both
//! engines. `max`/`min` are order-independent; degrees are integers (exact) so the GCN
//! `1/sqrt(deg_i·deg_j)` matches. A `both`-direction self-loop is counted once (the
//! in-side copy is dropped, mirroring `expand`).

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

/// Fold one contributor vector into `acc` under `op`, scaled by `coef` (the edge
/// weight × normalization factor; `1.0` for a plain unweighted aggregate). `coef_sum`
/// accumulates the coefficients (the denominator for a weighted `mean`); `started`
/// tracks whether `acc` holds a real value yet (for `max`/`min`, whose identity is the
/// first contributor, not zero). `max`/`min` ignore `coef` — they are scale-independent
/// and reject a weight/norm at the call site.
fn fold(op: Op, acc: &mut [f64], coef_sum: &mut f64, started: &mut bool, vec: &[f64], coef: f64) {
    match op {
        Op::Sum | Op::Mean => {
            for (a, x) in acc.iter_mut().zip(vec) {
                *a += coef * x;
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
    *coef_sum += coef;
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
    let gcn = match cfg.norm.as_deref().unwrap_or("none") {
        "none" => false,
        "gcn" => true,
        other => {
            return Err(format!(
                "neighborAggregate `norm` must be one of none|gcn, got '{other}'"
            ));
        }
    };
    // Per-edge weights (`None` = unweighted, coefficient 1.0). A weight or a `gcn` norm
    // SCALES each contributor, which is meaningless for the order/scale-independent
    // `max`/`min` — reject loudly rather than silently ignore.
    let weighted = cfg.weight_property.is_some();
    if (weighted || gcn) && matches!(op, Op::Max | Op::Min) {
        return Err(
            "neighborAggregate `weightProperty`/`norm` apply only to op=sum|mean \
             (max/min are scale-independent)"
                .to_string(),
        );
    }
    let weights: Option<Vec<f64>> = cfg
        .weight_property
        .as_deref()
        .map(|k| crate::algo::edge_weights(graph, k));
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

    // Gather a vertex's contributor `(eidx, nbr)` pairs by direction, sorted by edge
    // index for a canonical, engine-independent accumulation order. A `both`-direction
    // self-loop is counted once (the in-side copy is dropped, mirroring `expand`).
    let contributors = |v: u32| -> Vec<(u32, u32)> {
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
                if want_out && a.nbr == v {
                    continue;
                }
                if etype_ok(a.etype) {
                    contrib.push((a.eidx, a.nbr));
                }
            }
        }
        contrib.sort_unstable_by_key(|&(eidx, _)| eidx);
        contrib
    };

    // GCN degree per vertex: its contributor count under the SAME direction/etype filter,
    // plus the self-loop when `includeSelf` (the `Ã = A + I` self-loop). Floored at 1 so a
    // sink/source contributes a finite `1/sqrt(deg_i·deg_j)` instead of dividing by zero.
    let deg: Vec<f64> = if gcn {
        graph
            .vertex_indices()
            .map(|v| (contributors(v).len() + usize::from(include_self)).max(1) as f64)
            .collect()
    } else {
        Vec::new()
    };

    let mut out: Vec<(u32, Value)> = Vec::with_capacity(graph.vertex_count());
    for v in graph.vertex_indices() {
        let contrib = contributors(v);

        let mut acc = vec![0.0f64; d];
        let mut coef_sum = 0.0f64;
        let mut started = false;
        // The coefficient scaling a contributor at edge `eidx` from neighbour `nbr`:
        // edge weight (1.0 unweighted) × GCN factor (`1/sqrt(deg_i·deg_j)`, else 1.0).
        let coef_of = |eidx: u32, nbr: u32| -> f64 {
            let w = weights.as_ref().map_or(1.0, |ws| ws[eidx as usize]);
            let nf = if gcn {
                1.0 / (deg[v as usize] * deg[nbr as usize]).sqrt()
            } else {
                1.0
            };
            w * nf
        };
        if include_self {
            if let Some(sv) = &feats[v as usize] {
                // The self-loop has weight 1.0 and GCN factor `1/deg_i` (`sqrt(deg_i·deg_i)`).
                let coef = if gcn { 1.0 / deg[v as usize] } else { 1.0 };
                fold(op, &mut acc, &mut coef_sum, &mut started, sv, coef);
            }
        }
        for &(eidx, nbr) in &contrib {
            if let Some(nv) = &feats[nbr as usize] {
                fold(
                    op,
                    &mut acc,
                    &mut coef_sum,
                    &mut started,
                    nv,
                    coef_of(eidx, nbr),
                );
            }
        }
        if op == Op::Mean && coef_sum != 0.0 {
            for a in &mut acc {
                *a /= coef_sum;
            }
        }
        // No contributors → the zero vector (acc is already zeros; `max`/`min` also
        // leave zeros since `started` stayed false).
        out.push((v, Value::List(acc.into_iter().map(Value::Num).collect())));
    }
    Ok(out)
}
