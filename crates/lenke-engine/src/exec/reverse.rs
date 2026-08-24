use super::*;
use crate::batch::{Batch, Col};
use crate::ir::{Expr, Plan};
use crate::store::Store;
use crate::value::Value;

/// A hop: for each input row, expand the node in slot `from` along `dir`,
/// filtered by `edge_label`; emit one output row per matching neighbour with the
/// existing slots replicated and the neighbour appended as a new slot. This is
/// the bulk (lineage-free) strategy: `keep` records which input row each output
/// row came from, `nbrs` the landed node — the existing slots are gathered by
pub(super) fn reverse_dir(dir: Dir) -> Dir {
    match dir {
        Dir::Out => Dir::In,
        Dir::In => Dir::Out,
        Dir::Both => Dir::Both,
    }
}

/// Cardinality-driven ANCHOR FLIP for `Filter(target = lit, Expand(Scan(src), Out))`
/// — the "selective filter on the traversal TARGET" shape. The forward plan scans
/// EVERY source and expands to filter the target at the end; when the target is an
/// indexed `=` whose bucket is smaller than the source scan, it is far cheaper to
/// SEED the target (index) and walk the edges in REVERSE to the sources. The output
/// is the IDENTICAL `[source, target]` slot layout, so nothing downstream changes;
/// only the (unspecified) row order differs — the multiset is preserved. `None`
/// unless the shape matches, the target is index-seekable, the cost says flip, and
/// no path is tracked / no edge is bound.
/// The reverse-walk win for shortest paths: `Filter{endpoint == t} over ShortestPath`
/// resolves the target set `t` and hands it to the BFS as an early stop, so each source
/// stops sweeping the moment every target is settled rather than exploring the whole
/// reachable component. The filter is STILL applied to the result here, so the output is
/// byte-identical to the unbounded path — this only avoids materializing rows the filter
/// would drop. Conservative: only an indexed `endpoint.key == lit` over `min == 0`.
pub(super) fn try_shortest_early_stop(
    pred: &Expr,
    input: &Plan,
    store: &Store,
    track: bool,
) -> Option<Batch> {
    let Plan::ShortestPath {
        input: sp_in,
        from,
        dir,
        edge_label,
        min: 0, // `*` only — a `+` (min 1) has source-as-endpoint cycle cases
        max,
        selector,
        edge_pred,
    } = input
    else {
        return None;
    };
    let (key, value) = target_eq(pred, 1)?; // endpoint (slot 1) `== lit`
    if !store.has_hash_index(&key) {
        return None; // resolve the target set from the index, else keep the normal path
    }
    let targets = store.index_lookup(&key, &value)?;
    if targets.is_empty() {
        return None; // no target → the normal path yields empty; nothing to accelerate
    }
    let sp_batch = shortest_path(
        &pull(sp_in, store, track).ok()?,
        store,
        *from,
        *dir,
        edge_label,
        0,
        *max,
        *selector,
        edge_pred.as_deref(),
        Some(&targets),
    );
    // Apply the endpoint filter exactly as the general path would — the early stop only
    // changed which never-kept rows were produced, so this reproduces the same result.
    let keep: Vec<usize> = match try_filter_keep(pred, store, &sp_batch) {
        Some(k) => k,
        None => {
            let mask = eval(pred, store, &sp_batch).ok()?;
            match &mask {
                Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                other => (0..other.len())
                    .filter(|&i| other.value_at(i).is_true())
                    .collect(),
            }
        }
    };
    Some(sp_batch.gather(&keep))
}

/// The cardinality-approved decision for a reverse-seed: the hop chain (innermost-first,
/// carrying each hop's bind-edge flag), the source scan's label, the endpoint slot the
/// predicate seeds on, and the seeded endpoint bucket. Produced by [`reverse_seed_decide`],
/// which materializes the bucket to size the cardinality guard.
pub(super) struct RevSeed {
    hops: Vec<RevHop>,
    src_label: Option<String>,
    ep_slot: usize,
    bucket: Vec<u32>,
}

pub(super) fn reverse_seed_decide(
    pred: &Expr,
    input: &Plan,
    store: &Store,
    track: bool,
) -> Option<RevSeed> {
    if track {
        return None; // a path-reading query keeps the forward walk (lineage)
    }
    // Unwrap a chain of expands over a scan. A hop may bind its edge (appending an edge
    // slot before the landed node); `chain` collects them outermost-in as
    // (from, dir, edge_label, bind_edge).
    let mut chain: Vec<(usize, Dir, &[String], bool)> = Vec::new();
    let mut cur = input;
    let src_label = loop {
        match cur {
            Plan::Expand {
                input: inner,
                from,
                dir,
                edge_label,
                bind_edge,
                double_loops: false,
            } => {
                chain.push((*from, *dir, edge_label.as_slice(), *bind_edge));
                cur = inner.as_ref();
            }
            Plan::Scan { label } if !chain.is_empty() => break label.clone(),
            _ => return None, // source must bottom at an unfiltered scan
        }
    };
    // Build the hops innermost-first, verifying each feeds from the running node slot (a
    // straight chain, no branch/re-entry) and tracking where the endpoint node lands: a
    // bound hop appends an edge slot then the node (+2), an unbound hop just the node (+1).
    let mut hops: Vec<RevHop> = Vec::with_capacity(chain.len());
    let mut node_slot = 0usize;
    for &(from, dir, edge_label, bind_edge) in chain.iter().rev() {
        if from != node_slot {
            return None;
        }
        node_slot += if bind_edge { 2 } else { 1 };
        let want = match want_etypes(store, edge_label) {
            Ok(w) => w,
            Err(()) => return None,
        };
        hops.push(RevHop {
            dir,
            want,
            bind_edge,
        });
    }
    let ep_slot = node_slot;
    // Seed the endpoint from the index — an equality, range, IN, OR, or the more selective
    // conjunct of an AND; the residual filter (below) exacts the answer.
    let bucket = seed_bucket(pred, ep_slot, store)?;
    // Cardinality decision: flip only when the endpoint bucket is smaller than the source
    // scan (the reverse walks back only the paths that reach it).
    let source_rows = match &src_label {
        Some(l) => store.nodes_with_label(l).len(),
        None => store.live_node_count(),
    };
    // Loose only for a bare equality (no residual, and a smaller-than-scan bucket already
    // wins). Everything else — range / IN / OR / AND — MATERIALIZES the walked rows and
    // boxes a residual, so it needs the selectivity guard; a large OR union in particular
    // must NOT flip when a downstream LIMIT could stream the forward walk cheaply.
    let loose = target_eq(pred, ep_slot).is_some();
    // A SINGLE-hop non-loose seed has a cheap forward alternative: sweep the endpoint type's
    // edges off the per-type CSR (`fwd`). Reverse-seeding a NON-selective range instead seeds
    // its large bucket and walks the SPARSE type-in edges back — a random probe per seed that
    // costs more than the sequential forward sweep. Decline when the bucket is not smaller than
    // that forward cost (the `reverse_seed_worth` guard prices against the SOURCE scan, which
    // over-fires for a sparse type — `age >= 77` = 92k seeds vs 80k forward F edges). Byte-
    // identical either way — this only picks the cheaper equivalent plan.
    if !loose {
        if let Some(ep) = hops.last() {
            let fwd: usize = ep
                .want
                .iter()
                .filter_map(|&t| store.out_typed_flat(t).map(<[_]>::len))
                .sum();
            // A random type-in probe per seed costs ~10x a sequential forward edge read, so a
            // SINGLE-hop reverse-seed wins only when its bucket is well under a tenth of `fwd`
            // (the endpoint type's forward edge count). A MULTI-hop forward walk fans out again
            // past `fwd`, so it is only cheaper when the bucket is at least the whole endpoint
            // edge count — a stricter bar that keeps a mid-selective 2-hop range on the seed.
            let factor = if hops.len() == 1 { 8 } else { 1 };
            if fwd > 0 && bucket.len().saturating_mul(factor) >= fwd {
                return None;
            }
        }
    }
    if !reverse_seed_worth(bucket.len(), source_rows, loose, store) {
        return None;
    }
    Some(RevSeed {
        hops,
        src_label,
        ep_slot,
        bucket,
    })
}

/// A `DISTINCT` whose result cannot reach `cap` rows: its input projects a single bare
/// property that is a low-cardinality dict column with `distinct_count + 1 <= cap` (the
/// `+1` covers a possible NULL, which DISTINCT counts as a value but the dict does not).
/// A `LIMIT cap` over such a DISTINCT cannot bind.
pub(super) fn distinct_cap_cannot_bind(distinct_input: &Plan, cap: usize, store: &Store) -> bool {
    let Plan::Project { items, .. } = distinct_input else {
        return false;
    };
    if items.len() != 1 {
        return false;
    }
    let Expr::Prop { key, .. } = &items[0].1 else {
        return false;
    };
    store
        .distinct_count(key)
        .is_some_and(|d| d.saturating_add(1) <= cap)
}

/// Would `plan` reverse-seed? Peeks through the row-preserving wrappers that sit above
/// the `Filter` (Project) so a blocking op (OrderPage) can pick the reverse-seed over a
/// forward stream. Cheap — the underlying decision is O(1) index lookups, no walk.
pub(super) fn reverse_seed_applies(plan: &Plan, store: &Store, track: bool) -> bool {
    match plan {
        Plan::Project { input, .. } => reverse_seed_applies(input, store, track),
        Plan::Filter { input, pred } => reverse_seed_decide(pred, input, store, track).is_some(),
        _ => false,
    }
}

/// A reverse-walk hop: direction, edge-type want-set, and whether the forward hop BOUND
/// its edge (appending an edge slot before the landed node). Innermost-first.
pub(super) struct RevHop {
    dir: Dir,
    want: Vec<u32>,
    bind_edge: bool,
}

/// Reverse-walk a chain of hops, prepending each hop's source (and, for a bound-edge hop,
/// its edge) to every partial row. `rows` start as suffixes headed by the frontier node
/// (`row[0]`, the node to walk back from); a bound hop prepends `[src, edge]` so the row
/// stays in forward slot order `[…, src, edge, landed, …]` and `row[0]` remains a node.
/// Intermediate nodes are unconstrained (the forward plan filters only the scan's source),
/// so the scan's source label is enforced once, on the last prepend (hop 0 → s_0). An
/// empty `hops` returns `rows` unchanged.
pub(super) fn reverse_walk_chain(
    mut rows: Vec<Vec<u32>>,
    hops: &[RevHop],
    src_label: Option<&str>,
    store: &Store,
) -> Vec<Vec<u32>> {
    for k in (0..hops.len()).rev() {
        let hop = &hops[k];
        let rev = reverse_dir(hop.dir);
        let last_hop = k == 0;
        let mut next: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        for row in &rows {
            let head = row[0];
            for_each_nbr(store, head, rev, &hop.want, false, |a, eid| {
                if last_hop && !src_label.is_none_or(|l| store.is_labeled(a, l)) {
                    return;
                }
                let mut r = Vec::with_capacity(row.len() + if hop.bind_edge { 2 } else { 1 });
                r.push(a);
                if hop.bind_edge {
                    r.push(eid);
                }
                r.extend_from_slice(row);
                next.push(r);
            });
        }
        rows = next;
    }
    rows
}

/// The per-slot column kinds a reverse-walk of `hops` produces, in forward slot order:
/// the source node, then each hop's `[edge?, landed node]`. `false` = node, `true` = edge.
pub(super) fn chain_slot_kinds(hops: &[RevHop]) -> Vec<bool> {
    let mut kinds = vec![false]; // slot 0: the source node
    for hop in hops {
        if hop.bind_edge {
            kinds.push(true); // the bound edge slot
        }
        kinds.push(false); // the landed node
    }
    kinds
}

/// Transpose full rows into columns typed by `kinds` (`true` = edge slot, else node).
pub(super) fn rows_to_batch(rows: &[Vec<u32>], kinds: &[bool]) -> Batch {
    let mut cols: Vec<Vec<u32>> = (0..kinds.len())
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for row in rows {
        for (i, &v) in row.iter().enumerate() {
            cols[i].push(v);
        }
    }
    Batch::of(
        cols.into_iter()
            .zip(kinds)
            .map(|(c, &is_edge)| {
                if is_edge {
                    Col::Edges(c)
                } else {
                    Col::Nodes(c)
                }
            })
            .collect(),
    )
}

pub(super) fn try_reverse_expand(
    pred: &Expr,
    input: &Plan,
    store: &Store,
    track: bool,
) -> Option<Batch> {
    let RevSeed {
        hops,
        src_label,
        ep_slot,
        bucket,
    } = reverse_seed_decide(pred, input, store, track)?;
    // Reverse-walk the chain from the seeded endpoint bucket to the labeled sources, then
    // transpose into node/edge columns matching the forward slot layout.
    let rows = reverse_walk_chain(
        bucket.iter().map(|&t| vec![t]).collect(),
        &hops,
        src_label.as_deref(),
        store,
    );
    let b = rows_to_batch(&rows, &chain_slot_kinds(&hops));
    // A bare equality is fully satisfied by the seed. A conjunction / range / IN / OR — or
    // a bound-edge chain with an edge-property residual — needs the WHOLE predicate applied
    // over the (small) seeded batch. If the residual can't evaluate cleanly, decline.
    if target_eq(pred, ep_slot).is_some() {
        return Some(b);
    }
    let keep = residual_keep(pred, store, &b)?;
    Some(b.gather(&keep))
}

/// The reverse-seed for a VAR-LENGTH hop, possibly behind fixed leading hops:
/// `Filter(endpoint.key = lit, VarLength{from:L}(Expand·…·Expand(Scan)))` with L ≥ 0
/// plain fixed hops before the quantifier. The forward walk enumerates every path from
/// every source and filters the endpoint at the end — on a fanning graph that is the
/// ~34M-path materialization that trips the trail-limit guard. Instead, seed the
/// selective indexed endpoint and walk BACKWARD: the var-length in reverse (reusing
/// `var_length` on `reverse_dir(dir)`), then the fixed chain in reverse. Reversal
/// preserves every mode's path validity (Trail = same edge set, Simple/Acyclic = same
/// node set) and forward/reverse trails biject, so the whole row multiset — hence
/// `count(*)` and every projection — is identical.
///
/// The var-length reverse yields `(endpoint c, var-source b=slot L)` pairs; those feed
/// `reverse_walk_chain` for the fixed hops (which enforces the scan's source label on
/// slot 0). With no fixed hops (L = 0) the var-source IS the labeled source, so it is
/// filtered directly. A conjunction is seeded on its equality conjunct and residual-
/// filtered by the whole predicate. Plain hops only (no until/body_filter/both-loops,
/// no path lineage); anything else declines to the forward path.
pub(super) fn try_reverse_varlen(
    pred: &Expr,
    input: &Plan,
    store: &Store,
    track: bool,
) -> Option<Batch> {
    if track {
        return None; // a path-reading query keeps the forward walk (lineage)
    }
    let Plan::VarLength {
        input: vl_in,
        from: vl_from,
        dir,
        edge_label,
        min,
        max,
        mode,
        until,
        body_filter,
        double_loops,
    } = input
    else {
        return None;
    };
    if until.is_some() || body_filter.is_some() || *double_loops {
        return None; // Gremlin until()/body-filter/both() have no simple reverse
    }
    // Unwrap the fixed hops below the var-length (0+ plain expands over a scan),
    // outermost-in so their `from` runs L-1 down to 0.
    let mut chain: Vec<(usize, Dir, &[String])> = Vec::new();
    let mut cur = vl_in.as_ref();
    let src_label = loop {
        match cur {
            Plan::Expand {
                input: inner,
                from,
                dir,
                edge_label,
                bind_edge: false,
                double_loops: false,
            } => {
                chain.push((*from, *dir, edge_label.as_slice()));
                cur = inner.as_ref();
            }
            Plan::Scan { label } => break label.clone(),
            _ => return None, // source must bottom at an unfiltered scan
        }
    };
    let fixed = chain.len();
    // The var-length must expand from the fixed chain's last slot, and the chain must be a
    // straight run (from L-1, L-2, …, 0). The endpoint lands in slot L+1.
    if *vl_from != fixed
        || chain
            .iter()
            .enumerate()
            .any(|(i, (from, _, _))| *from != fixed - 1 - i)
    {
        return None;
    }
    let ep_slot = fixed + 1;
    // Seed the endpoint from the index (equality / range / IN / OR / more selective AND
    // conjunct); the residual filter (below) exacts the answer.
    let bucket = seed_bucket(pred, ep_slot, store)?;
    // Cardinality decision: flip only when the endpoint bucket is smaller than the source
    // scan (the same guard as the fixed-length seed).
    let source_rows = match &src_label {
        Some(l) => store.nodes_with_label(l).len(),
        None => store.live_node_count(),
    };
    if bucket.len() >= source_rows {
        return None;
    }
    // The fixed hops are plain (no bound edge), innermost-first — the var-length output is
    // all nodes, so the reverse-walk over them produces a node-only layout.
    let mut fixed_hops: Vec<RevHop> = Vec::with_capacity(fixed);
    for &(_, dir, edge_label) in chain.iter().rev() {
        let want = match want_etypes(store, edge_label) {
            Ok(w) => w,
            Err(()) => return None,
        };
        fixed_hops.push(RevHop {
            dir,
            want,
            bind_edge: false,
        });
    }
    // Walk the var-length in reverse from the endpoint bucket. Decline (forward path) if
    // the reverse walk itself trips the trail guard — a huge-in-degree endpoint.
    let seed = Batch::of(vec![Col::Nodes(bucket)]);
    let rev = var_length(
        &seed,
        store,
        0,
        reverse_dir(*dir),
        edge_label,
        *min,
        *max,
        *mode,
        &[],
        None,
        1,
        None,
        None,
        false,
    )
    .ok()?;
    // rev is [endpoint(c) gathered, ends(b = var-source = slot L)]. Build rows headed by
    // b (the frontier for the fixed-chain reverse) with c as the suffix. With no fixed
    // hops, b IS the source, so apply the scan label here; otherwise the chain applies it.
    let (Col::Nodes(cs), Col::Nodes(bs)) = (rev.slot(0), rev.slot(1)) else {
        return None;
    };
    let rows0: Vec<Vec<u32>> = bs
        .iter()
        .zip(cs.iter())
        .filter(|(&b, _)| fixed != 0 || src_label.as_deref().is_none_or(|l| store.is_labeled(b, l)))
        .map(|(&b, &c)| vec![b, c])
        .collect();
    let rows = reverse_walk_chain(rows0, &fixed_hops, src_label.as_deref(), store);
    let out = rows_to_batch(&rows, &vec![false; ep_slot + 1]);
    // A bare equality is fully satisfied by the seed; a conjunction needs its other
    // conjuncts applied over the (small) seeded batch.
    if target_eq(pred, ep_slot).is_some() {
        return Some(out);
    }
    let keep = residual_keep(pred, store, &out)?;
    Some(out.gather(&keep))
}

/// Keep-indices for a residual predicate over an already-materialized (small) batch —
/// the fast `try_filter_keep` pass, else a boxed `eval`. `None` if `eval` faults, so a
/// caller can decline and fall back to the forward path rather than swallow the error.
pub(super) fn residual_keep(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    if let Some(keep) = try_filter_keep(pred, store, batch) {
        return Some(keep);
    }
    let mask = eval_mask(pred, store, batch).ok()?;
    Some((0..mask.len()).filter(|&i| mask[i] == Some(true)).collect())
}

/// The candidate endpoint node ids for a seedable predicate on `slot` — a DEDUPED set
/// that is a SUPERSET of the predicate's exact matches. The caller's residual filter
/// (whenever `pred` isn't a bare equality) narrows it to the exact set, so this only has
/// to over-approximate, which keeps NULL / cross-type / ordering edge cases the residual's
/// job. Handles an indexed equality (hash), a range op (range index), a positive
/// `IN [lits]` (union of hash buckets), an `OR` of seedables (union), and an `AND` (the
/// more selective conjunct's bucket). `None` when nothing on `slot` is seedable.
pub(super) fn seed_bucket(pred: &Expr, slot: usize, store: &Store) -> Option<Vec<u32>> {
    if let Some(b) = seed_pure(pred, slot, store) {
        return Some(b);
    }
    if let Expr::And(l, r) = pred {
        // Two range bounds on the SAME key (one lower, one upper) seed their exact
        // intersection in one BTree walk — `k >= a AND k < b` narrows to [a, b) and a
        // contradictory pair (a > b) yields the empty set. Without this, each conjunct
        // seeds independently and we keep the wider single-bound bucket (or, when the
        // pair is unsatisfiable, materialize a large bucket only to filter it to zero).
        if let Some(ids) = seed_interval(l, r, slot, store) {
            return Some(ids);
        }
        // Seed the more selective conjunct; the residual applies the whole conjunction.
        return match (seed_bucket(l, slot, store), seed_bucket(r, slot, store)) {
            (Some(a), Some(b)) => Some(if a.len() <= b.len() { a } else { b }),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
    }
    None
}

/// Two range bounds on the SAME key with OPPOSITE directions (one lower, one upper)
/// → seed their exact intersection via a two-sided range seek. Byte-identical: the
/// reverse-seed re-applies the full conjunction as its residual, and the intersection
/// is exactly the set satisfying both bounds (empty when contradictory). Same-direction
/// pairs fall through to the generic per-conjunct seed, which already picks the tighter.
pub(super) fn seed_interval(l: &Expr, r: &Expr, slot: usize, store: &Store) -> Option<Vec<u32>> {
    use crate::ir::CompareOp::{Ge, Gt, Le, Lt};
    use std::ops::Bound::{Excluded, Included, Unbounded};
    let (kl, ol, vl) = endpoint_range(l, slot)?;
    let (kr, or, vr) = endpoint_range(r, slot)?;
    if kl != kr || !store.has_range_index(&kl) {
        return None;
    }
    let side = |op: CompareOp, v: &Value| match op {
        Gt => Some((true, Excluded(v.clone()))),
        Ge => Some((true, Included(v.clone()))),
        Lt => Some((false, Excluded(v.clone()))),
        Le => Some((false, Included(v.clone()))),
        _ => None,
    };
    let (l_low, lb) = side(ol, &vl)?;
    let (r_low, rb) = side(or, &vr)?;
    let (lo, hi) = match (l_low, r_low) {
        (true, false) => (lb, rb),
        (false, true) => (rb, lb),
        _ => return None, // same direction — generic seed picks the tighter conjunct
    };
    let lo_ref = match &lo {
        Included(v) => Included(v),
        Excluded(v) => Excluded(v),
        Unbounded => Unbounded,
    };
    let hi_ref = match &hi {
        Included(v) => Included(v),
        Excluded(v) => Excluded(v),
        Unbounded => Unbounded,
    };
    store.range_between(&kl, lo_ref, hi_ref)
}

/// Is a reverse-seed over a FIXED-hop chain worth it? The forward count/agg folds during
/// a single scan (cheap, no materialization); the reverse-seed materializes the walked
/// rows and boxes any residual over them. When the forward predicate ALSO folds cheaply
/// (a simple range/IN/AND that `try_filter_keep` handles), the reverse only wins on a
/// SMALL FRACTION of the scan — require its fan-out (bucket × degree²) to stay under the
/// forward scan. Only a `loose` predicate (a bare equality — no residual) wins on any
/// bucket smaller than the scan; range/IN/OR/AND all take the tight guard, so a large OR
/// union does not flip when a downstream LIMIT could stream the forward walk cheaply.
/// (Var-length seeds skip this — their forward path is a trail-limit blow-up.)
pub(super) fn reverse_seed_worth(bucket: usize, source: usize, loose: bool, store: &Store) -> bool {
    if bucket >= source {
        return false;
    }
    if loose {
        return true;
    }
    // The reverse-seed materializes ~bucket × reverse-degree rows; it wins when that stays
    // under the forward scan of `source`. `deg` is the GLOBAL avg degree — deliberately: a
    // per-edge-type degree was TRIED (2026-08-15) and REVERTED, because the sparse edge type
    // here (F, deg 1) made a NON-selective range bucket (`score >= 89` = 91% of nodes) pass
    // `bucket × 1 < source` and wrongly fire the seed, materializing ~all the graph (26ms vs
    // the 3.6ms forward win). The global degree correctly declines those. The residual is
    // vectorized (`eval_mask`), so a single `deg` factor (not `deg²`) already admits the
    // genuinely selective ranges while declining the dense/non-selective ones.
    let deg = (store.edge_count() as f64 / store.live_node_count().max(1) as f64).max(1.0);
    (bucket as f64) * deg < source as f64
}

/// A predicate whose ENTIRE match set is captured by one index bucket or a union of them
/// (equality / range / positive `IN` / `OR` of such) — the deduped candidate set. Never
/// descends an `AND` (that needs a residual only the caller applies).
pub(super) fn seed_pure(pred: &Expr, slot: usize, store: &Store) -> Option<Vec<u32>> {
    if let Some((k, v)) = target_eq(pred, slot) {
        if store.has_hash_index(&k) {
            return store.index_lookup(&k, &v);
        }
    }
    if let Some((k, op, v)) = endpoint_range(pred, slot) {
        if store.has_range_index(&k) {
            return store.range_lookup(&k, op, &v);
        }
    }
    if let Some((k, vals)) = endpoint_in(pred, slot) {
        if store.has_hash_index(&k) {
            let mut ids = Vec::new();
            for v in &vals {
                ids.extend(store.index_lookup(&k, v)?);
            }
            ids.sort_unstable();
            ids.dedup();
            return Some(ids);
        }
    }
    if let Expr::Or(l, r) = pred {
        let mut a = seed_pure(l, slot, store)?;
        a.extend(seed_pure(r, slot, store)?);
        a.sort_unstable();
        a.dedup();
        return Some(a);
    }
    None
}

/// `slot.key <op> lit` (or the mirror) with a range op — the endpoint-range analogue of
/// [`target_eq`]. The mirror flips the operator so `lit <op> slot.key` reads as the seek.
pub(super) fn endpoint_range(pred: &Expr, slot: usize) -> Option<(String, CompareOp, Value)> {
    let Expr::Compare { op, left, right } = pred else {
        return None;
    };
    if !matches!(
        op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return None;
    }
    let flip = |o: CompareOp| match o {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Ge => CompareOp::Le,
        other => other,
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot: s, key }, Expr::Lit(v)) if *s == slot => {
            (!v.is_null()).then(|| (key.clone(), *op, v.clone()))
        }
        (Expr::Lit(v), Expr::Prop { slot: s, key }) if *s == slot => {
            (!v.is_null()).then(|| (key.clone(), flip(*op), v.clone()))
        }
        _ => None,
    }
}

/// `slot.key IN [lit, lit, …]` → (key, literal values). Positive `IN` over a literal list
/// only (a `NOT … IN` is not a superset seed); every element must be a non-null literal.
pub(super) fn endpoint_in(pred: &Expr, slot: usize) -> Option<(String, Vec<Value>)> {
    let Expr::In { needle, haystack } = pred else {
        return None;
    };
    let Expr::Prop { slot: s, key } = needle.as_ref() else {
        return None;
    };
    if *s != slot {
        return None;
    }
    let Expr::List { items } = haystack.as_ref() else {
        return None;
    };
    let mut vals = Vec::with_capacity(items.len());
    for it in items {
        let Expr::Lit(v) = it else {
            return None;
        };
        if v.is_null() {
            return None;
        }
        vals.push(v.clone());
    }
    (!vals.is_empty()).then(|| (key.clone(), vals))
}

/// Parse `Prop{slot, key} = Lit(value)` (or its mirror) — an equality on the given slot.
pub(super) fn target_eq(pred: &Expr, slot: usize) -> Option<(String, Value)> {
    let Expr::Compare {
        op: CompareOp::Eq,
        left,
        right,
    } = pred
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot: s, key }, Expr::Lit(v))
        | (Expr::Lit(v), Expr::Prop { slot: s, key })
            if *s == slot =>
        {
            (!v.is_null()).then(|| (key.clone(), v.clone()))
        }
        _ => None,
    }
}
