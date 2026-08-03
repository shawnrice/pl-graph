//! Query-shape fast-paths: recognize special MATCH/RETURN shapes (bare
//! `count(*)`, two-hop counts, grouped var-length, distinct-reachable,
//! parallel scans/aggregates, …) and answer them without materializing the full
//! result. Each returns `None` to fall back to the general executor, and is
//! provably identical to it. Extracted from the evaluator (`super`); shares its
//! context/helpers via `use super::*`.
use super::*;

/// O(1) shortcut for `MATCH (n:Label) RETURN count(*)`: no WHERE, no path, no
/// grouping / extra aggregate / DISTINCT / ORDER BY / SKIP / LIMIT. The result is
/// exactly the label bucket's size, so read `vertices_with_label(l).len()` instead
/// of materializing and counting the whole id column — turning an O(n) scan into
/// an O(1) read. Provably identical to the general path, which counts that same
/// bucket; the difference is `bucket.len()` vs `bucket.iter().count()`.
/// Is the projection exactly a bare, un-grouped `count(*)` — no DISTINCT, ORDER BY,
/// SKIP, LIMIT, grouping, or any extra item/aggregate? Every scalar-count shortcut
/// below requires this shape before it may substitute a closed-form count for row
/// enumeration, and bails the moment any of these is present. (The grouped and
/// DISTINCT-count shortcuts have their own, different guards.)
fn is_bare_count_star(proj: &CProjection) -> bool {
    !proj.distinct
        && proj.order_by.is_empty()
        && proj.skip.is_none()
        && proj.limit.is_none()
        && proj.out_len == 1
        && proj.aggs.len() == 1
        && proj.items.len() == 1
        && matches!(proj.items[0].expr, CExpr::AggRef(0))
        && proj.group_by.is_empty()
        && {
            let agg = &proj.aggs[0];
            agg.star && !agg.distinct && matches!(agg.func, AggFn::Count)
        }
}

pub(super) fn try_count_star(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    // a single bare node `(n:Label)` — one pattern, no path segments, no inline
    // props / WHERE on the node.
    let [path] = patterns.as_slice() else {
        return None;
    };
    if !path.segments.is_empty() || !path.start.props.is_empty() || path.start.where_.is_some() {
        return None;
    }
    // exactly one label (no `|`, `!`, wildcard) — else the bucket isn't the count.
    let Some(CLabelExpr::Label(label_ref)) = &path.start.label else {
        return None;
    };
    // the projection is exactly `count(*)` and nothing else.
    if !is_bare_count_star(proj) {
        return None;
    }
    let ctx = resolve_ctx(graph, plan, params);
    let n = ctx.labels[*label_ref]
        .0
        .map_or(0, |lid| graph.vertices_with_label(lid).len());
    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(n as f64)));
    Some(rs)
}

/// Collect the edge-type ids named by a `:T` / `:A|B` relationship label into
/// `out` (deduped). Returns `false` for `And`/`Not`/wildcard — no cheap type
/// enumeration, so the caller must fall back to per-vertex expansion.
pub(super) fn collect_etype_ids(ctx: &Ctx, expr: &CLabelExpr, out: &mut Vec<u32>) -> bool {
    match expr {
        CLabelExpr::Label(r) => {
            if let Some(t) = ctx.labels[*r].1 {
                if !out.contains(&t) {
                    out.push(t);
                }
            }
            true
        }
        CLabelExpr::Or(l, r) => collect_etype_ids(ctx, l, out) && collect_etype_ids(ctx, r, out),
        _ => false,
    }
}

/// Edge-anchored shortcut for `MATCH (a)-[:T]->(b) RETURN count(*)`: one directed
/// fixed-length segment, no WHERE, no inline props/WHERE on either endpoint or the
/// relationship. Counts by scanning the relationship-**type** bucket(s) — the flat,
/// contiguous edge-id arrays — instead of pointer-chasing every vertex's adjacency
/// list. Unlabeled endpoints collapse to `bucket.len()` (O(1) per type); labelled
/// endpoints filter each candidate edge's two endpoints by label.
///
/// Provably identical to the general path: an edge has exactly one type, so the
/// per-type buckets are disjoint, and every stored edge of the type is exactly one
/// directed `a→b` match (self-loops included once, matching `out_adj`). `Both` is
/// left to the scalar path (its self-loop de-duplication differs from a bucket scan).
pub(super) fn try_count_edges(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    // Exactly one segment; no inline props / WHERE anywhere on the pattern.
    let [seg] = path.segments.as_slice() else {
        return None;
    };
    if !path.start.props.is_empty() || path.start.where_.is_some() {
        return None;
    }
    if !seg.node.props.is_empty() || seg.node.where_.is_some() {
        return None;
    }
    if !seg.rel.props.is_empty() || seg.rel.where_.is_some() || seg.rel.quantifier.is_some() {
        return None;
    }
    // Directed only — `Both`'s self-loop semantics differ from a bucket scan.
    let dir = seg.rel.direction;
    if !matches!(dir, Direction::Out | Direction::In) {
        return None;
    }
    // The projection is exactly `count(*)` (mirrors `try_count_star`).
    if !is_bare_count_star(proj) {
        return None;
    }
    // The relationship must name its type(s): `:T` or `:A|B`.
    let rel_label = seg.rel.label.as_ref()?;
    let ctx = resolve_ctx(graph, plan, params);
    let mut tids = Vec::new();
    if !collect_etype_ids(&ctx, rel_label, &mut tids) {
        return None;
    }

    let start_label = path.start.label.as_ref();
    let node_label = seg.node.label.as_ref();
    let unlabeled = start_label.is_none() && node_label.is_none();

    // Cardinality-based seed: when a labeled endpoint's bucket is smaller than the
    // whole edge-type set, seed from it and count matching adjacency — O(bucket·deg)
    // instead of O(E) scanning every edge. Order-independent, so this only affects
    // speed; each qualifying edge is counted once (from one endpoint's adjacency).
    let etype_total: usize = tids.iter().map(|&t| graph.edges_with_etype(t).len()).sum();
    let bucket_card = |lbl: Option<&CLabelExpr>| -> Option<usize> {
        lbl.and_then(seed_label)
            .and_then(|r| ctx.labels[r].0)
            .map(|lid| graph.vertices_with_label(lid).len())
    };
    let (start_card, node_card) = (bucket_card(start_label), bucket_card(node_label));
    // Seed from the smaller labeled endpoint (start on a tie).
    let seed_start = match (start_card, node_card) {
        (Some(s), Some(n)) => s <= n,
        (Some(_), None) => true,
        _ => false,
    };
    if let Some(sc) = if seed_start { start_card } else { node_card } {
        if sc < etype_total {
            // Seed side: which end we anchor, whether it's the edge source, and the
            // *other* end's label to validate. `dir` is Out/In (Both bailed above).
            let (seed_lbl, far_lbl, v_is_src) = if seed_start {
                (start_label, node_label, dir == Direction::Out)
            } else {
                (node_label, start_label, dir == Direction::In)
            };
            let seeds: &[u32] = seed_lbl
                .and_then(seed_label)
                .and_then(|r| ctx.labels[r].0)
                .map_or(&[], |lid| graph.vertices_with_label(lid));
            let mut count: usize = 0;
            for &v in seeds {
                // The bucket is only a *superset* for a conjunct label — re-validate.
                if !matches_label(graph, &ctx, v, seed_lbl) {
                    continue;
                }
                let hit =
                    |a: &Adj| tids.contains(&a.etype) && matches_label(graph, &ctx, a.nbr, far_lbl);
                count += if v_is_src {
                    graph.out_adj(v).filter(hit).count()
                } else {
                    graph.in_adj(v).filter(hit).count()
                };
            }
            let mut rs = RowSet::new(proj.out_names.clone());
            rs.push_row(std::iter::once(Value::Num(count as f64)));
            return Some(rs);
        }
    }

    let mut count: usize = 0;
    for tid in tids {
        let bucket = graph.edges_with_etype(tid);
        if unlabeled {
            count += bucket.len(); // every edge of this type is one match
            continue;
        }
        for &eid in bucket {
            let src = graph.e_src[eid as usize];
            let dst = graph.e_dst[eid as usize];
            // Out: `a` is the source, `b` the destination; In reverses them.
            let (a_end, b_end) = match dir {
                Direction::In => (dst, src),
                _ => (src, dst),
            };
            if matches_label(graph, &ctx, a_end, start_label)
                && matches_label(graph, &ctx, b_end, node_label)
            {
                count += 1;
            }
        }
    }
    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Some(rs)
}

/// The edge-type ids a relationship label admits: `None` = no `:T` constraint
/// (any type); `Some(v)` = exactly those types; the whole result `None` = an
/// `And`/`Not`/wildcard label with no cheap enumeration (caller bails).
pub(super) fn rel_type_set(ctx: &Ctx, label: Option<&CLabelExpr>) -> Option<Option<Vec<u32>>> {
    match label {
        None => Some(None),
        Some(expr) => {
            let mut v = Vec::new();
            collect_etype_ids(ctx, expr, &mut v).then_some(Some(v))
        }
    }
}

/// True if edge type `etype` is admitted by a `rel_type_set` result.
pub(super) fn etype_ok(set: &Option<Vec<u32>>, etype: u32) -> bool {
    set.as_ref().is_none_or(|v| v.contains(&etype))
}

/// Degree-product shortcut for a **two-hop count**:
/// `MATCH (a)-[:T1]->(b)-[:T2]->(c) RETURN count(*)`. A homomorphic two-hop count
/// is `Σ_b (edges into b that reach a valid a) × (edges out of b that reach a
/// valid c)` — every in/out edge pair at the middle vertex `b` is one path — so it
/// visits each edge O(1) times (O(E) total) instead of enumerating O(paths). No
/// materialisation, and single-threaded it beats even the parallel enumeration.
///
/// Applies only when the shape can't hide a distinctness/self-join constraint the
/// product would miss: both relationships anonymous (no var ⇒ no edge-uniqueness
/// check) and directed, no inline props/WHERE anywhere, and the three node
/// variables are pairwise distinct (no `(a)…->(a)` self-join). Endpoint/middle
/// labels are honoured by filtering the incident edges. Returns `None` otherwise.
pub(super) fn try_count_two_hop(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    let [seg1, seg2] = path.segments.as_slice() else {
        return None;
    };
    // The projection is exactly `count(*)` (mirrors `try_count_star`).
    if !is_bare_count_star(proj) {
        return None;
    }
    // Both relationships: anonymous (no edge-uniqueness to enforce), directed, no
    // inline props / WHERE / quantifier.
    for rel in [&seg1.rel, &seg2.rel] {
        if rel.var_slot.is_some()
            || !rel.props.is_empty()
            || rel.where_.is_some()
            || rel.quantifier.is_some()
            || !matches!(rel.direction, Direction::Out | Direction::In)
        {
            return None;
        }
    }
    // No inline node props / WHERE (labels are fine — applied below).
    for node in [&path.start, &seg1.node, &seg2.node] {
        if !node.props.is_empty() || node.where_.is_some() {
            return None;
        }
    }
    // Node variables must be pairwise distinct — a shared variable (e.g.
    // `(a)-[:T]->()-[:T]->(a)`) is a self-join the product can't express.
    let slots: Vec<usize> = [path.start.var_slot, seg1.node.var_slot, seg2.node.var_slot]
        .into_iter()
        .flatten()
        .collect();
    if (1..slots.len()).any(|i| slots[..i].contains(&slots[i])) {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let t1 = rel_type_set(&ctx, seg1.rel.label.as_ref())?;
    let t2 = rel_type_set(&ctx, seg2.rel.label.as_ref())?;
    let start_label = path.start.label.as_ref(); // `a`
    let mid_label = seg1.node.label.as_ref(); // `b`
    let end_label = seg2.node.label.as_ref(); // `c`

    // For the middle vertex `b`: seg1 edges reach `a` from b's *reverse* side (an
    // out-pattern `a->b` is an in-edge of b), seg2 edges reach `c` from b's
    // forward side. `Adj.nbr` is always the far endpoint, so it's the a / c to
    // label-check.
    let count_side = |b: u32, out_side: bool, tset: &Option<Vec<u32>>, far: Option<&CLabelExpr>| {
        // `out_adj`/`in_adj` are distinct opaque iterator types, so branch the whole
        // count rather than the iterator binding.
        let keep =
            |adj: &Adj| etype_ok(tset, adj.etype) && matches_label(graph, &ctx, adj.nbr, far);
        if out_side {
            graph.out_adj(b).filter(keep).count() as u64
        } else {
            graph.in_adj(b).filter(keep).count() as u64
        }
    };
    let to_a_out = seg1.rel.direction == Direction::In; // In ⇒ a via b's out-edges
    let from_c_out = seg2.rel.direction == Direction::Out; // Out ⇒ c via b's out-edges

    // Each middle vertex `b` contributes `ways_to(b) × ways_from(b)` paths.
    let contribution = |b: u32| -> u64 {
        if !matches_label(graph, &ctx, b, mid_label) {
            return 0;
        }
        let ways_to = count_side(b, to_a_out, &t1, start_label);
        if ways_to == 0 {
            return 0; // no incoming side ⇒ no paths through b
        }
        ways_to * count_side(b, from_c_out, &t2, end_label)
    };
    // Candidate middles: the middle label's bucket, else every live vertex.
    let candidates: Vec<u32> = match mid_label.and_then(seed_label) {
        Some(r) => match ctx.labels[r].0 {
            Some(lid) => graph.vertices_with_label(lid).to_vec(),
            None => Vec::new(), // unknown middle label → no rows
        },
        None => graph.vertex_indices().collect(),
    };
    // The middles are independent — split them across cores (opt-in) and sum.
    #[cfg(feature = "parallel-query")]
    let count: u64 = candidates.par_iter().map(|&b| contribution(b)).sum();
    #[cfg(not(feature = "parallel-query"))]
    let count: u64 = candidates.iter().map(|&b| contribution(b)).sum();

    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Some(rs)
}

/// Collect the variable slots referenced by `e` into `out`; `false` if `e` contains
/// a construct not analyzed here (subquery / aggregate / CASE / function call), so
/// the caller can't safely reason about which branch a predicate belongs to.
pub(super) fn expr_slot_refs(e: &CExpr, out: &mut Vec<usize>) -> bool {
    match e {
        CExpr::Var(s) => {
            out.push(*s);
            true
        }
        CExpr::Prop { var_slot, .. } => {
            out.push(*var_slot);
            true
        }
        CExpr::Param(_) | CExpr::Lit(_) => true,
        CExpr::List(xs) => xs.iter().all(|x| expr_slot_refs(x, out)),
        CExpr::Compare { left, right, .. } => {
            expr_slot_refs(left, out) && expr_slot_refs(right, out)
        }
        CExpr::Arith { head, tail } => {
            expr_slot_refs(head, out) && tail.iter().all(|(_, e)| expr_slot_refs(e, out))
        }
        CExpr::Concat(items) | CExpr::And(items) | CExpr::Or(items) | CExpr::Xor(items) => {
            items.iter().all(|e| expr_slot_refs(e, out))
        }
        CExpr::Neg(x) | CExpr::Not(x) => expr_slot_refs(x, out),
        CExpr::IsNull { expr, .. }
        | CExpr::IsTruth { expr, .. }
        | CExpr::IsLabeled { expr, .. }
        | CExpr::IsTyped { expr, .. } => expr_slot_refs(expr, out),
        CExpr::In { expr, list, .. } => expr_slot_refs(expr, out) && expr_slot_refs(list, out),
        _ => false, // Exists / CountSubquery / Case / Scalar / Aggregate / AggRef
    }
}

/// Flatten a top-level `AND` chain into its conjuncts.
pub(super) fn split_conjuncts<'a>(e: &'a CExpr, out: &mut Vec<&'a CExpr>) {
    if let CExpr::And(items) = e {
        for it in items {
            split_conjuncts(it, out);
        }
    } else {
        out.push(e);
    }
}

/// Filtered-degree-product shortcut for a comma-join count:
/// `MATCH (a:La?)-[:T1]->(b:Lb?), (a)-[:T2]->(c:Lc?) WHERE <φ> RETURN count(*)`.
///
/// The two branches share only the anchor `a`, so the number of matches at each `a`
/// is `|B(a)| · |C(a)|` — the product of the two independently-filtered out-degrees —
/// and the total is `Σ_a |B(a)|·|C(a)|`. Computing that per anchor is O(deg) instead
/// of enumerating the O(deg²) cross product the scalar join materializes. Requires
/// the `WHERE` to factor: every conjunct references at most one branch endpoint
/// (`b`-only or `c`-only, plus the anchor); a cross-branch conjunct (`b.x < c.y`)
/// can't factor, so it bails. Anonymous rels ⇒ no edge-uniqueness (homomorphism,
/// same as `try_count_two_hop`), so the plain product is exact. A global `count(*)`,
/// so there's no group order to preserve.
#[allow(clippy::too_many_lines, reason = "one self-contained count shortcut")]
pub(super) fn try_count_comma_join(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: Some(w),
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    // Exactly `count(*)`.
    if !is_bare_count_star(proj) {
        return None;
    }
    let [p1, p2] = patterns.as_slice() else {
        return None;
    };
    let ([seg1], [seg2]) = (p1.segments.as_slice(), p2.segments.as_slice()) else {
        return None;
    };
    for (p, seg) in [(p1, seg1), (p2, seg2)] {
        if !p.start.props.is_empty() || p.start.where_.is_some() {
            return None;
        }
        let rel = &seg.rel;
        if rel.var_slot.is_some()
            || !rel.props.is_empty()
            || rel.where_.is_some()
            || rel.quantifier.is_some()
            || rel.direction != Direction::Out
        {
            return None;
        }
        if !seg.node.props.is_empty() || seg.node.where_.is_some() {
            return None;
        }
    }
    // Shared anchor `a`; distinct named endpoints `b`, `c`.
    let a_slot = p1.start.var_slot?;
    if p2.start.var_slot != Some(a_slot) {
        return None;
    }
    let b_slot = seg1.node.var_slot?;
    let c_slot = seg2.node.var_slot?;
    if b_slot == c_slot || b_slot == a_slot || c_slot == a_slot {
        return None;
    }

    // Partition the WHERE conjuncts into anchor / b-branch / c-branch; bail on a
    // cross-branch conjunct or a reference to any variable other than a/b/c.
    let mut conjuncts = Vec::new();
    split_conjuncts(w, &mut conjuncts);
    let (mut a_preds, mut b_preds, mut c_preds): (Vec<&CExpr>, Vec<&CExpr>, Vec<&CExpr>) =
        (Vec::new(), Vec::new(), Vec::new());
    for conj in conjuncts {
        let mut slots = Vec::new();
        if !expr_slot_refs(conj, &mut slots) {
            return None;
        }
        if slots
            .iter()
            .any(|s| *s != a_slot && *s != b_slot && *s != c_slot)
        {
            return None;
        }
        let refs_b = slots.contains(&b_slot);
        let refs_c = slots.contains(&c_slot);
        match (refs_b, refs_c) {
            (true, true) => return None, // cross-branch — can't factor
            (true, false) => b_preds.push(conj),
            (false, true) => c_preds.push(conj),
            (false, false) => a_preds.push(conj),
        }
    }

    let ctx = resolve_ctx(graph, plan, params);
    let la1 = p1.start.label.as_ref();
    let la2 = p2.start.label.as_ref();
    let lb = seg1.node.label.as_ref();
    let lc = seg2.node.label.as_ref();
    let width = a_slot.max(b_slot).max(c_slot) + 1;

    // For one anchor `a`, the filtered out-degree of a branch: neighbours matching
    // the endpoint label and every branch predicate (with `a` + the endpoint bound).
    let branch_degree = |bind: &mut Binding,
                         a: u32,
                         dir_label: Option<&CLabelExpr>,
                         end_slot: usize,
                         end_label: Option<&CLabelExpr>,
                         preds: &[&CExpr]|
     -> u64 {
        let mut d = 0u64;
        for (_e, nbr) in expand(graph, &ctx, a, Direction::Out, dir_label) {
            if !matches_label(graph, &ctx, nbr, end_label) {
                continue;
            }
            bind.set(end_slot, Val::Node(nbr));
            let env = Env::new(graph, &ctx, bind);
            if preds.iter().all(|p| as_truth(&eval(&env, p)) == Some(true)) {
                d += 1;
            }
        }
        d
    };

    // Collect the anchors (matching both re-stated start labels), then fan the
    // independent per-anchor products across cores (opt-in `parallel-query`).
    let mut anchors: Vec<u32> = Vec::new();
    for_each_seed(graph, &ctx, la1, &mut |a| {
        if matches_label(graph, &ctx, a, la2) {
            anchors.push(a);
        }
        true
    });
    let per_anchor = |a: u32| -> u64 {
        let mut bind = Binding(vec![None; width]);
        bind.set(a_slot, Val::Node(a));
        // Anchor predicates (with only `a` bound).
        {
            let env = Env::new(graph, &ctx, &bind);
            if !a_preds
                .iter()
                .all(|p| as_truth(&eval(&env, p)) == Some(true))
            {
                return 0;
            }
        }
        let d1 = branch_degree(&mut bind, a, seg1.rel.label.as_ref(), b_slot, lb, &b_preds);
        if d1 == 0 {
            return 0;
        }
        let d2 = branch_degree(&mut bind, a, seg2.rel.label.as_ref(), c_slot, lc, &c_preds);
        d1 * d2
    };
    #[cfg(feature = "parallel-query")]
    let count: u64 = anchors.par_iter().map(|&a| per_anchor(a)).sum();
    #[cfg(not(feature = "parallel-query"))]
    let count: u64 = anchors.iter().map(|&a| per_anchor(a)).sum();

    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Some(rs)
}

/// Degree-product shortcut for a **var-length `{1,2}` count**:
/// `MATCH (a:La?)-[:T]->{1,2}(b:Lb?) RETURN count(*)` — count length-1 + length-2
/// trails without enumerating every trail (which is O(#trails), quadratic in
/// degree). Directed `Out`, no edge variable / inline props / WHERE.
///
/// - Length-1 trails = matching single edges: `Σ_{a:La} out_T→Lb(a)`.
/// - Length-2 trails `a→x→y` (edges distinct) = `Σ_x in_T←La(x) · out_T→Lb(x)`
///   minus the self-loop double-count: a self-loop `e` at `x` is both an in- and
///   out-edge, so the product counts the invalid `a→a→a` that reuses `e` for both
///   hops (forbidden — a trail traverses each edge at most once). It's subtracted
///   only when `x` matches both endpoints' labels (it is the `a` *and* the `b`).
///
/// Other quantifiers/directions fall through to the enumerating parallel count.
pub(super) fn try_count_varlen_1_2(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    let [seg] = path.segments.as_slice() else {
        return None;
    };
    // Exactly `count(*)` (mirrors try_count_star).
    if !is_bare_count_star(proj) {
        return None;
    }
    // The one relationship: `{1,2}`, directed Out, anonymous, no inline props/WHERE.
    let rel = &seg.rel;
    if rel.var_slot.is_some()
        || !rel.props.is_empty()
        || rel.where_.is_some()
        || rel.direction != Direction::Out
    {
        return None;
    }
    match rel.quantifier {
        Some(q) if q.min == 1 && q.max == Some(2) => {}
        _ => return None,
    }
    // Start / endpoint: no inline props/WHERE (labels are fine, applied below).
    if !path.start.props.is_empty()
        || path.start.where_.is_some()
        || !seg.node.props.is_empty()
        || seg.node.where_.is_some()
    {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let tset = rel_type_set(&ctx, rel.label.as_ref())?;
    let la = path.start.label.as_ref(); // the `a` end
    let lb = seg.node.label.as_ref(); // the `b` end
    let out_to_lb = |x: u32| -> u64 {
        graph
            .out_adj(x)
            .filter(|a| etype_ok(&tset, a.etype) && matches_label(graph, &ctx, a.nbr, lb))
            .count() as u64
    };
    let in_from_la = |x: u32| -> u64 {
        graph
            .in_adj(x)
            .filter(|a| etype_ok(&tset, a.etype) && matches_label(graph, &ctx, a.nbr, la))
            .count() as u64
    };
    let self_loops = |x: u32| -> u64 {
        graph
            .out_adj(x)
            .filter(|a| etype_ok(&tset, a.etype) && a.nbr == x)
            .count() as u64
    };
    // Per middle-vertex `x`: (length-1 from x as `a`, length-2 through x, self-loop
    // correction). Every live vertex is a candidate middle (the intermediate is
    // unconstrained); `a`/`b` labels gate the length-1 and correction terms.
    let contribution = |x: u32| -> (u64, u64, u64) {
        let out_lb = out_to_lb(x);
        let l2 = in_from_la(x) * out_lb;
        let mut l1 = 0;
        let mut corr = 0;
        if matches_label(graph, &ctx, x, la) {
            l1 = out_lb; // `x` is a valid start `a`
            if matches_label(graph, &ctx, x, lb) {
                corr = self_loops(x); // invalid a→a→a reusing the self-loop
            }
        }
        (l1, l2, corr)
    };
    let candidates: Vec<u32> = graph.vertex_indices().collect();
    let add = |a: (u64, u64, u64), b: (u64, u64, u64)| (a.0 + b.0, a.1 + b.1, a.2 + b.2);
    #[cfg(feature = "parallel-query")]
    let (l1, l2, corr) = candidates
        .par_iter()
        .map(|&x| contribution(x))
        .reduce(|| (0, 0, 0), add);
    #[cfg(not(feature = "parallel-query"))]
    let (l1, l2, corr) = candidates
        .iter()
        .map(|&x| contribution(x))
        .fold((0, 0, 0), add);
    let count = l1 + l2 - corr;

    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Some(rs)
}

/// Map `f` over `0..n` into a `Vec`, across rayon threads when `parallel-query` is
/// on (else serial). Used for the independent per-vertex degree passes in the
/// grouped count shortcuts — the same `par_iter`/`iter` split the other shortcuts
/// (`try_count_two_hop`, …) use, factored so the call sites stay `cfg`-free.
#[cfg(feature = "parallel-query")]
pub(super) fn par_map<T: Send>(n: usize, f: impl Fn(u32) -> T + Sync + Send) -> Vec<T> {
    (0..n as u32).into_par_iter().map(f).collect()
}
#[cfg(not(feature = "parallel-query"))]
pub(super) fn par_map<T>(n: usize, f: impl Fn(u32) -> T) -> Vec<T> {
    (0..n as u32).map(f).collect()
}

/// Does `e` reference only slot `slot`? Conservative: a bare var or a direct
/// property of it. Anything else (arithmetic, another var) bails — so the grouped
/// var-length shortcut only fires when every group key is a value of the endpoint.
pub(super) fn expr_refs_only_slot(e: &CExpr, slot: usize) -> bool {
    match e {
        CExpr::Var(s) => *s == slot,
        CExpr::Prop { var_slot, .. } => *var_slot == slot,
        _ => false,
    }
}

/// Grouped var-length count shortcut:
/// `MATCH (a:La?)-[:T]->{lo,hi}(b:Lb?) RETURN <key(b)…>, count(*)` with `hi <= 2`.
///
/// **Why it's exact.** At bound ≤2, ISO trail semantics (each edge once) coincides
/// with walk semantics — the shortest edge-reusing walk has length 3 — so per-
/// endpoint *trail* multiplicity is just the walk count, a guarded frequency
/// propagation: `into[x]` = #`T`-edges into `x` from a valid start; a `b`'s
/// multiplicity is `[len-0] + into[b] (len-1) + Σ_{x→b} into[x] (len-2) − self-loop
/// correction`. `count(*)` grouped by a value of the endpoint is a guarded
/// aggregate, so each endpoint's multiplicity is added to its own group. This is
/// O(V+E) instead of enumerating every trail endpoint (the scalar path's cost).
///
/// **Order.** Group *counts* are exact; the group *first-seen order* (contractual
/// for a non-`ORDER BY` aggregate) is recovered by replaying the scalar walk order
/// (`reachable` per seed, endpoint filtered by `Lb`) only until every group — whose
/// full set is already known from the O(V+E) pass — has appeared. For a low-
/// cardinality group key that stops almost immediately; worst case it costs no more
/// than the scalar enumeration it replaces (and it never groups a 14M-row stream).
#[allow(clippy::too_many_lines, reason = "one self-contained count shortcut")]
pub(super) fn try_grouped_varlen_1_2(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    let [seg] = path.segments.as_slice() else {
        return None;
    };
    let rel = &seg.rel;
    if rel.var_slot.is_some()
        || !rel.props.is_empty()
        || rel.where_.is_some()
        || rel.direction != Direction::Out
    {
        return None;
    }
    // A bounded quantifier with `hi <= 2` (where trail == walk); `*`/`+`/`hi>2` stay
    // scalar (edge-uniqueness bites at length ≥3).
    let q = rel.quantifier?;
    let hi = q.max?;
    if hi > 2 || q.min > hi {
        return None;
    }
    let (lo, hi) = (q.min, hi);
    if !path.start.props.is_empty()
        || path.start.where_.is_some()
        || !seg.node.props.is_empty()
        || seg.node.where_.is_some()
    {
        return None;
    }
    let b_slot = seg.node.var_slot?;
    // A grouped `count(*)`: no DISTINCT/SKIP/LIMIT/`*`, no ORDER BY (first-seen only
    // for v1), at least one non-agg key, exactly one bare `count(*)`.
    if proj.distinct
        || proj.skip.is_some()
        || proj.limit.is_some()
        || proj.star
        || !proj.order_by.is_empty()
        || !proj.aggregating
        || proj.aggs.len() != 1
    {
        return None;
    }
    let agg = &proj.aggs[0];
    if !agg.star || agg.distinct || !matches!(agg.func, AggFn::Count) {
        return None;
    }
    // Every non-agg item is a group key over `b`; the one agg item is a bare count.
    let key_items = proj.group_keys();
    if key_items.is_empty() {
        return None; // a global count uses `try_count_varlen_1_2`
    }
    for it in &key_items {
        if !expr_refs_only_slot(&it.expr, b_slot) {
            return None;
        }
    }
    if !proj
        .items
        .iter()
        .any(|i| i.is_agg && matches!(i.expr, CExpr::AggRef(0)))
    {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let tset = rel_type_set(&ctx, rel.label.as_ref())?;
    let la = path.start.label.as_ref();
    let lb = seg.node.label.as_ref();
    let n = graph.n;

    // into[x] = # `T`-edges a→x with `a` matching La (the length-1 count into x).
    let into: Vec<u64> = par_map(n, |x| {
        graph
            .in_adj(x)
            .filter(|a| etype_ok(&tset, a.etype) && matches_label(graph, &ctx, a.nbr, la))
            .count() as u64
    });

    // Per-endpoint trail multiplicity (bound ≤2 ⇒ trail == walk).
    let mult: Vec<i64> = par_map(n, |b| {
        if !graph.is_vertex_live(b) {
            return 0;
        }
        let bi = b as usize;
        let mut m: i64 = 0;
        let in_la = matches_label(graph, &ctx, b, la);
        if lo == 0 && in_la {
            m += 1; // length-0: the start itself
        }
        if lo <= 1 {
            m += into[bi] as i64; // length-1: a→b
        }
        // length-2: a→x→b over in-edges x→b (hi is always ≥2 here when lo≤2<hi
        // is false; guard on hi).
        if hi >= 2 {
            let l2: i64 = graph
                .in_adj(b)
                .filter(|a| etype_ok(&tset, a.etype))
                .map(|a| into[a.nbr as usize] as i64)
                .sum();
            m += l2;
            if in_la {
                // Trail correction: a→b→b reusing the same self-loop edge.
                let sl = graph
                    .out_adj(b)
                    .filter(|a| etype_ok(&tset, a.etype) && a.nbr == b)
                    .count() as i64;
                m -= sl;
            }
        }
        m
    });

    // Accumulate group counts (order-independent): endpoints matching Lb with a
    // positive multiplicity, keyed by the `val_key` of their group-key values.
    let mut groups: FxHashMap<String, (Vec<Val>, i64)> = FxHashMap::default();
    let mut bb = Binding(vec![None; b_slot + 1]);
    let mut key_buf = String::new();
    for b in 0..n as u32 {
        let m = mult[b as usize];
        if m <= 0 || !matches_label(graph, &ctx, b, lb) {
            continue;
        }
        bb.set(b_slot, Val::Node(b));
        let vals: Vec<Val> = {
            let env = Env::new(graph, &ctx, &bb);
            key_items.iter().map(|it| eval_item(&env, it)).collect()
        };
        key_buf.clear();
        for v in &vals {
            val_key(v, &mut key_buf);
            key_buf.push('\u{1}');
        }
        let entry = groups.entry(key_buf.clone()).or_insert_with(|| (vals, 0));
        entry.1 += m;
    }

    // Recover first-seen group order by replaying the scalar walk order until every
    // group has appeared (the group set is already fixed by `groups`).
    let target = groups.len();
    let mut seen: FxHashSet<String> = HashSet::with_capacity_and_hasher(target, Default::default());
    let mut order: Vec<String> = Vec::with_capacity(target);
    let mut faulted = false;
    for_each_seed(graph, &ctx, la, &mut |a| {
        for end in reachable(graph, &ctx, a, rel, q, path.mode) {
            if !matches_label(graph, &ctx, end, lb) {
                continue;
            }
            bb.set(b_slot, Val::Node(end));
            key_buf.clear();
            {
                let env = Env::new(graph, &ctx, &bb);
                for it in &key_items {
                    val_key(&eval_item(&env, it), &mut key_buf);
                    key_buf.push('\u{1}');
                }
            }
            if seen.insert(key_buf.clone()) {
                order.push(key_buf.clone());
                if order.len() == target {
                    return false; // every group seen — stop the walk
                }
            }
        }
        if ctx.faulted() {
            faulted = true;
            return false;
        }
        true
    });
    if faulted {
        return None; // trail budget blew — let the scalar path surface it
    }

    // Emit one row per group in first-seen order: group-key values interleaved with
    // the count, following the projection's item order.
    let mut rs = RowSet::new(proj.out_names.clone());
    for key in &order {
        let (vals, cnt) = &groups[key];
        let mut ki = 0;
        rs.push_row(proj.items.iter().map(|it| {
            if it.is_agg {
                Value::Num(*cnt as f64)
            } else {
                let v = val_to_value(graph, &vals[ki]);
                ki += 1;
                v
            }
        }));
    }
    Some(rs)
}

/// Fixed two-hop count GROUPED by an endpoint value:
/// `MATCH (a:La?)-[:T1]->(b:Lb?)-[:T2]->(c:Lc?) RETURN <key(c)…>, count(*)`.
///
/// Analogous to [`try_grouped_varlen_1_2`] but for two *fixed* directed segments,
/// so — anonymous rels ⇒ homomorphism (no edge-uniqueness, same as
/// [`try_count_two_hop`]) — it's plain **walk** counting with NO self-loop
/// correction. Per endpoint `c`: `Σ_{b→c via T2} into[b]`, where `into[b]` = #`T1`-
/// edges into a valid middle `b` from a valid start `a`. O(V+E) instead of
/// enumerating the O(deg²) two-hop rows. Counts exact; first-seen group order
/// recovered by replaying the scalar nested expansion until every (already-known)
/// group appears.
#[allow(clippy::too_many_lines, reason = "one self-contained count shortcut")]
pub(super) fn try_grouped_2hop(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    let [seg1, seg2] = path.segments.as_slice() else {
        return None;
    };
    for rel in [&seg1.rel, &seg2.rel] {
        if rel.var_slot.is_some()
            || !rel.props.is_empty()
            || rel.where_.is_some()
            || rel.quantifier.is_some()
            || rel.direction != Direction::Out
        {
            return None;
        }
    }
    for node in [&path.start, &seg1.node, &seg2.node] {
        if !node.props.is_empty() || node.where_.is_some() {
            return None;
        }
    }
    // Distinct node variables; the endpoint `c` must be named (it's the group key).
    let a_slot = path.start.var_slot;
    let b_slot = seg1.node.var_slot;
    let c_slot = seg2.node.var_slot?;
    let named: Vec<usize> = [a_slot, b_slot, Some(c_slot)]
        .into_iter()
        .flatten()
        .collect();
    if (1..named.len()).any(|i| named[..i].contains(&named[i])) {
        return None;
    }
    if proj.distinct
        || proj.skip.is_some()
        || proj.limit.is_some()
        || proj.star
        || !proj.order_by.is_empty()
        || !proj.aggregating
        || proj.aggs.len() != 1
    {
        return None;
    }
    let agg = &proj.aggs[0];
    if !agg.star || agg.distinct || !matches!(agg.func, AggFn::Count) {
        return None;
    }
    let key_items = proj.group_keys();
    if key_items.is_empty() {
        return None;
    }
    for it in &key_items {
        if !expr_refs_only_slot(&it.expr, c_slot) {
            return None;
        }
    }
    if !proj
        .items
        .iter()
        .any(|i| i.is_agg && matches!(i.expr, CExpr::AggRef(0)))
    {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let t1 = rel_type_set(&ctx, seg1.rel.label.as_ref())?;
    let t2 = rel_type_set(&ctx, seg2.rel.label.as_ref())?;
    let la = path.start.label.as_ref();
    let lb = seg1.node.label.as_ref();
    let lc = seg2.node.label.as_ref();
    let n = graph.n;

    // into[b] = #`T1`-edges a→b with `a` matching La, if `b` is a valid middle.
    let into: Vec<u64> = par_map(n, |b| {
        if !matches_label(graph, &ctx, b, lb) {
            return 0;
        }
        graph
            .in_adj(b)
            .filter(|e| etype_ok(&t1, e.etype) && matches_label(graph, &ctx, e.nbr, la))
            .count() as u64
    });
    // Per endpoint `c` (matching Lc): Σ over T2-edges b→c of into[b] (walk count).
    let mult: Vec<i64> = par_map(n, |c| {
        if !graph.is_vertex_live(c) || !matches_label(graph, &ctx, c, lc) {
            return 0;
        }
        graph
            .in_adj(c)
            .filter(|e| etype_ok(&t2, e.etype))
            .map(|e| into[e.nbr as usize] as i64)
            .sum()
    });

    // Accumulate group counts, then recover first-seen order (both share the tail
    // shape with `try_grouped_varlen_1_2`).
    let mut groups: FxHashMap<String, (Vec<Val>, i64)> = FxHashMap::default();
    let mut bb = Binding(vec![None; c_slot + 1]);
    let mut key_buf = String::new();
    for c in 0..n as u32 {
        let m = mult[c as usize];
        if m <= 0 {
            continue;
        }
        bb.set(c_slot, Val::Node(c));
        let vals: Vec<Val> = {
            let env = Env::new(graph, &ctx, &bb);
            key_items.iter().map(|it| eval_item(&env, it)).collect()
        };
        key_buf.clear();
        for v in &vals {
            val_key(v, &mut key_buf);
            key_buf.push('\u{1}');
        }
        let entry = groups.entry(key_buf.clone()).or_insert_with(|| (vals, 0));
        entry.1 += m;
    }

    let target = groups.len();
    let mut seen: FxHashSet<String> = HashSet::with_capacity_and_hasher(target, Default::default());
    let mut order: Vec<String> = Vec::with_capacity(target);
    'seeds: for a in {
        let mut starts: Vec<u32> = Vec::new();
        for_each_seed(graph, &ctx, la, &mut |v| {
            starts.push(v);
            true
        });
        starts
    } {
        for be in expand(graph, &ctx, a, Direction::Out, seg1.rel.label.as_ref()) {
            if !matches_label(graph, &ctx, be.1, lb) {
                continue;
            }
            for ce in expand(graph, &ctx, be.1, Direction::Out, seg2.rel.label.as_ref()) {
                if !matches_label(graph, &ctx, ce.1, lc) {
                    continue;
                }
                bb.set(c_slot, Val::Node(ce.1));
                key_buf.clear();
                {
                    let env = Env::new(graph, &ctx, &bb);
                    for it in &key_items {
                        val_key(&eval_item(&env, it), &mut key_buf);
                        key_buf.push('\u{1}');
                    }
                }
                if seen.insert(key_buf.clone()) {
                    order.push(key_buf.clone());
                    if order.len() == target {
                        break 'seeds;
                    }
                }
            }
        }
    }

    let mut rs = RowSet::new(proj.out_names.clone());
    for key in &order {
        let (vals, cnt) = &groups[key];
        let mut ki = 0;
        rs.push_row(proj.items.iter().map(|it| {
            if it.is_agg {
                Value::Num(*cnt as f64)
            } else {
                let v = val_to_value(graph, &vals[ki]);
                ki += 1;
                v
            }
        }));
    }
    Some(rs)
}

/// `MATCH (m:La?)-[:T]->(n:Lb?) WITH n [, <aggs>] RETURN count(*)`: the outer
/// `count(*)` counts the WITH's rows — one per distinct `n` — and the aggregates
/// are computed only to be discarded. So the answer is just the number of distinct
/// endpoints `n` (matching `Lb`) with at least one `T`-edge from a start matching
/// `La`. That's a per-vertex membership test, O(V+E), instead of materializing +
/// grouping every `(m,n)` row. A global count ⇒ no group order to preserve.
pub(super) fn try_count_distinct_endpoint(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::With {
        projection: wp,
        where_: None,
        ..
    }, CClause::Return(rp)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    let [seg] = path.segments.as_slice() else {
        return None;
    };
    let rel = &seg.rel;
    if rel.var_slot.is_some()
        || !rel.props.is_empty()
        || rel.where_.is_some()
        || rel.quantifier.is_some()
        || rel.direction != Direction::Out
        || !path.start.props.is_empty()
        || path.start.where_.is_some()
        || !seg.node.props.is_empty()
        || seg.node.where_.is_some()
    {
        return None;
    }
    let n_slot = seg.node.var_slot?;
    // The WITH groups by exactly the bare endpoint `n` (its aggregates are discarded
    // by the outer count). A property key / extra key / non-aggregating WITH is a
    // different distinct set.
    if !wp.aggregating {
        return None;
    }
    let key_items = wp.group_keys();
    if key_items.len() != 1 || !matches!(key_items[0].expr, CExpr::Var(s) if s == n_slot) {
        return None;
    }
    // The RETURN is exactly `count(*)`.
    if rp.distinct
        || !rp.order_by.is_empty()
        || rp.skip.is_some()
        || rp.limit.is_some()
        || rp.out_len != 1
        || rp.aggs.len() != 1
        || rp.items.len() != 1
        || !matches!(rp.items[0].expr, CExpr::AggRef(0))
    {
        return None;
    }
    let agg = &rp.aggs[0];
    if !agg.star || agg.distinct || !matches!(agg.func, AggFn::Count) {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let tset = rel_type_set(&ctx, rel.label.as_ref())?;
    let la = path.start.label.as_ref();
    let lb = seg.node.label.as_ref();
    // A distinct endpoint `n`: matches `Lb` and has ≥1 `T`-edge from an `La` start.
    let reached = |n: u32| -> u64 {
        if !matches_label(graph, &ctx, n, lb) {
            return 0;
        }
        u64::from(
            graph
                .in_adj(n)
                .any(|e| etype_ok(&tset, e.etype) && matches_label(graph, &ctx, e.nbr, la)),
        )
    };
    // Candidate endpoints: the `Lb` bucket, else every live vertex.
    let candidates: Vec<u32> = match lb.and_then(seed_label) {
        Some(r) => match ctx.labels[r].0 {
            Some(lid) => graph.vertices_with_label(lid).to_vec(),
            None => Vec::new(),
        },
        None => graph.vertex_indices().collect(),
    };
    #[cfg(feature = "parallel-query")]
    let count: u64 = candidates.par_iter().map(|&n| reached(n)).sum();
    #[cfg(not(feature = "parallel-query"))]
    let count: u64 = candidates.iter().map(|&n| reached(n)).sum();

    let mut rs = RowSet::new(rp.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Some(rs)
}

/// Reverse semi-join for a correlated `EXISTS` count:
/// `MATCH (a:La?) WHERE [NOT] EXISTS { (a)-[:T]->(b:Lb) } RETURN count(*)`.
///
/// The satisfying `a`s are exactly the `T`-predecessors of the `Lb` vertices, so
/// when `Lb` is more selective than `La`, seed the small `Lb` bucket and collect
/// the distinct `a`s from its reverse adjacency — O(|Lb|·degree) — instead of
/// testing `EXISTS` for every one of the many `a`s (O(|La|·degree)). `EXISTS` →
/// the predecessor count; `NOT EXISTS` → `|La|` minus it.
///
/// Tightly gated: a single bare correlated start, a single directed non-var-length
/// inner segment with no edge variable / props / WHERE, a labeled (seedable) fresh
/// inner endpoint, and `Lb` smaller than `La`. Anything else falls through to the
/// per-row `any_match`.
pub(super) fn try_count_semi_join(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: Some(w),
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [outer] = patterns.as_slice() else {
        return None;
    };
    // Outer is a bare node `(a:La?)` — the rows are the `a`s.
    if !outer.segments.is_empty() || !outer.start.props.is_empty() || outer.start.where_.is_some() {
        return None;
    }
    let a_slot = outer.start.var_slot?;
    // Exactly `count(*)` (mirrors try_count_star).
    if !is_bare_count_star(proj) {
        return None;
    }
    // Unwrap `EXISTS { … }` or `NOT EXISTS { … }`.
    let (inner_patterns, inner_where, negated) = match w {
        CExpr::Exists {
            patterns, where_, ..
        } => (patterns, where_, false),
        CExpr::Not(inner) => match inner.as_ref() {
            CExpr::Exists {
                patterns, where_, ..
            } => (patterns, where_, true),
            _ => return None,
        },
        _ => return None,
    };
    if inner_where.is_some() {
        return None;
    }
    let [inner] = inner_patterns.as_slice() else {
        return None;
    };
    let [seg] = inner.segments.as_slice() else {
        return None;
    };
    // Inner start is the correlated `a` (bare, same slot). Inner endpoint `b` is a
    // fresh selective node — not `a` (no self-referential `(a)-[:T]->(a)`).
    if inner.start.var_slot != Some(a_slot)
        || inner.start.label.is_some()
        || !inner.start.props.is_empty()
        || inner.start.where_.is_some()
        || seg.node.var_slot == Some(a_slot)
        || !seg.node.props.is_empty()
        || seg.node.where_.is_some()
    {
        return None;
    }
    let rel = &seg.rel;
    if rel.var_slot.is_some()
        || !rel.props.is_empty()
        || rel.where_.is_some()
        || rel.quantifier.is_some()
        || !matches!(rel.direction, Direction::Out | Direction::In)
    {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let la = outer.start.label.as_ref();
    let lb = seg.node.label.as_ref();
    let tset = rel_type_set(&ctx, rel.label.as_ref())?;
    // `Lb` must seed a bucket; only reverse-seed when it's smaller than `La`.
    let lb_bucket: &[u32] = lb
        .and_then(seed_label)
        .and_then(|r| ctx.labels[r].0)
        .map_or(&[], |lid| graph.vertices_with_label(lid));
    let la_card = match la.and_then(seed_label).and_then(|r| ctx.labels[r].0) {
        Some(lid) => graph.vertices_with_label(lid).len(),
        None => graph.vertex_count(),
    };
    if lb.is_none() || lb_bucket.len() >= la_card {
        return None;
    }

    // Distinct `a`s reachable back from the `Lb` bucket over `T`. For `(a)-[:T]->b`
    // (Out) `a` is `b`'s in-neighbor; for `(a)<-[:T]-b` (In) `a` is `b`'s out-neighbor.
    let out_side = rel.direction == Direction::In;
    let mut preds: FxHashSet<u32> = FxHashSet::default();
    for &b in lb_bucket {
        if !matches_label(graph, &ctx, b, lb) {
            continue; // conjunct label: the bucket is only a superset
        }
        let keep =
            |adj: &Adj| etype_ok(&tset, adj.etype) && matches_label(graph, &ctx, adj.nbr, la);
        if out_side {
            for adj in graph.out_adj(b).filter(keep) {
                preds.insert(adj.nbr);
            }
        } else {
            for adj in graph.in_adj(b).filter(keep) {
                preds.insert(adj.nbr);
            }
        }
    }
    let semi = preds.len();
    let count = if negated {
        la_card.saturating_sub(semi)
    } else {
        semi
    };

    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Some(rs)
}

/// Frontier-marking shortcut for `count(DISTINCT <endpoint node>)` over a plain
/// fixed-length traversal: `MATCH (a:La?)-[:T]->…->(c:Lc?) RETURN count(DISTINCT c)`.
///
/// The answer is the size of the **set of vertices reachable** as `c` — path
/// *multiplicity* is irrelevant to a DISTINCT count, so instead of enumerating
/// every path (O(#paths), exponential in hops) propagate a deduped frontier level
/// by level (each level dedups, so a vertex expands once) and return the final
/// frontier size — O(depth·E). Walk-vs-trail doesn't matter: both reach the same
/// vertex set.
///
/// Gated: single plain MATCH (no WHERE), a fixed-length non-var-length path with no
/// edge variable / props / WHERE, no repeated node variable (self-join), and the
/// DISTINCT argument is exactly the final node's variable.
pub(super) fn try_count_distinct_reachable(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<RowSet> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    if path.segments.is_empty() || path.segments.iter().any(|s| s.rel.quantifier.is_some()) {
        return None;
    }
    // Projection is exactly `count(DISTINCT <var>)`.
    if proj.distinct
        || !proj.order_by.is_empty()
        || proj.skip.is_some()
        || proj.limit.is_some()
        || proj.out_len != 1
        || proj.aggs.len() != 1
        || proj.items.len() != 1
        || !matches!(proj.items[0].expr, CExpr::AggRef(0))
        || !proj.group_by.is_empty()
    {
        return None;
    }
    let agg = &proj.aggs[0];
    if agg.star || !agg.distinct || !matches!(agg.func, AggFn::Count) {
        return None;
    }
    // The DISTINCT argument must be the final node's variable.
    let end_slot = path.segments[path.segments.len() - 1].node.var_slot?;
    if !matches!(&agg.arg, Some(CExpr::Var(s)) if *s == end_slot) {
        return None;
    }
    // No edge variable / props / WHERE on any relationship; no inline node
    // props / WHERE (labels are fine, applied per frontier level).
    for seg in &path.segments {
        if seg.rel.var_slot.is_some()
            || !seg.rel.props.is_empty()
            || seg.rel.where_.is_some()
            || !seg.node.props.is_empty()
            || seg.node.where_.is_some()
        {
            return None;
        }
    }
    if !path.start.props.is_empty() || path.start.where_.is_some() {
        return None;
    }
    // No repeated node variable — a self-join (`(a)…->(a)`) constrains endpoints in
    // a way plain reachability can't express.
    let slots: Vec<usize> = std::iter::once(&path.start)
        .chain(path.segments.iter().map(|s| &s.node))
        .filter_map(|n| n.var_slot)
        .collect();
    let mut seen_slots = HashSet::new();
    if slots.iter().any(|s| !seen_slots.insert(*s)) {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    // Seed frontier: distinct start vertices matching the start label.
    let mut cur: Vec<u32> = Vec::new();
    for_each_seed(graph, &ctx, path.start.label.as_ref(), &mut |v| {
        if matches_label(graph, &ctx, v, path.start.label.as_ref()) {
            cur.push(v);
        }
        true
    });
    // Expand level by level, deduping each frontier so every vertex expands once.
    for seg in &path.segments {
        let mut seen = crate::graph::BitSet::zeros(graph.vertex_count());
        let mut next: Vec<u32> = Vec::new();
        for &v in &cur {
            for (_e, w) in expand(graph, &ctx, v, seg.rel.direction, seg.rel.label.as_ref()) {
                if !seen.get(w as usize) && matches_label(graph, &ctx, w, seg.node.label.as_ref()) {
                    seen.set(w as usize);
                    next.push(w);
                }
            }
        }
        cur = next;
    }
    let count = cur.len();

    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Some(rs)
}

/// Start vertices matching a bare seed node (label + inline props/WHERE), using a
/// property index when the inline map / WHERE offers one, else a label scan.
pub(super) fn reach_seed_vertices(
    graph: &Graph,
    ctx: &Ctx,
    start: &CNode,
    scope_len: usize,
) -> Vec<u32> {
    let needs_check = !start.props.is_empty() || start.where_.is_some();
    let mut b = Binding(vec![None; scope_len.max(1)]);
    let mut out = Vec::new();
    let ok = |graph: &Graph, vi: u32, b: &mut Binding| -> bool {
        if !matches_label(graph, ctx, vi, start.label.as_ref()) {
            return false;
        }
        if needs_check {
            if let Some(s) = start.var_slot {
                b.set(s, Val::Node(vi));
            }
            if !satisfies(
                graph,
                ctx,
                &Val::Node(vi),
                &start.props,
                start.where_.as_ref(),
                b,
            ) {
                return false;
            }
        }
        true
    };
    // The node's own WHERE anchors the seek just as an inline `{k: v}` does; see
    // the note in `scan_start_seed`. `satisfies` above re-checks it either way, so
    // seeking only narrows the candidates.
    match node_index_seed(graph, ctx, start, start.where_.as_ref()) {
        Some(cands) => {
            for vi in cands {
                if graph.is_vertex_live(vi) && ok(graph, vi, &mut b) {
                    out.push(vi);
                }
            }
        }
        None => {
            for_each_seed(graph, ctx, start.label.as_ref(), &mut |vi| {
                if ok(graph, vi, &mut b) {
                    out.push(vi);
                }
                true
            });
        }
    }
    out
}

/// Whether `expr` reads only the endpoint variable `b` (a bare `b` or `b.<prop>`).
/// A projection that also reads the start `a` (or an intermediate) can't be served
/// by a reachability set, which loses the per-path source correspondence.
pub(super) fn refs_only_endpoint(expr: &CExpr, b: usize) -> bool {
    match expr {
        CExpr::Var(s) => *s == b,
        CExpr::Prop { var_slot, .. } => *var_slot == b,
        CExpr::Lit(_) => true,
        _ => false,
    }
}

/// Reachability shortcut for **unbounded var-length with DISTINCT**:
/// `MATCH (a:La {..})-[:T]->+(b:Lb?) RETURN DISTINCT <b…>` (also `->*` and
/// `count(DISTINCT b)`). Trail enumeration is exponential on a connected graph and
/// hits the trail budget (a *fault*), but a DISTINCT result only wants the reachable
/// *set* — multiplicity is collapsed — which a plain O(V+E) graph search answers.
/// `->+` = reachable via ≥1 hop; `->*` also includes the seed(s).
///
/// Gated to a single unbounded (`max = None`) directed segment (no edge var / props
/// / WHERE), a DISTINCT projection with no ORDER BY that reads only the endpoint,
/// and the endpoint bound. Bounded quantifiers keep enumerating (already small).
pub(super) fn try_reachable_distinct(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        scope_len,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    let [seg] = path.segments.as_slice() else {
        return None;
    };
    // Only an *unbounded* quantifier blows up; bounded `{lo,hi}` enumeration is small.
    let q = seg.rel.quantifier?;
    if q.max.is_some() {
        return None;
    }
    if seg.rel.var_slot.is_some()
        || !seg.rel.props.is_empty()
        || seg.rel.where_.is_some()
        || !matches!(seg.rel.direction, Direction::Out | Direction::In)
        || !seg.node.props.is_empty()
        || seg.node.where_.is_some()
    {
        return None;
    }
    let b_slot = seg.node.var_slot?;
    if !proj.order_by.is_empty() {
        return None;
    }
    // DISTINCT rows over `b`, or `count(DISTINCT <b…>)`.
    let rows_mode = proj.distinct
        && !proj.aggregating
        && proj
            .items
            .iter()
            .all(|it| refs_only_endpoint(&it.expr, b_slot));
    let count_mode = proj.aggregating
        && proj.aggs.len() == 1
        && proj.items.len() == 1
        && matches!(proj.items[0].expr, CExpr::AggRef(0))
        && {
            let a = &proj.aggs[0];
            a.distinct
                && !a.star
                && matches!(a.func, AggFn::Count)
                && a.arg
                    .as_ref()
                    .is_some_and(|e| refs_only_endpoint(e, b_slot))
        };
    if !rows_mode && !count_mode {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let seeds = reach_seed_vertices(graph, &ctx, &path.start, *scope_len);
    // Forward reachability (≥1 hop) as a closed DFS — each vertex expands once, in
    // discovery order.
    let (dir, el) = (seg.rel.direction, seg.rel.label.as_ref());
    let mut reached: Vec<u32> = Vec::new();
    reach_dfs_forward(graph, &ctx, seeds.iter().copied(), dir, el, |w| {
        reached.push(w);
        true
    });
    // `->*` also admits the zero-length path — the seeds themselves. `reach_dfs_forward`
    // dedups internally but its `seen` isn't exposed; a seed already ≥1-hop reachable is
    // therefore re-added here, so guard against duplicating it.
    if q.min == 0 {
        let mut have: FxHashSet<u32> = reached.iter().copied().collect();
        for &s in &seeds {
            if have.insert(s) {
                reached.push(s);
            }
        }
    }

    let lb = seg.node.label.as_ref();
    let width = (*scope_len).max(1);
    let mut bind = Binding(vec![None; width]);

    if count_mode {
        let arg = proj.aggs[0].arg.as_ref();
        let (mut ids, mut strs) = (HashSet::new(), HashSet::new());
        let mut n = 0u64;
        for &v in &reached {
            if !matches_label(graph, &ctx, v, lb) {
                continue;
            }
            bind.set(b_slot, Val::Node(v));
            let val = match arg {
                Some(e) => eval(&Env::new(graph, &ctx, &bind), e),
                None => Val::Node(v),
            };
            if is_nullish(&val) {
                continue;
            }
            let novel = match &val {
                Val::Node(i) => ids.insert(*i as u64),
                Val::Edge(i) => ids.insert(*i as u64 | EDGE_ID_TAG),
                _ => {
                    let mut k = String::new();
                    val_key(&val, &mut k);
                    strs.insert(k)
                }
            };
            if novel {
                n += 1;
            }
        }
        let mut rs = RowSet::new(proj.out_names.clone());
        rs.push_row(std::iter::once(Value::Num(n as f64)));
        return Some(Ok(rs));
    }

    // rows_mode: project the endpoint per reached vertex, dedup the output tuples.
    let mut rs = RowSet::new(proj.out_names.clone());
    let mut seen_rows: FxHashSet<String> = FxHashSet::default();
    for &v in &reached {
        if !matches_label(graph, &ctx, v, lb) {
            continue;
        }
        bind.set(b_slot, Val::Node(v));
        let env = Env::new(graph, &ctx, &bind);
        let vals: Vec<Val> = proj.items.iter().map(|it| eval_item(&env, it)).collect();
        let mut key = String::new();
        for val in &vals {
            val_key(val, &mut key);
            key.push('\u{1}');
        }
        if seen_rows.insert(key) {
            rs.push_row(vals.iter().map(|val| val_to_value(graph, val)));
        }
    }
    rs.apply_skip_limit(proj.skip_val(&ctx), proj.limit_val(&ctx));
    Some(Ok(rs))
}

/// Intra-query parallel count for `MATCH <path with ≥1 segment> [WHERE …] RETURN
/// count(*)` — the read-only traversal count that stays scalar (a pure aggregate
/// over a traversal isn't vectorized, and `try_count_edges` only covers a single
/// WHERE-less segment). The seed vertices are split across rayon threads; each
/// runs the **same** single-threaded matcher over its chunk with a thread-local
/// counter and its own binding, then the partials are summed — the "accumulator"
/// model. `Graph`/`Ctx` are `Sync` and the walk is read-only, so this is a pure
/// latency win (the outer seed loop is embarrassingly parallel). Any WHERE fault
/// is recorded atomically and surfaced via `check_fault` exactly as the serial
/// path would. Returns `None` below a seed threshold (serial keeps small queries
/// off the thread hand-off) or when the shape doesn't qualify.
#[cfg(feature = "parallel-query")]
pub(super) fn try_parallel_count(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_,
        where_prog,
        scope_len,
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    // Traversals only — a bare-node scan/filter count has its own fast paths.
    if path.segments.is_empty() {
        return None;
    }
    // The projection is exactly `count(*)` (mirrors `try_count_star`).
    if !is_bare_count_star(proj) {
        return None;
    }

    let threads = rayon::current_num_threads();
    if threads <= 1 {
        return None;
    }
    let ctx = resolve_ctx(graph, plan, params);

    // Seed set — mirror `match_one_path`: a bare start label seeds its bucket,
    // otherwise every live vertex.
    let seeds: Vec<u32> = match path.start.label.as_ref().and_then(seed_label) {
        Some(r) => match ctx.labels[r].0 {
            Some(lid) => graph.vertices_with_label(lid).to_vec(),
            None => return Some(count_rows(proj, 0)), // unknown label → 0
        },
        None => graph.vertex_indices().collect(),
    };
    // Below this, the thread hand-off would dominate the walk — stay serial.
    const MIN_SEEDS: usize = 8_192;
    if seeds.len() < MIN_SEEDS {
        return None;
    }

    let cwhere = where_.as_ref();
    let cwhere_prog = where_prog.as_ref();
    let width = (*scope_len).max(1);
    // Chunk for work-stealing balance while keeping per-chunk overhead low.
    let chunk = (seeds.len() / (threads * 4)).max(1_024);
    let count: u64 = seeds
        .par_chunks(chunk)
        .map(|chunk| {
            let mut local = 0u64;
            let mut b = Binding(vec![None; width]);
            for &s in chunk {
                if ctx.faulted() {
                    break; // a sibling chunk already faulted — stop early
                }
                match_node_continue(graph, &ctx, &mut b, &path.start, s, path, 0, &mut |bnd| {
                    if where_keep(&Env::new(graph, &ctx, bnd), cwhere, cwhere_prog) {
                        local += 1;
                    }
                    true // never stop — a full count visits every match
                });
            }
            local
        })
        .sum();

    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }
    Some(count_rows(proj, count))
}

/// Build the single-row `count(*)` result for a projection.
#[cfg(feature = "parallel-query")]
pub(super) fn count_rows(proj: &CProjection, count: u64) -> CodeResult<RowSet> {
    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Ok(rs)
}

/// Intra-query parallel **aggregation** for `MATCH <traversal> [WHERE …] RETURN
/// <group keys>, <aggregates>` — the general form of [`try_parallel_count`].
/// Aggregating over a traversal isn't vectorized, so it stream-folds one match at
/// a time on a single thread; here the seed vertices are split across rayon
/// threads, each folds its matches into a thread-local [`ProjAccum`], and the
/// partials are reduced in seed order (`ProjAccum::merge`) — which reproduces the
/// serial first-seen group order exactly, so the result is byte-identical.
///
/// Gated to traversals (a bare-node scan aggregate is already vectorized) with
/// **non-DISTINCT** aggregates (a distinct fold can't be merged from partials).
/// Var-length is fine (same per-seed matcher as `try_parallel_count`).
/// `ORDER BY`/`SKIP`/`LIMIT` are applied by `finish` after the merge, so they're
/// fine.
#[cfg(feature = "parallel-query")]
pub(super) fn try_parallel_agg(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_,
        where_prog,
        scope_len,
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    if !proj.aggregating || proj.star {
        return None;
    }
    // DISTINCT aggregates can't merge from partial (sum, seen) state — stay serial.
    if proj.aggs.iter().any(|a| a.distinct) {
        return None;
    }
    // Anchor at the first pattern's start node: every complete match binds it, so
    // partitioning the seeds by it is a clean split (no double-count, no miss). A
    // single path uses the direct matcher; a comma-join pre-binds the anchor and
    // drives all patterns via `drive_matches`.
    let anchor = &patterns[0].start;
    let single = patterns.len() == 1;
    // A comma-join needs a variable anchor to pre-bind. Traversals only (a bare-node
    // aggregate is already vectorized). Var-length is fine — the matcher (`reachable`)
    // is all-local plus the shared atomic fault, exactly as `try_parallel_count` runs
    // it per-seed, so splitting the seed loop is still a pure latency win.
    if !single && anchor.var_slot.is_none() {
        return None;
    }
    if patterns.iter().all(|p| p.segments.is_empty()) {
        return None;
    }

    let threads = rayon::current_num_threads();
    if threads <= 1 {
        return None;
    }
    let ctx = resolve_ctx(graph, plan, params);
    let seeds: Vec<u32> = match anchor.label.as_ref().and_then(seed_label) {
        Some(r) => match ctx.labels[r].0 {
            Some(lid) => graph.vertices_with_label(lid).to_vec(),
            None => Vec::new(), // unknown label → no matches (finish emits the empty result)
        },
        None => graph.vertex_indices().collect(),
    };
    const MIN_SEEDS: usize = 8_192;
    if seeds.len() < MIN_SEEDS {
        return None;
    }

    let cwhere = where_.as_ref();
    let cwhere_prog = where_prog.as_ref();
    let width = (*scope_len).max(1);
    let anchor_slot = anchor.var_slot;
    let match_clause: [&CClause; 1] = [&linear.clauses[0]]; // for drive_matches
    let chunk = (seeds.len() / (threads * 4)).max(1_024);
    // Per-chunk accumulator; rayon preserves chunk order, so the reduce below sees
    // chunks in seed order and reproduces the serial first-seen group order.
    let accs: Vec<ProjAccum> = seeds
        .par_chunks(chunk)
        .map(|chunk| {
            let mut acc = ProjAccum::new(proj, &ctx);
            let mut b = Binding(vec![None; width]);
            for &s in chunk {
                if ctx.faulted() {
                    break;
                }
                if single {
                    // Direct matcher; the clause WHERE is applied per emitted match.
                    match_node_continue(
                        graph,
                        &ctx,
                        &mut b,
                        anchor,
                        s,
                        &patterns[0],
                        0,
                        &mut |bnd| {
                            if where_keep(&Env::new(graph, &ctx, bnd), cwhere, cwhere_prog) {
                                acc.accept(graph, &ctx, bnd);
                            }
                            true
                        },
                    );
                } else {
                    // Comma-join: pre-bind the anchor, drive every pattern (which
                    // applies the clause WHERE itself), fold each complete match.
                    b.0.iter_mut().for_each(|c| *c = None);
                    b.set(anchor_slot.unwrap(), Val::Node(s));
                    drive_matches(graph, &ctx, &match_clause, 0, &mut b, &mut |bnd| {
                        acc.accept(graph, &ctx, bnd);
                        true
                    });
                }
            }
            acc
        })
        .collect();

    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }
    let mut merged = ProjAccum::new(proj, &ctx);
    for a in accs {
        merged.merge(a);
    }
    let bindings = merged.finish(graph, &ctx);

    let mut rs = RowSet::new(proj.out_names.clone());
    for b in bindings {
        rs.push_row((0..proj.out_len).map(|i| {
            b.get(i)
                .map(|v| val_to_value(graph, v))
                .unwrap_or(Value::Null)
        }));
    }
    Some(Ok(rs))
}

/// Intra-query parallel **row materialization** for `MATCH <traversal> [WHERE …]
/// RETURN <plain projection>` — the row-returning analogue of [`try_parallel_agg`].
/// The vectorized builder ([`build_scan`]) enumerates the whole join into columns
/// on one thread; here the filtered start seeds are split across rayon threads,
/// each runs [`expand_scan`] over its chunk (+ the clause WHERE mask) and projects
/// its slice to a [`RowSet`] fragment, and the fragments are concatenated in seed
/// order — reproducing the serial row order exactly, so the result is byte-identical.
///
/// Gated to a single fresh traversal MATCH with a plain projection (no aggregate /
/// DISTINCT / ORDER BY — those reorder or fold and stay on the existing paths) and
/// no var-length. A LIMIT with no WHERE is left to the serial scan, which early-
/// stops it cheaply (parallel would build every row first). Below a seed threshold
/// it declines so small queries skip the thread hand-off.
#[cfg(feature = "parallel-query")]
pub(super) fn try_parallel_scan(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    let [CClause::Match {
        optional: false,
        patterns,
        where_,
        scope_len,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    // Plain row projection only — aggregate/DISTINCT/ORDER BY reorder or fold and
    // are handled by try_parallel_agg / the vectorized column path.
    if proj.star || proj.aggregating || proj.distinct || !proj.order_by.is_empty() {
        return None;
    }
    let [path] = patterns.as_slice() else {
        return None;
    };
    // Traversals only (an isolated-node scan is a cheap bucket clone), non-var-length.
    if path.segments.is_empty() || path.segments.iter().any(|s| s.rel.quantifier.is_some()) {
        return None;
    }
    // A LIMIT with no WHERE lets the serial scan stop early — don't intercept it
    // (parallel would materialize every row before truncating). With a WHERE the
    // scan can't early-stop, so building all rows in parallel is a pure win.
    if where_.is_none() && proj.limit.is_some() {
        return None;
    }

    let threads = rayon::current_num_threads();
    if threads <= 1 {
        return None;
    }
    let ctx = resolve_ctx(graph, plan, params);
    // Orient exactly as build_scan does; decline (→ serial path) for any shape it
    // wouldn't orient (an index / edge-property seek), so serial and parallel seed
    // from the identical end and produce the identical row order.
    let oriented = try_orient_node_seed(graph, &ctx, path, where_.as_ref())?;
    let start_ids = scan_start_seed(graph, &ctx, &oriented.start, *scope_len, where_.as_ref());
    const MIN_SEEDS: usize = 8_192;
    if start_ids.len() < MIN_SEEDS {
        return None;
    }

    let w = where_.as_ref();
    let chunk = (start_ids.len() / (threads * 4)).max(1_024);
    // Each chunk builds + filters + projects independently; rayon preserves chunk
    // order, so concatenating the fragments reproduces the serial row order.
    let frags: Vec<Option<RowSet>> = start_ids
        .par_chunks(chunk)
        .map(|c| {
            let mut sc = expand_scan(
                graph,
                &ctx,
                &oriented,
                *scope_len,
                c.to_vec(),
                super::scan::SeedFrom {
                    cap: None,
                    needed: None,
                    carry: None,
                },
            )?;
            if let Some(w) = w {
                let keep: Vec<bool> = eval_vec(graph, &ctx, &sc, w)
                    .into_truth()
                    .iter()
                    .map(|t| *t == Some(true))
                    .collect();
                compact(&mut sc, &keep);
            }
            Some(project_scan_rows(graph, &ctx, &sc, proj))
        })
        .collect();

    // A `None` fragment = a self-join expand_scan can't vectorize (shape-based, so
    // every chunk agrees) — decline and let the serial vectorized/scalar path run.
    if frags.iter().any(Option::is_none) {
        return None;
    }
    // A data exception during the vectorized WHERE can't return `Err` from here;
    // decline so the scalar path re-evaluates and surfaces the `CodeError` — the
    // same fallback the serial vectorized path uses.
    if ctx.faulted() {
        return None;
    }

    let total: usize = frags.iter().flatten().map(|f| f.nrows).sum();
    let mut rs = RowSet::new(proj.out_names.clone());
    rs.data.reserve(total * proj.out_len.max(1));
    for f in frags.into_iter().flatten() {
        rs.nrows += f.nrows;
        rs.data.extend(f.data);
    }
    rs.apply_skip_limit(proj.skip_val(&ctx), proj.limit_val(&ctx));
    Some(Ok(rs))
}
