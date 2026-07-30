//! Index-seeded scanning: pick the most selective index seed for a pattern
//! (prop_index_hint / node_index_seed / try_orient_node_seed), build the scan,
//! expand it edge-first, and run vectorized grouping/aggregation over the result
//! (fused_global_agg / vectorized_aggregate / fold_group_agg_cols). Extracted from
//! the evaluator (`super`); shares its context/helpers via `use super::*`.
use super::*;

/// Scatter the frame's bound element slots into binding `b` at row `ri`: each
/// bound slot becomes the Node/Edge id sitting at that row. Value/scalar slots are
/// filled separately by the caller. Shared by every place that materializes a
/// representative row from the columnar frame (grouped aggregation, sort binding).
#[inline]
fn bind_frame_row(b: &mut Binding, sc: &ScanCols, ri: usize) {
    for (slot, col) in sc.slots.iter().enumerate() {
        if let Some((elem, ids)) = col {
            b.set(
                slot,
                match elem {
                    Elem::Node => Val::Node(ids[ri]),
                    Elem::Edge => Val::Edge(ids[ri]),
                },
            );
        }
    }
}

/// Resolve `(var_slot, key_ref)` in the scan frame to the property `Column` the
/// slot's element type stores it in, plus the row→element-id map for that slot.
/// Outer `None` = the slot isn't bound to an element; inner `None` = that property
/// column is absent (the caller decides how to treat an absent column).
fn prop_col<'a>(
    graph: &'a Graph,
    ctx: &Ctx,
    sc: &'a ScanCols,
    var_slot: usize,
    key_ref: usize,
) -> Option<(Option<&'a Column>, &'a [u32])> {
    let (elem, ids) = sc.slot(var_slot)?;
    let (store, kid) = match elem {
        Elem::Node => (&graph.props, ctx.prop_keys[key_ref].0),
        Elem::Edge => (&graph.edge_props, ctx.prop_keys[key_ref].1),
    };
    Some((kid.and_then(|k| store.cols.get(k as usize)), ids))
}

// --- vertex/edge-agnostic index seeks (dispatched by an `edge` flag) ---------
pub(super) fn idx_indexed(graph: &Graph, name: &str, edge: bool) -> bool {
    if edge {
        graph.edge_indexed(name)
    } else {
        graph.vertex_indexed(name)
    }
}
pub(super) fn idx_eq(
    graph: &Graph,
    name: &str,
    k: &crate::graph::IdxKey,
    edge: bool,
) -> Option<Vec<u32>> {
    if edge {
        graph.edges_by_prop(name, k).map(<[u32]>::to_vec)
    } else {
        graph.vertices_by_prop(name, k).map(<[u32]>::to_vec)
    }
}
pub(super) fn idx_range(
    graph: &Graph,
    name: &str,
    rb: &crate::graph::RangeBound,
    edge: bool,
) -> Option<Vec<u32>> {
    if edge {
        graph.edges_by_prop_range(name, rb)
    } else {
        graph.vertices_by_prop_range(name, rb)
    }
}
/// The property name a `Prop` key-ref resolves to (vertex or edge store).
pub(super) fn prop_name<'a>(
    graph: &'a Graph,
    ctx: &Ctx,
    key_ref: usize,
    edge: bool,
) -> Option<&'a str> {
    let (vk, ek) = ctx.prop_keys[key_ref];
    if edge {
        Some(graph.edge_props.keys.text(ek?))
    } else {
        Some(graph.props.keys.text(vk?))
    }
}

/// An index seek from a WHERE comparison `var.key OP <literal>` where `var` is at
/// `want_slot` (`None` = any), against the vertex or edge index. An `AND` of two
/// same-var/same-key comparisons coalesces into one tight range seek; else the
/// first usable conjunct. Returns candidate element ids.
pub(super) fn prop_index_hint(
    graph: &Graph,
    ctx: &Ctx,
    e: &CExpr,
    want_slot: Option<usize>,
    edge: bool,
) -> Option<Vec<u32>> {
    use crate::graph::RangeBound;
    let slot_ok = |s: usize| want_slot.is_none_or(|w| w == s);
    match e {
        CExpr::Compare { op, left, right } => {
            // Handles a bare `var.key` AND a nested `var.a.b` (dotted-path index).
            let (vslot, path) = prop_path(left, graph, ctx, edge)?;
            if !slot_ok(vslot) {
                return None;
            }
            let key = expr_to_idxkey(right, ctx)?;
            if !idx_indexed(graph, &path, edge) {
                return None;
            }
            if *op == CompareOp::Eq {
                return idx_eq(graph, &path, &key, edge);
            }
            let mut rb = RangeBound::default();
            apply_bound(&mut rb, *op, key);
            idx_range(graph, &path, &rb, edge)
        }
        CExpr::And(items) => {
            // Contender B: an as-of over an edge RI-tree interval index (`lo_key <= $v
            // AND hi_key > $v`, same var, same probe) seeds from `stab($v)` — the small
            // "active at v" set directly — instead of the BTreeMap seeks A would pick.
            // Same recognizer style, different seek; falls through to A when no interval
            // index covers the pair. The stab is inclusive of `hi == v`, a superset the
            // final `WHERE hi > v` then verifies.
            if edge {
                let probe = |name: &str, want_lo: bool| -> Option<(usize, i128)> {
                    items.iter().find_map(|it| {
                        let (s, kref, op, key) = cmp_bound(it, ctx)?;
                        let ok = if want_lo {
                            matches!(op, CompareOp::Le | CompareOp::Lt)
                        } else {
                            matches!(op, CompareOp::Gt | CompareOp::Ge)
                        };
                        if !ok || !slot_ok(s) || prop_name(graph, ctx, kref, true)? != name {
                            return None;
                        }
                        match key {
                            crate::graph::IdxKey::Temporal(_, q) => Some((s, q)),
                            _ => None,
                        }
                    })
                };
                // Each edge interval index whose `lo_key ≤/< qlo AND hi_key >/≥ qhi` pair
                // is present in the conjunction can seed a candidate SUPERSET (a point stab
                // when qlo==qhi, else a window-normalized overlap — min/max is essential, or
                // `overlap(qhi,qlo)` on a contains shape would be min>max → empty → a silent
                // miss). For the bitemporal 4-way we DON'T intersect the axes: one axis is
                // often non-selective — e.g. "believed now" (`tt=∞`) matches ~every version,
                // so materializing its stab and intersecting is O(all rows) and can lose to
                // a scan. Instead compare axes' sizes CHEAPLY via *_len (no materialization),
                // seed from the MOST SELECTIVE axis only, and let the final WHERE verify the
                // rest. (An enum: stab point, or overlap window.)
                enum Probe {
                    Stab(i128),
                    Overlap(i128, i128),
                }
                let mut best: Option<(usize, Probe, usize)> = None; // (index, probe, size)
                for (n, (lo_key, hi_key)) in graph.edge_interval_index_specs().iter().enumerate() {
                    let (Some((slo, qlo)), Some((shi, qhi))) =
                        (probe(lo_key, true), probe(hi_key, false))
                    else {
                        continue;
                    };
                    if slo != shi {
                        continue;
                    }
                    let (pr, len) = if qlo == qhi {
                        (Probe::Stab(qlo), graph.edge_interval_stab_len_nth(n, qlo))
                    } else {
                        let (d1, d2) = (qlo.min(qhi), qlo.max(qhi));
                        (
                            Probe::Overlap(d1, d2),
                            graph.edge_interval_overlap_len_nth(n, d1, d2),
                        )
                    };
                    if best.as_ref().is_none_or(|(_, _, b)| len < *b) {
                        best = Some((n, pr, len));
                    }
                }
                if let Some((n, pr, _)) = best {
                    return Some(match pr {
                        Probe::Stab(q) => graph.edge_interval_stab_nth(n, q),
                        Probe::Overlap(d1, d2) => graph.edge_interval_overlap_nth(n, d1, d2),
                    });
                }
            }
            // Group every indexed single-column comparison by (var slot, key ref) and
            // fold all comparisons on that key into one tight `RangeBound`; then SEEK
            // each group and INTERSECT the candidate sets across groups, driven from
            // the smallest. A conjunction ANDs necessary conditions, so the true
            // matches lie in every group's seek — the intersection is a correct
            // candidate superset (the final WHERE re-verifies, so extra rows are
            // dropped and none are missed). One group reproduces the old same-key
            // band seek (`x>=a AND x<=b`); two groups make an interval-containment
            // as-of (`vf<=v AND vt>v`) a real seek instead of only the first column;
            // four groups do the bitemporal 4-way.
            let mut groups: std::collections::HashMap<(usize, usize), RangeBound> =
                std::collections::HashMap::new();
            for it in items {
                let Some((s, kref, op, key)) = cmp_bound(it, ctx) else {
                    continue;
                };
                if !slot_ok(s) || matches!(op, CompareOp::Ne) {
                    continue;
                }
                match prop_name(graph, ctx, kref, edge) {
                    Some(name) if idx_indexed(graph, name, edge) => {
                        apply_bound(groups.entry((s, kref)).or_default(), op, key);
                    }
                    _ => {}
                }
            }
            // Return the MOST SELECTIVE single group's seek (smallest candidate set);
            // the final WHERE verifies the remaining conjuncts. Blind intersection of
            // the groups was measured to TANK on a low-selectivity conjunct — building
            // and probing two ~200k halves costs more than the scan it replaces — so
            // pick the best column instead of AND-ing them. (Getting the small "active
            // at v" set *without* materializing the huge halves is exactly the RI-tree's
            // job; this is the non-structure baseline it must beat.) The old code
            // picked the *first* usable conjunct, which is why an as-of seeded on the
            // non-selective `vf`; choosing by result size fixes that.
            if let Some(best) = groups
                .iter()
                .filter_map(|((_, kref), rb)| {
                    idx_range(graph, prop_name(graph, ctx, *kref, edge)?, rb, edge)
                })
                .min_by_key(Vec::len)
            {
                return Some(best);
            }
            // No indexed conjunct grouped (e.g. only dotted paths) — try the first
            // usable single conjunct via the recursive single-Compare path.
            items
                .iter()
                .find_map(|it| prop_index_hint(graph, ctx, it, want_slot, edge))
        }
        _ => None,
    }
}

/// Candidate vertices for a single-node scan: an indexed inline `{key: lit}`
/// equality, or a WHERE comparison on the node. `None` ⇒ full scan.
pub(super) fn node_index_seed(
    graph: &Graph,
    ctx: &Ctx,
    node: &CNode,
    where_: Option<&CExpr>,
) -> Option<Vec<u32>> {
    for pc in &node.props {
        if graph.vertex_indexed(&pc.key) {
            // Inline `{key: lit}` OR `{key: $param}` — both resolve to a seek.
            if let Some(k) = expr_to_idxkey(&pc.value, ctx) {
                return graph.vertices_by_prop(&pc.key, &k).map(<[u32]>::to_vec);
            }
        }
    }
    where_.and_then(|w| prop_index_hint(graph, ctx, w, node.var_slot, false))
}

/// Candidate edges for a single-segment pattern: an indexed inline `[r {key:lit}]`
/// equality, or a WHERE comparison on the relationship var. `None` ⇒ no edge seed.
/// Seed the candidate edges of a pattern's relationship from the always-on edge
/// **type** index (`by_etype`) — the analogue of seeding a node scan from its
/// label bucket. Handles a single type `:T` (one bucket) and a disjunction
/// `:A|B` (union of buckets; an edge has one type, so the buckets are disjoint).
/// A missing type name yields an empty seed (no edge matches — itself a win).
/// `And`/`Not`/wildcard fall through to `None` (no cheap enumeration / no gain).
pub(super) fn etype_label_seed(graph: &Graph, ctx: &Ctx, expr: &CLabelExpr) -> Option<Vec<u32>> {
    match expr {
        CLabelExpr::Label(r) => Some(
            ctx.labels[*r]
                .1
                .map_or_else(Vec::new, |t| graph.edges_with_etype(t).to_vec()),
        ),
        CLabelExpr::Or(l, r) => {
            let mut a = etype_label_seed(graph, ctx, l)?;
            a.extend(etype_label_seed(graph, ctx, r)?);
            Some(a)
        }
        _ => None,
    }
}

/// A *selective* edge seed: an indexed edge-property equality (inline `{k: v}`) or
/// a seekable WHERE hint on the edge variable. Excludes the edge-type fallback, so
/// the caller can decide between this true seek and node-side seeding.
pub(super) fn edge_prop_seed(
    graph: &Graph,
    ctx: &Ctx,
    rel: &CRel,
    where_: Option<&CExpr>,
) -> Option<Vec<u32>> {
    for pc in &rel.props {
        if graph.edge_indexed(&pc.key) {
            if let CExpr::Lit(lit) = &pc.value {
                if let Some(k) = lit_to_idxkey(lit) {
                    return graph.edges_by_prop(&pc.key, &k).map(<[u32]>::to_vec);
                }
            }
        }
    }
    where_.and_then(|w| prop_index_hint(graph, ctx, w, rel.var_slot, true))
}

pub(super) fn edge_index_seed(
    graph: &Graph,
    ctx: &Ctx,
    rel: &CRel,
    where_: Option<&CExpr>,
) -> Option<Vec<u32>> {
    // Prefer a (usually more selective) property hint; otherwise seed from the
    // edge type. edge_first_build re-validates label + props, so a type seed is
    // a correct superset for any extra constraints.
    edge_prop_seed(graph, ctx, rel, where_).or_else(|| {
        rel.label
            .as_ref()
            .and_then(|lbl| etype_label_seed(graph, ctx, lbl))
    })
}

/// Flip a relationship direction for path reversal (`Out`↔`In`; `Both` fixed).
pub(super) fn flip_direction(d: Direction) -> Direction {
    match d {
        Direction::Out => Direction::In,
        Direction::In => Direction::Out,
        Direction::Both => Direction::Both,
    }
}

/// Walk a fixed-length path from its other end: reverse the segment order and flip
/// each relationship's direction. The matched bindings are identical (same edges /
/// nodes) — only the seed side, and thus enumeration order, change. Mirrors the TS
/// engine's `reversePath` so both engines can seed the same end.
pub(super) fn reverse_path(path: &CPath) -> CPath {
    // Nodes in written order: [start, seg0.node, seg1.node, …].
    let n = path.segments.len();
    let node_at = |i: usize| -> &CNode {
        if i == 0 {
            &path.start
        } else {
            &path.segments[i - 1].node
        }
    };
    let mut segments = Vec::with_capacity(n);
    for i in (0..n).rev() {
        let seg = &path.segments[i];
        segments.push(CSegment {
            rel: CRel {
                direction: flip_direction(seg.rel.direction),
                ..seg.rel.clone()
            },
            node: node_at(i).clone(),
            unit: seg.unit.clone(),
        });
    }
    CPath {
        start: path.segments[n - 1].node.clone(),
        segments,
        // Reversing swaps the endpoints but not what the path binds to.
        path_var_slot: path.path_var_slot,
        selector: path.selector,
        mode: path.mode,
    }
}

/// Estimated seed count for anchoring a pattern at `node`: its label bucket size,
/// or all live vertices when unlabeled. Drives orientation; index hints are handled
/// separately (a hinted node keeps the pattern on the index-seed path).
pub(super) fn estimate_seed_card(graph: &Graph, ctx: &Ctx, node: &CNode) -> usize {
    match node.label.as_ref().and_then(seed_label) {
        Some(r) => ctx.labels[r]
            .0
            .map_or(0, |lid| graph.vertices_with_label(lid).len()),
        None => graph.vertex_count(),
    }
}

/// Cardinality-based orientation for a **label-only** fixed-length traversal: pick
/// the more selective node end to seed from, reversing the path if the far end is
/// smaller. Returns the (possibly reversed) path to seed via `scan_start_seed` +
/// `expand_scan`, or `None` to leave the pattern on its existing path.
///
/// Bails for anything with an index seek or edge/where property hint (those are
/// handled by `edge_first_build` / the isolated seek) or a var-length segment, so
/// this only ever *replaces* the O(E) edge-type-bucket scan with an O(seeds·degree)
/// node walk — never abandons a more selective seek. Used by both `build_scan` and
/// `try_parallel_scan`, so serial and parallel seed identically.
pub(super) fn try_orient_node_seed(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    where_: Option<&CExpr>,
) -> Option<CPath> {
    if path.segments.is_empty() || path.segments.iter().any(|s| s.rel.quantifier.is_some()) {
        return None;
    }
    let end_node = &path.segments[path.segments.len() - 1].node;
    // Any edge property / WHERE hint means edge_first_build has a selective seed,
    // which beats any node seed — checked first so it still wins outright.
    for seg in &path.segments {
        if !seg.rel.props.is_empty()
            || seg.rel.where_.is_some()
            || edge_prop_seed(graph, ctx, &seg.rel, where_).is_some()
        {
            return None;
        }
    }
    // A real index seek on an endpoint is the best seed available, so orient
    // TOWARD it rather than declining to act. This used to bail whenever either
    // endpoint was seekable ("don't interfere with a real index seek") — which
    // left a *target*-anchored pattern, `(e:Emp)-[:T]->(m:Emp {id: $m})`, seeding
    // from the unindexed source and scanning its entire label bucket on every
    // lookup. Seeding the start was fixed separately; this is the mirror case.
    let start_seek = node_index_seed(graph, ctx, &path.start, where_).is_some();
    let end_seek = node_index_seed(graph, ctx, end_node, where_).is_some();

    if start_seek {
        return Some(path.clone()); // already leads with the seekable end
    }

    if end_seek {
        return Some(reverse_path(path)); // flip so the seekable end leads
    }
    // Orient to the smaller end. A strict `<` keeps the written orientation on a
    // tie, matching the TS engine's `orient`.
    let start_est = estimate_seed_card(graph, ctx, &path.start);
    let end_est = estimate_seed_card(graph, ctx, end_node);
    Some(if end_est < start_est {
        reverse_path(path)
    } else {
        path.clone()
    })
}

/// Whether `build_scan` will turn this scan into an index seek (so a LIMIT cap
/// can't early-stop it and should be dropped). Only a *genuine* seek counts: a
/// node/edge property index. A label-only traversal seeds a label bucket and
/// expands, which `expand_scan` **can** early-stop at the cap — so it is not
/// "hinted" (the edge-type fallback must not drop the cap, else `LIMIT n` with no
/// WHERE materializes every row before slicing).
pub(super) fn scan_is_hinted(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    where_: Option<&CExpr>,
) -> bool {
    if path.segments.is_empty() {
        node_index_seed(graph, ctx, &path.start, where_).is_some()
    } else if path.segments.len() == 1 {
        edge_prop_seed(graph, ctx, &path.segments[0].rel, where_).is_some()
    } else {
        false
    }
}

pub(super) fn build_scan(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    scope_len: usize,
    cap: Option<usize>,
    where_: Option<&CExpr>,
) -> Option<ScanCols> {
    // A path selector (`ANY SHORTEST`) or a bound path variable is handled only by
    // the scalar driver — only it builds the Path value.
    if path.selector != PathSelector::Walk || path.path_var_slot.is_some() {
        return None;
    }
    // Fast path: an isolated node is a tight scan. An index hint (inline `{k:v}`
    // eq or a WHERE comparison on the node) seeds just the candidate vertices;
    // otherwise the label bucket / all-live range. Either way the node's label +
    // inline constraints are re-checked.
    if path.segments.is_empty() {
        let node = &path.start;
        let seed = node_index_seed(graph, ctx, node, where_);
        let mut ids = Vec::new();
        let needs_check = !node.props.is_empty() || node.where_.is_some();

        // Fast path: no index seed, no inline props/WHERE, and the label is either
        // absent or a **bare** single label. Then bucket membership already implies
        // the label (`matches_label` would be a redundant per-vertex re-check), so
        // clone the live-vertex / label-bucket slice straight into the id column —
        // skipping 1M closure calls + label scans. Anything richer (And/Or/Not
        // label, inline constraints, index seed) falls through to the general loop.
        // `Some(None)` = all live vertices; `Some(Some(slice))` = a label bucket;
        // `None` = not fast-path-eligible (fall through to the general loop).
        let fast_bucket: Option<Option<&[u32]>> = if seed.is_some() || needs_check {
            None
        } else {
            match node.label.as_ref() {
                None => Some(None),
                Some(CLabelExpr::Label(r)) => Some(Some(match ctx.labels[*r].0 {
                    Some(lid) => graph.vertices_with_label(lid),
                    None => &[], // unknown label → no rows
                })),
                Some(_) => None, // And/Or/Not label needs the per-vertex re-check
            }
        };
        if let Some(bucket) = fast_bucket {
            let ids: Vec<u32> = match (bucket, cap) {
                (Some(b), Some(c)) => b.iter().take(c).copied().collect(),
                (Some(b), None) => b.to_vec(),
                (None, Some(c)) => graph.vertex_indices().take(c).collect(),
                (None, None) => graph.vertex_indices().collect(),
            };
            let mut sc = ScanCols::new(scope_len);
            sc.n = ids.len();
            if let Some(s) = node.var_slot {
                sc.slots[s] = Some((Elem::Node, ids));
            }
            return Some(sc);
        }

        let mut b = Binding(vec![None; scope_len.max(1)]);
        let consider = |graph: &Graph, vi: u32, ids: &mut Vec<u32>, b: &mut Binding| -> bool {
            if !matches_label(graph, ctx, vi, node.label.as_ref()) {
                return true;
            }
            if needs_check {
                if let Some(s) = node.var_slot {
                    b.set(s, Val::Node(vi));
                }
                if !satisfies(
                    graph,
                    ctx,
                    &Val::Node(vi),
                    &node.props,
                    node.where_.as_ref(),
                    b,
                ) {
                    return true;
                }
            }
            ids.push(vi);
            cap.is_none_or(|c| ids.len() < c)
        };
        match seed {
            Some(cands) => {
                for vi in cands {
                    if graph.is_vertex_live(vi) && !consider(graph, vi, &mut ids, &mut b) {
                        break;
                    }
                }
            }
            None => {
                for_each_seed(graph, ctx, node.label.as_ref(), &mut |vi| {
                    consider(graph, vi, &mut ids, &mut b)
                });
            }
        }
        let mut sc = ScanCols::new(scope_len);
        sc.n = ids.len();
        if let Some(s) = node.var_slot {
            sc.slots[s] = Some((Elem::Node, ids));
        }
        return Some(sc);
    }
    if path.segments.iter().any(|s| s.rel.quantifier.is_some()) {
        return None;
    }
    // Cardinality-based orientation: a label-only traversal seeds from its more
    // selective node end and walks its adjacency (O(seeds·degree)) instead of
    // scanning the whole edge-type bucket (O(E)). Same decision as
    // `try_parallel_scan`, so the serial and parallel paths seed identically.
    if let Some(oriented) = try_orient_node_seed(graph, ctx, path, where_) {
        let endpoint = scan_start_seed(graph, ctx, &oriented.start, scope_len);
        return expand_scan(graph, ctx, &oriented, scope_len, endpoint, cap);
    }
    // Edge-first: a single segment with an indexed edge-property hint → seek the
    // matching edges and validate the surrounding (a)-[r]->(b) pattern, instead
    // of expanding every vertex's adjacency.
    if path.segments.len() == 1 {
        // A *selective* edge seed (an indexed edge property) is always worth taking.
        // The `by_etype` fallback is not: it materializes every edge of the type,
        // O(E_type), which loses badly whenever an endpoint is index-seekable —
        // that seeds a handful of vertices and walks their adjacency, O(seeds·deg).
        // `try_orient_node_seed` above deliberately bails on an indexed endpoint so
        // as "not to interfere with a real index seek"; without this guard control
        // fell straight through to here and an indexed anchor *diverted* the plan
        // into the whole-type scan — making the index actively harmful.
        let endpoint_seekable = node_index_seed(graph, ctx, &path.start, where_).is_some()
            || node_index_seed(graph, ctx, &path.segments[0].node, where_).is_some();
        let seed = if endpoint_seekable {
            edge_prop_seed(graph, ctx, &path.segments[0].rel, where_)
        } else {
            edge_index_seed(graph, ctx, &path.segments[0].rel, where_)
        };
        if let Some(edges) = seed {
            return edge_first_build(graph, ctx, path, scope_len, &edges);
        }
    }
    // Seed the start-node endpoints, then expand the segments into columns.
    let endpoint = scan_start_seed(graph, ctx, &path.start, scope_len);
    expand_scan(graph, ctx, path, scope_len, endpoint, cap)
}

/// The filtered start-node endpoints for a traversal scan: every live vertex that
/// matches the start node's label + inline props/WHERE, in seed order. Split off
/// from [`build_scan`] so the parallel driver can chunk it — a contiguous slice of
/// this feeds [`expand_scan`] to build a contiguous slice of the full result.
pub(super) fn scan_start_seed(
    graph: &Graph,
    ctx: &Ctx,
    start: &CNode,
    scope_len: usize,
) -> Vec<u32> {
    let start_check = !start.props.is_empty() || start.where_.is_some();
    let mut sb = Binding(vec![None; scope_len.max(1)]);
    let mut endpoint: Vec<u32> = Vec::new();
    {
        let mut keep = |vi: u32| -> bool {
            if !matches_label(graph, ctx, vi, start.label.as_ref()) {
                return true;
            }
            if start_check {
                if let Some(s) = start.var_slot {
                    sb.set(s, Val::Node(vi));
                }
                if !satisfies(
                    graph,
                    ctx,
                    &Val::Node(vi),
                    &start.props,
                    start.where_.as_ref(),
                    &sb,
                ) {
                    return true;
                }
            }
            endpoint.push(vi);
            true
        };
        // An indexed inline `{k: lit}` / `{k: $param}` pins the start to a handful
        // of candidates — seek them rather than walking the whole label bucket.
        // Without this a traversal from an indexed anchor costs O(label bucket)
        // instead of O(degree): `(s:Employee {id:$x})-[:T]->(t)` scanned every
        // Employee to reach one vertex. The per-vertex label + props re-check above
        // still runs, so the seek only ever *narrows* the same candidate set and
        // seed order is unchanged for the unindexed path.
        match node_index_seed(graph, ctx, start, None) {
            Some(cands) => {
                for vi in cands {
                    keep(vi);
                }
            }
            None => {
                for_each_seed(graph, ctx, start.label.as_ref(), &mut keep);
            }
        }
    }
    endpoint
}

/// Expand a traversal `path` from the given start-node `endpoint` ids into
/// columnar [`ScanCols`], replicating bound columns as each segment fans out. The
/// row order is fully determined by (`endpoint` order, per-segment `expand` order),
/// so a chunk of `endpoint` yields a contiguous slice of the full result in the
/// same order — the parallel driver builds chunks independently and concatenates.
/// Returns `None` for a self-join (a slot bound twice); caller falls back to scalar.
pub(super) fn expand_scan(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    scope_len: usize,
    mut endpoint: Vec<u32>,
    cap: Option<usize>,
) -> Option<ScanCols> {
    // Bound slots and their element kind, in path order.
    let mut kinds: Vec<(usize, Elem)> = Vec::new();
    if let Some(s) = path.start.var_slot {
        kinds.push((s, Elem::Node));
    }
    for seg in &path.segments {
        if let Some(s) = seg.rel.var_slot {
            kinds.push((s, Elem::Edge));
        }
        if let Some(s) = seg.node.var_slot {
            kinds.push((s, Elem::Node));
        }
    }
    let mut seen = HashSet::new();
    if kinds.iter().any(|(s, _)| !seen.insert(*s)) {
        return None; // a slot bound twice (self-join) — not vectorized
    }

    // Per-bound-slot columns built so far; `endpoint` is the current last-node id
    // per row (tracked even for anonymous nodes, to expand the next segment).
    let mut cols: Vec<Option<Vec<u32>>> = (0..scope_len.max(1)).map(|_| None).collect();
    for &(s, _) in &kinds {
        cols[s] = Some(Vec::new());
    }

    // Which slots are populated so far. A later segment's rel/node slots are in
    // `kinds` (and pre-allocated in `cols`) but their columns stay empty until
    // that segment runs, so the per-row copy loops below must skip them.
    let mut bound = vec![false; scope_len.max(1)];
    if let Some(s) = path.start.var_slot {
        bound[s] = true;
        cols[s] = Some(endpoint.clone()); // start col = the seeded endpoints
    }

    // Expand each segment: every frontier row fans out to its matching neighbors,
    // replicating the already-bound columns and appending this segment's ids.
    let nseg = path.segments.len();
    let mut nb = Binding(vec![None; scope_len.max(1)]);
    for (si, seg) in path.segments.iter().enumerate() {
        let rel = &seg.rel;
        let node = &seg.node;
        let rel_check = !rel.props.is_empty() || rel.where_.is_some();
        let node_check = !node.props.is_empty() || node.where_.is_some();
        let need_bind = rel_check || node_check;
        let is_last = si + 1 == nseg;
        let mut new_cols: Vec<Option<Vec<u32>>> = (0..scope_len.max(1)).map(|_| None).collect();
        for &(s, _) in &kinds {
            new_cols[s] = Some(Vec::new());
        }
        let mut new_endpoint: Vec<u32> = Vec::new();
        'rows: for i in 0..endpoint.len() {
            // Prior slots are constant across this row's neighbors — set them once.
            if need_bind {
                for &(s, knd) in &kinds {
                    if !bound[s] || Some(s) == rel.var_slot || Some(s) == node.var_slot {
                        continue;
                    }
                    if let Some(col) = &cols[s] {
                        let v = match knd {
                            Elem::Node => Val::Node(col[i]),
                            Elem::Edge => Val::Edge(col[i]),
                        };
                        nb.set(s, v);
                    }
                }
            }
            for (eidx, nbr) in expand(graph, ctx, endpoint[i], rel.direction, rel.label.as_ref()) {
                if !matches_label(graph, ctx, nbr, node.label.as_ref()) {
                    continue;
                }
                if need_bind {
                    if let Some(s) = rel.var_slot {
                        nb.set(s, Val::Edge(eidx));
                    }
                    if let Some(s) = node.var_slot {
                        nb.set(s, Val::Node(nbr));
                    }
                    if rel_check
                        && !satisfies(
                            graph,
                            ctx,
                            &Val::Edge(eidx),
                            &rel.props,
                            rel.where_.as_ref(),
                            &nb,
                        )
                    {
                        continue;
                    }
                    if node_check
                        && !satisfies(
                            graph,
                            ctx,
                            &Val::Node(nbr),
                            &node.props,
                            node.where_.as_ref(),
                            &nb,
                        )
                    {
                        continue;
                    }
                }
                for &(s, _) in &kinds {
                    let v = if Some(s) == rel.var_slot {
                        eidx
                    } else if Some(s) == node.var_slot {
                        nbr
                    } else if bound[s] {
                        cols[s].as_ref().unwrap()[i]
                    } else {
                        // Slot bound by a later segment — not present in this row yet.
                        continue;
                    };
                    new_cols[s].as_mut().unwrap().push(v);
                }
                new_endpoint.push(nbr);
                // No WHERE ⇒ every built row survives, so a LIMIT can stop here.
                if is_last && cap.is_some_and(|c| new_endpoint.len() >= c) {
                    break 'rows;
                }
                // Bound the frontier before it takes the host down. The cross-product
                // of partial matches can reach billions of rows on a dense graph, and
                // only the *last* segment's LIMIT prunes early. Checked here inside the
                // build — not after the segment — so a single layer that would jump to
                // a billion rows caps at the ceiling instead of materializing the whole
                // layer first. Faults (surfaced as `E_RESOURCE_EXHAUSTED` at the row
                // boundary) and bails; returning drops `new_cols`/`new_endpoint`, so
                // the memory is released rather than continuing to grow.
                if new_endpoint.len() > INTERMEDIATE_BUDGET {
                    ctx.set_fault(FAULT_INTERMEDIATE);
                    return None;
                }
            }
        }

        // This segment's rel/node columns are now populated for every row.
        if let Some(s) = rel.var_slot {
            bound[s] = true;
        }
        if let Some(s) = node.var_slot {
            bound[s] = true;
        }
        cols = new_cols;
        endpoint = new_endpoint;
    }

    let mut sc = ScanCols::new(scope_len);
    sc.n = endpoint.len();
    for &(s, e) in &kinds {
        sc.slots[s] = Some((e, cols[s].take().unwrap()));
    }
    Some(sc)
}

/// Edge-first build for a single segment `(a)-[r]->(b)` seeded from the edge
/// index: for each candidate edge, validate its type + direction + the inline
/// node/rel constraints, and emit one `(a, r, b)` row. The clause WHERE is still
/// re-applied by the caller, so the edge seed only has to be a superset.
pub(super) fn edge_first_build(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    scope_len: usize,
    edges: &[u32],
) -> Option<ScanCols> {
    let seg = &path.segments[0];
    let (start, rel, node) = (&path.start, &seg.rel, &seg.node);
    // A slot bound twice (self-join) — leave to the scalar path.
    let slots: Vec<usize> = [start.var_slot, rel.var_slot, node.var_slot]
        .into_iter()
        .flatten()
        .collect();
    let mut seen = HashSet::new();
    if slots.iter().any(|s| !seen.insert(*s)) {
        return None;
    }
    let (start_check, rel_check, node_check) = (
        !start.props.is_empty() || start.where_.is_some(),
        !rel.props.is_empty() || rel.where_.is_some(),
        !node.props.is_empty() || node.where_.is_some(),
    );
    let mut a_ids = Vec::new();
    let mut r_ids = Vec::new();
    let mut b_ids = Vec::new();
    let mut bind = Binding(vec![None; scope_len.max(1)]);
    for &e in edges {
        let ei = e as usize;
        if !graph.is_edge_live(e) {
            continue;
        }
        if !rel
            .label
            .as_ref()
            .is_none_or(|lbl| eval_label_edge(ctx, graph.e_type[ei], lbl))
        {
            continue;
        }
        let (src, dst) = (graph.e_src[ei], graph.e_dst[ei]);
        let orients: &[(u32, u32)] = match rel.direction {
            Direction::Out => &[(src, dst)],
            Direction::In => &[(dst, src)],
            // A self-loop's two orientations are identical, so emit it once.
            Direction::Both if src == dst => &[(src, dst)],
            Direction::Both => &[(src, dst), (dst, src)],
        };
        for &(a, bn) in orients {
            if !matches_label(graph, ctx, a, start.label.as_ref())
                || !matches_label(graph, ctx, bn, node.label.as_ref())
            {
                continue;
            }
            if start_check || rel_check || node_check {
                if let Some(s) = start.var_slot {
                    bind.set(s, Val::Node(a));
                }
                if let Some(s) = rel.var_slot {
                    bind.set(s, Val::Edge(e));
                }
                if let Some(s) = node.var_slot {
                    bind.set(s, Val::Node(bn));
                }
                if start_check
                    && !satisfies(
                        graph,
                        ctx,
                        &Val::Node(a),
                        &start.props,
                        start.where_.as_ref(),
                        &bind,
                    )
                {
                    continue;
                }
                if rel_check
                    && !satisfies(
                        graph,
                        ctx,
                        &Val::Edge(e),
                        &rel.props,
                        rel.where_.as_ref(),
                        &bind,
                    )
                {
                    continue;
                }
                if node_check
                    && !satisfies(
                        graph,
                        ctx,
                        &Val::Node(bn),
                        &node.props,
                        node.where_.as_ref(),
                        &bind,
                    )
                {
                    continue;
                }
            }
            a_ids.push(a);
            r_ids.push(e);
            b_ids.push(bn);
        }
    }
    let nrows = r_ids.len();
    let mut sc = ScanCols::new(scope_len);
    sc.n = nrows;
    if let Some(s) = start.var_slot {
        sc.slots[s] = Some((Elem::Node, a_ids));
    }
    if let Some(s) = rel.var_slot {
        sc.slots[s] = Some((Elem::Edge, r_ids));
    }
    if let Some(s) = node.var_slot {
        sc.slots[s] = Some((Elem::Node, b_ids));
    }
    Some(sc)
}

/// Build a new row set holding only rows `idx`, in that order (for ORDER BY: the
/// sorted window — gathers the few output rows instead of projecting all of `sc`).
pub(super) fn gather_rows(sc: &ScanCols, idx: &[usize]) -> ScanCols {
    let mut out = ScanCols::new(sc.slots.len());
    out.n = idx.len();
    for (s, col) in sc.slots.iter().enumerate() {
        if let Some((elem, ids)) = col {
            out.slots[s] = Some((*elem, idx.iter().map(|&i| ids[i]).collect()));
        } else if let Some(vals) = &sc.vals[s] {
            out.vals[s] = Some(idx.iter().map(|&i| vals[i].clone()).collect());
        }
    }
    out
}

/// A contiguous row-range view of a frame as its own (owned) `ScanCols` — used to
/// split a large frame into chunks for parallel column evaluation.
#[cfg(feature = "parallel-query")]
pub(super) fn slice_rows(sc: &ScanCols, lo: usize, hi: usize) -> ScanCols {
    let mut out = ScanCols::new(sc.slots.len());
    out.n = hi - lo;
    for s in 0..sc.slots.len() {
        if let Some((e, ids)) = &sc.slots[s] {
            out.slots[s] = Some((*e, ids[lo..hi].to_vec()));
        } else if let Some(v) = &sc.vals[s] {
            out.vals[s] = Some(v[lo..hi].to_vec());
        }
    }
    out
}

/// Evaluate each projection item as a `Val` column over the whole frame. For a
/// large frame (and the opt-in `parallel-query` feature) the rows are split into
/// chunks evaluated concurrently, then the per-item columns concatenated in order —
/// the expression eval is embarrassingly parallel and `Graph`/`Ctx` are `Sync`.
///
/// Measured (52k rows, 16 threads): ~1.7x on heavy projections (expr-heavy 4.4ms
/// → 2.5ms; single num/str col ~1.7x). It does NOT scale to core count — these
/// loops stream `f64`/`Val` columns and are memory-bandwidth-bound, plus the
/// concat and the caller's RowSet transpose are serial tails. Two consequences:
/// (1) the threshold keeps small queries on the serial path (thread hand-off
/// would dominate); (2) on a server already saturated with concurrent queries,
/// *inter*-query parallelism uses the cores better — this trades a single query's
/// latency for throughput, so it's a win mainly when cores would otherwise idle.
pub(super) fn par_project(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    items: &[CReturnItem],
) -> Vec<Vec<Val>> {
    let serial = || {
        items
            .iter()
            .map(|it| eval_vec(graph, ctx, sc, &it.expr).into_vals())
            .collect()
    };
    #[cfg(feature = "parallel-query")]
    {
        // Threshold: only worth splitting once there's enough per-row work to
        // amortize chunk slicing + thread hand-off.
        const MIN_ROWS: usize = 16_384;
        let threads = rayon::current_num_threads();
        if sc.n >= MIN_ROWS && threads > 1 {
            let nchunks = threads.min(sc.n / 4096).max(1);
            if nchunks > 1 {
                let chunk = sc.n.div_ceil(nchunks);
                let ranges: Vec<(usize, usize)> = (0..nchunks)
                    .map(|c| (c * chunk, ((c + 1) * chunk).min(sc.n)))
                    .filter(|&(lo, hi)| lo < hi)
                    .collect();
                let parts: Vec<Vec<Vec<Val>>> = ranges
                    .par_iter()
                    .map(|&(lo, hi)| {
                        let sub = slice_rows(sc, lo, hi);
                        items
                            .iter()
                            .map(|it| eval_vec(graph, ctx, &sub, &it.expr).into_vals())
                            .collect()
                    })
                    .collect();
                let mut cols: Vec<Vec<Val>> =
                    (0..items.len()).map(|_| Vec::with_capacity(sc.n)).collect();
                for mut part in parts {
                    for (j, c) in part.drain(..).enumerate() {
                        cols[j].extend(c); // moves Vals (no clone), preserves order
                    }
                }
                return cols;
            }
        }
    }
    serial()
}

/// Drop the rows where `keep[i]` is false, compacting every slot column in place.
pub(super) fn compact(sc: &mut ScanCols, keep: &[bool]) {
    for (_, v) in sc.slots.iter_mut().flatten() {
        let mut w = 0;
        for i in 0..v.len() {
            if keep[i] {
                v[w] = v[i];
                w += 1;
            }
        }
        v.truncate(w);
    }
    for v in sc.vals.iter_mut().flatten() {
        let mut w = 0;
        #[allow(
            clippy::needless_range_loop,
            reason = "bound by the column length; `i` indexes the keep mask and is the swap target"
        )]
        for i in 0..v.len() {
            if keep[i] {
                v.swap(w, i);
                w += 1;
            }
        }
        v.truncate(w);
    }
    sc.n = keep.iter().filter(|&&k| k).count();
}

/// Vectorized grouped / global aggregate over an already-matched (and WHERE-
/// filtered) row set. Supports a single direct-`Prop` group key over a typed
/// column (keys hash on raw ids, no string build) and non-distinct `count(*)` /
/// `count`/`sum`/`avg`/`min`/`max` over a column. Returns `None` (→ scalar) for
/// anything else (multi-key, expr keys, DISTINCT, collect, non-numeric min/max).
/// Raw key bits per row for a group-key item that is a direct `Prop` over a
/// typed column (string-id / f64-bits / bool). `None` per row = absent (its own
/// NULL group); `None` overall = the key isn't a typed direct property, so the
/// caller must fall back to the scalar path.
pub(super) fn key_raw_col(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    item: &CReturnItem,
) -> Option<Vec<Option<u64>>> {
    raw_bits_of(graph, ctx, sc, &item.expr)
}

/// Per-row raw key bits for a direct `Prop` over a typed column: the interned
/// string **id** (`Str`), the `f64` bits (`Num`), or the bool (`Bool`) — `None`
/// per row where the value is absent, `None` overall if the expr isn't a direct
/// typed-column property (Mixed / absent). Both the vectorized group-by key
/// ([`key_raw_col`]) and `count(DISTINCT …)` fold on this — dedup on an integer id
/// with no string materialization/hashing.
pub(super) fn raw_bits_of(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    expr: &CExpr,
) -> Option<Vec<Option<u64>>> {
    // Grouping / DISTINCT by element *identity* (`WITH p, …`, `count(DISTINCT p)`):
    // the vertex/edge id is already a dense integer key — no property lookup, never
    // absent. (A single key column is one element type, so a node id and an edge id
    // never share a refinement pass; matches the scalar `@v{id}` / `@e{id}` key.)
    if let CExpr::Var(slot) = expr {
        let (_elem, ids) = sc.slot(*slot)?;
        return Some(ids.iter().map(|&id| Some(id as u64)).collect());
    }
    let CExpr::Prop { var_slot, key_ref } = expr else {
        return None;
    };
    let (col, ids) = prop_col(graph, ctx, sc, *var_slot, *key_ref)?;
    let bits = |i: usize, present: &crate::graph::BitSet, raw: u64| {
        present.get(ids[i] as usize).then_some(raw)
    };
    match col {
        Some(Column::Str { data, present }) => Some(
            (0..sc.n)
                .map(|i| bits(i, present, data[ids[i] as usize] as u64))
                .collect(),
        ),
        Some(Column::Num { data, present }) => Some(
            (0..sc.n)
                .map(|i| bits(i, present, data[ids[i] as usize].to_bits()))
                .collect(),
        ),
        Some(Column::Bool { data, present }) => Some(
            (0..sc.n)
                .map(|i| bits(i, present, data[ids[i] as usize] as u64))
                .collect(),
        ),
        _ => None, // Mixed / absent column — can't cheaply raw-key it
    }
}

/// Assign a dense group id per row by grouping on `key_items`. Multi-key grouping
/// is done by *refinement*: start with one group, then split each current group
/// by each key column's value in turn. Because the final pass numbers groups in
/// row order by first appearance of (prev-group, last-key) — which uniquely
/// identifies the full key tuple — this reproduces the scalar engine's first-seen
/// group order exactly. Each key must be a direct `Prop` over a typed column
/// (raw-id hashing, no string build); otherwise `None` → scalar fallback.
/// Returns `(gid per row, representative row per group, group count)`.
pub(super) fn group_ids(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    key_items: &[&CReturnItem],
) -> Option<(Vec<usize>, Vec<usize>, usize)> {
    let n = sc.n;
    let mut gid_of_row = vec![0usize; n];
    let mut ngroups = 1; // global group (overwritten once any key column refines)
    for &item in key_items {
        let col = key_raw_col(graph, ctx, sc, item)?;
        // FxHash, not SipHash: this refines groups by hashing a short key per row —
        // exactly the case the module's FxHash was introduced for. Group numbering
        // is by first-appearance in row order, so iteration order is irrelevant →
        // byte-identity-safe. (See the FxHasher rationale in `eval.rs`.)
        let mut map: FxHashMap<(usize, Option<u64>), usize> = FxHashMap::default();
        let mut next = 0usize;
        let mut refined = vec![0usize; n];
        for i in 0..n {
            let g = *map.entry((gid_of_row[i], col[i])).or_insert_with(|| {
                let g = next;
                next += 1;
                g
            });
            refined[i] = g;
        }
        gid_of_row = refined;
        ngroups = next;
    }
    // Representative row per group (first occurrence).
    let mut rep_row = vec![usize::MAX; ngroups];
    #[allow(
        clippy::needless_range_loop,
        reason = "bound by row count `n`; `i` indexes gid_of_row and is stored as the representative row"
    )]
    for i in 0..n {
        let g = gid_of_row[i];
        if rep_row[g] == usize::MAX {
            rep_row[g] = i;
        }
    }
    Some((gid_of_row, rep_row, ngroups))
}

/// Resolve `arg` to a direct typed **numeric** column read: `(data, present, ids)`
/// where `data[ids[i]]` is row `i`'s value. `None` unless `arg` is a bare `Prop`
/// over a `Column::Num` — the shape the fused global aggregate can read straight
/// out of storage with no per-row `Val` boxing or gathered copy.
pub(super) fn num_col_of<'a>(
    graph: &'a Graph,
    ctx: &Ctx,
    sc: &'a ScanCols,
    arg: &CExpr,
) -> Option<(&'a [f64], &'a crate::graph::BitSet, &'a [u32])> {
    let CExpr::Prop { var_slot, key_ref } = arg else {
        return None;
    };
    let (col, ids) = prop_col(graph, ctx, sc, *var_slot, *key_ref)?;
    match col {
        Some(Column::Num { data, present }) => Some((data.as_slice(), present, ids)),
        _ => None,
    }
}

/// If `ids` is exactly `[base, base+1, …, base+len-1]`, return `base`. Lets a fused
/// scan reduce over a contiguous `&data[base..]` slice (fully autovectorizable)
/// instead of gathering `data[ids[i]]` one index at a time. O(len) but branch-free
/// bar the compare — cheap next to the gather+alloc it replaces.
pub(super) fn contiguous_base(ids: &[u32]) -> Option<usize> {
    let base = *ids.first()? as usize;
    ids.iter()
        .enumerate()
        .all(|(k, &id)| id as usize == base + k)
        .then_some(base)
}

/// One fused global (un-grouped) aggregate, computed by reducing straight over the
/// stored column — no `eval_vec` gather, no materialized `f64`/validity vectors,
/// no second pass. Handles `count(*)`, and `count`/`sum`/`avg`/`min`/`max` over a
/// direct numeric property. Returns `None` (→ caller's general path) for anything
/// else (non-numeric min/max, DISTINCT, collect, expression args, Mixed columns).
///
/// Three tiers by column density: a fully-present column over a contiguous id run
/// reduces over a flat slice (SIMD); a fully-present column at arbitrary ids gathers
/// with no presence branch; otherwise the presence bit is probed per element.
pub(super) fn fused_global_agg(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    spec: &CAgg,
) -> Option<Val> {
    // count(DISTINCT prop): dedup the interned **ids** (string id / f64 bits / bool)
    // in an integer set — no `Val` build, no string hashing (the scalar path keys a
    // `HashSet<String>` of formatted values per row).
    if spec.distinct {
        if spec.func != AggFn::Count {
            return None; // sum/avg/min/max DISTINCT stay scalar
        }
        let bits = raw_bits_of(graph, ctx, sc, spec.arg.as_ref()?)?;
        // FxHash: a per-row insert to count distinct values (membership only → order
        // irrelevant → byte-identity-safe).
        let seen: FxHashSet<u64> = bits.into_iter().flatten().collect();
        return Some(Val::Num(seen.len() as f64));
    }
    if spec.func == AggFn::Count && spec.star {
        // count(*) over one global group is just the live row count.
        return Some(Val::Num(sc.n as f64));
    }
    if matches!(
        spec.func,
        AggFn::CollectList | AggFn::PercentileCont | AggFn::PercentileDisc
    ) {
        return None; // collect-then-compute aggregates aren't vectorized
    }
    // Temporal aggregates over a typed temporal column: min/max via the total
    // order and sum via DURATION addition compute here; avg (and sum over a
    // non-DURATION kind) faults loud. The numeric `num_col_of` fold below can't
    // read a temporal column (it would silently NaN → null).
    if let Some(v) = temporal_agg(graph, ctx, sc, spec) {
        return Some(v);
    }
    let (data, present, ids) = num_col_of(graph, ctx, sc, spec.arg.as_ref()?)?;
    let dense = present.all_set(data.len());

    // count(prop): number of present values (all rows when the column is dense).
    if spec.func == AggFn::Count {
        let c = if dense {
            ids.len()
        } else {
            ids.iter().filter(|&&i| present.get(i as usize)).count()
        };
        return Some(Val::Num(c as f64));
    }

    // sum/avg/min/max fold. `sum`+`n` cover sum and avg; min/max track an extremum.
    let mut sum = 0.0f64;
    let mut n = 0usize;
    let mut ext: Option<f64> = None;
    let is_min = spec.func == AggFn::Min;
    let mut fold = |x: f64| {
        sum += x;
        n += 1;
        ext = Some(match ext {
            Some(e) => {
                if is_min {
                    e.min(x)
                } else {
                    e.max(x)
                }
            }
            None => x,
        });
    };

    match (dense, contiguous_base(ids)) {
        // Tier 1: dense + contiguous — reduce a flat slice (autovectorizes).
        (true, Some(base)) => {
            for &x in &data[base..base + ids.len()] {
                fold(x);
            }
        }
        // Tier 2: dense, scattered ids — gather, but no presence branch.
        (true, None) => {
            for &i in ids {
                fold(data[i as usize]);
            }
        }
        // Tier 3: sparse — probe presence per element.
        (false, _) => {
            for &i in ids {
                let i = i as usize;
                if present.get(i) {
                    fold(data[i]);
                }
            }
        }
    }

    Some(match spec.func {
        AggFn::Sum => Val::Num(sum),
        AggFn::Avg => {
            if n == 0 {
                Val::Null
            } else {
                Val::Num(sum / n as f64)
            }
        }
        AggFn::Min | AggFn::Max => ext.map_or(Val::Null, Val::Num),
        _ => return None,
    })
}

/// Fold every aggregate in `proj` into a per-group column (`Vec<Val>` of length
/// `ngroups`, one per `proj.aggs` spec), given the row→group map from
/// [`group_ids`]. The tight loops (`count`/`sum`/`avg`/`min`/`max`) index the
/// group id directly — no per-row `eval` or string key. Returns `None` (→ caller
/// falls back to the scalar accumulator) for a shape not vectorized here: grouped
/// DISTINCT, non-numeric `min`/`max`, `collect`/percentile. A single global group
/// (`ngroups == 1`) folds straight over storage via [`fused_global_agg`]. Shared
/// by the terminal [`vectorized_aggregate`] and the pipeline [`with_frame`].
pub(super) fn fold_group_agg_cols(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    proj: &CProjection,
    gid_of_row: &[usize],
    ngroups: usize,
) -> Option<Vec<Vec<Val>>> {
    let n = sc.n;
    let mut agg_cols: Vec<Vec<Val>> = Vec::with_capacity(proj.aggs.len());
    for spec in &proj.aggs {
        // Global (single-group) aggregates fold straight over the stored column —
        // no gather, no materialized f64/validity vectors, no second pass. This also
        // covers `count(DISTINCT prop)` (dedup on interned ids), so it's tried
        // before the distinct bail below.
        if ngroups == 1 {
            if let Some(v) = fused_global_agg(graph, ctx, sc, spec) {
                agg_cols.push(vec![v]);
                continue;
            }
        }
        if spec.distinct {
            return None; // grouped distinct / non-count distinct → scalar
        }
        let col: Vec<Val> = if spec.func == AggFn::Count && spec.star {
            let mut cnt = vec![0u64; ngroups];
            for &g in gid_of_row {
                cnt[g] += 1;
            }
            cnt.into_iter().map(|c| Val::Num(c as f64)).collect()
        } else {
            let arg = spec.arg.as_ref()?;
            let av = eval_vec(graph, ctx, sc, arg);
            // min/max compare by value; only correct here for numeric columns.
            if matches!(spec.func, AggFn::Min | AggFn::Max) && !matches!(av, VVec::Num { .. }) {
                return None;
            }
            // Temporal (gathered → `Gen`) sum/avg can't go through the numeric fold
            // (it would NaN → null); bail to the scalar accumulator, which sums
            // DURATIONs and faults on avg / non-summable kinds.
            if matches!(spec.func, AggFn::Sum | AggFn::Avg) && matches!(av, VVec::Gen(_)) {
                return None;
            }
            let (d, valid) = av.into_num();
            match spec.func {
                AggFn::Count => {
                    let mut c = vec![0u64; ngroups];
                    for i in 0..n {
                        if valid[i] {
                            c[gid_of_row[i]] += 1;
                        }
                    }
                    c.into_iter().map(|x| Val::Num(x as f64)).collect()
                }
                AggFn::Sum => {
                    let mut s = vec![0f64; ngroups];
                    for i in 0..n {
                        if valid[i] {
                            s[gid_of_row[i]] += d[i];
                        }
                    }
                    s.into_iter().map(Val::Num).collect()
                }
                AggFn::Avg => {
                    let mut s = vec![0f64; ngroups];
                    let mut c = vec![0u64; ngroups];
                    for i in 0..n {
                        if valid[i] {
                            let g = gid_of_row[i];
                            s[g] += d[i];
                            c[g] += 1;
                        }
                    }
                    (0..ngroups)
                        .map(|g| {
                            if c[g] == 0 {
                                Val::Null
                            } else {
                                Val::Num(s[g] / c[g] as f64)
                            }
                        })
                        .collect()
                }
                AggFn::Min | AggFn::Max => {
                    let is_min = spec.func == AggFn::Min;
                    let mut m: Vec<Option<f64>> = vec![None; ngroups];
                    for i in 0..n {
                        if valid[i] {
                            let g = gid_of_row[i];
                            m[g] = Some(match m[g] {
                                Some(x) => {
                                    if is_min {
                                        x.min(d[i])
                                    } else {
                                        x.max(d[i])
                                    }
                                }
                                None => d[i],
                            });
                        }
                    }
                    m.into_iter()
                        .map(|o| o.map_or(Val::Null, Val::Num))
                        .collect()
                }
                _ => return None, // CollectList etc.
            }
        };
        agg_cols.push(col);
    }
    Some(agg_cols)
}

pub(super) fn vectorized_aggregate(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    proj: &CProjection,
) -> Option<Vec<Vec<Val>>> {
    let key_items = proj.group_keys();
    let (gid_of_row, rep_row, ngroups) = group_ids(graph, ctx, sc, &key_items)?;
    let agg_cols = fold_group_agg_cols(graph, ctx, sc, proj, &gid_of_row, ngroups)?;

    // Bind group `g`'s representative row's element slots into `b` (for a computed
    // group key / an aggregate expr that references a key). `usize::MAX` = the empty
    // global group (no input rows) — leave unbound; only pure aggregates read it.
    let bind_rep = |b: &mut Binding, g: usize| {
        if let Some(&ri) = rep_row.get(g).filter(|&&ri| ri != usize::MAX) {
            bind_frame_row(b, sc, ri);
        }
    };

    let mut b = Binding(vec![None; sc.slots.len()]);
    if proj.order_by.is_empty() {
        // No ORDER BY: emit groups in first-seen order, applying SKIP/LIMIT directly.
        let start = proj.skip_val(ctx).min(ngroups);
        let end = proj
            .limit_val(ctx)
            .map(|l| (start + l).min(ngroups))
            .unwrap_or(ngroups);
        let mut out: Vec<Vec<Val>> = vec![Vec::with_capacity(end - start); proj.items.len()];
        for g in start..end {
            bind_rep(&mut b, g);
            let agg_values: Vec<Val> = agg_cols.iter().map(|c| c[g].clone()).collect();
            let env = Env {
                graph,
                ctx,
                binding: &b,
                group: None,
                agg_values: Some(&agg_values),
            };
            for (item_idx, item) in proj.items.iter().enumerate() {
                out[item_idx].push(eval(&env, &item.expr));
            }
        }
        return Some(out);
    }

    // ORDER BY: materialize every group's projected row + its sort keys, then sort
    // (input-keyed, exactly like the scalar `sort_keys`: keys evaluated over the
    // projected output + `order_overlay` input slots + the folded aggregates), then
    // SKIP/LIMIT, then transpose the selected rows to columns.
    let mut rows: Vec<Vec<Val>> = Vec::with_capacity(ngroups);
    let mut keys: Vec<Vec<Val>> = Vec::with_capacity(ngroups);
    for g in 0..ngroups {
        bind_rep(&mut b, g);
        let agg_values: Vec<Val> = agg_cols.iter().map(|c| c[g].clone()).collect();
        let env = Env {
            graph,
            ctx,
            binding: &b,
            group: None,
            agg_values: Some(&agg_values),
        };
        let row: Vec<Val> = proj
            .items
            .iter()
            .map(|item| eval(&env, &item.expr))
            .collect();
        // Sort-key env: projected output at slots 0..out_len, then the order_overlay
        // input slots appended — matches `ProjAccum::sort_keys` exactly.
        let mut sort_binding = Binding(row.iter().map(|v| Some(v.clone())).collect());
        for &islot in &proj.order_overlay {
            sort_binding.0.push(b.get(islot).cloned());
        }
        let senv = Env {
            graph,
            ctx,
            binding: &sort_binding,
            group: None,
            agg_values: Some(&agg_values),
        };
        keys.push(proj.order_by.iter().map(|s| eval(&senv, &s.expr)).collect());
        rows.push(row);
    }

    // Total order: ORDER BY keys, then the group's first-seen index as the final
    // tiebreak — so ties resolve to first-seen group order (a stable sort's result),
    // which lets the partial sort below stay unstable yet deterministic. Mirrors the
    // non-aggregate ORDER BY branch and the scalar path's group order.
    let cmp = |&i: &usize, &j: &usize| -> Ordering {
        for (k, s) in proj.order_by.iter().enumerate() {
            let o = compare_sort(&keys[i][k], &keys[j][k], s.descending, s.nulls_first);
            if o != Ordering::Equal {
                return o;
            }
        }
        i.cmp(&j)
    };
    let start = proj.skip_val(ctx).min(ngroups);
    let end = proj
        .limit_val(ctx)
        .map(|l| (start + l).min(ngroups))
        .unwrap_or(ngroups);
    let mut idx: Vec<usize> = (0..ngroups).collect();
    // Partial sort for a LIMIT: quickselect the smallest `end`, then sort only those.
    if end >= 1 && end < idx.len() {
        idx.select_nth_unstable_by(end - 1, cmp);
        idx.truncate(end);
    }
    idx.sort_by(cmp);
    let sel = &idx[start.min(idx.len())..end.min(idx.len())];
    let mut out: Vec<Vec<Val>> = vec![Vec::with_capacity(sel.len()); proj.items.len()];
    for &gi in sel {
        for (c, v) in rows[gi].iter().enumerate() {
            out[c].push(v.clone());
        }
    }
    Some(out)
}

/// Try the vectorized path for a single fresh `MATCH` of one fixed-length path,
/// producing the projection's output **as column-major `Val` columns** (each the
/// final output rows, in order, after WHERE / aggregate / DISTINCT / ORDER BY /
/// SKIP+LIMIT). The caller turns these into a terminal `RowSet` (flattening
/// elements to ids) or into carried `Binding`s for a `WITH` (preserving element
/// handles). Returns `None` (→ scalar driver) unless the shape qualifies: one
/// fresh `MATCH` of a buildable (non-var-length, no self-join) path, no `RETURN *`.
pub(super) fn vectorized_cols(
    graph: &Graph,
    ctx: &Ctx,
    incoming: &[Binding],
    matches: &[&CClause],
    proj: &CProjection,
) -> Option<Vec<Vec<Val>>> {
    let sc = vectorized_frame(graph, ctx, incoming, matches, proj)?;
    project_frame_cols(graph, ctx, &sc, proj)
}

/// Build (and WHERE-filter) the columnar frame for a single fresh `MATCH … RETURN`
/// — the shared front half of the vectorized terminal paths ([`vectorized_cols`]
/// and [`vectorized_rowset`]). Returns `None` (→ scalar driver) unless the shape
/// qualifies: one fresh `MATCH` of a buildable (non-var-length, no self-join)
/// path, no `RETURN *`.
pub(super) fn vectorized_frame(
    graph: &Graph,
    ctx: &Ctx,
    incoming: &[Binding],
    matches: &[&CClause],
    proj: &CProjection,
) -> Option<ScanCols> {
    if incoming.len() != 1 || incoming[0].0.iter().any(|c| c.is_some()) {
        return None; // a prior WITH/INSERT already produced bindings
    }
    if matches.len() != 1 || proj.star {
        return None;
    }
    // ORDER BY: an aggregate sorts its group rows internally ([`vectorized_aggregate`],
    // which resolves output aliases + aggregates), so it's allowed. A non-aggregate
    // sort only vectorizes when the keys read input vars (not output aliases);
    // DISTINCT + ORDER BY stays scalar.
    let has_order = !proj.order_by.is_empty();
    if has_order && (proj.distinct || (!proj.aggregating && proj.order_needs_output)) {
        return None;
    }
    let CClause::Match {
        optional: false,
        patterns,
        where_,
        scope_len,
        ..
    } = matches[0]
    else {
        return None;
    };
    if patterns.len() != 1 {
        return None;
    }
    let path = &patterns[0];

    // A bound path variable needs the scalar driver — only it builds the Path
    // value (`all_walk`/`shortest_walk`); the vectorized frame yields columns.
    // (A selector / non-default mode already routes here via the run_part guard.)
    if path.path_var_slot.is_some() {
        return None;
    }

    // A pure aggregate over a traversal with no WHERE stays scalar: the scalar
    // engine stream-folds the join without materializing it, and there's no
    // per-row expression to vectorize. With a WHERE, the batched build + masked
    // count can pay for itself.
    if !path.segments.is_empty() && proj.aggregating && where_.is_none() {
        return None;
    }

    // A multi-segment pattern with a LIMIT and a plain projection is answered far
    // better by the scalar depth-first driver, so defer to it. This path is
    // breadth-first: it materializes each segment's full frontier and the LIMIT
    // only prunes the *last* one, so a dense multi-hop chain builds the entire
    // cross-product of partial matches — millions of rows to return a handful, and
    // on a large graph an OOM. DFS filters during traversal and stops the instant
    // the LIMIT fills, at every level, matching the TS engine's streaming
    // semantics. Aggregation / DISTINCT / ORDER BY genuinely need every row, so
    // they stay here; a limitless multi-hop is enumerate-all and the intermediate
    // budget in `expand_scan` bounds it.
    if path.segments.len() >= 2
        && !proj.aggregating
        && !proj.distinct
        && !has_order
        && proj.limit_val(ctx).is_some()
    {
        return None;
    }

    // With no clause WHERE (and no aggregation/DISTINCT), a LIMIT lets us stop the
    // scan early — preserving the scalar path's streaming advantage for small
    // LIMITs. (DISTINCT/aggregation need every row before producing output.)
    let cap = (where_.is_none() && !proj.aggregating && !proj.distinct && !has_order)
        .then(|| proj.limit_val(ctx).map(|l| proj.skip_val(ctx) + l))
        .flatten();
    // Seed an isolated-node scan from a property index when an indexed eq/range
    // hint applies (cap can't early-stop a seeded scan, so drop it then).
    // An index hint (vertex or edge) makes the scan a seek, so the LIMIT cap
    // can't early-stop it — drop the cap when a hint applies.
    let cap = if scan_is_hinted(graph, ctx, path, where_.as_ref()) {
        None
    } else {
        cap
    };
    let mut sc = build_scan(graph, ctx, path, *scope_len, cap, where_.as_ref())?;

    // Clause WHERE → keep mask (vectorized), compacting the row set.
    if let Some(w) = where_ {
        let keep: Vec<bool> = eval_vec(graph, ctx, &sc, w)
            .into_truth()
            .iter()
            .map(|t| *t == Some(true))
            .collect();
        compact(&mut sc, &keep);
    }
    Some(sc)
}

/// Terminal `MATCH … RETURN` straight to a [`RowSet`], skipping the intermediate
/// `Vec<Val>` columns: each item is evaluated as a `VVec`, then rows are
/// transposed reading `Value`s directly out of the typed buffers — a numeric
/// column goes `f64 → Value::Num` with no `Val` boxing pass, halving the
/// materialization for a numeric projection. Only the **plain** (non-aggregating,
/// non-DISTINCT, non-ORDER-BY) shape qualifies; the others reorder/dedup and need
/// the materialized-column path. `None` ⇒ caller falls back to `vectorized_cols`.
pub(super) fn vectorized_rowset(
    graph: &Graph,
    ctx: &Ctx,
    incoming: &[Binding],
    matches: &[&CClause],
    proj: &CProjection,
) -> Option<RowSet> {
    if proj.aggregating || proj.distinct || !proj.order_by.is_empty() {
        return None;
    }
    let sc = vectorized_frame(graph, ctx, incoming, matches, proj)?;
    let vvs: Vec<VVec> = proj
        .items
        .iter()
        .map(|it| eval_vec(graph, ctx, &sc, &it.expr))
        .collect();
    let start = proj.skip_val(ctx).min(sc.n);
    let end = proj
        .limit_val(ctx)
        .map(|l| (start + l).min(sc.n))
        .unwrap_or(sc.n);
    let mut rs = RowSet::new(proj.out_names.clone());
    for i in start..end {
        rs.push_row(vvs.iter().map(|vv| vv.value_at(i, graph)));
    }
    Some(rs)
}

/// Transpose every row of an already-built (and WHERE-filtered) frame `sc` into a
/// [`RowSet`] via the plain projection — no SKIP/LIMIT (the parallel driver applies
/// those globally after concatenating chunk fragments). The `Val`-boxing-free
/// analogue of [`vectorized_rowset`]'s tail, factored out for the parallel path.
#[cfg(feature = "parallel-query")]
pub(super) fn project_scan_rows(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    proj: &CProjection,
) -> RowSet {
    let vvs: Vec<VVec> = proj
        .items
        .iter()
        .map(|it| eval_vec(graph, ctx, sc, &it.expr))
        .collect();
    let mut rs = RowSet::new(proj.out_names.clone());
    rs.data.reserve(sc.n * proj.items.len().max(1));
    for i in 0..sc.n {
        rs.push_row(vvs.iter().map(|vv| vv.value_at(i, graph)));
    }
    rs
}

/// Project an already-built (and WHERE-filtered) frame `sc` to column-major output
/// — aggregate / ORDER BY / DISTINCT / plain projection + SKIP/LIMIT. Shared by
/// the single-scan entry ([`vectorized_cols`]) and a pipeline's terminal RETURN
/// (where `sc` may carry computed value columns from upstream `WITH`s).
pub(super) fn project_frame_cols(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    proj: &CProjection,
) -> Option<Vec<Vec<Val>>> {
    let has_order = !proj.order_by.is_empty();
    // Aggregating + ORDER BY is handled inside `vectorized_aggregate` (it sorts the
    // group rows, resolving output aliases + aggregates); DISTINCT + ORDER BY and a
    // non-aggregate sort over output aliases stay scalar.
    if has_order && (proj.distinct || (!proj.aggregating && proj.order_needs_output)) {
        return None;
    }
    if proj.aggregating {
        // HAVING filters groups post-fold; that path lives in the scalar
        // `ProjAccum::finish`, so bail to it rather than duplicate it here.
        if proj.having.is_some() {
            return None;
        }
        return vectorized_aggregate(graph, ctx, sc, proj);
    }

    // ORDER BY (input-keyed): evaluate the sort keys as columns, sort row indices,
    // then project only the SKIP/LIMIT window — so a small LIMIT never materializes
    // the full (e.g. string) output columns, just the keys.
    if has_order {
        // A sort-scope view of `sc`: alias each overlay input column at its
        // sort-scope slot (out_len + j), so the sort exprs resolve directly.
        let mut sort_sc = ScanCols::new(proj.out_len + proj.order_overlay.len());
        sort_sc.n = sc.n;
        for (j, &islot) in proj.order_overlay.iter().enumerate() {
            if let Some((elem, ids)) = &sc.slots[islot] {
                sort_sc.slots[proj.out_len + j] = Some((*elem, ids.clone()));
            } else if let Some(vals) = &sc.vals[islot] {
                sort_sc.vals[proj.out_len + j] = Some(vals.clone());
            }
        }
        // Fast path: a single temporal ORDER BY key sorts packed Copy temporals via
        // `cmp_total`, skipping the `Val` keycol + dispatch. Falls through to the
        // generic `Vec<Val>` sort for multi-key / non-temporal / mixed keys (that
        // path is left exactly as-is).
        let single = (proj.order_by.len() == 1).then(|| &proj.order_by[0]);
        // Densest: a single instant key sorts a flat i128 array (cache-friendly,
        // like a numeric sort) — the top-k fast path.
        let dense_key: Option<DenseSortCol> = single.and_then(|s| {
            dense_sort_key(graph, ctx, &sort_sc, &s.expr)
                .map(|(k, v)| (k, v, s.descending, s.nulls_first))
        });
        // Duration (no dense key): compare Copy temporals via `cmp_total`.
        let temporal_key: Option<TypedSortCol> = (dense_key.is_none())
            .then(|| {
                single.and_then(|s| {
                    temporal_sort_key(graph, ctx, &sort_sc, &s.expr)
                        .map(|k| (k, s.descending, s.nulls_first))
                })
            })
            .flatten();
        // Only the generic path needs the `Vec<Val>` keycols.
        let keycols: Vec<Vec<Val>> = if dense_key.is_some() || temporal_key.is_some() {
            Vec::new()
        } else {
            proj.order_by
                .iter()
                .map(|s| eval_vec(graph, ctx, &sort_sc, &s.expr).into_vals())
                .collect()
        };
        // Total-order comparator: the ORDER BY keys, then the original row index as
        // a final tiebreak. The index tiebreak makes ties resolve to scan order —
        // identical to the previous *stable* full sort — while allowing an unstable
        // partial sort below (which needs a strict weak order to be deterministic).
        let cmp = |&i: &usize, &j: &usize| -> Ordering {
            if let Some((key, valid, descending, nulls_first)) = &dense_key {
                let o = dense_compare_sort(
                    key[i],
                    valid[i],
                    key[j],
                    valid[j],
                    *descending,
                    *nulls_first,
                );
                return if o != Ordering::Equal { o } else { i.cmp(&j) };
            }
            if let Some((key, descending, nulls_first)) = &temporal_key {
                let o = temporal_compare_sort(&key[i], &key[j], *descending, *nulls_first);
                return if o != Ordering::Equal { o } else { i.cmp(&j) };
            }
            for (k, s) in proj.order_by.iter().enumerate() {
                let o = compare_sort(&keycols[k][i], &keycols[k][j], s.descending, s.nulls_first);
                if o != Ordering::Equal {
                    return o;
                }
            }
            i.cmp(&j)
        };
        let start = proj.skip_val(ctx).min(sc.n);
        let end = proj
            .limit_val(ctx)
            .map(|l| (start + l).min(sc.n))
            .unwrap_or(sc.n);
        let mut idx: Vec<usize> = (0..sc.n).collect();
        // Partial sort for a LIMIT: partition the top `end` rows out in O(n), then
        // fully sort just that window — instead of an O(n log n) sort of every row
        // to keep only a small prefix. No LIMIT ⇒ a full sort (all rows returned).
        if end >= 1 && end < idx.len() {
            idx.select_nth_unstable_by(end - 1, cmp);
            idx.truncate(end);
        }
        idx.sort_by(cmp);
        let sub = gather_rows(sc, &idx[start..end.min(idx.len())]);
        return Some(
            proj.items
                .iter()
                .map(|item| eval_vec(graph, ctx, &sub, &item.expr).into_vals())
                .collect(),
        );
    }

    // DISTINCT fast path: when every output item is a direct typed-Prop column,
    // DISTINCT ≡ group-by-all-columns with no aggregates — reuse the raw-id
    // grouping and emit one representative row per group (first-seen order, no
    // per-row string key). Falls through to the generic dedup otherwise.
    if proj.distinct {
        let all_items: Vec<&CReturnItem> = proj.items.iter().collect();
        if let Some((_, rep_row, ngroups)) = group_ids(graph, ctx, sc, &all_items) {
            let start = proj.skip_val(ctx).min(ngroups);
            let end = proj
                .limit_val(ctx)
                .map(|l| (start + l).min(ngroups))
                .unwrap_or(ngroups);
            let mut out: Vec<Vec<Val>> = vec![Vec::with_capacity(end - start); proj.items.len()];
            let mut b = Binding(vec![None; sc.slots.len()]);
            for &ri in &rep_row[start..end] {
                bind_frame_row(&mut b, sc, ri);
                let env = Env::new(graph, ctx, &b);
                for (item_idx, item) in proj.items.iter().enumerate() {
                    out[item_idx].push(eval(&env, &item.expr));
                }
            }
            return Some(out);
        }
    }

    // Non-aggregating projection: evaluate each item as a column (parallel over
    // row-chunks for a large frame).
    let mut cols: Vec<Vec<Val>> = par_project(graph, ctx, sc, &proj.items);
    if proj.distinct {
        // Generic DISTINCT (expression / non-typed items): keep the first
        // occurrence of each row in scan order, dedup on a composite cell key.
        // FxHash: membership only; the kept order comes from the scan, not the set.
        let mut seen: FxHashSet<String> = FxHashSet::default();
        let skip = proj.skip_val(ctx);
        let mut seen_count = 0usize;
        let mut kept: Vec<usize> = Vec::new();
        for i in 0..sc.n {
            let mut key = String::new();
            for c in &cols {
                val_key(&c[i], &mut key);
                key.push('\u{1}');
            }
            if !seen.insert(key) {
                continue;
            }
            if seen_count >= skip {
                if proj.limit_val(ctx).is_some_and(|l| kept.len() >= l) {
                    break;
                }
                kept.push(i);
            }
            seen_count += 1;
        }
        Some(
            cols.iter()
                .map(|c| kept.iter().map(|&i| c[i].clone()).collect())
                .collect(),
        )
    } else {
        // Window each column to the SKIP/LIMIT row range (no ORDER BY ⇒ scan order).
        let start = proj.skip_val(ctx).min(sc.n);
        let end = proj
            .limit_val(ctx)
            .map(|l| (start + l).min(sc.n))
            .unwrap_or(sc.n);
        for c in &mut cols {
            c.truncate(end);
            c.drain(0..start);
        }
        Some(cols)
    }
}

/// Project a frame through a non-aggregating `WITH` into a new frame: bare element
/// variables are carried forward as fast element columns (so downstream prop reads
/// and filters stay vectorized), and every other item becomes a computed value
/// column. Returns `None` for shapes a mid-pipeline `WITH` shouldn't carry
/// (aggregate / DISTINCT / ORDER BY / SKIP / LIMIT / `*`) — those end the pipeline
/// or fall back to scalar.
pub(super) fn with_frame(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    proj: &CProjection,
) -> Option<ScanCols> {
    if proj.distinct
        || !proj.order_by.is_empty()
        || proj.skip.is_some()
        || proj.limit.is_some()
        || proj.star
    {
        return None;
    }
    if proj.aggregating {
        return with_frame_aggregate(graph, ctx, sc, proj);
    }
    let mut out = ScanCols::new(proj.out_len);
    out.n = sc.n;
    for (i, item) in proj.items.iter().enumerate() {
        if let CExpr::Var(slot) = &item.expr {
            if let Some((elem, ids)) = sc.slot(*slot) {
                out.slots[i] = Some((elem, ids.to_vec())); // carry element column forward
                continue;
            }
            if let Some(vals) = sc.val_slot(*slot) {
                out.vals[i] = Some(vals.to_vec()); // carry a prior computed column
                continue;
            }
        }
        out.vals[i] = Some(eval_vec(graph, ctx, sc, &item.expr).into_vals());
    }
    Some(out)
}

/// A grouped/global aggregating `WITH` as a columnar frame → frame transform: one
/// output row per group (first-seen order), replacing the scalar per-row
/// accumulator. Groups by raw ids ([`group_ids`], now including element identity),
/// folds each aggregate columnar ([`fold_group_agg_cols`]), then materializes each
/// output item at the group's representative row. Bare element group keys carry
/// their element column forward (so downstream `p.name` / `RETURN p` still resolve
/// the handle); computed keys and aggregate expressions eval per group (few groups)
/// against the rep binding + folded values. `None` (→ scalar `run_linear`) when the
/// keys/aggregates aren't raw-vectorizable — identical fallback surface to the
/// terminal [`vectorized_aggregate`].
pub(super) fn with_frame_aggregate(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    proj: &CProjection,
) -> Option<ScanCols> {
    let key_items = proj.group_keys();
    let (gid_of_row, rep_row, ngroups) = group_ids(graph, ctx, sc, &key_items)?;
    let agg_cols = fold_group_agg_cols(graph, ctx, sc, proj, &gid_of_row, ngroups)?;

    let mut out = ScanCols::new(proj.out_len);
    out.n = ngroups;
    // Items that can't be read straight from a carried column — computed group
    // keys and aggregate expressions — are evaluated per group below.
    let mut need_eval: Vec<usize> = Vec::new();
    for (i, item) in proj.items.iter().enumerate() {
        if !item.is_agg {
            if let CExpr::Var(slot) = &item.expr {
                // A bare element group key: carry its column, gathered at each
                // group's representative row. (Bare element keys ⇒ `ngroups` is the
                // real group count, so every `rep_row` entry is a live row.)
                if let Some((elem, ids)) = sc.slot(*slot) {
                    out.slots[i] = Some((elem, rep_row.iter().map(|&ri| ids[ri]).collect()));
                    continue;
                }
                // A bare carried value column (a key from an upstream WITH): gather.
                if let Some(vals) = sc.val_slot(*slot) {
                    out.vals[i] = Some(rep_row.iter().map(|&ri| vals[ri].clone()).collect());
                    continue;
                }
            }
        }
        need_eval.push(i);
    }

    if !need_eval.is_empty() {
        let mut cols: Vec<Vec<Val>> = need_eval
            .iter()
            .map(|_| Vec::with_capacity(ngroups))
            .collect();
        let mut b = Binding(vec![None; sc.slots.len()]);
        for g in 0..ngroups {
            // Rebind the representative row's element slots so a computed group key
            // (`p.age`) or an aggregate expr that references a key resolves.
            // `usize::MAX` = the empty global group (no rows); leave unbound.
            if let Some(&ri) = rep_row.get(g).filter(|&&ri| ri != usize::MAX) {
                bind_frame_row(&mut b, sc, ri);
            }
            let agg_values: Vec<Val> = agg_cols.iter().map(|c| c[g].clone()).collect();
            let env = Env {
                graph,
                ctx,
                binding: &b,
                group: None,
                agg_values: Some(&agg_values),
            };
            for (k, &i) in need_eval.iter().enumerate() {
                cols[k].push(eval(&env, &proj.items[i].expr));
            }
        }
        for (k, &i) in need_eval.iter().enumerate() {
            out.vals[i] = Some(std::mem::take(&mut cols[k]));
        }
    }

    Some(out)
}

/// Expand a frame by a `MATCH` whose start node is an already-bound element column
/// (e.g. `… WITH a MATCH (a)-[:KNOWS]->(b) …`): for each frame row, walk the
/// path's segments from that row's start vertex, fanning out to matching
/// neighbors and replicating the frame's other columns. Returns `None` for a
/// fresh/unbound start (cartesian), var-length, or a segment slot already bound.
/// Which per-segment checks an edge must pass, bundled to keep `seg_edge_accepts`
/// under the arg limit: whether the relationship/target need to be *bound* into the
/// candidate binding, and whether the relationship / node property predicates need
/// re-checking.
pub(super) struct SegChecks {
    pub need_bind: bool,
    pub rel_check: bool,
    pub node_check: bool,
}

/// Does neighbor edge `(eidx → nbr)` pass segment `seg`'s node label and — when
/// `need_bind` — its inline rel/node property + WHERE predicates? When `need_bind`, the
/// rel/node slots are set into `nb` as a side effect (so a caller that keeps the row
/// sees them bound). `rel_check`/`node_check` are the precomputed "has a predicate"
/// flags. The shared per-edge accept test of `expand_frame` and `expand_frame_optional`;
/// the divergent output (column scatter vs null-fillable push) stays in each caller.
/// The `rel_check`/`node_check` flags are precomputed once per segment by the caller to
/// keep this per-edge test free of repeated `props`/`where_` inspection.
#[inline]
pub(super) fn seg_edge_accepts(
    graph: &Graph,
    ctx: &Ctx,
    seg: &CSegment,
    eidx: u32,
    nbr: u32,
    checks: SegChecks,
    nb: &mut Binding,
) -> bool {
    let SegChecks {
        need_bind,
        rel_check,
        node_check,
    } = checks;
    let (rel, node) = (&seg.rel, &seg.node);
    if !matches_label(graph, ctx, nbr, node.label.as_ref()) {
        return false;
    }
    if need_bind {
        if let Some(s) = rel.var_slot {
            nb.set(s, Val::Edge(eidx));
        }
        if let Some(s) = node.var_slot {
            nb.set(s, Val::Node(nbr));
        }
        if rel_check
            && !satisfies(
                graph,
                ctx,
                &Val::Edge(eidx),
                &rel.props,
                rel.where_.as_ref(),
                nb,
            )
        {
            return false;
        }
        if node_check
            && !satisfies(
                graph,
                ctx,
                &Val::Node(nbr),
                &node.props,
                node.where_.as_ref(),
                nb,
            )
        {
            return false;
        }
    }
    true
}

pub(super) fn expand_frame(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    path: &CPath,
    scope_len: usize,
) -> Option<ScanCols> {
    let start = &path.start;
    let start_slot = start.var_slot?;
    let start_ids: Vec<u32> = match sc.slot(start_slot) {
        Some((Elem::Node, ids)) => ids.to_vec(), // start must be a bound node column
        _ => return None,
    };
    if path.segments.iter().any(|s| s.rel.quantifier.is_some()) {
        return None;
    }
    // Segment-introduced slots must be fresh (not already bound) — no self-join.
    let mut seen = HashSet::new();
    for seg in &path.segments {
        for s in [seg.rel.var_slot, seg.node.var_slot].into_iter().flatten() {
            if !seen.insert(s) || sc.slot(s).is_some() || sc.val_slot(s).is_some() {
                return None;
            }
        }
    }
    let width = scope_len.max(sc.slots.len());

    // cur = the frame widened to `width`; endpoint = each row's start vertex.
    let mut cur = ScanCols::new(width);
    cur.n = sc.n;
    for s in 0..sc.slots.len() {
        if let Some((e, ids)) = &sc.slots[s] {
            cur.slots[s] = Some((*e, ids.clone()));
        } else if let Some(v) = &sc.vals[s] {
            cur.vals[s] = Some(v.clone());
        }
    }
    let mut endpoint = start_ids;

    // Sets a binding from `cur` at row `i` (for inline WHERE/props referencing
    // frame variables during constraint checks).
    let bind_row = |b: &mut Binding, cur: &ScanCols, i: usize| {
        for s in 0..cur.slots.len() {
            if let Some((e, ids)) = &cur.slots[s] {
                b.set(
                    s,
                    match e {
                        Elem::Node => Val::Node(ids[i]),
                        Elem::Edge => Val::Edge(ids[i]),
                    },
                );
            } else if let Some(v) = &cur.vals[s] {
                b.set(s, v[i].clone());
            }
        }
    };

    // The restated start node may add label/props/WHERE — filter rows by them.
    if start.label.is_some() || !start.props.is_empty() || start.where_.is_some() {
        let mut b = Binding(vec![None; width]);
        let mut keep = vec![false; cur.n];
        for i in 0..cur.n {
            bind_row(&mut b, &cur, i);
            keep[i] = matches_label(graph, ctx, endpoint[i], start.label.as_ref())
                && satisfies(
                    graph,
                    ctx,
                    &Val::Node(endpoint[i]),
                    &start.props,
                    start.where_.as_ref(),
                    &b,
                );
        }
        endpoint = endpoint
            .iter()
            .zip(&keep)
            .filter_map(|(&v, &k)| k.then_some(v))
            .collect();
        compact(&mut cur, &keep);
    }

    let mut nb = Binding(vec![None; width]);
    for seg in &path.segments {
        let rel = &seg.rel;
        let node = &seg.node;
        let rel_check = !rel.props.is_empty() || rel.where_.is_some();
        let node_check = !node.props.is_empty() || node.where_.is_some();
        let need_bind = rel_check || node_check;
        // Pre-init the next frame's columns: new rel/node slots + carried columns.
        let mut nxt = ScanCols::new(width);
        for s in 0..width {
            if Some(s) == rel.var_slot {
                nxt.slots[s] = Some((Elem::Edge, Vec::new()));
            } else if Some(s) == node.var_slot {
                nxt.slots[s] = Some((Elem::Node, Vec::new()));
            } else if let Some((e, _)) = &cur.slots[s] {
                nxt.slots[s] = Some((*e, Vec::new()));
            } else if cur.vals[s].is_some() {
                nxt.vals[s] = Some(Vec::new());
            }
        }
        let mut nxt_end: Vec<u32> = Vec::new();
        for i in 0..cur.n {
            if need_bind {
                bind_row(&mut nb, &cur, i);
            }
            for (eidx, nbr) in expand(graph, ctx, endpoint[i], rel.direction, rel.label.as_ref()) {
                if !seg_edge_accepts(
                    graph,
                    ctx,
                    seg,
                    eidx,
                    nbr,
                    SegChecks {
                        need_bind,
                        rel_check,
                        node_check,
                    },
                    &mut nb,
                ) {
                    continue;
                }
                for s in 0..width {
                    if Some(s) == rel.var_slot {
                        nxt.slots[s].as_mut().unwrap().1.push(eidx);
                    } else if Some(s) == node.var_slot {
                        nxt.slots[s].as_mut().unwrap().1.push(nbr);
                    } else if let Some((_, ids)) = &cur.slots[s] {
                        nxt.slots[s].as_mut().unwrap().1.push(ids[i]);
                    } else if let Some(v) = &cur.vals[s] {
                        nxt.vals[s].as_mut().unwrap().push(v[i].clone());
                    }
                }
                nxt_end.push(nbr);
            }
        }
        nxt.n = nxt_end.len();
        cur = nxt;
        endpoint = nxt_end;
    }
    Some(cur)
}

/// OPTIONAL single-segment expansion as a columnar frame transform: like
/// [`expand_frame`], but **every outer row survives** — one output row per match,
/// or a single NULL-filled row when an outer row has no match (ISO `OPTIONAL
/// MATCH`). The segment's new rel/node slots become **value** columns: they must
/// hold `Val::Null` for the unmatched rows, which an element `slots` column (a bare
/// `Vec<u32>`) can't. Downstream reads them via `val_slot` (bare var) /`scalar_col`
/// (property access), and `count(f)` counts the non-null rows (`num_of` marks a
/// node valid, null invalid) — matching the scalar accumulator exactly.
///
/// Scoped to a **single fixed-length segment from a bare re-stated start** (no
/// start label/props/WHERE — those would need null-fill semantics, not a compacting
/// filter; no var-length; no self-join). Anything else → `None` (scalar fallback).
/// A clause-level WHERE on the OPTIONAL clause is refused by the caller (it, too,
/// would have to null-fill rather than drop).
pub(super) fn expand_frame_optional(
    graph: &Graph,
    ctx: &Ctx,
    sc: &ScanCols,
    path: &CPath,
    scope_len: usize,
) -> Option<ScanCols> {
    if path.segments.len() != 1 {
        return None;
    }
    let start = &path.start;
    if start.label.is_some() || !start.props.is_empty() || start.where_.is_some() {
        return None;
    }
    let start_slot = start.var_slot?;
    let start_ids: Vec<u32> = match sc.slot(start_slot) {
        Some((Elem::Node, ids)) => ids.to_vec(),
        _ => return None,
    };
    let seg = &path.segments[0];
    if seg.rel.quantifier.is_some() {
        return None;
    }
    let rel = &seg.rel;
    let node = &seg.node;
    // The new rel/node slots must be fresh (no self-join back onto a bound column).
    let mut seen = HashSet::new();
    for s in [rel.var_slot, node.var_slot].into_iter().flatten() {
        if !seen.insert(s) || sc.slot(s).is_some() || sc.val_slot(s).is_some() {
            return None;
        }
    }
    let width = scope_len.max(sc.slots.len());
    let rel_check = !rel.props.is_empty() || rel.where_.is_some();
    let node_check = !node.props.is_empty() || node.where_.is_some();
    let need_bind = rel_check || node_check;

    // Carried columns keep their kind (element/value); the segment's rel/node slots
    // are nullable value columns.
    let mut out = ScanCols::new(width);
    for s in 0..width {
        if Some(s) == rel.var_slot || Some(s) == node.var_slot {
            out.vals[s] = Some(Vec::new());
        } else if s < sc.slots.len() {
            if let Some((e, _)) = &sc.slots[s] {
                out.slots[s] = Some((*e, Vec::new()));
            } else if sc.vals[s].is_some() {
                out.vals[s] = Some(Vec::new());
            }
        }
    }

    // Append one output row: carried columns read from outer row `i`; the segment's
    // rel/node value columns take `rv`/`nv` (both `Val::Null` for the no-match fill).
    let push = |out: &mut ScanCols, i: usize, rv: &Val, nv: &Val| {
        for s in 0..width {
            if Some(s) == rel.var_slot {
                out.vals[s].as_mut().unwrap().push(rv.clone());
            } else if Some(s) == node.var_slot {
                out.vals[s].as_mut().unwrap().push(nv.clone());
            } else if s < sc.slots.len() {
                if let Some((_, ids)) = &sc.slots[s] {
                    out.slots[s].as_mut().unwrap().1.push(ids[i]);
                } else if let Some(v) = &sc.vals[s] {
                    out.vals[s].as_mut().unwrap().push(v[i].clone());
                }
            }
        }
    };

    let mut nb = Binding(vec![None; width]);
    let mut nrows = 0usize;
    for i in 0..sc.n {
        if need_bind {
            for s in 0..sc.slots.len() {
                if let Some((e, ids)) = &sc.slots[s] {
                    nb.set(
                        s,
                        match e {
                            Elem::Node => Val::Node(ids[i]),
                            Elem::Edge => Val::Edge(ids[i]),
                        },
                    );
                } else if let Some(v) = &sc.vals[s] {
                    nb.set(s, v[i].clone());
                }
            }
        }
        let mut matched = false;
        for (eidx, nbr) in expand(graph, ctx, start_ids[i], rel.direction, rel.label.as_ref()) {
            if !seg_edge_accepts(
                graph,
                ctx,
                seg,
                eidx,
                nbr,
                SegChecks {
                    need_bind,
                    rel_check,
                    node_check,
                },
                &mut nb,
            ) {
                continue;
            }
            push(&mut out, i, &Val::Edge(eidx), &Val::Node(nbr));
            nrows += 1;
            matched = true;
        }
        if !matched {
            push(&mut out, i, &Val::Null, &Val::Null);
            nrows += 1;
        }
    }
    out.n = nrows;
    Some(out)
}
