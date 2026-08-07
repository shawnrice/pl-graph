//! Index-seeded scanning: pick the most selective index seed for a pattern
//! (prop_index_hint / node_index_seed / try_orient_node_seed), build the scan,
//! expand it edge-first, and run vectorized grouping/aggregation over the result
//! (fused_global_agg / vectorized_aggregate / fold_group_agg_cols). Extracted from
//! the evaluator (`super`); shares its context/helpers via `use super::*`.
use super::seek_lower;
use super::*;
use crate::seek::ElementSeek;

/// Scatter the frame's bound element slots into binding `b` at row `ri`: each
/// bound slot becomes the Node/Edge id sitting at that row. Value/scalar slots are
/// filled separately by the caller. Shared by every place that materializes a
/// representative row from the columnar frame (grouped aggregation, sort binding).
#[inline]
fn bind_frame_row(b: &mut Binding, sc: &ScanCols, ri: usize) {
    for (slot, col) in sc.cols.iter().enumerate() {
        if let Some(Col::Elems { ids, is_edge }) = col {
            b.set(
                slot,
                if *is_edge {
                    Val::Edge(ids[ri])
                } else {
                    Val::Node(ids[ri])
                },
            );
        }
    }
}

/// [`bind_frame_row`] plus the frame's COMPUTED columns.
///
/// The two are separate because they do different things, and until the frame's
/// two column vectors became one that was easy to miss: the element loop was
/// written out twice, once here and once there, and only one copy also bound a
/// computed column.
///
/// The difference is NOT observable today, and the reason is worth writing down
/// because it is not obvious: the only thing that sets a computed column is a
/// bound path variable, and `vectorized_frame` declines outright when there is
/// one (only the scalar driver builds a `Path`). So `bind_frame_row` never meets
/// one. Checked rather than assumed — `ORDER BY path_length(p)` and `GROUP BY p`
/// over `p = (x)-[:R]->{1,2}(y)` both answer correctly, on the scalar path.
///
/// It becomes reachable the moment the frame learns to carry a path, so the two
/// callers keep the behavior they had rather than being quietly unified into
/// whichever one someone reads first.
#[inline]
pub(super) fn bind_frame_row_with_vals(b: &mut Binding, sc: &ScanCols, ri: usize) {
    for (slot, col) in sc.cols.iter().enumerate() {
        match col {
            Some(Col::Elems { ids, is_edge }) => b.set(
                slot,
                if *is_edge {
                    Val::Edge(ids[ri])
                } else {
                    Val::Node(ids[ri])
                },
            ),
            Some(Col::Gen(vals)) => b.set(slot, vals[ri].clone()),
            _ => {}
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

/// `a OP b` as `b OP' a` — the same predicate with the operands exchanged.
/// Equality and inequality are symmetric; the orderings invert.
pub(super) fn flip_compare(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Ge => CompareOp::Le,
        other => other,
    }
}

/// An as-of / overlap seek from an edge RI-tree INTERVAL index.
///
/// Deliberately not an `ElementSeek`: it is a different structure with its own
/// selectivity rule (compare the axes by `*_len` and stab the most selective
/// ONE — never materialize and intersect both, which measured worse than a
/// scan). Folding it into the shared layer would mean teaching that layer
/// about temporal axes, so it stays a pre-check ahead of the ordinary path.
///
/// This was briefly lost when the seeds moved to the shared layer, and no test
/// caught it: the index is a pure performance feature, so the fallback returns
/// identical rows. It showed up only as 1.00x vs scan in `bench_temporal_index`,
/// where it should be ~27x.
pub(super) fn interval_index_seed(
    graph: &Graph,
    ctx: &Ctx,
    e: &CExpr,
    want_slot: Option<usize>,
) -> Option<Vec<u32>> {
    let CExpr::And(items) = e else {
        return None;
    };
    let slot_ok = |s: usize| want_slot.is_none_or(|w| w == s);

    // Contender B: an as-of over an edge RI-tree interval index (`lo_key <= $v
    // AND hi_key > $v`, same var, same probe) seeds from `stab($v)` — the small
    // "active at v" set directly — instead of the BTreeMap seeks A would pick.
    // Same recognizer style, different seek; falls through to A when no interval
    // index covers the pair. The stab is inclusive of `hi == v`, a superset the
    // final `WHERE hi > v` then verifies.
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
        let (Some((slo, qlo)), Some((shi, qhi))) = (probe(lo_key, true), probe(hi_key, false))
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
    None
}

/// Candidate vertices for a single-node scan: an indexed inline `{key: lit}`
/// equality, or a WHERE comparison on the node. `None` ⇒ full scan.
pub(super) fn node_index_seed(
    graph: &Graph,
    ctx: &Ctx,
    node: &CNode,
    where_: Option<&CExpr>,
) -> Option<Vec<u32>> {
    // Inline `{k: v}` and a clause `WHERE u.k = v` now lower to the SAME
    // structure, so they cannot take different paths. Previously the inline form
    // returned on the FIRST indexed constraint and the WHERE form went through a
    // separate recogniser; the two only agreed by coincidence.
    seeded(
        graph,
        ctx,
        where_,
        &seek_lower::inline_of(node),
        node.var_slot,
        false,
    )
}

/// Whether this element WOULD seed from an index, without building the seed.
///
/// The boolean-only callers (orientation, the LIMIT cap) used to call
/// `node_index_seed(..).is_some()`, which performs the whole seek and discards
/// it. On a traversal that meant three real seeks per execution for one result.
pub(super) fn node_seekable(
    graph: &Graph,
    ctx: &Ctx,
    node: &CNode,
    where_: Option<&CExpr>,
) -> bool {
    seek_lower::element_seek(
        where_,
        &seek_lower::inline_of(node),
        graph,
        ctx,
        node.var_slot,
        false,
    )
    .can_seek(graph, &seek_lower::GqlBindings(ctx.params))
}

/// Candidate ids for a lowered seek, or `None` to scan.
pub(super) fn resolve_seek(graph: &Graph, ctx: &Ctx, seek: ElementSeek) -> Option<Vec<u32>> {
    if seek.is_empty() {
        return None;
    }

    seek.resolve(graph, &seek_lower::GqlBindings(ctx.params))
}

/// A seek for one element: the edge interval index first (it answers a shape the
/// ordinary property indexes answer badly), then the shared access path.
fn seeded(
    graph: &Graph,
    ctx: &Ctx,
    where_: Option<&CExpr>,
    inline: &[(&str, &CExpr)],
    var_slot: Option<usize>,
    edge: bool,
) -> Option<Vec<u32>> {
    if edge {
        if let Some(ids) = where_.and_then(|w| interval_index_seed(graph, ctx, w, var_slot)) {
            return Some(ids);
        }
    }

    resolve_seek(
        graph,
        ctx,
        seek_lower::element_seek(where_, inline, graph, ctx, var_slot, edge),
    )
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
        // NOTE: this union is NOT deduped. An edge is bucketed under every label
        // it carries, so `[:X|Y]` over an edge labelled [X, Y] would take it from
        // both sides — the double-count that had to be fixed in
        // `try_count_edges` and `ElementSeek::label_seed`. It is left alone here
        // because no query reaches this branch: instrumenting it across the
        // disjunction tests and the `edge_first_build` shapes produced zero hits,
        // so a dedup would be code no test can exercise. If a change routes a
        // multi-type seed through here, dedup it — and `a_label_disjunction_over_
        // multi_label_elements_counts_each_once` is where to assert it.
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
    seeded(
        graph,
        ctx,
        where_,
        &seek_lower::inline_of_rel(rel),
        rel.var_slot,
        true,
    )
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
    let start_seek = node_seekable(graph, ctx, &path.start, where_);
    let end_seek = node_seekable(graph, ctx, end_node, where_);

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
        node_seekable(graph, ctx, &path.start, where_)
    } else if path.segments.len() == 1 {
        edge_prop_seed(graph, ctx, &path.segments[0].rel, where_).is_some()
    } else {
        false
    }
}

/// Collect a driver-produced frame into columns.
///
/// `visit_pattern` owns the walk for path selectors and multi-segment path
/// variables — choosing among many walks is exactly what it is for. What it
/// hands back is one binding per row, so those rows become a `ScanCols` and the
/// rest of the pipeline runs columnar rather than staying scalar to the end.
fn driven_scan(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    scope_len: usize,
    where_: Option<&CExpr>,
    cap: Option<usize>,
) -> Option<ScanCols> {
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

    // A self-join names a slot twice; one column each.
    let mut seen: HashSet<usize> = HashSet::new();

    kinds.retain(|(s, _)| seen.insert(*s));

    let mut cols: Vec<Vec<u32>> = kinds.iter().map(|_| Vec::new()).collect();
    let mut paths: Vec<Val> = Vec::new();
    let path_slot = path.path_var_slot;
    let mut b = Binding::with_len(scope_len.max(1));
    let mut n = 0usize;
    let mut ragged = false;

    crate::gql::eval::pathfind::visit_pattern(graph, ctx, path, where_, &mut b, &mut |bind| {
        for (k, &(slot, kind)) in kinds.iter().enumerate() {
            // A slot the driver left unbound cannot be columnized.
            let id = match bind.get(slot) {
                Some(Val::Node(i)) if kind == Elem::Node => *i,
                Some(Val::Edge(e)) if kind == Elem::Edge => *e,
                _ => {
                    ragged = true;
                    return false;
                }
            };

            cols[k].push(id);
        }

        if let Some(s) = path_slot {
            match bind.get(s) {
                Some(v) => paths.push(v.clone()),
                None => {
                    ragged = true;
                    return false;
                }
            }
        }

        n += 1;
        cap.is_none_or(|c| n < c)
    });

    // Any row the driver could not fully bind leaves the columns ragged. The
    // scalar path handles those correctly, so hand back to it rather than
    // emitting a short column beside full ones.
    if ragged || cols.iter().any(|c| c.len() != n) || (path_slot.is_some() && paths.len() != n) {
        return None;
    }

    let mut sc = ScanCols::new(scope_len);

    sc.n = n;

    for (k, &(slot, kind)) in kinds.iter().enumerate() {
        sc.set_elems(slot, kind, std::mem::take(&mut cols[k]));
    }

    if let Some(slot) = path_slot {
        sc.set_vals(slot, paths);
    }

    Some(sc)
}

/// Which slots the rest of the statement will READ, when that is known.
///
/// `None` means "assume everything". Only ever used to SKIP building a column,
/// so an under-populated set would silently drop one — see
/// [`crate::gql::plan::Program::read_slots`].
pub(super) type Needed<'a> = Option<&'a [usize]>;

/// Everything about how an expansion STARTS, beside its endpoints: the LIMIT cap,
/// the slots downstream will read, and any columns an earlier clause already
/// bound. Grouped because they travel together through [`expand_scan`].
pub(super) struct SeedFrom<'a> {
    pub cap: Option<usize>,
    pub needed: Needed<'a>,
    /// Columns from the frame this expansion continues (a mid-pipeline `MATCH`
    /// after a `WITH`), or `None` for a fresh scan.
    pub carry: Option<&'a ScanCols>,
}

/// A frame carrying only the LAST slot, walked rather than joined.
///
/// The middle ground between the columnar frame and a row-at-a-time stream. A
/// frame materializes a column per bound slot and pairs them per row; a walk
/// materializes the frontier and nothing else. When the only slot anything reads
/// is the one the walk lands on, the pairing is work whose result is discarded —
/// `MATCH (a:V)-[:R]->(b) WHERE a.n > 900 RETURN b.n` built 14,850 `(a, b)` pairs
/// to project `b`.
///
/// The rows are IDENTICAL to the frame's projection onto that slot: `expand`
/// emits one endpoint per traversed edge, so multiplicity is preserved and no
/// downstream terminal can tell the difference. Every one of them — rows,
/// aggregates, GROUP BY, ORDER BY — works unchanged, which is the reason to
/// produce a frame here rather than answer any single shape.
///
/// This is what Gremlin's `try_values` has always done: expand to ids, gather the
/// column. Two tight loops over `u32` and `f64`, no per-row binding at all — the
/// distinction that matters is not stream-vs-columnar but how many columns the
/// pairing has to carry.
///
/// Declines unless the walk can carry the whole meaning: every segment bare (a
/// filter needs a row to reject), and the clause WHERE about the start alone,
/// which is applied HERE because the caller's re-application reads a start column
/// this frame does not have.
pub(super) fn streamed_frame(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    where_: Option<&CExpr>,
    scope_len: usize,
    proj: &CProjection,
) -> Option<ScanCols> {
    if path.segments.is_empty()
        || path.path_var_slot.is_some()
        || path.selector != PathSelector::Walk
        || path.mode != PathMode::Trail
    {
        return None;
    }

    let last = path.segments.last()?.node.var_slot?;

    // What the PROJECTION reads — deliberately not the caller's `needed`, which
    // folds in the clause WHERE. The WHERE is applied here, over the seed, so its
    // slots are not slots this frame has to carry; counting them made every
    // filtered shape decline and the walk never fire at all.
    let mut reads = Vec::new();

    for it in proj.items.iter().chain(&proj.group_by) {
        if !it.prog.read_slots(&mut reads) {
            return None;
        }
    }

    for k in &proj.order_by {
        if !crate::gql::plan::compile_program(&k.expr).read_slots(&mut reads) {
            return None;
        }
    }

    for a in &proj.aggs {
        if let Some(arg) = a.arg.as_ref() {
            if !crate::gql::plan::compile_program(arg).read_slots(&mut reads) {
                return None;
            }
        }
    }

    if proj.star {
        reads.extend(proj.star_cols.iter().copied());
    }

    // The overlay is every input slot, so a sort key can name one the projection
    // dropped. Only a sorting query reads it.
    if !proj.order_by.is_empty() {
        reads.extend(proj.order_overlay.iter().copied());
    }

    if reads.iter().any(|s| *s != last) {
        return None;
    }

    // A multi-segment chain with a LIMIT stays with the scalar depth-first
    // driver, which stops the instant the limit fills AT EVERY LEVEL. The walk is
    // breadth-first: `expand` emits one row per traversed edge, so after k hops it
    // holds every k-path — the cross-product the equivalent bail below was written
    // for, "millions of rows to return a handful, and on a large graph an OOM".
    //
    // Measured on 50k vertices at degree 6, `RETURN … LIMIT 5`:
    //
    //   two hops    0.001ms guarded, 5.462ms walked
    //   three hops  0.001ms guarded, 73.700ms walked
    //
    // — the growth the cross-product argument predicts, and the reason this is
    // not left to a correctness test: the ROWS are identical either way, so
    // removing this guard fails nothing.
    //
    // Aggregation, DISTINCT and ORDER BY genuinely need every row, so they still
    // walk.
    if path.segments.len() >= 2
        && !proj.aggregating
        && !proj.distinct
        && proj.order_by.is_empty()
        && proj.limit.is_some()
    {
        return None;
    }

    let start_slot = path.start.var_slot;

    // The clause WHERE is applied here, over the seeded start, so it may not say
    // anything about a slot the walk does not bind.
    if where_.is_some_and(|w| crate::gql::plan::refs_slot(w, &|s| Some(s) != start_slot)) {
        return None;
    }

    let mut hops: Vec<(crate::seek::Dir, Option<Vec<u32>>)> =
        Vec::with_capacity(path.segments.len());

    for seg in &path.segments {
        if seg.unit.is_some()
            || seg.rel.quantifier.is_some()
            || seg.rel.var_slot.is_some_and(|s| reads.contains(&s))
            || !seg.rel.props.is_empty()
            || seg.rel.where_.is_some()
            || !seg.node.props.is_empty()
            || seg.node.where_.is_some()
            || seg.node.label.is_some()
        {
            return None;
        }

        // `None` is ANY type here and `Some(vec![])` is a name that resolved to
        // NOTHING — the walk reads it that way and yields nothing, where a bare
        // `Vec` would have read empty as "any" and returned every edge.
        let etypes = match &seg.rel.label {
            None => None,
            Some(l) => Some(seek_lower::lower_labels(l, ctx, true)?),
        };

        hops.push((
            match seg.rel.direction {
                Direction::Out => crate::seek::Dir::Out,
                Direction::In => crate::seek::Dir::In,
                Direction::Both => crate::seek::Dir::Both,
            },
            etypes,
        ));
    }

    let mut ids = seek_lower::scan_node(graph, ctx, &path.start, where_, scope_len, None);

    // The seed carries the start; filter it before walking, since the caller's
    // pass cannot.
    if let Some(w) = where_ {
        let mut start_frame = ScanCols::new(scope_len);

        start_frame.n = ids.len();
        start_frame.set_elems(start_slot?, Elem::Node, ids);

        let keep: Vec<bool> = eval_vec(graph, ctx, &start_frame, w)
            .into_truth()
            .iter()
            .map(|t| *t == Some(true))
            .collect();

        compact(&mut start_frame, &keep);
        ids = match start_frame.take_slot(start_slot?) {
            Some((_, v)) => v,
            None => return None,
        };
    }

    // The shared streaming walk — the same one `walk_count` folds over and
    // Gremlin's lowered prefix drives.
    let ids = crate::seek::walk_ids(graph, &ids, &hops, ctx.loops);

    let mut out = ScanCols::new(scope_len);

    out.n = ids.len();
    out.set_elems(last, Elem::Node, ids);

    Some(out)
}

pub(super) fn build_scan(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    scope_len: usize,
    cap: Option<usize>,
    where_: Option<&CExpr>,
    needed: Needed<'_>,
) -> Option<ScanCols> {
    // A path selector (`ANY SHORTEST`) or a bound path variable is handled only by
    // the scalar driver — only it builds the Path value.
    // A path SELECTOR (`ANY SHORTEST`) picks which of many walks to keep, which
    // is the scalar driver's job. A bound path VARIABLE is not — the walker can
    // hand back each walk's `(vertices, edges)`, and `ScanCols` already carries
    // `Val` columns beside the id ones, so the Path value rides along.
    // A path SELECTOR (`ANY SHORTEST`, `ALL SHORTEST`, `SHORTEST k`) and a path
    // variable over a multi-segment pattern both need `visit_pattern` — it is the
    // one entry point that knows which of many walks to keep. But what it hands
    // back is still one binding per row, so the walk stays scalar while the frame
    // becomes COLUMNS and everything downstream (projection, grouping,
    // aggregation) runs vectorized over it instead of continuing scalar.
    let needs_driver = path.selector != PathSelector::Walk
        || (path.path_var_slot.is_some()
            && !(path.segments.len() == 1
                && path.segments[0].rel.quantifier.is_some()
                && path.segments[0].unit.is_none()
                && path.segments[0].rel.var_slot.is_none()));

    if needs_driver {
        return driven_scan(graph, ctx, path, scope_len, where_, cap);
    }
    // Fast path: an isolated node is a tight scan. An index hint (inline `{k:v}`
    // eq or a WHERE comparison on the node) seeds just the candidate vertices;
    // otherwise the label bucket / all-live range. Either way the node's label +
    // inline constraints are re-checked.
    if path.segments.is_empty() {
        let node = &path.start;

        // One lowering for both scan shapes — see `seek_lower::scan_node`.
        let ids = seek_lower::scan_node(graph, ctx, node, where_, scope_len, cap);

        let mut sc = ScanCols::new(scope_len);
        sc.n = ids.len();
        if let Some(s) = node.var_slot {
            sc.set_elems(s, Elem::Node, ids);
        }
        return Some(sc);
    }
    // The ABBREVIATED form's edge variable (`-[e]->{n,m}`) still keeps the
    // matcher: it binds `e` per HOP and unbinds it again, so there is no
    // per-repetition value for a column to hold — it must read back as NULL, and
    // the frontier has nowhere to say that. A parenthesized unit is different:
    // every variable it exposes IS a per-repetition list, and `bind_group_vars`
    // already builds exactly that, so those vectorize (see `expand_scan`).
    if path
        .segments
        .iter()
        .any(|seg| seg.rel.quantifier.is_some() && seg.unit.is_none() && seg.rel.var_slot.is_some())
    {
        return None;
    }
    // Cardinality-based orientation: a label-only traversal seeds from its more
    // selective node end and walks its adjacency (O(seeds·degree)) instead of
    // scanning the whole edge-type bucket (O(E)). Same decision as
    // `try_parallel_scan`, so the serial and parallel paths seed identically.
    if let Some(oriented) = try_orient_node_seed(graph, ctx, path, where_) {
        let endpoint = scan_start_seed(graph, ctx, &oriented.start, scope_len, where_);
        return expand_scan(
            graph,
            ctx,
            &oriented,
            scope_len,
            endpoint,
            SeedFrom {
                cap,
                needed,
                carry: None,
            },
        );
    }
    // Edge-first: a single segment with an indexed edge-property hint → seek the
    // matching edges and validate the surrounding (a)-[r]->(b) pattern, instead
    // of expanding every vertex's adjacency.
    // …and never for a quantified segment: this builds ONE `(a)-[r]->(b)` row
    // per seeded edge, so a `{n,m}` walk would silently collapse to a single hop.
    // It has no quantifier guard of its own, which is why relaxing the one above
    // sent var-length here instead of to the expansion below.
    if path.segments.len() == 1 && path.segments[0].rel.quantifier.is_none() {
        // A *selective* edge seed (an indexed edge property) is always worth taking.
        // The `by_etype` fallback is not: it materializes every edge of the type,
        // O(E_type), which loses badly whenever an endpoint is index-seekable —
        // that seeds a handful of vertices and walks their adjacency, O(seeds·deg).
        // `try_orient_node_seed` above deliberately bails on an indexed endpoint so
        // as "not to interfere with a real index seek"; without this guard control
        // fell straight through to here and an indexed anchor *diverted* the plan
        // into the whole-type scan — making the index actively harmful.
        let endpoint_seekable = node_seekable(graph, ctx, &path.start, where_)
            || node_seekable(graph, ctx, &path.segments[0].node, where_);
        let rel = &path.segments[0].rel;
        // Inline edge constraints with NO index behind them are the case the
        // type-bucket fallback serves worst: it materializes every edge of the
        // type, O(E), and the constraints then throw most of them away — while a
        // node-seeded expansion applies the same constraints DURING the walk and
        // never builds the vector. Measured on 50k vertices / 150k edges with no
        // index on `w`, `MATCH ()-[r:R {w: 1}]->() RETURN count(*)`: 2.84ms
        // seeding 150000 edges here, 1.33ms expanding from the nodes.
        //
        // `try_orient_node_seed` above declines this shape (it treats any written
        // edge property as evidence that this branch has a selective seed), so
        // falling through lands on the plain start-seeded expansion at the bottom
        // — which is the one that wins.
        //
        // Relaxing that guard instead was tried and REJECTED: it is provably a
        // no-op for gql_bench (all 202 evaluations take the same branch either
        // way, checked) and still cost 11% on `edge prop filter` and 12% on
        // `with then match expand`, because trimming a term changed the
        // function's inlining — forcing `#[inline(never)]` on the ORIGINAL
        // reproduced the same slowdown, and `#[inline]` on the new one did not
        // recover it. This branch is the narrower place to say it.
        //
        // Even here, `with then match expand` reads 7% slower and the condition
        // below NEVER fires in that benchmark (instrumented: zero firings), so no
        // plan changed. Under the 10% floor, and recorded rather than chased.
        let unselective_inline =
            !rel.props.is_empty() && edge_prop_seed(graph, ctx, rel, where_).is_none();
        let seed = if unselective_inline {
            None
        } else if endpoint_seekable {
            edge_prop_seed(graph, ctx, rel, where_)
        } else {
            edge_index_seed(graph, ctx, rel, where_)
        };
        if let Some(edges) = seed {
            return edge_first_build(graph, ctx, path, scope_len, &edges);
        }
    }
    // Seed the start-node endpoints, then expand the segments into columns.
    let endpoint = scan_start_seed(graph, ctx, &path.start, scope_len, where_);
    expand_scan(
        graph,
        ctx,
        path,
        scope_len,
        endpoint,
        SeedFrom {
            cap,
            needed,
            carry: None,
        },
    )
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
    where_: Option<&CExpr>,
) -> Vec<u32> {
    // An indexed anchor pins the start to a handful of candidates rather than the
    // whole label bucket: `(s:Employee {id:$x})-[:T]->(t)` used to scan every
    // Employee to reach one. A clause WHERE counts as an anchor too, not just an
    // inline `{k: lit}` — that was worth 60x, and only in the traversal case,
    // which is what made it easy to miss.
    seek_lower::scan_node(
        graph,
        ctx,
        start,
        where_.or(start.where_.as_ref()),
        scope_len,
        None,
    )
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
    endpoint: Vec<u32>,
    seed: SeedFrom<'_>,
) -> Option<ScanCols> {
    let SeedFrom { cap, needed, carry } = seed;
    // A quantified unit's variables are GROUP variables — one list per row, held
    // as `Val` columns, not one element per row. They must not also be claimed
    // here as element columns: a segment carrying a unit still reports its edge
    // on `seg.rel.var_slot`, and registering that slot would install an empty id
    // column beside the real list one, which then wins in `scalar_col`.
    let mut group_slots: Vec<usize> = Vec::new();

    for seg in &path.segments {
        if let Some(unit) = seg.unit.as_ref() {
            unit.group_slots(&mut group_slots);
        }
    }

    // Building a group variable's list costs an allocation per row per variable,
    // and a query that binds `((x)-[e]->(y)){1,4}` but only returns the landing
    // node reads none of them. Skip the columns nothing downstream will look at.
    // `kinds` below still excludes every group slot, read or not — those slots are
    // never element columns.
    let built: Vec<usize> = match needed {
        Some(n) => group_slots
            .iter()
            .copied()
            .filter(|s| n.contains(s))
            .collect(),
        None => group_slots.clone(),
    };

    // Bound slots and their element kind, in path order.
    let mut kinds: Vec<(usize, Elem)> = Vec::new();

    if let Some(s) = path.start.var_slot {
        kinds.push((s, Elem::Node));
    }

    for seg in &path.segments {
        if let Some(s) = seg.rel.var_slot.filter(|s| !group_slots.contains(s)) {
            kinds.push((s, Elem::Edge));
        }

        if let Some(s) = seg.node.var_slot.filter(|s| !group_slots.contains(s)) {
            kinds.push((s, Elem::Node));
        }
    }

    // A slot named twice (`(a)-[:R]->(b)-[:S]->(a)`) is a SELF-JOIN: the second
    // occurrence is an equality against what the first bound, which the frontier
    // enforces per row. This used to refuse the pattern and fall back.
    let mut seen: HashSet<usize> = HashSet::new();
    let mut rejoins: Vec<(bool, bool)> = Vec::with_capacity(path.segments.len());

    if let Some(s) = path.start.var_slot {
        seen.insert(s);
    }

    for seg in &path.segments {
        let rel_seen = seg.rel.var_slot.is_some_and(|s| !seen.insert(s));
        let node_seen = seg.node.var_slot.is_some_and(|s| !seen.insert(s));

        rejoins.push((rel_seen, node_seen));
    }

    // The fan-out, the column replication, the LIMIT stop and the
    // intermediate-size ceiling are the SHARED `Frontier` — the same structure
    // Gremlin's lowered prefix carries with a single column. What stays here is
    // the part that is GQL: the per-segment label test and inline/WHERE
    // constraints, supplied as a `RowFilter`.
    let mut frontier = crate::seek::Frontier::seed(endpoint, path.start.var_slot, scope_len.max(1));

    // Columns an earlier clause already bound ride along, so the expansion fans
    // them out with everything else. This is what lets a mid-pipeline `MATCH`
    // (after a `WITH`) use the shared frontier instead of a second expander.
    if let Some(src) = carry {
        for s in 0..scope_len.max(1).min(src.cols.len()) {
            if Some(s) == path.start.var_slot {
                continue;
            }

            if let Some((_, ids)) = src.slot(s) {
                frontier.set_column(s, ids.to_vec());
            } else if let Some(v) = src.val_slot(s) {
                frontier.set_values(s, v.to_vec());
            }
        }
    }
    let nseg = path.segments.len();
    // A bound path variable is a `Val` column, not an id column.
    let mut path_col: Option<(usize, Vec<Val>)> = None;

    for (si, seg) in path.segments.iter().enumerate() {
        let rel = &seg.rel;
        let node = &seg.node;
        // Only the LAST segment may stop early: an earlier one's rows can still
        // be dropped by a later segment, so capping there would lose matches.
        let hop_cap = (si + 1 == nseg).then_some(cap).flatten();
        let etypes = rel
            .label
            .as_ref()
            .and_then(|l| seek_lower::lower_labels(l, ctx, true));
        let hop = crate::seek::Hop {
            dir: match rel.direction {
                Direction::Out => crate::seek::Dir::Out,
                Direction::In => crate::seek::Dir::In,
                Direction::Both => crate::seek::Dir::Both,
            },
            etypes: etypes.as_deref(),
            // Per-language: GQL once, Gremlin twice. See `Ctx::loops`.
            loops: ctx.loops,
            rel_slot: rel.var_slot,
            node_slot: node.var_slot,
            rejoin_rel: rejoins[si].0,
            rejoin_node: rejoins[si].1,
        };
        let mut filter = SegmentFilter {
            graph,
            ctx,
            rel,
            node,
            // An edge-type expression the IR cannot hold stays a per-edge test.
            residual_type: etypes.is_none(),
            rel_check: !rel.props.is_empty() || rel.where_.is_some(),
            node_check: !node.props.is_empty() || node.where_.is_some(),
            rel_consts: rel
                .where_
                .is_none()
                .then(|| crate::gql::eval::const_props(graph, ctx, &rel.props))
                .flatten(),
            node_consts: node
                .where_
                .is_none()
                .then(|| crate::gql::eval::const_props(graph, ctx, &node.props))
                .flatten(),
            rel_eqs: rel
                .where_
                .is_none()
                .then(|| crate::gql::eval::const_props(graph, ctx, &rel.props))
                .flatten()
                .and_then(|c| const_eqs(&graph.edge_props, &graph.strs, ctx, &c, true)),
            node_eqs: node
                .where_
                .is_none()
                .then(|| crate::gql::eval::const_props(graph, ctx, &node.props))
                .flatten()
                .and_then(|c| const_eqs(&graph.props, &graph.strs, ctx, &c, false)),
            kinds: &kinds,
            binding: Binding::with_len(scope_len.max(1)),
        };

        // A quantified hop drives `reachable_each` — the SAME walker the scalar
        // matcher uses — once per frontier row, and vectorizes only the fan-out
        // around it. Its bounds, the path MODE's repeated-element restriction and
        // the zero-length case are subtle enough that a second implementation
        // diverged immediately when tried.
        if let Some(q) = rel.quantifier {
            let path_slot = path.path_var_slot;
            // A parenthesized unit's group variables are per-repetition LISTS.
            // `bind_group_vars` already builds them for the scalar matcher, so
            // this asks for the same thing and reads the result straight out of
            // the binding into a column — one list per row, no second binder.
            let mut seg_groups: Vec<usize> = Vec::new();
            if let Some(unit) = seg.unit.as_ref() {
                unit.group_slots(&mut seg_groups);
                seg_groups.retain(|s| built.contains(s));
            }
            let spec = crate::gql::eval::pathfind::WalkSpec {
                q,
                mode: path.mode,
                // Rebuilding each trail is O(depth), so only ask for it when a
                // path variable or a group variable will actually hold it.
                want_path: path_slot.is_some() || !seg_groups.is_empty(),
            };
            let mut ends: Vec<u32> = Vec::new();
            let mut src: Vec<usize> = Vec::new();
            let mut paths: Vec<Val> = Vec::new();
            let mut groups: Vec<Vec<Val>> = vec![Vec::new(); seg_groups.len()];
            let mut wb = Binding::with_len(scope_len.max(1));
            let node_check = !node.props.is_empty() || node.where_.is_some();

            for i in 0..frontier.rows() {
                let from = frontier.endpoint()[i];
                let mut on_end =
                    |b: &mut Binding,
                     end: u32,
                     v: &[u32],
                     e: &[u32],
                     steps: &[crate::gql::eval::pathfind::StepRec]| {
                        // The landing node's own label and constraints still apply.
                        if !matches_label(graph, ctx, end, node.label.as_ref()) {
                            return true;
                        }

                        if node_check {
                            if let Some(slot) = node.var_slot {
                                b.set(slot, Val::Node(end));
                            }

                            if !satisfies(
                                graph,
                                ctx,
                                &Val::Node(end),
                                &node.props,
                                node.where_.as_ref(),
                                b,
                            ) {
                                return true;
                            }
                        }

                        if let Some(unit) = seg.unit.as_ref() {
                            if !seg_groups.is_empty() {
                                // A FLAT unit gets empty `steps` on purpose — it binds
                                // from the walk directly, which is the hot path and
                                // avoids a per-hop allocation. Only a nested unit
                                // needs the structured records. Same split the scalar
                                // matcher makes; picking one binder for both would
                                // silently bind nothing for every flat unit.
                                let restores = if unit.is_flat() {
                                    crate::gql::eval::pathfind::bind_group_vars_flat(b, unit, v, e)
                                } else {
                                    crate::gql::eval::pathfind::bind_group_vars(b, unit, steps)
                                };

                                for (col, slot) in groups.iter_mut().zip(&seg_groups) {
                                    col.push(b.take(*slot).unwrap_or(Val::Null));
                                }

                                for (slot, prev) in restores {
                                    match prev {
                                        Some(v) => b.set(slot, v),
                                        None => b.unset(slot),
                                    }
                                }
                            }
                        }

                        if path_slot.is_some() {
                            paths.push(Val::path(v.to_vec(), e.to_vec()));
                        }

                        ends.push(end);
                        src.push(i);
                        true
                    };

                if let Some(unit) = seg.unit.as_ref() {
                    crate::gql::eval::pathfind::reachable_each_unit(
                        graph,
                        ctx,
                        &mut wb,
                        from,
                        unit,
                        spec,
                        &mut on_end,
                    );
                } else {
                    crate::gql::eval::pathfind::reachable_each(
                        graph,
                        ctx,
                        &mut wb,
                        from,
                        rel,
                        spec,
                        &mut on_end,
                    );
                }

                if ends.len() as u64 > graph.limits().intermediate {
                    ctx.set_fault(FAULT_INTERMEDIATE);
                    return None;
                }
            }

            frontier.replicate(&src, &ends, node.var_slot);

            for (slot, col) in seg_groups.iter().zip(groups) {
                frontier.set_values(*slot, col);
            }

            if let Some(slot) = path_slot {
                path_col = Some((slot, std::mem::take(&mut paths)));
            }

            continue;
        }

        if frontier
            .expand(
                graph,
                &hop,
                graph.limits().intermediate,
                hop_cap,
                &mut filter,
            )
            .is_err()
        {
            // Surfaced as `E_RESOURCE_EXHAUSTED` at the row boundary. Returning
            // drops the partial frontier, so the memory is released rather than
            // continuing to grow.
            ctx.set_fault(FAULT_INTERMEDIATE);
            return None;
        }
    }

    let mut sc = ScanCols::new(scope_len);

    sc.n = frontier.rows();

    // `kinds` lists a self-joined slot TWICE, and a column can only be taken
    // once — the second take would install an empty column beside full ones.
    let mut taken: HashSet<usize> = HashSet::new();

    for &(s, e) in &kinds {
        if taken.insert(s) {
            sc.set_elems(s, e, frontier.take_column(s).unwrap_or_default());
        }
    }

    // …and the carried ones, fanned out by the same hops.
    if let Some(src) = carry {
        for s in 0..sc.cols.len().min(src.cols.len()) {
            if !taken.insert(s) {
                continue;
            }

            if let Some((e, _)) = src.slot(s) {
                if let Some(ids) = frontier.take_column(s) {
                    sc.set_elems(s, e, ids);
                }
            } else if src.val_slot(s).is_some() {
                if let Some(v) = frontier.take_values(s) {
                    sc.set_vals(s, v);
                }
            }
        }
    }

    if let Some((slot, vals)) = path_col {
        sc.set_vals(slot, vals);
    }

    for &s in &built {
        if let Some(vals) = frontier.take_values(s) {
            sc.set_vals(s, vals);
        }
    }

    Some(sc)
}

/// The per-segment part of an expansion that is GQL rather than IR: the node's
/// label, an edge-type expression too rich to lower, and the inline / `WHERE`
/// constraints on either end.
/// One inline `{k: v}` equality with its COLUMN already resolved.
///
/// `satisfies_const` still has to find the column per candidate —
/// `store.cols.get(kid)` plus the type match — and that is per NEIGHBOUR, so per
/// edge traversed. Hoisting it is what closes the rest of the gap between the
/// inline spelling and the `WHERE` spelling, which filters the built frame with
/// one dispatch for the whole column.
///
/// Only the two column types worth specializing; anything else keeps the general
/// path rather than growing a second copy of the equality rules here.
enum ConstEq<'a> {
    Num {
        data: &'a [f64],
        present: &'a crate::graph::BitSet,
        want: f64,
    },
    /// Interned ids compare as `u32`. `want` is `None` when the string is not in
    /// the dictionary at all — then nothing can equal it, and the row is dropped
    /// without touching the column.
    Str {
        data: &'a [u32],
        present: &'a crate::graph::BitSet,
        want: Option<u32>,
    },
}

impl ConstEq<'_> {
    fn holds(&self, idx: usize) -> bool {
        match self {
            Self::Num {
                data,
                present,
                want,
            } => present.get(idx) && data[idx] == *want,
            Self::Str {
                data,
                present,
                want,
            } => want.is_some_and(|w| present.get(idx) && data[idx] == w),
        }
    }
}

/// Resolve every constraint against `store`, or `None` if any one of them is not
/// a plain typed equality this can answer.
fn const_eqs<'a>(
    store: &'a crate::graph::Properties,
    strs: &crate::graph::Dict,
    ctx: &Ctx,
    consts: &[(usize, Val)],
    edge: bool,
) -> Option<Vec<ConstEq<'a>>> {
    consts
        .iter()
        .map(|(key_ref, want)| {
            let kid = if edge {
                ctx.prop_keys[*key_ref].1
            } else {
                ctx.prop_keys[*key_ref].0
            }?;

            match (store.cols.get(kid as usize)?, want) {
                (crate::graph::Column::Num { data, present }, Val::Num(w)) => Some(ConstEq::Num {
                    data,
                    present,
                    want: *w,
                }),
                (crate::graph::Column::Str { data, present }, Val::Str(w)) => Some(ConstEq::Str {
                    data,
                    present,
                    want: strs.get(w),
                }),
                _ => None,
            }
        })
        .collect()
}

struct SegmentFilter<'a> {
    graph: &'a Graph,
    ctx: &'a Ctx<'a>,
    rel: &'a CRel,
    node: &'a CNode,
    residual_type: bool,
    rel_check: bool,
    node_check: bool,
    /// Inline constraints with their values already evaluated — see
    /// [`crate::gql::eval::const_props`]. `None` when a value reads a slot, or
    /// when there is a WHERE beside the props, either of which needs the row.
    rel_consts: Option<Vec<(usize, Val)>>,
    node_consts: Option<Vec<(usize, Val)>>,
    /// The same constraints with their columns resolved — the hot form.
    rel_eqs: Option<Vec<ConstEq<'a>>>,
    node_eqs: Option<Vec<ConstEq<'a>>>,
    kinds: &'a [(usize, Elem)],
    binding: Binding,
}

impl crate::seek::RowFilter for SegmentFilter<'_> {
    fn row(&mut self, cols: &[Option<Vec<u32>>], row: usize) {
        if !(self.rel_check || self.node_check) {
            return;
        }

        // Slots already known are constant across this row's neighbours, so bind
        // them once rather than per neighbour.
        for &(s, kind) in self.kinds {
            if Some(s) == self.rel.var_slot || Some(s) == self.node.var_slot {
                continue;
            }

            if let Some(col) = cols.get(s).and_then(Option::as_ref) {
                if let Some(&id) = col.get(row) {
                    self.binding.set(
                        s,
                        match kind {
                            Elem::Node => Val::Node(id),
                            Elem::Edge => Val::Edge(id),
                        },
                    );
                }
            }
        }
    }

    fn keep(&mut self, eidx: u32, nbr: u32) -> bool {
        if self.residual_type {
            if let Some(l) = self.rel.label.as_ref() {
                if !eval_label_edge_at(self.graph, self.ctx, eidx, l) {
                    return false;
                }
            }
        }

        if !matches_label(self.graph, self.ctx, nbr, self.node.label.as_ref()) {
            return false;
        }

        if !(self.rel_check || self.node_check) {
            return true;
        }

        if let Some(s) = self.rel.var_slot {
            self.binding.set(s, Val::Edge(eidx));
        }

        if let Some(s) = self.node.var_slot {
            self.binding.set(s, Val::Node(nbr));
        }

        if self.rel_check {
            if let Some(eqs) = &self.rel_eqs {
                if !eqs.iter().all(|e| e.holds(eidx as usize)) {
                    return false;
                }
            } else {
                let ok = match &self.rel_consts {
                    Some(consts) => crate::gql::eval::satisfies_const(
                        self.graph,
                        self.ctx,
                        &Val::Edge(eidx),
                        consts,
                    ),
                    None => satisfies(
                        self.graph,
                        self.ctx,
                        &Val::Edge(eidx),
                        &self.rel.props,
                        self.rel.where_.as_ref(),
                        &self.binding,
                    ),
                };

                if !ok {
                    return false;
                }
            }
        }

        if !self.node_check {
            return true;
        }

        if let Some(eqs) = &self.node_eqs {
            return eqs.iter().all(|e| e.holds(nbr as usize));
        }

        match &self.node_consts {
            Some(consts) => {
                crate::gql::eval::satisfies_const(self.graph, self.ctx, &Val::Node(nbr), consts)
            }
            None => satisfies(
                self.graph,
                self.ctx,
                &Val::Node(nbr),
                &self.node.props,
                self.node.where_.as_ref(),
                &self.binding,
            ),
        }
    }
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
            .is_none_or(|lbl| eval_label_edge_at(graph, ctx, ei as u32, lbl))
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
        sc.set_elems(s, Elem::Node, a_ids);
    }
    if let Some(s) = rel.var_slot {
        sc.set_elems(s, Elem::Edge, r_ids);
    }
    if let Some(s) = node.var_slot {
        sc.set_elems(s, Elem::Node, b_ids);
    }
    Some(sc)
}

/// Build a new row set holding only rows `idx`, in that order (for ORDER BY: the
/// sorted window — gathers the few output rows instead of projecting all of `sc`).
pub(super) fn gather_rows(sc: &ScanCols, idx: &[usize]) -> ScanCols {
    let mut out = ScanCols::new(sc.cols.len());
    out.n = idx.len();
    for (s, col) in sc.cols.iter().enumerate() {
        match col {
            Some(Col::Elems { ids, is_edge }) => out.set_elems(
                s,
                if *is_edge { Elem::Edge } else { Elem::Node },
                idx.iter().map(|&i| ids[i]).collect(),
            ),
            Some(Col::Gen(vals)) => {
                out.set_vals(s, idx.iter().map(|&i| vals[i].clone()).collect());
            }
            _ => {}
        }
    }
    out
}

/// A contiguous row-range view of a frame as its own (owned) `ScanCols` — used to
/// split a large frame into chunks for parallel column evaluation.
#[cfg(feature = "parallel-query")]
pub(super) fn slice_rows(sc: &ScanCols, lo: usize, hi: usize) -> ScanCols {
    let mut out = ScanCols::new(sc.cols.len());
    out.n = hi - lo;
    for s in 0..sc.cols.len() {
        if let Some((e, ids)) = sc.slot(s) {
            out.set_elems(s, e, ids[lo..hi].to_vec());
        } else if let Some(v) = &sc.vals[s] {
            out.set_vals(s, v[lo..hi].to_vec());
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
) -> Vec<Col> {
    // Columns stay COLUMNS. `eval_vec` already produces one, and boxing it here
    // meant everything downstream — the window, DISTINCT, the row build — worked
    // on `Vec<Val>` and could not use any of the shared column operations. A
    // projection of numbers now reaches the output still unboxed.
    let serial = || {
        items
            .iter()
            .map(|it| eval_vec(graph, ctx, sc, &it.expr))
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
                let mut parts: Vec<Vec<Col>> = ranges
                    .par_iter()
                    .map(|&(lo, hi)| {
                        let sub = slice_rows(sc, lo, hi);
                        items
                            .iter()
                            .map(|it| eval_vec(graph, ctx, &sub, &it.expr))
                            .collect()
                    })
                    .collect();
                // Chunks agree on representation by construction — same
                // expression, same frame — so `append` keeps a number column a
                // number column across the join.
                let mut cols = parts.remove(0);

                for part in parts {
                    for (j, c) in part.into_iter().enumerate() {
                        cols[j].append(c);
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
    for c in sc.cols.iter_mut().flatten() {
        c.retain_rows(keep);
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
        // Canonicalized like the scalar key (`group_num_bits`): a stored -0 must
        // group with 0, and a stored NaN with any other NaN.
        Some(Column::Num { data, present }) => Some(
            (0..sc.n)
                .map(|i| {
                    bits(
                        i,
                        present,
                        crate::value::group_key_bits(data[ids[i] as usize]),
                    )
                })
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
    let want = if spec.func == AggFn::Min {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    let mut fold = |x: f64| {
        sum += x;
        n += 1;
        ext = Some(match ext {
            // `num_total_cmp`, not `f64::min`/`max`: those DROP a NaN, so `max`
            // would return the largest real number where the sort/aggregate
            // policy makes NaN the greatest value and `max` keeps it.
            Some(e) => {
                if num_total_cmp(x, e) == want {
                    x
                } else {
                    e
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
            if matches!(spec.func, AggFn::Min | AggFn::Max) && !matches!(av, Col::Num { .. }) {
                return None;
            }
            // Temporal (gathered → `Gen`) sum/avg can't go through the numeric fold
            // (it would NaN → null); bail to the scalar accumulator, which sums
            // DURATIONs and faults on avg / non-summable kinds.
            if matches!(spec.func, AggFn::Sum | AggFn::Avg) && matches!(av, Col::Gen(_)) {
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
                                // The same TOTAL order the scalar `cmp_total` path
                                // uses — NaN greatest, NaN == NaN — so `max` keeps a
                                // NaN and `min` never picks one.
                                //
                                // This was `partial_cmp`, which answers None against a
                                // NaN: never the wanted ordering, so a first-seen NaN
                                // STUCK and `min(sqrt(x))` returned it. (`f64::min/max`
                                // is the opposite error — it drops NaN, so `max` misses
                                // one.) The vectorized and scalar paths have to agree
                                // here; nothing tells a caller which one answered.
                                Some(x) => {
                                    let want = if is_min {
                                        Ordering::Less
                                    } else {
                                        Ordering::Greater
                                    };
                                    if num_total_cmp(d[i], x) == want {
                                        d[i]
                                    } else {
                                        x
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

    let mut b = Binding(vec![None; sc.cols.len()]);
    if proj.order_by.is_empty() {
        // No ORDER BY: emit groups in first-seen order, applying SKIP/LIMIT directly.
        let (start, end) = proj.window(ctx, ngroups);
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
    let (start, end) = proj.window(ctx, ngroups);
    let mut idx: Vec<usize> = (0..ngroups).collect();
    // Partial sort for a LIMIT — `keep_smallest`, shared with the scalar
    // projection and with Gremlin's `order()`.
    crate::value::keep_smallest(&mut idx, Some(end), cmp);
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
/// `RETURN *` rewritten as an explicit item list, or `None` if it isn't a star.
///
/// A star projection carries every input slot across in scope order — which is
/// `RETURN a, b, c` with the names filled in. `proj.items` is EMPTY for a star,
/// so every path that projects through `items` (the sort, DISTINCT, the terminal
/// column build) would emit nothing; desugaring here means none of them needs a
/// star case at all.
fn desugar_star(proj: &CProjection) -> Option<CProjection> {
    if !proj.star {
        return None;
    }

    let items = proj
        .star_cols
        .iter()
        .zip(&proj.out_names)
        .map(|(&slot, name)| {
            let expr = CExpr::Var(slot);

            CReturnItem {
                prog: crate::gql::plan::compile_program(&expr),
                expr,
                name: name.clone(),
                is_agg: false,
            }
        })
        .collect();

    Some(CProjection {
        star: false,
        items,
        ..proj.clone()
    })
}

pub(super) fn vectorized_cols(
    graph: &Graph,
    ctx: &Ctx,
    incoming: &[Binding],
    matches: &[&CClause],
    proj: &CProjection,
) -> Option<Vec<Col>> {
    if let Some(p) = desugar_star(proj) {
        return vectorized_cols(graph, ctx, incoming, matches, &p);
    }
    let sc = vectorized_frame(graph, ctx, incoming, matches, proj)?;
    project_frame_cols(graph, ctx, &sc, proj)
}

thread_local! {
    static FUSE_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn fusion_enabled() -> bool {
    !FUSE_OFF.with(std::cell::Cell::get)
}

/// Run `f` with comma-pattern fusion disabled, then restore it — so a test can
/// execute a query both ways and compare rows IN ORDER. Test-only.
#[cfg(test)]
pub(super) fn without_fusion<T>(f: impl FnOnce() -> T) -> T {
    let prev = FUSE_OFF.with(|c| c.replace(true));
    let out = f();
    FUSE_OFF.with(|c| c.set(prev));
    out
}

/// Fuse comma patterns that CHAIN into one path, or `None` if they don't.
///
/// Only the straightforward case: each pattern after the first must start on the
/// node the previous one ended on, named by the same slot, and must add nothing
/// to it — no label, no inline properties, no `WHERE`. A back-reference like the
/// `(b)` in `…, (b)-[s]->(c)` is exactly that, and re-checking constraints the
/// first pattern already applied is where a wrong fusion would hide.
///
/// Declines a path variable, a non-default selector or mode, since those change
/// what a path MEANS rather than just where it starts.
/// Every binding slot `p` names, in path order.
fn path_slots(p: &CPath, out: &mut Vec<usize>) {
    out.extend(p.start.var_slot);

    for seg in &p.segments {
        out.extend(seg.rel.var_slot);
        out.extend(seg.node.var_slot);

        if let Some(unit) = seg.unit.as_ref() {
            unit.group_slots(out);
        }
    }
}

/// A clause whose patterns fall into several disconnected groups: build each
/// group's frame, cross them, then apply the clause `WHERE` to the product.
///
/// The `WHERE` runs LAST, on the crossed rows, which is where the scalar join
/// applies it too — it is checked once every pattern has bound. Filtering a group
/// early would be a real optimization and is deliberately not done here: a
/// predicate over one group's variables alone is not distinguished yet, and
/// getting that wrong silently drops rows.
fn crossed_frame(
    graph: &Graph,
    ctx: &Ctx,
    groups: &[CPath],
    where_: &Option<CExpr>,
    scope_len: usize,
    proj: &CProjection,
) -> Option<ScanCols> {
    // Only a group that shares NO variable with another may be crossed. Groups
    // split whenever a pattern does not attach at the accumulated END, which also
    // catches shapes that DO share a variable elsewhere — diverging
    // (`(b)-[]->(a), (b)-[]->(c)`) or joining mid-path. Crossing those would drop
    // the equality the shared name asserts and return the full product, which is
    // exactly what happened the first time this was written: 8 rows where the
    // join gives 2.
    let mut slots: Vec<usize> = Vec::new();

    for g in groups {
        let before = slots.len();

        path_slots(g, &mut slots);

        if slots[..before].iter().any(|a| slots[before..].contains(a)) {
            return None;
        }
    }

    // A cap can't early-stop a cross product: the LAST group's rows are the inner
    // loop, so stopping there would truncate every outer row's pairing.
    let mut acc: Option<ScanCols> = None;

    for g in groups {
        let one = build_scan(graph, ctx, g, scope_len, None, where_.as_ref(), None)?;

        acc = Some(match acc {
            None => one,
            Some(prev) => cross_frames(&prev, &one, graph.limits().intermediate)?,
        });
    }

    let mut sc = acc?;

    if let Some(w) = where_ {
        let keep: Vec<bool> = eval_vec(graph, ctx, &sc, w)
            .into_truth()
            .iter()
            .map(|t| *t == Some(true))
            .collect();

        compact(&mut sc, &keep);
    }

    let _ = proj;

    Some(sc)
}

/// The CROSS PRODUCT of two frames: every row of `a` paired with every row of
/// `b`.
///
/// `a`'s columns repeat each value `b.n` times and `b`'s tile, which is exactly
/// the order the scalar join produces — it nests pattern 2 inside pattern 1 —
/// so this is a drop-in for a clause whose patterns share no variable.
///
/// `budget` bounds the result the way an expansion's does: `n * m` reaches
/// billions on two large scans, and a cartesian product is usually a mistake, so
/// it must cap rather than materialize.
fn cross_frames(a: &ScanCols, b: &ScanCols, budget: u64) -> Option<ScanCols> {
    let n = a.n.checked_mul(b.n)?;

    if n as u64 > budget {
        return None;
    }

    let width = a.cols.len().max(b.cols.len());
    let mut out = ScanCols::new(width);

    out.n = n;

    // Four cases collapse to two: which SIDE the slot came from decides how it is
    // laid out, and the column knows how to repeat or tile itself whatever it
    // holds. This used to be a branch per (side, representation) pair, and the
    // pair that tiled a value column was the one that had to be written twice.
    for s in 0..width {
        let col = if let Some(c) = a.cols.get(s).and_then(Option::as_ref) {
            // `a` repeats: each of its rows held for a whole pass of `b`.
            c.clone().repeat_each(b.n)
        } else if let Some(c) = b.cols.get(s).and_then(Option::as_ref) {
            // `b` tiles: its whole column once per row of `a`.
            c.clone().tile(a.n)
        } else {
            continue;
        };

        out.cols[s] = Some(col);
    }

    Some(out)
}

/// Split `patterns` into the fewest paths that cover them: each consecutive run
/// that CHAINS is spliced into one, and a pattern that joins nowhere starts a new
/// group. `None` if any pattern is not plain enough to fuse at all.
///
/// Splicing is what makes `MATCH (a)-[]->(b), (b)-[]->(c)` cost the same as
/// `MATCH (a)-[]->(b)-[]->(c)`: each pattern after the first must start on the
/// node the previous ended on, in either direction (`reverse_path` flips one when
/// they converge), and whatever that shared node constrains is carried across by
/// `merge_node`. A path variable, a non-default selector or mode declines — those
/// change what a path MEANS, not just where it starts.
///
/// One group means the whole clause is a single path. Several means the clause is
/// a CROSS PRODUCT of those paths — patterns that share no variable, like
/// `MATCH (a)-[]->(b), (c)-[]->(d)`. That is not a join to be spliced; it is a
/// different operation, and `cross_frames` performs it.
pub(super) fn fuse_groups(patterns: &[CPath]) -> Option<Vec<CPath>> {
    // One pattern has nothing to group, and returning `None` lets the caller
    // BORROW it. Building a one-element Vec here clones the whole `CPath` —
    // labels, props, `WHERE` trees — on every execution. Invisible on a 500us
    // join; 2.7x on `RETURN n.name LIMIT 100`, where the query is 1.6us and a
    // couple of allocations are the whole cost.
    if patterns.len() < 2 || !fusion_enabled() {
        return None;
    }

    let plain = |p: &CPath| {
        p.path_var_slot.is_none() && p.selector == PathSelector::Walk && p.mode == PathMode::Trail
    };

    if !patterns.iter().all(plain) {
        return None;
    }

    let mut groups: Vec<CPath> = Vec::new();
    let mut out = patterns[0].clone();

    for next in &patterns[1..] {
        let Some(end) = out
            .segments
            .last()
            .map_or(out.start.var_slot, |s| s.node.var_slot)
        else {
            groups.push(std::mem::replace(&mut out, next.clone()));
            continue;
        };

        if attach(next, end).is_none() {
            groups.push(std::mem::replace(&mut out, next.clone()));
            continue;
        }
        let end = out
            .segments
            .last()
            .map_or(out.start.var_slot, |s| s.node.var_slot)?;

        // Either side may need walking backwards. `(a)-[]->(b), (c)-[]->(b)`
        // converges on `b`, so the SECOND reverses; `(b)-[]->(a), (b)-[]->(c)`
        // diverges from `b`, so the FIRST does. `reverse_path` flips each
        // direction, binding the same edges and nodes from the other end.
        // Reversing the ACCUMULATED path is not an option, though it would fuse
        // the diverging shape `(b)-[]->(a), (b)-[]->(c)` into
        // `(a)<-[]-(b)-[]->(c)`. A linear path can only be enumerated from an
        // end, so that fused form drives from `a` while the join drives from `b`,
        // and the rows come out in a different ORDER — same multiset, regrouped.
        // `fusing_comma_patterns_preserves_rows_and_order` catches it; it was
        // written before this was attempted, and is why it did not ship.
        let joined = attach(next, end).expect("checked just above");

        // The shared node may be constrained on BOTH sides —
        // `(a)-[]->(b:N), (b:M {k: 1})-[]->(c)`. They name one variable, so the
        // node must satisfy both; MERGE them onto the fused node rather than
        // declining. Dropping either would silently widen the match.
        merge_node(end_node_mut(&mut out), &joined.start);
        out.segments.extend(joined.segments);
    }

    groups.push(out);

    Some(groups)
}

/// The node the accumulated path currently ends on.
fn end_node_mut(p: &mut CPath) -> &mut CNode {
    match p.segments.last_mut() {
        Some(seg) => &mut seg.node,
        None => &mut p.start,
    }
}

/// Conjoin `from`'s constraints onto `into` — the two name the same variable.
fn merge_node(into: &mut CNode, from: &CNode) {
    into.label = match (into.label.take(), from.label.clone()) {
        (Some(a), Some(b)) => Some(CLabelExpr::And(Box::new(a), Box::new(b))),
        (a, b) => a.or(b),
    };
    into.props.extend(from.props.iter().cloned());
    into.where_ = match (into.where_.take(), from.where_.clone()) {
        (Some(a), Some(b)) => Some(CExpr::And(vec![a, b])),
        (a, b) => a.or(b),
    };
}

/// `p` oriented to start on slot `end`, either as written or reversed, or `None`
/// if it does not join there.
fn attach(p: &CPath, end: usize) -> Option<CPath> {
    if joins_at(p, end) {
        return Some(p.clone());
    }

    let flipped = (!p.segments.is_empty()).then(|| reverse_path(p))?;

    joins_at(&flipped, end).then_some(flipped)
}

/// Does `p` start on slot `end`? Whatever that node constrains is carried across
/// by [`merge_node`], so a constrained back-reference joins like a bare one.
fn joins_at(p: &CPath, end: usize) -> bool {
    p.start.var_slot == Some(end)
}

/// Decline the columnar frame and fall to the scalar driver.
macro_rules! bail {
    () => {
        return None
    };
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
    // A prior `WITH`/`INSERT` already produced bindings.
    //
    // REJECTED (measured neutral): seeding the frame from those bindings and
    // continuing through `expand_frame`, which is what the design note proposed
    // for the "carrying a `WITH` costs 4.8x" finding. It was written and
    // measured: 1.70x vs 1.72x, 3.47x vs 3.55x — noise. The reason is that the
    // 4.8x is already gone. `vectorized_linear` is a SECOND columnar pipeline
    // that takes the whole `MATCH … WITH … MATCH … RETURN` shape before this
    // function is ever reached, and every shape in `with_carry_bench` hits it.
    // The entry added here fired 4 times in the entire GQL suite.
    //
    // What remains (1.7x on a carried scalar, 3.5x on a carried element) is the
    // cost of the `WITH` BARRIER — materializing the intermediate binding table —
    // not a fall back to the scalar driver. Fixing that means not materializing
    // at the barrier, which is a different piece of work; re-seeding a frame that
    // was just materialized cannot recover it.
    if incoming.len() != 1 || incoming[0].0.iter().any(|c| c.is_some()) {
        bail!();
    }
    // Consecutive MATCH clauses are a join, exactly like comma patterns:
    // `MATCH (a)-[]->(b) MATCH (b)-[]->(c)` and
    // `MATCH (a)-[]->(b), (b)-[]->(c)` bind the same rows in the same order.
    // Flatten them into one clause so `fuse_chain` sees the whole join.
    // Consecutive MATCH clauses arrive here ALREADY merged into one clause with
    // several patterns — the planner does that — so this only refuses a genuine
    // multi-clause shape, such as a MATCH after a WITH. Flattening them here was
    // written and removed: it fired three times in its own test and never once on
    // the benchmark queries, because there was nothing left to flatten.
    if matches.len() != 1 {
        bail!();
    }

    // ORDER BY: an aggregate sorts its group rows internally ([`vectorized_aggregate`],
    // which resolves output aliases + aggregates). A non-aggregate sort over
    // OUTPUT aliases projects first and sorts the projected columns (see
    // `vectorized_rowset`). DISTINCT + ORDER BY stays scalar.
    let has_order = !proj.order_by.is_empty();
    let CClause::Match {
        optional: false,
        patterns,
        where_,
        scope_len,
        ..
    } = matches[0]
    else {
        bail!();
    };
    // `MATCH (a)-[r]->(b), (b)-[s]->(c)` is a JOIN written as two patterns, and
    // `MATCH (a)-[r]->(b)-[s]->(c)` is the same query written as one. Fuse the
    // first into the second so the single-pattern frame answers both — the
    // "collapse equivalent spellings into one shape" this IR is for.
    //
    // Sound because the trail restriction (no repeated edge) applies to a
    // QUANTIFIED walk, not to a fixed-length path: on a self-loop both spellings
    // return the same row with `r = s`, verified before this was written. And the
    // row ORDER is unchanged — the scalar join nests pattern 2 inside pattern 1,
    // which is exactly the order one fused path enumerates.
    let groups = fuse_groups(patterns);
    let path = match groups.as_deref() {
        Some([one]) => one,
        // Patterns that share no variable: build each group's frame and cross
        // them. Handled below, after the shape checks that apply to any of them.
        Some(_) => return crossed_frame(graph, ctx, groups.as_deref()?, where_, *scope_len, proj),
        None if patterns.len() == 1 => &patterns[0],
        None => return None,
    };

    // A bound path variable needs the scalar driver — only it builds the Path
    // value (`all_walk`/`shortest_walk`); the vectorized frame yields columns.
    // (A selector / non-default mode already routes here via the run_part guard.)
    if path.path_var_slot.is_some() {
        bail!();
    }

    // The WALK: one materialized column instead of a whole frame, when the only
    // slot anything reads is the one it lands on. Ahead of both bails below —
    // it beats the scalar stream-fold they defer to (`RETURN sum(b.n)` over a hop
    // was 1.16ms scalar against 0.50ms walked) and it applies the clause WHERE
    // itself, so it must return before the re-application at the end.
    if let Some(sc) = streamed_frame(graph, ctx, path, where_.as_ref(), *scope_len, proj) {
        return Some(sc);
    }

    // A pure aggregate over a traversal with no WHERE stays scalar: the scalar
    // engine stream-folds the join without materializing it, and there's no
    // per-row expression to vectorize. With a WHERE, the batched build + masked
    // count can pay for itself.
    if !path.segments.is_empty() && proj.aggregating && where_.is_none() {
        bail!();
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
        bail!();
    }

    // Every input slot the rest of this statement reads: the projection items,
    // the GROUP BY / ORDER BY keys, the lifted aggregates and the clause WHERE.
    // `None` the moment any of them hides an opaque `Op::Tree` (a subquery or
    // aggregate the flattener kept as an expression), because this is only used
    // to SKIP building a column and a missed slot would read back as null.
    // Only a parenthesized quantified unit produces group columns, and `needed`
    // exists solely to skip building the ones nothing reads. Computing it
    // otherwise is pure overhead — it COMPILES the sort keys and clause WHERE on
    // every execution, which measured 2.7x on `RETURN n.name LIMIT 100` (1.6us,
    // where a few allocations are the whole query).
    let needed: Option<Vec<usize>> = (|| {
        // Two consumers, so two reasons to compute it. Group columns cost an
        // allocation per row per variable; a bare-hop chain can skip the whole
        // JOIN when only its landing slot is read (`streamed_frame`). The gate
        // stays a cheap structural test either way — computing this eagerly
        // COMPILES the sort keys and clause WHERE on every execution, which
        // measured 2.7x on `RETURN n.name LIMIT 100`.
        let quantified = patterns
            .iter()
            .any(|p| p.segments.iter().any(|s| s.unit.is_some()));
        let walkable = patterns.len() == 1
            && !patterns[0].segments.is_empty()
            && patterns[0].path_var_slot.is_none()
            && patterns[0].segments.iter().all(|s| {
                s.unit.is_none()
                    && s.rel.quantifier.is_none()
                    && s.rel.props.is_empty()
                    && s.rel.where_.is_none()
                    && s.node.props.is_empty()
                    && s.node.where_.is_none()
                    && s.node.label.is_none()
            });

        if !quantified && !walkable {
            return None;
        }

        let mut out = Vec::new();

        for it in proj.items.iter().chain(&proj.group_by) {
            if !it.prog.read_slots(&mut out) {
                return None;
            }
        }

        for k in &proj.order_by {
            if !crate::gql::plan::compile_program(&k.expr).read_slots(&mut out) {
                return None;
            }
        }

        for a in &proj.aggs {
            if let Some(arg) = a.arg.as_ref() {
                if !crate::gql::plan::compile_program(arg).read_slots(&mut out) {
                    return None;
                }
            }
        }

        if let Some(w) = where_.as_ref() {
            if !crate::gql::plan::compile_program(w).read_slots(&mut out) {
                return None;
            }
        }

        // `RETURN *` carries these across verbatim.
        if proj.star {
            out.extend(proj.star_cols.iter().copied());
        }

        // The ORDER BY overlay is EVERY input slot — it exists so a sort key can
        // name an input the projection dropped. Only a query that actually sorts
        // reads it; folding it in unconditionally made `needed` universal and the
        // elision a no-op.
        if !proj.order_by.is_empty() {
            out.extend(proj.order_overlay.iter().copied());
        }

        Some(out)
    })();

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
    let mut sc = build_scan(
        graph,
        ctx,
        path,
        *scope_len,
        cap,
        where_.as_ref(),
        needed.as_deref(),
    )?;

    // Clause WHERE → keep mask (vectorized), compacting the row set.
    //
    // This re-applies a predicate the scan has usually ALREADY applied — the
    // clause WHERE is lowered into the seek by `seek_lower::scan_node`. Skipping
    // it when the seek answered exactly was written and measured: `scan_node`
    // reporting completeness (every conjunct lowered, `columnar()` for
    // answerable-from-a-typed-column) and this pass skipped on it. The flag fired
    // on 6303 of 6303 evaluations in `gql_bench` — and gained 2.5%.
    //
    // So the second pass is not the cost: `eval_vec` over a typed column is
    // genuinely cheap. Reverted rather than carry a signature change through four
    // functions plus a flag whose wrong value silently drops a filter, for a gain
    // under the noise floor. `[b] scan+count+pred` remains 1.2-1.3x against main
    // across runs — re-measured 2026-08-05 at 1.31x on a quiet machine — and is
    // NOT explained by this, by the residual pass (also measured, also neutral),
    // or by `Val::from_column` (measured, neutral — LTO already inlines it).
    // Three hypotheses, three disproofs; it is still open.
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
    let vvs: Vec<Col> = proj
        .items
        .iter()
        .map(|it| eval_vec(graph, ctx, sc, &it.expr))
        .collect();
    let mut rs = RowSet::with_capacity(proj.out_names.clone(), sc.n);
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
) -> Option<Vec<Col>> {
    let has_order = !proj.order_by.is_empty();
    // Aggregating + ORDER BY is handled inside `vectorized_aggregate` (it sorts
    // the group rows, resolving output aliases + aggregates).
    if proj.aggregating && has_order && proj.distinct {
        return None;
    }
    if proj.aggregating {
        // HAVING filters groups post-fold; that path lives in the scalar
        // `ProjAccum::finish`, so bail to it rather than duplicate it here.
        if proj.having.is_some() {
            return None;
        }
        // An aggregate produces one row per group, so there is no column-scale
        // win in keeping it typed — `Col::Gen` wraps what it already built, at no
        // cost, so both halves of this function return the same thing.
        return Some(
            vectorized_aggregate(graph, ctx, sc, proj)?
                .into_iter()
                .map(Col::Gen)
                .collect(),
        );
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
            if let Some((elem, ids)) = sc.slot(islot) {
                sort_sc.set_elems(proj.out_len + j, elem, ids.to_vec());
            } else if let Some(vals) = sc.val_slot(islot) {
                sort_sc.set_vals(proj.out_len + j, vals.to_vec());
            }
        }

        // `RETURN a.x AS n ORDER BY n` sorts by an OUTPUT column, which does not
        // exist yet — sort scope slots 0..out_len are empty above. Project first
        // and install the results there so the sort keys resolve.
        //
        // Installed by MOVE. Cloning the columns in cost 112us of the 400us that
        // separated `ORDER BY a` from `ORDER BY u.k` over 20k rows, for nothing:
        // the only other reader is DISTINCT, which reads them back out of the
        // sort scope just as well.
        //
        // This gives up the "a small LIMIT never materializes the full output
        // columns" property the input-keyed path has. That is the right trade
        // here and only here: this shape was 100% scalar before, so it is a win
        // against the scalar driver even having materialized.
        // Only the columns a sort key actually READS — the rest are projected
        // once, from the window, at the end. `RETURN u.k AS a, u.n AS b ORDER BY
        // a` needs `a` here and nothing else; projecting `b` too cost a full
        // column for a value the sort never looks at.
        //
        // DISTINCT is the exception: it keys on the whole projected row, so it
        // needs all of them.
        if proj.order_needs_output || proj.distinct {
            for (i, item) in proj.items.iter().enumerate() {
                let read_by_sort = proj
                    .order_by
                    .iter()
                    .any(|s| crate::gql::plan::refs_slot(&s.expr, &|slot| slot == i));

                if proj.distinct || read_by_sort {
                    sort_sc.set_vals(i, eval_vec(graph, ctx, sc, &item.expr).into_vals());
                }
            }
        }
        // REJECTED (measured neutral): the same dense treatment for a NUMERIC key
        // — a flat `f64` array instead of the boxed `Val` comparator, mirroring
        // the temporal one below. It was written for both spellings, the property
        // and the output alias, and moved nothing: 6.54ms vs 6.61ms on
        // `WITH a.n AS x RETURN x ORDER BY x DESC`, 17.0ms vs 16.6ms with an
        // aggregate beside it, over 50k rows.
        //
        // The reason is that the shape it was aimed at never arrives here.
        // `MATCH (a:V) RETURN a.n AS x ORDER BY x DESC` is a plain non-aggregating
        // `MATCH … RETURN`, which `vectorized_linear` declines by design, so the
        // SCALAR matcher sorts it — and that path has its own top-k (matcher.rs).
        // A full sort of 50k numbers there costs 6.4ms against Gremlin's 1.0ms,
        // which is a real gap and is NOT in this function.
        //
        // Also worth recording: at top-k the two engines already agree — 0.98ms
        // against 0.96ms for `ORDER BY … LIMIT 5` over 50k distinct values. The
        // 2x on `cross_engine_shortcuts` is that fixture having only 1000
        // distinct values, not a difference in the selection.
        //
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
        // A NUMERIC key, the same idea one type over: a flat `f64` array instead
        // of the boxed comparator.
        let num_key: Option<NumSortCol> = (dense_key.is_none())
            .then(|| {
                single.and_then(|s| {
                    num_sort_key(graph, ctx, &sort_sc, &s.expr).map(|k| (k, s.descending))
                })
            })
            .flatten();
        // Duration (no dense key): compare Copy temporals via `cmp_total`.
        let temporal_key: Option<TypedSortCol> = (dense_key.is_none() && num_key.is_none())
            .then(|| {
                single.and_then(|s| {
                    temporal_sort_key(graph, ctx, &sort_sc, &s.expr)
                        .map(|k| (k, s.descending, s.nulls_first))
                })
            })
            .flatten();
        // Only the generic path needs the `Vec<Val>` keycols.
        let keycols: Vec<Vec<Val>> =
            if dense_key.is_some() || num_key.is_some() || temporal_key.is_some() {
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
            if let Some((key, descending)) = &num_key {
                // No NaN reaches here — `num_sort_key` declines a column holding
                // one — so this is a plain comparison.
                let o = key[i].partial_cmp(&key[j]).unwrap_or(Ordering::Equal);
                let o = if *descending { o.reverse() } else { o };

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
        // DISTINCT + ORDER BY: dedup on the projected row, THEN sort. Which
        // duplicate survives is irrelevant — they are the same row by definition
        // — so this does not need the scan-order first-seen rule the unsorted
        // DISTINCT path keeps.
        let mut idx: Vec<usize> = if proj.distinct {
            // Read back out of the sort scope, where the projection above put
            // them (DISTINCT forces `projected_cols`).
            let cols: Vec<&[Val]> = (0..proj.items.len())
                .map(|i| {
                    sort_sc
                        .val_slot(i)
                        .expect("distinct forces the projection above")
                })
                .collect();
            let mut seen: FxHashSet<String> = FxHashSet::default();
            (0..sc.n)
                .filter(|&i| {
                    let mut key = String::new();
                    for c in &cols {
                        val_key(&c[i], &mut key);
                        key.push('\u{1}');
                    }
                    seen.insert(key)
                })
                .collect()
        } else {
            (0..sc.n).collect()
        };
        let (start, end) = proj.window(ctx, idx.len());
        // Partition the top `end` rows out in O(n), then sort just that window,
        // instead of an O(n log n) sort of every row to keep a small prefix. No
        // LIMIT ⇒ `end` is the row count and it is a full sort.
        crate::value::keep_smallest(&mut idx, Some(end), cmp);
        let window = &idx[start..end.min(idx.len())];

        // REJECTED (measured): windowing the columns projected above — "they are
        // already computed, so take the window out of them rather than evaluate a
        // second time". It is 2.7x SLOWER than evaluating again: 175us against
        // gather 8us + project 57us, over 20k rows. `window` is in SORTED order,
        // so indexing the columns by it is a random-access gather over 40-byte
        // boxed `Val`s, while `gather_rows` permutes a `Vec<u32>` and the
        // re-projection then reads the property column densely. Re-evaluating
        // wins outright, and it also restores the LIMIT property: only the window
        // is ever projected.
        let sub = gather_rows(sc, window);
        return Some(
            proj.items
                .iter()
                .map(|item| eval_vec(graph, ctx, &sub, &item.expr))
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
            let (start, end) = proj.window(ctx, ngroups);
            let mut out: Vec<Vec<Val>> = vec![Vec::with_capacity(end - start); proj.items.len()];
            let mut b = Binding(vec![None; sc.cols.len()]);
            for &ri in &rep_row[start..end] {
                bind_frame_row(&mut b, sc, ri);
                let env = Env::new(graph, ctx, &b);
                for (item_idx, item) in proj.items.iter().enumerate() {
                    out[item_idx].push(eval(&env, &item.expr));
                }
            }
            return Some(out.into_iter().map(Col::Gen).collect());
        }
    }

    // Non-aggregating projection: evaluate each item as a column (parallel over
    // row-chunks for a large frame).
    let mut cols: Vec<Col> = par_project(graph, ctx, sc, &proj.items);
    if proj.distinct {
        // Generic DISTINCT (expression / non-typed items): the first occurrence of
        // each row in scan order, keyed on a composite of its cells. The bucketing
        // is `group_first_seen`, shared — what is GQL's here is the KEY, `val_key`
        // over every projected cell.
        //
        // The cap is `skip + limit` GROUPS rather than rows, because the window is
        // taken over distinct rows: stopping at `limit` alone would cut the window
        // short whenever a SKIP precedes it.
        let skip = proj.skip_val(ctx);
        let cap = proj.limit_val(ctx).map(|l| skip + l);
        let kept: Vec<usize> = crate::value::group_first_seen(
            sc.n,
            |i| {
                let mut key = String::new();

                for c in &cols {
                    c.with_val_at(i, |v| val_key(v, &mut key));
                    key.push('\u{1}');
                }

                Some(key)
            },
            || (),
            |(), _| (),
            cap,
        )
        .into_iter()
        .skip(skip)
        .map(|(rep, ())| rep)
        .collect();

        Some(
            cols.iter()
                .map(|c| Col::Gen(kept.iter().map(|&i| c.val_at(i)).collect()))
                .collect(),
        )
    } else {
        // Window each column to the SKIP/LIMIT row range (no ORDER BY ⇒ scan
        // order) — `Col::page`, the same slice Gremlin's paging steps take.
        let (start, end) = proj.window(ctx, sc.n);

        Some(cols.drain(..).map(|c| c.page(start, end)).collect())
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
                out.set_elems(i, elem, ids.to_vec()); // carry element column forward
                continue;
            }
            if let Some(vals) = sc.val_slot(*slot) {
                out.set_vals(i, vals.to_vec()); // carry a prior computed column
                continue;
            }
        }
        out.set_vals(i, eval_vec(graph, ctx, sc, &item.expr).into_vals());
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
                    out.set_elems(i, elem, rep_row.iter().map(|&ri| ids[ri]).collect());
                    continue;
                }
                // A bare carried value column (a key from an upstream WITH): gather.
                if let Some(vals) = sc.val_slot(*slot) {
                    out.set_vals(i, rep_row.iter().map(|&ri| vals[ri].clone()).collect());
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
        let mut b = Binding(vec![None; sc.cols.len()]);
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
            out.set_vals(i, std::mem::take(&mut cols[k]));
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
    let Some((Elem::Node, start_ids)) = sc.slot(start_slot) else {
        return None; // the start must be an already-bound node column
    };
    let mut cur = sc.clone();
    let mut endpoint = start_ids.to_vec();

    // The RESTATED start node may add a label / inline props / WHERE. Apply them
    // here, before the shared expansion, because `expand_scan` seeds its start
    // from a scan and so checks those at seed time — there is no seed to check
    // when the start is a column that already exists.
    if start.label.is_some() || !start.props.is_empty() || start.where_.is_some() {
        let width = scope_len.max(sc.cols.len());
        let mut b = Binding::with_len(width);
        let mut keep = vec![false; cur.n];

        for i in 0..cur.n {
            bind_frame_row(&mut b, &cur, i);
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

        endpoint.retain({
            let mut it = keep.iter();
            move |_| *it.next().unwrap_or(&false)
        });
        compact(&mut cur, &keep);
    }

    expand_scan(
        graph,
        ctx,
        path,
        scope_len.max(sc.cols.len()),
        endpoint,
        SeedFrom {
            cap: None,
            needed: None,
            carry: Some(&cur),
        },
    )
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
    let width = scope_len.max(sc.cols.len());
    let rel_check = !rel.props.is_empty() || rel.where_.is_some();
    let node_check = !node.props.is_empty() || node.where_.is_some();
    let need_bind = rel_check || node_check;

    // Carried columns keep their kind (element/value); the segment's rel/node slots
    // are nullable value columns.
    let mut out = ScanCols::new(width);
    for s in 0..width {
        if Some(s) == rel.var_slot || Some(s) == node.var_slot {
            out.set_vals(s, Vec::new());
        } else if s < sc.cols.len() {
            if let Some((e, _)) = sc.slot(s) {
                out.set_elems(s, e, Vec::new());
            } else if sc.val_slot(s).is_some() {
                out.set_vals(s, Vec::new());
            }
        }
    }

    // Append one output row: carried columns read from outer row `i`; the segment's
    // rel/node value columns take `rv`/`nv` (both `Val::Null` for the no-match fill).
    let push = |out: &mut ScanCols, i: usize, rv: &Val, nv: &Val| {
        for s in 0..width {
            let v = if Some(s) == rel.var_slot {
                rv.clone()
            } else if Some(s) == node.var_slot {
                nv.clone()
            } else if s < sc.cols.len() {
                match sc.cols[s].as_ref() {
                    Some(c) => match c {
                        Col::Elems { ids, is_edge } => {
                            if *is_edge {
                                Val::Edge(ids[i])
                            } else {
                                Val::Node(ids[i])
                            }
                        }
                        Col::Gen(vals) => vals[i].clone(),
                        _ => continue,
                    },
                    None => continue,
                }
            } else {
                continue;
            };

            if let Some(c) = out.cols[s].as_mut() {
                c.push_val(&v);
            }
        }
    };

    let mut nb = Binding(vec![None; width]);
    let mut nrows = 0usize;
    for i in 0..sc.n {
        if need_bind {
            for s in 0..sc.cols.len() {
                if let Some((e, ids)) = sc.slot(s) {
                    nb.set(
                        s,
                        match e {
                            Elem::Node => Val::Node(ids[i]),
                            Elem::Edge => Val::Edge(ids[i]),
                        },
                    );
                } else if let Some(v) = sc.val_slot(s) {
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
