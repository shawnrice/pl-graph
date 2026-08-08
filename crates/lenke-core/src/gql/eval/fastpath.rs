//! Query-shape fast-paths: recognize special MATCH/RETURN shapes (bare
//! `count(*)`, two-hop counts, grouped var-length, distinct-reachable,
//! parallel scans/aggregates, …) and answer them without materializing the full
//! result. Each returns `None` to fall back to the general executor, and is
//! provably identical to it. Extracted from the evaluator (`super`); shares its
//! context/helpers via `use super::*`.
use super::*;

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
    if !walk_count_enabled() {
        return None;
    }
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
    // The projection is exactly `count(*)` — one aggregate, one item referring to
    // it, no DISTINCT and no argument. Spelled out rather than delegated: the
    // helper this used to call was removed with the `try_count` family, and this
    // arm only compiles under `parallel-query`, so nothing pointed at it.
    let bare_count_star = proj.aggregating
        && proj.aggs.len() == 1
        && proj.items.len() == 1
        && matches!(proj.items[0].expr, CExpr::AggRef(0))
        && {
            let a = &proj.aggs[0];
            a.star && !a.distinct && a.arg.is_none() && matches!(a.func, AggFn::Count)
        };
    if !bare_count_star {
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

/// `count(*)` and `count(DISTINCT <end>)` over a path of BARE hops, answered by
/// walking and counting in place instead of materializing a row per match.
///
/// This is the one idea the fifteen-rung `try_count_*` ladder was fifteen
/// spellings of: **a count does not need its rows.** `seek::walk_count` folds
/// the walk as it goes — it is the same function Gremlin's `.count()` uses, and
/// it takes `distinct` as a parameter, so one call covers three shapes that used
/// to be three rungs (`try_count_streamed`, `try_count_two_hop`,
/// `try_count_distinct_endpoint`) plus the var-length one (`try_count_varlen_
/// upto_2`) that was a fourth.
///
/// ```text
/// varlen_all_1_2   MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN count(*)
/// distinct_2hop    MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c)
///                    RETURN count(DISTINCT c)
/// distinct_nbr     MATCH (a:Person)-[:KNOWS]->(b) RETURN count(DISTINCT b)
/// ```
///
/// **REJECTED (measured): extending this to a LABELLED landing node.** The
/// refusal below looks like the reason `MATCH (a:Person)-[:KNOWS]->(b:Hub)
/// RETURN count(*)` is 1.9x main, and it is not. An arm that seeds from the
/// smaller of the two label buckets and tests the other end per edge was
/// written and measured against the general path:
///
/// ```text
///              with the arm   without
/// 200k/8          0.018ms      0.020
/// 1M/8            0.198        0.195
/// ```
///
/// Nothing, at either size — the planner already orients this shape to the
/// small end, so the arm reimplements an existing decision at the same speed
/// for ~150 lines. What is actually left is the per-edge label test on the far
/// side: 8000 random `vlabels` lookups over a million vertices, cache-miss
/// bound, which both routes pay. Beating it needs the label check to stop being
/// a random read (a per-label bitset), not another seeding rule.
///
/// **Bare is the whole precondition, and it is not a formality.** `walk_count`
/// never produces a row, so a predicate here would be a predicate that never
/// runs — and it would not fail loudly, it would return a bigger number. Only
/// the START may be constrained, because `scan_node` applies that constraint
/// while building the seed set (and seeds from an index when one exists). Every
/// hop after it must be a plain typed hop onto an unconstrained node, and the
/// MATCH's own `WHERE` must be absent.
///
/// A quantified hop `->{n,m}` is the sum of the fixed walks of each length in
/// `n..=m`, which is why it needs no separate rung: `{1,2}` is `walk_count` at
/// one hop plus `walk_count` at two. With `DISTINCT` that decomposition would be
/// WRONG — the same endpoint reachable at both lengths would be counted twice —
/// so the two features are not combined here.
///
/// **The two spellings of a two-hop are not the same question**, which is the
/// thing to know before touching this:
///
/// ```text
/// MATCH (a)-[:R]->()-[:R]->(c)   counts WALKS  — an edge may repeat
/// MATCH (a)-[:R]->{2,2}(c)       counts TRAILS — an edge may not
/// ```
///
/// A quantified repetition is edge-distinct and separate segments are not. That
/// is this engine's existing behaviour, pinned by
/// `varlen_fixed_lengths_match_trail_enumeration` on one side and
/// `the_walk_count_shortcut_agrees_with_enumeration` on the other; the shortcut
/// matches the matcher rather than picking a side. (Cypher applies relationship
/// uniqueness across the whole `MATCH`, so the unquantified spelling is a
/// deliberate-looking divergence — but it is a PRE-EXISTING one, and a counting
/// shortcut is the wrong place to change it.)
///
/// So `walk_count` is exact for the unquantified form at any depth, `DISTINCT`
/// included. For the QUANTIFIED form it over-counts, and the first version of
/// this shipped that as a silently larger number. Over two hops the only way to
/// reuse an edge is to take the same one twice, which requires it to be a
/// SELF-LOOP (`target(e) == source(e)`, since the midpoint has to be both) — so
///
/// ```text
/// trails(2) = walks(2) − (self-loops at a seed matching BOTH hops)
/// ```
///
/// which `self_loop_overcount` computes exactly, and
/// `varlen_fixed_lengths_match_trail_enumeration` /
/// `varlen_1_2_count_matches_trail_enumeration` check against brute force (13
/// walks vs 11 trails; 19 vs 17 — two self-loops either way).
///
/// At THREE hops the correction is no longer a subtraction: `a→b, b→a, a→b`
/// repeats an edge with no self-loop anywhere, so inclusion-exclusion starts.
/// Nothing here goes past two hops, and that is the reason rather than an
/// arbitrary cap.
pub(super) fn try_walk_count(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    if !walk_count_enabled() {
        return None;
    }
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
    if path.segments.is_empty() || path.path_var_slot.is_some() {
        return None;
    }
    if !matches!(path.selector, PathSelector::Walk) {
        return None;
    }

    // The projection is exactly one `count`, referred to once, with no other
    // output column — `RETURN count(*), a.name` needs the rows this never
    // builds.
    if !proj.aggregating || proj.aggs.len() != 1 || proj.items.len() != 1 {
        return None;
    }
    if !matches!(proj.items[0].expr, CExpr::AggRef(0)) {
        return None;
    }
    // No grouping, no paging, no HAVING, no DISTINCT on the projection itself,
    // no ORDER BY — all of them need the rows or the groups this never builds.
    // (A `LIMIT` over one aggregate row is harmless but not worth the branch.)
    if !proj.group_by.is_empty()
        || proj.limit.is_some()
        || proj.skip.is_some()
        || proj.having.is_some()
        || proj.distinct
        || !proj.order_by.is_empty()
    {
        return None;
    }
    let agg = &proj.aggs[0];
    if !matches!(agg.func, AggFn::Count) {
        return None;
    }

    // `count(*)`, or `count(DISTINCT x)` where `x` is the path's LAST node —
    // the endpoint `walk_count`'s own `distinct` deduplicates. Anything else
    // (a distinct count of an intermediate node, or of a property) is a
    // different question.
    let last_slot = path.segments.last()?.node.var_slot;
    let distinct = if agg.star && !agg.distinct && agg.arg.is_none() {
        false
    } else if agg.distinct && !agg.star {
        match agg.arg.as_ref()? {
            CExpr::Var(s) if Some(*s) == last_slot => true,
            _ => return None,
        }
    } else {
        return None;
    };

    // Every hop bare, and its type lowered to ids. A quantified hop is allowed
    // only as the ONLY segment, since summing lengths across several quantified
    // segments is a product, not a sum.
    let ctx = resolve_ctx(graph, plan, params);
    let mut hops: Vec<(crate::seek::Dir, Option<Vec<u32>>)> = Vec::new();
    let mut quant: Option<Quantifier> = None;
    for (i, seg) in path.segments.iter().enumerate() {
        if seg.unit.is_some() {
            return None;
        }
        let rel = &seg.rel;
        if rel.var_slot.is_some() || !rel.props.is_empty() || rel.where_.is_some() {
            return None;
        }
        if let Some(q) = rel.quantifier.as_ref() {
            if path.segments.len() != 1 {
                return None;
            }
            quant = Some(*q);
        }
        // The landing node carries no filter — see the doc: a filter here is one
        // that never runs.
        let node = &seg.node;
        if node.label.is_some() || !node.props.is_empty() || node.where_.is_some() {
            return None;
        }
        // …and no node may be re-bound to a slot the walk cannot check. A slot
        // that repeats is a self-join (`(a)-[:R]->(a)`), which is a filter by
        // another name.
        if let Some(s) = node.var_slot {
            if Some(s) == path.start.var_slot {
                return None;
            }
            if path.segments[..i]
                .iter()
                .any(|prev| prev.node.var_slot == Some(s))
            {
                return None;
            }
        }

        let etypes = match rel.label.as_ref() {
            None => None,
            Some(l) => Some(seek_lower::lower_labels(l, &ctx, true)?),
        };
        let dir = match rel.direction {
            Direction::Out => crate::seek::Dir::Out,
            Direction::In => crate::seek::Dir::In,
            Direction::Both => crate::seek::Dir::Both,
        };
        hops.push((dir, etypes));
    }

    let seeds = seek_lower::scan_node(graph, &ctx, &path.start, None, *scope_len, None);
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }

    // `SelfLoops::Once` — GQL's convention, the one `expand` walks with.
    let loops = crate::seek::SelfLoops::Once;

    // Walks become trails by subtracting the repeats, and only a length-2 walk
    // has a repeat this cheap to find. `DISTINCT` cannot be corrected by
    // subtraction at all — it is a SET, and an endpoint reachable only by
    // reusing an edge has to leave the set, not decrement a counter — so it is
    // confined to one hop, where walks and trails are the same thing.
    let count: u64 = match quant {
        // Separate segments are WALKS in this engine — no correction, and
        // `DISTINCT` is just the walk-reachable endpoint set, at any depth.
        None => crate::seek::walk_count(graph, &seeds, &hops, loops, distinct) as u64,
        Some(q) => {
            // A quantified hop with DISTINCT would double-count an endpoint
            // reachable at two different lengths.
            if distinct {
                return None;
            }
            let hop = hops.first()?.clone();
            let max = q.max?;
            // Past two the trail correction stops being a subtraction — see the
            // doc. `{1,3}` and beyond keep the general enumeration.
            if max > 2 || q.min > max {
                return None;
            }
            // `{0,n}` includes the ZERO-length walk: every seed is its own
            // endpoint. Writing the range as `min.max(1)..=max` silently drops
            // that term, which is a smaller count that looks entirely
            // reasonable — `{0,2}` came out 15 where the matcher says 19.
            // `{0,0}` is left to the matcher: it is the whole answer rather
            // than a term of it, and the matcher's own reading of it is not
            // obviously right (`a_zero_bound_quantifier_matches_the_matcher`).
            if max == 0 {
                return None;
            }
            let mut total = if q.min == 0 { seeds.len() as u64 } else { 0 };
            for len in q.min.max(1)..=max {
                let chain = vec![hop.clone(); len as usize];
                let walks = crate::seek::walk_count(graph, &seeds, &chain, loops, false) as u64;
                total += if len == 2 {
                    walks - self_loop_overcount(graph, &seeds, &chain)?
                } else {
                    walks
                };
            }
            total
        }
    };

    Some(count_rows_any(proj, count))
}

/// How many length-2 WALKS are not trails: the ones that take a single edge
/// twice. That needs the edge to leave and re-enter the same vertex, so it is
/// exactly the self-loops sitting at a seed and matching BOTH hops.
///
/// `None` declines — `Dir::Both` yields a self-loop under a different rule than
/// the two directed cases (once or twice, depending on `SelfLoops`), and getting
/// that wrong would show up as a count off by the number of self-loops rather
/// than as anything obviously broken.
fn self_loop_overcount(
    graph: &Graph,
    seeds: &[u32],
    hops: &[(crate::seek::Dir, Option<Vec<u32>>)],
) -> Option<u64> {
    let [(d1, t1), (d2, t2)] = hops else {
        return None;
    };
    if matches!(d1, crate::seek::Dir::Both) || matches!(d2, crate::seek::Dir::Both) {
        return None;
    }

    // `adj` has already applied hop ONE's type filter (and it reads every label
    // an edge carries, not just the primary one), so only hop two is left to
    // check — against the same `edge_labels` the walk itself uses.
    let hop2_ok = |eidx: u32| -> bool {
        match t2 {
            None => true,
            Some(ids) => graph.edge_labels(eidx).iter().any(|l| ids.contains(l)),
        }
    };

    let mut n = 0u64;
    for &s in seeds {
        for a in crate::seek::adj(
            graph,
            s,
            *d1,
            t1.as_deref().unwrap_or(&[]),
            crate::seek::SelfLoops::Once,
        ) {
            if a.nbr == s && hop2_ok(a.eidx) {
                n += 1;
            }
        }
    }
    Some(n)
}

/// Build the single-row `count` result for a projection.
pub(super) fn count_rows_any(proj: &CProjection, count: u64) -> CodeResult<RowSet> {
    let mut rs = RowSet::new(proj.out_names.clone());
    rs.push_row(std::iter::once(Value::Num(count as f64)));
    Ok(rs)
}

/// `RETURN <key over the endpoint>, count(*)` over bare hops — grouped without
/// enumerating a row per walk, and without even VISITING one.
///
/// `MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN b.city, count(*)` produces fifty
/// rows out of 72M walks, and the enumerating path builds a binding and
/// evaluates the key expression once per walk to get there.
///
/// Two independent savings, and the second is the large one:
///
/// 1. **The key runs once per distinct ENDPOINT, not once per walk** — a tally
///    per endpoint vertex, folded into groups afterwards.
/// 2. **The tally is a DEGREE PRODUCT, so it is O(E) rather than O(walks).** A
///    length-2 walk is an arrival at a midpoint followed by a departure from it,
///    so the number of walks ending at `c` is `Σ_m arrivals(m) · [edges m→c]`.
///    Adding 1 per pair computes that same number `degree` times over; 8M edge
///    visits do what 72M walk visits did.
///
/// ```text
///                     enumerating      here      main (pre-IR)
/// varlen_group          11453.84ms   295.23ms       430.14ms
/// trav2_group            1457.27      185.30        413.68
/// ```
///
/// **Group order is first-seen, the engine pins it, and the degree product does
/// not know it** — it sums per midpoint in vertex order, which says nothing
/// about which row came first. That is the constraint that blocked this shape
/// for a long time, and the way through is to stop asking the counting pass for
/// it:
///
/// - The count pass yields the group SET and its totals, in any order.
/// - A second pass then walks in TRUE enumeration order — depth-first per seed,
///   the order the matcher emits in — recording each group the first time it
///   appears, and **stops as soon as it has seen them all**. That bound exists
///   only because the first pass already counted the groups, and it is what
///   makes the second pass nearly free: fifty groups turn up in the first
///   handful of walks. The worst case, a group whose only row is the very last
///   walk, degrades to the enumeration this replaces and no further.
///
/// `grouped_walk_count_emits_groups_in_first_seen_order` builds a fixture where
/// vertex order and first-seen order disagree, which is the failure this would
/// otherwise ship: a wrong answer that looks entirely reasonable.
///
/// Depth stops at two hops, for the same reason the scalar shortcut does — past
/// that the trail correction stops being a subtraction (see `try_walk_count`).
/// Here that correction is applied distributed: the product counts a self-loop
/// at a seed once as a length-2 walk that reuses its edge, so it comes back off.
pub(super) fn try_grouped_walk_count(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    if !walk_count_enabled() {
        return None;
    }
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
    if path.segments.is_empty() || path.path_var_slot.is_some() {
        return None;
    }
    if !matches!(path.selector, PathSelector::Walk) {
        return None;
    }

    // Exactly `RETURN <key>, count(*)` — one grouping item and one bare count,
    // in that order, with no paging, ordering, HAVING or DISTINCT on top.
    if !proj.aggregating
        || proj.aggs.len() != 1
        || proj.items.len() != 2
        || !proj.group_by.is_empty()
        || proj.limit.is_some()
        || proj.skip.is_some()
        || proj.having.is_some()
        || proj.distinct
        || !proj.order_by.is_empty()
    {
        return None;
    }
    let agg = &proj.aggs[0];
    if !(agg.star && !agg.distinct && agg.arg.is_none() && matches!(agg.func, AggFn::Count)) {
        return None;
    }
    if proj.items[0].is_agg || !matches!(proj.items[1].expr, CExpr::AggRef(0)) {
        return None;
    }

    // The key must read the path's LAST node and nothing else — that is what
    // makes it computable from an endpoint id alone.
    let end_slot = path.segments.last()?.node.var_slot?;
    if !refs_only_endpoint(&proj.items[0].expr, end_slot) {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    let (hops, quant) = bare_hops(path, &ctx)?;
    // See the doc: two hops is where the trail correction stops being exact.
    let depth_ok = match quant {
        None => hops.len() <= 2,
        Some(q) => hops.len() == 1 && q.max.is_some_and(|m| m <= 2) && q.min <= q.max?,
    };
    if !depth_ok {
        return None;
    }

    let seeds = seek_lower::scan_node(graph, &ctx, &path.start, None, *scope_len, None);
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }

    let loops = crate::seek::SelfLoops::Once;
    let hop_at = |i: usize| -> &(crate::seek::Dir, Option<Vec<u32>>) {
        if quant.is_some() {
            &hops[0]
        } else {
            &hops[i]
        }
    };
    // `{0,n}` includes the ZERO-length walk — every seed is its own endpoint.
    // See `try_walk_count` for why that term cannot be folded into the range.
    let zero_len = quant.is_some_and(|q| q.min == 0);
    if quant.is_some_and(|q| q.max == Some(0)) {
        return None;
    }
    let lengths: Vec<usize> = match quant {
        None => vec![hops.len()],
        Some(q) => (q.min.max(1)..=q.max?).map(|n| n as usize).collect(),
    };
    // A quantified repetition is edge-distinct; separate segments are not.
    let trail = quant.is_some();

    // COUNTS, as a degree product — O(E), not O(walks).
    //
    // A length-2 walk is an arrival at a midpoint followed by a departure from
    // it, and the count of walks ending at `c` is the sum over midpoints of
    // `arrivals(m) · [edges m→c]`. Enumerating the pairs to add 1 each time
    // computes the same number `degree` times over: 72M walks where 8M edge
    // visits suffice.
    let slots = graph.vertex_slots();
    let mut tally = vec![0u64; slots];
    if zero_len {
        for &s in &seeds {
            tally[s as usize] += 1;
        }
    }
    for &len in &lengths {
        let (d1, t1) = hop_at(0);
        if len == 1 {
            for &s in &seeds {
                for a in crate::seek::adj(graph, s, *d1, t1.as_deref().unwrap_or(&[]), loops) {
                    tally[a.nbr as usize] += 1;
                }
            }
            continue;
        }
        // Arrivals per midpoint, then one pass over each midpoint's departures.
        let mut arrivals = vec![0u64; slots];
        for &s in &seeds {
            for a in crate::seek::adj(graph, s, *d1, t1.as_deref().unwrap_or(&[]), loops) {
                arrivals[a.nbr as usize] += 1;
            }
        }
        let (d2, t2) = hop_at(1);
        for (m, &n) in arrivals.iter().enumerate() {
            if n == 0 {
                continue;
            }
            for b in crate::seek::adj(graph, m as u32, *d2, t2.as_deref().unwrap_or(&[]), loops) {
                tally[b.nbr as usize] += n;
            }
        }
        // The trail correction, distributed: the only length-2 walk that reuses
        // an edge takes a SELF-LOOP twice, which the product above counted once
        // for every self-loop sitting at a seed.
        if trail {
            for &s in &seeds {
                for a in crate::seek::adj(graph, s, *d1, t1.as_deref().unwrap_or(&[]), loops) {
                    if a.nbr == s {
                        tally[s as usize] -= 1;
                    }
                }
            }
        }
    }

    // GROUPS. The key runs once per distinct endpoint, and the fold is done
    // twice for two different reasons: once over every touched endpoint to learn
    // the group SET and its counts, and once in walk order to learn the group
    // ORDER. See below for why the second pass is cheap.
    let mut b = Binding(vec![None; (*scope_len).max(end_slot + 1)]);
    let key_of = |b: &mut Binding, v: u32| -> (String, Val) {
        b.set(end_slot, Val::Node(v));
        let val = eval(&Env::new(graph, &ctx, b), &proj.items[0].expr);
        let mut k = String::new();
        super::val_key(&val, &mut k);
        (k, val)
    };
    let mut totals: std::collections::HashMap<String, (Val, u64)> =
        std::collections::HashMap::new();
    for (v, &n) in tally.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let (k, val) = key_of(&mut b, v as u32);
        totals.entry(k).or_insert((val, 0)).1 += n;
    }
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }

    // ORDER is first-seen and the engine pins it, so the groups have to come out
    // in the order their first ROW appeared — which the degree product, summing
    // per midpoint in vertex order, does not know.
    //
    // Walk in enumeration order and stop as soon as every group has been seen.
    // That bound is available only because the pass above already counted them,
    // and it is what makes this cheap: a query with fifty groups finds them in
    // the first handful of walks. The worst case — a group whose only row is the
    // last walk — degrades to the enumeration this replaces, and no further.
    let want = totals.len();
    let mut order: Vec<String> = Vec::with_capacity(want);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut cached: Vec<Option<u32>> = vec![None; slots];
    let note = |b: &mut Binding,
                v: u32,
                order: &mut Vec<String>,
                seen: &mut std::collections::HashSet<String>,
                cached: &mut Vec<Option<u32>>| {
        if cached[v as usize].is_some() {
            return;
        }
        cached[v as usize] = Some(0);
        let (k, _) = key_of(b, v);
        if seen.insert(k.clone()) {
            order.push(k);
        }
    };
    'outer: for &s in &seeds {
        if zero_len {
            note(&mut b, s, &mut order, &mut seen, &mut cached);
            if order.len() == want {
                break 'outer;
            }
        }
        for &len in &lengths {
            let (d1, t1) = hop_at(0);
            for a in crate::seek::adj(graph, s, *d1, t1.as_deref().unwrap_or(&[]), loops) {
                if len == 1 {
                    note(&mut b, a.nbr, &mut order, &mut seen, &mut cached);
                } else {
                    let (d2, t2) = hop_at(1);
                    for c in
                        crate::seek::adj(graph, a.nbr, *d2, t2.as_deref().unwrap_or(&[]), loops)
                    {
                        if trail && c.eidx == a.eidx {
                            continue;
                        }
                        note(&mut b, c.nbr, &mut order, &mut seen, &mut cached);
                    }
                }
                if order.len() == want {
                    break 'outer;
                }
            }
            if order.len() == want {
                break 'outer;
            }
        }
    }
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }

    let keys: Vec<(String, Val, u64)> = order
        .into_iter()
        .map(|k| {
            let (val, n) = totals.remove(&k).expect("every ordered key was counted");
            (k, val, n)
        })
        .collect();

    let mut rs = RowSet::new(proj.out_names.clone());
    for (_, val, n) in keys {
        rs.push_row([val_to_value(graph, &val), Value::Num(n as f64)]);
    }
    Some(Ok(rs))
}

/// A chain of hops as `seek` takes them: `(direction, edge types)`, where the
/// types are `None` for ANY and `Some(&[])` for NONE.
type HopChain = Vec<(crate::seek::Dir, Option<Vec<u32>>)>;

/// The `(direction, edge types)` chain of a path whose hops are all BARE, plus
/// its quantifier if it has one. `None` if anything on the way is filtered — see
/// `try_walk_count`'s doc for why a filter here is one that never runs.
fn bare_hops(path: &CPath, ctx: &Ctx) -> Option<(HopChain, Option<Quantifier>)> {
    let mut hops = Vec::new();
    let mut quant = None;
    for (i, seg) in path.segments.iter().enumerate() {
        if seg.unit.is_some() {
            return None;
        }
        let rel = &seg.rel;
        if rel.var_slot.is_some() || !rel.props.is_empty() || rel.where_.is_some() {
            return None;
        }
        if let Some(q) = rel.quantifier.as_ref() {
            if path.segments.len() != 1 {
                return None;
            }
            quant = Some(*q);
        }
        let node = &seg.node;
        if node.label.is_some() || !node.props.is_empty() || node.where_.is_some() {
            return None;
        }
        if let Some(s) = node.var_slot {
            if Some(s) == path.start.var_slot
                || path.segments[..i]
                    .iter()
                    .any(|prev| prev.node.var_slot == Some(s))
            {
                return None;
            }
        }
        let etypes = match rel.label.as_ref() {
            None => None,
            Some(l) => Some(seek_lower::lower_labels(l, ctx, true)?),
        };
        let dir = match rel.direction {
            Direction::Out => crate::seek::Dir::Out,
            Direction::In => crate::seek::Dir::In,
            Direction::Both => crate::seek::Dir::Both,
        };
        hops.push((dir, etypes));
    }
    Some((hops, quant))
}

/// `MATCH (a)-[:R]->(b), (a)-[:R]->(c) … RETURN count(*)` — a comma join whose
/// patterns all branch off the SAME start, counted as a PRODUCT instead of a
/// cross product enumerated row by row.
///
/// Each branch is independent given the shared start, so for one seed the number
/// of rows is `|b-candidates| · |c-candidates|` — no pair ever has to exist. Over
/// a 1M/8M graph with a filter on each far end, that is a couple of adjacency
/// walks per seed instead of the 8 × 8 pairs they generate.
///
/// This is what `try_count_comma_join` used to do. It comes back because the
/// product is exact HERE and nowhere near obvious in general — the engine builds
/// the full cross product across comma patterns, with no relationship-uniqueness
/// rule between them (`(a)-[:R]->(b), (a)-[:R]->(c)` over a two-edge fan is 4
/// rows, not 2), which is precisely what makes a multiplication right.
///
/// The `WHERE` is split into conjuncts and each is assigned to the branch whose
/// far end it reads. A conjunct that reads TWO branches (`b.age > c.age`) is a
/// correlation between them — the factors stop being independent and the product
/// stops being the answer — so it declines. So does one that reads the start and
/// a far end together, which is the same problem.
pub(super) fn try_count_comma_join(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    if !walk_count_enabled() {
        return None;
    }
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
    if patterns.len() < 2 {
        return None;
    }
    if !proj.aggregating
        || proj.aggs.len() != 1
        || proj.items.len() != 1
        || !proj.group_by.is_empty()
        || proj.limit.is_some()
        || proj.skip.is_some()
        || proj.having.is_some()
        || proj.distinct
        || !proj.order_by.is_empty()
        || !matches!(proj.items[0].expr, CExpr::AggRef(0))
    {
        return None;
    }
    let agg = &proj.aggs[0];
    if !(agg.star && !agg.distinct && agg.arg.is_none() && matches!(agg.func, AggFn::Count)) {
        return None;
    }

    // Every pattern is ONE bare hop off the same start slot, landing on its own
    // slot. The first pattern's start carries the seed constraint; the others
    // must be the same variable and add nothing.
    let start_slot = patterns[0].start.var_slot?;
    let mut branches: Vec<(&CSegment, usize)> = Vec::new();
    for (i, p) in patterns.iter().enumerate() {
        if p.path_var_slot.is_some() || !matches!(p.selector, PathSelector::Walk) {
            return None;
        }
        if p.start.var_slot != Some(start_slot) {
            return None;
        }
        if i > 0
            && (p.start.label.is_some() || !p.start.props.is_empty() || p.start.where_.is_some())
        {
            return None;
        }
        let [seg] = p.segments.as_slice() else {
            return None;
        };
        if seg.unit.is_some() {
            return None;
        }
        let rel = &seg.rel;
        if rel.var_slot.is_some()
            || !rel.props.is_empty()
            || rel.where_.is_some()
            || rel.quantifier.is_some()
        {
            return None;
        }
        let far = seg.node.var_slot?;
        if far == start_slot || branches.iter().any(|(_, s)| *s == far) {
            return None;
        }
        branches.push((seg, far));
    }

    // Split the WHERE and hand each conjunct to the branch it reads.
    let mut per_branch: Vec<Vec<&CExpr>> = vec![Vec::new(); branches.len()];
    if let Some(w) = where_.as_ref() {
        for conj in conjuncts(w) {
            // A conjunct belongs to branch `i` when it says nothing about ANY
            // other slot — `refs_slot` is the one traversal over slot
            // references, and asking it the negative question ("does this read
            // something that is not my far end") is what makes a predicate about
            // two branches decline instead of being silently assigned to one of
            // them. `refs_only_endpoint` cannot answer this: it bottoms out at
            // `Var`/`Prop`/`Lit`, so every COMPARISON reads as false.
            let owner = branches
                .iter()
                .position(|(_, far)| !crate::gql::plan::refs_slot(conj, &|s| s != *far));
            per_branch[owner?].push(conj);
        }
    }

    let ctx = resolve_ctx(graph, plan, params);
    let seeds = seek_lower::scan_node(graph, &ctx, &patterns[0].start, None, *scope_len, None);
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }

    let width = (*scope_len).max(1);
    let mut b = Binding(vec![None; width]);
    let mut total: u64 = 0;
    for &s in &seeds {
        let mut product: u64 = 1;
        for (i, (seg, far)) in branches.iter().enumerate() {
            let node = &seg.node;
            let mut n: u64 = 0;
            for (_e, nbr) in expand(graph, &ctx, s, seg.rel.direction, seg.rel.label.as_ref()) {
                if !matches_label(graph, &ctx, nbr, node.label.as_ref()) {
                    continue;
                }
                b.set(*far, Val::Node(nbr));
                if !satisfies(graph, &ctx, &Val::Node(nbr), &node.props, None, &b) {
                    continue;
                }
                let env = Env::new(graph, &ctx, &b);
                if per_branch[i]
                    .iter()
                    .any(|c| !where_keep(&env, Some(c), None))
                {
                    continue;
                }
                n += 1;
            }
            product = product.saturating_mul(n);
            // A zero factor makes the whole product zero — skip the rest.
            if product == 0 {
                break;
            }
        }
        total += product;
    }
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }

    Some(count_rows_any(proj, total))
}

/// The top-level AND conjuncts of a predicate, flattened.
fn conjuncts(e: &CExpr) -> Vec<&CExpr> {
    match e {
        CExpr::And(parts) => parts.iter().flat_map(|p| conjuncts(p)).collect(),
        other => vec![other],
    }
}

/// `MATCH … WITH <endpoint>, <aggregates> RETURN count(*)` — the number of
/// GROUPS, which does not depend on what the aggregates computed.
///
/// This is the PageRank/CC-shaped gather as a benchmark writes it: group 8M
/// edges by their endpoint, aggregate over the sources, then ask how many
/// groups there were. The grouping is the answer; the aggregate is dead.
///
/// So it is dead-value elimination, with the usual guard — an aggregate may be
/// dropped only if nothing can OBSERVE it, and in this engine an aggregate has
/// exactly one observable effect besides its value: it can FAULT.
///
/// - `count`, `min` and `max` never fault. `min`/`max` fold through `cmp_total`,
///   which is a total order by construction.
/// - `sum` and `avg` fault on a temporal or a list value
///   (`FAULT_NONNUMERIC_AGG`, `FAULT_TEMPORAL_AGG`). They are safe exactly when
///   the argument's stored column cannot hold one — a typed `Num`, `Str` or
///   `Bool` column. A `Mixed` column, or anything else, declines.
///
/// With that established the answer is the number of distinct endpoints, which
/// `walk_count` already computes with a bitmap:
///
/// ```text
/// gather_by_node   145.16ms -> 17.4   (main 35.28)
/// ```
///
/// **What this does NOT speed up, and it is worth being plain about it:** the
/// shape where the aggregate is actually READ — `WITH n, sum(m.age) AS s RETURN
/// sum(s)` — still materializes a row per edge, and that is the shape the
/// benchmark's name is really about. This makes the benchmark fast by not doing
/// arithmetic nobody asked for, which is a real and general optimization but a
/// narrower one than the row it lands on suggests. A streaming grouped fold,
/// which would fix both, is the next thing here.
pub(super) fn try_count_groups(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    if !walk_count_enabled() {
        return None;
    }
    let [CClause::Match {
        optional: false,
        patterns,
        where_: None,
        scope_len,
        ..
    }, CClause::With {
        projection: with,
        where_: None,
        ..
    }, CClause::Return(out)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    if path.segments.is_empty() || path.path_var_slot.is_some() {
        return None;
    }
    if !matches!(path.selector, PathSelector::Walk) {
        return None;
    }

    // The outer projection is exactly `count(*)`.
    if !out.aggregating
        || out.aggs.len() != 1
        || out.items.len() != 1
        || !out.group_by.is_empty()
        || out.limit.is_some()
        || out.skip.is_some()
        || out.having.is_some()
        || out.distinct
        || !out.order_by.is_empty()
        || !matches!(out.items[0].expr, CExpr::AggRef(0))
    {
        return None;
    }
    let outer = &out.aggs[0];
    if !(outer.star && !outer.distinct && outer.arg.is_none() && matches!(outer.func, AggFn::Count))
    {
        return None;
    }

    // The WITH groups by the path's endpoint and nothing else, with no paging,
    // ordering, HAVING or DISTINCT of its own — each of which would change how
    // many groups survive.
    if !with.aggregating
        || !with.group_by.is_empty()
        || with.limit.is_some()
        || with.skip.is_some()
        || with.having.is_some()
        || with.distinct
        || !with.order_by.is_empty()
    {
        return None;
    }
    let end_slot = path.segments.last()?.node.var_slot?;
    let mut keys = with.items.iter().filter(|i| !i.is_agg);
    let key = keys.next()?;
    if keys.next().is_some() || !matches!(key.expr, CExpr::Var(s) if s == end_slot) {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    for agg in &with.aggs {
        if !agg_cannot_fault(graph, &ctx, agg) {
            return None;
        }
    }

    let (hops, quant) = bare_hops(path, &ctx)?;
    // A quantified pattern reaches the same endpoint at several lengths, and the
    // distinct count would have to union them rather than sum.
    if quant.is_some() {
        return None;
    }

    let seeds = seek_lower::scan_node(graph, &ctx, &path.start, None, *scope_len, None);
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }
    let n = crate::seek::walk_count(graph, &seeds, &hops, crate::seek::SelfLoops::Once, true);

    Some(count_rows_any(out, n as u64))
}

/// Whether an aggregate can be dropped without losing an error — see
/// [`try_count_groups`]. Conservative: an argument this cannot type declines.
fn agg_cannot_fault(graph: &Graph, ctx: &Ctx, agg: &CAgg) -> bool {
    match agg.func {
        // `count` counts; `min`/`max` fold through `cmp_total`, a total order.
        AggFn::Count | AggFn::Min | AggFn::Max => true,
        // `sum`/`avg` fault on a temporal or a list, so they are safe exactly
        // when the stored column cannot hold one.
        AggFn::Sum | AggFn::Avg => {
            let Some(CExpr::Prop { key_ref, .. }) = agg.arg.as_ref() else {
                return false;
            };
            let Some(kid) = ctx.prop_keys[*key_ref].0 else {
                // The key does not exist on any vertex, so every value is absent
                // and the fold sees nothing to fault on.
                return true;
            };
            matches!(
                graph.props.cols.get(kid as usize),
                Some(Column::Num { .. } | Column::Str { .. } | Column::Bool { .. })
            )
        }
        _ => false,
    }
}

/// `MATCH (a:L) WHERE [NOT] EXISTS { (a)-[:R]->(far) } RETURN count(*)` — count
/// the mask, do not build rows to count them.
///
/// The backward sweep already answers this as a per-vertex bitmap (see
/// `semi_join_back`), and the columnar path then spends three passes over a
/// million rows turning that bitmap into a number: a `Col::Bool` of a million
/// bools, a filter, and a count. Testing the bit while walking the seeds is one
/// pass and no allocation.
///
/// ```text
/// not_exists_hub   2.91ms -> 0.30   (main 0.25)
/// exists_semi      1.11   -> 0.13   (main 0.25)
/// ```
///
/// `NOT` is a negation of the same mask rather than its own path — the sweep
/// computes "reaches", and the two polarities differ only in which answer is
/// kept, which is also why neither can be right if the other is wrong.
pub(super) fn try_count_semi_join(
    linear: &CLinear,
    graph: &Graph,
    plan: &CQuery,
    params: &[Val],
) -> Option<CodeResult<RowSet>> {
    if !walk_count_enabled() {
        return None;
    }
    let [CClause::Match {
        optional: false,
        patterns,
        where_: Some(pred),
        scope_len,
        ..
    }, CClause::Return(proj)] = linear.clauses.as_slice()
    else {
        return None;
    };
    let [path] = patterns.as_slice() else {
        return None;
    };
    // A BARE node: the pattern contributes no rows of its own, so the count is
    // exactly how many seeds pass the predicate.
    if !path.segments.is_empty() || path.path_var_slot.is_some() {
        return None;
    }
    if !proj.aggregating
        || proj.aggs.len() != 1
        || proj.items.len() != 1
        || !proj.group_by.is_empty()
        || proj.limit.is_some()
        || proj.skip.is_some()
        || proj.having.is_some()
        || proj.distinct
        || !proj.order_by.is_empty()
        || !matches!(proj.items[0].expr, CExpr::AggRef(0))
    {
        return None;
    }
    let agg = &proj.aggs[0];
    if !(agg.star && !agg.distinct && agg.arg.is_none() && matches!(agg.func, AggFn::Count)) {
        return None;
    }

    // `EXISTS { … }` or `NOT EXISTS { … }`, and nothing else — any further
    // conjunct is a predicate the mask does not carry.
    let (want, inner) = match pred {
        CExpr::Exists { patterns, .. } => (true, patterns),
        CExpr::Not(e) => match e.as_ref() {
            CExpr::Exists { patterns, .. } => (false, patterns),
            _ => return None,
        },
        _ => return None,
    };
    let [inner_path] = inner.as_slice() else {
        return None;
    };
    if inner_path.path_var_slot.is_some() || !matches!(inner_path.selector, PathSelector::Walk) {
        return None;
    }
    let [seg] = inner_path.segments.as_slice() else {
        return None;
    };
    if seg.unit.is_some() || seg.rel.quantifier.is_some() {
        return None;
    }
    // The subquery's root must be the outer variable itself, unconstrained —
    // the same precondition `exists_semi_join_vec` checks before it reads the
    // frame's element column.
    let start = &inner_path.start;
    if start.var_slot != path.start.var_slot
        || start.label.is_some()
        || !start.props.is_empty()
        || start.where_.is_some()
    {
        return None;
    }
    let rel = &seg.rel;
    if rel.var_slot.is_some() || !rel.props.is_empty() || rel.where_.is_some() {
        return None;
    }
    if seg.node.where_.is_some() {
        return None;
    }

    let ctx = resolve_ctx(graph, plan, params);
    // A bare label seeds from its BUCKET, borrowed — `scan_node` would copy a
    // million ids into a `Vec` to then read each one once. Anything richer (an
    // inline constraint, a WHERE, an index seek) goes through `scan_node`, which
    // is where that logic belongs.
    let bare_label = path.start.props.is_empty() && path.start.where_.is_none();
    let bucket = if bare_label {
        path.start
            .label
            .as_ref()
            .and_then(seed_label)
            .and_then(|r| ctx.labels[r].0)
            .map(|lid| graph.vertices_with_label(lid))
    } else {
        None
    };
    let owned;
    let seeds: &[u32] = match bucket {
        Some(b) => b,
        None => {
            owned = seek_lower::scan_node(graph, &ctx, &path.start, None, *scope_len, None);
            &owned
        }
    };
    if let Err(e) = ctx.check_fault() {
        return Some(Err(e));
    }
    // Declines unless the far end is narrow enough to sweep — the same cost
    // model, so this arm never makes a query slower than the columnar one.
    let reached = semi_join_back(graph, &ctx, seg, seeds.len())?;
    let n = seeds
        .iter()
        .filter(|&&v| reached[v as usize] == want)
        .count();

    Some(count_rows_any(proj, n as u64))
}
