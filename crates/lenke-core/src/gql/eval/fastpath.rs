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
