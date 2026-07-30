//! The match-execution pipeline: drive a MATCH over the columnar adjacency
//! (any_match/drive_matches/match_path), fold aggregates over the matched rows
//! (Agg/step_aggs/passes_having), and project them to output rows
//! (ProjAccum/project_matches). Extracted from the evaluator (`super`); shares its
//! context/helpers via `use super::*`.
use super::*;

pub(super) fn any_match(
    graph: &Graph,
    ctx: &Ctx,
    patterns: &[CPath],
    where_: Option<&CExpr>,
    binding: &Binding,
    sub_len: usize,
) -> bool {
    // The unified meet-in-the-middle first (bound-both-endpoints existence, k ≥ 1); then the
    // single-segment forward search for the cases it declines (unbound end, inner WHERE).
    if let Some(res) = exists_bidir(graph, ctx, patterns, where_, binding) {
        return res;
    }
    if let Some(res) = any_match_reachable(graph, ctx, patterns, where_, binding, sub_len) {
        return res;
    }
    let mut found = false;
    let mut work = binding.clone();
    work.resize(sub_len);
    visit_patterns(graph, ctx, patterns, 0, where_, &mut work, &mut |_| {
        found = true;
        false
    });
    found
}

/// Count matches of the (correlated) sub-pattern.
pub(super) fn count_matches(
    graph: &Graph,
    ctx: &Ctx,
    patterns: &[CPath],
    where_: Option<&CExpr>,
    binding: &Binding,
    sub_len: usize,
) -> u64 {
    // Degree fast path: `COUNT { (n)-[:T]->(m) }` with `n` already bound and a
    // single plain directed segment (no quantifier, no inline props/WHERE, no inner
    // WHERE, a fresh endpoint) is just `n`'s matching adjacency count — skip the
    // per-call binding clone and the recursive matcher. `COUNT { … }` in a `SET` /
    // `RETURN` runs once per outer row, so this turns an O(rows·degree) enumeration
    // (with a clone each) into a tight adjacency scan.
    if where_.is_none() {
        if let [path] = patterns {
            if let [seg] = path.segments.as_slice() {
                let plain = |n: &CNode| n.props.is_empty() && n.where_.is_none();
                let ok_rel = seg.rel.quantifier.is_none()
                    && seg.rel.props.is_empty()
                    && seg.rel.where_.is_none()
                    && matches!(seg.rel.direction, Direction::Out | Direction::In);
                // The live `Node` a pattern node is already bound to (a fresh sub-scope
                // slot sits beyond the outer binding, so guard the index).
                let bound_of = |n: &CNode| -> Option<u32> {
                    n.var_slot
                        .filter(|&s| s < binding.0.len())
                        .and_then(|s| binding.get(s))
                        .and_then(|v| match v {
                            Val::Node(i) => Some(*i),
                            _ => None,
                        })
                };
                if ok_rel && plain(&path.start) && plain(&seg.node) {
                    // Anchor at whichever end is the bound correlated vertex; the other
                    // (free) end supplies a label filter. Its matching-adjacency count
                    // is the (reverse-)degree — no per-row clone, no recursion.
                    let cnt = |anchor: u32, dir: Direction, far: Option<&CLabelExpr>| -> u64 {
                        expand(graph, ctx, anchor, dir, seg.rel.label.as_ref())
                            .filter(|(_e, nbr)| matches_label(graph, ctx, *nbr, far))
                            .count() as u64
                    };
                    match (bound_of(&path.start), bound_of(&seg.node)) {
                        // `(a)-[:T]{dir}-(m)`, `a` bound → a's `dir` adjacency to m's label.
                        (Some(a), None) => {
                            return cnt(a, seg.rel.direction, seg.node.label.as_ref())
                        }
                        // `(m)-[:T]{dir}-(b)`, `b` bound → b's *reverse*-side adjacency to
                        // m's (start's) label: the reverse degree (e.g. `COUNT { (:U)->(b) }`).
                        (None, Some(b)) => {
                            return cnt(
                                b,
                                flip_direction(seg.rel.direction),
                                path.start.label.as_ref(),
                            );
                        }
                        _ => {} // both bound (specific edge) / both free (global) → enumerate
                    }
                }
            }
        }
    }

    let mut count = 0u64;
    let mut work = binding.clone();
    work.resize(sub_len);
    visit_patterns(graph, ctx, patterns, 0, where_, &mut work, &mut |_| {
        count += 1;
        true
    });
    count
}

/// The compiled `VALUE { … }` subquery, bundled to keep `value_subquery` under the
/// arg limit: the correlated pattern(s), the optional inner WHERE, the RETURN
/// expression, whether that RETURN aggregates, and the subquery's binding width.
pub(super) struct SubqueryPlan<'a> {
    pub patterns: &'a [CPath],
    pub where_: Option<&'a CExpr>,
    pub ret: &'a CExpr,
    pub is_agg: bool,
    pub sub_len: usize,
}

/// Evaluate a `VALUE { … RETURN <expr> }` scalar subquery: a single value.
///
/// Collect every correlated match (read-only, via `visit_patterns`), then:
/// - an aggregate RETURN folds the whole group to one value (0 rows → the
///   aggregate's empty answer, e.g. `count` → 0, `sum` → NULL);
/// - a non-aggregate RETURN yields NULL for 0 rows, the value for exactly one
///   row, and a **cardinality fault** for more than one (ISO: a scalar subquery
///   must not deliver more than one row) — loud, never a silent first-of-many.
pub(super) fn value_subquery(
    graph: &Graph,
    ctx: &Ctx,
    plan: SubqueryPlan,
    binding: &Binding,
) -> Val {
    let SubqueryPlan {
        patterns,
        where_,
        ret,
        is_agg,
        sub_len,
    } = plan;
    let mut work = binding.clone();
    work.resize(sub_len);
    let mut matches: Vec<Binding> = Vec::new();
    visit_patterns(graph, ctx, patterns, 0, where_, &mut work, &mut |b| {
        matches.push(b.clone());
        // A non-aggregate scalar subquery is over the moment a second row appears
        // (it's already a cardinality error); an aggregate needs the full group.
        is_agg || matches.len() < 2
    });

    if is_agg {
        // Fold over the group. The tree-walk `CExpr::Aggregate` arm reads
        // `env.group`; a plain sub-expression around it reads the first match.
        let base = matches.first().cloned().unwrap_or_else(|| {
            let mut b = binding.clone();
            b.resize(sub_len);
            b
        });
        let mut env = Env::new(graph, ctx, &base);
        env.group = Some(&matches);
        return eval(&env, ret);
    }

    match matches.as_slice() {
        [] => Val::Null,
        [b] => eval(&Env::new(graph, ctx, b), ret),
        _ => {
            ctx.set_fault(FAULT_CARDINALITY);
            Val::Null
        }
    }
}

/// Slots a pattern set introduces (for OPTIONAL MATCH null-binding).
pub(super) fn pattern_slots(patterns: &[CPath]) -> Vec<usize> {
    let mut slots = Vec::new();
    let mut push = |s: Option<usize>| {
        if let Some(s) = s {
            slots.push(s);
        }
    };
    for p in patterns {
        push(p.path_var_slot);
        push(p.start.var_slot);
        for CSegment { rel, node, .. } in &p.segments {
            push(rel.var_slot);
            push(node.var_slot);
        }
    }
    slots
}

/// Stream every binding produced by a chain of MATCH clauses (extending `binding`
/// in place, backtracking) into `sink`. No intermediate `Vec<Binding>`: matches
/// nest directly into the consumer. Returns `false` to propagate a stop request.
pub(super) fn drive_matches(
    graph: &Graph,
    ctx: &Ctx,
    matches: &[&CClause],
    idx: usize,
    binding: &mut Binding,
    sink: &mut dyn FnMut(&Binding) -> bool,
) -> bool {
    let Some(clause) = matches.get(idx) else {
        return sink(binding);
    };
    let CClause::Match {
        optional,
        patterns,
        where_,
        scope_len,
        ..
    } = clause
    else {
        return true; // only MATCH clauses are streamed
    };
    binding.resize(*scope_len);
    let mut matched = false;
    let cont = visit_patterns(
        graph,
        ctx,
        patterns,
        0,
        where_.as_ref(),
        binding,
        &mut |b| {
            matched = true;
            drive_matches(graph, ctx, matches, idx + 1, b, sink)
        },
    );
    if !cont {
        return false;
    }
    if !matched && *optional {
        // OPTIONAL with no match: null-fill this clause's slots and continue —
        // then UNDO the fill, exactly as a successful match backtracks its
        // bindings. Without this, the stale nulls leak into the NEXT outer
        // binding, where `bind_slot` mistakes them for a join conflict and silently
        // drops that row's real matches.
        let mut filled = Vec::new();
        for s in pattern_slots(patterns) {
            if !binding.bound(s) {
                binding.set(s, Val::Null);
                filled.push(s);
            }
        }
        let keep = drive_matches(graph, ctx, matches, idx + 1, binding, sink);
        for s in filled {
            binding.unset(s);
        }
        return keep;
    }
    true
}

// --- specialized single-path matcher (monomorphized, no per-segment dyn) -----
//
// The general matcher above passes `&mut dyn FnMut` down each segment, so a
// K-segment path does K dynamic calls per match. This generic variant inlines
// node/edge matching and recurses with the *same* `&mut F`, so it monomorphizes
// once per concrete sink and the per-edge hot loop has no dynamic dispatch — the
// dyn boundary collapses to a single call per emitted match. Used for the common
// shape: one MATCH clause, one path (quantifiers fine).

/// Match `node` at vertex `vi`; on success continue matching `path` from segment
/// `next_idx`. Restores the binding on backtrack. Generic over the sink `F`.
#[allow(
    clippy::too_many_arguments,
    reason = "recursive backtracking matcher; bundling its args into a struct would obscure the hot recursion"
)]
/// Bind `vi` to `node` and continue into `path[next_idx..]` — the specialized
/// (next-segment) continuation of the shared node-binder [`match_node_then`].
pub(super) fn match_node_continue<F: FnMut(&mut Binding) -> bool + ?Sized>(
    graph: &Graph,
    ctx: &Ctx,
    binding: &mut Binding,
    node: &CNode,
    vi: u32,
    path: &CPath,
    next_idx: usize,
    emit: &mut F,
) -> bool {
    match_node_then(graph, ctx, binding, node, vi, &mut |b| {
        match_path(graph, ctx, path, next_idx, vi, b, emit)
    })
}

/// Walk segments `idx..` of `path` from `from`, emitting each complete binding. Generic
/// over the emit sink AND `?Sized`, so it is the ONE segment-walker for both the hot
/// top-level caller (a concrete `F` → monomorphized, static dispatch) and the subquery
/// caller (`F = dyn FnMut` → one instantiation, dynamic dispatch) — no duplicated body.
pub(super) fn match_path<F: FnMut(&mut Binding) -> bool + ?Sized>(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    idx: usize,
    from: u32,
    binding: &mut Binding,
    emit: &mut F,
) -> bool {
    if idx >= path.segments.len() {
        return emit(binding);
    }
    let CSegment { rel, node, unit } = &path.segments[idx];
    if let Some(q) = rel.quantifier {
        // A parenthesized SUBPATH repeats a unit
        // and exposes its group variables at each trail end; the abbreviated form is
        // the plain single-edge walk. Same `on_end` contract.
        let sink =
            &mut |b: &mut Binding, end: u32, verts: &[u32], edges: &[u32], steps: &[StepRec]| {
                let restores = unit
                    .as_ref()
                    .filter(|u| u.exposes())
                    .map(|u| {
                        if u.is_flat() {
                            bind_group_vars_flat(b, u, verts, edges)
                        } else {
                            bind_group_vars(b, u, steps)
                        }
                    })
                    .unwrap_or_default();
                let keep = match_node_continue(graph, ctx, b, node, end, path, idx + 1, emit);
                for (s, prev) in restores.into_iter().rev() {
                    match prev {
                        Some(v) => b.set(s, v),
                        None => b.unset(s),
                    }
                }
                keep
            };
        let spec = WalkSpec {
            q,
            mode: path.mode,
            want_path: unit.as_ref().is_some_and(|u| u.exposes()),
        };
        return match unit {
            Some(u) => reachable_each_unit(graph, ctx, binding, from, u, spec, sink),
            None => reachable_each(graph, ctx, binding, from, rel, spec, sink),
        };
    }
    for (eidx, nbr) in expand(graph, ctx, from, rel.direction, rel.label.as_ref()) {
        let Some(eset) = bind_slot(binding, rel.var_slot, &Val::Edge(eidx)) else {
            continue;
        };
        let keep = if satisfies(
            graph,
            ctx,
            &Val::Edge(eidx),
            &rel.props,
            rel.where_.as_ref(),
            binding,
        ) {
            match_node_continue(graph, ctx, binding, node, nbr, path, idx + 1, emit)
        } else {
            true
        };
        if eset {
            binding.unset(rel.var_slot.unwrap());
        }
        if !keep {
            return false;
        }
    }
    true
}

/// Seed and match a single path, emitting each complete binding via `emit`.
pub(super) fn match_one_path<F: FnMut(&mut Binding) -> bool>(
    graph: &Graph,
    ctx: &Ctx,
    path: &CPath,
    binding: &mut Binding,
    emit: &mut F,
) -> bool {
    match path.start.var_slot {
        Some(sl) if binding.bound(sl) => match binding.get(sl) {
            Some(Val::Node(i)) => {
                match_node_continue(graph, ctx, binding, &path.start, *i, path, 0, emit)
            }
            _ => true,
        },
        _ => match path.start.label.as_ref().and_then(seed_label) {
            Some(r) => match ctx.labels[r].0 {
                Some(lid) => {
                    let seeds = graph.vertices_with_label(lid);
                    for &s in seeds {
                        if !match_node_continue(graph, ctx, binding, &path.start, s, path, 0, emit)
                        {
                            return false;
                        }
                    }
                    true
                }
                None => true,
            },
            None => {
                for s in graph.vertex_indices() {
                    if !match_node_continue(graph, ctx, binding, &path.start, s, path, 0, emit) {
                        return false;
                    }
                }
                true
            }
        },
    }
}

/// Recognize the common shape a single MATCH clause + single path so the
/// monomorphized matcher can drive it directly (returns path, clause WHERE, and
/// the binding slot count to size the working binding).
type SimpleWhere<'a> = (&'a CPath, Option<&'a CExpr>, Option<&'a Program>, usize);
pub(super) fn single_simple_clause<'a>(matches: &[&'a CClause]) -> Option<SimpleWhere<'a>> {
    if matches.len() != 1 {
        return None;
    }
    match matches[0] {
        CClause::Match {
            optional: false,
            patterns,
            where_,
            where_prog,
            scope_len,
            // A path selector (`ANY SHORTEST`) or a bound path variable needs the
            // general `visit_pattern` driver (which knows `shortest_walk`/`all_walk`
            // and builds the Path value); `match_one_path` only yields endpoints, so
            // decline the fast path for those.
        } if patterns.len() == 1
            && patterns[0].selector == PathSelector::Walk
            && patterns[0].path_var_slot.is_none() =>
        {
            Some((
                &patterns[0],
                where_.as_ref(),
                where_prog.as_ref(),
                *scope_len,
            ))
        }
        _ => None,
    }
}

/// Evaluate a fast-path clause WHERE (`true` = keep the row), per [`USE_VM`].
#[inline]
pub(super) fn where_keep(env: &Env, cw: Option<&CExpr>, cwp: Option<&Program>) -> bool {
    if USE_VM {
        cwp.is_none_or(|w| as_truth(&run(env, w)) == Some(true))
    } else {
        cw.is_none_or(|w| as_truth(&eval(env, w)) == Some(true))
    }
}

/// An aggregate's running state, folded one value at a time (no stored group).
pub(super) struct Agg {
    func: AggFn,
    star: bool,
    distinct: bool,
    n: u64,
    sum: f64,
    /// `sum()` only: a non-DURATION scalar (number/bool) has contributed to `sum`.
    /// Paired with `tsum`, it detects a number-and-DURATION mix, which is unsummable
    /// and faults (matches TS and the scalar `temporal_values_sum`) rather than
    /// silently returning just the duration.
    saw_num: bool,
    /// Running DURATION sum for `sum()` over a temporal column (`None` until the
    /// first duration; keeps `sum` for the numeric path).
    tsum: Option<crate::temporal::Duration>,
    /// A pending temporal-aggregate fault (`avg`/non-summable-kind → unsupported;
    /// duration overflow), surfaced via `ctx` by [`step_aggs`].
    tfault: Option<u8>,
    /// Running Σx² for the one-pass `stddev_pop` / `stddev_samp`; 0 for others.
    sum_sq: f64,
    extreme: Option<Val>,
    list: Vec<Val>,
    seen: FxHashSet<String>,
    /// DISTINCT fast path for element values: a node/edge is identified by its
    /// dense id, so dedup by a tagged `u64` (no per-value string key). Scalars fall
    /// back to `seen`.
    seen_ids: FxHashSet<u64>,
    /// Percentile fraction (clamped `[0, 1]`); unused by other aggregates.
    frac: f64,
}

/// Tag bit distinguishing an edge id from a node id in [`Agg::seen_ids`] (dense ids
/// are `u32`, so the tag never collides with the value).
pub(super) const EDGE_ID_TAG: u64 = 1 << 32;

impl Agg {
    fn new(spec: &crate::gql::plan::CAgg) -> Self {
        Self {
            func: spec.func,
            star: spec.star,
            distinct: spec.distinct,
            n: 0,
            sum: 0.0,
            saw_num: false,
            tsum: None,
            tfault: None,
            sum_sq: 0.0,
            extreme: None,
            list: Vec::new(),
            seen: FxHashSet::default(),
            seen_ids: FxHashSet::default(),
            frac: spec.frac.unwrap_or(0.0),
        }
    }
    fn step(&mut self, value: Option<Val>) {
        if self.func == AggFn::Count && self.star {
            self.n += 1; // count(*) counts rows
            return;
        }
        let Some(val) = value else { return };
        if is_nullish(&val) {
            return;
        }
        if self.distinct {
            // Element values dedup by dense id (no string key); scalars by `val_key`.
            let novel = match &val {
                Val::Node(i) => self.seen_ids.insert(*i as u64),
                Val::Edge(i) => self.seen_ids.insert(*i as u64 | EDGE_ID_TAG),
                _ => {
                    let mut k = String::new();
                    val_key(&val, &mut k);
                    self.seen.insert(k)
                }
            };
            if !novel {
                return;
            }
        }
        match self.func {
            AggFn::Count => self.n += 1,
            // `sum` over DURATIONs folds component-wise (like `dur + dur`); over a
            // non-summable temporal kind it faults. `avg` over any temporal faults.
            AggFn::Sum if matches!(val, Val::Temporal(_)) => {
                if let Val::Temporal(crate::temporal::Temporal::Duration(d)) = &val {
                    self.tsum = Some(match self.tsum {
                        None => *d,
                        Some(a) => match a.add(d) {
                            Some(s) => s,
                            None => {
                                self.tfault = Some(FAULT_DURATION_OVERFLOW);
                                a
                            }
                        },
                    });
                    // A DURATION summed alongside a plain number is a type mix — not
                    // summable. Fault (matches TS) instead of dropping the numeric.
                    if self.saw_num {
                        self.tfault.get_or_insert(FAULT_TEMPORAL_AGG);
                    }
                } else {
                    self.tfault = Some(FAULT_TEMPORAL_AGG);
                }
            }
            AggFn::Avg if matches!(val, Val::Temporal(_)) => {
                self.tfault = Some(FAULT_TEMPORAL_AGG);
            }
            // A list (or other non-scalar) isn't summable — fault loud rather than
            // silently NaN → null, matching the temporal rule (and the TS twin).
            AggFn::Sum | AggFn::Avg if matches!(val, Val::List(_)) => {
                self.tfault = Some(FAULT_NONNUMERIC_AGG);
            }
            AggFn::Sum => {
                self.saw_num = true;
                self.sum += num_of(&val).unwrap_or(f64::NAN);
                // A number after a DURATION already accumulated is the same type mix.
                if self.tsum.is_some() {
                    self.tfault.get_or_insert(FAULT_TEMPORAL_AGG);
                }
            }
            AggFn::Avg => {
                self.sum += num_of(&val).unwrap_or(f64::NAN);
                self.n += 1;
            }
            AggFn::StddevPop | AggFn::StddevSamp => {
                let x = num_of(&val).unwrap_or(f64::NAN);
                self.sum += x;
                self.sum_sq += x * x;
                self.n += 1;
            }
            AggFn::Min => {
                if self
                    .extreme
                    .as_ref()
                    .is_none_or(|m| cmp_total(&val, m) == Ordering::Less)
                {
                    self.extreme = Some(val);
                }
            }
            AggFn::Max => {
                if self
                    .extreme
                    .as_ref()
                    .is_none_or(|m| cmp_total(&val, m) == Ordering::Greater)
                {
                    self.extreme = Some(val);
                }
            }
            AggFn::CollectList | AggFn::PercentileCont | AggFn::PercentileDisc => {
                self.list.push(val)
            }
        }
    }
    fn finish(self) -> Val {
        match self.func {
            AggFn::Count => Val::Num(self.n as f64),
            AggFn::Sum => match self.tsum {
                Some(d) => Val::Temporal(crate::temporal::Temporal::Duration(d)),
                None => Val::Num(self.sum),
            },
            AggFn::Avg => {
                if self.n == 0 {
                    Val::Null
                } else {
                    Val::Num(self.sum / self.n as f64)
                }
            }
            AggFn::Min | AggFn::Max => self.extreme.unwrap_or(Val::Null),
            AggFn::CollectList => Val::List(self.list),
            AggFn::PercentileCont => percentile(&self.list, self.frac, true),
            AggFn::PercentileDisc => percentile(&self.list, self.frac, false),
            AggFn::StddevPop => stddev_of(self.n, self.sum, self.sum_sq, false),
            AggFn::StddevSamp => stddev_of(self.n, self.sum, self.sum_sq, true),
        }
    }

    /// Fold `other`'s partial into `self` — the reduce step for parallel
    /// aggregation. `other` must be the same `func` (and non-DISTINCT: distinct
    /// aggregates can't merge from `(sum, seen)` alone, so they stay serial). Only
    /// the fields the func uses are non-default, so the unconditional `n`/`sum`/
    /// `list` merges are correct; `Min`/`Max` take the better extreme. Because a
    /// group's members share their group-key values, keeping either representative
    /// binding is equivalent, so only the fold state needs merging. Merging chunks
    /// in seed order reproduces the serial first-seen order exactly.
    #[cfg(feature = "parallel-query")]
    fn merge(&mut self, other: Self) {
        self.n += other.n;
        self.sum += other.sum;
        self.saw_num |= other.saw_num;
        self.sum_sq += other.sum_sq;
        self.list.extend(other.list);
        // DURATION sum folds across partials (same `Duration::add`); a fault in
        // either partial wins.
        self.tfault = self.tfault.or(other.tfault);
        if let Some(o) = other.tsum {
            self.tsum = Some(match self.tsum {
                None => o,
                Some(a) => match a.add(&o) {
                    Some(s) => s,
                    None => {
                        self.tfault = self.tfault.or(Some(FAULT_DURATION_OVERFLOW));
                        a
                    }
                },
            });
        }
        // A numeric-only partial merged with a DURATION-only partial is the same
        // number-and-DURATION mix a serial fold would have faulted on.
        if self.saw_num && self.tsum.is_some() {
            self.tfault = self.tfault.or(Some(FAULT_TEMPORAL_AGG));
        }
        if let Some(o) = other.extreme {
            let take = match self.func {
                AggFn::Min => self
                    .extreme
                    .as_ref()
                    .is_none_or(|m| cmp_total(&o, m) == Ordering::Less),
                AggFn::Max => self
                    .extreme
                    .as_ref()
                    .is_none_or(|m| cmp_total(&o, m) == Ordering::Greater),
                _ => false,
            };
            if take {
                self.extreme = Some(o);
            }
        }
    }
}

/// Fold one input binding into a group's aggregate states (one per extracted
/// aggregate), evaluating each aggregate's argument against the binding.
pub(super) fn step_aggs(
    aggs: &mut [Agg],
    specs: &[crate::gql::plan::CAgg],
    graph: &Graph,
    ctx: &Ctx,
    binding: &Binding,
) {
    for (agg, spec) in aggs.iter_mut().zip(specs) {
        let v = spec
            .arg
            .as_ref()
            .map(|a| eval(&Env::new(graph, ctx, binding), a));
        agg.step(v);
        // Surface a temporal-aggregate fault (avg/non-summable → unsupported;
        // duration overflow) to the row boundary. `set_fault` is first-wins.
        if let Some(f) = agg.tfault {
            ctx.set_fault(f);
        }
    }
}

/// ISO `HAVING`: does a group survive its post-aggregation predicate? Evaluated
/// with the group's representative binding (its group keys / input vars) and the
/// folded `agg_values`. Three-valued — only TRUE keeps the group. `None` HAVING
/// (the `RETURN`/`WITH` case) always passes.
pub(super) fn passes_having(
    proj: &CProjection,
    graph: &Graph,
    ctx: &Ctx,
    rep: &Binding,
    agg_values: &[Val],
) -> bool {
    let Some(cond) = proj.having.as_ref() else {
        return true;
    };
    let env = Env {
        graph,
        ctx,
        binding: rep,
        group: None,
        agg_values: Some(agg_values),
    };
    as_truth(&eval(&env, cond)) == Some(true)
}

/// A streaming projection: accepts bindings one at a time (folding aggregates
/// incrementally; never storing the full input), then `finish`es to result rows.
pub(super) struct ProjAccum<'p> {
    proj: &'p CProjection,
    /// Whether grouping keys exist (some non-aggregate item). When false but
    /// aggregating, it's a single global group (no map, no key string).
    grouped: bool,
    /// Top-k mode: `ORDER BY … LIMIT n` whose keys don't reference output, so we
    /// keep only the top-k *input* bindings (sort keys computed without
    /// projecting) and project just those at finish. `cap` = skip+limit.
    topk: bool,
    cap: usize,
    /// Top-k: the worst (largest) kept sort key once at capacity — a new row not
    /// better than this can't make the top-k, so it's skipped without cloning.
    threshold: Option<Vec<Val>>,
    /// Reused scratch binding for computing a top-k sort key (no per-row alloc).
    sort_scratch: Binding,
    /// Non-aggregating: projected rows (+ ORDER BY keys); in top-k mode, instead
    /// the kept *input* bindings (+ keys) until `finish` projects them.
    rows: Vec<(Binding, Vec<Val>)>,
    /// Global aggregate (no group keys): one running accumulator set.
    global: Option<(Binding, Vec<Agg>)>,
    /// Grouped aggregate: groups in first-seen order (the `Vec` *is* the order —
    /// no separate order list), plus a `key -> index` map for reappearing keys.
    /// Holding an index (not a `&mut` into a map) lets the streaming fast path
    /// keep a pointer to the current group across rows without a borrow conflict.
    group_vec: Vec<(String, Binding, Vec<Agg>)>,
    group_index: FxHashMap<String, usize>,
    /// Streaming fast path: the previous row's grouping values + its group index.
    /// `WITH <driving-var>, <agg>` emits rows contiguous by key, so a plain value
    /// compare against these accumulates the whole run with no key string/hash.
    last_key_vals: Vec<Val>,
    last_idx: Option<usize>,
    /// Reused scratch: current row's grouping values, and the built string key.
    key_vals: Vec<Val>,
    key_buf: String,
    distinct_seen: FxHashSet<String>,
}

impl<'p> ProjAccum<'p> {
    fn new(proj: &'p CProjection, ctx: &Ctx) -> Self {
        let topk = !proj.aggregating
            && !proj.order_by.is_empty()
            && proj.limit.is_some()
            && !proj.distinct
            && !proj.order_needs_output;
        ProjAccum {
            proj,
            grouped: proj.aggregating
                && (!proj.group_by.is_empty() || proj.items.iter().any(|i| !i.is_agg)),
            topk,
            cap: proj.skip_val(ctx) + proj.limit_val(ctx).unwrap_or(0),
            threshold: None,
            sort_scratch: Binding::default(),
            rows: Vec::new(),
            global: None,
            group_vec: Vec::new(),
            group_index: FxHashMap::default(),
            last_key_vals: Vec::new(),
            last_idx: None,
            key_vals: Vec::new(),
            key_buf: String::new(),
            distinct_seen: FxHashSet::default(),
        }
    }

    fn project_row(
        &self,
        graph: &Graph,
        ctx: &Ctx,
        input: &Binding,
        agg_values: Option<&[Val]>,
    ) -> Binding {
        let proj = self.proj;
        let mut out = Binding(vec![None; proj.out_len]);
        if proj.star {
            for (i, &islot) in proj.star_cols.iter().enumerate() {
                if let Some(v) = input.get(islot) {
                    out.0[i] = Some(v.clone());
                }
            }
        } else {
            let env = Env {
                graph,
                ctx,
                binding: input,
                group: None,
                agg_values,
            };
            for (i, item) in proj.items.iter().enumerate() {
                out.0[i] = Some(eval_item(&env, item));
            }
        }
        out
    }

    fn sort_keys(
        &self,
        graph: &Graph,
        ctx: &Ctx,
        input: &Binding,
        projected: &Binding,
        agg_values: Option<&[Val]>,
    ) -> Vec<Val> {
        let proj = self.proj;
        if proj.order_by.is_empty() {
            return Vec::new();
        }
        let mut sort_binding = projected.clone();
        for &islot in &proj.order_overlay {
            sort_binding.0.push(input.get(islot).cloned());
        }
        let env = Env {
            graph,
            ctx,
            binding: &sort_binding,
            group: None,
            agg_values,
        };
        proj.order_by.iter().map(|s| eval(&env, &s.expr)).collect()
    }

    /// Accept one input binding. Returns `false` to request a stop (streamable
    /// LIMIT: non-aggregating, no ORDER BY, enough rows collected).
    fn accept(&mut self, graph: &Graph, ctx: &Ctx, binding: &Binding) -> bool {
        let proj = self.proj;
        if self.topk {
            // Sort key from the input alone (output slots absent + input overlay),
            // built into the reused scratch binding (no per-row alloc).
            self.sort_scratch.0.clear();
            self.sort_scratch.0.resize(proj.out_len, None);
            for &islot in &proj.order_overlay {
                let v = binding.get(islot).cloned();
                self.sort_scratch.0.push(v);
            }
            let keys: Vec<Val> = {
                let env = Env {
                    graph,
                    ctx,
                    binding: &self.sort_scratch,
                    group: None,
                    agg_values: None,
                };
                proj.order_by.iter().map(|s| eval(&env, &s.expr)).collect()
            };
            // Once at capacity, skip (no clone) anything not better than the worst kept.
            if let Some(th) = &self.threshold {
                if cmp_keys(&keys, th, &proj.order_by) != Ordering::Less {
                    return true;
                }
            }
            self.rows.push((binding.clone(), keys));
            if self.cap >= 1 && self.rows.len() >= self.cap * 2 {
                let cap = self.cap;
                self.rows
                    .select_nth_unstable_by(cap - 1, |a, b| cmp_keyed(a, b, &proj.order_by));
                self.rows.truncate(cap);
                self.threshold = Some(self.rows[cap - 1].1.clone());
            }
            return true;
        }
        if proj.aggregating {
            if !self.grouped {
                // Global aggregate: one accumulator set, no key/map per row.
                let entry = self.global.get_or_insert_with(|| {
                    (binding.clone(), proj.aggs.iter().map(Agg::new).collect())
                });
                step_aggs(&mut entry.1, &proj.aggs, graph, ctx, binding);
                return true;
            }
            // Evaluate this row's grouping values into the reused scratch.
            self.key_vals.clear();
            {
                let env = Env::new(graph, ctx, binding);
                // Explicit GROUP BY keys drive grouping; else the non-agg items.
                if proj.group_by.is_empty() {
                    for item in proj.items.iter().filter(|i| !i.is_agg) {
                        self.key_vals.push(eval_item(&env, item));
                    }
                } else {
                    for item in &proj.group_by {
                        self.key_vals.push(eval_item(&env, item));
                    }
                }
            }
            // Streaming fast path: rows for one group usually arrive contiguously
            // (grouping by the driving variable), so if this row's values equal the
            // previous row's, fold straight into that group — no key string, no hash.
            if let Some(li) = self.last_idx {
                if group_vals_eq(&self.key_vals, &self.last_key_vals) {
                    step_aggs(&mut self.group_vec[li].2, &proj.aggs, graph, ctx, binding);
                    return true;
                }
            }
            // Key changed (or is out of order): build the string key and consult the
            // index. Only a run boundary (≪ every row) pays the build + hash here.
            self.key_buf.clear();
            for v in &self.key_vals {
                val_key(v, &mut self.key_buf);
                self.key_buf.push('\u{1}');
            }
            let idx = match self.group_index.get(self.key_buf.as_str()) {
                Some(&idx) => {
                    step_aggs(&mut self.group_vec[idx].2, &proj.aggs, graph, ctx, binding);
                    idx
                }
                None => {
                    let idx = self.group_vec.len();
                    let mut aggs: Vec<Agg> = proj.aggs.iter().map(Agg::new).collect();
                    step_aggs(&mut aggs, &proj.aggs, graph, ctx, binding);
                    self.group_vec
                        .push((self.key_buf.clone(), binding.clone(), aggs));
                    self.group_index.insert(self.key_buf.clone(), idx);
                    idx
                }
            };
            self.last_idx = Some(idx);
            self.last_key_vals.clear();
            self.last_key_vals.extend_from_slice(&self.key_vals);
            return true;
        }
        // Non-aggregating: project the row now (no full-binding clone retained).
        let projected = self.project_row(graph, ctx, binding, None);
        if proj.distinct && !self.distinct_seen.insert(row_key(&projected)) {
            return true;
        }
        let keys = self.sort_keys(graph, ctx, binding, &projected, None);
        self.rows.push((projected, keys));
        // Streamable LIMIT: with no ORDER BY, match order is result order.
        if proj.order_by.is_empty() {
            if let Some(limit) = proj.limit_val(ctx) {
                if self.rows.len() >= proj.skip_val(ctx) + limit {
                    return false;
                }
            }
        }
        true
    }

    fn finish(mut self, graph: &Graph, ctx: &Ctx) -> Vec<Binding> {
        let proj = self.proj;
        if proj.aggregating {
            if !self.grouped {
                // Global aggregate always emits exactly one row (0/null over no input)
                // — unless a HAVING on the whole-input group filters it out.
                let (rep, aggs) = self.global.take().unwrap_or_else(|| {
                    (Binding::default(), proj.aggs.iter().map(Agg::new).collect())
                });
                let agg_values: Vec<Val> = aggs.into_iter().map(Agg::finish).collect();
                if passes_having(proj, graph, ctx, &rep, &agg_values) {
                    let projected = self.project_row(graph, ctx, &rep, Some(&agg_values));
                    let keys = self.sort_keys(graph, ctx, &rep, &projected, Some(&agg_values));
                    self.rows.push((projected, keys));
                }
            } else {
                let groups = std::mem::take(&mut self.group_vec);
                for (_key, rep, aggs) in groups {
                    let agg_values: Vec<Val> = aggs.into_iter().map(Agg::finish).collect();
                    // ISO HAVING: drop a group whose post-aggregation predicate is
                    // not TRUE (three-valued — NULL/false both drop).
                    if !passes_having(proj, graph, ctx, &rep, &agg_values) {
                        continue;
                    }
                    let projected = self.project_row(graph, ctx, &rep, Some(&agg_values));
                    let keys = self.sort_keys(graph, ctx, &rep, &projected, Some(&agg_values));
                    self.rows.push((projected, keys));
                }
                if proj.distinct {
                    let mut seen: FxHashSet<String> = FxHashSet::default();
                    self.rows.retain(|(b, _)| seen.insert(row_key(b)));
                }
            }
        } else if self.topk {
            // Trim to the top-k input bindings, then project only those.
            if self.cap >= 1 && self.rows.len() > self.cap {
                let cap = self.cap;
                self.rows
                    .select_nth_unstable_by(cap - 1, |a, b| cmp_keyed(a, b, &proj.order_by));
                self.rows.truncate(cap);
            }
            let buf = std::mem::take(&mut self.rows);
            self.rows = buf
                .into_iter()
                .map(|(inb, keys)| (self.project_row(graph, ctx, &inb, None), keys))
                .collect();
        }
        if !proj.order_by.is_empty() {
            let cmp =
                |a: &(Binding, Vec<Val>), b: &(Binding, Vec<Val>)| cmp_keyed(a, b, &proj.order_by);
            // ORDER BY + LIMIT: partition the smallest `cap` with quickselect
            // (O(n)), then sort only those — instead of a full O(n log n) sort.
            let n = self.rows.len();
            if let Some(cap) = proj.limit_val(ctx).map(|l| proj.skip_val(ctx) + l) {
                if cap >= 1 && cap < n {
                    self.rows.select_nth_unstable_by(cap - 1, cmp);
                    self.rows.truncate(cap);
                }
            }
            self.rows.sort_by(cmp);
        }
        let start = proj.skip_val(ctx);
        let mut rows: Vec<Binding> = self.rows.into_iter().map(|(b, _)| b).skip(start).collect();
        if let Some(n) = proj.limit_val(ctx) {
            rows.truncate(n);
        }
        rows
    }

    /// Fold another chunk's aggregate state into this one — the reduce step for
    /// parallel aggregation. Merges the global accumulator and each group's
    /// accumulators (appending `other`'s new groups in its first-seen order);
    /// caller gates to the aggregating, non-topk, non-DISTINCT-agg case. Merging
    /// chunks in seed order reproduces the serial first-seen group order exactly.
    #[cfg(feature = "parallel-query")]
    fn merge(&mut self, mut other: Self) {
        if let Some((rep, other_aggs)) = other.global.take() {
            match &mut self.global {
                Some((_, aggs)) => {
                    for (a, o) in aggs.iter_mut().zip(other_aggs) {
                        a.merge(o);
                    }
                }
                None => self.global = Some((rep, other_aggs)),
            }
        }
        for (key, rep, other_aggs) in other.group_vec {
            match self.group_index.get(&key) {
                Some(&idx) => {
                    for (a, o) in self.group_vec[idx].2.iter_mut().zip(other_aggs) {
                        a.merge(o);
                    }
                }
                None => {
                    let idx = self.group_vec.len();
                    self.group_index.insert(key.clone(), idx);
                    self.group_vec.push((key, rep, other_aggs));
                }
            }
        }
    }
}

/// Project the binding stream from `incoming × pending matches` (streamed) into
/// `proj`, returning result rows. The hot path: no intermediate `Vec<Binding>`.
pub(super) fn project_matches(
    graph: &Graph,
    ctx: &Ctx,
    incoming: &[Binding],
    matches: &[&CClause],
    proj: &CProjection,
) -> Vec<Binding> {
    if use_vec() {
        if let Some(cols) = vectorized_cols(graph, ctx, incoming, matches, proj) {
            // WITH stage: carry output forward as bindings, *preserving* element
            // handles (a carried node stays `Val::Node`, not flattened to an id).
            let nrows = cols.first().map_or(0, |c| c.len());
            return (0..nrows)
                .map(|i| Binding(cols.iter().map(|c| Some(c[i].clone())).collect()))
                .collect();
        }
    }
    let mut acc = ProjAccum::new(proj, ctx);
    let simple = single_simple_clause(matches);
    for inb in incoming {
        let mut work = inb.clone();
        let cont = match simple {
            Some((path, cwhere, cwhere_prog, scope_len)) => {
                work.resize(scope_len);
                match_one_path(graph, ctx, path, &mut work, &mut |b| {
                    if !where_keep(&Env::new(graph, ctx, b), cwhere, cwhere_prog) {
                        return true;
                    }
                    acc.accept(graph, ctx, b)
                })
            }
            None => drive_matches(graph, ctx, matches, 0, &mut work, &mut |b| {
                acc.accept(graph, ctx, b)
            }),
        };
        if !cont {
            break;
        }
    }
    acc.finish(graph, ctx)
}
