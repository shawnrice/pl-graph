//! Top-level statement execution: projection to output rows (with ORDER BY sort
//! keys), running a linear query and the set operators (UNION/EXCEPT/INTERSECT),
//! and the write clauses (INSERT/SET/REMOVE/DELETE, incl. _MERGE). Extracted from
//! the evaluator (`super`); shares its context/helpers via `use super::*`.
use super::*;

// --- projection --------------------------------------------------------------

/// Compare two ORDER BY key vectors lexicographically (per-key direction/nulls).
pub(super) fn cmp_keys(a: &[Val], b: &[Val], order: &[crate::gql::plan::CSortItem]) -> Ordering {
    for (i, s) in order.iter().enumerate() {
        let o = compare_sort(&a[i], &b[i], s.descending, s.nulls_first);
        if o != Ordering::Equal {
            return o;
        }
    }
    Ordering::Equal
}

/// Compare two keyed rows by their ORDER BY keys.
pub(super) fn cmp_keyed(
    a: &(Binding, Vec<Val>),
    b: &(Binding, Vec<Val>),
    order: &[crate::gql::plan::CSortItem],
) -> Ordering {
    cmp_keys(&a.1, &b.1, order)
}

/// Compare two ORDER BY keys, honoring direction and ISO NULLS FIRST/LAST.
pub(super) fn compare_sort(
    a: &Val,
    b: &Val,
    descending: bool,
    nulls_first: Option<bool>,
) -> Ordering {
    let a_null = is_nullish(a);
    let b_null = is_nullish(b);
    if a_null && b_null {
        return Ordering::Equal;
    }
    if a_null || b_null {
        // Null placement is absolute (independent of ASC/DESC). With no explicit
        // NULLS FIRST/LAST, nulls sort LAST — ISO GQL leaves the default
        // unspecified, so we pin one for cross-engine determinism.
        let first = nulls_first.unwrap_or(false);
        return if a_null == first {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let base = cmp_total(a, b);
    if descending {
        base.reverse()
    } else {
        base
    }
}

// --- linear query & set ops --------------------------------------------------

/// The result of running one linear query part. A top-level part produces a
/// `RowSet` of output `Value`s; an inline `CALL` body produces projected
/// `Binding`s instead — so element columns (`RETURN *`, `RETURN n`) keep their
/// `Val::Node`/`Val::Edge` identity across the merge-back into the outer row (a
/// serialized `Value::Map` can't round-trip to a `Val`). Selected by the
/// `want_binds` flag on [`run_linear_from`].
pub(super) enum LinearOut {
    Rows(RowSet),
    Binds(Vec<Binding>),
}

pub(super) fn run_linear(
    linear: &CLinear,
    graph: &mut Graph,
    plan: &CQuery,
    params: &[Val],
) -> CodeResult<RowSet> {
    match run_linear_from(
        linear,
        graph,
        plan,
        params,
        vec![Binding::default()],
        None,
        false,
    )? {
        LinearOut::Rows(rs) => Ok(rs),
        LinearOut::Binds(_) => unreachable!("top-level run requests rows"),
    }
}

/// [`run_linear`] starting from a given set of bindings — the seed for an inline
/// subquery's correlated run (the imported scope variables live in `initial`).
pub(super) fn run_linear_from(
    linear: &CLinear,
    graph: &mut Graph,
    plan: &CQuery,
    params: &[Val],
    initial: Vec<Binding>,
    shared: Option<&Ctx>,
    // When true, a terminal RETURN projects to `Binding`s (element-preserving,
    // for an inline `CALL` merge-back) instead of a `RowSet` of output values.
    want_binds: bool,
) -> CodeResult<LinearOut> {
    // `bindings` is the materialized row set at the last barrier; `pending` are
    // MATCH clauses deferred so a projection (or write) can stream them directly.
    let mut bindings: Vec<Binding> = initial;
    let mut pending: Vec<&CClause> = Vec::new();
    // Refs (keys/labels) resolved to ids. A correlated inline subquery reuses the
    // caller's ctx (`shared`) — it shares the plan's tables, so resolving per
    // outer row is pure waste — and only OWNS a ctx if it writes (re-resolved
    // after each mutation). A top-level run always owns its ctx.
    let mut owned: Option<Ctx> = match shared {
        Some(_) => None,
        None => Some(resolve_ctx(graph, plan, params)),
    };
    // The current read ctx: the owned one if we've resolved (top-level or after a
    // write), else the shared borrow. Expands inline, so it never holds a borrow
    // across the write arms' `owned.as_mut()` / re-resolve.
    macro_rules! ctx {
        () => {
            owned
                .as_ref()
                .unwrap_or_else(|| shared.expect("a shared ctx"))
        };
    }

    for clause in &linear.clauses {
        match clause {
            CClause::Match { .. } => pending.push(clause), // defer; consumed at a barrier
            CClause::With {
                projection,
                where_,
                where_prog,
            } => {
                let projected = project_matches(graph, ctx!(), &bindings, &pending, projection);
                pending.clear();
                bindings = if where_.is_none() {
                    projected
                } else {
                    projected
                        .into_iter()
                        .filter(|b| {
                            where_keep(
                                &Env::new(graph, ctx!(), b),
                                where_.as_ref(),
                                where_prog.as_ref(),
                            )
                        })
                        .collect()
                };
            }
            CClause::Filter { pred, prog } => {
                // Flush deferred matches (the predicate may reference their vars),
                // then drop every row where the condition is not TRUE.
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                bindings
                    .retain(|b| where_keep(&Env::new(graph, ctx!(), b), Some(pred), Some(prog)));
            }
            CClause::Page {
                order_by,
                skip,
                limit,
            } => {
                // ISO `<order by and page statement>` in statement position. Flush
                // deferred matches (the sort keys may reference their vars), then
                // sort and slice the working table. Because this runs BEFORE any
                // projection, a later RETURN only ever projects the surviving rows.
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                if !order_by.is_empty() {
                    // Key each row once, then a STABLE sort — same comparator (and
                    // so the same total order and NULLS FIRST/LAST handling) as a
                    // projection's ORDER BY.
                    let mut keyed: Vec<(Binding, Vec<Val>)> = bindings
                        .drain(..)
                        .map(|b| {
                            let keys = {
                                let env = Env::new(graph, ctx!(), &b);
                                order_by.iter().map(|s| eval(&env, &s.expr)).collect()
                            };
                            (b, keys)
                        })
                        .collect();
                    keyed.sort_by(|a, b| cmp_keyed(a, b, order_by));
                    bindings = keyed.into_iter().map(|(b, _)| b).collect();
                }
                let start = count_of(skip.as_ref(), ctx!())
                    .unwrap_or(0)
                    .min(bindings.len());
                bindings.drain(..start);
                if let Some(n) = count_of(limit.as_ref(), ctx!()) {
                    bindings.truncate(n);
                }
            }
            CClause::Let(items) => {
                // Flush deferred matches, then bind each new variable into every
                // row (left-to-right, so a later item sees an earlier one).
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                for b in &mut bindings {
                    for (slot, expr, _prog) in items {
                        let v = {
                            let env = Env::new(graph, ctx!(), b);
                            eval(&env, expr)
                        };
                        b.set(*slot, v);
                    }
                }
                ctx!().check_fault()?;
            }
            CClause::For {
                list,
                alias_slot,
                ord,
                scope_len,
            } => {
                // FOR's list can reference a deferred MATCH var, so flush pending
                // first, then unwind: each incoming binding fans out to one row
                // per list element (ISO GQL's UNWIND). A list unwinds its
                // elements; null yields zero rows; any other scalar unwinds as a
                // one-element list.
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                let mut out = Vec::new();
                for inb in &bindings {
                    let mut work = inb.clone();
                    work.resize(*scope_len);
                    let listv = {
                        let env = Env::new(graph, ctx!(), &work);
                        eval(&env, list)
                    };
                    let elems = match listv {
                        Val::List(items) => items.to_vec(),
                        Val::Null => Vec::new(),
                        scalar => vec![scalar],
                    };
                    for (i, elem) in elems.into_iter().enumerate() {
                        work.set(*alias_slot, elem);
                        if let Some((is_ordinality, ord_slot)) = ord {
                            let counter = if *is_ordinality {
                                (i + 1) as f64
                            } else {
                                i as f64
                            };
                            work.set(*ord_slot, Val::Num(counter));
                        }
                        out.push(work.clone());
                    }
                }
                bindings = out;
                ctx!().check_fault()?;
            }
            CClause::CallNamed {
                optional,
                proc_name,
                algo,
                config,
                binds,
                scope_len,
            } => {
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                let Some(dispatch) = algo else {
                    let msg = match crate::gql::plan::suggest_procedure(proc_name) {
                        Some(s) => format!("unknown procedure: {proc_name} (did you mean '{s}'?)"),
                        None => format!("unknown procedure: {proc_name}"),
                    };
                    return Err(CodeError::new(ErrorCode::Unsupported, msg));
                };
                // Build the algorithm config from the (constant) config exprs.
                let cfg = {
                    let scratch = Binding::default();
                    let env = Env::new(graph, ctx!(), &scratch);
                    let mut cfg = crate::algo::AlgoConfig::default();
                    for (field, expr) in config {
                        apply_algo_config(&mut cfg, field, &eval(&env, expr))?;
                    }
                    cfg
                };
                // Raw `(vertex, result)` rows — no RowSet, so `node` binds as a
                // live `Val::Node` handle (hydrated to `{id,labels,properties}`
                // only for rows that survive to output) rather than a stringified id.
                let (result_col, results) = crate::algo::run_columns(graph, dispatch, &cfg)
                    .map_err(|e| CodeError::new(ErrorCode::InvalidValue, e))?;
                // Resolve each YIELD bind to its source: the vertex handle or the
                // result value.
                let mut bind_src: Vec<(bool, usize)> = Vec::with_capacity(binds.len());
                for b in binds {
                    let is_node = if b.column == "node" {
                        true
                    } else if b.column == result_col {
                        false
                    } else {
                        return Err(CodeError::new(
                            ErrorCode::InvalidValue,
                            format!(
                                "procedure `{proc_name}` has no output column `{}`",
                                b.column
                            ),
                        ));
                    };
                    bind_src.push((is_node, b.slot));
                }
                // Cross-join incoming bindings with the procedure's rows (the call
                // is uncorrelated); OPTIONAL keeps the outer row (null-filled) when
                // the procedure yields nothing.
                let mut out = Vec::new();
                for inb in &bindings {
                    let mut work = inb.clone();
                    work.resize(*scope_len);
                    if results.is_empty() && *optional {
                        for (_, slot) in &bind_src {
                            work.set(*slot, Val::Null);
                        }
                        out.push(work);
                        continue;
                    }
                    for (vertex, value) in &results {
                        let mut w = work.clone();
                        for (is_node, slot) in &bind_src {
                            let bound = if *is_node {
                                Val::Node(*vertex)
                            } else {
                                value_to_val(value)
                            };
                            w.set(*slot, bound);
                        }
                        out.push(w);
                    }
                }
                bindings = out;
                ctx!().check_fault()?;
                owned = Some(resolve_ctx(graph, plan, params)); // writeProperty may have mutated
            }
            CClause::CallInline {
                optional,
                imports,
                body,
                body_more,
                out_binds,
                body_star,
                body_read_only,
            } => {
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                // Run the nested query once per outer row (correlated), seeding it
                // with only the imported scope variables, and merge its RETURN
                // columns back — one merged row per nested row. A read-only body
                // reuses this ctx (shared plan tables) so it never re-resolves.
                let mut out = Vec::new();
                for outer in &bindings {
                    let mut seed = Binding::default();
                    for (outer_slot, nested_slot) in imports {
                        if let Some(v) = outer.get(*outer_slot) {
                            seed.set(*nested_slot, v.clone());
                        }
                    }
                    let reuse = if *body_read_only { Some(ctx!()) } else { None };
                    // Run the body to element-preserving bindings so a returned
                    // node/edge/`*` merges back with its `Val` identity intact.
                    let LinearOut::Binds(mut rows) = run_linear_from(
                        body,
                        graph,
                        plan,
                        params,
                        vec![seed.clone()],
                        reuse,
                        true,
                    )?
                    else {
                        unreachable!("inline body requests binds")
                    };
                    // Fold in any set-op parts (`… UNION/EXCEPT/INTERSECT …`), each run
                    // against the same seed, matching the top-level set-op semantics.
                    for (op, part) in body_more {
                        let LinearOut::Binds(right) = run_linear_from(
                            part,
                            graph,
                            plan,
                            params,
                            vec![seed.clone()],
                            reuse,
                            true,
                        )?
                        else {
                            unreachable!("inline body requests binds")
                        };
                        rows = combine_binds(*op, rows, right, out_binds.len());
                    }
                    if rows.is_empty() && *optional {
                        // A named RETURN null-fills its produced columns; a `RETURN *`
                        // produces no new named columns (its columns are scope vars,
                        // imports included), so keep the outer row untouched — leaving
                        // freshly-introduced vars unbound, matching the TS engine.
                        let mut w = outer.clone();
                        if !*body_star {
                            for slot in out_binds {
                                w.set(*slot, Val::Null);
                            }
                        }
                        out.push(w);
                        continue;
                    }
                    for row in &rows {
                        let mut w = outer.clone();
                        for (i, slot) in out_binds.iter().enumerate() {
                            w.set(*slot, row.get(i).cloned().unwrap_or(Val::Null));
                        }
                        out.push(w);
                    }
                }
                bindings = out;
                owned = Some(resolve_ctx(graph, plan, params)); // a nested write may have mutated
                ctx!().check_fault()?;
            }
            CClause::Return(proj) => {
                let out = if want_binds {
                    // Inline-CALL body: project to element-preserving bindings (same
                    // path a WITH uses), so a returned node/edge/`*` keeps identity.
                    LinearOut::Binds(project_matches(graph, ctx!(), &bindings, &pending, proj))
                } else {
                    LinearOut::Rows(project_to_rows(graph, ctx!(), &bindings, &pending, proj))
                };
                ctx!().check_fault()?;
                return Ok(out);
            }
            CClause::Finish => {
                return Ok(if want_binds {
                    LinearOut::Binds(Vec::new())
                } else {
                    LinearOut::Rows(RowSet::new(Vec::new()))
                });
            }
            // Mutations run eagerly, exactly once per binding. Flush deferred
            // matches first, then re-resolve refs against the mutated graph.
            CClause::Insert(patterns) => {
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                let mut inserted = Vec::with_capacity(bindings.len());
                for b in &bindings {
                    inserted.push(run_insert(
                        graph,
                        owned.as_mut().expect("a write clause owns its ctx"),
                        plan,
                        patterns,
                        b,
                    ));
                }
                bindings = inserted;
                ctx!().check_fault()?;
                owned = Some(resolve_ctx(graph, plan, params));
            }
            CClause::Merge(m) => {
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                let mut merged = Vec::with_capacity(bindings.len());
                for b in &bindings {
                    merged.push(run_merge(graph, ctx!(), m, b));
                }
                bindings = merged;
                ctx!().check_fault()?;
                owned = Some(resolve_ctx(graph, plan, params));
            }
            CClause::Set(items) => {
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                for b in &bindings {
                    run_set(graph, ctx!(), items, b);
                }
                ctx!().check_fault()?;
                owned = Some(resolve_ctx(graph, plan, params));
            }
            CClause::Remove(items) => {
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                for b in &bindings {
                    run_remove(graph, ctx!(), items, b);
                }
                ctx!().check_fault()?;
            }
            CClause::Delete { detach, targets } => {
                if !pending.is_empty() {
                    bindings = materialize_matches(graph, ctx!(), &bindings, &pending);
                    pending.clear();
                }
                for b in &bindings {
                    run_delete(graph, ctx!(), *detach, targets, b)?;
                }
            }
        }
    }
    ctx!().check_fault()?;
    // write-only / no RETURN
    Ok(if want_binds {
        LinearOut::Binds(Vec::new())
    } else {
        LinearOut::Rows(RowSet::new(Vec::new()))
    })
}

// --- write execution ---------------------------------------------------------

/// Concrete labels a (lowered) label expression names, for element creation;
/// resolves each ref back to its name. `|`/`!`/`%` can't name a creatable set.
/// Labels to CREATE for an INSERT element: `None` for no label expression
/// (a legitimately unlabelled node), the conjunction for `A`/`A&B`, and `None`
/// for a disjunction/negation/wildcard — an ambiguous form that can't be created
/// (the caller raises FAULT_BAD_LABEL). A non-INSERT (MATCH) label expression is
/// handled elsewhere; this deliberately refuses the ambiguous forms.
pub(super) fn creatable_labels(expr: Option<&CLabelExpr>, names: &[String]) -> Option<Vec<String>> {
    match expr {
        None => Some(Vec::new()),
        Some(CLabelExpr::Label(r)) => Some(vec![names[*r].clone()]),
        Some(CLabelExpr::And(l, r)) => {
            let mut v = creatable_labels(Some(l), names)?;
            v.extend(creatable_labels(Some(r), names)?);
            Some(v)
        }
        Some(_) => None, // |, !, % — not a concrete label set
    }
}

/// Evaluate a pattern property map to concrete core `Value`s (for create/set).
pub(super) fn eval_props(
    graph: &Graph,
    ctx: &Ctx,
    props: &[CPropConstraint],
    binding: &Binding,
) -> Vec<(String, Value)> {
    let env = Env::new(graph, ctx, binding);
    props
        .iter()
        .map(|pc| (pc.key.clone(), val_to_value(graph, &eval(&env, &pc.value))))
        .collect()
}

/// Insert a vertex, using a string `id` property as the element's external id.
///
/// A domain `id` (`INSERT (:P {id: 'alice'})`) becomes the engine's identity — so
/// `element_id(n)` equals it and `toNdjson` round-trips by domain identity instead
/// of a synthetic `_n{k}` — while `id` is still stored as an ordinary property
/// (`RETURN n.id` works, exactly as an NDJSON top-level id + `properties.id` do).
/// A non-string or absent id mints a synthetic one. A duplicate string id faults
/// (ids are unique); the fault rolls the statement back, so the throwaway synthetic
/// vertex created to keep evaluation well-formed leaves no trace.
pub(super) fn insert_vertex_with_id(
    graph: &mut Graph,
    ctx: &Ctx,
    labels: &[String],
    props: Vec<(String, Value)>,
) -> u32 {
    if let Some((_, Value::Str(id))) = props.iter().find(|(k, _)| k == "id") {
        let id = id.clone();
        if graph.vertex_by_id(&id).is_some() {
            ctx.set_fault(FAULT_ID_DUP);
            return graph.add_vertex(labels, props); // synth id; rolled back by the fault
        }
        return graph.add_vertex_with_id(&id, labels, props);
    }
    graph.add_vertex(labels, props)
}

/// Insert an edge, using a string `id` property as its external identity — the
/// edge analogue of [`insert_vertex_with_id`]. Edge ids are unique among edges; a
/// duplicate faults (rolled back with the throwaway edge). A rollback removes the
/// edge and its id overlay together (`remove_edge` drops `eid_fwd`/`eid_rev`), so
/// `add_edge` + `set_edge_id` needs no separate undo.
pub(super) fn insert_edge_with_id(
    graph: &mut Graph,
    ctx: &Ctx,
    from: u32,
    to: u32,
    etype: &str,
    props: Vec<(String, Value)>,
) -> u32 {
    if let Some((_, Value::Str(id))) = props.iter().find(|(k, _)| k == "id") {
        let id = id.clone();
        if graph.edge_by_id(&id).is_some() {
            ctx.set_fault(FAULT_ID_DUP);
            return graph.add_edge(from, to, etype, props); // synth; rolled back by the fault
        }
        let ei = graph.add_edge(from, to, etype, props);
        graph.set_edge_id(ei, &id);
        return ei;
    }
    graph.add_edge(from, to, etype, props)
}

/// Create a node from a pattern, reusing an already-bound variable.
pub(super) fn ensure_node(
    graph: &mut Graph,
    ctx: &Ctx,
    binding: &mut Binding,
    node: &CNode,
) -> u32 {
    if let Some(slot) = node.var_slot {
        if let Some(Val::Node(vi)) = binding.get(slot) {
            return *vi;
        }
    }
    // A node may be unlabelled, but a non-conjunction label expression
    // (`A|B`, `!A`, `%`) is ambiguous — reject it rather than silently create an
    // unlabelled node.
    let labels = creatable_labels(node.label.as_ref(), ctx.label_names).unwrap_or_else(|| {
        ctx.set_fault(FAULT_BAD_LABEL);
        Vec::new()
    });
    let props = eval_props(graph, ctx, &node.props, binding);
    // Create eagerly and note the vertex for the commit-time constraint check
    // (unique / required / type). Inside the statement's auto-commit frame the
    // checks defer to end-of-statement, so a multi-row INSERT whose rows only
    // collide with each other — or a node inserted before a sibling supplies its
    // key — is judged against the fully-staged graph, and a violation rolls the
    // whole statement back (per-statement atomicity) instead of leaving a partial
    // write. `_MERGE` reconciles instead; see docs/design/gql-extensions.md §3.
    let vi = insert_vertex_with_id(graph, ctx, &labels, props);
    graph.tx_note_touched(vi);
    if let Some(slot) = node.var_slot {
        binding.set(slot, Val::Node(vi));
    }
    vi
}

pub(super) fn run_insert(
    graph: &mut Graph,
    ctx: &mut Ctx,
    plan: &CQuery,
    patterns: &[CPath],
    binding: &Binding,
) -> Binding {
    let mut out = binding.clone();
    for pattern in patterns {
        // Refresh id resolution so this element's property expressions can read
        // a sibling created earlier in the same INSERT (forward reference).
        ctx.refresh_ids(graph, plan);
        let mut prev = ensure_node(graph, ctx, &mut out, &pattern.start);
        for CSegment { rel, node, .. } in &pattern.segments {
            ctx.refresh_ids(graph, plan);
            let next = ensure_node(graph, ctx, &mut out, node);
            let (from, to) = if rel.direction == Direction::In {
                (next, prev)
            } else {
                (prev, next)
            };
            // An edge MUST carry exactly one type: reject a typeless edge or a
            // non-conjunction type expression (empty → FAULT_BAD_LABEL) instead
            // of silently creating an empty-type edge that won't round-trip.
            let etype = creatable_labels(rel.label.as_ref(), ctx.label_names)
                .and_then(|ls| ls.into_iter().next());
            let etype = etype.unwrap_or_else(|| {
                ctx.set_fault(FAULT_BAD_LABEL);
                String::new()
            });
            ctx.refresh_ids(graph, plan);
            let eprops = eval_props(graph, ctx, &rel.props, &out);
            let ei = insert_edge_with_id(graph, ctx, from, to, &etype, eprops);
            // Note the edge for the commit-time edge-constraint check (unique /
            // required / type), mirroring `ensure_node`'s vertex handling.
            graph.tx_note_touched_edge(ei);
            if let Some(slot) = rel.var_slot {
                out.set(slot, Val::Edge(ei));
            }
            prev = next;
        }
    }
    out
}

/// Infer the conflict key for `_MERGE`: the single unique-constrained key present
/// in the pattern's props. `None` if none apply (can't define the key) or if more
/// than one does (ambiguous) — both surface as `FAULT_MERGE_KEY`
/// (`InvalidGraphOp`), matching the TS engine's code. See gql-extensions.md §2.2.
pub(super) fn infer_merge_key(
    graph: &Graph,
    labels: &[String],
    props: &[(String, Value)],
) -> Option<(String, String, Value)> {
    let mut found: Option<(String, String, Value)> = None;
    for label in labels {
        for key in graph.unique_keys(label) {
            if let Some((_, value)) = props.iter().find(|(k, _)| k == key) {
                if found.is_some() {
                    return None; // ambiguous — more than one constrained key present
                }
                found = Some((label.clone(), key.clone(), value.clone()));
            }
        }
    }
    found
}

/// Apply `_ON_CREATE` / `_ON_UPDATE` SET items to the node or edge bound in
/// `binding` (mirrors [`run_set`]).
pub(super) fn apply_merge_sets(
    graph: &mut Graph,
    ctx: &Ctx,
    items: &[CSetItem],
    binding: &Binding,
) {
    for item in items {
        match item {
            CSetItem::Prop {
                var_slot,
                key,
                value,
            } => {
                let target = binding.get(*var_slot).cloned();
                let v = {
                    let env = Env::new(graph, ctx, binding);
                    val_to_value(graph, &eval(&env, value))
                };
                match target {
                    Some(Val::Node(vi)) => graph.set_vertex_prop(vi, key, v),
                    Some(Val::Edge(ei)) => {
                        graph.set_edge_prop(ei, key, v);
                        graph.tx_note_touched_edge(ei);
                    }
                    _ => {}
                }
            }
            CSetItem::Label { var_slot, label } => match binding.get(*var_slot).cloned() {
                Some(Val::Node(vi)) => graph.add_vertex_label(vi, label),
                Some(Val::Edge(ei)) => {
                    graph.add_edge_label(ei, label);
                    graph.tx_note_touched_edge(ei);
                }
                _ => {}
            },
        }
    }
}

/// Resolve a `_MERGE` edge endpoint: the vertex matched by the endpoint's
/// unique-constraint key. `None` if no key can be inferred or no vertex matches
/// (surfaced as `FAULT_MERGE_KEY` by the caller).
pub(super) fn resolve_merge_endpoint(
    graph: &Graph,
    ctx: &Ctx,
    node: &CNode,
    binding: &Binding,
) -> Option<u32> {
    // An endpoint bound by a preceding clause — `MATCH (a), (b) _MERGE (a)-[:R]->(b)`,
    // the natural way to merge an edge between two known vertices — is already a
    // resolved vertex. Use it directly rather than re-inferring a unique key from
    // the (empty) node pattern, which would fail with FAULT_MERGE_KEY and made the
    // bound-variable form of edge `_MERGE` unusable.
    if let Some(slot) = node.var_slot {
        if let Some(Val::Node(vi)) = binding.get(slot) {
            return Some(*vi);
        }
    }
    let labels = creatable_labels(node.label.as_ref(), ctx.label_names)?;
    let props = eval_props(graph, ctx, &node.props, binding);
    let (label, key, value) = infer_merge_key(graph, &labels, &props)?;
    graph.unique_lookup(&label, &key, &value)
}

/// `_MERGE` edge form (v1): match both endpoints by key, then upsert the single
/// edge between them keyed structurally by `(from, to, type)`. Dispositions apply
/// to the edge (which has no key prop, so the default clobbers all its props).
/// Byte-identical to the TS `runMergeEdge`.
pub(super) fn run_merge_edge(
    graph: &mut Graph,
    ctx: &Ctx,
    clause: &CMerge,
    binding: &Binding,
) -> Binding {
    let mut out = binding.clone();
    let seg = &clause.pattern.segments[0];

    let (Some(a), Some(b)) = (
        resolve_merge_endpoint(graph, ctx, &clause.pattern.start, binding),
        resolve_merge_endpoint(graph, ctx, &seg.node, binding),
    ) else {
        ctx.set_fault(FAULT_MERGE_KEY);
        return out;
    };

    let (from, to) = if seg.rel.direction == Direction::In {
        (b, a)
    } else {
        (a, b)
    };
    let Some(etype) = creatable_labels(seg.rel.label.as_ref(), ctx.label_names)
        .and_then(|ls| ls.into_iter().next())
    else {
        ctx.set_fault(FAULT_BAD_LABEL);
        return out;
    };
    let eprops = eval_props(graph, ctx, &seg.rel.props, binding);

    // Bind the resolved endpoints so the dispositions' expressions can read them.
    if let Some(s) = clause.pattern.start.var_slot {
        out.set(s, Val::Node(a));
    }
    if let Some(s) = seg.node.var_slot {
        out.set(s, Val::Node(b));
    }

    let ei = if let Some(ei) = graph.find_edge(from, to, &etype) {
        // Update path. An edge has no key prop → the default clobbers all props.
        match &clause.on_update {
            None => {
                for (k, v) in &eprops {
                    graph.set_edge_prop(ei, k, v.clone());
                }
            }
            Some(CMergeUpdate::Nothing) => {}
            Some(CMergeUpdate::Set { items, where_ }) => {
                if let Some(s) = seg.rel.var_slot {
                    out.set(s, Val::Edge(ei));
                }
                let passes = match where_ {
                    None => true,
                    Some(w) => {
                        let env = Env::new(graph, ctx, &out);
                        as_truth(&eval(&env, w)) == Some(true)
                    }
                };
                if passes {
                    apply_merge_sets(graph, ctx, items, &out);
                }
            }
        }
        ei
    } else {
        // Create path.
        let ei = graph.add_edge(from, to, &etype, eprops);
        if let Some(s) = seg.rel.var_slot {
            out.set(s, Val::Edge(ei));
        }
        if let Some(items) = &clause.on_create {
            apply_merge_sets(graph, ctx, items, &out);
        }
        ei
    };

    if let Some(s) = seg.rel.var_slot {
        out.set(s, Val::Edge(ei));
    }
    // Note the merged edge for the commit-time edge-constraint check.
    graph.tx_note_touched_edge(ei);
    out
}

/// `_MERGE` keyed upsert (v1: node form). Match by the constraint key; on miss,
/// insert the pattern (key + payload) then `_ON_CREATE`; on hit, apply the update
/// disposition — default clobbers the non-key payload, `_ON_UPDATE SET … [WHERE]`
/// replaces it, `_ON_UPDATE_NOTHING` leaves it. Byte-identical to the TS
/// `runMerge`. (Edge form arrives in a later slice.)
pub(super) fn run_merge(
    graph: &mut Graph,
    ctx: &Ctx,
    clause: &CMerge,
    binding: &Binding,
) -> Binding {
    let mut out = binding.clone();

    // Edge form = exactly one segment `(a)-(rel)->(b)`. Multi-hop compound
    // patterns are deferred (v2).
    match clause.pattern.segments.len() {
        0 => {}
        1 => return run_merge_edge(graph, ctx, clause, &out),
        _ => {
            ctx.set_fault(FAULT_MERGE_EDGE);
            return out;
        }
    }

    let node = &clause.pattern.start;
    let labels = creatable_labels(node.label.as_ref(), ctx.label_names).unwrap_or_else(|| {
        ctx.set_fault(FAULT_BAD_LABEL);
        Vec::new()
    });
    let props = eval_props(graph, ctx, &node.props, binding);

    let Some((label, key, value)) = infer_merge_key(graph, &labels, &props) else {
        ctx.set_fault(FAULT_MERGE_KEY);
        return out;
    };

    let vi = if let Some(vi) = graph.unique_lookup(&label, &key, &value) {
        // Update path.
        match &clause.on_update {
            None => {
                // Default clobber: write every non-key payload prop.
                for (k, v) in &props {
                    if *k != key {
                        graph.set_vertex_prop(vi, k, v.clone());
                    }
                }
            }
            Some(CMergeUpdate::Nothing) => {}
            Some(CMergeUpdate::Set { items, where_ }) => {
                if let Some(slot) = node.var_slot {
                    out.set(slot, Val::Node(vi));
                }
                let passes = match where_ {
                    None => true,
                    Some(w) => {
                        let env = Env::new(graph, ctx, &out);
                        as_truth(&eval(&env, w)) == Some(true)
                    }
                };
                if passes {
                    apply_merge_sets(graph, ctx, items, &out);
                }
            }
        }
        vi
    } else {
        // Create path: insert the pattern (key + payload), then `_ON_CREATE`.
        let vi = graph.add_vertex(&labels, props);
        if let Some(slot) = node.var_slot {
            out.set(slot, Val::Node(vi));
        }
        if let Some(items) = &clause.on_create {
            apply_merge_sets(graph, ctx, items, &out);
        }
        vi
    };

    if let Some(slot) = node.var_slot {
        out.set(slot, Val::Node(vi));
    }
    out
}

pub(super) fn run_set(graph: &mut Graph, ctx: &Ctx, items: &[CSetItem], binding: &Binding) {
    for item in items {
        match item {
            CSetItem::Prop {
                var_slot,
                key,
                value,
            } => {
                let Some(el) = binding.get(*var_slot).cloned() else {
                    continue;
                };
                let v = {
                    let env = Env::new(graph, ctx, binding);
                    val_to_value(graph, &eval(&env, value))
                };
                match el {
                    // An element keyed by a string `id` has that id as its identity,
                    // fixed at creation — re-keying it would break `element_id` /
                    // round-trip stability, so reject the SET (the fault rolls the
                    // statement back). A numeric/absent `id` is an ordinary
                    // (possibly unique-constrained) property and stays SET-able.
                    Val::Node(vi) if key == "id" && graph.vertex_id_is_identity(vi) => {
                        ctx.set_fault(FAULT_ID_IMMUTABLE);
                    }
                    Val::Edge(ei) if key == "id" && graph.edge_id_is_identity(ei) => {
                        ctx.set_fault(FAULT_ID_IMMUTABLE);
                    }
                    // Apply eagerly, then note the vertex — a SET that nulls a
                    // required key, breaks a type constraint, or collides under a
                    // unique constraint surfaces as ConstraintViolation at the
                    // frame's commit-time recheck (deferring it lets a
                    // momentarily-colliding intermediate settle first).
                    Val::Node(vi) => {
                        graph.set_vertex_prop(vi, key, v);
                        graph.tx_note_touched(vi);
                    }
                    Val::Edge(ei) => {
                        graph.set_edge_prop(ei, key, v);
                        graph.tx_note_touched_edge(ei);
                    }
                    _ => {}
                }
            }
            CSetItem::Label { var_slot, label } => match binding.get(*var_slot) {
                // Adding a label brings its required keys into force for this node;
                // the commit-time recheck flags one that's now missing.
                Some(Val::Node(vi)) => {
                    graph.add_vertex_label(*vi, label);
                    graph.tx_note_touched(*vi);
                }
                Some(Val::Edge(ei)) => {
                    // Relabelling an edge replaces its type — bring the new type's
                    // constraints into force at the commit-time recheck.
                    graph.add_edge_label(*ei, label);
                    graph.tx_note_touched_edge(*ei);
                }
                _ => {}
            },
        }
    }
}

pub(super) fn run_remove(graph: &mut Graph, _ctx: &Ctx, items: &[CRemoveItem], binding: &Binding) {
    for item in items {
        match item {
            CRemoveItem::Prop { var_slot, key } => match binding.get(*var_slot) {
                // Removing a required key surfaces as ConstraintViolation at the
                // frame's commit-time recheck (the key is then absent → missing).
                Some(Val::Node(vi)) => {
                    graph.remove_vertex_prop(*vi, key);
                    graph.tx_note_touched(*vi);
                }
                Some(Val::Edge(ei)) => {
                    graph.remove_edge_prop(*ei, key);
                    graph.tx_note_touched_edge(*ei);
                }
                _ => {}
            },
            CRemoveItem::Label { var_slot, label } => match binding.get(*var_slot) {
                Some(Val::Node(vi)) => graph.remove_vertex_label(*vi, label),
                Some(Val::Edge(ei)) => graph.remove_edge_label(*ei, label),
                _ => {}
            },
        }
    }
}

pub(super) fn run_delete(
    graph: &mut Graph,
    ctx: &Ctx,
    detach: bool,
    targets: &[CExpr],
    binding: &Binding,
) -> CodeResult<()> {
    for target in targets {
        let v = {
            let env = Env::new(graph, ctx, binding);
            eval(&env, target)
        };
        match v {
            Val::Edge(ei) => graph.remove_edge(ei),
            Val::Node(vi) => graph.remove_vertex(vi, detach)?,
            _ => {}
        }
    }
    Ok(())
}

/// Keep only rows whose key passes `keep`, into a fresh flat RowSet.
pub(super) fn filter_rows(rs: RowSet, mut keep: impl FnMut(&str) -> bool) -> RowSet {
    let mut out = RowSet::new(rs.cols.clone());
    for r in rs.rows() {
        if keep(&value_row_key(r)) {
            out.push_row(r.iter().cloned());
        }
    }
    out
}

pub(super) fn distinct_rows(rs: RowSet) -> RowSet {
    let mut seen = HashSet::new();
    filter_rows(rs, |k| seen.insert(k.to_string()))
}

/// Dedup key over the first `n` output slots of a projected binding (the inline
/// body's output columns). Element-aware via [`val_key`], so two rows are equal
/// iff their columns hold the same scalars / the same node/edge handles.
pub(super) fn binds_row_key(b: &Binding, n: usize) -> String {
    let mut s = String::new();
    for i in 0..n {
        match b.get(i) {
            Some(v) => val_key(v, &mut s),
            None => s.push('\u{2}'),
        }
        s.push('\u{1}');
    }
    s
}

pub(super) fn distinct_binds(rows: Vec<Binding>, n: usize) -> Vec<Binding> {
    let mut seen = HashSet::new();
    rows.into_iter()
        .filter(|b| seen.insert(binds_row_key(b, n)))
        .collect()
}

/// Set-op fold over inline-body result bindings (the binding twin of [`combine`]),
/// keeping element identity. `n` is the output column count. Mirrors the
/// top-level `combine` semantics exactly (first-seen distinct, ALL keeps dups).
pub(super) fn combine_binds(
    op: SetOp,
    left: Vec<Binding>,
    right: Vec<Binding>,
    n: usize,
) -> Vec<Binding> {
    match op.op {
        SetOpKind::Union => {
            let mut all = left;
            all.extend(right);
            if op.all {
                all
            } else {
                distinct_binds(all, n)
            }
        }
        SetOpKind::Except => {
            let rk: HashSet<String> = right.iter().map(|b| binds_row_key(b, n)).collect();
            let kept: Vec<Binding> = left
                .into_iter()
                .filter(|b| !rk.contains(&binds_row_key(b, n)))
                .collect();
            if op.all {
                kept
            } else {
                distinct_binds(kept, n)
            }
        }
        SetOpKind::Intersect => {
            let rk: HashSet<String> = right.iter().map(|b| binds_row_key(b, n)).collect();
            let kept: Vec<Binding> = left
                .into_iter()
                .filter(|b| rk.contains(&binds_row_key(b, n)))
                .collect();
            if op.all {
                kept
            } else {
                distinct_binds(kept, n)
            }
        }
    }
}

pub(super) fn combine(op: SetOp, left: RowSet, right: RowSet) -> RowSet {
    let right_keys: HashSet<String> = right.rows().map(value_row_key).collect();
    match op.op {
        SetOpKind::Union => {
            let mut all = RowSet::new(left.cols.clone());
            for r in left.rows().chain(right.rows()) {
                all.push_row(r.iter().cloned());
            }
            if op.all {
                all
            } else {
                distinct_rows(all)
            }
        }
        SetOpKind::Except => {
            let kept = filter_rows(left, |k| !right_keys.contains(k));
            if op.all {
                kept
            } else {
                distinct_rows(kept)
            }
        }
        SetOpKind::Intersect => {
            let kept = filter_rows(left, |k| right_keys.contains(k));
            if op.all {
                kept
            } else {
                distinct_rows(kept)
            }
        }
    }
}

/// Map a deferred-check failure at commit into the coded error the per-binding
/// gates used to raise inline, so the surfaced `ConstraintViolation` (and its
/// message) is unchanged whether a single statement checks eagerly or at commit.
pub(super) fn tx_commit_error(e: TxCommitError) -> CodeError {
    match e {
        TxCommitError::Required => CodeError::new(
            ErrorCode::ConstraintViolation,
            "write violates a required-property constraint (a required key is missing, null, or being removed)",
        ),
        TxCommitError::Type => CodeError::new(
            ErrorCode::ConstraintViolation,
            "write violates a type constraint (a value is not of the declared scalar type)",
        ),
        TxCommitError::Unique => CodeError::new(
            ErrorCode::ConstraintViolation,
            "write would duplicate a value under a unique constraint (use _MERGE to upsert)",
        ),
        TxCommitError::Cardinality => CodeError::new(
            ErrorCode::ConstraintViolation,
            "write violates a cardinality constraint (a vertex's edge degree is outside its declared min..max bound)",
        ),
        // A custom validator carries its own error verbatim — a `ConstraintViolation`
        // for a definite-`false` predicate, or an evaluation fault's own code.
        TxCommitError::Validator(e) => e,
        // A graph-level invariant carries its own error verbatim — a
        // `ConstraintViolation` for a `false` result cell, or an evaluation fault.
        TxCommitError::Invariant(e) => e,
        // A malformed label / edge type / property key introduced by a write.
        TxCommitError::MalformedName(e) => e,
        TxCommitError::NoTx => {
            CodeError::new(ErrorCode::InvalidGraphOp, "commit called with no open transaction")
        }
    }
}

/// Close a statement's auto-commit frame: on success commit (running the deferred
/// constraint checks — a failure has already rolled the statement's writes back);
/// on error roll the statement's partial writes back. This gives every top-level
/// statement per-statement atomicity: a faulting INSERT/SET/DELETE leaves no trace.
pub(super) fn finish_statement<T>(
    graph: &mut Graph,
    result: CodeResult<T>,
    mark: usize,
) -> CodeResult<T> {
    match result {
        Ok(v) => match graph.commit_tx() {
            Ok(()) | Err(TxCommitError::NoTx) => Ok(v),
            Err(e) => Err(tx_commit_error(e)),
        },
        Err(err) => {
            // Undo only this statement's writes and close only this frame. An
            // enclosing explicit transaction stays open, so a caught error does not
            // silently drop the caller out of its transaction (which would then
            // auto-commit every later write and make the closing rollback a no-op).
            graph.rollback_statement(mark);
            Err(err)
        }
    }
}

/// Execute a lowered plan against a graph with positional params, inside a
/// per-statement auto-commit transaction frame (see [`finish_statement`]). Nesting
/// joins an outer explicit transaction opened over the FFI boundary — the inner
/// commit is a no-op and the outermost commit runs the deferred checks.
pub(super) fn run_cquery(plan: &CQuery, graph: &mut Graph, params: &[Val]) -> CodeResult<RowSet> {
    let mark = graph.tx_undo_mark();
    graph.begin_tx();
    let result = run_cquery_body(plan, graph, params);
    finish_statement(graph, result, mark)
}

/// The statement body — runs each linear part and combines set-op results. Its
/// writes apply eagerly inside the frame [`run_cquery`] opened; a fault propagates
/// out and rolls them back.
/// Eagerly reject any unknown/unimplemented function the plan references, BEFORE
/// running a single row. An unknown function is never valid regardless of row
/// count, so an empty result set must fault exactly like a non-empty one (the
/// per-row `FAULT_UNKNOWN_FN` path would otherwise never fire over zero rows and
/// silently return no rows). Surfaced here at the execute entry — the prepare
/// entry returns only a `SyntaxError`, so the coded fault is raised before the
/// first `run_part`. Matches the TS engine's compile-time `assertKnownScalarFn`.
pub(super) fn check_unknown_fns(unknown_fns: &[String]) -> CodeResult<()> {
    if unknown_fns.is_empty() {
        return Ok(());
    }
    let names = unknown_fns
        .iter()
        .map(|n| format!("{n}()"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(CodeError::new(
        ErrorCode::UnknownFunction,
        format!("call to an unknown or unimplemented function: {names}"),
    ))
}

/// Fault on a `GROUP BY` naming something no clause binds.
///
/// ISO GQL's `groupingElement` is a `bindingVariableReference`, so the name has
/// to be bound already. An unbound one used to read as NULL, which keys every
/// row the same: the query silently returned ONE group holding the first row's
/// values instead of the grouping the reader asked for.
///
/// The message names `LET`/`WITH` because the usual way in is a RETURN alias —
/// `RETURN n.t AS t … GROUP BY t` — and the fix is to bind the value BEFORE the
/// projection, which is how the ISO examples spell it.
pub(super) fn check_unbound_group_keys(unbound: &[String]) -> CodeResult<()> {
    if unbound.is_empty() {
        return Ok(());
    }

    let names = unbound.join(", ");

    Err(CodeError::new(
        ErrorCode::UnknownFunction,
        format!(
            "GROUP BY references an unbound variable: {names}. GROUP BY groups the \
             INPUT bindings, so it cannot see a RETURN alias (ORDER BY, which runs \
             after the projection, can). Bind it first — `LET {names} = …` or \
             `WITH … AS {names}` — or group by the property directly."
        ),
    ))
}

/// Push every `CCount::Param` slot referenced by a projection anywhere in the
/// clause list — including nested CALL-subquery bodies — into `out`.
pub(super) fn collect_count_param_slots(clauses: &[CClause], out: &mut Vec<usize>) {
    for clause in clauses {
        match clause {
            CClause::With { projection, .. } | CClause::Return(projection) => {
                for b in [&projection.skip, &projection.limit].into_iter().flatten() {
                    if let CCount::Param(slot) = b {
                        out.push(*slot);
                    }
                }
            }
            CClause::CallInline {
                body, body_more, ..
            } => {
                collect_count_param_slots(&body.clauses, out);
                for (_, part) in body_more {
                    collect_count_param_slots(&part.clauses, out);
                }
            }
            _ => {}
        }
    }
}

/// Eagerly validate every `LIMIT` / `OFFSET` `$param` bound: its value must be a
/// non-negative integer. Checked BEFORE any row is produced, so a bad bound faults
/// identically over zero rows or many — mirroring the TS engine's up-front check
/// in `compile`. A missing bound param is already caught by `positional`.
pub(super) fn check_count_params(plan: &CQuery, params: &[Val]) -> CodeResult<()> {
    let mut slots = Vec::new();
    for part in &plan.parts {
        collect_count_param_slots(&part.clauses, &mut slots);
    }
    for slot in slots {
        let v = params.get(slot);
        let ok = matches!(v, Some(Val::Num(n)) if n.is_finite() && n.fract() == 0.0 && *n >= 0.0);
        if !ok {
            return Err(CodeError::new(
                ErrorCode::InvalidValue,
                "a LIMIT/OFFSET parameter must resolve to a non-negative integer",
            ));
        }
    }
    Ok(())
}

pub(super) fn run_cquery_body(
    plan: &CQuery,
    graph: &mut Graph,
    params: &[Val],
) -> CodeResult<RowSet> {
    check_unknown_fns(&plan.unknown_fns)?;
    check_unbound_group_keys(&plan.unbound_group_keys)?;
    check_count_params(plan, params)?;
    if has_nested_aggregate(plan) {
        return Err(CodeError::new(
            ErrorCode::Unsupported,
            "aggregate functions cannot be nested",
        ));
    }
    if has_argless_aggregate(plan) {
        return Err(CodeError::new(
            ErrorCode::Unsupported,
            "aggregate function requires an argument (only count(*) is argless)",
        ));
    }
    let first = plan
        .parts
        .first()
        .ok_or_else(|| CodeError::new(ErrorCode::Syntax, "empty query"))?;
    let mut rs = run_part(first, graph, plan, params)?;
    for (i, op) in plan.ops.iter().enumerate() {
        let right = run_part(&plan.parts[i + 1], graph, plan, params)?;
        rs = combine(*op, rs, right);
    }
    Ok(rs)
}

/// Does any MATCH in this part carry a path selector (`ANY SHORTEST`)? Such a
/// part must take the general scalar driver, which is the only one that honors it.
/// True if any MATCH carries a path selector (`ANY`/`ALL SHORTEST`) or a
/// non-default path mode (`SIMPLE`/`ACYCLIC`/`WALK`). Both are implemented only in
/// the general scalar driver; the count / vectorized / parallel fast paths below
/// enumerate trails (edge-uniqueness), which is wrong for either.
pub(super) fn linear_needs_general_matcher(linear: &CLinear) -> bool {
    linear.clauses.iter().any(|c| {
        matches!(c, CClause::Match { patterns, .. }
        if patterns.iter().any(|p| p.selector != PathSelector::Walk
            || p.mode != PathMode::Trail
            // A bound path variable needs the general matcher — only it builds
            // the Path value (via `all_walk`/`shortest_walk`).
            || p.path_var_slot.is_some()
            // A per-hop edge predicate on a quantified segment is evaluated only
            // by the general matcher's `reachable_each`; the count / vectorized
            // shortcuts count or scan without it and would over-count.
            || p.segments.iter().any(|s| {
                s.rel.quantifier.is_some()
                    && (!s.rel.props.is_empty() || s.rel.where_.is_some())
            })))
    })
}

/// Run one linear part: try the fully-vectorized pipeline executor first (it
/// handles read-only `MATCH … WITH … RETURN` chains end-to-end), else the scalar
/// binding-based driver.
pub(super) fn run_part(
    linear: &CLinear,
    graph: &mut Graph,
    plan: &CQuery,
    params: &[Val],
) -> CodeResult<RowSet> {
    // A path selector (`ANY`/`ALL SHORTEST`) or a non-default mode
    // (`SIMPLE`/`ACYCLIC`/`WALK`) is only implemented in the general scalar driver.
    // Skip every count / vectorized / parallel fast path below — they enumerate
    // trails (edge-uniqueness) or ignore the selector, wrong for either.
    if linear_needs_general_matcher(linear) {
        return run_linear(linear, graph, plan, params);
    }
    // Unbounded var-length with a DISTINCT result → BFS the reachable set instead of
    // enumerating trails (which is exponential and hits the trail budget / faults).
    if let Some(res) = try_reachable_distinct(linear, graph, plan, params) {
        return res;
    }
    // A count over BARE hops needs no rows at all — walk and fold in place
    // (`seek::walk_count`, the same fold Gremlin's `.count()` uses). Tried ahead
    // of everything below because those all build the rows this never needs:
    // even the parallel counter enumerates, it just does so on more cores.
    if let Some(res) = try_walk_count(linear, graph, plan, params) {
        return res;
    }
    // A comma join off one start is a PRODUCT of its branches, not a cross
    // product to enumerate.
    if let Some(res) = try_count_comma_join(linear, graph, plan, params) {
        return res;
    }
    // …and the same fold, grouped: one row per GROUP instead of one per walk.
    if let Some(res) = try_grouped_walk_count(linear, graph, plan, params) {
        return res;
    }
    // Intra-query parallel count over a traversal (opt-in `parallel-query`). Tried
    // before the vectorized pipeline: for a pure `count(*)` over a multi-hop or
    // filtered traversal the vectorized path *materializes* every intermediate row
    // into a frame just to count it, whereas this streams the walk across all
    // cores with per-thread counters — no materialization. Only fires above a seed
    // threshold, so small queries still take the vectorized/scalar path below.
    #[cfg(feature = "parallel-query")]
    if let Some(res) = try_parallel_count(linear, graph, plan, params) {
        return res;
    }
    // General parallel aggregation over a traversal (group-by / sum / avg / …) —
    // the scalar aggregating path stream-folds one match at a time; this splits
    // the seed loop across cores with per-thread accumulators merged in seed order.
    #[cfg(feature = "parallel-query")]
    if let Some(res) = try_parallel_agg(linear, graph, plan, params) {
        return res;
    }
    // Parallel row materialization over a traversal: the vectorized builder below
    // enumerates the whole join into columns on one thread, whereas this splits the
    // seed loop across cores, each building + projecting its slice, then concats.
    #[cfg(feature = "parallel-query")]
    if let Some(res) = try_parallel_scan(linear, graph, plan, params) {
        return res;
    }
    if use_vec() {
        if let Some(rs) = vectorized_linear(linear, graph, plan, params) {
            return Ok(rs);
        }
    }
    run_linear(linear, graph, plan, params)
}

/// Typed Arrow fast path: a single fresh `MATCH` + plain `RETURN` (no WITH /
/// aggregate / DISTINCT / ORDER BY / `*`). Produces Arrow columns straight from
/// the vectorized `Col`s, so numeric/bool columns skip the `Val`→`Value` boxing
/// the RowSet path would do. Returns `(columns, nrows)` or `None` to fall back.
#[cfg(feature = "arrow")]
pub(super) fn vectorized_arrow(
    graph: &Graph,
    ctx: &Ctx,
    matches: &[&CClause],
    proj: &CProjection,
) -> Option<(Vec<ArrowColumn>, usize)> {
    if matches.len() != 1
        || proj.star
        || proj.aggregating
        || proj.distinct
        || !proj.order_by.is_empty()
    {
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
    let cap = where_
        .is_none()
        .then(|| proj.limit_val(ctx).map(|l| proj.skip_val(ctx) + l))
        .flatten();
    // An index hint (vertex or edge) makes the scan a seek, so the LIMIT cap
    // can't early-stop it — drop the cap when a hint applies.
    let cap = if scan_is_hinted(graph, ctx, path, where_.as_ref()) {
        None
    } else {
        cap
    };
    let mut sc = build_scan(graph, ctx, path, *scope_len, cap, where_.as_ref(), None)?;
    if let Some(w) = where_ {
        let keep: Vec<bool> = eval_vec(graph, ctx, &sc, w)
            .into_truth()
            .iter()
            .map(|t| *t == Some(true))
            .collect();
        compact(&mut sc, &keep);
    }
    let start = proj.skip_val(ctx).min(sc.n);
    let end = proj
        .limit_val(ctx)
        .map(|l| (start + l).min(sc.n))
        .unwrap_or(sc.n);
    let cols = proj
        .items
        .iter()
        .map(|it| {
            eval_vec(graph, ctx, &sc, &it.expr)
                .page(start, end)
                .into_arrow(graph)
        })
        .collect();
    Some((cols, end - start))
}

/// Execute a plan and return an Arrow columnar blob. Uses the typed boxing-free
/// fast path for a single-part `MATCH … RETURN`; otherwise runs the normal
/// executor and converts its `RowSet` (correct for aggregate / WITH / UNION /
/// scalar — just not boxing-free).
#[cfg(feature = "arrow")]
pub(super) fn run_cquery_arrow(
    plan: &CQuery,
    graph: &mut Graph,
    params: &[Val],
) -> CodeResult<Vec<u8>> {
    check_unknown_fns(&plan.unknown_fns)?;
    check_unbound_group_keys(&plan.unbound_group_keys)?;
    if has_nested_aggregate(plan) {
        return Err(CodeError::new(
            ErrorCode::Unsupported,
            "aggregate functions cannot be nested",
        ));
    }
    if has_argless_aggregate(plan) {
        return Err(CodeError::new(
            ErrorCode::Unsupported,
            "aggregate function requires an argument (only count(*) is argless)",
        ));
    }
    if use_vec() && plan.ops.is_empty() && plan.parts.len() == 1 {
        let linear = &plan.parts[0];
        if let Some((CClause::Return(proj), rest)) = linear.clauses.split_last() {
            if rest.iter().all(|c| {
                matches!(
                    c,
                    CClause::Match {
                        optional: false,
                        ..
                    }
                )
            }) {
                let ctx = resolve_ctx(graph, plan, params);
                let matches: Vec<&CClause> = rest.iter().collect();
                if let Some((cols, nrows)) = vectorized_arrow(graph, &ctx, &matches, proj) {
                    // A recorded data exception can't return Err from the typed
                    // fast path; fall through to the scalar path (read-only shape,
                    // safe to re-run), which surfaces the CodeError.
                    if !ctx.faulted() {
                        return Ok(crate::arrow::to_arrow_cols(&proj.out_names, &cols, nrows));
                    }
                }
            }
        }
    }
    let rs = run_cquery(plan, graph, params)?;
    Ok(crate::arrow::to_arrow(&rs))
}

/// Bind named params into the plan's positional slot order. A `$name` the query
/// references but the caller didn't supply is an error (not a silent NULL) — a
/// missing binding is a programming mistake, so fail loud. Mirrors the TS
/// engine's eager check.
pub(super) fn positional(param_names: &[String], params: &Params) -> CodeResult<Vec<Val>> {
    param_names
        .iter()
        .map(|n| {
            match params.get(n).cloned() {
                Some(v) => Ok(v),
                // The reserved `$__now` (from a bare `current_*` function) is
                // optional: if the host didn't supply a `now`, it reads as NULL
                // (so `current_date` → null) rather than a missing-param error.
                None if n == "__now" => Ok(Val::Null),
                None => Err(CodeError::new(
                    ErrorCode::MissingParameter,
                    format!("missing parameter: ${n}"),
                )),
            }
        })
        .collect()
}
