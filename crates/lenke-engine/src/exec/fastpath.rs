use super::aggregation::*;
use super::evaluator::*;
use super::*;
use crate::batch::{Batch, Col};
use crate::gstr::GStr;
use crate::ir::{Expr, Plan};
use crate::store::{Column, Store};
use crate::value::{self, Value};

/// Fused `count(*)` over `Filter(<numeric conj on the frontier>, Expand(Scan))` — the
/// single-hop analogue of [`try_filtered_count`]'s streaming. Sweep the type's edges (per-type
/// CSR) and test the numeric predicate INLINE against the neighbour's column, counting per
/// (src, nbr) PATH — no `[src, nbr]` batch, no keep list, no separate filter pass. Byte-
/// identical: the same typed compare per path as the materialize path (multiplicity kept,
/// present-null dropped). Declined for a reverse-seedable (selective) predicate — the
/// `reverse_seed_worth` guard prices the reverse walk against the SOURCE scan (node count),
/// which over-fires for a sparse type, so `reverse_seed_decide` returning `Some` here means the
/// endpoint is genuinely more selective than sweeping the type's edges forward.
pub(super) fn try_fused_hop_num_count(input: &Plan, store: &Store) -> Option<u64> {
    let (label, w, pred, exp) = fused_hop_shape(input, store)?;
    let (key, bounds) = num_conj_on_slot(pred, 1)?;
    let Some(Column::Num { data, present, .. }) = store.column(&key) else {
        return None;
    };
    if reverse_seed_decide(pred, exp, store, false).is_some() {
        return None;
    }
    let mut count = 0u64;
    for_each_typed_out(store, label, w, |nbr| {
        let j = nbr as usize;
        if present[j] && bounds.iter().all(|&(op, t)| num_pred(op, data[j], t)) {
            count += 1;
        }
    })?;
    Some(count)
}

/// Fused numeric-filtered PROJECTION over `Project(Filter(<numeric conj>, Expand(Scan)))`.
/// The general path pulls the whole expand into an `[src, nbr]` batch, filters, and GATHERS
/// the survivors — a fixed ~0.8ms for an 80k-edge hop no matter how few rows survive, which
/// loses to the TS engine's streaming when the filter is mid-selective (survivors ≪ edges) but not
/// selective enough to reverse-seed. Instead STREAM the type's edges (per-type CSR), test the
/// numeric predicate inline, collect just the surviving TARGET ids, and evaluate the projection
/// over that survivor frontier — the survivor count of output rows, never the `[src, nbr]`
/// intermediate. Byte-identical: same typed test per (src, nbr) PATH, survivors in the same
/// (source, out_adj) order the expand emits, projection unchanged. `None` unless every projected
/// item reads only the endpoint (slot 1), lineage-free, single-type Out hop, per-type CSR fresh,
/// and the endpoint is not reverse-seedable.
pub(super) fn try_fused_hop_project(
    input: &Plan,
    items: &[(String, Expr)],
    store: &Store,
    track: bool,
) -> Option<Batch> {
    if track || items.is_empty() {
        return None;
    }
    let (label, w, pred, exp) = fused_hop_shape(input, store)?;
    if !items.iter().all(|(_, e)| refs_only_slot(e, 1)) {
        return None;
    }
    let (key, bounds) = num_conj_on_slot(pred, 1)?;
    let Some(Column::Num { data, present, .. }) = store.column(&key) else {
        return None;
    };
    if reverse_seed_decide(pred, exp, store, false).is_some() {
        return None;
    }
    // Stream the type's edges, collecting just the surviving TARGET ids (output-proportional,
    // never the `[src, nbr]` intermediate); then evaluate the projection over that frontier.
    let mut survivors: Vec<u32> = Vec::new();
    for_each_typed_out(store, label, w, |nbr| {
        let j = nbr as usize;
        if present[j] && bounds.iter().all(|&(op, t)| num_pred(op, data[j], t)) {
            survivors.push(nbr);
        }
    })?;
    // Evaluate the projection over the survivor frontier (endpoint at slot 1).
    let cols = vec![
        Col::Nodes(vec![0u32; survivors.len()]),
        Col::Nodes(survivors),
    ];
    let out = eval_all(items.iter().map(|(_, e)| e), store, &Batch::of(cols)).ok()?;
    Some(Batch::of(out))
}

/// Fused scalar aggregate — `count(*)` / `sum` / `min` / `max` — over
/// `Filter(<any endpoint-only pred>, Expand(Scan))` for a predicate the inline numeric count
/// can't take (an OR, a mixed-key disjunction, a string search). Sweep the type's edges off
/// the per-type CSR into just the TARGET-id column, run the SAME vectorized `eval_mask` the
/// materialize path would, and fold the aggregate over the TRUE cells — skipping the
/// `[src, nbr]` batch the general Aggregate builds AND the keep-gather it discards. Byte-
/// identical: the mask is per (src, nbr) PATH exactly as the materialize filter, and the flat
/// partition is in the SAME (source, out_adj) order the expand emits, so a float `sum` folds in
/// the identical order. Declined for a reverse-seedable (selective) endpoint.
pub(super) fn try_fused_hop_mask_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct {
        return None;
    }
    // count(*) has no arg; sum/min/max fold a Num property of the frontier (slot 1).
    let arg_key: Option<&String> = match (&agg.func, agg.arg.as_ref()) {
        (AggFn::Count, None) => None,
        (AggFn::Sum | AggFn::Min | AggFn::Max, Some(Expr::Prop { slot: 1, key })) => Some(key),
        _ => return None,
    };
    let (label, w, pred, exp) = fused_hop_shape(input, store)?;
    if !refs_only_slot(pred, 1) {
        return None;
    }
    // The agg property must be a plain Num column (min/max/sum semantics; a NULL cell is
    // skipped, matching the general aggregate).
    let agg_col: Option<(&[f64], &[bool])> = match arg_key {
        Some(k) => match store.column(k)? {
            Column::Num { data, present, .. } => Some((data, present)),
            _ => return None,
        },
        None => None,
    };
    if reverse_seed_decide(pred, exp, store, false).is_some() {
        return None;
    }
    let mut targets: Vec<u32> = Vec::new();
    for_each_typed_out(store, label, w, |nbr| targets.push(nbr))?;
    // Frontier at slot 1; slot 0 is a dummy the endpoint-only predicate never reads.
    let cols = vec![Col::Nodes(vec![0u32; targets.len()]), Col::Nodes(targets)];
    let batch = Batch::of(cols);
    let mask = eval_mask(pred, store, &batch).ok()?;
    let Col::Nodes(targets) = batch.slot(1) else {
        return None;
    };
    let is_true = |i: usize| mask.get(i) == Some(&Some(true));
    match (&agg.func, agg_col) {
        (AggFn::Count, _) => {
            let c = (0..targets.len()).filter(|&i| is_true(i)).count();
            Some(scalar_num(c as f64))
        }
        (AggFn::Sum, Some((data, present))) => {
            let mut total = 0f64;
            for (i, &t) in targets.iter().enumerate() {
                let j = t as usize;
                if is_true(i) && present[j] {
                    total += data[j];
                }
            }
            Some(scalar_num(total))
        }
        (AggFn::Min | AggFn::Max, Some((data, present))) => {
            let want_min = matches!(agg.func, AggFn::Min);
            let mut best: Option<f64> = None;
            for (i, &t) in targets.iter().enumerate() {
                let j = t as usize;
                if is_true(i) && present[j] {
                    let x = data[j];
                    best = Some(match best {
                        None => x,
                        Some(b) => {
                            let ord = value::cmp_num_total(x, b);
                            if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                                x
                            } else {
                                b
                            }
                        }
                    });
                }
            }
            Some(Batch::single(Col::Gen(vec![
                best.map_or(Value::Null, Value::Num)
            ])))
        }
        _ => None,
    }
}

/// Count nodes of `label` whose `pred` holds, STREAMING the label bucket with raw
/// f64 compares — no scan-id materialization, no keep vector. Handles a single
/// `prop OP num` compare and a same-column numeric range (`lo <= x AND x < hi`), the
/// hot filtered-count shapes; `None` for anything else (the caller materializes and
/// runs the general filter). Every survivor test matches `try_filter_keep`'s typed
/// paths exactly (present gates NULL; a NaN cell fails ordering → dropped), so the
/// count is identical.
/// Recognize a filter predicate that is a CONJUNCTION of numeric compares all on the
/// SAME property of one `slot` — `prop OP num` (either operand order) — returning
/// `(key, bounds)`. Shared by the streaming node/edge count fast paths; `None` for a
/// string / disjunction / multi-slot / multi-key / non-numeric predicate.
pub(super) fn num_conj_on_slot(
    pred: &Expr,
    slot: usize,
) -> Option<(String, Vec<(CompareOp, f64)>)> {
    // An atom on `slot` is either a numeric compare (a bound) or a `PropertyExists`
    // presence gate (NO bound — redundant with the streaming count's own `present[i]`
    // check, and implied by any compare on the same key). `has(k, pred)` desugars to
    // `And(PropertyExists{k}, <compare>)`, so accepting the presence atom is what keeps
    // a non-selective `has('age', neq(60)).count()` on the streaming path instead of
    // materializing a 99% keep-list.
    let atom = |e: &Expr| -> Option<(String, Option<(CompareOp, f64)>)> {
        match e {
            Expr::PropertyExists { slot: s, key } if *s == slot => Some((key.clone(), None)),
            Expr::Compare { op, left, right } => {
                let (key, op, lit) = match (left.as_ref(), right.as_ref()) {
                    (Expr::Prop { slot: s, key }, Expr::Lit(v)) if *s == slot => {
                        (key.clone(), *op, v)
                    }
                    (Expr::Lit(v), Expr::Prop { slot: s, key }) if *s == slot => {
                        (key.clone(), flip_op(*op), v)
                    }
                    _ => return None,
                };
                match lit {
                    Value::Num(t) => Some((key, Some((op, *t)))),
                    _ => None,
                }
            }
            _ => None,
        }
    };
    let mut conjuncts = Vec::new();
    flatten_and(pred, &mut conjuncts);
    let mut key0: Option<String> = None;
    let mut bounds: Vec<(CompareOp, f64)> = Vec::with_capacity(conjuncts.len());
    for c in &conjuncts {
        let (key, bound) = atom(c)?;
        match &key0 {
            Some(k) if *k != key => return None, // a second key can't stream one column
            _ => key0 = Some(key),
        }
        if let Some(b) = bound {
            bounds.push(b);
        }
    }
    Some((key0?, bounds))
}

/// Answer `count(*)` over `Filter(edge-pred, Expand{bind_edge})` by STREAMING the
/// expansion — for each source, test each matching out-edge's property inline and
/// count — instead of materializing every `(source, edge, target)` row and filtering
/// (an O(edges) Batch). Edge properties are boxed (a per-key eid→Value map), so the
/// per-edge lookup stays, but the row materialization is what dominated. The survivor
/// test matches the general Filter (a present Num edge prop tests the bounds;
/// null/non-numeric → UNKNOWN → dropped), so the count is identical. Only the pred on
/// the bound EDGE slot (not the target node) is handled; anything else falls through.
pub(super) fn try_edge_filtered_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    let Plan::Filter {
        input: expand,
        pred,
    } = input
    else {
        return None;
    };
    let Plan::Expand {
        input: src,
        from,
        dir,
        edge_label,
        bind_edge,
        double_loops: false,
    } = expand.as_ref()
    else {
        return None;
    };
    if !bind_edge {
        return None; // the edge must be bound for the filter to read its property
    }
    // A bind_edge Expand appends the edge at the slot just past its input (then the
    // target node); the pred must be a numeric conjunction on that edge slot.
    let edge_slot = from + 1;
    let (key, bounds) = num_conj_on_slot(pred, edge_slot)?;
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no rows
    };
    let src_ids = frontier_ids(src, store)?;
    let mut count = 0u64;
    // Typed overlay: read the edge property as a raw f64 (no per-edge hash probe +
    // Value unbox). Falls back to the boxed edge_prop when the overlay is stale or the
    // key is not homogeneously numeric.
    if let Some((data, present)) = store.edge_num_column(&key) {
        for &v in &src_ids {
            for_each_nbr(store, v, *dir, &want, false, |_nbr, eid| {
                let i = eid as usize;
                if present[i] && bounds.iter().all(|&(op, t)| num_pred(op, data[i], t)) {
                    count += 1;
                }
            });
        }
    } else {
        for &v in &src_ids {
            for_each_nbr(store, v, *dir, &want, false, |_nbr, eid| {
                if let Value::Num(x) = store.edge_prop(eid, &key) {
                    if bounds.iter().all(|&(op, t)| num_pred(op, x, t)) {
                        count += 1;
                    }
                }
            });
        }
    }
    Some(scalar_num(count as f64))
}

/// A searched `CASE` that is a categorical remap: EVERY branch condition is
/// `<dict col> = <string literal>` on ONE key. Returns the slot, key, and a code →
/// first-matching-branch-index table (`None` where no branch matches that dict value).
pub(super) fn case_dict_lookup(
    branches: &[(Expr, Expr)],
    store: &Store,
) -> Option<(usize, String, Vec<Option<usize>>)> {
    if branches.is_empty() {
        return None;
    }
    let mut slot_key: Option<(usize, String)> = None;
    let mut lits: Vec<&str> = Vec::with_capacity(branches.len());
    for (cond, _) in branches {
        let Expr::Compare {
            op: CompareOp::Eq,
            left,
            right,
        } = cond
        else {
            return None;
        };
        let (s, k, v) = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot, key }, Expr::Lit(Value::Str(v)))
            | (Expr::Lit(Value::Str(v)), Expr::Prop { slot, key }) => (*slot, key, v),
            _ => return None,
        };
        match &slot_key {
            None => slot_key = Some((s, k.clone())),
            Some((s0, k0)) if *s0 == s && k0 == k => {}
            Some(_) => return None,
        }
        lits.push(v.as_ref());
    }
    let (slot, key) = slot_key?;
    let Some(Column::Dict { dict, .. }) = store.column(&key) else {
        return None;
    };
    let code_to_branch: Vec<Option<usize>> = dict
        .iter()
        .map(|dstr| lits.iter().position(|lit| dstr.as_ref() == *lit))
        .collect();
    Some((slot, key, code_to_branch))
}

/// Is `pred` a disjunction (or a single term) of `slot.key == <literal>`, all on ONE
/// key? Returns that key and the literal values — the shape `x IN [a, b, …]` desugars to.
pub(super) fn eq_disjunction_on_slot(pred: &Expr, slot: usize) -> Option<(String, Vec<Value>)> {
    fn collect(e: &Expr, slot: usize, key: &mut Option<String>, vals: &mut Vec<Value>) -> bool {
        match e {
            Expr::Or(a, b) => collect(a, slot, key, vals) && collect(b, slot, key, vals),
            Expr::Compare {
                op: CompareOp::Eq,
                left,
                right,
            } => {
                let (k, v) = match (left.as_ref(), right.as_ref()) {
                    (Expr::Prop { slot: s, key }, Expr::Lit(v))
                    | (Expr::Lit(v), Expr::Prop { slot: s, key })
                        if *s == slot =>
                    {
                        (key, v)
                    }
                    _ => return false,
                };
                if v.is_null() {
                    return false; // a NULL term makes non-matches UNKNOWN, not FALSE
                }
                match key.as_deref() {
                    None => *key = Some(k.clone()),
                    Some(existing) if existing == k => {}
                    Some(_) => return false, // mixed keys — not this shape
                }
                vals.push(v.clone());
                true
            }
            _ => false,
        }
    }
    let (mut key, mut vals) = (None, Vec::new());
    collect(pred, slot, &mut key, &mut vals).then_some(())?;
    Some((key?, vals))
}

/// If every leaf of `pred` that touches a row is `Prop { slot, key }` for ONE `(slot,
/// key)` (no label tests, presence tests, or other columns), return it — the predicate is
/// a pure function of a single property, evaluable once per distinct value.
pub(super) fn sole_prop_ref(pred: &Expr) -> Option<(usize, String)> {
    fn walk(e: &Expr, seen: &mut Option<(usize, String)>) -> bool {
        match e {
            Expr::Lit(_) => true,
            Expr::Prop { slot, key } => match seen {
                None => {
                    *seen = Some((*slot, key.clone()));
                    true
                }
                Some((s0, k0)) => *s0 == *slot && k0 == key,
            },
            Expr::Not(a) => walk(a, seen),
            Expr::And(a, b) | Expr::Or(a, b) | Expr::Xor(a, b) => walk(a, seen) && walk(b, seen),
            Expr::Compare { left, right, .. } => walk(left, seen) && walk(right, seen),
            Expr::Call { args, .. } => args.iter().all(|a| walk(a, seen)),
            Expr::In { needle, haystack } => walk(needle, seen) && walk(haystack, seen),
            _ => false, // other slots, subqueries, label/presence tests → not pure-in-key
        }
    }
    let mut seen = None;
    walk(pred, &mut seen).then_some(())?;
    seen
}

/// [`sole_prop_ref`] constrained to a specific `slot` (returns just the key).
pub(super) fn sole_prop_key(pred: &Expr, slot: usize) -> Option<String> {
    sole_prop_ref(pred)
        .filter(|(s, _)| *s == slot)
        .map(|(_, k)| k)
}

/// Evaluate a single-property predicate for one concrete value of that property, by
/// substituting the literal and folding the now-constant expression. `None` on a faulting
/// eval; `Some(true)` only when the result is definitely TRUE (3VL — the keep condition).
pub(super) fn dict_pred_value(
    pred: &Expr,
    slot: usize,
    key: &str,
    v: &Value,
    store: &Store,
) -> Option<bool> {
    let e = subst_prop(pred, slot, key, v);
    let col = eval(&e, store, &Batch::single(Col::Num(vec![0.0]))).ok()?;
    Some(matches!(col.value_at(0), Value::Bool(true)))
}

/// Evaluate a scalar expression that is a pure function of one DICT column by computing it
/// once per distinct dict value (≤ dict size) and mapping each row to the shared result —
/// so `upper(city)`, `substring(city, …)`, etc. over a categorical column do dict.len()
/// string allocations instead of one per row (the result `Value`s, including `Arc<str>`,
/// are cloned per row = a refcount bump, not a new allocation). Byte-identical: the value
/// is a function of the property alone (absent → the NULL case, computed once).
/// Fold the boxed `Vec<Value>` a scalar function produced into a TYPED column when every
/// cell is the same non-null primitive (`Num`/`Str`/`Bool`). A downstream sort, DISTINCT or
/// GROUP BY on the computed value then takes the typed fast path (raw f64 / `Arc<str>`
/// compare, dict-code dedup) instead of boxing every cell per comparison. A single null or a
/// mixed type keeps it `Gen`; either way `value_at(i)` is byte-identical for every row, so
/// this is purely an internal representation choice — never an observable one.
pub(super) fn typed_col_from_values(out: Vec<Value>) -> Col {
    let Some(first) = out.first() else {
        return Col::Gen(out);
    };
    match first {
        Value::Num(_) if out.iter().all(|v| matches!(v, Value::Num(_))) => Col::Num(
            out.iter()
                .map(|v| {
                    if let Value::Num(x) = v {
                        *x
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ),
        Value::Str(_) if out.iter().all(|v| matches!(v, Value::Str(_))) => Col::Str(
            out.into_iter()
                .map(|v| {
                    if let Value::Str(s) = v {
                        s
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ),
        Value::Bool(_) if out.iter().all(|v| matches!(v, Value::Bool(_))) => Col::Bool(
            out.iter()
                .map(|v| {
                    if let Value::Bool(b) = v {
                        *b
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ),
        _ => Col::Gen(out),
    }
}

pub(super) fn try_eval_dict_scalar(expr: &Expr, store: &Store, batch: &Batch) -> Option<Col> {
    let (slot, key) = sole_prop_ref(expr)?;
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(&key)
    else {
        return None;
    };
    let ev = |v: &Value| -> Option<Value> {
        let e = subst_prop(expr, slot, &key, v);
        Some(
            eval(&e, store, &Batch::single(Col::Num(vec![0.0])))
                .ok()?
                .value_at(0),
        )
    };
    let mut per_code = Vec::with_capacity(dict.len());
    for dv in dict.iter() {
        per_code.push(ev(&Value::Str(dv.clone()))?);
    }
    let null_val = ev(&Value::Null)?;
    Some(Col::Gen(
        ids.iter()
            .map(|&id| {
                if id != u32::MAX && present[id as usize] {
                    per_code[codes[id as usize] as usize].clone()
                } else {
                    null_val.clone()
                }
            })
            .collect(),
    ))
}

/// Keep-list for a predicate that is a pure function of one DICT column: evaluate it once
/// per distinct dict value, then keep rows whose code matches — the projection sibling of
/// [`try_stream_dict_pred_count`], for `WHERE <dict pred> RETURN …`. Byte-identical to the
/// per-row boxed filter (same 3VL TRUE-keeps rule).
pub(super) fn try_filter_keep_dict(
    pred: &Expr,
    store: &Store,
    batch: &Batch,
) -> Option<Vec<usize>> {
    let (slot, key) = sole_prop_ref(pred)?;
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(&key)
    else {
        return None;
    };
    let mut matches = Vec::with_capacity(dict.len());
    for dv in dict.iter() {
        matches.push(dict_pred_value(
            pred,
            slot,
            &key,
            &Value::Str(dv.clone()),
            store,
        )?);
    }
    let null_match = dict_pred_value(pred, slot, &key, &Value::Null, store)?;
    Some(
        ids.iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                let hit = if id != u32::MAX && present[id as usize] {
                    matches[codes[id as usize] as usize]
                } else {
                    null_match
                };
                hit.then_some(i)
            })
            .collect(),
    )
}

/// Replace every `Prop { slot, key }` in `e` with `Lit(value)` — so a predicate that is a
/// pure function of one property becomes a constant expression, evaluable once.
pub(super) fn subst_prop(e: &Expr, slot: usize, key: &str, value: &Value) -> Expr {
    match e {
        Expr::Prop { slot: s, key: k } if *s == slot && k == key => Expr::Lit(value.clone()),
        Expr::Not(a) => Expr::Not(Box::new(subst_prop(a, slot, key, value))),
        Expr::And(a, b) => Expr::And(
            Box::new(subst_prop(a, slot, key, value)),
            Box::new(subst_prop(b, slot, key, value)),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(subst_prop(a, slot, key, value)),
            Box::new(subst_prop(b, slot, key, value)),
        ),
        Expr::Xor(a, b) => Expr::Xor(
            Box::new(subst_prop(a, slot, key, value)),
            Box::new(subst_prop(b, slot, key, value)),
        ),
        Expr::Compare { op, left, right } => Expr::Compare {
            op: *op,
            left: Box::new(subst_prop(left, slot, key, value)),
            right: Box::new(subst_prop(right, slot, key, value)),
        },
        Expr::Call { name, args } => Expr::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| subst_prop(a, slot, key, value))
                .collect(),
        },
        Expr::In { needle, haystack } => Expr::In {
            needle: Box::new(subst_prop(needle, slot, key, value)),
            haystack: Box::new(subst_prop(haystack, slot, key, value)),
        },
        other => other.clone(),
    }
}

/// Streaming count for ANY predicate that is a pure function of one DICT column: evaluate
/// it once per distinct dict value (≤ dict size), then count code membership in a single
/// pass — never materializing rows. Covers `STARTS WITH … OR …`, `CONTAINS`, ranges, and
/// arbitrary boolean combinations over a categorical column. Byte-identical: a row is
/// counted iff the predicate is definitely TRUE for its value (or NULL, evaluated once).
pub(super) fn try_stream_dict_pred_count(
    store: &Store,
    label: &Option<String>,
    pred: &Expr,
) -> Option<u64> {
    let key = sole_prop_key(pred, 0)?;
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(&key)
    else {
        return None;
    };
    let eval_const = |v: &Value| -> Option<bool> {
        let e = subst_prop(pred, 0, &key, v);
        let col = eval(&e, store, &Batch::single(Col::Num(vec![0.0]))).ok()?;
        Some(matches!(col.value_at(0), Value::Bool(true)))
    };
    let mut matches = Vec::with_capacity(dict.len());
    for dv in dict.iter() {
        matches.push(eval_const(&Value::Str(dv.clone()))?);
    }
    let null_match = eval_const(&Value::Null)?;
    let mut count = 0u64;
    scan_visit(store, label, |i| {
        let hit = if present[i] {
            matches[codes[i] as usize]
        } else {
            null_match
        };
        if hit {
            count += 1;
        }
    });
    Some(count)
}

/// Streaming count for categorical membership — `col IN [a, b, …]` (desugared to an
/// OR-chain of equals) on a `Dict` or `Str` column — the string sibling of
/// `try_stream_num_count`. Maps the literals to dict CODES once, then counts matches in
/// one pass over the bucket, never materializing an id vector or keep list. Byte-identical
/// to the OR filter (a literal absent from the dict simply matches nothing).
pub(super) fn try_stream_membership_count(
    store: &Store,
    label: &Option<String>,
    pred: &Expr,
) -> Option<u64> {
    let (key, vals) = eq_disjunction_on_slot(pred, 0)?;
    let mut count = 0u64;
    match store.column(&key)? {
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            // Every literal must be a string (else it can't equal a dict value). Resolve
            // to codes; a literal not in the dict contributes no code (never matches).
            let mut targets: Vec<u32> = Vec::with_capacity(vals.len());
            for v in &vals {
                let Value::Str(s) = v else { return None };
                if let Some(c) = dict.iter().position(|d| d.as_ref() == s.as_ref()) {
                    targets.push(c as u32);
                }
            }
            scan_visit(store, label, |i| {
                if present[i] && targets.contains(&codes[i]) {
                    count += 1;
                }
            });
        }
        Column::Str { data, present, .. } => {
            let mut targets: Vec<&str> = Vec::with_capacity(vals.len());
            for v in &vals {
                let Value::Str(s) = v else { return None };
                targets.push(s.as_ref());
            }
            scan_visit(store, label, |i| {
                if present[i] && targets.iter().any(|t| data[i].as_ref() == *t) {
                    count += 1;
                }
            });
        }
        _ => return None,
    }
    Some(count)
}

/// `DISTINCT`/`dedup()` over `values(<dict col>)`: emit the distinct dict values in
/// FIRST-SEEN order by scanning codes against a `dict.len()` bitset — never decoding or
/// hashing the per-row strings. Byte-identical to the general first-seen dedup (first
/// occurrence of a code == first occurrence of its string). Bails (→ general path) on any
/// absent value or null-sentinel id, whose NULL dedup this fast path doesn't model.
pub(super) fn try_distinct_dict_col(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project { input: pin, items } = input else {
        return None;
    };
    let [(_, Expr::Prop { slot, key })] = items.as_slice() else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(key)
    else {
        return None;
    };
    let frontier = pull(pin, store, false).ok()?;
    // The property may sit on any bound slot — slot 0 for `values(k).dedup()`, but the
    // hop endpoint (e.g. slot 2) for `out().out().values(k).dedup()`.
    let Col::Nodes(ids) = frontier.slot(*slot) else {
        return None;
    };
    let mut seen = vec![false; dict.len()];
    let mut out: Vec<GStr> = Vec::new();
    for &id in ids {
        if id == u32::MAX {
            return None; // a NULL value in the dedup — let the general path handle it
        }
        let i = id as usize;
        if !present[i] {
            return None;
        }
        let c = codes[i] as usize;
        if !seen[c] {
            seen[c] = true;
            out.push(dict[c].clone());
        }
    }
    Some(Batch::of(vec![Col::Str(out)]))
}

/// `MATCH (a)-[]->(b) WHERE b.k <op> a.k RETURN count(*)` — a hop whose survival compares
/// the two ENDPOINTS' numeric properties. Stream the edges and compare the source/neighbor
/// num columns directly, never building the neighbor frontier or a boxed compare. A count
/// is order-free, so byte-identical (NaN / absent → not counted, matching the 3VL filter).
pub(super) fn try_edge_cross_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None;
    }
    let Plan::Filter {
        input: expand,
        pred,
    } = input
    else {
        return None;
    };
    let Plan::Expand {
        input: scan,
        from: 0, // source at slot 0, neighbour appended at slot 1
        dir,
        edge_label,
        bind_edge: false,
        double_loops: _,
    } = expand.as_ref()
    else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    // The predicate is the endpoint compare, optionally AND a neighbour-label test (the
    // `(b:Label)` pattern) — peel that off and check the label per edge.
    let (cmp, nbr_labels): (&Expr, Option<&[String]>) = match pred {
        Expr::Compare { .. } => (pred, None),
        Expr::And(a, b) => match (a.as_ref(), b.as_ref()) {
            (c @ Expr::Compare { .. }, Expr::IsLabeled { slot: 1, labels })
            | (Expr::IsLabeled { slot: 1, labels }, c @ Expr::Compare { .. }) => (c, Some(labels)),
            _ => return None,
        },
        _ => return None,
    };
    // The compare relates two properties, each on slot 0 (source) or slot 1 (neighbour).
    let Expr::Compare { op, left, right } = cmp else {
        return None;
    };
    let (Expr::Prop { slot: ls, key: lk }, Expr::Prop { slot: rs, key: rk }) =
        (left.as_ref(), right.as_ref())
    else {
        return None;
    };
    if *ls > 1 || *rs > 1 {
        return None;
    }
    let (
        Some(Column::Num {
            data: ld,
            present: lp,
            ..
        }),
        Some(Column::Num {
            data: rd,
            present: rp,
            ..
        }),
    ) = (store.column(lk), store.column(rk))
    else {
        return None; // unboxed numeric endpoints only
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)),
    };
    // O(1) neighbour-label check: a membership bitset beats a `is_labeled` binary-search
    // per edge (which made this LOSE to the general path).
    let nbr_bits: Option<Vec<bool>> = nbr_labels.map(|labels| {
        let mut b = vec![false; store.node_count()];
        for l in labels {
            for &id in store.nodes_with_label(l) {
                b[id as usize] = true;
            }
        }
        b
    });
    let mut count = 0u64;
    scan_visit(store, label, |src| {
        let src = src as u32;
        for_each_nbr(store, src, *dir, &want, false, |nbr, _| {
            if let Some(bits) = &nbr_bits {
                if !bits[nbr as usize] {
                    return; // neighbour fails its `(b:Label)` constraint
                }
            }
            let li = if *ls == 0 { src } else { nbr } as usize;
            let ri = if *rs == 0 { src } else { nbr } as usize;
            if lp[li] && rp[ri] && num_pred(*op, ld[li], rd[ri]) {
                count += 1;
            }
        });
    });
    Some(scalar_num(count as f64))
}

pub(super) fn try_stream_num_count(
    store: &Store,
    label: &Option<String>,
    pred: &Expr,
) -> Option<u64> {
    let (key, bounds) = num_conj_on_slot(pred, 0)?;
    let Some(Column::Num {
        data,
        present,
        nulls,
    }) = store.column(&key)
    else {
        return None;
    };
    let mut count = 0u64;
    scan_visit(store, label, |i| {
        // A bare presence gate (`has(k)`, no bounds) counts PRESENCE — a stored
        // present-null included (`present || nulls`). A bounded compare counts only
        // typed values (`present`): a null satisfies no numeric predicate.
        if bounds.is_empty() {
            if present[i] || nulls[i] {
                count += 1;
            }
        } else if present[i] && bounds.iter().all(|&(op, t)| num_pred(op, data[i], t)) {
            count += 1;
        }
    });
    Some(count)
}

/// Answer a scalar `count(*)` over a `VarLength` hop by DFS-counting the emitted
/// paths per source row, WITHOUT materializing the (up to millions of) keep/ends
/// vectors or gathering the input slots — which the general VarLength → Aggregate
/// path builds and immediately discards for a count. Same traversal, edge-type
/// filter and trail bookkeeping as `var_length`, so the count is exact and
/// identical. `None` for a grouped / arg'd / DISTINCT aggregate or a non-`VarLength`
/// input (handled elsewhere).
pub(super) fn try_varlen_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    let Plan::VarLength {
        input: inner,
        from,
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
    if until.is_some() || body_filter.is_some() {
        return None; // an until(pred) walk emits a filtered subset — no closed-form count
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no paths
    };
    let batch = pull(inner, store, false).ok()?;
    let Col::Nodes(src) = batch.slot(*from) else {
        return None;
    };

    // ALGEBRAIC count: for a bounded OUT walk/trail with max<=2, count(*) is the sum of
    // per-hop path counts computed from degrees in O(V+E) — NOT by enumerating the
    // O(paths) walks. 1-hop = the source out-edges; 2-hop = for each source out-edge
    // s->y, the neighbour's out-degree. A TRAIL (no edge reuse) then excludes the one
    // reused-self-loop path s->s->s over the same edge; a WALK (repeat()'s default)
    // permits it, so it makes no correction. Only taken when the enumeration would be
    // the MORE expensive path (a large source set); a filtered / small source stays on
    // the DFS below, where enumeration is already cheap. WALK/TRAIL only: the degree
    // algebra counts node-repeating paths, which SIMPLE / ACYCLIC forbid — those must
    // enumerate via the DFS below.
    let is_trail = matches!(mode, PathMode::Trail);
    if matches!(dir, Dir::Out)
        && *max <= 2
        && *max >= 1
        && matches!(mode, PathMode::Trail | PathMode::Walk)
    {
        let (nc, ec) = (store.node_count(), store.edge_count());
        let avg_deg = if nc == 0 { 0.0 } else { ec as f64 / nc as f64 };
        let est_paths = src.len() as f64 * avg_deg.powi(*max as i32);
        if est_paths > 2.0 * (nc + ec) as f64 {
            let mut outdeg = vec![0u64; nc];
            for (v, d) in outdeg.iter_mut().enumerate() {
                *d = if want.is_empty() {
                    store.out(v as u32).len() as u64
                } else {
                    store
                        .out(v as u32)
                        .iter()
                        .filter(|a| edge_carries_wanted(store, a, &want))
                        .count() as u64
                };
            }
            let mut total: u64 = 0;
            for &s in src {
                for a in store.out(s) {
                    if !edge_carries_wanted(store, a, &want) {
                        continue;
                    }
                    if *min <= 1 {
                        total += 1; // the 1-hop path s -> a.nbr
                    }
                    if *max >= 2 {
                        total += outdeg[a.nbr as usize]; // 2-hop paths s -> a.nbr -> z
                        if is_trail && a.nbr == s {
                            total -= 1; // a trail excludes the reused self-loop s -> s -> s
                        }
                    }
                }
            }
            return Some(scalar_num(total as f64));
        }
    }

    let mut total: u64 = 0;
    let mut used: Vec<u32> = Vec::new();
    let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
    for &v in src {
        if node_unique {
            used.push(v); // mark the start node
        }
        varlen_count_dfs(
            store,
            v,
            0,
            *min,
            *max,
            *dir,
            &want,
            *mode,
            v,
            &mut used,
            &mut total,
            *double_loops,
        );
        if node_unique {
            used.pop();
        }
        debug_assert!(used.is_empty());
    }
    Some(scalar_num(total as f64))
}

/// The shared LEAN iterative walker behind the count/agg var-length fast-paths: like
/// `varlen_walk` but with neither the path stacks nor an emit sink — it just calls
/// `visit(v)` once per "row" the materializing path would emit (every length in
/// `min..=max`, plus each `Close` endpoint). An explicit heap frame stack, so it uses
/// O(1) CALL stack however deep the closure — the recursive twins it replaced went one
/// frame per hop, so a deep count/agg could overflow (or commit the 1 GiB big stack).
#[allow(clippy::too_many_arguments)]
pub(super) fn varlen_scan_walk(
    store: &Store,
    v0: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    double_loops: bool,
    visit: &mut dyn FnMut(u32),
) {
    // Root pre-work: the length-0 source is a row iff `min == 0`; no descent past `max`.
    if min == 0 {
        visit(v0);
    }
    if max == 0 {
        return;
    }
    let drop_loop = matches!(dir, Dir::Both) && !double_loops;
    struct SF {
        v: u32,
        len: u32,
        cursor: usize,
        pending: Option<bool>, // Some(pop_used) once we've descended into a child
    }
    let mut stack = vec![SF {
        v: v0,
        len: 0,
        cursor: 0,
        pending: None,
    }];
    'frames: while let Some(top) = stack.last_mut() {
        if let Some(pop_used) = top.pending.take() {
            if pop_used {
                used.pop();
            }
        }
        let (v, len) = (top.v, top.len);
        loop {
            let Some((is_inc, a)) = adj_nth(store, v, dir, top.cursor) else {
                stack.pop();
                continue 'frames;
            };
            top.cursor += 1;
            if !want.is_empty()
                && !want.iter().any(|&w| {
                    w == a.etype
                        || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
                })
            {
                continue;
            }
            if is_inc && drop_loop && a.nbr == v {
                continue;
            }
            let mark = match varlen_step(mode, start, &a, used) {
                VarStep::Skip => continue,
                VarStep::Close => {
                    if len + 1 >= min {
                        visit(a.nbr);
                    }
                    continue;
                }
                VarStep::Go(mark) => mark,
            };
            // Child pre-work at len+1: it is a row iff in range, then descend unless at
            // `max` (the recursion's `len == max` early return — its mark push/pop around
            // an immediately-returning child cannot affect the tally, so we skip it).
            let clen = len + 1;
            if clen >= min {
                visit(a.nbr);
            }
            if clen == max {
                continue;
            }
            if let Some(m) = mark {
                used.push(m);
            }
            top.pending = Some(mark.is_some());
            stack.push(SF {
                v: a.nbr,
                len: clen,
                cursor: 0,
                pending: None,
            });
            continue 'frames;
        }
    }
}

/// The counting twin of `varlen_dfs`: tallies every row the materializing path would
/// emit. Iterative (see [`varlen_scan_walk`]) so a deep count can't overflow the stack.
#[allow(clippy::too_many_arguments)]
pub(super) fn varlen_count_dfs(
    store: &Store,
    v: u32,
    _len: u32, // always 0 at the call sites — the walk starts at the source
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    total: &mut u64,
    double_loops: bool,
) {
    varlen_scan_walk(
        store,
        v,
        min,
        max,
        dir,
        want,
        mode,
        start,
        used,
        double_loops,
        &mut |_| {
            *total += 1;
        },
    );
}

/// The outcome of the per-hop reuse gate ([`varlen_step`]).
pub(super) enum VarStep {
    /// The hop is forbidden — skip this neighbour.
    Skip,
    /// A SIMPLE closing hop (`nbr == start`): emit the endpoint but do NOT descend —
    /// the cycle is closed, and extending it would repeat an interior node (mirrors
    /// the TS engine's `is_close` early-`continue`).
    Close,
    /// Descend. `Some(id)` is pushed onto the reuse stack before recursing (Trail:
    /// the edge id; Simple/Acyclic: the node id); `None` pushes nothing (Walk).
    Go(Option<u32>),
}

/// The per-hop reuse gate shared by every var-length DFS. Decides whether the hop
/// across `a` is legal under `mode`, and whether it closes a Simple cycle.
///
/// For the node modes `used` is a NODE stack (the driver seeds it with `start`); for
/// Trail it is an EDGE stack. `Simple` permits a hop that closes the cycle on the
/// walk's `start` even though `start` is already marked — that hop emits (via
/// [`VarStep::Close`]) but terminates the path.
#[inline]
pub(super) fn varlen_step(
    mode: PathMode,
    start: u32,
    a: &crate::store::Adj,
    used: &[u32],
) -> VarStep {
    if matches!(mode, PathMode::Simple) && a.nbr == start {
        return VarStep::Close;
    }
    let collide = match mode {
        PathMode::Trail => used.contains(&a.eid),
        PathMode::Simple | PathMode::Acyclic => used.contains(&a.nbr),
        PathMode::Walk => false,
    };
    if collide {
        return VarStep::Skip;
    }
    let mark = match mode {
        PathMode::Walk => None,
        PathMode::Trail => Some(a.eid),
        PathMode::Simple | PathMode::Acyclic => Some(a.nbr),
    };
    VarStep::Go(mark)
}

/// Answer `count(DISTINCT endpoint)` over a bounded var-length hop by MULTI-SOURCE
/// BFS with a visited bitset — O(V+E) — instead of enumerating every path (with its
/// full multiplicity) and deduping the endpoints, which explodes with fan-out. The
/// DISTINCT endpoint set is exactly the nodes at shortest distance in `min..=max`
/// from the source set: a node with ANY walk of length `L ≤ max` has shortest
/// distance ≤ L, so a `min ≤ 1` reachability is the same set whether paths are
/// walks or trails (the shortest path is simple, reusing no edge). That equivalence
/// only holds for `min ≤ 1` (a node discovered at its shortest distance `< min`
/// might still be a valid longer-walk endpoint, which BFS would miss), so deeper
/// lower bounds fall back to the general path. The count is a set size, so it is
/// byte-identical to the TS engine's regardless of visitation order.
pub(super) fn try_varlen_distinct_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    // count(DISTINCT <endpoint slot>) only.
    if agg.func != AggFn::Count || !agg.distinct {
        return None;
    }
    let Some(Expr::Slot(want_slot)) = agg.arg.as_ref() else {
        return None;
    };
    let Plan::VarLength {
        input: inner,
        from,
        dir,
        edge_label,
        min,
        max,
        mode,
        until,
        body_filter,
        double_loops: _, // a distinct endpoint set is blind to edge multiplicity
    } = input
    else {
        return None;
    };
    if until.is_some() || body_filter.is_some() {
        return None; // an until(pred) walk emits a filtered subset — no closed-form count
    }
    // A distinct ENDPOINT set is blind to edge multiplicity, so `double_loops` (a
    // both()-crossed self-loop counted twice) is irrelevant here — the self is reached
    // either way. It is deliberately NOT a bail condition (unlike the multiplicity
    // counts).
    //
    // The set-reachability fusion relies on nodes being allowed to repeat (Walk /
    // Trail). SIMPLE / ACYCLIC forbid node reuse, so a distinct-endpoint count must
    // enumerate — fall through.
    if !matches!(mode, PathMode::Walk | PathMode::Trail) {
        return None;
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no endpoints
    };
    let batch = pull(inner, store, false).ok()?;
    // The endpoint the VarLength appends lands at the slot just past the inner width;
    // the DISTINCT arg must be exactly that endpoint (not, say, the source slot).
    if *want_slot != batch.slots.len() {
        return None;
    }
    let Col::Nodes(src) = batch.slot(*from) else {
        return None;
    };
    let count = varlen_distinct_endpoint_count(store, src, *dir, &want, *min, *max);
    Some(scalar_num(count as f64))
}

/// `<walk>.dedup().count()` — `count(*)` over a `DistinctBy` (on the endpoint slot) of a
/// var-length walk. Same distinct-endpoint count as [`try_varlen_distinct_count`]'s
/// `count(DISTINCT endpoint)`, but the front-end spells `dedup().count()` as a separate
/// `DistinctBy` node rather than a distinct aggregate, so it needs its own matcher. The
/// dedup key must be exactly the walk's appended endpoint slot; anything else (dedup on
/// the source, a multi-key dedup) is a different question and falls through.
pub(super) fn try_varlen_distinctby_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // plain count(*)
    }
    let Plan::DistinctBy {
        input: inner,
        key_slots,
    } = input
    else {
        return None;
    };
    let [dedup_slot] = key_slots.as_slice() else {
        return None;
    };
    // The dedup'd source is either a var-length WALK (repeat(...).times(k)) or a single
    // Expand hop (`both().dedup()`) — both a distinct-endpoint question. Normalize to
    // (inner-plan, from, dir, edge_label, min, max); a single Expand is a 1-hop walk.
    let (src_plan, from, dir, edge_label, min, max) = match inner.as_ref() {
        Plan::VarLength {
            input: vl_inner,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
            until,
            body_filter,
            double_loops: _, // a distinct endpoint set is blind to edge multiplicity
        } => {
            if until.is_some()
                || body_filter.is_some()
                || !matches!(mode, PathMode::Walk | PathMode::Trail)
            {
                return None;
            }
            (vl_inner.as_ref(), *from, *dir, edge_label, *min, *max)
        }
        Plan::Expand {
            input: ex_inner,
            from,
            dir,
            edge_label,
            bind_edge,
            double_loops: _,
        } => {
            if *bind_edge {
                return None; // the bound edge slot shifts the endpoint; not this shape
            }
            // A single hop below the dedup; a deeper chain (`both().both().dedup()`) keeps
            // its already-good path — materialize the earlier hops, BFS the last — because
            // a full multi-level BFS from the scan re-expands the whole (dense) graph and
            // measured ~8x SLOWER on a dense 2-hop. (Rejected optimization, kept for the
            // note.)
            (ex_inner.as_ref(), *from, *dir, edge_label, 1u32, 1u32)
        }
        _ => return None,
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)),
    };
    let batch = pull(src_plan, store, false).ok()?;
    // The dedup key must be exactly the endpoint the hop appends (slot == inner width).
    if *dedup_slot != batch.slots.len() {
        return None;
    }
    let Col::Nodes(src) = batch.slot(from) else {
        return None;
    };
    let count = varlen_distinct_endpoint_count(store, src, dir, &want, min, max);
    Some(scalar_num(count as f64))
}

/// The number of DISTINCT nodes reachable from `src` by a Walk/Trail of `min..=max`
/// hops along `dir`/`want` edges — the shared kernel behind both `count(DISTINCT
/// endpoint)` and `count(*)` over a `dedup()` of a var-length walk. Two regimes:
///
/// - `min ≤ 1`: cumulative shortest-distance BFS. Each node is expanded at most once
///   (at its shortest distance), so an edge is traversed once — O(E) — and every node
///   within `max` hops is an endpoint (every hop ≥ 1 ≥ min).
/// - `min ≥ 2`: a walk may revisit, so the endpoints at EXACTLY h hops are the h-th
///   neighbour-set iterate N^h(src), NOT the distance-h set. Expand the DISTINCT
///   frontier one level at a time (each level a set — a node expands at most once per
///   level) and union the levels in `min..=max`. O(hops · E), versus the
///   product-of-degrees an enumeration of every walk would pay.
///
/// Blind to edge multiplicity (a set), so a both()-crossed self-loop needs no special
/// casing. `min == 0` also counts the sources as their own 0-hop endpoints.
pub(super) fn varlen_distinct_endpoint_count(
    store: &Store,
    src: &[u32],
    dir: Dir,
    want: &[u32],
    min: u32,
    max: u32,
) -> usize {
    let n = store.node_count();
    if min >= 2 {
        let mut reached = vec![false; n];
        let mut in_next = vec![false; n];
        let mut seen = vec![false; n];
        let mut frontier: Vec<u32> = Vec::with_capacity(src.len());
        for &s in src {
            if !seen[s as usize] {
                seen[s as usize] = true;
                frontier.push(s);
            }
        }
        let mut next: Vec<u32> = Vec::new();
        for hop in 1..=max {
            if frontier.is_empty() {
                break;
            }
            next.clear();
            for &v in &frontier {
                for_each_nbr(store, v, dir, want, false, |nbr, _| {
                    if !in_next[nbr as usize] {
                        in_next[nbr as usize] = true;
                        next.push(nbr);
                    }
                });
            }
            if hop >= min {
                for &w in &next {
                    reached[w as usize] = true;
                }
            }
            // Reset the level-set for reuse (only the touched entries), then advance.
            for &w in &next {
                in_next[w as usize] = false;
            }
            std::mem::swap(&mut frontier, &mut next);
        }
        return reached.iter().filter(|&&r| r).count();
    }
    let mut visited = vec![false; n]; // added to a frontier (expansion dedup)
    let mut reached = vec![false; n]; // a valid endpoint (hop in min..=max)
    let mut frontier: Vec<u32> = Vec::with_capacity(src.len());
    for &s in src {
        if !visited[s as usize] {
            visited[s as usize] = true;
            frontier.push(s);
        }
        if min == 0 {
            reached[s as usize] = true; // the 0-hop path a=b
        }
    }
    let mut next: Vec<u32> = Vec::new();
    for _hop in 1..=max {
        if frontier.is_empty() {
            break;
        }
        for &v in &frontier {
            for_each_nbr(store, v, dir, want, false, |nbr, _| {
                reached[nbr as usize] = true;
                if !visited[nbr as usize] {
                    visited[nbr as usize] = true;
                    next.push(nbr);
                }
            });
        }
        std::mem::swap(&mut frontier, &mut next);
        next.clear();
    }
    reached.iter().filter(|&&r| r).count()
}

/// The fold twin of `varlen_count_dfs`: calls `emit(endpoint)` at every length in
/// `min..=max` instead of counting. Traversal / edge-type / trail logic — and thus
/// the EMISSION ORDER — are identical to `var_length`, so a `sum` folded here lands
/// the same value as materializing then summing.
#[allow(clippy::too_many_arguments)]
pub(super) fn varlen_agg_dfs(
    store: &Store,
    v: u32,
    _len: u32, // always 0 at the call sites — the walk starts at the source
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    emit: &mut dyn FnMut(u32),
) {
    // The fold twin visits the same endpoints in the same order as the materializing
    // path (this fast-path is only taken without a `both()` double-loop, so `false`).
    varlen_scan_walk(
        store, v, min, max, dir, want, mode, start, used, false, emit,
    );
}

/// A scalar `sum`/`avg`/`min`/`max`/`count(arg)` over a bare var-length's ENDPOINT
/// property, folded DURING the DFS — no keep/ends, no gather, no intermediate batch
/// (which `try_frontier_aggregate`/`aggregate` all build, ~3x the traversal). The
/// emission order matches `var_length`, so `sum` folds in the same order and the
/// value contract (`cmp_num_total`) drives min/max — byte-identical to the
/// materializing path. `None` unless the aggregate reads exactly the appended
/// endpoint slot (block-streaming the general chain was measured a net regression;
/// this surgical fold is the low-overhead win).
pub(super) fn try_varlen_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct
        || !matches!(
            agg.func,
            AggFn::Sum | AggFn::Avg | AggFn::Min | AggFn::Max | AggFn::Count
        )
    {
        return None;
    }
    let Plan::VarLength {
        input: inner,
        from,
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
        return None; // an until(pred) walk emits a filtered subset — no closed-form agg
    }
    // The aggregate argument must be a property of the ENDPOINT (the appended slot).
    let Some(Expr::Prop { slot, key }) = agg.arg.as_ref() else {
        return None; // count(*) is `try_varlen_count`
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        // Unknown edge type → no paths. A non-empty want of a non-existent id
        // (etype ids are dense, so u32::MAX is none) matches nothing, yielding the
        // empty-aggregate value without a special-cased early return here.
        Err(()) => vec![u32::MAX],
    };
    let batch = pull(inner, store, false).ok()?;
    if *slot != batch.slots.len() {
        return None; // arg is not the endpoint
    }
    let Col::Nodes(src) = batch.slot(*from) else {
        return None;
    };
    let column = store.column(key)?; // property absent everywhere → fall back
    let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
    let dfs = |emit: &mut dyn FnMut(u32)| {
        let mut used: Vec<u32> = Vec::new();
        for &v in src {
            if node_unique {
                used.push(v); // mark the start node
            }
            varlen_agg_dfs(
                store, v, 0, *min, *max, *dir, &want, *mode, v, &mut used, emit,
            );
            if node_unique {
                used.pop();
            }
        }
    };

    let val = match (agg.func, column) {
        (AggFn::Sum | AggFn::Avg, Column::Num { data, present, .. }) => {
            let mut total = 0.0f64;
            let mut cnt = 0u64;
            dfs(&mut |v| {
                let i = v as usize;
                if present[i] {
                    total += data[i];
                    cnt += 1;
                }
            });
            if agg.func == AggFn::Sum {
                Value::Num(total)
            } else if cnt == 0 {
                Value::Null
            } else {
                Value::Num(total / cnt as f64)
            }
        }
        (AggFn::Min | AggFn::Max, Column::Num { data, present, .. }) => {
            let want_min = agg.func == AggFn::Min;
            let mut best: Option<f64> = None;
            dfs(&mut |v| {
                let i = v as usize;
                if present[i] {
                    let x = data[i];
                    best = Some(match best {
                        None => x,
                        Some(b) => {
                            let ord = value::cmp_num_total(x, b);
                            if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                                x
                            } else {
                                b
                            }
                        }
                    });
                }
            });
            best.map_or(Value::Null, Value::Num)
        }
        (AggFn::Min | AggFn::Max, Column::Str { data, present, .. }) => {
            // Track the best endpoint id (not a borrow into `data`), comparing `&str`
            // directly — the value contract's order for two strings is lexicographic,
            // so this equals the materializing min/max. `<`/`>` on equal keeps the
            // first (`cmp_total(..).is_lt()` semantics).
            let want_min = agg.func == AggFn::Min;
            let mut best: Option<u32> = None;
            dfs(&mut |v| {
                let i = v as usize;
                if present[i] {
                    best = Some(match best {
                        None => v,
                        Some(b) => {
                            let (sv, sb) = (data[i].as_ref(), data[b as usize].as_ref());
                            if (want_min && sv < sb) || (!want_min && sv > sb) {
                                v
                            } else {
                                b
                            }
                        }
                    });
                }
            });
            best.map_or(Value::Null, |v| Value::Str(data[v as usize].clone()))
        }
        (
            AggFn::Min | AggFn::Max,
            Column::Dict {
                dict,
                codes,
                present,
                ..
            },
        ) => {
            let want_min = agg.func == AggFn::Min;
            let str_of = |v: u32| dict[codes[v as usize] as usize].as_ref();
            let mut best: Option<u32> = None;
            dfs(&mut |v| {
                if present[v as usize] {
                    best = Some(match best {
                        None => v,
                        Some(b) => {
                            if (want_min && str_of(v) < str_of(b))
                                || (!want_min && str_of(v) > str_of(b))
                            {
                                v
                            } else {
                                b
                            }
                        }
                    });
                }
            });
            best.map_or(Value::Null, |v| {
                Value::Str(dict[codes[v as usize] as usize].clone())
            })
        }
        (AggFn::Min | AggFn::Max, _) => return None, // Temporal/Bool/Gen → general path
        (AggFn::Count, col) => {
            // count(arg): endpoints whose property is present (non-null).
            let present: &[bool] = match col {
                Column::Num { present, .. }
                | Column::Str { present, .. }
                | Column::Bool { present, .. }
                | Column::Dict { present, .. } => present,
                _ => return None, // Temporal/Gen → the general path
            };
            let mut cnt = 0u64;
            dfs(&mut |v| {
                if present[v as usize] {
                    cnt += 1;
                }
            });
            Value::Num(cnt as f64)
        }
        _ => return None,
    };
    Some(Batch::single(Col::Gen(vec![val])))
}

/// `<hop-chain>.values(k).min()/max()` over a pure Scan/Expand chain: fold the numeric
/// property `k` over the per-node PATH-COUNT frontier — WITHOUT materializing the
/// exploding frontier (the min/max analog of the count fast-path). MIN/MAX only: they
/// are order-INDEPENDENT, so collapsing the frontier to per-node multiplicity (which
/// loses row order) is byte-identical; SUM/AVG would change the summation order, so they
/// stay on `try_frontier_aggregate` (`frontier_ids`, which keeps order). Numeric columns
/// only; a filtered chain / edge hop / non-numeric column returns None.
/// `count(*)` over any plan `frontier_counts` can fold — including a `hasLabel(L)`-
/// filtered hop chain (`<hops>.hasLabel(L).count()`) — as the SUM of the per-node path
/// multiplicities, never materializing the frontier. Order-independent (an integer row
/// count), so byte-identical. `None` for a non-fusable shape.
pub(super) fn try_frontier_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    // Only worth the frontier fold when a plain filter/hop chain would otherwise
    // materialize — i.e. there IS a frontier filter (the bare-chain counts already have
    // their own fast-paths). Require a top-level IsLabeled filter.
    if !matches!(
        input,
        Plan::Filter {
            pred: Expr::IsLabeled { .. },
            ..
        }
    ) {
        return None;
    }
    let counts = frontier_counts(input, store)?;
    let mut total = 0f64;
    counts.for_each(|_, c| total += c);
    Some(scalar_num(total))
}

pub(super) fn try_frontier_prop_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct || !matches!(agg.func, AggFn::Min | AggFn::Max) {
        return None;
    }
    let Some(Expr::Prop { slot, key }) = agg.arg.as_ref() else {
        return None;
    };
    let width = chain_width(input)?;
    if *slot != width - 1 {
        return None; // arg must be a property of the chain frontier
    }
    let Some(Column::Num { data, present, .. }) = store.column(key) else {
        return None; // non-numeric / absent-everywhere → general path
    };
    let counts = frontier_counts(input, store)?;
    let want_min = agg.func == AggFn::Min;
    let mut best: Option<f64> = None;
    counts.for_each(|v, _c| {
        let i = v as usize;
        if present[i] {
            let x = data[i];
            best = Some(match best {
                None => x,
                Some(b) => {
                    let ord = value::cmp_num_total(x, b);
                    if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                        x
                    } else {
                        b
                    }
                }
            });
        }
    });
    Some(Batch::single(Col::Gen(vec![
        best.map_or(Value::Null, Value::Num)
    ])))
}

/// Answer a scalar `count(*)` over a bare labelled/unlabelled `Scan` in O(1) (a
/// label bucket length — buckets hold only live ids) or a single tombstone-bitmap
/// sweep (unlabelled), WITHOUT materializing the id vector. `None` for any other
/// shape (a WHERE seed, an Expand, `count(arg)`), which the other paths handle.
pub(super) fn try_scan_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() {
        return None; // count(*) only; count(arg)/DISTINCT need the values
    }
    let n = match input {
        Plan::Scan { label: Some(l) } => store.nodes_with_label(l).len(),
        Plan::Scan { label: None } => store.live_node_count(),
        _ => return None,
    };
    Some(scalar_num(n as f64))
}

/// Answer a scalar `sum`/`avg`/`count(arg)` over a bare `Scan`'s Num property by
/// summing the RAW f64 column (present cells only), WITHOUT materializing the
/// frontier or boxing each cell into a `Value`. `None` (fall back) for a grouped
/// aggregate, a DISTINCT, `min`/`max` (need the value-contract order), a non-`Num`
/// column (which may need poison handling), or any non-`Scan` input.
/// A scalar numeric aggregate (`sum`/`avg`/`min`/`max`/`count(prop)`) over a FILTERED
/// scan — `has(...).values(k).sum()` and friends. Get the survivors from the filter fast
/// path (`try_filter_keep`, which raw-passes num/str/And/Not/dict predicates), then
/// accumulate the aggregate directly over their column values — no gather of a survivor
/// frontier, no Project into a Col::Num, no boxed fold. `None` for a non-`Filter{Scan}`
/// input, a non-fast-pathable predicate, or an unsupported agg — the general path runs.
pub(super) fn try_filtered_scan_num_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct
        || !matches!(
            agg.func,
            AggFn::Sum | AggFn::Avg | AggFn::Min | AggFn::Max | AggFn::Count
        )
    {
        return None;
    }
    let Plan::Filter { input: scan, pred } = input else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    let Some(Expr::Prop { slot: 0, key }) = agg.arg.as_ref() else {
        return None; // count(*) is try_filtered_count; only a prop agg here
    };
    let Some(Column::Num { data, present, .. }) = store.column(key) else {
        return None;
    };
    // Survivor ROWS of the scan frontier (row index == the node id for a bare scan).
    let ids: Vec<u32> = match label {
        Some(l) => store.nodes_with_label(l).to_vec(),
        None => store.all_nodes(),
    };
    let batch = Batch::of(vec![Col::Nodes(ids)]);
    let keep = try_filter_keep(pred, store, &batch)?;
    let Col::Nodes(sids) = batch.slot(0) else {
        return None;
    };
    let (mut total, mut cnt, mut best): (f64, u64, Option<f64>) = (0.0, 0, None);
    for &row in &keep {
        let i = sids[row] as usize;
        if !present[i] {
            continue; // the agg's own prop may be NULL even when the filter passed
        }
        let x = data[i];
        total += x;
        cnt += 1;
        best = Some(match best {
            None => x,
            Some(b) => {
                let keep_new = (agg.func == AggFn::Min && value::cmp_num_total(x, b).is_lt())
                    || (agg.func == AggFn::Max && value::cmp_num_total(x, b).is_gt());
                if keep_new {
                    x
                } else {
                    b
                }
            }
        });
    }
    let result = match agg.func {
        AggFn::Sum if agg.null_on_empty && cnt == 0 => Value::Null,
        AggFn::Sum => Value::Num(total),
        AggFn::Count => Value::Num(cnt as f64),
        AggFn::Avg => {
            if cnt == 0 {
                Value::Null
            } else {
                Value::Num(total / cnt as f64)
            }
        }
        _ => best.map_or(Value::Null, Value::Num), // min/max of nothing → NULL
    };
    Some(Batch::single(Col::Gen(vec![result])))
}

pub(super) fn try_scan_num_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.distinct || !matches!(agg.func, AggFn::Sum | AggFn::Avg | AggFn::Count) {
        return None;
    }
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    let Some(Expr::Prop { slot: 0, key }) = agg.arg.as_ref() else {
        return None;
    };
    let Some(Column::Num { data, present, .. }) = store.column(key) else {
        return None; // non-numeric column: the general path handles poison
    };
    let (mut total, mut cnt) = (0f64, 0u64);
    // Whole-column fast path: when the scan covers EVERY live node (an unlabelled
    // scan, or a label all nodes carry) with nothing deleted, sum the raw
    // `data`/`present` slices directly — no per-row id indirection, so the loop
    // auto-vectorizes. Otherwise walk the label's id list.
    let all_live = store.live_node_count() == store.node_count();
    let whole = all_live
        && match label {
            None => true,
            Some(l) => store.nodes_with_label(l).len() == store.node_count(),
        };
    if whole {
        for (i, &x) in data.iter().enumerate() {
            if present[i] {
                total += x;
                cnt += 1;
            }
        }
    } else {
        let mut visit = |i: usize| {
            if present[i] {
                total += data[i];
                cnt += 1;
            }
        };
        match label {
            Some(l) => store
                .nodes_with_label(l)
                .iter()
                .for_each(|&id| visit(id as usize)),
            None => (0..store.node_count()).for_each(|i| {
                if store.is_alive(i as u32) {
                    visit(i);
                }
            }),
        }
    }
    let result = match agg.func {
        AggFn::Sum if agg.null_on_empty && cnt == 0 => Value::Null, // Gremlin sum() of nothing
        AggFn::Sum => Value::Num(total), // 0.0 over an empty/all-null set (K0a)
        AggFn::Count => Value::Num(cnt as f64), // count(arg) = present count
        _ => {
            if cnt == 0 {
                Value::Null // avg of nothing
            } else {
                Value::Num(total / cnt as f64)
            }
        }
    };
    Some(Batch::of(vec![Col::Gen(vec![result])]))
}

/// Visit each scanned node's dense id (as `usize`) for a bare `Scan`. Iterates the
/// raw `0..node_count` range directly when the scan covers every live node (an
/// unlabelled scan, or a label all nodes carry, nothing deleted) — sequential and
/// vectorizable — otherwise walks the label's id list. Generic over `F` so there is
/// no per-node dynamic dispatch. Shared by the scan-aggregate fast paths.
pub(super) fn scan_visit<F: FnMut(usize)>(store: &Store, label: &Option<String>, mut f: F) {
    let all_live = store.live_node_count() == store.node_count();
    let whole = all_live
        && match label {
            None => true,
            Some(l) => store.nodes_with_label(l).len() == store.node_count(),
        };
    if whole {
        (0..store.node_count()).for_each(&mut f);
    } else {
        match label {
            Some(l) => store
                .nodes_with_label(l)
                .iter()
                .for_each(|&id| f(id as usize)),
            None => (0..store.node_count()).for_each(|i| {
                if store.is_alive(i as u32) {
                    f(i);
                }
            }),
        }
    }
}

/// A group's accumulators: row count (for `count(*)`) plus `(total, count, best)`
/// per numeric aggregate.
struct GroupAcc {
    rows: u64,
    aggs: Vec<(f64, u64, Option<f64>)>,
}

/// Fused single-key grouped aggregate over a bare `Scan`: `RETURN n.k AS key,
/// <aggs> …` where the group key is a `Str`/`Num`/`Bool` column and each aggregate
/// is `count(*)` or a numeric reduction over a `Num` column. Reads the storage
/// columns directly and groups by the TYPED key value (first-seen order, matching
/// the grouping contract), so the frontier and projected columns are never
/// materialized. `None` for any other shape (Temporal/Gen key, non-numeric agg
/// arg, DISTINCT, multi-key). The per-key string hashing is the residual floor.
/// The TIGHT case of [`try_scan_group_agg`]: a plain `count(*) GROUP BY <col>` where the
/// group column is DICTIONARY-encoded (a categorical `city`/`status`). Count directly per
/// dict CODE into a `Vec<u64>` — no per-group `GroupAcc` struct, no `accumulate` closure,
/// no bounds-checked `acc[group]` write per row, just `counts[code] += 1`.
///
/// This exists because the general `GroupAcc` path, while fine natively, is
/// DISPROPORTIONATELY slow on wasm: its nested closure + per-row struct indexing compile
/// to indirect calls / bounds-checked accesses that wasm penalizes several times more
/// than native, which flipped `groupCount().by('city')` and `GROUP BY city` from wins to
/// losses on the wasm surface while they won on FFI/native. The lean loop closes that.
/// Numeric aggregates, multi-agg, and non-dict keys stay on `try_scan_group_agg`.
pub(super) fn try_scan_dict_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    let [(_, Expr::Prop { slot: 0, key })] = keys else {
        return None;
    };
    let [agg] = aggs else {
        return None;
    };
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None; // count(*) only
    }
    let Plan::Scan { label } = input else {
        return None;
    };
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(key)
    else {
        return None;
    };
    // `usize` counters, NOT u64: a count cannot exceed node_count, and `usize` is 32-bit
    // (native i32) on wasm32 where a u64 add is EMULATED — the general path's u64
    // GroupAcc.rows is part of why grouping was disproportionately slow on wasm.
    let mut counts = vec![0usize; dict.len()];
    let mut null_count = 0usize;
    // Group output order is FIRST-SEEN (the grouping contract, matching the general path):
    // record a code the first time it is counted (its count goes 0 -> 1); -1 = null group.
    let mut order: Vec<i32> = Vec::new();
    let mut seen_null = false;
    scan_visit(store, label, |i| {
        if present[i] {
            let c = codes[i] as usize;
            if counts[c] == 0 {
                order.push(c as i32);
            }
            counts[c] += 1;
        } else {
            if !seen_null {
                seen_null = true;
                order.push(-1);
            }
            null_count += 1;
        }
    });
    let mut key_col: Vec<Value> = Vec::with_capacity(order.len());
    let mut cnt_col: Vec<Value> = Vec::with_capacity(order.len());
    for &code in &order {
        if code < 0 {
            key_col.push(Value::Null);
            cnt_col.push(Value::Num(null_count as f64));
        } else {
            key_col.push(Value::Str(dict[code as usize].clone()));
            cnt_col.push(Value::Num(counts[code as usize] as f64));
        }
    }
    Some(Batch::of(vec![Col::Gen(key_col), Col::Gen(cnt_col)]))
}

/// `count(*) GROUP BY <dict col>` over ANY frontier — a hop, a filter — not just a bare
/// Scan (which [`try_scan_dict_count`] already streams). Pull the frontier once, then
/// count per dict CODE, instead of the general group_by decoding each cell to a
/// `Value::Str` and HASHING it (`group_by_arc`) — the string hash over a big hop frontier
/// was the whole cost (`inE('R').outV().groupCount().by('city')` at 0.08x). First-seen
/// group order and null handling (absent OR the u32::MAX optional-match sentinel → the
/// null group) match the general path exactly.
pub(super) fn try_frontier_dict_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
    track: bool,
) -> Option<Batch> {
    let [(_, Expr::Prop { slot, key })] = keys else {
        return None;
    };
    let [agg] = aggs else {
        return None;
    };
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None;
    }
    if matches!(input, Plan::Scan { .. }) {
        return None; // the bare-scan case streams via try_scan_dict_count
    }
    let Some(Column::Dict {
        dict,
        codes,
        present,
        ..
    }) = store.column(key)
    else {
        return None;
    };
    let batch = pull(input, store, track).ok()?;
    let Col::Nodes(ids) = batch.slot(*slot) else {
        return None;
    };
    let mut counts = vec![0usize; dict.len()];
    let mut null_count = 0usize;
    let mut order: Vec<i32> = Vec::new();
    let mut seen_null = false;
    for &id in ids {
        if id != u32::MAX && present[id as usize] {
            let c = codes[id as usize] as usize;
            if counts[c] == 0 {
                order.push(c as i32);
            }
            counts[c] += 1;
        } else {
            if !seen_null {
                seen_null = true;
                order.push(-1);
            }
            null_count += 1;
        }
    }
    let mut key_col: Vec<Value> = Vec::with_capacity(order.len());
    let mut cnt_col: Vec<Value> = Vec::with_capacity(order.len());
    for &code in &order {
        if code < 0 {
            key_col.push(Value::Null);
            cnt_col.push(Value::Num(null_count as f64));
        } else {
            key_col.push(Value::Str(dict[code as usize].clone()));
            cnt_col.push(Value::Num(counts[code as usize] as f64));
        }
    }
    Some(Batch::of(vec![Col::Gen(key_col), Col::Gen(cnt_col)]))
}

pub(super) fn try_scan_group_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    let [(_, Expr::Prop { slot: 0, key: gkey })] = keys else {
        return None;
    };
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    // Agg specs: the Num column (None for count(*)) and function.
    type Spec<'a> = (Option<(&'a [f64], &'a [bool])>, AggFn);
    let mut specs: Vec<Spec> = Vec::with_capacity(aggs.len());
    for agg in aggs {
        if agg.distinct {
            return None;
        }
        match (agg.func, agg.arg.as_ref()) {
            (AggFn::Count, None) => specs.push((None, AggFn::Count)),
            (
                AggFn::Sum | AggFn::Avg | AggFn::Count | AggFn::Min | AggFn::Max,
                Some(Expr::Prop { slot: 0, key }),
            ) => {
                let Some(Column::Num { data, present, .. }) = store.column(key) else {
                    return None;
                };
                specs.push((Some((data.as_slice(), present.as_slice())), agg.func));
            }
            _ => return None,
        }
    }

    let mut group_keys: Vec<Value> = Vec::new();
    let mut acc: Vec<GroupAcc> = Vec::new();
    let na = specs.len();
    // Add one row (dense group id `g`) to the accumulators.
    let accumulate = |acc: &mut Vec<GroupAcc>, g: usize, i: usize| {
        let a = &mut acc[g];
        a.rows += 1;
        for (k, &(col, func)) in specs.iter().enumerate() {
            let Some((data, present)) = col else { continue };
            if !present[i] {
                continue;
            }
            let x = data[i];
            let s = &mut a.aggs[k];
            s.0 += x;
            s.1 += 1;
            s.2 = Some(match s.2 {
                None => x,
                Some(b) => match func {
                    AggFn::Min if value::cmp_num_total(x, b).is_lt() => x,
                    AggFn::Max if value::cmp_num_total(x, b).is_gt() => x,
                    _ => b,
                },
            });
        }
    };

    // Resolve a row to a dense group id (first-seen), creating the group on demand.
    macro_rules! run {
        ($present:expr, $lookup:expr, $keyval:expr, $nullkey:expr) => {{
            let present = $present;
            let mut map: FnvMap<_, u32> = FnvMap::default();
            let mut null_group: Option<u32> = None;
            scan_visit(store, label, |i| {
                let g = if present[i] {
                    let k = $lookup(i);
                    match map.get(&k) {
                        Some(&g) => g as usize,
                        None => {
                            let g = group_keys.len() as u32;
                            map.insert(k, g);
                            group_keys.push($keyval(i));
                            acc.push(GroupAcc {
                                rows: 0,
                                aggs: vec![(0.0, 0, None); na],
                            });
                            g as usize
                        }
                    }
                } else {
                    match null_group {
                        Some(g) => g as usize,
                        None => {
                            let g = group_keys.len() as u32;
                            null_group = Some(g);
                            group_keys.push(Value::Null);
                            acc.push(GroupAcc {
                                rows: 0,
                                aggs: vec![(0.0, 0, None); na],
                            });
                            g as usize
                        }
                    }
                };
                accumulate(&mut acc, g, i);
            });
            let _ = $nullkey; // silence unused when the key type has no null path
        }};
    }
    // Only a STRING group key: reading the storage column directly avoids
    // materializing 100k `Arc<str>` (the win). A Num/Bool key already groups via
    // `assign_groups`' typed fast path over the materialized column, which is as
    // fast — so leave those to the general aggregate (this fused path's per-agg
    // accumulator loop is slightly heavier and would regress them).
    match store.column(gkey)? {
        Column::Str { data, present, .. } => {
            run!(
                present,
                |i: usize| data[i].as_ref(),
                |i: usize| Value::Str(data[i].clone()),
                ()
            );
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            // Group by CODE, mapped to a dense group id in first-seen (scan) order —
            // a per-code slot, no per-row string hash. First-seen (not dict order) is
            // what the pinned GROUP BY order requires, since the dict was built over
            // all nodes and the scan may visit a label subset in a different order.
            let mut code_to_group: Vec<u32> = vec![u32::MAX; dict.len()];
            let mut null_group: Option<u32> = None;
            scan_visit(store, label, |i| {
                let g = if present[i] {
                    let c = codes[i] as usize;
                    if code_to_group[c] == u32::MAX {
                        let g = group_keys.len() as u32;
                        code_to_group[c] = g;
                        group_keys.push(Value::Str(dict[c].clone()));
                        acc.push(GroupAcc {
                            rows: 0,
                            aggs: vec![(0.0, 0, None); na],
                        });
                        g as usize
                    } else {
                        code_to_group[c] as usize
                    }
                } else {
                    match null_group {
                        Some(g) => g as usize,
                        None => {
                            let g = group_keys.len() as u32;
                            null_group = Some(g);
                            group_keys.push(Value::Null);
                            acc.push(GroupAcc {
                                rows: 0,
                                aggs: vec![(0.0, 0, None); na],
                            });
                            g as usize
                        }
                    }
                };
                accumulate(&mut acc, g, i);
            });
        }
        _ => return None,
    }

    // Build the output: the key column, then one column per aggregate.
    let key_col = Col::Gen(group_keys);
    let mut cols = vec![key_col];
    for (k, &(col, func)) in specs.iter().enumerate() {
        let vals: Vec<Value> = acc
            .iter()
            .map(|a| {
                let (total, cnt, best) = a.aggs[k];
                match func {
                    AggFn::Count if col.is_none() => Value::Num(a.rows as f64),
                    AggFn::Count => Value::Num(cnt as f64),
                    AggFn::Sum => Value::Num(total),
                    AggFn::Avg => {
                        if cnt == 0 {
                            Value::Null
                        } else {
                            Value::Num(total / cnt as f64)
                        }
                    }
                    _ => best.map_or(Value::Null, Value::Num),
                }
            })
            .collect();
        cols.push(Col::Gen(vals));
    }
    Some(Batch::of(cols))
}

/// Answer `count(DISTINCT n.k)` over a bare `Scan` by deduping the RAW column into
/// a typed set (a `&str`, the f64 group bits, or a bool) and returning its size —
/// no frontier materialization and no per-cell byte-key serialization. Nulls are
/// skipped (as `count(DISTINCT)` does). `None` for a non-`Scan` input, a
/// Temporal/Gen column, or a non-distinct/`count(*)` agg.
/// A membership bitset over the DISTINCT integer values of a Num column: returns
/// `(min, bits)` where `bits[k]` is set iff the value `min + k` is present. Used
/// instead of hashing when every present value is a finite INTEGER in a small span
/// — `count(DISTINCT age)` / `DISTINCT age` over 100 ages then sets 100 bits rather
/// than hashing 200k cells. One pass finds the span + integrality (a non-integer,
/// NaN, or Inf value disqualifies via `fract()`/`is_finite`), a second sets the
/// bits. Distinct finite integers map to distinct offsets, so a popcount equals the
/// FnvSet's `len` and the set bits recover every distinct value exactly. `None`
/// (fall back to hashing) when the column is empty, non-integer, or spans too wide.
pub(super) fn low_card_int_bitset(
    store: &Store,
    label: &Option<String>,
    data: &[f64],
    present: &[bool],
) -> Option<(f64, Vec<bool>, bool)> {
    const MAX_SPAN: usize = 1 << 20; // cap the bitset at ~1M bits (128 KB)
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut any, mut all_int, mut saw_absent) = (false, true, false);
    scan_visit(store, label, |i| {
        if present[i] {
            let x = data[i];
            any = true;
            if x.is_finite() && x.fract() == 0.0 {
                lo = lo.min(x);
                hi = hi.max(x);
            } else {
                all_int = false;
            }
        } else {
            saw_absent = true; // a NULL cell — DISTINCT keeps one, count ignores it
        }
    });
    if !any || !all_int {
        return None;
    }
    let span = (hi - lo) as usize;
    if span >= MAX_SPAN {
        return None;
    }
    let mut bits = vec![false; span + 1];
    scan_visit(store, label, |i| {
        if present[i] {
            bits[(data[i] - lo) as usize] = true;
        }
    });
    Some((lo, bits, saw_absent))
}

pub(super) fn try_scan_distinct_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || !agg.distinct {
        return None;
    }
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    let Some(Expr::Prop { slot: 0, key }) = agg.arg.as_ref() else {
        return None;
    };
    let count = match store.column(key)? {
        Column::Str { data, present, .. } => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            scan_visit(store, label, |i| {
                if present[i] {
                    seen.insert(data[i].as_ref());
                }
            });
            seen.len()
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            // A distinct value == a distinct code: mark a per-code bitset, no hashing.
            let mut seen = vec![false; dict.len()];
            scan_visit(store, label, |i| {
                if present[i] {
                    seen[codes[i] as usize] = true;
                }
            });
            seen.iter().filter(|&&b| b).count()
        }
        Column::Num { data, present, .. } => {
            // Low-cardinality integer fast path: dedup with a bitset (popcount), no
            // hashing. Falls back to the FnvSet when values are wide-ranged or
            // non-integer. The distinct count is identical either way.
            if let Some((_, bits, _)) = low_card_int_bitset(store, label, data, present) {
                bits.iter().filter(|&&b| b).count()
            } else {
                let mut seen: FnvSet<u64> = FnvSet::default();
                scan_visit(store, label, |i| {
                    if present[i] {
                        seen.insert(value::num_group_bits(data[i]));
                    }
                });
                seen.len()
            }
        }
        Column::Bool { data, present, .. } => {
            let mut seen = [false; 2];
            scan_visit(store, label, |i| {
                if present[i] {
                    seen[usize::from(data[i])] = true;
                }
            });
            usize::from(seen[0]) + usize::from(seen[1])
        }
        _ => return None, // Temporal / Gen → the general aggregate
    };
    Some(scalar_num(count as f64))
}

/// The frontier sibling of [`try_scan_distinct_count`]: `count(DISTINCT <frontier prop>)`
/// over a hop chain, deduped over the DISTINCT reached endpoints (`frontier_counts`) rather
/// than materializing the exploded path multiset and byte-keying every row through the
/// general grouped fold. Path multiplicity is irrelevant to DISTINCT, so visiting each
/// endpoint once yields the identical value set — byte-identical count, far less work.
pub(super) fn try_frontier_distinct_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || !agg.distinct {
        return None;
    }
    let Some(Expr::Prop { slot, key }) = agg.arg.as_ref() else {
        return None;
    };
    let width = chain_width(input)?;
    if *slot != width - 1 {
        return None; // arg must be a property of the chain frontier
    }
    let counts = frontier_counts(input, store)?;
    let count = match store.column(key)? {
        Column::Str { data, present, .. } => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen.insert(data[i].as_ref());
                }
            });
            seen.len()
        }
        Column::Dict {
            dict,
            codes,
            present,
            ..
        } => {
            let mut seen = vec![false; dict.len()];
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen[codes[i] as usize] = true;
                }
            });
            seen.iter().filter(|&&b| b).count()
        }
        Column::Num { data, present, .. } => {
            let mut seen: FnvSet<u64> = FnvSet::default();
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen.insert(value::num_group_bits(data[i]));
                }
            });
            seen.len()
        }
        Column::Bool { data, present, .. } => {
            let mut seen = [false; 2];
            counts.for_each(|v, _| {
                let i = v as usize;
                if present[i] {
                    seen[usize::from(data[i])] = true;
                }
            });
            usize::from(seen[0]) + usize::from(seen[1])
        }
        _ => return None, // Temporal / Gen → the general aggregate
    };
    Some(scalar_num(count as f64))
}

/// Answer several scalar numeric aggregates (`sum`/`avg`/`min`/`max`/`count`) over
/// a bare `Scan` in ONE pass over the Num columns — e.g. `min(age), max(age)` or
/// `count(*), avg(age)`. `None` if any agg is grouped/DISTINCT or not a numeric
/// reduction over a `Num` property (or `count(*)`). Complements the single-agg
/// [`try_scan_num_agg`], which keeps the tighter auto-vectorized loop.
pub(super) fn try_scan_multi_agg(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.is_empty() {
        return None;
    }
    let label = match input {
        Plan::Scan { label } => label,
        _ => return None,
    };
    // Per agg: its Num column slices (None for `count(*)`) and function.
    type AggSpec<'a> = (Option<(&'a [f64], &'a [bool])>, AggFn);
    let mut specs: Vec<AggSpec> = Vec::with_capacity(aggs.len());
    for agg in aggs {
        if agg.distinct {
            return None;
        }
        match (agg.func, agg.arg.as_ref()) {
            (AggFn::Count, None) => specs.push((None, AggFn::Count)), // count(*)
            (
                AggFn::Sum | AggFn::Avg | AggFn::Count | AggFn::Min | AggFn::Max,
                Some(Expr::Prop { slot: 0, key }),
            ) => {
                let Some(Column::Num { data, present, .. }) = store.column(key) else {
                    return None;
                };
                specs.push((Some((data.as_slice(), present.as_slice())), agg.func));
            }
            _ => return None,
        }
    }
    // Fast path: every value-aggregate reads ONE Num column (e.g. `sum(age),
    // min(age), max(age)`) — a single BRANCH-FREE pass computing sum/cnt/min/max
    // with straight f64 ops (stored Nums are finite, so `x < mn` == cmp_num_total),
    // instead of the per-element per-spec match in the general loop below.
    let used: Vec<*const f64> = specs
        .iter()
        .filter_map(|(c, _)| c.map(|(d, _)| d.as_ptr()))
        .collect();
    if !used.is_empty() && used.iter().all(|&p| p == used[0]) {
        let (data, present) = specs.iter().find_map(|(c, _)| *c).expect("used non-empty");
        let (mut sum, mut cnt, mut mn, mut mx, mut rows) =
            (0.0f64, 0u64, f64::INFINITY, f64::NEG_INFINITY, 0u64);
        scan_visit(store, label, |i| {
            rows += 1;
            if present[i] {
                let x = data[i];
                sum += x;
                cnt += 1;
                if x < mn {
                    mn = x;
                }
                if x > mx {
                    mx = x;
                }
            }
        });
        let cols: Vec<Col> = specs
            .iter()
            .map(|&(col, func)| {
                let v = match func {
                    AggFn::Count if col.is_none() => Value::Num(rows as f64),
                    AggFn::Count => Value::Num(cnt as f64),
                    AggFn::Sum => Value::Num(sum),
                    AggFn::Avg if cnt == 0 => Value::Null,
                    AggFn::Avg => Value::Num(sum / cnt as f64),
                    AggFn::Min if cnt == 0 => Value::Null,
                    AggFn::Min => Value::Num(mn),
                    AggFn::Max if cnt == 0 => Value::Null,
                    _ => Value::Num(mx),
                };
                Col::Gen(vec![v])
            })
            .collect();
        return Some(Batch::of(cols));
    }
    // (total, count, best) per agg; `rows` counts scanned nodes for count(*).
    let mut acc: Vec<(f64, u64, Option<f64>)> = vec![(0.0, 0, None); specs.len()];
    let mut rows = 0u64;
    let mut visit = |i: usize| {
        rows += 1;
        for (k, (col, func)) in specs.iter().enumerate() {
            let Some((data, present)) = col else { continue };
            if !present[i] {
                continue;
            }
            let x = data[i];
            let a = &mut acc[k];
            a.0 += x;
            a.1 += 1;
            a.2 = Some(match a.2 {
                None => x,
                Some(b) => match func {
                    AggFn::Min if value::cmp_num_total(x, b).is_lt() => x,
                    AggFn::Max if value::cmp_num_total(x, b).is_gt() => x,
                    _ => b,
                },
            });
        }
    };
    let all_live = store.live_node_count() == store.node_count();
    let whole = all_live
        && match label {
            None => true,
            Some(l) => store.nodes_with_label(l).len() == store.node_count(),
        };
    if whole {
        (0..store.node_count()).for_each(&mut visit);
    } else {
        match label {
            Some(l) => store
                .nodes_with_label(l)
                .iter()
                .for_each(|&id| visit(id as usize)),
            None => (0..store.node_count()).for_each(|i| {
                if store.is_alive(i as u32) {
                    visit(i);
                }
            }),
        }
    }
    // One output COLUMN per aggregate, each a single row (a scalar aggregate emits
    // exactly one row).
    let cols: Vec<Col> = specs
        .iter()
        .zip(&acc)
        .map(|(&(col, func), &(total, cnt, best))| {
            let v = match func {
                AggFn::Count if col.is_none() => Value::Num(rows as f64), // count(*)
                AggFn::Count => Value::Num(cnt as f64),                   // count(arg)
                AggFn::Sum => Value::Num(total),                          // 0.0 over empty (K0a)
                AggFn::Avg => {
                    if cnt == 0 {
                        Value::Null
                    } else {
                        Value::Num(total / cnt as f64)
                    }
                }
                _ => best.map_or(Value::Null, Value::Num), // min/max of nothing → NULL
            };
            Col::Gen(vec![v])
        })
        .collect();
    Some(Batch::of(cols))
}

/// Try to answer a scalar `count(*)` / `count(DISTINCT <last slot>)` sitting on
/// an Expand of a Scan/Expand chain WITHOUT materializing the wide intermediate
/// batch: the frontier feeding the final hop is produced by [`frontier_ids`],
/// then `count(*)` sums the final hop's matching degree and `count(DISTINCT c)`
/// marks endpoints in a bitset over node ids. Returns `None` (fall back to the
/// general aggregate) for any shape it does not recognize — so it is an
/// optimization, never a semantic fork.
/// Peel exactly `n` OUTgoing frontier hops (no bound edge) ending at a bare Scan,
/// returning the per-hop edge labels FIRST-to-LAST and the Scan's label. `None`
/// unless the plan is precisely that chain (used by the 3-hop edge-product count).
pub(super) fn peel_out_hops(plan: &Plan, n: usize) -> Option<(Vec<Vec<String>>, Option<String>)> {
    if n == 0 {
        return match plan {
            Plan::Scan { label } => Some((Vec::new(), label.clone())),
            _ => None,
        };
    }
    let Plan::Expand {
        input,
        from,
        dir: Dir::Out,
        edge_label,
        bind_edge: false,
        double_loops: false,
    } = plan
    else {
        return None;
    };
    if *from + 1 != chain_width(input)? {
        return None; // must expand the current frontier
    }
    let (mut labels, base) = peel_out_hops(input, n - 1)?;
    labels.push(edge_label.clone());
    Some((labels, base))
}

/// count(*) over a 3-hop OUT chain via the identity `1ᵀA₁A₂A₃1 = Σ` over the MIDDLE
/// edges (b→c, hop 2) of `(source→b walks over hop 1) × (out-degree of c over hop
/// 3)` — O(V+E), replacing the 2-hop count-propagation SCATTER (the 3-hop
/// bottleneck: random `next[nbr] += c` writes) with degree products. A fixed chain
/// is a WALK (edges may repeat), so there is NO trail correction — byte-identical
/// to the propagation. Per-hop edge types are handled independently.
pub(super) fn try_3hop_product_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count || agg.arg.is_some() || agg.distinct {
        return None;
    }
    let (labels, base) = peel_out_hops(input, 3)?;
    let mut wants: Vec<Vec<u32>> = Vec::with_capacity(3);
    for l in &labels {
        match want_etypes(store, l) {
            Ok(w) => wants.push(w),
            Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no paths
        }
    }
    let (w1, w2, w3) = (&wants[0], &wants[1], &wants[2]);
    let nc = store.node_count();
    // Empty want = any type; else the edge must carry one of the hop's labels
    // (primary or, on a multi-label graph, secondary).
    let hit = |a: &crate::store::Adj, w: &[u32]| edge_carries_wanted(store, a, w);

    // level1[b] = number of hop-1 edges from a SOURCE into b (= counts after 1 hop).
    let mut level1 = vec![0u64; nc];
    let bump = |s: u32, level1: &mut [u64]| {
        for a in store.out(s) {
            if hit(a, w1) {
                level1[a.nbr as usize] += 1;
            }
        }
    };
    match &base {
        Some(l) => {
            for &s in store.nodes_with_label(l) {
                bump(s, &mut level1);
            }
        }
        None => {
            for s in 0..nc as u32 {
                if store.is_alive(s) {
                    bump(s, &mut level1);
                }
            }
        }
    }
    // outdeg3[c] = number of hop-3 out-edges of c.
    let mut outdeg3 = vec![0u64; nc];
    for (c, d) in outdeg3.iter_mut().enumerate() {
        *d = store.out(c as u32).iter().filter(|a| hit(a, w3)).count() as u64;
    }
    // Σ over hop-2 middle edges (b→c) of level1[b] × outdeg3[c].
    let mut total = 0u64;
    for (b, &lvl) in level1.iter().enumerate() {
        if lvl == 0 {
            continue;
        }
        for a in store.out(b as u32) {
            if hit(a, w2) {
                total += lvl * outdeg3[a.nbr as usize];
            }
        }
    }
    Some(scalar_num(total as f64))
}

pub(super) fn try_fused_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    if !keys.is_empty() || aggs.len() != 1 {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Count {
        return None;
    }
    let Plan::Expand {
        input: inner,
        from,
        dir,
        edge_label,
        double_loops,
        ..
    } = input
    else {
        return None;
    };
    // Gremlin `both()` walks a self-loop twice — the final-hop degree counts it twice.
    let dl = *double_loops;
    let w = chain_width(inner)?; // slot count feeding the final hop
    if *from + 1 != w {
        return None; // the final Expand must expand the current frontier
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown label: zero rows
    };
    let src = frontier_ids(inner, store)?; // ids feeding the final hop, w/ multiplicity

    if agg.arg.is_none() {
        // DEEP chain (≥2 hops feed the final hop, so the intermediate frontier would
        // explode with path multiplicity): propagate a per-node count array instead
        // of materializing the frontier ids — O(hops * edges) time, O(node_count)
        // space. The count is Σ_v counts[v] * matching-out-degree(v).
        if count_hops(inner) >= 2 {
            if let Some(counts) = frontier_counts(inner, store) {
                let mut total = 0f64;
                counts.for_each(|v, c| {
                    let mut deg = 0f64;
                    for_each_nbr(store, v, *dir, &want, dl, |_, _| deg += 1.0);
                    total += c * deg;
                });
                return Some(scalar_num(total));
            }
        }
        // count(*): number of final-hop paths = sum over sources of matching
        // out-degree. When the sources come from an Expand they repeat (many paths
        // reach the same node), and a node's degree is the same each time — so
        // collapse to distinct nodes with multiplicity and walk each adjacency
        // once, scaled. When they come from a Scan they are already distinct, so
        // that dedup is pure overhead: sum degrees directly.
        //
        // When the hop's type set matches EVERY edge (an unlabeled hop, or the
        // graph's only type), the degree is the raw adjacency length — no per-edge
        // type check. That is the common "count all my out-neighbours" shape; the
        // per-edge walk it replaces was the one 1-hop-count regression vs the TS engine.
        let all_types = want_covers_all_etypes(store, &want);
        let mut total = 0f64;
        if matches!(inner.as_ref(), Plan::Expand { .. }) {
            let (distinct, mult) = distinct_with_mult(&src, store.node_count());
            for (i, &v) in distinct.iter().enumerate() {
                total += mult[i] * matching_degree(store, v, *dir, &want, dl, all_types);
            }
        } else {
            for &v in &src {
                total += matching_degree(store, v, *dir, &want, dl, all_types);
            }
        }
        return Some(scalar_num(total));
    }
    if agg.distinct {
        // count(DISTINCT c) where c is the final (last) slot, index == w: distinct
        // endpoints deduped in a bitset — no per-row hashing, no boxed values.
        match agg.arg.as_ref() {
            Some(Expr::Slot(s)) if *s == w => {}
            _ => return None,
        }
        // The distinct endpoints depend only on the SET of last-hop sources, not
        // their multiplicity: a source reached by many paths yields the same
        // neighbours each time. When the sources come from an Expand they repeat,
        // so collapse them to distinct nodes first — a 2-hop's millions of repeated
        // intermediates down to the distinct nodes, each final hop walked once.
        // Sources from a Scan are already distinct, so skip that pass.
        let nc = store.node_count();
        let deduped;
        let sources: &[u32] = if matches!(inner.as_ref(), Plan::Expand { .. }) {
            let mut seen_src = vec![false; nc];
            let mut distinct_src = Vec::new();
            for &v in &src {
                if !seen_src[v as usize] {
                    seen_src[v as usize] = true;
                    distinct_src.push(v);
                }
            }
            deduped = distinct_src;
            &deduped
        } else {
            &src
        };
        let mut seen = vec![false; nc];
        let mut cnt = 0f64;
        for &v in sources {
            for_each_nbr(store, v, *dir, &want, false, |nbr, _| {
                if !seen[nbr as usize] {
                    seen[nbr as usize] = true;
                    cnt += 1.0;
                }
            });
        }
        return Some(scalar_num(cnt));
    }
    None // count(arg) non-distinct on the final slot: not fused (uncommon)
}

/// A one-row, one-column batch holding a single number — a scalar aggregate's
/// result.
pub(super) fn scalar_num(x: f64) -> Batch {
    Batch::of(vec![Col::Gen(vec![Value::Num(x)])])
}

/// A Gremlin `tree()` accumulator: a nested, INSERTION-ORDERED map (children keyed by
/// value, matched via `value::equals`), materialized into nested `Value::Map`s.
#[derive(Default)]
pub(super) struct GremlinTree {
    // Children in FIRST-SEEN order (the Gremlin tree contract) …
    order: Vec<(Value, GremlinTree)>,
    // … plus a grouping-key → index map so a level with many children is an O(1) hash
    // lookup, not a linear scan comparing full element-map keys (which made a wide
    // tree O(paths · children · map-size) — the dominant `tree()` cost).
    index: FnvMap<Vec<u8>, usize>,
}

impl GremlinTree {
    pub(super) fn insert(&mut self, keys: &[Value]) {
        let Some((first, rest)) = keys.split_first() else {
            return;
        };
        let mut kb = Vec::new();
        crate::value::group_key_into(first, &mut kb);
        let i = match self.index.get(&kb) {
            Some(&i) => i,
            None => {
                let idx = self.order.len();
                self.order.push((first.clone(), GremlinTree::default()));
                self.index.insert(kb, idx);
                idx
            }
        };
        self.order[i].1.insert(rest);
    }

    pub(super) fn to_value(&self) -> Value {
        Value::Map(std::sync::Arc::new(
            self.order
                .iter()
                .map(|(k, c)| (k.clone(), c.to_value()))
                .collect(),
        ))
    }
}

/// A component/cluster id as the ROOT vertex's external-id string — the value the TS engine's
/// `connectedComponent`/`peerPressure` write (`Value::Str(vid.arc(root))`). A root
/// with no external id (never, for a loaded node) reads back NULL.
pub(super) fn root_ext_id(store: &Store, root: u32) -> Value {
    store.node_ext_id(root).map_or(Value::Null, Value::Str)
}

/// Collapse a node-id multiset to (distinct ids in first-seen order, their
/// multiplicities) via a direct-mapped array — node ids are dense, so no hashing.
pub(super) fn distinct_with_mult(nodes: &[u32], node_count_total: usize) -> (Vec<u32>, Vec<f64>) {
    let mut group_of = vec![u32::MAX; node_count_total];
    let mut distinct: Vec<u32> = Vec::new();
    let mut mult: Vec<f64> = Vec::new();
    for &id in nodes {
        let slot = &mut group_of[id as usize];
        if *slot == u32::MAX {
            *slot = u32::try_from(distinct.len()).expect("distinct count fits in u32");
            distinct.push(id);
            mult.push(1.0);
        } else {
            mult[*slot as usize] += 1.0;
        }
    }
    (distinct, mult)
}

/// Does `expr` reference no slot other than `s` (and never the path)? Literals
/// and comparisons over slot `s` qualify; any other slot, or `Expr::Path`,
/// disqualifies — the signal that the frontier alone is enough to evaluate it.
pub(super) fn refs_only_slot(expr: &Expr, s: usize) -> bool {
    match expr {
        Expr::Lit(_) | Expr::Param(_) => true,
        Expr::Slot(n) => *n == s,
        Expr::Prop { slot, .. } => *slot == s,
        Expr::Path
        | Expr::PathAccess { .. }
        | Expr::GremlinPath { .. }
        | Expr::GremlinFullPath { .. } => false,
        Expr::Not(x) => refs_only_slot(x, s),
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Arith {
            left: a, right: b, ..
        }
        | Expr::In {
            needle: a,
            haystack: b,
        } => refs_only_slot(a, s) && refs_only_slot(b, s),
        Expr::Call { args, .. } | Expr::GraphPred { args, .. } | Expr::List { items: args } => {
            args.iter().all(|a| refs_only_slot(a, s))
        }
        Expr::Record { fields }
        | Expr::MapLit {
            entries: fields, ..
        } => fields.iter().all(|(_, e)| refs_only_slot(e, s)),
        Expr::Field { base, .. } => refs_only_slot(base, s),
        Expr::Index { base, index, .. } => refs_only_slot(base, s) && refs_only_slot(index, s),
        Expr::Case {
            branches,
            otherwise,
        } => {
            branches
                .iter()
                .all(|(c, v)| refs_only_slot(c, s) && refs_only_slot(v, s))
                && otherwise.as_deref().is_none_or(|e| refs_only_slot(e, s))
        }
        Expr::Compare { left, right, .. } => refs_only_slot(left, s) && refs_only_slot(right, s),
        Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => refs_only_slot(expr, s),
        Expr::PropertyExists { slot, .. } | Expr::IsLabeled { slot, .. } => *slot == s,
        // An EXISTS correlates on outer slots below `outer_width`; conservatively
        // treat it as touching more than one, so it never rides the frontier-only
        // aggregate fast path.
        Expr::Exists { .. }
        | Expr::CountSubquery { .. }
        | Expr::ScalarSubquery { .. }
        | Expr::CollectSubquery { .. }
        | Expr::AggSubquery { .. }
        | Expr::UncorrelatedExists { .. }
        | Expr::UncorrelatedCount { .. }
        | Expr::UncorrelatedScalar { .. } => false,
    }
}

/// Rewrite every reference to slot `from` in `expr` to slot `to`. Used to retarget
/// frontier-only expressions onto a one-slot frontier batch. Callers guarantee
/// (via [`refs_only_slot`]) that no other slot appears.
pub(super) fn remap_slot(expr: &Expr, from: usize, to: usize) -> Expr {
    let go = |e| Box::new(remap_slot(e, from, to));
    match expr {
        Expr::Slot(n) if *n == from => Expr::Slot(to),
        Expr::Prop { slot, key } if *slot == from => Expr::Prop {
            slot: to,
            key: key.clone(),
        },
        Expr::Slot(_)
        | Expr::Prop { .. }
        | Expr::Lit(_)
        | Expr::Path
        | Expr::PathAccess { .. }
        | Expr::GremlinPath { .. }
        | Expr::GremlinFullPath { .. } => expr.clone(),
        Expr::Not(x) => Expr::Not(go(x)),
        Expr::And(a, b) => Expr::And(go(a), go(b)),
        Expr::Or(a, b) => Expr::Or(go(a), go(b)),
        Expr::Xor(a, b) => Expr::Xor(go(a), go(b)),
        Expr::In { needle, haystack } => Expr::In {
            needle: go(needle),
            haystack: go(haystack),
        },
        Expr::Compare { op, left, right } => Expr::Compare {
            op: *op,
            left: go(left),
            right: go(right),
        },
        Expr::Arith { op, left, right } => Expr::Arith {
            op: *op,
            left: go(left),
            right: go(right),
        },
        Expr::Call { name, args } => Expr::Call {
            name: name.clone(),
            args: args.iter().map(|a| remap_slot(a, from, to)).collect(),
        },
        Expr::GraphPred { op, args, negated } => Expr::GraphPred {
            op: *op,
            args: args.iter().map(|a| remap_slot(a, from, to)).collect(),
            negated: *negated,
        },
        Expr::List { items } => Expr::List {
            items: items.iter().map(|a| remap_slot(a, from, to)).collect(),
        },
        Expr::Record { fields } => Expr::Record {
            fields: fields
                .iter()
                .map(|(k, e)| (k.clone(), remap_slot(e, from, to)))
                .collect(),
        },
        Expr::MapLit {
            entries,
            omit_absent,
        } => Expr::MapLit {
            entries: entries
                .iter()
                .map(|(k, e)| (k.clone(), remap_slot(e, from, to)))
                .collect(),
            omit_absent: *omit_absent,
        },
        Expr::Index { base, index, elem } => Expr::Index {
            base: go(base),
            index: go(index),
            elem: *elem,
        },
        Expr::Field { base, key } => Expr::Field {
            base: go(base),
            key: key.clone(),
        },
        Expr::Case {
            branches,
            otherwise,
        } => Expr::Case {
            branches: branches
                .iter()
                .map(|(c, v)| (remap_slot(c, from, to), remap_slot(v, from, to)))
                .collect(),
            otherwise: otherwise
                .as_ref()
                .map(|e| Box::new(remap_slot(e, from, to))),
        },
        Expr::Cast { target, expr } => Expr::Cast {
            target: *target,
            expr: go(expr),
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: go(expr),
            negated: *negated,
        },
        Expr::PropertyExists { slot, key } => Expr::PropertyExists {
            slot: if *slot == from { to } else { *slot },
            key: key.clone(),
        },
        Expr::IsLabeled { slot, labels } => Expr::IsLabeled {
            slot: if *slot == from { to } else { *slot },
            labels: labels.clone(),
        },
        // Never reached: `refs_only_slot` rejects EXISTS, so the frontier remap
        // that calls this is never handed one. Clone rather than rewrite a body.
        Expr::Exists { .. }
        | Expr::CountSubquery { .. }
        | Expr::ScalarSubquery { .. }
        | Expr::CollectSubquery { .. }
        | Expr::AggSubquery { .. }
        | Expr::UncorrelatedExists { .. }
        | Expr::UncorrelatedCount { .. }
        | Expr::UncorrelatedScalar { .. }
        | Expr::Param(_) => expr.clone(),
    }
}

/// `count(*)` grouped by a single property of the frontier node, computed by
/// grouping on the integer node id FIRST, then merging node groups by the
/// property value. The property is a function of the node, so two rows on the
/// same node share a property value: counting 8M endpoints by their (cheap,
/// dense) node id and reading/hashing the property for only the distinct nodes
/// replaces millions of string hashes and `Arc` clones with a direct-mapped
/// array index each. The final hop is fused into the count — endpoints are
/// streamed straight into the array, never materialized as a column. First-seen
/// order is preserved: the distinct nodes are visited in first-appearance order,
/// so a property value is first seen at the earliest node — hence earliest row —
/// carrying it. `None` for any other shape (non-count aggregate, key that is not
/// a lone frontier property), which falls through to the general frontier path.
///
/// Rejected optimization: for a DICT-encoded key, counting straight into per-code
/// buckets during the traversal (`counts[codes[nbr]] += 1`), skipping this per-node
/// intermediate and the Level-2 merge. It moved `c.city, count(*)` on the 2-hop
/// 100k/deg-5 fixture only 24.5ms -> 23.0ms (0.54x -> 0.57x of the TS engine) — a consistent
/// ~7% but still far from parity, and it TRADES the per-node scatter for reading the
/// property once PER PATH (2.5M reads) instead of once per distinct endpoint (100k).
/// The shape is memory-bound on ~2.5M random accesses either way; the TS engine's remaining
/// edge is its CSR adjacency (sequential neighbour reads), which the per-node `Vec`
/// adjacency here cannot match without a layout change (deferred, large blast radius).
/// Not worth a second grouped-count path for a sub-10% move that leaves it slowest.
pub(super) fn try_node_grouped_count(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Option<Batch> {
    let [agg] = aggs else { return None };
    if agg.func != AggFn::Count || agg.arg.is_some() {
        return None;
    }
    let [(_, key_expr)] = keys else { return None };
    // The group node is the endpoint of a final Expand over a Scan/Expand chain.
    let Plan::Expand {
        input: inner,
        from,
        dir,
        edge_label,
        bind_edge,
        double_loops: false,
    } = input
    else {
        return None;
    };
    if *bind_edge {
        // With the edge bound, the endpoint node sits at slot w+1 and slot w is the
        // EDGE — so a `Prop{slot: w}` key is an edge property, not a node one. This
        // fast path reads NODE properties of the endpoint; hand a bound-edge group
        // (e.g. `RETURN r.w, count(*)`) to the general aggregate, which reads the
        // edge slot correctly. (Found by the differential fuzzer: this used to read
        // the edge key as an absent node property and bucket every row under NULL.)
        return None;
    }
    let w = chain_width(inner)?;
    if *from + 1 != w {
        return None; // the final Expand must expand the current frontier
    }
    let Expr::Prop { slot, key } = key_expr else {
        return None;
    };
    if *slot != w {
        return None; // key must read the endpoint (last) slot, index == w
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(Batch::of(vec![Col::Nodes(vec![]), Col::Gen(vec![])])),
    };
    let src = frontier_ids(inner, store)?; // nodes feeding the final hop, w/ multiplicity

    // Level 1: count per endpoint node id via a direct-mapped array (no hashing —
    // node ids are dense), with the final hop fused in so endpoints never
    // materialize. Distinct ids come out in first-seen order.
    let mut group_of = vec![u32::MAX; store.node_count()];
    let mut rep_ids: Vec<u32> = Vec::new();
    let mut node_count: Vec<f64> = Vec::new();
    for &v in &src {
        for_each_nbr(store, v, *dir, &want, false, |nbr, _| {
            let slot = &mut group_of[nbr as usize];
            if *slot == u32::MAX {
                *slot = u32::try_from(rep_ids.len()).expect("group count fits in u32");
                rep_ids.push(nbr);
                node_count.push(1.0);
            } else {
                node_count[*slot as usize] += 1.0;
            }
        });
    }

    // Read the grouping property for the DISTINCT endpoint nodes only.
    let key_col = read_property(store, &Col::Nodes(rep_ids), key);

    // Level 2: merge node groups by property value, summing their counts.
    let (val_of, val_first) = assign_groups(std::slice::from_ref(&key_col), key_col.len());
    let mut counts = vec![0f64; val_first.len()];
    for (node_group, &vg) in val_of.iter().enumerate() {
        counts[vg as usize] += node_count[node_group];
    }
    let key_out = key_col.gather(&val_first);
    Some(Batch::of(vec![
        key_out,
        Col::Gen(counts.into_iter().map(Value::Num).collect()),
    ]))
}

/// Run a grouped/scalar aggregate over a Scan/Expand chain WITHOUT materializing
/// the earlier slots: when every key and aggregate argument reads only the
/// frontier (last) slot, the chain's frontier is all the aggregate needs. The
/// frontier ([`frontier_ids`]) is produced in the same row order the full batch
/// would have, so first-seen group order — and every value — is identical to the
/// general path; this only drops the wasted slot columns. `None` for any shape it
/// does not handle (a filter/join in the chain, an expression over an earlier
/// slot), which falls back to the general aggregate.
pub(super) fn try_frontier_aggregate(
    input: &Plan,
    keys: &[(String, Expr)],
    aggs: &[Agg],
    store: &Store,
) -> Result<Option<Batch>, String> {
    let Some(width) = chain_width(input) else {
        return Ok(None);
    };
    let last = width - 1; // frontier slot index of the whole chain
    let key_ok = keys.iter().all(|(_, e)| refs_only_slot(e, last));
    let agg_ok = aggs
        .iter()
        .all(|a| a.arg.as_ref().is_none_or(|e| refs_only_slot(e, last)));
    if !key_ok || !agg_ok {
        return Ok(None);
    }
    let Some(frontier) = frontier_ids(input, store) else {
        return Ok(None);
    };
    let batch = Batch::of(vec![Col::Nodes(frontier)]);
    // Retarget the frontier-only expressions onto the one-slot frontier batch.
    let keys: Vec<(String, Expr)> = keys
        .iter()
        .map(|(n, e)| (n.clone(), remap_slot(e, last, 0)))
        .collect();
    let aggs: Vec<Agg> = aggs
        .iter()
        .map(|a| Agg {
            func: a.func,
            arg: a.arg.as_ref().map(|e| remap_slot(e, last, 0)),
            distinct: a.distinct,
            name: a.name.clone(),
            frac: a.frac,
            null_on_empty: a.null_on_empty,
            // Preserve min()/max()'s cross-type-fault contract: dropping it let a grouped
            // `by(__.values(v).max())` return a value where the streamed spelling faulted.
            numeric_only: a.numeric_only,
        })
        .collect();
    Ok(Some(aggregate(&batch, store, &keys, &aggs)?))
}
