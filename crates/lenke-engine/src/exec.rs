//! Execution: pull a batch up through the plan, then materialize the projection.
//!
//! Expression evaluation is columnar — `eval` produces a `Col` over the whole
//! batch, reading typed storage columns in bulk where it can. It calls the value
//! contract for every comparison and equality; it never restates those rules.
//! This is the lineage-FREE strategy; the lineage-preserving strategy for the
//! same operators lands with the operators (path/tags) that need it.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::batch::{Batch, Col, Lineage};
use crate::ir::{Agg, AggFn, CompareOp, Dir, Expr, Plan};
use crate::store::{Column, Store};
use crate::value::{self, Value};

/// A fast, dependency-free hasher for the engine's INTERNAL grouping, distinct,
/// and join maps. The default `HashMap` hasher (SipHash) is DoS-resistant, which
/// these maps — built and dropped inside one operator over trusted, already-
/// materialized keys — do not need; FNV-1a is several times faster on the short
/// byte/integer keys grouping produces. It never escapes the executor, so hash
/// quality only affects speed, never results.
mod fnv {
    use std::collections::{HashMap, HashSet};
    use std::hash::{BuildHasherDefault, Hasher};

    pub type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fnv>>;
    pub type Set<K> = HashSet<K, BuildHasherDefault<Fnv>>;

    pub struct Fnv(u64);

    impl Default for Fnv {
        fn default() -> Self {
            Self(0xcbf2_9ce4_8422_2325) // FNV-1a 64-bit offset basis
        }
    }

    impl Hasher for Fnv {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            let mut h = self.0;
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
            }
            self.0 = h;
        }
    }
}
use fnv::{Map as FnvMap, Set as FnvSet};

/// A materialized result: column names and rows of values. `Value` intentionally
/// has no `PartialEq` (f64/NaN policy lives in the value contract, not a derive),
/// so compare results through `value::equals`/`cmp_total`, not `==`.
#[derive(Debug)]
pub struct Rows {
    pub names: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Run `plan` over `store`, returning materialized rows. Output column names come
/// from the outermost naming operator (`Project` or `Aggregate`, seen through
/// `Distinct`/`OrderPage`); a plan with none surfaces slot 0 under a single
/// implicit column so partial plans stay runnable in tests.
#[must_use]
pub fn run(plan: &Plan, store: &Store) -> Rows {
    try_run(plan, store).expect("read plan evaluation faulted")
}

/// The fallible core of [`run`]: an expression can fault at runtime (a failed
/// `CAST` throws `E_INVALID_VALUE`), so the read pipeline threads a `Result`. A
/// plan that never evaluates a fallible expression cannot error, which is why
/// [`run`] can wrap this with `.expect` — the panic path is unreachable for such
/// plans, and callers that may run user CASTs use `try_run` (or `execute`).
pub fn try_run(plan: &Plan, store: &Store) -> Result<Rows, String> {
    // Lineage is plan-global: if anything reads the path, the whole plan tracks
    // it (Scan seeds, Expand extends); otherwise no operator builds a sidecar and
    // the query pays nothing for lineage.
    let track = needs_lineage(plan);
    let batch = pull(plan, store, track)?;
    let n = batch.rows();
    Ok(match output_names(plan) {
        Some(names) => {
            let rows = (0..n)
                .map(|i| batch.slots.iter().map(|c| c.value_at(i)).collect())
                .collect();
            Rows { names, rows }
        }
        None => {
            let slot0 = batch.slot(0);
            let rows = (0..n).map(|i| vec![slot0.value_at(i)]).collect();
            Rows {
                names: vec!["_".to_string()],
                rows,
            }
        }
    })
}

/// Run a plan that MAY write, against a mutable store. A write plan (`Insert`)
/// mutates the store and returns no rows; any other plan is a pure read and is
/// dispatched to [`run`] over a shared borrow. This is the entry point for
/// statements that can mutate; read-only callers can keep using [`run`]. Returns
/// `Err` when a write violates a constraint (the write is rolled back); reads and
/// successful writes are `Ok` (a write's result is the empty row set).
pub fn execute(plan: &Plan, store: &mut Store) -> Result<Rows, String> {
    match plan {
        Plan::Insert { nodes, edges } => {
            // In a transaction so a constraint violation rolls the whole INSERT
            // back rather than leaving a partial write.
            store.begin();
            let mut ids = Vec::with_capacity(nodes.len());
            for spec in nodes {
                let labels: Vec<&str> = spec.labels.iter().map(String::as_str).collect();
                let props: Vec<(&str, Value)> = spec
                    .props
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.clone()))
                    .collect();
                ids.push(store.add_node(&labels, &props));
            }
            for e in edges {
                let eid = store.add_edge(ids[e.from], ids[e.to], &e.etype);
                for (k, v) in &e.props {
                    store.set_edge_prop(eid, k, v.clone());
                }
            }
            // Enforce unique AND required constraints on every label this INSERT
            // touched (roll the whole INSERT back on the first violation).
            let mut labels: Vec<&str> = nodes
                .iter()
                .flat_map(|s| s.labels.iter().map(String::as_str))
                .collect();
            labels.sort_unstable();
            labels.dedup();
            for l in labels {
                if let Err(e) = store
                    .check_unique_for_label(l)
                    .and_then(|()| store.check_required_for_label(l))
                {
                    store.rollback();
                    return Err(e);
                }
            }
            store.commit();
            Ok(empty_rows())
        }
        Plan::Update { input, ops } => {
            // Read phase: run the match and compute every write into OWNED data —
            // so the immutable borrow ends before the write phase mutates. A slot
            // may be a node frontier or (bound relationship) an edge frontier;
            // SET/REMOVE dispatch on which, so `r.weight` writes an edge property.
            enum Applied {
                Set(u32, String, Value),
                Remove(u32, String),
                Delete(u32),
                SetEdge(u32, String, Value),
                RemoveEdge(u32, String),
            }
            let mut applied: Vec<Applied> = Vec::new();
            {
                let track = needs_lineage(input);
                let batch = pull(input, store, track)?;
                for op in ops {
                    match op {
                        crate::ir::SetOp::Set { slot, key, value } => {
                            let vals = eval(value, store, &batch)?;
                            match batch.slot(*slot) {
                                Col::Nodes(ids) => {
                                    for (i, &id) in ids.iter().enumerate() {
                                        applied.push(Applied::Set(
                                            id,
                                            key.clone(),
                                            vals.value_at(i),
                                        ));
                                    }
                                }
                                Col::Edges(eids) => {
                                    for (i, &e) in eids.iter().enumerate() {
                                        applied.push(Applied::SetEdge(
                                            e,
                                            key.clone(),
                                            vals.value_at(i),
                                        ));
                                    }
                                }
                                _ => {}
                            }
                        }
                        crate::ir::SetOp::Remove { slot, key } => match batch.slot(*slot) {
                            Col::Nodes(ids) => {
                                for &id in ids {
                                    applied.push(Applied::Remove(id, key.clone()));
                                }
                            }
                            Col::Edges(eids) => {
                                for &e in eids {
                                    applied.push(Applied::RemoveEdge(e, key.clone()));
                                }
                            }
                            _ => {}
                        },
                        crate::ir::SetOp::Delete { slot } => {
                            // Edge deletion via drop() needs the endpoints (not just
                            // the eid); deferred with Gremlin addE. Node delete here.
                            if let Col::Nodes(ids) = batch.slot(*slot) {
                                for &id in ids {
                                    applied.push(Applied::Delete(id));
                                }
                            }
                        }
                    }
                }
            }
            // Write phase: apply in op/row order (last write wins per element+key;
            // delete_node is idempotent for a node matched by several rows).
            for a in applied {
                match a {
                    Applied::Set(node, key, value) => store.set_prop(node, &key, value),
                    Applied::Remove(node, key) => store.remove_prop(node, &key),
                    Applied::SetEdge(eid, key, value) => store.set_edge_prop(eid, &key, value),
                    Applied::RemoveEdge(eid, key) => store.remove_edge_prop(eid, &key),
                    Applied::Delete(node) => store.delete_node(node),
                }
            }
            Ok(empty_rows())
        }
        Plan::Merge {
            label,
            props,
            on_create,
            on_update,
        } => execute_merge(store, label, props, on_create, on_update),
        Plan::AddEdge {
            from,
            to,
            etype,
            props,
        } => {
            let nc = u32::try_from(store.node_count()).unwrap_or(u32::MAX);
            if *from >= nc || *to >= nc || !store.is_alive(*from) || !store.is_alive(*to) {
                return Err(format!(
                    "addE: endpoint out of range or deleted ({from} -> {to})"
                ));
            }
            let eid = store.add_edge(*from, *to, etype);
            for (k, v) in props {
                store.set_edge_prop(eid, k, v.clone());
            }
            Ok(empty_rows())
        }
        _ => try_run(plan, store),
    }
}

/// Execute a `_MERGE`: infer the key from a unique constraint, find the existing
/// node by its key values, and take the create or update path. Runs in a
/// transaction so a constraint violation (or a no-applicable-constraint error)
/// leaves the store untouched.
fn execute_merge(
    store: &mut Store,
    label: &str,
    props: &[(String, Value)],
    on_create: &[(String, Expr)],
    on_update: &crate::ir::MergeUpdate,
) -> Result<Rows, String> {
    use crate::ir::MergeUpdate;
    store.begin();
    let have: Vec<String> = props.iter().map(|(k, _)| k.clone()).collect();
    let key_keys = match store.infer_merge_key(label, &have) {
        Ok(k) => k,
        Err(e) => {
            store.rollback();
            return Err(e);
        }
    };
    // The pattern's key-tuple bytes, and a finder that matches an existing node.
    let want = key_bytes(&key_keys, |k| pattern_value(props, k));
    let found = store
        .nodes_with_label(label)
        .iter()
        .copied()
        .find(|&id| key_bytes(&key_keys, |k| store.prop(id, k)) == want);

    match found {
        Some(id) => match on_update {
            MergeUpdate::Nothing => {}
            MergeUpdate::Clobber => {
                // Set every non-key payload property to the pattern's value.
                for (k, v) in props {
                    if !key_keys.contains(k) {
                        store.set_prop(id, k, v.clone());
                    }
                }
            }
            MergeUpdate::Set { assigns, filter } => {
                let batch = Batch::of(vec![Col::Nodes(vec![id])]);
                // Evaluate the gate and every assignment BEFORE mutating; a fault
                // (e.g. a failed CAST) rolls the whole MERGE back rather than
                // leaving the begun transaction open.
                let gate = match filter.as_ref().map(|f| eval(f, store, &batch)).transpose() {
                    Ok(g) => g.is_none_or(|c| matches!(c.value_at(0), Value::Bool(true))),
                    Err(e) => {
                        store.rollback();
                        return Err(e);
                    }
                };
                if gate {
                    let writes: Result<Vec<(String, Value)>, String> = assigns
                        .iter()
                        .map(|(k, e)| Ok((k.clone(), eval(e, store, &batch)?.value_at(0))))
                        .collect();
                    match writes {
                        Ok(writes) => {
                            for (k, v) in writes {
                                store.set_prop(id, &k, v);
                            }
                        }
                        Err(e) => {
                            store.rollback();
                            return Err(e);
                        }
                    }
                }
            }
        },
        None => {
            let props_ref: Vec<(&str, Value)> =
                props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            let id = store.add_node(&[label], &props_ref);
            let batch = Batch::of(vec![Col::Nodes(vec![id])]);
            let writes: Result<Vec<(String, Value)>, String> = on_create
                .iter()
                .map(|(k, e)| Ok((k.clone(), eval(e, store, &batch)?.value_at(0))))
                .collect();
            match writes {
                Ok(writes) => {
                    for (k, v) in writes {
                        store.set_prop(id, &k, v);
                    }
                }
                Err(e) => {
                    store.rollback();
                    return Err(e);
                }
            }
        }
    }

    if let Err(e) = store
        .check_unique_for_label(label)
        .and_then(|()| store.check_required_for_label(label))
    {
        store.rollback();
        return Err(e);
    }
    store.commit();
    Ok(empty_rows())
}

/// The grouping-key bytes of `keys`, reading each key's value via `get`.
fn key_bytes(keys: &[String], mut get: impl FnMut(&str) -> Value) -> Vec<u8> {
    let mut buf = Vec::new();
    for k in keys {
        value::group_key_into(&get(k), &mut buf);
    }
    buf
}

/// A pattern property's value by key (NULL if the pattern does not name it).
fn pattern_value(props: &[(String, Value)], key: &str) -> Value {
    props
        .iter()
        .find(|(k, _)| k == key)
        .map_or(Value::Null, |(_, v)| v.clone())
}

/// The empty result a write statement returns (no columns, no rows).
fn empty_rows() -> Rows {
    Rows {
        names: Vec::new(),
        rows: Vec::new(),
    }
}

/// The output column names a plan produces, seen through row-shape-preserving
/// operators (`Distinct`, `OrderPage`) down to the naming one. `None` means no
/// explicit projection — the row is the raw slot-0 frontier.
fn output_names(plan: &Plan) -> Option<Vec<String>> {
    match plan {
        Plan::Project { items, .. } => Some(items.iter().map(|(n, _)| n.clone()).collect()),
        Plan::Aggregate { keys, aggs, .. } => {
            let mut names: Vec<String> = keys.iter().map(|(n, _)| n.clone()).collect();
            names.extend(aggs.iter().map(|a| a.name.clone()));
            Some(names)
        }
        Plan::Distinct { input }
        | Plan::OrderPage { input, .. }
        | Plan::SortLocal { input, .. } => output_names(input),
        _ => None,
    }
}

/// Whether any expression in the plan reads the path (`Expr::Path`) — the signal
/// that lineage must be tracked. Computed once, for the whole plan.
fn needs_lineage(plan: &Plan) -> bool {
    fn reads_path(e: &Expr) -> bool {
        match e {
            // Reading any part of the path needs the lineage, just like `Path`.
            Expr::Path | Expr::PathAccess { .. } => true,
            Expr::Compare { left, right, .. } => reads_path(left) || reads_path(right),
            Expr::Not(x) => reads_path(x),
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Arith {
                left: a, right: b, ..
            } => reads_path(a) || reads_path(b),
            Expr::Call { args, .. } | Expr::List { items: args } => args.iter().any(reads_path),
            Expr::Record { fields } | Expr::MapLit { entries: fields } => {
                fields.iter().any(|(_, e)| reads_path(e))
            }
            Expr::Field { base, .. } => reads_path(base),
            Expr::Case {
                branches,
                otherwise,
            } => {
                branches.iter().any(|(c, v)| reads_path(c) || reads_path(v))
                    || otherwise.as_deref().is_some_and(reads_path)
            }
            Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => reads_path(expr),
            // An EXISTS body reads its OWN (sub-)path, never the outer one, and the
            // seed is built without lineage — so it never forces outer tracking.
            Expr::Slot(_)
            | Expr::Prop { .. }
            | Expr::Lit(_)
            | Expr::PropertyExists { .. }
            | Expr::Exists { .. } => false,
        }
    }
    match plan {
        Plan::Scan { .. }
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. } => false,
        Plan::Expand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Distinct { input }
        | Plan::SortLocal { input, .. } => needs_lineage(input),
        Plan::IntervalExpand {
            input, qlo, qhi, ..
        } => reads_path(qlo) || reads_path(qhi) || needs_lineage(input),
        Plan::Filter { input, pred } => reads_path(pred) || needs_lineage(input),
        Plan::Project { input, items } => {
            items.iter().any(|(_, e)| reads_path(e)) || needs_lineage(input)
        }
        Plan::Aggregate { input, keys, aggs } => {
            keys.iter().any(|(_, e)| reads_path(e))
                || aggs.iter().any(|a| a.arg.as_ref().is_some_and(reads_path))
                || needs_lineage(input)
        }
        Plan::OrderPage { input, keys, .. } => {
            keys.iter().any(|k| reads_path(&k.expr)) || needs_lineage(input)
        }
        Plan::Join { left, right, .. } => needs_lineage(left) || needs_lineage(right),
        // The subquery yields append columns; whether the OUTER plan needs a path
        // depends on its input (a path read inside the subquery is not surfaced).
        Plan::CallInline { input, yields, .. } => {
            needs_lineage(input) || yields.iter().any(|(_, e)| reads_path(e))
        }
        Plan::Update { input, ops } => {
            needs_lineage(input)
                || ops.iter().any(|op| match op {
                    crate::ir::SetOp::Set { value, .. } => reads_path(value),
                    crate::ir::SetOp::Remove { .. } | crate::ir::SetOp::Delete { .. } => false,
                })
        }
    }
}

/// Pull a batch up through a (non-terminal) plan node. `track` is the plan-global
/// lineage decision: when true, row-producing operators build the path sidecar.
fn pull(plan: &Plan, store: &Store, track: bool) -> Result<Batch, String> {
    Ok(match plan {
        // A write plan is never pulled (a read sub-plan cannot contain one); it
        // is run through `execute`. Yield an empty batch if it somehow reaches
        // here so `run` on a bare write is a harmless no-op rather than a panic.
        Plan::Insert { .. } | Plan::Update { .. } | Plan::Merge { .. } | Plan::AddEdge { .. } => {
            Batch::of(Vec::new())
        }
        // `Row` is the leaf of an EXISTS body and is only ever fed a batch by
        // `pull_body`; reaching it through the main pipeline is a bug.
        Plan::Row => {
            return Err("Plan::Row is only valid inside an EXISTS body".into());
        }
        Plan::Scan { label } => {
            let ids = match label {
                Some(l) => store.nodes_with_label(l).to_vec(),
                None => store.all_nodes(),
            };
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::IndexSeek { label, key, value } => {
            let ids = index_seek_ids(store, label, key, value);
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => {
            let ids = range_seek_ids(store, label, key, *op, value);
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge,
        } => expand(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label.as_deref(),
            *bind_edge,
        ),
        Plan::IntervalExpand {
            input,
            from,
            dir,
            edge_label,
            lo_key,
            hi_key,
            qlo,
            qhi,
            bind_edge,
        } => {
            let batch = pull(input, store, track)?;
            let qlo_col = eval(qlo, store, &batch)?;
            let qhi_col = eval(qhi, store, &batch)?;
            interval_expand(
                &batch,
                store,
                *from,
                *dir,
                edge_label.as_deref(),
                lo_key,
                hi_key,
                &qlo_col,
                &qhi_col,
                *bind_edge,
            )
        }
        Plan::Filter { input, pred } => {
            let batch = pull(input, store, track)?;
            // Fast path: `<prop> <cmp> <literal>` reads storage in one pass to
            // keep-indices; otherwise evaluate the predicate as a full column.
            let keep: Vec<usize> = match try_filter_keep(pred, store, &batch) {
                Some(keep) => keep,
                None => {
                    let mask = eval(pred, store, &batch)?;
                    match &mask {
                        Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                        other => (0..other.len())
                            .filter(|&i| other.value_at(i).is_true())
                            .collect(),
                    }
                }
            };
            batch.gather(&keep)
        }
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            trail,
        } => var_length(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label.as_deref(),
            *min,
            *max,
            *trail,
        ),
        Plan::ShortestPath {
            input,
            from,
            dir,
            edge_label,
            max,
        } => shortest_path(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label.as_deref(),
            *max,
        ),
        Plan::Aggregate { input, keys, aggs } => {
            // Frontier fast path: a scalar count over an Expand chain need not
            // build the wide intermediate batch. Falls back to the general
            // aggregate for every shape it does not recognize. (The fused paths
            // never evaluate arbitrary expressions, so they cannot fault.)
            if let Some(b) = try_fused_count(input, keys, aggs, store)
                .or_else(|| try_node_grouped_count(input, keys, aggs, store))
            {
                b
            } else if let Some(b) = try_frontier_aggregate(input, keys, aggs, store)? {
                b
            } else {
                aggregate(&pull(input, store, track)?, store, keys, aggs)?
            }
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } => order_page(&pull(input, store, track)?, store, keys, *skip, *limit)?,
        Plan::Project { input, items } => {
            // Project produces a batch whose slots ARE the projected columns, so
            // an operator above it (Distinct, OrderPage) works on the output
            // values, not the pre-projection bindings.
            let batch = pull(input, store, track)?;
            let cols = eval_all(items.iter().map(|(_, e)| e), store, &batch)?;
            Batch::of(cols)
        }
        Plan::Distinct { input } => {
            let batch = pull(input, store, track)?;
            let n = batch.rows();
            let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
            let mut buf = Vec::new();
            let keep: Vec<usize> = (0..n)
                .filter(|&i| {
                    // Key the whole row across every slot, via the grouping
                    // notion (NaN/-0 collapse) — never predicate equality. The
                    // buffer is reused across rows; only a NEW row allocates.
                    buf.clear();
                    for c in &batch.slots {
                        value::group_key_into(&c.value_at(i), &mut buf);
                    }
                    if seen.contains(buf.as_slice()) {
                        false
                    } else {
                        seen.insert(buf.clone());
                        true
                    }
                })
                .collect();
            batch.gather(&keep)
        }
        Plan::SortLocal { input, descending } => {
            // Gremlin `order(local)`: sort inside each row's slot-0 cell, leaving
            // the batch shape and every other slot untouched. Ordering is the value
            // contract's `cmp_total` (the single home for order); DESC reverses it.
            let batch = pull(input, store, track)?;
            let n = batch.rows();
            let sorted: Vec<Value> = (0..n)
                .map(|i| sort_local_cell(batch.slot(0).value_at(i), *descending))
                .collect();
            let mut slots: Vec<Col> = batch.slots.clone();
            if !slots.is_empty() {
                slots[0] = Col::Gen(sorted);
            }
            let mut out = Batch::of(slots);
            out.lineage = batch.lineage;
            out
        }
        Plan::Join { left, right, on } => {
            hash_join(&pull(left, store, track)?, &pull(right, store, track)?, on)
        }
        Plan::CallInline {
            input,
            body,
            yields,
            outer_width,
        } => {
            // Inline correlated (lateral) subquery: run `body` over the outer rows
            // (it is rooted at `Plan::Row`, which yields them), then emit one row
            // per sub-row — the outer slots the sub-row still carries, followed by
            // the yield expressions. Outer rows with no sub-row drop out (inner
            // lateral join). The subquery's internal variables are NOT surfaced.
            let outer = pull(input, store, track)?;
            let ow = *outer_width;
            let sub = pull_body(body, store, &outer)?;
            let mut out_slots: Vec<Col> = (0..ow).map(|i| sub.slot(i).clone()).collect();
            for (_, e) in yields {
                out_slots.push(eval(e, store, &sub)?);
            }
            let mut out = Batch::of(out_slots);
            // Carry any path the sub-rows accumulated (present only under lineage).
            out.lineage = sub.lineage;
            out
        }
        Plan::CallProcedure { name, config } => {
            // Run the named graph algorithm over the whole store into a two-slot
            // batch: node ids, then the per-node result. The parser validates the
            // name, so an unknown one here is defensive.
            let results = crate::algo::run_procedure(store, name, config)
                .ok_or_else(|| format!("unknown procedure `{name}`"))?;
            let ids: Vec<u32> = results.iter().map(|(v, _)| *v).collect();
            let vals: Vec<f64> = results.iter().map(|(_, r)| *r).collect();
            Batch::of(vec![Col::Nodes(ids), Col::Num(vals)])
        }
    })
}

/// Hash-join two batches on `(left_slot, right_slot)` key equalities. Output is
/// every left slot gathered by the matched left rows, followed by every right
/// slot gathered by the matched right rows — so right slot `j` lands at output
/// slot `left.len() + j`. Keys are `group_key`-hashed (bound-variable identity),
/// consistent with grouping/distinct.
fn join_key(batch: &Batch, slots: impl Iterator<Item = usize>, row: usize) -> Vec<u8> {
    let mut k = Vec::new();
    for s in slots {
        value::group_key_into(&batch.slot(s).value_at(row), &mut k);
    }
    k
}

fn hash_join(lb: &Batch, rb: &Batch, on: &[(usize, usize)]) -> Batch {
    // Index the right side by its join key.
    let mut index: FnvMap<Vec<u8>, Vec<usize>> = FnvMap::default();
    for j in 0..rb.rows() {
        let k = join_key(rb, on.iter().map(|&(_, r)| r), j);
        index.entry(k).or_default().push(j);
    }
    // Probe with the left side, emitting one combined row per match (a shared key
    // with several matches on each side fans out to their product).
    let mut keep_l = Vec::new();
    let mut keep_r = Vec::new();
    for i in 0..lb.rows() {
        let k = join_key(lb, on.iter().map(|&(l, _)| l), i);
        if let Some(js) = index.get(&k) {
            for &j in js {
                keep_l.push(i);
                keep_r.push(j);
            }
        }
    }
    let mut slots: Vec<Col> = lb.slots.iter().map(|c| c.gather(&keep_l)).collect();
    slots.extend(rb.slots.iter().map(|c| c.gather(&keep_r)));
    Batch::of(slots)
}

/// Sort the batch by `keys` (stable; ascending via `cmp_total`, descending its
/// reverse), then keep the window `[skip, skip+limit)`. Reorders every slot
/// together, so bound variables stay row-aligned.
fn order_page(
    batch: &Batch,
    store: &Store,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
) -> Result<Batch, String> {
    let n = batch.rows();
    let mut idx: Vec<usize> = (0..n).collect();
    if !keys.is_empty() {
        let key_cols: Vec<Col> = eval_all(keys.iter().map(|k| &k.expr), store, batch)?;
        // Stable sort: equal keys keep input order, so the last key's ties fall
        // back to arrival order deterministically.
        idx.sort_by(|&a, &b| {
            for (kc, k) in key_cols.iter().zip(keys) {
                let (va, vb) = (kc.value_at(a), kc.value_at(b));
                // NULL placement is set by the front-end (GQL last, Gremlin first)
                // and is INDEPENDENT of direction — so it is decided here, before
                // the total order, not by reversing `cmp_total` (whose rank would
                // otherwise flip NULLs to the wrong end under DESC).
                let ord = match (va.is_null(), vb.is_null()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => {
                        if k.nulls_first {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    }
                    (false, true) => {
                        if k.nulls_first {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    }
                    (false, false) => {
                        let o = value::cmp_total(&va, &vb);
                        if k.descending {
                            o.reverse()
                        } else {
                            o
                        }
                    }
                };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    let start = skip.unwrap_or(0).min(idx.len());
    let end = limit.map_or(idx.len(), |l| start.saturating_add(l).min(idx.len()));
    Ok(batch.gather(&idx[start..end]))
}

/// Group `batch` by `keys` and compute `aggs` per group. Output slots are the key
/// columns (one value per group, taken from each group's first row) followed by
/// the aggregate columns. Group order is first-seen: a group's index is the
/// order its first row arrived, which is the order it is emitted.
///
/// Rows are labelled with a dense group id in a single pass ([`assign_groups`]),
/// then each aggregate is a single streaming pass over that labelling — so an
/// aggregate never materializes its group's rows, and `count(*)` is a tally, not
/// a bucketed list of row indices.
fn aggregate(
    batch: &Batch,
    store: &Store,
    keys: &[(String, Expr)],
    aggs: &[Agg],
) -> Result<Batch, String> {
    let n = batch.rows();
    let key_cols: Vec<Col> = eval_all(keys.iter().map(|(_, e)| e), store, batch)?;

    // With no keys the whole input is one group, and a scalar aggregate over
    // EMPTY input still emits that one group (SQL: `count(*)` over nothing is 0,
    // one row) — hence n_groups is forced to 1 there. A GROUPED aggregate over
    // empty input has no groups and emits zero rows, which falls out naturally.
    let (group_of, first_row, n_groups) = if keys.is_empty() {
        (vec![0u32; n], Vec::new(), 1)
    } else {
        let (g, fr) = assign_groups(&key_cols, n);
        let ng = fr.len();
        (g, fr, ng)
    };

    // Key output columns: each key's value at its group's first row.
    let mut slots: Vec<Col> = if keys.is_empty() {
        Vec::new()
    } else {
        key_cols.iter().map(|c| c.gather(&first_row)).collect()
    };

    for agg in aggs {
        let arg_col = agg
            .arg
            .as_ref()
            .map(|e| eval(e, store, batch))
            .transpose()?;
        slots.push(Col::Gen(fold_grouped(
            agg,
            arg_col.as_ref(),
            &group_of,
            n_groups,
        )));
    }

    Ok(Batch::of(slots))
}

/// Assign a dense, first-seen group id to every row from its key columns.
/// Returns `(group_of, first_row)`: `group_of[i]` is row `i`'s group,
/// `first_row[g]` the row that opened group `g`. A single native key column is
/// grouped on its raw type with no boxing; anything else falls back to a reused
/// byte key ([`value::group_key_into`]). Both honor the one grouping contract.
fn assign_groups(key_cols: &[Col], n: usize) -> (Vec<u32>, Vec<usize>) {
    if let [only] = key_cols {
        match only {
            Col::Num(v) => return group_by(n, v.iter().map(|&x| value::num_group_bits(x))),
            // Node ids are small non-negative integers: their id IS the key, and
            // it matches `value_at` (which surfaces a node as `Num(id)`, whose
            // group bits are the same integer).
            Col::Nodes(v) | Col::Edges(v) => return group_by(n, v.iter().map(|&x| u64::from(x))),
            Col::Bool(v) => return group_by(n, v.iter().map(|&b| u64::from(b))),
            Col::Str(v) => return group_by_arc(v),
            Col::Gen(_) => {} // mixed: fall through to the byte-key path
        }
    }
    // General path: self-delimiting byte key per row, reused buffer, allocate
    // only when a row opens a new group.
    let mut of: FnvMap<Vec<u8>, u32> = FnvMap::default();
    let mut group_of = Vec::with_capacity(n);
    let mut first_row = Vec::new();
    let mut buf = Vec::new();
    for i in 0..n {
        buf.clear();
        for kc in key_cols {
            value::group_key_into(&kc.value_at(i), &mut buf);
        }
        let g = match of.get(buf.as_slice()) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(buf.clone(), g);
                first_row.push(i);
                g
            }
        };
        group_of.push(g);
    }
    (group_of, first_row)
}

/// Group by a per-row key of a `Hash + Eq` type (the typed fast path).
fn group_by<K: std::hash::Hash + Eq>(
    n: usize,
    keys: impl Iterator<Item = K>,
) -> (Vec<u32>, Vec<usize>) {
    let mut of: FnvMap<K, u32> = FnvMap::default();
    let mut group_of = Vec::with_capacity(n);
    let mut first_row = Vec::new();
    for (i, k) in keys.enumerate() {
        let g = match of.get(&k) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(k, g);
                first_row.push(i);
                g
            }
        };
        group_of.push(g);
    }
    (group_of, first_row)
}

/// Group a string column keyed on the `Arc<str>` itself: a row that opens a new
/// group stores a clone of the shared pointer (a refcount bump), NOT a freshly
/// allocated `Box<str>` — so a million distinct strings cost a million refcount
/// bumps, not a million heap allocations + copies. Lookups borrow `&str`, so a
/// repeated string never touches the allocator.
fn group_by_arc(keys: &[Arc<str>]) -> (Vec<u32>, Vec<usize>) {
    // Pre-size for the worst case (all-distinct) so the map never rehashes while
    // filling — the rehash chain dominated an all-unique million-key merge.
    let mut of: FnvMap<Arc<str>, u32> =
        FnvMap::with_capacity_and_hasher(keys.len(), Default::default());
    let mut group_of = Vec::with_capacity(keys.len());
    let mut first_row = Vec::new();
    for (i, k) in keys.iter().enumerate() {
        let g = match of.get(k.as_ref()) {
            Some(&g) => g,
            None => {
                let g = first_row.len() as u32;
                of.insert(Arc::clone(k), g);
                first_row.push(i);
                g
            }
        };
        group_of.push(g);
    }
    (group_of, first_row)
}

/// Sort one cell in place for `order(local)`: a `List` by its elements, a `Map`
/// by its values (TinkerPop's default local map ordering), anything else
/// unchanged. Order is the value contract's `cmp_total`; `descending` reverses.
fn sort_local_cell(v: Value, descending: bool) -> Value {
    let dir = |ord: std::cmp::Ordering| if descending { ord.reverse() } else { ord };
    match v {
        Value::List(mut items) => {
            items.sort_by(|a, b| dir(value::cmp_total(a, b)));
            Value::List(items)
        }
        Value::Map(pairs) => {
            let mut pairs = (*pairs).clone();
            pairs.sort_by(|a, b| dir(value::cmp_total(&a.1, &b.1)));
            Value::Map(std::sync::Arc::new(pairs))
        }
        other => other,
    }
}

/// Fold one aggregate to one value per group in a single streaming pass over the
/// group labelling. Null policy and ordering come from the value contract;
/// nothing here restates them.
fn fold_grouped(agg: &Agg, arg_col: Option<&Col>, group_of: &[u32], n_groups: usize) -> Vec<Value> {
    // `count(*)` — no argument — is each group's row count: a pure tally.
    if agg.func == AggFn::Count && agg.arg.is_none() {
        let mut tally = vec![0f64; n_groups];
        for &g in group_of {
            tally[g as usize] += 1.0;
        }
        return tally.into_iter().map(Value::Num).collect();
    }
    let Some(col) = arg_col else {
        return vec![Value::Null; n_groups]; // sum/min/max/avg with no argument
    };

    match agg.func {
        AggFn::Count if agg.distinct => {
            // Per-group distinct count. A dedicated set per group, keyed by the
            // grouping bytes; a group entry is allocated only for a new value.
            let mut sets: Vec<FnvSet<Vec<u8>>> = (0..n_groups).map(|_| FnvSet::default()).collect();
            let mut buf = Vec::new();
            for (i, &g) in group_of.iter().enumerate() {
                let v = col.value_at(i);
                if v.is_null() {
                    continue;
                }
                buf.clear();
                value::group_key_into(&v, &mut buf);
                let set = &mut sets[g as usize];
                if !set.contains(buf.as_slice()) {
                    set.insert(buf.clone());
                }
            }
            sets.iter().map(|s| Value::Num(s.len() as f64)).collect()
        }
        AggFn::Count => {
            // count(arg): non-null values per group.
            let mut tally = vec![0f64; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                if !col.value_at(i).is_null() {
                    tally[g as usize] += 1.0;
                }
            }
            tally.into_iter().map(Value::Num).collect()
        }
        AggFn::Sum | AggFn::Avg => {
            // total + count of non-null NUMERIC values; a non-null non-numeric
            // poisons its group to NULL (never coerced to NaN). SUM and AVG differ
            // over nothing, matching lenke-core (the GQL/Cypher convention): SUM of
            // an empty/all-null group is 0, AVG is NULL (no values to divide).
            let mut total = vec![0f64; n_groups];
            let mut cnt = vec![0u64; n_groups];
            let mut poison = vec![false; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                match col.value_at(i) {
                    Value::Null => {}
                    Value::Num(x) => {
                        total[g as usize] += x;
                        cnt[g as usize] += 1;
                    }
                    _ => poison[g as usize] = true,
                }
            }
            (0..n_groups)
                .map(|g| {
                    if poison[g] {
                        Value::Null
                    } else if agg.func == AggFn::Sum {
                        Value::Num(total[g]) // 0.0 when cnt == 0
                    } else if cnt[g] == 0 {
                        Value::Null // AVG of nothing
                    } else {
                        Value::Num(total[g] / cnt[g] as f64)
                    }
                })
                .collect()
        }
        AggFn::Collect => {
            // Gremlin `fold()`: gather every value into each group's list, in row
            // order (a preceding sort carries through), nulls kept. An empty group
            // — which the no-key case still emits — folds to the empty list.
            let mut lists: Vec<Vec<Value>> = vec![Vec::new(); n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                lists[g as usize].push(col.value_at(i));
            }
            lists.into_iter().map(Value::List).collect()
        }
        AggFn::Min | AggFn::Max => {
            let want_min = agg.func == AggFn::Min;
            let mut best: Vec<Option<Value>> = vec![None; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                let v = col.value_at(i);
                if v.is_null() {
                    continue;
                }
                match &best[g as usize] {
                    None => best[g as usize] = Some(v),
                    Some(cur) => {
                        let ord = value::cmp_total(&v, cur);
                        if (want_min && ord.is_lt()) || (!want_min && ord.is_gt()) {
                            best[g as usize] = Some(v);
                        }
                    }
                }
            }
            best.into_iter().map(|o| o.unwrap_or(Value::Null)).collect()
        }
    }
}

/// A hop: for each input row, expand the node in slot `from` along `dir`,
/// filtered by `edge_label`; emit one output row per matching neighbour with the
/// existing slots replicated and the neighbour appended as a new slot. This is
/// the bulk (lineage-free) strategy: `keep` records which input row each output
/// row came from, `nbrs` the landed node — the existing slots are gathered by
/// `keep`, so no per-row struct is built.
fn expand(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: Option<&str>,
    bind_edge: bool,
) -> Batch {
    // An empty expand still appends the landed slot(s), so the output has the same
    // shape a successful expand would (K+1 slots, or K+2 with the edge bound) — a
    // projection referencing a new slot must not go out of bounds.
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        if bind_edge {
            slots.push(Col::Edges(vec![]));
        }
        slots.push(Col::Nodes(vec![]));
        let mut b = Batch::of(slots);
        if batch.lineage.is_some() {
            b.lineage = Some(Lineage::empty());
        }
        b
    };
    // Resolve the edge label to an interned id up front; an unknown label matches
    // nothing (not everything).
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        // Only a node frontier can be expanded; anything else yields nothing.
        return empty();
    };

    // Collect edge ids only when something needs them — a bound edge slot or
    // lineage — so the lineage-free hot path pushes nothing extra per neighbour.
    let track = batch.lineage.is_some();
    let need_eids = bind_edge || track;
    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    let mut eids = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        for_each_nbr(store, v, dir, want, |nbr, eid| {
            keep.push(row);
            nbrs.push(nbr);
            if need_eids {
                eids.push(eid);
            }
        });
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    if bind_edge {
        slots.push(Col::Edges(eids.clone())); // edge slot at index W
    }
    slots.push(Col::Nodes(nbrs.clone())); // node slot at index W (or W+1)
    let mut out = Batch::of(slots);
    // Lineage strategy: when the input carried a path, extend each output row's
    // path by the neighbour it landed on AND the edge it crossed, so both
    // `nodes(p)` and `relationships(p)` are recoverable.
    if let Some(lin) = &batch.lineage {
        out.lineage = Some(lin.extend(&keep, &nbrs, &eids));
    }
    out
}

/// Interval-overlap hop (`Plan::IntervalExpand`): like [`expand`], but keeps only
/// edges whose `[lo_key, hi_key]` interval overlaps `[qlo, qhi]` (per input row).
/// Seek-or-scan: an OUT hop over a store with a matching interval index seeks
/// (`for_each_overlap`); otherwise it scans the adjacency and applies the overlap
/// itself — the rows are identical either way, so the optimizer can fuse without
/// knowing whether the index exists. A non-numeric/absent bound or edge interval
/// yields no match for that edge (matching what the `<=`/`>=` filter would do).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors `expand` plus the two bounds and keys"
)]
fn interval_expand(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: Option<&str>,
    lo_key: &str,
    hi_key: &str,
    qlo_col: &Col,
    qhi_col: &Col,
    bind_edge: bool,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        if bind_edge {
            slots.push(Col::Edges(vec![]));
        }
        slots.push(Col::Nodes(vec![]));
        let mut b = Batch::of(slots);
        if batch.lineage.is_some() {
            b.lineage = Some(Lineage::empty());
        }
        b
    };
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };
    // Seek only an OUT hop over a matching index (the index is over out-edges);
    // any other case scans and applies the overlap.
    let can_seek = matches!(dir, Dir::Out) && store.has_interval_index(lo_key, hi_key);

    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    let mut eids = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        // The bounds for this row; a non-numeric bound can never satisfy the
        // numeric interval comparison, so the row contributes nothing.
        let (Value::Num(qlo), Value::Num(qhi)) = (qlo_col.value_at(row), qhi_col.value_at(row))
        else {
            continue;
        };
        if can_seek {
            store.for_each_overlap(v, qlo, qhi, |eid, nbr| {
                keep.push(row);
                nbrs.push(nbr);
                eids.push(eid);
            });
        } else {
            for_each_nbr(store, v, dir, want, |nbr, eid| {
                if let (Value::Num(lo), Value::Num(hi)) =
                    (store.edge_prop(eid, lo_key), store.edge_prop(eid, hi_key))
                {
                    if lo <= qhi && hi >= qlo {
                        keep.push(row);
                        nbrs.push(nbr);
                        eids.push(eid);
                    }
                }
            });
        }
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    if bind_edge {
        slots.push(Col::Edges(eids.clone()));
    }
    slots.push(Col::Nodes(nbrs.clone()));
    let mut out = Batch::of(slots);
    if let Some(lin) = &batch.lineage {
        out.lineage = Some(lin.extend(&keep, &nbrs, &eids));
    }
    out
}

/// The nodes with `label` whose property `key` equals `value` under predicate
/// `=` — the rows an `IndexSeek` produces. Uses a property index when present
/// (candidates intersected with the label), else scans the label and filters by
/// `value::equals`. A NaN/NULL literal matches nothing (as `=` does).
fn index_seek_ids(store: &Store, label: &str, key: &str, value: &Value) -> Vec<u32> {
    if value.is_null() || matches!(value, Value::Num(x) if x.is_nan()) {
        return Vec::new();
    }
    match store.index_lookup(key, value) {
        Some(cands) => {
            // group_key == equals for a finite, non-null value, so the index
            // bucket is exact; just intersect with the label.
            let in_label: std::collections::HashSet<u32> =
                store.nodes_with_label(label).iter().copied().collect();
            cands
                .into_iter()
                .filter(|id| in_label.contains(id))
                .collect()
        }
        None => store
            .nodes_with_label(label)
            .iter()
            .copied()
            // `key` may be a dotted record path — resolve it (plain keys read as
            // `prop`), so the no-index fallback matches a dotted seek too.
            .filter(|&id| value::equals(&store.prop_path(id, key), value))
            .collect(),
    }
}

/// Whether `prop <op> value` holds — the exact test the `Filter` executor applies
/// for a range comparison. Three-valued via `cmp_partial`: a NULL operand OR
/// incomparable operands (different types / NaN) are UNKNOWN → false (the row
/// drops), matching the general `compare` path. `op` must be a range op; `value`
/// is non-null.
fn range_pass(prop: &Value, op: CompareOp, value: &Value) -> bool {
    if prop.is_null() {
        return false;
    }
    let Some(ord) = value::cmp_partial(prop, value) else {
        return false; // incomparable → UNKNOWN → drop
    };
    match op {
        CompareOp::Lt => ord.is_lt(),
        CompareOp::Le => ord.is_le(),
        CompareOp::Gt => ord.is_gt(),
        CompareOp::Ge => ord.is_ge(),
        CompareOp::Eq | CompareOp::Ne => false,
    }
}

/// The nodes with `label` whose property `key` satisfies `key <op> value` — the
/// rows a `RangeSeek` produces. Uses a range index when present (candidates
/// intersected with the label), else scans and filters via `range_pass`. A NULL
/// `value` matches nothing (predicate UNKNOWN), matching a scan+filter.
fn range_seek_ids(store: &Store, label: &str, key: &str, op: CompareOp, value: &Value) -> Vec<u32> {
    if value.is_null() {
        return Vec::new();
    }
    match store.range_lookup(key, op, value) {
        Some(cands) => {
            let in_label: std::collections::HashSet<u32> =
                store.nodes_with_label(label).iter().copied().collect();
            cands
                .into_iter()
                .filter(|id| in_label.contains(id))
                .collect()
        }
        None => store
            .nodes_with_label(label)
            .iter()
            .copied()
            .filter(|&id| range_pass(&store.prop(id, key), op, value))
            .collect(),
    }
}

/// Visit each neighbour of `v` along `dir` matching edge type `want`, calling `f`
/// with `(neighbour, eid)`. The one place Expand's adjacency walk is spelled —
/// shared by the batch operator and the frontier executor so the two can never
/// disagree on what an Expand reaches.
fn for_each_nbr(store: &Store, v: u32, dir: Dir, want: Option<u32>, mut f: impl FnMut(u32, u32)) {
    // A type-filtered hop over an indexed store seeks the type bucket directly
    // (O(matching), not O(degree)) — the whole point of the opt-in edge-type index.
    if let Some(w) = want {
        if store.has_edge_type_index() {
            if matches!(dir, Dir::Out | Dir::Both) {
                for a in store.out_typed(v, w) {
                    f(a.nbr, a.eid);
                }
            }
            if matches!(dir, Dir::In | Dir::Both) {
                for a in store.in_typed(v, w) {
                    f(a.nbr, a.eid);
                }
            }
            return;
        }
    }
    let type_ok = |et: u32| want.is_none_or(|w| w == et);
    if matches!(dir, Dir::Out | Dir::Both) {
        for a in store.out(v) {
            if type_ok(a.etype) {
                f(a.nbr, a.eid);
            }
        }
    }
    if matches!(dir, Dir::In | Dir::Both) {
        for a in store.inc(v) {
            if type_ok(a.etype) {
                f(a.nbr, a.eid);
            }
        }
    }
}

/// Slot count of a pure Scan/Expand chain; `None` for anything else (Filter,
/// Join, VarLength, …). The frontier executor only handles such chains.
fn chain_width(plan: &Plan) -> Option<usize> {
    match plan {
        // A seek, like a scan, seeds a single-slot frontier.
        Plan::Scan { .. } | Plan::IndexSeek { .. } | Plan::RangeSeek { .. } => Some(1),
        // A bind_edge Expand appends TWO slots (edge then node), else one.
        Plan::Expand {
            input, bind_edge, ..
        } => Some(chain_width(input)? + if *bind_edge { 2 } else { 1 }),
        _ => None,
    }
}

/// The current node frontier of a pure Scan/Expand chain — the last slot's node
/// ids, WITH multiplicity (one entry per path reaching the node) — produced
/// without ever materializing the earlier slots. `None` if the plan is not such
/// a chain. This is the batch model's payoff: when nothing above the chain reads
/// an earlier slot, the chain need only carry its frontier.
///
/// Rejected optimization: replacing this Vec with a streaming `for_each_frontier`
/// callback (so the fused counts never build the intermediate at all). It had to
/// pass the callback as `&mut dyn FnMut` — a generic bound blows monomorphization
/// on the recursion — and the resulting per-node indirect call, nested one level
/// per hop, cost MORE than building and rescanning the vector: at 1M/8 it moved
/// 2-hop count(*) 40->64ms and count(DISTINCT) 54->62ms. The sequential Vec push
/// is cheap; per-element dynamic dispatch over tens of millions of nodes is not.
fn frontier_ids(plan: &Plan, store: &Store) -> Option<Vec<u32>> {
    match plan {
        Plan::Scan { label } => Some(match label {
            Some(l) => store.nodes_with_label(l).to_vec(),
            None => store.all_nodes(),
        }),
        Plan::IndexSeek { label, key, value } => Some(index_seek_ids(store, label, key, value)),
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => Some(range_seek_ids(store, label, key, *op, value)),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            ..
        } => {
            // Must expand the CURRENT frontier (the last slot); a linear pattern
            // always does, but a hand-built plan might not.
            if *from + 1 != chain_width(input)? {
                return None;
            }
            let src = frontier_ids(input, store)?;
            let want = match edge_label {
                None => None,
                Some(name) => match store.etype_id(name) {
                    Some(id) => Some(id),
                    None => return Some(Vec::new()), // unknown label matches nothing
                },
            };
            let mut out = Vec::new();
            for &v in &src {
                for_each_nbr(store, v, *dir, want, |nbr, _eid| out.push(nbr));
            }
            Some(out)
        }
        _ => None,
    }
}

/// Try to answer a scalar `count(*)` / `count(DISTINCT <last slot>)` sitting on
/// an Expand of a Scan/Expand chain WITHOUT materializing the wide intermediate
/// batch: the frontier feeding the final hop is produced by [`frontier_ids`],
/// then `count(*)` sums the final hop's matching degree and `count(DISTINCT c)`
/// marks endpoints in a bitset over node ids. Returns `None` (fall back to the
/// general aggregate) for any shape it does not recognize — so it is an
/// optimization, never a semantic fork.
fn try_fused_count(
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
        ..
    } = input
    else {
        return None;
    };
    let w = chain_width(inner)?; // slot count feeding the final hop
    if *from + 1 != w {
        return None; // the final Expand must expand the current frontier
    }
    let want = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return Some(scalar_num(0.0)), // unknown label: zero rows
        },
    };
    let src = frontier_ids(inner, store)?; // ids feeding the final hop, w/ multiplicity

    if agg.arg.is_none() {
        // count(*): number of final-hop paths = sum over sources of matching
        // out-degree. When the sources come from an Expand they repeat (many paths
        // reach the same node), and a node's degree is the same each time — so
        // collapse to distinct nodes with multiplicity and walk each adjacency
        // once, scaled. When they come from a Scan they are already distinct, so
        // that dedup is pure overhead: sum degrees directly.
        let mut total = 0f64;
        if matches!(inner.as_ref(), Plan::Expand { .. }) {
            let (distinct, mult) = distinct_with_mult(&src, store.node_count());
            for (i, &v) in distinct.iter().enumerate() {
                let mut deg = 0f64;
                for_each_nbr(store, v, *dir, want, |_, _| deg += 1.0);
                total += mult[i] * deg;
            }
        } else {
            for &v in &src {
                for_each_nbr(store, v, *dir, want, |_, _| total += 1.0);
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
            for_each_nbr(store, v, *dir, want, |nbr, _| {
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
fn scalar_num(x: f64) -> Batch {
    Batch::of(vec![Col::Gen(vec![Value::Num(x)])])
}

/// Collapse a node-id multiset to (distinct ids in first-seen order, their
/// multiplicities) via a direct-mapped array — node ids are dense, so no hashing.
fn distinct_with_mult(nodes: &[u32], node_count_total: usize) -> (Vec<u32>, Vec<f64>) {
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
fn refs_only_slot(expr: &Expr, s: usize) -> bool {
    match expr {
        Expr::Lit(_) => true,
        Expr::Slot(n) => *n == s,
        Expr::Prop { slot, .. } => *slot == s,
        Expr::Path | Expr::PathAccess { .. } => false,
        Expr::Not(x) => refs_only_slot(x, s),
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Arith {
            left: a, right: b, ..
        } => refs_only_slot(a, s) && refs_only_slot(b, s),
        Expr::Call { args, .. } | Expr::List { items: args } => {
            args.iter().all(|a| refs_only_slot(a, s))
        }
        Expr::Record { fields } | Expr::MapLit { entries: fields } => {
            fields.iter().all(|(_, e)| refs_only_slot(e, s))
        }
        Expr::Field { base, .. } => refs_only_slot(base, s),
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
        Expr::PropertyExists { slot, .. } => *slot == s,
        // An EXISTS correlates on outer slots below `outer_width`; conservatively
        // treat it as touching more than one, so it never rides the frontier-only
        // aggregate fast path.
        Expr::Exists { .. } => false,
    }
}

/// Rewrite every reference to slot `from` in `expr` to slot `to`. Used to retarget
/// frontier-only expressions onto a one-slot frontier batch. Callers guarantee
/// (via [`refs_only_slot`]) that no other slot appears.
fn remap_slot(expr: &Expr, from: usize, to: usize) -> Expr {
    let go = |e| Box::new(remap_slot(e, from, to));
    match expr {
        Expr::Slot(n) if *n == from => Expr::Slot(to),
        Expr::Prop { slot, key } if *slot == from => Expr::Prop {
            slot: to,
            key: key.clone(),
        },
        Expr::Slot(_) | Expr::Prop { .. } | Expr::Lit(_) | Expr::Path | Expr::PathAccess { .. } => {
            expr.clone()
        }
        Expr::Not(x) => Expr::Not(go(x)),
        Expr::And(a, b) => Expr::And(go(a), go(b)),
        Expr::Or(a, b) => Expr::Or(go(a), go(b)),
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
        Expr::List { items } => Expr::List {
            items: items.iter().map(|a| remap_slot(a, from, to)).collect(),
        },
        Expr::Record { fields } => Expr::Record {
            fields: fields
                .iter()
                .map(|(k, e)| (k.clone(), remap_slot(e, from, to)))
                .collect(),
        },
        Expr::MapLit { entries } => Expr::MapLit {
            entries: entries
                .iter()
                .map(|(k, e)| (k.clone(), remap_slot(e, from, to)))
                .collect(),
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
        // Never reached: `refs_only_slot` rejects EXISTS, so the frontier remap
        // that calls this is never handed one. Clone rather than rewrite a body.
        Expr::Exists { .. } => expr.clone(),
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
fn try_node_grouped_count(
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
    let want = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return Some(Batch::of(vec![Col::Nodes(vec![]), Col::Gen(vec![])])),
        },
    };
    let src = frontier_ids(inner, store)?; // nodes feeding the final hop, w/ multiplicity

    // Level 1: count per endpoint node id via a direct-mapped array (no hashing —
    // node ids are dense), with the final hop fused in so endpoints never
    // materialize. Distinct ids come out in first-seen order.
    let mut group_of = vec![u32::MAX; store.node_count()];
    let mut rep_ids: Vec<u32> = Vec::new();
    let mut node_count: Vec<f64> = Vec::new();
    for &v in &src {
        for_each_nbr(store, v, *dir, want, |nbr, _| {
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
fn try_frontier_aggregate(
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
        })
        .collect();
    Ok(Some(aggregate(&batch, store, &keys, &aggs)?))
}

/// A quantified hop: for each input row, enumerate every path of length in
/// `min..=max` from the node in `from`, and emit one output row per path with the
/// reached endpoint appended as a new slot. `min == 0` emits the source itself.
///
/// `trail` chooses the semantics and nothing else does: true forbids reusing an
/// edge within one path (a trail), false allows it (a walk). They diverge on a
/// cycle/self-loop — pinned by the tests — and are never conflated with a chain
/// of separate fixed `Expand`s (which is always a walk).
#[allow(clippy::too_many_arguments)]
fn var_length(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: Option<&str>,
    min: u32,
    max: u32,
    trail: bool,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };

    let mut keep = Vec::new();
    let mut ends = Vec::new();
    let mut used: Vec<u32> = Vec::new(); // edge ids on the current path (trail only)
    for (row, &v) in src.iter().enumerate() {
        varlen_dfs(
            store, v, 0, min, max, dir, want, trail, &mut used, row, &mut keep, &mut ends,
        );
        debug_assert!(used.is_empty());
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    Batch::of(slots)
}

/// Depth-first path enumeration for `var_length`. Emits `(row, endpoint)` at
/// every length in `min..=max` reached from the source, pushing straight into
/// `keep`/`ends` (a recursion-friendly alternative to a closure). For a trail,
/// `used` holds the edge ids on the current path and blocks reuse.
#[allow(clippy::too_many_arguments)]
fn varlen_dfs(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: Option<u32>,
    trail: bool,
    used: &mut Vec<u32>,
    row: usize,
    keep: &mut Vec<usize>,
    ends: &mut Vec<u32>,
) {
    if len >= min {
        keep.push(row);
        ends.push(v);
    }
    if len == max {
        return;
    }
    let mut adjs: Vec<crate::store::Adj> = Vec::new();
    if matches!(dir, Dir::Out | Dir::Both) {
        adjs.extend_from_slice(store.out(v));
    }
    if matches!(dir, Dir::In | Dir::Both) {
        adjs.extend_from_slice(store.inc(v));
    }
    for a in adjs {
        if want.is_some_and(|w| w != a.etype) {
            continue;
        }
        if trail && used.contains(&a.eid) {
            continue; // a trail may not reuse an edge
        }
        if trail {
            used.push(a.eid);
        }
        varlen_dfs(
            store,
            a.nbr,
            len + 1,
            min,
            max,
            dir,
            want,
            trail,
            used,
            row,
            keep,
            ends,
        );
        if trail {
            used.pop();
        }
    }
}

/// Shortest-path reach: a BFS from each input row's source node, emitting each
/// reachable target ONCE at its shortest distance (the first BFS reach), with the
/// target appended as a new slot. ANY-shortest — one representative per target,
/// not every shortest path. The source is not emitted; `max` caps hop distance.
#[allow(clippy::too_many_arguments)]
fn shortest_path(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: Option<&str>,
    max: Option<u32>,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };

    // When the input carries a path, reconstruct each shortest path (BFS
    // predecessors) so `Expr::Path` — hence `nodes(p)`/`path_length(p)` — sees the
    // whole node chain, not just the endpoint.
    let track = batch.lineage.is_some();
    let mut path_values: Vec<Value> = Vec::new();
    let mut path_offsets: Vec<usize> = vec![0];
    let mut path_edges: Vec<Value> = Vec::new();
    let mut path_edge_offsets: Vec<usize> = vec![0];

    let mut keep = Vec::new();
    let mut ends = Vec::new();
    for (row, &start) in src.iter().enumerate() {
        let mut visited: FnvSet<u32> = FnvSet::default();
        visited.insert(start);
        // child -> parent, and child -> edge used to reach it, for reconstructing
        // the shortest path (nodes AND relationships) back to `start`.
        let mut pred: FnvMap<u32, u32> = FnvMap::default();
        let mut pred_edge: FnvMap<u32, u32> = FnvMap::default();
        let mut q: VecDeque<(u32, u32)> = VecDeque::new();
        q.push_back((start, 0));
        while let Some((v, d)) = q.pop_front() {
            if max.is_some_and(|m| d >= m) {
                continue; // reached the hop cap; do not expand further
            }
            let mut adjs: Vec<crate::store::Adj> = Vec::new();
            if matches!(dir, Dir::Out | Dir::Both) {
                adjs.extend_from_slice(store.out(v));
            }
            if matches!(dir, Dir::In | Dir::Both) {
                adjs.extend_from_slice(store.inc(v));
            }
            for a in adjs {
                if want.is_some_and(|w| w != a.etype) {
                    continue;
                }
                // First reach = shortest reach (BFS), and only the first is kept.
                if visited.insert(a.nbr) {
                    keep.push(row);
                    ends.push(a.nbr);
                    if track {
                        pred.insert(a.nbr, v);
                        pred_edge.insert(a.nbr, a.eid);
                        // Walk parents back to `start` (its pred is never set),
                        // collecting the edge crossed at each step, then append
                        // `start..target` (and its edges) to the input row's path.
                        let mut chain = vec![a.nbr];
                        let mut edge_chain: Vec<u32> = Vec::new();
                        let mut cur = a.nbr;
                        while cur != start {
                            edge_chain.push(pred_edge[&cur]);
                            cur = pred[&cur];
                            chain.push(cur);
                        }
                        chain.reverse(); // start .. target
                        edge_chain.reverse(); // e0 .. e(k-1)
                        let lin = batch.lineage.as_ref().expect("track");
                        path_values.extend_from_slice(lin.path_at(row)); // ends at start
                        for &node in &chain[1..] {
                            path_values.push(Value::Num(f64::from(node)));
                        }
                        path_offsets.push(path_values.len());
                        path_edges.extend_from_slice(lin.edges_at(row));
                        for &edge in &edge_chain {
                            path_edges.push(Value::Num(f64::from(edge)));
                        }
                        path_edge_offsets.push(path_edges.len());
                    }
                    q.push_back((a.nbr, d + 1));
                }
            }
        }
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    let mut out = Batch::of(slots);
    if track {
        out.lineage = Some(Lineage {
            values: path_values,
            offsets: path_offsets,
            edges: path_edges,
            edge_offsets: path_edge_offsets,
        });
    }
    out
}

/// Evaluate `expr` over every row of `batch`, producing a column.
fn eval(expr: &Expr, store: &Store, batch: &Batch) -> Result<Col, String> {
    Ok(match expr {
        Expr::Slot(n) => batch.slot(*n).clone(),
        Expr::Lit(v) => broadcast(v.clone(), batch.rows()),
        Expr::Prop { slot, key } => read_property(store, batch.slot(*slot), key),
        // `<base>.key` — evaluate the base to a column, then read the field/property
        // from it (the general form of `Prop`).
        Expr::Field { base, key } => {
            let col = eval(base, store, batch)?;
            read_property(store, &col, key)
        }
        Expr::Path => match &batch.lineage {
            // Each row's path as a List of node ids; NULL when the plan tracks no
            // lineage (which `needs_lineage` prevents when Path is actually read).
            Some(lin) => Col::Gen(
                (0..batch.rows())
                    .map(|i| Value::List(lin.path_at(i).to_vec()))
                    .collect(),
            ),
            None => Col::Gen(vec![Value::Null; batch.rows()]),
        },
        Expr::PathAccess { part } => {
            use crate::ir::PathPart;
            match &batch.lineage {
                Some(lin) => Col::Gen(
                    (0..batch.rows())
                        .map(|i| {
                            let nodes = lin.path_at(i);
                            let edges = lin.edges_at(i);
                            match part {
                                PathPart::Nodes => Value::List(nodes.to_vec()),
                                PathPart::Relationships => Value::List(edges.to_vec()),
                                // Hops == number of relationships.
                                PathPart::Length => Value::Num(edges.len() as f64),
                                PathPart::Elements => {
                                    // n0, e0, n1, e1, …, nk
                                    let mut items = Vec::with_capacity(nodes.len() + edges.len());
                                    for (j, node) in nodes.iter().enumerate() {
                                        items.push(node.clone());
                                        if let Some(e) = edges.get(j) {
                                            items.push(e.clone());
                                        }
                                    }
                                    Value::List(items)
                                }
                            }
                        })
                        .collect(),
                ),
                None => Col::Gen(vec![Value::Null; batch.rows()]),
            }
        }
        Expr::Not(inner) => {
            let c = eval(inner, store, batch)?;
            map_bool(&c, |b| b.map(|x| !x))
        }
        Expr::And(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        })?,
        Expr::Or(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        })?,
        Expr::Compare { op, left, right } => {
            let l = eval(left, store, batch)?;
            let r = eval(right, store, batch)?;
            compare(*op, &l, &r)
        }
        Expr::Arith { op, left, right } => {
            // f64 math via the value contract's `as_num` (finite Num only); any
            // NULL / non-numeric / non-finite operand OR result yields NULL. When
            // either operand is a temporal, `temporal_arith` takes over (and may
            // THROW on a result out of the representable range).
            use crate::ir::ArithOp::{Add, Div, Mul, Rem, Sub};
            let l = eval(left, store, batch)?;
            let r = eval(right, store, batch)?;
            let n = l.len().min(r.len());
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let a = l.value_at(i);
                let b = r.value_at(i);
                let v = if matches!(a, Value::Temporal(_)) || matches!(b, Value::Temporal(_)) {
                    if a.is_null() || b.is_null() {
                        Value::Null
                    } else {
                        temporal_arith(*op, &a, &b)?
                    }
                } else {
                    match (value::as_num(&a), value::as_num(&b)) {
                        (Some(x), Some(y)) => {
                            // Division / modulo by zero THROWS (matches lenke-core's
                            // DataException), rather than producing IEEE Inf/NaN.
                            if matches!(op, Div | Rem) && y == 0.0 {
                                return Err("division by zero".into());
                            }
                            let res = match op {
                                Add => x + y,
                                Sub => x - y,
                                Mul => x * y,
                                Div => x / y,
                                Rem => x % y,
                            };
                            if res.is_finite() {
                                Value::Num(res)
                            } else {
                                Value::Null
                            }
                        }
                        _ => Value::Null,
                    }
                };
                out.push(v);
            }
            Col::Gen(out)
        }
        Expr::Call { name, args } => {
            // Evaluate each argument to a column, then dispatch per row. Arity is
            // validated at parse time, so `call_scalar` can index its args.
            let cols = eval_all(args, store, batch)?;
            let n = cols.iter().map(Col::len).min().unwrap_or(0);
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    let row: Vec<Value> = cols.iter().map(|c| c.value_at(i)).collect();
                    call_scalar(name, &row)
                })
                .collect();
            Col::Gen(out)
        }
        Expr::List { items } => {
            // Per row, build a Value::List of each element's value.
            let cols = eval_all(items, store, batch)?;
            let n = batch.rows();
            Col::Gen(
                (0..n)
                    .map(|i| Value::List(cols.iter().map(|c| c.value_at(i)).collect()))
                    .collect(),
            )
        }
        Expr::Record { fields } => {
            // Per row, evaluate each field then canonicalize into a Value::Record
            // (keys sorted, last-wins) via the value contract.
            let cols = eval_all(fields.iter().map(|(_, e)| e), store, batch)?;
            let n = batch.rows();
            Col::Gen(
                (0..n)
                    .map(|i| {
                        let pairs = fields
                            .iter()
                            .zip(&cols)
                            .map(|((k, _), c)| (Arc::from(k.as_str()), c.value_at(i)))
                            .collect();
                        value::make_record(pairs)
                    })
                    .collect(),
            )
        }
        Expr::MapLit { entries } => {
            // Per row, an insertion-ordered Value::Map with string keys.
            let cols = eval_all(entries.iter().map(|(_, e)| e), store, batch)?;
            let n = batch.rows();
            Col::Gen(
                (0..n)
                    .map(|i| {
                        let pairs = entries
                            .iter()
                            .zip(&cols)
                            .map(|((k, _), c)| (Value::Str(Arc::from(k.as_str())), c.value_at(i)))
                            .collect();
                        Value::Map(Arc::new(pairs))
                    })
                    .collect(),
            )
        }
        Expr::Case {
            branches,
            otherwise,
        } => {
            let conds = eval_all(branches.iter().map(|(c, _)| c), store, batch)?;
            let vals = eval_all(branches.iter().map(|(_, v)| v), store, batch)?;
            let else_col = otherwise
                .as_ref()
                .map(|e| eval(e, store, batch))
                .transpose()?;
            let n = batch.rows();
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    // First branch whose condition is literally TRUE (three-valued).
                    conds
                        .iter()
                        .position(|c| matches!(c.value_at(i), Value::Bool(true)))
                        .map(|bi| vals[bi].value_at(i))
                        .or_else(|| else_col.as_ref().map(|c| c.value_at(i)))
                        .unwrap_or(Value::Null)
                })
                .collect();
            Col::Gen(out)
        }
        Expr::Cast { target, expr } => {
            // Evaluate the input, then cast per row via the value contract. A
            // failed conversion aborts the whole evaluation (E_INVALID_VALUE) —
            // the read pipeline is fallible precisely so this can throw.
            let col = eval(expr, store, batch)?;
            let t = value::CastTarget::from(*target);
            let mut out = Vec::with_capacity(col.len());
            for i in 0..col.len() {
                out.push(value::cast(&col.value_at(i), t)?);
            }
            Col::Gen(out)
        }
        Expr::IsNull { expr, negated } => {
            // A definite Bool per row (never NULL): `IS NULL` is TRUE exactly when
            // the value is Null; `IS NOT NULL` flips it.
            let col = eval(expr, store, batch)?;
            Col::Bool(
                (0..col.len())
                    .map(|i| col.value_at(i).is_null() != *negated)
                    .collect(),
            )
        }
        Expr::PropertyExists { slot, key } => {
            // Presence, not value: true iff the element carries a stored value for
            // `key`. Only nodes carry the presence bitmap here (edge-property
            // existence is deferred); a non-node slot has no property → FALSE.
            let out: Vec<bool> = match batch.slot(*slot) {
                Col::Nodes(ids) => ids.iter().map(|&id| store.has_prop(id, key)).collect(),
                other => vec![false; other.len()],
            };
            Col::Bool(out)
        }
        Expr::Exists { body, .. } => {
            // Correlated existence: run the sub-pattern over ALL outer rows at once,
            // tagging each with a unique provenance id so surviving sub-rows point
            // back to the outer row they came from. An outer row is TRUE iff at
            // least one sub-row carries its id.
            let n = batch.rows();
            let prov = batch.slots.len(); // provenance rides at the first free slot
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            // The body reads no path (EXISTS discards lineage), so seed without one.
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let mut hit = vec![false; n];
            if let Col::Num(ids) = survivors.slot(prov) {
                for &id in ids {
                    let i = id as usize;
                    if i < n {
                        hit[i] = true;
                    }
                }
            }
            Col::Bool(hit)
        }
    })
}

/// Evaluate an `EXISTS` body against a correlated `seed` batch (the outer rows
/// plus a provenance column). The body is a chain of the operators an EXISTS
/// pattern can contain — `Expand`/`VarLength`/`Filter` — rooted at `Plan::Row`,
/// which yields `seed`. Every operator gathers the whole input row, so the
/// provenance column rides through untouched; the caller reads it off the result.
fn pull_body(plan: &Plan, store: &Store, seed: &Batch) -> Result<Batch, String> {
    Ok(match plan {
        Plan::Row => seed.clone(),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge,
        } => expand(
            &pull_body(input, store, seed)?,
            store,
            *from,
            *dir,
            edge_label.as_deref(),
            *bind_edge,
        ),
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            trail,
        } => var_length(
            &pull_body(input, store, seed)?,
            store,
            *from,
            *dir,
            edge_label.as_deref(),
            *min,
            *max,
            *trail,
        ),
        Plan::Filter { input, pred } => {
            let b = pull_body(input, store, seed)?;
            let mask = eval(pred, store, &b)?;
            let keep: Vec<usize> = match &mask {
                Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                other => (0..other.len())
                    .filter(|&i| other.value_at(i).is_true())
                    .collect(),
            };
            b.gather(&keep)
        }
        other => {
            return Err(format!("unsupported operator in EXISTS body: {other:?}"));
        }
    })
}

/// Evaluate several expressions to columns, short-circuiting on the first error.
fn eval_all<'a>(
    exprs: impl IntoIterator<Item = &'a Expr>,
    store: &Store,
    batch: &Batch,
) -> Result<Vec<Col>, String> {
    exprs.into_iter().map(|e| eval(e, store, batch)).collect()
}

/// Dispatch a scalar function over its already-evaluated argument row. Arity is
/// enforced by the parser, so indexing `args` here is safe. NULL / wrong-type
/// arguments yield NULL (no coercion, no throw).
fn call_scalar(name: &str, args: &[Value]) -> Value {
    match name {
        // variadic
        "coalesce" => args
            .iter()
            .find(|v| !v.is_null())
            .cloned()
            .unwrap_or(Value::Null),
        // numeric (1 arg)
        "abs" | "sign" | "floor" | "ceil" | "round" | "sqrt" => scalar_num_fn(name, &args[0]),
        // string (1 arg → string/number)
        "upper" => str_map(&args[0], str::to_uppercase),
        "lower" => str_map(&args[0], str::to_lowercase),
        "trim" => str_map(&args[0], |s| s.trim().to_string()),
        "length" => match &args[0] {
            Value::Str(s) => Value::Num(s.chars().count() as f64),
            _ => Value::Null,
        },
        // string predicates (2 args → bool)
        "starts_with" => str_bool(&args[0], &args[1], |s, sub| s.starts_with(sub)),
        "ends_with" => str_bool(&args[0], &args[1], |s, sub| s.ends_with(sub)),
        "contains" => str_bool(&args[0], &args[1], |s, sub| s.contains(sub)),
        // replace(s, from, to) (3 args → string)
        "replace" => match (&args[0], &args[1], &args[2]) {
            (Value::Str(s), Value::Str(f), Value::Str(t)) => {
                Value::Str(s.replace(f.as_ref(), t).into())
            }
            _ => Value::Null,
        },
        // substring(s, start[, len]) — 0-based, char-indexed
        "substring" => substring(args),
        // list (1 arg)
        "size" => match &args[0] {
            Value::List(v) => Value::Num(v.len() as f64),
            _ => Value::Null,
        },
        "head" => match &args[0] {
            Value::List(v) => v.first().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        "last" => match &args[0] {
            Value::List(v) => v.last().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // Temporal component accessors (1 arg → number, or NULL when the component
        // is undefined for that kind, e.g. year() of a time).
        "year" | "month" | "day" | "hour" | "minute" | "second" => match &args[0] {
            Value::Temporal(t) => date_part(name, *t).map_or(Value::Null, |n| Value::Num(n as f64)),
            _ => Value::Null,
        },
        // Temporal constructors: parse a string, or coerce between kinds.
        "date" => temporal_ctor(&args[0], "date"),
        "local_time" => temporal_ctor(&args[0], "localtime"),
        "datetime" | "local_datetime" => temporal_ctor(&args[0], "datetime"),
        "zoned_time" => temporal_ctor(&args[0], "zoned_time"),
        "zoned_datetime" => temporal_ctor(&args[0], "zoned_datetime"),
        "duration" => temporal_ctor(&args[0], "duration"),
        // The exact span from a to b (b - a), in fixed units; cross-kind → NULL.
        "duration_between" => match (&args[0], &args[1]) {
            (Value::Temporal(x), Value::Temporal(y)) => duration_between(*x, *y),
            _ => Value::Null,
        },
        // Path accessors (nodes/relationships/path_length/elements) are not scalar
        // Call functions — they read the lineage sidecar via `Expr::PathAccess`.
        _ => Value::Null, // parser rejects unknown names; defensive
    }
}

/// Extract a calendar/clock component from a temporal value. `None` when the
/// component is undefined for that kind (`year`/`month`/`day` of a time-only
/// value, or `hour`/`minute`/`second` of a date). Zoned values decompose in their
/// stored offset (the local wall clock), as they render; euclidean division so
/// pre-epoch instants floor correctly. Ported from lenke-core for agreement.
fn date_part(func: &str, t: crate::temporal::Temporal) -> Option<i64> {
    use crate::temporal::{civil_from_days, Temporal};
    const SPD: i64 = 86_400;
    match func {
        "year" | "month" | "day" => {
            let days = match t {
                Temporal::Date(x) => i64::from(x.days),
                Temporal::DateTime(x) => x.secs.div_euclid(SPD),
                Temporal::ZonedDateTime(x) => (x.secs + i64::from(x.offset) * 60).div_euclid(SPD),
                _ => return None,
            };
            let (y, m, d) = civil_from_days(days);
            Some(match func {
                "year" => y,
                "month" => i64::from(m),
                _ => i64::from(d),
            })
        }
        "hour" | "minute" | "second" => {
            let tod = match t {
                Temporal::Time(x) => i64::from(x.secs),
                Temporal::DateTime(x) => x.secs.rem_euclid(SPD),
                Temporal::ZonedTime(x) => {
                    (i64::from(x.secs) + i64::from(x.offset) * 60).rem_euclid(SPD)
                }
                Temporal::ZonedDateTime(x) => (x.secs + i64::from(x.offset) * 60).rem_euclid(SPD),
                _ => return None,
            };
            Some(match func {
                "hour" => tod / 3600,
                "minute" => (tod / 60) % 60,
                _ => tod % 60,
            })
        }
        _ => None,
    }
}

/// Temporal constructor: build a temporal of `kind` from a string (parsed) or
/// coerce another temporal into it (`date(datetime)` → the date part,
/// `datetime(date)` → midnight, `local_time(datetime)` → the time-of-day). A
/// bare `YYYY-MM-DD` string to a datetime target coerces to midnight. Anything
/// with no sensible conversion → NULL. Ported from lenke-core for agreement.
fn temporal_ctor(v: &Value, kind: &str) -> Value {
    use crate::temporal::{Date, DateTime, Temporal, Time};
    const SPD: i64 = 86_400;
    match v {
        // A date-only string to a datetime target → midnight.
        Value::Str(s) if kind == "datetime" && !s.contains(['T', ' ']) => Date::parse(s)
            .map(|d| {
                Value::Temporal(Temporal::DateTime(DateTime {
                    secs: i64::from(d.days) * SPD,
                    nanos: 0,
                }))
            })
            .unwrap_or(Value::Null),
        Value::Str(s) => Temporal::parse(kind, s)
            .map(Value::Temporal)
            .unwrap_or(Value::Null),
        Value::Temporal(t) => match (kind, t) {
            ("date", Temporal::Date(_))
            | ("localtime", Temporal::Time(_))
            | ("datetime", Temporal::DateTime(_))
            | ("duration", Temporal::Duration(_)) => Value::Temporal(*t),
            ("date", Temporal::DateTime(dt)) => Value::Temporal(Temporal::Date(Date {
                days: dt.secs.div_euclid(SPD) as i32,
            })),
            ("localtime", Temporal::DateTime(dt)) => Value::Temporal(Temporal::Time(Time {
                secs: u32::try_from(dt.secs.rem_euclid(SPD)).expect("0..86_400"),
                nanos: dt.nanos,
            })),
            ("datetime", Temporal::Date(d)) => Value::Temporal(Temporal::DateTime(DateTime {
                secs: i64::from(d.days) * SPD,
                nanos: 0,
            })),
            _ => Value::Null, // e.g. duration(date) — no sensible conversion
        },
        _ => Value::Null,
    }
}

/// The EXACT span from `a` to `b` (b − a), in fixed units only: whole days for
/// two dates, seconds+nanos for two datetimes. Any cross-kind pair (or a
/// duration operand) → NULL. Ported from lenke-core.
fn duration_between(a: crate::temporal::Temporal, b: crate::temporal::Temporal) -> Value {
    use crate::temporal::{Duration, Temporal};
    match (a, b) {
        (Temporal::Date(x), Temporal::Date(y)) => Value::Temporal(Temporal::Duration(Duration {
            months: 0,
            days: i64::from(y.days) - i64::from(x.days),
            secs: 0,
            nanos: 0,
        })),
        (Temporal::DateTime(x), Temporal::DateTime(y)) => {
            let mut secs = y.secs - x.secs;
            let mut nanos = i64::from(y.nanos) - i64::from(x.nanos);
            if nanos < 0 {
                nanos += 1_000_000_000;
                secs -= 1;
            }
            Value::Temporal(Temporal::Duration(Duration {
                months: 0,
                days: 0,
                secs,
                nanos: u32::try_from(nanos).expect("0..1e9 after carry"),
            }))
        }
        _ => Value::Null,
    }
}

/// Temporal `+`/`-`/`*` when either operand is temporal: instant ± duration
/// (anchored — months clamped, then days, then time), instant − instant (the
/// exact span), duration ± duration (component-wise), duration × integer. An
/// undefined combination is `Ok(Null)`; a result outside the representable
/// range is a THROWN fault (`Err`) — not a silent null. Ported from lenke-core.
fn temporal_arith(op: crate::ir::ArithOp, lv: &Value, rv: &Value) -> Result<Value, String> {
    use crate::ir::ArithOp::{Add, Mul, Sub};
    use crate::temporal::Temporal as T;
    use Value::Temporal as VT;
    let dur = |r: Option<crate::temporal::Duration>| {
        r.map(|d| VT(T::Duration(d)))
            .ok_or_else(|| "E_INVALID_VALUE: duration component out of range".to_string())
    };
    let inst = |r: Option<T>| {
        r.map(VT)
            .ok_or_else(|| "E_INVALID_VALUE: temporal result out of range".to_string())
    };
    match (op, lv, rv) {
        // duration ± duration (component-wise).
        (Add, VT(T::Duration(a)), VT(T::Duration(b))) => dur(a.add(b)),
        (Sub, VT(T::Duration(a)), VT(T::Duration(b))) => dur(a.add(&b.negate())),
        // instant ± duration (either order for +; dur±dur already handled above).
        (Add, VT(t), VT(T::Duration(d))) | (Add, VT(T::Duration(d)), VT(t)) => {
            inst(t.add_duration(d))
        }
        (Sub, VT(t), VT(T::Duration(d))) => inst(t.add_duration(&d.negate())),
        // instant − instant → the exact span from b to a (a − b).
        (Sub, VT(a), VT(b)) => Ok(duration_between(*b, *a)),
        // duration × INTEGER (either order); a non-integer factor is NULL.
        (Mul, VT(T::Duration(d)), Value::Num(n)) | (Mul, Value::Num(n), VT(T::Duration(d))) => {
            if n.is_finite() && n.fract() == 0.0 {
                dur(d.scale(*n as i64))
            } else {
                Ok(Value::Null)
            }
        }
        _ => Ok(Value::Null),
    }
}

/// Map a string value through `f`; NULL/non-string yields NULL.
fn str_map(v: &Value, f: impl Fn(&str) -> String) -> Value {
    match v {
        Value::Str(s) => Value::Str(f(s).into()),
        _ => Value::Null,
    }
}

/// A two-string predicate; NULL/non-string operand yields NULL.
fn str_bool(a: &Value, b: &Value, f: impl Fn(&str, &str) -> bool) -> Value {
    match (a, b) {
        (Value::Str(s), Value::Str(sub)) => Value::Bool(f(s, sub)),
        _ => Value::Null,
    }
}

/// `substring(s, start[, len])` — 0-based, Unicode-char indexed. `start` clamps to
/// `[0, len]`; a negative `start`/`len`, or a NULL/non-string/non-numeric arg,
/// yields NULL. Omitted `len` runs to the end.
fn substring(args: &[Value]) -> Value {
    let (Value::Str(s), Some(start)) = (&args[0], value::as_num(&args[1])) else {
        return Value::Null;
    };
    if start < 0.0 || start.fract() != 0.0 {
        return Value::Null;
    }
    let chars: Vec<char> = s.chars().collect();
    let begin = (start as usize).min(chars.len());
    let end = match args.get(2) {
        None => chars.len(),
        Some(lv) => match value::as_num(lv) {
            Some(l) if l >= 0.0 && l.fract() == 0.0 => (begin + l as usize).min(chars.len()),
            _ => return Value::Null,
        },
    };
    Value::Str(chars[begin..end].iter().collect::<String>().into())
}

/// Apply a unary numeric scalar function. Finite-or-null: a NULL / non-numeric /
/// non-finite argument OR result (e.g. `sqrt(-1)`) yields NULL. `sign(0)` is 0
/// (unlike `f64::signum`); rounding is f64's round-half-away-from-zero.
fn scalar_num_fn(name: &str, v: &Value) -> Value {
    let Some(x) = value::as_num(v) else {
        return Value::Null;
    };
    let r = match name {
        "abs" => x.abs(),
        "sign" => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "floor" => x.floor(),
        "ceil" => x.ceil(),
        "round" => x.round(),
        "sqrt" => x.sqrt(),
        _ => return Value::Null, // parser rejects unknown names; defensive
    };
    if r.is_finite() {
        Value::Num(r)
    } else {
        Value::Null
    }
}

/// `op` with its operands swapped — used to normalize `literal <cmp> prop` to
/// `prop <cmp> literal`. Equality is symmetric; the orderings mirror.
fn flip_op(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::Ne => CompareOp::Ne,
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Ge => CompareOp::Le,
    }
}

/// One-pass predicate for the common `<prop> <cmp> <literal>` (either operand
/// order) over a node frontier: read the storage property per row and emit the
/// kept row indices, without building a full value column AND a full boolean mask
/// as intermediates. Every comparison goes through the value contract, so results
/// match the general path exactly: an absent property is NULL → UNKNOWN → dropped,
/// a NULL literal makes every comparison UNKNOWN → all dropped, and cross-type is
/// the contract's `equals`/`cmp_total`. `None` if the predicate is not this shape.
fn try_filter_keep(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    let Expr::Compare { op, left, right } = pred else {
        return None;
    };
    let (slot, key, op, lit) = match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot, key }, Expr::Lit(v)) => (*slot, key, *op, v),
        (Expr::Lit(v), Expr::Prop { slot, key }) => (*slot, key, flip_op(*op), v),
        _ => return None,
    };
    let Col::Nodes(ids) = batch.slot(slot) else {
        return None;
    };
    // A NULL literal makes every comparison UNKNOWN — no row is kept.
    if lit.is_null() {
        return Some(Vec::new());
    }
    let column = store.column(key);
    let mut keep = Vec::new();
    for (row, &id) in ids.iter().enumerate() {
        let v = match column {
            Some(c) => c.read(id as usize),
            None => continue, // property absent everywhere → UNKNOWN → dropped
        };
        if v.is_null() {
            continue;
        }
        let hit = match op {
            CompareOp::Eq => value::equals(&v, lit),
            CompareOp::Ne => !value::equals(&v, lit),
            CompareOp::Lt => value::cmp_total(&v, lit).is_lt(),
            CompareOp::Le => value::cmp_total(&v, lit).is_le(),
            CompareOp::Gt => value::cmp_total(&v, lit).is_gt(),
            CompareOp::Ge => value::cmp_total(&v, lit).is_ge(),
        };
        if hit {
            keep.push(row);
        }
    }
    Some(keep)
}

/// Read `key` off an element frontier as a column, bulk-gathering the typed
/// storage column and staying unboxed when it and every read entry are
/// present-and-typed; fall to `Gen` (with nulls) otherwise.
fn read_property(store: &Store, col: &Col, key: &str) -> Col {
    // An edge slot reads an EDGE property (boxed map, keyed by eid).
    if let Col::Edges(eids) = col {
        return Col::Gen(eids.iter().map(|&e| store.edge_prop(e, key)).collect());
    }
    let Col::Nodes(ids) = col else {
        // A non-element column (e.g. a projected Record): `x.key` reads the record
        // field; anything else has no property and reads NULL.
        return Col::Gen(
            (0..col.len())
                .map(|i| match col.value_at(i) {
                    Value::Record(fields) => value::record_field(&fields, key),
                    // A Map `.key` reads the entry under the string key `key`.
                    Value::Map(pairs) => pairs
                        .iter()
                        .find(|(k, _)| matches!(k, Value::Str(s) if s.as_ref() == key))
                        .map_or(Value::Null, |(_, v)| v.clone()),
                    _ => Value::Null,
                })
                .collect(),
        );
    };
    let Some(column) = store.column(key) else {
        return Col::Gen(vec![Value::Null; ids.len()]);
    };
    match column {
        Column::Num { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Num(ids.iter().map(|&i| data[i as usize]).collect())
        }
        Column::Str { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Str(ids.iter().map(|&i| data[i as usize].clone()).collect())
        }
        Column::Bool { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Bool(ids.iter().map(|&i| data[i as usize]).collect())
        }
        _ => Col::Gen(ids.iter().map(|&i| store.prop(i, key)).collect()),
    }
}

fn broadcast(v: Value, n: usize) -> Col {
    match v {
        Value::Num(x) => Col::Num(vec![x; n]),
        Value::Bool(b) => Col::Bool(vec![b; n]),
        Value::Str(s) => Col::Str(vec![s; n]),
        // Null and List have no unboxed column form.
        other => Col::Gen(vec![other; n]),
    }
}

/// Compare two columns elementwise into a `Bool` column. `=`/`<>` use the value
/// contract's `equals`; ordering uses `cmp_total`. A NULL operand yields UNKNOWN,
/// carried as a `Gen` cell of `Null` so the three-valued logic upstream sees it.
fn compare(op: CompareOp, l: &Col, r: &Col) -> Col {
    let n = l.len().min(r.len());
    let mut out = Vec::with_capacity(n);
    let mut any_unknown = false;
    for i in 0..n {
        let a = l.value_at(i);
        let b = r.value_at(i);
        if a.is_null() || b.is_null() {
            any_unknown = true;
            out.push(None);
            continue;
        }
        // Equality uses the value contract's `equals` (cross-type = false, not
        // unknown). Ordering uses `cmp_partial` (3VL): incomparable operands —
        // different types or a NaN — make the comparison UNKNOWN (→ NULL), NOT a
        // Bool from the total order. (The total order is only for sort/min/max.)
        let res = match op {
            CompareOp::Eq => Some(value::equals(&a, &b)),
            CompareOp::Ne => Some(!value::equals(&a, &b)),
            CompareOp::Lt => value::cmp_partial(&a, &b).map(std::cmp::Ordering::is_lt),
            CompareOp::Le => value::cmp_partial(&a, &b).map(std::cmp::Ordering::is_le),
            CompareOp::Gt => value::cmp_partial(&a, &b).map(std::cmp::Ordering::is_gt),
            CompareOp::Ge => value::cmp_partial(&a, &b).map(std::cmp::Ordering::is_ge),
        };
        if res.is_none() {
            any_unknown = true;
        }
        out.push(res);
    }
    if any_unknown {
        Col::Gen(
            out.into_iter()
                .map(|o| o.map_or(Value::Null, Value::Bool))
                .collect(),
        )
    } else {
        Col::Bool(out.into_iter().map(|o| o.expect("no unknowns")).collect())
    }
}

/// Read a column as three-valued booleans (None = UNKNOWN).
fn as_truth(col: &Col) -> Vec<Option<bool>> {
    match col {
        Col::Bool(bs) => bs.iter().map(|&b| Some(b)).collect(),
        other => (0..other.len())
            .map(|i| match other.value_at(i) {
                Value::Bool(b) => Some(b),
                _ => None,
            })
            .collect(),
    }
}

fn map_bool(col: &Col, f: impl Fn(Option<bool>) -> Option<bool>) -> Col {
    truth_to_col(as_truth(col).into_iter().map(f).collect())
}

fn zip_bool(
    store: &Store,
    batch: &Batch,
    l: &Expr,
    r: &Expr,
    f: impl Fn(Option<bool>, Option<bool>) -> Option<bool>,
) -> Result<Col, String> {
    let lc = as_truth(&eval(l, store, batch)?);
    let rc = as_truth(&eval(r, store, batch)?);
    let n = lc.len().min(rc.len());
    Ok(truth_to_col((0..n).map(|i| f(lc[i], rc[i])).collect()))
}

fn truth_to_col(out: Vec<Option<bool>>) -> Col {
    if out.iter().all(Option::is_some) {
        Col::Bool(out.into_iter().map(|o| o.expect("all some")).collect())
    } else {
        Col::Gen(
            out.into_iter()
                .map(|o| o.map_or(Value::Null, Value::Bool))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Plan;
    use crate::store::Builder;
    use std::sync::Arc;

    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }
    fn prop(slot: usize, key: &str) -> Expr {
        Expr::Prop {
            slot,
            key: key.to_string(),
        }
    }
    fn lit(v: Value) -> Expr {
        Expr::Lit(v)
    }
    fn cmp(op: CompareOp, l: Expr, r: Expr) -> Expr {
        Expr::Compare {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }
    fn scan(label: &str) -> Plan {
        Plan::Scan {
            label: Some(label.to_string()),
        }
    }
    fn names_of(out: &Rows, col: usize) -> Vec<String> {
        out.rows
            .iter()
            .map(|r| match &r[col] {
                Value::Str(x) => x.to_string(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    fn social() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
        let bob = b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
        let c = b.node(&["Person"], &[("name", s("carol")), ("age", n(40.0))]);
        let proj = b.node(&["Project"], &[("name", s("graphdb"))]);
        b.edge(a, bob, "KNOWS");
        b.edge(a, c, "KNOWS");
        b.edge(bob, c, "KNOWS");
        b.edge(a, proj, "WORKS_ON");
        b.build()
    }

    /// The opt-in edge-type index is a pure optimization: a type-filtered hop
    /// returns the SAME rows with it on as with it off (for_each_nbr routes to the
    /// bucket, but the answer is identical).
    #[test]
    fn edge_type_index_gives_identical_query_results() {
        let mut store = social();
        let plan = crate::gql::parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS b").unwrap();
        let mut before = names_of(&run(&plan, &store), 0);
        before.sort();
        store.create_edge_type_index();
        let mut after = names_of(&run(&plan, &store), 0);
        after.sort();
        assert_eq!(before, after);
        // alice KNOWS bob & carol; bob KNOWS carol → bob, carol, carol.
        assert_eq!(after, vec!["bob", "carol", "carol"]);
        // A count through the fused fast path also matches.
        let cplan =
            crate::gql::parse("MATCH (a:Person)-[:KNOWS]->() RETURN count(*) AS c").unwrap();
        assert!(matches!(run(&cplan, &store).rows[0][0], Value::Num(x) if x == 3.0));
    }

    /// Does the plan tree contain an `IntervalExpand` (the fused interval hop)?
    fn has_interval_expand(p: &Plan) -> bool {
        match p {
            Plan::IntervalExpand { .. } => true,
            Plan::Expand { input, .. }
            | Plan::VarLength { input, .. }
            | Plan::ShortestPath { input, .. }
            | Plan::Filter { input, .. }
            | Plan::Aggregate { input, .. }
            | Plan::OrderPage { input, .. }
            | Plan::Project { input, .. }
            | Plan::Distinct { input }
            | Plan::SortLocal { input, .. }
            | Plan::Update { input, .. } => has_interval_expand(input),
            Plan::Join { left, right, .. } => {
                has_interval_expand(left) || has_interval_expand(right)
            }
            _ => false,
        }
    }

    fn interval_store() -> Store {
        // Emp 0 with 5 HELD edges to role 1, intervals [d, d+2] for d in 0..5.
        let mut b = Builder::default();
        b.node(&["Emp"], &[]);
        b.node(&["Role"], &[]);
        let mut st = b.build();
        for d in 0..5u32 {
            let e = st.add_edge(0, 1, "HELD");
            st.set_edge_prop(e, "vf", n(f64::from(d)));
            st.set_edge_prop(e, "vt", n(f64::from(d) + 2.0));
        }
        st
    }

    /// The optimizer fuses `r.vf <= X AND r.vt >= Y` over a bound-edge hop into an
    /// `IntervalExpand`, which returns the SAME rows via the scan fallback (no
    /// index) and via the index seek — and both equal the hand-computed answer.
    #[test]
    fn interval_expand_fuses_and_matches_scan_and_seek() {
        use crate::opt::optimize;
        let mut st = interval_store();
        // As of t=3: [0,2] no, [1,3] yes, [2,4] yes, [3,5] yes, [4,6] no → 3.
        let q = "MATCH (p:Emp)-[r:HELD]->(x) WHERE r.vf <= 3 AND r.vt >= 3 RETURN count(*) AS c";
        let plan = optimize(crate::gql::parse(q).unwrap());
        assert!(
            has_interval_expand(&plan),
            "optimizer did not fuse: {plan:?}"
        );
        // scan fallback (no interval index yet)
        assert!(matches!(run(&plan, &st).rows[0][0], Value::Num(x) if x == 3.0));
        // index seek (same plan, index present)
        st.create_interval_index("vf", "vt");
        assert!(matches!(run(&plan, &st).rows[0][0], Value::Num(x) if x == 3.0));

        // Row-level equivalence: the matching intervals' vf are {1,2,3}, seek == scan.
        let rq = "MATCH (p:Emp)-[r:HELD]->(x) WHERE r.vf <= 3 AND r.vt >= 3 RETURN r.vf AS f";
        let rplan = optimize(crate::gql::parse(rq).unwrap());
        let mut seek: Vec<String> = names_of(&run(&rplan, &st), 0);
        seek.sort();
        let scan_only = interval_store(); // fresh, no index
        let mut scan: Vec<String> = names_of(&run(&rplan, &scan_only), 0);
        scan.sort();
        assert_eq!(seek, scan);
        // vf of the matching intervals ([1,3],[2,4],[3,5]) — `names_of` renders a
        // Num via its debug form.
        assert_eq!(seek, vec!["Num(1.0)", "Num(2.0)", "Num(3.0)"]);
    }

    /// Grouping by an EDGE property counts per distinct edge-prop value — the
    /// bound edge sits at slot W and the endpoint node at W+1, so the count
    /// fast-path must not read the edge key as an (absent) node property. (The
    /// differential fuzzer found this bucketing every row under one NULL group.)
    #[test]
    fn group_by_edge_property_counts_per_value() {
        let mut b = Builder::default();
        let x = b.node(&["N"], &[]);
        let y = b.node(&["N"], &[]);
        let z = b.node(&["N"], &[]);
        b.edge(x, y, "R");
        b.edge(x, z, "R");
        b.edge(y, z, "R");
        let mut store = b.build();
        // Set weights: two edges w=2, one w=7 (eids 0,1,2 in insertion order).
        store.set_edge_prop(0, "w", n(2.0));
        store.set_edge_prop(1, "w", n(2.0));
        store.set_edge_prop(2, "w", n(7.0));
        let plan =
            crate::gql::parse("MATCH (a:N)-[r:R]->(b) RETURN r.w AS w, count(*) AS c").unwrap();
        // Group {2.0 → 2 edges, 7.0 → 1 edge}, order-independent.
        let mut got: Vec<(String, f64)> = run(&plan, &store)
            .rows
            .iter()
            .map(|row| (format!("{:?}", row[0]), num(&row[1])))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            got,
            vec![("Num(2.0)".to_string(), 2.0), ("Num(7.0)".to_string(), 1.0)]
        );
    }

    // --- relational core (unchanged behavior, now slot-addressed) ---

    #[test]
    fn scan_label_and_project() {
        let store = social();
        let out = run(
            &scan("Person").project(vec![("name".into(), prop(0, "name"))]),
            &store,
        );
        assert_eq!(out.rows.len(), 3);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob", "carol"]);
    }

    #[test]
    fn cast_projects_per_row() {
        let store = social();
        // Cast each Person's numeric age to INTEGER (identity here; the ages are
        // already whole) — verifies the per-row Cast arm wires through Project.
        let plan = scan("Person").project(vec![(
            "a".into(),
            Expr::Cast {
                target: crate::ir::CastTarget::Integer,
                expr: Box::new(prop(0, "age")),
            },
        )]);
        let out = run(&plan, &store);
        let mut got: Vec<f64> = out
            .rows
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                ref o => panic!("expected Num, got {o:?}"),
            })
            .collect();
        got.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        assert_eq!(got, vec![25.0, 30.0, 40.0]);
    }

    #[test]
    fn cast_fault_surfaces_through_try_run() {
        let store = social();
        // "alice" has no numeric form → the CAST throws E_INVALID_VALUE, and the
        // fallible `try_run` returns that Err (this is why the read pipeline
        // threads Result at all; `run` would panic on the same plan).
        let plan = scan("Person").project(vec![(
            "n".into(),
            Expr::Cast {
                target: crate::ir::CastTarget::Integer,
                expr: Box::new(prop(0, "name")),
            },
        )]);
        let err = try_run(&plan, &store).unwrap_err();
        assert!(err.contains("E_INVALID_VALUE"), "got: {err}");
    }

    #[test]
    fn is_null_projects_definite_bools() {
        // A scan of all nodes: the three Persons carry `age`, the Project node
        // does not. `age IS NULL` must be a definite Bool for EVERY row (never a
        // Null/UNKNOWN), TRUE only where the value is absent.
        let store = social();
        let plan = Plan::Scan { label: None }.project(vec![(
            "n".into(),
            Expr::IsNull {
                expr: Box::new(prop(0, "age")),
                negated: false,
            },
        )]);
        let out = run(&plan, &store);
        // Every value is a concrete boolean — none is Null.
        assert!(out.rows.iter().all(|r| matches!(r[0], Value::Bool(_))));
        let trues = out
            .rows
            .iter()
            .filter(|r| matches!(r[0], Value::Bool(true)))
            .count();
        assert_eq!(trues, 1); // only the Project node lacks `age`
    }

    #[test]
    fn property_exists_separates_present_null_from_absent() {
        // node 0: age present-null, node 1: age absent. PROPERTY_EXISTS is a
        // presence test, so it is TRUE for the present-null and FALSE for absent —
        // the distinction `IS NOT NULL` (both FALSE) cannot draw.
        let mut b = Builder::default();
        b.node(&["P"], &[("name", s("null"))]);
        b.node(&["P"], &[("name", s("absent"))]);
        let mut store = b.build();
        store.set_prop(0, "age", Value::Null);

        let exists = Plan::Scan {
            label: Some("P".into()),
        }
        .project(vec![(
            "e".into(),
            Expr::PropertyExists {
                slot: 0,
                key: "age".into(),
            },
        )]);
        let out = run(&exists, &store);
        assert!(matches!(out.rows[0][0], Value::Bool(true))); // present-null
        assert!(matches!(out.rows[1][0], Value::Bool(false))); // absent
    }

    #[test]
    fn filter_numeric_then_project() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Gt, prop(0, "age"), lit(n(28.0))))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    #[test]
    fn absent_property_is_null_and_filters_as_unknown() {
        let store = social();
        // Project has no age → `age >= 0` is UNKNOWN for it → dropped.
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Ge, prop(0, "age"), lit(n(0.0))))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 3);
    }

    #[test]
    fn equality_is_cross_type_false() {
        let store = social();
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Eq, prop(0, "age"), lit(s("30"))))
            .project(vec![("name".into(), prop(0, "name"))]);
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    /// A hand-built `Insert` plan writes nodes and edges through `execute`.
    #[test]
    fn execute_insert_writes_store() {
        use crate::ir::{InsertEdge, InsertNode};
        let mut store = Builder::default().build();
        let plan = Plan::Insert {
            nodes: vec![
                InsertNode {
                    labels: vec!["P".into()],
                    props: vec![("name".into(), s("a"))],
                },
                InsertNode {
                    labels: vec!["P".into()],
                    props: vec![],
                },
            ],
            edges: vec![InsertEdge {
                from: 0,
                to: 1,
                etype: "R".into(),
                props: vec![],
            }],
        };
        let out = execute(&plan, &mut store).unwrap();
        assert_eq!(out.rows.len(), 0); // a write returns no rows
        assert_eq!(store.node_count(), 2);
        assert_eq!(store.nodes_with_label("P"), &[0, 1]);
        assert_eq!(store.out(0).len(), 1);
        assert_eq!(store.out(0)[0].nbr, 1);
        assert!(matches!(store.prop(0, "name"), Value::Str(x) if &*x == "a"));
    }

    /// A hand-built `Update` plan sets and removes properties on matched nodes.
    /// SET carol.age = 41; REMOVE alice.age — over a Person scan.
    #[test]
    fn execute_update_sets_and_removes() {
        use crate::ir::SetOp;
        let mut store = social();
        let plan = Plan::Update {
            input: Box::new(scan("Person")),
            ops: vec![
                SetOp::Set {
                    slot: 0,
                    key: "seen".into(),
                    value: lit(n(1.0)),
                },
                SetOp::Remove {
                    slot: 0,
                    key: "age".into(),
                },
            ],
        };
        execute(&plan, &mut store).unwrap();
        // every Person got seen=1 and lost age
        for id in 0..3u32 {
            assert!(matches!(store.prop(id, "seen"), Value::Num(x) if x == 1.0));
            assert!(store.prop(id, "age").is_null());
        }
    }

    /// INSERT enforces unique constraints: the second insert of the same key
    /// errors and is rolled back (the graph keeps exactly the first node).
    #[test]
    fn insert_enforces_unique_constraint() {
        use crate::ir::{InsertNode, Plan};
        let mut store = Builder::default().build();
        store.create_unique_constraint("User", &["email"]).unwrap();
        let ins = |email: &str| Plan::Insert {
            nodes: vec![InsertNode {
                labels: vec!["User".into()],
                props: vec![("email".into(), s(email))],
            }],
            edges: vec![],
        };
        assert!(execute(&ins("a@x"), &mut store).is_ok());
        let err = execute(&ins("a@x"), &mut store); // duplicate
        assert!(err.is_err());
        // rolled back: still exactly one User, and node_count did not grow.
        assert_eq!(store.node_count(), 1);
        assert_eq!(store.nodes_with_label("User").len(), 1);
        // a different key still inserts fine.
        assert!(execute(&ins("b@x"), &mut store).is_ok());
        assert_eq!(store.node_count(), 2);
    }

    /// A single INSERT that creates two colliding nodes is rejected atomically.
    #[test]
    fn insert_rejects_intra_statement_duplicate() {
        use crate::ir::{InsertNode, Plan};
        let mut store = Builder::default().build();
        store.create_unique_constraint("User", &["email"]).unwrap();
        let plan = Plan::Insert {
            nodes: vec![
                InsertNode {
                    labels: vec!["User".into()],
                    props: vec![("email".into(), s("same"))],
                },
                InsertNode {
                    labels: vec!["User".into()],
                    props: vec![("email".into(), s("same"))],
                },
            ],
            edges: vec![],
        };
        assert!(execute(&plan, &mut store).is_err());
        assert_eq!(store.node_count(), 0); // both rolled back
    }

    /// A deleted node is absent from a label scan through the query path — build
    /// the social graph, delete bob (id 1), and the Person scan yields alice+carol.
    #[test]
    fn scan_skips_deleted_node() {
        let mut store = social();
        store.delete_node(1); // bob
        let out = run(
            &scan("Person").project(vec![("name".into(), prop(0, "name"))]),
            &store,
        );
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    // --- Arithmetic (E1) ---

    fn arith(op: crate::ir::ArithOp, l: Expr, r: Expr) -> Expr {
        Expr::Arith {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    /// `age * 2 + 1` for alice(30) = 61 — precedence honored in the hand plan.
    #[test]
    fn arith_eval_computes() {
        use crate::ir::ArithOp::{Add, Mul};
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))))
            .project(vec![(
                "x".into(),
                arith(Add, arith(Mul, prop(0, "age"), lit(n(2.0))), lit(n(1.0))),
            )]);
        assert_eq!(num(&run(&plan, &store).rows[0][0]), 61.0);
    }

    /// A NULL / missing / non-numeric operand yields NULL — the Project node has no
    /// `age`, so `age + 1` is NULL for exactly it.
    #[test]
    fn arith_null_propagates() {
        use crate::ir::ArithOp::Add;
        let store = social();
        let plan = Plan::Scan { label: None }
            .project(vec![("x".into(), arith(Add, prop(0, "age"), lit(n(1.0))))]);
        let nulls = run(&plan, &store)
            .rows
            .iter()
            .filter(|r| r[0].is_null())
            .count();
        assert_eq!(nulls, 1); // only the Project node lacks age
    }

    /// Division / modulo by zero THROWS (matches lenke-core's DataException), via
    /// the fallible read path — `try_run` surfaces the error (K3).
    #[test]
    fn arith_div_or_mod_by_zero_throws() {
        use crate::ir::ArithOp::{Div, Rem};
        let store = social();
        let one = scan("Person").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))));
        for op in [Div, Rem] {
            let plan = one
                .clone()
                .project(vec![("x".into(), arith(op, prop(0, "age"), lit(n(0.0))))]);
            let err = crate::exec::try_run(&plan, &store).unwrap_err();
            assert!(err.contains("division by zero"), "op {op:?}: {err}");
        }
    }

    /// A product that overflows f64 to Inf currently collapses to NULL (the
    /// finite-or-null policy). NOTE: K4 revisits this to KEEP Inf like lenke-core.
    #[test]
    fn arith_overflow_is_null() {
        use crate::ir::ArithOp::Mul;
        let store = social();
        let one = scan("Person").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))));
        let big = one.project(vec![("x".into(), arith(Mul, lit(n(1e308)), lit(n(1e308))))]);
        assert!(run(&big, &store).rows[0][0].is_null());
    }

    // --- Property index + IndexSeek (D1a) ---

    /// A store with two labels sharing an `age` property (some age 30).
    fn indexed_store() -> Store {
        let mut st = Builder::default().build();
        st.add_node(&["P"], &[("age", n(30.0)), ("name", s("a"))]);
        st.add_node(&["P"], &[("age", n(25.0)), ("name", s("b"))]);
        st.add_node(&["P"], &[("age", n(30.0)), ("name", s("c"))]);
        st.add_node(&["Q"], &[("age", n(30.0)), ("name", s("d"))]); // other label
        st
    }

    /// `IndexSeek` returns the SAME rows as `Scan + Filter(=)`, with and without
    /// an index. P nodes with age 30 are a and c (d is a Q, excluded).
    #[test]
    fn index_seek_matches_scan_filter() {
        let mut st = indexed_store();
        let seek = Plan::IndexSeek {
            label: "P".into(),
            key: "age".into(),
            value: n(30.0),
        }
        .project(vec![("name".into(), prop(0, "name"))]);
        let filt = scan("P")
            .filter(cmp(CompareOp::Eq, prop(0, "age"), lit(n(30.0))))
            .project(vec![("name".into(), prop(0, "name"))]);

        let mut want = names_of(&run(&filt, &st), 0);
        want.sort();
        assert_eq!(want, vec!["a", "c"]);
        let mut got = names_of(&run(&seek, &st), 0);
        got.sort();
        assert_eq!(got, want); // no index yet (scan fallback)

        st.create_index("age");
        let mut got = names_of(&run(&seek, &st), 0);
        got.sort();
        assert_eq!(got, want); // index path, same rows
    }

    /// The index is maintained through set/remove/delete.
    #[test]
    fn index_maintained_on_writes() {
        let mut st = indexed_store();
        st.create_index("age");
        let sorted = |st: &Store| {
            let mut v = st.index_lookup("age", &n(30.0)).unwrap();
            v.sort_unstable();
            v
        };
        assert_eq!(sorted(&st), vec![0, 2, 3]); // any-label candidates
        st.set_prop(0, "age", n(25.0)); // 0 leaves the 30 bucket
        assert_eq!(sorted(&st), vec![2, 3]);
        st.delete_node(2); // 2 gone
        assert_eq!(sorted(&st), vec![3]);
        st.remove_prop(3, "age"); // 3 loses the prop
        assert!(st.index_lookup("age", &n(30.0)).unwrap().is_empty());
    }

    /// A transaction rollback restores the index (writes replay through the
    /// primitives, which maintain it).
    #[test]
    fn index_consistent_after_rollback() {
        let mut st = indexed_store();
        st.create_index("age");
        st.begin();
        st.set_prop(0, "age", n(99.0));
        st.delete_node(2);
        st.rollback();
        let mut v = st.index_lookup("age", &n(30.0)).unwrap();
        v.sort_unstable();
        assert_eq!(v, vec![0, 2, 3]);
    }

    /// A NaN / NULL seek value matches nothing (predicate `=` semantics), same as
    /// the filter — even though those values live in a group_key bucket.
    #[test]
    fn index_seek_nan_and_null_match_nothing() {
        let mut st = indexed_store();
        st.create_index("age");
        let seek = |v: Value| {
            Plan::IndexSeek {
                label: "P".into(),
                key: "age".into(),
                value: v,
            }
            .project(vec![("name".into(), prop(0, "name"))])
        };
        assert_eq!(run(&seek(n(f64::NAN)), &st).rows.len(), 0);
        assert_eq!(run(&seek(Value::Null), &st).rows.len(), 0);
    }

    /// `RangeSeek` returns the SAME rows as `Scan + Filter(<op>)` for every range
    /// op, with and without a range index. Hand: ages 30,25,40 (a,b,c).
    #[test]
    fn range_seek_matches_scan_filter_all_ops() {
        let mut st = indexed_store(); // P: a=30, b=25, c=30; Q: d=30
        let ops = [
            (CompareOp::Gt, 25.0, vec!["a", "c"]), // >25 → 30,30
            (CompareOp::Ge, 30.0, vec!["a", "c"]), // >=30
            (CompareOp::Lt, 30.0, vec!["b"]),      // <30 → 25
            (CompareOp::Le, 25.0, vec!["b"]),      // <=25
        ];
        for indexed in [false, true] {
            if indexed {
                st.create_range_index("age");
            }
            for (op, v, want) in &ops {
                let seek = Plan::RangeSeek {
                    label: "P".into(),
                    key: "age".into(),
                    op: *op,
                    value: n(*v),
                }
                .project(vec![("name".into(), prop(0, "name"))]);
                let filt = scan("P")
                    .filter(cmp(*op, prop(0, "age"), lit(n(*v))))
                    .project(vec![("name".into(), prop(0, "name"))]);
                let mut a = names_of(&run(&seek, &st), 0);
                a.sort();
                let mut b = names_of(&run(&filt, &st), 0);
                b.sort();
                assert_eq!(a, *want, "op {op:?} v {v}");
                assert_eq!(a, b, "seek vs filter disagree for {op:?} {v}");
            }
        }
    }

    /// Range ordering is the value contract's total order: a NULL value matches
    /// nothing, and (this engine's design) cross-type compares by rank, so a
    /// string property is > a numeric literal — seek and filter agree on both.
    #[test]
    fn range_seek_null_and_cross_type_match_filter() {
        let mut st = Builder::default().build();
        st.add_node(&["P"], &[("v", n(10.0))]);
        st.add_node(&["P"], &[("v", s("zzz"))]); // string > any number by rank
        st.add_node(&["P"], &[]); // v absent → null
        st.create_range_index("v");
        let check = |st: &Store, op, val: Value| {
            let seek = Plan::RangeSeek {
                label: "P".into(),
                key: "v".into(),
                op,
                value: val.clone(),
            };
            let filt = scan("P").filter(cmp(op, prop(0, "v"), lit(val)));
            (run(&seek, st).rows.len(), run(&filt, st).rows.len())
        };
        // v > 5: 10 and "zzz" (string outranks number); null excluded → 2, agree.
        assert_eq!(check(&st, CompareOp::Gt, n(5.0)), (2, 2));
        // v > null: UNKNOWN for all → 0, agree.
        assert_eq!(check(&st, CompareOp::Gt, Value::Null), (0, 0));
    }

    /// The range index is maintained through set/delete and a transaction rollback.
    #[test]
    fn range_index_maintained_and_rolls_back() {
        let mut st = indexed_store();
        st.create_range_index("age");
        // Candidates > 25 across any label (index_lookup is any-label).
        let cand = |st: &Store, v: f64| st.range_lookup("age", CompareOp::Gt, &n(v)).unwrap().len();
        assert_eq!(cand(&st, 25.0), 3); // a,c (P,30) + d (Q,30)
        st.set_prop(0, "age", n(10.0)); // a drops below 25
        assert_eq!(cand(&st, 25.0), 2);
        st.begin();
        st.delete_node(2); // c gone
        assert_eq!(cand(&st, 25.0), 1);
        st.rollback();
        assert_eq!(cand(&st, 25.0), 2); // restored
    }

    /// A scalar count over an IndexSeek is correct (the seek seeds like a scan).
    #[test]
    fn count_over_index_seek() {
        let mut st = indexed_store();
        st.create_index("age");
        let plan = Plan::IndexSeek {
            label: "P".into(),
            key: "age".into(),
            value: n(30.0),
        }
        .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        assert_eq!(num(&run(&plan, &st).rows[0][0]), 2.0);
    }

    /// Reversed operand order (`literal < prop`) must match `prop > literal` —
    /// exercises the fused filter's operand flip. `28 < age` → alice(30),carol(40).
    #[test]
    fn filter_literal_on_left_flips() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Lt, lit(n(28.0)), prop(0, "age")))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    // --- Expand ---

    /// `MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name` — two slots bound,
    /// row per matching edge.
    #[test]
    fn expand_binds_both_ends() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![
                ("a".into(), prop(0, "name")),
                ("b".into(), prop(1, "name")),
            ]);
        let out = run(&plan, &store);
        let mut pairs: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), as_str(&r[1])))
            .collect();
        pairs.sort();
        // a→b, a→c, b→c (KNOWS only; the WORKS_ON edge is excluded)
        assert_eq!(
            pairs,
            vec![
                ("alice".into(), "bob".into()),
                ("alice".into(), "carol".into()),
                ("bob".into(), "carol".into()),
            ]
        );
    }

    /// An edge-label filter selects: WORKS_ON reaches only the Project.
    #[test]
    fn expand_filters_by_edge_label() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("WORKS_ON"))
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["graphdb"]);
    }

    /// Filtering on the FAR end after an expand — the far slot's property.
    #[test]
    fn filter_on_the_expanded_end() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .filter(cmp(CompareOp::Ge, prop(1, "age"), lit(n(40.0))))
            .project(vec![("a".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // Only edges landing on carol(40): alice→carol, bob→carol.
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob"]);
    }

    /// Incoming direction: who KNOWS carol.
    #[test]
    fn expand_incoming() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("carol"))))
            .expand(0, Dir::In, Some("KNOWS"))
            .project(vec![("who".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob"]);
    }

    /// An unknown edge label matches nothing.
    #[test]
    fn expand_unknown_label_is_empty() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("NOPE"))
            .project(vec![("x".into(), prop(1, "name"))]);
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    /// `expand_edge` binds the traversed edge as a slot: for `(a)-[r:R]->(b)` the
    /// edge is slot 1 and the node slot 2, so `r.weight` reads an edge property and
    /// `b.name` reads a node property.
    #[test]
    fn expand_edge_binds_edge_and_reads_edge_prop() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        st.set_edge_prop(eid, "weight", n(0.5));
        let plan = scan("P").expand_edge(0, Dir::Out, Some("R")).project(vec![
            ("w".into(), prop(1, "weight")), // edge slot
            ("b".into(), prop(2, "name")),   // node slot
        ]);
        let out = run(&plan, &st);
        assert_eq!(out.rows.len(), 1);
        assert!(matches!(&out.rows[0][0], Value::Num(x) if *x == 0.5));
        assert_eq!(as_str(&out.rows[0][1]), "b");
    }

    /// An edge slot with no such property reads NULL.
    #[test]
    fn expand_edge_absent_prop_is_null() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]);
        let b = st.add_node(&["P"], &[]);
        st.add_edge(a, b, "R");
        let plan = scan("P")
            .expand_edge(0, Dir::Out, Some("R"))
            .project(vec![("w".into(), prop(1, "weight"))]);
        let out = run(&plan, &st);
        assert_eq!(out.rows.len(), 1);
        assert!(out.rows[0][0].is_null());
    }

    /// Filtering on an edge property keeps only matching edges. a→b (w=0.5),
    /// a→c (w=0.2); `WHERE r.w > 0.4` → only b.
    #[test]
    fn filter_on_edge_property() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let c = st.add_node(&["P"], &[("name", s("c"))]);
        st.add_edge(a, b, "R");
        let e1 = st.out(a)[0].eid;
        st.add_edge(a, c, "R");
        let e2 = st.out(a)[1].eid;
        st.set_edge_prop(e1, "w", n(0.5));
        st.set_edge_prop(e2, "w", n(0.2));
        let plan = scan("P")
            .expand_edge(0, Dir::Out, Some("R"))
            .filter(cmp(CompareOp::Gt, prop(1, "w"), lit(n(0.4))))
            .project(vec![("b".into(), prop(2, "name"))]);
        assert_eq!(names_of(&run(&plan, &st), 0), vec!["b"]);
    }

    fn as_str(v: &Value) -> String {
        match v {
            Value::Str(x) => x.to_string(),
            other => format!("{other:?}"),
        }
    }

    fn num(v: &Value) -> f64 {
        match v {
            Value::Num(x) => *x,
            other => panic!("expected a number, got {other:?}"),
        }
    }

    // --- Aggregate / group-by ---

    fn agg(func: AggFn, arg: Option<Expr>, distinct: bool, name: &str) -> Agg {
        Agg {
            func,
            arg,
            distinct,
            name: name.to_string(),
        }
    }

    /// Scalar `count(*)` over a label — one row, the count.
    #[test]
    fn scalar_count_star() {
        let store = social();
        let plan = scan("Person").aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        let out = run(&plan, &store);
        assert_eq!(out.names, vec!["c"]);
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&out.rows[0][0]), 3.0); // alice, bob, carol
    }

    /// sum / min / max / avg over the age column, hand-computed: 30,25,40.
    #[test]
    fn scalar_sum_min_max_avg() {
        let store = social();
        let plan = scan("Person").aggregate(
            vec![],
            vec![
                agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
                agg(AggFn::Min, Some(prop(0, "age")), false, "lo"),
                agg(AggFn::Max, Some(prop(0, "age")), false, "hi"),
                agg(AggFn::Avg, Some(prop(0, "age")), false, "av"),
            ],
        );
        let out = run(&plan, &store);
        let r = &out.rows[0];
        assert_eq!(num(&r[0]), 95.0); // 30+25+40
        assert_eq!(num(&r[1]), 25.0);
        assert_eq!(num(&r[2]), 40.0);
        assert_eq!(num(&r[3]), 95.0 / 3.0);
    }

    /// `count(*)` grouped by a property — a row per distinct value, first-seen
    /// order. Group on `city`: alice/carol="nyc", bob="sf".
    #[test]
    fn group_count_by_property() {
        let mut b = Builder::default();
        b.node(&["P"], &[("city", s("nyc"))]);
        b.node(&["P"], &[("city", s("sf"))]);
        b.node(&["P"], &[("city", s("nyc"))]);
        let store = b.build();
        let plan = scan("P").aggregate(
            vec![("city".into(), prop(0, "city"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        let out = run(&plan, &store);
        assert_eq!(out.names, vec!["city", "c"]);
        // first-seen order: nyc (row 0), then sf (row 1).
        assert_eq!(as_str(&out.rows[0][0]), "nyc");
        assert_eq!(num(&out.rows[0][1]), 2.0);
        assert_eq!(as_str(&out.rows[1][0]), "sf");
        assert_eq!(num(&out.rows[1][1]), 1.0);
    }

    /// `count(arg)` ignores nulls; `count(DISTINCT arg)` ignores nulls AND
    /// duplicates. Ages: 10, 10, null, 20 → count=3, distinct=2.
    #[test]
    fn count_arg_and_count_distinct_skip_nulls() {
        let mut b = Builder::default();
        b.node(&["P"], &[("v", n(10.0))]);
        b.node(&["P"], &[("v", n(10.0))]);
        b.node(&["P"], &[]); // no v → null
        b.node(&["P"], &[("v", n(20.0))]);
        let store = b.build();
        let plan = scan("P").aggregate(
            vec![],
            vec![
                agg(AggFn::Count, Some(prop(0, "v")), false, "c"),
                agg(AggFn::Count, Some(prop(0, "v")), true, "cd"),
            ],
        );
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][0]), 3.0); // non-null count
        assert_eq!(num(&out.rows[0][1]), 2.0); // distinct non-null: {10, 20}
    }

    /// Over nothing, `count` and `sum` are both 0 but `avg` is NULL — matching
    /// lenke-core (the GQL/Cypher convention; the differential fuzzer flagged the
    /// earlier SQL-style `sum → NULL`).
    #[test]
    fn sum_over_empty_is_zero_avg_is_null() {
        let store = social();
        // No node has this label → empty input to the scalar aggregate.
        let plan = Plan::Scan {
            label: Some("Nonexistent".into()),
        }
        .aggregate(
            vec![],
            vec![
                agg(AggFn::Count, None, false, "c"),
                agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
                agg(AggFn::Avg, Some(prop(0, "age")), false, "a"),
            ],
        );
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1); // scalar aggregate still emits one row
        assert_eq!(num(&out.rows[0][0]), 0.0); // count(*) = 0
        assert_eq!(num(&out.rows[0][1]), 0.0); // sum = 0
        assert!(out.rows[0][2].is_null()); // avg = NULL
    }

    /// A grouped aggregate over empty input emits ZERO rows (unlike the scalar
    /// case) — there are no groups.
    #[test]
    fn grouped_over_empty_is_zero_rows() {
        let store = social();
        let plan = Plan::Scan {
            label: Some("Nonexistent".into()),
        }
        .aggregate(
            vec![("k".into(), prop(0, "age"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    /// Aggregate after an Expand: out-degree per person (count of KNOWS edges),
    /// grouped by the source. alice→2, bob→1, carol→0(absent).
    #[test]
    fn count_out_degree_grouped_by_source() {
        let store = social();
        let plan = scan("Person").expand(0, Dir::Out, Some("KNOWS")).aggregate(
            vec![("who".into(), prop(0, "name"))],
            vec![agg(AggFn::Count, None, false, "deg")],
        );
        let out = run(&plan, &store);
        let mut got: Vec<(String, f64)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), num(&r[1])))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        // carol has no outgoing KNOWS, so she is absent from the expanded rows.
        assert_eq!(got, vec![("alice".into(), 2.0), ("bob".into(), 1.0)]);
    }

    /// Scalar `count(*)` over a single Expand — the frontier fast path. Hand
    /// count of KNOWS edges: alice→{bob,carol}, bob→{carol} = 3.
    #[test]
    fn fused_count_star_one_hop() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&out.rows[0][0]), 3.0);
    }

    /// Scalar `count(*)` over a two-hop chain. Hand count of length-2 KNOWS
    /// walks: only alice→bob→carol (bob is the only reached node with an
    /// outgoing KNOWS) = 1.
    #[test]
    fn fused_count_star_two_hop() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .expand(1, Dir::Out, Some("KNOWS"))
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][0]), 1.0);
    }

    /// 2-hop `count(*)` where an intermediate is reached by MULTIPLE paths — the
    /// dedup-with-multiplicity path must scale by how many times it was reached.
    /// a→x, b→x (x reached twice), x→p, x→q. Length-2 walks: a→x→{p,q} and
    /// b→x→{p,q} = 4 (x itself reaches p,q which are sinks).
    #[test]
    fn fused_count_star_two_hop_with_multiplicity() {
        let mut bld = Builder::default();
        let a = bld.node(&["P"], &[]);
        let b = bld.node(&["P"], &[]);
        let x = bld.node(&["P"], &[]);
        let p = bld.node(&["P"], &[]);
        let q = bld.node(&["P"], &[]);
        bld.edge(a, x, "R");
        bld.edge(b, x, "R");
        bld.edge(x, p, "R");
        bld.edge(x, q, "R");
        let store = bld.build();
        let plan = scan("P")
            .expand(0, Dir::Out, Some("R"))
            .expand(1, Dir::Out, Some("R"))
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        assert_eq!(num(&run(&plan, &store).rows[0][0]), 4.0);
    }

    /// `count(DISTINCT c)` over the two-hop chain: the distinct endpoints are
    /// {carol} = 1, deduped in the bitset path.
    #[test]
    fn fused_count_distinct_endpoint() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .expand(1, Dir::Out, Some("KNOWS"))
            .aggregate(
                vec![],
                vec![agg(AggFn::Count, Some(Expr::Slot(2)), true, "c")],
            );
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][0]), 1.0);
    }

    /// Grouped count over an Expand chain (the frontier-mode aggregate). Group
    /// the reached KNOWS neighbours by name: alice→{bob,carol}, bob→{carol}, so
    /// the frontier is [bob, carol, carol] → bob:1, carol:2, first-seen order.
    #[test]
    fn frontier_grouped_count_matches() {
        let store = social();
        let plan = scan("Person").expand(0, Dir::Out, Some("KNOWS")).aggregate(
            vec![("who".into(), prop(1, "name"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        let out = run(&plan, &store);
        assert_eq!(as_str(&out.rows[0][0]), "bob");
        assert_eq!(num(&out.rows[0][1]), 1.0);
        assert_eq!(as_str(&out.rows[1][0]), "carol");
        assert_eq!(num(&out.rows[1][1]), 2.0);
    }

    /// The node-grouped count path when DISTINCT nodes share a property value —
    /// the level-2 merge must combine them. a→{b,c,d}; b,d are in nyc, c in sf.
    /// Group reached neighbours by city: nyc:2 (b,d), sf:1, first-seen order.
    #[test]
    fn node_grouped_count_merges_shared_value() {
        let mut b = Builder::default();
        let a = b.node(&["P"], &[("name", s("a"))]);
        let n1 = b.node(&["P"], &[("city", s("nyc"))]);
        let n2 = b.node(&["P"], &[("city", s("sf"))]);
        let n3 = b.node(&["P"], &[("city", s("nyc"))]);
        b.edge(a, n1, "R");
        b.edge(a, n2, "R");
        b.edge(a, n3, "R");
        let store = b.build();
        let plan = scan("P").expand(0, Dir::Out, Some("R")).aggregate(
            vec![("city".into(), prop(1, "city"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        let out = run(&plan, &store);
        assert_eq!(as_str(&out.rows[0][0]), "nyc");
        assert_eq!(num(&out.rows[0][1]), 2.0);
        assert_eq!(as_str(&out.rows[1][0]), "sf");
        assert_eq!(num(&out.rows[1][1]), 1.0);
    }

    /// A grouped SUM over the frontier's property, to exercise a non-count agg on
    /// the frontier path: sum the neighbours' ages by name. bob(25) reached once;
    /// carol(40) reached twice → 80.
    #[test]
    fn frontier_grouped_sum_matches() {
        let store = social();
        let plan = scan("Person").expand(0, Dir::Out, Some("KNOWS")).aggregate(
            vec![("who".into(), prop(1, "name"))],
            vec![agg(AggFn::Sum, Some(prop(1, "age")), false, "s")],
        );
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][1]), 25.0); // bob
        assert_eq!(num(&out.rows[1][1]), 80.0); // carol twice
    }

    /// An unknown final edge label fuses to zero rows.
    #[test]
    fn fused_count_unknown_label_is_zero() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("NOPE"))
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        assert_eq!(num(&run(&plan, &store).rows[0][0]), 0.0);
    }

    // --- Order + Page ---

    fn asc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: false,
            nulls_first: false,
        }
    }
    fn desc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: true,
            nulls_first: false,
        }
    }

    /// NULL placement is a language contract independent of direction: GQL keeps
    /// NULLs LAST in both ASC and DESC (a NULL prop must not float to the front
    /// under DESC). Uses a graph where one node lacks `age`.
    #[test]
    fn gql_order_by_desc_keeps_nulls_last() {
        let mut b = Builder::default();
        b.node(&["P"], &[("age", n(30.0))]);
        b.node(&["P"], &[("age", n(10.0))]);
        b.node(&["P"], &[]); // no age → NULL
        let store = b.build();
        let ages =
            |q: &str| -> Vec<String> { names_of(&run(&crate::gql::parse(q).unwrap(), &store), 1) };
        // DESC: 30, 10, then NULL last (not first).
        assert_eq!(
            ages("MATCH (p:P) RETURN p.age AS a0, p.age AS a1 ORDER BY a0 DESC"),
            vec!["Num(30.0)", "Num(10.0)", "Null"]
        );
        // ASC: 10, 30, NULL last.
        assert_eq!(
            ages("MATCH (p:P) RETURN p.age AS a0, p.age AS a1 ORDER BY a0 ASC"),
            vec!["Num(10.0)", "Num(30.0)", "Null"]
        );
    }

    /// Gremlin's `order()` places NULLs FIRST (the other language default) — the
    /// same shared OrderPage, driven by `SortKey.nulls_first`.
    #[test]
    fn gremlin_order_keeps_nulls_first() {
        let mut b = Builder::default();
        b.node(&["P"], &[("age", n(30.0)), ("name", s("a"))]);
        b.node(&["P"], &[("age", n(10.0)), ("name", s("b"))]);
        b.node(&["P"], &[("name", s("c"))]); // no age → NULL
        let store = b.build();
        let out = run(
            &crate::gremlin::parse("g.V().hasLabel('P').order().by('age').values('name')").unwrap(),
            &store,
        );
        // NULL-age node ('c') sorts FIRST, then 10 ('b'), 30 ('a').
        assert_eq!(names_of(&out, 0), vec!["c", "b", "a"]);
    }

    /// ORDER BY age ascending, then project name.
    #[test]
    fn order_by_ascending() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![asc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // ages 30,25,40 -> bob(25), alice(30), carol(40)
        assert_eq!(names_of(&out, 0), vec!["bob", "alice", "carol"]);
    }

    /// Descending reverses it.
    #[test]
    fn order_by_descending() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![desc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["carol", "alice", "bob"]);
    }

    /// ORDER BY ... LIMIT is a top-k prefix of the sorted order.
    #[test]
    fn order_then_limit_is_top_k() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![desc(0, "age")], None, Some(2))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["carol", "alice"]); // two oldest
    }

    /// SKIP then LIMIT is a paging window over the sorted order.
    #[test]
    fn order_skip_limit_paging_window() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![asc(0, "age")], Some(1), Some(1))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // sorted bob,alice,carol; skip 1, take 1 -> alice
        assert_eq!(names_of(&out, 0), vec!["alice"]);
    }

    /// Nulls sort LAST in ascending order (the value contract's policy).
    #[test]
    fn nulls_sort_last_ascending() {
        let mut b = Builder::default();
        b.node(&["P"], &[("name", s("has30")), ("age", n(30.0))]);
        b.node(&["P"], &[("name", s("noage"))]); // null age
        b.node(&["P"], &[("name", s("has10")), ("age", n(10.0))]);
        let store = b.build();
        let plan = scan("P")
            .order_page(vec![asc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // 10, 30, then null last
        assert_eq!(names_of(&out, 0), vec!["has10", "has30", "noage"]);
    }

    /// Multi-key: city ascending, then age descending within a city.
    #[test]
    fn multi_key_order() {
        let mut b = Builder::default();
        b.node(
            &["P"],
            &[("name", s("a")), ("city", s("nyc")), ("age", n(30.0))],
        );
        b.node(
            &["P"],
            &[("name", s("b")), ("city", s("sf")), ("age", n(40.0))],
        );
        b.node(
            &["P"],
            &[("name", s("c")), ("city", s("nyc")), ("age", n(50.0))],
        );
        let store = b.build();
        let plan = scan("P")
            .order_page(vec![asc(0, "city"), desc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // nyc: c(50) before a(30); then sf: b(40)
        assert_eq!(names_of(&out, 0), vec!["c", "a", "b"]);
    }

    // --- Distinct ---

    /// `RETURN DISTINCT city` over nyc/sf/nyc -> two rows, first-seen order.
    #[test]
    fn distinct_dedups_projected_column() {
        let mut b = Builder::default();
        b.node(&["P"], &[("city", s("nyc"))]);
        b.node(&["P"], &[("city", s("sf"))]);
        b.node(&["P"], &[("city", s("nyc"))]);
        let store = b.build();
        let plan = scan("P")
            .project(vec![("city".into(), prop(0, "city"))])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["nyc", "sf"]);
    }

    /// DISTINCT is over the WHOLE projected row: (city, tier) tuples dedup, so a
    /// repeated city with a different tier is NOT collapsed.
    #[test]
    fn distinct_is_over_the_whole_row() {
        let mut b = Builder::default();
        b.node(&["P"], &[("city", s("nyc")), ("tier", n(1.0))]);
        b.node(&["P"], &[("city", s("nyc")), ("tier", n(2.0))]);
        b.node(&["P"], &[("city", s("nyc")), ("tier", n(1.0))]); // dup of row 0
        let store = b.build();
        let plan = scan("P")
            .project(vec![
                ("city".into(), prop(0, "city")),
                ("tier".into(), prop(0, "tier")),
            ])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2);
        assert_eq!(num(&out.rows[0][1]), 1.0);
        assert_eq!(num(&out.rows[1][1]), 2.0);
    }

    /// DISTINCT uses the grouping notion, not predicate equality: two NaNs
    /// collapse to one row.
    #[test]
    fn distinct_collapses_nans() {
        let mut b = Builder::default();
        b.node(&["P"], &[("v", n(f64::NAN))]);
        b.node(&["P"], &[("v", n(f64::NAN))]);
        b.node(&["P"], &[("v", n(1.0))]);
        let store = b.build();
        let plan = scan("P")
            .project(vec![("v".into(), prop(0, "v"))])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2); // one NaN row + one 1.0 row
    }

    /// DISTINCT after Expand: the set of nodes reached by KNOWS from anyone.
    #[test]
    fn distinct_reached_set() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![("who".into(), prop(1, "name"))])
            .distinct();
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["bob", "carol"]);
    }

    // --- Join (multi-pattern / shared variable) ---

    /// `MATCH (a)-[:KNOWS]->(b), (a)-[:WORKS_ON]->(c)` sharing `a`. Left slots
    /// [a,b], right slots [a,c]; join on left a (0) == right a (0); output slots
    /// [a, b, a', c]. Only alice has a WORKS_ON, so only her KNOWS rows survive.
    #[test]
    fn join_shared_start_variable() {
        let store = social();
        let left = scan("Person").expand(0, Dir::Out, Some("KNOWS"));
        let right = scan("Person").expand(0, Dir::Out, Some("WORKS_ON"));
        let plan = Plan::join(left, right, vec![(0, 0)]).project(vec![
            ("a".into(), prop(0, "name")),
            ("b".into(), prop(1, "name")),
            ("c".into(), prop(3, "name")), // right slot 1 -> output slot 2+1=3
        ]);
        let out = run(&plan, &store);
        let mut pairs: Vec<(String, String, String)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), as_str(&r[1]), as_str(&r[2])))
            .collect();
        pairs.sort();
        // alice KNOWS {bob,carol}, WORKS_ON {graphdb}: 2x1 = 2 rows. bob has no
        // WORKS_ON, so bob->carol drops.
        assert_eq!(
            pairs,
            vec![
                ("alice".into(), "bob".into(), "graphdb".into()),
                ("alice".into(), "carol".into(), "graphdb".into()),
            ]
        );
    }

    /// The join fans out to the PRODUCT per shared key: a person with 2 R and 2 S
    /// neighbours yields 4 combined rows.
    #[test]
    fn join_is_product_per_shared_key() {
        let mut b = Builder::default();
        let a = b.node(&["P"], &[("name", s("a"))]);
        let r1 = b.node(&["P"], &[("name", s("r1"))]);
        let r2 = b.node(&["P"], &[("name", s("r2"))]);
        let s1 = b.node(&["P"], &[("name", s("s1"))]);
        let s2 = b.node(&["P"], &[("name", s("s2"))]);
        b.edge(a, r1, "R");
        b.edge(a, r2, "R");
        b.edge(a, s1, "S");
        b.edge(a, s2, "S");
        let store = b.build();
        let left = scan("P").expand(0, Dir::Out, Some("R"));
        let right = scan("P").expand(0, Dir::Out, Some("S"));
        let plan = Plan::join(left, right, vec![(0, 0)]).project(vec![
            ("r".into(), prop(1, "name")),
            ("s".into(), prop(3, "name")),
        ]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 4); // {r1,r2} x {s1,s2}
        let mut pairs: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), as_str(&r[1])))
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("r1".into(), "s1".into()),
                ("r1".into(), "s2".into()),
                ("r2".into(), "s1".into()),
                ("r2".into(), "s2".into()),
            ]
        );
    }

    /// A left key with no right match drops (inner join).
    #[test]
    fn join_drops_unmatched() {
        let store = social();
        // Everyone with a KNOWS edge, joined to everyone with a WORKS_ON edge on
        // the SAME person. Only alice has both, so bob (KNOWS only) drops.
        let left = scan("Person").expand(0, Dir::Out, Some("KNOWS"));
        let right = scan("Person").expand(0, Dir::Out, Some("WORKS_ON"));
        let plan = Plan::join(left, right, vec![(0, 0)])
            .project(vec![("a".into(), prop(0, "name"))])
            .distinct();
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["alice"]);
    }

    // --- VarLength (quantified hops) ---

    /// A linear chain a->b->c. `{1,2}` from a reaches b (len 1) and c (len 2):
    /// two rows.
    #[test]
    fn varlen_chain_one_to_two() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, Some("R"), 1, 2, true)
            .project(vec![("end".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c"]);
    }

    /// `{0,2}` includes the source itself at length 0: a, b, c.
    #[test]
    fn varlen_zero_includes_source() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, Some("R"), 0, 2, true)
            .project(vec![("end".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["a", "b", "c"]); // a at length 0
    }

    /// THE trail-vs-walk discriminator: a single self-loop a->a. `{1,2}`:
    /// - walk (trail=false) reuses the edge, so len1 AND len2 both reach a -> 2 rows;
    /// - trail (trail=true) may not reuse it, so only len1 -> 1 row.
    #[test]
    fn varlen_trail_vs_walk_on_a_self_loop() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        b.edge(a, a, "R"); // self-loop
        let store = b.build();
        let base = scan("N");

        let walk = base
            .clone()
            .var_length(0, Dir::Out, Some("R"), 1, 2, false)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 2, "walk reuses the edge");

        let trail = base
            .var_length(0, Dir::Out, Some("R"), 1, 2, true)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(
            run(&trail, &store).rows.len(),
            1,
            "trail may not reuse the edge"
        );
    }

    /// A 2-cycle a<->b (two directed edges a->b, b->a). `{1,3}` as a TRAIL from a:
    /// len1 a->b (edge0); len2 a->b->a (edge0,edge1); len3 a->b->a->b (edge0,
    /// edge1, then edge0 again -> reused -> blocked). So endpoints b, a -> 2 rows.
    /// As a WALK, len3 a->b->a->b is allowed -> endpoints b, a, b -> 3 rows.
    #[test]
    fn varlen_two_cycle_trail_bounds_edge_reuse() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        b.edge(a, bb, "R"); // edge 0
        b.edge(bb, a, "R"); // edge 1
        let store = b.build();
        let from_a = scan("N").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))));

        let trail = from_a
            .clone()
            .var_length(0, Dir::Out, Some("R"), 1, 3, true)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&trail, &store).rows.len(), 2); // b (len1), a (len2)

        let walk = from_a
            .var_length(0, Dir::Out, Some("R"), 1, 3, false)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 3); // b, a, b
    }

    /// Exact length `{2,2}` emits only the 2-hop endpoints.
    #[test]
    fn varlen_exact_length() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, Some("R"), 2, 2, true)
            .project(vec![("end".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["c"]); // only the 2-hop endpoint
    }

    // --- ShortestPath ---

    /// A diamond a->b, a->c, b->d, c->d. Shortest from a: b(1), c(1), d(2). `d` is
    /// reachable two ways at distance 2 but emitted ONCE (ANY-shortest).
    #[test]
    fn shortest_path_diamond_reaches_each_once() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        let d = b.node(&["N"], &[("name", s("d"))]);
        b.edge(a, bb, "R");
        b.edge(a, c, "R");
        b.edge(bb, d, "R");
        b.edge(c, d, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(0, Dir::Out, Some("R"), None)
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c", "d"]); // d once, not twice
    }

    /// The source is not emitted, and a direct edge wins over a longer path: with
    /// a->c direct AND a->b->c, c is reached at distance 1, once.
    #[test]
    fn shortest_path_takes_the_short_route() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(a, c, "R"); // direct shortcut
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(0, Dir::Out, Some("R"), None)
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c"]); // both at distance 1; source a not emitted
    }

    /// `max` caps the hop distance: on a chain a->b->c->d with max 2, d (distance
    /// 3) is unreachable.
    #[test]
    fn shortest_path_respects_max_hops() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        let d = b.node(&["N"], &[("name", s("d"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, d, "R");
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(0, Dir::Out, Some("R"), Some(2))
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["b", "c"]); // d (distance 3) beyond the cap
    }

    /// A cycle does not loop forever — each node is reached once.
    #[test]
    fn shortest_path_terminates_on_a_cycle() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, a, "R"); // cycle back
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .shortest_path(0, Dir::Out, Some("R"), None)
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        // b(1), c(2); a is the source, not re-emitted despite the cycle back.
        assert_eq!(got, vec!["b", "c"]);
    }

    // --- Lineage (path) ---

    /// A chain a->b->c. `RETURN path` over the 2-hop expand yields the hand-
    /// computed path [a, b, c] (node ids), and the path grows one node per hop.
    #[test]
    fn path_is_the_hop_sequence() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        let store = b.build();
        // (a)-[:R]->(x)-[:R]->(y) starting at a, RETURN path.
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .expand(0, Dir::Out, Some("R"))
            .expand(1, Dir::Out, Some("R"))
            .project(vec![("p".into(), Expr::Path)]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        // path = [a, b, c] as node ids (a=0, b=1, c=2).
        match &out.rows[0][0] {
            Value::List(items) => {
                let ids: Vec<f64> = items
                    .iter()
                    .map(|v| match v {
                        Value::Num(x) => *x,
                        other => panic!("path element not a node id: {other:?}"),
                    })
                    .collect();
                assert_eq!(ids, vec![f64::from(a), f64::from(bb), f64::from(c)]);
            }
            other => panic!("expected a path list, got {other:?}"),
        }
    }

    /// Expand tracks the traversed EDGE in the lineage too: over a->b->c the
    /// relationships accessor recovers edge ids [0, 1] (creation order), the
    /// parallel of `path_is_the_hop_sequence` for edges.
    #[test]
    fn expand_lineage_tracks_edges() {
        use crate::ir::PathPart;
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        b.edge(a, bb, "R"); // edge id 0
        b.edge(bb, c, "R"); // edge id 1
        let store = b.build();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .expand(0, Dir::Out, Some("R"))
            .expand(1, Dir::Out, Some("R"))
            .project(vec![(
                "es".into(),
                Expr::PathAccess {
                    part: PathPart::Relationships,
                },
            )]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1);
        match &out.rows[0][0] {
            Value::List(items) => {
                let eids: Vec<f64> = items
                    .iter()
                    .map(|v| match v {
                        Value::Num(x) => *x,
                        other => panic!("edge element not an id: {other:?}"),
                    })
                    .collect();
                assert_eq!(eids, vec![0.0, 1.0]);
            }
            other => panic!("expected an edge list, got {other:?}"),
        }
    }

    /// A one-hop path is two nodes; the source's own path (length-0 walk via a
    /// bare scan) is one node.
    #[test]
    fn path_length_grows_with_hops() {
        let store = social();
        // alice -KNOWS-> {bob, carol}; RETURN path per edge.
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))))
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![("p".into(), Expr::Path)]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2); // alice->bob, alice->carol
        for row in &out.rows {
            match &row[0] {
                Value::List(items) => assert_eq!(items.len(), 2), // [alice, neighbour]
                other => panic!("expected path list, got {other:?}"),
            }
        }
    }

    /// GATING: a lineage-free plan builds NO sidecar (pays nothing). Only a plan
    /// that reads Path tracks it. Checked at the batch level via `needs_lineage`
    /// and the pulled batch's `lineage` field.
    #[test]
    fn lineage_free_plan_builds_no_sidecar() {
        let store = social();
        let plain = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![("b".into(), prop(1, "name"))]);
        assert!(!super::needs_lineage(&plain), "no Path read -> no lineage");
        // The pulled batch (before the lineage-dropping Project) has no sidecar.
        let inner = scan("Person").expand(0, Dir::Out, Some("KNOWS"));
        assert!(super::pull(&inner, &store, false)
            .unwrap()
            .lineage
            .is_none());

        let with_path = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![("p".into(), Expr::Path)]);
        assert!(super::needs_lineage(&with_path), "Path read -> lineage");
        // With track=true the expand carries a sidecar.
        assert!(super::pull(&inner, &store, true).unwrap().lineage.is_some());
    }

    /// Lineage survives a reorder: ORDER BY over a path-tracking plan keeps each
    /// row's path aligned with its row.
    #[test]
    fn lineage_follows_a_reorder() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a")), ("age", n(1.0))]);
        let bb = b.node(&["N"], &[("name", s("b")), ("age", n(3.0))]);
        let c = b.node(&["N"], &[("name", s("c")), ("age", n(2.0))]);
        b.edge(a, bb, "R");
        b.edge(a, c, "R");
        let store = b.build();
        // a -> {b(age3), c(age2)}; order by the neighbour's age asc, RETURN path.
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .expand(0, Dir::Out, Some("R"))
            .order_page(vec![asc(1, "age")], None, None)
            .project(vec![
                ("last".into(), prop(1, "name")),
                ("p".into(), Expr::Path),
            ]);
        let out = run(&plan, &store);
        // sorted by neighbour age: c(2) then b(3). Each path ends at its own node.
        assert_eq!(as_str(&out.rows[0][0]), "c");
        assert_eq!(as_str(&out.rows[1][0]), "b");
        let last_of = |row: &[Value]| match &row[1] {
            Value::List(items) => match items.last() {
                Some(Value::Num(x)) => *x,
                other => panic!("path tail not a node: {other:?}"),
            },
            other => panic!("expected path, got {other:?}"),
        };
        assert_eq!(last_of(&out.rows[0]), f64::from(c)); // path for c ends at c
        assert_eq!(last_of(&out.rows[1]), f64::from(bb)); // path for b ends at b
    }
}

#[cfg(test)]
mod perf {
    use crate::opt::optimize;
    use crate::store::{Builder, Store};
    use crate::value::Value;
    use std::sync::Arc;
    use std::time::Instant;

    fn build(nodes: usize, deg: usize) -> Store {
        let mut b = Builder::default();
        for i in 0..nodes {
            b.node(
                &["Person"],
                &[
                    ("name", Value::Str(Arc::from(format!("n{i}").as_str()))),
                    ("age", Value::Num((i % 100) as f64)),
                ],
            );
        }
        for i in 0..nodes {
            for d in 0..deg {
                b.edge(i as u32, ((i * 7 + d * 13 + 1) % nodes) as u32, "R");
            }
        }
        b.build()
    }

    #[test]
    #[ignore = "perf probe"]
    fn zzz_perf() {
        let (nodes, deg) = (200_000usize, 4usize);
        let t = Instant::now();
        let store = build(nodes, deg);
        eprintln!(
            "PERF build {nodes} nodes / {} edges: {:?}",
            nodes * deg,
            t.elapsed()
        );
        for q in [
            "MATCH (p:Person) WHERE p.age > 90 RETURN p.name",
            "MATCH (a:Person)-[:R]->(b) RETURN count(*) AS c",
            "MATCH (a:Person)-[:R]->(b) RETURN b.name AS who, count(*) AS c",
            "MATCH (a:Person)-[:R]->(b) RETURN b.age AS age, count(*) AS c",
            "MATCH (a:Person)-[:R]->()-[:R]->(c) RETURN count(DISTINCT c) AS c",
            "MATCH (a:Person)-[:R]->(b)-[:R]->(c) RETURN count(*) AS c",
        ] {
            let plan = optimize(super::super::gql::parse(q).unwrap());
            let mut best = f64::MAX;
            let mut rows = 0;
            for _ in 0..5 {
                let t = Instant::now();
                let out = super::run(&plan, &store);
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                rows = out.rows.len();
            }
            eprintln!("PERF {best:>9.2} ms  rows {rows:>8}  {q}");
        }
    }
}
