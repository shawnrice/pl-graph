use super::evaluator::*;
use super::*;
use crate::batch::{Batch, Col};
use crate::store::Store;
use crate::value::Value;

/// Run a `MATCH … SET/REMOVE/DELETE` update: pull the frontier, apply the ops in a
/// statement transaction (with the deferred-constraint recheck), and RETURN the
/// frontier batch it operated on — so `UpdateReturn` can read the just-written values
/// over the same bindings. `Plan::Update` discards the returned batch.
pub(super) fn run_update(
    store: &mut Store,
    input: &Plan,
    ops: &[crate::ir::SetOp],
    track: bool,
) -> Result<Batch, String> {
    // Read phase: run the match and compute every write into OWNED data — the pulled
    // frontier batch owns its columns (no store borrow), so it survives the mutation
    // and seeds an `UpdateReturn` tail.
    let frontier = pull(input, store, track)?;
    let mut applied: Vec<Applied> = Vec::new();
    {
        let batch = &frontier;
        for op in ops {
            match op {
                crate::ir::SetOp::Set { slot, key, value } => {
                    let vals = eval(value, store, batch)?;
                    match batch.slot(*slot) {
                        Col::Nodes(ids) => {
                            for (i, &id) in ids.iter().enumerate() {
                                // A string `id` is the element's fixed identity —
                                // re-keying it would break element_id / round-trip.
                                if key == "id" && node_id_is_identity(store, id) {
                                    return Err(ID_IMMUTABLE_ERR.into());
                                }
                                applied.push(Applied::Set(id, key.clone(), vals.value_at(i)));
                            }
                        }
                        Col::Edges(eids) => {
                            for (i, &e) in eids.iter().enumerate() {
                                if key == "id" && edge_id_is_identity(store, e) {
                                    return Err(ID_IMMUTABLE_ERR.into());
                                }
                                applied.push(Applied::SetEdge(e, key.clone(), vals.value_at(i)));
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
                crate::ir::SetOp::AddLabel { slot, label } => {
                    if let Col::Nodes(ids) = batch.slot(*slot) {
                        for &id in ids {
                            applied.push(Applied::AddLabel(id, label.clone()));
                        }
                    }
                }
                crate::ir::SetOp::RemoveLabel { slot, label } => {
                    if let Col::Nodes(ids) = batch.slot(*slot) {
                        for &id in ids {
                            applied.push(Applied::RemoveLabel(id, label.clone()));
                        }
                    }
                }
                crate::ir::SetOp::Delete { slot, detach } => match batch.slot(*slot) {
                    Col::Nodes(ids) => {
                        for &id in ids {
                            applied.push(Applied::DeleteNode(id, *detach));
                        }
                    }
                    Col::Edges(eids) => {
                        for &e in eids {
                            applied.push(Applied::DeleteEdge(e));
                        }
                    }
                    _ => {}
                },
            }
        }
    }
    // Write phase, as a TRANSACTION so a constraint violation rolls the whole
    // statement back — matching INSERT/_MERGE. Previously SET/REMOVE applied
    // with no recheck, so `SET u.email = <existing>` silently violated a
    // unique constraint and `REMOVE u.email` a required one.
    let scope = stmt_begin(store);
    // Pass 1: property writes and EDGE deletes. Node deletes are deferred to
    // pass 2 so an edge deleted here (`DELETE r, a, b`) leaves its endpoints
    // relationship-free before the non-DETACH node-delete check runs.
    let mut node_deletes: Vec<(u32, bool)> = Vec::new();
    for a in applied {
        match a {
            Applied::Set(node, key, value) => store.set_prop(node, &key, value),
            Applied::Remove(node, key) => store.remove_prop(node, &key),
            Applied::AddLabel(node, label) => store.add_label(node, &label),
            Applied::RemoveLabel(node, label) => store.remove_label(node, &label),
            Applied::SetEdge(eid, key, value) => store.set_edge_prop(eid, &key, value),
            Applied::RemoveEdge(eid, key) => store.remove_edge_prop(eid, &key),
            Applied::DeleteEdge(eid) => {
                if let Some((u, v)) = store.edge_endpoints(eid) {
                    store.delete_edge(u, v, eid);
                }
            }
            Applied::DeleteNode(node, detach) => node_deletes.push((node, detach)),
        }
    }
    // Pass 2: node deletes. A non-DETACH delete of a node that still has
    // relationships is an error (Cypher/core semantics); DETACH deletes the
    // incident edges too (delete_node cascades). A node matched by several
    // rows is deleted once (skip if already gone).
    for (node, detach) in node_deletes {
        if !store.is_alive(node) {
            continue;
        }
        if !detach && (!store.out(node).is_empty() || !store.inc(node).is_empty()) {
            stmt_rollback(store, scope);
            return Err(
                "E_INVALID_GRAPH_OP: cannot DELETE a node that still has relationships; \
                         use DETACH DELETE"
                    .into(),
            );
        }
        store.delete_node(node);
    }
    // Declared-constraint checks: NOW if standalone, else deferred to the
    // enclosing transaction's COMMIT. Roll back on the first violation.
    if let Err(e) = check_deferred_if_standalone(store, scope) {
        stmt_rollback(store, scope);
        return Err(e);
    }
    stmt_commit(store, scope);
    Ok(frontier)
}

/// Execute a `_MERGE`: infer the key from a unique constraint, find the existing
/// node by its key values, and take the create or update path. Runs in a
/// transaction so a constraint violation (or a no-applicable-constraint error)
/// leaves the store untouched.
pub(super) fn execute_merge(
    store: &mut Store,
    label: &str,
    props: &[(String, Value)],
    on_create: &[(String, Expr)],
    on_update: &crate::ir::MergeUpdate,
) -> Result<Rows, String> {
    use crate::ir::MergeUpdate;
    let scope = stmt_begin(store);
    let have: Vec<String> = props.iter().map(|(k, _)| k.clone()).collect();
    let key_keys = match store.infer_merge_key(label, &have) {
        Ok(k) => k,
        Err(e) => {
            stmt_rollback(store, scope);
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
                        stmt_rollback(store, scope);
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
                            stmt_rollback(store, scope);
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
                    stmt_rollback(store, scope);
                    return Err(e);
                }
            }
        }
    }

    if let Err(e) = check_deferred_if_standalone(store, scope) {
        stmt_rollback(store, scope);
        return Err(e);
    }
    stmt_commit(store, scope);
    Ok(empty_rows())
}

/// Resolve a `_MERGE` edge endpoint: the vertex whose inferred unique key matches
/// the pattern's `props`. A missing key or no match is an error (mirrors the TS
/// engine's `resolveMergeEndpoint`).
fn resolve_merge_endpoint(
    store: &Store,
    label: &str,
    props: &[(String, Value)],
) -> Result<u32, String> {
    let have: Vec<String> = props.iter().map(|(k, _)| k.clone()).collect();
    let key_keys = store.infer_merge_key(label, &have)?;
    let want = key_bytes(&key_keys, |k| pattern_value(props, k));
    store
        .nodes_with_label(label)
        .iter()
        .copied()
        .find(|&id| key_bytes(&key_keys, |k| store.prop(id, k)) == want)
        .ok_or_else(|| {
            format!(
                "E_INVALID_GRAPH_OP: _MERGE endpoint (:{label} {{…}}) not found — its key must match an existing vertex"
            )
        })
}

/// Execute a `_MERGE` edge upsert: resolve both endpoints by their unique key,
/// then upsert the single edge between them keyed by (from, to, `etype`).
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_merge_edge(
    store: &mut Store,
    start_label: &str,
    start_props: &[(String, Value)],
    end_label: &str,
    end_props: &[(String, Value)],
    dir: Dir,
    etype: &str,
    edge_props: &[(String, Value)],
    on_create: &[(usize, String, Expr)],
    on_update: &crate::ir::MergeEdgeUpdate,
) -> Result<Rows, String> {
    use crate::ir::MergeEdgeUpdate;
    let scope = stmt_begin(store);

    let start_id = match resolve_merge_endpoint(store, start_label, start_props) {
        Ok(id) => id,
        Err(e) => {
            stmt_rollback(store, scope);
            return Err(e);
        }
    };
    let end_id = match resolve_merge_endpoint(store, end_label, end_props) {
        Ok(id) => id,
        Err(e) => {
            stmt_rollback(store, scope);
            return Err(e);
        }
    };
    // The pattern positions (start=slot 0, end=slot 1) are fixed; direction only
    // decides which is the edge's tail vs head.
    let (from, to) = match dir {
        Dir::In => (end_id, start_id),
        _ => (start_id, end_id),
    };

    let existing = store
        .out(from)
        .iter()
        .find(|a| a.nbr == to && store.edge_type_name(a.eid).as_deref() == Some(etype))
        .map(|a| a.eid);

    let (eid, created) = match existing {
        Some(e) => (e, false),
        None => {
            let e = store.add_edge(from, to, etype);
            for (k, v) in edge_props {
                store.set_edge_prop(e, k, v.clone());
            }
            (e, true)
        }
    };

    // slot 0 = start node, slot 1 = end node, slot 2 = edge — the disposition
    // expressions read and write through this batch.
    let batch = Batch::of(vec![
        Col::Nodes(vec![start_id]),
        Col::Nodes(vec![end_id]),
        Col::Edges(vec![eid]),
    ]);

    // Evaluate every assignment BEFORE mutating, so a fault rolls the whole MERGE
    // back rather than leaving a partial write.
    let eval_writes = |store: &Store,
                       assigns: &[(usize, String, Expr)]|
     -> Result<Vec<(usize, String, Value)>, String> {
        assigns
            .iter()
            .map(|(slot, k, e)| Ok((*slot, k.clone(), eval(e, store, &batch)?.value_at(0))))
            .collect()
    };
    let apply = |store: &mut Store, writes: Vec<(usize, String, Value)>| {
        for (slot, k, v) in writes {
            match slot {
                0 => store.set_prop(start_id, &k, v),
                1 => store.set_prop(end_id, &k, v),
                _ => store.set_edge_prop(eid, &k, v),
            }
        }
    };

    if created {
        match eval_writes(store, on_create) {
            Ok(w) => apply(store, w),
            Err(e) => {
                stmt_rollback(store, scope);
                return Err(e);
            }
        }
    } else {
        match on_update {
            MergeEdgeUpdate::Nothing => {}
            MergeEdgeUpdate::Clobber => {
                for (k, v) in edge_props {
                    store.set_edge_prop(eid, k, v.clone());
                }
            }
            MergeEdgeUpdate::Set { assigns, filter } => {
                let gate = match filter.as_ref().map(|f| eval(f, store, &batch)).transpose() {
                    Ok(g) => g.is_none_or(|c| matches!(c.value_at(0), Value::Bool(true))),
                    Err(e) => {
                        stmt_rollback(store, scope);
                        return Err(e);
                    }
                };
                if gate {
                    match eval_writes(store, assigns) {
                        Ok(w) => apply(store, w),
                        Err(e) => {
                            stmt_rollback(store, scope);
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    if let Err(e) = check_deferred_if_standalone(store, scope) {
        stmt_rollback(store, scope);
        return Err(e);
    }
    stmt_commit(store, scope);
    Ok(empty_rows())
}
