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
use crate::ir::{Agg, AggFn, CombineOp, CompareOp, Dir, Expr, PathMode, Plan};
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
/// Result cells in a FLAT row-major buffer (`data[i*ncols + j]`) — one allocation
/// for the whole result instead of a `Vec` per row. The nested `Vec<Vec<Value>>`
/// layout measured ~4x slower to build (a malloc per row), and this matches core's
/// `RowSet`. It still indexes and iterates like the old nested layout —
/// `flat[i]` / `flat[i][j]` yield a row slice / cell, `flat.len()` the row count,
/// `flat.iter()` (and `&flat`) yield `&[Value]` rows — so read sites are unchanged;
/// only construction goes through [`Flat::from_rows`] or the direct push in `run`.
#[derive(Debug, Clone, Default)]
pub struct Flat {
    data: Vec<Value>,
    ncols: usize,
}

impl Flat {
    fn with_capacity(nrows: usize, ncols: usize) -> Self {
        Self {
            data: Vec::with_capacity(nrows.saturating_mul(ncols)),
            ncols,
        }
    }
    /// The number of rows (`data.len() / ncols`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len().checked_div(self.ncols).unwrap_or(0)
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Iterate rows as `&[Value]` slices — the drop-in for the old `Vec::iter`.
    pub fn iter(&self) -> std::slice::Chunks<'_, Value> {
        self.data.chunks(self.ncols.max(1))
    }
    /// Build from the nested layout — the construction path for tests and callers
    /// that already hold a `Vec<Vec<Value>>`.
    #[must_use]
    pub fn from_rows(rows: Vec<Vec<Value>>) -> Self {
        let ncols = rows.first().map_or(0, Vec::len);
        let mut data = Vec::with_capacity(rows.len().saturating_mul(ncols));
        for r in rows {
            data.extend(r);
        }
        Self { data, ncols }
    }
}

impl std::ops::Index<usize> for Flat {
    type Output = [Value];
    fn index(&self, i: usize) -> &[Value] {
        let c = self.ncols.max(1);
        &self.data[i * c..i * c + c]
    }
}

impl<'a> IntoIterator for &'a Flat {
    type Item = &'a [Value];
    type IntoIter = std::slice::Chunks<'a, Value>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug)]
pub struct Rows {
    pub names: Vec<String>,
    pub rows: Flat,
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
            // One flat allocation, row-major: for each row, push every slot's cell.
            let ncols = names.len();
            let mut rows = Flat::with_capacity(n, ncols);
            for i in 0..n {
                for c in &batch.slots {
                    rows.data.push(render_cell(c, i, store));
                }
            }
            Rows { names, rows }
        }
        None => {
            let slot0 = batch.slot(0);
            let mut rows = Flat::with_capacity(n, 1);
            for i in 0..n {
                rows.data.push(render_cell(slot0, i, store));
            }
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
/// Run an `Insert`'s writes (nodes then edges) inside a transaction, enforcing
/// unique + required constraints on every touched label and rolling the whole
/// statement back on the first violation. Returns the ids of the created nodes,
/// in creation order (index i is the node declared at position i). Shared by
/// `Plan::Insert` and `Plan::InsertReturn`.
fn run_insert(
    store: &mut Store,
    nodes: &[crate::ir::InsertNode],
    edges: &[crate::ir::InsertEdge],
) -> Result<Vec<u32>, String> {
    // In a transaction so a constraint violation rolls the whole INSERT back
    // rather than leaving a partial write.
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
    // Enforce unique AND required constraints on every label this INSERT touched
    // (roll the whole INSERT back on the first violation).
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
    Ok(ids)
}

pub fn execute(plan: &Plan, store: &mut Store) -> Result<Rows, String> {
    match plan {
        Plan::Insert { nodes, edges } => {
            run_insert(store, nodes, edges)?;
            Ok(empty_rows())
        }
        Plan::InsertReturn { nodes, edges, tail } => {
            // First write-then-return path: run the INSERT, then bind each created
            // node into the slot equal to its creation index and project the tail.
            let ids = run_insert(store, nodes, edges)?;
            // A one-row seed: slot i carries the id of the i-th created node, so the
            // tail's `Expr::Prop{slot}` reads the node just created at that index.
            let seed = Batch::of(ids.iter().map(|&id| Col::Nodes(vec![id])).collect());
            // The tail is restricted (by the parser + this guard) to pure
            // projections; `pull_body` covers Row/Project/Filter, not the read
            // pipeline's grouping/paging operators.
            let store_ref: &Store = store;
            let batch = pull_body(tail, store_ref, &seed)?;
            let n = batch.rows();
            Ok(match output_names(tail) {
                Some(names) => {
                    let ncols = names.len();
                    let mut rows = Flat::with_capacity(n, ncols);
                    for i in 0..n {
                        for c in &batch.slots {
                            rows.data.push(render_cell(c, i, store_ref));
                        }
                    }
                    Rows { names, rows }
                }
                None => {
                    let slot0 = batch.slot(0);
                    let mut rows = Flat::with_capacity(n, 1);
                    for i in 0..n {
                        rows.data.push(render_cell(slot0, i, store_ref));
                    }
                    Rows {
                        names: vec!["_".to_string()],
                        rows,
                    }
                }
            })
        }
        Plan::Update { input, ops } => {
            // Read phase: run the match and compute every write into OWNED data —
            // so the immutable borrow ends before the write phase mutates. A slot
            // may be a node frontier or (bound relationship) an edge frontier;
            // SET/REMOVE dispatch on which, so `r.weight` writes an edge property.
            enum Applied {
                Set(u32, String, Value),
                Remove(u32, String),
                DeleteNode(u32, bool), // (node, detach)
                DeleteEdge(u32),       // eid
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
            store.begin();
            // Nodes whose properties changed (Set/Remove) — the ones that can newly
            // violate a unique/required constraint. A Delete can't create a violation
            // on another node, so it doesn't seed a recheck.
            let touched: Vec<u32> = applied
                .iter()
                .filter_map(|a| match a {
                    Applied::Set(n, _, _) | Applied::Remove(n, _) => Some(*n),
                    _ => None,
                })
                .collect();
            // Pass 1: property writes and EDGE deletes. Node deletes are deferred to
            // pass 2 so an edge deleted here (`DELETE r, a, b`) leaves its endpoints
            // relationship-free before the non-DETACH node-delete check runs.
            let mut node_deletes: Vec<(u32, bool)> = Vec::new();
            for a in applied {
                match a {
                    Applied::Set(node, key, value) => store.set_prop(node, &key, value),
                    Applied::Remove(node, key) => store.remove_prop(node, &key),
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
                    store.rollback();
                    return Err(
                        "E_INVALID_GRAPH_OP: cannot DELETE a node that still has relationships; \
                         use DETACH DELETE"
                            .into(),
                    );
                }
                store.delete_node(node);
            }
            // Recheck unique + required on every label a touched (still-live) node
            // carries; roll the statement back on the first violation.
            let mut labels: Vec<String> = touched
                .iter()
                .filter(|&&n| store.is_alive(n))
                .flat_map(|&n| store.labels_of(n))
                .collect();
            labels.sort_unstable();
            labels.dedup();
            for l in &labels {
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

/// Render a batch cell (slot `col`, row `i`) to a result `Value`. A NODE frontier
/// slot renders as core's element MAP `{id, labels, properties}` (not its bare id),
/// so `RETURN n` / `RETURN *` match core byte-for-byte. Everything else materializes
/// as its plain value. (Edge frontier rendering — `{id, from, to, labels,
/// properties}` — needs an eid→endpoints accessor and is a separate step.)
fn render_cell(col: &Col, i: usize, store: &Store) -> Value {
    match col {
        // `u32::MAX` is the OPTIONAL-MATCH null sentinel → NULL, not an element map.
        Col::Nodes(ids) if ids[i] == u32::MAX => Value::Null,
        Col::Edges(eids) if eids[i] == u32::MAX => Value::Null,
        Col::Nodes(ids) => node_result_value(store, ids[i]),
        Col::Edges(eids) => edge_result_value(store, eids[i]),
        _ => col.value_at(i),
    }
}

/// The canonical result map for an edge — `{id, from, to, labels(sorted),
/// properties(sorted by key)}`, byte-identical to lenke-core's `val_to_value(Edge)`.
/// `from`/`to` are the endpoint EXTERNAL ids.
fn edge_result_value(store: &Store, eid: u32) -> Value {
    use std::sync::Arc;
    let id = store
        .edge_ext_id(eid)
        .unwrap_or_else(|| Arc::from(format!("e{eid}")));
    let (src, dst) = store.edge_endpoints(eid).unwrap_or((0, 0));
    let ext = |n: u32| {
        store
            .node_ext_id(n)
            .unwrap_or_else(|| Arc::from(n.to_string()))
    };
    // Single edge type here → a one-element (trivially sorted) labels list.
    let labels = Value::List(
        store
            .edge_type_name(eid)
            .into_iter()
            .map(|t| Value::Str(t.into()))
            .collect(),
    );
    let mut props: Vec<(String, Value)> = store
        .edge_prop_keys()
        .into_iter()
        .filter(|k| store.has_edge_prop(eid, k))
        .map(|k| {
            let v = store.edge_prop(eid, &k);
            (k, v)
        })
        .collect();
    props.sort_by(|a, b| a.0.cmp(&b.0));
    let props_map = Value::Map(Arc::new(
        props
            .into_iter()
            .map(|(k, v)| (Value::Str(k.into()), v))
            .collect(),
    ));
    Value::Map(Arc::new(vec![
        (Value::Str("id".into()), Value::Str(id)),
        (Value::Str("from".into()), Value::Str(ext(src))),
        (Value::Str("to".into()), Value::Str(ext(dst))),
        (Value::Str("labels".into()), labels),
        (Value::Str("properties".into()), props_map),
    ]))
}

/// The canonical result map for a node — `{id, labels(sorted), properties(sorted by
/// key)}`, byte-identical to lenke-core's `val_to_value(Node)`.
fn node_result_value(store: &Store, id: u32) -> Value {
    use std::sync::Arc;
    let ext = store
        .node_ext_id(id)
        .unwrap_or_else(|| Arc::from(id.to_string()));
    let mut labels = store.labels_of(id);
    labels.sort_unstable();
    let labels_list = Value::List(labels.into_iter().map(|l| Value::Str(l.into())).collect());
    // Present properties on this node, sorted by key (core's props_map ordering).
    let mut props: Vec<(String, Value)> = store
        .prop_keys()
        .into_iter()
        .filter(|k| store.has_prop(id, k))
        .map(|k| {
            let v = store.prop(id, &k);
            (k, v)
        })
        .collect();
    props.sort_by(|a, b| a.0.cmp(&b.0));
    let props_map = Value::Map(Arc::new(
        props
            .into_iter()
            .map(|(k, v)| (Value::Str(k.into()), v))
            .collect(),
    ));
    Value::Map(Arc::new(vec![
        (Value::Str("id".into()), Value::Str(ext)),
        (Value::Str("labels".into()), labels_list),
        (Value::Str("properties".into()), props_map),
    ]))
}

/// The empty result a write statement returns (no columns, no rows).
fn empty_rows() -> Rows {
    Rows {
        names: Vec::new(),
        rows: Flat::default(),
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
        // UNION names come from the LEFT arm (core's rule).
        Plan::Union { left, .. } => output_names(left),
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
            Expr::Compare { left, right, .. }
            | Expr::In {
                needle: left,
                haystack: right,
            } => reads_path(left) || reads_path(right),
            Expr::Not(x) => reads_path(x),
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Xor(a, b)
            | Expr::Arith {
                left: a, right: b, ..
            } => reads_path(a) || reads_path(b),
            Expr::Call { args, .. } | Expr::List { items: args } => args.iter().any(reads_path),
            Expr::Record { fields } | Expr::MapLit { entries: fields } => {
                fields.iter().any(|(_, e)| reads_path(e))
            }
            Expr::Field { base, .. } => reads_path(base),
            Expr::Index { base, index } => reads_path(base) || reads_path(index),
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
            | Expr::Exists { .. }
            | Expr::CountSubquery { .. } => false,
        }
    }
    match plan {
        Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. } => false,
        Plan::Expand { input, .. }
        | Plan::OptionalExpand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Distinct { input }
        | Plan::Tail { input, .. }
        | Plan::SortLocal { input, .. } => needs_lineage(input),
        Plan::Branch { input, bodies } => needs_lineage(input) || bodies.iter().any(needs_lineage),
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
        Plan::Join { left, right, .. } | Plan::Union { left, right, .. } => {
            needs_lineage(left) || needs_lineage(right)
        }
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
        Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::Update { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. } => Batch::of(Vec::new()),
        // `Row` is the leaf of an EXISTS body and is only ever fed a batch by
        // `pull_body`; reaching it through the main pipeline is a bug.
        Plan::Row => {
            // ONE unit row (a single dummy cell so `rows()` == 1) — the input to a
            // bare `RETURN <items>` with no MATCH. A row with no bound variables; the
            // projected items reference no slots. (Inside an EXISTS body, `Plan::Row`
            // is seeded by `pull_body`, not this path.)
            Batch::single(Col::Num(vec![0.0]))
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
        Plan::NodeSeed { ext_ids } => {
            // Resolve each external id to a LIVE node; an unknown/deleted id is
            // silently dropped (Gremlin `g.V(<missing>)` yields nothing for it).
            let ids: Vec<u32> = ext_ids
                .iter()
                .filter_map(|e| store.node_by_ext(e).filter(|&id| store.is_alive(id)))
                .collect();
            let mut batch = Batch::single(Col::Nodes(ids.clone()));
            if track {
                batch.lineage = Some(Lineage::seed(&ids));
            }
            batch
        }
        Plan::EdgeScan => {
            // The frontier is EDGES, not nodes. `track` is never set for a bare
            // g.E() read (no path()/lineage step targets an edge frontier yet), so
            // no lineage is seeded here — a path over g.E() is a later item.
            Batch::single(Col::Edges(store.all_edges()))
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
            edge_label,
            *bind_edge,
        ),
        Plan::OptionalExpand {
            input,
            from,
            dir,
            edge_label,
            keep_source,
        } => optional_expand(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *keep_source,
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
                &batch, store, *from, *dir, edge_label, lo_key, hi_key, &qlo_col, &qhi_col,
                *bind_edge,
            )
        }
        Plan::Filter { input, pred } => {
            // Anchor flip: a selective indexed `=` on the traversal TARGET is far
            // cheaper to seed-and-walk-in-reverse than to scan every source and
            // filter. Same slot layout, multiset-preserving.
            if let Some(b) = try_reverse_expand(pred, input, store, track) {
                return Ok(b);
            }
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
            mode,
        } => var_length(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *min,
            *max,
            *mode,
        ),
        Plan::ShortestPath {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            selector,
        } => shortest_path(
            &pull(input, store, track)?,
            store,
            *from,
            *dir,
            edge_label,
            *min,
            *max,
            *selector,
        ),
        Plan::Aggregate { input, keys, aggs } => {
            // Frontier fast path: a scalar count over an Expand chain need not
            // build the wide intermediate batch. Falls back to the general
            // aggregate for every shape it does not recognize. (The fused paths
            // never evaluate arbitrary expressions, so they cannot fault.)
            if let Some(b) = try_scan_count(input, keys, aggs, store)
                .or_else(|| try_filtered_count(input, keys, aggs, store))
                .or_else(|| try_edge_filtered_count(input, keys, aggs, store))
                .or_else(|| try_varlen_count(input, keys, aggs, store))
                .or_else(|| try_varlen_distinct_count(input, keys, aggs, store))
                .or_else(|| try_varlen_agg(input, keys, aggs, store))
                .or_else(|| try_scan_num_agg(input, keys, aggs, store))
                .or_else(|| try_scan_multi_agg(input, keys, aggs, store))
                .or_else(|| try_scan_distinct_count(input, keys, aggs, store))
                .or_else(|| try_3hop_product_count(input, keys, aggs, store))
                .or_else(|| try_fused_count(input, keys, aggs, store))
                .or_else(|| try_node_grouped_count(input, keys, aggs, store))
                .or_else(|| try_scan_group_agg(input, keys, aggs, store))
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
        } if *limit == Some(0) => {
            // LIMIT 0 keeps no rows, so the input's projection is never evaluated —
            // a faulting expression (`RETURN 1/0 AS x LIMIT 0`) must yield the empty
            // result, not the fault. Short-circuit without pulling the input. One
            // empty slot keeps the unnamed-output path (`batch.slot(0)`) valid; an
            // empty result carries no column identity anyway.
            let _ = (input, keys, skip);
            Batch::of(vec![Col::Nodes(vec![])])
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } => {
            // A keyless page (LIMIT/SKIP without ORDER BY) keeps the first
            // `skip+limit` rows in scan order — so cap the input at that many rows
            // instead of materializing the whole scan and slicing. Only safe for a
            // row-preserving Scan/Seek/Project chain (a Filter/Expand would need
            // MORE rows to still yield `limit`), which `pull_capped` recognizes.
            let cap = limit.map(|l| skip.unwrap_or(0).saturating_add(l));
            let capped = match cap {
                Some(c) if keys.is_empty() => match pull_capped(input, store, track, c)? {
                    Some(b) => Some(b),
                    None if track => None,
                    // The row-preserving cap didn't apply (a Filter/Expand/VarLength in
                    // the chain). Stream the chain block-by-block until `c` rows land —
                    // identical rows, computed early. A `DISTINCT … LIMIT` streams with
                    // incremental dedup, stopping at `c` distinct rows.
                    None => match input.as_ref() {
                        Plan::Distinct { input: inner } => {
                            pull_distinct_capped_stream(inner, store, c)?
                        }
                        _ => pull_capped_stream(input, store, c)?,
                    },
                },
                _ => None,
            };
            if let Some(b) = capped {
                order_page(&b, store, keys, *skip, *limit)?
            } else if let Some(b) = try_late_materialize(input, keys, *skip, *limit, store, track)?
            {
                // Sorted top-K over a projection: project only the surviving rows.
                b
            } else {
                order_page(&pull(input, store, track)?, store, keys, *skip, *limit)?
            }
        }
        Plan::Tail { input, n } => {
            // The last `n` rows in input order (Gremlin tail): gather the tail window,
            // computing its start from the materialized row count. `gather` carries
            // the slots AND the lineage sidecar, so a path survives the trim.
            let b = pull(input, store, track)?;
            let rows = b.rows();
            let start = rows.saturating_sub(*n);
            b.gather(&(start..rows).collect::<Vec<usize>>())
        }
        Plan::Branch { input, bodies } => {
            // Gremlin union: run every branch body over the SAME input frontier (each
            // is Row-rooted, correlating on the current slot) and concatenate their
            // sub-rows. Every branch lands its element at the same slot, so the
            // concatenated column keeps its node/edge type — a continuable frontier.
            let inb = pull(input, store, track)?;
            let subs: Vec<Batch> = bodies
                .iter()
                .map(|b| pull_body(b, store, &inb))
                .collect::<Result<_, _>>()?;
            concat_batches(&subs)
        }
        Plan::Project { input, items } => {
            // Project produces a batch whose slots ARE the projected columns, so
            // an operator above it (Distinct, OrderPage) works on the output
            // values, not the pre-projection bindings.
            let batch = pull(input, store, track)?;
            let cols = eval_all(items.iter().map(|(_, e)| e), store, &batch)?;
            Batch::of(cols)
        }
        Plan::Union {
            left,
            right,
            all,
            op,
        } => {
            // Run both arms, materialize each row (render_cell → nodes/edges as maps),
            // pad to the LEFT arm's width. UNION concatenates (deduped unless ALL);
            // EXCEPT keeps left rows absent from the right; INTERSECT keeps left rows
            // present in the right (both deduped). Column names come from the left arm.
            let bl = pull(left, store, track)?;
            let br = pull(right, store, track)?;
            let ncols = bl.slots.len();
            // Fast path: UNION ALL of same-width arms concatenates COLUMN-wise.
            if matches!(op, CombineOp::Union) && *all && br.slots.len() == ncols {
                return Ok(concat_batches(&[bl, br]));
            }
            let row_of = |b: &Batch, i: usize| -> Vec<Value> {
                let mut row: Vec<Value> =
                    b.slots.iter().map(|c| render_cell(c, i, store)).collect();
                row.resize(ncols, Value::Null);
                row
            };
            let key_of = |row: &[Value]| -> Vec<u8> {
                let mut buf = Vec::new();
                for v in row {
                    value::group_key_into(v, &mut buf);
                }
                buf
            };
            let rows: Vec<Vec<Value>> = match op {
                CombineOp::Union => {
                    let mut rows: Vec<Vec<Value>> = Vec::with_capacity(bl.rows() + br.rows());
                    for b in [&bl, &br] {
                        for i in 0..b.rows() {
                            rows.push(row_of(b, i));
                        }
                    }
                    if !*all {
                        let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
                        rows.retain(|row| seen.insert(key_of(row)));
                    }
                    rows
                }
                CombineOp::Except | CombineOp::Intersect => {
                    // The right arm's key set; keep a LEFT row iff its key is absent
                    // (EXCEPT) or present (INTERSECT). Always deduped.
                    let mut right_keys: FnvSet<Vec<u8>> = FnvSet::default();
                    for i in 0..br.rows() {
                        right_keys.insert(key_of(&row_of(&br, i)));
                    }
                    let want_present = matches!(op, CombineOp::Intersect);
                    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
                    let mut rows = Vec::new();
                    for i in 0..bl.rows() {
                        let row = row_of(&bl, i);
                        let k = key_of(&row);
                        if right_keys.contains(&k) == want_present && seen.insert(k) {
                            rows.push(row);
                        }
                    }
                    rows
                }
            };
            let mut cols: Vec<Vec<Value>> = vec![Vec::with_capacity(rows.len()); ncols.max(1)];
            for row in rows {
                for (j, v) in row.into_iter().enumerate() {
                    cols[j].push(v);
                }
            }
            Batch::of(cols.into_iter().map(Col::Gen).collect())
        }
        Plan::Distinct { input } => {
            // Fused `DISTINCT n.k` over a bare Scan: read the storage column and
            // dedup in one pass, emitting ONLY the distinct values — never
            // materializing the 100k-row projected column.
            if let Some(b) = try_distinct_scan_prop(input, store) {
                return Ok(b);
            }
            // The multi-column sibling: dedup several storage columns on a composite
            // key without materializing any of them (the `Arc<str>` dept column above
            // all). Single-column shapes are already handled above; this catches
            // `DISTINCT n.a, n.b, …`.
            if let Some(b) = try_distinct_scan_multi(input, store) {
                return Ok(b);
            }
            let batch = pull(input, store, track)?;
            // Typed single-column fast path: dedup by the raw value (a `&str`, the
            // f64 group bits, or a dense id) — no per-row byte-key serialization.
            if let Some(keep) = try_distinct_typed(&batch) {
                return Ok(batch.gather(&keep));
            }
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
            // The result column carries per-node Values (a scalar Num for most
            // procedures, a List for neighbor_aggregate's feature vectors).
            let vals: Vec<Value> = results.into_iter().map(|(_, r)| r).collect();
            Batch::of(vec![Col::Nodes(ids), Col::Gen(vals)])
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

/// Pull at most `cap` rows from a ROW-PRESERVING chain (Scan / IndexSeek /
/// RangeSeek / Project over one), so a keyless `LIMIT` need not materialize the
/// whole scan. `Ok(None)` when the input is not cap-safe (a Filter/Expand/Distinct
/// changes the row count, so an input cap could under-produce) — the caller then
/// does the full pull. Faults propagate as `Err`.
fn pull_capped(
    plan: &Plan,
    store: &Store,
    track: bool,
    cap: usize,
) -> Result<Option<Batch>, String> {
    Ok(match plan {
        Plan::Scan { label } => {
            let ids: Vec<u32> = match label {
                Some(l) => store
                    .nodes_with_label(l)
                    .iter()
                    .copied()
                    .take(cap)
                    .collect(),
                None => (0..store.node_count() as u32)
                    .filter(|&i| store.is_alive(i))
                    .take(cap)
                    .collect(),
            };
            let mut b = Batch::single(Col::Nodes(ids.clone()));
            if track {
                b.lineage = Some(Lineage::seed(&ids));
            }
            Some(b)
        }
        Plan::IndexSeek { label, key, value } => {
            let ids: Vec<u32> = index_seek_ids(store, label, key, value)
                .into_iter()
                .take(cap)
                .collect();
            let mut b = Batch::single(Col::Nodes(ids.clone()));
            if track {
                b.lineage = Some(Lineage::seed(&ids));
            }
            Some(b)
        }
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => {
            let ids: Vec<u32> = range_seek_ids(store, label, key, *op, value)
                .into_iter()
                .take(cap)
                .collect();
            let mut b = Batch::single(Col::Nodes(ids.clone()));
            if track {
                b.lineage = Some(Lineage::seed(&ids));
            }
            Some(b)
        }
        Plan::Project { input, items } => match pull_capped(input, store, track, cap)? {
            Some(batch) => {
                let cols = eval_all(items.iter().map(|(_, e)| e), store, &batch)?;
                let mut out = Batch::of(cols);
                out.lineage = batch.lineage;
                Some(out)
            }
            None => None,
        },
        _ => None, // Filter/Expand/Aggregate/Distinct/… change the row count
    })
}

/// If `plan` is a STREAMABLE chain — Project/Filter/Expand/VarLength over a
/// chunkable leaf (Scan/IndexSeek/RangeSeek) — return the chain with the leaf
/// replaced by `Plan::Row` (so it runs from a seeded frontier) plus the leaf's full
/// id list. `None` for any operator the row-by-row `pull_body` cannot stream
/// (Aggregate/Distinct/Join/OrderPage/OptionalExpand/…).
fn streaming_chain(plan: &Plan, store: &Store) -> Option<(Plan, Vec<u32>)> {
    match plan {
        Plan::Scan { label } => {
            let ids: Vec<u32> = match label {
                Some(l) => store.nodes_with_label(l).to_vec(),
                None => (0..store.node_count() as u32)
                    .filter(|&i| store.is_alive(i))
                    .collect(),
            };
            Some((Plan::Row, ids))
        }
        Plan::IndexSeek { label, key, value } => {
            Some((Plan::Row, index_seek_ids(store, label, key, value)))
        }
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => Some((Plan::Row, range_seek_ids(store, label, key, *op, value))),
        Plan::Filter { input, pred } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::Filter {
                    input: Box::new(body),
                    pred: pred.clone(),
                },
                ids,
            ))
        }
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge,
        } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::Expand {
                    input: Box::new(body),
                    from: *from,
                    dir: *dir,
                    edge_label: edge_label.clone(),
                    bind_edge: *bind_edge,
                },
                ids,
            ))
        }
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
        } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::VarLength {
                    input: Box::new(body),
                    from: *from,
                    dir: *dir,
                    edge_label: edge_label.clone(),
                    min: *min,
                    max: *max,
                    mode: *mode,
                },
                ids,
            ))
        }
        Plan::Project { input, items } => {
            let (body, ids) = streaming_chain(input, store)?;
            Some((
                Plan::Project {
                    input: Box::new(body),
                    items: items.clone(),
                },
                ids,
            ))
        }
        _ => None,
    }
}

/// Short-circuit a keyless `LIMIT`/`SKIP` (no `ORDER BY`) over a streamable chain
/// that `pull_capped` can't cap because it filters/expands: run the chain over
/// successive BLOCKS of the source, stopping once `cap` rows have accumulated. The
/// blocks are taken in source-id order and concatenated in order, and per-block
/// operators are the same row-wise ones — so the accumulated rows are IDENTICAL
/// (same order) to materializing the whole input and slicing, just computed early.
/// Only for `!track` (a path-reading LIMIT keeps the full path via the slow path).
fn pull_capped_stream(plan: &Plan, store: &Store, cap: usize) -> Result<Option<Batch>, String> {
    if cap == 0 {
        return Ok(None); // LIMIT 0 handled by the general path (empty, right width)
    }
    let Some((body, ids)) = streaming_chain(plan, store) else {
        return Ok(None);
    };
    if ids.is_empty() {
        return Ok(None); // empty source → let the full path build the right shape
    }
    // ADAPTIVE block size, starting at 1 and doubling. A high-fan-out chain (double
    // var-length) makes even one source overshoot `cap`, so the first block must be
    // tiny — else a fixed block materializes thousands of rows per source. A
    // selective filter / low fan-out grows the block geometrically, so the overhead
    // stays logarithmic. This mirrors a lazy engine producing just past `cap`.
    let mut acc: Vec<Batch> = Vec::new();
    let mut total = 0usize;
    let mut start = 0usize;
    let mut block = 1usize;
    while start < ids.len() && total < cap {
        let end = (start + block).min(ids.len());
        let seed = Batch::single(Col::Nodes(ids[start..end].to_vec()));
        let b = pull_body(&body, store, &seed)?;
        total += b.rows();
        acc.push(b);
        start = end;
        block = block.saturating_mul(2).min(8192);
    }
    Ok(Some(concat_batches(&acc)))
}

/// Streaming `DISTINCT … LIMIT k` over a streamable chain: dedup incrementally
/// (the same whole-row grouping key as `Plan::Distinct`) while streaming source
/// blocks, stopping once `cap` DISTINCT rows are collected. First-occurrence order
/// is preserved (blocks in source-id order), so the result matches a full
/// distinct-then-slice. Lets "give me N distinct X" short-circuit instead of
/// materializing every reachable row before deduping.
fn pull_distinct_capped_stream(
    inner: &Plan,
    store: &Store,
    cap: usize,
) -> Result<Option<Batch>, String> {
    if cap == 0 {
        return Ok(None);
    }
    let Some((body, ids)) = streaming_chain(inner, store) else {
        return Ok(None);
    };
    if ids.is_empty() {
        return Ok(None);
    }
    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
    let mut buf = Vec::new();
    let mut acc: Vec<Batch> = Vec::new();
    let mut distinct = 0usize;
    let mut start = 0usize;
    let mut block = 1usize;
    while start < ids.len() && distinct < cap {
        let end = (start + block).min(ids.len());
        let b = pull_body(
            &body,
            store,
            &Batch::single(Col::Nodes(ids[start..end].to_vec())),
        )?;
        let mut keep = Vec::new();
        for i in 0..b.rows() {
            buf.clear();
            for c in &b.slots {
                value::group_key_into(&c.value_at(i), &mut buf);
            }
            if !seen.contains(buf.as_slice()) {
                seen.insert(buf.clone());
                keep.push(i);
                distinct += 1;
                if distinct >= cap {
                    break;
                }
            }
        }
        acc.push(b.gather(&keep));
        start = end;
        block = block.saturating_mul(2).min(8192);
    }
    Ok(Some(concat_batches(&acc)))
}

/// Compare two rows by the sort keys only (`Equal` on a full tie). NULL placement
/// is the front-end's language contract (GQL last, Gremlin first), decided here
/// BEFORE the total order (not by reversing `cmp_total` under DESC).
#[inline]
fn row_cmp(
    key_cols: &[Col],
    keys: &[crate::ir::SortKey],
    a: usize,
    b: usize,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    for (kc, k) in key_cols.iter().zip(keys) {
        let (va, vb) = (kc.value_at(a), kc.value_at(b));
        let ord = match (va.is_null(), vb.is_null()) {
            (true, true) => Ordering::Equal,
            (true, false) => {
                if k.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (false, true) => {
                if k.nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
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
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Order `idx` (initially `0..idx.len()`) by `keys` so that `idx[..end]` are the
/// `end` rows the sort window needs, correctly ordered. When `end < n` only the
/// top-`end` are selected (partition + partial sort — O(n) + O(end log end)); the
/// full case keeps a stable sort (ties keep arrival order). A single numeric key
/// compares raw f64 via `cmp_num_total` (no per-comparison `value_at` boxing).
fn sort_idx(idx: &mut [usize], key_cols: &[Col], keys: &[crate::ir::SortKey], end: usize) {
    let n = idx.len();
    if keys.is_empty() {
        return;
    }
    if let [Col::Num(vals)] = key_cols {
        // Col::Num carries no nulls, so null placement is moot; an arrival-index
        // tiebreak makes the order total (deterministic, == the stable sort on ties).
        let desc = keys[0].descending;
        let cmp = |&a: &usize, &b: &usize| {
            let o = if desc {
                value::cmp_num_total(vals[b], vals[a])
            } else {
                value::cmp_num_total(vals[a], vals[b])
            };
            o.then(a.cmp(&b))
        };
        if end < n {
            idx.select_nth_unstable_by(end - 1, cmp);
            idx[..end].sort_unstable_by(cmp);
        } else {
            idx.sort_unstable_by(cmp);
        }
    } else if let [Col::Str(vals)] = key_cols {
        // A single string key: compare the `Arc<str>` cells directly (lexicographic,
        // == `cmp_total` for strings) instead of boxing each cell through `value_at`.
        // Col::Str carries no nulls; the arrival-index tiebreak keeps it total/stable.
        let desc = keys[0].descending;
        let cmp = |&a: &usize, &b: &usize| {
            let o = if desc {
                vals[b].as_ref().cmp(vals[a].as_ref())
            } else {
                vals[a].as_ref().cmp(vals[b].as_ref())
            };
            o.then(a.cmp(&b))
        };
        if end < n {
            idx.select_nth_unstable_by(end - 1, cmp);
            idx[..end].sort_unstable_by(cmp);
        } else {
            idx.sort_unstable_by(cmp);
        }
    } else if end < n {
        let total = |&a: &usize, &b: &usize| row_cmp(key_cols, keys, a, b).then(a.cmp(&b));
        idx.select_nth_unstable_by(end - 1, total);
        idx[..end].sort_unstable_by(total);
    } else {
        idx.sort_by(|&a, &b| row_cmp(key_cols, keys, a, b));
    }
}

/// LATE MATERIALIZATION for a sorted `LIMIT` over a `Project`: when the window is
/// a strict PREFIX of the rows (`skip+limit < n`) and every sort key is an output
/// alias (`Slot(i)` into the projection), evaluate ONLY the sort-key expressions
/// over the projection's input to find the top-K rows, then project the FULL item
/// list for just those K survivors — so the non-key columns (a `name` string per
/// row, say) are built for K rows, not all N. `Ok(None)` when the shape doesn't
/// fit (no limit, input not a Project, a key that isn't a projected alias, or the
/// window is the whole set so there is nothing to save).
fn try_late_materialize(
    input: &Plan,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
    store: &Store,
    track: bool,
) -> Result<Option<Batch>, String> {
    let Some(limit) = limit else { return Ok(None) };
    if keys.is_empty() {
        return Ok(None);
    }
    let Plan::Project {
        input: pinput,
        items,
    } = input
    else {
        return Ok(None);
    };
    // Every sort key must be an output alias `Slot(i)` — map it to that item's
    // expression, so it can be evaluated over the projection's INPUT.
    let key_exprs: Option<Vec<&Expr>> = keys
        .iter()
        .map(|k| match &k.expr {
            Expr::Slot(i) => items.get(*i).map(|(_, e)| e),
            _ => None,
        })
        .collect();
    let Some(key_exprs) = key_exprs else {
        return Ok(None);
    };

    let base = pull(pinput, store, track)?;
    let n = base.rows();
    let start = skip.unwrap_or(0).min(n);
    let end = start.saturating_add(limit).min(n);
    if end >= n {
        // The window is the whole set — a full projection is unavoidable; nothing
        // to late-materialize.
        return Ok(None);
    }
    if end <= start {
        return Ok(Some(Batch::of(
            items.iter().map(|_| Col::Nodes(vec![])).collect(),
        )));
    }

    // Sort by the key columns evaluated over the base, take the window's rows.
    let key_cols: Vec<Col> = key_exprs
        .iter()
        .map(|e| eval(e, store, &base))
        .collect::<Result<_, _>>()?;
    let mut idx: Vec<usize> = (0..n).collect();
    sort_idx(&mut idx, &key_cols, keys, end);
    let sub = base.gather(&idx[start..end]);

    // NOW project every item, but only over the K surviving rows.
    let cols = eval_all(items.iter().map(|(_, e)| e), store, &sub)?;
    let mut out = Batch::of(cols);
    out.lineage = sub.lineage;
    Ok(Some(out))
}

/// Sort the batch by `keys`, then keep the window `[skip, skip+limit)`. Reorders
/// every slot together, so bound variables stay row-aligned.
fn order_page(
    batch: &Batch,
    store: &Store,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
) -> Result<Batch, String> {
    let n = batch.rows();
    let start = skip.unwrap_or(0).min(n);
    let end = limit.map_or(n, |l| start.saturating_add(l).min(n));
    if end <= start {
        return Ok(batch.gather(&[]));
    }
    let mut idx: Vec<usize> = (0..n).collect();
    if !keys.is_empty() {
        let key_cols: Vec<Col> = eval_all(keys.iter().map(|k| &k.expr), store, batch)?;
        sort_idx(&mut idx, &key_cols, keys, end);
    }
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
        AggFn::Collect | AggFn::CollectList => {
            // Gather each group's values into a list, in row order (a preceding sort
            // carries through). `Collect` (Gremlin fold) KEEPS nulls; `CollectList`
            // (GQL collect_list) SKIPS them, matching core. An empty (or all-null,
            // for CollectList) group folds to the empty list.
            let skip_nulls = agg.func == AggFn::CollectList;
            let mut lists: Vec<Vec<Value>> = vec![Vec::new(); n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                let v = col.value_at(i);
                if skip_nulls && v.is_null() {
                    continue;
                }
                lists[g as usize].push(v);
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
        AggFn::StddevPop | AggFn::StddevSamp => {
            // One-pass moments per group: a present non-null value contributes as a
            // number (a non-numeric one as NaN, which propagates — matching core's
            // stddev over a non-numeric column). NULLs are skipped.
            let sample = agg.func == AggFn::StddevSamp;
            let mut sum = vec![0f64; n_groups];
            let mut sum_sq = vec![0f64; n_groups];
            let mut cnt = vec![0u64; n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                let v = col.value_at(i);
                if v.is_null() {
                    continue;
                }
                let x = value::num_of(&v).unwrap_or(f64::NAN);
                let g = g as usize;
                sum[g] += x;
                sum_sq[g] += x * x;
                cnt[g] += 1;
            }
            (0..n_groups)
                .map(|g| stddev_of(cnt[g], sum[g], sum_sq[g], sample))
                .collect()
        }
        AggFn::PercentileCont | AggFn::PercentileDisc => {
            // Ordered-set: gather each group's finite numeric values, sort, and take
            // the `frac`-th percentile (interpolated for cont, discrete for disc) —
            // replicated from core's `percentile`. Empty group → NULL.
            let cont = agg.func == AggFn::PercentileCont;
            let frac = agg.frac.unwrap_or(0.0);
            let mut per_group: Vec<Vec<f64>> = vec![Vec::new(); n_groups];
            for (i, &g) in group_of.iter().enumerate() {
                if let Some(x) = value::num_of(&col.value_at(i)) {
                    if x.is_finite() {
                        per_group[g as usize].push(x);
                    }
                }
            }
            per_group
                .into_iter()
                .map(|nums| percentile_of(nums, frac, cont))
                .collect()
        }
    }
}

/// The `frac`-th percentile of `nums` — interpolated (`cont`) or discrete (`disc`) —
/// replicated exactly from core's `percentile`. Empty input → NULL.
fn percentile_of(mut nums: Vec<f64>, frac: f64, cont: bool) -> Value {
    if nums.is_empty() {
        return Value::Null;
    }
    nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = nums.len();
    let result = if cont {
        let rn = frac * (n - 1) as f64;
        let lo = rn.floor() as usize;
        let hi = rn.ceil() as usize;
        if lo == hi {
            nums[lo]
        } else {
            nums[lo] + (rn - lo as f64) * (nums[hi] - nums[lo])
        }
    } else {
        let idx = ((frac * n as f64).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        nums[idx]
    };
    Value::Num(result)
}

/// Population / sample standard deviation from one-pass moments — replicated exactly
/// from core's `stddev_of`. `pop` is NULL over 0 rows, `samp` over fewer than 2; the
/// summed squared deviation is clamped at 0 (preserving NaN) so f64 cancellation
/// can't slip a tiny negative into `sqrt`.
fn stddev_of(n: u64, sum: f64, sum_sq: f64, sample: bool) -> Value {
    let denom = if sample {
        if n < 2 {
            return Value::Null;
        }
        (n - 1) as f64
    } else {
        if n == 0 {
            return Value::Null;
        }
        n as f64
    };
    let nf = n as f64;
    let variance = (sum_sq - sum * sum / nf) / denom;
    let clamped = if variance.is_nan() {
        f64::NAN
    } else {
        variance.max(0.0)
    };
    Value::Num(clamped.sqrt())
}

/// A hop: for each input row, expand the node in slot `from` along `dir`,
/// filtered by `edge_label`; emit one output row per matching neighbour with the
/// existing slots replicated and the neighbour appended as a new slot. This is
/// the bulk (lineage-free) strategy: `keep` records which input row each output
/// row came from, `nbrs` the landed node — the existing slots are gathered by
fn reverse_dir(dir: Dir) -> Dir {
    match dir {
        Dir::Out => Dir::In,
        Dir::In => Dir::Out,
        Dir::Both => Dir::Both,
    }
}

/// Cardinality-driven ANCHOR FLIP for `Filter(target = lit, Expand(Scan(src), Out))`
/// — the "selective filter on the traversal TARGET" shape. The forward plan scans
/// EVERY source and expands to filter the target at the end; when the target is an
/// indexed `=` whose bucket is smaller than the source scan, it is far cheaper to
/// SEED the target (index) and walk the edges in REVERSE to the sources. The output
/// is the IDENTICAL `[source, target]` slot layout, so nothing downstream changes;
/// only the (unspecified) row order differs — the multiset is preserved. `None`
/// unless the shape matches, the target is index-seekable, the cost says flip, and
/// no path is tracked / no edge is bound.
fn try_reverse_expand(pred: &Expr, input: &Plan, store: &Store, track: bool) -> Option<Batch> {
    if track {
        return None; // a path-reading query keeps the forward walk (lineage)
    }
    let Plan::Expand {
        input: src,
        from: 0,
        dir,
        edge_label,
        bind_edge: false,
    } = input
    else {
        return None;
    };
    let Plan::Scan { label: src_label } = src.as_ref() else {
        return None; // source must be an unfiltered scan (else seed the source)
    };
    // pred must be `target.key = lit` on the appended slot (slot 1 over a 1-wide
    // scan) with a hash index on `key`.
    let (key, value) = target_eq(pred)?;
    if !store.has_hash_index(&key) {
        return None;
    }
    // Cardinality decision: seed the SMALLER side. Forward seeds the source scan;
    // reverse seeds the target bucket. Flip only when the target bucket is smaller.
    let target_rows = store.index_bucket_len(&key, &value)?;
    let source_rows = match src_label {
        Some(l) => store.nodes_with_label(l).len(),
        None => store.live_node_count(),
    };
    if target_rows >= source_rows {
        return None;
    }
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return None,
    };
    let rev = reverse_dir(*dir);
    // Seed the targets (raw index bucket, any label — the forward path does not
    // constrain the target's label either), walk reverse edges to the sources, and
    // keep only sources carrying the scan's label.
    let targets = store.index_lookup(&key, &value)?;
    let mut sources = Vec::new();
    let mut ends = Vec::new();
    for &t in &targets {
        for_each_nbr(store, t, rev, &want, |a, _| {
            if src_label.as_deref().is_none_or(|l| store.is_labeled(a, l)) {
                sources.push(a);
                ends.push(t);
            }
        });
    }
    Some(Batch::of(vec![Col::Nodes(sources), Col::Nodes(ends)]))
}

/// Parse `Prop{slot 1, key} = Lit(value)` (or its mirror) — a target-slot equality.
fn target_eq(pred: &Expr) -> Option<(String, Value)> {
    let Expr::Compare {
        op: CompareOp::Eq,
        left,
        right,
    } = pred
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot: 1, key }, Expr::Lit(v))
        | (Expr::Lit(v), Expr::Prop { slot: 1, key }) => {
            (!v.is_null()).then(|| (key.clone(), v.clone()))
        }
        _ => None,
    }
}

/// `keep`, so no per-row struct is built.
fn expand(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
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
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return empty(),
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
        for_each_nbr(store, v, dir, &want, |nbr, eid| {
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

/// LEFT-OUTER single hop (`Plan::OptionalExpand`, GQL `OPTIONAL MATCH`): like
/// [`expand`], but a source row with NO matching neighbour is KEPT, its appended
/// node slot holding the `u32::MAX` null sentinel (read back as NULL everywhere).
/// Node-only, no lineage. So every input row yields at least one output row.
fn optional_expand(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
    keep_source: bool,
) -> Batch {
    // The value a missed row lands: the source element (Gremlin optional) or the
    // null sentinel (GQL OPTIONAL MATCH). `miss(v)` picks per row.
    let miss = |v: u32| if keep_source { v } else { u32::MAX };
    // Every left row gets exactly one neighbour-less row — used when the edge type
    // is unknown, or the `from` slot isn't a node frontier.
    let all_miss = || {
        let keep: Vec<usize> = (0..batch.rows()).collect();
        let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
        let landed: Vec<u32> = match (keep_source, batch.slot(from)) {
            (true, Col::Nodes(src)) => src.clone(),
            _ => vec![u32::MAX; batch.rows()],
        };
        slots.push(Col::Nodes(landed));
        Batch::of(slots)
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return all_miss(), // unknown edge type → no match for any row
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return all_miss();
    };
    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        let before = nbrs.len();
        for_each_nbr(store, v, dir, &want, |nbr, _| {
            keep.push(row);
            nbrs.push(nbr);
        });
        if nbrs.len() == before {
            // No neighbour — keep the row, landing the miss value (source or null).
            keep.push(row);
            nbrs.push(miss(v));
        }
    }
    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(nbrs));
    Batch::of(slots)
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
    edge_label: &[String],
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
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return empty(),
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
            for_each_nbr(store, v, dir, &want, |nbr, eid| {
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
            // group_key == equals for a finite, non-null value, so the index bucket
            // is exact; intersect with the label. The label bucket is sorted
            // ascending, so binary-search each (usually few) candidate rather than
            // building a HashSet of the WHOLE label per query — that O(label) build
            // made an indexed seek SLOWER than the typed scan it was meant to beat.
            let bucket = store.nodes_with_label(label);
            cands
                .into_iter()
                .filter(|id| bucket.binary_search(id).is_ok())
                .collect()
        }
        None => {
            let ids = store.nodes_with_label(label);
            // Typed fast paths for a plain (non-dotted) key: compare the raw column
            // — a `&str`/`f64`/`bool` compare, no per-cell `Value` boxing or `Arc`
            // clone. Equality semantics match `value::equals` (a present cell of the
            // literal's type; a NULL cell — `present == false` — never equals).
            if !key.contains('.') {
                match (store.column(key), value) {
                    (Some(Column::Str { data, present }), Value::Str(t)) => {
                        let t: &str = t;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && &*data[id as usize] == t)
                            .collect();
                    }
                    (
                        Some(Column::Dict {
                            dict,
                            codes,
                            present,
                        }),
                        Value::Str(t),
                    ) => {
                        // Resolve the literal to its code ONCE, then match rows on the
                        // `u32` — no per-row string compare. A literal absent from the
                        // dict matches nothing.
                        let t: &str = t;
                        let Some(want) = dict.iter().position(|s| &**s == t) else {
                            return Vec::new();
                        };
                        let want = want as u32;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && codes[id as usize] == want)
                            .collect();
                    }
                    (Some(Column::Num { data, present }), Value::Num(t)) => {
                        let t = *t;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && data[id as usize] == t)
                            .collect();
                    }
                    (Some(Column::Bool { data, present }), Value::Bool(t)) => {
                        let t = *t;
                        return ids
                            .iter()
                            .copied()
                            .filter(|&id| present[id as usize] && data[id as usize] == t)
                            .collect();
                    }
                    _ => {}
                }
            }
            // `key` may be a dotted record path — resolve it (plain keys read as
            // `prop`), so the no-index fallback matches a dotted seek too.
            ids.iter()
                .copied()
                .filter(|&id| value::equals(&store.prop_path(id, key), value))
                .collect()
        }
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
/// Raw-f64 comparison matching the value contract for two present numbers: `==`/
/// `!=` for equality, `<`/`<=`/`>`/`>=` for ordering (a NaN operand makes ordering
/// false — 3VL "unknown → drop", as `cmp_partial` gives). Used by the typed
/// scan/filter fast paths so a numeric predicate never boxes a `Value`.
fn num_pred(op: CompareOp, x: f64, t: f64) -> bool {
    match op {
        CompareOp::Eq => x == t,
        CompareOp::Ne => x != t,
        CompareOp::Lt => x < t,
        CompareOp::Le => x <= t,
        CompareOp::Gt => x > t,
        CompareOp::Ge => x >= t,
    }
}

fn range_seek_ids(store: &Store, label: &str, key: &str, op: CompareOp, value: &Value) -> Vec<u32> {
    if value.is_null() {
        return Vec::new();
    }
    match store.range_lookup(key, op, value) {
        Some(cands) => {
            // Test label membership with a per-candidate binary search of the sorted
            // label bucket (O(cands·log|label|)), NOT a HashSet built from the whole
            // bucket — that build is O(|label|) and dominates when the label covers
            // most of the graph (a single-label store makes it pure waste: the range
            // index already narrowed to `cands`, then we'd rebuild a set of everything
            // to intersect back down). When the bucket covers ALL non-deleted nodes,
            // every candidate is in-label, so skip the test entirely.
            let bucket = store.nodes_with_label(label);
            let all_in_label = bucket.len() == store.live_node_count();
            cands
                .into_iter()
                .filter(|&id| all_in_label || store.is_labeled(id, label))
                // The index orders by the TOTAL order (cross-type by rank), but the
                // OPERATOR is three-valued (cross-type → UNKNOWN → drop). Re-check
                // each candidate with `range_pass` so an indexed seek returns
                // exactly the scan-filter rows (the equivalent-spellings invariant);
                // for a homogeneous column this keeps every candidate.
                .filter(|&id| range_pass(&store.prop(id, key), op, value))
                .collect()
        }
        None => {
            let ids = store.nodes_with_label(label);
            // Typed fast path: a Num column vs a Num bound compares RAW f64 (no
            // per-cell Value boxing) — the no-index scan is the common case.
            if let (Some(Column::Num { data, present }), Value::Num(t)) = (store.column(key), value)
            {
                let t = *t;
                return ids
                    .iter()
                    .copied()
                    .filter(|&id| present[id as usize] && num_pred(op, data[id as usize], t))
                    .collect();
            }
            ids.iter()
                .copied()
                .filter(|&id| range_pass(&store.prop(id, key), op, value))
                .collect()
        }
    }
}

/// Visit each neighbour of `v` along `dir` matching edge type `want`, calling `f`
/// with `(neighbour, eid)`. The one place Expand's adjacency walk is spelled —
/// shared by the batch operator and the frontier executor so the two can never
/// disagree on what an Expand reaches.
/// Resolve an edge-type constraint (the plan's `edge_label` list) to the matching
/// type ids the walkers filter on. An EMPTY list is untyped — any edge — so it
/// returns `Ok(vec![])` (an empty `want` slice reads as "any"). A typed list whose
/// names ALL fail to resolve matches no edge, so it returns `Err(())` and the caller
/// short-circuits to its own empty result. Otherwise the known ids, unknown names
/// dropped — mirroring core's `lower_labels`.
fn want_etypes(store: &Store, edge_label: &[String]) -> Result<Vec<u32>, ()> {
    if edge_label.is_empty() {
        return Ok(Vec::new());
    }
    let ids: Vec<u32> = edge_label
        .iter()
        .filter_map(|n| store.etype_id(n))
        .collect();
    if ids.is_empty() {
        return Err(());
    }
    Ok(ids)
}

/// Does adjacency entry `a` carry one of the `want` labels? Empty `want` = any
/// edge. Checks the primary type (`a.etype`, already in a register) then, only on a
/// multi-label graph, the eid's secondary set. The one predicate every edge-type
/// filter shares, so a `:Y` hop over a multi-label edge matches everywhere.
#[inline]
fn edge_carries_wanted(store: &Store, a: &crate::store::Adj, want: &[u32]) -> bool {
    want.is_empty()
        || want.iter().any(|&w| {
            w == a.etype || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
        })
}

fn for_each_nbr(store: &Store, v: u32, dir: Dir, want: &[u32], mut f: impl FnMut(u32, u32)) {
    // An undirected walk reaches a self-loop from BOTH the out- and the in-index;
    // emit it once (from the out-side) by dropping its in-side copy. Directed walks
    // touch one index, so they keep it either way. Matches core's SelfLoops::Once.
    let drop_loop = matches!(dir, Dir::Both);
    // A SINGLE-type hop over an indexed store seeks the type bucket directly
    // (O(matching), not O(degree)) — the whole point of the opt-in edge-type index.
    // A disjunction (`want.len() >= 2`) must NOT union buckets: that reorders vs the
    // flat stored-order scan and would break byte-identity, so it falls through.
    // The type-index bucket keys on an edge's PRIMARY label only, so it cannot see
    // a `:Y` match on a multi-label edge whose first label is `X`. Skip it whenever
    // the graph has any multi-label edge (rare) and fall to the flat scan below,
    // which consults the secondary labels.
    if let [w] = want {
        if store.has_edge_type_index() && !store.has_multi_label_edges() {
            if matches!(dir, Dir::Out | Dir::Both) {
                for a in store.out_typed(v, *w) {
                    f(a.nbr, a.eid);
                }
            }
            if matches!(dir, Dir::In | Dir::Both) {
                for a in store.in_typed(v, *w) {
                    if !(drop_loop && a.nbr == v) {
                        f(a.nbr, a.eid);
                    }
                }
            }
            return;
        }
    }
    // Empty `want` = any type; otherwise the edge must carry one of the wanted
    // labels. `edge_has_label` checks the primary type (already in `a.etype`) then,
    // only when the graph has multi-label edges, the eid's secondary set.
    let has_extra = store.has_multi_label_edges();
    let type_ok = |et: u32, eid: u32| {
        want.is_empty()
            || want
                .iter()
                .any(|&w| w == et || (has_extra && store.edge_has_label(eid, w)))
    };
    if matches!(dir, Dir::Out | Dir::Both) {
        for a in store.out(v) {
            if type_ok(a.etype, a.eid) {
                f(a.nbr, a.eid);
            }
        }
    }
    if matches!(dir, Dir::In | Dir::Both) {
        for a in store.inc(v) {
            if type_ok(a.etype, a.eid) && !(drop_loop && a.nbr == v) {
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
            let want = match want_etypes(store, edge_label) {
                Ok(w) => w,
                Err(()) => return Some(Vec::new()), // unknown label matches nothing
            };
            let mut out = Vec::new();
            for &v in &src {
                for_each_nbr(store, v, *dir, &want, |nbr, _eid| out.push(nbr));
            }
            Some(out)
        }
        _ => None,
    }
}

/// The number of `Expand` hops in a pure Scan/Expand chain (0 for a bare seed).
fn count_hops(plan: &Plan) -> usize {
    match plan {
        Plan::Expand { input, .. } => 1 + count_hops(input),
        _ => 0,
    }
}

/// A per-node PATH-COUNT frontier, stored SPARSE (a list of active `(node, count)`)
/// while few nodes carry a path, DENSE (indexed by node id) once the frontier
/// covers a large fraction of the graph. A 5-hop count from a SINGLE source touches
/// at most fan-out^hops distinct nodes — kept sparse, it costs O(active) per hop
/// instead of the O(node_count) alloc + full scan a dense array pays every hop
/// (that made `aml/chain5` 30x SLOWER than core: an 8 MB zeroed array and a 1M-entry
/// scan, five times, for a frontier of a few hundred nodes). Counts are exact
/// integers (< 2^53), so the f64 sums are order-independent and the representation
/// switch is byte-identical.
enum Counts {
    Sparse(Vec<(u32, f64)>),
    Dense(Vec<f64>),
}

impl Counts {
    /// Call `f(node, count)` for every node carrying a non-zero count.
    fn for_each(&self, mut f: impl FnMut(u32, f64)) {
        match self {
            Counts::Sparse(v) => {
                for &(id, c) in v {
                    f(id, c);
                }
            }
            Counts::Dense(a) => {
                for (i, &c) in a.iter().enumerate() {
                    if c != 0.0 {
                        f(i as u32, c);
                    }
                }
            }
        }
    }

    /// Number of nodes carrying a path — the cost driver for the next hop.
    fn active(&self) -> usize {
        match self {
            Counts::Sparse(v) => v.len(),
            Counts::Dense(a) => a.iter().filter(|&&c| c != 0.0).count(),
        }
    }
}

/// The per-node PATH-COUNT frontier of a pure Scan/Expand chain: `counts[v]` is the
/// number of chain paths whose last node is `v`. Propagated one hop at a time
/// (`next[nbr] += counts[v]` over each matching edge) so it never materializes the
/// exploding path multiset that [`frontier_ids`] carries — O(hops * edges) time.
/// Sparse until the frontier is large (see [`Counts`]), so a narrow deep chain pays
/// O(active) not O(node_count) per hop. `None` for a non-chain.
fn frontier_counts(plan: &Plan, store: &Store) -> Option<Counts> {
    let n = store.node_count();
    // Go dense once the active set is a large fraction of the graph: past this a
    // dense array's O(1) scatter beats an FnvMap's hashing, and a full-scan seed is
    // dense from the start. Below it the sparse list wins by touching only live nodes.
    let dense_cut = (n / 16).max(1024);
    match plan {
        Plan::Scan { label } => {
            let seed: &[u32] = match label {
                Some(l) => store.nodes_with_label(l),
                None => return Some(dense_from(store.all_nodes().into_iter(), n)),
            };
            if seed.len() > dense_cut {
                let mut counts = vec![0.0f64; n];
                for &v in seed {
                    counts[v as usize] = 1.0;
                }
                Some(Counts::Dense(counts))
            } else {
                Some(Counts::Sparse(seed.iter().map(|&v| (v, 1.0)).collect()))
            }
        }
        Plan::IndexSeek { label, key, value } => Some(sparse_or_dense(
            index_seek_ids(store, label, key, value),
            n,
            dense_cut,
        )),
        Plan::RangeSeek {
            label,
            key,
            op,
            value,
        } => Some(sparse_or_dense(
            range_seek_ids(store, label, key, *op, value),
            n,
            dense_cut,
        )),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            ..
        } => {
            if *from + 1 != chain_width(input)? {
                return None;
            }
            let prev = frontier_counts(input, store)?;
            let want = match want_etypes(store, edge_label) {
                Ok(w) => w,
                Err(()) => return Some(Counts::Sparse(Vec::new())), // unknown label → no paths
            };
            // Estimate the next frontier's fan-out from the source count and the
            // average degree; go dense when it will be large, sparse otherwise. The
            // scatter itself is identical either way — only the accumulator differs.
            let avg_deg = if n == 0 {
                0.0
            } else {
                store.edge_count() as f64 / n as f64
            };
            let est_next = prev.active() as f64 * avg_deg.max(1.0);
            if est_next > dense_cut as f64 {
                let mut next = vec![0.0f64; n];
                prev.for_each(|v, c| {
                    for_each_nbr(store, v, *dir, &want, |nbr, _| next[nbr as usize] += c);
                });
                Some(Counts::Dense(next))
            } else {
                // Sparse scatter into an FnvMap keyed by neighbour — touches only the
                // few nodes a narrow frontier reaches, no O(node_count) allocation.
                let mut next: FnvMap<u32, f64> = FnvMap::default();
                prev.for_each(|v, c| {
                    for_each_nbr(store, v, *dir, &want, |nbr, _| {
                        *next.entry(nbr).or_insert(0.0) += c;
                    });
                });
                Some(Counts::Sparse(next.into_iter().collect()))
            }
        }
        _ => None,
    }
}

/// A DENSE count frontier with each listed id set to 1.0 (for a full unlabeled scan).
fn dense_from(ids: impl Iterator<Item = u32>, n: usize) -> Counts {
    let mut counts = vec![0.0f64; n];
    for v in ids {
        counts[v as usize] = 1.0;
    }
    Counts::Dense(counts)
}

/// Seed a count frontier from a seek's id list: sparse when the result is small
/// (the common selective seek), dense when it is large. Duplicate ids accumulate.
fn sparse_or_dense(ids: Vec<u32>, n: usize, dense_cut: usize) -> Counts {
    if ids.len() > dense_cut {
        let mut counts = vec![0.0f64; n];
        for v in ids {
            counts[v as usize] += 1.0;
        }
        Counts::Dense(counts)
    } else {
        // A seek CAN repeat an id (an index bucket with dups); fold so each node
        // appears once with its multiplicity, matching the dense accumulation.
        let mut map: FnvMap<u32, f64> = FnvMap::default();
        for v in ids {
            *map.entry(v).or_insert(0.0) += 1.0;
        }
        Counts::Sparse(map.into_iter().collect())
    }
}

/// Answer a scalar `count(*)` over `Filter(Scan)` by running only the filter and
/// returning the number of survivors — NO gather of the surviving rows' columns
/// (which the general Filter → Aggregate path builds and immediately discards for a
/// count). Relies on `try_filter_keep`'s vectorized filter; falls back (`None`) when
/// the predicate isn't a fast-path shape or the input isn't `Filter(Scan)`.
fn try_filtered_count(
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
    let Plan::Filter { input: scan, pred } = input else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    // STREAM a numeric predicate over the label bucket — count matches with raw-f64
    // compares, never materializing the scan's id vector or a keep list. This is
    // core's structure (it iterates the bucket and tests inline) but with the
    // engine's typed compare instead of core's boxed CExpr tree-walk: measured 3.67x
    // core (and ~5x the engine's own materialize-then-filter) on a 200k range count.
    if let Some(c) = try_stream_num_count(store, label, pred) {
        return Some(scalar_num(c as f64));
    }
    // Fallback: materialize the scan and run the general vectorized filter (string /
    // disjunction / NOT predicates the streaming path does not special-case).
    let batch = pull(scan, store, false).ok()?;
    let keep = try_filter_keep(pred, store, &batch)?;
    Some(scalar_num(keep.len() as f64))
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
fn num_conj_on_slot(pred: &Expr, slot: usize) -> Option<(String, Vec<(CompareOp, f64)>)> {
    let atom = |e: &Expr| -> Option<(String, CompareOp, f64)> {
        let Expr::Compare { op, left, right } = e else {
            return None;
        };
        let (key, op, lit) = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot: s, key }, Expr::Lit(v)) if *s == slot => (key.clone(), *op, v),
            (Expr::Lit(v), Expr::Prop { slot: s, key }) if *s == slot => {
                (key.clone(), flip_op(*op), v)
            }
            _ => return None,
        };
        match lit {
            Value::Num(t) => Some((key, op, *t)),
            _ => None,
        }
    };
    let mut conjuncts = Vec::new();
    flatten_and(pred, &mut conjuncts);
    let mut key0: Option<String> = None;
    let mut bounds: Vec<(CompareOp, f64)> = Vec::with_capacity(conjuncts.len());
    for c in &conjuncts {
        let (key, op, t) = atom(c)?;
        match &key0 {
            Some(k) if *k != key => return None,
            _ => key0 = Some(key),
        }
        bounds.push((op, t));
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
fn try_edge_filtered_count(
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
            for_each_nbr(store, v, *dir, &want, |_nbr, eid| {
                let i = eid as usize;
                if present[i] && bounds.iter().all(|&(op, t)| num_pred(op, data[i], t)) {
                    count += 1;
                }
            });
        }
    } else {
        for &v in &src_ids {
            for_each_nbr(store, v, *dir, &want, |_nbr, eid| {
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

fn try_stream_num_count(store: &Store, label: &Option<String>, pred: &Expr) -> Option<u64> {
    let (key, bounds) = num_conj_on_slot(pred, 0)?;
    let Some(Column::Num { data, present }) = store.column(&key) else {
        return None;
    };
    let mut count = 0u64;
    scan_visit(store, label, |i| {
        if present[i] {
            let x = data[i];
            if bounds.iter().all(|&(op, t)| num_pred(op, x, t)) {
                count += 1;
            }
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
fn try_varlen_count(
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
    } = input
    else {
        return None;
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return Some(scalar_num(0.0)), // unknown edge type → no paths
    };
    let batch = pull(inner, store, false).ok()?;
    let Col::Nodes(src) = batch.slot(*from) else {
        return None;
    };

    // ALGEBRAIC count: for a bounded OUT trail with max<=2, count(*) is the sum of
    // per-hop trail counts computed from degrees in O(V+E) — NOT by enumerating the
    // O(paths) walks. 1-hop = the source out-edges; 2-hop = for each source out-edge
    // s->y, the neighbour's out-degree, minus the one reused-self-loop path
    // (s->s->s over the same edge, which a trail forbids). Only taken when the
    // enumeration would be the MORE expensive path (a large source set); a filtered
    // / small source stays on the DFS below, where enumeration is already cheap.
    // TRAIL only: the degree algebra counts node-repeating trails, which SIMPLE /
    // ACYCLIC forbid — those must enumerate via the DFS below.
    if matches!(dir, Dir::Out) && *max <= 2 && *max >= 1 && matches!(mode, PathMode::Trail) {
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
                        total += 1; // the 1-hop trail s -> a.nbr
                    }
                    if *max >= 2 {
                        total += outdeg[a.nbr as usize]; // 2-hop trails s -> a.nbr -> z
                        if a.nbr == s {
                            total -= 1; // exclude the reused self-loop s -> s -> s
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
            store, v, 0, *min, *max, *dir, &want, *mode, v, &mut used, &mut total,
        );
        if node_unique {
            used.pop();
        }
        debug_assert!(used.is_empty());
    }
    Some(scalar_num(total as f64))
}

/// The counting twin of `varlen_dfs`: increments `total` at every length in
/// `min..=max` instead of pushing `(row, endpoint)`. Traversal order, edge-type
/// filtering and trail (no-edge-reuse) logic are identical, so the tally equals
/// the number of rows the materializing path would emit.
#[allow(clippy::too_many_arguments)]
fn varlen_count_dfs(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    total: &mut u64,
) {
    if len >= min {
        *total += 1;
    }
    if len == max {
        return;
    }
    let out: &[crate::store::Adj] = if matches!(dir, Dir::Out | Dir::Both) {
        store.out(v)
    } else {
        &[]
    };
    let inc: &[crate::store::Adj] = if matches!(dir, Dir::In | Dir::Both) {
        store.inc(v)
    } else {
        &[]
    };
    // Undirected: a self-loop sits in both indexes; drop its in-side copy so it is
    // walked once (matches core's SelfLoops::Once).
    let drop_loop = matches!(dir, Dir::Both);
    for (is_inc, a) in out
        .iter()
        .map(|a| (false, a))
        .chain(inc.iter().map(|a| (true, a)))
    {
        // The edge must carry a wanted label — its primary type (`a.etype`) or, on
        // a multi-label graph, a secondary one (`edge_has_label`).
        if !want.is_empty()
            && !want.iter().any(|&w| {
                w == a.etype || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
            })
        {
            continue;
        }
        if is_inc && drop_loop && a.nbr == v {
            continue;
        }
        let mark = match varlen_step(mode, start, a, used) {
            VarStep::Skip => continue,
            VarStep::Close => {
                if len + 1 >= min {
                    *total += 1;
                }
                continue;
            }
            VarStep::Go(mark) => mark,
        };
        if let Some(m) = mark {
            used.push(m);
        }
        varlen_count_dfs(
            store,
            a.nbr,
            len + 1,
            min,
            max,
            dir,
            want,
            mode,
            start,
            used,
            total,
        );
        if mark.is_some() {
            used.pop();
        }
    }
}

/// The outcome of the per-hop reuse gate ([`varlen_step`]).
enum VarStep {
    /// The hop is forbidden — skip this neighbour.
    Skip,
    /// A SIMPLE closing hop (`nbr == start`): emit the endpoint but do NOT descend —
    /// the cycle is closed, and extending it would repeat an interior node (mirrors
    /// core's `is_close` early-`continue`).
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
fn varlen_step(mode: PathMode, start: u32, a: &crate::store::Adj, used: &[u32]) -> VarStep {
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
/// byte-identical to core's regardless of visitation order.
fn try_varlen_distinct_count(
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
    } = input
    else {
        return None;
    };
    // The BFS-reachability fusion relies on shortest-distance == walk equivalence,
    // which only holds when nodes may repeat (Walk / Trail). SIMPLE / ACYCLIC forbid
    // node reuse, so a distinct-endpoint count must enumerate — fall through.
    if !matches!(mode, PathMode::Walk | PathMode::Trail) {
        return None;
    }
    // Only min ≤ 1 (see the invariant above); a min-0 lower bound also counts the
    // sources themselves as 0-hop endpoints.
    if *min > 1 {
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
    let n = store.node_count();
    let mut visited = vec![false; n]; // added to a frontier (expansion dedup)
    let mut reached = vec![false; n]; // a valid endpoint (hop in min..=max)
    let mut frontier: Vec<u32> = Vec::with_capacity(src.len());
    for &s in src {
        if !visited[s as usize] {
            visited[s as usize] = true;
            frontier.push(s);
        }
        if *min == 0 {
            reached[s as usize] = true; // the 0-hop path a=b
        }
    }
    // BFS `max` levels. Each node is expanded at most once (at its shortest distance),
    // so an edge is traversed once from its source — O(E). Every neighbour reached is
    // an endpoint (hop ≥ 1 ≥ min); newly-seen ones seed the next level.
    let mut next: Vec<u32> = Vec::new();
    for _hop in 1..=*max {
        if frontier.is_empty() {
            break;
        }
        for &v in &frontier {
            for_each_nbr(store, v, *dir, &want, |nbr, _| {
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
    let count = reached.iter().filter(|&&r| r).count();
    Some(scalar_num(count as f64))
}

/// The fold twin of `varlen_count_dfs`: calls `emit(endpoint)` at every length in
/// `min..=max` instead of counting. Traversal / edge-type / trail logic — and thus
/// the EMISSION ORDER — are identical to `var_length`, so a `sum` folded here lands
/// the same value as materializing then summing.
#[allow(clippy::too_many_arguments)]
fn varlen_agg_dfs(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    emit: &mut dyn FnMut(u32),
) {
    if len >= min {
        emit(v);
    }
    if len == max {
        return;
    }
    let out: &[crate::store::Adj] = if matches!(dir, Dir::Out | Dir::Both) {
        store.out(v)
    } else {
        &[]
    };
    let inc: &[crate::store::Adj] = if matches!(dir, Dir::In | Dir::Both) {
        store.inc(v)
    } else {
        &[]
    };
    // Undirected: drop the in-side copy of a self-loop so it is walked once.
    let drop_loop = matches!(dir, Dir::Both);
    for (is_inc, a) in out
        .iter()
        .map(|a| (false, a))
        .chain(inc.iter().map(|a| (true, a)))
    {
        // The edge must carry a wanted label — its primary type (`a.etype`) or, on
        // a multi-label graph, a secondary one (`edge_has_label`).
        if !want.is_empty()
            && !want.iter().any(|&w| {
                w == a.etype || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
            })
        {
            continue;
        }
        if is_inc && drop_loop && a.nbr == v {
            continue;
        }
        let mark = match varlen_step(mode, start, a, used) {
            VarStep::Skip => continue,
            VarStep::Close => {
                if len + 1 >= min {
                    emit(a.nbr);
                }
                continue;
            }
            VarStep::Go(mark) => mark,
        };
        if let Some(m) = mark {
            used.push(m);
        }
        varlen_agg_dfs(
            store,
            a.nbr,
            len + 1,
            min,
            max,
            dir,
            want,
            mode,
            start,
            used,
            emit,
        );
        if mark.is_some() {
            used.pop();
        }
    }
}

/// A scalar `sum`/`avg`/`min`/`max`/`count(arg)` over a bare var-length's ENDPOINT
/// property, folded DURING the DFS — no keep/ends, no gather, no intermediate batch
/// (which `try_frontier_aggregate`/`aggregate` all build, ~3x the traversal). The
/// emission order matches `var_length`, so `sum` folds in the same order and the
/// value contract (`cmp_num_total`) drives min/max — byte-identical to the
/// materializing path. `None` unless the aggregate reads exactly the appended
/// endpoint slot (block-streaming the general chain was measured a net regression;
/// this surgical fold is the low-overhead win).
fn try_varlen_agg(
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
    } = input
    else {
        return None;
    };
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
        (AggFn::Sum | AggFn::Avg, Column::Num { data, present }) => {
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
        (AggFn::Min | AggFn::Max, Column::Num { data, present }) => {
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
        (AggFn::Min | AggFn::Max, Column::Str { data, present }) => {
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

/// Answer a scalar `count(*)` over a bare labelled/unlabelled `Scan` in O(1) (a
/// label bucket length — buckets hold only live ids) or a single tombstone-bitmap
/// sweep (unlabelled), WITHOUT materializing the id vector. `None` for any other
/// shape (a WHERE seed, an Expand, `count(arg)`), which the other paths handle.
fn try_scan_count(
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
fn try_scan_num_agg(
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
    let Some(Column::Num { data, present }) = store.column(key) else {
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
fn scan_visit<F: FnMut(usize)>(store: &Store, label: &Option<String>, mut f: F) {
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
fn try_scan_group_agg(
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
                let Some(Column::Num { data, present }) = store.column(key) else {
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
        Column::Str { data, present } => {
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
fn low_card_int_bitset(
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

fn try_scan_distinct_count(
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
        Column::Str { data, present } => {
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
        Column::Num { data, present } => {
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
        Column::Bool { data, present } => {
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

/// Answer several scalar numeric aggregates (`sum`/`avg`/`min`/`max`/`count`) over
/// a bare `Scan` in ONE pass over the Num columns — e.g. `min(age), max(age)` or
/// `count(*), avg(age)`. `None` if any agg is grouped/DISTINCT or not a numeric
/// reduction over a `Num` property (or `count(*)`). Complements the single-agg
/// [`try_scan_num_agg`], which keeps the tighter auto-vectorized loop.
fn try_scan_multi_agg(
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
                let Some(Column::Num { data, present }) = store.column(key) else {
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
fn peel_out_hops(plan: &Plan, n: usize) -> Option<(Vec<Vec<String>>, Option<String>)> {
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
fn try_3hop_product_count(
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
                    for_each_nbr(store, v, *dir, &want, |_, _| deg += 1.0);
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
        let mut total = 0f64;
        if matches!(inner.as_ref(), Plan::Expand { .. }) {
            let (distinct, mult) = distinct_with_mult(&src, store.node_count());
            for (i, &v) in distinct.iter().enumerate() {
                let mut deg = 0f64;
                for_each_nbr(store, v, *dir, &want, |_, _| deg += 1.0);
                total += mult[i] * deg;
            }
        } else {
            for &v in &src {
                for_each_nbr(store, v, *dir, &want, |_, _| total += 1.0);
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
            for_each_nbr(store, v, *dir, &want, |nbr, _| {
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
        | Expr::Xor(a, b)
        | Expr::Arith {
            left: a, right: b, ..
        }
        | Expr::In {
            needle: a,
            haystack: b,
        } => refs_only_slot(a, s) && refs_only_slot(b, s),
        Expr::Call { args, .. } | Expr::List { items: args } => {
            args.iter().all(|a| refs_only_slot(a, s))
        }
        Expr::Record { fields } | Expr::MapLit { entries: fields } => {
            fields.iter().all(|(_, e)| refs_only_slot(e, s))
        }
        Expr::Field { base, .. } => refs_only_slot(base, s),
        Expr::Index { base, index } => refs_only_slot(base, s) && refs_only_slot(index, s),
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
        Expr::Exists { .. } | Expr::CountSubquery { .. } => false,
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
        Expr::Index { base, index } => Expr::Index {
            base: go(base),
            index: go(index),
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
        Expr::Exists { .. } | Expr::CountSubquery { .. } => expr.clone(),
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
/// 100k/deg-5 fixture only 24.5ms -> 23.0ms (0.54x -> 0.57x of core) — a consistent
/// ~7% but still far from parity, and it TRADES the per-node scatter for reading the
/// property once PER PATH (2.5M reads) instead of once per distinct endpoint (100k).
/// The shape is memory-bound on ~2.5M random accesses either way; core's remaining
/// edge is its CSR adjacency (sequential neighbour reads), which the per-node `Vec`
/// adjacency here cannot match without a layout change (deferred, large blast radius).
/// Not worth a second grouped-count path for a sub-10% move that leaves it slowest.
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
        for_each_nbr(store, v, *dir, &want, |nbr, _| {
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
            frac: a.frac,
        })
        .collect();
    Ok(Some(aggregate(&batch, store, &keys, &aggs)?))
}

/// A quantified hop: for each input row, enumerate every path of length in
/// `min..=max` from the node in `from`, and emit one output row per path with the
/// reached endpoint appended as a new slot. `min == 0` emits the source itself.
///
/// `mode` chooses the semantics and nothing else does (see [`PathMode`]): `Trail`
/// forbids reusing an edge, `Walk` allows anything, `Simple`/`Acyclic` forbid
/// reusing a node (Simple permits the closing `start == end`). They diverge on a
/// cycle/self-loop — pinned by the tests — and are never conflated with a chain
/// of separate fixed `Expand`s (which is always a walk).
#[allow(clippy::too_many_arguments)]
fn var_length(
    batch: &Batch,
    store: &Store,
    from: usize,
    dir: Dir,
    edge_label: &[String],
    min: u32,
    max: u32,
    mode: PathMode,
) -> Batch {
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return empty(),
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };

    // A named path over the pattern (`MATCH p = (a)-[:R]->{1,3}(b)`) needs the
    // per-row node/edge chain so path_length(p)/nodes(p)/edges(p) resolve; the
    // input carries a lineage exactly when the plan reads the path.
    let track = batch.lineage.is_some();
    let mut keep = Vec::new();
    let mut ends = Vec::new();
    // For Trail this is the EDGE stack; for Simple/Acyclic the NODE stack (seeded
    // with the start). Empty for Walk.
    let mut used: Vec<u32> = Vec::new();
    let mut bufs = PathBufs::new();
    let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
    for (row, &v) in src.iter().enumerate() {
        if node_unique {
            used.push(v); // mark the start node
        }
        // The node/edge chain from the source to the DFS frontier — seeded with the
        // source so `push_path` (which skips `chain[0]`, already the input path's
        // tail) reconstructs the whole path.
        let mut node_stack = vec![v];
        let mut edge_stack: Vec<u32> = Vec::new();
        varlen_dfs(
            store,
            v,
            0,
            min,
            max,
            dir,
            &want,
            mode,
            v,
            &mut used,
            row,
            &mut keep,
            &mut ends,
            if track { Some(batch) } else { None },
            &mut node_stack,
            &mut edge_stack,
            &mut bufs,
        );
        if node_unique {
            used.pop();
        }
        debug_assert!(used.is_empty());
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(ends));
    let mut out = Batch::of(slots);
    if track {
        out.lineage = Some(Lineage {
            values: bufs.values,
            offsets: bufs.offsets,
            edges: bufs.edges,
            edge_offsets: bufs.edge_offsets,
        });
    }
    out
}

/// Depth-first path enumeration for `var_length`. Emits `(row, endpoint)` at
/// every length in `min..=max` reached from the source, pushing straight into
/// `keep`/`ends` (a recursion-friendly alternative to a closure). `used` holds the
/// on-path elements that block reuse under `mode` — edge ids for a trail, node ids
/// for Simple/Acyclic (seeded with `start`). See [`varlen_step`].
#[allow(clippy::too_many_arguments)]
fn varlen_dfs(
    store: &Store,
    v: u32,
    len: u32,
    min: u32,
    max: u32,
    dir: Dir,
    want: &[u32],
    mode: PathMode,
    start: u32,
    used: &mut Vec<u32>,
    row: usize,
    keep: &mut Vec<usize>,
    ends: &mut Vec<u32>,
    // Lineage recording (a named path): `Some(input_batch)` records each emitted
    // path into `bufs`; `node_stack`/`edge_stack` hold the chain source..`v`.
    track_batch: Option<&Batch>,
    node_stack: &mut Vec<u32>,
    edge_stack: &mut Vec<u32>,
    bufs: &mut PathBufs,
) {
    if len >= min {
        keep.push(row);
        ends.push(v);
        if let Some(b) = track_batch {
            push_path(
                b,
                row,
                node_stack,
                edge_stack,
                &mut bufs.values,
                &mut bufs.offsets,
                &mut bufs.edges,
                &mut bufs.edge_offsets,
            );
        }
    }
    if len == max {
        return;
    }
    // Iterate the OUT then IN adjacency slices directly — chained, not copied into
    // a per-visit `Vec` (that allocation, once per node on a path of which there are
    // millions, dominated). Order is unchanged (out first, then in), so the emitted
    // path multiset and its order are bit-identical.
    let out: &[crate::store::Adj] = if matches!(dir, Dir::Out | Dir::Both) {
        store.out(v)
    } else {
        &[]
    };
    let inc: &[crate::store::Adj] = if matches!(dir, Dir::In | Dir::Both) {
        store.inc(v)
    } else {
        &[]
    };
    // Undirected: drop the in-side copy of a self-loop so it is walked once.
    let drop_loop = matches!(dir, Dir::Both);
    for (is_inc, a) in out
        .iter()
        .map(|a| (false, a))
        .chain(inc.iter().map(|a| (true, a)))
    {
        // The edge must carry a wanted label — its primary type (`a.etype`) or, on
        // a multi-label graph, a secondary one (`edge_has_label`).
        if !want.is_empty()
            && !want.iter().any(|&w| {
                w == a.etype || (store.has_multi_label_edges() && store.edge_has_label(a.eid, w))
            })
        {
            continue;
        }
        if is_inc && drop_loop && a.nbr == v {
            continue;
        }
        let mark = match varlen_step(mode, start, a, used) {
            VarStep::Skip => continue,
            VarStep::Close => {
                // Emit the closing endpoint (the start) at this length, no descent.
                if len + 1 >= min {
                    keep.push(row);
                    ends.push(a.nbr);
                    if let Some(b) = track_batch {
                        node_stack.push(a.nbr);
                        edge_stack.push(a.eid);
                        push_path(
                            b,
                            row,
                            node_stack,
                            edge_stack,
                            &mut bufs.values,
                            &mut bufs.offsets,
                            &mut bufs.edges,
                            &mut bufs.edge_offsets,
                        );
                        node_stack.pop();
                        edge_stack.pop();
                    }
                }
                continue;
            }
            VarStep::Go(mark) => mark,
        };
        if let Some(m) = mark {
            used.push(m);
        }
        node_stack.push(a.nbr);
        edge_stack.push(a.eid);
        varlen_dfs(
            store,
            a.nbr,
            len + 1,
            min,
            max,
            dir,
            want,
            mode,
            start,
            used,
            row,
            keep,
            ends,
            track_batch,
            node_stack,
            edge_stack,
            bufs,
        );
        node_stack.pop();
        edge_stack.pop();
        if mark.is_some() {
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
    edge_label: &[String],
    min: u32,
    max: Option<u32>,
    selector: crate::ir::ShortestSelector,
) -> Batch {
    use crate::ir::ShortestSelector;
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    let want = match want_etypes(store, edge_label) {
        Ok(w) => w,
        Err(()) => return empty(),
    };
    let Col::Nodes(src) = batch.slot(from) else {
        return empty();
    };

    // When the input carries a path, each emitted row reconstructs the node/edge
    // chain start..endpoint so `Expr::Path` (nodes(p)/path_length(p)) sees the whole
    // path. Endpoint MULTIPLICITY (how many shortest paths reach it) is emitted for
    // `ALL` regardless of lineage.
    let track = batch.lineage.is_some();
    let mut path_values: Vec<Value> = Vec::new();
    let mut path_offsets: Vec<usize> = vec![0];
    let mut path_edges: Vec<Value> = Vec::new();
    let mut path_edge_offsets: Vec<usize> = vec![0];

    let mut keep = Vec::new();
    let mut ends = Vec::new();

    for (row, &start) in src.iter().enumerate() {
        // BFS from `start`: shortest distance per node, plus ALL predecessors that
        // lie on a shortest path (an edge prev->node with dist[prev] + 1 == dist[node]).
        // `order` is BFS discovery order — every node's predecessors precede it, so a
        // bottom-up pass over it computes path counts / enumerations.
        let mut dist: FnvMap<u32, u32> = FnvMap::default();
        dist.insert(start, 0);
        let mut preds: FnvMap<u32, Vec<(u32, u32)>> = FnvMap::default();
        let mut order: Vec<u32> = vec![start];
        let mut q: VecDeque<u32> = VecDeque::new();
        q.push_back(start);
        while let Some(v) = q.pop_front() {
            let dv = dist[&v];
            if max.is_some_and(|m| dv >= m) {
                continue; // hop cap: do not expand past `max`
            }
            let mut adjs: Vec<crate::store::Adj> = Vec::new();
            if matches!(dir, Dir::Out | Dir::Both) {
                adjs.extend_from_slice(store.out(v));
            }
            if matches!(dir, Dir::In | Dir::Both) {
                adjs.extend_from_slice(store.inc(v));
            }
            for a in adjs {
                if !edge_carries_wanted(store, &a, &want) {
                    continue;
                }
                match dist.get(&a.nbr).copied() {
                    None => {
                        dist.insert(a.nbr, dv + 1);
                        preds.entry(a.nbr).or_default().push((v, a.eid));
                        order.push(a.nbr);
                        q.push_back(a.nbr);
                    }
                    // Another edge onto a node at its shortest distance — a second
                    // shortest-path predecessor.
                    Some(dn) if dn == dv + 1 => {
                        preds.entry(a.nbr).or_default().push((v, a.eid));
                    }
                    _ => {}
                }
            }
        }

        // `ALL` without lineage: emit each endpoint as many times as it has distinct
        // shortest paths (count DP over `order`).
        let mut pcount: FnvMap<u32, u64> = FnvMap::default();
        if matches!(selector, ShortestSelector::All) && !track {
            pcount.insert(start, 1);
            for &node in &order {
                if node == start {
                    continue;
                }
                let c = preds
                    .get(&node)
                    .map(|ps| {
                        ps.iter()
                            .map(|&(p, _)| pcount.get(&p).copied().unwrap_or(0))
                            .sum()
                    })
                    .unwrap_or(0);
                pcount.insert(node, c);
            }
        }

        for &node in &order {
            let dn = dist[&node];
            if dn < min {
                continue; // a `+` quantifier (min 1) excludes the zero-length seed
            }
            match selector {
                ShortestSelector::Any => {
                    keep.push(row);
                    ends.push(node);
                    if track {
                        let (chain, echain) = first_pred_chain(node, start, &preds);
                        push_path(
                            batch,
                            row,
                            &chain,
                            &echain,
                            &mut path_values,
                            &mut path_offsets,
                            &mut path_edges,
                            &mut path_edge_offsets,
                        );
                    }
                }
                ShortestSelector::All => {
                    if track {
                        for (chain, echain) in enumerate_shortest_paths(node, start, &preds) {
                            keep.push(row);
                            ends.push(node);
                            push_path(
                                batch,
                                row,
                                &chain,
                                &echain,
                                &mut path_values,
                                &mut path_offsets,
                                &mut path_edges,
                                &mut path_edge_offsets,
                            );
                        }
                    } else {
                        for _ in 0..pcount.get(&node).copied().unwrap_or(0) {
                            keep.push(row);
                            ends.push(node);
                        }
                    }
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

/// The single shortest path start..node via the FIRST predecessor of each node (the
/// BFS-tree parent) — the representative `ANY SHORTEST` keeps. Returns the node chain
/// (start..node inclusive) and its edge chain. `node == start` gives `([start], [])`.
fn first_pred_chain(
    node: u32,
    start: u32,
    preds: &FnvMap<u32, Vec<(u32, u32)>>,
) -> (Vec<u32>, Vec<u32>) {
    let mut chain = vec![node];
    let mut echain = Vec::new();
    let mut cur = node;
    while cur != start {
        let (prev, e) = preds[&cur][0];
        echain.push(e);
        cur = prev;
        chain.push(cur);
    }
    chain.reverse();
    echain.reverse();
    (chain, echain)
}

/// Every distinct shortest path start..node through the predecessor DAG, each as a
/// (node chain start..node, edge chain) pair. Exponential on a wide lattice — the
/// same cost core's `enumerate_shortest_paths` pays; no case in scope hits it.
fn enumerate_shortest_paths(
    node: u32,
    start: u32,
    preds: &FnvMap<u32, Vec<(u32, u32)>>,
) -> Vec<(Vec<u32>, Vec<u32>)> {
    if node == start {
        return vec![(vec![start], Vec::new())];
    }
    let mut out = Vec::new();
    if let Some(ps) = preds.get(&node) {
        for &(prev, e) in ps {
            for (mut chain, mut echain) in enumerate_shortest_paths(prev, start, preds) {
                chain.push(node);
                echain.push(e);
                out.push((chain, echain));
            }
        }
    }
    out
}

/// Append one shortest-path row's lineage: the input row's carried path (ending at
/// `start`) followed by the reconstructed `start..node` chain and its edges.
#[allow(clippy::too_many_arguments)]
/// The four parallel buffers a [`Lineage`] is assembled from, seeded with the
/// leading `0` offset each side needs. Shared by the var-length DFS to accumulate a
/// per-emitted-row path.
struct PathBufs {
    values: Vec<Value>,
    offsets: Vec<usize>,
    edges: Vec<Value>,
    edge_offsets: Vec<usize>,
}

impl PathBufs {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            offsets: vec![0],
            edges: Vec::new(),
            edge_offsets: vec![0],
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_path(
    batch: &Batch,
    row: usize,
    chain: &[u32],
    echain: &[u32],
    path_values: &mut Vec<Value>,
    path_offsets: &mut Vec<usize>,
    path_edges: &mut Vec<Value>,
    path_edge_offsets: &mut Vec<usize>,
) {
    let lin = batch.lineage.as_ref().expect("track");
    path_values.extend_from_slice(lin.path_at(row));
    for &n in &chain[1..] {
        path_values.push(Value::Num(f64::from(n)));
    }
    path_offsets.push(path_values.len());
    path_edges.extend_from_slice(lin.edges_at(row));
    for &e in echain {
        path_edges.push(Value::Num(f64::from(e)));
    }
    path_edge_offsets.push(path_edges.len());
}

/// Elementwise `l OP r` over two already-evaluated columns — the general arithmetic
/// body, shared by `Expr::Arith` and its scalar fast path's non-numeric fallback.
/// Raw f64 when both are `Col::Num`; otherwise per-cell via the value contract (a
/// NULL / non-numeric operand → NULL, a temporal operand → `temporal_arith`). Div/Rem
/// by a zero divisor (the RIGHT operand) throws, matching core's DataException.
fn arith_general(op: crate::ir::ArithOp, l: &Col, r: &Col) -> Result<Col, String> {
    use crate::ir::ArithOp::{Add, Div, Mul, Rem, Sub};
    if let (Col::Num(xs), Col::Num(ys)) = (l, r) {
        let n = xs.len().min(ys.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let (x, y) = (xs[i], ys[i]);
            if matches!(op, Div | Rem) && y == 0.0 {
                return Err("division by zero".into());
            }
            out.push(match op {
                Add => x + y,
                Sub => x - y,
                Mul => x * y,
                Div => x / y,
                Rem => x % y,
            });
        }
        return Ok(Col::Num(out));
    }
    let n = l.len().min(r.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = l.value_at(i);
        let b = r.value_at(i);
        let v = if matches!(a, Value::Temporal(_)) || matches!(b, Value::Temporal(_)) {
            if a.is_null() || b.is_null() {
                Value::Null
            } else {
                temporal_arith(op, &a, &b)?
            }
        } else {
            match (value::num_of(&a), value::num_of(&b)) {
                (Some(x), Some(y)) => {
                    if matches!(op, Div | Rem) && y == 0.0 {
                        return Err("division by zero".into());
                    }
                    Value::Num(match op {
                        Add => x + y,
                        Sub => x - y,
                        Mul => x * y,
                        Div => x / y,
                        Rem => x % y,
                    })
                }
                _ => Value::Null,
            }
        };
        out.push(v);
    }
    Ok(Col::Gen(out))
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
        // `base[index]` — 0-based list element or record/map field. Out of range /
        // negative / non-integer index → NULL; null-safe. Mirrors core.
        //
        // Special case: `nodes(p)[i]` / relationships(p)[i]` (an Index over a path
        // accessor) must keep the ELEMENT typing so a following `.prop` resolves the
        // node/edge property (`edges(p)[0].w`). The path lists carry ids as `Num`,
        // which a generic list-index would flatten to an untyped scalar. Emit a typed
        // `Col::Nodes`/`Col::Edges` instead (out-of-range → `u32::MAX` null sentinel).
        Expr::Index { base, index }
            if matches!(
                base.as_ref(),
                Expr::PathAccess {
                    part: crate::ir::PathPart::Nodes | crate::ir::PathPart::Relationships
                }
            ) =>
        {
            let is_nodes = matches!(
                base.as_ref(),
                Expr::PathAccess {
                    part: crate::ir::PathPart::Nodes
                }
            );
            let icol = eval(index, store, batch)?;
            let ids: Vec<u32> = match &batch.lineage {
                Some(lin) => (0..batch.rows())
                    .map(|i| {
                        let elems = if is_nodes {
                            lin.path_at(i)
                        } else {
                            lin.edges_at(i)
                        };
                        match icol.value_at(i) {
                            Value::Num(n)
                                if n >= 0.0 && n.fract() == 0.0 && (n as usize) < elems.len() =>
                            {
                                match elems[n as usize] {
                                    Value::Num(x) => x as u32,
                                    _ => u32::MAX,
                                }
                            }
                            _ => u32::MAX,
                        }
                    })
                    .collect(),
                None => vec![u32::MAX; batch.rows()],
            };
            if is_nodes {
                Col::Nodes(ids)
            } else {
                Col::Edges(ids)
            }
        }
        Expr::Index { base, index } => {
            let bcol = eval(base, store, batch)?;
            let icol = eval(index, store, batch)?;
            Col::Gen(
                (0..batch.rows())
                    .map(|i| match bcol.value_at(i) {
                        Value::List(items) => match icol.value_at(i) {
                            Value::Num(n)
                                if n >= 0.0 && n.fract() == 0.0 && (n as usize) < items.len() =>
                            {
                                items[n as usize].clone()
                            }
                            _ => Value::Null,
                        },
                        Value::Record(fields) => match icol.value_at(i) {
                            Value::Str(k) => fields
                                .iter()
                                .find(|(fk, _)| *fk == k)
                                .map_or(Value::Null, |(_, v)| v.clone()),
                            _ => Value::Null,
                        },
                        Value::Map(entries) => match icol.value_at(i) {
                            Value::Str(k) => entries
                                .iter()
                                .find(|(ek, _)| matches!(ek, Value::Str(s) if *s == k))
                                .map_or(Value::Null, |(_, v)| v.clone()),
                            _ => Value::Null,
                        },
                        _ => Value::Null,
                    })
                    .collect(),
            )
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
        // Three-valued XOR: both known → `a != b`; any UNKNOWN operand → UNKNOWN.
        Expr::Xor(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(x), Some(y)) => Some(x != y),
            _ => None,
        })?,
        Expr::Compare { op, left, right } => {
            let l = eval(left, store, batch)?;
            let r = eval(right, store, batch)?;
            compare(*op, &l, &r)
        }
        Expr::In { needle, haystack } => {
            // Runtime three-valued membership (a literal list desugars to an
            // OR-chain instead; this matches its semantics). Per row: TRUE if any
            // element equals the needle; else UNKNOWN (NULL) if the needle or any
            // element is null (the answer can't be decided); else FALSE. A
            // non-list haystack is NULL.
            let nd = eval(needle, store, batch)?;
            let hs = eval(haystack, store, batch)?;
            let n = batch.rows();
            let out: Vec<Value> = (0..n)
                .map(|i| {
                    let needle = nd.value_at(i);
                    let Value::List(items) = hs.value_at(i) else {
                        return Value::Null;
                    };
                    let mut saw_unknown = needle.is_null();
                    for el in items.iter() {
                        if el.is_null() || needle.is_null() {
                            saw_unknown = true;
                        } else if value::equals(&needle, el) {
                            return Value::Bool(true);
                        }
                    }
                    if saw_unknown {
                        Value::Null
                    } else {
                        Value::Bool(false)
                    }
                })
                .collect();
            Col::Gen(out)
        }
        Expr::Arith { op, left, right } => {
            // f64 math via the value contract's `as_num` (finite Num only); any
            // NULL / non-numeric / non-finite operand OR result yields NULL. When
            // either operand is a temporal, `temporal_arith` takes over (and may
            // THROW on a result out of the representable range).
            use crate::ir::ArithOp::{Add, Div, Mul, Rem, Sub};
            // Scalar-literal fast path: `col OP num` / `num OP col`. Evaluate ONLY the
            // non-literal operand and fold the constant into the loop — never
            // materializing an n-length broadcast column for the literal. A chain like
            // `age * 2 + 1` then costs one gather + two scalar passes instead of two
            // 8 MB constant columns plus a boxed intermediate; at 1M that alloc traffic
            // was the whole gap (proj/arith 0.55x). Semantics match the general arm
            // below: div/rem by a zero DIVISOR throws (the divisor is the RIGHT
            // operand), every other f64 result is kept.
            let lit_num = |e: &Expr| match e {
                Expr::Lit(Value::Num(t)) if t.is_finite() => Some(*t),
                _ => None,
            };
            let scalar = match (lit_num(left), lit_num(right)) {
                (_, Some(t)) => Some((t, false)), // col OP num (num is the divisor)
                (Some(t), None) => Some((t, true)), // num OP col (col is the divisor)
                _ => None,
            };
            if let Some((t, num_on_left)) = scalar {
                let other = if num_on_left { right } else { left };
                let col = eval(other, store, batch)?;
                if let Col::Num(xs) = col {
                    let mut out = Vec::with_capacity(xs.len());
                    if matches!(op, Div | Rem) && num_on_left {
                        // num OP col → the COLUMN is the divisor; a zero cell throws.
                        for &x in &xs {
                            if x == 0.0 {
                                return Err("division by zero".into());
                            }
                            out.push(if matches!(op, Div) { t / x } else { t % x });
                        }
                    } else if matches!(op, Div | Rem) {
                        // col OP num → the LITERAL is the divisor; throw once if zero.
                        if t == 0.0 {
                            return Err("division by zero".into());
                        }
                        for &x in &xs {
                            out.push(if matches!(op, Div) { x / t } else { x % t });
                        }
                    } else {
                        for &x in &xs {
                            let (a, b) = if num_on_left { (t, x) } else { (x, t) };
                            out.push(match op {
                                Add => a + b,
                                Sub => a - b,
                                Mul => a * b,
                                _ => unreachable!(),
                            });
                        }
                    }
                    return Ok(Col::Num(out));
                }
                // The non-literal side is not a raw Num column (a null / boxed / temporal
                // operand): reuse the evaluated `col` and a broadcast literal through the
                // general loop rather than re-evaluating.
                let lit_col = broadcast(Value::Num(t), col.len());
                let (l, r) = if num_on_left {
                    (lit_col, col)
                } else {
                    (col, lit_col)
                };
                return arith_general(*op, &l, &r);
            }
            let l = eval(left, store, batch)?;
            let r = eval(right, store, batch)?;
            return arith_general(*op, &l, &r);
        }
        Expr::Call { name, args } => {
            // `element_id(node|edge)` → the element's PRESERVED external id string.
            if name == "element_id" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                            store.node_ext_id(id as u32).map_or(Value::Null, Value::Str)
                        }
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => store
                            .edge_ext_id(eid as u32)
                            .map_or(Value::Null, Value::Str),
                        _ => Value::Null,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `type(edge)` needs the store + the edge identity (an eid), so it is
            // handled here (off the evaluated arg column), not in `call_scalar`.
            if name == "type" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => store
                            .edge_type_name(eid as u32)
                            .map_or(Value::Null, |t| Value::Str(t.into())),
                        _ => Value::Null,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `element_label(node|edge)` → a SINGLE label string (Gremlin `label()`):
            // a vertex's label, an edge's type. Not user-callable from GQL (which has
            // list-valued `labels()` and `type()`); emitted only by the Gremlin
            // front-end. A vertex with several labels yields the first in the store's
            // canonical (sorted) order, consistent with GQL `labels()`; a vertex with
            // no label yields Null.
            if name == "element_label" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                            let mut ls = store.labels_of(id as u32);
                            ls.sort();
                            ls.into_iter()
                                .next()
                                .map_or(Value::Null, |l| Value::Str(l.into()))
                        }
                        Value::Num(eid) if matches!(arg, Col::Edges(_)) => store
                            .edge_type_name(eid as u32)
                            .map_or(Value::Null, |t| Value::Str(t.into())),
                        _ => Value::Null,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `element_map(element[, 'k1', …])` → Gremlin `elementMap()`: core's FLAT
            // shape — `{id, label, <props…>}` for a node, plus `IN`/`OUT` endpoint
            // stubs for an edge — where `label` is SINGULAR (the first label / edge
            // type) and the present properties are flattened alongside the tokens
            // (so a property named `id`/`label` would shadow one; that's the lossy
            // flat form, distinct from the nested `{id, labels, properties}` render).
            // An optional trailing key list filters the properties. Gremlin-only.
            if name == "element_map" {
                let filter: Vec<String> = args[1..]
                    .iter()
                    .filter_map(|e| match e {
                        Expr::Lit(Value::Str(s)) => Some(s.to_string()),
                        _ => None,
                    })
                    .collect();
                // The first (sorted) label of a node, or an edge's type.
                let node_label = |id: u32| -> Value {
                    let mut ls = store.labels_of(id);
                    ls.sort();
                    ls.into_iter()
                        .next()
                        .map_or(Value::Null, |l| Value::Str(l.into()))
                };
                let node_id = |id: u32| store.node_ext_id(id).map_or(Value::Null, Value::Str);
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let mut entries: Vec<(Value, Value)> = Vec::new();
                        match arg.value_at(i) {
                            Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                                let id = id as u32;
                                entries.push((Value::Str("id".into()), node_id(id)));
                                entries.push((Value::Str("label".into()), node_label(id)));
                                let keys = if filter.is_empty() {
                                    store.prop_keys()
                                } else {
                                    filter.clone()
                                };
                                let mut props: Vec<(String, Value)> = keys
                                    .into_iter()
                                    .filter(|k| store.has_prop(id, k))
                                    .map(|k| {
                                        let v = store.prop(id, &k);
                                        (k, v)
                                    })
                                    .collect();
                                props.sort_by(|a, b| a.0.cmp(&b.0));
                                for (k, v) in props {
                                    entries.push((Value::Str(k.into()), v));
                                }
                            }
                            Value::Num(eid) if matches!(arg, Col::Edges(_)) => {
                                let eid = eid as u32;
                                entries.push((
                                    Value::Str("id".into()),
                                    store.edge_ext_id(eid).map_or(Value::Null, Value::Str),
                                ));
                                entries.push((
                                    Value::Str("label".into()),
                                    store
                                        .edge_type_name(eid)
                                        .map_or(Value::Null, |t| Value::Str(t.into())),
                                ));
                                if let Some((src, dst)) = store.edge_endpoints(eid) {
                                    let stub = |v: u32| {
                                        Value::Map(Arc::new(vec![
                                            (Value::Str("id".into()), node_id(v)),
                                            (Value::Str("label".into()), node_label(v)),
                                        ]))
                                    };
                                    // Core: IN is the destination, OUT the source.
                                    entries.push((Value::Str("IN".into()), stub(dst)));
                                    entries.push((Value::Str("OUT".into()), stub(src)));
                                }
                                let keys = if filter.is_empty() {
                                    store.edge_prop_keys()
                                } else {
                                    filter.clone()
                                };
                                let mut props: Vec<(String, Value)> = keys
                                    .into_iter()
                                    .filter(|k| store.has_edge_prop(eid, k))
                                    .map(|k| {
                                        let v = store.edge_prop(eid, &k);
                                        (k, v)
                                    })
                                    .collect();
                                props.sort_by(|a, b| a.0.cmp(&b.0));
                                for (k, v) in props {
                                    entries.push((Value::Str(k.into()), v));
                                }
                            }
                            _ => return Value::Null,
                        }
                        Value::Map(Arc::new(entries))
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `value_map(element[, 'k1', …])` → Gremlin `valueMap()`: a Value::Map of
            // the element's PRESENT properties (no id/label tokens), with SCALAR
            // values (core's `propertyMap()`, not built here, is the list-wrapped
            // form). An optional trailing key list filters; no keys = every present
            // property. Keys are sorted (the engine's element-map convention; map key
            // order is set-based per policy). Gremlin-only — not in the GQL whitelist.
            if name == "value_map" {
                // The filter keys are constant string literals after the element arg.
                let filter: Vec<String> = args[1..]
                    .iter()
                    .filter_map(|e| match e {
                        Expr::Lit(Value::Str(s)) => Some(s.to_string()),
                        _ => None,
                    })
                    .collect();
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let mut pairs: Vec<(String, Value)> = match arg.value_at(i) {
                            Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                                let id = id as u32;
                                let keys = if filter.is_empty() {
                                    store.prop_keys()
                                } else {
                                    filter.clone()
                                };
                                keys.into_iter()
                                    .filter(|k| store.has_prop(id, k))
                                    .map(|k| {
                                        let v = store.prop(id, &k);
                                        (k, v)
                                    })
                                    .collect()
                            }
                            Value::Num(eid) if matches!(arg, Col::Edges(_)) => {
                                let eid = eid as u32;
                                let keys = if filter.is_empty() {
                                    store.edge_prop_keys()
                                } else {
                                    filter.clone()
                                };
                                keys.into_iter()
                                    .filter(|k| store.has_edge_prop(eid, k))
                                    .map(|k| {
                                        let v = store.edge_prop(eid, &k);
                                        (k, v)
                                    })
                                    .collect()
                            }
                            _ => return Value::Null,
                        };
                        pairs.sort_by(|a, b| a.0.cmp(&b.0));
                        Value::Map(Arc::new(
                            pairs
                                .into_iter()
                                .map(|(k, v)| (Value::Str(k.into()), v))
                                .collect(),
                        ))
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `path_nodes(path)` → Gremlin `path()` over a vertex-hop chain: render
            // each node id in the lineage path as its element map, so the path is a
            // list of vertex elements (not bare ids). The argument is `Expr::Path`
            // (a per-row list of node-id Nums); a Null row (no lineage) stays Null.
            // Gremlin-only — not in the GQL whitelist.
            if name == "path_nodes" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(ids) => Value::List(
                            ids.into_iter()
                                .map(|v| match v {
                                    Value::Num(id) => node_result_value(store, id as u32),
                                    other => other,
                                })
                                .collect(),
                        ),
                        other => other,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `path_has_dup(path)` → Gremlin `cyclicPath`/`simplePath` support: TRUE if
            // the lineage node path repeats any vertex, FALSE if all distinct. The
            // argument is `Expr::Path` (a per-row list of node-id Nums); a Null row
            // (no lineage) is Null. Gremlin-only — not in the GQL whitelist.
            if name == "path_has_dup" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(ids) => {
                            let mut seen: std::collections::HashSet<u64> =
                                std::collections::HashSet::new();
                            let dup = ids.iter().any(|v| match v {
                                Value::Num(id) => !seen.insert(id.to_bits()),
                                _ => false,
                            });
                            Value::Bool(dup)
                        }
                        other => other,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // `list_{sum,mean,min,max}(list)` → Gremlin's scope-LOCAL aggregates over
            // a list cell (e.g. after `fold()`): reduce the list's NUMERIC elements
            // (nulls/non-numerics skipped), yielding Null for a list with no number —
            // matching core's `local_num`/`local_extreme` on the numeric case.
            // Gremlin-only. (Mixed numeric+non-numeric lists are the held cross-type
            // territory; here the non-numerics are simply skipped.)
            // `list_count(list)` → Gremlin `count(local)`: the number of local
            // elements (a list's length, or 1 for a scalar cell — core's
            // `local_elems(v).len()`). Gremlin-only.
            if name == "list_count" {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        Value::List(items) => Value::Num(items.len() as f64),
                        _ => Value::Num(1.0),
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            if matches!(
                name.as_str(),
                "list_sum" | "list_mean" | "list_min" | "list_max"
            ) {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| {
                        let nums: Vec<f64> = match arg.value_at(i) {
                            Value::List(items) => items
                                .iter()
                                .filter_map(|v| match v {
                                    Value::Num(x) => Some(*x),
                                    _ => None,
                                })
                                .collect(),
                            // A scalar cell is a one-element local list.
                            Value::Num(x) => vec![x],
                            _ => Vec::new(),
                        };
                        if nums.is_empty() {
                            return Value::Null;
                        }
                        let v = match name.as_str() {
                            "list_sum" => nums.iter().sum(),
                            "list_mean" => nums.iter().sum::<f64>() / nums.len() as f64,
                            "list_min" => nums.iter().copied().fold(f64::INFINITY, f64::min),
                            _ => nums.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        };
                        Value::Num(v)
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // Element functions need the STORE and the element identity (a node/edge
            // slot), which the pure-value `call_scalar` cannot see — handle them
            // here off the evaluated argument column.
            if matches!(name.as_str(), "keys" | "labels" | "property_names") {
                let arg = eval(&args[0], store, batch)?;
                let n = batch.rows();
                let out: Vec<Value> = (0..n)
                    .map(|i| match arg.value_at(i) {
                        // A node surfaces as Num(id); its keys / property_names are
                        // the SORTED present property keys, its labels the SORTED
                        // labels — both as string lists (matching core).
                        Value::Num(id) if matches!(arg, Col::Nodes(_)) => {
                            let id = id as u32;
                            let mut items: Vec<Value> = if name == "labels" {
                                let mut ls = store.labels_of(id);
                                ls.sort();
                                ls.into_iter().map(|l| Value::Str(l.into())).collect()
                            } else {
                                store
                                    .prop_keys()
                                    .into_iter()
                                    .filter(|k| store.has_prop(id, k))
                                    .map(|k| Value::Str(k.into()))
                                    .collect()
                            };
                            items.sort_by(value::cmp_total);
                            Value::List(items)
                        }
                        _ => Value::Null,
                    })
                    .collect();
                return Ok(Col::Gen(out));
            }
            // Vectorized unary numeric functions that map finite→finite over a raw
            // `Num` column stay a `Num` column (no per-row boxing), so a downstream
            // aggregate/compare keeps the f64 fast path — e.g. `sum(abs(x - k))`.
            if args.len() == 1 {
                if let Some(f) = unary_finite_num_fn(name) {
                    if let Col::Num(xs) = eval(&args[0], store, batch)? {
                        return Ok(Col::Num(xs.iter().map(|&x| f(x)).collect()));
                    }
                    // A non-`Num` arg (nulls / mixed) falls through to the boxed path.
                }
            }
            // Evaluate each argument to a column, then dispatch per row. Arity is
            // validated at parse time, so `call_scalar` can index its args. The row
            // count is the BATCH's, not the min over args — a niladic function
            // (`pi()`, `e()`) has no arg columns yet still yields one value per row.
            let cols = eval_all(args, store, batch)?;
            let n = batch.rows();
            let out = (0..n)
                .map(|i| {
                    let row: Vec<Value> = cols.iter().map(|c| c.value_at(i)).collect();
                    call_scalar_checked(name, &row)
                })
                .collect::<Result<Vec<Value>, String>>()?;
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
        Expr::CountSubquery { body, .. } => {
            // Correlated count: same provenance-tagged sub-run as EXISTS, but TALLY
            // the sub-rows per outer row instead of a boolean any().
            let n = batch.rows();
            let prov = batch.slots.len();
            let mut slots = batch.slots.clone();
            slots.push(Col::Num((0..n).map(|i| i as f64).collect()));
            let seed = Batch::of(slots);
            let survivors = pull_body(body, store, &seed)?;
            let mut counts = vec![0f64; n];
            if let Col::Num(ids) = survivors.slot(prov) {
                for &id in ids {
                    let i = id as usize;
                    if i < n {
                        counts[i] += 1.0;
                    }
                }
            }
            Col::Num(counts)
        }
    })
}

/// Evaluate an `EXISTS` body against a correlated `seed` batch (the outer rows
/// plus a provenance column). The body is a chain of the operators an EXISTS
/// pattern can contain — `Expand`/`VarLength`/`Filter` — rooted at `Plan::Row`,
/// which yields `seed`. Every operator gathers the whole input row, so the
/// provenance column rides through untouched; the caller reads it off the result.
/// Concatenate several same-shaped batches row-wise (Gremlin `union`'s reconverge).
/// Each output slot is the type-preserving concatenation of that slot across the
/// batches — so a column that is `Col::Nodes` in every branch stays a node frontier
/// (continuable), falling back to `Col::Gen` only when the branch column types
/// differ. Empty input → an empty batch.
fn concat_batches(subs: &[Batch]) -> Batch {
    let Some(first) = subs.first() else {
        return Batch::of(Vec::new());
    };
    let ncols = first.slots.len();
    let cols: Vec<Col> = (0..ncols)
        .map(|j| concat_cols(&subs.iter().map(|b| b.slot(j)).collect::<Vec<_>>()))
        .collect();
    Batch::of(cols)
}

/// Concatenate columns of (ideally) the same variant. Same variant → keep it and
/// extend the inner vector; mixed variants → materialize every value into `Gen`.
fn concat_cols(cols: &[&Col]) -> Col {
    macro_rules! same {
        ($variant:ident) => {{
            let mut v = Vec::new();
            for c in cols {
                if let Col::$variant(xs) = c {
                    v.extend(xs.iter().cloned());
                } else {
                    return Col::Gen(
                        cols.iter()
                            .flat_map(|c| (0..c.len()).map(|i| c.value_at(i)))
                            .collect(),
                    );
                }
            }
            Col::$variant(v)
        }};
    }
    match cols.first() {
        None => Col::Gen(Vec::new()),
        Some(Col::Nodes(_)) => same!(Nodes),
        Some(Col::Edges(_)) => same!(Edges),
        Some(Col::Num(_)) => same!(Num),
        Some(Col::Bool(_)) => same!(Bool),
        Some(Col::Str(_)) => same!(Str),
        Some(Col::Gen(_)) => Col::Gen(
            cols.iter()
                .flat_map(|c| (0..c.len()).map(|i| c.value_at(i)))
                .collect(),
        ),
    }
}

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
            edge_label,
            *bind_edge,
        ),
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            mode,
        } => var_length(
            &pull_body(input, store, seed)?,
            store,
            *from,
            *dir,
            edge_label,
            *min,
            *max,
            *mode,
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
        // A projection is streamable too (used by the LIMIT short-circuit driver;
        // EXISTS bodies never contain one). Evaluate the items over the sub-frontier.
        Plan::Project { input, items } => {
            let b = pull_body(input, store, seed)?;
            let cols = eval_all(items.iter().map(|(_, e)| e), store, &b)?;
            let mut out = Batch::of(cols);
            out.lineage = b.lineage;
            out
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
/// The fallible wrapper around [`call_scalar`]: nearly every scalar function is
/// total, but a temporal component accessor of a kind that lacks that component
/// (`_year` of a time, `_hour` of a date) FAULTS with `E_INVALID_VALUE` — matching
/// core, which returns an error there rather than NULL. A non-temporal / null arg
/// still yields NULL (nullish propagation), not a fault.
fn call_scalar_checked(name: &str, args: &[Value]) -> Result<Value, String> {
    if matches!(
        name,
        "year"
            | "month"
            | "day"
            | "hour"
            | "minute"
            | "second"
            | "_year"
            | "_month"
            | "_day"
            | "_hour"
            | "_minute"
            | "_second"
    ) {
        if let Value::Temporal(t) = &args[0] {
            return date_part(name.trim_start_matches('_'), *t)
                .map(|n| Value::Num(n as f64))
                .ok_or_else(|| {
                    format!(
                        "E_INVALID_VALUE: {} is undefined for this temporal kind",
                        name.trim_start_matches('_')
                    )
                });
        }
    }
    Ok(call_scalar(name, args))
}

fn call_scalar(name: &str, args: &[Value]) -> Value {
    match name {
        // variadic
        "coalesce" => args
            .iter()
            .find(|v| !v.is_null())
            .cloned()
            .unwrap_or(Value::Null),
        // `x IS [NOT] TYPED <type> [NOT NULL]` desugars here: args are (value,
        // category, not_null). A NULL value conforms to any nullable type (so it is
        // `!not_null`); else the value's runtime type must match the category —
        // replicated from core's `category_matches`/`value_is_typed_ty`.
        "__is_typed" => {
            let v = &args[0];
            let category = match &args[1] {
                Value::Str(s) => s.as_ref(),
                _ => return Value::Null,
            };
            let not_null = matches!(args[2], Value::Bool(true));
            if v.is_null() {
                return Value::Bool(!not_null);
            }
            let ok = match category {
                "any" => true,
                "null" => false, // v is non-null here
                "bool" => matches!(v, Value::Bool(_)),
                "string" => matches!(v, Value::Str(_)),
                "integer" => matches!(v, Value::Num(n) if n.is_finite() && n.fract() == 0.0),
                "float" => matches!(v, Value::Num(_)),
                "list" => matches!(v, Value::List(_)),
                "record" => matches!(v, Value::Record(_)),
                "date" | "local_time" | "local_datetime" | "zoned_time" | "zoned_datetime"
                | "duration" => {
                    use crate::temporal::TemporalKind as K;
                    if let Value::Temporal(t) = v {
                        let want = match category {
                            "date" => K::Date,
                            "local_time" => K::Time,
                            "local_datetime" => K::DateTime,
                            "zoned_time" => K::ZonedTime,
                            "zoned_datetime" => K::ZonedDateTime,
                            _ => K::Duration,
                        };
                        t.kind() == want
                    } else {
                        false
                    }
                }
                _ => false,
            };
            Value::Bool(ok)
        }
        // `a || b || …` — left-associative concat (the parser folds a `||` run into
        // one call). Matches core's `concat_step` fold: ANY null operand → NULL; two
        // lists concatenate element-wise; otherwise both sides JS-string-coerce (via
        // `to_string_fn`) and join.
        "concat" => {
            let mut acc = args.first().cloned().unwrap_or(Value::Null);
            for r in &args[1..] {
                acc = concat_step(&acc, r);
            }
            acc
        }
        // numeric constants (0 args)
        "e" => Value::Num(std::f64::consts::E),
        "pi" => Value::Num(std::f64::consts::PI),
        // numeric (1 arg)
        "abs" | "sign" | "floor" | "ceil" | "ceiling" | "sqrt" | "exp" | "ln" | "log10" | "sin"
        | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "cot"
        | "degrees" | "radians" => scalar_num_fn(name, &args[0]),
        // `round(x)` rounds to an integer; `round(x, digits)` to `digits` decimal
        // places (negative rounds left of the point). Half away from zero, matching
        // core; the `(x*f).round()/f` form is bit-identical (do not reformulate).
        "round" => match value::num_of(&args[0]) {
            Some(x) => {
                let digits = args
                    .get(1)
                    .and_then(value::num_of)
                    .map_or(0, |d| d.trunc() as i32);
                let f = 10f64.powi(digits);
                Value::Num((x * f).round() / f)
            }
            None => Value::Null,
        },
        // numeric (2 args). `log(a, b)` is log-base-a of b = ln(b)/ln(a) (matches
        // core's argument order); `mod` is the fn form of `%` (NaN on a zero
        // divisor — it does NOT throw like the `%` OPERATOR, which core reserves for
        // the operator); `atan2(y, x)` is the two-argument arctangent. NaN/Inf
        // results are KEPT (K4), coerced only at JSON egress.
        "log" | "power" | "mod" | "atan2" => {
            match (value::num_of(&args[0]), value::num_of(&args[1])) {
                (Some(x), Some(y)) => Value::Num(match name {
                    "log" => y.ln() / x.ln(),
                    "power" => x.powf(y),
                    "atan2" => x.atan2(y),
                    _ => x % y,
                }),
                _ => Value::Null,
            }
        }
        // nullif(a, b): NULL when a == b (value-contract equality), else a.
        "nullif" => {
            if !args[0].is_null() && !args[1].is_null() && value::equals(&args[0], &args[1]) {
                Value::Null
            } else {
                args[0].clone()
            }
        }
        // Cast FUNCTIONS: NULL on a failed/inapplicable conversion (unlike `CAST`,
        // which throws — and unlike `CAST`, these do NOT coerce a Bool to a number).
        "to_integer" | "tointeger" => to_number(&args[0], true),
        "to_float" | "tofloat" => to_number(&args[0], false),
        "to_string" | "tostring" => to_string_fn(&args[0]),
        "to_boolean" | "toboolean" => to_boolean_fn(&args[0]),
        // string (1 arg → string/number)
        "upper" => str_map(&args[0], str::to_uppercase),
        "lower" => str_map(&args[0], str::to_lowercase),
        // `trim` is both-sides; a 2nd (char-set) arg from the SQL-spec form is
        // honored by routing through btrim (identical to core's Trim).
        "trim" => trim_fn("btrim", args),
        // ltrim/rtrim/btrim: 1 arg trims WHITESPACE from that side; a 2nd string
        // arg is the set of characters to strip instead.
        "ltrim" | "rtrim" | "btrim" => trim_fn(name, args),
        // reverse is polymorphic: a string reverses by char, a list by element;
        // anything else is NULL (matches core, e.g. reverse(number) → NULL).
        // reverse: a string reverses by UTF-16 unit (JS model — a surrogate pair
        // reversed decodes lossily to U+FFFD, byte-identical to core), a list by
        // element; anything else is NULL.
        "reverse" => match &args[0] {
            Value::Str(s) => {
                let mut units: Vec<u16> = s.encode_utf16().collect();
                units.reverse();
                Value::Str(String::from_utf16_lossy(&units).into())
            }
            Value::List(v) => Value::List(v.iter().rev().cloned().collect()),
            _ => Value::Null,
        },
        // left/right(s, n): the first / last n UTF-16 units (n ≥ len → the whole
        // string; n ≤ 0 → empty).
        "left" | "right" => match (&args[0], value::num_of(&args[1])) {
            (Value::Str(s), Some(k)) => {
                let units = utf16_len(s);
                let take = (k.max(0.0) as usize).min(units);
                let out = if name == "left" {
                    utf16_slice(s, 0, take)
                } else {
                    utf16_slice(s, units - take, take)
                };
                Value::Str(out.into())
            }
            _ => Value::Null,
        },
        // split(s, delim) → a list of substrings. An EMPTY delimiter splits into one
        // element per UTF-16 unit (JS model), matching core — NOT Rust's `split("")`.
        "split" => match (&args[0], &args[1]) {
            (Value::Str(s), Value::Str(d)) => {
                let parts: Vec<Value> = if d.is_empty() {
                    s.encode_utf16()
                        .map(|u| Value::Str(String::from_utf16_lossy(&[u]).into()))
                        .collect()
                } else {
                    s.split(d.as_ref()).map(|p| Value::Str(p.into())).collect()
                };
                Value::List(parts)
            }
            _ => Value::Null,
        },
        // Length of a string in UTF-16 code units (JS `.length` model), matching
        // core; `byte_length`/`octet_length` count UTF-8 bytes.
        "length" | "char_length" | "character_length" => match &args[0] {
            Value::Str(s) => Value::Num(utf16_len(s) as f64),
            _ => Value::Null,
        },
        "byte_length" | "octet_length" => match &args[0] {
            Value::Str(s) => Value::Num(s.len() as f64),
            _ => Value::Null,
        },
        // string predicates (2 args → bool)
        "starts_with" => str_bool(&args[0], &args[1], |s, sub| s.starts_with(sub)),
        "ends_with" => str_bool(&args[0], &args[1], |s, sub| s.ends_with(sub)),
        "contains" => str_bool(&args[0], &args[1], |s, sub| s.contains(sub)),
        // replace(s, from[, to]) — `to` defaults to "" (core); an EMPTY search
        // returns the string unchanged (core), NOT Rust's insert-everywhere.
        "replace" => match (&args[0], &args[1]) {
            (Value::Str(s), Value::Str(f)) => {
                let t = match args.get(2) {
                    Some(Value::Str(t)) => t.to_string(),
                    Some(v) if !v.is_null() => return Value::Null,
                    _ => String::new(),
                };
                if f.is_empty() {
                    Value::Str(s.clone())
                } else {
                    Value::Str(s.replace(f.as_ref(), &t).into())
                }
            }
            _ => Value::Null,
        },
        // substring(s, start[, len]) — ISO 1-based, UTF-16-unit indexed
        "substring" => substring(args),
        // `size` is polymorphic over a collection OR a string (UTF-16 units), like
        // lenke-core; a non-collection non-string is NULL.
        "size" => match &args[0] {
            Value::List(v) => Value::Num(v.len() as f64),
            Value::Str(s) => Value::Num(utf16_len(s) as f64),
            _ => Value::Null,
        },
        "head" => match &args[0] {
            Value::List(v) => v.first().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // tail: all but the first element (empty list → empty).
        "tail" => match &args[0] {
            Value::List(v) => Value::List(v.iter().skip(1).cloned().collect()),
            _ => Value::Null,
        },
        // append(list, x) → the list with x appended.
        "append" => match &args[0] {
            Value::List(v) => {
                let mut out = v.clone();
                out.push(args[1].clone());
                Value::List(out)
            }
            _ => Value::Null,
        },
        // list_contains(list, x) → 1.0 if any element equals x, else 0.0 (a NUMBER,
        // not a bool — matching core; `null` matches `null` via `equals`).
        "list_contains" => match &args[0] {
            Value::List(v) => Value::Num(f64::from(v.iter().any(|e| value::equals(e, &args[1])))),
            _ => Value::Null,
        },
        // list_sort(list, [order], [nullOrder]) — the value contract's total order,
        // reversed for `'desc'`, with absolute null placement (`'first'`/`'last'`,
        // default last). Mirrors ORDER BY / core's compare_sort byte-for-byte. A
        // stored list never holds NaN (it becomes null at ingest), so `is_null`
        // covers every nullish element.
        "list_sort" => {
            match &args[0] {
                Value::List(v) => {
                    let descending = matches!(args.get(1), Some(Value::Str(s)) if s.eq_ignore_ascii_case("desc"));
                    let nulls_first = matches!(args.get(2), Some(Value::Str(s)) if s.eq_ignore_ascii_case("first"));
                    let mut out = v.clone();
                    out.sort_by(|x, y| {
                        use std::cmp::Ordering;
                        match (x.is_null(), y.is_null()) {
                            (true, true) => Ordering::Equal,
                            (true, false) => {
                                if nulls_first {
                                    Ordering::Less
                                } else {
                                    Ordering::Greater
                                }
                            }
                            (false, true) => {
                                if nulls_first {
                                    Ordering::Greater
                                } else {
                                    Ordering::Less
                                }
                            }
                            (false, false) => {
                                let o = value::cmp_total(x, y);
                                if descending {
                                    o.reverse()
                                } else {
                                    o
                                }
                            }
                        }
                    });
                    Value::List(out)
                }
                _ => Value::Null,
            }
        }
        // Set algebra over lists — all DEDUPED (by value equality), matching core.
        // union: a's elements then b's, deduped. intersection: elements of a also
        // in b, deduped. difference: elements of a not in b, deduped.
        "list_union" | "difference" | "intersection" => match (&args[0], &args[1]) {
            (Value::List(a), Value::List(b)) => Value::List(list_set_op(name, a, b)),
            _ => Value::Null,
        },
        // range(start, end[, step]) — INCLUSIVE of both ends; default step 1; a
        // zero step is NULL; a start past end with the wrong sign yields an empty
        // list (matches core).
        "range" => {
            let step = if args.len() == 3 {
                value::as_num(&args[2])
            } else {
                Some(1.0)
            };
            match (value::as_num(&args[0]), value::as_num(&args[1]), step) {
                (Some(a), Some(b), Some(st)) if st != 0.0 => {
                    let (mut cur, mut out) = (a, Vec::new());
                    // Guard the element count so a pathological range can't OOM.
                    while (st > 0.0 && cur <= b) || (st < 0.0 && cur >= b) {
                        out.push(Value::Num(cur));
                        if out.len() > 10_000_000 {
                            break;
                        }
                        cur += st;
                    }
                    Value::List(out)
                }
                _ => Value::Null,
            }
        }
        "last" => match &args[0] {
            Value::List(v) => v.last().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // Temporal component accessors (1 arg → number, or NULL when the component
        // is undefined for that kind, e.g. year() of a time).
        // Core spells these with the leading-underscore extension sigil (`_year`);
        // accept that (parity) plus the bare name (kept as a superset alias).
        "year" | "month" | "day" | "hour" | "minute" | "second" | "_year" | "_month" | "_day"
        | "_hour" | "_minute" | "_second" => match &args[0] {
            Value::Temporal(t) => date_part(name.trim_start_matches('_'), *t)
                .map_or(Value::Null, |n| Value::Num(n as f64)),
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

/// Slice `s` by UTF-16 code UNITS `[start, start+len)` (JS `String.slice` /
/// `.length` model), decoding back to UTF-8. A slice that splits a surrogate pair
/// yields U+FFFD there (lossy) — byte-identical to lenke-core (`utf16_slice`) and
/// the TS engine. The whole string model here counts UTF-16 units, NOT `chars()`,
/// so `size('😀')` is 2 (a surrogate pair), matching core.
fn utf16_slice(s: &str, start: usize, len: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    let end = start.saturating_add(len).min(units.len());
    let start = start.min(end);
    String::from_utf16_lossy(&units[start..end])
}

/// Length of `s` in UTF-16 code units — the JS `.length` model core uses.
fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `substring(s, start[, len])` — ISO/SQL **1-based** start, indexed by UTF-16 code
/// UNIT (matching lenke-core exactly). A `start <= 0` shrinks the window from the
/// front (SQL semantics); an omitted `len` runs to the end. NULL for a null string
/// or start.
fn substring(args: &[Value]) -> Value {
    if args[0].is_null() || args[1].is_null() {
        return Value::Null;
    }
    let Value::Str(s) = &args[0] else {
        return Value::Null;
    };
    // 1-based → 0-based offset; a start <= 0 shrinks the window from the front.
    let zero_start = value::num_of(&args[1]).unwrap_or(0.0) - 1.0;
    let from = zero_start.max(0.0) as usize;
    let count = match args.get(2) {
        Some(z) if !z.is_null() => {
            let end = (zero_start + value::num_of(z).unwrap_or(0.0)).max(0.0) as usize;
            end.saturating_sub(from)
        }
        _ => usize::MAX,
    };
    Value::Str(utf16_slice(s, from, count).into())
}

/// Apply a unary numeric scalar function. A NULL / non-numeric argument yields
/// NULL; a computed NaN/Inf result (e.g. `sqrt(-1)`, `ln(0)`) is KEPT (IEEE, like
/// lenke-core — coerced to null only at JSON egress). `sign(0)` is 0 (unlike
/// `f64::signum`); rounding is f64's round-half-away-from-zero.
/// The finite→finite unary numeric functions, as raw `f64 -> f64` closures that
/// match [`scalar_num_fn`] EXACTLY. Restricted to functions that cannot introduce
/// NaN/Inf from a finite input (`sqrt`/`ln`/`exp`/… can, so they are excluded):
/// the result column then keeps the all-finite invariant of a stored `Num` column,
/// and the vectorized path is byte-identical to the boxed one. `None` = not eligible.
fn unary_finite_num_fn(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        "abs" => f64::abs,
        "floor" => f64::floor,
        "ceil" | "ceiling" => f64::ceil,
        "round" => f64::round,
        "sign" => |x: f64| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
        _ => return None,
    })
}

fn scalar_num_fn(name: &str, v: &Value) -> Value {
    let Some(x) = value::num_of(v) else {
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
        "ceil" | "ceiling" => x.ceil(),
        "round" => x.round(),
        "sqrt" => x.sqrt(),
        // Transcendentals — native libm, matching lenke-core's native build. A
        // domain-invalid result (e.g. `ln(-1)`, `cot(0)`) is NaN/Inf and, for now,
        // falls to NULL through the finite gate below (K4 will KEEP it, like core).
        "exp" => x.exp(),
        "ln" => x.ln(),
        "log10" => x.log10(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "cot" => 1.0 / x.tan(),
        // Multiply-then-divide, NOT `to_degrees`/`to_radians`: the latter pre-round
        // the 180/PI (resp. PI/180) constant and land one ULP off core's byte-exact
        // `(n*180)/PI` / `(n*PI)/180`.
        "degrees" => (x * 180.0) / std::f64::consts::PI,
        "radians" => (x * std::f64::consts::PI) / 180.0,
        _ => return Value::Null, // parser rejects unknown names; defensive
    };
    // NaN/Inf are KEPT (K4) — a computed NaN (`sqrt(-1)`, `ln(-1)`) is a real
    // signal, coerced to null only at the JSON egress boundary, matching core.
    Value::Num(r)
}

/// `to_integer`/`to_float` FUNCTION: Num (truncated for integer) or a parseable
/// string; anything else — INCLUDING a Bool — is NULL (the fn forms do not coerce
/// bools, unlike `CAST`). Matches lenke-core.
fn to_number(v: &Value, integer: bool) -> Value {
    let n = match v {
        Value::Num(x) => *x,
        Value::Str(s) => match s.trim().parse::<f64>() {
            Ok(x) => x,
            Err(_) => return Value::Null,
        },
        _ => return Value::Null,
    };
    if integer {
        if n.is_finite() {
            Value::Num(n.trunc())
        } else {
            Value::Null
        }
    } else {
        Value::Num(n)
    }
}

/// `to_string` FUNCTION: NULL→NULL, finite Num→its egress text, Bool→"true"/
/// "false", Str→itself, Temporal→its ISO form; a non-finite number is NULL.
/// One step of the `||` fold, matching core's `concat_step`: null propagates, two
/// lists concatenate, otherwise both operands JS-string-coerce and join.
fn concat_step(l: &Value, r: &Value) -> Value {
    if l.is_null() || r.is_null() {
        return Value::Null;
    }
    if let (Value::List(a), Value::List(b)) = (l, r) {
        return Value::List(a.iter().chain(b.iter()).cloned().collect());
    }
    match (to_string_fn(l), to_string_fn(r)) {
        (Value::Str(a), Value::Str(b)) => Value::Str(format!("{a}{b}").into()),
        // A non-stringable operand (e.g. a map) → NULL, as core's js_str-of-unknown does.
        _ => Value::Null,
    }
}

fn to_string_fn(v: &Value) -> Value {
    match v {
        Value::Null => Value::Null,
        Value::Str(s) => Value::Str(s.clone()),
        Value::Bool(b) => Value::Str((if *b { "true" } else { "false" }).into()),
        // Finite number as text; -0.0 renders as "0" (matches core — the sign of
        // zero is dropped on string egress).
        Value::Num(x) if x.is_finite() => {
            let s = if *x == 0.0 {
                "0".to_string()
            } else {
                x.to_string()
            };
            Value::Str(s.into())
        }
        Value::Temporal(t) => Value::Str(t.format().into()),
        _ => Value::Null,
    }
}

/// `to_boolean` FUNCTION: a Bool passes through; the strings "true"/"false"
/// (trimmed, case-insensitive) convert; anything else is NULL.
fn to_boolean_fn(v: &Value) -> Value {
    match v {
        Value::Bool(b) => Value::Bool(*b),
        // A number coerces like C truthiness: nonzero → true, zero → false.
        Value::Num(x) => Value::Bool(*x != 0.0),
        Value::Str(s) => {
            let t = s.trim();
            if t.eq_ignore_ascii_case("true") {
                Value::Bool(true)
            } else if t.eq_ignore_ascii_case("false") {
                Value::Bool(false)
            } else {
                Value::Null
            }
        }
        _ => Value::Null,
    }
}

/// Set algebra over two lists, all producing a DEDUPED result (by the value
/// contract's `equals`, so `null` collapses with `null`): `list_union` = a then
/// the b-elements not already present; `intersection` = a-elements also in b;
/// `difference` = a-elements not in b. Order follows first appearance in `a`
/// (then `b` for union). O(n·m) — lists are small.
fn list_set_op(name: &str, a: &[Value], b: &[Value]) -> Vec<Value> {
    let contains = |xs: &[Value], v: &Value| xs.iter().any(|x| value::equals(x, v));
    let mut out: Vec<Value> = Vec::new();
    let push_unique = |out: &mut Vec<Value>, v: &Value| {
        if !contains(out, v) {
            out.push(v.clone());
        }
    };
    match name {
        "intersection" => {
            for v in a {
                if contains(b, v) {
                    push_unique(&mut out, v);
                }
            }
        }
        "difference" => {
            for v in a {
                if !contains(b, v) {
                    push_unique(&mut out, v);
                }
            }
        }
        _ => {
            // union: everything in a, then b's new elements, deduped throughout.
            for v in a.iter().chain(b.iter()) {
                push_unique(&mut out, v);
            }
        }
    }
    out
}

/// `ltrim`/`rtrim`/`btrim`: strip whitespace (1 arg) or a given char set (2 args)
/// from the left / right / both ends of a string. Non-string → NULL.
fn trim_fn(name: &str, args: &[Value]) -> Value {
    let Value::Str(s) = &args[0] else {
        return Value::Null;
    };
    // A 2nd string arg is the set of chars to strip; otherwise strip whitespace.
    let set: Option<Vec<char>> = match args.get(1) {
        None => None,
        Some(Value::Str(cs)) => Some(cs.chars().collect()),
        Some(_) => return Value::Null, // a non-string char set
    };
    let strip = |c: char| {
        set.as_ref()
            .map_or_else(|| c.is_whitespace(), |v| v.contains(&c))
    };
    let trimmed = match name {
        "ltrim" => s.trim_start_matches(strip),
        "rtrim" => s.trim_end_matches(strip),
        _ => s.trim_matches(strip), // btrim
    };
    Value::Str(trimmed.into())
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

/// The negation of a compare operator — `NOT (x op y)` ≡ `x invert_op(op) y` for
/// present, finite operands. Stored Num/Str cells always are (NaN/absent → NULL,
/// gated by `present`), so the raw fast paths keep the exact keep-TRUE semantics.
fn invert_op(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Eq => CompareOp::Ne,
        CompareOp::Ne => CompareOp::Eq,
        CompareOp::Lt => CompareOp::Ge,
        CompareOp::Ge => CompareOp::Lt,
        CompareOp::Gt => CompareOp::Le,
        CompareOp::Le => CompareOp::Gt,
    }
}

/// De Morgan push-down of `NOT` for the keep-TRUE filter: an equivalent positive
/// predicate (`NOT` eliminated) when `e` is built from compares / AND / OR / NOT,
/// else `None`. Exact in Kleene 3-valued logic for "keep rows where TRUE": `NOT e`
/// is TRUE iff `e` is FALSE, and each rule preserves that (`AND` is FALSE iff an
/// operand is FALSE → `OR` of the negations; `>` inverts to `<=`; etc.). Absent /
/// NaN cells stay dropped on both sides because every compare is UNKNOWN there.
fn invert_pred(e: &Expr) -> Option<Expr> {
    Some(match e {
        Expr::Compare { op, left, right } => Expr::Compare {
            op: invert_op(*op),
            left: left.clone(),
            right: right.clone(),
        },
        Expr::And(a, b) => Expr::Or(Box::new(invert_pred(a)?), Box::new(invert_pred(b)?)),
        Expr::Or(a, b) => Expr::And(Box::new(invert_pred(a)?), Box::new(invert_pred(b)?)),
        Expr::Not(inner) => (**inner).clone(),
        _ => return None,
    })
}

/// A typed reader over ONE storage column for the multi-column distinct fast path:
/// it appends a row's grouping-key bytes (byte-identical to
/// [`value::group_key_into`] over the boxed value, so the induced equivalence is the
/// same) and produces the row's output `Value` — both reading the column directly,
/// borrowing a `&str` for the key rather than boxing or cloning per row. A `Dict`
/// column keys on its decoded string, exactly as a `Str` would.
enum ColKeyer<'a> {
    Dict {
        dict: &'a [std::sync::Arc<str>],
        codes: &'a [u32],
        present: &'a [bool],
    },
    Num {
        data: &'a [f64],
        present: &'a [bool],
    },
    Str {
        data: &'a [std::sync::Arc<str>],
        present: &'a [bool],
    },
    Bool {
        data: &'a [bool],
        present: &'a [bool],
    },
}

impl<'a> ColKeyer<'a> {
    /// A keyer for a Num/Str/Bool/Dict column; `None` for Temporal/Gen/missing (which
    /// may carry present-null or need typed compare — left to the general path).
    fn of(col: Option<&'a Column>) -> Option<Self> {
        match col? {
            Column::Dict {
                dict,
                codes,
                present,
            } => Some(Self::Dict {
                dict,
                codes,
                present,
            }),
            Column::Num { data, present } => Some(Self::Num { data, present }),
            Column::Str { data, present } => Some(Self::Str { data, present }),
            Column::Bool { data, present } => Some(Self::Bool { data, present }),
            _ => None,
        }
    }

    /// Append row `i`'s grouping-key bytes. Str/Num/Bool mirror `group_key_into`
    /// tag-for-tag (absent → `0`, bool → `1`, num → `2`, str → `3`). A `Dict` column
    /// instead keys on its `u32` CODE (tag `8`): the dict assigns exactly one code
    /// per distinct string, so two rows share a code iff they share the string —
    /// the same equivalence a string key induces, but 4 bytes and no string hash.
    /// Codes never cross columns (each column keys at its own fixed offset).
    fn key_into(&self, i: usize, out: &mut Vec<u8>) {
        let push_str = |out: &mut Vec<u8>, s: &str| {
            out.push(3);
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        };
        match self {
            Self::Dict { codes, present, .. } => {
                if present[i] {
                    out.push(8);
                    out.extend_from_slice(&codes[i].to_le_bytes());
                } else {
                    out.push(0);
                }
            }
            Self::Str { data, present } => {
                if present[i] {
                    push_str(out, &data[i]);
                } else {
                    out.push(0);
                }
            }
            Self::Num { data, present } => {
                if present[i] {
                    out.push(2);
                    out.extend_from_slice(&value::num_group_bits(data[i]).to_le_bytes());
                } else {
                    out.push(0);
                }
            }
            Self::Bool { data, present } => {
                if present[i] {
                    out.push(1);
                    out.push(u8::from(data[i]));
                } else {
                    out.push(0);
                }
            }
        }
    }

    /// Row `i`'s output value (absent → `Null`). Clones an `Arc` only here — called
    /// once per SURVIVING distinct tuple, not per scanned row.
    fn value_at(&self, i: usize) -> Value {
        match self {
            Self::Dict {
                dict,
                codes,
                present,
            } => {
                if present[i] {
                    Value::Str(dict[codes[i] as usize].clone())
                } else {
                    Value::Null
                }
            }
            Self::Str { data, present } => {
                if present[i] {
                    Value::Str(data[i].clone())
                } else {
                    Value::Null
                }
            }
            Self::Num { data, present } => {
                if present[i] {
                    Value::Num(data[i])
                } else {
                    Value::Null
                }
            }
            Self::Bool { data, present } => {
                if present[i] {
                    Value::Bool(data[i])
                } else {
                    Value::Null
                }
            }
        }
    }
}

/// Fused multi-column `RETURN DISTINCT n.a, n.b, …` over a bare `Scan`: read the
/// storage columns directly and dedup on a composite grouping key, emitting only the
/// distinct tuples (first-seen order) — so the 100k-row projected columns (a `dept`
/// of `Arc<str>` above all) are never materialized and no `Value` is boxed per
/// scanned row. `None` unless the input is a `Project(Scan, [prop, …])` whose every
/// key is a plain (non-dotted) property backed by a Num/Str/Bool/Dict column.
fn try_distinct_scan_multi(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project { input: scan, items } = input else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    if items.is_empty() {
        return None;
    }
    let mut readers: Vec<ColKeyer> = Vec::with_capacity(items.len());
    for (_, e) in items {
        let Expr::Prop { slot: 0, key } = e else {
            return None;
        };
        if key.contains('.') {
            return None; // a dotted record path — leave to the general path
        }
        readers.push(ColKeyer::of(store.column(key))?);
    }

    let ncol = readers.len();
    let mut outs: Vec<Vec<Value>> = vec![Vec::new(); ncol];
    let mut seen: FnvSet<Vec<u8>> = FnvSet::default();
    let mut buf: Vec<u8> = Vec::new();
    scan_visit(store, label, |i| {
        buf.clear();
        for r in &readers {
            r.key_into(i, &mut buf);
        }
        if !seen.contains(buf.as_slice()) {
            seen.insert(buf.clone());
            for (c, r) in readers.iter().enumerate() {
                outs[c].push(r.value_at(i));
            }
        }
    });
    Some(Batch::of(outs.into_iter().map(Col::Gen).collect()))
}

/// One-pass predicate for the common `<prop> <cmp> <literal>` (either operand
/// order) over a node frontier: read the storage property per row and emit the
/// kept row indices, without building a full value column AND a full boolean mask
/// as intermediates. Every comparison goes through the value contract, so results
/// match the general path exactly: an absent property is NULL → UNKNOWN → dropped,
/// a NULL literal makes every comparison UNKNOWN → all dropped, and cross-type is
/// the contract's `equals`/`cmp_total`. `None` if the predicate is not this shape.
/// Fused `RETURN DISTINCT n.k` — a `Distinct` over a `Project(Scan, [one prop])` —
/// reading the storage column directly and deduping to just the distinct values
/// (first-seen order), so the 100k-row projected column is never materialized.
/// Absence is a distinct value (a present-null / missing prop → one `Null` row, as
/// grouping treats it). `None` unless the shape is exactly that over a `Num`/`Str`/
/// `Bool` column.
fn try_distinct_scan_prop(input: &Plan, store: &Store) -> Option<Batch> {
    let Plan::Project { input: scan, items } = input else {
        return None;
    };
    let [(_, Expr::Prop { slot: 0, key })] = items.as_slice() else {
        return None;
    };
    let Plan::Scan { label } = scan.as_ref() else {
        return None;
    };
    let mut out: Vec<Value> = Vec::new();
    let mut saw_null = false;
    match store.column(key)? {
        Column::Str { data, present } => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            scan_visit(store, label, |i| {
                if present[i] {
                    if seen.insert(data[i].as_ref()) {
                        out.push(Value::Str(data[i].clone()));
                    }
                } else if !saw_null {
                    saw_null = true;
                    out.push(Value::Null);
                }
            });
        }
        Column::Dict {
            dict,
            codes,
            present,
        } => {
            // First-seen order is preserved by pushing when a code is first observed
            // during the scan (NOT dict order, which can differ from scan order under
            // deletes / a label subset).
            let mut seen = vec![false; dict.len()];
            scan_visit(store, label, |i| {
                if present[i] {
                    let c = codes[i] as usize;
                    if !std::mem::replace(&mut seen[c], true) {
                        out.push(Value::Str(dict[c].clone()));
                    }
                } else if !saw_null {
                    saw_null = true;
                    out.push(Value::Null);
                }
            });
        }
        Column::Num { data, present } => {
            // Low-card integer fast path: recover the distinct values from a bitset
            // (ascending) instead of hashing every cell. DISTINCT output order is
            // unspecified (compared as a set), so ascending is fine; a NULL is still
            // emitted once if any cell is absent.
            if let Some((lo, bits, saw_absent)) = low_card_int_bitset(store, label, data, present) {
                if saw_absent {
                    out.push(Value::Null);
                }
                for (k, &set) in bits.iter().enumerate() {
                    if set {
                        out.push(Value::Num(lo + k as f64));
                    }
                }
            } else {
                let mut seen: FnvSet<u64> = FnvSet::default();
                scan_visit(store, label, |i| {
                    if present[i] {
                        if seen.insert(value::num_group_bits(data[i])) {
                            out.push(Value::Num(data[i]));
                        }
                    } else if !saw_null {
                        saw_null = true;
                        out.push(Value::Null);
                    }
                });
            }
        }
        Column::Bool { data, present } => {
            let mut seen = [false; 2];
            scan_visit(store, label, |i| {
                if present[i] {
                    let b = data[i];
                    if !std::mem::replace(&mut seen[usize::from(b)], true) {
                        out.push(Value::Bool(b));
                    }
                } else if !saw_null {
                    saw_null = true;
                    out.push(Value::Null);
                }
            });
        }
        _ => return None, // Temporal / Gen → the general Distinct path
    }
    Some(Batch::of(vec![Col::Gen(out)]))
}

/// Row indices of the first occurrence of each distinct value in a SINGLE-column
/// batch, keyed by the raw value (`&str`, f64 group bits, or a dense id) rather
/// than a serialized byte key — the common `RETURN DISTINCT n.k` shape. `None` for
/// a multi-column batch or a `Gen` column (which may hold nulls/mixed types, where
/// the grouping-byte key is needed). First-seen order preserved.
fn try_distinct_typed(batch: &Batch) -> Option<Vec<usize>> {
    let [col] = batch.slots.as_slice() else {
        return None;
    };
    let mut keep = Vec::new();
    match col {
        Col::Str(v) => {
            let mut seen: FnvSet<&str> = FnvSet::default();
            for (i, s) in v.iter().enumerate() {
                if seen.insert(s.as_ref()) {
                    keep.push(i);
                }
            }
        }
        Col::Num(v) => {
            // f64 group bits collapse NaN payloads and signed zero, matching the
            // grouping contract.
            let mut seen: FnvSet<u64> = FnvSet::default();
            for (i, &x) in v.iter().enumerate() {
                if seen.insert(value::num_group_bits(x)) {
                    keep.push(i);
                }
            }
        }
        Col::Nodes(v) | Col::Edges(v) => {
            let mut seen: FnvSet<u32> = FnvSet::default();
            for (i, &id) in v.iter().enumerate() {
                if seen.insert(id) {
                    keep.push(i);
                }
            }
        }
        Col::Bool(v) => {
            let mut seen = [false; 2];
            for (i, &b) in v.iter().enumerate() {
                if !std::mem::replace(&mut seen[usize::from(b)], true) {
                    keep.push(i);
                }
            }
        }
        Col::Gen(_) => return None, // nulls / mixed types → the grouping-byte key
    }
    Some(keep)
}

/// Flatten a conjunction into its atoms (`a AND b AND c` → `[a, b, c]`); a
/// non-`And` expression is a single atom.
fn flatten_and<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::And(a, b) => {
            flatten_and(a, out);
            flatten_and(b, out);
        }
        _ => out.push(e),
    }
}

fn flatten_or<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::Or(a, b) => {
            flatten_or(a, out);
            flatten_or(b, out);
        }
        _ => out.push(e),
    }
}

/// Keep rows satisfying a DISJUNCTION of `prop <op> num-literal` compares, all on
/// the same node slot and all reading `Num` columns — one raw-f64 pass keeping a
/// row when ANY disjunct is TRUE. This is the OR mirror of [`try_num_conjunction`],
/// and it also catches `x IN [a, b, …]`, which the parser desugars to an OR-chain of
/// equalities. 3VL WHERE semantics hold: a disjunct over a NULL/NaN cell is never
/// TRUE, so the row is kept iff some disjunct is definitely TRUE (else FALSE/UNKNOWN
/// → dropped), matching the general `Or` evaluator under `is_true`.
fn try_num_disjunction(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    if !matches!(pred, Expr::Or(..)) {
        return None;
    }
    let mut atoms = Vec::new();
    flatten_or(pred, &mut atoms);
    let mut slot0: Option<usize> = None;
    let mut specs: Vec<(&[f64], &[bool], CompareOp, f64)> = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let Expr::Compare { op, left, right } = atom else {
            return None;
        };
        let (slot, key, op, lit) = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot, key }, Expr::Lit(v)) => (*slot, key, *op, v),
            (Expr::Lit(v), Expr::Prop { slot, key }) => (*slot, key, flip_op(*op), v),
            _ => return None,
        };
        match slot0 {
            Some(s) if s != slot => return None, // all disjuncts on the same slot
            _ => slot0 = Some(slot),
        }
        let Value::Num(t) = lit else { return None };
        let Some(Column::Num { data, present }) = store.column(key) else {
            return None;
        };
        specs.push((data, present, op, *t));
    }
    let Col::Nodes(ids) = batch.slot(slot0?) else {
        return None;
    };
    Some(
        ids.iter()
            .enumerate()
            .filter(|&(_, &id)| {
                let i = id as usize;
                specs
                    .iter()
                    .any(|&(data, present, op, t)| present[i] && num_pred(op, data[i], t))
            })
            .map(|(row, _)| row)
            .collect(),
    )
}

/// Keep rows satisfying a CONJUNCTION of `prop <op> num-literal` compares, all on
/// the same node slot and all reading `Num` columns — one raw-f64 pass over the id
/// list, each conjunct a `num_pred` (a NULL/NaN cell fails its conjunct → the row
/// drops, matching AND's 3VL). `None` unless every atom fits that shape (the
/// caller then tries the single-compare / general paths).
fn try_num_conjunction(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    if !matches!(pred, Expr::And(..)) {
        return None; // a single compare is handled by the caller's typed path
    }
    let mut atoms = Vec::new();
    flatten_and(pred, &mut atoms);
    let mut slot0: Option<usize> = None;
    let mut specs: Vec<(&[f64], &[bool], CompareOp, f64)> = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let Expr::Compare { op, left, right } = atom else {
            return None;
        };
        let (slot, key, op, lit) = match (left.as_ref(), right.as_ref()) {
            (Expr::Prop { slot, key }, Expr::Lit(v)) => (*slot, key, *op, v),
            (Expr::Lit(v), Expr::Prop { slot, key }) => (*slot, key, flip_op(*op), v),
            _ => return None,
        };
        match slot0 {
            Some(s) if s != slot => return None, // all atoms on the same slot
            _ => slot0 = Some(slot),
        }
        let Value::Num(t) = lit else { return None };
        let Some(Column::Num { data, present }) = store.column(key) else {
            return None;
        };
        specs.push((data, present, op, *t));
    }
    let Col::Nodes(ids) = batch.slot(slot0?) else {
        return None;
    };
    // Same-column range (`lo <= x AND x < hi`) — the overwhelmingly common
    // conjunction. Normalize the two bounds to a concrete lower/upper with
    // inclusivity, then run ONE loop of LITERAL f64 comparisons: no per-element `match
    // op` and no runtime spec loop. NaN fails both compares (dropped), matching
    // `num_pred`'s 3VL; `present` gates nulls — byte-identical to the general path
    // below. At 1M this turns range filter+project from 0.68x to 1.10x.
    //
    // The 200k cache-resident `scan/range-and` PROJECTION sits at ~0.85x, and that is
    // projection-bound, not filter-bound: the FILTER, streamed, is 3.67x core (see
    // `try_stream_num_count` — the win was skipping the scan-id materialization, core's
    // trick), but this shape returns 20k names and the ~0.66ms of string projection
    // dominates the ~0.12ms filter, so both engines pay it and the ratio parks near a
    // tie. REJECTED, all measured NEUTRAL at 200k:
    //   - Streaming the projection too (collect survivors by borrowing the bucket, then
    //     project): neutral for the range (projection-bound) and it REGRESSED the
    //     single-compare `scan/gt` 1.05x -> 0.78x (a lone compare loses the vectorized
    //     `try_filter_keep` path for no materialization win).
    //   - Sequential column read when the id list is the contiguous full scan
    //     (`id == row`), to drop the `d0[ids[row]]` gather: no change.
    //   - Mask-then-compact: the default release target has no SIMD gather, so it would
    //     not vectorize either.
    if specs.len() == 2 {
        let (d0, p0, op0, t0) = specs[0];
        let (d1, _, op1, t1) = specs[1];
        if std::ptr::eq(d0.as_ptr(), d1.as_ptr()) {
            let bound = |op, t| match op {
                CompareOp::Ge => Some((true, t, true)), // (is_lower, value, inclusive)
                CompareOp::Gt => Some((true, t, false)),
                CompareOp::Le => Some((false, t, true)),
                CompareOp::Lt => Some((false, t, false)),
                _ => None, // Eq/Ne is not a range bound
            };
            if let (Some((lo_a, va, ia)), Some((lo_b, vb, ib))) = (bound(op0, t0), bound(op1, t1)) {
                // One bound must be lower and the other upper (else e.g. `x>=5 AND x>=10`,
                // not a range — fall through to the general path).
                let lohi = match (lo_a, lo_b) {
                    (true, false) => Some(((va, ia), (vb, ib))),
                    (false, true) => Some(((vb, ib), (va, ia))),
                    _ => None,
                };
                if let Some(((lo, lo_inc), (hi, hi_inc))) = lohi {
                    macro_rules! range_loop {
                        ($lo_cmp:tt, $hi_cmp:tt) => {{
                            let mut keep = Vec::new();
                            for (row, &id) in ids.iter().enumerate() {
                                let i = id as usize;
                                let x = d0[i];
                                if p0[i] && x $lo_cmp lo && x $hi_cmp hi {
                                    keep.push(row);
                                }
                            }
                            keep
                        }};
                    }
                    let keep = match (lo_inc, hi_inc) {
                        (true, true) => range_loop!(>=, <=),
                        (true, false) => range_loop!(>=, <),
                        (false, true) => range_loop!(>, <=),
                        (false, false) => range_loop!(>, <),
                    };
                    return Some(keep);
                }
            }
        }
    }
    Some(
        ids.iter()
            .enumerate()
            .filter(|&(_, &id)| {
                let i = id as usize;
                specs
                    .iter()
                    .all(|&(data, present, op, t)| present[i] && num_pred(op, data[i], t))
            })
            .map(|(row, _)| row)
            .collect(),
    )
}

fn try_filter_keep(pred: &Expr, store: &Store, batch: &Batch) -> Option<Vec<usize>> {
    // `NOT p` pushes into the raw fast paths by inverting `p` (De Morgan + operator
    // flip), exact for the keep-TRUE filter. If the inverted form is not itself
    // fast-pathable, this returns None and the caller evaluates the original `NOT`
    // through the general (boxed) path.
    if let Expr::Not(inner) = pred {
        return invert_pred(inner).and_then(|pos| try_filter_keep(&pos, store, batch));
    }
    // A CONJUNCTION of typed-numeric `prop <op> literal` compares on one node slot
    // (e.g. `age >= 30 AND age < 40`) keeps rows satisfying ALL, in one raw-f64
    // pass — no per-cell boxing, and no falling to the general And evaluator.
    if let Some(keep) = try_num_conjunction(pred, store, batch) {
        return Some(keep);
    }
    // The OR mirror — `age < 5 OR age > 95`, and `age IN [1, 2, …]` (an OR-chain).
    if let Some(keep) = try_num_disjunction(pred, store, batch) {
        return Some(keep);
    }
    // A string-search predicate `col STARTS WITH / ENDS WITH / CONTAINS lit` (which
    // desugars to a `starts_with`/`ends_with`/`contains` call) over a raw Str/Dict
    // column — scan `&str` directly, no per-cell `Value` boxing through
    // `call_scalar` (the same win the Str compare path gets). Semantics match
    // `str_bool`: a present string cell tests, an absent/NULL cell is UNKNOWN →
    // dropped; a non-string column has no match and falls to the general path.
    if let Expr::Call { name, args } = pred {
        if let (
            test @ ("starts_with" | "ends_with" | "contains"),
            [Expr::Prop { slot, key }, Expr::Lit(Value::Str(sub))],
        ) = (name.as_str(), args.as_slice())
        {
            if let Col::Nodes(ids) = batch.slot(*slot) {
                let f: fn(&str, &str) -> bool = match test {
                    "starts_with" => |s, t| s.starts_with(t),
                    "ends_with" => |s, t| s.ends_with(t),
                    _ => |s, t| s.contains(t),
                };
                let sub = sub.as_ref();
                let mut keep = Vec::new();
                match store.column(key) {
                    Some(Column::Str { data, present }) => {
                        for (row, &id) in ids.iter().enumerate() {
                            let i = id as usize;
                            if present[i] && f(data[i].as_ref(), sub) {
                                keep.push(row);
                            }
                        }
                        return Some(keep);
                    }
                    Some(Column::Dict {
                        dict,
                        codes,
                        present,
                    }) => {
                        for (row, &id) in ids.iter().enumerate() {
                            let i = id as usize;
                            if present[i] && f(dict[codes[i] as usize].as_ref(), sub) {
                                keep.push(row);
                            }
                        }
                        return Some(keep);
                    }
                    // Column absent everywhere → every cell UNKNOWN → dropped.
                    None => return Some(Vec::new()),
                    // A non-string column: `str_bool` yields NULL → no match.
                    _ => return Some(Vec::new()),
                }
            }
        }
    }
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
    let Some(column) = store.column(key) else {
        return Some(Vec::new()); // property absent everywhere → UNKNOWN → all dropped
    };
    let mut keep = Vec::new();
    // Typed fast path: a Num column vs a Num literal compares RAW f64 — no per-cell
    // `Value` boxing (the eval-vs-columnar cost). Semantics match the general
    // `compare`: ordering is 3VL (a NaN cell is unordered → dropped, via `<`/`>`
    // being false on NaN); equality via `==`/`!=`.
    if let (Column::Num { data, present }, Value::Num(t)) = (column, lit) {
        let t = *t;
        for (row, &id) in ids.iter().enumerate() {
            let i = id as usize;
            if !present[i] {
                continue; // NULL → UNKNOWN → dropped
            }
            let x = data[i];
            let hit = match op {
                CompareOp::Eq => x == t,
                CompareOp::Ne => x != t,
                CompareOp::Lt => x < t,
                CompareOp::Le => x <= t,
                CompareOp::Gt => x > t,
                CompareOp::Ge => x >= t,
            };
            if hit {
                keep.push(row);
            }
        }
        return Some(keep);
    }
    // Typed fast path: a Str column vs a Str literal compares `&str` directly — no
    // per-cell `Value` boxing. `=`/`<>` are byte equality (== `value::equals`);
    // ordering is lexicographic (== `cmp_partial` for two strings). A NULL cell is
    // gated by `present`; a NULL literal was handled above.
    if let (Column::Str { data, present }, Value::Str(t)) = (column, lit) {
        let t = t.as_ref();
        for (row, &id) in ids.iter().enumerate() {
            let i = id as usize;
            if !present[i] {
                continue; // NULL → UNKNOWN → dropped
            }
            let x = data[i].as_ref();
            let hit = match op {
                CompareOp::Eq => x == t,
                CompareOp::Ne => x != t,
                CompareOp::Lt => x < t,
                CompareOp::Le => x <= t,
                CompareOp::Gt => x > t,
                CompareOp::Ge => x >= t,
            };
            if hit {
                keep.push(row);
            }
        }
        return Some(keep);
    }
    // General path (Bool/Temporal/Gen columns): read the cell, then compare via the
    // value contract. Ordering uses `cmp_partial` (3VL — cross-type/NaN → drop,
    // matching `compare`), NOT the total order.
    for (row, &id) in ids.iter().enumerate() {
        let v = column.read(id as usize);
        if v.is_null() {
            continue;
        }
        let hit = match op {
            CompareOp::Eq => value::equals(&v, lit),
            CompareOp::Ne => !value::equals(&v, lit),
            _ => match value::cmp_partial(&v, lit) {
                Some(o) => match op {
                    CompareOp::Lt => o.is_lt(),
                    CompareOp::Le => o.is_le(),
                    CompareOp::Gt => o.is_gt(),
                    CompareOp::Ge => o.is_ge(),
                    _ => unreachable!("Eq/Ne handled above"),
                },
                None => continue, // incomparable → UNKNOWN → dropped
            },
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
    // An edge slot reads an EDGE property (boxed map, keyed by eid). A `u32::MAX`
    // eid is the OPTIONAL null sentinel → NULL.
    if let Col::Edges(eids) = col {
        // Fastest path: the typed numeric overlay — read `data[eid]` as a raw f64 with
        // NO per-edge hash probe (the boxed `map.get` below). Only when every edge has
        // a present value (the null sentinel `u32::MAX` indexes past `present`, so it
        // fails the check and falls through to the null-carrying general column).
        if let Some((data, present)) = store.edge_num_column(key) {
            if eids
                .iter()
                .all(|&e| (e as usize) < present.len() && present[e as usize])
            {
                return Col::Num(eids.iter().map(|&e| data[e as usize]).collect());
            }
        }
        // Fast path: a fully-present NUMERIC edge property → a raw `Col::Num`, so the
        // downstream compare / aggregate hits the unboxed f64 path (the same win the
        // node columns already get). One outer hash lookup for `key`, then a probe
        // per edge; bail to the boxed `Gen` path the moment any edge is missing the
        // key, is the OPTIONAL null sentinel (`u32::MAX`, absent from the map), or is
        // non-numeric — those need the null-carrying general column.
        if let Some(map) = store.edge_prop_map(key) {
            let mut nums = Vec::with_capacity(eids.len());
            let ok = eids.iter().all(|&e| match map.get(&e) {
                Some(Value::Num(x)) => {
                    nums.push(*x);
                    true
                }
                _ => false,
            });
            if ok {
                return Col::Num(nums);
            }
        }
        return Col::Gen(
            eids.iter()
                .map(|&e| {
                    if e == u32::MAX {
                        Value::Null
                    } else {
                        store.edge_prop(e, key)
                    }
                })
                .collect(),
        );
    }
    // A node column carrying any OPTIONAL null sentinel reads per row (sentinel →
    // NULL, else the stored property); u32::MAX would index the property column out
    // of bounds on the fast path below.
    if let Col::Nodes(ids) = col {
        if ids.contains(&u32::MAX) {
            return Col::Gen(
                ids.iter()
                    .map(|&id| {
                        if id == u32::MAX {
                            Value::Null
                        } else {
                            store.prop(id, key)
                        }
                    })
                    .collect(),
            );
        }
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
        Column::Dict {
            dict,
            codes,
            present,
        } if ids.iter().all(|&i| present[i as usize]) => Col::Str(
            ids.iter()
                .map(|&i| dict[codes[i as usize] as usize].clone())
                .collect(),
        ),
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

    /// K4: computed NaN/Inf are KEPT in the result value (matching lenke-core, so a
    /// caller can detect the signal), and coerced to null only at JSON egress.
    #[test]
    fn nan_and_inf_kept_in_results_coerced_at_egress() {
        let mut b = Builder::default();
        b.node(&["N"], &[("a", n(-4.0))]);
        let store = b.build();
        let val = |e: &str| {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        assert!(matches!(val("sqrt(x.a)"), Value::Num(y) if y.is_nan())); // sqrt(-4) → NaN kept
        assert!(matches!(val("sqrt(x.a) + 1"), Value::Num(y) if y.is_nan())); // NaN propagates
        assert!(matches!(val("power(10, 400)"), Value::Num(y) if y.is_infinite())); // overflow → Inf
                                                                                    // But the JSON egress renders both as null (no JSON form for NaN/Inf).
        let ndjson = crate::ndjson::to_ndjson(&store);
        assert!(!ndjson.contains("NaN") && !ndjson.to_lowercase().contains("inf"));
    }

    /// Newly added scalar functions (K6 casts, K8 nullif, K9 math/constants,
    /// K5 size-on-string) match hand-computed values. One node with a=4, b="Carol".
    #[test]
    fn added_scalar_functions() {
        let mut b = Builder::default();
        b.node(&["N"], &[("a", n(4.0)), ("b", s("Carol"))]);
        let store = b.build();
        let val = |e: &str| -> Value {
            let q = format!("MATCH (n:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        let num = |e: &str| match val(e) {
            Value::Num(x) => x,
            o => panic!("{e} → {o:?}"),
        };
        // constants + math (native libm, matching core)
        assert!((num("pi()") - std::f64::consts::PI).abs() < 1e-12);
        assert!((num("e()") - std::f64::consts::E).abs() < 1e-12);
        assert_eq!(num("power(2, 10)"), 1024.0);
        assert_eq!(num("log(2, 8)"), 3.0); // log base 2 of 8 = ln(8)/ln(2)
        assert_eq!(num("mod(7, 3)"), 1.0);
        assert!((num("ln(e())") - 1.0).abs() < 1e-12);
        assert_eq!(num("degrees(pi())").round(), 180.0);
        // casts (NULL on failure; no bool→number coercion)
        assert_eq!(num("to_integer('7')"), 7.0);
        assert_eq!(num("to_integer(4.9)"), 4.0);
        assert_eq!(num("to_float('2.5')"), 2.5);
        assert!(matches!(val("to_string(n.a)"), Value::Str(x) if &*x == "4"));
        assert!(matches!(val("to_boolean('true')"), Value::Bool(true)));
        assert!(matches!(val("to_boolean(0)"), Value::Bool(false)));
        assert!(val("to_integer('nope')").is_null());
        assert!(val("to_integer(true)").is_null()); // fn form does NOT coerce bool
                                                    // nullif
        assert!(val("nullif(n.a, 4)").is_null());
        assert_eq!(num("nullif(n.a, 5)"), 4.0);
        // size / char_length on a string (K5)
        assert_eq!(num("size(n.b)"), 5.0);
        assert_eq!(num("char_length(n.b)"), 5.0);
    }

    /// Subscript `base[index]` (ISO 0-based) over a list literal, record and map:
    /// in-range element, out-of-range / negative / non-integer → NULL, null-safe.
    #[test]
    fn subscript_list_record_map() {
        let mut b = Builder::default();
        b.node(&["N"], &[("z", n(1.0))]);
        let store = b.build();
        let num = |e: &str| -> f64 {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            match run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("{e} → {o:?}"),
            }
        };
        let isnull = |e: &str| -> bool {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].is_null()
        };
        assert_eq!(num("[10,20,30][0]"), 10.0);
        assert_eq!(num("[10,20,30][2]"), 30.0);
        assert!(isnull("[10,20,30][9]")); // out of range
        assert!(isnull("[10,20,30][-1]")); // negative
        assert!(isnull("[10,20,30][1.5]")); // non-integer
        assert_eq!(num("{a:1,b:2}['b']"), 2.0); // record field by string key
        assert!(isnull("{a:1,b:2}['zzz']")); // missing field
    }

    /// `edges(p)[i]` / `nodes(p)[i]` keep element typing so a following `.prop`
    /// resolves the edge/node property. Path n0 -R(w=5)-> n1 -R(w=7)-> n2.
    #[test]
    fn subscript_path_element_property() {
        let nd = concat!(
            "{\"id\":\"n0\",\"labels\":[\"N\"],\"props\":{\"id\":\"n0\"}}\n",
            "{\"id\":\"n1\",\"labels\":[\"N\"],\"props\":{\"id\":\"n1\"}}\n",
            "{\"id\":\"n2\",\"labels\":[\"N\"],\"props\":{\"id\":\"n2\"}}\n",
            "{\"id\":\"e1\",\"from\":\"n0\",\"to\":\"n1\",\"type\":\"R\",\"props\":{\"w\":5}}\n",
            "{\"id\":\"e2\",\"from\":\"n1\",\"to\":\"n2\",\"type\":\"R\",\"props\":{\"w\":7}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let val = |e: &str| -> Value {
            let q = format!(
                "MATCH p = ANY SHORTEST (a:N {{id:'n0'}})-[:R]->*(b:N {{id:'n2'}}) RETURN {e} AS v"
            );
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        assert!(matches!(val("edges(p)[0].w"), Value::Num(x) if x == 5.0)); // first edge property
        assert!(matches!(val("edges(p)[1].w"), Value::Num(x) if x == 7.0)); // second edge property
        assert!(val("edges(p)[9].w").is_null()); // out-of-range edge → NULL prop
        assert!(matches!(val("nodes(p)[2].id"), Value::Str(x) if &*x == "n2"));
        assert!(val("nodes(p)[0].nope").is_null()); // missing node property
        assert!(matches!(
            val("edges(p)[1].w > edges(p)[0].w"),
            Value::Bool(true)
        ));
    }

    /// Multi-label edges: an edge's type is its FIRST label; the rest are secondary
    /// labels a `-[:label]->` hop must still match. `a-[:X,:Y]->b`, `a-[:Y]->c`.
    #[test]
    fn multi_label_edge_matching() {
        // a -r0[X,Y]-> b ; a -r1[Y]-> c ; b -r2[Z,Y]-> c
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"r0\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"X\",\"Y\"],\"props\":{}}\n",
            "{\"id\":\"r1\",\"from\":\"a\",\"to\":\"c\",\"labels\":[\"Y\"],\"props\":{}}\n",
            "{\"id\":\"r2\",\"from\":\"b\",\"to\":\"c\",\"labels\":[\"Z\",\"Y\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        assert!(store.has_multi_label_edges());
        let ids = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        // `:Y` reaches every edge (all three carry Y, two only as a secondary label).
        assert_eq!(
            ids("MATCH (a:N)-[:Y]->(b) RETURN b.id AS x"),
            vec!["b", "c", "c"]
        );
        // `:X` only r0 (its primary), `:Z` only r2 (its primary).
        assert_eq!(ids("MATCH (a:N)-[:X]->(b) RETURN b.id AS x"), vec!["b"]);
        assert_eq!(ids("MATCH (a:N)-[:Z]->(b) RETURN b.id AS x"), vec!["c"]);
        // A var-length `:Y` hop crosses secondary-label edges too: a-Y->b-Y->c.
        assert!(
            ids("MATCH (a:N {id:'a'})-[:Y]->{2}(b) RETURN b.id AS x").contains(&"c".to_string())
        );
        // type(edge) is the FIRST label, not a secondary one.
        let ty = |q: &str| -> Vec<String> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v = names_of(&run(&plan, &store), 0);
            v.sort();
            v
        };
        assert_eq!(
            ty("MATCH (a:N)-[e:Y]->(b) RETURN type(e) AS t"),
            vec!["X", "Y", "Z"]
        );
    }

    /// An explicit `GROUP BY` after the RETURN list parses and groups the same as
    /// the implicit (non-aggregate items are the keys). n=1,1,2 over three P nodes.
    #[test]
    fn explicit_group_by_after_return() {
        let mut b = Builder::default();
        b.node(&["P"], &[("n", n(1.0))]);
        b.node(&["P"], &[("n", n(1.0))]);
        b.node(&["P"], &[("n", n(2.0))]);
        let store = b.build();
        // GROUP BY the underlying expression, ORDER BY the alias then the expr.
        let rows = |q: &str| -> Vec<(f64, f64)> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            run(&plan, &store)
                .rows
                .iter()
                .map(|r| match (&r[0], &r[1]) {
                    (Value::Num(a), Value::Num(c)) => (*a, *c),
                    o => panic!("{o:?}"),
                })
                .collect()
        };
        assert_eq!(
            rows("MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY a"),
            vec![(1.0, 2.0), (2.0, 1.0)]
        );
        assert_eq!(
            rows("MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY u.n"),
            vec![(1.0, 2.0), (2.0, 1.0)]
        );
    }

    /// LIMIT 0 yields the empty result WITHOUT evaluating the projection, so a
    /// faulting expression (`1/0`) under LIMIT 0 does not error (matches core).
    #[test]
    fn limit_zero_short_circuits_before_projection() {
        let mut b = Builder::default();
        b.node(&["T"], &[("x", n(3.0))]);
        let store = b.build();
        // Without LIMIT 0, `1/0` faults; with it, the projection is never reached.
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (n:T) RETURN 1/0 AS x LIMIT 0").unwrap(),
            &store,
        );
        let out = try_run(&plan, &store).expect("LIMIT 0 must not fault");
        assert_eq!(out.rows.len(), 0);
        // DISTINCT … LIMIT 0 too.
        let plan = crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (n:T) RETURN DISTINCT 1/0 AS x LIMIT 0").unwrap(),
            &store,
        );
        assert_eq!(try_run(&plan, &store).unwrap().rows.len(), 0);
    }

    /// A named path over a NON-shortest var-length pattern binds the walk lineage,
    /// so path_length(p)/edges(p)/nodes(p) resolve. Fixture a->b->c->a (cycle) + a->d.
    #[test]
    fn named_path_over_var_length_binds_lineage() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"c\",\"to\":\"a\",\"labels\":[\"R\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let run_q = |q: &str, col: usize| -> Vec<f64> {
            let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
            let mut v: Vec<f64> = run(&plan, &store)
                .rows
                .iter()
                .map(|r| match r[col] {
                    Value::Num(x) => x,
                    ref o => panic!("{o:?}"),
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        // paths from a of length 1..3: a-b (1), a-d (1), a-b-c (2), a-b-c-a (3).
        assert_eq!(
            run_q(
                "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN path_length(p) AS len",
                0
            ),
            vec![1.0, 1.0, 2.0, 3.0]
        );
        // size(edges(p)) tracks the hop count (path_length).
        assert_eq!(
            run_q(
                "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN size(edges(p)) AS es",
                0
            ),
            vec![1.0, 1.0, 2.0, 3.0]
        );
        // min 0 binds the length-0 seed path (a itself) too.
        assert_eq!(
            run_q(
                "MATCH p = (a:N {id:'a'})-[:R]->{0,1}(x) RETURN path_length(p) AS len",
                0
            ),
            vec![0.0, 1.0, 1.0]
        );
    }

    /// String (K10) and list/element (K11) functions match hand-computed values.
    #[test]
    fn added_string_and_list_functions() {
        let mut b = Builder::default();
        b.node(&["N", "M"], &[("z", n(1.0)), ("a", n(2.0))]);
        let store = b.build();
        let val = |e: &str| -> Value {
            let q = format!("MATCH (x:N) RETURN {e} AS v");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        let str_of = |e: &str| match val(e) {
            Value::Str(s) => s.to_string(),
            o => panic!("{e} → {o:?}"),
        };
        // `Value` has no `PartialEq` (the value contract owns equality), so compare
        // list contents via debug strings.
        let list_of = |e: &str| -> Vec<String> {
            match val(e) {
                Value::List(v) => v.iter().map(|x| format!("{x:?}")).collect(),
                o => panic!("{e} → {o:?}"),
            }
        };
        let dbg = |xs: &[Value]| -> Vec<String> { xs.iter().map(|x| format!("{x:?}")).collect() };
        // trims (whitespace, and explicit char set)
        assert_eq!(str_of("ltrim('  hi ')"), "hi ");
        assert_eq!(str_of("rtrim('  hi ')"), "  hi");
        assert_eq!(str_of("btrim('xxhixx', 'x')"), "hi");
        // reverse (string + list), left/right, split
        assert_eq!(str_of("reverse('abc')"), "cba");
        assert_eq!(str_of("left('abcd', 2)"), "ab");
        assert_eq!(str_of("right('abcd', 2)"), "cd");
        assert_eq!(str_of("left('ab', 5)"), "ab"); // n > len → whole
        assert_eq!(
            list_of("split('a,b,c', ',')"),
            dbg(&[s("a"), s("b"), s("c")])
        );
        // list fns
        assert_eq!(
            list_of("reverse([1, 2, 3])"),
            dbg(&[n(3.0), n(2.0), n(1.0)])
        );
        assert_eq!(list_of("tail([1, 2, 3])"), dbg(&[n(2.0), n(3.0)]));
        assert_eq!(
            list_of("range(1, 4)"),
            dbg(&[n(1.0), n(2.0), n(3.0), n(4.0)])
        );
        assert_eq!(list_of("range(5, 1, -1)").len(), 5);
        assert!(val("range(1, 4, 0)").is_null()); // zero step
        assert_eq!(list_of("range(5, 1)"), Vec::<String>::new()); // wrong-sign default step
                                                                  // element fns: keys (sorted present props), labels (sorted)
        assert_eq!(list_of("keys(x)"), dbg(&[s("a"), s("z")]));
        assert_eq!(list_of("labels(x)"), dbg(&[s("M"), s("N")]));
    }

    /// `IN` / `NOT IN` over a list literal (K7), desugared to an OR-chain of
    /// equals — including three-valued behavior with a NULL in the list.
    #[test]
    fn in_operator() {
        let mut b = Builder::default();
        b.node(&["N"], &[("a", n(1.0))]);
        b.node(&["N"], &[("a", n(2.0))]);
        b.node(&["N"], &[("a", n(9.0))]);
        b.node(&["N"], &[]); // a is NULL
        let store = b.build();
        let ids = |q: &str| -> Vec<String> {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
            v.sort();
            v
        };
        // a IN [1,2] → the 1 and 2 nodes.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN [1, 2] RETURN n.a AS a"),
            vec!["Num(1.0)", "Num(2.0)"]
        );
        // NOT IN → the 9 node only (NULL-a is UNKNOWN, dropped, not returned).
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a NOT IN [1, 2] RETURN n.a AS a"),
            vec!["Num(9.0)"]
        );
        // A NULL element makes a non-match UNKNOWN → row drops (3VL): only the
        // literal 1 matches; 2/9 are UNKNOWN (could equal the null), dropped.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN [1, null] RETURN n.a AS a"),
            vec!["Num(1.0)"]
        );
        // Empty list → nobody matches.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN [] RETURN n.a AS a"),
            Vec::<String>::new()
        );
    }

    /// Dynamic (non-literal) IN over a list PROPERTY — the runtime `Expr::In`, with
    /// the same three-valued behavior as the literal OR-chain.
    #[test]
    fn in_operator_dynamic() {
        let mut b = Builder::default();
        b.node(
            &["N"],
            &[
                ("a", n(2.0)),
                ("xs", Value::List(vec![n(1.0), n(2.0), n(3.0)])),
            ],
        );
        b.node(
            &["N"],
            &[
                ("a", n(9.0)),
                ("xs", Value::List(vec![n(1.0), n(2.0), n(3.0)])),
            ],
        );
        b.node(
            &["N"],
            &[
                ("a", n(5.0)),
                ("xs", Value::List(vec![n(1.0), Value::Null, n(3.0)])),
            ],
        );
        let store = b.build();
        let ids = |q: &str| -> Vec<String> {
            let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
            v.sort();
            v
        };
        // n.a IN n.xs: only the a=2 node (2 ∈ [1,2,3]); a=9 not in; a=5 vs [1,null,3]
        // is UNKNOWN (null element) → dropped.
        assert_eq!(
            ids("MATCH (n:N) WHERE n.a IN n.xs RETURN n.a AS a"),
            vec!["Num(2.0)"]
        );
        // 2 IN n.xs: the two nodes whose list has 2; the [1,null,3] node lacks 2 and
        // is UNKNOWN → dropped.
        assert_eq!(
            ids("MATCH (n:N) WHERE 2 IN n.xs RETURN n.a AS a"),
            vec!["Num(2.0)", "Num(9.0)"]
        );
    }

    /// Undirected `~` traversal is `Dir::Both`: a normal edge is reached from both
    /// endpoints (two rows), but a self-loop is walked ONCE (its in-side copy is
    /// dropped), matching core's `SelfLoops::Once`.
    #[test]
    fn undirected_tilde_self_loop_counted_once() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"a\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // Self-loop once (1) + a-b both orientations (2) = 3.
        assert_eq!(count("MATCH (a)~[r]~(b) RETURN count(*) AS c"), 3.0);
        // The same over a single-hop var-length spelling routes through the DFS
        // walker, which also drops the self-loop's in-side copy.
        assert_eq!(count("MATCH (a)~[:R]~{1,1}(b) RETURN count(*) AS c"), 3.0);
        // A directed self-loop is walked once either way (one index touched).
        assert_eq!(count("MATCH (a)-[r:R]->(b) RETURN count(*) AS c"), 2.0);
    }

    /// `SELECT … GROUP BY … HAVING …` filters grouped rows: on an aggregate
    /// (`count(*) > 1`), on a group key (`n.age >= 35`), globally (no GROUP BY), and
    /// `HAVING null` drops every group. An aggregate may appear only in HAVING.
    #[test]
    fn select_having() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{\"age\":30}}\n",
            "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{\"age\":30}}\n",
            "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{\"age\":40}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let rows = |q: &str| -> Vec<String> {
            run(&crate::gql::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| {
                    r.iter()
                        .map(|c| format!("{c:?}"))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .collect()
        };
        // HAVING on an aggregate: only the age-30 group (count 2 > 1).
        assert_eq!(
            rows("SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) GROUP BY n.age HAVING count(*) > 1 ORDER BY age"),
            vec!["Num(30.0),Num(2.0)"]
        );
        // Aggregate only in HAVING, not in the SELECT list.
        assert_eq!(
            rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING count(*) > 1"),
            vec!["Num(30.0)"]
        );
        // HAVING on a group key.
        assert_eq!(
            rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING n.age >= 35 ORDER BY age"),
            vec!["Num(40.0)"]
        );
        // Global HAVING (no GROUP BY): 3 people — passes >2, fails >100.
        assert_eq!(
            rows("SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 2"),
            vec!["Num(3.0)"]
        );
        assert!(
            rows("SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 100").is_empty()
        );
        // HAVING null drops every group.
        assert!(
            rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING null").is_empty()
        );
    }

    /// `ALL SHORTEST` emits one row per distinct shortest path (so a target reached
    /// by two equal-length paths appears twice), while `ANY SHORTEST` emits one row
    /// per reachable target. A `*` quantifier includes the zero-length seed.
    #[test]
    fn all_shortest_multiplicity() {
        // Diamond: a->b, a->c, b->d, c->d — d is reachable by two 2-hop paths.
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"c\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e3\",\"from\":\"b\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e4\",\"from\":\"c\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // ANY: seed a (len 0) + b + c + d(once) = 4 rows.
        assert_eq!(
            count("MATCH ANY SHORTEST (a {id:'a'})-[:R]->*(x) RETURN count(*) AS c"),
            4.0
        );
        // ALL: a + b + c + d TWICE (two shortest paths) = 5 rows.
        assert_eq!(
            count("MATCH ALL SHORTEST (a {id:'a'})-[:R]->*(x) RETURN count(*) AS c"),
            5.0
        );
        // ALL restricted to endpoint d: two shortest paths → 2 rows.
        assert_eq!(
            count("MATCH ALL SHORTEST (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
            2.0
        );
        // SHORTEST 1 reduces to ANY (one row for d); SHORTEST 1 GROUP to ALL (two).
        assert_eq!(
            count("MATCH SHORTEST 1 (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
            1.0
        );
        assert_eq!(
            count("MATCH SHORTEST 1 GROUP (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
            2.0
        );
    }

    /// `SELECT … [FROM MATCH …]` is sugar for MATCH…RETURN: a constant projection
    /// with no FROM, a plain projection, a global aggregate with WHERE, and a
    /// GROUP BY (via implicit grouping) with ORDER BY over an output alias.
    #[test]
    fn select_from_match() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Alice\",\"age\":30}}\n",
            "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Bob\",\"age\":40}}\n",
            "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Cara\",\"age\":30}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let one =
            |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
        // Constant projection, no FROM.
        assert!(matches!(one("SELECT 1 + 2 AS v"), Value::Num(n) if n == 3.0));
        // Plain projection with an inline filter.
        assert!(
            matches!(one("SELECT n.name AS nm FROM MATCH (n:Person {name: 'Alice'})"), Value::Str(s) if &*s == "Alice")
        );
        // Global aggregate with WHERE (>= 30 → all three).
        assert!(
            matches!(one("SELECT count(*) AS c FROM MATCH (n:Person) WHERE n.age >= 30"), Value::Num(n) if n == 3.0)
        );
        // GROUP BY age with ORDER BY the output alias: ages 30 (×2), 40 (×1).
        let grouped = run(
            &crate::gql::parse(
                "SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) GROUP BY n.age ORDER BY age",
            )
            .unwrap(),
            &store,
        );
        let rows: Vec<String> = grouped
            .rows
            .iter()
            .map(|r| format!("{:?},{:?}", r[0], r[1]))
            .collect();
        assert_eq!(rows, vec!["Num(30.0),Num(2.0)", "Num(40.0),Num(1.0)"]);
    }

    /// Scalar functions: 2-arg round (incl. negative digits), atan2 (arg order +
    /// null propagation), log10, TRIM spec forms, and list_sort with order/nullOrder.
    #[test]
    fn scalar_fns_batch() {
        let store =
            crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
        let val =
            |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
        let num = |q: &str| -> f64 {
            match val(q) {
                Value::Num(n) => n,
                other => panic!("want num, got {other:?}"),
            }
        };
        // round to N decimal places; negative digits round left of the point.
        assert_eq!(num("RETURN round(1.2345, 2) AS r"), 1.23);
        assert_eq!(num("RETURN round(1234.5678, -2) AS r"), 1200.0);
        assert_eq!(num("RETURN round(2.5) AS r"), 3.0); // 1-arg still works
                                                        // atan2(y, x): arg order matters; a null arg → NULL.
        assert!((num("RETURN atan2(1, 1) AS r") - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert_eq!(num("RETURN atan2(0, 1) AS r"), 0.0);
        assert!(matches!(val("RETURN atan2(null, 1) AS r"), Value::Null));
        // log10.
        assert_eq!(num("RETURN log10(1000) AS r"), 3.0);
        // TRIM spec forms desugar to trim/ltrim/rtrim with the char as 2nd arg.
        let s = |q: &str| -> String {
            match val(q) {
                Value::Str(x) => x.to_string(),
                other => panic!("want str, got {other:?}"),
            }
        };
        assert_eq!(s("RETURN TRIM('  hi  ') AS r"), "hi");
        assert_eq!(s("RETURN TRIM(BOTH FROM '  hi  ') AS r"), "hi");
        assert_eq!(s("RETURN TRIM(LEADING 'x' FROM 'xxhi') AS r"), "hi");
        assert_eq!(s("RETURN TRIM(TRAILING 'x' FROM 'hixx') AS r"), "hi");
        assert_eq!(s("RETURN TRIM('x' FROM 'xxhixx') AS r"), "hi");
        // list_sort: default ascending, 'desc' reverses, nullOrder places nulls.
        // Compare list results by their debug rendering (Value is not PartialEq).
        let list = |q: &str| -> String { format!("{:?}", val(q)) };
        assert_eq!(
            list("RETURN list_sort([3,1,2], 'desc') AS r"),
            "List([Num(3.0), Num(2.0), Num(1.0)])"
        );
        assert_eq!(
            list("RETURN list_sort([3,1,null,2], 'asc', 'first') AS r"),
            "List([Null, Num(1.0), Num(2.0), Num(3.0)])"
        );
        // default null placement is LAST.
        assert_eq!(
            list("RETURN list_sort([2,null,1]) AS r"),
            "List([Num(1.0), Num(2.0), Null])"
        );
    }

    /// An edge-type disjunction `-[:A|B]->` matches an edge whose type is ANY of the
    /// listed types; a typed-but-all-unknown disjunction matches nothing (it is NOT
    /// read as "any"); an unknown name in a partial disjunction is dropped.
    #[test]
    fn edge_type_disjunction() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"KNOWS\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"c\",\"type\":\"CREATED\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // Both edge types match → both neighbours.
        assert_eq!(
            count("MATCH (a)-[:KNOWS|CREATED]->(x) RETURN count(*) AS c"),
            2.0
        );
        // Order is irrelevant to the set.
        assert_eq!(
            count("MATCH (a)-[:CREATED|KNOWS]->(x) RETURN count(*) AS c"),
            2.0
        );
        // A single named type still matches only that one.
        assert_eq!(count("MATCH (a)-[:KNOWS]->(x) RETURN count(*) AS c"), 1.0);
        // A partial disjunction drops the unknown name, keeping the known one.
        assert_eq!(
            count("MATCH (a)-[:KNOWS|BOGUS]->(x) RETURN count(*) AS c"),
            1.0
        );
        // Typed but ALL-unknown matches nothing (NOT read as "any type").
        assert_eq!(
            count("MATCH (a)-[:BOGUS|NOPE]->(x) RETURN count(*) AS c"),
            0.0
        );
    }

    /// `MATCH WALK` lets a variable-length hop reuse an edge; `TRAIL` (the default)
    /// forbids it. Over a self-loop, a length-2 hop exists as a WALK (reuse the loop)
    /// but not as a TRAIL.
    #[test]
    fn path_mode_walk_vs_trail_edge_reuse() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"a\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let count = |q: &str| -> f64 {
            match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        // WALK: a->a->a reuses the loop edge — one length-2 walk.
        assert_eq!(
            count("MATCH WALK (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
            1.0
        );
        // TRAIL (default): the loop edge can't repeat — no length-2 trail.
        assert_eq!(
            count("MATCH TRAIL (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
            0.0
        );
        assert_eq!(
            count("MATCH (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
            0.0
        );
    }

    /// `~` resolves to `Dir::Both` regardless of which side (or a `-`/`~` mix) is
    /// used, matching either traversal direction of the edge.
    #[test]
    fn undirected_tilde_matches_either_direction() {
        let nd = concat!(
            "{\"id\":\"josh\",\"labels\":[\"P\"],\"props\":{\"name\":\"josh\"}}\n",
            "{\"id\":\"vadas\",\"labels\":[\"P\"],\"props\":{\"name\":\"vadas\"}}\n",
            "{\"id\":\"e1\",\"from\":\"josh\",\"to\":\"vadas\",\"type\":\"KNOWS\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        // josh has an OUT edge; the undirected walk still reaches vadas.
        let mut a = names_of(
            &run(
                &crate::gql::parse(
                    "MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'josh' RETURN b.name AS n",
                )
                .unwrap(),
                &store,
            ),
            0,
        );
        a.sort();
        assert_eq!(a, vec!["vadas"]);
        // vadas has only an IN edge; the undirected walk reaches josh.
        let b = names_of(
            &run(
                &crate::gql::parse(
                    "MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'vadas' RETURN b.name AS n",
                )
                .unwrap(),
                &store,
            ),
            0,
        );
        assert_eq!(b, vec!["josh"]);
    }

    /// External ids are PRESERVED through ingest and returned by element_id (nodes
    /// and edges), and survive an NDJSON round-trip.
    #[test]
    fn element_id_preserves_external_ids() {
        let nd = concat!(
            "{\"id\":\"alice\",\"labels\":[\"P\"],\"props\":{}}\n",
            "{\"id\":\"bob\",\"labels\":[\"P\"],\"props\":{}}\n",
            "{\"id\":\"e42\",\"from\":\"alice\",\"to\":\"bob\",\"type\":\"KNOWS\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        // element_id(node) returns the preserved string id.
        let mut ns = names_of(
            &run(
                &crate::gql::parse("MATCH (n:P) RETURN element_id(n) AS a0").unwrap(),
                &store,
            ),
            0,
        );
        ns.sort();
        assert_eq!(ns, vec!["alice", "bob"]);
        // element_id(edge) returns the preserved edge id.
        let es = run(
            &crate::gql::parse("MATCH (a:P)-[r:KNOWS]->(b) RETURN element_id(r) AS a0").unwrap(),
            &store,
        );
        assert!(matches!(&es.rows[0][0], Value::Str(s) if &**s == "e42"));
        // NDJSON round-trip preserves those ids (dump contains them, reload keeps).
        let dump = crate::ndjson::to_ndjson(&store);
        assert!(dump.contains("\"id\":\"alice\"") && dump.contains("\"id\":\"e42\""));
        assert_eq!(
            crate::ndjson::to_ndjson(&crate::ndjson::from_ndjson(&dump).unwrap()),
            dump
        );
    }

    /// `type(edge)` and the list-algebra functions (previously deferred) match
    /// hand-computed values.
    #[test]
    fn type_and_list_algebra_functions() {
        let mut b = Builder::default();
        let x = b.node(&["N"], &[]);
        let y = b.node(&["N"], &[]);
        b.edge(x, y, "KNOWS");
        let store = b.build();
        // type(edge)
        let t = run(
            &crate::gql::parse("MATCH (a:N)-[r:KNOWS]->(b) RETURN type(r) AS t").unwrap(),
            &store,
        );
        assert!(matches!(&t.rows[0][0], Value::Str(s) if &**s == "KNOWS"));

        let list = |e: &str| -> Vec<String> {
            let q = format!("MATCH (a:N) RETURN {e} AS v LIMIT 1");
            match &run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0] {
                Value::List(v) => v.iter().map(|x| format!("{x:?}")).collect(),
                o => panic!("{e} → {o:?}"),
            }
        };
        let one = |e: &str| -> Value {
            let q = format!("MATCH (a:N) RETURN {e} AS v LIMIT 1");
            run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
        };
        let dbg = |xs: &[Value]| -> Vec<String> { xs.iter().map(|x| format!("{x:?}")).collect() };
        assert_eq!(list("append([1, 2], 3)"), dbg(&[n(1.0), n(2.0), n(3.0)]));
        assert!(matches!(one("list_contains([1, 2, 3], 2)"), Value::Num(x) if x == 1.0));
        assert!(matches!(one("list_contains([1, 2], 5)"), Value::Num(x) if x == 0.0));
        assert!(matches!(one("list_contains([1, null], null)"), Value::Num(x) if x == 1.0));
        assert_eq!(list("list_sort([3, 1, 2])"), dbg(&[n(1.0), n(2.0), n(3.0)]));
        assert_eq!(
            list("list_union([1, 1, 2], [2, 3])"),
            dbg(&[n(1.0), n(2.0), n(3.0)])
        );
        assert_eq!(
            list("difference([1, 1, 2, 3], [2])"),
            dbg(&[n(1.0), n(3.0)])
        );
        assert_eq!(
            list("intersection([1, 2, 2, 3], [2, 3, 4])"),
            dbg(&[n(2.0), n(3.0)])
        );
    }

    /// Late materialization (sorted top-K over a Project) returns the SAME rows as
    /// the eager path — the non-key column is projected only for the survivors.
    #[test]
    fn late_materialize_top_k_matches_eager() {
        let mut b = Builder::default();
        for i in 0..50u32 {
            b.node(
                &["P"],
                &[("age", n(f64::from(i % 10))), ("name", s(&format!("p{i}")))],
            );
        }
        let store = b.build();
        // Top-3 by age DESC, then name — a non-key column (name) is projected.
        let q = "MATCH (p:P) RETURN p.name AS name, p.age AS age ORDER BY age DESC, name LIMIT 3";
        let got = run(&crate::gql::parse(q).unwrap(), &store);
        // Highest ages are 9 (p9, p19, p29, p39, p49) → name-sorted first three.
        assert_eq!(names_of(&got, 0), vec!["p19", "p29", "p39"]);
        assert!(got
            .rows
            .iter()
            .all(|r| matches!(r[1], Value::Num(x) if x == 9.0)));
        // With SKIP: rows 3..6 of the same order.
        let q2 = "MATCH (p:P) RETURN p.name AS name, p.age AS age ORDER BY age DESC, name SKIP 3 LIMIT 2";
        assert_eq!(
            names_of(&run(&crate::gql::parse(q2).unwrap(), &store), 0),
            vec!["p49", "p9"]
        );
    }

    /// A low-cardinality string column dictionary-encodes, and every read shape
    /// (DISTINCT / GROUP BY / equality filter / ORDER BY) returns exactly what the
    /// plain `Str` column would — while a high-cardinality column stays `Str`.
    #[test]
    fn dict_encoded_column_round_trips() {
        let depts = ["eng", "sales", "ops"];
        let mut b = Builder::default();
        for i in 0..30u32 {
            b.node(
                &["P"],
                &[
                    ("dept", s(depts[i as usize % 3])),
                    ("name", s(&format!("p{i}"))), // 30 distinct -> stays Str
                ],
            );
        }
        let store = b.build();
        // The low-card column encoded; the high-card one did not.
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Dict { .. })
        ));
        assert!(matches!(
            store.column("name"),
            Some(crate::store::Column::Str { .. })
        ));

        let rows = |q: &str| {
            let mut r: Vec<String> = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
            r.sort();
            r
        };
        // DISTINCT over the dict column.
        assert_eq!(
            rows("MATCH (n:P) RETURN DISTINCT n.dept AS d"),
            vec!["eng", "ops", "sales"]
        );
        // GROUP BY the dict column: 10 of each.
        let g = run(
            &crate::gql::parse("MATCH (n:P) RETURN n.dept AS d, count(*) AS c").unwrap(),
            &store,
        );
        assert_eq!(g.rows.len(), 3);
        assert!(g
            .rows
            .iter()
            .all(|r| matches!(r[1], Value::Num(x) if x == 10.0)));
        // count(DISTINCT) over the dict column.
        let c = run(
            &crate::gql::parse("MATCH (n:P) RETURN count(DISTINCT n.dept) AS c").unwrap(),
            &store,
        );
        assert!(matches!(c.rows[0][0], Value::Num(x) if x == 3.0));
        // Equality filter resolves through the dict; a miss matches nothing.
        let count_where = |q: &str| match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            _ => panic!("count is not a number"),
        };
        assert_eq!(
            count_where("MATCH (n:P) WHERE n.dept = 'eng' RETURN count(*) AS c"),
            10.0
        );
        assert_eq!(
            count_where("MATCH (n:P) WHERE n.dept = 'zzz' RETURN count(*) AS c"),
            0.0
        );
        // ORDER BY the dict column sorts by VALUE, not code.
        let o = run(
            &crate::gql::parse("MATCH (n:P) RETURN DISTINCT n.dept AS d ORDER BY d").unwrap(),
            &store,
        );
        assert_eq!(names_of(&o, 0), vec!["eng", "ops", "sales"]);
    }

    /// Writing a value to a dict-encoded column decodes it to `Str` in place, and the
    /// new value reads back correctly alongside the untouched ones.
    #[test]
    fn dict_column_decodes_on_write() {
        let mut b = Builder::default();
        for _ in 0..6u32 {
            b.node(&["P"], &[("dept", s("eng"))]);
        }
        let mut store = b.build();
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Dict { .. })
        ));
        let id = store.nodes_with_label("P")[0];
        store.set_prop(id, "dept", s("legal"));
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Str { .. })
        ));
        assert!(matches!(store.prop(id, "dept"), Value::Str(x) if &*x == "legal"));
        let other = store.nodes_with_label("P")[1];
        assert!(matches!(store.prop(other, "dept"), Value::Str(x) if &*x == "eng"));
    }

    /// Multi-column `DISTINCT` over a dict-encoded string column plus a numeric one
    /// (and an absent cell) dedups on the composite code+bits key exactly as the
    /// general byte-key path would — same distinct tuples, absence as its own value.
    #[test]
    fn multi_col_distinct_over_dict_and_num() {
        let depts = ["eng", "sales"];
        let mut b = Builder::default();
        // 20 rows: dept in {eng,sales} (cycles every row), age in {30,40} (flips
        // every 2 rows) — decoupled, so all 4 present tuples occur...
        for i in 0..20u32 {
            b.node(
                &["P"],
                &[
                    ("dept", s(depts[i as usize % 2])),
                    ("age", n(f64::from(30 + ((i / 2) % 2) * 10))),
                ],
            );
        }
        // ...plus two rows whose dept is ABSENT (age 30) -> a 5th tuple (Null, 30).
        b.node(&["P"], &[("age", n(30.0))]);
        b.node(&["P"], &[("age", n(30.0))]);
        let store = b.build();
        assert!(matches!(
            store.column("dept"),
            Some(crate::store::Column::Dict { .. })
        ));

        let out = run(
            &crate::gql::parse("MATCH (n:P) RETURN DISTINCT n.dept AS d, n.age AS age").unwrap(),
            &store,
        );
        // Render each (dept, age) tuple to a stable string and compare as a set.
        let mut got: Vec<String> = out
            .rows
            .iter()
            .map(|r| format!("{:?}|{:?}", r[0], r[1]))
            .collect();
        got.sort();
        let mut want = vec![
            format!("{:?}|{:?}", Value::Str("eng".into()), Value::Num(30.0)),
            format!("{:?}|{:?}", Value::Str("eng".into()), Value::Num(40.0)),
            format!("{:?}|{:?}", Value::Str("sales".into()), Value::Num(30.0)),
            format!("{:?}|{:?}", Value::Str("sales".into()), Value::Num(40.0)),
            format!("{:?}|{:?}", Value::Null, Value::Num(30.0)),
        ];
        want.sort();
        assert_eq!(got, want);
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

    /// A `SET` that collides with a unique constraint is REJECTED and rolled back —
    /// the Update path enforces constraints like INSERT/_MERGE, not silently apply.
    #[test]
    fn set_enforces_unique_constraint() {
        let mut b = Builder::default();
        b.node(&["User"], &[("email", s("a@x"))]);
        b.node(&["User"], &[("email", s("b@x"))]);
        let mut store = b.build();
        store.create_unique_constraint("User", &["email"]).unwrap();
        let go = |q: &str, store: &mut Store| execute(&crate::gql::parse(q).unwrap(), store);

        // Colliding SET → error, rolled back (still exactly one 'a@x').
        assert!(go(
            "MATCH (u:User) WHERE u.email='b@x' SET u.email='a@x'",
            &mut store
        )
        .is_err());
        let count = |store: &Store, v: &str| {
            store
                .nodes_with_label("User")
                .iter()
                .filter(|&&n| matches!(store.prop(n, "email"), Value::Str(e) if &*e == v))
                .count()
        };
        assert_eq!(count(&store, "a@x"), 1, "collision must have rolled back");
        assert_eq!(count(&store, "b@x"), 1);
        // A non-colliding SET still applies.
        assert!(go(
            "MATCH (u:User) WHERE u.email='b@x' SET u.email='c@x'",
            &mut store
        )
        .is_ok());
        assert_eq!(count(&store, "c@x"), 1);
    }

    /// `REMOVE` of a required-constraint key is rejected and rolled back.
    #[test]
    fn remove_enforces_required_constraint() {
        let mut b = Builder::default();
        b.node(&["User"], &[("name", s("alice"))]);
        let mut store = b.build();
        store.create_required_constraint("User", "name").unwrap();
        let id = store.nodes_with_label("User")[0];
        assert!(execute(
            &crate::gql::parse("MATCH (u:User) REMOVE u.name").unwrap(),
            &mut store
        )
        .is_err());
        assert!(
            store.has_prop(id, "name"),
            "required key must survive rollback"
        );
    }

    /// GQL DELETE / DETACH DELETE, matching core: a non-DETACH delete of a node with
    /// relationships errors and rolls back; DETACH cascades the edges; an edge delete
    /// leaves the endpoints; a node with no edges deletes plainly.
    #[test]
    fn gql_delete_and_detach_delete() {
        let build = || {
            let mut b = Builder::default();
            let a = b.node(&["P"], &[("n", s("a"))]);
            let z = b.node(&["P"], &[("n", s("b"))]);
            let iso = b.node(&["P"], &[("n", s("iso"))]);
            b.edge(a, z, "R");
            let _ = iso;
            b.build()
        };
        let go = |q: &str, store: &mut Store| execute(&crate::gql::parse(q).unwrap(), store);

        // Non-DETACH delete of a node WITH an edge → error, nothing removed.
        let mut s1 = build();
        assert!(go("MATCH (p:P) WHERE p.n='a' DELETE p", &mut s1).is_err());
        assert_eq!(s1.live_node_count(), 3, "rolled back");

        // DETACH DELETE removes the node and its edge; the neighbour survives.
        let mut s2 = build();
        assert!(go("MATCH (p:P) WHERE p.n='a' DETACH DELETE p", &mut s2).is_ok());
        assert_eq!(s2.live_node_count(), 2);

        // A node with NO edges deletes plainly (no DETACH needed).
        let mut s3 = build();
        assert!(go("MATCH (p:P) WHERE p.n='iso' DELETE p", &mut s3).is_ok());
        assert_eq!(s3.live_node_count(), 2);

        // Deleting the EDGE leaves both endpoints; then a plain DELETE works.
        let mut s4 = build();
        assert!(go("MATCH (a:P)-[r:R]->(b) DELETE r", &mut s4).is_ok());
        assert_eq!(s4.live_node_count(), 3);
        assert!(go("MATCH (p:P) WHERE p.n='a' DELETE p", &mut s4).is_ok());
        assert_eq!(s4.live_node_count(), 2);
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

    /// A product that overflows f64 to +Inf is KEPT (IEEE), matching lenke-core —
    /// NaN/Inf are coerced to null only at the JSON egress boundary, not here (K4).
    #[test]
    fn arith_overflow_keeps_inf() {
        use crate::ir::ArithOp::Mul;
        let store = social();
        let one = scan("Person").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))));
        let big = one.project(vec![("x".into(), arith(Mul, lit(n(1e308)), lit(n(1e308))))]);
        assert!(
            matches!(run(&big, &store).rows[0][0], Value::Num(x) if x.is_infinite() && x > 0.0)
        );
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

    /// An indexed range seek returns EXACTLY the scan-filter rows (the
    /// equivalent-spellings invariant): a NULL value matches nothing, and a
    /// cross-type comparison is UNKNOWN → dropped (a string property vs a numeric
    /// bound does NOT match, per the 3VL operator semantics — K2).
    #[test]
    fn range_seek_null_and_cross_type_match_filter() {
        let mut st = Builder::default().build();
        st.add_node(&["P"], &[("v", n(10.0))]);
        st.add_node(&["P"], &[("v", s("zzz"))]); // string: cross-type vs a number
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
        // v > 5: only 10 matches; "zzz" is cross-type → UNKNOWN → dropped; null
        // excluded → 1, and seek agrees with filter.
        assert_eq!(check(&st, CompareOp::Gt, n(5.0)), (1, 1));
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
            .expand(0, Dir::Out, &["KNOWS".to_string()])
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
            .expand(0, Dir::Out, &["WORKS_ON".to_string()])
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["graphdb"]);
    }

    /// Filtering on the FAR end after an expand — the far slot's property.
    #[test]
    fn filter_on_the_expanded_end() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
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
            .expand(0, Dir::In, &["KNOWS".to_string()])
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
            .expand(0, Dir::Out, &["NOPE".to_string()])
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
        let plan = scan("P")
            .expand_edge(0, Dir::Out, &["R".to_string()])
            .project(vec![
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
            .expand_edge(0, Dir::Out, &["R".to_string()])
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
            .expand_edge(0, Dir::Out, &["R".to_string()])
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
            frac: None,
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
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .aggregate(
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
            .expand(0, Dir::Out, &["KNOWS".to_string()])
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
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .expand(1, Dir::Out, &["KNOWS".to_string()])
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
            .expand(0, Dir::Out, &["R".to_string()])
            .expand(1, Dir::Out, &["R".to_string()])
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        assert_eq!(num(&run(&plan, &store).rows[0][0]), 4.0);
    }

    /// `count(DISTINCT c)` over the two-hop chain: the distinct endpoints are
    /// {carol} = 1, deduped in the bitset path.
    #[test]
    fn fused_count_distinct_endpoint() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .expand(1, Dir::Out, &["KNOWS".to_string()])
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
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .aggregate(
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
        let plan = scan("P").expand(0, Dir::Out, &["R".to_string()]).aggregate(
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
        let plan = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .aggregate(
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
            .expand(0, Dir::Out, &["NOPE".to_string()])
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
            .expand(0, Dir::Out, &["KNOWS".to_string()])
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
        let left = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
        let right = scan("Person").expand(0, Dir::Out, &["WORKS_ON".to_string()]);
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
        let left = scan("P").expand(0, Dir::Out, &["R".to_string()]);
        let right = scan("P").expand(0, Dir::Out, &["S".to_string()]);
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
        let left = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
        let right = scan("Person").expand(0, Dir::Out, &["WORKS_ON".to_string()]);
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
            .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail)
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
            .var_length(0, Dir::Out, &["R".to_string()], 0, 2, PathMode::Trail)
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
            .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Walk)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 2, "walk reuses the edge");

        let trail = base
            .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail)
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
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Trail)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&trail, &store).rows.len(), 2); // b (len1), a (len2)

        let walk = from_a
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Walk)
            .project(vec![("end".into(), prop(1, "name"))]);
        assert_eq!(run(&walk, &store).rows.len(), 3); // b, a, b
    }

    /// Build the triangle a->b->c->a with a spur a->d — the ACYCLIC/SIMPLE fixture.
    #[cfg(test)]
    fn triangle_with_spur() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        let d = b.node(&["N"], &[("name", s("d"))]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, a, "R"); // closes the cycle back to the start
        b.edge(a, d, "R");
        b.build()
    }

    /// ACYCLIC forbids repeating ANY node — the hop c->a back to the start is
    /// rejected, so from a over `{1,3}` the endpoints are b, c, d (never a).
    #[test]
    fn varlen_acyclic_forbids_revisiting_the_start() {
        let store = triangle_with_spur();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Acyclic)
            .project(vec![("end".into(), prop(1, "name"))]);
        let mut got = names_of(&run(&plan, &store), 0);
        got.sort();
        assert_eq!(got, vec!["b", "c", "d"]); // no `a`: acyclic can't cycle back
    }

    /// SIMPLE forbids repeating an INTERIOR node but PERMITS a path that closes on
    /// its own start (start == end). From a over `{1,3}` the cycle a->b->c->a is a
    /// legal simple (closed) path, so `a` is emitted alongside b, c, d.
    #[test]
    fn varlen_simple_allows_the_closing_cycle() {
        let store = triangle_with_spur();
        let plan = scan("N")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
            .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Simple)
            .project(vec![("end".into(), prop(1, "name"))]);
        let mut got = names_of(&run(&plan, &store), 0);
        got.sort();
        assert_eq!(got, vec!["a", "b", "c", "d"]); // `a` via the closing cycle
    }

    /// Over a 2-cycle a<->b from a with `{1,4}`, the count driver must respect the
    /// node modes (not the algebraic trail shortcut): SIMPLE emits b (len1) and the
    /// closing a (len2) = 2; ACYCLIC emits only b = 1 (a would repeat the start).
    #[test]
    fn varlen_count_honors_node_modes() {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        b.edge(a, bb, "R");
        b.edge(bb, a, "R");
        let store = b.build();
        let from_a = scan("N").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))));
        let count = |mode| {
            let plan = from_a
                .clone()
                .var_length(0, Dir::Out, &["R".to_string()], 1, 4, mode)
                .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
            match run(&plan, &store).rows[0][0] {
                Value::Num(n) => n,
                ref other => panic!("want num, got {other:?}"),
            }
        };
        assert_eq!(count(PathMode::Simple), 2.0);
        assert_eq!(count(PathMode::Acyclic), 1.0);
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
            .var_length(0, Dir::Out, &["R".to_string()], 2, 2, PathMode::Trail)
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
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                None,
                crate::ir::ShortestSelector::Any,
            )
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
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                None,
                crate::ir::ShortestSelector::Any,
            )
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
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                Some(2),
                crate::ir::ShortestSelector::Any,
            )
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
            .shortest_path(
                0,
                Dir::Out,
                &["R".to_string()],
                1,
                None,
                crate::ir::ShortestSelector::Any,
            )
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
            .expand(0, Dir::Out, &["R".to_string()])
            .expand(1, Dir::Out, &["R".to_string()])
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
            .expand(0, Dir::Out, &["R".to_string()])
            .expand(1, Dir::Out, &["R".to_string()])
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
            .expand(0, Dir::Out, &["KNOWS".to_string()])
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
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .project(vec![("b".into(), prop(1, "name"))]);
        assert!(!super::needs_lineage(&plain), "no Path read -> no lineage");
        // The pulled batch (before the lineage-dropping Project) has no sidecar.
        let inner = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
        assert!(super::pull(&inner, &store, false)
            .unwrap()
            .lineage
            .is_none());

        let with_path = scan("Person")
            .expand(0, Dir::Out, &["KNOWS".to_string()])
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
            .expand(0, Dir::Out, &["R".to_string()])
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
