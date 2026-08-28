use super::*;
use crate::batch::Col;
use crate::ir::Plan;
use crate::store::Store;
use crate::value::Value;

/// Whether any cell in a result is boolean `false` — the invariant-violation test.
pub(super) fn rows_have_false(rows: &Rows) -> bool {
    rows.rows
        .iter()
        .any(|r| r.iter().any(|c| matches!(c, Value::Bool(false))))
}

/// The grouping-key bytes of `keys`, reading each key's value via `get`.
pub(super) fn key_bytes(keys: &[String], mut get: impl FnMut(&str) -> Value) -> Vec<u8> {
    let mut buf = Vec::new();
    for k in keys {
        value::group_key_into(&get(k), &mut buf);
    }
    buf
}

/// A pattern property's value by key (NULL if the pattern does not name it).
pub(super) fn pattern_value(props: &[(String, Value)], key: &str) -> Value {
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
/// Like [`render_cell`] but keeps a graph element UNBOXED — `Value::Node`/`Value::Edge`
/// carrying the dense id, NOT the rendered element map. Used when a node/edge flows into a
/// heterogeneous `Col::Gen` (a mixed branch or `inject`) that a downstream step still has to
/// traverse; egress resolves the ref to its map via [`render_cell`]. This is the un-boxed
/// alternative to eagerly rendering the map (which lost node identity, so `out()`/`hasLabel()`
/// on the result yielded nothing).
pub(super) fn cell_value(col: &Col, i: usize, _store: &Store) -> Value {
    match col {
        Col::Nodes(ids) if ids[i] == u32::MAX => Value::Null,
        Col::Edges(eids) if eids[i] == u32::MAX => Value::Null,
        Col::Nodes(ids) => Value::Node(ids[i]),
        Col::Edges(eids) => Value::Edge(eids[i]),
        _ => col.value_at(i),
    }
}

pub(super) fn render_cell(col: &Col, i: usize, store: &Store) -> Value {
    match col {
        // `u32::MAX` is the OPTIONAL-MATCH null sentinel → NULL, not an element map.
        Col::Nodes(ids) if ids[i] == u32::MAX => Value::Null,
        Col::Edges(eids) if eids[i] == u32::MAX => Value::Null,
        Col::Nodes(ids) => node_result_value(store, ids[i]),
        Col::Edges(eids) => edge_result_value(store, eids[i]),
        // A Gen cell may carry an UNBOXED element ref (Value::Node/Edge from a heterogeneous
        // branch/inject) — resolve it to its element map at egress, the same map a Nodes/Edges
        // cell renders. (SPIKE: top-level only; a ref nested in a list/map is not yet resolved.)
        _ => match col.value_at(i) {
            Value::Node(id) => node_result_value(store, id),
            Value::Edge(id) => edge_result_value(store, id),
            v => v,
        },
    }
}

/// The `id` field of a bare-VERTEX element map (`{id, labels, properties}`), or `None`.
pub(super) fn vertex_map_ext_id(v: &Value) -> Option<&str> {
    let Value::Map(pairs) = v else { return None };
    let keys: std::collections::BTreeSet<&str> = pairs
        .iter()
        .filter_map(|(k, _)| match k {
            Value::Str(s) => Some(s.as_ref()),
            _ => None,
        })
        .collect();
    if keys.len() != pairs.len() || keys != ["id", "labels", "properties"].into_iter().collect() {
        return None;
    }
    pairs.iter().find_map(|(k, val)| match (k, val) {
        (Value::Str(k), Value::Str(id)) if k.as_ref() == "id" => Some(id.as_ref()),
        _ => None,
    })
}

/// The `id` field of a bare-EDGE element map (`{id, from, to, labels, properties}`).
pub(super) fn edge_map_ext_id(v: &Value) -> Option<&str> {
    let Value::Map(pairs) = v else { return None };
    let keys: std::collections::BTreeSet<&str> = pairs
        .iter()
        .filter_map(|(k, _)| match k {
            Value::Str(s) => Some(s.as_ref()),
            _ => None,
        })
        .collect();
    if keys.len() != pairs.len()
        || keys
            != ["from", "id", "labels", "properties", "to"]
                .into_iter()
                .collect()
    {
        return None;
    }
    pairs.iter().find_map(|(k, val)| match (k, val) {
        (Value::Str(k), Value::Str(id)) if k.as_ref() == "id" => Some(id.as_ref()),
        _ => None,
    })
}

/// Reconstitute an `unfold`ed element column: when every element is a resolvable
/// bare VERTEX (or EDGE) element map (the fold().unfold() round-trip), resolve each
/// `id` back to a live dense id and return a `Col::Nodes` (or `Col::Edges`) so
/// downstream steps operate on the elements again. Otherwise keep the raw `Col::Gen`.
pub(super) fn reunfold_elements(elems: &[Value], store: &Store) -> Col {
    if elems.is_empty() {
        return Col::Gen(Vec::new());
    }
    let nodes: Option<Vec<u32>> = elems
        .iter()
        .map(|v| vertex_map_ext_id(v).and_then(|ext| store.node_by_ext(ext)))
        .collect();
    if let Some(ids) = nodes {
        return Col::Nodes(ids);
    }
    // Try edges — build a lazy ext→edge map (no reverse map is stored).
    if elems.iter().all(|v| edge_map_ext_id(v).is_some()) {
        let mut by_ext: std::collections::HashMap<Arc<str>, u32> = std::collections::HashMap::new();
        for e in store.all_edges() {
            if let Some(x) = store.edge_ext_id(e) {
                by_ext.entry(x).or_insert(e);
            }
        }
        let eids: Option<Vec<u32>> = elems
            .iter()
            .map(|v| edge_map_ext_id(v).and_then(|ext| by_ext.get(ext).copied()))
            .collect();
        if let Some(eids) = eids {
            return Col::Edges(eids);
        }
    }
    Col::Gen(elems.to_vec())
}

/// The canonical result map for an edge — `{id, from, to, labels(sorted),
/// properties(sorted by key)}`, byte-identical to lenke-core's `val_to_value(Edge)`.
/// `from`/`to` are the endpoint EXTERNAL ids.
pub(super) fn edge_result_value(store: &Store, eid: u32) -> Value {
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
    // ALL of the edge's labels, sorted — mirroring `node_result_value`. A
    // multi-label edge (`[KNOWS, CREATED]`) must render its whole label set
    // regardless of which type it was reached through; `edge_type_name` returns
    // only the primary type, which silently dropped the rest.
    let mut labels = store.edge_labels_of(eid);
    labels.sort_unstable();
    let labels = Value::List(labels.into_iter().map(|t| Value::Str(t.into())).collect());
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
        (Value::Str("id".into()), Value::Str(id.into())),
        (Value::Str("from".into()), Value::Str(ext(src).into())),
        (Value::Str("to".into()), Value::Str(ext(dst).into())),
        (Value::Str("labels".into()), labels),
        (Value::Str("properties".into()), props_map),
    ]))
}

/// Render one element of an interleaved Gremlin `path()` per its (cycled) `by`
/// modulator. A vertex or an edge (`is_edge`); `Element` → the element map,
/// `Prop` → a property value, `Id`/`Label` → the ext-id / label string.
// Extract a dense id (`Value::Num`) as `u32` for path element rendering.
pub(super) fn num_as_u32(v: &Value) -> u32 {
    match v {
        Value::Num(n) => *n as u32,
        _ => 0,
    }
}

pub(super) fn render_gpath_elem(
    store: &Store,
    id: u32,
    is_edge: bool,
    by: &crate::ir::GPathBy,
) -> Value {
    use crate::ir::GPathBy;
    match by {
        GPathBy::Element => {
            if is_edge {
                edge_result_value(store, id)
            } else {
                node_result_value(store, id)
            }
        }
        GPathBy::Prop(k) => {
            if is_edge {
                store.edge_prop(id, k)
            } else {
                store.prop(id, k)
            }
        }
        GPathBy::Id => {
            let ext = if is_edge {
                store.edge_ext_id(id)
            } else {
                store.node_ext_id(id)
            };
            ext.map_or(Value::Null, |s| Value::Str(s.into()))
        }
        GPathBy::Label => {
            if is_edge {
                store
                    .edge_type_name(id)
                    .map_or(Value::Null, |t| Value::Str(t.into()))
            } else {
                store
                    .labels_of(id)
                    .into_iter()
                    .next()
                    .map_or(Value::Null, |l| Value::Str(l.into()))
            }
        }
    }
}

/// Render MANY nodes to their result maps, resolving the property columns ONCE for the
/// whole batch instead of two HashMap-by-key lookups per node per key (what calling
/// [`node_result_value`] per node costs). Byte-identical to per-node rendering: the
/// property map keeps `prop_keys` (sorted) order, filtered to present, and labels stay
/// sorted. The big win for element-materializing shapes — `fold()`, `path`, `valueMap`,
/// `elementMap`, and a bare `g.V()` frontier — where the per-node column re-resolution
/// dominated.
pub(super) fn render_nodes(store: &Store, ids: &[u32]) -> Vec<Value> {
    use crate::store::Column;
    use std::sync::Arc;
    let keys = store.prop_keys_arc();
    let cols: Vec<(&Arc<str>, &Column)> = keys
        .iter()
        .filter_map(|k| store.column(k).map(|c| (k, c)))
        .collect();
    ids.iter()
        .map(|&id| {
            if id == u32::MAX {
                return Value::Null;
            }
            let i = id as usize;
            let ext = store
                .node_ext_id(id)
                .unwrap_or_else(|| Arc::from(id.to_string()));
            let mut labels = store.labels_of(id);
            labels.sort_unstable();
            let labels_list =
                Value::List(labels.into_iter().map(|l| Value::Str(l.into())).collect());
            let props: Vec<(Value, Value)> = cols
                .iter()
                .filter(|(_, c)| c.present_at(i))
                .map(|(k, c)| (Value::Str(Arc::clone(k).into()), c.read(i)))
                .collect();
            Value::Map(Arc::new(vec![
                (Value::Str("id".into()), Value::Str(ext.into())),
                (Value::Str("labels".into()), labels_list),
                (Value::Str("properties".into()), Value::Map(Arc::new(props))),
            ]))
        })
        .collect()
}

/// Resolve the node property columns an element/value map reads, in the SAME order and
/// membership the per-node path produced — sorted keys (every present property, or the
/// `filter` list sorted), each paired with its column. Hoists the per-node
/// `prop_keys()` clone+sort and per-key HashMap probes out of the row loop; the caller
/// then does one `present_at`/`read` per column per node. Byte-identical: `prop_keys_arc`
/// is already sorted, a filter list is sorted here, and a filtered-then-sorted present
/// subset is the same set in the same order.
pub(super) fn resolve_node_cols<'a>(
    store: &'a Store,
    filter: &[String],
) -> Vec<(std::sync::Arc<str>, &'a crate::store::Column)> {
    use std::sync::Arc;
    if filter.is_empty() {
        store
            .prop_keys_arc()
            .iter()
            .filter_map(|k| store.column(k).map(|c| (Arc::clone(k), c)))
            .collect()
    } else {
        let mut keys = filter.to_vec();
        keys.sort();
        keys.into_iter()
            .filter_map(|k| store.column(&k).map(|c| (Arc::from(k.as_str()), c)))
            .collect()
    }
}

/// The canonical result map for a node — `{id, labels(sorted), properties(sorted by
/// key)}`, byte-identical to lenke-core's `val_to_value(Node)`.
/// Map a lineage node-id slice (`Value::Num(dense_id)` entries) to full vertex
/// element maps — the materialization behind `nodes(p)` and a Path's `vertices`.
pub(super) fn path_node_values(store: &Store, ids: &[Value]) -> Vec<Value> {
    ids.iter()
        .map(|v| match v {
            Value::Num(n) => node_result_value(store, *n as u32),
            other => other.clone(),
        })
        .collect()
}

/// Map a lineage edge-id slice to full edge element maps — behind `edges(p)` and a
/// Path's `edges`.
pub(super) fn path_edge_values(store: &Store, ids: &[Value]) -> Vec<Value> {
    ids.iter()
        .map(|v| match v {
            Value::Num(n) => edge_result_value(store, *n as u32),
            other => other.clone(),
        })
        .collect()
}

pub(super) fn node_result_value(store: &Store, id: u32) -> Value {
    use std::sync::Arc;
    let ext = store
        .node_ext_id(id)
        .unwrap_or_else(|| Arc::from(id.to_string()));
    let mut labels = store.labels_of(id);
    labels.sort_unstable();
    let labels_list = Value::List(labels.into_iter().map(|l| Value::Str(l.into())).collect());
    // Present properties on this node, keyed in `prop_keys()` order — which is ALREADY
    // sorted, so the filtered subset stays sorted (core's props_map ordering) with no
    // re-sort and no intermediate Vec.
    let props_map = Value::Map(Arc::new(
        store
            .prop_keys_arc()
            .iter()
            .filter(|k| store.has_prop(id, k))
            .map(|k| {
                let v = store.prop(id, k);
                (Value::Str(Arc::clone(k).into()), v)
            })
            .collect(),
    ));
    Value::Map(Arc::new(vec![
        (Value::Str("id".into()), Value::Str(ext.into())),
        (Value::Str("labels".into()), labels_list),
        (Value::Str("properties".into()), props_map),
    ]))
}

/// A self-describing edge record `{id, label, outV, inV, properties}` — the shape
/// core's `subgraph_edge` builds (single `label` string; endpoints as external ids;
/// properties sorted by key).
pub(super) fn subgraph_edge_value(store: &Store, eid: u32) -> Value {
    use std::sync::Arc;
    let ext = |id: u32| {
        store
            .node_ext_id(id)
            .map_or(Value::Null, |s| Value::Str(s.into()))
    };
    let (src, dst) = store.edge_endpoints(eid).unwrap_or((0, 0));
    let mut keys: Vec<String> = store
        .edge_prop_keys()
        .into_iter()
        .filter(|k| store.has_edge_prop(eid, k))
        .collect();
    keys.sort();
    let props: Vec<(Value, Value)> = keys
        .into_iter()
        .map(|k| {
            let v = store.edge_prop(eid, &k);
            (Value::Str(k.into()), v)
        })
        .collect();
    Value::Map(Arc::new(vec![
        (
            Value::Str("id".into()),
            store
                .edge_ext_id(eid)
                .map_or(Value::Null, |s| Value::Str(s.into())),
        ),
        (
            Value::Str("label".into()),
            store
                .edge_type_name(eid)
                .map_or(Value::Null, |s| Value::Str(s.into())),
        ),
        (Value::Str("outV".into()), ext(src)),
        (Value::Str("inV".into()), ext(dst)),
        (Value::Str("properties".into()), Value::Map(Arc::new(props))),
    ]))
}

/// The empty result a write statement returns (no columns, no rows).
pub(super) fn empty_rows() -> Rows {
    Rows {
        names: Vec::new(),
        rows: Flat::default(),
    }
}

/// The output column names a plan produces, seen through row-shape-preserving
/// operators (`Distinct`, `OrderPage`) down to the naming one. `None` means no
/// explicit projection — the row is the raw slot-0 frontier.
pub(super) fn output_names(plan: &Plan) -> Option<Vec<String>> {
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
pub(super) fn needs_lineage(plan: &Plan) -> bool {
    fn reads_path(e: &Expr) -> bool {
        match e {
            // Reading any part of the path needs the lineage, just like `Path`.
            Expr::Path
            | Expr::PathAccess { .. }
            | Expr::GremlinPath { .. }
            | Expr::GremlinFullPath { .. } => true,
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
            Expr::Call { args, .. } | Expr::GraphPred { args, .. } | Expr::List { items: args } => {
                args.iter().any(reads_path)
            }
            Expr::Record { fields } | Expr::MapLit { entries: fields } => {
                fields.iter().any(|(_, e)| reads_path(e))
            }
            Expr::Field { base, .. } => reads_path(base),
            Expr::Index { base, index, .. } => reads_path(base) || reads_path(index),
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
            | Expr::Param(_)
            | Expr::PropertyExists { .. }
            | Expr::IsLabeled { .. }
            | Expr::Exists { .. }
            | Expr::CountSubquery { .. }
            | Expr::ScalarSubquery { .. }
            | Expr::CollectSubquery { .. }
            | Expr::AggSubquery { .. }
            | Expr::UncorrelatedExists { .. }
            | Expr::UncorrelatedCount { .. }
            | Expr::UncorrelatedScalar { .. } => false,
        }
    }
    match plan {
        Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::EdgeSeed { .. }
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::Merge { .. }
        | Plan::MergeEdge { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. }
        | Plan::TxControl { .. }
        | Plan::InsertFrom { .. } => false,
        // otherV off a bare edge reads the lineage reference vertex.
        Plan::EdgeVertex { input, other, .. } => *other || needs_lineage(input),
        Plan::Sample { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::Expand { input, .. }
        | Plan::OptionalExpand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::RepeatGroup { input, .. }
        | Plan::NestedGroup { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Distinct { input }
        | Plan::DistinctBy { input, .. }
        | Plan::Tail { input, .. }
        | Plan::NullPadIfEmpty { input, .. }
        | Plan::GroupToMap { input }
        | Plan::AlgoAnnotate { input, .. }
        | Plan::SortLocal { input, .. } => needs_lineage(input),
        // tree() reads the path lineage itself, so its INPUT must track it.
        Plan::Tree { .. } => true,
        Plan::MapSlot { input, value, .. } => reads_path(value) || needs_lineage(input),
        Plan::Subgraph { input, .. } => needs_lineage(input),
        Plan::ShortestPathEnum { input, .. } => needs_lineage(input),
        Plan::OptionalScan { input, filters, .. } => {
            filters.iter().any(|(_, e)| reads_path(e)) || needs_lineage(input)
        }
        Plan::Unwind { input, list, .. } => reads_path(list) || needs_lineage(input),
        Plan::Branch { input, bodies } => needs_lineage(input) || bodies.iter().any(needs_lineage),
        Plan::PerElementBranch {
            input, cond, arms, ..
        } => {
            needs_lineage(input)
                || cond.as_deref().is_some_and(needs_lineage)
                || arms.iter().any(needs_lineage)
        }
        Plan::Reconverge { input, .. } => needs_lineage(input),
        Plan::IntervalExpand {
            input, qlo, qhi, ..
        } => reads_path(qlo) || reads_path(qhi) || needs_lineage(input),
        Plan::Filter { input, pred } => reads_path(pred) || needs_lineage(input),
        // A `PathRecord` writes the step-history, so the plan must track lineage; the `input`
        // is walked for the same reason `Filter` walks its own (the decision is plan-global).
        Plan::PathRecord { .. } => true,
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
                    crate::ir::SetOp::Remove { .. }
                    | crate::ir::SetOp::AddLabel { .. }
                    | crate::ir::SetOp::RemoveLabel { .. }
                    | crate::ir::SetOp::Delete { .. } => false,
                })
        }
        Plan::UpdateReturn { input, ops, tail } => {
            needs_lineage(input)
                || needs_lineage(tail)
                || ops.iter().any(|op| match op {
                    crate::ir::SetOp::Set { value, .. } => reads_path(value),
                    crate::ir::SetOp::Remove { .. }
                    | crate::ir::SetOp::AddLabel { .. }
                    | crate::ir::SetOp::RemoveLabel { .. }
                    | crate::ir::SetOp::Delete { .. } => false,
                })
        }
    }
}
