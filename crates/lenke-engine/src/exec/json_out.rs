use super::*;
use crate::batch::{Batch, Col};
use crate::store::Store;
use crate::value::Value;

pub fn try_stream_gremlin_json(
    plan: &Plan,
    store: &Store,
    track: bool,
    min_rows: f64,
) -> Option<String> {
    if track {
        return None; // a path/lineage result is not this shape
    }
    let Plan::Project { input, items } = plan else {
        return None;
    };
    if items.len() != 1 {
        return None; // a Gremlin result is a single column
    }
    // STRUCTURAL gate (measured — see `plan_probe`): stream ONLY the one shape that
    // reliably wins — a SINGLE hop with NO filter. A deeper chain re-runs every hop
    // per block and loses (1.3-7.6x); a filtered chain defeats the row estimate
    // (`has(eq(..))` estimates 144k but matches ~1, so a row gate would wave a 7.6x
    // regression through). Requiring one bare Expand sidesteps both — and there the
    // estimate is trustworthy, so the `min_rows` floor gates the rest.
    let (inner_body, ids) = streaming_chain(input, store)?;
    if !single_hop_no_filter(&inner_body) {
        return None;
    }
    // Deliberately HIGH floor: below it the block/serialize overhead loses to the
    // vectorized materialized path — we do NOT opt in early.
    if crate::cost::estimate(input, store).rows < min_rows {
        return None;
    }
    // Run the chain BELOW the projection per block, then serialize the one output value
    // per row directly — fusing render + serialize so the string heap is touched once,
    // not once to `clone` into a `Col::Str`/`Vec<Value>` and again to serialize. When the
    // sole item is a bare `Prop` over the block's node frontier, `write_nodes_prop_json`
    // reads and writes each property in one scattered pass (no `Arc` bump, no `Value`);
    // any other item falls back to project-then-`write_col_json` (still one pass, no
    // `Vec<Value>`). Byte-identical to `pull_body(Project{..})` then `write_value` each.
    const BLOCK: usize = 8192;
    let single_prop = match items.as_slice() {
        [(_, Expr::Prop { slot, key })] => Some((*slot, key.as_str())),
        _ => None,
    };
    let mut out = String::from("[");
    let mut first = true;
    let mut start = 0usize;
    while start < ids.len() {
        let end = (start + BLOCK).min(ids.len());
        let batch = pull_body(
            &inner_body,
            store,
            &Batch::single(Col::Nodes(ids[start..end].to_vec())),
        )
        .ok()?;
        // Fused fast path: one Prop over a fully-present scalar node column.
        if let Some((slot, key)) = single_prop {
            if let Col::Nodes(nids) = batch.slot(slot) {
                if write_nodes_prop_json(&mut out, store, nids, key, &mut first) {
                    start = end;
                    continue;
                }
            }
        }
        // General path: project the item(s) to one column, serialize it in place.
        let cols = eval_all(items.iter().map(|(_, e)| e), store, &batch).ok()?;
        let col = cols.into_iter().next()?;
        write_col_json(&mut out, &col, store, &mut first);
        start = end;
    }
    out.push(']');
    Some(out)
}

/// Serialize `nids`'s value for property `key` straight into `out` (comma-separated,
/// `first` tracking whether a leading comma is due), when the property is a
/// fully-present scalar column — the fused render+serialize fast path. Returns `false`
/// WITHOUT writing anything when the shape isn't handled (a sentinel id, an absent
/// value, or a non-scalar column), so the caller can fall back for that block. The
/// bytes written are identical to `read_property` → `write_value`: a `Str`/`Dict` cell
/// as a JSON string, `Num` per the number rules, `Bool` as a literal.
fn write_nodes_prop_json(
    out: &mut String,
    store: &Store,
    nids: &[u32],
    key: &str,
    first: &mut bool,
) -> bool {
    if nids.contains(&u32::MAX) {
        return false; // a null sentinel needs the general NULL-carrying path
    }
    let Some(column) = store.column(key) else {
        return false; // missing column → all NULL, let the general path emit nulls
    };
    // All-present check up front: a partial column must not emit a half-written block.
    let present = match column {
        Column::Num { present, .. }
        | Column::Str { present, .. }
        | Column::Dict { present, .. }
        | Column::Bool { present, .. } => present,
        _ => return false,
    };
    if nids.iter().any(|&id| !present[id as usize]) {
        return false;
    }
    let sep = |out: &mut String, first: &mut bool| {
        if !*first {
            out.push(',');
        }
        *first = false;
    };
    match column {
        Column::Str { data, .. } => {
            for &id in nids {
                sep(out, first);
                crate::json::write_string(out, &data[id as usize]);
            }
        }
        Column::Dict { dict, codes, .. } => {
            for &id in nids {
                sep(out, first);
                crate::json::write_string(out, &dict[codes[id as usize] as usize]);
            }
        }
        Column::Num { data, .. } => {
            for &id in nids {
                sep(out, first);
                crate::json::write_value(out, &Value::Num(data[id as usize]));
            }
        }
        Column::Bool { data, .. } => {
            for &id in nids {
                sep(out, first);
                out.push_str(if data[id as usize] { "true" } else { "false" });
            }
        }
        _ => return false,
    }
    true
}

/// Serialize a whole projected `Col` into `out` (comma-separated, `first`-tracked),
/// without the `Col` → `Vec<Value>` step `col_into_values` would take: a typed column
/// (`Str`/`Num`/`Bool`) writes each cell straight, and only a boxed `Gen` column defers
/// to `write_value`. Byte-identical to serializing `col_into_values(col)` cell by cell.
fn write_col_json(out: &mut String, col: &Col, store: &Store, first: &mut bool) {
    let sep = |out: &mut String, first: &mut bool| {
        if !*first {
            out.push(',');
        }
        *first = false;
    };
    match col {
        Col::Str(v) => {
            for s in v {
                sep(out, first);
                crate::json::write_string(out, s);
            }
        }
        Col::Num(v) => {
            for &x in v {
                sep(out, first);
                crate::json::write_value(out, &Value::Num(x));
            }
        }
        Col::Bool(v) => {
            for &b in v {
                sep(out, first);
                out.push_str(if b { "true" } else { "false" });
            }
        }
        // Nodes/Edges render as element maps, Gen carries arbitrary values — both need
        // the full renderer. Reuse `col_into_values` (the identical cells) for these.
        _ => {
            for v in col_into_values(col.clone(), store) {
                sep(out, first);
                crate::json::write_value(out, &v);
            }
        }
    }
}

/// Entry point the FFI's `lnk_query` (Gremlin, JSON) uses: stream the result JSON when the shape
/// and cost allow, else materialize + serialize as before. Kept here (not in the FFI) so
/// it has the plan + streaming machinery; falls back transparently.
pub fn run_gremlin_json(plan: &Plan, store: &Store) -> String {
    try_run_gremlin_json(plan, store).expect("read plan evaluation faulted")
}

/// Fallible Gremlin-JSON entry point: an evaluation fault (a bad cast, a cross-type
/// order, …) returns `Err` instead of panicking, so the FFI can surface it as a null
/// result rather than unwinding across the C boundary (which aborts the process).
pub fn try_run_gremlin_json(plan: &Plan, store: &Store) -> Result<String, String> {
    // The fused/streamed sinks below call `pull` directly (off `try_run`'s path), so a
    // var-length traversal here would recurse on the normal stack — route it to the big
    // one, same as `try_run`.
    on_big_stack(plan, || try_run_gremlin_json_inner(plan, store))
}

/// GQL egress (`lnk_query`, JSON rows): stream a var-length endpoint projection to the
/// `{columns, rows}` document when it applies — so a large closure completes without
/// materializing the row batch — else materialize + serialize. Big-stack-dispatched like
/// `try_run`. Byte-identical to `gql_rows_json(try_run(...))`.
pub fn try_run_gql_json(plan: &Plan, store: &Store) -> Result<String, String> {
    on_big_stack(plan, || {
        if let Some(res) = try_stream_varlen_json(plan, store, true) {
            return res;
        }
        Ok(crate::json::gql_rows_json(&try_run_inner(plan, store)?))
    })
}

fn try_run_gremlin_json_inner(plan: &Plan, store: &Store) -> Result<String, String> {
    // Fused element/value-map serialization: for a terminal node-map projection, write
    // the JSON straight from the columns and skip building a `Value::Map` tree per row
    // (the dominant cost of these shapes — ~8-10 heap allocs/row otherwise). No cost
    // gate: it is strictly less work than build-then-serialize, so it wins at all sizes.
    if let Some(json) = try_fused_map_json(plan, store) {
        return Ok(json);
    }
    if let Some(json) = try_fused_fold_json(plan, store) {
        return Ok(json);
    }
    if let Some(json) = try_fused_maplit_json(plan, store) {
        return Ok(json);
    }
    // Measured crossover (see `plan_probe`): below ~1M output rows the materialized
    // path ties or wins; the streamed sink pulls 20-29% ahead only at 1M-2.7M+, where
    // it also never builds the full frontier column. Deliberately high — we do NOT opt
    // in eagerly (matches the `pull_top_output_streamed` precedent).
    const STREAM_JSON_ROWS: f64 = 1_000_000.0;
    let track = needs_lineage(plan);
    if let Some(json) = try_stream_gremlin_json(plan, store, track, STREAM_JSON_ROWS) {
        return Ok(json);
    }
    // Stream a var-length endpoint projection straight to JSON — no giant row batch, so a
    // large closure completes (up to a byte cap) where the materialized path would trip
    // the 1M-row trail limit.
    if let Some(res) = try_stream_varlen_json(plan, store, false) {
        return res;
    }
    Ok(crate::json::gremlin_results_json(&try_run(plan, store)?))
}

/// The node-map projection shapes the fused serializer handles, each byte-identical to
/// building the corresponding `Value::Map` then serializing it.
enum NodeMapKind {
    /// A bare node frontier → the NESTED `{id, labels:[…], properties:{…}}` render
    /// (`node_result_value`), where `id` falls back to the dense id as a string.
    Nested,
    /// Gremlin `elementMap()` → the FLAT `{id, label, <props…>}`, where `id`/`label`
    /// are NULL (not a dense-id fallback) when absent.
    Flat,
    /// Gremlin `valueMap()` (`wrap=false`) / `propertyMap()` (`wrap=true`, each value in
    /// a one-element list) → the present properties. `tokens` (from `valueMap(true)`)
    /// also prepends id + label, like `Flat` but without an edge's IN/OUT submaps.
    Value { wrap: bool, tokens: bool },
}

/// `g.V().project(k…).by(e…)` and GQL map projections — a terminal `Project{[MapLit]}`
/// — serialized straight to `[{k:v,…},…]`, skipping the per-row `Value::Map` (its Vec,
/// Arc, and freshly-allocated key Arcs). Values are computed vectorized (`eval_all`) once;
/// a scalar cell writes directly, an element cell falls back to `render_cell` (its element
/// map). Byte-identical to building the map then serializing: same key order, same values.
fn try_fused_maplit_json(plan: &Plan, store: &Store) -> Option<String> {
    let Plan::Project { input, items } = plan else {
        return None;
    };
    let [(_, Expr::MapLit { entries })] = items.as_slice() else {
        return None;
    };
    if needs_lineage(plan) {
        return None;
    }
    let batch = pull(input, store, false).ok()?;
    let cols = eval_all(entries.iter().map(|(_, e)| e), store, &batch).ok()?;
    let mut out = String::from("[");
    for i in 0..batch.rows() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        for (j, ((k, _), col)) in entries.iter().zip(&cols).enumerate() {
            if j > 0 {
                out.push(',');
            }
            crate::json::write_string(&mut out, k);
            out.push(':');
            match col {
                Col::Str(v) => crate::json::write_string(&mut out, &v[i]),
                Col::Num(v) => crate::json::write_value(&mut out, &Value::Num(v[i])),
                Col::Bool(v) => out.push_str(if v[i] { "true" } else { "false" }),
                other => crate::json::write_value(&mut out, &render_cell(other, i, store)),
            }
        }
        out.push('}');
    }
    out.push(']');
    Some(out)
}

/// `g.V().fold()` / `g.E().fold()` — a whole node/edge frontier collected into ONE list —
/// serialized straight to `[[<element maps>]]`, skipping the `Value::List` of `Value::Map`
/// trees the general `Collect` aggregate builds (the same per-node allocation the terminal
/// map writer eliminates, here for the list-wrapped fold). Only a keyless, non-distinct
/// `Collect(Slot)` over a node/edge frontier; anything else falls back.
fn try_fused_fold_json(plan: &Plan, store: &Store) -> Option<String> {
    let Plan::Aggregate { input, keys, aggs } = plan else {
        return None;
    };
    if !keys.is_empty() || aggs.len() != 1 || needs_lineage(plan) {
        return None;
    }
    let agg = &aggs[0];
    if agg.func != AggFn::Collect || agg.distinct {
        return None;
    }
    let Some(Expr::Slot(s)) = &agg.arg else {
        return None;
    };
    let batch = pull(input, store, false).ok()?;
    let cols = resolve_node_cols(store, &[]);
    let mut out = String::from("[[");
    // An empty frontier narrowed below this slot (`outE('X').limit(0).fold()` — limit(0)
    // leaves a zero-row batch that may drop the edge column) folds to the empty list. Guard
    // the slot read: `batch.slot()` would panic-index (crashing the FFI as E_INVALID_VALUE).
    let Some(frontier) = batch.slots.get(*s) else {
        out.push_str("]]");
        return Some(out);
    };
    match frontier {
        Col::Nodes(ids) => {
            for (n, &id) in ids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if id == u32::MAX {
                    out.push_str("null");
                } else {
                    write_node_nested_map(&mut out, store, id, &cols);
                }
            }
        }
        Col::Edges(eids) => {
            let ecols = resolve_edge_cols(store, &[]);
            for (n, &eid) in eids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if eid == u32::MAX {
                    out.push_str("null");
                } else {
                    write_edge_nested_map(&mut out, store, eid, &ecols);
                }
            }
        }
        _ => return None, // a folded scalar list is not an element-map fold
    }
    out.push_str("]]");
    Some(out)
}

/// Serialize a terminal single-column node-map projection directly to JSON, skipping the
/// per-row `Value::Map` tree. Returns `None` (→ the caller's slower path) for anything
/// not a node frontier rendered as an element/value map: an edge frontier, a scalar
/// projection, a lineage-tracked plan, or a non-`Slot` map argument.
fn try_fused_map_json(plan: &Plan, store: &Store) -> Option<String> {
    if needs_lineage(plan) {
        return None;
    }
    // `input` is the plan whose batch to pull; `slot` the frontier column within it.
    let (input, kind, slot, filter): (&Plan, NodeMapKind, usize, Vec<String>) = match plan {
        Plan::Project { input, items } if items.len() == 1 => match &items[0].1 {
            // A bare frontier projection renders as the nested element map (render_cell).
            Expr::Slot(s) => (input.as_ref(), NodeMapKind::Nested, *s, Vec::new()),
            Expr::Call { name, args }
                if matches!(name.as_str(), "element_map" | "value_map" | "property_map") =>
            {
                let Some(Expr::Slot(s)) = args.first() else {
                    return None; // a non-slot element arg (rare) keeps the general path
                };
                let filter = args[1..]
                    .iter()
                    .filter_map(|e| match e {
                        Expr::Lit(Value::Str(s)) => Some(s.to_string()),
                        _ => None,
                    })
                    .collect();
                // value_map/property_map carry an include-tokens Bool (skipped as a key above).
                let tokens = args[1..]
                    .iter()
                    .any(|e| matches!(e, Expr::Lit(Value::Bool(true))));
                let kind = match name.as_str() {
                    "element_map" => NodeMapKind::Flat,
                    "value_map" => NodeMapKind::Value {
                        wrap: false,
                        tokens,
                    },
                    _ => NodeMapKind::Value { wrap: true, tokens },
                };
                (input.as_ref(), kind, *s, filter)
            }
            _ => return None,
        },
        // A projection-LESS read frontier (`g.V()`, `g.E()`, a bare filtered/dedup'd
        // frontier): `try_run` renders slot 0 as its nested element map when
        // `output_names` is None. Mirror that exact single-column path, fused.
        other if output_names(other).is_none() => (other, NodeMapKind::Nested, 0, Vec::new()),
        _ => return None,
    };
    let batch = pull(input, store, false).ok()?;
    let mut out = String::from("[");
    // An empty frontier narrowed below this slot (a zero-row batch after `limit(0)`) renders
    // as no rows. Guard the slot read — `batch.slot()` would panic-index the FFI.
    let Some(frontier) = batch.slots.get(slot) else {
        return Some(String::from("[]"));
    };
    match frontier {
        Col::Nodes(ids) => {
            let cols = resolve_node_cols(store, &filter);
            for (n, &id) in ids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if id == u32::MAX {
                    out.push_str("null"); // the OPTIONAL-match null sentinel
                    continue;
                }
                match &kind {
                    NodeMapKind::Nested => write_node_nested_map(&mut out, store, id, &cols),
                    NodeMapKind::Flat => write_node_flat_map(&mut out, store, id, &cols),
                    NodeMapKind::Value { wrap, tokens } => {
                        write_node_value_map(&mut out, store, id, &cols, *wrap, *tokens);
                    }
                }
            }
        }
        Col::Edges(eids) => {
            let cols = resolve_edge_cols(store, &filter);
            for (n, &eid) in eids.iter().enumerate() {
                if n > 0 {
                    out.push(',');
                }
                if eid == u32::MAX {
                    out.push_str("null");
                    continue;
                }
                match &kind {
                    NodeMapKind::Nested => write_edge_nested_map(&mut out, store, eid, &cols),
                    NodeMapKind::Flat => write_edge_flat_map(&mut out, store, eid, &cols),
                    NodeMapKind::Value { wrap, tokens } => {
                        write_edge_value_map(&mut out, store, eid, &cols, *wrap, *tokens);
                    }
                }
            }
        }
        // A scalar column (e.g. a projected value) is not an element map — fall back.
        _ => return None,
    }
    out.push(']');
    Some(out)
}

/// A resolved read handle for one edge property — the dense numeric overlay when fresh,
/// else the boxed per-eid map. `cell(eid)` returns the present value or `None`, matching
/// `store.edge_prop`/`has_edge_prop` byte-for-byte (the overlay is kept in step with the
/// boxed source, and only homogeneously-numeric keys get one).
enum EdgeCol<'a> {
    Num(&'a [f64], &'a [bool]),
    Boxed(&'a crate::store::EdgeMap),
}

impl EdgeCol<'_> {
    fn cell(&self, eid: u32) -> Option<Value> {
        match self {
            EdgeCol::Num(data, present) => {
                let i = eid as usize;
                (i < present.len() && present[i]).then(|| Value::Num(data[i]))
            }
            EdgeCol::Boxed(map) => map.get(&eid).cloned(),
        }
    }
}

/// Resolve the edge property read handles once, in the SAME sorted-key order and
/// membership the per-edge path produced — hoisting `edge_prop_keys()` clone+sort and
/// the per-key `edge_prop_map`/`edge_num_column` lookups out of the row loop.
fn resolve_edge_cols<'a>(
    store: &'a Store,
    filter: &[String],
) -> Vec<(std::sync::Arc<str>, EdgeCol<'a>)> {
    use std::sync::Arc;
    let keys: Vec<String> = if filter.is_empty() {
        store.edge_prop_keys() // already sorted
    } else {
        let mut k = filter.to_vec();
        k.sort();
        k
    };
    keys.into_iter()
        .filter_map(|k| {
            let col = match store.edge_num_column(&k) {
                Some((d, p)) => EdgeCol::Num(d, p),
                None => EdgeCol::Boxed(store.edge_prop_map(&k)?),
            };
            Some((Arc::from(k.as_str()), col))
        })
        .collect()
}

/// A node's external id as a JSON string, falling back to its dense id (the `id`/`from`/
/// `to` rule for the NESTED renders).
fn write_node_ext_or_dense(out: &mut String, store: &Store, n: u32) {
    match store.node_ext_id(n) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => crate::json::write_string(out, &n.to_string()),
    }
}

/// A `{id, label}` endpoint stub for the flat edge `elementMap` — `id`/`label` NULL when
/// absent (matching the `node_id`/`node_label` closures).
fn write_node_stub(out: &mut String, store: &Store, v: u32) {
    out.push_str("{\"id\":");
    match store.node_ext_id(v) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => out.push_str("null"),
    }
    out.push_str(",\"label\":");
    match store.labels_of(v).first() {
        Some(l) => crate::json::write_string(out, l),
        None => out.push_str("null"),
    }
    out.push('}');
}

/// `{id, from, to, labels:[type?], properties:{sorted present}}` — the nested edge render,
/// byte-identical to `edge_result_value(store, eid)` serialized.
fn write_edge_nested_map(
    out: &mut String,
    store: &Store,
    eid: u32,
    cols: &[(std::sync::Arc<str>, EdgeCol<'_>)],
) {
    out.push_str("{\"id\":");
    match store.edge_ext_id(eid) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => crate::json::write_string(out, &format!("e{eid}")),
    }
    let (src, dst) = store.edge_endpoints(eid).unwrap_or((0, 0));
    out.push_str(",\"from\":");
    write_node_ext_or_dense(out, store, src);
    out.push_str(",\"to\":");
    write_node_ext_or_dense(out, store, dst);
    out.push_str(",\"labels\":[");
    let mut labels = store.edge_labels_of(eid);
    labels.sort_unstable();
    for (i, t) in labels.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        crate::json::write_string(out, t);
    }
    out.push_str("],\"properties\":{");
    let mut first = true;
    for (k, col) in cols {
        if let Some(v) = col.cell(eid) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            crate::json::write_value(out, &v);
        }
    }
    out.push_str("}}");
}

/// `{id, label, IN:{…}, OUT:{…}, <sorted props flat>}` — the flat Gremlin `elementMap()`
/// on an edge (IN is the destination, OUT the source; `id`/`label` NULL when absent).
fn write_edge_flat_map(
    out: &mut String,
    store: &Store,
    eid: u32,
    cols: &[(std::sync::Arc<str>, EdgeCol<'_>)],
) {
    out.push_str("{\"id\":");
    match store.edge_ext_id(eid) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => out.push_str("null"),
    }
    out.push_str(",\"label\":");
    match store.edge_type_name(eid) {
        Some(t) => crate::json::write_string(out, &t),
        None => out.push_str("null"),
    }
    if let Some((src, dst)) = store.edge_endpoints(eid) {
        out.push_str(",\"IN\":");
        write_node_stub(out, store, dst);
        out.push_str(",\"OUT\":");
        write_node_stub(out, store, src);
    }
    for (k, col) in cols {
        if let Some(v) = col.cell(eid) {
            out.push(',');
            crate::json::write_string(out, k);
            out.push(':');
            crate::json::write_value(out, &v);
        }
    }
    out.push('}');
}

/// `{sorted present edge props}` — Gremlin `valueMap()`/`propertyMap()` on an edge.
/// With `tokens` (from `valueMap(true)`) id + label are prepended (no IN/OUT, unlike
/// the flat elementMap); the token values are never list-wrapped.
fn write_edge_value_map(
    out: &mut String,
    store: &Store,
    eid: u32,
    cols: &[(std::sync::Arc<str>, EdgeCol<'_>)],
    wrap: bool,
    tokens: bool,
) {
    out.push('{');
    let mut first = true;
    if tokens {
        out.push_str("\"id\":");
        match store.edge_ext_id(eid) {
            Some(ext) => crate::json::write_string(out, &ext),
            None => out.push_str("null"),
        }
        out.push_str(",\"label\":");
        match store.edge_type_name(eid) {
            Some(t) => crate::json::write_string(out, &t),
            None => out.push_str("null"),
        }
        first = false;
    }
    for (k, col) in cols {
        if let Some(v) = col.cell(eid) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            if wrap {
                out.push('[');
                crate::json::write_value(out, &v);
                out.push(']');
            } else {
                crate::json::write_value(out, &v);
            }
        }
    }
    out.push('}');
}

/// Write one present property column cell straight to `out` (the caller guarantees
/// `present_at(i)`), avoiding the `Arc`/`Value` a `Column::read` would build for the
/// scalar cases. Byte-identical to `write_value(&col.read(i))`.
fn write_col_cell_json(out: &mut String, col: &crate::store::Column, i: usize) {
    use crate::store::Column;
    match col {
        Column::Str { data, .. } => crate::json::write_string(out, &data[i]),
        Column::Dict { dict, codes, .. } => {
            crate::json::write_string(out, &dict[codes[i] as usize]);
        }
        Column::Num { data, .. } => crate::json::write_value(out, &Value::Num(data[i])),
        Column::Bool { data, .. } => out.push_str(if data[i] { "true" } else { "false" }),
        // Temporal / Gen: defer to the value renderer (the leaf types the fast path skips).
        other => crate::json::write_value(out, &other.read(i)),
    }
}

/// `{id, labels:[sorted], properties:{sorted present}}` — the nested node render, written
/// directly. Byte-identical to `node_result_value(store, id)` serialized.
fn write_node_nested_map(
    out: &mut String,
    store: &Store,
    id: u32,
    cols: &[(std::sync::Arc<str>, &crate::store::Column)],
) {
    let i = id as usize;
    out.push_str("{\"id\":");
    match store.node_ext_id(id) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => crate::json::write_string(out, &id.to_string()),
    }
    out.push_str(",\"labels\":[");
    for (j, l) in store.labels_of(id).iter().enumerate() {
        if j > 0 {
            out.push(',');
        }
        crate::json::write_string(out, l);
    }
    out.push_str("],\"properties\":{");
    let mut first = true;
    for (k, col) in cols {
        if col.present_at(i) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            write_col_cell_json(out, col, i);
        }
    }
    out.push_str("}}");
}

/// `{id, label, <sorted present props flat>}` — the flat Gremlin `elementMap()` shape.
/// `id`/`label` are NULL when absent (no dense-id fallback, unlike the nested render).
fn write_node_flat_map(
    out: &mut String,
    store: &Store,
    id: u32,
    cols: &[(std::sync::Arc<str>, &crate::store::Column)],
) {
    let i = id as usize;
    out.push_str("{\"id\":");
    match store.node_ext_id(id) {
        Some(ext) => crate::json::write_string(out, &ext),
        None => out.push_str("null"),
    }
    out.push_str(",\"label\":");
    match store.labels_of(id).first() {
        Some(l) => crate::json::write_string(out, l),
        None => out.push_str("null"),
    }
    // id and label are always emitted, so every property is comma-prefixed.
    for (k, col) in cols {
        if col.present_at(i) {
            out.push(',');
            crate::json::write_string(out, k);
            out.push(':');
            write_col_cell_json(out, col, i);
        }
    }
    out.push('}');
}

/// `{sorted present props}` — Gremlin `valueMap()` (`wrap=false`) or `propertyMap()`
/// (`wrap=true`, each value wrapped in a one-element list). With `tokens` (from
/// `valueMap(true)`) id + label are prepended (NULL when absent), like the flat map
/// but without an edge's IN/OUT — the token values are never list-wrapped.
fn write_node_value_map(
    out: &mut String,
    store: &Store,
    id: u32,
    cols: &[(std::sync::Arc<str>, &crate::store::Column)],
    wrap: bool,
    tokens: bool,
) {
    let i = id as usize;
    out.push('{');
    let mut first = true;
    if tokens {
        out.push_str("\"id\":");
        match store.node_ext_id(id) {
            Some(ext) => crate::json::write_string(out, &ext),
            None => out.push_str("null"),
        }
        out.push_str(",\"label\":");
        match store.labels_of(id).first() {
            Some(l) => crate::json::write_string(out, l),
            None => out.push_str("null"),
        }
        first = false;
    }
    for (k, col) in cols {
        if col.present_at(i) {
            if !first {
                out.push(',');
            }
            first = false;
            crate::json::write_string(out, k);
            out.push(':');
            if wrap {
                out.push('[');
                write_col_cell_json(out, col, i);
                out.push(']');
            } else {
                write_col_cell_json(out, col, i);
            }
        }
    }
    out.push('}');
}
