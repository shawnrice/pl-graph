//! NDJSON egress/ingest: dump/load a store as newline-delimited JSON — one object
//! per live node, then one per edge. Dependency-free (a small hand-rolled JSON
//! writer, no serde) and deterministic (nodes by id; labels and property keys
//! sorted; edges in adjacency order).
//!
//! Emits the SHIPPED shape (byte-shape-compatible with lenke-core /
//! `@lenke/serialization`) so NDJSON is interchangeable across the engines (ids
//! are PRESERVED external ids — strings; a numeric id on ingest is kept as text):
//! - node: `{"type":"node","id":"N","labels":[...],"properties":{...}}`
//! - edge: `{"type":"edge","id":"E","from":"F","to":"T","labels":["R",...],"properties":{...}}`
//!
//! Ingest is lenient: it ALSO reads the engine's earlier shape (node `{"id",…,
//! "props"}`, edge `{"from","to","type":"<etype>","props"}`), so anything the
//! engine wrote before still loads. Property key ORDER is unspecified (each engine
//! emits its own), so cross-engine comparison is structural, not byte-identical.
//!
//! This module SERIALIZES values; it does not define value semantics (order,
//! equality) — those stay in [`crate::value`]. A non-finite number (NaN/Inf) has
//! no JSON form and is written as `null`, consistent with the engine's
//! NaN/Inf→null policy.

use crate::gstr::GStr;
use std::sync::Arc;

use crate::store::Store;
use crate::value::Value;

/// The store as NDJSON: a line per live node, then a line per edge. Ends with a
/// trailing newline when non-empty.
#[must_use]
pub fn to_ndjson(store: &Store) -> String {
    let mut out = String::new();
    let node_keys = store.prop_keys();
    let edge_keys = store.edge_prop_keys();

    for id in 0..u32::try_from(store.node_count()).unwrap_or(u32::MAX) {
        if !store.is_alive(id) {
            continue;
        }
        // The SHIPPED shape (lenke-core / @lenke/serialization): a `type`
        // discriminator, `properties` (not `props`), and the PRESERVED external id.
        out.push_str("{\"type\":\"node\",\"id\":");
        encode_string(&mut out, &store.node_ext_id(id).unwrap_or_default());
        out.push_str(",\"labels\":");
        encode_str_array(&mut out, &store.labels_of(id));
        out.push_str(",\"properties\":");
        encode_object(&mut out, &node_keys, |k| {
            store.has_prop(id, k).then(|| store.prop(id, k))
        });
        out.push_str("}\n");
    }

    for from in 0..u32::try_from(store.node_count()).unwrap_or(u32::MAX) {
        if !store.is_alive(from) {
            continue;
        }
        for a in store.out(from) {
            let eid = a.eid;
            // An edge carries a type SET in `labels` (first = primary type), like
            // core — not a single `type` string.
            out.push_str("{\"type\":\"edge\",\"id\":");
            encode_string(&mut out, &store.edge_ext_id(eid).unwrap_or_default());
            out.push_str(",\"from\":");
            encode_string(&mut out, &store.node_ext_id(from).unwrap_or_default());
            out.push_str(",\"to\":");
            encode_string(&mut out, &store.node_ext_id(a.nbr).unwrap_or_default());
            out.push_str(",\"labels\":");
            encode_str_array(&mut out, &store.edge_labels_of(eid));
            out.push_str(",\"properties\":");
            encode_object(&mut out, &edge_keys, |k| {
                store.has_edge_prop(eid, k).then(|| store.edge_prop(eid, k))
            });
            out.push_str("}\n");
        }
    }
    out
}

/// The store's SCHEMA as NDJSON — the unique and required constraints, one per
/// line (`{"schema":"unique","label":..,"keys":[..]}` /
/// `{"schema":"required","label":..,"key":..}`). Each group sorted for
/// determinism. These lines lead a [`snapshot`] so they apply before the data.
#[must_use]
pub fn dump_schema(store: &Store) -> String {
    let mut out = String::new();
    let mut uniques = store.unique_constraints();
    uniques.sort();
    for (label, keys) in uniques {
        out.push_str("{\"schema\":\"unique\",\"label\":");
        encode_string(&mut out, &label);
        out.push_str(",\"keys\":");
        encode_str_array(&mut out, &keys);
        out.push_str("}\n");
    }
    let mut required = store.required_constraints();
    required.sort();
    for (label, key) in required {
        out.push_str("{\"schema\":\"required\",\"label\":");
        encode_string(&mut out, &label);
        out.push_str(",\"key\":");
        encode_string(&mut out, &key);
        out.push_str("}\n");
    }
    out
}

/// A full snapshot: schema lines first, then the data (nodes + edges). Reloading
/// with [`load_snapshot`] applies the schema before the data, so INSERT-time
/// enforcement on the reloaded store matches the original.
#[must_use]
pub fn snapshot(store: &Store) -> String {
    let mut out = dump_schema(store);
    out.push_str(&to_ndjson(store));
    out
}

/// Load a full snapshot (schema + data). Equivalent to [`from_ndjson`], which
/// already recognizes schema lines; named for symmetry with [`snapshot`].
pub fn load_snapshot(text: &str) -> Result<Store, String> {
    from_ndjson(text)
}

/// Write a JSON object from `keys`, including only those a present value exists
/// for (via `get`), in `keys` order.
fn encode_object(out: &mut String, keys: &[String], get: impl Fn(&str) -> Option<Value>) {
    out.push('{');
    let mut first = true;
    for k in keys {
        let Some(v) = get(k) else { continue };
        if !first {
            out.push(',');
        }
        first = false;
        encode_string(out, k);
        out.push(':');
        encode_value(out, &v);
    }
    out.push('}');
}

/// Append a JSON array of strings to `out` (shared with the schema-op vocabulary).
pub fn encode_str_array(out: &mut String, items: &[String]) {
    out.push('[');
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        encode_string(out, s);
    }
    out.push(']');
}

/// Encode a value as JSON. A non-finite number becomes `null` (no JSON form).
pub fn encode_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Node(_) | Value::Edge(_) => {
            unreachable!("element ref is never a stored property value")
        }
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => {
            if x.is_finite() {
                out.push_str(&x.to_string());
            } else {
                out.push_str("null");
            }
        }
        Value::Str(s) => encode_string(out, s),
        // Tagged temporal: {"@date":"2024-01-15"} — the ISO string under a kind
        // key, matching lenke-core's json_tagged form.
        Value::Temporal(t) => {
            out.push_str("{\"@");
            out.push_str(t.tag());
            out.push_str("\":");
            encode_string(out, &t.format());
            out.push('}');
        }
        Value::List(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_value(out, it);
            }
            out.push(']');
        }
        // A record is a JSON object (keys already sorted, so deterministic).
        Value::Record(fields) => {
            out.push('{');
            for (i, (k, v)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_string(out, k);
                out.push(':');
                encode_value(out, v);
            }
            out.push('}');
        }
        // A Gremlin map is not a STORED property type (no producer persists one);
        // egress is best-effort as an object using each key's string form, so the
        // arm is total. (It decodes back as a record, not a map — acceptable since
        // maps are never written to a property.)
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                match k {
                    Value::Str(s) => encode_string(out, s),
                    other => encode_string(out, &format!("{other:?}")),
                }
                out.push(':');
                encode_value(out, v);
            }
            out.push('}');
        }
    }
}

/// Encode a JSON string with the required escapes (shared with the schema-op vocabulary).
pub fn encode_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A staged node record: `(external id, labels, props)`.
pub(crate) type NodeRec = (String, Vec<String>, Vec<(String, Value)>);
/// A staged edge record: `(from-id, to-id, edge id?, labels, props)` — an edge's
/// first label is its type, the rest are secondary (multi-label edges).
pub(crate) type EdgeRec = (
    String,
    String,
    Option<String>,
    Vec<String>,
    Vec<(String, Value)>,
);

/// The decoded-but-not-yet-applied contents of a full-graph document. Shared by
/// [`from_ndjson`] / [`merge_ndjson`] and the binary codec ([`crate::binary`]) so
/// every full-graph decoder feeds the one [`build_store`] path.
pub(crate) struct StagedNdjson {
    pub(crate) constraints: Vec<(String, Vec<String>)>,
    pub(crate) required: Vec<(String, String)>,
    pub(crate) nodes: Vec<NodeRec>,
    pub(crate) edges: Vec<EdgeRec>,
}

/// Parse an NDJSON document into staged records. External ids are PRESERVED
/// verbatim (no remap), so element_id / egress round-trip.
fn stage_ndjson(text: &str) -> Result<StagedNdjson, String> {
    let mut constraints: Vec<(String, Vec<String>)> = Vec::new();
    let mut required: Vec<(String, String)> = Vec::new();
    let mut nodes: Vec<NodeRec> = Vec::new();
    let mut edges: Vec<EdgeRec> = Vec::new();

    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let err = |m: String| format!("line {}: {m}", lineno + 1);
        let Json::Obj(fields) = parse_line(line).map_err(err)? else {
            return Err(err("expected a JSON object".into()));
        };
        if let Some(kind) = field(&fields, "schema") {
            // Schema line (leads the snapshot): "unique" or "required".
            match json_string(kind).map_err(err)?.as_str() {
                "unique" => {
                    let label = json_string(req(&fields, "label").map_err(err)?).map_err(err)?;
                    let keys = json_str_array(req(&fields, "keys").map_err(err)?).map_err(err)?;
                    constraints.push((label, keys));
                }
                "required" => {
                    let label = json_string(req(&fields, "label").map_err(err)?).map_err(err)?;
                    let key = json_string(req(&fields, "key").map_err(err)?).map_err(err)?;
                    required.push((label, key));
                }
                _ => return Err(err("unknown schema kind".into())),
            }
        } else {
            // A data line. Two shapes are accepted so the engine is a drop-in for
            // core's NDJSON AND keeps reading anything it wrote earlier:
            //   core / shipped:  {"type":"node|edge", …, "labels":[…], "properties":{…}}
            //   engine (legacy): node {"id",…,"props"}; edge {"from","to","type":"<etype>","props"}
            // Edges are told by a `from` field (both shapes have it). Properties come
            // from `properties` OR `props` (default empty); an edge's type SET is its
            // `labels` array, or — legacy — a single `type` string (not "edge").
            let props_of =
                |f: &[(String, Json)]| match field(f, "properties").or_else(|| field(f, "props")) {
                    Some(p) => json_props(p),
                    None => Ok(Vec::new()),
                };
            if field(&fields, "from").is_some() {
                let from = json_id(req(&fields, "from").map_err(err)?).map_err(err)?;
                let to = json_id(req(&fields, "to").map_err(err)?).map_err(err)?;
                let edge_id = field(&fields, "id").map(json_id).transpose().map_err(err)?;
                let ty = field(&fields, "type")
                    .map(json_string)
                    .transpose()
                    .map_err(err)?;
                let labels: Vec<String> = match field(&fields, "labels") {
                    Some(l) => json_str_array(l).map_err(err)?,
                    // Legacy single-type edge: `type` is the edge type (never "edge").
                    None => match ty.filter(|t| t != "edge") {
                        Some(t) => vec![t],
                        None => return Err(err("edge needs `labels` (or a legacy `type`)".into())),
                    },
                };
                if labels.is_empty() {
                    return Err(err("edge `labels` must have at least one entry".into()));
                }
                edges.push((from, to, edge_id, labels, props_of(&fields).map_err(err)?));
            } else {
                let ext = json_id(req(&fields, "id").map_err(err)?).map_err(err)?;
                let labels = json_str_array(req(&fields, "labels").map_err(err)?).map_err(err)?;
                nodes.push((ext, labels, props_of(&fields).map_err(err)?));
            }
        }
    }
    Ok(StagedNdjson {
        constraints,
        required,
        nodes,
        edges,
    })
}

/// What a [`merge_ndjson`] applied vs. skipped — so a caller (the sync demand-fill
/// path) can surface anything that did not land cleanly. Empty skip/phantom lists
/// mean a clean merge. Mirrors lenke-core's `MergeReport`.
#[derive(Default)]
pub struct MergeReport {
    pub nodes_added: usize,
    pub edges_added: usize,
    /// Batch node ids skipped because the id already existed (first-wins).
    pub nodes_skipped: Vec<String>,
    /// Batch edge ids skipped because the explicit id already existed.
    pub edges_skipped: Vec<String>,
    /// Endpoint ids an edge referenced that were not declared — created as bare
    /// vertices (the lenient endpoint policy) and reported.
    pub phantom_vertices: Vec<String>,
}

/// Merge an NDJSON document into an EXISTING store with **first-wins** semantics
/// (matching lenke-core's `ndjson::append`): a node whose id already exists is
/// SKIPPED (the graph's copy kept) and reported; an edge with an already-present
/// explicit id is skipped; an undeclared edge endpoint is created as a bare vertex
/// and reported as a phantom; explicit edge ids are preserved. An id-less edge is
/// always inserted (it has no identity to dedup on). Schema lines (constraints)
/// are applied if present. This is the bulk demand-fill path — NOT an upsert; the
/// keyed upsert is GQL `_MERGE`.
pub fn merge_ndjson(store: &mut Store, text: &str) -> Result<MergeReport, String> {
    let staged = stage_ndjson(text)?;
    let mut report = MergeReport::default();

    for (label, keys) in &staged.constraints {
        let krefs: Vec<&str> = keys.iter().map(String::as_str).collect();
        store.create_unique_constraint(label, &krefs)?;
    }
    for (label, key) in &staged.required {
        store.create_required_constraint(label, key)?;
    }

    // Nodes first (first-wins), so a same-batch edge can reference one.
    for (ext, labels, props) in &staged.nodes {
        if store.node_by_ext(ext).is_some() {
            report.nodes_skipped.push(ext.clone()); // first-wins: existing kept
            continue;
        }
        let lrefs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let prefs: Vec<(&str, Value)> =
            props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        store.add_node_with_id(&Arc::from(ext.as_str()), &lrefs, &prefs);
        report.nodes_added += 1;
    }

    // Edge external id → eid over live edges, so an explicit-id dedup is O(1).
    let mut ext_to_eid: std::collections::HashMap<Arc<str>, u32> = std::collections::HashMap::new();
    for eid in store.all_edges() {
        if let Some(ext) = store.edge_ext_id(eid) {
            ext_to_eid.insert(ext, eid);
        }
    }

    for (from, to, edge_id, labels, props) in &staged.edges {
        // An edge whose explicit id already exists is a duplicate — drop it.
        if let Some(id) = edge_id {
            if ext_to_eid.contains_key(id.as_str()) {
                report.edges_skipped.push(id.clone());
                continue;
            }
        }
        let f = resolve_or_phantom(store, from, &mut report);
        let t = resolve_or_phantom(store, to, &mut report);
        let etype = &labels[0];
        let eid = match edge_id {
            Some(id) => store.add_edge_with_id(&Arc::from(id.as_str()), f, t, etype),
            None => store.add_edge(f, t, etype),
        };
        if labels.len() > 1 {
            let extra: Vec<&str> = labels[1..].iter().map(String::as_str).collect();
            store.set_edge_extra_labels(eid, &extra);
        }
        for (k, v) in props {
            store.set_edge_prop(eid, k, v.clone());
        }
        if let Some(e) = edge_id {
            ext_to_eid.insert(Arc::from(e.as_str()), eid);
        }
        report.edges_added += 1;
    }
    store.rebuild_csr();
    store.rebuild_edge_num();
    Ok(report)
}

/// An endpoint id → its node, creating a bare vertex (and recording a phantom) for
/// an undeclared one — the lenient endpoint policy, matching core.
fn resolve_or_phantom(store: &mut Store, ext: &str, report: &mut MergeReport) -> u32 {
    if let Some(id) = store.node_by_ext(ext) {
        return id;
    }
    report.phantom_vertices.push(ext.to_string());
    store.add_node_with_id(&Arc::from(ext), &[], &[])
}

/// Load a store from NDJSON in the [`to_ndjson`] format. Node lines
/// (`{id,labels,props}`) come first, then edge lines (`{from,to,type,props}`).
///
/// The file's `id` values may have GAPS (a dump omits deleted nodes), so ids are
/// NOT preserved: nodes are inserted in file order and get fresh dense ids, and
/// edges are remapped through that file-id → new-id map. Consequently a dump of a
/// graph with deletions re-densifies on load — round-trip is exact for a gap-free
/// dump and STABLE from the first reload otherwise.
pub fn from_ndjson(text: &str) -> Result<Store, String> {
    build_store(stage_ndjson(text)?)
}

/// Build a `Store` from staged records — the shared tail of every full-graph
/// decoder ([`from_ndjson`] and [`crate::binary::from_binary`]). Applies schema
/// BEFORE data (the store is empty, so declaration always succeeds — INSERT-time
/// enforcement on the reloaded store matches), then finalizes the read overlays.
pub(crate) fn build_store(staged: StagedNdjson) -> Result<Store, String> {
    let StagedNdjson {
        constraints,
        required,
        nodes,
        edges,
    } = staged;

    let mut store = Store::default();
    for (label, keys) in &constraints {
        let krefs: Vec<&str> = keys.iter().map(String::as_str).collect();
        store.create_unique_constraint(label, &krefs)?;
    }
    for (label, key) in &required {
        store.create_required_constraint(label, key)?;
    }
    for (ext, labels, props) in &nodes {
        let lrefs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let prefs: Vec<(&str, Value)> =
            props.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        store.add_node_with_id(&Arc::from(ext.as_str()), &lrefs, &prefs);
    }
    for (from, to, edge_id, labels, props) in &edges {
        let f = store
            .node_by_ext(from)
            .ok_or_else(|| format!("edge references unknown node id {from}"))?;
        let t = store
            .node_by_ext(to)
            .ok_or_else(|| format!("edge references unknown node id {to}"))?;
        let etype = &labels[0];
        let eid = match edge_id {
            Some(id) => store.add_edge_with_id(&Arc::from(id.as_str()), f, t, etype),
            None => store.add_edge(f, t, etype),
        };
        if labels.len() > 1 {
            let extra: Vec<&str> = labels[1..].iter().map(String::as_str).collect();
            store.set_edge_extra_labels(eid, &extra);
        }
        for (k, v) in props {
            store.set_edge_prop(eid, k, v.clone());
        }
    }
    // Incremental loading left the CSR + numeric-edge overlays stale (edges arrive via
    // add_edge, which invalidates); rebuild both once so a loaded snapshot gets the
    // contiguous read path and the typed edge-property reads without a later rebuild.
    store.rebuild_csr();
    store.rebuild_edge_num();
    // Dictionary-encode categorical string columns now that every value is in — the
    // finalize that gives a bulk-loaded `city`/`dept`/`status` the code-based encoding
    // incremental adds skip (turns GROUP BY / DISTINCT / equality over them into u32
    // work instead of string hashing).
    store.dict_encode_columns();
    Ok(store)
}

// --- a tiny dependency-free JSON parser (one value per line) -----------------

/// A parsed JSON value. Public so other engine modules (e.g. [`crate::schema_op`])
/// can reuse the one hand-rolled, tested JSON parser instead of duplicating it.
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// Look up a field by key in a JSON object's `(key, value)` list.
pub fn field<'a>(fields: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}
/// Like [`field`], but a missing key is an `Err`.
pub fn req<'a>(fields: &'a [(String, Json)], key: &str) -> Result<&'a Json, String> {
    field(fields, key).ok_or_else(|| format!("missing field `{key}`"))
}
/// An element id, accepted as a JSON string OR a non-negative integer (rendered as
/// its integer text) — so both `"id":"e0"` and `"id":0` preserve a stable id.
fn json_id(j: &Json) -> Result<String, String> {
    match j {
        Json::Str(s) => Ok(s.clone()),
        Json::Num(n) if n.fract() == 0.0 => Ok((*n as i64).to_string()),
        _ => Err("expected an id (string or integer)".into()),
    }
}
/// A JSON string value, or `Err` for any other shape.
pub fn json_string(j: &Json) -> Result<String, String> {
    match j {
        Json::Str(s) => Ok(s.clone()),
        _ => Err("expected a string".into()),
    }
}
/// A JSON array of strings, or `Err` for any other shape.
pub fn json_str_array(j: &Json) -> Result<Vec<String>, String> {
    match j {
        Json::Arr(items) => items.iter().map(json_string).collect(),
        _ => Err("expected an array of strings".into()),
    }
}
fn json_props(j: &Json) -> Result<Vec<(String, Value)>, String> {
    match j {
        Json::Obj(fields) => fields
            .iter()
            .map(|(k, v)| Ok((k.clone(), json_value(v)?)))
            .collect(),
        _ => Err("expected an object".into()),
    }
}
/// A JSON value as a property `Value`. There is no map type, so a nested object
/// as a property value is rejected (the egress never emits one).
fn json_value(j: &Json) -> Result<Value, String> {
    Ok(match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Num(x) => Value::Num(*x),
        Json::Str(s) => Value::Str(GStr::from(s.as_str())),
        Json::Arr(items) => Value::List(items.iter().map(json_value).collect::<Result<_, _>>()?),
        // A single-key `{"@<tag>":"<iso>"}` object is a tagged temporal — the
        // inverse of the egress in `encode_value`. Any other object shape is not a
        // property value.
        Json::Obj(fields) => {
            if let [(key, Json::Str(s))] = fields.as_slice() {
                if let Some(tag) = key.strip_prefix('@') {
                    if matches!(
                        tag,
                        "date"
                            | "localtime"
                            | "datetime"
                            | "zoned_time"
                            | "zoned_datetime"
                            | "duration"
                    ) {
                        return crate::temporal::Temporal::parse(tag, s)
                            .map(Value::Temporal)
                            .map_err(|e| format!("bad temporal value: {e}"));
                    }
                }
            }
            // Any other object is a record: decode each field recursively, then
            // canonicalize (sorted, last-wins) via the value contract.
            let pairs = fields
                .iter()
                .map(|(k, v)| json_value(v).map(|v| (Arc::from(k.as_str()), v)))
                .collect::<Result<Vec<_>, _>>()?;
            crate::value::make_record(pairs)
        }
    })
}

/// Parse exactly one JSON value from `text` (trailing whitespace allowed). The
/// public entry to the engine's one JSON parser, for callers outside NDJSON decode.
pub fn parse_json(text: &str) -> Result<Json, String> {
    parse_line(text)
}

/// Parse a JSON object of query parameters `{"name": value, …}` into `(name, value)`
/// pairs. Each value decodes with the same rules as a stored property value (scalars,
/// lists, records, tagged temporals). A non-object is an error.
pub fn parse_params(json: &str) -> Result<Vec<(String, Value)>, String> {
    params_from_obj(&parse_json(json)?)
}

/// Convert an ALREADY-parsed JSON object into query-parameter `(name, value)`
/// pairs — the nested-value counterpart of [`parse_params`] (which takes text),
/// for callers that hold a `params` sub-object (e.g. a prepared-statement payload).
pub fn params_from_obj(obj: &Json) -> Result<Vec<(String, Value)>, String> {
    match obj {
        Json::Obj(fields) => fields
            .iter()
            .map(|(k, v)| Ok((k.clone(), json_value(v)?)))
            .collect(),
        _ => Err("query parameters must be a JSON object".into()),
    }
}

/// Parse exactly one JSON value from `line` (trailing whitespace allowed).
fn parse_line(line: &str) -> Result<Json, String> {
    let mut p = JsonParser {
        b: line.chars().collect(),
        i: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing input at char {}", p.i));
    }
    Ok(v)
}

struct JsonParser {
    b: Vec<char>,
    i: usize,
}

impl JsonParser {
    fn ws(&mut self) {
        while self.b.get(self.i).is_some_and(|c| c.is_whitespace()) {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some('t') | Some('f') => self.boolean(),
            Some('n') => self.keyword("null", Json::Null),
            Some(c) if *c == '-' || c.is_ascii_digit() => self.number(),
            other => Err(format!("unexpected {other:?} at char {}", self.i)),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // '{'
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&'}') {
            self.i += 1;
            return Ok(Json::Obj(out));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&':') {
                return Err(format!("expected ':' at char {}", self.i));
            }
            self.i += 1;
            let val = self.value()?;
            out.push((key, val));
            self.ws();
            match self.b.get(self.i) {
                Some(',') => self.i += 1,
                Some('}') => {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                _ => return Err(format!("expected ',' or '}}' at char {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // '['
        let mut out = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&']') {
            self.i += 1;
            return Ok(Json::Arr(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(',') => self.i += 1,
                Some(']') => {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                _ => return Err(format!("expected ',' or ']' at char {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&'"') {
            return Err(format!("expected a string at char {}", self.i));
        }
        self.i += 1;
        let mut s = String::new();
        loop {
            match self.b.get(self.i) {
                None => return Err("unterminated string".into()),
                Some('"') => {
                    self.i += 1;
                    return Ok(s);
                }
                Some('\\') => {
                    self.i += 1;
                    match self.b.get(self.i) {
                        Some('"') => s.push('"'),
                        Some('\\') => s.push('\\'),
                        Some('/') => s.push('/'),
                        Some('n') => s.push('\n'),
                        Some('r') => s.push('\r'),
                        Some('t') => s.push('\t'),
                        Some('b') => s.push('\u{8}'),
                        Some('f') => s.push('\u{c}'),
                        Some('u') => {
                            let hex: String =
                                (1..=4).filter_map(|d| self.b.get(self.i + d)).collect();
                            let cp = u32::from_str_radix(&hex, 16)
                                .map_err(|_| "bad \\u escape".to_string())?;
                            s.push(char::from_u32(cp).ok_or("bad code point")?);
                            self.i += 4;
                        }
                        other => return Err(format!("bad escape {other:?}")),
                    }
                    self.i += 1;
                }
                Some(&c) => {
                    s.push(c);
                    self.i += 1;
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self
            .b
            .get(self.i)
            .is_some_and(|c| matches!(c, '0'..='9' | '-' | '+' | '.' | 'e' | 'E'))
        {
            self.i += 1;
        }
        let text: String = self.b[start..self.i].iter().collect();
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("bad number `{text}`"))
    }

    fn boolean(&mut self) -> Result<Json, String> {
        if self.b.get(self.i) == Some(&'t') {
            self.keyword("true", Json::Bool(true))
        } else {
            self.keyword("false", Json::Bool(false))
        }
    }

    fn keyword(&mut self, word: &str, val: Json) -> Result<Json, String> {
        for c in word.chars() {
            if self.b.get(self.i) != Some(&c) {
                return Err(format!("expected `{word}` at char {}", self.i));
            }
            self.i += 1;
        }
        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use super::{from_ndjson, merge_ndjson, snapshot, to_ndjson};
    use crate::store::Builder;
    use crate::value::Value;
    use std::sync::Arc;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }

    #[test]
    fn merge_into_empty_equals_from_ndjson() {
        let doc = "{\"id\":\"1\",\"labels\":[\"P\"],\"props\":{\"n\":\"a\"}}\n\
                   {\"id\":\"2\",\"labels\":[\"P\"],\"props\":{\"n\":\"b\"}}\n\
                   {\"from\":\"1\",\"to\":\"2\",\"id\":\"e0\",\"type\":\"R\",\"props\":{}}\n";
        let mut merged = Builder::default().build();
        merge_ndjson(&mut merged, doc).unwrap();
        assert_eq!(to_ndjson(&merged), to_ndjson(&from_ndjson(doc).unwrap()));
    }

    #[test]
    fn merge_first_wins_on_existing_node() {
        let mut st = from_ndjson(
            "{\"id\":\"1\",\"labels\":[\"P\"],\"props\":{\"n\":\"a\",\"k\":\"keep\"}}\n",
        )
        .unwrap();
        let report = merge_ndjson(
            &mut st,
            "{\"id\":\"1\",\"labels\":[\"P\"],\"props\":{\"n\":\"z\",\"age\":5}}\n",
        )
        .unwrap();
        let id = st.node_by_ext("1").unwrap();
        let v = |x: Value| format!("{x:?}"); // Value is not PartialEq — compare via Debug
                                             // First-wins: the existing node is kept UNCHANGED and reported as skipped.
        assert_eq!(
            v(st.prop(id, "n")),
            v(s("a")),
            "existing kept, not overwritten"
        );
        assert_eq!(
            v(st.prop(id, "age")),
            v(Value::Null),
            "no new key merged in"
        );
        assert_eq!(v(st.prop(id, "k")), v(s("keep")));
        assert_eq!(st.node_count(), 1);
        assert_eq!(report.nodes_skipped, vec!["1"]);
        assert_eq!(report.nodes_added, 0);
    }

    #[test]
    fn merge_adds_unknown_node_and_skips_edge_by_id() {
        let mut st = from_ndjson(
            "{\"id\":\"1\",\"labels\":[\"P\"],\"props\":{}}\n\
             {\"id\":\"2\",\"labels\":[\"P\"],\"props\":{}}\n\
             {\"from\":\"1\",\"to\":\"2\",\"id\":\"e0\",\"type\":\"R\",\"props\":{\"w\":1}}\n",
        )
        .unwrap();
        let report = merge_ndjson(
            &mut st,
            "{\"id\":\"3\",\"labels\":[\"P\"],\"props\":{}}\n\
             {\"from\":\"1\",\"to\":\"2\",\"id\":\"e0\",\"type\":\"R\",\"props\":{\"w\":9}}\n",
        )
        .unwrap();
        assert_eq!(st.node_count(), 3, "node 3 inserted");
        assert_eq!(
            st.edge_count(),
            1,
            "e0 skipped (first-wins), not duplicated"
        );
        let eid = st.all_edges()[0];
        // First-wins: the existing edge's property is kept.
        assert_eq!(
            format!("{:?}", st.edge_prop(eid, "w")),
            format!("{:?}", Value::Num(1.0))
        );
        assert_eq!(report.nodes_added, 1);
        assert_eq!(report.edges_skipped, vec!["e0"]);
    }

    #[test]
    fn merge_auto_creates_a_phantom_endpoint() {
        // An edge to an undeclared node creates a bare vertex + reports it.
        let mut st = from_ndjson("{\"id\":\"1\",\"labels\":[\"P\"],\"props\":{}}\n").unwrap();
        let report = merge_ndjson(
            &mut st,
            "{\"from\":\"1\",\"to\":\"ghost\",\"type\":\"R\",\"props\":{}}\n",
        )
        .unwrap();
        assert!(
            st.node_by_ext("ghost").is_some(),
            "phantom endpoint created"
        );
        assert_eq!(report.phantom_vertices, vec!["ghost"]);
        assert_eq!(report.edges_added, 1);
    }

    /// A required constraint survives a snapshot round trip: it dumps a schema
    /// line and the reloaded store re-enforces it.
    #[test]
    fn required_constraint_survives_snapshot() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("a@x"))]);
        st.create_required_constraint("User", "email").unwrap();
        let snap = snapshot(&st);
        assert!(
            snap.contains("{\"schema\":\"required\",\"label\":\"User\",\"key\":\"email\"}"),
            "schema was: {snap}"
        );
        let mut st2 = from_ndjson(&snap).unwrap();
        // The reloaded constraint still bites: a User with no email violates it.
        st2.add_node(&["User"], &[("name", s("b"))]);
        assert!(st2.check_required_for_label("User").is_err());
    }
    fn n(x: f64) -> Value {
        Value::Num(x)
    }

    /// A small graph dumps to exactly these lines — hand-written. Property keys
    /// are sorted (age before name); a node without a key omits it.
    #[test]
    fn dumps_nodes_and_edges() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a")), ("age", n(1.0))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let eid = st.add_edge(a, b, "R");
        st.set_edge_prop(eid, "weight", n(0.5));
        let expected = "{\"type\":\"node\",\"id\":\"0\",\"labels\":[\"P\"],\"properties\":{\"age\":1,\"name\":\"a\"}}\n\
             {\"type\":\"node\",\"id\":\"1\",\"labels\":[\"P\"],\"properties\":{\"name\":\"b\"}}\n\
             {\"type\":\"edge\",\"id\":\"e0\",\"from\":\"0\",\"to\":\"1\",\"labels\":[\"R\"],\"properties\":{\"weight\":0.5}}\n";
        assert_eq!(to_ndjson(&st), expected);
    }

    /// A temporal property survives an NDJSON round trip: it dumps to the tagged
    /// form and decodes back to the same value.
    #[test]
    fn temporal_props_round_trip() {
        use crate::temporal::{Date, Temporal};
        let mut st = Builder::default().build();
        st.add_node(
            &["P"],
            &[(
                "born",
                Value::Temporal(Temporal::Date(Date::parse("1990-05-01").unwrap())),
            )],
        );
        let text = to_ndjson(&st);
        assert!(
            text.contains("\"born\":{\"@date\":\"1990-05-01\"}"),
            "egress was: {text}"
        );
        let st2 = from_ndjson(&text).unwrap();
        match st2.prop(0, "born") {
            Value::Temporal(Temporal::Date(d)) => assert_eq!(d.format(), "1990-05-01"),
            o => panic!("expected a Date after round trip, got {o:?}"),
        }
    }

    /// A record property survives an NDJSON round trip: it dumps to a JSON object
    /// (keys sorted) and decodes back to the same record.
    #[test]
    fn record_props_round_trip() {
        use crate::value::make_record;
        let mut st = Builder::default().build();
        st.add_node(
            &["P"],
            &[(
                "meta",
                make_record(vec![(Arc::from("y"), s("hi")), (Arc::from("x"), n(1.0))]),
            )],
        );
        let text = to_ndjson(&st);
        assert!(
            text.contains("\"meta\":{\"x\":1,\"y\":\"hi\"}"),
            "egress was: {text}"
        );
        let st2 = from_ndjson(&text).unwrap();
        match st2.prop(0, "meta") {
            Value::Record(f) => {
                assert_eq!(f[0].0.as_ref(), "x");
                assert!(crate::value::equals(&f[0].1, &n(1.0)));
                assert_eq!(f[1].0.as_ref(), "y");
                assert!(crate::value::equals(&f[1].1, &s("hi")));
            }
            o => panic!("expected a Record after round trip, got {o:?}"),
        }
    }

    /// A deleted node (and its edges) is absent from the dump.
    #[test]
    fn deleted_node_excluded() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        st.add_edge(a, b, "R");
        st.delete_node(b);
        let expected =
            "{\"type\":\"node\",\"id\":\"0\",\"labels\":[\"P\"],\"properties\":{\"name\":\"a\"}}\n";
        assert_eq!(to_ndjson(&st), expected);
    }

    /// Strings are escaped; a node with no labels/props emits empty `[]`/`{}`;
    /// bool, null, and list values encode as JSON.
    #[test]
    fn escaping_and_value_kinds() {
        let mut st = Builder::default().build();
        st.add_node(
            &[],
            &[
                ("q", s("a\"b\nc")),
                ("ok", Value::Bool(true)),
                ("z", Value::Null),
                ("xs", Value::List(vec![n(1.0), s("y")])),
            ],
        );
        let out = to_ndjson(&st);
        // keys sorted: ok, q, xs, z
        let expected = "{\"type\":\"node\",\"id\":\"0\",\"labels\":[],\"properties\":\
             {\"ok\":true,\"q\":\"a\\\"b\\nc\",\"xs\":[1,\"y\"],\"z\":null}}\n";
        assert_eq!(out, expected);
    }

    /// Non-finite numbers have no JSON form and are written as null.
    #[test]
    fn non_finite_number_is_null() {
        let mut st = Builder::default().build();
        st.add_node(&[], &[("v", n(f64::NAN))]);
        assert_eq!(
            to_ndjson(&st),
            "{\"type\":\"node\",\"id\":\"0\",\"labels\":[],\"properties\":{\"v\":null}}\n"
        );
    }

    /// An empty store dumps to the empty string.
    #[test]
    fn empty_store() {
        let st = Builder::default().build();
        assert_eq!(to_ndjson(&st), "");
    }

    // --- ingest / round-trip (C2) ---

    /// dump → parse → dump is IDENTITY for a gap-free graph.
    #[test]
    fn round_trip_is_identity_gap_free() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a")), ("age", n(1.0))]);
        let b = st.add_node(&["P", "Q"], &[("name", s("b"))]);
        let e = st.add_edge(a, b, "R");
        st.set_edge_prop(e, "weight", n(0.5));
        let d1 = to_ndjson(&st);
        let st2 = from_ndjson(&d1).unwrap();
        assert_eq!(d1, to_ndjson(&st2));
    }

    /// Parsing a hand-written document reconstructs the graph exactly.
    #[test]
    fn parse_reconstructs_graph() {
        let doc = "{\"id\":0,\"labels\":[\"P\"],\"props\":{\"age\":1,\"name\":\"a\"}}\n\
                   {\"id\":1,\"labels\":[],\"props\":{}}\n\
                   {\"from\":0,\"to\":1,\"type\":\"R\",\"props\":{\"weight\":0.5}}\n";
        let st = from_ndjson(doc).unwrap();
        assert_eq!(st.node_count(), 2);
        assert_eq!(st.nodes_with_label("P"), &[0]);
        assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "a"));
        assert!(matches!(st.prop(0, "age"), Value::Num(x) if x == 1.0));
        assert_eq!(st.out(0).len(), 1);
        assert_eq!(st.out(0)[0].nbr, 1);
        let eid = st.out(0)[0].eid;
        assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 0.5));
    }

    /// A dump with a deleted node has an id GAP; ingest remaps to dense ids
    /// (edges follow the remap) and is stable from the first reload.
    #[test]
    fn gapped_ids_remap_and_stabilize() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let c = st.add_node(&["P"], &[("name", s("c"))]);
        st.add_edge(a, c, "R"); // 0 -> 2
        st.delete_node(b); // gap at id 1
        let d1 = to_ndjson(&st); // node ids 0 and 2
        let st2 = from_ndjson(&d1).unwrap(); // remapped to 0, 1
        assert_eq!(st2.node_count(), 2);
        assert_eq!(st2.out(0).len(), 1);
        assert!(matches!(st2.prop(st2.out(0)[0].nbr, "name"), Value::Str(x) if &*x == "c"));
        // stable from here on
        let d2 = to_ndjson(&st2);
        let st3 = from_ndjson(&d2).unwrap();
        assert_eq!(d2, to_ndjson(&st3));
    }

    /// Every value kind (escaped string, bool, null, list) round-trips.
    #[test]
    fn value_kinds_round_trip() {
        let mut st = Builder::default().build();
        st.add_node(
            &[],
            &[
                ("q", s("a\"b\nc\ttab\\end")),
                ("ok", Value::Bool(true)),
                ("z", Value::Null),
                ("xs", Value::List(vec![n(1.0), s("y"), Value::Bool(false)])),
            ],
        );
        let d1 = to_ndjson(&st);
        let st2 = from_ndjson(&d1).unwrap();
        assert_eq!(d1, to_ndjson(&st2));
        assert!(matches!(st2.prop(0, "q"), Value::Str(x) if &*x == "a\"b\nc\ttab\\end"));
        assert!(matches!(st2.prop(0, "xs"), Value::List(v) if v.len() == 3));
    }

    #[test]
    fn malformed_lines_error() {
        assert!(from_ndjson("{\"id\":0,\"labels\":[],\"props\":{}").is_err()); // missing }
        assert!(from_ndjson("not json").is_err());
        assert!(from_ndjson("{\"nope\":1}").is_err()); // neither schema/id/from
    }

    // --- schema / snapshot (C3) ---

    /// `dump_schema` emits one line per unique constraint, sorted by (label,keys).
    #[test]
    fn dump_schema_lines() {
        let mut st = Builder::default().build();
        st.create_unique_constraint("User", &["email"]).unwrap();
        st.create_unique_constraint("Doc", &["id", "rev"]).unwrap();
        let expected = "{\"schema\":\"unique\",\"label\":\"Doc\",\"keys\":[\"id\",\"rev\"]}\n\
             {\"schema\":\"unique\",\"label\":\"User\",\"keys\":[\"email\"]}\n";
        assert_eq!(super::dump_schema(&st), expected);
    }

    /// A snapshot is schema lines first, then the data.
    #[test]
    fn snapshot_is_schema_then_data() {
        let mut st = Builder::default().build();
        st.create_unique_constraint("P", &["k"]).unwrap();
        st.add_node(&["P"], &[("k", n(1.0))]);
        let expected = "{\"schema\":\"unique\",\"label\":\"P\",\"keys\":[\"k\"]}\n\
             {\"type\":\"node\",\"id\":\"0\",\"labels\":[\"P\"],\"properties\":{\"k\":1}}\n";
        assert_eq!(super::snapshot(&st), expected);
    }

    /// A full snapshot round-trips BOTH data and schema: reload is byte-identical,
    /// and the reloaded constraint still rejects a violating INSERT.
    #[test]
    fn snapshot_round_trip_preserves_schema_and_data() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        st.create_unique_constraint("User", &["email"]).unwrap();
        let a = st.add_node(&["User"], &[("email", s("a@x")), ("name", s("A"))]);
        let b = st.add_node(&["User"], &[("email", s("b@x"))]);
        st.add_edge(a, b, "KNOWS");
        let snap = super::snapshot(&st);

        let mut st2 = super::load_snapshot(&snap).unwrap();
        assert_eq!(super::snapshot(&st2), snap); // data + schema identical

        // The constraint survived: a duplicate INSERT on the reloaded store errors.
        let dup = crate::gql::parse("INSERT (:User {email: 'a@x'})").unwrap();
        assert!(execute(&dup, &mut st2).is_err());
        // …and a fresh email still inserts fine.
        let ok = crate::gql::parse("INSERT (:User {email: 'c@x'})").unwrap();
        assert!(execute(&ok, &mut st2).is_ok());
    }
}
