//! NDJSON codec for the columnar core. One JSON object per line, tagged
//! `type:"node"|"edge"`. Decoding parses lines **in parallel** (rayon) — the
//! axis single-threaded JS can't match — then assembles serially.
//!
//! Scope note: an edge's *type* is its first label. Edge **properties** are
//! supported (same columnar store as vertex properties). A property value that
//! is a nested JSON object is a first-class map/record value (canonicalized to
//! sorted keys on store); a single-key `{"@date":…}` object is still a tagged
//! temporal scalar.

use std::borrow::Cow;
use std::sync::Arc;

use crate::error::{CodeError, CodeResult};
use crate::error_codes::ErrorCode;
use crate::graph::{Builder, Column, Dict, EdgeRec, Graph, NodeRec, Properties, Value};
use crate::jsonfmt::{push_json_str, push_num};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::json::{self, Json};

fn to_value(j: &Json) -> CodeResult<Value> {
    Ok(match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        // A JSON number that overflowed to ±Infinity (e.g. `1e400`) — or a NaN —
        // is not representable in the LPG numeric model, so it maps to `null` at
        // this entry point (matching the TS ndjson/pg-json codecs and the shared
        // `codec::json_to_value`). Storing a real non-finite float would silently
        // corrupt count/sum/min/max/`IS NULL`/comparisons and diverge from TS.
        Json::Num(n) => {
            if n.is_finite() {
                Value::Num(*n)
            } else {
                Value::Null
            }
        }
        Json::Str(s) => Value::Str(Arc::from(s.as_ref())),
        Json::Arr(a) => Value::List(a.iter().map(to_value).collect::<CodeResult<Vec<_>>>()?),
        // A tagged temporal `{"@date":"…"}` (single key) round-trips as a scalar;
        // any other JSON object is a record/map value (canonicalized on store).
        Json::Obj(pairs) => match json::temporal_from_pairs(pairs) {
            Some(res) => {
                Value::Temporal(res.map_err(|e| CodeError::new(ErrorCode::InvalidValue, e))?)
            }
            None => Value::Map(
                pairs
                    .iter()
                    .map(|(k, v)| Ok((Arc::from(k.as_ref()), to_value(v)?)))
                    .collect::<CodeResult<Vec<_>>>()?,
            ),
        },
    })
}

/// Decode a JSON object's `properties` field into core property pairs, or an
/// empty vec when absent. A nested-object value becomes a map/record property.
fn props_of<'a>(obj: &Json<'a>) -> CodeResult<Vec<(Cow<'a, str>, Value)>> {
    match obj.get("properties").and_then(Json::as_object) {
        Some(m) => m
            .iter()
            .map(|(k, v)| Ok((k.clone(), to_value(v)?)))
            .collect(),
        None => Ok(Vec::new()),
    }
}

/// A JSON id field as a string (a string verbatim; a number/bool/null via its
/// JSON text — matching serde_json's `Display`).
fn as_id<'a>(j: &Json<'a>) -> Cow<'a, str> {
    match j {
        Json::Str(s) => s.clone(),
        Json::Num(n) => Cow::Owned(crate::jsonfmt::js_number(*n)),
        Json::Bool(b) => Cow::Owned(b.to_string()),
        _ => Cow::Borrowed("null"),
    }
}

enum Rec<'a> {
    Node(NodeRec<'a>),
    Edge(EdgeRec<'a>),
}

/// Parse one line. A blank line is skipped (`Ok(None)`); everything else is
/// strict and matches the TS codec: invalid JSON → `InvalidJson`, a non-object
/// or an unknown/missing `type` → `InvalidShape`. (Previously these all silently
/// skipped the line, which could mask corrupt fixtures since `decode` is the
/// crate's test-fixture loader.)
fn parse_line(line: &str) -> CodeResult<Option<Rec<'_>>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let snippet = || line.chars().take(80).collect::<String>();
    let Ok(j) = json::parse(line) else {
        return Err(CodeError::new(
            ErrorCode::InvalidJson,
            format!("ndjson: invalid JSON: {}", snippet()),
        ));
    };
    if j.as_object().is_none() {
        return Err(CodeError::new(
            ErrorCode::InvalidShape,
            format!(
                "ndjson: each line must be a node or edge object: {}",
                snippet()
            ),
        ));
    }
    let rec = match j.get("type").and_then(Json::as_str) {
        Some("node") => {
            let id = j.get("id").map(as_id).unwrap_or_default();
            let labels = j
                .get("labels")
                .and_then(Json::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|x| match x {
                            Json::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            Rec::Node(NodeRec {
                id,
                labels,
                props: props_of(&j)?,
            })
        }
        Some("edge") => {
            let src = j.get("from").map(as_id).unwrap_or_default();
            let dst = j.get("to").map(as_id).unwrap_or_default();
            // Edges are MULTI-label, like vertices. This used to take only
            // `.first()` and silently drop the rest, so a two-label edge
            // round-tripped as one and `[:SECOND]` matched nothing — while the
            // TS engine, whose `Edge.labels` is a `Set`, kept both.
            let mut names = j
                .get("labels")
                .and_then(Json::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|x| match x {
                            Json::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
                .into_iter();
            let etype = names.next().unwrap_or(Cow::Borrowed(""));
            let extra_labels: Vec<Cow<'_, str>> = names.collect();
            // Optional external edge id (absent ⇒ id-less, stays lazy).
            let id = match j.get("id") {
                Some(Json::Str(s)) => Some(s.clone()),
                _ => None,
            };
            Rec::Edge(EdgeRec {
                src,
                dst,
                etype,
                extra_labels,
                props: props_of(&j)?,
                id,
            })
        }
        _ => {
            return Err(CodeError::new(
                ErrorCode::InvalidShape,
                format!(
                    "ndjson: line is not a 'node' or 'edge' record: {}",
                    snippet()
                ),
            ))
        }
    };
    Ok(Some(rec))
}

/// Decode NDJSON into a columnar graph. Lines parse in parallel; the build is
/// serial (shared dictionaries).
pub fn decode(text: &str) -> CodeResult<Graph> {
    // `collect` into a Result short-circuits on the first InvalidValue (rayon's
    // parallel collect supports this), so one bad line fails the whole decode.
    #[cfg(feature = "parallel")]
    let recs: Vec<Option<Rec>> = text
        .par_lines()
        .map(parse_line)
        .collect::<CodeResult<_>>()?;
    #[cfg(not(feature = "parallel"))]
    let recs: Vec<Option<Rec>> = text.lines().map(parse_line).collect::<CodeResult<_>>()?;
    let mut b = Builder::default();
    for r in recs.into_iter().flatten() {
        match r {
            Rec::Node(n) => b.nodes.push(n),
            Rec::Edge(e) => b.edges.push(e),
        }
    }
    // A one-shot decode rejects a dangling edge (an endpoint never declared as a
    // node) instead of fabricating a phantom vertex — a truly-missing endpoint is
    // bad input, and this matches the pg-json/graphson/csv document codecs. The
    // streaming *merge* path (`decode_into`) keeps the lenient create-and-report
    // policy, where an endpoint can legitimately arrive in a later batch.
    let g = b.finalize_strict()?;
    g.validate_wellformed()?; // reject a malformed label / edge type / property key
    Ok(g)
}

/// Bulk-append the NDJSON records in `text` to an **existing** graph — a
/// `COPY FROM` into a live store, the incremental twin of [`decode`].
///
/// Semantics match `decode(encode(graph) + "\n" + text)`: a node whose id already
/// exists is **first-wins** (skipped, the graph's copy kept), an edge with an
/// already-present explicit id is dropped, an undeclared edge endpoint gets a
/// bare vertex (the lenient policy `decode` uses), and explicit edge ids are
/// preserved. It drives the graph's own append machinery (so property indexes
/// stay current and the version bumps per element) — but with no per-record
/// parse or FFI crossing, so it runs at bulk speed, not per-`INSERT` speed.
pub fn append(graph: &mut Graph, text: &str) -> CodeResult<MergeReport> {
    // Parse the lines in parallel — the same rayon fan-out `decode` uses, closing
    // the ~25% gap the serial parse cost. Order is preserved (so first-wins node
    // dedupe is deterministic), and the apply below stays serial (it mutates the
    // shared graph). One bad line short-circuits the whole batch.
    #[cfg(feature = "parallel")]
    let parsed: Vec<Option<Rec>> = text
        .par_lines()
        .map(parse_line)
        .collect::<CodeResult<_>>()?;
    #[cfg(not(feature = "parallel"))]
    let parsed: Vec<Option<Rec>> = text.lines().map(parse_line).collect::<CodeResult<_>>()?;
    let recs: Vec<Rec> = parsed.into_iter().flatten().collect();
    let mut report = MergeReport::default();

    // Nodes first, so an edge may reference a same-batch node declared in any
    // order (mirrors decode's "declared nodes, then edge endpoints" indexing).
    for r in &recs {
        if let Rec::Node(n) = r {
            if graph.vertex_by_id(&n.id).is_some() {
                report.nodes_skipped.push(n.id.to_string()); // first-wins: existing kept
            } else {
                // The incremental path writes through the graph's owned mutation
                // API, so the borrowed record is materialized here. Only the bulk
                // `decode` path builds columns directly and keeps the borrow.
                let labels: Vec<String> = n.labels.iter().map(ToString::to_string).collect();
                let props: Vec<(String, Value)> = n
                    .props
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect();

                graph.add_vertex_with_id(&n.id, &labels, props);
                report.nodes_added += 1;
            }
        }
    }
    for r in &recs {
        if let Rec::Edge(e) = r {
            if let Some(id) = &e.id {
                if graph.edge_by_id(id).is_some() {
                    report.edges_skipped.push(id.to_string()); // duplicate id → drop
                    continue;
                }
            }
            let from = resolve_or_create(graph, &e.src, &mut report);
            let to = resolve_or_create(graph, &e.dst, &mut report);
            let eprops: Vec<(String, Value)> = e
                .props
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();
            let ei = if e.extra_labels.is_empty() {
                graph.add_edge(from, to, &e.etype, eprops)
            } else {
                let mut names: Vec<&str> = vec![&e.etype];

                names.extend(e.extra_labels.iter().map(std::convert::AsRef::as_ref));
                graph.add_edge_labelled(from, to, &names, eprops)
            };
            if let Some(id) = &e.id {
                graph.set_edge_id(ei, id);
            }
            report.edges_added += 1;
        }
    }

    graph.validate_wellformed()?;
    Ok(report)
}

/// A vertex id → its dense index, creating a bare (label-less, prop-less) vertex
/// on demand — the lenient endpoint policy `decode` uses for an edge that names
/// an undeclared node — and recording it as a phantom in the report.
fn resolve_or_create(graph: &mut Graph, id: &str, report: &mut MergeReport) -> u32 {
    match graph.vertex_by_id(id) {
        Some(vi) => vi,
        None => {
            report.phantom_vertices.push(id.to_string());
            graph.add_vertex_with_id(id, &[], Vec::new())
        }
    }
}

/// What an [`append`] applied vs. skipped — so a caller sees anything that
/// didn't land cleanly. Empty `*_skipped`/`phantom_vertices` = a clean merge.
#[derive(Debug, Default, Clone)]
pub struct MergeReport {
    /// Vertices actually inserted.
    pub nodes_added: usize,
    /// Edges actually inserted.
    pub edges_added: usize,
    /// Batch node ids skipped because the id already existed (first-wins).
    pub nodes_skipped: Vec<String>,
    /// Batch edge ids dropped because that explicit id already existed.
    pub edges_skipped: Vec<String>,
    /// Ids the batch used as an edge endpoint but never declared as a node —
    /// created as bare (label-less, prop-less) vertices.
    pub phantom_vertices: Vec<String>,
}

impl MergeReport {
    /// The report as JSON (camelCase keys) for the FFI / napi boundary.
    pub fn to_json(&self) -> String {
        let arr = |out: &mut String, items: &[String]| {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_str(out, it);
            }
            out.push(']');
        };
        let mut s = format!(
            "{{\"nodesAdded\":{},\"edgesAdded\":{},\"nodesSkipped\":",
            self.nodes_added, self.edges_added
        );
        arr(&mut s, &self.nodes_skipped);
        s.push_str(",\"edgesSkipped\":");
        arr(&mut s, &self.edges_skipped);
        s.push_str(",\"phantomVertices\":");
        arr(&mut s, &self.phantom_vertices);
        s.push('}');
        s
    }
}

/// Decode without parallelism — for isolating rayon's contribution in the bench.
pub fn decode_serial(text: &str) -> CodeResult<Graph> {
    let mut b = Builder::default();
    for line in text.lines() {
        match parse_line(line)? {
            Some(Rec::Node(n)) => b.nodes.push(n),
            Some(Rec::Edge(e)) => b.edges.push(e),
            None => {}
        }
    }
    let g = b.finalize_strict()?; // reject a dangling edge — see `decode`
    g.validate_wellformed()?;
    Ok(g)
}

fn push_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => push_num(out, *x),
        Value::Str(s) => push_json_str(out, s),
        Value::Temporal(t) => out.push_str(&t.json_tagged()),
        Value::List(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_value(out, e);
            }
            out.push(']');
        }
        // A record/map property → a JSON object. Keys are already canonical
        // (sorted) from the store, so emit in order.
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, e)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_str(out, k);
                out.push(':');
                push_value(out, e);
            }
            out.push('}');
        }
    }
}

/// Is property `col` present at element `idx`?
fn col_present(col: &Column, idx: usize) -> bool {
    match col {
        Column::Num { present, .. }
        | Column::Str { present, .. }
        | Column::Bool { present, .. }
        | Column::Temporal { present, .. }
        | Column::Vec { present, .. } => present.get(idx),
        Column::Mixed { data } => data[idx].is_some(),
        Column::Record {
            present, escaped, ..
        } => escaped.contains_key(&(idx as u32)) || present.get(idx),
    }
}

/// Emit the `{...}` body of an element's properties from a columnar store —
/// shared by node and edge encoding. `strs` backs the string columns.
fn push_props(out: &mut String, store: &Properties, strs: &Dict, idx: usize) {
    out.push('{');
    let mut first = true;
    for (kid, col) in store.cols.iter().enumerate() {
        if !col_present(col, idx) {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        push_json_str(out, store.keys.text(kid as u32));
        out.push(':');
        match col {
            Column::Num { data, .. } => push_num(out, data[idx]),
            Column::Str { data, .. } => push_json_str(out, strs.text(data[idx])),
            Column::Bool { data, .. } => out.push_str(if data[idx] { "true" } else { "false" }),
            Column::Temporal { data, .. } => push_value(out, &Value::Temporal(data.get(idx))),
            // Reconstruct the list and reuse the `push_value` path, so a vector column
            // encodes byte-for-byte identically to the same list boxed in `Mixed`.
            Column::Vec { data, dim, .. } => push_value(
                out,
                &Value::List(
                    data[idx * *dim..idx * *dim + *dim]
                        .iter()
                        .map(|x| Value::Num(*x))
                        .collect(),
                ),
            ),
            Column::Mixed { data } => push_value(out, data[idx].as_ref().unwrap()),
            // Synthesize the record (or read an escapee) and reuse `push_value`, so a
            // de-boxed record encodes byte-for-byte identically to a boxed map.
            Column::Record { .. } => push_value(out, &store.value_id(idx, kid as u32, strs)),
        }
    }
    out.push('}');
}

/// Encode a columnar graph back to NDJSON (nodes then edges). Builds the string
/// directly — no per-record `serde_json::Value` allocation.
pub fn encode(g: &Graph) -> String {
    let mut out = String::with_capacity(g.vertex_count() * 64 + g.edge_count() * 48);
    for vi in 0..g.n {
        if !g.is_vertex_live(vi as u32) {
            continue; // skip tombstoned vertices
        }
        out.push_str("{\"type\":\"node\",\"id\":");
        push_json_str(&mut out, g.vid.text(vi as u32));
        out.push_str(",\"labels\":[");
        for (i, &l) in g.vertex_labels(vi as u32).iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            push_json_str(&mut out, g.labels.text(l));
        }
        out.push_str("],\"properties\":");
        push_props(&mut out, &g.props, &g.strs, vi);
        out.push_str("}\n");
    }
    for i in 0..g.edge_slots() {
        if !g.is_edge_live(i as u32) {
            continue; // skip tombstoned edges
        }
        out.push_str("{\"type\":\"edge\"");
        // Every edge has an id (assigned, or canonical `e{index}`) — always emit.
        out.push_str(",\"id\":");
        push_json_str(&mut out, &g.edge_id(i as u32));
        out.push_str(",\"from\":");
        push_json_str(&mut out, g.vid.text(g.e_src[i]));
        out.push_str(",\"to\":");
        push_json_str(&mut out, g.vid.text(g.e_dst[i]));
        out.push_str(",\"labels\":[");

        for (k, lid) in g.edge_labels(i as u32).into_iter().enumerate() {
            if k > 0 {
                out.push(',');
            }

            push_json_str(&mut out, g.etype.text(lid));
        }

        out.push_str("],\"properties\":");
        push_props(&mut out, &g.edge_props, &g.strs, i);
        out.push_str("}\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_props_round_trip() {
        // Tagged temporals decode to `Value::Temporal` and re-serialize to their
        // canonical tagged form (the duration normalizes to total months/days).
        let doc = "{\"type\":\"node\",\"id\":\"1\",\"labels\":[\"Event\"],\"properties\":{\
            \"at\":{\"@datetime\":\"2020-02-29T13:45:06.5\"},\
            \"on\":{\"@date\":\"2020-02-29\"},\
            \"took\":{\"@duration\":\"P1Y2M3DT4H5M6S\"}}}";
        let g = decode(doc).unwrap();
        let enc = encode(&g);
        assert!(enc.contains("{\"@date\":\"2020-02-29\"}"), "{enc}");
        assert!(
            enc.contains("{\"@datetime\":\"2020-02-29T13:45:06.5\"}"),
            "{enc}"
        );
        assert!(enc.contains("{\"@duration\":\"P14M3DT14706S\"}"), "{enc}");
        // Stable: re-decoding the output re-encodes identically.
        assert_eq!(encode(&decode(&enc).unwrap()), enc);
    }

    #[test]
    fn append_matches_decode_of_concatenation() {
        let a = "{\"type\":\"node\",\"id\":\"1\",\"labels\":[\"P\"],\"properties\":{\"name\":\"a\",\"age\":1}}\n\
                 {\"type\":\"node\",\"id\":\"2\",\"labels\":[\"P\"],\"properties\":{\"name\":\"b\"}}\n\
                 {\"type\":\"edge\",\"id\":\"e0\",\"from\":\"1\",\"to\":\"2\",\"labels\":[\"K\"],\"properties\":{\"w\":0.5}}";
        let b = "{\"type\":\"node\",\"id\":\"3\",\"labels\":[\"P\",\"Q\"],\"properties\":{\"name\":\"c\"}}\n\
                 {\"type\":\"edge\",\"id\":\"e1\",\"from\":\"2\",\"to\":\"3\",\"labels\":[\"K\"],\"properties\":{\"w\":1.0}}\n\
                 {\"type\":\"edge\",\"from\":\"3\",\"to\":\"1\",\"labels\":[\"K\"],\"properties\":{}}";

        // Appending b into decode(a) equals decoding the concatenation.
        let mut merged = decode(a).unwrap();
        let rep = append(&mut merged, b).unwrap();
        let combined = decode(&format!("{a}\n{b}")).unwrap();
        assert_eq!(encode(&merged), encode(&combined));
        // A clean merge: everything applied, nothing skipped, no phantoms.
        assert_eq!(rep.nodes_added, 1);
        assert_eq!(rep.edges_added, 2);
        assert!(rep.nodes_skipped.is_empty());
        assert!(rep.edges_skipped.is_empty());
        assert!(rep.phantom_vertices.is_empty());

        // Appending into an empty graph equals a fresh decode.
        let mut empty = decode("").unwrap();
        append(&mut empty, a).unwrap();
        assert_eq!(encode(&empty), encode(&decode(a).unwrap()));

        // A pre-existing id is first-wins (skipped) and REPORTED; an undeclared
        // edge endpoint is created as a phantom and reported too.
        let mut g = decode(a).unwrap();
        let before = g.vertex_count();
        let rep = append(
            &mut g,
            "{\"type\":\"node\",\"id\":\"1\",\"labels\":[\"Z\"],\"properties\":{\"name\":\"OVERWRITE\"}}\n\
             {\"type\":\"edge\",\"id\":\"e0\",\"from\":\"1\",\"to\":\"2\",\"labels\":[\"K\"],\"properties\":{}}\n\
             {\"type\":\"edge\",\"from\":\"1\",\"to\":\"ghost\",\"labels\":[\"K\"],\"properties\":{}}",
        )
        .unwrap();
        assert_eq!(g.vertex_count(), before + 1); // only the phantom `ghost`
        assert_eq!(rep.nodes_skipped, vec!["1".to_string()]);
        assert_eq!(rep.edges_skipped, vec!["e0".to_string()]);
        assert_eq!(rep.phantom_vertices, vec!["ghost".to_string()]);
        assert_eq!(rep.nodes_added, 0);
        assert_eq!(rep.edges_added, 1);

        // Indexes survive an append AND are maintained: the new node is findable.
        let mut idx = decode(a).unwrap();
        idx.create_vertex_index("name");
        append(&mut idx, b).unwrap();
        assert!(idx.vertex_indexed("name"));
        assert_eq!(
            idx.vertices_by_prop("name", &crate::graph::IdxKey::Str("c".into()))
                .map(<[u32]>::len),
            Some(1)
        );
    }

    #[test]
    fn nested_object_property_is_a_map() {
        // A nested object is now a first-class map/record property (canonicalized
        // to sorted keys on store). Both parse paths agree.
        let line = r#"{"type":"node","id":"a","labels":[],"properties":{"m":{"b":2,"a":1}}}"#;
        for g in [decode(line).unwrap(), decode_serial(line).unwrap()] {
            assert_eq!(
                g.props.value(0, "m", &g.strs),
                Value::Map(vec![
                    ("a".into(), Value::Num(1.0)),
                    ("b".into(), Value::Num(2.0)),
                ]),
            );
        }
    }

    #[test]
    fn round_trip_preserves_content() {
        let input = "\
{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"ann\",\"age\":30,\"active\":true}}
{\"type\":\"node\",\"id\":\"b\",\"labels\":[\"Person\"],\"properties\":{\"name\":\"bo\",\"age\":25,\"active\":false}}
{\"type\":\"edge\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"KNOWS\"],\"properties\":{}}";
        let g = decode(input).unwrap();
        assert_eq!(g.n, 2);
        assert_eq!(g.edge_count(), 1);
        // re-decoding the encoding yields the same shape
        let g2 = decode(&encode(&g)).unwrap();
        assert_eq!(g2.n, 2);
        assert_eq!(g2.edge_count(), 1);
        // age column present and correct
        let age_kid = g2.props.keys.get("age").unwrap();
        match &g2.props.cols[age_kid as usize] {
            Column::Num { data, .. } => {
                let a = g2.vid.get("a").unwrap() as usize;
                assert_eq!(data[a], 30.0);
            }
            _ => panic!("age should be a Num column"),
        }
    }

    #[test]
    fn strict_decode_rejects_malformed_lines() {
        use crate::error_codes::ErrorCode;
        // `.err().unwrap()` rather than `unwrap_err()` (Graph has no Debug impl).
        let code = |s: &str| decode(s).err().unwrap().code;
        // Invalid JSON → InvalidJson (matches TS, instead of a silent skip).
        assert_eq!(code("{not json"), ErrorCode::InvalidJson);
        // Valid JSON but not an object → InvalidShape.
        assert_eq!(code("42"), ErrorCode::InvalidShape);
        assert_eq!(code("[1,2]"), ErrorCode::InvalidShape);
        // Object with unknown/missing `type` → InvalidShape.
        assert_eq!(code(r#"{"type":"banana"}"#), ErrorCode::InvalidShape);
        assert_eq!(code(r#"{"id":"a"}"#), ErrorCode::InvalidShape);
        // Blank lines are still skipped (not an error).
        let g = decode("\n  \n{\"type\":\"node\",\"id\":\"a\",\"labels\":[],\"properties\":{}}\n")
            .unwrap();
        assert_eq!(g.n, 1);
    }

    #[test]
    fn deeply_nested_array_is_rejected_not_overflow() {
        use crate::error_codes::ErrorCode;
        // serde caps nesting at 128 levels during parse → a clean InvalidJson,
        // never a stack overflow or a silent accept (matches the TS depth guard).
        let deep = format!("{}1{}", "[".repeat(2000), "]".repeat(2000));
        let line = format!(r#"{{"type":"node","id":"a","labels":[],"properties":{{"x":{deep}}}}}"#);
        assert_eq!(decode(&line).err().unwrap().code, ErrorCode::InvalidJson);
    }

    #[test]
    fn duplicate_ids_first_wins_node_drop_second_edge() {
        // Matches the TS core: a duplicate node id is first-wins (later labels/
        // props ignored), and an edge with an already-seen id is dropped.
        let g = decode(
            "{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"L1\"],\"properties\":{\"x\":1}}\n\
             {\"type\":\"node\",\"id\":\"a\",\"labels\":[\"L2\"],\"properties\":{\"x\":2}}\n\
             {\"type\":\"node\",\"id\":\"b\",\"labels\":[],\"properties\":{}}\n\
             {\"type\":\"edge\",\"id\":\"x\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"properties\":{}}\n\
             {\"type\":\"edge\",\"id\":\"x\",\"from\":\"b\",\"to\":\"a\",\"labels\":[\"S\"],\"properties\":{}}",
        )
        .unwrap();
        assert_eq!(g.n, 2); // a (first-wins) + b
        let a = g.vid.get("a").unwrap();
        let labels: Vec<&str> = g
            .vertex_labels(a)
            .iter()
            .map(|&l| g.labels.text(l))
            .collect();
        assert_eq!(labels, vec!["L1"]); // first-wins: L2 ignored
        assert_eq!(g.props.value(a as usize, "x", &g.strs), Value::Num(1.0)); // first-wins
        assert_eq!(g.edge_count(), 1); // drop-second: only the first edge id 'x'
        assert_eq!(g.etype.text(g.e_type[0]), "R");
    }

    // ===== decode characterization =====
    //
    // Assert the exact `Value` that `decode` parses each JSON property into, so
    // the hand-rolled parser (which will replace `serde_json::from_str`) can be
    // proven equivalent. Covers escape decoding (incl. `\uXXXX` + surrogate
    // pairs), number forms, list/bool/null, and whitespace. Malformed-input
    // rejection is locked by `strict_decode_rejects_malformed_lines`,
    // `deeply_nested_array_is_rejected_not_overflow`, and the lone-surrogate
    // test below.

    fn decoded(props: &str, key: &str) -> Value {
        let line = format!(r#"{{"type":"node","id":"a","labels":["N"],"properties":{props}}}"#);
        let g = decode(&line).unwrap();
        let a = g.vid.get("a").unwrap() as usize;
        g.props.value(a, key, &g.strs)
    }
    fn str_val(s: &str) -> Value {
        Value::Str(s.into())
    }

    #[test]
    fn one_shot_decode_rejects_a_dangling_edge() {
        // A one-shot `decode` no longer fabricates a phantom for an edge whose
        // endpoint was never declared — it rejects it (like pg-json/graphson/csv).
        let dangling =
            "{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"N\"],\"properties\":{}}\n\
             {\"type\":\"edge\",\"from\":\"a\",\"to\":\"ghost\",\"labels\":[\"K\"],\"properties\":{}}";
        assert_eq!(
            decode(dangling).err().map(|e| e.code),
            Some(ErrorCode::MissingVertex)
        );
        // Endpoints declared in ANY order within the batch are fine (edge first).
        let out_of_order =
            "{\"type\":\"edge\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"K\"],\"properties\":{}}\n\
             {\"type\":\"node\",\"id\":\"a\",\"labels\":[\"N\"],\"properties\":{}}\n\
             {\"type\":\"node\",\"id\":\"b\",\"labels\":[\"N\"],\"properties\":{}}";
        assert!(decode(out_of_order).is_ok());
    }

    #[test]
    fn ingestion_rejects_malformed_labels_and_keys() {
        // A `::` label is unrepresentable in GraphSON; an empty label/key is
        // unrepresentable too. Ingestion rejects them with a coded error.
        let cases = [
            r#"{"type":"node","id":"a","labels":["x::y"],"properties":{}}"#,
            r#"{"type":"node","id":"a","labels":[""],"properties":{}}"#,
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"":1}}"#,
            // declare the endpoint so this isolates the malformed edge LABEL (a
            // dangling edge is now a separate `MissingVertex` rejection).
            "{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"N\"],\"properties\":{}}\n{\"type\":\"edge\",\"id\":\"e\",\"from\":\"a\",\"to\":\"a\",\"labels\":[\"A::B\"],\"properties\":{}}",
        ];
        for c in cases {
            assert_eq!(
                decode(c).err().map(|e| e.code),
                Some(ErrorCode::InvalidValue),
                "should reject: {c}"
            );
        }
        // A single colon and a well-formed graph are fine.
        assert!(
            decode(r#"{"type":"node","id":"a","labels":["a:b"],"properties":{"k":1}}"#).is_ok()
        );
    }

    #[test]
    fn decode_string_escapes() {
        assert_eq!(decoded(r#"{"s":"a\"b"}"#, "s"), str_val("a\"b"));
        assert_eq!(decoded(r#"{"s":"a\\b"}"#, "s"), str_val("a\\b"));
        assert_eq!(decoded(r#"{"s":"a\/b"}"#, "s"), str_val("a/b")); // \/ → /
        assert_eq!(decoded(r#"{"s":"a\tb\nc\rd"}"#, "s"), str_val("a\tb\nc\rd"));
        assert_eq!(
            decoded(r#"{"s":"a\bb\fc"}"#, "s"),
            str_val("a\u{08}b\u{0c}c")
        );
        // \uXXXX (BMP) and surrogate pairs (astral) decode to real chars.
        assert_eq!(decoded(r#"{"s":"\u0041\u00e9"}"#, "s"), str_val("A\u{e9}"));
        assert_eq!(
            decoded(r#"{"s":"\ud83e\udd80"}"#, "s"),
            str_val("\u{1F980}")
        );
    }

    #[test]
    fn decode_number_forms() {
        assert_eq!(decoded(r#"{"n":42}"#, "n"), Value::Num(42.0));
        assert_eq!(decoded(r#"{"n":-7}"#, "n"), Value::Num(-7.0));
        assert_eq!(decoded(r#"{"n":1.5}"#, "n"), Value::Num(1.5));
        assert_eq!(decoded(r#"{"n":1.5e3}"#, "n"), Value::Num(1500.0)); // exponent
        assert_eq!(decoded(r#"{"n":2.5e-3}"#, "n"), Value::Num(0.0025));
    }

    #[test]
    fn decode_nonfinite_number_maps_to_null() {
        // A JSON literal that overflows f64 to ±Infinity is not representable in
        // the LPG numeric model → stored as `null` (matching TS ndjson/pg-json,
        // whose `normalizeValue` maps NaN/±Inf → null). Storing a real non-finite
        // float would corrupt count/sum/min/max/`IS NULL` and diverge from TS.
        assert_eq!(decoded(r#"{"n":1e400}"#, "n"), Value::Null); // +Infinity
        assert_eq!(decoded(r#"{"n":-1e400}"#, "n"), Value::Null); // -Infinity
                                                                  // The stored null is a PRESENT value (first-class), not absence.
        let line = r#"{"type":"node","id":"a","labels":["N"],"properties":{"v":1e400}}"#;
        let g = decode(line).unwrap();
        let a = g.vid.get("a").unwrap() as usize;
        assert_eq!(g.props.value(a, "v", &g.strs), Value::Null);
        assert!(g.props.is_present(a, "v"), "the coerced null is present");
        // Same coercion via the COPY-FROM `append` path.
        let mut m = decode("").unwrap();
        append(&mut m, line).unwrap();
        let ai = m.vid.get("a").unwrap() as usize;
        assert_eq!(m.props.value(ai, "v", &m.strs), Value::Null);
        // Inside a list, a non-finite element is coerced too.
        assert_eq!(
            decoded(r#"{"xs":[1,1e400,2]}"#, "xs"),
            Value::List(vec![Value::Num(1.0), Value::Null, Value::Num(2.0)])
        );
    }

    #[test]
    fn decode_bool_null_and_lists() {
        assert_eq!(decoded(r#"{"b":true}"#, "b"), Value::Bool(true));
        assert_eq!(decoded(r#"{"c":false}"#, "c"), Value::Bool(false));
        // A top-level null property is a PRESENT, first-class value (distinct
        // from absent) — it reads back as Null AND is present; a key never set
        // is not present. (Divergence from the old "null = absent" model.)
        {
            let line = r#"{"type":"node","id":"a","labels":["N"],"properties":{"d":null}}"#;
            let g = decode(line).unwrap();
            let a = g.vid.get("a").unwrap() as usize;
            assert_eq!(g.props.value(a, "d", &g.strs), Value::Null);
            assert!(g.props.is_present(a, "d"), "a stored null is present");
            assert!(
                !g.props.is_present(a, "never_set"),
                "an unset key is absent"
            );
        }
        // …but null INSIDE a list value is preserved.
        assert_eq!(
            decoded(r#"{"xs":[1,2,3]}"#, "xs"),
            Value::List(vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)])
        );
        assert_eq!(
            decoded(r#"{"xs":["a",2,true,null]}"#, "xs"),
            Value::List(vec![
                str_val("a"),
                Value::Num(2.0),
                Value::Bool(true),
                Value::Null
            ])
        );
        assert_eq!(
            decoded(r#"{"xs":[[1],[2,3]]}"#, "xs"),
            Value::List(vec![
                Value::List(vec![Value::Num(1.0)]),
                Value::List(vec![Value::Num(2.0), Value::Num(3.0)]),
            ])
        );
    }

    #[test]
    fn decode_tolerates_whitespace() {
        assert_eq!(
            decoded("{ \"n\" : 1 , \"s\" : \"x\" }", "n"),
            Value::Num(1.0)
        );
        assert_eq!(decoded("{ \"n\" : 1 , \"s\" : \"x\" }", "s"), str_val("x"));
    }

    #[test]
    fn decode_rejects_lone_surrogate() {
        use crate::error_codes::ErrorCode;
        // A lone high surrogate is not valid JSON — must be rejected, not decoded
        // to a replacement char (serde behavior; the hand-rolled parser must match).
        let line = r#"{"type":"node","id":"a","labels":["N"],"properties":{"s":"\ud83e"}}"#;
        assert_eq!(decode(line).err().unwrap().code, ErrorCode::InvalidJson);
    }

    /// How fast COULD ingest go, and how close is it?
    ///
    /// A throughput number means nothing without a reference, so this measures
    /// one document at five levels:
    ///
    /// 1. scan — count newlines. LLVM vectorizes it, so this is roughly what the
    ///    machine pulls through memory: the hard ceiling.
    /// 2. copy — allocate and memcpy the input. The floor for anything that must
    ///    produce as much output as it consumed.
    /// 3. parse — run the JSON parser over every line and drop the result.
    ///    Structure recognized, nothing built.
    /// 4. decode — the whole thing: dictionaries, columns, adjacency.
    /// 5. encode — the way back out.
    ///
    /// The gap between 3 and 4 is the graph BUILD, which no JSON parser does —
    /// which is why comparing a graph loader's GiB/s against a JSON parser's is
    /// not like for like.
    ///
    /// **It sweeps sizes, and that matters.** The answer changes qualitatively
    /// with `n`, because the working set falls out of cache somewhere between
    /// 200k and 1M elements. At 10k-200k the document and its dictionaries stay
    /// resident; past 1M they do not, the per-element build cost roughly doubles,
    /// and ingest becomes bound by random access into the dictionaries rather
    /// than by hashing or allocation. Line 2 shows the transition directly:
    /// allocate-and-memcpy runs at 27 GiB/s over 22 MiB and 2.6 GiB/s over 112
    /// MiB. An optimization tuned at one end can measure as nothing at the other
    /// — which is exactly what happened to a faster hash function — so a change
    /// is only believable if it is measured across the whole range.
    ///
    /// A worked example of why: batching the id interning and prefetching each
    /// slot a few iterations ahead — the textbook fix for a miss-bound loop —
    /// measured 110 -> 130 ms at 200k (17% WORSE) and nothing at all at 1M. At
    /// 200k the dictionary is cache-resident, so there are no misses to hide and
    /// the extra load is overhead that also evicts something useful; at 1M the
    /// two-pass structure and the hash array it materializes cost about what the
    /// overlap saves. Measured at 1M alone it would have looked harmless, and at
    /// 200k alone like a serious regression. Neither is the whole answer.
    ///
    /// Reordering the adjacency FILL was tried against rows 6 and 7 and rejected.
    /// Decomposing the 1M penalty gives a ceiling: src sorted + dst sorted 1255
    /// ms, dst scattered +271, src scattered +163, both +335. A stable counting
    /// sort by endpoint — filling each direction in ascending vertex order
    /// instead of edge order — recovered 27 ms of that 335 (1230 -> 1203,
    /// ~2%) and nothing on already-clustered edges. The sort's own permutation
    /// pass scatters writes across a cursor array and reads back through
    /// `resolved`, which costs about what the ordered writes save. Not worth ~50
    /// lines of stable-counting-sort in the middle of graph construction, where
    /// getting it subtly wrong reorders traversal silently.
    ///
    /// The structural fix — CSR adjacency — is off the table for a different and
    /// better reason: it was tried historically and collapsed once the graph
    /// started changing. `interleaved_write_and_traverse_is_independent_of_graph_
    /// size` in the GQL tests exists to keep it that way.
    ///
    /// Row 8 is where the edge path becomes measurable at all. Merging the edge
    /// dedup table into the id overlay measured FLAT when this benchmark ran one
    /// edge per node and left the id off — 28 ms of effect against 35 ms of
    /// run-to-run spread. At five edges per node, with ids present, the same
    /// change is a clean 5% (894 -> 843 ms, four samples each, no overlap).
    ///
    /// The lesson is about the benchmark, not the change: every per-edge cost
    /// scales with the edge:node ratio while every per-node cost does not, so a
    /// sparse fixture systematically understates anything on the edge path. A
    /// 1:1 graph is not a small version of a real one.
    ///
    /// `INGEST_N=3000000,5000000` overrides the default sweep.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn ingest_throughput_against_the_ceiling() {
        let sizes: Vec<usize> = std::env::var("INGEST_N").map_or_else(
            |_| vec![10_000, 200_000, 1_000_000],
            |v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        );

        for n in sizes {
            ingest_throughput_at(n);
        }
    }

    #[cfg(test)]
    fn ingest_throughput_at(n: usize) {
        use std::time::Instant;

        fn bench(label: &str, bytes: usize, reps: usize, mut f: impl FnMut()) -> f64 {
            f();

            let mut best = f64::MAX;

            for _ in 0..reps {
                let t = Instant::now();

                f();

                let s = t.elapsed().as_secs_f64();

                if s < best {
                    best = s;
                }
            }

            let gibs = bytes as f64 / best / (1024.0 * 1024.0 * 1024.0);

            println!("  {label:<30} {:>8.1} ms  {gibs:>7.3} GiB/s", best * 1000.0);

            gibs
        }

        let text: String = (0..n)
            .map(|i| {
                format!(
                    concat!(
                        r#"{{"type":"node","id":"v{i}","labels":["Person"],"#,
                        r#""properties":{{"name":"person{i}","city":"Springfield","age":{a}}}}}"#
                    ),
                    i = i,
                    a = i % 90
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let bytes = text.len();
        // Fewer reps as the document grows; each is already a best-of.
        let reps = if n >= 1_000_000 { 3 } else { 5 };

        println!(
            "\n=== {n} nodes, {:.1} MiB ===",
            bytes as f64 / (1024.0 * 1024.0)
        );

        let ceiling = bench("1. scan (count newlines)", bytes, 20, || {
            std::hint::black_box(text.bytes().filter(|b| *b == b'\n').count());
        });

        bench("2. copy (allocate + memcpy)", bytes, 20, || {
            std::hint::black_box(text.to_string());
        });

        let parse = bench("3. parse only (drop result)", bytes, reps, || {
            for line in text.lines() {
                std::hint::black_box(crate::json::parse(line).is_ok());
            }
        });
        let decode_gibs = bench("4. decode (build the graph)", bytes, reps, || {
            std::hint::black_box(super::decode(&text).is_ok());
        });
        let g = super::decode(&text).expect("decodes");
        let obytes = encode(&g).len();

        bench("5. encode", obytes, reps, || {
            std::hint::black_box(encode(&g).len());
        });

        println!(
            "  parse reaches {:.0}% of the scan ceiling; decode reaches {:.0}% of parse.",
            parse / ceiling * 100.0,
            decode_gibs / parse * 100.0
        );

        // Edge ENDPOINT LOCALITY, which a nodes-only document hides entirely.
        // Filling adjacency writes to `out[src]`/`in_[dst]`, so scattered
        // endpoints scatter those writes across a per-vertex `Vec<Vec<Adj>>` —
        // one miss for the header, another for its buffer.
        //
        // Every edge here carries an id, because `encode` emits one and so every
        // reloaded snapshot has them. Omitting the id skips the external-id
        // bookkeeping entirely, which is a shape nothing actually loads — an
        // earlier version of this benchmark did exactly that and led to a wrong
        // conclusion about how much that bookkeeping costs.
        let with_edges = |per_node: usize, scatter: bool| -> String {
            let mut lines: Vec<String> = (0..n)
                .map(|i| {
                    format!(
                        r#"{{"type":"node","id":"v{i}","labels":["P"],"properties":{{"n":{i}}}}}"#
                    )
                })
                .collect();

            for k in 0..n.saturating_sub(1) * per_node {
                let (a, b) = if scatter {
                    ((k * 7919) % n, (k * 104_729) % n)
                } else {
                    (k % n, (k + 1) % n)
                };

                lines.push(format!(
                    r#"{{"type":"edge","id":"e{k}","labels":["R"],"from":"v{a}","to":"v{b}","properties":{{}}}}"#
                ));
            }

            lines.join("\n")
        };
        let seq = with_edges(1, false);
        let scattered = with_edges(1, true);

        bench("6. decode, adjacent endpoints", seq.len(), reps, || {
            std::hint::black_box(super::decode(&seq).is_ok());
        });
        bench(
            "7. decode, scattered endpoints",
            scattered.len(),
            reps,
            || {
                std::hint::black_box(super::decode(&scattered).is_ok());
            },
        );

        // EDGE-DENSE. One edge per node is a sparse graph; real ones carry many
        // more edges than nodes, and every per-edge cost scales with that ratio
        // while every per-node cost does not. A conclusion drawn at 1:1 about
        // anything on the edge path is drawn at the wrong ratio.
        let dense = with_edges(5, true);

        bench("8. decode, 5 edges per node", dense.len(), reps, || {
            std::hint::black_box(super::decode(&dense).is_ok());
        });
    }
    /// Where does a simple query's time go, per row?
    ///
    /// Every query pays projection and row materialization, so a per-row cost
    /// there is paid by everything. Compares shapes that do progressively more
    /// per row against one that produces almost none, over the same graph.
    ///
    /// Measured at 500k-1M rows: a numeric column is ~7-23 ns, a string column
    /// roughly double that (an `Arc` clone and a dictionary index), and `RETURN n`
    /// — the shape most application code actually issues — is 230-310 ns, an
    /// order of magnitude more. That decomposes into ~208 ns fixed per row plus
    /// ~23 ns per PRESENT property key.
    ///
    /// The sparse row is the reassuring one: 24 keys in the store with one on
    /// each element costs less than a one-key store, so an absent key is a
    /// bitset test rather than a walk. The fixed 208 ns is roughly four `Vec`
    /// allocations per row (labels twice over, the property map, the outer
    /// three-entry map) plus a handful of atomic `Arc` clones.
    ///
    /// **This harness cannot resolve a change worth ~10% of that.** Back-to-back
    /// runs of the `whole node` row have landed anywhere from 72 to 161 ms for
    /// the same binary, because it runs after several other shapes and inherits
    /// their allocator and cache state. Removing one of the four allocations was
    /// tried and could not be told apart from noise. Anything at that scale needs
    /// its own isolated microbenchmark, not another row here.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn query_row_cost() {
        use std::time::Instant;

        let n: usize = std::env::var("ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000_000);
        let text: String = (0..n)
            .map(|i| {
                format!(
                    r#"{{"type":"node","id":"v{i}","labels":["P"],"properties":{{"a":{i},"s":"str{i}","b":true}}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut g = super::decode(&text).expect("decodes");

        let mut run = |label: &str, q: &str| {
            let plan = crate::gql::parse(q).expect("parses");
            let mut best = f64::MAX;

            for _ in 0..5 {
                let t = Instant::now();
                let rs = plan
                    .execute(&mut g, &crate::gql::eval::Params::new())
                    .expect("runs");
                let rows = rs.rows().count();
                let s = t.elapsed().as_secs_f64();

                std::hint::black_box(rows);
                if s < best {
                    best = s;
                }
            }

            println!(
                "  {label:<30} {:>8.1} ms   {:>6.0} ns/row",
                best * 1000.0,
                best * 1e9 / n as f64
            );
        };

        println!("\n=== {n} rows ===");
        run("count(*) only", "MATCH (n:P) RETURN count(*) AS c");
        run("1 numeric column", "MATCH (n:P) RETURN n.a AS a");
        run("1 string column", "MATCH (n:P) RETURN n.s AS s");
        run(
            "3 columns",
            "MATCH (n:P) RETURN n.a AS a, n.s AS s, n.b AS b",
        );
        run(
            "1 col + filter (all pass)",
            "MATCH (n:P) WHERE n.a >= 0 RETURN n.a AS a",
        );
        run("1 col + arithmetic", "MATCH (n:P) RETURN n.a + 1 AS a");
        run("whole node", "MATCH (n:P) RETURN n AS n");

        // How `RETURN n` scales with the number of property KEYS in the store —
        // separating per-element work from per-key work. `props_map` walks every
        // key in the store per row (not just the ones present) and sorts what it
        // finds, though the relative order of two keys never changes.
        for keys in [1usize, 8, 24] {
            let doc: String = (0..n)
                .map(|i| {
                    let props: Vec<String> = (0..keys).map(|k| format!(r#""k{k}":{i}"#)).collect();

                    format!(
                        r#"{{"type":"node","id":"w{i}","labels":["Q"],"properties":{{{}}}}}"#,
                        props.join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let mut gk = super::decode(&doc).expect("decodes");
            let plan = crate::gql::parse("MATCH (n:Q) RETURN n AS n").expect("parses");
            let mut best = f64::MAX;

            for _ in 0..3 {
                let t = Instant::now();
                let rs = plan
                    .execute(&mut gk, &crate::gql::eval::Params::new())
                    .expect("runs");

                std::hint::black_box(rs.rows().count());

                let el = t.elapsed().as_secs_f64();

                if el < best {
                    best = el;
                }
            }

            println!(
                "  {:<30} {:>8.1} ms   {:>6.0} ns/row",
                format!("whole node, {keys} keys"),
                best * 1000.0,
                best * 1e9 / n as f64
            );
        }

        // SPARSE properties: 24 distinct keys in the store, but each element
        // carries only one of them. If the cost tracks the store's key count
        // rather than the element's, the per-row walk is over all keys.
        {
            let doc: String = (0..n)
                .map(|i| {
                    format!(
                        r#"{{"type":"node","id":"s{i}","labels":["S"],"properties":{{"k{}":{i}}}}}"#,
                        i % 24
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let mut gs = super::decode(&doc).expect("decodes");
            let plan = crate::gql::parse("MATCH (n:S) RETURN n AS n").expect("parses");
            let mut best = f64::MAX;

            for _ in 0..3 {
                let t = Instant::now();
                let rs = plan
                    .execute(&mut gs, &crate::gql::eval::Params::new())
                    .expect("runs");

                std::hint::black_box(rs.rows().count());

                let el = t.elapsed().as_secs_f64();

                if el < best {
                    best = el;
                }
            }

            println!(
                "  {:<30} {:>8.1} ms   {:>6.0} ns/row",
                "whole node, 24 keys / 1 each",
                best * 1000.0,
                best * 1e9 / n as f64
            );
        }
    }
}
