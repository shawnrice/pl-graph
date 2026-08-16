//! The bridge between the engine's `Store` and the shared [`lenke_codec`] crate.
//!
//! The pg-json / pg-text / graphson / csv format logic lives ONCE, in
//! `lenke_codec`, over a neutral graph model ([`GraphData`]/[`lenke_codec::Value`]).
//! This module only projects a `Store` into that model ([`to_graph_data`]) and
//! rebuilds a `Store` from it ([`from_graph_data`]) — so the engine and
//! `lenke-core` emit byte-identical bytes from identical data. NDJSON keeps its
//! own native module ([`crate::ndjson`]); binary is the engine's own format.

use std::sync::Arc;

use lenke_codec::{codes, CodecError, Edge as CEdge, GraphData, Node as CNode, Value as CValue};

use crate::store::Store;
use crate::temporal::Temporal;
use crate::value::{make_record, Value};

/// How a decoder that produced a temporal-tagged value wants a *malformed* ISO
/// string handled — the one place the codecs differ once the format logic is
/// shared. On a valid ISO every policy yields the same temporal value.
#[derive(Clone, Copy)]
enum TemporalOnErr {
    /// Reject (`E_INVALID_VALUE`) — the JSON document codecs (pg-json, graphson).
    Error,
    /// Fall back to the bare ISO string — CSV's lenient scalar decode.
    StringIso,
    /// Fall back to the `@tag:iso` token — pg-text's lenient scalar decode.
    TagToken,
}

/// The per-codec policy: endpoint strictness + temporal-parse fallback. Mirrors
/// `lenke-core`'s bridge so both engines interpret the same bytes identically.
fn policy(format: &str) -> (bool, TemporalOnErr) {
    match format {
        "csv" => (true, TemporalOnErr::StringIso),
        "pg-text" => (false, TemporalOnErr::TagToken),
        // pg-json, graphson (and any future document codec)
        _ => (true, TemporalOnErr::Error),
    }
}

// ------------------------------------------------------------- value bridge ---

/// An engine [`Value`] as a neutral [`CValue`]. A stored property is only ever a
/// scalar, a list, or a record; the Gremlin `Map` (a query-result value) never
/// reaches here, so it is handled defensively.
fn value_to_neutral(v: &Value) -> CValue {
    match v {
        Value::Null => CValue::Null,
        Value::Bool(b) => CValue::Bool(*b),
        Value::Num(x) => CValue::Num(*x),
        Value::Str(s) => CValue::Str(s.to_string()),
        Value::Temporal(t) => CValue::Temporal {
            tag: t.tag().to_string(),
            iso: t.format(),
        },
        Value::List(a) => CValue::List(a.iter().map(value_to_neutral).collect()),
        Value::Record(fields) => CValue::Map(
            fields
                .iter()
                .map(|(k, val)| (k.to_string(), value_to_neutral(val)))
                .collect(),
        ),
        // A Gremlin map is not a stored property value; render its entries with
        // best-effort string keys so this is lossless if it ever does appear.
        Value::Map(pairs) => CValue::Map(
            pairs
                .iter()
                .map(|(k, val)| (neutral_key(k), value_to_neutral(val)))
                .collect(),
        ),
    }
}

/// A best-effort string key for a Gremlin-map key (unreachable for stored props;
/// mirrors [`crate::ndjson`]'s map egress, which stringifies a non-string key).
fn neutral_key(v: &Value) -> String {
    match v {
        Value::Str(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

/// A neutral [`CValue`] as an engine [`Value`]. Only a temporal can fail (an ISO
/// string the neutral crate carried through without validating); `on_err` picks
/// the codec-appropriate handling. A neutral `Map` becomes a canonical record.
fn value_from_neutral(v: &CValue, on_err: TemporalOnErr) -> Result<Value, CodecError> {
    Ok(match v {
        CValue::Null => Value::Null,
        CValue::Bool(b) => Value::Bool(*b),
        CValue::Num(x) => Value::Num(*x),
        CValue::Str(s) => Value::Str(Arc::from(s.as_str())),
        CValue::Temporal { tag, iso } => match Temporal::parse(tag, iso) {
            Ok(t) => Value::Temporal(t),
            Err(e) => match on_err {
                TemporalOnErr::Error => return Err(CodecError::new(codes::INVALID_VALUE, e)),
                TemporalOnErr::StringIso => Value::Str(Arc::from(iso.as_str())),
                TemporalOnErr::TagToken => Value::Str(Arc::from(format!("@{tag}:{iso}"))),
            },
        },
        CValue::List(a) => Value::List(
            a.iter()
                .map(|e| value_from_neutral(e, on_err))
                .collect::<Result<_, _>>()?,
        ),
        CValue::Map(pairs) => {
            let fields = pairs
                .iter()
                .map(|(k, val)| Ok((Arc::from(k.as_str()), value_from_neutral(val, on_err)?)))
                .collect::<Result<Vec<_>, CodecError>>()?;
            make_record(fields)
        }
    })
}

// ------------------------------------------------------------- graph bridge ---

/// Project a store into the neutral model the shared codecs operate on. Nodes are
/// yielded in internal-id order and edges in adjacency order (the same order
/// [`crate::ndjson::to_ndjson`] uses), with property bags in `prop_keys` order.
fn to_graph_data(store: &Store) -> GraphData {
    let node_keys = store.prop_keys();
    let edge_keys = store.edge_prop_keys();
    let count = u32::try_from(store.node_count()).unwrap_or(u32::MAX);

    let bag = |present: &dyn Fn(&str) -> Option<Value>, keys: &[String]| -> Vec<(String, CValue)> {
        keys.iter()
            .filter_map(|k| present(k).map(|v| (k.clone(), value_to_neutral(&v))))
            .collect()
    };

    let mut nodes = Vec::new();
    for id in 0..count {
        if !store.is_alive(id) {
            continue;
        }
        nodes.push(CNode {
            id: store
                .node_ext_id(id)
                .map(|s| s.to_string())
                .unwrap_or_default(),
            labels: store.labels_of(id),
            props: bag(
                &|k| store.has_prop(id, k).then(|| store.prop(id, k)),
                &node_keys,
            ),
        });
    }

    let mut edges = Vec::new();
    for from in 0..count {
        if !store.is_alive(from) {
            continue;
        }
        let from_ext = store
            .node_ext_id(from)
            .map(|s| s.to_string())
            .unwrap_or_default();
        for a in store.out(from) {
            let eid = a.eid;
            edges.push(CEdge {
                id: Some(
                    store
                        .edge_ext_id(eid)
                        .map(|s| s.to_string())
                        .unwrap_or_default(),
                ),
                from: from_ext.clone(),
                to: store
                    .node_ext_id(a.nbr)
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                labels: store.edge_labels_of(eid),
                props: bag(
                    &|k| store.has_edge_prop(eid, k).then(|| store.edge_prop(eid, k)),
                    &edge_keys,
                ),
            });
        }
    }

    GraphData { nodes, edges }
}

/// Build a fresh store from neutral data. `strict` enforces the declared-nodes
/// contract (an edge endpoint must be a declared node → `E_MISSING_VERTEX`); the
/// lenient path (pg-text) fabricates a missing endpoint as a bare node.
fn from_graph_data(
    data: GraphData,
    strict: bool,
    on_err: TemporalOnErr,
) -> Result<Store, CodecError> {
    let mut store = Store::default();

    for n in data.nodes {
        let lrefs: Vec<&str> = n.labels.iter().map(String::as_str).collect();
        let props = n
            .props
            .iter()
            .map(|(k, v)| Ok((k.as_str(), value_from_neutral(v, on_err)?)))
            .collect::<Result<Vec<_>, CodecError>>()?;
        store.add_node_with_id(&Arc::from(n.id.as_str()), &lrefs, &props);
    }

    for e in data.edges {
        let from = endpoint(&mut store, &e.from, strict)?;
        let to = endpoint(&mut store, &e.to, strict)?;
        // Edges are MULTI-type: the first label is the type, the rest are extras.
        let mut labels = e.labels.iter();
        let etype = labels.next().map(String::as_str).unwrap_or("");
        let eid = match &e.id {
            Some(id) => store.add_edge_with_id(&Arc::from(id.as_str()), from, to, etype),
            None => store.add_edge(from, to, etype),
        };
        let extra: Vec<&str> = labels.map(String::as_str).collect();
        if !extra.is_empty() {
            store.set_edge_extra_labels(eid, &extra);
        }
        for (k, v) in &e.props {
            store.set_edge_prop(eid, k, value_from_neutral(v, on_err)?);
        }
    }

    store.rebuild_csr();
    store.rebuild_edge_num();
    store.dict_encode_columns();
    Ok(store)
}

/// Resolve an edge endpoint id to a node. Strict: a missing endpoint is an error;
/// lenient: it is created as a bare node (matching pg-text's `finalize`).
fn endpoint(store: &mut Store, ext: &str, strict: bool) -> Result<u32, CodecError> {
    if let Some(id) = store.node_by_ext(ext) {
        return Ok(id);
    }
    if strict {
        return Err(CodecError::new(
            "E_MISSING_VERTEX",
            format!("edge references unknown node id {ext}"),
        ));
    }
    Ok(store.add_node_with_id(&Arc::from(ext), &[], &[]))
}

// --------------------------------------------------------------- dispatch ---

/// Serialize a store in the named format (`pg-json | pg-text | graphson | csv`).
/// NDJSON/binary are handled by their own modules, not here.
pub fn serialize(store: &Store, format: &str) -> Result<String, CodecError> {
    lenke_codec::serialize(&to_graph_data(store), format)
}

/// Deserialize `input` in the named format into a fresh store.
pub fn deserialize(input: &str, format: &str) -> Result<Store, CodecError> {
    let data = lenke_codec::deserialize(input, format)?;
    let (strict, on_err) = policy(format);
    from_graph_data(data, strict, on_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The store's data as canonical NDJSON — an order-independent structural form
    /// (nodes by id, keys sorted) for asserting a round trip preserved everything.
    fn shape(store: &Store) -> String {
        crate::ndjson::to_ndjson(store)
    }

    fn err_code<T>(r: Result<T, CodecError>) -> String {
        r.err().expect("expected an error").code.to_string()
    }

    fn round_trip(format: &str) {
        let src = concat!(
            r#"{"id":"a","labels":["P","Q"],"props":{"n":42,"w":3.5,"ok":true,"tags":["x","y"]}}"#,
            "\n",
            r#"{"id":"b","labels":["P"],"props":{"name":"bo"}}"#,
            "\n",
            r#"{"id":"e0","from":"a","to":"b","type":"KNOWS","props":{"since":2020}}"#,
        );
        let store = crate::ndjson::from_ndjson(src).unwrap();
        let blob = serialize(&store, format).unwrap();
        let back = deserialize(&blob, format).unwrap();
        // pg-text drops the edge id (no id slot), so compare the data shape rather
        // than requiring byte-identical NDJSON; the values must all survive.
        assert_eq!(
            back.node_count(),
            store.node_count(),
            "{format}: node count"
        );
        let a = back.node_by_ext("a").unwrap();
        assert!(
            crate::value::equals(&back.prop(a, "n"), &Value::Num(42.0)),
            "{format}: scalar"
        );
        assert!(
            matches!(back.prop(a, "tags"), Value::List(items) if items.len() == 2),
            "{format}: list"
        );
        assert!(back.node_by_ext("b").is_some(), "{format}: node b");
        assert_eq!(back.edge_prop_keys().len(), 1, "{format}: edge prop kept");
    }

    #[test]
    fn every_codec_round_trips() {
        for f in ["pg-json", "pg-text", "graphson", "csv"] {
            round_trip(f);
        }
    }

    #[test]
    fn structured_codecs_preserve_the_ndjson_shape() {
        // pg-json/graphson/csv carry the edge id, so a full snapshot round-trips to
        // byte-identical NDJSON (the engine's own canonical form).
        let src = concat!(
            r#"{"id":"a","labels":["P"],"props":{"n":1}}"#,
            "\n",
            r#"{"id":"b","labels":[],"props":{}}"#,
            "\n",
            r#"{"id":"e0","from":"a","to":"b","type":"R","props":{}}"#,
        );
        let store = crate::ndjson::from_ndjson(src).unwrap();
        for f in ["pg-json", "graphson", "csv"] {
            let back = deserialize(&serialize(&store, f).unwrap(), f).unwrap();
            assert_eq!(shape(&back), shape(&store), "{f}: shape drifted");
        }
    }

    #[test]
    fn temporal_round_trips_pg_json() {
        let src = r#"{"id":"e","labels":["Event"],"props":{"on":{"@date":"2020-02-29"}}}"#;
        let store = crate::ndjson::from_ndjson(src).unwrap();
        let back = deserialize(&serialize(&store, "pg-json").unwrap(), "pg-json").unwrap();
        let e = back.node_by_ext("e").unwrap();
        assert!(matches!(back.prop(e, "on"), Value::Temporal(_)));
    }

    #[test]
    fn strict_codec_rejects_dangling_edge() {
        // graphson with an edge to an undeclared vertex → E_MISSING_VERTEX.
        let doc = r#"{"vertices":[{"@value":{"id":"a","label":""}}],"edges":[{"@value":{"id":"e","label":"R","inV":"z","outV":"a"}}]}"#;
        assert_eq!(err_code(deserialize(doc, "graphson")), "E_MISSING_VERTEX");
    }

    #[test]
    fn unknown_format_is_reported() {
        assert_eq!(err_code(deserialize("", "nope")), codes::UNKNOWN_FORMAT);
    }

    #[test]
    fn flat_codec_rejects_a_record_property() {
        let src = r#"{"id":"a","labels":["N"],"props":{"m":{"x":1}}}"#;
        let store = crate::ndjson::from_ndjson(src).unwrap();
        assert_eq!(err_code(serialize(&store, "csv")), codes::UNSUPPORTED);
    }
}
