//! Serialization codecs mirroring the TypeScript `@lenke/serialization`
//! package: **pg-json**, **pg-text**, **graphson**, and **csv**. (NDJSON has its
//! own module, [`crate::ndjson`].) [`serialize`] / [`deserialize`] dispatch by
//! format name (including `"ndjson"`).
//!
//! The pg-json / pg-text / graphson / csv format logic now lives in the shared,
//! zero-dep [`lenke_codec`] crate, over a neutral graph model — so this core and
//! the standalone `lenke-engine` share one byte-identical implementation. This
//! module keeps only the **bridge**: [`to_graph_data`] (Graph → neutral) and
//! [`from_graph_data`] (neutral → Graph), plus the format dispatch that wires
//! them to `lenke_codec` (NDJSON stays native — it has its own streaming module).
//!
//! ## Two faithful divergences from the TS core
//!
//! Both are pre-existing properties of this columnar core (see [`crate::ndjson`]),
//! not codec choices:
//!   - **An edge carries a single type** (`etype`), not a label *set*. Where a
//!     format models edge labels as a list (PG-JSON `labels`, CSV `:TYPE`,
//!     GraphSON `label`), we emit the one type and, on decode, take the first
//!     label as the type.
//!   - **Every edge has an id.** It is the assigned external id, or — computed on
//!     demand — the canonical `e{index}` derived from the dense index (see
//!     [`Graph::edge_id`](crate::graph::Graph::edge_id)); the explicit-id overlay
//!     stays lazy, so the load path is unaffected. Formats with an edge-id slot
//!     (PG-JSON, GraphSON, CSV, NDJSON) **always emit** it and round-trip it.
//!     PG-text has no id slot, so its edges re-derive `e{index}` on decode rather
//!     than round-tripping an assigned id. **Node** ids round-trip exactly.

#[cfg(test)]
mod conformance;

use std::sync::Arc;

use lenke_codec::{Edge as CEdge, GraphData, Node as CNode, Value as CValue};

use crate::error::{CodeError, CodeResult};
use crate::error_codes::ErrorCode;
use crate::graph::{Builder, Dict, EdgeRec, Graph, NodeRec, Properties, Value};

// ---------------------------------------------------------------------------
// Element/property access over the columnar store
// ---------------------------------------------------------------------------

/// Present properties of element `idx`, in key-id (intern) order. Gated on
/// PRESENCE (`is_present_id`), not on the value: a stored `Null` is a present
/// property (a first-class value — see `set_value`) and IS emitted as `null`;
/// only a genuinely absent key is skipped. Each codec's scalar writer renders
/// `Value::Null` in its own null form.
pub(crate) fn element_props<'a>(
    store: &'a Properties,
    strs: &Dict,
    idx: usize,
) -> Vec<(&'a str, Value)> {
    let mut out = Vec::new();
    for kid in 0..store.cols.len() as u32 {
        if store.is_present_id(idx, kid) {
            out.push((store.keys.text(kid), store.value_id(idx, kid, strs)));
        }
    }
    out
}

/// A node's labels as string slices, in stored order.
pub(crate) fn node_labels(g: &Graph, vi: u32) -> Vec<&str> {
    g.vertex_labels(vi)
        .iter()
        .map(|&l| g.labels.text(l))
        .collect()
}

/// Every type an edge carries, in storage order — the edge-side `node_labels`.
///
/// Edges are multi-type, and each format below already has a label-SET slot it
/// uses for vertices (a JSON array, a `::`-joined string, repeated `:label`
/// tokens, a `;`-joined cell). Writing `e_type` alone into that slot dropped
/// every type but the first on the way out, in all four.
pub(crate) fn edge_types(g: &Graph, ei: u32) -> Vec<&str> {
    g.edge_labels(ei)
        .into_iter()
        .map(|t| g.etype.text(t))
        .collect()
}

// ---------------------------------------------------------------------------
// Neutral-model bridge (Graph <-> lenke_codec::GraphData)
// ---------------------------------------------------------------------------

/// A core [`Value`] as a neutral [`CValue`]. A temporal becomes its `(tag, iso)`
/// strings; every other variant maps one-to-one (a `Map` keeps its stored,
/// canonical key order, so the codec re-emits it exactly as the store holds it).
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
        Value::Map(pairs) => CValue::Map(
            pairs
                .iter()
                .map(|(k, val)| (k.to_string(), value_to_neutral(val)))
                .collect(),
        ),
    }
}

/// How a decoder that produced a temporal-tagged value wants a *malformed* ISO
/// string handled — the one place the codecs differ once the format logic is
/// shared. On a valid ISO every policy yields the same `Value::Temporal`.
#[derive(Clone, Copy)]
enum TemporalOnErr {
    /// Reject (`E_INVALID_VALUE`) — the JSON document codecs (pg-json, graphson).
    Error,
    /// Fall back to the bare ISO string — CSV's lenient scalar decode.
    StringIso,
    /// Fall back to the `@tag:iso` token — pg-text's lenient scalar decode.
    TagToken,
}

/// A neutral [`CValue`] as a core [`Value`]. Only a temporal can fail (an ISO
/// string the neutral crate carried through without validating); `on_err` picks
/// the codec-appropriate handling.
fn value_from_neutral(v: &CValue, on_err: TemporalOnErr) -> CodeResult<Value> {
    Ok(match v {
        CValue::Null => Value::Null,
        CValue::Bool(b) => Value::Bool(*b),
        CValue::Num(x) => Value::Num(*x),
        CValue::Str(s) => Value::Str(Arc::from(s.as_str())),
        CValue::Temporal { tag, iso } => match crate::temporal::Temporal::parse(tag, iso) {
            Ok(t) => Value::Temporal(t),
            Err(e) => match on_err {
                TemporalOnErr::Error => return Err(CodeError::new(ErrorCode::InvalidValue, e)),
                TemporalOnErr::StringIso => Value::Str(Arc::from(iso.as_str())),
                TemporalOnErr::TagToken => Value::Str(Arc::from(format!("@{tag}:{iso}").as_str())),
            },
        },
        CValue::List(a) => Value::List(
            a.iter()
                .map(|e| value_from_neutral(e, on_err))
                .collect::<CodeResult<_>>()?,
        ),
        CValue::Map(pairs) => Value::Map(
            pairs
                .iter()
                .map(|(k, val)| Ok((Arc::from(k.as_str()), value_from_neutral(val, on_err)?)))
                .collect::<CodeResult<_>>()?,
        ),
    })
}

/// Project a graph into the neutral model the shared codecs operate on. Elements
/// are yielded in live-index order and each property bag in key-intern order
/// (via [`element_props`]) — the exact order the codecs used to read, so the
/// serialized bytes are unchanged.
fn to_graph_data(g: &Graph) -> GraphData {
    let neutral_props = |props: Vec<(&str, Value)>| -> Vec<(String, CValue)> {
        props
            .into_iter()
            .map(|(k, v)| (k.to_string(), value_to_neutral(&v)))
            .collect()
    };

    let mut nodes = Vec::with_capacity(g.vertex_count());
    for vi in 0..g.n as u32 {
        if !g.is_vertex_live(vi) {
            continue;
        }
        nodes.push(CNode {
            id: g.vid.text(vi).to_string(),
            labels: node_labels(g, vi)
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            props: neutral_props(element_props(&g.props, &g.strs, vi as usize)),
        });
    }

    let mut edges = Vec::with_capacity(g.edge_count());
    for i in 0..g.edge_slots() {
        if !g.is_edge_live(i as u32) {
            continue;
        }
        edges.push(CEdge {
            // Every edge has an id (assigned or canonical `e{index}`); the
            // id-less pg-text codec simply ignores it on encode.
            id: Some(g.edge_id(i as u32).into_owned()),
            from: g.vid.text(g.e_src[i]).to_string(),
            to: g.vid.text(g.e_dst[i]).to_string(),
            labels: edge_types(g, i as u32)
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            props: neutral_props(element_props(&g.edge_props, &g.strs, i)),
        });
    }

    GraphData { nodes, edges }
}

/// Build a fresh graph from neutral data. `strict` enforces the declared-nodes
/// contract (an edge endpoint must be a declared node → `MissingVertex`), used by
/// the document codecs; the lenient path (pg-text) fabricates missing endpoints.
fn from_graph_data(data: GraphData, strict: bool, on_err: TemporalOnErr) -> CodeResult<Graph> {
    let mut b = Builder::default();
    for n in data.nodes {
        let props = n
            .props
            .into_iter()
            .map(|(k, v)| Ok((k, value_from_neutral(&v, on_err)?)))
            .collect::<CodeResult<Vec<_>>>()?;
        b.nodes.push(NodeRec::owned(n.id, n.labels, props));
    }
    for e in data.edges {
        let props = e
            .props
            .into_iter()
            .map(|(k, v)| Ok((k, value_from_neutral(&v, on_err)?)))
            .collect::<CodeResult<Vec<_>>>()?;
        // Edges are MULTI-type: the first label is the type, the rest are extras.
        let mut labels = e.labels.into_iter();
        let etype = labels.next().unwrap_or_default();
        b.edges.push(EdgeRec {
            src: std::borrow::Cow::Owned(e.from),
            dst: std::borrow::Cow::Owned(e.to),
            etype: std::borrow::Cow::Owned(etype),
            extra_labels: labels.map(std::borrow::Cow::Owned).collect(),
            props: crate::graph::owned_props(props),
            id: e.id.map(std::borrow::Cow::Owned),
        });
    }
    if strict {
        b.finalize_strict()
    } else {
        Ok(b.finalize())
    }
}

// ---------------------------------------------------------------------------
// Format dispatch (mirrors the TS `serialize` / `deserialize`)
// ---------------------------------------------------------------------------

/// Map a shared-codec error (carrying an `E_*` wire string) to a core `CodeError`.
fn map_codec_err(e: lenke_codec::CodecError) -> CodeError {
    let code = ErrorCode::ALL
        .iter()
        .copied()
        .find(|c| c.as_str() == e.code)
        .unwrap_or(ErrorCode::Ffi);
    CodeError::new(code, e.message)
}

/// Serialize `g` in the named format: `pg-json | pg-text | graphson | csv | ndjson`.
pub fn serialize(g: &Graph, format: &str) -> CodeResult<String> {
    // NDJSON keeps its own native streaming module.
    if format == "ndjson" {
        return Ok(crate::ndjson::encode(g));
    }
    // The flat/map rejection lives in the shared crate (byte-identical message);
    // building the neutral projection first keeps one code path.
    lenke_codec::serialize(&to_graph_data(g), format).map_err(map_codec_err)
}

/// Deserialize `input` in the named format into a fresh graph. A bad format name
/// yields `UnknownFormat`; a malformed payload of a known format yields the
/// decoder's own code (`InvalidJson` / `InvalidShape` / …).
pub fn deserialize(input: &str, format: &str) -> CodeResult<Graph> {
    let g = match format {
        "ndjson" => crate::ndjson::decode(input)?,
        // The endpoint-strictness and temporal-fallback policies are per-codec:
        // the JSON document codecs are strict + reject a bad temporal; CSV is
        // strict but lenient on a bad temporal scalar; pg-text is lenient on both.
        "pg-json" | "graphson" => build_from(input, format, true, TemporalOnErr::Error)?,
        "csv" => build_from(input, format, true, TemporalOnErr::StringIso)?,
        "pg-text" => build_from(input, format, false, TemporalOnErr::TagToken)?,
        other => {
            return Err(CodeError::new(
                ErrorCode::UnknownFormat,
                format!("unknown serialization format '{other}'"),
            ))
        }
    };
    // Ingestion gate: reject loaded data holding a malformed label / edge type /
    // property key so it can't smuggle in a name that won't round-trip.
    g.validate_wellformed()?;
    Ok(g)
}

/// Deserialize via the shared crate, then bridge back to a core `Graph`.
fn build_from(input: &str, format: &str, strict: bool, on_err: TemporalOnErr) -> CodeResult<Graph> {
    let data = lenke_codec::deserialize(input, format).map_err(map_codec_err)?;
    from_graph_data(data, strict, on_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph with one explicitly-id'd edge, via pg-json.
    fn with_edge_id() -> Graph {
        deserialize(
            r#"{"nodes":[{"id":"a","labels":[],"properties":{}},{"id":"b","labels":[],"properties":{}}],"edges":[{"id":"pay-1","from":"a","to":"b","labels":["PAID"],"properties":{"amt":50}}]}"#,
            "pg-json",
        )
        .unwrap()
    }

    #[test]
    fn edge_id_round_trips_across_id_formats() {
        let g = with_edge_id();
        assert_eq!(g.edge_id(0).as_ref(), "pay-1");
        for format in ["pg-json", "graphson", "csv", "ndjson"] {
            let blob = serialize(&g, format).unwrap();
            let g2 = deserialize(&blob, format).unwrap();
            assert_eq!(g2.edge_id(0).as_ref(), "pay-1", "edge id lost via {format}");
            assert_eq!(
                g2.edge_by_id("pay-1"),
                Some(0),
                "reverse lookup lost via {format}"
            );
        }
    }

    #[test]
    fn dispatch_unknown_format_errs() {
        let g = with_edge_id();
        assert!(serialize(&g, "nope").is_err());
        assert!(deserialize("", "nope").is_err());
    }

    #[test]
    fn set_and_remove_edge_id() {
        let mut g = crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"a\",\"labels\":[],\"properties\":{}}\n{\"type\":\"node\",\"id\":\"b\",\"labels\":[],\"properties\":{}}\n{\"type\":\"edge\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"X\"],\"properties\":{}}",
        )
        .unwrap();
        assert_eq!(g.edge_id(0).as_ref(), "e0"); // canonical `e{index}` by default
        g.set_edge_id(0, "e-custom");
        assert_eq!(g.edge_id(0).as_ref(), "e-custom");
        assert_eq!(g.edge_by_id("e-custom"), Some(0));
        // removing the edge purges the overlay
        g.remove_edge(0);
        assert_eq!(g.edge_by_id("e-custom"), None);
    }

    #[test]
    fn every_edge_has_a_canonical_id() {
        // No edge is id-less: an unassigned edge has the canonical `e{index}`,
        // resolvable in both directions, and that id is emitted by every codec.
        let g = crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"a\",\"labels\":[],\"properties\":{}}\n\
             {\"type\":\"node\",\"id\":\"b\",\"labels\":[],\"properties\":{}}\n\
             {\"type\":\"edge\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"X\"],\"properties\":{}}\n\
             {\"type\":\"edge\",\"from\":\"b\",\"to\":\"a\",\"labels\":[\"Y\"],\"properties\":{}}",
        )
        .unwrap();
        assert_eq!(g.edge_id(0).as_ref(), "e0");
        assert_eq!(g.edge_id(1).as_ref(), "e1");
        assert_eq!(g.edge_by_id("e1"), Some(1));
        assert_eq!(g.edge_by_id("e9"), None); // out of range
                                              // The canonical id is emitted and round-trips through every id format.
        for format in ["pg-json", "graphson", "csv", "ndjson"] {
            let blob = serialize(&g, format).unwrap();
            let g2 = deserialize(&blob, format).unwrap();
            assert_eq!(
                g2.edge_id(0).as_ref(),
                "e0",
                "canonical id lost via {format}"
            );
            assert_eq!(
                g2.edge_by_id("e1"),
                Some(1),
                "reverse lookup lost via {format}"
            );
        }
    }

    // --- map/record properties (Phase 2) -----------------------------------
    // A node with a map property, authored with keys out of order and a nested
    // list-of-maps, so the round-trip proves canonicalization + structure survive.
    const MAP_NDJSON: &str = concat!(
        r#"{"type":"node","id":"a","labels":["Person"],"#,
        r#""properties":{"meta":{"name":"marko","age":29,"tags":[{"w":2,"k":"x"}]}}}"#,
    );

    #[test]
    fn map_property_round_trips_through_structured_codecs() {
        let g = crate::ndjson::decode(MAP_NDJSON).unwrap();
        // Stored + read back canonical (sorted keys, recursively).
        let want = "{\"meta\":{\"age\":29,\"name\":\"marko\",\"tags\":[{\"k\":\"x\",\"w\":2}]}}";
        let ndjson = crate::ndjson::encode(&g);
        assert!(ndjson.contains(want), "ndjson map shape: {ndjson}");

        // Round-trip through each structured codec back to identical ndjson.
        let base = crate::ndjson::encode(&g);
        for format in ["ndjson", "graphson", "pg-json"] {
            let blob = serialize(&g, format).unwrap();
            let g2 = deserialize(&blob, format).unwrap();
            assert_eq!(
                crate::ndjson::encode(&g2),
                base,
                "map lost/altered via {format}"
            );
        }
    }

    #[test]
    fn flat_codecs_reject_a_map_property_loudly() {
        let g = crate::ndjson::decode(MAP_NDJSON).unwrap();
        for format in ["pg-text", "csv"] {
            assert!(
                serialize(&g, format).is_err(),
                "{format} should reject a map property, not mangle it"
            );
        }
        // A graph with no map still serializes to the flat codecs fine.
        let plain = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"n":1}}"#,
        )
        .unwrap();
        assert!(serialize(&plain, "csv").is_ok());
        assert!(serialize(&plain, "pg-text").is_ok());
    }
}

#[cfg(test)]
mod multi_label_edge_sweep {
    //! EVERY codec, round-tripping an edge that carries more than one type.
    //!
    //! Vertices have been multi-label since the start and every format here
    //! already carries a label SET for them — a JSON array, a `::`-joined
    //! string, repeated `:label` tokens, a `;`-joined cell. Edges used the same
    //! slot but wrote only `e_type`, the first, so a two-type edge silently
    //! became single-type on the way out. The TS mirrors emit all of them, which
    //! made this a cross-engine divergence too; the codec fuzzer missed it
    //! because it never built a multi-type edge.
    //!
    //! Written as a sweep because checking one format at a time is what let the
    //! same omission sit in four of them.
    use crate::codec::{deserialize, serialize};

    const FORMATS: &[&str] = &["ndjson", "pg-json", "pg-text", "graphson", "csv"];

    fn fixture() -> crate::graph::Graph {
        crate::ndjson::decode(
            &[
                r#"{"type":"node","id":"a","labels":["V","W"],"properties":{}}"#,
                r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
                r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R","S"],"properties":{"w":1.0}}"#,
                r#"{"type":"edge","id":"e1","from":"b","to":"a","labels":["T"],"properties":{}}"#,
            ]
            .join("\n"),
        )
        .expect("fixture decodes")
    }

    /// An edge's types, sorted, so the assertion does not depend on which one
    /// happens to be stored first.
    fn types(g: &crate::graph::Graph, ei: u32) -> Vec<String> {
        let mut out: Vec<String> = g
            .edge_labels(ei)
            .into_iter()
            .map(|t| g.etype.text(t).to_string())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn every_codec_round_trips_all_of_an_edges_types() {
        let g = fixture();

        for fmt in FORMATS {
            let text =
                serialize(&g, fmt).unwrap_or_else(|e| panic!("`{fmt}` encode failed: {e:?}"));
            let back =
                deserialize(&text, fmt).unwrap_or_else(|e| panic!("`{fmt}` decode failed: {e:?}"));

            assert_eq!(back.edge_count(), 2, "`{fmt}` lost an edge:\n{text}");
            assert_eq!(
                types(&back, 0),
                vec!["R".to_string(), "S".to_string()],
                "`{fmt}` dropped a type from a two-type edge:\n{text}"
            );
            assert_eq!(
                types(&back, 1),
                vec!["T".to_string()],
                "`{fmt}` changed a single-type edge:\n{text}"
            );
        }
    }

    /// Encoding is idempotent through the round trip, so nothing is added either
    /// — a decoder that duplicated a type would grow the output each pass.
    #[test]
    fn a_multi_type_edge_re_encodes_identically() {
        let g = fixture();

        for fmt in FORMATS {
            let once = serialize(&g, fmt).expect("encodes");
            let twice =
                serialize(&deserialize(&once, fmt).expect("decodes"), fmt).expect("encodes");

            assert_eq!(once, twice, "`{fmt}` is not stable across a round trip");
        }
    }
}

#[cfg(test)]
mod separator_in_a_label {
    //! GraphSON joins a label SET into one `::`-separated string, so a label
    //! that CONTAINS `::` would be torn in two on the way back. The ingestion
    //! gate rejects it — and now has to reject it on EDGES too, since an edge's
    //! types go through that same join.
    use crate::codec::deserialize;

    fn err_of(doc: &str) -> String {
        match deserialize(doc, "ndjson") {
            Ok(_) => String::from("accepted"),
            Err(e) => format!("{:?}", e.code),
        }
    }

    #[test]
    fn a_type_containing_the_graphson_separator_is_rejected_on_edges_too() {
        let node = r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#;
        let node_b = r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#;

        // The vertex side, which has always been checked.
        assert_eq!(
            err_of(r#"{"type":"node","id":"a","labels":["has::sep"],"properties":{}}"#),
            "InvalidValue"
        );

        // An edge's FIRST type...
        assert_eq!(
            err_of(&[
                node,
                node_b,
                r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["has::sep"],"properties":{}}"#,
            ]
            .join("\n")),
            "InvalidValue"
        );

        // ...and any of the rest, which only became reachable once edges kept
        // more than one.
        assert_eq!(
            err_of(&[
                node,
                node_b,
                r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R","has::sep"],"properties":{}}"#,
            ]
            .join("\n")),
            "InvalidValue"
        );

        // A plain multi-type edge is still fine.
        assert_eq!(
            err_of(&[
                node,
                node_b,
                r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R","S"],"properties":{}}"#,
            ]
            .join("\n")),
            "accepted"
        );
    }
}
