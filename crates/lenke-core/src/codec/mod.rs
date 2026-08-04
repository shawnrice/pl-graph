//! Serialization codecs mirroring the TypeScript `@lenke/serialization`
//! package: **pg-json**, **pg-text**, **graphson**, and **csv**. (NDJSON has its
//! own module, [`crate::ndjson`].) Each codec exposes `encode(&Graph) -> String`
//! and `decode(&str) -> Result<Graph, String>`, and [`serialize`] /
//! [`deserialize`] dispatch by format name (including `"ndjson"`).
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
//!
//! Streaming variants (the TS `encodeStream`/`decodeStream`) are intentionally
//! omitted: the idiomatic bulk path here is the whole-string `encode`/`decode`
//! over the `Builder`, which is the codec-contract surface.

pub mod csv;
pub mod graphson;
pub mod pg_json;
pub mod pg_text;

#[cfg(test)]
mod conformance;

use std::borrow::Cow;
use std::sync::Arc;

use crate::error::{CodeError, CodeResult};
use crate::error_codes::ErrorCode;
use crate::graph::{Dict, Graph, Properties, Value};
use crate::json::Json;

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

/// True if a float is an exact integer value — GraphSON `g:Int64` vs `g:Double`,
/// CSV `integer` vs `float`. Mirrors JS `Number.isInteger`.
pub(crate) fn is_intish(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0
}

// ---------------------------------------------------------------------------
// JSON scalar emit (shared by pg-json and graphson; mirrors ndjson)
// ---------------------------------------------------------------------------

// JSON scalar emit is shared via [`crate::jsonfmt`], so every serde-free writer
// (gremlin, ndjson, codecs) escapes strings and formats numbers identically.
pub(crate) use crate::jsonfmt::{push_json_str, push_num};

/// Emit a core [`Value`] as a plain JSON value (used by pg-json).
pub(crate) fn push_value(out: &mut String, v: &Value) {
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
        // A record/map → a JSON object; keys are already canonical (sorted).
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

// ---------------------------------------------------------------------------
// JSON scalar parse (shared by pg-json; graphson has its own typed decode)
// ---------------------------------------------------------------------------

/// A `serde_json::Value` as a core [`Value`]. A nested JSON object is a
/// map/record property (a single-key `{"@date":…}` stays a tagged temporal).
pub(crate) fn json_to_value(j: &Json) -> CodeResult<Value> {
    Ok(match j {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        // A non-finite JSON number (±Infinity from an overflowing literal like
        // `1e400`, or NaN) is not representable in the LPG numeric model → `null`,
        // matching the TS `normalizeValue` contract. Storing a real non-finite
        // float would corrupt aggregates and `IS NULL` and diverge from TS.
        Json::Num(n) => {
            if n.is_finite() {
                Value::Num(*n)
            } else {
                Value::Null
            }
        }
        Json::Str(s) => Value::Str(Arc::from(s.as_ref())),
        Json::Arr(a) => Value::List(
            a.iter()
                .map(json_to_value)
                .collect::<CodeResult<Vec<_>>>()?,
        ),
        // A tagged temporal `{"@date":"…"}` (single key) round-trips as a scalar;
        // any other JSON object is a record/map value (canonicalized on store).
        Json::Obj(pairs) => match crate::json::temporal_from_pairs(pairs) {
            Some(res) => {
                Value::Temporal(res.map_err(|e| CodeError::new(ErrorCode::InvalidValue, e))?)
            }
            None => Value::Map(
                pairs
                    .iter()
                    .map(|(k, v)| Ok((Arc::from(k.as_ref()), json_to_value(v)?)))
                    .collect::<CodeResult<Vec<_>>>()?,
            ),
        },
    })
}

/// A JSON id field as a string (a string verbatim; a number/bool/null via its
/// JSON text — matching serde_json's `Display`).
pub(crate) fn json_id<'a>(j: &Json<'a>) -> Cow<'a, str> {
    match j {
        Json::Str(s) => s.clone(),
        Json::Num(n) => Cow::Owned(crate::jsonfmt::js_number(*n)),
        Json::Bool(b) => Cow::Owned(b.to_string()),
        _ => Cow::Borrowed("null"),
    }
}

/// A JSON array field as a `Vec<String>` (non-string elements dropped).
pub(crate) fn json_str_array<'a>(field: Option<&Json<'a>>) -> Vec<Cow<'a, str>> {
    field
        .and_then(Json::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| match x {
                    // Cloning the `Cow` keeps the INPUT's lifetime; `as_str()`
                    // would hand back a reference into the tree instead.
                    Json::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// A JSON object field as core property pairs (used by pg-json). A nested-object
/// value anywhere is an `InvalidValue` error (see [`json_to_value`]).
pub(crate) fn json_props<'a>(field: Option<&Json<'a>>) -> CodeResult<Vec<(Cow<'a, str>, Value)>> {
    match field.and_then(Json::as_object) {
        Some(m) => m
            .iter()
            .map(|(k, v)| Ok((k.clone(), json_to_value(v)?)))
            .collect(),
        None => Ok(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// Format dispatch (mirrors the TS `serialize` / `deserialize`)
// ---------------------------------------------------------------------------

/// An unrecognized format name. The codes are now structural: an unknown name is
/// distinct from a parse failure of a *known* format (which the decoders code
/// precisely), so the FFI layer can surface `e.code` directly.
fn unknown_format(format: &str) -> CodeError {
    CodeError::new(
        ErrorCode::UnknownFormat,
        format!("unknown serialization format '{format}'"),
    )
}

/// Serialize `g` in the named format: `pg-json | pg-text | graphson | csv | ndjson`.
pub fn serialize(g: &Graph, format: &str) -> CodeResult<String> {
    // The flat codecs can't faithfully carry a nested record; reject loudly rather
    // than mangle or drop it. The structured codecs (ndjson/graphson/pg-json)
    // round-trip maps, so point the caller there.
    if matches!(format, "pg-text" | "csv") && g.has_map_property() {
        return Err(CodeError::new(
            ErrorCode::Unsupported,
            "a map/record property can't be serialized to a flat format (pg-text/csv); \
             use a structured format: ndjson, graphson, or pg-json",
        ));
    }
    match format {
        "pg-json" => Ok(pg_json::encode(g)),
        "pg-text" => Ok(pg_text::encode(g)),
        "graphson" => Ok(graphson::encode(g)),
        "csv" => Ok(csv::encode(g)),
        "ndjson" => Ok(crate::ndjson::encode(g)),
        other => Err(unknown_format(other)),
    }
}

/// Deserialize `input` in the named format into a fresh graph. A bad format name
/// yields `UnknownFormat`; a malformed payload of a known format yields the
/// decoder's own code (`InvalidJson` / `InvalidShape` / …).
pub fn deserialize(input: &str, format: &str) -> CodeResult<Graph> {
    let g = match format {
        "pg-json" => pg_json::decode(input),
        "pg-text" => Ok(pg_text::decode(input)),
        "graphson" => graphson::decode(input),
        "csv" => csv::decode(input),
        "ndjson" => crate::ndjson::decode(input),
        other => Err(unknown_format(other)),
    }?;
    // Ingestion gate: reject loaded data holding a malformed label / edge type /
    // property key so it can't smuggle in a name that won't round-trip.
    g.validate_wellformed()?;
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph with one explicitly-id'd edge, via pg-json.
    fn with_edge_id() -> Graph {
        pg_json::decode(
            r#"{"nodes":[{"id":"a","labels":[],"properties":{}},{"id":"b","labels":[],"properties":{}}],"edges":[{"id":"pay-1","from":"a","to":"b","labels":["PAID"],"properties":{"amt":50}}]}"#,
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
