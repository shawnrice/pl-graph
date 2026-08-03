//! GraphSON v3.0 (Apache TinkerPop) codec for the columnar core.
//!
//! The whole graph is one JSON document `{ "vertices": [<g:Vertex>...],
//! "edges": [<g:Edge>...] }`; each element uses GraphSON v3 typed values of the
//! form `{ "@type": <type>, "@value": <value> }`.
//!
//! LPG ↔ GraphSON mapping (see [`crate::codec`] for the shared divergences):
//!   - **Single-value properties.** Each vertex key is emitted as a one-element
//!     `g:VertexProperty` array; decode reads the first element only.
//!   - **Multi-label `::` convention.** A vertex's label *set* is joined with
//!     `::` into GraphSON's single `label` string and split back on decode (empty
//!     set ⇄ `""`). Edges carry a single type, emitted as-is.
//!   - **int/float inference.** `Number.isInteger`-style: a whole float → `g:Int64`,
//!     else `g:Double`. Both decode back to the core's single float type.

use crate::json::{self, Json};

use std::borrow::Cow;

use crate::codec::{element_props, is_intish, node_labels, push_json_str, push_num};
use crate::error::{CodeError, CodeResult};
use crate::error_codes::ErrorCode;
use crate::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

const LABEL_SEP: &str = "::";

/// Emit one core [`Value`] as a GraphSON v3 typed value.
fn push_typed(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Str(s) => push_json_str(out, s),
        Value::Num(x) => {
            if is_intish(*x) {
                out.push_str("{\"@type\":\"g:Int64\",\"@value\":");
                push_num(out, *x);
                out.push('}');
            } else {
                out.push_str("{\"@type\":\"g:Double\",\"@value\":");
                push_num(out, *x);
                out.push('}');
            }
        }
        Value::Temporal(t) => {
            out.push_str("{\"@type\":\"");
            out.push_str(t.graphson_type());
            out.push_str("\",\"@value\":");
            push_json_str(out, &t.format());
            out.push('}');
        }
        Value::List(a) => {
            out.push_str("{\"@type\":\"g:List\",\"@value\":[");
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_typed(out, e);
            }
            out.push_str("]}");
        }
        // GraphSON v3 `g:Map`: a FLAT `[k1, v1, k2, v2, …]` value array (keys are
        // typed too, but a string key is bare, like `push_typed` for `Str`). Keys
        // are already canonical (sorted).
        Value::Map(pairs) => {
            out.push_str("{\"@type\":\"g:Map\",\"@value\":[");
            for (i, (k, e)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_str(out, k);
                out.push(',');
                push_typed(out, e);
            }
            out.push_str("]}");
        }
    }
}

/// Decode a GraphSON v3 typed value (or bare JSON scalar) back to a core value.
fn decode_typed(node: &Json) -> CodeResult<Value> {
    Ok(match node {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Str(s) => Value::Str(s.as_ref().into()),
        Json::Num(n) => Value::Num(*n),
        Json::Arr(a) => Value::List(a.iter().map(decode_typed).collect::<CodeResult<Vec<_>>>()?),
        Json::Obj(_) => {
            let value = node.get("@value");
            match node.get("@type").and_then(Json::as_str) {
                Some("g:Int64" | "g:Int32" | "g:Double" | "g:Float") => {
                    let n = value.and_then(Json::as_f64).ok_or_else(|| {
                        CodeError::new(
                            ErrorCode::InvalidShape,
                            "graphson: numeric typed value must be a number",
                        )
                    })?;
                    Value::Num(n)
                }
                Some("g:List") => {
                    let arr = value.and_then(Json::as_array).ok_or_else(|| {
                        CodeError::new(
                            ErrorCode::InvalidShape,
                            "graphson: g:List value must be an array",
                        )
                    })?;
                    Value::List(
                        arr.iter()
                            .map(decode_typed)
                            .collect::<CodeResult<Vec<_>>>()?,
                    )
                }
                // GraphSON v3 `g:Map`: a flat `[k1, v1, …]` array → a record value.
                // Stored maps are string-keyed, so a non-string key is rejected.
                Some("g:Map") => {
                    let arr = value.and_then(Json::as_array).ok_or_else(|| {
                        CodeError::new(
                            ErrorCode::InvalidShape,
                            "graphson: g:Map value must be an array",
                        )
                    })?;
                    if arr.len() % 2 != 0 {
                        return Err(CodeError::new(
                            ErrorCode::InvalidShape,
                            "graphson: g:Map value must have an even number of entries",
                        ));
                    }
                    let mut pairs = Vec::with_capacity(arr.len() / 2);
                    for ch in arr.chunks_exact(2) {
                        let key = match decode_typed(&ch[0])? {
                            Value::Str(s) => s,
                            _ => {
                                return Err(CodeError::new(
                                    ErrorCode::InvalidShape,
                                    "graphson: a stored g:Map key must be a string",
                                ))
                            }
                        };
                        pairs.push((key, decode_typed(&ch[1])?));
                    }
                    Value::Map(pairs)
                }
                // A temporal wrapper (`gx:LocalDate`/`gx:LocalDateTime`/`gx:Duration`)
                // whose `@value` is the ISO-8601 string.
                Some(ty) if crate::temporal::Temporal::graphson_tag(ty).is_some() => {
                    let tag = crate::temporal::Temporal::graphson_tag(ty).unwrap_or("");
                    let s = value.and_then(Json::as_str).ok_or_else(|| {
                        CodeError::new(
                            ErrorCode::InvalidShape,
                            "graphson: temporal @value must be a string",
                        )
                    })?;
                    Value::Temporal(
                        crate::temporal::Temporal::parse(tag, s)
                            .map_err(|e| CodeError::new(ErrorCode::InvalidValue, e))?,
                    )
                }
                // An unknown/missing wrapper is outside the LPG model — reject it
                // (matches the TS codec) rather than storing a raw out-of-model value.
                _ => {
                    return Err(CodeError::new(
                        ErrorCode::InvalidShape,
                        "graphson: unknown or missing typed-value wrapper",
                    ))
                }
            }
        }
    })
}

pub fn encode(g: &Graph) -> String {
    let mut out = String::with_capacity(g.vertex_count() * 96 + g.edge_count() * 96);
    out.push_str("{\"vertices\":[");
    let mut first = true;
    for vi in 0..g.n {
        if !g.is_vertex_live(vi as u32) {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str("{\"@type\":\"g:Vertex\",\"@value\":{\"id\":");
        push_json_str(&mut out, g.vid.text(vi as u32));
        out.push_str(",\"label\":");
        push_json_str(&mut out, &node_labels(g, vi as u32).join(LABEL_SEP));
        out.push_str(",\"properties\":{");
        for (pi, (k, v)) in element_props(&g.props, &g.strs, vi).iter().enumerate() {
            if pi > 0 {
                out.push(',');
            }
            push_json_str(&mut out, k);
            out.push_str(":[{\"@type\":\"g:VertexProperty\",\"@value\":{\"id\":");
            push_json_str(&mut out, &format!("{}/{k}", g.vid.text(vi as u32)));
            out.push_str(",\"value\":");
            push_typed(&mut out, v);
            out.push_str(",\"label\":");
            push_json_str(&mut out, k);
            out.push_str("}}]");
        }
        out.push_str("}}}");
    }
    out.push_str("],\"edges\":[");
    first = true;
    for i in 0..g.edge_slots() {
        if !g.is_edge_live(i as u32) {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str("{\"@type\":\"g:Edge\",\"@value\":{");
        // Every edge has an id (assigned, or canonical `e{index}`) — always emit.
        out.push_str("\"id\":");
        push_json_str(&mut out, &g.edge_id(i as u32));
        out.push(',');
        out.push_str("\"label\":");
        push_json_str(&mut out, g.etype.text(g.e_type[i]));
        out.push_str(",\"inV\":");
        push_json_str(&mut out, g.vid.text(g.e_dst[i]));
        out.push_str(",\"outV\":");
        push_json_str(&mut out, g.vid.text(g.e_src[i]));
        out.push_str(",\"properties\":{");
        for (pi, (k, v)) in element_props(&g.edge_props, &g.strs, i).iter().enumerate() {
            if pi > 0 {
                out.push(',');
            }
            push_json_str(&mut out, k);
            out.push_str(":{\"@type\":\"g:Property\",\"@value\":{\"key\":");
            push_json_str(&mut out, k);
            out.push_str(",\"value\":");
            push_typed(&mut out, v);
            out.push_str("}}");
        }
        out.push_str("}}}");
    }
    out.push_str("]}");
    out
}

/// The `value` slot inside a `g:VertexProperty` / `g:Property` `@value` object.
fn inner_value<'a, 'b>(prop_value: &'b Json<'a>) -> Option<&'b Json<'a>> {
    prop_value.get("@value").and_then(|v| v.get("value"))
}

pub fn decode(input: &str) -> CodeResult<Graph> {
    let j = json::parse(input)
        .map_err(|()| CodeError::new(ErrorCode::InvalidJson, "graphson: invalid JSON"))?;
    if j.as_object().is_none() {
        return Err(CodeError::new(
            ErrorCode::InvalidShape,
            "graphson: expected a top-level object",
        ));
    }
    let obj = &j;

    let mut b = Builder::default();

    let shape = |msg: &str| CodeError::new(ErrorCode::InvalidShape, format!("graphson: {msg}"));

    if let Some(vertices) = obj.get("vertices").and_then(Json::as_array) {
        for wrapper in vertices {
            let v = wrapper
                .get("@value")
                .filter(|x| x.as_object().is_some())
                .ok_or_else(|| shape("each vertex must have an @value object"))?;
            if !matches!(v.get("id"), Some(Json::Str(_)) | Some(Json::Num(_))) {
                return Err(shape("vertex @value.id must be a string or number"));
            }
            let id = v.get("id").map(crate::codec::json_id).unwrap_or_default();
            let labels: Vec<Cow<'_, str>> = match v.get("label") {
                Some(Json::Str(s)) if s.is_empty() => Vec::new(),
                // A multi-label vertex arrives `::`-joined. Splitting a slice of
                // the INPUT yields slices of the input, so those borrow; splitting
                // a label that had to be unescaped can only borrow from the tree,
                // so those pieces are copied.
                Some(Json::Str(Cow::Borrowed(s))) => {
                    s.split(LABEL_SEP).map(Cow::Borrowed).collect()
                }
                Some(Json::Str(s)) => s
                    .split(LABEL_SEP)
                    .map(|x| Cow::Owned(x.to_string()))
                    .collect(),
                _ => return Err(shape("vertex @value.label must be a string")),
            };
            let mut props = Vec::new();
            if let Some(pmap) = v.get("properties").and_then(Json::as_object) {
                for (k, entries) in pmap {
                    // single-value LPG: read the first element of the array
                    if let Some(first) = entries.as_array().and_then(<[Json]>::first) {
                        if let Some(val) = inner_value(first) {
                            props.push((k.clone(), decode_typed(val)?));
                        }
                    }
                }
            }
            b.nodes.push(NodeRec { id, labels, props });
        }
    }

    if let Some(edges) = obj.get("edges").and_then(Json::as_array) {
        for wrapper in edges {
            let e = wrapper
                .get("@value")
                .filter(|x| x.as_object().is_some())
                .ok_or_else(|| shape("each edge must have an @value object"))?;
            let src = e.get("outV").map(crate::codec::json_id).unwrap_or_default();
            let dst = e.get("inV").map(crate::codec::json_id).unwrap_or_default();
            // single type — split the `::` convention and take the first.
            let Some(Json::Str(label)) = e.get("label") else {
                return Err(shape("edge @value.label must be a string"));
            };
            let etype = label.split(LABEL_SEP).next().unwrap_or("").to_string();
            let mut props = Vec::new();
            if let Some(pmap) = e.get("properties").and_then(Json::as_object) {
                for (k, entry) in pmap {
                    if let Some(val) = inner_value(entry) {
                        props.push((k.clone(), decode_typed(val)?));
                    }
                }
            }
            let id = e.get("id").map(crate::codec::json_id);
            b.edges.push(EdgeRec {
                src,
                dst,
                etype: etype.into(),
                props,
                id,
                extra_labels: Vec::new(),
            });
        }
    }

    b.finalize_strict()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Value;

    fn build() -> Graph {
        // Use pg-json to get a graph with a multi-label node, a list, ints+floats.
        crate::codec::pg_json::decode(
            r#"{"nodes":[{"id":"a","labels":["P","Q"],"properties":{"n":42,"w":3.5,"tags":["x","y"]}},{"id":"b","labels":[],"properties":{}}],"edges":[{"from":"a","to":"b","labels":["KNOWS"],"properties":{"since":2020}}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn round_trip() {
        let g = build();
        let g2 = decode(&encode(&g)).unwrap();
        assert_eq!(g2.vertex_count(), 2);
        assert_eq!(g2.edge_count(), 1);
        let a = g2.vid.get("a").unwrap() as usize;
        assert_eq!(node_labels(&g2, a as u32).len(), 2); // multi-label via `::`
        assert_eq!(g2.props.value(a, "n", &g2.strs), Value::Num(42.0));
        assert_eq!(g2.props.value(a, "w", &g2.strs), Value::Num(3.5));
        assert_eq!(
            g2.props.value(a, "tags", &g2.strs),
            Value::List(vec![Value::Str("x".into()), Value::Str("y".into())]),
        );
        // empty label set round-trips to no labels
        let bn = g2.vid.get("b").unwrap();
        assert_eq!(node_labels(&g2, bn).len(), 0);
    }

    #[test]
    fn int_vs_float_typed() {
        let g = build();
        let s = encode(&g);
        assert!(s.contains("g:Int64"));
        assert!(s.contains("g:Double"));
    }

    #[test]
    fn strict_decode_rejects_malformed() {
        use crate::error_codes::ErrorCode;
        let code = |s: &str| decode(s).err().unwrap().code;
        // Invalid JSON and non-object top level.
        assert_eq!(code("{bad"), ErrorCode::InvalidJson);
        assert_eq!(code("[]"), ErrorCode::InvalidShape);
        // Vertex without @value / id / label.
        assert_eq!(
            code(r#"{"vertices":[{"@type":"g:Vertex"}]}"#),
            ErrorCode::InvalidShape
        );
        assert_eq!(
            code(r#"{"vertices":[{"@value":{"label":""}}]}"#),
            ErrorCode::InvalidShape // missing id
        );
        assert_eq!(
            code(r#"{"vertices":[{"@value":{"id":"a"}}]}"#),
            ErrorCode::InvalidShape // missing label
        );
        // Malformed g:List value and unknown wrapper in a property.
        assert_eq!(
            code(
                r#"{"vertices":[{"@value":{"id":"a","label":"","properties":{"k":[{"@value":{"value":{"@type":"g:List","@value":5}}}]}}}]}"#
            ),
            ErrorCode::InvalidShape
        );
        // A well-formed minimal document still decodes.
        assert!(decode(r#"{"vertices":[{"@value":{"id":"a","label":""}}]}"#).is_ok());
    }
}
