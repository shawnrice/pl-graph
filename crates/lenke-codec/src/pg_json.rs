//! PG-JSON codec over the neutral model — a faithful port of the now-removed
//! lenke-core's `codec::pg_json`, retyped from `Graph`/`Value` to [`GraphData`]/[`Value`].
//!
//! Wire shape (a single JSON document):
//! ```text
//! { "nodes": [{ "id", "labels": [...], "properties": {...} }],
//!   "edges": [{ "id", "from", "to", "undirected", "labels": [...], "properties": {...} }] }
//! ```

use crate::json::Json;
use crate::jsonfmt::{push_json_str, push_value};
use crate::model::{Edge, GraphData, Node, Value};
use crate::{json_id, json_props, json_str_array, CodeResult, CodecError, E_INVALID_SHAPE};

/// Emit an element's present properties as a JSON object.
fn push_props(out: &mut String, props: &[(String, Value)]) {
    out.push('{');
    for (i, (k, v)) in props.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_json_str(out, k);
        out.push(':');
        push_value(out, v);
    }
    out.push('}');
}

/// Serialize neutral graph data to a PG-JSON string (compact, single pass).
pub fn encode(g: &GraphData) -> String {
    let mut out = String::with_capacity(g.nodes.len() * 64 + g.edges.len() * 64);
    out.push_str("{\"nodes\":[");
    for (i, n) in g.nodes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"id\":");
        push_json_str(&mut out, &n.id);
        out.push_str(",\"labels\":[");
        for (j, l) in n.labels.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            push_json_str(&mut out, l);
        }
        out.push_str("],\"properties\":");
        push_props(&mut out, &n.props);
        out.push('}');
    }
    out.push_str("],\"edges\":[");
    for (i, e) in g.edges.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // Every edge has an id (the host supplies the assigned or canonical one).
        out.push_str("{\"id\":");
        push_json_str(&mut out, e.id.as_deref().unwrap_or(""));
        out.push_str(",\"from\":");
        push_json_str(&mut out, &e.from);
        out.push_str(",\"to\":");
        push_json_str(&mut out, &e.to);
        out.push_str(",\"undirected\":false,\"labels\":[");
        for (k, t) in e.labels.iter().enumerate() {
            if k > 0 {
                out.push(',');
            }
            push_json_str(&mut out, t);
        }
        out.push_str("],\"properties\":");
        push_props(&mut out, &e.props);
        out.push('}');
    }
    out.push_str("]}");
    out
}

/// Streaming decode: walk a PG-JSON string and push each element to `sink` as
/// borrowed views, skipping the owned `GraphData`. Same strict shape and same
/// element/property order as [`decode`], so the built graph is identical.
pub fn decode_into(input: &str, sink: &mut dyn crate::GraphSink) -> CodeResult<()> {
    use crate::decstream::json_to_decval;
    let j = crate::parse_json(input, "pg-json")?;
    if j.as_object().is_none() {
        return Err(CodecError::new(
            E_INVALID_SHAPE,
            "pg-json: expected a top-level object",
        ));
    }
    let shape = |msg: &str| CodecError::new(E_INVALID_SHAPE, format!("pg-json: {msg}"));

    let nodes_json = j
        .get("nodes")
        .and_then(Json::as_array)
        .ok_or_else(|| shape("'nodes' must be an array"))?;
    for o in nodes_json {
        if o.as_object().is_none() {
            return Err(shape("each node must be an object"));
        }
        if !is_id_value(o.get("id")) {
            return Err(shape("node 'id' must be a string or number"));
        }
        if !is_string_array(o.get("labels")) {
            return Err(shape("node 'labels' must be an array of strings"));
        }
        if !is_object_field(o.get("properties")) {
            return Err(shape("node 'properties' must be an object"));
        }
        let id_num;
        let id: &str = match o.get("id") {
            Some(Json::Str(s)) => s.as_ref(),
            other => {
                id_num = other.map(json_id).unwrap_or_default();
                &id_num
            }
        };
        let labels = borrowed_str_array(o.get("labels"));
        let props = borrowed_props(o.get("properties"), json_to_decval);
        sink.node(id, &labels, &props)?;
    }

    match j.get("edges") {
        None => {}
        Some(Json::Arr(edges_json)) => {
            for o in edges_json {
                if o.as_object().is_none() {
                    return Err(shape("each edge must be an object"));
                }
                if !is_string_array(o.get("labels")) {
                    return Err(shape("edge 'labels' must be an array of strings"));
                }
                if !is_object_field(o.get("properties")) {
                    return Err(shape("edge 'properties' must be an object"));
                }
                if !matches!(o.get("id"), None | Some(Json::Str(_)) | Some(Json::Num(_))) {
                    return Err(shape("edge 'id' must be a string or number"));
                }
                let id_num;
                let id: Option<&str> = match o.get("id") {
                    None => None,
                    Some(Json::Str(s)) => Some(s.as_ref()),
                    other => {
                        id_num = other.map(json_id).unwrap_or_default();
                        Some(&id_num)
                    }
                };
                let from_num;
                let from: &str = match o.get("from") {
                    Some(Json::Str(s)) => s.as_ref(),
                    other => {
                        from_num = other.map(json_id).unwrap_or_default();
                        &from_num
                    }
                };
                let to_num;
                let to: &str = match o.get("to") {
                    Some(Json::Str(s)) => s.as_ref(),
                    other => {
                        to_num = other.map(json_id).unwrap_or_default();
                        &to_num
                    }
                };
                let labels = borrowed_str_array(o.get("labels"));
                let props = borrowed_props(o.get("properties"), json_to_decval);
                sink.edge(id, from, to, &labels, &props)?;
            }
        }
        Some(_) => return Err(shape("'edges' must be an array")),
    }
    Ok(())
}

/// A JSON string array as borrowed `&str` (non-string elements dropped) — the
/// borrowed twin of [`json_str_array`](crate::json_str_array).
fn borrowed_str_array<'a>(field: Option<&'a Json<'a>>) -> Vec<&'a str> {
    field
        .and_then(Json::as_array)
        .map(|a| a.iter().filter_map(Json::as_str).collect())
        .unwrap_or_default()
}

/// A JSON object field as borrowed `(key, DecVal)` pairs via `conv`.
fn borrowed_props<'a>(
    field: Option<&'a Json<'a>>,
    conv: impl Fn(&'a Json<'a>) -> crate::decstream::DecVal<'a>,
) -> Vec<(&'a str, crate::decstream::DecVal<'a>)> {
    field
        .and_then(Json::as_object)
        .map(|m| m.iter().map(|(k, v)| (k.as_ref(), conv(v))).collect())
        .unwrap_or_default()
}

/// Deserialize a PG-JSON string into neutral graph data (strict shape).
pub fn decode(input: &str) -> CodeResult<GraphData> {
    let j = crate::parse_json(input, "pg-json")?;
    if j.as_object().is_none() {
        return Err(CodecError::new(
            E_INVALID_SHAPE,
            "pg-json: expected a top-level object",
        ));
    }

    let shape = |msg: &str| CodecError::new(E_INVALID_SHAPE, format!("pg-json: {msg}"));

    let nodes_json = j
        .get("nodes")
        .and_then(Json::as_array)
        .ok_or_else(|| shape("'nodes' must be an array"))?;
    let mut nodes = Vec::with_capacity(nodes_json.len());
    for o in nodes_json {
        if o.as_object().is_none() {
            return Err(shape("each node must be an object"));
        }
        if !is_id_value(o.get("id")) {
            return Err(shape("node 'id' must be a string or number"));
        }
        if !is_string_array(o.get("labels")) {
            return Err(shape("node 'labels' must be an array of strings"));
        }
        if !is_object_field(o.get("properties")) {
            return Err(shape("node 'properties' must be an object"));
        }
        nodes.push(Node {
            id: o.get("id").map(json_id).unwrap_or_default(),
            labels: json_str_array(o.get("labels")),
            props: json_props(o.get("properties"))?,
        });
    }

    let mut edges = Vec::new();
    match j.get("edges") {
        None => {}
        Some(Json::Arr(edges_json)) => {
            for o in edges_json {
                if o.as_object().is_none() {
                    return Err(shape("each edge must be an object"));
                }
                if !is_string_array(o.get("labels")) {
                    return Err(shape("edge 'labels' must be an array of strings"));
                }
                if !is_object_field(o.get("properties")) {
                    return Err(shape("edge 'properties' must be an object"));
                }
                if !matches!(o.get("id"), None | Some(Json::Str(_)) | Some(Json::Num(_))) {
                    return Err(shape("edge 'id' must be a string or number"));
                }
                edges.push(Edge {
                    id: o.get("id").map(json_id),
                    from: o.get("from").map(json_id).unwrap_or_default(),
                    to: o.get("to").map(json_id).unwrap_or_default(),
                    labels: json_str_array(o.get("labels")),
                    props: json_props(o.get("properties"))?,
                });
            }
        }
        Some(_) => return Err(shape("'edges' must be an array")),
    }

    Ok(GraphData { nodes, edges })
}

/// A JSON id field present and string-or-number.
fn is_id_value(j: Option<&Json>) -> bool {
    matches!(j, Some(Json::Str(_)) | Some(Json::Num(_)))
}

/// A present array whose every element is a string.
fn is_string_array(j: Option<&Json>) -> bool {
    matches!(j, Some(Json::Arr(a)) if a.iter().all(|x| matches!(x, Json::Str(_))))
}

/// A present JSON object (non-null, non-array).
fn is_object_field(j: Option<&Json>) -> bool {
    matches!(j, Some(Json::Obj(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_stable() {
        let doc = r#"{"nodes":[{"id":"a","labels":["Person"],"properties":{"name":"ann","age":30,"active":true,"tags":["x","y"]}},{"id":"b","labels":["Person","Admin"],"properties":{"name":"bo"}}],"edges":[{"id":"e0","from":"a","to":"b","undirected":false,"labels":["KNOWS"],"properties":{"since":2020}}]}"#;
        let g = decode(doc).unwrap();
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.nodes[1].labels, vec!["Person", "Admin"]);
        // Re-encoding the decoded doc reproduces it byte-for-byte (the input is
        // already in canonical emit order).
        assert_eq!(encode(&g), doc);
    }

    #[test]
    fn nested_object_is_a_map_and_tagged_temporal_is_a_temporal() {
        let g = decode(
            r#"{"nodes":[{"id":"a","labels":[],"properties":{"m":{"a":1},"d":{"@date":"2024-01-15"}}}],"edges":[]}"#,
        )
        .unwrap();
        assert_eq!(
            g.nodes[0].props,
            vec![
                (
                    "m".to_string(),
                    Value::Map(vec![("a".to_string(), Value::Num(1.0))])
                ),
                (
                    "d".to_string(),
                    Value::Temporal {
                        tag: "date".to_string(),
                        iso: "2024-01-15".to_string()
                    },
                ),
            ],
        );
    }

    #[test]
    fn strict_shape_is_rejected() {
        for doc in [
            r#"{}"#,
            r#"{"nodes":{}}"#,
            r#"{"nodes":[{"labels":[],"properties":{}}]}"#,
            r#"{"nodes":[{"id":true,"labels":[],"properties":{}}]}"#,
            r#"{"nodes":[{"id":"a","labels":["x",1],"properties":{}}]}"#,
            r#"{"nodes":[{"id":"a","labels":[],"properties":null}]}"#,
            r#"{"nodes":[42]}"#,
            r#"{"nodes":[],"edges":{}}"#,
        ] {
            assert_eq!(decode(doc).unwrap_err().code, E_INVALID_SHAPE, "for: {doc}");
        }
        assert!(decode(r#"{"nodes":[{"id":"a","labels":[],"properties":{}}]}"#).is_ok());
    }

    #[test]
    fn bad_json_is_invalid_json() {
        assert_eq!(decode("{not json").unwrap_err().code, crate::E_INVALID_JSON);
    }
}
