//! GraphSON v3.0 (Apache TinkerPop) codec over the neutral model — a faithful
//! port of the now-removed lenke-core's `codec::graphson`, retyped to [`GraphData`]/[`Value`].
//!
//! The whole graph is one document `{ "vertices": [<g:Vertex>...], "edges":
//! [<g:Edge>...] }`; each element uses GraphSON v3 typed values `{ "@type":…,
//! "@value":… }`. A vertex label SET is `::`-joined into GraphSON's single `label`
//! string (and split back on decode); a whole float → `g:Int64`, else `g:Double`.

use crate::json::Json;
use crate::jsonfmt::{push_json_str, push_num};
use crate::model::{graphson_tag, graphson_type, Edge, GraphData, Node, Value};
use crate::decstream::DecVal;
use crate::stream::ValueRef;
use crate::{is_intish, json_id, CodeResult, CodecError, E_INVALID_SHAPE, E_INVALID_VALUE};

const LABEL_SEP: &str = "::";

/// Accumulates a vertex/edge's labels into a single `::`-joined string (GraphSON
/// carries the label SET as one `label` string). The host pushes borrowed `&str`
/// labels in canonical order; the sink escapes the finished join once.
pub struct LabelJoin<'a> {
    buf: &'a mut String,
    any: bool,
}
impl LabelJoin<'_> {
    pub fn push(&mut self, label: &str) {
        if self.any {
            self.buf.push_str(LABEL_SEP);
        }
        self.any = true;
        self.buf.push_str(label);
    }
}

/// Emit a BORROWED value as a GraphSON v3 typed value — the streaming twin of
/// [`push_typed`]; both funnel scalars through the same writers, so the bytes match.
/// A nested list/record/map arrives as an owned [`Value`] and defers to `push_typed`.
fn push_typed_ref(out: &mut String, v: ValueRef) {
    match v {
        ValueRef::Null => out.push_str("null"),
        ValueRef::Bool(b) => out.push_str(if b { "true" } else { "false" }),
        ValueRef::Str(s) => push_json_str(out, s),
        ValueRef::Num(x) => {
            out.push_str(if is_intish(x) {
                "{\"@type\":\"g:Int64\",\"@value\":"
            } else {
                "{\"@type\":\"g:Double\",\"@value\":"
            });
            push_num(out, x);
            out.push('}');
        }
        ValueRef::Temporal { tag, iso } => {
            out.push_str("{\"@type\":\"");
            out.push_str(graphson_type(tag).unwrap_or("gx:LocalDate"));
            out.push_str("\",\"@value\":");
            push_json_str(out, iso);
            out.push('}');
        }
        ValueRef::Nested(v) => push_typed(out, v),
    }
}

/// A streaming GraphSON v3 encoder — the byte-identical twin of [`encode`] that
/// pulls each element from the host instead of an owned [`GraphData`]. Order of use
/// mirrors [`PgJsonSink`](crate::PgJsonSink): construct, `vertex` per node,
/// `begin_edges`, `edge` per edge, `finish`.
pub struct GraphsonSink {
    out: String,
    label_buf: String,
    any: bool,
}

impl GraphsonSink {
    #[must_use]
    pub fn new(nodes: usize, edges: usize) -> Self {
        let mut out = String::with_capacity(nodes * 96 + edges * 96 + 16);
        out.push_str("{\"vertices\":[");
        Self {
            out,
            label_buf: String::new(),
            any: false,
        }
    }

    fn sep(&mut self) {
        if self.any {
            self.out.push(',');
        }
        self.any = true;
    }

    /// Build the `::`-joined label string into the reusable buffer, then emit it as
    /// one escaped JSON string (join-then-escape, matching [`encode`]).
    fn write_labels(&mut self, labels: impl FnOnce(&mut LabelJoin)) {
        self.label_buf.clear();
        labels(&mut LabelJoin {
            buf: &mut self.label_buf,
            any: false,
        });
        push_json_str(&mut self.out, &self.label_buf);
    }

    /// Emit one vertex. `props` writes each present property with its `g:VertexProperty`
    /// wrapper (whose id is `"<vertex-id>/<key>"`).
    pub fn vertex(
        &mut self,
        id: &str,
        labels: impl FnOnce(&mut LabelJoin),
        props: impl FnOnce(&mut VProps),
    ) {
        self.sep();
        self.out.push_str("{\"@type\":\"g:Vertex\",\"@value\":{\"id\":");
        push_json_str(&mut self.out, id);
        self.out.push_str(",\"label\":");
        self.write_labels(labels);
        self.out.push_str(",\"properties\":{");
        props(&mut VProps {
            out: &mut self.out,
            id,
            any: false,
        });
        self.out.push_str("}}}");
    }

    pub fn begin_edges(&mut self) {
        self.out.push_str("],\"edges\":[");
        self.any = false;
    }

    /// Emit one edge. `props` writes each property with its `g:Property` wrapper.
    pub fn edge(
        &mut self,
        id: &str,
        from: &str,
        to: &str,
        labels: impl FnOnce(&mut LabelJoin),
        props: impl FnOnce(&mut EProps),
    ) {
        self.sep();
        self.out.push_str("{\"@type\":\"g:Edge\",\"@value\":{\"id\":");
        push_json_str(&mut self.out, id);
        self.out.push_str(",\"label\":");
        self.write_labels(labels);
        self.out.push_str(",\"inV\":");
        push_json_str(&mut self.out, to);
        self.out.push_str(",\"outV\":");
        push_json_str(&mut self.out, from);
        self.out.push_str(",\"properties\":{");
        props(&mut EProps {
            out: &mut self.out,
            any: false,
        });
        self.out.push_str("}}}");
    }

    #[must_use]
    pub fn finish(mut self) -> String {
        self.out.push_str("]}");
        self.out
    }
}

/// The vertex-property cursor: each entry is a single-element array of a
/// `g:VertexProperty` whose id is `"<vertex-id>/<key>"`.
pub struct VProps<'a> {
    out: &'a mut String,
    id: &'a str,
    any: bool,
}
impl VProps<'_> {
    pub fn push(&mut self, key: &str, value: ValueRef) {
        if self.any {
            self.out.push(',');
        }
        self.any = true;
        push_json_str(self.out, key);
        self.out
            .push_str(":[{\"@type\":\"g:VertexProperty\",\"@value\":{\"id\":");
        // Composite id "<vertex-id>/<key>" — the one small per-property allocation,
        // inherent to the format (the TS encoder builds the same string).
        let mut vpid = String::with_capacity(self.id.len() + 1 + key.len());
        vpid.push_str(self.id);
        vpid.push('/');
        vpid.push_str(key);
        push_json_str(self.out, &vpid);
        self.out.push_str(",\"value\":");
        push_typed_ref(self.out, value);
        self.out.push_str(",\"label\":");
        push_json_str(self.out, key);
        self.out.push_str("}}]");
    }
}

/// The edge-property cursor: each entry is a `g:Property` wrapper.
pub struct EProps<'a> {
    out: &'a mut String,
    any: bool,
}
impl EProps<'_> {
    pub fn push(&mut self, key: &str, value: ValueRef) {
        if self.any {
            self.out.push(',');
        }
        self.any = true;
        push_json_str(self.out, key);
        self.out
            .push_str(":{\"@type\":\"g:Property\",\"@value\":{\"key\":");
        push_json_str(self.out, key);
        self.out.push_str(",\"value\":");
        push_typed_ref(self.out, value);
        self.out.push_str("}}");
    }
}

/// Emit one neutral [`Value`] as a GraphSON v3 typed value.
fn push_typed(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Str(s) => push_json_str(out, s),
        Value::Num(x) => {
            out.push_str(if is_intish(*x) {
                "{\"@type\":\"g:Int64\",\"@value\":"
            } else {
                "{\"@type\":\"g:Double\",\"@value\":"
            });
            push_num(out, *x);
            out.push('}');
        }
        Value::Temporal { tag, iso } => {
            out.push_str("{\"@type\":\"");
            out.push_str(graphson_type(tag).unwrap_or("gx:LocalDate"));
            out.push_str("\",\"@value\":");
            push_json_str(out, iso);
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

/// Decode a GraphSON v3 typed value (or bare scalar) back to a neutral value.
fn decode_typed(node: &Json) -> CodeResult<Value> {
    let shape = |m: &str| CodecError::new(E_INVALID_SHAPE, format!("graphson: {m}"));
    Ok(match node {
        Json::Null => Value::Null,
        Json::Bool(b) => Value::Bool(*b),
        Json::Str(s) => Value::Str(s.to_string()),
        Json::Num(n) => Value::Num(*n),
        Json::Arr(a) => Value::List(a.iter().map(decode_typed).collect::<CodeResult<_>>()?),
        Json::Obj(_) => {
            let value = node.get("@value");
            match node.get("@type").and_then(Json::as_str) {
                Some("g:Int64" | "g:Int32" | "g:Double" | "g:Float") => Value::Num(
                    value
                        .and_then(Json::as_f64)
                        .ok_or_else(|| shape("numeric typed value must be a number"))?,
                ),
                Some("g:List") => Value::List(
                    value
                        .and_then(Json::as_array)
                        .ok_or_else(|| shape("g:List value must be an array"))?
                        .iter()
                        .map(decode_typed)
                        .collect::<CodeResult<_>>()?,
                ),
                Some("g:Map") => {
                    let arr = value
                        .and_then(Json::as_array)
                        .ok_or_else(|| shape("g:Map value must be an array"))?;
                    if arr.len() % 2 != 0 {
                        return Err(shape("g:Map value must have an even number of entries"));
                    }
                    let mut pairs = Vec::with_capacity(arr.len() / 2);
                    for ch in arr.chunks_exact(2) {
                        let key = match decode_typed(&ch[0])? {
                            Value::Str(s) => s,
                            _ => return Err(shape("a stored g:Map key must be a string")),
                        };
                        pairs.push((key, decode_typed(&ch[1])?));
                    }
                    Value::Map(pairs)
                }
                Some(ty) if graphson_tag(ty).is_some() => {
                    let tag = graphson_tag(ty).unwrap_or("");
                    let iso = value
                        .and_then(Json::as_str)
                        .ok_or_else(|| shape("temporal @value must be a string"))?;
                    if iso.is_empty() {
                        return Err(CodecError::new(
                            E_INVALID_VALUE,
                            "graphson: empty temporal value",
                        ));
                    }
                    Value::Temporal {
                        tag: tag.to_string(),
                        iso: iso.to_string(),
                    }
                }
                _ => return Err(shape("unknown or missing typed-value wrapper")),
            }
        }
    })
}

/// Decode a GraphSON v3 typed value to a BORROWED [`DecVal`] — the streaming twin
/// of [`decode_typed`], mirroring it arm for arm so the built value is identical.
fn decode_typed_ref<'a>(node: &'a Json<'a>) -> CodeResult<DecVal<'a>> {
    let shape = |m: &str| CodecError::new(E_INVALID_SHAPE, format!("graphson: {m}"));
    Ok(match node {
        Json::Null => DecVal::Null,
        Json::Bool(b) => DecVal::Bool(*b),
        Json::Str(s) => DecVal::Str(s.as_ref()),
        Json::Num(n) => DecVal::Num(*n),
        Json::Arr(a) => DecVal::List(a.iter().map(decode_typed_ref).collect::<CodeResult<_>>()?),
        Json::Obj(_) => {
            let value = node.get("@value");
            match node.get("@type").and_then(Json::as_str) {
                Some("g:Int64" | "g:Int32" | "g:Double" | "g:Float") => DecVal::Num(
                    value
                        .and_then(Json::as_f64)
                        .ok_or_else(|| shape("numeric typed value must be a number"))?,
                ),
                Some("g:List") => DecVal::List(
                    value
                        .and_then(Json::as_array)
                        .ok_or_else(|| shape("g:List value must be an array"))?
                        .iter()
                        .map(decode_typed_ref)
                        .collect::<CodeResult<_>>()?,
                ),
                Some("g:Map") => {
                    let arr = value
                        .and_then(Json::as_array)
                        .ok_or_else(|| shape("g:Map value must be an array"))?;
                    if arr.len() % 2 != 0 {
                        return Err(shape("g:Map value must have an even number of entries"));
                    }
                    let mut pairs = Vec::with_capacity(arr.len() / 2);
                    for ch in arr.chunks_exact(2) {
                        let key = match decode_typed_ref(&ch[0])? {
                            DecVal::Str(s) => s,
                            _ => return Err(shape("a stored g:Map key must be a string")),
                        };
                        pairs.push((key, decode_typed_ref(&ch[1])?));
                    }
                    DecVal::Map(pairs)
                }
                Some(ty) if graphson_tag(ty).is_some() => {
                    let tag = graphson_tag(ty).unwrap_or("");
                    let iso = value
                        .and_then(Json::as_str)
                        .ok_or_else(|| shape("temporal @value must be a string"))?;
                    if iso.is_empty() {
                        return Err(CodecError::new(
                            E_INVALID_VALUE,
                            "graphson: empty temporal value",
                        ));
                    }
                    DecVal::Temporal { tag, iso }
                }
                _ => return Err(shape("unknown or missing typed-value wrapper")),
            }
        }
    })
}

/// A `::`-joined label string as borrowed labels (empty string → no labels) — the
/// borrowed twin of [`split_labels`].
fn split_labels_ref(s: &str) -> Vec<&str> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(LABEL_SEP).collect()
    }
}

/// Streaming decode: walk a GraphSON v3 string and push each element to `sink` as
/// borrowed views. Same strict shape and element/property order as [`decode`].
pub fn decode_into(input: &str, sink: &mut dyn crate::GraphSink) -> CodeResult<()> {
    let j = crate::parse_json(input, "graphson")?;
    if j.as_object().is_none() {
        return Err(CodecError::new(
            E_INVALID_SHAPE,
            "graphson: expected a top-level object",
        ));
    }
    let shape = |m: &str| CodecError::new(E_INVALID_SHAPE, format!("graphson: {m}"));

    if let Some(vertices) = j.get("vertices").and_then(Json::as_array) {
        for wrapper in vertices {
            let v = wrapper
                .get("@value")
                .filter(|x| x.as_object().is_some())
                .ok_or_else(|| shape("each vertex must have an @value object"))?;
            if !matches!(v.get("id"), Some(Json::Str(_)) | Some(Json::Num(_))) {
                return Err(shape("vertex @value.id must be a string or number"));
            }
            let labels = match v.get("label") {
                Some(Json::Str(s)) => split_labels_ref(s),
                _ => return Err(shape("vertex @value.label must be a string")),
            };
            let mut props: Vec<(&str, DecVal)> = Vec::new();
            if let Some(pmap) = v.get("properties").and_then(Json::as_object) {
                for (k, entries) in pmap {
                    if let Some(first) = entries.as_array().and_then(<[Json]>::first) {
                        if let Some(val) = inner_value(first) {
                            props.push((k.as_ref(), decode_typed_ref(val)?));
                        }
                    }
                }
            }
            let id_num;
            let id: &str = match v.get("id") {
                Some(Json::Str(s)) => s.as_ref(),
                other => {
                    id_num = other.map(json_id).unwrap_or_default();
                    &id_num
                }
            };
            sink.node(id, &labels, &props)?;
        }
    }

    if let Some(edges_json) = j.get("edges").and_then(Json::as_array) {
        for wrapper in edges_json {
            let e = wrapper
                .get("@value")
                .filter(|x| x.as_object().is_some())
                .ok_or_else(|| shape("each edge must have an @value object"))?;
            let Some(Json::Str(label)) = e.get("label") else {
                return Err(shape("edge @value.label must be a string"));
            };
            let mut props: Vec<(&str, DecVal)> = Vec::new();
            if let Some(pmap) = e.get("properties").and_then(Json::as_object) {
                for (k, entry) in pmap {
                    if let Some(val) = inner_value(entry) {
                        props.push((k.as_ref(), decode_typed_ref(val)?));
                    }
                }
            }
            let labels = split_labels_ref(label);
            let id_num;
            let id: Option<&str> = match e.get("id") {
                None => None,
                Some(Json::Str(s)) => Some(s.as_ref()),
                other => {
                    id_num = other.map(json_id).unwrap_or_default();
                    Some(&id_num)
                }
            };
            let from_num;
            let from: &str = match e.get("outV") {
                Some(Json::Str(s)) => s.as_ref(),
                other => {
                    from_num = other.map(json_id).unwrap_or_default();
                    &from_num
                }
            };
            let to_num;
            let to: &str = match e.get("inV") {
                Some(Json::Str(s)) => s.as_ref(),
                other => {
                    to_num = other.map(json_id).unwrap_or_default();
                    &to_num
                }
            };
            sink.edge(id, from, to, &labels, &props)?;
        }
    }
    Ok(())
}

/// Serialize neutral graph data to a GraphSON v3 string.
pub fn encode(g: &GraphData) -> String {
    let mut out = String::with_capacity(g.nodes.len() * 96 + g.edges.len() * 96);
    out.push_str("{\"vertices\":[");
    for (i, n) in g.nodes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"@type\":\"g:Vertex\",\"@value\":{\"id\":");
        push_json_str(&mut out, &n.id);
        out.push_str(",\"label\":");
        push_json_str(&mut out, &n.labels.join(LABEL_SEP));
        out.push_str(",\"properties\":{");
        for (pi, (k, v)) in n.props.iter().enumerate() {
            if pi > 0 {
                out.push(',');
            }
            push_json_str(&mut out, k);
            out.push_str(":[{\"@type\":\"g:VertexProperty\",\"@value\":{\"id\":");
            push_json_str(&mut out, &format!("{}/{k}", n.id));
            out.push_str(",\"value\":");
            push_typed(&mut out, v);
            out.push_str(",\"label\":");
            push_json_str(&mut out, k);
            out.push_str("}}]");
        }
        out.push_str("}}}");
    }
    out.push_str("],\"edges\":[");
    for (i, e) in g.edges.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"@type\":\"g:Edge\",\"@value\":{\"id\":");
        push_json_str(&mut out, e.id.as_deref().unwrap_or(""));
        out.push_str(",\"label\":");
        push_json_str(&mut out, &e.labels.join(LABEL_SEP));
        out.push_str(",\"inV\":");
        push_json_str(&mut out, &e.to);
        out.push_str(",\"outV\":");
        push_json_str(&mut out, &e.from);
        out.push_str(",\"properties\":{");
        for (pi, (k, v)) in e.props.iter().enumerate() {
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

/// Split a `::`-joined label string into a label vec (empty string → no labels).
fn split_labels(s: &str) -> Vec<String> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(LABEL_SEP).map(str::to_string).collect()
    }
}

/// Deserialize a GraphSON v3 string into neutral graph data (strict shape).
pub fn decode(input: &str) -> CodeResult<GraphData> {
    let j = crate::parse_json(input, "graphson")?;
    if j.as_object().is_none() {
        return Err(CodecError::new(
            E_INVALID_SHAPE,
            "graphson: expected a top-level object",
        ));
    }
    let shape = |m: &str| CodecError::new(E_INVALID_SHAPE, format!("graphson: {m}"));

    let mut nodes = Vec::new();
    if let Some(vertices) = j.get("vertices").and_then(Json::as_array) {
        for wrapper in vertices {
            let v = wrapper
                .get("@value")
                .filter(|x| x.as_object().is_some())
                .ok_or_else(|| shape("each vertex must have an @value object"))?;
            if !matches!(v.get("id"), Some(Json::Str(_)) | Some(Json::Num(_))) {
                return Err(shape("vertex @value.id must be a string or number"));
            }
            let labels = match v.get("label") {
                Some(Json::Str(s)) => split_labels(s),
                _ => return Err(shape("vertex @value.label must be a string")),
            };
            let mut props = Vec::new();
            if let Some(pmap) = v.get("properties").and_then(Json::as_object) {
                for (k, entries) in pmap {
                    if let Some(first) = entries.as_array().and_then(<[Json]>::first) {
                        if let Some(val) = inner_value(first) {
                            props.push((k.to_string(), decode_typed(val)?));
                        }
                    }
                }
            }
            nodes.push(Node {
                id: v.get("id").map(json_id).unwrap_or_default(),
                labels,
                props,
            });
        }
    }

    let mut edges = Vec::new();
    if let Some(edges_json) = j.get("edges").and_then(Json::as_array) {
        for wrapper in edges_json {
            let e = wrapper
                .get("@value")
                .filter(|x| x.as_object().is_some())
                .ok_or_else(|| shape("each edge must have an @value object"))?;
            let Some(Json::Str(label)) = e.get("label") else {
                return Err(shape("edge @value.label must be a string"));
            };
            let mut props = Vec::new();
            if let Some(pmap) = e.get("properties").and_then(Json::as_object) {
                for (k, entry) in pmap {
                    if let Some(val) = inner_value(entry) {
                        props.push((k.to_string(), decode_typed(val)?));
                    }
                }
            }
            edges.push(Edge {
                id: e.get("id").map(json_id),
                from: e.get("outV").map(json_id).unwrap_or_default(),
                to: e.get("inV").map(json_id).unwrap_or_default(),
                labels: split_labels(label),
                props,
            });
        }
    }

    Ok(GraphData { nodes, edges })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GraphData {
        crate::pg_json::decode(
            r#"{"nodes":[{"id":"a","labels":["P","Q"],"properties":{"n":42,"w":3.5,"tags":["x","y"]}},{"id":"b","labels":[],"properties":{}}],"edges":[{"id":"e0","from":"a","to":"b","labels":["KNOWS"],"properties":{"since":2020}}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn round_trip() {
        let g = sample();
        let g2 = decode(&encode(&g)).unwrap();
        assert_eq!(g2.nodes.len(), 2);
        assert_eq!(g2.edges.len(), 1);
        assert_eq!(g2.nodes[0].labels, vec!["P", "Q"]); // multi-label via `::`
        assert_eq!(g2.nodes[1].labels, Vec::<String>::new()); // empty set
        assert_eq!(g2.nodes[0].props, g.nodes[0].props);
        assert_eq!(g2.edges[0].props, g.edges[0].props);
    }

    #[test]
    fn int_vs_float_typed() {
        let s = encode(&sample());
        assert!(s.contains("g:Int64"));
        assert!(s.contains("g:Double"));
    }

    #[test]
    fn strict_decode_rejects_malformed() {
        let code = |s: &str| decode(s).err().unwrap().code;
        assert_eq!(code("{bad"), crate::E_INVALID_JSON);
        assert_eq!(code("[]"), E_INVALID_SHAPE);
        assert_eq!(
            code(r#"{"vertices":[{"@type":"g:Vertex"}]}"#),
            E_INVALID_SHAPE
        );
        assert_eq!(
            code(r#"{"vertices":[{"@value":{"label":""}}]}"#),
            E_INVALID_SHAPE
        );
        assert_eq!(
            code(r#"{"vertices":[{"@value":{"id":"a"}}]}"#),
            E_INVALID_SHAPE
        );
        assert!(decode(r#"{"vertices":[{"@value":{"id":"a","label":""}}]}"#).is_ok());
    }
}
