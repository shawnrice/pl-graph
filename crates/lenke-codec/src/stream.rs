//! Streaming encoders: the host drives element-by-element, feeding BORROWED views
//! (`&str` ids/labels, scalar props as [`ValueRef`]) instead of first building an
//! owned [`GraphData`] copy of the whole graph. All format syntax still lives here
//! — the host only supplies data — so the streaming and [`GraphData`] paths emit
//! byte-identical output.
//!
//! Why it exists: `serialize(&to_graph_data(store))` deep-copies every id, label
//! and property value into owned `String`/`Vec` before writing a byte, so a native
//! encode paid for the graph twice and barely beat pure-JS. Streaming borrows the
//! store's own bytes (a `GStr` property derefs to `&str` with no allocation), so
//! the encoder does one pass and the copy is gone.

use crate::jsonfmt::{push_json_str, push_num, push_temporal, push_value};
use crate::model::Value;

/// A property value the host hands the encoder WITHOUT owning a codec [`Value`].
/// Scalars borrow (`Str(&str)`), so the common case allocates nothing; a nested
/// list/map/record falls back to an owned codec `Value` the caller keeps alive for
/// the single `push` call (rare in practice).
pub enum ValueRef<'a> {
    Null,
    Bool(bool),
    Num(f64),
    Str(&'a str),
    Temporal { tag: &'a str, iso: &'a str },
    Nested(&'a Value),
}

/// Write a [`ValueRef`] as a JSON value — the borrowed twin of
/// [`push_value`](crate::push_value); both share the same primitive writers, so
/// their output is byte-identical.
pub fn push_value_ref(out: &mut String, v: ValueRef) {
    match v {
        ValueRef::Null => out.push_str("null"),
        ValueRef::Bool(b) => out.push_str(if b { "true" } else { "false" }),
        ValueRef::Num(x) => push_num(out, x),
        ValueRef::Str(s) => push_json_str(out, s),
        ValueRef::Temporal { tag, iso } => push_temporal(out, tag, iso),
        ValueRef::Nested(v) => push_value(out, v),
    }
}

/// The label array of one element, written comma-separated inside `[...]`.
pub struct Labels<'a> {
    out: &'a mut String,
    any: bool,
}
impl Labels<'_> {
    /// Append one label, in the order the host yields them (already the store's
    /// canonical sorted order).
    pub fn push(&mut self, label: &str) {
        if self.any {
            self.out.push(',');
        }
        self.any = true;
        push_json_str(self.out, label);
    }
}

/// The property object of one element, written comma-separated inside `{...}`.
pub struct Props<'a> {
    out: &'a mut String,
    any: bool,
}
impl Props<'_> {
    /// Append one present property (`"key":value`), in the host's key order.
    pub fn push(&mut self, key: &str, value: ValueRef) {
        if self.any {
            self.out.push(',');
        }
        self.any = true;
        push_json_str(self.out, key);
        self.out.push(':');
        push_value_ref(self.out, value);
    }
}

/// A streaming PG-JSON encoder. Emits `{"nodes":[…],"edges":[…]}` — identical to
/// [`pg_json::encode`](crate::serialize) — but pulls each element from the host.
///
/// Order of use: construct, call [`node`](Self::node) once per live node,
/// [`begin_edges`](Self::begin_edges), [`edge`](Self::edge) once per edge, then
/// [`finish`](Self::finish).
pub struct PgJsonSink {
    out: String,
    any: bool,
}

impl PgJsonSink {
    /// A sink pre-sized like the `GraphData` encoder (`~64` bytes/element).
    #[must_use]
    pub fn new(nodes: usize, edges: usize) -> Self {
        let mut out = String::with_capacity(nodes * 64 + edges * 64 + 16);
        out.push_str("{\"nodes\":[");
        Self { out, any: false }
    }

    fn sep(&mut self) {
        if self.any {
            self.out.push(',');
        }
        self.any = true;
    }

    /// Emit one node. `labels`/`props` run against a live cursor, so the host reads
    /// borrowed values into locals and hands them over without owning a codec value.
    pub fn node(
        &mut self,
        id: &str,
        labels: impl FnOnce(&mut Labels),
        props: impl FnOnce(&mut Props),
    ) {
        self.sep();
        self.out.push_str("{\"id\":");
        push_json_str(&mut self.out, id);
        self.out.push_str(",\"labels\":[");
        labels(&mut Labels {
            out: &mut self.out,
            any: false,
        });
        self.out.push_str("],\"properties\":{");
        props(&mut Props {
            out: &mut self.out,
            any: false,
        });
        self.out.push_str("}}");
    }

    /// Close the node array and open the edge array. Call once, after the last node.
    pub fn begin_edges(&mut self) {
        self.out.push_str("],\"edges\":[");
        self.any = false;
    }

    /// Emit one edge (`labels[0]` is its type). Every edge carries an id.
    pub fn edge(
        &mut self,
        id: &str,
        from: &str,
        to: &str,
        labels: impl FnOnce(&mut Labels),
        props: impl FnOnce(&mut Props),
    ) {
        self.sep();
        self.out.push_str("{\"id\":");
        push_json_str(&mut self.out, id);
        self.out.push_str(",\"from\":");
        push_json_str(&mut self.out, from);
        self.out.push_str(",\"to\":");
        push_json_str(&mut self.out, to);
        self.out.push_str(",\"undirected\":false,\"labels\":[");
        labels(&mut Labels {
            out: &mut self.out,
            any: false,
        });
        self.out.push_str("],\"properties\":{");
        props(&mut Props {
            out: &mut self.out,
            any: false,
        });
        self.out.push_str("}}");
    }

    /// Close the edge array and return the finished document.
    #[must_use]
    pub fn finish(mut self) -> String {
        self.out.push_str("]}");
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, GraphData, Node};

    /// Drive `PgJsonSink` from a `GraphData` (what the host does from a store) and
    /// prove the streaming output is byte-identical to the `GraphData` encoder — the
    /// guarantee that lets the host skip the owned copy without changing a byte.
    fn stream_pg_json(g: &GraphData) -> String {
        fn vref(v: &Value) -> ValueRef<'_> {
            match v {
                Value::Null => ValueRef::Null,
                Value::Bool(b) => ValueRef::Bool(*b),
                Value::Num(x) => ValueRef::Num(*x),
                Value::Str(s) => ValueRef::Str(s),
                Value::Temporal { tag, iso } => ValueRef::Temporal { tag, iso },
                nested => ValueRef::Nested(nested),
            }
        }
        let mut sink = PgJsonSink::new(g.nodes.len(), g.edges.len());
        for n in &g.nodes {
            sink.node(
                &n.id,
                |l| n.labels.iter().for_each(|x| l.push(x)),
                |p| n.props.iter().for_each(|(k, v)| p.push(k, vref(v))),
            );
        }
        sink.begin_edges();
        for e in &g.edges {
            sink.edge(
                e.id.as_deref().unwrap_or(""),
                &e.from,
                &e.to,
                |l| e.labels.iter().for_each(|x| l.push(x)),
                |p| e.props.iter().for_each(|(k, v)| p.push(k, vref(v))),
            );
        }
        sink.finish()
    }

    #[test]
    fn streaming_matches_graphdata_encoder_byte_for_byte() {
        let g = GraphData {
            nodes: vec![
                Node {
                    id: "a".into(),
                    labels: vec!["Person".into()],
                    props: vec![
                        ("name".into(), Value::Str("ann \"x\"".into())),
                        ("age".into(), Value::Num(30.0)),
                        ("active".into(), Value::Bool(true)),
                        ("nick".into(), Value::Null),
                        (
                            "born".into(),
                            Value::Temporal {
                                tag: "date".into(),
                                iso: "2024-01-15".into(),
                            },
                        ),
                        (
                            "tags".into(),
                            Value::List(vec![Value::Str("x".into()), Value::Num(1.0)]),
                        ),
                        (
                            "meta".into(),
                            Value::Map(vec![("k".into(), Value::Num(2.0))]),
                        ),
                    ],
                },
                Node {
                    id: "b".into(),
                    labels: vec!["Person".into(), "Admin".into()],
                    props: vec![],
                },
            ],
            edges: vec![Edge {
                id: Some("e0".into()),
                from: "a".into(),
                to: "b".into(),
                labels: vec!["KNOWS".into()],
                props: vec![("since".into(), Value::Num(2020.0))],
            }],
        };
        assert_eq!(stream_pg_json(&g), crate::pg_json::encode(&g));
    }
}
