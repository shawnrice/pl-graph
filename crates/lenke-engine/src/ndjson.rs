//! NDJSON egress: dump a store as newline-delimited JSON — one object per live
//! node, then one per edge. Dependency-free (a small hand-rolled JSON writer, no
//! serde) and deterministic (nodes by id; labels and property keys sorted; edges
//! in adjacency order).
//!
//! Line shapes:
//! - node: `{"id":N,"labels":[...],"props":{...}}`
//! - edge: `{"from":F,"to":T,"type":"R","props":{...}}`
//!
//! This module SERIALIZES values; it does not define value semantics (order,
//! equality) — those stay in [`crate::value`]. A non-finite number (NaN/Inf) has
//! no JSON form and is written as `null`, consistent with the engine's
//! NaN/Inf→null policy.

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
        out.push_str("{\"id\":");
        out.push_str(&id.to_string());
        out.push_str(",\"labels\":");
        encode_str_array(&mut out, &store.labels_of(id));
        out.push_str(",\"props\":");
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
            out.push_str("{\"from\":");
            out.push_str(&from.to_string());
            out.push_str(",\"to\":");
            out.push_str(&a.nbr.to_string());
            out.push_str(",\"type\":");
            encode_string(&mut out, &store.etype_name(a.etype).unwrap_or_default());
            out.push_str(",\"props\":");
            let eid = a.eid;
            encode_object(&mut out, &edge_keys, |k| {
                store.has_edge_prop(eid, k).then(|| store.edge_prop(eid, k))
            });
            out.push_str("}\n");
        }
    }
    out
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

fn encode_str_array(out: &mut String, items: &[String]) {
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
fn encode_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => {
            if x.is_finite() {
                out.push_str(&x.to_string());
            } else {
                out.push_str("null");
            }
        }
        Value::Str(s) => encode_string(out, s),
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
    }
}

/// Encode a JSON string with the required escapes.
fn encode_string(out: &mut String, s: &str) {
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

#[cfg(test)]
mod tests {
    use super::to_ndjson;
    use crate::store::Builder;
    use crate::value::Value;
    use std::sync::Arc;

    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
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
        let expected = "{\"id\":0,\"labels\":[\"P\"],\"props\":{\"age\":1,\"name\":\"a\"}}\n\
             {\"id\":1,\"labels\":[\"P\"],\"props\":{\"name\":\"b\"}}\n\
             {\"from\":0,\"to\":1,\"type\":\"R\",\"props\":{\"weight\":0.5}}\n";
        assert_eq!(to_ndjson(&st), expected);
    }

    /// A deleted node (and its edges) is absent from the dump.
    #[test]
    fn deleted_node_excluded() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        st.add_edge(a, b, "R");
        st.delete_node(b);
        let expected = "{\"id\":0,\"labels\":[\"P\"],\"props\":{\"name\":\"a\"}}\n";
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
        let expected = "{\"id\":0,\"labels\":[],\"props\":\
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
            "{\"id\":0,\"labels\":[],\"props\":{\"v\":null}}\n"
        );
    }

    /// An empty store dumps to the empty string.
    #[test]
    fn empty_store() {
        let st = Builder::default().build();
        assert_eq!(to_ndjson(&st), "");
    }
}
