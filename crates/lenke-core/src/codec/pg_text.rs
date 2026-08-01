//! The PG textual format (`.pg`) — the line-based companion to PG-JSON. One
//! element per line:
//! ```text
//! <id> :Label* key:value*          ← a node (one leading id)
//! <from> <to> :Label* key:value*   ← an edge (two leading ids)
//! ```
//! Told apart by the second token: a bare id (no `:`) means an edge. `#` starts a
//! comment.
//!
//! Value mapping: strings are double-quoted (escaping `"` and `\`); numbers,
//! booleans, and `null` are bare; a list rides on **repeated keys**
//! (`tags:1 tags:2`). On decode a key seen once is a scalar, more than once a
//! list — so (as in the TS codec) an empty list emits nothing (decodes as absent)
//! and a single-element list is indistinguishable from a scalar. Node ids are
//! preserved; the textual format has no edge-id slot, so an *assigned* edge id is
//! not round-tripped — a decoded edge re-derives the canonical `e{index}` id
//! (use PG-JSON / GraphSON / CSV to round-trip an assigned edge id). An edge's
//! single type is its first `:Label`.

use crate::codec::{element_props, node_labels};
use crate::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Render one scalar value as a PG-text token value (never a list).
fn scalar_token(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => {
            if x.is_finite() {
                out.push_str(&crate::jsonfmt::js_number(*x));
            } else {
                out.push_str("null");
            }
        }
        // Temporals ride as an unquoted `@<kind>:<iso>` token — the ISO form has
        // no whitespace/newline, so it stays on one physical line, and the `@`
        // sigil lets the parser distinguish it from a quoted string.
        Value::Temporal(t) => {
            out.push('@');
            out.push_str(t.tag());
            out.push(':');
            out.push_str(&t.format());
        }
        Value::Str(s) => {
            out.push('"');
            for c in s.chars() {
                // Escape the quote/backslash AND the line/whitespace control chars
                // — pg-text is line-oriented, so an unescaped newline in a value
                // would split the token across physical lines and corrupt the
                // round-trip. Must match the TS codec's escape scheme exactly.
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Value::List(_) => {} // handled by the caller (one token per element)
        // A map has no flat-token form; `serialize` pre-rejects a pg-text export
        // that contains one, so this is unreachable in practice.
        Value::Map(_) => {
            unreachable!("pg-text cannot carry a map; serialize() rejects it up front")
        }
    }
}

/// Append `key:value` tokens for one property (a list expands to one per element).
fn push_property(tokens: &mut Vec<String>, key: &str, v: &Value) {
    let k = id_token(key); // arbitrary keys are quoted like ids
    match v {
        Value::List(elems) => {
            for el in elems {
                let mut t = format!("{k}:");
                scalar_token(&mut t, el);
                tokens.push(t);
            }
        }
        _ => {
            let mut t = format!("{k}:");
            scalar_token(&mut t, v);
            tokens.push(t);
        }
    }
}

/// Escape a string's quote/backslash/control chars into a quoted token body.
/// Shared by `id_token` and `label_token`; must match the TS escape scheme.
fn quote_escaped(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Render a label token. A label is emitted as `:label`, so an embedded `:` is
/// unambiguous and needs no quoting — but whitespace / quote / backslash still
/// do (whitespace splits the token; a leading quote reads as a quoted span).
fn label_token(s: &str) -> String {
    if s.chars()
        .any(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '"' | '\\'))
    {
        quote_escaped(s)
    } else {
        s.to_string()
    }
}

/// Render an id as a token, quoting + escaping it when it contains a `:`
/// (which would otherwise read as a `:label` / `key:value`), whitespace (which
/// would split the token), a quote/backslash, or a control char — so ids with
/// colons or spaces round-trip instead of corrupting the line shape.
fn id_token(s: &str) -> String {
    let needs_quote = s.is_empty()
        || s.chars()
            .any(|c| matches!(c, ':' | ' ' | '\t' | '\n' | '\r' | '"' | '\\'));
    if !needs_quote {
        return s.to_string();
    }
    quote_escaped(s)
}

fn element_line(leading: &[&str], labels: &[&str], props: &[(&str, Value)]) -> String {
    let mut tokens: Vec<String> = leading.iter().map(|s| id_token(s)).collect();
    for l in labels {
        tokens.push(format!(":{}", label_token(l)));
    }
    for (k, v) in props {
        push_property(&mut tokens, k, v);
    }
    tokens.join(" ")
}

/// Serialize a graph to PG-text: node lines, then edge lines.
pub fn encode(g: &Graph) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(g.vertex_count() + g.edge_count());
    for vi in 0..g.n {
        if !g.is_vertex_live(vi as u32) {
            continue;
        }
        let id = g.vid.text(vi as u32);
        lines.push(element_line(
            &[id],
            &node_labels(g, vi as u32),
            &element_props(&g.props, &g.strs, vi),
        ));
    }
    for i in 0..g.edge_slots() {
        if !g.is_edge_live(i as u32) {
            continue;
        }
        let from = g.vid.text(g.e_src[i]);
        let to = g.vid.text(g.e_dst[i]);
        let etype = g.etype.text(g.e_type[i]);
        lines.push(element_line(
            &[from, to],
            &[etype],
            &element_props(&g.edge_props, &g.strs, i),
        ));
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Split a line into tokens, keeping double-quoted spans (with `\` escapes) whole.
fn tokenize(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut started = false;
    let mut in_quote = false;
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_quote {
            current.push(c);
            if c == '\\' && i + 1 < chars.len() {
                current.push(chars[i + 1]);
                i += 2;
                continue;
            } else if c == '"' {
                in_quote = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_quote = true;
            started = true;
            current.push(c);
        } else if c == ' ' || c == '\t' {
            if started {
                tokens.push(std::mem::take(&mut current));
                started = false;
            }
        } else {
            current.push(c);
            started = true;
        }
        i += 1;
    }
    if started {
        tokens.push(current);
    }
    tokens
}

/// Looks like a JS-`Number`-shaped token (so a bare `1e3` parses, but `inf` does not).
fn is_number(raw: &str) -> bool {
    let first = raw.as_bytes().first().copied();
    // Require a FINITE parse: Rust's f64::from_str accepts "inf"/"-inf"/"nan", but
    // JS Number() maps those to NaN (→ treated as a string), so a foreign token like
    // "-inf" must not be read as a number here (byte-identity with the TS decoder).
    matches!(first, Some(b'0'..=b'9') | Some(b'-') | Some(b'.'))
        && raw.parse::<f64>().is_ok_and(|n| n.is_finite())
}

/// Parse the value half of a `key:value` token into a scalar value.
fn parse_scalar(raw: &str) -> Value {
    if let Some(rest) = raw.strip_prefix('"') {
        let body = rest.strip_suffix('"').unwrap_or(rest);
        // Undo the encode escapes: `\n`/`\r`/`\t` decode to the control chars,
        // `\\`/`\"` to themselves, and any other `\x` to a literal `x` (lenient
        // for foreign `.pg`). Must match the TS codec exactly.
        let mut out = String::with_capacity(body.len());
        let mut chars = body.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(other) => out.push(other),
                    None => {}
                }
            } else {
                out.push(c);
            }
        }
        return Value::Str(out.into());
    }
    // A tagged temporal `@<kind>:<iso>` (unquoted; the `@` sigil disambiguates it
    // from a bare string). A malformed tag/ISO falls through to string handling,
    // matching pg-text's lenient decode policy.
    if let Some(rest) = raw.strip_prefix('@') {
        if let Some((tag, iso)) = rest.split_once(':') {
            if let Ok(t) = crate::temporal::Temporal::parse(tag, iso) {
                return Value::Temporal(t);
            }
        }
    }
    match raw {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        _ if is_number(raw) => Value::Num(raw.parse().unwrap()),
        _ => Value::Str(raw.into()), // bare unquoted string (lenient for foreign .pg)
    }
}

/// Pull labels and properties (repeated keys → lists) from the trailing tokens.
fn parse_labels_props(tokens: &[String]) -> (Vec<String>, Vec<(String, Value)>) {
    let mut labels = Vec::new();
    // preserve first-seen key order, collecting repeats
    let mut order: Vec<String> = Vec::new();
    let mut collected: std::collections::HashMap<String, Vec<Value>> =
        std::collections::HashMap::new();
    for token in tokens {
        if let Some(rest) = token.strip_prefix(':') {
            // `:label` or `:"quoted label"` — unquote the latter.
            labels.push(if rest.starts_with('"') {
                parse_id(rest)
            } else {
                rest.to_string()
            });
            continue;
        }
        // Find the key:value separator, skipping a quoted key's own colons.
        let sep = if token.starts_with('"') {
            let end = quoted_span_end(token, 0);
            // A whole quoted span with no trailing `:value` is a stray id-shaped
            // token, not a property; ignore it.
            if end >= token.len() || token.as_bytes()[end] != b':' {
                continue;
            }
            end
        } else {
            match token.find(':') {
                Some(c) => c,
                None => continue,
            }
        };
        let key = parse_id(&token[..sep]); // unquotes a quoted key, else verbatim
        let value = parse_scalar(&token[sep + 1..]);
        if let Some(list) = collected.get_mut(&key) {
            list.push(value);
        } else {
            order.push(key.clone());
            collected.insert(key, vec![value]);
        }
    }
    let props = order
        .into_iter()
        .map(|k| {
            let mut vals = collected.remove(&k).unwrap();
            let v = if vals.len() == 1 {
                vals.pop().unwrap()
            } else {
                Value::List(vals)
            };
            (k, v)
        })
        .collect();
    (labels, props)
}

/// Index just past the closing `"` of a quoted span at `start` (`bytes[start]`
/// must be `"`), respecting `\`-escapes; `s.len()` if unterminated. Operates on
/// bytes — quote/backslash are ASCII, so this is UTF-8 safe.
fn quoted_span_end(s: &str, start: usize) -> usize {
    let b = s.as_bytes();
    let mut i = start + 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    s.len()
}

/// A leading token is an id iff it's a *whole* quoted span (`"…"` with nothing
/// after — an id) or has no `:` (a bare id). A quoted-then-`:value` token is a
/// quoted property key, NOT an id — this keeps node/edge detection correct now
/// that keys can be quoted.
fn is_id_token(t: &str) -> bool {
    if t.starts_with('"') {
        quoted_span_end(t, 0) == t.len()
    } else {
        !t.contains(':')
    }
}

/// A second token that is an id (not a `:label` / `key:value`) marks an edge line.
fn is_edge_line(tokens: &[String]) -> bool {
    tokens.len() >= 2 && is_id_token(&tokens[1])
}

/// Read an id token, unquoting + unescaping it if it was quoted.
fn parse_id(raw: &str) -> String {
    let Some(rest) = raw.strip_prefix('"') else {
        return raw.to_string();
    };
    let body = rest.strip_suffix('"').unwrap_or(rest);
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Deserialize a PG-text string into a fresh graph. Endpoints referenced by an
/// edge but never declared as a node line are created (bare) by `finalize` —
/// this leniency is intentional format semantics (matching the TS codec), so
/// decode is infallible and returns `Graph` directly (no coded error to carry).
pub fn decode(input: &str) -> Graph {
    let mut b = Builder::default();
    for raw in input.split('\n') {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens = tokenize(line);
        if tokens.is_empty() {
            continue;
        }
        if is_edge_line(&tokens) {
            let from = parse_id(&tokens[0]);
            let to = parse_id(&tokens[1]);
            let (labels, props) = parse_labels_props(&tokens[2..]);
            b.edges.push(EdgeRec {
                src: from.into(),
                dst: to.into(),
                etype: labels.into_iter().next().unwrap_or_default().into(),
                props: crate::graph::owned_props(props),
                id: None, // the .pg textual format has no edge-id slot
            });
        } else {
            let id = parse_id(&tokens[0]);
            let (labels, props) = parse_labels_props(&tokens[1..]);
            b.nodes.push(NodeRec {
                id: id.into(),
                labels: crate::graph::owned_labels(labels),
                props: crate::graph::owned_props(props),
            });
        }
    }
    b.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_scalars_and_lists() {
        let src = "\
a :Person name:\"Ann\" age:30 active:true tags:x tags:y
b :Person :Admin name:\"Bo\"
a b :KNOWS since:2020";
        let g = decode(src);
        assert_eq!(g.vertex_count(), 2);
        assert_eq!(g.edge_count(), 1);
        let a = g.vid.get("a").unwrap() as usize;
        assert_eq!(g.props.value(a, "age", &g.strs), Value::Num(30.0));
        assert_eq!(g.props.value(a, "active", &g.strs), Value::Bool(true));
        assert_eq!(
            g.props.value(a, "tags", &g.strs),
            Value::List(vec![Value::Str("x".into()), Value::Str("y".into())]),
        );
        // multi-label node
        assert_eq!(node_labels(&g, g.vid.get("b").unwrap()).len(), 2);

        // encode → decode is stable for scalars + multi-element lists
        let g2 = decode(&encode(&g));
        let a2 = g2.vid.get("a").unwrap() as usize;
        assert_eq!(
            g2.props.value(a2, "tags", &g2.strs),
            Value::List(vec![Value::Str("x".into()), Value::Str("y".into())]),
        );
        assert_eq!(g2.edge_count(), 1);
    }

    #[test]
    fn quoted_strings_and_comments() {
        let src = "# a comment\nx name:\"a b\\\"c\"";
        let g = decode(src);
        let x = g.vid.get("x").unwrap() as usize;
        assert_eq!(
            g.props.value(x, "name", &g.strs),
            Value::Str("a b\"c".into())
        );
    }

    #[test]
    fn edge_endpoint_autocreated() {
        let g = decode("a b :KNOWS");
        assert_eq!(g.vertex_count(), 2); // a and b created as bare nodes
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn ids_with_colons_and_spaces_round_trip() {
        // An endpoint id containing `:` (or a space) must be quoted so it is not
        // mis-read as a node line / split into tokens.
        let g = crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"a:b\",\"labels\":[\"N\"],\"properties\":{}}\n\
             {\"type\":\"node\",\"id\":\"c d\",\"labels\":[\"N\"],\"properties\":{}}\n\
             {\"type\":\"edge\",\"from\":\"a:b\",\"to\":\"c d\",\"labels\":[\"R\"],\"properties\":{}}",
        )
        .unwrap();
        let g2 = decode(&encode(&g));
        assert_eq!(
            g2.vertex_count(),
            2,
            "an id was mis-parsed into an extra node"
        );
        assert_eq!(g2.edge_count(), 1, "the edge was mis-classified as a node");
        let from = g2.vid.get("a:b").expect("node a:b");
        let to = g2.vid.get("c d").expect("node 'c d'");
        assert_eq!(g2.e_src[0], from);
        assert_eq!(g2.e_dst[0], to);
    }

    #[test]
    fn label_or_key_with_newline_cannot_forge_an_element() {
        // A raw newline in a label/key would split the physical line and inject a
        // second element on decode; quoting+escaping must prevent that.
        let g = crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"n1\",\"labels\":[\"ok\",\"evil\\n999 :Injected\"],\
              \"properties\":{\"weird key\\nx\":1}}",
        )
        .unwrap();
        let out = encode(&g);
        assert_eq!(out.lines().count(), 1, "encode leaked a raw newline");

        let g2 = decode(&out);
        assert_eq!(g2.vertex_count(), 1, "a forged element appeared");
        assert_eq!(g2.edge_count(), 0);
        let n1 = g2.vid.get("n1").unwrap() as usize;
        let mut labels = crate::codec::node_labels(&g2, n1 as u32);
        labels.sort();
        assert_eq!(labels, vec!["evil\n999 :Injected", "ok"]);
        assert_eq!(
            g2.props.value(n1, "weird key\nx", &g2.strs),
            Value::Num(1.0)
        );
    }

    #[test]
    fn labels_and_keys_with_delimiters_round_trip() {
        let g = crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"n1\",\"labels\":[\"has space\",\"has:colon\",\"has\\\"quote\"],\
              \"properties\":{\"key with space\":\"v\",\"key:with:colons\":2}}",
        )
        .unwrap();
        let g2 = decode(&encode(&g));
        let n1 = g2.vid.get("n1").unwrap() as usize;
        let mut labels = crate::codec::node_labels(&g2, n1 as u32);
        labels.sort();
        assert_eq!(labels, vec!["has space", "has\"quote", "has:colon"]);
        assert_eq!(
            g2.props.value(n1, "key with space", &g2.strs),
            Value::Str("v".into())
        );
        assert_eq!(
            g2.props.value(n1, "key:with:colons", &g2.strs),
            Value::Num(2.0)
        );
    }

    #[test]
    fn a_quoted_first_key_is_not_misread_as_an_edge() {
        let g = crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"solo\",\"labels\":[],\"properties\":{\"a b\":1}}",
        )
        .unwrap();
        let g2 = decode(&encode(&g));
        assert_eq!(g2.vertex_count(), 1);
        assert_eq!(g2.edge_count(), 0, "a node was mis-classified as an edge");
        let solo = g2.vid.get("solo").unwrap() as usize;
        assert_eq!(g2.props.value(solo, "a b", &g2.strs), Value::Num(1.0));
    }

    #[test]
    fn escapes_control_chars_in_strings() {
        // A value with newline/CR/tab/quote/backslash must survive a round trip;
        // an unescaped newline would split the line and corrupt the graph.
        let g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"note":"l1\nl2\tx\"q\\b\r"}}"#,
        )
        .unwrap();
        let text = encode(&g);
        // The value must not leak a raw newline into the output (single line).
        assert_eq!(
            text.trim_end_matches('\n').lines().count(),
            1,
            "value control char leaked into output: {text:?}"
        );
        let g2 = decode(&text);
        let a = g2.vid.get("a").unwrap() as usize;
        assert_eq!(
            g2.props.value(a, "note", &g2.strs),
            Value::Str("l1\nl2\tx\"q\\b\r".into())
        );
    }
}
