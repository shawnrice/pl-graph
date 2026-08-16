//! The PG textual format (`.pg`) over the neutral model — a faithful port of
//! lenke-core's `codec::pg_text`, retyped to [`GraphData`]/[`Value`]. One element
//! per line:
//! ```text
//! <id> :Label* key:value*          ← a node (one leading id)
//! <from> <to> :Label* key:value*   ← an edge (two leading ids)
//! ```
//! Told apart by the second token: a bare id (no `:`) means an edge. `#` starts a
//! comment.
//!
//! Values: strings are double-quoted (escaping `"`/`\` and the whitespace control
//! chars); numbers/booleans/`null` are bare; a temporal rides as an unquoted
//! `@<tag>:<iso>` token; a list rides on **repeated keys** (`tags:1 tags:2`), so
//! an empty list emits nothing and a single-element list is indistinguishable from
//! a scalar. The textual format has no edge-id slot, so a decoded edge carries no
//! id (the host re-derives the canonical `e{index}`).

use crate::model::{is_temporal_tag, Edge, GraphData, Node, Value};

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
                out.push_str(&crate::js_number(*x));
            } else {
                out.push_str("null");
            }
        }
        // A temporal rides as an unquoted `@<tag>:<iso>` token — the ISO form has
        // no whitespace/newline, so it stays on one physical line, and the `@`
        // sigil lets the parser tell it from a quoted string.
        Value::Temporal { tag, iso } => {
            out.push('@');
            out.push_str(tag);
            out.push(':');
            out.push_str(iso);
        }
        Value::Str(s) => {
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

/// Render a label token (`:label`); an embedded `:` needs no quoting, but
/// whitespace / quote / backslash still do.
fn label_token(s: &str) -> String {
    if s.chars()
        .any(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '"' | '\\'))
    {
        quote_escaped(s)
    } else {
        s.to_string()
    }
}

/// Render an id as a token, quoting + escaping it when it contains a `:`,
/// whitespace, a quote/backslash, or is empty — so ids round-trip instead of
/// corrupting the line shape.
fn id_token(s: &str) -> String {
    let needs_quote = s.is_empty()
        || s.chars()
            .any(|c| matches!(c, ':' | ' ' | '\t' | '\n' | '\r' | '"' | '\\'));
    if needs_quote {
        quote_escaped(s)
    } else {
        s.to_string()
    }
}

fn element_line(leading: &[&str], labels: &[String], props: &[(String, Value)]) -> String {
    let mut tokens: Vec<String> = leading.iter().map(|s| id_token(s)).collect();
    for l in labels {
        tokens.push(format!(":{}", label_token(l)));
    }
    for (k, v) in props {
        push_property(&mut tokens, k, v);
    }
    tokens.join(" ")
}

/// Serialize neutral graph data to PG-text: node lines, then edge lines.
pub fn encode(g: &GraphData) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(g.nodes.len() + g.edges.len());
    for n in &g.nodes {
        lines.push(element_line(&[&n.id], &n.labels, &n.props));
    }
    for e in &g.edges {
        lines.push(element_line(&[&e.from, &e.to], &e.labels, &e.props));
    }
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Split a line into tokens, keeping double-quoted spans (with `\` escapes) whole.
fn tokenize(line: &str) -> Vec<&str> {
    let b = line.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut start = 0;
    let mut started = false;
    let mut in_quote = false;

    while i < b.len() {
        let c = b[i];
        if in_quote {
            if c == b'\\' && i + 1 < b.len() {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_quote = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            if !started {
                start = i;
                started = true;
            }
            in_quote = true;
        } else if c == b' ' || c == b'\t' {
            if started {
                tokens.push(&line[start..i]);
                started = false;
            }
        } else if !started {
            start = i;
            started = true;
        }
        i += 1;
    }
    if started {
        tokens.push(&line[start..]);
    }
    tokens
}

/// Looks like a JS-`Number`-shaped token (so `1e3` parses, but `inf` does not).
fn is_number(raw: &str) -> bool {
    let first = raw.as_bytes().first().copied();
    matches!(first, Some(b'0'..=b'9') | Some(b'-') | Some(b'.'))
        && raw.parse::<f64>().is_ok_and(f64::is_finite)
}

/// Parse the value half of a `key:value` token into a scalar value.
fn parse_scalar(raw: &str) -> Value {
    if let Some(rest) = raw.strip_prefix('"') {
        let body = rest.strip_suffix('"').unwrap_or(rest);
        return Value::Str(unescape(body));
    }
    // A tagged temporal `@<tag>:<iso>` (unquoted; the `@` sigil disambiguates it
    // from a bare string). An UNKNOWN tag falls through to string handling — the
    // ISO string itself is validated by the host when it builds its graph, and a
    // parse failure there is reconstructed back to this exact `@tag:iso` token
    // (matching pg-text's lenient decode policy).
    if let Some(rest) = raw.strip_prefix('@') {
        if let Some((tag, iso)) = rest.split_once(':') {
            if is_temporal_tag(tag) {
                return Value::Temporal {
                    tag: tag.to_string(),
                    iso: iso.to_string(),
                };
            }
        }
    }
    match raw {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        _ if is_number(raw) => Value::Num(raw.parse().unwrap()),
        _ => Value::Str(raw.to_string()), // bare unquoted string (lenient for foreign .pg)
    }
}

/// A line's labels plus its first-seen-ordered properties (repeated keys → lists).
type LabelsAndProps = (Vec<String>, Vec<(String, Value)>);

fn parse_labels_props(tokens: &[&str]) -> LabelsAndProps {
    let mut labels = Vec::new();
    let mut props: Vec<(String, Value)> = Vec::new();
    let mut promoted: Vec<usize> = Vec::new();

    for token in tokens {
        if let Some(rest) = token.strip_prefix(':') {
            labels.push(parse_id(rest));
            continue;
        }
        let sep = if token.starts_with('"') {
            let end = quoted_span_end(token, 0);
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
        let key = parse_id(&token[..sep]);
        let value = parse_scalar(&token[sep + 1..]);

        match props.iter().position(|(k, _)| *k == key) {
            Some(pos) if promoted.contains(&pos) => {
                if let Value::List(items) = &mut props[pos].1 {
                    items.push(value);
                }
            }
            Some(pos) => {
                let prev = std::mem::replace(&mut props[pos].1, Value::Null);
                props[pos].1 = Value::List(vec![prev, value]);
                promoted.push(pos);
            }
            None => props.push((key, value)),
        }
    }
    (labels, props)
}

/// Index just past the closing `"` of a quoted span at `start`, respecting
/// `\`-escapes; `s.len()` if unterminated.
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

/// A leading token is an id iff it's a *whole* quoted span or has no `:`.
fn is_id_token(t: &str) -> bool {
    if t.starts_with('"') {
        quoted_span_end(t, 0) == t.len()
    } else {
        !t.contains(':')
    }
}

/// A second token that is an id (not a `:label` / `key:value`) marks an edge line.
fn is_edge_line(tokens: &[&str]) -> bool {
    tokens.len() >= 2 && is_id_token(tokens[1])
}

/// Read an id token, unquoting + unescaping it if it was quoted.
fn parse_id(raw: &str) -> String {
    let Some(rest) = raw.strip_prefix('"') else {
        return raw.to_string();
    };
    let body = rest.strip_suffix('"').unwrap_or(rest);
    unescape(body)
}

/// Undo the encode escapes: `\n`/`\r`/`\t` → the control chars, `\\`/`\"` → self,
/// any other `\x` → a literal `x` (lenient for foreign `.pg`).
fn unescape(body: &str) -> String {
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

/// Deserialize a PG-text string into neutral graph data. Endpoints referenced by
/// an edge but never declared as a node line are the host's concern (pg-text is
/// the lenient codec — the host auto-creates them). Decode is infallible.
pub fn decode(input: &str) -> GraphData {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
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
            let from = parse_id(tokens[0]);
            let to = parse_id(tokens[1]);
            let (labels, props) = parse_labels_props(&tokens[2..]);
            edges.push(Edge {
                id: None,
                from,
                to,
                labels,
                props,
            });
        } else {
            let id = parse_id(tokens[0]);
            let (labels, props) = parse_labels_props(&tokens[1..]);
            nodes.push(Node { id, labels, props });
        }
    }
    GraphData { nodes, edges }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_scalars_and_lists() {
        let g = decode(
            "a :Person name:\"Ann\" age:30 active:true tags:x tags:y\nb :Person :Admin name:\"Bo\"\na b :KNOWS since:2020",
        );
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.nodes[0].props[1], ("age".to_string(), Value::Num(30.0)));
        assert_eq!(
            g.nodes[0].props[2],
            ("active".to_string(), Value::Bool(true))
        );
        assert_eq!(
            g.nodes[0].props[3].1,
            Value::List(vec![Value::Str("x".into()), Value::Str("y".into())]),
        );
        assert_eq!(g.nodes[1].labels, vec!["Person", "Admin"]);
        // stable through a round trip
        let g2 = decode(&encode(&g));
        assert_eq!(g2.nodes[0].props[3].1, g.nodes[0].props[3].1);
        assert_eq!(g2.edges.len(), 1);
    }

    #[test]
    fn a_key_repeated_three_times_collects_in_order() {
        let g = decode("a t:x t:y t:z");
        assert_eq!(
            g.nodes[0].props[0].1,
            Value::List(vec![
                Value::Str("x".into()),
                Value::Str("y".into()),
                Value::Str("z".into())
            ]),
        );
    }

    #[test]
    fn temporal_token_round_trips() {
        let g = decode("a on:@date:2020-02-29 took:@duration:P3M10DT90S");
        assert_eq!(
            g.nodes[0].props[0],
            (
                "on".to_string(),
                Value::Temporal {
                    tag: "date".into(),
                    iso: "2020-02-29".into()
                }
            ),
        );
        assert_eq!(
            encode(&g),
            "a on:@date:2020-02-29 took:@duration:P3M10DT90S"
        );
    }

    #[test]
    fn unknown_temporal_tag_stays_a_string() {
        let g = decode("a x:@nope:foo");
        assert_eq!(g.nodes[0].props[0].1, Value::Str("@nope:foo".into()));
    }

    #[test]
    fn edge_endpoint_and_comments() {
        let g = decode("# a comment\na b :KNOWS");
        assert_eq!(g.nodes.len(), 0); // endpoints are the host's concern (lenient)
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.edges[0].labels, vec!["KNOWS"]);
    }

    #[test]
    fn quoted_control_chars_round_trip() {
        let g = decode("a name:\"a b\\\"c\\nd\"");
        assert_eq!(g.nodes[0].props[0].1, Value::Str("a b\"c\nd".into()));
        let g2 = decode(&encode(&g));
        assert_eq!(g2.nodes[0].props[0].1, Value::Str("a b\"c\nd".into()));
        assert_eq!(
            encode(&g).lines().count(),
            1,
            "a control char leaked a newline"
        );
    }
}
