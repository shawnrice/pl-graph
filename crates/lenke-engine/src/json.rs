//! JSON serialization of engine query results, for the CROSS-ENGINE COMPARISON harness
//! (`packages/native`'s `backend-engine`). The engine's `Value`s are already
//! byte-identical to core's, so this only has to match core's JSON STRUCTURE and key
//! order — the differential fuzzer compares `JSON.stringify(rows)` of the decoded
//! values, not raw bytes, so a number's exact text does not matter (it round-trips
//! through a JS `number`), but object key order and nesting do.
//!
//! Two entry points mirror the two FFI result shapes:
//!   - [`gremlin_results_json`] — a bare JSON ARRAY of per-result values (like core's
//!     `lnk_gremlin_json`).
//!   - [`gql_rows_json`] — a JSON ARRAY of ROW OBJECTS keyed by column name (the decoded
//!     shape of core's `lnk_query_rows`).

use crate::exec::Rows;
use crate::value::Value;
use std::fmt::Write;

/// Append `v` to `out` as JSON.
fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => write_number(out, *x),
        Value::Str(s) => write_string(out, s),
        Value::Temporal(t) => write_string(out, &t.format()),
        Value::List(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, it);
            }
            out.push(']');
        }
        Value::Record(fields) => {
            // A record renders as an object keyed by field name, in field order.
            out.push('{');
            for (i, (k, val)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, val);
            }
            out.push('}');
        }
        Value::Map(pairs) => {
            // A map (element map, group map, …) renders as an object in INSERTION order
            // — key order is significant for the value-level compare. Keys are stringified.
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                match k {
                    Value::Str(s) => write_string(out, s),
                    // A non-string key (a numeric group key, T.label token) → its JSON
                    // scalar text as the object key, matching core's map rendering.
                    other => {
                        let mut kb = String::new();
                        write_value(&mut kb, other);
                        write_string(out, kb.trim_matches('"'));
                    }
                }
                out.push(':');
                write_value(out, val);
            }
            out.push('}');
        }
    }
}

/// A finite number as its shortest round-tripping text; a non-finite one as `null`
/// (JSON has no NaN/Infinity — core emits null too, per the NaN policy).
fn write_number(out: &mut String, x: f64) {
    if !x.is_finite() {
        out.push_str("null");
    } else if x.fract() == 0.0 && x.abs() < 1e15 {
        // Integer-valued: emit without a decimal point (matches core; parses to the
        // same JS number regardless, but keeps the text tidy).
        let _ = write!(out, "{}", x as i64);
    } else {
        let _ = write!(out, "{x}");
    }
}

/// A JSON string literal (RFC 8259 escaping).
fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Gremlin results: a bare JSON array of per-result values (one column, `_`).
#[must_use]
pub fn gremlin_results_json(rows: &Rows) -> String {
    let mut out = String::from("[");
    for (i, row) in rows.rows.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // A single-column Gremlin result: the one cell.
        write_value(&mut out, row.first().unwrap_or(&Value::Null));
    }
    out.push(']');
    out
}

/// GQL rows: the `{columns, rows}` document core's `lnk_query_rows` returns (its
/// `RowSet::to_json`), which the TS `decodeRows` zips into per-row objects. `columns`
/// is the column-name list; `rows` is a positional matrix of cells.
#[must_use]
pub fn gql_rows_json(rows: &Rows) -> String {
    let mut out = String::from("{\"columns\":[");
    for (c, name) in rows.names.iter().enumerate() {
        if c > 0 {
            out.push(',');
        }
        write_string(&mut out, name);
    }
    out.push_str("],\"rows\":[");
    let ncols = rows.names.len();
    for (r, row) in rows.rows.iter().enumerate() {
        if r > 0 {
            out.push(',');
        }
        out.push('[');
        for c in 0..ncols {
            if c > 0 {
                out.push(',');
            }
            write_value(&mut out, row.get(c).unwrap_or(&Value::Null));
        }
        out.push(']');
    }
    out.push_str("]}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::Flat;
    use std::sync::Arc;

    fn rows(names: &[&str], data: Vec<Vec<Value>>) -> Rows {
        Rows {
            names: names.iter().map(|s| (*s).to_string()).collect(),
            rows: Flat::from_rows(data),
        }
    }

    #[test]
    fn scalars_render_as_json() {
        let mut s = String::new();
        write_value(&mut s, &Value::Num(3.0));
        write_value(&mut s, &Value::Bool(true));
        write_value(&mut s, &Value::Null);
        assert_eq!(s, "3truenull");
    }

    #[test]
    fn integer_valued_numbers_drop_the_decimal_point() {
        let mut s = String::new();
        write_number(&mut s, 42.0);
        write_number(&mut s, -1.0);
        write_number(&mut s, 2.5);
        assert_eq!(s, "42-12.5");
    }

    #[test]
    fn non_finite_numbers_are_null() {
        // JSON has no NaN/Infinity; core emits null too (NaN policy).
        let mut s = String::new();
        write_number(&mut s, f64::NAN);
        write_number(&mut s, f64::INFINITY);
        assert_eq!(s, "nullnull");
    }

    #[test]
    fn strings_are_escaped() {
        let mut s = String::new();
        write_string(&mut s, "a\"b\\c\n\t");
        assert_eq!(s, r#""a\"b\\c\n\t""#);
    }

    #[test]
    fn control_chars_use_u_escapes() {
        let mut s = String::new();
        write_string(&mut s, "\u{01}");
        assert_eq!(s, "\"\\u0001\"");
    }

    #[test]
    fn map_preserves_pair_order() {
        // A Value::Map renders in the pair order it holds — key order is significant
        // for the value-level compare, so the serializer must not reorder.
        let m = Value::Map(Arc::new(vec![
            (Value::Str(Arc::from("n")), Value::Num(1.0)),
            (Value::Str(Arc::from("s")), Value::Str(Arc::from("x"))),
        ]));
        let mut s = String::new();
        write_value(&mut s, &m);
        assert_eq!(s, r#"{"n":1,"s":"x"}"#);
    }

    #[test]
    fn record_renders_as_object_in_field_order() {
        let r = Value::Record(Arc::from(
            vec![
                (Arc::from("b"), Value::Num(2.0)),
                (Arc::from("a"), Value::Num(1.0)),
            ]
            .into_boxed_slice(),
        ));
        let mut s = String::new();
        write_value(&mut s, &r);
        assert_eq!(s, r#"{"b":2,"a":1}"#);
    }

    #[test]
    fn gremlin_results_are_a_bare_array() {
        let r = rows(&["_"], vec![vec![Value::Num(1.0)], vec![Value::Num(2.0)]]);
        assert_eq!(gremlin_results_json(&r), "[1,2]");
    }

    #[test]
    fn gql_rows_are_a_columns_rows_document() {
        let r = rows(
            &["a", "b"],
            vec![
                vec![Value::Num(1.0), Value::Str(Arc::from("x"))],
                vec![Value::Num(2.0), Value::Null],
            ],
        );
        assert_eq!(
            gql_rows_json(&r),
            r#"{"columns":["a","b"],"rows":[[1,"x"],[2,null]]}"#
        );
    }

    #[test]
    fn empty_result_sets_render() {
        let g = rows(&["_"], vec![]);
        assert_eq!(gremlin_results_json(&g), "[]");
        let q = rows(&["a"], vec![]);
        assert_eq!(gql_rows_json(&q), r#"{"columns":["a"],"rows":[]}"#);
    }
}
