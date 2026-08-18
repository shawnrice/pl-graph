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
pub(crate) fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => write_number(out, *x),
        Value::Str(s) => write_string(out, s),
        // A temporal renders TAGGED — `{"@date":"2020-01-01"}`, `{"@duration":"P1D"}`,
        // … — matching core's query-result form (and the NDJSON/elementMap wire form),
        // not a bare ISO string.
        Value::Temporal(t) => {
            out.push_str("{\"@");
            out.push_str(t.tag());
            out.push_str("\":");
            write_string(out, &t.format());
            out.push('}');
        }
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
        // Integer-valued: emit without a decimal point (matches `js_number` for this
        // range but skips the `{:e}` round-trip — a hot-path shortcut).
        let _ = write!(out, "{}", x as i64);
    } else {
        out.push_str(&js_number(x));
    }
}

/// Format `x` exactly as JavaScript's `Number.prototype.toString` does — the fixed /
/// exponential threshold (decimal-point position `n > 21` or `n <= -6` → exponential),
/// and `-0` → `"0"`. Rust's `{}` formats an f64 in decimal at ALL magnitudes
/// (`1e-7` → `0.0000001`, `1e21` → `1000000000000000000000`), which is a different STRING
/// from JS even though it parses to the same f64 — so any number that becomes a STRING
/// (`to_string` / `CAST AS STRING` / `||`) must route through here for byte-identity with
/// core / the TS engine. Ported from lenke-core's `jsonfmt::js_number`. `{:e}` gives the
/// shortest round-tripping mantissa; this just places the decimal point per the spec.
pub(crate) fn js_number(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string(); // also normalizes -0.0 → "0" (JS drops the sign)
    }
    let neg = x < 0.0;
    let sci = format!("{:e}", x.abs()); // e.g. "1.5e21", "1e-7"
    let (mant, exp_str) = sci.split_once('e').expect("{:e} always has an 'e'");
    let exp: i32 = exp_str.parse().expect("valid base-10 exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32; // significant digits
    let n = exp + 1; // ECMA `n`: position of the decimal point

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if k <= n && n <= 21 {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= 21 {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-n) as usize));
        out.push_str(&digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let e = n - 1;
        out.push(if e >= 0 { '+' } else { '-' });
        out.push_str(&e.abs().to_string());
    }
    out
}

/// Stringify a value like JS `String(v)` / core's `js_str` — the coercion behind
/// `to_string` and `CAST(x AS STRING)` for the COMPOSITE types (a scalar's own arm in
/// `to_string_fn` is the fast path). A LIST joins its elements with "," (a null element →
/// "", matching JS `Array.prototype.join`, NOT the top-level "null"); a RECORD/MAP renders
/// as its canonical JSON (byte-identical to how it serializes in a result). Numbers use
/// `js_number`, with non-finite ones spelled "NaN"/"Infinity"/"-Infinity" (JS `String`).
pub(crate) fn js_str(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Num(x) if x.is_nan() => "NaN".to_string(),
        Value::Num(x) if x.is_infinite() => {
            if *x > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
        }
        Value::Num(x) => js_number(*x),
        Value::Str(s) => s.to_string(),
        Value::Temporal(t) => t.format(),
        Value::List(items) => items
            .iter()
            .map(|x| match x {
                Value::Null => String::new(),
                other => js_str(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Record(_) | Value::Map(_) => {
            let mut s = String::new();
            write_value(&mut s, v);
            s
        }
    }
}

/// A JSON string literal (RFC 8259 escaping).
pub(crate) fn write_string(out: &mut String, s: &str) {
    out.push('"');
    // Fast path: if NO byte needs escaping — none is `"`, `\`, or a control byte
    // (< 0x20) — the literal is `s` verbatim, so copy it in one `push_str` (a memcpy)
    // instead of walking `char`s. UTF-8 continuation/lead bytes are all >= 0x80, so a
    // byte scan never false-flags a multi-byte char. Byte-identical to the loop below.
    if !s.bytes().any(|b| b < 0x20 || b == b'"' || b == b'\\') {
        out.push_str(s);
        out.push('"');
        return;
    }
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

    #[test]
    fn js_number_matches_js_tostring() {
        // Byte-for-byte JavaScript `Number.prototype.toString`, incl. the fixed /
        // exponential threshold (n > 21 or n <= -6) and -0 → "0". Same fixture as core.
        let cases: &[(f64, &str)] = &[
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (-1.5, "-1.5"),
            (100.0, "100"),
            (0.5, "0.5"),
            (1234.5, "1234.5"),
            (12300.0, "12300"),
            (0.1, "0.1"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1e20, "100000000000000000000"),
            (1e21, "1e+21"),
            (1.5e21, "1.5e+21"),
            (1e-10, "1e-10"),
            (1e100, "1e+100"),
            (-1e-7, "-1e-7"),
            (1.25, "1.25"),
        ];
        for &(x, want) in cases {
            assert_eq!(js_number(x), want, "js_number({x})");
        }
    }

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
