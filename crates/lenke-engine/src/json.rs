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

// Pure formatters moved to lenke-engine-core; re-exported so `crate::json::<fn>`
// paths are unchanged.
pub(crate) use lenke_engine_core::json_fmt::{js_number, js_str, write_string, write_value};
// write_number is only referenced by tests; gate the re-export so the lib build
// doesn't see it as unused.
#[cfg(test)]
pub(crate) use lenke_engine_core::json_fmt::write_number;

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
