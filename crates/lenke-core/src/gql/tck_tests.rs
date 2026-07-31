//! TCK (Technology Compatibility Kit) scenarios adapted from the openCypher TCK
//! (github.com/opencypher/openCypher, `tck/features`, Apache License 2.0),
//! re-expressed as Rust unit tests.
//!
//! Ports `packages/gql/src/tck.test.ts`, so a change to either side should change
//! both.
//! Each test is named `tck_<snake_case_description>` mirroring the TS test
//! name. The module is self-contained: helpers and fixtures are defined here.

use super::eval::Params;
use super::parse;
use crate::graph::{Graph, Value};
use crate::ndjson;

// ---------------------------------------------------------------------------
// Helpers (mirrored from tests.rs)
// ---------------------------------------------------------------------------

fn n(x: f64) -> Value {
    Value::Num(x)
}
fn s(x: &str) -> Value {
    Value::Str(x.into())
}
fn b(x: bool) -> Value {
    Value::Bool(x)
}

/// Run a query (no params) and return (columns, rows).
fn q(g: &mut Graph, query: &str) -> (Vec<String>, Vec<Vec<Value>>) {
    let parsed = parse(query).unwrap_or_else(|e| panic!("parse error for `{query}`: {e}"));
    let rs = parsed
        .execute(g, &Params::new())
        .unwrap_or_else(|e| panic!("exec error for `{query}`: {e}"));
    (rs.cols.clone(), rs.rows().map(|r| r.to_vec()).collect())
}

fn qp(g: &mut Graph, query: &str, params: Params) -> Vec<Vec<Value>> {
    parse(query)
        .unwrap()
        .execute(g, &params)
        .unwrap()
        .rows()
        .map(|r| r.to_vec())
        .collect()
}

fn rows(g: &mut Graph, query: &str) -> Vec<Vec<Value>> {
    q(g, query).1
}

/// Build the TinkerPop Modern graph via ndjson (ids are names; labels
/// Person/Software; KNOWS+CREATED edges with `weight`).
/// An empty graph (no nodes, no edges).
fn empty() -> Graph {
    ndjson::decode("").unwrap()
}

// ---------------------------------------------------------------------------
// TCK Boolean1 — AND (three-valued logic)
// ---------------------------------------------------------------------------

#[test]
fn tck_boolean1_and_three_valued() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN true AND true AS tt, true AND false AS tf, true AND null AS tn,
                false AND true AS ft, false AND false AS ff, false AND null AS fn,
                null AND true AS nt, null AND false AS nf, null AND null AS nn",
    );
    // tt  tf     tn          ft     ff     fn     nt          nf     nn
    assert_eq!(
        r,
        vec![vec![
            b(true),
            b(false),
            Value::Null,
            b(false),
            b(false),
            b(false),
            Value::Null,
            b(false),
            Value::Null,
        ]]
    );
}

// ---------------------------------------------------------------------------
// TCK Boolean2 — OR (three-valued logic)
// ---------------------------------------------------------------------------

#[test]
fn tck_boolean2_or_three_valued() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN true OR true AS tt, true OR false AS tf, true OR null AS tn,
                false OR true AS ft, false OR false AS ff, false OR null AS fn,
                null OR true AS nt, null OR false AS nf, null OR null AS nn",
    );
    assert_eq!(
        r,
        vec![vec![
            b(true),
            b(true),
            b(true),
            b(true),
            b(false),
            Value::Null,
            b(true),
            Value::Null,
            Value::Null,
        ]]
    );
}

// ---------------------------------------------------------------------------
// TCK Boolean3 — XOR (three-valued logic)
// ---------------------------------------------------------------------------

#[test]
fn tck_boolean3_xor_three_valued() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN true XOR true AS tt, true XOR false AS tf, true XOR null AS tn,
                false XOR true AS ft, false XOR false AS ff, false XOR null AS fn,
                null XOR true AS nt, null XOR false AS nf, null XOR null AS nn",
    );
    assert_eq!(
        r,
        vec![vec![
            b(false),
            b(true),
            Value::Null,
            b(true),
            b(false),
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
        ]]
    );
}

// ---------------------------------------------------------------------------
// TCK Boolean4 — NOT (three-valued logic)
// ---------------------------------------------------------------------------

#[test]
fn tck_boolean4_not_three_valued() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN NOT true AS nt, NOT false AS nf, NOT null AS nn",
    );
    assert_eq!(r, vec![vec![b(false), b(true), Value::Null]]);
}

// ---------------------------------------------------------------------------
// TCK Null3 — null evaluation, Scenarios [1]–[3]
// ---------------------------------------------------------------------------

#[test]
fn tck_null3_inverse_of_null_is_null() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN NOT null AS val");
    assert_eq!(r, vec![vec![Value::Null]]);
}

#[test]
fn tck_null3_null_equals_null_is_unknown() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN null = null AS val");
    assert_eq!(r, vec![vec![Value::Null]]);
}

#[test]
fn tck_null3_null_not_equals_null_is_unknown() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN null <> null AS val");
    assert_eq!(r, vec![vec![Value::Null]]);
}

// ---------------------------------------------------------------------------
// TCK Null3 — Scenario [4]: IN with null (three-valued)
// ---------------------------------------------------------------------------

#[test]
fn tck_null3_in_with_null_membership_three_valued_logic() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN 1 IN [1, 2] AS present, 3 IN [1, 2] AS absent,
                1 IN [1, null] AS foundDespiteNull, 3 IN [1, null] AS unknownDueToNull,
                null IN [1, 2] AS nullElt, null IN [] AS nullInEmpty, 1 IN [] AS valInEmpty",
    );
    assert_eq!(
        r,
        vec![vec![
            b(true),
            b(false),
            b(true),
            Value::Null,
            Value::Null,
            b(false),
            b(false),
        ]]
    );
}

// ---------------------------------------------------------------------------
// TCK Null1 — IS NULL validation, Scenarios [1]–[3]
// ---------------------------------------------------------------------------

#[test]
fn tck_null1_property_null_check_on_non_null_node() {
    let mut g = empty();
    // The TCK uses `exists` (reserved in ISO GQL); renamed to `present`.
    rows(&mut g, "INSERT ({present: 42})");
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.missing IS NULL AS missingNull, n.present IS NULL AS presentNull",
    );
    assert_eq!(r, vec![vec![b(true), b(false)]]);
}

#[test]
fn tck_null1_property_null_check_on_null_node_unmatched_optional() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "OPTIONAL MATCH (n) RETURN n.missing IS NULL AS missingNull",
    );
    assert_eq!(r, vec![vec![b(true)]]);
}

#[test]
fn tck_null1_literal_null_is_null() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN null IS NULL AS val");
    assert_eq!(r, vec![vec![b(true)]]);
}

// ---------------------------------------------------------------------------
// TCK Aggregation1 — count only non-null values, Scenario [1]
// ---------------------------------------------------------------------------

#[test]
fn tck_aggregation1_count_skips_null() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT ({name: 'a', num: 33}), ({name: 'a'}), ({name: 'b', num: 42})",
    );
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.name AS name, count(n.num) AS c ORDER BY name",
    );
    assert_eq!(r, vec![vec![s("a"), n(1.0)], vec![s("b"), n(1.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Precedence1 — boolean operator precedence
// ---------------------------------------------------------------------------

#[test]
fn tck_precedence1_iso_or_xor_share_left_associative_level() {
    // ISO: OR and XOR share one left-associative level, so
    // `true OR true XOR true` = `(true OR true) XOR true` = false (not true).
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN true OR true XOR true AS a, true OR (true XOR true) AS b, (true OR true) XOR true AS c",
    );
    assert_eq!(r, vec![vec![b(false), b(true), b(false)]]);
}

#[test]
fn tck_precedence1_and_takes_precedence_over_xor() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN true XOR false AND false AS a, true XOR (false AND false) AS b, (true XOR false) AND false AS c",
    );
    assert_eq!(r, vec![vec![b(true), b(true), b(false)]]);
}

#[test]
fn tck_precedence1_and_takes_precedence_over_or() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN true OR false AND false AS a, true OR (false AND false) AS b, (true OR false) AND false AS c",
    );
    assert_eq!(r, vec![vec![b(true), b(true), b(false)]]);
}

#[test]
fn tck_precedence1_not_takes_precedence_over_and() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN NOT true AND false AS a, (NOT true) AND false AS b, NOT (true AND false) AS c",
    );
    assert_eq!(r, vec![vec![b(false), b(false), b(true)]]);
}

#[test]
fn tck_precedence1_not_takes_precedence_over_or() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN NOT false OR true AS a, (NOT false) OR true AS b, NOT (false OR true) AS c",
    );
    assert_eq!(r, vec![vec![b(true), b(true), b(false)]]);
}

#[test]
fn tck_precedence1_comparison_takes_precedence_over_not() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN NOT false >= false AS a, NOT (false >= false) AS b, (NOT false) >= false AS c",
    );
    assert_eq!(r, vec![vec![b(false), b(false), b(true)]]);
}

// ---------------------------------------------------------------------------
// TCK Precedence2 — numeric operator precedence
// ---------------------------------------------------------------------------

#[test]
fn tck_precedence2_multiplication_takes_precedence_over_addition() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN 4 * 2 + 3 * 2 AS a, 4 * 2 + (3 * 2) AS b, 4 * (2 + 3) * 2 AS c",
    );
    assert_eq!(r, vec![vec![n(14.0), n(14.0), n(40.0)]]);
}

#[test]
fn tck_precedence2_unary_minus_takes_precedence_over_addition() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN -3 + 2 AS a, (-3) + 2 AS b, -(3 + 2) AS c");
    assert_eq!(r, vec![vec![n(-1.0), n(-1.0), n(-5.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Mathematical6 — modulo division
// ---------------------------------------------------------------------------

#[test]
fn tck_mathematical6_modulo_of_positive_integers() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN 7 % 3 AS a, 8 % 4 AS b, 5 % 3 AS c");
    assert_eq!(r, vec![vec![n(1.0), n(0.0), n(2.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Aggregation2/3/5 — null handling in aggregates
// ---------------------------------------------------------------------------

#[test]
fn tck_aggregation2_min_max_over_integers_ignore_null() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT ({num: 1}), ({num: 2}), ({num: 0}), ({other: 9}), ({num: -1})",
    );
    let r = rows(
        &mut g,
        "MATCH (n) RETURN max(n.num) AS mx, min(n.num) AS mn",
    );
    assert_eq!(r, vec![vec![n(2.0), n(-1.0)]]);
}

#[test]
fn tck_aggregation3_sum_only_non_null_values() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT ({name: 'a', num: 33}), ({name: 'a'}), ({name: 'a', num: 42})",
    );
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.name AS name, sum(n.num) AS total",
    );
    assert_eq!(r, vec![vec![s("a"), n(75.0)]]);
}

#[test]
fn tck_aggregation5_collect_filters_nulls() {
    let mut g = empty();
    rows(&mut g, "INSERT (:Lonely)");
    let r = rows(
        &mut g,
        "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN collect_list(x) AS xs",
    );
    assert_eq!(r, vec![vec![Value::List(vec![])]]);
}

// ---------------------------------------------------------------------------
// TCK Comparison3 — chained comparison rejected (ISO is binary-only)
// ---------------------------------------------------------------------------

#[test]
fn tck_comparison3_chained_comparison_rejected() {
    // ISO GQL: comparisons are strictly binary; `1 < 2 < 3` is a syntax error.
    let result = parse("RETURN 1 < 2 < 3 AS x");
    assert!(
        result.is_err(),
        "expected syntax error for chained comparison"
    );
}

// ---------------------------------------------------------------------------
// TCK Comparison1 — element identity
// ---------------------------------------------------------------------------

#[test]
fn tck_comparison1_comparing_nodes_to_nodes() {
    let mut g = empty();
    rows(&mut g, "INSERT (:N)");
    let r = rows(
        &mut g,
        "MATCH (a) WITH a MATCH (b) WHERE a = b RETURN count(b) AS c",
    );
    assert_eq!(r, vec![vec![n(1.0)]]);
}

#[test]
fn tck_comparison1_comparing_relationships_to_relationships() {
    let mut g = empty();
    rows(&mut g, "INSERT (:A)-[:T]->(:B)");
    let r = rows(
        &mut g,
        "MATCH ()-[a]->() WITH a MATCH ()-[b]->() WHERE a = b RETURN count(b) AS c",
    );
    assert_eq!(r, vec![vec![n(1.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Union1 / Union2 — DISTINCT union dedups; UNION ALL keeps duplicates
// ---------------------------------------------------------------------------

#[test]
fn tck_union1_two_unique_elements_distinct() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN 1 AS x UNION RETURN 2 AS x");
    assert_eq!(r, vec![vec![n(1.0)], vec![n(2.0)]]);
}

#[test]
fn tck_union1_three_elements_two_unique_distinct() {
    let mut g = empty();
    let r = rows(
        &mut g,
        "RETURN 2 AS x UNION RETURN 1 AS x UNION RETURN 2 AS x",
    );
    assert_eq!(r, vec![vec![n(2.0)], vec![n(1.0)]]);
}

#[test]
fn tck_union2_union_all_keeps_duplicates() {
    let mut g = empty();
    let r = rows(&mut g, "RETURN 1 AS x UNION ALL RETURN 1 AS x");
    assert_eq!(r, vec![vec![n(1.0)], vec![n(1.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Set1 — setting properties
// ---------------------------------------------------------------------------

#[test]
fn tck_set1_set_property_to_literal() {
    let mut g = empty();
    rows(&mut g, "INSERT (:A {name: 'Andres'})");
    rows(
        &mut g,
        "MATCH (n:A) WHERE n.name = 'Andres' SET n.name = 'Michael'",
    );
    let r = rows(&mut g, "MATCH (n:A) RETURN n.name AS name");
    assert_eq!(r, vec![vec![s("Michael")]]);
}

#[test]
fn tck_set1_set_property_to_expression_concat_iso() {
    // ISO GQL uses `||` for string concatenation (not `+` like Cypher).
    let mut g = empty();
    rows(&mut g, "INSERT (:A {name: 'Andres'})");
    rows(
        &mut g,
        "MATCH (n:A) WHERE n.name = 'Andres' SET n.name = n.name || ' was here'",
    );
    let r = rows(&mut g, "MATCH (n:A) RETURN n.name AS name");
    assert_eq!(r, vec![vec![s("Andres was here")]]);
}

// ---------------------------------------------------------------------------
// TCK Remove1 — removing a node property
// ---------------------------------------------------------------------------

#[test]
fn tck_remove1_removed_property_is_no_longer_present() {
    let mut g = empty();
    rows(&mut g, "INSERT (:L {num: 42})");
    let r = rows(
        &mut g,
        "MATCH (n) REMOVE n.num RETURN n.num IS NOT NULL AS stillThere",
    );
    assert_eq!(r, vec![vec![b(false)]]);
}

// ---------------------------------------------------------------------------
// TCK Delete1 — deleting nodes
// ---------------------------------------------------------------------------

#[test]
fn tck_delete1_delete_isolated_node() {
    let mut g = empty();
    rows(&mut g, "INSERT (:Doomed)");
    rows(&mut g, "MATCH (n:Doomed) DELETE n");
    let r = rows(&mut g, "MATCH (n) RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(0.0)]]);
}

#[test]
fn tck_delete1_detach_delete_removes_node_and_relationships() {
    let mut g = empty();
    rows(&mut g, "INSERT (:A)-[:T]->(:B)");
    rows(&mut g, "MATCH (a:A) DETACH DELETE a");
    // A node and T edge gone; only B remains.
    let c = rows(&mut g, "MATCH (n) RETURN count(*) AS c");
    assert_eq!(c, vec![vec![n(1.0)]]);
    let cr = rows(&mut g, "MATCH ()-[r]->() RETURN count(*) AS c");
    assert_eq!(cr, vec![vec![n(0.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Literals1/2/3/4/5/6 — literal evaluation
// ---------------------------------------------------------------------------

#[test]
fn tck_literals_booleans_and_null_case_insensitive() {
    let mut g = empty();
    assert_eq!(rows(&mut g, "RETURN true AS literal"), vec![vec![b(true)]]);
    assert_eq!(rows(&mut g, "RETURN TRUE AS literal"), vec![vec![b(true)]]);
    assert_eq!(
        rows(&mut g, "RETURN false AS literal"),
        vec![vec![b(false)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN FALSE AS literal"),
        vec![vec![b(false)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN null AS literal"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn tck_literals_decimal_integers() {
    let mut g = empty();
    assert_eq!(rows(&mut g, "RETURN 1 AS literal"), vec![vec![n(1.0)]]);
    assert_eq!(rows(&mut g, "RETURN 0 AS literal"), vec![vec![n(0.0)]]);
    assert_eq!(
        rows(&mut g, "RETURN 372036854 AS literal"),
        vec![vec![n(372036854.0)]]
    );
}

#[test]
fn tck_literals_hexadecimal_integers() {
    let mut g = empty();
    assert_eq!(rows(&mut g, "RETURN 0x1 AS literal"), vec![vec![n(1.0)]]);
    assert_eq!(
        rows(&mut g, "RETURN 0x162CD4F6 AS literal"),
        vec![vec![n(372036854.0)]]
    );
}

#[test]
fn tck_literals_octal_integers() {
    let mut g = empty();
    assert_eq!(rows(&mut g, "RETURN 0o1 AS literal"), vec![vec![n(1.0)]]);
    assert_eq!(
        rows(&mut g, "RETURN 0o2613152366 AS literal"),
        vec![vec![n(372036854.0)]]
    );
}

#[test]
fn tck_literals_floats_including_leading_dot_and_ieee754_rounding() {
    let mut g = empty();
    assert_eq!(rows(&mut g, "RETURN 1.0 AS literal"), vec![vec![n(1.0)]]);
    assert_eq!(rows(&mut g, "RETURN .1 AS literal"), vec![vec![n(0.1)]]);
    assert_eq!(
        rows(&mut g, "RETURN .3405892687 AS literal"),
        vec![vec![n(0.3405892687)]]
    );
    // The same double rounding the TCK expects (the last digit rounds to 6).
    assert_eq!(
        rows(&mut g, "RETURN 3985764.3405892687 AS literal"),
        vec![vec![n(3985764.3405892686)]]
    );
}

#[test]
fn tck_literals_strings() {
    let mut g = empty();
    assert_eq!(rows(&mut g, "RETURN '' AS literal"), vec![vec![s("")]]);
    assert_eq!(rows(&mut g, "RETURN 'a' AS literal"), vec![vec![s("a")]]);
}

// ---------------------------------------------------------------------------
// TCK Literals6 — string escape sequences
// ---------------------------------------------------------------------------

#[test]
fn tck_literals6_control_character_escapes_decode() {
    let mut g = empty();
    assert_eq!(rows(&mut g, r"RETURN '\n' AS literal"), vec![vec![s("\n")]]);
    assert_eq!(rows(&mut g, r"RETURN '\t' AS literal"), vec![vec![s("\t")]]);
    assert_eq!(rows(&mut g, r"RETURN '\r' AS literal"), vec![vec![s("\r")]]);
}

#[test]
fn tck_literals6_escaped_backslash_and_quotes_decode() {
    let mut g = empty();
    assert_eq!(rows(&mut g, r"RETURN '\\' AS literal"), vec![vec![s("\\")]]);
    assert_eq!(rows(&mut g, r"RETURN '\'' AS literal"), vec![vec![s("'")]]);
    assert_eq!(rows(&mut g, "RETURN '\"' AS literal"), vec![vec![s("\"")]]);
}

#[test]
fn tck_literals6_unicode_escapes_decode_to_code_points() {
    let mut g = empty();
    // \uXXXX — 4-digit unicode escape
    assert_eq!(
        rows(&mut g, "RETURN '\\u0041' AS literal"),
        vec![vec![s("A")]]
    );
    assert_eq!(
        rows(&mut g, "RETURN '\\u01FF' AS literal"),
        vec![vec![s("ǿ")]]
    );
    // \UXXXXXX — 6-digit unicode escape (emoji)
    assert_eq!(
        rows(&mut g, r"RETURN '\U01F600' AS literal"),
        vec![vec![s("😀")]]
    );
}

#[test]
fn tck_literals6_malformed_unicode_escape_is_syntax_error() {
    let result = parse(r"RETURN '\uH' AS x");
    assert!(result.is_err(), "expected syntax error for malformed \\uH");
}

// ---------------------------------------------------------------------------
// TCK ReturnSkipLimit2/3 — LIMIT and SKIP
// ---------------------------------------------------------------------------

fn skip_limit_seed() -> Graph {
    let mut g = ndjson::decode("").unwrap();
    rows(
        &mut g,
        "INSERT ({name:'A'}), ({name:'B'}), ({name:'C'}), ({name:'D'}), ({name:'E'})",
    );
    g
}

#[test]
fn tck_return_skip_limit_order_by_then_limit_2_keeps_first_two() {
    let mut g = skip_limit_seed();
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.name AS name ORDER BY n.name ASC LIMIT 2",
    );
    assert_eq!(r, vec![vec![s("A")], vec![s("B")]]);
}

#[test]
fn tck_return_skip_limit_limit_0_returns_no_rows() {
    let mut g = skip_limit_seed();
    let r = rows(&mut g, "MATCH (n) RETURN n.name AS name LIMIT 0");
    assert_eq!(r, vec![] as Vec<Vec<Value>>);
}

#[test]
fn tck_return_skip_limit_skip_then_limit_pages_through_ordering() {
    let mut g = skip_limit_seed();
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.name AS name ORDER BY n.name ASC SKIP 2 LIMIT 2",
    );
    assert_eq!(r, vec![vec![s("C")], vec![s("D")]]);
}

#[test]
fn tck_return_skip_limit_skip_past_end_yields_no_rows() {
    let mut g = skip_limit_seed();
    let r = rows(&mut g, "MATCH (n) RETURN n.name AS name SKIP 99");
    assert_eq!(r, vec![] as Vec<Vec<Value>>);
}

// ---------------------------------------------------------------------------
// TCK ReturnOrderBy6 — aggregation expressions inside ORDER BY
// ---------------------------------------------------------------------------

#[test]
fn tck_return_order_by6_empty_match_avg_is_null_order_by_aggregate_param() {
    let mut g = empty();
    let mut params = Params::new();
    params.insert("age".to_string(), super::eval::Val::Num(38.0));
    let r = qp(
        &mut g,
        "MATCH (person) RETURN avg(person.age) AS avgAge ORDER BY $age + avg(person.age) - 1000",
        params,
    );
    // Empty match but aggregate projection still yields one row.
    assert_eq!(r, vec![vec![Value::Null]]);
}

#[test]
fn tck_return_order_by6_order_by_aggregate_sorts_groups() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT ({city:'London'}), ({city:'London'}), ({city:'London'}),
                ({city:'Paris'}), ({city:'Berlin'}), ({city:'Berlin'})",
    );
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.city AS city, count(*) AS cnt ORDER BY count(*) DESC, city",
    );
    assert_eq!(
        r,
        vec![
            vec![s("London"), n(3.0)],
            vec![s("Berlin"), n(2.0)],
            vec![s("Paris"), n(1.0)],
        ]
    );
}

// ---------------------------------------------------------------------------
// TCK WithWhere1/3 — WITH then WHERE
// ---------------------------------------------------------------------------

#[test]
fn tck_with_where1_where_after_with_filters_carried_rows() {
    let mut g = empty();
    rows(&mut g, "INSERT ({name: 'A'}), ({name: 'B'}), ({name: 'C'})");
    let r = rows(
        &mut g,
        "MATCH (a) WITH a WHERE a.name = 'B' RETURN a.name AS name",
    );
    assert_eq!(r, vec![vec![s("B")]]);
}

#[test]
fn tck_with_where3_cartesian_self_join_filtered_by_identity() {
    let mut g = empty();
    rows(&mut g, "INSERT ({k: 'A'}), ({k: 'B'})");
    let r = rows(
        &mut g,
        "MATCH (a), (b) WITH a, b WHERE a = b RETURN a.k AS ak, b.k AS bk ORDER BY ak",
    );
    assert_eq!(r, vec![vec![s("A"), s("A")], vec![s("B"), s("B")]]);
}

// ---------------------------------------------------------------------------
// TCK MatchWhere2 / With2 — multi-variable joins
// ---------------------------------------------------------------------------

#[test]
fn tck_match_where2_undirected_4_cycle_with_chord_filtered_on_two_variables() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (a:A {n: 'a'}), (b:B {n: 'b', id: 1}), (c:C {n: 'c', id: 2}), (d:D {n: 'd'})",
    );
    rows(
        &mut g,
        "MATCH (a:A), (b:B), (c:C), (d:D)
         INSERT (a)-[:T]->(b), (a)-[:T]->(c), (a)-[:T]->(d),
                (b)-[:T]->(c), (b)-[:T]->(d), (c)-[:T]->(d)",
    );
    let result = rows(
        &mut g,
        "MATCH (a)~(b)~(c)~(d)~(a), (b)~(d) WHERE a.id = 1 AND c.id = 2 RETURN d.n AS dn ORDER BY dn",
    );
    assert_eq!(result, vec![vec![s("a")], vec![s("d")]]);
}

#[test]
fn tck_with2_with_forwards_property_as_join_key() {
    // Intra-INSERT forward reference: the third node's `num: a.id` reads sibling
    // node `a` created earlier in the same INSERT (sequential per-element eval).
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (a:Sink {num: 42, id: 0}), (:Sink {num: 3}), (:Source {num: a.id})",
    );
    let result = rows(
        &mut g,
        "MATCH (a:Source) WITH a.num AS property MATCH (b) WHERE b.id = property RETURN b.num AS num",
    );
    assert_eq!(result, vec![vec![n(42.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Mathematical3/4/5 — arithmetic with null propagation
// ---------------------------------------------------------------------------

#[test]
fn tck_mathematical_subtraction_multiplication_division() {
    let mut g = empty();
    assert_eq!(rows(&mut g, "RETURN 7 - 2 AS r"), vec![vec![n(5.0)]]);
    assert_eq!(rows(&mut g, "RETURN 3 * 4 AS r"), vec![vec![n(12.0)]]);
    assert_eq!(rows(&mut g, "RETURN 6 / 2 AS r"), vec![vec![n(3.0)]]);
}

#[test]
fn tck_mathematical_any_null_operand_yields_null() {
    let mut g = empty();
    assert_eq!(
        rows(&mut g, "RETURN null - 1 AS r"),
        vec![vec![Value::Null]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 2 - null AS r"),
        vec![vec![Value::Null]]
    );
    assert_eq!(
        rows(&mut g, "RETURN null * 3 AS r"),
        vec![vec![Value::Null]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 5 / null AS r"),
        vec![vec![Value::Null]]
    );
}

// ---------------------------------------------------------------------------
// TCK MatchWhere3 / With1 / ReturnOrderBy5
// ---------------------------------------------------------------------------

#[test]
fn tck_match_where3_equi_join_on_properties_of_disconnected_nodes() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (:A {id: 1}), (:A {id: 2}), (:B {id: 2}), (:B {id: 3})",
    );
    let r = rows(
        &mut g,
        "MATCH (a:A), (b:B) WHERE a.id = b.id RETURN a.id AS aid, b.id AS bid",
    );
    assert_eq!(r, vec![vec![n(2.0), n(2.0)]]);
}

#[test]
fn tck_with1_with_forwards_node_variable_into_next_match() {
    let mut g = empty();
    rows(&mut g, "INSERT (:A {n: 'a'})-[:REL]->(:B {n: 'b'})");
    let r = rows(
        &mut g,
        "MATCH (a:A) WITH a MATCH (a)->(b) RETURN a.n AS an, b.n AS bn",
    );
    assert_eq!(r, vec![vec![s("a"), s("b")]]);
}

#[test]
fn tck_return_order_by5_order_by_expression_over_renamed_column() {
    let mut g = empty();
    rows(&mut g, "INSERT ({num: 1}), ({num: 3}), ({num: -5})");
    // `n` is the output alias for `n.num`; the sort key `n + 2` uses it.
    // Order by: -5+2=-3, 1+2=3, 3+2=5 ⇒ ascending by expression → [-5, 1, 3].
    let r = rows(&mut g, "MATCH (n) RETURN n.num AS n ORDER BY n + 2");
    let vals: Vec<Value> = r
        .into_iter()
        .map(|row| row.into_iter().next().unwrap())
        .collect();
    assert_eq!(vals, vec![n(-5.0), n(1.0), n(3.0)]);
}

// ---------------------------------------------------------------------------
// TCK Match1 — matching nodes
// ---------------------------------------------------------------------------

#[test]
fn tck_match1_matching_non_existent_nodes_returns_empty() {
    let mut g = empty();
    let r = rows(&mut g, "MATCH (n) RETURN n.x AS x");
    assert_eq!(r, vec![] as Vec<Vec<Value>>);
}

#[test]
fn tck_match1_matching_all_nodes_regardless_of_label() {
    let mut g = empty();
    rows(&mut g, "INSERT (:A), (:B {name: 'b'}), ({name: 'c'})");
    let r = rows(&mut g, "MATCH (n) RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(3.0)]]);
}

#[test]
fn tck_match1_matching_by_conjunctive_label_expression() {
    // Cypher's `:A:B` → ISO `A&B`. Nodes with BOTH A and B: (:A&B&C) and (:A&B).
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (:A&B&C), (:A&B), (:A&C), (:B&C), (:A), (:B), (:C), ({name: ':A:B:C'}), ({abc: 'abc'}), ()",
    );
    let r = rows(&mut g, "MATCH (a:A&B) RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(2.0)]]);
}

// ---------------------------------------------------------------------------
// TCK Match2 / Match3 — relationships and fixed-length patterns
// ---------------------------------------------------------------------------

#[test]
fn tck_match2_matching_non_existent_relationships_returns_empty() {
    let mut g = empty();
    let r = rows(&mut g, "MATCH ()-[r]->() RETURN r.x AS x");
    assert_eq!(r, vec![] as Vec<Vec<Value>>);
}

#[test]
fn tck_match2_relationship_pattern_with_label_predicate_on_both_sides() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (:A)-[:T1 {id: 1}]->(:B), (:B)-[:T2 {id: 2}]->(:A),
                (:B)-[:T3 {id: 3}]->(:B), (:A)-[:T4 {id: 4}]->(:A)",
    );
    // Only the A→B edge (T1) satisfies both endpoint labels.
    let r = rows(&mut g, "MATCH (:A)-[r]->(:B) RETURN r.id AS id");
    assert_eq!(r, vec![vec![n(1.0)]]);
}

#[test]
fn tck_match3_get_neighbours_across_typed_relationship() {
    let mut g = empty();
    rows(&mut g, "INSERT (:A {num: 1})-[:KNOWS]->(:B {num: 2})");
    let r = rows(
        &mut g,
        "MATCH (n1)-[rel:KNOWS]->(n2) RETURN n1.num AS a, n2.num AS b",
    );
    assert_eq!(r, vec![vec![n(1.0), n(2.0)]]);
}

#[test]
fn tck_match3_directed_match_binds_source_relationship_and_target() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (:A {k: 'a'})-[:LOOP {k: 'r'}]->(:B {k: 'b'})",
    );
    let r = rows(
        &mut g,
        "MATCH (a)-[r]->(b) RETURN a.k AS a, r.k AS r, b.k AS b",
    );
    assert_eq!(r, vec![vec![s("a"), s("r"), s("b")]]);
}

#[test]
fn tck_match3_undirected_match_yields_both_orientations() {
    let mut g = empty();
    rows(&mut g, "INSERT (:A {n: 'a'})-[:LOOP]->(:B {n: 'b'})");
    let r = rows(
        &mut g,
        "MATCH (a)-[r]-(b) RETURN a.n AS a, b.n AS b ORDER BY a",
    );
    assert_eq!(r, vec![vec![s("a"), s("b")], vec![s("b"), s("a")]]);
}

#[test]
fn tck_match3_get_targets_of_typed_relationship_from_one_source() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (a:A {num: 1}), (a)-[:KNOWS]->(:B {num: 2}), (a)-[:KNOWS]->(:C {num: 3})",
    );
    let r = rows(
        &mut g,
        "MATCH ()-[rel:KNOWS]->(x) RETURN x.num AS num ORDER BY num",
    );
    assert_eq!(r, vec![vec![n(2.0)], vec![n(3.0)]]);
}

// ---------------------------------------------------------------------------
// TCK ReturnOrderBy3 — sort on an aggregate then a property
// ---------------------------------------------------------------------------

#[test]
fn tck_return_order_by3_count_star_desc_then_division_asc_tie_break() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT ({division: 'Sweden'}), ({division: 'Germany'}),
                ({division: 'England'}), ({division: 'Sweden'})",
    );
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt
         ORDER BY count(*) DESC, division ASC",
    );
    assert_eq!(
        r,
        vec![
            vec![s("Sweden"), n(2.0)],
            vec![s("England"), n(1.0)],
            vec![s("Germany"), n(1.0)],
        ]
    );
}

// ---------------------------------------------------------------------------
// TCK With4 / With6 / With7 — aliasing, grouping, chaining
// ---------------------------------------------------------------------------

#[test]
fn tck_with6_implicit_grouping_in_with() {
    let mut g = empty();
    rows(&mut g, "INSERT ({name: 'A'}), ({name: 'A'}), ({name: 'B'})");
    let r = rows(
        &mut g,
        "MATCH (a) WITH a.name AS name, count(*) AS relCount RETURN name, relCount ORDER BY name",
    );
    assert_eq!(r, vec![vec![s("A"), n(2.0)], vec![s("B"), n(1.0)]]);
}

#[test]
fn tck_with4_aliasing_relationship_variable_through_with() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT ()-[:T1 {n: 't1'}]->(), ()-[:T2 {n: 't2'}]->()",
    );
    let r = rows(
        &mut g,
        "MATCH ()-[r1]->() WITH r1 AS r2 RETURN r2.n AS n ORDER BY n",
    );
    assert_eq!(r, vec![vec![s("t1")], vec![s("t2")]]);
}

#[test]
fn tck_with7_with_on_with_swapping_variable_names_then_rematch() {
    let mut g = empty();
    rows(
        &mut g,
        "INSERT (:A {k: 'a'})-[:REL {k: 'r'}]->(:B {k: 'b'})",
    );
    let r = rows(
        &mut g,
        "MATCH (a:A)-[r:REL]->(b:B)
         WITH a AS b, b AS tmp, r AS r
         WITH b AS a, r LIMIT 1
         MATCH (a)-[r]->(b)
         RETURN a.k AS a, r.k AS r, b.k AS b",
    );
    assert_eq!(r, vec![vec![s("a"), s("r"), s("b")]]);
}
