//! Behavioral-parity port of `packages/gql/src/hardening.test.ts` into Rust.
//!
//! Each test is named `h_<snake_case>` and maps 1-to-1 to a TS test.
//! Tests that diverge from the Rust engine behaviour are commented out with
//! `// SKIPPED (divergence): ...` or `// SKIPPED (unsupported): ...`.

use super::eval::Params;
use super::{parse, parse_with_max_chain};
use crate::error_codes::ErrorCode;
use crate::graph::{Graph, Value};
use crate::ndjson;

// ---------------------------------------------------------------------------
// Fixtures & helpers (self-contained, mirroring tests.rs conventions)
// ---------------------------------------------------------------------------

/// The TinkerPop "Modern" graph — 4 Person, 2 Software, KNOWS + CREATED edges
/// with `weight` properties. Identical to `tests::modern()`.
fn modern() -> Graph {
    let lines = [
        r#"{"type":"node","id":"marko","labels":["Person"],"properties":{"name":"marko","age":29}}"#,
        r#"{"type":"node","id":"vadas","labels":["Person"],"properties":{"name":"vadas","age":27}}"#,
        r#"{"type":"node","id":"josh","labels":["Person"],"properties":{"name":"josh","age":32}}"#,
        r#"{"type":"node","id":"peter","labels":["Person"],"properties":{"name":"peter","age":35}}"#,
        r#"{"type":"node","id":"lop","labels":["Software"],"properties":{"name":"lop","lang":"java"}}"#,
        r#"{"type":"node","id":"ripple","labels":["Software"],"properties":{"name":"ripple","lang":"java"}}"#,
        r#"{"type":"edge","from":"marko","to":"vadas","labels":["KNOWS"],"properties":{"weight":0.5}}"#,
        r#"{"type":"edge","from":"marko","to":"josh","labels":["KNOWS"],"properties":{"weight":1.0}}"#,
        r#"{"type":"edge","from":"marko","to":"lop","labels":["CREATED"],"properties":{"weight":0.4}}"#,
        r#"{"type":"edge","from":"josh","to":"ripple","labels":["CREATED"],"properties":{"weight":1.0}}"#,
        r#"{"type":"edge","from":"josh","to":"lop","labels":["CREATED"],"properties":{"weight":0.4}}"#,
        r#"{"type":"edge","from":"peter","to":"lop","labels":["CREATED"],"properties":{"weight":0.2}}"#,
    ];
    ndjson::decode(&lines.join("\n")).unwrap()
}

fn n(x: f64) -> Value {
    Value::Num(x)
}
fn s(x: &str) -> Value {
    Value::Str(x.into())
}

fn q(g: &mut Graph, query: &str) -> (Vec<String>, Vec<Vec<Value>>) {
    let parsed = parse(query).unwrap_or_else(|e| panic!("parse error for `{query}`: {e}"));
    let rs = parsed
        .execute(g, &Params::new())
        .unwrap_or_else(|e| panic!("exec error for `{query}`: {e}"));
    (rs.cols.clone(), rs.rows().map(|r| r.to_vec()).collect())
}

fn rows(g: &mut Graph, query: &str) -> Vec<Vec<Value>> {
    q(g, query).1
}

// ---------------------------------------------------------------------------
// § 1 — SET / REMOVE keep the property index consistent
// TS: describe('hardening: SET/REMOVE maintain the property index', ...)
// ---------------------------------------------------------------------------

/// TS: "SET reindexes, so an indexed seek finds the new value"
#[test]
fn h_set_reindexes_indexed_seek_finds_new_value() {
    let mut plain = modern();
    let mut indexed = modern();
    indexed.create_vertex_index("age");

    for g in [&mut plain, &mut indexed] {
        rows(g, "MATCH (n:Person {name: 'marko'}) SET n.age = 31");
    }

    let qry = "MATCH (n:Person) WHERE n.age = 31 RETURN n.name";
    let plain_res = rows(&mut plain, qry);
    let idx_res = rows(&mut indexed, qry);
    assert_eq!(idx_res, vec![vec![s("marko")]]);
    assert_eq!(idx_res, plain_res);
}

/// TS: "SET reindexes, so the old value no longer seeks the node"
#[test]
fn h_set_reindexes_old_value_no_longer_seeks() {
    let mut indexed = modern();
    indexed.create_vertex_index("age");
    rows(
        &mut indexed,
        "MATCH (n:Person {name: 'marko'}) SET n.age = 31",
    );

    let result = rows(
        &mut indexed,
        "MATCH (n:Person) WHERE n.age = 29 RETURN n.name",
    );
    assert!(result.is_empty(), "old age=29 should no longer find marko");
}

/// TS: "REMOVE reindexes (indexed and unindexed agree)"
#[test]
fn h_remove_reindexes_indexed_and_unindexed_agree() {
    let mut plain = modern();
    let mut indexed = modern();
    indexed.create_vertex_index("age");

    for g in [&mut plain, &mut indexed] {
        rows(g, "MATCH (n:Person {name: 'marko'}) SET n.age = 100");
        rows(g, "MATCH (n:Person {name: 'marko'}) REMOVE n.age");
    }

    let qry = "MATCH (n:Person) WHERE n.age = 100 RETURN n.name";
    assert!(rows(&mut indexed, qry).is_empty());
    assert_eq!(rows(&mut indexed, qry), rows(&mut plain, qry));
}

// ---------------------------------------------------------------------------
// § 2 — Parser recursion-depth guard
// TS: describe('hardening: deep nesting is a syntax error, not a stack overflow', ...)
// ---------------------------------------------------------------------------

/// TS: "nested parentheses"
///
/// The TS checks `hasErrorCode(e, ErrorCode.Syntax)`; in Rust, `parse()` returns
/// `Result<_, SyntaxError>` and `SyntaxError` has no `.code` field (its type IS
/// the code). An `Err` from `parse()` is always a syntax error.
#[test]
fn h_deep_nested_parens_is_syntax_error() {
    let deep = format!("RETURN {}1{} AS r", "(".repeat(5000), ")".repeat(5000));
    assert!(
        parse(&deep).is_err(),
        "expected parse error for deeply nested parens"
    );
}

/// TS: "nested NOT"
#[test]
fn h_deep_nested_not_is_syntax_error() {
    let deep = format!("MATCH (n) WHERE {}n.x RETURN n", "NOT ".repeat(5000));
    assert!(
        parse(&deep).is_err(),
        "expected parse error for deeply nested NOT"
    );
}

/// TS: "nested label negation"
#[test]
fn h_deep_nested_label_negation_is_syntax_error() {
    let deep = format!("MATCH (n:{}A) RETURN n", "!".repeat(5000));
    assert!(
        parse(&deep).is_err(),
        "expected parse error for deeply nested label negation"
    );
}

/// TS: "nested lists"
#[test]
fn h_deep_nested_lists_is_syntax_error() {
    let deep = format!("RETURN {}1{} AS r", "[".repeat(5000), "]".repeat(5000));
    assert!(
        parse(&deep).is_err(),
        "expected parse error for deeply nested lists"
    );
}

/// TS: "a normally-nested query still parses"
#[test]
fn h_normally_nested_query_still_parses() {
    assert!(parse("RETURN (((1 + 2)) * 3) AS r").is_ok());
}

/// The associative operator AST is n-ary (a flat `Vec`), so a long
/// left-associative chain far past the old crash point (~40k) parses AND
/// *evaluates* — no stack overflow — instead of aborting the process (given a
/// ceiling that admits it). Regression test for the native SIGSEGV on
/// `RETURN true AND true AND … (100k)`.
#[test]
fn h_long_chain_evaluates_without_stack_overflow() {
    let val = |q: &str| -> Value {
        let mut g = ndjson::decode("").unwrap();
        let rs = parse_with_max_chain(q, 200_000)
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        let rows: Vec<Vec<Value>> = rs.rows().map(|r| r.to_vec()).collect();
        rows[0][0].clone()
    };
    assert_eq!(
        val(&format!(
            "RETURN {} AS r",
            vec!["true"; 50_000].join(" AND ")
        )),
        Value::Bool(true)
    );
    assert_eq!(
        val(&format!("RETURN {} AS s", vec!["1"; 50_000].join(" + "))),
        Value::Num(50_000.0)
    );
}

/// Past the anti-resource-abuse ceiling (default 10k), an operator chain is a
/// clean `SyntaxError` (identical in both engines), never an OOM.
#[test]
fn h_over_cap_operator_chain_is_syntax_error() {
    for chain in [
        vec!["true"; 10_002].join(" AND "), // 10_001 ops > DEFAULT_MAX_CHAIN
        vec!["1"; 10_002].join(" + "),
    ] {
        assert!(parse(&format!("RETURN {chain} AS r")).is_err());
    }
}

/// A chain at the default ceiling still parses (10_000 ops); the ceiling is
/// configurable per parse via [`parse_with_max_chain`].
#[test]
fn h_operator_chain_ceiling_is_configurable() {
    // default: 10_000 ops ok, 10_001 rejected
    assert!(parse(&format!(
        "RETURN {} AS r",
        vec!["true"; 10_001].join(" AND ")
    ))
    .is_ok());
    assert!(parse(&format!(
        "RETURN {} AS r",
        vec!["true"; 10_002].join(" AND ")
    ))
    .is_err());
    // a lower configured ceiling rejects sooner; a higher one admits more
    let five = format!("RETURN {} AS r", ["true"; 7].join(" AND ")); // 6 ops
    assert!(parse_with_max_chain(&five, 5).is_err());
    assert!(parse_with_max_chain(&five, 100).is_ok());
}

// ---------------------------------------------------------------------------
// § 3 — Lexer numeric-literal validation
// TS: describe('hardening: malformed numeric literals are rejected', ...)
// ---------------------------------------------------------------------------

/// TS: rejects '0x'
#[test]
fn h_rejects_literal_0x() {
    assert!(parse("RETURN 0x AS r").is_err());
}

/// TS: rejects '0b'
#[test]
fn h_rejects_literal_0b() {
    assert!(parse("RETURN 0b AS r").is_err());
}

/// TS: rejects '0o'
#[test]
fn h_rejects_literal_0o() {
    assert!(parse("RETURN 0o AS r").is_err());
}

/// TS: rejects '0b2'
#[test]
fn h_rejects_literal_0b2() {
    assert!(parse("RETURN 0b2 AS r").is_err());
}

/// TS: rejects '0o8'
#[test]
fn h_rejects_literal_0o8() {
    assert!(parse("RETURN 0o8 AS r").is_err());
}

/// TS: rejects '0o9'
#[test]
fn h_rejects_literal_0o9() {
    assert!(parse("RETURN 0o9 AS r").is_err());
}

/// TS: rejects '1e'
#[test]
fn h_rejects_literal_1e() {
    assert!(parse("RETURN 1e AS r").is_err());
}

/// TS: rejects '1e+'
#[test]
fn h_rejects_literal_1e_plus() {
    assert!(parse("RETURN 1e+ AS r").is_err());
}

/// TS: rejects '0xG'
#[test]
fn h_rejects_literal_0xg() {
    assert!(parse("RETURN 0xG AS r").is_err());
}

/// TS: "rejects an overflowing exponent (Infinity)"
#[test]
fn h_rejects_overflowing_exponent() {
    assert!(parse("RETURN 1e999 AS r").is_err());
}

/// TS: "rejects an integer beyond the safe range"
#[test]
fn h_rejects_oversized_integer_literal() {
    assert!(parse("RETURN 99999999999999999999 AS r").is_err());
}

/// TS: "still accepts valid integers, bases, and floats" (the litValue assertions)
///
/// The TS uses `litValue()` which inspects the AST literal directly.
/// In Rust we parse and execute to verify the value; the semantic is the same.
#[test]
fn h_valid_numeric_literals_accepted() {
    let mut g = modern();
    assert_eq!(rows(&mut g, "RETURN 0 AS r"), vec![vec![n(0.0)]]);
    assert_eq!(rows(&mut g, "RETURN 255 AS r"), vec![vec![n(255.0)]]);
    assert_eq!(rows(&mut g, "RETURN 0xFF AS r"), vec![vec![n(255.0)]]);
    assert_eq!(rows(&mut g, "RETURN 0o17 AS r"), vec![vec![n(15.0)]]);
    assert_eq!(rows(&mut g, "RETURN 0b101 AS r"), vec![vec![n(5.0)]]);
    assert_eq!(rows(&mut g, "RETURN 1_000 AS r"), vec![vec![n(1000.0)]]);
    assert_eq!(rows(&mut g, "RETURN 1.5 AS r"), vec![vec![n(1.5)]]);
    assert_eq!(rows(&mut g, "RETURN 1.5e2 AS r"), vec![vec![n(150.0)]]);
    assert_eq!(rows(&mut g, "RETURN .5 AS r"), vec![vec![n(0.5)]]);
}

// ---------------------------------------------------------------------------
// § 4 — SKIP / LIMIT / quantifier integer validation
// TS: describe('hardening: SKIP/LIMIT/quantifier require non-negative integers', ...)
// ---------------------------------------------------------------------------

/// TS: rejects 'LIMIT 2.5'
#[test]
fn h_rejects_limit_fractional_2_5() {
    assert!(parse("MATCH (n) RETURN n LIMIT 2.5").is_err());
}

/// TS: rejects 'SKIP 1.5'
#[test]
fn h_rejects_skip_fractional_1_5() {
    assert!(parse("MATCH (n) RETURN n SKIP 1.5").is_err());
}

/// TS: rejects 'LIMIT 0.5'
#[test]
fn h_rejects_limit_fractional_0_5() {
    assert!(parse("MATCH (n) RETURN n LIMIT 0.5").is_err());
}

/// TS: "rejects a fractional quantifier bound"
#[test]
fn h_rejects_fractional_quantifier_bound() {
    assert!(parse("MATCH (a)-[:R]->{1.5}(b) RETURN b").is_err());
}

/// TS: "rejects a quantifier whose upper bound is below its lower bound"
#[test]
fn h_rejects_quantifier_reversed_bounds() {
    assert!(parse("MATCH (a)-[:R]->{3,2}(b) RETURN b").is_err());
}

/// TS: "still accepts valid integer bounds"
#[test]
fn h_valid_skip_limit_quantifier_bounds_accepted() {
    assert!(parse("MATCH (n) RETURN n SKIP 1 LIMIT 2").is_ok());
    assert!(parse("MATCH (a)-[:R]->{1,3}(b) RETURN b").is_ok());
    assert!(parse("MATCH (a)-[:R]->{2}(b) RETURN b").is_ok());
}

// ---------------------------------------------------------------------------
// § 5 — plain DELETE must not orphan relationships
// TS: describe('hardening: DELETE vs DETACH DELETE', ...)
// ---------------------------------------------------------------------------

/// TS: "plain DELETE of a connected node throws and leaves the graph intact"
#[test]
fn h_plain_delete_connected_node_errors() {
    let mut g = modern();
    let err = parse("MATCH (n:Person {name: 'marko'}) DELETE n")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidGraphOp);
    // Graph is intact: still 4 persons
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN count(*) AS c"),
        vec![vec![n(4.0)]]
    );
}

/// TS: "plain DELETE of an isolated node succeeds"
#[test]
fn h_plain_delete_isolated_node_succeeds() {
    let mut g = modern();
    rows(&mut g, "INSERT (x:Loner {name: 'solo'})");
    rows(&mut g, "MATCH (n:Loner) DELETE n");
    assert_eq!(
        rows(&mut g, "MATCH (n:Loner) RETURN count(*) AS c"),
        vec![vec![n(0.0)]]
    );
}

/// TS: "DETACH DELETE still cascades incident edges"
#[test]
fn h_detach_delete_cascades_edges() {
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name: 'marko'}) DETACH DELETE n");
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN count(*) AS c"),
        vec![vec![n(3.0)]]
    );
}

// ---------------------------------------------------------------------------
// § 6 — variable-length segments can't bind an edge or carry a predicate
// TS: describe('hardening: variable-length segment restrictions', ...)
// ---------------------------------------------------------------------------

/// TS: "accepts a bound edge variable on a quantified segment (per-hop scope)"
///
/// The edge variable names each hop's edge in turn for the per-hop predicate; it
/// is not yet a group/list variable exposed to the outer query.
#[test]
fn h_accepts_edge_var_on_quantified_segment() {
    assert!(
        parse("MATCH (a)-[r:KNOWS]->*(b) RETURN b").is_ok(),
        "a per-hop edge variable on a quantified segment now parses"
    );
}

/// TS: "accepts a per-edge property predicate on a quantified segment"
#[test]
fn h_accepts_per_edge_predicate_on_quantified_segment() {
    assert!(
        parse("MATCH (a)-[:KNOWS {weight: 1}]->+(b) RETURN b").is_ok(),
        "a per-hop edge property predicate on a quantified segment now parses"
    );
}

/// TS: "a plain quantified segment (label only) still works"
#[test]
fn h_plain_quantified_segment_label_only_works() {
    assert!(parse("MATCH (a:Person {name: 'marko'})-[:KNOWS]->+(b) RETURN b.name").is_ok());
}

// ---------------------------------------------------------------------------
// § 7 — undirected traversal counts a self-loop once
// TS: describe('hardening: self-loop adjacency', ...)
// ---------------------------------------------------------------------------

fn loop_graph() -> Graph {
    let lines = [
        r#"{"type":"node","id":"n","labels":["N"],"properties":{"name":"n"}}"#,
        r#"{"type":"edge","from":"n","to":"n","labels":["LOOP"],"properties":{}}"#,
    ];
    ndjson::decode(&lines.join("\n")).unwrap()
}

/// TS: "an undirected (~) walk yields a self-loop once, not twice"
#[test]
fn h_undirected_self_loop_counted_once() {
    let mut g = loop_graph();
    assert_eq!(
        rows(&mut g, "MATCH (a)~[r]~(b) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
}

/// TS: "directed walks each yield the self-loop once"
#[test]
fn h_directed_self_loop_counted_once() {
    let mut g = loop_graph();
    assert_eq!(
        rows(&mut g, "MATCH (a)-[r]->(b) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (a)<-[r]-(b) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
}

// ---------------------------------------------------------------------------
// § ISO data exceptions in arithmetic
// TS: describe('hardening: ISO data exceptions in arithmetic', ...)
// ---------------------------------------------------------------------------

/// TS: "division by zero raises: RETURN 1 / 0 AS r"
#[test]
fn h_div_by_zero_integer_raises_data_exception() {
    let mut g = modern();
    let err = parse("RETURN 1 / 0 AS r")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::DataException);
}

/// TS: "division by zero raises: RETURN 5 % 0 AS r"
#[test]
fn h_mod_by_zero_raises_data_exception() {
    let mut g = modern();
    let err = parse("RETURN 5 % 0 AS r")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::DataException);
}

/// TS: "division by zero raises: RETURN 1.0 / 0 AS r"
#[test]
fn h_float_div_by_zero_raises_data_exception() {
    let mut g = modern();
    let err = parse("RETURN 1.0 / 0 AS r")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::DataException);
}

/// TS: "non-numeric operand raises: RETURN 'abc' + 1 AS r"
#[test]
fn h_string_plus_number_raises_data_exception() {
    let mut g = modern();
    let err = parse("RETURN 'abc' + 1 AS r")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::DataException);
}

/// TS: "non-numeric operand raises: RETURN true * 2 AS r"
#[test]
fn h_bool_times_number_raises_data_exception() {
    let mut g = modern();
    let err = parse("RETURN true * 2 AS r")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::DataException);
}

/// TS: "a NULL operand still propagates to NULL (not an error)"
#[test]
fn h_null_arithmetic_propagates_to_null() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN null + 1 AS r"),
        vec![vec![Value::Null]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 1 / null AS r"),
        vec![vec![Value::Null]]
    );
}

// ---------------------------------------------------------------------------
// § ISO three-valued comparison of mixed types
// TS: describe('hardening: ISO three-valued comparison of mixed types', ...)
// ---------------------------------------------------------------------------

/// TS: "ordering across incomparable types is UNKNOWN (null)"
#[test]
fn h_ordering_across_incomparable_types_is_unknown() {
    let mut g = modern();
    assert_eq!(rows(&mut g, "RETURN 1 < 'a' AS r"), vec![vec![Value::Null]]);
    assert_eq!(
        rows(&mut g, "RETURN 'a' >= 1 AS r"),
        vec![vec![Value::Null]]
    );
}

/// TS: "equality across types is simply false/true, not null"
#[test]
fn h_equality_across_types_is_false_not_null() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN 5 = '5' AS r"),
        vec![vec![Value::Bool(false)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 5 <> '5' AS r"),
        vec![vec![Value::Bool(true)]]
    );
}

/// TS: "same-type ordering (incl. booleans) still works"
#[test]
fn h_same_type_ordering_still_works() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN 1 < 2 AS r"),
        vec![vec![Value::Bool(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 'a' < 'b' AS r"),
        vec![vec![Value::Bool(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN false >= false AS r"),
        vec![vec![Value::Bool(true)]]
    );
}

// ---------------------------------------------------------------------------
// § Aggregate validation
// TS: describe('hardening: aggregate validation', ...)
// ---------------------------------------------------------------------------

/// TS: "nested aggregates are rejected"
#[test]
fn h_nested_aggregates_rejected() {
    let mut g = modern();
    let err = parse("MATCH (n:Person) RETURN sum(avg(n.age))")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Unsupported);
}

/// TS: "an argless aggregate (other than count(*)) is rejected"
/// (Previously skipped: the engine silently returned NULL; now rejected at
/// plan validation via has_argless_aggregate.)
#[test]
fn h_argless_aggregate_rejected() {
    let mut g = modern();
    let err = parse("MATCH (n:Person) RETURN sum()")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Unsupported);
}

/// TS: "count(*) and normal aggregates still work"
#[test]
fn h_count_star_and_normal_aggregates_work() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN count(*) AS c"),
        vec![vec![n(4.0)]]
    );
}

// ---------------------------------------------------------------------------
// § Group keys do not collide on non-finite numbers
// TS: describe('hardening: group keys do not collide on non-finite numbers', ...)
// ---------------------------------------------------------------------------

/// TS: "NaN and null form distinct groups"
///
/// sqrt(-1) produces NaN; sqrt(null) produces null. They must form separate groups,
/// each with count = 1.
#[test]
fn h_nan_and_null_form_distinct_groups() {
    let lines = [
        r#"{"type":"node","id":"t1","labels":["T"],"properties":{"v":-1}}"#,
        r#"{"type":"node","id":"t2","labels":["T"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let r = rows(&mut g, "MATCH (n:T) RETURN sqrt(n.v) AS k, count(*) AS c");
    assert_eq!(r.len(), 2, "expected 2 distinct groups (NaN and null)");
    // Each group must have count = 1.
    for row in &r {
        assert_eq!(
            row[1],
            n(1.0),
            "each group should have count=1, got row: {row:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// § Variable-length trail semantics
// TS: describe('hardening: variable-length trail semantics', ...)
// ---------------------------------------------------------------------------

/// Ring graph: a → b → c → a
fn ring_graph() -> Graph {
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"name":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"name":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"name":"c"}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"c","to":"a","labels":["R"],"properties":{}}"#,
    ];
    ndjson::decode(&lines.join("\n")).unwrap()
}

/// TS: "a cycle terminates and yields one row per trail"
///
/// From a, trails of ≥1 hop: a→b, a→b→c, a→b→c→a. Next step reuses a→b → stop.
/// Three trails total.
#[test]
fn h_trail_cycle_terminates_one_row_per_trail() {
    let mut g = ring_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {name:'a'})-[:R]->+(x) RETURN count(*) AS c"
        ),
        vec![vec![n(3.0)]]
    );
}

/// TS: "an endpoint reached by multiple trails appears once per trail"
///
/// Diamond: a→b→d and a→c→d. Two distinct 2-hop trails reach d.
#[test]
fn h_trail_endpoint_appears_once_per_trail() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"name":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"name":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"name":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"name":"d"}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"d","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"c","to":"d","labels":["R"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {name:'a'})-[:R]->{2,2}(d) RETURN count(*) AS c"
        ),
        vec![vec![n(2.0)]]
    );
}

/// TS: "an unbounded * on a dense graph hits the trail budget instead of hanging"
///
/// 8-node complete directed graph: every pair has an R edge.
/// The unbounded `*` would produce an astronomically large set of trails;
/// the engine must raise ResourceExhausted instead.
#[test]
fn h_trail_budget_guards_dense_unbounded_star() {
    let mut lines: Vec<String> = Vec::new();
    for i in 0..8u32 {
        lines.push(format!(
            r#"{{"type":"node","id":"{i}","labels":["N"],"properties":{{}}}}"#
        ));
    }
    for i in 0..8u32 {
        for j in 0..8u32 {
            if i != j {
                lines.push(format!(
                    r#"{{"type":"edge","from":"{i}","to":"{j}","labels":["R"],"properties":{{}}}}"#
                ));
            }
        }
    }
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let err = parse("MATCH (a)-[:R]->*(b) RETURN count(*) AS c")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::ResourceExhausted);
}
