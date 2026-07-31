//! Golden tests for operator-CHAIN semantics — precedence, associativity,
//! three-valued boolean folding, string-concat null propagation, error propagation
//! (the evaluator does NOT short-circuit), long-chain arithmetic results, and the
//! planner's AND-split index seed.
//!
//! Pinned BEFORE the n-ary AST flatten refactor so any regression in that
//! behavior is caught by a failing diff rather than noticed later.
//! Mirror of `packages/gql/src/operator-chains.test.ts` (byte-identity). Golden
//! values were captured from the pre-refactor engine, where TS and native agree.

use super::eval::{Params, Val};
use super::{parse, prepare, prepare_with_max_chain};
use crate::graph::{Graph, Value};
use crate::ndjson;

/// A prepared statement honours the operator-chain ceiling too: `prepare` uses the
/// default (10k), `prepare_with_max_chain` overrides it.
#[test]
fn oc_prepared_statement_ceiling_is_configurable() {
    let over = format!("RETURN {} AS r", vec!["true"; 10_002].join(" AND ")); // 10_001 ops
    assert!(prepare(&over).is_err()); // default 10k rejects
    let deep = format!("RETURN {} AS r", vec!["true"; 50_000].join(" AND "));
    let plan = prepare_with_max_chain(&deep, 200_000).unwrap(); // override admits it
    let mut g = ndjson::decode("").unwrap();
    let rs = plan.execute(&mut g, &Params::new()).unwrap();
    let rows: Vec<Vec<Value>> = rs.rows().map(|r| r.to_vec()).collect();
    assert_eq!(rows[0][0], Value::Bool(true));
}

fn empty() -> Graph {
    ndjson::decode("").unwrap()
}

/// Evaluate `RETURN <expr> AS r` on an empty graph to its single scalar value.
fn val(expr: &str) -> Value {
    let mut g = empty();
    let rs = parse(&format!("RETURN {expr} AS r"))
        .unwrap_or_else(|e| panic!("parse `{expr}`: {e}"))
        .execute(&mut g, &Params::new())
        .unwrap_or_else(|e| panic!("exec `{expr}`: {e}"));
    let rows: Vec<Vec<Value>> = rs.rows().map(|r| r.to_vec()).collect();
    rows[0][0].clone()
}

/// Does `RETURN <expr>` raise (a data exception, e.g. division by zero)?
fn errs(expr: &str) -> bool {
    let mut g = empty();
    match parse(&format!("RETURN {expr} AS r")) {
        Ok(p) => p.execute(&mut g, &Params::new()).is_err(),
        Err(_) => true,
    }
}

fn num(x: f64) -> Value {
    Value::Num(x)
}
fn b(x: bool) -> Value {
    Value::Bool(x)
}
fn s(x: &str) -> Value {
    Value::Str(x.into())
}

// ── three-valued boolean folding (null == UNKNOWN) ──────────────────────────

#[test]
fn oc_three_valued_and_or_xor() {
    let cases: &[(&str, Value)] = &[
        ("true AND true", b(true)),
        ("true AND false", b(false)),
        ("true AND null", Value::Null),
        ("false AND null", b(false)), // false dominates
        ("null AND null", Value::Null),
        ("true OR false", b(true)),
        ("false OR false", b(false)),
        ("true OR null", b(true)), // true dominates
        ("false OR null", Value::Null),
        ("null OR null", Value::Null),
        ("true XOR false", b(true)),
        ("true XOR true", b(false)),
        ("true XOR null", Value::Null),
        ("null XOR null", Value::Null),
    ];
    for (e, want) in cases {
        assert_eq!(&val(e), want, "`{e}`");
    }
}

// ── chains fold correctly (the values the n-ary form must reproduce) ─────────

#[test]
fn oc_boolean_chains() {
    assert_eq!(val("true AND true AND false"), b(false));
    assert_eq!(val("true OR false OR null"), b(true));
    assert_eq!(val("true XOR false XOR true"), b(false));
    assert_eq!(val("false AND false AND false"), b(false));
    assert_eq!(val("null OR null OR true"), b(true));
}

// ── precedence & associativity (boolean) ────────────────────────────────────

#[test]
fn oc_boolean_precedence() {
    assert_eq!(val("true OR false AND false"), b(true)); // AND binds tighter
    assert_eq!(val("NOT true AND false"), b(false)); // (NOT true) AND false
    assert_eq!(val("NOT (true AND false)"), b(true));
    assert_eq!(val("true XOR true XOR true"), b(true)); // left-assoc
}

// ── arithmetic left-associativity (non-regroupable ops) ─────────────────────

#[test]
fn oc_arithmetic_associativity() {
    assert_eq!(val("10 - 3 - 2"), num(5.0)); // (10-3)-2
    assert_eq!(val("100 / 5 / 2"), num(10.0)); // (100/5)/2
    assert_eq!(val("20 % 7 % 3"), num(0.0)); // (20%7)%3 = 6%3
    assert_eq!(val("10 - 2 + 3"), num(11.0));
    assert_eq!(val("2 * 3 % 4"), num(2.0)); // (2*3)%4 = 6%4
    assert_eq!(val("2 + 3 * 4"), num(14.0)); // * binds tighter
    assert_eq!(val("(2 + 3) * 4"), num(20.0));
    assert_eq!(val("100 / 10 * 5"), num(50.0));
    assert_eq!(val("7 - 3 - 2 - 1"), num(1.0));
}

// ── string concat chains + null propagation ─────────────────────────────────

#[test]
fn oc_concat_chains() {
    assert_eq!(val("'a' || 'b' || 'c'"), s("abc"));
    assert_eq!(val("'x' || null"), Value::Null);
    assert_eq!(val("null || 'y'"), Value::Null);
}

// ── the evaluator does NOT short-circuit: a fault in any operand propagates ──

#[test]
fn oc_no_short_circuit_error_propagates() {
    // `false AND …` and `true OR …` still evaluate the other operand, so a
    // division-by-zero in it raises rather than being skipped. The n-ary fold
    // must preserve this (evaluate every element).
    assert!(errs("false AND (1.0 / 0.0)"));
    assert!(errs("true OR (1.0 / 0.0)"));
}

// ── long chains still evaluate to the right value (not just "not a crash") ───

#[test]
fn oc_long_chains_evaluate() {
    let add = vec!["1"; 100].join(" + ");
    assert_eq!(val(&add), num(100.0));
    let and = vec!["true"; 50].join(" AND ");
    assert_eq!(val(&and), b(true));
    let or_false = vec!["false"; 200].join(" OR ");
    assert_eq!(val(&or_false), b(false));
}

// ── WHERE chains over a scan (vectorized `eval_vec` path) + planner AND-split ─

fn n_graph() -> Graph {
    let lines: Vec<String> = (0..12)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"n{i}","labels":["N"],"properties":{{"id":"n{i}","a":{i},"b":{}}}}}"#,
                i % 3
            )
        })
        .collect();
    ndjson::decode(&lines.join("\n")).unwrap()
}

fn col_a(g: &mut Graph, q: &str, params: &Params) -> Vec<f64> {
    parse(q)
        .unwrap()
        .execute(g, params)
        .unwrap()
        .rows()
        .map(|r| match &r.to_vec()[0] {
            Value::Num(x) => *x,
            other => panic!("non-num row {other:?}"),
        })
        .collect()
}

#[test]
fn oc_where_chains_vectorized() {
    let mut g = n_graph();
    let p = Params::new();
    assert_eq!(
        col_a(
            &mut g,
            "MATCH (n:N) WHERE n.a > 1 AND n.a < 8 AND n.b <> 0 RETURN n.a AS a ORDER BY a",
            &p
        ),
        vec![2.0, 4.0, 5.0, 7.0]
    );
    assert_eq!(
        col_a(
            &mut g,
            "MATCH (n:N) WHERE n.a = 0 OR n.a = 5 OR n.a = 11 RETURN n.a AS a ORDER BY a",
            &p
        ),
        vec![0.0, 5.0, 11.0]
    );
    // mixed precedence: `a<3 OR a>9 AND b=0` = `a<3 OR (a>9 AND b=0)`
    assert_eq!(
        col_a(
            &mut g,
            "MATCH (n:N) WHERE n.a < 3 OR n.a > 9 AND n.b = 0 RETURN n.a AS a ORDER BY a",
            &p
        ),
        vec![0.0, 1.0, 2.0]
    );
    assert_eq!(
        col_a(
            &mut g,
            "MATCH (n:N) WHERE NOT (n.a = 1 OR n.a = 2) AND n.a < 5 RETURN n.a AS a ORDER BY a",
            &p
        ),
        vec![0.0, 3.0, 4.0]
    );
}

#[test]
fn oc_and_split_index_seed_still_correct() {
    // The planner splits an AND chain to find seekable equality predicates; the
    // n-ary form must still yield the same rows whether or not an index exists.
    for indexed in [false, true] {
        let mut g = n_graph();
        if indexed {
            g.create_vertex_index("id");
        }
        let mut p = Params::new();
        p.insert("x".to_string(), Val::Str("n6".into()));
        p.insert("y".to_string(), Val::Num(0.0));
        assert_eq!(
            col_a(
                &mut g,
                "MATCH (n:N) WHERE n.id = $x AND n.b = $y RETURN n.a AS a",
                &p
            ),
            vec![6.0],
            "indexed={indexed}"
        );
    }
}
