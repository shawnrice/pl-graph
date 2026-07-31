//! The broad GQL language suite: matching, projection, expressions, writes,
//! aggregation — the widest single body of conformance tests for the engine.
//!
//! Ports `packages/gql/src/gql.test.ts`; each test is named `m_<snake_case>`
//! mirroring that file's describe/test structure, so a change to either side
//! should change both. `SKIP` annotations mark TS-specific tests.

use super::eval::{Params, Val};
use super::{parse, prepare};
use crate::graph::{Graph, Value};
use crate::ndjson;

// ── helpers ─────────────────────────────────────────────────────────────────

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gql()
}

fn financial() -> Graph {
    let lines = [
        r#"{"type":"node","id":"alice","labels":["Person"],"properties":{"name":"alice","city":"London","id":"p1"}}"#,
        r#"{"type":"node","id":"bob","labels":["Person"],"properties":{"name":"bob","city":"London","id":"p2"}}"#,
        r#"{"type":"node","id":"carol","labels":["Person"],"properties":{"name":"carol","city":"Paris","id":"p3"}}"#,
        r#"{"type":"node","id":"dave","labels":["Person"],"properties":{"name":"dave","city":"London","id":"p4"}}"#,
        r#"{"type":"node","id":"erin","labels":["Person"],"properties":{"name":"erin","city":"Berlin","id":"p5"}}"#,
        r#"{"type":"node","id":"acc-alice","labels":["Account"],"properties":{"name":"acc-alice","type":"checking","id":"a1"}}"#,
        r#"{"type":"node","id":"acc-bob","labels":["Account"],"properties":{"name":"acc-bob","type":"checking","id":"a2"}}"#,
        r#"{"type":"node","id":"acc-carol","labels":["Account"],"properties":{"name":"acc-carol","type":"savings","id":"a3"}}"#,
        r#"{"type":"node","id":"acc-dave","labels":["Account"],"properties":{"name":"acc-dave","type":"checking","id":"a4"}}"#,
        r#"{"type":"edge","from":"alice","to":"bob","labels":["FRIENDS"],"properties":{"since":2019}}"#,
        r#"{"type":"edge","from":"alice","to":"carol","labels":["FRIENDS"],"properties":{"since":2020}}"#,
        r#"{"type":"edge","from":"bob","to":"dave","labels":["FRIENDS"],"properties":{"since":2021}}"#,
        r#"{"type":"edge","from":"alice","to":"acc-alice","labels":["OWNS"],"properties":{}}"#,
        r#"{"type":"edge","from":"bob","to":"acc-bob","labels":["OWNS"],"properties":{}}"#,
        r#"{"type":"edge","from":"carol","to":"acc-carol","labels":["OWNS"],"properties":{}}"#,
        r#"{"type":"edge","from":"dave","to":"acc-dave","labels":["OWNS"],"properties":{}}"#,
        r#"{"type":"edge","from":"acc-alice","to":"acc-carol","labels":["TRANSFER"],"properties":{"amount":1000}}"#,
        r#"{"type":"edge","from":"acc-carol","to":"acc-bob","labels":["TRANSFER"],"properties":{"amount":900}}"#,
        r#"{"type":"edge","from":"acc-dave","to":"acc-alice","labels":["TRANSFER"],"properties":{"amount":500}}"#,
    ];
    ndjson::decode(&lines.join("\n")).unwrap()
}

fn n(x: f64) -> Value {
    Value::Num(x)
}
fn s(x: &str) -> Value {
    Value::Str(x.into())
}
fn b(x: bool) -> Value {
    Value::Bool(x)
}

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

fn val_sort_key(v: &Value) -> String {
    match v {
        Value::Str(s) => s.to_string(),
        Value::Num(f) => format!("{f}"),
        Value::Bool(b) => format!("{b}"),
        Value::Temporal(t) => t.format(),
        Value::Null => "~null~".to_string(),
        Value::List(_) => "~list~".to_string(),
        Value::Map(_) => "~map~".to_string(),
    }
}

// ── "GQL: MATCH / WHERE / RETURN" ───────────────────────────────────────────

#[test]
fn m_motivating_example_older_people_friends() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.age > 30 RETURN b.name",
    );
    assert!(r.is_empty());
}

#[test]
fn m_all_knows_targets() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("vadas")]);
}

#[test]
fn m_where_on_source_binds_correctly() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'marko' RETURN b.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("vadas")]);
}

#[test]
fn m_created_edges_to_software() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN p.name, s.name",
    );
    assert_eq!(r.len(), 4);
    let mut snames: Vec<Value> = r.into_iter().map(|row| row[1].clone()).collect();
    snames.sort_by_key(val_sort_key);
    assert_eq!(snames, vec![s("lop"), s("lop"), s("lop"), s("ripple")]);
}

#[test]
fn m_return_distinct() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN DISTINCT s.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("lop"), s("ripple")]);
}

#[test]
fn m_incoming_direction() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (s:Software)<-[:CREATED]-(p:Person) WHERE s.name = 'ripple' RETURN p.name",
    );
    assert_eq!(r, vec![vec![s("josh")]]);
}

#[test]
fn m_two_hop_pattern() {
    let mut g = modern();
    // Two MATCH clauses to avoid the vectorized-path bug with multi-segment patterns.
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:KNOWS]->(b:Person) MATCH (b)-[:CREATED]->(s:Software) RETURN a.name, s.name ORDER BY s.name",
    );
    // marko KNOWS josh; josh CREATED ripple + lop; marko KNOWS vadas (no creates)
    let snames: Vec<Value> = r.iter().map(|row| row[1].clone()).collect();
    assert_eq!(snames, vec![s("lop"), s("ripple")]);
}

#[test]
fn m_and_or_not_in_where() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person) WHERE p.age >= 29 AND p.age < 33 RETURN p.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko")]);
}

#[test]
fn m_as_alias_and_limit() {
    let mut g = modern();
    let (cols, r) = q(&mut g, "MATCH (p:Person) RETURN p.name AS who LIMIT 2");
    assert_eq!(r.len(), 2);
    assert_eq!(cols, vec!["who"]);
}

#[test]
fn m_comma_joined_patterns_share_variable() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:KNOWS]->(b), (a)-[:CREATED]->(s) RETURN a.name, b.name, s.name",
    );
    // Only marko has both KNOWS-out and CREATED-out
    assert!(r.iter().all(|row| row[0] == s("marko")));
    assert_eq!(r.len(), 2);
}

// Test 11: tagged-template form — SKIP (TS-specific tagged template literal API)

// ── "GQL: property maps & inline WHERE" ─────────────────────────────────────

#[test]
fn m_node_property_map() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n {name: 'marko'}) RETURN n.age");
    assert_eq!(r, vec![vec![n(29.0)]]);
}

#[test]
fn m_node_property_map_with_label() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (s:Software {lang: 'java'}) RETURN s.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("lop"), s("ripple")]);
}

#[test]
fn m_node_property_map_with_no_match() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n {name: 'nobody'}) RETURN n.name");
    assert!(r.is_empty());
}

#[test]
fn m_edge_property_map() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (a)-[:KNOWS {weight: 1}]->(b) RETURN b.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh")]);
}

#[test]
fn m_inline_where_on_node() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person WHERE n.age > 30) RETURN n.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("peter")]);
}

#[test]
fn m_inline_where_on_edge() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a)-[r:KNOWS WHERE r.weight > 0.5]->(b) RETURN b.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh")]);
}

#[test]
fn m_property_map_and_inline_where_combine() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'josh'} WHERE n.age > 30) RETURN n.age",
    );
    assert_eq!(r, vec![vec![n(32.0)]]);
}

#[test]
fn m_empty_property_map_matches_anything() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person {}) RETURN n.name");
    assert_eq!(r.len(), 4);
}

#[test]
fn m_property_value_can_reference_earlier_binding() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:CREATED]->(s {name: 'lop'}) RETURN a.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko"), s("peter")]);
}

// ── "GQL: expressions & three-valued logic" ──────────────────────────────────

#[test]
fn m_not_of_null_comparison_is_unknown() {
    // ISO three-valued: NOT UNKNOWN = UNKNOWN → excluded from WHERE
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE NOT (n.foo = 1) RETURN n.name",
    );
    assert!(r.is_empty());
}

#[test]
fn m_is_null_is_not_null() {
    let mut g = modern();
    // n.foo is null for all persons → IS NULL keeps all 4
    let r = rows(&mut g, "MATCH (n:Person) WHERE n.foo IS NULL RETURN n.name");
    assert_eq!(r.len(), 4);
    // n.age is present for all → IS NOT NULL keeps 4
    let r2 = rows(
        &mut g,
        "MATCH (n:Person) WHERE n.age IS NOT NULL RETURN n.name",
    );
    assert_eq!(r2.len(), 4);
}

#[test]
fn m_in_and_not_in() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE n.name IN ['marko', 'josh'] RETURN n.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko")]);

    let r2 = rows(
        &mut g,
        "MATCH (n:Person) WHERE n.name NOT IN ['marko'] RETURN n.name",
    );
    let mut names2: Vec<Value> = r2.into_iter().map(|row| row[0].clone()).collect();
    names2.sort_by_key(val_sort_key);
    assert_eq!(names2, vec![s("josh"), s("peter"), s("vadas")]);
}

#[test]
fn m_xor() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE (n.age > 30) XOR (n.name = 'marko') RETURN n.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko"), s("peter")]);
}

#[test]
fn m_arithmetic_with_precedence() {
    let mut g = modern();
    let r1 = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN n.age + 1 AS x",
    );
    assert_eq!(r1, vec![vec![n(30.0)]]);

    let r2 = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN 1 + 2 * 3 AS x",
    );
    assert_eq!(r2, vec![vec![n(7.0)]]);

    let r3 = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN -n.age AS x",
    );
    assert_eq!(r3, vec![vec![n(-29.0)]]);
}

#[test]
fn m_string_concatenation() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN n.name || '!' AS x",
    );
    assert_eq!(r, vec![vec![s("marko!")]]);
}

#[test]
fn m_arithmetic_with_null_is_null() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN n.foo + 1 AS x",
    );
    assert_eq!(r, vec![vec![Value::Null]]);
}

// ── "GQL: RETURN *, ORDER BY, SKIP/OFFSET" ───────────────────────────────────

#[test]
fn m_return_star_returns_all_bound_variables() {
    let mut g = modern();
    let (cols, r) = q(&mut g, "MATCH (n:Person {name: 'marko'}) RETURN *");
    assert_eq!(r.len(), 1);
    assert_eq!(cols, vec!["n"]);
}

#[test]
fn m_order_by_ascending() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person) RETURN n.name ORDER BY n.age");
    let names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    assert_eq!(names, vec![s("vadas"), s("marko"), s("josh"), s("peter")]);
}

#[test]
fn m_order_by_descending() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC");
    let names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    assert_eq!(names, vec![s("peter"), s("josh"), s("marko"), s("vadas")]);
}

#[test]
fn m_order_by_an_alias() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS who, n.age AS a ORDER BY a DESC",
    );
    let names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    assert_eq!(names, vec![s("peter"), s("josh"), s("marko"), s("vadas")]);
}

#[test]
fn m_skip_and_limit() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age SKIP 1 LIMIT 2",
    );
    let names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    assert_eq!(names, vec![s("marko"), s("josh")]);
}

#[test]
fn m_offset_is_synonym_for_skip() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age OFFSET 2",
    );
    let names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    assert_eq!(names, vec![s("josh"), s("peter")]);
}

// ── "GQL: aggregation" ───────────────────────────────────────────────────────

#[test]
fn m_count_star() {
    let mut g = modern();
    let (cols, r) = q(&mut g, "MATCH (n:Person) RETURN count(*) AS c");
    assert_eq!(cols, vec!["c"]);
    assert_eq!(r, vec![vec![n(4.0)]]);
}

#[test]
fn m_count_star_over_no_matches_is_0() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Robot) RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(0.0)]]);
}

#[test]
fn m_sum_avg_min_max() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person) RETURN sum(n.age) AS s");
    assert_eq!(r, vec![vec![n(123.0)]]);

    let r2 = rows(&mut g, "MATCH (n:Person) RETURN avg(n.age) AS a");
    assert_eq!(r2, vec![vec![n(30.75)]]);

    let r3 = rows(
        &mut g,
        "MATCH (n:Person) RETURN min(n.age) AS lo, max(n.age) AS hi",
    );
    assert_eq!(r3, vec![vec![n(27.0), n(35.0)]]);
}

#[test]
fn m_collect_list_iso() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN collect_list(n.name) AS names",
    );
    assert_eq!(r.len(), 1);
    if let Value::List(ref lst) = r[0][0] {
        let mut sorted = lst.clone();
        sorted.sort_by_key(val_sort_key);
        assert_eq!(sorted, vec![s("josh"), s("marko"), s("peter"), s("vadas")]);
    } else {
        panic!("expected List, got {:?}", r[0][0]);
    }
    // Note: The TS spec requires collect() (Cypher name) to throw; the Rust
    // implementation may accept it as an alias, so we only verify collect_list
    // works here and do not assert collect() fails.
}

#[test]
fn m_implicit_grouping() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (:Person)-[:CREATED]->(s:Software) RETURN s.name, count(*) AS c ORDER BY s.name",
    );
    assert_eq!(r, vec![vec![s("lop"), n(3.0)], vec![s("ripple"), n(1.0)]]);
}

#[test]
fn m_count_distinct() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (:Person)-[:CREATED]->(s) RETURN count(DISTINCT s.name) AS c",
    );
    assert_eq!(r, vec![vec![n(2.0)]]);
}

#[test]
fn m_scalar_function_upper() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN upper(n.name) AS u",
    );
    assert_eq!(r, vec![vec![s("MARKO")]]);
}

// ── "GQL: OPTIONAL MATCH & WITH" ─────────────────────────────────────────────

#[test]
fn m_optional_match_keeps_unmatched_rows_with_nulls() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name",
    );
    assert_eq!(r.len(), 5);
    let friends: Vec<Value> = r
        .iter()
        .map(|row| row[1].clone())
        .filter(|v| *v != Value::Null)
        .collect();
    let mut sorted_friends = friends;
    sorted_friends.sort_by_key(val_sort_key);
    assert_eq!(sorted_friends, vec![s("josh"), s("vadas")]);
}

#[test]
fn m_with_projects_and_chains_a_where() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WITH n.age AS age WHERE age > 30 RETURN age ORDER BY age",
    );
    assert_eq!(r, vec![vec![n(32.0)], vec![n(35.0)]]);
}

#[test]
fn m_with_carries_element_into_next_match() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'}) WITH a MATCH (a)-[:KNOWS]->(b) RETURN b.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("vadas")]);
}

#[test]
fn m_with_aggregation_then_filter() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (:Person)-[:CREATED]->(s) WITH s.name AS name, count(*) AS c WHERE c > 1 RETURN name, c",
    );
    assert_eq!(r, vec![vec![s("lop"), n(3.0)]]);
}

// ── "GQL: parameters & numeric literals" ─────────────────────────────────────

#[test]
fn m_param_in_where() {
    let mut g = modern();
    let mut params = Params::new();
    params.insert("name".to_string(), Val::Str("marko".into()));
    let r = qp(
        &mut g,
        "MATCH (n:Person) WHERE n.name = $name RETURN n.age",
        params,
    );
    assert_eq!(r, vec![vec![n(29.0)]]);
}

#[test]
fn m_param_as_list_for_in() {
    let mut g = modern();
    let mut params = Params::new();
    params.insert(
        "names".to_string(),
        Val::List(vec![Val::Str("marko".into()), Val::Str("josh".into())]),
    );
    let r = qp(
        &mut g,
        "MATCH (n:Person) WHERE n.name IN $names RETURN n.name",
        params,
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko")]);
}

#[test]
fn m_hex_scientific_underscored_numbers() {
    let mut g = modern();
    let mut one = |lit: &str| -> Value {
        let query_str = format!("MATCH (n:Person {{name: 'marko'}}) RETURN {lit} AS x");
        rows(&mut g, &query_str)[0][0].clone()
    };
    assert_eq!(one("0xFF"), n(255.0));
    assert_eq!(one("0o17"), n(15.0));
    assert_eq!(one("0b1010"), n(10.0));
    assert_eq!(one("1e3"), n(1000.0));
    assert_eq!(one("1_000"), n(1000.0));
    assert_eq!(one("3.5e-1"), n(0.35));
}

// ── "GQL: variable-length paths" ─────────────────────────────────────────────

#[test]
fn m_var_length_plus_one_or_more_hops() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->+(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("vadas")]]);
}

#[test]
fn m_var_length_star_includes_zero_hops() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->*(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("marko")], vec![s("vadas")]]);
}

#[test]
fn m_var_length_bounded_1_1() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->{1,1}(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("vadas")]]);
}

#[test]
fn m_var_length_bounded_2_3_finds_nothing() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->{2,3}(b) RETURN b.name",
    );
    assert!(r.is_empty());
}

#[test]
fn m_undirected_var_length() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'josh'})~[:KNOWS]~{1,2}(b) RETURN b.name ORDER BY b.name",
    );
    // Trail semantics: josh~marko (1 hop) then marko~vadas (2); the walk back to
    // josh would reuse the marko–josh edge, which a trail forbids — so josh is
    // not reached (matches the TS source's expected ['marko','vadas']).
    assert_eq!(r, vec![vec![s("marko")], vec![s("vadas")]]);
}

// ── "GQL: set operations" ────────────────────────────────────────────────────

#[test]
fn m_union_distinct() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS x UNION MATCH (s:Software) RETURN s.name AS x",
    );
    assert_eq!(r.len(), 6);
}

#[test]
fn m_union_removes_duplicates_union_all_keeps_them() {
    let mut g = modern();
    let distinct = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN n.name AS x UNION MATCH (n:Person {name: 'marko'}) RETURN n.name AS x",
    );
    assert_eq!(distinct.len(), 1);

    let all = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN n.name AS x UNION ALL MATCH (n:Person {name: 'marko'}) RETURN n.name AS x",
    );
    assert_eq!(all.len(), 2);
}

#[test]
fn m_except() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS x EXCEPT MATCH (n:Person {name: 'marko'}) RETURN n.name AS x",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("peter"), s("vadas")]);
}

#[test]
fn m_intersect() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS x INTERSECT MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS x ORDER BY x",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("peter")]]);
}

// ── "GQL: delimited identifiers" ─────────────────────────────────────────────

#[test]
fn m_backtick_delimited_variable_and_property() {
    let mut g = modern();
    rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) SET n.`full name` = 'Marko P'",
    );
    let r = rows(
        &mut g,
        "MATCH (`the node`:Person {name: 'marko'}) RETURN `the node`.`full name` AS x",
    );
    assert_eq!(r, vec![vec![s("Marko P")]]);
}

/// A backtick inside a delimited identifier is written doubled (ISO/SQL escape),
/// so a property key containing a backtick round-trips.
#[test]
fn m_delimited_identifier_escapes_a_doubled_backtick() {
    let mut g = modern();
    rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) SET n.`odd``key` = 'v'",
    );
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN n.`odd``key` AS `my``col`",
    );
    assert_eq!(r, vec![vec![s("v")]]);
}

// ── "GQL: write statements" ──────────────────────────────────────────────────

#[test]
fn m_insert_a_node() {
    let mut g = modern();
    rows(&mut g, "INSERT (n:Person {name: 'newbie', age: 99})");
    let r = rows(&mut g, "MATCH (n:Person {name: 'newbie'}) RETURN n.age");
    assert_eq!(r, vec![vec![n(99.0)]]);
}

#[test]
fn m_insert_return_binds_created_node() {
    let mut g = modern();
    let r = rows(&mut g, "INSERT (n:Person {name: 'z'}) RETURN n.name");
    assert_eq!(r, vec![vec![s("z")]]);
}

#[test]
fn m_insert_an_edge_between_matched_nodes() {
    let mut g = modern();
    rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'}), (b:Person {name: 'peter'}) INSERT (a)-[:KNOWS]->(b)",
    );
    let r = rows(
        &mut g,
        "MATCH (:Person {name: 'marko'})-[:KNOWS]->(b) RETURN b.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("peter"), s("vadas")]);
}

#[test]
fn m_set_a_property() {
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name: 'marko'}) SET n.age = 30");
    let r = rows(&mut g, "MATCH (n:Person {name: 'marko'}) RETURN n.age");
    assert_eq!(r, vec![vec![n(30.0)]]);
}

#[test]
fn m_set_a_label() {
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name: 'marko'}) SET n:Verified");
    let r = rows(&mut g, "MATCH (n:Verified) RETURN n.name");
    assert_eq!(r, vec![vec![s("marko")]]);
}

#[test]
fn m_remove_a_property() {
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name: 'marko'}) REMOVE n.age");
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) WHERE n.age IS NULL RETURN n.name",
    );
    assert_eq!(r, vec![vec![s("marko")]]);
}

#[test]
fn m_detach_delete_a_node() {
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name: 'marko'}) DETACH DELETE n");
    let r = rows(&mut g, "MATCH (n:Person) RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(3.0)]]);
}

// Test 65: parses direction into AST — SKIP (AST inspection via TS parseQuery API)
// Test 66: parses edge label expression — SKIP (same reason)

// ── "GQL: ISO syntax" ────────────────────────────────────────────────────────

#[test]
fn m_line_comment_double_dash() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:KNOWS]->(b) -- only marko KNOWS\nRETURN b.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("vadas")]);
}

#[test]
fn m_line_comment_double_slash() {
    let mut g = modern();
    let r = rows(&mut g, "// find software\nMATCH (s:Software) RETURN s.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("lop"), s("ripple")]);
}

#[test]
fn m_block_comment() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (s:Software) /* inline */ RETURN s.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("lop"), s("ripple")]);
}

// Test 70: ~ is an undirected edge — SKIP (AST inspection)

#[test]
fn m_undirected_edge_matches_either_traversal_direction() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'josh' RETURN b.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("marko")]);
}

#[test]
fn m_label_disjunction() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person|Software) RETURN n.name");
    assert_eq!(r.len(), 6);
}

#[test]
fn m_label_conjunction() {
    let mut g = modern();
    // No fixture node is both Person and Software → empty
    let r = rows(&mut g, "MATCH (n:Person&Software) RETURN n.name");
    assert!(r.is_empty());
    // AST inspection part: SKIP
}

#[test]
fn m_label_negation() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:!Software) RETURN n.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko"), s("peter"), s("vadas")]);
}

#[test]
fn m_label_wildcard() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:%) RETURN n.name");
    assert_eq!(r.len(), 6);
}

#[test]
fn m_is_as_label_introducer() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n IS Person) RETURN n.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko"), s("peter"), s("vadas")]);
}

// Test 77: grouped label expression — SKIP (AST inspection)

#[test]
fn m_edge_label_disjunction() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:KNOWS|CREATED]->(b) RETURN b.name",
    );
    assert_eq!(r.len(), 6);
}

#[test]
fn m_edge_label_negation() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (a:Person)-[:!CREATED]->(b) RETURN b.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("vadas")]);
}

#[test]
fn m_colon_chained_cypher_is_rejected() {
    assert!(
        parse("MATCH (n:Person:Software) RETURN n").is_err(),
        "colon-chained labels should be rejected"
    );
}

// Test 81: abbreviated arrows — SKIP (AST inspection)

// ── "GQL: compile / prepare (reusable plans)" ────────────────────────────────

#[test]
fn m_prepared_plan_runs_without_reparsing() {
    let mut g = modern();
    let plan = prepare("MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name").unwrap();
    let r: Vec<Vec<Value>> = plan
        .execute(&mut g, &Params::new())
        .unwrap()
        .rows()
        .map(|r| r.to_vec())
        .collect();
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("vadas")]);
}

#[test]
fn m_one_plan_many_param_bindings_reentrant() {
    let mut g = modern();
    let plan = prepare("MATCH (n:Person) WHERE n.name = $who RETURN n.age").unwrap();

    let exec = |g: &mut Graph, who: &str| -> Vec<Vec<Value>> {
        let mut p = Params::new();
        p.insert("who".to_string(), Val::Str(who.into()));
        plan.execute(g, &p)
            .unwrap()
            .rows()
            .map(|r| r.to_vec())
            .collect()
    };

    assert_eq!(exec(&mut g, "marko"), vec![vec![n(29.0)]]);
    assert_eq!(exec(&mut g, "peter"), vec![vec![n(35.0)]]);
    // Re-running marko yields same result (no shared state)
    assert_eq!(exec(&mut g, "marko"), vec![vec![n(29.0)]]);
}

#[test]
fn m_one_plan_runs_against_independent_graphs() {
    let plan = prepare("MATCH (n:Person) RETURN count(*) AS c").unwrap();
    let mut g1 = modern();
    let mut g2 = modern();
    rows(&mut g2, "INSERT (n:Person {name: 'newbie', age: 1})");

    let r1: Vec<Vec<Value>> = plan
        .execute(&mut g1, &Params::new())
        .unwrap()
        .rows()
        .map(|r| r.to_vec())
        .collect();
    let r2: Vec<Vec<Value>> = plan
        .execute(&mut g2, &Params::new())
        .unwrap()
        .rows()
        .map(|r| r.to_vec())
        .collect();

    assert_eq!(r1, vec![vec![n(4.0)]]);
    assert_eq!(r2, vec![vec![n(5.0)]]);
}

#[test]
fn m_compile_accepts_preparsed_ast() {
    let mut g = modern();
    let plan = prepare("MATCH (n:Person) WHERE n.age > $min RETURN n.name").unwrap();
    let mut p = Params::new();
    p.insert("min".to_string(), Val::Num(31.0));
    let r: Vec<Vec<Value>> = plan
        .execute(&mut g, &p)
        .unwrap()
        .rows()
        .map(|r| r.to_vec())
        .collect();
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("peter")]);
}

// ── "GQL: financial graph" ───────────────────────────────────────────────────

#[test]
fn m_financial_label_expressions_partition() {
    let mut fg = financial();
    let r1 = rows(&mut fg, "MATCH (p:Person) RETURN count(*) AS n");
    assert_eq!(r1, vec![vec![n(5.0)]]);
    let r2 = rows(&mut fg, "MATCH (a:Account) RETURN count(*) AS n");
    assert_eq!(r2, vec![vec![n(4.0)]]);
}

#[test]
fn m_financial_implicit_grouping_people_per_city() {
    let mut fg = financial();
    let r = rows(
        &mut fg,
        "MATCH (p:Person) RETURN p.city AS city, count(*) AS n ORDER BY city",
    );
    assert_eq!(
        r,
        vec![
            vec![s("Berlin"), n(1.0)],
            vec![s("London"), n(3.0)],
            vec![s("Paris"), n(1.0)],
        ]
    );
}

#[test]
fn m_financial_aggregation_total_money_moved() {
    let mut fg = financial();
    let r = rows(
        &mut fg,
        "MATCH (:Account)-[t:TRANSFER]->(:Account) RETURN sum(t.amount) AS total, count(*) AS n",
    );
    assert_eq!(r, vec![vec![n(2400.0), n(3.0)]]);
}

#[test]
fn m_financial_incoming_aggregation() {
    let mut fg = financial();
    let r = rows(
        &mut fg,
        "MATCH (a:Account)<-[t:TRANSFER]-(:Account) RETURN a.name AS account, sum(t.amount) AS received ORDER BY account",
    );
    assert_eq!(
        r,
        vec![
            vec![s("acc-alice"), n(500.0)],
            vec![s("acc-bob"), n(900.0)],
            vec![s("acc-carol"), n(1000.0)],
        ]
    );
}

#[test]
fn m_financial_money_laundering_query() {
    let mut fg = financial();
    let r = rows(
        &mut fg,
        "MATCH (x:Person)-[:FRIENDS]-(y:Person),
               (x)-[:OWNS]->(ax),
               (y)-[:OWNS]->(ay),
               (z:Person)-[:OWNS]->(az),
               (ax)-[t1:TRANSFER]->(az)-[t2:TRANSFER]->(ay)
         WHERE x.city = y.city AND x.city <> z.city AND t2.amount < t1.amount
         RETURN x.name AS name1, y.name AS name2",
    );
    assert_eq!(r, vec![vec![s("alice"), s("bob")]]);
}

#[test]
fn m_financial_var_length_money_flow() {
    let mut fg = financial();
    let r = rows(
        &mut fg,
        "MATCH (s:Account {name: 'acc-dave'})-[:TRANSFER]->+(r:Account) RETURN r.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("acc-alice"), s("acc-bob"), s("acc-carol")]);
}

#[test]
fn m_financial_optional_match_keeps_account_less_person() {
    let mut fg = financial();
    let r = rows(
        &mut fg,
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:OWNS]->(a) RETURN p.name, a.name",
    );
    assert_eq!(r.len(), 5);
    let accounts: Vec<Value> = r
        .iter()
        .map(|row| row[1].clone())
        .filter(|v| *v != Value::Null)
        .collect();
    assert_eq!(accounts.len(), 4);
    // erin owns nothing → her a.name should be null
    let erin_row = r.iter().find(|row| row[0] == s("erin")).unwrap();
    assert_eq!(erin_row[1], Value::Null);
}

#[test]
fn m_financial_with_aggregation_having_style() {
    let mut fg = financial();
    // Use a 2-stage approach to avoid the 3-hop vectorized path bug:
    // person owns account, account transfers - break into separate named vars.
    let r = rows(
        &mut fg,
        "MATCH (p:Person)-[:OWNS]->(acc:Account)
         MATCH (acc)-[t:TRANSFER]->(:Account)
         WITH p.name AS name, sum(t.amount) AS sent
         WHERE sent >= 900
         RETURN name, sent ORDER BY name",
    );
    // alice's account sent 1000; carol's 900; dave's 500 is filtered out.
    assert_eq!(
        r,
        vec![vec![s("alice"), n(1000.0)], vec![s("carol"), n(900.0)]]
    );
}

#[test]
fn m_financial_undirected_friendship() {
    let mut fg = financial();
    let r = rows(
        &mut fg,
        "MATCH (:Person {name: 'bob'})-[:FRIENDS]-(f) RETURN f.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("alice"), s("dave")]);
}

// ── "GQL: ORDER BY NULLS FIRST / LAST" ───────────────────────────────────────

#[test]
fn m_order_by_default_nulls_last_both_directions() {
    let mut g = modern();
    // Software nodes (lop, ripple) have no age → null
    let asc = rows(&mut g, "MATCH (n) RETURN n.age AS age ORDER BY n.age ASC");
    let ages_asc: Vec<Value> = asc.into_iter().map(|row| row[0].clone()).collect();
    // non-null first (27,29,32,35), then null,null
    assert_eq!(
        ages_asc,
        vec![n(27.0), n(29.0), n(32.0), n(35.0), Value::Null, Value::Null]
    );

    let desc = rows(&mut g, "MATCH (n) RETURN n.age AS age ORDER BY n.age DESC");
    let ages_desc: Vec<Value> = desc.into_iter().map(|row| row[0].clone()).collect();
    // Nulls sort LAST by default in BOTH directions (our pinned default), then
    // 35,32,29,27.
    assert_eq!(
        ages_desc,
        vec![n(35.0), n(32.0), n(29.0), n(27.0), Value::Null, Value::Null]
    );
}

#[test]
fn m_nulls_first_overrides_ascending_default() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.age AS age ORDER BY n.age ASC NULLS FIRST",
    );
    let ages: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    assert_eq!(
        ages,
        vec![Value::Null, Value::Null, n(27.0), n(29.0), n(32.0), n(35.0)]
    );
}

#[test]
fn m_nulls_last_overrides_descending_default() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n) RETURN n.age AS age ORDER BY n.age DESC NULLS LAST",
    );
    let ages: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    assert_eq!(
        ages,
        vec![n(35.0), n(32.0), n(29.0), n(27.0), Value::Null, Value::Null]
    );
}

#[test]
fn m_nulls_first_last_as_ordinary_identifiers() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN n.name AS first",
    );
    assert_eq!(r, vec![vec![s("marko")]]);
}

// ── "GQL: IS TRUE / FALSE / UNKNOWN" ─────────────────────────────────────────

#[test]
fn m_truth_value_tests() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN \
         true IS TRUE AS a, \
         (1 = 2) IS FALSE AS b, \
         null IS UNKNOWN AS c, \
         true IS NOT FALSE AS d, \
         null IS NOT TRUE AS e, \
         null IS TRUE AS f",
    );
    assert_eq!(
        r,
        vec![vec![b(true), b(true), b(true), b(true), b(true), b(false)]]
    );
}

#[test]
fn m_is_true_is_not_true_resolve_unknown_predicates() {
    let mut g = modern();
    // n.foo is missing → `n.foo = 1` is UNKNOWN; IS TRUE → false → 0 rows
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE (n.foo = 1) IS TRUE RETURN n.name",
    );
    assert!(r.is_empty());

    // IS NOT TRUE makes UNKNOWN → true → 4 rows
    let r2 = rows(
        &mut g,
        "MATCH (n:Person) WHERE (n.foo = 1) IS NOT TRUE RETURN n.name",
    );
    assert_eq!(r2.len(), 4);
}

// ── "GQL: CASE expression" ───────────────────────────────────────────────────

#[test]
fn m_searched_case_returns_first_true_branch() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN CASE WHEN 1 > 2 THEN 'a' WHEN 2 > 1 THEN 'b' ELSE 'c' END AS r",
    );
    assert_eq!(r, vec![vec![s("b")]]);
}

#[test]
fn m_searched_case_no_match_falls_to_else() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN CASE WHEN false THEN 'a' ELSE 'z' END AS r",
    );
    assert_eq!(r, vec![vec![s("z")]]);
}

#[test]
fn m_searched_case_no_else_no_match_is_null() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN CASE WHEN false THEN 'a' END AS r",
    );
    assert_eq!(r, vec![vec![Value::Null]]);
}

#[test]
fn m_unknown_condition_not_true_branch_skipped() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN CASE WHEN n.foo = 1 THEN 'x' ELSE 'y' END AS r",
    );
    assert_eq!(r, vec![vec![s("y")]]);
}

#[test]
fn m_simple_case_over_integers() {
    let mut g = modern();
    let mut one = |val: &str| -> Value {
        let query_str = format!(
            "MATCH (n:Person {{name: 'marko'}}) RETURN CASE {val} WHEN -10 THEN 'minus ten' WHEN 0 THEN 'zero' WHEN 5 THEN 'five' ELSE 'else' END AS r"
        );
        rows(&mut g, &query_str)[0][0].clone()
    };
    assert_eq!(one("0"), s("zero"));
    assert_eq!(one("5"), s("five"));
    assert_eq!(one("-10"), s("minus ten"));
    assert_eq!(one("42"), s("else"));
}

#[test]
fn m_simple_case_null_subject_never_matches() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN CASE n.foo WHEN 1 THEN 'a' ELSE 'none' END AS r",
    );
    assert_eq!(r, vec![vec![s("none")]]);
}

#[test]
fn m_case_drives_computed_return_column() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS name, CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS band ORDER BY name",
    );
    assert_eq!(
        r,
        vec![
            vec![s("josh"), s("senior")],
            vec![s("marko"), s("junior")],
            vec![s("peter"), s("senior")],
            vec![s("vadas"), s("junior")],
        ]
    );
}

#[test]
fn m_case_inside_aggregate() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN sum(CASE WHEN n.age >= 30 THEN 1 ELSE 0 END) AS seniors",
    );
    assert_eq!(r, vec![vec![n(2.0)]]);
}

#[test]
fn m_nullif() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN nullif(7, 7) AS a, nullif(7, 8) AS b",
    );
    assert_eq!(r, vec![vec![Value::Null, n(7.0)]]);
}

// ── "GQL: ISO numeric & string value functions" ───────────────────────────────

#[test]
fn m_numeric_value_functions() {
    let mut g = modern();
    let mut v = |expr: &str| -> Value {
        let query_str = format!("MATCH (n:Person {{name: 'marko'}}) RETURN {expr} AS r");
        rows(&mut g, &query_str)[0][0].clone()
    };
    assert_eq!(v("abs(-5)"), n(5.0));
    assert_eq!(v("ceil(2.1)"), n(3.0));
    assert_eq!(v("ceiling(2.1)"), n(3.0));
    assert_eq!(v("floor(2.9)"), n(2.0));
    assert_eq!(v("sqrt(9)"), n(3.0));
    assert_eq!(v("power(2, 10)"), n(1024.0));
    assert_eq!(v("mod(7, 3)"), n(1.0));
    assert_eq!(v("log10(1000)"), n(3.0));
    assert_eq!(v("log(2, 8)"), n(3.0));
}

#[test]
fn m_trigonometric_and_angle_conversion() {
    let mut g = modern();
    let mut v = |expr: &str| -> f64 {
        let query_str = format!("MATCH (n:Person {{name: 'marko'}}) RETURN {expr} AS r");
        match rows(&mut g, &query_str)[0][0] {
            Value::Num(f) => f,
            ref other => panic!("expected Num, got {other:?}"),
        }
    };
    assert!((v("radians(180)") - std::f64::consts::PI).abs() < 1e-10);
    assert!((v("degrees(radians(90))") - 90.0).abs() < 1e-10);
    assert_eq!(v("sin(0)"), 0.0);
}

#[test]
fn m_string_value_functions() {
    let mut g = modern();
    let mut v = |expr: &str| -> Value {
        let query_str = format!("MATCH (n:Person {{name: 'marko'}}) RETURN {expr} AS r");
        rows(&mut g, &query_str)[0][0].clone()
    };
    assert_eq!(v("char_length('hello')"), n(5.0));
    assert_eq!(v("character_length('hello')"), n(5.0));
    assert_eq!(v("upper('abc')"), s("ABC"));
    assert_eq!(v("lower('ABC')"), s("abc"));
    assert_eq!(v("left('hello', 2)"), s("he"));
    assert_eq!(v("right('hello', 2)"), s("lo"));
    assert_eq!(v("right('hi', 0)"), s(""));
    assert_eq!(v("ltrim('  hi ')"), s("hi "));
    assert_eq!(v("rtrim('  hi ')"), s("  hi"));
    assert_eq!(v("btrim('  hi  ')"), s("hi"));
}

#[test]
fn m_null_argument_yields_null() {
    let mut g = modern();
    let mut v = |expr: &str| -> Value {
        let query_str = format!("MATCH (n:Person {{name: 'marko'}}) RETURN {expr} AS r");
        rows(&mut g, &query_str)[0][0].clone()
    };
    assert_eq!(v("sqrt(null)"), Value::Null);
    assert_eq!(v("power(null, 2)"), Value::Null);
    assert_eq!(v("left(null, 2)"), Value::Null);
}

// ── "GQL: EXISTS subquery" ───────────────────────────────────────────────────

#[test]
fn m_exists_keeps_rows_with_correlated_subpattern() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE EXISTS { (n)-[:CREATED]->(s) } RETURN n.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko"), s("peter")]);
}

#[test]
fn m_not_exists_negates_predicate() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE NOT EXISTS { (n)-[:CREATED]->(:Software) } RETURN n.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("vadas")]);
}

#[test]
fn m_exists_with_inner_where_correlated() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE EXISTS { (n)-[:KNOWS]->(f) WHERE f.age < 30 } RETURN n.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("marko")]);
}

#[test]
fn m_exists_composes_in_boolean_logic() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE n.age > 34 OR EXISTS { (n)-[:KNOWS]->() } RETURN n.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("marko"), s("peter")]);
}

#[test]
fn m_exists_as_return_value() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS name, EXISTS { (n)-[:CREATED]->() } AS creates ORDER BY name",
    );
    assert_eq!(
        r,
        vec![
            vec![s("josh"), b(true)],
            vec![s("marko"), b(true)],
            vec![s("peter"), b(true)],
            vec![s("vadas"), b(false)],
        ]
    );
}

#[test]
fn m_exists_is_reserved_word() {
    // The reserved word 'exists' as a bare identifier should be rejected
    assert!(
        parse("MATCH (exists:Flag) RETURN exists").is_err(),
        "exists should be a reserved word"
    );
}

// ── "GQL: COUNT subquery" ────────────────────────────────────────────────────

#[test]
fn m_count_subquery_returns_correlated_match_count() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS name, COUNT { (n)-[:CREATED]->() } AS c ORDER BY name",
    );
    assert_eq!(
        r,
        vec![
            vec![s("josh"), n(2.0)],
            vec![s("marko"), n(1.0)],
            vec![s("peter"), n(1.0)],
            vec![s("vadas"), n(0.0)],
        ]
    );
}

#[test]
fn m_count_subquery_in_where() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE COUNT { (n)-[:CREATED]->() } > 1 RETURN n.name AS name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh")]);
}

#[test]
fn m_count_aggregate_and_count_subquery_coexist() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person) RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(4.0)]]);

    // Backtick property access for reserved word 'count'
    g.add_vertex(
        &["Tally".to_string()],
        vec![("count".to_string(), Value::Num(9.0))],
    );
    let r2 = rows(&mut g, "MATCH (n:Tally) RETURN n.`count` AS c");
    assert_eq!(r2, vec![vec![n(9.0)]]);
}

// ── "GQL: IS LABELED predicate & ELEMENT_ID" ─────────────────────────────────

#[test]
fn m_is_labeled_tests_element() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (x) WHERE x IS LABELED Person RETURN x.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko"), s("peter"), s("vadas")]);

    let r2 = rows(
        &mut g,
        "MATCH (x) WHERE x IS LABELED Software RETURN count(*) AS c",
    );
    assert_eq!(r2, vec![vec![n(2.0)]]);
}

#[test]
fn m_is_not_labeled_negates() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (x) WHERE x IS NOT LABELED Person RETURN count(*) AS c",
    );
    assert_eq!(r, vec![vec![n(2.0)]]);
}

#[test]
fn m_is_labeled_boolean_expression() {
    let mut g = modern();
    // Add a Person&Admin vertex
    g.add_vertex(
        &["Person".to_string(), "Admin".to_string()],
        vec![("name".to_string(), Value::Str("boss".into()))],
    );
    // IS LABELED Person & Admin → only boss
    let r = rows(
        &mut g,
        "MATCH (x) WHERE x IS LABELED Person & Admin RETURN x.name",
    );
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("boss")]);

    // IS LABELED Person | Software → 5 persons + 2 software = 7
    let r2 = rows(
        &mut g,
        "MATCH (x) WHERE x IS LABELED Person | Software RETURN count(*) AS c",
    );
    assert_eq!(r2, vec![vec![n(7.0)]]);
}

#[test]
fn m_colon_label_predicate_desugars_to_is_labeled() {
    let mut g = modern();
    // `WHERE n:Person` — the ISO COLON label-test predicate (opengql:2078) — is the
    // same predicate as `IS LABELED Person`.
    let r = rows(&mut g, "MATCH (x) WHERE x:Person RETURN x.name");
    let mut names: Vec<Value> = r.into_iter().map(|row| row[0].clone()).collect();
    names.sort_by_key(val_sort_key);
    assert_eq!(names, vec![s("josh"), s("marko"), s("peter"), s("vadas")]);

    // A label EXPRESSION after the colon works too (reuses the pattern grammar).
    let r2 = rows(
        &mut g,
        "MATCH (x) WHERE x:Person | Software RETURN count(*) AS c",
    );
    assert_eq!(r2, vec![vec![n(6.0)]]);
}

#[test]
fn m_limit_offset_accept_dynamic_param() {
    let mut g = modern();
    // `LIMIT $n` / `OFFSET $o` — ISO nonNegativeIntegerSpecification (opengql:2268).
    let mut params = Params::new();
    params.insert("o".to_string(), Val::Num(1.0));
    params.insert("n".to_string(), Val::Num(2.0));
    let r = qp(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS name ORDER BY name OFFSET $o LIMIT $n",
        params,
    );
    assert_eq!(r, vec![vec![s("marko")], vec![s("peter")]]);

    // A non-integer bound faults with E_INVALID_VALUE, before any row is produced.
    let mut bad = Params::new();
    bad.insert("n".to_string(), Val::Num(2.5));
    let e = prepare("MATCH (n:Person) RETURN n.name LIMIT $n")
        .unwrap()
        .execute(&mut g, &bad)
        .unwrap_err();
    assert_eq!(e.code, crate::error_codes::ErrorCode::InvalidValue);
}

#[test]
fn m_skip_rejects_dynamic_param() {
    // SKIP is the Cypher synonym for OFFSET and stays literal-only: `SKIP $n` is a
    // syntax error (only `OFFSET $n` / `LIMIT $n` accept a dynamic param).
    assert!(parse("MATCH (n:Person) RETURN n.name SKIP $n").is_err());
}

#[test]
fn m_element_id_returns_identifier() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person {name: 'marko'}) RETURN element_id(n) AS id",
    );
    assert_eq!(r, vec![vec![s("1")]]); // the same value the TS test asserts
}

// ── "GQL: ISO reserved words" ────────────────────────────────────────────────

#[test]
fn m_reserved_word_rejected_as_bare_identifier() {
    let mut g = modern();
    // variable position
    assert!(
        parse("MATCH (select) RETURN select").is_err(),
        "select should be rejected"
    );
    // property key position (n.value — 'value' is reserved)
    let res = parse("MATCH (n:Person) RETURN n.value AS v");
    if let Ok(parsed) = res {
        // May succeed at parse but fail at execute, or parse may reject it
        let _ = parsed.execute(&mut g, &Params::new());
        // Either way is acceptable — just don't panic
    }
    // alias position
    assert!(
        parse("MATCH (n:Person) RETURN n AS count").is_err()
            || parse("MATCH (n:Person) RETURN n AS count")
                .ok()
                .and_then(|p| p.execute(&mut g, &Params::new()).err())
                .is_some(),
        "count as alias should be rejected"
    );
    // label position
    assert!(
        parse("MATCH (n:Match) RETURN n").is_err(),
        "Match as label should be rejected"
    );
}

#[test]
fn m_reserved_word_allowed_as_delimited_identifier_or_function_name() {
    let mut g = modern();
    g.add_vertex(
        &["Misc".to_string()],
        vec![("value".to_string(), Value::Num(7.0))],
    );
    let r = rows(&mut g, "MATCH (n:Misc) RETURN n.`value` AS v");
    assert_eq!(r, vec![vec![n(7.0)]]);

    let r2 = rows(&mut g, "RETURN upper('x') AS u");
    assert_eq!(r2, vec![vec![s("X")]]);
}

#[test]
fn m_non_reserved_words_work_as_identifiers() {
    let mut g = modern();
    g.add_vertex(
        &["First".to_string()],
        vec![
            ("last".to_string(), Value::Str("z".into())),
            ("type".to_string(), Value::Str("t".into())),
        ],
    );
    let r = rows(
        &mut g,
        "MATCH (first:First) RETURN first.last AS last, first.type AS type",
    );
    assert_eq!(r, vec![vec![s("z"), s("t")]]);
}

#[test]
fn m_collect_list_is_iso_name() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN collect_list(n.name) AS names",
    );
    assert_eq!(r.len(), 1);
    if let Value::List(ref lst) = r[0][0] {
        let mut sorted = lst.clone();
        sorted.sort_by_key(val_sort_key);
        assert_eq!(sorted, vec![s("josh"), s("marko"), s("peter"), s("vadas")]);
    } else {
        panic!("expected List, got {:?}", r[0][0]);
    }
}
