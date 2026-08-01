//! The vectorized comparison kernels must agree with the scalar evaluator.
//!
//! `str_eq_vec` and `temporal_cmp_vec` exist purely to go faster: they answer a
//! `WHERE` over a typed column without rebuilding a binding per row. That makes
//! them invisible when right and silent when wrong — a kernel that disagrees
//! with the scalar path returns wrong rows with no error anywhere.
//!
//! So every test here pins a spelling that routes through a kernel against one
//! that cannot, and asserts the rows are identical. The literal form is the
//! reference: it has always taken the fast path, and the parameterized form is
//! what was added to it.

use super::eval::{Params, Val};
use super::parse;
use crate::graph::{Graph, Value};

/// Users with string names, one with the property absent, plus an edge property
/// so the edge side of the kernel is covered too.
fn users() -> Graph {
    let mut lines: Vec<String> = vec![
        r#"{"type":"node","id":"u0","labels":["User"],"properties":{"name":"ann","tier":1}}"#
            .into(),
        r#"{"type":"node","id":"u1","labels":["User"],"properties":{"name":"bob","tier":2}}"#
            .into(),
        r#"{"type":"node","id":"u2","labels":["User"],"properties":{"name":"ann","tier":3}}"#
            .into(),
        // No `name` at all — the three-valued case the kernel must not mistake
        // for a non-match.
        r#"{"type":"node","id":"u3","labels":["User"],"properties":{"tier":4}}"#.into(),
        // Named "1" as a STRING. Cross-type `u.name = 1` must not match it, so a
        // kernel that stringified a numeric param would be caught here.
        r#"{"type":"node","id":"u4","labels":["User"],"properties":{"name":"1","tier":5}}"#.into(),
    ];
    lines.push(
        r#"{"type":"edge","id":"e0","labels":["R"],"from":"u0","to":"u1","properties":{"kind":"x"}}"#
            .into(),
    );
    lines.push(
        r#"{"type":"edge","id":"e1","labels":["R"],"from":"u1","to":"u2","properties":{"kind":"y"}}"#
            .into(),
    );

    crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes")
}

fn rows(g: &mut Graph, query: &str, params: &Params) -> Vec<Vec<Value>> {
    let mut out: Vec<Vec<Value>> = parse(query)
        .unwrap_or_else(|e| panic!("parse error for `{query}`: {e}"))
        .execute(g, params)
        .unwrap_or_else(|e| panic!("exec error for `{query}`: {e}"))
        .rows()
        .map(<[Value]>::to_vec)
        .collect();

    out.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    out
}

fn one(key: &str, v: Val) -> Params {
    let mut p = Params::new();

    p.insert(key.to_string(), v);
    p
}

#[test]
fn a_string_param_equality_matches_the_literal_spelling() {
    let mut g = users();
    let by_param = rows(
        &mut g,
        "MATCH (u:User) WHERE u.name = $n RETURN u.tier AS t",
        &one("n", Val::Str("ann".into())),
    );

    assert_eq!(
        by_param,
        rows(
            &mut g,
            "MATCH (u:User) WHERE u.name = 'ann' RETURN u.tier AS t",
            &Params::new()
        )
    );
    assert_eq!(by_param.len(), 2);
}

#[test]
fn a_reversed_string_param_equality_matches_too() {
    let mut g = users();

    assert_eq!(
        rows(
            &mut g,
            "MATCH (u:User) WHERE $n = u.name RETURN u.tier AS t",
            &one("n", Val::Str("ann".into()))
        ),
        rows(
            &mut g,
            "MATCH (u:User) WHERE u.name = 'ann' RETURN u.tier AS t",
            &Params::new()
        )
    );
}

#[test]
fn string_param_inequality_excludes_the_absent_row() {
    let mut g = users();

    // `u3` has no `name`, so `u.name <> 'ann'` is UNKNOWN for it, not true — the
    // kernel reports validity separately from the comparison and a version that
    // conflated them would return u3 here.
    let by_param = rows(
        &mut g,
        "MATCH (u:User) WHERE u.name <> $n RETURN u.tier AS t",
        &one("n", Val::Str("ann".into())),
    );

    assert_eq!(
        by_param,
        rows(
            &mut g,
            "MATCH (u:User) WHERE u.name <> 'ann' RETURN u.tier AS t",
            &Params::new()
        )
    );
    // bob and the user named "1" — NOT the one with no `name` at all.
    assert_eq!(by_param, vec![vec![Value::Num(2.0)], vec![Value::Num(5.0)]]);
}

#[test]
fn a_param_string_that_was_never_interned_matches_nothing() {
    let mut g = users();

    // The interned-id path resolves the scalar to a dictionary id. A value no
    // element ever carried has no id, which must mean "matches nothing" rather
    // than "matches everything".
    assert!(rows(
        &mut g,
        "MATCH (u:User) WHERE u.name = $n RETURN u.tier AS t",
        &one("n", Val::Str("nobody".into()))
    )
    .is_empty());
}

#[test]
fn a_non_string_param_does_not_take_the_string_path() {
    let mut g = users();

    // A number compared against a string column is cross-type. It must produce
    // the same answer as the equivalent literal spelling, NOT be handed to a
    // kernel that only answers string-vs-string. The fixture holds a user whose
    // name is the STRING "1", so a kernel that stringified the param would match
    // it and this would diverge — without that row both sides are empty and the
    // test proves nothing.
    let by_param = rows(
        &mut g,
        "MATCH (u:User) WHERE u.name = $n RETURN u.tier AS t",
        &one("n", Val::Num(1.0)),
    );

    assert_eq!(
        by_param,
        rows(
            &mut g,
            "MATCH (u:User) WHERE u.name = 1 RETURN u.tier AS t",
            &Params::new()
        )
    );
    assert!(by_param.is_empty(), "cross-type equality must not match");
}

#[test]
fn a_null_param_compares_as_unknown() {
    let mut g = users();

    let by_param = rows(
        &mut g,
        "MATCH (u:User) WHERE u.name = $n RETURN u.tier AS t",
        &one("n", Val::Null),
    );

    assert_eq!(
        by_param,
        rows(
            &mut g,
            "MATCH (u:User) WHERE u.name = null RETURN u.tier AS t",
            &Params::new()
        )
    );
    assert!(by_param.is_empty());
}

#[test]
fn a_string_param_works_on_an_edge_property() {
    let mut g = users();

    assert_eq!(
        rows(
            &mut g,
            "MATCH ()-[e:R]->() WHERE e.kind = $k RETURN e.kind AS k",
            &one("k", Val::Str("x".into()))
        ),
        rows(
            &mut g,
            "MATCH ()-[e:R]->() WHERE e.kind = 'x' RETURN e.kind AS k",
            &Params::new()
        )
    );
}

#[test]
fn a_string_param_inside_a_conjunction_still_agrees() {
    let mut g = users();

    // The kernel is reached per comparison, so it has to compose with the
    // surrounding boolean structure rather than only working standalone.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (u:User) WHERE u.name = $n AND u.tier > 1 RETURN u.tier AS t",
            &one("n", Val::Str("ann".into()))
        ),
        rows(
            &mut g,
            "MATCH (u:User) WHERE u.name = 'ann' AND u.tier > 1 RETURN u.tier AS t",
            &Params::new()
        )
    );
}

#[test]
fn the_same_prepared_plan_answers_two_different_params() {
    let mut g = users();
    let plan = super::prepare("MATCH (u:User) WHERE u.name = $n RETURN u.tier AS t").unwrap();
    let count = |p: &Params, g: &mut Graph| plan.execute(g, p).unwrap().rows().count();

    // A cached plan must not capture the first param's resolved dictionary id.
    assert_eq!(count(&one("n", Val::Str("ann".into())), &mut g), 2);
    assert_eq!(count(&one("n", Val::Str("bob".into())), &mut g), 1);
    assert_eq!(count(&one("n", Val::Str("ann".into())), &mut g), 2);
}

#[test]
fn a_param_string_interned_only_after_planning_still_matches() {
    let mut g = users();
    let plan = super::prepare("MATCH (u:User) WHERE u.name = $n RETURN u.tier AS t").unwrap();
    let p = one("n", Val::Str("zoe".into()));

    assert_eq!(plan.execute(&mut g, &p).unwrap().rows().count(), 0);

    // Interning happens on write, so a value with no id at plan time can have one
    // by the next execution. Resolving the id once and caching it would answer 0
    // here forever.
    parse("INSERT (:User {name: 'zoe', tier: 9})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap();

    assert_eq!(plan.execute(&mut g, &p).unwrap().rows().count(), 1);
}
