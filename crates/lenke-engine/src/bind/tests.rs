use super::*;
use crate::exec::run;
use crate::gql::{parse_prepared, parse_with_params};
use crate::store::{Builder, Store};
use std::sync::Arc;

fn s(x: &str) -> Value {
    Value::Str(Arc::from(x))
}
fn n(x: f64) -> Value {
    Value::Num(x)
}
fn people() -> Store {
    let mut b = Builder::default();
    b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
    b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
    b.build()
}

/// Parse-in-prepared-mode + bind must yield the EXACT same plan as substituting
/// the value at parse time — so a prepared run is byte-identical to a direct one.
fn prepared_equals_direct(query: &str, params: &[(String, Value)]) {
    let mut prepared = parse_prepared(query).unwrap();
    bind_params(&mut prepared, params).unwrap();
    let direct = parse_with_params(query, params).unwrap();
    assert_eq!(
        format!("{prepared:?}"),
        format!("{direct:?}"),
        "prepared+bind != direct for `{query}`"
    );
}

#[test]
fn bound_plan_matches_direct_substitution() {
    prepared_equals_direct(
        "MATCH (n:Person) WHERE n.age = $a RETURN n.name AS x",
        &[("a".into(), n(30.0))],
    );
    // Inline prop: `{k: $p}` binds via the filter path, same plan as the literal.
    prepared_equals_direct(
        "MATCH (n:Person {name: $nm}) RETURN n.age AS a",
        &[("nm".into(), s("alice"))],
    );
    // A param nested in arithmetic / function position.
    prepared_equals_direct(
        "MATCH (n:Person) WHERE n.age > $a + 1 RETURN n.name AS x",
        &[("a".into(), n(20.0))],
    );
}

#[test]
fn one_prepared_plan_reused_with_different_params() {
    let store = people();
    let template = parse_prepared("MATCH (n:Person) WHERE n.name = $nm RETURN n.age AS a").unwrap();
    let age_for = |nm: &str| {
        let mut plan = template.clone(); // parse once, bind many
        bind_params(&mut plan, &[("nm".into(), s(nm))]).unwrap();
        let rows = run(&plan, &store);
        assert_eq!(rows.rows.len(), 1, "{nm}");
        format!("{:?}", rows.rows[0][0])
    };
    assert_eq!(age_for("alice"), format!("{:?}", n(30.0)));
    assert_eq!(age_for("bob"), format!("{:?}", n(25.0)));
}

#[test]
fn missing_param_is_an_error() {
    let mut plan = parse_prepared("MATCH (n) WHERE n.k = $x RETURN n").unwrap();
    let err = bind_params(&mut plan, &[]).unwrap_err();
    assert!(err.contains("$x"), "unhelpful error: {err}");
}

#[test]
fn binding_is_idempotent_across_clones() {
    // Binding a clone must not disturb the template (the reuse invariant).
    let template = parse_prepared("MATCH (n:Person) WHERE n.age = $a RETURN n").unwrap();
    let before = format!("{template:?}");
    let mut once = template.clone();
    bind_params(&mut once, &[("a".into(), n(1.0))]).unwrap();
    assert_eq!(
        format!("{template:?}"),
        before,
        "template mutated by a bind"
    );
}
