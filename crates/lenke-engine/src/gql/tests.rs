
use crate::exec::{run, Rows};
use crate::store::{Builder, Store};
use crate::value::Value;
use std::sync::Arc;

fn s(x: &str) -> Value {
    Value::Str(Arc::from(x))
}
fn n(x: f64) -> Value {
    Value::Num(x)
}

/// Run a GQL write through the same path real usage does (optimize then execute).
#[cfg(test)]
fn exec_gql(st: &mut Store, sql: &str) {
    let p = crate::opt::optimize_indexed(super::parse(sql).unwrap(), st);
    crate::exec::execute(&p, st).unwrap();
}

/// The three property states — present-value / present-null / absent — must stay
/// distinguishable, read correctly, and survive transitions. Pins the semantics for
/// the typed-column `nulls` bit: `has_prop` distinguishes present-null (true) from
/// absent (false); both read NULL; a value written over a null restores it; REMOVE of
/// a present-null makes it absent.
#[test]
fn present_null_tristate_and_transitions() {
    let mut st = Builder::default().build();
    exec_gql(
        &mut st,
        "INSERT (:P {id:'v', age: 10}), (:P {id:'z', age: 20}), (:P {id:'a', age: 30})",
    );
    exec_gql(&mut st, "MATCH (n:P {id:'z'}) SET n.age = null"); // z: present-null
    exec_gql(&mut st, "MATCH (n:P {id:'a'}) REMOVE n.age"); // a: absent
                                                            // ids by creation order: v=0, z=1, a=2.
    assert!(st.has_prop(0, "age"));
    assert!(matches!(st.prop(0, "age"), Value::Num(x) if x == 10.0));
    assert!(st.has_prop(1, "age"), "present-null IS present");
    assert!(st.prop(1, "age").is_null(), "present-null reads NULL");
    assert!(!st.has_prop(2, "age"), "REMOVEd is absent");
    assert!(st.prop(2, "age").is_null());

    // A real value written over a present-null restores both value and presence.
    exec_gql(&mut st, "MATCH (n:P {id:'z'}) SET n.age = 99");
    assert!(st.has_prop(1, "age"));
    assert!(matches!(st.prop(1, "age"), Value::Num(x) if x == 99.0));
    // REMOVE of a (re-nulled) present-null yields absent, distinct from the stored null.
    exec_gql(&mut st, "MATCH (n:P {id:'z'}) SET n.age = null");
    assert!(st.has_prop(1, "age"));
    exec_gql(&mut st, "MATCH (n:P {id:'z'}) REMOVE n.age");
    assert!(!st.has_prop(1, "age"), "REMOVE of a present-null -> absent");
}

/// A present-null must survive an NDJSON round-trip AS a present-null — not collapse
/// to absent (which would silently turn `has('k')` false). The distinct-from-absent
/// invariant has to hold through serialization, whatever column form backs it.
#[test]
fn present_null_survives_ndjson_roundtrip() {
    let mut st = Builder::default().build();
    exec_gql(
        &mut st,
        "INSERT (:P {id:'v', age: 10}), (:P {id:'z', age: 20}), (:P {id:'a', age: 30})",
    );
    exec_gql(&mut st, "MATCH (n:P {id:'z'}) SET n.age = null"); // present-null
    exec_gql(&mut st, "MATCH (n:P {id:'a'}) REMOVE n.age"); // absent

    let nd = crate::ndjson::to_ndjson(&st);
    // The present-null serializes as an explicit null; the absent node omits `age`.
    assert!(
        nd.contains("\"age\":null"),
        "present-null emits an explicit null: {nd}"
    );
    let st2 = crate::ndjson::from_ndjson(&nd).unwrap();
    // Dense ids are assigned in file order, which `to_ndjson` emits in id order.
    assert!(matches!(st2.prop(0, "age"), Value::Num(x) if x == 10.0));
    assert!(
        st2.has_prop(1, "age") && st2.prop(1, "age").is_null(),
        "present-null round-trips as a present null, not absent"
    );
    assert!(!st2.has_prop(2, "age"), "absent round-trips as absent");
}

/// Rollback restores the exact state a `SET k = null` overwrote — the value AND its
/// presence come back. And rolling back a `SET value` over a present-null restores the
/// null. The undo log must treat the null write like any other cell write.
#[test]
fn present_null_survives_rollback() {
    let mut st = Builder::default().build();
    exec_gql(&mut st, "INSERT (:P {id:'v', age: 10})"); // node 0, present value

    st.begin();
    st.set_prop(0, "age", Value::Null); // present-null inside the txn
    assert!(st.prop(0, "age").is_null());
    st.rollback();
    assert!(
        matches!(st.prop(0, "age"), Value::Num(x) if x == 10.0),
        "rollback restores the value a SET null overwrote"
    );

    // The reverse: a present-null, then SET a value, rolled back -> the null returns.
    st.set_prop(0, "age", Value::Null);
    assert!(st.has_prop(0, "age") && st.prop(0, "age").is_null());
    st.begin();
    st.set_prop(0, "age", Value::Num(7.0));
    st.rollback();
    assert!(
        st.has_prop(0, "age") && st.prop(0, "age").is_null(),
        "rollback restores a present null"
    );
}

/// A present-null interacts with constraints like core: a REQUIRED key must be
/// present AND non-null, so `SET k = null` on it is rejected (a present-null is not a
/// value); a UNIQUE key EXEMPTS nulls, so two present-nulls don't collide. Verifies
/// the write-path enforcement end to end.
#[test]
fn present_null_and_constraints() {
    let try_gql = |st: &mut Store, sql: &str| -> Result<(), String> {
        let p = crate::opt::optimize_indexed(super::parse(sql).unwrap(), st);
        crate::exec::execute(&p, st).map(|_| ())
    };

    // REQUIRED: setting the key to null is rejected; the prior value is rolled back.
    let mut st = Builder::default().build();
    exec_gql(&mut st, "INSERT (:User {id:'a', email: 'a@x'})");
    st.create_required_constraint("User", "email").unwrap();
    assert!(
        try_gql(&mut st, "MATCH (n:User {id:'a'}) SET n.email = null").is_err(),
        "SET a required key to null must be rejected (present-null is not 'present, non-null')"
    );
    assert!(
        matches!(st.prop(0, "email"), Value::Str(v) if &*v == "a@x"),
        "the rejected write rolled back — the value survives"
    );

    // UNIQUE: nulls are exempt — two present-nulls under a unique key are allowed.
    let mut st2 = Builder::default().build();
    exec_gql(
        &mut st2,
        "INSERT (:U {id:'b', k: 'v1'}), (:U {id:'c', k: 'v2'})",
    );
    st2.create_unique_constraint("U", &["k"]).unwrap();
    assert!(try_gql(&mut st2, "MATCH (n:U {id:'b'}) SET n.k = null").is_ok());
    assert!(
        try_gql(&mut st2, "MATCH (n:U {id:'c'}) SET n.k = null").is_ok(),
        "two present-nulls do NOT violate a unique constraint (nulls exempt)"
    );
}

/// `SET n:Label` / `REMOVE n:Label` (and the `IS` spelling) mutate a node's
/// label set; the change is re-checked against the constraints on the new label
/// — adding `:Acct` to a node with no email violates Acct's required-email, and
/// the whole statement rolls back.
#[test]
fn set_and_remove_label() {
    let try_gql = |st: &mut Store, sql: &str| -> Result<(), String> {
        let p = crate::opt::optimize_indexed(super::parse(sql).unwrap(), st);
        crate::exec::execute(&p, st).map(|_| ())
    };

    let mut st = Builder::default().build();
    exec_gql(&mut st, "INSERT (:Person {name: 'P'})");

    // SET adds a label (the original label stays).
    exec_gql(&mut st, "MATCH (p:Person {name: 'P'}) SET p:Staff");
    assert!(st.is_labeled(0, "Staff"));
    assert!(st.is_labeled(0, "Person"));
    // A repeat SET is idempotent, and the node is now seekable by the new label.
    exec_gql(&mut st, "MATCH (p:Staff) SET p:Staff");
    assert_eq!(st.nodes_with_label("Staff").len(), 1);

    // REMOVE (the `IS` spelling) strips it.
    exec_gql(&mut st, "MATCH (p:Person {name: 'P'}) REMOVE p IS Staff");
    assert!(!st.is_labeled(0, "Staff"));
    assert!(st.nodes_with_label("Staff").is_empty());

    // Adding a label whose required constraint the node cannot satisfy is
    // rejected, and the label add rolls back.
    st.create_required_constraint("Acct", "email").unwrap();
    assert!(try_gql(&mut st, "MATCH (p:Person {name: 'P'}) SET p:Acct").is_err());
    assert!(
        !st.is_labeled(0, "Acct"),
        "the rejected label add rolled back"
    );
}

/// Characterization of PRESENT-NULL semantics — the executable spec a future
/// typed-column `nulls` bitset must preserve. Today a stored null forces the column
/// to `Gen` (the correct oracle); every assertion here must STILL hold once a null
/// keeps the column typed. The landmine these guard against: a typed fast-path that
/// reads `data[i]` when `present[i]` — without checking the null bit — would surface
/// the placeholder (0.0), so `min`/`= 0`/`count` would silently go wrong.
#[test]
fn present_null_semantics_characterization() {
    use crate::exec::execute;
    let q = |st: &mut Store, sql: &str| {
        let p = crate::opt::optimize_indexed(super::parse(sql).unwrap(), st);
        execute(&p, st).unwrap();
    };
    let mut st = Builder::default().build();
    // ids 0..3 by creation order: a=0 b=1 c=2 d=3.
    q(
            &mut st,
            "INSERT (:P {id:'a', age: 10}), (:P {id:'b', age: 20}), (:P {id:'c', age: 99}), (:P {id:'d', age: 0})",
        );
    q(&mut st, "MATCH (n:P {id:'c'}) SET n.age = null"); // c: PRESENT null
    q(&mut st, "MATCH (n:P {id:'d'}) REMOVE n.age"); // d: ABSENT

    // --- presence: present-null IS present and distinct from absent ---
    assert!(
        st.has_prop(2, "age"),
        "a stored null is PRESENT (has('age') true)"
    );
    assert!(!st.has_prop(3, "age"), "a REMOVEd value is ABSENT");
    assert!(
        st.prop(2, "age").is_null(),
        "present-null reads NULL, not 99/0"
    );
    assert!(st.prop(3, "age").is_null(), "absent reads NULL");

    // --- query results: a present-null must NOT surface the placeholder 0.0 ---
    let one = |st: &Store, sql: &str| -> f64 {
        let p = crate::opt::optimize_indexed(super::parse(sql).unwrap(), st);
        match &run(&p, st).rows[0][0] {
            Value::Num(x) => *x,
            other => panic!("expected a number, got {other:?} for `{sql}`"),
        }
    };
    // count(expr) skips BOTH present-null and absent → only a,b.
    assert_eq!(one(&st, "MATCH (n:P) RETURN count(n.age) AS c"), 2.0);
    assert_eq!(one(&st, "MATCH (n:P) RETURN count(*) AS c"), 4.0);
    // min/sum skip nulls: min is 10, NOT 0 (the placeholder). This is THE leak catcher.
    assert_eq!(one(&st, "MATCH (n:P) RETURN min(n.age) AS m"), 10.0);
    assert_eq!(one(&st, "MATCH (n:P) RETURN sum(n.age) AS s"), 30.0);
    // equality against the placeholder value must NOT match a present-null.
    assert_eq!(
        one(&st, "MATCH (n:P) WHERE n.age = 0 RETURN count(*) AS c"),
        0.0
    );
    // IS NULL is true for BOTH present-null and absent (3VL); IS NOT NULL neither.
    assert_eq!(
        one(&st, "MATCH (n:P) WHERE n.age IS NULL RETURN count(*) AS c"),
        2.0
    );
    assert_eq!(
        one(
            &st,
            "MATCH (n:P) WHERE n.age IS NOT NULL RETURN count(*) AS c"
        ),
        2.0
    );

    // --- an index seek must also exclude the present-null (not seek it under 0) ---
    st.create_index("age");
    let idx_plan = crate::opt::optimize_indexed(
        super::parse("MATCH (n:P) WHERE n.age = 0 RETURN count(*) AS c").unwrap(),
        &st,
    );
    match &run(&idx_plan, &st).rows[0][0] {
        Value::Num(x) => assert_eq!(*x, 0.0, "indexed `= 0` must not return a present-null"),
        other => panic!("{other:?}"),
    }
}

/// The typed-vs-`Gen` EQUIVALENCE harness — the centerpiece guard for any change to
/// how a typed column stores values (e.g. adding a `nulls` bit). `Gen` is the boxed,
/// always-correct oracle; a typed column carrying the SAME values must produce
/// identical results under EVERY query shape. Running the battery below over a
/// column and its `force_gen` twin catches any typed fast-path (vectorized filter,
/// index seek, aggregate, string search, group/distinct) that reads a value wrong —
/// exactly the class of bug a mishandled null bit would introduce. Extend the fixture
/// with `SET k = null` once the typed column can hold a present null.
#[test]
fn typed_and_gen_columns_agree() {
    use crate::exec::execute;
    let exec_gql = |st: &mut Store, sql: &str| {
        let p = crate::opt::optimize_indexed(super::parse(sql).unwrap(), st);
        execute(&p, st).unwrap();
    };
    let build = |exec_gql: &dyn Fn(&mut Store, &str)| -> Store {
        let mut st = Builder::default().build();
        exec_gql(
            &mut st,
            "INSERT (:P {id:'a', age: 10, city:'oslo', vip: true}), \
                 (:P {id:'b', age: 20, city:'bergen', vip: false}), \
                 (:P {id:'c', age: 30, city:'oslo', vip: true}), \
                 (:P {id:'d', age: 20, city:'oslo', vip: false})",
        );
        // A STORED PRESENT NULL on each typed column — with the `nulls` side map the
        // column stays typed, so the typed-vs-Gen agreement must hold WITH nulls in
        // the mix (a present-null reads NULL, is skipped by aggregates, doesn't match
        // a value, but IS present for `PROPERTY_EXISTS`).
        exec_gql(&mut st, "MATCH (n:P {id:'c'}) SET n.age = null");
        exec_gql(&mut st, "MATCH (n:P {id:'b'}) SET n.city = null");
        exec_gql(&mut st, "MATCH (n:P {id:'d'}) SET n.vip = null");
        st.create_index("age");
        st.create_index("city");
        st
    };
    // One query per typed read fast-path the change could break.
    let queries = [
        "MATCH (n:P) RETURN n.age AS x, n.id AS t ORDER BY x, t", // materialize + order
        "MATCH (n:P) WHERE n.age > 15 RETURN n.id AS x ORDER BY x", // vectorized compare
        "MATCH (n:P) WHERE n.age = 20 RETURN n.id AS x ORDER BY x", // numeric index seek
        "MATCH (n:P) WHERE n.city = 'oslo' RETURN n.id AS x ORDER BY x", // string index seek
        "MATCH (n:P) WHERE n.city STARTS WITH 'os' RETURN n.id AS x ORDER BY x", // string search
        "MATCH (n:P) WHERE n.vip = true RETURN n.id AS x ORDER BY x", // bool
        "MATCH (n:P) RETURN min(n.age) AS a, max(n.age) AS b, sum(n.age) AS c, count(n.age) AS d",
        "MATCH (n:P) RETURN n.city AS x, count(*) AS c GROUP BY n.city ORDER BY x", // group by
        "MATCH (n:P) RETURN DISTINCT n.age AS x ORDER BY x",                        // distinct
        "MATCH (n:P) WHERE n.age IS NULL RETURN n.id AS x ORDER BY x", // present-null + absent
        "MATCH (n:P) WHERE PROPERTY_EXISTS(n, age) RETURN n.id AS x ORDER BY x", // presence
    ];
    for keys in [
        &["age"][..],
        &["city"][..],
        &["vip"][..],
        &["age", "city", "vip"][..],
    ] {
        let typed = build(&exec_gql);
        let mut boxed = build(&exec_gql);
        for k in keys {
            boxed.force_gen(k);
        }
        let repr = |r: &Rows| -> Vec<Vec<String>> {
            r.rows
                .iter()
                .map(|row| row.iter().map(|c| format!("{c:?}")).collect())
                .collect()
        };
        for query in queries {
            let tp = crate::opt::optimize_indexed(super::parse(query).unwrap(), &typed);
            let bp = crate::opt::optimize_indexed(super::parse(query).unwrap(), &boxed);
            assert_eq!(
                repr(&run(&tp, &typed)),
                repr(&run(&bp, &boxed)),
                "typed vs Gen diverged on `{query}` (Gen-forced keys: {keys:?})"
            );
        }
    }
}

fn social() -> Store {
    let mut b = Builder::default();
    let a = b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
    let bob = b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
    let c = b.node(&["Person"], &[("name", s("carol")), ("age", n(40.0))]);
    let proj = b.node(&["Project"], &[("name", s("graphdb"))]);
    b.edge(a, bob, "KNOWS");
    b.edge(a, c, "KNOWS");
    b.edge(bob, c, "KNOWS");
    b.edge(a, proj, "WORKS_ON");
    b.build()
}

/// Rows as a sorted multiset of `col=value;` strings — order-independent
/// comparison, since result order is unspecified without ORDER BY.
fn bag(rows: &Rows) -> Vec<String> {
    let mut out: Vec<String> = rows
        .rows
        .iter()
        .map(|r| {
            rows.names
                .iter()
                .zip(r)
                .map(|(k, v)| format!("{k}={v:?};"))
                .collect::<String>()
        })
        .collect();
    out.sort();
    out
}

/// The parser is correct iff parse->run reproduces the hand-built plan.
fn assert_same(query: &str, hand: &crate::ir::Plan, store: &Store) {
    let parsed = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    assert_eq!(
        bag(&run(&parsed, store)),
        bag(&run(hand, store)),
        "parsed plan differs for `{query}`"
    );
}

/// A `$name` param query returns the SAME rows as the equivalent inlined-literal
/// query, across every position params appear (WHERE, inline prop, IN, UNWIND,
/// LIMIT). The value is typed, so it is never spliced into the query text.
#[test]
fn query_params_match_inlined_literals() {
    let store = social();
    let same = |pq: &str, params: &[(String, Value)], lit: &str| {
        let a = bag(&run(&super::parse_with_params(pq, params).unwrap(), &store));
        let b = bag(&run(&super::parse(lit).unwrap(), &store));
        assert_eq!(a, b, "`{pq}` vs `{lit}`");
    };
    same(
        "MATCH (n:Person) WHERE n.age = $a RETURN n.name AS x",
        &[("a".into(), n(30.0))],
        "MATCH (n:Person) WHERE n.age = 30 RETURN n.name AS x",
    );
    same(
        "MATCH (n:Person {name: $nm}) RETURN n.age AS a",
        &[("nm".into(), s("alice"))],
        "MATCH (n:Person {name: 'alice'}) RETURN n.age AS a",
    );
    same(
        "MATCH (n:Person) WHERE n.name IN $names RETURN n.name AS x",
        &[("names".into(), Value::List(vec![s("alice"), s("carol")]))],
        "MATCH (n:Person) WHERE n.name IN ['alice', 'carol'] RETURN n.name AS x",
    );
    same(
        "MATCH (n:Person) RETURN n.name AS x ORDER BY n.name LIMIT $k",
        &[("k".into(), n(2.0))],
        "MATCH (n:Person) RETURN n.name AS x ORDER BY n.name LIMIT 2",
    );
}

/// The ISO niladic now-functions read the injected `$__now` clock: the timestamp
/// forms as a DATETIME, `current_date` as a DATE, `current_time`/`local_time` as a
/// time-of-day; an unsupplied clock → null. The engine never reads a wall clock.
#[test]
fn niladic_now_functions_read_injected_clock() {
    use crate::temporal::Temporal;
    let store = Builder::default().build();
    let run_p = |q: &str, params: &[(String, Value)]| -> Value {
        let plan = super::parse_with_params(q, params).unwrap();
        run(&plan, &store).rows[0][0].clone()
    };
    let now = Value::Temporal(Temporal::parse("datetime", "2026-07-12T10:30:45").unwrap());
    let p = vec![("__now".to_string(), now)];
    assert!(matches!(
        run_p("RETURN current_timestamp AS t", &p),
        Value::Temporal(Temporal::DateTime(_))
    ));
    assert!(matches!(
        run_p("RETURN local_timestamp AS t", &p),
        Value::Temporal(Temporal::DateTime(_))
    ));
    assert!(matches!(
        run_p("RETURN current_date AS t", &p),
        Value::Temporal(Temporal::Date(_))
    ));
    assert!(matches!(
        run_p("RETURN current_time AS t", &p),
        Value::Temporal(Temporal::Time(_))
    ));
    // `local_time()` (empty parens) is the niladic form; `local_time('…')` the ctor.
    assert!(matches!(
        run_p("RETURN local_time() AS t", &p),
        Value::Temporal(Temporal::Time(_))
    ));
    assert!(matches!(
        run_p("RETURN local_time('13:47:09') AS t", &[]),
        Value::Temporal(Temporal::Time(_))
    ));
    // No clock supplied → null.
    assert!(matches!(
        run_p("RETURN current_timestamp AS t", &[]),
        Value::Null
    ));
}

/// For a scalar / inline-prop param, substitution produces the EXACT same plan as
/// inlining the literal — so the param path is byte-identical to the literal path
/// by construction, index seeding included. (Compared via derived `Debug`; `Plan`
/// is not `PartialEq`.)
#[test]
fn scalar_param_yields_identical_plan() {
    let same_plan = |pq: &str, params: &[(String, Value)], lit: &str| {
        let a = format!("{:?}", super::parse_with_params(pq, params).unwrap());
        let b = format!("{:?}", super::parse(lit).unwrap());
        assert_eq!(a, b, "param plan != literal plan for `{pq}`");
    };
    same_plan(
        "MATCH (n:Person) WHERE n.age = $a RETURN n",
        &[("a".into(), n(30.0))],
        "MATCH (n:Person) WHERE n.age = 30 RETURN n",
    );
    same_plan(
        "MATCH (n:Person {name: $nm}) RETURN n",
        &[("nm".into(), s("alice"))],
        "MATCH (n:Person {name: 'alice'}) RETURN n",
    );
}

#[test]
fn unbound_param_is_a_parse_error() {
    let err = super::parse_with_params("MATCH (n) WHERE n.k = $missing RETURN n", &[]).unwrap_err();
    // A supplied-but-unbound `$param` carries the E_MISSING_PARAMETER wire code (the
    // FFI routes the prefix), not a bare syntax error — matching TS.
    assert!(
        err.starts_with("E_MISSING_PARAMETER:"),
        "expected the E_MISSING_PARAMETER prefix, got: {err}"
    );
    assert!(err.contains("missing"), "unhelpful error: {err}");
}

/// A well-formed JSON param whose VALUE is outside the param model — a malformed
/// tagged temporal (`{"@date":"nope"}`) — is a VALUE error, not a JSON-syntax one,
/// so `params_from_obj` rejects it (the FFI maps this to E_INVALID_VALUE).
#[test]
fn malformed_tagged_temporal_param_is_a_value_error() {
    let obj = crate::ndjson::parse_json(r#"{"x":{"@date":"nope"}}"#).unwrap();
    assert!(crate::ndjson::params_from_obj(&obj).is_err());
}

/// A string param carrying query-like text is a plain VALUE — matched literally,
/// never parsed as syntax. This is the safety guarantee: no injection surface.
#[test]
fn string_param_is_a_value_not_injected_syntax() {
    let store = social();
    let rows = bag(&run(
        &super::parse_with_params(
            "MATCH (n:Person) WHERE n.name = $nm RETURN n.name AS x",
            &[("nm".into(), s("alice' OR '1'='1"))],
        )
        .unwrap(),
        &store,
    ));
    assert!(
        rows.is_empty(),
        "injection-looking string matched: {rows:?}"
    );
}

/// Inline node-property maps `(n:L {k: v, …})` are a match filter — the same
/// rows as the `WHERE` spelling, on the seed node AND a hop's landing node.
#[test]
fn inline_property_maps_match_where() {
    let store = social();
    let same = |inline: &str, wher: &str| {
        let a = bag(&run(&super::parse(inline).unwrap(), &store));
        let b = bag(&run(&super::parse(wher).unwrap(), &store));
        assert_eq!(a, b, "`{inline}` vs `{wher}`");
    };
    // Seed node, single and multi-property.
    same(
        "MATCH (n:Person {name: 'alice'}) RETURN n.age AS a",
        "MATCH (n:Person) WHERE n.name = 'alice' RETURN n.age AS a",
    );
    same(
        "MATCH (n:Person {name: 'alice', age: 30}) RETURN n.name AS x",
        "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN n.name AS x",
    );
    // Landing node of a hop.
    same(
            "MATCH (a:Person {name: 'alice'})-[:KNOWS]->(b {name: 'carol'}) RETURN b.age AS a",
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'alice' AND b.name = 'carol' RETURN b.age AS a",
        );
    // Empty map is a no-op filter (all rows).
    same(
        "MATCH (n:Person {}) RETURN n.name AS x",
        "MATCH (n:Person) RETURN n.name AS x",
    );
    // A non-matching constraint yields nothing.
    assert!(bag(&run(
        &super::parse("MATCH (n:Person {name: 'nobody'}) RETURN n.name AS x").unwrap(),
        &store
    ))
    .is_empty());
}

/// A CORRELATED inline-property value `(b {k: a.k})` — an expression, not a
/// literal — lowers to the filter `b.k = a.k`, the exact equivalent of the
/// `(b WHERE b.k = a.k)` spelling (equivalent spellings cost the same).
#[test]
fn inline_property_expression_matches_where() {
    // A chain where a hop lands on a node whose age equals the source's.
    let mut b = Builder::default();
    let a = b.node(&["P"], &[("age", n(30.0))]);
    let same_age = b.node(&["P"], &[("age", n(30.0))]);
    let diff_age = b.node(&["P"], &[("age", n(40.0))]);
    b.edge(a, same_age, "R");
    b.edge(a, diff_age, "R");
    let store = b.build();
    let same = |inline: &str, wher: &str| {
        let x = bag(&run(&super::parse(inline).unwrap(), &store));
        let y = bag(&run(&super::parse(wher).unwrap(), &store));
        assert_eq!(x, y, "`{inline}` vs `{wher}`");
    };
    // Landing correlated on the source: only the same-age neighbour matches.
    same(
        "MATCH (a:P)-[:R]->(b {age: a.age}) RETURN b.age AS x",
        "MATCH (a:P)-[:R]->(b) WHERE b.age = a.age RETURN b.age AS x",
    );
    // An arithmetic expression in the value position.
    same(
        "MATCH (a:P)-[:R]->(b {age: a.age + 10}) RETURN b.age AS x",
        "MATCH (a:P)-[:R]->(b) WHERE b.age = a.age + 10 RETURN b.age AS x",
    );
    // A constant expression on the ANCHOR node (nothing bound before it).
    same(
        "MATCH (a:P {age: 20 + 20}) RETURN a.age AS x",
        "MATCH (a:P) WHERE a.age = 40 RETURN a.age AS x",
    );
}

/// A literal `{k: null}` stays an IS NULL structural test (not `k = null`), and a
/// plain literal value stays literal — the expression path must not swallow them.
#[test]
fn inline_literal_and_null_props_unchanged() {
    let mut b = Builder::default();
    b.node(&["P"], &[("age", n(1.0))]);
    b.node(&["P"], &[("age", Value::Null)]);
    let store = b.build();
    let count = |q: &str| bag(&run(&super::parse(q).unwrap(), &store)).len();
    // `{age: null}` matches the null-valued node only (IS NULL).
    assert_eq!(count("MATCH (n:P {age: null}) RETURN n.age AS x"), 1);
    // `{age: 1}` matches the numeric node only.
    assert_eq!(count("MATCH (n:P {age: 1}) RETURN n.age AS x"), 1);
}

/// A plain INSERT evaluates a CONSTANT property expression (`duration('P1D')`,
/// arithmetic) like TS — but a reference to an unbound variable (nothing is bound
/// in a plain INSERT) is still an error, now the precise "unknown variable" one.
#[test]
fn insert_constant_expr_ok_unbound_var_rejected() {
    // A constant expression is accepted and stored.
    let mut st = Builder::default().build();
    exec_gql(
        &mut st,
        "INSERT (:Z {id: 'z', d: duration('P1D'), n: 1 + 1})",
    );
    assert!(matches!(st.prop(0, "n"), Value::Num(x) if x == 2.0));
    assert!(matches!(st.prop(0, "d"), Value::Temporal(_)));
    // A reference to an unbound variable is rejected (nothing is bound here).
    let e = super::parse("INSERT (x:P {age: y.age})").unwrap_err();
    assert!(
        e.contains("unknown variable"),
        "expected an unbound-variable rejection, got: {e}"
    );
}

/// An INSERT relationship also evaluates a constant property expression
/// (`-[:R {at: date('…')}]->`), not only bare literals.
#[test]
fn insert_edge_constant_expr_prop() {
    let mut st = Builder::default().build();
    exec_gql(
        &mut st,
        "INSERT (:P {id: 'a'})-[:R {w: 1 + 2, at: date('2020-01-01')}]->(:P {id: 'b'})",
    );
    assert_eq!(st.edge_count(), 1);
    assert!(matches!(st.edge_prop(0, "w"), Value::Num(x) if x == 3.0));
    assert!(matches!(st.edge_prop(0, "at"), Value::Temporal(_)));
}

#[test]
fn single_node_return_property() {
    use crate::ir::{Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        // Unaliased `p.name` is named `p.name` (the expression text), like core.
        "p.name".into(),
        Expr::Prop {
            slot: 0,
            key: "name".into(),
        },
    )]);
    assert_same("MATCH (p:Person) RETURN p.name", &hand, &store);
}

#[test]
fn cast_parses_target_and_runs() {
    use crate::ir::{CastTarget, Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        "a".into(),
        Expr::Cast {
            target: CastTarget::String,
            expr: Box::new(Expr::Prop {
                slot: 0,
                key: "age".into(),
            }),
        },
    )]);
    assert_same(
        "MATCH (p:Person) RETURN CAST(p.age AS STRING) AS a",
        &hand,
        &store,
    );
}

#[test]
fn cast_integer_alias_and_bad_type() {
    use crate::ir::{CastTarget, Expr, Plan};
    let store = social();
    // The `INT` alias parses to the same `Integer` target as `INTEGER`.
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        "a".into(),
        Expr::Cast {
            target: CastTarget::Integer,
            expr: Box::new(Expr::Prop {
                slot: 0,
                key: "age".into(),
            }),
        },
    )]);
    assert_same(
        "MATCH (p:Person) RETURN CAST(p.age AS INT) AS a",
        &hand,
        &store,
    );
    // An unknown target type is a parse error, not a silent fallback.
    assert!(super::parse("MATCH (p:Person) RETURN CAST(p.age AS WIDGET) AS a").is_err());
}

/// A store with all three null states on `P.age`: present non-null, absent,
/// and present-null. These are what separate `IS NULL` (a value test) from
/// `PROPERTY_EXISTS` (a presence test).
fn null_states() -> Store {
    let mut b = Builder::default();
    b.node(&["P"], &[("name", s("has")), ("age", n(30.0))]);
    b.node(&["P"], &[("name", s("absent"))]);
    b.node(&["P"], &[("name", s("null"))]);
    let mut st = b.build();
    st.set_prop(2, "age", Value::Null); // node 2: present, but Null
    st
}

/// The set of `name` values a query returns, sorted — order is unspecified.
fn names(store: &Store, query: &str) -> Vec<String> {
    let out = run(&super::parse(query).unwrap(), store);
    let mut got: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    got.sort();
    got
}

#[test]
fn is_null_is_a_value_test() {
    use crate::ir::{Expr, Plan};
    let st = null_states();
    // IS NULL is TRUE for both absent and present-null; IS NOT NULL only for
    // the present non-null. (A definite predicate — no row is UNKNOWN.)
    assert_eq!(
        names(&st, "MATCH (p:P) WHERE p.age IS NULL RETURN p.name"),
        vec!["absent", "null"]
    );
    assert_eq!(
        names(&st, "MATCH (p:P) WHERE p.age IS NOT NULL RETURN p.name"),
        vec!["has"]
    );
    // Parse cross-check against the hand-built plan.
    let hand = Plan::Scan {
        label: Some("P".into()),
    }
    .filter(Expr::IsNull {
        expr: Box::new(Expr::Prop {
            slot: 0,
            key: "age".into(),
        }),
        negated: false,
    })
    .project(vec![(
        "name".into(),
        Expr::Prop {
            slot: 0,
            key: "name".into(),
        },
    )]);
    assert_same(
        "MATCH (p:P) WHERE p.age IS NULL RETURN p.name AS name",
        &hand,
        &st,
    );
}

#[test]
fn property_exists_is_a_presence_test() {
    use crate::ir::{Expr, Plan};
    let st = null_states();
    // PROPERTY_EXISTS is TRUE wherever the value is PRESENT — including the
    // present-null — and FALSE only for the absent node. This is the case
    // `IS NOT NULL` cannot express: "null" appears here but not above.
    assert_eq!(
        names(
            &st,
            "MATCH (p:P) WHERE PROPERTY_EXISTS(p, age) RETURN p.name"
        ),
        vec!["has", "null"]
    );
    let hand = Plan::Scan {
        label: Some("P".into()),
    }
    .filter(Expr::PropertyExists {
        slot: 0,
        key: "age".into(),
    })
    .project(vec![(
        "name".into(),
        Expr::Prop {
            slot: 0,
            key: "name".into(),
        },
    )]);
    assert_same(
        "MATCH (p:P) WHERE PROPERTY_EXISTS(p, age) RETURN p.name AS name",
        &hand,
        &st,
    );
}

#[test]
fn with_aggregate_then_having_filter() {
    use crate::ir::{AggFn, CompareOp, Dir, Expr, Plan};
    let store = social();
    // KNOWS out-degree: alice=2 (bob,carol), bob=1 (carol), carol=0. WITH
    // aggregates the degree, then WHERE filters it (HAVING) — which a single
    // RETURN cannot do. Only alice survives n >= 2.
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()])
    .aggregate(
        vec![("a".into(), Expr::Slot(0))],
        vec![crate::ir::Agg {
            func: AggFn::Count,
            arg: Some(Expr::Slot(1)),
            distinct: false,
            name: "n".into(),
            frac: None,
            null_on_empty: false,
            numeric_only: false,
        }],
    )
    .filter(Expr::Compare {
        op: CompareOp::Ge,
        left: Box::new(Expr::Slot(1)),
        right: Box::new(Expr::Lit(Value::Num(2.0))),
    })
    .project(vec![
        (
            "name".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        ),
        ("n".into(), Expr::Slot(1)),
    ]);
    let q = "MATCH (a:Person)-[:KNOWS]->(b) WITH a, count(b) AS n WHERE n >= 2 \
                 RETURN a.name AS name, n";
    assert_same(q, &hand, &store);
    // And the concrete answer: alice with degree 2.
    let out = run(&super::parse(q).unwrap(), &store);
    assert_eq!(out.rows.len(), 1);
    assert!(crate::value::equals(&col(&out, 0, "name"), &s("alice")));
    assert_eq!(num(&col(&out, 0, "n")), 2.0);
}

#[test]
fn with_carries_a_node_into_a_continuing_match() {
    use crate::ir::{CompareOp, Dir, Expr, Plan};
    let store = social();
    // Carry `a`, filter it (HAVING), then continue the pattern FROM `a`. Only
    // alice(30)/carol(40) pass age>=30; alice KNOWS bob,carol and carol knows
    // no one out, so the endpoints are bob and carol.
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![("a".into(), Expr::Slot(0))])
    .filter(Expr::Compare {
        op: CompareOp::Ge,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "age".into(),
        }),
        right: Box::new(Expr::Lit(Value::Num(30.0))),
    })
    .expand(0, Dir::Out, &["KNOWS".to_string()])
    .project(vec![(
        "name".into(),
        Expr::Prop {
            slot: 1,
            key: "name".into(),
        },
    )]);
    let q = "MATCH (a:Person) WITH a WHERE a.age >= 30 \
                 MATCH (a)-[:KNOWS]->(b) RETURN b.name AS name";
    assert_same(q, &hand, &store);
    assert_eq!(names(&store, q), vec!["bob", "carol"]);
}

#[test]
fn with_order_by_alias_and_limit_pages() {
    let store = social();
    // WITH projects age+name, pages by age DESC LIMIT 2 (carol 40, alice 30 —
    // bob 25 is dropped), then RETURN name. The surviving set is {alice,carol}.
    let q = "MATCH (p:Person) WITH p.age AS age, p.name AS name \
                 ORDER BY age DESC LIMIT 2 RETURN name";
    assert_eq!(names(&store, q), vec!["alice", "carol"]);
}

#[test]
fn continuing_match_from_fresh_variable_cross_joins() {
    // A continuing MATCH whose first node is a FRESH variable is a new independent
    // pattern cross-joined with the working table (valid ISO), not an error.
    assert!(
        super::parse("MATCH (a:Person) WITH a MATCH (z)-[:KNOWS]->(y) RETURN y.name AS name")
            .is_ok()
    );
}

#[test]
fn exists_correlated_subpattern() {
    use crate::ir::{Dir, Expr, Plan};
    let store = social();
    // Who has an outgoing KNOWS? alice (bob,carol) and bob (carol); carol has
    // none. EXISTS is a definite predicate over the correlated node `p`.
    let body = Plan::Row.expand(0, Dir::Out, &["KNOWS".to_string()]);
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .filter(Expr::Exists {
        body: Box::new(body),
        outer_width: 1,
    })
    .project(vec![(
        "name".into(),
        Expr::Prop {
            slot: 0,
            key: "name".into(),
        },
    )]);
    let q = "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(x) } RETURN p.name AS name";
    assert_same(q, &hand, &store);
    assert_eq!(names(&store, q), vec!["alice", "bob"]);
}

#[test]
fn exists_with_inner_where_on_body_var() {
    let store = social();
    // The sub-pattern's WHERE filters the reached node: only a KNOWS target
    // younger than 30 counts. alice knows bob(25) → yes; bob knows only
    // carol(40) → no; carol knows no one → no. So only alice qualifies.
    let q = "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(x) WHERE x.age < 30 } \
                 RETURN p.name AS name";
    assert_eq!(names(&store, q), vec!["alice"]);
}

#[test]
fn exists_where_correlates_on_the_outer_row() {
    let store = social();
    // The sub-WHERE references the OUTER node `p`: does p know someone older?
    // alice(30) knows carol(40) → yes; bob(25) knows carol(40) → yes;
    // carol(40) knows no one → no.
    let q = "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(x) WHERE x.age > p.age } \
                 RETURN p.name AS name";
    assert_eq!(names(&store, q), vec!["alice", "bob"]);
}

#[test]
fn not_exists_negates_the_predicate() {
    let store = social();
    // EXISTS is a definite Bool, so NOT composes cleanly: the Persons with NO
    // outgoing KNOWS. Only carol (alice and bob both know someone).
    let q = "MATCH (p:Person) WHERE NOT EXISTS { (p)-[:KNOWS]->(x) } RETURN p.name AS name";
    assert_eq!(names(&store, q), vec!["carol"]);
}

#[test]
fn exists_uncorrelated_body_is_accepted() {
    // An EXISTS body that binds NEITHER endpoint to an outer variable is a valid
    // UNCORRELATED existence check (run once) — not an error.
    assert!(super::parse(
        "MATCH (p:Person) WHERE EXISTS { (z)-[:KNOWS]->(x) } RETURN p.name AS name",
    )
    .is_ok());
}

/// COUNT / EXISTS correlated on the pattern's ENDPOINT with an ANONYMOUS start node
/// (`(:Person)-[:KNOWS]->(p)`) — a reverse expansion from the bound landing. KNOWS:
/// alice→bob, alice→carol, bob→carol, so in-degree is alice 0, bob 1, carol 2.
/// (Previously rejected with "must start from a bound variable".)
#[test]
fn count_exists_endpoint_anchor_anonymous_start() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN p.name AS name, \
                 COUNT { (:Person)-[:KNOWS]->(p) } AS indeg ORDER BY name",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "indeg")), 0.0); // alice
    assert_eq!(num(&col(&out, 1, "indeg")), 1.0); // bob
    assert_eq!(num(&col(&out, 2, "indeg")), 2.0); // carol
                                                  // A bare `()` start (no label) reverse-anchors the same way.
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN p.name AS name, \
                 COUNT { ()-[:KNOWS]->(p) } AS indeg ORDER BY name",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 2, "indeg")), 2.0); // carol
                                                  // EXISTS endpoint-anchor: who is known by at least one Person? (not alice)
    let q = "MATCH (p:Person) WHERE EXISTS { (:Person)-[:KNOWS]->(p) } RETURN p.name AS name";
    assert_eq!(names(&store, q), vec!["bob", "carol"]);
}

#[test]
fn call_inline_lateral_join() {
    use crate::ir::{Dir, Expr, Plan};
    let store = social();
    // For each Person, expand KNOWS in a subquery and yield the friend's name
    // — a lateral join. carol knows no one, so she drops out (inner join).
    let call = Plan::CallInline {
        input: Box::new(Plan::Scan {
            label: Some("Person".into()),
        }),
        // Slot 1 is the exec-seeded provenance column; the body's expanded node `x`
        // lands at slot 2 (`outer_width + 1`).
        body: Box::new(Plan::Row.expand(0, Dir::Out, &["KNOWS".to_string()])),
        yields: vec![(
            "friend".into(),
            Expr::Prop {
                slot: 2,
                key: "name".into(),
            },
        )],
        outer_width: 1,
        optional: false,
        parts: Vec::new(),
    };
    let hand = call.project(vec![
        (
            "name".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        ),
        ("friend".into(), Expr::Slot(1)),
    ]);
    let q = "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(x) RETURN x.name AS friend } \
                 RETURN p.name AS name, friend";
    assert_same(q, &hand, &store);
    assert_eq!(
        bag(&run(&super::parse(q).unwrap(), &store)),
        vec![
            "name=Str(\"alice\");friend=Str(\"bob\");",
            "name=Str(\"alice\");friend=Str(\"carol\");",
            "name=Str(\"bob\");friend=Str(\"carol\");",
        ]
    );
}

/// A single `COUNT(...)` aggregate RETURN in a correlated `CALL (p) { … }` is a
/// per-outer-row count with LEFT semantics: an outer row whose sub-pattern matches
/// nothing still survives with count 0.
#[test]
fn call_inline_correlated_count() {
    let store = social(); // alice KNOWS bob & carol; bob KNOWS carol; carol knows none
    let out = run(
        &super::parse(
            "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } \
                 RETURN p.name AS name, c ORDER BY name",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(
        bag(&out),
        vec![
            "name=Str(\"alice\");c=Num(2.0);",
            "name=Str(\"bob\");c=Num(1.0);",
            "name=Str(\"carol\");c=Num(0.0);", // no KNOWS → survives with 0
        ]
    );
}

/// `CALL () { … }` — an UNCORRELATED empty-scope subquery — runs once and
/// cross-joins the outer table. An aggregate body yields a single row (so each
/// outer row survives); a body referencing an outer variable sees it as NULL
/// (scope isolation), so `WHERE c = a` matches nothing.
#[test]
fn call_inline_uncorrelated_empty_scope() {
    let store = social(); // 4 nodes: alice, bob, carol, graphdb
    let total = run(
        &super::parse(
            "MATCH (p:Person {name: 'alice'}) \
                 CALL () { MATCH (n) RETURN count(n) AS total } RETURN total",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(total.rows.len(), 1);
    assert!(matches!(total.rows[0][0], crate::value::Value::Num(x) if x == 4.0));
    // Isolated outer reference: `c = a` with `a` NULL inside `()` → no rows.
    let iso = run(
        &super::parse(
            "MATCH (a:Person) WHERE a.name = 'alice' \
                 CALL () { MATCH (b:Person)-[:KNOWS]->(c) WHERE c = a RETURN c.name AS cn } \
                 RETURN cn",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(iso.rows.len(), 0);
}

#[test]
fn call_inline_subquery_where() {
    let store = social();
    // The subquery's WHERE filters the reached node: only friends older than 30
    // (carol) count. alice and bob both know carol; carol knows no one.
    let q = "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(x) WHERE x.age > 30 \
                 RETURN x.name AS friend } RETURN p.name AS name, friend";
    assert_eq!(
        bag(&run(&super::parse(q).unwrap(), &store)),
        vec![
            "name=Str(\"alice\");friend=Str(\"carol\");",
            "name=Str(\"bob\");friend=Str(\"carol\");",
        ]
    );
}

#[test]
fn call_inline_yield_correlates_on_outer() {
    let store = social();
    // The yield expression mixes the subquery node and the OUTER node: the age
    // gap x.age - p.age. alice(30)->bob(25)=-5, alice->carol(40)=10,
    // bob(25)->carol(40)=15. carol knows no one.
    let q = "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(x) \
                 RETURN x.age - p.age AS gap } RETURN gap";
    let out = run(&super::parse(q).unwrap(), &store);
    let mut gaps: Vec<f64> = out
        .rows
        .iter()
        .map(|r| match r[0] {
            Value::Num(x) => x,
            ref o => panic!("expected Num, got {o:?}"),
        })
        .collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    assert_eq!(gaps, vec![-5.0, 10.0, 15.0]);
}

#[test]
fn call_named_form_is_deferred() {
    // The named-procedure form has no catalog yet (the algorithms it calls are
    // a later phase); it is a clear error, not a silent no-op.
    let err = super::parse("MATCH (p:Person) CALL foo() RETURN p.name AS name").unwrap_err();
    assert!(
        err.contains("named-procedure CALL is deferred"),
        "got: {err}"
    );
}

#[test]
fn call_inline_unbound_scope_errors() {
    let err = super::parse(
        "MATCH (p:Person) CALL (z) { MATCH (z)-[:KNOWS]->(x) RETURN x.name AS f } \
             RETURN p.name AS name",
    )
    .unwrap_err();
    assert!(err.contains("not bound"), "got: {err}");
}

/// A straight chain a→b→c→d over LINK edges (node ids 0,1,2,3). Shortest
/// paths have distinct, checkable lengths — unlike the dense `social()`.
fn chain() -> Store {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    let d = b.node(&["N"], &[("name", s("d"))]);
    b.edge(a, bb, "LINK");
    b.edge(bb, c, "LINK");
    b.edge(c, d, "LINK");
    b.build()
}

/// The `id` field of each element map in a path-accessor result list. `chain()`
/// nodes/edges carry no external id, so the rendered id is the dense-id fallback
/// ("0".."3" for nodes, "e0".."e2" for edges).
fn elem_ids(list: &Value) -> Vec<String> {
    let Value::List(items) = list else {
        panic!("not a list: {list:?}")
    };
    items
        .iter()
        .map(|e| match e {
            Value::Map(m) => match m
                .iter()
                .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "id"))
                .map(|(_, v)| v)
            {
                Some(Value::Str(s)) => s.to_string(),
                o => panic!("id not a string: {o:?}"),
            },
            o => panic!("not an element map: {o:?}"),
        })
        .collect()
}

#[test]
fn any_shortest_path_length() {
    use crate::ir::{Dir, Expr, PathPart, Plan};
    let store = chain();
    // Shortest LINK paths from `a`: the `*` quantifier admits the zero-length
    // path to `a` itself (len 0), then b at 1 hop, c at 2, d at 3.
    let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) WHERE x.name = 'a' \
                 RETURN y.name AS y, path_length(p) AS len";
    assert_eq!(
        bag(&run(&super::parse(q).unwrap(), &store)),
        vec![
            "y=Str(\"a\");len=Num(0.0);",
            "y=Str(\"b\");len=Num(1.0);",
            "y=Str(\"c\");len=Num(2.0);",
            "y=Str(\"d\");len=Num(3.0);",
        ]
    );
    // Parse cross-check against the hand-built ShortestPath plan (all sources).
    // `*` is min 0 (the seed is a zero-length path to itself).
    let hand = Plan::Scan { label: None }
        .shortest_path(
            0,
            Dir::Out,
            &["LINK".to_string()],
            0,
            None,
            crate::ir::ShortestSelector::Any,
            None,
        )
        .project(vec![(
            "len".into(),
            Expr::PathAccess {
                part: PathPart::Length,
            },
        )]);
    assert_same(
        "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) RETURN path_length(p) AS len",
        &hand,
        &store,
    );
}

#[test]
fn any_shortest_nodes_reconstructs_the_chain() {
    let store = chain();
    // The full path a→b→c→d is reconstructed (BFS predecessors), so nodes(p)
    // is the node-id chain [0,1,2,3], not just the endpoint.
    let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) \
                 WHERE x.name = 'a' AND y.name = 'd' RETURN nodes(p) AS ns";
    let out = run(&super::parse(q).unwrap(), &store);
    assert_eq!(out.rows.len(), 1);
    // nodes(p) materializes the four vertices in chain order.
    assert_eq!(elem_ids(&out.rows[0][0]), vec!["0", "1", "2", "3"]);
}

#[test]
fn any_shortest_edges_are_the_traversed_edges() {
    let store = chain();
    // Edges are created a→b, b→c, c→d (ids 0,1,2). The shortest path a→d
    // traverses all three, in order — edges(p) recovers them. (`edges` is the ISO
    // name; the Cypher-ism `relationships` is not accepted.)
    let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) \
                 WHERE x.name = 'a' AND y.name = 'd' RETURN edges(p) AS es";
    let out = run(&super::parse(q).unwrap(), &store);
    assert_eq!(out.rows.len(), 1);
    // edges(p) materializes the three traversed edges, in order.
    assert_eq!(elem_ids(&out.rows[0][0]), vec!["e0", "e1", "e2"]);
}

#[test]
fn relationships_is_not_an_iso_function() {
    // The Cypher spelling `relationships` is NOT a GQL function — ISO uses `edges`.
    let err = super::parse("MATCH p = (a)-[:LINK]->(b) RETURN relationships(p) AS x").unwrap_err();
    assert!(err.contains("E_UNKNOWN_FUNCTION"), "got: {err}");
}

#[test]
fn any_shortest_elements_interleave_nodes_and_edges() {
    let store = chain();
    // elements(p) for a→d is n0,e0,n1,e1,n2,e2,n3 = 0,0,1,1,2,2,3 (node ids
    // 0..3, edge ids 0..2). Nodes and edges are both Num here, so compare the
    // flat sequence.
    let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) \
                 WHERE x.name = 'a' AND y.name = 'd' RETURN elements(p) AS els";
    let out = run(&super::parse(q).unwrap(), &store);
    assert_eq!(out.rows.len(), 1);
    // elements(p) interleaves n0,e0,n1,e1,n2,e2,n3 — each a full element map.
    assert_eq!(
        elem_ids(&out.rows[0][0]),
        vec!["0", "e0", "1", "e1", "2", "e2", "3"]
    );
}

#[test]
fn path_accessor_requires_a_path_variable() {
    // A path accessor on a non-path expression is a clear parse error.
    let err = super::parse("MATCH (a:Person) RETURN edges(a.name) AS x").unwrap_err();
    assert!(err.contains("path variable"), "got: {err}");
}

#[test]
fn named_path_over_plain_pattern_is_accepted() {
    // A named path does NOT require a shortest-path selector: `MATCH p = <plain
    // pattern>` binds the pattern's (WALK/TRAIL) lineage, readable via
    // path_length(p)/nodes(p)/edges(p). Both a fixed hop and a var-length body
    // parse.
    assert!(super::parse("MATCH p = (a)-[:LINK]->(b) RETURN p").is_ok());
    assert!(super::parse("MATCH p = (a)-[:LINK]->{1,3}(b) RETURN path_length(p) AS n").is_ok());
}

#[test]
fn any_shortest_requires_a_quantifier() {
    let err =
        super::parse("MATCH p = ANY SHORTEST (a)-[:LINK]->(b) RETURN a.name AS a").unwrap_err();
    assert!(err.contains("quantifier"), "got: {err}");
}

#[test]
fn temporal_literals_render_and_compare() {
    let store = social();
    // The three zone-less literals parse and round-trip to their ISO form.
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN DATE '2024-01-15' AS d, TIME '13:45:06' AS t, \
                 DATETIME '2024-01-15T09:00:00' AS dt",
        )
        .unwrap(),
        &store,
    );
    let iso = |v: &Value| match v {
        Value::Temporal(t) => t.format(),
        o => panic!("expected Temporal, got {o:?}"),
    };
    assert_eq!(iso(&col(&out, 0, "d")), "2024-01-15");
    assert_eq!(iso(&col(&out, 0, "t")), "13:45:06");
    assert_eq!(iso(&col(&out, 0, "dt")), "2024-01-15T09:00:00");

    // Date ordering as a constant predicate: earlier < later keeps all rows,
    // the reverse keeps none, and equality holds.
    let count = |q: &str| run(&super::parse(q).unwrap(), &store).rows.len();
    assert_eq!(
        count("MATCH (p:Person) WHERE DATE '2024-01-01' < DATE '2024-06-01' RETURN p.name"),
        3
    );
    assert_eq!(
        count("MATCH (p:Person) WHERE DATE '2024-06-01' < DATE '2024-01-01' RETURN p.name"),
        0
    );
    assert_eq!(
        count("MATCH (p:Person) WHERE DATE '2024-01-01' = DATE '2024-01-01' RETURN p.name"),
        3
    );
    // A malformed literal is a parse error.
    assert!(super::parse("MATCH (p:Person) RETURN DATE '2024-13-01' AS d").is_err());
}

#[test]
fn duration_and_zoned_literals() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN DURATION 'P1Y2M' AS d, \
                 ZONED DATETIME '2024-01-15T12:00:00+01:00' AS z, \
                 ZONED TIME '13:45:00Z' AS zt",
        )
        .unwrap(),
        &store,
    );
    let iso = |v: &Value| match v {
        Value::Temporal(t) => t.format(),
        o => panic!("expected Temporal, got {o:?}"),
    };
    assert_eq!(iso(&col(&out, 0, "d")), "P14M"); // 1Y2M = 14 months, canonical
    assert_eq!(iso(&col(&out, 0, "z")), "2024-01-15T12:00:00+01:00");
    assert_eq!(iso(&col(&out, 0, "zt")), "13:45:00Z");
    // A malformed duration literal is a parse error.
    assert!(super::parse("MATCH (p:Person) RETURN DURATION 'nope' AS d").is_err());
}

#[test]
fn temporal_component_accessors() {
    let store = social();
    // Component accessors carry the leading-underscore extension sigil (`_year`);
    // the bare ISO spelling `year()` is NOT a function (unknown-function error).
    assert!(super::parse("RETURN year(DATE '2024-03-15') AS y").is_err());
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN _year(DATE '2024-03-15') AS y, \
                 _month(DATE '2024-03-15') AS mo, _day(DATE '2024-03-15') AS d, \
                 _hour(TIME '13:45:06') AS h, _minute(TIME '13:45:06') AS mi, \
                 _second(TIME '13:45:06') AS se, _year(DATETIME '2020-07-04T09:30:00') AS dty",
        )
        .unwrap(),
        &store,
    );
    for (name, want) in [
        ("y", 2024.0),
        ("mo", 3.0),
        ("d", 15.0),
        ("h", 13.0),
        ("mi", 45.0),
        ("se", 6.0),
        ("dty", 2020.0),
    ] {
        assert_eq!(num(&col(&out, 0, name)), want, "{name}");
    }
    // A component undefined for the kind FAULTS with E_INVALID_VALUE (year of a
    // time, hour of a date) — matching core, which errors rather than NULLs.
    for q in [
        "MATCH (p:Person) RETURN _year(TIME '01:02:03') AS y",
        "MATCH (p:Person) RETURN _hour(DATE '2024-01-01') AS h",
    ] {
        let err = crate::exec::try_run(&super::parse(q).unwrap(), &store);
        assert!(
            matches!(&err, Err(e) if e.contains("E_INVALID_VALUE")),
            "expected fault for `{q}`, got {err:?}"
        );
    }
}

#[test]
fn temporal_constructors_and_coercion() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN \
                 date('2024-03-15') AS d1, \
                 datetime('2024-03-15') AS d2, \
                 date(DATETIME '2024-03-15T09:30:00') AS d3, \
                 datetime(DATE '2024-03-15') AS d4, \
                 local_time(DATETIME '2024-03-15T09:30:45') AS d5, \
                 duration('P1Y2M') AS d6",
        )
        .unwrap(),
        &store,
    );
    let iso = |v: &Value| match v {
        Value::Temporal(t) => t.format(),
        o => panic!("expected Temporal, got {o:?}"),
    };
    assert_eq!(iso(&col(&out, 0, "d1")), "2024-03-15"); // parse
    assert_eq!(iso(&col(&out, 0, "d2")), "2024-03-15T00:00:00"); // date-str → midnight
    assert_eq!(iso(&col(&out, 0, "d3")), "2024-03-15"); // datetime → date part
    assert_eq!(iso(&col(&out, 0, "d4")), "2024-03-15T00:00:00"); // date → midnight
    assert_eq!(iso(&col(&out, 0, "d5")), "09:30:45"); // datetime → time part
    assert_eq!(iso(&col(&out, 0, "d6")), "P14M"); // 1Y2M canonical
                                                  // A malformed constructor argument is NULL, not an error.
    let out2 = run(
        &super::parse("MATCH (p:Person) RETURN date('garbage') AS d").unwrap(),
        &store,
    );
    assert!(col(&out2, 0, "d").is_null());
}

#[test]
fn duration_between_is_exact() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN \
                 duration_between(DATE '2020-01-15', DATE '2020-04-20') AS a, \
                 duration_between(DATETIME '2020-01-01T00:00:00', \
                 DATETIME '2020-01-01T01:01:01') AS b, \
                 duration_between(DATE '2020-01-01', DATETIME '2020-01-01T00:00:00') AS c",
        )
        .unwrap(),
        &store,
    );
    let iso = |v: &Value| match v {
        Value::Temporal(t) => t.format(),
        o => panic!("expected Temporal, got {o:?}"),
    };
    assert_eq!(iso(&col(&out, 0, "a")), "P96D"); // 96 days (2020 is a leap year)
    assert_eq!(iso(&col(&out, 0, "b")), "PT3661S"); // 1h1m1s
    assert!(col(&out, 0, "c").is_null()); // cross-kind → NULL
}

#[test]
fn temporal_arithmetic() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN \
                 DATE '2024-01-31' + DURATION 'P1M' AS clamp_leap, \
                 DATE '2023-01-31' + DURATION 'P1M' AS clamp, \
                 DATE '2024-01-15' + DURATION 'P10D' AS plus_days, \
                 DATETIME '2024-01-15T10:00:00' + DURATION 'PT3661S' AS dt_plus, \
                 DATE '2024-04-20' - DATE '2024-01-15' AS span, \
                 DURATION 'P1M' + DURATION 'P2D' AS dsum, \
                 DURATION 'P2D' * 3 AS dscale",
        )
        .unwrap(),
        &store,
    );
    let iso = |v: &Value| match v {
        Value::Temporal(t) => t.format(),
        o => panic!("expected Temporal, got {o:?}"),
    };
    assert_eq!(iso(&col(&out, 0, "clamp_leap")), "2024-02-29"); // Jan31+1M → Feb29 (leap)
    assert_eq!(iso(&col(&out, 0, "clamp")), "2023-02-28"); // non-leap → Feb28
    assert_eq!(iso(&col(&out, 0, "plus_days")), "2024-01-25");
    assert_eq!(iso(&col(&out, 0, "dt_plus")), "2024-01-15T11:01:01"); // +1h1m1s
    assert_eq!(iso(&col(&out, 0, "span")), "P96D"); // Jan15→Apr20, leap year
    assert_eq!(iso(&col(&out, 0, "dsum")), "P1M2D");
    assert_eq!(iso(&col(&out, 0, "dscale")), "P6D");

    // A non-integer duration scale is NULL (no meaningful fractional month).
    let out2 = run(
        &super::parse("MATCH (p:Person) RETURN DURATION 'P2D' * 1.5 AS d").unwrap(),
        &store,
    );
    assert!(col(&out2, 0, "d").is_null());
}

#[test]
fn temporal_arithmetic_overflow_throws() {
    let store = social();
    // Adding ~8.3M years leaves the representable i32-day date range: a THROWN
    // fault (E_INVALID_VALUE via the fallible pipeline), not a silent null.
    let plan =
        super::parse("MATCH (p:Person) RETURN DATE '2024-01-01' + DURATION 'P100000000M' AS d")
            .unwrap();
    let err = crate::exec::try_run(&plan, &store).unwrap_err();
    assert!(err.contains("E_INVALID_VALUE"), "got: {err}");
}

#[test]
fn record_literal_and_field_access() {
    use crate::ir::{CompareOp, Expr, Plan};
    let store = social();
    // Build a record from a matched node, carry it through WITH, read fields.
    let out = run(
        &super::parse(
            "MATCH (p:Person) WHERE p.name = 'alice' \
                 WITH {name: p.name, age: p.age} AS r RETURN r.name AS n, r.age AS a, \
                 r.missing AS m",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(out.rows.len(), 1);
    assert!(crate::value::equals(&col(&out, 0, "n"), &s("alice")));
    assert_eq!(num(&col(&out, 0, "a")), 30.0);
    assert!(col(&out, 0, "m").is_null()); // absent field → NULL

    // A returned record has its keys sorted (canonical), whatever the literal
    // order; cross-checked against the hand-built Record plan.
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .filter(Expr::Compare {
        op: CompareOp::Eq,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "name".into(),
        }),
        right: Box::new(Expr::Lit(Value::Str("alice".into()))),
    })
    .project(vec![(
        "r".into(),
        Expr::Record {
            fields: vec![
                ("b".into(), Expr::Lit(Value::Num(2.0))),
                ("a".into(), Expr::Lit(Value::Num(1.0))),
            ],
        },
    )]);
    let q = "MATCH (p:Person) WHERE p.name = 'alice' RETURN {b: 2, a: 1} AS r";
    assert_same(q, &hand, &store);
    let out2 = run(&super::parse(q).unwrap(), &store);
    match &col(&out2, 0, "r") {
        Value::Record(f) => {
            assert_eq!(f[0].0.as_ref(), "a"); // sorted
            assert_eq!(f[1].0.as_ref(), "b");
        }
        o => panic!("expected a Record, got {o:?}"),
    }
}

#[test]
fn field_access_on_a_record_literal() {
    use crate::ir::{Expr, Plan};
    let store = social();
    // `{lit}.field` and a chained `.outer.inner` on nested record literals.
    let out = run(
        &super::parse(
            "MATCH (p:Person) WHERE p.name = 'alice' RETURN \
                 {a: 1, b: 2}.b AS x, {outer: {inner: 7}}.outer.inner AS y, \
                 {a: 1}.missing AS m",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "x")), 2.0);
    assert_eq!(num(&col(&out, 0, "y")), 7.0);
    assert!(col(&out, 0, "m").is_null()); // absent field → NULL

    // Cross-check `{a: 1}.a` against the hand-built Field(Record) plan.
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .filter(Expr::Compare {
        op: crate::ir::CompareOp::Eq,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "name".into(),
        }),
        right: Box::new(Expr::Lit(Value::Str("alice".into()))),
    })
    .project(vec![(
        "x".into(),
        Expr::Field {
            base: Box::new(Expr::Record {
                fields: vec![("a".into(), Expr::Lit(Value::Num(1.0)))],
            }),
            key: "a".into(),
        },
    )]);
    assert_same(
        "MATCH (p:Person) WHERE p.name = 'alice' RETURN {a: 1}.a AS x",
        &hand,
        &store,
    );
}

#[test]
fn nested_field_access_on_a_stored_record() {
    let mut store = social();
    // Store a record property, then read nested fields via `n.rec.field`.
    crate::exec::execute(
        &super::parse(
            "MATCH (p:Person) WHERE p.name = 'alice' \
                 SET p.meta = {city: 'NYC', zip: 10001}",
        )
        .unwrap(),
        &mut store,
    )
    .unwrap();
    let out = run(
        &super::parse(
            "MATCH (p:Person) WHERE p.name = 'alice' \
                 RETURN p.meta.city AS c, p.meta.zip AS z, p.meta.absent AS a",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(out.rows.len(), 1);
    assert!(crate::value::equals(&col(&out, 0, "c"), &s("NYC")));
    assert_eq!(num(&col(&out, 0, "z")), 10001.0);
    assert!(col(&out, 0, "a").is_null()); // absent nested field → NULL
}

#[test]
fn stored_dates_round_trip_and_filter() {
    let mut store = social();
    // Store birthdates on two Persons, then find those born before 2000.
    for (who, born) in [("alice", "1990-05-01"), ("bob", "2005-03-03")] {
        let q = format!("MATCH (p:Person) WHERE p.name = '{who}' SET p.born = DATE '{born}'");
        crate::exec::execute(&super::parse(&q).unwrap(), &mut store).unwrap();
    }
    // alice(1990) qualifies; bob(2005) does not; carol has no `born` (NULL,
    // so the comparison is UNKNOWN and she is filtered out).
    assert_eq!(
        names(
            &store,
            "MATCH (p:Person) WHERE p.born < DATE '2000-01-01' RETURN p.name AS name"
        ),
        vec!["alice"]
    );
    // The stored date reads back as its ISO string.
    let out = run(
        &super::parse("MATCH (p:Person) WHERE p.name = 'alice' RETURN p.born AS born").unwrap(),
        &store,
    );
    match &col(&out, 0, "born") {
        Value::Temporal(t) => assert_eq!(t.format(), "1990-05-01"),
        o => panic!("expected Temporal, got {o:?}"),
    }
}

#[test]
fn cdc_observes_a_committed_insert() {
    use crate::store::Change;
    let mut store = Builder::default().build();
    crate::exec::execute(
        &super::parse("INSERT (:P {name: 'a'}), (:P {name: 'b'})").unwrap(),
        &mut store,
    )
    .unwrap();
    // The INSERT is txn-wrapped, so its two node adds surface as CDC changes.
    assert_eq!(
        store.last_commit_changes(),
        &[Change::NodeAdded(0), Change::NodeAdded(1)]
    );
}

#[test]
fn required_constraint_rejects_insert_without_the_key() {
    let mut store = Builder::default().build();
    store.create_required_constraint("User", "email").unwrap();
    // INSERT carrying the required key succeeds.
    crate::exec::execute(
        &super::parse("INSERT (:User {email: 'a@x'})").unwrap(),
        &mut store,
    )
    .unwrap();
    // INSERT missing it is rejected and rolled back (node count unchanged).
    let before = store.node_count();
    let err = crate::exec::execute(
        &super::parse("INSERT (:User {name: 'b'})").unwrap(),
        &mut store,
    )
    .unwrap_err();
    assert!(err.contains("E_REQUIRED"), "got: {err}");
    assert_eq!(store.node_count(), before);
}

/// A directed triangle a→b→c→a (ids 0,1,2) + an isolated node d (3).
fn triangle_store() -> Store {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[]);
    let bb = b.node(&["N"], &[]);
    let c = b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    b.edge(c, a, "R");
    b.build()
}

#[test]
fn call_degree_procedure_yield_and_default() {
    use crate::ir::{Expr, Plan};
    let store = triangle_store();
    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    // Out-degrees: each triangle node 1, the isolated node 0.
    let want = vec![(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)];
    assert_eq!(rows_of("CALL degree() YIELD node, degree"), want);
    // No YIELD → the default [node, <result>] columns.
    assert_eq!(rows_of("CALL degree()"), want);
    // YIELD renames the output columns.
    let out = run(
        &super::parse("CALL degree() YIELD node AS n, degree AS d").unwrap(),
        &store,
    );
    assert_eq!(out.names, vec!["n".to_string(), "d".to_string()]);

    // Parse→run matches the hand-built plan (CallProcedure under a Project).
    let hand = Plan::CallProcedure {
        name: "degree".into(),
        config: vec![],
    }
    .project(vec![
        ("node".into(), Expr::Slot(0)),
        ("degree".into(), Expr::Slot(1)),
    ]);
    assert_same("CALL degree()", &hand, &store);
}

#[test]
fn call_closeness_procedure_yields_centrality() {
    let store = triangle_store();
    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    // Directed OUT triangle: each member Σdist=3 → 1/3; the isolated node → 0.
    let want = vec![
        (0.0, 1.0 / 3.0),
        (1.0, 1.0 / 3.0),
        (2.0, 1.0 / 3.0),
        (3.0, 0.0),
    ];
    assert_eq!(rows_of("CALL closeness() YIELD node, centrality"), want);
    // Default columns (no YIELD) are [node, centrality].
    assert_eq!(rows_of("CALL closeness()"), want);
    let out = run(&super::parse("CALL closeness()").unwrap(), &store);
    assert_eq!(
        out.names,
        vec!["node".to_string(), "centrality".to_string()]
    );
}

#[test]
fn call_scc_procedure_yields_component_id() {
    let store = triangle_store();
    let rows_of = |q: &str| -> Vec<(f64, String)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), str_val(&r[1])))
            .collect()
    };
    // The directed triangle {0,1,2} is one SCC (rep 0 → ext id "0"); the isolated
    // node is {3} (rep 3 → ext id "3").
    let want = vec![
        (0.0, "0".to_string()),
        (1.0, "0".to_string()),
        (2.0, "0".to_string()),
        (3.0, "3".to_string()),
    ];
    assert_eq!(
        rows_of("CALL strongly_connected_components() YIELD node, componentId"),
        want
    );
    assert_eq!(rows_of("CALL strongly_connected_components()"), want);
    let out = run(
        &super::parse("CALL strongly_connected_components()").unwrap(),
        &store,
    );
    assert_eq!(
        out.names,
        vec!["node".to_string(), "componentId".to_string()]
    );
}

#[test]
fn call_on_cycle_procedure_yields_on_cycle_flag() {
    let store = triangle_store();
    let rows_of = |q: &str| -> Vec<(f64, bool)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| {
                let b = match &r[1] {
                    Value::Bool(b) => *b,
                    other => panic!("onCycle should be a Bool, got {other:?}"),
                };
                (node_id(&r[0]), b)
            })
            .collect()
    };
    // The triangle members are on a cycle (Bool true, matching core's `onCycle`
    // type); the isolated node is not (Bool false).
    let want = vec![(0.0, true), (1.0, true), (2.0, true), (3.0, false)];
    assert_eq!(rows_of("CALL on_cycle() YIELD node, onCycle"), want);
    assert_eq!(rows_of("CALL on_cycle()"), want);
    let out = run(&super::parse("CALL on_cycle()").unwrap(), &store);
    assert_eq!(out.names, vec!["node".to_string(), "onCycle".to_string()]);
}

#[test]
fn call_betweenness_procedure_yields_centrality() {
    let store = triangle_store();
    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    // Directed triangle: each member is the sole intermediary of one 2-hop path
    // → 1.0; the isolated node → 0.0.
    let want = vec![(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)];
    assert_eq!(rows_of("CALL betweenness() YIELD node, centrality"), want);
    assert_eq!(rows_of("CALL betweenness()"), want);
    let out = run(&super::parse("CALL betweenness()").unwrap(), &store);
    assert_eq!(
        out.names,
        vec!["node".to_string(), "centrality".to_string()]
    );
}

#[test]
fn call_shortest_path_procedure_yields_distance() {
    let store = triangle_store();
    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    // OUT from source "0" on the triangle: hop distances 0,1,2 to the three
    // members; the isolated node is unreachable and absent.
    let want = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
    assert_eq!(
        rows_of("CALL shortest_path({source: '0'}) YIELD node, distance"),
        want
    );
    assert_eq!(rows_of("CALL shortest_path({source: '0'})"), want);
    let out = run(
        &super::parse("CALL shortest_path({source: '0'})").unwrap(),
        &store,
    );
    assert_eq!(out.names, vec!["node".to_string(), "distance".to_string()]);
    // A `target` restricts the result to just that vertex's distance.
    assert_eq!(
        rows_of("CALL shortest_path({source: '0', target: '2'}) YIELD node, distance"),
        vec![(2.0, 2.0)]
    );
    // An unreachable target (the isolated node) yields nothing.
    assert!(rows_of("CALL shortest_path({source: '0', target: '3'})").is_empty());
}

#[test]
fn call_shortest_path_astar() {
    // 0→1 (w=10), 0→2 (1), 2→1 (1): A* 0→1 returns the exact shortest distance (2),
    // the same as Dijkstra, guided by the algorithm:'astar' backend.
    let mut bld = Builder::default();
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[]);
    let mut store = bld.build();
    let e0 = store.add_edge(0, 1, "R");
    store.set_edge_prop(e0, "w", Value::Num(10.0));
    let e1 = store.add_edge(0, 2, "R");
    store.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = store.add_edge(2, 1, "R");
    store.set_edge_prop(e2, "w", Value::Num(1.0));

    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    assert_eq!(
        rows_of(
            "CALL shortest_path({source:'0', target:'1', weightProperty:'w', \
                 algorithm:'astar'}) YIELD node, distance"
        ),
        vec![(1.0, 2.0)]
    );
}

#[test]
fn call_shortest_path_weighted() {
    // 0→1 (w=10), 0→2 (w=1), 2→1 (w=1): weighted 0→1 = 2 (light detour), while
    // unweighted 0→1 = 1 (direct edge).
    let mut bld = Builder::default();
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[]);
    let mut store = bld.build();
    let e0 = store.add_edge(0, 1, "R");
    store.set_edge_prop(e0, "w", Value::Num(10.0));
    let e1 = store.add_edge(0, 2, "R");
    store.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = store.add_edge(2, 1, "R");
    store.set_edge_prop(e2, "w", Value::Num(1.0));

    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    assert_eq!(
        rows_of("CALL shortest_path({source: '0', weightProperty: 'w'}) YIELD node, distance"),
        vec![(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]
    );
    // Same query without the weight → the hop distance to 1 is 1.
    assert_eq!(
        rows_of("CALL shortest_path({source: '0'})"),
        vec![(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)]
    );
}

#[test]
fn call_closeness_weighted() {
    // 0→1 (w=10), 0→2 (1), 2→1 (1): weighted closeness of 0 is 1/3 (Dijkstra
    // sum 3), vs unweighted 1/2 (hop sum 2).
    let mut bld = Builder::default();
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[]);
    let mut store = bld.build();
    let e0 = store.add_edge(0, 1, "R");
    store.set_edge_prop(e0, "w", Value::Num(10.0));
    let e1 = store.add_edge(0, 2, "R");
    store.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = store.add_edge(2, 1, "R");
    store.set_edge_prop(e2, "w", Value::Num(1.0));

    let close0 = |q: &str| -> f64 { num(&run(&super::parse(q).unwrap(), &store).rows[0][1]) };
    assert!(
        (close0("CALL closeness({weightProperty: 'w'}) YIELD node, centrality") - 1.0 / 3.0).abs()
            < 1e-12
    );
    assert!((close0("CALL closeness()") - 1.0 / 2.0).abs() < 1e-12);
}

#[test]
fn call_betweenness_weighted() {
    // Diamond with a heavy 2→3 branch: weighted betweenness routes all 0→3
    // dependency through node 1 (1.0), where unweighted splits it 0.5/0.5.
    let mut bld = Builder::default();
    for _ in 0..4 {
        bld.node(&["N"], &[]);
    }
    let mut store = bld.build();
    let e0 = store.add_edge(0, 1, "R");
    store.set_edge_prop(e0, "w", Value::Num(1.0));
    let e1 = store.add_edge(0, 2, "R");
    store.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = store.add_edge(1, 3, "R");
    store.set_edge_prop(e2, "w", Value::Num(1.0));
    let e3 = store.add_edge(2, 3, "R");
    store.set_edge_prop(e3, "w", Value::Num(5.0));

    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    assert_eq!(
        rows_of("CALL betweenness({weightProperty: 'w'}) YIELD node, centrality"),
        vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 0.0)]
    );
    assert_eq!(
        rows_of("CALL betweenness()"),
        vec![(0.0, 0.0), (1.0, 0.5), (2.0, 0.5), (3.0, 0.0)]
    );
}

#[test]
fn call_neighbor_aggregate_weighted() {
    // 0→1 (w=1, h=[2]), 0→2 (w=3, h=[4]): weighted mean at 0 is 14/(1+3)=3.5.
    let mut bld = Builder::default();
    let f = |x: f64| Value::List(vec![Value::Num(x)]);
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[("h", f(2.0))]);
    bld.node(&["N"], &[("h", f(4.0))]);
    let mut store = bld.build();
    let e0 = store.add_edge(0, 1, "R");
    store.set_edge_prop(e0, "w", Value::Num(1.0));
    let e1 = store.add_edge(0, 2, "R");
    store.set_edge_prop(e1, "w", Value::Num(3.0));

    let out = run(
        &super::parse(
            "CALL neighbor_aggregate({feature: 'h', op: 'mean', direction: 'out', \
                 weightProperty: 'w'}) YIELD node, vector",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(format!("{:?}", out.rows[0][1]), "List([Num(3.5)])");
}

#[test]
fn call_neighbor_aggregate_gcn() {
    // 0→1, 0→2 (unweighted); h(1)=[2], h(2)=[4]. GCN sum at 0 folds each
    // contributor by 1/sqrt(deg_0·deg_nbr) = 1/sqrt(2).
    let mut bld = Builder::default();
    let f = |x: f64| Value::List(vec![Value::Num(x)]);
    bld.node(&["N"], &[]);
    bld.node(&["N"], &[("h", f(2.0))]);
    bld.node(&["N"], &[("h", f(4.0))]);
    let mut store = bld.build();
    store.add_edge(0, 1, "R");
    store.add_edge(0, 2, "R");

    let out = run(
        &super::parse(
            "CALL neighbor_aggregate({feature: 'h', op: 'sum', direction: 'out', \
                 norm: 'gcn'}) YIELD node, vector",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(
        format!("{:?}", out.rows[0][1]),
        "List([Num(4.242640687119285)])"
    );
}

#[test]
fn call_personalized_pagerank_yields_score() {
    let store = triangle_store();
    let rows_of = |q: &str| -> Vec<(f64, f64)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), num(&r[1])))
            .collect()
    };
    // Seeding node "0" via the sourceNodes list makes 0 the strict max and leaves
    // the unreachable isolated node at 0; the yield column is `score`.
    let seeded = rows_of("CALL personalized_pagerank({sourceNodes: ['0']}) YIELD node, score");
    assert_eq!(seeded.len(), 4);
    assert!(seeded[0].1 > seeded[1].1 && seeded[0].1 > seeded[2].1);
    assert_eq!(seeded[3], (3.0, 0.0));
    // Default columns are [node, score].
    let out = run(
        &super::parse("CALL personalized_pagerank({sourceNodes: ['0']})").unwrap(),
        &store,
    );
    assert_eq!(out.names, vec!["node".to_string(), "score".to_string()]);
}

#[test]
fn call_neighbor_aggregate_yields_vector() {
    // a(0)=[1,2], b(1)=[3,4]; a→b. OUT-sum at a folds b's vector; b has none.
    let mut bld = Builder::default();
    let vec = |xs: &[f64]| Value::List(xs.iter().map(|&x| Value::Num(x)).collect());
    let a = bld.node(&["N"], &[("h", vec(&[1.0, 2.0]))]);
    let b = bld.node(&["N"], &[("h", vec(&[3.0, 4.0]))]);
    bld.edge(a, b, "R");
    let store = bld.build();

    let out = run(
        &super::parse(
            "CALL neighbor_aggregate({feature: 'h', op: 'sum', direction: 'out'}) \
                 YIELD node, vector",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(out.names, vec!["node".to_string(), "vector".to_string()]);
    // Node a's aggregate is b's feature [3,4]; node b's is the zero vector.
    assert_eq!(out.rows.len(), 2);
    assert_eq!(
        format!("{:?}", out.rows[0][1]),
        "List([Num(3.0), Num(4.0)])"
    );
    assert_eq!(
        format!("{:?}", out.rows[1][1]),
        "List([Num(0.0), Num(0.0)])"
    );
}

#[test]
fn call_peer_pressure_yields_cluster() {
    // Sink 1→0, 2→0, 3→0: node 0 joins cluster 1 (tie to smallest ext id); the
    // sources keep their own cluster. The yield column is `cluster`.
    let mut bld = Builder::default();
    let a = bld.node(&["N"], &[]);
    let x = bld.node(&["N"], &[]);
    let y = bld.node(&["N"], &[]);
    let z = bld.node(&["N"], &[]);
    bld.edge(x, a, "R");
    bld.edge(y, a, "R");
    bld.edge(z, a, "R");
    let store = bld.build();

    let rows_of = |q: &str| -> Vec<(f64, String)> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| (node_id(&r[0]), str_val(&r[1])))
            .collect()
    };
    // Cluster is the rep node's ext id string: node 0 joins cluster "1", sources
    // keep their own ("1"/"2"/"3").
    let want = vec![
        (0.0, "1".to_string()),
        (1.0, "1".to_string()),
        (2.0, "2".to_string()),
        (3.0, "3".to_string()),
    ];
    assert_eq!(rows_of("CALL peer_pressure() YIELD node, cluster"), want);
    assert_eq!(rows_of("CALL peer_pressure()"), want);
    let out = run(&super::parse("CALL peer_pressure()").unwrap(), &store);
    assert_eq!(out.names, vec!["node".to_string(), "cluster".to_string()]);
}

#[test]
fn call_procedure_config_and_components() {
    let store = triangle_store();
    // degree with direction=both: each triangle node 2, isolated 0.
    let both: Vec<f64> = run(
        &super::parse("CALL degree({direction: 'both'}) YIELD degree").unwrap(),
        &store,
    )
    .rows
    .iter()
    .map(|r| num(&r[0]))
    .collect();
    assert_eq!(both, vec![2.0, 2.0, 2.0, 0.0]);
    // connected_components: triangle → component root ext id "0", isolated → "3".
    let comps: Vec<(f64, String)> = run(
        &super::parse("CALL connected_components() YIELD node, componentId").unwrap(),
        &store,
    )
    .rows
    .iter()
    .map(|r| (node_id(&r[0]), str_val(&r[1])))
    .collect();
    assert_eq!(
        comps,
        vec![
            (0.0, "0".to_string()),
            (1.0, "0".to_string()),
            (2.0, "0".to_string()),
            (3.0, "3".to_string()),
        ]
    );
}

#[test]
fn call_procedure_errors() {
    // Unknown procedure and unknown YIELD column are both parse errors.
    assert!(super::parse("CALL bogus()").is_err());
    assert!(super::parse("CALL degree() YIELD nope").is_err());
}

#[test]
fn where_filter_and_alias() {
    use crate::ir::{CompareOp, Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .filter(Expr::Compare {
        op: CompareOp::Gt,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "age".into(),
        }),
        right: Box::new(Expr::Lit(Value::Num(28.0))),
    })
    .project(vec![(
        "who".into(),
        Expr::Prop {
            slot: 0,
            key: "name".into(),
        },
    )]);
    assert_same(
        "MATCH (p:Person) WHERE p.age > 28 RETURN p.name AS who",
        &hand,
        &store,
    );
}

#[test]
fn one_hop_binds_both() {
    use crate::ir::{Dir, Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()])
    .project(vec![
        (
            "a".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        ),
        (
            "b".into(),
            Expr::Prop {
                slot: 1,
                key: "name".into(),
            },
        ),
    ]);
    assert_same(
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b",
        &hand,
        &store,
    );
}

#[test]
fn two_hops_and_where_conjunction() {
    use crate::ir::{Dir, Expr, Plan};
    let store = social();
    // (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE a.name='alice' AND c.age>=40
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()])
    .expand(1, Dir::Out, &["KNOWS".to_string()])
    .filter(Expr::And(
        Box::new(Expr::Compare {
            op: crate::ir::CompareOp::Eq,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "name".into(),
            }),
            right: Box::new(Expr::Lit(Value::Str("alice".into()))),
        }),
        Box::new(Expr::Compare {
            op: crate::ir::CompareOp::Ge,
            left: Box::new(Expr::Prop {
                slot: 2,
                key: "age".into(),
            }),
            right: Box::new(Expr::Lit(Value::Num(40.0))),
        }),
    ))
    .project(vec![(
        "c".into(),
        Expr::Prop {
            slot: 2,
            key: "name".into(),
        },
    )]);
    assert_same(
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) \
             WHERE a.name = 'alice' AND c.age >= 40 RETURN c.name AS c",
        &hand,
        &store,
    );
    // and the direct answer, hand-checked: alice->b->c with c.age>=40 is
    // alice->bob->carol and alice->carol->? carol KNOWS nobody, so only carol.
    let out = run(&super::parse("MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE a.name = 'alice' AND c.age >= 40 RETURN c.name AS c").unwrap(), &store);
    assert_eq!(out.rows.len(), 1);
}

#[test]
fn incoming_direction() {
    use crate::ir::{Dir, Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .filter(Expr::Compare {
        op: crate::ir::CompareOp::Eq,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "name".into(),
        }),
        right: Box::new(Expr::Lit(Value::Str("carol".into()))),
    })
    .expand(0, Dir::In, &["KNOWS".to_string()])
    .project(vec![(
        "who".into(),
        Expr::Prop {
            slot: 1,
            key: "name".into(),
        },
    )]);
    assert_same(
        "MATCH (c:Person)<-[:KNOWS]-(who) WHERE c.name = 'carol' RETURN who.name AS who",
        &hand,
        &store,
    );
}

#[test]
fn parse_errors_are_reported_not_panicked() {
    assert!(super::parse("MATCH (p:Person").is_err()); // unclosed
    assert!(super::parse("MATCH (p:Person) RETURN q.name").is_err()); // unknown var
    assert!(super::parse("RETURN 1").is_ok()); // bare RETURN is a valid statement
    assert!(super::parse("RETURN").is_err()); // …but RETURN needs at least one item
    assert!(super::parse("MATCH (p:Person) WHERE p.age > RETURN p.name").is_err());
}

// --- part 2: aggregation, DISTINCT, ORDER/SKIP/LIMIT ---

fn num(v: &Value) -> f64 {
    match v {
        Value::Num(x) => *x,
        other => panic!("expected number, got {other:?}"),
    }
}
// A node-id-valued procedure result (componentId / cluster / label) renders as the
// representative node's EXTERNAL id string (matching core), not a dense index.
fn str_val(v: &Value) -> String {
    match v {
        Value::Str(s) => s.to_string(),
        other => panic!("expected string, got {other:?}"),
    }
}
fn col(rows: &Rows, r: usize, name: &str) -> Value {
    let i = rows.names.iter().position(|n| n == name).expect("column");
    rows.rows[r][i].clone()
}

/// The numeric id of a NODE-element result map (`{id: "N", labels, properties}`),
/// which is how a node binding now renders (matching core).
fn node_id(v: &Value) -> f64 {
    match v {
        Value::Map(m) => m
            .iter()
            .find_map(|(k, val)| match (k, val) {
                (Value::Str(k), Value::Str(id)) if &**k == "id" => id.parse().ok(),
                _ => None,
            })
            .expect("node map carries a string id"),
        other => num(other),
    }
}

#[test]
fn scalar_count_star() {
    let store = social();
    let out = run(
        &super::parse("MATCH (p:Person) RETURN count(*) AS c").unwrap(),
        &store,
    );
    assert_eq!(out.rows.len(), 1);
    assert_eq!(num(&col(&out, 0, "c")), 3.0);
}

#[test]
fn group_count_by_property() {
    let store = social();
    // group people by age bucket... simpler: count Persons by their own name
    // (each unique) is 1 each — instead group by a shared value. Use KNOWS
    // out-degree: (a)-[:KNOWS]->(b) grouped by a.name.
    let out = run(
        &super::parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS who, count(*) AS deg")
            .unwrap(),
        &store,
    );
    let mut got: Vec<(String, f64)> = out
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Str(w), Value::Num(d)) => (w.to_string(), *d),
            _ => panic!("shape"),
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(got, vec![("alice".into(), 2.0), ("bob".into(), 1.0)]);
}

#[test]
fn sum_min_max_avg_match_hand_built() {
    use crate::ir::{Agg, AggFn, Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .aggregate(
        vec![],
        vec![
            Agg {
                func: AggFn::Sum,
                arg: Some(Expr::Prop {
                    slot: 0,
                    key: "age".into(),
                }),
                distinct: false,
                name: "s".into(),
                frac: None,
                null_on_empty: false,
                numeric_only: false,
            },
            Agg {
                func: AggFn::Avg,
                arg: Some(Expr::Prop {
                    slot: 0,
                    key: "age".into(),
                }),
                distinct: false,
                name: "a".into(),
                frac: None,
                null_on_empty: false,
                numeric_only: false,
            },
        ],
    );
    assert_same(
        "MATCH (p:Person) RETURN sum(p.age) AS s, avg(p.age) AS a",
        &hand,
        &store,
    );
}

#[test]
fn return_distinct() {
    let store = social();
    // distinct set of nodes reachable by KNOWS: {bob, carol}
    let out = run(
        &super::parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN DISTINCT b.name AS who").unwrap(),
        &store,
    );
    let mut got: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            _ => panic!(),
        })
        .collect();
    got.sort();
    assert_eq!(got, vec!["bob", "carol"]);
}

#[test]
fn order_by_limit_top_k() {
    let store = social();
    // oldest two people by age desc: carol(40), alice(30)
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age DESC LIMIT 2",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(out.rows.len(), 2);
    assert_eq!(num(&col(&out, 0, "age")), 40.0);
    assert_eq!(num(&col(&out, 1, "age")), 30.0);
}

#[test]
fn order_by_skip_limit_window() {
    let store = social();
    // ascending age: bob(25), alice(30), carol(40); skip 1 limit 1 -> alice
    let out = run(
        &super::parse(
            "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age ASC SKIP 1 LIMIT 1",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(out.rows.len(), 1);
    assert_eq!(num(&col(&out, 0, "age")), 30.0);
}

#[test]
fn order_by_aggregate_alias() {
    let store = social();
    // out-degree desc: alice(2) then bob(1)
    let out = run(
            &super::parse(
                "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS who, count(*) AS deg ORDER BY deg DESC",
            )
            .unwrap(),
            &store,
        );
    assert_eq!(out.rows.len(), 2);
    assert_eq!(num(&col(&out, 0, "deg")), 2.0);
    assert_eq!(num(&col(&out, 1, "deg")), 1.0);
}

#[test]
fn order_by_unknown_column_errors() {
    assert!(super::parse("MATCH (p:Person) RETURN p.name AS name ORDER BY age").is_err());
}

// --- part 3: comma-join and variable-length ---

#[test]
fn comma_join_shared_variable() {
    use crate::ir::{Dir, Expr, Plan};
    let store = social();
    // (a)-[:KNOWS]->(b), (a)-[:WORKS_ON]->(c) sharing a. Only alice has both.
    let left = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()]);
    let right = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["WORKS_ON".to_string()]);
    let hand = Plan::join(left, right, vec![(0, 0)]).project(vec![
        (
            "a".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        ),
        (
            "b".into(),
            Expr::Prop {
                slot: 1,
                key: "name".into(),
            },
        ),
        (
            "c".into(),
            Expr::Prop {
                slot: 3,
                key: "name".into(),
            },
        ),
    ]);
    assert_same(
        "MATCH (a:Person)-[:KNOWS]->(b), (a:Person)-[:WORKS_ON]->(c) \
             RETURN a.name AS a, b.name AS b, c.name AS c",
        &hand,
        &store,
    );
    // hand-checked: alice KNOWS {bob,carol} x WORKS_ON {graphdb} = 2 rows.
    let out = run(
        &super::parse(
            "MATCH (a:Person)-[:KNOWS]->(b), (a:Person)-[:WORKS_ON]->(c) RETURN c.name AS c",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(out.rows.len(), 2);
}

#[test]
fn var_length_range() {
    use crate::ir::{Dir, Expr, PathMode, Plan};
    let store = social();
    // (a)-[:KNOWS]->{1,2}(b) from alice: b(len1)={bob,carol}, then len2 from
    // those: bob->carol. Trail. Cross-check vs hand-built VarLength.
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .filter(Expr::Compare {
        op: crate::ir::CompareOp::Eq,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "name".into(),
        }),
        right: Box::new(Expr::Lit(Value::Str("alice".into()))),
    })
    .var_length(0, Dir::Out, &["KNOWS".to_string()], 1, 2, PathMode::Trail)
    .project(vec![(
        "b".into(),
        Expr::Prop {
            slot: 1,
            key: "name".into(),
        },
    )]);
    assert_same(
        "MATCH (a:Person)-[:KNOWS]->{1,2}(b) WHERE a.name = 'alice' RETURN b.name AS b",
        &hand,
        &store,
    );
}

#[test]
fn var_length_exact_and_open() {
    let store = social();
    // exact {2}: alice's 2-hop KNOWS endpoints = {carol} (alice->bob->carol).
    let out = run(
        &super::parse(
            "MATCH (a:Person)-[:KNOWS]->{2}(b) WHERE a.name = 'alice' RETURN b.name AS b",
        )
        .unwrap(),
        &store,
    );
    let mut got: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            _ => panic!(),
        })
        .collect();
    got.sort();
    assert_eq!(got, vec!["carol"]);

    // open {1,} is accepted (capped) and reaches everyone reachable.
    assert!(super::parse(
        "MATCH (a:Person)-[:KNOWS]->{1,}(b) WHERE a.name = 'alice' RETURN b.name AS b"
    )
    .is_ok());
}

#[test]
fn bad_quantifier_errors() {
    assert!(super::parse("MATCH (a:Person)-[:KNOWS]->{3,1}(b) RETURN a.name AS a").is_err());
}

/// The algebraic degree-product count for a bounded OUT var-length must equal the
/// DFS enumeration — on a graph large enough that the formula path FIRES, and
/// with a self-loop that exercises the reused-edge trail correction.
/// The 3-hop edge-product `count(*)` must equal the DFS enumeration — on a graph
/// with a cycle and a self-loop, since a FIXED chain is a WALK (edges may repeat,
/// no trail correction).
#[test]
fn three_hop_product_count_matches_enumeration() {
    use crate::store::Builder;
    let mut b = Builder::default();
    b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    b.node(&["N"], &[]); // a=0, b=1, c=2
    b.edge(0, 1, "R");
    b.edge(1, 2, "R");
    b.edge(2, 0, "R"); // 3-cycle
    b.edge(0, 0, "R"); // self-loop
    let st = b.build();
    let q = "MATCH (a:N)-[:R]->()-[:R]->()-[:R]->(d)";
    let count = match &run(
        &super::parse(&format!("{q} RETURN count(*) AS c")).unwrap(),
        &st,
    )
    .rows[0][0]
    {
        Value::Num(x) => *x as usize,
        other => panic!("not a count: {other:?}"),
    };
    let enumerated = run(&super::parse(&format!("{q} RETURN d.z AS d")).unwrap(), &st)
        .rows
        .len();
    assert_eq!(count, enumerated, "product count != enumerated walks");
}

#[test]
fn edge_filtered_count_matches_enumeration() {
    use crate::store::Builder;
    use crate::value::Value;
    // A small graph with a numeric edge property `w`; the streaming edge-filtered
    // count must equal the enumerated matching-row count for each predicate.
    let mut b = Builder::default();
    for _ in 0..200 {
        b.node(&["P"], &[]);
    }
    for i in 0u32..200 {
        for d in 0u32..3 {
            b.edge(i, (i * 7 + d * 13 + 1) % 200, "R");
        }
    }
    let mut st = b.build();
    for eid in st.all_edges() {
        st.set_edge_prop(eid, "w", Value::Num(f64::from(eid % 10)));
    }
    let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x as usize,
        other => panic!("not a count: {other:?}"),
    };
    for wc in ["r.w > 5", "r.w >= 2 AND r.w < 5", "r.w = 3", "3 > r.w"] {
        let c = count(&format!(
            "MATCH (a:P)-[r:R]->(b) WHERE {wc} RETURN count(*) AS c"
        ));
        let rows = run(
            &super::parse(&format!(
                "MATCH (a:P)-[r:R]->(b) WHERE {wc} RETURN r.w AS w"
            ))
            .unwrap(),
            &st,
        )
        .rows
        .len();
        assert_eq!(c, rows, "edge-filtered count != enumerated for `{wc}`");
    }
}

#[test]
fn streaming_num_filtered_count_matches_enumeration() {
    use crate::store::Builder;
    // Includes a NULL age (every 11th) and a NaN age (every 13th) so the count's
    // NULL-gating and NaN-drops-from-ordering rules are exercised; the streaming
    // count must equal the enumerated survivor count for each predicate spelling.
    let mut b = Builder::default();
    for i in 0..3000u32 {
        let mut props = vec![("name", Value::Str(format!("n{i}").into()))];
        let age = if i % 13 == 0 {
            Some(f64::NAN)
        } else if i % 11 == 0 {
            None
        } else {
            Some(f64::from(i % 100))
        };
        if let Some(a) = age {
            props.push(("age", Value::Num(a)));
        }
        b.node(&["P"], &props);
    }
    let st = b.build();
    let cnt = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x as usize,
        other => panic!("not a count: {other:?}"),
    };
    // Each: count(*) (streaming) == the enumerated row count (RETURN name).
    for wc in [
        "p.age > 50",
        "50 < p.age", // flipped operands, same predicate
        "p.age >= 10 AND p.age < 20",
        "p.age <= 10 AND p.age >= 5",
        "p.age = 42",
        "p.age <> 42",
    ] {
        let c = cnt(&format!("MATCH (p:P) WHERE {wc} RETURN count(*) AS c"));
        let rows = run(
            &super::parse(&format!("MATCH (p:P) WHERE {wc} RETURN p.name AS n")).unwrap(),
            &st,
        )
        .rows
        .len();
        assert_eq!(c, rows, "streaming count != enumerated for `{wc}`");
    }
}

#[test]
fn string_concat_operator() {
    use crate::store::Builder;
    let mut b = Builder::default();
    b.node(
        &["P"],
        &[("name", Value::Str("ab".into())), ("age", Value::Num(7.0))],
    );
    let st = b.build();
    let one = |q: &str| -> Value { run(&super::parse(q).unwrap(), &st).rows[0][0].clone() };
    // string || string, chain, null propagation, list concat.
    assert!(
        matches!(one("MATCH (p:P) RETURN p.name || '!' AS x"), Value::Str(ref s) if &**s == "ab!")
    );
    assert!(
        matches!(one("MATCH (p:P) RETURN 'a' || 'b' || 'c' AS x"), Value::Str(ref s) if &**s == "abc")
    );
    // `||` no longer JS-coerces a numeric operand — mixing a string with a number is a
    // type error (use to_string()), never "ab-7".
    assert!(crate::exec::try_run(
        &super::parse("MATCH (p:P) RETURN p.name || '-' || p.age AS x").unwrap(),
        &st
    )
    .unwrap_err()
    .contains("E_INVALID_VALUE"));
    assert!(matches!(
        one("MATCH (p:P) RETURN p.missing || 'x' AS x"),
        Value::Null
    ));
    assert!(matches!(
        one("MATCH (p:P) RETURN 'x' || p.missing AS x"),
        Value::Null
    ));
    assert!(
        matches!(one("MATCH (p:P) RETURN [1, 2] || [3] AS x"), Value::List(ref v) if v.len() == 3)
    );
    // Precedence: `||` binds looser than `+`. Strict typing means `(1+2) || 3`
    // (num||num) can no longer COERCE to "33" — but the PARSE still groups `+` under
    // `||`, which is what precedence is about. Assert the structure, not a coerced
    // value: the projected expr is a `concat` whose FIRST operand is the `1 + 2` add.
    let plan = super::parse("MATCH (p:P) RETURN 1 + 2 || 3 AS x").unwrap();
    let crate::ir::Plan::Project { items, .. } = &plan else {
        panic!("expected a Project: {plan:?}")
    };
    let crate::ir::Expr::Call { name, args } = &items[0].1 else {
        panic!("expected a concat call: {plan:?}")
    };
    assert_eq!(name, "concat");
    assert!(
        matches!(&args[0], crate::ir::Expr::Arith { .. }),
        "`+` must bind tighter than `||`: {plan:?}"
    );
    // A lone `|` is not an operator.
    assert!(super::parse("MATCH (p:P) RETURN p.age | 1 AS x").is_err());
}

#[test]
fn low_card_num_distinct_matches_hashing() {
    use crate::store::Builder;
    use std::collections::BTreeSet;
    // Columns exercising: low-card ints (age), a NULL every 5th (age absent),
    // high-card ints past the trivial range (uniq), and non-integers (frac, must
    // fall back to hashing). The bitset path must agree with the hashing path on
    // BOTH count(DISTINCT) and the DISTINCT value SET.
    let mut b = Builder::default();
    for i in 0..2000u32 {
        let mut props = vec![
            ("uniq", Value::Num(f64::from(i))),
            ("frac", Value::Num(f64::from(i % 9) + 0.25)),
        ];
        // age covers every value 0..49; the NULL condition (i%7) is independent of
        // the value so the present ages are still the full {0..49}.
        if i % 7 != 0 {
            props.push(("age", Value::Num(f64::from(i % 50))));
        }
        b.node(&["N"], &props);
    }
    let st = b.build();
    let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x as usize,
        other => panic!("not a count: {other:?}"),
    };
    let set = |q: &str| -> BTreeSet<String> {
        run(&super::parse(q).unwrap(), &st)
            .rows
            .iter()
            .map(|r| format!("{:?}", r[0]))
            .collect()
    };
    // count(DISTINCT k) == the size of the DISTINCT value set, for every column.
    for k in ["age", "uniq", "frac"] {
        let c = count(&format!("MATCH (n:N) RETURN count(DISTINCT n.{k}) AS c"));
        let s = set(&format!("MATCH (n:N) RETURN DISTINCT n.{k} AS v"));
        // The DISTINCT set includes a NULL for `age` (absent every 5th node), which
        // count(DISTINCT) excludes — so the set is one larger exactly there.
        let extra = usize::from(k == "age");
        assert_eq!(c + extra, s.len(), "count vs set mismatch for {k}");
    }
    // Concrete expected values: age = {0..49} plus NULL.
    let age = set("MATCH (n:N) RETURN DISTINCT n.age AS v");
    assert_eq!(age.len(), 51);
    assert!(age.contains("Null"));
    assert_eq!(count("MATCH (n:N) RETURN count(DISTINCT n.age) AS c"), 50);
}

#[test]
fn varlen_degree_formula_matches_enumeration() {
    use crate::store::Builder;
    // 1000 nodes, degree 4 → est_paths (1000·4²=16k) > 2·(V+E) (~10k), so the
    // formula fires for {1,2}/{2,2}; a self-loop on node 0 tests the correction.
    let mut b = Builder::default();
    for _ in 0..1000 {
        b.node(&["N"], &[]);
    }
    for i in 0u32..1000 {
        for d in 0u32..4 {
            b.edge(i, (i * 7 + d * 13 + 1) % 1000, "R");
        }
    }
    b.edge(0, 0, "R"); // self-loop → reused-edge trail exclusion
    let st = b.build();
    let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x as usize,
        other => panic!("not a count: {other:?}"),
    };
    // The count(*) (formula) must equal the enumerated row count (RETURN b).
    for (lo, hi) in [(1u32, 1u32), (1, 2), (2, 2)] {
        let formula = count(&format!(
            "MATCH (a:N)-[:R]->{{{lo},{hi}}}(b) RETURN count(*) AS c"
        ));
        let enumerated = run(
            &super::parse(&format!(
                "MATCH (a:N)-[:R]->{{{lo},{hi}}}(b) RETURN b.z AS b"
            ))
            .unwrap(),
            &st,
        )
        .rows
        .len();
        assert_eq!(formula, enumerated, "mismatch for {{{lo},{hi}}}");
    }
}

#[test]
fn varlen_distinct_count_bfs_matches_enumeration() {
    use crate::store::Builder;
    use std::collections::HashSet;
    // A graph with cycles and a self-loop, so shortest-distance reachability and
    // the walk-enumerated endpoint SET are non-trivially exercised.
    let mut b = Builder::default();
    for i in 0..300 {
        b.node(&["N"], &[("k", Value::Num(f64::from(i)))]); // unique per-node key
    }
    for i in 0u32..300 {
        for d in 0u32..3 {
            b.edge(i, (i * 7 + d * 11 + 1) % 300, "R");
        }
    }
    b.edge(0, 0, "R"); // self-loop
    let st = b.build();
    let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x as usize,
        other => panic!("not a count: {other:?}"),
    };
    // min≤1 fires the cumulative shortest-distance BFS; min≥2 fires the per-level
    // set-expansion (N^h(src), not the distance-h set — a walk may revisit). Both
    // regimes must equal the DISTINCT set of endpoints the enumerating path emits.
    // IN and BOTH exercise reverse hops; the self-loop stresses revisits.
    for (lo, hi) in [
        (0u32, 2u32),
        (1, 1),
        (1, 3),
        (1, 4),
        (2, 2),
        (2, 3),
        (3, 3),
        (2, 4),
    ] {
        for dir in ["->", "<-", "-"] {
            let (l, r) = match dir {
                "->" => ("-[:R]", "->"),
                "<-" => ("<-[:R]", "-"),
                _ => ("-[:R]", "-"),
            };
            let pat = format!("(a:N){l}{r}{{{lo},{hi}}}(b)");
            let fast = count(&format!("MATCH {pat} RETURN count(DISTINCT b) AS c"));
            let rows = run(
                &super::parse(&format!("MATCH {pat} RETURN b.k AS b")).unwrap(),
                &st,
            )
            .rows;
            let enumerated: HashSet<String> = rows.iter().map(|r| format!("{:?}", r[0])).collect();
            assert_eq!(
                fast,
                enumerated.len(),
                "BFS distinct != enumerated distinct for {pat}"
            );
        }
    }
}

/// Cardinality-driven anchor flip: a selective indexed `=` on the traversal
/// TARGET seeds the target and walks reverse edges instead of scanning every
/// source. The result multiset must equal the forward walk — INCLUDING excluding
/// a non-source-label node reached in reverse (`bot` is not a `Person`).
#[test]
fn anchor_flip_matches_forward_and_respects_source_label() {
    use crate::store::{Builder, Store};
    let mut b = Builder::default();
    // ids 0..3 in insertion order.
    b.node(&["Person"], &[("name", Value::Str("p1".into()))]); // 0
    b.node(&["Person"], &[("name", Value::Str("p2".into()))]); // 1
    b.node(&["Bot"], &[("name", Value::Str("bot".into()))]); // 2 (not Person)
    b.node(&["Person"], &[("name", Value::Str("target".into()))]); // 3
    b.edge(0, 3, "R");
    b.edge(1, 3, "R");
    b.edge(2, 3, "R"); // bot -> target
    let mut st = b.build();
    let q = "MATCH (a:Person)-[:R]->(b) WHERE b.name = 'target' RETURN a.name AS a";
    let names = |st: &Store| {
        let mut v: Vec<String> = run(&super::parse(q).unwrap(), st)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(s) => s.to_string(),
                o => format!("{o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    // Forward (no index): only the two Person sources reach the target.
    let forward = names(&st);
    assert_eq!(forward, vec!["p1".to_string(), "p2".to_string()]);
    // With an index on `name` the anchor flips (target count 1 < Person count 3);
    // it must give the SAME set — `bot`, reached walking reverse, is excluded.
    st.create_index("name");
    assert_eq!(names(&st), forward);
}

/// A write immediately followed by a traversal must NOT cost more as the graph
/// grows — the shape that repacked the read-side adjacency snapshot on every write
/// in the old engine (the mistake this guards against). Ported from lenke-core's
/// `interleaved_write_and_traverse_is_independent_of_graph_size`: run the
/// write+traverse cycle at 2k and 32k vertices and assert the cost scales
/// sub-linearly (16x more vertices, `< 6x` time). If a read after a write rebuilds
/// the whole adjacency instead of reading the delta, the ratio tracks graph size.
///
/// Timing-based, so MIN over reps (a single ~1ms sample against a 6x bound is below
/// this repo's noise floor); the minimum is the closest thing to an
/// interference-free run. `#[ignore]`d because it flakes under the CPU contention of
/// the full parallel `cargo test` run (as core's own copy notes: "failed 2 of 8 with
/// the box loaded") — a scaling assertion needs an isolated harness. Run it alone:
///   cargo test -p lenke-engine --release --lib \
///     interleaved_write_and_traverse_is_independent_of_graph_size -- --ignored --exact
#[test]
#[ignore = "timing/scaling guard — run isolated in release (see doc comment)"]
fn interleaved_write_and_traverse_is_independent_of_graph_size() {
    use crate::exec::execute;
    use std::time::Instant;

    const REPS: usize = 9;

    // Mirror real usage (the FFI's `lnk_query`): optimize the plan against the store
    // so index seeds are chosen, THEN execute. A raw `execute(parse(q))` skips the
    // optimizer and scans — which would scale with graph size for reasons unrelated
    // to the write-side snapshot this test guards.
    let q = |st: &mut Store, s: &str| {
        let plan = crate::opt::optimize_indexed(super::parse(s).unwrap(), st);
        execute(&plan, st).unwrap();
    };
    let mut st = Builder::default().build();
    st.create_index("id");
    q(&mut st, "INSERT (:Dept {id: 'D0'})");

    let grow_to = |st: &mut Store, upto: usize, from: usize| {
        for i in from..upto {
            q(st, &format!("INSERT (:Emp {{id: 'e{i}'}})"));
            q(
                    st,
                    &format!(
                        "MATCH (s:Emp {{id: 'e{i}'}}), (t:Dept {{id: 'D0'}}) INSERT (s)-[:IN_DEPT]->(t)"
                    ),
                );
        }
    };

    // One property write, then one traversal off the SAME anchor — the shape that
    // repacked. The write is a property SET (does not touch adjacency), so a correct
    // engine reads the unchanged adjacency; a broken one rebuilds it per cycle.
    let travq = "MATCH (s:Emp {id: 'e0'})-[r:IN_DEPT]->(t) RETURN t.id AS x";
    let cycle = |st: &mut Store, i: usize| {
        q(st, &format!("MATCH (s:Emp {{id: 'e0'}}) SET s.w = {i}"));
        let plan = crate::opt::optimize_indexed(super::parse(travq).unwrap(), st);
        let _ = run(&plan, st);
    };

    let time = |st: &mut Store| {
        for i in 0..20 {
            cycle(st, i); // warm up
        }
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            for i in 0..100 {
                cycle(st, i);
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    grow_to(&mut st, 2_000, 0);
    let small = time(&mut st);
    grow_to(&mut st, 32_000, 2_000);
    let large = time(&mut st);

    let ratio = large / small.max(f64::MIN_POSITIVE);
    assert!(
        ratio < 6.0,
        "interleaved write+traverse scaled with graph size: 16x more vertices cost \
             {ratio:.1}x more time ({small:.4}s -> {large:.4}s, min of {REPS}). A read after \
             a write is rebuilding the whole adjacency snapshot instead of reading the delta."
    );
}

/// The CSR adjacency cache must PAY FOR ITSELF: the engine WITH the cache must never
/// be slower than the engine WITHOUT it. Twin of the scaling guard above — that one
/// proves a write does not trigger a rebuild (no churn); this one proves the cache's
/// upkeep (building it, and flipping it stale on every write) never costs MORE than
/// simply not having a cache and always reading the per-node adjacency `Vec`s.
///
/// Runs two serving workloads — a read-heavy full traversal and an interleaved
/// write+traverse — each timed with the cache ON and OFF (`set_csr_enabled`), min over
/// reps. If the cache were a net loss (its bookkeeping outweighing its read speedup),
/// cache-ON would be the slower number. `#[ignore]`d: a timing guard, like its twin —
/// run isolated in release:
///   cargo test -p lenke-engine --release --lib cache_is_never_a_net_loss \
///     -- --ignored
#[test]
#[ignore = "timing guard — run isolated in release (see doc comment)"]
fn cache_is_never_a_net_loss() {
    use crate::exec::execute;
    use std::time::Instant;

    const REPS: usize = 9;

    // A mesh with real, MULTI-TYPE adjacency: 4k nodes, each with one out-edge of
    // each of 4 types. A single-type hop (`-[:A]->`) then reads 1/4 of the adjacency —
    // the shape the typed-CSR partition is built to win (touch only the A edges,
    // instead of scanning all four per node and filtering). Without the cache the hop
    // scans every edge and filters, so this is where the cache should pay off.
    let mut b = Builder::default();
    const N: u32 = 4_000;
    const TYPES: [&str; 4] = ["A", "B", "C", "D"];
    for i in 0..N {
        b.node(&["P"], &[("id", s(&format!("p{i}"))), ("w", n(0.0))]);
    }
    for i in 0..N {
        for (d, ty) in TYPES.iter().enumerate() {
            b.edge(i, (i + d as u32 + 1) % N, ty);
        }
    }
    let mut st = b.build();
    st.create_index("id");

    let readq = crate::opt::optimize_indexed(
        super::parse("MATCH (a:P)-[:A]->(b) RETURN count(*) AS c").unwrap(),
        &st,
    );
    let setq = "MATCH (s:P {id: 'p0'}) SET s.w = 1";
    let anchorq = "MATCH (s:P {id: 'p0'})-[:A]->(t) RETURN t.id AS x";

    // Time `f` after a warm-up, min over reps — the interference-free estimate.
    let bench = |st: &mut Store, f: &dyn Fn(&mut Store)| {
        for _ in 0..5 {
            f(st);
        }
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            for _ in 0..50 {
                f(st);
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    let read_only = |st: &mut Store| {
        let _ = run(&readq, st);
    };
    // Interleaved serving: a property write (does NOT change adjacency, so with the
    // cache ON the CSR stays fresh and the traversal reads it) then a traversal.
    let interleave = |st: &mut Store| {
        let plan = crate::opt::optimize_indexed(super::parse(setq).unwrap(), st);
        execute(&plan, st).unwrap();
        let tp = crate::opt::optimize_indexed(super::parse(anchorq).unwrap(), st);
        let _ = run(&tp, st);
    };

    let measure = |st: &mut Store, f: &dyn Fn(&mut Store)| {
        st.set_csr_enabled(true);
        let on = bench(st, f);
        st.set_csr_enabled(false);
        let off = bench(st, f);
        st.set_csr_enabled(true); // leave it as the default for the next measurement
        (on, off)
    };

    let (read_on, read_off) = measure(&mut st, &read_only);
    let (inter_on, inter_off) = measure(&mut st, &interleave);
    eprintln!(
            "read:  on={read_on:.5} off={read_off:.5} ({:.2}x)   interleave: on={inter_on:.5} off={inter_off:.5} ({:.2}x)",
            read_on / read_off,
            inter_on / inter_off,
        );

    // The cache must never make a workload SLOWER than no cache. A small slack absorbs
    // timing noise; a genuine net-loss cache (e.g. a rebuild churned on every read)
    // would blow well past it.
    assert!(
        read_on <= read_off * 1.10,
        "CSR cache made the read-heavy traversal SLOWER than no cache: \
             on={read_on:.5}s vs off={read_off:.5}s (min of {REPS})"
    );
    assert!(
        inter_on <= inter_off * 1.10,
        "the CSR cache machinery cost MORE than not having it on the interleaved \
             write+traverse workload: on={inter_on:.5}s vs off={inter_off:.5}s (min of {REPS}). \
             The cache's upkeep is outweighing its read speedup."
    );
}

/// The raw string-search filter fast path (STARTS WITH / ENDS WITH / CONTAINS)
/// must match the boxed `str_bool` for a dict-encoded (low-cardinality) column
/// and for a row missing the property (→ UNKNOWN → dropped).
#[test]
fn string_search_fast_path_dict_and_null() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    // `city` is low-cardinality → dict-encoded; one node omits it entirely.
    execute(
        &super::parse(
            "INSERT (:P {city: 'oslo'}), (:P {city: 'bergen'}), (:P {city: 'oslo'}), (:P {n: 1})",
        )
        .unwrap(),
        &mut st,
    )
    .unwrap();
    let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x as i64,
        other => panic!("not a count: {other:?}"),
    };
    assert_eq!(
        count("MATCH (p:P) WHERE p.city STARTS WITH 'os' RETURN count(*) AS c"),
        2
    );
    assert_eq!(
        count("MATCH (p:P) WHERE p.city ENDS WITH 'en' RETURN count(*) AS c"),
        1
    );
    assert_eq!(
        count("MATCH (p:P) WHERE p.city CONTAINS 'o' RETURN count(*) AS c"),
        2
    ); // oslo, oslo (bergen has no 'o')
       // The property-less node is UNKNOWN, never matched.
    assert_eq!(
        count("MATCH (p:P) WHERE p.city STARTS WITH '' RETURN count(*) AS c"),
        3
    );
}

/// A scalar / grouped aggregate over a traversal streams the source in blocks
/// into running accumulators (no full endpoint multiset). The result must equal
/// the materializing path — checked here on a hand-built chain graph.
#[test]
fn streaming_aggregate_over_traversal_is_exact() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    // a(1)->b(2)->c(3), a(1)->d(4); scores 2,3,4 reachable at 1..2 hops from a.
    execute(
        &super::parse(
            "INSERT (a:P {g: 0, s: 1})-[:R]->(b:P {g: 1, s: 2})-[:R]->(c:P {g: 1, s: 3}), \
                 (a)-[:R]->(d:P {g: 0, s: 4})",
        )
        .unwrap(),
        &mut st,
    )
    .unwrap();
    let one = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x,
        other => panic!("not a number: {other:?}"),
    };
    // Scalar over a 1..2-hop reach from a: endpoints b(2), d(4) at hop 1; c(3) at
    // hop 2 → {2,4,3}. min=2, max=4, sum=9, count=3.
    let base = "MATCH (a:P {s: 1})-[:R]->{1,2}(x)";
    assert_eq!(one(&format!("{base} RETURN min(x.s) AS v")), 2.0);
    assert_eq!(one(&format!("{base} RETURN max(x.s) AS v")), 4.0);
    assert_eq!(one(&format!("{base} RETURN sum(x.s) AS v")), 9.0);
    assert_eq!(one(&format!("{base} RETURN count(*) AS v")), 3.0);
    // Grouped by a property: g=1 for {b,c}=(2,3), g=0 for {d}=(4). sums 5 and 4.
    let rows = run(
        &super::parse("MATCH (p:P)-[:R]->(q) RETURN q.g AS g, sum(q.s) AS v").unwrap(),
        &st,
    );
    let mut got: Vec<(i64, i64)> = rows
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Num(g), Value::Num(v)) => (*g as i64, *v as i64),
            _ => panic!(),
        })
        .collect();
    got.sort();
    // q.g over edges a->b(1), b->c(1), a->d(0): group 1 sums s of {b,c}=2+3=5;
    // group 0 sums s of {d}=4.
    assert_eq!(got, vec![(0, 4), (1, 5)]);
}

/// `DISTINCT … LIMIT k` streams with incremental dedup and stops at `k` distinct
/// rows — it must equal the first `k` of the full distinct (first-seen order) and
/// never exceed the total distinct count.
#[test]
fn streaming_distinct_limit_equals_full_prefix() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    // Nodes with ages cycling 0..4 so there are exactly 5 distinct target ages.
    let mut q = String::from("INSERT ");
    for i in 0..20 {
        if i > 0 {
            q.push_str(", ");
        }
        q.push_str(&format!("(:P {{age: {}}})", i % 5));
    }
    execute(&super::parse(&q).unwrap(), &mut st).unwrap();
    // Give every P a self-ish edge so a hop exists (chain them a->a+1).
    execute(
        &super::parse("MATCH (a:P), (b:P) WHERE a.age = 0 AND b.age = 1 CREATE (a)-[:R]->(b)")
            .unwrap_or_else(|_| super::parse("MATCH (p:P) RETURN p.age AS a").unwrap()),
        &mut st,
    )
    .ok();
    let rows = |query: &str| {
        let mut v: Vec<String> = run(&super::parse(query).unwrap(), &st)
            .rows
            .iter()
            .map(|r| format!("{r:?}"))
            .collect();
        v.sort();
        v
    };
    // Over a plain scan (streamable): DISTINCT age has 5 values.
    let full = rows("MATCH (p:P) RETURN DISTINCT p.age AS x");
    assert_eq!(full.len(), 5);
    // LIMIT 3 yields exactly 3 distinct rows, all a subset of the full set.
    let lim = rows("MATCH (p:P) RETURN DISTINCT p.age AS x LIMIT 3");
    assert_eq!(lim.len(), 3);
    assert!(lim.iter().all(|r| full.contains(r)));
    // LIMIT beyond the total returns exactly the full distinct set.
    let big = rows("MATCH (p:P) RETURN DISTINCT p.age AS x LIMIT 999");
    assert_eq!(big, full);
}

/// A bare `LIMIT`/`SKIP` (no ORDER BY) over a filtered/expanded chain streams the
/// source in blocks and stops early — and must return EXACTLY the same rows the
/// full materialize-then-slice would (the block order preserves scan order).
#[test]
fn streaming_limit_equals_full_prefix() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    // A chain of nodes 0->1->…; each has age = id, so a filter + expand + limit
    // is non-trivial and the count fast-paths don't apply.
    execute(
        &super::parse(
            "INSERT (a:P {age: 1})-[:R]->(b:P {age: 2})-[:R]->(c:P {age: 3}), \
                 (d:P {age: 4})-[:R]->(e:P {age: 5}), (f:P {age: 6})-[:R]->(g:P {age: 7})",
        )
        .unwrap(),
        &mut st,
    )
    .unwrap();
    let rows = |q: &str| {
        run(&super::parse(q).unwrap(), &st)
            .rows
            .iter()
            .map(|r| format!("{r:?}"))
            .collect::<Vec<_>>()
    };
    let base = "MATCH (a:P)-[:R]->(b) WHERE b.age > 2 RETURN b.age AS x";
    let full = rows(base);
    assert!(full.len() >= 2, "need enough rows to slice");
    // LIMIT streams; it must equal the full result's prefix.
    let lim = rows(&format!("{base} LIMIT 2"));
    assert_eq!(lim, full[..2].to_vec());
    // SKIP + LIMIT streams to skip+limit then slices — equal to the full window.
    let win = rows(&format!("{base} SKIP 1 LIMIT 2"));
    let end = 3.min(full.len());
    assert_eq!(win, full[1..end].to_vec());
    // A LIMIT larger than the result returns everything (no truncation).
    let big = rows(&format!("{base} LIMIT 10000"));
    assert_eq!(big, full);
}

/// The vectorized finite→finite unary numeric functions (abs/floor/ceil/round/
/// sign) must produce exactly what the boxed `scalar_num_fn` does, so an
/// aggregate over them is correct. Known inputs, hand-computed sums.
#[test]
fn vectorized_unary_num_fns_are_exact() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    // `p.v - 5` yields -3 / 2.5 / 0 (a raw Num column via the Arith fast path),
    // which the unary functions then vectorize.
    execute(
        &super::parse("INSERT (:P {v: 2.0}), (:P {v: 7.5}), (:P {v: 5.0})").unwrap(),
        &mut st,
    )
    .unwrap();
    let s = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => *x,
        other => panic!("not a number: {other:?}"),
    };
    assert_eq!(s("MATCH (p:P) RETURN sum(abs(p.v - 5)) AS s"), 5.5); // 3 + 2.5 + 0
    assert_eq!(s("MATCH (p:P) RETURN sum(floor(p.v - 5)) AS s"), -1.0); // -3 + 2 + 0
    assert_eq!(s("MATCH (p:P) RETURN sum(ceil(p.v - 5)) AS s"), 0.0); // -3 + 3 + 0
    assert_eq!(s("MATCH (p:P) RETURN sum(sign(p.v - 5)) AS s"), 0.0); // -1 + 1 + 0
    assert_eq!(s("MATCH (p:P) RETURN sum(round(p.v - 5)) AS s"), 0.0); // -3 + 3(2.5→3) + 0
}

/// The `count(*)`-over-VarLength fast path must equal the materializing path's
/// row count for every quantifier / direction — including trail exclusion on the
/// cycles in `social`. Guards `try_varlen_count`.
#[test]
fn varlen_count_matches_materialized() {
    let store = social();
    let count_star = |q: &str| match &run(&super::parse(q).unwrap(), &store).rows[0][0] {
        Value::Num(x) => *x as usize,
        other => panic!("not a count: {other:?}"),
    };
    let materialized = |q: &str| run(&super::parse(q).unwrap(), &store).rows.len();
    for (ct, mt) in [
        (
            "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN count(*) AS c",
            "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN b.name AS b",
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->{1,3}(b) RETURN count(*) AS c",
            "MATCH (a:Person)-[:KNOWS]->{1,3}(b) RETURN b.name AS b",
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->{2}(b) RETURN count(*) AS c",
            "MATCH (a:Person)-[:KNOWS]->{2}(b) RETURN b.name AS b",
        ),
        (
            "MATCH (a:Person)<-[:KNOWS]-{1,2}(b) RETURN count(*) AS c",
            "MATCH (a:Person)<-[:KNOWS]-{1,2}(b) RETURN b.name AS b",
        ),
    ] {
        assert_eq!(count_star(ct), materialized(mt), "mismatch for `{ct}`");
    }
    // Unknown edge type → zero paths, and the fast path must agree.
    assert_eq!(
        count_star("MATCH (a:Person)-[:NOPE]->{1,2}(b) RETURN count(*) AS c"),
        0
    );
}

/// `NOT p` over compares / AND / OR is pushed into the raw filter fast paths by
/// inversion; each form must return the SAME rows as its hand-inverted twin.
#[test]
fn not_pushdown_equals_inverted() {
    let store = social();
    let rows = |q: &str| {
        let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(s) => s.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        v.sort();
        v
    };
    // NOT (compare)
    assert_eq!(
        rows("MATCH (p:Person) WHERE NOT p.age > 30 RETURN p.name AS n"),
        rows("MATCH (p:Person) WHERE p.age <= 30 RETURN p.name AS n"),
    );
    // NOT (a AND b) ≡ NOT a OR NOT b
    assert_eq!(
        rows("MATCH (p:Person) WHERE NOT (p.age >= 20 AND p.age < 40) RETURN p.name AS n"),
        rows("MATCH (p:Person) WHERE p.age < 20 OR p.age >= 40 RETURN p.name AS n"),
    );
    // NOT (a OR b) ≡ NOT a AND NOT b
    assert_eq!(
        rows("MATCH (p:Person) WHERE NOT (p.age < 25 OR p.age > 60) RETURN p.name AS n"),
        rows("MATCH (p:Person) WHERE p.age >= 25 AND p.age <= 60 RETURN p.name AS n"),
    );
}

/// A shared-start LINEAR comma pattern `…, (b)-[:R]->(c)` (b bound, c new) folds
/// into a chained expansion — no hash Join — and returns exactly what the join
/// spelling would. Guards the `join/tri` optimization.
#[test]
fn comma_join_linear_folds_to_chain() {
    use crate::ir::{Dir, Expr, Plan};
    let store = social();
    let q = "MATCH (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(c) RETURN c.name AS c";
    let plan = super::parse(q).unwrap();
    // The fold fired: the plan is a chain of Expands, with no Join operator.
    assert!(
        !format!("{plan:?}").contains("Join"),
        "expected the linear comma pattern to fold into a chain, got a Join"
    );
    // …and it equals the same shape written as one chained MATCH.
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .expand(0, Dir::Out, &["KNOWS".to_string()])
    .expand(1, Dir::Out, &["KNOWS".to_string()])
    .project(vec![(
        "c".into(),
        Expr::Prop {
            slot: 2,
            key: "name".into(),
        },
    )]);
    assert_same(q, &hand, &store);
}

/// A cycle-CLOSING comma pattern `(a)-[:R]->(b), (b)-[:R]->(a)` closes correctly:
/// the repeated landing variable `a` becomes an equality join (not a rebind), so
/// every returned `a` genuinely sits on a real 2-cycle. (The plan may fold to a
/// chained expand + equality now that the repeat is handled, rather than a Join.)
#[test]
fn comma_join_cycle_close_keeps_join() {
    let store = social();
    let q = "MATCH (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(a) RETURN a.name AS a";
    let plan = super::parse(q).unwrap();
    // Every returned `a` must genuinely sit on a 2-cycle a->b->a.
    let out = run(&plan, &store);
    for row in &out.rows {
        let Value::Str(name) = &row[0] else {
            panic!("expected a name")
        };
        let back = run(
            &super::parse(&format!(
                "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(a2) \
                     WHERE a.name = '{name}' AND a2.name = '{name}' RETURN a.name AS a"
            ))
            .unwrap(),
            &store,
        );
        assert!(!back.rows.is_empty(), "{name} is not on a real 2-cycle");
    }
}

// --- part 3.5: arithmetic (E1) ---

/// Precedence: `2 + 3 * 4` = 14 (multiply binds tighter).
#[test]
fn arithmetic_precedence() {
    let store = social();
    let out = run(
        &super::parse("MATCH (p:Person) RETURN 2 + 3 * 4 AS x").unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "x")), 14.0);
}

/// Parsed `p.age * 2 + 1` matches the hand-built nested Arith plan.
#[test]
fn arithmetic_parse_matches_hand() {
    use crate::ir::{ArithOp, Expr, Plan};
    let store = social();
    let mul = Expr::Arith {
        op: ArithOp::Mul,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "age".into(),
        }),
        right: Box::new(Expr::Lit(n(2.0))),
    };
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        "x".into(),
        Expr::Arith {
            op: ArithOp::Add,
            left: Box::new(mul),
            right: Box::new(Expr::Lit(n(1.0))),
        },
    )]);
    assert_same("MATCH (p:Person) RETURN p.age * 2 + 1 AS x", &hand, &store);
}

/// Arithmetic in WHERE: `p.age % 2 = 0` keeps even ages (alice 30, carol 40).
#[test]
fn arithmetic_in_where() {
    let store = social();
    let out = run(
        &super::parse("MATCH (p:Person) WHERE p.age % 2 = 0 RETURN p.name AS name").unwrap(),
        &store,
    );
    let mut got: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            _ => panic!(),
        })
        .collect();
    got.sort();
    assert_eq!(got, vec!["alice", "carol"]);
}

/// Unary minus: `-p.age` for alice(30) is -30.
#[test]
fn unary_minus() {
    let store = social();
    let out = run(
        &super::parse("MATCH (p:Person) WHERE p.name = 'alice' RETURN -p.age AS x").unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "x")), -30.0);
}

// --- part 3.6: scalar functions (E2) ---

/// Numeric functions compute; hand-checked on alice(age 30).
#[test]
fn scalar_numeric_functions() {
    let store = social();
    let q = "MATCH (p:Person) WHERE p.name = 'alice' \
                 RETURN abs(-p.age) AS a, floor(p.age / 4) AS f, ceil(p.age / 4) AS c, \
                        round(p.age / 4) AS r, sqrt(p.age - 5) AS s, sign(p.age - 100) AS g";
    let out = run(&super::parse(q).unwrap(), &store);
    assert_eq!(num(&col(&out, 0, "a")), 30.0); // abs(-30)
    assert_eq!(num(&col(&out, 0, "f")), 7.0); // floor(7.5)
    assert_eq!(num(&col(&out, 0, "c")), 8.0); // ceil(7.5)
    assert_eq!(num(&col(&out, 0, "r")), 8.0); // round(7.5) -> 8
    assert_eq!(num(&col(&out, 0, "s")), 5.0); // sqrt(25)
    assert_eq!(num(&col(&out, 0, "g")), -1.0); // sign(30-100)
}

/// A numeric fn on a NULL/non-numeric/negative-sqrt argument yields NULL.
#[test]
fn scalar_fn_null_and_domain() {
    let store = social();
    // proj node has no age → abs(age) is NULL for it.
    let out = run(
        &super::parse("MATCH (n) RETURN abs(n.age) AS a").unwrap(),
        &store,
    );
    assert_eq!(out.rows.iter().filter(|r| r[0].is_null()).count(), 1);
    // sqrt of a negative KEEPS NaN (a real signal), matching lenke-core — it is
    // coerced to null only at JSON egress, not in the result value (K4).
    let out = run(
        &super::parse("MATCH (p:Person) WHERE p.name='alice' RETURN sqrt(0 - p.age) AS s").unwrap(),
        &store,
    );
    assert!(matches!(col(&out, 0, "s"), crate::value::Value::Num(x) if x.is_nan()));
}

/// `_is_nan` / `_is_infinite` / `_is_finite` — TOTAL boolean classifiers over the
/// IEEE-754 special values (leading-underscore extensions). True iff the argument IS
/// that kind; false for everything else (a finite number, null, a string). Never NULL,
/// never a fault — GQL has no NaN/Infinity literal or `IS NAN`, so these are the way to
/// test for the non-finite values that only render as null at JSON egress.
#[test]
fn nonfinite_classifiers() {
    use crate::value::Value;
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) WHERE p.name='alice' RETURN \
                 _is_nan(sqrt(0 - p.age)) AS a, _is_nan(p.age) AS b, _is_nan(1e400) AS c, \
                 _is_infinite(1e400) AS d, _is_infinite(0 - 1e400) AS e, _is_infinite(p.age) AS f, \
                 _is_finite(p.age) AS g, _is_finite(1e400) AS h, _is_finite(sqrt(0 - p.age)) AS i, \
                 _is_nan(null) AS j, _is_infinite(null) AS k, _is_finite('x') AS l",
        )
        .unwrap(),
        &store,
    );
    let t = |k: &str| matches!(col(&out, 0, k), Value::Bool(true));
    assert!(t("a")); // sqrt(-30) is NaN
    assert!(!t("b") && !t("c")); // a finite / an infinity is not NaN
    assert!(t("d") && t("e")); // ±1e400 overflow to ±∞
    assert!(!t("f")); // a finite is not infinite
    assert!(t("g")); // a finite is finite
    assert!(!t("h") && !t("i")); // ∞ / NaN are not finite
                                 // Total: null / non-number → the DEFINITE boolean false, never null.
    assert!(matches!(col(&out, 0, "j"), Value::Bool(false)));
    assert!(matches!(col(&out, 0, "k"), Value::Bool(false)));
    assert!(matches!(col(&out, 0, "l"), Value::Bool(false)));
}

/// A stored ±Infinity (an overflowing property literal) is a DISTINCT present value
/// (Model B): `IS NULL` is false, it orders (−∞ < finite < +∞), aggregates count it,
/// and it renders null only at egress. (`sum` NaN-poisons on +∞ + −∞.)
#[test]
fn stored_infinity_is_a_value() {
    use crate::value::Value;
    let mut b = crate::store::Builder::default();
    b.node(
        &["N"],
        &[("k", Value::Num(1.0)), ("v", Value::Num(f64::INFINITY))],
    );
    b.node(
        &["N"],
        &[("k", Value::Num(2.0)), ("v", Value::Num(f64::NEG_INFINITY))],
    );
    b.node(&["N"], &[("k", Value::Num(3.0)), ("v", Value::Num(2.5))]);
    let store = b.build();
    // IS NULL is false for the ±∞ rows — they are present values.
    let out = run(
        &super::parse("MATCH (n:N) WHERE n.v IS NULL RETURN count(*) AS c").unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "c")), 0.0);
    // Ordered: −∞ (k=2), 2.5 (k=3), +∞ (k=1).
    let out = run(
        &super::parse("MATCH (n:N) RETURN n.k AS k ORDER BY n.v, n.k").unwrap(),
        &store,
    );
    let ks: Vec<f64> = (0..out.rows.len())
        .map(|i| num(&col(&out, i, "k")))
        .collect();
    assert_eq!(ks, vec![2.0, 3.0, 1.0]);
    // Aggregate: count all 3; sum = +∞ + −∞ = NaN.
    let out = run(
        &super::parse("MATCH (n:N) RETURN count(n.v) AS c, sum(n.v) AS s").unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "c")), 3.0);
    assert!(matches!(col(&out, 0, "s"), Value::Num(x) if x.is_nan()));
    // `_is_finite` filters to just the finite row.
    let out = run(
        &super::parse("MATCH (n:N) WHERE _is_finite(n.v) RETURN n.k AS k").unwrap(),
        &store,
    );
    assert_eq!(out.rows.len(), 1);
    assert_eq!(num(&col(&out, 0, "k")), 3.0);
}

/// `CONTAINS` / `STARTS WITH` / `ENDS WITH` infix predicates desugar to the
/// scalar functions and filter three-valued (a NULL operand drops the row).
#[test]
fn string_infix_predicates() {
    let mut b = crate::store::Builder::default();
    b.node(&["N"], &[("s", crate::value::Value::Str("carol".into()))]);
    b.node(&["N"], &[("s", crate::value::Value::Str("bob".into()))]);
    b.node(&["N"], &[]); // s absent → NULL
    let store = b.build();
    let names = |q: &str| -> Vec<String> {
        let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| match &r[0] {
                crate::value::Value::Str(x) => x.to_string(),
                o => format!("{o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        names("MATCH (n:N) WHERE n.s CONTAINS 'o' RETURN n.s AS s"),
        vec!["bob", "carol"]
    );
    assert_eq!(
        names("MATCH (n:N) WHERE n.s STARTS WITH 'ca' RETURN n.s AS s"),
        vec!["carol"]
    );
    assert_eq!(
        names("MATCH (n:N) WHERE n.s ENDS WITH 'ob' RETURN n.s AS s"),
        vec!["bob"]
    );
    // NULL operand → UNKNOWN → dropped (the s-absent node never matches).
    assert_eq!(
        names("MATCH (n:N) WHERE n.s CONTAINS 'zzz' RETURN n.s AS s"),
        Vec::<String>::new()
    );
}

/// `coalesce` returns the first non-null argument.
#[test]
fn coalesce_first_non_null() {
    let store = social();
    // proj has name but no age: coalesce(age, 99) = 99 for proj, real age else.
    let out = run(
        &super::parse("MATCH (n) WHERE n.name = 'graphdb' RETURN coalesce(n.age, 99) AS x")
            .unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "x")), 99.0);
    let out = run(
        &super::parse("MATCH (p:Person) WHERE p.name='alice' RETURN coalesce(p.age, 99) AS x")
            .unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "x")), 30.0);
}

/// Parsed `abs(p.age)` matches the hand-built Call plan.
#[test]
fn scalar_fn_parse_matches_hand() {
    use crate::ir::{Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        "a".into(),
        Expr::Call {
            name: "abs".into(),
            args: vec![Expr::Prop {
                slot: 0,
                key: "age".into(),
            }],
        },
    )]);
    assert_same("MATCH (p:Person) RETURN abs(p.age) AS a", &hand, &store);
}

#[test]
fn scalar_fn_errors() {
    assert!(super::parse("MATCH (p:Person) RETURN nope(p.age) AS x").is_err()); // unknown fn
    assert!(super::parse("MATCH (p:Person) RETURN abs(p.age, 1) AS x").is_err()); // arity
    assert!(super::parse("MATCH (p:Person) RETURN coalesce() AS x").is_err());
    // arity
}

// --- part 3.7: CASE (E3) ---

/// Searched CASE picks the first true branch; ELSE otherwise. Ages 30/25/40:
/// >=40 → "old", >=30 → "mid", else "young".
#[test]
fn case_branch_selection() {
    let store = social();
    let q = "MATCH (p:Person) RETURN p.name AS name, \
                 CASE WHEN p.age >= 40 THEN 'old' WHEN p.age >= 30 THEN 'mid' ELSE 'young' END AS band";
    let out = run(&super::parse(q).unwrap(), &store);
    let mut got: Vec<(String, String)> = out
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Str(a), Value::Str(b)) => (a.to_string(), b.to_string()),
            _ => panic!("shape"),
        })
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            ("alice".into(), "mid".into()), // 30
            ("bob".into(), "young".into()), // 25
            ("carol".into(), "old".into()), // 40
        ]
    );
}

/// No ELSE and no matching branch → NULL; a NULL/false condition is skipped
/// (proj has no age, so `p.age >= 30` is UNKNOWN → skipped → NULL, no ELSE).
#[test]
fn case_no_else_and_null_condition() {
    let store = social();
    let out = run(
        &super::parse("MATCH (n) RETURN CASE WHEN n.age >= 30 THEN 'y' END AS x").unwrap(),
        &store,
    );
    // alice(30),carol(40) → 'y'; bob(25) → NULL; proj(no age) → NULL.
    let ys = out
        .rows
        .iter()
        .filter(|r| matches!(&r[0], Value::Str(s) if &**s == "y"))
        .count();
    let nulls = out.rows.iter().filter(|r| r[0].is_null()).count();
    assert_eq!(ys, 2);
    assert_eq!(nulls, 2);
}

/// Parsed CASE matches the hand-built `Expr::Case`.
#[test]
fn case_parse_matches_hand() {
    use crate::ir::{CompareOp, Expr, Plan};
    let store = social();
    let cond = Expr::Compare {
        op: CompareOp::Ge,
        left: Box::new(Expr::Prop {
            slot: 0,
            key: "age".into(),
        }),
        right: Box::new(Expr::Lit(n(30.0))),
    };
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        "x".into(),
        Expr::Case {
            branches: vec![(cond, Expr::Lit(s("sr")))],
            otherwise: Some(Box::new(Expr::Lit(s("jr")))),
        },
    )]);
    assert_same(
        "MATCH (p:Person) RETURN CASE WHEN p.age >= 30 THEN 'sr' ELSE 'jr' END AS x",
        &hand,
        &store,
    );
}

#[test]
fn case_errors() {
    // The simple form is now supported (desugars to searched CASE).
    assert!(super::parse("MATCH (p:Person) RETURN CASE p.age WHEN 30 THEN 'x' END AS y").is_ok());
    assert!(super::parse("MATCH (p:Person) RETURN CASE WHEN p.age >= 30 THEN 'x' AS y").is_err());
    // no END
}

/// The simple CASE form `CASE <subject> WHEN <v> THEN …` desugars to searched
/// CASE (`WHEN subject = v`); a NULL subject matches no branch (3VL) → ELSE.
#[test]
fn simple_case_form() {
    let store = social();
    let val =
        |q: &str| -> String { format!("{:?}", run(&super::parse(q).unwrap(), &store).rows[0][0]) };
    assert_eq!(
        val("RETURN CASE 5 WHEN 1 THEN 'a' WHEN 5 THEN 'b' ELSE 'c' END AS r"),
        "Str(\"b\")"
    );
    assert_eq!(
        val("RETURN CASE 42 WHEN 1 THEN 'a' ELSE 'c' END AS r"),
        "Str(\"c\")"
    );
    // A NULL subject never equals a WHEN value → falls to ELSE.
    assert_eq!(
            val("MATCH (p:Person) WHERE p.name = 'alice' RETURN CASE p.nope WHEN 1 THEN 'a' ELSE 'none' END AS r"),
            "Str(\"none\")"
        );
}

/// `ORDER BY … NULLS FIRST|LAST` overrides the default null placement (last).
#[test]
fn order_by_nulls_first_last() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"age\":30}}\n",
        "{\"id\":\"b\",\"labels\":[\"P\"],\"props\":{\"age\":null}}\n",
        "{\"id\":\"c\",\"labels\":[\"P\"],\"props\":{\"age\":40}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let col0 = |q: &str| -> Vec<String> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| format!("{:?}", r[0]))
            .collect()
    };
    // ASC NULLS FIRST puts the null ahead of 30, 40.
    assert_eq!(
        col0("MATCH (n:P) RETURN n.age AS age ORDER BY n.age ASC NULLS FIRST"),
        vec!["Null", "Num(30.0)", "Num(40.0)"]
    );
    // DESC NULLS LAST keeps the null after 40, 30.
    assert_eq!(
        col0("MATCH (n:P) RETURN n.age AS age ORDER BY n.age DESC NULLS LAST"),
        vec!["Num(40.0)", "Num(30.0)", "Null"]
    );
}

/// String-literal backslash escapes decode to their characters; `\\uXXXX` /
/// `\\UXXXXXX` are code points; a malformed unicode escape is a syntax error.
#[test]
fn string_escapes() {
    let store = social();
    let val = |q: &str| -> String {
        match run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Str(ref s) => s.to_string(),
            ref o => panic!("want str, got {o:?}"),
        }
    };
    assert_eq!(val(r"RETURN '\n' AS r"), "\n");
    assert_eq!(val(r"RETURN '\t' AS r"), "\t");
    assert_eq!(val(r"RETURN '\\' AS r"), "\\");
    assert_eq!(val(r"RETURN '\'' AS r"), "'");
    assert_eq!(val(r"RETURN '\u0041' AS r"), "A");
    assert_eq!(val(r"RETURN '\U01F600' AS r"), "\u{1F600}");
    // A malformed \u escape is rejected (agreeing with core).
    assert!(super::parse(r"RETURN '\uH' AS x").is_err());
}

/// `x IS [NOT] LABELED L` tests the element's label set (the keyword form of
/// the `x:L` predicate).
#[test]
fn is_labeled_predicate() {
    let store = social();
    let n = |q: &str| -> f64 {
        match run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("want num, got {o:?}"),
        }
    };
    let total = n("MATCH (x) RETURN count(*) AS c");
    let persons = n("MATCH (x) WHERE x IS LABELED Person RETURN count(*) AS c");
    assert!(persons > 0.0 && persons <= total);
    // IS NOT LABELED is the complement.
    assert_eq!(
        n("MATCH (x) WHERE x IS NOT LABELED Person RETURN count(*) AS c"),
        total - persons
    );
    // Agrees with the `x:Label` predicate form.
    assert_eq!(persons, n("MATCH (x) WHERE x:Person RETURN count(*) AS c"));
}

/// Node label algebra: conjunction `:A&B`, disjunction `:A|B`, negation `:!A`,
/// wildcard `:%`, and the `IS L` introducer, over multi-label nodes.
#[test]
fn node_label_algebra() {
    let nd = concat!(
        "{\"id\":\"pa\",\"labels\":[\"Person\",\"Admin\"],\"props\":{}}\n",
        "{\"id\":\"p\",\"labels\":[\"Person\"],\"props\":{}}\n",
        "{\"id\":\"s\",\"labels\":[\"Software\"],\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let n = |q: &str| -> f64 {
        match run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("want num, got {o:?}"),
        }
    };
    assert_eq!(n("MATCH (x:Person&Admin) RETURN count(*) AS c"), 1.0); // only pa
    assert_eq!(n("MATCH (x:Person|Software) RETURN count(*) AS c"), 3.0); // pa, p, s
    assert_eq!(n("MATCH (x:!Software) RETURN count(*) AS c"), 2.0); // pa, p
    assert_eq!(n("MATCH (x:%) RETURN count(*) AS c"), 3.0); // any label
    assert_eq!(n("MATCH (x IS Person) RETURN count(*) AS c"), 2.0); // pa, p (= :Person)
}

/// A landing (non-seed) node's label constrains the hop, as core does — in a
/// plain MATCH and inside a COUNT{}/EXISTS{} subquery body.
#[test]
fn landing_node_label() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"name\":\"a\"}}\n",
        "{\"id\":\"t\",\"labels\":[\"N\",\"Target\"],\"props\":{}}\n",
        "{\"id\":\"x\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"t\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"x\",\"type\":\"R\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let n = |q: &str| -> f64 {
        match run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("want num, got {o:?}"),
        }
    };
    // Plain MATCH: a -> Target lands only on `t` (1), not `x`.
    assert_eq!(n("MATCH (a)-[:R]->(b:Target) RETURN count(*) AS c"), 1.0);
    // Without the label, both neighbours count.
    assert_eq!(n("MATCH (a)-[:R]->(b) RETURN count(*) AS c"), 2.0);
    // The same constraint inside a COUNT{} subquery body.
    assert_eq!(
        n("MATCH (a {name:'a'}) RETURN COUNT { (a)-[:R]->(:Target) } AS c"),
        1.0
    );
}

/// The cross-type total order (Num < Str < Bool < Temporal < compound < Null,
/// matching core) drives ORDER BY / min / max over a mixed-type column.
#[test]
fn mixed_type_total_order() {
    let nd = concat!(
        "{\"id\":\"1\",\"labels\":[\"X\"],\"props\":{\"v\":2}}\n",
        "{\"id\":\"2\",\"labels\":[\"X\"],\"props\":{\"v\":\"a\"}}\n",
        "{\"id\":\"3\",\"labels\":[\"X\"],\"props\":{\"v\":1}}\n",
        "{\"id\":\"4\",\"labels\":[\"X\"],\"props\":{\"v\":true}}\n",
        "{\"id\":\"5\",\"labels\":[\"X\"],\"props\":{\"v\":\"b\"}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let col0 = |q: &str| -> Vec<String> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| format!("{:?}", r[0]))
            .collect()
    };
    // ORDER BY asc: numbers, then strings, then bool.
    assert_eq!(
        col0("MATCH (n:X) RETURN n.v AS v ORDER BY n.v"),
        vec![
            "Num(1.0)",
            "Num(2.0)",
            "Str(\"a\")",
            "Str(\"b\")",
            "Bool(true)"
        ]
    );
    // min = the smallest number; max = the bool (highest rank present).
    assert_eq!(col0("MATCH (n:X) RETURN min(n.v) AS m"), vec!["Num(1.0)"]);
    assert_eq!(col0("MATCH (n:X) RETURN max(n.v) AS m"), vec!["Bool(true)"]);
}

/// `ORDER BY` over a group-key EXPRESSION works under implicit grouping — the
/// key column is ordered even though the bindings are gone post-aggregation.
#[test]
fn grouped_order_by_group_key() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"city\":\"z\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"P\"],\"props\":{\"city\":\"a\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"P\"],\"props\":{\"city\":\"a\"}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    // Group by city, count, ORDER BY the group-key expression n.city.
    let rows: Vec<String> = run(
        &super::parse("MATCH (n:P) RETURN n.city, count(*) AS c ORDER BY n.city").unwrap(),
        &store,
    )
    .rows
    .iter()
    .map(|r| format!("{:?},{:?}", r[0], r[1]))
    .collect();
    assert_eq!(rows, vec!["Str(\"a\"),Num(2.0)", "Str(\"z\"),Num(1.0)"]);
}

/// `TIMESTAMP '…'` is core's alias for a (local) DATETIME literal.
#[test]
fn timestamp_is_datetime_alias() {
    let store = social();
    let val = |q: &str| format!("{:?}", run(&super::parse(q).unwrap(), &store).rows[0][0]);
    // TIMESTAMP parses and compares equal to the same DATETIME literal.
    assert_eq!(
        val("RETURN TIMESTAMP '2021-06-15T08:30:00' = DATETIME '2021-06-15T08:30:00' AS x"),
        "Bool(true)"
    );
    assert_eq!(
        val("RETURN TIMESTAMP '2021-06-15T08:30:00.5' >= DATETIME '2021-06-15T08:30:00' AS x"),
        "Bool(true)"
    );
}

/// An aggregate nested in a projection expression (`count(*) + 1`) hoists the
/// aggregate into the group and projects the surrounding arithmetic over it.
#[test]
fn aggregate_in_projection_expression() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"T\"],\"props\":{}}\n",
        "{\"id\":\"b\",\"labels\":[\"T\"],\"props\":{}}\n",
        "{\"id\":\"c\",\"labels\":[\"T\"],\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let n = |q: &str| -> f64 {
        match run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("want num, got {o:?}"),
        }
    };
    assert_eq!(n("MATCH (t:T) RETURN count(*) + 1 AS c"), 4.0);
    assert_eq!(n("MATCH (t:T) RETURN count(*) * 2 - 1 AS c"), 5.0);
    // A bare aggregate is unaffected.
    assert_eq!(n("MATCH (t:T) RETURN count(*) AS c"), 3.0);
}

/// A label EXPRESSION in a WHERE predicate (`x:A|B`, `x:A&B`, `x:!A`) lowers via
/// the shared label-expression lowering, like the pattern-position label algebra.
#[test]
fn label_expr_in_predicate() {
    let nd = concat!(
        "{\"id\":\"pa\",\"labels\":[\"Person\",\"Admin\"],\"props\":{}}\n",
        "{\"id\":\"p\",\"labels\":[\"Person\"],\"props\":{}}\n",
        "{\"id\":\"s\",\"labels\":[\"Software\"],\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let n = |q: &str| -> f64 {
        match run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("want num, got {o:?}"),
        }
    };
    assert_eq!(
        n("MATCH (x) WHERE x:Person|Software RETURN count(*) AS c"),
        3.0
    );
    assert_eq!(
        n("MATCH (x) WHERE x:Person&Admin RETURN count(*) AS c"),
        1.0
    );
    assert_eq!(n("MATCH (x) WHERE x:!Software RETURN count(*) AS c"), 2.0);
    // A single label is unchanged.
    assert_eq!(n("MATCH (x) WHERE x:Person RETURN count(*) AS c"), 2.0);
}

/// A reverse-correlated COUNT/EXISTS subquery — the outer variable is the hop's
/// LANDING (`COUNT { (m)-[:R]->(n) }`), so the body traverses from the bound
/// endpoint backward. In-degree, incoming direction, and a local-node label all
/// resolve correctly.
#[test]
fn reverse_correlated_subquery() {
    let nd = concat!(
        "{\"id\":\"n0\",\"labels\":[\"Node\"],\"props\":{\"name\":\"n0\"}}\n",
        "{\"id\":\"n1\",\"labels\":[\"Node\"],\"props\":{\"name\":\"n1\"}}\n",
        "{\"id\":\"n2\",\"labels\":[\"Node\"],\"props\":{\"name\":\"n2\"}}\n",
        "{\"id\":\"e1\",\"from\":\"n0\",\"to\":\"n1\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e2\",\"from\":\"n0\",\"to\":\"n1\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e3\",\"from\":\"n0\",\"to\":\"n2\",\"type\":\"R\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let n = |q: &str| -> f64 {
        match run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("want num, got {o:?}"),
        }
    };
    // in-degree of n1 = 2.
    assert_eq!(
        n("MATCH (n:Node) WHERE n.name='n1' RETURN COUNT { (m)-[:R]->(n) } AS c"),
        2.0
    );
    // out-degree of n0 via incoming arrow at the local node = 3.
    assert_eq!(
        n("MATCH (n:Node) WHERE n.name='n0' RETURN COUNT { (m)<-[:R]-(n) } AS c"),
        3.0
    );
    // local-node label filter narrows the reverse hop.
    assert_eq!(
        n("MATCH (n:Node) WHERE n.name='n1' RETURN COUNT { (m:Node)-[:R]->(n) } AS c"),
        2.0
    );
}

/// A single-edge parenthesized subpath group `((x)-[e:R]->(y)){n,m}(t)` lowers
/// to a variable-length hop to the endpoint (endpoint-only; group vars ignored).
#[test]
fn subpath_group_single_edge() {
    // chain a -> b -> c -> d
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e2\",\"from\":\"b\",\"to\":\"c\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e3\",\"from\":\"c\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| format!("{:?}", r[0]))
            .collect();
        v.sort();
        v
    };
    // 1..2 reps from a: b (1), c (2).
    assert_eq!(
        ids("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,2} (t) RETURN t.id AS id"),
        vec!["Str(\"b\")", "Str(\"c\")"]
    );
    // Anonymous inner nodes + exact {2}: only c.
    assert_eq!(
        ids("MATCH (s:N {id:'a'}) (()-[:R]->()){2} (t) RETURN t.id AS id"),
        vec!["Str(\"c\")"]
    );
    // Unanchored group: every 1-rep landing = b, c, d.
    assert_eq!(
        ids("MATCH ((x)-[:R]->(y)){1,1} (t) RETURN t.id AS id"),
        vec!["Str(\"b\")", "Str(\"c\")", "Str(\"d\")"]
    );
}

// --- part 3.8: string functions (E4a) ---

/// upper/lower/trim/length/substring/replace on alice's name — hand-computed.
#[test]
fn string_functions() {
    let store = social();
    let q = "MATCH (p:Person) WHERE p.name = 'alice' RETURN \
                 upper(p.name) AS u, char_length(p.name) AS l, substring(p.name, 1, 3) AS sub, \
                 replace(p.name, 'a', 'A') AS rep";
    let out = run(&super::parse(q).unwrap(), &store);
    assert!(matches!(col(&out, 0, "u"), Value::Str(x) if &*x == "ALICE"));
    assert_eq!(num(&col(&out, 0, "l")), 5.0); // "alice"
    assert!(matches!(col(&out, 0, "sub"), Value::Str(x) if &*x == "ali")); // ISO 1-based [1,4)
    assert!(matches!(col(&out, 0, "rep"), Value::Str(x) if &*x == "Alice"));
}

/// String predicates return Bool; a non-string / null argument yields NULL.
#[test]
fn string_predicates_and_null() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) WHERE p.name='alice' \
                 RETURN starts_with(p.name,'al') AS s, contains(p.name,'zz') AS c",
        )
        .unwrap(),
        &store,
    );
    assert!(matches!(col(&out, 0, "s"), Value::Bool(true)));
    assert!(matches!(col(&out, 0, "c"), Value::Bool(false)));
    // upper() of a number is now a TYPE ERROR (not a silent NULL) — no implicit coercion.
    assert!(crate::exec::try_run(
        &super::parse("MATCH (p:Person) RETURN upper(p.age) AS bad").unwrap(),
        &store
    )
    .unwrap_err()
    .contains("E_INVALID_VALUE"));
}

/// substring past the end clamps; a negative index is NULL.
#[test]
fn substring_edges() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) WHERE p.name='alice' \
                 RETURN substring(p.name, 3) AS tail, substring(p.name, 10) AS past",
        )
        .unwrap(),
        &store,
    );
    assert!(matches!(col(&out, 0, "tail"), Value::Str(x) if &*x == "ice")); // ISO 1-based, from unit 2
    assert!(matches!(col(&out, 0, "past"), Value::Str(x) if x.is_empty())); // clamped
                                                                            // A start <= 0 shrinks the window from the front (SQL semantics), so it
                                                                            // returns the whole string — matching core, NOT NULL.
    let neg = run(
        &super::parse("MATCH (p:Person) WHERE p.name='alice' RETURN substring(p.name, -1) AS x")
            .unwrap(),
        &store,
    );
    assert!(matches!(col(&neg, 0, "x"), Value::Str(s) if &*s == "alice"));
}

/// OPTIONAL MATCH is a left-outer hop: a node with no matching neighbour survives
/// with the optional variable NULL; `count(x)` skips those nulls.
#[test]
fn optional_match_left_outer() {
    let store = social(); // alice-KNOWS->bob, alice-KNOWS->carol, bob-KNOWS->carol
    let q = "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name AS an, b.name AS bn";
    let out = run(&super::parse(q).unwrap(), &store);
    // carol has no outgoing KNOWS → one row (carol, null).
    let mut pairs: Vec<(String, Option<String>)> = out
        .rows
        .iter()
        .map(|r| {
            let a = match &r[0] {
                Value::Str(s) => s.to_string(),
                o => format!("{o:?}"),
            };
            let b = match &r[1] {
                Value::Str(s) => Some(s.to_string()),
                Value::Null => None,
                o => Some(format!("{o:?}")),
            };
            (a, b)
        })
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("alice".into(), Some("bob".into())),
            ("alice".into(), Some("carol".into())),
            ("bob".into(), Some("carol".into())),
            ("carol".into(), None), // left-outer: kept with NULL
        ]
    );
    // count(x) over the optional skips the null (carol → 0).
    let counts = run(
        &super::parse(
            "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name AS an, count(b) AS c",
        )
        .unwrap(),
        &store,
    );
    let carol = counts
        .rows
        .iter()
        .find(|r| matches!(&r[0], Value::Str(s) if &**s == "carol"))
        .unwrap();
    assert!(matches!(carol[1], Value::Num(x) if x == 0.0));
}

/// `UNION` concatenates two query arms' rows and dedups; `UNION ALL` keeps dups;
/// the result's column names come from the LEFT arm.
#[test]
fn union_and_union_all() {
    let mut b = Builder::default();
    b.node(&["P"], &[("v", s("a"))]);
    b.node(&["P"], &[("v", s("a"))]); // duplicate value
    b.node(&["Q"], &[("v", s("b"))]);
    let store = b.build();
    let vals = |q: &str| -> Vec<String> {
        let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(s) => s.to_string(),
                o => format!("{o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    // UNION dedups: {a, a} ∪ {b} → [a, b].
    assert_eq!(
        vals("MATCH (p:P) RETURN p.v AS x UNION MATCH (q:Q) RETURN q.v AS x"),
        vec!["a", "b"]
    );
    // UNION ALL keeps every row: a, a, b.
    assert_eq!(
        vals("MATCH (p:P) RETURN p.v AS x UNION ALL MATCH (q:Q) RETURN q.v AS x"),
        vec!["a", "a", "b"]
    );
    // Column names come from the LEFT arm even if the right differs.
    assert_eq!(
        run(
            &super::parse("MATCH (p:P) RETURN p.v AS x UNION MATCH (q:Q) RETURN q.v AS y").unwrap(),
            &store
        )
        .names,
        vec!["x".to_string()]
    );
}

/// `collect_list(x)` gathers a group's values into a list in row order, SKIPPING
/// nulls (core's semantics — distinct from Gremlin fold, which keeps them).
#[test]
fn collect_list_aggregate_skips_nulls() {
    let mut b = Builder::default();
    // dept eng: ages 1, (null), 3 ; dept ops: age 5
    b.node(&["P"], &[("d", s("eng")), ("age", n(1.0))]);
    b.node(&["P"], &[("d", s("eng"))]); // no age → null, dropped by collect_list
    b.node(&["P"], &[("d", s("eng")), ("age", n(3.0))]);
    b.node(&["P"], &[("d", s("ops")), ("age", n(5.0))]);
    let store = b.build();
    let out = run(
        &super::parse("MATCH (p:P) RETURN p.d AS d, collect_list(p.age) AS ages ORDER BY d")
            .unwrap(),
        &store,
    );
    // Groups ordered by d: eng then ops. eng's list is [1, 3] (null skipped).
    let list = |r: usize| match &out.rows[r][1] {
        Value::List(v) => v
            .iter()
            .map(|x| match x {
                Value::Num(n) => *n,
                _ => f64::NAN,
            })
            .collect::<Vec<_>>(),
        _ => panic!("expected a list"),
    };
    assert_eq!(list(0), vec![1.0, 3.0]);
    assert_eq!(list(1), vec![5.0]);
    // `collect` is a superset alias for the same thing.
    assert!(super::parse("MATCH (p:P) RETURN collect(p.age) AS a").is_ok());
}

/// ORDER BY can sort by an UNPROJECTED expression (`ORDER BY n.age` when only
/// `n.name` is returned) — projected as a hidden column, sorted, then dropped.
#[test]
fn order_by_unprojected_expression() {
    let mut b = Builder::default();
    for (nm, age) in [("c", 3.0), ("a", 1.0), ("b", 2.0)] {
        b.node(&["P"], &[("name", s(nm)), ("age", n(age))]);
    }
    let store = b.build();
    let names = |q: &str| -> Vec<String> {
        run(&super::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(s) => s.to_string(),
                o => format!("{o:?}"),
            })
            .collect()
    };
    // Sort by the unprojected age; only name is returned.
    assert_eq!(
        names("MATCH (p:P) RETURN p.name AS nm ORDER BY p.age"),
        vec!["a", "b", "c"]
    );
    assert_eq!(
        names("MATCH (p:P) RETURN p.name AS nm ORDER BY p.age DESC"),
        vec!["c", "b", "a"]
    );
    // The output is exactly the returned column (hidden sort column dropped).
    assert_eq!(
        run(
            &super::parse("MATCH (p:P) RETURN p.name AS nm ORDER BY p.age").unwrap(),
            &store
        )
        .names,
        vec!["nm".to_string()]
    );
}

/// `RETURN r` (a bound edge) renders core's edge element map —
/// `{id, from, to, labels, properties}`.
#[test]
fn return_edge_renders_element_map() {
    let store = social();
    let out = run(
        &super::parse("MATCH (a:Person {name:'alice'})-[r:KNOWS]->(b) RETURN r").unwrap(),
        &store,
    );
    let Value::Map(m) = &out.rows[0][0] else {
        panic!("expected an edge map, got {:?}", out.rows[0][0]);
    };
    let keys: Vec<&str> = m
        .iter()
        .map(|(k, _)| match k {
            Value::Str(s) => s.as_ref(),
            _ => "?",
        })
        .collect();
    assert_eq!(keys, vec!["id", "from", "to", "labels", "properties"]);
    // labels is a list carrying the edge type.
    assert!(matches!(&m[3].1, Value::List(l) if matches!(&l[0], Value::Str(s) if &**s == "KNOWS")));
}

/// `RETURN *` projects every bound variable, in slot (declaration) order, each
/// column named for its variable.
#[test]
fn return_star_expands_bound_vars() {
    let store = social();
    // Two bound node vars → two columns, `a` then `b`, both node maps.
    let out = run(
        &super::parse("MATCH (a:Person {name:'alice'})-[:KNOWS]->(b) RETURN *").unwrap(),
        &store,
    );
    assert_eq!(out.names, vec!["a".to_string(), "b".to_string()]);
    assert!(out
        .rows
        .iter()
        .all(|r| matches!(&r[0], Value::Map(_)) && matches!(&r[1], Value::Map(_))));
    // `*` composes with an explicit item.
    let out2 = run(
        &super::parse("MATCH (n:Person) RETURN *, n.name AS nm").unwrap(),
        &store,
    );
    assert_eq!(out2.names, vec!["n".to_string(), "nm".to_string()]);
}

/// `RETURN n` (a bare node binding) renders the element MAP core produces —
/// `{id, labels(sorted), properties(sorted)}` — not the bare node id.
#[test]
fn return_node_renders_element_map() {
    let store = social();
    let out = run(
        &super::parse("MATCH (p:Person {name:'alice'}) RETURN p").unwrap(),
        &store,
    );
    let Value::Map(m) = &out.rows[0][0] else {
        panic!("expected a node map, got {:?}", out.rows[0][0]);
    };
    // Top-level keys, in order.
    let keys: Vec<&str> = m
        .iter()
        .map(|(k, _)| match k {
            Value::Str(s) => s.as_ref(),
            _ => "?",
        })
        .collect();
    assert_eq!(keys, vec!["id", "labels", "properties"]);
    // labels is a List; properties is a Map carrying name='alice'.
    assert!(
        matches!(&m[1].1, Value::List(l) if matches!(&l[0], Value::Str(s) if &**s == "Person"))
    );
    let Value::Map(props) = &m[2].1 else {
        panic!("properties must be a map")
    };
    assert!(props
            .iter()
            .any(|(k, v)| matches!((k, v), (Value::Str(k), Value::Str(v)) if &**k == "name" && &**v == "alice")));
}

/// An untyped relationship `-[r]->` / `-[]->` traverses edges of ANY type;
/// `alice` has one KNOWS and one WORKS_ON out-edge, so untyped sees both while a
/// `:KNOWS` hop sees only one.
#[test]
fn untyped_relationship_traverses_all_types() {
    let store = social();
    let names = |q: &str| {
        let out = run(&super::parse(q).unwrap(), &store);
        let i = out.names.iter().position(|n| n == "n").expect("column n");
        let mut v: Vec<String> = out
            .rows
            .iter()
            .filter_map(|r| match &r[i] {
                Value::Str(s) => Some(s.to_string()),
                _ => None,
            })
            .collect();
        v.sort();
        v
    };
    // Bare-variable and empty-bracket untyped forms both traverse everything.
    assert_eq!(
        names("MATCH (a:Person {name:'alice'})-[r]->(b) RETURN b.name AS n"),
        vec!["bob", "carol", "graphdb"],
    );
    assert_eq!(
        names("MATCH (a:Person {name:'alice'})-[]->(b) RETURN b.name AS n"),
        vec!["bob", "carol", "graphdb"],
    );
    // A typed hop is narrower.
    assert_eq!(
        names("MATCH (a:Person {name:'alice'})-[:KNOWS]->(b) RETURN b.name AS n"),
        vec!["bob", "carol"],
    );
}

/// Parsed `upper(p.name)` matches the hand-built Call.
#[test]
fn string_fn_parse_matches_hand() {
    use crate::ir::{Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        "u".into(),
        Expr::Call {
            name: "upper".into(),
            args: vec![Expr::Prop {
                slot: 0,
                key: "name".into(),
            }],
        },
    )]);
    assert_same("MATCH (p:Person) RETURN upper(p.name) AS u", &hand, &store);
}

#[test]
fn string_fn_arity_errors() {
    assert!(super::parse("MATCH (p:Person) RETURN upper(p.name, 1) AS x").is_err());
    assert!(super::parse("MATCH (p:Person) RETURN replace(p.name, 'a') AS x").is_err());
}

// --- part 3.9: list literal + list functions (E4b) ---

/// A list literal can hold non-constant elements; size/head/last read it.
#[test]
fn list_literal_and_functions() {
    let store = social();
    let q = "MATCH (p:Person) WHERE p.name='alice' RETURN \
                 size([p.age, 1, 2]) AS n, head([p.age, 1, 2]) AS h, last([p.age, 1, 2]) AS t";
    let out = run(&super::parse(q).unwrap(), &store);
    assert_eq!(num(&col(&out, 0, "n")), 3.0);
    assert_eq!(num(&col(&out, 0, "h")), 30.0); // p.age (alice)
    assert_eq!(num(&col(&out, 0, "t")), 2.0);
}

/// Empty list: size 0, head/last NULL. A list fn on a non-list is NULL.
#[test]
fn empty_list_and_non_list() {
    let store = social();
    let out = run(
        &super::parse(
            "MATCH (p:Person) WHERE p.name='alice' \
                 RETURN size([]) AS z, head([]) AS h",
        )
        .unwrap(),
        &store,
    );
    assert_eq!(num(&col(&out, 0, "z")), 0.0);
    assert!(col(&out, 0, "h").is_null());
    // size() of a number is now a TYPE ERROR — it is polymorphic over string|list only.
    assert!(crate::exec::try_run(
        &super::parse("MATCH (p:Person) RETURN size(p.age) AS bad").unwrap(),
        &store
    )
    .unwrap_err()
    .contains("E_INVALID_VALUE"));
}

/// Parsed `[p.age, 1]` matches the hand-built `Expr::List`.
#[test]
fn list_literal_parse_matches_hand() {
    use crate::ir::{Expr, Plan};
    let store = social();
    let hand = Plan::Scan {
        label: Some("Person".into()),
    }
    .project(vec![(
        "xs".into(),
        Expr::List {
            items: vec![
                Expr::Prop {
                    slot: 0,
                    key: "age".into(),
                },
                Expr::Lit(n(1.0)),
            ],
        },
    )]);
    assert_same("MATCH (p:Person) RETURN [p.age, 1] AS xs", &hand, &store);
}

// --- part 4: INSERT (write statements) ---

/// Parsed INSERT matches the hand-built `Plan::Insert`: execute both onto
/// fresh stores and confirm they answer the same query identically (and that
/// the insert actually happened).
#[test]
fn insert_parse_matches_hand_plan() {
    use crate::exec::execute;
    use crate::ir::{InsertEdge, InsertNode, Plan};
    let hand = Plan::Insert {
        nodes: vec![
            InsertNode {
                labels: vec!["Person".into()],
                props: vec![("name".into(), s("x")), ("age".into(), n(1.0))],
            },
            InsertNode {
                labels: vec!["Person".into()],
                props: vec![("name".into(), s("y"))],
            },
        ],
        edges: vec![InsertEdge {
            from: 0,
            to: 1,
            etype: "KNOWS".into(),
            props: vec![],
        }],
    };
    let query = "INSERT (a:Person {name: 'x', age: 1})-[:KNOWS]->(b:Person {name: 'y'})";
    let mut st_p = Builder::default().build();
    let mut st_h = Builder::default().build();
    execute(&super::parse(query).unwrap(), &mut st_p).unwrap();
    execute(&hand, &mut st_h).unwrap();
    let probe = "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b, a.age AS age";
    let pp = super::parse(probe).unwrap();
    assert_eq!(bag(&run(&pp, &st_p)), bag(&run(&pp, &st_h)));
    assert_eq!(run(&pp, &st_p).rows.len(), 1); // the insert happened
}

/// A repeated variable references the same node, not a new one: `(a) … , (a)…`
/// creates ONE `a`.
#[test]
fn insert_reuses_variable() {
    use crate::exec::execute;
    let mut store = Builder::default().build();
    execute(
        &super::parse("INSERT (a:P {name: 'a'}), (a)-[:R]->(b:P {name: 'b'})").unwrap(),
        &mut store,
    )
    .unwrap();
    let edge = run(
        &super::parse("MATCH (x:P)-[:R]->(y) RETURN x.name AS x, y.name AS y").unwrap(),
        &store,
    );
    assert_eq!(edge.rows.len(), 1);
    let cnt = run(
        &super::parse("MATCH (p:P) RETURN count(*) AS c").unwrap(),
        &store,
    );
    assert_eq!(num(&col(&cnt, 0, "c")), 2.0); // a reused, not duplicated
}

#[test]
fn insert_errors() {
    assert!(super::parse("INSERT (a:P)-[:R]-(b:P)").is_err()); // undirected
    assert!(super::parse("INSERT (a:P {n: 1}), (a:P {n: 2})").is_err()); // redefine var
}

// --- part 5: SET / REMOVE (update statements) ---

/// Parsed SET/REMOVE matches the hand-built `Plan::Update`: run both onto
/// fresh copies and confirm the resulting property reads agree.
#[test]
fn update_parse_matches_hand_plan() {
    use crate::exec::execute;
    use crate::ir::{Expr, Plan, SetOp};
    let hand = Plan::Update {
        input: Box::new(
            Plan::Scan {
                label: Some("Person".into()),
            }
            .filter(Expr::Compare {
                op: crate::ir::CompareOp::Eq,
                left: Box::new(Expr::Prop {
                    slot: 0,
                    key: "name".into(),
                }),
                right: Box::new(Expr::Lit(s("alice"))),
            }),
        ),
        ops: vec![
            SetOp::Set {
                slot: 0,
                key: "age".into(),
                value: Expr::Lit(n(41.0)),
            },
            SetOp::Remove {
                slot: 0,
                key: "name".into(),
            },
        ],
    };
    let query = "MATCH (p:Person) WHERE p.name = 'alice' SET p.age = 41 REMOVE p.name";
    let mut st_p = social();
    let mut st_h = social();
    execute(&super::parse(query).unwrap(), &mut st_p).unwrap();
    execute(&hand, &mut st_h).unwrap();
    // Compare the whole Person table (age + name) between the two stores.
    let probe = "MATCH (p:Person) RETURN p.age AS age, p.name AS name";
    let pp = super::parse(probe).unwrap();
    assert_eq!(bag(&run(&pp, &st_p)), bag(&run(&pp, &st_h)));
}

/// SET only touches WHERE-matched rows; others are unchanged (hand-computed:
/// only alice's age becomes 100).
#[test]
fn update_respects_where() {
    use crate::exec::execute;
    let mut store = social();
    execute(
        &super::parse("MATCH (p:Person) WHERE p.name = 'alice' SET p.age = 100").unwrap(),
        &mut store,
    )
    .unwrap();
    let out = run(
        &super::parse("MATCH (p:Person) RETURN p.name AS name, p.age AS age").unwrap(),
        &store,
    );
    let mut got: Vec<(String, f64)> = out
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Str(nm), Value::Num(a)) => (nm.to_string(), *a),
            _ => panic!("shape"),
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        vec![
            ("alice".into(), 100.0),
            ("bob".into(), 25.0),
            ("carol".into(), 40.0)
        ]
    );
}

/// The null policy: `SET x = null` STORES a present null (has_prop true, reads
/// null); `REMOVE x` makes it absent (has_prop false). Distinct operations.
#[test]
fn set_null_stores_remove_deletes() {
    use crate::exec::execute;
    let mut store = social();
    // node 0 = alice. SET a present null on 'age', remove 'name'.
    execute(
        &super::parse("MATCH (p:Person) WHERE p.name = 'alice' SET p.age = null").unwrap(),
        &mut store,
    )
    .unwrap();
    assert!(store.has_prop(0, "age")); // present…
    assert!(store.prop(0, "age").is_null()); // …but null
    execute(
        &super::parse("MATCH (p:Person) WHERE p.age = 25 REMOVE p.age").unwrap(),
        &mut store,
    )
    .unwrap();
    // bob (age 25) had age removed → absent
    assert!(!store.has_prop(1, "age"));
}

#[test]
fn update_errors_on_unknown_var() {
    assert!(super::parse("MATCH (p:Person) SET q.age = 1").is_err());
}

// --- part 6: _MERGE (keyed upsert) ---

fn user_store() -> Store {
    let mut st = Builder::default().build();
    st.create_unique_constraint("User", &["email"]).unwrap();
    st
}
fn merge(store: &mut Store, q: &str) -> Result<(), String> {
    crate::exec::execute(&super::parse(q).unwrap(), store).map(|_| ())
}

/// Create path: absent key → node created with all pattern props.
#[test]
fn merge_creates_when_absent() {
    let mut st = user_store();
    merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
    assert_eq!(st.nodes_with_label("User").len(), 1);
    assert!(matches!(st.prop(0, "email"), Value::Str(x) if &*x == "a"));
    assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A"));
}

/// Idempotence + default clobber: a second _MERGE on the same key updates the
/// SAME node, clobbering the non-key payload to the new pattern value.
#[test]
fn merge_is_idempotent_and_clobbers_payload() {
    let mut st = user_store();
    merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
    merge(&mut st, "_MERGE (u:User {email: 'a', name: 'B'})").unwrap();
    assert_eq!(st.nodes_with_label("User").len(), 1); // no duplicate
    assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "B")); // clobbered
}

/// `_ON_CREATE SET` fires only on create; `_ON_UPDATE SET` REPLACES the
/// default clobber (so the pattern payload is NOT re-clobbered on update).
#[test]
fn merge_on_create_and_on_update_dispositions() {
    let mut st = user_store();
    merge(
        &mut st,
        "_MERGE (u:User {email: 'a', name: 'A'}) _ON_CREATE SET u.created = true",
    )
    .unwrap();
    assert!(matches!(st.prop(0, "created"), Value::Bool(true)));
    // update with a new name, but _ON_UPDATE replaces the default: name stays
    // 'A', only seen is written; created stays (on_create didn't re-fire).
    merge(
        &mut st,
        "_MERGE (u:User {email: 'a', name: 'C'}) _ON_UPDATE SET u.seen = 1",
    )
    .unwrap();
    assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A")); // NOT clobbered
    assert!(matches!(st.prop(0, "seen"), Value::Num(x) if x == 1.0));
    assert!(matches!(st.prop(0, "created"), Value::Bool(true))); // survived
}

/// A WHERE-gated `_ON_UPDATE` whose predicate is false is a no-op (not an
/// error): the existing value is left untouched.
#[test]
fn merge_on_update_where_gate_false_is_noop() {
    let mut st = user_store();
    merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
    merge(&mut st, "MATCH (u:User) SET u.version = 5").ok(); // seed a version
                                                             // incoming version 3 is not newer → gate false → name unchanged.
    merge(
        &mut st,
        "_MERGE (u:User {email: 'a'}) _ON_UPDATE SET u.name = 'Z' WHERE u.version < 3",
    )
    .unwrap();
    assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A"));
}

/// `_ON_UPDATE_NOTHING` leaves the existing node untouched.
#[test]
fn merge_on_update_nothing() {
    let mut st = user_store();
    merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
    merge(
        &mut st,
        "_MERGE (u:User {email: 'a', name: 'X'}) _ON_UPDATE_NOTHING",
    )
    .unwrap();
    assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A"));
}

/// `_MERGE` on a label with no applicable unique constraint errors.
#[test]
fn merge_without_constraint_errors() {
    let mut st = Builder::default().build(); // no constraint
    assert!(merge(&mut st, "_MERGE (u:User {email: 'a'})").is_err());
}

/// A two-label store (`User.id`, `Team.id` unique) with one vertex of each,
/// for the edge-form `_MERGE` tests.
fn edge_merge_store() -> Store {
    let mut st = Builder::default().build();
    st.create_unique_constraint("User", &["id"]).unwrap();
    st.create_unique_constraint("Team", &["id"]).unwrap();
    merge_all(&mut st, "INSERT (:User {id: 'u1'}), (:Team {id: 't1'})");
    st
}
fn merge_all(store: &mut Store, q: &str) {
    crate::exec::execute(&super::parse(q).unwrap(), store).unwrap();
}

/// Edge `_MERGE`: absent → the edge is created between the two key-matched
/// endpoints with its inline props, then `_ON_CREATE` fires on the edge.
#[test]
fn merge_edge_creates_between_key_matched_endpoints() {
    let mut st = edge_merge_store();
    merge(
        &mut st,
        "_MERGE (u:User {id:'u1'})-[m:MEMBER {since: 1}]->(t:Team {id:'t1'}) \
             _ON_CREATE SET m.role = 'admin'",
    )
    .unwrap();
    assert_eq!(st.edge_count(), 1);
    assert!(matches!(st.edge_prop(0, "since"), Value::Num(x) if x == 1.0));
    assert!(matches!(st.edge_prop(0, "role"), Value::Str(x) if &*x == "admin"));
}

/// Idempotent + default clobber of edge props; `_ON_CREATE` does NOT re-fire.
#[test]
fn merge_edge_is_idempotent_and_clobbers_props() {
    let mut st = edge_merge_store();
    merge(
        &mut st,
        "_MERGE (u:User {id:'u1'})-[m:MEMBER {since: 1}]->(t:Team {id:'t1'}) \
             _ON_CREATE SET m.role = 'admin'",
    )
    .unwrap();
    merge(
        &mut st,
        "_MERGE (u:User {id:'u1'})-[m:MEMBER {since: 2}]->(t:Team {id:'t1'})",
    )
    .unwrap();
    assert_eq!(st.edge_count(), 1); // no duplicate edge
    assert!(matches!(st.edge_prop(0, "since"), Value::Num(x) if x == 2.0)); // clobbered
    assert!(matches!(st.edge_prop(0, "role"), Value::Str(x) if &*x == "admin"));
    // on_create kept
}

/// `_ON_UPDATE SET … WHERE p` gates the edge update; a false gate is a no-op.
#[test]
fn merge_edge_on_update_where_gate() {
    let mut st = edge_merge_store();
    merge(
        &mut st,
        "_MERGE (u:User {id:'u1'})-[m:MEMBER {v: 1}]->(t:Team {id:'t1'})",
    )
    .unwrap();
    // Gate true → applies.
    merge(
        &mut st,
        "_MERGE (u:User {id:'u1'})-[m:MEMBER]->(t:Team {id:'t1'}) \
             _ON_UPDATE SET m.v = 9 WHERE m.v < 9",
    )
    .unwrap();
    assert!(matches!(st.edge_prop(0, "v"), Value::Num(x) if x == 9.0));
    // Gate false → no-op.
    merge(
        &mut st,
        "_MERGE (u:User {id:'u1'})-[m:MEMBER]->(t:Team {id:'t1'}) \
             _ON_UPDATE SET m.v = 2 WHERE m.v < 2",
    )
    .unwrap();
    assert!(matches!(st.edge_prop(0, "v"), Value::Num(x) if x == 9.0));
}

/// A missing endpoint (its key matches no vertex) is an error, not a silent
/// create — and it leaves no edge behind.
#[test]
fn merge_edge_missing_endpoint_errors() {
    let mut st = edge_merge_store();
    assert!(merge(
        &mut st,
        "_MERGE (u:User {id:'u1'})-[m:MEMBER]->(t:Team {id:'nope'})"
    )
    .is_err());
    assert_eq!(st.edge_count(), 0);
}

// --- part 7: relationship variables & edge properties (B5c) ---

/// INSERT writes inline edge properties; a bound relationship variable reads
/// them back (`r.weight`) alongside the landed node (`b.name`).
#[test]
fn insert_edge_props_then_read_via_rel_var() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    execute(
        &super::parse("INSERT (a:P {name: 'a'})-[:R {weight: 0.5}]->(b:P {name: 'b'})").unwrap(),
        &mut st,
    )
    .unwrap();
    let out = run(
        &super::parse("MATCH (a:P)-[r:R]->(b) RETURN r.weight AS w, b.name AS who").unwrap(),
        &st,
    );
    assert_eq!(out.rows.len(), 1);
    assert_eq!(num(&col(&out, 0, "w")), 0.5);
    assert!(matches!(col(&out, 0, "who"), Value::Str(x) if &*x == "b"));
}

/// An edge property present on SOME edges must not use the all-present raw
/// `Col::Num` fast path — the reader falls back to the null-carrying column, so
/// the missing cell reads NULL (and is dropped by a numeric filter).
#[test]
fn edge_property_partly_present_reads_null() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    execute(
        &super::parse(
            "INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'}), \
                 (a)-[:R]->(c:P {name: 'c'})",
        )
        .unwrap(),
        &mut st,
    )
    .unwrap();
    // Projection: the w-less edge reads NULL, not a panic or a stale 0.
    let out = run(
        &super::parse("MATCH (a:P)-[r:R]->(b) RETURN b.name AS who, r.w AS w").unwrap(),
        &st,
    );
    let mut got: Vec<(String, Value)> = out
        .rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Str(s), w) => (s.to_string(), w.clone()),
            _ => panic!(),
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(got[0].0, "b");
    assert_eq!(num(&got[0].1), 0.5);
    assert_eq!(got[1].0, "c");
    assert!(got[1].1.is_null());
    // Filter: `r.w > 0.4` keeps only the present-and-matching edge.
    let out = run(
        &super::parse("MATCH (a:P)-[r:R]->(b) WHERE r.w > 0.4 RETURN count(*) AS c").unwrap(),
        &st,
    );
    assert_eq!(num(&col(&out, 0, "c")), 1.0);
}

/// WHERE on an edge property filters edges: only the 0.5 edge passes `> 0.4`.
#[test]
fn where_on_edge_property() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    execute(
        &super::parse(
            "INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'}), \
                 (a)-[:R {w: 0.2}]->(c:P {name: 'c'})",
        )
        .unwrap(),
        &mut st,
    )
    .unwrap();
    let out = run(
        &super::parse("MATCH (a:P)-[r:R]->(b) WHERE r.w > 0.4 RETURN b.name AS who").unwrap(),
        &st,
    );
    let got: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(got, vec!["b"]);
}

/// SET on a bound relationship writes an EDGE property.
#[test]
fn set_edge_property_via_rel_var() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    execute(
        &super::parse("INSERT (a:P {name: 'a'})-[:R {w: 1}]->(b:P {name: 'b'})").unwrap(),
        &mut st,
    )
    .unwrap();
    execute(
        &super::parse("MATCH (a:P)-[r:R]->(b) SET r.w = 9").unwrap(),
        &mut st,
    )
    .unwrap();
    let out = run(
        &super::parse("MATCH (a:P)-[r:R]->(b) RETURN r.w AS w").unwrap(),
        &st,
    );
    assert_eq!(num(&col(&out, 0, "w")), 9.0);
}

/// An inline edge property in a MATCH pattern is a match filter on the edge.
#[test]
fn inline_edge_prop_is_a_match_filter() {
    use crate::exec::execute;
    let mut st = Builder::default().build();
    execute(
        &super::parse(
            "INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'}), \
                 (a)-[:R {w: 0.2}]->(c:P {name: 'c'})",
        )
        .unwrap(),
        &mut st,
    )
    .unwrap();
    let out = run(
        &super::parse("MATCH (a:P)-[r:R {w: 0.5}]->(b) RETURN b.name AS who").unwrap(),
        &st,
    );
    let got: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(got, vec!["b"]);
}

/// A bound relationship read lowers to `expand_edge` — cross-check vs the hand
/// plan (edge at slot 1, node at slot 2).
#[test]
fn rel_var_read_matches_hand_plan() {
    use crate::exec::execute;
    use crate::ir::{Expr, Plan};
    let mut st = Builder::default().build();
    execute(
        &super::parse("INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'})").unwrap(),
        &mut st,
    )
    .unwrap();
    let hand = Plan::Scan {
        label: Some("P".into()),
    }
    .expand_edge(0, crate::ir::Dir::Out, &["R".to_string()])
    .project(vec![(
        "w".into(),
        Expr::Prop {
            slot: 1,
            key: "w".into(),
        },
    )]);
    assert_same("MATCH (a:P)-[r:R]->(b) RETURN r.w AS w", &hand, &st);
}

/// A relationship variable on a variable-length pattern is rejected (deferred).
#[test]
fn rel_var_on_varlength_errors() {
    assert!(super::parse("MATCH (a:P)-[r:R]->{1,2}(b) RETURN r.w AS w").is_err());
}

/// Parsed `_MERGE` matches the hand-built `Plan::Merge` (create + on_update):
/// run both onto fresh constrained stores, confirm identical resulting props.
#[test]
fn merge_parse_matches_hand_plan() {
    use crate::exec::execute;
    use crate::ir::{Expr, MergeUpdate, Plan};
    let hand = Plan::Merge {
        label: "User".into(),
        props: vec![("email".into(), s("a")), ("name".into(), s("A"))],
        on_create: vec![("created".into(), Expr::Lit(Value::Bool(true)))],
        on_update: MergeUpdate::Set {
            assigns: vec![("seen".into(), Expr::Lit(n(1.0)))],
            filter: None,
        },
    };
    let query = "_MERGE (u:User {email: 'a', name: 'A'}) _ON_CREATE SET u.created = true \
                     _ON_UPDATE SET u.seen = 1";
    let mut st_p = user_store();
    let mut st_h = user_store();
    execute(&super::parse(query).unwrap(), &mut st_p).unwrap();
    execute(&hand, &mut st_h).unwrap();
    let probe = "MATCH (u:User) RETURN u.email AS e, u.name AS nm, u.created AS c";
    let pp = super::parse(probe).unwrap();
    assert_eq!(bag(&run(&pp, &st_p)), bag(&run(&pp, &st_h)));
}

/// `INSERT (n:…) RETURN n.…` binds the created node into scope so the trailing
/// projection reads it — the engine's first write-then-return path. The
/// returned row equals reading the same node back from the mutated store.
#[test]
fn insert_return_binds_created_node() {
    use crate::exec::execute;
    let mut st = social();
    let out = execute(
        &super::parse("INSERT (n:Person {name: 'z'}) RETURN n.name").unwrap(),
        &mut st,
    )
    .unwrap();
    // Exactly one projected row for the one created node.
    assert_eq!(out.rows.len(), 1);
    // …and it matches reading that node back (proving the bind, not a constant).
    let probe = super::parse("MATCH (n:Person {name: 'z'}) RETURN n.name").unwrap();
    assert_eq!(bag(&out), bag(&run(&probe, &st)));
}

/// The projection may read several properties of the created node.
#[test]
fn insert_return_projects_multiple_props() {
    use crate::exec::execute;
    let mut st = social();
    let out = execute(
        &super::parse("INSERT (n:Person {name: 'newbie', age: 99}) RETURN n.name, n.age").unwrap(),
        &mut st,
    )
    .unwrap();
    assert_eq!(out.rows.len(), 1);
    let probe = super::parse("MATCH (n:Person {name: 'newbie'}) RETURN n.name, n.age").unwrap();
    assert_eq!(bag(&out), bag(&run(&probe, &st)));
}

/// `&`-separated labels create a multi-labelled node (`n:Person&Admin`): the
/// created node answers a MATCH on EITHER label.
#[test]
fn insert_return_multi_label_ampersand() {
    use crate::exec::execute;
    let mut st = social();
    let out = execute(
        &super::parse("INSERT (n:Person&Admin {name: 'root'}) RETURN n.name").unwrap(),
        &mut st,
    )
    .unwrap();
    assert_eq!(out.rows.len(), 1);
    // The node carries BOTH labels — reachable via Admin and via Person.
    let by_admin = super::parse("MATCH (n:Admin) RETURN n.name").unwrap();
    assert_eq!(bag(&out), bag(&run(&by_admin, &st)));
    let by_person = super::parse("MATCH (n:Person {name: 'root'}) RETURN n.name").unwrap();
    assert_eq!(bag(&run(&by_person, &st)), bag(&run(&by_admin, &st)));
}
