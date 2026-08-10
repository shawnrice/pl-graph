//! J1 — cross-engine conformance.
//!
//! The parity plan's final gate: prove that `lenke-engine` (the from-scratch
//! engine under test) and `lenke-core` (the reference engine) return the SAME
//! answer for a matched set of GQL shapes. `lenke-core` is a dev-only dependency
//! (see Cargo.toml) so this runs only under `cargo test`.
//!
//! The two engines use DIFFERENT NDJSON dialects on the wire — engine lines are
//! `{"id":N,"labels":[...],"props":{...}}` / `{"from":N,"to":N,"type":T,...}`,
//! core lines are `{"type":"node","id":"N","labels":[...],"properties":{...}}` —
//! so the fixture is defined ONCE as Rust data (`NODES`/`EDGES`) and serialized
//! into each dialect. That keeps a single source of truth for the graph while
//! feeding each engine the bytes it expects.
//!
//! Comparison is by VALUE, not by column name or node identity: every shape
//! RETURNs scalars (names, ages, cities, counts, sums) so the two engines'
//! independent dense-id assignments never enter the comparison. Unordered shapes
//! compare as multisets (sorted); `ORDER BY` shapes compare in order, and every
//! such shape orders on a UNIQUE key so there is no tie whose resolution the two
//! engines could legitimately disagree on.
//!
//! Numbers compare through `num_key`, which is exact for integers and rounds
//! non-integers to 1e-9 — an aggregate (`avg`) can differ in the last ULP between
//! two independent summation orders, and cross-engine bit-identity of floats is
//! NOT a parity claim (byte-identity is a lenke-core-vs-TS invariant, not a
//! lenke-engine one). All fixture aggregates here are exact anyway.

use lenke_core::gql::eval::Params as CoreParams;
// The RESULT value type — `RowSet` yields `graph::Value` (a distinct type from
// the internal `value::Value`): Null/Bool/Num/Str/Temporal/List/Map, where a
// returned node/edge serializes to a Map. Our scalar shapes only ever hit the
// first five.
use lenke_core::graph::Value as CoreVal;
use lenke_engine::value::Value as EngVal;

// ── the fixture, defined once ────────────────────────────────────────────────

/// (id, label, name, age, city)
const NODES: &[(u32, &str, &str, f64, &str)] = &[
    (0, "Person", "alice", 30.0, "NYC"),
    (1, "Person", "bob", 25.0, "NYC"),
    (2, "Person", "carol", 45.0, "LA"),
    (3, "Person", "dave", 35.0, "LA"),
    (4, "Person", "eve", 28.0, "SF"),
];

/// (from, to, type)
const EDGES: &[(u32, u32, &str)] = &[
    (0, 1, "KNOWS"),
    (0, 2, "KNOWS"),
    (1, 3, "KNOWS"),
    (2, 4, "KNOWS"),
    (3, 0, "KNOWS"),
    (0, 4, "LIKES"),
];

fn engine_ndjson() -> String {
    let mut s = String::new();
    for (id, label, name, age, city) in NODES {
        s.push_str(&format!(
            r#"{{"id":{id},"labels":["{label}"],"props":{{"name":"{name}","age":{age},"city":"{city}"}}}}"#
        ));
        s.push('\n');
    }
    for (from, to, ty) in EDGES {
        s.push_str(&format!(
            r#"{{"from":{from},"to":{to},"type":"{ty}","props":{{}}}}"#
        ));
        s.push('\n');
    }
    s
}

fn core_ndjson() -> String {
    let mut s = String::new();
    for (id, label, name, age, city) in NODES {
        s.push_str(&format!(
            r#"{{"type":"node","id":"{id}","labels":["{label}"],"properties":{{"name":"{name}","age":{age},"city":"{city}"}}}}"#
        ));
        s.push('\n');
    }
    for (i, (from, to, ty)) in EDGES.iter().enumerate() {
        s.push_str(&format!(
            r#"{{"type":"edge","id":"e{i}","labels":["{ty}"],"from":"{from}","to":"{to}","properties":{{}}}}"#
        ));
        s.push('\n');
    }
    s
}

// ── a value form both engines' cells map into ────────────────────────────────

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Cmp {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Temporal(String),
    /// Anything a scalar shape should never produce (node/edge/list/record/map).
    /// Kept comparable so a stray one surfaces as an inequality, not a panic.
    Other(String),
}

/// Integers compare exactly; non-integers round to 1e-9 (see the module note).
fn num_key(n: f64) -> String {
    if n.is_finite() && n == n.trunc() {
        format!("i{}", n as i64)
    } else if n.is_nan() {
        "nan".to_string()
    } else {
        format!("f{n:.9}")
    }
}

fn norm_eng(v: &EngVal) -> Cmp {
    match v {
        EngVal::Null => Cmp::Null,
        EngVal::Bool(b) => Cmp::Bool(*b),
        EngVal::Num(n) => Cmp::Num(num_key(*n)),
        EngVal::Str(s) => Cmp::Str(s.to_string()),
        EngVal::Temporal(t) => Cmp::Temporal(t.format()),
        other => Cmp::Other(format!("{other:?}")),
    }
}

fn norm_core(v: &CoreVal) -> Cmp {
    match v {
        CoreVal::Null => Cmp::Null,
        CoreVal::Bool(b) => Cmp::Bool(*b),
        CoreVal::Num(n) => Cmp::Num(num_key(*n)),
        CoreVal::Str(s) => Cmp::Str(s.to_string()),
        CoreVal::Temporal(t) => Cmp::Temporal(t.format()),
        other => Cmp::Other(format!("{other:?}")),
    }
}

// ── run a shape through each engine ──────────────────────────────────────────

fn run_engine(store: &lenke_engine::store::Store, q: &str) -> Vec<Vec<Cmp>> {
    let plan = lenke_engine::gql::parse(q).unwrap_or_else(|e| panic!("engine parse `{q}`: {e}"));
    let out = lenke_engine::exec::run(&plan, store);
    out.rows
        .iter()
        .map(|r| r.iter().map(norm_eng).collect())
        .collect()
}

fn run_core(graph: &mut lenke_core::graph::Graph, q: &str) -> Vec<Vec<Cmp>> {
    let prepared = lenke_core::gql::prepare(q).unwrap_or_else(|e| panic!("core parse `{q}`: {e}"));
    let rs = prepared
        .execute(graph, &CoreParams::new())
        .unwrap_or_else(|e| panic!("core exec `{q}`: {e:?}"));
    rs.rows()
        .map(|r| r.iter().map(norm_core).collect())
        .collect()
}

fn sorted(mut rows: Vec<Vec<Cmp>>) -> Vec<Vec<Cmp>> {
    rows.sort();
    rows
}

/// Assert the two engines agree on `q` as a multiset (row order unspecified).
fn agree_unordered(
    store: &lenke_engine::store::Store,
    graph: &mut lenke_core::graph::Graph,
    q: &str,
) {
    let e = sorted(run_engine(store, q));
    let c = sorted(run_core(graph, q));
    assert_eq!(
        e, c,
        "multiset mismatch for `{q}`\n engine={e:?}\n   core={c:?}"
    );
}

/// Assert the two engines agree on `q` as an ordered sequence (`ORDER BY`).
fn agree_ordered(
    store: &lenke_engine::store::Store,
    graph: &mut lenke_core::graph::Graph,
    q: &str,
) {
    let e = run_engine(store, q);
    let c = run_core(graph, q);
    assert_eq!(
        e, c,
        "ordered mismatch for `{q}`\n engine={e:?}\n   core={c:?}"
    );
}

// ── the conformance shapes ───────────────────────────────────────────────────

#[test]
fn cross_engine_conformance() {
    let store = lenke_engine::ndjson::from_ndjson(&engine_ndjson()).expect("engine load");
    let mut graph = lenke_core::ndjson::decode(&core_ndjson()).expect("core load");

    // Sanity: the fixture loaded with the shape we expect, in BOTH engines, so a
    // later agreement on `[]` can't be two empty results agreeing on nothing.
    assert_eq!(
        run_engine(&store, "MATCH (n:Person) RETURN count(*) AS c"),
        vec![vec![Cmp::Num("i5".into())]]
    );
    assert_eq!(
        run_core(&mut graph, "MATCH (n:Person) RETURN count(*) AS c"),
        vec![vec![Cmp::Num("i5".into())]]
    );

    let unordered = [
        // projection
        "MATCH (n:Person) RETURN n.name AS name",
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age",
        // filters — the equivalent-spellings concern lives here
        "MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name",
        "MATCH (n:Person) WHERE n.age >= 30 AND n.age <= 40 RETURN n.name AS name",
        "MATCH (n:Person) WHERE n.name = 'alice' RETURN n.age AS age",
        "MATCH (n:Person) WHERE n.city = 'NYC' RETURN n.name AS name",
        "MATCH (n:Person) WHERE n.age < 30 OR n.age > 40 RETURN n.name AS name",
        // aggregates
        "MATCH (n:Person) RETURN count(*) AS c",
        "MATCH (n:Person) RETURN sum(n.age) AS s",
        "MATCH (n:Person) RETURN avg(n.age) AS a",
        "MATCH (n:Person) RETURN min(n.age) AS mn, max(n.age) AS mx",
        // grouped aggregate
        "MATCH (n:Person) RETURN n.city AS city, count(*) AS c",
        "MATCH (n:Person) RETURN n.city AS city, sum(n.age) AS s",
        // traversal
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS a, b.name AS b",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS b",
        "MATCH (a:Person)-[:LIKES]->(b) RETURN a.name AS a, b.name AS b",
        "MATCH ()-[:KNOWS]->() RETURN count(*) AS c",
        // two-hop
        "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c) RETURN a.name AS a, c.name AS c",
        // distinct
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN DISTINCT b.name AS b",
    ];
    for q in unordered {
        agree_unordered(&store, &mut graph, q);
    }

    let ordered = [
        // every ORDER BY key below is unique in the fixture (ages and names are
        // all distinct), so there is no tie whose order the engines could differ
        // on. The key is ALSO projected: lenke-engine scopes ORDER BY to output
        // columns (by alias), while lenke-core also accepts a non-projected
        // expression — a documented divergence (see engine-parity-plan.md J1), so
        // the matched shapes order by a projected alias, which BOTH accept.
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age",
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age DESC",
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY name",
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age DESC LIMIT 2",
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age LIMIT 3",
        // grouped, then ordered on the (unique) group key
        "MATCH (n:Person) RETURN n.city AS city, count(*) AS c ORDER BY city",
    ];
    for q in ordered {
        agree_ordered(&store, &mut graph, q);
    }
}
