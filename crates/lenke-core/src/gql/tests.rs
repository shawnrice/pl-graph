//! Conformance tests for the GQL engine, mirroring the TS `gql.test.ts` /
//! `tck.test.ts` spec over the TinkerPop "Modern" fixture. Covers the read
//! surface, edge properties, and write clauses (the graph is mutable).

use std::sync::Arc;

use super::eval::Params;
use super::parse;
use crate::graph::{Graph, Value};
use crate::ndjson;

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gql()
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

/// Build a graph from NDJSON lines (each already a JSON object literal).
fn graph_of(lines: &[&str]) -> Graph {
    ndjson::decode(&lines.join("\n")).unwrap()
}

/// Triangle a→b→c→a plus the tail a→d. Shared by the path-mode / bare-path tests.
fn triangle_tail() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e3","from":"c","to":"a","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e4","from":"a","to":"d","labels":["R"],"properties":{}}"#,
    ])
}

/// Two nodes joined by an edge each way: a→b (e0) and b→a (e1). The minimal
/// fixture for observing WALK edge-reuse vs the TRAIL/SIMPLE/ACYCLIC restrictors.
fn bidir_pair() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e1","from":"b","to":"a","labels":["R"],"properties":{}}"#,
    ])
}

/// Run a query and return column 0 of every row, sorted by debug repr — for
/// set-comparing unordered endpoint results deterministically.
fn sorted_col0(g: &mut Graph, query: &str) -> Vec<Value> {
    let mut v: Vec<Value> = rows(g, query).into_iter().map(|r| r[0].clone()).collect();
    v.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    v
}

/// Sorted endpoint ids from `MATCH <mode> (a{id:'a'})-[:R]->{lo,hi}(x) RETURN x.id`.
/// A convenience for asserting exactly which endpoints a path mode admits.
fn mode_ends(g: &mut Graph, mode: &str, lo: u32, hi: u32) -> Vec<Value> {
    let mut r: Vec<Value> = rows(
        g,
        &format!("MATCH {mode} (a:N {{id:'a'}})-[:R]->{{{lo},{hi}}}(x) RETURN x.id AS id"),
    )
    .into_iter()
    .map(|row| row[0].clone())
    .collect();
    r.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    r
}

/// Run a query expected to fail at EXECUTION time (it must still parse), and
/// report whether it did. Unlike [`q`], this does not panic on the error.
fn exec_err(g: &mut Graph, query: &str) -> bool {
    parse(query)
        .unwrap_or_else(|e| panic!("parse error for `{query}`: {e}"))
        .execute(g, &Params::new())
        .is_err()
}

/// `EXISTS { (a)-[:T]->+/*(b …) }` reachability: the BFS fast path must agree with
/// the enumerated answer. Cross-checked against a *bounded* `->{1,big}` EXISTS
/// (which the fast path does NOT intercept, so it enumerates) — and covers a
/// reachable target, an UNREACHABLE one (the fault the fast path fixes), an endpoint
/// WHERE, and `->*` zero-length self-inclusion.
#[test]
fn exists_reachability_matches_enumeration() {
    // s → x → y → c (a chain); z is isolated (unreachable).
    let lines = [
        r#"{"type":"node","id":"s","labels":["N"],"properties":{"name":"s"}}"#,
        r#"{"type":"node","id":"x","labels":["N"],"properties":{"name":"x"}}"#,
        r#"{"type":"node","id":"y","labels":["N"],"properties":{"name":"y"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"name":"c"}}"#,
        r#"{"type":"node","id":"z","labels":["N"],"properties":{"name":"z"}}"#,
        r#"{"type":"edge","from":"s","to":"x","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"x","to":"y","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"y","to":"c","labels":["R"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let ex = |g: &mut Graph, query: &str| -> bool {
        match q(g, query).1[0][0] {
            Value::Bool(b) => b,
            ref o => panic!("expected bool, got {o:?}"),
        }
    };
    let pairs = [
        // reachable target c (3 hops)
        (
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->+(b:N {name:'c'}) } AS r",
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->{1,20}(b:N {name:'c'}) } AS r",
            true,
        ),
        // UNREACHABLE target z — the enumeration explores every trail then says false;
        // on a bigger graph that faults, which the BFS fast path avoids.
        (
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->+(b:N {name:'z'}) } AS r",
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->{1,20}(b:N {name:'z'}) } AS r",
            false,
        ),
        // endpoint WHERE
        (
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->+(b) WHERE b.name = 'y' } AS r",
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->{1,20}(b) WHERE b.name = 'y' } AS r",
            true,
        ),
        // ->* admits the 0-hop start: s reaches itself
        (
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->*(b:N {name:'s'}) } AS r",
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->{0,20}(b:N {name:'s'}) } AS r",
            true,
        ),
        // ->+ from s to s needs a cycle — there is none.
        (
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->+(b:N {name:'s'}) } AS r",
            "MATCH (a:N {name:'s'}) RETURN EXISTS { (a)-[:R]->{1,20}(b:N {name:'s'}) } AS r",
            false,
        ),
    ];
    for (bfs, enumerated, want) in pairs {
        let b = ex(&mut g, bfs);
        let e = ex(&mut g, enumerated);
        assert_eq!(b, e, "BFS != enumerated:\n  {bfs}");
        assert_eq!(b, want, "wrong reachability for: {bfs}");
    }
}

/// ISO `percentile_cont` / `percentile_disc` ordered-set aggregates over known
/// values: cont interpolates between ranks, disc returns an actual element.
#[test]
fn percentile_aggregates() {
    let odd = ndjson::decode(
        &(1..=5)
            .map(|i| {
                format!(r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{{"v":{i}}}}}"#)
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let even = ndjson::decode(
        &(1..=4)
            .map(|i| {
                format!(r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{{"v":{i}}}}}"#)
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let num = |g: &mut Graph, query: &str| -> f64 {
        match q(g, query).1[0][0] {
            Value::Num(v) => v,
            ref o => panic!("expected number, got {o:?}"),
        }
    };
    let mut o = odd;
    // [1,2,3,4,5]: median = 3 (cont & disc). cont(0.25)=2 (rn=1.0, exact rank).
    assert_eq!(
        num(&mut o, "MATCH (n:V) RETURN percentile_cont(n.v, 0.5) AS x"),
        3.0
    );
    assert_eq!(
        num(&mut o, "MATCH (n:V) RETURN percentile_disc(n.v, 0.5) AS x"),
        3.0
    );
    assert_eq!(
        num(&mut o, "MATCH (n:V) RETURN percentile_cont(n.v, 0.25) AS x"),
        2.0
    );
    assert_eq!(
        num(&mut o, "MATCH (n:V) RETURN percentile_cont(n.v, 0.0) AS x"),
        1.0
    );
    assert_eq!(
        num(&mut o, "MATCH (n:V) RETURN percentile_cont(n.v, 1.0) AS x"),
        5.0
    );

    let mut e = even;
    // [1,2,3,4]: cont(0.5) interpolates to 2.5; disc(0.5) is the element 2.
    assert_eq!(
        num(&mut e, "MATCH (n:V) RETURN percentile_cont(n.v, 0.5) AS x"),
        2.5
    );
    assert_eq!(
        num(&mut e, "MATCH (n:V) RETURN percentile_disc(n.v, 0.5) AS x"),
        2.0
    );
    // Grouped: verify it folds per group too (all one group here → same answer).
    assert_eq!(
        num(&mut e, "MATCH (n:V) RETURN percentile_cont(n.v, 0.75) AS x"),
        3.25, // rn = 0.75*3 = 2.25 → 3 + 0.25*(4-3)
    );
}

/// The `COUNT { … }` degree fast path (single plain correlated segment) must equal
/// the enumerated count. Cross-checked by adding an always-true inner `WHERE`, which
/// bails the fast path onto the recursive matcher — so fast == enumerated. Covers a
/// labelled endpoint, the `In` direction, parallel edges and a self-loop.
#[test]
fn count_subquery_degree_matches_enumeration() {
    let lines = [
        r#"{"type":"node","id":"n0","labels":["Node"],"properties":{"name":"n0"}}"#,
        r#"{"type":"node","id":"n1","labels":["Node","Target"],"properties":{"name":"n1"}}"#,
        r#"{"type":"node","id":"n2","labels":["Node"],"properties":{"name":"n2"}}"#,
        r#"{"type":"edge","from":"n0","to":"n1","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"n0","to":"n1","labels":["R"],"properties":{}}"#, // parallel
        r#"{"type":"edge","from":"n0","to":"n2","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"n0","to":"n0","labels":["R"],"properties":{}}"#, // self-loop
        r#"{"type":"edge","from":"n2","to":"n1","labels":["R"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let val = |g: &mut Graph, query: &str| -> f64 {
        match q(g, query).1[0][0] {
            Value::Num(v) => v,
            ref o => panic!("expected number, got {o:?}"),
        }
    };
    // (fast COUNT{}, enumerated COUNT{ … WHERE true }) — must be equal, and match the
    // hand-computed degree from n0. n0 out-R: to n1, n1, n2, n0 (self) = 4.
    let pairs = [
        (
            "MATCH (n:Node) WHERE n.name = 'n0' RETURN COUNT { (n)-[:R]->() } AS c",
            "MATCH (n:Node) WHERE n.name = 'n0' RETURN COUNT { (n)-[:R]->(m) WHERE true } AS c",
            4.0,
        ),
        (
            // labelled endpoint: n0 → Target(n1) twice = 2.
            "MATCH (n:Node) WHERE n.name = 'n0' RETURN COUNT { (n)-[:R]->(:Target) } AS c",
            "MATCH (n:Node) WHERE n.name = 'n0' RETURN COUNT { (n)-[:R]->(m:Target) WHERE true } AS c",
            2.0,
        ),
        (
            // In direction: in-R of n1 = from n0 (×2) and n2 = 3.
            "MATCH (n:Node) WHERE n.name = 'n1' RETURN COUNT { (n)<-[:R]-() } AS c",
            "MATCH (n:Node) WHERE n.name = 'n1' RETURN COUNT { (n)<-[:R]-(m) WHERE true } AS c",
            3.0,
        ),
        (
            // Reverse degree — the correlated node is the ENDPOINT (like a popularity
            // shape `COUNT { (:User)-[:PURCHASED]->(y) }`). in-R of n1 from a free start = 3.
            "MATCH (n:Node) WHERE n.name = 'n1' RETURN COUNT { (m)-[:R]->(n) } AS c",
            "MATCH (n:Node) WHERE n.name = 'n1' RETURN COUNT { (m)-[:R]->(n) WHERE true } AS c",
            3.0,
        ),
        (
            // Reverse degree with a label on the free start: in-R of n1 whose source
            // is a Target. n1's in-neighbours n0/n2 are not Target → 0.
            "MATCH (n:Node) WHERE n.name = 'n1' RETURN COUNT { (m:Target)-[:R]->(n) } AS c",
            "MATCH (n:Node) WHERE n.name = 'n1' RETURN COUNT { (m:Target)-[:R]->(n) WHERE true } AS c",
            0.0,
        ),
        (
            // Reverse out-degree: `(m)<-[:R]-(n)` with n bound = n's out-edges. n0 → 4.
            "MATCH (n:Node) WHERE n.name = 'n0' RETURN COUNT { (m)<-[:R]-(n) } AS c",
            "MATCH (n:Node) WHERE n.name = 'n0' RETURN COUNT { (m)<-[:R]-(n) WHERE true } AS c",
            4.0,
        ),
    ];
    for (fast, enumerated, want) in pairs {
        let f = val(&mut g, fast);
        let e = val(&mut g, enumerated);
        assert_eq!(f, e, "fast != enumerated:\n  {fast}");
        assert_eq!(f, want, "wrong degree for: {fast}");
    }
}

/// The reverse semi-join fast path (`try_count_semi_join`) must equal the enumerated
/// answer: `EXISTS` count == `count(DISTINCT a)` of the same join, and `NOT EXISTS`
/// == all rows minus that. Both edge directions; some `a`s have a qualifying edge,
/// some don't, and one `a` reaches an Admin two ways (dedup must hold).
#[test]
fn exists_count_matches_reverse_semi_join() {
    let lines = [
        r#"{"type":"node","id":"p1","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"p2","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"p3","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"m1","labels":["Person","Admin"],"properties":{}}"#,
        r#"{"type":"node","id":"m2","labels":["Person","Admin"],"properties":{}}"#,
        // p1 → m1 and p1 → m2 (two ways to an Admin — must count p1 once).
        r#"{"type":"edge","from":"p1","to":"m1","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"p1","to":"m2","labels":["KNOWS"],"properties":{}}"#,
        // p2 → m1 (one Admin). p3 → p2 (no Admin). m1 → m2 (Admin → Admin).
        r#"{"type":"edge","from":"p2","to":"m1","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"p3","to":"p2","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"m1","to":"m2","labels":["KNOWS"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let c = |g: &mut Graph, query: &str| -> f64 {
        match q(g, query).1[0][0] {
            Value::Num(v) => v,
            ref o => panic!("expected count, got {o:?}"),
        }
    };
    // EXISTS (a)-[:KNOWS]->(:Admin): predecessors of Admins = {p1, p2, m1}. = 3.
    let exists = c(
        &mut g,
        "MATCH (a:Person) WHERE EXISTS { (a)-[:KNOWS]->(:Admin) } RETURN count(*) AS c",
    );
    let distinct = c(
        &mut g,
        "MATCH (a:Person)-[:KNOWS]->(b:Admin) RETURN count(DISTINCT a) AS c",
    );
    assert_eq!(exists, distinct, "EXISTS count != count(DISTINCT a)");
    assert_eq!(exists, 3.0);
    // NOT EXISTS = all 5 Person minus the 3 that reach an Admin.
    let not_exists = c(
        &mut g,
        "MATCH (a:Person) WHERE NOT EXISTS { (a)-[:KNOWS]->(:Admin) } RETURN count(*) AS c",
    );
    assert_eq!(not_exists, 2.0);
    // Reverse direction: (a)<-[:KNOWS]-(:Admin) — a is an out-neighbor of an Admin.
    // Admin out-neighbors: m1→m2. So {m2}. = 1.
    let exists_rev = c(
        &mut g,
        "MATCH (a:Person) WHERE EXISTS { (a)<-[:KNOWS]-(:Admin) } RETURN count(*) AS c",
    );
    let distinct_rev = c(
        &mut g,
        "MATCH (a:Person)<-[:KNOWS]-(b:Admin) RETURN count(DISTINCT a) AS c",
    );
    assert_eq!(exists_rev, distinct_rev);
    assert_eq!(exists_rev, 1.0);
}

/// `count(DISTINCT <endpoint>)` over a traversal (`try_count_distinct_reachable`,
/// frontier marking) must equal the size of the enumerated reachable set —
/// including convergent paths (two starts reaching one endpoint dedup to one) and
/// the endpoint label filter.
#[test]
fn count_distinct_reachable_matches_enumeration() {
    // p0,p1,p2 : Person ; t3,t4,t5 : Target.
    let lines = [
        r#"{"type":"node","id":"p0","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"p1","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"p2","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"t3","labels":["Target"],"properties":{}}"#,
        r#"{"type":"node","id":"t4","labels":["Target"],"properties":{}}"#,
        r#"{"type":"node","id":"t5","labels":["Target"],"properties":{}}"#,
        r#"{"type":"edge","from":"p0","to":"t3","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"p1","to":"t3","labels":["KNOWS"],"properties":{}}"#, // p0,p1 → t3
        r#"{"type":"edge","from":"p1","to":"t4","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"p2","to":"t4","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"t3","to":"t5","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"t4","to":"t5","labels":["KNOWS"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let edges: [(u32, u32); 6] = [(0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5)];
    let reachable = |starts: &[u32], depth: usize| -> usize {
        let mut cur: std::collections::HashSet<u32> = starts.iter().copied().collect();
        for _ in 0..depth {
            let mut next = std::collections::HashSet::new();
            for &(s, d) in &edges {
                if cur.contains(&s) {
                    next.insert(d);
                }
            }
            cur = next;
        }
        cur.len()
    };
    let c = |g: &mut Graph, query: &str| -> usize {
        match q(g, query).1[0][0] {
            Value::Num(v) => v as usize,
            ref o => panic!("expected count, got {o:?}"),
        }
    };
    // 1-hop distinct endpoints from Person {0,1,2} = {t3, t4} (t3 reached twice).
    assert_eq!(
        c(
            &mut g,
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(DISTINCT b) AS c"
        ),
        reachable(&[0, 1, 2], 1),
    );
    assert_eq!(reachable(&[0, 1, 2], 1), 2);
    // 2-hop distinct endpoints = out-neighbors of {t3,t4} = {t5}.
    assert_eq!(
        c(
            &mut g,
            "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c) RETURN count(DISTINCT c) AS c",
        ),
        reachable(&[0, 1, 2], 2),
    );
    assert_eq!(reachable(&[0, 1, 2], 2), 1);
    // Endpoint label filter: 1-hop distinct b that are Target = {t3, t4}.
    assert_eq!(
        c(
            &mut g,
            "MATCH (a:Person)-[:KNOWS]->(b:Target) RETURN count(DISTINCT b) AS c",
        ),
        2,
    );
}

/// Unbounded var-length with DISTINCT (`try_reachable_distinct`, BFS) must equal the
/// enumerated reachable set on a graph small enough that trail enumeration completes.
/// Covers `->+` vs `->*` (seed inclusion), a cycle (seed reachable from itself), and
/// the endpoint label filter.
#[test]
fn reachable_distinct_matches_enumeration() {
    // s0 → a1 → a2 → a1 (cycle a1↔a2 via a2→a1); a2 → t3 (Target). s0 is Node.
    let lines = [
        r#"{"type":"node","id":"s0","labels":["Node"],"properties":{"name":"s0"}}"#,
        r#"{"type":"node","id":"a1","labels":["Node"],"properties":{"name":"a1"}}"#,
        r#"{"type":"node","id":"a2","labels":["Node"],"properties":{"name":"a2"}}"#,
        r#"{"type":"node","id":"t3","labels":["Node","Target"],"properties":{"name":"t3"}}"#,
        r#"{"type":"node","id":"z9","labels":["Node"],"properties":{"name":"z9"}}"#, // unreachable
        r#"{"type":"edge","from":"s0","to":"a1","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"a1","to":"a2","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"a2","to":"a1","labels":["R"],"properties":{}}"#, // cycle
        r#"{"type":"edge","from":"a2","to":"t3","labels":["R"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let names = |g: &mut Graph, query: &str| -> Vec<String> {
        let mut v: Vec<String> = rows(g, query)
            .iter()
            .map(|r| match &r[0] {
                Value::Str(s) => s.to_string(),
                o => panic!("expected string, got {o:?}"),
            })
            .collect();
        v.sort();
        v
    };
    // ->+ from s0: reachable via ≥1 hop = {a1, a2, t3}. s0 NOT included (no path back to it).
    assert_eq!(
        names(
            &mut g,
            "MATCH (a:Node {name: 's0'})-[:R]->+(b) RETURN DISTINCT b.name AS n"
        ),
        vec!["a1", "a2", "t3"],
    );
    // ->* also includes the 0-hop seed s0.
    assert_eq!(
        names(
            &mut g,
            "MATCH (a:Node {name: 's0'})-[:R]->*(b) RETURN DISTINCT b.name AS n"
        ),
        vec!["a1", "a2", "s0", "t3"],
    );
    // Endpoint label filter: only Target endpoints = {t3}.
    assert_eq!(
        names(
            &mut g,
            "MATCH (a:Node {name: 's0'})-[:R]->+(b:Target) RETURN DISTINCT b.name AS n",
        ),
        vec!["t3"],
    );
    // A cycle makes a1 reachable from itself via ≥1 hop: ->+ from a1 includes a1.
    assert_eq!(
        names(
            &mut g,
            "MATCH (a:Node {name: 'a1'})-[:R]->+(b) RETURN DISTINCT b.name AS n"
        ),
        vec!["a1", "a2", "t3"],
    );
    // count(DISTINCT b) over the unbounded reach from s0 = |{a1,a2,t3}| = 3.
    let c = match q(
        &mut g,
        "MATCH (a:Node {name: 's0'})-[:R]->+(b) RETURN count(DISTINCT b) AS c",
    )
    .1[0][0]
    {
        Value::Num(v) => v as i64,
        ref o => panic!("{o:?}"),
    };
    assert_eq!(c, 3);
}

#[test]
fn count_star_alias() {
    let mut g = modern();
    let (cols, r) = q(&mut g, "MATCH (n:Person) RETURN count(*) AS c");
    assert_eq!(cols, vec!["c"]);
    assert_eq!(r, vec![vec![n(4.0)]]);
}

/// The var-length `{1,2}` count fast path (`try_count_varlen_upto_2`, degree products)
/// must equal a brute-force trail enumeration, including the tricky cases: parallel
/// edges (two `a→b`), self-loops (`a→a`, `b→b`, which the degree product would
/// double-count as an invalid edge-reusing `a→a→a` without the correction), and a
/// selective start label. `a=0, b=1, c=2` by insertion order.
#[test]
fn varlen_1_2_count_matches_trail_enumeration() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["Person","VIP"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"c","labels":["Person"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"b","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"a","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"c","to":"a","labels":["KNOWS"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    // The authored edge list (src, dst), by vertex index.
    let edges: [(u32, u32); 6] = [(0, 1), (0, 1), (1, 2), (1, 1), (0, 0), (2, 0)];

    // Brute force: length-1 trails = edges with a matching source; length-2 trails
    // = ordered edge pairs (e_i ending at m, e_j starting at m) with i != j (each
    // edge used at most once — trail semantics).
    let brute = |src_ok: &dyn Fn(u32) -> bool| -> u64 {
        let mut c = edges.iter().filter(|(s, _)| src_ok(*s)).count() as u64;
        for (i, &(s, m)) in edges.iter().enumerate() {
            if !src_ok(s) {
                continue;
            }
            for (j, &(s2, _)) in edges.iter().enumerate() {
                if i != j && s2 == m {
                    c += 1;
                }
            }
        }
        c
    };
    let count_of = |g: &mut Graph, query: &str| -> u64 {
        match q(g, query).1[0][0] {
            Value::Num(v) => v as u64,
            ref other => panic!("expected numeric count, got {other:?}"),
        }
    };

    // La = none (every vertex is a valid start).
    assert_eq!(
        count_of(&mut g, "MATCH (x)-[:KNOWS]->{1,2}(y) RETURN count(*) AS c"),
        brute(&|_| true),
    );
    // La = VIP (only `a`) — exercises the src-label filter on the length-1 and
    // in-side terms and the self-loop correction gating.
    assert_eq!(
        count_of(
            &mut g,
            "MATCH (x:VIP)-[:KNOWS]->{1,2}(y) RETURN count(*) AS c"
        ),
        brute(&|v| v == 0),
    );
    // Independent hand-computed anchors.
    assert_eq!(brute(&|_| true), 17);
    assert_eq!(brute(&|v| v == 0), 9);
}

/// The SAME degree-product pass answers `{1,1}` and `{2,2}`, not only `{1,2}`.
///
/// One pass gives the length-1 count, the length-2 count and the trail
/// correction, so a quantifier wanting one of them costs what the one wanting
/// both costs. Refusing the narrower ranges left `{2,2}` on the general trail
/// machinery at 16.3ms while `{1,2}` — strictly more work — answered in 0.97ms.
///
/// Brute-forced per LENGTH here, because the interesting term is the correction:
/// `corr` removes the `a→a→a` walks that reuse a self-loop, so it belongs to the
/// length-2 count and must not be subtracted from the length-1 one. The fixture
/// has two self-loops and a pair of parallel edges for exactly that reason.
#[test]
fn varlen_fixed_lengths_match_trail_enumeration() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["Person","VIP"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["Person"],"properties":{}}"#,
        r#"{"type":"node","id":"c","labels":["Person"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"b","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"a","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","from":"c","to":"a","labels":["KNOWS"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let edges: [(u32, u32); 6] = [(0, 1), (0, 1), (1, 2), (1, 1), (0, 0), (2, 0)];

    let len1 = |src_ok: &dyn Fn(u32) -> bool| -> u64 {
        edges.iter().filter(|(s, _)| src_ok(*s)).count() as u64
    };
    // Ordered edge pairs sharing a midpoint, each edge used at most once.
    let len2 = |src_ok: &dyn Fn(u32) -> bool| -> u64 {
        let mut c = 0;

        for (i, &(s, m)) in edges.iter().enumerate() {
            if !src_ok(s) {
                continue;
            }

            for (j, &(s2, _)) in edges.iter().enumerate() {
                if i != j && s2 == m {
                    c += 1;
                }
            }
        }

        c
    };
    let count_of = |g: &mut Graph, query: &str| -> u64 {
        match q(g, query).1[0][0] {
            Value::Num(v) => v as u64,
            ref other => panic!("expected numeric count, got {other:?}"),
        }
    };

    for (src_ok, pat) in [(&|_: u32| true, "(x)"), (&|v: u32| v == 0, "(x:VIP)")]
        as [(&dyn Fn(u32) -> bool, &str); 2]
    {
        assert_eq!(
            count_of(
                &mut g,
                &format!("MATCH {pat}-[:KNOWS]->{{1,1}}(y) RETURN count(*) AS c")
            ),
            len1(src_ok),
            "`{pat}` length-1"
        );
        assert_eq!(
            count_of(
                &mut g,
                &format!("MATCH {pat}-[:KNOWS]->{{2,2}}(y) RETURN count(*) AS c")
            ),
            len2(src_ok),
            "`{pat}` length-2"
        );
        // And the two still sum to the range that was already covered.
        assert_eq!(
            count_of(
                &mut g,
                &format!("MATCH {pat}-[:KNOWS]->{{1,2}}(y) RETURN count(*) AS c")
            ),
            len1(src_ok) + len2(src_ok),
            "`{pat}` the range is the sum of its lengths"
        );
    }

    // Hand-computed anchors, so a brute force that drifts is caught too. The
    // length-2 count is 11 and NOT 13: two of the fifteen midpoint-sharing pairs
    // reuse a self-loop.
    assert_eq!((len1(&|_| true), len2(&|_| true)), (6, 11));
    assert_eq!((len1(&|v| v == 0), len2(&|v| v == 0)), (3, 6));

    // `{1,1}` is one hop, so it must agree with the unquantified spelling.
    assert_eq!(
        count_of(&mut g, "MATCH (x)-[:KNOWS]->{1,1}(y) RETURN count(*) AS c"),
        count_of(&mut g, "MATCH (x)-[:KNOWS]->(y) RETURN count(*) AS c"),
    );
}

/// A numeric `score` present on some nodes, absent on others — so the column
/// carries NaN for the absent ones. Exercises the absent→NaN columnar path that
/// the vectorized aggregate executor now handles for plain `MATCH … WHERE …
/// RETURN <aggregate>` (no intermediate WITH): a numeric predicate must treat
/// `NaN <cmp> x` as false (matching GQL null semantics), and sum/avg/count must
/// skip absent values — exactly the scalar engine's behavior.
fn mixed_presence() -> Graph {
    let lines = [
        r#"{"type":"node","id":"a","labels":["T"],"properties":{"score":1,"age":10}}"#,
        r#"{"type":"node","id":"b","labels":["T"],"properties":{"age":20}}"#,
        r#"{"type":"node","id":"c","labels":["T"],"properties":{"score":9,"age":30}}"#,
        r#"{"type":"node","id":"d","labels":["T"],"properties":{"score":5}}"#,
        r#"{"type":"node","id":"e","labels":["T"],"properties":{"score":4,"age":20}}"#,
        r#"{"type":"node","id":"f","labels":["T"],"properties":{"age":40}}"#,
    ];
    ndjson::decode(&lines.join("\n")).unwrap()
}

#[test]
fn vectorized_aggregates_over_absent_and_nan() {
    let mut g = mixed_presence();
    // count(*) — every matched row
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN count(*) AS c"),
        vec![vec![n(6.0)]]
    );
    // numeric predicate where absent score is NaN in the column: NaN > x is false,
    // so absent nodes are excluded (b, f have no score).
    assert_eq!(
        rows(&mut g, "MATCH (n:T) WHERE n.score > 5 RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) WHERE n.score >= 5 RETURN count(*) AS c"
        ),
        vec![vec![n(2.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) WHERE n.age > 15 RETURN count(*) AS c"),
        vec![vec![n(4.0)]]
    );
    // aggregates skip absent values (4 nodes have score: 1,9,5,4)
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN sum(n.score) AS s"),
        vec![vec![n(19.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN avg(n.score) AS a"),
        vec![vec![n(4.75)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN min(n.score) AS lo, max(n.score) AS hi"
        ),
        vec![vec![n(1.0), n(9.0)]],
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN count(n.score) AS c"),
        vec![vec![n(4.0)]]
    );
    // filter on one property, aggregate another: age>=20 → {b,c,e,f}; present
    // scores among them are c=9, e=4 (b, f skipped).
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) WHERE n.age >= 20 RETURN sum(n.score) AS s"
        ),
        vec![vec![n(13.0)]]
    );
}

#[test]
fn count_star_shortcut_edges() {
    let mut g = mixed_presence();
    // bare count(*) over a label takes the O(1) `vertices_with_label(l).len()`
    // shortcut — must equal the general count.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN count(*) AS c"),
        vec![vec![n(6.0)]]
    );
    // a label with no vertices → count 0 (still one row, like the general path).
    assert_eq!(
        rows(&mut g, "MATCH (n:Ghost) RETURN count(*) AS c"),
        vec![vec![n(0.0)]]
    );
    // count(*)+1 is NOT the bare shortcut (output is an expression over the agg).
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN count(*) + 1 AS c"),
        vec![vec![n(7.0)]]
    );
    // a second aggregate / a grouping key / a WHERE all keep the general path.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) WHERE n.age > 15 RETURN count(*) AS c"),
        vec![vec![n(4.0)]]
    );
}

#[test]
fn order_by_letin_over_output_column_sorts_by_that_column() {
    // Regression (refs_slot_below): the vectorized-scan fast path is gated by
    // whether an ORDER BY key reads an output column. That check once missed
    // `LetIn` (and PropertyExists / correlated subqueries), so an ORDER BY key that
    // reads an OUTPUT column *through* a LET was mis-reported as input-only — the
    // query wrongly vectorized and sorted in the INPUT scope (by node identity)
    // instead of by the projected value. TS has no such fast path, so it was a
    // silent Rust-only divergence. The key `(LET x = a IN x END)` sorts by `a`.
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS nm, n.age AS a ORDER BY (LET x = a IN x END)",
    );
    let names: Vec<String> = r
        .iter()
        .map(|row| match &row[0] {
            Value::Str(name) => name.to_string(),
            other => panic!("expected a name, got {other:?}"),
        })
        .collect();
    // ages: vadas 27, marko 29, josh 32, peter 35 → ascending by age, not by id.
    assert_eq!(names, vec!["vadas", "marko", "josh", "peter"]);
}

#[test]
fn projection_column_names_and_order() {
    let mut g = modern();
    let (cols, r) = q(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN p.name, s.name ORDER BY p.name, s.name",
    );
    assert_eq!(cols, vec!["p.name", "s.name"]);
    assert_eq!(
        r,
        vec![
            vec![s("josh"), s("lop")],
            vec![s("josh"), s("ripple")],
            vec![s("marko"), s("lop")],
            vec![s("peter"), s("lop")],
        ]
    );
}

#[test]
fn two_hop_linear_pattern() {
    // Regression: a linear two-segment pattern `(a)-[r1]->(b)-[r2]->(c)` used to
    // panic in build_scan because the per-row column copy referenced `c`'s slot
    // (bound only by the second segment) while building the first segment.
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->(b)-[:CREATED]->(c) RETURN c.name ORDER BY c.name",
    );
    // marko KNOWS josh; josh CREATED lop + ripple.
    assert_eq!(r, vec![vec![s("lop")], vec![s("ripple")]]);
}

#[test]
fn three_hop_linear_pattern() {
    // A three-segment chain exercises copying multiple already-bound columns
    // across several future-bound slots.
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->(b)-[:CREATED]->(c)<-[:CREATED]-(d) RETURN d.name ORDER BY d.name",
    );
    // marko->josh; josh created lop+ripple; lop also created-by marko,josh,peter;
    // ripple created-by josh. Distinct d over both c's, ordered.
    assert_eq!(
        r,
        vec![
            vec![s("josh")],
            vec![s("josh")],
            vec![s("marko")],
            vec![s("peter")],
        ]
    );
}

#[test]
fn incoming_edge() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (s:Software)<-[:CREATED]-(p:Person) WHERE s.name = 'ripple' RETURN p.name",
    );
    assert_eq!(r, vec![vec![s("josh")]]);
}

#[test]
fn undirected_edge() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'josh' RETURN b.name",
    );
    assert_eq!(r, vec![vec![s("marko")]]);
}

#[test]
fn var_length_plus() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->+(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("vadas")]]);
}

#[test]
fn var_length_star_includes_self() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->*(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("marko")], vec![s("vadas")]]);
}

#[test]
fn var_length_bounded() {
    let mut g = modern();
    // exactly 1 hop of KNOWS from marko → vadas, josh
    let r = rows(
        &mut g,
        "MATCH (a:Person {name: 'marko'})-[:KNOWS]->{1,1}(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("vadas")]]);
}

#[test]
fn with_filter() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WITH n.age AS age WHERE age > 30 RETURN age ORDER BY age",
    );
    assert_eq!(r, vec![vec![n(32.0)], vec![n(35.0)]]);
}

#[test]
fn comma_join_shared_var() {
    let mut g = modern();
    // marko knows josh, and marko created lop
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:KNOWS]->(b), (a)-[:CREATED]->(s) RETURN a.name, b.name, s.name ORDER BY b.name",
    );
    // a=marko (only marko has both KNOWS-out and CREATED-out); b in {vadas, josh}; s=lop
    assert_eq!(
        r,
        vec![
            vec![s("marko"), s("josh"), s("lop")],
            vec![s("marko"), s("vadas"), s("lop")]
        ]
    );
}

#[test]
fn optional_match_keeps_unmatched() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name ORDER BY a.name, b.name",
    );
    // josh/peter/vadas have no KNOWS-out → b null; marko → josh, vadas
    assert_eq!(
        r,
        vec![
            vec![s("josh"), Value::Null],
            vec![s("marko"), s("josh")],
            vec![s("marko"), s("vadas")],
            vec![s("peter"), Value::Null],
            vec![s("vadas"), Value::Null],
        ]
    );
}

#[test]
fn union_distinct_and_all() {
    let mut g = modern();
    let d = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS x UNION MATCH (s:Software) RETURN s.name AS x",
    );
    assert_eq!(d.len(), 6);
    let a = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS x UNION ALL MATCH (n:Person) RETURN n.name AS x",
    );
    assert_eq!(a.len(), 8);
}

#[test]
fn except_and_intersect() {
    let mut g = modern();
    let e = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS x EXCEPT MATCH (n:Person {name:'marko'}) RETURN n.name AS x",
    );
    assert_eq!(e.len(), 3);
    let i = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS x INTERSECT MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS x ORDER BY x",
    );
    assert_eq!(i, vec![vec![s("josh")], vec![s("peter")]]);
}

#[test]
fn exists_subquery() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE EXISTS { (n)-[:CREATED]->(s) } RETURN n.name ORDER BY n.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("marko")], vec![s("peter")]]);
}

#[test]
fn count_subquery() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name, COUNT { (n)-[:CREATED]->() } AS c ORDER BY n.name",
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
fn case_searched() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name, CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS tier ORDER BY n.name",
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
fn in_and_not_in() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE n.name IN ['marko','josh'] RETURN n.name ORDER BY n.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("marko")]]);
    let r2 = rows(
        &mut g,
        "MATCH (n:Person) WHERE n.name NOT IN ['marko'] RETURN count(*) AS c",
    );
    assert_eq!(r2, vec![vec![n(3.0)]]);
}

#[test]
fn is_null_and_is_truth() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Software) WHERE n.age IS NULL RETURN count(*) AS c",
    );
    assert_eq!(r, vec![vec![n(2.0)]]);
    let t = rows(
        &mut g,
        "RETURN true IS TRUE AS a, (1 = 2) IS FALSE AS b, null IS UNKNOWN AS c",
    );
    assert_eq!(t, vec![vec![b(true), b(true), b(true)]]);
}

#[test]
fn arithmetic_concat_and_negation() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "RETURN 7 % 3 AS a, -5 AS b, 2 + 3 * 4 AS c, 'x' || '!' AS d",
    );
    assert_eq!(r, vec![vec![n(1.0), n(-5.0), n(14.0), s("x!")]]);
}

#[test]
fn xor_precedence() {
    let mut g = modern();
    // ISO: OR/XOR same level, left-assoc. true XOR false = true.
    let r = rows(&mut g, "RETURN true XOR false AS a, (1=1) XOR (2=2) AS b");
    assert_eq!(r, vec![vec![b(true), b(false)]]);
}

#[test]
fn group_by_aggregate() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN s.name, count(*) AS c ORDER BY s.name",
    );
    assert_eq!(r, vec![vec![s("lop"), n(3.0)], vec![s("ripple"), n(1.0)]]);
}

#[test]
fn aggregate_functions() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person) RETURN min(p.age) AS lo, max(p.age) AS hi, sum(p.age) AS tot, avg(p.age) AS mean",
    );
    assert_eq!(r, vec![vec![n(27.0), n(35.0), n(123.0), n(123.0 / 4.0)]]);
}

#[test]
fn collect_list_distinct() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN collect_list(DISTINCT s.name) AS langs",
    );
    // marko/josh/peter → lop, josh → ripple; distinct → {lop, ripple} in encounter order
    assert_eq!(r, vec![vec![Value::List(vec![s("lop"), s("ripple")])]]);
}

#[test]
fn order_desc_limit_skip() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC SKIP 1 LIMIT 2",
    );
    // ages desc: peter35, josh32, marko29, vadas27 → skip peter → josh, marko
    assert_eq!(r, vec![vec![s("josh")], vec![s("marko")]]);
}

#[test]
fn distinct_projection() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN DISTINCT s.lang",
    );
    assert_eq!(r, vec![vec![s("java")]]);
}

#[test]
fn label_expression_or_not_wildcard() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "MATCH (n:Person|Software) RETURN count(*) AS c"),
        vec![vec![n(6.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:!Software) RETURN count(*) AS c"),
        vec![vec![n(4.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:%) RETURN count(*) AS c"),
        vec![vec![n(6.0)]]
    );
}

#[test]
fn property_map_and_inline_where() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "MATCH (n {name: 'marko'}) RETURN n.age"),
        vec![vec![n(29.0)]]
    );
    let r = rows(
        &mut g,
        "MATCH (n:Person WHERE n.age > 30) RETURN n.name ORDER BY n.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("peter")]]);
}

#[test]
fn parameters() {
    let mut g = modern();
    let mut params = Params::new();
    params.insert("who".to_string(), super::eval::Val::Str("vadas".into()));
    let r = qp(
        &mut g,
        "MATCH (n:Person) WHERE n.name = $who RETURN n.age",
        params,
    );
    assert_eq!(r, vec![vec![n(27.0)]]);
}

#[test]
fn indexed_param_lookup_matches_scan() {
    // An index seek must return the SAME rows as a full scan — proving the
    // param-resolved seek (WHERE `.k = $p` and inline `{k: $p}`) is correct, not
    // just fast. Compare an indexed graph's results to an un-indexed one.
    let mk = || {
        let mut g = modern();
        g.create_vertex_index("name");
        g
    };
    let param = |name: &str| {
        let mut p = Params::new();
        p.insert("who".to_string(), super::eval::Val::Str(name.into()));
        p
    };
    // WHERE with a $param → index seek on `name`.
    assert_eq!(
        qp(
            &mut mk(),
            "MATCH (n:Person) WHERE n.name = $who RETURN n.age",
            param("josh")
        ),
        vec![vec![n(32.0)]]
    );
    // Inline `{name: $param}` → index seek.
    assert_eq!(
        qp(
            &mut mk(),
            "MATCH (n:Person {name: $who}) RETURN n.age",
            param("marko")
        ),
        vec![vec![n(29.0)]]
    );
    // A miss returns nothing (not a stale index hit).
    assert!(qp(
        &mut mk(),
        "MATCH (n:Person {name: $who}) RETURN n.age",
        param("nobody")
    )
    .is_empty());
}

#[test]
fn prepared_plan_reused_with_params() {
    use super::eval::Val;
    let mut g = modern();
    // Lower once, execute many with different params slotted in positionally.
    let plan = super::prepare("MATCH (n:Person) WHERE n.name = $who RETURN n.age AS age").unwrap();

    let mut p1 = Params::new();
    p1.insert("who".to_string(), Val::Str("marko".into()));
    assert_eq!(
        plan.execute(&mut g, &p1)
            .unwrap()
            .rows()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![n(29.0)]]
    );

    let mut p2 = Params::new();
    p2.insert("who".to_string(), Val::Str("josh".into()));
    assert_eq!(
        plan.execute(&mut g, &p2)
            .unwrap()
            .rows()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![n(32.0)]]
    );
}

#[test]
fn prepared_write_persists() {
    use super::eval::Val;
    let mut g = modern();
    let ins = super::prepare("INSERT (n:Person {name: $nm, age: $age}) RETURN n.name").unwrap();
    let mut p = Params::new();
    p.insert("nm".to_string(), Val::Str("zoe".into()));
    p.insert("age".to_string(), Val::Num(40.0));
    assert_eq!(
        ins.execute(&mut g, &p)
            .unwrap()
            .rows()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        vec![vec![s("zoe")]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN count(*) AS c"),
        vec![vec![n(5.0)]]
    );
}

#[test]
fn scalar_functions() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "RETURN upper('ab') AS u, abs(-3) AS a, coalesce(null, 5) AS c, size([1,2,3]) AS sz, left('hello', 3) AS l",
    );
    assert_eq!(r, vec![vec![s("AB"), n(3.0), n(5.0), n(3.0), s("hel")]]);
}

#[test]
fn element_id_and_identity() {
    let mut g = modern();
    // element_id of a vertex is its external id; a = b is identity.
    let r = rows(
        &mut g,
        "MATCH (a:Person {name:'marko'}) RETURN element_id(a) AS id",
    );
    assert_eq!(r, vec![vec![s("1")]]); // marko is node 1 (see `crate::fixtures`)
    let c = rows(
        &mut g,
        "MATCH (a:Person), (b:Person) WHERE a = b RETURN count(*) AS c",
    );
    assert_eq!(c, vec![vec![n(4.0)]]); // only the 4 self-pairs
}

#[test]
fn is_labeled_predicate() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n) WHERE n IS LABELED Software RETURN count(*) AS c",
    );
    assert_eq!(r, vec![vec![n(2.0)]]);
    let r2 = rows(
        &mut g,
        "MATCH (n) WHERE n IS NOT LABELED Software RETURN count(*) AS c",
    );
    assert_eq!(r2, vec![vec![n(4.0)]]);
}

#[test]
fn three_valued_null_comparison() {
    let mut g = modern();
    // n.age > 30 is UNKNOWN for Software (no age) → excluded; only josh, peter
    let r = rows(&mut g, "MATCH (n) WHERE n.age > 30 RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(2.0)]]);
}

#[test]
fn edge_property_projection() {
    let mut g = modern();
    // Bind the edge variable and read its property — now backed by edge columns.
    let r = rows(
        &mut g,
        "MATCH (a:Person {name:'marko'})-[r:KNOWS]->(b) RETURN b.name, r.weight ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh"), n(1.0)], vec![s("vadas"), n(0.5)]]);
}

#[test]
fn edge_property_inline_where() {
    let mut g = modern();
    // Inline edge predicate filters on the edge's own property.
    let r = rows(
        &mut g,
        "MATCH (a)-[r:CREATED WHERE r.weight >= 1.0]->(s) RETURN a.name, s.name",
    );
    assert_eq!(r, vec![vec![s("josh"), s("ripple")]]);
}

#[test]
fn edge_property_map_constraint() {
    let mut g = modern();
    // Property-map constraint on an edge.
    let r = rows(
        &mut g,
        "MATCH (a:Person)-[:CREATED {weight: 0.4}]->(s) RETURN a.name ORDER BY a.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("marko")]]);
}

#[test]
fn edge_property_aggregate() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH ()-[r:CREATED]->() RETURN sum(r.weight) AS total, count(*) AS c",
    );
    assert_eq!(r, vec![vec![n(0.4 + 1.0 + 0.4 + 0.2), n(4.0)]]);
}

#[test]
fn limit_pushdown_streams_match_order() {
    let mut g = modern();
    // No ORDER BY → streamable; LIMIT short-circuits matching in declaration order.
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN n.name LIMIT 2"),
        vec![vec![s("marko")], vec![s("vadas")]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN n.name SKIP 1 LIMIT 2"),
        vec![vec![s("vadas")], vec![s("josh")]]
    );
}

#[test]
fn order_by_limit_is_global_not_pushed_down() {
    let mut g = modern();
    // ORDER BY present → cap NOT applied; result is the globally smallest ages.
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name ORDER BY n.age LIMIT 2",
    );
    assert_eq!(r, vec![vec![s("vadas")], vec![s("marko")]]);
}

#[test]
fn group_by_expression() {
    let mut g = modern();
    // group key is an expression (age parity), not just a property.
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.age % 2 AS parity, count(*) AS c ORDER BY parity",
    );
    assert_eq!(r, vec![vec![n(0.0), n(1.0)], vec![n(1.0), n(3.0)]]);
}

#[test]
fn count_distinct_aggregate() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN count(DISTINCT s.lang) AS c",
    );
    assert_eq!(r, vec![vec![n(1.0)]]);
}

#[test]
fn nested_function_calls() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN upper(left('hello', 3)) AS x"),
        vec![vec![s("HEL")]]
    );
}

#[test]
fn case_simple_form() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "RETURN CASE 2 WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END AS x",
    );
    assert_eq!(r, vec![vec![s("b")]]);
}

#[test]
fn order_by_expression() {
    let mut g = modern();
    // ORDER BY an expression (negated age) → effectively descending by age.
    let r = rows(&mut g, "MATCH (n:Person) RETURN n.name ORDER BY n.age * -1");
    assert_eq!(
        r,
        vec![
            vec![s("peter")],
            vec![s("josh")],
            vec![s("marko")],
            vec![s("vadas")]
        ]
    );
}

#[test]
fn concat_coerces_number_and_comparison_projects_bool() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN 'age=' || 29 AS x"),
        vec![vec![s("age=29")]]
    );
    let r = rows(
        &mut g,
        "MATCH (n:Person {name:'josh'}) RETURN n.age > 30 AS old",
    );
    assert_eq!(r, vec![vec![b(true)]]);
}

#[test]
fn three_valued_boolean_logic() {
    let mut g = modern();
    // AND/OR/XOR/NOT with UNKNOWN (null), per ISO Kleene logic.
    let r = rows(
        &mut g,
        "RETURN null AND false AS a, null AND true AS b, null OR true AS c, \
         null OR false AS d, NOT null AS e, null XOR true AS f",
    );
    assert_eq!(
        r,
        vec![vec![
            b(false),
            Value::Null,
            b(true),
            Value::Null,
            Value::Null,
            Value::Null
        ]]
    );
}

#[test]
fn null_comparison_and_arithmetic_propagate() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "RETURN (1 = 1) AS a, (1 = 2) AS b, (null = 1) AS c, (1 + null) AS d",
    );
    assert_eq!(r, vec![vec![b(true), b(false), Value::Null, Value::Null]]);
}

#[test]
fn in_three_valued_logic() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "RETURN 1 IN [1,2] AS a, 3 IN [1,2] AS b, null IN [] AS c, \
         1 IN [null] AS d, 3 IN [1,null] AS e, 1 IN [1,null] AS f",
    );
    // null IN [] is FALSE (empty disjunction); a TRUE equality beats UNKNOWN.
    assert_eq!(
        r,
        vec![vec![
            b(true),
            b(false),
            b(false),
            Value::Null,
            Value::Null,
            b(true)
        ]]
    );
}

#[test]
fn two_match_clauses() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name:'marko'}) MATCH (a)-[:KNOWS]->(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("vadas")]]);
}

#[test]
fn with_carries_element_forward() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (a:Person {name:'josh'}) WITH a MATCH (a)-[:CREATED]->(x) RETURN x.name ORDER BY x.name");
    assert_eq!(r, vec![vec![s("lop")], vec![s("ripple")]]);
}

#[test]
fn with_then_match_expand_count() {
    let mut g = modern();
    // All CREATED edges: marko→lop, josh→ripple, josh→lop, peter→lop = 4.
    let r = rows(
        &mut g,
        "MATCH (a:Person) WITH a MATCH (a)-[:CREATED]->(x) RETURN count(*) AS c",
    );
    assert_eq!(r, vec![vec![n(4.0)]]);
}

#[test]
fn with_carry_computed_col_across_expand() {
    let mut g = modern();
    // Carry a.age forward, expand KNOWS, keep neighbors older than the carried age.
    // marko(29)→vadas(27): no; marko(29)→josh(32): yes ⇒ 1.
    let r = rows(&mut g, "MATCH (a:Person) WITH a, a.age AS aage MATCH (a)-[:KNOWS]->(b) WHERE b.age > aage RETURN count(*) AS c");
    assert_eq!(r, vec![vec![n(1.0)]]);
}

#[test]
fn with_value_col_survives_expand() {
    let mut g = modern();
    // The computed column `an` (a value column) must ride through the expand and
    // appear in output alongside the expanded b's property.
    let r = rows(&mut g, "MATCH (a:Person {name:'marko'}) WITH a, a.name AS an MATCH (a)-[:KNOWS]->(b {name:'josh'}) RETURN an, b.age");
    assert_eq!(r, vec![vec![s("marko"), n(32.0)]]);
}

#[test]
fn not_exists_subquery() {
    let mut g = modern();
    // vadas is the only Person who created nothing.
    let r = rows(
        &mut g,
        "MATCH (n:Person) WHERE NOT EXISTS { (n)-[:CREATED]->() } RETURN n.name ORDER BY n.name",
    );
    assert_eq!(r, vec![vec![s("vadas")]]);
}

#[test]
fn exists_with_inner_where() {
    let mut g = modern();
    let r = rows(&mut g, "MATCH (n:Person) WHERE EXISTS { (n)-[:CREATED]->(s) WHERE s.name = 'ripple' } RETURN n.name");
    assert_eq!(r, vec![vec![s("josh")]]);
}

#[test]
fn count_over_empty_is_zero() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "MATCH (n:Ghost) RETURN count(*) AS c"),
        vec![vec![n(0.0)]]
    );
}

#[test]
fn min_max_over_empty_is_null() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Ghost) RETURN min(n.age) AS lo, max(n.age) AS hi",
    );
    assert_eq!(r, vec![vec![Value::Null, Value::Null]]);
}

#[test]
fn order_by_multiple_keys() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) RETURN p.name, s.name ORDER BY s.name, p.name",
    );
    assert_eq!(
        r,
        vec![
            vec![s("josh"), s("lop")],
            vec![s("marko"), s("lop")],
            vec![s("peter"), s("lop")],
            vec![s("josh"), s("ripple")],
        ]
    );
}

#[test]
fn order_by_alias() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (n:Person) RETURN n.name AS who, n.age AS yrs ORDER BY yrs DESC LIMIT 2",
    );
    assert_eq!(r, vec![vec![s("peter"), n(35.0)], vec![s("josh"), n(32.0)]]);
}

#[test]
fn coalesce_and_nullif() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "RETURN coalesce(null, null, 7) AS a, nullif(3, 3) AS b, nullif(3, 4) AS c",
    );
    assert_eq!(r, vec![vec![n(7.0), Value::Null, n(3.0)]]);
}

#[test]
fn multi_stage_with_aggregate_filter() {
    let mut g = modern();
    // group, aggregate, filter on the aggregate, then return.
    let r = rows(
        &mut g,
        "MATCH (p:Person)-[:CREATED]->(s:Software) WITH s.name AS sw, count(*) AS c WHERE c > 1 RETURN sw, c",
    );
    assert_eq!(r, vec![vec![s("lop"), n(3.0)]]);
}

#[test]
fn return_star_columns_are_bound_vars() {
    let mut g = modern();
    // `*` projects every in-scope variable as a column (here just `n`).
    let (cols, r) = q(&mut g, "MATCH (n:Person {name:'marko'}) RETURN *");
    assert_eq!(cols, vec!["n"]);
    // A returned node serializes to a rich `{id, labels, properties}` map
    // (byte-identical to the TS engine); keys/labels are sorted.
    let node = Value::Map(vec![
        (Arc::from("id"), s("1")), // marko is node 1 (see `crate::fixtures`)
        (Arc::from("labels"), Value::List(vec![s("Person")])),
        (
            Arc::from("properties"),
            Value::Map(vec![
                (Arc::from("age"), n(29.0)),
                (Arc::from("name"), s("marko")),
            ]),
        ),
    ]);
    assert_eq!(r, vec![vec![node]]);
}

#[test]
fn with_star_carries_all_vars() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person {name:'marko'})-[:KNOWS]->(b) WITH * WHERE b.age > 28 RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")]]);
}

#[test]
fn undirected_var_length() {
    let mut g = modern();
    // From vadas, KNOWS is incoming (marko→vadas); undirected reaches marko,
    // then marko's other KNOWS reaches josh.
    let r = rows(
        &mut g,
        "MATCH (a:Person {name:'vadas'})-[:KNOWS]-*(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("marko")], vec![s("vadas")]]);
}

// --- write clauses (the graph is mutable) -----------------------------------

#[test]
fn insert_multi_label_node() {
    let mut g = modern();
    // ISO label conjunction `:A&B` names both labels on creation.
    let r = rows(
        &mut g,
        "INSERT (n:Person&Admin {name:'root'}) RETURN n.name",
    );
    assert_eq!(r, vec![vec![s("root")]]);
    assert_eq!(
        rows(&mut g, "MATCH (n:Admin) RETURN n.name"),
        vec![vec![s("root")]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:Person&Admin) RETURN n.name"),
        vec![vec![s("root")]]
    );
}

#[test]
fn insert_node_then_return() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "INSERT (n:Person {name: 'newbie', age: 99}) RETURN n.name, n.age",
    );
    assert_eq!(r, vec![vec![s("newbie"), n(99.0)]]);
    // The new node is matchable afterward, and Person count grew 4 → 5.
    assert_eq!(
        rows(&mut g, "MATCH (p:Person) RETURN count(*) AS c"),
        vec![vec![n(5.0)]]
    );
    assert_eq!(g.vertex_count(), 7);
}

#[test]
fn insert_edge_between_matched_nodes() {
    let mut g = modern();
    // marko does not yet know peter; create the edge, then traverse it.
    rows(&mut g, "MATCH (a:Person {name:'marko'}), (b:Person {name:'peter'}) INSERT (a)-[:KNOWS {weight: 0.9}]->(b)");
    let r = rows(
        &mut g,
        "MATCH (a:Person {name:'marko'})-[r:KNOWS]->(b) RETURN b.name, r.weight ORDER BY b.name",
    );
    assert_eq!(
        r,
        vec![
            vec![s("josh"), n(1.0)],
            vec![s("peter"), n(0.9)],
            vec![s("vadas"), n(0.5)]
        ]
    );
}

#[test]
fn set_property_and_label() {
    let mut g = modern();
    rows(
        &mut g,
        "MATCH (n:Person {name:'vadas'}) SET n.age = 28, n:Senior",
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:Person {name:'vadas'}) RETURN n.age"),
        vec![vec![n(28.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:Senior) RETURN n.name"),
        vec![vec![s("vadas")]]
    );
}

#[test]
fn set_new_property_creates_column() {
    let mut g = modern();
    // 'city' is a brand-new key — the column is created on demand.
    rows(
        &mut g,
        "MATCH (n:Person {name:'josh'}) SET n.city = 'berlin'",
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:Person) WHERE n.city = 'berlin' RETURN n.name"
        ),
        vec![vec![s("josh")]]
    );
}

#[test]
fn set_promotes_column_to_mixed_on_type_change() {
    let mut g = modern();
    // age is a Num column; setting a string promotes it to Mixed (lossless).
    rows(
        &mut g,
        "MATCH (n:Person {name:'marko'}) SET n.age = 'twenty-nine'",
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:Person {name:'marko'}) RETURN n.age"),
        vec![vec![s("twenty-nine")]]
    );
    // other rows keep their numeric ages
    assert_eq!(
        rows(&mut g, "MATCH (n:Person {name:'josh'}) RETURN n.age"),
        vec![vec![n(32.0)]]
    );
}

#[test]
fn remove_property() {
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name:'peter'}) REMOVE n.age");
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) WHERE n.age IS NULL RETURN n.name"),
        vec![vec![s("peter")]]
    );
}

#[test]
fn set_null_stores_a_present_null_and_remove_deletes_it() {
    // Divergence from Cypher: `SET n.k = null` STORES a present null — it does
    // NOT remove the property. `REMOVE` is the explicit deletion path. Both a
    // present null and an absent key satisfy `IS NULL` (three-valued logic).
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name:'marko'}) SET n.nick = null");

    let marko = g.vid.get("1").unwrap() as usize; // marko is node 1
    assert!(
        g.props.is_present(marko, "nick"),
        "SET null stores a PRESENT null, not a removal"
    );
    assert_eq!(g.props.value(marko, "nick", &g.strs), Value::Null);

    // IS NULL matches the stored null.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:Person {name:'marko'}) WHERE n.nick IS NULL RETURN n.name"
        ),
        vec![vec![s("marko")]]
    );

    // REMOVE actually deletes it.
    rows(&mut g, "MATCH (n:Person {name:'marko'}) REMOVE n.nick");
    assert!(
        !g.props.is_present(marko, "nick"),
        "REMOVE deletes the property outright"
    );
}

#[test]
fn delete_isolated_vertex() {
    let mut g = modern();
    // ripple has only an incoming CREATED edge, so plain DELETE needs DETACH;
    // delete an edge first, then a now-isolated vertex.
    rows(
        &mut g,
        "MATCH (:Person {name:'josh'})-[r:CREATED]->(:Software {name:'ripple'}) DELETE r",
    );
    rows(&mut g, "MATCH (n:Software {name:'ripple'}) DELETE n");
    assert_eq!(
        rows(&mut g, "MATCH (s:Software) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
}

#[test]
fn detach_delete_cascades_edges() {
    let mut g = modern();
    rows(&mut g, "MATCH (n:Person {name:'marko'}) DETACH DELETE n");
    // marko and all his edges are gone; remaining people = 3.
    assert_eq!(
        rows(&mut g, "MATCH (p:Person) RETURN count(*) AS c"),
        vec![vec![n(3.0)]]
    );
    // lop now has 2 creators (josh, peter) instead of 3.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (:Person)-[:CREATED]->(s:Software {name:'lop'}) RETURN count(*) AS c"
        ),
        vec![vec![n(2.0)]]
    );
}

#[test]
fn scalar_functions_graph_string_list_conversion() {
    let mut g = modern();
    // graph functions (label/key order is sorted for determinism)
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:Person {name:'marko'}) RETURN labels(n) AS l"
        ),
        vec![vec![Value::List(vec![s("Person")])]]
    );
    assert_eq!(
        rows(&mut g, "MATCH ()-[r:KNOWS]->() RETURN type(r) AS t LIMIT 1"),
        vec![vec![s("KNOWS")]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:Person {name:'marko'}) RETURN keys(n) AS k"
        ),
        vec![vec![Value::List(vec![s("age"), s("name")])]]
    );
    // conversion
    assert_eq!(
        rows(&mut g, "RETURN to_integer('42') AS x"),
        vec![vec![n(42.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN to_float('3.5') AS x"),
        vec![vec![n(3.5)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN to_string(42) AS x"),
        vec![vec![s("42")]]
    );
    // string / list — substring is 1-based (SQL / ISO GQL): positions 1..3.
    assert_eq!(
        rows(&mut g, "RETURN substring('hello', 1, 3) AS x"),
        vec![vec![s("hel")]]
    );
    assert_eq!(
        rows(&mut g, "RETURN substring('hello', 4) AS x"),
        vec![vec![s("lo")]]
    );
    assert_eq!(
        rows(&mut g, "RETURN substring('hello', 0, 3) AS x"),
        vec![vec![s("he")]]
    );
    assert_eq!(
        rows(&mut g, "RETURN split('a,b,c', ',') AS x"),
        vec![vec![Value::List(vec![s("a"), s("b"), s("c")])]]
    );
    assert_eq!(
        rows(&mut g, "RETURN replace('a.b.c', '.', '-') AS x"),
        vec![vec![s("a-b-c")]]
    );
    assert_eq!(
        rows(&mut g, "RETURN head([1, 2, 3]) AS x"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN last([1, 2, 3]) AS x"),
        vec![vec![n(3.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN reverse('abc') AS x"),
        vec![vec![s("cba")]]
    );
}

#[test]
fn math_round_sign_pi_e() {
    let mut g = modern();
    // round: half away from zero, optional digits (negative rounds to tens).
    assert_eq!(rows(&mut g, "RETURN round(2.5) AS x"), vec![vec![n(3.0)]]);
    assert_eq!(rows(&mut g, "RETURN round(-2.5) AS x"), vec![vec![n(-3.0)]]);
    assert_eq!(
        rows(&mut g, "RETURN round(1.2345, 2) AS x"),
        vec![vec![n(1.23)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN round(1234.5678, -2) AS x"),
        vec![vec![n(1200.0)]]
    );
    // sign: -1 | 0 | 1.
    assert_eq!(rows(&mut g, "RETURN sign(-3.7) AS x"), vec![vec![n(-1.0)]]);
    assert_eq!(rows(&mut g, "RETURN sign(0) AS x"), vec![vec![n(0.0)]]);
    assert_eq!(rows(&mut g, "RETURN sign(5) AS x"), vec![vec![n(1.0)]]);
    // 0-arg constants.
    assert_eq!(
        rows(&mut g, "RETURN pi() AS x"),
        vec![vec![n(std::f64::consts::PI)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN e() AS x"),
        vec![vec![n(std::f64::consts::E)]]
    );
    // null in → null out.
    assert_eq!(
        rows(&mut g, "RETURN round(null) AS x"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn sum_duration_computes_avg_duration_throws() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["T"],"properties":{"g":1,"d":{"@duration":"P1M10D"}}}"#,
        r#"{"type":"node","id":"b","labels":["T"],"properties":{"g":1,"d":{"@duration":"P2M5D"}}}"#,
        r#"{"type":"node","id":"c","labels":["T"],"properties":{"g":2,"d":{"@duration":"P7D"}}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let tdur = |s: &str| Value::Temporal(crate::temporal::Temporal::parse("duration", s).unwrap());
    // sum(DURATION): component-wise total (P1M10D + P2M5D + P7D = P3M22D), not NaN→null.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN sum(n.d) AS s"),
        vec![vec![tdur("P3M22D")]]
    );
    // Grouped sum (routes through the scalar accumulator): per-group totals.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN n.g AS g, sum(n.d) AS s ORDER BY n.g"
        ),
        vec![vec![n(1.0), tdur("P3M15D")], vec![n(2.0), tdur("P7D")],]
    );
    // avg(DURATION) is a loud data exception (needs unrepresentable duration÷count).
    let err = parse("MATCH (n:T) RETURN avg(n.d) AS a")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::DataException);
    // sum over a non-DURATION temporal (dates aren't summable) also throws.
    let lines2 =
        [r#"{"type":"node","id":"x","labels":["T"],"properties":{"dt":{"@date":"2020-01-01"}}}"#];
    let mut g2 = ndjson::decode(&lines2.join("\n")).unwrap();
    let err2 = parse("MATCH (n:T) RETURN sum(n.dt) AS s")
        .unwrap()
        .execute(&mut g2, &Params::new())
        .unwrap_err();
    assert_eq!(err2.code, crate::error_codes::ErrorCode::DataException);
}

#[test]
fn math_atan2_binary_arctangent() {
    let mut g = modern();
    // atan2(y, x): quadrant-correct angle. Exact/stable values.
    assert_eq!(
        rows(&mut g, "RETURN atan2(1, 1) AS x"),
        vec![vec![n(std::f64::consts::FRAC_PI_4)]]
    );
    assert_eq!(rows(&mut g, "RETURN atan2(0, 1) AS x"), vec![vec![n(0.0)]]);
    assert_eq!(
        rows(&mut g, "RETURN atan2(1, 0) AS x"),
        vec![vec![n(std::f64::consts::FRAC_PI_2)]]
    );
    // null operand → null.
    assert_eq!(
        rows(&mut g, "RETURN atan2(null, 1) AS x"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn order_by_and_minmax_total_order_across_types() {
    let lines = [
        r#"{"type":"node","id":"1","labels":["X"],"properties":{"v":2}}"#,
        r#"{"type":"node","id":"2","labels":["X"],"properties":{"v":"a"}}"#,
        r#"{"type":"node","id":"3","labels":["X"],"properties":{"v":1}}"#,
        r#"{"type":"node","id":"4","labels":["X"],"properties":{"v":true}}"#,
        r#"{"type":"node","id":"5","labels":["X"],"properties":{"v":"b"}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let col = |g: &mut Graph, q: &str| -> Vec<Value> {
        rows(g, q).into_iter().map(|r| r[0].clone()).collect()
    };
    // Total order across type groups: number < string < boolean.
    assert_eq!(
        col(&mut g, "MATCH (n:X) RETURN n.v AS v ORDER BY n.v"),
        vec![n(1.0), n(2.0), s("a"), s("b"), b(true)]
    );
    assert_eq!(
        col(&mut g, "MATCH (n:X) RETURN n.v AS v ORDER BY n.v DESC"),
        vec![b(true), s("b"), s("a"), n(2.0), n(1.0)]
    );
    // min / max use the same total order.
    assert_eq!(
        rows(&mut g, "MATCH (n:X) RETURN min(n.v) AS m"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:X) RETURN max(n.v) AS m"),
        vec![vec![b(true)]]
    );
}

#[test]
fn set_style_list_functions() {
    let mut g = modern();
    let list = |xs: Vec<Value>| vec![vec![Value::List(xs)]];
    assert_eq!(
        rows(&mut g, "RETURN list_union([1,2,2,3], [3,4,5]) AS x"),
        list(vec![n(1.0), n(2.0), n(3.0), n(4.0), n(5.0)])
    );
    assert_eq!(
        rows(&mut g, "RETURN intersection([1,2,3,3], [3,3,4,5]) AS x"),
        list(vec![n(3.0)])
    );
    assert_eq!(
        rows(&mut g, "RETURN difference([1,2,2,3], [3,4,5]) AS x"),
        list(vec![n(1.0), n(2.0)])
    );
    // ISO GQL: list_contains returns numeric 1 / 0.
    assert_eq!(
        rows(&mut g, "RETURN list_contains([1,2,3], 2) AS x"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN list_contains([1,2,3], 9) AS x"),
        vec![vec![n(0.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN list_sort([3,1,4,1,5]) AS x"),
        list(vec![n(1.0), n(1.0), n(3.0), n(4.0), n(5.0)])
    );
    assert_eq!(
        rows(&mut g, "RETURN list_sort([3,1,2], 'desc') AS x"),
        list(vec![n(3.0), n(2.0), n(1.0)])
    );
    assert_eq!(
        rows(&mut g, "RETURN list_sort([3,1,null,2]) AS x"),
        list(vec![n(1.0), n(2.0), n(3.0), Value::Null])
    );
    assert_eq!(
        rows(
            &mut g,
            "RETURN list_sort([3,1,null,2], 'asc', 'first') AS x"
        ),
        list(vec![Value::Null, n(1.0), n(2.0), n(3.0)])
    );
}

#[test]
fn infix_string_match_predicates() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN 'Hello World' CONTAINS 'World' AS x"),
        vec![vec![Value::Bool(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 'Hello World' STARTS WITH 'Hello' AS x"),
        vec![vec![Value::Bool(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 'Hello World' ENDS WITH 'World' AS x"),
        vec![vec![Value::Bool(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN 'abc' CONTAINS 'z' AS x"),
        vec![vec![Value::Bool(false)]]
    );
    // as a WHERE filter
    assert_eq!(
        rows(
            &mut g,
            "MATCH (p:Person) WHERE p.name STARTS WITH 'ma' RETURN p.name AS x"
        ),
        vec![vec![s("marko")]]
    );
}

#[test]
fn cast_desugars_to_conversion_functions() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN CAST('42' AS INTEGER) AS x"),
        vec![vec![n(42.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN CAST(3.7 AS INT) AS x"),
        vec![vec![n(3.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN CAST('3.5' AS FLOAT) AS x"),
        vec![vec![n(3.5)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN CAST(42 AS STRING) AS x"),
        vec![vec![s("42")]]
    );
    assert_eq!(
        rows(&mut g, "RETURN CAST('yes' AS BOOL) AS x"),
        vec![vec![Value::Bool(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN CAST('ab' AS LIST) AS x"),
        vec![vec![Value::List(vec![s("a"), s("b")])]]
    );
    assert_eq!(
        rows(&mut g, "RETURN CAST('nope' AS INT) AS x"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn cast_to_unrepresentable_type_is_a_syntax_error() {
    assert!(parse("RETURN CAST(1 AS BYTES) AS x").is_err());
    assert!(parse("RETURN CAST(1 AS RECORD) AS x").is_err());
    // A temporal target (DATE/DATETIME/…) is now a representable CAST that
    // desugars to the temporal constructor function — no longer a syntax error.
    assert!(parse("RETURN CAST('2020-01-01' AS DATE) AS x").is_ok());
    assert!(parse("RETURN CAST('2020-01-01T00:00:00' AS LOCAL DATETIME) AS x").is_ok());
}

#[test]
fn unknown_function_errors_instead_of_silent_null() {
    let mut g = modern();
    let err = parse("RETURN nope_fn(1) AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::UnknownFunction);
    // The message NAMES the offending function (parity with the TS engine).
    assert!(
        err.message.contains("nope_fn()"),
        "message should name the function, got: {}",
        err.message
    );
}

#[test]
fn camelcase_procedure_name_suggests_the_snake_case_one() {
    // The GQL `CALL` catalog is snake_case; a camelCase spelling of a real
    // algorithm (the JS/Gremlin surface name) faults E_UNSUPPORTED with a "did
    // you mean" hint pointing at the canonical name. The TS engine emits the
    // same message byte-for-byte. An unrelated name gets no suggestion.
    let mut g = modern();
    let err = parse("CALL connectedComponents({}) YIELD node RETURN node")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::Unsupported);
    assert_eq!(
        err.message,
        "unknown procedure: connectedComponents (did you mean 'connected_components'?)"
    );

    let err = parse("CALL pageRank({}) YIELD node RETURN node")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(
        err.message,
        "unknown procedure: pageRank (did you mean 'pagerank'?)"
    );

    let err = parse("CALL totallyBogus({}) YIELD node RETURN node")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.message, "unknown procedure: totallyBogus");
}

#[test]
fn call_betweenness_pivots_reaches_the_algorithm() {
    // `CALL betweenness({pivots: k})` must actually sample. The config-map path
    // (`apply_algo_config`) used to drop `pivots` (and `seedProperty`), silently
    // running the exact O(V·E) pass regardless — a documented, unit-tested
    // feature that was unreachable through its primary GQL surface. On a clean
    // directed path an 8-of-16 pivot sample scales differently from the exact
    // pass, so the two sums must differ. If `pivots` is dropped, both are exact
    // and equal and this fails.
    let mut g = ndjson::decode("").unwrap();
    let count = 16;
    for i in 0..count {
        rows(&mut g, &format!("INSERT (:P {{id: 'n{i}'}})"));
    }
    for i in 0..count - 1 {
        let j = i + 1;
        rows(
            &mut g,
            &format!("MATCH (a:P {{id:'n{i}'}}),(b:P {{id:'n{j}'}}) INSERT (a)-[:E]->(b)"),
        );
    }
    let exact = rows(
        &mut g,
        "CALL betweenness({}) YIELD node, centrality RETURN sum(centrality) AS s",
    );
    let sampled = rows(
        &mut g,
        "CALL betweenness({pivots: 2}) YIELD node, centrality RETURN sum(centrality) AS s",
    );
    assert_ne!(
        exact[0][0],
        n(0.0),
        "the path should have nonzero betweenness"
    );
    assert_ne!(
        sampled[0][0], exact[0][0],
        "pivots must change the result — the config must reach the algorithm"
    );
}

#[test]
fn string_vs_temporal_comparison_is_a_type_error() {
    // An ORDERED comparison between a temporal value and a non-temporal one — an
    // untagged string param vs a stored DATE, or a number vs a DATE — is a type
    // error (E_INVALID_VALUE), not a silent empty result. Equality is unaffected
    // (simply unequal), and a real DATE operand works. Byte-identical to TS.
    let mut g = ndjson::decode("").unwrap();
    rows(&mut g, "INSERT (:R {vf: DATE '2021-06-01'})");

    let mut p = Params::new();
    p.insert("x".to_string(), super::eval::Val::Str("2021-01-01".into()));
    let e = parse("MATCH (r:R) WHERE r.vf <= $x RETURN r")
        .unwrap()
        .execute(&mut g, &p)
        .unwrap_err();
    assert_eq!(e.code, crate::error_codes::ErrorCode::InvalidValue);

    // number vs temporal is likewise a type error …
    assert!(exec_err(&mut g, "MATCH (r:R) WHERE r.vf < 5 RETURN r"));
    // … but equality with a mismatched type is fine (just unequal → 0 rows) …
    let cnt = rows(&mut g, "MATCH (r:R) WHERE r.vf = 5 RETURN count(*) AS c");
    assert_eq!(cnt[0][0], n(0.0));
    // … and a proper DATE comparison still works.
    let ok = rows(
        &mut g,
        "MATCH (r:R) WHERE r.vf <= DATE '2022-01-01' RETURN count(*) AS c",
    );
    assert_eq!(ok[0][0], n(1.0));
}

#[test]
fn call_config_key_validation() {
    // An unknown or wrong-typed CALL config key faults E_INVALID_VALUE (a silently
    // dropped key once hid the pivots bug). A near-miss gets a "did you mean" hint;
    // the TS engine emits the same code and message byte-for-byte.
    let mut g = modern();
    use crate::error_codes::ErrorCode::InvalidValue;
    let msg = |g: &mut Graph, q: &str| -> (crate::error_codes::ErrorCode, String) {
        let e = parse(q).unwrap().execute(g, &Params::new()).unwrap_err();
        (e.code, e.message)
    };

    assert_eq!(
        msg(
            &mut g,
            "CALL betweenness({pivot: 8}) YIELD node RETURN node"
        ),
        (
            InvalidValue,
            "unknown config key 'pivot' (did you mean 'pivots'?)".to_string()
        )
    );
    assert_eq!(
        msg(
            &mut g,
            "CALL betweenness({bogusKey: 1}) YIELD node RETURN node"
        ),
        (InvalidValue, "unknown config key 'bogusKey'".to_string())
    );
    assert_eq!(
        msg(
            &mut g,
            "CALL betweenness({pivots: 'x'}) YIELD node RETURN node"
        ),
        (
            InvalidValue,
            "config key 'pivots' expects a number".to_string()
        )
    );
    // A valid key of the right type still works.
    rows(
        &mut g,
        "CALL pagerank({iterations: 5}) YIELD node RETURN count(*) AS c",
    );
}

/// A bound-both-endpoints reachability `EXISTS { (a)-[:R]->+(b) }` (both anchored) takes
/// the bidirectional fast path; it must give the SAME boolean as a one-directional search
/// across reachable / UNreachable (the exhaust-the-cone case) / cycles / self-loops / the
/// zero-length `*` / reversed direction.
#[test]
fn exists_bound_endpoint_reachability_is_bidirectional_and_exact() {
    // chain a→b→c→d; isolated e; self-loop f→f; 2-cycle g↔h.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"node","id":"e","labels":["N"],"properties":{"id":"e"}}"#,
        r#"{"type":"node","id":"f","labels":["N"],"properties":{"id":"f"}}"#,
        r#"{"type":"node","id":"g","labels":["N"],"properties":{"id":"g"}}"#,
        r#"{"type":"node","id":"h","labels":["N"],"properties":{"id":"h"}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["R"]}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["R"]}"#,
        r#"{"type":"edge","from":"c","to":"d","labels":["R"]}"#,
        r#"{"type":"edge","from":"f","to":"f","labels":["R"]}"#,
        r#"{"type":"edge","from":"g","to":"h","labels":["R"]}"#,
        r#"{"type":"edge","from":"h","to":"g","labels":["R"]}"#,
    ]);
    // Both endpoints bound (the outer MATCH anchors a AND b) → bidirectional fast path.
    let reaches = |g: &mut Graph, from: &str, to: &str, arrow: &str| -> bool {
        !rows(
            g,
            &format!(
                "MATCH (a:N {{id:'{from}'}}), (b:N {{id:'{to}'}}) \
                 WHERE EXISTS {{ MATCH (a)-[:R]->{arrow}(b) }} RETURN 1 AS x"
            ),
        )
        .is_empty()
    };
    assert!(reaches(&mut g, "a", "d", "+")); // a→→→d
    assert!(reaches(&mut g, "a", "b", "+"));
    assert!(!reaches(&mut g, "a", "e", "+")); // e unreachable — NEGATIVE, must exhaust
    assert!(!reaches(&mut g, "d", "a", "+")); // wrong direction — NEGATIVE
    assert!(reaches(&mut g, "g", "g", "+")); // g on a 2-cycle
    assert!(reaches(&mut g, "f", "f", "+")); // self-loop
    assert!(!reaches(&mut g, "e", "e", "+")); // no self-loop; `+` needs ≥1 hop
    assert!(reaches(&mut g, "e", "e", "*")); // `*` admits the zero-length self path
    assert!(reaches(&mut g, "a", "a", "*"));
    assert!(!reaches(&mut g, "a", "a", "+")); // a not on a cycle
                                              // Reversed direction: `<-[:R]-+` = follow edges backward.
    assert!(!rows(
        &mut g,
        "MATCH (a:N {id:'a'}), (d:N {id:'d'}) WHERE EXISTS { MATCH (d)<-[:R]-+(a) } RETURN 1 AS x",
    )
    .is_empty());
}

/// A MULTI-segment bound-both-endpoints EXISTS (the ReBAC `check()` shape) takes the
/// meet-in-the-middle fast path; it must give the SAME boolean as the general matcher.
/// Covers nested-group + resource-inheritance grants, a direct (0-hop) grant, a deny, and a
/// `->*` vs `->{0,N}` (general) differential.
#[test]
fn exists_multiseg_meet_in_middle_matches_general() {
    // u1∈g1∈g2; g2 OWNER f1; f1 PARENT d1. u2 OWNER d3 directly. d2 ungranted.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"u1","labels":["User"],"properties":{"id":"u1"}}"#,
        r#"{"type":"node","id":"u2","labels":["User"],"properties":{"id":"u2"}}"#,
        r#"{"type":"node","id":"g1","labels":["Grp"],"properties":{"id":"g1"}}"#,
        r#"{"type":"node","id":"g2","labels":["Grp"],"properties":{"id":"g2"}}"#,
        r#"{"type":"node","id":"f1","labels":["Res"],"properties":{"id":"f1"}}"#,
        r#"{"type":"node","id":"d1","labels":["Res"],"properties":{"id":"d1"}}"#,
        r#"{"type":"node","id":"d2","labels":["Res"],"properties":{"id":"d2"}}"#,
        r#"{"type":"node","id":"d3","labels":["Res"],"properties":{"id":"d3"}}"#,
        r#"{"type":"edge","from":"u1","to":"g1","labels":["MEMBER"]}"#,
        r#"{"type":"edge","from":"g1","to":"g2","labels":["MEMBER"]}"#,
        r#"{"type":"edge","from":"g2","to":"f1","labels":["OWNER"]}"#,
        r#"{"type":"edge","from":"f1","to":"d1","labels":["PARENT"]}"#,
        r#"{"type":"edge","from":"u2","to":"d3","labels":["OWNER"]}"#,
    ]);
    let check = |g: &mut Graph, u: &str, t: &str, arrow: &str| -> bool {
        !rows(
            g,
            &format!(
                "MATCH (u:User {{id:'{u}'}}), (t:Res {{id:'{t}'}}) WHERE EXISTS {{ MATCH \
                 (u)-[:MEMBER]->{arrow}(s)-[:OWNER|EDITOR|VIEWER]->(gr)-[:PARENT]->{arrow}(t) }} \
                 RETURN 1 AS x"
            ),
        )
        .is_empty()
    };
    // Known answers via the meet-in-the-middle path (`->*`).
    assert!(check(&mut g, "u1", "f1", "*")); // nested group grant, 0 PARENT hops
    assert!(check(&mut g, "u1", "d1", "*")); // + resource inheritance (1 PARENT hop)
    assert!(!check(&mut g, "u1", "d2", "*")); // ungranted — the NEGATIVE case
    assert!(!check(&mut g, "u1", "d3", "*")); // granted to u2, not u1
    assert!(check(&mut g, "u2", "d3", "*")); // DIRECT grant: 0 MEMBER hops (min-0)
                                             // Differential: `->*` (meet-in-the-middle) must equal `->{0,20}` (general matcher).
    for (u, t) in [
        ("u1", "f1"),
        ("u1", "d1"),
        ("u1", "d2"),
        ("u1", "d3"),
        ("u2", "d3"),
        ("u2", "f1"),
    ] {
        assert_eq!(
            check(&mut g, u, t, "*"),
            check(&mut g, u, t, "{0,20}"),
            "meet-in-the-middle diverged from general for {u} → {t}",
        );
    }
}

/// `neighborAggregate`'s `norm` (and `weightProperty`) must be accepted via GQL `CALL`,
/// not only via the direct-method API — the CALL path has its OWN config-key mapping.
/// Regression guard: `norm` shipped to the algorithm + `CONFIG_KEYS` but was initially
/// missing from the `CALL` mapper, so the documented GCN recipe faulted through GQL.
#[test]
fn call_neighbor_aggregate_accepts_norm_and_weight() {
    // a[1,2]↔b[3,4] via one edge; `both` + includeSelf + gcn → degree 2 each → coef 1/2 →
    // sum = ½([1,2]+[3,4]) = [2,3] for both.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a","h":[1.0,2.0]}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b","h":[3.0,4.0]}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{"w":2.0}}"#,
    ]);
    assert_eq!(
        rows(
            &mut g,
            "CALL neighbor_aggregate({feature:'h', op:'sum', direction:'both', includeSelf:true, norm:'gcn'}) \
             YIELD node, vector RETURN vector ORDER BY node",
        ),
        vec![
            vec![Value::List(vec![n(2.0), n(3.0)])],
            vec![Value::List(vec![n(2.0), n(3.0)])],
        ],
    );
    // `weightProperty` is likewise accepted through CALL (no fault).
    rows(
        &mut g,
        "CALL neighbor_aggregate({feature:'h', op:'mean', direction:'out', weightProperty:'w'}) \
         YIELD node RETURN count(*) AS c",
    );
}

#[test]
fn unknown_function_errors_even_over_empty_input_and_dead_branches() {
    // The fault is raised EAGERLY off the plan's `unknown_fns`, before the first
    // row — so an unknown function faults identically whether the result set is
    // empty or not, and even when the call sits in a never-taken branch. (A lazy
    // per-row fault would silently return `[]` over zero rows.) Matches the TS
    // engine's compile-time check.
    let mut g = modern();

    // Zero-row result: `MATCH (n) WHERE false` yields no rows, yet the unknown
    // function still faults.
    let err = parse("MATCH (n) WHERE false RETURN nope_fn(n) AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::UnknownFunction);
    assert!(err.message.contains("nope_fn()"), "got: {}", err.message);

    // A never-taken CASE branch: name resolution is reachability-independent.
    let err = parse("RETURN CASE WHEN false THEN bogus_fn(1) ELSE 1 END AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::UnknownFunction);
    assert!(err.message.contains("bogus_fn()"), "got: {}", err.message);
}

#[test]
fn unbound_param_errors_instead_of_silent_null() {
    let mut g = modern();
    // `$missing` is referenced but not supplied — a programming error, not a
    // silent empty result.
    let err = parse("MATCH (n) WHERE n.name = $missing RETURN n")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::MissingParameter);
}

#[test]
fn string_length_counts_utf16_units_like_js() {
    // Non-BMP chars are 2 UTF-16 units — matching JS `.length` (the TS engine),
    // not Unicode code points (which Rust's `chars().count()` gave before).
    let mut g = modern();
    assert_eq!(rows(&mut g, "RETURN size('😀') AS s"), vec![vec![n(2.0)]]);
    assert_eq!(
        rows(&mut g, "RETURN char_length('a😀b') AS s"),
        vec![vec![n(4.0)]]
    );
    // left/right slice on the same UTF-16 unit as JS `String.slice`.
    assert_eq!(
        rows(&mut g, "RETURN left('😀x', 2) AS s"),
        vec![vec![s("😀")]]
    );
    assert_eq!(
        rows(&mut g, "RETURN right('x😀', 2) AS s"),
        vec![vec![s("😀")]]
    );
}

#[test]
fn insert_rejects_ambiguous_label_and_typeless_edge() {
    // A non-conjunction node label (`|`/`!`/`%`) can't be created (which one?),
    // and an edge must carry exactly one type — both were silently accepted
    // (unlabelled node / empty-type edge) before.
    for q in [
        "INSERT (a:Foo|Bar)",
        "INSERT (a:!Foo)",
        "INSERT (a)-[r]->(b)",    // typeless edge
        "INSERT (a)-[:A|B]->(b)", // disjunction edge type
    ] {
        let mut g = modern();
        let err = parse(q)
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap_err();
        assert_eq!(
            err.code,
            crate::error_codes::ErrorCode::InvalidGraphOp,
            "should reject: {q}"
        );
    }
    // Sanity: conjunction, an unlabelled node, and a single-type edge all succeed.
    let mut g = modern();
    rows(&mut g, "INSERT (a:Foo&Bar)");
    rows(&mut g, "INSERT (a)"); // an unlabelled node is legitimate in GQL
    rows(&mut g, "INSERT (a:X)-[:REL]->(b:Y)");
}

#[test]
fn unique_constraint_enforced_on_insert_and_set() {
    // A UNIQUE constraint on (Acct, email): at most one live Acct per email. A
    // plain INSERT/SET that would duplicate faults with ConstraintViolation
    // (_MERGE, a later slice, reconciles instead). docs/design/gql-extensions.md §3.
    let mut g = modern(); // has no Acct/Other labels — a clean namespace.
    g.create_unique_constraint("Acct", "email").unwrap();

    rows(&mut g, "INSERT (:Acct {email: 'a@x.io', name: 'A'})");

    // Duplicate email under the same label → violation (no partial write: the
    // check precedes add_vertex).
    let err = parse("INSERT (:Acct {email: 'a@x.io', name: 'B'})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);

    // A different email is fine; a different label with the same email is fine
    // (the constraint is per-label).
    rows(&mut g, "INSERT (:Acct {email: 'b@x.io', name: 'B'})");
    rows(&mut g, "INSERT (:Other {email: 'a@x.io'})");

    // A SET that collides with an existing Acct email → violation …
    let err = parse("MATCH (n:Acct {email: 'b@x.io'}) SET n.email = 'a@x.io'")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
    // … but setting a row to its OWN current value is not a self-collision.
    rows(
        &mut g,
        "MATCH (n:Acct {email: 'b@x.io'}) SET n.email = 'b@x.io'",
    );
}

#[test]
fn wide_multi_clause_set_still_defers_constraint_checks() {
    // A K-clause `SET` pushes its vertex onto the touched list K times; the
    // commit-time recheck de-duplicates it (checking each touched element once
    // rather than once per clause) and skips the recheck entirely when no
    // constraint of a kind it checks is declared. Neither shortcut may change
    // observable behaviour — this pins that:
    //
    //  (a) with NO constraints, a wide multi-property SET commits and every value
    //      is readable (the skipped recheck loop didn't drop the write); and
    //  (b) with a UNIQUE constraint, the same wide SET whose LAST clause collides
    //      still faults (de-dup must not swallow the check).
    let mut g = modern();
    rows(&mut g, "INSERT (:Acct {email: 'a@x.io', name: 'A'})");
    rows(&mut g, "INSERT (:Acct {email: 'b@x.io', name: 'B'})");

    // (a) No constraints: set 10 fresh properties in one statement.
    let sets: Vec<String> = (0..10).map(|k| format!("n.f{k} = {k}")).collect();
    rows(
        &mut g,
        &format!("MATCH (n:Acct {{email: 'b@x.io'}}) SET {}", sets.join(", ")),
    );
    let vals = rows(&mut g, "MATCH (n:Acct {email: 'b@x.io'}) RETURN n.f0, n.f9");
    assert_eq!(vals, vec![vec![n(0.0), n(9.0)]]);

    // (b) Declare the constraint, then a wide SET whose final clause collides on
    // the constrained key must still surface ConstraintViolation.
    g.create_unique_constraint("Acct", "email").unwrap();
    let mut sets: Vec<String> = (0..10).map(|k| format!("n.g{k} = {k}")).collect();
    sets.push("n.email = 'a@x.io'".to_string());
    let err = parse(&format!(
        "MATCH (n:Acct {{email: 'b@x.io'}}) SET {}",
        sets.join(", ")
    ))
    .unwrap()
    .execute(&mut g, &Params::new())
    .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
}

#[test]
fn unique_constraint_null_values_are_exempt() {
    // SQL semantics: NULLs are distinct, so multiple null-emails don't collide
    // (lenke stores null first-class, but uniqueness still exempts it — matching
    // the value index, which never buckets null). An absent value is likewise ok.
    let mut g = modern();
    g.create_unique_constraint("Acct", "email").unwrap();
    rows(&mut g, "INSERT (:Acct {email: null, name: 'A'})");
    rows(&mut g, "INSERT (:Acct {email: null, name: 'B'})");
    rows(&mut g, "INSERT (:Acct {name: 'C'})");
}

#[test]
fn create_unique_constraint_rejects_preexisting_duplicates() {
    // Declaring a constraint the current data already violates is meaningless —
    // SQL rejects the unique-index build the same way.
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"1","labels":["Acct"],"properties":{"email":"dup@x.io"}}"#,
            r#"{"type":"node","id":"2","labels":["Acct"],"properties":{"email":"dup@x.io"}}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    let err = g.create_unique_constraint("Acct", "email").unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
}

#[test]
fn unique_constraint_introspection_and_drop() {
    let mut g = modern();
    g.create_unique_constraint("Acct", "email").unwrap();
    g.create_unique_constraint("Acct", "handle").unwrap();
    assert!(g.has_unique_constraint("Acct", "email"));
    assert_eq!(
        g.unique_keys("Acct"),
        &["email".to_string(), "handle".to_string()]
    );
    assert_eq!(
        g.unique_constraints(),
        vec![
            ("Acct".to_string(), "email".to_string()),
            ("Acct".to_string(), "handle".to_string()),
        ]
    );
    g.drop_unique_constraint("Acct", "email");
    assert!(!g.has_unique_constraint("Acct", "email"));
    assert!(g.has_unique_constraint("Acct", "handle"));
}

// --- edge-side constraints (edge types) -------------------------------------
// Direct mirror of the vertex constraint tests above, keyed by edge type and
// enforced against the edge property store. Byte-identical to the TS engine.

#[test]
fn edge_unique_constraint_enforced_on_insert_and_set() {
    let mut g = ndjson::decode("").unwrap();
    g.create_edge_unique_constraint("FOLLOWS", "tag").unwrap();

    rows(
        &mut g,
        "INSERT (:P {id: 'a'})-[:FOLLOWS {tag: 'x'}]->(:P {id: 'b'})",
    );

    // Duplicate tag on the same edge type → violation (whole statement rolls back).
    let err = parse("INSERT (:P {id: 'c'})-[:FOLLOWS {tag: 'x'}]->(:P {id: 'd'})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);

    // Different tag ok; a different edge type with the same tag ok (per-type).
    rows(
        &mut g,
        "INSERT (:P {id: 'e'})-[:FOLLOWS {tag: 'y'}]->(:P {id: 'f'})",
    );
    rows(
        &mut g,
        "INSERT (:P {id: 'g'})-[:LIKES {tag: 'x'}]->(:P {id: 'h'})",
    );

    // A SET that collides → violation …
    let err = parse("MATCH ()-[r:FOLLOWS {tag: 'y'}]->() SET r.tag = 'x'")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
    // … but re-setting an edge to its own value is not a self-collision.
    rows(
        &mut g,
        "MATCH ()-[r:FOLLOWS {tag: 'y'}]->() SET r.tag = 'y'",
    );
}

#[test]
fn edge_unique_constraint_null_values_are_exempt() {
    let mut g = ndjson::decode("").unwrap();
    g.create_edge_unique_constraint("FOLLOWS", "tag").unwrap();
    rows(
        &mut g,
        "INSERT (:P {id: 'a'})-[:FOLLOWS {tag: null}]->(:P {id: 'b'})",
    );
    rows(
        &mut g,
        "INSERT (:P {id: 'c'})-[:FOLLOWS {tag: null}]->(:P {id: 'd'})",
    );
    rows(&mut g, "INSERT (:P {id: 'e'})-[:FOLLOWS]->(:P {id: 'f'})");
}

#[test]
fn edge_required_constraint_enforced() {
    let mut g = ndjson::decode("").unwrap();
    g.create_edge_required_constraint("FOLLOWS", "since")
        .unwrap();

    rows(
        &mut g,
        "INSERT (:P {id: 'a'})-[:FOLLOWS {since: 1}]->(:P {id: 'b'})",
    );
    // Missing / null required → violation.
    let err = parse("INSERT (:P {id: 'c'})-[:FOLLOWS]->(:P {id: 'd'})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
    // Setting it to null, or removing it, → violation.
    let err = parse("MATCH ()-[r:FOLLOWS]->() SET r.since = null")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
    let err = parse("MATCH ()-[r:FOLLOWS]->() REMOVE r.since")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
}

#[test]
fn edge_type_constraint_enforced() {
    let mut g = ndjson::decode("").unwrap();
    g.create_edge_type_constraint("FOLLOWS", "since", "number")
        .unwrap();

    rows(
        &mut g,
        "INSERT (:P {id: 'a'})-[:FOLLOWS {since: 30}]->(:P {id: 'b'})",
    );
    // Wrong type → violation; null is exempt.
    let err = parse("INSERT (:P {id: 'c'})-[:FOLLOWS {since: 'old'}]->(:P {id: 'd'})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
    rows(
        &mut g,
        "INSERT (:P {id: 'e'})-[:FOLLOWS {since: null}]->(:P {id: 'f'})",
    );
    // A wrong-typed SET faults; unknown scalar type name is InvalidValue.
    let err = parse("MATCH ()-[r:FOLLOWS {since: 30}]->() SET r.since = 'nope'")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ConstraintViolation);
    assert_eq!(
        g.create_edge_type_constraint("FOLLOWS", "since", "int")
            .unwrap_err()
            .code,
        crate::error_codes::ErrorCode::InvalidValue
    );
}

#[test]
fn create_edge_constraint_rejects_preexisting_violations() {
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
            r#"{"type":"edge","from":"a","to":"b","labels":["FOLLOWS"],"properties":{"tag":"dup"}}"#,
            r#"{"type":"edge","from":"a","to":"b","labels":["FOLLOWS"],"properties":{"tag":"dup"}}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    assert_eq!(
        g.create_edge_unique_constraint("FOLLOWS", "tag")
            .unwrap_err()
            .code,
        crate::error_codes::ErrorCode::ConstraintViolation
    );
    assert_eq!(
        g.create_edge_required_constraint("FOLLOWS", "since")
            .unwrap_err()
            .code,
        crate::error_codes::ErrorCode::ConstraintViolation
    );
    assert_eq!(
        g.create_edge_type_constraint("FOLLOWS", "tag", "number")
            .unwrap_err()
            .code,
        crate::error_codes::ErrorCode::ConstraintViolation
    );
}

#[test]
fn edge_constraint_introspection_and_drop() {
    let mut g = ndjson::decode("").unwrap();
    g.create_edge_unique_constraint("FOLLOWS", "tag").unwrap();
    g.create_edge_required_constraint("FOLLOWS", "since")
        .unwrap();
    g.create_edge_type_constraint("FOLLOWS", "since", "number")
        .unwrap();
    assert!(g.has_edge_unique_constraint("FOLLOWS", "tag"));
    assert_eq!(g.edge_unique_keys("FOLLOWS"), &["tag".to_string()]);
    assert!(g.has_edge_required_constraint("FOLLOWS", "since"));
    assert_eq!(
        g.edge_type_constraint("FOLLOWS", "since"),
        Some(crate::graph::PropType::Num)
    );
    assert_eq!(
        g.edge_unique_constraints(),
        vec![("FOLLOWS".to_string(), "tag".to_string())]
    );
    g.drop_edge_unique_constraint("FOLLOWS", "tag");
    assert!(!g.has_edge_unique_constraint("FOLLOWS", "tag"));
}

#[test]
fn edge_constraint_deferred_within_transaction() {
    // An intermediate edge violation that resolves before commit is fine; one
    // left unresolved rolls the whole transaction back.
    let mut g = ndjson::decode("").unwrap();
    g.create_edge_required_constraint("FOLLOWS", "since")
        .unwrap();

    // Resolved-before-commit: insert an edge missing `since`, then supply it.
    g.begin_tx();
    rows(&mut g, "INSERT (:P {id: 'a'})-[:FOLLOWS]->(:P {id: 'b'})");
    rows(&mut g, "MATCH ()-[r:FOLLOWS]->() SET r.since = 2020");
    assert!(matches!(g.commit_tx(), Ok(())));
    assert_eq!(g.edge_count(), 1, "the resolved edge committed");

    // Unresolved: the missing required key survives to commit → rollback.
    g.begin_tx();
    rows(&mut g, "INSERT (:P {id: 'c'})-[:FOLLOWS]->(:P {id: 'd'})");
    assert!(matches!(
        g.commit_tx(),
        Err(crate::graph::TxCommitError::Required)
    ));
    assert_eq!(g.edge_count(), 1, "the unresolved edge rolled back");
}

// `_MERGE` keyed upsert (node form). Mirrors the TS `merge.test.ts` so the two
// engines stay byte-identical. See docs/design/gql-extensions.md §2.

/// A string `id` property in an INSERT becomes the element's external identity —
/// so `element_id(n)` equals it and `toNdjson` round-trips by domain identity
/// instead of a synthetic `_n{k}`, while `id` is still a stored property. A numeric
/// `id` stays an ordinary (SET-able) property; a duplicate string id, or a SET on a
/// string-identity id, is rejected. Byte-identical to the TS engine.
#[test]
fn string_id_property_is_the_element_identity() {
    let mut g = ndjson::decode("").unwrap();
    rows(&mut g, "INSERT (:P {id: 'alice', name: 'A'})");

    // element_id == the domain id, and `id` is still a readable property.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:P {id: 'alice'}) RETURN element_id(n) AS e, n.id AS p"
        ),
        vec![vec![s("alice"), s("alice")]]
    );

    // The whole point: toNdjson emits the domain id as the top-level id (not a
    // synthetic `_n{k}`), so an independently-built graph round-trips by identity.
    let dumped = ndjson::encode(&g);
    assert!(dumped.contains(r#""id":"alice""#));
    assert!(!dumped.contains("_n0"));
    let mut reloaded = ndjson::decode(&dumped).unwrap();
    assert_eq!(
        rows(
            &mut reloaded,
            "MATCH (n:P {id: 'alice'}) RETURN element_id(n) AS e"
        ),
        vec![vec![s("alice")]]
    );

    // A duplicate string id is rejected (ids are unique).
    let dup = parse("INSERT (:P {id: 'alice'})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(dup.code, crate::error_codes::ErrorCode::ConstraintViolation);

    // SET on the string-identity id is rejected — identity is fixed at creation.
    let set_id = parse("MATCH (n:P {id: 'alice'}) SET n.id = 'bob'")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(set_id.code, crate::error_codes::ErrorCode::InvalidGraphOp);

    // A numeric id is an ordinary property: not the identity, and SET-able.
    rows(&mut g, "INSERT (:Q {id: 7})");
    assert_ne!(
        rows(&mut g, "MATCH (n:Q {id: 7}) RETURN element_id(n) AS e"),
        vec![vec![s("7")]] // external id is synthetic, not "7"
    );
    rows(&mut g, "MATCH (n:Q {id: 7}) SET n.id = 8"); // allowed
    assert_eq!(
        rows(&mut g, "MATCH (n:Q) RETURN n.id AS i"),
        vec![vec![n(8.0)]]
    );
}

/// The edge analogue: a string `id` on an INSERT edge is its identity —
/// `element_id(r)` equals it, it's unique among edges, and SET on it is rejected.
/// A numeric edge id is an ordinary, SET-able property. Byte-identical to TS.
#[test]
fn string_id_property_is_the_edge_identity() {
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"id":"a"}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"id":"b"}}"#,
            r#"{"type":"node","id":"c","labels":["P"],"properties":{"id":"c"}}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    rows(
        &mut g,
        "MATCH (a:P {id: 'a'}), (b:P {id: 'b'}) INSERT (a)-[:R {id: 'e1', w: 5}]->(b)",
    );

    // element_id(r) == the domain id (not the synthetic `e{index}`), `id` readable.
    assert_eq!(
        rows(
            &mut g,
            "MATCH ()-[r:R]->() RETURN element_id(r) AS e, r.id AS p"
        ),
        vec![vec![s("e1"), s("e1")]]
    );
    // toNdjson uses the domain edge id.
    let dumped = ndjson::encode(&g);
    assert!(dumped.contains(r#""type":"edge","id":"e1""#));

    // A duplicate edge id is rejected (edge ids are unique).
    let dup = parse("MATCH (a:P {id: 'a'}), (c:P {id: 'c'}) INSERT (a)-[:R {id: 'e1'}]->(c)")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(dup.code, crate::error_codes::ErrorCode::ConstraintViolation);

    // SET on the string-identity edge id is rejected.
    let set_id = parse("MATCH ()-[r:R {id: 'e1'}]->() SET r.id = 'e2'")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(set_id.code, crate::error_codes::ErrorCode::InvalidGraphOp);

    // A numeric edge id is an ordinary, SET-able property.
    rows(
        &mut g,
        "MATCH (b:P {id: 'b'}), (c:P {id: 'c'}) INSERT (b)-[:R {id: 99}]->(c)",
    );
    rows(&mut g, "MATCH ()-[r:R {id: 99}]->() SET r.id = 100"); // allowed
}

/// Edge `_MERGE` with endpoints bound by a preceding MATCH — `MATCH (a), (b)
/// _MERGE (a)-[:R]->(b)`, the natural way to upsert an edge between two known
/// vertices. Regression: `resolve_merge_endpoint` ignored the binding and re-
/// inferred a unique key from the (empty) node pattern, so every bound-variable
/// edge merge failed with `_MERGE could not determine a unique key` — the whole
/// keyed-edge-upsert path was unreachable. Surfaced when replaying edges as
/// upserts during a merge.
#[test]
fn merge_edge_between_bound_endpoints_upserts() {
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["A"],"properties":{"id":"a"}}"#,
            r#"{"type":"node","id":"b","labels":["A"],"properties":{"id":"b"}}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    g.create_unique_constraint("A", "id").unwrap();

    let merge = "MATCH (a:A {id: 'a'}), (b:A {id: 'b'}) \
                 _MERGE (a)-[r:R]->(b) _ON_CREATE SET r.n = 1 _ON_UPDATE SET r.n = r.n + 100 \
                 RETURN r.n AS n";
    assert_eq!(rows(&mut g, merge), vec![vec![n(1.0)]]); // created
    assert_eq!(rows(&mut g, merge), vec![vec![n(101.0)]]); // updated, not duplicated
    assert_eq!(
        rows(&mut g, "MATCH (:A)-[r:R]->(:A) RETURN count(r) AS c"),
        vec![vec![n(1.0)]] // exactly one edge
    );
}

#[test]
fn merge_create_path_runs_on_create() {
    let mut g = modern();
    g.create_unique_constraint("Acct", "email").unwrap();
    rows(
        &mut g,
        "_MERGE (u:Acct {email: 'a@x.io', name: 'A'}) _ON_CREATE SET u.created = 1",
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name, u.created"
        ),
        vec![vec![s("A"), n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (u:Acct) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
}

#[test]
fn merge_update_default_clobbers_payload_keeps_one_node() {
    let mut g = modern();
    g.create_unique_constraint("Acct", "email").unwrap();
    rows(
        &mut g,
        "_MERGE (u:Acct {email: 'a@x.io', name: 'A'}) _ON_CREATE SET u.created = 1",
    );
    // Present → clobber payload (name); created stays (birth-only); one node.
    rows(&mut g, "_MERGE (u:Acct {email: 'a@x.io', name: 'A2'})");
    assert_eq!(
        rows(
            &mut g,
            "MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name, u.created"
        ),
        vec![vec![s("A2"), n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (u:Acct) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
}

#[test]
fn merge_on_update_set_replaces_default_clobber() {
    let mut g = modern();
    g.create_unique_constraint("Acct", "email").unwrap();
    rows(&mut g, "_MERGE (u:Acct {email: 'a@x.io', name: 'A'})");
    // Pattern payload 'IGNORED' is NOT written — _ON_UPDATE replaces the default.
    rows(
        &mut g,
        "_MERGE (u:Acct {email: 'a@x.io', name: 'IGNORED'}) _ON_UPDATE SET u.name = 'FromUpdate'",
    );
    assert_eq!(
        rows(&mut g, "MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name"),
        vec![vec![s("FromUpdate")]]
    );
}

#[test]
fn merge_on_update_nothing_leaves_untouched() {
    let mut g = modern();
    g.create_unique_constraint("Acct", "email").unwrap();
    rows(&mut g, "_MERGE (u:Acct {email: 'a@x.io', name: 'A'})");
    rows(
        &mut g,
        "_MERGE (u:Acct {email: 'a@x.io', name: 'IGNORED'}) _ON_UPDATE_NOTHING",
    );
    assert_eq!(
        rows(&mut g, "MATCH (u:Acct {email: 'a@x.io'}) RETURN u.name"),
        vec![vec![s("A")]]
    );
}

#[test]
fn merge_where_gated_update_is_last_write_wins() {
    let mut g = modern();
    g.create_unique_constraint("Doc", "id").unwrap();
    rows(&mut g, "_MERGE (d:Doc {id: 1, v: 1, body: 'first'})");
    // Incoming v (5) newer than stored (1) → applies.
    rows(
        &mut g,
        "_MERGE (d:Doc {id: 1}) _ON_UPDATE SET d.v = 5, d.body = 'newer' WHERE d.v < 5",
    );
    assert_eq!(
        rows(&mut g, "MATCH (d:Doc {id: 1}) RETURN d.v, d.body"),
        vec![vec![n(5.0), s("newer")]]
    );
    // Stored (5) not < 3 → predicate false → no-op.
    rows(
        &mut g,
        "_MERGE (d:Doc {id: 1}) _ON_UPDATE SET d.v = 3, d.body = 'older' WHERE d.v < 3",
    );
    assert_eq!(
        rows(&mut g, "MATCH (d:Doc {id: 1}) RETURN d.v, d.body"),
        vec![vec![n(5.0), s("newer")]]
    );
}

#[test]
fn merge_presence_idiom_clobbers() {
    let mut g = modern();
    g.create_unique_constraint("Presence", "sid").unwrap();
    rows(&mut g, "_MERGE (p:Presence {sid: 's1', x: 0, y: 0})");
    rows(&mut g, "_MERGE (p:Presence {sid: 's1', x: 10, y: 20})");
    assert_eq!(
        rows(&mut g, "MATCH (p:Presence) RETURN p.x, p.y"),
        vec![vec![n(10.0), n(20.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (p:Presence) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
}

#[test]
fn merge_without_constraint_errors() {
    let mut g = modern();
    let err = parse("_MERGE (x:Nope {k: 1})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::InvalidGraphOp);
}

#[test]
fn merge_conflicting_dispositions_is_parse_error() {
    assert!(
        parse("_MERGE (u:Acct {email: 'a'}) _ON_UPDATE SET u.n = 1 _ON_UPDATE_NOTHING").is_err()
    );
}

#[test]
fn merge_gated_off_under_iso_strict() {
    use super::ast::Dialect;
    use super::parser::parse_with_dialect;
    // Under iso-strict, `_MERGE` is a plain identifier → no clause → syntax error.
    assert!(parse_with_dialect("_MERGE (u:Acct {email: 'a'})", Dialect::IsoStrict).is_err());
    // …but it parses fine under the default (lenke) dialect.
    assert!(parse_with_dialect("_MERGE (u:Acct {email: 'a'})", Dialect::Lenke).is_ok());
}

#[test]
fn merge_edge_form_upserts_edge_between_matched_endpoints() {
    let mut g = modern();
    g.create_unique_constraint("User", "id").unwrap();
    g.create_unique_constraint("Team", "id").unwrap();
    rows(&mut g, "INSERT (:User {id: 'u1'}), (:Team {id: 'g1'})");

    // ensure-tuple: endpoints matched by key, the MEMBER edge is upserted.
    rows(
        &mut g,
        "_MERGE (u:User {id: 'u1'})-[m:MEMBER {since: 1}]->(g:Team {id: 'g1'}) _ON_CREATE SET m.role = 'admin'",
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (:User {id:'u1'})-[m:MEMBER]->(:Team {id:'g1'}) RETURN m.since, m.role"
        ),
        vec![vec![n(1.0), s("admin")]]
    );

    // Idempotent: second _MERGE clobbers edge props (default), no duplicate edge;
    // _ON_CREATE does not re-run, so role stays.
    rows(
        &mut g,
        "_MERGE (u:User {id: 'u1'})-[m:MEMBER {since: 2}]->(g:Team {id: 'g1'})",
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (:User {id:'u1'})-[m:MEMBER]->(:Team {id:'g1'}) RETURN m.since, m.role"
        ),
        vec![vec![n(2.0), s("admin")]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (:User)-[m:MEMBER]->(:Team) RETURN count(*) AS c"
        ),
        vec![vec![n(1.0)]]
    );

    // _ON_UPDATE_NOTHING leaves the edge untouched.
    rows(
        &mut g,
        "_MERGE (u:User {id: 'u1'})-[m:MEMBER {since: 99}]->(g:Team {id: 'g1'}) _ON_UPDATE_NOTHING",
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (:User {id:'u1'})-[m:MEMBER]->(:Team {id:'g1'}) RETURN m.since"
        ),
        vec![vec![n(2.0)]]
    );
}

#[test]
fn merge_edge_missing_endpoint_errors() {
    let mut g = modern();
    g.create_unique_constraint("User", "id").unwrap();
    g.create_unique_constraint("Team", "id").unwrap();
    rows(&mut g, "INSERT (:User {id: 'u1'})"); // no Team t1

    let err = parse("_MERGE (u:User {id:'u1'})-[m:MEMBER]->(g:Team {id:'g1'})")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::InvalidGraphOp);
}

#[test]
fn iso_strict_parses_iso_surface_rejects_extensions() {
    use super::ast::Dialect;
    use super::parser::parse_with_dialect;
    // The whole ISO surface parses under iso-strict (self-contained; no extension
    // leaked in).
    for q in [
        "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.age > 30 RETURN b.name",
        "INSERT (:Person {name: 'x', age: 1})",
        "MATCH (n:Person) SET n.age = 2",
        "MATCH (n:Person) REMOVE n.age",
        "MATCH (n:Person) DETACH DELETE n",
        "MATCH (n) RETURN count(*) AS c ORDER BY c DESC LIMIT 5",
    ] {
        assert!(
            parse_with_dialect(q, Dialect::IsoStrict).is_ok(),
            "should parse: {q}"
        );
    }
    // Every extension construct is a syntax error under iso-strict.
    for ext in [
        "_MERGE (u:Acct {email: 'a'})",
        "_MERGE (u:Acct {email: 'a'}) _ON_CREATE SET u.x = 1",
    ] {
        assert!(
            parse_with_dialect(ext, Dialect::IsoStrict).is_err(),
            "should reject: {ext}"
        );
    }
}

#[test]
fn delete_vertex_with_edges_errors_without_detach() {
    let mut g = modern();
    let err = parse("MATCH (n:Person {name:'marko'}) DELETE n")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    // The code is the contract; the message is just the human hint.
    assert_eq!(err.code, crate::error_codes::ErrorCode::InvalidGraphOp);
    assert!(err.message.contains("DETACH"), "got: {err}");
}

#[test]
fn missing_label_empty_with_columns() {
    let mut g = modern();
    let (cols, r) = q(&mut g, "MATCH (n:Ghost) RETURN n.name AS who");
    assert_eq!(cols, vec!["who"]);
    assert!(r.is_empty());
}

// --- property-index seeding (indexed result must equal the scan result) ---

#[test]
fn index_eq_inline_matches_scan() {
    let scan = {
        let mut g = modern();
        rows(&mut g, "MATCH (n:Person {name:'marko'}) RETURN n.age")
    };
    let idx = {
        let mut g = modern();
        g.create_vertex_index("name");
        rows(&mut g, "MATCH (n:Person {name:'marko'}) RETURN n.age")
    };
    assert_eq!(scan, idx);
    assert_eq!(idx, vec![vec![n(29.0)]]);
}

#[test]
fn index_where_eq_matches_scan() {
    let mut g = modern();
    g.create_vertex_index("name");
    assert_eq!(
        rows(&mut g, "MATCH (n) WHERE n.name = 'marko' RETURN n.age"),
        vec![vec![n(29.0)]]
    );
}

#[test]
fn index_where_range_matches_scan() {
    let mut g = modern();
    g.create_vertex_index("age");
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:Person) WHERE n.age > 30 RETURN n.name ORDER BY n.name"
        ),
        vec![vec![s("josh")], vec![s("peter")]]
    );
}

#[test]
fn index_where_and_range_matches_scan() {
    let mut g = modern();
    g.create_vertex_index("age");
    // AND of two comparisons on the indexed key — first conjunct seeds, WHERE re-filters.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:Person) WHERE n.age > 28 AND n.age < 33 RETURN n.name ORDER BY n.name"
        ),
        vec![vec![s("josh")], vec![s("marko")]]
    );
}

#[test]
fn index_range_does_not_bleed_into_software() {
    let mut g = modern();
    g.create_vertex_index("age");
    // age > 0 must not surface software (no age) — type-block bounded seed.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n) WHERE n.age > 0 RETURN n.name ORDER BY n.name"
        )
        .len(),
        4
    );
}

#[test]
fn index_live_under_gql_insert() {
    let mut g = modern();
    g.create_vertex_index("name");
    rows(&mut g, "INSERT (z:Person {name:'zoe', age:50})");
    // The new vertex is found via the (maintained) index seed.
    assert_eq!(
        rows(&mut g, "MATCH (n) WHERE n.name = 'zoe' RETURN n.age"),
        vec![vec![n(50.0)]]
    );
}

#[test]
fn index_live_under_gql_set() {
    let mut g = modern();
    g.create_vertex_index("name");
    rows(
        &mut g,
        "MATCH (n:Person) WHERE n.name = 'marko' SET n.name = 'mark'",
    );
    assert!(rows(&mut g, "MATCH (n) WHERE n.name = 'marko' RETURN n.name").is_empty());
    assert_eq!(
        rows(&mut g, "MATCH (n) WHERE n.name = 'mark' RETURN n.age"),
        vec![vec![n(29.0)]]
    );
}

// --- edge property index seeding (edge-first single-segment build) ---

#[test]
fn edge_index_where_eq() {
    let scan = {
        let mut g = modern();
        rows(
            &mut g,
            "MATCH (a)-[r:CREATED]->(s) WHERE r.weight = 1.0 RETURN s.name",
        )
    };
    let idx = {
        let mut g = modern();
        g.create_edge_index("weight");
        rows(
            &mut g,
            "MATCH (a)-[r:CREATED]->(s) WHERE r.weight = 1.0 RETURN s.name",
        )
    };
    assert_eq!(scan, idx);
    assert_eq!(idx, vec![vec![s("ripple")]]); // josh -created(1.0)-> ripple
}

#[test]
fn edge_index_inline_prop() {
    let mut g = modern();
    g.create_edge_index("weight");
    // inline edge prop drives the seek; label CREATED narrows the weight-1.0 edges.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:CREATED {weight:1.0}]->(s) RETURN s.name"
        ),
        vec![vec![s("ripple")]]
    );
}

#[test]
fn edge_index_range() {
    let mut g = modern();
    g.create_edge_index("weight");
    // CREATED weights {0.4,1.0,0.4,0.2}; >= 0.5 ⇒ only ripple's edge.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:CREATED]->(s) WHERE r.weight >= 0.5 RETURN s.name ORDER BY s.name"
        ),
        vec![vec![s("ripple")]]
    );
}

#[test]
fn edge_index_knows_eq() {
    let mut g = modern();
    g.create_edge_index("weight");
    // KNOWS weight 1.0 ⇒ marko -knows-> josh.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:KNOWS]->(b) WHERE r.weight = 1.0 RETURN b.name"
        ),
        vec![vec![s("josh")]]
    );
}

#[test]
fn edge_index_live_under_set() {
    let mut g = modern();
    g.create_edge_index("weight");
    // bump every CREATED edge to weight 2.0, then seek 2.0.
    rows(&mut g, "MATCH ()-[r:CREATED]->() SET r.weight = 2.0");
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:CREATED]->(s) WHERE r.weight = 2.0 RETURN s.name ORDER BY s.name"
        ),
        vec![
            vec![s("lop")],
            vec![s("lop")],
            vec![s("lop")],
            vec![s("ripple")]
        ]
    );
    // and 1.0 now finds nothing among CREATED (josh->ripple moved to 2.0).
    assert!(rows(
        &mut g,
        "MATCH (a)-[r:CREATED]->(s) WHERE r.weight = 1.0 RETURN s.name"
    )
    .is_empty());
}

// --- edge TYPE index seeding (always-on `by_etype`; `()-[:T]->()` patterns) ---

#[test]
fn edge_type_seed_single() {
    // marko -knows-> vadas, marko -knows-> josh. The type bucket seeds these two
    // edges directly instead of expanding every vertex's adjacency.
    let mut g = modern();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:KNOWS]->(b) RETURN b.name ORDER BY b.name"
        ),
        vec![vec![s("josh")], vec![s("vadas")]],
    );
}

#[test]
fn edge_type_seed_disjunction() {
    // `:KNOWS|CREATED` unions two type buckets (disjoint — an edge has one type).
    // KNOWS: 2 edges, CREATED: 4 edges ⇒ 6 rows.
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a)-[r:KNOWS|CREATED]->(b) RETURN count(*) AS c",
    );
    assert_eq!(r, vec![vec![n(6.0)]]);
}

#[test]
fn edge_type_seed_absent_is_empty() {
    // A type that was never interned seeds an empty candidate set (no scan).
    let mut g = modern();
    assert!(rows(&mut g, "MATCH (a)-[r:NONEXISTENT]->(b) RETURN b.name").is_empty());
}

#[test]
fn edge_type_seed_with_endpoint_filter() {
    // Type seed is a superset; edge_first_build re-validates the endpoint WHERE.
    // Of marko's two KNOWS targets, only josh is 32.
    let mut g = modern();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:KNOWS]->(b) WHERE b.age = 32 RETURN b.name"
        ),
        vec![vec![s("josh")]],
    );
}

#[test]
fn edge_type_seed_live_under_insert() {
    // A KNOWS edge created at runtime must land in the type bucket and be found.
    let mut g = modern();
    rows(&mut g, "MATCH (a:Person), (b:Person) WHERE a.name = 'peter' AND b.name = 'vadas' INSERT (a)-[:KNOWS]->(b)");
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:KNOWS]->(b) RETURN a.name ORDER BY a.name"
        ),
        vec![vec![s("marko")], vec![s("marko")], vec![s("peter")]],
    );
}

#[test]
fn edge_type_seed_live_under_delete() {
    // Deleting an edge must purge it from the type bucket, so the seed shrinks.
    let mut g = modern();
    rows(
        &mut g,
        "MATCH (a)-[r:KNOWS]->(b) WHERE b.name = 'vadas' DELETE r",
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[r:KNOWS]->(b) RETURN b.name ORDER BY b.name"
        ),
        vec![vec![s("josh")]],
    );
}

// --- reactive change tracking (version + per-token epochs) ---

#[test]
fn reactive_version_and_epoch() {
    let mut g = modern();
    let v0 = g.version();
    let person0 = g.epoch("Person");
    let age0 = g.epoch("age");
    let name0 = g.epoch("name");

    // A read does not bump anything.
    rows(&mut g, "MATCH (n:Person) RETURN n.name");
    assert_eq!(g.version(), v0);
    assert_eq!(g.epoch("Person"), person0);

    // Inserting a Person bumps the global version and the touched tokens.
    rows(&mut g, "INSERT (:Person {name: 'zoe', age: 99})");
    assert!(g.version() > v0);
    assert!(g.epoch("Person") > person0);
    assert!(g.epoch("age") > age0);
    assert!(g.epoch("name") > name0);

    // A property write bumps that key's epoch but NOT the label's (finer
    // invalidation: a label-only/topology query is not disturbed).
    let v1 = g.version();
    let person1 = g.epoch("Person");
    let age1 = g.epoch("age");
    let name1 = g.epoch("name");
    rows(
        &mut g,
        "MATCH (n:Person) WHERE n.name = 'marko' SET n.age = 30",
    );
    assert!(g.version() > v1);
    assert!(g.epoch("age") > age1);
    assert_eq!(g.epoch("Person"), person1); // label untouched by a value write
    assert_eq!(g.epoch("name"), name1); // unrelated key untouched
}

// --- hardening: parser/lexer robustness (ports of the TS hardening.test.ts) ---

#[test]
fn deep_nesting_errors_instead_of_stack_overflow() {
    // Each of these would overflow the native stack (an uncatchable abort)
    // without the recursion-depth guard; they must return a parse error.
    let parens = format!("RETURN {}1{} AS r", "(".repeat(5000), ")".repeat(5000));
    assert!(parse(&parens).is_err());

    let nots = format!("MATCH (n) WHERE {}n.x RETURN n", "NOT ".repeat(5000));
    assert!(parse(&nots).is_err());

    let bangs = format!("MATCH (n:{}A) RETURN n", "!".repeat(5000));
    assert!(parse(&bangs).is_err());

    let lists = format!("RETURN {}1{} AS r", "[".repeat(5000), "]".repeat(5000));
    assert!(parse(&lists).is_err());
}

#[test]
fn normally_nested_query_still_parses() {
    assert!(parse("RETURN (((1 + 2)) * 3) AS r").is_ok());
}

#[test]
fn malformed_numeric_literals_rejected() {
    for bad in [
        "0x", "0b", "0o", "0b2", "0o8", "0o9", "1e", "1e+", "0xG", "1e999",
    ] {
        assert!(
            parse(&format!("RETURN {bad} AS r")).is_err(),
            "expected a lex error for `{bad}`"
        );
    }
}

#[test]
fn oversized_integer_literal_rejected() {
    // Beyond 2^53 an integer literal loses precision as an f64.
    assert!(parse("RETURN 99999999999999999999 AS r").is_err());
}

#[test]
fn valid_numeric_literals_still_parse_and_eval() {
    let mut g = modern();
    assert_eq!(rows(&mut g, "RETURN 0xFF AS r"), vec![vec![n(255.0)]]);
    assert_eq!(rows(&mut g, "RETURN 0o17 AS r"), vec![vec![n(15.0)]]);
    assert_eq!(rows(&mut g, "RETURN 0b101 AS r"), vec![vec![n(5.0)]]);
    assert_eq!(rows(&mut g, "RETURN 1_000 AS r"), vec![vec![n(1000.0)]]);
    assert_eq!(rows(&mut g, "RETURN 1.5e2 AS r"), vec![vec![n(150.0)]]);
}

#[test]
fn skip_limit_reject_non_integers() {
    assert!(parse("MATCH (n) RETURN n LIMIT 2.5").is_err());
    assert!(parse("MATCH (n) RETURN n SKIP 1.5").is_err());
    assert!(parse("MATCH (n) RETURN n LIMIT 0.5").is_err());
}

#[test]
fn quantifier_rejects_fractional_and_reversed_bounds() {
    assert!(parse("MATCH (a)-[:R]->{1.5}(b) RETURN b").is_err());
    assert!(parse("MATCH (a)-[:R]->{3,2}(b) RETURN b").is_err());
}

#[test]
fn skip_limit_quantifier_valid_forms_still_parse() {
    assert!(parse("MATCH (n) RETURN n SKIP 1 LIMIT 2").is_ok());
    assert!(parse("MATCH (a)-[:R]->{1,3}(b) RETURN b").is_ok());
    assert!(parse("MATCH (a)-[:R]->{2}(b) RETURN b").is_ok());
}

#[test]
fn var_length_accepts_per_hop_predicate() {
    // A quantified segment may now carry a per-hop edge predicate (applied to every
    // edge of the walk) and name each hop's edge for it. See
    // `per_hop_predicate_filters_var_length_edges` for the execution semantics.
    assert!(parse("MATCH (a)-[r:KNOWS]->*(b) RETURN b").is_ok());
    assert!(parse("MATCH (a)-[:KNOWS {weight:1}]->+(b) RETURN b").is_ok());
    assert!(parse("MATCH (a)-[:KNOWS WHERE true]->+(b) RETURN b").is_ok());
    assert!(parse("MATCH (a)-[e:KNOWS WHERE e.weight > 0.5]->{1,5}(b) RETURN b").is_ok());
    // …including together with a shortest selector: the BFS now expands only over
    // predicate-passing edges (shortest path in the filtered subgraph).
    assert!(parse("MATCH ANY SHORTEST (a)-[e:R WHERE e.w > 1]->*(b) RETURN b").is_ok());
    assert!(parse("MATCH p = ALL SHORTEST (a)-[:R {w:1}]->*(b) RETURN p").is_ok());
}

#[test]
fn var_length_label_only_still_parses() {
    assert!(parse("MATCH (a:Person {name:'marko'})-[:KNOWS]->+(b) RETURN b.name").is_ok());
}

/// ANY/ALL SHORTEST now honour a per-hop edge predicate: the BFS expands only over
/// predicate-passing edges, so it finds the shortest path in the FILTERED subgraph.
/// (Sound because a per-hop predicate is element-local — not path-dependent — so the
/// filtered graph is well-defined and BFS's discover-once invariant still holds.)
#[test]
fn shortest_honours_per_hop_edge_predicate() {
    // a→b (w=1), a→c (w=10), c→b (w=10). Unfiltered shortest a→b is 1 hop; with
    // `WHERE e.w > 5` the direct edge is blocked, so the shortest is a→c→b (2 hops).
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"w":1.0}}"#,
        r#"{"type":"edge","id":"e2","from":"a","to":"c","labels":["R"],"properties":{"w":10.0}}"#,
        r#"{"type":"edge","id":"e3","from":"c","to":"b","labels":["R"],"properties":{"w":10.0}}"#,
    ]);
    let len = |g: &mut Graph, q: &str| rows(g, q)[0][0].clone();
    // Unfiltered: 1 hop.
    assert_eq!(
        len(
            &mut g,
            "MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R]->*(b:N {id:'b'}) RETURN path_length(p)",
        ),
        n(1.0),
    );
    // Filtered (w > 5): the direct edge is gone → 2 hops.
    assert_eq!(
        len(
            &mut g,
            "MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 5]->*(b:N {id:'b'}) RETURN path_length(p)",
        ),
        n(2.0),
    );
    assert_eq!(
        len(
            &mut g,
            "MATCH p = ALL SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 5]->*(b:N {id:'b'}) RETURN path_length(p)",
        ),
        n(2.0),
    );
    // A predicate that blocks EVERY edge out of the seed → no path.
    assert!(rows(
        &mut g,
        "MATCH ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 100]->*(b:N {id:'b'}) RETURN b.id AS id",
    )
    .is_empty());
}

#[test]
fn undirected_self_loop_counted_once() {
    let lines = [
        r#"{"type":"node","id":"n","labels":["N"],"properties":{"name":"n"}}"#,
        r#"{"type":"edge","from":"n","to":"n","labels":["LOOP"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    // Before the fix an undirected walk yielded the self-loop twice (once from
    // the out-index, once from the in-index).
    assert_eq!(
        rows(&mut g, "MATCH (a)~[r]~(b) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
    // Directed walks each see it exactly once.
    assert_eq!(
        rows(&mut g, "MATCH (a)-[r]->(b) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (a)<-[r]-(b) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
}

// --- ISO medium-conformance batch (mirrors TS hardening.test.ts) ------------

#[test]
fn ordering_across_incomparable_types_is_unknown() {
    let mut g = modern();
    // number vs string has no defined order → UNKNOWN (null), not a coerced bool.
    assert_eq!(
        rows(&mut g, "RETURN (1 < 'a') AS r"),
        vec![vec![Value::Null]]
    );
    assert_eq!(
        rows(&mut g, "RETURN ('a' >= 1) AS r"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn equality_across_types_is_false_not_null() {
    let mut g = modern();
    assert_eq!(rows(&mut g, "RETURN (5 = '5') AS r"), vec![vec![b(false)]]);
    assert_eq!(rows(&mut g, "RETURN (5 <> '5') AS r"), vec![vec![b(true)]]);
}

#[test]
fn same_type_ordering_including_booleans_still_works() {
    let mut g = modern();
    assert_eq!(rows(&mut g, "RETURN (1 < 2) AS r"), vec![vec![b(true)]]);
    assert_eq!(rows(&mut g, "RETURN ('a' < 'b') AS r"), vec![vec![b(true)]]);
    assert_eq!(
        rows(&mut g, "RETURN (false >= false) AS r"),
        vec![vec![b(true)]]
    );
}

#[test]
fn nested_aggregates_rejected() {
    let mut g = modern();
    let err = parse("MATCH (n:Person) RETURN sum(avg(n.age))")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::Unsupported);
}

#[test]
fn plain_aggregate_still_works() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN sum(n.age) AS s"),
        vec![vec![n(123.0)]]
    );
}

#[test]
fn division_by_zero_raises_data_exception() {
    let mut g = modern();
    for q in ["RETURN 1 / 0 AS r", "RETURN 5 % 0 AS r"] {
        let err = parse(q)
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap_err();
        assert_eq!(
            err.code,
            crate::error_codes::ErrorCode::DataException,
            "{q}"
        );
    }
}

#[test]
fn division_by_zero_raises_over_rows_vectorized() {
    let mut g = modern();
    // MATCH … RETURN n.age / 0 takes the vectorized path; the divisor scan must
    // surface the data exception (via scalar fallback).
    let err = parse("MATCH (n:Person) RETURN n.age / 0 AS r")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::DataException);
}

#[test]
fn null_dividend_over_zero_is_null_not_a_fault_vectorized() {
    // A null dividend short-circuits to null BEFORE the divide-by-zero check, so
    // `null / 0` (and `% 0`) over MATCH rows is null — NOT a data exception. The
    // vectorized fault scan once omitted the dividend-validity check and faulted
    // these, diverging from the scalar path and the TS engine (found by the
    // differential fuzzer). A real numeric dividend over zero must still fault.
    let mut g = modern();
    for q in [
        "MATCH (n:Person) RETURN null / 0 AS r",
        "MATCH (n:Person) RETURN null % 0 AS r",
        // n.age is present, but the null literal still short-circuits the row.
        "MATCH (n:Person) RETURN (null / (n.age - n.age)) AS r",
    ] {
        let out = rows(&mut g, q);
        assert!(!out.is_empty(), "{q}");
        assert!(
            out.iter().all(|row| row == &vec![Value::Null]),
            "{q}: expected all-null rows, got {out:?}"
        );
    }
}

#[test]
fn sum_mixing_number_and_duration_faults_not_drops() {
    // `sum()` over a mix of a plain number and a DURATION is an unsummable type mix —
    // a loud E_DATA_EXCEPTION, matching TS. The streaming accumulator kept separate
    // numeric (`sum`) and duration (`tsum`) totals and, at finish, returned just the
    // duration — silently dropping the number (found by the differential fuzzer).
    let mut g = modern();
    let tdur = |s: &str| Value::Temporal(crate::temporal::Temporal::parse("duration", s).unwrap());
    for q in [
        "MATCH (n:Person) RETURN sum(CASE WHEN n.age < 30 THEN 1 ELSE duration('P1M') END) AS r",
        // DISTINCT takes the same accumulator; the mix must still fault.
        "MATCH (n:Person) RETURN sum(DISTINCT CASE WHEN n.age < 30 THEN 1 ELSE duration('P1M') END) AS r",
        // a non-numeric scalar mixed with a duration faults too.
        "MATCH (n:Person) RETURN sum(CASE WHEN n.age < 30 THEN 'x' ELSE duration('P1M') END) AS r",
    ] {
        let err = parse(q)
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap_err();
        assert_eq!(err.code, crate::error_codes::ErrorCode::DataException, "{q}");
    }
    // The unmixed paths are unaffected: pure durations sum (4 people × P1M = P4M),
    // pure numbers sum (29+27+32+35).
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN sum(duration('P1M')) AS r"),
        vec![vec![tdur("P4M")]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN sum(n.age) AS r"),
        vec![vec![n(123.0)]]
    );
}

#[test]
fn date_arithmetic_overflow_raises_data_exception() {
    // A date/datetime shifted past the representable range (Date is i32 days,
    // ≈±5.88M years) is a loud data exception, not a silent null — same policy as
    // duration overflow and division by zero (supersedes the old D4 → null).
    let mut g = modern();
    for q in [
        "RETURN DATE '2020-01-01' + DURATION 'P10000000Y' AS d",
        "RETURN DATE '2020-01-01' - DURATION 'P10000000Y' AS d",
        "RETURN DATETIME '2020-01-01T00:00:00' + DURATION 'P10000000Y' AS d",
    ] {
        let err = parse(q)
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap_err();
        assert_eq!(
            err.code,
            crate::error_codes::ErrorCode::DataException,
            "{q}"
        );
    }
    // An in-range shift (~5M years) still succeeds.
    assert_eq!(
        rows(
            &mut g,
            "RETURN DATE '2020-01-01' + DURATION 'P5000000Y' AS d"
        ),
        vec![vec![tdate("5002020-01-01")]]
    );
}

#[test]
fn vectorized_temporal_filter_matches_canonical_compare() {
    // The typed vectorized temporal comparator (temporal_cmp_vec) must produce the
    // exact same counts as the canonical compare_vals: relational order on dates,
    // reversed operands, Eq, cross-kind → UNKNOWN (0 rows), duration unordered
    // (0 rows), and an absent value → UNKNOWN. Isolated-node scans take the
    // vectorized path.
    let lines = [
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"d":{"@date":"2020-01-01"}}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"d":{"@date":"2020-06-01"}}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"d":{"@date":"2020-12-01"}}}"#,
        // an absent `d` — the compare is UNKNOWN, so the row is excluded
        r#"{"type":"node","id":"e","labels":["P"],"properties":{"name":"x"}}"#,
        r#"{"type":"node","id":"f","labels":["P"],"properties":{"dur":{"@duration":"P30D"}}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let count = |g: &mut Graph, q: &str| -> f64 {
        match rows(g, q)[0][0] {
            Value::Num(n) => n,
            ref v => panic!("expected count, got {v:?}"),
        }
    };
    // relational order; reversed operand order; boundary (>= includes equal).
    assert_eq!(
        count(
            &mut g,
            "MATCH (n:P) WHERE n.d > DATE '2020-06-01' RETURN count(*) AS c"
        ),
        1.0
    );
    assert_eq!(
        count(
            &mut g,
            "MATCH (n:P) WHERE n.d >= DATE '2020-06-01' RETURN count(*) AS c"
        ),
        2.0
    );
    assert_eq!(
        count(
            &mut g,
            "MATCH (n:P) WHERE DATE '2020-06-01' < n.d RETURN count(*) AS c"
        ),
        1.0
    );
    assert_eq!(
        count(
            &mut g,
            "MATCH (n:P) WHERE n.d = DATE '2020-06-01' RETURN count(*) AS c"
        ),
        1.0
    );
    // cross-kind date vs datetime → UNKNOWN → 0 rows (not a coerced compare).
    assert_eq!(
        count(
            &mut g,
            "MATCH (n:P) WHERE n.d > DATETIME '2020-06-01T00:00:00' RETURN count(*) AS c"
        ),
        0.0
    );
    // `<>` must NOT count the absent-value row: `null <> x` is UNKNOWN, not true
    // (the bug the scalar-vs-vectorized differential caught). Only a,c differ; the
    // node with no `d` is excluded.
    assert_eq!(
        count(
            &mut g,
            "MATCH (n:P) WHERE n.d <> DATE '2020-06-01' RETURN count(*) AS c"
        ),
        2.0
    );
    // durations are relationally unordered → every compare is UNKNOWN → 0 rows.
    assert_eq!(
        count(
            &mut g,
            "MATCH (n:P) WHERE n.dur > DURATION 'P1D' RETURN count(*) AS c"
        ),
        0.0
    );
    // typed min/max fold (fused_global_agg → temporal_minmax): canonical total
    // order, absent values skipped.
    assert_eq!(
        rows(&mut g, "MATCH (n:P) RETURN min(n.d) AS m")[0][0],
        tdate("2020-01-01")
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:P) RETURN max(n.d) AS m")[0][0],
        tdate("2020-12-01")
    );
}

#[test]
fn non_numeric_arithmetic_raises_data_exception() {
    let mut g = modern();
    for q in ["RETURN 'abc' + 1 AS r", "RETURN true * 2 AS r"] {
        let err = parse(q)
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap_err();
        assert_eq!(
            err.code,
            crate::error_codes::ErrorCode::DataException,
            "{q}"
        );
    }
}

#[test]
fn non_numeric_arithmetic_raises_in_vectorized_path() {
    let mut g = modern();
    // n.name is a string column → arithmetic over it falls back to scalar eval,
    // which raises the type error.
    let err = parse("MATCH (n:Person) RETURN n.name + 1 AS r")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::DataException);
}

#[test]
fn null_arithmetic_still_propagates_to_null() {
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

// --- variable-length trail semantics ----------------------------------------

/// Build a graph from (id-label) nodes and (from,to) R-edges.
fn ring_graph() -> Graph {
    // a → b → c → a
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

#[test]
fn trail_excludes_repeated_relationship() {
    let mut g = modern();
    // From josh, undirected KNOWS reaches marko (1). The 2-hop step back to josh
    // would reuse the marko–josh edge, which a trail forbids — so josh is not
    // reached (Gremlin's walk semantics would include it).
    let r = rows(
        &mut g,
        "MATCH (a:Person {name:'josh'})-[:KNOWS]-{1,2}(b) RETURN b.name ORDER BY b.name",
    );
    assert_eq!(r, vec![vec![s("marko")], vec![s("vadas")]]);
}

#[test]
fn trail_cycle_terminates_one_row_per_trail() {
    let mut g = ring_graph();
    // From a the trails of ≥1 hop are a→b, a→b→c, a→b→c→a; the next step reuses
    // a→b, so it stops. Three trails.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {name:'a'})-[:R]->+(x) RETURN count(*) AS c"
        ),
        vec![vec![n(3.0)]]
    );
}

#[test]
fn trail_endpoint_appears_once_per_trail() {
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
    // d is reached by two distinct 2-hop trails: a→b→d and a→c→d.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {name:'a'})-[:R]->{2,2}(d) RETURN count(*) AS c"
        ),
        vec![vec![n(2.0)]]
    );
}

#[test]
fn trail_budget_guards_dense_unbounded_star() {
    let mut lines: Vec<String> = Vec::new();
    for i in 0..8 {
        lines.push(format!(
            r#"{{"type":"node","id":"{i}","labels":["N"],"properties":{{}}}}"#
        ));
    }
    for i in 0..8 {
        for j in 0..8 {
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
    assert_eq!(err.code, crate::error_codes::ErrorCode::ResourceExhausted);
}

/// A dense multi-element unit still enumerates `d^k` traversals from a vertex, but the
/// fused matcher walks them LAZILY (O(path) memory, no `d^k` buffer), so the concern is
/// now runaway TIME, not memory. The per-seed `TRAIL_BUDGET` (charged per hop) must
/// still fault with `E_RESOURCE_EXHAUSTED` — even for a single repetition `{1}`.
#[test]
fn unit_expansion_budget_guards_dense_multi_element() {
    // Complete digraph K_64: every vertex → every other (4032 edges). A single 4-hop
    // unit expansion from one seed fans out ~63⁴ ≈ 15M traversals.
    let n = 64;
    let mut lines: Vec<String> = Vec::new();
    for i in 0..n {
        lines.push(format!(
            r#"{{"type":"node","id":"{i}","labels":["N"],"properties":{{}}}}"#
        ));
    }
    for i in 0..n {
        for j in 0..n {
            if i != j {
                lines.push(format!(
                    r#"{{"type":"edge","from":"{i}","to":"{j}","labels":["R"],"properties":{{}}}}"#
                ));
            }
        }
    }
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    // `{1}` — exactly one repetition, so the whole fan-out is ONE `expand_unit` call:
    // proof the guard fires inside a single expansion, not just across repetitions.
    let err = parse(
        "MATCH (s:N) ((a)-[:R]->(b)-[:R]->(c)-[:R]->(d)-[:R]->(e)){1} (t) RETURN count(*) AS c",
    )
    .unwrap()
    .execute(&mut g, &Params::new())
    .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ResourceExhausted);
}

/// A fixed-length multi-hop pattern with a per-hop WHERE and a LIMIT must be
/// answered by the scalar depth-first driver — which filters during traversal and
/// stops the instant the LIMIT fills — NOT the breadth-first vectorized path,
/// which materializes the full cross-product of partial matches (millions of rows
/// on a dense graph, an OOM on a large one) before the LIMIT ever applies. This is
/// the routing that keeps native's result identical to the TS engine's and keeps
/// the host alive. The `INTERMEDIATE_BUDGET` fault in `expand_scan` is the
/// separate backstop for the enumerate-all case (no LIMIT, plain projection),
/// which must materialize the full cross-product to return it: past the ceiling it
/// faults with `E_RESOURCE_EXHAUSTED` rather than OOM-killing the host. That path
/// is impractical to unit-test cheaply — tripping the ceiling means materializing
/// tens of millions of rows — so this test covers the routing that makes the
/// common case both correct and bounded. Regression: a correlated multi-hop join
/// took the host down with an OOM kill.
#[test]
fn multi_hop_with_limit_streams_and_stays_correct() {
    // A small DAG: a -> {b,c} -> {d,e} -> f, with rising `amt` on exactly one
    // fully-increasing path (a-b-d-f: 1 < 3 < 6). The per-hop `<` filter selects
    // that path and no other, so the answer is deterministic and independent of
    // enumeration order.
    let lines = [
        r#"{"type":"node","id":"a","labels":["A"],"properties":{"nm":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["A"],"properties":{"nm":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["A"],"properties":{"nm":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["A"],"properties":{"nm":"d"}}"#,
        r#"{"type":"node","id":"e","labels":["A"],"properties":{"nm":"e"}}"#,
        r#"{"type":"node","id":"f","labels":["A"],"properties":{"nm":"f"}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["E"],"properties":{"amt":1}}"#,
        r#"{"type":"edge","from":"b","to":"d","labels":["E"],"properties":{"amt":3}}"#,
        r#"{"type":"edge","from":"d","to":"f","labels":["E"],"properties":{"amt":6}}"#,
        // a decoy path whose amounts do not strictly increase (2, 1, 9)
        r#"{"type":"edge","from":"a","to":"c","labels":["E"],"properties":{"amt":2}}"#,
        r#"{"type":"edge","from":"c","to":"e","labels":["E"],"properties":{"amt":1}}"#,
        r#"{"type":"edge","from":"e","to":"f","labels":["E"],"properties":{"amt":9}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();
    let out = rows(
        &mut g,
        "MATCH (v0:A)-[e1:E]->(v1:A)-[e2:E]->(v2:A)-[e3:E]->(v3:A) \
         WHERE e1.amt < e2.amt AND e2.amt < e3.amt \
         RETURN v0.nm AS s, v3.nm AS t LIMIT 100",
    );
    // Exactly the increasing path a->b->d->f, and nothing from the decoy.
    assert_eq!(out, vec![vec![s("a"), s("f")]]);
}

#[test]
fn list_value_equality_is_structural() {
    // Lists compare by size then element-wise (ISO); the TS engine matches this
    // (it previously used reference identity — a byte-identical violation).
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN [1, 2] = [1, 2] AS x"),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN [1, 2] = [1, 3] AS x"),
        vec![vec![b(false)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN [1, 2] = [1, 2, 3] AS x"),
        vec![vec![b(false)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN [[1], [2]] = [[1], [2]] AS x"),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN [1, 2] <> [1, 3] AS x"),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN [1] IN [[1], [2]] AS x"),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN [3] IN [[1], [2]] AS x"),
        vec![vec![b(false)]]
    );
}

// --- FOR (ISO GQL list unwind / UNWIND) -------------------------------------

#[test]
fn for_unwinds_a_literal_list() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "FOR x IN [1, 2, 3] RETURN x"),
        vec![vec![n(1.0)], vec![n(2.0)], vec![n(3.0)]]
    );
}

#[test]
fn for_ordinality_counts_from_one() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "FOR x IN ['a', 'b'] WITH ORDINALITY i RETURN x, i"),
        vec![vec![s("a"), n(1.0)], vec![s("b"), n(2.0)]]
    );
}

#[test]
fn for_offset_counts_from_zero() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "FOR x IN ['a', 'b'] WITH OFFSET i RETURN x, i"),
        vec![vec![s("a"), n(0.0)], vec![s("b"), n(1.0)]]
    );
}

#[test]
fn for_over_null_yields_no_rows() {
    let mut g = modern();
    assert!(rows(&mut g, "FOR x IN null RETURN x").is_empty());
}

#[test]
fn for_over_empty_list_yields_no_rows() {
    let mut g = modern();
    assert!(rows(&mut g, "FOR x IN [] RETURN x").is_empty());
}

#[test]
fn for_over_scalar_unwinds_as_singleton() {
    let mut g = modern();
    assert_eq!(rows(&mut g, "FOR x IN 5 RETURN x"), vec![vec![n(5.0)]]);
}

#[test]
fn for_multiplies_prior_match_rows() {
    let mut g = modern();
    // One matched row × a two-element list → two rows.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (p:Person {name: 'marko'}) FOR t IN ['x', 'y'] RETURN p.name, t"
        ),
        vec![vec![s("marko"), s("x")], vec![s("marko"), s("y")]]
    );
}

#[test]
fn for_list_can_reference_a_bound_var() {
    let mut g = modern();
    // The list expression sees the pending MATCH binding (`p`).
    assert_eq!(
        rows(
            &mut g,
            "MATCH (p:Person {name: 'marko'}) FOR x IN [p.name, p.age] RETURN x"
        ),
        vec![vec![s("marko")], vec![n(29.0)]]
    );
}

#[test]
fn for_bare_with_after_for_starts_a_new_clause() {
    // `WITH x AS y` is NOT a FOR modifier (no ORDINALITY/OFFSET) — it must be
    // parsed as a WITH clause, so the lookahead disambiguation matters.
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "FOR x IN [1, 2] WITH x AS y RETURN y"),
        vec![vec![n(1.0)], vec![n(2.0)]]
    );
}

#[test]
fn for_first_clause_needs_no_seed_row() {
    // FOR as the very first clause runs against the single empty seed binding.
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "FOR x IN ['only'] RETURN x"),
        vec![vec![s("only")]]
    );
}

#[test]
fn for_drives_batch_optional_match_allow_and_deny() {
    // `FOR` clause deny-side: one row per requested name, present or not. `josh`
    // exists (age 32); `nobody` does not, so OPTIONAL MATCH leaves `p` null.
    let mut g = modern();
    assert_eq!(
        rows(
            &mut g,
            "FOR name IN ['josh', 'nobody'] OPTIONAL MATCH (p:Person {name: name}) RETURN name, p.age"
        ),
        vec![vec![s("josh"), n(32.0)], vec![s("nobody"), Value::Null]]
    );
}

// --- temporal literals + comparison (Phase 1) -------------------------------

fn tdate(s: &str) -> Value {
    Value::Temporal(crate::temporal::Temporal::parse("date", s).unwrap())
}

#[test]
fn temporal_date_literal_returns_value() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN DATE '2020-02-29' AS d"),
        vec![vec![tdate("2020-02-29")]]
    );
}

#[test]
fn temporal_literals_compare_chronologically() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN DATE '2020-01-01' < DATE '2020-06-01' AS x"),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN DATE '2020-06-01' < DATE '2020-01-01' AS x"),
        vec![vec![b(false)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN DATE '2020-01-01' = DATE '2020-01-01' AS x"),
        vec![vec![b(true)]]
    );
    // TIMESTAMP is a DATETIME synonym; fractional seconds parse.
    assert_eq!(
        rows(
            &mut g,
            "RETURN TIMESTAMP '2021-06-15T08:30:00.5' >= DATETIME '2021-06-15T08:30:00' AS x"
        ),
        vec![vec![b(true)]]
    );
}

#[test]
fn temporal_cross_kind_comparison_is_unknown() {
    // date vs datetime relationally → UNKNOWN (null), like a cross-type compare.
    let mut g = modern();
    assert_eq!(
        rows(
            &mut g,
            "RETURN DATE '2020-01-01' < DATETIME '2020-01-01T00:00:00' AS x"
        ),
        vec![vec![Value::Null]]
    );
}

#[test]
fn temporal_as_of_where_filter() {
    // Valid-time modeling: keep the fact whose [vfrom, vto) contains the as-of date.
    let doc = concat!(
        r#"{"type":"node","id":"1","labels":["Fact"],"properties":{"name":"a","vfrom":{"@date":"2020-01-01"},"vto":{"@date":"2021-01-01"}}}"#,
        "\n",
        r#"{"type":"node","id":"2","labels":["Fact"],"properties":{"name":"b","vfrom":{"@date":"2021-01-01"},"vto":{"@date":"2022-01-01"}}}"#,
    );
    let mut g = crate::ndjson::decode(doc).unwrap();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (f:Fact) WHERE f.vfrom <= DATE '2020-06-01' AND DATE '2020-06-01' < f.vto RETURN f.name"
        ),
        vec![vec![s("a")]]
    );
}

#[test]
fn temporal_order_by_sorts_chronologically() {
    let mut g = modern();
    assert_eq!(
        rows(
            &mut g,
            "FOR d IN [DATE '2020-06-01', DATE '2020-01-01', DATE '2020-03-01'] RETURN d ORDER BY d"
        ),
        vec![
            vec![tdate("2020-01-01")],
            vec![tdate("2020-03-01")],
            vec![tdate("2020-06-01")]
        ]
    );
}

#[test]
fn temporal_bad_literal_is_a_syntax_error() {
    assert!(parse("RETURN DATE '2020-99-99'").is_err());
}

// --- temporal constructor functions (Phase 2 slice 1) -----------------------

#[test]
fn temporal_constructors_parse_strings() {
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN date('2020-02-29') AS d"),
        vec![vec![tdate("2020-02-29")]]
    );
    // local_datetime + duration return their kinds (checked via re-serialized form).
    let dt = rows(&mut g, "RETURN local_datetime('2021-06-15T08:30:00') AS d");
    assert_eq!(dt.len(), 1);
    let du = rows(&mut g, "RETURN duration('P1Y2M') AS d");
    assert_eq!(du.len(), 1);
    // A bad string is lenient → null (like to_integer).
    assert_eq!(
        rows(&mut g, "RETURN date('nope') AS d"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn temporal_constructors_convert_between_kinds() {
    let mut g = modern();
    // date(datetime) truncates to the date part.
    assert_eq!(
        rows(
            &mut g,
            "RETURN date(local_datetime('2020-02-29T13:45:00')) AS d"
        ),
        vec![vec![tdate("2020-02-29")]]
    );
    // local_datetime(date) is midnight; comparing to the explicit midnight literal.
    assert_eq!(
        rows(
            &mut g,
            "RETURN local_datetime(date('2020-02-29')) = DATETIME '2020-02-29T00:00:00' AS x"
        ),
        vec![vec![b(true)]]
    );
    // duration(date) has no sensible conversion → null.
    assert_eq!(
        rows(&mut g, "RETURN duration(date('2020-01-01')) AS d"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn temporal_constructor_converts_a_string_property() {
    // The point of the function form (vs the literal): convert loaded string data.
    let doc = r#"{"type":"node","id":"1","labels":["E"],"properties":{"hired":"2019-03-15"}}"#;
    let mut g = crate::ndjson::decode(doc).unwrap();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:E) RETURN date(n.hired) < DATE '2020-01-01' AS x"
        ),
        vec![vec![b(true)]]
    );
}

#[test]
fn temporal_duration_between_is_exact() {
    let mut g = modern();
    // Two dates → whole days.
    assert_eq!(
        rows(
            &mut g,
            "RETURN duration_between(DATE '2020-01-15', DATE '2020-04-20') AS d"
        )
        .into_iter()
        .flatten()
        .map(|v| match v {
            Value::Temporal(t) => t.format(),
            _ => "?".into(),
        })
        .collect::<Vec<_>>(),
        vec!["P96D"]
    );
    // Two datetimes → seconds.
    assert_eq!(
        rows(&mut g, "RETURN duration_between(DATETIME '2020-01-01T00:00:00', DATETIME '2020-01-01T01:01:01') AS d")
            .into_iter().flatten().map(|v| match v { Value::Temporal(t) => t.format(), _ => "?".into() }).collect::<Vec<_>>(),
        vec!["PT3661S"]
    );
    // Cross-kind → null.
    assert_eq!(
        rows(
            &mut g,
            "RETURN duration_between(DATE '2020-01-01', DATETIME '2020-01-01T00:00:00') AS d"
        ),
        vec![vec![Value::Null]]
    );
}

#[test]
fn temporal_arithmetic() {
    let mut g = modern();
    let fmt = |r: Vec<Vec<Value>>| -> String {
        match r.into_iter().flatten().next() {
            Some(Value::Temporal(t)) => t.format(),
            other => format!("{other:?}"),
        }
    };
    // month-add clamps to the new month's length (2020 is a leap year → Feb 29).
    assert_eq!(
        fmt(rows(
            &mut g,
            "RETURN DATE '2020-01-31' + DURATION 'P1M' AS d"
        )),
        "2020-02-29"
    );
    assert_eq!(
        fmt(rows(
            &mut g,
            "RETURN DATE '2021-01-31' + DURATION 'P1M' AS d"
        )),
        "2021-02-28"
    );
    assert_eq!(
        fmt(rows(
            &mut g,
            "RETURN DATE '2020-01-15' + DURATION 'P2M3D' AS d"
        )),
        "2020-03-18"
    );
    // datetime + time-duration.
    assert_eq!(
        fmt(rows(
            &mut g,
            "RETURN DATETIME '2020-01-01T10:00:00' + DURATION 'PT1H30M' AS d"
        )),
        "2020-01-01T11:30:00"
    );
    // subtraction is the inverse of addition.
    assert_eq!(
        fmt(rows(
            &mut g,
            "RETURN DATE '2020-03-18' - DURATION 'P2M3D' AS d"
        )),
        "2020-01-15"
    );
    // instant − instant → the exact span.
    assert_eq!(
        fmt(rows(
            &mut g,
            "RETURN DATE '2020-04-20' - DATE '2020-01-15' AS d"
        )),
        "P96D"
    );
    // duration ± duration (component-wise) and duration × integer.
    assert_eq!(
        fmt(rows(&mut g, "RETURN DURATION 'P1M' + DURATION 'P2D' AS d")),
        "P1M2D"
    );
    assert_eq!(
        fmt(rows(&mut g, "RETURN DURATION 'P1M2DT3S' * 3 AS d")),
        "P3M6DT9S"
    );
    // add then subtract the same duration round-trips (clamp is only on the add side).
    assert_eq!(
        fmt(rows(
            &mut g,
            "RETURN DATE '2020-01-15' + DURATION 'P1M' - DURATION 'P1M' AS d"
        )),
        "2020-01-15"
    );
}

#[test]
fn temporal_now_functions_read_injected_now() {
    let mut g = modern();
    let now =
        crate::gql::params_from_json(r#"{"__now":{"@datetime":"2026-07-12T10:30:45"}}"#).unwrap();
    // current_timestamp / local_timestamp → the injected datetime.
    assert_eq!(
        qp(&mut g, "RETURN current_timestamp AS t", now.clone()),
        vec![vec![Value::Temporal(
            crate::temporal::Temporal::parse("datetime", "2026-07-12T10:30:45").unwrap()
        )]]
    );
    assert_eq!(
        qp(&mut g, "RETURN local_timestamp AS t", now.clone()),
        vec![vec![Value::Temporal(
            crate::temporal::Temporal::parse("datetime", "2026-07-12T10:30:45").unwrap()
        )]]
    );
    // current_date → the date part (truncated).
    assert_eq!(
        qp(&mut g, "RETURN current_date AS d", now.clone()),
        vec![vec![tdate("2026-07-12")]]
    );
    // empty-parens form also parses.
    assert_eq!(
        qp(&mut g, "RETURN current_date() AS d", now),
        vec![vec![tdate("2026-07-12")]]
    );
    // without an injected now → null (the engine never reads a clock).
    assert_eq!(
        qp(
            &mut g,
            "RETURN current_date AS d",
            crate::gql::eval::Params::new()
        ),
        vec![vec![Value::Null]]
    );
    // a general temporal param also binds now.
    let p = crate::gql::params_from_json(r#"{"d":{"@date":"2020-02-29"}}"#).unwrap();
    assert_eq!(
        qp(&mut g, "RETURN $d AS d", p),
        vec![vec![tdate("2020-02-29")]]
    );
}

// --- ISO transaction keywords (START TRANSACTION / COMMIT / ROLLBACK) ---------
// Byte-identical with the TS `transaction-keywords.test.ts` and the cross-engine
// differential (packages/native/src/transaction-conformance.test.ts).

/// The sorted `id`s of every `:Acct` vertex — a compact view of committed state.
fn acct_ids(g: &mut Graph) -> Vec<String> {
    rows(g, "MATCH (n:Acct) RETURN n.id AS id ORDER BY n.id")
        .into_iter()
        .map(|r| match &r[0] {
            Value::Str(s) => s.to_string(),
            other => panic!("expected a string id, got {other:?}"),
        })
        .collect()
}

/// Execute a statement expecting a coded execution error; return its code.
fn tx_err(g: &mut Graph, query: &str) -> crate::error_codes::ErrorCode {
    parse(query)
        .unwrap_or_else(|e| panic!("parse error for `{query}`: {e}"))
        .execute(g, &Params::new())
        .unwrap_err()
        .code
}

#[test]
fn tx_keywords_start_insert_commit_persists() {
    let mut g = ndjson::decode("").unwrap();
    assert!(q(&mut g, "START TRANSACTION").1.is_empty());
    assert!(g.in_transaction());
    q(&mut g, "INSERT (:Acct {id: 'a'})");
    q(&mut g, "INSERT (:Acct {id: 'b'})");
    // Read-your-writes inside the transaction.
    assert_eq!(acct_ids(&mut g), vec!["a", "b"]);
    q(&mut g, "COMMIT");
    assert!(!g.in_transaction());
    assert_eq!(acct_ids(&mut g), vec!["a", "b"]);
}

/// The seekable endpoint may be the TARGET, not the source: `(e)-[:T]->(m {id:$m})`
/// must cost the same at any graph size. Regression: `try_orient_node_seed` bailed
/// whenever either endpoint had a real index seek ("don't interfere with a real
/// index seek"), so a target-anchored pattern seeded from the *unindexed* source
/// and scanned its whole label bucket — 26x at N=32k while the source-anchored
/// form was already flat. It now orients toward the seekable end, reversing the
/// path when that end is last. Regression: a target-anchored "direct reports of"
/// traversal scanned the whole label bucket.
#[test]
fn traversal_anchored_on_the_target_is_independent_of_graph_size() {
    use std::time::Instant;

    /// Repetitions to take the minimum over. This is a SCALING assertion, so the
    /// bound was never the problem — the SAMPLE was: one timing of ~1.5ms
    /// compared against a 6x ratio is below this repo's noise floor. All three of
    /// these passed 60 consecutive runs on an idle box and failed 5 of 12 with it
    /// loaded. The minimum over reps is the closest thing to an interference-free
    /// run, which is the number the assertion actually wants.
    const REPS: usize = 9;

    let mut g = ndjson::decode("").unwrap();
    g.create_vertex_index("id");
    q(&mut g, "INSERT (:Emp {id: 'BOSS'})");
    q(&mut g, "INSERT (:Emp {id: 'SOLO'})");
    q(&mut g, "INSERT (:Emp {id: 'ONE'})");
    q(
        &mut g,
        "MATCH (a:Emp {id: 'ONE'}), (b:Emp {id: 'SOLO'}) INSERT (a)-[:REPORTS_TO]->(b)",
    );

    let grow_to = |g: &mut Graph, upto: usize, from: usize| {
        for i in from..upto {
            q(g, &format!("INSERT (:Emp {{id: 'e{i}'}})"));
            q(
                g,
                &format!(
                    "MATCH (a:Emp {{id: 'e{i}'}}), (b:Emp {{id: 'BOSS'}}) INSERT (a)-[:REPORTS_TO]->(b)"
                ),
            );
        }
    };
    // Exactly one edge ever points at SOLO, whatever the graph size.
    let probe = "MATCH (e:Emp)-[r:REPORTS_TO]->(m:Emp {id: 'SOLO'}) RETURN e.id AS x";
    let time = |g: &mut Graph| {
        for _ in 0..20 {
            q(g, probe);
        }
        let mut best = f64::INFINITY;

        for _ in 0..REPS {
            let t = Instant::now();

            for _ in 0..200 {
                q(g, probe);
            }

            best = best.min(t.elapsed().as_secs_f64());
        }

        best
    };

    grow_to(&mut g, 2_000, 0);
    let small = time(&mut g);
    grow_to(&mut g, 32_000, 2_000);
    let large = time(&mut g);

    let ratio = large / small.max(f64::MIN_POSITIVE);
    assert!(
        ratio < 6.0,
        "target-anchored traversal scaled with graph size: 16x more vertices cost \
         {ratio:.1}x more time ({small:.4}s -> {large:.4}s, min of {REPS}). The plan \
         is seeding the unindexed source instead of orienting toward the seekable \
         target."
    );
}

/// Interleaving writes with traversals must not cost more as the graph grows.
/// Regression (the second, independent defect): every adjacency read went through
/// `csr()`'s `get_or_init`, and every topology mutation dropped the snapshot — so
/// the first read after each write repacked the WHOLE graph, O(V+E) per read, and
/// an ingest that reads as it writes went quadratic. Warm read-only scans hid it
/// completely, which is why the read-only benches never saw it.
///
/// `Graph::adj` now serves a single vertex from the `out`/`in_` delta and only
/// repacks once `CSR_WARM_READS` reads accumulate. Same 16x vertex spread, but
/// construction as the sibling test below. Regression: an ingest that reads as it
/// writes repacked the whole CSR on the first read after each write.
#[test]
fn interleaved_write_and_traverse_is_independent_of_graph_size() {
    use std::time::Instant;

    /// Repetitions to take the minimum over. Enough that a loaded box is very
    /// unlikely to interfere with EVERY rep; cheap because each is ~1ms.
    const REPS: usize = 9;

    let mut g = ndjson::decode("").unwrap();
    g.create_vertex_index("id");
    q(&mut g, "INSERT (:Dept {id: 'D0'})");

    let grow_to = |g: &mut Graph, upto: usize, from: usize| {
        for i in from..upto {
            q(g, &format!("INSERT (:Emp {{id: 'e{i}'}})"));
            q(
                g,
                &format!(
                    "MATCH (s:Emp {{id: 'e{i}'}}), (t:Dept {{id: 'D0'}}) INSERT (s)-[:IN_DEPT]->(t)"
                ),
            );
        }
    };

    // One write immediately followed by one traversal — the shape that repacked.
    let cycle = |g: &mut Graph, i: usize| {
        q(g, &format!("MATCH (s:Emp {{id: 'e0'}}) SET s.w = {i}"));
        q(
            g,
            "MATCH (s:Emp {id: 'e0'})-[r:IN_DEPT]->(t) RETURN t.id AS x",
        );
    };
    // MIN of several reps, not one sample. This is a scaling assertion, so the
    // bound was never the problem — the SAMPLE was: one timing of ~1ms compared
    // against a 6x ratio is below the noise floor this repo already warns about.
    // It passed 60 consecutive runs on an idle box and failed 2 of 8 with the box
    // loaded (7.4x, 9.4x). The minimum over reps is the closest thing to an
    // interference-free run, which is the number this assertion actually wants.
    //
    // Counting CSR repacks instead was tried and is INERT: the cycle's write is a
    // PROPERTY write, which does not invalidate the snapshot, so the count is 0
    // or 1 whatever the traversal does — it stayed 1 under a mutation that made
    // every read repack, and 0 under one that invalidated on every write.
    let time = |g: &mut Graph| {
        for i in 0..20 {
            cycle(g, i);
        }

        let mut best = f64::INFINITY;

        for _ in 0..REPS {
            let t = Instant::now();

            for i in 0..100 {
                cycle(g, i);
            }

            best = best.min(t.elapsed().as_secs_f64());
        }

        best
    };

    grow_to(&mut g, 2_000, 0);
    let small = time(&mut g);
    grow_to(&mut g, 32_000, 2_000);
    let large = time(&mut g);

    let ratio = large / small.max(f64::MIN_POSITIVE);
    assert!(
        ratio < 6.0,
        "interleaved write+traverse scaled with graph size: 16x more vertices cost \
         {ratio:.1}x more time ({small:.4}s -> {large:.4}s, min of {REPS}). A read \
         after a write is rebuilding the whole CSR snapshot instead of reading the \
         adjacency delta."
    );
}

/// A traversal from an index-anchored, degree-1 source must cost the same whether
/// the traversed edge type holds 2k edges or 32k. Regression: `build_scan` fell
/// through to `edge_index_seed`, whose `by_etype` fallback materializes EVERY edge
/// of the type — and `try_orient_node_seed` deliberately bails on an indexed
/// endpoint, so having a usable index actively *diverted* the plan into that scan.
/// A one-edge lookup cost O(edges of type); measured 193x at N=32k.
///
/// Timing-based, but the spread is 16x of graph size against a 6x bound: unfixed
/// this ratio is ~16, fixed it is ~1. The cost must track edge-type population,
/// not graph size.
#[test]
fn traversal_from_indexed_anchor_is_independent_of_edge_type_size() {
    use std::time::Instant;

    /// See the note on `REPS` in `traversal_anchored_on_the_target_…`: one ~1.5ms
    /// sample against a 6x ratio is below the noise floor, so take the minimum
    /// over reps instead of widening the bound.
    const REPS: usize = 9;

    // One probe vertex with exactly ONE outgoing edge, whatever else exists.
    let mut g = ndjson::decode("").unwrap();
    g.create_vertex_index("id");
    q(&mut g, "INSERT (:Dept {id: 'D0'})");
    q(&mut g, "INSERT (:Probe {id: 'PR'})");
    q(
        &mut g,
        "MATCH (s:Probe {id: 'PR'}), (t:Dept {id: 'D0'}) INSERT (s)-[:IN_DEPT]->(t)",
    );

    let grow_to = |g: &mut Graph, upto: usize, from: usize| {
        for i in from..upto {
            q(g, &format!("INSERT (:Emp {{id: 'e{i}'}})"));
            q(
                g,
                &format!(
                    "MATCH (s:Emp {{id: 'e{i}'}}), (t:Dept {{id: 'D0'}}) INSERT (s)-[:IN_DEPT]->(t)"
                ),
            );
        }
    };
    // The probe's own traversal is identical at both sizes — only the population
    // of the IN_DEPT edge type changes underneath it.
    let probe = "MATCH (s:Probe {id: 'PR'})-[r:IN_DEPT]->(t) RETURN t.id AS x";
    let time = |g: &mut Graph| {
        for _ in 0..20 {
            q(g, probe);
        }
        let mut best = f64::INFINITY;

        for _ in 0..REPS {
            let t = Instant::now();

            for _ in 0..200 {
                q(g, probe);
            }

            best = best.min(t.elapsed().as_secs_f64());
        }

        best
    };

    grow_to(&mut g, 2_000, 0);
    let small = time(&mut g);
    grow_to(&mut g, 32_000, 2_000);
    let large = time(&mut g);

    let ratio = large / small.max(f64::MIN_POSITIVE);
    assert!(
        ratio < 6.0,
        "one-edge traversal scaled with edge-type population: 16x more edges cost \
         {ratio:.1}x more time ({small:.4}s -> {large:.4}s, min of {REPS}). The \
         planner is scanning the by_etype bucket instead of seeking the indexed \
         anchor's adjacency."
    );
}

/// A statement that faults inside an explicit transaction must undo only its own
/// writes and leave the transaction OPEN. Regression: `finish_statement` used to
/// call `rollback_tx`, which resets `tx_depth` to 0 unconditionally — so an app
/// that caught a statement error silently fell out of its transaction, every later
/// write auto-committed, and the closing ROLLBACK became a no-op. Native-only;
/// pure-TS `@lenke/core` was never affected.
#[test]
fn tx_statement_error_does_not_end_the_enclosing_transaction() {
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "START TRANSACTION");
    q(&mut g, "INSERT (:Acct {id: 'a'})");

    // An execution-time fault, of the kind an application legitimately catches.
    assert!(exec_err(&mut g, "RETURN no_such_fn(1) AS x"));

    // The transaction must still be open — this is the whole bug.
    assert!(
        g.in_transaction(),
        "a faulting statement ended the enclosing transaction"
    );

    // A later write is still staged, not auto-committed...
    q(&mut g, "INSERT (:Acct {id: 'b'})");
    q(&mut g, "ROLLBACK");

    // ...so ROLLBACK discards BOTH inserts. Before the fix, 'b' survived.
    assert!(!g.in_transaction());
    assert!(
        acct_ids(&mut g).is_empty(),
        "ROLLBACK left writes behind after a caught statement error"
    );
}

/// The faulting statement's own partial writes are still undone (per-statement
/// atomicity), while the enclosing transaction's earlier writes survive to be
/// committed — the two halves the single `rollback_tx` call used to conflate.
#[test]
fn tx_statement_error_undoes_only_that_statement() {
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "START TRANSACTION");
    q(&mut g, "INSERT (:Acct {id: 'keep'})");
    // Writes, then faults, in one statement: the INSERT must leave no trace.
    assert!(exec_err(
        &mut g,
        "INSERT (:Acct {id: 'gone'}) RETURN no_such_fn(1) AS x"
    ));
    q(&mut g, "COMMIT");
    assert_eq!(acct_ids(&mut g), vec!["keep"]);
}

#[test]
fn tx_keywords_start_insert_rollback_discards() {
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "INSERT (:Acct {id: 'seed'})");
    q(&mut g, "START TRANSACTION");
    q(&mut g, "INSERT (:Acct {id: 'a'})");
    q(&mut g, "ROLLBACK");
    assert!(!g.in_transaction());
    assert_eq!(acct_ids(&mut g), vec!["seed"]);
}

#[test]
fn tx_keywords_commit_work_and_rollback_work() {
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "START TRANSACTION");
    q(&mut g, "INSERT (:Acct {id: 'a'})");
    q(&mut g, "COMMIT WORK");
    assert_eq!(acct_ids(&mut g), vec!["a"]);
    q(&mut g, "START TRANSACTION");
    q(&mut g, "INSERT (:Acct {id: 'b'})");
    q(&mut g, "ROLLBACK WORK");
    assert_eq!(acct_ids(&mut g), vec!["a"]);
}

#[test]
fn tx_keywords_deferred_required_commits_when_valid() {
    let mut g = ndjson::decode("").unwrap();
    g.create_required_constraint("Acct", "email").unwrap();
    // The intermediate state (an Acct with no email) is allowed until COMMIT.
    q(&mut g, "START TRANSACTION");
    q(&mut g, "INSERT (:Acct {id: 'a'})");
    q(&mut g, "MATCH (n:Acct {id: 'a'}) SET n.email = 'a@x.io'");
    q(&mut g, "COMMIT");
    assert_eq!(
        rows(&mut g, "MATCH (n:Acct) RETURN n.email AS e"),
        vec![vec![s("a@x.io")]]
    );
}

#[test]
fn tx_keywords_deferred_required_rolls_back_when_invalid() {
    let mut g = ndjson::decode("").unwrap();
    g.create_required_constraint("Acct", "email").unwrap();
    q(&mut g, "START TRANSACTION");
    q(&mut g, "INSERT (:Acct {id: 'a', email: 'a@x.io'})");
    q(&mut g, "INSERT (:Acct {id: 'b'})"); // never gets an email
    assert_eq!(
        tx_err(&mut g, "COMMIT"),
        crate::error_codes::ErrorCode::ConstraintViolation
    );
    assert!(!g.in_transaction());
    assert!(acct_ids(&mut g).is_empty());
}

#[test]
fn tx_keywords_nested_start_is_an_error() {
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "START TRANSACTION");
    assert_eq!(
        tx_err(&mut g, "START TRANSACTION"),
        crate::error_codes::ErrorCode::InvalidGraphOp
    );
    // The original transaction is untouched.
    assert!(g.in_transaction());
    q(&mut g, "ROLLBACK");
}

#[test]
fn tx_keywords_commit_or_rollback_with_no_tx_is_an_error() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        tx_err(&mut g, "COMMIT"),
        crate::error_codes::ErrorCode::InvalidGraphOp
    );
    assert_eq!(
        tx_err(&mut g, "ROLLBACK"),
        crate::error_codes::ErrorCode::InvalidGraphOp
    );
}

#[test]
fn tx_keywords_read_only_rejects_writes_allows_reads() {
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "INSERT (:Acct {id: 'seed'})");
    q(&mut g, "START TRANSACTION READ ONLY");
    // A read is fine.
    assert_eq!(acct_ids(&mut g), vec!["seed"]);
    // Every write shape is rejected before it applies.
    assert_eq!(
        tx_err(&mut g, "INSERT (:Acct {id: 'x'})"),
        crate::error_codes::ErrorCode::InvalidGraphOp
    );
    assert_eq!(
        tx_err(&mut g, "MATCH (n:Acct) SET n.touched = true"),
        crate::error_codes::ErrorCode::InvalidGraphOp
    );
    assert_eq!(
        tx_err(&mut g, "MATCH (n:Acct {id: 'seed'}) DELETE n"),
        crate::error_codes::ErrorCode::InvalidGraphOp
    );
    q(&mut g, "COMMIT");
    // After commit the read-only mode is cleared — writes work again.
    q(&mut g, "INSERT (:Acct {id: 'x'})");
    assert_eq!(acct_ids(&mut g), vec!["seed", "x"]);
}

#[test]
fn tx_keywords_read_write_allows_writes() {
    use super::ast::{AccessMode, Statement, TxKind};
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "START TRANSACTION READ WRITE");
    q(&mut g, "INSERT (:Acct {id: 'a'})");
    q(&mut g, "ROLLBACK");
    q(&mut g, "INSERT (:Acct {id: 'b'})");
    assert_eq!(acct_ids(&mut g), vec!["b"]);

    // The parsed access mode is READ WRITE.
    match parse("START TRANSACTION READ WRITE").unwrap() {
        Statement::Tx(tx) => {
            assert_eq!(tx.kind, TxKind::Start);
            assert_eq!(tx.access_mode, Some(AccessMode::ReadWrite));
        }
        other => panic!("expected a TxControl, got {other:?}"),
    }
}

#[test]
fn tx_keywords_parse_shapes() {
    use super::ast::{AccessMode, Statement, TxKind};
    let tx = |src: &str| match parse(src).unwrap() {
        Statement::Tx(tx) => tx,
        other => panic!("expected a TxControl for `{src}`, got {other:?}"),
    };
    assert_eq!(tx("START TRANSACTION").kind, TxKind::Start);
    assert_eq!(tx("START TRANSACTION").access_mode, None);
    assert_eq!(
        tx("START TRANSACTION READ ONLY").access_mode,
        Some(AccessMode::ReadOnly)
    );
    assert_eq!(
        tx("start transaction read only").access_mode,
        Some(AccessMode::ReadOnly)
    );
    assert_eq!(tx("COMMIT").kind, TxKind::Commit);
    assert_eq!(tx("COMMIT WORK").kind, TxKind::Commit);
    assert_eq!(tx("ROLLBACK").kind, TxKind::Rollback);
    assert_eq!(tx("ROLLBACK WORK").kind, TxKind::Rollback);
    // A linear query is NOT a TxControl.
    assert!(matches!(
        parse("MATCH (n) RETURN n").unwrap(),
        Statement::Query(_)
    ));
}

#[test]
fn tx_keywords_malformed_are_syntax_errors() {
    assert!(parse("START").is_err());
    assert!(parse("START FROBNICATE").is_err());
    assert!(parse("START TRANSACTION READ SIDEWAYS").is_err());
    assert!(parse("START TRANSACTION READ ONLY READ WRITE").is_err());
    assert!(parse("COMMIT ALL THE THINGS").is_err());
}

#[test]
fn tx_keywords_soft_words_stay_identifiers() {
    // `read` / `write` / `only` / `transaction` are NOT reserved — usable as a
    // label, variable, property key, and alias.
    let mut g = ndjson::decode("").unwrap();
    q(&mut g, "INSERT (:read {write: 1})");
    assert_eq!(
        rows(&mut g, "MATCH (read:read) RETURN read.write AS only"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:read) RETURN n.write AS transaction"),
        vec![vec![n(1.0)]]
    );
}

// --- reserved words in binding positions (C10) ------------------------------

/// The message a rejected parse produced (panics if it unexpectedly parsed).
fn reject_msg(src: &str) -> String {
    match parse(src) {
        Ok(_) => panic!("expected `{src}` to be rejected, but it parsed"),
        Err(e) => e.message,
    }
}

#[test]
fn reserved_word_labels_get_consistent_backtick_naming_errors() {
    // Both token classes: structural keywords (Order/Count/Match/Set) that used
    // to fail `expect(Ident)` with a generic lowercased message, and reserved
    // idents (Group/Product). Every one now names backticks and keeps casing.
    for word in ["Group", "Product", "Order", "Count", "Match", "Set"] {
        let msg = reject_msg(&format!("MATCH (x:{word}) RETURN x"));
        // Original casing, echoed in the name and the backtick suggestion.
        assert!(
            msg.contains(&format!("`{word}`")),
            "label `{word}`: message did not preserve casing / name backticks: {msg}"
        );
        assert!(
            msg.contains("delimited identifier"),
            "label `{word}`: message did not name the delimited-identifier remedy: {msg}"
        );
        assert!(msg.contains("reserved word"), "label `{word}`: {msg}");
    }
}

#[test]
fn reserved_word_delimited_label_parses() {
    assert!(parse("MATCH (x:`Order`) RETURN x").is_ok());
}

#[test]
fn plain_label_parses() {
    assert!(parse("MATCH (x:Person) RETURN x").is_ok());
}

#[test]
fn reserved_word_variable_gets_improved_message() {
    let msg = reject_msg("MATCH (Group) RETURN 1");
    assert!(msg.contains("`Group`"), "{msg}");
    assert!(msg.contains("variable"), "{msg}");
    assert!(msg.contains("delimited identifier"), "{msg}");
}

#[test]
fn reserved_keyword_alias_recovers_casing() {
    let msg = reject_msg("MATCH (x) RETURN x AS Order");
    assert!(msg.contains("`Order`"), "{msg}");
    assert!(msg.contains("delimited identifier"), "{msg}");
}

#[test]
fn reserved_word_property_key_gets_improved_message() {
    let msg = reject_msg("MATCH (x { Order: 1 }) RETURN x");
    assert!(msg.contains("`Order`"), "{msg}");
    assert!(msg.contains("delimited identifier"), "{msg}");
}

#[test]
fn reserved_words_as_function_names_still_parse() {
    // Reserved words remain valid in call position — this is not a regression.
    assert!(parse("MATCH (x) RETURN upper(x.name), count(*)").is_ok());
}

/// `ANY SHORTEST` over a single quantified segment: the correct reachable set,
/// and a genuinely *shortest* path (the 1-hop `marko→lop`, not the 2-hop
/// `marko→josh→lop`), with the path value bound to `p`.
#[test]
fn any_shortest_reachability_and_path() {
    let mut g = modern();

    // Reachable from marko over `->*` (min 0 → marko reaches itself too).
    let mut names: Vec<String> = rows(
        &mut g,
        "MATCH ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' RETURN b.name AS n",
    )
    .into_iter()
    .map(|r| match &r[0] {
        Value::Str(s) => s.to_string(),
        other => panic!("expected a name, got {other:?}"),
    })
    .collect();
    names.sort();
    assert_eq!(names, vec!["josh", "lop", "marko", "ripple", "vadas"]);

    // Shortest marko→lop is the direct 1-hop CREATED edge, not marko→josh→lop.
    let r = rows(
        &mut g,
        "MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name = 'marko' AND b.name = 'lop' RETURN p",
    );
    assert_eq!(r.len(), 1);
    let Value::Map(m) = &r[0][0] else {
        panic!("expected a path map, got {:?}", r[0][0]);
    };
    let field = |k: &str| m.iter().find(|(key, _)| &**key == k).map(|(_, v)| v);
    assert_eq!(field("length"), Some(&Value::Num(1.0))); // one hop
    let Some(Value::List(vs)) = field("vertices") else {
        panic!("expected a vertices list");
    };
    assert_eq!(vs.len(), 2); // marko, lop
    let Some(Value::List(es)) = field("edges") else {
        panic!("expected an edges list");
    };
    assert_eq!(es.len(), 1);
}

/// `+` (min 1) excludes the zero-length self path; a bounded ceiling (`->{1,1}`)
/// keeps only the direct out-neighbours.
#[test]
fn any_shortest_plus_and_bounded() {
    let mut g = modern();

    // `->+` from marko: every reachable vertex EXCEPT marko itself (no 0-hop).
    let mut plus: Vec<String> = rows(
        &mut g,
        "MATCH ANY SHORTEST (a)-[]->+(b) WHERE a.name = 'marko' RETURN b.name AS n",
    )
    .into_iter()
    .map(|r| match &r[0] {
        Value::Str(s) => s.to_string(),
        other => panic!("{other:?}"),
    })
    .collect();
    plus.sort();
    assert_eq!(plus, vec!["josh", "lop", "ripple", "vadas"]);

    // `->{1,1}` from marko: only the direct out-neighbours.
    let mut one: Vec<String> = rows(
        &mut g,
        "MATCH ANY SHORTEST (a)-[]->{1,1}(b) WHERE a.name = 'marko' RETURN b.name AS n",
    )
    .into_iter()
    .map(|r| match &r[0] {
        Value::Str(s) => s.to_string(),
        other => panic!("{other:?}"),
    })
    .collect();
    one.sort();
    assert_eq!(one, vec!["josh", "lop", "vadas"]);
}

/// `->+(a)` closing on the seed finds the shortest cycle back to it
/// (the seed is marked at BFS distance 0 yet is a valid endpoint via a cycle),
/// while `->*` still yields the zero-length self path.
#[test]
fn any_shortest_closes_on_the_seed_cycle() {
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
            r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e2","from":"b","to":"a","labels":["R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .unwrap();

    // `->+(a)`: the shortest cycle a→b→a, length 2.
    let plus = rows(
        &mut g,
        "MATCH p = ANY SHORTEST (a:N {id:'a'})-[:R]->+(a) RETURN path_length(p) AS len",
    );
    assert_eq!(plus, vec![vec![Value::Num(2.0)]]);

    // `->*(a)`: min 0 admits the zero-length self path (length 0), unchanged.
    let star = rows(
        &mut g,
        "MATCH p = ANY SHORTEST (a:N {id:'a'})-[:R]->*(a) RETURN path_length(p) AS len",
    );
    assert_eq!(star, vec![vec![Value::Num(0.0)]]);
}

#[test]
fn path_modes_restrict_repeated_elements() {
    // Triangle a→b→c→a plus a→d and a back-edge b→a. From a within 1..3 hops the
    // reachable-endpoint multiplicity differs by mode: WALK ⊃ TRAIL ⊃ SIMPLE ⊃
    // ACYCLIC. No mode == TRAIL (the default), so the shape is unchanged.
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
            r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
            r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
            r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e3","from":"c","to":"a","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e4","from":"a","to":"d","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e5","from":"b","to":"a","labels":["R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    let count = |g: &mut Graph, mode: &str| {
        rows(
            g,
            &format!("MATCH {mode} (a:N {{id:'a'}})-[:R]->{{1,3}}(x) RETURN x.id AS id"),
        )
        .len()
    };

    let default = count(&mut g, "");
    assert_eq!(count(&mut g, "TRAIL"), default, "no mode == TRAIL");
    assert!(
        count(&mut g, "WALK") > default,
        "WALK admits repeated edges"
    );
    // ACYCLIC forbids the cycle back to the seed; SIMPLE allows only that close.
    let acyclic = count(&mut g, "ACYCLIC");
    let simple = count(&mut g, "SIMPLE");
    assert!(acyclic < default);
    assert!(simple > acyclic && simple <= default);
    // ACYCLIC never revisits a node, so `a` (the cycle-back endpoint) is absent.
    let ac_ends = rows(
        &mut g,
        "MATCH ACYCLIC (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id",
    );
    assert!(ac_ends.iter().all(|r| r[0] != s("a")));
}

#[test]
fn all_shortest_enumerates_every_tied_path() {
    // Diamond a→b→d, a→c→d: two shortest paths a..d (length 2). `ALL SHORTEST`
    // returns both (ISO per-path multiplicity); `ANY SHORTEST` returns one.
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
            r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
            r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
            r#"{"type":"edge","id":"ea","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"eb","from":"a","to":"c","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"ec","from":"b","to":"d","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"ed","from":"c","to":"d","labels":["R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .unwrap();

    let all = rows(
        &mut g,
        "MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    assert_eq!(all.len(), 2, "both tied shortest paths a..d");
    assert!(all.iter().all(|r| r[0] == n(2.0)));

    let any = rows(
        &mut g,
        "MATCH p = ANY SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    assert_eq!(any.len(), 1, "ANY SHORTEST keeps one");

    // Endpoint multiplicity over `->*` (min 0): a(1) b(1) c(1) d(2) = 5 rows.
    let ends = rows(
        &mut g,
        "MATCH ALL SHORTEST (a:N {id:'a'})-[:R]->*(x) RETURN x.id AS id",
    );
    assert_eq!(ends.len(), 5);
}

#[test]
fn postfix_property_chains_off_a_subscript() {
    // `list[i].prop` — property access chained off a subscript (the per-hop path
    // accessor), incl. `list[i].prop <op> list[j].prop` (consecutive-hop
    // comparison, the LaunderHunt motif) and out-of-range → null.
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"n0","labels":["N"],"properties":{"id":"n0"}}"#,
            r#"{"type":"node","id":"n1","labels":["N"],"properties":{"id":"n1"}}"#,
            r#"{"type":"node","id":"n2","labels":["N"],"properties":{"id":"n2"}}"#,
            r#"{"type":"edge","id":"e1","from":"n0","to":"n1","labels":["R"],"properties":{"w":5.0}}"#,
            r#"{"type":"edge","id":"e2","from":"n1","to":"n2","labels":["R"],"properties":{"w":9.0}}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    let path = "MATCH p = ANY SHORTEST (a:N {id:'n0'})-[:R]->*(b:N {id:'n2'})";

    assert_eq!(
        rows(&mut g, &format!("{path} RETURN edges(p)[0].w AS w")),
        vec![vec![n(5.0)]]
    );
    assert_eq!(
        rows(&mut g, &format!("{path} RETURN nodes(p)[2].id AS id")),
        vec![vec![s("n2")]]
    );
    // out-of-range subscript → null, and `.prop` on that null stays null.
    assert_eq!(
        rows(&mut g, &format!("{path} RETURN edges(p)[9].w AS w")),
        vec![vec![Value::Null]]
    );
    // consecutive-hop comparison: hop 1's weight (9) > hop 0's (5).
    assert_eq!(
        rows(
            &mut g,
            &format!("{path} RETURN edges(p)[1].w > edges(p)[0].w AS inc")
        ),
        vec![vec![b(true)]]
    );
}

// ---------------------------------------------------------------------------
// Path modes (WALK / TRAIL / SIMPLE / ACYCLIC): one focused test per restrictor,
// on the minimal fixture that makes the distinction observable.
// ---------------------------------------------------------------------------

/// WALK places no restriction: on a→b, b→a it re-treads both edges, so `{1,4}`
/// from `a` yields four walks (b, a, b, a).
#[test]
fn walk_mode_admits_repeated_edges() {
    let mut g = bidir_pair();
    let ends = rows(
        &mut g,
        "MATCH WALK (a:N {id:'a'})-[:R]->{1,4}(x) RETURN x.id AS id",
    );
    assert_eq!(ends.len(), 4, "a-b, a-b-a, a-b-a-b, a-b-a-b-a");
}

/// TRAIL (the default) forbids reusing an edge: a→b then b→a is fine (distinct
/// edges), but a→b→a→b would reuse e0, so `{1,4}` stops at length 2.
#[test]
fn trail_mode_forbids_edge_reuse() {
    let mut g = bidir_pair();
    let walks = rows(
        &mut g,
        "MATCH TRAIL (a:N {id:'a'})-[:R]->{1,4}(x) RETURN x.id AS id",
    );
    assert_eq!(walks.len(), 2, "a-b, a-b-a only");
    // No mode == TRAIL: identical count.
    let bare = rows(
        &mut g,
        "MATCH (a:N {id:'a'})-[:R]->{1,4}(x) RETURN x.id AS id",
    );
    assert_eq!(bare.len(), 2, "default mode is TRAIL");
}

/// SIMPLE permits revisiting a node only when it closes the walk back to the
/// start. On the triangle a→b→c→a that admits the cycle-close; no other repeats.
#[test]
fn simple_mode_allows_only_the_closing_cycle() {
    let mut g = triangle_tail();
    // a-b-c-a closes at the seed → allowed; endpoint `a` is present.
    assert_eq!(
        mode_ends(&mut g, "SIMPLE", 1, 3),
        vec![s("a"), s("b"), s("c"), s("d")]
    );
}

/// ACYCLIC forbids every repeated node, so the cycle back to the seed (`a`) is
/// excluded even though it closes.
#[test]
fn acyclic_mode_excludes_the_cycle_back_to_seed() {
    let mut g = triangle_tail();
    let ends = mode_ends(&mut g, "ACYCLIC", 1, 3);
    assert!(!ends.contains(&s("a")), "seed is never an ACYCLIC endpoint");
    assert_eq!(ends, vec![s("b"), s("c"), s("d")]);
}

/// The mode words are contextual keywords: `walk`/`trail`/`simple`/`acyclic`
/// remain usable as ordinary identifiers (variable + property names).
#[test]
fn path_mode_words_stay_usable_as_identifiers() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"n","labels":["N"],"properties":{"walk":1,"trail":2,"simple":3,"acyclic":4}}"#,
    ]);
    let r = rows(
        &mut g,
        "MATCH (walk:N) RETURN walk.walk AS w, walk.trail AS t, walk.simple AS s, walk.acyclic AS a",
    );
    assert_eq!(r, vec![vec![n(1.0), n(2.0), n(3.0), n(4.0)]]);
}

/// count(*) over a non-TRAIL mode must route through the general matcher (the
/// count fast path assumes trail semantics); it agrees with the enumerated rows.
#[test]
fn non_trail_mode_count_matches_enumeration() {
    let mut g = bidir_pair();
    for mode in ["WALK", "SIMPLE", "ACYCLIC"] {
        let c = rows(
            &mut g,
            &format!("MATCH {mode} (a:N {{id:'a'}})-[:R]->{{1,4}}(x) RETURN count(*) AS c"),
        );
        let enumerated = rows(
            &mut g,
            &format!("MATCH {mode} (a:N {{id:'a'}})-[:R]->{{1,4}}(x) RETURN x.id AS id"),
        )
        .len() as f64;
        assert_eq!(c, vec![vec![n(enumerated)]], "{mode} count vs enumeration");
    }
}

// ---------------------------------------------------------------------------
// Bare path binding (`p = (a)-[]->{m,n}(b)`, no selector) — edge cases beyond
// `bare_path_binds_every_walk`.
// ---------------------------------------------------------------------------

/// An unreachable endpoint binds no path (empty result, not a fault).
#[test]
fn bare_path_empty_when_unreachable() {
    let mut g = triangle_tail();
    let r = rows(
        &mut g,
        "MATCH p = (a:N {id:'d'})-[:R]->{1,3}(x) RETURN path_length(p) AS len",
    );
    assert!(r.is_empty(), "d has no out-edges");
}

/// `{0,n}` includes the zero-length walk that stays at the seed (path_length 0).
#[test]
fn bare_path_min_zero_binds_the_seed() {
    let mut g = triangle_tail();
    let lens = rows(
        &mut g,
        "MATCH p = (a:N {id:'a'})-[:R]->{0,2}(x) RETURN path_length(p) AS len ORDER BY len",
    );
    assert_eq!(lens[0], vec![n(0.0)], "the empty walk binds path_length 0");
}

/// The bound path carries its actual edges: edges(p) length == hops.
#[test]
fn bare_path_edges_match_hop_count() {
    let mut g = triangle_tail();
    let r = rows(
        &mut g,
        "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) \
         RETURN path_length(p) AS len, size(edges(p)) AS es ORDER BY len, x.id",
    );
    // Every row's edge count equals its hop count.
    assert!(
        r.iter().all(|row| row[0] == row[1]),
        "edges == hops per path"
    );
}

// ---------------------------------------------------------------------------
// ALL SHORTEST — edge cases beyond the diamond enumeration.
// ---------------------------------------------------------------------------

/// When exactly one shortest path exists, ALL SHORTEST and ANY SHORTEST agree.
#[test]
fn all_shortest_single_path_agrees_with_any() {
    // a→b→c is the only a..c path.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{}}"#,
    ]);
    let all = rows(
        &mut g,
        "MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'c'}) RETURN path_length(p) AS len",
    );
    assert_eq!(all, vec![vec![n(2.0)]]);
}

/// No path to the target → empty result (not a fault, not a null row).
#[test]
fn all_shortest_no_path_is_empty() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"z","labels":["N"],"properties":{"id":"z"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"a","labels":["R"],"properties":{}}"#,
    ]);
    let all = rows(
        &mut g,
        "MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'z'}) RETURN path_length(p) AS len",
    );
    assert!(all.is_empty());
}

/// ALL SHORTEST to the seed itself is the zero-length path (min 0 via `->*`).
#[test]
fn all_shortest_zero_length_to_self() {
    let mut g = triangle_tail();
    let all = rows(
        &mut g,
        "MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(a) RETURN path_length(p) AS len",
    );
    assert_eq!(all, vec![vec![n(0.0)]]);
}

// ---------------------------------------------------------------------------
// Postfix `.prop` chaining — edge cases beyond the subscript motif.
// ---------------------------------------------------------------------------

/// `.prop` reads straight off a bound vertex/edge variable (no subscript).
#[test]
fn postfix_prop_off_a_bound_variable() {
    let mut g = triangle_tail();
    let r = rows(
        &mut g,
        "MATCH (a:N {id:'a'})-[e:R]->(b) RETURN (b).id AS bid ORDER BY bid",
    );
    assert_eq!(r, vec![vec![s("b")], vec![s("d")]]);
}

/// `.prop` of an absent key is null; chaining another `.prop` off null stays null.
#[test]
fn postfix_prop_of_missing_key_is_null() {
    let mut g = triangle_tail();
    let r = rows(
        &mut g,
        "MATCH p = ANY SHORTEST (a:N {id:'a'})-[:R]->*(b:N {id:'c'}) RETURN nodes(p)[0].nope AS x",
    );
    assert_eq!(r, vec![vec![Value::Null]]);
}

// ---------------------------------------------------------------------------
// Per-hop edge predicates on variable-length segments — the predicate (inline
// props or WHERE, optionally naming each hop's edge) filters EVERY edge of the
// walk. A weighted chain a→b→c→d with an increasing then dropping weight.
// ---------------------------------------------------------------------------

/// a=(amt 10)=>b=(amt 20)=>c=(amt 5)=>d. The per-hop predicate filters each edge.
fn weighted_chain() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"amt":10.0}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{"amt":20.0}}"#,
        r#"{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{"amt":5.0}}"#,
    ])
}

/// Chain with node balances + edge amounts, for the quantified parenthesized
/// subpath (per-hop node + cross-element predicates).
fn balanced_chain() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a","bal":100.0}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b","bal":200.0}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c","bal":5.0}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d","bal":200.0}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"amt":30.0}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{"amt":20.0}}"#,
        r#"{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{"amt":10.0}}"#,
    ])
}

/// ISO quantified parenthesized subpath `((x)-[e]->(y) WHERE …){n,m}(t)` — the
/// per-repetition predicate can name the hop's SOURCE `(x)`, edge `(e)`, and
/// TARGET `(y)` (which the abbreviated `-[e]->{n,m}` form cannot); `(t)` is the
/// endpoint (the inner `y` is a group variable, so the endpoint is a separate node).
#[test]
fn quantified_subpath_per_hop_node_and_cross_element_predicates() {
    let mut g = balanced_chain();
    // Cross-element: each hop's source can afford the edge (`e.amt <= x.bal`).
    // a→b 30<=100 ✓, b→c 20<=200 ✓, c→d 10<=5 ✗ → reaches {b, c}, not d.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt <= x.bal){1,3} (t) RETURN t.id AS id",
        ),
        vec![s("b"), s("c")],
    );
    // Per-hop TARGET node predicate (`y.bal >= 100`): a→b b.bal 200 ✓, b→c c.bal 5
    // ✗ → reaches {b} only.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE y.bal >= 100){1,3} (t) RETURN t.id AS id",
        ),
        vec![s("b")],
    );
    // A permissive predicate reaches the whole chain (b, c, d).
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt >= 1){1,3} (t) RETURN t.id AS id",
        ),
        vec![s("b"), s("c"), s("d")],
    );
}

/// Phase 2 — GROUP variables: `x`/`e`/`y` bound inside the quantifier are exposed
/// to the outer query as LISTS of every hop's value (edges, source nodes, target
/// nodes), while the endpoint `(t)` is a singleton and the per-hop `WHERE` still
/// sees each as a SCALAR.
#[test]
fn quantified_subpath_group_variables() {
    let mut g = balanced_chain();
    // Exactly 2 hops from a: a→b→c. Endpoint t=c; e=[e1,e2]; x=[a,b]; y=[b,c].
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} (t) \
             RETURN t.id AS tid, size(e) AS ne, size(x) AS nx, size(y) AS ny, \
             x[0].id AS x0, y[1].id AS y1, e[0].amt AS e0",
        ),
        vec![vec![
            s("c"),
            n(2.0),
            n(2.0),
            n(2.0),
            s("a"),
            s("c"),
            n(30.0)
        ]],
    );
    // The DUAL context: the per-hop `WHERE` reads `e`/`x` as SCALARS, while `size(e)`
    // reads the group-variable LIST. `e.amt >= 15` admits a→b (30) and b→c (20) but
    // not c→d (10) → endpoints {b (1 hop), c (2 hops)} with their edge-list sizes.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt >= 15){1,3} (t) \
             RETURN t.id AS tid, size(e) AS ne ORDER BY t.id",
        ),
        vec![vec![s("b"), n(1.0)], vec![s("c"), n(2.0)]],
    );
    // A `{0,1}` walk includes the zero-hop match: endpoint = the seed, empty groups.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){0,1} (t) \
             RETURN t.id AS tid, size(e) AS ne ORDER BY t.id, ne",
        ),
        vec![vec![s("a"), n(0.0)], vec![s("b"), n(1.0)]],
    );
    // The endpoint may be anonymous (no `(t)`): the group vars still bind.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} RETURN size(e) AS ne",
        ),
        vec![vec![n(2.0)]],
    );
}

// ---------------------------------------------------------------------------
// Quantified segments through the VECTORIZED scan. `expand_scan` drives the same
// walker (`reachable_each_unit`) and the same group binder (`bind_group_vars`) the
// scalar matcher uses, once per frontier row, and reads the bound group variables
// off into `Val::List` value columns. These pin what the columnar build must
// reproduce: the per-repetition ORDER inside each list, and row-for-row agreement
// with the scalar driver on every quantified shape.
// ---------------------------------------------------------------------------

/// a→b→c→d→e with a distinct `amt` per edge, so a group list's ORDER is observable
/// (a set-equal but mis-ordered list would still fail).
fn stepped_chain() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a","bal":1.0}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b","bal":2.0}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c","bal":3.0}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d","bal":4.0}}"#,
        r#"{"type":"node","id":"e","labels":["N"],"properties":{"id":"e","bal":5.0}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"amt":11.0}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{"amt":22.0}}"#,
        r#"{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{"amt":33.0}}"#,
        r#"{"type":"edge","id":"e4","from":"d","to":"e","labels":["R"],"properties":{"amt":44.0}}"#,
    ])
}

/// Run `query` under the vectorized scan and again under the scalar matcher, assert
/// the two agree row-for-row (order included — both enumerate seeds then trails in
/// the same order), and return the rows.
fn vec_eq_scalar(g: &mut Graph, query: &str) -> Vec<Vec<Value>> {
    let on = super::eval::with_vec_override(true, || rows(g, query));
    let off = super::eval::with_vec_override(false, || rows(g, query));
    assert_eq!(on, off, "vectorized != scalar for `{query}`");
    on
}

/// Every repetition's value lands in its group variable's list, IN WALK ORDER —
/// rep `i`'s source at `x[i]`, its edge at `e[i]`, its target at `y[i]` — and the
/// vectorized column build agrees with the scalar matcher.
#[test]
fn group_variables_bind_each_repetition_in_order() {
    let mut g = stepped_chain();
    // Three reps of a one-hop unit from `a`: a→b→c→d.
    // x = [a,b,c], y = [b,c,d], e = [e1,e2,e3] (amts 11, 22, 33).
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){3} (t) \
             RETURN t.id AS tid, size(x) AS nx, size(e) AS ne, size(y) AS ny, \
             x[0].id AS x0, x[1].id AS x1, x[2].id AS x2, \
             y[0].id AS y0, y[1].id AS y1, y[2].id AS y2, \
             e[0].amt AS a0, e[1].amt AS a1, e[2].amt AS a2",
        ),
        vec![vec![
            s("d"),
            n(3.0),
            n(3.0),
            n(3.0),
            s("a"),
            s("b"),
            s("c"),
            s("b"),
            s("c"),
            s("d"),
            n(11.0),
            n(22.0),
            n(33.0),
        ]],
    );
    // A TWO-hop unit: each rep contributes one value per element position, so the
    // mid node `m` strides by 2 over the walk while `y` takes every other target.
    // rep1 = a→b→c (m=b, y=c), rep2 = c→d→e (m=d, y=e).
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){2} (t) \
             RETURN t.id AS tid, x[0].id AS x0, x[1].id AS x1, \
             m[0].id AS m0, m[1].id AS m1, y[0].id AS y0, y[1].id AS y1, \
             e1[0].amt AS p0, e2[0].amt AS q0, e1[1].amt AS p1, e2[1].amt AS q1",
        ),
        vec![vec![
            s("e"),
            s("a"),
            s("c"),
            s("b"),
            s("d"),
            s("c"),
            s("e"),
            n(11.0),
            n(22.0),
            n(33.0),
            n(44.0),
        ]],
    );
}

/// Growing repetition counts: the SAME variable is a list of length 1, 2, 3 … on
/// successive rows, so the value column really is per-row (not a shape frozen by
/// the first row the builder saw).
#[test]
fn group_variable_list_length_tracks_each_row_repetition_count() {
    let mut g = stepped_chain();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,4} (t) \
             RETURN t.id AS tid, size(e) AS ne, e[0].amt AS first, \
             y[size(y) - 1].id AS last ORDER BY size(e)",
        ),
        vec![
            vec![s("b"), n(1.0), n(11.0), s("b")],
            vec![s("c"), n(2.0), n(11.0), s("c")],
            vec![s("d"), n(3.0), n(11.0), s("d")],
            vec![s("e"), n(4.0), n(11.0), s("e")],
        ],
    );
    // `{0,n}` — the zero-repetition match binds every group variable to an EMPTY
    // list (the endpoint is the seed itself), on the same column as the longer rows.
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){0,2} (t) \
             RETURN t.id AS tid, size(x) AS nx, size(e) AS ne ORDER BY size(e)",
        ),
        vec![
            vec![s("a"), n(0.0), n(0.0)],
            vec![s("b"), n(1.0), n(1.0)],
            vec![s("c"), n(2.0), n(2.0)],
        ],
    );
}

/// A nested quantifier's group variables nest one list level per enclosing
/// quantifier — the columnar build stores whatever `bind_group_vars` produced, so
/// the nesting survives unchanged into the value column.
#[test]
fn nested_quantifier_group_variables_vectorize() {
    let mut g = stepped_chain();
    // `( (x)-[e:R]->{2,2}(y) ){2}`: two outer reps of two inner hops. `x`/`y` sit at
    // the OUTER unit's depth (flat lists); `e` is inside the inner quantifier, so it
    // is a list of lists — one inner list per outer rep.
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (s:N {id:'a'}) ( (x)-[e:R]->{2,2}(y) ){2} (t) \
             RETURN t.id AS tid, x[0].id AS x0, x[1].id AS x1, \
             y[0].id AS y0, y[1].id AS y1, \
             size(e) AS ne, size(e[0]) AS ne0, e[0][0].amt AS a00, e[1][1].amt AS a11",
        ),
        vec![vec![
            s("e"),
            s("a"),
            s("c"),
            s("c"),
            s("e"),
            n(2.0),
            n(2.0),
            n(11.0),
            n(44.0),
        ]],
    );
}

/// Group-variable columns survive a `WITH` (the vectorized pipeline carries value
/// columns forward) and can be filtered/aggregated past it.
#[test]
fn group_variables_carry_through_a_with() {
    let mut g = stepped_chain();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,3} (t) \
             WITH e AS hops, t WHERE size(hops) >= 2 \
             RETURN t.id AS tid, size(hops) AS n, hops[1].amt AS amt2 ORDER BY size(hops)",
        ),
        vec![vec![s("c"), n(2.0), n(22.0)], vec![s("d"), n(3.0), n(22.0)],],
    );
}

/// The abbreviated `-[e]->{n,m}` form does NOT expose `e` at the trail end (the
/// matcher binds it per hop, for that hop's own predicate only) — so it stays NULL,
/// in the vectorized build exactly as in the scalar one. This is the one quantified
/// edge variable the columnar path must NOT turn into a column.
#[test]
fn abbreviated_quantified_edge_variable_stays_unbound() {
    let mut g = stepped_chain();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (s:N {id:'a'})-[e:R]->{2}(t) RETURN t.id AS tid, e AS e",
        ),
        vec![vec![s("c"), Value::Null]],
    );
}

// ---------------------------------------------------------------------------
// `EXISTS { (u)-[:R]->() }` vectorizes as a bulk per-id adjacency test
// (`exists_semi_join_vec`) instead of the scalar `any_match` running once per
// row. Every case below runs through `vec_eq_scalar`, so the assertion is
// against the SCALAR path's own answer (`any_match`, unaffected by this
// change), not a hand-computed expectation — a shared bug in `expand` /
// `matches_label` would still show up as a *different* scalar answer, since
// `with_vec_override(false)` disables the columnar frame entirely.
// ---------------------------------------------------------------------------

/// `a` has one `R` out-edge and one `S` out-edge; `b` carries an extra label
/// `W` so a far-end label constraint has something to distinguish. `d` has no
/// out-edges at all (the no-match case). `e` is a self-loop. `p,q,r` form an
/// `R` triangle so every vertex of that label matches (the all-match case).
fn semi_join_fixture() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["V"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["V","W"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["V"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["V"],"properties":{"id":"d"}}"#,
        r#"{"type":"node","id":"e","labels":["V"],"properties":{"id":"e"}}"#,
        r#"{"type":"node","id":"p","labels":["Tri"],"properties":{"id":"p"}}"#,
        r#"{"type":"node","id":"q","labels":["Tri"],"properties":{"id":"q"}}"#,
        r#"{"type":"node","id":"r","labels":["Tri"],"properties":{"id":"r"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"a","to":"c","labels":["S"],"properties":{}}"#,
        r#"{"type":"edge","id":"e3","from":"e","to":"e","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e4","from":"p","to":"q","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e5","from":"q","to":"r","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e6","from":"r","to":"p","labels":["R"],"properties":{}}"#,
    ])
}

/// No matching neighbour: `d` has no out-edges of any type.
#[test]
fn exists_semi_join_vec_no_match() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:V) WHERE u.id = 'd' RETURN EXISTS { (u)-[:R]->() } AS r",
        ),
        vec![vec![b(false)]],
    );
}

/// Some match, some don't: of `a,b,c,d,e`, only `a` (→b) and `e` (self-loop)
/// have an `R` out-edge.
#[test]
fn exists_semi_join_vec_some_match() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:V) WHERE EXISTS { (u)-[:R]->() } RETURN count(*) AS c",
        ),
        vec![vec![n(2.0)]],
    );
}

/// All match: every vertex of the `R` triangle has an out-edge.
#[test]
fn exists_semi_join_vec_all_match() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:Tri) WHERE EXISTS { (u)-[:R]->() } RETURN count(*) AS c",
        ),
        vec![vec![n(3.0)]],
    );
}

/// A typed edge is not "any type": `a` has an `S` out-edge but no `Q`, and the
/// untyped `-[]->` form matches either.
#[test]
fn exists_semi_join_vec_typed_edge_vs_any_type() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:V) WHERE u.id = 'a' RETURN \
             EXISTS { (u)-[:R]->() } AS r, \
             EXISTS { (u)-[:S]->() } AS s, \
             EXISTS { (u)-[:Q]->() } AS q, \
             EXISTS { (u)-[]->() } AS anyt",
        ),
        vec![vec![b(true), b(true), b(false), b(true)]],
    );
}

/// A label constraint on the far end: `a`'s `R` neighbour `b` carries `W`, its
/// `S` neighbour `c` does not.
#[test]
fn exists_semi_join_vec_far_end_label() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:V) WHERE u.id = 'a' RETURN \
             EXISTS { (u)-[:R]->(:W) } AS r, \
             EXISTS { (u)-[:S]->(:W) } AS s",
        ),
        vec![vec![b(true), b(false)]],
    );
}

/// A property constraint on the far end (`const_props` — the constraint
/// doesn't read a slot, so it pre-evaluates once rather than declining).
#[test]
fn exists_semi_join_vec_far_end_property() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:V) WHERE u.id = 'a' RETURN \
             EXISTS { (u)-[:R]->({id: 'b'}) } AS hit, \
             EXISTS { (u)-[:R]->({id: 'z'}) } AS miss",
        ),
        vec![vec![b(true), b(false)]],
    );
}

/// A self-loop counts as its own out-neighbour — `expand` (shared with the
/// scalar matcher) applies `SelfLoops::Once`, so this must be true, not an
/// accidental false from a walk that skips the loop.
#[test]
fn exists_semi_join_vec_self_loop() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:V) WHERE u.id = 'e' RETURN \
             EXISTS { (u)-[:R]->() } AS out, \
             EXISTS { (u)<-[:R]-() } AS in_",
        ),
        vec![vec![b(true), b(true)]],
    );
}

/// An edge type no edge in the graph carries matches NOTHING — not every
/// vertex. This is the `seek`/Gremlin type-set inversion the brief calls out:
/// getting it backwards would silently turn `NOPE` into "any type" and every
/// row would read true.
#[test]
fn exists_semi_join_vec_unknown_edge_type_matches_nothing() {
    let mut g = semi_join_fixture();
    assert_eq!(
        vec_eq_scalar(
            &mut g,
            "MATCH (u:V) WHERE EXISTS { (u)-[:NOPE]->() } RETURN count(*) AS c",
        ),
        vec![vec![n(0.0)]],
    );
}

/// The vectorized and scalar drivers must agree on every quantified shape the
/// columnar builder now accepts — bounds, path mode, per-repetition `WHERE`,
/// endpoint constraints, a quantified segment in the middle of a longer path, an
/// anonymous endpoint, and the self-join shapes the builder declines (which must
/// fall back cleanly rather than produce different rows).
#[test]
fn vectorized_quantified_scan_agrees_with_the_scalar_matcher() {
    let mut g = stepped_chain();
    // Every ORDER BY names an INPUT expression (`t.id`), never an output alias:
    // sorting by an alias routes the whole query to the materialized-column path,
    // which would silently stop exercising the columnar build this test is about.
    for q in [
        // Abbreviated form, every bound shape.
        "MATCH (s:N {id:'a'})-[:R]->{1,3}(t) RETURN t.id AS id ORDER BY t.id",
        "MATCH (s:N {id:'a'})-[:R]->{0,2}(t) RETURN t.id AS id ORDER BY t.id",
        "MATCH (s:N {id:'a'})-[:R]->{2}(t) RETURN t.id AS id",
        "MATCH (s:N)-[:R]->{1,2}(t:N) RETURN s.id AS a, t.id AS b ORDER BY s.id, t.id",
        // Undirected + inbound repetition.
        "MATCH (s:N {id:'c'})-[:R]-{1,2}(t) RETURN t.id AS id ORDER BY t.id",
        "MATCH (s:N {id:'c'})<-[:R]-{1,2}(t) RETURN t.id AS id ORDER BY t.id",
        // Parenthesized unit with group variables.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,3} (t) \
         RETURN t.id AS id, size(e) AS ne, x[0].id AS x0 ORDER BY size(e)",
        // Per-repetition WHERE over the unit's own variables.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt <= 33.0){1,4} (t) \
         RETURN t.id AS id, size(e) AS ne ORDER BY size(e)",
        // Endpoint label / inline props / WHERE on the landing node.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,4} (t:N {id:'d'}) RETURN size(e) AS ne",
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,4} (t WHERE t.bal > 3.0) \
         RETURN t.id AS id ORDER BY t.id",
        // Anonymous endpoint, anonymous unit variables.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} RETURN size(e) AS ne",
        "MATCH (s:N {id:'a'}) (()-[:R]->()){2} (t) RETURN t.id AS id",
        // A quantified segment surrounded by fixed hops (columns replicate on both
        // sides of the repetition).
        "MATCH (s:N {id:'a'})-[f:R]->(m) ((x)-[e:R]->(y)){1,2} (t) \
         RETURN m.id AS mid, t.id AS tid, size(e) AS ne ORDER BY t.id, size(e)",
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,2} (m)-[f:R]->(t) \
         RETURN t.id AS tid, size(e) AS ne ORDER BY t.id",
        // Two quantified segments in one path.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,2} (m) ((p)-[r:R]->(qq)){1,2} (t) \
         RETURN t.id AS tid, size(e) AS ne, size(r) AS nr \
         ORDER BY t.id, size(e), size(r)",
        // Nested quantifier.
        "MATCH (s:N {id:'a'}) ( (x)-[e:R]->{1,2}(y) ){1,2} (t) \
         RETURN t.id AS tid, size(x) AS nx ORDER BY t.id, size(x)",
        // Non-default path modes (routed to the scalar driver, kept here so the
        // corpus covers them if that ever changes).
        "MATCH WALK (s:N {id:'a'})-[:R]->{1,3}(t) RETURN t.id AS id ORDER BY t.id",
        "MATCH ACYCLIC (s:N {id:'a'}) ((x)-[e:R]->(y)){1,3} (t) \
         RETURN t.id AS id ORDER BY t.id",
        // Clause WHERE over a group variable + aggregation over the repetition.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,4} (t) WHERE size(e) = 2 \
         RETURN t.id AS id",
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,4} (t) WHERE t.bal > 2.0 \
         RETURN count(*) AS c",
        // LIMIT — the vectorized build may stop the walk early; same prefix either way.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,4} (t) RETURN t.id AS id LIMIT 2",
        // A group-variable column carried into a following clause: an expanding
        // MATCH, an OPTIONAL MATCH (its new slots become nullable value columns
        // beside the list columns), a WITH, and DISTINCT over a list-derived value.
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,2} (t) MATCH (t)-[f:R]->(u) \
         RETURN u.id AS id, size(e) AS ne ORDER BY u.id",
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,2} (t) OPTIONAL MATCH (t)-[f:R]->(u) \
         RETURN t.id AS tid, size(e) AS ne ORDER BY t.id",
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,4} (t) WITH size(e) AS ne \
         RETURN DISTINCT ne ORDER BY ne",
        // Shapes the columnar builder declines (a slot bound twice): must still
        // agree, via the scalar fallback.
        "MATCH (s:N {id:'a'}) ((s)-[e:R]->(y)){2} (t) RETURN t.id AS id",
        "MATCH (s:N {id:'a'}) ((x)-[e:R]->(x)){1,2} (t) RETURN t.id AS id ORDER BY t.id",
    ] {
        vec_eq_scalar(&mut g, q);
    }
}

/// A layered graph: `layers` ranks of `width` nodes each, every node fully
/// connected to the next rank (dense fan-out). Rank-0 node 0 is `src`.
fn layered_dense(layers: usize, width: usize) -> Graph {
    let mut lines: Vec<String> = Vec::new();
    for l in 0..layers {
        for w in 0..width {
            let id = l * width + w;
            lines.push(format!(
                r#"{{"type":"node","id":"n{id}","labels":["N"],"properties":{{"id":"n{id}"}}}}"#
            ));
        }
    }
    let mut e = 0;
    for l in 0..layers - 1 {
        for a in 0..width {
            for b in 0..width {
                let from = l * width + a;
                let to = (l + 1) * width + b;
                lines.push(format!(
                    r#"{{"type":"edge","id":"e{e}","from":"n{from}","to":"n{to}","labels":["R"],"properties":{{"amt":10.0}}}}"#
                ));
                e += 1;
            }
        }
    }
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    graph_of(&refs)
}

/// Throwaway micro-benchmark (ignored; run with `--ignored --nocapture`) for the HOT
/// abbreviated var-length path at k=1. Prints elapsed wall time — no assertion (wall
/// clock is flaky as a gate). The abbreviated form is a one-hop unit run by the single
/// fused matcher (`reachable_each` → `reachable_each_unit`): ~74µs/iter here, vs ~177µs
/// for the old hand-tuned materialize-then-step fast-path it replaced. The lazy fused
/// walk is both faster AND O(path)-memory (no `d^k` per-vertex buffer).
#[test]
#[ignore]
fn bench_k1_abbreviated_walk() {
    let mut g = layered_dense(6, 6);
    let q = "MATCH (s:N {id:'n0'})-[:R]->{1,4}(x) RETURN count(*) AS c";
    // warm
    for _ in 0..5 {
        let _ = rows(&mut g, q);
    }
    let t = std::time::Instant::now();
    let iters = 200;
    for _ in 0..iters {
        let _ = rows(&mut g, q);
    }
    let el = t.elapsed();
    eprintln!(
        "bench_k1_abbreviated_walk: {iters} iters in {:?} ({:?}/iter)",
        el,
        el / iters
    );
}

/// Micro-benchmark (ignored) covering the whole var-length matcher surface the
/// unification touches: abbreviated k=1, single-edge subpath (k=1), and multi-element
/// units (k=2, k=3). Same dense DAG. Run before AND after a matcher change to catch a
/// regression on the linear (degenerate) case. No assertion — wall clock is flaky.
#[test]
#[ignore]
fn bench_var_length_matcher() {
    let mut g = layered_dense(6, 6);
    let cases = [
        (
            "k1_abbrev  ",
            "MATCH (s:N {id:'n0'})-[:R]->{1,4}(x) RETURN count(*) AS c",
        ),
        (
            "k1_subpath ",
            "MATCH (s:N {id:'n0'}) ((x)-[:R]->(y)){1,4} (t) RETURN count(*) AS c",
        ),
        (
            "k2_subpath ",
            "MATCH (s:N {id:'n0'}) ((x)-[:R]->(m)-[:R]->(y)){1,2} (t) RETURN count(*) AS c",
        ),
        (
            "k2_groupvar",
            "MATCH (s:N {id:'n0'}) ((x)-[e:R]->(m)-[e2:R]->(y)){1,2} (t) RETURN count(size(e)) AS c",
        ),
        (
            "k3_subpath ",
            "MATCH (s:N {id:'n0'}) ((x)-[:R]->(m)-[:R]->(w)-[:R]->(y)){1} (t) RETURN count(*) AS c",
        ),
        // NESTED units take the general structured binder (not the flat k-stride fast path).
        (
            "nest_anon  ",
            "MATCH (s:N {id:'n0'}) ( ()-[:R]->{1,2}() ){1,2} (t) RETURN count(*) AS c",
        ),
        (
            "nest_grpvar",
            "MATCH (s:N {id:'n0'}) ( (x)-[:R]->{1,2}(y) ){1,2} (t) RETURN count(size(x)) AS c",
        ),
        // ROW-returning shapes: these choose between the columnar build (`expand_scan`
        // drives the walker per frontier row into columns) and the scalar streaming
        // driver, which the `count(*)` rows above cannot distinguish. See
        // `bench_quantified_vec_vs_scalar` for the side-by-side.
        (
            "rows_abbrev",
            "MATCH (s:N {id:'n0'})-[:R]->{1,4}(x) RETURN x.id AS id",
        ),
        (
            "rows_grpvar",
            "MATCH (s:N {id:'n0'}) ((x)-[e:R]->(y)){1,4} (t) RETURN t.id AS id, size(e) AS ne",
        ),
    ];
    for (label, q) in cases {
        for _ in 0..5 {
            let _ = rows(&mut g, q);
        }
        let iters = 200;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = rows(&mut g, q);
        }
        let el = t.elapsed();
        eprintln!("bench {label}: {iters} iters, {:?}/iter", el / iters);
    }
}

/// The routing question for a quantified segment: columnar build (`expand_scan`)
/// vs the scalar streaming driver, same query, same fixture, forced both ways.
///
/// Measured on `layered_dense(6, 6)` (100 iters, repeated — the ordering is stable
/// run to run, the absolute numbers move ~10%):
///
/// ```text
///                            columnar   scalar
///   -[:R]->{1,4}  RETURN id     131µs     143µs   ← no group variables: a win
///   ((x)-[]->(y)){1,4} rows     365µs     286µs
///   ((x)-[e]->(y)){1,4} +size   802µs     411µs
///   … WHERE size(e) >= 2       1093µs     412µs
///   ( (x)-[]->{1,2}(y) ){1,2}  1115µs     890µs
/// ```
///
/// A quantified segment that exposes NO group variables vectorizes at or better
/// than the scalar walker. One that DOES is 1.3-2.7x worse, and the reason is
/// structural, not a missing tune-up: `bind_group_vars` allocates a `Val::List` per
/// variable per trail in BOTH drivers, but the columnar build keeps every one of
/// them live in a column instead of dropping it at the end of the row, and no
/// vectorized kernel reads a list column — `size(e)`, `e[0].amt` and friends all
/// fall to `scalar_col`, which rebuilds a full `Binding` per row and DEEP-COPIES
/// every list into it (`Val::List(Vec<Val>)`; `Val::Map` already solved exactly this
/// by boxing its payload in an `Arc`).
///
/// So the two candidate fixes are (a) an `Arc`-ed list payload, so binding a list
/// column is a refcount bump like a map, and (b) list kernels in `eval_vec` for
/// `size`/index, so the common group-variable expressions never bind at all.
/// Narrowing the per-row bind to the slots an expression actually reads helps but
/// is not sufficient on its own (it removes 2 of 3 copies here, not the retention).
#[test]
#[ignore]
fn bench_quantified_vec_vs_scalar() {
    let mut g = layered_dense(6, 6);
    let cases = [
        (
            "abbrev_rows",
            "MATCH (s:N {id:'n0'})-[:R]->{1,4}(x) RETURN x.id AS id",
        ),
        (
            "subpath_row",
            "MATCH (s:N {id:'n0'}) ((x)-[:R]->(y)){1,4} (t) RETURN t.id AS id",
        ),
        (
            "groupvar   ",
            "MATCH (s:N {id:'n0'}) ((x)-[e:R]->(y)){1,4} (t) RETURN t.id AS id, size(e) AS ne",
        ),
        (
            "groupvar_wh",
            "MATCH (s:N {id:'n0'}) ((x)-[e:R]->(y)){1,4} (t) WHERE size(e) >= 2 RETURN t.id AS id",
        ),
        (
            "nested_gv  ",
            "MATCH (s:N {id:'n0'}) ( (x)-[:R]->{1,2}(y) ){1,2} (t) RETURN t.id AS id, size(x) AS nx",
        ),
    ];
    for (label, q) in cases {
        for columnar in [true, false] {
            super::eval::with_vec_override(columnar, || {
                for _ in 0..5 {
                    let _ = rows(&mut g, q);
                }
                let iters = 100;
                let t = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = rows(&mut g, q);
                }
                eprintln!(
                    "bench {label} columnar={columnar}: {:?}/iter",
                    t.elapsed() / iters
                );
            });
        }
    }
}

/// A five-node chain a→b→c→d→e (uniform edge amt 10) for exercising MULTI-element
/// repetition units, where each repetition of the unit advances more than one hop.
fn five_chain() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"node","id":"e","labels":["N"],"properties":{"id":"e"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"amt":10.0}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{"amt":10.0}}"#,
        r#"{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{"amt":10.0}}"#,
        r#"{"type":"edge","id":"e4","from":"d","to":"e","labels":["R"],"properties":{"amt":10.0}}"#,
    ])
}

/// The abbreviated `-[]->{n,m}` form (`reachable_each`, a one-hop-unit wrapper) and an
/// equivalent single-edge parenthesized subpath `((x)-[]->(y)){n,m}` (a `k = 1` unit
/// with group binding) must return IDENTICAL endpoints under EVERY path mode. They run
/// the SAME fused matcher now, so this guards that the `k = 1` reduction stays exact
/// (and that group binding doesn't perturb endpoints).
#[test]
fn abbreviated_and_single_edge_subpath_agree_k1() {
    // A graph with a cycle + a tail so TRAIL/SIMPLE/ACYCLIC/WALK genuinely diverge
    // from one another (and thus meaningfully test that BOTH matchers agree per mode).
    let mut g = triangle_tail();
    for mode in ["WALK", "TRAIL", "SIMPLE", "ACYCLIC"] {
        // Bounded quantifiers only: an unbounded `+`/`*` under WALK on the cycle is
        // legitimately infinite (trail-budget fault), which is orthogonal to drift.
        for quant in ["{1,3}", "{0,2}", "{2}", "{1,4}"] {
            let abbrev =
                format!("MATCH {mode} (s:N {{id:'a'}})-[:R]->{quant}(x) RETURN x.id AS id");
            let subpath = format!(
                "MATCH {mode} (s:N {{id:'a'}}) ((y)-[:R]->(z)){quant} (x) RETURN x.id AS id"
            );
            assert_eq!(
                sorted_col0(&mut g, &abbrev),
                sorted_col0(&mut g, &subpath),
                "abbreviated vs single-edge subpath diverged for mode={mode} quant={quant}",
            );
        }
    }
}

/// A pattern may BEGIN with a quantified subpath (no leading anchor node), and a path
/// variable may bind the whole repeated walk — ISO `<parenthesized path pattern
/// expression> <quantifier>` as the first path factor. The walk seeds from every vertex
/// (anonymous start); the eval already handled the anchored form, this is the parser
/// accepting the unanchored one.
#[test]
fn unanchored_quantified_subpath_and_path_variable() {
    let mut g = balanced_chain(); // a→b→c→d, plus the `bal` props
                                  // Path variable over a quantified subpath: `p` is the repeated walk; from every
                                  // seed a 2-hop (2-rep single-edge) walk. Endpoints c (a→b→c) and d (b→c→d).
    assert_eq!(
        rows(
            &mut g,
            "MATCH p = ((x)-[e:R]->(y)){2} (t) RETURN t.id AS tid, path_length(p) AS len ORDER BY tid",
        ),
        vec![vec![s("c"), n(2.0)], vec![s("d"), n(2.0)]],
    );
    // Unanchored, no path variable: bounded {1,2} from every seed.
    assert_eq!(
        sorted_col0(&mut g, "MATCH ((x)-[:R]->(y)){1,2} (t) RETURN t.id AS id",),
        vec![s("b"), s("c"), s("c"), s("d"), s("d")],
    );
    // `nodes(p)` exposes the whole repeated walk's vertices (a→b→c for the first).
    assert_eq!(
        rows(
            &mut g,
            "MATCH p = ((x)-[e:R]->(y)){2} (t) WHERE t.id = 'c' RETURN size(nodes(p)) AS n",
        ),
        vec![vec![n(3.0)]],
    );
}

/// The bare parenthesized-subpath GROUPING `( <path> [WHERE] )` (NO quantifier) must
/// still parse as a WHERE-scoped grouping — the lookahead only reroutes a `((…)){n}`.
#[test]
fn parenthesized_grouping_without_quantifier_still_works() {
    let mut g = balanced_chain();
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH ((a)-[:R]->(b) WHERE a.bal < b.bal) RETURN a.id AS id",
        ),
        vec![s("a"), s("c")], // a→b 100<200 ✓; b→c 200<5 ✗; c→d 5<200 ✓
    );
}

/// A SIMPLE close (a walk returning to the seed) EMITS the seed but must NOT extend
/// past it — the cycle is closed. Regression for the unified matcher (this shape
/// slipped past cargo tests but conformance caught it). Triangle a→b→c→a + a→d + b→a.
#[test]
fn simple_close_emits_but_does_not_extend() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"c","to":"a","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"d","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"a","labels":["R"],"properties":{}}"#,
    ]);
    // 1-hop: b, d. 2-hop: a→b→c (c), a→b→a (close → a). 3-hop: a→b→c→a (close → a).
    // Two distinct closing trails to a → a twice; the close never extends further.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH SIMPLE (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id",
        ),
        vec![s("a"), s("a"), s("b"), s("c"), s("d")],
    );
}

/// A five-node chain fixture a→b→c→d→e for nested-quantifier tests.
fn chain5() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"node","id":"e","labels":["N"],"properties":{"id":"e"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e3","from":"c","to":"d","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e4","from":"d","to":"e","labels":["R"],"properties":{}}"#,
    ])
}

/// NESTED quantifiers `( … {a,b} … ){n,m}` — the outer repetition repeats an inner
/// variable-length sub-walk. Exercises the pushdown matcher's `resolve_general` path
/// (a `CElem::Sub`); v1 is endpoint enumeration (anonymous inner nodes/edges).
#[test]
fn nested_quantifier_endpoints() {
    let mut g = chain5();
    // One outer rep of a 1-3 hop walk from a → b, c, d.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ()-[:R]->{1,3}() ){1} (t) RETURN t.id AS id",
        ),
        vec![s("b"), s("c"), s("d")],
    );
    // TWO outer reps of a 1-2 hop walk: (a→{b,c}) then (→{c,d} or →{d,e}) → c,d,d,e
    // (one path per trail: b→c, b→c→d, c→d, c→d→e).
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ()-[:R]->{1,2}() ){2} (t) RETURN t.id AS id",
        ),
        vec![s("c"), s("d"), s("d"), s("e")],
    );
    // A plain hop followed by a quantified hop, one outer rep: 2–3 hops → c, d.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ()-[:R]->()-[:R]->{1,2}() ){1} (t) RETURN t.id AS id",
        ),
        vec![s("c"), s("d")],
    );
}

/// Nested-quantifier surface: NODE group variables are exposed as flat lists, per-hop
/// EDGE predicates filter inner edges, and a subpath-level WHERE is now a per-outer-rep
/// predicate over the grouped variables (all supported).
#[test]
fn nested_quantifier_group_vars_and_where() {
    // A NAMED inner NODE inside a nested quantifier IS supported (its landing is a group
    // variable at the outer unit's depth — a flat list).
    assert!(parse("MATCH (s) ( (x)-[:R]->{1,2}(m) ){2} (t) RETURN size(x)").is_ok());
    // A per-hop EDGE predicate on a nested inner hop IS supported (filters inner edges).
    assert!(parse("MATCH (s) ( ()-[e:R WHERE e.amt >= 5]->{1,2}() ){2} (t) RETURN t").is_ok());
    assert!(parse("MATCH (s) ( ()-[:R {amt:10.0}]->{1,3}() ){1} (t) RETURN t").is_ok());
    // A subpath-level WHERE on a nested quantifier IS supported (per outer rep, inner
    // variables bound as lists — e.g. `size(e)`).
    assert!(parse("MATCH (s) ( ()-[e:R]->{1,2}() WHERE size(e) = 2 ){2} (t) RETURN t").is_ok());
    // A plain (non-nested) subpath keeps its group variables + WHERE.
    assert!(parse("MATCH (s) ((x)-[e:R]->(y) WHERE x = y){1,2} (t) RETURN size(e)").is_ok());
}

/// #1 — a nested quantifier's NODE group variables are exposed as FLAT lists at the
/// outer unit's depth: the source `x` (once per outer rep) and the nested hop's landing
/// `y` (`-[]->{a,b}(y)`, the whole sub-unit's endpoint per outer rep).
#[test]
fn nested_quantifier_outer_group_variables() {
    let mut g = five_chain();
    // `( (x)-[:R]->{2,2}(y) ){2}` on a→b→c→d→e: two outer reps of exactly two inner hops.
    // rep1 = a→b→c (x=a, landing y=c); rep2 = c→d→e (x=c, y=e). x=[a,c], y=[c,e].
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ( (x)-[:R]->{2,2}(y) ){2} (t) \
             RETURN t.id AS tid, size(x) AS nx, size(y) AS ny, \
             x[0].id AS x0, x[1].id AS x1, y[0].id AS y0, y[1].id AS y1",
        ),
        vec![vec![s("e"), n(2.0), n(2.0), s("a"), s("c"), s("c"), s("e"),]],
    );
    // Varying inner count: `( (x)-[:R]->{1,2}(y) ){1}`, one outer rep from a → landings
    // b, c (1 or 2 hops). x is a 1-element list [a] each; y is [b] or [c].
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ( (x)-[:R]->{1,2}(y) ){1} (t) \
             RETURN t.id AS tid, size(x) AS nx, x[0].id AS x0, y[0].id AS y0 \
             ORDER BY tid",
        ),
        vec![
            vec![s("b"), n(1.0), s("a"), s("b")],
            vec![s("c"), n(1.0), s("a"), s("c")],
        ],
    );
}

/// A per-hop EDGE predicate on a nested inner hop (`-[e WHERE …]->{a,b}`) filters every
/// edge of the inner walk — the tractable slice of "WHERE inside a nested quantifier".
#[test]
fn nested_quantifier_per_hop_edge_predicate() {
    // a→b(10)→c(1)→d(10): the inner walk only follows amt ≥ 5 edges, so b→c blocks it.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{"amt":10.0}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{"amt":1.0}}"#,
        r#"{"type":"edge","from":"c","to":"d","labels":["R"],"properties":{"amt":10.0}}"#,
    ]);
    // WHERE on the inner edge: from a, only a→b passes (b→c amt 1 < 5 blocks). Endpoint b.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ()-[e:R WHERE e.amt >= 5]->{1,2}() ){1,2} (t) RETURN t.id AS id",
        ),
        vec![s("b")],
    );
    // Inline property predicate on the inner edge (amt = 10): same block at b→c.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ()-[:R {amt:10.0}]->{1,3}() ){1} (t) RETURN t.id AS id",
        ),
        vec![s("b")],
    );
}

/// #2 — a nested PARENTHESIZED subpath `( ((x)-[e]->(y)){a,b} ){n,m}` exposes its inner
/// variables as LIST-OF-LISTS: one list level per enclosing quantifier. The structured
/// binder assembles this with no special case (a `Sub` whose inner unit itself exposes).
#[test]
fn nested_parenthesized_subpath_list_of_lists() {
    let mut g = five_chain();
    // `( ((x)-[:R]->(y)){2,2} ){2}` on a→b→c→d→e: outer 2 reps, each 2 inner hops.
    // rep1: a→b, b→c → x=[a,b], y=[b,c]; rep2: c→d, d→e → x=[c,d], y=[d,e].
    // So x=[[a,b],[c,d]], y=[[b,c],[d,e]].
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ((x)-[:R]->(y)){2,2} ){2} (t) \
             RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, \
             x[0][0].id AS a00, x[0][1].id AS a01, x[1][0].id AS a10, \
             y[0][1].id AS y01, y[1][1].id AS y11",
        ),
        vec![vec![
            s("e"),
            n(2.0),
            n(2.0),
            s("a"),
            s("b"),
            s("c"),
            s("c"),
            s("e"),
        ]],
    );
}

/// #2 — a nested subpath's inner EDGE variable is also list-of-lists, and the inner
/// count may VARY per outer rep (the nested lists are ragged, not rectangular).
#[test]
fn nested_parenthesized_subpath_varying_and_edges() {
    let mut g = five_chain();
    // `( ((x)-[e:R]->(y)){1,2} ){1}`: one outer rep, inner walk of 1 or 2 hops from a.
    // → endpoint b (1 inner hop: x=[[a]]) or c (2 inner hops: x=[[a,b]]).
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){1,2} ){1} (t) \
             RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, size(e[0]) AS ne0 \
             ORDER BY tid",
        ),
        vec![
            vec![s("b"), n(1.0), n(1.0), n(1.0)],
            vec![s("c"), n(1.0), n(2.0), n(2.0)],
        ],
    );
    // Two outer reps, exactly one inner hop each: `( ((x)-[e:R]->(y)){1,1} ){2}` on
    // a→b→c → x=[[a],[b]], y=[[b],[c]] (each inner list length 1).
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){1,1} ){2} (t) \
             RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, \
             x[0][0].id AS a00, x[1][0].id AS a10, y[1][0].id AS y10",
        ),
        vec![vec![s("c"), n(2.0), n(1.0), s("a"), s("b"), s("c")]],
    );
}

/// #2 WHERE surface: predicates at every level of a nested parenthesized subpath.
#[test]
fn nested_parenthesized_subpath_restrictions() {
    // Per-hop predicate inside the inner subpath → OK.
    assert!(parse("MATCH (s) ( ((x)-[e:R WHERE e.amt > 0]->(y)){1,2} ){2} (t) RETURN t").is_ok());
    // The inner subpath keeps its own WHERE (a per-rep predicate over its group vars).
    assert!(parse("MATCH (s) ( ((x)-[e:R]->(y) WHERE x = y){1,2} ){2} (t) RETURN t").is_ok());
    // A WHERE on the OUTER nested quantifier IS supported — per outer rep, the inner vars
    // (x, e, y) are bound as lists.
    assert!(parse("MATCH (s) ( ((x)-[e:R]->(y)){1,2} WHERE size(e) = 2 ){2} (t) RETURN t").is_ok());
}

/// A subpath-level WHERE on a NESTED quantifier is a PER-OUTER-REP predicate with the
/// inner variables bound as LISTS. `size(e)` constrains each outer rep's inner walk
/// length — the crux that distinguishes per-rep from per-edge / whole-match filtering.
#[test]
fn nested_quantifier_per_rep_where_over_grouped_vars() {
    let mut g = five_chain(); // a→b→c→d→e
                              // Abbreviated inner: each outer rep must be exactly TWO inner hops (`size(e)=2`), so
                              // two outer reps walk all four edges → endpoint e. (`=1` → each rep one hop → c.)
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ()-[e:R]->{1,2}() WHERE size(e) = 2 ){2} (t) RETURN t.id AS id",
        ),
        vec![s("e")],
    );
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ()-[e:R]->{1,2}() WHERE size(e) = 1 ){2} (t) RETURN t.id AS id",
        ),
        vec![s("c")],
    );
    // Nested parenthesized subpath: same per-outer-rep constraint, inner vars still exposed
    // as list-of-lists at the end (x = [[a,b],[c,d]]).
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){1,2} WHERE size(e) = 2 ){2} (t) \
             RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, x[1][0].id AS a10",
        ),
        vec![vec![s("e"), n(2.0), n(2.0), s("c")]],
    );
    // A per-rep WHERE referencing list ELEMENTS: keep only outer reps whose inner walk
    // starts at 'a' or 'c' (the actual rep sources). Both reps qualify → endpoint e.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ( ((x)-[e:R]->(y)){2,2} WHERE x[0].id <> y[1].id ){2} (t) \
             RETURN t.id AS id",
        ),
        vec![s("e")],
    );
}

/// The fused matcher marks PER HOP, so ACYCLIC/SIMPLE now forbid a multi-element unit
/// from repeating a vertex INTERNALLY — the correct ISO reading (a node at most once
/// on an acyclic path), which the old per-unit-set check missed. A 2-hop unit through
/// a self-loop (`s→p`, `p→p`) revisits `p` within one unit: TRAIL keeps it (distinct
/// edges), ACYCLIC and SIMPLE reject it.
#[test]
fn multi_element_acyclic_rejects_intra_unit_vertex_repeat() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"s","labels":["N"],"properties":{"id":"s"}}"#,
        r#"{"type":"node","id":"p","labels":["N"],"properties":{"id":"p"}}"#,
        r#"{"type":"edge","id":"e1","from":"s","to":"p","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"p","to":"p","labels":["R"],"properties":{}}"#,
    ]);
    // TRAIL (default): s→p→p uses two distinct edges → endpoint p.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'s'}) ((x)-[:R]->(m)-[:R]->(y)){1} (t) RETURN t.id AS id",
        ),
        vec![s("p")],
    );
    // ACYCLIC / SIMPLE: the unit revisits p internally → rejected → no endpoint.
    for mode in ["ACYCLIC", "SIMPLE"] {
        assert_eq!(
            sorted_col0(
                &mut g,
                &format!(
                    "MATCH {mode} (s:N {{id:'s'}}) ((x)-[:R]->(m)-[:R]->(y)){{1}} (t) RETURN t.id AS id"
                ),
            ),
            Vec::<Value>::new(),
            "mode={mode} should reject the intra-unit vertex repeat",
        );
    }
}

/// ISO MULTI-element repetition unit `((x)-[e1]->(m)-[e2]->(y)){n,m}`: each
/// repetition advances TWO hops, so the endpoints land on even hop counts only.
#[test]
fn quantified_subpath_multi_element_unit_endpoints() {
    let mut g = five_chain();
    // One repetition of the 2-hop unit from a: a→b→c → endpoint c (never b — that's
    // mid-unit, not a repetition boundary).
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){1} (t) RETURN t.id AS id",
        ),
        vec![s("c")],
    );
    // {1,2}: one rep → c, two reps → a→b→c→d→e → e. Endpoints {c, e}; d (odd) excluded.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){1,2} (t) RETURN t.id AS id",
        ),
        vec![s("c"), s("e")],
    );
}

/// The intermediate node `m` and BOTH edges of a multi-element unit are group
/// variables, exposed as LISTS whose length is the repetition count.
#[test]
fn quantified_subpath_multi_element_group_variables() {
    let mut g = five_chain();
    // Two repetitions (a→b→c→d→e): x=[a,c], m=[b,d], y=[c,e]; e1=[a→b,c→d], e2=[b→c,d→e].
    assert_eq!(
        rows(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){2} (t) \
             RETURN t.id AS tid, size(e1) AS n1, size(e2) AS n2, size(m) AS nm, \
             x[0].id AS x0, x[1].id AS x1, m[0].id AS m0, m[1].id AS m1, y[1].id AS y1",
        ),
        vec![vec![
            s("e"),
            n(2.0),
            n(2.0),
            n(2.0),
            s("a"),
            s("c"),
            s("b"),
            s("d"),
            s("e"),
        ]],
    );
}

/// A per-unit `WHERE` spanning BOTH hops of a multi-element unit (`m` is the shared
/// interior node): admit a repetition only when the second edge is no larger than
/// the first. On the uniform chain (all amt 10) every unit passes; a stricter test
/// with `<` would admit none.
#[test]
fn quantified_subpath_multi_element_cross_hop_predicate() {
    let mut g = five_chain();
    // `e2.amt <= e1.amt` (10<=10) holds for every unit → endpoints {c, e} as before.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt <= e1.amt){1,2} (t) \
             RETURN t.id AS id",
        ),
        vec![s("c"), s("e")],
    );
    // `e2.amt < e1.amt` (10<10) fails for the first unit → nothing reachable.
    assert_eq!(
        sorted_col0(
            &mut g,
            "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt < e1.amt){1,2} (t) \
             RETURN t.id AS id",
        ),
        Vec::<Value>::new(),
    );
}

/// `WHERE e.amt >= 10` per hop: e3 (amt 5) is excluded, so from `a` the walk can
/// reach b and c but never d.
#[test]
fn per_hop_predicate_filters_var_length_edges() {
    let mut g = weighted_chain();
    let ends = sorted_col0(
        &mut g,
        "MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10]->{1,3}(x) RETURN x.id AS id",
    );
    assert_eq!(
        ends,
        vec![s("b"), s("c")],
        "d is unreachable — e3 (amt 5) is filtered"
    );
}

/// Loosening the threshold to admit every edge restores the full reach (b, c, d).
#[test]
fn per_hop_predicate_admitting_all_edges_reaches_all() {
    let mut g = weighted_chain();
    let ends = sorted_col0(
        &mut g,
        "MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 1]->{1,3}(x) RETURN x.id AS id",
    );
    assert_eq!(ends, vec![s("b"), s("c"), s("d")]);
}

/// The threshold can come from a parameter (the AML motif: host-supplied cutoff).
#[test]
fn per_hop_predicate_reads_a_parameter() {
    let mut g = weighted_chain();
    let mut params = Params::new();
    params.insert("t".into(), super::eval::Val::Num(10.0));
    let mut ends: Vec<Value> = qp(
        &mut g,
        "MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= $t]->{1,3}(x) RETURN x.id AS id",
        params,
    )
    .into_iter()
    .map(|r| r[0].clone())
    .collect();
    ends.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    assert_eq!(ends, vec![s("b"), s("c")]);
}

/// Inline property equality is a per-hop predicate too: `{amt:20}` keeps only the
/// single edge whose amount is 20 (b→c), so from `a` no walk qualifies (a→b is 10).
#[test]
fn per_hop_inline_property_filters_edges() {
    let mut g = weighted_chain();
    // Only b→c has amt 20, but you can't reach b via a qualifying edge, so empty.
    let from_a = rows(
        &mut g,
        "MATCH (a:N {id:'a'})-[:R {amt:20.0}]->{1,3}(x) RETURN x.id AS id",
    );
    assert!(from_a.is_empty());
    // From b, the first hop b→c (amt 20) qualifies; c→d (amt 5) does not.
    let ends = sorted_col0(
        &mut g,
        "MATCH (b:N {id:'b'})-[:R {amt:20.0}]->{1,3}(x) RETURN x.id AS id",
    );
    assert_eq!(ends, vec![s("c")]);
}

/// The per-hop predicate composes with a bound path variable: the bound Path
/// contains only qualifying edges.
#[test]
fn per_hop_predicate_with_path_binding() {
    let mut g = weighted_chain();
    let r = rows(
        &mut g,
        "MATCH p = (a:N {id:'a'})-[e:R WHERE e.amt >= 10]->{1,3}(x) \
         RETURN path_length(p) AS len ORDER BY len",
    );
    // a-b (len 1) and a-b-c (len 2); a-b-c-d blocked at e3.
    assert_eq!(r, vec![vec![n(1.0)], vec![n(2.0)]]);
}

/// The per-hop predicate may reference an outer bound variable (correlation).
#[test]
fn per_hop_predicate_references_outer_variable() {
    let mut g = weighted_chain();
    // a.id = 'a'; a nonsensical-but-valid correlation: keep edges whose amt > 8
    // only when the seed is 'a'. Exercises outer-var visibility inside the filter.
    let ends = sorted_col0(
        &mut g,
        "MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10 AND a.id = 'a']->{1,3}(x) RETURN x.id AS id",
    );
    assert_eq!(ends, vec![s("b"), s("c")]);
}

// ---------------------------------------------------------------------------
// Bare `ALL` selector — the ISO default (every matching path), a synonym for
// writing no selector at all.
// ---------------------------------------------------------------------------

/// `ALL (a)-[]->{1,3}(x)` enumerates exactly what the bare pattern does.
#[test]
fn bare_all_selector_matches_default_enumeration() {
    let mut g = triangle_tail();
    let with_all = rows(
        &mut g,
        "MATCH ALL (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id ORDER BY x.id",
    );
    let bare = rows(
        &mut g,
        "MATCH (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id ORDER BY x.id",
    );
    assert_eq!(with_all, bare);
    assert!(!with_all.is_empty());
}

/// `ALL p = …` binds each walk as a Path, exactly like the bare path binding.
#[test]
fn bare_all_selector_binds_paths() {
    let mut g = triangle_tail();
    let r = rows(
        &mut g,
        "MATCH p = ALL (a:N {id:'a'})-[:R]->{1,3}(x) RETURN path_length(p) AS len ORDER BY len",
    );
    assert_eq!(
        r,
        vec![vec![n(1.0)], vec![n(1.0)], vec![n(2.0)], vec![n(3.0)]]
    );
}

/// `ALL` composes with a path mode: `ALL SIMPLE` keeps only the closing cycle
/// back to the seed.
#[test]
fn bare_all_selector_composes_with_mode() {
    let mut g = triangle_tail();
    let r = rows(
        &mut g,
        "MATCH p = ALL SIMPLE (a:N {id:'a'})-[:R]->{1,3}(a) RETURN path_length(p) AS len",
    );
    assert_eq!(r, vec![vec![n(3.0)]]);
}

/// `ALL` over a plain fixed pattern is accepted and returns every match.
#[test]
fn bare_all_selector_over_fixed_pattern() {
    let mut g = triangle_tail();
    let r = rows(
        &mut g,
        "MATCH ALL (a:N {id:'a'})-[:R]->(x) RETURN x.id AS id ORDER BY x.id",
    );
    assert_eq!(r, vec![vec![s("b")], vec![s("d")]]);
}

// ---------------------------------------------------------------------------
// Regression: a numeric-list property is stored in a typed `Column::Vec`. Reading
// it through GQL must reconstruct the list — the scalar read path `prop_of` (in
// eval.rs) matches `Column` variants DIRECTLY and originally lacked a `Vec` arm,
// so `RETURN n.h` came back NULL (while `props.value_id` handled it). These cover
// that path for node + edge props, subscript, list functions, and WHERE.
// ---------------------------------------------------------------------------

fn vector_props() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"name":"a","h":[1.0,2.0,3.0]}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"name":"b","h":[4.0,5.0,6.0]}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{"w":[7.0,8.0]}}"#,
    ])
}

#[test]
fn vector_column_returns_the_whole_list_through_gql() {
    let mut g = vector_props();
    // `RETURN n.h` — the exact shape that regressed to NULL.
    assert_eq!(
        rows(&mut g, "MATCH (n:N {name:'a'}) RETURN n.h AS h"),
        vec![vec![Value::List(vec![n(1.0), n(2.0), n(3.0)])]]
    );
    // An edge vector property goes through the same `prop_of` (Val::Edge branch).
    assert_eq!(
        rows(&mut g, "MATCH (a)-[e:R]->(b) RETURN e.w AS w"),
        vec![vec![Value::List(vec![n(7.0), n(8.0)])]]
    );
}

#[test]
fn vector_column_subscript_and_list_fns_through_gql() {
    let mut g = vector_props();
    // Subscript reads one element (0-based) off the reconstructed list.
    assert_eq!(
        rows(&mut g, "MATCH (n:N {name:'a'}) RETURN n.h[1] AS x"),
        vec![vec![n(2.0)]]
    );
    // `size` over the vector.
    assert_eq!(
        rows(&mut g, "MATCH (n:N {name:'b'}) RETURN size(n.h) AS s"),
        vec![vec![n(3.0)]]
    );
    // A WHERE predicate over the vector filters, and both rows carry a 3-vector.
    let mut names = rows(
        &mut g,
        "MATCH (n:N) WHERE size(n.h) = 3 RETURN n.name AS name",
    );
    names.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    assert_eq!(names, vec![vec![s("a")], vec![s("b")]]);
}

#[test]
fn vector_column_null_when_absent_through_gql() {
    // A vertex without the vector key reads NULL (present-gating, not the value).
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"name":"a","h":[1.0,2.0]}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"name":"b"}}"#,
    ]);
    assert_eq!(
        rows(&mut g, "MATCH (n:N {name:'b'}) RETURN n.h AS h"),
        vec![vec![Value::Null]]
    );
}

// ---------------------------------------------------------------------------
// Bare `ANY` selector (one arbitrary path per endpoint) and `SHORTEST k [GROUP]`
// (the k shortest paths / the k smallest length-groups). Fixture: to `d` there
// is one length-1 path (a→d) and two length-2 paths (a→b→d, a→c→d).
// ---------------------------------------------------------------------------

fn multi_length_to_d() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"d","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e3","from":"b","to":"d","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e4","from":"a","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e5","from":"c","to":"d","labels":["R"],"properties":{}}"#,
    ])
}

/// Bare `ANY` keeps one path per endpoint, so `->*` from `a` yields one row each
/// for the reachable endpoints (a itself via the zero-length path, b, c, d).
#[test]
fn bare_any_one_path_per_endpoint() {
    let mut g = multi_length_to_d();
    let ends = sorted_col0(
        &mut g,
        "MATCH ANY (a:N {id:'a'})-[:R]->*(x) RETURN x.id AS id",
    );
    assert_eq!(ends, vec![s("a"), s("b"), s("c"), s("d")]);
}

/// `ANY` over a bounded quantifier drops the zero-length self path (min 1); every
/// endpoint still appears exactly once.
#[test]
fn bare_any_dedups_over_bounded_quantifier() {
    let mut g = multi_length_to_d();
    let ends = sorted_col0(
        &mut g,
        "MATCH ANY (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id",
    );
    assert_eq!(ends, vec![s("b"), s("c"), s("d")], "one row per endpoint");
}

/// `ANY p = …` binds a single Path per endpoint; to `d` the witness is the first
/// walk discovered (the direct length-1 edge, first in adjacency order).
#[test]
fn bare_any_binds_one_path_to_d() {
    let mut g = multi_length_to_d();
    let r = rows(
        &mut g,
        "MATCH p = ANY (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    assert_eq!(
        r,
        vec![vec![n(1.0)]],
        "exactly one path, the shortest witness"
    );
}

/// `SHORTEST 1` to `d` keeps a single (shortest, length-1) path.
#[test]
fn shortest_1_keeps_one_shortest_path() {
    let mut g = multi_length_to_d();
    let r = rows(
        &mut g,
        "MATCH p = SHORTEST 1 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    assert_eq!(r, vec![vec![n(1.0)]]);
}

/// `SHORTEST 2` to `d` keeps the two shortest paths by (length, discovery): the
/// length-1 direct edge, then the first length-2 path.
#[test]
fn shortest_2_keeps_two_shortest_paths() {
    let mut g = multi_length_to_d();
    let r = rows(
        &mut g,
        "MATCH p = SHORTEST 2 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) \
         RETURN path_length(p) AS len ORDER BY len",
    );
    assert_eq!(r, vec![vec![n(1.0)], vec![n(2.0)]]);
}

/// `SHORTEST 2 GROUP` to `d` keeps EVERY path in the two smallest length groups:
/// the one length-1 path and both length-2 paths (three rows).
#[test]
fn shortest_2_group_keeps_all_in_two_length_groups() {
    let mut g = multi_length_to_d();
    let r = rows(
        &mut g,
        "MATCH p = SHORTEST 2 GROUP (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) \
         RETURN path_length(p) AS len ORDER BY len",
    );
    assert_eq!(r, vec![vec![n(1.0)], vec![n(2.0)], vec![n(2.0)]]);
}

/// `SHORTEST 1 GROUP` keeps every path of the single smallest length — here just
/// the one length-1 path, identical to what `ALL SHORTEST` returns.
#[test]
fn shortest_1_group_matches_all_shortest() {
    let mut g = multi_length_to_d();
    let grp = rows(
        &mut g,
        "MATCH p = SHORTEST 1 GROUP (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    let all = rows(
        &mut g,
        "MATCH p = ALL SHORTEST (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    assert_eq!(grp, vec![vec![n(1.0)]]);
    assert_eq!(grp, all);
}

/// `GROUPS` is accepted as a synonym for `GROUP`.
#[test]
fn shortest_k_groups_synonym() {
    let mut g = multi_length_to_d();
    let a = rows(
        &mut g,
        "MATCH (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN count(*) AS c",
    );
    let _ = a;
    let r = rows(
        &mut g,
        "MATCH p = SHORTEST 2 GROUPS (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    assert_eq!(r.len(), 3);
}

/// `SHORTEST k` clamps to however many paths exist (k larger than the path count).
#[test]
fn shortest_k_clamps_to_available_paths() {
    let mut g = multi_length_to_d();
    let r = rows(
        &mut g,
        "MATCH p = SHORTEST 10 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len",
    );
    assert_eq!(r.len(), 3, "only three paths a..d exist");
}

/// `SHORTEST k` composes with a per-hop predicate (it enumerates trails, so the
/// filter applies): excluding the direct edge leaves only the two length-2 paths.
#[test]
fn shortest_k_with_per_hop_predicate() {
    // Weight the direct a→d edge so a predicate can exclude it.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"d","labels":["R"],"properties":{"w":100.0}}"#,
        r#"{"type":"edge","id":"e2","from":"a","to":"b","labels":["R"],"properties":{"w":1.0}}"#,
        r#"{"type":"edge","id":"e3","from":"b","to":"d","labels":["R"],"properties":{"w":1.0}}"#,
        r#"{"type":"edge","id":"e4","from":"a","to":"c","labels":["R"],"properties":{"w":1.0}}"#,
        r#"{"type":"edge","id":"e5","from":"c","to":"d","labels":["R"],"properties":{"w":1.0}}"#,
    ]);
    let r = rows(
        &mut g,
        "MATCH p = SHORTEST 5 (a:N {id:'a'})-[e:R WHERE e.w < 10]->*(x:N {id:'d'}) \
         RETURN path_length(p) AS len ORDER BY len",
    );
    assert_eq!(
        r,
        vec![vec![n(2.0)], vec![n(2.0)]],
        "the length-1 direct edge is filtered"
    );
}

/// SHORTEST k parse/shape rejections.
#[test]
fn shortest_k_rejections() {
    assert!(parse("MATCH SHORTEST (a)-[]->*(b) RETURN b").is_err()); // needs a count
    assert!(parse("MATCH SHORTEST 0 (a)-[]->*(b) RETURN b").is_err()); // k >= 1
    assert!(parse("MATCH SHORTEST 3 (a)-[]->*(b) RETURN b").is_ok());
    assert!(parse("MATCH SHORTEST 3 GROUP (a)-[]->*(b) RETURN b").is_ok());
    assert!(parse("MATCH SHORTEST 3 GROUPS (a)-[]->*(b) RETURN b").is_ok());
    assert!(parse("MATCH ANY (a)-[]->*(b) RETURN b").is_ok()); // bare ANY now supported
                                                               // Selectors still need a single var-length segment.
    assert!(parse("MATCH SHORTEST 2 (a)-[]->(b) RETURN b").is_err());
    assert!(parse("MATCH ANY (a)-[]->(b) RETURN b").is_err());
}

/// The unsupported selector shapes fail to parse with a pointed message.
#[test]
fn shortest_unsupported_shapes_rejected() {
    assert!(parse("MATCH (a)-[]->*(b) RETURN b").is_ok()); // no selector: still fine
    assert!(parse("MATCH ALL SHORTEST (a)-[]->*(b) RETURN b").is_ok()); // now supported
    assert!(parse("MATCH ALL (a)-[]->*(b) RETURN b").is_ok()); // bare ALL = default selector
    assert!(parse("MATCH ANY (a)-[]->*(b) RETURN b").is_ok()); // bare ANY now supported
    assert!(parse("MATCH SHORTEST 2 (a)-[]->*(b) RETURN b").is_ok()); // SHORTEST k now supported
    assert!(parse("MATCH SHORTEST (a)-[]->*(b) RETURN b").is_err()); // …but needs a count
                                                                     // A shortest selector needs a single variable-length segment.
    assert!(parse("MATCH ANY SHORTEST (a)-[]->(b) RETURN b").is_err());
    assert!(parse("MATCH ANY SHORTEST (a)-[]->*(b)-[]->*(c) RETURN c").is_err());
    // min > 1 is not the shortest semantics yet.
    assert!(parse("MATCH ANY SHORTEST (a)-[]->{2,4}(b) RETURN b").is_err());
    // A named path over a *fixed*-length hop needs a selector; a single
    // var-length segment binds the path directly (see `bare_path_binds_every_walk`).
    assert!(parse("MATCH p = (a)-[]->(b) RETURN p").is_err());
    assert!(parse("MATCH p = (a)-[]->{1,3}(b) RETURN p").is_ok());
    assert!(parse("MATCH p = SIMPLE (a)-[]->{1,3}(b) RETURN p").is_ok());
}

/// A bare path variable over a single quantified segment (no selector) binds
/// EVERY walk under the pattern's mode as a full Path — the `all_walk` driver.
/// Triangle a→b→c→a plus a→d: from `a`, TRAIL (default) yields four walks up to
/// length 3, and SIMPLE back to `a` keeps only the closing cycle.
#[test]
fn bare_path_binds_every_walk() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"node","id":"d","labels":["N"],"properties":{"id":"d"}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"c","to":"a","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"d","labels":["R"],"properties":{}}"#,
    ];
    let mut g = ndjson::decode(&lines.join("\n")).unwrap();

    // TRAIL (default): a-b, a-b-c, a-b-c-a, a-d — four distinct walks.
    let r = rows(
        &mut g,
        "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN x.id AS id ORDER BY path_length(p), id",
    );
    assert_eq!(r.len(), 4);
    let ends: Vec<&Value> = r.iter().map(|row| &row[0]).collect();
    assert_eq!(ends, vec![&s("b"), &s("d"), &s("c"), &s("a")]);

    // The endpoint of a-b-c-a is the seed again (the trail closes the cycle).
    let lens = rows(
        &mut g,
        "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN path_length(p) AS len ORDER BY len",
    );
    assert_eq!(
        lens,
        vec![vec![n(1.0)], vec![n(1.0)], vec![n(2.0)], vec![n(3.0)]]
    );

    // SIMPLE p back to the seed keeps only the closing cycle a-b-c-a.
    let cyc = rows(
        &mut g,
        "MATCH p = SIMPLE (a:N {id:'a'})-[:R]->{1,3}(a) RETURN path_length(p) AS len",
    );
    assert_eq!(cyc, vec![vec![n(3.0)]]);
}

/// ISO GQL path functions over a bound path: `path_length`/`length` (hops),
/// `nodes`/`edges` (element lists), `elements` (interleaved).
#[test]
fn path_accessor_functions() {
    let mut g = modern();

    // marko→josh→ripple is the shortest marko→ripple path (2 hops).
    let r = rows(
        &mut g,
        "MATCH p = ANY SHORTEST (a)-[]->*(b) WHERE a.name='marko' AND b.name='ripple' \
         RETURN path_length(p) AS len, length(p) AS len2, \
                nodes(p) AS ns, edges(p) AS es, elements(p) AS el",
    );
    assert_eq!(r.len(), 1);
    let row = &r[0];

    assert_eq!(row[0], Value::Num(2.0)); // path_length
    assert_eq!(row[1], Value::Num(2.0)); // length synonym

    // nodes(p): the three vertices as rich element maps (marko, josh, ripple).
    let Value::List(ns) = &row[2] else {
        panic!("nodes(p) should be a list, got {:?}", row[2]);
    };
    assert_eq!(ns.len(), 3);
    let node_id = |v: &Value| match v {
        Value::Map(m) => match m.iter().find(|(k, _)| &**k == "id").map(|(_, x)| x) {
            Some(Value::Str(s)) => s.to_string(),
            _ => panic!("no id"),
        },
        _ => panic!("not a node map"),
    };
    // marko → josh → ripple, by element id (see `crate::fixtures`).
    assert_eq!(
        ns.iter().map(node_id).collect::<Vec<_>>(),
        vec!["1", "4", "5"]
    );

    // edges(p): the two edges.
    let Value::List(es) = &row[3] else {
        panic!("edges(p) should be a list");
    };
    assert_eq!(es.len(), 2);

    // elements(p): interleaved node, edge, node, edge, node → 5 items.
    let Value::List(el) = &row[4] else {
        panic!("elements(p) should be a list");
    };
    assert_eq!(el.len(), 5);
}

/// ISO GQL named procedure CALL invoking the built-in graph algorithms, with
/// YIELD, aliasing, config, and downstream clauses over the yielded columns.
#[test]
fn call_named_procedure_algorithms() {
    let mut g = modern();

    // pagerank: lop has the most incoming edges → the top score. `node` is a live
    // vertex handle, so `node.name` reads its property directly.
    let top = rows(
        &mut g,
        "CALL pagerank() YIELD node, score RETURN node.name AS n ORDER BY score DESC, n LIMIT 1",
    );
    assert_eq!(top, vec![vec![s("lop")]]);

    // degree: one row per vertex (YIELD-less binds node + degree automatically).
    let degs = rows(&mut g, "CALL degree() RETURN node.name AS n, degree");
    assert_eq!(degs.len(), 6);

    // YIELD aliasing, then ISO filtering via WITH … WHERE over the yielded column.
    let hi = rows(
        &mut g,
        "CALL degree() YIELD node AS v, degree AS d WITH v, d WHERE d >= 3 RETURN v.name AS n ORDER BY n",
    );
    // Default degree is out-degree; only marko has out-degree ≥ 3.
    assert_eq!(hi, vec![vec![s("marko")]]);

    // config: writeProperty mutates the graph; the property is then readable.
    let _ = rows(
        &mut g,
        "CALL degree({writeProperty: 'deg'}) YIELD node RETURN node",
    );
    let read = rows(&mut g, "MATCH (n) WHERE n.name = 'marko' RETURN n.deg AS d");
    assert_eq!(read, vec![vec![Value::Num(3.0)]]); // marko has out-degree 3
}

/// CALL parse-level behavior: unknown procedure faults; the inline-subquery form
/// is rejected with a pointed message (deferred).
#[test]
fn call_unknown_procedure_faults() {
    let mut g = modern();
    assert!(parse("CALL bogus() YIELD x RETURN x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .is_err());
}

/// ISO GQL inline subquery CALL: correlated lateral join, row duplication,
/// OPTIONAL null-fill vs drop, and scope isolation.
#[test]
fn call_inline_subquery() {
    // For each person, count what they created (marko 1, vadas 0, josh 2, peter 1).
    let mut g = modern();
    let counts = rows(
        &mut g,
        "MATCH (p:Person) \
         CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN count(w) AS created } \
         RETURN p.name AS name, created ORDER BY name",
    );
    assert_eq!(
        counts,
        vec![
            vec![s("josh"), Value::Num(2.0)],
            vec![s("marko"), Value::Num(1.0)],
            vec![s("peter"), Value::Num(1.0)],
            vec![s("vadas"), Value::Num(0.0)],
        ]
    );

    // Row duplication: marko knows vadas + josh → the outer row fans to two.
    let friends = rows(
        &mut g,
        "MATCH (p:Person {name: 'marko'}) \
         CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS friend } \
         RETURN friend ORDER BY friend",
    );
    assert_eq!(friends, vec![vec![s("josh")], vec![s("vadas")]]);

    // Non-OPTIONAL empty subquery drops the outer row (vadas created nothing).
    let dropped = rows(
        &mut g,
        "MATCH (p:Person) \
         CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN w.name AS thing } \
         RETURN p.name AS name ORDER BY name",
    );
    let names: Vec<String> = dropped
        .iter()
        .map(|r| match &r[0] {
            Value::Str(s) => s.to_string(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(names, vec!["josh", "josh", "marko", "peter"]);
}

/// Scope isolation: an unscoped outer variable is invisible inside the subquery.
#[test]
fn call_inline_scope_isolation() {
    let mut g = modern();
    // `p` is NOT imported → the inner `MATCH (p)` is a fresh unbound pattern over
    // all 6 vertices, not the outer marko.
    let total = rows(
        &mut g,
        "MATCH (p:Person {name: 'marko'}) \
         CALL () { MATCH (n) RETURN count(n) AS total } \
         RETURN total",
    );
    assert_eq!(total, vec![vec![Value::Num(6.0)]]);
}

/// Decorrelation safety: a non-aggregating inline CALL that references an UNSCOPED
/// outer var must stay correlated (the guard falls back). The empty `()` scope
/// isolates `a`, so `c = a` compares against NULL → no rows → the outer row is
/// dropped. If wrongly decorrelated (a made visible), it would match and return
/// rows — so a non-empty result here would be a correctness bug.
#[test]
fn decorrelate_respects_scope_isolation() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (a:Person) WHERE a.name = 'marko' \
         CALL () { MATCH (b:Person)-[:KNOWS]->(c) WHERE c = a RETURN c.name AS cn } \
         RETURN cn",
    );
    assert!(
        r.is_empty(),
        "unscoped `a` must be NULL inside the subquery, got {r:?}"
    );
}

/// The non-aggregating correlated CALL decorrelates to an order-identical flat
/// form; this pins the exact rows (the same assertion works correlated or flat).
#[test]
fn decorrelate_non_agg_rows_unchanged() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person {name: 'marko'}) \
         CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS friend } \
         RETURN friend ORDER BY friend",
    );
    assert_eq!(r, vec![vec![s("josh")], vec![s("vadas")]]);
}

/// Non-agg decorrelation over MULTIPLE start vertices (not anchored to one) —
/// the case that would expose a bad flattening of the correlation join.
#[test]
fn decorrelate_non_agg_multi_start() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person) \
         CALL (p) { MATCH (p)-[:CREATED]->(w) RETURN w.name AS thing } \
         RETURN p.name AS pn, thing ORDER BY pn, thing",
    );
    // marko→lop, josh→{lop,ripple}, peter→lop; vadas creates nothing (dropped, non-OPTIONAL).
    assert_eq!(
        r,
        vec![
            vec![s("josh"), s("lop")],
            vec![s("josh"), s("ripple")],
            vec![s("marko"), s("lop")],
            vec![s("peter"), s("lop")],
        ]
    );
}

/// Regression: a non-optional MATCH immediately followed by a correlated OPTIONAL
/// MATCH (no barrier between them) must find EVERY start vertex's matches. The
/// OPTIONAL null-fill used to leak stale nulls into the next start binding, where
/// `bind_slot` mistook them for a join conflict and dropped that row's real
/// matches (only the first start — marko — survived).
#[test]
fn optional_match_after_match_no_barrier() {
    let mut g = modern();
    let r = rows(
        &mut g,
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:CREATED]->(w) \
         RETURN p.name AS pn, w.name AS wn ORDER BY pn, wn",
    );
    assert_eq!(
        r,
        vec![
            vec![s("josh"), s("lop")],
            vec![s("josh"), s("ripple")],
            vec![s("marko"), s("lop")],
            vec![s("peter"), s("lop")],
            vec![s("vadas"), Value::Null], // creates nothing → kept with null
        ]
    );
}

// ---------------------------------------------------------------------------
// Temporal component extraction: _year()/_month()/_day()/_hour()/_minute()/_second()
// — the ISO GQL named-function form (NOT SQL `EXTRACT`, NOT Cypher `.year`). A
// string is NOT coerced (faults E_INVALID_VALUE); a temporal lacking the
// requested component faults too; zoned values read their OWN offset (local
// wall clock). Byte-identity with the TS engine is checked in the differential
// suite (packages/native/src/gql-conformance.test.ts).
// ---------------------------------------------------------------------------

#[test]
fn date_part_extracts_from_a_date() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN _year(DATE '2024-03-15') AS y, _month(DATE '2024-03-15') AS mo, \
             _day(DATE '2024-03-15') AS d"
        ),
        vec![vec![n(2024.0), n(3.0), n(15.0)]]
    );
    // A pre-epoch date decomposes correctly (negative epoch-day count).
    assert_eq!(
        rows(
            &mut g,
            "RETURN _year(DATE '1969-12-31') AS y, _month(DATE '1969-12-31') AS mo, \
             _day(DATE '1969-12-31') AS d"
        ),
        vec![vec![n(1969.0), n(12.0), n(31.0)]]
    );
}

#[test]
fn date_part_extracts_date_and_time_fields_from_a_datetime() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN _year(DATETIME '2024-03-15T13:47:09') AS y, \
             _hour(DATETIME '2024-03-15T13:47:09') AS h, \
             _minute(DATETIME '2024-03-15T13:47:09') AS mi, \
             _second(DATETIME '2024-03-15T13:47:09') AS s"
        ),
        vec![vec![n(2024.0), n(13.0), n(47.0), n(9.0)]]
    );
}

#[test]
fn date_part_extracts_time_fields_from_a_local_time() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN _hour(local_time('13:47:09')) AS h, _minute(local_time('13:47:09')) AS mi, \
             _second(local_time('13:47:09')) AS s"
        ),
        vec![vec![n(13.0), n(47.0), n(9.0)]]
    );
}

#[test]
fn date_part_zoned_reads_its_own_offset_wall_clock() {
    // A zoned value's components are its stored-offset wall clock, not UTC.
    // 23:30+05:00 → local hour 23, local day 15 (the UTC instant is 18:30Z).
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN _day(zoned_datetime('2024-03-15T23:30:00+05:00')) AS d, \
             _hour(zoned_datetime('2024-03-15T23:30:00+05:00')) AS h"
        ),
        vec![vec![n(15.0), n(23.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "RETURN _hour(zoned_time('01:15:00+02:00')) AS h, \
             _minute(zoned_time('01:15:00+02:00')) AS mi"
        ),
        vec![vec![n(1.0), n(15.0)]]
    );
}

// ---------------------------------------------------------------------------
// EXPECTED behavior — NOT bugs. lenke is an INSTANT engine, not a civil-time
// (wall-clock) engine: a duration adds elapsed SECONDS to the instant, the
// stored offset is FROZEN (no IANA tz database, so no DST re-derivation), and a
// day is exactly 86_400 s (so `PT24H` and `P1D` are the same operation). These
// tests pin the surprising-to-JS-users consequences so they can't silently
// regress. The fix for civil-time semantics is app-side: store UTC/`Z` and do
// zone/DST conversion at the boundary with a real tz library. See the `temporal`
// MCP guide's "Time zones, DST & instants" section and docs/guides/bitemporal.md.
// The scenario: 8pm on 2026-03-07, the evening BEFORE US "spring forward"
// (2026-03-08 02:00 local, EST -05:00 → EDT -04:00).
// ---------------------------------------------------------------------------

#[test]
fn duration_add_is_elapsed_seconds_pt24h_equals_p1d() {
    // Adding a duration advances the INSTANT by its seconds and keeps the offset
    // frozen — it does NOT spring forward. As UTC the result is 2026-03-09T01:00Z
    // (exactly +24h from the 2026-03-08T01:00Z start), which is correct elapsed
    // time; a tz-aware library would RENDER that same instant as 21:00-04:00
    // (9pm EDT). lenke renders it in the original -05:00, so the wall clock reads
    // 20:00. And because a lenke "day" is 86_400 s, `P1D` == `PT24H` exactly —
    // there is no shorter/longer calendar day across the DST boundary.
    let mut g = ndjson::decode("").unwrap();
    let frozen = Value::Temporal(
        crate::temporal::Temporal::parse("zoned_datetime", "2026-03-08T20:00:00-05:00").unwrap(),
    );
    assert_eq!(
        rows(
            &mut g,
            "RETURN zoned_datetime('2026-03-07T20:00:00-05:00') + duration('PT24H') AS r"
        ),
        vec![vec![frozen.clone()]]
    );
    assert_eq!(
        rows(
            &mut g,
            "RETURN zoned_datetime('2026-03-07T20:00:00-05:00') + duration('P1D') AS r"
        ),
        vec![vec![frozen]],
        "P1D and PT24H are the same operation — a day is 86_400 s, no DST-shortened day"
    );
}

#[test]
fn date_part_hour_reads_frozen_offset_across_a_dst_boundary() {
    // `_hour` reads the value's OWN (frozen) offset, so after crossing spring-
    // forward the hour is 20 (the stale -05:00 wall clock), not 21 (the real EDT
    // -04:00 local hour of the identical instant). A `GROUP BY _hour` over data
    // that crossed a DST edge therefore buckets an hour off — expected, because
    // lenke never re-derived the offset.
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN _hour(zoned_datetime('2026-03-07T20:00:00-05:00') + duration('P1D')) AS h"
        ),
        vec![vec![n(20.0)]] // reality in America/New_York is 21 (9pm EDT)
    );
}

#[test]
fn date_part_day_does_not_roll_across_a_dst_boundary() {
    // The nastier bucketing case: a value near midnight. 23:30 EST + P1D stays
    // 23:30-05:00, so `_day` is 8 — but the identical instant in real EDT is
    // 2026-03-09T00:30-04:00, i.e. day 9. A daily rollup over cross-DST data can
    // put a row on the wrong DATE. Expected: the offset is frozen, so no rollover.
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN _day(zoned_datetime('2026-03-07T23:30:00-05:00') + duration('P1D')) AS d"
        ),
        vec![vec![n(8.0)]] // reality in America/New_York is day 9
    );
}

#[test]
fn date_part_null_in_null_out() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(&mut g, "RETURN _year(null) AS y"),
        vec![vec![Value::Null]]
    );
    // An absent property → null in → null out (no fault), so the row survives.
    rows(&mut g, "INSERT (:H {name: 'x'})");
    assert_eq!(
        rows(&mut g, "MATCH (h:H) RETURN _year(h.hired) AS y"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn date_part_rejects_strings_and_missing_components() {
    use crate::error_codes::ErrorCode::InvalidValue;
    let mut g = ndjson::decode("").unwrap();
    let code = |g: &mut Graph, q: &str| {
        parse(q)
            .unwrap()
            .execute(g, &Params::new())
            .unwrap_err()
            .code
    };

    // A string is NOT coerced — it must be wrapped with date()/local_datetime()/…
    assert_eq!(
        code(&mut g, "RETURN _year('2024-03-15') AS y"),
        InvalidValue
    );
    // A number is not a temporal.
    assert_eq!(code(&mut g, "RETURN _month(5) AS m"), InvalidValue);
    // _year() of a time-only value has no date component.
    assert_eq!(
        code(&mut g, "RETURN _year(local_time('13:47:09')) AS y"),
        InvalidValue
    );
    // _hour() of a date has no time component.
    assert_eq!(
        code(&mut g, "RETURN _hour(DATE '2024-03-15') AS h"),
        InvalidValue
    );
    // a duration carries neither.
    assert_eq!(
        code(&mut g, "RETURN _day(duration('P1Y2M3D')) AS d"),
        InvalidValue
    );
    // The bare (sigil-less) names are NOT functions — date-parts are a lenke
    // extension, so only the `_`-prefixed form resolves.
    assert_eq!(
        code(&mut g, "RETURN year(DATE '2024-03-15') AS y"),
        crate::error_codes::ErrorCode::UnknownFunction
    );
}

#[test]
fn date_part_group_by_year_buckets_rows() {
    // The headline use case: cohort/bucket rows by a calendar component.
    let mut g = ndjson::decode("").unwrap();
    rows(&mut g, "INSERT (:H {hired: DATE '2021-05-01'})");
    rows(&mut g, "INSERT (:H {hired: DATE '2021-11-30'})");
    rows(&mut g, "INSERT (:H {hired: DATE '2023-02-14'})");
    assert_eq!(
        rows(
            &mut g,
            "MATCH (h:H) RETURN _year(h.hired) AS yr, count(*) AS c ORDER BY yr"
        ),
        vec![vec![n(2021.0), n(2.0)], vec![n(2023.0), n(1.0)]]
    );
}

// ---------------------------------------------------------------------------
// PROPERTY_EXISTS(n, key) — ISO presence predicate. The point: it distinguishes
// an ABSENT key from a PRESENT-but-null one, which `n.key IS NOT NULL` cannot
// (null is a first-class stored value in lenke).
// ---------------------------------------------------------------------------

#[test]
fn property_exists_distinguishes_absent_from_present_null() {
    // n1 has {a:1, z:null(stored)}; n2 has {a:2} (no z).
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"n1","labels":["N"],"properties":{"a":1,"z":null}}"#,
            r#"{"type":"node","id":"n2","labels":["N"],"properties":{"a":2}}"#,
        ]
        .join("\n"),
    )
    .unwrap();

    // present key → true; a stored null still EXISTS → true; absent key → false.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:N {a:1}) RETURN property_exists(n, a) AS ha, property_exists(n, z) AS hz, \
             property_exists(n, nope) AS hn",
        ),
        vec![vec![b(true), b(true), b(false)]]
    );
    // n2 has no `z` at all → false, even though n1's `z IS NULL` reads the same.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:N {a:2}) RETURN property_exists(n, z) AS hz"
        ),
        vec![vec![b(false)]]
    );
    // Contrast with `IS NOT NULL`, which CANNOT tell absent from stored-null:
    // both n1.z (stored null) and n2.z (absent) read as null → both false.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:N) RETURN property_exists(n, z) AS present, (n.z IS NOT NULL) AS notnull \
             ORDER BY n.a",
        ),
        vec![vec![b(true), b(false)], vec![b(false), b(false)]]
    );
}

#[test]
fn property_exists_on_edges_and_non_elements() {
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{}}"#,
            r#"{"type":"edge","id":"e","from":"a","to":"b","labels":["R"],"properties":{"w":5.0}}"#,
        ]
        .join("\n"),
    )
    .unwrap();

    assert_eq!(
        rows(
            &mut g,
            "MATCH ()-[e:R]->() RETURN property_exists(e, w) AS hw, property_exists(e, gone) AS hg",
        ),
        vec![vec![b(true), b(false)]]
    );
    // A NULL element (unbound OPTIONAL) → NULL (three-valued), not false.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:N) OPTIONAL MATCH (n)-[:NOSUCH]->(m) RETURN property_exists(m, x) AS hx",
        ),
        vec![vec![Value::Null], vec![Value::Null]]
    );
}

// ---------------------------------------------------------------------------
// IS [NOT] TYPED <type> [NOT NULL] — the ISO value-type predicate. Null conforms
// to any nullable type (Neo4j-verified reading); `NOT NULL` excludes it. Numeric
// split is boundary-inferred (INTEGER = whole number) since lenke has one f64 type.
// ---------------------------------------------------------------------------

#[test]
fn is_typed_scalar_and_numeric_inference() {
    let mut g = ndjson::decode("").unwrap();
    let row = rows(
        &mut g,
        "RETURN 5 IS TYPED INTEGER AS a, 5.5 IS TYPED INTEGER AS b, 5.5 IS TYPED FLOAT AS c, \
         5 IS TYPED FLOAT AS d, 'x' IS TYPED STRING AS e, true IS TYPED BOOL AS f, \
         [1,2] IS TYPED LIST AS h, 5 IS TYPED STRING AS i",
    );
    assert_eq!(
        row,
        vec![vec![
            b(true),  // 5 is a whole number → INTEGER
            b(false), // 5.5 is not whole → not INTEGER
            b(true),  // 5.5 is a number → FLOAT
            b(true),  // 5 is a number → FLOAT (one f64 type: a whole number is both)
            b(true),
            b(true),
            b(true),
            b(false), // 5 is not a STRING
        ]]
    );
}

#[test]
fn is_typed_temporal_and_negation() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN DATE '2020-01-01' IS TYPED DATE AS a, \
             DATE '2020-01-01' IS TYPED LOCAL DATETIME AS b, \
             DATETIME '2020-01-01T00:00:00' IS TYPED LOCAL DATETIME AS c, \
             duration('P1D') IS TYPED DURATION AS d, \
             5 IS NOT TYPED STRING AS e, 5 IS NOT TYPED INTEGER AS f",
        ),
        vec![vec![b(true), b(false), b(true), b(true), b(true), b(false)]]
    );
}

#[test]
fn is_typed_null_conformance_and_not_null() {
    let mut g = ndjson::decode("").unwrap();
    // Null conforms to any nullable type → true; NOT NULL excludes it → false.
    assert_eq!(
        rows(
            &mut g,
            "RETURN null IS TYPED INTEGER AS a, null IS TYPED INTEGER NOT NULL AS b, \
             null IS TYPED STRING AS c, null IS TYPED NULL AS d, null IS TYPED ANY AS e, \
             null IS TYPED ANY NOT NULL AS f",
        ),
        vec![vec![b(true), b(false), b(true), b(true), b(true), b(false)]]
    );
    // A non-null value: NOT NULL makes no difference; ANY always matches.
    assert_eq!(
        rows(
            &mut g,
            "RETURN 5 IS TYPED INTEGER NOT NULL AS a, 5 IS TYPED ANY AS b, 5 IS TYPED NULL AS c",
        ),
        vec![vec![b(true), b(true), b(false)]]
    );
}

#[test]
fn is_typed_rejects_unknown_type() {
    // An unknown type name is a loud parse error, not a silent false.
    assert!(parse("RETURN 5 IS TYPED FROBNICATE AS x").is_err());
}

#[test]
fn is_typed_open_record() {
    // ISO `IS TYPED [ANY] RECORD` — the OPEN record type tests "is this a map".
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN {a: 1} IS TYPED ANY RECORD AS a, {a: 1} IS TYPED RECORD AS b, \
             5 IS TYPED ANY RECORD AS c, [1,2] IS TYPED RECORD AS d, \
             5 IS NOT TYPED RECORD AS e",
        ),
        vec![vec![b(true), b(true), b(false), b(false), b(true)]]
    );
    // NOT NULL / null semantics match the scalar predicate.
    assert_eq!(
        rows(
            &mut g,
            "RETURN null IS TYPED ANY RECORD AS a, null IS TYPED ANY RECORD NOT NULL AS b",
        ),
        vec![vec![b(true), b(false)]]
    );
    // `ANY` without `RECORD` is a parse error.
    assert!(parse("RETURN {a:1} IS TYPED ANY FOO AS x").is_err());
}

#[test]
fn is_typed_closed_record() {
    // ISO `IS TYPED RECORD { field :: type [NOT NULL], … }` — closed on extras,
    // fields nullable/optional unless NOT NULL, with the predicate's INTEGER/FLOAT
    // split and nesting.
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN {a: 1, b: 'x'} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS a, \
             {a: 1} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS b, \
             {a: 1, b: 'x', c: 9} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS c, \
             {a: 1.5} IS TYPED RECORD {a :: INTEGER} AS d, \
             {a: 1.5} IS TYPED RECORD {a :: FLOAT} AS e",
        ),
        // a: exact match. b: `b` nullable-absent → OK. c: extra field → closed → false.
        // d: 1.5 not INTEGER → false. e: 1.5 IS FLOAT → true.
        vec![vec![b(true), b(true), b(false), b(false), b(true)]]
    );
    // NOT NULL fields + nested records.
    assert_eq!(
        rows(
            &mut g,
            "RETURN {} IS TYPED RECORD {a :: INTEGER NOT NULL} AS a, \
             {a: null} IS TYPED RECORD {a :: INTEGER NOT NULL} AS b, \
             {geo: {lat: 1, lng: 2}} IS TYPED RECORD {geo :: RECORD {lat :: INTEGER, lng :: INTEGER}} AS c, \
             {geo: {lat: 'x'}} IS TYPED RECORD {geo :: RECORD {lat :: INTEGER, lng :: INTEGER}} AS d",
        ),
        // a: NOT NULL absent → false. b: NOT NULL null → false. c: nested match → true.
        // d: nested wrong type + extra-missing → false.
        vec![vec![b(false), b(false), b(true), b(false)]]
    );
}

// ---------------------------------------------------------------------------
// Graph-element predicates: IS DIRECTED, IS SOURCE/DESTINATION OF, ALL_DIFFERENT,
// SAME — plus the `!` unary-not operator (tight-binding).
// ---------------------------------------------------------------------------

fn diamond() -> Graph {
    // a -> b, a -> c (so a is SOURCE of both; b,c are DESTINATIONs).
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{"id":"c"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"a","to":"c","labels":["R"],"properties":{}}"#,
    ])
}

#[test]
fn graph_pred_is_directed() {
    let mut g = diamond();
    // Every lenke edge is directed → true; IS NOT DIRECTED → false.
    assert_eq!(
        rows(
            &mut g,
            "MATCH ()-[e:R]->() RETURN e IS DIRECTED AS d, e IS NOT DIRECTED AS nd ORDER BY d LIMIT 1",
        ),
        vec![vec![b(true), b(false)]]
    );
}

#[test]
fn graph_pred_source_destination_of() {
    let mut g = diamond();
    // a is the SOURCE of e1(a->b); b is the DESTINATION; b is NOT the source.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {id:'a'})-[e:R]->(b:N {id:'b'}) \
             RETURN a IS SOURCE OF e AS asrc, b IS DESTINATION OF e AS bdst, \
             b IS SOURCE OF e AS bsrc, a IS NOT DESTINATION OF e AS anotdst",
        ),
        vec![vec![b(true), b(true), b(false), b(true)]]
    );
}

#[test]
fn graph_pred_all_different_and_same() {
    let mut g = diamond();
    // In (a)-->(b), a and b are different elements; SAME(a,a) is true.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {id:'a'})-[e:R]->(b:N {id:'b'}) \
             RETURN ALL_DIFFERENT(a, b) AS diff, SAME(a, a) AS same_aa, SAME(a, b) AS same_ab",
        ),
        vec![vec![b(true), b(true), b(false)]]
    );
    // ALL_DIFFERENT is false when two operands are the same element.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {id:'a'}) RETURN ALL_DIFFERENT(a, a) AS d"
        ),
        vec![vec![b(false)]]
    );
}

#[test]
fn graph_pred_null_is_three_valued() {
    let mut g = diamond();
    // A NULL element (unmatched OPTIONAL) → NULL, not false.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:N {id:'a'}) OPTIONAL MATCH (a)-[:NOSUCH]->(m) \
             RETURN m IS DIRECTED AS d, ALL_DIFFERENT(a, m) AS ad",
        ),
        vec![vec![Value::Null, Value::Null]]
    );
}

#[test]
fn bang_unary_not_binds_tightly() {
    let mut g = ndjson::decode("").unwrap();
    // `!` is a TIGHT unary-not: `!(1=2)` true; `!true` false; and it binds harder
    // than comparison, so `!(1=2) = true` parses as `((!(1=2)) = true)` = true.
    assert_eq!(
        rows(
            &mut g,
            "RETURN !(1=2) AS a, !true AS b, (!(1=2) = true) AS c"
        ),
        vec![vec![b(true), b(false), b(true)]]
    );
}

// ---------------------------------------------------------------------------
// Case sensitivity (regression guard). ISO GQL is case-SENSITIVE for labels,
// edge types, property keys, and string values — `:Person` != `:person`,
// `:CREATED` != `:created`. (A latent test bug once used `:created` against a
// `CREATED` edge and silently matched nothing.) Keywords stay case-INSENSITIVE.
// ---------------------------------------------------------------------------

#[test]
fn labels_and_edge_types_are_case_sensitive() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["Person"],"properties":{"name":"marko"}}"#,
        r#"{"type":"node","id":"b","labels":["Software"],"properties":{"name":"lop"}}"#,
        r#"{"type":"edge","id":"e","from":"a","to":"b","labels":["CREATED"],"properties":{}}"#,
    ]);

    // Exact-case label matches; a different-case label matches nothing.
    assert_eq!(
        rows(&mut g, "MATCH (n:Person) RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:person) RETURN count(*) AS c"),
        vec![vec![n(0.0)]]
    );
    // Exact-case edge type matches; the lowercase form matches nothing.
    assert_eq!(
        rows(&mut g, "MATCH ()-[e:CREATED]->() RETURN count(*) AS c"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH ()-[e:created]->() RETURN count(*) AS c"),
        vec![vec![n(0.0)]]
    );
}

#[test]
fn property_keys_and_values_are_case_sensitive() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"name":"marko","Name":"OTHER"}}"#,
    ]);

    // `name` and `Name` are distinct keys.
    assert_eq!(
        rows(&mut g, "MATCH (n:N) RETURN n.name AS a, n.Name AS b"),
        vec![vec![s("marko"), s("OTHER")]]
    );
    // String value equality is case-sensitive.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:N) WHERE n.name = 'marko' RETURN count(*) AS c"
        ),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:N) WHERE n.name = 'MARKO' RETURN count(*) AS c"
        ),
        vec![vec![n(0.0)]]
    );
    // But structural keywords remain case-INSENSITIVE (match/return/where).
    assert_eq!(
        rows(
            &mut g,
            "match (n:N) where n.name = 'marko' return count(*) AS c"
        ),
        vec![vec![n(1.0)]]
    );
}

#[test]
fn trim_char_set_two_arg() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN btrim('xxhixx','x') AS b, ltrim('xxhixx','x') AS l, rtrim('xxhixx','x') AS r"
        ),
        vec![vec![s("hi"), s("hixx"), s("xxhi")]]
    );
    // multi-char set + whitespace default still works.
    assert_eq!(
        rows(
            &mut g,
            "RETURN btrim('xyxhixyx','xy') AS a, trim('  hi  ') AS b, ltrim('  hi') AS c"
        ),
        vec![vec![s("hi"), s("hi"), s("hi")]]
    );
}

#[test]
fn trim_sql_spec_form() {
    let mut g = ndjson::decode("").unwrap();
    assert_eq!(
        rows(
            &mut g,
            "RETURN TRIM('  hi  ') AS a, TRIM(BOTH FROM '  hi  ') AS b, \
             TRIM(LEADING FROM '  hi') AS c, TRIM(TRAILING FROM 'hi  ') AS d, \
             TRIM(LEADING 'x' FROM 'xxhi') AS e, TRIM('x' FROM 'xxhixx') AS f, \
             TRIM(TRAILING 'x' FROM 'hixx') AS gg",
        ),
        vec![vec![
            s("hi"),
            s("hi"),
            s("hi"),
            s("hi"),
            s("hi"),
            s("hi"),
            s("hi"),
        ]]
    );
}

// ---------------------------------------------------------------------------
// Explicit GROUP BY (ISO — on the RETURN statement). Drives grouping (forces it
// on, even without an aggregate); an empty GROUP BY keeps implicit grouping.
// ---------------------------------------------------------------------------

fn hires() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["E"],"properties":{"dept":"eng","name":"a","sal":100.0}}"#,
        r#"{"type":"node","id":"b","labels":["E"],"properties":{"dept":"eng","name":"b","sal":200.0}}"#,
        r#"{"type":"node","id":"c","labels":["E"],"properties":{"dept":"sales","name":"c","sal":50.0}}"#,
    ])
}

#[test]
fn group_by_with_aggregate() {
    let mut g = hires();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (e:E) RETURN e.dept AS d, count(*) AS c GROUP BY e.dept ORDER BY d"
        ),
        vec![vec![s("eng"), n(2.0)], vec![s("sales"), n(1.0)]]
    );
}

#[test]
fn group_by_non_returned_key() {
    // `RETURN count(*) GROUP BY e.dept` — group by a key that is NOT returned.
    // Implicit grouping can't express this (there'd be no key); explicit GROUP BY
    // must → one count per dept.
    let mut g = hires();
    let mut out = rows(&mut g, "MATCH (e:E) RETURN count(*) AS c GROUP BY e.dept");
    out.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    assert_eq!(out, vec![vec![n(1.0)], vec![n(2.0)]]);
}

#[test]
fn group_by_without_aggregate_is_distinct() {
    // `RETURN e.dept GROUP BY e.dept` — no aggregate → one row per distinct dept.
    let mut g = hires();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (e:E) RETURN e.dept AS d GROUP BY e.dept ORDER BY d"
        ),
        vec![vec![s("eng")], vec![s("sales")]]
    );
}

#[test]
fn exists_multi_match() {
    // ISO `EXISTS { MATCH … MATCH … }` — a conjunction of MATCH blocks, each with
    // its own optional WHERE. True iff all jointly match.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["M"],"properties":{"id":"b"}}"#,
    ]);
    // Both an N and an M exist → true; an N and a (missing) Z → false.
    assert_eq!(
        rows(&mut g, "RETURN EXISTS { MATCH (x:N) MATCH (y:M) } AS e"),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN EXISTS { MATCH (x:N) MATCH (y:Z) } AS e"),
        vec![vec![b(false)]]
    );
    // Per-block WHERE, ANDed: N with id 'a' AND M with id 'b' → true; wrong id → false.
    assert_eq!(
        rows(
            &mut g,
            "RETURN EXISTS { MATCH (x:N) WHERE x.id='a' MATCH (y:M) WHERE y.id='b' } AS e",
        ),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "RETURN EXISTS { MATCH (x:N) WHERE x.id='a' MATCH (y:M) WHERE y.id='nope' } AS e",
        ),
        vec![vec![b(false)]]
    );
    // COUNT { } over multi-MATCH counts the joint solutions (1 N × 1 M = 1).
    assert_eq!(
        rows(&mut g, "RETURN COUNT { MATCH (x:N) MATCH (y:M) } AS c"),
        vec![vec![n(1.0)]]
    );
}

/// Fixture for the VALUE scalar-subquery tests: alice→bob (KNOWS), carol with no
/// out-edges, and dave→erin + dave→frank (KNOWS) so `dave` has two neighbours.
fn value_graph() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"alice","labels":["Person"],"properties":{"id":"alice","name":"Alice"}}"#,
        r#"{"type":"node","id":"bob","labels":["Person"],"properties":{"id":"bob","name":"Bob"}}"#,
        r#"{"type":"node","id":"carol","labels":["Person"],"properties":{"id":"carol","name":"Carol"}}"#,
        r#"{"type":"node","id":"dave","labels":["Person"],"properties":{"id":"dave","name":"Dave"}}"#,
        r#"{"type":"node","id":"erin","labels":["Person"],"properties":{"id":"erin","name":"Erin"}}"#,
        r#"{"type":"node","id":"frank","labels":["Person"],"properties":{"id":"frank","name":"Frank"}}"#,
        r#"{"type":"edge","id":"k1","from":"alice","to":"bob","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","id":"k2","from":"dave","to":"erin","labels":["KNOWS"],"properties":{}}"#,
        r#"{"type":"edge","id":"k3","from":"dave","to":"frank","labels":["KNOWS"],"properties":{}}"#,
    ])
}

#[test]
fn value_subquery_correlated_scalar() {
    // Correlated single-row VALUE: the one neighbour's name, NULL when there is
    // none. `alice`→`bob` yields "Bob"; `carol` has no KNOWS edge → NULL.
    let mut g = value_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='alice' \
             RETURN VALUE { MATCH (a)-[:KNOWS]->(b) RETURN b.name } AS friend",
        ),
        vec![vec![s("Bob")]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='carol' \
             RETURN VALUE { MATCH (a)-[:KNOWS]->(b) RETURN b.name } AS friend",
        ),
        vec![vec![Value::Null]]
    );
}

#[test]
fn value_subquery_aggregate_folds_group() {
    // An aggregate RETURN folds the whole matched group to one value regardless
    // of row count — no cardinality error. Six people; dave has two neighbours.
    let mut g = value_graph();
    assert_eq!(
        rows(
            &mut g,
            "RETURN VALUE { MATCH (n:Person) RETURN count(*) } AS c"
        ),
        vec![vec![n(6.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='dave' \
             RETURN VALUE { MATCH (a)-[:KNOWS]->(b) RETURN count(*) } AS deg",
        ),
        vec![vec![n(2.0)]]
    );
    // count() over zero matches is 0 (the aggregate's empty answer), not NULL.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='carol' \
             RETURN VALUE { MATCH (a)-[:KNOWS]->(b) RETURN count(*) } AS deg",
        ),
        vec![vec![n(0.0)]]
    );
}

#[test]
fn value_subquery_multi_row_is_cardinality_error() {
    // A non-aggregate RETURN that matches >1 row is an ISO cardinality violation:
    // loud error, never a silent first-of-many. `dave` has two neighbours.
    let mut g = value_graph();
    assert!(exec_err(
        &mut g,
        "MATCH (a:Person) WHERE a.id='dave' \
         RETURN VALUE { MATCH (a)-[:KNOWS]->(b) RETURN b.name } AS friend",
    ));
    // The global form matches all six people → also a cardinality error.
    assert!(exec_err(
        &mut g,
        "RETURN VALUE { MATCH (n:Person) RETURN n.name } AS nm",
    ));
}

#[test]
fn value_subquery_constant_and_where() {
    // No patterns → a constant scalar. And a WHERE filters the correlated match.
    let mut g = value_graph();
    assert_eq!(
        rows(&mut g, "RETURN VALUE { RETURN 1 + 2 } AS v"),
        vec![vec![n(3.0)]]
    );
    // WHERE narrows dave's two neighbours to exactly one → no cardinality error.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='dave' \
             RETURN VALUE { MATCH (a)-[:KNOWS]->(b) WHERE b.name='Erin' RETURN b.name } AS f",
        ),
        vec![vec![s("Erin")]]
    );
}

#[test]
fn let_in_expression_binds_scoped_locals() {
    // ISO `<let value expression>`: LET x = e IN body END. A constant fold.
    let mut g = value_graph();
    assert_eq!(
        rows(&mut g, "RETURN LET x = 2 + 3 IN x * x END AS v"),
        vec![vec![n(25.0)]]
    );
    // Multiple bindings, later sees earlier (y references x).
    assert_eq!(
        rows(&mut g, "RETURN LET x = 4, y = x + 1 IN x * y END AS v"),
        vec![vec![n(20.0)]]
    );
    // Correlated: reads an outer variable in the binding.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='alice' \
             RETURN LET u = a.name IN u || '!' END AS greet",
        ),
        vec![vec![s("Alice!")]]
    );
}

#[test]
fn let_in_binding_suppresses_bare_in_operator() {
    // The binding RHS ends at the structural IN: `LET x = a.id IN [...] END` binds
    // `x = a.id` (NOT `a.id IN [...]`), then the body is the list membership isn't
    // even here — the body is what follows IN. Here body = a truthy compare.
    let mut g = value_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='alice' \
             RETURN LET nm = a.name IN nm = 'Alice' END AS ok",
        ),
        vec![vec![b(true)]]
    );
    // A parenthesized IN predicate inside a binding still works (parens re-enable
    // the operator): x = whether alice.name is in a list.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:Person) WHERE a.id='alice' \
             RETURN LET hit = (a.name IN ['Alice', 'Bob']) IN hit END AS present",
        ),
        vec![vec![b(true)]]
    );
}

/// Fixture for subpath-WHERE tests: three KNOWS edges with ages + weights.
///   alice(30) -w0.9-> bob(25)     (older → younger)
///   carol(20) -w0.3-> dave(40)    (younger → older)
///   erin(35)  -w0.7-> frank(35)   (equal ages)
fn ages_graph() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"alice","labels":["Person"],"properties":{"id":"alice","name":"Alice","age":30}}"#,
        r#"{"type":"node","id":"bob","labels":["Person"],"properties":{"id":"bob","name":"Bob","age":25}}"#,
        r#"{"type":"node","id":"carol","labels":["Person"],"properties":{"id":"carol","name":"Carol","age":20}}"#,
        r#"{"type":"node","id":"dave","labels":["Person"],"properties":{"id":"dave","name":"Dave","age":40}}"#,
        r#"{"type":"node","id":"erin","labels":["Person"],"properties":{"id":"erin","name":"Erin","age":35}}"#,
        r#"{"type":"node","id":"frank","labels":["Person"],"properties":{"id":"frank","name":"Frank","age":35}}"#,
        r#"{"type":"edge","id":"k1","from":"alice","to":"bob","labels":["KNOWS"],"properties":{"weight":0.9}}"#,
        r#"{"type":"edge","id":"k2","from":"carol","to":"dave","labels":["KNOWS"],"properties":{"weight":0.3}}"#,
        r#"{"type":"edge","id":"k3","from":"erin","to":"frank","labels":["KNOWS"],"properties":{"weight":0.7}}"#,
    ])
}

#[test]
fn subpath_where_filters_across_elements() {
    // ISO parenthesized-subpath WHERE: the predicate spans both endpoints and is
    // part of the pattern. Only carol(20)→dave(40) has age(x) < age(y).
    let mut g = ages_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) RETURN x.name AS n",
        ),
        vec![vec![s("Carol")]]
    );
    // Referencing the edge inside the subpath WHERE: weight > 0.5 → alice, erin.
    assert_eq!(
        rows(
            &mut g,
            "MATCH ((x:Person)-[e:KNOWS]->(y:Person) WHERE e.weight > 0.5) \
             RETURN x.name AS n ORDER BY n",
        ),
        vec![vec![s("Alice")], vec![s("Erin")]]
    );
}

#[test]
fn subpath_where_equals_clause_where_when_unquantified() {
    // For a single non-quantified pattern, a subpath WHERE and a clause WHERE are
    // semantically identical — same rows, proving we didn't misinterpret either.
    let mut g = ages_graph();
    let subpath = rows(
        &mut g,
        "MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) RETURN x.name AS n ORDER BY n",
    );
    let clause = rows(
        &mut g,
        "MATCH (x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age RETURN x.name AS n ORDER BY n",
    );
    assert_eq!(subpath, clause);
    assert_eq!(subpath, vec![vec![s("Carol")]]);
}

#[test]
fn subpath_where_and_clause_where_compose() {
    // Both a subpath WHERE (inside the parens) AND a trailing clause WHERE — they
    // are distinct and AND together. Subpath: x.age < y.age (→ carol→dave); then
    // clause: y.name = 'Dave' (still carol). Flipping the clause to a non-match
    // (y.name = 'Bob') yields nothing, proving the clause WHERE is really applied.
    let mut g = ages_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) WHERE y.name = 'Dave' \
             RETURN x.name AS n",
        ),
        vec![vec![s("Carol")]]
    );
    assert!(rows(
        &mut g,
        "MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) WHERE y.name = 'Bob' \
             RETURN x.name AS n",
    )
    .is_empty());
}

#[test]
fn regular_clause_where_still_works_alongside_subpath() {
    // A plain clause WHERE (no subpath parens) is unaffected — it filters rows and
    // sees every pattern variable, exactly as before.
    let mut g = ages_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (x:Person)-[:KNOWS]->(y:Person) WHERE x.age > y.age RETURN x.name AS n",
        ),
        vec![vec![s("Alice")]]
    );
    // Inline element WHERE (per-node) also still works.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (x:Person WHERE x.age >= 40) RETURN x.name AS n",
        ),
        vec![vec![s("Dave")]]
    );
}

#[test]
fn subpath_where_single_node() {
    // A single-node subpath: `((x:Person) WHERE p)` attaches the WHERE to the start
    // node. Ages ≥ 35 → dave(40), erin(35), frank(35).
    let mut g = ages_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH ((x:Person) WHERE x.age >= 35) RETURN x.name AS n ORDER BY n",
        ),
        vec![vec![s("Dave")], vec![s("Erin")], vec![s("Frank")]]
    );
}

#[test]
fn subpath_quantifier_and_pathvar_rejected() {
    // A quantified subpath as the FIRST path factor (no leading anchor) now parses —
    // it is ISO `<parenthesized path pattern expression> <quantifier>`.
    assert!(parse("MATCH ((x)-[:KNOWS]->(y) WHERE x.age < y.age)+ RETURN x").is_ok());
    // A path variable is fine over a QUANTIFIED subpath (it binds the repeated walk)…
    assert!(parse("MATCH p = ((x)-[:KNOWS]->(y)){1,3} (z) RETURN p").is_ok());
    // …but a path variable on a bare NON-quantified grouping is still not supported.
    assert!(parse("MATCH p = ((x)-[:KNOWS]->(y) WHERE x.age < y.age) RETURN p").is_err());
}

#[test]
fn subpath_where_referencing_only_start_or_end() {
    // The subpath WHERE may reference any subset of the subpath's variables.
    let mut g = ages_graph();
    // Only the start: x.age < 30 → carol(20).
    assert_eq!(
        rows(
            &mut g,
            "MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < 30) RETURN x.name AS n",
        ),
        vec![vec![s("Carol")]]
    );
    // Only the end: y.age < 30 → bob's predecessor alice.
    assert_eq!(
        rows(
            &mut g,
            "MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE y.age < 30) RETURN x.name AS n",
        ),
        vec![vec![s("Alice")]]
    );
}

#[test]
fn select_statement_basic_and_grouping() {
    // ISO SELECT desugars to MATCH + RETURN. Ages: 30,25,20,40,35,35 (erin=frank=35).
    let mut g = ages_graph();
    // Plain projection (no FROM) → a one-row constant.
    assert_eq!(rows(&mut g, "SELECT 1 + 2 AS v"), vec![vec![n(3.0)]]);
    // SELECT * over a single matched node.
    assert_eq!(
        rows(
            &mut g,
            "SELECT n.name AS nm FROM MATCH (n:Person {name: 'Alice'})"
        ),
        vec![vec![s("Alice")]]
    );
    // A pre-aggregation WHERE filters rows before the count.
    assert_eq!(
        rows(
            &mut g,
            "SELECT count(*) AS c FROM MATCH (n:Person) WHERE n.age >= 30",
        ),
        vec![vec![n(4.0)]] // alice30, dave40, erin35, frank35
    );
    // GROUP BY a property, with an aggregate and ORDER BY for determinism.
    assert_eq!(
        rows(
            &mut g,
            "SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) GROUP BY n.age ORDER BY age",
        ),
        vec![
            vec![n(20.0), n(1.0)],
            vec![n(25.0), n(1.0)],
            vec![n(30.0), n(1.0)],
            vec![n(35.0), n(2.0)], // erin + frank
            vec![n(40.0), n(1.0)],
        ]
    );
}

#[test]
fn select_having_filters_groups_post_aggregation() {
    let mut g = ages_graph();
    // HAVING on the grouped count: keep only ages shared by >1 person → 35.
    assert_eq!(
        rows(
            &mut g,
            "SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) \
             GROUP BY n.age HAVING count(*) > 1 ORDER BY age",
        ),
        vec![vec![n(35.0), n(2.0)]]
    );
    // HAVING referencing an aggregate NOT in the SELECT list — still folded.
    assert_eq!(
        rows(
            &mut g,
            "SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING count(*) > 1",
        ),
        vec![vec![n(35.0)]]
    );
    // HAVING on a group key (not an aggregate).
    assert_eq!(
        rows(
            &mut g,
            "SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) \
             GROUP BY n.age HAVING n.age >= 35 ORDER BY age",
        ),
        vec![vec![n(35.0), n(2.0)], vec![n(40.0), n(1.0)]]
    );
}

#[test]
fn select_having_on_global_aggregate() {
    let mut g = ages_graph();
    // No GROUP BY → a single global group; HAVING keeps or drops the whole row.
    assert_eq!(
        rows(
            &mut g,
            "SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 3",
        ),
        vec![vec![n(6.0)]]
    );
    // The same query with a failing threshold → zero rows (not a NULL/0 row).
    assert!(rows(
        &mut g,
        "SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 100",
    )
    .is_empty());
}

#[test]
fn select_having_three_valued_and_ordering() {
    let mut g = ages_graph();
    // ORDER BY + LIMIT compose after HAVING: all five age-groups, keep the two
    // youngest by age. (HAVING true for all here — count(*) >= 1.)
    assert_eq!(
        rows(
            &mut g,
            "SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age \
             HAVING count(*) >= 1 ORDER BY age LIMIT 2",
        ),
        vec![vec![n(20.0)], vec![n(25.0)]]
    );
    // A NULL HAVING (three-valued) drops the group: `HAVING null` keeps nothing.
    assert!(rows(
        &mut g,
        "SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING null",
    )
    .is_empty());
}

#[test]
fn having_is_select_only_not_on_return() {
    // HAVING is the SELECT statement's; a bare RETURN must not accept it (it is
    // left as trailing input → a parse error), matching ISO where HAVING lives on
    // the SELECT statement, not the return statement.
    assert!(parse("MATCH (n:Person) RETURN count(*) AS c HAVING count(*) > 1").is_err());
}

#[test]
fn match_mode_repeatable_vs_different_edges() {
    // 2-cycle a<->b. Under DIFFERENT EDGES (= default TRAIL) a 3-hop walk can't
    // re-tread an edge; under REPEATABLE ELEMENTS (WALK) it can re-tread.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"id":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"id":"b"}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e2","from":"b","to":"a","labels":["R"],"properties":{}}"#,
    ]);
    // REPEATABLE ELEMENTS: a→b→a→b re-treads e1,e2 → reaches b at hop 3.
    let rep = rows(
        &mut g,
        "MATCH REPEATABLE ELEMENTS (x:N {id:'a'})-[:R]->{3}(y) RETURN y.id AS id",
    );
    assert_eq!(rep, vec![vec![s("b")]]);
    // DIFFERENT EDGES (= default): only 2 distinct edges exist, so a 3-hop TRAIL
    // has no solution.
    let diff = rows(
        &mut g,
        "MATCH DIFFERENT EDGES (x:N {id:'a'})-[:R]->{3}(y) RETURN y.id AS id",
    );
    assert!(
        diff.is_empty(),
        "DIFFERENT EDGES 3-hop should have no trail: {diff:?}"
    );
    // DIFFERENT EDGES is exactly the default (no mode) behavior.
    let default = rows(
        &mut g,
        "MATCH (x:N {id:'a'})-[:R]->{3}(y) RETURN y.id AS id",
    );
    assert_eq!(diff, default);
}

// --- map/record runtime value (Phase 4) --------------------------------------

/// Build a result `Value::Map` for assertions (keys given in any order; the
/// engine canonicalizes on store, so expectations use sorted order).
fn vmap(pairs: &[(&str, Value)]) -> Value {
    Value::Map(
        pairs
            .iter()
            .map(|(k, v)| ((*k).into(), v.clone()))
            .collect(),
    )
}

fn map_graph() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"id":"a","meta":{"city":"NYC","n":1}}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"id":"b","meta":{"city":"NYC","n":1}}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"id":"c","meta":{"city":"LA","n":2}}}"#,
    ])
}

#[test]
fn read_and_return_a_stored_map() {
    let mut g = map_graph();
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta AS m"),
        vec![vec![vmap(&[("city", s("NYC")), ("n", n(1.0))])]],
    );
    // An absent field of a map is not reachable yet (Phase 5); the whole map reads.
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'c'}) RETURN n.meta AS m"),
        vec![vec![vmap(&[("city", s("LA")), ("n", n(2.0))])]],
    );
}

#[test]
fn map_equality_is_structural() {
    // a.meta == b.meta (same fields/values), != c.meta.
    let mut g = map_graph();
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:P {id:'a'}), (b:P {id:'b'}) RETURN a.meta = b.meta AS eq",
        ),
        vec![vec![b(true)]],
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:P {id:'a'}), (c:P {id:'c'}) RETURN a.meta = c.meta AS eq",
        ),
        vec![vec![b(false)]],
    );
}

#[test]
fn distinct_over_map_values_collapses_equal_maps() {
    // a and b share an identical map → DISTINCT yields two maps, not three.
    let mut g = map_graph();
    let mut got: Vec<Value> = rows(&mut g, "MATCH (n:P) RETURN DISTINCT n.meta AS m")
        .into_iter()
        .map(|r| r[0].clone())
        .collect();
    got.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    assert_eq!(
        got,
        vec![
            vmap(&[("city", s("LA")), ("n", n(2.0))]),
            vmap(&[("city", s("NYC")), ("n", n(1.0))]),
        ],
    );
}

#[test]
fn order_by_over_map_values_is_deterministic() {
    // A total order on maps (sorted-field lexicographic) makes ORDER BY well-
    // defined: "LA" < "NYC" on the first field. a and b tie (equal maps).
    let mut g = map_graph();
    let ids: Vec<Value> = rows(
        &mut g,
        "MATCH (n:P) RETURN n.id AS id, n.meta AS m ORDER BY m, id",
    )
    .into_iter()
    .map(|r| r[0].clone())
    .collect();
    assert_eq!(ids, vec![s("c"), s("a"), s("b")]); // LA first, then the two NYC
}

// --- record constructor + field access (Phase 5) -----------------------------

#[test]
fn record_constructor_builds_a_canonical_map() {
    let mut g = map_graph();
    // Fields authored out of order → canonical (sorted) map on output.
    assert_eq!(
        rows(&mut g, "RETURN {name: 'marko', age: 29} AS r"),
        vec![vec![vmap(&[("age", n(29.0)), ("name", s("marko"))])]],
    );
    // Empty record.
    assert_eq!(
        rows(&mut g, "RETURN {} AS r"),
        vec![vec![Value::Map(vec![])]],
    );
    // Duplicate field name → last write wins.
    assert_eq!(
        rows(&mut g, "RETURN {a: 1, a: 2} AS r"),
        vec![vec![vmap(&[("a", n(2.0))])]],
    );
    // A field value can be any expression (incl. a variable / property).
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:P {id:'a'}) RETURN {who: n.id, city: n.meta.city} AS r",
        ),
        vec![vec![vmap(&[("city", s("NYC")), ("who", s("a"))])]],
    );
}

#[test]
fn field_access_on_a_record() {
    let mut g = map_graph();
    // Dot access, subscript access, and a missing field (→ null).
    assert_eq!(
        rows(&mut g, "RETURN {a: 1, b: 2}.a AS x"),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN {a: 1, b: 2}['b'] AS x"),
        vec![vec![n(2.0)]],
    );
    assert_eq!(
        rows(&mut g, "RETURN {a: 1}.zzz AS x"),
        vec![vec![Value::Null]],
    );
    // Nested construction + chained access.
    assert_eq!(
        rows(&mut g, "RETURN {p: {n: 5}}.p.n AS x"),
        vec![vec![n(5.0)]],
    );
}

#[test]
fn field_access_on_a_stored_map() {
    let mut g = map_graph();
    // Nested access into a stored map property.
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta.city AS c"),
        vec![vec![s("NYC")]],
    );
    // A missing nested field reads as null.
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta.zip AS z"),
        vec![vec![Value::Null]],
    );
}

#[test]
fn where_on_a_nested_map_field_scans_correctly() {
    // Nested-field predicate works via scan (the planner index seek is a later
    // phase; correctness must hold regardless).
    let mut g = map_graph();
    let mut ids: Vec<Value> = rows(
        &mut g,
        "MATCH (n:P) WHERE n.meta.city = 'NYC' RETURN n.id AS id",
    )
    .into_iter()
    .map(|r| r[0].clone())
    .collect();
    ids.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    assert_eq!(ids, vec![s("a"), s("b")]);
}

#[test]
fn set_a_map_property_then_read_it_back() {
    // The write path: SET n.x = {record} stores a canonical map; read it back.
    let mut g = map_graph();
    let out = rows(
        &mut g,
        "MATCH (n:P {id:'a'}) SET n.tag = {b: 2, a: 1} RETURN n.tag AS t",
    );
    assert_eq!(out, vec![vec![vmap(&[("a", n(1.0)), ("b", n(2.0))])]]);
    // And it persists (a second read sees it).
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.tag.a AS a"),
        vec![vec![n(1.0)]],
    );
}

#[test]
fn nested_field_where_uses_the_dotted_path_index() {
    // With a `meta.city` index, `WHERE n.meta.city = 'NYC'` seeks it (Phase 3
    // proved the seek primitive; this proves the planner routes to it without
    // altering results — same rows as the scan path).
    let mut g = map_graph();
    g.create_vertex_index("meta.city");
    let mut ids: Vec<Value> = rows(
        &mut g,
        "MATCH (n:P) WHERE n.meta.city = 'NYC' RETURN n.id AS id",
    )
    .into_iter()
    .map(|r| r[0].clone())
    .collect();
    ids.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    assert_eq!(ids, vec![s("a"), s("b")]);
    // A param seeks the same index (`WHERE n.meta.city = $c`).
    let mut p = Params::new();
    p.insert("c".to_string(), super::eval::Val::Str("LA".into()));
    let out = qp(
        &mut g,
        "MATCH (n:P) WHERE n.meta.city = $c RETURN n.id AS id",
        p,
    );
    assert_eq!(out, vec![vec![s("c")]]);
}

#[test]
fn deep_stored_field_access_reads_only_the_leaf() {
    // A 3-level nested stored map — the collapsed `PropField` navigates the stored
    // `Value` in place and materializes only the addressed leaf.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"id":"a","deep":{"x":{"y":{"z":42}}}}}"#,
    ]);
    assert_eq!(
        rows(&mut g, "MATCH (n:P) RETURN n.deep.x.y.z AS v"),
        vec![vec![n(42.0)]],
    );
    // A mid-path leaf reads back as the nested map (materialized only there).
    assert_eq!(
        rows(&mut g, "MATCH (n:P) RETURN n.deep.x.y AS m"),
        vec![vec![vmap(&[("z", n(42.0))])]],
    );
    // A missing segment anywhere → null (not an error).
    assert_eq!(
        rows(&mut g, "MATCH (n:P) RETURN n.deep.x.nope.z AS v"),
        vec![vec![Value::Null]],
    );
    // Subscript form collapses the same way.
    assert_eq!(
        rows(&mut g, "MATCH (n:P) RETURN n.deep['x']['y']['z'] AS v"),
        vec![vec![n(42.0)]],
    );
    // A field access on a non-map property is null (root isn't a stored map).
    assert_eq!(
        rows(&mut g, "MATCH (n:P) RETURN n.id.foo AS v"),
        vec![vec![Value::Null]],
    );
}

#[test]
fn deboxed_record_queries_identically_to_the_boxed_map() {
    // Record-typed constraints, step 2: declaring a RECORD constraint de-boxes `meta` into
    // typed sub-columns. Every read path (whole map, field access, nested WHERE,
    // dotted-path index seek) must return exactly what the boxed map did.
    let mut g = map_graph();
    g.create_type_constraint("P", "meta", "record{city::string,n::number}")
        .unwrap();
    // Whole-map read (synthesized from sub-columns).
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta AS m"),
        vec![vec![vmap(&[("city", s("NYC")), ("n", n(1.0))])]],
    );
    // Direct field read (sub-column, no map materialization) + a missing field.
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta.city AS c"),
        vec![vec![s("NYC")]],
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta.zip AS z"),
        vec![vec![Value::Null]],
    );
    // Nested-field predicate (scan over the de-boxed column).
    let mut ids: Vec<Value> = rows(
        &mut g,
        "MATCH (n:P) WHERE n.meta.city = 'NYC' RETURN n.id AS id",
    )
    .into_iter()
    .map(|r| r[0].clone())
    .collect();
    ids.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    assert_eq!(ids, vec![s("a"), s("b")]);
    // The dotted-path index still seeks after de-boxing (build reads via synthesis).
    g.create_vertex_index("meta.city");
    let mut p = Params::new();
    p.insert("c".to_string(), super::eval::Val::Str("LA".into()));
    let out = qp(
        &mut g,
        "MATCH (n:P) WHERE n.meta.city = $c RETURN n.id AS id",
        p,
    );
    assert_eq!(out, vec![vec![s("c")]]);
    // A SET that scatters into sub-columns round-trips.
    let updated = rows(
        &mut g,
        "MATCH (n:P {id:'a'}) SET n.meta = {city: 'SF', n: 9} RETURN n.meta.city AS c",
    );
    assert_eq!(updated, vec![vec![s("SF")]]);
}

#[test]
fn deboxed_record_survives_transaction_rollback() {
    // A SET that scatters into the record sub-columns must undo exactly on
    // rollback — the undo log captures the synthesized prior map and re-scatters.
    let mut g = map_graph();
    g.create_type_constraint("P", "meta", "record{city::string,n::number}")
        .unwrap();
    let before = rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta AS m");
    g.begin_tx();
    let _ = rows(
        &mut g,
        "MATCH (n:P {id:'a'}) SET n.meta = {city: 'SF', n: 9}",
    );
    // Read-your-writes inside the transaction.
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta.city AS c"),
        vec![vec![s("SF")]],
    );
    g.rollback_tx();
    assert_eq!(
        rows(&mut g, "MATCH (n:P {id:'a'}) RETURN n.meta AS m"),
        before,
        "the record is exactly restored after rollback",
    );
}

/// `par_project` (the opt-in `parallel-query` feature) splits a large projection
/// frame across rayon threads. Over more than `MIN_ROWS` (16_384) rows, prove the
/// parallel result equals the serial one: no row dropped or duplicated, and each
/// row's columns stay paired (no cross-chunk mixing). `w == 2v+1` on every row is the
/// pairing witness; the `seen` sieve is the completeness witness.
#[cfg(feature = "parallel-query")]
#[test]
fn parallel_projection_preserves_every_row_over_a_large_frame() {
    let count = 20_000usize; // > MIN_ROWS, so the frame is chunked across threads
    let lines: Vec<String> = (0..count)
        .map(|i| {
            format!(r#"{{"type":"node","id":"v{i}","labels":["T"],"properties":{{"val":{i}}}}}"#)
        })
        .collect();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);

    let out = rows(&mut g, "MATCH (a:T) RETURN a.val AS v, a.val * 2 + 1 AS w");
    assert_eq!(out.len(), count);
    let mut seen = vec![false; count];
    for r in &out {
        let vi = match &r[0] {
            Value::Num(x) => *x as usize,
            other => panic!("v not numeric: {other:?}"),
        };
        // w == 2v+1 proves this row's two columns came from the same source row.
        assert_eq!(r[1], n(2.0 * vi as f64 + 1.0), "row {vi}: columns unpaired");
        assert!(
            !std::mem::replace(&mut seen[vi], true),
            "val {vi} duplicated"
        );
    }
    assert!(seen.iter().all(|&x| x), "a row was dropped");
}

/// Same large-frame parallel projection, but each row runs a var-length subquery
/// (`EXISTS { (a)-[:R]->+(:T) }`) — a per-row traversal evaluated concurrently across
/// the rayon projection threads, exercising the shared trail-mark pool (a `Mutex`
/// under this feature) under real contention. Isolated pairs keep each answer
/// determinate and O(1): even vertices reach exactly one `T`, odd ones none.
#[cfg(feature = "parallel-query")]
#[test]
fn parallel_projection_with_per_row_subquery_over_a_large_frame() {
    let count = 20_000usize; // even, > MIN_ROWS
    let mut lines: Vec<String> = (0..count)
        .map(|i| {
            format!(r#"{{"type":"node","id":"v{i}","labels":["T"],"properties":{{"val":{i}}}}}"#)
        })
        .collect();
    for i in (0..count).step_by(2) {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"v{i}","to":"v{}","labels":["R"],"properties":{{}}}}"#,
            i + 1
        ));
    }
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);

    let out = rows(
        &mut g,
        "MATCH (a:T) RETURN a.val AS v, EXISTS { MATCH (a)-[:R]->+(b:T) } AS reach",
    );
    assert_eq!(out.len(), count);
    let mut reachable = 0usize;
    for r in &out {
        let vi = match &r[0] {
            Value::Num(x) => *x as usize,
            other => panic!("v not numeric: {other:?}"),
        };
        assert_eq!(r[1], b(vi % 2 == 0), "row {vi}: wrong reachability");
        if vi % 2 == 0 {
            reachable += 1;
        }
    }
    assert_eq!(reachable, count / 2);
}

/// Serial-vs-parallel timing for `parallel-query`, across the shapes it parallelizes
/// (large-frame projection, per-row subqueries, GROUP BY aggregation). Ignored — it's a
/// benchmark, not a check. Run the SAME binary twice to isolate the parallelization
/// effect (only the rayon thread count changes):
///   RAYON_NUM_THREADS=1 cargo test --release --features parallel-query bench_parallel_query -- --ignored --nocapture
///   cargo test --release --features parallel-query bench_parallel_query -- --ignored --nocapture
#[cfg(feature = "parallel-query")]
#[test]
#[ignore = "benchmark; run with --ignored --nocapture and vary RAYON_NUM_THREADS"]
fn bench_parallel_query_speedup() {
    use std::time::Instant;

    // N vertices in bounded C-chains: each `-[:R]->+` walk stays within its chain, so
    // a per-row subquery does real but O(C) traversal — the "advanced query that fans
    // out" shape. Frame is N rows (>> the 16_384 par_project threshold).
    let n = 100_000usize;
    let chain = 32usize;
    let mut lines: Vec<String> = (0..n)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"v{i}","labels":["T"],"properties":{{"val":{i},"grp":{}}}}}"#,
                i % 256
            )
        })
        .collect();
    for i in 0..n {
        if i + 1 < n && (i + 1) % chain != 0 {
            lines.push(format!(
                r#"{{"type":"edge","id":"e{i}","from":"v{i}","to":"v{}","labels":["R"],"properties":{{}}}}"#,
                i + 1
            ));
        }
    }
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);
    eprintln!(
        "\nrayon threads = {} | graph {} nodes / {} edges",
        rayon::current_num_threads(),
        g.vertex_count(),
        g.edge_count()
    );

    let bench = |g: &mut Graph, name: &str, query: &str, iters: usize| {
        let _ = rows(g, query); // warmup
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let out = rows(g, query);
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(out);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "  {:<34} p50={:>8.2}ms  min={:>8.2}ms",
            name,
            ms[ms.len() / 2],
            ms[0]
        );
    };

    bench(
        &mut g,
        "light scalar projection",
        "MATCH (a:T) RETURN a.val * 2 + 1 AS w",
        25,
    );
    bench(
        &mut g,
        "heavy scalar projection",
        "MATCH (a:T) RETURN (a.val * 2 + 1) * (a.val - 3) + a.val * a.val - a.val / 2 AS w",
        25,
    );
    bench(
        &mut g,
        "per-row EXISTS subquery",
        "MATCH (a:T) RETURN a.val AS v, EXISTS { MATCH (a)-[:R]->+(b:T) } AS r",
        15,
    );
    bench(
        &mut g,
        "per-row COUNT subquery",
        "MATCH (a:T) RETURN a.val AS v, COUNT { MATCH (a)-[:R]->+(b:T) } AS c",
        10,
    );
    bench(
        &mut g,
        "GROUP BY aggregation",
        "MATCH (a:T) RETURN a.grp AS g, count(*) AS c, avg(a.val) AS m",
        25,
    );
    bench(
        &mut g,
        "WHERE arith filter",
        "MATCH (a:T) WHERE a.val * 3 > 100000 AND a.val < 250000 RETURN a.val AS v",
        25,
    );
    bench(
        &mut g,
        "1-hop expansion join",
        "MATCH (a:T)-[:R]->(b:T) RETURN a.val AS x, b.val AS y",
        20,
    );
    bench(
        &mut g,
        "2-hop expansion join",
        "MATCH (a:T)-[:R]->(b:T)-[:R]->(c:T) RETURN c.val AS v",
        20,
    );
    bench(
        &mut g,
        "1-hop COUNT subquery",
        "MATCH (a:T) RETURN a.val AS v, COUNT { MATCH (a)-[:R]->(b:T) } AS c",
        15,
    );
    bench(
        &mut g,
        "global aggregate",
        "MATCH (a:T) RETURN count(*) AS c, avg(a.val) AS m, sum(a.val) AS s",
        25,
    );
    bench(
        &mut g,
        "heavy GROUP BY (5 aggs)",
        "MATCH (a:T) RETURN a.grp AS g, count(*) AS c, avg(a.val) AS m, min(a.val) AS lo, max(a.val) AS hi",
        25,
    );
}

/// AML-shaped workload (the gnarly layering/structuring patterns): a transaction network,
/// then the layering / structuring / circular-flow queries. Same serial-vs-parallel
/// protocol (vary RAYON_NUM_THREADS) to gauge whether real AML queries speed up.
#[cfg(feature = "parallel-query")]
#[test]
#[ignore = "benchmark; run with --ignored --nocapture and vary RAYON_NUM_THREADS"]
fn bench_aml_shapes() {
    use std::time::Instant;

    // 50k accounts, fan-out 3 (i→i+1, i→7i+3, i→13i+5 mod n) with amount + ts. Gives
    // chains, fan-in/out, and cycles — the AML substrate.
    let n = 50_000usize;
    let mut lines: Vec<String> = (0..n)
        .map(|i| {
            format!(r#"{{"type":"node","id":"a{i}","labels":["A"],"properties":{{"id":{i}}}}}"#)
        })
        .collect();
    let mut e = 0usize;
    for i in 0..n {
        for &t in &[(i + 1) % n, (i * 7 + 3) % n, (i * 13 + 5) % n] {
            lines.push(format!(
                r#"{{"type":"edge","id":"t{e}","from":"a{i}","to":"a{t}","labels":["TX"],"properties":{{"amt":{},"ts":{i}}}}}"#,
                (i % 900) + 100
            ));
            e += 1;
        }
    }
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);
    eprintln!(
        "\nrayon threads = {} | AML graph {} accts / {} tx",
        rayon::current_num_threads(),
        g.vertex_count(),
        g.edge_count()
    );

    let bench = |g: &mut Graph, name: &str, query: &str, iters: usize| {
        let _ = rows(g, query);
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            std::hint::black_box(rows(g, query));
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "  {:<40} p50={:>8.2}ms  min={:>8.2}ms",
            name,
            ms[ms.len() / 2],
            ms[0]
        );
    };

    // Per-account subquery shapes (the outer fan-out over 50k accounts parallelizes).
    bench(
        &mut g,
        "layering spread COUNT ->{1,3}",
        "MATCH (a:A) RETURN a.id AS id, COUNT { MATCH (a)-[:TX]->{1,3}(b:A) } AS spread",
        8,
    );
    bench(
        &mut g,
        "circular-flow EXISTS ->{2,4}(a)",
        "MATCH (a:A) WHERE EXISTS { MATCH (a)-[:TX]->{2,4}(a) } RETURN a.id AS id",
        8,
    );
    bench(
        &mut g,
        "fundedby depth COUNT <-{1,3}",
        "MATCH (a:A) RETURN a.id AS id, COUNT { MATCH (a)<-[:TX]-{1,3}(s:A) } AS sources",
        8,
    );
    // Unrolled fixed-3-hop layering with monotone-decreasing amounts (structuring).
    bench(
        &mut g,
        "3-hop decreasing-amount chain",
        "MATCH (a:A)-[e1:TX]->(b:A)-[e2:TX]->(c:A)-[e3:TX]->(d:A) \
         WHERE e1.amt > e2.amt AND e2.amt > e3.amt RETURN a.id AS s, d.id AS t LIMIT 5000",
        8,
    );
}

/// HRIS-shaped workload: an org hierarchy (tree, each person REPORTS_TO one manager)
/// plus departments, then the span-of-control / depth / chain queries. Sparse tree
/// (vs AML's dense network), so per-entity work is cheaper and imbalanced — measure
/// whether it still parallelizes. Same RAYON_NUM_THREADS protocol.
#[cfg(feature = "parallel-query")]
#[test]
#[ignore = "benchmark; run with --ignored --nocapture and vary RAYON_NUM_THREADS"]
fn bench_hris_shapes() {
    use std::time::Instant;

    // 100k employees, branching factor 6: employee i REPORTS_TO (i-1)/6 (i=0 is the CEO).
    // dept = i % 50. A realistic ~7-level org tree.
    let n = 100_000usize;
    let mut lines: Vec<String> = (0..n)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"e{i}","labels":["Emp"],"properties":{{"id":{i},"dept":{}}}}}"#,
                i % 50
            )
        })
        .collect();
    for i in 1..n {
        lines.push(format!(
            r#"{{"type":"edge","id":"r{i}","from":"e{i}","to":"e{}","labels":["REPORTS_TO"],"properties":{{}}}}"#,
            (i - 1) / 6
        ));
    }
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);
    eprintln!(
        "\nrayon threads = {} | HRIS org {} emps / {} reports-to",
        rayon::current_num_threads(),
        g.vertex_count(),
        g.edge_count()
    );

    let bench = |g: &mut Graph, name: &str, query: &str, iters: usize| {
        let _ = rows(g, query);
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            std::hint::black_box(rows(g, query));
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "  {:<40} p50={:>8.2}ms  min={:>8.2}ms",
            name,
            ms[ms.len() / 2],
            ms[0]
        );
    };

    bench(
        &mut g,
        "span-of-control COUNT descendants",
        "MATCH (m:Emp) RETURN m.id AS id, COUNT { MATCH (m)<-[:REPORTS_TO]-*(r:Emp) } AS reports",
        8,
    );
    bench(
        &mut g,
        "management depth COUNT ancestors",
        "MATCH (e:Emp) RETURN e.id AS id, COUNT { MATCH (e)-[:REPORTS_TO]->+(m:Emp) } AS levels",
        8,
    );
    bench(
        &mut g,
        "in-chain-of-exec EXISTS (bound)",
        "MATCH (e:Emp) WHERE EXISTS { MATCH (e)-[:REPORTS_TO]->+(m:Emp {id:5}) } RETURN e.id AS id",
        8,
    );
    bench(
        &mut g,
        "headcount by dept (GROUP BY)",
        "MATCH (e:Emp) RETURN e.dept AS d, count(*) AS n",
        20,
    );
}

/// M0 end-to-end: a temporal property index must be transparent — a range seek
/// returns exactly the rows the unindexed scan does. Proves the query-side key
/// (from the `DATE` literal) matches the stored column key and that
/// `prop_index_hint` actually picks up the temporal comparison.
#[test]
fn temporal_property_index_seek_agrees_with_scan() {
    let dates = [
        "2019-03-01",
        "2020-06-15",
        "2021-01-01",
        "2022-11-30",
        "2023-07-04",
        "2024-02-29",
        "2025-12-31",
    ];
    let lines: Vec<String> = dates
        .iter()
        .enumerate()
        .map(|(i, d)| {
            format!(
                r#"{{"type":"node","id":"n{i}","labels":["E"],"properties":{{"id":{i},"d":{{"@date":"{d}"}}}}}}"#
            )
        })
        .collect();
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let q = "MATCH (v:E) WHERE v.d <= DATE '2022-11-30' RETURN v.id AS id ORDER BY id";

    let mut g_scan = graph_of(&refs);
    let scan = rows(&mut g_scan, q);

    let mut g_seek = graph_of(&refs);
    g_seek.create_vertex_index("d");
    let seek = rows(&mut g_seek, q);

    assert_eq!(
        scan, seek,
        "temporal index seek must return the same rows as the scan"
    );
    assert_eq!(
        scan.len(),
        4,
        "sanity: 2019,2020,2021,2022 are <= 2022-11-30; 2023-25 excluded"
    );
}

/// Bitemporal temporal-index bake-off harness. Builds an SCD-2 org (REPORTS_TO
/// edge versions with valid [vf,vt) + transaction [tf,tt) DATE intervals, most
/// current via a vt=INF sentinel, churn producing closed versions), then times
/// the hard as-of / overlap / range queries with and without the temporal index,
/// plus the write-interleaved build cost (index maintained on every insert). Run:
///   cargo test --release bench_temporal_index -- --ignored --nocapture
/// Bump N / VERSIONS toward the xxl (212k) / xxxl (1.75M) version counts.
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_temporal_index() {
    use crate::temporal::{Date, Temporal};
    use std::time::Instant;

    const N: usize = 25_000; // employees
    const VERSIONS: usize = 8; // REPORTS_TO versions per employee → N*VERSIONS edge versions
    const BASE: i32 = 18_000; // ~2019-04
    const PERIOD: i32 = 250; // days between reorgs
    const INF: i32 = 3_000_000; // far-future "current" sentinel (days)

    let dval = |d: i32| Value::Temporal(Temporal::Date(Date { days: d }));
    let dparam = |d: i32| super::eval::Val::Temporal(Temporal::Date(Date { days: d }));

    // Build the org; if `indexed`, create the vf/vt edge indexes BEFORE inserting so
    // every add_edge maintains them — the write-interleaved cost. Returns (graph, write_ms).
    // mode: 0 = no index, 1 = A (vf+vt edge indexes), 2 = B (RI-tree interval index).
    let build = |mode: u8| -> (Graph, f64) {
        let mut g = ndjson::decode("").unwrap();
        match mode {
            1 => {
                g.create_edge_index("vf");
                g.create_edge_index("vt");
            }
            2 => g.create_edge_interval_index("vf", "vt"),
            _ => {}
        }
        let t0 = Instant::now();
        let vids: Vec<u32> = (0..N)
            .map(|i| {
                g.add_vertex(
                    &["Emp".to_string()],
                    vec![("id".to_string(), Value::Num(i as f64))],
                )
            })
            .collect();
        for i in 0..N {
            let mgr = vids[(i / 6).min(N - 1)];
            for v in 0..VERSIONS {
                let vf = BASE + (v as i32) * PERIOD;
                let vt = if v + 1 == VERSIONS {
                    INF
                } else {
                    BASE + ((v + 1) as i32) * PERIOD
                };
                g.add_edge(
                    vids[i],
                    mgr,
                    "REPORTS_TO",
                    vec![
                        ("vf".to_string(), dval(vf)),
                        ("vt".to_string(), dval(vt)),
                        ("tf".to_string(), dval(vf)),
                        ("tt".to_string(), dval(INF)),
                    ],
                );
            }
        }
        (g, t0.elapsed().as_secs_f64() * 1000.0)
    };

    let (mut g_plain, w_plain) = build(0);
    let (mut g_idx, w_idx) = build(1);
    let (mut g_ri, w_ri) = build(2);
    eprintln!(
        "\nbitemporal org: {} emps / {} edge versions",
        g_plain.vertex_count(),
        g_plain.edge_count()
    );
    eprintln!(
        "WRITE-INTERLEAVED build: no-index {w_plain:.0}ms | A vf+vt index {w_idx:.0}ms ({:.2}x) | B RI-tree {w_ri:.0}ms ({:.2}x)",
        w_idx / w_plain,
        w_ri / w_plain
    );

    let bench = |g: &mut Graph, name: &str, q: &str, params: &Params| -> f64 {
        let prepared = parse(q).unwrap();
        // The count(*) value = how many edge versions matched (the selectivity).
        let matched = prepared
            .execute(g, params)
            .unwrap()
            .rows()
            .next()
            .and_then(|r| match r.to_vec().first() {
                Some(Value::Num(n)) => Some(*n as i64),
                _ => None,
            })
            .unwrap_or(-1);
        let mut ms = Vec::with_capacity(20);
        for _ in 0..20 {
            let t = Instant::now();
            let rs = prepared.execute(g, params).unwrap();
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(rs.rows().count());
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = ms[ms.len() / 2];
        eprintln!("  {name:<28} p50={p50:>8.3}ms  (matched={matched})");
        p50
    };

    // Probe params for the hard cases.
    let mut p_now = Params::new();
    p_now.insert("v".into(), dparam(BASE + PERIOD * (VERSIONS as i32) - 10)); // near "now"
    let mut p_hist = Params::new();
    p_hist.insert("v".into(), dparam(BASE + PERIOD * 3 + 20)); // mid-history
    let mut p_narrow = Params::new();
    p_narrow.insert("d1".into(), dparam(BASE + PERIOD * 3));
    p_narrow.insert("d2".into(), dparam(BASE + PERIOD * 3 + 60)); // 60-day window
    let mut p_wide = Params::new();
    p_wide.insert("d1".into(), dparam(BASE));
    p_wide.insert("d2".into(), dparam(INF));
    let mut p_range = Params::new();
    p_range.insert("d1".into(), dparam(BASE + PERIOD * 2));
    p_range.insert("d2".into(), dparam(BASE + PERIOD * 4));

    let asof = "MATCH ()-[r:REPORTS_TO]->() WHERE r.vf <= $v AND r.vt > $v RETURN count(*) AS n";
    let overlap =
        "MATCH ()-[r:REPORTS_TO]->() WHERE r.vf < $d2 AND r.vt > $d1 RETURN count(*) AS n";
    let range =
        "MATCH ()-[r:REPORTS_TO]->() WHERE r.vf >= $d1 AND r.vf <= $d2 RETURN count(*) AS n";

    let cases: [(&str, &str, &Params); 5] = [
        ("as-of now", asof, &p_now),
        ("as-of historical", asof, &p_hist),
        ("narrow overlap (straddlers)", overlap, &p_narrow),
        ("wide overlap", overlap, &p_wide),
        ("single-col range (vf)", range, &p_range),
    ];
    eprintln!("== NO INDEX (scan baseline) ==");
    let base: Vec<f64> = cases
        .iter()
        .map(|(n, q, p)| bench(&mut g_plain, n, q, p))
        .collect();
    eprintln!("== A: vf+vt INDEX (most-selective seek) ==");
    for (i, (n, q, p)) in cases.iter().enumerate() {
        let idx = bench(&mut g_idx, n, q, p);
        eprintln!("      └─ {:.2}x vs scan", base[i] / idx);
    }

    // Contender B — now LIVE through the query path: prop_index_hint recognizes the
    // as-of predicate and seeds from the RI-tree stab. Run the same GQL query the other
    // configs run; `matched` proves correctness (== the scan's 25000).
    eprintln!("== B: RI-tree interval index (as-of + overlap, via GQL query path) ==");
    let b_now = bench(&mut g_ri, "as-of now", asof, &p_now);
    eprintln!("      └─ {:.2}x vs scan", base[0] / b_now);
    let b_hist = bench(&mut g_ri, "as-of historical", asof, &p_hist);
    eprintln!("      └─ {:.2}x vs scan", base[1] / b_hist);
    let b_narrow = bench(&mut g_ri, "narrow overlap (straddlers)", overlap, &p_narrow);
    eprintln!("      └─ {:.2}x vs scan", base[2] / b_narrow);
    let b_wide = bench(&mut g_ri, "wide overlap", overlap, &p_wide);
    eprintln!("      └─ {:.2}x vs scan", base[3] / b_wide);
}

/// A key bitemporal BLOCKER was that variable-length paths couldn't filter
/// edges, forcing a materialize-a-slice-then-traverse hack. The per-repetition WHERE
/// (ISO `parenthesizedPathPatternWhereClause`) should retire it:
/// each hop's edge is filtered by the as-of predicate inline, no slice, no mutation.
/// Chain v4→v3→v2→v1→v0; the v3→v2 edge is EXPIRED at the query date, so an as-of
/// traversal from v4 must stop at v3.
#[test]
fn bitemporal_per_rep_where_filters_variable_length_traversal() {
    let e = |from: &str, to: &str, vf: &str, vt: &str| {
        format!(
            r#"{{"type":"edge","from":"{from}","to":"{to}","labels":["REPORTS_TO"],"properties":{{"vf":{{"@date":"{vf}"}},"vt":{{"@date":"{vt}"}}}}}}"#
        )
    };
    let mut lines: Vec<String> = (0..5)
        .map(|i| {
            format!(r#"{{"type":"node","id":"v{i}","labels":["E"],"properties":{{"id":"v{i}"}}}}"#)
        })
        .collect();
    lines.push(e("v1", "v0", "2020-01-01", "9999-12-31")); // valid at 2024
    lines.push(e("v2", "v1", "2020-01-01", "9999-12-31")); // valid at 2024
    lines.push(e("v3", "v2", "2020-01-01", "2022-01-01")); // EXPIRED by 2024
    lines.push(e("v4", "v3", "2020-01-01", "9999-12-31")); // valid at 2024
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);

    // As-of 2024: per-rep WHERE keeps only edges valid at the date. From v4 the walk
    // follows v4→v3 (valid) then finds v3→v2 expired, so it stops — reachable = {v3}.
    let asof = "MATCH (a:E {id:'v4'}) \
        ((x)-[r:REPORTS_TO]->(y) WHERE r.vf <= DATE '2024-01-01' AND r.vt > DATE '2024-01-01'){1,4} \
        (z:E) RETURN z.id AS id ORDER BY id";
    assert_eq!(
        rows(&mut g, asof),
        vec![vec![s("v3")]],
        "per-rep as-of must stop at the expired edge"
    );

    // Unfiltered, the same shape reaches every ancestor — proving the filter is what
    // pruned the chain, not the topology.
    let plain =
        "MATCH (a:E {id:'v4'}) ((x)-[:REPORTS_TO]->(y)){1,4} (z:E) RETURN z.id AS id ORDER BY id";
    assert_eq!(
        rows(&mut g, plain),
        vec![vec![s("v0")], vec![s("v1")], vec![s("v2")], vec![s("v3")]],
    );
}

/// The RI-tree interval index must stay correct under mutation. The dangerous case is
/// a SET that EXTENDS an interval to newly cover the query point — without remove+insert
/// maintenance the stab would MISS the row (a wrong answer, not just a stale extra).
/// Also covers SET-shrink and DELETE.
#[test]
fn interval_index_maintained_under_set_and_delete() {
    let edge = |id: &str, from: &str, to: &str, vf: &str, vt: &str| {
        format!(
            r#"{{"type":"edge","id":"{id}","from":"{from}","to":"{to}","labels":["R"],"properties":{{"id":"{id}","vf":{{"@date":"{vf}"}},"vt":{{"@date":"{vt}"}}}}}}"#
        )
    };
    let lines = [
        r#"{"type":"node","id":"v0","labels":["N"],"properties":{"id":"v0"}}"#.to_string(),
        r#"{"type":"node","id":"v1","labels":["N"],"properties":{"id":"v1"}}"#.to_string(),
        r#"{"type":"node","id":"v2","labels":["N"],"properties":{"id":"v2"}}"#.to_string(),
        edge("e0", "v1", "v0", "2020-01-01", "2022-01-01"), // expired at 2024
        edge("e1", "v2", "v1", "2020-01-01", "9999-12-31"), // current
    ];
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);
    g.create_edge_interval_index("vf", "vt");

    let asof =
        "MATCH ()-[r:R]->() WHERE r.vf <= DATE '2024-01-01' AND r.vt > DATE '2024-01-01' RETURN count(*) AS n";
    let count = |g: &mut Graph| -> i64 {
        match rows(g, asof)[0][0] {
            Value::Num(n) => n as i64,
            _ => -1,
        }
    };

    assert_eq!(count(&mut g), 1, "only e1 is valid at 2024");
    // SET-extend e0 past 2024 — without remove+insert this MISSES the row.
    rows(
        &mut g,
        "MATCH ()-[r:R {id:'e0'}]->() SET r.vt = DATE '2026-01-01'",
    );
    assert_eq!(
        count(&mut g),
        2,
        "extended e0 must now appear (SET maintenance / no miss)"
    );
    // SET-shrink e0 back before 2024.
    rows(
        &mut g,
        "MATCH ()-[r:R {id:'e0'}]->() SET r.vt = DATE '2021-01-01'",
    );
    assert_eq!(count(&mut g), 1, "re-expired e0 must drop out");
    // DELETE e1 — the only remaining valid edge.
    rows(&mut g, "MATCH ()-[r:R {id:'e1'}]->() DELETE r");
    assert_eq!(
        count(&mut g),
        0,
        "deleted e1 must not be returned (delete maintenance)"
    );
}

/// Shared fixture for the per-relation Allen tests: 13 stored intervals (one in each
/// Allen relation to Q=[2024-04-01, 2024-08-01), mutually exclusive) built three ways,
/// asserting `where_` returns exactly `{want}` under the RI-tree interval index, under
/// regular vf/vt indexes, AND under a full scan. Each relation gets its own `#[test]`
/// below so the count reflects the coverage and a failure names the exact relation.
fn allen_check(where_: &str, want: &str) {
    // (id, vf, vt) — one interval per relation to Q=[2024-04-01, 2024-08-01).
    let ivals = [
        ("before", "2024-01-01", "2024-03-01"),
        ("meets", "2024-01-01", "2024-04-01"),
        ("overlaps", "2024-01-01", "2024-06-01"),
        ("finished_by", "2024-01-01", "2024-08-01"),
        ("contains", "2024-01-01", "2024-12-01"),
        ("starts", "2024-04-01", "2024-06-01"),
        ("equals", "2024-04-01", "2024-08-01"),
        ("started_by", "2024-04-01", "2024-12-01"),
        ("during", "2024-05-01", "2024-07-01"),
        ("finishes", "2024-06-01", "2024-08-01"),
        ("overlapped_by", "2024-06-01", "2024-12-01"),
        ("met_by", "2024-08-01", "2024-12-01"),
        ("after", "2024-09-01", "2024-12-01"),
    ];
    let mut lines = vec![
        r#"{"type":"node","id":"n0","labels":["N"],"properties":{}}"#.to_string(),
        r#"{"type":"node","id":"n1","labels":["N"],"properties":{}}"#.to_string(),
    ];
    for (id, vf, vt) in ivals {
        lines.push(format!(
            r#"{{"type":"edge","id":"{id}","from":"n0","to":"n1","labels":["R"],"properties":{{"id":"{id}","vf":{{"@date":"{vf}"}},"vt":{{"@date":"{vt}"}}}}}}"#
        ));
    }
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();

    // Three index configurations that must all agree: interval index, regular indexes, scan.
    let make = |cfg: u8| {
        let mut g = graph_of(&refs);
        match cfg {
            1 => g.create_edge_interval_index("vf", "vt"),
            2 => {
                g.create_edge_index("vf");
                g.create_edge_index("vt");
            }
            _ => {}
        }
        g
    };
    let q = format!("MATCH ()-[r:R]->() WHERE {where_} RETURN r.id AS id ORDER BY id");
    let scan = rows(&mut make(0), &q);
    assert_eq!(
        scan,
        vec![vec![s(want)]],
        "scan returned wrong rows for `{where_}`"
    );
    assert_eq!(
        rows(&mut make(1), &q),
        scan,
        "interval-index seed disagrees for `{where_}`"
    );
    assert_eq!(
        rows(&mut make(2), &q),
        scan,
        "regular-index seed disagrees for `{where_}`"
    );
}

// One #[test] per Allen relation (Q=[2024-04-01, 2024-08-01)); each is independently
// named and runnable, and the cargo count reflects the 13-way coverage.
#[test]
fn allen_before() {
    allen_check("r.vt < DATE '2024-04-01'", "before");
}
#[test]
fn allen_meets() {
    allen_check("r.vt = DATE '2024-04-01'", "meets");
}
#[test]
fn allen_overlaps() {
    allen_check(
        "r.vf < DATE '2024-04-01' AND r.vt > DATE '2024-04-01' AND r.vt < DATE '2024-08-01'",
        "overlaps",
    );
}
#[test]
fn allen_finished_by() {
    allen_check(
        "r.vf < DATE '2024-04-01' AND r.vt = DATE '2024-08-01'",
        "finished_by",
    );
}
#[test]
fn allen_contains() {
    allen_check(
        "r.vf < DATE '2024-04-01' AND r.vt > DATE '2024-08-01'",
        "contains",
    );
}
#[test]
fn allen_starts() {
    allen_check(
        "r.vf = DATE '2024-04-01' AND r.vt < DATE '2024-08-01'",
        "starts",
    );
}
#[test]
fn allen_equals() {
    allen_check(
        "r.vf = DATE '2024-04-01' AND r.vt = DATE '2024-08-01'",
        "equals",
    );
}
#[test]
fn allen_started_by() {
    allen_check(
        "r.vf = DATE '2024-04-01' AND r.vt > DATE '2024-08-01'",
        "started_by",
    );
}
#[test]
fn allen_during() {
    allen_check(
        "r.vf > DATE '2024-04-01' AND r.vt < DATE '2024-08-01'",
        "during",
    );
}
#[test]
fn allen_finishes() {
    allen_check(
        "r.vf > DATE '2024-04-01' AND r.vt = DATE '2024-08-01'",
        "finishes",
    );
}
#[test]
fn allen_overlapped_by() {
    allen_check(
        "r.vf > DATE '2024-04-01' AND r.vf < DATE '2024-08-01' AND r.vt > DATE '2024-08-01'",
        "overlapped_by",
    );
}
#[test]
fn allen_met_by() {
    allen_check("r.vf = DATE '2024-08-01'", "met_by");
}
#[test]
fn allen_after() {
    allen_check("r.vf > DATE '2024-08-01'", "after");
}

/// Comprehensive Allen-relations bitemporal benchmark: a batch of edge-versions in
/// EACH of the 13 relations to a fixed query interval Q (so every relation is flexed
/// with a real, isolated count), on a graph carrying BOTH the RI-tree interval index
/// and regular vf/vt indexes, write-interleaved. Times each relation + combinations,
/// scan vs indexed, showing which index serves which shape. Run:
///   cargo test --release bench_allen_relations -- --ignored --nocapture
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_allen_relations() {
    use crate::temporal::{Date, Temporal};
    use std::time::Instant;

    const QF: i32 = 19_723; // ~2024-01-01 (days since epoch)
    const QT: i32 = 20_088; // ~2025-01-01
    const PER: usize = 4_000; // edge-versions per relation

    // Deterministic LCG (no rng crate, no Math.random); each build owns a fresh one.
    struct Rng(u64);
    impl Rng {
        fn r(&mut self, lo: i32, hi: i32) -> i32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((self.0 >> 20) % ((hi - lo).max(1) as u64)) as i32
        }
    }
    const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
    // (vf, vt) strictly inside relation `r` to Q=[QF,QT). Mutually exclusive by construction.
    let gen = |g: &mut Rng, r: usize| -> (i32, i32) {
        match r {
            0 => {
                let vf = QF - g.r(2000, 6000);
                (vf, QF - g.r(100, 1000))
            } // before
            1 => (QF - g.r(100, 4000), QF), // meets (vt=QF)
            2 => (QF - g.r(100, 4000), QF + g.r(100, QT - QF - 100)), // overlaps
            3 => (QF - g.r(100, 4000), QT), // finished_by (vt=QT)
            4 => (QF - g.r(100, 4000), QT + g.r(100, 4000)), // contains
            5 => (QF, QF + g.r(100, QT - QF - 100)), // starts (vf=QF)
            6 => (QF, QT),                  // equals
            7 => (QF, QT + g.r(100, 4000)), // started_by
            8 => {
                let vf = QF + g.r(100, QT - QF - 2000);
                (vf, vf + g.r(50, QT - vf - 100))
            } // during
            9 => (QF + g.r(100, QT - QF - 100), QT), // finishes (vt=QT)
            10 => (QF + g.r(100, QT - QF - 100), QT + g.r(100, 4000)), // overlapped_by
            11 => (QT, QT + g.r(100, 4000)), // met_by (vf=QT)
            _ => {
                let vf = QT + g.r(100, 4000);
                (vf, vf + g.r(30, 4000))
            } // after
        }
    };

    let names = [
        "before",
        "meets",
        "overlaps",
        "finished_by",
        "contains",
        "starts",
        "equals",
        "started_by",
        "during",
        "finishes",
        "overlapped_by",
        "met_by",
        "after",
    ];
    let dval = |d: i32| Value::Temporal(Temporal::Date(Date { days: d }));

    let build = |indexed: bool| -> (Graph, f64) {
        let mut g = ndjson::decode("").unwrap();
        if indexed {
            g.create_edge_interval_index("vf", "vt");
            g.create_edge_index("vf");
            g.create_edge_index("vt");
        }
        let a = g.add_vertex(&["N".to_string()], vec![]);
        let b = g.add_vertex(&["N".to_string()], vec![]);
        let t0 = Instant::now();
        // regenerate the same intervals deterministically for both builds
        let mut rng = Rng(SEED);
        for r in 0..13 {
            for _ in 0..PER {
                let (vf, vt) = gen(&mut rng, r);
                g.add_edge(
                    a,
                    b,
                    "R",
                    vec![("vf".to_string(), dval(vf)), ("vt".to_string(), dval(vt))],
                );
            }
        }
        (g, t0.elapsed().as_secs_f64() * 1000.0)
    };

    let (mut g_scan, w_scan) = build(false);
    let (mut g_idx, w_idx) = build(true);
    eprintln!(
        "\nAllen bench: {} edge-versions ({} per relation) | write: scan {w_scan:.0}ms | indexed {w_idx:.0}ms ({:.2}x)",
        g_scan.edge_count(), PER, w_idx / w_scan
    );

    let mut q = Params::new();
    q.insert(
        "qf".into(),
        super::eval::Val::Temporal(Temporal::Date(Date { days: QF })),
    );
    q.insert(
        "qt".into(),
        super::eval::Val::Temporal(Temporal::Date(Date { days: QT })),
    );

    let clauses = [
        "r.vt < $qf",                               // before
        "r.vt = $qf",                               // meets
        "r.vf < $qf AND r.vt > $qf AND r.vt < $qt", // overlaps
        "r.vf < $qf AND r.vt = $qt",                // finished_by
        "r.vf < $qf AND r.vt > $qt",                // contains
        "r.vf = $qf AND r.vt < $qt",                // starts
        "r.vf = $qf AND r.vt = $qt",                // equals
        "r.vf = $qf AND r.vt > $qt",                // started_by
        "r.vf > $qf AND r.vt < $qt",                // during
        "r.vf > $qf AND r.vt = $qt",                // finishes
        "r.vf > $qf AND r.vf < $qt AND r.vt > $qt", // overlapped_by
        "r.vf = $qt",                               // met_by
        "r.vf > $qt",                               // after
    ];

    let bench = |g: &mut Graph, q: &str, params: &Params, iters: usize| -> (f64, i64) {
        let prep = parse(q).unwrap();
        let matched = match prep
            .execute(g, params)
            .unwrap()
            .rows()
            .next()
            .and_then(|r| r.first().cloned())
        {
            Some(Value::Num(n)) => n as i64,
            _ => -1,
        };
        let mut ms = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            std::hint::black_box(prep.execute(g, params).unwrap().rows().count());
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
        (ms[ms.len() / 2], matched)
    };

    eprintln!(
        "  {:<14} {:>10} {:>10} {:>8}  {:>8}",
        "relation", "scan", "indexed", "speedup", "matched"
    );
    for (i, cl) in clauses.iter().enumerate() {
        let query = format!("MATCH ()-[r:R]->() WHERE {cl} RETURN count(*) AS n");
        let (scan, mc) = bench(&mut g_scan, &query, &q, 30);
        let (idx, mi) = bench(&mut g_idx, &query, &q, 30);
        assert_eq!(mc, mi, "{}: indexed count != scan count", names[i]);
        eprintln!(
            "  {:<14} {scan:>8.3}ms {idx:>8.3}ms {:>7.2}x  {mc:>8}",
            names[i],
            scan / idx
        );
    }

    // Combinations.
    let combos: [(&str, &str); 4] = [
        ("intersects Q (o∪s∪d∪…)", "r.vf < $qt AND r.vt > $qf"),
        (
            "overlaps OR contains",
            "(r.vf < $qf AND r.vt > $qf AND r.vt < $qt) OR (r.vf < $qf AND r.vt > $qt)",
        ),
        (
            "during ∧ ends-early",
            "r.vf > $qf AND r.vt < $qt AND r.vt < $qf + 100",
        ),
        (
            "not(before) ∧ not(after)",
            "NOT (r.vt < $qf) AND NOT (r.vf > $qt)",
        ),
    ];
    eprintln!("  -- combinations --");
    for (label, cl) in combos {
        let query = format!("MATCH ()-[r:R]->() WHERE {cl} RETURN count(*) AS n");
        let (scan, mc) = bench(&mut g_scan, &query, &q, 20);
        let (idx, mi) = bench(&mut g_idx, &query, &q, 20);
        assert_eq!(mc, mi, "{label}: indexed count != scan count");
        eprintln!(
            "  {label:<26} {scan:>8.3}ms {idx:>8.3}ms {:>7.2}x  {mc:>8}",
            scan / idx
        );
    }
}

/// Bitemporal 4-way as-of: two RI-tree interval indexes (valid [vf,vt) + transaction
/// [tf,tt)) intersect so a query "as of valid V, as believed at T" returns the version
/// believed at T — the *same* valid point V yields different rows for different T, which
/// is the whole point of transaction time. e0 and e1 share a valid interval but e1 is a
/// later belief (correction) that supersedes e0. Correct across index configs.
#[test]
fn bitemporal_4way_asof_intersects_valid_and_tx() {
    // e0: valid [2020,2025) believed [2020,2023);  e1: valid [2020,2025) believed [2023,9999).
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{"id":"e0","vf":{"@date":"2020-01-01"},"vt":{"@date":"2025-01-01"},"tf":{"@date":"2020-01-01"},"tt":{"@date":"2023-01-01"}}}"#,
        r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"id":"e1","vf":{"@date":"2020-01-01"},"vt":{"@date":"2025-01-01"},"tf":{"@date":"2023-01-01"},"tt":{"@date":"9999-12-31"}}}"#,
    ];
    let q = "MATCH ()-[r:R]->() \
        WHERE r.vf <= $v AND r.vt > $v AND r.tf <= $t AND r.tt > $t \
        RETURN r.id AS id ORDER BY id";
    let make = |indexed: bool| {
        let mut g = graph_of(&lines);
        if indexed {
            g.create_edge_interval_index("vf", "vt"); // valid-time
            g.create_edge_interval_index("tf", "tt"); // transaction-time
        }
        g
    };
    let params = |v: &str, t: &str| {
        let mut m = Params::new();
        let d = |s: &str| {
            super::eval::Val::Temporal(crate::temporal::Temporal::parse("date", s).unwrap())
        };
        m.insert("v".into(), d(v));
        m.insert("t".into(), d(t));
        m
    };

    // valid 2022, believed 2021 → e0 (e1 not yet believed).
    for indexed in [false, true] {
        let mut g = make(indexed);
        assert_eq!(
            qp(&mut g, q, params("2022-01-01", "2021-01-01")),
            vec![vec![s("e0")]],
            "as-of valid 2022 believed 2021 (indexed={indexed})"
        );
        // same valid point, later belief 2024 → e1 (the correction).
        assert_eq!(
            qp(&mut g, q, params("2022-01-01", "2024-01-01")),
            vec![vec![s("e1")]],
            "as-of valid 2022 believed 2024 (indexed={indexed})"
        );
    }
}

/// contains-window ("valid THROUGHOUT [d1,d2]"): `vf <= d1 AND vt >= d2` returns exactly
/// the intervals covering the whole window, seeking via the interval index (a superset
/// the WHERE refines), identical across index configs.
#[test]
fn contains_window_valid_throughout() {
    let edge = |id: &str, vf: &str, vt: &str| {
        format!(
            r#"{{"type":"edge","id":"{id}","from":"a","to":"b","labels":["R"],"properties":{{"id":"{id}","vf":{{"@date":"{vf}"}},"vt":{{"@date":"{vt}"}}}}}}"#
        )
    };
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#.to_string(),
        r#"{"type":"node","id":"b","labels":["N"],"properties":{}}"#.to_string(),
        edge("covers", "2024-01-01", "2024-12-01"), // ⊇ [Apr,Aug)
        edge("exact", "2024-04-01", "2024-08-01"),  // ⊇ (equal)
        edge("starts_inside", "2024-06-01", "2024-12-01"), // vf>Apr → not throughout
        edge("ends_inside", "2024-01-01", "2024-06-01"), // vt<Aug → not throughout
        edge("disjoint", "2024-01-01", "2024-03-01"), // before
    ];
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let q = "MATCH ()-[r:R]->() WHERE r.vf <= DATE '2024-04-01' AND r.vt >= DATE '2024-08-01' RETURN r.id AS id ORDER BY id";
    for indexed in [false, true] {
        let mut g = graph_of(&refs);
        if indexed {
            g.create_edge_interval_index("vf", "vt");
        }
        assert_eq!(
            rows(&mut g, q),
            vec![vec![s("covers")], vec![s("exact")]],
            "contains-window must return only the throughout-covering intervals (indexed={indexed})"
        );
    }
}

/// Bitemporal 4-way as-of at scale: valid AND transaction intervals varied so
/// valid-at-V and believed-at-T each select ~half, their intersection ~a quarter.
/// Compares scan vs one interval index (valid only, then filter tx) vs two interval
/// indexes (valid ∩ tx). Run: cargo test --release bench_bitemporal_4way -- --ignored --nocapture
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_bitemporal_4way() {
    use crate::temporal::{Date, Temporal};
    use std::time::Instant;
    const N: usize = 40_000;
    const V: i32 = 19_900; // query valid point
    const T: i32 = 20_000; // query belief point

    struct Rng(u64);
    impl Rng {
        fn r(&mut self, lo: i32, hi: i32) -> i32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((self.0 >> 20) % ((hi - lo).max(1) as u64)) as i32
        }
    }
    let dval = |d: i32| Value::Temporal(Temporal::Date(Date { days: d }));
    // mode: 0 none, 1 valid-only interval index, 2 valid+tx interval indexes.
    let build = |mode: u8| -> Graph {
        let mut g = ndjson::decode("").unwrap();
        if mode >= 1 {
            g.create_edge_interval_index("vf", "vt");
        }
        if mode == 2 {
            g.create_edge_interval_index("tf", "tt");
        }
        let a = g.add_vertex(&["N".to_string()], vec![]);
        let b = g.add_vertex(&["N".to_string()], vec![]);
        const INF: i32 = 3_000_000; // far-future "believed now" sentinel
        let mut rng = Rng(0x51ed_2701);
        for _ in 0..N {
            // Valid axis is selective (wide start spread, narrow durations → ~7% at V).
            // Transaction axis is deliberately NON-selective: ~90% are current belief
            // (tt = ∞), so tx-stab(T="now") matches ~everything — the "current org chart"
            // case where the old two-stab-intersect LOST to a scan. The fix must seed
            // from the selective valid axis and verify tx, never materialize the tx stab.
            let vf = V - rng.r(0, 30_000);
            let vt = vf + rng.r(50, 2000);
            let tf = T - rng.r(0, 30_000);
            let tt = if rng.r(0, 10) == 0 {
                tf + rng.r(50, 2000)
            } else {
                INF
            };
            g.add_edge(
                a,
                b,
                "R",
                vec![
                    ("vf".to_string(), dval(vf)),
                    ("vt".to_string(), dval(vt)),
                    ("tf".to_string(), dval(tf)),
                    ("tt".to_string(), dval(tt)),
                ],
            );
        }
        g
    };
    let (mut g0, mut g1, mut g2) = (build(0), build(1), build(2));
    let mut q = Params::new();
    let td = |d: i32| super::eval::Val::Temporal(Temporal::Date(Date { days: d }));
    q.insert("v".into(), td(V));
    q.insert("t".into(), td(T));
    let query = "MATCH ()-[r:R]->() WHERE r.vf <= $v AND r.vt > $v AND r.tf <= $t AND r.tt > $t RETURN count(*) AS n";

    let bench = |g: &mut Graph| -> (f64, i64) {
        let prep = parse(query).unwrap();
        let mc = match prep
            .execute(g, &q)
            .unwrap()
            .rows()
            .next()
            .and_then(|r| r.first().cloned())
        {
            Some(Value::Num(n)) => n as i64,
            _ => -1,
        };
        let mut ms = Vec::with_capacity(30);
        for _ in 0..30 {
            let t = Instant::now();
            std::hint::black_box(prep.execute(g, &q).unwrap().rows().count());
            ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (ms[ms.len() / 2], mc)
    };
    let (s, ms) = bench(&mut g0);
    let (o, mo) = bench(&mut g1);
    let (b, mb) = bench(&mut g2);
    assert_eq!(ms, mo);
    assert_eq!(ms, mb); // all three agree on the answer
    eprintln!("\nbitemporal 4-way ({N} versions), matched={ms}:");
    eprintln!("  scan                {s:>8.3}ms  1.00x");
    eprintln!(
        "  valid index only    {o:>8.3}ms  {:.2}x  (stab(V), filter tx)",
        s / o
    );
    eprintln!(
        "  valid ∩ tx indexes  {b:>8.3}ms  {:.2}x  (stab(V) ∩ stab(T))",
        s / b
    );
}

// --- fuzzer-found byte-identity regressions ----------------------------------
// Each of these pins a divergence the differential fuzzer surfaced between this
// engine and the TS twin. They are the permanent guards; the fuzzer itself is
// randomized and only rediscovers a regression by luck.

#[test]
fn stddev_over_non_numeric_values_is_null_not_zero() {
    // A non-numeric value coerces to NaN, so the variance is NaN — and must STAY
    // NaN (→ null on output), like `avg`. The negative-variance clamp used
    // `f64::max`, which returns the non-NaN operand (`NAN.max(0.0)` == 0.0) and so
    // reported a real 0. JS `Math.max(0, NaN)` is NaN, so the TS twin said null.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a"}}"#,
        r#"{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z"}}"#,
    ]);
    // NaN is the in-engine spelling of "no value"; it serializes to JSON null,
    // which is what the TS twin returns. A real 0 would serialize as 0.
    let is_nan = |g: &mut Graph, q: &str| match rows(g, q).as_slice() {
        [row] => matches!(row.as_slice(), [Value::Num(x)] if x.is_nan()),
        _ => false,
    };
    assert!(is_nan(&mut g, "MATCH (n:T) RETURN stddev_pop(n.s) AS x"));
    assert!(is_nan(&mut g, "MATCH (n:T) RETURN stddev_samp(n.s) AS x"));
    // A single non-numeric value among numbers poisons the whole aggregate.
    assert!(is_nan(
        &mut g,
        "MATCH (n:T) RETURN stddev_pop(CASE WHEN n.n = 3 THEN 3 ELSE 'a' END) AS x"
    ));
    // Not a zero — the bug reported a real 0 here.
    assert_ne!(
        rows(&mut g, "MATCH (n:T) RETURN stddev_pop(n.s) AS x"),
        vec![vec![n(0.0)]]
    );
}

#[test]
fn stddev_over_numbers_is_unaffected_by_the_nan_guard() {
    // The NaN guard must not disturb the ordinary numeric path, including the
    // genuinely-zero case (all values equal → variance 0) it has to stay distinct
    // from, and the tiny-negative cancellation the clamp exists for.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"k":5}}"#,
        r#"{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"k":5}}"#,
    ]);
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN stddev_pop(n.n) AS x"),
        vec![vec![n(2.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN stddev_pop(n.k) AS x"),
        vec![vec![n(0.0)]]
    );
    // stddev_samp needs 2+ rows; stddev_pop needs 1+. Neither is NaN here.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN stddev_samp(n.k) AS x"),
        vec![vec![n(0.0)]]
    );
}

#[test]
fn degrees_and_radians_use_the_multiply_then_divide_association() {
    // `f64::to_degrees` is `n * (180/PI)` — it pre-rounds the constant and lands
    // one ulp from `(n * 180) / PI`, which is what the TS twin computes. Multiply
    // and divide are exactly specified by IEEE 754, so the two forms are
    // reproducible and must be spelled the same way in both engines.
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN degrees(1e100) AS x"),
        vec![vec![n((1e100 * 180.0) / std::f64::consts::PI)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN radians(3) AS x"),
        vec![vec![n((3.0 * std::f64::consts::PI) / 180.0)]]
    );
    // The association actually differs for these inputs — guard against a silent
    // revert to `to_degrees`/`to_radians`.
    assert_ne!(
        1e100_f64.to_degrees(),
        (1e100 * 180.0) / std::f64::consts::PI
    );
    assert_ne!(3.0_f64.to_radians(), (3.0 * std::f64::consts::PI) / 180.0);
}

#[test]
fn degrees_and_radians_round_trip_the_common_angles() {
    let mut g = modern();
    assert_eq!(rows(&mut g, "RETURN degrees(0) AS x"), vec![vec![n(0.0)]]);
    assert_eq!(rows(&mut g, "RETURN radians(0) AS x"), vec![vec![n(0.0)]]);
    assert_eq!(
        rows(&mut g, "RETURN degrees(pi()) AS x"),
        vec![vec![n(180.0)]]
    );
}

#[test]
fn zero_limit_returns_no_rows_without_evaluating_the_projection() {
    // `LIMIT 0` emits no rows, so a faulting projection never runs — the same rule
    // the engine already follows for a non-zero LIMIT ("project exactly the rows you
    // emit"). Both engines used to decide this by whether an ORDER BY was present,
    // and took OPPOSITE branches.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"1","labels":["T"],"properties":{"n":3}}"#,
        r#"{"type":"node","id":"2","labels":["T"],"properties":{"n":7}}"#,
    ]);
    let empty: Vec<Vec<Value>> = vec![];
    assert_eq!(rows(&mut g, "MATCH (n:T) RETURN 1/0 AS x LIMIT 0"), empty);
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN 1/0 AS x, n.n AS t ORDER BY t LIMIT 0"
        ),
        empty
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN DISTINCT 1/0 AS x LIMIT 0"),
        empty
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN sum(1/0) AS x LIMIT 0"),
        empty
    );
    // An intermediate WITH carries the same rule.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) WITH 1/0 AS x LIMIT 0 RETURN x"),
        empty
    );
}

#[test]
fn a_nonzero_limit_still_evaluates_and_still_faults() {
    // The short-circuit is for LIMIT 0 ONLY — a limit that keeps any row must
    // still project it (and still raise the projection's fault).
    let mut g = graph_of(&[
        r#"{"type":"node","id":"1","labels":["T"],"properties":{"n":3}}"#,
        r#"{"type":"node","id":"2","labels":["T"],"properties":{"n":7}}"#,
    ]);
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN n.n AS x ORDER BY x LIMIT 1"),
        vec![vec![n(3.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN n.n AS x ORDER BY x LIMIT 2"),
        vec![vec![n(3.0)], vec![n(7.0)]]
    );
    assert!(parse("MATCH (n:T) RETURN 1/0 AS x LIMIT 1")
        .unwrap()
        .execute(&mut g, &Params::new())
        .is_err());
    // SKIP past the end still evaluates — only a zero LIMIT short-circuits.
    assert!(parse("MATCH (n:T) RETURN 1/0 AS x SKIP 5")
        .unwrap()
        .execute(&mut g, &Params::new())
        .is_err());
}

#[test]
fn distinct_and_group_by_treat_every_nan_as_one_value() {
    // The grouping key was the raw bit pattern, but NaNs differ by sign and
    // payload depending on which operation made them (`ln(-1)` vs `x / NaN`), so
    // two NaNs landed in different DISTINCT groups. The engine's total order says
    // NaN == NaN, and the TS twin keys all non-finite values by their rendered
    // form, so both must collapse to one.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"x":-1}}"#,
        r#"{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"x":4}}"#,
    ]);
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN count(DISTINCT log('nan', n.x)) AS x"
        ),
        vec![vec![n(1.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN count(DISTINCT (log('inf', n.x) * n.x)) AS x"
        ),
        vec![vec![n(1.0)]]
    );
    // A DISTINCT collect over the same NaNs yields ONE element.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN size(collect_list(DISTINCT log('nan', n.x))) AS x"
        ),
        vec![vec![n(1.0)]]
    );
}

#[test]
fn distinct_collapses_signed_zero_but_not_ordinary_values() {
    // The canonicalization must not collapse anything else.
    let mut g = graph_of(&[
        r#"{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a"}}"#,
        r#"{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z"}}"#,
    ]);
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN count(DISTINCT n.n) AS x"),
        vec![vec![n(2.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN count(DISTINCT n.s) AS x"),
        vec![vec![n(2.0)]]
    );
    // -0.0 and 0.0 are ONE group: `-0 = 0` is true, and the distinction is
    // normalized everywhere else (ORDER BY, sign(), the result JSON, the property
    // index), so two groups here would both render as `0`.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN count(DISTINCT CASE WHEN n.n = 3 THEN 0.0 ELSE -0.0 END) AS x"
        ),
        vec![vec![n(1.0)]]
    );
    // And a GROUP BY over the same values yields one group holding both rows.
    //
    // Bound with `LET` so the grouping is REAL. Spelled `RETURN … AS k … GROUP BY
    // k` this asserted nothing: `GROUP BY` cannot see a RETURN alias, the key read
    // as NULL, and every row landed in one group whatever the values were — so the
    // -0.0/0.0 question it exists to ask was never put.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) LET k = CASE WHEN n.n = 3 THEN 0.0 ELSE -0.0 END RETURN k, count(*) AS c GROUP BY k"
        ),
        vec![vec![n(0.0), n(2.0)]]
    );
    // The same shape over values that are genuinely different must NOT collapse —
    // otherwise the assertion above still passes when grouping is broken.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) LET k = CASE WHEN n.n = 3 THEN 1.0 ELSE 2.0 END RETURN k, count(*) AS c GROUP BY k ORDER BY k"
        ),
        vec![vec![n(1.0), n(1.0)], vec![n(2.0), n(1.0)]]
    );
    // Infinities stay distinct from each other and from NaN.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) RETURN count(DISTINCT CASE WHEN n.n = 3 THEN 1e100 ELSE -1e100 END) AS x"
        ),
        vec![vec![n(2.0)]]
    );
}

#[test]
fn a_numeric_string_that_overflows_is_not_a_number() {
    // '1e1000' is a syntactically-valid literal that parses to Infinity. Both
    // engines read a non-finite parse as "not a number" (NaN → null), so it never
    // reaches a math function as Infinity.
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN (to_float('1e1000') IS NULL) AS x"),
        vec![vec![b(true)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN (to_float('-1e1000') IS NULL) AS x"),
        vec![vec![b(true)]]
    );
    // Just inside the range still converts.
    assert_eq!(
        rows(&mut g, "RETURN (to_float('1e300') IS NULL) AS x"),
        vec![vec![b(false)]]
    );
    assert_eq!(
        rows(&mut g, "RETURN sqrt('1e300') AS x"),
        vec![vec![n(1e150)]]
    );
}

#[test]
fn the_total_order_ties_every_pair_in_the_catch_all_rank() {
    // `type_rank` 4 holds graph elements, lists, and records. Same-kind pairs
    // compare structurally (element-wise / field-wise); a MIXED pair — a list vs a
    // record — is Equal, so a stable sort leaves it in input order. The TS twin
    // fell through to JS `<`, which compared their string coercions ("1,2" vs
    // "[object Map]") and invented an order this engine does not have.
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN list_sort([{a: 1}, [1, 2]]) AS x"),
        rows(&mut g, "RETURN [{a: 1}, [1, 2]] AS x")
    );
    assert_eq!(
        rows(&mut g, "RETURN list_sort([[1, 2], {a: 1}]) AS x"),
        rows(&mut g, "RETURN [[1, 2], {a: 1}] AS x")
    );
}

#[test]
fn the_total_order_still_separates_the_type_groups_and_sorts_within_them() {
    // The rank-4 tie must not weaken the rest of the order: the groups stay
    // number < string < boolean < temporal < other, and same-kind values inside
    // rank 4 still sort structurally.
    let mut g = modern();
    assert_eq!(
        rows(
            &mut g,
            "RETURN list_sort([[1, 2], date('2020-01-01'), {a: 1}, true, 'z', 3]) AS x"
        ),
        rows(
            &mut g,
            "RETURN [3, 'z', true, date('2020-01-01'), [1, 2], {a: 1}] AS x"
        )
    );
    // Lists sort element-wise, records field-wise — both still total.
    assert_eq!(
        rows(&mut g, "RETURN list_sort([[3], [1], [2]]) AS x"),
        rows(&mut g, "RETURN [[1], [2], [3]] AS x")
    );
    assert_eq!(
        rows(&mut g, "RETURN list_sort([{b: 1}, {a: 1}]) AS x"),
        rows(&mut g, "RETURN [{a: 1}, {b: 1}] AS x")
    );
}

#[test]
fn range_is_bounded_and_faults_past_the_budget() {
    // A GQL list is materialized, so an unbounded `range` is an OOM kill rather
    // than a query error. Past `RANGE_BUDGET` the call faults loudly.
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN size(range(0, 999999)) AS x"),
        vec![vec![n(1_000_000.0)]]
    );
    let err = parse("RETURN size(range(0, 1000000)) AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ResourceExhausted);
    // The case that used to hang the process outright.
    let err = parse("RETURN size(range(0, 1e21)) AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ResourceExhausted);
    // A wide step brings a wide span back under the budget.
    assert_eq!(
        rows(&mut g, "RETURN size(range(0, 1000000, 2)) AS x"),
        vec![vec![n(500_001.0)]]
    );
}

#[test]
fn range_terminates_past_the_float_step_stall() {
    // `i += 1.0` is a NO-OP once `i` reaches 2^53, so a comparison-driven loop
    // (`while i <= e`) never terminates there — even for a 3-element span. The
    // loop is count-driven, so it ends; the repeated-addition values themselves
    // stall, which is the honest f64 result and identical in both engines.
    let mut g = modern();
    assert_eq!(
        rows(
            &mut g,
            "RETURN size(range(to_float('9007199254740992'), to_float('9007199254740994'))) AS x"
        ),
        vec![vec![n(3.0)]]
    );
}

#[test]
fn range_keeps_its_ordinary_semantics() {
    // The bound must not disturb the normal cases: inclusive of both bounds, a
    // negative step counts down, an empty span yields [], a zero step is null.
    let mut g = modern();
    assert_eq!(
        rows(&mut g, "RETURN range(0, 5) AS x"),
        rows(&mut g, "RETURN [0, 1, 2, 3, 4, 5] AS x")
    );
    assert_eq!(
        rows(&mut g, "RETURN range(5, 0, -1) AS x"),
        rows(&mut g, "RETURN [5, 4, 3, 2, 1, 0] AS x")
    );
    assert_eq!(
        rows(&mut g, "RETURN range(0, 10, 3) AS x"),
        rows(&mut g, "RETURN [0, 3, 6, 9] AS x")
    );
    assert_eq!(
        rows(&mut g, "RETURN range(0, 0) AS x"),
        rows(&mut g, "RETURN [0] AS x")
    );
    assert_eq!(
        rows(&mut g, "RETURN range(5, 0) AS x"),
        rows(&mut g, "RETURN [] AS x")
    );
    assert_eq!(
        rows(&mut g, "RETURN range(0, 10, 0) AS x"),
        vec![vec![Value::Null]]
    );
}

#[test]
fn settings_are_configurable_per_graph() {
    // The budgets are host policy, not semantics: a query under the ceiling
    // behaves identically whatever the ceiling is, and tripping one is always a
    // loud E_RESOURCE_EXHAUSTED.
    use crate::graph::ConfigId;
    let mut g = modern();
    assert_eq!(g.limits().range, 1_000_000);
    // Over the default ceiling → faults.
    assert!(parse("RETURN size(range(0, 1000000)) AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .is_err());
    // Raise it and the same query runs.
    assert!(g.set_config(ConfigId::LimitsRange, 5_000_000));
    assert_eq!(
        rows(&mut g, "RETURN size(range(0, 1000000)) AS x"),
        vec![vec![n(1_000_001.0)]]
    );
    // Lower it and a much smaller range now trips.
    assert!(g.set_config(ConfigId::LimitsTrail, 25));
    assert!(g.set_config(ConfigId::LimitsRange, 10));
    assert!(parse("RETURN size(range(0, 20)) AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .is_err());
    // Under the new ceiling it still works, unchanged.
    assert_eq!(
        rows(&mut g, "RETURN size(range(0, 5)) AS x"),
        vec![vec![n(6.0)]]
    );
    assert_eq!(g.limits().trail, 25);
}

#[test]
fn an_unknown_or_zero_setting_is_rejected_not_silently_ignored() {
    // A host talking to an older artifact must be able to tell that its limit was
    // not applied — silently running with the default is the failure mode this
    // guards against. Zero is rejected for the same reason: it would fail every
    // query, so it is never the intent.
    use crate::graph::ConfigId;
    let mut g = modern();
    assert!(crate::graph::ConfigId::from_u32(0).is_some());
    assert!(crate::graph::ConfigId::from_u32(3).is_some());
    assert!(crate::graph::ConfigId::from_u32(99).is_none());
    assert!(!g.set_config(ConfigId::LimitsRange, 0));
    // The rejected write left the ceiling alone.
    assert_eq!(g.limits().range, 1_000_000);
}

#[test]
fn settings_survive_a_graph_clone() {
    use crate::graph::ConfigId;
    let mut g = modern();
    assert!(g.set_config(ConfigId::LimitsRange, 42));
    assert!(g.set_config(ConfigId::LimitsIntermediate, 4242));
    let copy = g.clone();
    assert_eq!(copy.limits().range, 42);
    assert_eq!(copy.limits().intermediate, 4242);
    assert_eq!(copy.limits().trail, 1_000_000);
}

#[test]
fn the_operator_chain_ceiling_lives_in_the_same_config_space() {
    // It used to be its own field with its own FFI export; folding it in is the
    // point of the config space. The named getter still reads it.
    use crate::graph::ConfigId;
    let mut g = modern();
    assert_eq!(g.max_operator_chain(), 10_000);
    assert_eq!(g.limits().operator_chain, 10_000);
    assert!(g.set_config(ConfigId::LimitsOperatorChain, 25));
    assert_eq!(g.max_operator_chain(), 25);
    // The named alias writes the same slot.
    g.set_max_operator_chain(77);
    assert_eq!(g.config().limits.operator_chain, 77);
}

// --- ISO <order by and page statement> in STATEMENT position ------------------
// The ISO grammar puts `orderByAndPageStatement` in TWO places: trailing a RETURN
// (`primitiveResultStatement`) and as a pipeline step of its own
// (`primitiveQueryStatement`). These cover the second form, which sorts/slices the
// working BINDING table before any projection runs.

fn paged() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"1","labels":["T"],"properties":{"n":3,"s":"a"}}"#,
        r#"{"type":"node","id":"2","labels":["T"],"properties":{"n":7,"s":"z"}}"#,
        r#"{"type":"node","id":"3","labels":["T"],"properties":{"n":5,"s":"m"}}"#,
    ])
}

#[test]
fn a_standalone_order_by_statement_sorts_the_binding_table() {
    let mut g = paged();
    assert_eq!(
        rows(&mut g, "MATCH (n:T) ORDER BY n.n RETURN n.n AS x"),
        vec![vec![n(3.0)], vec![n(5.0)], vec![n(7.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) ORDER BY n.n DESC RETURN n.n AS x"),
        vec![vec![n(7.0)], vec![n(5.0)], vec![n(3.0)]]
    );
    // A string key sorts by the same total order as a projection's ORDER BY.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) ORDER BY n.s RETURN n.s AS x"),
        vec![vec![s("a")], vec![s("m")], vec![s("z")]]
    );
}

#[test]
fn a_standalone_limit_or_offset_statement_slices_the_binding_table() {
    let mut g = paged();
    assert_eq!(
        rows(&mut g, "MATCH (n:T) ORDER BY n.n LIMIT 2 RETURN n.n AS x"),
        vec![vec![n(3.0)], vec![n(5.0)]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n:T) ORDER BY n.n OFFSET 1 RETURN n.n AS x"),
        vec![vec![n(5.0)], vec![n(7.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) ORDER BY n.n OFFSET 1 LIMIT 1 RETURN n.n AS x"
        ),
        vec![vec![n(5.0)]]
    );
    // OFFSET past the end is empty, not an error.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) ORDER BY n.n OFFSET 99 RETURN n.n AS x"),
        Vec::<Vec<Value>>::new()
    );
}

#[test]
fn a_standalone_page_statement_runs_before_the_projection() {
    // The semantic point of the statement form: paging trims the binding table, so
    // the RETURN only ever projects the survivors — a faulting projection on a
    // dropped row never runs. (The trailing form reaches the same answer for
    // LIMIT 0, but for a different reason; see the note on `project_to_rows`.)
    let mut g = paged();
    assert_eq!(
        rows(&mut g, "MATCH (n:T) ORDER BY n.n LIMIT 0 RETURN 1/0 AS x"),
        Vec::<Vec<Value>>::new()
    );
    // Only the kept row is projected, so dividing by `n.n - 7` is safe here.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) ORDER BY n.n LIMIT 1 RETURN 1/(n.n - 7) AS x"
        ),
        vec![vec![n(-0.25)]]
    );
}

#[test]
fn a_page_statement_composes_with_the_other_statements() {
    let mut g = paged();
    // After a FILTER, and feeding an aggregate.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) FILTER n.n > 3 ORDER BY n.n LIMIT 1 RETURN n.n AS x"
        ),
        vec![vec![n(5.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) ORDER BY n.n LIMIT 2 RETURN count(*) AS c"
        ),
        vec![vec![n(2.0)]]
    );
    // Two paging statements in a row: the second applies to the first's output.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (n:T) ORDER BY n.n LIMIT 2 ORDER BY n.n DESC LIMIT 1 RETURN n.n AS x"
        ),
        vec![vec![n(5.0)]]
    );
    // The trailing form still works, and is unaffected.
    assert_eq!(
        rows(&mut g, "MATCH (n:T) RETURN n.n AS x ORDER BY x LIMIT 2"),
        vec![vec![n(3.0)], vec![n(5.0)]]
    );
}

#[test]
fn a_page_statement_takes_a_dynamic_bound() {
    // OFFSET/LIMIT accept a `$param` here exactly as they do trailing a RETURN —
    // including the up-front bound check, so an unbound one is a clean
    // MissingParameter rather than a surprise at row time.
    let mut g = paged();
    let mut params = Params::new();
    params.insert("k".to_string(), super::eval::Val::Num(2.0));
    assert_eq!(
        qp(
            &mut g,
            "MATCH (n:T) ORDER BY n.n LIMIT $k RETURN n.n AS x",
            params
        ),
        vec![vec![n(3.0)], vec![n(5.0)]]
    );
    let err = parse("MATCH (n:T) ORDER BY n.n LIMIT $k RETURN n.n AS x")
        .unwrap()
        .execute(&mut g, &Params::new())
        .unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::MissingParameter);
}

/// Columnar vs scalar for a quantified unit that exposes GROUP variables.
///
/// The port was first measured 1.25-2.65x SLOWER here, which is why it was held
/// back; the cause was `Val::List` deep-copying on every per-row `Binding` clone.
/// Run: `cargo test --release bench_quantified_group_vars -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_quantified_group_vars() {
    let mut g = layered_dense(6, 6);
    let cases: &[(&str, &str)] = &[
        (
            "-[:R]->{1,4} (no group vars)",
            "MATCH (a:N)-[:R]->{1,4}(b) RETURN element_id(b) AS i",
        ),
        (
            "((x)-[]->(y)){1,4} rows",
            "MATCH (a:N)((x)-[]->(y)){1,4}(b) RETURN element_id(b) AS i",
        ),
        (
            "((x)-[e]->(y)){1,4} + size",
            "MATCH (a:N)((x)-[e]->(y)){1,4}(b) RETURN size(e) AS n",
        ),
        (
            "… WHERE size(e) >= 2",
            "MATCH (a:N)((x)-[e]->(y)){1,4}(b) WHERE size(e) >= 2 RETURN element_id(b) AS i",
        ),
    ];

    println!(
        "\n{:<30} {:>10} {:>10} {:>8}",
        "query", "columnar", "scalar", "ratio"
    );

    for (name, q) in cases {
        let mut best = [f64::MAX; 2];

        for (k, on) in [true, false].iter().enumerate() {
            for _ in 0..7 {
                let t = std::time::Instant::now();
                let n = super::eval::with_vec_override(*on, || rows(&mut g, q).len());
                assert!(n > 0, "[{name}] produced no rows");
                best[k] = best[k].min(t.elapsed().as_secs_f64() * 1e6);
            }
        }

        println!(
            "{name:<30} {:>9.0}us {:>9.0}us {:>7.2}x",
            best[0],
            best[1],
            best[0] / best[1]
        );
    }
}

/// Fusing comma patterns must not change the ROWS or their ORDER.
///
/// Fusion rewrites `MATCH (a)-[]->(b), (b)-[]->(c)` into one path. The rows are
/// the same set by construction; the risk is ORDER, which nothing else in the
/// suite pins for a multi-pattern MATCH — and which the TS engine, joining
/// unfused, would then disagree with. Each case is run with fusion on and off
/// and compared position-by-position.
#[test]
fn fusing_comma_patterns_preserves_rows_and_order() {
    let mut g = layered_dense(4, 4);
    let queries = [
        // chains — fused
        "MATCH (a:N)-[:R]->(b), (b)-[:R]->(c) RETURN element_id(a), element_id(b), element_id(c)",
        "MATCH (a:N)-[:R]->(b), (b)-[:R]->(c), (c)-[:R]->(d) RETURN element_id(a), element_id(d)",
        "MATCH (a:N)-[r:R]->(b), (b)-[s:R]->(c) RETURN element_id(r), element_id(s)",
        // the shared node is constrained on BOTH sides — the constraints merge.
        // The inline property is what makes this discriminating: every node here
        // is an `N`, so a dropped label would not change the answer, but a
        // dropped `{id: …}` would return far more rows.
        "MATCH (a:N)-[:R]->(b), (b:N {id: 'n5'})-[:R]->(c) RETURN element_id(a), element_id(c)",
        "MATCH (a:N)-[:R]->(b {id: 'n5'}), (b)-[:R]->(c) RETURN element_id(a), element_id(c)",
        "MATCH (a:N)-[:R]->(b {id: 'n5'}), (b {id: 'n5'})-[:R]->(c) RETURN element_id(a), element_id(c)",
        "MATCH (a:N)-[:R]->(b), (b)-[:R]->(c) WHERE b.id = 'n5' RETURN element_id(a), element_id(c)",
        // converging: (c) points AT the shared node — the second pattern reverses
        "MATCH (a:N)-[:R]->(b), (c)-[:R]->(b) RETURN element_id(a), element_id(c)",
        // diverging: both patterns leave the SAME node — the first reverses
        "MATCH (b)-[:R]->(a:N), (b)-[:R]->(c) RETURN element_id(a), element_id(c)",
        "MATCH (b)-[:R]->(a:N), (b)-[:R]->(c), (c)-[:R]->(d) RETURN element_id(a), element_id(d)",
        // disconnected — a cartesian product, never fusable
        "MATCH (a:N)-[:R]->(b), (c:N)-[:R]->(d) RETURN element_id(a), element_id(c)",
        // with a filter that references both patterns' variables
        "MATCH (a:N)-[:R]->(b), (b)-[:R]->(c) WHERE a.id <> c.id RETURN element_id(a), element_id(c)",
    ];

    for q in queries {
        let fused = rows(&mut g, q);
        let plain = super::eval::without_fusion(|| rows(&mut g, q));

        assert!(!fused.is_empty(), "`{q}` produced no rows — inert test");
        assert_eq!(fused, plain, "fusion changed the result of `{q}`");
    }
}

/// The shared node's constraints must all survive a fusion.
///
/// Split from `fusing_comma_patterns_preserves_rows_and_order` because that
/// test's fixture cannot discriminate them: every node there carries the single
/// label `N`, so a dropped label conjunction changes nothing, and it has no
/// inline node `WHERE`. Deleting the label merge or the `where_` merge left that
/// test green. This fixture gives nodes two labels and uses inline `WHERE`, so
/// each branch of `merge_node` has a case that fails without it.
#[test]
fn fusing_merges_every_constraint_on_the_shared_node() {
    let lines: Vec<String> = (0..8)
        .map(|i| {
            // Half the nodes are `N` only, half are both `N` and `M`.
            let labels = if i % 2 == 0 { r#"["N"]"# } else { r#"["N","M"]"# };
            format!(
                r#"{{"type":"node","id":"n{i}","labels":{labels},"properties":{{"v":{i}}}}}"#
            )
        })
        .chain((0..8).map(|i| {
            format!(
                r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"n{i}","to":"n{}","properties":{{}}}}"#,
                (i + 1) % 8
            )
        }))
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let queries = [
        // two DIFFERENT labels on the shared node — only the `N`+`M` nodes qualify
        "MATCH (a:N)-[:R]->(b:N), (b:M)-[:R]->(c) RETURN element_id(a), element_id(c)",
        // an inline node WHERE on each side
        "MATCH (a:N)-[:R]->(b WHERE b.v > 2), (b WHERE b.v < 6)-[:R]->(c) \
         RETURN element_id(a), element_id(c)",
        // label on one side, inline WHERE on the other
        "MATCH (a:N)-[:R]->(b:M), (b WHERE b.v > 2)-[:R]->(c) RETURN element_id(a), element_id(c)",
    ];

    for q in queries {
        let fused = rows(&mut g, q);
        let plain = super::eval::without_fusion(|| rows(&mut g, q));

        assert!(!fused.is_empty(), "`{q}` produced no rows — inert test");
        assert_eq!(fused, plain, "fusion changed the result of `{q}`");
    }
}

/// `DISTINCT` combined with `ORDER BY` must agree with the scalar driver.
///
/// The columnar path dedups on the projected row and then sorts, while the
/// scalar one dedups in scan order — so they only agree if the sort is total
/// enough to pin the result. Every case here is compared against the scalar
/// driver row-for-row, including the ties.
#[test]
fn distinct_with_order_by_agrees_with_the_scalar_driver() {
    let lines: Vec<String> = (0..40)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"n{i}","labels":["N"],"properties":{{"d":"d{}","v":{},"s":"s{}"}}}}"#,
                i % 5,
                i % 7,
                i % 3
            )
        })
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let queries = [
        "MATCH (n:N) RETURN DISTINCT n.d ORDER BY n.d",
        "MATCH (n:N) RETURN DISTINCT n.d AS a ORDER BY a",
        "MATCH (n:N) RETURN DISTINCT n.d AS a ORDER BY a DESC",
        "MATCH (n:N) RETURN DISTINCT n.d, n.v ORDER BY n.d, n.v",
        "MATCH (n:N) RETURN DISTINCT n.d, n.v ORDER BY n.v DESC, n.d",
        "MATCH (n:N) RETURN DISTINCT n.v AS a ORDER BY a LIMIT 3",
        "MATCH (n:N) RETURN DISTINCT n.v AS a ORDER BY a SKIP 2 LIMIT 3",
        "MATCH (n:N) RETURN DISTINCT n.s AS a ORDER BY a",
        // a computed alias, so the dedup key is not a raw column
        "MATCH (n:N) RETURN DISTINCT n.v * 2 AS a ORDER BY a",
    ];

    for q in queries {
        let vec_on = super::eval::with_vec_override(true, || rows(&mut g, q));
        let scalar = super::eval::with_vec_override(false, || rows(&mut g, q));

        assert!(!vec_on.is_empty(), "`{q}` produced no rows — inert test");
        assert_eq!(
            vec_on, scalar,
            "vectorized DISTINCT+ORDER BY differs for `{q}`"
        );
    }
}

/// Columnar vs scalar for a projection that returns ELEMENTS rather than values.
///
/// `RETURN *` was measured 23x faster than the equivalent `RETURN v`, which is
/// the wrong way round: star took the scalar driver and the explicit spelling
/// took the columnar frame.
/// Run: `cargo test --release bench_element_projection -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_element_projection() {
    let lines: Vec<String> = (0..200_000)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"v{i}","labels":["V"],"properties":{{"n":{}}}}}"#,
                i % 1000
            )
        })
        .chain((0..200_000).map(|i| {
            format!(
                r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"v{i}","to":"v{}","properties":{{}}}}"#,
                (i + 1) % 200_000
            )
        }))
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let cases: &[(&str, &str)] = &[
        ("RETURN v (element)", "MATCH (v:V) RETURN v"),
        ("RETURN * (same query)", "MATCH (v:V) RETURN *"),
        ("RETURN v.n (value)", "MATCH (v:V) RETURN v.n"),
        ("RETURN a, b", "MATCH (a:V)-[:R]->(b) RETURN a, b"),
        ("RETURN a.n, b.n", "MATCH (a:V)-[:R]->(b) RETURN a.n, b.n"),
    ];

    println!(
        "\n{:<24} {:>10} {:>10} {:>8}",
        "query", "columnar", "scalar", "ratio"
    );

    for (name, q) in cases {
        let mut best = [f64::MAX; 2];

        for (k, on) in [true, false].iter().enumerate() {
            let plan = super::parse(q).expect("parses");
            let params = super::eval::Params::new();

            for _ in 0..5 {
                let t = std::time::Instant::now();
                let n = super::eval::with_vec_override(*on, || {
                    plan.execute(&mut g, &params).expect("executes").nrows
                });
                assert!(n > 0);
                best[k] = best[k].min(t.elapsed().as_secs_f64() * 1e3);
            }
        }

        println!(
            "{name:<24} {:>9.2}ms {:>9.2}ms {:>7.2}x",
            best[0],
            best[1],
            best[0] / best[1]
        );
    }
}

/// Multi-clause `MATCH` must bind the same rows, in the same order, whichever
/// engine runs it.
///
/// Consecutive `MATCH`es arrive at the frame ALREADY merged into one clause with
/// several patterns, so these exercise the comma-fusion path via a spelling that
/// does not use commas — plus the shapes that must not merge (`OPTIONAL`) and
/// must not fuse (a cartesian product).
#[test]
fn multi_clause_match_agrees_with_the_scalar_driver() {
    let mut g = layered_dense(4, 4);
    let queries = [
        "MATCH (a:N)-[:R]->(b) MATCH (b)-[:R]->(c) RETURN element_id(a), element_id(c)",
        "MATCH (a:N)-[:R]->(b) MATCH (b)-[:R]->(c) MATCH (c)-[:R]->(d) \
         RETURN element_id(a), element_id(d)",
        // a WHERE on the FIRST clause, which flattening moves after the second
        "MATCH (a:N)-[:R]->(b) WHERE a.id <> 'n0' MATCH (b)-[:R]->(c) \
         RETURN element_id(a), element_id(c)",
        // a WHERE on each clause
        "MATCH (a:N)-[:R]->(b) WHERE a.id <> 'n0' MATCH (b)-[:R]->(c) WHERE c.id <> 'n15' \
         RETURN element_id(a), element_id(c)",
        // clauses that do NOT share a variable — a cartesian product
        "MATCH (a:N)-[:R]->(b) MATCH (c:N)-[:R]->(d) RETURN element_id(a), element_id(c)",
        // OPTIONAL must not flatten — it is a left join
        "MATCH (a:N)-[:R]->(b) OPTIONAL MATCH (b)-[:NONE]->(c) \
         RETURN element_id(a), element_id(c)",
        // mixed with a comma pattern in one of the clauses
        "MATCH (a:N)-[:R]->(b), (b)-[:R]->(c) MATCH (c)-[:R]->(d) \
         RETURN element_id(a), element_id(d)",
        "MATCH (a:N)-[:R]->(b) MATCH (b)-[:R]->(c) ORDER BY element_id(c) LIMIT 5 \
         RETURN element_id(a), element_id(c)",
    ];

    for q in queries {
        let vec_on = super::eval::with_vec_override(true, || rows(&mut g, q));
        let scalar = super::eval::with_vec_override(false, || rows(&mut g, q));

        assert!(!vec_on.is_empty(), "`{q}` produced no rows — inert test");
        assert_eq!(vec_on, scalar, "engines disagree on `{q}`");
    }
}

/// A carried value column only becomes a TYPED numeric column when every cell is
/// numeric — a mixed one must keep its cross-type behaviour.
///
/// `typed_val_col` exists so `WITH a.n AS m … WHERE b.n > m` drives the same
/// typed kernels a property column does (5.8x on arithmetic). Coercing a mixed
/// column with `num_of` would have turned a string into "not a number" and
/// quietly yielded no-match, where ordering a number against a string is
/// specified to RAISE. These pin both halves.
#[test]
fn a_carried_column_is_typed_only_when_uniformly_numeric() {
    let lines: Vec<String> = (0..6)
        .map(|i| {
            format!(r#"{{"type":"node","id":"n{i}","labels":["N"],"properties":{{"v":{i}}}}}"#)
        })
        .chain((0..6).map(|i| {
            format!(
                r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"n{i}","to":"n{}","properties":{{}}}}"#,
                (i + 1) % 6
            )
        }))
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    // Uniformly numeric: takes the typed path, and still answers correctly.
    let typed =
        "MATCH (a:N) WITH a, a.v AS m MATCH (a)-[:R]->(b) WHERE b.v > m RETURN count(*) AS c";
    assert_eq!(
        rows(&mut g, typed),
        super::eval::with_vec_override(false, || rows(&mut g, typed)),
        "typed carried column disagrees with the scalar driver"
    );

    // A carried STRING compared to a number keeps whatever cross-type behaviour
    // the general path has — the point is that the two engines still AGREE, which
    // is what coercing the column with `num_of` would have broken. (Here they
    // agree on no-match rather than a raise; that is the engine's existing rule,
    // not something this test asserts a preference about.)
    let mixed =
        "MATCH (a:N) WITH a, 'x' AS m MATCH (a)-[:R]->(b) WHERE b.v > m RETURN count(*) AS c";
    assert_eq!(
        super::eval::with_vec_override(true, || exec_err(&mut g, mixed)),
        super::eval::with_vec_override(false, || exec_err(&mut g, mixed)),
        "engines disagree on whether a number-vs-carried-string comparison raises"
    );
    assert_eq!(
        super::eval::with_vec_override(true, || rows(&mut g, mixed)),
        super::eval::with_vec_override(false, || rows(&mut g, mixed)),
        "engines disagree on a number-vs-carried-string comparison"
    );

    // A carried NULL is numeric-shaped (invalid), not a reason to fall back.
    let nulled = "MATCH (a:N) WITH a, CASE WHEN a.v > 2 THEN a.v ELSE null END AS m \
                  MATCH (a)-[:R]->(b) WHERE b.v > m RETURN count(*) AS c";
    assert_eq!(
        rows(&mut g, nulled),
        super::eval::with_vec_override(false, || rows(&mut g, nulled)),
        "a carried column with nulls disagrees with the scalar driver"
    );
}

/// A clause whose patterns share NO variable is a cross product, and the
/// columnar cross must match the scalar join row-for-row.
///
/// The first version of `cross_frames` crossed any groups that failed to CHAIN,
/// which silently included shapes that share a variable elsewhere (diverging, or
/// joining mid-path) — it returned the full product, 8 rows where the join gives
/// 2. The disjointness check is what makes it sound, so the shapes that must NOT
/// cross are tested here beside the ones that must.
#[test]
fn crossing_disconnected_patterns_agrees_with_the_scalar_driver() {
    let mut g = layered_dense(4, 4);
    let queries = [
        // genuinely disconnected — crosses
        "MATCH (a:N)-[:R]->(b), (c:N)-[:R]->(d) RETURN element_id(a), element_id(c)",
        "MATCH (a:N)-[:R]->(b), (c:N)-[:R]->(d) WHERE a.id < c.id RETURN element_id(a), element_id(c)",
        "MATCH (a:N)-[:R]->(b), (c:N)-[:R]->(d), (e:N)-[:R]->(f) RETURN count(*) AS n",
        // shares a variable but does not chain — must NOT cross
        "MATCH (b)-[:R]->(a:N), (b)-[:R]->(c) RETURN element_id(a), element_id(c)",
        "MATCH (a:N)-[:R]->(b), (c)-[:R]->(b) RETURN element_id(a), element_id(c)",
        // chains — fuses, never reaches the cross
        "MATCH (a:N)-[:R]->(b), (b)-[:R]->(c) RETURN element_id(a), element_id(c)",
        // disconnected AND ordered
        "MATCH (a:N)-[:R]->(b), (c:N)-[:R]->(d) RETURN element_id(a) AS x, element_id(c) AS y \
         ORDER BY x, y LIMIT 12",
    ];

    for q in queries {
        let vec_on = super::eval::with_vec_override(true, || rows(&mut g, q));
        let scalar = super::eval::with_vec_override(false, || rows(&mut g, q));

        assert!(!vec_on.is_empty(), "`{q}` produced no rows — inert test");
        assert_eq!(vec_on, scalar, "engines disagree on `{q}`");
    }
}

/// A capped scan may only bound its SEED when nothing after the seed can reject
/// a row.
///
/// `label_seed` collects at most `cap` ids so `MATCH (n:Person) … LIMIT 100`
/// need not materialize a 50,000-id bucket. That is only sound with no conjunct
/// and no residual: here the matching rows are at the END of the bucket, so a
/// seed bounded to the limit would find none of them and return zero rows.
#[test]
fn a_capped_scan_does_not_bound_its_seed_when_a_filter_follows() {
    // 200 people; only the LAST five carry age 99, so any bounded seed misses them.
    let lines: Vec<String> = (0..200)
        .map(|i| {
            let age = if i >= 195 { 99 } else { 30 };
            format!(
                r#"{{"type":"node","id":"p{i}","labels":["Person"],"properties":{{"age":{age},"nm":"n{i}"}}}}"#
            )
        })
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let q = "MATCH (n:Person) WHERE n.age = 99 RETURN n.nm LIMIT 3";
    assert_eq!(
        rows(&mut g, q).len(),
        3,
        "a bounded seed lost the matching rows"
    );

    // Same with an inline property rather than a clause WHERE.
    let inline = "MATCH (n:Person {age: 99}) RETURN n.nm LIMIT 3";
    assert_eq!(
        rows(&mut g, inline).len(),
        3,
        "a bounded seed lost the matching rows"
    );

    // And the shape the bound EXISTS for: no filter, so bounding is sound.
    let plain = "MATCH (n:Person) RETURN n.nm LIMIT 3";
    assert_eq!(rows(&mut g, plain).len(), 3);
}

/// A label DISJUNCTION over multi-label elements must not double-count.
///
/// An element is bucketed under every label it carries, so `[:X|Y]` reaches an
/// edge labelled `[X, Y]` from both buckets. Three separate paths unioned those
/// buckets and each returned it twice — `label_seed`, GQL's `etype_label_seed`,
/// and `try_count_edges` summing bucket LENGTHS. The vertex side had always
/// deduped for exactly this reason; edges only started needing it when they
/// stopped being single-label.
///
/// The whole composite is here — "walk the FROM edges labelled X or Y to any
/// vertex labelled W or Z" — because that is one operation over two element
/// kinds, and each kind had its own copy of the rule.
#[test]
fn a_label_disjunction_over_multi_label_elements_counts_each_once() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#,
            // `b` carries BOTH target labels, so the node side is exercised too.
            r#"{"type":"node","id":"b","labels":["W","Z"],"properties":{}}"#,
            r#"{"type":"node","id":"c","labels":["Z"],"properties":{}}"#,
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["X","Y"],"properties":{}}"#,
            r#"{"type":"edge","id":"e1","from":"a","to":"c","labels":["Y"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let n = |g: &mut Graph, q: &str| match rows(g, q).first().and_then(|r| r.first()) {
        Some(Value::Num(x)) => *x,
        other => panic!("expected a count, got {other:?}"),
    };

    // Each label alone.
    assert_eq!(n(&mut g, "MATCH ()-[:X]->() RETURN count(*) AS c"), 1.0);
    assert_eq!(n(&mut g, "MATCH ()-[:Y]->() RETURN count(*) AS c"), 2.0);

    // The disjunction: two edges, not three.
    assert_eq!(n(&mut g, "MATCH ()-[:X|Y]->() RETURN count(*) AS c"), 2.0);

    // …and the same through the NAMED-edge form, which takes a different path.
    assert_eq!(
        rows(&mut g, "MATCH ()-[e:X|Y]->() RETURN element_id(e) AS i").len(),
        2
    );

    // Vertex disjunction over a vertex carrying both.
    assert_eq!(n(&mut g, "MATCH (v:W|Z) RETURN count(*) AS c"), 2.0);

    // The composite: FROM-edges labelled X or Y, to vertices labelled W or Z.
    assert_eq!(
        n(&mut g, "MATCH ()-[:X|Y]->(:W|Z) RETURN count(*) AS c"),
        2.0
    );

    // A residual label test over an edge must see PAST its first label.
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[e]->() WHERE e IS LABELED Y RETURN count(*) AS c"
        ),
        2.0
    );
}

/// The `count(*)` shortcuts must see PAST an edge's first label.
///
/// They tested `a.etype` — an edge's FIRST label — which was the whole story
/// until edges became multi-label. `MATCH (a)-[:Y]->(b)-[:Y]->(c)` over an edge
/// labelled [X, Y] then answered 0 where the answer is 1: a wrong count, from a
/// path taken only when the shape qualifies for a shortcut, so the general
/// matcher answered correctly and nothing disagreed.
#[test]
fn the_count_shortcuts_see_every_edge_label() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{}}"#,
            r#"{"type":"node","id":"c","labels":["N"],"properties":{}}"#,
            // FIRST label X, second Y — a `[:Y]` shortcut must still find it.
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["X","Y"],"properties":{}}"#,
            r#"{"type":"edge","id":"e1","from":"b","to":"c","labels":["Y"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let n = |g: &mut Graph, q: &str| match rows(g, q).first().and_then(|r| r.first()) {
        Some(Value::Num(x)) => *x,
        other => panic!("expected a count, got {other:?}"),
    };

    // try_count_edges
    assert_eq!(
        n(&mut g, "MATCH (a:N)-[:Y]->(b:N) RETURN count(*) AS c"),
        2.0
    );
    assert_eq!(
        n(&mut g, "MATCH (a:N)-[:X]->(b:N) RETURN count(*) AS c"),
        1.0
    );

    // try_count_two_hop — the one that answered 0
    assert_eq!(
        n(
            &mut g,
            "MATCH (a:N)-[:Y]->(b:N)-[:Y]->(c:N) RETURN count(*) AS c"
        ),
        1.0
    );

    // …and each shortcut agrees with the general matcher, which never used the
    // first-label shorthand.
    for q in [
        "MATCH (a:N)-[:Y]->(b:N) RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b:N)-[:Y]->(c:N) RETURN count(*) AS c",
    ] {
        assert_eq!(
            rows(&mut g, q),
            super::eval::with_vec_override(false, || rows(&mut g, q)),
            "shortcut disagrees with the general matcher on `{q}`"
        );
    }
}

/// EVERY `count(*)` shortcut must agree with the general matcher on a graph with
/// a multi-label edge.
///
/// There are eight of them, each triggered by a different query shape, and each
/// carried its own edge-type test. Checking them one at a time is how the
/// two-hop one stayed wrong; this drives every shape and compares against the
/// scalar driver, which never used the first-label shorthand.
#[test]
fn every_count_shortcut_agrees_on_multi_label_edges() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"k":1}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"k":2}}"#,
            r#"{"type":"node","id":"c","labels":["M"],"properties":{"k":3}}"#,
            r#"{"type":"node","id":"d","labels":["N"],"properties":{"k":4}}"#,
            // Each of these has Y as a NON-first label, so any shortcut testing
            // only `e_type` sees a different graph than the matcher does.
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["X","Y"],"properties":{"w":1}}"#,
            r#"{"type":"edge","id":"e1","from":"b","to":"c","labels":["X","Y"],"properties":{"w":2}}"#,
            r#"{"type":"edge","id":"e2","from":"b","to":"d","labels":["Y"],"properties":{"w":3}}"#,
            r#"{"type":"edge","id":"e3","from":"c","to":"d","labels":["Z","Y"],"properties":{"w":4}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // One query per shortcut shape, all over the non-first label `Y`.
    let queries = [
        "MATCH (n:N) RETURN count(*) AS c",
        "MATCH ()-[:Y]->() RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b) RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b)-[:Y]->(c) RETURN count(*) AS c",
        "MATCH (a)-[:Y]->(b), (b)-[:Y]->(c) RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->{1,2}(b) RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b) RETURN count(DISTINCT b) AS c",
        "MATCH (a:N) WHERE EXISTS { (a)-[:Y]->() } RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->{1,3}(b) RETURN count(DISTINCT b) AS c",
        "MATCH (a)-[:X|Y]->(b) RETURN count(*) AS c",
        "MATCH (a:N)-[e:Y]->(b) WHERE e.w > 1 RETURN count(*) AS c",
    ];

    for q in queries {
        let fast = rows(&mut g, q);
        let general = super::eval::with_vec_override(false, || rows(&mut g, q));

        assert_eq!(
            fast, general,
            "a count shortcut disagrees with the matcher on `{q}`"
        );
    }
}

/// The unboxed numeric sort agrees with the boxed one.
///
/// `ORDER BY` cost 9-18x the same query without one, and the comparator was why:
/// `compare_sort` over two boxed `Val`s does two `is_nullish` matches, two
/// `type_rank` calls and a `Vec` index, where Gremlin compares `f64`. Extracting
/// a flat `f64` key is 1.9x on 150k rows.
///
/// Everything that could differ is a tie, a null or a NaN, so all three are in
/// the fixture: ties must resolve to SCAN ORDER (the index tiebreak, matching
/// the stable boxed sort), a null must make the unboxed path DECLINE rather than
/// guess a placement, and a NaN must sort LAST — it is a value, the greatest
/// one, not a null.
///
/// The NaN has to be COMPUTED into the projection. A stored one is no longer
/// reachable: every write coerces a non-finite to null.
#[test]
fn the_unboxed_numeric_sort_agrees_with_the_boxed_one() {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..40 {
        // Heavy duplication, so most comparisons are TIES.
        let mut props = format!(r#""k":{}"#, i % 5);

        // Some rows carry no `m` at all — a null key, which the unboxed path
        // must DECLINE rather than guess a placement for.
        if i % 3 != 0 {
            props.push_str(&format!(r#","m":{}"#, i % 7));
        }

        lines.push(format!(
            r#"{{"type":"node","id":"n{i}","labels":["P"],"properties":{{{props}}}}}"#
        ));
    }

    for i in 0..39 {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"n{i}","to":"n{}","labels":["R"],"properties":{{}}}}"#,
            (i * 7 + 1) % 40
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    // A third column that is numeric on most rows and NULL on a few — `sqrt(-1)`
    // computes a NaN and the write coerces it, which is the only way a null
    // lands in the middle of an otherwise numeric column.
    let _ = q(
        &mut g,
        "MATCH (n:P) WHERE n.k = 2 SET n.z = sqrt(-1) RETURN count(*) AS c",
    );
    let _ = q(
        &mut g,
        "MATCH (n:P) WHERE n.k <> 2 SET n.z = n.k RETURN count(*) AS c",
    );

    for query in [
        // Ties everywhere, both directions, alias and property spellings.
        "MATCH (n:P) RETURN n.k AS x ORDER BY x",
        "MATCH (n:P) RETURN n.k AS x ORDER BY x DESC",
        "MATCH (n:P) RETURN n.k AS x ORDER BY n.k",
        "MATCH (n:P)-[:R]->(b) RETURN b.k AS x ORDER BY x",
        "MATCH (n:P)-[:R]->(b) RETURN b.k AS x ORDER BY x DESC",
        // Paging over the sorted result.
        "MATCH (n:P) RETURN n.k AS x ORDER BY x LIMIT 5",
        "MATCH (n:P) RETURN n.k AS x ORDER BY x DESC LIMIT 5",
        "MATCH (n:P) RETURN n.k AS x ORDER BY x OFFSET 5",
        "MATCH (n:P) RETURN n.k AS x ORDER BY x OFFSET 3 LIMIT 4",
        "MATCH (n:P) RETURN n.k AS x ORDER BY x LIMIT 0",
        // A NULL key: the unboxed path declines, the boxed one places nulls.
        "MATCH (n:P) RETURN n.m AS x ORDER BY x",
        "MATCH (n:P) RETURN n.m AS x ORDER BY x DESC",
        "MATCH (n:P) RETURN n.m AS x ORDER BY x LIMIT 5",
        // Nulls interleaved with numbers in one column.
        "MATCH (n:P) RETURN n.z AS x ORDER BY x",
        "MATCH (n:P) RETURN n.z AS x ORDER BY x DESC",
        "MATCH (n:P) RETURN n.z AS x ORDER BY x LIMIT 6",
        // A COMPUTED NaN, which is a value and not a null — the unboxed path
        // declines it and the boxed comparator sorts it last.
        "MATCH (n:P) RETURN sqrt(n.k - 2) AS x ORDER BY x",
        "MATCH (n:P) RETURN sqrt(n.k - 2) AS x ORDER BY x DESC",
        "MATCH (n:P) RETURN sqrt(n.k - 2) AS x ORDER BY x LIMIT 6",
        "MATCH (n:P) RETURN sqrt(n.k - 2) AS x, n.k AS y ORDER BY x",
        // Carrying another column along, so the PERMUTATION is observable and
        // not just the sorted key.
        "MATCH (n:P) RETURN n.k AS x, n.m AS y ORDER BY x",
        "MATCH (n:P) RETURN n.k AS x, n.m AS y ORDER BY x DESC LIMIT 7",
        // Two keys — the unboxed path takes only a single key.
        "MATCH (n:P) RETURN n.k AS x, n.m AS y ORDER BY x, y",
        // A non-numeric key stays boxed.
        "MATCH (n:P) RETURN n.k AS x ORDER BY x, n.k",
    ] {
        // Compared as DEBUG STRINGS, not values: `NaN != NaN` under `PartialEq`,
        // so `assert_eq!` on rows holding one fails while printing two identical
        // lists. The projections above compute NaNs on purpose.
        let fast = format!("{:?}", rows(&mut g, query));
        let boxed = format!(
            "{:?}",
            super::eval::with_vec_override(false, || rows(&mut g, query))
        );

        assert_eq!(
            fast, boxed,
            "the unboxed numeric sort disagrees with the boxed one on `{query}`"
        );
    }
}

/// A computed NaN sorts LAST, and it is the sort that has to say so.
///
/// The fast-vs-boxed comparison next door is an internal-consistency oracle: it
/// passes whenever the two paths agree, including when both are wrong. This
/// pins the answer itself.
///
/// It was wrong. `cmp_total` compared a NaN Equal to every number while the
/// numbers stayed ordered among themselves — not a total order — so
/// `ORDER BY x DESC` returned `NaN, 2, NaN, 3`: every adjacent pair compared
/// Equal, and the stable sort left the rows where it found them. The TS engine
/// scrambled the same input differently (`null, 3, 2, null`), because its
/// comparator fell through to `x < y` / `x > y`, both false, giving 0. We own
/// the comparator and not the sort ALGORITHM, so only a TOTAL order makes the
/// two agree. It was also the one input that could ABORT the process: Rust's
/// sort detects the inconsistency and panics.
///
/// NaN is the GREATEST value, not an absolute-last like null, so DESC puts it
/// first — matching `Double.compareTo` and SQL float order.
#[test]
fn a_computed_nan_sorts_last() {
    let lines: Vec<String> = [-1, 4, -9, 9]
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(r#"{{"type":"node","id":"n{i}","labels":["P"],"properties":{{"m":{m}}}}}"#)
        })
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    for (query, want) in [
        (
            "MATCH (n:P) RETURN sqrt(n.m) AS x ORDER BY x",
            "[[Num(2.0)], [Num(3.0)], [Num(NaN)], [Num(NaN)]]",
        ),
        (
            "MATCH (n:P) RETURN sqrt(n.m) AS x ORDER BY x DESC",
            "[[Num(NaN)], [Num(NaN)], [Num(3.0)], [Num(2.0)]]",
        ),
        // A LIMIT takes the top-k path, which must read the same order.
        (
            "MATCH (n:P) RETURN sqrt(n.m) AS x ORDER BY x LIMIT 2",
            "[[Num(2.0)], [Num(3.0)]]",
        ),
        (
            "MATCH (n:P) RETURN sqrt(n.m) AS x ORDER BY x DESC LIMIT 2",
            "[[Num(NaN)], [Num(NaN)]]",
        ),
        // Both halves of the NaN split in one query. `nullif` compares with
        // PREDICATE equality, where NaN = NaN is UNKNOWN, so the NaN rows are
        // NOT nulled while the real numbers are. Then the SORT uses the total
        // order, where NaN == NaN and is greatest — so under DESC the surviving
        // NaNs come first and the nulls stay last, null placement being absolute
        // rather than direction-relative.
        (
            "MATCH (n:P) RETURN nullif(sqrt(n.m), sqrt(n.m)) AS x ORDER BY x DESC",
            "[[Num(NaN)], [Num(NaN)], [Null], [Null]]",
        ),
    ] {
        assert_eq!(
            format!("{:?}", rows(&mut g, query)),
            want,
            "`{query}` did not sort a computed NaN last"
        );

        // The scalar driver has to agree — it is a different comparator call site.
        assert_eq!(
            format!(
                "{:?}",
                super::eval::with_vec_override(false, || rows(&mut g, query))
            ),
            want,
            "the scalar driver disagreed on `{query}`"
        );
    }
}

/// A merged multi-pattern first clause agrees with the scalar driver.
///
/// `vectorized_linear` used to refuse any first `MATCH` with more than one
/// pattern, which sent the WHOLE pipeline to the scalar driver. Since ac0d6c2
/// merges adjacent `MATCH`es at plan time — and an inline `CALL` desugars to
/// exactly that, `[Match, Match, With, Return]` — the refusal fired on shapes
/// people write, and cost 4.1x on the CALL form.
///
/// Fusing first makes those shapes reach the pipeline, which means a class of
/// queries changed which engine answers them. This asserts the answers did not.
#[test]
fn a_fused_multi_pattern_pipeline_agrees_with_the_scalar_driver() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"k":1,"s":"x"}}"#,
            r#"{"type":"node","id":"b","labels":["P","W"],"properties":{"k":2,"s":"y"}}"#,
            r#"{"type":"node","id":"c","labels":["P"],"properties":{"k":2,"s":"x"}}"#,
            r#"{"type":"node","id":"d","labels":["Q"],"properties":{"k":3}}"#,
            r#"{"type":"edge","id":"r0","from":"a","to":"b","labels":["R"],"properties":{"w":1}}"#,
            r#"{"type":"edge","id":"r1","from":"a","to":"c","labels":["R"],"properties":{"w":2}}"#,
            r#"{"type":"edge","id":"r2","from":"b","to":"d","labels":["R"],"properties":{"w":3}}"#,
            r#"{"type":"edge","id":"r3","from":"c","to":"b","labels":["S"],"properties":{"w":4}}"#,
            r#"{"type":"edge","id":"r4","from":"b","to":"b","labels":["R"],"properties":{"w":5}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    for q in [
        // The shape that regressed, and the two spellings that merge into it.
        "MATCH (p:P) CALL (p) { MATCH (p)-[:R]->(f) RETURN f.k AS fk } RETURN count(*) AS c",
        "MATCH (p:P) MATCH (p)-[:R]->(f) WITH f.k AS fk RETURN count(*) AS c",
        "MATCH (p:P), (p)-[:R]->(f) WITH f.k AS fk RETURN count(*) AS c",
        // Carrying the projection forward rather than counting it.
        "MATCH (p:P) MATCH (p)-[:R]->(f) WITH f.k AS fk RETURN fk ORDER BY fk",
        "MATCH (p:P) MATCH (p)-[:R]->(f) WITH DISTINCT f.k AS fk RETURN fk ORDER BY fk",
        "MATCH (p:P) MATCH (p)-[:R]->(f) WITH f.k AS fk WHERE fk > 1 RETURN count(*) AS c",
        "MATCH (p:P) MATCH (p)-[:R]->(f) WITH f AS f2 MATCH (f2)-[:R]->(g) RETURN count(*) AS c",
        // A three-way merge, and one where the second pattern extends the first.
        "MATCH (p:P) MATCH (p)-[:R]->(f) MATCH (f)-[:R]->(g) WITH g.k AS gk RETURN count(*) AS c",
        "MATCH (p:P)-[:R]->(f) MATCH (f)-[:S]->(h) WITH h.k AS hk RETURN count(*) AS c",
        // Aggregates and grouping over the merged shape.
        "MATCH (p:P) MATCH (p)-[:R]->(f) WITH f.k AS fk RETURN sum(fk) AS s",
        "MATCH (p:P) MATCH (p)-[:R]->(f) RETURN f.k AS fk, count(*) AS c ORDER BY fk",
        // Patterns that share NO variable are a cross product, which this
        // pipeline must still decline rather than get wrong.
        "MATCH (p:P), (q:Q) WITH p.k AS pk, q.k AS qk RETURN count(*) AS c",
        "MATCH (p:P) MATCH (q:Q) WITH p.k AS pk RETURN count(*) AS c",
        // OPTIONAL must not merge at all.
        "MATCH (p:P) OPTIONAL MATCH (p)-[:S]->(f) WITH f.k AS fk RETURN count(*) AS c",
    ] {
        let fused = rows(&mut g, q);
        let scalar = super::eval::with_vec_override(false, || rows(&mut g, q));

        assert_eq!(
            fused, scalar,
            "the fused pipeline disagrees with the scalar driver on `{q}`"
        );
    }
}

/// The one-slot WALK agrees with the joined frame on every shape it takes.
///
/// `streamed_frame` replaces the join with a frontier walk when the only slot
/// anything reads is the one the walk lands on. The rows must be identical to the
/// frame's projection onto that slot — `expand` emits one endpoint per traversed
/// edge, so multiplicity is preserved — and every terminal then runs unchanged.
/// This asserts that against `with_vec_override(false, …)`, which takes the
/// matcher instead.
///
/// The fixture has a self-loop (an undirected hop counts it once in GQL), a
/// non-first edge label (a walk reading only `e_type` sees a different graph than
/// the matcher does), repeated property values (so DISTINCT and GROUP BY have
/// something to collapse) and fan-in (so multiplicity is observable at all).
#[test]
fn the_walked_frame_agrees_with_the_joined_one() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"k":1,"s":"p"}}"#,
            r#"{"type":"node","id":"b","labels":["N","W"],"properties":{"k":7,"s":"q"}}"#,
            r#"{"type":"node","id":"c","labels":["M"],"properties":{"k":3,"s":"p"}}"#,
            r#"{"type":"node","id":"d","labels":["N"],"properties":{"k":7,"s":"q"}}"#,
            r#"{"type":"node","id":"e","labels":["N"],"properties":{}}"#,
            r#"{"type":"edge","id":"r0","from":"a","to":"b","labels":["X","Y"],"properties":{"w":1}}"#,
            r#"{"type":"edge","id":"r1","from":"a","to":"c","labels":["Y"],"properties":{"w":2}}"#,
            r#"{"type":"edge","id":"r2","from":"b","to":"d","labels":["Y"],"properties":{"w":3}}"#,
            r#"{"type":"edge","id":"r3","from":"c","to":"d","labels":["Z","Y"],"properties":{"w":4}}"#,
            r#"{"type":"edge","id":"r4","from":"e","to":"d","labels":["Y"],"properties":{"w":5}}"#,
            r#"{"type":"edge","id":"r5","from":"b","to":"b","labels":["Y"],"properties":{"w":6}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    for q in [
        // Plain rows off the landing slot — the shape the walk exists for.
        "MATCH (a:N)-[:Y]->(b) RETURN b.k AS x",
        "MATCH (a:N)-[:Y]->(b) WHERE a.k = 1 RETURN b.k AS x",
        "MATCH (a)-[:Y]->(b) WHERE a.k > 1 RETURN b.s AS x",
        "MATCH (a:N)-[:Y]->(b)-[:Y]->(c) RETURN c.k AS x",
        // A property that is ABSENT on some rows, so the null path is exercised.
        "MATCH (a)-[:Y]->(b) RETURN b.k AS x",
        // Aggregates: one fold, and one per group.
        "MATCH (a:N)-[:Y]->(b) RETURN sum(b.k) AS s",
        "MATCH (a:N)-[:Y]->(b) RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b) RETURN avg(b.k) AS s",
        "MATCH (a:N)-[:Y]->(b) RETURN min(b.k) AS lo, max(b.k) AS hi",
        "MATCH (a)-[:Y]->(b) RETURN b.k AS k, count(*) AS c",
        "MATCH (a)-[:Y]->(b) RETURN b.s AS s, sum(b.k) AS t",
        // The boundaries: DISTINCT, ORDER BY, paging.
        "MATCH (a)-[:Y]->(b) RETURN DISTINCT b.k AS x",
        "MATCH (a)-[:Y]->(b) RETURN b.k AS x ORDER BY x",
        "MATCH (a)-[:Y]->(b) RETURN b.k AS x ORDER BY x DESC LIMIT 2",
        "MATCH (a)-[:Y]->(b) RETURN b.k AS x LIMIT 3",
        "MATCH (a)-[:Y]->(b) RETURN b.k AS x OFFSET 2",
        "MATCH (a)-[:Y]->(b) RETURN count(DISTINCT b.k) AS c",
        // Undirected, over the self-loop; GQL counts it once.
        "MATCH (a:N)-[:Y]-(b) RETURN b.k AS x",
        "MATCH (a)-[:Y]-(b) RETURN count(*) AS c",
        // Reverse, untyped, disjoint, and a type that resolves to nothing.
        "MATCH (a:N)<-[:Y]-(b) RETURN b.k AS x",
        "MATCH (a:N)-[]->(b) RETURN b.k AS x",
        "MATCH (a:N)-[:X|Y]->(b) RETURN b.k AS x",
        "MATCH (a)-[:NONEXISTENT]->(b) RETURN b.k AS x",
        "MATCH (a)-[r:NONEXISTENT]->(b) RETURN count(*) AS c",
        // Reads a slot the walk does not carry — must DECLINE, and be right.
        "MATCH (a:N)-[:Y]->(b) RETURN a.k AS x",
        "MATCH (a:N)-[:Y]->(b) RETURN a.k AS x, b.k AS y",
        "MATCH (a:N)-[r:Y]->(b) RETURN r.w AS w",
        "MATCH (a:N)-[:Y]->(b) RETURN b.k AS x ORDER BY a.k",
        "MATCH (a:N)-[:Y]->(b) RETURN a.k AS k, count(*) AS c",
        "MATCH (a:N)-[:Y]->(b) RETURN *",
        // Constrained segments — a filter needs a row, so these decline too.
        "MATCH (a:N)-[:Y]->(b:W) RETURN b.k AS x",
        "MATCH (a:N)-[:Y]->(b {k: 7}) RETURN b.k AS x",
        "MATCH (a:N)-[:Y {w: 3}]->(b) RETURN b.k AS x",
        "MATCH (a)-[:Y]->(b) WHERE b.k = 7 RETURN b.k AS x",
        // Multi-segment with a LIMIT stays depth-first — the walk would hold every
        // k-path to return a handful.
        "MATCH (a)-[:Y]->(b)-[:Y]->(c) RETURN c.k AS x LIMIT 2",
        // Var-length and bound path variables are different walks entirely.
        "MATCH (a:N)-[:Y]->{1,2}(b) RETURN b.k AS x",
        "MATCH p = (a:N)-[:Y]->{1,2}(b) RETURN b.k AS x",
    ] {
        let walked = rows(&mut g, q);
        let joined = super::eval::with_vec_override(false, || rows(&mut g, q));

        assert_eq!(
            walked, joined,
            "the walked frame disagrees with the matcher on `{q}`"
        );
    }
}

/// The STREAM route for `count(*)` agrees with the matcher on every shape it takes.
///
/// `try_count_streamed` answers a pure count over bare hops by walking and
/// counting in place — `seek::walk_count`, the same function Gremlin's
/// `.count()` uses. Nothing is materialized, which is the point and also the
/// risk: a predicate that never gets a row to reject is a predicate that does
/// not run. The seed is filtered by `scan_node`, and `scan_node` filters the
/// START and nothing else.
///
/// That is not a hypothetical. `(a)-[:R]->(b) WHERE b.n = 7` has a bare `b` — no
/// label, no inline prop — and a WHERE about it, and the first version counted
/// every row the predicate excludes. Three suite tests caught it; the shapes
/// below are the ones that should have.
#[test]
fn the_streamed_count_agrees_with_the_matcher() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"k":1}}"#,
            r#"{"type":"node","id":"b","labels":["N","W"],"properties":{"k":7}}"#,
            r#"{"type":"node","id":"c","labels":["M"],"properties":{"k":3}}"#,
            r#"{"type":"node","id":"d","labels":["N"],"properties":{"k":7}}"#,
            // Non-first label `Y`, so a shortcut reading only `e_type` sees a
            // different graph than the matcher does.
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["X","Y"],"properties":{"w":1}}"#,
            r#"{"type":"edge","id":"e1","from":"b","to":"c","labels":["X","Y"],"properties":{"w":2}}"#,
            r#"{"type":"edge","id":"e2","from":"b","to":"d","labels":["Y"],"properties":{"w":3}}"#,
            r#"{"type":"edge","id":"e3","from":"c","to":"d","labels":["Z","Y"],"properties":{"w":4}}"#,
            // A SELF-LOOP, because an undirected hop counts one differently in
            // each language and the walk must keep GQL's answer.
            r#"{"type":"edge","id":"e4","from":"b","to":"b","labels":["Y"],"properties":{"w":5}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    for q in [
        // The shape this exists for: a filtered start, then bare hops.
        "MATCH (a:N)-[:Y]->(b) WHERE a.k = 1 RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b)-[:Y]->(c) WHERE a.k = 1 RETURN count(*) AS c",
        "MATCH (a)-[:Y]->(b) WHERE a.k > 1 RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b) RETURN count(*) AS c",
        // Untyped, disjoint, and a type that resolves to nothing.
        "MATCH (a:N)-[]->(b) RETURN count(*) AS c",
        "MATCH (a:N)-[:X|Y]->(b) RETURN count(*) AS c",
        "MATCH (a:N)-[:NOPE]->(b) RETURN count(*) AS c",
        // Undirected, over the self-loop.
        "MATCH (a:N)-[:Y]-(b) RETURN count(*) AS c",
        "MATCH (a)-[:Y]-(b) RETURN count(*) AS c",
        "MATCH (a)-[:Y]-(b)-[:Y]-(c) RETURN count(*) AS c",
        // Reverse direction.
        "MATCH (a:N)<-[:Y]-(b) RETURN count(*) AS c",
        // A WHERE about a slot the walk never binds a row for. Every one of these
        // must DECLINE — the walk has no row to test — and still be right.
        "MATCH (a)-[:Y]->(b) WHERE b.k = 7 RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b) WHERE a.k = 1 AND b.k = 7 RETURN count(*) AS c",
        "MATCH (a)-[e:Y]->(b) WHERE e.w > 1 RETURN count(*) AS c",
        "MATCH (a)-[:Y]->(b)-[:Y]->(c) WHERE c.k = 7 RETURN count(*) AS c",
        "MATCH (a)-[:Y]->(b) WHERE a.k = 1 OR b.k = 7 RETURN count(*) AS c",
        // Constrained SEGMENTS, which also decline: a filter needs a row.
        "MATCH (a:N)-[:Y]->(b:W) RETURN count(*) AS c",
        "MATCH (a:N)-[:Y]->(b {k: 7}) RETURN count(*) AS c",
        "MATCH (a:N)-[:Y {w: 3}]->(b) RETURN count(*) AS c",
        // Not a pure count, so not a reducing terminal.
        "MATCH (a:N)-[:Y]->(b) RETURN count(DISTINCT b) AS c",
        "MATCH (a:N)-[:Y]->(b) RETURN b.k AS k, count(*) AS c",
        "MATCH (a:N)-[:Y]->(b) RETURN count(*) AS c ORDER BY c",
    ] {
        let fast = rows(&mut g, q);
        let general = super::eval::with_vec_override(false, || rows(&mut g, q));

        assert_eq!(
            fast, general,
            "the streamed count disagrees with the matcher on `{q}`"
        );
    }
}

/// The seeded start of a streamed count is filtered; the far end is not reached.
///
/// Stated as numbers rather than as agreement, so the test says what the shapes
/// above only imply: a WHERE about the far end changes the answer, which is why
/// the walk must decline it rather than ignore it.
#[test]
fn a_streamed_count_declines_a_predicate_it_cannot_apply() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"n":1}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"n":7}}"#,
            r#"{"type":"node","id":"c","labels":["N"],"properties":{"n":9}}"#,
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e1","from":"a","to":"c","labels":["R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // Both edges leave `a`, so a start filter keeps both …
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[:R]->(b) WHERE a.n = 1 RETURN count(*) AS c"
        ),
        vec![vec![Value::Num(2.0)]]
    );

    // … and a far-end filter keeps ONE. A walk that dropped the predicate would
    // answer 2 here, and nothing about the query's shape would look wrong.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[:R]->(b) WHERE b.n = 7 RETURN count(*) AS c"
        ),
        vec![vec![Value::Num(1.0)]]
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a)-[:R]->(b) WHERE a.n = 1 AND b.n = 7 RETURN count(*) AS c"
        ),
        vec![vec![Value::Num(1.0)]]
    );
}

/// `labels()` takes an ELEMENT — a node or an edge — and returns its label set.
///
/// ISO GQL does not define `labels()` at all: the standard interrogates labels
/// with the `IS LABELED` predicate, and its only element function is
/// `element_id`. `labels` is a Cypher inheritance vendors added, and the two
/// that ship it define it over an element, not just a node — Spanner's
/// `LABELS(GRAPH_ELEMENT) -> ARRAY<STRING>` and Fabric's `labels(node_or_edge)`
/// "the labels of a node or edge as a list of strings". Both return a length-1
/// list for an edge because neither has multi-label edges; this engine does, so
/// it returns the whole set. Returning NULL, as this did, made it the only
/// implementation that does.
#[test]
fn labels_of_an_edge_is_its_type_set() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["W","V"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["S","R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e1","from":"b","to":"a","labels":["T"],"properties":{}}"#,
    ]);

    // Sorted, like the node arm — which type is stored first is not observable.
    assert_eq!(
        rows(&mut g, "MATCH ()-[e]->() RETURN labels(e) AS x ORDER BY x"),
        vec![
            vec![Value::List(vec![s("R"), s("S")])],
            vec![Value::List(vec![s("T")])],
        ]
    );

    // A single-type edge still gets a one-element LIST, not a bare string.
    assert_eq!(
        rows(&mut g, "MATCH ()-[e:T]->() RETURN labels(e) AS x"),
        vec![vec![Value::List(vec![s("T")])]]
    );

    // The node arm is unchanged.
    assert_eq!(
        rows(&mut g, "MATCH (n) RETURN labels(n) AS x ORDER BY x"),
        vec![
            vec![Value::List(vec![s("V")])],
            vec![Value::List(vec![s("V"), s("W")])],
        ]
    );

    // `type` stays SINGULAR — it is openCypher's `type(relationship) -> String`,
    // which cannot express a set, so it reports the first type. `labels(e)` is
    // the accessor for all of them.
    assert_eq!(
        rows(&mut g, "MATCH ()-[e:R]->() RETURN type(e) AS x"),
        vec![vec![s("S")]]
    );

    // Neither function accepts a non-element.
    assert_eq!(
        rows(&mut g, "MATCH (n) RETURN labels(n.missing) AS x LIMIT 1"),
        vec![vec![Value::Null]]
    );
    assert_eq!(
        rows(&mut g, "MATCH (n) RETURN type(n) AS x LIMIT 1"),
        vec![vec![Value::Null]]
    );
}

/// `labels(e)` agrees with the pattern matcher: every type it reports selects
/// that edge, and `size()` of it counts them.
#[test]
fn labels_of_an_edge_agrees_with_the_pattern_matcher() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R","S"],"properties":{}}"#,
    ]);

    for t in ["R", "S"] {
        assert_eq!(
            rows(
                &mut g,
                &format!("MATCH ()-[e:{t}]->() RETURN labels(e) AS x")
            ),
            vec![vec![Value::List(vec![s("R"), s("S")])]],
            "`[:{t}]` selected the edge but `labels` disagreed"
        );
    }

    assert_eq!(
        rows(&mut g, "MATCH ()-[e]->() RETURN size(labels(e)) AS x"),
        vec![vec![Value::Num(2.0)]]
    );
}

/// The GQL edge-type count reads the same buckets Gremlin's `.count()` now does,
/// so it must survive the same mutations.
///
/// Promoting a secondary type to first (by removing the first) re-pushed the
/// edge into a bucket it was already in. Walking the bucket hid it; taking its
/// LENGTH — which both engines' edge-type count shortcuts do — counted the edge
/// twice. Found by lowering the Gremlin count onto the shared primitive and
/// checking it against enumeration.
#[test]
fn the_edge_type_count_survives_label_mutation() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R","S"],"properties":{}}"#,
        r#"{"type":"edge","id":"e1","from":"b","to":"a","labels":["S"],"properties":{}}"#,
    ]);

    // The count shortcut (unlabeled endpoints) and the enumeration it stands in
    // for must agree at every step.
    let both = |g: &mut Graph, t: &str| {
        let counted = rows(g, &format!("MATCH ()-[:{t}]->() RETURN count(*) AS c"));
        let listed = rows(g, &format!("MATCH ()-[e:{t}]->(x) RETURN x.missing AS c"));

        assert_eq!(
            counted,
            vec![vec![Value::Num(listed.len() as f64)]],
            "`[:{t}]` count disagrees with enumerating it"
        );
    };

    both(&mut g, "R");
    both(&mut g, "S");
    both(&mut g, "R|S");

    // Remove the FIRST type, promoting `S`.
    g.remove_edge_label(0, "R");

    both(&mut g, "R");
    both(&mut g, "S");
    both(&mut g, "R|S");

    // Add it back, then delete an edge entirely.
    g.add_edge_label(0, "R");
    both(&mut g, "R|S");
    g.remove_edge(1);
    both(&mut g, "S");
    both(&mut g, "R|S");
}

/// The vectorized min/max fold and the scalar one agree on a computed NaN.
///
/// Nothing tells a caller which path answered, so a NaN rule that lives in only
/// one of them is a wrong answer half the time. Each fold had its OWN, and each
/// was wrong in a different direction: the ungrouped column fold used
/// `f64::min`/`max`, which silently DROP a NaN, so `max` returned the largest
/// real number; the grouped fold used `partial_cmp`, which answers None against
/// a NaN — never the wanted ordering — so a first-seen NaN STUCK and `min`
/// returned it. Both now call `num_total_cmp`, the one definition.
///
/// The policy: NaN is the greatest value, so `max` keeps it and `min` never
/// picks one — unless every value is a NaN, when there is nothing else to pick.
#[test]
fn the_vectorized_extreme_agrees_with_the_scalar_one_on_a_nan() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"age":29,"t":"x"}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"age":27,"t":"x"}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"age":32,"t":"x"}}"#,
        r#"{"type":"node","id":"d","labels":["P"],"properties":{"age":28,"t":"y"}}"#,
        r#"{"type":"node","id":"e","labels":["P"],"properties":{"age":36,"t":"y"}}"#,
    ]);

    for (query, want) in [
        // Ungrouped: two NaNs and one real.
        (
            "MATCH (n:P) WHERE n.t = 'x' RETURN min(sqrt(n.age - 30)) AS lo, max(sqrt(n.age - 30)) AS hi",
            "[[Num(1.4142135623730951), Num(NaN)]]",
        ),
        // GROUPED, which is a SECOND fold with its own history: group 'x' mixes
        // NaNs with a real, group 'y' does too.
        (
            "MATCH (n:P) RETURN n.t AS t, min(sqrt(n.age - 30)) AS lo, max(sqrt(n.age - 30)) AS hi GROUP BY n.t ORDER BY t",
            "[[Str(\"x\"), Num(1.4142135623730951), Num(NaN)], [Str(\"y\"), Num(2.449489742783178), Num(NaN)]]",
        ),
        // EVERY value a NaN: there is no real extreme to prefer, so both ends
        // are the NaN rather than null.
        (
            "MATCH (n:P) WHERE n.age < 30 RETURN min(sqrt(n.age - 40)) AS lo, max(sqrt(n.age - 40)) AS hi",
            "[[Num(NaN), Num(NaN)]]",
        ),
    ] {
        let vectorized = format!("{:?}", rows(&mut g, query));
        let scalar = format!(
            "{:?}",
            super::eval::with_vec_override(false, || rows(&mut g, query))
        );

        assert_eq!(vectorized, want, "the vectorized fold on `{query}`");
        assert_eq!(scalar, want, "the scalar fold on `{query}`");
    }
}

/// `GROUP BY` names a BOUND variable, and an unbound one faults rather than
/// silently collapsing the result.
///
/// ISO GQL's `groupingElement` is a `bindingVariableReference`. An unbound name
/// lowered to the `UNBOUND` slot, which reads as NULL — so every row keyed the
/// same and the query returned ONE group holding the first row's values. It
/// returned that for `GROUP BY zzz` as readily as for a RETURN alias, and a
/// caller could not tell either from a real answer.
#[test]
fn a_group_by_must_name_a_bound_variable() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"t":"x"}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"t":"x"}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"t":"y"}}"#,
    ]);

    for query in [
        // A RETURN alias is an OUTPUT column; `GROUP BY` runs over the INPUT
        // bindings, so it cannot see one. This is the spelling people write.
        "MATCH (n:P) RETURN n.t AS t, count(*) AS c GROUP BY t",
        // A name that exists nowhere — the same fault, which is the point: the
        // old behaviour could not tell these apart from a correct query.
        "MATCH (n:P) RETURN n.t AS t, count(*) AS c GROUP BY zzz",
        // The property spelling has to check its VARIABLE too.
        "MATCH (n:P) RETURN n.t AS t, count(*) AS c GROUP BY zzz.t",
    ] {
        let err = crate::gql::prepare(query)
            .expect("plans")
            .execute(&mut g, &crate::gql::eval::Params::new())
            .expect_err("an unbound GROUP BY key must fault");

        assert_eq!(
            err.code,
            crate::error_codes::ErrorCode::UnknownFunction,
            "on `{query}`"
        );
        assert!(
            err.message.contains("unbound variable"),
            "the message must name the problem: {}",
            err.message
        );
    }
}

/// Every spelling that DOES bind the key groups the same way.
///
/// `LET` is how ISO spells it (a grouping element must already be a binding
/// variable, so a computed key has to be bound before the projection); `WITH`
/// is the same thing across a clause boundary; the property form is our
/// superset. All three have to agree, and none of them may agree with the
/// broken one — hence the count assertion, which a collapsed grouping fails.
#[test]
fn the_bound_group_by_spellings_agree() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"t":"x"}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"t":"x"}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"t":"y"}}"#,
    ]);

    let want = vec![
        vec![Value::Str("x".into()), n(2.0)],
        vec![Value::Str("y".into()), n(1.0)],
    ];

    for query in [
        "MATCH (n:P) LET t = n.t RETURN t, count(*) AS c GROUP BY t ORDER BY t",
        "MATCH (n:P) WITH n.t AS t RETURN t, count(*) AS c GROUP BY t ORDER BY t",
        "MATCH (n:P) RETURN n.t AS t, count(*) AS c GROUP BY n.t ORDER BY t",
    ] {
        assert_eq!(rows(&mut g, query), want, "on `{query}`");
    }
}

/// `ORDER BY <alias>` answers exactly what `ORDER BY <what the alias names>`
/// answers — it is now the same PLAN, so this checks the rewrite did not change
/// the query on the way.
///
/// The rewrite substitutes a sort key that is a bare output alias with the
/// expression that alias names, remapped into the sort scope. It fires only for
/// the direct column shapes, and the cases below are the ones where "the same
/// values" could stop being true: a shadowed name, an aggregate, a group key, a
/// computed alias, nulls, and `RETURN *`.
#[test]
fn ordering_by_an_alias_matches_ordering_by_what_it_names() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"k":"m","n":3,"m":{"city":"z"}}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"k":"a","n":1,"m":{"city":"y"}}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"k":"z","n":3,"m":{"city":"x"}}}"#,
        // No `k`, no `m` — a NULL sort key, whose placement is absolute.
        r#"{"type":"node","id":"d","labels":["P"],"properties":{"n":2}}"#,
    ]);

    // Each pair means the same thing; the alias spelling is the rewritten one.
    for (alias, named) in [
        (
            "MATCH (u:P) RETURN u.k AS a ORDER BY a",
            "MATCH (u:P) RETURN u.k AS a ORDER BY u.k",
        ),
        (
            "MATCH (u:P) RETURN u.k AS a ORDER BY a DESC",
            "MATCH (u:P) RETURN u.k AS a ORDER BY u.k DESC",
        ),
        // Nulls: placement is absolute, so it must survive the rewrite in BOTH
        // directions and under an explicit NULLS FIRST.
        (
            "MATCH (u:P) RETURN u.k AS a ORDER BY a NULLS FIRST",
            "MATCH (u:P) RETURN u.k AS a ORDER BY u.k NULLS FIRST",
        ),
        // A dotted column.
        (
            "MATCH (u:P) RETURN u.m.city AS a ORDER BY a",
            "MATCH (u:P) RETURN u.m.city AS a ORDER BY u.m.city",
        ),
        // The whole element as the alias.
        (
            "MATCH (u:P) RETURN u AS a ORDER BY a",
            "MATCH (u:P) RETURN u AS a ORDER BY u",
        ),
        // A second column the sort does not read is still projected.
        (
            "MATCH (u:P) RETURN u.k AS a, u.n AS b ORDER BY a",
            "MATCH (u:P) RETURN u.k AS a, u.n AS b ORDER BY u.k",
        ),
        // Ties in the key: the rewrite must not disturb which row wins them.
        (
            "MATCH (u:P) RETURN u.n AS a, u.k AS b ORDER BY a",
            "MATCH (u:P) RETURN u.n AS a, u.k AS b ORDER BY u.n",
        ),
        // Paging over the sorted result.
        (
            "MATCH (u:P) RETURN u.k AS a ORDER BY a LIMIT 2",
            "MATCH (u:P) RETURN u.k AS a ORDER BY u.k LIMIT 2",
        ),
        (
            "MATCH (u:P) RETURN u.k AS a ORDER BY a OFFSET 1 LIMIT 2",
            "MATCH (u:P) RETURN u.k AS a ORDER BY u.k OFFSET 1 LIMIT 2",
        ),
        // DISTINCT keys on the projected row and still needs every column.
        (
            "MATCH (u:P) RETURN DISTINCT u.n AS a ORDER BY a",
            "MATCH (u:P) RETURN DISTINCT u.n AS a ORDER BY u.n",
        ),
        // AGGREGATING: the alias is a GROUP KEY, so every row in a group shares
        // it — substituting reads it off the group's binding instead.
        (
            "MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY a",
            "MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY u.n",
        ),
    ] {
        assert_eq!(
            rows(&mut g, alias),
            rows(&mut g, named),
            "`{alias}` disagreed with `{named}`"
        );

        // And the scalar driver, which is a different sort site entirely.
        assert_eq!(
            super::eval::with_vec_override(false, || rows(&mut g, alias)),
            rows(&mut g, named),
            "the scalar driver disagreed on `{alias}`"
        );
    }
}

/// The shapes the alias rewrite must NOT touch, each for its own reason.
///
/// These are where "the alias names an input expression" stops holding, and
/// substituting anyway would sort by something the query did not ask for.
#[test]
fn the_alias_rewrite_leaves_alone_what_it_cannot_prove() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"k":"m","n":3}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"k":"a","n":1}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"k":"z","n":1}}"#,
    ]);

    // An alias that SHADOWS the input variable it is derived from. The input `u`
    // is not in the sort scope at all, so there is nothing to rewrite to — and
    // `ORDER BY u` here means the output column, the strings, not the element.
    assert_eq!(
        rows(&mut g, "MATCH (u:P) RETURN u.k AS u ORDER BY u"),
        rows(&mut g, "MATCH (u:P) RETURN u.k AS x ORDER BY x"),
        "a shadowing alias must still sort by the OUTPUT column"
    );

    // An AGGREGATE alias is defined over the group, not over any one input row.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY c DESC, a"
        ),
        vec![vec![n(1.0), n(2.0)], vec![n(3.0), n(1.0)],],
        "ordering by an aggregate alias"
    );

    // A COMPUTED alias is left unsubstituted on purpose — it must still answer.
    assert_eq!(
        rows(&mut g, "MATCH (u:P) RETURN upper(u.k) AS a ORDER BY a"),
        rows(
            &mut g,
            "MATCH (u:P) RETURN upper(u.k) AS a ORDER BY upper(u.k)"
        ),
        "a computed alias"
    );

    // `RETURN *` has no item list to substitute FROM.
    assert_eq!(
        rows(&mut g, "MATCH (u:P) RETURN * ORDER BY u.n, u.k"),
        rows(&mut g, "MATCH (u:P) RETURN u AS u ORDER BY u.n, u.k"),
        "RETURN * with an ORDER BY"
    );
}

/// An inline `{k: v}` constraint answers exactly what the general path answers,
/// including every case the column-resolved shortcut must NOT take.
///
/// The shortcut resolves the property column and the wanted value once per
/// segment instead of per neighbour, and then compares raw column entries — an
/// `f64` or an interned `u32`. That is only equality for the shapes it accepts,
/// so what matters is the cases it declines: a value that reads a slot, a
/// heterogeneous column, a key the store has never seen, a string absent from
/// the dictionary, and a type that does not line up with its column.
#[test]
fn an_inline_constraint_agrees_with_the_general_path() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":7,"s":"x","mix":1}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"n":0,"s":"y","mix":"one"}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"n":7,"s":"x"}}"#,
        r#"{"type":"node","id":"d","labels":["P"],"properties":{"other":1}}"#,
        r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{"w":7,"t":"k"}}"#,
        r#"{"type":"edge","id":"e1","labels":["R"],"from":"b","to":"c","properties":{"w":0,"t":"j"}}"#,
        r#"{"type":"edge","id":"e2","labels":["R"],"from":"c","to":"d","properties":{"w":7}}"#,
    ]);

    for query in [
        // The shapes the shortcut takes.
        "MATCH ()-[:R]->(b {n: 7}) RETURN count(*) AS c",
        "MATCH ()-[:R]->(b {s: 'x'}) RETURN count(*) AS c",
        "MATCH ()-[r:R {w: 7}]->() RETURN count(*) AS c",
        "MATCH ()-[r:R {t: 'k'}]->() RETURN count(*) AS c",
        // Two constraints on one element.
        "MATCH ()-[:R]->(b {n: 7, s: 'x'}) RETURN count(*) AS c",
        // On BOTH ends of the hop at once.
        "MATCH ()-[r:R {w: 7}]->(b {n: 0}) RETURN count(*) AS c",
        // A string the dictionary has never seen matches nothing — and must not
        // be read as "no constraint".
        "MATCH ()-[:R]->(b {s: 'nope'}) RETURN count(*) AS c",
        // An absent property. Zero, not a match against null.
        "MATCH ()-[:R]->(b {other: 1}) RETURN count(*) AS c",
        // A key NO element carries, so the store has no column for it.
        "MATCH ()-[:R]->(b {missing: 1}) RETURN count(*) AS c",
        // A HETEROGENEOUS column (number on one node, string on another) is
        // boxed, so the typed shortcut declines and the general path answers.
        "MATCH ()-[:R]->(b {mix: 1}) RETURN count(*) AS c",
        "MATCH ()-[:R]->(b {mix: 'one'}) RETURN count(*) AS c",
        // A type that does not line up with its column.
        "MATCH ()-[:R]->(b {n: 'seven'}) RETURN count(*) AS c",
        "MATCH ()-[:R]->(b {s: 7}) RETURN count(*) AS c",
        // Signed zero is one value, whichever way it is written.
        "MATCH ()-[:R]->(b {n: -0.0}) RETURN count(*) AS c",
        // Null is never equal to anything, including an absent property.
        "MATCH ()-[:R]->(b {n: null}) RETURN count(*) AS c",
        // The rows themselves, not just a count — the shortcut decides which
        // neighbours survive, so the SET has to match too.
        "MATCH (a)-[:R]->(b {n: 7}) RETURN a.s AS x, b.s AS y ORDER BY x, y",
    ] {
        assert_eq!(
            rows(&mut g, query),
            super::eval::with_vec_override(false, || rows(&mut g, query)),
            "the expansion filter disagreed with the scalar driver on `{query}`"
        );
    }

    // The counts themselves, so the test cannot pass by both paths being wrong.
    assert_eq!(
        rows(&mut g, "MATCH ()-[:R]->(b {n: 7}) RETURN count(*) AS c"),
        vec![vec![n(1.0)]],
        "only c has n = 7 at the far end of an R edge"
    );
    assert_eq!(
        rows(
            &mut g,
            "MATCH ()-[:R]->(b {s: 'nope'}) RETURN count(*) AS c"
        ),
        vec![vec![n(0.0)]],
        "a string absent from the dictionary matches nothing"
    );
    assert_eq!(
        rows(&mut g, "MATCH ()-[r:R {w: 7}]->() RETURN count(*) AS c"),
        vec![vec![n(2.0)]],
        "two R edges carry w = 7"
    );
}

/// A constraint whose value READS A SLOT is not constant, and keeps the general
/// path — it genuinely differs per row.
///
/// The shortcut evaluates the value once per segment. That is only sound while
/// the value cannot change between rows, so this is the case that would break
/// silently if the check were dropped: every row would be compared against the
/// first row's value.
#[test]
fn an_inline_constraint_that_reads_a_slot_is_evaluated_per_row() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":1}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"n":2}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"n":1}}"#,
        r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{}}"#,
        r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"c","properties":{}}"#,
        r#"{"type":"edge","id":"e2","labels":["R"],"from":"b","to":"c","properties":{}}"#,
    ]);

    // `{n: a.n}` is a correlated equality: b must carry the SAME n as a. Only
    // a→c (1 = 1) qualifies; a→b (1 vs 2) and b→c (2 vs 1) do not.
    let query = "MATCH (a:P)-[:R]->(b {n: a.n}) RETURN count(*) AS c";

    assert_eq!(rows(&mut g, query), vec![vec![n(1.0)]]);
    assert_eq!(
        rows(&mut g, query),
        super::eval::with_vec_override(false, || rows(&mut g, query)),
        "a per-row constraint disagreed with the scalar driver"
    );

    // A PARAM is constant across rows, so it does take the shortcut — with the
    // param's value, not a placeholder.
    let mut params = Params::new();
    params.insert("v".to_string(), super::eval::Val::Num(1.0));

    assert_eq!(
        qp(
            &mut g,
            "MATCH ()-[:R]->(b {n: $v}) RETURN count(*) AS c",
            params
        ),
        vec![vec![n(2.0)]],
        "two R edges land on a node with n = 1"
    );
}

/// An inline edge constraint with no index behind it seeds from the NODES, not
/// from the edge-type bucket.
///
/// The bucket fallback materializes every edge of the type and then throws most
/// of them away; the node-seeded expansion applies the same constraint during
/// the walk and never builds the vector. On 50k vertices / 150k edges that was
/// 2.84ms against 1.33ms — and it reached Gremlin too, since
/// `g.V().outE('R').has('w', 1)` compiles to exactly this pattern.
///
/// The rows are what this asserts: changing which end a pattern seeds from is
/// the classic way to silently drop or duplicate matches.
#[test]
fn an_unindexed_inline_edge_constraint_still_matches_every_row() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"k":"a"}}"#,
        r#"{"type":"node","id":"b","labels":["P"],"properties":{"k":"b"}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"k":"c"}}"#,
        r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{"w":1}}"#,
        r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"c","properties":{"w":2}}"#,
        r#"{"type":"edge","id":"e2","labels":["R"],"from":"b","to":"c","properties":{"w":1}}"#,
        // A SELF LOOP carrying the constraint — one row, not two, and easy to
        // lose when the seed side changes.
        r#"{"type":"edge","id":"e3","labels":["R"],"from":"c","to":"c","properties":{"w":1}}"#,
        // Another type entirely, which the constraint must not reach.
        r#"{"type":"edge","id":"e4","labels":["S"],"from":"a","to":"b","properties":{"w":1}}"#,
    ]);

    assert_eq!(
        rows(
            &mut g,
            "MATCH (x)-[r:R {w: 1}]->(y) RETURN x.k AS a, y.k AS b ORDER BY a, b"
        ),
        vec![
            vec![Value::Str("a".into()), Value::Str("b".into())],
            vec![Value::Str("b".into()), Value::Str("c".into())],
            vec![Value::Str("c".into()), Value::Str("c".into())],
        ]
    );

    // The same query with an INDEX on the key takes the edge seek instead, and
    // has to answer identically — the two seeds are the equivalent spellings
    // here, chosen by what the store happens to carry.
    let unindexed = rows(
        &mut g,
        "MATCH (x)-[r:R {w: 1}]->(y) RETURN x.k AS a ORDER BY a",
    );

    g.create_edge_index("w");

    assert_eq!(
        rows(
            &mut g,
            "MATCH (x)-[r:R {w: 1}]->(y) RETURN x.k AS a ORDER BY a"
        ),
        unindexed,
        "an index changed the answer, not just the plan"
    );

    // And the scalar driver agrees with both.
    assert_eq!(
        super::eval::with_vec_override(false, || rows(
            &mut g,
            "MATCH (x)-[r:R {w: 1}]->(y) RETURN x.k AS a ORDER BY a"
        )),
        unindexed
    );
}

/// The reverse semi-join agrees with the general path, including on everything
/// it must decline.
///
/// `try_count_semi_join` answers `count(*)` over `[NOT] EXISTS { (a)-[:T]->(b) }`
/// by seeding `b` and walking BACK, instead of testing every `a`. It only
/// understood a label bucket as the seed — so `EXISTS { (a)-[:R]->(b) WHERE
/// b.n = 7 }`, a constraint that makes `b` far more selective and is exactly
/// what a reverse join wants, declined and tested all 20k outer rows: 3.7ms,
/// against 0.037ms once it seeds through the shared node scan.
///
/// `with_vec_override(false, …)` turns the shortcut off, so each query below is
/// answered twice by two different machines.
#[test]
fn the_reverse_semi_join_agrees_with_the_general_path() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":7,"s":"x"}}"#,
        r#"{"type":"node","id":"b","labels":["P","W"],"properties":{"n":7,"s":"y"}}"#,
        r#"{"type":"node","id":"c","labels":["P"],"properties":{"n":1,"s":"x"}}"#,
        r#"{"type":"node","id":"d","labels":["Q"],"properties":{"n":7}}"#,
        // No `n` at all.
        r#"{"type":"node","id":"e","labels":["P"],"properties":{"s":"z"}}"#,
        r#"{"type":"edge","id":"r0","labels":["R"],"from":"a","to":"b","properties":{}}"#,
        r#"{"type":"edge","id":"r1","labels":["R"],"from":"a","to":"c","properties":{}}"#,
        r#"{"type":"edge","id":"r2","labels":["R"],"from":"c","to":"d","properties":{}}"#,
        r#"{"type":"edge","id":"r3","labels":["S"],"from":"e","to":"b","properties":{}}"#,
        // A SELF LOOP, which the shortcut refuses (it would count `a` as its own
        // predecessor).
        r#"{"type":"edge","id":"r4","labels":["R"],"from":"e","to":"e","properties":{}}"#,
    ]);

    for query in [
        // The shape the seed now reaches.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        "MATCH (a:P) WHERE NOT EXISTS { MATCH (a)-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        // The inline spelling of the same constraint.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b {n: 7}) } RETURN count(*) AS c",
        // A label AND a where.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b:W) WHERE b.n = 7 } RETURN count(*) AS c",
        // A key no element carries, and one that matches nothing.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.missing = 1 } RETURN count(*) AS c",
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.n = 999 } RETURN count(*) AS c",
        // NOT seekable — a range, which `scan_node` scans and residual-filters.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.n > 1 } RETURN count(*) AS c",
        // The other direction.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)<-[:R]-(b) WHERE b.n = 7 } RETURN count(*) AS c",
        // A CORRELATED inner where reads `a`, which a backward walk cannot apply
        // — it has to decline and still answer.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.n = a.n } RETURN count(*) AS c",
        // No outer label: `a` is every vertex.
        "MATCH (a) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        // Nothing selective about `b` — the forward test is cheaper and this
        // declines.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b) } RETURN count(*) AS c",
        // CHAINS. Run forward per row these are O(rows · degree^hops) exactly
        // where no walk exists; backwards they are a level per hop.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->()-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        "MATCH (a:P) WHERE NOT EXISTS { MATCH (a)-[:R]->()-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->()-[:R]->()-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        // Mixed directions along the chain, so reversing has to flip each hop.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->()<-[:R]-(b) WHERE b.n = 7 } RETURN count(*) AS c",
        "MATCH (a:P) WHERE EXISTS { MATCH (a)<-[:R]-()-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        // Different types per hop, and a type nothing carries.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:S]->()-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:NOPE]->()-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        // An INTERMEDIATE node that is constrained — a backward walk keeps no
        // rows to filter, so this must decline.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(m:W)-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c",
        // The far end bound by an inline prop rather than a WHERE.
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->()-[:R]->(b {n: 7}) } RETURN count(*) AS c",
    ] {
        assert_eq!(
            rows(&mut g, query),
            super::eval::with_vec_override(false, || rows(&mut g, query)),
            "the reverse semi-join disagrees with the general path on `{query}`"
        );
    }

    // The answers themselves, so the pairs cannot agree by both being wrong.
    // `a` reaches `b` (n = 7) and `c` reaches `d` (n = 7); `b` has no out-R and
    // `e` only loops to itself, which has no `n`.
    assert_eq!(
        rows(
            &mut g,
            "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c"
        ),
        vec![vec![n(2.0)]]
    );
    // Four P vertices, so three do not.
    assert_eq!(
        rows(&mut g, "MATCH (a:P) WHERE NOT EXISTS { MATCH (a)-[:R]->(b) WHERE b.n = 7 } RETURN count(*) AS c"),
        vec![vec![n(2.0)]]
    );
}

/// Counting edges under a WHERE that talks only about the relationship.
///
/// The shortcut answers straight from the candidate set, so it is only correct
/// when the seek IS the predicate. Every case here where it is not — a
/// comparison the columns cannot run, half of an AND, a constrained endpoint —
/// has to fall back rather than count candidates.
#[test]
fn counting_edges_under_a_relationship_where() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["M"],"properties":{}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{"w":1,"s":"x"}}"#,
        r#"{"type":"edge","id":"e1","from":"b","to":"c","labels":["R"],"properties":{"w":2,"s":"y"}}"#,
        r#"{"type":"edge","id":"e2","from":"c","to":"a","labels":["R"],"properties":{"w":1}}"#,
        // a self-loop, and an edge of another type
        r#"{"type":"edge","id":"e3","from":"a","to":"a","labels":["R"],"properties":{"w":1}}"#,
        r#"{"type":"edge","id":"e4","from":"a","to":"b","labels":["S"],"properties":{"w":1}}"#,
    ];
    let mut g = graph_of(&lines);
    let n = |g: &mut Graph, q: &str| -> i64 {
        match rows(g, q)[0][0] {
            Value::Num(x) => x as i64,
            _ => -1,
        }
    };

    // The lowered shape: three R edges have w = 1.
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w = 1 RETURN count(*) AS c"
        ),
        3
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w > 1 RETURN count(*) AS c"
        ),
        1
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.s = 'x' RETURN count(*) AS c"
        ),
        1
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w IN [1, 2] RETURN count(*) AS c"
        ),
        4
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w = 1 OR r.w = 2 RETURN count(*) AS c"
        ),
        4
    );
    // Type sets, and a type nothing carries.
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R|S]->() WHERE r.w = 1 RETURN count(*) AS c"
        ),
        4
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:NOPE]->() WHERE r.w = 1 RETURN count(*) AS c"
        ),
        0
    );

    // Predicates the columns cannot run. These must NOT be read off the
    // candidate set — `<>` and `IS NULL` lower to nothing at all, so the seek
    // would be every edge of the type.
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w <> 1 RETURN count(*) AS c"
        ),
        1
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.s IS NULL RETURN count(*) AS c"
        ),
        2
    );
    // HALF of an AND lowers; the other half decides the answer.
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w = 1 AND r.s IS NULL RETURN count(*) AS c"
        ),
        2
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w = 1 AND r.s = 'x' RETURN count(*) AS c"
        ),
        1
    );

    // A WHERE that reaches an ENDPOINT is a filter on the pattern, not the edge.
    assert_eq!(
        n(
            &mut g,
            "MATCH (u)-[r:R]->() WHERE r.w = 1 AND u:M RETURN count(*) AS c"
        ),
        0
    );
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->(v:N) WHERE r.w = 1 RETURN count(*) AS c"
        ),
        2
    );
    // Both ends bound to the SAME name is a self-loop test, which the edge set
    // knows nothing about.
    assert_eq!(
        n(
            &mut g,
            "MATCH (u)-[r:R]->(u) WHERE r.w = 1 RETURN count(*) AS c"
        ),
        1
    );
    // An inline constraint on the relationship means the same as the WHERE.
    assert_eq!(
        n(&mut g, "MATCH ()-[r:R {w: 1}]->() RETURN count(*) AS c"),
        3
    );

    // A deleted edge is not a candidate.
    rows(&mut g, "MATCH ()-[r:R]->() WHERE r.w = 2 DELETE r");
    assert_eq!(
        n(
            &mut g,
            "MATCH ()-[r:R]->() WHERE r.w > 0 RETURN count(*) AS c"
        ),
        3
    );
}

/// DISTINCT windows over DISTINCT rows, not over scanned rows — so `SKIP 2` drops
/// the first two distinct values, not the first two rows.
///
/// An expression item keeps this on the GENERIC dedup path (the `group_ids` fast
/// path takes only direct property columns), which is the one where the skip, the
/// limit and the dedup are interleaved in a single pass.
#[test]
fn distinct_with_skip_and_limit_windows_over_distinct_rows() {
    let mut lines = Vec::new();

    // Values 0,0,0,1,1,1,2,2,2,… so every distinct value spans three rows.
    for i in 0..30usize {
        lines.push(format!(
            r#"{{"type":"node","id":"n{i}","labels":["N"],"properties":{{"n":{}}}}}"#,
            i / 3
        ));
    }

    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let mut g = graph_of(&refs);
    let nums = |g: &mut Graph, q: &str| -> Vec<f64> {
        rows(g, q)
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                _ => f64::NAN,
            })
            .collect()
    };

    let all = "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v ORDER BY u.n + 0";

    assert_eq!(nums(&mut g, all).len(), 10);
    // No ORDER BY: scan order, which for this fixture is 0..9 ascending.
    assert_eq!(
        nums(&mut g, "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v"),
        (0..10).map(f64::from).collect::<Vec<_>>()
    );
    assert_eq!(
        nums(&mut g, "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v LIMIT 3"),
        vec![0.0, 1.0, 2.0]
    );
    // SKIP counts DISTINCT rows: 2 skipped means the third distinct value first,
    // not the third scanned row (which is still 0).
    assert_eq!(
        nums(&mut g, "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v SKIP 2"),
        (2..10).map(f64::from).collect::<Vec<_>>()
    );
    assert_eq!(
        nums(
            &mut g,
            "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v SKIP 2 LIMIT 3"
        ),
        vec![2.0, 3.0, 4.0]
    );
    // Past the end, and exactly at it.
    assert_eq!(
        nums(
            &mut g,
            "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v SKIP 9 LIMIT 5"
        ),
        vec![9.0]
    );
    assert!(nums(&mut g, "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v SKIP 10").is_empty());
    assert!(nums(&mut g, "MATCH (u:N) RETURN DISTINCT u.n + 0 AS v LIMIT 0").is_empty());
}

/// Control characters in a string cell serialize the way JS `JSON.stringify`
/// does: the short escapes where JSON defines one, `\uXXXX` otherwise.
///
/// `RowSet::to_json` is the FFI query-result path, so these are the bytes the TS
/// side sees. It used its own escaper until the crate's three copies were merged,
/// and that copy had no `\b`/`\f` arms — it fell through to the
/// control-character rule and emitted the long form. Same string, different
/// bytes, on the one surface where the bytes are the contract.
#[test]
fn control_characters_serialize_the_way_javascript_does() {
    let mut g =
        graph_of(&[r#"{"type":"node","id":"a","labels":["N"],"properties":{"s":"x\by\fz"}}"#]);
    let json = crate::gql::parse("MATCH (u:N) RETURN u.s AS s")
        .expect("parses")
        .execute(&mut g, &crate::gql::eval::Params::new())
        .expect("runs")
        .to_json();

    assert_eq!(json, r#"{"columns":["s"],"rows":[["x\by\fz"]]}"#);

    // A control character with no short escape keeps the `\u` form, which is
    // also what JS does.
    let mut g2 =
        graph_of(&[r#"{"type":"node","id":"a","labels":["N"],"properties":{"s":"\u0001"}}"#]);

    assert_eq!(
        crate::gql::parse("MATCH (u:N) RETURN u.s AS s")
            .expect("parses")
            .execute(&mut g2, &crate::gql::eval::Params::new())
            .expect("runs")
            .to_json(),
        r#"{"columns":["s"],"rows":[["\u0001"]]}"#
    );
}

/// `-0.0` and `0.0` are ONE grouping key, and both signs really do reach the
/// column.
///
/// Ported here 2026-08-07 from `query.rs`, which held it as an agreement test
/// between the real engine and a second, hand-rolled one that existed to produce
/// a benchmark fingerprint. That engine keyed on raw bits and gave two rows where
/// this one gives one — two implementations in one crate disagreeing on a settled
/// decision, which is the argument for having deleted it rather than fixing it
/// again. The half that survives is the half about the engine users reach.
///
/// The `assert_ne!` on the bit patterns is load-bearing: without it the test
/// passes just as well on a fixture where the two signs never both got stored.
#[test]
fn signed_zeros_are_one_group() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"z":0.0}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"z":-0.0}}"#,
    ]
    .join("\n");
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    let kid = g.props.keys.get("z").expect("the key exists");
    let bits: Vec<u64> = (0..2)
        .map(
            |i| match crate::value::Value::from_column(&g.props, kid, i, &g.strs, false) {
                crate::value::Value::Num(x) => x.to_bits(),
                other => panic!("expected a number, got {other:?}"),
            },
        )
        .collect();

    assert_ne!(bits[0], bits[1], "both zero signs must reach the column");

    assert_eq!(
        crate::gql::parse("MATCH (u:N) RETURN DISTINCT u.z")
            .expect("parses")
            .execute(&mut g, &crate::gql::eval::Params::new())
            .expect("runs")
            .nrows,
        1
    );
}

/// Is the MIGRATION route — a Gremlin tail compiled to a GQL projection — cheaper
/// than the Gremlin terminals it replaces?
///
/// This is the question that decides whether the migration is worth finishing, and
/// the answer is yes. Over 50k vertices / 150k edges, against `MIGRATE_OFF`:
///
///   values(n)                0.123ms  0.124   1.01x
///   values(n).sum()          0.124    0.128   1.04x
///   values(n).dedup()        0.182    0.164   0.90x
///   groupCount().by(n)       0.190    1.189   6.27x
///   values(k)                0.344    0.344   1.00x
///
/// Parity everywhere except grouping, where GQL's `GROUP BY` beats Gremlin's
/// hand-rolled tally by 6x — one engine's optimization arriving in the other,
/// which is the entire argument for doing this.
///
/// `dedup` is 0.90x — 11% SLOWER, which is at the noise floor this repo warns
/// about but is the one row trending the wrong way. GQL's DISTINCT builds a hash
/// per row where `Col::dedup` keys on a bit pattern. Re-measure it in isolation
/// before finishing the migration; do not let it ride on "within noise".
///
/// An earlier note on `pattern.rs` REJECTED routing unconstrained prefixes here,
/// measuring the frame at 0.578ms against 0.188 for the old path. That measurement
/// predates this route: it was taken when `plan_pattern_ids` materialized ids and
/// handed them to Gremlin's terminals, where `run_pattern_projection` keeps the
/// frame and projects from it. The rejection deserves re-testing on those terms.
#[test]
#[ignore = "probe"]
fn migration_route_cost() {
    let mut lines = String::new();
    for i in 0..50_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"k\":\"key{i:06}\"}}}}\n",
            i % 97
        ));
    }
    let mut e = 0;
    for i in 0..50_000usize {
        for d in 0..3usize {
            lines.push_str(&format!(
                "{{\"type\":\"edge\",\"id\":\"e{e}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
                (i * 31 + d * 7 + 1) % 50_000
            ));
            e += 1;
        }
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    println!(
        "{:<52} {:>10} {:>10} {:>7}",
        "traversal", "migrated", "terminals", "x"
    );
    for q in [
        "g.V().out('R').hasLabel('W').values('n')",
        "g.V().out('R').hasLabel('W').values('n').sum()",
        "g.V().out('R').hasLabel('W').values('n').dedup()",
        "g.V().out('R').hasLabel('W').groupCount().by('n')",
        "g.V().out('R').hasLabel('W').values('k')",
    ] {
        let (mut a, mut b) = (f64::MAX, f64::MAX);
        for _ in 0..7 {
            crate::gremlin::exec::MIGRATE_OFF.with(|c| c.set(false));
            let p = crate::gremlin::parse(q).expect("parses");
            let t = std::time::Instant::now();
            let x = p.run(&mut g);
            a = a.min(t.elapsed().as_secs_f64() * 1000.0);

            crate::gremlin::exec::MIGRATE_OFF.with(|c| c.set(true));
            let p2 = crate::gremlin::parse(q).expect("parses");
            let t2 = std::time::Instant::now();
            let y = p2.run(&mut g);
            b = b.min(t2.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(x, y, "{q}");
        }
        crate::gremlin::exec::MIGRATE_OFF.with(|c| c.set(false));
        println!("{q:<52} {a:>9.3}ms {b:>9.3}ms {:>6.2}x", b / a);
    }
}

/// Does the `pattern.rs` "nothing past the start is constrained" decline still
/// cost more than it saves, now that the TAIL compiles to a GQL projection too?
///
/// It does. Measured 2026-08-07 by flipping the decline off, 50k vertices with one
/// out-edge each — the numbers are on `pattern.rs` beside the decline itself:
/// every shape got 1.5-2.7x WORSE.
///
/// The question mattered because this is the gate on deleting a CONTAINER rather
/// than an arm. `elem_terminal`'s navigation (`OutV`/`InV`/`BothV`) and filter
/// (`Where`/`Not`) arms exist only because the LINEAR route hands ids over
/// MID-traversal, so they can never move into a tail translator; they would go if
/// every prefix compiled as a pattern. At 1.5-2.7x they do not go that way.
///
/// So the migration's reach is bounded: it takes TAIL arms, and the container
/// survives. Re-running this after a change to `build_scan`'s setup cost is the
/// only thing that would reopen it.
#[test]
#[ignore = "probe"]
fn the_unconstrained_prefix_decline_still_pays() {
    let mut lines = String::new();
    for i in 0..50_000usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{}}}}}\n",
            i % 97
        ));
    }
    for i in 0..50_000usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
            (i * 31 + 1) % 50_000
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    // These are the shapes the decline sends down the LINEAR route: nothing past
    // the start is constrained, so `pattern::compile` refuses them today.
    for q in [
        "g.V().out('R').count()",
        "g.V().out('R')",
        "g.V().out('R').values('n')",
        "g.V().out('R').values('n').sum()",
        "g.V().out('R').dedup().count()",
    ] {
        let mut best = f64::MAX;
        let mut rows = 0;
        for _ in 0..7 {
            let p = crate::gremlin::parse(q).expect("parses");
            let t = std::time::Instant::now();
            let out = p.run(&mut g);
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            rows = out.len();
        }
        println!("UNC {best:>8.3}ms {rows:>7} rows  {q}");
    }
}

/// INVESTIGATION PROBE for the "Gremlin as a clause sequence, not a
/// prefix+tail split" question (see `docs/design/query-ir.md`).
///
/// Three shapes, GQL text parsed with `crate::gql::eval::prepare_plan` (parse +
/// lower — the real pipeline, not a hand-built `CClause`) so the plan is
/// guaranteed correct, timed three ways: (a) the Gremlin traversal as it runs
/// today, (b) the equivalent GQL plan through `execute` (`RowSet`-boxing), (c)
/// the same plan through whichever columnar exit (if any) the clause shape
/// reaches — `vectorized_linear_cols` for the aggregate/WITH-chain shape,
/// `vectorized_single_match_cols` for a plain single-`MATCH`+`RETURN`.
///
/// Measured 2026-08-07, 50k vertices / 150k edges, min of 7 (two runs, stable):
///
/// ```text
/// case                       gremlin   gql/rows   gql/cols  cols-via
/// seeded 1-hop count         0.061ms    0.061ms    0.061ms  linear (aggregate/WITH-chain)
/// far-end label, values      0.153ms    0.245ms    0.119ms  single-match (plain projection)
/// semi-join count (EXISTS)   1.245ms    3.583ms    3.606ms  linear (aggregate/WITH-chain)
/// ```
///
/// Two different stories, not one:
///
/// - The plain-projection shape (`values(n)`) is exactly what the
///   `RowSet`-boxing tax was theorized to cost: `gql/rows` is 1.6x SLOWER than
///   Gremlin, and skipping the box (`gql/cols`) flips that into 1.29x FASTER —
///   the columnar exit is the whole gap.
/// - The EXISTS semi-join gets NO benefit from the columnar exit (3.583 vs
///   3.606ms, noise-level) because `eval_vec` has no vectorized arm for
///   `CExpr::Exists` (see `gql/eval.rs` — `eval_vec`'s `_ => gen(e)` catch-all
///   falls through to a per-row scalar-VM re-evaluation of the sub-pattern for
///   every candidate row). `RowSet` boxing was never the cost there; the
///   per-shape count shortcuts that used to answer this (`try_count_semi_join`
///   et al., named in the now-stale doc comment on
///   `examples/cross_engine_shortcuts.rs`) are gone from `fastpath.rs` on this
///   branch — retired in favor of the general planner — and nothing replaced
///   them for a sub-pattern inside a filter. That is a real, separate gap: it
///   would need `eval_vec` to gain an `Exists` arm, not a wider columnar exit.
/// - The single-row aggregate (`count()`) shows parity all three ways — with
///   one output row `RowSet` boxing is one allocation, not a cost worth
///   chasing.
///
/// Checked separately, because `fastpath.rs` shrank from 2,065 lines to 600 on
/// this branch and took a family of count/semi-join shortcuts with it: the
/// semi-join row above is NOT a regression from that removal. Measured on `main`
/// in a scratch worktree, the same query costs 3.549ms against 3.583 here — the
/// deleted shortcuts never covered this shape, so the gap is pre-existing and
/// belongs to the missing `eval_vec` arm. (`MATCH (u:V) RETURN u.n` went the
/// other way on this branch, 0.402ms -> 0.245.)
///
/// SUPERSEDED for the semi-join row, and by the fix this probe motivated: once
/// `eval_vec` gained a vectorized `CExpr::Exists` arm, that row went 3.648ms ->
/// 0.370, which is 3.4x FASTER than Gremlin's 1.274 rather than 2.9x slower. So
/// all three shapes now favour the clause route or tie it, and the conclusion
/// "the semi-join needs separate work before it can migrate" is answered rather
/// than outstanding. Re-run this probe after touching the columnar exit; the
/// numbers above the fold are the pre-fix ones and are kept for the contrast.
#[test]
#[ignore = "probe"]
fn clause_sequence_columnar_exit_probe() {
    use crate::gql::eval::{vectorized_linear_cols, vectorized_single_match_cols, Val};

    // Same shape as `migration_route_cost`'s fixture, so k/n/labels/degree line
    // up with the three traversals below: 50k vertices (label V, every 10th
    // also W), properties n (i%97, numeric) and k ("key{i:06}", a unique
    // string key), 3 out-edges of type R each -> 150k edges.
    let mut lines = String::new();
    for i in 0..50_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"k\":\"key{i:06}\"}}}}\n",
            i % 97
        ));
    }
    let mut e = 0;
    for i in 0..50_000usize {
        for d in 0..3usize {
            lines.push_str(&format!(
                "{{\"type\":\"edge\",\"id\":\"e{e}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
                (i * 31 + d * 7 + 1) % 50_000
            ));
            e += 1;
        }
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    const REPS: usize = 7;

    fn as_num(v: &Val) -> f64 {
        match v {
            Val::Num(x) => *x,
            other => panic!("expected a Num, got {other:?}"),
        }
    }
    fn row_num(v: &Value) -> f64 {
        match v {
            Value::Num(x) => *x,
            other => panic!("expected a Num, got {other:?}"),
        }
    }

    // (name, gremlin, gql)
    let cases: &[(&str, &str, &str)] = &[
        (
            "seeded 1-hop count",
            "g.V().has('k','key000005').out('R').count()",
            "MATCH (a)-[:R]->(b) WHERE a.k = 'key000005' RETURN count(*) AS c",
        ),
        (
            "far-end label, values",
            "g.V().out('R').hasLabel('W').values('n')",
            "MATCH ()-[:R]->(b:W) RETURN b.n AS n",
        ),
        (
            "semi-join count (EXISTS)",
            "g.V().hasLabel('V').where(__.out('R')).count()",
            "MATCH (a:V) WHERE EXISTS { MATCH (a)-[:R]->() } RETURN count(*) AS c",
        ),
    ];

    println!(
        "50000 vertices, degree 3, min of {REPS}\n\n{:<26} {:>10} {:>10} {:>10}  cols-via",
        "case", "gremlin", "gql/rows", "gql/cols"
    );

    for (label, gq, gql) in cases {
        // (a) Gremlin, as it runs today.
        let gp = crate::gremlin::parse(gq).unwrap_or_else(|e| panic!("gremlin parses `{gq}`: {e}"));
        let mut t_grem = f64::MAX;
        let mut grem_out: Vec<Val> = Vec::new();
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            grem_out = gp.run(&mut g);
            t_grem = t_grem.min(t.elapsed().as_secs_f64() * 1000.0);
        }

        // (b) The equivalent GQL statement through `execute` (`RowSet`-boxed).
        let prepared =
            crate::gql::prepare(gql).unwrap_or_else(|e| panic!("gql prepares `{gql}`: {e}"));
        let params = Params::new();
        let mut t_rows = f64::MAX;
        let mut rows_out = prepared.execute(&mut g, &params).expect("gql runs");
        for _ in 0..REPS {
            let t = std::time::Instant::now();
            rows_out = prepared.execute(&mut g, &params).expect("gql runs");
            t_rows = t_rows.min(t.elapsed().as_secs_f64() * 1000.0);
        }

        // Answers, normalized for container shape: a Gremlin `count()` is one
        // scalar, a GQL count is a one-row/one-column `RowSet`; `values(n)` is a
        // Gremlin list in TRAVERSAL order against an unordered GQL row set
        // (order is unspecified across both engines — see
        // `docs/design/query-ir.md`), so sort both before comparing as
        // multisets rather than sequences.
        let mut grem_nums: Vec<f64> = grem_out.iter().map(as_num).collect();
        let mut gql_nums: Vec<f64> = rows_out.rows().map(|r| row_num(&r[0])).collect();
        grem_nums.sort_by(f64::total_cmp);
        gql_nums.sort_by(f64::total_cmp);
        assert_eq!(
            grem_nums, gql_nums,
            "{label}: gremlin `{gq}` and gql `{gql}` disagree"
        );

        // (c) The same lowered plan through whichever columnar exit the clause
        // shape reaches, if any.
        let plan = crate::gql::eval::prepare_plan(gql).expect("gql lowers");
        let linear = &plan.parts[0];
        let mut cols_via = "declined (scalar/general driver)";
        let mut t_cols = f64::NAN;

        if vectorized_linear_cols(linear, &g, &plan, &[]).is_some() {
            cols_via = "linear (aggregate/WITH-chain)";
            let mut best = f64::MAX;
            let mut cols = None;
            for _ in 0..REPS {
                let t = std::time::Instant::now();
                cols = vectorized_linear_cols(linear, &g, &plan, &[]);
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            t_cols = best;
            let cols = cols.expect("engaged above");
            let mut nums: Vec<f64> = (0..cols.first().map_or(0, |c| c.len()))
                .flat_map(|i| cols.iter().map(move |c| c.with_val_at(i, as_num)))
                .collect();
            nums.sort_by(f64::total_cmp);
            assert_eq!(nums, gql_nums, "{label}: linear columnar exit disagrees");
        } else if vectorized_single_match_cols(linear, &g, &plan, &[]).is_some() {
            cols_via = "single-match (plain projection)";
            let mut best = f64::MAX;
            let mut cols = None;
            for _ in 0..REPS {
                let t = std::time::Instant::now();
                cols = vectorized_single_match_cols(linear, &g, &plan, &[]);
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            }
            t_cols = best;
            let cols = cols.expect("engaged above");
            let mut nums: Vec<f64> = (0..cols.first().map_or(0, |c| c.len()))
                .flat_map(|i| cols.iter().map(move |c| c.with_val_at(i, as_num)))
                .collect();
            nums.sort_by(f64::total_cmp);
            assert_eq!(
                nums, gql_nums,
                "{label}: single-match columnar exit disagrees"
            );
        }

        println!(
            "{label:<26} {t_grem:>8.3}ms {t_rows:>8.3}ms {t_cols:>8.3}ms  {cols_via}   ({} rows)",
            rows_out.nrows
        );
    }
}

/// A fixture with the shapes a BACKWARD semi-join sweep can get wrong, all in a
/// graph small enough to check by hand: a self-loop (`e`), a directed triangle
/// (`p→q→r→p`), a multi-label far end (`b` is `V,W`), two edge types on the same
/// source (`a-R->b`, `a-S->c`), and a vertex with no edges at all (`d`).
fn back_semi_fixture() -> Graph {
    semi_join_fixture()
}

/// The two routes must agree ROW FOR ROW on every shape the sweep accepts.
///
/// `forcing_backward_semi_join` is what makes this test mean anything: the cost
/// model would decline all of these on an eight-vertex graph, so without the
/// force this would compare the forward walk against itself and pass no matter
/// what the sweep computed.
#[test]
fn a_backward_semi_join_agrees_with_the_forward_walk() {
    let mut g = back_semi_fixture();
    for q in [
        // Labeled far end — the shape the sweep exists for.
        "MATCH (u:V) WHERE EXISTS { (u)-[:R]->(:W) } RETURN u.id AS id",
        "MATCH (u:V) WHERE NOT EXISTS { (u)-[:R]->(:W) } RETURN u.id AS id",
        // The far end is the SOURCE's own label, so the self-loop at `e` and
        // every triangle edge qualify — a hop reversed the wrong way shows up
        // here as the triangle answering backwards.
        "MATCH (u:V) WHERE EXISTS { (u)-[:R]->(:V) } RETURN u.id AS id",
        "MATCH (u:Tri) WHERE EXISTS { (u)-[:R]->(:Tri) } RETURN u.id AS id",
        // Reversed direction, and undirected — `Both` must not become `Out`.
        "MATCH (u:V) WHERE EXISTS { (u)<-[:R]-(:V) } RETURN u.id AS id",
        "MATCH (u:Tri) WHERE EXISTS { (u)<-[:R]-(:Tri) } RETURN u.id AS id",
        "MATCH (u:V) WHERE EXISTS { (u)-[:R]-(:V) } RETURN u.id AS id",
        // An inline property constraint instead of a label, and both together.
        "MATCH (u:V) WHERE EXISTS { (u)-[:R]->({id: 'b'}) } RETURN u.id AS id",
        "MATCH (u:V) WHERE EXISTS { (u)-[:R]->(:W {id: 'b'}) } RETURN u.id AS id",
        // A constraint that matches NOTHING: an empty far set must sweep to an
        // all-false map, not to "unconstrained".
        "MATCH (u:V) WHERE EXISTS { (u)-[:R]->(:W {id: 'zzz'}) } RETURN u.id AS id",
        // An edge type that does not exist matches nothing — the conflation this
        // module has written five times reads it as ANY type instead.
        "MATCH (u:V) WHERE EXISTS { (u)-[:NOPE]->(:W) } RETURN u.id AS id",
        // A type UNION lowers to two ids; `S` only reaches `c`, `R` only `b`.
        "MATCH (u:V) WHERE EXISTS { (u)-[:R|S]->(:V) } RETURN u.id AS id",
        // EXISTS as a projected value, not a filter — same column, different
        // consumer.
        "MATCH (u:V) RETURN u.id AS id, EXISTS { (u)-[:R]->(:W) } AS r",
        // …and folded, which is the shape the whole thing was slow for.
        "MATCH (u:V) WHERE EXISTS { (u)-[:R]->(:W) } RETURN count(*) AS c",
    ] {
        let back = super::eval::forcing_backward_semi_join(|| rows(&mut g, q));
        let fwd = super::eval::without_backward_semi_join(|| rows(&mut g, q));
        assert_eq!(back, fwd, "backward != forward for `{q}`");
    }
}

/// The self-loop is the case the two `SelfLoops` conventions disagree about:
/// GQL walks it ONCE and Gremlin twice, and `reach_back` takes that as a
/// parameter. `e-[:R]->e` means `e` reaches a `:V`, exactly once.
#[test]
fn a_backward_semi_join_walks_a_self_loop_once() {
    let mut g = back_semi_fixture();
    let q = "MATCH (u:V) WHERE EXISTS { (u)-[:R]->(:V) } RETURN u.id AS id";
    let back = super::eval::forcing_backward_semi_join(|| rows(&mut g, q));
    // `a→b` and the `e→e` loop; `b`, `c`, `d` have no `R` out-edge to a `:V`.
    assert_eq!(back, vec![vec![s("a")], vec![s("e")]]);
}

/// An UNCONSTRAINED far end declines the sweep — with no label and no property
/// the far set is every vertex, so the sweep would mark the whole graph to
/// answer what the forward walk answers on the first edge. The answer must stay
/// right either way.
#[test]
fn an_unconstrained_far_end_stays_on_the_forward_walk() {
    let mut g = back_semi_fixture();
    let q = "MATCH (u:V) WHERE EXISTS { (u)-[:R]->() } RETURN u.id AS id";
    let forced = super::eval::forcing_backward_semi_join(|| rows(&mut g, q));
    let fwd = super::eval::without_backward_semi_join(|| rows(&mut g, q));
    assert_eq!(
        forced, fwd,
        "an unconstrained far end must not change the answer"
    );
    assert_eq!(forced, vec![vec![s("a")], vec![s("e")]]);
}

/// The cost model declines a far end that is not narrower than the outer set.
/// Forcing the sweep must not change the ANSWER — only which route computes it —
/// so a graph where every vertex is a valid far end is still correct both ways.
#[test]
fn a_wide_far_end_agrees_with_the_narrow_route() {
    let mut g = back_semi_fixture();
    // Every `:V` is a far-end candidate here, so the far set is as wide as the
    // outer set — the case the `cap` exists to turn away.
    let q = "MATCH (u:V) WHERE EXISTS { (u)-[:R]->(:V) } RETURN count(*) AS c";
    let auto = rows(&mut g, q);
    let forced = super::eval::forcing_backward_semi_join(|| rows(&mut g, q));
    assert_eq!(auto, forced);
    assert_eq!(auto, vec![vec![n(2.0)]]);
}

/// A graph built so that WALK and TRAIL counts differ: two self-loops, parallel
/// edges, and a cycle back to the start. `try_walk_count` folds walks and GQL
/// counts trails, so every assertion here is really about the correction.
fn trail_vs_walk_fixture() -> Graph {
    graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"k":1}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{"k":2}}"#,
        r#"{"type":"node","id":"c","labels":["M"],"properties":{"k":3}}"#,
        r#"{"type":"edge","id":"x1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"x2","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"x3","from":"b","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"x4","from":"a","to":"a","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"x5","from":"b","to":"c","labels":["S"],"properties":{}}"#,
    ])
}

/// The counting shortcut must agree with the enumerating matcher on every shape
/// it accepts — most of all the two-hop ones, where a walk may take one edge
/// twice and a trail may not.
#[test]
fn the_walk_count_shortcut_agrees_with_enumeration() {
    let mut g = trail_vs_walk_fixture();
    for q in [
        // One hop: walks and trails are the same, with and without DISTINCT.
        "MATCH (a)-[:R]->(b) RETURN count(*) AS c",
        "MATCH (a:N)-[:R]->(b) RETURN count(*) AS c",
        "MATCH (a)-[:R]->(b) RETURN count(DISTINCT b) AS c",
        "MATCH (a)<-[:R]-(b) RETURN count(*) AS c",
        "MATCH (a)-[:R]-(b) RETURN count(*) AS c",
        // Two hops — the self-loops at `a` and `b` make walks exceed trails.
        "MATCH (a)-[:R]->()-[:R]->(c) RETURN count(*) AS c",
        "MATCH (a:N)-[:R]->()-[:R]->(c) RETURN count(*) AS c",
        "MATCH (a)<-[:R]-()<-[:R]-(c) RETURN count(*) AS c",
        // Mixed types across the hops: the same edge can only be reused when it
        // matches BOTH, which `S` does not.
        "MATCH (a)-[:R]->()-[:S]->(c) RETURN count(*) AS c",
        "MATCH (a)-[:R|S]->()-[:R]->(c) RETURN count(*) AS c",
        // Quantified, which decomposes into the fixed lengths and corrects each.
        "MATCH (a)-[:R]->{1,2}(b) RETURN count(*) AS c",
        "MATCH (a:N)-[:R]->{1,2}(b) RETURN count(*) AS c",
        "MATCH (a)-[:R]->{2,2}(b) RETURN count(*) AS c",
        // A seed the shortcut narrows through `scan_node`, not by filtering rows.
        "MATCH (a:N {k: 1})-[:R]->()-[:R]->(c) RETURN count(*) AS c",
        // An edge type that resolves to nothing counts nothing.
        "MATCH (a)-[:NOPE]->()-[:R]->(c) RETURN count(*) AS c",
    ] {
        let shortcut = rows(&mut g, q);
        let enumerated = super::eval::without_walk_count(|| rows(&mut g, q));
        assert_eq!(
            shortcut, enumerated,
            "walk-count shortcut != matcher for `{q}`"
        );
    }
}

/// The two spellings of a two-hop count DIFFERENT things, and the shortcut has
/// to keep that difference rather than tidy it up.
///
/// `R` edges: `x1`,`x2` (a→b), `x3` (b→b), `x4` (a→a).
///
/// As separate segments the engine counts WALKS — an edge may repeat:
/// x1→x3, x2→x3, x3→x3, x4→x1, x4→x2, x4→x4 = 6.
///
/// Quantified, it counts TRAILS — x3→x3 and x4→x4 are the same edge twice and
/// drop out = 4.
#[test]
fn a_quantified_two_hop_excludes_the_edge_taken_twice() {
    let mut g = trail_vs_walk_fixture();
    let walks = rows(&mut g, "MATCH (a)-[:R]->()-[:R]->(c) RETURN count(*) AS c");
    let trails = rows(&mut g, "MATCH (a)-[:R]->{2,2}(c) RETURN count(*) AS c");
    assert_eq!(walks, vec![vec![n(6.0)]], "separate segments count walks");
    assert_eq!(
        trails,
        vec![vec![n(4.0)]],
        "a quantified repetition counts trails"
    );
    // …and both agree with the matcher that enumerates.
    assert_eq!(
        walks,
        super::eval::without_walk_count(|| rows(
            &mut g,
            "MATCH (a)-[:R]->()-[:R]->(c) RETURN count(*) AS c"
        ))
    );
    assert_eq!(
        trails,
        super::eval::without_walk_count(|| rows(
            &mut g,
            "MATCH (a)-[:R]->{2,2}(c) RETURN count(*) AS c"
        ))
    );
}

/// `count(DISTINCT <endpoint>)` over separate segments is the walk-reachable
/// SET, which `walk_count` deduplicates itself. From `a` and `b` over two `R`
/// hops that set is `{a, b}`: `b` via x1→x3, and `a` via x4→x4.
#[test]
fn a_distinct_two_hop_count_is_the_reachable_set() {
    let mut g = trail_vs_walk_fixture();
    let q = "MATCH (a)-[:R]->()-[:R]->(c) RETURN count(DISTINCT c) AS c";
    let shortcut = rows(&mut g, q);
    let enumerated = super::eval::without_walk_count(|| rows(&mut g, q));
    assert_eq!(shortcut, enumerated);
    assert_eq!(shortcut, vec![vec![n(2.0)]]);
}

/// A quantified repetition with `DISTINCT` declines the shortcut — the lengths
/// would be summed, and an endpoint reachable at both would be counted twice.
/// The answer still has to be right.
#[test]
fn a_distinct_quantified_count_stays_on_the_matcher() {
    let mut g = trail_vs_walk_fixture();
    let q = "MATCH (a)-[:R]->{1,2}(c) RETURN count(DISTINCT c) AS c";
    let shortcut = rows(&mut g, q);
    let enumerated = super::eval::without_walk_count(|| rows(&mut g, q));
    assert_eq!(shortcut, enumerated);
}

/// The grouped counting shortcut must agree with the enumerating path, and
/// agree IN ORDER — `GROUP BY` emits groups in the order their first row
/// appeared, so a tally that folded in vertex order would be a wrong answer
/// that looks completely reasonable.
#[test]
fn the_grouped_walk_count_agrees_with_enumeration_in_order() {
    let mut g = trail_vs_walk_fixture();
    for q in [
        "MATCH (a)-[:R]->(b) RETURN b.k AS k, count(*) AS n",
        "MATCH (a:N)-[:R]->(b) RETURN b.k AS k, count(*) AS n",
        "MATCH (a)<-[:R]-(b) RETURN b.k AS k, count(*) AS n",
        "MATCH (a)-[:R]-(b) RETURN b.k AS k, count(*) AS n",
        "MATCH (a)-[:R]->()-[:R]->(c) RETURN c.k AS k, count(*) AS n",
        "MATCH (a)-[:R]->()-[:S]->(c) RETURN c.k AS k, count(*) AS n",
        "MATCH (a)-[:R]->{1,2}(b) RETURN b.k AS k, count(*) AS n",
        "MATCH (a)-[:R]->{2,2}(b) RETURN b.k AS k, count(*) AS n",
        "MATCH (a:N {k: 1})-[:R]->()-[:R]->(c) RETURN c.k AS k, count(*) AS n",
        // The key is the ELEMENT, not a property of it.
        "MATCH (a)-[:R]->(b) RETURN b AS b, count(*) AS n",
        // A key that is an expression over the endpoint, and one that is
        // constant — both still read only the endpoint slot.
        "MATCH (a)-[:R]->(b) RETURN b.k + 1 AS k, count(*) AS n",
        // A missing property groups under NULL rather than dropping the row.
        "MATCH (a)-[:R]->(b) RETURN b.nope AS k, count(*) AS n",
    ] {
        let shortcut = rows(&mut g, q);
        let enumerated = super::eval::without_walk_count(|| rows(&mut g, q));
        assert_eq!(
            shortcut, enumerated,
            "grouped shortcut != matcher for `{q}`"
        );
    }
}

/// Group order is FIRST-SEEN, and the fixture is built so vertex order and
/// first-seen order disagree: `z` sorts last by id but is the first endpoint
/// reached from the first seed, so it must be the first group.
#[test]
fn grouped_walk_count_emits_groups_in_first_seen_order() {
    let mut g = graph_of(&[
        r#"{"type":"node","id":"a","labels":["N"],"properties":{"g":"first"}}"#,
        r#"{"type":"node","id":"m","labels":["N"],"properties":{"g":"second"}}"#,
        r#"{"type":"node","id":"z","labels":["N"],"properties":{"g":"first"}}"#,
        // `a`'s only edge lands on `z` (the LAST vertex), so first-seen order is
        // z's group, then m's.
        r#"{"type":"edge","id":"q1","from":"a","to":"z","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"q2","from":"a","to":"m","labels":["R"],"properties":{}}"#,
    ]);
    let q = "MATCH (a:N)-[:R]->(b) RETURN b.g AS g, count(*) AS n";
    let shortcut = rows(&mut g, q);
    assert_eq!(
        shortcut,
        super::eval::without_walk_count(|| rows(&mut g, q)),
        "the shortcut must not reorder groups"
    );
    assert_eq!(
        shortcut,
        vec![vec![s("first"), n(1.0)], vec![s("second"), n(1.0)]]
    );
}
