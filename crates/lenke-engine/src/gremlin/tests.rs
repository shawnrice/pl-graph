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

/// The VALUE cells of each row, order-independent and NAME-independent — so a
/// GQL result and a Gremlin result can be compared even though their column
/// names differ.
fn value_bag(rows: &Rows) -> Vec<String> {
    let mut out: Vec<String> = rows
        .rows
        .iter()
        .map(|r| r.iter().map(|v| format!("{v:?};")).collect::<String>())
        .collect();
    out.sort();
    out
}

fn gremlin_rows(q: &str, store: &Store) -> Rows {
    let plan = super::parse(q).unwrap_or_else(|e| panic!("parse gremlin `{q}`: {e}"));
    run(&plan, store)
}
fn gql_rows(q: &str, store: &Store) -> Rows {
    let plan = crate::gql::parse(q).unwrap_or_else(|e| panic!("parse gql `{q}`: {e}"));
    run(&plan, store)
}

/// A string compare on a DICTIONARY-encoded column (built by from_ndjson for a
/// categorical property) must match the general/`Str` path exactly for every operator
/// — `=`/`<>` via code equality, ordering via the decoded string. Absent nodes drop.
#[test]
fn dict_column_string_filter_matches_all_ops() {
    let cities = ["oslo", "bergen", "trondheim", "oslo", "bergen"];
    let mut nd = String::new();
    for (i, c) in (0..250u32).map(|i| (i, cities[(i % 5) as usize])) {
        nd.push_str(&format!(
            r#"{{"id":"{i}","labels":["N"],"props":{{"city":"{c}"}}}}"#
        ));
        nd.push('\n');
    }
    // two nodes with NO city (must drop from every compare)
    nd.push_str(r#"{"id":"250","labels":["N"],"props":{"z":"x"}}"#);
    nd.push('\n');
    nd.push_str(r#"{"id":"251","labels":["N"],"props":{"z":"y"}}"#);
    nd.push('\n');
    let st = crate::ndjson::from_ndjson(&nd).unwrap();
    assert!(
        matches!(st.column("city"), Some(crate::store::Column::Dict { .. })),
        "city should be dict-encoded"
    );
    let cnt = |q: &str| match &gremlin_rows(q, &st).rows[0][0] {
        Value::Num(x) => *x as i64,
        other => panic!("not a count: {other:?}"),
    };
    let want = |f: &dyn Fn(&str) -> bool| {
        (0..250u32).filter(|&i| f(cities[(i % 5) as usize])).count() as i64
    };
    assert_eq!(
        cnt("g.V().has('city', eq('oslo')).count()"),
        want(&|c| c == "oslo")
    );
    // neq keeps present-and-unequal PLUS the 2 absent nodes (Not(And(exists,eq)) 3VL).
    assert_eq!(
        cnt("g.V().has('city', neq('oslo')).count()"),
        want(&|c| c != "oslo") + 2
    );
    assert_eq!(cnt("g.V().has('city', eq('nowhere')).count()"), 0);
    assert_eq!(
        cnt("g.V().has('city', gt('c')).count()"),
        want(&|c| c > "c")
    );
    assert_eq!(
        cnt("g.V().has('city', lte('oslo')).count()"),
        want(&|c| c <= "oslo")
    );
}

/// `repeat(x).times(1)` is rewritten to a single `Expand` at plan time — so it MUST
/// return exactly what the explicit one-hop `x()` does, INCLUDING a self-loop (Walk
/// keeps A->A). Compare the rewritten var-length form to the explicit hop across
/// out/in/both and typed edges, count + distinct + a value projection.
#[test]
fn repeat_times_one_equals_explicit_hop() {
    let mut b = Builder::default();
    for i in 0..40u32 {
        b.node(&["N"], &[("v", n(f64::from(i)))]);
    }
    for i in 0u32..40 {
        b.edge(i, (i + 1) % 40, "R");
        b.edge(i, (i * 3 + 7) % 40, "F");
    }
    b.edge(0, 0, "R"); // self-loop — Walk keeps it on a 1-hop
    let st = b.build();
    let bag = |q: &str| value_bag(&gremlin_rows(q, &st));
    for (rep, hop) in [
        (
            "g.V().repeat(__.out()).times(1).count()",
            "g.V().out().count()",
        ),
        (
            "g.V().repeat(__.both()).times(1).count()",
            "g.V().both().count()",
        ),
        (
            "g.V().repeat(__.in('R')).times(1).count()",
            "g.V().in('R').count()",
        ),
        (
            "g.V().repeat(__.both()).times(1).dedup().count()",
            "g.V().both().dedup().count()",
        ),
        (
            "g.V().repeat(__.out()).times(1).values('v')",
            "g.V().out().values('v')",
        ),
        ("g.V('0').repeat(__.out('R')).times(1)", "g.V('0').out('R')"),
    ] {
        assert_eq!(bag(rep), bag(hop), "{rep} vs {hop}");
    }
}

/// `has(k, neq(v))` desugars to `Not(And(PropertyExists{k}, Eq{k,v}))`; the raw fast
/// path must match that 3VL exactly — an absent-`k` node IS kept (false AND null =
/// false, Not = true), a present `k != v` is kept, a present `k == v` is dropped.
/// The GQL differential engine already pins this against core; here we lock the row
/// set directly on a graph where some nodes lack the key.
#[test]
fn has_neq_keeps_absent_and_unequal() {
    let mut b = Builder::default();
    // ids 0..30: age = i % 5; ids 30..40: NO age at all.
    for i in 0..30u32 {
        b.node(&["N"], &[("age", n(f64::from(i % 5)))]);
    }
    for _ in 30..40u32 {
        b.node(&["N"], &[("other", s("x"))]);
    }
    let st = b.build();
    let cnt = |q: &str| match &gremlin_rows(q, &st).rows[0][0] {
        Value::Num(x) => *x as i64,
        other => panic!("not a count: {other:?}"),
    };
    // neq(2): drop the 6 present nodes with age==2 (i in {2,7,12,17,22,27}); keep the
    // other 24 present + ALL 10 absent = 34.
    assert_eq!(cnt("g.V().has('age', neq(2)).count()"), 34);
    // neq over a value NO present node has (age==99): keep all 30 present + 10 absent.
    assert_eq!(cnt("g.V().has('age', neq(99)).count()"), 40);
    // Sanity: eq(2) is the complement over PRESENT nodes only — 6.
    assert_eq!(cnt("g.V().has('age', eq(2)).count()"), 6);
}

/// A MIXED conjunction filter — the shape a projection creates (`values(k)` AND-s a
/// `PropertyExists{k}` onto the user's `has(...)`) — must keep EXACTLY the rows
/// satisfying every conjunct, whichever order the fast path evaluates them in. Some
/// nodes lack `name` (so PropertyExists actually gates), and the selective `age`
/// equality vs the non-selective presence gate exercises the ordering heuristic.
#[test]
fn mixed_conjunction_filter_keeps_exactly_all_conjuncts() {
    let mut b = Builder::default();
    // 60 nodes; age = i%6 (so age==3 hits 10 of them); name present only when i%2==0.
    for i in 0..60u32 {
        let mut props: Vec<(&str, Value)> =
            vec![("age", n(f64::from(i % 6))), ("score", n(f64::from(i)))];
        if i % 2 == 0 {
            props.push(("name", s("nm")));
        }
        b.node(&["N"], &props);
    }
    let st = b.build();
    let cnt = |q: &str| match &gremlin_rows(q, &st).rows[0][0] {
        Value::Num(x) => *x as i64,
        other => panic!("not a count: {other:?}"),
    };
    // age==3 → i in {3,9,15,...,57} (10 nodes), of which name present (i%2==0) → NONE
    // (all are odd). So values('name') over age==3 yields 0 rows.
    assert_eq!(cnt("g.V().has('age', eq(3)).count()"), 10);
    assert_eq!(cnt("g.V().has('age', eq(3)).values('name').count()"), 0);
    // age==2 → i in {2,8,...,56} (10 nodes), all even → name present → 10 names.
    assert_eq!(cnt("g.V().has('age', eq(2)).values('name').count()"), 10);
    // A two-Compare conjunction: age<3 AND score>=30 (rows i%6<3 and i>=30).
    let expect = (0..60u32).filter(|&i| i % 6 < 3 && i >= 30).count() as i64;
    assert_eq!(
        cnt("g.V().has('age', lt(3)).has('score', gte(30)).count()"),
        expect
    );
    // …and with a values() presence gate AND-ed on (name present too).
    let expect2 = (0..60u32)
        .filter(|&i| i % 6 < 3 && i >= 30 && i % 2 == 0)
        .count() as i64;
    assert_eq!(
        cnt("g.V().has('age', lt(3)).has('score', gte(30)).values('name').count()"),
        expect2
    );
}

/// The streaming bounded top-K (`order().by(prop).limit(k)` over a bare scan) must
/// return EXACTLY what a full sort would — same tie order (arrival = node-id order for
/// a V() scan), across ascending/descending and with a skip. Ages have heavy ties, so
/// the tiebreak is exercised. Ground truth: a stable sort by (age, id) in the test.
#[test]
fn streaming_top_k_matches_full_sort_order() {
    let mut b = Builder::default();
    let ages: Vec<f64> = (0..500).map(|i| f64::from(i % 7)).collect(); // many ties
    for &a in &ages {
        b.node(&["N"], &[("age", n(a))]);
    }
    let st = b.build();
    let ids = |q: &str| -> Vec<u32> {
        gremlin_rows(q, &st)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Num(x) => *x as u32,
                Value::Str(s) => s.parse().unwrap(),
                other => panic!("want an id: {other:?}"),
            })
            .collect()
    };
    // Expected: stable sort of (age, id), the arrival tiebreak being id order.
    let mut asc: Vec<u32> = (0..500u32).collect();
    asc.sort_by(|&a, &b2| {
        ages[a as usize]
            .partial_cmp(&ages[b2 as usize])
            .unwrap()
            .then(a.cmp(&b2))
    });
    let desc: Vec<u32> = {
        let mut d: Vec<u32> = (0..500u32).collect();
        d.sort_by(|&a, &b2| {
            ages[b2 as usize]
                .partial_cmp(&ages[a as usize])
                .unwrap()
                .then(a.cmp(&b2))
        });
        d
    };
    // `id()` returns the external id (the dense-id string here) — compare as u32.
    assert_eq!(ids("g.V().order().by('age').limit(10).id()"), asc[..10]);
    assert_eq!(ids("g.V().order().by('age').limit(5).id()"), asc[..5]);
    assert_eq!(
        ids("g.V().order().by('age', desc).limit(10).id()"),
        desc[..10]
    );
    // With a skip (range): `range(3, 13)` == skip 3, limit 10.
    assert_eq!(ids("g.V().order().by('age').range(3, 13).id()"), asc[3..13]);
}

/// The streaming JSON sink (`try_stream_gremlin_json`) must be BYTE-IDENTICAL to
/// materializing the whole result then serializing — same rows, same order, same
/// per-cell rendering. Forced on (`min_rows = 0`) so the gate can't hide a mismatch,
/// over a single-hop `values` chain (the shape it accepts). This is the invariant the
/// cost gate exists to exploit safely, not to protect.
#[test]
fn streaming_json_sink_matches_materialized_bytes() {
    let mut b = Builder::default();
    let n = 200u32;
    for i in 0..n {
        b.node(&["N"], &[("name", s(&format!("p{i}")))]);
    }
    for i in 0..n {
        for d in 1..=3u32 {
            b.edge(i, (i + d) % n, "R"); // out() fans out 3x — a real frontier
        }
    }
    let st = b.build();
    let plan =
        crate::opt::optimize_indexed(super::parse("g.V().out().values('name')").unwrap(), &st);
    let materialized = crate::json::gremlin_results_json(&run(&plan, &st));
    let streamed = crate::exec::try_stream_gremlin_json(&plan, &st, false, 0.0)
        .expect("single-hop values is a streamable shape");
    assert_eq!(streamed, materialized);
}

/// The fused element/value-map JSON writer (`run_gremlin_json`'s node-map fast path)
/// must be BYTE-IDENTICAL to building the `Value::Map` tree then serializing it
/// (`gremlin_results_json(run(..))`) — across the nested render, flat `elementMap`,
/// `valueMap`, a key filter, absent properties, and multi-label nodes. This is the
/// invariant the whole optimization rests on: it may skip the tree only if the bytes
/// are the same.
#[test]
fn fused_map_json_matches_value_tree_bytes() {
    let mut b = Builder::default();
    // Mixed graph: some nodes miss `age`/`city`; every 5th node is also VIP; a bool.
    for i in 0..120u32 {
        let mut props: Vec<(&str, Value)> = vec![
            ("name", s(&format!("p{i}"))),
            ("active", Value::Bool(i % 2 == 0)),
        ];
        if i % 3 != 0 {
            props.push(("age", n(f64::from(i % 40 + 18))));
        }
        if i % 4 != 0 {
            props.push(("city", s(["oslo", "bergen", "tromso"][(i % 3) as usize])));
        }
        let labels: &[&str] = if i % 5 == 0 { &["N", "VIP"] } else { &["N"] };
        b.node(labels, &props);
    }
    for i in 0..120u32 {
        b.edge(i, (i + 1) % 120, "R");
        b.edge(i, (i + 7) % 120, "R");
    }
    let st = b.build();
    for q in [
        "g.V().out().elementMap()",
        "g.V().out().valueMap()",
        "g.V().out().elementMap('name', 'age')",
        "g.V().out().valueMap('city', 'missingkey')",
        "g.V().out()",
        "g.V().hasLabel('VIP').elementMap()",
    ] {
        let plan = crate::opt::optimize_indexed(super::parse(q).unwrap(), &st);
        let fused = crate::exec::run_gremlin_json(&plan, &st);
        let tree = crate::json::gremlin_results_json(&run(&plan, &st));
        assert_eq!(fused, tree, "fused vs value-tree diverged for `{q}`");
    }
}

/// Target-aware shortest-path early stop must be byte-identical to the full BFS: a
/// `Filter{endpoint == t}` over `ShortestPath` bounds the search to the target's
/// distance, but the filter still runs, so the KEPT rows (paths + multiplicity) are
/// unchanged. Compare an INDEXED store (early stop fires) against an UNINDEXED one
/// (full BFS) across ANY/ALL and endpoint/path returns, including a diamond that gives
/// a node two distinct shortest paths (exercises `ALL` multiplicity).
#[test]
fn shortest_path_early_stop_matches_full_bfs() {
    // 0->1->4 and 0->2->4 (two shortest paths to 4), then 4->5->6->7 (a tail).
    let edges = [
        (0, 1),
        (0, 2),
        (1, 4),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 3),
        (3, 4),
    ];
    let mut nd = String::new();
    for i in 0..8u32 {
        nd.push_str(&format!(
            "{{\"id\":\"{i}\",\"labels\":[\"N\"],\"props\":{{\"k\":{i}}}}}\n"
        ));
    }
    for (j, (a, b)) in edges.iter().enumerate() {
        nd.push_str(&format!(
                "{{\"id\":\"e{j}\",\"from\":\"{a}\",\"to\":\"{b}\",\"labels\":[\"R\"],\"props\":{{}}}}\n"
            ));
    }
    let plain = crate::ndjson::from_ndjson(&nd).unwrap();
    let mut indexed = crate::ndjson::from_ndjson(&nd).unwrap();
    indexed.create_index("k"); // makes `try_shortest_early_stop` fire

    let gql_json = |st: &crate::store::Store, q: &str| {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), st);
        crate::json::gql_rows_json(&run(&plan, st))
    };
    for tgt in [4u32, 7, 3] {
        for sel in ["ANY", "ALL"] {
            for ret in ["b.k", "path_length(p)"] {
                let q = format!(
                    "MATCH p = {sel} SHORTEST (a:N {{k:0}})-[]->*(b:N {{k:{tgt}}}) RETURN {ret}"
                );
                assert_eq!(
                    gql_json(&plain, &q),
                    gql_json(&indexed, &q),
                    "early-stop diverged from full BFS for `{q}`"
                );
            }
        }
    }
}

/// `repeat(...).dedup().count()` — the order-free reachability-COUNT shape — must run
/// as a per-level BFS (`varlen_distinct_endpoint_count`), NOT by materializing the
/// exponential walk. Two assertions: (1) the count equals the distinct endpoints a
/// shallow walk actually produces; (2) a DEEP count completes and stays under the trail
/// ceiling — proof it never enumerates the (astronomical) walk.
#[test]
fn reachability_count_is_bfs_not_walk_enumeration() {
    let mut nd = String::new();
    for i in 0..300u32 {
        nd.push_str(&format!(
            "{{\"id\":\"{i}\",\"labels\":[\"N\"],\"props\":{{}}}}\n"
        ));
    }
    for e in 0..900u32 {
        let (from, to) = (e % 300, (e * 11 + 3) % 300);
        nd.push_str(&format!(
                "{{\"id\":\"e{e}\",\"from\":\"{from}\",\"to\":\"{to}\",\"labels\":[\"R\"],\"props\":{{}}}}\n"
            ));
    }
    let st = crate::ndjson::from_ndjson(&nd).unwrap();
    let count = |q: &str| -> f64 {
        let plan = crate::opt::optimize_indexed(super::parse(q).unwrap(), &st);
        crate::exec::run_gremlin_json(&plan, &st)
            .trim_matches(|c| c == '[' || c == ']')
            .parse()
            .unwrap()
    };
    // (1) BFS count == distinct endpoints of the actual 2-step walk.
    let plan =
        crate::opt::optimize_indexed(super::parse("g.V().repeat(both()).times(2)").unwrap(), &st);
    let walk = run(&plan, &st);
    let endpoint_col = walk.names.len() - 1;
    let mut distinct: std::collections::HashSet<String> = std::collections::HashSet::new();
    for row in walk.rows.iter() {
        distinct.insert(format!("{:?}", row[endpoint_col]));
    }
    assert_eq!(
        count("g.V().repeat(both()).times(2).dedup().count()"),
        distinct.len() as f64
    );

    // (2) A DEEP reach-count still completes — impossible via walk enumeration (the
    // default trail ceiling would trip long before depth 30 if it enumerated).
    let deep = count("g.V().repeat(both()).times(30).dedup().count()");
    assert!(
        deep > 0.0 && deep <= 300.0,
        "deep reach-count out of range: {deep}"
    );
}

/// Streaming a var-length endpoint projection to JSON must be (1) BYTE-IDENTICAL to
/// serializing the materialized result — including `values()`'s drop-if-absent
/// semantics (some nodes here lack `name`) — and (2) able to COMPLETE a closure larger
/// than the row cap, since it never materializes the batch (the memory win over core's
/// materialize-then-serialize).
#[test]
fn streamed_varlen_values_matches_and_bypasses_the_row_cap() {
    use crate::store::ConfigId;
    let mut nd = String::new();
    for i in 0..60u32 {
        let labels: &str = if i % 4 == 0 {
            "[\"P\",\"V\"]"
        } else {
            "[\"P\"]"
        };
        // ~1/3 of nodes have NO `name` — the PropertyExists guard must drop them.
        let props = if i % 3 == 0 {
            format!("\"age\":{}", i % 20)
        } else {
            format!("\"age\":{},\"name\":\"p{i}\"", i % 20)
        };
        nd.push_str(&format!(
            "{{\"id\":\"n{i}\",\"labels\":{labels},\"props\":{{{props}}}}}\n"
        ));
    }
    for e in 0..120u32 {
        nd.push_str(&format!(
                "{{\"id\":\"e{e}\",\"from\":\"n{}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"props\":{{}}}}\n",
                e % 60,
                (e * 7 + 1) % 60
            ));
    }
    let mut st = crate::ndjson::from_ndjson(&nd).unwrap();
    // (1) byte-identity across shapes (default cap, both paths complete).
    for q in [
        "g.V().repeat(out()).times(3).values('name')",
        "g.V().repeat(out()).emit().times(2).values('name')",
        "g.V().repeat(both()).times(2).values('name')",
        "g.V().repeat(out()).until(__.hasLabel('V')).times(4).values('name')",
    ] {
        let plan = crate::opt::optimize_indexed(super::parse(q).unwrap(), &st);
        let streamed = crate::exec::run_gremlin_json(&plan, &st);
        let material = crate::json::gremlin_results_json(&run(&plan, &st));
        assert_eq!(
            streamed, material,
            "streamed vs materialized diverged for `{q}`"
        );
    }
    // (2) with a tiny row cap, a modest closure trips the MATERIALIZED path but the
    // streamed path (bounded by output bytes, not rows) still completes.
    st.set_limit(ConfigId::LimitsTrail, 100);
    let plan = crate::opt::optimize_indexed(
        super::parse("g.V().repeat(out()).times(3).values('name')").unwrap(),
        &st,
    );
    assert!(crate::exec::try_run(&plan, &st).is_err()); // materialized: E_RESOURCE
    let streamed = crate::exec::run_gremlin_json(&plan, &st); // streamed: completes
    assert!(streamed.starts_with('[') && streamed.ends_with(']') && streamed.len() > 2);
}

/// The GQL egress streams a var-length endpoint projection to the `{columns,rows}`
/// document, byte-identical to `gql_rows_json(try_run(..))` — including emitting `null`
/// for an absent value (GQL has no `values()` PropertyExists guard) — and completes a
/// closure past the row cap that the materialized path rejects.
#[test]
fn streamed_varlen_gql_matches_and_bypasses_the_row_cap() {
    use crate::store::ConfigId;
    let mut nd = String::new();
    for i in 0..60u32 {
        // ~1/3 of nodes have no `name` -> GQL emits `[null]` (not dropped).
        let props = if i % 3 == 0 {
            format!("\"age\":{}", i % 20)
        } else {
            format!("\"age\":{},\"name\":\"p{i}\"", i % 20)
        };
        nd.push_str(&format!(
            "{{\"id\":\"n{i}\",\"labels\":[\"P\"],\"props\":{{{props}}}}}\n"
        ));
    }
    for e in 0..120u32 {
        nd.push_str(&format!(
                "{{\"id\":\"e{e}\",\"from\":\"n{}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"props\":{{}}}}\n",
                e % 60,
                (e * 7 + 1) % 60
            ));
    }
    let mut st = crate::ndjson::from_ndjson(&nd).unwrap();
    for q in [
        "MATCH (a:P)-[:R]->{1,3}(t) RETURN t.name AS r",
        "MATCH (a:P)-[:R]->{2,2}(t) RETURN t.age AS a",
    ] {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &st);
        let streamed = crate::exec::try_run_gql_json(&plan, &st).unwrap();
        let material = crate::json::gql_rows_json(&crate::exec::try_run(&plan, &st).unwrap());
        assert_eq!(
            streamed, material,
            "GQL streamed vs materialized diverged for `{q}`"
        );
    }
    st.set_limit(ConfigId::LimitsTrail, 100);
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("MATCH (a:P)-[:R]->{3,3}(t) RETURN t.name AS r").unwrap(),
        &st,
    );
    assert!(crate::exec::try_run(&plan, &st).is_err()); // materialized: E_RESOURCE
    let streamed = crate::exec::try_run_gql_json(&plan, &st).unwrap(); // streamed: completes
    assert!(streamed.starts_with("{\"columns\":") && streamed.ends_with("]}"));
}

/// A DEEP traversal (recursion depth = hop count) must not overflow the stack — the
/// recursive var-length DFS runs on a large stack (`on_big_stack`). A 25k-node chain
/// walked end to end recurses ~25k frames, well past the default 8 MB stack, yet must
/// complete. Tiny result (one path), so this exercises DEPTH, not fan-out. Guards both
/// the `try_run` path and (via the JSON entry) the Gremlin sinks that call `pull`
/// directly.
#[test]
fn deep_traversal_runs_on_a_large_stack() {
    let n = 25_000u32;
    let mut nd = String::new();
    for i in 0..n {
        nd.push_str(&format!(
            "{{\"id\":\"n{i}\",\"labels\":[\"P\"],\"props\":{{\"name\":\"p{i}\"}}}}\n"
        ));
    }
    for i in 0..n - 1 {
        nd.push_str(&format!(
                "{{\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"props\":{{}}}}\n",
                i + 1
            ));
    }
    let st = crate::ndjson::from_ndjson(&nd).unwrap();
    // From n0, exactly n-1 R-hops reaches the single last node.
    let q = format!(
        "MATCH (s:P {{name:'p0'}})-[:R]->{{{}}}(t) RETURN t.name AS r",
        n - 1
    );
    let plan = crate::opt::optimize_indexed(crate::gql::parse(&q).unwrap(), &st);
    assert_eq!(run(&plan, &st).rows.len(), 1);
}

/// A runaway per-path `repeat` must trip the `trail` limit with a loud
/// `E_RESOURCE_EXHAUSTED` — never a truncated result, never an OOM — and the ceiling
/// must be configurable (the same anti-runaway contract, and defaults, as core).
#[test]
fn trail_limit_caps_runaway_traversal_and_is_configurable() {
    use crate::store::ConfigId;
    // A dense-ish graph so `repeat(both())` fans out fast.
    let mut nd = String::new();
    for i in 0..200u32 {
        nd.push_str(&format!(
            "{{\"id\":\"{i}\",\"labels\":[\"N\"],\"props\":{{}}}}\n"
        ));
    }
    for e in 0..800u32 {
        let (from, to) = (e % 200, (e * 7 + 1) % 200);
        nd.push_str(&format!(
                "{{\"id\":\"e{e}\",\"from\":\"{from}\",\"to\":\"{to}\",\"labels\":[\"R\"],\"props\":{{}}}}\n"
            ));
    }
    let mut st = crate::ndjson::from_ndjson(&nd).unwrap();
    assert_eq!(st.limits().trail, 1_000_000); // core-matching default

    let run_q = |st: &crate::store::Store, q: &str| {
        let plan = crate::opt::optimize_indexed(super::parse(q).unwrap(), st);
        crate::exec::try_run(&plan, st)
    };
    let is_exhausted = |r: &Result<crate::exec::Rows, String>| matches!(r, Err(e) if e.contains("E_RESOURCE_EXHAUSTED"));

    // A shallow walk is well under the default ceiling — it completes.
    assert!(run_q(&st, "g.V().repeat(both()).times(1)").is_ok());

    // Tighten the ceiling: a walk that exceeds it now errors (not truncates).
    st.set_limit(ConfigId::LimitsTrail, 2_000);
    let tight = run_q(&st, "g.V().repeat(both()).times(3)");
    assert!(
        is_exhausted(&tight),
        "expected E_RESOURCE_EXHAUSTED, got {tight:?}"
    );

    // Raise it back above the result size: the SAME query now completes.
    st.set_limit(ConfigId::LimitsTrail, 100_000_000);
    assert!(run_q(&st, "g.V().repeat(both()).times(3)").is_ok());
}

/// The fused writer must also match the value-tree bytes for EDGE frontiers — the
/// nested `{id, from, to, labels, properties}`, the flat `elementMap` with its
/// `IN`/`OUT` endpoint stubs, and `valueMap` — over edges carrying numeric and string
/// properties (some absent), and for bare `g.E()`. Built from ndjson so the edges have
/// real properties (the test `Builder::edge` sets none).
#[test]
fn fused_edge_map_json_matches_value_tree_bytes() {
    let mut nd = String::new();
    for i in 0..40u32 {
        nd.push_str(&format!(
            "{{\"id\":\"{i}\",\"labels\":[\"N\"],\"props\":{{\"name\":\"p{i}\"}}}}\n"
        ));
    }
    for e in 0..80u32 {
        let (from, to) = (e % 40, (e + 3) % 40);
        // Half the edges carry a string `kind`; all carry numeric `w`; some typed T2.
        let mut props = format!("\"w\":{}", e % 50);
        if e % 2 == 0 {
            props.push_str(&format!(
                ",\"kind\":\"{}\"",
                ["buy", "sell"][(e % 2) as usize]
            ));
        }
        let lbl = if e % 5 == 0 { "T2" } else { "R" };
        nd.push_str(&format!(
                "{{\"id\":\"edge{e}\",\"from\":\"{from}\",\"to\":\"{to}\",\"labels\":[\"{lbl}\"],\"props\":{{{props}}}}}\n"
            ));
    }
    let st = crate::ndjson::from_ndjson(&nd).unwrap();
    for q in [
        "g.E().elementMap()",
        "g.E().valueMap()",
        "g.E().elementMap('w')",
        "g.E().valueMap('kind', 'absent')",
        "g.E()",
        "g.V().outE().elementMap()",
        "g.V().outE().valueMap()",
    ] {
        let plan = crate::opt::optimize_indexed(super::parse(q).unwrap(), &st);
        let fused = crate::exec::run_gremlin_json(&plan, &st);
        let tree = crate::json::gremlin_results_json(&run(&plan, &st));
        assert_eq!(fused, tree, "fused vs value-tree diverged for `{q}`");
    }
}

/// The gate is STRUCTURAL, not just a row count: a chain the streamed form would
/// regress on must be rejected regardless of size. A deeper (2-hop) chain re-runs
/// every hop per block and loses; a VALUE-comparison filter (`has(eq(..))`) defeats
/// the row estimate the floor trusts. Both must return `None` even at `min_rows = 0`,
/// while the one accepted shape (single hop + presence filter) passes.
#[test]
fn streaming_gate_rejects_deep_and_value_filtered_chains() {
    let mut b = Builder::default();
    for i in 0..50u32 {
        b.node(&["N"], &[("name", s(&format!("p{i}")))]);
        b.edge(i, (i + 1) % 50, "R");
    }
    let st = b.build();
    let opt = |q: &str| crate::opt::optimize_indexed(super::parse(q).unwrap(), &st);
    // Accepted: one hop, presence filter from `values`.
    assert!(crate::exec::try_stream_gremlin_json(
        &opt("g.V().out().values('name')"),
        &st,
        false,
        0.0
    )
    .is_some());
    // Rejected: two hops (per-block re-expansion tax).
    assert!(crate::exec::try_stream_gremlin_json(
        &opt("g.V().out().out().values('name')"),
        &st,
        false,
        0.0
    )
    .is_none());
    // Rejected: a value comparison the estimate over-counts.
    assert!(crate::exec::try_stream_gremlin_json(
        &opt("g.V().out().has('name', eq('p5')).values('name')"),
        &st,
        false,
        0.0
    )
    .is_none());
}

/// A type filter must match an edge's SECONDARY label, not just its primary type —
/// the per-edge `edge_has_extra` bit (which skips the extras probe for single-label
/// edges) must be SET for a multi-label edge, or the secondary match is missed.
/// Edge 0→1 is `[F, R]` (R secondary); `out('R')` must still reach node 1.
#[test]
fn edge_type_filter_matches_secondary_label() {
    let nd = [
        r#"{"id":"0","labels":["N"],"props":{}}"#,
        r#"{"id":"1","labels":["N"],"props":{}}"#,
        r#"{"id":"2","labels":["N"],"props":{}}"#,
        r#"{"from":"0","to":"1","labels":["F","R"],"props":{}}"#, // R is SECONDARY
        r#"{"from":"0","to":"2","labels":["R"],"props":{}}"#,     // R is primary
        r#"{"from":"0","to":"1","labels":["F"],"props":{}}"#,     // no R at all
    ]
    .join("\n");
    let st = crate::ndjson::from_ndjson(&nd).unwrap();
    let count = |q: &str| match &gremlin_rows(q, &st).rows[0][0] {
        Value::Num(x) => *x as i64,
        other => panic!("not a count: {other:?}"),
    };
    // Two R-edges from node 0 (the [F,R] secondary + the primary R); the bare F is out.
    assert_eq!(count("g.V('0').out('R').count()"), 2);
    // The [F,R] edge is reached by BOTH out('F') and out('R').
    assert_eq!(count("g.V('0').out('F').count()"), 2); // [F,R] + [F]
                                                       // Distinct R-neighbours of 0: nodes 1 and 2.
    assert_eq!(count("g.V('0').out('R').dedup().count()"), 2);
}

/// `repeat(dir).times(k).dedup().count()` — the distinct-endpoint fast path
/// (per-level set expansion) must equal the count from ENUMERATING the same walk
/// with explicit hops (`both().both()…`, which are ordinary Expand steps, not a
/// VarLength, so they enumerate and dedup for real). A triangle + a self-loop make
/// revisits non-trivial, so distinct < total.
#[test]
fn repeat_dedup_count_matches_explicit_hop_enumeration() {
    let mut b = Builder::default();
    for i in 0..40 {
        b.node(&["N"], &[("k", n(f64::from(i)))]);
    }
    for i in 0u32..40 {
        b.edge(i, (i + 1) % 40, "R"); // a ring
        b.edge(i, (i * 7 + 3) % 40, "R"); // chords → cycles
    }
    b.edge(0, 0, "R"); // self-loop
    let st = b.build();
    let count = |q: &str| match &gremlin_rows(q, &st).rows[0][0] {
        Value::Num(x) => *x as i64,
        other => panic!("not a count: {other:?}"),
    };
    // (fast path via repeat/VarLength, enumerated via explicit Expand hops).
    let cases = [
        (
            "g.V().repeat(__.both()).times(2).dedup().count()",
            "g.V().both().both().dedup().count()",
        ),
        (
            "g.V().repeat(__.out()).times(2).dedup().count()",
            "g.V().out().out().dedup().count()",
        ),
        (
            "g.V().repeat(__.both()).times(3).dedup().count()",
            "g.V().both().both().both().dedup().count()",
        ),
        (
            "g.V().repeat(__.in()).times(2).dedup().count()",
            "g.V().in().in().dedup().count()",
        ),
    ];
    for (fast, enumerated) in cases {
        assert_eq!(count(fast), count(enumerated), "{fast} vs {enumerated}");
    }
    // 1-hop dedup (a single Expand, not a VarLength) also takes the distinct-
    // endpoint fast path. On this ring every node is some node's neighbour AND some
    // node's out-target, so the distinct 1-hop set is ALL 40 — an independent check.
    assert_eq!(count("g.V().both().dedup().count()"), 40);
    assert_eq!(count("g.V().out().dedup().count()"), 40);
}

/// `repeat(out()).times(k).count()` — the WALK degree-algebra count (O(V+E), fired
/// only above the enumeration-cost threshold) must equal the count from ENUMERATING
/// the same walk with explicit `out().out()…` hops. Sized (3000 nodes, deg 4) so the
/// algebra actually fires; self-loops stress the walk's edge-reuse (no subtraction).
#[test]
fn repeat_out_count_algebra_matches_enumeration() {
    let mut b = Builder::default();
    for _ in 0..3000 {
        b.node(&["N"], &[]);
    }
    for i in 0u32..3000 {
        for d in 0u32..4 {
            let ty = if d % 2 == 0 { "R" } else { "F" };
            b.edge(i, (i * 7 + d * 13 + 1) % 3000, ty);
        }
    }
    b.edge(0, 0, "R"); // self-loop (walk may reuse it)
    b.edge(5, 5, "R");
    let st = b.build();
    let count = |q: &str| match &gremlin_rows(q, &st).rows[0][0] {
        Value::Num(x) => *x as i64,
        other => panic!("not a count: {other:?}"),
    };
    let cases = [
        (
            "g.V().repeat(__.out()).times(2).count()",
            "g.V().out().out().count()",
        ),
        (
            "g.V().repeat(__.out()).times(1).count()",
            "g.V().out().count()",
        ),
        (
            "g.V().repeat(__.out('R')).times(2).count()",
            "g.V().out('R').out('R').count()",
        ),
    ];
    for (algebra, enumerated) in cases {
        assert_eq!(
            count(algebra),
            count(enumerated),
            "{algebra} vs {enumerated}"
        );
    }
}

/// `is(P)` filters the VALUE stream by a predicate (like `where`); `is(literal)`
/// is an equality test. Ages are alice 30, bob 25, carol 40.
#[test]
fn gremlin_is_value_predicate() {
    let store = social();
    let gt = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').values('age').is(gt(28))",
        &store,
    ));
    assert_eq!(gt, vec!["Num(30.0);", "Num(40.0);"]);
    // Bare literal → equality.
    let eq = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').values('age').is(25)",
        &store,
    ));
    assert_eq!(eq, vec!["Num(25.0);"]);
}

/// `g.V('id', …)` is a READ source: seed the frontier with exactly the vertices
/// carrying those external ids (dense-id strings here), then traverse as usual.
/// A missing id contributes nothing — like core's `g.V(<absent>)`.
#[test]
fn gremlin_v_by_external_id_read_source() {
    let store = social();
    // Single id → that vertex.
    let one = value_bag(&gremlin_rows("g.V('0').values('name')", &store));
    assert_eq!(one, vec!["Str(\"alice\");"]);
    // Several ids → their union, order-independent.
    let many = value_bag(&gremlin_rows("g.V('1', '2').values('name')", &store));
    assert_eq!(many, vec!["Str(\"bob\");", "Str(\"carol\");"]);
    // Seeds a real frontier: hops compose off it.
    let alice_out = value_bag(&gremlin_rows(
        "g.V('0').out('KNOWS').values('name')",
        &store,
    ));
    assert_eq!(alice_out, vec!["Str(\"bob\");", "Str(\"carol\");"]);
    // A non-existent id yields nothing (no error).
    let gone = value_bag(&gremlin_rows("g.V('999').values('name')", &store));
    assert!(
        gone.is_empty(),
        "missing id must contribute nothing: {gone:?}"
    );
}

/// The 3-arg `has('Label','k',pred)` is a label check AND a property predicate,
/// and `hasLabel` now works anywhere (after a hop, and with multiple labels =
/// ANY of them) via a runtime `label ∈ labels(n)` membership — not just folded
/// into the scan right after `V()`.
#[test]
fn gremlin_has_label_forms() {
    let store = social();
    // has(label, key, pred) == 'Label' IN labels(n) AND n.key = pred.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().has('Person','name','alice').values('name')",
            &store,
        )),
        value_bag(&gql_rows(
            "MATCH (n) WHERE 'Person' IN labels(n) AND n.name='alice' RETURN n.name",
            &store,
        )),
    );
    // hasLabel after a hop (not right after V()): alice's non-Person neighbour.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V('0').out().hasLabel('Project').values('name')",
            &store,
        )),
        vec!["Str(\"graphdb\");"],
    );
    // Multi-label hasLabel matches ANY of the labels (all 4 nodes here).
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person','Project').count()",
            &store
        )),
        vec!["Num(4.0);"],
    );
    // The single-label-after-V() fast path and 2-arg has still work.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Project').values('name')",
            &store
        )),
        vec!["Str(\"graphdb\");"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().has('name','bob').values('name')",
            &store
        )),
        vec!["Str(\"bob\");"],
    );
}

/// `elementMap()` is core's FLAT element map — `{id, label, <props…>}` for a
/// node, plus `IN`/`OUT` endpoint stubs for an edge — with a SINGULAR label and
/// the properties flattened alongside the tokens. `elementMap('k',…)` filters the
/// properties. This is the Gremlin/TinkerPop shape (distinct from the nested
/// `{id, labels, properties}` render used for a bare returned element).
#[test]
fn gremlin_element_map_flat_shape() {
    let store = social();
    // Node: id + singular label + flattened (sorted) present properties.
    assert_eq!(
        value_bag(&gremlin_rows("g.V('0').elementMap()", &store)),
        vec![
            "Map([(Str(\"id\"), Str(\"0\")), (Str(\"label\"), Str(\"Person\")), \
                 (Str(\"age\"), Num(30.0)), (Str(\"name\"), Str(\"alice\"))]);",
        ],
    );
    // A key filter restricts the flattened properties.
    assert_eq!(
        value_bag(&gremlin_rows("g.V('0').elementMap('name')", &store)),
        vec![
            "Map([(Str(\"id\"), Str(\"0\")), (Str(\"label\"), Str(\"Person\")), \
                 (Str(\"name\"), Str(\"alice\"))]);",
        ],
    );
    // Edge: id + type label + IN (destination) / OUT (source) stubs, matching
    // core's element_map_val (IN = e_dst, OUT = e_src). alice(0)→bob(1) KNOWS = e0.
    let edge = value_bag(&gremlin_rows("g.V('0').outE('KNOWS').elementMap()", &store));
    assert!(edge.iter().any(|s| s.contains(
        "(Str(\"id\"), Str(\"e0\")), (Str(\"label\"), Str(\"KNOWS\")), \
             (Str(\"IN\"), Map([(Str(\"id\"), Str(\"1\")), (Str(\"label\"), Str(\"Person\"))])), \
             (Str(\"OUT\"), Map([(Str(\"id\"), Str(\"0\")), (Str(\"label\"), Str(\"Person\"))]))"
    )));
}

/// `coalesce(<hop>, …)` takes the FIRST branch that yields per element (an Exists
/// guard chain over the same Branch reconverge); `choose(<pred>, <thenHop>,
/// <elseHop>)` routes by a predicate; `optional(<hop>)` advances if the hop
/// yields, else keeps the element (OptionalExpand keep_source). All keep a
/// continuable frontier.
#[test]
fn gremlin_coalesce_choose_optional() {
    let store = social();
    // coalesce: WORKS_ON if present (alice→graphdb), else out KNOWS (bob→carol);
    // carol has neither → nothing.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').coalesce(out('WORKS_ON'), out('KNOWS')).values('name')",
            &store,
        )),
        vec!["Str(\"carol\");", "Str(\"graphdb\");"],
    );
    // choose: alice routes to out KNOWS (bob, carol); the others to out WORKS_ON
    // (none, since only alice has it).
    assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').choose(has('name','alice'), out('KNOWS'), out('WORKS_ON')).values('name')",
                &store,
            )),
            vec!["Str(\"bob\");", "Str(\"carol\");"],
        );
    // optional: alice→bob,carol; bob→carol; carol has no out KNOWS → stays carol.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').optional(out('KNOWS')).values('name')",
            &store,
        )),
        vec![
            "Str(\"bob\");",
            "Str(\"carol\");",
            "Str(\"carol\");",
            "Str(\"carol\");",
        ],
    );
    // A missed optional keeps the element (frontier continues).
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V('2').optional(out('KNOWS')).values('name')",
            &store
        )),
        vec!["Str(\"carol\");"],
    );
}

/// `union(<hop>, …)` concatenates each branch's frontier per element and — unlike
/// GQL's materializing UNION — keeps it a node frontier, so the traversal
/// CONTINUES (`.values()`, `.count()`, another hop). This is core's per-traverser
/// branch-and-reconverge, expressed columnar via Plan::Branch over pull_body.
#[test]
fn gremlin_union_of_hops() {
    let store = social();
    // alice's KNOWS targets (bob, carol) unioned with her WORKS_ON target (graphdb).
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V('0').union(out('KNOWS'), out('WORKS_ON')).values('name')",
            &store,
        )),
        vec!["Str(\"bob\");", "Str(\"carol\");", "Str(\"graphdb\");"],
    );
    // The union frontier continues: count() sees all three, values() reads them.
    assert_eq!(
        value_bag(&gremlin_rows("g.V('0').union(out(), in()).count()", &store)),
        vec!["Num(3.0);"],
    );
    // bob: out KNOWS (carol) unioned with in KNOWS (alice).
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V('1').union(out('KNOWS'), in('KNOWS')).values('name')",
            &store,
        )),
        vec!["Str(\"alice\");", "Str(\"carol\");"],
    );
    // Multi-step branch bodies are now supported (arbitrary sub-traversals per
    // branch, not just a single hop): alice's 2-hop KNOWS reach unioned with a
    // value body still parses and runs.
    assert!(super::parse("g.V().union(out().out(), in())").is_ok());
    assert!(super::parse("g.V().union(values('name'), out('KNOWS').values('name'))").is_ok());
}

/// The scope-LOCAL aggregates `count`/`sum`/`mean`/`min`/`max`(local) reduce the
/// current list cell PER ROW (after `fold()`), where the bare/global forms fold
/// the whole stream to one scalar. Over the folded Person ages [30,25,40]: local
/// count 3, sum 95, mean 95/3, min 25, max 40 — and the local sum equals the
/// global sum of the same values.
#[test]
fn gremlin_local_scope_aggregates() {
    let store = social();
    let folded = "g.V().hasLabel('Person').values('age').fold()";
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{folded}.count(local)"), &store)),
        vec!["Num(3.0);"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{folded}.sum(local)"), &store)),
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').values('age').sum()",
            &store
        )),
    );
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{folded}.min(local)"), &store)),
        vec!["Num(25.0);"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{folded}.max(local)"), &store)),
        vec!["Num(40.0);"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{folded}.mean(local)"), &store)),
        vec!["Num(31.666666666666668);"],
    );
}

/// `tail(n)` keeps the LAST n rows of the committed order — the mirror of
/// `limit(n)` (the first n). After `order().by('age')` it is the top-n by age.
#[test]
fn gremlin_tail_is_the_last_n() {
    let store = social();
    let ages = "g.V().hasLabel('Person').order().by('age').values('age')";
    // tail(2) = the two largest ages (the tail of the ascending order); limit(2)
    // = the two smallest — different windows of the same order.
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{ages}.tail(2)"), &store)),
        vec!["Num(30.0);", "Num(40.0);"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{ages}.limit(2)"), &store)),
        vec!["Num(25.0);", "Num(30.0);"],
    );
    // Default n = 1 (the single largest); an oversized n keeps everything.
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{ages}.tail()"), &store)),
        vec!["Num(40.0);"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(&format!("{ages}.tail(99)"), &store)),
        vec!["Num(25.0);", "Num(30.0);", "Num(40.0);"],
    );
}

/// `hasId('a', …)` keeps the element iff its external id is one of the given ids
/// — an `element_id`-in-list filter, verified equal to the GQL `element_id(n) = …`
/// predicate. Works on nodes and edges.
#[test]
fn gremlin_has_id_filters_by_external_id() {
    let store = social();
    assert_eq!(
        value_bag(&gremlin_rows("g.V().hasId('0','1').values('name')", &store)),
        value_bag(&gql_rows(
            "MATCH (n) WHERE element_id(n)='0' OR element_id(n)='1' RETURN n.name",
            &store,
        )),
    );
    // A single id, and an edge id.
    assert_eq!(
        value_bag(&gremlin_rows("g.V().hasId('2').values('name')", &store)),
        vec!["Str(\"carol\");"],
    );
    assert_eq!(
        value_bag(&gremlin_rows("g.E().hasId('e0').count()", &store)),
        vec!["Num(1.0);"],
    );
}

/// `simplePath()` keeps traversers whose vertex path has NO repeat; `cyclicPath()`
/// keeps those that DO — a partition of the stream. They read the lineage node
/// path (like `path()`), so they are scoped to pure vertex-hop chains. A 2-hop
/// `both` walk from a node returns to it on half the paths (the cyclic ones).
#[test]

fn gremlin_simple_and_cyclic_path() {
    let store = social();
    // 2-hop BOTH from alice: [0,1,0] and [0,2,0] return to alice (cyclic); [0,1,2]
    // and [0,2,1] reach a new node (simple).
    let base = "g.V('0').both('KNOWS').both('KNOWS')";
    assert_eq!(
        value_bag(&gremlin_rows(
            &format!("{base}.simplePath().values('name')"),
            &store
        )),
        vec!["Str(\"bob\");", "Str(\"carol\");"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(
            &format!("{base}.cyclicPath().values('name')"),
            &store
        )),
        vec!["Str(\"alice\");", "Str(\"alice\");"],
    );
    // The two are complementary: together they are the whole stream (4 paths).
    let all = value_bag(&gremlin_rows(&format!("{base}.count()"), &store));
    assert_eq!(all, vec!["Num(4.0);"]);
    assert_eq!(
        value_bag(&gremlin_rows(
            &format!("{base}.simplePath().count()"),
            &store
        )),
        vec!["Num(2.0);"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(
            &format!("{base}.cyclicPath().count()"),
            &store
        )),
        vec!["Num(2.0);"],
    );
    // Only over a pure vertex-hop chain — a value stream is deferred.
    assert!(super::parse("g.V().values('name').simplePath()").is_err());
}

/// `and`/`or`/`not` accept navigating hop children too, each a semi-join
/// `Expr::Exists` (the same construction as `where(<hop>)`). So `not(out('L'))`
/// is the ANTI-join (elements without such an edge) and `and(out('L'), has(…))`
/// mixes an edge test with a property test — verified equal to the GQL
/// EXISTS/NOT EXISTS forms.
#[test]
fn gremlin_and_or_not_hop_children() {
    let store = social();
    // not(out(KNOWS)) = the anti-join: vertices WITHOUT an out-KNOWS edge.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().not(out('KNOWS')).values('name')",
            &store
        )),
        value_bag(&gql_rows(
            "MATCH (n) WHERE NOT EXISTS { (n)-[:KNOWS]->() } RETURN n.name",
            &store,
        )),
    );
    // and(<hop>, <property>) mixes a semi-join with a predicate.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().and(out('KNOWS'), has('name','alice')).values('name')",
            &store,
        )),
        value_bag(&gql_rows(
            "MATCH (n) WHERE EXISTS { (n)-[:KNOWS]->() } AND n.name='alice' RETURN n.name",
            &store,
        )),
    );
    // or of two different hops.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().or(out('WORKS_ON'), in('KNOWS')).values('name')",
            &store,
        )),
        value_bag(&gql_rows(
            "MATCH (n) WHERE EXISTS { (n)-[:WORKS_ON]->() } OR EXISTS { (n)<-[:KNOWS]-() } \
                 RETURN n.name",
            &store,
        )),
    );
    // A multi-label hop child is now supported (Exists over a disjunction hop).
    assert!(super::parse("g.V().and(out('A','B'), has('k'))").is_ok());
}

/// `where(<hop>)` is a semi-join: keep the current element iff it HAS such an
/// adjacency. It lowers to an `Expr::Exists` whose body seeds `Plan::Row` and
/// expands from the current slot — the same shape GQL's `EXISTS { … }` builds —
/// so it equals the GQL `WHERE EXISTS { (n)-[:L]->() }` form. `where(P)` (the
/// value-predicate form) is unchanged.
#[test]
fn gremlin_where_hop_semijoin() {
    let store = social();
    // Vertices with an out-KNOWS edge (alice, bob) == the GQL EXISTS form.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().where(out('KNOWS')).values('name')",
            &store
        )),
        value_bag(&gql_rows(
            "MATCH (n) WHERE EXISTS { (n)-[:KNOWS]->() } RETURN n.name",
            &store,
        )),
    );
    // Incoming KNOWS (bob, carol) == the reverse EXISTS form.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().where(in('KNOWS')).values('name')",
            &store
        )),
        value_bag(&gql_rows(
            "MATCH (n) WHERE EXISTS { (n)<-[:KNOWS]-() } RETURN n.name",
            &store,
        )),
    );
    // Argless where(out()) is any out-edge; both() is either direction.
    assert_eq!(
        value_bag(&gremlin_rows("g.V().where(out()).values('name')", &store)),
        vec!["Str(\"alice\");", "Str(\"bob\");"],
    );
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().where(both('KNOWS')).values('name')",
            &store
        )),
        vec!["Str(\"alice\");", "Str(\"bob\");", "Str(\"carol\");"],
    );
    // The value-predicate form still works (age > 28 → carol, alice).
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').values('age').where(gt(28))",
            &store,
        )),
        vec!["Num(30.0);", "Num(40.0);"],
    );
    // A multi-label where-hop is a semi-join over either edge type.
    assert!(super::parse("g.V().where(out('A','B'))").is_ok());
}

/// `and(f1,f2,…)`, `or(f1,f2,…)`, `not(f)` combine element filters (has/hasNot,
/// nested and/or/not) into one predicate over the current element — the direct
/// Gremlin spelling of a boolean `WHERE`. Verified equal to the equivalent GQL
/// `WHERE … AND/OR/NOT …`. Navigating child traversals (semi-joins) are deferred.
#[test]
fn gremlin_and_or_not_filter_combinators() {
    let store = social();
    // and: two conjoined predicates == GQL AND.
    assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').and(has('age', gt(28)), has('name', neq('carol'))).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n:Person) WHERE n.age > 28 AND n.name <> 'carol' RETURN n.name",
                &store,
            )),
        );
    // or: disjunction == GQL OR.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').or(has('age', lt(28)), has('name', 'carol')).values('name')",
            &store,
        )),
        value_bag(&gql_rows(
            "MATCH (n:Person) WHERE n.age < 28 OR n.name = 'carol' RETURN n.name",
            &store,
        )),
    );
    // not: negation == GQL NOT.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').not(has('age', gt(28))).values('name')",
            &store,
        )),
        value_bag(&gql_rows(
            "MATCH (n:Person) WHERE NOT (n.age > 28) RETURN n.name",
            &store,
        )),
    );
    // Nested and/or compose (an or inside an and) == the GQL parenthesized form.
    assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').and(has('age', gt(20)), or(has('name','bob'), has('name','carol'))).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n:Person) WHERE n.age > 20 AND (n.name = 'bob' OR n.name = 'carol') RETURN n.name",
                &store,
            )),
        );
    // Navigating child traversals are now semi-joins (see
    // `gremlin_and_or_not_hop_children`); they parse rather than error.
    assert!(super::parse("g.V().and(out('KNOWS'), has('age', gt(1)))").is_ok());
    assert!(super::parse("g.V().not(out('KNOWS'))").is_ok());
}

/// `group().by(key).by(value)` is a grouped aggregation. Core shapes it as one
/// {key: value} map; the engine represents a grouped result as ROWS of (key,
/// value), consistent with `groupCount()`. `by(count())` reduces to a count (so
/// `group().by('k').by(count())` == `groupCount().by('k')`); a property/element
/// value folds the group (collect, Gremlin fold), elements folded as their ids.
#[test]
fn gremlin_group_by_key_and_value() {
    let store = social();
    // by(count()) reduces — identical to groupCount().by('k').
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').out('KNOWS').group().by('name').by(count())",
            &store,
        )),
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').out('KNOWS').groupCount().by('name')",
            &store,
        )),
    );
    // Default value-by folds the group's ELEMENTS, each rendered as its element
    // map (like a top-level vertex, so a folded vertex canonicalizes the same way).
    // Names are unique here, so each group holds one element: alice(0), bob(1),
    // carol(2).
    assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').group().by('name')",
                &store
            )),
            // group() folds to ONE Gremlin Map {name: [elements]} (first-seen key
            // order), matching core.
            vec![
                "Map([(Str(\"alice\"), List([Map([(Str(\"id\"), Str(\"0\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(30.0)), (Str(\"name\"), Str(\"alice\"))]))])])), (Str(\"bob\"), List([Map([(Str(\"id\"), Str(\"1\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(25.0)), (Str(\"name\"), Str(\"bob\"))]))])])), (Str(\"carol\"), List([Map([(Str(\"id\"), Str(\"2\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(40.0)), (Str(\"name\"), Str(\"carol\"))]))])]))]);",
            ],
        );
    // A property value-by folds that property per group.
    assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').group().by('name').by('age')",
                &store,
            )),
            vec![
                "Map([(Str(\"alice\"), List([Num(30.0)])), (Str(\"bob\"), List([Num(25.0)])), (Str(\"carol\"), List([Num(40.0)]))]);",
            ],
        );
    // A <hop>.count() value-by is the group's total degree (a Sum of per-element
    // degrees).
    assert!(super::parse("g.V().group().by('name').by(out().count())").is_ok());
}

/// `project('a','b').by(x).by(y)` builds one insertion-ordered Map per traverser:
/// key i takes the i-th `by` modulator, or the current element when there is no
/// i-th `by` (core's `bys.get(i)`, not cycled). `by('key')` reads a property; a
/// key with no `by` yields the element as its id, consistent with `select()`.
#[test]
fn gremlin_project_by_modulators() {
    let store = social();
    // Two by-modulators → {n: name, a: age}, keys in project order.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').project('n','a').by('name').by('age')",
            &store,
        )),
        vec![
            "Map([(Str(\"n\"), Str(\"alice\")), (Str(\"a\"), Num(30.0))]);",
            "Map([(Str(\"n\"), Str(\"bob\")), (Str(\"a\"), Num(25.0))]);",
            "Map([(Str(\"n\"), Str(\"carol\")), (Str(\"a\"), Num(40.0))]);",
        ],
    );
    // A key with fewer bys than keys defaults to the current element, rendered as
    // its element map (like a top-level vertex): project('n','self').by('name') →
    // {n: name, self: {id,labels,properties}}.
    assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().has('name','bob').project('n','self').by('name')",
                &store,
            )),
            vec![
                "Map([(Str(\"n\"), Str(\"bob\")), (Str(\"self\"), Map([(Str(\"id\"), Str(\"1\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(25.0)), (Str(\"name\"), Str(\"bob\"))]))]))]);"
            ],
        );
    // A degree `<hop>.count()` by-modulator is a correlated count subquery.
    assert!(super::parse("g.V().project('c').by(out().count())").is_ok());
}

/// `path()` over a pure vertex-hop chain yields the sequence of vertices visited,
/// each rendered as its element map (not a bare id). Verified structurally: the
/// path elements ARE node maps whose id sequence is the hop sequence; and every
/// non-vertex-hop shape (edge steps, the E source, value projections, var-length)
/// is deferred rather than mis-answered.
#[test]
fn gremlin_path_vertex_hop_chain() {
    let store = social();
    // Pull the "id" of each node-map element, per path, as a sorted set of seqs.
    fn id_seqs(q: &str, store: &Store) -> Vec<Vec<String>> {
        let rows = gremlin_rows(q, store);
        let mut out: Vec<Vec<String>> = rows
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::List(elems) => elems
                    .iter()
                    .map(|e| match e {
                        Value::Map(pairs) => pairs
                            .iter()
                            .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "id"))
                            .and_then(|(_, v)| match v {
                                Value::Str(s) => Some(s.to_string()),
                                _ => None,
                            })
                            .expect("path element is a node map with an id"),
                        o => panic!("path element not a node map: {o:?}"),
                    })
                    .collect(),
                o => panic!("path is not a list: {o:?}"),
            })
            .collect();
        out.sort();
        out
    }
    // Single vertex → a one-element path.
    assert_eq!(
        id_seqs("g.V('0').path()", &store),
        vec![vec!["0".to_string()]]
    );
    // One hop from alice(0) → [0,1] and [0,2].
    assert_eq!(
        id_seqs("g.V('0').out('KNOWS').path()", &store),
        vec![
            vec!["0".to_string(), "1".to_string()],
            vec!["0".to_string(), "2".to_string()],
        ],
    );
    // Two hops: alice->bob->carol is the only length-2 KNOWS walk from alice.
    assert_eq!(
        id_seqs("g.V('0').out('KNOWS').out('KNOWS').path()", &store),
        vec![vec!["0".to_string(), "1".to_string(), "2".to_string()]],
    );
    // An interleaved node/edge path (`outE().inV()`) works.
    assert!(super::parse("g.V().outE().inV().path()").is_ok());
    // The full per-step history (PathRecord) now covers what was once deferred: an `E()`
    // source (`[e]` per edge) and a value projection before `path()` (`[v, 'name']`).
    assert!(super::parse("g.E().path()").is_ok());
    assert!(super::parse("g.V().values('name').path()").is_ok());
    // A `repeat(<vertex-hop>)` walk records full path lineage, so path() over it
    // parses (the VarLength endpoints carry their node chains).
    assert!(super::parse("g.V().repeat(out('KNOWS')).times(2).path()").is_ok());
}

/// `valueMap()` projects a PROPERTIES-only map (no id/label tokens) with scalar
/// values; `valueMap('k',…)` filters keys. Present-properties only — the Project
/// node has no `age`, so its map omits it. The maps equal the `properties`
/// sub-map of the engine's GQL element render, which is byte-identical to core.
#[test]
fn gremlin_valuemap_properties_only() {
    let store = social();
    // All properties (keys sorted): the three Persons carry name+age, the
    // Project only name.
    assert_eq!(
        value_bag(&gremlin_rows("g.V().valueMap()", &store)),
        vec![
            "Map([(Str(\"age\"), Num(25.0)), (Str(\"name\"), Str(\"bob\"))]);",
            "Map([(Str(\"age\"), Num(30.0)), (Str(\"name\"), Str(\"alice\"))]);",
            "Map([(Str(\"age\"), Num(40.0)), (Str(\"name\"), Str(\"carol\"))]);",
            "Map([(Str(\"name\"), Str(\"graphdb\"))]);",
        ],
    );
    // Key filter: only the named property, when present.
    assert_eq!(
        value_bag(&gremlin_rows("g.V().valueMap('name')", &store)),
        vec![
            "Map([(Str(\"name\"), Str(\"alice\"))]);",
            "Map([(Str(\"name\"), Str(\"bob\"))]);",
            "Map([(Str(\"name\"), Str(\"carol\"))]);",
            "Map([(Str(\"name\"), Str(\"graphdb\"))]);",
        ],
    );
    // Filtering an absent key drops it: the Project keeps only name under
    // valueMap('name','age').
    assert_eq!(
        value_bag(&gremlin_rows("g.V().valueMap('name','age')", &store)),
        value_bag(&gremlin_rows("g.V().valueMap()", &store)),
    );
}

/// `id()` projects the element's preserved external id (via `element_id`), and
/// `label()` a single label string — a vertex's label or an edge's type (via
/// `element_label`), both polymorphic over the current node/edge slot. Verified
/// vs the engine's own GQL `element_id`/`type` and vs the fixture's known labels.
#[test]
fn gremlin_id_and_label_accessors() {
    let store = social();
    // id() == element_id over the same elements.
    assert_eq!(
        value_bag(&gremlin_rows("g.V().id()", &store)),
        value_bag(&gql_rows("MATCH (n) RETURN element_id(n)", &store)),
    );
    assert_eq!(
        value_bag(&gremlin_rows("g.V().outE().id()", &store)),
        value_bag(&gql_rows("MATCH ()-[r]->() RETURN element_id(r)", &store)),
    );
    // Vertex label() == its single label (Person x3, Project x1).
    assert_eq!(
        value_bag(&gremlin_rows("g.V().label().dedup()", &store)),
        vec!["Str(\"Person\");", "Str(\"Project\");"],
    );
    // Edge label() == edge type, matching GQL type().
    assert_eq!(
        value_bag(&gremlin_rows("g.V().outE().label()", &store)),
        value_bag(&gql_rows("MATCH ()-[r]->() RETURN type(r)", &store)),
    );
}

/// `repeat(<hop>).times(n)` applies a single anonymous hop exactly n times — a
/// walk of length n. Verified against the equivalent chain of n plain hops (both
/// are walks, so this exercises `VarLength{min:n,max:n,trail:false}` end to end);
/// GQL var-length is a TRAIL, so the chained-hop equivalent is the right oracle.
#[test]
fn gremlin_repeat_times_fixed_length_walk() {
    let store = social();
    // times(n) == n chained hops, for out and both, by rows and by count.
    for (repeat_form, chain_form) in [
        (
            "g.V().repeat(out('KNOWS')).times(1).values('name')",
            "g.V().out('KNOWS').values('name')",
        ),
        (
            "g.V().repeat(out('KNOWS')).times(2).values('name')",
            "g.V().out('KNOWS').out('KNOWS').values('name')",
        ),
        (
            "g.V().repeat(__.out('KNOWS')).times(2).values('name')",
            "g.V().out('KNOWS').out('KNOWS').values('name')",
        ),
        (
            "g.V().repeat(both('KNOWS')).times(2).count()",
            "g.V().both('KNOWS').both('KNOWS').count()",
        ),
    ] {
        assert_eq!(
            value_bag(&gremlin_rows(repeat_form, &store)),
            value_bag(&gremlin_rows(chain_form, &store)),
            "{repeat_form} must equal {chain_form}",
        );
    }
    // Deferred / malformed forms error rather than silently mis-answer.
    assert!(super::parse("g.V().repeat(out('KNOWS'))").is_err()); // bare repeat = unbounded
    assert!(super::parse("g.V().repeat(out('KNOWS')).values('name')").is_err()); // not closed by times
    assert!(super::parse("g.V().times(2)").is_err()); // times without repeat
    assert!(super::parse("g.V().repeat(out('A').out('B')).times(2)").is_err());
    // multi-step body
}

/// `outE`/`inE`/`bothE` bind the traversed edge (current element becomes the
/// edge), and the canonical vertex move steps back onto the endpoint the hop
/// landed: `outE().inV()` == `out()`, `inE().outV()` == `in()`, `*E().otherV()`
/// == the corresponding both/out/in. Verified against the plain hops (same IR),
/// and `outE().count()` == `g.E().count()` (every edge is one node's out-edge).
#[test]
fn gremlin_edge_hops_and_endpoint_moves() {
    let store = social();
    // The edge frontier: outE over all vertices touches every edge once.
    assert_eq!(
        value_bag(&gremlin_rows("g.V().outE().count()", &store)),
        value_bag(&gremlin_rows("g.E().count()", &store)),
    );
    // Canonical edge-step + vertex-move pairs equal the plain hops.
    for (edge_form, hop_form) in [
        (
            "g.V().outE('KNOWS').inV().values('name')",
            "g.V().out('KNOWS').values('name')",
        ),
        (
            "g.V().inE('KNOWS').outV().values('name')",
            "g.V().in('KNOWS').values('name')",
        ),
        (
            "g.V().bothE('KNOWS').otherV().values('name')",
            "g.V().both('KNOWS').values('name')",
        ),
    ] {
        assert_eq!(
            value_bag(&gremlin_rows(edge_form, &store)),
            value_bag(&gremlin_rows(hop_form, &store)),
            "{edge_form} must equal {hop_form}",
        );
    }
    // Origin-returning combinations read the endpoint straight off the edge.
    assert!(super::parse("g.V().outE().outV().values('name')").is_ok());
    assert!(super::parse("g.V().inE().inV().values('name')").is_ok());
    assert!(super::parse("g.V().bothE().inV().values('name')").is_ok());
    // A vertex move with no edge frontier is still an error.
    assert!(super::parse("g.V().inV()").is_err());
}

/// `g.E()` is an all-edges READ source: it seeds the frontier with every live
/// edge (`social()` has 4: three KNOWS + one WORKS_ON). Cross-checked against the
/// engine's own GQL front-end — the anonymous directed pattern `()-[r]->()` — so
/// both lowerings of "every edge" agree; the GQL side is itself proven vs core by
/// the differential fuzzer. Counting through g.E() exercises the Col::Edges
/// frontier end to end.
#[test]
fn gremlin_e_all_edges_read_source() {
    let store = social();
    let ge = value_bag(&gremlin_rows("g.E().count()", &store));
    assert_eq!(ge, vec!["Num(4.0);"]);
    // Same "count every edge" via GQL's directed anonymous pattern.
    let gql = value_bag(&gql_rows("MATCH ()-[r]->() RETURN count(r)", &store));
    assert_eq!(ge, gql);
    // g.E('id') (edges by external id) seeds the named edge.
    assert!(super::parse("g.E('e0')").is_ok());
}

/// `has(k)` filters elements that CARRY property `k`; `hasNot(k)` those that
/// don't — matching core. Only the `Project` node (graphdb) lacks `age`.
#[test]
fn gremlin_has_key_existence_and_hasnot() {
    let store = social();
    let has_age = value_bag(&gremlin_rows("g.V().has('age').values('name')", &store));
    assert_eq!(
        has_age,
        vec!["Str(\"alice\");", "Str(\"bob\");", "Str(\"carol\");"]
    );
    let no_age = value_bag(&gremlin_rows("g.V().hasNot('age').values('name')", &store));
    assert_eq!(no_age, vec!["Str(\"graphdb\");"]);
    // has(k, pred) — the value-predicate form — still works.
    assert!(super::parse("g.V().has('age', gt(28)).values('name')").is_ok());
}

/// Argless `out()`/`in()`/`both()` traverse edges of ANY type (matching core),
/// where a labelled hop is narrower — alice's WORKS_ON target only shows up
/// through the untyped hop.
#[test]
fn gremlin_argless_out_traverses_all_edge_types() {
    let store = social();
    let all = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').out().values('name')",
        &store,
    ));
    assert!(
        all.iter().any(|r| r.contains("graphdb")),
        "argless out() must follow WORKS_ON too: {all:?}"
    );
    let knows = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').out('KNOWS').values('name')",
        &store,
    ));
    assert!(
        !knows.iter().any(|r| r.contains("graphdb")),
        "typed out('KNOWS') must not: {knows:?}"
    );
    // in()/both() accept the argless form too.
    assert!(super::parse("g.V().hasLabel('Person').in().values('name')").is_ok());
    assert!(super::parse("g.V().hasLabel('Person').both().values('name')").is_ok());
}

/// THE PAYOFF: equivalent GQL and Gremlin queries lower to plans producing
/// the same rows. Both are thin front-ends over one neutral IR.
#[test]
fn gql_and_gremlin_agree() {
    let store = social();
    let pairs = [
        (
            "MATCH (p:Person) RETURN p.name",
            "g.V().hasLabel('Person').values('name')",
        ),
        (
            "MATCH (p:Person) WHERE p.age > 28 RETURN p.name",
            "g.V().hasLabel('Person').has('age', P.gt(28)).values('name')",
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name",
            "g.V().hasLabel('Person').out('KNOWS').values('name')",
        ),
        (
            "MATCH (p:Person) RETURN count(*) AS c",
            "g.V().hasLabel('Person').count()",
        ),
        (
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN DISTINCT b.name",
            "g.V().hasLabel('Person').out('KNOWS').values('name').dedup()",
        ),
        // NOTE: GQL GROUP BY and Gremlin groupCount() do NOT agree by design —
        // GQL yields relational (key, count) ROWS, Gremlin yields a single
        // {key: count} Map (see `bare_group_count_groups_by_the_current_element`).
        // So no groupCount pair belongs in this row-equality list.
    ];
    for (gql, gremlin) in pairs {
        assert_eq!(
            value_bag(&gql_rows(gql, &store)),
            value_bag(&gremlin_rows(gremlin, &store)),
            "GQL `{gql}` and Gremlin `{gremlin}` disagree",
        );
    }
}

/// order().by(...).values(...) sorts elements, then projects — hand-checked.
#[test]
fn order_by_then_values() {
    let store = social();
    let out = gremlin_rows(
        "g.V().hasLabel('Person').order().by('age', desc).values('name')",
        &store,
    );
    let names: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(names, vec!["carol", "alice", "bob"]); // 40, 30, 25
}

/// range(lo, hi) is a paging window.
#[test]
fn range_is_a_window() {
    let store = social();
    let out = gremlin_rows(
        "g.V().hasLabel('Person').order().by('age').range(1, 2).values('name')",
        &store,
    );
    // ages asc: bob(25), alice(30), carol(40); range(1,2) -> alice
    assert_eq!(out.rows.len(), 1);
    match &out.rows[0][0] {
        Value::Str(x) => assert_eq!(&**x, "alice"),
        _ => panic!(),
    }
}

/// The single list cell of a folded result: exactly one row, one column,
/// which is a `Value::List`. Returned as debug strings, since `Value` has no
/// `PartialEq` (equality is the value contract's, not derived) — the exact,
/// ordered element sequence is what an `order(local)` test needs to pin.
fn fold_list(out: &Rows) -> Vec<String> {
    assert_eq!(out.rows.len(), 1, "fold emits exactly one row");
    assert_eq!(out.rows[0].len(), 1, "fold emits one column");
    match &out.rows[0][0] {
        Value::List(items) => items.iter().map(|v| format!("{v:?}")).collect(),
        other => panic!("expected a list cell, got {other:?}"),
    }
}

/// The same debug-string projection for an expected element list.
fn dbg(items: &[Value]) -> Vec<String> {
    items.iter().map(|v| format!("{v:?}")).collect()
}

/// fold() collects the whole value stream into one list. Order is unspecified
/// without a preceding sort, so the SET of names is what's pinned here.
#[test]
fn fold_collects_the_stream() {
    let store = social();
    let out = gremlin_rows("g.V().hasLabel('Person').values('name').fold()", &store);
    let mut got = fold_list(&out);
    got.sort();
    let mut want = dbg(&[s("alice"), s("bob"), s("carol")]);
    want.sort();
    // Person names are alice/bob/carol; graphdb is a Project, excluded.
    assert_eq!(got, want);
}

/// fold() over an empty stream still emits exactly one row: the empty list.
#[test]
fn fold_of_empty_is_one_empty_list() {
    let store = social();
    let out = gremlin_rows("g.V().hasLabel('Nope').values('name').fold()", &store);
    assert_eq!(fold_list(&out), Vec::<String>::new());
}

/// order(local) sorts WITHIN the folded list — ascending by the value contract.
#[test]
fn order_local_sorts_the_list_ascending() {
    let store = social();
    let out = gremlin_rows(
        "g.V().hasLabel('Person').values('name').fold().order(local)",
        &store,
    );
    // names sorted ascending, exact order (not a set)
    assert_eq!(fold_list(&out), dbg(&[s("alice"), s("bob"), s("carol")]));
}

/// order(local).by(desc) reverses the within-list order; numeric elements sort
/// numerically (the value contract), not lexically.
#[test]
fn order_local_by_desc_on_numbers() {
    let store = social();
    let out = gremlin_rows(
        "g.V().hasLabel('Person').values('age').fold().order(local).by(desc)",
        &store,
    );
    // ages 25/30/40 descending
    assert_eq!(fold_list(&out), dbg(&[n(40.0), n(30.0), n(25.0)]));
}

/// `Scope.local` is an accepted spelling of the local scope.
#[test]
fn order_scope_local_spelling() {
    let store = social();
    let out = gremlin_rows(
        "g.V().hasLabel('Person').values('name').fold().order(Scope.local)",
        &store,
    );
    assert_eq!(fold_list(&out), dbg(&[s("alice"), s("bob"), s("carol")]));
}

/// order(local) faults nothing on a scalar cell — it passes through unchanged
/// (there is no list to sort), so a global order() is still the stream sort.
#[test]
fn order_local_passthrough_on_scalar() {
    let store = social();
    // No fold: each row's slot-0 is a scalar name; order(local) leaves it be.
    let out = gremlin_rows(
        "g.V().hasLabel('Person').values('name').order(local)",
        &store,
    );
    let mut got: Vec<String> = out
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(x) => x.to_string(),
            other => panic!("{other:?}"),
        })
        .collect();
    got.sort();
    assert_eq!(got, vec!["alice", "bob", "carol"]);
}

#[test]
fn errors_not_panics() {
    assert!(super::parse("g.V(").is_err());
    assert!(super::parse("g.E().values('x')").is_ok()); // g.E() is an all-edges source now
    assert!(super::parse("g.E('e0')").is_ok()); // edges by external id
    assert!(super::parse("g.V().frobnicate()").is_err()); // unknown step
    assert!(super::parse("g.V().has('k')").is_ok()); // has(k) is key-existence now
    assert!(super::parse("g.V().has()").is_err()); // has still needs a key
}

// --- writes: addV / property / drop ---

fn exec(q: &str, store: &mut Store) {
    crate::exec::execute(&super::parse(q).unwrap(), store).unwrap();
}

/// `g.addV('L').property(...)` creates one vertex with those properties.
#[test]
fn add_vertex_with_properties() {
    let mut st = Builder::default().build();
    exec(
        "g.addV('Person').property('name', 'x').property('age', 1)",
        &mut st,
    );
    assert_eq!(st.node_count(), 1);
    assert_eq!(st.nodes_with_label("Person"), &[0]);
    assert!(matches!(st.prop(0, "name"), Value::Str(v) if &*v == "x"));
    assert!(matches!(st.prop(0, "age"), Value::Num(v) if v == 1.0));
}

/// `g.V()...property(k, v)` sets the property on the matched vertices only.
#[test]
fn property_step_sets_matched() {
    let mut st = social();
    exec(
        "g.V().hasLabel('Person').has('name', 'alice').property('age', 99)",
        &mut st,
    );
    assert!(matches!(st.prop(0, "age"), Value::Num(v) if v == 99.0)); // alice
    assert!(matches!(st.prop(1, "age"), Value::Num(v) if v == 25.0)); // bob unchanged
}

/// `g.V()...drop()` deletes the matched vertices.
#[test]
fn drop_step_deletes_matched() {
    let mut st = social();
    exec(
        "g.V().hasLabel('Person').has('name', 'bob').drop()",
        &mut st,
    );
    assert!(!st.is_alive(1)); // bob
    assert_eq!(st.nodes_with_label("Person"), &[0, 2]); // alice, carol
}

/// Cross-language agreement: a Gremlin `addV` and the equivalent GQL `INSERT`
/// produce the same graph.
#[test]
fn gremlin_and_gql_writes_agree() {
    let mut g1 = Builder::default().build();
    let mut g2 = Builder::default().build();
    exec("g.addV('P').property('name', 'z')", &mut g1);
    crate::exec::execute(
        &crate::gql::parse("INSERT (:P {name: 'z'})").unwrap(),
        &mut g2,
    )
    .unwrap();
    let probe = crate::gql::parse("MATCH (p:P) RETURN p.name AS n").unwrap();
    assert_eq!(value_bag(&run(&probe, &g1)), value_bag(&run(&probe, &g2)));
}

#[test]
fn write_step_errors() {
    assert!(super::parse("g.addE('R')").is_err()); // no from/to
    assert!(super::parse("g.V().drop().count()").is_err()); // read after write
    assert!(super::parse("g.addV('P').out('R')").is_err()); // read after write
}

// --- addE (B6) ---

/// `g.V(a).addE('T').to(V(b)).property(...)` creates one edge with props.
#[test]
fn add_edge_anchored() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[]);
    let b = st.add_node(&["P"], &[]);
    exec("g.V(0).addE('R').to(V(1)).property('weight', 0.5)", &mut st);
    assert_eq!(st.out(a).len(), 1);
    assert_eq!(st.out(a)[0].nbr, b);
    let eid = st.out(a)[0].eid;
    assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 0.5));
}

/// `g.addE('T').from(V(a)).to(V(b))` is the unanchored form.
#[test]
fn add_edge_from_to() {
    let mut st = Builder::default().build();
    st.add_node(&["P"], &[]);
    st.add_node(&["P"], &[]);
    exec("g.addE('R').from(V(0)).to(V(1))", &mut st);
    assert_eq!(st.out(0).len(), 1);
    assert_eq!(st.out(0)[0].nbr, 1);
}

#[test]
fn add_edge_errors() {
    // Missing `to` is a parse error (finish_add_edge requires both endpoints).
    assert!(super::parse("g.addE('R').from(V(0))").is_err());
    // Out-of-range endpoint is a runtime error.
    let mut st = Builder::default().build();
    st.add_node(&["P"], &[]);
    assert!(
        crate::exec::execute(&super::parse("g.V(0).addE('R').to(V(9))").unwrap(), &mut st).is_err()
    );
}

#[test]
fn local_count_is_per_element() {
    let store = social();
    // local(out('KNOWS').count()) is the per-vertex out-degree, keeping vertices
    // with zero (unlike a global count). alice→2, bob→1, carol→0, +Project has 0.
    let counts = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').local(out('KNOWS').count())",
        &store,
    ));
    // social() has 3 persons; every one contributes a count (0 kept). KNOWS is
    // alice→bob, alice→carol, bob→carol → out-degrees carol=0, bob=1, alice=2.
    assert_eq!(
        counts,
        vec!["Num(0.0);", "Num(1.0);", "Num(2.0);"],
        "one count per person, zeros kept",
    );
    // A non-reducing hop chain is applied per element (local is transparent to it).
    assert!(super::parse("g.V().local(out('KNOWS').values('name'))").is_ok());
}

/// `project(...)` rows are Maps; `order().by(select('k'))` sorts by the entry `k`
/// (a sub-traversal that reads the Map), and the trailing `select('name')` projects
/// the entry from the Map via the tag-fallback. Byte-identical to core's Scoping.
#[test]
fn order_by_select_sorts_project_rows_and_select_reads_the_map_entry() {
    let store = social();
    let rows = gremlin_rows(
        "g.V().hasLabel('Person').project('name','age').by('name').by('age')\
             .order().by(select('age'), desc).select('name')",
        &store,
    );
    let names: Vec<String> = rows
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(s) => s.to_string(),
            other => panic!("expected a name string, got {other:?}"),
        })
        .collect();
    // social() ages: alice 30, bob 25, carol 40 → desc by age.
    assert_eq!(names, vec!["carol", "alice", "bob"]);
}

/// `select('k')` on a Map traverser (a `project()` row) with no step labelled `k`
/// projects the entry rather than dropping every row (core's Scoping fallback).
#[test]
fn select_key_falls_back_to_the_map_entry() {
    let store = social();
    let got = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').project('name','age').by('name').by('age').select('name')",
        &store,
    ));
    assert_eq!(
        got,
        vec!["Str(\"alice\");", "Str(\"bob\");", "Str(\"carol\");"]
    );
}

/// `property(key, <traversal>)` accepts a traversal-induced value: `constant(v)`
/// lowers to a literal SET, a degree sub-traversal `outE().count()` to a
/// `CountSubquery` SET (evaluated per element at write time — see `exec`'s SET path).
#[test]
fn property_value_accepts_constant_and_degree_traversal() {
    use crate::ir::{Expr, Plan, SetOp};
    let set_value = |q: &str| -> (String, Expr) {
        // A terminal property() is an UpdateReturn now (it EMITS the mutated
        // element); the write ops are the same.
        match super::parse(q).unwrap() {
            Plan::Update { ops, .. } | Plan::UpdateReturn { ops, .. } => {
                match ops.into_iter().next().unwrap() {
                    SetOp::Set { key, value, .. } => (key, value),
                    other => panic!("expected SetOp::Set, got {other:?}"),
                }
            }
            other => panic!("expected Plan::Update(Return), got {other:?}"),
        }
    };
    let (k, v) = set_value("g.V().hasLabel('Person').property('flag', constant(1.0))");
    assert_eq!(k, "flag");
    assert!(matches!(v, Expr::Lit(Value::Num(n)) if n == 1.0));
    let (k, v) = set_value("g.V().property('deg', outE().count())");
    assert_eq!(k, "deg");
    assert!(matches!(v, Expr::CountSubquery { .. }));
}

/// Read-after-write (an `UpdateReturn`): TinkerPop `property(k, v).values(k)` and GQL
/// `MATCH … SET … RETURN` apply the write, then read the just-written value over the
/// SAME frontier. constant() and a degree `count()` both flow through.
#[test]
fn read_after_write_reads_the_written_values() {
    let nd = "{\"type\":\"node\",\"id\":\"marko\",\"labels\":[\"P\"],\"properties\":{\"id\":\"marko\"}}\n\
                  {\"type\":\"node\",\"id\":\"a\",\"labels\":[\"P\"],\"properties\":{\"id\":\"a\"}}\n\
                  {\"type\":\"node\",\"id\":\"b\",\"labels\":[\"P\"],\"properties\":{\"id\":\"b\"}}\n\
                  {\"type\":\"edge\",\"id\":\"e1\",\"from\":\"marko\",\"to\":\"a\",\"labels\":[\"KNOWS\"],\"properties\":{}}\n\
                  {\"type\":\"edge\",\"id\":\"e2\",\"from\":\"marko\",\"to\":\"b\",\"labels\":[\"KNOWS\"],\"properties\":{}}";
    // Gremlin `property(k, constant(v)).values(k)` — the write, then the read of it.
    let mut st = crate::ndjson::from_ndjson(nd).unwrap();
    let flags = crate::exec::execute(
        &super::parse("g.V().hasLabel('P').property('flag', constant(1.0)).values('flag')")
            .unwrap(),
        &mut st,
    )
    .unwrap();
    assert_eq!(
        value_bag(&flags),
        vec!["Num(1.0);", "Num(1.0);", "Num(1.0);"]
    );
    // Gremlin `property(k, outE().count()).values(k)` — a traversal-induced out-degree.
    let deg = crate::exec::execute(
        &super::parse("g.V().has('id', eq('marko')).property('deg', outE().count()).values('deg')")
            .unwrap(),
        &mut st,
    )
    .unwrap();
    assert_eq!(value_bag(&deg), vec!["Num(2.0);"]);
    // GQL `MATCH … SET … RETURN` reads the mutated value over the same binding.
    let mut st2 = crate::ndjson::from_ndjson(nd).unwrap();
    let gql = crate::exec::execute(
        &crate::gql::parse("MATCH (n:P) SET n.x = 7 RETURN n.x AS v").unwrap(),
        &mut st2,
    )
    .unwrap();
    assert_eq!(value_bag(&gql), vec!["Num(7.0);", "Num(7.0);", "Num(7.0);"]);
}

#[test]
fn olap_annotate_steps_attach_a_readable_property() {
    let store = social();
    // pageRank(): every vertex gets a numeric score under the default property.
    let pr = gremlin_rows(
        "g.V().pageRank().values('gremlin.pageRankVertexProgram.pageRank')",
        &store,
    );
    assert_eq!(pr.rows.len(), 4, "one score per person");
    assert!(
        pr.rows.iter().all(|r| matches!(r[0], Value::Num(_))),
        "pageRank values are numbers: {:?}",
        pr.rows
    );
    // connectedComponent(): the whole social graph is one component → one id.
    let cc = value_bag(&gremlin_rows(
            "g.V().connectedComponent().values('gremlin.connectedComponentVertexProgram.component').dedup()",
            &store,
        ));
    assert_eq!(cc.len(), 1, "one component id (all connected): {cc:?}");
    // The component id is an external-id STRING (the root vertex), like core.
    assert!(
        cc[0].starts_with("Str("),
        "component id is a string: {cc:?}"
    );
    // A non-algo property still reads the store after an annotate (pass-through).
    assert_eq!(
        value_bag(&gremlin_rows("g.V().pageRank().values('name')", &store)),
        value_bag(&gremlin_rows("g.V().values('name')", &store)),
    );
}

#[test]
fn aggregate_store_cap_side_effect_bag() {
    let store = social();
    // aggregate('x') fills a bag with the value stream; cap('x') reveals it as one
    // list. store is an alias in this eager executor.
    let agg = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').values('name').aggregate('x').cap('x')",
        &store,
    ));
    let sto = value_bag(&gremlin_rows(
        "g.V().hasLabel('Person').values('name').store('x').cap('x')",
        &store,
    ));
    assert_eq!(agg, sto, "aggregate and store are interchangeable");
    assert_eq!(agg.len(), 1, "cap yields exactly one row (a list)");
    // aggregate() alone is a pass-through side effect (no effect on results).
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V('0').out('KNOWS').aggregate('x').values('name')",
            &store,
        )),
        value_bag(&gremlin_rows(
            "g.V('0').out('KNOWS').values('name')",
            &store
        )),
    );
    // cap of an unfilled key yields a single EMPTY list (core), not an error.
    assert_eq!(
        value_bag(&gremlin_rows("g.V('1').cap('nope')", &store)),
        vec!["List([]);"],
    );
    // barrier() and identity() are pass-throughs.
    for step in ["barrier", "identity"] {
        assert_eq!(
            value_bag(&gremlin_rows(
                &format!("g.V().hasLabel('Person').{step}().values('name')"),
                &store,
            )),
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').values('name')",
                &store
            )),
        );
    }
}

#[test]
fn value_aggregates_fold_the_stream() {
    let store = social();
    // Person ages are 30, 25, 40.
    for (step, want) in [("max", 40.0), ("min", 25.0), ("sum", 95.0)] {
        let q = format!("g.V().hasLabel('Person').values('age').{step}()");
        let out = gremlin_rows(&q, &store);
        assert_eq!(out.rows.len(), 1, "{step}");
        match out.rows[0][0] {
            Value::Num(x) => assert_eq!(x, want, "{step}"),
            ref o => panic!("{step}: expected Num, got {o:?}"),
        }
    }
    // mean = 95 / 3.
    let out = gremlin_rows("g.V().hasLabel('Person').values('age').mean()", &store);
    match out.rows[0][0] {
        Value::Num(x) => assert!((x - 95.0 / 3.0).abs() < 1e-9, "mean was {x}"),
        ref o => panic!("mean: expected Num, got {o:?}"),
    }
}

#[test]
fn where_filters_the_value_stream() {
    let store = social();
    // Ages > 28: 30 and 40. Both the bare and P.-prefixed spellings work.
    for q in [
        "g.V().hasLabel('Person').values('age').where(gt(28))",
        "g.V().hasLabel('Person').values('age').where(P.gt(28))",
    ] {
        assert_eq!(
            value_bag(&gremlin_rows(q, &store)),
            vec!["Num(30.0);", "Num(40.0);"],
            "{q}"
        );
    }
}

#[test]
fn as_labels_and_select_projects_it() {
    let store = social();
    // Label the source as `p`, hop, then select `p` back and read its name.
    // KNOWS edges: alice->bob, alice->carol, bob->carol, so the sources are
    // alice, alice, bob.
    let q = "g.V().hasLabel('Person').as('p').out('KNOWS').select('p').values('name')";
    assert_eq!(
        value_bag(&gremlin_rows(q, &store)),
        vec!["Str(\"alice\");", "Str(\"alice\");", "Str(\"bob\");"]
    );
}

#[test]
fn within_and_without_membership() {
    let store = social();
    // within is an OR-of-equals; without is its negation.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').has('name', within('alice','carol')).values('age')",
            &store
        )),
        vec!["Num(30.0);", "Num(40.0);"]
    );
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').has('name', without('alice','carol')).values('age')",
            &store
        )),
        vec!["Num(25.0);"]
    );
    // within also works in where(...) on the value stream.
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').values('age').where(within(25, 40))",
            &store
        )),
        vec!["Num(25.0);", "Num(40.0);"]
    );
}

#[test]
fn bare_group_count_groups_by_the_current_element() {
    let store = social();
    // KNOWS targets are bob, carol, carol → one Map {bob:1, carol:2} (Gremlin
    // groupCount is a single Map, not (key,count) rows). Bare groupCount() over
    // the name stream and the .by('name') form agree.
    let want = vec!["Map([(Str(\"bob\"), Num(1.0)), (Str(\"carol\"), Num(2.0))]);"];
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').out('KNOWS').values('name').groupCount()",
            &store
        )),
        want
    );
    assert_eq!(
        value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').out('KNOWS').groupCount().by('name')",
            &store
        )),
        want
    );
}

#[test]
fn select_errors() {
    // A select over an UNKNOWN label drops every traverser (core yields nothing),
    // rather than erroring — whether alone or inside a multi-select.
    assert!(super::parse("g.V().as('p').select('q')").is_ok());
    assert!(super::parse("g.V().as('a').out('R').as('b').select('a','z')").is_ok());
}

#[test]
fn multi_select_builds_an_ordered_map() {
    let store = social();
    // bob KNOWS carol only: select('p','f') is a Map {p: bob, f: carol},
    // insertion-ordered (p then f), the values being the vertices' element maps.
    let out = gremlin_rows(
        "g.V().hasLabel('Person').has('name', 'bob').as('p').out('KNOWS').as('f') \
             .select('p', 'f')",
        &store,
    );
    assert_eq!(out.rows.len(), 1);
    // The `id` field of a vertex element-map value.
    let map_id = |v: &Value| match v {
        Value::Map(m) => m
            .iter()
            .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "id"))
            .map(|(_, id)| id.clone()),
        _ => None,
    };
    match &out.rows[0][0] {
        Value::Map(pairs) => {
            assert_eq!(pairs.len(), 2);
            assert!(matches!(&pairs[0].0, Value::Str(s) if &**s == "p"));
            assert!(matches!(&pairs[1].0, Value::Str(s) if &**s == "f"));
            assert!(matches!(map_id(&pairs[0].1), Some(Value::Str(s)) if &*s == "1")); // bob
            assert!(matches!(map_id(&pairs[1].1), Some(Value::Str(s)) if &*s == "2"));
            // carol
        }
        o => panic!("expected a Map, got {o:?}"),
    }
}

#[test]
fn has_accepts_a_bare_predicate() {
    let store = social();
    // has('age', gt(28)) (no P. prefix) now parses and agrees with P.gt(28).
    let bare = gremlin_rows(
        "g.V().hasLabel('Person').has('age', gt(28)).values('name')",
        &store,
    );
    let with_p = gremlin_rows(
        "g.V().hasLabel('Person').has('age', P.gt(28)).values('name')",
        &store,
    );
    assert_eq!(value_bag(&bare), value_bag(&with_p));
    assert_eq!(value_bag(&bare), vec!["Str(\"alice\");", "Str(\"carol\");"]);
}

/// A bare graph-ELEMENT frontier cannot be summed/averaged/min/max'd or sorted —
/// TinkerPop faults (a Vertex/Edge is not a number and has no natural order). The
/// frontier kind is known statically from the step chain, so this is a PARSE error.
/// A `values('<key>')` projection (or `order().by('<key>')`) makes it comparable/numeric.
#[test]
fn element_frontier_agg_and_order_fault() {
    // sum/mean/min/max over a DEFINITE raw element frontier → parse error (static). A
    // POST-BRANCH element frontier is unknown, so `coalesce(...).sum()` reaches the
    // runtime check instead (covered in the ported suite).
    for q in [
        "g.V().sum()",
        "g.V().mean()",
        "g.V().max()",
        "g.V().min()",
        "g.V().has('age', gt(1)).sum()", // has() preserves the element frontier
        "g.E().sum()",
        "g.V().out().max()",
    ] {
        assert!(super::parse(q).is_err(), "expected fault: {q}");
    }
    // Bare order() (or a direction-only by) over elements → parse error (static).
    for q in ["g.V().order()", "g.V().order().by(desc)", "g.E().order()"] {
        assert!(super::parse(q).is_err(), "expected fault: {q}");
    }
    // A scalar/projected frontier is fine — no false positives.
    for q in [
        "g.V().values('age').sum()",
        "g.V().values('age').max()",
        "g.V().values('age').order()",
        "g.V().count()",
        "g.V().order().by('name')",
        "g.V().out().count()",
    ] {
        assert!(super::parse(q).is_ok(), "expected ok: {q}");
    }
}

/// A tiny graph with ONE multi-label edge (`[KNOWS, CREATED]`).
fn multi_label_edge_store() -> Store {
    let nd = concat!(
        r#"{"id":"1","labels":["PERSON"],"props":{"name":"marko"}}"#,
        "\n",
        r#"{"id":"2","labels":["PERSON"],"props":{"name":"vadas"}}"#,
        "\n",
        r#"{"id":"e1","from":"1","to":"2","labels":["KNOWS","CREATED"],"props":{"w":0.5}}"#,
        "\n",
    );
    crate::ndjson::from_ndjson(nd).expect("fixture decodes")
}

/// The `labels` list of the single rendered edge in `rows`.
fn edge_labels(rows: &Rows) -> Vec<String> {
    let Value::Map(pairs) = &rows.rows[0][0] else {
        panic!("expected an edge map, got {:?}", rows.rows[0][0]);
    };
    let labels = pairs
        .iter()
        .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "labels"))
        .map(|(_, v)| v)
        .expect("edge map has a labels entry");
    let Value::List(items) = labels else {
        panic!("labels is a list");
    };
    items
        .iter()
        .map(|v| match v {
            Value::Str(s) => s.to_string(),
            o => panic!("label is a string, got {o:?}"),
        })
        .collect()
}

/// A multi-label edge renders its WHOLE label set (sorted), no matter which
/// type it was reached through — `edge_type_name` used to drop all but the
/// primary type, so `outE('KNOWS')` on a `[KNOWS, CREATED]` edge showed only
/// `[KNOWS]`. Both the nested map render and its streaming twin are checked.
#[test]
fn multi_label_edge_renders_all_labels_sorted() {
    let st = multi_label_edge_store();
    for q in [
        "g.E().hasLabel('KNOWS')",
        "g.E().hasLabel('CREATED')",
        "g.V().outE('KNOWS')",
        "g.E()",
    ] {
        let rows = gremlin_rows(q, &st);
        assert_eq!(
            edge_labels(&rows),
            vec!["CREATED".to_string(), "KNOWS".to_string()],
            "edge labels for `{q}`",
        );
    }
}

/// `path().sum()` (and min/max/mean) faults: a path is not a number. Known
/// statically from the chain, like the bare-element aggregate fault. `count()`
/// over a path is fine (it counts, not reduces numerically).
#[test]
fn path_aggregate_faults() {
    let st = multi_label_edge_store();
    for reduce in ["sum", "mean", "min", "max"] {
        let q = format!("g.V().out('KNOWS').path().{reduce}()");
        assert!(super::parse(&q).is_err(), "expected fault: {q}");
    }
    // count/fold over a path are fine.
    for q in [
        "g.V().out('KNOWS').path().count()",
        "g.V().out('KNOWS').path().fold()",
    ] {
        let _ = gremlin_rows(q, &st); // parses AND runs without faulting
    }
}

/// Branch bodies of MORE than a single hop now parse and reconverge correctly: a
/// 2-hop arm beside a 1-hop arm lands its element at the SAME slot, so a downstream
/// `values()` reads the true endpoint (it used to read a mid-hop node). Each branch
/// is checked against its manually-unrolled equivalent, so the assertion needs no
/// hand-computed expected set.
#[test]
fn multi_hop_branch_bodies_reconverge() {
    let st = social();
    // union(A, B).values == A.values (multiset) + B.values.
    let u = value_bag(&gremlin_rows(
        "g.V().union(out('KNOWS').out('KNOWS'), out('KNOWS')).values('name')",
        &st,
    ));
    let mut ab = value_bag(&gremlin_rows(
        "g.V().out('KNOWS').out('KNOWS').values('name')",
        &st,
    ));
    ab.extend(value_bag(&gremlin_rows(
        "g.V().out('KNOWS').values('name')",
        &st,
    )));
    ab.sort();
    assert_eq!(u, ab);

    // coalesce(A, B): the FIRST non-empty arm per element. Every result is a real
    // name (not a dense id / mid-hop node) — a 2-hop then-arm reconverges cleanly.
    let c = value_bag(&gremlin_rows(
        "g.V().coalesce(out('KNOWS').out('KNOWS'), out('KNOWS')).values('name')",
        &st,
    ));
    assert!(
        c.iter().all(|s| s.starts_with("Str(")),
        "coalesce values are names: {c:?}"
    );

    // choose with a 3-arm / traversal cond / multi-hop then-arm parses and runs.
    for q in [
        "g.V().choose(out('KNOWS'), out('KNOWS').out('KNOWS'), values('name')).count()",
        "g.V().optional(out('KNOWS').out('KNOWS')).count()",
    ] {
        let _ = gremlin_rows(q, &st);
    }

    // optional keeps the SOURCE where the multi-hop body is empty: every input row is
    // represented exactly once when the body reaches nothing new.
    let opt = gremlin_rows(
        "g.V().optional(out('WORKS_ON').out('WORKS_ON')).count()",
        &st,
    );
    // No 2-hop WORKS_ON exists, so optional yields every source unchanged (4 nodes).
    assert!(
        matches!(opt.rows[0][0], Value::Num(n) if n == 4.0),
        "optional fallback: {:?}",
        opt.rows[0][0]
    );
}

/// An ELEMENT step applied straight to a PATH faults — a path is not a vertex/edge,
/// so `path().values(...)`/`.hasLabel(...)`/`.inV()`/`.out(...)` throw (TinkerPop
/// raises ClassCastException). `unfold()` turns the path into its elements (element
/// steps then fine); `count`/`fold`/`range`/`order` are path-safe.
#[test]
fn element_step_on_a_path_faults() {
    let st = social();
    for q in [
        "g.V().out('KNOWS').path().values('name')",
        "g.V().out('KNOWS').path().hasLabel('Person').count()",
        "g.V().outE('KNOWS').path().inV()",
        "g.V().out('KNOWS').path().out('KNOWS').count()",
        "g.V().out('KNOWS').path().order().values('name')", // order preserves the path
        "g.V().out('KNOWS').path().id()",
    ] {
        assert!(super::parse(q).is_err(), "expected fault: {q}");
    }
    // Path-safe: the path is counted / consumed, not treated as an element.
    for q in [
        "g.V().out('KNOWS').path().count()",
        "g.V().out('KNOWS').path().fold()",
        "g.V().out('KNOWS').path().range(0, 1).count()",
        "g.V().out('KNOWS').path().unfold().values('name')", // unfold consumes the path
    ] {
        let _ = gremlin_rows(q, &st);
    }
}

/// `otherV()` inside a branch arm resolves against the edge's ORIGIN when the edge
/// was reached THROUGH a vertex (`V().outE().optional(otherV())`) — matching real
/// TinkerPop (verified against gremlin-console on createModern(): count 2, names
/// [josh,vadas]). Off a BARE edge frontier (`E().otherV()`) there is no origin, so it
/// faults (TinkerPop throws "path history ... does not contain a previous vertex").
#[test]
fn otherv_in_branch_resolves_against_edge_origin() {
    let st = social(); // 3 KNOWS edges (a->bob, a->c, bob->c), 1 WORKS_ON
                       // Reached through a vertex → otherV resolves the far endpoint for every edge.
    let c = gremlin_rows("g.V().outE('KNOWS').optional(otherV()).count()", &st);
    assert!(
        matches!(c.rows[0][0], Value::Num(n) if n == 3.0),
        "got {:?}",
        c.rows[0][0]
    );
    // A bare edge frontier (or a vertex) has no reference vertex → otherV faults at PARSE,
    // as in TinkerPop. (A reconverged BRANCH of edges resolves it — covered by the fuzzer.)
    for q in [
        "g.E().otherV()",
        "g.E().optional(otherV()).count()",
        "g.V().optional(otherV()).count()",
    ] {
        assert!(super::parse(q).is_err(), "expected fault: {q}");
    }
}

/// coalesce falls through EXACTLY: an arm whose leading hop exists but whose FULL
/// (multi-hop) body produces nothing must NOT consume the element — a later arm
/// still gets it. The prov-safe body gets a full-body EXISTS guard rather than the
/// leading-hop approximation, so no element is wrongly dropped.
#[test]
fn coalesce_falls_through_on_a_dead_multi_hop_arm() {
    let st = social();
    // Every node yields exactly once: its 2-hop KNOWS target if any, else its name.
    // The leading-hop approximation dropped nodes with a KNOWS out but no 2-hop.
    let c = gremlin_rows(
        "g.V().coalesce(out('KNOWS').out('KNOWS'), values('name')).count()",
        &st,
    );
    let n = gremlin_rows("g.V().count()", &st);
    assert_eq!(format!("{:?}", c.rows[0][0]), format!("{:?}", n.rows[0][0]));
}

/// A heterogeneous branch (an element arm beside a scalar arm) renders its
/// vertices/edges as ELEMENT MAPS, not dense ids. The mixed column falls into
/// `concat_cols`' Gen path, which used `value_at` (a node → its dense id) — now
/// `render_cell` (a node → `{id, labels, properties}`).
#[test]
fn mixed_type_branch_renders_elements_not_dense_ids() {
    let st = social();
    let rows = gremlin_rows("g.V().union(out('KNOWS'), values('name'))", &st);
    let mut saw_map = false;
    for r in &rows.rows {
        match &r[0] {
            // A vertex from the element arm: a map, NEVER a bare Num (a dense id).
            Value::Map(_) => saw_map = true,
            Value::Str(_) => {} // a name from the value arm
            other => panic!("branch element rendered as {other:?}, expected a map or a name"),
        }
    }
    assert!(saw_map, "the element arm must contribute vertex maps");
}

/// A coalesce/union whose arms reconverge at DIFFERENT widths (an `out()` expand
/// beside a `limit(3).label()` scalar projection) used to index a column a narrow
/// arm lacks and PANIC (`batch.slot` out of bounds). The concat now pads short arms
/// with NULLs, so these run and terminate cleanly. (The reconverged VALUES over a
/// mismatched-shape branch are a separate, fuzz-tracked divergence — this only
/// asserts no crash and a well-formed terminal.)
#[test]
fn mismatched_width_branch_does_not_panic() {
    let st = multi_label_edge_store();
    for q in [
        "g.V().coalesce(out('CREATED'), limit(3).label()).count()",
        "g.E().coalesce(hasLabel('NOPE'), range(0, 1)).count()",
        "g.V().coalesce(id().has('name', gte(-1)), out('NOPE')).values('lang')",
        "g.V().coalesce(out('CREATED'), limit(3).label())",
        "g.V().hasLabel('NOPE').union(limit(0).both('NOPE'), id()).dedup().count()",
    ] {
        // Parses AND runs without a slot-out-of-bounds panic.
        let _ = gremlin_rows(q, &st);
    }
    // A count terminal over a mismatched-width coalesce is a single well-formed row.
    let c = gremlin_rows(
        "g.V().coalesce(out('CREATED'), limit(3).label()).count()",
        &st,
    );
    assert_eq!(c.rows.len(), 1);
    assert!(matches!(c.rows[0][0], Value::Num(_)));
}

/// `dedup()` after an empty/zero-slice hop does not panic. An empty `inE('X')`
/// over an UNKNOWN edge type narrows the batch below the endpoint slot that
/// `otherV()` tagged, so `DistinctBy` read a slot past the 0-row batch's width.
/// A zero-row dedup is trivially the empty input.
#[test]
fn dedup_after_empty_slice_does_not_panic() {
    let st = multi_label_edge_store();
    for q in [
        "g.V().inE('NOPE').otherV().range(0, 0).dedup().count()",
        "g.V().inE('NOPE').otherV().range(0, 0).dedup()",
        "g.V().inE('KNOWS').otherV().range(0, 0).dedup().count()",
        "g.V().outE('NOPE').inV().dedup().count()",
    ] {
        let _ = gremlin_rows(q, &st);
    }
    // The count is 0 (nothing survives the zero slice).
    let c = gremlin_rows(
        "g.V().inE('NOPE').otherV().range(0, 0).dedup().count()",
        &st,
    );
    assert!(matches!(c.rows[0][0], Value::Num(n) if n == 0.0));
}

/// `inject` PREPENDS its literals, and a downstream element step reads THROUGH
/// the boxed element maps the heterogeneous union produces — `V().inject(0)`
/// used to surface each vertex as its dense id (a bare number), so
/// `.values('name')` saw numbers and returned nothing.
#[test]
fn inject_prepends_and_values_reads_boxed_elements() {
    let st = multi_label_edge_store();

    // values('name') AFTER inject is a type error: inject prepends a literal (0), so the
    // frontier is no longer purely vertices, and values() reads off an element — TinkerPop
    // throws ClassCastException, and both engines reject it at parse (element-type algebra).
    assert!(
        super::parse("g.V().inject(0).values('name')").is_err(),
        "values() on a post-inject scalar frontier must be rejected"
    );

    // inject(0) alone: the literal comes FIRST (TinkerPop prepends), then the
    // vertices render as element maps (not dense ids).
    let alone = gremlin_rows("g.V().inject(0)", &st);
    assert!(
        matches!(alone.rows[0][0], Value::Num(n) if n == 0.0),
        "inject prepends the literal, got {:?}",
        alone.rows[0][0],
    );
    assert!(
        matches!(&alone.rows[1][0], Value::Map(_)),
        "a vertex renders as an element map, got {:?}",
        alone.rows[1][0],
    );
}
