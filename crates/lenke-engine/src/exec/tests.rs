use super::*;
use crate::ir::Plan;
use crate::store::Builder;

/// The iterative `varlen_walk` must emit the exact same paths, in the exact same
/// order, as the recursive `varlen_dfs` it replaced — over every mode/direction/bound
/// combination on a spread of random graphs. Byte-identity is the hard invariant; this
/// is the direct A/B guard (the corpus + differential fuzzer cover the predicate hooks).
#[test]
fn iterative_varlen_matches_recursive() {
    struct RecordEmit {
        paths: Vec<(Vec<u32>, Vec<u32>)>,
    }
    impl VarlenEmit for RecordEmit {
        fn emit(&mut self, _row: usize, node_stack: &[u32], edge_stack: &[u32]) {
            self.paths.push((node_stack.to_vec(), edge_stack.to_vec()));
        }
        fn should_stop(&self) -> bool {
            false
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn collect(
        store: &Store,
        n_nodes: u32,
        mode: PathMode,
        dir: Dir,
        min: u32,
        max: u32,
        k: u32,
        double_loops: bool,
        iterative: bool,
    ) -> Vec<(Vec<u32>, Vec<u32>)> {
        let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
        let mut sink = RecordEmit { paths: Vec::new() };
        let mut used: Vec<u32> = Vec::new();
        for v in 0..n_nodes {
            if node_unique {
                used.push(v);
            }
            let mut ns = vec![v];
            let mut es: Vec<u32> = Vec::new();
            if iterative {
                varlen_walk(
                    store,
                    v,
                    min,
                    max,
                    dir,
                    &[],
                    mode,
                    v,
                    &mut used,
                    v as usize,
                    &mut ns,
                    &mut es,
                    None,
                    k,
                    None,
                    None,
                    double_loops,
                    &mut sink,
                );
            } else {
                varlen_dfs(
                    store,
                    v,
                    0,
                    min,
                    max,
                    dir,
                    &[],
                    mode,
                    v,
                    &mut used,
                    v as usize,
                    &mut ns,
                    &mut es,
                    None,
                    k,
                    None,
                    None,
                    double_loops,
                    &mut sink,
                );
            }
            if node_unique {
                used.pop();
            }
            assert!(used.is_empty(), "used stack left dirty after a source");
        }
        sink.paths
    }

    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _trial in 0..60 {
        let n_nodes = 4 + (rng() % 7) as u32;
        let mut nd = String::new();
        for i in 0..n_nodes {
            nd.push_str(&format!(
                "{{\"id\":\"n{i}\",\"labels\":[\"P\"],\"props\":{{}}}}\n"
            ));
        }
        let ecount = (rng() % (u64::from(n_nodes) * 3 + 1)) as u32;
        for e in 0..ecount {
            let f = (rng() % u64::from(n_nodes)) as u32;
            let t = (rng() % u64::from(n_nodes)) as u32;
            nd.push_str(&format!(
                    "{{\"id\":\"e{e}\",\"from\":\"n{f}\",\"to\":\"n{t}\",\"labels\":[\"R\"],\"props\":{{}}}}\n"
                ));
        }
        let store = crate::ndjson::from_ndjson(&nd).unwrap();
        for mode in [
            PathMode::Walk,
            PathMode::Trail,
            PathMode::Simple,
            PathMode::Acyclic,
        ] {
            for dir in [Dir::Out, Dir::In, Dir::Both] {
                for (min, max, k) in [
                    (0u32, 3u32, 1u32),
                    (1, 4, 1),
                    (2, 2, 1),
                    (1, 6, 2),
                    (0, 4, 1),
                ] {
                    for double_loops in [false, true] {
                        let rec =
                            collect(&store, n_nodes, mode, dir, min, max, k, double_loops, false);
                        let itr =
                            collect(&store, n_nodes, mode, dir, min, max, k, double_loops, true);
                        assert_eq!(
                                rec, itr,
                                "mode={mode:?} dir={dir:?} {min}..={max} k={k} dl={double_loops} ecount={ecount}"
                            );
                        // The count/agg fast-path twins (k=1, no preds) must equal the
                        // materialized rows: count == #rows, agg-sum == sum of endpoints.
                        if k == 1 {
                            let node_unique = matches!(mode, PathMode::Simple | PathMode::Acyclic);
                            let mut total = 0u64;
                            let mut used: Vec<u32> = Vec::new();
                            for src in 0..n_nodes {
                                if node_unique {
                                    used.push(src);
                                }
                                varlen_count_dfs(
                                    &store,
                                    src,
                                    0,
                                    min,
                                    max,
                                    dir,
                                    &[],
                                    mode,
                                    src,
                                    &mut used,
                                    &mut total,
                                    double_loops,
                                );
                                if node_unique {
                                    used.pop();
                                }
                            }
                            assert_eq!(
                                    total as usize,
                                    itr.len(),
                                    "count twin vs materialized rows: mode={mode:?} dir={dir:?} {min}..={max} dl={double_loops}"
                                );
                            // The agg fast-path is only taken without a both()-doubled
                            // self-loop, so compare it against the dl=false rows only.
                            if !double_loops {
                                let mut sum = 0u64;
                                let mut used2: Vec<u32> = Vec::new();
                                for src in 0..n_nodes {
                                    if node_unique {
                                        used2.push(src);
                                    }
                                    varlen_agg_dfs(
                                        &store,
                                        src,
                                        0,
                                        min,
                                        max,
                                        dir,
                                        &[],
                                        mode,
                                        src,
                                        &mut used2,
                                        &mut |v| sum += u64::from(v),
                                    );
                                    if node_unique {
                                        used2.pop();
                                    }
                                }
                                let want: u64 = itr
                                    .iter()
                                    .map(|(ns, _)| u64::from(*ns.last().unwrap()))
                                    .sum();
                                assert_eq!(
                                        sum, want,
                                        "agg-sum twin vs materialized endpoints: mode={mode:?} dir={dir:?} {min}..={max}"
                                    );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The iterative walk uses O(1) call stack regardless of closure depth: a 40k-deep
/// traversal — which the old recursive DFS would have driven ~40k frames deep, blowing
/// any normal stack — completes on a deliberately TINY (512 KiB) thread. This is why a
/// deep closure can no longer overflow (and why its peak memory is now bounded heap,
/// not committed stack). Drives `run_varlen` directly, off any big stack.
#[test]
fn deep_varlen_walk_runs_on_a_tiny_stack() {
    // A 40k-long chain n0->n1->...; recursing it would need ~20 MB of stack.
    let mut nd = String::new();
    for i in 0..40_000u32 {
        nd.push_str(&format!(
            "{{\"id\":\"n{i}\",\"labels\":[\"P\"],\"props\":{{}}}}\n"
        ));
    }
    for i in 0..39_999u32 {
        nd.push_str(&format!(
                "{{\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"props\":{{}}}}\n",
                i + 1
            ));
    }
    let store = crate::ndjson::from_ndjson(&nd).unwrap();
    // Drive the walk directly (bypassing the pull machinery) so the test isolates
    // varlen_walk's OWN stack use. An unbounded out-walk from n0 emits one path per
    // reachable prefix (39 999 of them) and recurses 40k deep in the old DFS.
    struct CountEmit(usize);
    impl VarlenEmit for CountEmit {
        fn emit(&mut self, _row: usize, _node_stack: &[u32], _edge_stack: &[u32]) {
            self.0 += 1;
        }
        fn should_stop(&self) -> bool {
            false
        }
    }
    let handle = std::thread::Builder::new()
        .stack_size(512 * 1024) // far below the ~20 MB a recursive DFS would need
        .spawn(move || {
            let mut sink = CountEmit(0);
            run_varlen(
                &[0], // source = n0
                &store,
                &[],      // any edge label
                1,        // min
                u32::MAX, // max (unbounded)
                Dir::Out,
                PathMode::Walk,
                None,
                1, // k
                None,
                None,
                false,
                &mut sink,
            );
            // The count fast-path twin must ALSO be O(1) stack (it shares varlen_scan_walk).
            let mut total = 0u64;
            let mut used: Vec<u32> = Vec::new();
            varlen_count_dfs(
                &store,
                0,
                0,
                1,
                u32::MAX,
                Dir::Out,
                &[],
                PathMode::Walk,
                0,
                &mut used,
                &mut total,
                false,
            );
            (sink.0, total)
        })
        .unwrap();
    assert_eq!(
        handle.join().expect("must not overflow the tiny stack"),
        (39_999, 39_999)
    );
}

fn n(x: f64) -> Value {
    Value::Num(x)
}
fn s(x: &str) -> Value {
    Value::Str(x.into())
}
fn prop(slot: usize, key: &str) -> Expr {
    Expr::Prop {
        slot,
        key: key.to_string(),
    }
}
fn lit(v: Value) -> Expr {
    Expr::Lit(v)
}
fn cmp(op: CompareOp, l: Expr, r: Expr) -> Expr {
    Expr::Compare {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}
fn scan(label: &str) -> Plan {
    Plan::Scan {
        label: Some(label.to_string()),
    }
}
fn names_of(out: &Rows, col: usize) -> Vec<String> {
    out.rows
        .iter()
        .map(|r| match &r[col] {
            Value::Str(x) => x.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
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

/// The opt-in edge-type index is a pure optimization: a type-filtered hop
/// returns the SAME rows with it on as with it off (for_each_nbr routes to the
/// bucket, but the answer is identical).
#[test]
fn edge_type_index_gives_identical_query_results() {
    let mut store = social();
    let plan = crate::gql::parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS b").unwrap();
    let mut before = names_of(&run(&plan, &store), 0);
    before.sort();
    store.create_edge_type_index();
    let mut after = names_of(&run(&plan, &store), 0);
    after.sort();
    assert_eq!(before, after);
    // alice KNOWS bob & carol; bob KNOWS carol → bob, carol, carol.
    assert_eq!(after, vec!["bob", "carol", "carol"]);
    // A count through the fused fast path also matches.
    let cplan = crate::gql::parse("MATCH (a:Person)-[:KNOWS]->() RETURN count(*) AS c").unwrap();
    assert!(matches!(run(&cplan, &store).rows[0][0], Value::Num(x) if x == 3.0));
}

/// A store whose only node named "target" (n0) is reachable by several 2- and
/// 3-hop R-paths, plus decoy paths that never reach it. Used to prove the
/// multi-hop reverse-seed returns the SAME multiset as the forward walk.
fn reverse_seed_store() -> Store {
    let mut b = Builder::default();
    let t = b.node(&["N"], &[("name", s("target"))]);
    let m1 = b.node(&["N"], &[("name", s("m1"))]);
    let m2 = b.node(&["N"], &[("name", s("m2"))]);
    let s3 = b.node(&["N"], &[("name", s("s3"))]);
    let s4 = b.node(&["N"], &[("name", s("s4"))]);
    let r8 = b.node(&["N"], &[("name", s("r8"))]);
    let r9 = b.node(&["N"], &[("name", s("r9"))]);
    // decoy chain that never reaches the target
    let d0 = b.node(&["N"], &[("name", s("other"))]);
    let d1 = b.node(&["N"], &[("name", s("d1"))]);
    let d2 = b.node(&["N"], &[("name", s("d2"))]);
    b.edge(m1, t, "R");
    b.edge(m2, t, "R");
    b.edge(s3, m1, "R");
    b.edge(s3, m2, "R"); // s3 reaches target two ways (diamond)
    b.edge(s4, m1, "R");
    b.edge(r8, s3, "R");
    b.edge(r9, s4, "R");
    b.edge(d1, d0, "R"); // decoys
    b.edge(d2, d1, "R");
    b.build()
}

/// The multi-hop reverse-seed (an indexed selective endpoint over an Expand chain)
/// returns exactly the rows the forward walk does — same multiset, index on or off,
/// at two and three hops. Index off ⇒ forward; index on ⇒ seed-and-reverse.
#[test]
fn reverse_seed_multihop_matches_forward() {
    let mut st = reverse_seed_store();
    let two = "MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'target' RETURN a.name AS a";
    let three =
        "MATCH (a)-[:R]->(b)-[:R]->(c)-[:R]->(d) WHERE d.name = 'target' RETURN a.name AS a";
    let sorted = |st: &Store, q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
        v.sort();
        v
    };

    // Forward (no index): 2-hop reaches target via m1 (from s3,s4) and m2 (from s3).
    let fwd2 = sorted(&st, two);
    assert_eq!(fwd2, vec!["s3", "s3", "s4"]);
    let fwd3 = sorted(&st, three); // r8→s3→{m1,m2}→t, r9→s4→m1→t
    assert_eq!(fwd3, vec!["r8", "r8", "r9"]);

    // Index on: the reverse-seed fires and must return the identical multiset.
    st.create_index("name");
    assert_eq!(sorted(&st, two), fwd2);
    assert_eq!(sorted(&st, three), fwd3);

    // count(*) rides the same seed and matches the row count.
    let cnt = |q: &str| match run(&crate::gql::parse(q).unwrap(), &st).rows[0][0] {
        Value::Num(x) => x,
        _ => panic!("count not numeric"),
    };
    assert_eq!(
        cnt("MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'target' RETURN count(*) AS c"),
        3.0
    );
    assert_eq!(
        cnt("MATCH (a)-[:R]->(b)-[:R]->(c)-[:R]->(d) WHERE d.name = 'target' RETURN count(*) AS c"),
        3.0
    );
}

/// A keyless `LIMIT` over a reverse-seeded chain (the OrderPage fast path) returns
/// the same rows the forward walk + LIMIT does — capped below the result size, and
/// unchanged above it — index on or off. Guards against the OrderPage stream path
/// silently bypassing the seed.
#[test]
fn reverse_seed_under_limit_matches_forward() {
    let mut st = reverse_seed_store();
    let q = |lim: usize| {
        format!(
            "MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'target' RETURN a.name AS a LIMIT {lim}"
        )
    };
    let rows = |st: &Store, lim: usize| run(&crate::gql::parse(&q(lim)).unwrap(), st).rows.len();
    // Forward (no index): 3 matching rows, so LIMIT 2 caps to 2, LIMIT 10 keeps 3.
    assert_eq!(rows(&st, 2), 2);
    assert_eq!(rows(&st, 10), 3);
    st.create_index("name"); // reverse-seed now fires under the LIMIT
    assert_eq!(rows(&st, 2), 2);
    assert_eq!(rows(&st, 10), 3);
    // Above the result size the full multiset must match the forward walk exactly.
    let mut got = names_of(&run(&crate::gql::parse(&q(10)).unwrap(), &st), 0);
    got.sort();
    assert_eq!(got, vec!["s3", "s3", "s4"]);
}

/// The reverse VAR-LENGTH seed returns the forward walk's exact multiset — including
/// duplicate-path multiplicity (s3 reaches the target two ways at length 2). Index
/// off ⇒ forward var-length; index on ⇒ seed the endpoint and walk the quantifier
/// backward. Covers a low and a high hop window.
#[test]
fn reverse_varlen_seed_matches_forward() {
    let mut st = reverse_seed_store();
    let cases: [(&str, Vec<&str>); 2] = [
        (
            "MATCH (a)-[:R]->{1,2}(b) WHERE b.name = 'target' RETURN a.name AS a",
            vec!["m1", "m2", "s3", "s3", "s4"],
        ),
        (
            "MATCH (a)-[:R]->{2,3}(b) WHERE b.name = 'target' RETURN a.name AS a",
            vec!["r8", "r8", "r9", "s3", "s3", "s4"],
        ),
    ];
    let sorted = |st: &Store, q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
        v.sort();
        v
    };
    // Forward (no index) matches the hand-computed multiset.
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "forward {q}");
    }
    // Index on → the reverse var-length seed fires and returns the identical multiset.
    st.create_index("name");
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "reverse {q}");
    }
}

/// A store for the COMPOUND reverse var-length: a fixed `-[:F]->` hop feeds an R
/// var-length to a unique "target". `x` reaches the target two ways at length 2, so
/// `a1` (which F-reaches `x`) must appear with that multiplicity.
fn compound_varlen_store() -> Store {
    let mut b = Builder::default();
    let t = b.node(&["N"], &[("name", s("target"))]);
    let m1 = b.node(&["N"], &[("name", s("m1"))]);
    let m2 = b.node(&["N"], &[("name", s("m2"))]);
    let x = b.node(&["N"], &[("name", s("x"))]);
    let a1 = b.node(&["N"], &[("name", s("a1"))]);
    let a2 = b.node(&["N"], &[("name", s("a2"))]);
    b.edge(m1, t, "R");
    b.edge(m2, t, "R");
    b.edge(x, m1, "R");
    b.edge(x, m2, "R"); // x reaches target two ways at length 2
    b.edge(a1, m1, "F");
    b.edge(a2, m2, "F");
    b.edge(a1, x, "F");
    b.build()
}

/// The reverse var-length seed behind a leading fixed hop returns the forward walk's
/// exact multiset. Reversal walks the var-length back to the fixed hop's target, then
/// the fixed hop back to the labeled source — `a1` appears three times at {1,2}
/// (via m1 once, via x twice) and twice at {2,3} (both length-2 paths through x).
#[test]
fn reverse_varlen_compound_matches_forward() {
    let mut st = compound_varlen_store();
    let cases: [(&str, Vec<&str>); 2] = [
        (
            "MATCH (a)-[:F]->(v)-[:R]->{1,2}(c) WHERE c.name = 'target' RETURN a.name AS a",
            vec!["a1", "a1", "a1", "a2"],
        ),
        (
            "MATCH (a)-[:F]->(v)-[:R]->{2,3}(c) WHERE c.name = 'target' RETURN a.name AS a",
            vec!["a1", "a1"],
        ),
    ];
    let sorted = |st: &Store, q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
        v.sort();
        v
    };
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "forward {q}");
    }
    st.create_index("name"); // compound reverse var-length fires
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "reverse {q}");
    }
}

/// `DISTINCT <low-card dict prop> LIMIT n` with `n` above the distinct count returns
/// the same rows as the uncapped DISTINCT (the LIMIT is a no-op) — the fast path must
/// not drop or reorder values.
#[test]
fn distinct_dict_limit_noop_matches_uncapped() {
    let mut b = Builder::default();
    for i in 0..40u32 {
        b.node(&["N"], &[("c", s(["x", "y", "z"][(i % 3) as usize]))]);
    }
    let st = b.build();
    let rows = |q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        v.sort();
        v
    };
    let full = rows("MATCH (n) RETURN DISTINCT n.c AS x");
    assert_eq!(full, vec!["x", "y", "z"]);
    // LIMIT 10 > 3 distinct → no-op; the vectorized fast path yields the same set.
    assert_eq!(rows("MATCH (n) RETURN DISTINCT n.c AS x LIMIT 10"), full);
    // A binding LIMIT still caps (3 distinct, LIMIT 2 → 2 rows).
    assert_eq!(
        run(
            &crate::gql::parse("MATCH (n) RETURN DISTINCT n.c AS x LIMIT 2").unwrap(),
            &st
        )
        .rows
        .len(),
        2
    );
}

/// The vectorized filter mask (`eval_mask`) keeps three-valued (Kleene) logic exact for
/// a complex predicate over a NULL-bearing column: an UNKNOWN row is dropped, `OR` with a
/// TRUE is TRUE even when the other side is UNKNOWN, and `NOT UNKNOWN` stays UNKNOWN.
#[test]
fn eval_mask_three_valued_semantics() {
    let mut b = Builder::default();
    b.node(&["N"], &[("age", n(60.0)), ("city", s("oslo"))]); // n0
    b.node(&["N"], &[("age", n(10.0)), ("city", s("bergen"))]); // n1
    b.node(&["N"], &[("city", s("oslo"))]); // n2: age absent (null)
    b.node(&["N"], &[("city", s("bergen"))]); // n3: age absent
    let st = b.build();
    let cities = |q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        v.sort();
        v
    };
    // n0 age>50=T; n1 both F; n2 null OR city=T → T; n3 null OR F → null (dropped).
    assert_eq!(
        cities("MATCH (n) WHERE (n.age > 50 OR n.city = 'oslo') RETURN n.city AS c"),
        vec!["oslo", "oslo"]
    );
    // NOT of the above: n0 F, n1 T, n2 F, n3 NOT null = null (dropped).
    assert_eq!(
        cities("MATCH (n) WHERE NOT (n.age > 50 OR n.city = 'oslo') RETURN n.city AS c"),
        vec!["bergen"]
    );
}

/// A string-search leaf (`STARTS WITH`/`ENDS WITH`/`CONTAINS`) inside a complex
/// predicate keeps three-valued semantics through `eval_mask`: a null string cell is
/// UNKNOWN (dropped, or `NOT UNKNOWN` = UNKNOWN), matching the boxed `str_bool`.
#[test]
fn eval_mask_string_search_three_valued() {
    let mut b = Builder::default();
    b.node(&["N"], &[("name", s("apple")), ("city", s("oslo"))]); // n0
    b.node(&["N"], &[("name", s("banana")), ("city", s("bergen"))]); // n1
    b.node(&["N"], &[("city", s("bergen"))]); // n2: name null
    let st = b.build();
    let cities = |q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        v.sort();
        v
    };
    // n0 name STARTS 'a' = T; n1 F or 'bergen' ENDS 'o' = F; n2 null OR F = null (drop).
    assert_eq!(
        cities(
            "MATCH (n) WHERE (n.name STARTS WITH 'a' OR n.city ENDS WITH 'o') RETURN n.city AS c"
        ),
        vec!["oslo"]
    );
    // NOT (name STARTS 'a'): n0 F, n1 T, n2 NOT null = null (drop).
    assert_eq!(
        cities("MATCH (n) WHERE NOT (n.name STARTS WITH 'a') RETURN n.city AS c"),
        vec!["bergen"]
    );
}

/// A chain that BINDS an edge (`-[e:R]->`) with an edge-property residual reverse-seeds
/// correctly: the reverse-walk must capture each hop's edge and land it in the right
/// column, so both the edge residual (`e.w < 100`) and a `RETURN e.w` see the true edge.
#[test]
fn reverse_bound_edge_matches_forward() {
    let mut bd = Builder::default();
    let t = bd.node(&["N"], &[("name", s("target"))]);
    let m1 = bd.node(&["N"], &[("name", s("m1"))]);
    let m2 = bd.node(&["N"], &[("name", s("m2"))]);
    let a1 = bd.node(&["N"], &[("name", s("a1"))]);
    let a2 = bd.node(&["N"], &[("name", s("a2"))]);
    bd.edge(m1, t, "R"); // eid 0
    bd.edge(m2, t, "R"); // eid 1
    bd.edge(a1, m1, "F"); // eid 2
    bd.edge(a2, m2, "F"); // eid 3
    let mut st = bd.build();
    st.set_edge_prop(0, "w", n(5.0));
    st.set_edge_prop(1, "w", n(500.0));
    let names = |st: &Store, q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
        v.sort();
        v
    };
    // a1→m1-(w5)→target passes `e.w < 100`; a2→m2-(w500) fails.
    let q =
        "MATCH (a)-[:F]->(m)-[e:R]->(c) WHERE c.name = 'target' AND e.w < 100 RETURN a.name AS a";
    assert_eq!(names(&st, q), vec!["a1"]);
    st.create_index("name");
    assert_eq!(names(&st, q), vec!["a1"]); // reverse-seed with the edge residual applied

    // The bound-edge column must carry the actual edges (RETURN reads slot 2 = edge).
    let qe = "MATCH (a)-[:F]->(m)-[e:R]->(c) WHERE c.name = 'target' RETURN e.w AS w";
    let mut ws: Vec<f64> = run(&crate::gql::parse(qe).unwrap(), &st)
        .rows
        .iter()
        .map(|r| match r[0] {
            Value::Num(x) => x,
            _ => panic!("w not numeric"),
        })
        .collect();
    ws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(ws, vec![5.0, 500.0]);
}

/// The generalized seeds — range (`>`), positive `IN`, and `OR` of seedables — each
/// return the forward walk's exact multiset (the `OR` case has s1 twice: it reaches an
/// age-matched endpoint and a city-matched one). Forward with no index; seeded with a
/// hash index (equality/IN) and a range index (range) present.
#[test]
fn reverse_range_in_or_seeds_match_forward() {
    let mut b = Builder::default();
    let e1 = b.node(
        &["N"],
        &[("name", s("e1")), ("age", n(95.0)), ("city", s("oslo"))],
    );
    let e2 = b.node(
        &["N"],
        &[("name", s("e2")), ("age", n(99.0)), ("city", s("bergen"))],
    );
    let e3 = b.node(
        &["N"],
        &[("name", s("e3")), ("age", n(50.0)), ("city", s("oslo"))],
    );
    let s1 = b.node(&["N"], &[("name", s("s1"))]);
    let s2 = b.node(&["N"], &[("name", s("s2"))]);
    let s3 = b.node(&["N"], &[("name", s("s3"))]);
    b.edge(s1, e1, "R");
    b.edge(s2, e2, "R");
    b.edge(s1, e3, "R");
    b.edge(s3, e1, "R");
    let mut st = b.build();
    let cases: [(&str, Vec<&str>); 3] = [
        (
            "MATCH (a)-[:R]->(b) WHERE b.age > 90 RETURN a.name AS a",
            vec!["s1", "s2", "s3"],
        ),
        (
            "MATCH (a)-[:R]->(b) WHERE b.age IN [95, 99] RETURN a.name AS a",
            vec!["s1", "s2", "s3"],
        ),
        (
            "MATCH (a)-[:R]->(b) WHERE (b.age > 90 OR b.city = 'oslo') RETURN a.name AS a",
            vec!["s1", "s1", "s2", "s3"],
        ),
    ];
    let sorted = |st: &Store, q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
        v.sort();
        v
    };
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "forward {q}");
    }
    st.create_index("age"); // IN over the hash index
    st.create_index("city"); // OR's equality disjunct
    st.create_range_index("age"); // range / OR's range disjunct
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "seeded {q}");
    }
}

/// Two range bounds on the SAME key seed their exact intersection via the two-sided
/// range seek. A narrow interval keeps only the in-range endpoints, a contradictory
/// pair (lo > hi) seeds the empty set, and a same-direction pair falls through to the
/// generic per-conjunct seed — all matching the forward walk.
#[test]
fn reverse_seed_interval_intersection_matches_forward() {
    let mut b = Builder::default();
    let e1 = b.node(&["N"], &[("name", s("e1")), ("age", n(95.0))]);
    let e2 = b.node(&["N"], &[("name", s("e2")), ("age", n(99.0))]);
    let e3 = b.node(&["N"], &[("name", s("e3")), ("age", n(50.0))]);
    let s1 = b.node(&["N"], &[("name", s("s1"))]);
    let s2 = b.node(&["N"], &[("name", s("s2"))]);
    let s3 = b.node(&["N"], &[("name", s("s3"))]);
    b.edge(s1, e1, "R");
    b.edge(s2, e2, "R");
    b.edge(s3, e1, "R");
    b.edge(s2, e3, "R"); // e3 (age 50) is reachable but filtered out by every case
    let mut st = b.build();
    let cases: [(&str, Vec<&str>); 3] = [
        // narrow interval [>90, <98] → only e1 (95); e2=99 and e3=50 excluded
        (
            "MATCH (a)-[:R]->(b) WHERE (b.age > 90 AND b.age < 98) RETURN a.name AS a",
            vec!["s1", "s3"],
        ),
        // contradictory (>98 AND <90) → empty
        (
            "MATCH (a)-[:R]->(b) WHERE (b.age > 98 AND b.age < 90) RETURN a.name AS a",
            vec![],
        ),
        // same direction (>40 AND >60) → e1,e2 (e3=50 fails >60); generic per-conjunct seed
        (
            "MATCH (a)-[:R]->(b) WHERE (b.age > 40 AND b.age > 60) RETURN a.name AS a",
            vec!["s1", "s2", "s3"],
        ),
    ];
    let sorted = |st: &Store, q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
        v.sort();
        v
    };
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "forward {q}");
    }
    st.create_range_index("age");
    for (q, want) in &cases {
        assert_eq!(&sorted(&st, q), want, "seeded {q}");
    }
}

/// `NOT (x <op> v)` normalizes to `x <neg op> v` — the negated spelling returns exactly
/// the positive one's rows, including the 3VL NULL case (an absent operand is UNKNOWN and
/// dropped either way, never resurrected) and stacked `NOT NOT NOT` collapsing.
#[test]
fn negated_comparison_matches_positive_spelling() {
    let mut b = Builder::default();
    let n1 = b.node(&["N"], &[("name", s("keep")), ("age", n(30.0))]);
    let n2 = b.node(&["N"], &[("name", s("skip")), ("age", n(40.0))]);
    let n3 = b.node(&["N"], &[("name", s("noage"))]); // age ABSENT
    let src = b.node(&["N"], &[("name", s("src"))]);
    b.edge(src, n1, "R");
    b.edge(src, n2, "R");
    b.edge(src, n3, "R");
    let st = b.build();
    let sorted = |q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        v.sort();
        v
    };
    // NOT (age <> 30) == age = 30 → keep n1; absent n3 is UNKNOWN, dropped both ways.
    assert_eq!(
        sorted("MATCH (a)-[:R]->(x) WHERE NOT x.age <> 30 RETURN x.name AS n"),
        sorted("MATCH (a)-[:R]->(x) WHERE x.age = 30 RETURN x.name AS n"),
    );
    assert_eq!(
        sorted("MATCH (a)-[:R]->(x) WHERE NOT x.age <> 30 RETURN x.name AS n"),
        vec!["keep"]
    );
    // NOT (age >= 40) == age < 40 → keep n1 (30); n2 (40) excluded; absent dropped.
    assert_eq!(
        sorted("MATCH (a)-[:R]->(x) WHERE NOT x.age >= 40 RETURN x.name AS n"),
        sorted("MATCH (a)-[:R]->(x) WHERE x.age < 40 RETURN x.name AS n"),
    );
    // Stacked negation collapses: NOT NOT NOT (age >= 40) == age < 40.
    assert_eq!(
        sorted("MATCH (a)-[:R]->(x) WHERE NOT NOT NOT x.age >= 40 RETURN x.name AS n"),
        vec!["keep"]
    );
}

/// `DISTINCT x, x` (identical projection items) routes to the single-column path and
/// replicates the result column — same distinct set, and the second column is the exact
/// replica of the first (not a separately-keyed composite).
#[test]
fn distinct_identical_columns_replicate() {
    let mut b = Builder::default();
    let m1 = b.node(&["N"], &[("name", s("m1"))]);
    let m2 = b.node(&["N"], &[("name", s("m2"))]);
    let t = b.node(&["N"], &[("name", s("t"))]);
    let src = b.node(&["N"], &[("name", s("src"))]);
    b.edge(src, m1, "R");
    b.edge(src, m2, "R");
    b.edge(src, t, "R");
    b.edge(m1, m2, "R"); // m1 also reaches m2 — forces the endpoint dedup
    let st = b.build();
    let batch = run(
        &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS p, x.name AS q").unwrap(),
        &st,
    );
    let mut c0 = names_of(&batch, 0);
    let c1 = names_of(&batch, 1);
    assert_eq!(
        c0.clone()
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        c0.len(),
        "col0 is distinct"
    );
    c0.sort();
    let mut c1s = c1.clone();
    c1s.sort();
    assert_eq!(c0, vec!["m1", "m2", "t"]);
    assert_eq!(
        c1,
        names_of(&batch, 0),
        "second column replicates the first row-for-row"
    );
    assert_eq!(c0, c1s);
}

/// `X AND (Y OR Z)` where `X ∧ Z` is numerically contradictory drops the Z branch, and a
/// NON-contradictory Z is preserved — both must return the SAME rows as the un-simplified
/// predicate (the simplification is logically exact, not just a fast path).
#[test]
fn contradictory_or_branch_pruned_matches_semantics() {
    let mut b = Builder::default();
    let hi = b.node(&["N"], &[("name", s("hit")), ("age", n(50.0))]);
    let lo = b.node(&["N"], &[("name", s("hit")), ("age", n(10.0))]);
    let ms = b.node(&["N"], &[("name", s("miss")), ("age", n(50.0))]);
    let src = b.node(&["N"], &[("name", s("src"))]);
    b.edge(src, hi, "R");
    b.edge(src, lo, "R");
    b.edge(src, ms, "R");
    let st = b.build();
    let sorted = |q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        v.sort();
        v
    };
    // age>=40 AND age<20 is contradictory → the OR collapses to name='hit'. Only hit(50)
    // satisfies age>=40 AND name='hit'; miss(50) fails the name, lo(10) fails the age.
    assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE (b.age >= 40 AND (b.name = 'hit' OR b.age < 20)) RETURN b.name AS n"),
            vec!["hit"]
        );
    // age>=45 is NOT contradictory with age>=40 → NO pruning: miss(50) satisfies the OR via
    // age>=45, so it MUST survive (guards against over-pruning).
    assert_eq!(
            sorted("MATCH (a)-[:R]->(b) WHERE (b.age >= 40 AND (b.name = 'hit' OR b.age >= 45)) RETURN b.name AS n"),
            vec!["hit", "miss"]
        );
}

/// The fused single-hop `count(*)` over a numeric-filtered typed hop counts exactly the
/// (source, neighbour) PATHS whose neighbour passes the filter — the same value the
/// materialize path yields — for BOTH a universal source label (flat-edge sweep) and a
/// labelled SUBSET (per-source slice), never counting an edge from an out-of-label source.
#[test]
fn fused_hop_count_matches_materialize_value() {
    let count_of = |q: &str, st: &Store| -> f64 {
        let out = run(&crate::gql::parse(q).unwrap(), st);
        match &out.rows.iter().next().expect("one count row")[0] {
            Value::Num(n) => *n,
            other => panic!("not a number: {other:?}"),
        }
    };
    // Non-universal: p* are Person, o0 is Other. Only Person sources' F edges count.
    let mut b = Builder::default();
    let p0 = b.node(&["Person"], &[("age", n(50.0))]);
    let p1 = b.node(&["Person"], &[("age", n(80.0))]);
    let p2 = b.node(&["Person"], &[("age", n(90.0))]);
    let o0 = b.node(&["Other"], &[("age", n(30.0))]);
    b.edge(p0, p1, "F"); // 80 >= 60 ✓
    b.edge(p0, o0, "F"); // 30 >= 60 ✗
    b.edge(p1, p2, "F"); // 90 >= 60 ✓
    b.edge(o0, p1, "F"); // source o0 is NOT a Person → excluded
    let st = b.build();
    let q = "MATCH (a:Person)-[:F]->(b) WHERE b.age >= 60 RETURN count(*) AS c";
    assert_eq!(count_of(q, &st), 2.0, "labelled-subset path");

    // Universal: every node is a Person → the flat-edge sweep fires. Same graph, all Person.
    let mut b2 = Builder::default();
    let q0 = b2.node(&["Person"], &[("age", n(50.0))]);
    let q1 = b2.node(&["Person"], &[("age", n(80.0))]);
    let q2 = b2.node(&["Person"], &[("age", n(90.0))]);
    let q3 = b2.node(&["Person"], &[("age", n(30.0))]);
    b2.edge(q0, q1, "F");
    b2.edge(q0, q3, "F");
    b2.edge(q1, q2, "F");
    b2.edge(q3, q1, "F"); // now q3 IS a Person → this edge counts (target q1 age80 ✓)
    let st2 = b2.build();
    // targets passing age>=60: q0→q1(80✓), q0→q3(30✗), q1→q2(90✓), q3→q1(80✓) = 3.
    assert_eq!(count_of(q, &st2), 3.0, "universal flat-sweep path");
}

/// The 1-hop `count(*)` degree-sum reads raw adjacency lengths when the hop's type
/// set covers EVERY edge type (the `matching_degree` fast path), and still filters
/// by type when it does not — the counts must agree with the per-edge walk in both
/// cases. Regression: this shape (`(a)-[:R]->(b) RETURN count(*)`) walked every edge
/// with a per-edge type check, ~1.8x slower than the TS engine; the raw-length path fixes it
/// WITHOUT changing the value.
#[test]
fn one_hop_count_uses_raw_degree_only_when_type_set_is_universal() {
    let count_of = |q: &str, st: &Store| -> f64 {
        match &run(&crate::gql::parse(q).unwrap(), st)
            .rows
            .iter()
            .next()
            .expect("one count row")[0]
        {
            Value::Num(n) => *n,
            other => panic!("not a number: {other:?}"),
        }
    };
    // Single edge type `R`: `[:R]` covers all types → raw-degree fast path.
    // Also a MULTI-type graph so a `[:R]` hop is a PARTIAL want (must still filter),
    // plus a directed self-loop (kept once by Out) and an unlabeled `-->` (any type).
    let mut b = Builder::default();
    let a = b.node(&["N"], &[]);
    let c = b.node(&["N"], &[]);
    let d = b.node(&["N"], &[]);
    b.edge(a, c, "R");
    b.edge(a, d, "R");
    b.edge(a, a, "R"); // directed self-loop: an out-edge counted once
    b.edge(c, d, "S"); // a SECOND edge type
    let st = b.build();
    // Out over R from every N: a has 3 R-out (c, d, self), c has 0 R-out → 3.
    assert_eq!(
        count_of("MATCH (x:N)-[:R]->(y) RETURN count(*) AS c", &st),
        3.0
    );
    // In over R: c←1 (a), d←1 (a), a←1 (self) → 3.
    assert_eq!(
        count_of("MATCH (x:N)<-[:R]-(y) RETURN count(*) AS c", &st),
        3.0
    );
    // Partial want `[:S]` in a multi-type graph: only the one S edge (c→d) → 1.
    assert_eq!(
        count_of("MATCH (x:N)-[:S]->(y) RETURN count(*) AS c", &st),
        1.0
    );
    // Anonymous edge `-[]->` = any type (empty want) → all 4 edges.
    assert_eq!(
        count_of("MATCH (x:N)-[]->(y) RETURN count(*) AS c", &st),
        4.0
    );
}

/// A two-column DISTINCT with a high-card Str column dedups on the (Str, other) tuple key
/// exactly as the byte-key would: same distinct tuples, first-seen order, and a present-null
/// component collapses with an absent one.
#[test]
fn str_composite_distinct_dedups_correctly() {
    let mut b = Builder::default();
    let src = b.node(&["N"], &[("name", s("src"))]);
    // (alice,30) twice via two neighbours, (alice,40) once, (bob,30) once → 3 distinct tuples.
    let n1 = b.node(&["N"], &[("name", s("alice")), ("age", n(30.0))]);
    let n2 = b.node(&["N"], &[("name", s("alice")), ("age", n(30.0))]);
    let n3 = b.node(&["N"], &[("name", s("alice")), ("age", n(40.0))]);
    let n4 = b.node(&["N"], &[("name", s("bob")), ("age", n(30.0))]);
    b.edge(src, n1, "R");
    b.edge(src, n2, "R");
    b.edge(src, n3, "R");
    b.edge(src, n4, "R");
    let st = b.build();
    let out = run(
        &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS n, x.age AS g").unwrap(),
        &st,
    );
    let mut tuples: Vec<(String, String)> = out
        .rows
        .iter()
        .map(|r| (format!("{:?}", r[0]), format!("{:?}", r[1])))
        .collect();
    assert_eq!(tuples.len(), 3, "three distinct (name, age) tuples");
    tuples.sort();
    assert_eq!(
        tuples,
        vec![
            ("Str(\"alice\")".into(), "Num(30.0)".into()),
            ("Str(\"alice\")".into(), "Num(40.0)".into()),
            ("Str(\"bob\")".into(), "Num(30.0)".into()),
        ]
    );
}

/// The fused numeric-filtered projection returns the SAME rows (as a multiset) as the
/// general materialize+filter+gather+project path — same survivors, same projected values.
#[test]
fn fused_hop_projection_matches_general() {
    let mut b = Builder::default();
    let src = b.node(&["Person"], &[("name", s("src"))]);
    for (sc, nm) in [
        (10.0, "a"),
        (60.0, "b"),
        (30.0, "c"),
        (90.0, "d"),
        (20.0, "e"),
    ] {
        let t = b.node(&["Person"], &[("score", n(sc)), ("name", s(nm))]);
        b.edge(src, t, "F");
    }
    let st = b.build();
    // score < 50 keeps a(10,→'a'), c(30,→'c'), e(20,→'e'); b,d dropped.
    let mut got = names_of(
        &run(
            &crate::gql::parse("MATCH (a:Person)-[:F]->(b) WHERE b.score < 50 RETURN b.name AS n")
                .unwrap(),
            &st,
        ),
        0,
    );
    got.sort();
    assert_eq!(got, vec!["a", "c", "e"]);
    // A projected expression (not a bare prop) over the survivor frontier still works.
    let mut up = names_of(
        &run(
            &crate::gql::parse(
                "MATCH (a:Person)-[:F]->(b) WHERE b.score < 50 RETURN upper(b.name) AS n",
            )
            .unwrap(),
            &st,
        ),
        0,
    );
    up.sort();
    assert_eq!(up, vec!["A", "C", "E"]);
}

/// The fused mask-aggregate (count/sum/min/max over a complex-predicate typed hop) returns
/// the SAME scalar as the general materialize+filter+aggregate path — checked against the
/// un-fused plan on the same graph, so any row-order or skip divergence would show.
#[test]
fn fused_mask_agg_matches_general_aggregate() {
    let mut b = Builder::default();
    let src = b.node(&["Person"], &[("name", s("src"))]);
    // targets with (score, age); the OR (score >= 50 OR age < 5) keeps some, drops others.
    for (sc, ag) in [
        (10.0, 3.0),
        (60.0, 90.0),
        (55.0, 2.0),
        (20.0, 40.0),
        (99.0, 10.0),
    ] {
        let t = b.node(&["Person"], &[("score", n(sc)), ("age", n(ag))]);
        b.edge(src, t, "F");
    }
    let st = b.build();
    let scalar = |q: &str| -> f64 {
        match &run(&crate::gql::parse(q).unwrap(), &st)
            .rows
            .iter()
            .next()
            .expect("one row")[0]
        {
            Value::Num(n) => *n,
            other => panic!("not a number: {other:?}"),
        }
    };
    // Kept scores: 60(≥50), 55(≥50 & age<5), 99(≥50), 10(age<5) → {60,55,99,10}. 20 dropped.
    let base = "MATCH (a:Person)-[:F]->(b) WHERE (b.score >= 50 OR b.age < 5)";
    assert_eq!(scalar(&format!("{base} RETURN count(*) AS c")), 4.0);
    assert_eq!(scalar(&format!("{base} RETURN sum(b.score) AS v")), 224.0);
    assert_eq!(scalar(&format!("{base} RETURN max(b.score) AS v")), 99.0);
    assert_eq!(scalar(&format!("{base} RETURN min(b.score) AS v")), 10.0);
}

/// A single-type hop over the per-type CSR returns the type's neighbours in the SAME
/// order (and multiplicity) as the flat scan filtering on the edge type — the byte-identity
/// the partition must preserve. Interleaves F and R out-edges from one source.
#[test]
fn per_type_hop_preserves_flat_scan_order() {
    let mut b = Builder::default();
    let src = b.node(&["N"], &[("name", s("src"))]);
    let f1 = b.node(&["N"], &[("name", s("f1"))]);
    let r1 = b.node(&["N"], &[("name", s("r1"))]);
    let f2 = b.node(&["N"], &[("name", s("f2"))]);
    let r2 = b.node(&["N"], &[("name", s("r2"))]);
    let f3 = b.node(&["N"], &[("name", s("f3"))]);
    // Interleave the two types in insertion order: F R F R F.
    b.edge(src, f1, "F");
    b.edge(src, r1, "R");
    b.edge(src, f2, "F");
    b.edge(src, r2, "R");
    b.edge(src, f3, "F");
    let st = b.build();
    // The F hop must yield f1, f2, f3 in insertion order (ORDER matters — no sort).
    let names = names_of(
        &run(
            &crate::gql::parse("MATCH (a)-[:F]->(b) RETURN b.name AS n").unwrap(),
            &st,
        ),
        0,
    );
    assert_eq!(names, vec!["f1", "f2", "f3"]);
    // And the R hop yields r1, r2 in insertion order.
    let rnames = names_of(
        &run(
            &crate::gql::parse("MATCH (a)-[:R]->(b) RETURN b.name AS n").unwrap(),
            &st,
        ),
        0,
    );
    assert_eq!(rnames, vec!["r1", "r2"]);
}

/// `NOT (col STARTS/ENDS/CONTAINS lit)` over a hop keeps the complement via the raw
/// scan, and — critically — an ABSENT cell stays dropped (UNKNOWN under the inner
/// search, so NOT-UNKNOWN is UNKNOWN), matching the general eval_mask path exactly.
#[test]
fn not_strsearch_keep_matches_general() {
    let mut b = Builder::default();
    let e1 = b.node(&["N"], &[("name", s("alpha")), ("city", s("oslo"))]);
    let e2 = b.node(&["N"], &[("name", s("beta")), ("city", s("bergen"))]);
    let e3 = b.node(&["N"], &[("name", s("gamma"))]); // city ABSENT
    let src = b.node(&["N"], &[("name", s("src"))]);
    b.edge(src, e1, "R");
    b.edge(src, e2, "R");
    b.edge(src, e3, "R");
    let st = b.build();
    let sorted = |q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        v.sort();
        v
    };
    // ENDS WITH 'ta': only beta; NOT keeps alpha, gamma.
    assert_eq!(
        sorted("MATCH (a)-[:R]->(b) WHERE NOT b.name ENDS WITH 'ta' RETURN b.name AS n"),
        vec!["alpha", "gamma"]
    );
    // STARTS WITH 'a': only alpha; NOT keeps beta, gamma.
    assert_eq!(
        sorted("MATCH (a)-[:R]->(b) WHERE NOT b.name STARTS WITH 'a' RETURN b.name AS n"),
        vec!["beta", "gamma"]
    );
    // CONTAINS 'mm': only gamma; NOT keeps alpha, beta.
    assert_eq!(
        sorted("MATCH (a)-[:R]->(b) WHERE NOT b.name CONTAINS 'mm' RETURN b.name AS n"),
        vec!["alpha", "beta"]
    );
    // NOT city CONTAINS 'o': oslo fails, bergen keeps (beta), gamma's city ABSENT is
    // UNKNOWN → dropped (not resurrected by NOT).
    assert_eq!(
        sorted("MATCH (a)-[:R]->(b) WHERE NOT b.city CONTAINS 'o' RETURN b.name AS n"),
        vec!["beta"]
    );
}

/// A conjunction seeded on its equality conjunct (`c.name = 'hit' AND c.age > 50`)
/// applies the remaining conjuncts as a residual filter over the seeded rows, so it
/// returns exactly the forward walk's rows — the seed bucket holds two 'hit' nodes
/// and only the one passing `age > 50` (and the paths reaching it) survive.
#[test]
fn reverse_seed_conjunction_residual_matches_forward() {
    let mut b = Builder::default();
    let t1 = b.node(&["N"], &[("name", s("hit")), ("age", n(99.0))]);
    let t2 = b.node(&["N"], &[("name", s("hit")), ("age", n(10.0))]); // same name, fails age>50
    let s1 = b.node(&["N"], &[("name", s("s1"))]);
    let s2 = b.node(&["N"], &[("name", s("s2"))]);
    b.edge(s1, t1, "R");
    b.edge(s2, t2, "R");
    b.edge(s1, t2, "R"); // s1 also reaches the filtered-out target
    let mut st = b.build();
    let q = "MATCH (a)-[:R]->(c) WHERE c.name = 'hit' AND c.age > 50 RETURN a.name AS a";
    let sorted = |st: &Store| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), st), 0);
        v.sort();
        v
    };
    // Forward (no index): only s1→t1 survives (t1.age = 99).
    assert_eq!(sorted(&st), vec!["s1"]);
    // Index on: seed name='hit' (bucket {t1,t2}), residual age>50 keeps only t1's path.
    st.create_index("name");
    assert_eq!(sorted(&st), vec!["s1"]);
}

/// The reverse-seed only fires when the target bucket is smaller than the source
/// scan. A non-selective endpoint (every node named "x") must keep the forward
/// walk — and either way the rows are identical.
#[test]
fn reverse_seed_declines_non_selective_endpoint() {
    let mut b = Builder::default();
    let ids: Vec<u32> = (0..6)
        .map(|_| b.node(&["N"], &[("name", s("x"))]))
        .collect();
    for w in ids.windows(2) {
        b.edge(w[0], w[1], "R");
    }
    let mut st = b.build();
    let q = "MATCH (a)-[:R]->(b)-[:R]->(c) WHERE c.name = 'x' RETURN a.name AS a";
    let fwd = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
    st.create_index("name"); // bucket = all 6 nodes >= source, so no flip
    let idx = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
    let (mut fwd, mut idx) = (fwd, idx);
    fwd.sort();
    idx.sort();
    assert_eq!(fwd, idx);
}

/// `r.vf <= X AND r.vt >= Y` fuses to an `IntervalExpand` whose scan fallback now
/// compares TEMPORAL bounds (not just numeric) via the value contract — so a
/// "contains the window" query over date edges returns the covering edges instead
/// of nothing (the numeric-only guard used to skip every temporal edge).
#[test]
fn interval_expand_scan_handles_temporal_bounds() {
    let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"id\":\"covers\",\"vf\":{\"@date\":\"2024-01-01\"},\"vt\":{\"@date\":\"2024-12-01\"}}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"id\":\"exact\",\"vf\":{\"@date\":\"2024-04-01\"},\"vt\":{\"@date\":\"2024-08-01\"}}}\n",
            "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"id\":\"disjoint\",\"vf\":{\"@date\":\"2024-01-01\"},\"vt\":{\"@date\":\"2024-03-01\"}}}"
        );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let q = "MATCH ()-[r:R]->() WHERE r.vf <= DATE '2024-04-01' AND r.vt >= DATE '2024-08-01' RETURN r.id AS id ORDER BY id";
    let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
    // The optimizer must have fused it into the interval hop.
    assert!(has_interval_expand(&plan));
    assert_eq!(names_of(&run(&plan, &store), 0), vec!["covers", "exact"]);
}

/// Does the plan tree contain an `IntervalExpand` (the fused interval hop)?
fn has_interval_expand(p: &Plan) -> bool {
    match p {
        Plan::IntervalExpand { .. } => true,
        Plan::Sample { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::EdgeVertex { input, .. }
        | Plan::Expand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Filter { input, .. }
        | Plan::Aggregate { input, .. }
        | Plan::OrderPage { input, .. }
        | Plan::Project { input, .. }
        | Plan::Distinct { input }
        | Plan::SortLocal { input, .. }
        | Plan::Update { input, .. }
        | Plan::UpdateReturn { input, .. } => has_interval_expand(input),
        Plan::Join { left, right, .. } => has_interval_expand(left) || has_interval_expand(right),
        _ => false,
    }
}

fn interval_store() -> Store {
    // Emp 0 with 5 HELD edges to role 1, intervals [d, d+2] for d in 0..5.
    let mut b = Builder::default();
    b.node(&["Emp"], &[]);
    b.node(&["Role"], &[]);
    let mut st = b.build();
    for d in 0..5u32 {
        let e = st.add_edge(0, 1, "HELD");
        st.set_edge_prop(e, "vf", n(f64::from(d)));
        st.set_edge_prop(e, "vt", n(f64::from(d) + 2.0));
    }
    st
}

/// The optimizer fuses `r.vf <= X AND r.vt >= Y` over a bound-edge hop into an
/// `IntervalExpand`, which returns the SAME rows via the scan fallback (no
/// index) and via the index seek — and both equal the hand-computed answer.
#[test]
fn interval_expand_fuses_and_matches_scan_and_seek() {
    use crate::opt::optimize;
    let mut st = interval_store();
    // As of t=3: [0,2] no, [1,3] yes, [2,4] yes, [3,5] yes, [4,6] no → 3.
    let q = "MATCH (p:Emp)-[r:HELD]->(x) WHERE r.vf <= 3 AND r.vt >= 3 RETURN count(*) AS c";
    let plan = optimize(crate::gql::parse(q).unwrap());
    assert!(
        has_interval_expand(&plan),
        "optimizer did not fuse: {plan:?}"
    );
    // scan fallback (no interval index yet)
    assert!(matches!(run(&plan, &st).rows[0][0], Value::Num(x) if x == 3.0));
    // index seek (same plan, index present)
    st.create_interval_index("vf", "vt");
    assert!(matches!(run(&plan, &st).rows[0][0], Value::Num(x) if x == 3.0));

    // Row-level equivalence: the matching intervals' vf are {1,2,3}, seek == scan.
    let rq = "MATCH (p:Emp)-[r:HELD]->(x) WHERE r.vf <= 3 AND r.vt >= 3 RETURN r.vf AS f";
    let rplan = optimize(crate::gql::parse(rq).unwrap());
    let mut seek: Vec<String> = names_of(&run(&rplan, &st), 0);
    seek.sort();
    let scan_only = interval_store(); // fresh, no index
    let mut scan: Vec<String> = names_of(&run(&rplan, &scan_only), 0);
    scan.sort();
    assert_eq!(seek, scan);
    // vf of the matching intervals ([1,3],[2,4],[3,5]) — `names_of` renders a
    // Num via its debug form.
    assert_eq!(seek, vec!["Num(1.0)", "Num(2.0)", "Num(3.0)"]);
}

/// Grouping by an EDGE property counts per distinct edge-prop value — the
/// bound edge sits at slot W and the endpoint node at W+1, so the count
/// fast-path must not read the edge key as an (absent) node property. (The
/// differential fuzzer found this bucketing every row under one NULL group.)
#[test]
fn group_by_edge_property_counts_per_value() {
    let mut b = Builder::default();
    let x = b.node(&["N"], &[]);
    let y = b.node(&["N"], &[]);
    let z = b.node(&["N"], &[]);
    b.edge(x, y, "R");
    b.edge(x, z, "R");
    b.edge(y, z, "R");
    let mut store = b.build();
    // Set weights: two edges w=2, one w=7 (eids 0,1,2 in insertion order).
    store.set_edge_prop(0, "w", n(2.0));
    store.set_edge_prop(1, "w", n(2.0));
    store.set_edge_prop(2, "w", n(7.0));
    let plan = crate::gql::parse("MATCH (a:N)-[r:R]->(b) RETURN r.w AS w, count(*) AS c").unwrap();
    // Group {2.0 → 2 edges, 7.0 → 1 edge}, order-independent.
    let mut got: Vec<(String, f64)> = run(&plan, &store)
        .rows
        .iter()
        .map(|row| (format!("{:?}", row[0]), num(&row[1])))
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        vec![("Num(2.0)".to_string(), 2.0), ("Num(7.0)".to_string(), 1.0)]
    );
}

/// K4: computed NaN/Inf are KEPT in the result value (matching the TS engine, so a
/// caller can detect the signal), and coerced to null only at JSON egress.
#[test]
fn nan_and_inf_kept_in_results_coerced_at_egress() {
    let mut b = Builder::default();
    b.node(&["N"], &[("a", n(-4.0))]);
    let store = b.build();
    let val = |e: &str| {
        let q = format!("MATCH (x:N) RETURN {e} AS v");
        run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
    };
    assert!(matches!(val("sqrt(x.a)"), Value::Num(y) if y.is_nan())); // sqrt(-4) → NaN kept
    assert!(matches!(val("sqrt(x.a) + 1"), Value::Num(y) if y.is_nan())); // NaN propagates
    assert!(matches!(val("power(10, 400)"), Value::Num(y) if y.is_infinite())); // overflow → Inf
                                                                                // But the JSON egress renders both as null (no JSON form for NaN/Inf).
    let ndjson = crate::ndjson::to_ndjson(&store);
    assert!(!ndjson.contains("NaN") && !ndjson.to_lowercase().contains("inf"));
}

/// Newly added scalar functions (K6 casts, K8 nullif, K9 math/constants,
/// K5 size-on-string) match hand-computed values. One node with a=4, b="Carol".
#[test]
fn added_scalar_functions() {
    let mut b = Builder::default();
    b.node(&["N"], &[("a", n(4.0)), ("b", s("Carol"))]);
    let store = b.build();
    let val = |e: &str| -> Value {
        let q = format!("MATCH (n:N) RETURN {e} AS v");
        run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
    };
    let num = |e: &str| match val(e) {
        Value::Num(x) => x,
        o => panic!("{e} → {o:?}"),
    };
    // constants + math (native libm, matching the TS engine)
    assert!((num("pi()") - std::f64::consts::PI).abs() < 1e-12);
    assert!((num("e()") - std::f64::consts::E).abs() < 1e-12);
    assert_eq!(num("power(2, 10)"), 1024.0);
    assert_eq!(num("log(2, 8)"), 3.0); // log base 2 of 8 = ln(8)/ln(2)
    assert_eq!(num("mod(7, 3)"), 1.0);
    assert!((num("ln(e())") - 1.0).abs() < 1e-12);
    assert_eq!(num("degrees(pi())").round(), 180.0);
    // casts (NULL on a non-convertible input; a BOOLEAN converts to 1/0)
    assert_eq!(num("to_integer('7')"), 7.0);
    assert_eq!(num("to_integer(4.9)"), 4.0);
    assert_eq!(num("to_float('2.5')"), 2.5);
    assert!(matches!(val("to_string(n.a)"), Value::Str(x) if &*x == "4"));
    assert!(matches!(val("to_boolean('true')"), Value::Bool(true)));
    assert!(matches!(val("to_boolean(0)"), Value::Bool(false)));
    assert!(val("to_integer('nope')").is_null());
    assert_eq!(num("to_integer(true)"), 1.0); // explicit conversion coerces bool → 1/0
                                              // nullif
    assert!(val("nullif(n.a, 4)").is_null());
    assert_eq!(num("nullif(n.a, 5)"), 4.0);
    // size / char_length on a string (K5)
    assert_eq!(num("size(n.b)"), 5.0);
    // `cardinality` is the ISO/SQL alias for `size` (a reserved word AND a function).
    assert_eq!(num("cardinality(n.b)"), 5.0);
    assert_eq!(num("cardinality([1, 2, 3])"), 3.0);
    assert_eq!(num("char_length(n.b)"), 5.0);
}

/// Subscript `base[index]` (ISO 0-based) over a list literal, record and map:
/// in-range element, out-of-range / negative / non-integer → NULL, null-safe.
#[test]
fn subscript_list_record_map() {
    let mut b = Builder::default();
    b.node(&["N"], &[("z", n(1.0))]);
    let store = b.build();
    let num = |e: &str| -> f64 {
        let q = format!("MATCH (x:N) RETURN {e} AS v");
        match run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("{e} → {o:?}"),
        }
    };
    let isnull = |e: &str| -> bool {
        let q = format!("MATCH (x:N) RETURN {e} AS v");
        run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].is_null()
    };
    assert_eq!(num("[10,20,30][0]"), 10.0);
    assert_eq!(num("[10,20,30][2]"), 30.0);
    assert!(isnull("[10,20,30][9]")); // out of range
    assert!(isnull("[10,20,30][-1]")); // negative
    assert!(isnull("[10,20,30][1.5]")); // non-integer
    assert_eq!(num("{a:1,b:2}['b']"), 2.0); // record field by string key
    assert!(isnull("{a:1,b:2}['zzz']")); // missing field
}

/// `edges(p)[i]` / `nodes(p)[i]` keep element typing so a following `.prop`
/// resolves the edge/node property. Path n0 -R(w=5)-> n1 -R(w=7)-> n2.
#[test]
fn subscript_path_element_property() {
    let nd = concat!(
        "{\"id\":\"n0\",\"labels\":[\"N\"],\"props\":{\"id\":\"n0\"}}\n",
        "{\"id\":\"n1\",\"labels\":[\"N\"],\"props\":{\"id\":\"n1\"}}\n",
        "{\"id\":\"n2\",\"labels\":[\"N\"],\"props\":{\"id\":\"n2\"}}\n",
        "{\"id\":\"e1\",\"from\":\"n0\",\"to\":\"n1\",\"type\":\"R\",\"props\":{\"w\":5}}\n",
        "{\"id\":\"e2\",\"from\":\"n1\",\"to\":\"n2\",\"type\":\"R\",\"props\":{\"w\":7}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let val = |e: &str| -> Value {
        let q = format!(
            "MATCH p = ANY SHORTEST (a:N {{id:'n0'}})-[:R]->*(b:N {{id:'n2'}}) RETURN {e} AS v"
        );
        run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
    };
    assert!(matches!(val("edges(p)[0].w"), Value::Num(x) if x == 5.0)); // first edge property
    assert!(matches!(val("edges(p)[1].w"), Value::Num(x) if x == 7.0)); // second edge property
    assert!(val("edges(p)[9].w").is_null()); // out-of-range edge → NULL prop
    assert!(matches!(val("nodes(p)[2].id"), Value::Str(x) if &*x == "n2"));
    assert!(val("nodes(p)[0].nope").is_null()); // missing node property
    assert!(matches!(
        val("edges(p)[1].w > edges(p)[0].w"),
        Value::Bool(true)
    ));
}

/// `VALUE { MATCH (a)-[:R]->(b) RETURN count(*) }` is a correlated count subquery
/// (a degree), lowering to the same result as `COUNT { (a)-[:R]->(b) }`.
#[test]
fn value_count_subquery() {
    let nd = concat!(
        "{\"id\":\"dave\",\"labels\":[\"P\"],\"props\":{\"id\":\"dave\"}}\n",
        "{\"id\":\"carol\",\"labels\":[\"P\"],\"props\":{\"id\":\"carol\"}}\n",
        "{\"id\":\"x\",\"labels\":[\"P\"],\"props\":{\"id\":\"x\"}}\n",
        "{\"id\":\"y\",\"labels\":[\"P\"],\"props\":{\"id\":\"y\"}}\n",
        "{\"from\":\"dave\",\"to\":\"x\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
        "{\"from\":\"dave\",\"to\":\"y\",\"labels\":[\"KNOWS\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let deg = |id: &str| -> f64 {
        let q = format!(
                "MATCH (a:P) WHERE a.id='{id}' RETURN VALUE {{ MATCH (a)-[:KNOWS]->(b) RETURN count(*) }} AS deg"
            );
        let plan = crate::opt::optimize_indexed(crate::gql::parse(&q).unwrap(), &store);
        match run(&plan, &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("{o:?}"),
        }
    };
    assert_eq!(deg("dave"), 2.0); // dave knows x and y
    assert_eq!(deg("carol"), 0.0); // carol knows no one
}

/// Multi-label edges: an edge's type is its FIRST label; the rest are secondary
/// labels a `-[:label]->` hop must still match. `a-[:X,:Y]->b`, `a-[:Y]->c`.
#[test]
fn multi_label_edge_matching() {
    // a -r0[X,Y]-> b ; a -r1[Y]-> c ; b -r2[Z,Y]-> c
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"r0\",\"from\":\"a\",\"to\":\"b\",\"labels\":[\"X\",\"Y\"],\"props\":{}}\n",
        "{\"id\":\"r1\",\"from\":\"a\",\"to\":\"c\",\"labels\":[\"Y\"],\"props\":{}}\n",
        "{\"id\":\"r2\",\"from\":\"b\",\"to\":\"c\",\"labels\":[\"Z\",\"Y\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    assert!(store.has_multi_label_edges());
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // `:Y` reaches every edge (all three carry Y, two only as a secondary label).
    assert_eq!(
        ids("MATCH (a:N)-[:Y]->(b) RETURN b.id AS x"),
        vec!["b", "c", "c"]
    );
    // `:X` only r0 (its primary), `:Z` only r2 (its primary).
    assert_eq!(ids("MATCH (a:N)-[:X]->(b) RETURN b.id AS x"), vec!["b"]);
    assert_eq!(ids("MATCH (a:N)-[:Z]->(b) RETURN b.id AS x"), vec!["c"]);
    // A var-length `:Y` hop crosses secondary-label edges too: a-Y->b-Y->c.
    assert!(ids("MATCH (a:N {id:'a'})-[:Y]->{2}(b) RETURN b.id AS x").contains(&"c".to_string()));
    // type(edge) is the FIRST label, not a secondary one.
    let ty = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    assert_eq!(
        ty("MATCH (a:N)-[e:Y]->(b) RETURN type(e) AS t"),
        vec!["X", "Y", "Z"]
    );
}

/// Edge-label NEGATION `-[:!T]->` matches any edge whose type is NOT `T` (the
/// complement of the named types), and `:!(A|B)` negates a disjunction.
#[test]
fn edge_label_negation() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"P\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"P\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"s\",\"labels\":[\"P\"],\"props\":{\"id\":\"s\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
        "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"CREATED\"],\"props\":{}}\n",
        "{\"from\":\"a\",\"to\":\"s\",\"labels\":[\"LIKES\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // NOT CREATED → the KNOWS and LIKES targets.
    assert_eq!(
        ids("MATCH (a:P {id:'a'})-[:!CREATED]->(x) RETURN x.id AS x"),
        vec!["b", "s"]
    );
    // NOT (CREATED|LIKES) → only the KNOWS target.
    assert_eq!(
        ids("MATCH (a:P {id:'a'})-[:!(CREATED|LIKES)]->(x) RETURN x.id AS x"),
        vec!["b"]
    );
    // A negated unknown type excludes nothing → every out-edge.
    assert_eq!(
        ids("MATCH (a:P {id:'a'})-[:!NOSUCH]->(x) RETURN x.id AS x"),
        vec!["b", "c", "s"]
    );
}

/// Inline edge properties on a plain var-length hop filter every edge on the
/// path. a-e(10)->b-e(20)->c-e(5)->d; only b->c has amt 20.
#[test]
fn var_length_inline_edge_props() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":5.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // From a, no outgoing amt=20 edge → no path.
    assert!(ids("MATCH (a:N {id:'a'})-[:R {amt:20.0}]->{1,3}(x) RETURN x.id AS id").is_empty());
    // From b, b->c has amt 20 → x = c (c->d is amt 5, excluded).
    assert_eq!(
        ids("MATCH (b:N {id:'b'})-[:R {amt:20.0}]->{1,3}(x) RETURN x.id AS id"),
        vec!["c"]
    );
}

/// A per-hop edge WHERE on a plain var-length hop filters each hop's edge.
/// a-e(20)->b-e(5)->c: e.amt>=10 admits only a->b.
#[test]
fn plain_var_length_per_hop_where() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":5.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // e.amt >= 10 blocks b->c → only a->b reaches b.
    assert_eq!(
        ids("MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10]->{1,3}(x) RETURN x.id AS id"),
        vec!["b"]
    );
    // e.amt >= 1 admits all → b, c.
    assert_eq!(
        ids("MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 1]->{1,3}(x) RETURN x.id AS id"),
        vec!["b", "c"]
    );
}

/// A per-hop edge WHERE may also reference the hop's SOURCE variable: `(a)-[e
/// WHERE a.id = … AND e.amt >= …]->{1,3}(x)`. The anchor `a` maps to the path
/// source at eval time, so a true condition admits the walk and a false one
/// blocks every path.
#[test]
fn per_hop_where_references_outer_source() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // a.id = 'a' holds → both hops admitted (b, c).
    assert_eq!(
            ids("MATCH (a:N {id:'a'})-[e:R WHERE e.amt >= 10 AND a.id = 'a']->{1,3}(x) RETURN x.id AS id"),
            vec!["b", "c"]
        );
    // a.id = 'zzz' is false → no path survives.
    assert!(
        ids("MATCH (a:N {id:'a'})-[e:R WHERE a.id = 'zzz']->{1,3}(x) RETURN x.id AS id").is_empty()
    );
}

/// Graph-element predicates: IS DIRECTED, IS SOURCE/DESTINATION OF, ALL_DIFFERENT,
/// SAME — three-valued over element identity (a null operand → NULL).
#[test]
fn graph_element_predicates() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let row = |q: &str| -> Vec<Value> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store).rows[0].to_vec()
    };
    let r = row(
        "MATCH (a:N {id:'a'})-[e:R]->(b:N {id:'b'}) RETURN e IS DIRECTED AS d, \
             a IS SOURCE OF e AS asrc, b IS DESTINATION OF e AS bdst, b IS SOURCE OF e AS bsrc, \
             ALL_DIFFERENT(a, b) AS diff, SAME(a, a) AS saa, SAME(a, b) AS sab",
    );
    assert!(matches!(r[0], Value::Bool(true))); // e IS DIRECTED
    assert!(matches!(r[1], Value::Bool(true))); // a IS SOURCE OF e
    assert!(matches!(r[2], Value::Bool(true))); // b IS DESTINATION OF e
    assert!(matches!(r[3], Value::Bool(false))); // b IS SOURCE OF e
    assert!(matches!(r[4], Value::Bool(true))); // ALL_DIFFERENT(a,b)
    assert!(matches!(r[5], Value::Bool(true))); // SAME(a,a)
    assert!(matches!(r[6], Value::Bool(false))); // SAME(a,b)
                                                 // Three-valued: a null element → NULL.
    let r = row("MATCH (a:N {id:'a'}) OPTIONAL MATCH (a)-[:NOSUCH]->(m) \
             RETURN m IS DIRECTED AS d, ALL_DIFFERENT(a, m) AS ad");
    assert!(r[0].is_null());
    assert!(r[1].is_null());
}

/// Bare ALL/ANY selectors: ALL is the default (every path — a duplicate endpoint
/// per path), ANY keeps one per endpoint (dedup). Diamond a->b->d, a->c->d.
#[test]
fn bare_all_any_selectors() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"b\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // ALL: d reached by two 2-hop paths → d appears twice; b, c once each.
    assert_eq!(
        ids("MATCH ALL (a:N {id:'a'})-[:R]->{1,2}(x) RETURN x.id AS id"),
        vec!["b", "c", "d", "d"]
    );
    // ANY: one per endpoint → b, c, d once each.
    assert_eq!(
        ids("MATCH ANY (a:N {id:'a'})-[:R]->{1,2}(x) RETURN x.id AS id"),
        vec!["b", "c", "d"]
    );
}

/// FOR..IN list unwind: literal list, ordinal (1-based ORDINALITY / 0-based
/// OFFSET), null/empty → no rows, a scalar singleton, and multiplying a MATCH.
#[test]
fn for_in_unwind() {
    let mut b = Builder::default();
    b.node(&["P"], &[("name", s("marko"))]);
    let store = b.build();
    let nums = |q: &str| -> Vec<f64> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store)
            .rows
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                ref o => panic!("{o:?}"),
            })
            .collect()
    };
    assert_eq!(nums("FOR x IN [1, 2, 3] RETURN x"), vec![1.0, 2.0, 3.0]);
    // ORDINALITY is 1-based, OFFSET 0-based.
    assert_eq!(
        nums("FOR x IN ['a','b'] WITH ORDINALITY i RETURN i"),
        vec![1.0, 2.0]
    );
    assert_eq!(
        nums("FOR x IN ['a','b'] WITH OFFSET i RETURN i"),
        vec![0.0, 1.0]
    );
    // null and empty list → no rows; a non-list scalar → one row.
    assert_eq!(nums("FOR x IN null RETURN x").len(), 0);
    assert_eq!(nums("FOR x IN [] RETURN x").len(), 0);
    assert_eq!(nums("FOR x IN 5 RETURN x"), vec![5.0]);
    // Multiplies a prior MATCH (one row per (match, element)).
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("MATCH (p:P) FOR t IN ['x','y'] RETURN t").unwrap(),
        &store,
    );
    assert_eq!(run(&plan, &store).rows.len(), 2);
}

/// A FOR-driven fresh-variable `OPTIONAL MATCH (p:Label {k: expr})` is a left-outer
/// correlated scan: each unwound name finds the matching node (its age), or a NULL
/// node when none matches.
#[test]
fn for_driven_optional_scan() {
    let mut b = Builder::default();
    b.node(&["Person"], &[("name", s("josh")), ("age", n(32.0))]);
    b.node(&["Person"], &[("name", s("marko")), ("age", n(29.0))]);
    let store = b.build();
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse(
            "FOR name IN ['josh', 'nobody'] \
                 OPTIONAL MATCH (p:Person {name: name}) RETURN name, p.age",
        )
        .unwrap(),
        &store,
    );
    let rows: Vec<(String, Value)> = run(&plan, &store)
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Str(nm) => (nm.to_string(), r[1].clone()),
            o => panic!("{o:?}"),
        })
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, "josh");
    assert!(matches!(rows[0].1, Value::Num(x) if x == 32.0));
    assert_eq!(rows[1].0, "nobody");
    assert!(rows[1].1.is_null());
}

/// A single-outer-rep endpoint-only nested group `( ()-[:R]->{1,3}() ){1} (t)`
/// desugars to a var-length {1,3}. Chain a->b->c->d.
#[test]
fn nested_endpoint_only_single_rep() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("MATCH (s:N {id:'a'}) ( ()-[:R]->{1,3}() ){1} (t) RETURN t.id AS id")
            .unwrap(),
        &store,
    );
    let mut v = names_of(&run(&plan, &store), 0);
    v.sort();
    assert_eq!(v, vec!["b", "c", "d"]); // reachable in 1..3 hops
                                        // A MULTI-repetition endpoint-only nested group now enumerates each
                                        // rep-decomposition (`Plan::NestedGroup`): `( ()-[:R]->{1,2}() ){2}` from a on
                                        // the chain a->b->c->d = 2 outer reps, each 1-2 hops (trail). Endpoints (with
                                        // multiplicity, one row per decomposition): 2+2=c, 2+... only c and d reach.
                                        // a->b then b->c (c), a->b then b->c->d (d), a->b->c then c->d (d).
    let plan2 = crate::opt::optimize_indexed(
        crate::gql::parse("MATCH (s:N {id:'a'}) ( ()-[:R]->{1,2}() ){2} (t) RETURN t.id AS id")
            .unwrap(),
        &store,
    );
    let mut v2 = names_of(&run(&plan2, &store), 0);
    v2.sort();
    assert_eq!(v2, vec!["c", "d", "d"]);
}

/// A repeated pattern variable on a var-length landing is an equality join: an
/// EXISTS correlated on BOTH anchors `EXISTS { (a)-[:R]->+(b) }`, and a cycle
/// `(a)-[:R]->{1,3}(a)`. Chain a->b->c (no cycle back to a).
#[test]
fn repeated_variable_landing_equality() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let rows = |q: &str| -> usize {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store).rows.len()
    };
    // a reaches c → EXISTS true → 1 row.
    assert_eq!(
            rows("MATCH (a:N {id:'a'}), (b:N {id:'c'}) WHERE EXISTS { MATCH (a)-[:R]->+(b) } RETURN 1 AS x"),
            1
        );
    // a does NOT reach a (no cycle) → EXISTS false → 0 rows.
    assert_eq!(
            rows("MATCH (a:N {id:'a'}), (b:N {id:'a'}) WHERE EXISTS { MATCH (a)-[:R]->+(b) } RETURN 1 AS x"),
            0
        );
    // A named cycle `(a)…(a)`: a can't return to a → no path.
    assert_eq!(
        rows("MATCH p = SIMPLE (a:N {id:'a'})-[:R]->{1,3}(a) RETURN path_length(p) AS len"),
        0
    );
}

/// An uncorrelated VALUE subquery runs a self-contained body once: a constant
/// (`VALUE { RETURN 1+2 }`) or a global aggregate (`VALUE { MATCH (n) RETURN
/// count(*) }`).
#[test]
fn uncorrelated_value_subquery() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{}}\n",
        "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{}}\n",
        "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let num = |q: &str| -> f64 {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        match run(&plan, &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("{o:?}"),
        }
    };
    assert_eq!(num("RETURN VALUE { RETURN 1 + 2 } AS v"), 3.0);
    assert_eq!(
        num("RETURN VALUE { MATCH (n:Person) RETURN count(*) } AS c"),
        3.0
    );
}

/// An uncorrelated multi-pattern EXISTS `EXISTS { MATCH (x:N) MATCH (y:M) }` is a
/// self-contained cross-join existence check, run once and broadcast.
#[test]
fn uncorrelated_multi_match_exists() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"M\"],\"props\":{\"id\":\"b\"}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let b = |q: &str| -> bool {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        matches!(run(&plan, &store).rows[0][0], Value::Bool(true))
    };
    // N and M both non-empty → true.
    assert!(b("RETURN EXISTS { MATCH (x:N) MATCH (y:M) } AS e"));
    // Z is empty → the cross-join is empty → false.
    assert!(!b("RETURN EXISTS { MATCH (x:N) MATCH (y:Z) } AS e"));
    // Per-clause WHERE: x.id='a' and y.id='b' both match → true.
    assert!(b(
        "RETURN EXISTS { MATCH (x:N) WHERE x.id='a' MATCH (y:M) WHERE y.id='b' } AS e"
    ));
    // y.id='nope' matches nothing → false.
    assert!(!b(
        "RETURN EXISTS { MATCH (x:N) WHERE x.id='a' MATCH (y:M) WHERE y.id='nope' } AS e"
    ));
}

/// A correlated scalar VALUE subquery returns the body's single value per outer
/// row (NULL if empty), and ERRORS if the body matches more than one row.
#[test]
fn scalar_value_subquery() {
    let nd = concat!(
            "{\"id\":\"alice\",\"labels\":[\"Person\"],\"props\":{\"id\":\"alice\",\"name\":\"Alice\"}}\n",
            "{\"id\":\"carol\",\"labels\":[\"Person\"],\"props\":{\"id\":\"carol\",\"name\":\"Carol\"}}\n",
            "{\"id\":\"dave\",\"labels\":[\"Person\"],\"props\":{\"id\":\"dave\",\"name\":\"Dave\"}}\n",
            "{\"id\":\"bob\",\"labels\":[\"Person\"],\"props\":{\"id\":\"bob\",\"name\":\"Bob\"}}\n",
            "{\"id\":\"erin\",\"labels\":[\"Person\"],\"props\":{\"id\":\"erin\",\"name\":\"Erin\"}}\n",
            "{\"from\":\"alice\",\"to\":\"bob\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
            "{\"from\":\"dave\",\"to\":\"bob\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
            "{\"from\":\"dave\",\"to\":\"erin\",\"labels\":[\"KNOWS\"],\"props\":{}}"
        );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let one = |id: &str| -> Value {
        let q = format!(
                "MATCH (a:Person) WHERE a.id='{id}' RETURN VALUE {{ MATCH (a)-[:KNOWS]->(b) RETURN b.name }} AS f"
            );
        let plan = crate::opt::optimize_indexed(crate::gql::parse(&q).unwrap(), &store);
        run(&plan, &store).rows[0][0].clone()
    };
    assert!(matches!(one("alice"), Value::Str(s) if &*s == "Bob")); // one friend
    assert!(one("carol").is_null()); // no friend → NULL
                                     // dave knows two → the subquery returns >1 row → execute errors.
    let q = "MATCH (a:Person) WHERE a.id='dave' RETURN VALUE { MATCH (a)-[:KNOWS]->(b) RETURN b.name } AS f";
    let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
    assert!(try_run(&plan, &store).is_err());
}

/// OPTIONAL MATCH binding an edge variable `(a)-[f:R]->(b)` binds the edge slot
/// too (left-outer: null edge + null node on a miss). a->b->c, c has no out edge.
#[test]
fn optional_match_binds_edge() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":7.0}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"w\":9.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    // For each N, OPTIONAL MATCH one outgoing R edge; RETURN the node id + f.w.
    // a->b (w7), b->c (w9), c has none → f.w NULL.
    let q = "MATCH (n:N) OPTIONAL MATCH (n)-[f:R]->(u) RETURN n.id AS id, f.w AS w ORDER BY n.id";
    let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
    let out = run(&plan, &store);
    let got: Vec<(String, Value)> = out
        .rows
        .iter()
        .map(|r| {
            let id = match &r[0] {
                Value::Str(s) => s.to_string(),
                o => format!("{o:?}"),
            };
            (id, r[1].clone())
        })
        .collect();
    assert_eq!(got.len(), 3);
    assert!(matches!(&got[0], (id, Value::Num(w)) if id == "a" && *w == 7.0));
    assert!(matches!(&got[1], (id, Value::Num(w)) if id == "b" && *w == 9.0));
    assert!(matches!(&got[2], (id, v) if id == "c" && v.is_null())); // no edge → f null
}

/// A per-repetition WHERE on a MULTI-HOP unit references every edge of the rep
/// (e1 AND e2), checked at the rep boundary. Chain a-b-c-d-e, all amt 10.
#[test]
fn multi_hop_group_per_rep_where() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"id\":\"e\",\"labels\":[\"N\"],\"props\":{\"id\":\"e\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}\n",
        "{\"from\":\"d\",\"to\":\"e\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // e2.amt <= e1.amt (10<=10) holds → 1 rep (t=c), 2 reps (t=e).
    assert_eq!(
            ids("MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt <= e1.amt){1,2} (t) RETURN t.id AS id"),
            vec!["c", "e"]
        );
    // e2.amt < e1.amt (10<10) fails every rep → no path.
    assert!(
            ids("MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y) WHERE e2.amt < e1.amt){1,2} (t) RETURN t.id AS id")
                .is_empty()
        );
}

/// NESTED subpath groups (`Plan::NestedGroup`): group variables materialize as
/// (nested) lists — one list level per enclosing quantifier. On the triangle
/// a->b->c->a: family 4 `( (x)-[e:R]->{1,2}(y) ){1,2}` binds x/y once per OUTER rep
/// (depth 1) and e as a list-of-lists (depth 2); family 3 `( ((x)-[e]->(y)){1,2}
/// ){1,2}` binds x as a list-of-lists (depth 2).
#[test]
fn nested_subpath_groups() {
    // Chain a->b->c->d->e (ids 0..4).
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":0}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":1}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":2}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":3}}\n",
        "{\"id\":\"e\",\"labels\":[\"N\"],\"props\":{\"id\":4}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"d\",\"to\":\"e\",\"labels\":[\"R\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let rows = |q: &str| -> Vec<Vec<f64>> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v: Vec<Vec<f64>> = run(&plan, &store)
            .rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|c| match c {
                        Value::Num(x) => *x,
                        o => panic!("{o:?}"),
                    })
                    .collect()
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    // Family 4 from a: the 6 trail decompositions of outer{1,2}×inner{1,2}.
    // (tid, size(x), size(e)) — size(x)/size(e) = the outer rep count.
    assert_eq!(
        rows(
            "MATCH (s:N {id:0}) ( (x)-[e:R]->{1,2}(y) ){1,2} (t) \
                  RETURN t.id AS tid, size(x) AS nx, size(e) AS ne"
        ),
        vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 2.0, 2.0],
            vec![3.0, 2.0, 2.0],
            vec![4.0, 2.0, 2.0],
        ]
    );
    // Family 3 `( ((x)-[e]->(y)){2,2} ){2} (t)` from a: exactly one match — 2 outer
    // reps of 2 inner hops = the 4-hop trail a->b->c->d->e, endpoint e(4). x is
    // depth-2: size(x)=2 (outer), size(x[0])=2 (inner), x[0][0]=a(0).
    assert_eq!(
        rows(
            "MATCH (s:N {id:0}) ( ((x)-[e:R]->(y)){2,2} ){2} (t) \
                  RETURN t.id AS tid, size(x) AS nx, size(x[0]) AS nx0, x[0][0].id AS a00"
        ),
        vec![vec![4.0, 2.0, 2.0, 0.0]]
    );
}

/// A MULTI-HOP group unit `((x)-[e1]->(m)-[e2]->(y)){2}` binds each inner var to
/// a list strided by the unit hop count k; the endpoint lands at a rep boundary.
/// Chain a-11->b-22->c-33->d-44->e. 2 reps of 2 hops → t=e.
#[test]
fn repeat_group_multi_hop_unit() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"id\":\"e\",\"labels\":[\"N\"],\"props\":{\"id\":\"e\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":11.0}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":22.0}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":33.0}}\n",
        "{\"from\":\"d\",\"to\":\"e\",\"labels\":[\"R\"],\"props\":{\"amt\":44.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let q = "MATCH (s:N {id:'a'}) ((x)-[e1:R]->(m)-[e2:R]->(y)){2} (t) \
                 RETURN t.id AS tid, x[0].id AS x0, x[1].id AS x1, m[1].id AS m1, \
                 y[1].id AS y1, e1[0].amt AS p0, e2[1].amt AS q1";
    let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 1);
    let r = &out.rows[0];
    assert!(matches!(&r[0], Value::Str(s) if &**s == "e")); // t = e
    assert!(matches!(&r[1], Value::Str(s) if &**s == "a")); // x[0] = a
    assert!(matches!(&r[2], Value::Str(s) if &**s == "c")); // x[1] = c
    assert!(matches!(&r[3], Value::Str(s) if &**s == "d")); // m[1] = d
    assert!(matches!(&r[4], Value::Str(s) if &**s == "e")); // y[1] = e
    assert!(matches!(r[5], Value::Num(x) if x == 11.0)); // e1[0].amt
    assert!(matches!(r[6], Value::Num(x) if x == 44.0)); // e2[1].amt
}

/// A per-repetition WHERE prunes each hop by the rep's scalar x/e/y. Path
/// a-e1(30)->b-e2(20)->c-e3(10)->d; bals a=100,b=200,c=5,d=200.
#[test]
fn repeat_group_per_rep_where() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\",\"bal\":100.0}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\",\"bal\":200.0}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\",\"bal\":5.0}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\",\"bal\":200.0}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":30.0}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20.0}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{\"amt\":10.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ids = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // e.amt >= 1 holds for every edge → reach b, c, d.
    assert_eq!(
        ids("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt >= 1){1,3} (t) RETURN t.id AS id"),
        vec!["b", "c", "d"]
    );
    // e.amt <= x.bal fails at c->d (10 <= 5 false) → only b, c.
    assert_eq!(
        ids(
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE e.amt <= x.bal){1,3} (t) RETURN t.id AS id"
        ),
        vec!["b", "c"]
    );
    // y.bal >= 100 fails when y=c (bal 5) → only b (a->b).
    assert_eq!(
        ids("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y) WHERE y.bal >= 100){1,3} (t) RETURN t.id AS id"),
        vec!["b"]
    );
}

/// A per-hop edge WHERE in a shortest path filters which edges may be traversed.
/// a-e1(w1)->b, a-e2(w10)->c, c-e3(w10)->b. With w>5, e1 is blocked → a->c->b (2).
#[test]
fn shortest_path_per_hop_edge_where() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":1.0}}\n",
        "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"w\":10.0}}\n",
        "{\"from\":\"c\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":10.0}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let lens = |q: &str| -> Vec<f64> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store)
            .rows
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                ref o => panic!("{o:?}"),
            })
            .collect()
    };
    // e.w > 5 blocks a->b (w1); shortest a->b is a->c->b, length 2.
    assert_eq!(
            lens("MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 5]->*(b:N {id:'b'}) RETURN path_length(p) AS len"),
            vec![2.0]
        );
    // e.w > 100 blocks every edge → b unreachable.
    assert!(
            lens("MATCH p = ANY SHORTEST (a:N {id:'a'})-[e:R WHERE e.w > 100]->*(b:N {id:'b'}) RETURN path_length(p) AS len")
                .is_empty()
        );
}

/// SHORTEST k (k>=2) keeps the k shortest trails per endpoint by (length,
/// discovery); GROUP keeps every trail in the k smallest distinct lengths.
/// a->d (1), a->b->d (2), a->c->d (2).
#[test]
fn shortest_k_selector() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"from\":\"a\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"b\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"a\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"c\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let lens = |q: &str| -> Vec<f64> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v: Vec<f64> = run(&plan, &store)
            .rows
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                ref o => panic!("{o:?}"),
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    // SHORTEST 2 → the two shortest: len 1 and one len 2.
    assert_eq!(
            lens("MATCH p = SHORTEST 2 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len"),
            vec![1.0, 2.0]
        );
    // SHORTEST 2 GROUP → all trails in the 2 smallest lengths (1 and 2): 1,2,2.
    assert_eq!(
            lens("MATCH p = SHORTEST 2 GROUP (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len"),
            vec![1.0, 2.0, 2.0]
        );
    // SHORTEST 10 clamps to the 3 available.
    assert_eq!(
            lens("MATCH p = SHORTEST 10 (a:N {id:'a'})-[:R]->*(x:N {id:'d'}) RETURN path_length(p) AS len"),
            vec![1.0, 2.0, 2.0]
        );
}

/// A quantified subpath group binds its inner variables as GROUP lists: each
/// becomes a list over the repetitions, with `size()` the hop count and `v[i]`
/// a typed node/edge element so `x[i].prop` resolves. Path a-R(10)->b-R(20)->c.
#[test]
fn repeat_group_binds_group_variables() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":10}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":20}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let row = |q: &str| -> Vec<Value> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1, "{q}");
        out.rows[0].to_vec()
    };
    // {2}: t=c, size(e)=size(x)=size(y)=2, x[0]=a, y[1]=c, e[0].amt=10.
    let r = row(
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} (t) \
             RETURN t.id AS tid, size(e) AS ne, size(x) AS nx, x[0].id AS x0, y[1].id AS y1, e[0].amt AS e0",
        );
    assert!(matches!(&r[0], Value::Str(s) if &**s == "c")); // tid
    assert!(matches!(r[1], Value::Num(x) if x == 2.0)); // size(e)
    assert!(matches!(r[2], Value::Num(x) if x == 2.0)); // size(x)
    assert!(matches!(&r[3], Value::Str(s) if &**s == "a")); // x[0].id
    assert!(matches!(&r[4], Value::Str(s) if &**s == "c")); // y[1].id
    assert!(matches!(r[5], Value::Num(x) if x == 10.0)); // e[0].amt
                                                         // Anonymous endpoint: only the group vars are used.
    let r = row("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} RETURN size(e) AS ne");
    assert!(matches!(r[0], Value::Num(x) if x == 2.0));
}

/// `IS TYPED RECORD { f :: TYPE [NOT NULL], … }` is a CLOSED record type: no
/// extra fields, every declared field present-and-typed or absent-and-nullable,
/// recursing into nested records. INTEGER requires an integral number.
#[test]
fn is_typed_closed_record() {
    let store =
        crate::ndjson::from_ndjson("{\"id\":\"1\",\"labels\":[\"X\"],\"props\":{}}").unwrap();
    let row = |q: &str| -> Vec<bool> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store).rows[0]
            .iter()
            .map(|v| matches!(v, Value::Bool(true)))
            .collect()
    };
    assert_eq!(
        row(
            "RETURN {a: 1, b: 'x'} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS a, \
                 {a: 1} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS b, \
                 {a: 1, b: 'x', c: 9} IS TYPED RECORD {a :: INTEGER, b :: STRING} AS c, \
                 {a: 1.5} IS TYPED RECORD {a :: INTEGER} AS d, \
                 {a: 1.5} IS TYPED RECORD {a :: FLOAT} AS e"
        ),
        vec![true, true, false, false, true]
    );
    assert_eq!(
            row("RETURN {} IS TYPED RECORD {a :: INTEGER NOT NULL} AS a, \
                 {a: null} IS TYPED RECORD {a :: INTEGER NOT NULL} AS b, \
                 {geo: {lat: 1, lng: 2}} IS TYPED RECORD {geo :: RECORD {lat :: INTEGER, lng :: INTEGER}} AS c, \
                 {geo: {lat: 'x'}} IS TYPED RECORD {geo :: RECORD {lat :: INTEGER, lng :: INTEGER}} AS d"),
            vec![false, false, true, false]
        );
}

/// A group-variable EDGE list keeps its element typing across a `WITH … AS`
/// rename: `WITH e AS hops` leaves `hops[i].amt` resolving the edge property, not
/// NULL. (The parser remaps the edge-/node-list slot sets through the WITH.)
#[test]
fn group_variable_edge_typing_survives_with_rename() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"amt\":11}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{\"amt\":22}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse(
            "MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){2} (t) \
                 WITH e AS hops, t RETURN t.id AS tid, hops[1].amt AS amt2, hops[0].amt AS amt1",
        )
        .unwrap(),
        &store,
    );
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 1);
    let r = out.rows[0].to_vec();
    assert!(matches!(&r[0], Value::Str(s) if &**s == "c"));
    assert!(matches!(r[1], Value::Num(x) if x == 22.0)); // hops[1].amt
    assert!(matches!(r[2], Value::Num(x) if x == 11.0)); // hops[0].amt
}

/// A standalone FILTER clause filters the working table, and repeated statement-
/// position ORDER BY … LIMIT compose (page then re-page). n = 1,5,9.
#[test]
fn filter_clause_and_composed_paging() {
    let mut b = Builder::default();
    b.node(&["T"], &[("n", n(1.0))]);
    b.node(&["T"], &[("n", n(5.0))]);
    b.node(&["T"], &[("n", n(9.0))]);
    let store = b.build();
    let one = |q: &str| -> f64 {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        match run(&plan, &store).rows[0][0] {
            Value::Num(x) => x,
            ref o => panic!("{o:?}"),
        }
    };
    // FILTER keeps n>3 (5,9); ORDER BY n LIMIT 1 -> 5.
    assert_eq!(
        one("MATCH (t:T) FILTER t.n > 3 ORDER BY t.n LIMIT 1 RETURN t.n AS x"),
        5.0
    );
    // Page (asc, top 2 -> {1,5}) then re-page (desc, top 1 -> 5).
    assert_eq!(
        one("MATCH (t:T) ORDER BY t.n LIMIT 2 ORDER BY t.n DESC LIMIT 1 RETURN t.n AS x"),
        5.0
    );
}

/// An UNQUANTIFIED subpath group `(( pattern [WHERE p] ))` is a scoping paren:
/// the inner pattern + trailing WHERE filter, no repetition. A NAMED path over
/// one is rejected (the TS engine does). Fixture: Amy(25)->Bob(40), Bob(40)->Amy(25).
#[test]
fn unquantified_subpath_group() {
    let nd = concat!(
        "{\"id\":\"amy\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Amy\",\"age\":25}}\n",
        "{\"id\":\"bob\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Bob\",\"age\":40}}\n",
        "{\"from\":\"amy\",\"to\":\"bob\",\"labels\":[\"KNOWS\"],\"props\":{}}\n",
        "{\"from\":\"bob\",\"to\":\"amy\",\"labels\":[\"KNOWS\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let names = |q: &str| -> Vec<String> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v = names_of(&run(&plan, &store), 0);
        v.sort();
        v
    };
    // Only Amy(25)->Bob(40) satisfies x.age < y.age.
    assert_eq!(
        names("MATCH ((x:Person)-[:KNOWS]->(y:Person) WHERE x.age < y.age) RETURN x.name AS n"),
        vec!["Amy"]
    );
    // Single-node group with WHERE.
    assert_eq!(
        names("MATCH ((x:Person) WHERE x.age >= 35) RETURN x.name AS n"),
        vec!["Bob"]
    );
    // A named path over an unquantified group is rejected (matches the TS engine).
    assert!(
        crate::gql::parse("MATCH p = ((x)-[:KNOWS]->(y) WHERE x.age < y.age) RETURN p").is_err()
    );
}

/// The `LET name = expr` clause adds a binding, carrying existing bindings
/// forward, so a later RETURN/GROUP BY can reference it. t = 5,5,9.
#[test]
fn let_clause_binds_and_carries_forward() {
    let mut b = Builder::default();
    b.node(&["P"], &[("t", n(5.0))]);
    b.node(&["P"], &[("t", n(5.0))]);
    b.node(&["P"], &[("t", n(9.0))]);
    let store = b.build();
    let rows = |q: &str| -> Vec<(f64, f64)> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store)
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Num(a), Value::Num(c)) => (*a, *c),
                o => panic!("{o:?}"),
            })
            .collect()
    };
    // LET-bound key used in RETURN + GROUP BY + ORDER BY.
    assert_eq!(
        rows("MATCH (n:P) LET t = n.t RETURN t, count(*) AS c GROUP BY t ORDER BY t"),
        vec![(5.0, 2.0), (9.0, 1.0)]
    );
    // The original binding `n` survives the LET (still usable downstream).
    assert_eq!(
        rows("MATCH (n:P) LET t = n.t RETURN n.t AS a, count(*) AS c ORDER BY a"),
        vec![(5.0, 2.0), (9.0, 1.0)]
    );
}

/// ORDER BY resolves an output alias even before `NULLS FIRST|LAST`, and ORDER
/// BY the underlying expression of a projected alias sorts by that output column
/// (so it composes with DISTINCT). k = 3, (null), 7 over three P nodes.
#[test]
fn order_by_alias_with_nulls_and_projected_expr() {
    let mut b = Builder::default();
    b.node(&["P"], &[("k", n(3.0)), ("nn", n(1.0))]);
    b.node(&["P"], &[("nn", n(2.0))]); // k absent -> null
    b.node(&["P"], &[("k", n(7.0)), ("nn", n(1.0))]);
    let store = b.build();
    let col0 = |q: &str| -> Vec<Value> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store)
            .rows
            .iter()
            .map(|r| r[0].clone())
            .collect()
    };
    // NULLS FIRST after a bare alias: null sorts first, then 3, 7.
    let got = col0("MATCH (u:P) RETURN u.k AS a ORDER BY a NULLS FIRST");
    assert!(got[0].is_null());
    assert!(matches!(got[1], Value::Num(x) if x == 3.0));
    assert!(matches!(got[2], Value::Num(x) if x == 7.0));
    // DISTINCT with ORDER BY the underlying expression of the projected alias.
    let got = col0("MATCH (u:P) RETURN DISTINCT u.nn AS a ORDER BY u.nn");
    assert_eq!(got.len(), 2); // distinct {1,2}
    assert!(matches!(got[0], Value::Num(x) if x == 1.0));
    assert!(matches!(got[1], Value::Num(x) if x == 2.0));
    // An ORDER BY *expression* may reference an output alias by name (`a` inside
    // a LET-IN): it inlines to the alias's definition (u.k), so the rows sort by
    // k — null first only under NULLS FIRST; default nulls last.
    let got = col0("MATCH (u:P) RETURN u.k AS a ORDER BY (LET x = a IN x END)");
    assert!(matches!(got[0], Value::Num(x) if x == 3.0));
    assert!(matches!(got[1], Value::Num(x) if x == 7.0));
    assert!(got[2].is_null());
}

/// An explicit `GROUP BY` after the RETURN list parses and groups the same as
/// the implicit (non-aggregate items are the keys). n=1,1,2 over three P nodes.
#[test]
fn explicit_group_by_after_return() {
    let mut b = Builder::default();
    b.node(&["P"], &[("n", n(1.0))]);
    b.node(&["P"], &[("n", n(1.0))]);
    b.node(&["P"], &[("n", n(2.0))]);
    let store = b.build();
    // GROUP BY the underlying expression, ORDER BY the alias then the expr.
    let rows = |q: &str| -> Vec<(f64, f64)> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store)
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Num(a), Value::Num(c)) => (*a, *c),
                o => panic!("{o:?}"),
            })
            .collect()
    };
    assert_eq!(
        rows("MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY a"),
        vec![(1.0, 2.0), (2.0, 1.0)]
    );
    assert_eq!(
        rows("MATCH (u:P) RETURN u.n AS a, count(*) AS c GROUP BY u.n ORDER BY u.n"),
        vec![(1.0, 2.0), (2.0, 1.0)]
    );
}

/// GROUP BY a key that is NOT among the RETURN items still groups: it becomes a
/// hidden grouping key, dropped from the output. `RETURN count(*) GROUP BY
/// e.dept` yields one row per dept. An aggregate ORDER BY resolves to its true
/// (keys-then-aggs) schema column, not its RETURN position.
#[test]
fn group_by_non_returned_key() {
    let mut b = Builder::default();
    b.node(&["E"], &[("dept", s("eng")), ("sal", n(100.0))]);
    b.node(&["E"], &[("dept", s("eng")), ("sal", n(200.0))]);
    b.node(&["E"], &[("dept", s("sales")), ("sal", n(50.0))]);
    let store = b.build();
    let nums = |q: &str| -> Vec<f64> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store)
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Num(x) => *x,
                o => panic!("{o:?}"),
            })
            .collect()
    };
    // count per dept, key not returned: two groups (eng=2, sales=1), any order.
    let mut c = nums("MATCH (e:E) RETURN count(*) AS c GROUP BY e.dept");
    c.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(c, vec![1.0, 2.0]);
    // sum per dept ordered by the aggregate: the ORDER BY alias must hit the sum
    // column (after the hidden key), so ascending is [50, 300] not [300, 50].
    assert_eq!(
        nums("MATCH (e:E) RETURN sum(e.sal) AS s GROUP BY e.dept ORDER BY s"),
        vec![50.0, 300.0]
    );
}

/// A leading `OPTIONAL MATCH` with no prior binding: on an EMPTY graph it still
/// yields one row, with the pattern variable NULL — so `n.missing IS NULL` is
/// true. On a non-empty graph it behaves like an ordinary scan (one row per
/// node), no null padding.
#[test]
fn leading_optional_match_pads_one_null_row_when_empty() {
    let empty = Builder::default().build();
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("OPTIONAL MATCH (n) RETURN n.missing IS NULL AS m").unwrap(),
        &empty,
    );
    let out = run(&plan, &empty);
    assert_eq!(out.rows.len(), 1);
    assert!(matches!(out.rows[0][0], Value::Bool(true)));

    let mut b = Builder::default();
    b.node(&["X"], &[("a", n(5.0))]);
    b.node(&["X"], &[("a", n(6.0))]);
    let store = b.build();
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("OPTIONAL MATCH (n) RETURN n.a AS a ORDER BY a").unwrap(),
        &store,
    );
    let got: Vec<f64> = run(&plan, &store)
        .rows
        .iter()
        .map(|r| match &r[0] {
            Value::Num(x) => *x,
            o => panic!("{o:?}"),
        })
        .collect();
    assert_eq!(got, vec![5.0, 6.0]);
}

/// LIMIT 0 yields the empty result WITHOUT evaluating the projection, so a
/// faulting expression (`1/0`) under LIMIT 0 does not error (matches the TS engine).
#[test]
fn limit_zero_short_circuits_before_projection() {
    let mut b = Builder::default();
    b.node(&["T"], &[("x", n(3.0))]);
    let store = b.build();
    // Without LIMIT 0, `1/0` faults; with it, the projection is never reached.
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("MATCH (n:T) RETURN 1/0 AS x LIMIT 0").unwrap(),
        &store,
    );
    let out = try_run(&plan, &store).expect("LIMIT 0 must not fault");
    assert_eq!(out.rows.len(), 0);
    // DISTINCT … LIMIT 0 too.
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("MATCH (n:T) RETURN DISTINCT 1/0 AS x LIMIT 0").unwrap(),
        &store,
    );
    assert_eq!(try_run(&plan, &store).unwrap().rows.len(), 0);
}

/// A named path over a NON-shortest var-length pattern binds the walk lineage,
/// so path_length(p)/edges(p)/nodes(p) resolve. Fixture a->b->c->a (cycle) + a->d.
#[test]
fn named_path_over_var_length_binds_lineage() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"b\",\"to\":\"c\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"c\",\"to\":\"a\",\"labels\":[\"R\"],\"props\":{}}\n",
        "{\"from\":\"a\",\"to\":\"d\",\"labels\":[\"R\"],\"props\":{}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let run_q = |q: &str, col: usize| -> Vec<f64> {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        let mut v: Vec<f64> = run(&plan, &store)
            .rows
            .iter()
            .map(|r| match r[col] {
                Value::Num(x) => x,
                ref o => panic!("{o:?}"),
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    // paths from a of length 1..3: a-b (1), a-d (1), a-b-c (2), a-b-c-a (3).
    assert_eq!(
        run_q(
            "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN path_length(p) AS len",
            0
        ),
        vec![1.0, 1.0, 2.0, 3.0]
    );
    // size(edges(p)) tracks the hop count (path_length).
    assert_eq!(
        run_q(
            "MATCH p = (a:N {id:'a'})-[:R]->{1,3}(x) RETURN size(edges(p)) AS es",
            0
        ),
        vec![1.0, 1.0, 2.0, 3.0]
    );
    // min 0 binds the length-0 seed path (a itself) too.
    assert_eq!(
        run_q(
            "MATCH p = (a:N {id:'a'})-[:R]->{0,1}(x) RETURN path_length(p) AS len",
            0
        ),
        vec![0.0, 1.0, 1.0]
    );
}

/// String (K10) and list/element (K11) functions match hand-computed values.
#[test]
fn added_string_and_list_functions() {
    let mut b = Builder::default();
    b.node(&["N", "M"], &[("z", n(1.0)), ("a", n(2.0))]);
    let store = b.build();
    let val = |e: &str| -> Value {
        let q = format!("MATCH (x:N) RETURN {e} AS v");
        run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
    };
    let str_of = |e: &str| match val(e) {
        Value::Str(s) => s.to_string(),
        o => panic!("{e} → {o:?}"),
    };
    // `Value` has no `PartialEq` (the value contract owns equality), so compare
    // list contents via debug strings.
    let list_of = |e: &str| -> Vec<String> {
        match val(e) {
            Value::List(v) => v.iter().map(|x| format!("{x:?}")).collect(),
            o => panic!("{e} → {o:?}"),
        }
    };
    let dbg = |xs: &[Value]| -> Vec<String> { xs.iter().map(|x| format!("{x:?}")).collect() };
    // trims (whitespace, and explicit char set)
    assert_eq!(str_of("ltrim('  hi ')"), "hi ");
    assert_eq!(str_of("rtrim('  hi ')"), "  hi");
    assert_eq!(str_of("btrim('xxhixx', 'x')"), "hi");
    // reverse (string + list), left/right, split
    assert_eq!(str_of("reverse('abc')"), "cba");
    assert_eq!(str_of("left('abcd', 2)"), "ab");
    assert_eq!(str_of("right('abcd', 2)"), "cd");
    assert_eq!(str_of("left('ab', 5)"), "ab"); // n > len → whole
    assert_eq!(
        list_of("split('a,b,c', ',')"),
        dbg(&[s("a"), s("b"), s("c")])
    );
    // list fns
    assert_eq!(
        list_of("reverse([1, 2, 3])"),
        dbg(&[n(3.0), n(2.0), n(1.0)])
    );
    assert_eq!(list_of("tail([1, 2, 3])"), dbg(&[n(2.0), n(3.0)]));
    assert_eq!(
        list_of("range(1, 4)"),
        dbg(&[n(1.0), n(2.0), n(3.0), n(4.0)])
    );
    assert_eq!(list_of("range(5, 1, -1)").len(), 5);
    assert!(val("range(1, 4, 0)").is_null()); // zero step
    assert_eq!(list_of("range(5, 1)"), Vec::<String>::new()); // wrong-sign default step
                                                              // element fns: keys (sorted present props), labels (sorted)
    assert_eq!(list_of("keys(x)"), dbg(&[s("a"), s("z")]));
    assert_eq!(list_of("labels(x)"), dbg(&[s("M"), s("N")]));
}

/// `IN` / `NOT IN` over a list literal (K7), desugared to an OR-chain of
/// equals — including three-valued behavior with a NULL in the list.
#[test]
fn in_operator() {
    let mut b = Builder::default();
    b.node(&["N"], &[("a", n(1.0))]);
    b.node(&["N"], &[("a", n(2.0))]);
    b.node(&["N"], &[("a", n(9.0))]);
    b.node(&["N"], &[]); // a is NULL
    let store = b.build();
    let ids = |q: &str| -> Vec<String> {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
        v.sort();
        v
    };
    // a IN [1,2] → the 1 and 2 nodes.
    assert_eq!(
        ids("MATCH (n:N) WHERE n.a IN [1, 2] RETURN n.a AS a"),
        vec!["Num(1.0)", "Num(2.0)"]
    );
    // NOT IN → the 9 node only (NULL-a is UNKNOWN, dropped, not returned).
    assert_eq!(
        ids("MATCH (n:N) WHERE n.a NOT IN [1, 2] RETURN n.a AS a"),
        vec!["Num(9.0)"]
    );
    // A NULL element makes a non-match UNKNOWN → row drops (3VL): only the
    // literal 1 matches; 2/9 are UNKNOWN (could equal the null), dropped.
    assert_eq!(
        ids("MATCH (n:N) WHERE n.a IN [1, null] RETURN n.a AS a"),
        vec!["Num(1.0)"]
    );
    // Empty list → nobody matches.
    assert_eq!(
        ids("MATCH (n:N) WHERE n.a IN [] RETURN n.a AS a"),
        Vec::<String>::new()
    );
}

/// Dynamic (non-literal) IN over a list PROPERTY — the runtime `Expr::In`, with
/// the same three-valued behavior as the literal OR-chain.
#[test]
fn in_operator_dynamic() {
    let mut b = Builder::default();
    b.node(
        &["N"],
        &[
            ("a", n(2.0)),
            ("xs", Value::List(vec![n(1.0), n(2.0), n(3.0)])),
        ],
    );
    b.node(
        &["N"],
        &[
            ("a", n(9.0)),
            ("xs", Value::List(vec![n(1.0), n(2.0), n(3.0)])),
        ],
    );
    b.node(
        &["N"],
        &[
            ("a", n(5.0)),
            ("xs", Value::List(vec![n(1.0), Value::Null, n(3.0)])),
        ],
    );
    let store = b.build();
    let ids = |q: &str| -> Vec<String> {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
        v.sort();
        v
    };
    // n.a IN n.xs: only the a=2 node (2 ∈ [1,2,3]); a=9 not in; a=5 vs [1,null,3]
    // is UNKNOWN (null element) → dropped.
    assert_eq!(
        ids("MATCH (n:N) WHERE n.a IN n.xs RETURN n.a AS a"),
        vec!["Num(2.0)"]
    );
    // 2 IN n.xs: the two nodes whose list has 2; the [1,null,3] node lacks 2 and
    // is UNKNOWN → dropped.
    assert_eq!(
        ids("MATCH (n:N) WHERE 2 IN n.xs RETURN n.a AS a"),
        vec!["Num(2.0)", "Num(9.0)"]
    );
}

/// Undirected `~` traversal is `Dir::Both`: a normal edge is reached from both
/// endpoints (two rows), but a self-loop is walked ONCE (its in-side copy is
/// dropped), matching the TS engine's `SelfLoops::Once`.
#[test]
fn undirected_tilde_self_loop_counted_once() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"a\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let count = |q: &str| -> f64 {
        match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(n) => n,
            ref other => panic!("want num, got {other:?}"),
        }
    };
    // Self-loop once (1) + a-b both orientations (2) = 3.
    assert_eq!(count("MATCH (a)~[r]~(b) RETURN count(*) AS c"), 3.0);
    // The same over a single-hop var-length spelling routes through the DFS
    // walker, which also drops the self-loop's in-side copy.
    assert_eq!(count("MATCH (a)~[:R]~{1,1}(b) RETURN count(*) AS c"), 3.0);
    // A directed self-loop is walked once either way (one index touched).
    assert_eq!(count("MATCH (a)-[r:R]->(b) RETURN count(*) AS c"), 2.0);
}

/// `SELECT … GROUP BY … HAVING …` filters grouped rows: on an aggregate
/// (`count(*) > 1`), on a group key (`n.age >= 35`), globally (no GROUP BY), and
/// `HAVING null` drops every group. An aggregate may appear only in HAVING.
#[test]
fn select_having() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{\"age\":30}}\n",
        "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{\"age\":30}}\n",
        "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{\"age\":40}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let rows = |q: &str| -> Vec<String> {
        run(&crate::gql::parse(q).unwrap(), &store)
            .rows
            .iter()
            .map(|r| {
                r.iter()
                    .map(|c| format!("{c:?}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect()
    };
    // HAVING on an aggregate: only the age-30 group (count 2 > 1).
    assert_eq!(
            rows("SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) GROUP BY n.age HAVING count(*) > 1 ORDER BY age"),
            vec!["Num(30.0),Num(2.0)"]
        );
    // Aggregate only in HAVING, not in the SELECT list.
    assert_eq!(
        rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING count(*) > 1"),
        vec!["Num(30.0)"]
    );
    // HAVING on a group key.
    assert_eq!(
            rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING n.age >= 35 ORDER BY age"),
            vec!["Num(40.0)"]
        );
    // Global HAVING (no GROUP BY): 3 people — passes >2, fails >100.
    assert_eq!(
        rows("SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 2"),
        vec!["Num(3.0)"]
    );
    assert!(rows("SELECT count(*) AS c FROM MATCH (n:Person) HAVING count(*) > 100").is_empty());
    // HAVING null drops every group.
    assert!(
        rows("SELECT n.age AS age FROM MATCH (n:Person) GROUP BY n.age HAVING null").is_empty()
    );
}

/// `ALL SHORTEST` emits one row per distinct shortest path (so a target reached
/// by two equal-length paths appears twice), while `ANY SHORTEST` emits one row
/// per reachable target. A `*` quantifier includes the zero-length seed.
#[test]
fn all_shortest_multiplicity() {
    // Diamond: a->b, a->c, b->d, c->d — d is reachable by two 2-hop paths.
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
        "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
        "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"c\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e3\",\"from\":\"b\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
        "{\"id\":\"e4\",\"from\":\"c\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let count = |q: &str| -> f64 {
        match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(n) => n,
            ref other => panic!("want num, got {other:?}"),
        }
    };
    // ANY: seed a (len 0) + b + c + d(once) = 4 rows.
    assert_eq!(
        count("MATCH ANY SHORTEST (a {id:'a'})-[:R]->*(x) RETURN count(*) AS c"),
        4.0
    );
    // ALL: a + b + c + d TWICE (two shortest paths) = 5 rows.
    assert_eq!(
        count("MATCH ALL SHORTEST (a {id:'a'})-[:R]->*(x) RETURN count(*) AS c"),
        5.0
    );
    // ALL restricted to endpoint d: two shortest paths → 2 rows.
    assert_eq!(
        count("MATCH ALL SHORTEST (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
        2.0
    );
    // SHORTEST 1 reduces to ANY (one row for d); SHORTEST 1 GROUP to ALL (two).
    assert_eq!(
        count("MATCH SHORTEST 1 (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
        1.0
    );
    assert_eq!(
        count("MATCH SHORTEST 1 GROUP (a {id:'a'})-[:R]->*(x {id:'d'}) RETURN count(*) AS c"),
        2.0
    );
}

/// `SELECT … [FROM MATCH …]` is sugar for MATCH…RETURN: a constant projection
/// with no FROM, a plain projection, a global aggregate with WHERE, and a
/// GROUP BY (via implicit grouping) with ORDER BY over an output alias.
#[test]
fn select_from_match() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Alice\",\"age\":30}}\n",
        "{\"id\":\"b\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Bob\",\"age\":40}}\n",
        "{\"id\":\"c\",\"labels\":[\"Person\"],\"props\":{\"name\":\"Cara\",\"age\":30}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let one = |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
    // Constant projection, no FROM.
    assert!(matches!(one("SELECT 1 + 2 AS v"), Value::Num(n) if n == 3.0));
    // Plain projection with an inline filter.
    assert!(
        matches!(one("SELECT n.name AS nm FROM MATCH (n:Person {name: 'Alice'})"), Value::Str(s) if &*s == "Alice")
    );
    // Global aggregate with WHERE (>= 30 → all three).
    assert!(
        matches!(one("SELECT count(*) AS c FROM MATCH (n:Person) WHERE n.age >= 30"), Value::Num(n) if n == 3.0)
    );
    // GROUP BY age with ORDER BY the output alias: ages 30 (×2), 40 (×1).
    let grouped = run(
        &crate::gql::parse(
            "SELECT n.age AS age, count(*) AS c FROM MATCH (n:Person) GROUP BY n.age ORDER BY age",
        )
        .unwrap(),
        &store,
    );
    let rows: Vec<String> = grouped
        .rows
        .iter()
        .map(|r| format!("{:?},{:?}", r[0], r[1]))
        .collect();
    assert_eq!(rows, vec!["Num(30.0),Num(2.0)", "Num(40.0),Num(1.0)"]);
}

/// A NaN operand makes ordering (`< > <= >=`) definitely FALSE (IEEE), NOT unknown —
/// matching JS and the pure-TS engine. Equality with NaN stays false (NaN != NaN).
#[test]
fn nan_ordering_is_ieee_false() {
    let store =
        crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
    let val = |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
    // log10(-1) is NaN. Every ordering against it is FALSE, not null.
    assert!(matches!(
        val("RETURN (log10(-1) < 5) AS x"),
        Value::Bool(false)
    ));
    assert!(matches!(
        val("RETURN (log10(-1) >= 5) AS x"),
        Value::Bool(false)
    ));
    assert!(matches!(
        val("RETURN (5 < log10(-1)) AS x"),
        Value::Bool(false)
    ));
    assert!(matches!(
        val("RETURN (0.0 > log10(-1)) AS x"),
        Value::Bool(false)
    ));
    // Equality with NaN is still FALSE, and its negation TRUE — ordering is the only
    // thing that changed from 3-valued to IEEE.
    assert!(matches!(
        val("RETURN (log10(-1) = log10(-1)) AS x"),
        Value::Bool(false)
    ));
    assert!(matches!(
        val("RETURN (log10(-1) <> log10(-1)) AS x"),
        Value::Bool(true)
    ));
}

/// -0 and +0 are one value: atan2 (the only fn whose result distinguishes the sign of
/// a zero operand) folds -0 to +0 on both inputs, whether the -0 came from a literal
/// or from arithmetic (`0 * -1`). So atan2(±0, -1) is always +PI, never -PI.
#[test]
fn signed_zero_folds_in_atan2() {
    let store =
        crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
    let num = |q: &str| -> f64 {
        match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() {
            Value::Num(n) => n,
            other => panic!("want num, got {other:?}"),
        }
    };
    let pi = std::f64::consts::PI;
    assert!((num("RETURN atan2(-0.0, -1.0) AS r") - pi).abs() < 1e-12);
    assert!((num("RETURN atan2(0.0 * -1, -1.0) AS r") - pi).abs() < 1e-12);
    assert!(num("RETURN atan2(-0.0, -0.0) AS r").abs() < 1e-12); // atan2(+0, +0) = 0
}

/// NaN has no sign: `sign(NaN)` is NaN (→ null at egress), NOT the 0 that a naive
/// `>0 / <0 / else` would give. Finite inputs are unchanged.
#[test]
fn sign_of_nan_is_nan() {
    let store =
        crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
    let val = |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
    // asin(1e100) and log10(-1) are NaN; sign keeps NaN at the row level (it is only
    // coerced to null at the JSON egress boundary, per the K4 policy).
    assert!(matches!(val("RETURN sign(asin(1e100)) AS x"), Value::Num(n) if n.is_nan()));
    assert!(matches!(val("RETURN sign(log10(-1)) AS x"), Value::Num(n) if n.is_nan()));
    // Finite inputs are unaffected.
    assert!(matches!(val("RETURN sign(-5.0) AS x"), Value::Num(n) if n == -1.0));
    assert!(matches!(val("RETURN sign(5.0) AS x"), Value::Num(n) if n == 1.0));
    assert!(matches!(val("RETURN sign(0.0) AS x"), Value::Num(n) if n == 0.0));
}

/// Scalar functions: 2-arg round (incl. negative digits), atan2 (arg order +
/// null propagation), log10, TRIM spec forms, and list_sort with order/nullOrder.
#[test]
fn scalar_fns_batch() {
    let store =
        crate::ndjson::from_ndjson("{\"id\":\"n\",\"labels\":[\"V\"],\"props\":{}}").unwrap();
    let val = |q: &str| -> Value { run(&crate::gql::parse(q).unwrap(), &store).rows[0][0].clone() };
    let num = |q: &str| -> f64 {
        match val(q) {
            Value::Num(n) => n,
            other => panic!("want num, got {other:?}"),
        }
    };
    // round to N decimal places; negative digits round left of the point.
    assert_eq!(num("RETURN round(1.2345, 2) AS r"), 1.23);
    assert_eq!(num("RETURN round(1234.5678, -2) AS r"), 1200.0);
    assert_eq!(num("RETURN round(2.5) AS r"), 3.0); // 1-arg still works
                                                    // atan2(y, x): arg order matters; a null arg → NULL.
    assert!((num("RETURN atan2(1, 1) AS r") - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    assert_eq!(num("RETURN atan2(0, 1) AS r"), 0.0);
    assert!(matches!(val("RETURN atan2(null, 1) AS r"), Value::Null));
    // log10.
    assert_eq!(num("RETURN log10(1000) AS r"), 3.0);
    // TRIM spec forms desugar to trim/ltrim/rtrim with the char as 2nd arg.
    let s = |q: &str| -> String {
        match val(q) {
            Value::Str(x) => x.to_string(),
            other => panic!("want str, got {other:?}"),
        }
    };
    assert_eq!(s("RETURN TRIM('  hi  ') AS r"), "hi");
    assert_eq!(s("RETURN TRIM(BOTH FROM '  hi  ') AS r"), "hi");
    assert_eq!(s("RETURN TRIM(LEADING 'x' FROM 'xxhi') AS r"), "hi");
    assert_eq!(s("RETURN TRIM(TRAILING 'x' FROM 'hixx') AS r"), "hi");
    assert_eq!(s("RETURN TRIM('x' FROM 'xxhixx') AS r"), "hi");
    // list_sort: default ascending, 'desc' reverses, nullOrder places nulls.
    // Compare list results by their debug rendering (Value is not PartialEq).
    let list = |q: &str| -> String { format!("{:?}", val(q)) };
    assert_eq!(
        list("RETURN list_sort([3,1,2], 'desc') AS r"),
        "List([Num(3.0), Num(2.0), Num(1.0)])"
    );
    assert_eq!(
        list("RETURN list_sort([3,1,null,2], 'asc', 'first') AS r"),
        "List([Null, Num(1.0), Num(2.0), Num(3.0)])"
    );
    // default null placement is LAST.
    assert_eq!(
        list("RETURN list_sort([2,null,1]) AS r"),
        "List([Num(1.0), Num(2.0), Null])"
    );
}

/// An edge-type disjunction `-[:A|B]->` matches an edge whose type is ANY of the
/// listed types; a typed-but-all-unknown disjunction matches nothing (it is NOT
/// read as "any"); an unknown name in a partial disjunction is dropped.
#[test]
fn edge_type_disjunction() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"KNOWS\",\"props\":{}}\n",
        "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"c\",\"type\":\"CREATED\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let count = |q: &str| -> f64 {
        match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(n) => n,
            ref other => panic!("want num, got {other:?}"),
        }
    };
    // Both edge types match → both neighbours.
    assert_eq!(
        count("MATCH (a)-[:KNOWS|CREATED]->(x) RETURN count(*) AS c"),
        2.0
    );
    // Order is irrelevant to the set.
    assert_eq!(
        count("MATCH (a)-[:CREATED|KNOWS]->(x) RETURN count(*) AS c"),
        2.0
    );
    // A single named type still matches only that one.
    assert_eq!(count("MATCH (a)-[:KNOWS]->(x) RETURN count(*) AS c"), 1.0);
    // A partial disjunction drops the unknown name, keeping the known one.
    assert_eq!(
        count("MATCH (a)-[:KNOWS|BOGUS]->(x) RETURN count(*) AS c"),
        1.0
    );
    // Typed but ALL-unknown matches nothing (NOT read as "any type").
    assert_eq!(
        count("MATCH (a)-[:BOGUS|NOPE]->(x) RETURN count(*) AS c"),
        0.0
    );
}

/// `MATCH WALK` lets a variable-length hop reuse an edge; `TRAIL` (the default)
/// forbids it. Over a self-loop, a length-2 hop exists as a WALK (reuse the loop)
/// but not as a TRAIL.
#[test]
fn path_mode_walk_vs_trail_edge_reuse() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
        "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"a\",\"type\":\"R\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let count = |q: &str| -> f64 {
        match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(n) => n,
            ref other => panic!("want num, got {other:?}"),
        }
    };
    // WALK: a->a->a reuses the loop edge — one length-2 walk.
    assert_eq!(
        count("MATCH WALK (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
        1.0
    );
    // TRAIL (default): the loop edge can't repeat — no length-2 trail.
    assert_eq!(
        count("MATCH TRAIL (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
        0.0
    );
    assert_eq!(
        count("MATCH (a {id:'a'})-[:R]->{2,2}(x) RETURN count(*) AS c"),
        0.0
    );
}

/// `~` resolves to `Dir::Both` regardless of which side (or a `-`/`~` mix) is
/// used, matching either traversal direction of the edge.
#[test]
fn undirected_tilde_matches_either_direction() {
    let nd = concat!(
        "{\"id\":\"josh\",\"labels\":[\"P\"],\"props\":{\"name\":\"josh\"}}\n",
        "{\"id\":\"vadas\",\"labels\":[\"P\"],\"props\":{\"name\":\"vadas\"}}\n",
        "{\"id\":\"e1\",\"from\":\"josh\",\"to\":\"vadas\",\"type\":\"KNOWS\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    // josh has an OUT edge; the undirected walk still reaches vadas.
    let mut a = names_of(
        &run(
            &crate::gql::parse("MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'josh' RETURN b.name AS n")
                .unwrap(),
            &store,
        ),
        0,
    );
    a.sort();
    assert_eq!(a, vec!["vadas"]);
    // vadas has only an IN edge; the undirected walk reaches josh.
    let b = names_of(
        &run(
            &crate::gql::parse("MATCH (a)~[:KNOWS]~(b) WHERE a.name = 'vadas' RETURN b.name AS n")
                .unwrap(),
            &store,
        ),
        0,
    );
    assert_eq!(b, vec!["josh"]);
}

/// External ids are PRESERVED through ingest and returned by element_id (nodes
/// and edges), and survive an NDJSON round-trip.
#[test]
fn element_id_preserves_external_ids() {
    let nd = concat!(
        "{\"id\":\"alice\",\"labels\":[\"P\"],\"props\":{}}\n",
        "{\"id\":\"bob\",\"labels\":[\"P\"],\"props\":{}}\n",
        "{\"id\":\"e42\",\"from\":\"alice\",\"to\":\"bob\",\"type\":\"KNOWS\",\"props\":{}}\n",
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    // element_id(node) returns the preserved string id.
    let mut ns = names_of(
        &run(
            &crate::gql::parse("MATCH (n:P) RETURN element_id(n) AS a0").unwrap(),
            &store,
        ),
        0,
    );
    ns.sort();
    assert_eq!(ns, vec!["alice", "bob"]);
    // element_id(edge) returns the preserved edge id.
    let es = run(
        &crate::gql::parse("MATCH (a:P)-[r:KNOWS]->(b) RETURN element_id(r) AS a0").unwrap(),
        &store,
    );
    assert!(matches!(&es.rows[0][0], Value::Str(s) if &**s == "e42"));
    // NDJSON round-trip preserves those ids (dump contains them, reload keeps).
    let dump = crate::ndjson::to_ndjson(&store);
    assert!(dump.contains("\"id\":\"alice\"") && dump.contains("\"id\":\"e42\""));
    assert_eq!(
        crate::ndjson::to_ndjson(&crate::ndjson::from_ndjson(&dump).unwrap()),
        dump
    );
}

/// `type(edge)` and the list-algebra functions (previously deferred) match
/// hand-computed values.
#[test]
fn type_and_list_algebra_functions() {
    let mut b = Builder::default();
    let x = b.node(&["N"], &[]);
    let y = b.node(&["N"], &[]);
    b.edge(x, y, "KNOWS");
    let store = b.build();
    // type(edge)
    let t = run(
        &crate::gql::parse("MATCH (a:N)-[r:KNOWS]->(b) RETURN type(r) AS t").unwrap(),
        &store,
    );
    assert!(matches!(&t.rows[0][0], Value::Str(s) if &**s == "KNOWS"));

    let list = |e: &str| -> Vec<String> {
        let q = format!("MATCH (a:N) RETURN {e} AS v LIMIT 1");
        match &run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0] {
            Value::List(v) => v.iter().map(|x| format!("{x:?}")).collect(),
            o => panic!("{e} → {o:?}"),
        }
    };
    let one = |e: &str| -> Value {
        let q = format!("MATCH (a:N) RETURN {e} AS v LIMIT 1");
        run(&crate::gql::parse(&q).unwrap(), &store).rows[0][0].clone()
    };
    let dbg = |xs: &[Value]| -> Vec<String> { xs.iter().map(|x| format!("{x:?}")).collect() };
    assert_eq!(list("append([1, 2], 3)"), dbg(&[n(1.0), n(2.0), n(3.0)]));
    assert!(matches!(one("list_contains([1, 2, 3], 2)"), Value::Num(x) if x == 1.0));
    assert!(matches!(one("list_contains([1, 2], 5)"), Value::Num(x) if x == 0.0));
    assert!(matches!(one("list_contains([1, null], null)"), Value::Num(x) if x == 1.0));
    assert_eq!(list("list_sort([3, 1, 2])"), dbg(&[n(1.0), n(2.0), n(3.0)]));
    assert_eq!(
        list("list_union([1, 1, 2], [2, 3])"),
        dbg(&[n(1.0), n(2.0), n(3.0)])
    );
    assert_eq!(
        list("difference([1, 1, 2, 3], [2])"),
        dbg(&[n(1.0), n(3.0)])
    );
    assert_eq!(
        list("intersection([1, 2, 2, 3], [2, 3, 4])"),
        dbg(&[n(2.0), n(3.0)])
    );
}

/// Late materialization (sorted top-K over a Project) returns the SAME rows as
/// the eager path — the non-key column is projected only for the survivors.
#[test]
fn late_materialize_top_k_matches_eager() {
    let mut b = Builder::default();
    for i in 0..50u32 {
        b.node(
            &["P"],
            &[("age", n(f64::from(i % 10))), ("name", s(&format!("p{i}")))],
        );
    }
    let store = b.build();
    // Top-3 by age DESC, then name — a non-key column (name) is projected.
    let q = "MATCH (p:P) RETURN p.name AS name, p.age AS age ORDER BY age DESC, name LIMIT 3";
    let got = run(&crate::gql::parse(q).unwrap(), &store);
    // Highest ages are 9 (p9, p19, p29, p39, p49) → name-sorted first three.
    assert_eq!(names_of(&got, 0), vec!["p19", "p29", "p39"]);
    assert!(got
        .rows
        .iter()
        .all(|r| matches!(r[1], Value::Num(x) if x == 9.0)));
    // With SKIP: rows 3..6 of the same order.
    let q2 =
        "MATCH (p:P) RETURN p.name AS name, p.age AS age ORDER BY age DESC, name SKIP 3 LIMIT 2";
    assert_eq!(
        names_of(&run(&crate::gql::parse(q2).unwrap(), &store), 0),
        vec!["p49", "p9"]
    );
}

/// A low-cardinality string column dictionary-encodes, and every read shape
/// (DISTINCT / GROUP BY / equality filter / ORDER BY) returns exactly what the
/// plain `Str` column would — while a high-cardinality column stays `Str`.
#[test]
fn dict_encoded_column_round_trips() {
    let depts = ["eng", "sales", "ops"];
    let mut b = Builder::default();
    for i in 0..30u32 {
        b.node(
            &["P"],
            &[
                ("dept", s(depts[i as usize % 3])),
                ("name", s(&format!("p{i}"))), // 30 distinct -> stays Str
            ],
        );
    }
    let store = b.build();
    // The low-card column encoded; the high-card one did not.
    assert!(matches!(
        store.column("dept"),
        Some(crate::store::Column::Dict { .. })
    ));
    assert!(matches!(
        store.column("name"),
        Some(crate::store::Column::Str { .. })
    ));

    let rows = |q: &str| {
        let mut r: Vec<String> = names_of(&run(&crate::gql::parse(q).unwrap(), &store), 0);
        r.sort();
        r
    };
    // DISTINCT over the dict column.
    assert_eq!(
        rows("MATCH (n:P) RETURN DISTINCT n.dept AS d"),
        vec!["eng", "ops", "sales"]
    );
    // GROUP BY the dict column: 10 of each.
    let g = run(
        &crate::gql::parse("MATCH (n:P) RETURN n.dept AS d, count(*) AS c").unwrap(),
        &store,
    );
    assert_eq!(g.rows.len(), 3);
    assert!(g
        .rows
        .iter()
        .all(|r| matches!(r[1], Value::Num(x) if x == 10.0)));
    // count(DISTINCT) over the dict column.
    let c = run(
        &crate::gql::parse("MATCH (n:P) RETURN count(DISTINCT n.dept) AS c").unwrap(),
        &store,
    );
    assert!(matches!(c.rows[0][0], Value::Num(x) if x == 3.0));
    // Equality filter resolves through the dict; a miss matches nothing.
    let count_where = |q: &str| match run(&crate::gql::parse(q).unwrap(), &store).rows[0][0] {
        Value::Num(x) => x,
        _ => panic!("count is not a number"),
    };
    assert_eq!(
        count_where("MATCH (n:P) WHERE n.dept = 'eng' RETURN count(*) AS c"),
        10.0
    );
    assert_eq!(
        count_where("MATCH (n:P) WHERE n.dept = 'zzz' RETURN count(*) AS c"),
        0.0
    );
    // ORDER BY the dict column sorts by VALUE, not code.
    let o = run(
        &crate::gql::parse("MATCH (n:P) RETURN DISTINCT n.dept AS d ORDER BY d").unwrap(),
        &store,
    );
    assert_eq!(names_of(&o, 0), vec!["eng", "ops", "sales"]);
}

/// Writing a value to a dict-encoded column decodes it to `Str` in place, and the
/// new value reads back correctly alongside the untouched ones.
#[test]
fn dict_column_decodes_on_write() {
    let mut b = Builder::default();
    for _ in 0..6u32 {
        b.node(&["P"], &[("dept", s("eng"))]);
    }
    let mut store = b.build();
    assert!(matches!(
        store.column("dept"),
        Some(crate::store::Column::Dict { .. })
    ));
    let id = store.nodes_with_label("P")[0];
    store.set_prop(id, "dept", s("legal"));
    assert!(matches!(
        store.column("dept"),
        Some(crate::store::Column::Str { .. })
    ));
    assert!(matches!(store.prop(id, "dept"), Value::Str(x) if &*x == "legal"));
    let other = store.nodes_with_label("P")[1];
    assert!(matches!(store.prop(other, "dept"), Value::Str(x) if &*x == "eng"));
}

/// Multi-column `DISTINCT` over a dict-encoded string column plus a numeric one
/// (and an absent cell) dedups on the composite code+bits key exactly as the
/// general byte-key path would — same distinct tuples, absence as its own value.
#[test]
fn multi_col_distinct_over_dict_and_num() {
    let depts = ["eng", "sales"];
    let mut b = Builder::default();
    // 20 rows: dept in {eng,sales} (cycles every row), age in {30,40} (flips
    // every 2 rows) — decoupled, so all 4 present tuples occur...
    for i in 0..20u32 {
        b.node(
            &["P"],
            &[
                ("dept", s(depts[i as usize % 2])),
                ("age", n(f64::from(30 + ((i / 2) % 2) * 10))),
            ],
        );
    }
    // ...plus two rows whose dept is ABSENT (age 30) -> a 5th tuple (Null, 30).
    b.node(&["P"], &[("age", n(30.0))]);
    b.node(&["P"], &[("age", n(30.0))]);
    let store = b.build();
    assert!(matches!(
        store.column("dept"),
        Some(crate::store::Column::Dict { .. })
    ));

    let out = run(
        &crate::gql::parse("MATCH (n:P) RETURN DISTINCT n.dept AS d, n.age AS age").unwrap(),
        &store,
    );
    // Render each (dept, age) tuple to a stable string and compare as a set.
    let mut got: Vec<String> = out
        .rows
        .iter()
        .map(|r| format!("{:?}|{:?}", r[0], r[1]))
        .collect();
    got.sort();
    let mut want = vec![
        format!("{:?}|{:?}", Value::Str("eng".into()), Value::Num(30.0)),
        format!("{:?}|{:?}", Value::Str("eng".into()), Value::Num(40.0)),
        format!("{:?}|{:?}", Value::Str("sales".into()), Value::Num(30.0)),
        format!("{:?}|{:?}", Value::Str("sales".into()), Value::Num(40.0)),
        format!("{:?}|{:?}", Value::Null, Value::Num(30.0)),
    ];
    want.sort();
    assert_eq!(got, want);
}

// --- relational core (unchanged behavior, now slot-addressed) ---

#[test]
fn scan_label_and_project() {
    let store = social();
    let out = run(
        &scan("Person").project(vec![("name".into(), prop(0, "name"))]),
        &store,
    );
    assert_eq!(out.rows.len(), 3);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["alice", "bob", "carol"]);
}

#[test]
fn cast_projects_per_row() {
    let store = social();
    // Cast each Person's numeric age to INTEGER (identity here; the ages are
    // already whole) — verifies the per-row Cast arm wires through Project.
    let plan = scan("Person").project(vec![(
        "a".into(),
        Expr::Cast {
            target: crate::ir::CastTarget::Integer,
            expr: Box::new(prop(0, "age")),
        },
    )]);
    let out = run(&plan, &store);
    let mut got: Vec<f64> = out
        .rows
        .iter()
        .map(|r| match r[0] {
            Value::Num(x) => x,
            ref o => panic!("expected Num, got {o:?}"),
        })
        .collect();
    got.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    assert_eq!(got, vec![25.0, 30.0, 40.0]);
}

#[test]
fn cast_fault_surfaces_through_try_run() {
    let store = social();
    // "alice" has no numeric form → the CAST throws E_INVALID_VALUE, and the
    // fallible `try_run` returns that Err (this is why the read pipeline
    // threads Result at all; `run` would panic on the same plan).
    let plan = scan("Person").project(vec![(
        "n".into(),
        Expr::Cast {
            target: crate::ir::CastTarget::Integer,
            expr: Box::new(prop(0, "name")),
        },
    )]);
    let err = try_run(&plan, &store).unwrap_err();
    assert!(err.contains("E_INVALID_VALUE"), "got: {err}");
}

#[test]
fn is_null_projects_definite_bools() {
    // A scan of all nodes: the three Persons carry `age`, the Project node
    // does not. `age IS NULL` must be a definite Bool for EVERY row (never a
    // Null/UNKNOWN), TRUE only where the value is absent.
    let store = social();
    let plan = Plan::Scan { label: None }.project(vec![(
        "n".into(),
        Expr::IsNull {
            expr: Box::new(prop(0, "age")),
            negated: false,
        },
    )]);
    let out = run(&plan, &store);
    // Every value is a concrete boolean — none is Null.
    assert!(out.rows.iter().all(|r| matches!(r[0], Value::Bool(_))));
    let trues = out
        .rows
        .iter()
        .filter(|r| matches!(r[0], Value::Bool(true)))
        .count();
    assert_eq!(trues, 1); // only the Project node lacks `age`
}

#[test]
fn property_exists_separates_present_null_from_absent() {
    // node 0: age present-null, node 1: age absent. PROPERTY_EXISTS is a
    // presence test, so it is TRUE for the present-null and FALSE for absent —
    // the distinction `IS NOT NULL` (both FALSE) cannot draw.
    let mut b = Builder::default();
    b.node(&["P"], &[("name", s("null"))]);
    b.node(&["P"], &[("name", s("absent"))]);
    let mut store = b.build();
    store.set_prop(0, "age", Value::Null);

    let exists = Plan::Scan {
        label: Some("P".into()),
    }
    .project(vec![(
        "e".into(),
        Expr::PropertyExists {
            slot: 0,
            key: "age".into(),
        },
    )]);
    let out = run(&exists, &store);
    assert!(matches!(out.rows[0][0], Value::Bool(true))); // present-null
    assert!(matches!(out.rows[1][0], Value::Bool(false))); // absent
}

/// PROPERTY_EXISTS works on an EDGE slot (not just nodes), and a NULL element
/// (the OPTIONAL unmatched sentinel) yields NULL, not FALSE — matching the TS engine.
#[test]
fn property_exists_on_edges_and_null_element() {
    let nd = concat!(
        "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{}}\n",
        "{\"from\":\"a\",\"to\":\"b\",\"labels\":[\"R\"],\"props\":{\"w\":3}}"
    );
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let val = |q: &str| -> Value {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        run(&plan, &store).rows[0][0].clone()
    };
    // edge carries `w`, not `gone`.
    assert!(matches!(
        val("MATCH ()-[e:R]->() RETURN property_exists(e, w) AS x"),
        Value::Bool(true)
    ));
    assert!(matches!(
        val("MATCH ()-[e:R]->() RETURN property_exists(e, gone) AS x"),
        Value::Bool(false)
    ));
    // OPTIONAL MATCH that finds nothing → m is NULL → property_exists is NULL.
    assert!(
        val("MATCH (n:N) OPTIONAL MATCH (n)-[:NOSUCH]->(m) RETURN property_exists(m, x) AS x")
            .is_null()
    );
}

#[test]
fn filter_numeric_then_project() {
    let store = social();
    let plan = scan("Person")
        .filter(cmp(CompareOp::Gt, prop(0, "age"), lit(n(28.0))))
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["alice", "carol"]);
}

#[test]
fn absent_property_is_null_and_filters_as_unknown() {
    let store = social();
    // Project has no age → `age >= 0` is UNKNOWN for it → dropped.
    let plan = Plan::Scan { label: None }
        .filter(cmp(CompareOp::Ge, prop(0, "age"), lit(n(0.0))))
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 3);
}

#[test]
fn equality_is_cross_type_false() {
    let store = social();
    let plan = Plan::Scan { label: None }
        .filter(cmp(CompareOp::Eq, prop(0, "age"), lit(s("30"))))
        .project(vec![("name".into(), prop(0, "name"))]);
    assert_eq!(run(&plan, &store).rows.len(), 0);
}

/// A hand-built `Insert` plan writes nodes and edges through `execute`.
#[test]
fn execute_insert_writes_store() {
    use crate::ir::{InsertEdge, InsertNode};
    let mut store = Builder::default().build();
    let plan = Plan::Insert {
        nodes: vec![
            InsertNode {
                labels: vec!["P".into()],
                props: vec![("name".into(), s("a"))],
            },
            InsertNode {
                labels: vec!["P".into()],
                props: vec![],
            },
        ],
        edges: vec![InsertEdge {
            from: 0,
            to: 1,
            etype: "R".into(),
            props: vec![],
        }],
    };
    let out = execute(&plan, &mut store).unwrap();
    assert_eq!(out.rows.len(), 0); // a write returns no rows
    assert_eq!(store.node_count(), 2);
    assert_eq!(store.nodes_with_label("P"), &[0, 1]);
    assert_eq!(store.out(0).len(), 1);
    assert_eq!(store.out(0)[0].nbr, 1);
    assert!(matches!(store.prop(0, "name"), Value::Str(x) if &*x == "a"));
}

/// A hand-built `Update` plan sets and removes properties on matched nodes.
/// SET carol.age = 41; REMOVE alice.age — over a Person scan.
#[test]
fn execute_update_sets_and_removes() {
    use crate::ir::SetOp;
    let mut store = social();
    let plan = Plan::Update {
        input: Box::new(scan("Person")),
        ops: vec![
            SetOp::Set {
                slot: 0,
                key: "seen".into(),
                value: lit(n(1.0)),
            },
            SetOp::Remove {
                slot: 0,
                key: "age".into(),
            },
        ],
    };
    execute(&plan, &mut store).unwrap();
    // every Person got seen=1 and lost age
    for id in 0..3u32 {
        assert!(matches!(store.prop(id, "seen"), Value::Num(x) if x == 1.0));
        assert!(store.prop(id, "age").is_null());
    }
}

/// INSERT enforces unique constraints: the second insert of the same key
/// errors and is rolled back (the graph keeps exactly the first node).
#[test]
fn insert_enforces_unique_constraint() {
    use crate::ir::{InsertNode, Plan};
    let mut store = Builder::default().build();
    store.create_unique_constraint("User", &["email"]).unwrap();
    let ins = |email: &str| Plan::Insert {
        nodes: vec![InsertNode {
            labels: vec!["User".into()],
            props: vec![("email".into(), s(email))],
        }],
        edges: vec![],
    };
    assert!(execute(&ins("a@x"), &mut store).is_ok());
    let err = execute(&ins("a@x"), &mut store); // duplicate
    assert!(err.is_err());
    // rolled back: still exactly one User, and node_count did not grow.
    assert_eq!(store.node_count(), 1);
    assert_eq!(store.nodes_with_label("User").len(), 1);
    // a different key still inserts fine.
    assert!(execute(&ins("b@x"), &mut store).is_ok());
    assert_eq!(store.node_count(), 2);
}

/// A validator (a per-element `CHECK` predicate) is enforced on write and
/// rolled back on violation; a null/absent value passes (SQL-`CHECK` semantics).
#[test]
fn validator_enforced_on_write() {
    let mut store = Builder::default().build();
    apply_schema_op(
        &mut store,
        r#"{"op":"validator","label":"P","var":"p","predicate":"p.age >= 0"}"#,
    )
    .unwrap();
    let ins = |gql: &str| crate::gql::parse(gql).unwrap();
    assert!(execute(&ins("INSERT (:P {age: 5})"), &mut store).is_ok());
    let err = execute(&ins("INSERT (:P {age: -1})"), &mut store).unwrap_err();
    assert!(err.starts_with("E_VALIDATOR"), "{err}");
    assert_eq!(
        store.nodes_with_label("P").len(),
        1,
        "violating insert rolled back"
    );
    // A null age passes (unknown, not false).
    assert!(execute(&ins("INSERT (:P {name: 'x'})"), &mut store).is_ok());
}

/// Declaring a validator the current data already breaks is rejected.
#[test]
fn validator_rejects_existing_violation() {
    let mut store = Builder::default().build();
    execute(
        &crate::gql::parse("INSERT (:P {age: -5})").unwrap(),
        &mut store,
    )
    .unwrap();
    let err = apply_schema_op(
        &mut store,
        r#"{"op":"validator","label":"P","var":"p","predicate":"p.age >= 0"}"#,
    );
    assert!(
        matches!(err, Err(crate::schema_op::SchemaError::Rejected(_))),
        "{err:?}"
    );
}

/// An invariant (a whole-graph query) is enforced on write; a boolean-`false`
/// cell in its result rolls the write back.
#[test]
fn invariant_enforced_on_write() {
    let mut store = Builder::default().build();
    apply_schema_op(
        &mut store,
        r#"{"op":"invariant","name":"nonneg","query":"MATCH (p:P) RETURN p.age >= 0"}"#,
    )
    .unwrap();
    let ins = |gql: &str| crate::gql::parse(gql).unwrap();
    assert!(execute(&ins("INSERT (:P {age: 1})"), &mut store).is_ok());
    let err = execute(&ins("INSERT (:P {age: -1})"), &mut store).unwrap_err();
    assert!(err.starts_with("E_INVARIANT"), "{err}");
    assert_eq!(
        store.nodes_with_label("P").len(),
        1,
        "violating insert rolled back"
    );
}

/// A bad validator predicate / invariant query is a syntax error at declaration.
#[test]
fn bad_predicate_and_query_are_syntax_errors() {
    let mut store = Builder::default().build();
    let v = apply_schema_op(
        &mut store,
        r#"{"op":"validator","label":"P","var":"p","predicate":"p.age >=>= 0"}"#,
    );
    assert!(
        matches!(v, Err(crate::schema_op::SchemaError::Syntax(_))),
        "{v:?}"
    );
    let i = apply_schema_op(
        &mut store,
        r#"{"op":"invariant","name":"x","query":"NOT A QUERY"}"#,
    );
    assert!(
        matches!(i, Err(crate::schema_op::SchemaError::Syntax(_))),
        "{i:?}"
    );
}

/// A DISTINCT aggregate (other than count) dedups its values per group before
/// folding — `collect_list(DISTINCT …)`/`min(DISTINCT …)` were folding over
/// duplicates. Covers the keyless fast-path (which used to skip the dedup) too.
#[test]
fn distinct_aggregate_dedups_values() {
    let mut b = Builder::default();
    b.node(&["T"], &[("g", s("a"))]);
    b.node(&["T"], &[("g", s("a"))]);
    b.node(&["T"], &[("g", s("b"))]);
    let st = b.build();
    let list_len = |q: &str| match &run(&crate::gql::parse(q).unwrap(), &st).rows[0][0] {
        Value::List(items) => items.len(),
        o => panic!("expected a list, got {o:?}"),
    };
    // Two distinct `g` values ("a","b"); a constant collapses to one.
    assert_eq!(
        list_len("MATCH (n:T) RETURN collect_list(DISTINCT n.g) AS x"),
        2
    );
    assert_eq!(
        list_len("MATCH (n:T) RETURN collect_list(DISTINCT true) AS x"),
        1
    );
    // Grouped: each group dedups independently (here one group of 2 distinct).
    assert_eq!(
        list_len("MATCH (n:T) RETURN collect_list(DISTINCT n.g) AS x"),
        2
    );
}

/// A non-boolean WHERE/FILTER value is a data exception — a number / string / map is
/// NOT coerced to a truth value (strict typing; `CAST AS BOOLEAN` / `to_boolean`
/// converts). Two paths, matching Postgres and the pure-TS engine:
///   - STATIC: a value whose type is known at parse time (a literal, arithmetic, a
///     map/list constructor) is rejected during PARSE, even on an empty match.
///   - DYNAMIC: a property / function result — type unknown in a schemaless engine —
///     parses, then throws per-row at evaluation via `as_truth`.
#[test]
fn where_rejects_a_non_boolean_condition() {
    let mut b = Builder::default();
    b.node(&["T"], &[("n", n(1.0)), ("s", s("x"))]);
    b.node(&["T"], &[("n", n(0.0)), ("s", s(""))]);
    let st = b.build();
    // STATIC: the parse itself fails (no row data needed).
    let parse_errs = |q: &str| {
        crate::gql::parse(q)
            .unwrap_err()
            .contains("E_INVALID_VALUE")
    };
    assert!(parse_errs("MATCH (n:T) WHERE 5 RETURN n.n AS x"));
    assert!(parse_errs(
        "MATCH (n:T) WHERE (n.n > 0 AND 100) RETURN n.n AS x"
    ));
    assert!(parse_errs("MATCH (n:T) WHERE (n.n + 1) RETURN n.n AS x"));
    assert!(parse_errs("MATCH (n:T) WHERE {a: n.n} RETURN n.n AS x"));
    // STATIC fires even when the pattern matches NOTHING (plan-time, like Postgres).
    assert!(parse_errs("MATCH (n:NONE) WHERE 5 RETURN n.n AS x"));
    // DYNAMIC: a bare property parses, then throws at runtime.
    let run_errs = |q: &str| {
        try_run(&crate::gql::parse(q).unwrap(), &st)
            .unwrap_err()
            .contains("E_INVALID_VALUE")
    };
    assert!(run_errs("MATCH (n:T) WHERE n.n RETURN n.n AS x"));
    assert!(run_errs("MATCH (n:T) WHERE n.s RETURN n.s AS x"));
    // A string-returning function is a Call — dynamic (not statically classified),
    // so `n.s || 'q'` (concat) is caught per-row at runtime, not at parse.
    assert!(run_errs("MATCH (n:T) WHERE (n.s || 'q') RETURN n.n AS x"));
    // A proper boolean condition still works; null is UNKNOWN (allowed), not an error.
    let count = |q: &str| run(&crate::gql::parse(q).unwrap(), &st).rows.len();
    assert_eq!(count("MATCH (n:T) WHERE n.n > 0 RETURN n.n AS x"), 1);
    assert_eq!(count("MATCH (n:T) WHERE null RETURN n.n AS x"), 0);
    assert_eq!(
        count("MATCH (n:T) WHERE CAST(n.n AS BOOLEAN) RETURN n.n AS x"),
        1
    );
}

/// A temporal renders TAGGED in a query result (`{"@duration":"P1D"}`), matching
/// the TS engine — not a bare ISO string. Covers every temporal kind.
#[test]
fn query_result_renders_temporals_tagged() {
    let mut b = Builder::default();
    b.node(&["T"], &[]);
    let st = b.build();
    let json = try_run_gql_json(
        &crate::gql::parse(
            "MATCH (n:T) RETURN duration('P1D') AS a, date('2020-01-01') AS b, \
                 local_time('08:30:00') AS c",
        )
        .unwrap(),
        &st,
    )
    .unwrap();
    assert!(json.contains(r#"{"@duration":"P1D"}"#), "{json}");
    assert!(json.contains(r#"{"@date":"2020-01-01"}"#), "{json}");
    assert!(json.contains(r#"{"@localtime":"08:30:00"}"#), "{json}");
}

/// INSERT accepts a record/map literal as a property value (a constant record),
/// stored canonically as a `Value::Record` — the seedable-literal path handles
/// `{…}`, not just scalars and lists.
#[test]
fn insert_writes_a_record_literal_property() {
    let mut store = Builder::default().build();
    execute(
        &crate::gql::parse("INSERT (:P {n: 1, m: {y: 'hi', x: 2}})").unwrap(),
        &mut store,
    )
    .unwrap();
    match store.prop(0, "m") {
        Value::Record(f) => {
            // Canonical: keys sorted (x before y), values preserved.
            assert_eq!(f.len(), 2);
            assert_eq!(f[0].0.as_ref(), "x");
            assert_eq!(format!("{:?}", f[0].1), format!("{:?}", Value::Num(2.0)));
            assert_eq!(f[1].0.as_ref(), "y");
        }
        other => panic!("expected a record, got {other:?}"),
    }
    // A nested field is queryable back out.
    let out = run(
        &crate::gql::parse("MATCH (p:P) RETURN p.m.x AS x").unwrap(),
        &store,
    );
    assert_eq!(
        format!("{:?}", out.rows[0][0]),
        format!("{:?}", Value::Num(2.0))
    );
}

/// A single INSERT that creates two colliding nodes is rejected atomically.
#[test]
fn insert_rejects_intra_statement_duplicate() {
    use crate::ir::{InsertNode, Plan};
    let mut store = Builder::default().build();
    store.create_unique_constraint("User", &["email"]).unwrap();
    let plan = Plan::Insert {
        nodes: vec![
            InsertNode {
                labels: vec!["User".into()],
                props: vec![("email".into(), s("same"))],
            },
            InsertNode {
                labels: vec!["User".into()],
                props: vec![("email".into(), s("same"))],
            },
        ],
        edges: vec![],
    };
    assert!(execute(&plan, &mut store).is_err());
    assert_eq!(store.node_count(), 0); // both rolled back
}

/// `MATCH (a),(b) INSERT (a)-[:E]->(b)` CONNECTS the matched nodes — a bare pattern node
/// naming a bound variable is a reference, not a fresh node. Previously it created two
/// NEW nodes and an edge between them, leaving the matched nodes untouched (which also
/// silently skipped their edge-cardinality constraints, since no edge reached them).
#[test]
fn insert_edge_between_bound_match_vars_connects_them() {
    let nd = "{\"id\":\"1\",\"labels\":[\"A\"],\"props\":{\"id\":\"a1\"}}\n\
                  {\"id\":\"2\",\"labels\":[\"B\"],\"props\":{\"id\":\"b1\"}}";
    let mut store = crate::ndjson::from_ndjson(nd).unwrap();
    let plan = crate::opt::optimize_indexed(
        crate::gql::parse("MATCH (a:A {id:'a1'}), (b:B {id:'b1'}) INSERT (a)-[:E]->(b)").unwrap(),
        &store,
    );
    execute(&plan, &mut store).unwrap();
    // No new nodes (still 2), exactly one edge, and it joins the two MATCHED nodes.
    assert_eq!(store.node_count(), 2);
    assert_eq!(store.edge_count(), 1);
}

/// A `SET` that collides with a unique constraint is REJECTED and rolled back —
/// the Update path enforces constraints like INSERT/_MERGE, not silently apply.
#[test]
fn set_enforces_unique_constraint() {
    let mut b = Builder::default();
    b.node(&["User"], &[("email", s("a@x"))]);
    b.node(&["User"], &[("email", s("b@x"))]);
    let mut store = b.build();
    store.create_unique_constraint("User", &["email"]).unwrap();
    let go = |q: &str, store: &mut Store| execute(&crate::gql::parse(q).unwrap(), store);

    // Colliding SET → error, rolled back (still exactly one 'a@x').
    assert!(go(
        "MATCH (u:User) WHERE u.email='b@x' SET u.email='a@x'",
        &mut store
    )
    .is_err());
    let count = |store: &Store, v: &str| {
        store
            .nodes_with_label("User")
            .iter()
            .filter(|&&n| matches!(store.prop(n, "email"), Value::Str(e) if &*e == v))
            .count()
    };
    assert_eq!(count(&store, "a@x"), 1, "collision must have rolled back");
    assert_eq!(count(&store, "b@x"), 1);
    // A non-colliding SET still applies.
    assert!(go(
        "MATCH (u:User) WHERE u.email='b@x' SET u.email='c@x'",
        &mut store
    )
    .is_ok());
    assert_eq!(count(&store, "c@x"), 1);
}

// ---- ISO transaction-control keywords (START TRANSACTION / COMMIT / ROLLBACK) ----

/// Run one GQL statement exactly as `lnk_query`'s GQL path does — through the
/// shared [`run_query`] dispatcher (transaction control, READ ONLY enforcement,
/// write/read split), materializing a returned read the way the FFI read path
/// streams it. So these tests exercise the real integration, not the pieces.
fn stmt(store: &mut Store, q: &str) -> Result<Rows, String> {
    let plan = crate::gql::parse(q)?;
    match run_query(plan, store)? {
        Executed::Rows(rows) => Ok(rows),
        Executed::Read(plan) => Ok(run(&plan, store)),
    }
}

/// Parse `q` and extract the `(kind, read_only)` of the resulting `TxControl`
/// plan (panicking if it is not one). `Plan` has no `PartialEq`, so the parse
/// tests compare the extracted parts, not whole plans.
fn tx_parts(q: &str) -> (TxKind, bool) {
    match crate::gql::parse(q).unwrap_or_else(|e| panic!("parse `{q}`: {e}")) {
        Plan::TxControl { kind, read_only } => (kind, read_only),
        other => panic!("expected TxControl for `{q}`, got {other:?}"),
    }
}

#[test]
fn tx_keywords_parse_to_the_right_plan() {
    assert_eq!(tx_parts("START TRANSACTION"), (TxKind::Start, false));
    assert_eq!(
        tx_parts("START TRANSACTION READ ONLY"),
        (TxKind::Start, true)
    );
    // Case-insensitive; READ WRITE is the (default) read-write mode.
    assert_eq!(
        tx_parts("start transaction read write"),
        (TxKind::Start, false)
    );
    assert_eq!(tx_parts("COMMIT"), (TxKind::Commit, false));
    assert_eq!(tx_parts("COMMIT WORK"), (TxKind::Commit, false));
    assert_eq!(tx_parts("ROLLBACK"), (TxKind::Rollback, false));
    assert_eq!(tx_parts("ROLLBACK WORK"), (TxKind::Rollback, false));
}

#[test]
fn tx_keyword_parse_errors() {
    // START without TRANSACTION, and a bad access mode, are syntax errors.
    assert!(crate::gql::parse("START").is_err());
    assert!(crate::gql::parse("START TRANSACTION READ").is_err());
    assert!(crate::gql::parse("START TRANSACTION READ SOMETIMES").is_err());
    // Trailing input after a complete command is rejected.
    assert!(crate::gql::parse("COMMIT EXTRA").is_err());
}

#[test]
fn commit_keyword_persists_the_transactions_writes() {
    let mut store = Builder::default().build();
    assert!(!store.in_transaction());
    stmt(&mut store, "START TRANSACTION").unwrap();
    assert!(store.in_transaction());
    stmt(&mut store, "INSERT (:Acct {bal: 100})").unwrap();
    stmt(&mut store, "INSERT (:Acct {bal: 200})").unwrap();
    stmt(&mut store, "COMMIT").unwrap();
    assert!(!store.in_transaction(), "COMMIT closes the transaction");
    assert_eq!(store.live_node_count(), 2, "both inserts persisted");
}

#[test]
fn rollback_keyword_discards_the_transactions_writes() {
    let mut store = Builder::default().build();
    stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap(); // committed implicitly (no tx)
    stmt(&mut store, "START TRANSACTION").unwrap();
    stmt(&mut store, "INSERT (:Acct {bal: 100})").unwrap();
    stmt(&mut store, "INSERT (:Acct {bal: 200})").unwrap();
    stmt(&mut store, "ROLLBACK").unwrap();
    assert!(!store.in_transaction());
    assert_eq!(
        store.live_node_count(),
        1,
        "only the pre-transaction insert survives"
    );
}

#[test]
fn transaction_state_persists_across_separate_statements() {
    // The store IS the session: a START stays open across statement boundaries.
    let mut store = Builder::default().build();
    stmt(&mut store, "START TRANSACTION").unwrap();
    stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap();
    assert!(store.in_transaction(), "still open between statements");
    stmt(&mut store, "INSERT (:Acct {bal: 2})").unwrap();
    assert!(store.in_transaction());
    stmt(&mut store, "COMMIT").unwrap();
    assert_eq!(store.live_node_count(), 2);
}

#[test]
fn nested_start_transaction_is_a_coded_error() {
    let mut store = Builder::default().build();
    stmt(&mut store, "START TRANSACTION").unwrap();
    let err = stmt(&mut store, "START TRANSACTION").unwrap_err();
    assert!(
        err.starts_with("E_INVALID_GRAPH_OP:"),
        "nested START is E_INVALID_GRAPH_OP, got: {err}"
    );
    assert!(store.in_transaction(), "the original tx is untouched");
    stmt(&mut store, "ROLLBACK").unwrap(); // clean up
}

#[test]
fn commit_or_rollback_with_no_active_transaction_is_a_coded_error() {
    let mut store = Builder::default().build();
    let c = stmt(&mut store, "COMMIT").unwrap_err();
    assert!(c.starts_with("E_INVALID_GRAPH_OP:"), "COMMIT no-tx: {c}");
    let r = stmt(&mut store, "ROLLBACK").unwrap_err();
    assert!(r.starts_with("E_INVALID_GRAPH_OP:"), "ROLLBACK no-tx: {r}");
}

#[test]
fn read_only_transaction_rejects_writes_but_allows_reads() {
    let mut store = Builder::default().build();
    stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap(); // seed (no tx)
    stmt(&mut store, "START TRANSACTION READ ONLY").unwrap();
    assert!(store.tx_read_only());
    // A read is allowed.
    assert!(stmt(&mut store, "MATCH (n:Acct) RETURN n.bal").is_ok());
    // Every write kind is rejected with the coded error, and nothing changes.
    for w in [
        "INSERT (:Acct {bal: 9})",
        "MATCH (n:Acct) SET n.bal = 5",
        "MATCH (n:Acct) REMOVE n.bal",
        "MATCH (n:Acct) DELETE n",
    ] {
        let e = stmt(&mut store, w).unwrap_err();
        assert!(e.starts_with("E_INVALID_GRAPH_OP:"), "{w} → {e}");
    }
    assert_eq!(
        store.live_node_count(),
        1,
        "read-only left the graph intact"
    );
    stmt(&mut store, "COMMIT").unwrap();
    assert!(!store.tx_read_only(), "COMMIT clears the read-only mode");
    // After commit the mode is cleared — a write applies.
    stmt(&mut store, "INSERT (:Acct {bal: 9})").unwrap();
    assert_eq!(store.live_node_count(), 2);
}

#[test]
fn rollback_clears_the_read_only_mode() {
    let mut store = Builder::default().build();
    stmt(&mut store, "START TRANSACTION READ ONLY").unwrap();
    assert!(store.tx_read_only());
    stmt(&mut store, "ROLLBACK").unwrap();
    assert!(!store.tx_read_only(), "ROLLBACK clears read-only");
    stmt(&mut store, "INSERT (:Acct {bal: 1})").unwrap(); // now allowed
    assert_eq!(store.live_node_count(), 1);
}

#[test]
fn an_immediate_fault_inside_a_transaction_isolates_to_its_own_statement() {
    // A string-`id` collision is an IMMEDIATE fault (the element's identity), so it
    // rolls back only ITS statement's savepoint. The app can swallow it and the
    // writes around it still commit with the transaction — the "skip the bad row,
    // commit the good ones" pattern. (Declared constraints DEFER to commit instead;
    // see the next test.)
    let mut store = Builder::default().build();
    stmt(&mut store, "INSERT (:User {id: 'taken'})").unwrap(); // external id 'taken'
    stmt(&mut store, "START TRANSACTION").unwrap();
    stmt(&mut store, "INSERT (:User {id: 'a'})").unwrap();
    // Collides with the existing 'taken' id → an IMMEDIATE fault; the app ignores it.
    assert!(stmt(&mut store, "INSERT (:User {id: 'taken'})").is_err());
    stmt(&mut store, "INSERT (:User {id: 'b'})").unwrap();
    stmt(&mut store, "COMMIT").unwrap();

    assert!(
        store.node_by_ext("a").is_some(),
        "the write before the fault committed"
    );
    assert!(
        store.node_by_ext("b").is_some(),
        "the write after the fault committed"
    );
    assert!(store.node_by_ext("taken").is_some());
    assert_eq!(store.live_node_count(), 3, "taken + a + b (no duplicate)");
}

#[test]
fn a_deferred_constraint_violation_surfaces_at_commit_and_rolls_the_whole_transaction_back() {
    // A DECLARED unique constraint is checked at COMMIT (deferred), matching the TS engine.
    // So the colliding write itself SUCCEEDS mid-transaction, and the violation
    // surfaces only at COMMIT — rolling back the WHOLE transaction (you cannot
    // swallow it row-by-row, unlike an immediate fault).
    let mut b = Builder::default();
    b.node(&["User"], &[("email", s("taken@x"))]);
    let mut store = b.build();
    store.create_unique_constraint("User", &["email"]).unwrap();

    stmt(&mut store, "START TRANSACTION").unwrap();
    stmt(&mut store, "INSERT (:User {email: 'a@x'})").unwrap();
    // Deferred: the duplicate insert itself does NOT fault here.
    stmt(&mut store, "INSERT (:User {email: 'taken@x'})")
        .expect("a deferred unique constraint does not fault at the statement");
    stmt(&mut store, "INSERT (:User {email: 'b@x'})").unwrap();
    let commit = stmt(&mut store, "COMMIT");
    assert!(commit.is_err(), "the deferred violation surfaces at COMMIT");

    let count = |store: &Store, v: &str| {
        store
            .nodes_with_label("User")
            .iter()
            .filter(|&&nd| matches!(store.prop(nd, "email"), Value::Str(e) if &*e == v))
            .count()
    };
    assert_eq!(count(&store, "a@x"), 0, "whole tx rolled back");
    assert_eq!(count(&store, "b@x"), 0, "whole tx rolled back");
    assert_eq!(count(&store, "taken@x"), 1, "only the seed remains");
}

#[test]
fn a_deferred_constraint_completed_by_a_later_statement_commits() {
    // The point of deferral: a row that is temporarily invalid mid-transaction (no
    // required key) becomes valid before COMMIT (a later statement fills it), so the
    // transaction commits — the pattern immediate checking would reject.
    let mut store = Builder::default().build();
    store.create_required_constraint("Acct", "email").unwrap();
    stmt(&mut store, "START TRANSACTION").unwrap();
    // No email yet — deferred, so this SUCCEEDS (immediate checking would reject it).
    stmt(&mut store, "INSERT (:Acct {id: 'u'})")
        .expect("a deferred required constraint does not fault at the statement");
    stmt(
        &mut store,
        "MATCH (n:Acct {id: 'u'}) SET n.email = 'u@x.io'",
    )
    .unwrap();
    stmt(&mut store, "COMMIT").expect("valid by commit time");
    assert_eq!(store.live_node_count(), 1, "the completed row committed");
}

#[test]
fn a_required_violation_in_a_transaction_never_persists() {
    // A required constraint on Acct.email. A row that never fills it must NOT
    // persist — whether the engine rejects it at the statement (its per-statement
    // constraint check) or would defer to COMMIT, the invalid row leaves no trace
    // and the transaction ends cleanly. (Engine checks per-statement; the TS engine defers
    // to commit — a separate constraint-deferral divergence — but both are safe.)
    let mut store = Builder::default().build();
    store.create_required_constraint("Acct", "email").unwrap();
    stmt(&mut store, "START TRANSACTION").unwrap();
    let insert = stmt(&mut store, "INSERT (:Acct {bal: 1})"); // no email
    let commit = stmt(&mut store, "COMMIT");
    assert!(
        insert.is_err() || commit.is_err(),
        "a required violation must surface (at the statement or at commit)"
    );
    assert_eq!(store.live_node_count(), 0, "the invalid row left no trace");
    assert!(
        !store.in_transaction(),
        "the transaction is closed either way"
    );
}

// ---- row-driven INSERT (`FOR … IN <list> INSERT (…)`) --------------------

/// Count live nodes carrying label `l`.
fn label_count(store: &Store, l: &str) -> usize {
    store.nodes_with_label(l).len()
}

#[test]
fn for_insert_parses_to_insert_from() {
    match crate::gql::parse("FOR x IN [1, 2] INSERT (:N {v: x})").unwrap() {
        Plan::InsertFrom { nodes, edges, .. } => {
            assert_eq!(nodes.len(), 1);
            assert_eq!(nodes[0].labels, vec!["N".to_string()]);
            assert_eq!(nodes[0].props.len(), 1);
            assert_eq!(nodes[0].props[0].0, "v");
            assert!(edges.is_empty());
        }
        other => panic!("expected InsertFrom, got {other:?}"),
    }
}

#[test]
fn for_insert_creates_one_node_per_element_with_the_bound_variable() {
    let mut store = Builder::default().build();
    stmt(&mut store, "FOR x IN [1, 2, 3] INSERT (:Acct {bal: x})").unwrap();
    assert_eq!(label_count(&store, "Acct"), 3);
    // Read the values back — one per unwound element.
    let rows = run(
        &crate::gql::parse("MATCH (n:Acct) RETURN n.bal AS bal").unwrap(),
        &store,
    );
    let mut vals: Vec<f64> = rows.rows.iter().map(|r| num(&r[0])).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn for_insert_evaluates_property_expressions_per_row() {
    let mut store = Builder::default().build();
    // `b: x * 2` is an EXPRESSION over the unwound `x`, not a literal.
    stmt(
        &mut store,
        "FOR x IN [10, 20] INSERT (:Pair {a: x, b: x * 2})",
    )
    .unwrap();
    let rows = run(
        &crate::gql::parse("MATCH (n:Pair) RETURN n.a AS a, n.b AS b").unwrap(),
        &store,
    );
    let mut pairs: Vec<(f64, f64)> = rows.rows.iter().map(|r| (num(&r[0]), num(&r[1]))).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert_eq!(pairs, vec![(10.0, 20.0), (20.0, 40.0)]);
}

#[test]
fn for_insert_mixes_literal_and_expression_properties() {
    let mut store = Builder::default().build();
    stmt(
        &mut store,
        "FOR x IN [1, 2] INSERT (:Acct {kind: 'k', bal: x})",
    )
    .unwrap();
    let rows = run(
        &crate::gql::parse("MATCH (n:Acct) RETURN n.kind AS kind, n.bal AS bal").unwrap(),
        &store,
    );
    assert_eq!(rows.rows.len(), 2);
    for r in rows.rows.iter() {
        // the literal `kind` is the same string on every row
        assert!(
            matches!(&r[0], Value::Str(s) if &**s == "k"),
            "kind should be 'k', got {:?}",
            r[0]
        );
    }
}

#[test]
fn for_insert_creates_an_edge_per_row() {
    let mut store = Builder::default().build();
    stmt(
        &mut store,
        "FOR x IN [1, 2] INSERT (:A {v: x})-[:R]->(:B {v: x})",
    )
    .unwrap();
    assert_eq!(label_count(&store, "A"), 2);
    assert_eq!(label_count(&store, "B"), 2);
    // Two R edges, one per row (A_x → B_x).
    let n = run(
        &crate::gql::parse("MATCH (:A)-[:R]->(:B) RETURN count(*) AS c").unwrap(),
        &store,
    );
    assert_eq!(num(&n.rows[0][0]), 2.0);
}

#[test]
fn for_insert_over_an_empty_list_creates_nothing() {
    let mut store = Builder::default().build();
    stmt(&mut store, "FOR x IN [] INSERT (:Acct {bal: x})").unwrap();
    assert_eq!(label_count(&store, "Acct"), 0);
}

#[test]
fn for_insert_is_atomic_a_unique_violation_rolls_back_every_row() {
    // Both rows carry id='dup'; a unique constraint on (Acct, id) means the second
    // collides. Per-statement atomicity must roll the FIRST row back too — zero rows.
    let mut store = Builder::default().build();
    store.create_unique_constraint("Acct", &["id"]).unwrap();
    let err = stmt(
        &mut store,
        "FOR x IN [1, 2] INSERT (:Acct {id: 'dup', bal: x})",
    )
    .unwrap_err();
    assert!(
        err.starts_with("E_UNIQUE:") || err.starts_with("E_CONSTRAINT"),
        "duplicate unique value violates: {err}"
    );
    assert_eq!(
        label_count(&store, "Acct"),
        0,
        "the whole FOR-INSERT rolled back — no partial write"
    );
}

#[test]
fn for_insert_inside_a_transaction_commits_and_rolls_back_as_a_unit() {
    // Committed: the FOR-INSERT's rows persist with the transaction.
    let mut store = Builder::default().build();
    stmt(&mut store, "START TRANSACTION").unwrap();
    stmt(&mut store, "FOR x IN [1, 2, 3] INSERT (:Acct {bal: x})").unwrap();
    stmt(&mut store, "COMMIT").unwrap();
    assert_eq!(label_count(&store, "Acct"), 3);

    // Rolled back: START, FOR-INSERT, ROLLBACK → nothing persists.
    let mut store2 = Builder::default().build();
    stmt(&mut store2, "START TRANSACTION").unwrap();
    stmt(&mut store2, "FOR x IN [1, 2] INSERT (:Acct {bal: x})").unwrap();
    stmt(&mut store2, "ROLLBACK").unwrap();
    assert_eq!(label_count(&store2, "Acct"), 0);
}

#[test]
fn for_insert_is_rejected_in_a_read_only_transaction() {
    let mut store = Builder::default().build();
    stmt(&mut store, "START TRANSACTION READ ONLY").unwrap();
    let err = stmt(&mut store, "FOR x IN [1, 2] INSERT (:Acct {bal: x})").unwrap_err();
    assert!(
        err.starts_with("E_INVALID_GRAPH_OP:"),
        "read-only rejects: {err}"
    );
    assert_eq!(label_count(&store, "Acct"), 0);
}

#[test]
fn insert_string_id_is_the_unique_external_identity() {
    let mut store = Builder::default().build();
    // A string `id` sets the external identity AND stays a queryable property.
    stmt(&mut store, "INSERT (:Acct {id: 'x', bal: 5})").unwrap();
    let rows = run(
        &crate::gql::parse("MATCH (n:Acct) RETURN n.id AS id, n.bal AS bal").unwrap(),
        &store,
    );
    assert!(
        matches!(&rows.rows[0][0], Value::Str(s) if &**s == "x"),
        "n.id stays a stored property"
    );
    assert_eq!(num(&rows.rows[0][1]), 5.0);
    assert!(
        store.node_by_ext("x").is_some(),
        "external id is registered"
    );

    // A duplicate string id is a constraint violation; the graph is unchanged.
    let err = stmt(&mut store, "INSERT (:Acct {id: 'x'})").unwrap_err();
    assert!(err.starts_with("E_UNIQUE:"), "duplicate string id: {err}");
    assert_eq!(label_count(&store, "Acct"), 1);

    // A NUMERIC id is a plain property — no identity, no uniqueness (two coexist).
    stmt(&mut store, "INSERT (:Num {id: 7})").unwrap();
    stmt(&mut store, "INSERT (:Num {id: 7})").unwrap();
    assert_eq!(label_count(&store, "Num"), 2);

    // No id → a synthetic external id; two such nodes coexist.
    stmt(&mut store, "INSERT (:Plain {bal: 1})").unwrap();
    stmt(&mut store, "INSERT (:Plain {bal: 2})").unwrap();
    assert_eq!(label_count(&store, "Plain"), 2);
}

#[test]
fn a_string_id_collision_within_one_insert_rolls_the_whole_statement_back() {
    let mut store = Builder::default().build();
    // Two nodes in ONE INSERT sharing a new string id → the second collides with
    // the first; per-statement atomicity leaves neither.
    let err = stmt(&mut store, "INSERT (:A {id: 'k'}), (:B {id: 'k'})").unwrap_err();
    assert!(
        err.starts_with("E_UNIQUE:"),
        "intra-statement dup id: {err}"
    );
    assert_eq!(store.live_node_count(), 0, "the whole INSERT rolled back");
}

#[test]
fn edge_string_id_is_the_unique_external_identity() {
    let mut store = Builder::default().build();
    stmt(
        &mut store,
        "INSERT (:A {id: 'a'})-[:R {id: 'e1'}]->(:B {id: 'b'})",
    )
    .unwrap();
    // A duplicate EDGE id is a constraint violation; the statement rolls back.
    let err = stmt(
        &mut store,
        "INSERT (:A {id: 'a2'})-[:R {id: 'e1'}]->(:B {id: 'b2'})",
    )
    .unwrap_err();
    assert!(err.starts_with("E_UNIQUE:"), "duplicate edge id: {err}");
    let n = run(
        &crate::gql::parse("MATCH ()-[:R]->() RETURN count(*) AS c").unwrap(),
        &store,
    );
    assert_eq!(num(&n.rows[0][0]), 1.0, "only the first R edge exists");
}

#[test]
fn set_on_a_string_identity_id_is_rejected_but_other_and_numeric_ids_are_settable() {
    let mut store = Builder::default().build();
    stmt(&mut store, "INSERT (:A {id: 'a', bal: 1})").unwrap();
    // SET on a string identity `id` → immutable error; the id is unchanged.
    let err = stmt(&mut store, "MATCH (n:A {id: 'a'}) SET n.id = 'z'").unwrap_err();
    assert!(
        err.starts_with("E_INVALID_GRAPH_OP:"),
        "SET id immutable: {err}"
    );
    assert!(store.node_by_ext("a").is_some(), "the id is unchanged");
    assert!(store.node_by_ext("z").is_none());
    // A NON-id property is still SET-able on an identity node.
    stmt(&mut store, "MATCH (n:A {id: 'a'}) SET n.bal = 5").unwrap();
    // A NUMERIC id is a plain property (not an identity) → SET-able.
    stmt(&mut store, "INSERT (:N {id: 7})").unwrap();
    stmt(&mut store, "MATCH (n:N) SET n.id = 8").unwrap();
}

#[test]
fn set_on_a_string_identity_edge_id_is_rejected() {
    let mut store = Builder::default().build();
    stmt(&mut store, "INSERT (:A)-[:R {id: 'e1', w: 1}]->(:B)").unwrap();
    let err = stmt(&mut store, "MATCH ()-[r:R]->() SET r.id = 'e2'").unwrap_err();
    assert!(
        err.starts_with("E_INVALID_GRAPH_OP:"),
        "SET edge id immutable: {err}"
    );
    // A non-id edge property is still SET-able.
    stmt(&mut store, "MATCH ()-[r:R]->() SET r.w = 9").unwrap();
}

/// `REMOVE` of a required-constraint key is rejected and rolled back.
#[test]
fn remove_enforces_required_constraint() {
    let mut b = Builder::default();
    b.node(&["User"], &[("name", s("alice"))]);
    let mut store = b.build();
    store.create_required_constraint("User", "name").unwrap();
    let id = store.nodes_with_label("User")[0];
    assert!(execute(
        &crate::gql::parse("MATCH (u:User) REMOVE u.name").unwrap(),
        &mut store
    )
    .is_err());
    assert!(
        store.has_prop(id, "name"),
        "required key must survive rollback"
    );
}

/// GQL DELETE / DETACH DELETE, matching the TS engine: a non-DETACH delete of a node with
/// relationships errors and rolls back; DETACH cascades the edges; an edge delete
/// leaves the endpoints; a node with no edges deletes plainly.
#[test]
fn gql_delete_and_detach_delete() {
    let build = || {
        let mut b = Builder::default();
        let a = b.node(&["P"], &[("n", s("a"))]);
        let z = b.node(&["P"], &[("n", s("b"))]);
        let iso = b.node(&["P"], &[("n", s("iso"))]);
        b.edge(a, z, "R");
        let _ = iso;
        b.build()
    };
    let go = |q: &str, store: &mut Store| execute(&crate::gql::parse(q).unwrap(), store);

    // Non-DETACH delete of a node WITH an edge → error, nothing removed.
    let mut s1 = build();
    assert!(go("MATCH (p:P) WHERE p.n='a' DELETE p", &mut s1).is_err());
    assert_eq!(s1.live_node_count(), 3, "rolled back");

    // DETACH DELETE removes the node and its edge; the neighbour survives.
    let mut s2 = build();
    assert!(go("MATCH (p:P) WHERE p.n='a' DETACH DELETE p", &mut s2).is_ok());
    assert_eq!(s2.live_node_count(), 2);

    // A node with NO edges deletes plainly (no DETACH needed).
    let mut s3 = build();
    assert!(go("MATCH (p:P) WHERE p.n='iso' DELETE p", &mut s3).is_ok());
    assert_eq!(s3.live_node_count(), 2);

    // Deleting the EDGE leaves both endpoints; then a plain DELETE works.
    let mut s4 = build();
    assert!(go("MATCH (a:P)-[r:R]->(b) DELETE r", &mut s4).is_ok());
    assert_eq!(s4.live_node_count(), 3);
    assert!(go("MATCH (p:P) WHERE p.n='a' DELETE p", &mut s4).is_ok());
    assert_eq!(s4.live_node_count(), 2);
}

/// A deleted node is absent from a label scan through the query path — build
/// the social graph, delete bob (id 1), and the Person scan yields alice+carol.
#[test]
fn scan_skips_deleted_node() {
    let mut store = social();
    store.delete_node(1); // bob
    let out = run(
        &scan("Person").project(vec![("name".into(), prop(0, "name"))]),
        &store,
    );
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["alice", "carol"]);
}

// --- Arithmetic (E1) ---

fn arith(op: crate::ir::ArithOp, l: Expr, r: Expr) -> Expr {
    Expr::Arith {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}

/// `age * 2 + 1` for alice(30) = 61 — precedence honored in the hand plan.
#[test]
fn arith_eval_computes() {
    use crate::ir::ArithOp::{Add, Mul};
    let store = social();
    let plan = scan("Person")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))))
        .project(vec![(
            "x".into(),
            arith(Add, arith(Mul, prop(0, "age"), lit(n(2.0))), lit(n(1.0))),
        )]);
    assert_eq!(num(&run(&plan, &store).rows[0][0]), 61.0);
}

/// A NULL / missing / non-numeric operand yields NULL — the Project node has no
/// `age`, so `age + 1` is NULL for exactly it.
#[test]
fn arith_null_propagates() {
    use crate::ir::ArithOp::Add;
    let store = social();
    let plan = Plan::Scan { label: None }
        .project(vec![("x".into(), arith(Add, prop(0, "age"), lit(n(1.0))))]);
    let nulls = run(&plan, &store)
        .rows
        .iter()
        .filter(|r| r[0].is_null())
        .count();
    assert_eq!(nulls, 1); // only the Project node lacks age
}

/// Arithmetic follows the TS engine's SQL rule: a NULL operand yields NULL, but a non-null
/// NON-numeric operand (string/bool) is a DATA EXCEPTION (never coerced) — an
/// explicit CAST is the escape hatch. Aggregates sum()/avg() likewise throw over a
/// non-numeric value.
#[test]
fn arith_and_agg_throw_on_non_numeric() {
    let store = social();
    let ok = |q: &str| {
        let plan = crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store);
        try_run(&plan, &store)
    };
    // null operand → NULL (not an error).
    assert!(ok("RETURN 1 + null AS r").unwrap().rows[0][0].is_null());
    // non-null non-numeric → error.
    assert!(ok("RETURN 'abc' + 1 AS r").is_err());
    assert!(ok("RETURN true * 2 AS r").is_err());
    assert!(ok("MATCH (p:Person) RETURN p.name + 1 AS r").is_err());
    // CAST is the escape hatch.
    assert!(matches!(
        ok("RETURN CAST('2' AS INT) * 3 AS r").unwrap().rows[0][0],
        Value::Num(x) if x == 6.0
    ));
    // sum/avg over numbers still work; over a non-numeric they throw.
    assert!(ok("MATCH (p:Person) RETURN sum(p.age) AS r").is_ok());
    assert!(ok("MATCH (p:Person) RETURN sum(p.name) AS r").is_err());
    assert!(ok("MATCH (p:Person) RETURN avg(p.name) AS r").is_err());
}

/// Division / modulo by zero THROWS (matches the TS engine's DataException), via
/// the fallible read path — `try_run` surfaces the error (K3).
#[test]
fn arith_div_or_mod_by_zero_throws() {
    use crate::ir::ArithOp::{Div, Rem};
    let store = social();
    let one = scan("Person").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))));
    for op in [Div, Rem] {
        let plan = one
            .clone()
            .project(vec![("x".into(), arith(op, prop(0, "age"), lit(n(0.0))))]);
        let err = crate::exec::try_run(&plan, &store).unwrap_err();
        assert!(err.contains("division by zero"), "op {op:?}: {err}");
    }
}

/// A product that overflows f64 to +Inf is KEPT (IEEE), matching the TS engine —
/// NaN/Inf are coerced to null only at the JSON egress boundary, not here (K4).
#[test]
fn arith_overflow_keeps_inf() {
    use crate::ir::ArithOp::Mul;
    let store = social();
    let one = scan("Person").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))));
    let big = one.project(vec![("x".into(), arith(Mul, lit(n(1e308)), lit(n(1e308))))]);
    assert!(matches!(run(&big, &store).rows[0][0], Value::Num(x) if x.is_infinite() && x > 0.0));
}

// --- Property index + IndexSeek (D1a) ---

/// A store with two labels sharing an `age` property (some age 30).
fn indexed_store() -> Store {
    let mut st = Builder::default().build();
    st.add_node(&["P"], &[("age", n(30.0)), ("name", s("a"))]);
    st.add_node(&["P"], &[("age", n(25.0)), ("name", s("b"))]);
    st.add_node(&["P"], &[("age", n(30.0)), ("name", s("c"))]);
    st.add_node(&["Q"], &[("age", n(30.0)), ("name", s("d"))]); // other label
    st
}

/// `IndexSeek` returns the SAME rows as `Scan + Filter(=)`, with and without
/// an index. P nodes with age 30 are a and c (d is a Q, excluded).
#[test]
fn index_seek_matches_scan_filter() {
    let mut st = indexed_store();
    let seek = Plan::IndexSeek {
        label: "P".into(),
        key: "age".into(),
        value: n(30.0),
    }
    .project(vec![("name".into(), prop(0, "name"))]);
    let filt = scan("P")
        .filter(cmp(CompareOp::Eq, prop(0, "age"), lit(n(30.0))))
        .project(vec![("name".into(), prop(0, "name"))]);

    let mut want = names_of(&run(&filt, &st), 0);
    want.sort();
    assert_eq!(want, vec!["a", "c"]);
    let mut got = names_of(&run(&seek, &st), 0);
    got.sort();
    assert_eq!(got, want); // no index yet (scan fallback)

    st.create_index("age");
    let mut got = names_of(&run(&seek, &st), 0);
    got.sort();
    assert_eq!(got, want); // index path, same rows
}

/// The index is maintained through set/remove/delete.
#[test]
fn index_maintained_on_writes() {
    let mut st = indexed_store();
    st.create_index("age");
    let sorted = |st: &Store| {
        let mut v = st.index_lookup("age", &n(30.0)).unwrap();
        v.sort_unstable();
        v
    };
    assert_eq!(sorted(&st), vec![0, 2, 3]); // any-label candidates
    st.set_prop(0, "age", n(25.0)); // 0 leaves the 30 bucket
    assert_eq!(sorted(&st), vec![2, 3]);
    st.delete_node(2); // 2 gone
    assert_eq!(sorted(&st), vec![3]);
    st.remove_prop(3, "age"); // 3 loses the prop
    assert!(st.index_lookup("age", &n(30.0)).unwrap().is_empty());
}

/// A transaction rollback restores the index (writes replay through the
/// primitives, which maintain it).
#[test]
fn index_consistent_after_rollback() {
    let mut st = indexed_store();
    st.create_index("age");
    st.begin();
    st.set_prop(0, "age", n(99.0));
    st.delete_node(2);
    st.rollback();
    let mut v = st.index_lookup("age", &n(30.0)).unwrap();
    v.sort_unstable();
    assert_eq!(v, vec![0, 2, 3]);
}

/// A NaN / NULL seek value matches nothing (predicate `=` semantics), same as
/// the filter — even though those values live in a group_key bucket.
#[test]
fn index_seek_nan_and_null_match_nothing() {
    let mut st = indexed_store();
    st.create_index("age");
    let seek = |v: Value| {
        Plan::IndexSeek {
            label: "P".into(),
            key: "age".into(),
            value: v,
        }
        .project(vec![("name".into(), prop(0, "name"))])
    };
    assert_eq!(run(&seek(n(f64::NAN)), &st).rows.len(), 0);
    assert_eq!(run(&seek(Value::Null), &st).rows.len(), 0);
}

/// `RangeSeek` returns the SAME rows as `Scan + Filter(<op>)` for every range
/// op, with and without a range index. Hand: ages 30,25,40 (a,b,c).
#[test]
fn range_seek_matches_scan_filter_all_ops() {
    let mut st = indexed_store(); // P: a=30, b=25, c=30; Q: d=30
    let ops = [
        (CompareOp::Gt, 25.0, vec!["a", "c"]), // >25 → 30,30
        (CompareOp::Ge, 30.0, vec!["a", "c"]), // >=30
        (CompareOp::Lt, 30.0, vec!["b"]),      // <30 → 25
        (CompareOp::Le, 25.0, vec!["b"]),      // <=25
    ];
    for indexed in [false, true] {
        if indexed {
            st.create_range_index("age");
        }
        for (op, v, want) in &ops {
            let seek = Plan::RangeSeek {
                label: "P".into(),
                key: "age".into(),
                op: *op,
                value: n(*v),
            }
            .project(vec![("name".into(), prop(0, "name"))]);
            let filt = scan("P")
                .filter(cmp(*op, prop(0, "age"), lit(n(*v))))
                .project(vec![("name".into(), prop(0, "name"))]);
            let mut a = names_of(&run(&seek, &st), 0);
            a.sort();
            let mut b = names_of(&run(&filt, &st), 0);
            b.sort();
            assert_eq!(a, *want, "op {op:?} v {v}");
            assert_eq!(a, b, "seek vs filter disagree for {op:?} {v}");
        }
    }
}

/// An indexed range seek returns EXACTLY the scan-filter rows (the
/// equivalent-spellings invariant): a NULL value matches nothing, and a
/// cross-type comparison is UNKNOWN → dropped (a string property vs a numeric
/// bound does NOT match, per the 3VL operator semantics — K2).
#[test]
fn range_seek_null_and_cross_type_match_filter() {
    let mut st = Builder::default().build();
    st.add_node(&["P"], &[("v", n(10.0))]);
    st.add_node(&["P"], &[("v", s("zzz"))]); // string: cross-type vs a number
    st.add_node(&["P"], &[]); // v absent → null
    st.create_range_index("v");
    let check = |st: &Store, op, val: Value| {
        let seek = Plan::RangeSeek {
            label: "P".into(),
            key: "v".into(),
            op,
            value: val.clone(),
        };
        let filt = scan("P").filter(cmp(op, prop(0, "v"), lit(val)));
        (run(&seek, st).rows.len(), run(&filt, st).rows.len())
    };
    // v > 5: only 10 matches; "zzz" is cross-type → UNKNOWN → dropped; null
    // excluded → 1, and seek agrees with filter.
    assert_eq!(check(&st, CompareOp::Gt, n(5.0)), (1, 1));
    // v > null: UNKNOWN for all → 0, agree.
    assert_eq!(check(&st, CompareOp::Gt, Value::Null), (0, 0));
}

/// The range index is maintained through set/delete and a transaction rollback.
#[test]
fn range_index_maintained_and_rolls_back() {
    let mut st = indexed_store();
    st.create_range_index("age");
    // Candidates > 25 across any label (index_lookup is any-label).
    let cand = |st: &Store, v: f64| st.range_lookup("age", CompareOp::Gt, &n(v)).unwrap().len();
    assert_eq!(cand(&st, 25.0), 3); // a,c (P,30) + d (Q,30)
    st.set_prop(0, "age", n(10.0)); // a drops below 25
    assert_eq!(cand(&st, 25.0), 2);
    st.begin();
    st.delete_node(2); // c gone
    assert_eq!(cand(&st, 25.0), 1);
    st.rollback();
    assert_eq!(cand(&st, 25.0), 2); // restored
}

/// A scalar count over an IndexSeek is correct (the seek seeds like a scan).
#[test]
fn count_over_index_seek() {
    let mut st = indexed_store();
    st.create_index("age");
    let plan = Plan::IndexSeek {
        label: "P".into(),
        key: "age".into(),
        value: n(30.0),
    }
    .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
    assert_eq!(num(&run(&plan, &st).rows[0][0]), 2.0);
}

/// Reversed operand order (`literal < prop`) must match `prop > literal` —
/// exercises the fused filter's operand flip. `28 < age` → alice(30),carol(40).
#[test]
fn filter_literal_on_left_flips() {
    let store = social();
    let plan = scan("Person")
        .filter(cmp(CompareOp::Lt, lit(n(28.0)), prop(0, "age")))
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["alice", "carol"]);
}

// --- Expand ---

/// `MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name` — two slots bound,
/// row per matching edge.
#[test]
fn expand_binds_both_ends() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .project(vec![
            ("a".into(), prop(0, "name")),
            ("b".into(), prop(1, "name")),
        ]);
    let out = run(&plan, &store);
    let mut pairs: Vec<(String, String)> = out
        .rows
        .iter()
        .map(|r| (as_str(&r[0]), as_str(&r[1])))
        .collect();
    pairs.sort();
    // a→b, a→c, b→c (KNOWS only; the WORKS_ON edge is excluded)
    assert_eq!(
        pairs,
        vec![
            ("alice".into(), "bob".into()),
            ("alice".into(), "carol".into()),
            ("bob".into(), "carol".into()),
        ]
    );
}

/// An edge-label filter selects: WORKS_ON reaches only the Project.
#[test]
fn expand_filters_by_edge_label() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["WORKS_ON".to_string()])
        .project(vec![("t".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    assert_eq!(names_of(&out, 0), vec!["graphdb"]);
}

/// Filtering on the FAR end after an expand — the far slot's property.
#[test]
fn filter_on_the_expanded_end() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .filter(cmp(CompareOp::Ge, prop(1, "age"), lit(n(40.0))))
        .project(vec![("a".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    // Only edges landing on carol(40): alice→carol, bob→carol.
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["alice", "bob"]);
}

/// Incoming direction: who KNOWS carol.
#[test]
fn expand_incoming() {
    let store = social();
    let plan = scan("Person")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("carol"))))
        .expand(0, Dir::In, &["KNOWS".to_string()])
        .project(vec![("who".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["alice", "bob"]);
}

/// An unknown edge label matches nothing.
#[test]
fn expand_unknown_label_is_empty() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["NOPE".to_string()])
        .project(vec![("x".into(), prop(1, "name"))]);
    assert_eq!(run(&plan, &store).rows.len(), 0);
}

/// `expand_edge` binds the traversed edge as a slot: for `(a)-[r:R]->(b)` the
/// edge is slot 1 and the node slot 2, so `r.weight` reads an edge property and
/// `b.name` reads a node property.
#[test]
fn expand_edge_binds_edge_and_reads_edge_prop() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("name", s("a"))]);
    let b = st.add_node(&["P"], &[("name", s("b"))]);
    st.add_edge(a, b, "R");
    let eid = st.out(a)[0].eid;
    st.set_edge_prop(eid, "weight", n(0.5));
    let plan = scan("P")
        .expand_edge(0, Dir::Out, &["R".to_string()])
        .project(vec![
            ("w".into(), prop(1, "weight")), // edge slot
            ("b".into(), prop(2, "name")),   // node slot
        ]);
    let out = run(&plan, &st);
    assert_eq!(out.rows.len(), 1);
    assert!(matches!(&out.rows[0][0], Value::Num(x) if *x == 0.5));
    assert_eq!(as_str(&out.rows[0][1]), "b");
}

/// An edge slot with no such property reads NULL.
#[test]
fn expand_edge_absent_prop_is_null() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[]);
    let b = st.add_node(&["P"], &[]);
    st.add_edge(a, b, "R");
    let plan = scan("P")
        .expand_edge(0, Dir::Out, &["R".to_string()])
        .project(vec![("w".into(), prop(1, "weight"))]);
    let out = run(&plan, &st);
    assert_eq!(out.rows.len(), 1);
    assert!(out.rows[0][0].is_null());
}

/// Filtering on an edge property keeps only matching edges. a→b (w=0.5),
/// a→c (w=0.2); `WHERE r.w > 0.4` → only b.
#[test]
fn filter_on_edge_property() {
    let mut st = Builder::default().build();
    let a = st.add_node(&["P"], &[("name", s("a"))]);
    let b = st.add_node(&["P"], &[("name", s("b"))]);
    let c = st.add_node(&["P"], &[("name", s("c"))]);
    st.add_edge(a, b, "R");
    let e1 = st.out(a)[0].eid;
    st.add_edge(a, c, "R");
    let e2 = st.out(a)[1].eid;
    st.set_edge_prop(e1, "w", n(0.5));
    st.set_edge_prop(e2, "w", n(0.2));
    let plan = scan("P")
        .expand_edge(0, Dir::Out, &["R".to_string()])
        .filter(cmp(CompareOp::Gt, prop(1, "w"), lit(n(0.4))))
        .project(vec![("b".into(), prop(2, "name"))]);
    assert_eq!(names_of(&run(&plan, &st), 0), vec!["b"]);
}

fn as_str(v: &Value) -> String {
    match v {
        Value::Str(x) => x.to_string(),
        other => format!("{other:?}"),
    }
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Num(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

// --- Aggregate / group-by ---

fn agg(func: AggFn, arg: Option<Expr>, distinct: bool, name: &str) -> Agg {
    Agg {
        func,
        arg,
        distinct,
        name: name.to_string(),
        frac: None,
        null_on_empty: false,
        numeric_only: false,
    }
}

/// Scalar `count(*)` over a label — one row, the count.
#[test]
fn scalar_count_star() {
    let store = social();
    let plan = scan("Person").aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
    let out = run(&plan, &store);
    assert_eq!(out.names, vec!["c"]);
    assert_eq!(out.rows.len(), 1);
    assert_eq!(num(&out.rows[0][0]), 3.0); // alice, bob, carol
}

/// sum / min / max / avg over the age column, hand-computed: 30,25,40.
#[test]
fn scalar_sum_min_max_avg() {
    let store = social();
    let plan = scan("Person").aggregate(
        vec![],
        vec![
            agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
            agg(AggFn::Min, Some(prop(0, "age")), false, "lo"),
            agg(AggFn::Max, Some(prop(0, "age")), false, "hi"),
            agg(AggFn::Avg, Some(prop(0, "age")), false, "av"),
        ],
    );
    let out = run(&plan, &store);
    let r = &out.rows[0];
    assert_eq!(num(&r[0]), 95.0); // 30+25+40
    assert_eq!(num(&r[1]), 25.0);
    assert_eq!(num(&r[2]), 40.0);
    assert_eq!(num(&r[3]), 95.0 / 3.0);
}

/// `count(*)` grouped by a property — a row per distinct value, first-seen
/// order. Group on `city`: alice/carol="nyc", bob="sf".
#[test]
fn group_count_by_property() {
    let mut b = Builder::default();
    b.node(&["P"], &[("city", s("nyc"))]);
    b.node(&["P"], &[("city", s("sf"))]);
    b.node(&["P"], &[("city", s("nyc"))]);
    let store = b.build();
    let plan = scan("P").aggregate(
        vec![("city".into(), prop(0, "city"))],
        vec![agg(AggFn::Count, None, false, "c")],
    );
    let out = run(&plan, &store);
    assert_eq!(out.names, vec!["city", "c"]);
    // first-seen order: nyc (row 0), then sf (row 1).
    assert_eq!(as_str(&out.rows[0][0]), "nyc");
    assert_eq!(num(&out.rows[0][1]), 2.0);
    assert_eq!(as_str(&out.rows[1][0]), "sf");
    assert_eq!(num(&out.rows[1][1]), 1.0);
}

/// `count(arg)` ignores nulls; `count(DISTINCT arg)` ignores nulls AND
/// duplicates. Ages: 10, 10, null, 20 → count=3, distinct=2.
#[test]
fn count_arg_and_count_distinct_skip_nulls() {
    let mut b = Builder::default();
    b.node(&["P"], &[("v", n(10.0))]);
    b.node(&["P"], &[("v", n(10.0))]);
    b.node(&["P"], &[]); // no v → null
    b.node(&["P"], &[("v", n(20.0))]);
    let store = b.build();
    let plan = scan("P").aggregate(
        vec![],
        vec![
            agg(AggFn::Count, Some(prop(0, "v")), false, "c"),
            agg(AggFn::Count, Some(prop(0, "v")), true, "cd"),
        ],
    );
    let out = run(&plan, &store);
    assert_eq!(num(&out.rows[0][0]), 3.0); // non-null count
    assert_eq!(num(&out.rows[0][1]), 2.0); // distinct non-null: {10, 20}
}

/// Over nothing, `count` and `sum` are both 0 but `avg` is NULL — matching
/// the TS engine (the GQL/Cypher convention; the differential fuzzer flagged the
/// earlier SQL-style `sum → NULL`).
#[test]
fn sum_over_empty_is_zero_avg_is_null() {
    let store = social();
    // No node has this label → empty input to the scalar aggregate.
    let plan = Plan::Scan {
        label: Some("Nonexistent".into()),
    }
    .aggregate(
        vec![],
        vec![
            agg(AggFn::Count, None, false, "c"),
            agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
            agg(AggFn::Avg, Some(prop(0, "age")), false, "a"),
        ],
    );
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 1); // scalar aggregate still emits one row
    assert_eq!(num(&out.rows[0][0]), 0.0); // count(*) = 0
    assert_eq!(num(&out.rows[0][1]), 0.0); // sum = 0
    assert!(out.rows[0][2].is_null()); // avg = NULL
}

/// SUM folds a group of DURATIONs component-wise (ISO-8601 addition is total — months
/// and days add independently, no normalization). AVG over durations (fractional months
/// are ill-defined), a mixed number+duration group, and a non-DURATION temporal fault.
#[test]
fn sum_over_durations_folds_component_wise() {
    let nd = "{\"id\":\"1\",\"labels\":[\"T\"],\"props\":{\"d\":{\"@duration\":\"P1M\"}}}\n\
                  {\"id\":\"2\",\"labels\":[\"T\"],\"props\":{\"d\":{\"@duration\":\"P2M\"}}}\n\
                  {\"id\":\"3\",\"labels\":[\"T\"],\"props\":{\"d\":{\"@duration\":\"P1D\"}}}";
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ok = |q: &str| {
        try_run(
            &crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store),
            &store,
        )
    };
    // P1M + P2M + P1D = P3M1D — months (3) and days (1) accumulate independently.
    match &ok("MATCH (n:T) RETURN sum(n.d) AS x").unwrap().rows[0][0] {
        Value::Temporal(crate::temporal::Temporal::Duration(d)) => {
            assert_eq!((d.months, d.days, d.secs), (3, 1, 0));
        }
        other => panic!("want a duration, got {other:?}"),
    }
    // AVG over durations faults; a non-DURATION temporal in SUM faults.
    assert!(ok("MATCH (n:T) RETURN avg(n.d) AS x").is_err());
    assert!(ok("MATCH (n:T) RETURN sum(date('2020-01-01')) AS x").is_err());
}

/// A grouped RETURN preserves the ITEM column order even when an aggregate precedes a
/// group key (`count(*) AS c, n.k AS g` → the count column is FIRST). The plan's schema
/// is `[keys…, aggs…]`, so this relies on the final reorder-to-visible projection —
/// which previously ran only when there were hidden group keys to drop.
#[test]
fn grouped_return_preserves_aggregate_first_column_order() {
    let nd = "{\"id\":\"1\",\"labels\":[\"T\"],\"props\":{\"n\":3}}\n\
                  {\"id\":\"2\",\"labels\":[\"T\"],\"props\":{\"n\":3}}\n\
                  {\"id\":\"3\",\"labels\":[\"T\"],\"props\":{\"n\":5}}";
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let rows = try_run(
        &crate::opt::optimize_indexed(
            crate::gql::parse("MATCH (n:T) RETURN count(*) AS c, n.n AS g GROUP BY n.n ORDER BY g")
                .unwrap(),
            &store,
        ),
        &store,
    )
    .unwrap()
    .rows;
    // Columns are [c, g] in item order — group n=3 has count 2, n=5 has count 1.
    assert_eq!((num(&rows[0][0]), num(&rows[0][1])), (2.0, 3.0));
    assert_eq!((num(&rows[1][0]), num(&rows[1][1])), (1.0, 5.0));
}

/// CASE is LAZY (SQL-standard): a type error in a branch a row never takes does NOT
/// fire. The eager fast path evaluates all branches vectorized, but on ANY error retries
/// with masked evaluation (each branch only over the rows that reach it). A branch that
/// a row genuinely takes, or a non-boolean WHEN reached before a match, still faults.
#[test]
fn case_is_lazy_over_unreached_branches() {
    let store = social();
    let ok = |q: &str| {
        try_run(
            &crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store),
            &store,
        )
    };
    // The ELSE holds an arithmetic type error but is never taken (WHEN true) → no fault.
    assert!(matches!(
        ok("MATCH (p:Person) RETURN CASE WHEN true THEN 1 ELSE (2 + 'abc') END AS x LIMIT 1")
            .unwrap()
            .rows[0][0],
        Value::Num(x) if x == 1.0
    ));
    // A row that genuinely TAKES the ill-typed branch faults.
    assert!(ok("MATCH (p:Person) RETURN CASE WHEN true THEN (2 + 'abc') END AS x").is_err());
    // A non-boolean WHEN reached before any match faults (a WHEN must be boolean).
    assert!(ok("MATCH (p:Person) RETURN CASE WHEN 5 THEN 1 ELSE 2 END AS x").is_err());
}

/// A SKIP whose window reaches the end of the data still late-materializes: only the
/// surviving `[skip, end)` rows are projected, so a FALLIBLE projection never runs for a
/// PAGED-OUT row. Previously `end >= n` bailed to a full projection and faulted on the
/// skipped rows even when no surviving row was ill-typed.
#[test]
fn skip_window_at_end_does_not_project_paged_out_rows() {
    // Sorted by n: n=1 → s='abc' (a CAST-to-int fault), n=2 → s='5' (castable).
    let nd = "{\"id\":\"1\",\"labels\":[\"T\"],\"props\":{\"n\":1,\"s\":\"abc\"}}\n\
                  {\"id\":\"2\",\"labels\":[\"T\"],\"props\":{\"n\":2,\"s\":\"5\"}}";
    let store = crate::ndjson::from_ndjson(nd).unwrap();
    let ok = |q: &str| {
        try_run(
            &crate::opt::optimize_indexed(crate::gql::parse(q).unwrap(), &store),
            &store,
        )
    };
    // SKIP 1 LIMIT 1 → window [1, 2) is only the n=2 row (CAST('5')=5). The paged-out
    // n=1 row (CAST('abc') would fault) is never evaluated.
    let rows =
        ok("MATCH (n:T) RETURN CAST(n.s AS INTEGER) AS x, n.n AS t ORDER BY t SKIP 1 LIMIT 1")
            .unwrap()
            .rows;
    assert_eq!(num(&rows[0][0]), 5.0);
    // Sanity: WITHOUT the skip the n=1 row IS projected and the CAST faults.
    assert!(ok("MATCH (n:T) RETURN CAST(n.s AS INTEGER) AS x, n.n AS t ORDER BY t").is_err());
}

/// A grouped aggregate over empty input emits ZERO rows (unlike the scalar
/// case) — there are no groups.
#[test]
fn grouped_over_empty_is_zero_rows() {
    let store = social();
    let plan = Plan::Scan {
        label: Some("Nonexistent".into()),
    }
    .aggregate(
        vec![("k".into(), prop(0, "age"))],
        vec![agg(AggFn::Count, None, false, "c")],
    );
    assert_eq!(run(&plan, &store).rows.len(), 0);
}

/// Aggregate after an Expand: out-degree per person (count of KNOWS edges),
/// grouped by the source. alice→2, bob→1, carol→0(absent).
#[test]
fn count_out_degree_grouped_by_source() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .aggregate(
            vec![("who".into(), prop(0, "name"))],
            vec![agg(AggFn::Count, None, false, "deg")],
        );
    let out = run(&plan, &store);
    let mut got: Vec<(String, f64)> = out
        .rows
        .iter()
        .map(|r| (as_str(&r[0]), num(&r[1])))
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    // carol has no outgoing KNOWS, so she is absent from the expanded rows.
    assert_eq!(got, vec![("alice".into(), 2.0), ("bob".into(), 1.0)]);
}

/// Scalar `count(*)` over a single Expand — the frontier fast path. Hand
/// count of KNOWS edges: alice→{bob,carol}, bob→{carol} = 3.
#[test]
fn fused_count_star_one_hop() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 1);
    assert_eq!(num(&out.rows[0][0]), 3.0);
}

/// Scalar `count(*)` over a two-hop chain. Hand count of length-2 KNOWS
/// walks: only alice→bob→carol (bob is the only reached node with an
/// outgoing KNOWS) = 1.
#[test]
fn fused_count_star_two_hop() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .expand(1, Dir::Out, &["KNOWS".to_string()])
        .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
    let out = run(&plan, &store);
    assert_eq!(num(&out.rows[0][0]), 1.0);
}

/// 2-hop `count(*)` where an intermediate is reached by MULTIPLE paths — the
/// dedup-with-multiplicity path must scale by how many times it was reached.
/// a→x, b→x (x reached twice), x→p, x→q. Length-2 walks: a→x→{p,q} and
/// b→x→{p,q} = 4 (x itself reaches p,q which are sinks).
#[test]
fn fused_count_star_two_hop_with_multiplicity() {
    let mut bld = Builder::default();
    let a = bld.node(&["P"], &[]);
    let b = bld.node(&["P"], &[]);
    let x = bld.node(&["P"], &[]);
    let p = bld.node(&["P"], &[]);
    let q = bld.node(&["P"], &[]);
    bld.edge(a, x, "R");
    bld.edge(b, x, "R");
    bld.edge(x, p, "R");
    bld.edge(x, q, "R");
    let store = bld.build();
    let plan = scan("P")
        .expand(0, Dir::Out, &["R".to_string()])
        .expand(1, Dir::Out, &["R".to_string()])
        .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
    assert_eq!(num(&run(&plan, &store).rows[0][0]), 4.0);
}

/// `count(DISTINCT c)` over the two-hop chain: the distinct endpoints are
/// {carol} = 1, deduped in the bitset path.
#[test]
fn fused_count_distinct_endpoint() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .expand(1, Dir::Out, &["KNOWS".to_string()])
        .aggregate(
            vec![],
            vec![agg(AggFn::Count, Some(Expr::Slot(2)), true, "c")],
        );
    let out = run(&plan, &store);
    assert_eq!(num(&out.rows[0][0]), 1.0);
}

/// Grouped count over an Expand chain (the frontier-mode aggregate). Group
/// the reached KNOWS neighbours by name: alice→{bob,carol}, bob→{carol}, so
/// the frontier is [bob, carol, carol] → bob:1, carol:2, first-seen order.
#[test]
fn frontier_grouped_count_matches() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .aggregate(
            vec![("who".into(), prop(1, "name"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
    let out = run(&plan, &store);
    assert_eq!(as_str(&out.rows[0][0]), "bob");
    assert_eq!(num(&out.rows[0][1]), 1.0);
    assert_eq!(as_str(&out.rows[1][0]), "carol");
    assert_eq!(num(&out.rows[1][1]), 2.0);
}

/// The node-grouped count path when DISTINCT nodes share a property value —
/// the level-2 merge must combine them. a→{b,c,d}; b,d are in nyc, c in sf.
/// Group reached neighbours by city: nyc:2 (b,d), sf:1, first-seen order.
#[test]
fn node_grouped_count_merges_shared_value() {
    let mut b = Builder::default();
    let a = b.node(&["P"], &[("name", s("a"))]);
    let n1 = b.node(&["P"], &[("city", s("nyc"))]);
    let n2 = b.node(&["P"], &[("city", s("sf"))]);
    let n3 = b.node(&["P"], &[("city", s("nyc"))]);
    b.edge(a, n1, "R");
    b.edge(a, n2, "R");
    b.edge(a, n3, "R");
    let store = b.build();
    let plan = scan("P").expand(0, Dir::Out, &["R".to_string()]).aggregate(
        vec![("city".into(), prop(1, "city"))],
        vec![agg(AggFn::Count, None, false, "c")],
    );
    let out = run(&plan, &store);
    assert_eq!(as_str(&out.rows[0][0]), "nyc");
    assert_eq!(num(&out.rows[0][1]), 2.0);
    assert_eq!(as_str(&out.rows[1][0]), "sf");
    assert_eq!(num(&out.rows[1][1]), 1.0);
}

/// A grouped SUM over the frontier's property, to exercise a non-count agg on
/// the frontier path: sum the neighbours' ages by name. bob(25) reached once;
/// carol(40) reached twice → 80.
#[test]
fn frontier_grouped_sum_matches() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .aggregate(
            vec![("who".into(), prop(1, "name"))],
            vec![agg(AggFn::Sum, Some(prop(1, "age")), false, "s")],
        );
    let out = run(&plan, &store);
    assert_eq!(num(&out.rows[0][1]), 25.0); // bob
    assert_eq!(num(&out.rows[1][1]), 80.0); // carol twice
}

/// An unknown final edge label fuses to zero rows.
#[test]
fn fused_count_unknown_label_is_zero() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["NOPE".to_string()])
        .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
    assert_eq!(num(&run(&plan, &store).rows[0][0]), 0.0);
}

// --- Order + Page ---

fn asc(slot: usize, key: &str) -> crate::ir::SortKey {
    crate::ir::SortKey {
        expr: prop(slot, key),
        descending: false,
        nulls_first: false,
    }
}
fn desc(slot: usize, key: &str) -> crate::ir::SortKey {
    crate::ir::SortKey {
        expr: prop(slot, key),
        descending: true,
        nulls_first: false,
    }
}

/// NULL placement is a language contract independent of direction: GQL keeps
/// NULLs LAST in both ASC and DESC (a NULL prop must not float to the front
/// under DESC). Uses a graph where one node lacks `age`.
#[test]
fn gql_order_by_desc_keeps_nulls_last() {
    let mut b = Builder::default();
    b.node(&["P"], &[("age", n(30.0))]);
    b.node(&["P"], &[("age", n(10.0))]);
    b.node(&["P"], &[]); // no age → NULL
    let store = b.build();
    let ages =
        |q: &str| -> Vec<String> { names_of(&run(&crate::gql::parse(q).unwrap(), &store), 1) };
    // DESC: 30, 10, then NULL last (not first).
    assert_eq!(
        ages("MATCH (p:P) RETURN p.age AS a0, p.age AS a1 ORDER BY a0 DESC"),
        vec!["Num(30.0)", "Num(10.0)", "Null"]
    );
    // ASC: 10, 30, NULL last.
    assert_eq!(
        ages("MATCH (p:P) RETURN p.age AS a0, p.age AS a1 ORDER BY a0 ASC"),
        vec!["Num(10.0)", "Num(30.0)", "Null"]
    );
}

/// Gremlin's `order()` places a stored PRESENT null FIRST (the other language default) —
/// the same shared OrderPage, driven by `SortKey.nulls_first`. An ABSENT property, by
/// contrast, FILTERS the traverser (TinkerPop: a by() yielding no value drops the row) —
/// so a present-null and an absent value are NOT conflated.
#[test]
fn gremlin_order_present_null_first_absent_filtered() {
    let mut b = Builder::default();
    b.node(&["P"], &[("age", n(30.0)), ("name", s("a"))]);
    b.node(&["P"], &[("age", n(10.0)), ("name", s("b"))]);
    b.node(&["P"], &[("age", Value::Null), ("name", s("c"))]); // PRESENT null → kept, first
    b.node(&["P"], &[("name", s("d"))]); // no age at all → ABSENT → filtered out
    let store = b.build();
    let out = run(
        &crate::gremlin::parse("g.V().hasLabel('P').order().by('age').values('name')").unwrap(),
        &store,
    );
    // present-null ('c') sorts FIRST, then 10 ('b'), 30 ('a'); absent-age ('d') is dropped.
    assert_eq!(names_of(&out, 0), vec!["c", "b", "a"]);
}

/// ORDER BY age ascending, then project name.
#[test]
fn order_by_ascending() {
    let store = social();
    let plan = scan("Person")
        .order_page(vec![asc(0, "age")], None, None)
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    // ages 30,25,40 -> bob(25), alice(30), carol(40)
    assert_eq!(names_of(&out, 0), vec!["bob", "alice", "carol"]);
}

/// Descending reverses it.
#[test]
fn order_by_descending() {
    let store = social();
    let plan = scan("Person")
        .order_page(vec![desc(0, "age")], None, None)
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    assert_eq!(names_of(&out, 0), vec!["carol", "alice", "bob"]);
}

/// ORDER BY ... LIMIT is a top-k prefix of the sorted order.
#[test]
fn order_then_limit_is_top_k() {
    let store = social();
    let plan = scan("Person")
        .order_page(vec![desc(0, "age")], None, Some(2))
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    assert_eq!(names_of(&out, 0), vec!["carol", "alice"]); // two oldest
}

/// SKIP then LIMIT is a paging window over the sorted order.
#[test]
fn order_skip_limit_paging_window() {
    let store = social();
    let plan = scan("Person")
        .order_page(vec![asc(0, "age")], Some(1), Some(1))
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    // sorted bob,alice,carol; skip 1, take 1 -> alice
    assert_eq!(names_of(&out, 0), vec!["alice"]);
}

/// Nulls sort LAST in ascending order (the value contract's policy).
#[test]
fn nulls_sort_last_ascending() {
    let mut b = Builder::default();
    b.node(&["P"], &[("name", s("has30")), ("age", n(30.0))]);
    b.node(&["P"], &[("name", s("noage"))]); // null age
    b.node(&["P"], &[("name", s("has10")), ("age", n(10.0))]);
    let store = b.build();
    let plan = scan("P")
        .order_page(vec![asc(0, "age")], None, None)
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    // 10, 30, then null last
    assert_eq!(names_of(&out, 0), vec!["has10", "has30", "noage"]);
}

/// Multi-key: city ascending, then age descending within a city.
#[test]
fn multi_key_order() {
    let mut b = Builder::default();
    b.node(
        &["P"],
        &[("name", s("a")), ("city", s("nyc")), ("age", n(30.0))],
    );
    b.node(
        &["P"],
        &[("name", s("b")), ("city", s("sf")), ("age", n(40.0))],
    );
    b.node(
        &["P"],
        &[("name", s("c")), ("city", s("nyc")), ("age", n(50.0))],
    );
    let store = b.build();
    let plan = scan("P")
        .order_page(vec![asc(0, "city"), desc(0, "age")], None, None)
        .project(vec![("name".into(), prop(0, "name"))]);
    let out = run(&plan, &store);
    // nyc: c(50) before a(30); then sf: b(40)
    assert_eq!(names_of(&out, 0), vec!["c", "a", "b"]);
}

// --- Distinct ---

/// `RETURN DISTINCT city` over nyc/sf/nyc -> two rows, first-seen order.
#[test]
fn distinct_dedups_projected_column() {
    let mut b = Builder::default();
    b.node(&["P"], &[("city", s("nyc"))]);
    b.node(&["P"], &[("city", s("sf"))]);
    b.node(&["P"], &[("city", s("nyc"))]);
    let store = b.build();
    let plan = scan("P")
        .project(vec![("city".into(), prop(0, "city"))])
        .distinct();
    let out = run(&plan, &store);
    assert_eq!(names_of(&out, 0), vec!["nyc", "sf"]);
}

/// DISTINCT is over the WHOLE projected row: (city, tier) tuples dedup, so a
/// repeated city with a different tier is NOT collapsed.
#[test]
fn distinct_is_over_the_whole_row() {
    let mut b = Builder::default();
    b.node(&["P"], &[("city", s("nyc")), ("tier", n(1.0))]);
    b.node(&["P"], &[("city", s("nyc")), ("tier", n(2.0))]);
    b.node(&["P"], &[("city", s("nyc")), ("tier", n(1.0))]); // dup of row 0
    let store = b.build();
    let plan = scan("P")
        .project(vec![
            ("city".into(), prop(0, "city")),
            ("tier".into(), prop(0, "tier")),
        ])
        .distinct();
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 2);
    assert_eq!(num(&out.rows[0][1]), 1.0);
    assert_eq!(num(&out.rows[1][1]), 2.0);
}

/// DISTINCT uses the grouping notion, not predicate equality: two NaNs
/// collapse to one row.
#[test]
fn distinct_collapses_nans() {
    let mut b = Builder::default();
    b.node(&["P"], &[("v", n(f64::NAN))]);
    b.node(&["P"], &[("v", n(f64::NAN))]);
    b.node(&["P"], &[("v", n(1.0))]);
    let store = b.build();
    let plan = scan("P")
        .project(vec![("v".into(), prop(0, "v"))])
        .distinct();
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 2); // one NaN row + one 1.0 row
}

/// DISTINCT after Expand: the set of nodes reached by KNOWS from anyone.
/// DISTINCT over a VAR-LENGTH endpoint keeps only the reachable-endpoint SET (the
/// endpoint-dedup walk), byte-identical to materialize-then-dedup: `t` is reached by two
/// length-2 paths (s→m1→t, s→m2→t) yet appears once.
#[test]
fn distinct_varlength_endpoint_set() {
    let mut b = Builder::default();
    let src = b.node(&["N"], &[("name", s("s")), ("score", n(1.0))]);
    let m1 = b.node(&["N"], &[("name", s("m1")), ("score", n(50.0))]);
    let m2 = b.node(&["N"], &[("name", s("m2")), ("score", n(99.0))]);
    let t = b.node(&["N"], &[("name", s("t")), ("score", n(5.0))]);
    b.edge(src, m1, "R");
    b.edge(src, m2, "R");
    b.edge(m1, t, "R");
    b.edge(m2, t, "R");
    let st = b.build();
    let sorted = |q: &str| {
        let mut v = names_of(&run(&crate::gql::parse(q).unwrap(), &st), 0);
        v.sort();
        v
    };
    // {1,2}: len1 → m1,m2 ; len2 → t (via m1 and via m2, deduped). s is never reached.
    assert_eq!(
        sorted("MATCH (a)-[:R]->{1,2}(x) RETURN DISTINCT x.name AS n"),
        vec!["m1", "m2", "t"]
    );
    // {2,2}: exactly two hops → only t (once, despite two paths).
    assert_eq!(
        sorted("MATCH (a)-[:R]->{2,2}(x) RETURN DISTINCT x.name AS n"),
        vec!["t"]
    );
    // Endpoint WHERE (score < 60) applied over the deduped endpoints: m1(50), t(5) pass;
    // m2(99) fails. Reachable set {m1,m2,t} → {m1,t}.
    assert_eq!(
        sorted("MATCH (a)-[:R]->{1,2}(x) WHERE x.score < 60 RETURN DISTINCT x.name AS n"),
        vec!["m1", "t"]
    );
    // DISTINCT over an ENDPOINT EXPRESSION: upper(name) over {m1,m2,t} → {M1,M2,T}.
    // Exercises try_distinct_varlen_expr (dedup endpoints, then project the expression).
    assert_eq!(
        sorted("MATCH (a)-[:R]->{1,2}(x) RETURN DISTINCT upper(x.name) AS n"),
        vec!["M1", "M2", "T"]
    );
    // Var-length in the MIDDLE (a fixed R hop AFTER it): reachable x {m1,m2,t}, then
    // x→y: m1→t, m2→t, t→none. Distinct y set = {t}. Dedups at every hop.
    assert_eq!(
        sorted("MATCH (a)-[:R]->{1,2}(x)-[:R]->(y) RETURN DISTINCT y.name AS n"),
        vec!["t"]
    );
}

#[test]
fn distinct_reached_set() {
    let store = social();
    let plan = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .project(vec![("who".into(), prop(1, "name"))])
        .distinct();
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["bob", "carol"]);
}

/// DISTINCT over a hop endpoint with DUPLICATE endpoints (a node reached by several
/// edges) and a shared high-card value: the node-dedup fast path must skip the
/// duplicate node yet still collapse two DIFFERENT nodes carrying the same string —
/// single-column and composite (multi-column).
#[test]
fn distinct_frontier_dedups_duplicate_endpoints() {
    let mut bd = Builder::default();
    let b0 = bd.node(&["N"], &[("name", s("alpha")), ("city", s("x"))]);
    let b1 = bd.node(&["N"], &[("name", s("beta")), ("city", s("y"))]);
    let b2 = bd.node(&["N"], &[("name", s("alpha")), ("city", s("x"))]); // diff node, same values
    let a0 = bd.node(&["N"], &[]);
    let a1 = bd.node(&["N"], &[]);
    let a2 = bd.node(&["N"], &[]);
    let a3 = bd.node(&["N"], &[]);
    bd.edge(a0, b0, "R");
    bd.edge(a1, b0, "R"); // b0 reached twice → duplicate endpoint node
    bd.edge(a2, b1, "R");
    bd.edge(a3, b2, "R"); // same (name, city) via a different node
    let st = bd.build();
    // Single-column (Str frontier path): distinct names collapse b0's duplicate AND
    // b2's shared name → {alpha, beta}.
    let mut got = names_of(
        &run(
            &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS n").unwrap(),
            &st,
        ),
        0,
    );
    got.sort();
    assert_eq!(got, vec!["alpha", "beta"]);
    // Composite (multi-column frontier path): (alpha,x) appears via b0 and b2 → one row.
    assert_eq!(
        run(
            &crate::gql::parse("MATCH (a)-[:R]->(x) RETURN DISTINCT x.name AS n, x.city AS c")
                .unwrap(),
            &st,
        )
        .rows
        .len(),
        2
    );
}

// --- Join (multi-pattern / shared variable) ---

/// `MATCH (a)-[:KNOWS]->(b), (a)-[:WORKS_ON]->(c)` sharing `a`. Left slots
/// [a,b], right slots [a,c]; join on left a (0) == right a (0); output slots
/// [a, b, a', c]. Only alice has a WORKS_ON, so only her KNOWS rows survive.
#[test]
fn join_shared_start_variable() {
    let store = social();
    let left = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
    let right = scan("Person").expand(0, Dir::Out, &["WORKS_ON".to_string()]);
    let plan = Plan::join(left, right, vec![(0, 0)]).project(vec![
        ("a".into(), prop(0, "name")),
        ("b".into(), prop(1, "name")),
        ("c".into(), prop(3, "name")), // right slot 1 -> output slot 2+1=3
    ]);
    let out = run(&plan, &store);
    let mut pairs: Vec<(String, String, String)> = out
        .rows
        .iter()
        .map(|r| (as_str(&r[0]), as_str(&r[1]), as_str(&r[2])))
        .collect();
    pairs.sort();
    // alice KNOWS {bob,carol}, WORKS_ON {graphdb}: 2x1 = 2 rows. bob has no
    // WORKS_ON, so bob->carol drops.
    assert_eq!(
        pairs,
        vec![
            ("alice".into(), "bob".into(), "graphdb".into()),
            ("alice".into(), "carol".into(), "graphdb".into()),
        ]
    );
}

/// The join fans out to the PRODUCT per shared key: a person with 2 R and 2 S
/// neighbours yields 4 combined rows.
#[test]
fn join_is_product_per_shared_key() {
    let mut b = Builder::default();
    let a = b.node(&["P"], &[("name", s("a"))]);
    let r1 = b.node(&["P"], &[("name", s("r1"))]);
    let r2 = b.node(&["P"], &[("name", s("r2"))]);
    let s1 = b.node(&["P"], &[("name", s("s1"))]);
    let s2 = b.node(&["P"], &[("name", s("s2"))]);
    b.edge(a, r1, "R");
    b.edge(a, r2, "R");
    b.edge(a, s1, "S");
    b.edge(a, s2, "S");
    let store = b.build();
    let left = scan("P").expand(0, Dir::Out, &["R".to_string()]);
    let right = scan("P").expand(0, Dir::Out, &["S".to_string()]);
    let plan = Plan::join(left, right, vec![(0, 0)]).project(vec![
        ("r".into(), prop(1, "name")),
        ("s".into(), prop(3, "name")),
    ]);
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 4); // {r1,r2} x {s1,s2}
    let mut pairs: Vec<(String, String)> = out
        .rows
        .iter()
        .map(|r| (as_str(&r[0]), as_str(&r[1])))
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("r1".into(), "s1".into()),
            ("r1".into(), "s2".into()),
            ("r2".into(), "s1".into()),
            ("r2".into(), "s2".into()),
        ]
    );
}

/// A left key with no right match drops (inner join).
#[test]
fn join_drops_unmatched() {
    let store = social();
    // Everyone with a KNOWS edge, joined to everyone with a WORKS_ON edge on
    // the SAME person. Only alice has both, so bob (KNOWS only) drops.
    let left = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
    let right = scan("Person").expand(0, Dir::Out, &["WORKS_ON".to_string()]);
    let plan = Plan::join(left, right, vec![(0, 0)])
        .project(vec![("a".into(), prop(0, "name"))])
        .distinct();
    let out = run(&plan, &store);
    assert_eq!(names_of(&out, 0), vec!["alice"]);
}

// --- VarLength (quantified hops) ---

/// A linear chain a->b->c. `{1,2}` from a reaches b (len 1) and c (len 2):
/// two rows.
#[test]
fn varlen_chain_one_to_two() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail)
        .project(vec![("end".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["b", "c"]);
}

/// `{0,2}` includes the source itself at length 0: a, b, c.
#[test]
fn varlen_zero_includes_source() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .var_length(0, Dir::Out, &["R".to_string()], 0, 2, PathMode::Trail)
        .project(vec![("end".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["a", "b", "c"]); // a at length 0
}

/// THE trail-vs-walk discriminator: a single self-loop a->a. `{1,2}`:
/// - walk (trail=false) reuses the edge, so len1 AND len2 both reach a -> 2 rows;
/// - trail (trail=true) may not reuse it, so only len1 -> 1 row.
#[test]
fn varlen_trail_vs_walk_on_a_self_loop() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    b.edge(a, a, "R"); // self-loop
    let store = b.build();
    let base = scan("N");

    let walk = base
        .clone()
        .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Walk)
        .project(vec![("end".into(), prop(1, "name"))]);
    assert_eq!(run(&walk, &store).rows.len(), 2, "walk reuses the edge");

    let trail = base
        .var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail)
        .project(vec![("end".into(), prop(1, "name"))]);
    assert_eq!(
        run(&trail, &store).rows.len(),
        1,
        "trail may not reuse the edge"
    );
}

/// A 2-cycle a<->b (two directed edges a->b, b->a). `{1,3}` as a TRAIL from a:
/// len1 a->b (edge0); len2 a->b->a (edge0,edge1); len3 a->b->a->b (edge0,
/// edge1, then edge0 again -> reused -> blocked). So endpoints b, a -> 2 rows.
/// As a WALK, len3 a->b->a->b is allowed -> endpoints b, a, b -> 3 rows.
#[test]
fn varlen_two_cycle_trail_bounds_edge_reuse() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    b.edge(a, bb, "R"); // edge 0
    b.edge(bb, a, "R"); // edge 1
    let store = b.build();
    let from_a = scan("N").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))));

    let trail = from_a
        .clone()
        .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Trail)
        .project(vec![("end".into(), prop(1, "name"))]);
    assert_eq!(run(&trail, &store).rows.len(), 2); // b (len1), a (len2)

    let walk = from_a
        .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Walk)
        .project(vec![("end".into(), prop(1, "name"))]);
    assert_eq!(run(&walk, &store).rows.len(), 3); // b, a, b
}

/// Build the triangle a->b->c->a with a spur a->d — the ACYCLIC/SIMPLE fixture.
#[cfg(test)]
fn triangle_with_spur() -> Store {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    let d = b.node(&["N"], &[("name", s("d"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    b.edge(c, a, "R"); // closes the cycle back to the start
    b.edge(a, d, "R");
    b.build()
}

/// ACYCLIC forbids repeating ANY node — the hop c->a back to the start is
/// rejected, so from a over `{1,3}` the endpoints are b, c, d (never a).
#[test]
fn varlen_acyclic_forbids_revisiting_the_start() {
    let store = triangle_with_spur();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Acyclic)
        .project(vec![("end".into(), prop(1, "name"))]);
    let mut got = names_of(&run(&plan, &store), 0);
    got.sort();
    assert_eq!(got, vec!["b", "c", "d"]); // no `a`: acyclic can't cycle back
}

/// SIMPLE forbids repeating an INTERIOR node but PERMITS a path that closes on
/// its own start (start == end). From a over `{1,3}` the cycle a->b->c->a is a
/// legal simple (closed) path, so `a` is emitted alongside b, c, d.
#[test]
fn varlen_simple_allows_the_closing_cycle() {
    let store = triangle_with_spur();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .var_length(0, Dir::Out, &["R".to_string()], 1, 3, PathMode::Simple)
        .project(vec![("end".into(), prop(1, "name"))]);
    let mut got = names_of(&run(&plan, &store), 0);
    got.sort();
    assert_eq!(got, vec!["a", "b", "c", "d"]); // `a` via the closing cycle
}

/// Over a 2-cycle a<->b from a with `{1,4}`, the count driver must respect the
/// node modes (not the algebraic trail shortcut): SIMPLE emits b (len1) and the
/// closing a (len2) = 2; ACYCLIC emits only b = 1 (a would repeat the start).
#[test]
fn varlen_count_honors_node_modes() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    b.edge(a, bb, "R");
    b.edge(bb, a, "R");
    let store = b.build();
    let from_a = scan("N").filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))));
    let count = |mode| {
        let plan = from_a
            .clone()
            .var_length(0, Dir::Out, &["R".to_string()], 1, 4, mode)
            .aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        match run(&plan, &store).rows[0][0] {
            Value::Num(n) => n,
            ref other => panic!("want num, got {other:?}"),
        }
    };
    assert_eq!(count(PathMode::Simple), 2.0);
    assert_eq!(count(PathMode::Acyclic), 1.0);
}

/// Exact length `{2,2}` emits only the 2-hop endpoints.
#[test]
fn varlen_exact_length() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .var_length(0, Dir::Out, &["R".to_string()], 2, 2, PathMode::Trail)
        .project(vec![("end".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    assert_eq!(names_of(&out, 0), vec!["c"]); // only the 2-hop endpoint
}

// --- ShortestPath ---

/// A diamond a->b, a->c, b->d, c->d. Shortest from a: b(1), c(1), d(2). `d` is
/// reachable two ways at distance 2 but emitted ONCE (ANY-shortest).
#[test]
fn shortest_path_diamond_reaches_each_once() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    let d = b.node(&["N"], &[("name", s("d"))]);
    b.edge(a, bb, "R");
    b.edge(a, c, "R");
    b.edge(bb, d, "R");
    b.edge(c, d, "R");
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .shortest_path(
            0,
            Dir::Out,
            &["R".to_string()],
            1,
            None,
            crate::ir::ShortestSelector::Any,
            None,
        )
        .project(vec![("t".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["b", "c", "d"]); // d once, not twice
}

/// The source is not emitted, and a direct edge wins over a longer path: with
/// a->c direct AND a->b->c, c is reached at distance 1, once.
#[test]
fn shortest_path_takes_the_short_route() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    b.edge(a, c, "R"); // direct shortcut
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .shortest_path(
            0,
            Dir::Out,
            &["R".to_string()],
            1,
            None,
            crate::ir::ShortestSelector::Any,
            None,
        )
        .project(vec![("t".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["b", "c"]); // both at distance 1; source a not emitted
}

/// `max` caps the hop distance: on a chain a->b->c->d with max 2, d (distance
/// 3) is unreachable.
#[test]
fn shortest_path_respects_max_hops() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    let d = b.node(&["N"], &[("name", s("d"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    b.edge(c, d, "R");
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .shortest_path(
            0,
            Dir::Out,
            &["R".to_string()],
            1,
            Some(2),
            crate::ir::ShortestSelector::Any,
            None,
        )
        .project(vec![("t".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    assert_eq!(got, vec!["b", "c"]); // d (distance 3) beyond the cap
}

/// A cycle does not loop forever — each node is reached once. With a `+`
/// (min 1) quantifier the source IS a valid endpoint at the shortest CYCLE
/// length back to it (a->b->c->a is length 3), matching the TS engine.
#[test]
fn shortest_path_terminates_on_a_cycle() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    b.edge(c, a, "R"); // cycle back
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .shortest_path(
            0,
            Dir::Out,
            &["R".to_string()],
            1,
            None,
            crate::ir::ShortestSelector::Any,
            None,
        )
        .project(vec![("t".into(), prop(1, "name"))]);
    let out = run(&plan, &store);
    let mut got = names_of(&out, 0);
    got.sort();
    // b(1), c(2), and a(3) — the source closes the shortest cycle back to itself.
    assert_eq!(got, vec!["a", "b", "c"]);
}

// --- Lineage (path) ---

/// Look up a key in a rich Path object `{vertices, edges, length}` (a `Value::Map`).
fn path_field<'a>(v: &'a Value, k: &str) -> &'a Value {
    match v {
        Value::Map(m) => {
            &m.iter()
                .find(|(key, _)| matches!(key, Value::Str(s) if &**s == k))
                .unwrap_or_else(|| panic!("path map has no `{k}`: {v:?}"))
                .1
        }
        other => panic!("not a path map: {other:?}"),
    }
}
/// The `name` property of each element map in a `vertices`/`edges` list.
fn elem_names(list: &Value) -> Vec<String> {
    let Value::List(items) = list else {
        panic!("not a list: {list:?}")
    };
    items
        .iter()
        .map(|e| match path_field(e, "properties") {
            Value::Map(pm) => match pm
                .iter()
                .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "name"))
            {
                Some((_, Value::Str(s))) => s.to_string(),
                _ => String::new(),
            },
            _ => String::new(),
        })
        .collect()
}
/// The `id` field of each element map in a list.
fn elem_ids(list: &Value) -> Vec<String> {
    let Value::List(items) = list else {
        panic!("not a list: {list:?}")
    };
    items
        .iter()
        .map(|e| match path_field(e, "id") {
            Value::Str(s) => s.to_string(),
            other => panic!("id not a string: {other:?}"),
        })
        .collect()
}

/// A chain a->b->c. `RETURN path` over the 2-hop expand yields a rich Path whose
/// vertices are [a, b, c] and length grows one hop per expand.
#[test]
fn path_is_the_hop_sequence() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    let store = b.build();
    // (a)-[:R]->(x)-[:R]->(y) starting at a, RETURN path.
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .expand(0, Dir::Out, &["R".to_string()])
        .expand(1, Dir::Out, &["R".to_string()])
        .project(vec![("p".into(), Expr::Path)]);
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 1);
    // A rich Path `{vertices:[a,b,c], edges:[e0,e1], length:2}`.
    let p = &out.rows[0][0];
    let _ = (a, bb, c);
    assert_eq!(elem_names(path_field(p, "vertices")), vec!["a", "b", "c"]);
    assert!(matches!(path_field(p, "length"), Value::Num(x) if *x == 2.0));
}

/// Expand tracks the traversed EDGE in the lineage too: over a->b->c the
/// relationships accessor recovers edge ids [0, 1] (creation order), the
/// parallel of `path_is_the_hop_sequence` for edges.
#[test]
fn expand_lineage_tracks_edges() {
    use crate::ir::PathPart;
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a"))]);
    let bb = b.node(&["N"], &[("name", s("b"))]);
    let c = b.node(&["N"], &[("name", s("c"))]);
    b.edge(a, bb, "R"); // edge id 0
    b.edge(bb, c, "R"); // edge id 1
    let store = b.build();
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .expand(0, Dir::Out, &["R".to_string()])
        .expand(1, Dir::Out, &["R".to_string()])
        .project(vec![(
            "es".into(),
            Expr::PathAccess {
                part: PathPart::Relationships,
            },
        )]);
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 1);
    // `edges(p)` materializes each traversed edge; these have no ext id, so the
    // rendered id is the dense id as a string ("e0", "e1" — the store's fallback).
    assert_eq!(elem_ids(&out.rows[0][0]), vec!["e0", "e1"]);
}

/// A one-hop path is two nodes; the source's own path (length-0 walk via a
/// bare scan) is one node.
#[test]
fn path_length_grows_with_hops() {
    let store = social();
    // alice -KNOWS-> {bob, carol}; RETURN path per edge.
    let plan = scan("Person")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("alice"))))
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .project(vec![("p".into(), Expr::Path)]);
    let out = run(&plan, &store);
    assert_eq!(out.rows.len(), 2); // alice->bob, alice->carol
    for row in &out.rows {
        let Value::List(verts) = path_field(&row[0], "vertices") else {
            panic!("expected a path map, got {:?}", row[0])
        };
        assert_eq!(verts.len(), 2); // [alice, neighbour]
        assert!(matches!(path_field(&row[0], "length"), Value::Num(x) if *x == 1.0));
    }
}

/// GATING: a lineage-free plan builds NO sidecar (pays nothing). Only a plan
/// that reads Path tracks it. Checked at the batch level via `needs_lineage`
/// and the pulled batch's `lineage` field.
#[test]
fn lineage_free_plan_builds_no_sidecar() {
    let store = social();
    let plain = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .project(vec![("b".into(), prop(1, "name"))]);
    assert!(!super::needs_lineage(&plain), "no Path read -> no lineage");
    // The pulled batch (before the lineage-dropping Project) has no sidecar.
    let inner = scan("Person").expand(0, Dir::Out, &["KNOWS".to_string()]);
    assert!(super::pull(&inner, &store, false)
        .unwrap()
        .lineage
        .is_none());

    let with_path = scan("Person")
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .project(vec![("p".into(), Expr::Path)]);
    assert!(super::needs_lineage(&with_path), "Path read -> lineage");
    // With track=true the expand carries a sidecar.
    assert!(super::pull(&inner, &store, true).unwrap().lineage.is_some());
}

/// Lineage survives a reorder: ORDER BY over a path-tracking plan keeps each
/// row's path aligned with its row.
#[test]
fn lineage_follows_a_reorder() {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[("name", s("a")), ("age", n(1.0))]);
    let bb = b.node(&["N"], &[("name", s("b")), ("age", n(3.0))]);
    let c = b.node(&["N"], &[("name", s("c")), ("age", n(2.0))]);
    b.edge(a, bb, "R");
    b.edge(a, c, "R");
    let store = b.build();
    // a -> {b(age3), c(age2)}; order by the neighbour's age asc, RETURN path.
    let plan = scan("N")
        .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("a"))))
        .expand(0, Dir::Out, &["R".to_string()])
        .order_page(vec![asc(1, "age")], None, None)
        .project(vec![
            ("last".into(), prop(1, "name")),
            ("p".into(), Expr::Path),
        ]);
    let out = run(&plan, &store);
    // sorted by neighbour age: c(2) then b(3). Each path ends at its own node.
    assert_eq!(as_str(&out.rows[0][0]), "c");
    assert_eq!(as_str(&out.rows[1][0]), "b");
    // Each row's path ends at its own neighbour (lineage stays aligned after the
    // reorder) — check the last vertex's `name`.
    let last_name = |row: &[Value]| elem_names(path_field(&row[1], "vertices")).pop().unwrap();
    let _ = (a, bb, c);
    assert_eq!(last_name(&out.rows[0]), "c"); // path for c ends at c
    assert_eq!(last_name(&out.rows[1]), "b"); // path for b ends at b
}

#[cfg(test)]
mod perf {
    use crate::opt::optimize;
    use crate::store::{Builder, Store};
    use crate::value::Value;
    use std::time::Instant;

    fn build(nodes: usize, deg: usize) -> Store {
        let mut b = Builder::default();
        for i in 0..nodes {
            b.node(
                &["Person"],
                &[
                    ("name", Value::Str(format!("n{i}").as_str().into())),
                    ("age", Value::Num((i % 100) as f64)),
                ],
            );
        }
        for i in 0..nodes {
            for d in 0..deg {
                b.edge(i as u32, ((i * 7 + d * 13 + 1) % nodes) as u32, "R");
            }
        }
        b.build()
    }

    #[test]
    #[ignore = "perf probe"]
    fn zzz_perf() {
        let (nodes, deg) = (200_000usize, 4usize);
        let t = Instant::now();
        let store = build(nodes, deg);
        eprintln!(
            "PERF build {nodes} nodes / {} edges: {:?}",
            nodes * deg,
            t.elapsed()
        );
        for q in [
            "MATCH (p:Person) WHERE p.age > 90 RETURN p.name",
            "MATCH (a:Person)-[:R]->(b) RETURN count(*) AS c",
            "MATCH (a:Person)-[:R]->(b) RETURN b.name AS who, count(*) AS c",
            "MATCH (a:Person)-[:R]->(b) RETURN b.age AS age, count(*) AS c",
            "MATCH (a:Person)-[:R]->()-[:R]->(c) RETURN count(DISTINCT c) AS c",
            "MATCH (a:Person)-[:R]->(b)-[:R]->(c) RETURN count(*) AS c",
        ] {
            let plan = optimize(crate::gql::parse(q).unwrap());
            let mut best = f64::MAX;
            let mut rows = 0;
            for _ in 0..5 {
                let t = Instant::now();
                let out = super::run(&plan, &store);
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                rows = out.rows.len();
            }
            eprintln!("PERF {best:>9.2} ms  rows {rows:>8}  {q}");
        }
    }
}
