use super::*;
use crate::store::Builder;

/// A directed triangle a→b→c→a (ids 0,1,2) plus an isolated node d (id 3).
fn triangle_plus_isolated() -> Store {
    let mut b = Builder::default();
    let a = b.node(&["N"], &[]);
    let bb = b.node(&["N"], &[]);
    let c = b.node(&["N"], &[]);
    let _d = b.node(&["N"], &[]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    b.edge(c, a, "R");
    b.build()
}

#[test]
fn degree_out_in_both() {
    let st = triangle_plus_isolated();
    // Each triangle node: 1 out, 1 in, 2 both; the isolated node: 0.
    assert_eq!(
        degree(&st, Dir::Out, None, 1),
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
    );
    assert_eq!(
        degree(&st, Dir::In, None, 1),
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
    );
    assert_eq!(
        degree(&st, Dir::Both, None, 1),
        vec![(0, 2.0), (1, 2.0), (2, 2.0), (3, 0.0)]
    );
    // An unknown edge type → all zero.
    assert_eq!(
        degree(&st, Dir::Out, Some("NOPE"), 1),
        vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    );
}

#[test]
fn closeness_reciprocal_of_summed_distances() {
    let st = triangle_plus_isolated();
    // Directed OUT on a→b→c→a: each triangle node reaches the other two at hop
    // distances 1 and 2, so Σ = 3 and closeness = 1/3. The isolated node reaches
    // nothing → sum 0 → closeness 0.
    assert_eq!(
        closeness(&st, None, None, 1),
        vec![(0, 1.0 / 3.0), (1, 1.0 / 3.0), (2, 1.0 / 3.0), (3, 0.0)]
    );
    // A named-but-unknown edge type reaches only each source → every closeness 0.
    assert_eq!(
        closeness(&st, Some("NOPE"), None, 1),
        vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    );
}

#[test]
fn scc_partitions_cycles_and_singletons() {
    let st = triangle_plus_isolated();
    // The directed triangle {0,1,2} is one SCC (rep = min member 0); the isolated
    // node {3} is its own singleton (rep 3).
    assert_eq!(
        strongly_connected_components(&st, None),
        vec![(0, 0), (1, 0), (2, 0), (3, 3)]
    );
    // A DAG (drop the closing c→a edge) has no non-trivial cycle → all singletons.
    let mut b = Builder::default();
    let a = b.node(&["N"], &[]);
    let bb = b.node(&["N"], &[]);
    let c = b.node(&["N"], &[]);
    b.edge(a, bb, "R");
    b.edge(bb, c, "R");
    let dag = b.build();
    assert_eq!(
        strongly_connected_components(&dag, None),
        vec![(0, 0), (1, 1), (2, 2)]
    );
    // A named-but-unknown edge type → every vertex is its own singleton.
    assert_eq!(
        strongly_connected_components(&st, Some("NOPE")),
        vec![(0, 0), (1, 1), (2, 2), (3, 3)]
    );
}

#[test]
fn on_cycle_flags_cycle_members_and_self_loops() {
    let st = triangle_plus_isolated();
    // Triangle members are on a cycle (1.0); the isolated node is not (0.0).
    assert_eq!(
        on_cycle(&st, None),
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
    );
    // A lone self-loop is a 1-cycle: node 1 loops on itself, node 0 does not.
    let mut b = Builder::default();
    let _a = b.node(&["N"], &[]);
    let bb = b.node(&["N"], &[]);
    b.edge(bb, bb, "R");
    let loops = b.build();
    assert_eq!(on_cycle(&loops, None), vec![(0, 0.0), (1, 1.0)]);
    // A named-but-unknown edge type → nothing is on a cycle.
    assert_eq!(
        on_cycle(&st, Some("NOPE")),
        vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    );
}

#[test]
fn betweenness_directed_triangle_and_diamond() {
    // Directed triangle 0→1→2→0: each vertex is the sole intermediary of exactly
    // one 2-hop shortest path, so every betweenness is 1.0; the isolated node 0.
    let st = triangle_plus_isolated();
    assert_eq!(
        betweenness(&st, None, None, None, 1),
        vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
    );
    // Diamond 0→1, 0→2, 1→3, 2→3: from 0 there are TWO shortest paths to 3, so
    // each middle vertex carries HALF the dependency — exercises sigma division.
    let mut b = Builder::default();
    let a = b.node(&["N"], &[]);
    let l = b.node(&["N"], &[]);
    let r = b.node(&["N"], &[]);
    let d = b.node(&["N"], &[]);
    b.edge(a, l, "R");
    b.edge(a, r, "R");
    b.edge(l, d, "R");
    b.edge(r, d, "R");
    let diamond = b.build();
    assert_eq!(
        betweenness(&diamond, None, None, None, 1),
        vec![(0, 0.0), (1, 0.5), (2, 0.5), (3, 0.0)]
    );
    // A named-but-unknown edge type → no paths → every vertex 0.0.
    assert_eq!(
        betweenness(&st, Some("NOPE"), None, None, 1),
        vec![(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    );
}

#[test]
fn personalized_pagerank_restarts_at_the_seed() {
    let st = triangle_plus_isolated();
    let ppr = |seeds: &[&str]| {
        let owned: Vec<String> = seeds.iter().map(|s| (*s).to_string()).collect();
        personalized_pagerank(
            &st,
            None,
            &owned,
            DEFAULT_DAMPING,
            DEFAULT_PAGERANK_ITERATIONS,
            1,
        )
    };
    // Seeding node 0 concentrates mass at 0, tapering around the directed cycle;
    // the unreachable isolated node gets none. (Exact values pin the f64 result.)
    let seed0 = ppr(&["0"]);
    assert_eq!(
        seed0,
        vec![
            (0, 0.375_920_077_192_677_5),
            (1, 0.319_532_065_613_775_9),
            (2, 0.304_547_857_193_546_66),
            (3, 0.0),
        ]
    );
    // Deterministic.
    assert_eq!(ppr(&["0"]), seed0);
    // Mass is conserved (a proper distribution over the reachable component).
    let mass: f64 = seed0.iter().map(|&(_, x)| x).sum();
    assert!((mass - 1.0).abs() < 1e-12, "mass {mass} should be ~1");
    // The seed is the strict maximum.
    assert!(seed0[0].1 > seed0[1].1 && seed0[0].1 > seed0[2].1);
    // Seeding node 1 is the rotational image (triangle symmetry).
    let seed1 = ppr(&["1"]);
    assert_eq!(seed1[1].1, seed0[0].1);
    assert_eq!(seed1[2].1, seed0[1].1);
    // With no resolvable seed it degenerates to global PageRank — so an unknown
    // id equals the empty-seed run, and there the isolated node gets teleport mass.
    let none = ppr(&[]);
    assert_eq!(ppr(&["999"]), none);
    assert!(none[3].1 > 0.0);
}

#[test]
fn neighbor_aggregate_gcn_normalization() {
    // 0→1, 0→2 (unweighted); features 1=[2], 2=[4]. Degrees under the OUT filter:
    // deg[0]=2 (two contributors), deg[1]=deg[2]=1 (no out-neighbours, floored to
    // 1). Each contributor's GCN factor is 1/sqrt(deg_0·deg_nbr) = 1/sqrt(2), so
    // the GCN sum at 0 is 2/sqrt(2) + 4/sqrt(2) — folded in that order (the exact
    // f64 the TS engine produces, NOT a re-derived 3·sqrt(2), which rounds one ULP
    // off in the last place).
    let mut b = Builder::default();
    let f = |x: f64| Value::List(vec![Value::Num(x)]);
    b.node(&["N"], &[]);
    b.node(&["N"], &[("h", f(2.0))]);
    b.node(&["N"], &[("h", f(4.0))]);
    let mut st = b.build();
    st.add_edge(0, 1, "R");
    st.add_edge(0, 2, "R");

    let cfg = |op: &str, gcn: bool| {
        let mut c = vec![
            ("feature".to_string(), Value::Str("h".into())),
            ("op".to_string(), Value::Str(op.into())),
            ("direction".to_string(), Value::Str("out".into())),
        ];
        if gcn {
            c.push(("norm".to_string(), Value::Str("gcn".into())));
        }
        c
    };
    let node0 =
        |op: &str, gcn: bool| format!("{:?}", neighbor_aggregate(&st, &cfg(op, gcn)).unwrap()[0].1);
    // GCN sum: 2·(1/√2) + 4·(1/√2), in fold order.
    assert_eq!(node0("sum", true), "List([Num(4.242640687119285)])");
    // Un-normalized sum is just 2 + 4 = 6.
    assert_eq!(node0("sum", false), "List([Num(6.0)])");
    // A GCN norm is meaningless for max/min → Err; a bad norm value → Err.
    assert!(neighbor_aggregate(&st, &cfg("max", true)).is_err());
    assert!(neighbor_aggregate(
        &st,
        &[
            ("feature".into(), Value::Str("h".into())),
            ("norm".into(), Value::Str("l2".into())),
        ]
    )
    .is_err());
}

#[test]
fn neighbor_aggregate_weighted_scales_by_edge_weight() {
    // 0→1 (w=1, feature [2]), 0→2 (w=3, feature [4]). Weighted sum at 0 is
    // 1·2 + 3·4 = 14; weighted mean divides by the WEIGHT sum: 14/(1+3) = 3.5 —
    // where the unweighted mean is (2+4)/2 = 3.
    let mut b = Builder::default();
    let f = |x: f64| Value::List(vec![Value::Num(x)]);
    b.node(&["N"], &[]);
    b.node(&["N"], &[("h", f(2.0))]);
    b.node(&["N"], &[("h", f(4.0))]);
    let mut st = b.build();
    let e0 = st.add_edge(0, 1, "R");
    st.set_edge_prop(e0, "w", Value::Num(1.0));
    let e1 = st.add_edge(0, 2, "R");
    st.set_edge_prop(e1, "w", Value::Num(3.0));

    let cfg = |op: &str, weighted: bool| {
        let mut c = vec![
            ("feature".to_string(), Value::Str("h".into())),
            ("op".to_string(), Value::Str(op.into())),
            ("direction".to_string(), Value::Str("out".into())),
        ];
        if weighted {
            c.push(("weightProperty".to_string(), Value::Str("w".into())));
        }
        c
    };
    let node0 = |op: &str, weighted: bool| {
        format!(
            "{:?}",
            neighbor_aggregate(&st, &cfg(op, weighted)).unwrap()[0].1
        )
    };
    assert_eq!(node0("sum", true), "List([Num(14.0)])");
    assert_eq!(node0("mean", true), "List([Num(3.5)])");
    assert_eq!(node0("mean", false), "List([Num(3.0)])");
    // A weight is meaningless for the scale-independent max/min → Err.
    assert!(neighbor_aggregate(&st, &cfg("max", true)).is_err());
    assert!(neighbor_aggregate(&st, &cfg("min", true)).is_err());
}

#[test]
fn neighbor_aggregate_folds_feature_vectors() {
    // a(0)=[1,2], b(1)=[3,4], c(2)=[5,6]; edges a→b, a→c. OUT-aggregation at a
    // folds b and c; b and c have no out-neighbour → the zero vector.
    let mut b = Builder::default();
    let vec = |xs: &[f64]| Value::List(xs.iter().map(|&x| Value::Num(x)).collect());
    let a = b.node(&["N"], &[("h", vec(&[1.0, 2.0]))]);
    let bb = b.node(&["N"], &[("h", vec(&[3.0, 4.0]))]);
    let c = b.node(&["N"], &[("h", vec(&[5.0, 6.0]))]);
    b.edge(a, bb, "R");
    b.edge(a, c, "R");
    let st = b.build();

    let run = |op: &str, extra: &[(&str, Value)]| {
        let mut cfg: Vec<(String, Value)> = vec![
            ("feature".into(), Value::Str("h".into())),
            ("op".into(), Value::Str(op.into())),
            ("direction".into(), Value::Str("out".into())),
        ];
        cfg.extend(extra.iter().map(|(k, v)| ((*k).to_string(), v.clone())));
        neighbor_aggregate(&st, &cfg).unwrap()
    };
    let node0 = |op: &str, extra: &[(&str, Value)]| match &run(op, extra)[0].1 {
        Value::List(xs) => xs
            .iter()
            .map(|v| match v {
                Value::Num(n) => *n,
                _ => panic!("non-numeric"),
            })
            .collect::<Vec<f64>>(),
        _ => panic!("not a list"),
    };
    assert_eq!(node0("sum", &[]), vec![8.0, 10.0]); // 3+5, 4+6
    assert_eq!(node0("mean", &[]), vec![4.0, 5.0]); // /2 contributors
    assert_eq!(node0("max", &[]), vec![5.0, 6.0]);
    assert_eq!(node0("min", &[]), vec![3.0, 4.0]);
    // includeSelf folds a's own [1,2] into the sum → [9,12].
    assert_eq!(
        node0("sum", &[("includeSelf", Value::Bool(true))]),
        vec![9.0, 12.0]
    );
    // b(1) has no out-neighbour → the zero vector (mean does not divide by 0).
    assert_eq!(
        format!("{:?}", run("mean", &[])[1].1),
        "List([Num(0.0), Num(0.0)])"
    );
    // Config errors surface as Err.
    assert!(neighbor_aggregate(&st, &[]).is_err()); // missing feature
    assert!(neighbor_aggregate(
        &st,
        &[
            ("feature".into(), Value::Str("h".into())),
            ("op".into(), Value::Str("median".into())),
        ]
    )
    .is_err()); // bad op
}

#[test]
fn peer_pressure_adopts_max_energy_cluster() {
    // Sink: 1→0, 2→0, 3→0. Node 0 sees equal vote energy from clusters 1, 2, 3
    // (each source has out-degree 1 → vote 1.0), so the tie goes to the smallest
    // external id, "1"; the sources have no in-edge and keep their own cluster.
    let mut b = Builder::default();
    let a = b.node(&["N"], &[]);
    let x = b.node(&["N"], &[]);
    let y = b.node(&["N"], &[]);
    let z = b.node(&["N"], &[]);
    b.edge(x, a, "R");
    b.edge(y, a, "R");
    b.edge(z, a, "R");
    let sink = b.build();
    assert_eq!(
        peer_pressure(&sink, None, DEFAULT_PEER_PRESSURE_ITERATIONS, 1),
        vec![(0, 1), (1, 1), (2, 2), (3, 3)]
    );
    // A named-but-unknown edge type → every vertex its own cluster.
    assert_eq!(
        peer_pressure(&sink, Some("NOPE"), DEFAULT_PEER_PRESSURE_ITERATIONS, 1),
        vec![(0, 0), (1, 1), (2, 2), (3, 3)]
    );
}

#[test]
fn closeness_weighted_uses_dijkstra_distances() {
    // 0→1 (w=10), 0→2 (1), 2→1 (1). Weighted closeness of 0 sums the Dijkstra
    // distances (0 + 2 + 1 = 3 → 1/3), differing from the unweighted hop sum
    // (0 + 1 + 1 = 2 → 1/2). Node 2 reaches only 1 at cost 1 → 1/1; node 1 → 0.
    let mut b = Builder::default();
    b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    let mut st = b.build();
    let e0 = st.add_edge(0, 1, "R");
    st.set_edge_prop(e0, "w", Value::Num(10.0));
    let e1 = st.add_edge(0, 2, "R");
    st.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = st.add_edge(2, 1, "R");
    st.set_edge_prop(e2, "w", Value::Num(1.0));

    assert_eq!(
        closeness(&st, None, Some("w"), 1),
        vec![(0, 1.0 / 3.0), (1, 0.0), (2, 1.0)]
    );
    assert_eq!(
        closeness(&st, None, None, 1),
        vec![(0, 1.0 / 2.0), (1, 0.0), (2, 1.0)]
    );
}

#[test]
fn betweenness_weighted_reroutes_dependency() {
    // Diamond 0→1, 0→2, 1→3, 2→3 with the 2→3 branch heavy (w=5): the unique
    // weighted shortest 0→3 goes via 1, so node 1 carries the FULL dependency
    // (1.0) and node 2 none — where the UNWEIGHTED graph splits it 0.5/0.5.
    let mut b = Builder::default();
    for _ in 0..4 {
        b.node(&["N"], &[]);
    }
    let mut st = b.build();
    let e0 = st.add_edge(0, 1, "R");
    st.set_edge_prop(e0, "w", Value::Num(1.0));
    let e1 = st.add_edge(0, 2, "R");
    st.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = st.add_edge(1, 3, "R");
    st.set_edge_prop(e2, "w", Value::Num(1.0));
    let e3 = st.add_edge(2, 3, "R");
    st.set_edge_prop(e3, "w", Value::Num(5.0));

    assert_eq!(
        betweenness(&st, None, Some("w"), None, 1),
        vec![(0, 0.0), (1, 1.0), (2, 0.0), (3, 0.0)]
    );
    assert_eq!(
        betweenness(&st, None, None, None, 1),
        vec![(0, 0.0), (1, 0.5), (2, 0.5), (3, 0.0)]
    );
}

#[test]
fn astar_matches_dijkstra_distance() {
    // 0→1 (w=10), 0→2 (1), 2→1 (1): the shortest 0→1 is the light detour (2).
    let mut b = Builder::default();
    b.node(&["N"], &[("hdist", Value::Num(1.0))]); // admissible heuristic to node 1
    b.node(&["N"], &[("hdist", Value::Num(0.0))]);
    b.node(&["N"], &[("hdist", Value::Num(0.5))]);
    let mut st = b.build();
    let e0 = st.add_edge(0, 1, "R");
    st.set_edge_prop(e0, "w", Value::Num(10.0));
    let e1 = st.add_edge(0, 2, "R");
    st.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = st.add_edge(2, 1, "R");
    st.set_edge_prop(e2, "w", Value::Num(1.0));

    // A* returns the exact shortest distance — the same as target-restricted
    // weighted Dijkstra — with or without a (admissible) heuristic.
    let dij = shortest_path(&st, Some("0"), Dir::Out, None, Some("w"), Some("1"));
    assert_eq!(dij, vec![(1, 2.0)]);
    assert_eq!(
        astar_search(&st, Some("0"), Some("1"), Dir::Out, None, Some("w"), None),
        dij
    );
    assert_eq!(
        astar_search(
            &st,
            Some("0"),
            Some("1"),
            Dir::Out,
            None,
            Some("w"),
            Some("hdist")
        ),
        dij
    );
    // Unweighted A* is the hop distance (direct edge to 1).
    assert_eq!(
        astar_search(&st, Some("0"), Some("1"), Dir::Out, None, None, None),
        vec![(1, 1.0)]
    );
    // A missing/unknown source or target → empty.
    assert!(astar_search(&st, Some("0"), Some("999"), Dir::Out, None, None, None).is_empty());
    assert!(astar_search(&st, None, Some("1"), Dir::Out, None, None, None).is_empty());
}

#[test]
fn shortest_path_weighted_dijkstra() {
    // 0→1 (weight 10), 0→2 (1), 2→1 (1). The weighted shortest 0→1 is the light
    // 2-hop detour (1+1=2), NOT the direct heavy edge — so it differs from the
    // unweighted hop distance (1).
    let mut b = Builder::default();
    b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    b.node(&["N"], &[]);
    let mut st = b.build();
    let e0 = st.add_edge(0, 1, "R");
    st.set_edge_prop(e0, "w", Value::Num(10.0));
    let e1 = st.add_edge(0, 2, "R");
    st.set_edge_prop(e1, "w", Value::Num(1.0));
    let e2 = st.add_edge(2, 1, "R");
    st.set_edge_prop(e2, "w", Value::Num(1.0));

    assert_eq!(
        shortest_path(&st, Some("0"), Dir::Out, None, Some("w"), None),
        vec![(0, 0.0), (1, 2.0), (2, 1.0)]
    );
    // Without the weight it is the plain hop distance (direct edge to 1).
    assert_eq!(
        shortest_path(&st, Some("0"), Dir::Out, None, None, None),
        vec![(0, 0.0), (1, 1.0), (2, 1.0)]
    );
    // A negative weight makes Dijkstra unsound → the empty result (the TS engine errs).
    let mut b2 = Builder::default();
    b2.node(&["N"], &[]);
    b2.node(&["N"], &[]);
    let mut st2 = b2.build();
    let en = st2.add_edge(0, 1, "R");
    st2.set_edge_prop(en, "w", Value::Num(-1.0));
    assert!(shortest_path(&st2, Some("0"), Dir::Out, None, Some("w"), None).is_empty());
}

#[test]
fn shortest_path_bfs_layers_from_source() {
    let st = triangle_plus_isolated();
    // OUT from a(ext "0") on 0→1→2→0: 0@0, 1@1, 2@2; the isolated node is
    // unreachable, so it is absent from the result.
    assert_eq!(
        shortest_path(&st, Some("0"), Dir::Out, None, None, None),
        vec![(0, 0.0), (1, 1.0), (2, 2.0)]
    );
    // IN from a walks the cycle backwards: 0@0, then c(2)@1, then b(1)@2.
    assert_eq!(
        shortest_path(&st, Some("0"), Dir::In, None, None, None),
        vec![(0, 0.0), (1, 2.0), (2, 1.0)]
    );
    // Unknown source → nothing.
    assert!(shortest_path(&st, Some("999"), Dir::Out, None, None, None).is_empty());
    assert!(shortest_path(&st, None, Dir::Out, None, None, None).is_empty());
    // A named-but-unknown edge type reaches only the source.
    assert_eq!(
        shortest_path(&st, Some("0"), Dir::Out, Some("NOPE"), None, None),
        vec![(0, 0.0)]
    );
    // A `target` restricts the result to just that vertex's distance.
    assert_eq!(
        shortest_path(&st, Some("0"), Dir::Out, None, None, Some("2")),
        vec![(2, 2.0)]
    );
    // An unreachable target (isolated node 3) or an unknown id → nothing.
    assert!(shortest_path(&st, Some("0"), Dir::Out, None, None, Some("3")).is_empty());
    assert!(shortest_path(&st, Some("0"), Dir::Out, None, None, Some("999")).is_empty());
}

#[test]
fn wcc_labels_by_smallest_member() {
    let st = triangle_plus_isolated();
    // Triangle {0,1,2} → component 0 (min); isolated 3 → component 3.
    assert_eq!(
        weakly_connected_components(&st, None),
        vec![(0, 0), (1, 0), (2, 0), (3, 3)]
    );
    // No edges considered (unknown type) → every node its own component.
    assert_eq!(
        weakly_connected_components(&st, Some("NOPE")),
        vec![(0, 0), (1, 1), (2, 2), (3, 3)]
    );
}

#[test]
fn bfs_distances_from_a_source() {
    let st = triangle_plus_isolated();
    // Out from 0: 0→b(1)→c(2); d unreachable.
    assert_eq!(
        bfs_distances(&st, 0, Dir::Out, None),
        vec![(0, 0), (1, 1), (2, 2)]
    );
    // Both from 0: c is one hop away via the incoming c→a edge.
    assert_eq!(
        bfs_distances(&st, 0, Dir::Both, None),
        vec![(0, 0), (1, 1), (2, 1)]
    );
    // A deleted / out-of-range source reaches nothing.
    assert!(bfs_distances(&st, 9, Dir::Out, None).is_empty());
}

#[test]
fn label_propagation_converges_a_triangle() {
    let st = triangle_plus_isolated();
    // The undirected triangle collapses to one label (its smallest member id,
    // 0); the isolated node keeps its own label (3).
    assert_eq!(
        label_propagation(&st, None, DEFAULT_LABEL_ITERATIONS, None, 1),
        vec![(0, 0), (1, 0), (2, 0), (3, 3)]
    );
    // No edges (unknown type) → every node keeps its own label.
    assert_eq!(
        label_propagation(&st, Some("NOPE"), DEFAULT_LABEL_ITERATIONS, None, 1),
        vec![(0, 0), (1, 1), (2, 2), (3, 3)]
    );
}

#[test]
fn pagerank_two_cycle_is_uniform_and_sums_to_one() {
    // a ↔ b (a→b, b→a): symmetric, no dangling → each rank is exactly 0.5.
    let mut b = Builder::default();
    let a = b.node(&["N"], &[]);
    let bb = b.node(&["N"], &[]);
    b.edge(a, bb, "R");
    b.edge(bb, a, "R");
    let st = b.build();
    let pr = pagerank(
        &st,
        None,
        None,
        DEFAULT_DAMPING,
        DEFAULT_PAGERANK_ITERATIONS,
        1,
    );
    assert!((pr[0].1 - 0.5).abs() < 1e-12);
    assert!((pr[1].1 - 0.5).abs() < 1e-12);
    let total: f64 = pr.iter().map(|(_, r)| r).sum();
    assert!((total - 1.0).abs() < 1e-9);
}

#[test]
fn pagerank_ranks_higher_in_degree_and_is_reproducible() {
    // 0→2, 1→2, 0→1: node 2 (in-degree 2) outranks node 1 (in-degree 1), which
    // outranks the source-only node 0.
    let mut b = Builder::default();
    let n0 = b.node(&["N"], &[]);
    let n1 = b.node(&["N"], &[]);
    let n2 = b.node(&["N"], &[]);
    b.edge(n0, n2, "R");
    b.edge(n1, n2, "R");
    b.edge(n0, n1, "R");
    let st = b.build();
    let pr = pagerank(
        &st,
        None,
        None,
        DEFAULT_DAMPING,
        DEFAULT_PAGERANK_ITERATIONS,
        1,
    );
    assert!(pr[2].1 > pr[1].1, "hub should outrank {pr:?}");
    assert!(
        pr[1].1 > pr[0].1,
        "middle should outrank the source-only {pr:?}"
    );
    let total: f64 = pr.iter().map(|(_, r)| r).sum();
    assert!((total - 1.0).abs() < 1e-9);
    // Deterministic: the same input gives a bit-identical result.
    assert_eq!(
        pr,
        pagerank(
            &st,
            None,
            None,
            DEFAULT_DAMPING,
            DEFAULT_PAGERANK_ITERATIONS,
            1
        )
    );
}

/// Two `(u32, f64)` result vectors are BIT-for-bit identical (not just `==`, which
/// would treat `-0.0`/`0.0` alike and mishandle NaN) — the byte-identity bar.
fn feq(a: &[(u32, f64)], b: &[(u32, f64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && x.1.to_bits() == y.1.to_bits())
}

/// A deterministic, RNG-free, moderately-dense directed graph (ring + fixed chords)
/// with enough structure that every algorithm produces varied per-node values —
/// the fixture for the parallel-vs-serial byte-identity check.
fn parallel_fixture() -> Store {
    let mut b = Builder::default();
    let n = 60usize;
    let ids: Vec<u32> = (0..n).map(|_| b.node(&["N"], &[])).collect();
    for i in 0..n {
        b.edge(ids[i], ids[(i + 1) % n], "R"); // ring
        b.edge(ids[i], ids[(i * 7 + 3) % n], "R"); // chords
        if i % 3 == 0 {
            b.edge(ids[i], ids[(i * 13 + 5) % n], "R");
        }
    }
    b.build()
}

/// The load-bearing invariant for this feature: every parallelized algorithm returns
/// a BIT-identical result at 8 threads and at 1 (serial). If a parallel reduction ever
/// reassociates a sum, this fails. (On a build without the `parallel` feature both
/// sides run serial, so it still holds trivially.)
#[test]
fn algorithms_are_byte_identical_across_thread_counts() {
    let g = parallel_fixture();

    assert!(
        feq(
            &degree(&g, Dir::Both, None, 1),
            &degree(&g, Dir::Both, None, 8)
        ),
        "degree diverged across thread counts"
    );
    assert!(
        feq(&closeness(&g, None, None, 1), &closeness(&g, None, None, 8)),
        "closeness diverged across thread counts"
    );
    assert!(
        feq(
            &pagerank(
                &g,
                None,
                None,
                DEFAULT_DAMPING,
                DEFAULT_PAGERANK_ITERATIONS,
                1
            ),
            &pagerank(
                &g,
                None,
                None,
                DEFAULT_DAMPING,
                DEFAULT_PAGERANK_ITERATIONS,
                8
            )
        ),
        "pagerank diverged across thread counts"
    );
    let seeds = vec!["0".to_string(), "5".to_string(), "17".to_string()];
    assert!(
        feq(
            &personalized_pagerank(
                &g,
                None,
                &seeds,
                DEFAULT_DAMPING,
                DEFAULT_PAGERANK_ITERATIONS,
                1
            ),
            &personalized_pagerank(
                &g,
                None,
                &seeds,
                DEFAULT_DAMPING,
                DEFAULT_PAGERANK_ITERATIONS,
                8
            )
        ),
        "personalized_pagerank diverged across thread counts"
    );
    assert!(
        feq(
            &betweenness(&g, None, None, None, 1),
            &betweenness(&g, None, None, None, 8)
        ),
        "betweenness (exact) diverged across thread counts"
    );
    assert!(
        feq(
            &betweenness(&g, None, None, Some(10), 1),
            &betweenness(&g, None, None, Some(10), 8)
        ),
        "betweenness (pivots) diverged across thread counts"
    );
    // Integer-labelled algorithms: exact equality across thread counts.
    assert_eq!(
        label_propagation(&g, None, DEFAULT_LABEL_ITERATIONS, None, 1),
        label_propagation(&g, None, DEFAULT_LABEL_ITERATIONS, None, 8),
        "label_propagation diverged across thread counts"
    );
    assert_eq!(
        peer_pressure(&g, None, DEFAULT_PEER_PRESSURE_ITERATIONS, 1),
        peer_pressure(&g, None, DEFAULT_PEER_PRESSURE_ITERATIONS, 8),
        "peer_pressure diverged across thread counts"
    );
}

/// The `parallelism` graph config flows through the keyed setter into
/// `effective_parallelism` (default serial).
#[test]
fn parallelism_config_sets_effective_thread_count() {
    let mut g = triangle_plus_isolated();
    assert_eq!(g.effective_parallelism(), 1); // unset ⇒ serial
    g.set_limit(crate::store::ConfigId::Parallelism, 4);
    assert_eq!(g.effective_parallelism(), 4);
}
