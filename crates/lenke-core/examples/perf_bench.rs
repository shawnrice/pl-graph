//! Perf-lever benchmark: isolates the query shapes targeted by the four
//! optimization levers so before/after numbers stay comparable across changes.
//!
//!   #2 fused aggregate scan  -> agg_avg / agg_sum / agg_minmax
//!   #3 relationship-first    -> trav_count / trav_filter
//!   #1 intra-query parallel  -> scan_filter / group_by / trav_* (scale w/ cores)
//!   #4 CSR read-snapshot     -> trav_2hop (cache-locality sensitive)
//!
//! Build + run (from crates/lenke-core):
//!   cargo build --release --example perf_bench
//!   ./target/release/examples/perf_bench [vertices] [edges_per_vertex]
//! With the parallel-query feature (lever #1):
//!   cargo build --release --features parallel-query --example perf_bench
//!
//! One graph is built once; each shape is timed with auto-scaled iterations so a
//! fast query runs many times and a slow one a few, all to ~the same wall budget.

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

/// xorshift — deterministic edges so every run/lever sees the identical graph.
struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x % n as u64) as usize
    }
}

fn build(n: usize, eper: usize) -> Graph {
    let mut b = Builder::default();
    for i in 0..n {
        // Every 1000th vertex is also a `Hub` — a small, selective second label
        // (~n/1000 of them) so a pattern can be anchored at the big `Person` end or
        // the tiny `Hub` end, exposing the cost of seed/anchor selection.
        let labels = if i % 1000 == 0 {
            vec!["Person".to_string(), "Hub".to_string()]
        } else {
            vec!["Person".to_string()]
        };
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            labels,
            vec![
                ("age".to_string(), Value::Num((18 + (i % 62)) as f64)),
                // `name`: high cardinality (unique). `city`: low cardinality (~50).
                ("name".to_string(), Value::Str(format!("name{i}").into())),
                (
                    "city".to_string(),
                    Value::Str(format!("city{}", i % 50).into()),
                ),
            ],
        ));
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for i in 0..n {
        for _ in 0..eper {
            b.edges.push(EdgeRec::owned(
                format!("p{i}"),
                format!("p{}", rng.below(n)),
                "KNOWS".to_string(),
                vec![],
                None,
            ));
        }
    }
    b.finalize()
}

/// Time `q` over `g`, auto-scaling iterations to ~a fixed wall budget. Returns
/// (mean_ms, rows). The first execute warms caches / any lazy structures.
fn bench(g: &mut Graph, q: &str) -> (f64, i64) {
    let plan = prepare(q).unwrap();
    let p = Params::new();
    let first = plan.execute(g, &p).unwrap(); // warm
    let rows = first.nrows as i64;
    let t0 = Instant::now();
    let _ = plan.execute(g, &p).unwrap();
    let one = t0.elapsed().as_secs_f64();
    let iters = (0.4 / one).clamp(3.0, 500.0) as u32;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(g, &p).unwrap();
    }
    (t.elapsed().as_secs_f64() * 1e3 / iters as f64, rows)
}

/// Write-path throughput: raw mutation rate (ops/sec) for the three hot writes,
/// outside a transaction and (for property writes) inside one — so the undo-log
/// cost of the transaction path is visible. Run last, on a copy of the graph, so
/// it doesn't perturb the read timings above.
fn bench_writes(ops: usize) {
    let mut g = build(200_000, 4);
    let n = g.vertex_count() as u32;
    let mut rng = Rng(0xDEAD_BEEF_CAFE_1234);
    let rate = |ops: usize, secs: f64| ops as f64 / secs / 1e6; // M ops/sec

    // add_edge (grows adjacency + invalidates the CSR each call).
    let t = Instant::now();
    for _ in 0..ops {
        let a = rng.below(n as usize) as u32;
        let b = rng.below(n as usize) as u32;
        g.add_edge(a, b, "NEW", vec![]);
    }
    let add_edge = rate(ops, t.elapsed().as_secs_f64());

    // set_vertex_prop, outside any transaction (no undo recorded).
    let t = Instant::now();
    for i in 0..ops {
        g.set_vertex_prop((i as u32) % n, "w", Value::Num(i as f64));
    }
    let set_prop = rate(ops, t.elapsed().as_secs_f64());

    // set_vertex_prop inside a transaction (each write records an undo op).
    g.begin_tx();
    let t = Instant::now();
    for i in 0..ops {
        g.set_vertex_prop((i as u32) % n, "w2", Value::Num(i as f64));
    }
    let set_prop_tx = rate(ops, t.elapsed().as_secs_f64());
    let _ = g.commit_tx();

    println!("\n  write throughput (M ops/sec, higher is better):");
    println!("    add_edge              {add_edge:>7.2}");
    println!("    set_prop (no tx)      {set_prop:>7.2}");
    println!("    set_prop (in tx)      {set_prop_tx:>7.2}");
    std::hint::black_box(&g);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let eper: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    let t = Instant::now();
    let mut g = build(n, eper);
    println!(
        "\nN={} vertices, {} edges  (built {:.1}s)   parallel-query={}",
        g.vertex_count(),
        g.edge_count(),
        t.elapsed().as_secs_f64(),
        cfg!(feature = "parallel-query"),
    );

    // (shape label, lever tag, query)
    let shapes: &[(&str, &str, &str)] = &[
        ("agg_avg", "#2", "MATCH (n:Person) RETURN avg(n.age) AS a"),
        ("agg_sum", "#2", "MATCH (n:Person) RETURN sum(n.age) AS s"),
        (
            "agg_minmax",
            "#2",
            "MATCH (n:Person) RETURN min(n.age) AS a, max(n.age) AS b",
        ),
        (
            "scan_filter",
            "#1/#2",
            "MATCH (n:Person) WHERE n.age = 42 RETURN count(*) AS c",
        ),
        (
            "group_by",
            "#1",
            "MATCH (n:Person) RETURN n.age AS age, count(*) AS c",
        ),
        (
            "trav_count",
            "#3/#4",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c",
        ),
        (
            "trav_count_bare",
            "#3",
            "MATCH ()-[:KNOWS]->() RETURN count(*) AS c",
        ),
        (
            "trav_filter",
            "#1/#3",
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE b.age > 40 RETURN count(*) AS c",
        ),
        (
            "trav_2hop",
            "#4",
            "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(b) RETURN count(*) AS c",
        ),
        // String comparison: present literal, absent literal (all-false fast path),
        // and a DISTINCT count that folds through the scalar val_key path.
        (
            "str_eq_present",
            "#6b",
            "MATCH (n:Person) WHERE n.name = 'name500000' RETURN count(*) AS c",
        ),
        (
            "str_eq_absent",
            "#6b",
            "MATCH (n:Person) WHERE n.name = 'zzz_absent' RETURN count(*) AS c",
        ),
        (
            "str_distinct",
            "#6b",
            "MATCH (n:Person) RETURN count(DISTINCT n.city) AS c",
        ),
        // Row-returning shapes — exercise projection + output materialization
        // (Val→Value boxing, node→Map building), not just a scalar count.
        (
            "rows_scalar",
            "out",
            "MATCH (n:Person) WHERE n.age > 40 RETURN n.name AS name, n.age AS age",
        ),
        (
            "rows_orderby",
            "out",
            "MATCH (n:Person) RETURN n.name AS name ORDER BY n.age DESC, n.name LIMIT 100",
        ),
        (
            "rows_node",
            "out",
            "MATCH (n:Person) WHERE n.age > 60 RETURN n",
        ),
        (
            "trav_rows",
            "out",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS an, b.age AS ba",
        ),
        // --- unmeasured territory: the gaps that decide where to push next ---
        // Anchor asymmetry: the SAME edges reached from the big Person end vs the
        // tiny Hub end. A large fwd/bwd gap = the win an anchor-chooser would give;
        // a small gap = seed selection isn't worth a planner. (Rows, not count, so
        // it goes through seed+expand rather than the edge-anchored count shortcut.)
        // diagnostics: is the Hub bucket actually the seed? hub_count is O(1)-ish;
        // hub_out anchors 1000 Hubs and expands out — if fast, Hub-seeding works.
        ("hub_count", "diag", "MATCH (b:Hub) RETURN count(*) AS c"),
        (
            "hub_out",
            "diag",
            "MATCH (b:Hub)-[:KNOWS]->(x) RETURN x.name AS n",
        ),
        (
            "asym_fwd",
            "plan",
            "MATCH (a:Person)-[:KNOWS]->(b:Hub) RETURN a.name AS an, b.name AS bn",
        ),
        (
            "asym_bwd",
            "plan",
            "MATCH (b:Hub)<-[:KNOWS]-(a:Person) RETURN a.name AS an, b.name AS bn",
        ),
        // Labeled-endpoint COUNT: seed the tiny Hub bucket + count adjacency,
        // instead of scanning every KNOWS edge (the count-side of orientation).
        (
            "asym_cnt_fwd",
            "plan",
            "MATCH (a:Person)-[:KNOWS]->(b:Hub) RETURN count(*) AS c",
        ),
        (
            "asym_cnt_bwd",
            "plan",
            "MATCH (b:Hub)<-[:KNOWS]-(a:Person) RETURN count(*) AS c",
        ),
        // Var-length / recursive traversal — entirely on the scalar path today.
        // Full-graph depth-2 (shows scale), and reachability-from-Hubs at depth-3
        // (bounded, realistic — "everything within 3 hops of a hub").
        (
            "varlen_all_1_2",
            "varlen",
            "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN count(*) AS c",
        ),
        (
            "varlen_hub_1_3",
            "varlen",
            "MATCH (a:Hub)-[:KNOWS]->{1,3}(b) RETURN count(*) AS c",
        ),
        // Var-length grouped aggregation — goes through try_parallel_agg (not the
        // count shortcut), so it exercises the quantified per-seed accumulator.
        (
            "varlen_group",
            "varlen",
            "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN b.city AS city, count(*) AS n",
        ),
        // Multi-hop that RETURNs (not count): the degree-product shortcut can't
        // help; full 2-hop expansion + grouping.
        (
            "trav2_group",
            "multihop",
            "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.city AS city, count(*) AS n",
        ),
        // Comma-pattern join sharing `a` — the fixed left-to-right join order.
        (
            "join_multi",
            "join",
            "MATCH (a:Person)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) WHERE b.age > 60 AND c.age < 25 RETURN count(*) AS c",
        ),
        // --- unmeasured common patterns: semi/anti-joins, DISTINCT, LIMIT ---
        // EXISTS semi-join: keep `a` if it has ANY qualifying edge — should stop at
        // the first match per `a`, not enumerate every neighbor.
        (
            "exists_semi",
            "semi",
            "MATCH (a:Person) WHERE EXISTS { (a)-[:KNOWS]->(:Hub) } RETURN count(*) AS c",
        ),
        // Anti-join: sink vertices (no outgoing KNOWS).
        (
            "not_exists",
            "semi",
            "MATCH (a:Person) WHERE NOT EXISTS { (a)-[:KNOWS]->() } RETURN count(*) AS c",
        ),
        // DISTINCT neighbors of the Hubs — dedup over an expansion.
        (
            "distinct_nbr",
            "distinct",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(DISTINCT b) AS c",
        ),
        // Traversal + small LIMIT with no WHERE — should early-stop the scan.
        (
            "limit_trav",
            "limit",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS an, b.name AS bn LIMIT 100",
        ),
        // Grouped count on the START node = out-degree; group order is start-seed
        // order, so a degree shortcut *could* preserve it (unlike end-grouped).
        (
            "group_deg",
            "group",
            "MATCH (a:Hub)-[:KNOWS]->(b) RETURN a.name AS an, count(*) AS c",
        ),
        // --- round 3: wider net ---
        // 2-hop ROWS to a selective end — does orient reverse the whole path?
        (
            "multihop_sel",
            "plan",
            "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c:Hub) RETURN a.name AS an, c.name AS cn",
        ),
        // NOT EXISTS with a labeled (selective) inner endpoint — reverse anti-join.
        (
            "not_exists_hub",
            "semi",
            "MATCH (a:Person) WHERE NOT EXISTS { (a)-[:KNOWS]->(:Hub) } RETURN count(*) AS c",
        ),
        // count(*) grouped by the START node = out-degree; 1M groups, first-seen =
        // seed order (a degree shortcut could preserve it — unlike end-grouped).
        (
            "start_group",
            "group",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS an, count(*) AS c",
        ),
        // count(DISTINCT) over a 2-hop from the big end (scalar, no parallel).
        (
            "distinct_2hop",
            "distinct",
            "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c) RETURN count(DISTINCT c) AS c",
        ),
        // Substring scan over the high-cardinality name column (full 1M scan).
        (
            "contains_scan",
            "scan",
            "MATCH (n:Person) WHERE n.name CONTAINS '999' RETURN count(*) AS c",
        ),
        // Full ORDER BY over 1M rows, no LIMIT (whole-column sort).
        (
            "order_big",
            "sort",
            "MATCH (n:Person) RETURN n.name AS name ORDER BY n.age",
        ),
        // 3-hop count from the tiny Hub end (degree-product territory?).
        (
            "three_hop",
            "multihop",
            "MATCH (a:Hub)-[:KNOWS]->()-[:KNOWS]->()-[:KNOWS]->(d) RETURN count(*) AS c",
        ),
        // Anti-self-join: pairs of Persons sharing a common KNOWS target (co-citation).
        (
            "cocite",
            "join",
            "MATCH (a:Hub)-[:KNOWS]->(x)<-[:KNOWS]-(b:Person) RETURN count(*) AS c",
        ),
        // --- round 4: subqueries, ORDER-BY-count grouping, aggregates over var-len ---
        // Per-row COUNT{} subquery = out-degree; over all 1M Person.
        (
            "count_subq",
            "subq",
            "MATCH (n:Person) RETURN sum(COUNT { (n)-[:KNOWS]->() }) AS s",
        ),
        // Reverse degree — the correlated node is the ENDPOINT (a popularity
        // shape `COUNT { (:User)-[:PURCHASED]->(y) }`): per-row in-degree.
        (
            "count_subq_rev",
            "subq",
            "MATCH (n:Person) RETURN sum(COUNT { (a)-[:KNOWS]->(n) }) AS s",
        ),
        // PageRank/CC-style gather: group 8M edges by the ENDPOINT node, aggregate
        // over sources (marcus/analytics.ts). The group key is a node — stringified
        // per edge today.
        (
            "gather_by_node",
            "group",
            "MATCH (m:Person)-[:KNOWS]->(n) WITH n, sum(m.age) AS s RETURN count(*) AS c",
        ),
        // End-grouped count with ORDER BY count DESC — order is by count, not
        // first-seen, so a degree shortcut would be legal (unlike no-ORDER-BY).
        (
            "group_order_cnt",
            "group",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.age AS age, count(*) AS c ORDER BY c DESC",
        ),
        // Recommendation 2-hop through a Hub's neighbors (co-purchase shape),
        // seeded from the tiny Hub end.
        (
            "recommend",
            "join",
            "MATCH (a:Hub)-[:KNOWS]->(u)-[:KNOWS]->(y) RETURN y.name AS n",
        ),
        // Numeric range scan (age is unindexed here → full column scan).
        (
            "range_scan",
            "scan",
            "MATCH (n:Person) WHERE n.age >= 50 AND n.age <= 55 RETURN count(*) AS c",
        ),
        // Aggregate over a bounded var-length reach from the Hub end.
        (
            "varlen_sum",
            "varlen",
            "MATCH (a:Hub)-[:KNOWS]->{1,3}(b) RETURN sum(b.age) AS s",
        ),
        // Whole-node materialization: the rich node ({id,labels,properties}) over a
        // filtered scan (ISO GQL has no `properties()`; `RETURN n` materializes it).
        (
            "props_scan",
            "out",
            "MATCH (n:Person) WHERE n.age > 60 RETURN n",
        ),
    ];

    println!("  {:<14} {:<8} {:>11}   rows", "shape", "lever", "ms");
    for (label, lever, q) in shapes {
        let (ms, rows) = bench(&mut g, q);
        println!("  {label:<14} {lever:<8} {ms:>11.3}   {rows}");
    }

    std::hint::black_box(&g);
    bench_writes(500_000);
}
