//! Micro-benchmark for the newly-added path features: per-hop edge predicates on
//! variable-length segments, and the ISO path selectors (ANY / ANY SHORTEST /
//! ALL SHORTEST / SHORTEST k [GROUP] / bare enumeration).
//!
//! Two fixtures:
//!  - a sparse weighted random graph (out-degree 3) for the per-hop predicate
//!    cost, seeded from one vertex with a bounded `{1,4}` quantifier;
//!  - a "chain with skips" DAG (i→i+1, i→i+2) for the selector comparison: it is
//!    acyclic (so `->*` terminates) and reaches each endpoint by Fibonacci-many
//!    paths of several lengths — the shape that separates the BFS drivers
//!    (ANY/ALL SHORTEST) from the trail-enumerating ones (ANY / SHORTEST k).
//!
//! Run:  cargo run --release --example path_selector_bench

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sparse weighted random graph: `n` vertices, `deg` out-edges each, edge weight
/// in [0,1). A single labelled "seed" vertex (`s0`) anchors the var-length query.
fn weighted_graph(n: usize, deg: usize) -> Graph {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let mut b = Builder::default();
    b.nodes.push(NodeRec {
        id: "s0".to_string(),
        labels: vec!["Seed".to_string()],
        props: vec![("id".to_string(), Value::Str("s0".into()))],
    });
    for i in 0..n {
        b.nodes.push(NodeRec {
            id: format!("n{i}"),
            labels: vec!["N".to_string()],
            props: vec![("id".to_string(), Value::Num(i as f64))],
        });
    }
    // The seed points at `deg` random N vertices; every N vertex points at `deg`
    // more, so a `{1,4}` walk fans out ~deg^k.
    let out = |b: &mut Builder, src: String, rng: &mut Rng| {
        for _ in 0..deg {
            b.edges.push(EdgeRec {
                src: src.clone(),
                dst: format!("n{}", rng.below(n)),
                etype: "R".to_string(),
                props: vec![("w".to_string(), Value::Num(rng.unit()))],
                id: None,
            });
        }
    };
    out(&mut b, "s0".to_string(), &mut rng);
    for i in 0..n {
        out(&mut b, format!("n{i}"), &mut rng);
    }
    b.finalize()
}

/// Chain with skips: `m` vertices `c0..c{m-1}`, edges i→i+1 and i→i+2. Acyclic, so
/// `(c0)-[:R]->*(x)` terminates; endpoint `cj` is reached by fib(j)-many paths of
/// lengths ranging from ⌈j/2⌉ to j.
fn skip_chain(m: usize) -> Graph {
    let mut b = Builder::default();
    for i in 0..m {
        b.nodes.push(NodeRec {
            id: format!("c{i}"),
            labels: vec!["C".to_string()],
            props: vec![("id".to_string(), Value::Num(i as f64))],
        });
    }
    for i in 0..m {
        for step in 1..=2 {
            if i + step < m {
                b.edges.push(EdgeRec {
                    src: format!("c{i}"),
                    dst: format!("c{}", i + step),
                    etype: "R".to_string(),
                    props: vec![],
                    id: None,
                });
            }
        }
    }
    b.finalize()
}

/// Run `q` `iters` times, return (avg microseconds, row count).
fn bench(g: &mut Graph, q: &str, iters: u32) -> (f64, usize) {
    let plan = prepare(q).unwrap_or_else(|e| panic!("prepare `{q}`: {e}"));
    let params = Params::new();
    let rows = plan.execute(g, &params).unwrap().nrows; // warm up + row count
    let t = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(g, &params).unwrap();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    (us, rows)
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
    println!("{:<40} {:>12} {:>10}", "query", "avg", "rows");
}

fn row(g: &mut Graph, label: &str, q: &str, iters: u32) {
    let (us, rows) = bench(g, q, iters);
    let pretty = if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{us:.1} us")
    };
    println!("{label:<40} {pretty:>12} {rows:>10}");
}

fn main() {
    // --- per-hop predicate overhead ---------------------------------------
    let mut wg = weighted_graph(20_000, 3);
    eprintln!(
        "weighted graph: {} vertices, {} edges",
        wg.vertex_count(),
        wg.edge_count()
    );
    section("per-hop predicate on var-length {1,4} from one seed (deg 3 → ~120 trails)");
    row(
        &mut wg,
        "bare (no predicate)",
        "MATCH (a:Seed)-[:R]->{1,4}(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut wg,
        "predicate, always-true (w >= 0)",
        "MATCH (a:Seed)-[e:R WHERE e.w >= 0]->{1,4}(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut wg,
        "predicate, ~50% selective (w > 0.5)",
        "MATCH (a:Seed)-[e:R WHERE e.w > 0.5]->{1,4}(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut wg,
        "predicate, ~10% selective (w > 0.9)",
        "MATCH (a:Seed)-[e:R WHERE e.w > 0.9]->{1,4}(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut wg,
        "inline prop (never matches, w=0.5)",
        "MATCH (a:Seed)-[:R {w:0.5}]->{1,4}(x) RETURN count(*) AS c",
        2000,
    );

    // --- selector comparison ----------------------------------------------
    let mut sc = skip_chain(24);
    eprintln!(
        "\nskip-chain: {} vertices, {} edges",
        sc.vertex_count(),
        sc.edge_count()
    );
    section("selectors over (c0)-[:R]->*(x), all endpoints [BFS drivers vs trail enum]");
    row(
        &mut sc,
        "ANY SHORTEST count (BFS)",
        "MATCH ANY SHORTEST (a:C {id:0})-[:R]->*(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut sc,
        "ANY count (trail enum + dedup)",
        "MATCH ANY (a:C {id:0})-[:R]->*(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut sc,
        "ALL SHORTEST count (BFS DAG)",
        "MATCH ALL SHORTEST (a:C {id:0})-[:R]->*(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut sc,
        "SHORTEST 1 GROUP count (== ALL SHORTEST)",
        "MATCH SHORTEST 1 GROUP (a:C {id:0})-[:R]->*(x) RETURN count(*) AS c",
        2000,
    );
    row(
        &mut sc,
        "SHORTEST 3 count (trail enum)",
        "MATCH SHORTEST 3 (a:C {id:0})-[:R]->*(x) RETURN count(*) AS c",
        100,
    );
    row(
        &mut sc,
        "SHORTEST 3 GROUP count (trail enum)",
        "MATCH SHORTEST 3 GROUP (a:C {id:0})-[:R]->*(x) RETURN count(*) AS c",
        100,
    );
    row(
        &mut sc,
        "bare enumeration count (every trail)",
        "MATCH (a:C {id:0})-[:R]->*(x) RETURN count(*) AS c",
        100,
    );

    section("selectors binding a Path — p = … RETURN path_length(p) (materialization)");
    row(
        &mut sc,
        "ANY SHORTEST p, one per endpoint",
        "MATCH p = ANY SHORTEST (a:C {id:0})-[:R]->*(x) RETURN path_length(p) AS l",
        2000,
    );
    row(
        &mut sc,
        "ANY p, one per endpoint",
        "MATCH p = ANY (a:C {id:0})-[:R]->*(x) RETURN path_length(p) AS l",
        2000,
    );
    row(
        &mut sc,
        "SHORTEST 3 p",
        "MATCH p = SHORTEST 3 (a:C {id:0})-[:R]->*(x) RETURN path_length(p) AS l",
        100,
    );
}
