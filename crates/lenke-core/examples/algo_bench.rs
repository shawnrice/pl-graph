//! Benchmark the in-engine graph algorithms (native path) at scale. One graph is
//! built once; each algorithm is timed with a single warm run then a measured run
//! (these are whole-graph computations, not per-query, so one measured pass is the
//! signal). Weights are attached to every edge so weighted PageRank / Dijkstra are
//! covered too.
//!
//! Build + run (from crates/lenke-core):
//!   cargo build --release --example algo_bench
//!   ./target/release/examples/algo_bench [vertices] [edges_per_vertex]
//! With parallelism (rayon):
//!   cargo build --release --features parallel --example algo_bench

use std::time::Instant;

use lenke_core::algo;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

/// xorshift — deterministic so every run/lever sees the identical graph.
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

fn build(n: usize, eper: usize) -> Graph {
    let mut b = Builder::default();
    for i in 0..n {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["N".to_string()],
            vec![],
        ));
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for i in 0..n {
        for _ in 0..eper {
            b.edges.push(EdgeRec::owned(
                format!("p{i}"),
                format!("p{}", rng.below(n)),
                "KNOWS".to_string(),
                vec![("w".to_string(), Value::Num(rng.unit() + 0.001))],
                None,
            ));
        }
    }
    b.finalize()
}

fn bench(g: &mut Graph, label: &str, name: &str, cfg: &str) {
    let _ = algo::run(g, name, cfg).unwrap(); // warm
    let t = Instant::now();
    let rs = algo::run(g, name, cfg).unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!("  [{:>9.1} ms] {label} ({} rows)", ms, rs.rows().count());
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let eper: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let feat = if cfg!(feature = "parallel") {
        "parallel"
    } else {
        "serial"
    };
    println!("building {n} vertices x {eper} edges/vertex ({feat}) ...");
    let t = Instant::now();
    let mut g = build(n, eper);
    println!(
        "  built {} vertices, {} edges in {:.1} ms\n",
        g.vertex_count(),
        g.edge_count(),
        t.elapsed().as_secs_f64() * 1e3
    );

    bench(
        &mut g,
        "degree (out, all types)",
        "degree",
        r#"{"direction":"out"}"#,
    );
    bench(
        &mut g,
        "degree (both, typed)",
        "degree",
        r#"{"direction":"both","edgeLabel":"KNOWS"}"#,
    );
    bench(&mut g, "connectedComponents", "connectedComponents", "{}");
    bench(
        &mut g,
        "labelPropagation (10 iters)",
        "labelPropagation",
        "{}",
    );
    bench(&mut g, "peerPressure (<=30 iters)", "peerPressure", "{}");
    bench(&mut g, "pagerank (20 iters, unweighted)", "pagerank", "{}");
    bench(
        &mut g,
        "pagerank (20 iters, weighted)",
        "pagerank",
        r#"{"weightProperty":"w"}"#,
    );
    bench(
        &mut g,
        "shortestPath BFS (from p0)",
        "shortestPath",
        r#"{"source":"p0"}"#,
    );
    bench(
        &mut g,
        "shortestPath Dijkstra (from p0)",
        "shortestPath",
        r#"{"source":"p0","weightProperty":"w"}"#,
    );
}
