//! Query-performance-at-scale probe: build N persons with `deg` KNOWS edges each
//! and time representative query shapes, to see how latency scales as the graph
//! grows toward this machine's memory ceiling. One size per process (so freed
//! memory from a previous size can't inflate the next). Run, e.g.:
//!   cargo build --release --example scale_bench
//!   for n in 10000000 30000000 50000000; do ./target/release/examples/scale_bench $n 4; done
//! Args: <vertices> [edges_per_vertex].

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

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
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["Person".to_string()],
            vec![("age".to_string(), Value::Num((18 + (i % 62)) as f64))],
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

/// Time `q` over `g`, auto-scaling iterations to ~the same wall budget.
fn bench(g: &mut Graph, q: &str) -> (f64, i64) {
    let plan = prepare(q).unwrap();
    let p = Params::new();
    let first = plan.execute(g, &p).unwrap(); // warm
    let rows = first.nrows as i64;
    // pick iters so a slow query runs a few times, a fast one more.
    let t0 = Instant::now();
    let _ = plan.execute(g, &p).unwrap();
    let one = t0.elapsed().as_secs_f64();
    let iters = (0.5 / one).clamp(3.0, 200.0) as u32;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(g, &p).unwrap();
    }
    (t.elapsed().as_secs_f64() * 1e3 / iters as f64, rows)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);
    let eper: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let t = Instant::now();
    let mut g = build(n, eper);
    println!(
        "\nN={} vertices, {} edges  (built in {:.1}s)",
        g.vertex_count(),
        g.edge_count(),
        t.elapsed().as_secs_f64(),
    );

    // Read-only shapes first.
    for (label, q) in [
        ("label scan count", "MATCH (n:Person) RETURN count(*) AS c"),
        ("aggregate avg", "MATCH (n:Person) RETURN avg(n.age) AS a"),
        (
            "group by age",
            "MATCH (n:Person) RETURN n.age AS age, count(*) AS c",
        ),
        (
            "type traversal count",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c",
        ),
        (
            "prop scan (no index)",
            "MATCH (n:Person) WHERE n.age = 42 RETURN count(*) AS c",
        ),
    ] {
        let (ms, rows) = bench(&mut g, q);
        println!("  {label:<22} {ms:>9.3} ms   rows {rows}");
    }

    // Then the indexed point lookup (build once, A/B against the scan above).
    g.create_vertex_index("age");
    let (ms, rows) = bench(
        &mut g,
        "MATCH (n:Person) WHERE n.age = 42 RETURN count(*) AS c",
    );
    println!("  {:<22} {ms:>9.3} ms   rows {rows}", "prop seek (indexed)");

    std::hint::black_box(&g);
}
