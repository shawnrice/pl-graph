//! The `perf_bench` rows that the count-shortcut deletion (`03a873b8`) left
//! slower, pulled out so they can be worked on without paying for the whole
//! suite. Same fixture and same timing harness as `perf_bench`, so the numbers
//! are directly comparable to that file's.
//!
//! It exists because those rows are OUTSTANDING WORK, not a finished result.
//! `03a873b8` deleted fifteen count fastpaths down to four on purpose — the
//! point of the exercise being to re-derive what is worth having once, on the
//! single path, instead of eight times across shape recognizers. Most of that
//! re-derivation landed; these shapes are what is left.
//!
//! Against main at 1M vertices / 8 edges (`perf_bench`):
//!
//!     exists_semi        0.249ms -> 505.095   2028x
//!     not_exists_hub     0.248   -> 485.705   1958x
//!     varlen_group     430.142   -> 11453.840   27x
//!     varlen_all_1_2   433.541   -> 6244.285    14x
//!     join_multi       562.279   -> 5576.692    10x
//!     gather_by_node    35.280   -> 322.735      9x
//!     distinct_2hop    209.725   -> 1749.041     8x
//!
//! Each one is "count without enumerating" — a bucket length, a degree product,
//! a filtered edge set, a reachability test that stops at the first hit. That
//! is ONE idea, which is why it is worth deriving once on the shared path.
//!
//! Run it at a smaller size while iterating; 300k/8 reproduces every one of
//! them in seconds:
//!
//!     cargo run --release --example exists_probe -- 300000 8
//!
//! It prints the ANSWER next to the time on purpose. The first thing to rule
//! out for any of these is that the fast side was fast because it was wrong —
//! for all of them, both sides agree.
//!
//! The fixture builder and `bench` harness below are copied from
//! `perf_bench` so the two stay comparable.
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

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let eper: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let mut g = build(n, eper);
    // Query text is COPIED from `perf_bench`, verbatim. Retyping it from the
    // row name is how the first version of this file ended up timing different
    // questions than the table it claims to reproduce — `varlen_group` groups by
    // `b.city` (50 groups), not by a unique name (300k).
    for (name, q) in [
        ("not_exists_hub", "MATCH (a:Person) WHERE NOT EXISTS { (a)-[:KNOWS]->(:Hub) } RETURN count(*) AS c"),
        ("exists_semi", "MATCH (a:Person) WHERE EXISTS { (a)-[:KNOWS]->(:Hub) } RETURN count(*) AS c"),
        ("varlen_group", "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN b.city AS city, count(*) AS n"),
        ("varlen_all_1_2", "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN count(*) AS c"),
        ("join_multi", "MATCH (a:Person)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) WHERE b.age > 60 AND c.age < 25 RETURN count(*) AS c"),
        ("gather_by_node", "MATCH (m:Person)-[:KNOWS]->(n) WITH n, sum(m.age) AS s RETURN count(*) AS c"),
        ("distinct_2hop", "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c) RETURN count(DISTINCT c) AS c"),
        ("trav2_group", "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.city AS city, count(*) AS n"),
        ("distinct_nbr", "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(DISTINCT b) AS c"),
    ] {
        let (ms, rows) = bench(&mut g, q);
        // The ANSWER, not just the row count — a "regression" that returns a
        // different number is a semantic change, not a slowdown.
        let plan = prepare(q).unwrap();
        let rs = plan.execute(&mut g, &Params::new()).unwrap();
        let v = format!("{:?}", rs.row(0));
        println!("PROBE {name:<16} {ms:>10.3} ms  rows {rows}  answer {v}");
    }
}
