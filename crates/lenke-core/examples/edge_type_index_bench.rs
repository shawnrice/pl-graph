//! Edge **type** index (`by_etype`) A/B: how much the always-on type bucket
//! speeds up `()-[:T]->()` patterns. Both sides run through the *same* query
//! engine and return the *same* rows — the only difference is whether the type
//! seed fires:
//!   * seek: `MATCH (a)-[r:T]->(b)` — the pattern label seeds the type bucket;
//!   * scan: `MATCH (a)-[r]->(b) WHERE r IS LABELED T` — the label moves to the
//!     WHERE, which does not seed, so the engine expands every vertex's
//!     adjacency and filters inline (the pre-change cost).
//!
//! Run: cargo run --release --example edge_type_index_bench
//!
//! The gain is selectivity-bound: a rare type seeks a tiny bucket while the scan
//! still walks all out-edges; a common type approaches parity (the bucket ≈ all
//! the edges you'd touch anyway).

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

const N: usize = 100_000; // persons
const KNOWS_PER: usize = 4; // common out-edges per person
const RARE_TOTAL: usize = 100; // a sparse, highly selective type

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

fn build() -> Graph {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut b = Builder::default();
    for i in 0..N {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["Person".to_string()],
            vec![("age".to_string(), Value::Num((18 + (i % 62)) as f64))],
        ));
    }
    for i in 0..N {
        for _ in 0..KNOWS_PER {
            b.edges.push(EdgeRec::owned(
                format!("p{i}"),
                format!("p{}", rng.below(N)),
                "KNOWS".to_string(),
                vec![],
                None,
            ));
        }
    }
    for _ in 0..RARE_TOTAL {
        b.edges.push(EdgeRec::owned(
            format!("p{}", rng.below(N)),
            format!("p{}", rng.below(N)),
            "RARE".to_string(),
            vec![],
            None,
        ));
    }
    b.finalize()
}

/// Time an engine query `iters` times; returns (avg microseconds, row count).
fn bench(g: &mut Graph, q: &str, iters: u32) -> (f64, usize) {
    let plan = prepare(q).unwrap();
    let p = Params::new();
    let rows = plan.execute(g, &p).unwrap().nrows; // warm + count
    let t = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(g, &p).unwrap();
    }
    (t.elapsed().as_secs_f64() * 1e6 / iters as f64, rows)
}

fn pretty(us: f64) -> String {
    if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{us:.1} us")
    }
}

fn main() {
    let mut g = build();
    println!(
        "graph: {} vertices, {} edges ({} KNOWS + {} RARE)\n",
        g.vertex_count(),
        g.edge_count(),
        N * KNOWS_PER,
        RARE_TOTAL,
    );
    println!("edge-type seed (vertex-first scan vs by_etype seek):");
    for (ty, bucket, iters) in [
        ("RARE", RARE_TOTAL, 1000u32),
        ("KNOWS", N * KNOWS_PER, 100u32),
    ] {
        let (scan_us, _) = bench(
            &mut g,
            &format!("MATCH (a)-[r]->(b) WHERE r IS LABELED {ty} RETURN count(*) AS c"),
            iters,
        );
        let (seek_us, _) = bench(
            &mut g,
            &format!("MATCH (a)-[r:{ty}]->(b) RETURN count(*) AS c"),
            iters,
        );
        println!(
            "  :{ty:<6} scan {:>10}   seek {:>10}   ({:.1}x)   bucket {bucket}/{}",
            pretty(scan_us),
            pretty(seek_us),
            scan_us / seek_us,
            N * KNOWS_PER + RARE_TOTAL,
        );
    }
}
