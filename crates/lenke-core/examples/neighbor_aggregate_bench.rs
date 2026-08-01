//! Benchmark `neighborAggregate` (one native pass over a D-dim feature block)
//! against the pre-existing host approach: one GQL statement per feature dimension
//! (`OPTIONAL MATCH … WITH n, avg(m.rK) … SET`). This is the message-passing layer
//! a host-driven GCN runs each layer, so the speedup compounds over layers × dims.
//!
//! Run:  cargo run --release --example neighbor_aggregate_bench

use std::time::Instant;

use lenke_core::algo;
use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

const N: usize = 20_000;
const DEG: usize = 8;
const D: usize = 16;

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

/// `N` nodes each with scalar features `r0..r{D-1}` AND a packed list feature `h`,
/// wired with `DEG` random out-edges each.
fn build() -> Graph {
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);
    let mut b = Builder::default();
    for i in 0..N {
        let mut props: Vec<(String, Value)> = Vec::with_capacity(D + 1);
        let mut vec: Vec<Value> = Vec::with_capacity(D);
        for d in 0..D {
            let v = rng.unit();
            props.push((format!("r{d}"), Value::Num(v)));
            vec.push(Value::Num(v));
        }
        props.push(("h".to_string(), Value::List(vec)));
        b.nodes.push(NodeRec::owned(
            format!("n{i}"),
            vec!["N".to_string()],
            props,
        ));
    }
    for i in 0..N {
        for _ in 0..DEG {
            b.edges.push(EdgeRec::owned(
                format!("n{i}"),
                format!("n{}", rng.below(N)),
                "R".to_string(),
                vec![],
                None,
            ));
        }
    }
    b.finalize()
}

fn main() {
    let mut g = build();
    eprintln!(
        "graph: {} vertices, {} edges, D={D}\n",
        g.vertex_count(),
        g.edge_count()
    );

    // --- approach A: one `neighborAggregate` call over the whole D-dim block ---
    let cfg = r#"{"feature":"h","op":"mean","direction":"both","writeProperty":"h_out"}"#;
    // warm up
    algo::run(&mut g, "neighborAggregate", cfg).unwrap();
    let iters = 50u32;
    let t = Instant::now();
    for _ in 0..iters {
        algo::run(&mut g, "neighborAggregate", cfg).unwrap();
    }
    let native_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // --- approach B: one GQL statement per dimension (the host loop today) ---
    // avg over BOTH directions matching neighborAggregate's `both`.
    let plans: Vec<_> = (0..D)
        .map(|d| {
            prepare(&format!(
                "MATCH (n:N) OPTIONAL MATCH (n)-[]-(m:N) WITH n, avg(m.r{d}) AS a \
                 SET n.o{d} = coalesce(a, 0.0)"
            ))
            .unwrap()
        })
        .collect();
    let params = Params::new();
    // warm up
    for p in &plans {
        p.execute(&mut g, &params).unwrap();
    }
    let iters_b = 10u32;
    let t = Instant::now();
    for _ in 0..iters_b {
        for p in &plans {
            p.execute(&mut g, &params).unwrap();
        }
    }
    let gql_ms = t.elapsed().as_secs_f64() * 1e3 / iters_b as f64;

    println!(
        "one message-passing layer over {N} nodes / {} edges, D={D}:",
        g.edge_count()
    );
    println!("  neighborAggregate (1 native pass) : {native_ms:.2} ms");
    println!("  GQL, {D} statements (1 per dim)     : {gql_ms:.2} ms");
    println!(
        "  speedup                           : {:.1}x",
        gql_ms / native_ms
    );
}
