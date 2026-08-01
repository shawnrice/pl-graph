//! CALL micro-benchmark: named procedure calls (graph algorithms as ISO GQL
//! procedures) and inline subqueries (correlated lateral joins). Each inline
//! query is paired with the flat, non-subquery equivalent so the per-outer-row
//! subquery overhead is the difference. Run:
//!   cargo run --release --example call_bench

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

const N: usize = 20_000; // persons
const SOFTWARE: usize = 1_000;
const KNOWS_PER: usize = 4;

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
}

fn build() -> Graph {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut b = Builder::default();
    for i in 0..N {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["Person".to_string()],
            vec![("name".to_string(), Value::Str(format!("name{i}").into()))],
        ));
    }
    for j in 0..SOFTWARE {
        b.nodes.push(NodeRec::owned(
            format!("s{j}"),
            vec!["Software".to_string()],
            vec![("name".to_string(), Value::Str(format!("sw{j}").into()))],
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
        if i % 2 == 0 {
            b.edges.push(EdgeRec::owned(
                format!("p{i}"),
                format!("s{}", rng.below(SOFTWARE)),
                "CREATED".to_string(),
                vec![],
                None,
            ));
        }
    }
    b.finalize()
}

fn bench(g: &mut Graph, q: &str, iters: u32) -> (f64, usize) {
    let plan = prepare(q).unwrap();
    let params = Params::new();
    let rows = plan.execute(g, &params).unwrap().nrows;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(g, &params).unwrap();
    }
    let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    (us, rows)
}

fn pretty(u: f64) -> String {
    if u >= 1000.0 {
        format!("{:.2} ms", u / 1000.0)
    } else {
        format!("{u:.1} us")
    }
}

fn main() {
    let t = Instant::now();
    let mut g = build();
    eprintln!(
        "built graph: {} vertices, {} edges in {:.1} ms\n",
        g.vertex_count(),
        g.edge_count(),
        t.elapsed().as_secs_f64() * 1e3
    );

    println!("=== named procedure CALL (algorithms as procedures) ===");
    println!("{:<44} {:>12} {:>10}", "query", "avg", "rows");
    println!("{}", "-".repeat(68));
    let named: &[(&str, &str, u32)] = &[
        (
            "degree YIELD node,degree count(*)",
            "CALL degree() YIELD node, degree RETURN count(*) AS c",
            30,
        ),
        (
            "degree YIELD node only, LIMIT 10",
            "CALL degree() YIELD node RETURN node LIMIT 10",
            30,
        ),
        (
            "degree RETURN node [ALL 21k, full hydration]",
            "CALL degree() YIELD node RETURN node",
            10,
        ),
        (
            "degree RETURN node.name [ALL, one prop]",
            "CALL degree() YIELD node RETURN node.name AS n",
            10,
        ),
        (
            "degree count(node) [ALL, no hydration]",
            "CALL degree() YIELD node RETURN count(node) AS c",
            30,
        ),
        (
            "degree top-10 by degree",
            "CALL degree() YIELD node, degree RETURN node ORDER BY degree DESC, node LIMIT 10",
            30,
        ),
        (
            "pagerank top-10 by score",
            "CALL pagerank() YIELD node, score RETURN node ORDER BY score DESC, node LIMIT 10",
            10,
        ),
        (
            "connected_components count",
            "CALL connected_components() YIELD node RETURN count(*) AS c",
            30,
        ),
    ];
    for (label, q, iters) in named {
        let (us, rows) = bench(&mut g, q, *iters);
        println!("{label:<44} {:>12} {rows:>10}", pretty(us));
    }

    println!("\n=== inline subquery CALL vs flat equivalent ===");
    println!("{:<44} {:>12} {:>10}", "query", "avg", "rows");
    println!("{}", "-".repeat(68));
    let inline: &[(&str, &str, u32)] = &[
        (
            "[inline] per-person KNOWS count → sum",
            "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } RETURN sum(c) AS total",
            5,
        ),
        (
            "[flat]   equivalent, no subquery",
            "MATCH (p:Person)-[:KNOWS]->(f) RETURN count(*) AS total",
            50,
        ),
        (
            "[inline] per-person friends → count rows",
            "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(f) RETURN f.name AS friend } RETURN count(*) AS c",
            5,
        ),
        (
            "[flat]   equivalent lateral, no subquery",
            "MATCH (p:Person)-[:KNOWS]->(f) RETURN count(*) AS c",
            50,
        ),
        (
            "[manual] OPTIONAL MATCH + WITH group-by-p",
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f) WITH p, count(f) AS c RETURN sum(c) AS total",
            60,
        ),
        (
            "[manual] inner MATCH + WITH group-by-p",
            "MATCH (p:Person)-[:KNOWS]->(f) WITH p, count(f) AS c RETURN sum(c) AS total",
            60,
        ),
        (
            "[term]   terminal grouped agg, no ORDER BY",
            "MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name AS n, count(f) AS c",
            60,
        ),
        (
            "[term]   terminal grouped agg + ORDER BY",
            "MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name AS n, count(f) AS c ORDER BY n",
            60,
        ),
        (
            "[term]   terminal grouped agg + ORDER BY count DESC LIMIT 10",
            "MATCH (p:Person)-[:KNOWS]->(f) RETURN p.name AS n, count(f) AS c ORDER BY c DESC, n LIMIT 10",
            60,
        ),
    ];
    for (label, q, iters) in inline {
        let (us, rows) = bench(&mut g, q, *iters);
        println!("{label:<44} {:>12} {rows:>10}", pretty(us));
    }
}
