//! Broad Gremlin traversal micro-benchmark — the per-step `Vec<Trav>` model over
//! representative shapes (label scan, filter, 1/2-hop, values, dedup, path).
//! Run: cargo run --release --example gremlin_bench

use std::time::Instant;

use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};
use lenke_core::gremlin::{g, Traversal, P};

const N: usize = 50_000;
const SOFTWARE: usize = 2_000;
const KNOWS_PER: usize = 4;

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
            vec![
                ("name".to_string(), Value::Str(format!("name{i}").into())),
                ("age".to_string(), Value::Num((18 + (i % 62)) as f64)),
            ],
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

fn bench(graph: &mut Graph, label: &str, t: &Traversal, iters: u32) {
    let rows = t.run(graph).len();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = t.run(graph);
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let pretty = if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{us:.1} us")
    };
    println!("  {label:<32} {pretty:>11}   rows {rows}");
}

fn main() {
    let t0 = Instant::now();
    let mut graph = build();
    eprintln!(
        "built {} vertices, {} edges in {:.0} ms\n",
        graph.vertex_count(),
        graph.edge_count(),
        t0.elapsed().as_secs_f64() * 1e3
    );

    bench(
        &mut graph,
        "V().hasLabel(P).count",
        &g().V().has_label(&["Person"]).count(),
        200,
    );
    bench(
        &mut graph,
        "V().has(age>50).count",
        &g().V().has("age", P::gt(50)).count(),
        200,
    );
    bench(
        &mut graph,
        "V().hasLabel.values(name)",
        &g().V().has_label(&["Person"]).values(&["name"]),
        100,
    );
    bench(
        &mut graph,
        "V().has(age>50).values(name)",
        &g().V().has("age", P::gt(50)).values(&["name"]),
        100,
    );
    bench(
        &mut graph,
        "V(P).out(KNOWS).count",
        &g().V().has_label(&["Person"]).out(&["KNOWS"]).count(),
        100,
    );
    bench(
        &mut graph,
        "V(P).out.out(KNOWS).count",
        &g().V()
            .has_label(&["Person"])
            .out(&["KNOWS"])
            .out(&["KNOWS"])
            .count(),
        20,
    );
    bench(
        &mut graph,
        "V(P).out(KNOWS).values(name)",
        &g().V()
            .has_label(&["Person"])
            .out(&["KNOWS"])
            .values(&["name"]),
        50,
    );
    bench(
        &mut graph,
        "V(P).out(KNOWS).dedup.count",
        &g().V()
            .has_label(&["Person"])
            .out(&["KNOWS"])
            .dedup()
            .count(),
        50,
    );
    bench(
        &mut graph,
        "V(P).both(KNOWS).count",
        &g().V().has_label(&["Person"]).both(&["KNOWS"]).count(),
        50,
    );
    bench(
        &mut graph,
        "V(P).out(CREATED).hasLabel(SW).count",
        &g().V()
            .has_label(&["Person"])
            .out(&["CREATED"])
            .has_label(&["Software"])
            .count(),
        100,
    );
}
