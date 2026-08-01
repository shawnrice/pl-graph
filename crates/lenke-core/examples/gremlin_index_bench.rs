//! Property-index seeding micro-benchmark for the Gremlin engine: compare a
//! `V().has(key, pred)` full scan vs an index seek, on a large vertex set.
//! Run: cargo run --release --example gremlin_index_bench

use std::time::Instant;

use lenke_core::graph::{Builder, Graph, NodeRec, Value};
use lenke_core::gremlin::{g, Traversal, P};

const N: usize = 100_000;

fn build() -> Graph {
    let mut b = Builder::default();
    for i in 0..N {
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["Person".to_string()],
            vec![
                ("name".to_string(), Value::Str(format!("name{i}").into())),
                ("age".to_string(), Value::Num((18 + (i % 62)) as f64)),
                (
                    "dept".to_string(),
                    Value::Str(format!("d{}", i % 50).into()),
                ),
            ],
        ));
    }
    b.finalize()
}

fn bench(g: &mut Graph, label: &str, t: &Traversal, iters: u32) -> usize {
    let rows = t.run(g).len();
    let start = Instant::now();
    for _ in 0..iters {
        let _ = t.run(g);
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let pretty = if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{us:.1} us")
    };
    println!("  {label:<34} {pretty:>11}   rows {rows}");
    rows
}

fn main() {
    let t0 = Instant::now();
    let mut graph = build();
    eprintln!(
        "built {} vertices in {:.0} ms\n",
        graph.vertex_count(),
        t0.elapsed().as_secs_f64() * 1e3
    );

    // The queries (built once; the seeding optimization fires at run time).
    let eq = g().V().has("name", P::eq("name54321")).values(&["name"]);
    let within = g()
        .V()
        .has("name", P::within(["name1", "name50000", "name99999"]))
        .values(&["name"]);
    let range = g().V().has("age", P::gt(75)).count(); // age in [18,79]; >75 ≈ 4/62 of rows
    let between = g().V().has("age", P::between(30, 40)).count();
    let prefix = g().V().has("name", P::starts_with("name999")).count();

    println!("=== full scan (no index) ===");
    bench(&mut graph, "eq point lookup", &eq, 200);
    bench(&mut graph, "within (3 values)", &within, 200);
    bench(&mut graph, "range age > 75", &range, 200);
    bench(&mut graph, "between age [30,40)", &between, 200);
    bench(&mut graph, "startsWith 'name999'", &prefix, 200);

    let ti = Instant::now();
    graph.create_vertex_index("name");
    graph.create_vertex_index("age");
    eprintln!(
        "\nbuilt name+age indexes in {:.1} ms\n",
        ti.elapsed().as_secs_f64() * 1e3
    );

    println!("=== index seek ===");
    bench(&mut graph, "eq point lookup", &eq, 2000);
    bench(&mut graph, "within (3 values)", &within, 2000);
    bench(&mut graph, "range age > 75", &range, 2000);
    bench(&mut graph, "between age [30,40)", &between, 2000);
    bench(&mut graph, "startsWith 'name999'", &prefix, 2000);
}
