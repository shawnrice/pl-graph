//! Map/record micro-benchmark. Each Person carries BOTH scalar properties (age,
//! name, dept) AND a `meta` record `{city, tier}`, so a scalar-only query is a
//! regression baseline (maps must not slow the scalar columns) and the map
//! queries measure construction / stored-read / nested access / index seek. Run:
//!   cargo run --release --example map_bench

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::{parse, prepare};
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

const N: usize = 50_000; // persons
const KNOWS_PER: usize = 4;
const CITIES: usize = 50;

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

fn map(pairs: &[(&str, Value)]) -> Value {
    Value::Map(
        pairs
            .iter()
            .map(|(k, v)| ((*k).into(), v.clone()))
            .collect(),
    )
}

fn build() -> Graph {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut b = Builder::default();
    for i in 0..N {
        let age = 18 + (i % 62);
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["Person".to_string()],
            vec![
                ("age".to_string(), Value::Num(age as f64)),
                ("name".to_string(), Value::Str(format!("name{i}").into())),
                (
                    "dept".to_string(),
                    Value::Str(format!("d{}", i % 12).into()),
                ),
                // A stored record property on every vertex.
                (
                    "meta".to_string(),
                    map(&[
                        ("city", Value::Str(format!("c{}", i % CITIES).into())),
                        ("tier", Value::Num((i % 5) as f64)),
                    ]),
                ),
            ],
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

fn main() {
    let t = Instant::now();
    let mut g = build();
    eprintln!(
        "built graph: {} vertices ({} with a `meta` record), {} edges in {:.1} ms\n",
        g.vertex_count(),
        g.vertex_count(),
        g.edge_count(),
        t.elapsed().as_secs_f64() * 1e3
    );

    let queries: &[(&str, &str, u32)] = &[
        // --- scalar baseline (regression check: maps must not touch these) ---
        (
            "[scalar] filter count",
            "MATCH (n:Person) WHERE n.age > 50 RETURN count(*) AS c",
            200,
        ),
        (
            "[scalar] project 2 cols",
            "MATCH (n:Person) WHERE n.age > 30 RETURN n.name, n.age",
            100,
        ),
        (
            "[scalar] group by dept",
            "MATCH (n:Person) RETURN n.dept, count(*) AS c",
            100,
        ),
        (
            "[scalar] eq scan (no index)",
            "MATCH (n:Person) WHERE n.dept = 'd7' RETURN count(*) AS c",
            100,
        ),
        // --- record construction ---
        (
            "construct record/row",
            "MATCH (n:Person) RETURN {a: n.age, nm: n.name} AS r",
            50,
        ),
        (
            "construct + field access",
            "MATCH (n:Person) RETURN {a: n.age, nm: n.name}.a AS x",
            50,
        ),
        // --- stored map read / access ---
        (
            "read whole stored map",
            "MATCH (n:Person) RETURN n.meta AS m",
            50,
        ),
        (
            "nested field access",
            "MATCH (n:Person) RETURN n.meta.city AS c, n.meta.tier AS t",
            50,
        ),
        (
            "nested WHERE (scan)",
            "MATCH (n:Person) WHERE n.meta.city = 'c5' RETURN count(*) AS c",
            100,
        ),
        // --- whole-map ops ---
        (
            "distinct over maps",
            "MATCH (n:Person) RETURN DISTINCT n.meta",
            50,
        ),
        (
            "order by map",
            "MATCH (n:Person) RETURN n.meta AS m ORDER BY m LIMIT 20",
            50,
        ),
        (
            "map equality filter",
            "MATCH (n:Person) WHERE n.meta = {city:'c5', tier:2.0} RETURN count(*) AS c",
            100,
        ),
    ];

    println!("{:<32} {:>10} {:>10}", "query", "us/iter", "rows");
    println!("{}", "-".repeat(54));
    for (name, q, iters) in queries {
        let (us, rows) = bench(&mut g, q, *iters);
        println!("{name:<32} {us:>10.1} {rows:>10}");
    }

    // --- nested-field index: scan vs dotted-path seek ---
    println!("\n-- nested-field predicate: scan vs dotted-path index seek --");
    let q = "MATCH (n:Person) WHERE n.meta.city = 'c5' RETURN count(*) AS c";
    let (scan_us, rows) = bench(&mut g, q, 100);
    let _ = parse(q).unwrap();
    g.create_vertex_index("meta.city");
    let (seek_us, _) = bench(&mut g, q, 100);
    println!(
        "{:<32} {:>10.1} {:>10}",
        "meta.city = 'c5'  (scan)", scan_us, rows
    );
    println!(
        "{:<32} {:>10.1} {:>10}   ({:.1}x)",
        "meta.city = 'c5'  (index seek)",
        seek_us,
        rows,
        scan_us / seek_us
    );
}
