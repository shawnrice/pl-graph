//! GQL engine micro-benchmark. Builds a synthetic social graph and times
//! representative query shapes (label scan, join, group/aggregate, projection,
//! EXISTS, var-length), plus prepared-plan vs lower-per-call. Run:
//!   cargo run --release --example gql_bench

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::{parse, prepare};
use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec, Value};

const N: usize = 50_000; // persons
const SOFTWARE: usize = 2_000;
const KNOWS_PER: usize = 4; // out-edges per person

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
        // ~half the people create one piece of software
        if i % 2 == 0 {
            b.edges.push(EdgeRec::owned(
                format!("p{i}"),
                format!("s{}", rng.below(SOFTWARE)),
                "CREATED".to_string(),
                vec![("weight".to_string(), Value::Num(0.5))],
                None,
            ));
        }
    }
    b.finalize()
}

/// Run `q` `iters` times against `g`, return (avg microseconds, row count).
fn bench(g: &mut Graph, q: &str, iters: u32) -> (f64, usize) {
    let plan = prepare(q).unwrap();
    let params = Params::new();
    let rows = plan.execute(g, &params).unwrap().nrows; // warm up + row count
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
        "built graph: {} vertices, {} edges in {:.1} ms\n",
        g.vertex_count(),
        g.edge_count(),
        t.elapsed().as_secs_f64() * 1e3
    );

    let queries: &[(&str, &str, u32)] = &[
        ("label scan + count", "MATCH (n:Person) RETURN count(*) AS c", 200),
        ("scan + filter count", "MATCH (n:Person) WHERE n.age > 50 RETURN count(*) AS c", 200),
        ("projection LIMIT 100", "MATCH (n:Person) RETURN n.name LIMIT 100", 2000),
        ("project many rows", "MATCH (n:Person) WHERE n.age > 30 RETURN n.name, n.age", 100),
        ("1-hop join count", "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c", 100),
        ("group by + aggregate", "MATCH (n:Person) RETURN n.dept, count(*) AS c, avg(n.age) AS a", 100),
        ("group by 2 keys", "MATCH (n:Person) RETURN n.dept, n.age, count(*) AS c", 100),
        ("exists subquery", "MATCH (n:Person) WHERE EXISTS { (n)-[:KNOWS]->() } RETURN count(*) AS c", 50),
        ("edge prop filter", "MATCH (a:Person)-[r:CREATED]->(s) WHERE r.weight > 0.4 RETURN count(*) AS c", 100),
        ("project over join", "MATCH (a:Person)-[r:CREATED]->(s) RETURN a.age * 2 + 1 AS x, r.weight + 1 AS w", 100),
        ("var-length 1..2", "MATCH (a:Person {name:'name0'})-[:KNOWS]->{1,2}(b) RETURN count(*) AS c", 200),
        ("order by + limit", "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC LIMIT 20", 100),
        ("order by num, no limit", "MATCH (n:Person) RETURN n.age ORDER BY n.age DESC", 100),
        ("distinct 1 col", "MATCH (n:Person) RETURN DISTINCT n.dept", 100),
        ("distinct 2 col", "MATCH (n:Person) RETURN DISTINCT n.dept, n.age", 100),
        ("with filter carry", "MATCH (n:Person) WITH n WHERE n.age > 30 RETURN n.name", 100),
        ("with agg then filter", "MATCH (n:Person) WITH n.dept AS d, count(*) AS c WHERE c > 4000 RETURN d, c", 100),
        ("with then match expand", "MATCH (a:Person) WITH a WHERE a.age > 40 MATCH (a)-[:KNOWS]->(b) RETURN count(*) AS c", 50),
        ("with carry then match", "MATCH (a:Person) WITH a, a.age AS age MATCH (a)-[:KNOWS]->(b) WHERE b.age > age RETURN count(*) AS c", 50),
        // --- expression-heavy (isolates expression eval; the bytecode-VM target) ---
        (
            "expr-heavy filter count",
            "MATCH (n:Person) WHERE (n.age * 2 + 1) % 3 = 0 AND n.age > 20 AND abs(n.age - 40) < 15 RETURN count(*) AS c",
            200,
        ),
        (
            "expr-heavy project",
            "MATCH (n:Person) RETURN n.age * 2 + 10 AS x, abs(n.age - 30) AS y, \
             CASE WHEN n.age >= 30 THEN 'sr' ELSE 'jr' END AS t, (n.age % 7) + sqrt(n.age) AS z",
            100,
        ),
        // --- attribution A/B pairs (subtract to isolate one cost) ---
        ("[a] scan+count", "MATCH (n:Person) RETURN count(*) AS c", 300),
        ("[b] scan+count+pred", "MATCH (n:Person) WHERE n.age >= 0 RETURN count(*) AS c", 300),
        ("[c] project num col", "MATCH (n:Person) RETURN n.age", 200),
        ("[d] project str col", "MATCH (n:Person) RETURN n.name", 200),
        ("[e] 2-hop join count", "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN count(*) AS c", 30),
    ];

    println!("{:<26} {:>12} {:>12}", "query", "avg", "rows");
    println!("{}", "-".repeat(52));
    for (label, q, iters) in queries {
        let (us, rows) = bench(&mut g, q, *iters);
        let pretty = if us >= 1000.0 {
            format!("{:.2} ms", us / 1000.0)
        } else {
            format!("{us:.1} us")
        };
        println!("{label:<26} {pretty:>12} {rows:>12}");
    }

    // Prepared (lower once) vs per-call (lower every run).
    let q = "MATCH (n:Person) WHERE n.name = $who RETURN n.age AS age";
    let mut p = Params::new();
    p.insert(
        "who".to_string(),
        lenke_core::gql::eval::Val::Str("name123".into()),
    );
    let iters = 2000u32;

    let plan = prepare(q).unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(&mut g, &p).unwrap();
    }
    let prepared_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        let _ = parse(q).unwrap().execute(&mut g, &p).unwrap();
    }
    let percall_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    println!("\nprepared vs per-call (point lookup, {iters} iters):");
    println!("  prepared.execute : {prepared_us:.1} us");
    println!(
        "  parse+execute    : {percall_us:.1} us   (+{:.1}us parse/lower)",
        percall_us - prepared_us
    );

    // Arrow result encoding: typed `execute_arrow` (numeric/bool columns kept as
    // f64/bool, no Val/Value boxing) vs the RowSet path (execute → to_arrow).
    println!("\nArrow result encoding (typed vs RowSet→arrow):");
    for (label, q) in [
        (
            "3 numeric cols",
            "MATCH (n:Person) RETURN n.age, n.age * 2 AS x, n.age + 1 AS y",
        ),
        ("num + str col", "MATCH (n:Person) RETURN n.age, n.name"),
    ] {
        let plan = prepare(q).unwrap();
        let p = Params::new();
        let iters = 200u32;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = plan.execute_arrow(&mut g, &p).unwrap();
        }
        let typed = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            let rs = plan.execute(&mut g, &p).unwrap();
            let _ = lenke_core::arrow::to_arrow(&rs);
        }
        let viarow = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "  {label:<16} typed {typed:7.1} us   rowset {viarow:7.1} us   ({:.2}x)",
            viarow / typed
        );
    }

    // Property-index seeding: a single-node MATCH with an indexed eq/range hint.
    // Two passes so every query gets a true scan baseline (the index, once built,
    // is shared by `g` and would otherwise leak into later scan timings).
    let queries: &[(&str, &str, u32)] = &[
        (
            "eq inline {name}",
            "MATCH (n:Person {name:'name25000'}) RETURN n.age AS a",
            200,
        ),
        (
            "where name =",
            "MATCH (n:Person) WHERE n.name = 'name25000' RETURN n.age AS a",
            200,
        ),
        (
            "where age > 78",
            "MATCH (n:Person) WHERE n.age > 78 RETURN count(*) AS c",
            200,
        ),
        (
            "where age 30..40",
            "MATCH (n:Person) WHERE n.age >= 30 AND n.age < 40 RETURN count(*) AS c",
            200,
        ),
    ];
    let scans: Vec<f64> = queries
        .iter()
        .map(|(_, q, it)| bench(&mut g, q, *it).0)
        .collect();
    g.create_vertex_index("name");
    g.create_vertex_index("age");
    println!("\nproperty-index seeding (scan vs index seek):");
    let pretty = |u: f64| {
        if u >= 1000.0 {
            format!("{:.2} ms", u / 1000.0)
        } else {
            format!("{u:.1} us")
        }
    };
    for ((label, q, it), &scan_us) in queries.iter().zip(&scans) {
        let (idx_us, rows) = bench(&mut g, q, it * 10);
        println!(
            "  {label:<18} scan {:>10}   index {:>10}   ({:.0}x)   rows {rows}",
            pretty(scan_us),
            pretty(idx_us),
            scan_us / idx_us
        );
    }
}
