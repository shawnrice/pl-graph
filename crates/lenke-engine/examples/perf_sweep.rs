//! Broad head-to-head sweep: run a battery of GQL query shapes (drawn from
//! lenke-core's `perf_bench` / `gql_bench`) through `lenke-engine` (with its
//! optimizer) and `lenke-core` on one identical Person/KNOWS graph, and print the
//! ratio core_ms/engine_ms — sorted SLOWEST-FIRST (smallest ratio = the engine's
//! worst relative shape, i.e. the next thing to optimize).
//!
//! A shape that fails to parse/run on either side is marked `n/a` (the engine
//! doesn't support that syntax yet — a separate gap, not a perf finding).
//!
//! Native only. Run (size via env):
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml --example perf_sweep
//!   BENCH_N=500000 BENCH_DEG=6 cargo run --release ... --example perf_sweep

use lenke_core::gql::eval::Params as CoreParams;
use lenke_engine::store::{Builder, Store};
use lenke_engine::value::Value;
use std::time::Instant;

const CITIES: &[&str] = &["NYC", "LA", "SF", "CHI", "SEA", "BOS", "AUS", "DEN"];
const DEPTS: &[&str] = &["eng", "sales", "ops", "hr", "legal"];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, bound: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as u32) % bound.max(1)
    }
}

fn engine_fixture(n: u32, deg: u32) -> Store {
    let mut b = Builder::default();
    for i in 0..n {
        b.node(
            &["Person"],
            &[
                ("name", Value::Str(format!("name{i}").into())),
                ("age", Value::Num(f64::from(i % 100))),
                (
                    "city",
                    Value::Str((*CITIES.get((i % 8) as usize).unwrap()).into()),
                ),
                (
                    "dept",
                    Value::Str((*DEPTS.get((i % 5) as usize).unwrap()).into()),
                ),
            ],
        );
    }
    let mut rng = Lcg(0x1234_5678);
    for i in 0..n {
        for _ in 0..deg {
            b.edge(i, rng.next(n), "KNOWS");
        }
    }
    b.build()
}

fn core_fixture(n: u32, deg: u32) -> lenke_core::graph::Graph {
    let mut s = String::new();
    for i in 0..n {
        let city = CITIES[(i % 8) as usize];
        let dept = DEPTS[(i % 5) as usize];
        s.push_str(&format!(
            r#"{{"type":"node","id":"{i}","labels":["Person"],"properties":{{"name":"name{i}","age":{},"city":"{city}","dept":"{dept}"}}}}"#,
            i % 100
        ));
        s.push('\n');
    }
    let mut rng = Lcg(0x1234_5678);
    let mut e = 0u64;
    for i in 0..n {
        for _ in 0..deg {
            let to = rng.next(n);
            s.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","labels":["KNOWS"],"from":"{i}","to":"{to}","properties":{{}}}}"#
            ));
            s.push('\n');
            e += 1;
        }
    }
    lenke_core::ndjson::decode(&s).expect("core load")
}

fn time_engine(store: &Store, q: &str, reps: usize) -> Option<(f64, usize)> {
    let plan = lenke_engine::opt::optimize(lenke_engine::gql::parse(q).ok()?);
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            lenke_engine::exec::try_run(&plan, store)
        }))
        .ok()?
        .ok()?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = out.rows.len();
    }
    Some((best, rows))
}

fn time_core(graph: &mut lenke_core::graph::Graph, q: &str, reps: usize) -> Option<(f64, usize)> {
    let prepared = lenke_core::gql::prepare(q).ok()?;
    let params = CoreParams::new();
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let rs = prepared.execute(graph, &params).ok()?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = rs.nrows;
    }
    Some((best, rows))
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    const REPS: usize = 5;
    let n = env_u32("BENCH_N", 100_000);
    let deg = env_u32("BENCH_DEG", 5);
    eprintln!(
        "fixture: {n} Person, degree {deg} ({} KNOWS edges)",
        u64::from(n) * u64::from(deg)
    );

    let estore = engine_fixture(n, deg);
    let mut cgraph = core_fixture(n, deg);

    // Shapes the engine's surface supports (WITH/EXISTS/COUNT{}/comma-patterns are
    // exercised too — they simply come back `n/a` if unsupported).
    let queries: &[&str] = &[
        // scans / filters / projection
        "MATCH (n:Person) RETURN count(*) AS c",
        "MATCH (n:Person) RETURN n.name AS name",
        "MATCH (n:Person) RETURN n.age AS age",
        "MATCH (n:Person) WHERE n.age > 50 RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.age >= 30 AND n.age < 40 RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.age > 40 RETURN n.name AS name, n.age AS age",
        "MATCH (n:Person) WHERE n.name = 'name500' RETURN count(*) AS c",
        "MATCH (n:Person) WHERE n.name CONTAINS '999' RETURN count(*) AS c",
        // aggregates
        "MATCH (n:Person) RETURN sum(n.age) AS s",
        "MATCH (n:Person) RETURN avg(n.age) AS a",
        "MATCH (n:Person) RETURN min(n.age) AS a, max(n.age) AS b",
        "MATCH (n:Person) RETURN n.age AS age, count(*) AS c",
        "MATCH (n:Person) RETURN n.city AS city, count(*) AS c",
        "MATCH (n:Person) RETURN n.dept AS d, count(*) AS c, avg(n.age) AS a",
        "MATCH (n:Person) RETURN count(DISTINCT n.city) AS c",
        // distinct
        "MATCH (n:Person) RETURN DISTINCT n.dept AS d",
        "MATCH (n:Person) RETURN DISTINCT n.dept AS d, n.age AS age",
        // order / limit
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age",
        "MATCH (n:Person) RETURN n.name AS name, n.age AS age ORDER BY age DESC LIMIT 20",
        "MATCH (n:Person) RETURN n.name AS name LIMIT 100",
        // traversal
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS n",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.age AS age, count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b) WHERE b.age > 40 RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(DISTINCT b) AS c",
        "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c) RETURN count(*) AS c",
        "MATCH (a:Person)-[:KNOWS]->()-[:KNOWS]->(c) RETURN count(DISTINCT c) AS c",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.city AS city, count(*) AS n",
        "MATCH ()-[:KNOWS]->() RETURN count(*) AS c",
    ];

    let mut rows: Vec<(f64, String, String)> = Vec::new();
    for q in queries {
        let e = time_engine(&estore, q, REPS);
        let c = time_core(&mut cgraph, q, REPS);
        let (ratio, detail) = match (e, c) {
            (Some((em, er)), Some((cm, cr))) => {
                let flag = if er == cr { "" } else { " ROWS!" };
                (
                    cm / em,
                    format!("{em:>9.3} {cm:>9.3} {:>7.2}x{flag}", cm / em),
                )
            }
            _ => (f64::INFINITY, format!("{:>9} {:>9} {:>8}", "-", "-", "n/a")),
        };
        rows.push((ratio, detail, (*q).to_string()));
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    println!("{:>9} {:>9} {:>8}  query", "engine_ms", "core_ms", "ratio");
    for (_, detail, q) in &rows {
        println!("{detail}  {q}");
    }
}
