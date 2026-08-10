//! Head-to-head: `lenke-engine` (the from-scratch engine, with its optimizer) vs
//! `lenke-core` (the reference engine) on identical data and identical GQL.
//!
//! Both engines get the SAME logical graph — built from one deterministic model,
//! fed to lenke-engine via its `Builder` and to lenke-core via NDJSON — and the
//! SAME query text. Each shape is timed min-of-`REPS` (release) on both; the ratio
//! is core_ms / engine_ms (>1 means the new engine is faster). Every shape returns
//! scalars, so the two engines' independent id assignments never matter.
//!
//! Native only. Run (size/degree via env):
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example vs_core_bench
//!   BENCH_N=500000 BENCH_DEG=8 cargo run --release ... --example vs_core_bench

use lenke_core::gql::eval::Params as CoreParams;
use lenke_engine::store::{Builder, Store};
use lenke_engine::value::Value;
use std::time::Instant;

fn engine_fixture(n: u32, deg: u32) -> Store {
    let mut b = Builder::default();
    for i in 0..n {
        b.node(
            &["Person"],
            &[
                ("name", Value::Str(format!("n{i}").into())),
                ("age", Value::Num(f64::from(i % 100))),
            ],
        );
    }
    for i in 0..n {
        for d in 0..deg {
            b.edge(
                i,
                (i.wrapping_mul(7)
                    .wrapping_add(d.wrapping_mul(13))
                    .wrapping_add(1))
                    % n,
                "R",
            );
        }
    }
    b.build()
}

fn core_fixture(n: u32, deg: u32) -> lenke_core::graph::Graph {
    // The same graph in lenke-core's NDJSON dialect.
    let mut s = String::new();
    for i in 0..n {
        s.push_str(&format!(
            r#"{{"type":"node","id":"{i}","labels":["Person"],"properties":{{"name":"n{i}","age":{}}}}}"#,
            i % 100
        ));
        s.push('\n');
    }
    let mut e = 0u64;
    for i in 0..n {
        for d in 0..deg {
            let to = (i
                .wrapping_mul(7)
                .wrapping_add(d.wrapping_mul(13))
                .wrapping_add(1))
                % n;
            s.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","labels":["R"],"from":"{i}","to":"{to}","properties":{{}}}}"#
            ));
            s.push('\n');
            e += 1;
        }
    }
    lenke_core::ndjson::decode(&s).expect("core load")
}

fn time_engine(store: &Store, q: &str, reps: usize) -> (f64, usize) {
    let plan = lenke_engine::opt::optimize_indexed(lenke_engine::gql::parse(q).unwrap(), store);
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let out = lenke_engine::exec::run(&plan, store);
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = out.rows.len();
    }
    (best, rows)
}

fn time_core(graph: &mut lenke_core::graph::Graph, q: &str, reps: usize) -> (f64, usize) {
    let prepared = lenke_core::gql::prepare(q).unwrap();
    let params = CoreParams::new();
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let rs = prepared.execute(graph, &params).unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = rs.nrows;
    }
    (best, rows)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    const REPS: usize = 5;
    let n = env_u32("BENCH_N", 200_000);
    let deg = env_u32("BENCH_DEG", 4);
    eprintln!(
        "fixture: {n} Person nodes, degree {deg} ({} R edges)",
        u64::from(n) * u64::from(deg)
    );

    let estore = engine_fixture(n, deg);
    let mut cgraph = core_fixture(n, deg);

    let queries = [
        "MATCH (p:Person) WHERE p.age > 90 RETURN p.name AS name",
        "MATCH (p:Person) RETURN count(*) AS c",
        "MATCH (p:Person) RETURN avg(p.age) AS a",
        "MATCH (a:Person)-[:R]->(b) RETURN count(*) AS c",
        "MATCH (a:Person)-[:R]->(b) RETURN b.name AS who",
        "MATCH (a:Person)-[:R]->(b) RETURN b.age AS age, count(*) AS c",
        "MATCH (a:Person)-[:R]->()-[:R]->(c) RETURN count(*) AS c",
    ];

    println!("  engine_ms     core_ms    ratio      rows  query");
    for q in queries {
        let (e_ms, e_rows) = time_engine(&estore, q, REPS);
        let (c_ms, c_rows) = time_core(&mut cgraph, q, REPS);
        let flag = if e_rows == c_rows {
            ""
        } else {
            " (ROW COUNT DIFF!)"
        };
        println!(
            "{e_ms:>11.3} {c_ms:>11.3} {:>8.2} {e_rows:>9}  {q}{flag}",
            c_ms / e_ms
        );
    }
}
