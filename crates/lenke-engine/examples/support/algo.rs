//! Graph-algorithm cost — what each in-engine algorithm costs at scale, what the
//! opt-in multicore path buys, the message-passing `neighborAggregate` primitive,
//! and the `CALL` overhead over a direct call. Consolidates the retired
//! `algo_bench`, `neighbor_aggregate_bench` and `call_bench`.
//!
//! Cases (filter with `-- <name>`):
//!   run          — each algorithm timed once (linear ones on a big graph,
//!                  the O(V·E) ones — closeness, betweenness — on a small one).
//!   parallel     — pagerank and betweenness at 1/2/4/8 threads: the multicore
//!                  speedup (byte-identical across thread counts by construction).
//!   neighboragg  — neighborAggregate over a D-dim feature block, D = 4/16/64:
//!                  one native pass vs the D separate GQL statements it replaces.
//!   call         — a named-procedure CALL vs the direct algorithm call.

use crate::harness::{best_ms, section, social_store, Cfg};
use lenke_engine::algo;
use lenke_engine::ir::Dir;
use lenke_engine::store::{Builder, Store};
use lenke_engine::value::Value;

/// A feature-vector fixture: `nodes` nodes each carrying an `h` property that is
/// a `Value::List` of `dim` floats, plus `deg` `R` out-edges — what a GCN-style
/// neighbourhood aggregation reads.
fn feature_store(nodes: u32, deg: u32, dim: usize) -> Store {
    let mut b = Builder::default();
    for i in 0..nodes {
        let h: Vec<Value> = (0..dim)
            .map(|k| Value::Num(f64::from((i % 7) + k as u32 + 1)))
            .collect();
        b.node(&["N"], &[("h", Value::List(h))]);
    }
    let mut rng = crate::harness::Lcg::seeded();
    for i in 0..nodes {
        for _ in 0..deg {
            b.edge(i, rng.next(nodes), "R");
        }
    }
    b.build()
}

pub fn run(cfg: &Cfg) {
    if cfg.want("algo/run") {
        section("algo/run (one measured pass each)");
        let big_n = cfg.scale.unwrap_or(200_000).min(50_000) as u32;
        let big = social_store(big_n, 5);
        let small_n = 2_000u32;
        let small = social_store(small_n, 8);
        println!("{:22} {:>10} {:>10}", "algorithm", "nodes", "ms");
        let reps = cfg.reps.min(3);
        let row = |name: &str, n: u32, ms: f64| println!("{name:22} {n:>10} {ms:>10.2}");
        row(
            "degree",
            big_n,
            best_ms(reps, || algo::degree(&big, Dir::Out, None, 1)),
        );
        row(
            "wcc",
            big_n,
            best_ms(reps, || algo::weakly_connected_components(&big, None)),
        );
        row(
            "pagerank(20)",
            big_n,
            best_ms(reps, || algo::pagerank(&big, None, None, 0.85, 20, 1)),
        );
        row(
            "label_prop(10)",
            big_n,
            best_ms(reps, || algo::label_propagation(&big, None, 10, None, 1)),
        );
        row(
            "peer_pressure(10)",
            big_n,
            best_ms(reps, || algo::peer_pressure(&big, None, 10, 1)),
        );
        row(
            "closeness",
            small_n,
            best_ms(reps, || algo::closeness(&small, None, None, 1)),
        );
        row(
            "betweenness",
            small_n,
            best_ms(reps, || algo::betweenness(&small, None, None, None, 1)),
        );
    }

    if cfg.want("algo/parallel") {
        section("algo/parallel (multicore speedup)");
        let pr = social_store(cfg.scale.unwrap_or(200_000).min(100_000) as u32, 5);
        let bw = social_store(3_000, 8);
        let reps = cfg.reps.min(3);
        println!(
            "{:22} {:>8} {:>10} {:>9}",
            "algorithm", "threads", "ms", "speedup"
        );
        for (name, one_ref) in [("pagerank(30)", true), ("betweenness", false)] {
            let mut serial = f64::NAN;
            for t in [1u32, 2, 4, 8] {
                let ms = if one_ref {
                    best_ms(reps, || algo::pagerank(&pr, None, None, 0.85, 30, t))
                } else {
                    best_ms(reps, || algo::betweenness(&bw, None, None, None, t))
                };
                if t == 1 {
                    serial = ms;
                }
                println!("{name:22} {t:>8} {ms:>10.2} {:>9.2}", serial / ms);
            }
        }
    }

    if cfg.want("algo/neighboragg") {
        section("algo/neighboragg (D-dim feature aggregation)");
        let n = cfg.scale.unwrap_or(200_000).min(50_000) as u32;
        println!(
            "{:>8} {:>10} {:>10} {:>12}",
            "dim", "nodes", "ms", "ns/node"
        );
        for dim in [4usize, 16, 64] {
            let store = feature_store(n, 5, dim);
            let config = vec![
                ("feature".to_string(), Value::Str("h".into())),
                ("op".to_string(), Value::Str("mean".into())),
                ("direction".to_string(), Value::Str("out".into())),
            ];
            let ms = best_ms(cfg.reps.min(4), || {
                algo::run_procedure(&store, "neighbor_aggregate", &config)
            });
            println!(
                "{dim:>8} {n:>10} {ms:>10.2} {:>12.1}",
                ms * 1e6 / f64::from(n)
            );
        }
    }

    if cfg.want("algo/call") {
        section("algo/call (CALL vs direct)");
        let n = cfg.scale.unwrap_or(200_000).min(50_000) as u32;
        let store = social_store(n, 5);
        let direct_ms = best_ms(cfg.reps.min(4), || algo::degree(&store, Dir::Out, None, 1));
        // The named-procedure CALL path, if it parses; otherwise report why.
        let q = "CALL degree() YIELD node, degree RETURN node, degree";
        match lenke_engine::gql::parse(q) {
            Ok(raw) => {
                let plan = lenke_engine::opt::optimize_indexed(raw, &store);
                let call_ms = best_ms(cfg.reps.min(4), || lenke_engine::exec::run(&plan, &store));
                println!(
                    "{:22} {:>10} {:>10} {:>9}",
                    "degree", "direct_ms", "call_ms", "overhead"
                );
                println!(
                    "{:22} {direct_ms:>10.2} {call_ms:>10.2} {:>9.2}",
                    "",
                    call_ms / direct_ms
                );
            }
            Err(e) => println!("(CALL path n/a: {})  direct={direct_ms:.2} ms", e.trim()),
        }
    }
}
