//! Storage-side cost — what a bulk write pays, and what a multi-label edge costs
//! the whole graph. Consolidates the write side of the retired `storage_probe`
//! and `edge_label_transition_bench`.
//!
//! (The read-ceiling half of `storage_probe` is the adjacency walk that
//! `query_bench -- gql`/`gremlin` already prices via 1-hop/2-hop, and the
//! eval-vs-columnar floor needs crate-private `eval_vec` access — it lives as an
//! ignored test, not here. See examples/README.md.)
//!
//! Cases (filter with `-- <name>`):
//!   write       — a bulk SET over every node: write throughput (elem/s). Each
//!                 rep re-parses (cheap) and re-runs the write on the same store.
//!   multilabel  — single-label vs two-label edges: resident bytes/edge and the
//!                 1-hop query cost, i.e. what a second label on every edge costs.

use crate::harness::{best_ms, rss_bytes, section, social_store, Cfg};
use lenke_engine::ndjson::from_ndjson;

pub fn run(cfg: &Cfg) {
    if cfg.want("storage/write") {
        section("storage/write (bulk SET throughput)");
        let n = cfg.scale.unwrap_or(200_000).min(200_000) as u32;
        let mut store = social_store(n, 5);
        let q = "MATCH (p:Person) SET p.age = p.age + 1";
        let ms = best_ms(cfg.reps, || {
            let plan = lenke_engine::gql::parse(q).expect("parses");
            lenke_engine::exec::run_query(plan, &mut store).expect("writes")
        });
        println!(
            "{:22} {:>10} {:>10} {:>12}",
            "SET p.age", "nodes", "ms", "Melem/s"
        );
        println!(
            "{:22} {n:>10} {ms:>10.2} {:>12.1}",
            "",
            f64::from(n) / (ms / 1e3) / 1e6
        );
    }

    if cfg.want("storage/multilabel") {
        section("storage/multilabel (a second label on every edge)");
        let n = cfg.scale.unwrap_or(200_000).min(100_000);
        // Same nodes and edge endpoints; only the edge label set differs.
        let build = |labels: &str| -> String {
            let mut lines: Vec<String> = (0..n)
                .map(|i| {
                    format!(
                        r#"{{"type":"node","id":"v{i}","labels":["P"],"properties":{{"age":{}}}}}"#,
                        i % 100
                    )
                })
                .collect();
            let mut rng = crate::harness::Lcg::seeded();
            for i in 0..n {
                let to = rng.next(n as u32);
                lines.push(format!(
                    r#"{{"type":"edge","id":"e{i}","labels":{labels},"from":"v{i}","to":"v{to}","properties":{{}}}}"#
                ));
            }
            lines.join("\n")
        };
        let hop = "MATCH (p:P)-[:KNOWS]->() RETURN count(*) AS c";
        println!(
            "{:22} {:>12} {:>12} {:>10}",
            "edges", "rss_MiB", "bytes/edge", "1hop_us"
        );
        for (name, labels) in [
            ("single [KNOWS]", r#"["KNOWS"]"#),
            ("double [KNOWS,X]", r#"["KNOWS","X"]"#),
        ] {
            let text = build(labels);
            let before = rss_bytes().unwrap_or(0);
            let store = from_ndjson(&text).expect("decodes");
            drop(text);
            let after = rss_bytes().unwrap_or(before);
            let used = after.saturating_sub(before) as f64;
            let plan =
                lenke_engine::opt::optimize_indexed(lenke_engine::gql::parse(hop).unwrap(), &store);
            let us = best_ms(cfg.reps, || lenke_engine::exec::run(&plan, &store)) * 1e3;
            println!(
                "{name:22} {:>12.1} {:>12.1} {us:>10.1}",
                used / (1024.0 * 1024.0),
                used / f64::from(n as u32)
            );
            std::hint::black_box(&store);
        }
    }
}
