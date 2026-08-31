//! Scaling & serving cost — how the same shapes scale across the cache
//! transition, and what content-derived CDC scope extraction costs per write.
//! Consolidates the retired `scale_bench` and `cdc_extract_bench`.
//!
//! (The AML / HRIS domain-shaped workloads — `bench_aml_shapes` /
//! `bench_hris_shapes` — are large bespoke fixtures; they are deferred rather
//! than stubbed. See examples/README.md.)
//!
//! Cases (filter with `-- <name>`):
//!   sweep — label-count / 1-hop / filtered across 50k / 200k / 500k nodes, so
//!           the cache transition (200k–1M elements) is visible in one table.
//!   cdc   — the cost of reading a content-derived scope off the last write's
//!           touched elements (`last_write_scope_json`), the per-write CDC tax.

use crate::harness::{best_ms, section, social_store, Cfg};

pub fn run(cfg: &Cfg) {
    if cfg.want("scale/sweep") {
        section("scale/sweep (shapes across size)");
        println!(
            "{:>10} {:>12} {:>12} {:>12}",
            "nodes", "labelcnt_us", "1hop_us", "filter_us"
        );
        let sizes = cfg
            .scale
            .map_or_else(|| vec![50_000u32, 200_000, 500_000], |s| vec![s as u32]);
        let shapes = [
            "MATCH (p:Person) RETURN count(*) AS c",
            "MATCH (p:Person)-[:KNOWS]->() RETURN count(*) AS c",
            "MATCH (p:Person) WHERE p.age > 50 RETURN count(*) AS c",
        ];
        for n in sizes {
            let store = social_store(n, 5);
            let mut us = [0.0f64; 3];
            for (k, q) in shapes.iter().enumerate() {
                let plan = lenke_engine::opt::optimize_indexed(
                    lenke_engine::gql::parse(q).unwrap(),
                    &store,
                );
                us[k] = best_ms(cfg.reps, || lenke_engine::exec::run(&plan, &store)) * 1e3;
            }
            println!("{n:>10} {:>12.1} {:>12.1} {:>12.1}", us[0], us[1], us[2]);
        }
    }

    if cfg.want("scale/cdc") {
        section("scale/cdc (content-derived scope extraction per write)");
        let n = cfg.scale.unwrap_or(200_000).min(200_000) as u32;
        let mut store = social_store(n, 5);
        // A write, so the last-write touched set is populated; then time reading a
        // scope (`city`) off it — the per-write CDC tax the serving path pays.
        let plan = lenke_engine::gql::parse("MATCH (p:Person) WHERE p.age = 42 SET p.age = 43")
            .expect("parses");
        lenke_engine::exec::run_query(plan, &mut store).expect("writes");
        let us = best_ms(cfg.reps, || store.last_write_scope_json("city")) * 1e3;
        println!("{:30} {:>10}", "last_write_scope_json(city)", "us");
        println!("{:30} {us:>10.3}", "");
    }
}
