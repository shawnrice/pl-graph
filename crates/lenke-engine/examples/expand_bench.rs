//! Benchmark: type-filtered expand across degree × edge-type count.
//!
//! The question G5 (edge-type index) must answer BEFORE building anything, per
//! the repo's measure-first rule: how much does a type-filtered `expand` pay for
//! scanning a node's whole adjacency and filtering by edge type? The answer scales
//! with degree and with how many types the degree is spread across — a selective
//! type over a high-degree, many-type node is where an index could win, and a
//! degree-4 single-type fixture is where it can only lose.
//!
//! Native only (uses `std::time::Instant`). Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example expand_bench
//!
//! It prints, per (nodes, degree, n_types), the min-of-`REPS` wall time to count
//! all edges of ONE type via `MATCH (n:V)-[:T0]->() RETURN count(*)`. Compare the
//! same rows before and after an edge-type index lands (G5b).

use lenke_engine::store::{Builder, Store};
use lenke_engine::value::Value;
use std::time::Instant;

/// A tiny deterministic LCG so target selection is reproducible without a dep
/// (Math.random()/rand are both off-limits here — determinism matters for a bench
/// that is compared across builds).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, bound: u32) -> u32 {
        // Numerical Recipes constants.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as u32) % bound
    }
}

/// Build `nodes` V-nodes, each with `degree` out-edges to random targets, the edge
/// types cycled across `n_types` (`T0..T{n_types-1}`). Type `T0` therefore owns
/// about `degree / n_types` of each node's out-edges — the selective slice the
/// query counts.
fn fixture(nodes: u32, degree: u32, n_types: u32) -> Store {
    let mut b = Builder::default();
    for i in 0..nodes {
        b.node(&["V"], &[("n", Value::Num(f64::from(i)))]);
    }
    let types: Vec<String> = (0..n_types).map(|t| format!("T{t}")).collect();
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    for from in 0..nodes {
        for d in 0..degree {
            let to = rng.next(nodes);
            let ty = &types[(d % n_types) as usize];
            b.edge(from, to, ty);
        }
    }
    b.build()
}

fn time_count(store: &Store, reps: usize) -> f64 {
    let plan = lenke_engine::gql::parse("MATCH (n:V)-[:T0]->() RETURN count(*) AS c").unwrap();
    let mut best = f64::INFINITY;
    let mut sink = 0.0;
    for _ in 0..reps {
        let t = Instant::now();
        let rows = lenke_engine::exec::run(&plan, store);
        let us = t.elapsed().as_secs_f64() * 1e6;
        // Touch the result so the run cannot be optimized away.
        if let Value::Num(c) = rows.rows[0][0] {
            sink += c;
        }
        best = best.min(us);
    }
    std::hint::black_box(sink);
    best
}

fn main() {
    const REPS: usize = 7;
    println!(
        "{:>8} {:>7} {:>7} {:>12} {:>14}",
        "nodes", "degree", "types", "min_us", "us_per_node"
    );
    // Sweep the size across the cache transition (per CLAUDE.md: 200k–1M elements),
    // degree (per-edge costs scale with degree), and type spread (an index only
    // helps when a selective type sits inside a high, many-type degree).
    for &(nodes, degree) in &[
        (50_000u32, 4u32),
        (50_000, 32),
        (50_000, 256),
        (200_000, 32),
    ] {
        for &n_types in &[1u32, 8, 64] {
            if n_types > degree {
                continue;
            }
            let store = fixture(nodes, degree, n_types);
            let us = time_count(&store, REPS);
            println!(
                "{nodes:>8} {degree:>7} {n_types:>7} {us:>12.1} {:>14.4}",
                us / f64::from(nodes)
            );
        }
    }
}
