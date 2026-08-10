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
//! all edges of ONE type via `MATCH (n:V)-[:T0]->() RETURN count(*)`, for the
//! scan baseline vs. the opt-in edge-type index, plus the index build cost.
//!
//! MEASURED (min of 7, release; keep these numbers next to the code, per
//! CLAUDE.md — this is where a wrong conclusion about the index would be drawn):
//! ```text
//!  nodes  degree  types   scan_us  index_us  speedup   build_us
//!  50000       4      1     202.9     625.4     0.32     20505   ← index LOSES
//!  50000      32      1    2457.0    1063.6     2.31     65553
//!  50000      32      8    1485.7     842.5     1.76     85851
//!  50000     256      1   10415.9    7025.7     1.48    366070
//!  50000     256      8    8732.6    1074.1     8.13    471868   ← index WINS big
//!  50000     256     64    8776.4    1341.1     6.54    639465   ← index WINS big
//! 200000      32      1    9856.6   11375.2     0.87    233458   ← index LOSES
//! 200000      32      8   10134.3   15596.6     0.65    277717   ← index LOSES
//! ```
//! Conclusion: the index wins (6–8×) ONLY in the high-degree × many-type ×
//! selective-query regime at cache-resident sizes. At degree 4 it loses (a 4-entry
//! scan beats a hashmap probe), and at 200k nodes it loses even at degree 32 — the
//! per-node `HashMap` chases scattered heap while the flat adjacency scans
//! contiguously (the classic cache-transition effect CLAUDE.md warns of). This is
//! precisely why the index is OPT-IN (`create_edge_type_index`): a graph that does
//! not fit the winning regime simply never creates it and pays nothing. A CSR /
//! sorted-by-type adjacency (contiguous, no per-node hashmap) is the future
//! representation that would make it a broad win; deferred until a workload needs it.

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
        "{:>8} {:>7} {:>6} {:>11} {:>11} {:>8} {:>10}",
        "nodes", "degree", "types", "scan_us", "index_us", "speedup", "build_us"
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
            // Baseline: full-adjacency scan-and-filter.
            let store = fixture(nodes, degree, n_types);
            let scan_us = time_count(&store, REPS);
            // Indexed: build the opt-in edge-type index, then re-time. `build_us`
            // is the one-time cost of `create_edge_type_index` over the fixture.
            let mut indexed = fixture(nodes, degree, n_types);
            let t = Instant::now();
            indexed.create_edge_type_index();
            let build_us = t.elapsed().as_secs_f64() * 1e6;
            let index_us = time_count(&indexed, REPS);
            println!(
                "{nodes:>8} {degree:>7} {n_types:>6} {scan_us:>11.1} {index_us:>11.1} {:>8.2} {build_us:>10.1}",
                scan_us / index_us
            );
        }
    }
}
