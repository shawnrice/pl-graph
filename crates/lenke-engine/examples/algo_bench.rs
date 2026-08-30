//! Graph-algorithm benchmarks — per-algorithm cost, the multicore speedup, the
//! neighborAggregate message-passing primitive, and CALL overhead. See the group
//! module for the per-case questions.
//!
//! Native only (`std::time::Instant`). Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example algo_bench [-- <case-filter>]
//! Env: BENCH_REPS (samples, default 7), BENCH_N (fixture size, capped per case).

// The harness is a shared toolkit; a single binary uses only part of it.
#[path = "support/algo.rs"]
mod algo;
#[path = "support/harness.rs"]
#[allow(dead_code)]
mod harness;

fn main() {
    algo::run(&harness::Cfg::from_env());
}
