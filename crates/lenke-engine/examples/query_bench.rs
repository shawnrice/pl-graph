//! Query-shape benchmarks — per-shape GQL and Gremlin cost, which counts shortcut
//! vs enumerate, per-row materialization cost, plan (text->plan) cost, and indexed
//! seek vs scan. See the group module for the per-case questions.
//!
//! Native only (`std::time::Instant`). Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example query_bench [-- <case-filter>]
//! Env: BENCH_REPS (samples, default 7), BENCH_N (fixture size, capped at 200k).

// The harness is a shared toolkit; a single binary uses only part of it.
#[path = "support/harness.rs"]
#[allow(dead_code)]
mod harness;
#[path = "support/query.rs"]
mod query;

fn main() {
    query::run(&harness::Cfg::from_env());
}
