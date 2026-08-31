//! value benchmarks — see the group module for the per-case questions.
//!
//! Native only (`std::time::Instant`). Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example value_bench [-- <case-filter>]
//! Env: BENCH_REPS (samples, default 7), BENCH_N (fixture size, capped per case).

// The harness is a shared toolkit; a single binary uses only part of it.
#[path = "support/harness.rs"]
#[allow(dead_code)]
mod harness;
#[path = "support/value.rs"]
mod value;

fn main() {
    value::run(&harness::Cfg::from_env());
}
