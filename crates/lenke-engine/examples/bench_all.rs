//! Run the WHOLE bench corpus in one process — the regression sweep.
//!
//! The per-group binaries (`ingest_bench`, `query_bench`, …) are for iterating on
//! one subsystem; this runs every group so an unrelated regression shows up. Each
//! group prints its own `=== group ===` section. A filter argument still applies,
//! matched against every case's `group/case` label, so `bench_all -- index` runs
//! just the index group across the whole corpus in one go.
//!
//! Native only. Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example bench_all [-- <case-filter>]
//! Env: BENCH_REPS (samples, default 7), BENCH_N (sweep size, default 200k).

// The harness is a shared toolkit; even bench_all may not use every helper.
#[path = "support/algo.rs"]
mod algo;
#[path = "support/harness.rs"]
#[allow(dead_code)]
mod harness;
#[path = "support/ingest.rs"]
mod ingest;
#[path = "support/query.rs"]
mod query;

fn main() {
    let cfg = harness::Cfg::from_env();
    // Groups are added here as each is built; every group honours `cfg.filter`.
    ingest::run(&cfg);
    query::run(&cfg);
    algo::run(&cfg);
}
