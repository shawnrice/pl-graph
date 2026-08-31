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
#[path = "support/scale.rs"]
mod scale;
#[path = "support/storage.rs"]
mod storage;
#[path = "support/value.rs"]
mod value;

fn main() {
    let cfg = harness::Cfg::from_env();
    // Every group honours `cfg.filter`, so `bench_all -- <case>` narrows the sweep.
    ingest::run(&cfg);
    query::run(&cfg);
    storage::run(&cfg);
    value::run(&cfg);
    algo::run(&cfg);
    scale::run(&cfg);
}
