//! Ingest & footprint benchmarks — where NDJSON decode time goes, the distance
//! from the machine's raw-scan ceiling, the parallel decode/encode speedup, and
//! resident bytes per element. See the group module for the per-case questions.
//!
//! Native only (`std::time::Instant`). Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example ingest_bench [-- <case-filter>]
//! Env: BENCH_REPS (samples, default 7), BENCH_N (sweep size, default 200k).

// The harness is a shared toolkit; a single binary uses only part of it.
#[path = "support/harness.rs"]
#[allow(dead_code)]
mod harness;
#[path = "support/ingest.rs"]
mod ingest;

fn main() {
    ingest::run(&harness::Cfg::from_env());
}
