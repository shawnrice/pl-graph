//! Benchmark: as-of interval-overlap over a node's time-versioned edges.
//!
//! The question G4 (edge interval index) must answer measured-first: how much does
//! an "as of T" query pay for expanding ALL of a node's edges and post-filtering by
//! `vf <= T AND vt >= T`? This is the bitemporal shape — a high-degree node whose
//! edges each carry a validity interval, asked "which held at time T". The cost
//! scales with degree; an interval index could seek only the overlapping edges.
//!
//! Native only. Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example interval_bench
//!
//! Prints, per (nodes, degree), the min-of-`REPS` wall time for the as-of count:
//! the boxed-post-filter scan vs. the interval-index seek, plus the build cost.
//!
//! MEASURED (min of 7, release; keep these next to the code, per CLAUDE.md):
//! ```text
//!  nodes  degree     scan_us     seek_us  speedup   build_us
//!  20000       8     29996.0       155.6   192.79      32994
//!  20000      64    364022.9      2841.9   128.09     347092
//!  20000     512   4151669.4     18349.2   226.26    4267859
//! 100000      64   2192915.6     22714.1    96.54    2395323
//! ```
//! The seek is 96–226× faster: it stores the intervals INLINE (no boxed edge-prop
//! probe) and seeks only the overlapping edges from the selective axis. Unlike the
//! edge-type index (G5), this is a large unambiguous win — because the baseline
//! cost was the boxed post-filter, not the adjacency scan. Build cost (one pass
//! reading the boxed props) is a one-time amortized expense. The seek is timed in
//! isolation here; wiring it into the query planner transparently is G4c.

use lenke_engine::store::{Builder, Store};
use lenke_engine::value::Value;
use std::time::Instant;

/// `nodes` Emp nodes, each with `degree` HELD edges to a shared pool of role
/// nodes, each edge carrying a validity interval `[vf, vt]` tiled along a timeline
/// of length `SPAN`. An "as of T" query counts edges whose interval covers T; with
/// tiled intervals of width `W`, about `W` of each node's `degree` edges overlap a
/// given point — the selective slice an index would seek.
fn fixture(nodes: u32, degree: u32) -> Store {
    const ROLES: u32 = 16;
    const W: i64 = 4; // interval width (in timeline steps)
    let mut b = Builder::default();
    for _ in 0..ROLES {
        b.node(&["Role"], &[]);
    }
    let emp0 = ROLES; // first Emp id
    for _ in 0..nodes {
        b.node(&["Emp"], &[]);
    }
    let mut st = b.build();
    for e in 0..nodes {
        let emp = emp0 + e;
        for d in 0..degree {
            let role = d % ROLES;
            let eid = st.add_edge(emp, role, "HELD");
            let vf = i64::from(d);
            st.set_edge_prop(eid, "vf", Value::Num(vf as f64));
            st.set_edge_prop(eid, "vt", Value::Num((vf + W) as f64));
        }
    }
    st
}

fn time_asof(store: &Store, t: i64, reps: usize) -> f64 {
    let q = format!(
        "MATCH (p:Emp)-[r:HELD]->() WHERE r.vf <= {t} AND r.vt >= {t} RETURN count(*) AS c"
    );
    let plan = lenke_engine::gql::parse(&q).unwrap();
    let mut best = f64::INFINITY;
    let mut sink = 0.0;
    for _ in 0..reps {
        let t0 = Instant::now();
        let rows = lenke_engine::exec::run(&plan, store);
        let us = t0.elapsed().as_secs_f64() * 1e6;
        if let Value::Num(c) = rows.rows[0][0] {
            sink += c;
        }
        best = best.min(us);
    }
    std::hint::black_box(sink);
    best
}

/// The same as-of count via the interval index seek (in isolation — G4b ships the
/// index + seek; wiring it into the query planner transparently is G4c). Counts,
/// over every Emp node, the out-edges whose `[vf, vt]` covers the point `t`.
fn time_asof_indexed(store: &Store, first_emp: u32, n_emp: u32, t: i64, reps: usize) -> f64 {
    let q = t as f64;
    let mut best = f64::INFINITY;
    let mut sink = 0u64;
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut c = 0u64;
        for e in 0..n_emp {
            store.for_each_overlap(first_emp + e, q, q, |_, _| c += 1);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6;
        sink += c;
        best = best.min(us);
    }
    std::hint::black_box(sink);
    best
}

fn main() {
    const REPS: usize = 7;
    const ROLES: u32 = 16; // must match `fixture`
    println!(
        "{:>8} {:>7} {:>11} {:>11} {:>8} {:>10}",
        "nodes", "degree", "scan_us", "seek_us", "speedup", "build_us"
    );
    for &(nodes, degree) in &[
        (20_000u32, 8u32),
        (20_000, 64),
        (20_000, 512),
        (100_000, 64),
    ] {
        // Query as of the timeline midpoint (a representative selective point).
        let t = i64::from(degree) / 2;
        let store = fixture(nodes, degree);
        let scan_us = time_asof(&store, t, REPS);
        // Indexed seek (isolated): build the interval index, then time the seek.
        let mut indexed = fixture(nodes, degree);
        let t0 = Instant::now();
        indexed.create_interval_index("vf", "vt");
        let build_us = t0.elapsed().as_secs_f64() * 1e6;
        let seek_us = time_asof_indexed(&indexed, ROLES, nodes, t, REPS);
        println!(
            "{nodes:>8} {degree:>7} {scan_us:>11.1} {seek_us:>11.1} {:>8.2} {build_us:>10.1}",
            scan_us / seek_us
        );
    }
}
