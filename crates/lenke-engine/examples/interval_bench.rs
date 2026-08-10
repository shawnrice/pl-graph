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
//! Prints, per (nodes, degree), the min-of-`REPS` wall time for the as-of count.
//! Compare before/after an interval index (G4b).

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

fn main() {
    const REPS: usize = 7;
    println!(
        "{:>8} {:>7} {:>12} {:>14}",
        "nodes", "degree", "asof_us", "us_per_node"
    );
    for &(nodes, degree) in &[
        (20_000u32, 8u32),
        (20_000, 64),
        (20_000, 512),
        (100_000, 64),
    ] {
        let store = fixture(nodes, degree);
        // Query as of the timeline midpoint (a representative selective point).
        let t = i64::from(degree) / 2;
        let us = time_asof(&store, t, REPS);
        println!(
            "{nodes:>8} {degree:>7} {us:>12.1} {:>14.4}",
            us / f64::from(nodes)
        );
    }
}
