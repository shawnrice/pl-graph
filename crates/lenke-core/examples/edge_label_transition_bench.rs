//! **What does one multi-label edge cost the whole graph?**
//!
//! Edges carry several labels, with the first dense (mirrored into every
//! adjacency entry) and the rest in a sparse side table. `edge_has_label`
//! short-circuits on that table being empty, so a graph with no multi-label edge
//! filters exactly as it did before edges became multi-label — but a SINGLE
//! multi-label edge anywhere disarms that check for every traversal in the graph.
//!
//! This measures the crossing, in both directions, which is the part a
//! correctness test cannot see: leaving a stale empty entry behind on removal
//! answers correctly and silently keeps the slow path forever.
//!
//! Run: `cargo run --release --example edge_label_transition_bench`

use std::time::Instant;

use lenke_core::graph::{Builder, EdgeRec, Graph, NodeRec};

fn build(n: usize) -> Graph {
    let mut b = Builder::default();

    for i in 0..n {
        b.nodes
            .push(NodeRec::owned(format!("v{i}"), vec!["V".into()], vec![]));
    }

    for i in 0..n {
        for (d, lbl) in [(1, "A"), (2, "B")] {
            b.edges.push(EdgeRec::owned(
                format!("v{i}"),
                format!("v{}", (i + d) % n),
                lbl.to_string(),
                vec![],
                None,
            ));
        }
    }

    b.finalize()
}

/// Best-of-7 milliseconds for a label-filtered traversal over the whole graph.
fn walk(g: &mut Graph) -> f64 {
    let mut best = f64::MAX;

    for _ in 0..7 {
        let clock = Instant::now();
        let n = lenke_core::gremlin::g()
            .V()
            .out(&["A"])
            .count()
            .run(g)
            .len();

        std::hint::black_box(n);
        best = best.min(clock.elapsed().as_secs_f64() * 1e3);
    }

    best
}

fn main() {
    const N: usize = 50_000;

    let mut g = build(N);

    println!("{N} vertices, {} edges\n", g.edge_count());
    println!("{:<38} {:>10}", "state", "out('A')");
    println!("{}", "-".repeat(50));

    let single = walk(&mut g);

    println!("{:<38} {single:>8.2} ms", "all edges single-label");

    // One edge gains a second label: the fast path disarms for the WHOLE graph.
    g.add_edge_label(0, "B");
    assert!(g.has_multi_label_edges());

    let one_multi = walk(&mut g);

    println!(
        "{:<38} {one_multi:>8.2} ms   {:.2}x",
        "…one edge made multi-label",
        one_multi / single
    );

    // Every edge multi-label — the worst case, to show it scales with the
    // number of multi-label edges and not with something worse.
    for e in 0..g.edge_count() as u32 {
        g.add_edge_label(e, "B");
    }

    let all_multi = walk(&mut g);

    println!(
        "{:<38} {all_multi:>8.2} ms   {:.2}x",
        "…every edge multi-label",
        all_multi / single
    );

    // …and back down. Removing the last extra must RE-ARM the fast path, not
    // leave an empty entry that keeps it disarmed for good.
    for e in 0..g.edge_count() as u32 {
        g.remove_edge_label(e, "B");
    }

    assert!(
        !g.has_multi_label_edges(),
        "fast path did not re-arm — a stale empty entry in `e_extra`"
    );

    let back = walk(&mut g);

    println!(
        "{:<38} {back:>8.2} ms   {:.2}x",
        "…all back to single-label",
        back / single
    );
}
