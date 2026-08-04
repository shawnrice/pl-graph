//! Does a `MATCH` that FOLLOWS a `WITH` cost more than the same query written
//! without one?
//!
//! Carrying a value through a `WITH` — instead of reading it off the element in
//! place — used to drop everything after the `WITH` to the scalar driver, which
//! measured 4.8x. That is long fixed: `vectorized_linear` keeps the whole
//! `MATCH … WITH … MATCH … RETURN` shape columnar, calling the same `build_scan`
//! / `with_frame` / `expand_frame` / `project_frame_cols` the single-clause path
//! does. Each pair here is two spellings of ONE question, so the answer must
//! match and the times should too — the `equivalent_spellings_cost_the_same`
//! idea applied to clause composition rather than to predicates.
//!
//! **What the remaining gap is.** Not the barrier. Measured on the two pairs
//! below, which differ only in what the `WITH` carries:
//!
//! ```text
//!   plain, no WITH                       1.756ms
//!   WITH a          (element only)       1.761ms   <- the barrier is FREE
//!   WITH a, a.n AS m (computed value)    3.095ms   <- 1.76x
//! ```
//!
//! A carried ELEMENT is a `Vec<u32>` column the expansion replicates by index. A
//! carried COMPUTED value is evaluated into a materialized `Val` column over
//! every row of the intermediate frame, then replicated through the expansion —
//! here 50k evaluations fanned out to 150k `Val`s, where the plain spelling reads
//! `a.n` off the element column once, at the end.
//!
//! So the lever is not the clause boundary; it is carrying `a.n AS m` lazily (as
//! the expression it is, over an element already in the frame) instead of
//! materializing it eagerly.
//!
//! Run: `cargo run --release --example with_carry_bench`

use std::time::Instant;

use lenke_core::gql::eval::Params;
use lenke_core::gql::prepare;
use lenke_core::graph::Graph;

const N: usize = 50_000;
const DEGREE: usize = 3;
const REPS: usize = 7;

/// `N` vertices labelled `V` with a numeric `n`, each with `DEGREE` out-edges of
/// type `R` to pseudo-random targets. Edges get ids: `encode` emits them, so
/// every reloaded snapshot has them, and omitting them skips the external-id
/// path entirely.
fn fixture() -> Graph {
    let mut lines = String::with_capacity(N * 96);

    for i in 0..N {
        lines.push_str(&format!(
            r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{{"n":{}}}}}"#,
            (i * 2_654_435_761) % 1_000
        ));
        lines.push('\n');
    }

    let mut e = 0usize;

    for i in 0..N {
        for d in 0..DEGREE {
            let to = (i * 31 + d * 7 + 1) % N;
            lines.push_str(&format!(
                r#"{{"type":"edge","id":"e{e}","from":"n{i}","to":"n{to}","labels":["R"],"properties":{{"w":1.0}}}}"#
            ));
            lines.push('\n');
            e += 1;
        }
    }

    lenke_core::ndjson::decode(&lines).expect("fixture decodes")
}

/// Min over `REPS`, plus the row count and column count — a shape that emits
/// zero-column rows looks fast for the wrong reason, and a benchmark that
/// printed only `nrows` once hid exactly that.
fn run(g: &mut Graph, q: &str) -> (f64, usize, usize) {
    let plan = prepare(q).expect("query plans");
    let p = Params::new();
    let warm = plan.execute(g, &p).expect("query runs");
    let (rows, cols) = (warm.nrows, warm.ncols());
    let mut best = f64::INFINITY;

    for _ in 0..REPS {
        let t = Instant::now();
        let _ = plan.execute(g, &p).expect("query runs");
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }

    (best, rows, cols)
}

fn main() {
    let mut g = fixture();

    // Each pair: the same question with and without a carrying `WITH`.
    let pairs: &[(&str, &str, &str)] = &[
        (
            "count, compare to a carried scalar",
            "MATCH (a:V)-[:R]->(b) WHERE b.n > a.n RETURN count(*) AS c",
            "MATCH (a:V) WITH a, a.n AS m MATCH (a)-[:R]->(b) WHERE b.n > m RETURN count(*) AS c",
        ),
        (
            "rows, compare to a carried scalar",
            "MATCH (a:V)-[:R]->(b) WHERE b.n > a.n RETURN b.n AS x",
            "MATCH (a:V) WITH a, a.n AS m MATCH (a)-[:R]->(b) WHERE b.n > m RETURN b.n AS x",
        ),
        (
            // The discriminating pair: same query, same WHERE, the only
            // difference being whether the `WITH` carries a COMPUTED value or
            // just the element. If the barrier itself cost anything, this would
            // be slower than the plain spelling. It is not — see below.
            "carry the element, read the prop late",
            "MATCH (a:V)-[:R]->(b) WHERE b.n > a.n RETURN count(*) AS c",
            "MATCH (a:V) WITH a MATCH (a)-[:R]->(b) WHERE b.n > a.n RETURN count(*) AS c",
        ),
        (
            // NOTE: the plain side here takes a count SHORTCUT (no WHERE), so
            // this row compares "shortcut" against "no shortcut", not the cost of
            // a `WITH`. Kept because that contrast is worth seeing, but do not
            // read its ratio as WITH overhead.
            "carry the element (plain side shortcuts)",
            "MATCH (a:V)-[:R]->(b) RETURN count(*) AS c",
            "MATCH (a:V) WITH a MATCH (a)-[:R]->(b) RETURN count(*) AS c",
        ),
        (
            "carry through a filtering WITH",
            "MATCH (a:V)-[:R]->(b) WHERE a.n > 500 RETURN count(*) AS c",
            "MATCH (a:V) WHERE a.n > 500 WITH a MATCH (a)-[:R]->(b) RETURN count(*) AS c",
        ),
    ];

    println!(
        "{N} vertices, degree {DEGREE}, min of {REPS}\n\n{:<38} {:>9} {:>9} {:>8}  rows/cols",
        "shape", "plain", "with WITH", "ratio"
    );

    for (name, plain, withed) in pairs {
        // Interleaved: run both in the same loop so machine drift hits each
        // equally. A non-interleaved A/B produced a phantom 5% here before.
        let (a, ra, ca) = run(&mut g, plain);
        let (b, rb, cb) = run(&mut g, withed);

        assert_eq!(
            (ra, ca),
            (rb, cb),
            "`{name}`: the two spellings disagree — {ra} rows x {ca} cols vs {rb} x {cb}"
        );

        println!(
            "{name:<38} {a:>8.3}ms {b:>8.3}ms {:>7.2}x  {ra}x{ca}",
            b / a
        );
    }
}
