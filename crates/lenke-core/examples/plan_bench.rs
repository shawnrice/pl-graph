//! **What does it cost to turn query TEXT into a plan?** — lex + parse + lower,
//! with no graph and no execution anywhere near it.
//!
//! Every other benchmark in this directory measures a plan being *run*. That
//! makes plan-time changes invisible: `gql_bench`'s "+50us parse/lower" line is
//! measured against a 2.7 ms execute, so a pass that doubled lowering would move
//! that row by under 2% and read as noise.
//!
//! This one exists because the predicate-normalization work in
//! `docs/design/query-ir.md` adds a rewrite pass to the lower step. That pass is
//! meant to be free — it runs once per plan, and prepared plans are cached — but
//! "meant to be" is not a measurement, and the pathological shapes for a tree
//! rewrite (a wide conjunction, a long `IN` list, a deep `OR`) are exactly the
//! ones a hand-written recogniser walks cheaply today.
//!
//! Three columns:
//!   parse   lex + parse to AST          — a normalization pass must not move this
//!   prepare lex + parse + LOWER         — where such a pass lands
//!   lower   prepare - parse             — the number to watch
//!
//! Run: `cargo run --release --example plan_bench`

use std::time::Instant;

use lenke_core::gql::{parse, prepare};

/// Best-of-`reps` nanoseconds for one call of `f`, averaged over `iters`.
fn best_ns(reps: u32, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..iters {
        f(); // warm
    }

    let mut best = f64::MAX;

    for _ in 0..reps {
        let clock = Instant::now();

        for _ in 0..iters {
            f();
        }

        let ns = clock.elapsed().as_secs_f64() * 1e9 / f64::from(iters);

        if ns < best {
            best = ns;
        }
    }

    best
}

fn main() {
    // Long-form shapes are built rather than written out; `leak` keeps them
    // `&'static str` so they sit in the same table as the literals.
    let wide_and: &str = Box::leak(
        (0..32)
            .map(|i| format!("u.k{i} = {i}"))
            .collect::<Vec<_>>()
            .join(" AND ")
            .pipe(|w| format!("MATCH (u:P) WHERE {w} RETURN count(*) AS c"))
            .into_boxed_str(),
    );
    let deep_or: &str = Box::leak(
        (0..32)
            .map(|i| format!("u.k = 'v{i}'"))
            .collect::<Vec<_>>()
            .join(" OR ")
            .pipe(|w| format!("MATCH (u:P) WHERE {w} RETURN count(*) AS c"))
            .into_boxed_str(),
    );
    let long_in: &str = Box::leak(
        (0..256)
            .map(|i| format!("'v{i}'"))
            .collect::<Vec<_>>()
            .join(", ")
            .pipe(|l| format!("MATCH (u:P) WHERE u.k IN [{l}] RETURN count(*) AS c"))
            .into_boxed_str(),
    );

    let cases: &[(&str, &str)] = &[
        // The spellings the seek recogniser has to canonicalize. These are the
        // ones a normalization pass touches on every plan.
        ("point equality", "MATCH (u:P) WHERE u.k = $x RETURN count(*) AS c"),
        ("reversed operands", "MATCH (u:P) WHERE $x = u.k RETURN count(*) AS c"),
        ("inline property", "MATCH (u:P {k: $x}) RETURN count(*) AS c"),
        ("IN list of 2", "MATCH (u:P) WHERE u.k IN [$a, $b] RETURN count(*) AS c"),
        ("OR of equalities", "MATCH (u:P) WHERE u.k = $a OR u.k = $b RETURN count(*) AS c"),
        ("range pair", "MATCH (u:P) WHERE u.n >= 5 AND u.n <= 9 RETURN count(*) AS c"),
        ("reversed range pair", "MATCH (u:P) WHERE 5 <= u.n AND 9 >= u.n RETURN count(*) AS c"),
        ("dotted path", "MATCH (u:P) WHERE u.m.city = $c RETURN count(*) AS c"),
        // Ordinary shapes, to prove a plan-time pass does not tax queries that
        // have nothing for it to rewrite.
        ("bare label scan", "MATCH (u:P) RETURN count(*) AS c"),
        ("one-hop traversal", "MATCH (u:P)-[:R]->(x) WHERE u.k = $x RETURN count(*) AS c"),
        ("three-hop traversal", "MATCH (a:P)-[:R]->(b)-[:R]->(c)-[:R]->(d) RETURN count(*) AS n"),
        ("var-length", "MATCH (u:P)-[:R]->{1,3}(x) WHERE u.k = $x RETURN count(*) AS c"),
        (
            "group + aggregate",
            "MATCH (u:P) RETURN u.n AS n, count(*) AS c, sum(u.w) AS s GROUP BY n ORDER BY c DESC LIMIT 20",
        ),
        (
            "multi-clause",
            "MATCH (u:P) WHERE u.k = $x WITH u, u.n AS n WHERE n > 5 MATCH (u)-[:R]->(v) RETURN v.k AS k",
        ),
        // Where a tree rewrite gets expensive, if it is going to.
        ("wide AND (32 terms)", wide_and),
        ("deep OR (32 terms)", deep_or),
        ("IN list of 256", long_in),
    ];

    println!("GQL plan-time cost — no graph, no execution.\n");
    println!(
        "{:<28}{:>12}{:>12}{:>12}{:>10}",
        "shape", "parse ns", "prepare ns", "lower ns", "lower %"
    );
    println!("{}", "-".repeat(74));

    for (name, q) in cases {
        // Fail loudly rather than reporting the cost of an error path.
        assert!(prepare(q).is_ok(), "`{name}` does not prepare: {q}");

        // Fewer iterations for the shapes that are 100x the others, so the whole
        // run stays interactive.
        let iters = if q.len() > 400 { 2_000 } else { 20_000 };
        let parse_ns = best_ns(5, iters, || {
            std::hint::black_box(parse(q).ok());
        });
        let prep_ns = best_ns(5, iters, || {
            std::hint::black_box(prepare(q).ok());
        });
        // Clamped at zero: the two are separate measurements and on the cheapest
        // shapes their noise floors overlap.
        let lower_ns = (prep_ns - parse_ns).max(0.0);
        let pct = if prep_ns > 0.0 {
            lower_ns / prep_ns * 100.0
        } else {
            0.0
        };

        println!("{name:<28}{parse_ns:>12.0}{prep_ns:>12.0}{lower_ns:>12.0}{pct:>9.0}%");
    }

    // Gremlin has no separate lower step — `parse` yields the step vector the
    // executor walks directly, and its index recogniser runs per EXECUTION, not
    // per plan. That difference is the whole reason the shared access-path layer
    // needs measuring on both sides: what is a once-per-plan cost for GQL is a
    // once-per-run cost for Gremlin.
    let gremlin: &[(&str, &str)] = &[
        ("point equality", "g.V().has('k', 'v5').count()"),
        ("label then has", "g.V().hasLabel('P').has('k', 'v5').count()"),
        ("within of 2", "g.V().has('k', within('v5', 'v9')).count()"),
        ("range", "g.V().has('n', gt(5)).count()"),
        ("has then traverse", "g.V().has('k', 'v5').out('R').values('k')"),
        (
            "ten steps",
            "g.V().hasLabel('P').has('k', 'v5').out('R').in('R').dedup().order().by('n').limit(20).values('k')",
        ),
    ];

    println!("\nGremlin parse cost — no lower step; the seek recogniser runs per execution.\n");
    println!("{:<28}{:>12}", "traversal", "parse ns");
    println!("{}", "-".repeat(40));

    for (name, t) in gremlin {
        assert!(
            lenke_core::gremlin::parse(t).is_ok(),
            "`{name}` does not parse: {t}"
        );

        let ns = best_ns(5, 20_000, || {
            std::hint::black_box(lenke_core::gremlin::parse(t).ok());
        });

        println!("{name:<28}{ns:>12.0}");
    }
}

/// `x.pipe(f)` — reads better than nesting `format!` calls three deep.
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}
