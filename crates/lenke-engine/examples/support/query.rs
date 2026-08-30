//! Query-shape cost — what individual GQL and Gremlin shapes pay, which counts
//! shortcut vs enumerate, what a row costs by what is returned, and what turning
//! text into a plan costs before anything runs. Consolidates the retired
//! `gql_bench`, `gremlin_bench`, `exists_probe`, `query_row_cost`, `plan_bench`
//! and `seeded_traversal_bench`.
//!
//! Cases (filter with `-- <name>`):
//!   gql      — label scan / filter / project / 1-hop / 2-hop / group / EXISTS.
//!   gremlin  — count / has-filter / out / out.out / values / dedup.
//!   counts   — count(*) over each shape: which still enumerate since the
//!              shortcut ladder (a fast one is a tally, a slow one walks rows).
//!   perrow   — RETURN count vs a scalar vs a string vs the whole element: the
//!              per-row materialization cost by what is projected.
//!   plan     — lex+parse+lower only, no graph, no exec: the cost of the text.
//!   seeded   — an equality filter on an indexed key: seek vs full scan.

use crate::harness::{best_us, section, social_store, Cfg};
use lenke_engine::store::Store;

/// Time a GQL or Gremlin query the way the FFI runs one: parse, then the
/// store-aware `optimize_indexed` (which is where an equality/range seek is
/// chosen against the store's indexes), then `exec::run`. Optimize is one-time,
/// so it sits OUTSIDE the timing loop — the number is the per-run exec cost.
/// Returns `(best_us, rows)` or a parse error — a bad shape prints `n/a` instead
/// of aborting the whole group.
fn time_query(q: &str, gremlin: bool, store: &Store, reps: usize) -> Result<(f64, usize), String> {
    let raw = if gremlin {
        lenke_engine::gremlin::parse(q)?
    } else {
        lenke_engine::gql::parse(q)?
    };
    let plan = lenke_engine::opt::optimize_indexed(raw, store);
    let mut rows = 0;
    let us = best_us(reps, || {
        let r = lenke_engine::exec::run(&plan, store);
        rows = r.rows.len();
        r
    });
    Ok((us, rows))
}

fn table(title: &str, store: &Store, gremlin: bool, cfg: &Cfg, shapes: &[(&str, &str)]) {
    section(title);
    println!("{:22} {:>11} {:>10}", "shape", "best_us", "rows");
    for (name, q) in shapes {
        match time_query(q, gremlin, store, cfg.reps) {
            Ok((us, rows)) => println!("{name:22} {us:>11.1} {rows:>10}"),
            Err(e) => println!("{name:22} {:>11} {:>10}  ({})", "n/a", "-", e.trim()),
        }
    }
}

pub fn run(cfg: &Cfg) {
    // Cap the fixture: 2-hop over deg-5 is quadratic in degree, so a huge graph
    // makes the group slow without changing the shape being measured.
    let nodes = cfg.scale.unwrap_or(200_000).min(200_000) as u32;
    let store = social_store(nodes, 5);

    if cfg.want("query/gql") {
        table(
            "query/gql",
            &store,
            false,
            cfg,
            &[
                ("label scan", "MATCH (p:Person) RETURN count(*) AS c"),
                (
                    "filter age>50",
                    "MATCH (p:Person) WHERE p.age > 50 RETURN count(*) AS c",
                ),
                ("project 3", "MATCH (p:Person) RETURN p.name, p.age, p.city"),
                (
                    "1-hop",
                    "MATCH (p:Person)-[:KNOWS]->(q) RETURN count(*) AS c",
                ),
                (
                    "2-hop",
                    "MATCH (p:Person)-[:KNOWS]->()-[:KNOWS]->(r) RETURN count(*) AS c",
                ),
                (
                    "group by dept",
                    "MATCH (p:Person) RETURN p.dept, count(*) AS c",
                ),
                (
                    "EXISTS",
                    "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->() } RETURN count(*) AS c",
                ),
            ],
        );
    }

    if cfg.want("query/gremlin") {
        table(
            "query/gremlin",
            &store,
            true,
            cfg,
            &[
                ("V count", "g.V().count()"),
                ("has age>50", "g.V().has('age', gt(50)).count()"),
                ("out", "g.V().out('KNOWS').count()"),
                ("out.out", "g.V().out('KNOWS').out('KNOWS').count()"),
                ("values", "g.V().values('name').count()"),
                ("dedup city", "g.V().values('city').dedup().count()"),
            ],
        );
    }

    if cfg.want("query/counts") {
        table(
            "query/counts (shortcut vs enumerate)",
            &store,
            false,
            cfg,
            &[
                ("label", "MATCH (p:Person) RETURN count(*) AS c"),
                (
                    "filtered",
                    "MATCH (p:Person) WHERE p.age > 50 RETURN count(*) AS c",
                ),
                (
                    "1-hop",
                    "MATCH (p:Person)-[:KNOWS]->() RETURN count(*) AS c",
                ),
                (
                    "2-hop",
                    "MATCH (p:Person)-[:KNOWS]->()-[:KNOWS]->() RETURN count(*) AS c",
                ),
                ("grouped", "MATCH (p:Person) RETURN p.dept, count(*) AS c"),
            ],
        );
    }

    if cfg.want("query/perrow") {
        section("query/perrow (per-row by projection)");
        println!(
            "{:22} {:>11} {:>10} {:>11}",
            "return", "best_us", "rows", "ns/row"
        );
        for (name, q) in [
            ("count(*)", "MATCH (p:Person) RETURN count(*) AS c"),
            ("scalar p.age", "MATCH (p:Person) RETURN p.age"),
            ("string p.name", "MATCH (p:Person) RETURN p.name"),
            ("element p", "MATCH (p:Person) RETURN p"),
        ] {
            match time_query(q, false, &store, cfg.reps) {
                Ok((us, rows)) => {
                    let ns = if rows > 0 {
                        us * 1e3 / rows as f64
                    } else {
                        0.0
                    };
                    println!("{name:22} {us:>11.1} {rows:>10} {ns:>11.1}");
                }
                Err(e) => println!("{name:22} {:>11}  ({})", "n/a", e.trim()),
            }
        }
    }

    if cfg.want("query/plan") {
        section("query/plan (text -> plan, no exec)");
        println!("{:30} {:>11}", "text", "best_us");
        for (name, q) in [
            ("label scan", "MATCH (p:Person) RETURN count(*) AS c"),
            (
                "2-hop + filter",
                "MATCH (p:Person)-[:KNOWS]->(q) WHERE p.age > 50 RETURN q.name",
            ),
            (
                "group + order + limit",
                "MATCH (p:Person) RETURN p.dept, count(*) AS c ORDER BY c DESC LIMIT 3",
            ),
            (
                "gremlin 2-hop",
                "g.V().out('KNOWS').out('KNOWS').values('name').dedup()",
            ),
        ] {
            let gremlin = q.starts_with("g.");
            let us = best_us(cfg.reps, || {
                if gremlin {
                    lenke_engine::gremlin::parse(q).map(|_| ())
                } else {
                    lenke_engine::gql::parse(q).map(|_| ())
                }
            });
            println!("{name:30} {us:>11.2}");
        }
    }

    if cfg.want("query/seeded") {
        section("query/seeded (indexed equality: seek vs scan)");
        // Same query, same rows — the only difference is whether the key is
        // indexed, so the optimize step turns the scan into a seek. (An inline
        // literal is a seekable spelling; per the equal-spellings rule it must
        // cost the same as the `$param` form.)
        let q = "MATCH (p:Person) WHERE p.name = 'name12345' RETURN p.age";
        let scan = social_store(nodes, 5);
        let mut indexed = social_store(nodes, 5);
        indexed.create_index("name");
        let scan_us = time_query(q, false, &scan, cfg.reps).map_or(f64::NAN, |(u, _)| u);
        let seek_us = time_query(q, false, &indexed, cfg.reps).map_or(f64::NAN, |(u, _)| u);
        println!(
            "{:22} {:>11} {:>11} {:>8}",
            "p.name = 'name12345'", "scan_us", "seek_us", "speedup"
        );
        println!(
            "{:22} {scan_us:>11.1} {seek_us:>11.1} {:>8.1}",
            "",
            scan_us / seek_us
        );
    }
}
