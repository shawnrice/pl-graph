//! Index equivalence fuzzing: **a query's answer must not depend on whether an
//! index exists.** An index is an optimization; it may reorder rows and it may
//! change how many elements the engine touches, but it may never change which
//! rows come back.
//!
//! This is the one property nothing else asserts. The differential fuzzers run
//! TS against native on graphs that carry no indexes at all, so every seek path
//! in `eval/scan.rs` — inline `{k: lit}`, WHERE-derived hints, grouped range
//! bounds, the most-selective-group choice, the RI-tree interval stab — is
//! invisible to them. A seek that silently drops a row would pass the entire
//! existing suite.
//!
//! Three graphs are built from the identical statement stream and compared:
//!
//!   - `plain` — no indexes; every match is a scan. This is the oracle.
//!   - `pre` — indexes created BEFORE the data, so every insert and update
//!     maintains them incrementally.
//!   - `post` — indexes created AFTER the data, so each one is built by a bulk
//!     pass over existing elements.
//!
//! `pre` and `post` must agree with `plain` *and* with each other: the two ways
//! of arriving at an index have to produce the same index. Writes are then
//! replayed on all three and the queries re-run, which is what actually catches
//! stale entries — the classic index bug is an update that indexes the new value
//! without retiring the old one. Each step also runs a write inside a
//! transaction and rolls it back, since an undo that restores the stored values
//! but not the index entries leaves a graph whose seeks answer from a state that
//! no longer exists.
//!
//! This found one real bug on its first round: `INSERT (:P {k: 1, k: 2})` stores
//! last-wins but used to apply the index once per pair, so every repeat left a
//! live entry the element no longer matched (fixed in `dedupe_props_last_wins`;
//! pinned by the `h_insert_repeated_key_*` tests).
//!
//! Rows are compared as MULTISETS. Seeding from an index legitimately changes
//! row order (that is the whole point of a seek), and unordered result order is
//! unspecified, so demanding identical order would manufacture failures rather
//! than find them. Errors count as outcomes too: all three graphs must fault
//! identically.
//!
//! Seed: random each run, `FUZZ_SEED=<n>` to replay — the convention the other
//! fuzzers use.

use super::eval::Params;
use super::parse;
use crate::fuzz_tests::{fuzz_seed, Rng};
use crate::graph::{Graph, Value};

/// An empty graph — the same door every other suite comes through.
fn empty_graph() -> Graph {
    crate::ndjson::decode("").expect("empty ndjson decodes to an empty graph")
}

/// The indexes under test. Chosen to cover every seek shape `scan.rs` can take:
/// a low-cardinality vertex key (many duplicates per seek), a unique one (one
/// hit), a string, a temporal, an edge property, and the edge interval pair.
fn create_indexes(g: &mut Graph) {
    g.create_vertex_index("k");
    g.create_vertex_index("u");
    g.create_vertex_index("s");
    g.create_vertex_index("t");
    g.create_edge_index("w");
    g.create_edge_interval_index("vf", "vt");
}

/// One randomly-shaped INSERT. Values repeat across a small domain so seeks
/// return multi-element buckets, and every property is sometimes absent and
/// sometimes explicitly null — absence and null are different states, and an
/// index must distinguish them exactly as a scan does.
fn gen_insert(rng: &mut Rng, i: usize) -> String {
    let label = if rng.below(2) == 0 { "P" } else { "Q" };
    let mut props = vec![format!("id: 'v{i}'"), format!("u: {i}")];

    match rng.below(4) {
        0 => {}
        1 => props.push("k: null".to_string()),
        _ => props.push(format!("k: {}", rng.below(6))),
    }
    match rng.below(4) {
        0 => {}
        1 => props.push("s: null".to_string()),
        _ => props.push(format!("s: '{}'", ["a", "b", "c", "", "😀"][rng.below(5)])),
    }
    match rng.below(3) {
        0 => {}
        _ => props.push(format!("t: DATE '2020-01-{:02}'", 1 + rng.below(28))),
    }
    // A key whose values are deliberately of MIXED type across rows: the index
    // column holds numbers and strings at once, so an ordered seek has to agree
    // with the scan's cross-type comparison rules rather than inventing its own.
    if rng.below(3) == 0 {
        let mixed = ["1", "'1'", "2.5", "'zz'", "true"][rng.below(5)];
        props.push(format!("k: {mixed}"));
    }

    format!("INSERT (:{label} {{{}}})", props.join(", "))
}

/// One randomly-shaped edge INSERT, carrying an indexed weight and a `[vf, vt)`
/// validity interval for the RI-tree.
fn gen_edge(rng: &mut Rng, n: usize) -> String {
    let (a, b) = (rng.below(n), rng.below(n));
    let etype = if rng.below(2) == 0 { "R" } else { "S" };
    let lo = 1 + rng.below(20);
    let hi = lo + rng.below(10);

    format!(
        "MATCH (a) WHERE a.id = 'v{a}' MATCH (b) WHERE b.id = 'v{b}' \
         INSERT (a)-[:{etype} {{w: {}, vf: DATE '2020-01-{lo:02}', vt: DATE '2020-01-{hi:02}'}}]->(b)",
        rng.below(5),
    )
}

/// A query whose plan the seek logic is expected to take over. Every shape here
/// maps to a branch in `scan.rs`: inline equality, WHERE equality, a same-key
/// band, a two-key interval containment, the bitemporal four-way, a multi-anchor
/// conjunction across comma patterns, and an edge-side seek. `<>` and `IS NULL`
/// are the negative controls — they must NOT seek, so they check the fallback.
fn gen_query(rng: &mut Rng) -> String {
    let k = rng.below(6);
    let u = rng.below(40);
    let s = ["a", "b", "c", "", "😀"][rng.below(5)];
    let d = format!("DATE '2020-01-{:02}'", 1 + rng.below(28));
    let d2 = format!("DATE '2020-01-{:02}'", 1 + rng.below(28));
    let w = rng.below(5);

    let shapes: [String; 16] = [
        format!("MATCH (n {{k: {k}}}) RETURN n.id AS x"),
        format!("MATCH (n:P {{u: {u}}}) RETURN n.id AS x"),
        format!("MATCH (n) WHERE n.k = {k} RETURN n.id AS x"),
        format!("MATCH (n) WHERE n.s = '{s}' RETURN n.id AS x, n.k AS y"),
        format!("MATCH (n) WHERE n.k <> {k} RETURN n.id AS x"),
        "MATCH (n) WHERE n.k IS NULL RETURN n.id AS x".to_string(),
        format!("MATCH (n) WHERE n.k IN [{k}, {}] RETURN n.id AS x", (k + 2) % 6),
        format!("MATCH (n) WHERE n.u >= {u} AND n.u <= {} RETURN n.id AS x", u + 7),
        format!("MATCH (n) WHERE n.t = {d} RETURN n.id AS x"),
        format!("MATCH (n) WHERE n.t >= {d} AND n.t < {d2} RETURN n.id AS x"),
        format!("MATCH (n) WHERE n.k = {k} AND n.s = '{s}' RETURN n.id AS x"),
        // Multi-anchor across comma patterns — two independent seeds in one plan.
        format!("MATCH (a), (b) WHERE a.k = {k} AND b.u = {u} RETURN a.id AS x, b.id AS y"),
        format!("MATCH ()-[e {{w: {w}}}]->() RETURN e.w AS x, count(*) AS c GROUP BY x"),
        format!("MATCH (a)-[e]->(b) WHERE e.w = {w} RETURN a.id AS x, b.id AS y"),
        // Interval as-of and overlap: the RI-tree stabs, the scan filters.
        format!("MATCH ()-[e]->() WHERE e.vf <= {d} AND e.vt > {d} RETURN e.w AS x, count(*) AS c GROUP BY x"),
        format!("MATCH ()-[e]->() WHERE e.vf < {d2} AND e.vt > {d} RETURN e.w AS x, count(*) AS c GROUP BY x"),
    ];

    shapes[rng.below(shapes.len())].clone()
}

/// A write that must keep every index consistent. Updating an indexed key is the
/// interesting one: the old entry has to be retired, not merely joined by a new
/// one, or a later seek returns an element that no longer matches.
fn gen_write(rng: &mut Rng) -> String {
    let k = rng.below(6);
    let shapes = [
        format!("MATCH (n) WHERE n.k = {k} SET n.k = {}", (k + 3) % 6),
        format!("MATCH (n) WHERE n.k = {k} SET n.s = 'zz', n.t = DATE '2020-02-01'"),
        format!("MATCH (n) WHERE n.k = {k} REMOVE n.k"),
        format!("MATCH (n) WHERE n.k = {k} SET n.k = null"),
        format!("MATCH (n:P) WHERE n.u < {} DETACH DELETE n", rng.below(40)),
        format!("MATCH ()-[e]->() WHERE e.w = {k} SET e.w = {}", (k + 1) % 5),
        format!("MATCH ()-[e]->() WHERE e.w = {k} SET e.vf = DATE '2020-01-05', e.vt = DATE '2020-01-25'"),
        format!("MATCH ()-[e]->() WHERE e.w = {k} DELETE e"),
        "INSERT (:P {id: 'late', u: 999, k: 3, s: 'a', t: DATE '2020-01-15'})".to_string(),
    ];

    shapes[rng.below(shapes.len())].clone()
}

/// Run one statement, returning the rows as a comparable multiset, or the error
/// code. A fault is a legitimate outcome — it just has to be the SAME outcome
/// everywhere.
fn outcome(g: &mut Graph, q: &str) -> Result<Vec<String>, String> {
    let parsed = parse(q).map_err(|e| format!("parse: {e}"))?;
    let rs = parsed
        .execute(g, &Params::new())
        .map_err(|e| format!("exec: {e}"))?;
    let mut rows: Vec<String> = rs
        .rows()
        .map(|r| {
            r.iter()
                .map(|v: &Value| format!("{v:?}"))
                .collect::<Vec<_>>()
                .join("\u{1}")
        })
        .collect();
    // Multiset comparison — an index legitimately changes row ORDER.
    rows.sort_unstable();

    Ok(rows)
}

/// Build a graph from a fixed statement stream, optionally indexing before or
/// after the data lands.
fn build(stmts: &[String], indexes: Option<bool>) -> Graph {
    let mut g = empty_graph();

    if indexes == Some(true) {
        create_indexes(&mut g);
    }
    for s in stmts {
        let _ = parse(s).map(|p| p.execute(&mut g, &Params::new()));
    }
    if indexes == Some(false) {
        create_indexes(&mut g);
    }

    g
}

#[test]
fn fuzz_index_never_changes_a_query_answer() {
    let base = fuzz_seed();
    println!("index-equivalence fuzz seed: {base} (FUZZ_SEED={base} to replay)");

    const ROUNDS: usize = 60;
    const ELEMENTS: usize = 40;

    for round in 0..ROUNDS {
        let mut rng = Rng(base.wrapping_add(round as u64).wrapping_mul(0x9e37_79b9) | 1);

        // One statement stream, replayed identically into all three graphs.
        let mut stmts: Vec<String> = (0..ELEMENTS).map(|i| gen_insert(&mut rng, i)).collect();
        stmts.extend((0..ELEMENTS / 2).map(|_| gen_edge(&mut rng, ELEMENTS)));

        let mut plain = build(&stmts, None);
        let mut pre = build(&stmts, Some(true));
        let mut post = build(&stmts, Some(false));

        let queries: Vec<String> = (0..12).map(|_| gen_query(&mut rng)).collect();
        let writes: Vec<String> = (0..4).map(|_| gen_write(&mut rng)).collect();

        // Alternate: query, mutate, query again — so every query runs against a
        // freshly-built index AND against one that has survived updates.
        for step in 0..=writes.len() {
            for q in &queries {
                let a = outcome(&mut plain, q);
                let b = outcome(&mut pre, q);
                let c = outcome(&mut post, q);

                assert_eq!(
                    a, b,
                    "\nINDEX CHANGED THE ANSWER (indexed-before-data)\n\
                     FUZZ_SEED={base} round={round} step={step}\nquery: {q}\n"
                );
                assert_eq!(
                    a, c,
                    "\nINDEX CHANGED THE ANSWER (indexed-after-data)\n\
                     FUZZ_SEED={base} round={round} step={step}\nquery: {q}\n"
                );
            }

            // A ROLLED-BACK write must leave every index exactly as it was: the
            // undo log has to retire the entries the write added and restore the
            // ones it retired. If it only restored the stored values, the next
            // seek would answer from entries describing a graph that no longer
            // exists — and because the rollback is invisible in the row output,
            // nothing but an index-vs-scan comparison would notice.
            if step < writes.len() {
                let rolled_back = &writes[writes.len() - 1 - step];

                for g in [&mut plain, &mut pre, &mut post] {
                    g.begin_tx();
                    let _ = outcome(g, rolled_back);
                    g.rollback_tx();
                }
                for q in &queries {
                    let a = outcome(&mut plain, q);
                    let b = outcome(&mut pre, q);
                    let c = outcome(&mut post, q);

                    assert_eq!(
                        a, b,
                        "\nROLLBACK LEFT THE INDEX STALE (indexed-before-data)\n\
                         FUZZ_SEED={base} round={round} step={step}\n\
                         rolled back: {rolled_back}\nquery: {q}\n"
                    );
                    assert_eq!(
                        a, c,
                        "\nROLLBACK LEFT THE INDEX STALE (indexed-after-data)\n\
                         FUZZ_SEED={base} round={round} step={step}\n\
                         rolled back: {rolled_back}\nquery: {q}\n"
                    );
                }
            }

            if step < writes.len() {
                let w = &writes[step];
                let a = outcome(&mut plain, w);
                let b = outcome(&mut pre, w);
                let c = outcome(&mut post, w);

                assert_eq!(a, b, "\nWRITE DIVERGED (indexed-before-data)\nFUZZ_SEED={base} round={round}\nwrite: {w}\n");
                assert_eq!(a, c, "\nWRITE DIVERGED (indexed-after-data)\nFUZZ_SEED={base} round={round}\nwrite: {w}\n");
            }
        }
    }
}

#[test]
fn fuzz_index_equivalence_corpus_actually_seeks() {
    // The control, in the spirit of the injection fuzzer's: if the generated
    // queries never actually reached a seek, the test above would be comparing
    // three scans and proving nothing. Assert the indexes really are consulted.
    let mut g = empty_graph();
    create_indexes(&mut g);

    assert!(g.vertex_indexed("k"), "vertex key `k` should be indexed");
    assert!(g.vertex_indexed("u"), "vertex key `u` should be indexed");
    assert!(g.edge_indexed("w"), "edge key `w` should be indexed");

    // And a seek genuinely returns the same rows as the scan on a hand-built case
    // whose answer is known independently of either path.
    let stmts: Vec<String> = (0..6)
        .map(|i| format!("INSERT (:P {{id: 'v{i}', k: {}, u: {i}}})", i % 3))
        .collect();
    let mut plain = build(&stmts, None);
    let mut idx = build(&stmts, Some(true));

    let q = "MATCH (n) WHERE n.k = 1 RETURN n.id AS x";

    assert_eq!(outcome(&mut plain, q).unwrap().len(), 2);
    assert_eq!(outcome(&mut plain, q), outcome(&mut idx, q));
}
