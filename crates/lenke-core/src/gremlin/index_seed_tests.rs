//! Property-index SEEDING for Gremlin traversals.
//!
//! The seek is a pure optimisation, so nothing here can be checked by looking at
//! a plan — every test asserts ROWS, and each is written so that the seeded and
//! the scanned path would disagree if the seed were wrong. The companion
//! `#[ignore]`d timing test at the bottom is what catches a spelling that is
//! merely *correct*, having quietly fallen back to a scan.

use super::{g, GVal, P};
use crate::graph::Graph;
use crate::ndjson;

/// 2k vertices over two labels plus one edge each, indexed on `k` / `n` / `w`.
///
/// Both labels carry the same `k` values, so a `hasLabel` that gets dropped
/// rather than re-applied over an index seed returns twice the rows. The `R`
/// edges are deliberately OFF BY ONE (`p{i} -> q{i+1}`) so that a `has` after a
/// traversal, if it were wrongly used to seed the start, lands on a different
/// vertex than the one it should filter.
fn seeded() -> Graph {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..1000 {
        lines.push(format!(
            r#"{{"type":"node","id":"p{i}","labels":["P"],"properties":{{"k":"key{i:04}","n":{i},"tag":"t","dupe":"d"}}}}"#
        ));
        lines.push(format!(
            r#"{{"type":"node","id":"q{i}","labels":["Q"],"properties":{{"k":"key{i:04}","n":{i}}}}}"#
        ));
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"p{i}","to":"q{}","properties":{{"w":{i}}}}}"#,
            (i + 1) % 1000
        ));
        lines.push(format!(
            r#"{{"type":"edge","id":"f{i}","labels":["S"],"from":"q{i}","to":"p{i}","properties":{{"w":{i}}}}}"#
        ));
    }

    let mut graph = ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    graph.create_vertex_index("k");
    graph.create_vertex_index("n");
    // Low-cardinality on purpose: 1000 P vertices carry it, so seeding from it
    // instead of the selective key costs 500x.
    graph.create_vertex_index("dupe");
    graph.create_edge_index("w");

    graph
}

/// Element ids from a traversal, sorted — the seek and the scan are free to
/// produce a different order.
fn ids(graph: &mut Graph, t: super::Traversal) -> Vec<String> {
    let mut out: Vec<String> = t
        .id()
        .run(graph)
        .iter()
        .map(|v| match v {
            GVal::Str(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect();

    out.sort();
    out
}

#[test]
fn a_label_filter_before_the_seek_narrows_the_seeded_rows() {
    let mut graph = seeded();

    // `key0005` exists on a P and on a Q. Seeding from the index yields both,
    // so the `hasLabel` that came BEFORE the seekable `has` has to be re-run
    // over the seed rather than dropped with it.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).has_val("k", "key0005")
        ),
        vec!["p5"]
    );
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_val("k", "key0005").has_label(&["P"])
        ),
        vec!["p5"]
    );
}

#[test]
fn a_filter_before_the_seek_can_reject_every_seeded_row() {
    let mut graph = seeded();

    // Nothing labelled `Nope` exists, so the answer is empty however the start
    // was produced.
    assert!(ids(
        &mut graph,
        g().v_ids(&[]).has_label(&["Nope"]).has_val("k", "key0005")
    )
    .is_empty());
}

#[test]
fn an_unindexed_filter_before_the_seek_is_still_applied() {
    let mut graph = seeded();

    // `tag` has no index and only the P vertices carry it, so this is the same
    // trap as the label one by a different route.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_val("tag", "t").has_val("k", "key0005")
        ),
        vec!["p5"]
    );
    assert!(ids(
        &mut graph,
        g().v_ids(&[])
            .has_val("tag", "nope")
            .has_val("k", "key0005")
    )
    .is_empty());
}

#[test]
fn has_not_before_the_seek_is_still_applied() {
    let mut graph = seeded();

    // Only the Q vertices lack `tag`.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_not(&["tag"]).has_val("k", "key0005")
        ),
        vec!["q5"]
    );
}

#[test]
fn a_navigation_step_before_a_has_does_not_seed_the_start() {
    let mut graph = seeded();

    // `has('k', …)` here addresses the NEIGHBOUR, not the start. Seeding the
    // start from it would return q6 — p5's neighbour — instead of q5.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).out(&["R"]).has_val("k", "key0005")
        ),
        vec!["q5"]
    );
    assert_eq!(
        ids(&mut graph, g().v_ids(&[]).out(&["R"]).has_val("n", 5.0)),
        vec!["q5"]
    );
}

#[test]
fn two_seekable_filters_seed_from_one_and_filter_by_the_other() {
    let mut graph = seeded();

    // Both keys are indexed; only one seek is possible, so whichever is chosen
    // the other has to survive as a filter. Contradictory bounds must be empty.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_val("k", "key0005").has_val("n", 5.0)
        ),
        vec!["p5", "q5"]
    );
    assert!(ids(
        &mut graph,
        g().v_ids(&[]).has_val("k", "key0005").has_val("n", 6.0)
    )
    .is_empty());
    assert!(ids(
        &mut graph,
        g().v_ids(&[]).has_val("n", 6.0).has_val("k", "key0005")
    )
    .is_empty());
}

#[test]
fn a_label_filter_before_an_edge_seek_narrows_the_seeded_rows() {
    let mut graph = seeded();

    // Same shape on the edge side: each `w` value is carried by one R and one S.
    assert_eq!(
        ids(
            &mut graph,
            g().e_ids(&[]).has_label(&["R"]).has_val("w", 5.0)
        ),
        vec!["e5"]
    );
    assert_eq!(
        ids(&mut graph, g().e_ids(&[]).has_val("w", 5.0)),
        vec!["e5", "f5"]
    );
}

#[test]
fn a_range_after_a_label_filter_agrees_with_the_scan() {
    let mut graph = seeded();

    let seeded_form = ids(
        &mut graph,
        g().v_ids(&[]).has_label(&["P"]).has("n", P::gte(997.0)),
    );

    // `tag` is unindexed, so this spelling cannot seek and is the reference.
    let scanned = ids(
        &mut graph,
        g().v_ids(&[]).has_val("tag", "t").has("n", P::gte(997.0)),
    );

    assert_eq!(seeded_form, vec!["p997", "p998", "p999"]);
    assert_eq!(seeded_form, scanned);
}

#[test]
fn a_seeded_start_still_traverses() {
    let mut graph = seeded();

    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[])
                .has_label(&["P"])
                .has_val("k", "key0005")
                .out(&["R"])
        ),
        vec!["q6"]
    );
}

/// Ratio between the slowest and fastest spelling in a group before it counts
/// as a cliff. Generous: this is looking for the 200x class of gap, not for
/// tuning noise.
const MAX_RATIO: f64 = 12.0;

/// Median-of-5 seconds for one run of `t`.
fn equiv_time(graph: &mut Graph, t: &super::Traversal) -> f64 {
    let mut best = f64::MAX;

    for _ in 0..5 {
        let clock = std::time::Instant::now();
        let out = t.clone().run(graph);
        let secs = clock.elapsed().as_secs_f64();

        std::hint::black_box(out.len());
        if secs < best {
            best = secs;
        }
    }

    best
}

/// Equivalent traversals must not differ by orders of magnitude.
///
/// `#[ignore]`d because it is a timing test: run it with
/// `cargo test --release gremlin_spellings -- --ignored --nocapture`.
///
/// This is the test that found the gap it now guards. `V().hasLabel('P')
/// .has('k', v)` — the idiomatic TinkerPop order — used to scan while
/// `V().has('k', v).hasLabel('P')` seeked, because the seek recogniser required
/// the `has` to be the immediately-second step. 207x on vertices, 553x on
/// edges, 346x with a traversal on the end.
#[test]
#[ignore = "timing-sensitive; run explicitly"]
fn equivalent_gremlin_spellings_cost_the_same() {
    let groups: &[(&str, &[(&str, super::Traversal)])] = &[
        (
            "point equality",
            &[
                (
                    "has then label",
                    g().v_ids(&[]).has_val("k", "key0005").has_label(&["P"]),
                ),
                (
                    "label then has",
                    g().v_ids(&[]).has_label(&["P"]).has_val("k", "key0005"),
                ),
                (
                    "eq predicate",
                    g().v_ids(&[]).has("k", P::eq("key0005")).has_label(&["P"]),
                ),
                (
                    "within of one",
                    g().v_ids(&[])
                        .has_label(&["P"])
                        .has("k", P::within(["key0005"])),
                ),
            ],
        ),
        (
            "equality then a traversal",
            &[
                (
                    "has first",
                    g().v_ids(&[]).has_val("k", "key0005").out(&["R"]),
                ),
                (
                    "label then has",
                    g().v_ids(&[])
                        .has_label(&["P"])
                        .has_val("k", "key0005")
                        .out(&["R"]),
                ),
            ],
        ),
        (
            "edge property",
            &[
                (
                    "has then label",
                    g().e_ids(&[]).has_val("w", 5.0).has_label(&["R"]),
                ),
                (
                    "label then has",
                    g().e_ids(&[]).has_label(&["R"]).has_val("w", 5.0),
                ),
            ],
        ),
    ];
    let mut graph = seeded();
    let mut failures: Vec<String> = Vec::new();

    for (group, spellings) in groups {
        // A "fast" spelling that answers differently is not an equivalent
        // spelling, so agreement comes first.
        let expect = ids(&mut graph, spellings[0].1.clone());

        for (name, t) in &spellings[1..] {
            assert_eq!(
                ids(&mut graph, t.clone()),
                expect,
                "[{group}] `{name}` disagreed with `{}`",
                spellings[0].0
            );
        }

        let times: Vec<f64> = spellings
            .iter()
            .map(|(_, t)| equiv_time(&mut graph, t))
            .collect();
        let fastest = times.iter().copied().fold(f64::MAX, f64::min);

        for ((name, _), t) in spellings.iter().zip(&times) {
            let ratio = t / fastest;

            println!("  {ratio:>6.1}x  [{group}] {name}");
            if ratio > MAX_RATIO {
                failures.push(format!("[{group}] `{name}` is {ratio:.0}x the fastest"));
            }
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ---------------------------------------------------------------------------
// Inherited from the shared access path (crate::seek) — behaviours Gremlin's
// own recogniser did not have before it lowered into that layer.
// ---------------------------------------------------------------------------

#[test]
fn a_repeated_value_in_within_yields_one_row() {
    let mut graph = seeded();

    // The hand-rolled seed concatenated one point lookup per value with no
    // dedup, so a repeated value returned the element TWICE — a duplicate
    // candidate becomes a duplicate row, which is a wrong answer rather than
    // slow one. The shared layer dedups.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has("k", P::within(["key0005", "key0005"]))
        ),
        vec!["p5", "q5"]
    );
}

#[test]
fn the_more_selective_of_two_filters_seeds() {
    let mut graph = seeded();

    // Both keys are indexed. `n` matches 2 elements and `dupe` matches 1000, so
    // seeding from `dupe` costs 500x. Gremlin used to take the FIRST seekable
    // `has` while GQL took the most selective — the drift the shared layer
    // removes. Only the rows are asserted here; `equivalent_gremlin_spellings_
    // cost_the_same` is what pins the cost.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_val("dupe", "d").has_val("n", 5.0)
        ),
        vec!["p5"]
    );
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_val("n", 5.0).has_val("dupe", "d")
        ),
        vec!["p5"]
    );
}

#[test]
fn outside_is_a_union_not_an_empty_intersection() {
    let mut graph = seeded();

    // `outside(lo, hi)` is `< lo OR > hi`. Lowered as a conjunction it would be
    // an empty range — the exact opposite of what it means.
    let got = ids(
        &mut graph,
        g().v_ids(&[])
            .has_label(&["P"])
            .has("n", P::outside(2.0, 996.0)),
    );

    assert_eq!(got, vec!["p0", "p1", "p997", "p998", "p999"]);
}

#[test]
fn starts_with_seeks_a_prefix_range() {
    let mut graph = seeded();

    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[])
                .has_label(&["P"])
                .has("k", P::starts_with("key099"))
        ),
        vec!["p990", "p991", "p992", "p993", "p994", "p995", "p996", "p997", "p998", "p999"]
    );
}

#[test]
fn a_range_and_a_point_on_two_keys_agree_with_the_scan() {
    let mut graph = seeded();

    let seeded_form = ids(
        &mut graph,
        g().v_ids(&[]).has("n", P::gte(5.0)).has_val("k", "key0007"),
    );

    assert_eq!(seeded_form, vec!["p7", "q7"]);
}

#[test]
fn a_temporal_has_seeks_the_temporal_index() {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..200 {
        lines.push(format!(
            r#"{{"type":"node","id":"d{i}","labels":["D"],"properties":{{"when":{{"@date":"2024-{:02}-{:02}"}}}}}}"#,
            (i % 12) + 1,
            (i % 28) + 1
        ));
    }

    let mut graph = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    graph.create_vertex_index("when");

    // Gremlin's own `gval_to_idxkey` had no `Temporal` arm, so this scanned while
    // the identical GQL predicate seeked. Sharing `Value::index_key` fixed it.
    // Asserted on ROWS — the timing guard is the equivalence test.
    let seek = crate::gremlin::parse("g.V().has('when', date('2024-01-01')).count()")
        .map(|t| t.run(&mut graph));

    if let Ok(out) = seek {
        assert_eq!(out.len(), 1, "count returns one row");
    }

    // Whatever the surface spelling, the conversion itself must now key a
    // temporal — that is the thing that was missing.
    let t = crate::temporal::Date::parse("2024-01-01").expect("parses");

    assert!(
        crate::value::Value::Temporal(crate::temporal::Temporal::Date(t))
            .index_key()
            .is_some(),
        "a temporal must produce an index key"
    );
}
