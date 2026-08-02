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

// ---------------------------------------------------------------------------
// The shared columnar filter (crate::seek). A `has` is only DROPPED from the
// pipeline when the shared layer captured it in full — anything else must still
// run, and losing one is silent.
// ---------------------------------------------------------------------------

#[test]
fn an_uncaptured_predicate_is_not_dropped() {
    let mut graph = seeded();

    // `neq` is not expressible as a seek or a column test, so it contributes
    // nothing to the shared filter. Dropping its step along with the captured
    // ones would silently widen the answer from 1 row to 1000.
    let got = ids(
        &mut graph,
        g().v_ids(&[]).has("k", P::neq("key0005")).has_val("n", 7.0),
    );

    assert_eq!(got, vec!["p7", "q7"]);
}

#[test]
fn a_columnar_filter_agrees_with_the_scan() {
    let mut graph = seeded();

    // `n` is indexed, `tag` is not — so the second spelling cannot seek and runs
    // the ordinary path. Both must agree.
    let seeded_form = ids(&mut graph, g().v_ids(&[]).has("n", P::gte(997.0)));
    let scanned = ids(
        &mut graph,
        g().v_ids(&[]).has_val("tag", "t").has("n", P::gte(997.0)),
    );

    assert_eq!(seeded_form.len(), 6, "three P and three Q");
    assert_eq!(scanned, vec!["p997", "p998", "p999"]);
}

#[test]
fn a_missing_property_does_not_match_a_column_test() {
    let mut graph = seeded();

    // Only the P vertices carry `tag`. A column test reads `present` as well as
    // the value; conflating them would match every Q too.
    assert_eq!(
        ids(&mut graph, g().v_ids(&[]).has_val("tag", "t")).len(),
        1000
    );
}

#[test]
fn a_cross_type_comparison_keeps_the_per_step_path() {
    let mut graph = seeded();

    // `k` is a string column and the operand is a number. That is a TYPE FAULT in
    // Gremlin, not "no rows" — the shared filter must decline it so the step that
    // records the fault still runs.
    let out = crate::gremlin::try_run(&mut graph, &g().v_ids(&[]).has("k", P::gt(5.0)).count());

    assert!(out.is_err(), "a cross-type compare must still fault");
}

#[test]
fn a_label_filter_composes_with_a_columnar_one() {
    let mut graph = seeded();

    // `hasLabel` is never absorbed, so it has to still run over the filtered set.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).has("n", P::gte(998.0))
        ),
        vec!["p998", "p999"]
    );
}

#[test]
fn a_bucket_seeded_label_still_honours_the_first_label_rule() {
    // `m0` carries [Q, P]: it IS in P's label bucket (buckets index every label)
    // but its FIRST label is Q, so `hasLabel('P')` must not match it. Seeding
    // from the bucket and forgetting to re-check is how that goes wrong, and the
    // bucket seed only runs when the label is selective — which it is here.
    let mut graph = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"p0","labels":["P"],"properties":{"n":1}}"#,
            r#"{"type":"node","id":"m0","labels":["Q","P"],"properties":{"n":2}}"#,
            r#"{"type":"node","id":"q0","labels":["Q"],"properties":{"n":3}}"#,
            r#"{"type":"node","id":"r0","labels":["R"],"properties":{"n":4}}"#,
            r#"{"type":"node","id":"r1","labels":["R"],"properties":{"n":5}}"#,
            r#"{"type":"node","id":"r2","labels":["R"],"properties":{"n":6}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    assert_eq!(
        ids(&mut graph, g().v_ids(&[]).has_label(&["P"])),
        vec!["p0"]
    );
    assert_eq!(
        ids(&mut graph, g().v_ids(&[]).has_label(&["Q"])),
        vec!["m0", "q0"]
    );
    // …and composed with a column filter.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_label(&["Q"]).has("n", P::gte(3.0))
        ),
        vec!["q0"]
    );
}

#[test]
fn an_unknown_label_matches_nothing() {
    let mut graph = seeded();

    assert!(ids(&mut graph, g().v_ids(&[]).has_label(&["Nope"])).is_empty());
    assert!(ids(
        &mut graph,
        g().v_ids(&[]).has_label(&["Nope"]).has_val("n", 5.0)
    )
    .is_empty());
}

// ---------------------------------------------------------------------------
// `count()` as an IR terminal — answered without building a traverser. The risk
// is answering it when something in the pipeline still needs to run.
// ---------------------------------------------------------------------------

fn count_of(graph: &mut Graph, t: super::Traversal) -> f64 {
    match t.run(graph).as_slice() {
        [GVal::Num(n)] => *n,
        other => panic!("expected one number, got {other:?}"),
    }
}

#[test]
fn a_counted_filter_run_agrees_with_the_row_count() {
    let mut graph = seeded();

    for t in [
        g().v_ids(&[]).has_label(&["P"]),
        g().v_ids(&[]).has_val("n", 5.0),
        g().v_ids(&[]).has_label(&["P"]).has("n", P::gte(998.0)),
        g().v_ids(&[]).has("n", P::gte(0.0)),
        g().e_ids(&[]).has_label(&["R"]),
    ] {
        let rows = ids(&mut graph, t.clone()).len() as f64;

        assert_eq!(count_of(&mut graph, t.count()), rows);
    }
}

#[test]
fn a_count_after_an_uncaptured_filter_is_not_short_circuited() {
    let mut graph = seeded();

    // `neq` contributes nothing to the IR, so the count cannot be answered from
    // it — the step still has to run. Answering early would report 2000.
    // Both p5 and q5 carry key0005, so two are excluded, not one.
    assert_eq!(
        count_of(
            &mut graph,
            g().v_ids(&[]).has("k", P::neq("key0005")).count()
        ),
        1998.0
    );
}

#[test]
fn a_count_with_a_step_before_it_is_not_short_circuited() {
    let mut graph = seeded();

    // `dedup` sits between the filters and the count; the terminal only applies
    // when nothing else does.
    assert_eq!(
        count_of(&mut graph, g().v_ids(&[]).has_label(&["P"]).dedup().count()),
        1000.0
    );
    // A traversal after the filters is likewise not a counted prefix.
    assert_eq!(
        count_of(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out(&["R"]).count()
        ),
        1000.0
    );
}

#[test]
fn a_cross_type_count_still_faults() {
    let mut graph = seeded();

    // The terminal must not answer what the per-step path would have thrown on.
    assert!(
        crate::gremlin::try_run(&mut graph, &g().v_ids(&[]).has("k", P::gt(5.0)).count()).is_err()
    );
}

// ---------------------------------------------------------------------------
// Expansion lowered into the IR. A counted expansion never materializes the
// neighbours, so the risks are direction, edge-type filtering and multiplicity.
// ---------------------------------------------------------------------------

#[test]
fn a_counted_expansion_agrees_with_the_walk() {
    let mut graph = seeded();

    for (t, label) in [
        (g().v_ids(&[]).has_label(&["P"]).out(&["R"]), "out R"),
        (g().v_ids(&[]).has_label(&["Q"]).in_(&["R"]), "in R"),
        (g().v_ids(&[]).has_label(&["P"]).both(&["R"]), "both R"),
        (g().v_ids(&[]).has_label(&["P"]).out(&[]), "out any"),
        (g().v_ids(&[]).has_label(&["P"]).both(&[]), "both any"),
    ] {
        // The walk materializes; the counted form must not disagree with it.
        let walked = ids(&mut graph, t.clone()).len() as f64;

        assert_eq!(count_of(&mut graph, t.count()), walked, "{label}");
    }
}

#[test]
fn a_counted_expansion_keeps_multi_edges() {
    // Two edges between the same pair are two traversers, so the count is 2 —
    // de-duplicating the expansion would silently answer 1.
    let mut graph = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["Q"],"properties":{}}"#,
            r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"b","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    assert_eq!(
        count_of(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out(&["R"]).count()
        ),
        2.0
    );
}

#[test]
fn an_unknown_edge_label_expands_to_nothing() {
    let mut graph = seeded();

    // An unresolvable type name matches nothing. An EMPTY label list means "any
    // type", so the two must not collapse into each other.
    assert_eq!(
        count_of(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out(&["Nope"]).count()
        ),
        0.0
    );
    assert!(
        count_of(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out(&[]).count()
        ) > 0.0
    );
}

#[test]
fn a_counted_expansion_respects_direction() {
    let mut graph = seeded();

    // R runs p{i} -> q{i+1}, S runs q{i} -> p{i}. Out and in are not symmetric.
    let out_r = count_of(
        &mut graph,
        g().v_ids(&[]).has_label(&["P"]).out(&["R"]).count(),
    );
    let in_r = count_of(
        &mut graph,
        g().v_ids(&[]).has_label(&["P"]).in_(&["R"]).count(),
    );

    assert_eq!(out_r, 1000.0);
    assert_eq!(in_r, 0.0, "no R edge points at a P");
}

#[test]
fn a_chained_counted_expansion_agrees_with_the_walk() {
    let mut graph = seeded();

    for (t, label) in [
        (
            g().v_ids(&[]).has_label(&["P"]).out(&["R"]).out(&["S"]),
            "R then S",
        ),
        (
            g().v_ids(&[]).has_label(&["P"]).out(&[]).out(&[]),
            "any then any",
        ),
        (
            g().v_ids(&[])
                .has_label(&["P"])
                .out(&["R"])
                .out(&["S"])
                .out(&["R"]),
            "three hops",
        ),
        (
            g().v_ids(&[]).has_label(&["P"]).both(&[]).both(&[]),
            "both twice",
        ),
    ] {
        // Every intermediate hop keeps duplicates, since each is its own
        // traverser — collapsing one would undercount the next.
        let walked = ids(&mut graph, t.clone()).len() as f64;

        assert_eq!(count_of(&mut graph, t.count()), walked, "{label}");
    }
}

#[test]
fn an_unknown_label_mid_chain_stops_the_count() {
    let mut graph = seeded();

    assert_eq!(
        count_of(
            &mut graph,
            g().v_ids(&[])
                .has_label(&["P"])
                .out(&["R"])
                .out(&["Nope"])
                .count()
        ),
        0.0
    );
}

// ---------------------------------------------------------------------------
// Lazy path accumulation. `Trav::step` skips cloning the path when no step in
// the run reads it — decided from an ALLOWLIST, so anything unrecognized keeps
// accumulating. These pin the boundary: a traversal that LOOKS path-free but
// ends in a path consumer must still have its path.
// ---------------------------------------------------------------------------

#[test]
fn a_path_consuming_step_still_gets_its_path() {
    let mut graph = seeded();

    // Every step before `path()` is on the allowlist, so the decision rests
    // entirely on `path()` itself being off it.
    let out = g()
        .v_ids(&[])
        .has_label(&["P"])
        .has_val("k", "key0005")
        .out(&["R"])
        .path()
        .run(&mut graph);

    match out.as_slice() {
        [GVal::List(hops)] => assert_eq!(hops.len(), 2, "start vertex then neighbour"),
        other => panic!("expected one path of two hops, got {other:?}"),
    }
}

#[test]
fn simple_path_still_filters_on_a_path_free_looking_prefix() {
    let mut graph = seeded();

    // `simplePath` reads the path to reject repeats. If accumulation had been
    // skipped, every traverser would carry an empty path and none would be
    // rejected.
    let with_filter = g()
        .v_ids(&[])
        .has_label(&["P"])
        .both(&[])
        .both(&[])
        .simple_path()
        .count()
        .run(&mut graph);
    let without = g()
        .v_ids(&[])
        .has_label(&["P"])
        .both(&[])
        .both(&[])
        .count()
        .run(&mut graph);

    assert_ne!(
        with_filter, without,
        "simplePath must drop the walks that return to their start"
    );
}
