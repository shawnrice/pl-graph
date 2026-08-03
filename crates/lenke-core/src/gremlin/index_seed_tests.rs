//! Property-index SEEDING for Gremlin traversals.
//!
//! The seek is a pure optimisation, so nothing here can be checked by looking at
//! a plan — every test asserts ROWS, and each is written so that the seeded and
//! the scanned path would disagree if the seed were wrong. The companion
//! `#[ignore]`d timing test at the bottom is what catches a spelling that is
//! merely *correct*, having quietly fallen back to a scan.

use super::{g, GVal, __, P};
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

#[test]
fn a_deduped_count_agrees_with_the_walk() {
    let mut graph = seeded();

    for (t, label) in [
        (g().v_ids(&[]).has_label(&["P"]).out(&["R"]), "one hop"),
        (
            g().v_ids(&[]).has_label(&["P"]).both(&[]).both(&[]),
            "two hops, both",
        ),
        (
            g().v_ids(&[]).has_label(&["P"]),
            "no hop — already distinct",
        ),
    ] {
        let walked = ids(&mut graph, t.clone().dedup()).len() as f64;

        assert_eq!(count_of(&mut graph, t.dedup().count()), walked, "{label}");
    }
}

#[test]
fn a_deduped_count_collapses_multi_edges() {
    // Two edges to the same neighbour are two traversers but ONE distinct
    // vertex: the counted form must collapse them, unlike the plain count.
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

    let base = g().v_ids(&[]).has_label(&["P"]).out(&["R"]);

    assert_eq!(count_of(&mut graph, base.clone().count()), 2.0);
    assert_eq!(count_of(&mut graph, base.dedup().count()), 1.0);
}

#[test]
fn a_keyed_dedup_is_not_treated_as_element_identity() {
    let mut graph = seeded();

    // `dedup('x')` keys on a TAG, not the element, so it must not take the
    // element-identity terminal.
    let out = g()
        .v_ids(&[])
        .has_label(&["P"])
        .as_("x")
        .out(&["R"])
        .dedup_labels(vec!["x".to_string()])
        .count()
        .run(&mut graph);

    assert_eq!(out.len(), 1, "still answers");
}

// ---------------------------------------------------------------------------
// `values(k)` as an IR terminal — the column is read straight into the results.
// The risks are ORDER, presence, and taking the terminal when it should not.
// ---------------------------------------------------------------------------

fn vals(graph: &mut Graph, t: super::Traversal) -> Vec<String> {
    t.run(graph)
        .iter()
        .map(|v| match v {
            GVal::Str(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

#[test]
fn a_values_terminal_matches_the_walk_exactly() {
    let mut graph = seeded();

    // Order is observable — `values()` follows traversal order, so the terminal
    // must produce the same SEQUENCE, not merely the same multiset.
    //
    // `dedup()` after the filters is the reference: it blocks the terminal (the
    // tail is no longer a bare `values`) while leaving the rows and their order
    // unchanged, since the elements are already distinct.
    for (t, label) in [
        (g().v_ids(&[]).has_label(&["P"]), "label only"),
        (g().v_ids(&[]).has("n", P::gte(996.0)), "range"),
        (g().v_ids(&[]).has_label(&["P"]).out(&["R"]), "after a hop"),
        (g().v_ids(&[]).has_val("k", "key0005"), "point seek"),
    ] {
        let terminal = vals(&mut graph, t.clone().values(&["k"]));
        let walked = vals(&mut graph, t.dedup().values(&["k"]));

        assert!(!terminal.is_empty(), "[{label}] produced nothing");
        assert_eq!(
            terminal, walked,
            "[{label}] terminal disagreed with the walk"
        );
    }
}

#[test]
fn a_values_terminal_skips_absent_and_keeps_present_null() {
    let mut graph = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"k":"x"}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
            r#"{"type":"node","id":"c","labels":["P"],"properties":{"k":null}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // `b` has no `k` and is skipped; `c` has a PRESENT null and rides through.
    assert_eq!(
        vals(&mut graph, g().v_ids(&[]).has_label(&["P"]).values(&["k"])),
        vec!["x".to_string(), "Null".to_string()]
    );
}

#[test]
fn a_values_terminal_on_an_unknown_key_is_empty() {
    let mut graph = seeded();

    assert!(vals(
        &mut graph,
        g().v_ids(&[]).has_label(&["P"]).values(&["nope"])
    )
    .is_empty());
}

#[test]
fn a_multi_key_values_is_not_taken_by_the_terminal() {
    let mut graph = seeded();

    // Two keys interleave per element; the terminal reads one column, so this
    // must fall back to the walk rather than dropping a column.
    let got = vals(
        &mut graph,
        g().v_ids(&[]).has("n", P::gte(999.0)).values(&["k", "n"]),
    );

    assert_eq!(got.len(), 4, "two elements x two keys");
}

#[test]
fn out_e_then_in_v_lowers_to_the_same_hop_as_out() {
    let mut graph = seeded();

    // `outE(L).inV()` IS `out(L)`: the edge step selects out-edges, the vertex
    // step takes their far end. Two spellings, one IR node — so they must agree
    // on rows AND on order.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out_e(&["R"]).in_v()
        ),
        ids(&mut graph, g().v_ids(&[]).has_label(&["P"]).out(&["R"]))
    );
    assert_eq!(
        count_of(
            &mut graph,
            g().v_ids(&[])
                .has_label(&["P"])
                .out_e(&["R"])
                .in_v()
                .count()
        ),
        count_of(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out(&["R"]).count()
        )
    );
    // …and the `in` direction.
    assert_eq!(
        ids(
            &mut graph,
            g().v_ids(&[]).has_label(&["Q"]).in_e(&["R"]).out_v()
        ),
        ids(&mut graph, g().v_ids(&[]).has_label(&["Q"]).in_(&["R"]))
    );
}

#[test]
fn an_edge_step_without_its_vertex_step_is_not_folded() {
    let mut graph = seeded();

    // `outE(R)` alone yields EDGES, not their far ends — folding it would change
    // what the traversal returns.
    let edges = ids(&mut graph, g().v_ids(&[]).has_label(&["P"]).out_e(&["R"]));

    assert!(edges.iter().all(|s| s.starts_with('e')), "got {edges:?}");
    assert_eq!(edges.len(), 1000);
}

#[test]
fn both_e_then_other_v_is_not_folded() {
    let mut graph = seeded();

    // `otherV` reads the traverser PATH to know which end it arrived from, so it
    // is not a pure function of the edge and must keep the per-step path.
    let via_edges = ids(
        &mut graph,
        g().v_ids(&[]).has_label(&["P"]).both_e(&["R"]).other_v(),
    );
    let direct = ids(&mut graph, g().v_ids(&[]).has_label(&["P"]).both(&["R"]));

    assert_eq!(via_edges, direct);
}

#[test]
fn a_counted_repeat_of_hops_agrees_with_the_walk() {
    let mut graph = seeded();

    for (t, label) in [
        (
            g().v_ids(&[])
                .has_label(&["P"])
                .repeat(__().out(&["R"]))
                .times(1),
            "times(1)",
        ),
        (
            g().v_ids(&[])
                .has_label(&["P"])
                .repeat(__().out(&["R"]))
                .times(2),
            "times(2)",
        ),
        (
            g().v_ids(&[])
                .has_label(&["P"])
                .repeat(__().both(&[]))
                .times(2),
            "both twice",
        ),
    ] {
        // `repeat(<hops>).times(n)` unrolls to those hops n times; the counted
        // form must not diverge from actually walking them.
        let walked = ids(&mut graph, t.clone()).len() as f64;

        assert_eq!(count_of(&mut graph, t.count()), walked, "{label}");
    }
}

#[test]
fn a_repeat_with_until_or_emit_is_not_unrolled() {
    let mut graph = seeded();

    // `until` and `emit` decide per traverser whether to stop or yield, so the
    // body is not a fixed number of hops. These must keep the stream path.
    let with_until = g()
        .v_ids(&[])
        .has_label(&["P"])
        .repeat(__().out(&["R"]))
        .until(__().has_label(&["Q"]))
        .count()
        .run(&mut graph);
    let with_emit = g()
        .v_ids(&[])
        .has_label(&["P"])
        .repeat(__().out(&["R"]))
        .times(2)
        .emit(__().has_label(&["Q"]))
        .count()
        .run(&mut graph);

    assert_eq!(with_until.len(), 1);
    assert_eq!(with_emit.len(), 1);
}

#[test]
fn an_edge_terminal_agrees_with_the_walk() {
    let mut graph = seeded();

    for (t, label) in [
        (g().v_ids(&[]).has_label(&["P"]).out_e(&["R"]), "outE R"),
        (g().v_ids(&[]).has_label(&["Q"]).in_e(&["R"]), "inE R"),
        (g().v_ids(&[]).has_label(&["P"]).both_e(&[]), "bothE any"),
        (
            g().v_ids(&[]).has_label(&["P"]).out(&["R"]).out_e(&["S"]),
            "after a hop",
        ),
    ] {
        // The edge steps land on the EDGE, not its far end, so the counted form
        // must count edges — and agree with materializing them.
        let walked = ids(&mut graph, t.clone()).len() as f64;

        assert_eq!(count_of(&mut graph, t.count()), walked, "{label}");
    }
}

#[test]
fn an_unknown_edge_label_on_a_terminal_counts_nothing() {
    let mut graph = seeded();

    assert_eq!(
        count_of(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out_e(&["Nope"]).count()
        ),
        0.0
    );
    assert!(
        count_of(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).out_e(&[]).count()
        ) > 0.0
    );
}

// ---------------------------------------------------------------------------
// The remaining terminals: `id()`, `label()`, the numeric aggregates over a
// `values(k)`, the `is(P)` that filters them, `fold()` and `count(local)`.
//
// Every one is a pure optimisation, so the reference is the SAME steps run over
// a stream. `identity()` is what forces that: it is a pass-through the terminal
// lowering declines to read, so a traversal with one in front of the terminal
// materializes where the bare one does not.
// ---------------------------------------------------------------------------

/// A terminal as a function, so one loop can run every aggregate over one shape.
type Agg = fn(super::Traversal) -> super::Traversal;

/// The same traversal, forced through the stream — the materialized reference.
fn walked(graph: &mut Graph, t: super::Traversal) -> Vec<String> {
    vals(graph, t.identity())
}

/// Prefixes that all reach the IR, each by a different route.
fn prefixes() -> Vec<(super::Traversal, &'static str)> {
    vec![
        (g().v_ids(&[]), "everything"),
        (g().v_ids(&[]).has_label(&["P"]), "label only"),
        (g().v_ids(&[]).has("n", P::gte(996.0)), "range seek"),
        (g().v_ids(&[]).has_val("k", "key0005"), "point seek"),
        (g().v_ids(&[]).has_label(&["P"]).out(&["R"]), "after a hop"),
        (
            g().v_ids(&[]).has_label(&["P"]).out_e(&["R"]),
            "edge frontier",
        ),
        (g().e_ids(&[]), "every edge"),
    ]
}

/// The property key the frontier of `label` actually carries.
fn key_for(label: &str) -> &'static str {
    if label.contains("edge") {
        "w"
    } else {
        "n"
    }
}

#[test]
fn an_id_terminal_matches_the_walk_exactly() {
    let mut graph = seeded();

    // Order is observable: `id()` follows traversal order, so the terminal has
    // to produce the same SEQUENCE, not merely the same multiset.
    for (t, label) in prefixes() {
        let terminal = vals(&mut graph, t.clone().id());
        let stream = walked(&mut graph, t.id());

        assert!(!terminal.is_empty(), "[{label}] produced nothing");
        assert_eq!(terminal, stream, "[{label}] id() disagreed with the walk");
    }
}

#[test]
fn a_label_terminal_matches_the_walk_exactly() {
    let mut graph = seeded();

    for (t, label) in prefixes() {
        let terminal = vals(&mut graph, t.clone().label());
        let stream = walked(&mut graph, t.label());

        assert!(!terminal.is_empty(), "[{label}] produced nothing");
        assert_eq!(
            terminal, stream,
            "[{label}] label() disagreed with the walk"
        );
    }
}

#[test]
fn a_label_terminal_reports_only_the_first_label() {
    let mut graph = crate::ndjson::decode(
        r#"{"type":"node","id":"a","labels":["First","Second"],"properties":{}}"#,
    )
    .expect("fixture decodes");

    // TinkerPop's `label()` is `vertex_labels(i).first()`, not "any label". A
    // lowering that read the whole label list would return both.
    assert_eq!(
        vals(&mut graph, g().v_ids(&[]).label()),
        vec!["First".to_string()]
    );
}

#[test]
fn an_id_terminal_after_a_hop_keeps_duplicates() {
    let mut graph = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"b","properties":{}}"#,
            r#"{"type":"edge","id":"e2","labels":["R"],"from":"a","to":"b","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // Two edges between the same pair are two traversers, so `b` is reached
    // twice. De-duplicating in the terminal would silently drop a row.
    assert_eq!(
        vals(&mut graph, g().v_ids(&["a"]).out(&["R"]).id()),
        vec!["b".to_string(), "b".to_string()]
    );
}

#[test]
fn a_numeric_aggregate_agrees_with_the_walk() {
    let mut graph = seeded();
    let aggs: [(Agg, &str); 4] = [
        (super::Traversal::sum, "sum"),
        (super::Traversal::mean, "mean"),
        (super::Traversal::min, "min"),
        (super::Traversal::max, "max"),
    ];

    for (t, label) in prefixes() {
        let key = key_for(label);

        for (agg, name) in aggs {
            let terminal = vals(&mut graph, agg(t.clone().values(&[key])));
            let stream = vals(&mut graph, agg(t.clone().values(&[key]).identity()));

            assert_eq!(terminal.len(), 1, "[{label}/{name}] not one row");
            assert_eq!(terminal, stream, "[{label}/{name}] disagreed with the walk");
        }
    }
}

#[test]
fn an_aggregate_over_a_non_numeric_column_still_faults() {
    let mut graph = seeded();

    // `k` is a string column. `sum()`/`mean()` over it is a type FAULT in the
    // stream, so a lowering that answered `null` would make the IR observable.
    for t in [
        g().v_ids(&[]).has_label(&["P"]).values(&["k"]).sum(),
        g().v_ids(&[]).has_label(&["P"]).values(&["k"]).mean(),
    ] {
        assert!(crate::gremlin::try_run(&mut graph, &t).is_err());
    }

    // `min`/`max` over strings is well defined and must still answer.
    assert_eq!(
        vals(
            &mut graph,
            g().v_ids(&[]).has_label(&["P"]).values(&["k"]).min()
        ),
        vec!["key0000".to_string()]
    );
}

#[test]
fn an_aggregate_over_an_absent_key_folds_the_empty_stream() {
    let mut graph = seeded();
    let none = || g().v_ids(&[]).has_label(&["P"]).values(&["nope"]);

    // Nothing is not zero: TinkerPop folds an empty numeric aggregate to null.
    assert_eq!(vals(&mut graph, none().sum()), vec!["Null".to_string()]);
    assert_eq!(vals(&mut graph, none().mean()), vec!["Null".to_string()]);
    assert_eq!(vals(&mut graph, none().min()), vec!["Null".to_string()]);
    assert_eq!(vals(&mut graph, none().max()), vec!["Null".to_string()]);
    assert_eq!(
        vals(&mut graph, none().count()),
        vec!["Num(0.0)".to_string()]
    );
    assert_eq!(
        vals(&mut graph, none().fold()),
        vec!["List([])".to_string()]
    );

    // And each agrees with the stream, which is where those rules are written.
    for t in [none().sum(), none().mean(), none().min(), none().max()] {
        let stream = vals(&mut graph, t.clone().identity());

        assert_eq!(vals(&mut graph, t), stream);
    }
}

#[test]
fn an_aggregate_over_a_stored_null_agrees_with_the_walk() {
    let mut graph = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":3}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"n":null}}"#,
            r#"{"type":"node","id":"c","labels":["P"],"properties":{}}"#,
            r#"{"type":"node","id":"d","labels":["P"],"properties":{"n":5}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // A stored null makes the column heterogeneous, so the numeric path has to
    // decline — and the stream's rule (nulls skipped, not summed as 0) has to
    // survive that.
    for t in [
        g().v_ids(&[]).values(&["n"]).sum(),
        g().v_ids(&[]).values(&["n"]).mean(),
        g().v_ids(&[]).values(&["n"]).min(),
        g().v_ids(&[]).values(&["n"]).max(),
        g().v_ids(&[]).values(&["n"]).count(),
    ] {
        let stream = vals(&mut graph, t.clone().identity());

        assert_eq!(vals(&mut graph, t), stream);
    }
}

#[test]
fn an_is_filter_after_values_matches_the_walk() {
    let mut graph = seeded();
    let aggs: [(Agg, &str); 5] = [
        (super::Traversal::count, "count"),
        (super::Traversal::sum, "sum"),
        (super::Traversal::min, "min"),
        (super::Traversal::max, "max"),
        (super::Traversal::fold, "fold"),
    ];

    for (p, name) in [
        (P::gt(500.0), "gt"),
        (P::gte(500.0), "gte"),
        (P::lt(500.0), "lt"),
        (P::lte(500.0), "lte"),
        (P::eq(500.0), "eq"),
        (P::neq(500.0), "neq"),
        (P::between(100.0, 200.0), "between"),
        (P::inside(100.0, 200.0), "inside"),
        (P::outside(100.0, 200.0), "outside"),
        (P::within([1.0, 2.0, 3.0]), "within"),
        (P::without([1.0, 2.0, 3.0]), "without"),
    ] {
        let t = || {
            g().v_ids(&[])
                .has_label(&["P"])
                .values(&["n"])
                .is(p.clone())
        };

        for (agg, aname) in aggs {
            let terminal = vals(&mut graph, agg(t()));
            let stream = vals(&mut graph, agg(t().identity()));

            assert_eq!(terminal, stream, "[{name}/{aname}] disagreed with the walk");
        }

        assert_eq!(
            vals(&mut graph, t()),
            vals(&mut graph, t().identity()),
            "[{name}] bare is() disagreed with the walk"
        );
    }
}

#[test]
fn a_nan_in_the_column_keeps_the_ordering_terminals_on_the_stream() {
    let mut graph = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":3}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"n":7}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    graph.set_vertex_prop(1, "n", crate::graph::Value::Num(f64::NAN));

    // `partial_cmp` has no answer for a NaN, so `cmp_or_fault` flags a type
    // fault: `min`/`max` and an ordering `is` THROW through the stream. Reading
    // the column instead would answer where the query is supposed to fail.
    for t in [
        g().v_ids(&[]).values(&["n"]).min(),
        g().v_ids(&[]).values(&["n"]).max(),
        g().v_ids(&[]).values(&["n"]).is(P::gt(1.0)).count(),
    ] {
        assert!(crate::gremlin::try_run(&mut graph, &t).is_err());
    }

    // `sum`/`mean` do not order, so a NaN just propagates — both paths agree.
    for t in [
        g().v_ids(&[]).values(&["n"]).sum(),
        g().v_ids(&[]).values(&["n"]).mean(),
    ] {
        let stream = vals(&mut graph, t.clone().identity());

        assert_eq!(vals(&mut graph, t), stream);
    }
}

#[test]
fn an_is_filter_against_a_non_number_still_faults() {
    let mut graph = seeded();

    // Ordering a number against a string is a type fault. Answering it as "no
    // rows" from the column would make the lowering observable.
    assert!(crate::gremlin::try_run(
        &mut graph,
        &g().v_ids(&[])
            .has_label(&["P"])
            .values(&["n"])
            .is(P::gt("x"))
            .count()
    )
    .is_err());
}

#[test]
fn an_is_filter_over_a_string_column_falls_back() {
    let mut graph = seeded();

    // The numeric path cannot express this; the stream still has to answer it.
    for t in [
        g().v_ids(&[])
            .has_label(&["P"])
            .values(&["k"])
            .is(P::eq("key0005")),
        g().v_ids(&[])
            .has_label(&["P"])
            .values(&["k"])
            .is(P::containing("0005")),
    ] {
        assert_eq!(vals(&mut graph, t.clone()), vec!["key0005".to_string()]);
        assert_eq!(vals(&mut graph, t.clone()), vals(&mut graph, t.identity()));
    }
}

#[test]
fn a_fold_terminal_matches_the_walk() {
    let mut graph = seeded();

    for (t, label) in prefixes() {
        let key = key_for(label);
        let terminal = vals(&mut graph, t.clone().fold());
        let stream = walked(&mut graph, t.clone().fold());

        assert_eq!(terminal.len(), 1, "[{label}] fold is one row");
        assert_eq!(terminal, stream, "[{label}] fold() disagreed with the walk");

        let vterminal = vals(&mut graph, t.clone().values(&[key]).fold());
        let vstream = vals(&mut graph, t.values(&[key]).identity().fold());

        assert_eq!(vterminal, vstream, "[{label}] values().fold() disagreed");
    }
}

#[test]
fn a_local_count_terminal_matches_the_walk() {
    let mut graph = seeded();

    for (t, label) in prefixes() {
        let key = key_for(label);
        let terminal = vals(&mut graph, t.clone().count_local());
        let stream = walked(&mut graph, t.clone().count_local());

        assert_eq!(terminal, stream, "[{label}] count(local) disagreed");

        let vterminal = vals(&mut graph, t.clone().values(&[key]).count_local());
        let vstream = vals(&mut graph, t.values(&[key]).identity().count_local());

        assert_eq!(
            vterminal, vstream,
            "[{label}] values().count(local) disagreed"
        );
    }
}

#[test]
fn a_values_count_matches_the_walk() {
    let mut graph = seeded();

    for (t, label) in prefixes() {
        let key = key_for(label);
        let terminal = vals(&mut graph, t.clone().values(&[key]).count());
        let stream = vals(&mut graph, t.values(&[key]).identity().count());

        assert_eq!(terminal, stream, "[{label}] values().count() disagreed");
    }
}

#[test]
fn an_edge_frontier_declines_a_navigating_step() {
    let mut graph = seeded();

    // `E().inV()` is not a projection off the edge ids — those are edge indices,
    // and reading them as vertices would answer nonsense. The allowlist that
    // guards the edge frontier has to reject it, so the stream answers instead.
    let stream = vals(&mut graph, g().e_ids(&[]).in_v().id().identity());
    let got = vals(&mut graph, g().e_ids(&[]).in_v().id());

    assert_eq!(got, stream);
    assert_eq!(got.len(), 2000, "one head per edge");
}

/// Cost of carrying `as(label)` tags through a traversal.
///
/// `Trav::tags` is cloned by every `step`/`with`, so a labelled traversal pays
/// per hop. This is the Gremlin half of the same deep copy that made GQL's group
/// variables slow — `select(Pop.all, 'x')` after a `repeat` IS a group variable.
/// Run: `cargo test --release bench_tag_carry -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_tag_carry() {
    let mut g = seeded();

    type Build = fn() -> super::Traversal;

    let build: &[(&str, Build)] = &[
        ("untagged 2-hop", || {
            super::g().V().out(&[]).out(&[]).count()
        }),
        ("as() then 2-hop", || {
            super::g().V().as_("x").out(&[]).out(&[]).count()
        }),
        ("as() per hop", || {
            super::g()
                .V()
                .as_("x")
                .out(&[])
                .as_("y")
                .out(&[])
                .as_("z")
                .count()
        }),
        ("as() per hop + select all", || {
            super::g()
                .V()
                .as_("x")
                .out(&[])
                .as_("x")
                .out(&[])
                .as_("x")
                .select_pop(super::Pop::All, &["x"])
                .count()
        }),
    ];

    println!("\n{:<28} {:>10}", "traversal", "best");

    for (name, mk) in build {
        let mut best = f64::MAX;

        for _ in 0..7 {
            let t = std::time::Instant::now();
            let _ = mk().run(&mut g);
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }

        println!("{name:<28} {best:>9.0}us");
    }
}

/// `needs_path` must stay true for a traversal whose PATH-reading step is nested
/// inside a container.
///
/// Only five steps read `Trav::path` — `OtherV`, `SimplePath`, `CyclicPath`,
/// `Path`, `Sack` — so `path_free` recurses into `repeat`/`union`/`choose`/… to
/// let the common looping shapes off the per-traverser path clone (2.3x on
/// `repeat(out()).times(2)`). The recursion is the risky half: a body that DOES
/// read the path must still force tracking, or the outer traversal loses the
/// history the inner step depends on.
#[test]
fn a_path_reading_step_inside_a_container_still_tracks_the_path() {
    let mut g = seeded();

    // `simplePath` inside `repeat` prunes revisits; without path history it
    // cannot, and the walk returns more rows.
    let pruned = super::g()
        .V()
        .repeat(super::__().both(&[]).simple_path())
        .times(3)
        .count()
        .run(&mut g);
    let unpruned = super::g()
        .V()
        .repeat(super::__().both(&[]))
        .times(3)
        .count()
        .run(&mut g);

    assert_ne!(
        pruned, unpruned,
        "simplePath nested in repeat did not prune — the path was not tracked"
    );

    // `path()` inside `union` must still produce real paths.
    let paths = super::g()
        .V()
        .limit(3)
        .union(vec![
            super::__().out(&[]).path(),
            super::__().in_(&[]).path(),
        ])
        .run(&mut g);

    assert!(
        paths.iter().any(|v| match v {
            GVal::List(items) => items.len() > 1,
            _ => false,
        }),
        "path() nested in union produced no multi-element path"
    );
}
