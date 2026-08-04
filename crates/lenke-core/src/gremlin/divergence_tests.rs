//! Regression tests for behaviors where this engine once diverged from the TS
//! `@lenke/gremlin` engine. Each was surfaced by the step-test parity port, then
//! aligned; these lock the agreed behavior in place so it cannot drift back.

use super::{g, GVal, __, P};
use crate::graph::Graph;

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gremlin()
}

fn q(t: super::Traversal) -> Vec<GVal> {
    let mut g = modern();
    t.run(&mut g)
}

fn one_num(r: Vec<GVal>) -> f64 {
    match r.as_slice() {
        [GVal::Num(n)] => *n,
        _ => panic!("expected single number, got {r:?}"),
    }
}

/// Sorted string results (order-independent).
fn sorted_names(r: Vec<GVal>) -> Vec<String> {
    let mut v: Vec<String> = r
        .iter()
        .map(|g| match g {
            GVal::Str(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    v.sort();
    v
}

// --- min/max skip nulls (TS: Comparable ignores null) -----------------------

#[test]
fn min_skips_nulls() {
    let r = g()
        .inject([GVal::Null, GVal::Num(10.0), GVal::Num(9.0), GVal::Null])
        .min();
    assert_eq!(one_num(q(r)), 9.0);
}

#[test]
fn max_skips_nulls() {
    let r = g()
        .inject([GVal::Null, GVal::Num(10.0), GVal::Num(9.0), GVal::Null])
        .max();
    assert_eq!(one_num(q(r)), 10.0);
}

#[test]
fn min_all_null_is_null() {
    let r = g().inject([GVal::Null, GVal::Null]).min();
    assert!(matches!(q(r).as_slice(), [GVal::Null]));
}

// --- sum/mean over all-null collapse to [null] ------------------------------

#[test]
fn sum_all_null_is_null() {
    let r = g().inject([GVal::Null, GVal::Null]).sum();
    assert!(matches!(q(r).as_slice(), [GVal::Null]));
}

#[test]
fn mean_all_null_is_null() {
    let r = g().inject([GVal::Null]).mean();
    assert!(matches!(q(r).as_slice(), [GVal::Null]));
}

// --- E() resolves external "e<n>" edge ids ----------------------------------

#[test]
fn e_external_id_resolves() {
    let r = g().e_ids(&["e0"]);
    assert_eq!(q(r).len(), 1);
}

// --- hasKey works on a property stream --------------------------------------

#[test]
fn has_key_on_property_stream() {
    // marko has name + age; hasKey("name") keeps just the name property.
    let r = g().v_ids(&["1"]).properties(&[]).has_key(&["name"]);
    assert_eq!(q(r).len(), 1);
}

// --- dedup().by(a).by(b) keys on the full tuple, not just the first by -------

#[test]
fn dedup_multi_by_keys_on_full_tuple() {
    // lop and ripple share lang=java but differ on name.
    let by_lang = g().v_ids(&["3", "5"]).dedup().by("lang");
    assert_eq!(q(by_lang).len(), 1);

    let by_lang_name = g().v_ids(&["3", "5"]).dedup().by("lang").by("name");
    assert_eq!(q(by_lang_name).len(), 2);
}

// --- value() is identity on a non-property traverser ------------------------

#[test]
fn value_identity_on_non_property() {
    let r = g().inject([GVal::Num(5.0)]).value();
    assert_eq!(one_num(q(r)), 5.0);
}

// --- property() drops non-element traversers (TS) ---------------------------

#[test]
fn property_drops_non_element() {
    let r = g().inject([GVal::Num(5.0)]).property("k", GVal::Num(1.0));
    assert!(q(r).is_empty());
}

// --- loops() counts from 1 in the first body pass (TinkerPop) ----------------

#[test]
fn repeat_until_loops_stops_after_first_pass() {
    // loops()==2 fires one body pass in: marko's neighbors, not their neighbors.
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .until(__().loops().is(P::eq(2)))
        .values(&["name"]);
    assert_eq!(sorted_names(q(r)), vec!["josh", "lop", "vadas"]);
}

#[test]
fn repeat_emit_before_yields_every_level() {
    // Pre-form emit (emit().repeat(out()).times(2)) emits the start vertex and
    // every level's frontier, not just the initial + final frontier.
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .times(2)
        .emit_before(__())
        .values(&["name"]);
    assert_eq!(
        sorted_names(q(r)),
        vec!["josh", "lop", "lop", "marko", "ripple", "vadas"]
    );
}

#[test]
fn textual_emit_before_repeat_yields_every_level() {
    // TEXTUAL pre-form emit: `emit().repeat(out()).times(2)` — the emit modulator
    // PRECEDES its repeat (TinkerPop allows this). It must match the builder's
    // `.repeat(...).emit_before(...)` above (start vertex + every level), not
    // silently drop the emit because it came before the repeat step.
    // (`emit().repeat(out('MEMBER_OF'))` needs the zero-hop start.)
    let t = super::parse("g.V('1').emit().repeat(out()).times(2).values('name')").unwrap();
    assert_eq!(
        sorted_names(q(t)),
        vec!["josh", "lop", "lop", "marko", "ripple", "vadas"]
    );
}

#[test]
fn textual_until_before_repeat_attaches() {
    // Same fix, the other pre-form modulator: `until(cond).repeat(out())` — until
    // precedes its repeat and must ATTACH (stop at the first match), not be
    // dropped and run to natural termination. From marko, until(name=josh) stops
    // the walk at josh; without the fix it'd drop until and yield the final
    // frontier (["lop","ripple"]).
    let t =
        super::parse("g.V('1').until(has('name','josh')).repeat(out()).values('name')").unwrap();
    assert_eq!(sorted_names(q(t)), vec!["josh"]);
}

// --- repeat().until() is do-while: the body runs at least once ----------------
// The repeat/until do-while form: post-form `repeat(body).until(cond)` checks the
// condition AFTER the body (TinkerPop), so a start already satisfying `until`
// still runs the body once. Pre-form `until(cond).repeat(body)` stays while-do.
#[test]
fn repeat_until_post_form_is_do_while() {
    // From marko (a PERSON): the body runs once → out('KNOWS') → josh, vadas (both
    // PERSON → satisfy until and exit). The old while-do returned [marko].
    let built = q(super::g()
        .v_ids(&["1"])
        .repeat(__().out(&["KNOWS"]))
        .until(__().has_label(&["PERSON"]))
        .values(&["name"]));
    assert_eq!(sorted_names(built), vec!["josh", "vadas"]);

    // Textual post-form is byte-identical to the builder.
    let t = super::parse("g.V('1').repeat(out('KNOWS')).until(hasLabel('PERSON')).values('name')")
        .unwrap();
    assert_eq!(sorted_names(q(t)), vec!["josh", "vadas"]);

    // Pre-form `until(cond).repeat(body)` is while-do → marko exits before the body.
    let pre =
        super::parse("g.V('1').until(hasLabel('PERSON')).repeat(out('KNOWS')).values('name')")
            .unwrap();
    assert_eq!(sorted_names(q(pre)), vec!["marko"]);
}

// --- order(Scope.local): rank a group Map by value (was a silent no-op) -------
// Gremlin local aggregation: order(Scope.local) sorts WITHIN each traverser's value
// instead of across the stream — the canonical use is ranking a groupCount() Map
// by its counts. It was silently ignored; now it matches @lenke/gremlin's
// orderLocalStep (Map entries sorted by value; a list's elements sorted).
#[test]
fn order_local_ranks_group_map_by_value() {
    // Builder form: groupCount → Map{PERSON:4, SOFTWARE:2}; order(local) by value desc.
    let out = q(super::g()
        .V()
        .group_count()
        .by_label()
        .order_local()
        .by_identity_dir(super::Order::Desc));
    let entries = match &out[0] {
        GVal::Map(e) => e,
        _ => panic!("expected a Map, got {out:?}"),
    };
    let got: Vec<(String, f64)> = entries
        .iter()
        .map(|(k, v)| {
            (
                match k {
                    GVal::Str(s) => s.to_string(),
                    _ => panic!("non-string key"),
                },
                match v {
                    GVal::Num(n) => *n,
                    _ => panic!("non-number value"),
                },
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![("PERSON".to_string(), 4.0), ("SOFTWARE".to_string(), 2.0)]
    );

    // Textual form must parse to the same thing (Scope.local routing on `order`).
    let t =
        super::parse("g.V().groupCount().by(T.label).order(Scope.local).by(Order.desc)").unwrap();
    assert_eq!(q(t), out);
}

#[test]
fn order_local_sorts_a_folded_list() {
    let t =
        super::parse("g.V().hasLabel('PERSON').values('age').fold().order(Scope.local)").unwrap();
    let out = q(t);
    let nums: Vec<f64> = match &out[0] {
        GVal::List(xs) => xs
            .iter()
            .map(|x| match x {
                GVal::Num(n) => *n,
                _ => panic!("non-number"),
            })
            .collect(),
        _ => panic!("expected a List, got {out:?}"),
    };
    assert_eq!(nums, vec![27.0, 29.0, 32.0, 35.0]);
}

// --- group().by(k).by(reduce) folds each group to one value (was a list) ------
// Gremlin local aggregation: a reducing value-by (count/sum/min/max/mean/
// fold) folds over the group as a barrier — group().by(k).by(count()) yields
// {k: n}, not {k: [1,1,...]}. A mapping value-by still collects a list.
#[test]
fn group_reducing_value_by_folds_the_group() {
    let entries = |out: &[GVal]| -> Vec<(GVal, GVal)> {
        match out.first() {
            Some(GVal::Map(e)) => e.clone().into_pairs(),
            other => panic!("expected a Map, got {other:?}"),
        }
    };
    let get = |es: &[(GVal, GVal)], k: &str| -> GVal {
        es.iter()
            .find(|(key, _)| matches!(key, GVal::Str(s) if &**s == k))
            .map(|(_, v)| v.clone())
            .unwrap_or(GVal::Null)
    };

    // by(count()) → a per-bucket count; the textual form is byte-identical.
    let by_count = q(super::g().V().group().by_label().by_t(super::__().count()));
    let es = entries(&by_count);
    assert_eq!(get(&es, "PERSON"), GVal::Num(4.0));
    assert_eq!(get(&es, "SOFTWARE"), GVal::Num(2.0));
    assert_eq!(
        q(super::parse("g.V().group().by(T.label).by(count())").unwrap()),
        by_count
    );

    // by(values('age').sum()) → sum per bucket; SOFTWARE has no ages → Null.
    let by_sum = q(super::g()
        .V()
        .group()
        .by_label()
        .by_t(super::__().values(&["age"]).sum()));
    let es = entries(&by_sum);
    assert_eq!(get(&es, "PERSON"), GVal::Num(123.0));
    assert_eq!(get(&es, "SOFTWARE"), GVal::Null);

    // A mapping value-by (a plain key) still collects a list (unchanged).
    let by_name = q(super::g().V().group().by_label().by("name"));
    assert!(matches!(get(&entries(&by_name), "SOFTWARE"), GVal::List(v) if v.len() == 2));
}

#[test]
fn repeat_emit_loops_predicate_offset() {
    // emit(loops().is(gt(1))) emits both body levels of a times(3) walk.
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .times(3)
        .emit(__().loops().is(P::gt(1)))
        .values(&["name"]);
    assert_eq!(
        sorted_names(q(r)),
        vec!["josh", "lop", "lop", "ripple", "vadas"]
    );
}

#[test]
fn id_of_a_non_element_is_null() {
    // `id()` used to pass a non-element THROUGH unchanged, so `path().id()` handed
    // the paths back instead of nulls — and a following numeric step then faulted
    // on them where the TS engine summed nulls. Its sibling `label()` already
    // returned null here, and so does the TS engine. Found by the Gremlin
    // differential fuzzer.
    let ids = q(g().E().path().id());
    assert_eq!(ids.len(), 6);
    assert!(
        ids.iter().all(|v| matches!(v, GVal::Null)),
        "path().id() must be null, got {ids:?}"
    );

    // The sibling accessor agrees — that symmetry is the actual invariant.
    assert_eq!(ids, q(g().E().path().label()));
}

#[test]
fn summing_the_ids_of_non_elements_is_an_all_null_fold_not_a_fault() {
    // The shape the fuzzer hit. `try_run` (not the `q` helper) on purpose: `run`
    // SWALLOWS a data fault, so summing the un-nulled paths would still have
    // looked fine here — and that fault is exactly what this rules out.
    let mut graph = modern();
    let out = super::try_run(&mut graph, &g().E().path().id().sum());

    assert!(matches!(out.as_deref(), Ok([GVal::Null])), "got {out:?}");
}

#[test]
fn id_of_a_real_element_still_reports_it() {
    // The null case must not swallow the real one.
    let ids = q(g().V().has_label(&["SOFTWARE"]).id());
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|v| matches!(v, GVal::Str(_))), "got {ids:?}");
}

/// A multi-label vertex must be reachable by EVERY label it carries, in both
/// engines.
///
/// Native once matched only the FIRST label in `hasLabel`, so a vertex labelled
/// `[Q, P]` was found by the TS engine's `hasLabel('P')` and not by this one.
/// That is a byte-identity divergence, and no fuzzer caught it because none of
/// them generates a vertex with more than one label — which is also why this test
/// is here rather than left to them.
///
/// `label()` still returns the FIRST label in both engines: it has to return one.
#[test]
fn has_label_finds_a_vertex_by_any_of_its_labels() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["A"],"properties":{}}"#,
            r#"{"type":"node","id":"ab","labels":["A","B"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["B"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let ids = |t: super::Traversal, g: &mut crate::graph::Graph| {
        let mut v: Vec<String> = t
            .id()
            .run(g)
            .iter()
            .map(|x| match x {
                GVal::Str(s) => s.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        v.sort();
        v
    };

    assert_eq!(
        ids(super::g().V().has_label(&["A"]), &mut g),
        vec!["a", "ab"]
    );
    assert_eq!(
        ids(super::g().V().has_label(&["B"]), &mut g),
        vec!["ab", "b"]
    );

    // The GQL side of the same graph agrees — this is the pair that diverged.
    let rows = crate::gql::parse("MATCH (n:B) RETURN count(*) AS c")
        .expect("parses")
        .execute(&mut g, &crate::gql::eval::Params::new())
        .expect("executes");

    assert_eq!(rows.nrows, 1);

    // `label()` is still first-label-only, in both engines.
    assert_eq!(
        super::g().V().has_label(&["B"]).label().run(&mut g).len(),
        2
    );
}

/// Naming several edge labels is a disjunction over ONE edge, not a walk per
/// name — an edge labelled `[R, S]` must traverse ONCE under `outE('R','S')`.
///
/// The TS engine buckets an edge under every label it carries and walked one
/// bucket per named label, so it emitted that edge twice while this engine (one
/// adjacency pass, an any-of predicate) emitted it once. Same shape as the GQL
/// `[:R|S]` double-count. Pinned on both sides.
#[test]
fn naming_several_edge_labels_traverses_a_multi_label_edge_once() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R","S"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let n = |t: super::Traversal, g: &mut crate::graph::Graph| t.count().run(g);
    let one = vec![GVal::Num(1.0)];

    // Every spelling that selects this edge selects it exactly once.
    assert_eq!(n(super::g().v_ids(&["a"]).out_e(&["R"]), &mut g), one);
    assert_eq!(n(super::g().v_ids(&["a"]).out_e(&["S"]), &mut g), one);
    assert_eq!(n(super::g().v_ids(&["a"]).out_e(&[]), &mut g), one);
    assert_eq!(
        n(super::g().v_ids(&["a"]).out_e(&["R", "S"]), &mut g),
        one,
        "`outE('R','S')` walked the edge once per matching label"
    );
    assert_eq!(n(super::g().v_ids(&["a"]).out(&["R", "S"]), &mut g), one);

    // ...in both directions, and for `both`, which sees it from each end once.
    assert_eq!(n(super::g().v_ids(&["b"]).in_e(&["R", "S"]), &mut g), one);
    assert_eq!(n(super::g().v_ids(&["a"]).both_e(&["R", "S"]), &mut g), one);
    assert_eq!(n(super::g().v_ids(&["b"]).both_e(&["R", "S"]), &mut g), one);

    // A name no edge carries contributes nothing rather than suppressing the rest.
    assert_eq!(
        n(super::g().v_ids(&["a"]).out_e(&["R", "ABSENT"]), &mut g),
        one
    );
    assert_eq!(
        n(super::g().v_ids(&["a"]).out_e(&["ABSENT"]), &mut g),
        vec![GVal::Num(0.0)]
    );
}

/// The TEXT form of a multi-name adjacency step must agree with the builder.
///
/// `outE('KNOWS','NOPE')` is a disjunction: the name that resolves to nothing
/// contributes nothing, and the one that resolves still matches. Found by the
/// gremlin differential fuzzer once its steps started naming more than one type
/// — the builder-level test above passes, so anything wrong here is in parsing
/// or lowering, not in the traversal.
#[test]
fn a_text_step_naming_an_unknown_type_alongside_a_real_one_still_matches() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let n = |src: &str, g: &mut crate::graph::Graph| {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` failed to parse: {e}"))
            .run(g)
    };
    // Adding a name that resolves to nothing changes NOTHING — the reference is
    // the same query without it, across every plan these take (a vertex start,
    // an edge start, a chain of hops, a dedup, a projection instead of a count).
    for (with_nope, plain) in [
        ("g.V().outE('R','NOPE').count()", "g.V().outE('R').count()"),
        ("g.V().outE('NOPE','R').count()", "g.V().outE('R').count()"),
        ("g.V().out('R','NOPE').count()", "g.V().out('R').count()"),
        ("g.V().inE('R','NOPE').count()", "g.V().inE('R').count()"),
        (
            "g.V().bothE('R','NOPE').count()",
            "g.V().bothE('R').count()",
        ),
        ("g.V().both('R','NOPE').count()", "g.V().both('R').count()"),
        (
            "g.E().hasLabel('R','NOPE').count()",
            "g.E().hasLabel('R').count()",
        ),
        (
            "g.V().hasLabel('V','NOPE').count()",
            "g.V().hasLabel('V').count()",
        ),
        (
            "g.V().out('R','NOPE').out('R','NOPE').count()",
            "g.V().out('R').out('R').count()",
        ),
        (
            "g.V().outE('R','NOPE').dedup().count()",
            "g.V().outE('R').dedup().count()",
        ),
        (
            "g.V().outE('R','NOPE').inV().count()",
            "g.V().outE('R').inV().count()",
        ),
        (
            "g.V().hasLabel('V','NOPE').out('R','NOPE').count()",
            "g.V().hasLabel('V').out('R').count()",
        ),
        // A count shortcut that disagrees with the stream it stands in for is
        // the failure mode this whole family has, so pin enumeration too.
        (
            "g.V().outE('R','NOPE').values('w').fold()",
            "g.V().outE('R').values('w').fold()",
        ),
        ("g.V().out('R','NOPE').fold()", "g.V().out('R').fold()"),
    ] {
        assert_eq!(
            n(with_nope, &mut g),
            n(plain, &mut g),
            "`{with_nope}` disagrees with `{plain}`"
        );
    }

    // ...and the plain spelling really does select something, so the pairs above
    // cannot agree by both being empty.
    assert_eq!(n("g.V().outE('R').count()", &mut g), vec![GVal::Num(1.0)]);

    // A step naming ONLY unknown types still matches nothing.
    for src in [
        "g.V().outE('NOPE','ALSO_NOPE').count()",
        "g.V().out('NOPE','ALSO_NOPE').count()",
        "g.V().hasLabel('NOPE','ALSO_NOPE').count()",
        "g.E().hasLabel('NOPE','ALSO_NOPE').count()",
    ] {
        assert_eq!(
            n(src, &mut g),
            vec![GVal::Num(0.0)],
            "`{src}` matched something"
        );
    }
}

/// A returned edge carries EVERY type it has, sorted — like a returned vertex.
///
/// The result serialization emitted only `e_type`, an edge's first type, so a
/// two-type edge read back as single-type through both engines' JSON while the
/// TS mirror rendered both. Found by the gremlin differential fuzzer once its
/// fixture held a multi-type edge. `label()` still returns ONE type: it has to.
#[test]
fn a_returned_edge_carries_every_type_it_has() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["S","R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let json = |g: &mut crate::graph::Graph| {
        let plan = super::parse::parse("g.E()").expect("parses");
        let vals = plan.clone().run(g);
        super::exec::results_to_json(g, &vals)
    };

    let rendered = json(&mut g);

    assert!(
        rendered.contains(r#""labels":["R","S"]"#),
        "an edge lost a type crossing the result boundary: {rendered}"
    );

    // Removing one leaves the other, and the rendering follows.
    g.remove_edge_label(0, "R");

    let rendered = json(&mut g);

    assert!(
        rendered.contains(r#""labels":["S"]"#),
        "a removed type still rendered: {rendered}"
    );
}

/// A `dedup()` BEFORE an edge step deduplicates what precedes it, not the edges.
///
/// The count shortcut peeled an optional `dedup()` off the front of what
/// remained and then applied it to the edges it counted, so
/// `V().dedup().bothE(T).count()` — dedup the vertices, then count every
/// incident edge from both ends — collapsed each edge to one. Found by the
/// gremlin differential fuzzer; nothing about it is multi-type, the fuzzer had
/// simply never put a `dedup()` in front of an edge step before.
#[test]
fn a_dedup_before_an_edge_step_does_not_deduplicate_the_edges() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{}}"#,
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e1","from":"b","to":"c","labels":["R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let n = |src: &str, g: &mut crate::graph::Graph| {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` failed to parse: {e}"))
            .run(g)
    };

    // The reference is the same query without the count shortcut in play: a
    // `fold()` materializes the stream, so its length is what the count must be.
    for (counted, folded) in [
        (
            "g.V().dedup().bothE('R').count()",
            "g.V().dedup().bothE('R').fold()",
        ),
        (
            "g.V().dedup().outE('R').count()",
            "g.V().dedup().outE('R').fold()",
        ),
        (
            "g.V().dedup().inE('R').count()",
            "g.V().dedup().inE('R').fold()",
        ),
        // ...and the form where the dedup really IS on the edges still dedupes.
        (
            "g.V().bothE('R').dedup().count()",
            "g.V().bothE('R').dedup().fold()",
        ),
    ] {
        let want = match n(folded, &mut g).as_slice() {
            [GVal::List(items)] => items.len() as f64,
            other => panic!("`{folded}` did not fold to a list: {other:?}"),
        };

        assert_eq!(
            n(counted, &mut g),
            vec![GVal::Num(want)],
            "`{counted}` disagrees with the stream it stands in for"
        );
    }

    // Each edge is incident to two vertices, so `bothE` over every vertex sees
    // it twice — the number the dedup must NOT collapse.
    assert_eq!(
        n("g.V().dedup().bothE('R').count()", &mut g),
        vec![GVal::Num(4.0)]
    );
    assert_eq!(
        n("g.V().bothE('R').dedup().count()", &mut g),
        vec![GVal::Num(2.0)]
    );
}

/// `g.V().outE(T).count()` reads the edge-type buckets instead of walking every
/// vertex's adjacency — and must agree with the walk in every case.
///
/// The shortcut is only valid when the traverser is still the whole vertex
/// universe and no hop has been taken, because only then is each edge the
/// out-edge of exactly one seeded vertex. Every row below is checked against
/// `fold()` — the same traversal materialized — so the reference is the
/// enumeration the shortcut replaces, not a hand-computed number.
#[test]
fn the_edge_type_count_agrees_with_enumerating_the_edges() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V","W"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{}}"#,
            // A self-loop, a parallel pair, a two-type edge, and an unrelated type.
            r#"{"type":"edge","id":"e0","from":"a","to":"a","labels":["R"],"properties":{"w":1}}"#,
            r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{"w":2}}"#,
            r#"{"type":"edge","id":"e2","from":"a","to":"b","labels":["R"],"properties":{"w":3}}"#,
            r#"{"type":"edge","id":"e3","from":"b","to":"c","labels":["R","S"],"properties":{"w":4}}"#,
            r#"{"type":"edge","id":"e4","from":"c","to":"a","labels":["S"],"properties":{"w":5}}"#,
            r#"{"type":"edge","id":"e5","from":"c","to":"b","labels":["OTHER"],"properties":{"w":6}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
    };

    // `count()` must equal the number of rows the same traversal yields. The
    // reference is the bare traversal drained — the enumeration the shortcut
    // replaces — not a hand-computed number.
    let check = |traversal: &str, g: &mut crate::graph::Graph| {
        let enumerated = run(&format!("g.{traversal}"), g).len() as f64;
        let counted = run(&format!("g.{traversal}.count()"), g);

        assert_eq!(
            counted,
            vec![GVal::Num(enumerated)],
            "`{traversal}.count()` disagrees with enumerating it"
        );
    };

    for t in [
        // The shortcut's own shape, in both directions.
        "V().outE('R')",
        "V().inE('R')",
        "V().outE('S')",
        "V().outE('OTHER')",
        "V().outE('NOPE')",
        // A multi-type ask: e3 carries R AND S, and is still ONE edge.
        "V().outE('R','S')",
        "V().outE('S','R')",
        "V().outE('R','NOPE')",
        // `both` sees an edge from each endpoint — the shortcut must NOT fire.
        "V().bothE('R')",
        "V().bothE('R','S')",
        // A filter before the step: no longer the whole universe.
        "V().hasLabel('W').outE('R')",
        "V().hasLabel('V').outE('R')",
        "V().has('missing', 1).outE('R')",
        // A hop before the step: no longer one seeded vertex per edge.
        "V().out('R').outE('R')",
        "V().out('R').inE('R')",
        // An edge start, and a dedup.
        "E().hasLabel('R')",
        "V().outE('R').dedup()",
        "V().bothE('R').dedup()",
    ] {
        check(t, &mut g);
    }

    // A deleted edge leaves its buckets, so the shortcut cannot count it.
    g.remove_edge(1); // e1: a -> b
    for t in ["V().outE('R')", "V().outE('R','S')", "V().bothE('R')"] {
        check(t, &mut g);
    }

    // ...and so does one that loses the type it was counted under.
    g.remove_edge_label(3, "R"); // e3 keeps only S
    for t in ["V().outE('R')", "V().outE('S')", "V().outE('R','S')"] {
        check(t, &mut g);
    }
}

/// `groupCount()` over a lowered column must equal the same query through the
/// stream — same pairs, same COUNTS, same first-seen ORDER.
///
/// The lowered path tallies a property column directly; the stream path builds
/// one traverser per value and tallies those. Two implementations of an
/// observably-ordered map is two chances to disagree, so they now share
/// `tally_group_count` and this pins that they do. A step the lowering declines
/// forces the stream path on an otherwise identical query — `barrier()`, which
/// is `stream => stream`, because anything that REORDERS (`order()`) changes
/// first-seen order and so compares two different questions.
#[test]
fn a_lowered_group_count_matches_the_streamed_one() {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..60 {
        // Deliberately skewed and repeating, with a gap: vertex 7k carries no
        // `n` at all, so PRESENCE (not null-ness) decides what is tallied.
        let props = if i % 7 == 0 {
            String::from("{}")
        } else {
            format!(r#"{{"n":{}}}"#, i % 5)
        };

        lines.push(format!(
            r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{props}}}"#
        ));
    }

    for i in 0..59 {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"n{i}","to":"n{}","labels":["R"],"properties":{{}}}}"#,
            i + 1
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
    };

    for (lowered, streamed) in [
        // A plain column tally.
        (
            "g.V().values('n').groupCount()",
            "g.V().values('n').barrier().groupCount()",
        ),
        // ...behind a label filter, and behind a hop.
        (
            "g.V().hasLabel('V').values('n').groupCount()",
            "g.V().hasLabel('V').values('n').barrier().groupCount()",
        ),
        (
            "g.V().out('R').values('n').groupCount()",
            "g.V().out('R').values('n').barrier().groupCount()",
        ),
        // A key no element carries tallies nothing.
        (
            "g.V().values('missing').groupCount()",
            "g.V().values('missing').barrier().groupCount()",
        ),
    ] {
        let a = run(lowered, &mut g);
        let b = run(streamed, &mut g);

        assert_eq!(
            a, b,
            "`{lowered}` disagrees with the same query through the stream"
        );
    }

    // ...and the tally is actually right, not merely self-consistent. 60 vertices,
    // every 7th without `n` (9 of them: 0,7,…,56), the rest cycling 0..4.
    let counted = match run("g.V().values('n').groupCount()", &mut g).as_slice() {
        [GVal::Map(pairs)] => pairs
            .iter()
            .map(|(_, v)| match v {
                GVal::Num(n) => *n,
                other => panic!("a count is a number, got {other:?}"),
            })
            .sum::<f64>(),
        other => panic!("groupCount did not return a map: {other:?}"),
    };

    assert_eq!(counted, 51.0, "every present `n` is tallied exactly once");

    // A pure NUMBER column takes a raw-bits tally, so it needs its own check.
    // `-0.0` and `0.0` are ONE key here and that IS observable: merged they are a
    // single entry counting 2, split they are two entries counting 1 each — and
    // both render as `0`, so only the counts show it.
    let mut z = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"z":0}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"z":-0.0}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{"z":2.5}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    assert_eq!(
        run("g.V().values('z').groupCount()", &mut z),
        run("g.V().values('z').barrier().groupCount()", &mut z),
        "the numeric tally disagrees with the stream"
    );

    match run("g.V().values('z').groupCount()", &mut z).as_slice() {
        [GVal::Map(pairs)] => {
            let counts: Vec<f64> = pairs
                .iter()
                .map(|(_, v)| match v {
                    GVal::Num(n) => *n,
                    other => panic!("a count is a number, got {other:?}"),
                })
                .collect();

            assert_eq!(
                counts,
                vec![2.0, 1.0],
                "`-0.0` and `0.0` are one key, so the first entry counts both"
            );
        }
        other => panic!("groupCount did not return a map: {other:?}"),
    }
}

/// A map renders in INSERTION order, which makes `order(local)` on it
/// observable and matches every other map renderer in the codebase.
///
/// Serializing sorted lexicographically (to mirror `serde_json::Map`) undid the
/// sort the user had just asked for, so `order(local).by(keys, desc)` came back
/// key-ascending — a step that silently did nothing. It also disagreed with
/// GQL's `codec::push_value`, with this module's own `push_result_value`, and
/// with the TS engine, whose maps have always been insertion-ordered.
#[test]
fn a_map_renders_in_insertion_order_so_order_local_is_observable() {
    let lines: Vec<String> = (0..12)
        .map(|i| {
            format!(
                r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{{"n":{}}}}}"#,
                i % 4
            )
        })
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let json = |src: &str, g: &mut crate::graph::Graph| {
        let vals = super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g);

        super::exec::results_to_json(g, &vals)
    };

    // First-seen order is vertex order: 0, 1, 2, 3.
    assert_eq!(
        json("g.V().values('n').groupCount()", &mut g),
        r#"[{"0":3,"1":3,"2":3,"3":3}]"#
    );

    // ...and `order(local)` actually reorders it. Under the old output sort both
    // of these came back key-ascending, whatever was asked for.
    assert_eq!(
        json(
            "g.V().values('n').groupCount().order(local).by(keys, desc)",
            &mut g
        ),
        r#"[{"3":3,"2":3,"1":3,"0":3}]"#
    );

    // A map whose insertion order is NOT sorted stays that way.
    assert_eq!(
        json("g.V().has('n', 3).values('n').groupCount()", &mut g),
        r#"[{"3":3}]"#
    );
}

/// `dedup()` over a lowered column matches the same query through the stream.
///
/// Both keep FIRST-SEEN order. `barrier()` forces the stream path — a step that
/// reorders would compare two different questions.
///
/// The keyless corner (`dedup_key` returning `None`, for a `NaN`) IS covered
/// below. A NaN cannot be INGESTED — every entry point normalizes it to null —
/// but it can be COMPUTED into a property from inside the graph, which this
/// engine documented as impossible and is not.
#[test]
fn a_lowered_dedup_matches_the_streamed_one() {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..40 {
        // Repeating values, a gap (no `n` at all), and a present null — which is
        // a value like any other here, not an absence.
        let props = match i % 8 {
            0 => String::from("{}"),
            1 => String::from(r#"{"n":null}"#),
            _ => format!(r#"{{"n":{}}}"#, i % 5),
        };

        lines.push(format!(
            r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{props}}}"#
        ));
    }

    for i in 0..39 {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"n{i}","to":"n{}","labels":["R"],"properties":{{}}}}"#,
            i + 1
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
    };

    for (lowered, streamed) in [
        (
            "g.V().values('n').dedup()",
            "g.V().values('n').barrier().dedup()",
        ),
        (
            "g.V().values('n').dedup().count()",
            "g.V().values('n').barrier().dedup().count()",
        ),
        (
            "g.V().hasLabel('V').values('n').dedup().count()",
            "g.V().hasLabel('V').values('n').barrier().dedup().count()",
        ),
        (
            "g.V().out('R').values('n').dedup()",
            "g.V().out('R').values('n').barrier().dedup()",
        ),
        (
            "g.V().values('missing').dedup().count()",
            "g.V().values('missing').barrier().dedup().count()",
        ),
    ] {
        assert_eq!(
            run(lowered, &mut g),
            run(streamed, &mut g),
            "`{lowered}` disagrees with the same query through the stream"
        );
    }

    // ...and it is actually distinct: values cycle 0..4 with a present null, so
    // five numbers plus the null.
    assert_eq!(
        run("g.V().values('n').dedup().count()", &mut g),
        vec![GVal::Num(6.0)]
    );

    // A pure NUMBER column takes a separate raw-bits path, so it needs its own
    // check — including that `-0.0` and `0.0` are ONE value, which keying on
    // `to_bits` alone would split into two zeroes.
    let mut z = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"z":0}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"z":-0.0}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{"z":1.5}}"#,
            r#"{"type":"node","id":"d","labels":["V"],"properties":{"z":1.5}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    assert_eq!(
        run("g.V().values('z').dedup()", &mut z),
        run("g.V().values('z').barrier().dedup()", &mut z),
        "the numeric dedup disagrees with the stream"
    );
    assert_eq!(
        run("g.V().values('z').dedup().count()", &mut z),
        vec![GVal::Num(2.0)],
        "`-0.0` and `0.0` are one value, so this is 0 and 1.5"
    );
}

/// A NaN can be COMPUTED into a stored property, and then every column terminal
/// has to treat it the way the stream does.
///
/// It cannot be ingested — `NaN` is normalized to null at every entry point —
/// so "a stored property is never NaN" reads as true and is not: `SET x =
/// sqrt(-1)` stores one, as do `asin(2)`, `acos(2)` and `power(-1, 0.5)`.
///
/// It matters most for `dedup`, because a NaN has NO dedup key and so is never a
/// duplicate — the stream keeps every one. The lowered numeric path keys on raw
/// bits, which collapsed them all into one: `dedup().count()` was 1 where the
/// stream said 2.
#[test]
fn a_computed_nan_column_matches_the_stream_in_every_terminal() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"m":-1}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"m":-1}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{"m":4}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // Compute the column: two NaNs (sqrt of a negative) and one real number.
    crate::gql::prepare("MATCH (a:V) SET a.x = sqrt(a.m)")
        .expect("plans")
        .execute(&mut g, &crate::gql::eval::Params::new())
        .expect("runs");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
    };

    for tail in [
        "count()",
        "sum()",
        "mean()",
        "min()",
        "max()",
        "fold()",
        "dedup()",
        "dedup().count()",
        "groupCount()",
        "is(gt(0)).count()",
        "order()",
    ] {
        let lowered = run(&format!("g.V().values('x').{tail}"), &mut g);
        let streamed = run(&format!("g.V().values('x').barrier().{tail}"), &mut g);

        assert_eq!(
            format!("{lowered:?}"),
            format!("{streamed:?}"),
            "`values('x').{tail}` over a NaN column disagrees with the stream"
        );
    }

    // The one that actually broke: each NaN is its own value, so three rows
    // deduplicate to three, not to two.
    assert_eq!(
        run("g.V().values('x').dedup().count()", &mut g),
        vec![GVal::Num(3.0)],
        "a NaN is never a duplicate, so both survive alongside the real number"
    );
}

/// A lowered `order()` over a column matches the stream — direction, ties, and
/// the type fault a NaN raises.
///
/// Three things have to hold and each has its own way of going wrong. The
/// direction can come from the step (`order(desc)`) or from the modulator
/// (`order().by(desc)`), and a `by` overrides the step. The sort is STABLE, so
/// equal keys keep INPUT order — observable because `-0.0` and `0.0` compare
/// equal and render differently, and because reversing the RESULT instead of the
/// comparator would flip them. And a NaN makes the stream record a type fault,
/// so the lowered path must decline rather than answer.
#[test]
fn a_lowered_order_matches_the_streamed_one() {
    let vals = [3.0_f64, -0.0, 1.0, 0.0, 3.0, -2.5, 1.0];
    let lines: Vec<String> = vals
        .iter()
        .enumerate()
        .map(|(i, v)| {
            format!(r#"{{"type":"node","id":"n{i}","labels":["V"],"properties":{{"n":{v}}}}}"#)
        })
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
    };

    for tail in [
        "order()",
        "order().by(asc)",
        "order().by(desc)",
        "order().limit(3)",
        "order().by(desc).limit(3)",
        "order().by(desc).limit(0)",
        "order().by(asc).limit(100)",
    ] {
        assert_eq!(
            run(&format!("g.V().values('n').{tail}"), &mut g),
            run(&format!("g.V().values('n').barrier().{tail}"), &mut g),
            "`values('n').{tail}` disagrees with the stream"
        );
    }

    // The prefix is actually sorted, and the LIMIT takes the smallest.
    assert_eq!(
        run("g.V().values('n').order().limit(3)", &mut g),
        vec![GVal::Num(-2.5), GVal::Num(-0.0), GVal::Num(0.0)]
    );

    // A NaN column declines to the stream, which raises the type fault — so both
    // spellings agree there too, rather than one answering and one throwing.
    let mut nan = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"m":-1}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"m":9}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    crate::gql::prepare("MATCH (a:V) SET a.x = sqrt(a.m)")
        .expect("plans")
        .execute(&mut nan, &crate::gql::eval::Params::new())
        .expect("runs");

    for tail in ["order()", "order().by(desc).limit(1)"] {
        assert_eq!(
            format!("{:?}", run(&format!("g.V().values('x').{tail}"), &mut nan)),
            format!(
                "{:?}",
                run(&format!("g.V().values('x').barrier().{tail}"), &mut nan)
            ),
            "`values('x').{tail}` over a NaN column disagrees with the stream"
        );
    }
}

/// A traversal PLANNED as a pattern answers exactly what the streamed one does.
///
/// `g.V().out('R').hasLabel('W')` is `MATCH ()-[:R]->(b:W)`, so it is handed to
/// the planner — which seeds the SELECTIVE end and walks the adjacency
/// backwards, instead of expanding every vertex and discarding most of the rows.
/// That reorders the work completely, so every shape it accepts is checked
/// against the same traversal forced through the stream.
///
/// `barrier()` right after `V()` is what forces it: it defeats the pattern
/// compile (the prefix no longer starts with a hop) without reordering anything.
/// Results compare as multisets because seeding the far end visits endpoints in
/// a different order, which is unspecified — the ROWS are the contract.
#[test]
fn a_planned_pattern_matches_the_streamed_traversal() {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..60 {
        let l = match i % 4 {
            0 => r#"["P","W"]"#,
            1 => r#"["P"]"#,
            2 => r#"["Q"]"#,
            _ => r#"["P","Q"]"#,
        };

        lines.push(format!(
            r#"{{"type":"node","id":"n{i}","labels":{l},"properties":{{"k":{},"s":"v{}"}}}}"#,
            i % 7,
            i % 3
        ));
    }

    for i in 0..59 {
        let t = if i % 3 == 0 { "R" } else { "S" };

        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"n{i}","to":"n{}","labels":["{t}"],"properties":{{}}}}"#,
            (i * 7 + 1) % 60
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let rows = |src: &str, g: &mut crate::graph::Graph| {
        let mut v: Vec<String> = super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
            .iter()
            .map(|x| format!("{x:?}"))
            .collect();
        v.sort();
        v
    };

    for q in [
        "g.V().out('R')",
        // The shape this exists for: the deciding filter is at the FAR end.
        "g.V().out('R').hasLabel('W')",
        "g.V().out('R').has('k', 3)",
        "g.V().out('R').hasLabel('W').has('k', 0)",
        // Filters at the near end, at both ends, and across two hops.
        "g.V().hasLabel('P').out('R')",
        "g.V().hasLabel('P').out('R').hasLabel('Q')",
        "g.V().hasLabel('P').has('k',1).out('S').has('s','v0')",
        "g.V().out('R').out('S')",
        "g.V().out('R').out('S').hasLabel('P')",
        "g.V().both('R').both('S')",
        // Every direction, an untyped hop, a type disjunction, and names that
        // resolve to nothing at either end.
        "g.V().in('S')",
        "g.V().both('R')",
        "g.V().out()",
        "g.V().out('R','S')",
        "g.V().out('NOPE')",
        "g.V().out('R').hasLabel('NOPE')",
        // NON-equality predicates must stay OUT of the pattern. `has(k, gt(3))`
        // over a string column is a type FAULT in Gremlin and three-valued
        // unknown in GQL, so folding it into a `CPropConstraint` would both
        // change which queries throw and — since a constraint is an EQUALITY —
        // silently answer `k = 3`. The compiler stops at one, leaving it to the
        // stream; these shapes are what proves it.
        "g.V().out('R').has('k', gt(3))",
        "g.V().out('R').has('k', lt(3))",
        "g.V().out('R').has('k', neq(3))",
        "g.V().out('R').has('k', within(1, 2))",
        "g.V().hasLabel('P').has('k', gte(4)).out('S')",
        "g.V().out('R').has('k', gt(3)).hasLabel('W')",
    ] {
        // `barrier()` after `V()` keeps the same rows and defeats the compile.
        let streamed = q.replacen("g.V()", "g.V().barrier()", 1);

        assert_eq!(
            rows(q, &mut g),
            rows(&streamed, &mut g),
            "`{q}` planned as a pattern disagrees with the streamed traversal"
        );
    }
}
