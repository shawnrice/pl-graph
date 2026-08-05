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
        // COMPOSED tails, which peel one step and recurse. Each peel has to keep
        // the NaN rule its direct arm has: `dedup` never keys a NaN (so every one
        // survives) and `order` DECLINES on one (so the stream raises the fault
        // rather than this swallowing it).
        "dedup().limit(2)",
        "dedup().count()",
        "dedup().limit(2).count()",
        "limit(2).count()",
        "skip(1).count()",
        "tail(2)",
        "tail(2).count()",
        "range(0, 2).count()",
        "limit(2).fold()",
        "dedup().order()",
        "order().count()",
        "order().by(desc).count()",
        "order().limit(2).count()",
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
            r#"{{"type":"edge","id":"e{i}","from":"n{i}","to":"n{}","labels":["{t}"],"properties":{{"w":{}}}}}"#,
            (i * 7 + 1) % 60,
            i % 3
        ));
    }

    // SELF-LOOPS, deliberately. An undirected hop is where the two languages
    // genuinely differ — Gremlin walks a loop twice, GQL once — so without one
    // in the fixture nothing here can tell a correct `both` lowering from one
    // that silently adopts GQL's count. That is exactly what happened: `both()`
    // lowered from the first version of the compiler and answered 3 where the
    // answer is 4, and every test passed.
    for (i, v) in [(59, 0), (60, 4), (61, 8)] {
        let t = if v == 4 { "S" } else { "R" };

        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"n{v}","to":"n{v}","labels":["{t}"],"properties":{{"w":1}}}}"#
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
        // UNDIRECTED hops with a far-end constraint, so they actually COMPILE.
        // Without one they decline and the fixture's self-loops never reach the
        // lowering — which is how `both()` returned 3 where the answer is 4 for
        // as long as it did. Gremlin walks a loop twice, GQL once, and the shared
        // segment now carries the difference through `Ctx::loops`.
        "g.V().both('R').hasLabel('W')",
        "g.V().both('R').has('k', 3)",
        "g.V().hasLabel('P').both('R').hasLabel('W')",
        "g.V().out('R').both('S').hasLabel('W')",
        "g.V().both('R').both('S').hasLabel('P')",
        "g.V().bothE('R').otherV().has('k', 3)",
        // Undirected AND stopping on the edge: declines, because the planner
        // enumerates rather than expands there. Still asserted, for the answer.
        "g.V().bothE('R').has('w', 1)",
        "g.V().bothE('S').has('w', 0)",
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
        // Edge hops. `outE('R').inV()` IS `out('R')` — the same pattern with the
        // edge named — and spelled apart it can also stop ON the edge, which no
        // vertex hop expresses.
        "g.V().outE('R').inV().hasLabel('W')",
        "g.V().outE('R').inV().has('k', 3)",
        "g.V().inE('S').outV().hasLabel('P')",
        "g.V().bothE('R').otherV().hasLabel('W')",
        "g.V().outE('R','S').inV().hasLabel('W')",
        "g.V().outE().inV().hasLabel('W')",
        "g.V().outE('NOPE').inV().hasLabel('W')",
        // Stopping on the edge, with and without an edge filter.
        "g.V().outE('R')",
        "g.V().outE('R').has('w', 1)",
        "g.V().bothE('S').has('w', 0)",
        // An edge filter that does NOT end there.
        "g.V().outE('R').has('w', 1).inV().hasLabel('W')",
        // Two segments, the second ending on an edge.
        "g.V().out('R').outE('S').has('w', 0)",
        "g.V().outE('R').inV().outE('S').has('w', 1)",
        // Landings that are NOT the far end: `outV()` walks back to where the
        // edge came from and `bothV()` emits both ends. Neither is the segment a
        // pattern would compile them into, so both must decline to the stream.
        "g.V().outE('R').outV()",
        "g.V().outE('R').outV().hasLabel('W')",
        "g.V().inE('S').inV().hasLabel('P')",
        "g.V().bothE('R').bothV().hasLabel('W')",
        "g.V().outE('R').bothV()",
        // `repeat(<hops>).times(n)` is n segments. The fixture's self-loops are
        // what make this mean something: a repeat is a WALK, so traversing one
        // twice is a row, where GQL's `{2,2}` trail would drop it.
        "g.V().repeat(__.out('R')).times(2).hasLabel('W')",
        "g.V().repeat(__.out('R')).times(1).hasLabel('W')",
        "g.V().repeat(__.out('R')).times(3).hasLabel('W')",
        "g.V().repeat(__.out('R')).times(2).has('k', 3)",
        "g.V().hasLabel('P').repeat(__.out('R')).times(2).hasLabel('W')",
        "g.V().repeat(__.out('R').out('S')).times(2).hasLabel('W')",
        "g.V().out('S').repeat(__.out('R')).times(2).hasLabel('W')",
        "g.V().repeat(__.out('R')).times(2).out('S').hasLabel('W')",
        "g.V().repeat(__.outE('R').inV()).times(2).hasLabel('W')",
        "g.V().repeat(__.out()).times(2).hasLabel('W')",
        "g.V().repeat(__.out('R')).times(2).as('b').hasLabel('W').select('b')",
        // A repeat the pattern must NOT unroll: a per-traverser predicate, an
        // unbounded walk, or a body that is more than hops.
        "g.V().repeat(__.out('R')).until(__.hasLabel('W'))",
        "g.V().repeat(__.out('R')).times(2).emit().hasLabel('W')",
        "g.V().repeat(__.out('R').hasLabel('P')).times(2).hasLabel('W')",
        "g.V().repeat(__.out('R').simplePath()).times(2).hasLabel('W')",
        "g.V().repeat(__.both('R')).times(2).hasLabel('W')",
        "g.V().repeat(__.outE('R')).times(2).hasLabel('W')",
        "g.V().repeat(__.outE('R').outV()).times(2).hasLabel('W')",
        "g.V().as('x').as('x').out('R').hasLabel('W')",
        // `as(x)` is a var_slot like any other. It filters nothing, so it can sit
        // anywhere among the filters; stopping the prefix at one is what made
        // `V().as('a').out('R').hasLabel('W')` decline with zero segments.
        "g.V().as('a').out('R').hasLabel('W')",
        "g.V().out('R').as('b').hasLabel('W')",
        "g.V().out('R').hasLabel('W').as('b')",
        "g.V().as('a').out('R').as('b').hasLabel('W')",
        "g.V().as('a').hasLabel('P').out('R').hasLabel('W')",
        "g.V().hasLabel('P').as('a').out('R').as('b').has('k', 3)",
        "g.V().as('a').out('R').as('b').out('S').as('c').hasLabel('P')",
        "g.V().outE('R').as('e').inV().hasLabel('W')",
        "g.V().outE('R').as('e').inV().as('b').hasLabel('W')",
        // The tag READ back, which is the whole point of binding it.
        "g.V().as('a').out('R').hasLabel('W').select('a')",
        "g.V().as('a').out('R').as('b').hasLabel('W').select('a','b')",
        "g.V().as('a').out('R').hasLabel('W').select('a').values('k')",
        "g.V().as('a').out('R').as('b').hasLabel('W').select('a','b').count()",
        "g.V().outE('R').as('e').inV().hasLabel('W').select('e')",
        "g.V().as('a').out('R').as('b').out('S').hasLabel('P').select('a','b')",
        // Two labels on ONE element, and a label reused later.
        "g.V().as('a').as('x').out('R').hasLabel('W').select('a','x')",
        // `where(eq('a'))` and `math('a + b')` resolve a LABEL against the tag
        // map exactly as `select` does. Both were missing from the tag list while
        // being listed correctly for paths, so a lowered prefix dropped the
        // bindings out from under them and they matched nothing.
        "g.V().as('a').out('R').hasLabel('W').where(eq('a'))",
        "g.V().as('a').out('R').hasLabel('W').where(neq('a'))",
        "g.V().as('a').out('R').as('b').hasLabel('W').where('a', neq('b'))",
        // `dedup(labels)` keys on TAGS, so the tags have to be there.
        "g.V().as('a').out('R').hasLabel('W').dedup('a')",
        "g.V().as('a').out('R').as('b').hasLabel('W').dedup('a','b')",
        // A `by()` body runs on a FRESH root (see `eval_by`), so `by(__.path())`
        // yields the selected value's own one-element path — not the outer
        // traverser's. Pinned because it reads like it should do the opposite.
        "g.V().as('a').out('R').hasLabel('W').select('a').by(__.path())",
        "g.V().as('a').out('R').hasLabel('W').select('a').by(__.values('k'))",
        // A non-equality edge filter stays out, same as on a node.
        "g.V().outE('R').has('w', gt(0))",
        "g.V().outE('R').has('w', gt(0)).inV().hasLabel('W')",
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

/// Planning is for the traversals whose selective end is NOT where written
/// order would seed.
///
/// Both sides of this are invisible to a result comparison — the rows are the
/// same either way, only the cost differs — so it takes its own test. Compiling
/// a start-only-constrained traversal cost 1.28x on
/// `V().hasLabel(P).out(KNOWS).values(name)`: a full pattern scan and a
/// multi-slot frame, to arrive at the seed the step list already named.
#[test]
fn only_a_constraint_past_the_start_is_worth_planning() {
    let compiles = |src: &str| {
        let t = super::parse::parse(src).unwrap_or_else(|e| panic!("`{src}` parses: {e}"));

        super::pattern::compile(&t.steps).is_some()
    };

    for q in [
        // Nothing past the start to orient toward — written order already seeds
        // the only constrained end.
        "g.V().hasLabel('P').out('R')",
        "g.V().has('k', 1).out('R')",
        "g.V().hasLabel('P').has('k', 1).out('R').out('S')",
        // Nothing constrained anywhere.
        "g.V().out('R')",
        "g.V().out('R').out('S')",
        // A typed hop is not something to orient TOWARD: both paths push the
        // type into the adjacency already.
        "g.V().hasLabel('P').out('KNOWS')",
    ] {
        assert!(
            !compiles(q),
            "`{q}` has nothing past the start to orient to"
        );
    }

    for q in [
        "g.V().out('R').hasLabel('W')",
        "g.V().out('R').has('k', 3)",
        "g.V().hasLabel('P').out('R').hasLabel('Q')",
        // The constraint is on the SECOND hop's node, two segments out.
        "g.V().out('R').out('S').hasLabel('P')",
        // `repeat(<hops>).times(n)` unrolls to n segments.
        "g.V().repeat(__.out('R')).times(2).hasLabel('W')",
        "g.V().repeat(__.out('R').out('S')).times(2).hasLabel('W')",
        "g.V().repeat(__.outE('R').inV()).times(2).has('k', 3)",
        "g.V().repeat(__.out('R')).times(2).as('b').hasLabel('W')",
        // `as(x)` binds a slot and no longer stops the prefix.
        "g.V().as('a').out('R').hasLabel('W')",
        "g.V().out('R').as('b').hasLabel('W')",
        "g.V().as('a').out('R').as('b').hasLabel('W')",
        "g.V().as('a').out('R').as('b').hasLabel('W').select('a','b')",
        "g.V().outE('R').as('e').inV().hasLabel('W')",
        // Edge hops: `outE('R').inV().hasLabel('W')` is `out('R').hasLabel('W')`
        // with the edge named, and an edge PROPERTY is worth orienting to on its
        // own because the planner can seek an edge property index.
        "g.V().outE('R').inV().hasLabel('W')",
        "g.V().inE('R').outV().has('k', 3)",
        "g.V().outE('R').has('w', 1)",
        "g.V().outE('R').has('w', 1).inV()",
        "g.V().out('R').outE('S').has('w', 0)",
        // UNDIRECTED hops lower, because `Ctx::loops` carries Gremlin's
        // self-loop contract into the shared segment. These are the shapes that
        // silently returned GQL's count until it did.
        "g.V().both('R').hasLabel('W')",
        "g.V().both('R').has('k', 3)",
        "g.V().hasLabel('P').both('R').hasLabel('W')",
        "g.V().bothE('R').otherV().hasLabel('W')",
    ] {
        assert!(compiles(q), "`{q}` has a far constraint worth orienting to");
    }
}

/// `count()` and `dedup().count()` over a PLANNED frontier, against the stream.
///
/// Both read the answer straight off the id list rather than building a
/// traverser per element — `out('R').hasLabel('W').count()` spent 0.59ms of its
/// 0.70ms constructing 15,000 `Trav`s to fold them away. The risk in reading an
/// answer off the frontier instead of running the step is that the frontier is
/// not what the step would have seen, so every shape here is asserted against
/// `barrier()`, which keeps the same rows and defeats the compile.
#[test]
fn a_lowered_count_matches_the_streamed_one() {
    // Explicit, because a generated one was too weak to discriminate: nine `W`
    // nodes carrying only THREE distinct `k` values, reached with fan-in. So
    // `count` (11), `dedup().count()` (9) and `dedup().by('k').count()` (3) are
    // three different numbers, and an arm that confuses them shows up.
    let mut lines: Vec<String> = Vec::new();

    for i in 0..4 {
        lines.push(format!(
            r#"{{"type":"node","id":"s{i}","labels":["S"],"properties":{{"k":{}}}}}"#,
            100 + i
        ));
    }

    for i in 0..9 {
        lines.push(format!(
            r#"{{"type":"node","id":"w{i}","labels":["P","W"],"properties":{{"k":{}}}}}"#,
            i % 3
        ));
    }

    // w0 and w1 are each reached twice — without that, `count` and
    // `dedup().count()` agree by accident.
    let r_edges = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 3),
        (1, 4),
        (2, 1),
        (2, 5),
        (3, 6),
        (3, 7),
        (3, 8),
    ];

    for (i, (from, to)) in r_edges.iter().enumerate() {
        lines.push(format!(
            r#"{{"type":"edge","id":"r{i}","from":"s{from}","to":"w{to}","labels":["R"],"properties":{{}}}}"#
        ));
    }

    // A second hop, also with fan-in, for the two-segment shapes.
    for i in 0..9 {
        lines.push(format!(
            r#"{{"type":"edge","id":"t{i}","from":"w{i}","to":"w{}","labels":["S"],"properties":{{}}}}"#,
            (i + 1) % 4
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let val = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    for q in [
        "g.V().out('R').hasLabel('W').count()",
        "g.V().out('R').hasLabel('W').dedup().count()",
        "g.V().out('R').has('k', 1).count()",
        "g.V().out('R').has('k', 1).dedup().count()",
        // Two hops, so the duplicates compound.
        "g.V().out('R').out('S').hasLabel('W').count()",
        "g.V().out('R').out('S').hasLabel('W').dedup().count()",
        // Matching nothing must count 0, not decline into a wrong answer.
        "g.V().out('R').hasLabel('NOPE').count()",
        "g.V().out('R').hasLabel('NOPE').dedup().count()",
        // `dedup().by(k)` keys on a PROPERTY, not on element identity, so the
        // bare-dedup arm must not take it. Several elements share each `k`.
        //
        // These pin the ANSWER, not the arm: a `by()` makes the traversal
        // path-bound, so `TRACK_PATH` keeps the pattern branch out and they
        // never reach `column_paths` at all. Dropping the arm's own
        // `bys.is_empty()` guard therefore does NOT fail this test — verified.
        "g.V().out('R').hasLabel('W').dedup().by('k').count()",
        "g.V().out('R').has('k', 1).dedup().by('k').count()",
        "g.V().out('R').out('S').hasLabel('W').dedup().by('k').count()",
        // `count(local)` is per row and must stay per row.
        "g.V().out('R').hasLabel('W').count(local)",
    ] {
        let streamed = q.replacen("g.V()", "g.V().barrier()", 1);

        assert_eq!(
            val(q, &mut g),
            val(&streamed, &mut g),
            "`{q}` answered from the frontier disagrees with the streamed traversal"
        );
    }

    // The fixture has to actually exercise the differences, or each arm is
    // asserting against itself. All three must be distinct numbers.
    let counts = [
        val("g.V().out('R').hasLabel('W').count()", &mut g),
        val("g.V().out('R').hasLabel('W').dedup().count()", &mut g),
        val(
            "g.V().out('R').hasLabel('W').dedup().by('k').count()",
            &mut g,
        ),
    ];

    assert_eq!(
        counts,
        ["[Num(11.0)]", "[Num(9.0)]", "[Num(3.0)]"].map(String::from),
        "fixture no longer separates count / dedup / dedup-by, so the arms \
         above would agree by accident"
    );
}

/// What an UNDIRECTED hop means in each language, stated as a number.
///
/// Gremlin's `both('R')` and GQL's `MATCH (a)-[:R]-(b)` agree on every ordinary
/// edge, which is what makes them look like one operation. On a SELF-LOOP they
/// do not: Gremlin traverses the loop twice, because it is an out-edge and an
/// in-edge of the same vertex, and GQL yields it once. Over a two-vertex graph
/// with one loop that is 4 against 3.
///
/// Neither is wrong — undirected traversal is a per-language contract, the same
/// category as ordering and equality. What that does NOT mean is that the shared
/// segment cannot carry it: `crate::seek::SelfLoops` had already parameterized
/// exactly this, with its two variants named for the two languages, and nothing had
/// plumbed it. `Ctx::loops` now does, so `both()` lowers and each language keeps its
/// own answer.
///
/// The finding this pins is that the two answers differ at all — which is what makes
/// a lowering that adopts the wrong one invisible. `both()` lowered from the first
/// version of the pattern compiler and returned 3 where Gremlin's answer is 4, and
/// every test passed, because no fixture had a self-loop.
#[test]
fn an_undirected_hop_counts_a_self_loop_differently_in_each_language() {
    let grem = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };
    let gql = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            crate::gql::prepare(src)
                .unwrap_or_else(|e| panic!("`{src}` plans: {e}"))
                .execute(g, &crate::gql::eval::Params::new())
                .unwrap_or_else(|e| panic!("`{src}` runs: {e}"))
                .rows()
                .next()
                .map(<[_]>::to_vec)
        )
    };
    let fixture = |loops: bool| {
        let mut l = vec![
            r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#.to_string(),
            r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#.to_string(),
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{}}"#
                .to_string(),
        ];

        if loops {
            l.push(
                r#"{"type":"edge","id":"e1","from":"a","to":"a","labels":["R"],"properties":{}}"#
                    .to_string(),
            );
        }

        crate::ndjson::decode(&l.join(
            "
",
        ))
        .expect("fixture decodes")
    };

    let mut looped = fixture(true);

    assert_eq!(grem("g.V().both('R').count()", &mut looped), "[Num(4.0)]");
    assert_eq!(
        gql("MATCH (a)-[:R]-(b) RETURN count(*) AS c", &mut looped),
        "Some([Num(3.0)])"
    );

    // The DIRECTED spellings agree on the same graph, which is what narrows the
    // finding to undirected traversal rather than to self-loops generally.
    assert_eq!(grem("g.V().out('R').count()", &mut looped), "[Num(2.0)]");
    assert_eq!(
        gql("MATCH (a)-[:R]->(b) RETURN count(*) AS c", &mut looped),
        "Some([Num(2.0)])"
    );

    // Without the loop the undirected spellings agree — the trap.
    let mut plain = fixture(false);

    assert_eq!(grem("g.V().both('R').count()", &mut plain), "[Num(2.0)]");
    assert_eq!(
        gql("MATCH (a)-[:R]-(b) RETURN count(*) AS c", &mut plain),
        "Some([Num(2.0)])"
    );
}

/// Dropping the traverser path never changes an answer.
///
/// `path_free` is an allowlist, and an allowlist is a claim: this step neither
/// reads `Trav::path` nor nests something that might. Nothing tested the claim.
/// The planned-vs-streamed comparison CANNOT — it runs the same step list twice,
/// so both spellings reach the same wrong conclusion and agree with each other.
///
/// `with_forced_path` is the oracle that can: run the traversal again with the
/// accumulation pinned on. If the answers differ, the analysis dropped a path
/// something reads, and the step that allowed it does not belong in the list.
///
/// Every shape below reaches the allowlist through a DIFFERENT arm — a bare
/// terminal, a `by()` modulator, a nested sub-traversal, a repeat body, and the
/// two steps this round added (`as` and `select`).
#[test]
fn dropping_the_path_never_changes_an_answer() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N","W"],"properties":{"k":1}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"k":2}}"#,
            r#"{"type":"node","id":"c","labels":["N","W"],"properties":{"k":1}}"#,
            r#"{"type":"node","id":"d","labels":["M"],"properties":{"k":3}}"#,
            r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{"w":1}}"#,
            r#"{"type":"edge","id":"e1","from":"b","to":"c","labels":["R"],"properties":{"w":2}}"#,
            r#"{"type":"edge","id":"e2","from":"c","to":"d","labels":["S"],"properties":{"w":3}}"#,
            r#"{"type":"edge","id":"e3","from":"a","to":"c","labels":["S"],"properties":{"w":4}}"#,
            r#"{"type":"edge","id":"e4","from":"c","to":"c","labels":["R"],"properties":{"w":5}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

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
        // Steps that genuinely read the path: these must AGREE because the
        // analysis keeps the path for them, and they are what proves the oracle
        // is wired up at all.
        "g.V().out('R').path()",
        "g.V().out('R').out('R').simplePath()",
        "g.V().out('R').out('R').cyclicPath()",
        "g.V().bothE('R').otherV()",
        "g.V().repeat(__.out('R')).times(2).path()",
        // The allowlist's own entries, one per arm.
        "g.V().out('R').hasLabel('W')",
        "g.V().out('R').values('k')",
        "g.V().out('R').count()",
        "g.V().out('R').dedup().count()",
        "g.V().out('R').order().by('k')",
        "g.V().out('R').fold()",
        "g.V().hasLabel('N').groupCount().by('k')",
        "g.V().hasLabel('N').group().by('k').by('k')",
        "g.V().hasLabel('N').project('x').by('k')",
        "g.V().out('R').limit(2).values('k')",
        "g.V().out('R').aggregate('s').cap('s')",
        // `as` and `select`, added to the allowlist this round.
        "g.V().as('a').out('R').hasLabel('W').select('a')",
        "g.V().as('a').out('R').as('b').select('a','b')",
        "g.V().as('a').out('R').hasLabel('W').select('a').by('k')",
        "g.V().as('a').out('R').hasLabel('W').select('a').by(__.path())",
        "g.V().as('a').out('R').hasLabel('W').dedup('a')",
        "g.V().as('a').out('R').select('a').by(__.out('S').count())",
        // `dedup().by(<sub>)` — the modulator, like every other, runs on a fresh
        // root, so it cannot read the outer path either.
        "g.V().out('R').dedup().by(__.path())",
        "g.V().out('R').dedup().by(__.out('S').count())",
        "g.V().out('R').dedup().by('k').values('k')",
        // Sub-traversals, which the allowlist admits only when their bodies are
        // path-free — so each of these is really a test of the recursion.
        "g.V().where(__.out('R')).values('k')",
        "g.V().not(__.out('R')).values('k')",
        "g.V().union(__.out('R'), __.out('S')).values('k')",
        "g.V().choose(__.hasLabel('W'), __.out('R'), __.out('S')).values('k')",
        "g.V().coalesce(__.out('R'), __.out('S')).values('k')",
        "g.V().local(__.out('R')).values('k')",
        "g.V().repeat(__.out('R')).times(2).values('k')",
        // A sub-traversal that DOES read the path — the recursion must keep it.
        "g.V().where(__.out('R').simplePath()).values('k')",
        "g.V().local(__.out('R').path()).count()",
        "g.V().repeat(__.out('R').simplePath()).times(2).count()",
    ] {
        assert_eq!(
            rows(q, &mut g),
            super::exec::with_forced_path(|| rows(q, &mut g)),
            "`{q}` answers differently once the path is kept, so `path_free` \
             dropped history it reads"
        );
    }
}

/// Paging a column agrees with paging the stream.
///
/// `values(k).limit(n)` had no arm in either column terminal, so it declined
/// both and streamed every row it was about to discard — 32.8ms against 1.2ms
/// for the same query with NO limit. A limit made it 28x slower than no limit,
/// which is the shape of a missing arm rather than a slow one.
///
/// The oracle is `.identity()` before the paging step: it changes nothing and
/// makes the tail a shape neither terminal recognizes, so the same question runs
/// down both paths.
#[test]
fn paging_a_column_agrees_with_paging_the_stream() {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..40 {
        // A gap in `n` and a gap in `s`, at different indices: paging counts
        // VALUES, and `values(k)` skips the elements that lack `k`.
        let mut props = Vec::new();

        if i % 5 != 0 {
            props.push(format!(r#""n":{}"#, i % 7));
        }
        if i % 4 != 0 {
            props.push(format!(r#""s":"v{}""#, i % 3));
        }

        lines.push(format!(
            r#"{{"type":"node","id":"n{i}","labels":["P"],"properties":{{{}}}}}"#,
            props.join(",")
        ));
    }

    for i in 0..39 {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"n{i}","to":"n{}","labels":["R"],"properties":{{}}}}"#,
            (i * 7 + 1) % 40
        ));
    }

    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let rows = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    for (col, page) in [("n", "tail(5)"), ("s", "tail(5)")].into_iter().chain([
        ("n", "skip(5)"),
        ("s", "skip(5)"),
        ("n", "range(2, 7)"),
        ("s", "range(2, 7)"),
        // Degenerate bounds: zero, past the end, and an empty window.
        ("n", "limit(0)"),
        ("n", "skip(1000)"),
        ("n", "range(5, 5)"),
        ("n", "range(1000, 1001)"),
        ("s", "limit(0)"),
        ("s", "skip(1000)"),
        // More than the column holds.
        ("n", "limit(1000)"),
    ]) {
        for src in [
            format!("g.V().values('{col}').{page}"),
            format!("g.V().hasLabel('P').out('R').values('{col}').{page}"),
            // A key nothing carries: the column is empty and paging it is still
            // whatever paging an empty stream is.
            format!("g.V().values('nope').{page}"),
        ] {
            let slow = src.replace(&format!(".{page}"), &format!(".identity().{page}"));

            assert_ne!(slow, src, "the oracle did not rewrite `{src}`");
            assert_eq!(
                rows(&src, &mut g),
                rows(&slow, &mut g),
                "`{src}` paged from the column disagrees with paging the stream"
            );
        }
    }

    // COMPOSITIONS. Each part had an arm; the combination had none, so it
    // streamed the whole traversal — `values('n').order().by(desc).count()` cost
    // 58x the same query without the count. The terminal peels one step and
    // recurses, so these are the shapes that peel more than once.
    for src in [
        "g.V().values('n').dedup().limit(5)",
        "g.V().values('n').limit(5).count()",
        "g.V().values('n').order().by(desc).count()",
        "g.V().values('n').order().limit(3).count()",
        "g.V().values('n').skip(2).limit(3)",
        "g.V().values('n').limit(10).skip(2).count()",
        "g.V().values('n').tail(5).count()",
        "g.V().values('n').dedup().count()",
        "g.V().values('n').dedup().order().limit(2)",
        "g.V().values('n').range(2, 9).dedup().count()",
        "g.V().values('n').order().by(desc).limit(4).sum()",
        "g.V().values('n').limit(6).fold()",
        "g.V().values('n').dedup().limit(3).max()",
        "g.V().values('s').dedup().limit(5)",
        "g.V().values('s').limit(5).count()",
        "g.V().values('s').tail(5).count()",
        "g.V().values('s').skip(2).limit(3)",
        "g.V().values('s').range(1, 8).dedup().count()",
        "g.V().hasLabel('P').out('R').values('n').dedup().limit(4).count()",
    ] {
        // `.identity()` right after `values(...)`: the tail now STARTS with a
        // step neither terminal recognizes, so nothing composes and the whole
        // thing streams — the same question down the other path.
        let slow = src
            .replace("values('n')", "values('n').identity()")
            .replace("values('s')", "values('s').identity()");

        assert_ne!(slow, src, "the oracle did not rewrite `{src}`");
        assert_eq!(
            rows(src, &mut g),
            rows(&slow, &mut g),
            "`{src}` composed off the column disagrees with the stream"
        );
    }

    // `local` pages WITHIN each value and must not take the column arm.
    for src in [
        "g.V().values('n').limit(local, 5)",
        "g.V().values('n').range(local, 0, 2)",
    ] {
        let slow = src.replace(".limit(local", ".identity().limit(local");
        let slow = slow.replace(".range(local", ".identity().range(local");

        assert_eq!(
            rows(src, &mut g),
            rows(&slow, &mut g),
            "`{src}` local paging"
        );
    }
}

/// `where(<one hop>)` answered from the adjacency agrees with running the body.
///
/// The oracle is `.identity()` appended to the body: it means exactly the same
/// thing and makes the body two steps, which the shortcut refuses — so each pair
/// is the same question asked of both paths. (`barrier()` cannot serve here; it
/// defeats the PATTERN compile, not this.)
#[test]
fn a_single_hop_where_agrees_with_running_it() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"k":1}}"#,
            r#"{"type":"node","id":"b","labels":["N","W"],"properties":{"k":2}}"#,
            r#"{"type":"node","id":"c","labels":["M"],"properties":{"k":3}}"#,
            r#"{"type":"node","id":"d","labels":["N"],"properties":{"k":4}}"#,
            r#"{"type":"edge","id":"r0","from":"a","to":"b","labels":["X","Y"],"properties":{}}"#,
            r#"{"type":"edge","id":"r1","from":"b","to":"c","labels":["Y"],"properties":{}}"#,
            r#"{"type":"edge","id":"r2","from":"c","to":"a","labels":["Z"],"properties":{}}"#,
            // A self-loop, since `both()` sees one from both sides.
            r#"{"type":"edge","id":"r3","from":"d","to":"d","labels":["Y"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

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

    for body in [
        // Every direction, in both the vertex and the edge spelling — for
        // EXISTENCE the two ask the same question.
        "__.out('Y')",
        "__.outE('Y')",
        "__.in('Y')",
        "__.inE('Y')",
        "__.both('Y')",
        "__.bothE('Y')",
        // Untyped, a disjunction, and a type that resolves to NOTHING — which
        // must match nothing rather than everything, the conflation this
        // codebase has written five times.
        "__.out()",
        "__.outE()",
        "__.out('X','Y')",
        "__.out('NOPE')",
        "__.outE('NOPE')",
        "__.out('NOPE','Y')",
        // Bodies the shortcut must refuse: something after the hop, several
        // hops, or a filter.
        "__.out('Y').hasLabel('W')",
        "__.out('Y').out('Y')",
        "__.out('Y').has('k', 2)",
        "__.not(__.out('Y'))",
    ] {
        for q in [
            format!("g.V().where({body})"),
            // `not()` asks the same question, negated — including for a
            // non-vertex traverser, which HAS no such edge and so survives it.
            format!("g.V().not({body})"),
            format!("g.E().not({body})"),
            format!("g.V().hasLabel('N').not({body}).values('k')"),
            format!("g.V().hasLabel('N').where({body}).values('k')"),
            format!("g.V().where({body}).count()"),
            // A non-vertex traverser: the hop yields nothing, so the filter drops
            // it — the shortcut has to agree about that too.
            format!("g.E().where({body})"),
        ] {
            // Rewrite the BODY, not the step, so `not(…)` gets the same oracle
            // as `where(…)`. Keying on `where(` compared every `not` shape
            // against ITSELF — a vacuous pass that reads exactly like a real one.
            let slow = q.replace(&format!("({body})"), &format!("({body}.identity())"));

            assert_ne!(slow, q, "the oracle did not rewrite `{q}`");

            assert_eq!(
                rows(&q, &mut g),
                rows(&slow, &mut g),
                "`{q}` disagrees with running the body"
            );
        }
    }
}

/// Median-of-5 seconds for one run of `src`.
fn grem_time(g: &mut Graph, src: &str) -> f64 {
    let plan = super::parse::parse(src).unwrap_or_else(|e| panic!("`{src}` parses: {e}"));
    let mut best = f64::MAX;

    for _ in 0..5 {
        let t = std::time::Instant::now();
        let out = plan.clone().run(g);
        let secs = t.elapsed().as_secs_f64();

        std::hint::black_box(out.len());
        if secs < best {
            best = secs;
        }
    }

    best
}

/// Traversals that mean the same thing must cost the same.
///
/// The Gremlin counterpart of `gql::index_seed_tests::
/// equivalent_spellings_cost_the_same`, and it exists because every lowering
/// this engine gained arrived as a spelling that was 100x off its twin while
/// returning identical rows — which no correctness test can catch:
///
///   `outE('R').inV()`            was 455x `out('R')`, a missing `path_free` entry
///   `as('a').out('R')…`          was 188x the untagged spelling
///   `repeat(out('R')).times(2)`  was 128x the written-out two hops
///
/// Each of those was found by hand, one at a time, after the fact. Grouped here,
/// the next one fails a test instead.
///
/// Ignored like its GQL twin: it is timing-sensitive and the ratios need a
/// quiet machine. Run with `--ignored --nocapture`.
#[test]
#[ignore = "timing-sensitive; run with --ignored --nocapture"]
fn equivalent_traversals_cost_the_same() {
    // Generous, and still 30x tighter than every gap above. This is a bound on
    // "one spelling plans and the other enumerates", not a microbenchmark.
    const MAX_RATIO: f64 = 4.0;

    let mut lines = String::new();

    for i in 0..20_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };

        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{}}}}}\n",
            i % 97
        ));
    }

    let mut e = 0;

    for i in 0..20_000usize {
        for d in 0..3usize {
            lines.push_str(&format!(
                "{{\"type\":\"edge\",\"id\":\"e{e}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{d}}}}}\n",
                (i * 31 + d * 7 + 1) % 20_000
            ));
            e += 1;
        }
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    let groups: &[(&str, &[&str])] = &[
        (
            "a far-end label decides the seed",
            &[
                "g.V().out('R').hasLabel('W').count()",
                "g.V().outE('R').inV().hasLabel('W').count()",
                "g.V().as('a').out('R').hasLabel('W').count()",
                "g.V().out('R').as('b').hasLabel('W').count()",
                "g.V().repeat(__.out('R')).times(1).hasLabel('W').count()",
            ],
        ),
        (
            "two hops to a labelled end",
            &[
                "g.V().out('R').out('R').hasLabel('W').count()",
                "g.V().repeat(__.out('R')).times(2).hasLabel('W').count()",
                "g.V().outE('R').inV().outE('R').inV().hasLabel('W').count()",
            ],
        ),
        (
            "values off a far-end-filtered hop",
            &[
                "g.V().out('R').hasLabel('W').values('n')",
                "g.V().outE('R').inV().hasLabel('W').values('n')",
                "g.V().as('a').out('R').hasLabel('W').values('n')",
            ],
        ),
        (
            "a far-end property decides the seed",
            &[
                "g.V().out('R').has('n', 7).count()",
                "g.V().outE('R').inV().has('n', 7).count()",
                "g.V().repeat(__.out('R')).times(1).has('n', 7).count()",
            ],
        ),
    ];
    let mut failures: Vec<String> = Vec::new();

    for (name, queries) in groups {
        let expect = {
            let mut v: Vec<String> = super::parse::parse(queries[0])
                .expect("parses")
                .run(&mut g)
                .iter()
                .map(|x| format!("{x:?}"))
                .collect();
            v.sort();
            v
        };

        for q in *queries {
            let mut got: Vec<String> = super::parse::parse(q)
                .unwrap_or_else(|e| panic!("`{q}` parses: {e}"))
                .run(&mut g)
                .iter()
                .map(|x| format!("{x:?}"))
                .collect();
            got.sort();

            assert_eq!(
                got, expect,
                "[{name}] `{q}` disagreed with `{}`",
                queries[0]
            );
        }

        let times: Vec<f64> = queries.iter().map(|q| grem_time(&mut g, q)).collect();
        let fastest = times.iter().copied().fold(f64::MAX, f64::min);

        for (q, t) in queries.iter().zip(&times) {
            let ratio = t / fastest;

            println!("  {ratio:>6.1}x  [{name}] {q}");

            if ratio > MAX_RATIO {
                failures.push(format!(
                    "[{name}] {ratio:.0}x slower than the best spelling in its group:\n    {q}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "\ntraversals that mean the same thing but do not cost the same:\n\n{}\n",
        failures.join("\n\n")
    );
}
