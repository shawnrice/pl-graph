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
