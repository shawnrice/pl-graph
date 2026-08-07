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

/// One question, written in GQL and in Gremlin — the table two probes read.
///
/// `cross_language_cost_probe` prices the pair; `the_two_languages_compute_the_same_values`
/// checks the two sides return the same answers. It lives here rather than inside
/// either because a second copy is exactly the drift this file exists to catch:
/// the two probes disagreeing about what "the same question" means would make
/// both of them lie.
///
/// `(name, gql, gremlin)`. Add a pair whenever a question is expressible in both.
const CROSS_LANGUAGE_PAIRS: &[(&str, &str, &str)] = &[
    (
        "any edge of a type exists",
        "MATCH ()-[:R]->() RETURN 1 AS x LIMIT 1",
        "g.E().hasLabel('R').limit(1).count()",
    ),
    (
        "tally a hop by an endpoint property",
        "MATCH ()-[:R]->(b) RETURN b.n AS k, count(*) AS c GROUP BY b.n",
        "g.V().out('R').groupCount().by('n')",
    ),
    (
        "top-k by a property",
        "MATCH (u:V) RETURN u.n AS n ORDER BY u.n DESC LIMIT 10",
        "g.V().hasLabel('V').order().by('n', desc).limit(10).values('n')",
    ),
    (
        "count of a 2-hop",
        "MATCH ()-[:R]->()-[:R]->() RETURN count(*) AS c",
        "g.V().out('R').out('R').count()",
    ),
    (
        "a join anchored on an indexed key",
        "MATCH (u:V)-[:R]->(x) WHERE u.k = 'key000005' RETURN count(*) AS c",
        "g.V().hasLabel('V').has('k', 'key000005').out('R').count()",
    ),
    (
        "distinct far ends of a hop",
        "MATCH ()-[:R]->(b) RETURN count(DISTINCT b) AS c",
        "g.V().out('R').dedup().count()",
    ),
    (
        "sum a property over a hop",
        "MATCH ()-[:R]->(b) RETURN sum(b.n) AS s",
        "g.V().out('R').values('n').sum()",
    ),
    (
        "max a property over all vertices",
        "MATCH (u:V) RETURN max(u.n) AS m",
        "g.V().hasLabel('V').values('n').max()",
    ),
    (
        "count vertices by label",
        "MATCH (u:W) RETURN count(*) AS c",
        "g.V().hasLabel('W').count()",
    ),
    (
        "tally by label of the far end",
        "MATCH ()-[:R]->(b) RETURN count(*) AS c",
        "g.V().out('R').count()",
    ),
    (
        "edge property tally",
        "MATCH ()-[r:R]->() RETURN r.w AS w, count(*) AS c GROUP BY r.w",
        "g.E().hasLabel('R').groupCount().by('w')",
    ),
    (
        "values of a property, all vertices",
        "MATCH (u:V) RETURN u.n AS n",
        "g.V().hasLabel('V').values('n')",
    ),
    (
        "two properties, all vertices",
        "MATCH (u:V) RETURN u.n AS n, u.k AS k",
        "g.V().hasLabel('V').project('n','k').by('n').by('k')",
    ),
    (
        "the elements themselves",
        "MATCH (u:V) RETURN u",
        "g.V().hasLabel('V')",
    ),
    (
        "the elements, rendered",
        "MATCH (u:V) RETURN u",
        "g.V().hasLabel('V').elementMap()",
    ),
    (
        "the edges, rendered",
        "MATCH ()-[r:R]->() RETURN r",
        "g.E().hasLabel('R').elementMap()",
    ),
    (
        "does any edge of a type exist",
        "MATCH ()-[:R]->() RETURN 1 AS x LIMIT 1",
        "g.E().hasLabel('R').limit(1).count()",
    ),
    (
        "a property of a filtered scan",
        "MATCH (u:V) WHERE u.n > 50 RETURN u.n AS n",
        "g.V().hasLabel('V').has('n', gt(50)).values('n')",
    ),
    (
        "a string property, all vertices",
        "MATCH (u:V) RETURN u.k AS k",
        "g.V().hasLabel('V').values('k')",
    ),
    (
        "a range predicate",
        "MATCH (u:V) WHERE u.n >= 20 AND u.n < 60 RETURN count(*) AS c",
        "g.V().hasLabel('V').has('n', between(20, 60)).count()",
    ),
    (
        "an OR of two equalities",
        "MATCH (u:V) WHERE u.n = 3 OR u.n = 9 RETURN count(*) AS c",
        "g.V().hasLabel('V').or(__.has('n', 3), __.has('n', 9)).count()",
    ),
    (
        "a negated predicate",
        "MATCH (u:V) WHERE NOT u.n = 3 RETURN count(*) AS c",
        "g.V().hasLabel('V').not(__.has('n', 3)).count()",
    ),
    (
        "a not-equal predicate",
        "MATCH (u:V) WHERE u.n <> 3 RETURN count(*) AS c",
        "g.V().hasLabel('V').has('n', neq(3)).count()",
    ),
    (
        "does the key exist at all",
        "MATCH (u:V) WHERE u.n IS NOT NULL RETURN count(*) AS c",
        "g.V().hasLabel('V').has('n').count()",
    ),
    (
        "a string prefix scan",
        "MATCH (u:V) WHERE u.k = 'key000005' RETURN count(*) AS c",
        "g.V().hasLabel('V').has('k', 'key000005').count()",
    ),
    (
        "where on the element's own property",
        "MATCH (u:V) WHERE u.n > 50 RETURN count(*) AS c",
        "g.V().hasLabel('V').where(__.values('n').is(gt(50))).count()",
    ),
    (
        "not on the element's own property",
        "MATCH (u:V) WHERE NOT u.n > 50 RETURN count(*) AS c",
        "g.V().hasLabel('V').not(__.values('n').is(gt(50))).count()",
    ),
    (
        "a barrier in the middle",
        "MATCH (u:V) RETURN count(*) AS c",
        "g.V().hasLabel('V').barrier().count()",
    ),
    (
        "identity in the middle",
        "MATCH (u:V) RETURN u.n AS n",
        "g.V().hasLabel('V').identity().values('n')",
    ),
    (
        "where on an EDGE's own property",
        "MATCH ()-[r:R]->() WHERE r.w = 1 RETURN count(*) AS c",
        "g.E().hasLabel('R').where(__.values('w').is(1)).count()",
    ),
    (
        "edge endpoints off an E() frontier",
        "MATCH ()-[:R]->(b) RETURN count(*) AS c",
        "g.E().hasLabel('R').inV().count()",
    ),
    (
        "edge endpoint values",
        "MATCH ()-[:R]->(b) RETURN b.n AS n",
        "g.E().hasLabel('R').inV().values('n')",
    ),
    (
        "select over a hop (pattern tags)",
        "MATCH (u:V)-[:R]->(x) RETURN count(*) AS c",
        "g.V().as('x').out('R').select('x').count()",
    ),
    (
        "select with a by() modulator",
        "MATCH (u:V)-[:R]->(x) RETURN count(*) AS c",
        "g.V().as('x').out('R').select('x').by('n').count()",
    ),
    (
        "select with Pop.all",
        "MATCH (u:V)-[:R]->(x) RETURN count(*) AS c",
        "g.V().as('x').out('R').select(all, 'x').count()",
    ),
    (
        "select with no hop at all",
        "MATCH (u:V) RETURN count(*) AS c",
        "g.V().hasLabel('V').as('x').select('x').count()",
    ),
    (
        "not(hasLabel) on the element itself",
        "MATCH (u:V) WHERE NOT u:W RETURN count(*) AS c",
        "g.V().hasLabel('V').not(__.hasLabel('W')).count()",
    ),
    (
        "an unread as() tag",
        "MATCH (u:V) RETURN count(*) AS c",
        "g.V().hasLabel('V').as('x').count()",
    ),
    (
        "a READ as() tag",
        "MATCH (u:V) RETURN count(*) AS c",
        "g.V().hasLabel('V').as('x').select('x').count()",
    ),
    (
        "plain count (baseline)",
        "MATCH (u:V) RETURN count(*) AS c",
        "g.V().hasLabel('V').count()",
    ),
    (
        "fold().unfold() still streams",
        "MATCH (u:V) RETURN count(*) AS c",
        "g.V().hasLabel('V').fold().unfold().count()",
    ),
    (
        "unfold with no fold before it",
        "MATCH (u:V) RETURN count(*) AS c",
        "g.V().hasLabel('V').values('n').unfold().count()",
    ),
    (
        "filter on the far end of a hop",
        "MATCH ()-[:R]->(b) WHERE b.n = 7 RETURN count(*) AS c",
        "g.V().out('R').has('n', 7).count()",
    ),
    (
        "filter on an edge property",
        "MATCH ()-[r:R]->() WHERE r.w = 1 RETURN count(*) AS c",
        "g.E().hasLabel('R').has('w', 1).count()",
    ),
    (
        "sum over a grouped hop",
        "MATCH ()-[:R]->(b) RETURN b.n AS k, sum(b.n) AS s GROUP BY b.n",
        "g.V().out('R').group().by('n').by(__.values('n').sum())",
    ),
    (
        "the degree of every vertex",
        "MATCH (u:V) RETURN u.k AS k, count(*) AS c GROUP BY u.k",
        "g.V().hasLabel('V').groupCount().by('k')",
    ),
    (
        "order by a property, no limit",
        "MATCH (u:V) RETURN u.n AS n ORDER BY u.n",
        "g.V().hasLabel('V').order().by('n').values('n')",
    ),
    (
        "distinct property values",
        "MATCH (u:V) RETURN DISTINCT u.n AS n",
        "g.V().hasLabel('V').values('n').dedup()",
    ),
    (
        "count of a hop from a narrow seed",
        "MATCH (u:V)-[:R]->(x) WHERE u.k = 'key000005' RETURN count(*) AS c",
        "g.V().has('k', 'key000005').out('R').count()",
    ),
];

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

/// `id()` / `label()` on a PATH is an error, and one raised from the PLAN.
///
/// This reverses what this test used to assert. It required null, on the
/// grounds that the TS engine returned null too — which it did, so the engines
/// agreed with each other and with nothing else. TinkerPop types
/// `IdStep<S extends Element>` and does `traverser.get().id()`, so on a path the
/// erased generic gives a bare `ClassCastException` ("ImmutablePath cannot be
/// cast to ...Element"). A path has no id and no label; answering null said it
/// had one, and it was null.
///
/// Raised from the step list before any walk, because it is a property of the
/// plan: `path().id()` cannot succeed on any graph, so there is nothing to
/// evaluate. `DataException`, not `Syntax` (the traversal parses) and not
/// `InvalidValue` (a path is a perfectly good value that has no id) — the same
/// code the TS engine raises from the same check.
#[test]
fn id_of_a_path_faults_from_the_plan() {
    let mut graph = modern();

    for t in [g().E().path().id(), g().E().path().label()] {
        let err = super::try_run(&mut graph, &t).expect_err("a path has no id or label");

        assert_eq!(err.code, crate::error_codes::ErrorCode::DataException);
        assert!(
            err.message.contains("not an element"),
            "the message must say why: {}",
            err.message
        );
    }

    // Through steps that hand the value on unchanged, the path is still a path.
    assert!(super::try_run(&mut graph, &g().E().path().limit(2).id()).is_err());

    // But `unfold()` turns it into its ELEMENTS, and those do have ids — so the
    // check must not simply look for a later `id()`.
    let unfolded = super::try_run(&mut graph, &g().E().path().unfold().id())
        .expect("elements of a path have ids");

    assert!(
        unfolded.iter().all(|v| matches!(v, GVal::Str(_))),
        "got {unfolded:?}"
    );
}

/// The fault reaches the caller through whatever follows it.
///
/// This test previously asserted the OPPOSITE — that summing the ids of paths
/// was an all-null fold and explicitly "not a fault" — which is the decision
/// reversed above. A terminal downstream must not swallow it: the plan is
/// unsatisfiable whatever `sum()` would have done with the nulls.
#[test]
fn a_plan_fault_survives_the_steps_after_it() {
    let mut graph = modern();

    for t in [
        g().E().path().id().sum(),
        g().E().path().id().count(),
        g().E().path().id().fold(),
    ] {
        assert_eq!(
            super::try_run(&mut graph, &t).map(|_| ()).unwrap_err().code,
            crate::error_codes::ErrorCode::DataException
        );
    }

    // `run` is infallible and cannot say why, so it yields nothing rather than
    // an answer that was never computable.
    assert!(super::run(&mut graph, &g().E().path().id().sum()).is_empty());
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

/// A computed non-finite never reaches storage — it is coerced to null at the
/// write, in every shape that can produce one.
///
/// The codecs always said so ("storing a real non-finite would silently corrupt
/// count/sum/min/max/`IS NULL` and diverge from TS"), and `Column::Num` goes
/// further by using NaN as its own ABSENT sentinel. The *computed*-write paths
/// did not honor it: every one of these stored a live NaN in native while the
/// TS engine stored null. The differential fuzzer cannot see it because it
/// fuzzes reads.
///
/// This is also what keeps the lowered column terminals safe. They read
/// `Column::Num` directly, so "a numeric column never holds a NaN" is the
/// premise under which their ordering fast paths are allowed to skip the
/// NaN-vs-number question entirely.
#[test]
fn a_computed_non_finite_never_reaches_storage() {
    let gql = |g: &mut crate::graph::Graph, src: &str| {
        crate::gql::prepare(src)
            .unwrap_or_else(|e| panic!("`{src}` plans: {e}"))
            .execute(g, &crate::gql::eval::Params::new())
            .unwrap_or_else(|e| panic!("`{src}` runs: {e}"));
    };

    for (setup, write, read, want) in [
        // NaN, every element the write paths can touch.
        (
            "",
            "INSERT (:V {x: sqrt(-1)})",
            "MATCH (n:V) RETURN n.x",
            "[[Null]]",
        ),
        (
            "INSERT (:V {x: 1})",
            "MATCH (n:V) SET n.x = sqrt(-1)",
            "MATCH (n:V) RETURN n.x",
            "[[Null]]",
        ),
        (
            "INSERT (:A)-[:R {w: 1}]->(:B)",
            "MATCH ()-[e:R]->() SET e.w = sqrt(-1)",
            "MATCH ()-[e:R]->() RETURN e.w",
            "[[Null]]",
        ),
        // ±Infinity is the same rule. (`1e400` cannot get this far — the lexer
        // rejects the literal — and `x / 0` throws, so `exp` is the way in.)
        (
            "INSERT (:V {x: 1})",
            "MATCH (n:V) SET n.x = exp(1000)",
            "MATCH (n:V) RETURN n.x",
            "[[Null]]",
        ),
        // NESTED, which a scalar-only check would miss.
        (
            "",
            "INSERT (:V {x: [sqrt(-1), 2]})",
            "MATCH (n:V) RETURN n.x",
            "[[List([Null, Num(2.0)])]]",
        ),
        (
            "INSERT (:V {x: 1})",
            "MATCH (n:V) SET n.x = {a: sqrt(-1)}",
            "MATCH (n:V) RETURN n.x",
            "[[Map([(\"a\", Null)])]]",
        ),
    ] {
        let mut g = crate::ndjson::decode("").expect("an empty graph decodes");

        if !setup.is_empty() {
            gql(&mut g, setup);
        }

        gql(&mut g, write);

        let got = crate::gql::prepare(read)
            .expect("plans")
            .execute(&mut g, &crate::gql::eval::Params::new())
            .expect("runs");
        let got: Vec<Vec<_>> = got.rows().map(<[_]>::to_vec).collect();

        assert_eq!(format!("{got:?}"), want, "`{write}` stored a non-finite");
    }

    // The graph API is the same boundary — the FFI reaches it without a query.
    let mut g =
        crate::ndjson::decode(r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":3}}"#)
            .expect("fixture decodes");

    g.set_vertex_prop(0, "n", crate::graph::Value::Num(f64::NAN));

    assert_eq!(
        g.props.value(0, "n", &g.strs),
        crate::graph::Value::Null,
        "the graph setter stored a NaN"
    );
}

/// A NaN can still be COMPUTED into a stream — `math()` makes one — and there it
/// obeys the sort/aggregate policy: a TOTAL order with NaN greatest.
///
/// Both halves matter and each broke differently. `order()` disagreed across
/// engines (native put every NaN first, because `f64::total_cmp` is sign-aware
/// and `sqrt(-1)` is a NEGATIVE NaN; TS left them scattered, because its sort
/// comparator returned `NaN`, which `Array.sort` reads as 0). And `min()`
/// returned a NaN in native, because the partial comparator answers `None`
/// against one — never the wanted ordering — so whichever NaN arrived first held
/// `best` forever.
#[test]
fn a_computed_nan_sorts_last_and_never_wins_min() {
    let lines: Vec<String> = [-1, 4, -9, 9, 16, -4, 25, -16, 1, -25]
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(r#"{{"type":"node","id":"v{i}","labels":["V"],"properties":{{"m":{m}}}}}"#)
        })
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    // Five NaNs and five reals. NaN LAST under ASC...
    assert_eq!(
        run("g.V().values('m').math('sqrt _').order()", &mut g),
        "[Num(1.0), Num(2.0), Num(3.0), Num(4.0), Num(5.0), \
         Num(NaN), Num(NaN), Num(NaN), Num(NaN), Num(NaN)]"
    );

    // ...and FIRST under DESC, because NaN is the greatest VALUE rather than an
    // absolute-last like null. Reversing the comparator is all it takes.
    assert_eq!(
        run("g.V().values('m').math('sqrt _').order().by(desc)", &mut g),
        "[Num(NaN), Num(NaN), Num(NaN), Num(NaN), Num(NaN), \
         Num(5.0), Num(4.0), Num(3.0), Num(2.0), Num(1.0)]"
    );

    // The same order, read by the aggregates: greatest means `max` keeps it and
    // `min` never picks it.
    assert_eq!(
        run("g.V().values('m').math('sqrt _').max()", &mut g),
        "[Num(NaN)]"
    );
    assert_eq!(
        run("g.V().values('m').math('sqrt _').min()", &mut g),
        "[Num(1.0)]"
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
        // Navigating OFF an edge frontier. `lowered_ids` guards this with an
        // allowlist of what may follow one, because `lower_hops` would otherwise
        // read the hop as an expansion of EDGE ids through the VERTEX adjacency
        // — the right shape of answer, off the wrong array. Nothing tested it
        // until a mutation that admitted `out` to that list broke nothing.
        "g.E().out('R')",
        "g.E().in('R')",
        "g.E().hasLabel('R').out('R')",
        "g.E().outV().out('R')",
        "g.E().out('R').count()",
        "g.E().out('R').values('k')",
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
        // `barrier()` after the source keeps the same rows and defeats the
        // compile. It has to match the SOURCE — keying on `g.V()` alone left
        // every `g.E()` shape compared against itself, which is a pass that
        // looks exactly like a real one. The assert below is the cheap guard.
        let streamed = if q.starts_with("g.E()") {
            q.replacen("g.E()", "g.E().barrier()", 1)
        } else {
            q.replacen("g.V()", "g.V().barrier()", 1)
        };

        assert_ne!(streamed, q, "the oracle did not rewrite `{q}`");

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

/// An edge's id is the stored one, assigned or synthesized.
///
/// `element_id` rebuilt it with `Arc::from(graph.edge_id(e).as_ref())` — a fresh
/// allocation of a string that is ALREADY an `Arc<str>` in `eid_fwd`. A vertex
/// id has always been a refcount bump, which is why `g.E().id()` cost 47ns an
/// edge where `g.V().id()` cost 9ns a vertex. Handing the `Arc` back directly is
/// 2.4x, and this pins that it is the same string either way — including the
/// SYNTHESIZED `e{n}` form, which has no stored `Arc` to hand back.
#[test]
fn an_edge_id_is_the_stored_one_however_it_is_read() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
            // An ASSIGNED id, a canonical-looking assigned id, and one with no id
            // at all — the third gets `e{n}` synthesized on demand.
            r#"{"type":"edge","id":"pay-1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","id":"e7","from":"b","to":"a","labels":["R"],"properties":{}}"#,
            r#"{"type":"edge","from":"a","to":"a","labels":["R"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let ids = |src: &str, g: &mut crate::graph::Graph| {
        let mut v: Vec<String> = super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
            .iter()
            .map(|x| format!("{x:?}"))
            .collect();
        v.sort();
        v
    };

    let direct = ids("g.E().id()", &mut g);

    // The same ids the graph reports, and the same ones the stream produces.
    assert_eq!(direct, ids("g.E().barrier().id()", &mut g));
    assert_eq!(
        direct,
        vec![
            r#"Str("e2")"#.to_string(),
            r#"Str("e7")"#.to_string(),
            r#"Str("pay-1")"#.to_string(),
        ]
    );

    // And they round-trip: each id finds its edge again.
    for id in ["pay-1", "e7", "e2"] {
        assert_eq!(
            ids(&format!("g.E('{id}').id()"), &mut g),
            vec![format!(r#"Str("{id}")"#)],
            "`{id}` did not resolve back to its edge"
        );
    }

    // `elementMap` reads the same accessor, nested for the endpoints.
    assert_eq!(
        ids("g.E().elementMap()", &mut g),
        ids("g.E().barrier().elementMap()", &mut g)
    );
}

/// Reading the FRONTIER agrees with streaming it.
///
/// `column_paths` had no arm for an empty tail, so a traversal with no terminal
/// — `g.V().hasLabel('V').out('R')` — built a `Trav` per element to hand the
/// elements back: 5.2ms for 150k where reading them off the frontier is 0.5ms.
/// `dedup()` had an arm only when followed by `count()`, so `dedup()` alone and
/// `dedup().limit(5)` streamed too.
///
/// Compared UNSORTED, unlike most of this file: `limit`/`skip`/`tail` decide
/// WHICH elements come back, so frontier order is part of the answer here rather
/// than incidental to it.
#[test]
fn reading_the_frontier_agrees_with_streaming_it() {
    let mut lines: Vec<String> = Vec::new();

    for i in 0..60 {
        let l = if i % 4 == 0 {
            r#"["P","W"]"#
        } else {
            r#"["P"]"#
        };

        lines.push(format!(
            r#"{{"type":"node","id":"n{i}","labels":{l},"properties":{{"k":{}}}}}"#,
            i % 7
        ));
    }

    // Fan-in, so the frontier carries DUPLICATES and `dedup` has work to do.
    for i in 0..59 {
        lines.push(format!(
            r#"{{"type":"edge","id":"e{i}","from":"n{i}","to":"n{}","labels":["R"],"properties":{{"w":{}}}}}"#,
            (i * 5 + 1) % 20,
            i % 3
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

    for tail in [
        "",
        ".limit(5)",
        ".skip(5)",
        ".range(2, 7)",
        ".tail(5)",
        ".dedup()",
        ".dedup().limit(5)",
        ".dedup().count()",
        ".limit(10).dedup()",
        ".dedup().tail(3)",
        ".skip(2).limit(3)",
        ".range(1, 9).dedup().count()",
        ".limit(5).values('k')",
        ".dedup().values('k').sum()",
        ".tail(4).values('k').fold()",
        ".limit(5).id()",
        ".dedup().label()",
        ".limit(3).fold()",
        ".groupCount()",
        ".dedup().groupCount()",
        ".limit(5).groupCount()",
        // Degenerate bounds.
        ".limit(0)",
        ".skip(1000)",
        ".range(5, 5)",
        ".range(1000, 1001)",
        ".tail(1000)",
        ".limit(1000)",
    ] {
        for base in [
            "g.V().hasLabel('P').out('R')",
            // An EDGE frontier takes the same arms.
            "g.V().hasLabel('P').outE('R')",
            "g.V().hasLabel('P')",
            // An `E()` SOURCE reaches them through `lowered_ids`, whose allowlist
            // decides what may follow an edge frontier — every arm added after
            // that list was written was unreachable from here.
            "g.E()",
            "g.E().hasLabel('R')",
        ] {
            let q = format!("{base}{tail}");
            // `barrier()` stops the prefix lowering, so the same question runs
            // down the stream instead.
            let slow = if q.starts_with("g.E()") {
                q.replacen("g.E()", "g.E().barrier()", 1)
            } else {
                q.replacen("g.V()", "g.V().barrier()", 1)
            };

            assert_ne!(slow, q, "the oracle did not rewrite `{q}`");

            assert_eq!(
                rows(&q, &mut g),
                rows(&slow, &mut g),
                "`{q}` read off the frontier disagrees with the stream"
            );
        }
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

/// `order(desc)` sorts descending, and an argument `order` cannot honor is an
/// error rather than a shrug.
///
/// TinkerPop's `order()` takes a `Scope`, not an `Order` — the direction belongs
/// in the modulator, `order().by(desc)`. But `order(desc)` gets written, the
/// lexer already turned `desc` into an `Arg::Order`, and the step arm then
/// DROPPED every argument: the traversal sorted ASCENDING and reported nothing.
/// Accepting the direction is a superset; dropping it was a wrong answer.
///
/// The TS engine threw `Expected Scope.local or Scope.global` here, so the two
/// engines disagreed about the same traversal. Both now sort descending.
#[test]
fn an_order_direction_argument_is_honored() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"m":-1}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"m":4}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{"m":9}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    let asc = "[Num(-1.0), Num(4.0), Num(9.0)]";
    let desc = "[Num(9.0), Num(4.0), Num(-1.0)]";

    for (src, want) in [
        ("g.V().values('m').order()", asc),
        ("g.V().values('m').order(asc)", asc),
        ("g.V().values('m').order(desc)", desc),
        // The canonical spelling, which always worked — asserted alongside so
        // the two cannot drift apart.
        ("g.V().values('m').order().by(desc)", desc),
        // A scope argument still means a scope, not a direction.
        ("g.V().values('m').order(Scope.local)", asc),
    ] {
        assert_eq!(run(src, &mut g), want, "`{src}`");
    }

    // `local` sorts WITHIN the traverser's value, so it needs a list to be
    // observable at all — and it still takes a direction.
    assert_eq!(
        run("g.V().values('m').fold().order(local)", &mut g),
        "[List([Num(-1.0), Num(4.0), Num(9.0)])]"
    );

    // Anything `order` cannot act on is rejected. Ignoring an argument is how
    // the direction went missing in the first place.
    assert!(
        super::parse::parse("g.V().values('m').order('bogus')").is_err(),
        "an argument order cannot honor must not be silently dropped"
    );
}

/// An edge-source traversal answers what the same walk answers, step for step.
///
/// `g.E()` lowers by DESUGARING to `g.V().outE()` — every edge appears exactly
/// once as an out-edge of its source — which is what lets an edge source reach
/// the planner at all. `g.E().hasLabel('R').inV().hasLabel('W').count()` went
/// from 23.2ms to 0.11ms on 50k vertices / 150k edges, because the declined form
/// made one traverser per edge and filtered afterwards.
///
/// A rewrite that changes which end a walk starts from is the classic way to
/// drop or duplicate rows, so every shape here is checked against the SAME
/// traversal behind a `barrier()`, which defeats the pattern compiler.
#[test]
fn an_edge_source_traversal_matches_the_streamed_walk() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V","W"],"properties":{"n":1}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":2}}"#,
            r#"{"type":"node","id":"c","labels":["V","W"],"properties":{"n":3}}"#,
            r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{"w":1}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"b","to":"c","properties":{"w":2}}"#,
            // A SELF LOOP. `g.E()` yields it once and `outE()` yields it once —
            // the equivalence turns on that, and Gremlin counts a loop TWICE for
            // an UNDIRECTED hop, so the two rules must not be confused.
            r#"{"type":"edge","id":"e2","labels":["R"],"from":"c","to":"c","properties":{"w":3}}"#,
            // A second type, and a MULTI-LABEL edge, so `hasLabel` has to do real work.
            r#"{"type":"edge","id":"e3","labels":["S"],"from":"a","to":"c","properties":{"w":4}}"#,
            r#"{"type":"edge","id":"e4","labels":["R","S"],"from":"b","to":"a","properties":{"w":5}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };
    // As a MULTISET. `g.E()` enumerates in edge-id order and `g.V().outE()` walks
    // each vertex's adjacency, so the two visit the same edges in different
    // orders — and an unordered result does not promise one (the engines' rule,
    // like SQL without ORDER BY). What has to match is which rows come back and
    // how many, which is exactly what sorting the rendered values compares.
    let bag = |src: &str, g: &mut crate::graph::Graph| {
        let mut out: Vec<String> = super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
            .iter()
            .map(|v| format!("{v:?}"))
            .collect();

        out.sort();
        out
    };

    for tail in [
        // The shape the lowering exists for.
        "hasLabel('R').inV().hasLabel('W').count()",
        "hasLabel('R').inV().count()",
        "hasLabel('R').inV().values('n')",
        "hasLabel('R').inV().dedup().count()",
        // An edge FILTER before the landing.
        "hasLabel('R').has('w', 1).inV().count()",
        "has('w', 3).inV().values('n')",
        // Multi-label edges must be found by either of their labels.
        "hasLabel('S').inV().count()",
        // A landing that is NOT the far end — declines, and must still answer.
        "hasLabel('R').outV().count()",
        "hasLabel('R').bothV().count()",
        // Tags across the landing.
        "hasLabel('R').as('x').inV().select('x').count()",
        // Continuing to hop after the landing.
        "hasLabel('R').inV().out('R').count()",
        // Staying ON the edge — deliberately NOT lowered, the edge column is
        // better — but it still has to agree.
        "hasLabel('R').count()",
        "hasLabel('R').values('w')",
        "has('w', 1).count()",
    ] {
        // Against the SAME traversal with the compiler defeated: same query, so
        // the rows must match exactly, order included.
        assert_eq!(
            run(&format!("g.E().{tail}"), &mut g),
            run(&format!("g.E().barrier().{tail}"), &mut g),
            "`g.E().{tail}` disagrees with the same walk behind a barrier"
        );

        // And against the spelling it desugars TO — as a multiset, that being a
        // different traversal with its own enumeration order.
        assert_eq!(
            bag(&format!("g.E().{tail}"), &mut g),
            bag(&format!("g.V().outE().{tail}"), &mut g),
            "`g.E().{tail}` disagrees with `g.V().outE().{tail}`"
        );
    }

    // The counts themselves, so the comparisons above cannot all be wrong
    // together. Four R edges (e0, e1, e2 the loop, e4 multi-labelled); their far
    // ends are b, c, c, a — three of which are W (c, c, a).
    assert_eq!(run("g.E().hasLabel('R').count()", &mut g), "[Num(4.0)]");
    assert_eq!(
        run("g.E().hasLabel('R').inV().hasLabel('W').count()", &mut g),
        "[Num(3.0)]"
    );
    // The self loop is ONE edge, so one row — not two.
    assert_eq!(run("g.E().has('w', 3).inV().count()", &mut g), "[Num(1.0)]");
}

/// The `g.E()` plan is taken exactly where the row ORDER cannot be observed.
///
/// The walk comparisons next door check that the answers agree; this checks that
/// the gate is where it should be. Without it, `g.E().inV().hasLabel('PERSON')`
/// returned the right three vertices in adjacency order where the streamed
/// traversal — and the TS engine — return them in edge-id order. The gremlin
/// differential fuzzer caught it, generating an `E()` source one time in five.
#[test]
fn an_edge_source_plan_is_gated_on_the_order_being_unobservable() {
    let compiled = |src: &str| {
        let t = super::parse::parse(src).unwrap_or_else(|e| panic!("`{src}` parses: {e}"));
        super::pattern::compile(&t.steps).map(|c| {
            let rest_is_safe = super::pattern::order_insensitive(&t.steps[c.consumed..]);

            (c.reorders, rest_is_safe)
        })
    };

    // Reordering, and the tail cannot tell — the shape the lowering exists for.
    assert_eq!(
        compiled("g.E().hasLabel('R').inV().hasLabel('W').count()"),
        Some((true, true))
    );
    assert_eq!(
        compiled("g.E().hasLabel('R').inV().hasLabel('W').dedup().count()"),
        Some((true, true))
    );

    // Reordering, and the tail CAN tell. Compiled, but the gate refuses it.
    assert_eq!(
        compiled("g.E().hasLabel('R').inV().hasLabel('W').values('n')"),
        Some((true, false))
    );
    assert_eq!(
        compiled("g.E().hasLabel('R').inV().hasLabel('W').fold()"),
        Some((true, false))
    );
    assert_eq!(
        compiled("g.E().hasLabel('R').inV().hasLabel('W').limit(2).count()"),
        Some((true, false))
    );

    // A `V()` source never reorders, so its tail is unrestricted — this is what
    // keeps the gate from costing the shapes that were already lowered.
    assert_eq!(
        compiled("g.V().out('R').hasLabel('W').values('n')"),
        Some((false, false))
    );
    assert_eq!(
        compiled("g.V().out('R').hasLabel('W').count()"),
        Some((false, true))
    );

    // Staying on the edge is declined outright: the edge column answers it
    // better, and the pattern branch runs first.
    assert_eq!(compiled("g.E().has('w', 1).count()"), None);
}

/// `order()` places NULLS FIRST and never faults on them.
///
/// TinkerPop splits Comparability from Orderability, and only the first rejects
/// a null. `GremlinValueComparator.ORDERABILITY` is explicit:
///
/// ```text
///   // nulls first
///   if (f == null || s == null)
///       return f == s ? 0 : f == null ? -1 : 1;
/// ```
///
/// This engine has the same split — `gcmp` for predicates, `gcmp_total` for
/// `order()`, where `gval_type_rank` puts `Null` at 0 — but the TS engine used
/// its PREDICATE comparator for sorting too, and that one has no null arm, so it
/// threw "cannot order null with null" where this sorted. Any traversal that
/// orders a stream with a missing property hit it; the gremlin differential
/// fuzzer found it through `path().id().order()`.
///
/// Nulls FIRST, not last. Nulls-last is the ISO contract GQL follows, and
/// ordering is a per-language contract — the two deliberately disagree.
#[test]
fn ordering_places_nulls_first_without_faulting() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":3}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":1}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // `path().id()` is null for every row (a path is not an element), unioned
    // with real numbers so the PLACEMENT is observable and not just the absence
    // of a fault.
    let mixed = "g.V().union(__.values('n'), __.path().id())";

    assert_eq!(
        format!(
            "{:?}",
            super::parse::parse(&format!("{mixed}.order()"))
                .unwrap()
                .run(&mut g)
        ),
        "[Null, Null, Num(1.0), Num(3.0)]"
    );

    // Descending reverses the comparator, so the nulls go to the end — they are
    // the bottom of one total order, not pinned to an end like GQL's nulls.
    assert_eq!(
        format!(
            "{:?}",
            super::parse::parse(&format!("{mixed}.order().by(desc)"))
                .unwrap()
                .run(&mut g)
        ),
        "[Num(3.0), Num(1.0), Null, Null]"
    );

    // And it is not a fault — `run` would swallow one, so ask `try_run`.
    let t = super::parse::parse(&format!("{mixed}.order()")).unwrap();

    assert!(
        crate::gremlin::try_run(&mut g, &t).is_ok(),
        "ordering a null must not fault"
    );
}

/// A `where()` / `not()` the adjacency can answer filters the FRONTIER, and
/// answers what running the body answers.
///
/// The body used to have to be one bare hop. A hop plus `hasLabel` on where it
/// landed — the shape people write — fell to the stream, and so did the whole
/// traversal around it: `g.V().hasLabel('V').where(__.out('R').hasLabel('W'))
/// .count()` cost 9.9ms over 50k vertices against 0.33ms for the GQL spelling.
/// Now the body reads the adjacency and the result stays a column, at 1.2ms.
///
/// `barrier()` defeats the column path, so each pair below is the same question
/// asked twice, once lowered and once streamed.
#[test]
fn a_semi_join_over_the_frontier_matches_the_streamed_body() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":1}}"#,
            // Multi-label, so `hasLabel` has to look past the first.
            r#"{"type":"node","id":"b","labels":["X","W"],"properties":{"n":2}}"#,
            r#"{"type":"node","id":"c","labels":["V","W"],"properties":{"n":3}}"#,
            // No out-edges at all — the `not()` side.
            r#"{"type":"node","id":"d","labels":["V"],"properties":{"n":4}}"#,
            r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"d","properties":{}}"#,
            r#"{"type":"edge","id":"e2","labels":["S"],"from":"c","to":"b","properties":{}}"#,
            // A SELF LOOP: `c` reaches a W (itself) without leaving.
            r#"{"type":"edge","id":"e3","labels":["R"],"from":"c","to":"c","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    for body in [
        // The shape the lowering exists for.
        "__.out('R').hasLabel('W')",
        "__.in('R').hasLabel('W')",
        "__.both('R').hasLabel('W')",
        // A bare hop still works — that arm predates this.
        "__.out('R')",
        // A property EQUALITY on the landed vertex, the sibling of the label
        // test. Streamed it cost 3.71ms over 20k vertices against 0.646ms here.
        "__.out('R').has('n', 2)",
        "__.in('R').has('n', 2)",
        "__.both('R').has('n', 2)",
        // A value nothing carries, and a key nothing carries.
        "__.out('R').has('n', 999)",
        "__.out('R').has('nope', 1)",
        // A RANGE is `P`'s business, not the adjacency's — it must decline.
        "__.out('R').has('n', gt(1))",
        // On an EDGE hop `has` reads the EDGE's property, not a vertex's.
        "__.outE('R').has('w', 1)",
        "__.out('S')",
        // An UNKNOWN label matches no vertex. It must not read as "no label
        // test": an unknown name resolves to an empty id list, and empty is also
        // what "no test at all" looks like.
        "__.out('R').hasLabel('NOPE')",
        // Mixed known and unknown — the known one still selects.
        "__.out('R').hasLabel('W', 'NOPE')",
        // An unknown EDGE type reaches nothing.
        "__.out('NOPE').hasLabel('W')",
        // An EDGE hop: `hasLabel` here tests the EDGE's label, not a vertex's, so
        // the vertex-label shortcut must not claim it.
        "__.outE('R').hasLabel('R')",
        "__.outE('R').hasLabel('S')",
        // Bodies the adjacency cannot answer, which must still work.
        "__.out('R').has('n', 2)",
        "__.out('R').out('R')",
        "__.out('R').hasLabel('W').hasLabel('X')",
    ] {
        for shape in [
            format!("g.V().where({body}).count()"),
            format!("g.V().not({body}).count()"),
            // The identities, not just the counts — the filter decides WHICH
            // rows survive, and a count can agree by accident.
            format!("g.V().where({body}).values('n')"),
            format!("g.V().not({body}).values('n')"),
            // Filtered frontier, then more column work.
            format!("g.V().hasLabel('V').where({body}).dedup().count()"),
        ] {
            let streamed = shape.replacen("g.V()", "g.V().barrier()", 1);

            assert_eq!(
                run(&shape, &mut g),
                run(&streamed, &mut g),
                "`{shape}` disagrees with the streamed body"
            );
        }
    }

    // The answers themselves, so the pairs above cannot agree by both being
    // wrong. `a` reaches b (W); `c` reaches itself (W) through the loop.
    assert_eq!(
        run("g.V().where(__.out('R').hasLabel('W')).values('n')", &mut g),
        "[Num(1.0), Num(3.0)]"
    );
    // An unknown label selects nobody — the bug this test exists for.
    assert_eq!(
        run("g.V().where(__.out('R').hasLabel('NOPE')).count()", &mut g),
        "[Num(0.0)]"
    );
    assert_eq!(
        run("g.V().not(__.out('R').hasLabel('NOPE')).count()", &mut g),
        "[Num(4.0)]"
    );
}

/// `groupCount().by(k)` tallies a property off the frontier and agrees with the
/// streamed fold — including the KEY ORDER, which is first-seen and observable.
///
/// Only `groupCount()` on the element itself had a column arm, so keying on a
/// property built a traverser per element to run the modulator over:
/// `g.V().out('R').groupCount().by('n')` cost 14.3ms over 150k rows against
/// 1.26ms for the GQL spelling. Now 4.6ms.
#[test]
fn a_keyed_group_count_matches_the_streamed_fold() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":1,"s":"x"}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":2,"s":"y"}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{"n":1,"s":"x"}}"#,
            // No `n` and no `s` — a NULL key, which is a group like any other.
            r#"{"type":"node","id":"d","labels":["V"],"properties":{}}"#,
            r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{"w":5}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"c","properties":{"w":5}}"#,
            r#"{"type":"edge","id":"e2","labels":["R"],"from":"b","to":"d","properties":{"w":7}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    for shape in [
        "g.V().groupCount().by('n')",
        "g.V().groupCount().by('s')",
        // A key NO element carries — every group is null.
        "g.V().groupCount().by('missing')",
        "g.V().out('R').groupCount().by('n')",
        "g.V().hasLabel('V').out('R').groupCount().by('n')",
        // An EDGE frontier keys off edge properties.
        "g.V().outE('R').groupCount().by('w')",
        // The identity form, which had the only arm before — asserted alongside
        // so the two cannot drift.
        "g.V().groupCount()",
        // A `by()` carrying an ORDER sorts rather than tallies, so it must NOT
        // take the arm — and must still answer.
        "g.V().groupCount().by('n').order(local)",
    ] {
        let streamed = shape.replacen("g.V()", "g.V().barrier()", 1);

        assert_eq!(
            run(shape, &mut g),
            run(&streamed, &mut g),
            "`{shape}` disagrees with the streamed fold"
        );
    }

    // The tally itself, key order included: first-seen, so 1 before 2 before the
    // null that `d` contributes.
    assert_eq!(
        run("g.V().groupCount().by('n')", &mut g),
        "[Map(MapVal { keys: [Num(1.0), Num(2.0), Null], vals: [Num(2.0), Num(1.0), Num(1.0)] })]"
    );
}

/// Every temporal constructor this dialect spells parses through the ONE shared
/// dispatch, and an unknown one still names itself.
///
/// `parse_temporal_literal` used to repeat `Temporal::parse`'s six arms — six
/// chances for the dialects to disagree about which constructors exist or how
/// each parses, in a codebase where a temporal has to decode identically
/// everywhere. Only the SPELLING differs now: Gremlin writes `time(…)` where
/// the shared tag is `localtime`.
#[test]
fn every_temporal_constructor_parses_through_the_shared_dispatch() {
    let mut g =
        crate::ndjson::decode(r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":1}}"#)
            .expect("fixture decodes");

    // `inject` carries the literal straight through, so what comes back IS what
    // the constructor parsed.
    for (src, want) in [
        (
            "date('2020-01-02')",
            "[Temporal(Date(Date { days: 18263 }))]",
        ),
        (
            "time('01:02:03')",
            "[Temporal(Time(Time { secs: 3723, nanos: 0 }))]",
        ),
        (
            "duration('P1D')",
            "[Temporal(Duration(Duration { months: 0, days: 1, secs: 0, nanos: 0 }))]",
        ),
    ] {
        assert_eq!(
            format!(
                "{:?}",
                super::parse::parse(&format!("g.inject({src})"))
                    .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                    .run(&mut g)
            ),
            want,
            "`{src}`"
        );
    }

    // The remaining three round-trip through their own formatter, which is the
    // check that matters for them: the value survives the shared parse.
    for src in [
        "datetime('2020-01-02T03:04:05')",
        "zoned_time('01:02:03+01:00')",
        "zoned_datetime('2020-01-02T03:04:05+01:00')",
    ] {
        let out = super::parse::parse(&format!("g.inject({src})"))
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(&mut g);

        assert!(
            matches!(out.as_slice(), [GVal::Temporal(_)]),
            "`{src}` gave {out:?}"
        );
    }

    // A malformed value reports the PARSER's message, not a constructor error.
    let bad = super::parse::parse("g.inject(date('not-a-date'))");

    assert!(bad.is_err(), "a malformed date must not parse");

    // An unknown constructor still names itself rather than reporting a parse
    // failure for a kind that does not exist.
    let unknown = super::parse::parse("g.inject(fortnight('P14D'))");

    assert!(unknown.is_err(), "an unknown constructor must not parse");
}

/// `min`/`max` answer the same whatever the SCOPE, and a NaN never wins a `min`.
///
/// There were three copies of the fold-to-an-extreme loop — Gremlin's global,
/// Gremlin's `local`, and GQL's — and the third had drifted. `local` compared
/// with `cmp_or_fault(..) == Some(want)` and nothing else, so a NaN answered
/// `None`, never matched `want`, and whichever NaN arrived first held the
/// extreme forever:
///
/// ```text
///   math('sqrt _').min()                 2.0
///   math('sqrt _').fold().min(local)     NaN
/// ```
///
/// One question, two scopes, two answers. The fold is shared now
/// (`crate::value::fold_extreme`) and only the comparator is per-language, so
/// the scopes cannot disagree again.
#[test]
fn an_extreme_is_the_same_in_every_scope() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"m":-1}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"m":4}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{"m":9}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    // `sqrt(-1)` is a NaN; the other two are 2 and 3.
    let nums = "g.V().values('m').math('sqrt _')";

    // NaN is the GREATEST value, so `min` never picks it and `max` always does —
    // in BOTH scopes.
    assert_eq!(run(&format!("{nums}.min()"), &mut g), "[Num(2.0)]");
    assert_eq!(
        run(&format!("{nums}.fold().min(local)"), &mut g),
        "[Num(2.0)]"
    );
    assert_eq!(run(&format!("{nums}.max()"), &mut g), "[Num(NaN)]");
    assert_eq!(
        run(&format!("{nums}.fold().max(local)"), &mut g),
        "[Num(NaN)]"
    );

    // Without a NaN the two scopes already agreed; asserted so a future change
    // cannot fix one and break the other.
    let plain = "g.V().values('m')";

    assert_eq!(run(&format!("{plain}.min()"), &mut g), "[Num(-1.0)]");
    assert_eq!(
        run(&format!("{plain}.fold().min(local)"), &mut g),
        "[Num(-1.0)]"
    );
    assert_eq!(run(&format!("{plain}.max()"), &mut g), "[Num(9.0)]");
    assert_eq!(
        run(&format!("{plain}.fold().max(local)"), &mut g),
        "[Num(9.0)]"
    );

    // An empty fold is null, not an error, in both scopes.
    assert_eq!(run("g.V().values('nope').min()", &mut g), "[Null]");
    assert_eq!(
        run("g.V().values('nope').fold().min(local)", &mut g),
        "[Null]"
    );
}

/// A chain-shaped `match(…)` answers what the backtracking solver answers.
///
/// `match` is Gremlin's only join and it had no planner: it picks a runnable
/// pattern and runs its sub-traversal per binding, so an INDEXED anchor is
/// scanned rather than sought. On 20k vertices,
/// `match(as('a').has('k', <indexed>), as('a').out('R').as('b')).count()` cost
/// 3.755ms against 0.000ms for the GQL spelling of the same question — 9842x.
///
/// The patterns people write are a chain, and a chain is a linear traversal:
/// `match(as('a').out('R').as('b'), as('b').has('n', 1))` IS
/// `as('a').out('R').as('b').has('n', 1)`. Rewriting it that way hands the whole
/// thing to the shared planner. 0.002ms after, and 12.7ms -> 0.281ms unanchored.
///
/// `identity()` ahead of the match defeats the rewrite (it fires only directly
/// off `V()`), so each pair below is the same question asked both ways.
#[test]
fn a_chain_shaped_match_agrees_with_the_solver() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"name":"marko","age":29}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"name":"vadas","age":27}}"#,
            r#"{"type":"node","id":"c","labels":["S"],"properties":{"name":"lop","age":29}}"#,
            r#"{"type":"node","id":"d","labels":["P"],"properties":{"name":"josh","age":32}}"#,
            r#"{"type":"edge","id":"e0","labels":["CREATED"],"from":"a","to":"c","properties":{}}"#,
            r#"{"type":"edge","id":"e1","labels":["CREATED"],"from":"d","to":"c","properties":{}}"#,
            r#"{"type":"edge","id":"e2","labels":["KNOWS"],"from":"a","to":"b","properties":{}}"#,
            r#"{"type":"edge","id":"e3","labels":["KNOWS"],"from":"a","to":"d","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // As a MULTISET: the rewrite enumerates in planner order and the solver in
    // backtracking order, and `match` promises neither (the step tests sort too).
    let bag = |src: &str, g: &mut crate::graph::Graph| {
        let mut out: Vec<String> = super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
            .iter()
            .map(|v| format!("{v:?}"))
            .collect();

        out.sort();
        out
    };

    for tail in [
        // Chains the rewrite takes.
        "match(__.as('a').out('KNOWS').as('b')).count()",
        "match(__.as('a').out('KNOWS').as('b')).select('a','b').by('name')",
        "match(__.as('a').out('CREATED').as('b'), __.as('b').has('name','lop')).select('a').by('name')",
        "match(__.as('a').has('name','marko'), __.as('a').out('KNOWS').as('b')).select('b').by('name')",
        // A filter BEFORE the hop, and one after — both attach at their tag.
        "match(__.as('a').has('age',29), __.as('a').out('CREATED').as('b'), __.as('b').has('age',29)).select('a','b').by('name')",
        // The binding MAP itself, which is the step's own value.
        "match(__.as('a').out('KNOWS').as('b'))",
        // Longer chain.
        "match(__.as('a').out('KNOWS').as('b'), __.as('b').out('CREATED').as('c')).select('a','c').by('name')",
        // Shapes the rewrite DECLINES and the solver keeps — they must still answer.
        // A branch: two patterns leaving `a`.
        "match(__.as('a').out('KNOWS').as('b'), __.as('a').out('CREATED').as('c')).select('b','c').by('name')",
        // A negated pattern.
        "match(__.as('a').out('KNOWS').as('b'), __.not(__.as('b').has('name','vadas'))).select('b').by('name')",
        // An inner the pattern compiler will not take.
        "match(__.as('a').out('KNOWS').count().as('b')).select('b')",
    ] {
        let rewritten = format!("g.V().{tail}");
        // `identity()` is a step, so the `[V, Match, …]` shape no longer holds
        // and the solver runs.
        let solved = format!("g.V().identity().{tail}");

        assert_eq!(
            bag(&rewritten, &mut g),
            bag(&solved, &mut g),
            "`{tail}` disagrees with the solver"
        );
    }

    // The answers themselves, so the pairs cannot agree by both being wrong.
    // marko KNOWS vadas and josh.
    assert_eq!(
        bag(
            "g.V().match(__.as('a').out('KNOWS').as('b')).select('b').by('name')",
            &mut g
        ),
        vec!["Str(\"josh\")", "Str(\"vadas\")"]
    );
    // Both creators of lop.
    assert_eq!(
        bag(
            "g.V().match(__.as('a').out('CREATED').as('b'), __.as('b').has('name','lop')).select('a').by('name')",
            &mut g
        ),
        vec!["Str(\"josh\")", "Str(\"marko\")"]
    );
}

/// A keyed `dedup`/`order` and a `select` over tags answer what the stream
/// answers.
///
/// All three read something the planner ALREADY produced and were re-deriving
/// it per row. Over 20k vertices: `dedup().by('n')` 2.22ms -> 0.52ms (it built a
/// traverser and a `DedupKey` per element), `order().by('n').limit(5)` 2.58ms ->
/// 1.06ms, and `as('a')…as('b').select('a','b').count()` 23.4ms -> 0.40ms — that
/// last one looked each label up by string scan per row and allocated a fresh
/// `Arc<str>` per label per row, to key a map a `count()` never reads.
///
/// `barrier()` defeats the column path, so each pair is the same question twice.
#[test]
fn keyed_terminals_and_select_match_the_stream() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":2,"s":"x"}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"n":1,"s":"y"}}"#,
            r#"{"type":"node","id":"c","labels":["P"],"properties":{"n":2,"s":"x"}}"#,
            // No `n` — a NULL key, which sorts and dedups like any other value.
            r#"{"type":"node","id":"d","labels":["P"],"properties":{"s":"z"}}"#,
            r#"{"type":"edge","id":"e0","labels":["R"],"from":"a","to":"b","properties":{}}"#,
            r#"{"type":"edge","id":"e1","labels":["R"],"from":"a","to":"c","properties":{}}"#,
            r#"{"type":"edge","id":"e2","labels":["R"],"from":"b","to":"d","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    for tail in [
        // dedup keyed on a property: FIRST-SEEN survives, which is observable.
        "dedup().by('n').values('s')",
        "dedup().by('n').count()",
        "dedup().by('s').values('s')",
        // A key some element lacks.
        "dedup().by('nope').count()",
        // order keyed on a property, both directions and both spellings of them.
        "order().by('n').values('s')",
        "order().by('n', desc).values('s')",
        "order().by('s').values('s')",
        // Ties keep frontier order — visible through the limit.
        "order().by('n').limit(2).values('s')",
        "order().by('n', desc).limit(2).values('s')",
        // A MIXED-TYPE key faults in the stream, so the column path must decline
        // rather than order it.
        "order().by('mix').count()",
        // select over tags the prefix bound.
        "as('a').out('R').as('b').select('a','b').count()",
        "as('a').out('R').as('b').select('b').values('s')",
        "as('a').out('R').as('b').select('a','b')",
        // A label the prefix never bound drops every row.
        "as('a').out('R').as('b').select('a','zz').count()",
        // `Pop.all` yields a list per label, which is not a zip.
        "as('a').out('R').as('b').select(Pop.all, 'a').count()",
    ] {
        assert_eq!(
            run(&format!("g.V().{tail}"), &mut g),
            run(&format!("g.V().barrier().{tail}"), &mut g),
            "`{tail}` disagrees with the stream"
        );
    }

    // The answers themselves, so the pairs cannot agree by both being wrong.
    // n = 2, 1, 2, null → first-seen keeps a (2), b (1), d (null).
    assert_eq!(
        run("g.V().dedup().by('n').values('s')", &mut g),
        "[Str(\"x\"), Str(\"y\"), Str(\"z\")]"
    );
    // Ascending by n with the null: nulls rank first in Gremlin's order.
    assert_eq!(
        run("g.V().order().by('n').values('s')", &mut g),
        "[Str(\"z\"), Str(\"y\"), Str(\"x\"), Str(\"x\")]"
    );
    assert_eq!(
        run(
            "g.V().as('a').out('R').as('b').select('a','b').count()",
            &mut g
        ),
        "[Num(3.0)]"
    );
}

/// A MULTI-HOP `where`/`not` body walked backwards agrees with running it.
///
/// One hop probes an adjacency; several had no adjacency to probe, so the body
/// ran per traverser and built a traverser per intermediate —
/// `g.V().where(__.out('R').out('R')).count()` cost 5.1ms over 20k vertices.
/// Backwards it is one level per hop and no tree: 0.881ms, and 0.169ms once a
/// property narrows the far end.
///
/// Forward-per-row is bounded by finding ONE walk, but that bound is the whole
/// `degree^hops` tree exactly when no walk exists — which is the rows a `where`
/// discards and a `not` keeps. So the two must agree precisely there, which is
/// what the isolated and dead-ended vertices in this fixture are for.
#[test]
fn a_multi_hop_semi_join_agrees_with_running_the_body() {
    let mut g = crate::ndjson::decode(
        &[
            // A 3-chain: a -> b -> c -> d, so `a` has 3 hops, `b` has 2, `c` 1.
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"n":1}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"n":2}}"#,
            r#"{"type":"node","id":"c","labels":["P","W"],"properties":{"n":3}}"#,
            r#"{"type":"node","id":"d","labels":["P"],"properties":{"n":7}}"#,
            // Isolated: no walk of any length.
            r#"{"type":"node","id":"e","labels":["P"],"properties":{"n":7}}"#,
            // A SELF LOOP, which has a walk of every length.
            r#"{"type":"node","id":"f","labels":["P"],"properties":{"n":1}}"#,
            r#"{"type":"edge","id":"r0","labels":["R"],"from":"a","to":"b","properties":{}}"#,
            r#"{"type":"edge","id":"r1","labels":["R"],"from":"b","to":"c","properties":{}}"#,
            r#"{"type":"edge","id":"r2","labels":["R"],"from":"c","to":"d","properties":{}}"#,
            r#"{"type":"edge","id":"r3","labels":["R"],"from":"f","to":"f","properties":{}}"#,
            // Another type, so a typed chain is not just "any edge".
            r#"{"type":"edge","id":"s0","labels":["S"],"from":"a","to":"d","properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    let run = |src: &str, g: &mut crate::graph::Graph| {
        format!(
            "{:?}",
            super::parse::parse(src)
                .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
                .run(g)
        )
    };

    for body in [
        "__.out('R').out('R')",
        "__.out('R').out('R').out('R')",
        "__.in('R').in('R')",
        "__.both('R').both('R')",
        // Mixed directions, where flipping the chain has to flip each hop.
        "__.out('R').in('R')",
        "__.in('R').out('R')",
        // A landed test on the far end only.
        "__.out('R').out('R').hasLabel('W')",
        "__.out('R').out('R').has('n', 7)",
        "__.out('R').out('R').has('n', 999)",
        // An UNKNOWN edge type matches nothing, and must not read as "any".
        "__.out('NOPE').out('R')",
        "__.out('R').out('NOPE')",
        // Mixed known and unknown on one hop is still the known one.
        "__.out('R', 'NOPE').out('R')",
        // Typed differently per hop.
        "__.out('S').out('R')",
        // The edge-spelled form of a hop.
        "__.outE('R').inV().outE('R').inV()",
        // `repeat` unrolls to the same chain.
        "__.repeat(__.out('R')).times(2)",
        // Bodies the chain cannot take, which must still answer.
        "__.out('R').has('n', gt(1)).out('R')",
        "__.out('R').count()",
    ] {
        for shape in [
            format!("g.V().where({body}).values('n')"),
            format!("g.V().not({body}).values('n')"),
            format!("g.V().where({body}).count()"),
        ] {
            let streamed = shape.replacen("g.V()", "g.V().barrier()", 1);

            assert_eq!(
                run(&shape, &mut g),
                run(&streamed, &mut g),
                "`{shape}` disagrees with running the body"
            );
        }
    }

    // The answers themselves. Two hops of R exist from `a` (a→b→c) and from `f`
    // (its own loop, twice); `b` reaches c→d; `c`, `d`, `e` do not.
    assert_eq!(
        run("g.V().where(__.out('R').out('R')).values('n')", &mut g),
        "[Num(1.0), Num(2.0), Num(1.0)]"
    );
    // An unknown type reaches nothing at all.
    assert_eq!(
        run("g.V().where(__.out('NOPE').out('R')).count()", &mut g),
        "[Num(0.0)]"
    );
    assert_eq!(
        run("g.V().not(__.out('NOPE').out('R')).count()", &mut g),
        "[Num(6.0)]"
    );
}

/// The same QUESTION in both languages, priced side by side.
///
/// Not a gate — `equivalent_traversals_cost_the_same` and
/// `equivalent_spellings_cost_the_same` are the gates, and each compares
/// spellings WITHIN one language. This compares ACROSS them, which is a different
/// question and the one that found every lowering in this area: where one engine
/// has a shortcut for something the other enumerates, the pair shows a 3-150x
/// ratio and the slower side names the missing arm.
///
/// Run it with:
///
/// ```text
/// cargo test --release -- --ignored --nocapture cross_language_cost_probe
/// ```
///
/// Read the ratios, not the times — the fixture is 20k vertices / 60k edges and
/// the absolute numbers move with the machine. And read them SKEPTICALLY: a
/// `RETURN u` that renders every element against a `g.V()` that returns handles
/// the boundary renders later reads as 168x and is not a gap at all. Ask both
/// sides for the same thing before believing the ratio.
///
/// Open gaps as of this writing:
///
/// ```text
///   MATCH (u:V) RETURN u.n              0.121   vs 0.028    4.4x   GQL
/// ```
///
/// That one is GQL paying for a frame it then transposes where the Gremlin side
/// reads one column — not a missing shortcut, and the smallest of the set in
/// absolute terms (0.1ms over 20k rows).
#[test]
#[ignore = "timing"]
fn cross_language_cost_probe() {
    let mut lines = String::new();

    for i in 0..20_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };

        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"k\":\"key{i:06}\"}}}}\n",
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

    let pairs = CROSS_LANGUAGE_PAIRS;

    println!();

    for (name, gq, gr) in pairs {
        let tg = {
            let plan = crate::gql::parse(gq).unwrap_or_else(|e| panic!("`{gq}`: {e}"));
            let mut best = f64::MAX;

            for _ in 0..5 {
                let t = std::time::Instant::now();
                let rs = plan
                    .execute(&mut g, &crate::gql::eval::Params::new())
                    .unwrap_or_else(|e| panic!("`{gq}`: {e}"));
                let secs = t.elapsed().as_secs_f64();

                std::hint::black_box(rs.rows().count());
                if secs < best {
                    best = secs;
                }
            }

            best
        };
        let tr = grem_time(&mut g, gr);
        let (slow, ratio) = if tg > tr {
            ("GQL ", tg / tr)
        } else {
            ("GREM", tr / tg)
        };

        println!(
            "PROBE {ratio:>6.1}x {slow} slower  gql {:>8.3}ms  grem {:>8.3}ms  [{name}]",
            tg * 1e3,
            tr * 1e3
        );
    }
}

// --- `out(T).count()` counts the type bucket, and only when that IS the count --

/// A fixture whose shape makes every exclusion visible: a self-loop, a vertex
/// with no edges, two edge types, and a labelled subset.
fn bucket_fixture() -> Graph {
    let mut lines = String::new();

    for (i, l) in [r#"["V","W"]"#, r#"["V"]"#, r#"["V"]"#, r#"["V"]"#]
        .iter()
        .enumerate()
    {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{i}}}}}\n"
        ));
    }
    // n0->n1, n1->n2, n2->n2 (a self-loop), n0->n2 of the OTHER type. n3 is
    // isolated.
    for (i, (from, to, t)) in [(0, 1, "R"), (1, 2, "R"), (2, 2, "R"), (0, 2, "S")]
        .iter()
        .enumerate()
    {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{from}\",\"to\":\"n{to}\",\"labels\":[\"{t}\"],\"properties\":{{}}}}\n"
        ));
    }

    crate::ndjson::decode(&lines).expect("fixture decodes")
}

fn count_of(g: &mut Graph, src: &str) -> f64 {
    one_num(
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g),
    )
}

/// `out(T).count()` is the size of the T bucket — one traverser per edge.
#[test]
fn counting_a_bare_hop_is_the_edge_bucket() {
    let mut g = bucket_fixture();

    assert_eq!(count_of(&mut g, "g.V().out('R').count()"), 3.0);
    assert_eq!(count_of(&mut g, "g.V().in('R').count()"), 3.0);
    assert_eq!(count_of(&mut g, "g.V().out('S').count()"), 1.0);
    assert_eq!(count_of(&mut g, "g.V().out('R','S').count()"), 4.0);
}

/// The self-loop is ONE out-edge of one vertex, so the bucket length is right;
/// `both()` sees it from both ends and the bucket length is not, which is why
/// that direction takes the walk.
#[test]
fn an_undirected_hop_count_is_not_the_bucket_length() {
    let mut g = bucket_fixture();

    // n0-n1, n1-n2, n2-n2 seen from each end: 2 + 2 + 2.
    assert_eq!(count_of(&mut g, "g.V().both('R').count()"), 6.0);
}

/// A filter before the hop means the traversers are not the whole universe, so
/// the shortcut must not fire. Both of these once counted the bucket.
#[test]
fn a_filtered_hop_count_is_not_the_bucket_length() {
    let mut g = bucket_fixture();

    // Only n0 is a W, and it has one R out-edge.
    assert_eq!(
        count_of(&mut g, "g.V().hasLabel('W').out('R').count()"),
        1.0
    );
    assert_eq!(count_of(&mut g, "g.V().has('n', 1).out('R').count()"), 1.0);
}

/// `dedup()` after the hop counts distinct FAR ENDS, which is a different
/// question from how many edges there are: n1 and n2 are each landed on twice
/// across `R`'s three edges — n2 by `n1->n2` and by its own self-loop.
#[test]
fn a_deduped_hop_count_is_not_the_bucket_length() {
    let mut g = bucket_fixture();

    assert_eq!(count_of(&mut g, "g.V().out('R').dedup().count()"), 2.0);
}

/// A type no edge carries is zero, not "any type".
#[test]
fn counting_a_hop_of_an_unknown_type_is_zero() {
    let mut g = bucket_fixture();

    assert_eq!(count_of(&mut g, "g.V().out('NOPE').count()"), 0.0);
    assert_eq!(count_of(&mut g, "g.V().out('NOPE','R').count()"), 3.0);
}

/// A second hop is not a bucket length — it depends on the far ends' degrees.
#[test]
fn counting_two_hops_is_not_the_bucket_length() {
    let mut g = bucket_fixture();

    // n0->n1->n2, n1->n2->n2, n2->n2->n2.
    assert_eq!(count_of(&mut g, "g.V().out('R').out('R').count()"), 3.0);
}

// --- ORDER BY + LIMIT partitions, and ties keep their input order -----------

/// A top-k must return the same rows in the same order as the full sort it
/// replaces, INCLUDING across a tie at the boundary — quickselect is unstable,
/// so a comparator that stops at the key would be free to return either of two
/// equal-keyed elements and the two engines would disagree.
#[test]
fn a_limited_order_agrees_with_the_full_sort() {
    let mut lines = String::new();

    // Every key appears four times, so every limit lands mid-tie.
    for i in 0..40usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{}}}}}\n",
            i % 10
        ));
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    let ids = |g: &mut Graph, src: &str| -> Vec<String> {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
            .iter()
            .map(|v| format!("{v:?}"))
            .collect()
    };

    for (dir, spelling) in [("", "order().by('n')"), ("desc", "order().by('n', desc)")] {
        let full = ids(&mut g, &format!("g.V().{spelling}.id()"));

        for k in [1usize, 3, 7, 40, 100] {
            let got = ids(&mut g, &format!("g.V().{spelling}.limit({k}).id()"));

            assert_eq!(
                got,
                full[..k.min(full.len())].to_vec(),
                "limit({k}) after {spelling} {dir} disagreed with the full sort"
            );
        }
        // `range` bounds it from the top the same way; `tail` does not, and takes
        // the full sort.
        assert_eq!(
            ids(&mut g, &format!("g.V().{spelling}.range(5, 9).id()")),
            full[5..9].to_vec()
        );
        assert_eq!(
            ids(&mut g, &format!("g.V().{spelling}.tail(3).id()")),
            full[full.len() - 3..].to_vec()
        );
    }
}

// --- the map steps off a frontier agree with the map steps over a stream ------

/// `fold().unfold()` means the same thing and takes the STREAM: the fold makes
/// the frontier a single list value, so the column route ends there and the steps
/// after it run per traverser. Pairing the two spellings is how these tests stay
/// non-vacuous — checked by breaking each column arm and watching them fail.
fn same_via_stream(g: &mut Graph, src: &str) {
    let column = super::parse::parse(src)
        .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
        .run(g);
    let (head, tail) = src.split_once('.').expect("a traversal has a step");
    let streamed_src = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));
    let streamed = super::parse::parse(&streamed_src)
        .unwrap_or_else(|e| panic!("`{streamed_src}` parses: {e}"))
        .run(g);

    assert_eq!(
        format!("{column:?}"),
        format!("{streamed:?}"),
        "`{src}` disagreed with its streamed spelling `{streamed_src}`"
    );
    assert!(!column.is_empty(), "`{src}` returned nothing to compare");
}

#[test]
fn element_map_off_a_frontier_matches_the_stream() {
    let mut g = modern();

    same_via_stream(&mut g, "g.V().elementMap()");
    same_via_stream(&mut g, "g.V().elementMap('name')");
    // An edge map carries the IN/OUT endpoint stubs as well.
    same_via_stream(&mut g, "g.E().elementMap()");
    same_via_stream(&mut g, "g.V().hasLabel('PERSON').elementMap('name','age')");
    // A key nothing carries is absent, not null.
    same_via_stream(&mut g, "g.V().elementMap('nope')");
}

#[test]
fn project_off_a_frontier_matches_the_stream() {
    let mut g = modern();

    same_via_stream(&mut g, "g.V().project('name').by('name')");
    same_via_stream(&mut g, "g.V().project('name','age').by('name').by('age')");
    // Fewer `by()`s than keys: the rest project the element itself.
    same_via_stream(&mut g, "g.V().project('self','name').by().by('name')");
    same_via_stream(&mut g, "g.V().project('self')");
    // A key nothing carries.
    same_via_stream(&mut g, "g.V().project('nope').by('nope')");
    // A sub-traversal `by()` is not a column and stays on the stream — the two
    // spellings must still agree, which is what says the guard declines rather
    // than mis-reads it.
    same_via_stream(&mut g, "g.V().project('out').by(__.out().count())");
    same_via_stream(&mut g, "g.V().project('id').by(__.id())");
    same_via_stream(
        &mut g,
        "g.E().project('label','weight').by(__.label()).by('weight')",
    );
}

/// The maps a `project()` builds share ONE key vector, and sharing must not let
/// one row's write reach another's — `MapVal::push` is copy-on-write for exactly
/// this. (`tree()` grows a map it was handed; nothing else does.)
#[test]
fn projected_rows_share_keys_without_sharing_values() {
    let mut g = modern();
    let rows = super::parse::parse("g.V().project('name','age').by('name').by('age')")
        .expect("parses")
        .run(&mut g);

    assert_eq!(rows.len(), 6);

    for r in &rows {
        let GVal::Map(m) = r else {
            panic!("expected a map, got {r:?}");
        };

        assert_eq!(m.keys().len(), 2);
        assert_eq!(m.len(), 2);
    }
    // Distinct values behind the shared keys.
    let names: Vec<String> = rows
        .iter()
        .map(|r| match r {
            GVal::Map(m) => format!("{:?}", m.values()[0]),
            _ => unreachable!(),
        })
        .collect();
    let mut uniq = names.clone();
    uniq.sort();
    uniq.dedup();

    assert_eq!(
        uniq.len(),
        names.len(),
        "rows shared a value vector: {names:?}"
    );
}

// --- a leading LIMIT bounds the scan, and only where it may ------------------

/// A capped scan must return the same ids, in the same order, as scanning
/// everything and slicing — including when a property filter is what decides
/// which rows survive, since the cap counts SURVIVORS rather than candidates.
#[test]
fn a_capped_scan_agrees_with_scanning_and_slicing() {
    let mut lines = String::new();

    for i in 0..200usize {
        let l = if i % 5 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };

        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{}}}}}\n",
            i % 7
        ));
    }
    for i in 0..200usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{}}}}}\n",
            (i * 7 + 1) % 200,
            i % 3
        ));
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    let run = |g: &mut Graph, src: &str| -> Vec<String> {
        super::parse::parse(src)
            .unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
            .run(g)
            .iter()
            .map(|v| format!("{v:?}"))
            .collect()
    };

    for source in [
        "g.V().hasLabel('V')",
        "g.V().hasLabel('W')",
        "g.V().has('n', 3)",
        "g.E().hasLabel('R')",
        "g.E().has('w', 1)",
    ] {
        let all = run(&mut g, &format!("{source}.id()"));

        for k in [0usize, 1, 5, 40, 1000] {
            assert_eq!(
                run(&mut g, &format!("{source}.limit({k}).id()")),
                all[..k.min(all.len())].to_vec(),
                "{source}.limit({k}) disagreed with the full scan"
            );
        }
        assert_eq!(
            run(&mut g, &format!("{source}.range(3, 11).id()")),
            all[3..11.min(all.len())].to_vec()
        );
        // Neither of these bounds the scan from the top; both must still be right.
        assert_eq!(
            run(&mut g, &format!("{source}.skip(190).id()")),
            all[190.min(all.len())..].to_vec()
        );
        assert_eq!(
            run(&mut g, &format!("{source}.tail(3).id()")),
            all[all.len() - 3..].to_vec()
        );
        // The count of a capped scan is the cap, not the bucket.
        assert_eq!(
            run(&mut g, &format!("{source}.limit(1).count()")),
            vec!["Num(1.0)".to_string()]
        );
    }
}

/// A hop between the source and the LIMIT means the cap is not the scan's — the
/// frontier it bounds is the one AFTER the walk.
#[test]
fn a_limit_past_a_hop_does_not_cap_the_scan() {
    let mut lines = String::new();

    // Every edge leaves the LAST vertex. A scan capped at 3 would keep the first
    // three, which have no out-edges at all, and answer 0 where the right answer
    // is 3 — the fixture has to put the edges where a wrongly capped scan cannot
    // reach them, or capping and not capping agree by luck.
    for i in 0..4usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{}}}}\n"
        ));
    }
    for (i, (a, b)) in [(3, 0), (3, 1), (3, 2), (3, 3)].iter().enumerate() {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{a}\",\"to\":\"n{b}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n"
        ));
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    // Four edges, so four traversers land; the limit takes three of them.
    assert_eq!(count_of(&mut g, "g.V().out('R').limit(3).count()"), 3.0);
    assert_eq!(count_of(&mut g, "g.V().out('R').limit(10).count()"), 4.0);
    assert_eq!(count_of(&mut g, "g.V().out('R').range(1, 3).count()"), 2.0);
}

// --- key presence lowers into the shared seek --------------------------------

/// n0 carries `a` and `b`; n1 carries `a` only; n2 carries a STORED NULL under
/// `a`; n3 carries nothing. Edges mirror it under `w`.
fn presence_fixture() -> Graph {
    let lines = [
        r#"{"type":"node","id":"n0","labels":["V"],"properties":{"a":1,"b":2}}"#,
        r#"{"type":"node","id":"n1","labels":["V"],"properties":{"a":3}}"#,
        r#"{"type":"node","id":"n2","labels":["V"],"properties":{"a":null}}"#,
        r#"{"type":"node","id":"n3","labels":["V"],"properties":{}}"#,
        r#"{"type":"edge","id":"e0","from":"n0","to":"n1","labels":["R"],"properties":{"w":1}}"#,
        r#"{"type":"edge","id":"e1","from":"n1","to":"n2","labels":["R"],"properties":{}}"#,
    ]
    .join("\n");

    crate::ndjson::decode(&lines).expect("fixture decodes")
}

/// A stored null is PRESENT — the engine's null model is a value, not an
/// absence, so `has('a')` counts it and `hasNot('a')` does not. Getting this
/// backwards is the whole risk of lowering presence into the columns, since the
/// packed columns flag it exactly and a naive "is it null" test would not.
#[test]
fn a_stored_null_is_present() {
    let mut g = presence_fixture();

    assert_eq!(count_of(&mut g, "g.V().has('a').count()"), 3.0);
    assert_eq!(count_of(&mut g, "g.V().hasNot('a').count()"), 1.0);
    // And it is still a null when read.
    let vals = super::parse::parse("g.V().has('a').values('a')")
        .expect("parses")
        .run(&mut g);

    assert!(
        vals.iter().any(|v| matches!(v, GVal::Null)),
        "the stored null did not survive: {vals:?}"
    );
}

#[test]
fn presence_narrows_the_same_rows_the_stream_would() {
    let mut g = presence_fixture();

    assert_eq!(count_of(&mut g, "g.V().has('b').count()"), 1.0);
    assert_eq!(count_of(&mut g, "g.V().hasNot('b').count()"), 3.0);
    // A key no element carries: nothing has it, everything lacks it.
    assert_eq!(count_of(&mut g, "g.V().has('zz').count()"), 0.0);
    assert_eq!(count_of(&mut g, "g.V().hasNot('zz').count()"), 4.0);
    // Edges have their own store.
    assert_eq!(count_of(&mut g, "g.E().has('w').count()"), 1.0);
    assert_eq!(count_of(&mut g, "g.E().hasNot('w').count()"), 1.0);
    // Composed with a label and with a value predicate.
    assert_eq!(
        count_of(&mut g, "g.V().hasLabel('V').has('a').count()"),
        3.0
    );
    assert_eq!(count_of(&mut g, "g.V().has('a').has('b').count()"), 1.0);
    assert_eq!(count_of(&mut g, "g.V().has('a', 3).has('b').count()"), 0.0);
    assert_eq!(count_of(&mut g, "g.V().hasNot('b').has('a').count()"), 2.0);
}

/// `hasKey('a','b')` means EITHER key — a disjunction the presence list cannot
/// express, so it must decline to lower rather than read as "both".
#[test]
fn a_multi_key_presence_test_is_any_of_them() {
    let mut g = presence_fixture();

    assert_eq!(count_of(&mut g, "g.V().hasKey('a','b').count()"), 3.0);
    assert_eq!(count_of(&mut g, "g.V().hasKey('a').count()"), 3.0);
    assert_eq!(count_of(&mut g, "g.V().hasKey('b').count()"), 1.0);
    assert_eq!(count_of(&mut g, "g.V().hasNot('a','b').count()"), 1.0);
}

/// The paging cap and presence compose: the cap counts SURVIVORS.
#[test]
fn a_capped_scan_counts_rows_that_survive_presence() {
    let mut g = presence_fixture();

    assert_eq!(count_of(&mut g, "g.V().has('a').limit(2).count()"), 2.0);
    assert_eq!(count_of(&mut g, "g.V().has('b').limit(2).count()"), 1.0);
    assert_eq!(count_of(&mut g, "g.V().hasNot('a').limit(5).count()"), 1.0);
}

// --- an `or()` of comparisons lowers into the seek's branches ----------------

/// The disjunction has to hold with NO index to answer it — it used to be a seed
/// and nothing else, so an unindexed one simply did not apply. These fixtures
/// have no index at all, which is the case that would have silently returned
/// every row.
#[test]
fn an_or_of_comparisons_narrows_without_an_index() {
    let mut lines = String::new();

    for i in 0..60usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{},\"s\":\"s{}\"}}}}\n",
            i % 10,
            i % 4
        ));
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    // 6 vertices per residue class.
    assert_eq!(count_of(&mut g, "g.V().or(__.has('n', 3)).count()"), 6.0);
    assert_eq!(
        count_of(&mut g, "g.V().or(__.has('n', 3), __.has('n', 9)).count()"),
        12.0
    );
    // The same value twice is still that value's rows, once each.
    assert_eq!(
        count_of(&mut g, "g.V().or(__.has('n', 3), __.has('n', 3)).count()"),
        6.0
    );
    // Different keys, and a mix of ops.
    assert_eq!(
        count_of(
            &mut g,
            "g.V().or(__.has('n', 0), __.has('s', 's1')).count()"
        ),
        21.0
    );
    assert_eq!(
        count_of(
            &mut g,
            "g.V().or(__.has('n', lt(2)), __.has('n', gte(8))).count()"
        ),
        24.0
    );
    // Composed with a conjunct and a label: the AND of the two.
    assert_eq!(
        count_of(
            &mut g,
            "g.V().hasLabel('V').has('s', 's1').or(__.has('n', 3), __.has('n', 9)).count()"
        ),
        // n=3 lands on i=3,13,23,33,43,53 and n=9 on i=9,…,59; three of each also
        // carry s1 (i % 4 == 1).
        6.0
    );
    // And with the cap, which applies the disjunction on a different code path.
    assert_eq!(
        count_of(
            &mut g,
            "g.V().or(__.has('n', 3), __.has('n', 9)).limit(5).count()"
        ),
        5.0
    );
    assert_eq!(
        count_of(
            &mut g,
            "g.V().or(__.has('n', 3), __.has('n', 9)).limit(50).count()"
        ),
        12.0
    );
}

/// A branch that is not a single comparison must decline to lower rather than be
/// dropped — dropping a BRANCH loses rows, and capturing the step leaves nothing
/// to re-check.
#[test]
fn an_or_of_anything_else_still_answers() {
    let mut lines = String::new();

    for i in 0..20usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{}}}}}\n",
            i % 5
        ));
    }
    for i in 0..20usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
            (i + 1) % 20
        ));
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    // A branch with two comparisons in it (an AND inside the OR).
    assert_eq!(
        count_of(
            &mut g,
            "g.V().or(__.has('n', 1).has('n', 1), __.has('n', 2)).count()"
        ),
        8.0
    );
    // A branch that walks.
    assert_eq!(
        count_of(&mut g, "g.V().or(__.out('R'), __.has('n', 0)).count()"),
        20.0
    );
    // A branch that is a label test.
    assert_eq!(
        count_of(&mut g, "g.V().or(__.hasLabel('V'), __.has('n', 0)).count()"),
        20.0
    );
    // An empty `or()` matches nothing, as TinkerPop's does.
    assert_eq!(count_of(&mut g, "g.V().or().count()"), 0.0);
}

// --- a grouped numeric fold off the frontier ---------------------------------

/// Every case where "read two columns and bucket them" could differ from
/// "bucket the traversers and run a sub-traversal per group": a group key that is
/// absent, a value that is absent, a STORED null on either side, a group with no
/// number in it at all, and a NaN key (which has no dedup key and falls back to a
/// linear scan).
fn grouped_fold_fixture() -> Graph {
    let lines = [
        r#"{"type":"node","id":"n0","labels":["V"],"properties":{"k":"a","v":1}}"#,
        r#"{"type":"node","id":"n1","labels":["V"],"properties":{"k":"a","v":2}}"#,
        r#"{"type":"node","id":"n2","labels":["V"],"properties":{"k":"b","v":10}}"#,
        // the key, no value
        r#"{"type":"node","id":"n3","labels":["V"],"properties":{"k":"b"}}"#,
        // no key, a value
        r#"{"type":"node","id":"n4","labels":["V"],"properties":{"v":100}}"#,
        // neither
        r#"{"type":"node","id":"n5","labels":["V"],"properties":{}}"#,
        // a stored null key
        r#"{"type":"node","id":"n6","labels":["V"],"properties":{"k":null,"v":7}}"#,
        // a group whose every value is a stored null
        r#"{"type":"node","id":"n7","labels":["V"],"properties":{"k":"z","v":null}}"#,
        r#"{"type":"node","id":"n8","labels":["V"],"properties":{"k":"z","v":null}}"#,
        // a value that is not a number at all
        r#"{"type":"node","id":"n9","labels":["V"],"properties":{"k":"s","v":"text"}}"#,
        r#"{"type":"node","id":"n10","labels":["V"],"properties":{"k":"s","v":4}}"#,
        // A group whose sum DEPENDS on the order it is folded in. At 1e16 the
        // gap between representable doubles is 2, so `1e16 + 1 + 1` keeps losing
        // the 1 and stays 1e16, while `1 + 1 + 1e16` carries a 2 in and gives
        // 1.0000000000000002e16. Without a group like this every sum here is
        // exact in either direction, and a fold that visited members backwards
        // passed the test.
        r#"{"type":"node","id":"n11","labels":["V"],"properties":{"k":"f","v":1e16}}"#,
        r#"{"type":"node","id":"n12","labels":["V"],"properties":{"k":"f","v":1}}"#,
        r#"{"type":"node","id":"n13","labels":["V"],"properties":{"k":"f","v":1}}"#,
        r#"{"type":"edge","id":"e0","from":"n0","to":"n1","labels":["R"],"properties":{"ek":"x","ev":3}}"#,
        r#"{"type":"edge","id":"e1","from":"n1","to":"n2","labels":["R"],"properties":{"ek":"x","ev":4}}"#,
        r#"{"type":"edge","id":"e2","from":"n2","to":"n0","labels":["R"],"properties":{"ek":"y"}}"#,
    ]
    .join("\n");

    crate::ndjson::decode(&lines).expect("fixture decodes")
}

#[test]
fn a_grouped_fold_off_a_frontier_matches_the_stream() {
    let mut g = grouped_fold_fixture();

    for reduce in ["sum", "max", "min", "mean"] {
        same_via_stream(
            &mut g,
            &format!("g.V().group().by('k').by(__.values('v').{reduce}())"),
        );
        same_via_stream(
            &mut g,
            &format!("g.E().group().by('ek').by(__.values('ev').{reduce}())"),
        );
        // Grouping BY the value and folding the key, so the absent side swaps.
        same_via_stream(
            &mut g,
            &format!("g.V().group().by('v').by(__.values('v').{reduce}())"),
        );
    }
}

/// The rules the fold has to reproduce, spelled out rather than only compared —
/// a pairing test says the two agree, not that either is right.
#[test]
fn a_grouped_fold_skips_nulls_and_keys_absences_as_null() {
    let mut g = grouped_fold_fixture();
    let entries = |g: &mut Graph, src: &str| -> Vec<(String, String)> {
        match super::parse::parse(src).expect("parses").run(g).as_slice() {
            [GVal::Map(m)] => m
                .iter()
                .map(|(k, v)| (format!("{k:?}"), format!("{v:?}")))
                .collect(),
            other => panic!("expected one map, got {other:?}"),
        }
    };
    let got = entries(&mut g, "g.V().group().by('k').by(__.values('v').sum())");

    assert_eq!(
        got,
        vec![
            // 1 + 2
            ("Str(\"a\")".into(), "Num(3.0)".into()),
            // n3 has the key and no value: the group exists, the value is skipped.
            ("Str(\"b\")".into(), "Num(10.0)".into()),
            // An ABSENT key and a STORED NULL key are the same group, which is
            // TinkerPop 3.5's rule for the first (a missing property groups under
            // null) and this engine's null model for the second. 100 + 7.
            ("Null".into(), "Num(107.0)".into()),
            // Every value in the group is a stored null: no number, so null —
            // NOT zero. (TinkerPop yields no result at all here; see
            // `reduce_nums`, which records the divergence.)
            ("Str(\"z\")".into(), "Null".into()),
            // A non-number is skipped like a null, leaving the one number.
            ("Str(\"s\")".into(), "Num(4.0)".into()),
            // Folded in FRONTIER order: 1e16 + 1 + 1 stays 1e16, where the
            // other order gives 1.0000000000000002e16. Float addition is not
            // associative, so the order is part of the answer.
            ("Str(\"f\")".into(), "Num(1e16)".into()),
        ]
    );
    // Groups come out in FIRST-SEEN order, which the vec above is asserting.
    assert_eq!(
        got.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
        vec![
            "Str(\"a\")",
            "Str(\"b\")",
            "Null",
            "Str(\"z\")",
            "Str(\"s\")",
            "Str(\"f\")"
        ]
    );
}

/// `count()` as the value-`by()` must NOT take the column arm: a column read
/// cannot tell an absent key from a stored null, and a count is the one reducer
/// that has to. n7/n8 hold stored nulls, so the `z` group counts 2 either way —
/// the case that discriminates is the null-KEY group, whose members are an absent
/// key twice and a stored null once.
#[test]
fn a_grouped_count_is_not_a_column_fold() {
    let mut g = grouped_fold_fixture();

    same_via_stream(&mut g, "g.V().group().by('k').by(__.count())");
    same_via_stream(&mut g, "g.V().group().by('k').by(__.values('v').count())");
    // Two keys read per element is not one column.
    same_via_stream(&mut g, "g.V().group().by('k').by(__.values('v','k').sum())");
    // No value-by at all: the members themselves.
    same_via_stream(&mut g, "g.V().group().by('k')");
    // A value-by that walks.
    same_via_stream(&mut g, "g.V().group().by('k').by(__.out('R').count())");
}

// --- a negated comparison lowers, and it is not a not-equal ------------------

/// `not(has(k, v))` is satisfied by an element that does not carry `k` at all.
/// TinkerPop's framing: if you have no sister, "is your sister older than 30" is
/// no and "is your sister younger than 30" is also no — so such an element fails
/// every `has(k, …)` and therefore satisfies every `not(has(k, …))`.
///
/// The fixture needs an element missing the key, or the assertion means nothing —
/// and it must NOT contain a stored null, which promotes the column to `Mixed`
/// and makes the whole seek decline to the stream. That was the first version of
/// this test: it asserted the right numbers, and the numbers came from the path
/// it was written to check the OTHER one against.
#[test]
fn a_negated_has_includes_elements_without_the_key() {
    let lines = [
        r#"{"type":"node","id":"n0","labels":["V"],"properties":{"n":3}}"#,
        r#"{"type":"node","id":"n1","labels":["V"],"properties":{"n":4}}"#,
        r#"{"type":"node","id":"n2","labels":["V"],"properties":{"n":5}}"#,
        // no `n` at all
        r#"{"type":"node","id":"n3","labels":["V"],"properties":{"other":1}}"#,
        r#"{"type":"node","id":"n4","labels":["V"],"properties":{}}"#,
    ]
    .join("\n");
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    // n1, n2 (present and not 3) PLUS n3, n4 (no key at all).
    assert_eq!(count_of(&mut g, "g.V().not(__.has('n', 3)).count()"), 4.0);
    // In THIS engine `neq` reads the same way — see the divergence recorded on
    // the `P::Neq` arm of `lower_predicate`. TinkerPop gives 2 here, and the TS
    // engine gives 4 like this one, so the two spellings are interchangeable here
    // and are not in TinkerPop.
    assert_eq!(count_of(&mut g, "g.V().has('n', neq(3)).count()"), 4.0);
    assert_eq!(count_of(&mut g, "g.V().has('n', 3).count()"), 1.0);

    // Every ordering comparison, negated; n3/n4 are in all of them.
    assert_eq!(
        count_of(&mut g, "g.V().not(__.has('n', gt(3))).count()"),
        3.0
    );
    assert_eq!(
        count_of(&mut g, "g.V().not(__.has('n', gte(4))).count()"),
        3.0
    );
    assert_eq!(
        count_of(&mut g, "g.V().not(__.has('n', lt(5))).count()"),
        3.0
    );
    assert_eq!(
        count_of(&mut g, "g.V().not(__.has('n', lte(3))).count()"),
        4.0
    );
    // A key nothing carries: the negation keeps everything.
    assert_eq!(count_of(&mut g, "g.V().not(__.has('zz', 1)).count()"), 5.0);
    assert_eq!(count_of(&mut g, "g.V().has('zz', neq(1)).count()"), 5.0);
    // Presence AND the negation is the pair that DOES exclude a missing key, and
    // it is what TinkerPop's `neq` means. Spelled out, it works here too.
    assert_eq!(
        count_of(&mut g, "g.V().has('n').not(__.has('n', 3)).count()"),
        2.0
    );
    // Two negations, and the scan cap (a different code path in the seek).
    assert_eq!(
        count_of(
            &mut g,
            "g.V().not(__.has('n', 3)).not(__.has('n', 4)).count()"
        ),
        3.0
    );
    // The CAPPED scan path. A count alone cannot see whether the cap applied the
    // negation — `limit(2)` of five candidates and `limit(2)` of four matches are
    // both 2 — so this asks WHICH rows, and asks for a limit above the match
    // count as well.
    assert_eq!(
        super::parse::parse("g.V().not(__.has('n', 3)).limit(2).id()")
            .expect("parses")
            .run(&mut g)
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>(),
        vec!["Str(\"n1\")", "Str(\"n2\")"]
    );
    assert_eq!(
        count_of(&mut g, "g.V().not(__.has('n', 3)).limit(5).count()"),
        4.0
    );
    // Labels compose with it, which is the shape that actually lowers in the
    // benchmark.
    assert_eq!(
        count_of(&mut g, "g.V().hasLabel('V').not(__.has('n', 3)).count()"),
        4.0
    );
}

/// A STORED NULL is present and satisfies no comparison, so it satisfies every
/// negation of one. A NaN would be the same case, and cannot arise: every write
/// entry point coerces a non-finite number to null, so there is no NaN in a
/// column to test against.
#[test]
fn a_stored_null_satisfies_every_negated_comparison() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":1}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":null}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // `a` matches each comparison and is excluded; the stored null never does.
    assert_eq!(count_of(&mut g, "g.V().not(__.has('n', 1)).count()"), 1.0);
    assert_eq!(
        count_of(&mut g, "g.V().not(__.has('n', gt(0))).count()"),
        1.0
    );
    assert_eq!(
        count_of(&mut g, "g.V().not(__.has('n', lt(9))).count()"),
        1.0
    );
    // And it IS present, so presence and negation disagree about it on purpose.
    assert_eq!(count_of(&mut g, "g.V().has('n').count()"), 2.0);
}

/// What a NEGATIVE `has(k, …)` predicate does with an element that lacks `k`.
///
/// TinkerPop's rule is uniform: `has(k, P)` filters out an element without `k`
/// whatever `P` is, because there is no value for the predicate to be applied to.
/// This engine is NOT uniform, and this test pins which is which rather than
/// leaving it to be discovered:
///
/// ```text
///                          here   TinkerPop
///   neq(v)                  keeps   drops
///   without(v)              keeps   drops
///   outside(lo, hi)         drops   drops
///   notContaining(s)        drops   drops
/// ```
///
/// The TS engine gives the same five answers, so the two are consistent with each
/// other and no differential fuzzer can see any of this. That is also why it is
/// not fixed here: `neq` and `without` returning fewer rows is a behavior change
/// to both engines, and it is a decision, not a bug fix to one side.
///
/// `not(__.has(k, v))` is a different question and DOES keep such an element in
/// both this engine and TinkerPop — see
/// `a_negated_has_includes_elements_without_the_key`.
#[test]
fn a_negative_predicate_does_not_treat_a_missing_key_uniformly() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":3,"s":"xy"}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":4,"s":"zz"}}"#,
            // no properties at all
            r#"{"type":"node","id":"c","labels":["V"],"properties":{}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    // KEEPS `c`. TinkerPop answers 1 for both of these.
    assert_eq!(count_of(&mut g, "g.V().has('n', neq(3)).count()"), 2.0);
    assert_eq!(count_of(&mut g, "g.V().has('n', without(3)).count()"), 2.0);
    // DROPS `c`, which is what TinkerPop does for all four.
    assert_eq!(
        count_of(&mut g, "g.V().has('n', outside(2, 4)).count()"),
        0.0
    );
    assert_eq!(
        count_of(&mut g, "g.V().has('s', notContaining('x')).count()"),
        1.0
    );
    // The positive predicates agree with TinkerPop and with each other.
    assert_eq!(
        count_of(&mut g, "g.V().has('n', within(3, 4)).count()"),
        2.0
    );
    assert_eq!(count_of(&mut g, "g.V().has('n', gt(0)).count()"), 2.0);
}

/// Sorting a numeric column on the raw `f64` has to give the same order as
/// sorting the boxed values, tie for tie.
///
/// It does by construction — `gcmp_total`'s non-NaN numeric arm IS `total_cmp`,
/// and the arm declines a column with a NaN in it before reaching either. The
/// case worth pinning anyway is `-0.0`, which `total_cmp` orders BEFORE `0.0`
/// while equality calls them equal: if the fast path had used `partial_cmp` the
/// two would tie, the index tie-break would put them in frontier order, and a
/// `limit` across that boundary would return a different row.
#[test]
fn ordering_a_numeric_column_matches_the_boxed_sort() {
    let mut lines = String::new();

    // Duplicates so ties are everywhere, and both zeroes, and both infinities.
    for (i, v) in [
        "1", "-0.0", "0.0", "3", "1", "-1", "1e308", "-1e308", "0.0", "-0.0", "2", "2",
    ]
    .iter()
    .enumerate()
    {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{v}}}}}\n"
        ));
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    for spelling in ["order().by('n')", "order().by('n', desc)"] {
        same_via_stream(&mut g, &format!("g.V().{spelling}.id()"));
        same_via_stream(&mut g, &format!("g.V().{spelling}.values('n')"));

        for k in [1usize, 2, 5, 11, 12, 50] {
            same_via_stream(&mut g, &format!("g.V().{spelling}.limit({k}).id()"));
        }
        same_via_stream(&mut g, &format!("g.V().{spelling}.range(2, 6).id()"));
        same_via_stream(&mut g, &format!("g.V().{spelling}.tail(3).id()"));
    }
    // A column that is NOT all numbers keeps the boxed comparator.
    let mut mixed = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":"s"}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":"t"}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    same_via_stream(&mut mixed, "g.V().order().by('n').id()");
}

/// A NaN cannot be STORED, by any route.
///
/// It is normalized to null on ingest, and — since the write path started
/// applying that same rule to computed values — on write too. The four functions
/// here are the ones a comment on `distinct_values` named as the way a NaN got
/// into a column, back when they did.
///
/// This is pinned because two guards depend on knowing which way it is: the
/// numeric column arms DECLINE when a NaN is present, and `dedup` treats a NaN as
/// never-a-duplicate. Both are still right — a COMPUTED column holds NaNs
/// (`RETURN sqrt(-1)` is `Num(NaN)`) even though a stored one cannot — but a
/// comment claiming the storage route is open would send the next person looking
/// for a case that no longer exists.
#[test]
fn a_nan_cannot_be_stored_by_any_route() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":0}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":0}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    for f in ["sqrt(-1)", "asin(2)", "acos(2)", "power(-1, 0.5)"] {
        let set = format!("MATCH (u:V) SET u.z = {f} RETURN count(*) AS c");

        crate::gql::parse(&set)
            .expect("parses")
            .execute(&mut g, &crate::gql::eval::Params::new())
            .expect("runs");

        let read = crate::gql::parse("MATCH (u:V) RETURN u.z AS z")
            .expect("parses")
            .execute(&mut g, &crate::gql::eval::Params::new())
            .expect("runs");

        for row in read.rows() {
            assert!(
                matches!(row[0], crate::graph::Value::Null),
                "`{f}` stored something that is not null: {:?}",
                row[0]
            );
        }
    }
    // The Gremlin write path agrees.
    super::parse::parse("g.V().property('m', __.math('0/0'))")
        .expect("parses")
        .run(&mut g);

    assert!(
        super::parse::parse("g.V().values('m')")
            .expect("parses")
            .run(&mut g)
            .iter()
            .all(|v| matches!(v, GVal::Null)),
        "a computed NaN reached a stored column through Gremlin"
    );
    // But a COMPUTED column still holds one, which is why the guards stay.
    let computed = crate::gql::parse("MATCH (u:V) RETURN sqrt(-1) AS z")
        .expect("parses")
        .execute(&mut g, &crate::gql::eval::Params::new())
        .expect("runs");

    assert!(
        computed
            .rows()
            .all(|r| matches!(r[0], crate::graph::Value::Num(x) if x.is_nan())),
        "a computed NaN stopped being a NaN"
    );
}

/// `where`/`not` over the element's OWN property runs as a column test, and both
/// spellings of it agree with the stream.
///
/// Two things had to change for this: the column layer only understood a
/// `where` body that HOPS, and `lower_prefix` consumed a `not()` it could not
/// capture — which declined the whole lowering rather than leaving the step for
/// the arm that handles it. The `not` case measured 2.488ms against 0.058 after.
#[test]
fn a_self_predicate_where_matches_the_stream() {
    let mut lines = String::new();

    for i in 0..200usize {
        let l = if i % 5 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };

        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"s\":\"k{}\"}}}}\n",
            i % 13,
            i % 3
        ));
    }
    // A vertex with no `n` at all, and one whose `n` is a stored null.
    lines.push_str(r#"{"type":"node","id":"x","labels":["V"],"properties":{"s":"k0"}}"#);
    lines.push('\n');
    lines.push_str(r#"{"type":"node","id":"y","labels":["V"],"properties":{"n":null,"s":"k0"}}"#);
    lines.push('\n');

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    for q in [
        "g.V().where(__.values('n').is(gt(5))).count()",
        "g.V().not(__.values('n').is(gt(5))).count()",
        "g.V().where(__.has('n', gt(5))).count()",
        "g.V().not(__.has('n', gt(5))).count()",
        "g.V().hasLabel('V').where(__.values('n').is(7)).count()",
        "g.V().hasLabel('V').not(__.values('n').is(7)).count()",
        // A key nothing carries, and a string predicate.
        "g.V().where(__.values('zz').is(1)).count()",
        "g.V().not(__.values('zz').is(1)).count()",
        "g.V().where(__.values('s').is('k1')).count()",
        // Composed with a hop on either side.
        "g.V().where(__.values('n').is(gt(5))).count()",
        "g.V().not(__.values('n').is(gt(5))).dedup().count()",
    ] {
        let column = count_of(&mut g, q);
        // `fold().unfold()` means the same and takes the STREAM.
        let (head, tail) = q.split_once('.').expect("a traversal has a step");
        let streamed = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));

        assert_eq!(
            column,
            count_of(&mut g, &streamed),
            "`{q}` disagreed with its streamed spelling"
        );
    }
}

/// A step that does NOTHING must not change the answer or the route.
///
/// `barrier()` and `identity()` are `=> stream` in the step interpreter — pure
/// identities. The column layer had no arm for them, so a traversal containing
/// one fell off the columnar route entirely and ran as a stream: `barrier()`
/// before a `count()` cost 0.838ms against 0.002, and `identity()` before
/// `values()` 1.178ms against 0.036.
///
/// The rule this pins is the general one: inserting an identity step anywhere is
/// not allowed to change what comes back.
#[test]
fn an_identity_step_changes_nothing() {
    let mut g = modern();

    for (plain, with_noop) in [
        ("g.V().count()", "g.V().barrier().count()"),
        ("g.V().count()", "g.V().identity().count()"),
        ("g.V().values('name')", "g.V().barrier().values('name')"),
        ("g.V().values('name')", "g.V().identity().values('name')"),
        (
            "g.V().out('KNOWS').id()",
            "g.V().out('KNOWS').barrier().id()",
        ),
        (
            "g.V().values('age').sum()",
            "g.V().identity().values('age').sum()",
        ),
        (
            "g.V().order().by('name').id()",
            "g.V().barrier().order().by('name').id()",
        ),
        ("g.V().dedup().count()", "g.V().identity().dedup().count()"),
        (
            "g.V().groupCount().by('name')",
            "g.V().barrier().groupCount().by('name')",
        ),
        // Two in a row, and one at the very end.
        ("g.V().count()", "g.V().barrier().identity().count()"),
        ("g.V().values('name')", "g.V().values('name').barrier()"),
    ] {
        let a = super::parse::parse(plain)
            .unwrap_or_else(|e| panic!("`{plain}`: {e}"))
            .run(&mut g);
        let b = super::parse::parse(with_noop)
            .unwrap_or_else(|e| panic!("`{with_noop}`: {e}"))
            .run(&mut g);

        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "`{with_noop}` differed from `{plain}`"
        );
        assert!(!a.is_empty(), "`{plain}` returned nothing to compare");
    }
}

/// A `where`/`not` on an EDGE's own property is a column test too.
///
/// Two things kept it on the stream: `self_predicate` built a vertex-side seek,
/// and the `E()` allowlist — which says what may follow an edge frontier without
/// being read as a hop off edge ids — did not list the filters. A filter cannot
/// navigate, so it belongs there.
///
/// The adjacency-shaped bodies must still DECLINE from an edge frontier, because
/// `has_adj` walks a vertex's adjacency and an edge id read as one would walk
/// whatever vertex shares its number. That is what the last cases check.
#[test]
fn a_where_on_an_edge_property_matches_the_stream() {
    let mut lines = String::new();

    for i in 0..60usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{}}}}\n"
        ));
    }
    for i in 0..60usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{}}}}}\n",
            (i * 7 + 1) % 60,
            i % 4
        ));
    }
    // An edge with no `w` at all.
    lines.push_str(
        r#"{"type":"edge","id":"ex","from":"n0","to":"n1","labels":["R"],"properties":{}}"#,
    );
    lines.push('\n');

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    for q in [
        "g.E().where(__.values('w').is(1)).count()",
        "g.E().not(__.values('w').is(1)).count()",
        "g.E().where(__.has('w', gt(1))).count()",
        "g.E().not(__.has('w', gt(1))).count()",
        "g.E().hasLabel('R').where(__.values('w').is(0)).count()",
        "g.E().where(__.values('w').is(1)).id()",
        // A key no edge carries.
        "g.E().where(__.values('zz').is(1)).count()",
        "g.E().not(__.values('zz').is(1)).count()",
        // Adjacency-shaped from an EDGE frontier: must decline, and still answer.
        "g.E().where(__.out('R')).count()",
        "g.E().not(__.out('R')).count()",
    ] {
        let (head, tail) = q.split_once('.').expect("a traversal has a step");
        let streamed = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));
        let a = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}`: {e}"))
            .run(&mut g);
        let b = super::parse::parse(&streamed)
            .unwrap_or_else(|e| panic!("`{streamed}`: {e}"))
            .run(&mut g);

        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "`{q}` disagreed with its streamed spelling"
        );
    }
}

/// An edge frontier's endpoints, off the column.
///
/// `e_src`/`e_dst` are indexed BY EDGE, so `outV`/`inV`/`bothV` are gathers
/// rather than walks — which is why they are safe after an `E()` source where a
/// step that read an edge id as a vertex id would not be. 6.994ms -> 0.153 for
/// `E().hasLabel(R).inV().count()`.
///
/// `otherV()` must still take the stream: "the end I did not come from" is a
/// question about the PATH, and a column has none. The last cases check it still
/// answers.
#[test]
fn edge_endpoints_off_a_column_match_the_stream() {
    let lines = [
        r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":1}}"#,
        r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":2}}"#,
        r#"{"type":"node","id":"c","labels":["V"],"properties":{"n":3}}"#,
        r#"{"type":"edge","id":"e0","from":"a","to":"b","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e1","from":"b","to":"c","labels":["R"],"properties":{}}"#,
        // a self-loop: both ends are the same vertex
        r#"{"type":"edge","id":"e2","from":"c","to":"c","labels":["R"],"properties":{}}"#,
        r#"{"type":"edge","id":"e3","from":"a","to":"c","labels":["S"],"properties":{}}"#,
    ];
    let mut g = crate::ndjson::decode(&lines.join("\n")).expect("fixture decodes");

    for q in [
        "g.E().inV().id()",
        "g.E().outV().id()",
        "g.E().bothV().id()",
        "g.E().hasLabel('R').inV().id()",
        "g.E().hasLabel('R').outV().values('n')",
        "g.E().inV().count()",
        "g.E().bothV().count()",
        "g.E().bothV().dedup().count()",
        "g.E().inV().dedup().id()",
        "g.E().hasLabel('R').inV().hasLabel('V').count()",
        // Path-dependent: stays on the stream, must still be right.
        "g.V().bothE('R').otherV().id()",
    ] {
        let (head, tail) = q.split_once('.').expect("a traversal has a step");
        let streamed = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));
        let a = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}`: {e}"))
            .run(&mut g);
        let b = super::parse::parse(&streamed)
            .unwrap_or_else(|e| panic!("`{streamed}`: {e}"))
            .run(&mut g);

        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "`{q}` disagreed with its streamed spelling"
        );
        assert!(!a.is_empty(), "`{q}` returned nothing to compare");
    }
}

/// `unfold()` off a column: a LIST expands to its elements, anything else passes
/// through.
///
/// An element column and an unboxed scalar column can never hold a list, so for
/// those it is the identity. Only a boxed column actually flattens.
///
/// Also pins the fact the OTHER tests in this file depend on: `fold()` with
/// anything after it still declines to the stream, which is what makes
/// `fold().unfold()` a reliable way to force the streamed spelling of a
/// traversal. Measured: 1.192ms against 0.002 for the same count without it. If
/// that ever stops being true, every `same_via_stream` pairing here quietly
/// starts comparing the column path against itself.
#[test]
fn unfold_off_a_column_matches_the_stream() {
    let mut g = crate::ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"n":1,"xs":[1,2,3]}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"n":2,"xs":[]}}"#,
            r#"{"type":"node","id":"c","labels":["V"],"properties":{"n":3}}"#,
        ]
        .join("\n"),
    )
    .expect("fixture decodes");

    for q in [
        // Identity cases: never a list.
        "g.V().unfold().count()",
        "g.V().values('n').unfold().count()",
        "g.V().values('n').unfold().sum()",
        "g.V().unfold().id()",
        // A real stored list, including an EMPTY one (which contributes nothing).
        "g.V().values('xs').unfold().count()",
        "g.V().values('xs').unfold().sum()",
        "g.V().values('xs').unfold().dedup().count()",
        // Two in a row.
        "g.V().values('n').unfold().unfold().count()",
    ] {
        let (head, tail) = q.split_once('.').expect("a traversal has a step");
        let streamed = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));
        let a = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}`: {e}"))
            .run(&mut g);
        let b = super::parse::parse(&streamed)
            .unwrap_or_else(|e| panic!("`{streamed}`: {e}"))
            .run(&mut g);

        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "`{q}` disagreed with its streamed spelling"
        );
    }
}

/// `as('x')` writes a tag onto every traverser. When nothing downstream READS a
/// tag, that is work with no reader, and the column path can carry on — 2.128ms
/// against 0.002 for a count behind an unread tag.
///
/// The other half is checked too — a traversal that DOES read one must still be
/// right — but be clear about WHY it is right today: not because of the
/// `reads_tags` guard, which a mutation can remove without failing anything, but
/// because every tag reader (`select`, `dedup('a')`, a tag `where`) has no column
/// arm and declines further down, so the stream re-runs the traversal from the
/// start. The guard is what will keep it right when the column path learns those
/// steps. Recorded here so nobody reads these cases as covering it.
#[test]
fn an_unread_tag_is_free_and_a_read_one_still_works() {
    let mut g = modern();

    for q in [
        // Nothing reads the tag.
        "g.V().as('x').count()",
        "g.V().as('x').values('name')",
        "g.V().as('x').out('KNOWS').id()",
        "g.V().as('x').as('y').count()",
        // Something does.
        "g.V().as('x').select('x').id()",
        "g.V().as('x').out('KNOWS').select('x').id()",
        "g.V().as('x').out('KNOWS').as('y').select('x', 'y').count()",
        "g.V().as('x').dedup('x').count()",
        "g.V().as('x').where(__.select('x')).count()",
    ] {
        let (head, tail) = q.split_once('.').expect("a traversal has a step");
        let streamed = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));
        let a = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}`: {e}"))
            .run(&mut g);
        let b = super::parse::parse(&streamed)
            .unwrap_or_else(|e| panic!("`{streamed}`: {e}"))
            .run(&mut g);

        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "`{q}` disagreed with its streamed spelling"
        );
        assert!(!a.is_empty(), "`{q}` returned nothing to compare");
    }
}

/// `select` off a tag COLUMN, and the alignment that makes it valid.
///
/// A tag is a snapshot of the column at the point `as()` ran, so `select` is a
/// column read — `V().hasLabel(L).as('x').select('x').count()` was 3.776ms and is
/// 0.009. (`select` across a HOP was already columnar via the pattern planner;
/// this is the hop-free case it declines.)
///
/// The snapshot is only valid while it stays ROW-ALIGNED with the current
/// column, and every case below with a step between the `as` and the `select`
/// exists because it was WRONG first: `as('x').limit(2).select('x')` returned six
/// rows against the stream's two, because the page applied to the current column
/// and not to the tag. Those steps now decline while a tag is live.
#[test]
fn select_off_a_tag_column_matches_the_stream() {
    let mut g = modern();

    for q in [
        // No step in between: the column read.
        "g.V().as('x').select('x').id()",
        "g.V().hasLabel('PERSON').as('x').select('x').count()",
        "g.V().as('x').select('x').values('name')",
        // Two tags, and the map that selecting both builds.
        "g.V().as('x').as('y').select('x','y').count()",
        // A step BETWEEN the tag and the read — each of these was wrong.
        "g.V().as('x').limit(2).select('x').id()",
        "g.V().as('x').skip(2).select('x').id()",
        "g.V().as('x').range(1,3).select('x').id()",
        "g.V().as('x').tail(2).select('x').id()",
        "g.V().as('x').dedup().select('x').id()",
        "g.V().as('x').out('KNOWS').select('x').id()",
        "g.V().as('x').order().by('name').select('x').id()",
        "g.V().as('x').where(__.has('name','marko')).select('x').id()",
        // The two spellings that are not a column read at all.
        "g.V().as('x').select(all, 'x').count()",
        "g.V().as('x').out('KNOWS').select('x').by('name').count()",
        // A label that was never bound: `select` drops every row.
        "g.V().as('x').select('zz').count()",
    ] {
        let (head, tail) = q.split_once('.').expect("a traversal has a step");
        let streamed = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));
        let a = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}`: {e}"))
            .run(&mut g);
        let b = super::parse::parse(&streamed)
            .unwrap_or_else(|e| panic!("`{streamed}`: {e}"))
            .run(&mut g);

        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "`{q}` disagreed with its streamed spelling"
        );
    }
}

/// A LABEL test on the element itself is a column question, like a property one.
///
/// `where(__.hasLabel(L))` / `not(__.hasLabel(L))` — 2.277ms against 0.028.
///
/// It needed the label-only case added to the answerable gate as well as the arm:
/// a seek holding nothing but a label has no property predicate, so `columnar`
/// says no by definition, and the arm built a perfectly good seek and then threw
/// it away. The measurement was identical before and after until that was fixed,
/// which is the only reason it was noticed.
#[test]
fn a_label_self_predicate_matches_the_stream() {
    let mut lines = String::new();

    for i in 0..120usize {
        let l = match i % 4 {
            0 => r#"["V","W"]"#,
            1 => r#"["V"]"#,
            2 => r#"["W"]"#,
            _ => r#"["V","X"]"#,
        };

        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{}}}}}\n",
            i % 7
        ));
    }
    for i in 0..40usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
            (i * 3 + 1) % 120
        ));
    }

    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    for q in [
        "g.V().where(__.hasLabel('W')).count()",
        "g.V().not(__.hasLabel('W')).count()",
        "g.V().hasLabel('V').not(__.hasLabel('W')).count()",
        "g.V().where(__.hasLabel('W')).id()",
        // Several names is a disjunction over ONE element.
        "g.V().where(__.hasLabel('W','X')).count()",
        "g.V().not(__.hasLabel('W','X')).count()",
        // A label NOTHING carries: `where` keeps none, `not` keeps all. This is
        // where an empty id list read as "no filter" would show.
        "g.V().where(__.hasLabel('ZZ')).count()",
        "g.V().not(__.hasLabel('ZZ')).count()",
        // One real name and one unknown: the unknown drops out.
        "g.V().where(__.hasLabel('W','ZZ')).count()",
        // Edges have labels too.
        "g.E().where(__.hasLabel('R')).count()",
        "g.E().not(__.hasLabel('R')).count()",
    ] {
        let (head, tail) = q.split_once('.').expect("a traversal has a step");
        let streamed = format!("{head}.{}", tail.replacen('.', ".fold().unfold().", 1));
        let a = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}`: {e}"))
            .run(&mut g);
        let b = super::parse::parse(&streamed)
            .unwrap_or_else(|e| panic!("`{streamed}`: {e}"))
            .run(&mut g);

        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "`{q}` disagreed with its streamed spelling"
        );
    }
}

/// `inject` inside a SUB-traversal, which the differential fuzzer does not reach.
///
/// It generates no `inject` and no `union`/`coalesce`/`choose` sub-traversals, so
/// TS-vs-native agreement here rests on tests like this one. The expectations
/// below were taken from the TS engine and match it value for value.
///
/// The semantics worth pinning: `inject` ADDS to the stream rather than replacing
/// it, so a sub-traversal that injects also passes its incoming traverser
/// through. That is why `coalesce(values('nope'), inject('fallback'))` yields the
/// fallback AND the vertex — the `constant()` idiom is the one that yields only a
/// fallback — and why `where(__.inject(1))` never filters anything, since a
/// sub-traversal that always emits is always true.
#[test]
fn inject_inside_a_subtraversal_agrees_with_the_ts_engine() {
    let mut g = modern();
    let run = |g: &mut Graph, q: &str| -> Vec<String> {
        super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}`: {e}"))
            .run(g)
            .iter()
            .map(|v| match v {
                GVal::Str(s) => s.to_string(),
                GVal::Num(n) => format!("{n}"),
                GVal::Node(_) => "<vertex>".to_string(),
                other => format!("{other:?}"),
            })
            .collect()
    };

    // The injected value, then the incoming vertex, then the other branch.
    assert_eq!(
        run(
            &mut g,
            "g.V().limit(1).union(__.inject('x'), __.values('name'))"
        ),
        vec!["x", "<vertex>", "marko"]
    );
    // 6 vertices: each yields the injection plus itself, plus 6 names.
    assert_eq!(
        run(
            &mut g,
            "g.V().union(__.inject('x'), __.values('name')).count()"
        ),
        vec!["18"]
    );
    // The fallback AND the vertex — `inject` is not `constant`.
    assert_eq!(
        run(
            &mut g,
            "g.V().limit(1).coalesce(__.values('nope'), __.inject('fallback'))"
        ),
        vec!["fallback", "<vertex>"]
    );
    // A sub-traversal that always emits is always true, so this filters nothing.
    assert_eq!(
        run(&mut g, "g.V().limit(1).where(__.inject(1)).count()"),
        vec!["1"]
    );
    // `map` takes the FIRST result, which is the injected value.
    assert_eq!(run(&mut g, "g.V().limit(1).map(__.inject('m'))"), vec!["m"]);
}

/// Ordering ELEMENTS against each other is a type fault, and BOTH engines agree.
///
/// Worth pinning because the agreement is not obvious and the cheap-looking "fix"
/// is wrong. The sort comparator calls `cmp_or_fault`, so `g.V().order()` — no
/// `by`, keys are the vertices themselves — raises E_INVALID_VALUE. That reads
/// like a bug: TinkerPop 3.5's ORDERABILITY is total and orders elements by id,
/// and `run()` returns correct rows while only the FALLIBLE path (the one the FFI
/// uses) errors. It is not a bug on our side alone: the TS engine raises "cannot
/// order an element with an element" for the same traversal. So the two engines
/// agree, and it is a SHARED deviation from TinkerPop — changing it is a
/// both-engines decision, not a one-side fix.
///
/// Swapping the comparator to the non-faulting `gcmp` was tried and reverted; it
/// makes native answer `g.V().order().count()` where TS still throws.
#[test]
fn order_over_elements_faults_like_the_ts_engine() {
    let mut g = modern();
    for q in [
        "g.V().order().count()",
        "g.V().order().by(desc).count()",
        "g.V().order().by(desc).id()",
    ] {
        let plan = super::parse::parse(q).expect("parses");
        assert_eq!(
            super::exec::try_run(&mut g, &plan).unwrap_err().code,
            crate::error_codes::ErrorCode::InvalidValue,
            "{q}"
        );
        // The infallible path is best-effort and must not panic.
        let _ = super::parse::parse(q).expect("parses").run(&mut g);
    }
}

/// The other half of the split still faults: an aggregate over genuinely
/// incomparable values raises, which is TinkerPop and what TS does.
#[test]
fn aggregate_over_mixed_types_still_faults() {
    let mut g = modern();
    let plan = super::parse::parse("g.V().values('name').inject(1).min()").expect("parses");
    assert!(
        super::exec::try_run(&mut g, &plan).is_err(),
        "min() over a string and a number should raise"
    );
}

/// A `limit(0)` does NOT excuse the work above it: this engine is EAGER, so a
/// side effect upstream of a zero-row slice still happens.
///
/// This is the one place the two engines genuinely disagree, and the fuzzer found
/// it (`FUZZ_SEED=4`, only reachable after the seeding fix). The TS engine is
/// lazy with a one-element pull-ahead — `limit(n)` pulls `n+1` — so its `limit(0)`
/// normally still touches the upstream and agrees with us. It stops agreeing when
/// that single pull can be satisfied WITHOUT reaching the upstream, which is
/// exactly what an `inject` in between does: it yields an injected value first.
///
///     g.V().aggregate('x').limit(0).cap('x')            both → 6 vertices
///     g.V().aggregate('x').inject(1).limit(1).cap('x')  both → 6 vertices
///     g.V().aggregate('x').inject(1).limit(0).cap('x')  native → 6, TS → []
///
/// So it is not a `limit(0)` rule and not an error-path curiosity — it is
/// lazy-vs-eager, observable on a perfectly valid query, narrowed to the case
/// where injected values alone cover the pull-ahead. Deciding it means making
/// native lazy or TS eager; both are architectural, so this pins what native does
/// and the fuzzer skips the shape by name rather than either engine changing
/// quietly.
/// The same lazy-vs-eager split, shielded by a BRANCH rather than an `inject`.
///
/// Native evaluates every branch of a `union` even when a `limit(0)` downstream
/// throws the rows away, so a fault in a LATER branch still surfaces. The TS
/// engine yields the first branch's row to satisfy its one-element pull-ahead and
/// never reaches the second, so it answers `[]`. Pinned so the eager behavior is
/// deliberate rather than incidental; see
/// `a_zero_limit_does_not_cancel_an_upstream_side_effect` for the full account.
#[test]
fn a_zero_limit_does_not_cancel_a_later_union_branch() {
    let mut g = modern();
    // `gte('lop')` compares a number against a string — a fault, in branch TWO.
    let q = "g.V().union(out('KNOWS'), has('age', gte('lop')).label()).limit(0).values('age')";
    let plan = super::parse::parse(q).expect("parses");
    assert_eq!(
        super::exec::try_run(&mut g, &plan).unwrap_err().code,
        crate::error_codes::ErrorCode::InvalidValue,
    );
    // And the infallible path must not panic.
    let _ = super::parse::parse(q).expect("parses").run(&mut g);
}

#[test]
fn a_zero_limit_does_not_cancel_an_upstream_side_effect() {
    let mut g = modern();
    for q in [
        "g.V().aggregate('x').limit(0).cap('x')",
        "g.V().aggregate('x').inject(1).limit(0).cap('x')",
    ] {
        let plan = super::parse::parse(q).expect("parses");
        let got = super::exec::try_run(&mut g, &plan).expect("no fault");
        let GVal::List(items) = &got[0] else {
            panic!("{q}: expected a list, got {got:?}");
        };
        assert_eq!(items.len(), 6, "{q}: the aggregate ran over every vertex");
    }
}

/// Every lowered BRANCH traversal must return exactly what the stream returns —
/// same rows, same order.
///
/// `fold().unfold()` is the lever: `fold()` with a tail declines to lower, so the
/// right-hand spelling runs the streamed interpreter over the same question while
/// the left-hand one takes the column arms. Comparing an engine against ITSELF
/// this way is what catches a lowering that is merely plausible.
///
/// ORDER is half the point. The stream is row-major
/// (`for t in stream { for p in plans { … } }`), so row 0's branches precede
/// row 1's; emitting branch-major returns the same multiset in a different
/// sequence, and every count- or set-based assertion would pass anyway.
#[test]
fn lowered_branch_steps_agree_with_the_stream() {
    let mut g = modern();

    for q in [
        // union: elements, values, mixed widths, and a per-row reducer
        "g.V().union(out('KNOWS'), out('CREATED'))",
        "g.V().union(out('KNOWS').values('name'), values('name'))",
        "g.V().union(out('KNOWS').count(), out('CREATED').count())",
        "g.V().union(values('name'), values('age'))",
        "g.V().union(out('KNOWS'), identity())",
        "g.V().union(out('KNOWS'), out('CREATED'), out())",
        // A branch that matches NOTHING still has to hold its row slot.
        "g.V().union(out('NOSUCH'), values('name'))",
        "g.V().union(out('KNOWS').has('age', gt(30)), values('name'))",
        // coalesce: first body that produces anything, per row
        "g.V().coalesce(out('KNOWS'), out('CREATED'))",
        "g.V().coalesce(values('age'), values('lang'))",
        "g.V().coalesce(out('NOSUCH'), out('KNOWS'), values('name'))",
        // optional: body rows, or the element itself
        "g.V().optional(out('KNOWS'))",
        "g.V().optional(out('KNOWS').values('name'))",
        "g.V().optional(outE('KNOWS'))",
        "g.V().optional(out('NOSUCH'))",
        // choose: both arities, and the no-else passthrough
        "g.V().choose(out('KNOWS'), values('name'), values('lang'))",
        "g.V().choose(has('age', gt(30)), values('name'))",
        "g.V().choose(hasLabel('PERSON'), out('KNOWS'), out('CREATED'))",
        // a tail after the branch, so the result feeds the rest of the column path
        "g.V().union(out('KNOWS'), out('CREATED')).count()",
        "g.V().union(out('KNOWS'), out('CREATED')).dedup().count()",
        "g.V().union(out('KNOWS'), out('CREATED')).limit(3)",
        "g.V().hasLabel('PERSON').union(out('KNOWS').fold(), values('name').fold())",
        // an edge frontier, and the endpoint gather inside a body
        "g.E().union(inV(), outV())",
        "g.E().optional(inV())",
        // `local()`, the one-body case
        "g.V().local(out('KNOWS'))",
        "g.V().local(values('name'))",
        "g.V().local(out('KNOWS').count())",
        // a NAVIGATING tail off a branch result, which needs the tightening back
        // into an element column
        "g.V().optional(out('KNOWS')).out('CREATED')",
        "g.V().union(out('KNOWS'), out('CREATED')).values('name')",
        "g.V().local(out('KNOWS')).outE('CREATED')",
        // a branch whose rows are a MIX of kinds stays boxed and must still agree
        "g.V().union(out('KNOWS'), outE('CREATED'))",
        "g.V().union(values('name'), out('KNOWS'))",
        // an empty frontier must not index past the end of the bounds
        "g.V().hasLabel('NOSUCH').union(out('KNOWS'), values('name'))",
        "g.V().hasLabel('NOSUCH').optional(out('KNOWS'))",
    ] {
        let lowered = super::parse::parse(q).expect("parses").run(&mut g);
        let streamed = super::parse::parse(&format!("{q}.fold().unfold()"))
            .expect("parses")
            .run(&mut g);

        assert_eq!(lowered, streamed, "{q}");
    }
}

/// What the branch steps cost lowered against the STREAM they replace, as an A/B
/// of the same engine on the same question — `fold().unfold()` forces the
/// streamed spelling, since `fold()` with a tail declines to lower.
///
/// Ignored (it builds an 80k-element fixture, and a 800k one when the sweep is
/// widened). The numbers it produced are recorded on the branch arms in
/// `gremlin/exec.rs`; re-run it after touching `fanout` or `col_of_branch`.
#[test]
#[ignore = "probe"]
fn branch_lowering_cost_probe() {
    // Both sizes were run; 20k is kept so the probe stays quick. The ratios were
    // within a point of each other either side of the cache-resident boundary.
    for n in [20_000usize] {
        let mut lines = String::new();
        for i in 0..n {
            let l = if i % 10 == 0 {
                r#"["V","W"]"#
            } else {
                r#"["V"]"#
            };
            lines.push_str(&format!(
                "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"k\":\"key{i:06}\"}}}}\n",
                i % 97
            ));
        }
        let mut e = 0;
        for i in 0..n {
            for d in 0..3usize {
                lines.push_str(&format!(
                    "{{\"type\":\"edge\",\"id\":\"e{e}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{d}}}}}\n",
                    (i * 31 + d * 7 + 1) % n
                ));
                e += 1;
            }
        }
        let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

        println!("\n=== {n} nodes / {} edges ({} elements)", n * 3, n * 4);
        println!(
            "{:<50} {:>10} {:>10} {:>7}",
            "traversal", "lowered", "streamed", "x"
        );
        for q in [
            "g.V().union(out('R'), values('n'))",
            "g.V().union(out('R').count(), values('n'))",
            "g.V().coalesce(out('R'), values('n'))",
            "g.V().optional(out('R'))",
            "g.V().choose(hasLabel('W'), out('R'), values('n'))",
            "g.V().local(out('R'))",
            "g.V().optional(out('R')).out('R').count()",
            "g.V().union(out('R'), out('R')).values('n')",
            "g.V().optional(out('R')).id()",
            "g.V().local(out('R')).groupCount().by('n')",
            "g.V().union(out('R'), out('R')).count()",
        ] {
            let streamed_q = format!("{q}.fold().unfold()");
            let (mut lo, mut st) = (f64::MAX, f64::MAX);
            for _ in 0..5 {
                let p = super::parse::parse(q).expect("parses");
                let t = std::time::Instant::now();
                let a = p.run(&mut g);
                lo = lo.min(t.elapsed().as_secs_f64() * 1000.0);
                let p2 = super::parse::parse(&streamed_q).expect("parses");
                let t2 = std::time::Instant::now();
                let b = p2.run(&mut g);
                st = st.min(t2.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(a, b, "{q}");
            }
            println!("{q:<50} {lo:>9.3}ms {st:>9.3}ms {:>6.1}x", st / lo);
        }
    }
}

/// Every arm of the column path, priced against the stream it replaces.
///
/// The question this answers is "which of these lowerings actually matter?" —
/// asked because the branch that added them grew the engine by thousands of
/// lines, and an arm that buys nothing is pure cost. Run 2026-08-07 over 20k
/// vertices / 60k edges, the answer was ALL of them: the worst is 1.9x
/// (`where`/`not`, which the stream already short-circuits per vertex) and the
/// rest run 2.1x to 353x. There is no dead lowering to remove here, and that is
/// worth knowing before someone deletes one to save lines.
///
/// Add a row when adding an arm. A row near 1.0x means the arm is not paying for
/// itself and should come out.
#[test]
#[ignore = "probe"]
fn arm_audit() {
    let mut lines = String::new();
    for i in 0..20_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"k\":\"key{i:06}\"}}}}\n",
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

    let cases: &[(&str, &str)] = &[
        ("[] bare frontier", "g.V().hasLabel('V')"),
        ("Fold", "g.V().fold()"),
        ("Count(Global)", "g.V().hasLabel('V').count()"),
        ("Count(Local)", "g.V().hasLabel('V').count(local)"),
        ("GroupCount identity", "g.V().groupCount()"),
        ("Barrier/Identity", "g.V().identity().count()"),
        ("As unread", "g.V().as('x').count()"),
        ("As + select", "g.V().as('x').select('x')"),
        ("Unfold", "g.V().values('n').unfold().count()"),
        ("Limit", "g.V().limit(5)"),
        ("Skip", "g.V().skip(5).count()"),
        ("Range", "g.V().range(2, 7)"),
        ("Tail", "g.V().tail(5)"),
        ("Id", "g.V().id()"),
        ("Label", "g.V().label()"),
        (
            "Group by+reduce",
            "g.V().group().by('n').by(values('n').sum())",
        ),
        ("Project", "g.V().project('a', 'b').by('n').by('k')"),
        ("OutV/InV", "g.E().inV()"),
        ("BothV", "g.E().bothV()"),
        ("ElementMap", "g.V().elementMap()"),
        ("Values", "g.V().values('n')"),
        ("GroupCount by key", "g.V().groupCount().by('n')"),
        ("Union", "g.V().union(out('R'), values('n'))"),
        ("Local", "g.V().local(out('R'))"),
        ("Coalesce", "g.V().coalesce(out('R'), values('n'))"),
        ("Optional", "g.V().optional(out('R'))"),
        (
            "Choose",
            "g.V().choose(hasLabel('W'), out('R'), values('n'))",
        ),
        ("Where", "g.V().where(__.out('R')).count()"),
        ("Not", "g.V().not(__.out('R')).count()"),
        ("Dedupe", "g.V().values('n').dedup()"),
        ("Order", "g.V().order().by('n').limit(10)"),
    ];

    println!(
        "{:<22} {:>10} {:>10} {:>7}",
        "arm", "lowered", "stream", "x"
    );
    for (name, q) in cases {
        let (mut lo, mut st) = (f64::MAX, f64::MAX);
        let mut same = true;
        for _ in 0..5 {
            super::exec::LOWERING_OFF.with(|c| c.set(false));
            let p = super::parse::parse(q).expect("parses");
            let t = std::time::Instant::now();
            let a = p.run(&mut g);
            lo = lo.min(t.elapsed().as_secs_f64() * 1000.0);

            super::exec::LOWERING_OFF.with(|c| c.set(true));
            let p2 = super::parse::parse(q).expect("parses");
            let t2 = std::time::Instant::now();
            let b = p2.run(&mut g);
            st = st.min(t2.elapsed().as_secs_f64() * 1000.0);
            same &= a == b;
        }
        super::exec::LOWERING_OFF.with(|c| c.set(false));
        println!(
            "{name:<22} {lo:>9.3}ms {st:>9.3}ms {:>6.1}x{}",
            st / lo,
            if same {
                ""
            } else {
                "   *** DIFFERENT RESULT ***"
            }
        );
    }
}

/// The three Gremlin routes, priced apart: the PATTERN planner
/// (`gremlin::pattern`, which compiles a linear prefix into GQL's pattern IR),
/// the linear COLUMN path, and the stream.
///
/// Asked because three routes is two more than the branch set out to have, so
/// each has to earn its place. `all` is everything on; `no pattern` leaves the
/// column arms but takes the planner away; `stream` is both off.
#[test]
#[ignore = "probe"]
fn route_audit() {
    let mut lines = String::new();
    for i in 0..50_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"k\":\"key{i:06}\"}}}}\n",
            i % 97
        ));
    }
    let mut e = 0;
    for i in 0..50_000usize {
        for d in 0..3usize {
            lines.push_str(&format!(
                "{{\"type\":\"edge\",\"id\":\"e{e}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{d}}}}}\n",
                (i * 31 + d * 7 + 1) % 50_000
            ));
            e += 1;
        }
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    println!(
        "{:<50} {:>9} {:>11} {:>9}",
        "traversal", "all", "no pattern", "stream"
    );
    for q in [
        "g.V().hasLabel('V').out('R').hasLabel('W').count()",
        "g.V().has('k', 'key000042').out('R').count()",
        "g.V().hasLabel('W').out('R').hasLabel('W').count()",
        "g.V().out('R').hasLabel('W').values('n')",
        "g.V().hasLabel('V').as('a').out('R').select('a').count()",
        "g.V().hasLabel('V').count()",
        "g.V().values('n').sum()",
    ] {
        let mut t = [f64::MAX; 3];
        let mut same = true;
        let mut first: Option<Vec<GVal>> = None;
        for _ in 0..5 {
            for (k, (lo, po)) in [(false, false), (false, true), (true, true)]
                .iter()
                .enumerate()
            {
                super::exec::LOWERING_OFF.with(|c| c.set(*lo));
                super::exec::PATTERN_OFF.with(|c| c.set(*po));
                let p = super::parse::parse(q).expect("parses");
                let start = std::time::Instant::now();
                let out = p.run(&mut g);
                t[k] = t[k].min(start.elapsed().as_secs_f64() * 1000.0);
                match &first {
                    None => first = Some(out),
                    Some(f) => same &= *f == out,
                }
            }
        }
        super::exec::LOWERING_OFF.with(|c| c.set(false));
        super::exec::PATTERN_OFF.with(|c| c.set(false));
        println!(
            "{q:<50} {:>8.3}ms {:>9.3}ms {:>8.3}ms{}",
            t[0],
            t[1],
            t[2],
            if same { "" } else { "  *** DIFFERENT ***" }
        );
    }
}

/// The PATTERN route reorders rows, and that is observable.
///
/// `gremlin::pattern` seeds at the selective end and walks the adjacency
/// backwards — which is where its 32-45x comes from — so its rows come out
/// grouped by the far end rather than in source order. Adding the post-hop filter
/// is what turns the route on, and the order changes with it:
///
///   g.V().out('CREATED').values('name')                      lop ripple lop lop
///   g.V().out('CREATED').hasLabel('SOFTWARE').values('name')  lop lop lop ripple
///
/// The TS engine, which has no planner, returns source order for both — so the
/// two engines disagree on the sequence of an ordinary traversal, and under
/// `fold()` that is a single row whose LIST differs. Confirmed through the real
/// FFI, not just in-process.
///
/// This is PERMITTED: row order without an explicit `order()` is unspecified in
/// both engines, like SQL without ORDER BY, and that is settled policy — the
/// multiset is identical. It is pinned here because it looks exactly like a bug,
/// because the trigger (a filter AFTER the hop) is not obvious, and because the
/// engine already treats reordering as observable elsewhere: the `g.E()` desugar
/// is admitted "just where the sequence cannot be observed". By that standard
/// far-end seeding deserves the same guard, which would keep the win for
/// order-insensitive shapes (`count()`, aggregates, `dedup()`) and give it up for
/// the rest. That is a perf-vs-agreement decision, not a fix to make quietly.
#[test]
fn the_pattern_route_reorders_rows() {
    let mut g = modern();
    let q = "g.V().out('CREATED').hasLabel('SOFTWARE').values('name')";

    super::exec::PATTERN_OFF.with(|c| c.set(false));
    let planned = super::parse::parse(q).expect("parses").run(&mut g);
    super::exec::PATTERN_OFF.with(|c| c.set(true));
    let walked = super::parse::parse(q).expect("parses").run(&mut g);
    super::exec::PATTERN_OFF.with(|c| c.set(false));

    assert_ne!(planned, walked, "the planner is expected to reorder");

    let key = |v: &Vec<GVal>| {
        let mut k: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
        k.sort();
        k
    };

    assert_eq!(key(&planned), key(&walked), "but only the sequence differs");
}

/// The two languages compute the SAME VALUES for the same question.
///
/// `cross_language_cost_probe` already asserts that each pair below is one
/// question written twice; it only ever PRICED them. This runs both sides and
/// compares the answers, which is the load-bearing claim behind sharing an IR at
/// all — if the algebras differed, no amount of shared machinery would help.
///
/// Both sides are flattened to a bag of scalars first, because the two languages
/// PACKAGE results differently and that difference is not a semantic one: GQL
/// returns 194 rows of `(k, c)` where Gremlin returns one `Map` holding the same
/// 194 entries, and an element renders as `{id,labels,properties}` on one side
/// and is a handle on the other. Flattened, 45 of 49 agree outright; the last four
/// are the normalizer's own asymmetry, not the engines' — it flattens a `GVal`
/// recursively (so a `project()` map contributes its KEYS as well as its values,
/// and `elementMap()` contributes every field) while rendering a GQL
/// `graph::Value` shallowly. Every one of the four is an element-rendering or
/// map-flattening artifact; none is a differing computation.
///
/// That is the argument for translating rather than re-implementing: the reason
/// `gremlin::pattern` can compile a traversal prefix into GQL's `CPath` and
/// inherit its planner (4-45x, see `route_audit`) is that the two languages mean
/// the same thing. What they do NOT share is the result boundary, and the
/// remaining per-language contracts — null ordering, NaN, cross-type comparison,
/// unspecified row order — which are parameters of an executor, not reasons for
/// two of them.
#[test]
#[ignore = "probe"]
fn the_two_languages_compute_the_same_values() {
    let mut lines = String::new();
    for i in 0..2_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{},\"k\":\"key{i:06}\"}}}}\n",
            i % 97
        ));
    }
    let mut e = 0;
    for i in 0..2_000usize {
        for d in 0..3usize {
            lines.push_str(&format!(
                "{{\"type\":\"edge\",\"id\":\"e{e}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{d}}}}}\n",
                (i * 31 + d * 7 + 1) % 2_000
            ));
            e += 1;
        }
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    let pairs = CROSS_LANGUAGE_PAIRS;

    // Normalize both sides to a sorted multiset of scalar renderings, so only the
    // VALUES are compared — not column names, row shape, or ordering.
    // Flatten a result to its SCALARS, so only the computed values are compared:
    // a Gremlin `Map` (one row holding N entries) and a GQL two-column row set
    // (N rows holding 2 cells) become the same bag of scalars. An element becomes
    // its id, since one side renders `{id,labels,properties}` and the other a
    // stringified id.
    fn flat_g(v: &GVal, out: &mut Vec<String>) {
        match v {
            GVal::Map(m) => {
                for (k, val) in m.clone().into_pairs() {
                    flat_g(&k, out);
                    flat_g(&val, out);
                }
            }
            GVal::List(items) => {
                for it in items.iter() {
                    flat_g(it, out);
                }
            }
            GVal::Node(i) => out.push(format!("elem{i}")),
            GVal::Edge(i) => out.push(format!("elem{i}")),
            GVal::Num(n) => out.push(format!("{n}")),
            GVal::Str(x) => out.push(x.to_string()),
            other => out.push(format!("{other:?}")),
        }
    }
    let norm = |mut v: Vec<String>| {
        v.sort();
        v
    };

    let (mut agree, mut differ, mut skipped) = (0, 0, 0);
    for (name, gql, grem) in pairs {
        let gq = crate::gql::parse(gql)
            .ok()
            .and_then(|p| p.execute(&mut g, &crate::gql::eval::Params::new()).ok());
        let Some(rs) = gq else {
            skipped += 1;
            continue;
        };
        // The GQL side: cells are `Value`; render them the same way, and drop the
        // element-vs-id distinction by keying elements on their id.
        let gvals: Vec<String> = rs
            .data
            .iter()
            .map(|c| match c {
                crate::graph::Value::Num(n) => format!("{n}"),
                crate::graph::Value::Str(x) => x.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        let gr = super::parse::parse(grem)
            .unwrap_or_else(|e| panic!("`{grem}` parses: {e}"))
            .run(&mut g);
        let mut rvals: Vec<String> = Vec::new();
        for v in &gr {
            flat_g(v, &mut rvals);
        }

        let (a, b) = (norm(gvals), norm(rvals));
        if a == b {
            agree += 1;
        } else {
            differ += 1;
            println!("DIFFER  {name}");
            println!(
                "   gql     [{}] {:?}",
                a.len(),
                a.iter().take(3).collect::<Vec<_>>()
            );
            println!(
                "   gremlin [{}] {:?}",
                b.len(),
                b.iter().take(3).collect::<Vec<_>>()
            );
        }
    }
    println!("\nAGREE {agree}   DIFFER {differ}   SKIPPED {skipped}");
}

/// `PROPERTY_EXISTS` is a bitset read, and used to run the scalar VM per row.
///
/// Measured over 50k vertices, two thirds carrying the key, with and without the
/// `eval_vec` arm:
///
///   WHERE property_exists(u, k) RETURN count(*)   0.849ms -> 0.109   7.8x
///   RETURN property_exists(u, k)                  1.135   -> 0.413   2.7x
///   WHERE property_exists(u, k) RETURN u.n        1.112   -> 0.374   3.0x
///
/// Found from the GREMLIN side: `values(k)` drops a row whose key is absent, which
/// as a projection is `RETURN k, PROPERTY_EXISTS(elem, k)`, and that spelling cost
/// 11x against a direct column read. The missing arm was the whole of it — so a
/// Gremlin question turned up a GQL optimization, which is what sharing a column
/// layer is for.
#[test]
#[ignore = "probe"]
fn property_exists_is_a_column_read() {
    let mut lines = String::new();
    for i in 0..50_000usize {
        // Two thirds carry `k`, so presence is a real filter rather than a constant.
        let props = if i % 3 == 0 {
            format!("{{\"n\":{}}}", i % 97)
        } else {
            format!("{{\"n\":{},\"k\":\"key{i:06}\"}}", i % 97)
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{props}}}\n"
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    for q in [
        "MATCH (u:V) WHERE property_exists(u, k) RETURN count(*) AS c",
        "MATCH (u:V) RETURN property_exists(u, k) AS p",
        "MATCH (u:V) WHERE property_exists(u, k) RETURN u.n AS n",
    ] {
        let mut best = f64::MAX;
        let mut rows = 0;
        for _ in 0..9 {
            let p = crate::gql::parse(q).expect("parses");
            let t = std::time::Instant::now();
            let rs = p
                .execute(&mut g, &crate::gql::eval::Params::new())
                .expect("runs");
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            rows = rs.nrows;
        }
        println!("PE {best:>8.3}ms  {rows:>6} rows  {q}");
    }
}

/// A `$param` is constant for the whole query, and used to be re-read through the
/// scalar VM once per row.
///
/// Over 50k vertices, with and without the `eval_vec` arm:
///
///   WHERE u.n > $x RETURN count(*)   0.961ms -> 0.120   8.0x
///   WHERE u.n > $x RETURN u.n        1.124   -> 0.292   3.8x
///   RETURN u.n + $x                  2.300   -> 0.451   5.1x
///   WHERE u.s = $s RETURN count(*)   0.054   -> 0.054   unchanged
///
/// The last row is the control: an equality on an INDEXED key is seeded, so it
/// never reaches the column path and the arm cannot help it. Without that row the
/// other three could be read as "params got faster" rather than "the column path
/// stopped re-evaluating a constant".
#[test]
#[ignore = "probe"]
fn a_param_is_broadcast_not_re_evaluated() {
    let mut lines = String::new();
    for i in 0..50_000usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{},\"s\":\"v{}\"}}}}\n",
            i % 97,
            i % 7
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    let mut params = crate::gql::eval::Params::new();
    params.insert("x".to_string(), crate::gql::eval::Val::Num(50.0));
    params.insert("s".to_string(), crate::gql::eval::Val::Str("v3".into()));

    for q in [
        "MATCH (u:V) WHERE u.n > $x RETURN count(*) AS c",
        "MATCH (u:V) WHERE u.n > $x RETURN u.n AS n",
        "MATCH (u:V) RETURN u.n + $x AS n",
        "MATCH (u:V) WHERE u.s = $s RETURN count(*) AS c",
    ] {
        let mut best = f64::MAX;
        let mut rows = 0;
        for _ in 0..9 {
            let p = crate::gql::parse(q).expect("parses");
            let t = std::time::Instant::now();
            let rs = p.execute(&mut g, &params).expect("runs");
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            rows = rs.nrows;
        }
        println!("PARAM {best:>8.3}ms {rows:>6} rows  {q}");
    }
}

/// `IS LABELED` walks a label set per element — irreducible — but used to carry VM
/// dispatch and `Val` boxing around it. Over 50k vertices, a third multi-labeled:
///
///   WHERE u IS LABELED W RETURN count(*)      1.033ms -> 0.184   5.6x
///   RETURN u IS LABELED W                     1.336   -> 0.501   2.7x
///   WHERE u IS NOT LABELED W RETURN u.n       1.313   -> 0.446   2.9x
#[test]
#[ignore = "probe"]
fn is_labeled_is_a_column_walk() {
    let mut lines = String::new();
    for i in 0..50_000usize {
        let l = if i % 3 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{}}}}}\n",
            i % 97
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    for q in [
        "MATCH (u:V) WHERE u IS LABELED W RETURN count(*) AS c",
        "MATCH (u:V) RETURN u IS LABELED W AS w",
        "MATCH (u:V) WHERE u IS NOT LABELED W RETURN u.n AS n",
    ] {
        let mut best = f64::MAX;
        let mut rows = 0;
        for _ in 0..9 {
            let p = crate::gql::parse(q).expect("parses");
            let t = std::time::Instant::now();
            let rs = p
                .execute(&mut g, &crate::gql::eval::Params::new())
                .expect("runs");
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            rows = rs.nrows;
        }
        println!("LBL {best:>8.3}ms {rows:>6} rows  {q}");
    }
}

/// `select` over pattern tags, with a `by('k')` modulator and with `Pop.all`,
/// must return exactly what the stream returns.
///
/// These two shapes used to decline to the stream and were the largest gaps left
/// in `cross_language_cost_probe` — 79x and 83x against the identical question in
/// GQL. Now they zip columns, so they need an agreement test of their own:
/// `PATTERN_OFF` runs the same traversal without the pattern route, which is the
/// streamed spelling.
#[test]
fn lowered_select_with_modulators_agrees_with_the_stream() {
    let mut g = modern();

    for q in [
        // by('k') — one label and several, and a tail after it
        "g.V().as('x').out('KNOWS').select('x').by('name')",
        "g.V().as('x').out('KNOWS').select('x').by('age')",
        "g.V().as('x').out('KNOWS').as('y').select('x', 'y').by('name').by('name')",
        "g.V().as('x').out('KNOWS').select('x').by('name').count()",
        "g.V().as('x').out('KNOWS').select('x').by('name').dedup()",
        // an absent key on some rows
        "g.V().as('x').out('CREATED').select('x').by('lang')",
        // Pop.all — a list per label
        "g.V().as('x').out('KNOWS').select(all, 'x')",
        "g.V().as('x').out('KNOWS').as('y').select(all, 'x', 'y')",
        "g.V().as('x').out('KNOWS').select(all, 'x').count()",
        // the bare form still works
        "g.V().as('x').out('KNOWS').select('x')",
        "g.V().as('x').out('KNOWS').as('y').select('x', 'y')",
        // a by() this must NOT take: it carries an order
        "g.V().as('x').out('KNOWS').select('x').by('name', desc)",
    ] {
        super::exec::PATTERN_OFF.with(|c| c.set(false));
        let lowered = super::parse::parse(q).expect("parses").run(&mut g);
        super::exec::MIGRATE_OFF.with(|c| c.set(true));
        let streamed = super::parse::parse(q).expect("parses").run(&mut g);
        super::exec::MIGRATE_OFF.with(|c| c.set(false));

        assert_eq!(lowered, streamed, "{q}");
    }
}

/// What the expressions `eval_vec` does NOT vectorize actually cost, over 50k
/// vertices. The list came from enumerating the 32 `CExpr` variants against the
/// arms; this is what says which are worth an arm.
///
/// Measured 2026-08-07, against ~0.12ms for a vectorized count and ~0.165ms for a
/// vectorized projection:
///
///   In                 0.70ms      PropField         1.40  (was 1.93)
///   IsTyped            1.16        PropField filter  5.13
///   Case               2.24        Concat            3.99
///   List               4.40
///
/// `PropField` got an arm and moved 1.4x; the rest of ITS cost is `field_at`
/// navigating the record, which no dispatch change reaches — and the filter form
/// barely moved at all, so the WHERE path for a dotted key is not going through
/// here. `Concat`, `List` and `Case` allocate per row by construction (a string, a
/// list, a branch), so an arm would remove dispatch and nothing else. Recorded so
/// the next person prices them before writing one.
#[test]
#[ignore = "probe"]
fn what_the_unvectorized_expressions_cost() {
    let mut lines = String::new();
    for i in 0..50_000usize {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{},\"s\":\"v{}\",\"meta\":{{\"city\":\"c{}\"}}}}}}\n",
            i % 97,
            i % 7,
            i % 11
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    for (name, q) in [
        (
            "In",
            "MATCH (u:V) WHERE u.n IN [1, 2, 3, 4, 5] RETURN count(*) AS c",
        ),
        ("PropField", "MATCH (u:V) RETURN u.meta.city AS c"),
        (
            "PropField filter",
            "MATCH (u:V) WHERE u.meta.city = 'c3' RETURN count(*) AS c",
        ),
        ("Concat", "MATCH (u:V) RETURN u.s || 'x' AS c"),
        (
            "IsTyped",
            "MATCH (u:V) WHERE u.n IS TYPED INTEGER RETURN count(*) AS c",
        ),
        (
            "Case",
            "MATCH (u:V) RETURN CASE WHEN u.n > 50 THEN 1 ELSE 0 END AS c",
        ),
        ("List", "MATCH (u:V) RETURN [u.n, u.n] AS c"),
    ] {
        let mut best = f64::MAX;
        let mut rows = 0;
        let mut ok = true;
        for _ in 0..7 {
            let Ok(p) = crate::gql::parse(q) else {
                ok = false;
                break;
            };
            let t = std::time::Instant::now();
            match p.execute(&mut g, &crate::gql::eval::Params::new()) {
                Ok(rs) => {
                    best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                    rows = rs.nrows;
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            println!("EXPR {name:<18} {best:>8.3}ms {rows:>6} rows");
        } else {
            println!("EXPR {name:<18}    (unsupported syntax)");
        }
    }
}

/// The MIGRATION route — a Gremlin tail compiled to a GQL projection — must
/// return exactly what the route it replaces returns.
///
/// Compared against `MIGRATE_OFF` (the column terminals, still on the pattern
/// plan) and NOT against the stream: the pattern route enumerates in a different
/// order by design (`the_pattern_route_reorders_rows`), so comparing against the
/// stream drowns a real ordering bug in permitted ones. That is not theoretical —
/// it is how `order().by(k)` first looked correct: a sort key referencing the
/// raw input slot instead of the `order_overlay` position it publishes through
/// sorted wrong (`n = 0` after `n = 10`), and only comparing against the route
/// it replaces (not the stream) caught it.
///
/// `migrated` asserts the route actually FIRED. Without it every row passes on an
/// arm that silently declined, which has happened twice on this branch.
#[test]
fn the_migration_route_agrees_with_the_route_it_replaces() {
    let mut lines = String::new();
    for i in 0..2_000usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        // `m` is ABSENT on about a third of rows (`i % 3 == 0`), which two arms
        // need for different reasons: `project()` must hold a NULL for it (not
        // drop the row, unlike `values(k)`), and the paging/`fold()` arms must
        // drop those rows BEFORE windowing or collecting. `n`, present on every
        // row, exercises neither.
        let m = if i % 3 == 0 {
            String::new()
        } else {
            format!(",\"m\":{}", i % 97)
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{}{m},\"k\":\"key{i:06}\"}}}}\n",
            i % 97
        ));
    }
    let mut e = 0;
    for i in 0..2_000usize {
        for d in 0..3usize {
            lines.push_str(&format!(
                "{{\"type\":\"edge\",\"id\":\"e{e}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
                (i * 31 + d * 7 + 1) % 2_000
            ));
            e += 1;
        }
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");
    for q in [
        "g.V().out('R').hasLabel('W').values('n')",
        "g.V().out('R').hasLabel('W').values('n').sum()",
        "g.V().out('R').hasLabel('W').values('n').min()",
        "g.V().out('R').hasLabel('W').values('n').max()",
        "g.V().out('R').hasLabel('W').values('n').mean()",
        "g.V().out('R').hasLabel('W').values('n').dedup()",
        "g.V().out('R').hasLabel('W').values('k')",
        // `id()` — GQL's `element_id(n)` calls the SAME `Value::element_id` this
        // engine's `elem_id` already does (`scalar_fns.rs`'s own comment:
        // "Shared with Gremlin's `id()`"), for either element kind, so both a
        // vertex and an edge frontier are worth checking.
        "g.V().out('R').hasLabel('W').id()",
        "g.V().outE('R').id()",
        // `label()` migrates ONLY off an EDGE frontier — GQL's `type(e)` mirrors
        // `elem_label`'s single-value edge rendering exactly (both docs say so
        // in matching words), but a VERTEX has no equivalent: `labels(n)`
        // returns every label SORTED as a `List`, while `elem_label` reads the
        // first-INSERTED one off `vertex_labels()`. `to_gql::tail`'s `[Step::
        // Label]` arm declines whenever `is_edge` is false for exactly that
        // reason, so only the edge form belongs in this loop — a vertex
        // `label()` still takes the `elem_terminal` route this migrates away
        // from, unchanged, and is covered elsewhere (`step_tests_2.rs`,
        // `tests.rs`).
        "g.V().outE('R').label()",
        // `id_label_prefix` — `id()`/`label()` followed by MORE steps, split
        // the same way `values_prefix` splits `values(k)`: project the id/
        // label column, hand the rest back to the column terminals.
        "g.V().out('R').hasLabel('W').id().count()",
        "g.V().out('R').hasLabel('W').id().dedup()",
        "g.V().out('R').hasLabel('W').id().limit(5)",
        "g.V().out('R').hasLabel('W').id().fold()",
        "g.V().outE('R').label().count()",
        "g.V().outE('R').label().dedup()",
        "g.V().outE('R').label().groupCount()",
        // `values(k).count()` — the count of rows whose key is NOT ABSENT, not
        // `count(*)` over the frontier. `n` is dense (no absent rows, a sanity
        // case with nothing to drop); `m` is absent on ~1/3 of rows, which is
        // the case that actually discriminates a row-drop-then-count from a
        // bare frontier count.
        "g.V().out('R').hasLabel('W').values('n').count()",
        // `values(k).is(P)` — the narrowing arm, with and without a window, on a
        // dense property and on one absent from a third of rows (where the
        // absent-key drop has to happen BEFORE the predicate, and the validity
        // mask has to be cleared afterwards or the arm declines a column with
        // nothing missing).
        "g.V().out('R').hasLabel('W').values('n').is(gt(40))",
        "g.V().out('R').hasLabel('W').values('n').is(lt(10))",
        "g.V().out('R').hasLabel('W').values('n').is(gt(40)).limit(5)",
        "g.V().out('R').hasLabel('W').values('m').is(gt(40))",
        "g.V().out('R').hasLabel('W').values('m').is(gt(40)).range(2, 6)",
        "g.V().out('R').hasLabel('W').values('m').count()",
        "g.V().out('R').hasLabel('W').groupCount().by('n')",
        "g.V().out('R').hasLabel('W').groupCount().by('k')",
        // `group().by(k).by(values(v).<reduce>())` — one row per group, a
        // REDUCED VALUE in place of `groupCount`'s tally. `k` (unique per node)
        // first: every group is a singleton, exercising the basic column-fold
        // shape with nothing to actually reduce...
        "g.V().out('R').hasLabel('W').group().by('k').by(__.values('n').sum())",
        "g.V().out('R').hasLabel('W').group().by('k').by(__.values('n').min())",
        "g.V().out('R').hasLabel('W').group().by('k').by(__.values('n').max())",
        "g.V().out('R').hasLabel('W').group().by('k').by(__.values('n').mean())",
        // ...then `n` (`i % 97`) as the key: real multi-member groups, so ties
        // in the key, FIRST-SEEN group order, and the reducer's fold order
        // (float addition is not associative) all have to agree, not just the
        // final numbers.
        "g.V().out('R').hasLabel('W').group().by('n').by(__.values('n').sum())",
        "g.V().out('R').hasLabel('W').group().by('n').by(__.values('n').mean())",
        // The VALUE side (`m`) absent on ~1/3 of rows: the reducer skips a
        // `null` ARGUMENT, not a row — the group still has every member.
        "g.V().out('R').hasLabel('W').group().by('n').by(__.values('m').sum())",
        // The KEY side (`m`) absent on ~1/3 of rows: those rows group under a
        // single NULL key rather than being dropped — same rule as
        // `groupCount().by('m')` would use, exercised here for the reducer arm.
        "g.V().out('R').hasLabel('W').group().by('m').by(__.values('n').sum())",
        // `n` (present on every row, i % 97 over 2000 nodes): many ties — the
        // frontier/scan-order tiebreak has to agree between the two routes,
        // not just the keys.
        "g.V().out('R').hasLabel('W').order().by('n')",
        "g.V().out('R').hasLabel('W').order().by('n').limit(5)",
        "g.V().out('R').hasLabel('W').order().by('n', desc)",
        "g.V().out('R').hasLabel('W').order().by('n', desc).limit(5)",
        // `k` is a unique string per node: no ties, both directions.
        "g.V().out('R').hasLabel('W').order().by('k')",
        "g.V().out('R').hasLabel('W').order().by('k', desc)",
        // `m` is absent on ~1/3 of the frontier (`i % 3 == 0`) — nulls-first
        // ascending, nulls-last descending, with and without a LIMIT that
        // lands inside the null run.
        "g.V().out('R').hasLabel('W').order().by('m')",
        "g.V().out('R').hasLabel('W').order().by('m', desc)",
        "g.V().out('R').hasLabel('W').order().by('m').limit(5)",
        "g.V().out('R').hasLabel('W').order().by('m', desc).limit(5)",
        // `project()` — two keys, a numeric and a string property.
        "g.V().out('R').hasLabel('W').project('n','k').by('n').by('k')",
        // three keys, the third (`m`) absent on some rows — a null in the
        // map, not a dropped row (unlike `values(k)`).
        "g.V().out('R').hasLabel('W').project('n','k','m').by('n').by('k').by('m')",
        // an explicit identity `by()` alongside a keyed one.
        "g.V().out('R').hasLabel('W').project('v','n').by().by('n')",
        // a MISSING `by()` (fewer bys than keys) defaults to identity too.
        "g.V().out('R').hasLabel('W').project('v','n').by('n')",
        // Paging and `fold()`, first over `n` (present on every row) so the
        // window/list math is checked with no absent-row drop in play...
        "g.V().out('R').hasLabel('W').values('n').limit(5)",
        "g.V().out('R').hasLabel('W').values('n').limit(0)",
        "g.V().out('R').hasLabel('W').values('n').skip(5)",
        "g.V().out('R').hasLabel('W').values('n').range(3, 8)",
        "g.V().out('R').hasLabel('W').values('n').range(8, 3)",
        "g.V().out('R').hasLabel('W').values('n').range(0, 1000000)",
        "g.V().out('R').hasLabel('W').values('n').fold()",
        // ...then over `m` (absent on ~1/3 of rows), which is the one that catches
        // a window taken over the SCANNED rows instead of the ones `values(m)`
        // actually kept — `limit(0)`, a `range` with `hi <= lo`, and one past the
        // end are the edge cases most likely to hide that off-by-a-window bug.
        "g.V().out('R').hasLabel('W').values('m').limit(5)",
        "g.V().out('R').hasLabel('W').values('m').limit(0)",
        "g.V().out('R').hasLabel('W').values('m').skip(5)",
        "g.V().out('R').hasLabel('W').values('m').range(3, 8)",
        "g.V().out('R').hasLabel('W').values('m').range(8, 3)",
        "g.V().out('R').hasLabel('W').values('m').range(0, 1000000)",
        "g.V().out('R').hasLabel('W').values('m').fold()",
    ] {
        super::exec::MIGRATED.with(|c| c.set(0));
        super::exec::PATTERN_OFF.with(|c| c.set(false));
        let via_gql = super::parse::parse(q).expect("parses").run(&mut g);
        let took = super::exec::MIGRATED.with(std::cell::Cell::get);

        super::exec::MIGRATE_OFF.with(|c| c.set(true));
        let streamed = super::parse::parse(q).expect("parses").run(&mut g);
        super::exec::MIGRATE_OFF.with(|c| c.set(false));

        assert_eq!(via_gql, streamed, "{q}");
        // Every case here now goes through the new route, `order()` included.
        assert_eq!(took, 1, "{q}: migration route taken {took} times");

        // `limit(0)` and a `range` with `hi <= lo` are the only cases meant to page
        // to nothing — everything else should return rows, so an empty result
        // there is the bug this loop exists to catch.
        let expect_empty = q.contains(".limit(0)") || q.contains(".range(8, 3)");
        assert_eq!(
            via_gql.is_empty(),
            expect_empty,
            "{q}: expected {} rows",
            if expect_empty { "no" } else { "some" }
        );
    }
}

/// `exec.rs::elem_terminal`'s `[Step::Group(bys)]` arm (the one `to_gql`'s new
/// `group().by(k).by(values(v).<reduce>())` arm was meant to replace) is still
/// REACHABLE, empirically, not just by reasoning — so it must stay.
///
/// A non-DURATION temporal `v` is the witness: GQL's grouped `sum()` FAULTS on
/// it (`FAULT_TEMPORAL_AGG`, `matcher.rs`), so `project_ids` sees
/// `ctx.fault != FAULT_NONE` and returns `None` — the migration declines at
/// RUN time even though `to_gql::tail` (a COMPILE-time check) accepted the
/// shape. `elem_terminal`'s own arm uses the stream's `reduce_nums`/
/// `strict_num`, which silently SKIPS a non-numeric value instead of
/// faulting, so it still has to answer this query and disagree with GQL's
/// aggregator on how the divergence is handled (fault vs. skip) — the exact
/// case that keeps this arm alive.
#[test]
fn old_group_reduce_arm_is_still_reachable_on_a_temporal_fault() {
    let doc = "{\"type\":\"node\",\"id\":\"1\",\"labels\":[\"V\"],\"properties\":{\"k\":\"a\",\"v\":{\"@date\":\"2020-01-01\"}}}\n\
               {\"type\":\"node\",\"id\":\"2\",\"labels\":[\"V\"],\"properties\":{\"k\":\"a\",\"v\":{\"@date\":\"2020-01-02\"}}}\n";
    let mut g = crate::ndjson::decode(doc).expect("fixture decodes");

    super::exec::MIGRATED.with(|c| c.set(0));
    let out = super::parse::parse("g.V().group().by('k').by(__.values('v').sum())")
        .expect("parses")
        .run(&mut g);
    let took = super::exec::MIGRATED.with(std::cell::Cell::get);

    // The migration route did NOT complete...
    assert_eq!(
        took, 0,
        "expected the migration to decline on a temporal sum"
    );
    // ...and the OLD arm still answered correctly: a non-numeric value is
    // skipped, not summed, leaving no number in the group.
    assert_eq!(
        out,
        vec![GVal::map(vec![(
            GVal::Str(std::sync::Arc::from("a")),
            GVal::Null
        )])]
    );
}

/// A peelable step in front (`dedup()`) still reaches the NEW `to_gql` arm —
/// `col_terminal_tagged` retries `to_gql::tail` on every recursive call,
/// including after a peel, so the migration is not limited to the bare
/// `group()` shape.
#[test]
fn new_group_reduce_arm_migrates_behind_a_peelable_step() {
    let doc = "{\"type\":\"node\",\"id\":\"1\",\"labels\":[\"V\"],\"properties\":{\"k\":\"a\",\"n\":1}}\n\
               {\"type\":\"node\",\"id\":\"2\",\"labels\":[\"V\"],\"properties\":{\"k\":\"a\",\"n\":2}}\n";
    let mut g = crate::ndjson::decode(doc).expect("fixture decodes");

    super::exec::MIGRATED.with(|c| c.set(0));
    let out = super::parse::parse("g.V().dedup().group().by('k').by(__.values('n').sum())")
        .expect("parses")
        .run(&mut g);
    let took = super::exec::MIGRATED.with(std::cell::Cell::get);

    assert_eq!(took, 1, "expected the migration to fire behind dedup()");
    assert_eq!(
        out,
        vec![GVal::map(vec![(
            GVal::Str(std::sync::Arc::from("a")),
            GVal::Num(3.0)
        )])]
    );
}

/// `values(k).count()` on a `Column::Mixed` property: a real TYPE conflict (a
/// number and a string) forces the promotion, and `Mixed`'s `Vec<Option<Value>>`
/// really does distinguish "never set" (`None`) from "stored null"
/// (`Some(Value::Null)`) — unlike a typed column, where a null write just
/// clears the same presence bit as absence (`Column::Num`'s "absent = NaN,
/// also flagged in `present`" — one bit, not two states).
///
/// `GQL`'s `count(prop)` cannot make that distinction (both box to
/// `Val::Null`), so it would undercount by one against Gremlin's "drop only
/// the truly absent row" if it ever ran here — proving out why the arm's
/// `absent_key`/`homogeneous_or_absent` check in `shape_projection` has to
/// decline for `Mixed`, not just for the row-drop shapes.
#[test]
fn values_count_declines_on_a_mixed_column_with_a_stored_null() {
    let doc = "{\"type\":\"node\",\"id\":\"1\",\"labels\":[\"V\"],\"properties\":{\"v\":1}}\n\
               {\"type\":\"node\",\"id\":\"2\",\"labels\":[\"V\"],\"properties\":{\"v\":\"x\"}}\n\
               {\"type\":\"node\",\"id\":\"3\",\"labels\":[\"V\"],\"properties\":{\"v\":null}}\n\
               {\"type\":\"node\",\"id\":\"4\",\"labels\":[\"V\"],\"properties\":{}}\n";
    let mut g = crate::ndjson::decode(doc).expect("fixture decodes");

    let via_gql = super::parse::parse("g.V().values('v').count()")
        .expect("parses")
        .run(&mut g);

    super::exec::MIGRATE_OFF.with(|c| c.set(true));
    let streamed = super::parse::parse("g.V().values('v').count()")
        .expect("parses")
        .run(&mut g);
    super::exec::MIGRATE_OFF.with(|c| c.set(false));

    // 3 present rows (number, string, stored null); the 4th is truly absent
    // and dropped.
    assert_eq!(via_gql, vec![GVal::Num(3.0)]);
    assert_eq!(via_gql, streamed);
}

/// Whether `exec.rs::elem_terminal`'s `[Step::Values(keys), tail @ ..]` arm —
/// `column_terminal`, its ONE call site — is still reachable, checked
/// EMPIRICALLY rather than by reasoning about the peel machinery.
///
/// `to_gql::tail` compiles a bare `values(k)` and several of its modulators
/// (`dedup`, paging, `fold`, a reducer), and `col_terminal_tagged` retries
/// that translation at EVERY peel (`dedup()`, `barrier()`, the pagers,
/// an unread `as()`) — not just at the entry — so a peelable step in front of
/// `values(k)` does not by itself keep this arm alive. What DOES: a tag read
/// (`select` after `as`), which the whole-traversal migration declines up
/// front (`run_collect`'s `c.tags.is_empty()` guard), and a frontier that is
/// no longer `Col::Elems` by the time `values(k)` runs (`fold().unfold()`
/// reboxes into `Col::Gen`, which the retry's `if let Col::Elems { .. } = &col`
/// does not match).
///
/// `COLUMN_TERMINAL_HIT` is a raw entry counter on `column_terminal` itself —
/// the same "did the OLD route actually run" question `MIGRATED` answers for
/// the new one, kept separate so a case that fires NEITHER (declining to the
/// stream entirely) is visible instead of silently passing.
#[test]
fn values_arm_reachability_probe() {
    let mut lines = String::new();
    for i in 0..500usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{}}}}}\n",
            i % 37
        ));
    }
    for i in 0..500usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{i}}}}}\n",
            (i * 31 + 1) % 500
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    let shapes: &[(&str, &str)] = &[
        (
            "baseline (no peel)",
            "g.V().out('R').hasLabel('W').values('n')",
        ),
        (
            "dedup() in front",
            "g.V().out('R').hasLabel('W').dedup().values('n')",
        ),
        (
            "barrier() in front",
            "g.V().out('R').hasLabel('W').barrier().values('n')",
        ),
        (
            "limit() in front",
            "g.V().out('R').hasLabel('W').limit(2).values('n')",
        ),
        (
            "tagged path (select)",
            "g.V().as('x').out('R').select('x').values('n')",
        ),
        (
            "post-fold (fold().unfold())",
            "g.V().out('R').hasLabel('W').fold().unfold().values('n')",
        ),
        ("edge frontier", "g.E().hasLabel('R').values('w')"),
        // Not in the brief, but the case `to_gql::tail` genuinely cannot
        // express: an `is(P)` filter chained straight off `values(k)`, which
        // `column_terminal` handles ITSELF (splitting the predicate off
        // before delegating the bare read) rather than through `to_gql::tail`.
        (
            "values().is() filter",
            "g.V().out('R').hasLabel('W').values('n').is(gt(20))",
        ),
    ];

    println!();
    for (label, q) in shapes {
        super::exec::MIGRATED.with(|c| c.set(0));
        super::exec::COLUMN_TERMINAL_HIT.with(|c| c.set(0));

        let out = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}` parses: {e}"))
            .run(&mut g);

        let migrated = super::exec::MIGRATED.with(std::cell::Cell::get);
        let old_arm = super::exec::COLUMN_TERMINAL_HIT.with(std::cell::Cell::get);
        let route = if migrated > 0 {
            "MIGRATED"
        } else if old_arm > 0 {
            "OLD_ARM"
        } else {
            "NEITHER (third route / stream)"
        };

        println!(
            "PROBE migrated={migrated} column_terminal={old_arm} rows={} route={route} [{label}] {q}",
            out.len()
        );
    }
}

/// Whether `exec.rs::elem_terminal`'s bare `[Step::Id]` / `[Step::Label]` arms
/// are still reachable now that `id_label_prefix` widens the split
/// `values_prefix` already does for `values(k)` — checked EMPIRICALLY, the same
/// way as `values_arm_reachability_probe`.
///
/// `ID_LABEL_ARM_HIT` is a raw entry counter on those two arms themselves.
#[test]
fn id_label_arm_reachability_probe() {
    let mut lines = String::new();
    for i in 0..500usize {
        let l = if i % 10 == 0 {
            r#"["V","W"]"#
        } else {
            r#"["V"]"#
        };
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":{l},\"properties\":{{\"n\":{}}}}}\n",
            i % 37
        ));
    }
    for i in 0..500usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{\"w\":{i}}}}}\n",
            (i * 31 + 1) % 500
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    let shapes: &[(&str, &str)] = &[
        ("bare id(), vertex", "g.V().out('R').hasLabel('W').id()"),
        ("bare id(), edge", "g.E().hasLabel('R').id()"),
        ("bare label(), edge", "g.E().hasLabel('R').label()"),
        // `label()` off a VERTEX has no `to_gql` equivalent at all (see
        // `to_gql::tail`'s own comment) — `id_label_prefix` declines it too,
        // so it must stay on the old arm every time.
        (
            "bare label(), vertex",
            "g.V().out('R').hasLabel('W').label()",
        ),
        ("id().count()", "g.V().out('R').hasLabel('W').id().count()"),
        ("id().dedup()", "g.V().out('R').hasLabel('W').id().dedup()"),
        (
            "label().groupCount(), edge",
            "g.E().hasLabel('R').label().groupCount()",
        ),
        (
            "dedup() in front of id()",
            "g.V().out('R').hasLabel('W').dedup().id()",
        ),
        (
            "barrier() in front of id()",
            "g.V().out('R').hasLabel('W').barrier().id()",
        ),
        (
            "limit() in front of id()",
            "g.V().out('R').hasLabel('W').limit(2).id()",
        ),
        (
            "tagged path (select) before id()",
            "g.V().as('x').out('R').select('x').id()",
        ),
        (
            "post-fold (fold().unfold()) before id()",
            "g.V().out('R').hasLabel('W').fold().unfold().id()",
        ),
        ("id().limit()", "g.V().out('R').hasLabel('W').id().limit(5)"),
        ("id().fold()", "g.V().out('R').hasLabel('W').id().fold()"),
    ];

    println!();
    for (label, q) in shapes {
        super::exec::MIGRATED.with(|c| c.set(0));
        super::exec::ID_LABEL_ARM_HIT.with(|c| c.set(0));

        let out = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}` parses: {e}"))
            .run(&mut g);

        let migrated = super::exec::MIGRATED.with(std::cell::Cell::get);
        let old_arm = super::exec::ID_LABEL_ARM_HIT.with(std::cell::Cell::get);
        let route = if migrated > 0 {
            "MIGRATED"
        } else if old_arm > 0 {
            "OLD_ARM"
        } else {
            "NEITHER (third route / stream)"
        };

        println!(
            "PROBE migrated={migrated} id_label_arm={old_arm} rows={} route={route} [{label}] {q}",
            out.len()
        );
    }

    // The reference-comparison switch: with the whole migration block gated
    // off (`col_terminal_tagged`'s `!migrate_off()` guard covers BOTH the
    // whole-tail attempt and the prefix split equally), the bare forms have
    // to reach the old arm directly — `id_label_prefix` cannot change this,
    // since it lives entirely inside the gated block.
    println!();
    super::exec::MIGRATE_OFF.with(|c| c.set(true));
    for q in [
        "g.V().out('R').hasLabel('W').id()",
        "g.E().hasLabel('R').id()",
        "g.E().hasLabel('R').label()",
        "g.V().out('R').hasLabel('W').label()",
    ] {
        super::exec::ID_LABEL_ARM_HIT.with(|c| c.set(0));

        let out = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}` parses: {e}"))
            .run(&mut g);
        let old_arm = super::exec::ID_LABEL_ARM_HIT.with(std::cell::Cell::get);

        println!(
            "PROBE MIGRATE_OFF id_label_arm={old_arm} rows={} [{q}]",
            out.len()
        );
        assert!(old_arm > 0, "{q}: expected the old arm under MIGRATE_OFF");
    }
    super::exec::MIGRATE_OFF.with(|c| c.set(false));
}

/// The headline comparison for the Gremlin migration:
/// `g.V().out('R').hasLabel('W').values('n')` against the identical GQL
/// statement, and against `MIGRATE_OFF` (the `elem_terminal`/`column_terminal`
/// route the migration replaces).
///
/// Both Gremlin routes already avoid the `RowSet` box GQL's own text-execute
/// path pays for: `column_terminal` calls `to_gql::tail` itself
/// (unconditionally, not gated on `migrate_off`) for the bare `values(k)`
/// read, so `MIGRATE_OFF` does not fall back to a naive stream here — it
/// falls back to a route that already shares the same columnar machinery,
/// which is why the two Gremlin numbers read close together while both beat
/// the `RowSet` number.
///
/// Run with:
///
/// ```text
/// cargo test --release -- --ignored --nocapture headline_query_timing_probe
/// ```
#[test]
#[ignore = "timing"]
fn headline_query_timing_probe() {
    let mut lines = String::new();
    for i in 0..150_000usize {
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
    for i in 0..150_000usize {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{i}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
            (i * 31 + 1) % 150_000
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    let q = "g.V().out('R').hasLabel('W').values('n')";
    let gq = "MATCH ()-[:R]->(b:W) RETURN b.n AS n";

    // migrated route (current default)
    let mut best_migrated = f64::MAX;
    for _ in 0..7 {
        let t = std::time::Instant::now();
        let out = super::parse::parse(q).expect("parses").run(&mut g);
        let secs = t.elapsed().as_secs_f64();
        std::hint::black_box(out.len());
        if secs < best_migrated {
            best_migrated = secs;
        }
    }

    // MIGRATE_OFF: the route being replaced (elem_terminal/column_terminal)
    super::exec::MIGRATE_OFF.with(|c| c.set(true));
    let mut best_old = f64::MAX;
    for _ in 0..7 {
        let t = std::time::Instant::now();
        let out = super::parse::parse(q).expect("parses").run(&mut g);
        let secs = t.elapsed().as_secs_f64();
        std::hint::black_box(out.len());
        if secs < best_old {
            best_old = secs;
        }
    }
    super::exec::MIGRATE_OFF.with(|c| c.set(false));

    // The identical GQL statement.
    let mut best_gql = f64::MAX;
    let plan = crate::gql::parse(gq).unwrap_or_else(|e| panic!("`{gq}`: {e}"));
    for _ in 0..7 {
        let t = std::time::Instant::now();
        let rs = plan
            .execute(&mut g, &crate::gql::eval::Params::new())
            .unwrap_or_else(|e| panic!("`{gq}`: {e}"));
        let secs = t.elapsed().as_secs_f64();
        std::hint::black_box(rs.rows().count());
        if secs < best_gql {
            best_gql = secs;
        }
    }

    println!();
    println!(
        "PROBE headline: gremlin(migrated) {:>8.4}ms  gremlin(MIGRATE_OFF) {:>8.4}ms  gql(RowSet) {:>8.4}ms  [{q}]",
        best_migrated * 1e3, best_old * 1e3, best_gql * 1e3
    );
}

/// Investigation for the `where`/`not` filter-arm migration: can
/// `where(__.out('R'))` be expressed as `CExpr::Exists` and answered through
/// `gql::eval::project_ids`, the ONE seam the Gremlin migration is allowed to
/// call without also editing `gql/eval.rs`?
///
/// No. `project_ids` builds its `Ctx` with `labels: Vec::new()` (it only
/// resolves PROPERTY keys, via `key_names` — see its own body) because nothing
/// that calls it has ever needed a LABEL. `exists_semi_join_vec`'s fast path
/// happily accepts a labelled relationship (`rel.label`) — a typed hop is
/// exactly the shape its own doc comment calls out as "the one shape that
/// matters" — but reaching that label test means indexing `ctx.labels[r]`
/// (`eval_label_adj`, `gql/eval.rs`), and an empty `Vec` has no index `0`.
/// This is not a semantic decline (`None`); it is a PANIC, empirically
/// reproduced here rather than assumed from reading the source, because a
/// wrong reading here would silently ship a crash instead of a decline.
///
/// The fix — giving `project_ids` a `label_names` parameter and populating
/// `ctx.labels` the way `plan_pattern_ids` already does two functions up —
/// is a `gql/eval.rs` change, out of scope for a migration confined to
/// `to_gql.rs`/`exec.rs`/`divergence_tests.rs`. So `to_gql::tail` must NEVER
/// emit `CLabelExpr::Label` for a body reached through `project_ids`, which
/// rules out both a typed hop (`out('R')`) and a landed `hasLabel(...)` test —
/// exactly the shapes `elem_terminal`'s existing `Where`/`Not` arms exist to
/// answer, via `semi_join_hop`/`semi_join_chain` (vertex-only, matching
/// `exists_semi_join_vec`'s own restriction for a different reason: `has_adj`
/// walks a vertex's adjacency).
#[test]
fn exists_via_project_ids_cannot_carry_a_label_ctx_has_none() {
    use crate::gql::ast::{Direction, PathMode, PathSelector};
    use crate::gql::plan::{
        compile_program, CExpr, CLabelExpr, CNode, CPath, CProjection, CRel, CReturnItem, CSegment,
    };

    let g = modern();
    // Every vertex: marko (id "1") has two KNOWS/CREATED out-edges, so the walk
    // below is guaranteed to test the label predicate at least once rather than
    // short-circuit past it on an empty adjacency.
    let ids: Vec<u32> = g.vertex_indices().collect();

    // `(u)-[:<ref 0>]->()` — precisely `where(__.out('R'))`'s shape, with the
    // edge-type ref left unresolved (any ref is unresolvable against an empty
    // `ctx.labels`, so which name it names is irrelevant to the point).
    let expr = CExpr::Exists {
        patterns: vec![CPath {
            start: CNode {
                var_slot: Some(0),
                label: None,
                props: Vec::new(),
                where_: None,
            },
            segments: vec![CSegment {
                rel: CRel {
                    var_slot: None,
                    label: Some(CLabelExpr::Label(0)),
                    direction: Direction::Out,
                    props: Vec::new(),
                    where_: None,
                    quantifier: None,
                },
                node: CNode {
                    var_slot: None,
                    label: None,
                    props: Vec::new(),
                    where_: None,
                },
                unit: None,
            }],
            path_var_slot: None,
            selector: PathSelector::Walk,
            mode: PathMode::Trail,
        }],
        where_: None,
        sub_len: 0,
    };
    let item = CReturnItem {
        prog: compile_program(&expr),
        expr,
        name: "v".to_string(),
        is_agg: false,
    };
    let proj = CProjection {
        star: false,
        distinct: false,
        aggregating: false,
        group_by: Vec::new(),
        aggs: Vec::new(),
        out_len: 1,
        star_cols: Vec::new(),
        order_by: Vec::new(),
        order_overlay: Vec::new(),
        order_needs_output: false,
        having: None,
        skip: None,
        limit: None,
        out_names: vec!["v".to_string()],
        items: vec![item],
    };

    // Resolved label names, which is what `project_ids` now takes. It used to
    // build its `Ctx` with `labels: Vec::new()`, so this call PANICKED indexing
    // `ctx.labels[0]` — a landmine rather than a decline, waiting for the first
    // arm to project anything label-bearing.
    let labels = vec!["KNOWS".to_string()];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::gql::eval::project_ids(&g, &ids, false, &[], &labels, &proj)
    }));

    assert!(
        result.is_ok(),
        "project_ids must RESOLVE a label rather than panic on it — it takes \
         `label_names` now, as `plan_pattern_ids` always has. This test asserted \
         the PANIC when it was written, which is what identified the landmine; it \
         pins the fix now."
    );
}

/// Before/after for the `where`/`not` filter-arm migration this file's
/// `exists_via_project_ids_cannot_carry_a_label_ctx_has_none` investigated and
/// declined: `g.V().hasLabel('V').where(__.out('R')).count()` and its `not()`
/// twin, against the equivalent `EXISTS`/`NOT EXISTS` GQL statement, on a
/// 50k-vertex / 150k-edge fixture.
///
/// "After" is identical to "before" by construction — `exec.rs`'s
/// `elem_terminal` `Where`/`Not` arms are UNCHANGED, because the migration
/// this investigated cannot reach a typed hop (`out('R')`) without a
/// `gql/eval.rs` change outside this task's file scope (see the test above).
/// Run to VERIFY the claim that motivated the investigation — that GQL's
/// vectorized `CExpr::Exists` is meaningfully faster than Gremlin's own
/// `semi_join_hop`/`has_adj` path — rather than trust the cited numbers.
///
/// Run with:
///
/// ```text
/// cargo test --release -- --ignored --nocapture where_not_typed_hop_timing_probe
/// ```
#[test]
#[ignore = "timing"]
fn where_not_typed_hop_timing_probe() {
    const VERTICES: usize = 50_000;
    const EDGES: usize = 150_000;

    let mut lines = String::new();
    for i in 0..VERTICES {
        lines.push_str(&format!(
            "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"V\"],\"properties\":{{\"n\":{}}}}}\n",
            i % 97
        ));
    }
    for i in 0..EDGES {
        lines.push_str(&format!(
            "{{\"type\":\"edge\",\"id\":\"e{i}\",\"from\":\"n{}\",\"to\":\"n{}\",\"labels\":[\"R\"],\"properties\":{{}}}}\n",
            i % VERTICES,
            (i * 31 + 1) % VERTICES,
        ));
    }
    let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

    let cases: &[(&str, &str, &str)] = &[
        (
            "where (semi-join keep)",
            "g.V().hasLabel('V').where(__.out('R')).count()",
            "MATCH (u:V) WHERE EXISTS { (u)-[:R]->() } RETURN count(*) AS c",
        ),
        (
            "not (semi-join drop)",
            "g.V().hasLabel('V').not(__.out('R')).count()",
            "MATCH (u:V) WHERE NOT EXISTS { (u)-[:R]->() } RETURN count(*) AS c",
        ),
    ];

    println!();
    for (label, gq, gqlq) in cases {
        let mut best_gremlin = f64::MAX;
        let mut last_out: Vec<GVal> = Vec::new();
        for _ in 0..7 {
            let t = std::time::Instant::now();
            let out = super::parse::parse(gq).expect("parses").run(&mut g);
            let secs = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            last_out = out;
            if secs < best_gremlin {
                best_gremlin = secs;
            }
        }

        let plan = crate::gql::parse(gqlq).unwrap_or_else(|e| panic!("`{gqlq}`: {e}"));
        let mut best_gql = f64::MAX;
        for _ in 0..7 {
            let t = std::time::Instant::now();
            let rs = plan
                .execute(&mut g, &crate::gql::eval::Params::new())
                .unwrap_or_else(|e| panic!("`{gqlq}`: {e}"));
            let secs = t.elapsed().as_secs_f64();
            std::hint::black_box(rs.rows().count());
            if secs < best_gql {
                best_gql = secs;
            }
        }

        println!(
            "PROBE where/not: gremlin {:>8.4}ms  gql {:>8.4}ms  ratio(gremlin/gql) {:>5.2}x  out={:?}  [{label}] {gq}",
            best_gremlin * 1e3,
            best_gql * 1e3,
            best_gremlin / best_gql,
            last_out,
        );
    }
}

/// `elem_terminal`'s `[Step::Where(sub), ..]` / `[Step::Not(sub), ..]` arms,
/// empirically, across the shapes the migration investigation
/// (`exists_via_project_ids_cannot_carry_a_label_ctx_has_none`) found it
/// cannot reach: a typed hop, a landed `hasLabel`, a landed property test, and
/// a multi-hop chain.
///
/// No route change was made here — this is not an agreement check against a
/// replacement route, only confirmation that `WHERE_NOT_ARM_HIT` still fires
/// for exactly the shapes it always has, since the migration this file
/// investigated was never applied (evidenced by
/// `exists_via_project_ids_cannot_carry_a_label_ctx_has_none` and
/// `where_not_typed_hop_timing_probe`, above).
#[test]
fn where_not_arm_is_unchanged_and_still_reachable() {
    let mut g = modern();

    let shapes: &[(&str, &str)] = &[
        ("typed hop (where)", "g.V().where(__.out('KNOWS')).count()"),
        ("typed hop (not)", "g.V().not(__.out('KNOWS')).count()"),
        (
            "landed hasLabel",
            "g.V().where(__.out().hasLabel('SOFTWARE')).count()",
        ),
        (
            "landed property test",
            "g.V().where(__.out().has('lang', eq('java'))).count()",
        ),
        ("multi-hop chain", "g.V().where(__.out().out()).count()"),
        (
            "edge frontier self predicate",
            "g.E().where(__.hasLabel('KNOWS')).count()",
        ),
    ];

    println!();
    for (label, q) in shapes {
        super::exec::WHERE_NOT_ARM_HIT.with(|c| c.set(0));
        let out = super::parse::parse(q)
            .unwrap_or_else(|e| panic!("`{q}` parses: {e}"))
            .run(&mut g);
        let hit = super::exec::WHERE_NOT_ARM_HIT.with(std::cell::Cell::get);
        println!("PROBE where_not_arm_hit={hit} out={out:?} [{label}] {q}");
        // The edge-frontier case is a self predicate that never reaches
        // `has_adj` (`semi_join_test`'s vertex-only hop/chain branches are
        // skipped for `is_edge`), but it still runs the SAME `[Step::Where]`
        // arm in `elem_terminal` — `self_predicate` works for either kind.
        assert!(hit > 0, "{q}: WHERE_NOT_ARM_HIT never fired");
    }
}
