//! Per-step conformance tests, part 2 of 6: `repeat`, `elementMap`, `textP`,
//! `aggregate`, `select`-pop, `choose`, `min`, `coalesce`, side-effects in
//! closures, `mean`, `flatMap`, `addE`, `label`, `fail`, `subgraph`, `cyclicPath`.
//!
//! The six parts are a SIZE split, not a thematic one — the number carries no
//! meaning. Self-contained: own `modern()` fixture + helpers.
//!
//! Mirrors the TS Gremlin step tests. TS closures (`filter((v,t)=>…)`,
//! `choose((p)=>…)`) are expressed as sub-traversals. Tests that hit a genuine
//! Rust gap or behavioral divergence are omitted rather than asserted failing.

use super::{GVal, Step, Traversal};
use crate::graph::Graph;

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gremlin_edge_ids()
}

/// Parse a Gremlin string and run it against a fresh Modern graph.
fn qs(query: &str) -> Vec<GVal> {
    let mut g = modern();
    let t = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

/// Run a fluent traversal against a fresh Modern graph.
fn q(t: Traversal) -> Vec<GVal> {
    let mut g = modern();
    t.run(&mut g)
}

fn s(g: &GVal) -> String {
    match g {
        GVal::Str(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

/// Sorted string results (order-independent traversals).
fn names(r: Vec<GVal>) -> Vec<String> {
    let mut v: Vec<String> = r.iter().map(s).collect();
    v.sort();
    v
}

/// String results in stream order (order-dependent traversals).
fn ordered(r: Vec<GVal>) -> Vec<String> {
    r.iter().map(s).collect()
}

/// Sort a result map's entries by string key.
fn map_sorted(g: &GVal) -> Vec<(String, GVal)> {
    match g {
        GVal::Map(entries) => {
            let mut v: Vec<(String, GVal)> =
                entries.iter().map(|(k, val)| (s(k), val.clone())).collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        }
        _ => panic!("expected map, got {g:?}"),
    }
}

/// Resolve element-ids in a result list of vertices.
fn ids(g: &Graph, r: &[GVal]) -> Vec<String> {
    r.iter()
        .map(|v| match v {
            GVal::Node(i) => g.vid.text(*i).to_string(),
            GVal::Edge(e) => format!("e{e}"),
            other => format!("{other:?}"),
        })
        .collect()
}

// ===== repeat (repeat.test.ts) ============================================

#[test]
fn p2_repeat_times_two() {
    // marko.repeat(out()).times(2) → grandchildren {ripple, lop}.
    assert_eq!(
        names(qs("g.V('1').repeat(__.out()).times(2).values('name')")),
        vec!["lop", "ripple"]
    );
}

#[test]
fn p2_repeat_until_software() {
    let r = qs("g.V('1').repeat(__.out()).until(__.hasLabel('SOFTWARE')).values('name')");
    assert_eq!(names(r), vec!["lop", "lop", "ripple"]);
}

#[test]
fn p2_repeat_until_ripple_from_start() {
    // Pre-form `until(cond).repeat(body)` is while-do — checked BEFORE the body,
    // so starting AT ripple yields ripple without running out(). (Post-form
    // `.until()` is do-while: ripple is a sink → out() drains it → [].)
    let r = qs("g.V('5').until(__.has('name', eq('ripple'))).repeat(__.out()).values('name')");
    assert_eq!(ordered(r), vec!["ripple"]);
}

#[test]
fn p2_repeat_times_two_emit() {
    // post-form emit: AFTER each body application; input (marko) not emitted.
    let r = qs("g.V('1').repeat(__.out()).times(2).emit().values('name')");
    assert_eq!(names(r), vec!["josh", "lop", "lop", "ripple", "vadas"]);
}

#[test]
fn p2_repeat_emit_filtered_software() {
    let r = qs("g.V('1').repeat(__.out()).times(2).emit(__.hasLabel('SOFTWARE')).values('name')");
    assert_eq!(names(r), vec!["lop", "lop", "ripple"]);
}

// NOTE: `emitBefore() yields the start vertex plus every level` now matches TS —
// see divergence_tests::repeat_emit_before_yields_every_level.

#[test]
fn p2_repeat_times_two_path() {
    // repeat(out()).times(2).path().by('name') → full two-hop paths.
    let r = qs("g.V('1').repeat(__.out()).times(2).path().by('name')");
    let mut paths: Vec<String> = r
        .iter()
        .map(|p| match p {
            GVal::List(items) => items.iter().map(s).collect::<Vec<_>>().join(","),
            _ => panic!("expected path list"),
        })
        .collect();
    paths.sort();
    assert_eq!(paths, vec!["marko,josh,lop", "marko,josh,ripple"]);
}

#[test]
fn p2_repeat_times_two_emit_path_starts_marko() {
    let r = qs("g.V('1').repeat(__.out()).times(2).emit().path().by('name')");
    assert!(!r.is_empty());
    for p in &r {
        match p {
            GVal::List(items) => assert_eq!(s(&items[0]), "marko"),
            _ => panic!("expected path list"),
        }
    }
}

#[test]
fn p2_repeat_until_sinks_oute_count() {
    let r = qs("g.V('1').repeat(__.out()).until(__.outE().count().is(eq(0))).values('name')");
    assert_eq!(names(r), vec!["lop", "lop", "ripple", "vadas"]);
}

#[test]
fn p2_repeat_times_three_empty() {
    let r = qs("g.V('1').repeat(__.out()).times(3).values('name')");
    assert!(r.is_empty());
}

#[test]
fn p2_repeat_times_three_emit() {
    let r = qs("g.V('1').repeat(__.out()).times(3).emit().values('name')");
    assert_eq!(names(r), vec!["josh", "lop", "lop", "ripple", "vadas"]);
}

#[test]
fn p2_repeat_times_three_emit_software() {
    let r = qs("g.V('1').repeat(__.out()).times(3).emit(__.hasLabel('SOFTWARE')).values('name')");
    assert_eq!(names(r), vec!["lop", "lop", "ripple"]);
}

#[test]
fn p2_repeat_times_three_until_software() {
    let r = qs("g.V('1').repeat(__.out()).times(3).until(__.hasLabel('SOFTWARE')).values('name')");
    assert_eq!(names(r), vec!["lop", "lop", "ripple"]);
}

#[test]
fn p2_repeat_loops_self_limit() {
    // repeat(out().where(loops().is(lt(2)))).times(5).emit()
    let r =
        qs("g.V('1').repeat(__.out().where(__.loops().is(lt(2)))).times(5).emit().values('name')");
    assert_eq!(names(r), vec!["josh", "lop", "vadas"]);
}

#[test]
fn p2_repeat_empty_input() {
    let r = qs("g.V('999').repeat(__.out()).times(3).values('name')");
    assert!(r.is_empty());
}

#[test]
fn p2_repeat_times_zero_passthrough() {
    let r = qs("g.V('1').repeat(__.out()).times(0).values('name')");
    assert_eq!(ordered(r), vec!["marko"]);
}

#[test]
fn p2_repeat_until_true_on_input() {
    // Pre-form `until(cond).repeat(body)` is while-do: starting at lop (SOFTWARE),
    // the pre-form until is checked first → the input passes through unchanged.
    let r = qs("g.V('3').until(__.hasLabel('SOFTWARE')).repeat(__.out()).values('name')");
    assert_eq!(ordered(r), vec!["lop"]);
}

#[test]
fn p2_repeat_times_cap_high() {
    let r = qs("g.V('1').repeat(__.out()).times(50).values('name')");
    assert!(r.is_empty());
}

// ===== elementMap (elementMap.test.ts) ====================================

/// Compare an elementMap result vertex-map to an expected key→value list.
fn assert_emap(got: &GVal, want: &[(&str, GVal)]) {
    let m = map_sorted(got);
    let mut w: Vec<(String, GVal)> = want
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    w.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(m, w);
}

#[test]
fn p2_element_map_one_key() {
    let r = qs("g.V().elementMap('name')");
    assert_eq!(r.len(), 6);
    // order = marko, vadas, josh, peter, lop, ripple
    assert_emap(
        &r[0],
        &[
            ("id", GVal::Str("1".into())),
            ("label", GVal::Str("PERSON".into())),
            ("name", GVal::Str("marko".into())),
        ],
    );
    assert_emap(
        &r[4],
        &[
            ("id", GVal::Str("3".into())),
            ("label", GVal::Str("SOFTWARE".into())),
            ("name", GVal::Str("lop".into())),
        ],
    );
}

#[test]
fn p2_element_map_no_keys_all_props() {
    let r = qs("g.V().elementMap()");
    assert_emap(
        &r[0],
        &[
            ("id", GVal::Str("1".into())),
            ("label", GVal::Str("PERSON".into())),
            ("name", GVal::Str("marko".into())),
            ("age", GVal::Num(29.0)),
        ],
    );
    assert_emap(
        &r[4],
        &[
            ("id", GVal::Str("3".into())),
            ("label", GVal::Str("SOFTWARE".into())),
            ("name", GVal::Str("lop".into())),
            ("lang", GVal::Str("java".into())),
        ],
    );
}

#[test]
fn p2_element_map_missing_key_on_some() {
    // elementMap('age') — software has no age → just id+label.
    let r = qs("g.V().elementMap('age')");
    assert_emap(
        &r[0],
        &[
            ("id", GVal::Str("1".into())),
            ("label", GVal::Str("PERSON".into())),
            ("age", GVal::Num(29.0)),
        ],
    );
    assert_emap(
        &r[4],
        &[
            ("id", GVal::Str("3".into())),
            ("label", GVal::Str("SOFTWARE".into())),
        ],
    );
}

#[test]
fn p2_element_map_skips_unknown_key() {
    let r = qs("g.V().elementMap('age', 'blah')");
    assert_emap(
        &r[0],
        &[
            ("id", GVal::Str("1".into())),
            ("label", GVal::Str("PERSON".into())),
            ("age", GVal::Num(29.0)),
        ],
    );
}

#[test]
fn p2_element_map_after_has_within() {
    let r = qs("g.V().has('name', within('josh','marko')).elementMap()");
    assert_eq!(r.len(), 2);
    let got = names(
        r.iter()
            .map(|m| match m {
                GVal::Map(e) => e
                    .iter()
                    .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == "name"))
                    .map(|(_, v)| v.clone())
                    .unwrap(),
                _ => panic!(),
            })
            .collect(),
    );
    assert_eq!(got, vec!["josh", "marko"]);
}

#[test]
fn p2_element_map_after_not_haslabel() {
    let r = qs("g.V().not(__.hasLabel('PERSON')).elementMap()");
    assert_eq!(r.len(), 2);
    assert_emap(
        &r[0],
        &[
            ("id", GVal::Str("3".into())),
            ("label", GVal::Str("SOFTWARE".into())),
            ("name", GVal::Str("lop".into())),
            ("lang", GVal::Str("java".into())),
        ],
    );
}

#[test]
fn p2_element_map_on_edge_in_out_submaps() {
    // marko -[CREATED #9]-> lop, weight 0.4. Edge elementMap has IN/OUT submaps.
    let r = qs("g.V('1').outE('CREATED').elementMap()");
    assert_eq!(r.len(), 1);
    let m = map_sorted(&r[0]);
    let get = |k: &str| m.iter().find(|(key, _)| key == k).map(|(_, v)| v.clone());
    assert_eq!(get("id"), Some(GVal::Str("9".into())));
    assert_eq!(get("label"), Some(GVal::Str("CREATED".into())));
    assert_eq!(get("weight"), Some(GVal::Num(0.4)));
    // IN endpoint = lop (3, SOFTWARE), OUT = marko (1, PERSON).
    assert_eq!(
        map_sorted(&get("IN").unwrap()),
        vec![
            ("id".to_string(), GVal::Str("3".into())),
            ("label".to_string(), GVal::Str("SOFTWARE".into())),
        ]
    );
    assert_eq!(
        map_sorted(&get("OUT").unwrap()),
        vec![
            ("id".to_string(), GVal::Str("1".into())),
            ("label".to_string(), GVal::Str("PERSON".into())),
        ]
    );
}

#[test]
fn p2_element_map_all_edges() {
    let r = qs("g.E().elementMap('weight')");
    assert_eq!(r.len(), 6);
}

// ===== TextP predicates (textP.test.ts) ===================================

#[test]
fn p2_textp_containing_o() {
    let r = qs("g.V().has('name', containing('o')).values('name')");
    assert_eq!(names(r), vec!["josh", "lop", "marko"]);
}

#[test]
fn p2_textp_not_containing_o() {
    let r = qs("g.V().has('name', notContaining('o')).values('name')");
    assert_eq!(names(r), vec!["peter", "ripple", "vadas"]);
}

#[test]
fn p2_textp_ending_with_o() {
    let r = qs("g.V().hasLabel('PERSON').has('name', endingWith('o')).values('name')");
    assert_eq!(ordered(r), vec!["marko"]);
}

#[test]
fn p2_textp_starts_with_m() {
    let r = qs("g.V().hasLabel('PERSON').has('name', startingWith('m')).values('name')");
    assert_eq!(ordered(r), vec!["marko"]);
}

// NOTE: regex(...) predicate tests are SKIPPED — the Rust parser/engine has no
// `regex` TextP predicate (no P::Regex variant). See delivery notes.

// ===== aggregate / cap (aggregate.test.ts) ================================

#[test]
fn p2_aggregate_passthrough() {
    let r = qs("g.V('1').out('CREATED').aggregate('x').values('name')");
    assert_eq!(ordered(r), vec!["lop"]);
}

#[test]
fn p2_aggregate_transparent_downstream() {
    let r = qs("g.V('1').out('CREATED').aggregate('x').in('CREATED').id()");
    assert_eq!(names(r), vec!["1", "4", "6"]);
}

#[test]
fn p2_cap_reads_bag() {
    let r = qs("g.V().out('KNOWS').aggregate('x').cap('x')");
    assert_eq!(r.len(), 1);
    let bag = match &r[0] {
        GVal::List(items) => items,
        _ => panic!("expected list bag"),
    };
    let g = modern();
    let mut got = ids(&g, bag);
    got.sort();
    assert_eq!(got, vec!["2", "4"]);
}

#[test]
fn p2_cap_empty_key() {
    let r = qs("g.V('1').cap('never-set')");
    assert_eq!(r, vec![GVal::List(vec![])]);
}

#[test]
fn p2_aggregate_full_stream_before_cap() {
    let r = qs("g.V().aggregate('all').cap('all')");
    let bag = match &r[0] {
        GVal::List(items) => items,
        _ => panic!(),
    };
    let g = modern();
    let mut got = ids(&g, bag);
    got.sort();
    assert_eq!(got, vec!["1", "2", "3", "4", "5", "6"]);
}

#[test]
fn p2_aggregate_transparent_long_chain() {
    let r = qs("g.V('1').out('CREATED').aggregate('x').in('CREATED').out('CREATED').id()");
    assert_eq!(names(r), vec!["3", "3", "3", "5"]);
}

#[test]
fn p2_multiple_aggregates_independent_keys() {
    let r = qs("g.V().aggregate('persons').aggregate('all').cap('persons')");
    let bag = match &r[0] {
        GVal::List(items) => items,
        _ => panic!(),
    };
    let g = modern();
    let mut got = ids(&g, bag);
    got.sort();
    assert_eq!(got, vec!["1", "2", "3", "4", "5", "6"]);
}

// NOTE: `filter(withoutBag('x'))` / `withinBag` tests are SKIPPED — they rely on
// JS closures reading `t.sideEffects`; the data-plan engine has no closure/bag-
// filter step. See delivery notes.

// ===== select with Pop (select-pop.test.ts) ===============================

#[test]
fn p2_select_pop_default_last_single() {
    let r = qs("g.V('1').as('start').select('start').values('name')");
    assert_eq!(ordered(r), vec!["marko"]);
}

#[test]
fn p2_select_pop_first_single() {
    let r = qs("g.V('1').as('start').select(Pop.first, 'start').values('name')");
    assert_eq!(ordered(r), vec!["marko"]);
}

#[test]
fn p2_select_pop_all_single() {
    let r = qs("g.V('1').as('start').select(Pop.all, 'start')");
    assert_eq!(r.len(), 1);
    match &r[0] {
        GVal::List(items) => assert_eq!(items.len(), 1),
        _ => panic!("expected list"),
    }
}

#[test]
fn p2_select_pop_last_inside_repeat() {
    let r = qs("g.V('4').repeat(__.out('CREATED').as('a')).times(1).select('a').values('name')");
    assert_eq!(names(r), vec!["lop", "ripple"]);
}

#[test]
fn p2_select_pop_first_inside_repeat() {
    let r =
        qs("g.V('1').repeat(__.out().as('hop')).times(2).select(Pop.first, 'hop').values('name')");
    assert_eq!(names(r), vec!["josh", "josh"]);
}

#[test]
fn p2_select_pop_all_inside_repeat() {
    let r = qs("g.V('1').repeat(__.out().as('hop')).times(2).select(Pop.all, 'hop')");
    assert_eq!(r.len(), 2);
    for list in &r {
        match list {
            GVal::List(items) => assert_eq!(items.len(), 2),
            _ => panic!("expected list"),
        }
    }
}

// ===== choose (choose.test.ts) ============================================

#[test]
fn p2_choose_then_else() {
    // choose(has('name','marko'), values('age'), values('name'))
    let r = qs("g.V().choose(__.has('name', eq('marko')), __.values('age'), __.values('name'))");
    assert_eq!(
        r,
        vec![
            GVal::Num(29.0),
            GVal::Str("vadas".into()),
            GVal::Str("josh".into()),
            GVal::Str("peter".into()),
            GVal::Str("lop".into()),
            GVal::Str("ripple".into()),
        ]
    );
}

#[test]
fn p2_choose_haslabel_branches() {
    // choose(hasLabel('PERSON'), out('CREATED'), identity()).values('name')
    let r =
        qs("g.V().choose(__.hasLabel('PERSON'), __.out('CREATED'), __.identity()).values('name')");
    assert_eq!(
        ordered(r),
        vec!["lop", "ripple", "lop", "lop", "lop", "ripple"]
    );
}

#[test]
fn p2_choose_by_age_predicate() {
    // hasLabel('PERSON').choose(values('age').is(lte(30)), in(), out()).values('name')
    let r = qs(
        "g.V().hasLabel('PERSON').choose(__.values('age').is(lte(30)), __.in(), __.out()).values('name')",
    );
    assert_eq!(ordered(r), vec!["marko", "ripple", "lop", "lop"]);
}

#[test]
fn p2_choose_on_oute_count() {
    // choose(outE('KNOWS').count().is(gt(0)), out('KNOWS'), identity())
    let r = qs(
        "g.V().hasLabel('PERSON').choose(__.outE('KNOWS').count().is(gt(0)), __.out('KNOWS'), __.identity()).values('name')",
    );
    assert_eq!(ordered(r), vec!["vadas", "josh", "vadas", "josh", "peter"]);
}

#[test]
fn p2_choose_no_else_is_identity() {
    // choose(hasLabel('PERSON'), out('CREATED')) — missing else = identity.
    let r = qs("g.V().choose(__.hasLabel('PERSON'), __.out('CREATED')).values('name')");
    assert_eq!(
        ordered(r),
        vec!["lop", "ripple", "lop", "lop", "lop", "ripple"]
    );
}

#[test]
fn p2_choose_no_else_test_fails_passthrough() {
    let r = qs(
        "g.V().hasLabel('PERSON').choose(__.has('name', eq('nonexistent')), __.out('CREATED')).values('name')",
    );
    assert_eq!(ordered(r), vec!["marko", "vadas", "josh", "peter"]);
}

// ===== min (min.test.ts) ==================================================

#[test]
fn p2_min_numbers() {
    let r = qs("g.V().values('age').min()");
    assert_eq!(r, vec![GVal::Num(27.0)]);
}

#[test]
fn p2_min_strings() {
    let r = qs("g.V().values('name').min()");
    assert_eq!(r, vec![GVal::Str("josh".into())]);
}

#[test]
fn p2_min_after_repeat_both_times_three() {
    let r = qs("g.V().repeat(__.both()).times(3).values('age').min()");
    assert_eq!(r, vec![GVal::Num(27.0)]);
}

#[test]
fn p2_min_all_null_yields_null() {
    // inject(null,null,null,null).min() → null (no textual null literal; build
    // the step directly).
    let t = Traversal {
        steps: vec![
            Step::Inject(vec![GVal::Null, GVal::Null, GVal::Null, GVal::Null]),
            Step::Min(super::Scope::Global),
        ],
    };
    assert_eq!(q(t), vec![GVal::Null]);
}

// NOTE: `min filters out null` (inject(null,10,9,null).min() → 9) now matches
// TS — see divergence_tests::min_skips_nulls / min_all_null_is_null.

// ===== coalesce (coalesce.test.ts) ========================================

#[test]
fn p2_coalesce_falls_back_to_name() {
    let r = qs("g.V().hasLabel('PERSON').coalesce(__.values('nickname'), __.values('name'))");
    assert_eq!(ordered(r), vec!["marko", "vadas", "josh", "peter"]);
}

#[test]
fn p2_coalesce_first_nonempty_created() {
    let r = qs("g.V('1').coalesce(__.outE('CREATED'), __.outE('KNOWS')).inV().values('name')");
    assert_eq!(ordered(r), vec!["lop"]);
}

#[test]
fn p2_coalesce_knows_first_paths() {
    let r = qs(
        "g.V('1').coalesce(__.outE('KNOWS'), __.outE('CREATED')).inV().path().by('name').by(__.label())",
    );
    let paths: Vec<Vec<String>> = r
        .iter()
        .map(|p| match p {
            GVal::List(items) => items.iter().map(s).collect(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(
        paths,
        vec![
            vec!["marko", "KNOWS", "vadas"],
            vec!["marko", "KNOWS", "josh"],
        ]
    );
}

#[test]
fn p2_coalesce_created_first_path() {
    let r = qs(
        "g.V('1').coalesce(__.outE('CREATED'), __.outE('KNOWS')).inV().path().by('name').by(__.label())",
    );
    let paths: Vec<Vec<String>> = r
        .iter()
        .map(|p| match p {
            GVal::List(items) => items.iter().map(s).collect(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(paths, vec![vec!["marko", "CREATED", "lop"]]);
}

#[test]
fn p2_coalesce_knows_first_names() {
    let r = qs("g.V('1').coalesce(__.outE('KNOWS'), __.outE('CREATED')).inV().values('name')");
    assert_eq!(ordered(r), vec!["vadas", "josh"]);
}

// NOTE: sideeffects-in-closures.test.ts is fully SKIPPED — all four tests use JS
// closures reading `t.sideEffects` (filter((v,t)=>…), withinBag/withoutBag,
// capturing the side-effect map); no data-plan equivalent. See delivery notes.

// ===== mean (mean.test.ts) ================================================

#[test]
fn p2_mean_numbers() {
    let r = qs("g.V().values('age').mean()");
    assert_eq!(r, vec![GVal::Num(30.75)]);
}

#[test]
fn p2_mean_after_repeat_both_times_three() {
    let r = qs("g.V().repeat(__.both()).times(3).values('age').mean()");
    assert_eq!(r, vec![GVal::Num(1471.0 / 48.0)]);
}

#[test]
fn p2_mean_filters_null() {
    // inject(null,10,9,null).mean() → 9.5 (numbers-only fold).
    let t = Traversal {
        steps: vec![
            Step::Inject(vec![
                GVal::Null,
                GVal::Num(10.0),
                GVal::Num(9.0),
                GVal::Null,
            ]),
            Step::Mean(super::Scope::Global),
        ],
    };
    assert_eq!(q(t), vec![GVal::Num(9.5)]);
}

// NOTE: `mean/sum take null if that is all they got` now matches TS — see
// divergence_tests::mean_all_null_is_null / sum_all_null_is_null.

// ===== flatMap (flatMap.test.ts) ==========================================

#[test]
fn p2_flatmap_expands_via_subplan() {
    let r = qs("g.V('1').flatMap(__.out()).values('name')");
    assert_eq!(names(r), vec!["josh", "lop", "vadas"]);
}

#[test]
fn p2_flatmap_drops_empty() {
    let r = qs("g.V().hasLabel('SOFTWARE').flatMap(__.out())");
    assert!(r.is_empty());
}

#[test]
fn p2_flatmap_values_equiv() {
    let r = qs("g.V().hasLabel('PERSON').flatMap(__.values('name'))");
    assert_eq!(names(r), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p2_flatmap_many_per_input() {
    let r = qs("g.V().hasLabel('PERSON').flatMap(__.out('CREATED')).values('name')");
    assert_eq!(names(r), vec!["lop", "lop", "lop", "ripple"]);
}

// ===== addE (addE.test.ts) ================================================

#[test]
fn p2_adde_to_subplan() {
    // marko -[NEMESIS]-> peter; input is FROM, sub-plan is TO.
    let mut g = modern();
    let before = q(super::g().E().count());
    let r = super::parse("g.V('1').addE('NEMESIS').to(__.V('6'))")
        .unwrap()
        .run(&mut g);
    assert_eq!(r.len(), 1);
    // edge count went up by one
    let after = super::g().E().count().run(&mut g);
    assert_eq!(after, vec![GVal::Num(7.0)]);
    assert_eq!(before, vec![GVal::Num(6.0)]);
    // the new edge connects marko -> peter with label NEMESIS
    let names_out = super::parse("g.V('1').out('NEMESIS').values('name')")
        .unwrap()
        .run(&mut g);
    assert_eq!(ordered(names_out), vec!["peter"]);
}

#[test]
fn p2_adde_from_tag() {
    // tag marko, hop to out-neighbors, addE('META').from('start').to(V('6')).
    let mut g = modern();
    let r =
        super::parse("g.V('1').as('start').out('KNOWS').addE('META').from('start').to(__.V('6'))")
            .unwrap()
            .run(&mut g);
    assert_eq!(r.len(), 2); // marko knows vadas + josh → 2 edges
    let count = super::g().E().count().run(&mut g);
    assert_eq!(count, vec![GVal::Num(8.0)]);
    // both new META edges go marko -> peter
    let metas = super::parse("g.V('1').out('META').values('name')")
        .unwrap()
        .run(&mut g);
    assert_eq!(names(metas), vec!["peter", "peter"]);
}

#[test]
fn p2_adde_with_property() {
    let mut g = modern();
    super::parse("g.V('1').addE('KNOWS').to(__.V('6')).property('weight', 0.42)")
        .unwrap()
        .run(&mut g);
    let w = super::parse("g.V('1').outE('KNOWS').has('weight', eq(0.42)).values('weight')")
        .unwrap()
        .run(&mut g);
    assert_eq!(w, vec![GVal::Num(0.42)]);
}

// addE.test.ts: an unresolvable endpoint is a data fault (MissingVertex via
// try_run), matching TS, rather than a silent drop. (The bare-`addE()` "neither
// .from nor .to" case differs by design: Rust's endpoints default to the current
// traverser rather than being unset, so it has no "both unspecified" state.)
#[test]
fn p2_add_e_unresolvable_endpoint_faults() {
    let mut g = modern();
    let t = super::parse("g.V('1').addE('NEMESIS').to(__.V('999'))").unwrap();
    let err = super::try_run(&mut g, &t).unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::MissingVertex);
}

// ===== label (label.test.ts) ==============================================

#[test]
fn p2_label_vertices() {
    let r = qs("g.V().label()");
    assert_eq!(
        ordered(r),
        vec!["PERSON", "PERSON", "PERSON", "PERSON", "SOFTWARE", "SOFTWARE"]
    );
}

#[test]
fn p2_label_edges() {
    let r = qs("g.V('1').outE().label()");
    assert_eq!(ordered(r), vec!["KNOWS", "KNOWS", "CREATED"]);
}

#[test]
fn p2_label_on_property_returns_key() {
    let r = qs("g.V('1').properties().label()");
    assert_eq!(names(r), vec!["age", "name"]);
}

// ===== fail (fail.test.ts) ================================================

#[test]
fn p2_fail_throws_with_message() {
    // fail() on a non-empty stream is a DataException surfaced by try_run —
    // carrying the user's message — NOT a process-aborting panic. TS throws a
    // catchable error here too. (`run` ignores it, matching the addV/addE faults.)
    let mut g = modern();
    let t =
        super::parse("g.V().hasLabel('PERSON').has('name', eq('peter')).fold().fail('Test Fail')")
            .unwrap();
    let err = super::try_run(&mut g, &t).unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::DataException);
    assert!(err.message.contains("Test Fail"), "got: {}", err.message);
}

#[test]
fn p2_fail_no_throw_on_empty_stream() {
    // Empty stream: fail() is a pass-through — no fault, even via try_run.
    let mut g = modern();
    let t = super::parse("g.V().has('name', eq('nobody')).fail('should not fire')").unwrap();
    assert!(super::try_run(&mut g, &t).unwrap().is_empty());
}

#[test]
fn p2_fail_default_message() {
    let mut g = modern();
    let t = super::parse("g.V().fail()").unwrap();
    let err = super::try_run(&mut g, &t).unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::DataException);
    assert!(
        err.message.contains("fail() reached"),
        "got: {}",
        err.message
    );
}

// ===== subgraph (subgraph.test.ts) ========================================
//
// The Rust GVal has no Graph type, so cap() of a subgraph key yields a
// {vertices, edges} id-list map; the membership counts match the TS Graph.

fn subgraph_counts(r: Vec<GVal>) -> (usize, usize) {
    match r.as_slice() {
        [GVal::Map(entries)] => {
            let get = |k: &str| {
                entries
                    .iter()
                    .find(|(key, _)| matches!(key, GVal::Str(s) if s.as_ref() == k))
                    .map(|(_, v)| v)
            };
            let len = |v: Option<&GVal>| match v {
                Some(GVal::List(l)) => l.len(),
                _ => 0,
            };
            (len(get("vertices")), len(get("edges")))
        }
        _ => panic!("expected a subgraph map, got {r:?}"),
    }
}

#[test]
fn p2_subgraph_collect_knows_edges() {
    let r = qs("g.E().hasLabel('KNOWS').subgraph('sg').cap('sg')");
    assert_eq!(subgraph_counts(r), (3, 2));
}

#[test]
fn p2_subgraph_chained_accumulation() {
    let r = qs("g.V().outE('KNOWS').subgraph('knowsG').inV().outE('CREATED').subgraph('createdG').inV().cap('createdG')");
    assert_eq!(subgraph_counts(r), (3, 2));
}

// ===== cyclicPath (cyclicPath.test.ts) ====================================

#[test]
fn p2_cyclic_path_keeps_repeats() {
    // V(1).both().both().cyclicPath() → marko thrice.
    let r = qs("g.V('1').both().both().cyclicPath().id()");
    assert_eq!(ordered(r), vec!["1", "1", "1"]);
}

#[test]
fn p2_cyclic_path_then_path() {
    let r = qs("g.V('1').both().both().cyclicPath().path()");
    assert_eq!(r.len(), 3);
    let g = modern();
    for p in &r {
        match p {
            GVal::List(items) => {
                let pids = ids(&g, items);
                assert_eq!(pids.first().map(String::as_str), Some("1"));
                assert_eq!(pids.last().map(String::as_str), Some("1"));
            }
            _ => panic!("expected path list"),
        }
    }
}
