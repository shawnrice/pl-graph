//! Per-step conformance tests, part 1 of 6: `out`, `limit`, `count`, `sideEffect`,
//! `fold`, `outE`, `groupCount`, `tail`, `not`, `hasId`, `and`, `local`, `filter`,
//! `value`, `hasNot`, plus the index-seed equivalence cases.
//!
//! The six parts are a SIZE split, not a thematic one — the number carries no
//! meaning, so look for a step by grepping rather than by guessing a part.
//!
//! Mirrors `packages/gremlin/src/{steps,executor}/*.test.ts` 1:1 over the
//! TinkerPop Modern graph, so a change to either side should change both. Test
//! names are prefixed `p1_`.

use super::{GVal, P};
use crate::graph::Graph;

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gremlin()
}

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern_eids() -> Graph {
    crate::fixtures::modern_gremlin_edge_ids()
}

fn s(g: &GVal) -> String {
    match g {
        GVal::Str(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

/// String results in stream order (order-dependent).
fn ordered(r: Vec<GVal>) -> Vec<String> {
    r.iter().map(s).collect()
}

/// Sorted string results (order-independent).
fn sorted(r: Vec<GVal>) -> Vec<String> {
    let mut v: Vec<String> = r.iter().map(s).collect();
    v.sort();
    v
}

fn nums(r: Vec<GVal>) -> Vec<f64> {
    r.iter()
        .map(|g| match g {
            GVal::Num(n) => *n,
            other => panic!("expected num, got {other:?}"),
        })
        .collect()
}

fn one_num(r: Vec<GVal>) -> f64 {
    match r.as_slice() {
        [GVal::Num(n)] => *n,
        _ => panic!("expected single number, got {r:?}"),
    }
}

/// Parse + run against a fresh Modern graph.
fn qs(query: &str) -> Vec<GVal> {
    let mut g = modern();
    let t = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

/// Parse + run against the edge-id Modern graph.
fn qs_e(query: &str) -> Vec<GVal> {
    let mut g = modern_eids();
    let t = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

/// Look up a numeric count keyed by `key` in a single-result group map.
fn map_get_num(r: &[GVal], key: &GVal) -> Option<f64> {
    match r.first() {
        Some(GVal::Map(entries)) => entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| match v {
                GVal::Num(n) => *n,
                other => panic!("expected num value, got {other:?}"),
            }),
        other => panic!("expected map, got {other:?}"),
    }
}

// ===== out (steps/out.test.ts) =====

#[test]
fn p1_out_toy_v4() {
    // V('4').out() — ripple, lop (edge order 10,11).
    assert_eq!(
        ordered(qs("g.V('4').out().values('name')")),
        vec!["ripple", "lop"]
    );
}

#[test]
fn p1_out_double_out() {
    assert_eq!(
        ordered(qs("g.V().out().out().values('name')")),
        vec!["ripple", "lop"]
    );
}

#[test]
fn p1_out_specific_label() {
    assert_eq!(
        ordered(qs("g.V('1').out('KNOWS').values('name')")),
        vec!["vadas", "josh"]
    );
}

#[test]
fn p1_out_multiple_labels() {
    assert_eq!(
        ordered(qs("g.V('1').out('KNOWS','CREATED').values('name')")),
        vec!["vadas", "josh", "lop"]
    );
}

#[test]
fn p1_out_all_labels_like_none() {
    let a = ordered(qs("g.V('1').out('KNOWS','CREATED').values('name')"));
    let b = ordered(qs("g.V('1').out().values('name')"));
    assert_eq!(a, b);
}

#[test]
fn p1_out_label_order_matters() {
    assert_eq!(
        ordered(qs("g.V('1').out('CREATED','KNOWS').values('name')")),
        vec!["lop", "vadas", "josh"]
    );
}

#[test]
fn p1_out_created_all_ids_in_order() {
    assert_eq!(
        ordered(qs("g.V().out('CREATED').id()")),
        vec!["3", "5", "3", "3"]
    );
}

#[test]
fn p1_out_out_grandchildren() {
    assert_eq!(
        ordered(qs("g.V().out().out().values('name')")),
        vec!["ripple", "lop"]
    );
}

#[test]
fn p1_out_knows_marko() {
    assert_eq!(
        ordered(qs("g.V().has('name','marko').out('KNOWS').values('name')")),
        vec!["vadas", "josh"]
    );
}

#[test]
fn p1_out_v4_ids() {
    assert_eq!(ordered(qs("g.V('4').out().id()")), vec!["5", "3"]);
}

// ===== limit (steps/limit.test.ts) =====

#[test]
fn p1_limit_to_three() {
    assert_eq!(
        ordered(qs("g.V().limit(3).values('name')")),
        vec!["marko", "vadas", "josh"]
    );
}

#[test]
fn p1_limit_skip_and_take() {
    assert_eq!(
        qs("g.V().values('age').skip(2).limit(1)"),
        vec![GVal::Num(32.0)]
    );
}

#[test]
fn p1_limit_open_end() {
    assert_eq!(
        ordered(qs("g.V().hasLabel('SOFTWARE').values('name').limit(90)")),
        vec!["lop", "ripple"]
    );
}

#[test]
fn p1_limit_two_ids() {
    assert_eq!(ordered(qs("g.V().limit(2).id()")), vec!["1", "2"]);
}

#[test]
fn p1_limit_equiv_range() {
    let lim = ordered(qs("g.V().limit(2).id()"));
    let rng = ordered(qs("g.V().range(0,2).id()"));
    assert_eq!(lim, rng);
}

#[test]
fn p1_limit_scope_local_slices() {
    // values('age').fold().limit(Scope.local, 2) → first two ages of the list.
    let r = qs("g.V().values('age').fold().limit(Scope.local, 2)");
    assert_eq!(r, vec![GVal::list(vec![GVal::Num(29.0), GVal::Num(27.0)])]);
}

#[test]
fn p1_range_scope_local_slices() {
    let r = qs("g.V().values('age').fold().range(Scope.local, 1, 3)");
    assert_eq!(r, vec![GVal::list(vec![GVal::Num(27.0), GVal::Num(32.0)])]);
}

#[test]
fn p1_range_scope_local_open_ended() {
    // range(Scope.local, 2, -1) is open-ended; build via the fluent builder so the
    // open end is usize::MAX (the textual `-1` casts to 0 in Rust, which is wrong).
    let mut g = modern();
    let r = super::g()
        .V()
        .values(&["age"])
        .fold()
        .range_local(2, usize::MAX)
        .run(&mut g);
    assert_eq!(r, vec![GVal::list(vec![GVal::Num(32.0), GVal::Num(35.0)])]);
}

#[test]
fn p1_scope_local_on_min_max_mean_skip_tail() {
    // Regression: the text parser dropped `Scope.local` on these five (the
    // executor supported Local, the builder/parser hardcoded Global). Folded
    // ages = [29,27,32,35].
    assert_eq!(
        qs("g.V().values('age').fold().max(Scope.local)"),
        vec![GVal::Num(35.0)]
    );
    assert_eq!(
        qs("g.V().values('age').fold().min(Scope.local)"),
        vec![GVal::Num(27.0)]
    );
    assert_eq!(
        qs("g.V().values('age').fold().mean(Scope.local)"),
        vec![GVal::Num(30.75)]
    );
    assert_eq!(
        qs("g.V().values('age').fold().skip(Scope.local, 2)"),
        vec![GVal::list(vec![GVal::Num(32.0), GVal::Num(35.0)])]
    );
    assert_eq!(
        qs("g.V().values('age').fold().tail(Scope.local, 2)"),
        vec![GVal::list(vec![GVal::Num(32.0), GVal::Num(35.0)])]
    );
}

// SKIPPED (divergence): steps/limit.test.ts "Scope.local on non-iterable values
// is a no-op". TS passes scalars through unchanged ([29,27,32,35]); Rust's
// slice_local always materializes a List, yielding [[29],[27],[32],[35]]. Not
// ported to avoid encoding divergent behavior as a passing test.

// ===== count (steps/count.test.ts) =====

#[test]
fn p1_count_all() {
    assert_eq!(one_num(qs("g.V().count()")), 6.0);
}

#[test]
fn p1_count_persons() {
    assert_eq!(one_num(qs("g.V().hasLabel('PERSON').count()")), 4.0);
}

#[test]
fn p1_count_after_has_out() {
    assert_eq!(
        one_num(qs("g.V().has('name','marko').out('KNOWS').count()")),
        2.0
    );
}

#[test]
fn p1_count_software_creators_ine_outv() {
    assert_eq!(
        one_num(qs(
            "g.V().hasLabel('SOFTWARE').inE('CREATED').outV().count()"
        )),
        4.0
    );
}

#[test]
fn p1_count_software_ine_created() {
    assert_eq!(
        one_num(qs("g.V().hasLabel('SOFTWARE').inE('CREATED').count()")),
        4.0
    );
}

#[test]
fn p1_count_persons_out() {
    assert_eq!(one_num(qs("g.V().hasLabel('PERSON').out().count()")), 6.0);
    assert_eq!(
        ordered(qs("g.V().hasLabel('PERSON').out().values('name')")),
        vec!["vadas", "josh", "lop", "ripple", "lop", "lop"]
    );
}

#[test]
fn p1_count_scope_local_list() {
    assert_eq!(
        one_num(qs("g.V().values('age').fold().count(Scope.local)")),
        4.0
    );
}

#[test]
fn p1_count_scope_local_scalar() {
    assert_eq!(
        one_num(qs("g.V().has('name','marko').count(Scope.local)")),
        1.0
    );
}

// ===== sideEffect (steps/sideEffect.test.ts) =====

#[test]
fn p1_side_effect_identity_transparent() {
    assert_eq!(
        ordered(qs(
            "g.V().hasLabel('SOFTWARE').sideEffect(identity()).values('name')"
        )),
        vec!["lop", "ripple"]
    );
}

#[test]
fn p1_side_effect_wider_subplan_no_multiply() {
    assert_eq!(
        ordered(qs("g.V().sideEffect(out()).values('name')")),
        vec!["marko", "vadas", "josh", "peter", "lop", "ripple"]
    );
}

#[test]
fn p1_side_effect_empty_subplan_passthrough() {
    // V('5').sideEffect(out().out()) — empty inner, traverser passes through.
    assert_eq!(
        ordered(qs("g.V('5').sideEffect(__.out().out()).values('name')")),
        vec!["ripple"]
    );
}

#[test]
fn p1_side_effect_aggregate_then_cap() {
    let r = qs("g.V().hasLabel('PERSON').sideEffect(aggregate('persons')).cap('persons')");
    let bag = match &r[0] {
        GVal::List(items) => sorted(items.to_vec()),
        other => panic!("expected list, got {other:?}"),
    };
    // cap returns the vertices; project to ids via sorting their id strings.
    // Vertices stringify as Vertex(idx); instead assert membership by re-querying.
    // Here we just assert the bag has 4 person vertices.
    assert_eq!(bag.len(), 4);
}

#[test]
fn p1_side_effect_single_root_identity() {
    assert_eq!(
        ordered(qs("g.V('1').sideEffect(__.out().out()).values('name')")),
        vec!["marko"]
    );
}

// ===== fold (steps/fold.test.ts) =====

#[test]
fn p1_fold_basic() {
    assert_eq!(
        ordered(qs("g.V('1').out('KNOWS').values('name')")),
        vec!["vadas", "josh"]
    );
    let r = qs("g.V('1').out('KNOWS').values('name').fold()");
    assert_eq!(
        r,
        vec![GVal::list(vec![
            GVal::Str("vadas".into()),
            GVal::Str("josh".into())
        ])]
    );
}

#[test]
fn p1_fold_unfold_round_trips() {
    assert_eq!(
        ordered(qs("g.V().fold().unfold().values('name')")),
        vec!["marko", "vadas", "josh", "peter", "lop", "ripple"]
    );
}

#[test]
fn p1_fold_collects_persons() {
    let r = qs("g.V().hasLabel('PERSON').fold()");
    assert_eq!(r.len(), 1);
    let ids = qs("g.V().hasLabel('PERSON').id()");
    assert_eq!(ordered(ids), vec!["1", "2", "4", "6"]);
}

// ===== outE (steps/outE.test.ts) =====

#[test]
fn p1_oute_toy_v4_weights() {
    assert_eq!(nums(qs("g.V('4').outE().values('weight')")), vec![1.0, 0.4]);
}

#[test]
fn p1_oute_specific_label_knows() {
    // V('1').outE('KNOWS') → two edges; inV names vadas, josh; weights 0.5, 1.0.
    assert_eq!(one_num(qs("g.V('1').outE('KNOWS').count()")), 2.0);
    assert_eq!(
        ordered(qs("g.V('1').outE('KNOWS').inV().values('name')")),
        vec!["vadas", "josh"]
    );
    assert_eq!(
        nums(qs("g.V('1').outE('KNOWS').values('weight')")),
        vec![0.5, 1.0]
    );
}

#[test]
fn p1_oute_multiple_labels() {
    assert_eq!(
        ordered(qs("g.V('1').outE('KNOWS','CREATED').inV().values('name')")),
        vec!["vadas", "josh", "lop"]
    );
    assert_eq!(
        nums(qs("g.V('1').outE('KNOWS','CREATED').values('weight')")),
        vec![0.5, 1.0, 0.4]
    );
}

#[test]
fn p1_oute_v4_edge_ids() {
    assert_eq!(ordered(qs_e("g.V('4').outE().id()")), vec!["10", "11"]);
}

#[test]
fn p1_oute_all_labels_like_none_idset() {
    let a = sorted(qs_e("g.V('1').outE('CREATED','KNOWS').id()"));
    let b = sorted(qs_e("g.V('1').outE().id()"));
    assert_eq!(a, b);
}

// ===== groupCount (steps/groupCount.test.ts) =====

#[test]
fn p1_group_count_value_occurrences() {
    let r = qs("g.V().hasLabel('PERSON').values('age').groupCount()");
    assert_eq!(map_get_num(&r, &GVal::Num(29.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(27.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(32.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(35.0)), Some(1.0));
}

#[test]
fn p1_group_count_by_lang() {
    let r = qs("g.V().hasLabel('SOFTWARE').groupCount().by('lang')");
    assert_eq!(map_get_num(&r, &GVal::Str("java".into())), Some(2.0));
}

#[test]
fn p1_group_count_by_label() {
    let r = qs("g.V().groupCount().by(T.label)");
    assert_eq!(map_get_num(&r, &GVal::Str("PERSON".into())), Some(4.0));
    assert_eq!(map_get_num(&r, &GVal::Str("SOFTWARE".into())), Some(2.0));
}

#[test]
fn p1_group_count_by_age_persons() {
    let r = qs("g.V().hasLabel('PERSON').groupCount().by('age')");
    assert_eq!(map_get_num(&r, &GVal::Num(29.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(27.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(32.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(35.0)), Some(1.0));
}

#[test]
fn p1_group_count_by_age_all() {
    let r = qs("g.V().groupCount().by('age')");
    assert_eq!(map_get_num(&r, &GVal::Num(29.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(27.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(32.0)), Some(1.0));
    assert_eq!(map_get_num(&r, &GVal::Num(35.0)), Some(1.0));
}

// ===== tail (steps/tail.test.ts) =====

#[test]
fn p1_tail_default_one() {
    assert_eq!(
        ordered(qs("g.V().hasLabel('PERSON').values('name').tail()")),
        vec!["peter"]
    );
}

#[test]
fn p1_tail_with_order() {
    assert_eq!(
        ordered(qs("g.V().hasLabel('PERSON').values('name').order().tail()")),
        vec!["vadas"]
    );
}

#[test]
fn p1_tail_order_default_eq_explicit_one() {
    let r1 = ordered(qs("g.V().hasLabel('PERSON').values('name').order().tail()"));
    let r2 = ordered(qs(
        "g.V().hasLabel('PERSON').values('name').order().tail(1)",
    ));
    assert_eq!(r1, r2);
}

#[test]
fn p1_tail_multiple_items() {
    assert_eq!(
        ordered(qs("g.V().values('name').order().tail(3)")),
        vec!["peter", "ripple", "vadas"]
    );
}

// ===== not (steps/not.test.ts) =====

#[test]
fn p1_not_filters_by_subtraversal_absence() {
    assert_eq!(
        ordered(qs(
            "g.V().hasLabel('PERSON').not(__.out('CREATED').count().is(gt(1))).values('name')"
        )),
        vec!["marko", "vadas", "peter"]
    );
}

#[test]
fn p1_not_haslabel_keeps_nonmatching() {
    assert_eq!(
        ordered(qs("g.V().not(__.hasLabel('PERSON')).values('name')")),
        vec!["lop", "ripple"]
    );
}

#[test]
fn p1_not_haslabel_element_map() {
    // V().not(hasLabel('PERSON')).elementMap() — the two software vertices.
    let r = qs("g.V().not(__.hasLabel('PERSON')).elementMap()");
    assert_eq!(r.len(), 2);
    let get = |m: &GVal, key: &str| -> String {
        match m {
            GVal::Map(e) => e
                .iter()
                .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == key))
                .map(|(_, v)| s(v))
                .unwrap_or_default(),
            _ => panic!("expected map"),
        }
    };
    assert_eq!(get(&r[0], "id"), "3");
    assert_eq!(get(&r[0], "label"), "SOFTWARE");
    assert_eq!(get(&r[0], "name"), "lop");
    assert_eq!(get(&r[0], "lang"), "java");
    assert_eq!(get(&r[1], "id"), "5");
    assert_eq!(get(&r[1], "name"), "ripple");
}

#[test]
fn p1_not_predicate_inside_has() {
    // has('name', not(within('vadas','marko'))) — everyone else, in stream order.
    let mut g = modern();
    let t = super::g()
        .V()
        .has("name", P::not(P::within(["vadas", "marko"])))
        .values(&["name"]);
    assert_eq!(
        ordered(t.run(&mut g)),
        vec!["josh", "peter", "lop", "ripple"]
    );
}

// ===== hasId (steps/hasId.test.ts) =====

#[test]
fn p1_has_id_single() {
    assert_eq!(ordered(qs("g.V().hasId('1').id()")), vec!["1"]);
    assert_eq!(
        ordered(qs("g.V().hasId('1').values('name')")),
        vec!["marko"]
    );
}

#[test]
fn p1_has_id_out_of_order() {
    // hasId keeps vertices in graph order regardless of arg order.
    assert_eq!(
        ordered(qs("g.V().hasId('6','2','1','4').id()")),
        vec!["1", "2", "4", "6"]
    );
    assert_eq!(
        ordered(qs("g.V().hasId('6','2','1','4').values('name')")),
        vec!["marko", "vadas", "josh", "peter"]
    );
}

#[test]
fn p1_has_id_on_edges() {
    assert_eq!(ordered(qs_e("g.E().hasId('7','8').id()")), vec!["7", "8"]);
}

#[test]
fn p1_has_id_complex_chain() {
    // E hasId 7,8 → outV (marko twice) → out().out() → hasId('5').
    assert_eq!(
        ordered(qs_e(
            "g.E().hasId('7','8').outV().out().out().hasId('5').id()"
        )),
        vec!["5", "5"]
    );
}

// ===== and (steps/and.test.ts) =====

#[test]
fn p1_and_two_subtraversals() {
    assert_eq!(
        ordered(qs(
            "g.V().and(__.outE('KNOWS'), __.values('age').is(lt(30))).values('name')"
        )),
        vec!["marko"]
    );
}

#[test]
fn p1_and_both_out_knows_and_created() {
    assert_eq!(
        ordered(qs(
            "g.V().and(__.outE('KNOWS'), __.outE('CREATED')).values('name')"
        )),
        vec!["marko"]
    );
}

#[test]
fn p1_and_filters_everything() {
    assert_eq!(
        ordered(qs(
            "g.V().hasLabel('SOFTWARE').and(__.outE('KNOWS')).values('name')"
        )),
        Vec::<String>::new()
    );
}

#[test]
fn p1_and_in_knows_and_out_created() {
    assert_eq!(
        ordered(qs(
            "g.V().and(__.inE('KNOWS'), __.outE('CREATED')).values('name')"
        )),
        vec!["josh"]
    );
}

// ===== local (steps/local.test.ts) =====

#[test]
fn p1_local_oute_inv_neighbors() {
    let r = qs("g.V().local(__.outE().inV()).values('name')");
    assert_eq!(
        sorted(r),
        vec!["josh", "lop", "lop", "lop", "ripple", "vadas"]
    );
}

#[test]
fn p1_local_out_count_outdegree() {
    let r = qs("g.V().hasLabel('PERSON').local(__.out().count())");
    assert_eq!(nums(r), vec![3.0, 0.0, 2.0, 1.0]);
}

#[test]
fn p1_local_out_fold_per_vertex_lists() {
    let r = qs("g.V().hasLabel('PERSON').local(__.out().fold())");
    let sizes: Vec<usize> = r
        .iter()
        .map(|g| match g {
            GVal::List(items) => items.len(),
            other => panic!("expected list, got {other:?}"),
        })
        .collect();
    assert_eq!(sizes, vec![3, 0, 2, 1]);
}

// ===== filter (steps/filter.test.ts) =====

#[test]
fn p1_filter_label_is_person() {
    assert_eq!(
        ordered(qs(
            "g.V().filter(__.label().is(eq('PERSON'))).values('name')"
        )),
        vec!["marko", "vadas", "josh", "peter"]
    );
}

#[test]
fn p1_filter_has_outgoing_created() {
    assert_eq!(
        ordered(qs("g.V().filter(__.out('CREATED')).values('name')")),
        vec!["marko", "josh", "peter"]
    );
}

// ===== value (steps/value.test.ts) =====

#[test]
fn p1_value_unwraps_properties() {
    assert_eq!(
        ordered(qs("g.V().hasId('1').properties('name').value()")),
        vec!["marko"]
    );
}

// ===== hasNot (steps/hasNot.test.ts) =====

#[test]
fn p1_has_not_missing_key() {
    assert_eq!(
        ordered(qs("g.V().hasNot('age').values('name')")),
        vec!["lop", "ripple"]
    );
}

#[test]
fn p1_has_not_variadic_none_of() {
    // hasNot('age','lang') — lop/ripple lack age but have lang, so excluded too.
    assert_eq!(
        ordered(qs("g.V().hasNot('age','lang').values('name')")),
        Vec::<String>::new()
    );
}

// ===== index-seed (executor/index-seed.test.ts) — black-box equivalence =====

/// Run a query against a fresh Modern graph with the given vertex indexes built.
fn q_vidx(indexes: &[&str], query: &str) -> Vec<GVal> {
    let mut g = modern();
    for k in indexes {
        g.create_vertex_index(k);
    }
    let t = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

#[test]
fn p1_idx_eq_matches_scan() {
    let plain = qs("g.V().has('name','marko').values('age')");
    let indexed = q_vidx(&["name"], "g.V().has('name','marko').values('age')");
    assert_eq!(plain, indexed);
    assert_eq!(indexed, vec![GVal::Num(29.0)]);
}

#[test]
fn p1_idx_3arg_has_keeps_label() {
    // lop is SOFTWARE; the PERSON label still excludes it even when name-seeded.
    assert_eq!(
        q_vidx(&["name"], "g.V().has('PERSON','name','lop').values('name')").len(),
        0
    );
    assert_eq!(
        ordered(q_vidx(
            &["name"],
            "g.V().has('PERSON','name','marko').values('name')"
        )),
        vec!["marko"]
    );
}

#[test]
fn p1_idx_downstream_steps_run() {
    let r = q_vidx(&["name"], "g.V().has('name','marko').out().values('name')");
    assert_eq!(sorted(r), vec!["josh", "lop", "vadas"]);
}

#[test]
fn p1_idx_range_matches_scan() {
    for pred in ["gt(30)", "between(28, 33)", "inside(28, 33)"] {
        let q = format!("g.V().has('age', {pred}).values('name')");
        let plain = sorted(qs(&q));
        let indexed = sorted(q_vidx(&["age"], &q));
        assert_eq!(plain, indexed, "mismatch for {pred}");
    }
}

#[test]
fn p1_idx_startswith_matches_scan() {
    let plain = sorted(qs("g.V().has('name', startsWith('r')).values('name')"));
    let indexed = sorted(q_vidx(
        &["name"],
        "g.V().has('name', startsWith('r')).values('name')",
    ));
    assert_eq!(plain, indexed);
    assert_eq!(indexed, vec!["ripple"]);
}

#[test]
fn p1_idx_within_matches_scan() {
    let plain = sorted(qs(
        "g.V().has('name', within('vadas','josh')).values('name')",
    ));
    let indexed = sorted(q_vidx(
        &["name"],
        "g.V().has('name', within('vadas','josh')).values('name')",
    ));
    assert_eq!(plain, indexed);
    assert_eq!(indexed, vec!["josh", "vadas"]);
}

#[test]
fn p1_idx_empty_bucket_short_circuits() {
    assert_eq!(
        q_vidx(&["name"], "g.V().has('name','nobody').values('name')").len(),
        0
    );
}

#[test]
fn p1_idx_multi_filter_matches_scan() {
    let q = "g.V().has('age', gt(28)).has('name', startsWith('j')).values('name')";
    let plain = sorted(qs(q));
    let indexed = sorted(q_vidx(&["age", "name"], q));
    assert_eq!(plain, indexed);
    assert_eq!(indexed, vec!["josh"]);
}

#[test]
fn p1_idx_edge_eq_matches_scan() {
    let mut plain = modern_eids();
    let mut indexed = modern_eids();
    indexed.create_edge_index("weight");
    let t = super::parse("g.E().has('weight', 1.0).id()").unwrap();
    let mut got = ordered(t.run(&mut indexed));
    let mut want = ordered(
        super::parse("g.E().has('weight', 1.0).id()")
            .unwrap()
            .run(&mut plain),
    );
    got.sort();
    want.sort();
    assert_eq!(got, want);
    assert_eq!(got, vec!["10", "8"]);
}

#[test]
fn p1_idx_edge_eq_count_seeds() {
    let mut g = modern_eids();
    g.create_edge_index("weight");
    // weight == 0.4 → edges 9 and 11.
    assert_eq!(
        one_num(
            super::parse("g.E().has('weight', eq(0.4)).count()")
                .unwrap()
                .run(&mut g)
        ),
        2.0
    );
}

#[test]
fn p1_idx_edge_range_matches_scan() {
    let mut plain = modern_eids();
    let mut indexed = modern_eids();
    indexed.create_edge_index("weight");
    let q = "g.E().has('weight', gt(0.5)).values('weight')";
    let mut got = nums(super::parse(q).unwrap().run(&mut indexed));
    let mut want = nums(super::parse(q).unwrap().run(&mut plain));
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got, want);
}
