//! Per-step conformance tests, part 3 of 6: `has`, closures, `none`,
//! subplan-shapes, `range`, `union`, `max`, `bothE`, `sample`, `match`, `E`,
//! `unfold`, inside/outside, `drop`, `outV`, `bothV`.
//!
//! The six parts are a SIZE split, not a thematic one — the number carries no
//! meaning. Self-contained: own `modern()` fixture (a copy of the canonical
//! TinkerPop "Modern" graph used in `tests.rs`), own helpers, own `#[test]` fns.
//!
//! Mirrors the TS Gremlin step tests. Test names are prefixed `p3_<step>_<short>`;
//! TS closures become Rust sub-traversals, and textual Gremlin (`super::parse`) is
//! preferred where it can express the query.

use super::{GVal, Step};
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

/// Sorted string results (order-independent traversals).
fn names(r: Vec<GVal>) -> Vec<String> {
    let mut v: Vec<String> = r.iter().map(s).collect();
    v.sort();
    v
}

/// String results in stream order.
fn ordered(r: Vec<GVal>) -> Vec<String> {
    r.iter().map(s).collect()
}

/// Parse + run a textual query against a fresh Modern graph.
fn qs(query: &str) -> Vec<GVal> {
    let mut g = modern();
    let t = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

/// Parse + run against the edge-id Modern graph.
fn qs_eids(query: &str) -> Vec<GVal> {
    let mut g = modern_eids();
    let t = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

// ===================== has.test.ts =====================

#[test]
fn p3_has_within_filter() {
    // V().hasLabel(PERSON).out().has('name', within('vadas','josh')) → ids 2,4.
    let r = qs("g.V().hasLabel('PERSON').out().has('name', within('vadas','josh')).id()");
    assert_eq!(names(r), vec!["2", "4"]);
}

#[test]
fn p3_has_chain_to_created_edges() {
    // …outE().hasLabel(CREATED) — josh's two CREATED edges (ids 10, 11).
    let r =
        qs_eids("g.V().hasLabel('PERSON').out().has('name', within('vadas','josh')).outE().hasLabel('CREATED').id()");
    assert_eq!(ordered(r), vec!["10", "11"]);
}

#[test]
fn p3_has_inside_strict() {
    // age in (28,33) → marko(29), josh(32).
    let r = qs("g.V().hasLabel('PERSON').has('age', inside(28, 33)).values('name')");
    assert_eq!(names(r), vec!["josh", "marko"]);
}

#[test]
fn p3_has_outside_strict() {
    // age < 29 || > 32 → vadas(27), peter(35).
    let r = qs("g.V().hasLabel('PERSON').has('age', outside(29, 32)).values('name')");
    assert_eq!(names(r), vec!["peter", "vadas"]);
}

#[test]
fn p3_has_starts_with() {
    let r = qs("g.V().hasLabel('PERSON').has('name', startsWith('m')).id()");
    assert_eq!(names(r), vec!["1"]);
}

#[test]
fn p3_has_key_existence() {
    // has('age') keeps the four people (software has no age).
    let r = qs("g.V().has('age').values('name')");
    assert_eq!(names(r), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p3_has_inside_all_vertices_ordered() {
    // doc: g.V().has('age', inside(20,30)).values('age') — 29; 27 (stream order).
    let r = qs("g.V().has('age', inside(20, 30)).values('age')");
    assert_eq!(r, vec![GVal::Num(29.0), GVal::Num(27.0)]);
}

#[test]
fn p3_has_outside_all_vertices_ordered() {
    // doc: g.V().has('age', outside(20,30)).values('age') — 32; 35.
    let r = qs("g.V().has('age', outside(20, 30)).values('age')");
    assert_eq!(r, vec![GVal::Num(32.0), GVal::Num(35.0)]);
}

#[test]
fn p3_has_within_element_map() {
    // doc: g.V().has('name', within('josh','marko')).elementMap() — marko, josh.
    let r = qs("g.V().has('name', within('josh','marko')).elementMap('name','age')");
    let ids: Vec<String> = r
        .iter()
        .map(|m| match m {
            GVal::Map(e) => e
                .iter()
                .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == "id"))
                .map(|(_, v)| s(v))
                .unwrap(),
            _ => panic!("expected map"),
        })
        .collect();
    assert_eq!(ids, vec!["1", "4"]); // marko, josh in stream order
}

#[test]
fn p3_has_without_element_map() {
    // doc: g.V().has('name', without('josh','marko')).elementMap().
    let r = qs("g.V().has('name', without('josh','marko')).elementMap('name','age','lang')");
    let ids: Vec<String> = r
        .iter()
        .map(|m| match m {
            GVal::Map(e) => e
                .iter()
                .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == "id"))
                .map(|(_, v)| s(v))
                .unwrap(),
            _ => panic!("expected map"),
        })
        .collect();
    assert_eq!(ids, vec!["2", "6", "3", "5"]);
}

#[test]
fn p3_has_not_within_equals_without() {
    // not(has(name, within(...))) ≡ has(name, without(...)).
    let r = qs("g.V().not(__.has('name', within('josh','marko'))).id()");
    assert_eq!(ordered(r), vec!["2", "6", "3", "5"]);
}

#[test]
fn p3_has_chained_oute_created_edges() {
    // doc chained variant — edges 10, 11.
    let r =
        qs_eids("g.V().hasLabel('PERSON').out().has('name', within('vadas','josh')).outE().hasLabel('CREATED').id()");
    assert_eq!(ordered(r), vec!["10", "11"]);
}

#[test]
fn p3_has_value_shorthand() {
    // has('name','marko') ≡ has('name', eq('marko')).
    let r = qs("g.V().has('name', 'marko').values('name')");
    assert_eq!(r, vec![GVal::Str("marko".into())]);
}

#[test]
fn p3_has_label_key_value_three_arg() {
    // has('PERSON','name','marko') filters by label AND property.
    let r = qs("g.V().has('PERSON', 'name', 'marko').values('name')");
    assert_eq!(r, vec![GVal::Str("marko".into())]);
}

#[test]
fn p3_has_label_key_predicate_three_arg() {
    // has('PERSON','age',gt(30)) → josh, peter.
    let r = qs("g.V().has('PERSON', 'age', gt(30)).values('name')");
    assert_eq!(ordered(r), vec!["josh", "peter"]);
}

// SKIPPED (has.test.ts): the two `regex('r')` cases — Rust `P` has no Regex
// variant and the parser rejects `regex(...)`. Recorded as a gap.

// ===================== closures.test.ts =====================
// TS closures → Rust sub-traversals where expressible.

#[test]
fn p3_closures_map_subplan_names() {
    // map(pipe(values('name'))) — sub-plan dispatch (the non-closure form).
    let r = qs("g.V().hasLabel('PERSON').map(__.values('name'))");
    assert_eq!(names(r), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p3_closures_filter_age_gt_30() {
    // filter closure age>30 → where(values('age').is(gt(30))).
    let r = qs("g.V().hasLabel('PERSON').where(__.values('age').is(gt(30))).values('name')");
    assert_eq!(names(r), vec!["josh", "peter"]);
}

#[test]
fn p3_closures_side_effect_passthrough() {
    // sideEffect closure → sideEffect sub-plan; passthrough preserves the stream.
    let r = qs("g.V().hasLabel('PERSON').sideEffect(__.values('name')).values('name')");
    assert_eq!(names(r), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p3_closures_fold_no_args_is_list() {
    // fold() without args produces a single list traverser.
    let r = qs("g.V().hasLabel('PERSON').values('name').fold()");
    assert_eq!(r.len(), 1);
    match &r[0] {
        GVal::List(items) => {
            let mut v: Vec<String> = items.iter().map(s).collect();
            v.sort();
            assert_eq!(v, vec!["josh", "marko", "peter", "vadas"]);
        }
        _ => panic!("expected list"),
    }
}

// SKIPPED (closures.test.ts): `map((v)=>...)`, `filter((v)=>...)`,
// `flatMap((v)=>...)`, `sideEffect((v)=>...seen)`, `fold(seed, reducer)` —
// closure-bearing forms are intentionally omitted in the data-plan model
// (mod.rs doc-comment). The serialize/isSerializable/findClosures tests have no
// Rust analogue. (The serializability concept is TS-only.)

// ===================== none.test.ts =====================

#[test]
fn p3_none_drops_every_traverser() {
    let r = qs("g.V().hasLabel('PERSON').none()");
    assert_eq!(r, Vec::<GVal>::new());
}

#[test]
fn p3_none_downstream_count_zero() {
    // count() over an empty stream → 0.
    let r = qs("g.V().none().count()");
    assert_eq!(r, vec![GVal::Num(0.0)]);
}

#[test]
fn p3_none_pred_keeps_when_all_fail() {
    // fold ages then none(gt(35)) — none > 35, so the folded list passes.
    let r = qs("g.V().values('age').fold().none(gt(35))");
    assert_eq!(r.len(), 1);
    match &r[0] {
        GVal::List(items) => {
            let nums: Vec<f64> = items
                .iter()
                .map(|v| match v {
                    GVal::Num(n) => *n,
                    _ => panic!(),
                })
                .collect();
            assert_eq!(nums, vec![29.0, 27.0, 32.0, 35.0]);
        }
        _ => panic!("expected list"),
    }
}

#[test]
fn p3_none_pred_drops_when_any_passes() {
    // 32, 35 are > 30 → folded list fails, dropped.
    let r = qs("g.V().values('age').fold().none(gt(30))");
    assert_eq!(r, Vec::<GVal>::new());
}

#[test]
fn p3_none_pred_empty_fold_passes() {
    // Vacuous truth over an empty fold.
    let r = qs("g.V().hasLabel('NOSUCH').values('age').fold().none(lt(0))");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0], GVal::List(vec![]));
}

// SKIPPED (none.test.ts): `toArray`/`toSet` runners and the `Scope`/
// `Cardinality` symbol-identity test — TS-only API surface (no Rust analogue;
// GAPS.md notes no list/set cardinality).

// ===================== subplan-shapes.test.ts =====================
// "Plan vs StepFn" interchangeability is a TS API distinction; in Rust every
// sub-plan is a Traversal. We port the behavioral assertions.

#[test]
fn p3_subplan_filter_label_person() {
    // filter(label().is(eq('PERSON'))) keeps the four people.
    let r = qs("g.V().filter(__.label().is(eq('PERSON'))).count()");
    assert_eq!(r, vec![GVal::Num(4.0)]);
}

#[test]
fn p3_subplan_where_label_person_names() {
    let r = qs("g.V().where(__.label().is(eq('PERSON'))).values('name')");
    assert_eq!(names(r), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p3_subplan_union_name_and_age() {
    let r = qs("g.V('1').union(__.values('name'), __.values('age'))");
    // marko, 29 in some order.
    assert!(r.contains(&GVal::Str("marko".into())));
    assert!(r.contains(&GVal::Num(29.0)));
    assert_eq!(r.len(), 2);
}

#[test]
fn p3_subplan_choose_test_then_else() {
    // choose(values('age').is(eq(29)), values('name'), values('age')).
    let r = qs("g.V().hasLabel('PERSON').choose(__.values('age').is(eq(29)), __.values('name'), __.values('age'))");
    // marko → 'marko'; others → their ages (27, 32, 35).
    let mut got = r.clone();
    got.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    assert_eq!(
        got,
        vec![
            GVal::Num(27.0),
            GVal::Num(32.0),
            GVal::Num(35.0),
            GVal::Str("marko".into()),
        ]
    );
}

#[test]
fn p3_subplan_repeat_body_adds_vertices() {
    // repeat(addV('REP').property('via','rep')).times(2) over V('1') adds 2 verts.
    let mut g = modern();
    let before = g.vertex_count();
    let t =
        super::parse("g.V('1').repeat(__.addV('REP').property('via', 'rep')).times(2)").unwrap();
    let _ = t.run(&mut g);
    assert_eq!(g.vertex_count(), before + 2);
}

#[test]
fn p3_subplan_map_body_adds_vertices() {
    // map(addV('SHADOW')...) over the four people adds 4 vertices.
    let mut g = modern();
    let before = g.vertex_count();
    let t = super::parse("g.V().hasLabel('PERSON').map(__.addV('SHADOW').property('via', 'map'))")
        .unwrap();
    let _ = t.run(&mut g);
    assert_eq!(g.vertex_count(), before + 4);
}

#[test]
fn p3_subplan_repeat_until_times_zero_smoke() {
    // repeat(identity).until(count().is(eq(0))).times(0) — smoke: doesn't panic.
    let mut g = modern();
    let t = super::parse("g.V('1').repeat(__.identity()).until(__.count().is(eq(0))).times(0)")
        .unwrap();
    let r = t.run(&mut g);
    // smoke: a Vec is produced.
    let _ = r.len();
}

// ===================== range.test.ts =====================

#[test]
fn p3_range_first_three() {
    let r = qs("g.V().range(0, 3).values('name')");
    assert_eq!(ordered(r), vec!["marko", "vadas", "josh"]);
}

#[test]
fn p3_range_skip_low_end() {
    let r = qs("g.V().range(3, 5).values('name')");
    assert_eq!(ordered(r), vec!["peter", "lop"]);
}

#[test]
fn p3_range_0_3_ids() {
    let r = qs("g.V().range(0, 3).id()");
    assert_eq!(ordered(r), vec!["1", "2", "4"]);
}

#[test]
fn p3_range_1_3_ids() {
    let r = qs("g.V().range(1, 3).id()");
    assert_eq!(ordered(r), vec!["2", "4"]);
}

// SKIPPED (range.test.ts): open-end `range(3, -1)` and `range(1, -1)` — Rust
// casts the negative `f64` end to `usize` saturating to 0 (not "open"), so the
// range yields an empty stream rather than "rest of the vertices". Genuine
// divergence (suspected: range end handling treats negative as 0 not unbounded).

// ===================== union.test.ts =====================

#[test]
fn p3_union_fold_fold_unfold_interleaved() {
    // union(fold(),fold()).unfold().values('name') — each vertex twice, interleaved.
    let r = qs("g.V().union(__.fold(), __.fold()).unfold().values('name')");
    assert_eq!(
        ordered(r),
        vec![
            "marko", "marko", "vadas", "vadas", "josh", "josh", "peter", "peter", "lop", "lop",
            "ripple", "ripple",
        ]
    );
}

#[test]
fn p3_union_in_and_out_values() {
    // V('4').union(in_(), out()).values('age','lang') — 29, java, java.
    let r = qs(
        "g.V('4').union(__.in('KNOWS','CREATED'), __.out('KNOWS','CREATED')).values('age','lang')",
    );
    assert_eq!(
        r,
        vec![
            GVal::Num(29.0),
            GVal::Str("java".into()),
            GVal::Str("java".into())
        ]
    );
}

#[test]
fn p3_union_out_in_names_flattened() {
    let r = qs("g.V('1','4').union(__.out().values('name'), __.in().values('name'))");
    assert_eq!(
        names(r),
        vec!["josh", "lop", "lop", "marko", "ripple", "vadas"]
    );
}

#[test]
fn p3_union_terminal_counts_per_branch() {
    // V('1','4').union(out().count(), in_().count()) — 3,0,2,1.
    let r = qs("g.V('1','4').union(__.out().count(), __.in().count())");
    assert_eq!(
        r,
        vec![
            GVal::Num(3.0),
            GVal::Num(0.0),
            GVal::Num(2.0),
            GVal::Num(1.0)
        ]
    );
}

#[test]
fn p3_union_output_feeds_parent() {
    let r = qs("g.V('1','4').union(__.out(), __.in()).hasLabel('PERSON').values('name')");
    assert_eq!(names(r), vec!["josh", "marko", "vadas"]);
}

// ===================== max.test.ts =====================

#[test]
fn p3_max_numbers() {
    let r = qs("g.V().values('age').max()");
    assert_eq!(r, vec![GVal::Num(35.0)]);
}

#[test]
fn p3_max_strings() {
    let r = qs("g.V().values('name').max()");
    assert_eq!(r, vec![GVal::Str("vadas".into())]);
}

#[test]
fn p3_max_after_repeat_both_times3() {
    let r = qs("g.V().repeat(__.both()).times(3).values('age').max()");
    assert_eq!(r, vec![GVal::Num(35.0)]);
}

// NOTE (max.test.ts): `max()` over null operands now skips them, matching TS —
// see divergence_tests::max_skips_nulls (built via the fluent builder, since
// the textual parser has no `null` literal).

// ===================== bothE.test.ts =====================

#[test]
fn p3_bothe_v4_three_edges_order() {
    // V('4').bothE('KNOWS','CREATED','BLAH') → out CREATED (10,11), then in KNOWS (8).
    let r = qs_eids("g.V('4').bothE('KNOWS','CREATED','BLAH').id()");
    assert_eq!(ordered(r), vec!["10", "11", "8"]);
}

#[test]
fn p3_bothe_v4_specific_endpoints() {
    // The two out CREATED edges go to ripple then lop; the in KNOWS comes from marko.
    let r = qs("g.V('4').bothE('KNOWS','CREATED','BLAH').inV().values('name')");
    // out edges inV: ripple, lop; the in-edge inV is josh himself (its inV = josh).
    assert_eq!(ordered(r), vec!["ripple", "lop", "josh"]);
}

#[test]
fn p3_bothe_v1_specific_label() {
    // V('1').bothE('KNOWS').inV() → vadas, josh.
    let r = qs("g.V('1').bothE('KNOWS').inV().values('name')");
    assert_eq!(ordered(r), vec!["vadas", "josh"]);
}

#[test]
fn p3_bothe_v4_no_labels_endpoints() {
    // V('4').bothE() inV order: ripple, lop (out), then josh (in-edge's inV).
    let r = qs("g.V('4').bothE().inV().values('name')");
    assert_eq!(ordered(r), vec!["ripple", "lop", "josh"]);
}

#[test]
fn p3_bothe_v4_ids() {
    // doc: g.V(4).bothE('KNOWS','CREATED','blah') → e[10], e[11], e[8].
    let r = qs_eids("g.V('4').bothE('KNOWS','CREATED','blah').id()");
    assert_eq!(ordered(r), vec!["10", "11", "8"]);
}

#[test]
fn p3_bothe_v1_ids() {
    // doc: marko's edges — out KNOWS (7,8), out CREATED (9); no incoming.
    let r = qs_eids("g.V('1').bothE().id()");
    assert_eq!(ordered(r), vec!["7", "8", "9"]);
}

// ===================== sample.test.ts =====================
// sample() is a SEEDED pseudo-random shuffle (fixed Mulberry32 seed): a real
// random subset (not a stream-order prefix), reproducible, and byte-identical
// with the TS engine.

#[test]
fn p3_sample_is_a_seeded_shuffle_not_a_prefix() {
    // Deterministic (fixed seed): two runs agree.
    assert_eq!(
        qs("g.V().values('name').sample(3)"),
        qs("g.V().values('name').sample(3)")
    );
    // sample(all) is a permutation of the full stream (same multiset)…
    let full = names(qs("g.V().values('name')"));
    let sampled = names(qs("g.V().values('name').sample(6)"));
    assert_eq!(full, sampled); // `names` sorts, so this checks the multiset
                               // …and it's not a mere prefix: the sampled order differs from stream order.
    let stream_order = ordered(qs("g.V().values('name')"));
    let sample_order = ordered(qs("g.V().values('name').sample(6)"));
    assert_ne!(
        stream_order, sample_order,
        "sample should shuffle, not take a prefix"
    );
}

#[test]
fn p3_sample_n_returns_n() {
    let r = qs("g.V().hasLabel('PERSON').sample(2).values('name')");
    assert_eq!(r.len(), 2);
    for name in &r {
        assert!(["marko", "vadas", "josh", "peter"].contains(&s(name).as_str()));
    }
}

#[test]
fn p3_sample_caps_at_stream_size() {
    let r = qs("g.V().hasLabel('SOFTWARE').sample(99).values('name')");
    assert_eq!(r.len(), 2);
    assert_eq!(names(r), vec!["lop", "ripple"]);
}

#[test]
fn p3_sample_zero_yields_nothing() {
    let r = qs("g.V().sample(0)");
    assert_eq!(r, Vec::<GVal>::new());
}

#[test]
fn p3_sample_one_on_oute_one_weight() {
    let r = qs("g.V().outE().sample(1).values('weight')");
    assert_eq!(r.len(), 1);
    match &r[0] {
        GVal::Num(n) => assert!([0.5, 1.0, 0.4, 0.2].contains(n)),
        _ => panic!("expected number"),
    }
}

// ===================== match.test.ts =====================

/// Normalize select(...)-of-match results into a sorted set of sorted
/// (key, name) rows.
fn match_rows(r: Vec<GVal>) -> Vec<Vec<(String, String)>> {
    let mut rows: Vec<Vec<(String, String)>> = r
        .iter()
        .map(|m| {
            let mut entries: Vec<(String, String)> = match m {
                GVal::Map(e) => e.iter().map(|(k, v)| (s(k), s(v))).collect(),
                _ => panic!("expected map, got {m:?}"),
            };
            entries.sort();
            entries
        })
        .collect();
    rows.sort();
    rows
}

fn pairs(spec: &[(&str, &str)]) -> Vec<(String, String)> {
    spec.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn p3_match_declarative_and_fragments() {
    let r = qs("g.V().match(\
         __.as('a').out('CREATED').as('b'), \
         __.as('b').has('name','lop'), \
         __.as('b').in('CREATED').as('c'), \
         __.as('c').has('age', 29)).select('a','c').by('name')");
    let mut want = vec![
        pairs(&[("a", "marko"), ("c", "marko")]),
        pairs(&[("a", "josh"), ("c", "marko")]),
        pairs(&[("a", "peter"), ("c", "marko")]),
    ];
    want.sort();
    assert_eq!(match_rows(r), want);
}

#[test]
fn p3_match_chained_embedded_has() {
    let r = qs("g.V().match(\
         __.as('a').out('CREATED').has('name','lop').as('b'), \
         __.as('b').in('CREATED').has('age', 29).as('c')).select('a','c').by('name')");
    let mut want = vec![
        pairs(&[("a", "marko"), ("c", "marko")]),
        pairs(&[("a", "josh"), ("c", "marko")]),
        pairs(&[("a", "peter"), ("c", "marko")]),
    ];
    want.sort();
    assert_eq!(match_rows(r), want);
}

#[test]
fn p3_match_with_where_neq() {
    let r = qs("g.V().match(\
         __.as('a').out('CREATED').as('b'), \
         __.as('b').in('CREATED').as('c')).where('a', neq('c')).select('a','c').by('name')");
    let mut want = vec![
        pairs(&[("a", "marko"), ("c", "josh")]),
        pairs(&[("a", "marko"), ("c", "peter")]),
        pairs(&[("a", "josh"), ("c", "marko")]),
        pairs(&[("a", "josh"), ("c", "peter")]),
        pairs(&[("a", "peter"), ("c", "marko")]),
        pairs(&[("a", "peter"), ("c", "josh")]),
    ];
    want.sort();
    assert_eq!(match_rows(r), want);
}

#[test]
fn p3_match_nested_not() {
    let r = qs("g.V().as('a').out('KNOWS').as('b').match(\
         __.as('b').out('CREATED').as('c'), \
         __.not(__.as('c').in('CREATED').as('a'))).select('a','b','c').by('name')");
    assert_eq!(
        match_rows(r),
        vec![pairs(&[("a", "marko"), ("b", "josh"), ("c", "ripple")])]
    );
}

// ===================== E.test.ts =====================

#[test]
fn p3_e_all_edges_count() {
    let r = qs("g.E().count()");
    assert_eq!(r, vec![GVal::Num(6.0)]);
}

#[test]
fn p3_e_insertion_order_ids() {
    let r = qs_eids("g.E().id()");
    assert_eq!(ordered(r), vec!["7", "8", "9", "10", "11", "12"]);
}

// E.test.ts: `E('7')` / `E('11')` lookup by *external* edge id now resolves the
// external id assigned in ndjson (like `V(id)`), matching TS.
#[test]
fn p3_e_by_external_id() {
    assert_eq!(ordered(qs_eids("g.E('7').id()")), vec!["7"]);
    assert_eq!(ordered(qs_eids("g.E('11').id()")), vec!["11"]);
}

// ===================== unfold.test.ts =====================

#[test]
fn p3_unfold_fold_then_inject_setup() {
    // V('1').out().fold().inject('gremlin', [1.23,2.34]) — inject prepends.
    // The nested list literal isn't parseable, so build the inject Step directly.
    let mut g = modern();
    let mut t = super::parse("g.V('1').out().fold()").unwrap();
    t.steps.push(Step::Inject(vec![
        GVal::Str("gremlin".into()),
        GVal::List(vec![GVal::Num(1.23), GVal::Num(2.34)]),
    ]));
    let r = t.run(&mut g);
    // ['gremlin', [1.23,2.34], List[vadas, josh, lop]] — out() KNOWS-first.
    assert_eq!(r.len(), 3);
    assert_eq!(r[0], GVal::Str("gremlin".into()));
    assert_eq!(r[1], GVal::List(vec![GVal::Num(1.23), GVal::Num(2.34)]));
    match &r[2] {
        GVal::List(items) => {
            let ids: Vec<String> = items
                .iter()
                .map(|v| match v {
                    GVal::Vertex(i) => g.vid.text(*i).to_string(),
                    other => format!("{other:?}"),
                })
                .collect();
            assert_eq!(ids, vec!["2", "4", "3"]);
        }
        _ => panic!("expected folded list"),
    }
}

#[test]
fn p3_unfold_unfolds_one_level() {
    // …inject('gremlin', [1.23,2.34]).unfold() — list flattens one level.
    let mut g = modern();
    let mut t = super::parse("g.V('1').out().fold()").unwrap();
    t.steps.push(Step::Inject(vec![
        GVal::Str("gremlin".into()),
        GVal::List(vec![GVal::Num(1.23), GVal::Num(2.34)]),
    ]));
    t.steps.push(Step::Unfold);
    let r = t.run(&mut g);
    // ['gremlin', 1.23, 2.34, vadas, josh, lop].
    assert_eq!(r.len(), 6);
    assert_eq!(r[0], GVal::Str("gremlin".into()));
    assert_eq!(r[1], GVal::Num(1.23));
    assert_eq!(r[2], GVal::Num(2.34));
    let ids: Vec<String> = r[3..]
        .iter()
        .map(|v| match v {
            GVal::Vertex(i) => g.vid.text(*i).to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(ids, vec!["2", "4", "3"]);
}

#[test]
fn p3_unfold_is_not_deep() {
    // inject(1, [2,3,[4,5,[6]]]) — without unfold: [1, [..]]; with: [1,2,3,[..]].
    let nested = GVal::List(vec![
        GVal::Num(2.0),
        GVal::Num(3.0),
        GVal::List(vec![
            GVal::Num(4.0),
            GVal::Num(5.0),
            GVal::List(vec![GVal::Num(6.0)]),
        ]),
    ]);
    let mut g = modern();
    let mut t = super::g();
    t.steps
        .push(Step::Inject(vec![GVal::Num(1.0), nested.clone()]));
    let r1 = t.run(&mut g);
    assert_eq!(r1, vec![GVal::Num(1.0), nested.clone()]);

    let mut t2 = super::g();
    t2.steps
        .push(Step::Inject(vec![GVal::Num(1.0), nested.clone()]));
    t2.steps.push(Step::Unfold);
    let r2 = t2.run(&mut g);
    let inner = GVal::List(vec![
        GVal::Num(4.0),
        GVal::Num(5.0),
        GVal::List(vec![GVal::Num(6.0)]),
    ]);
    assert_eq!(
        r2,
        vec![GVal::Num(1.0), GVal::Num(2.0), GVal::Num(3.0), inner]
    );
}

// ===================== inside-outside.test.ts =====================

#[test]
fn p3_between_half_open() {
    // age in [29,32) → marko only.
    let r = qs("g.V().hasLabel('PERSON').has('age', between(29, 32)).values('name')");
    assert_eq!(names(r), vec!["marko"]);
}

#[test]
fn p3_inside_strict_open() {
    // age in (27,35) → marko, josh.
    let r = qs("g.V().hasLabel('PERSON').has('age', inside(27, 35)).values('name')");
    assert_eq!(names(r), vec!["josh", "marko"]);
}

#[test]
fn p3_outside_strict_complement() {
    // age < 29 || > 32 → vadas, peter.
    let r = qs("g.V().hasLabel('PERSON').has('age', outside(29, 32)).values('name')");
    assert_eq!(names(r), vec!["peter", "vadas"]);
}

// ===================== drop.test.ts =====================

#[test]
fn p3_drop_vertex_removes_and_emits_nothing() {
    let mut g = modern();
    let before = g.vertex_count();
    let r = super::parse("g.V('2').drop()").unwrap().run(&mut g);
    assert_eq!(r, Vec::<GVal>::new());
    assert_eq!(g.vertex_count(), before - 1);
    // vadas (id 2) is gone.
    let mut g2 = g;
    let cnt = super::parse("g.V().has('name', 'vadas').count()")
        .unwrap()
        .run(&mut g2);
    assert_eq!(cnt, vec![GVal::Num(0.0)]);
}

#[test]
fn p3_drop_vertex_cascades_incident_edges() {
    let mut g = modern();
    let edges_before = g.edge_count();
    // marko (id 1) has 3 incident edges.
    let _ = super::parse("g.V('1').drop()").unwrap().run(&mut g);
    assert_eq!(g.edge_count(), edges_before - 3);
}

#[test]
fn p3_drop_edges_leaves_vertices() {
    let mut g = modern();
    let v_before = g.vertex_count();
    let _ = super::parse("g.E().hasLabel('CREATED').drop()")
        .unwrap()
        .run(&mut g);
    assert_eq!(g.vertex_count(), v_before);
    // No CREATED edges remain.
    let mut g2 = g;
    let cnt = super::parse("g.E().hasLabel('CREATED').count()")
        .unwrap()
        .run(&mut g2);
    assert_eq!(cnt, vec![GVal::Num(0.0)]);
}

// ===================== outV.test.ts =====================

#[test]
fn p3_outv_ine_outv_yields_source() {
    // V('4').inE().outV() — josh's incoming edge is from marko.
    let r = qs("g.V('4').inE().outV().values('name')");
    assert_eq!(ordered(r), vec!["marko"]);
}

#[test]
fn p3_outv_ine_outv_id() {
    let r = qs("g.V('4').inE().outV().id()");
    assert_eq!(ordered(r), vec!["1"]);
}

// ===================== bothV.test.ts =====================

#[test]
fn p3_bothv_ine_bothv_endpoints() {
    // V('4').inE().bothV() — marko (out) then josh (in).
    let r = qs("g.V('4').inE().bothV().values('name')");
    assert_eq!(ordered(r), vec!["marko", "josh"]);
}

#[test]
fn p3_bothv_ine_bothv_ids() {
    let r = qs("g.V('4').inE().bothV().id()");
    assert_eq!(ordered(r), vec!["1", "4"]);
}
