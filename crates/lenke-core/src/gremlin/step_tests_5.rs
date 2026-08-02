//! Per-step conformance tests, part 5 of 6: `values`, `sum`, `dedup`,
//! path-tracking, `properties`, `inject`, `valueMap`, `propertyMap`, `loops`,
//! `branch`, `shortestPath`, `id`, `as`, `inV`.
//!
//! The six parts are a SIZE split, not a thematic one — the number carries no
//! meaning.
//!
//! Each test translates a TS fluent traversal into the Rust fluent/text builder,
//! runs it over a local `modern()` graph, and asserts the equivalent `GVal` shape.
//! Intentional divergences (per the engine's documented behavior) are respected:
//!   - valueMap/propertyMap flat-value behavior already matches the TS v2 impl.
//!   - dedupe() only keys on its first by() modulator and By::Key on a Map is a
//!     no-op, so dedupe(a,b)/dedupe(a) over select-maps are not expressible.
//!   - global sum/min/max over an all-null stream returns [] (not [null]).
//!   - math(), branch(), and closure map() steps don't exist in the Rust engine.

use super::{g, GVal, Scope, Step, Token, Traversal, __, P};
use crate::ndjson;

/// Append a raw step (for local min/max/mean — no fluent builder helpers).
fn with_step(mut t: Traversal, s: Step) -> Traversal {
    t.steps.push(s);
    t
}

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> crate::graph::Graph {
    crate::fixtures::modern_gremlin_edge_ids()
}

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

/// String results in stream order.
fn ordered(r: Vec<GVal>) -> Vec<String> {
    r.iter().map(s).collect()
}

/// String results, sorted (order-independent assertions).
fn sorted(r: Vec<GVal>) -> Vec<String> {
    let mut v: Vec<String> = r.iter().map(s).collect();
    v.sort();
    v
}

fn one_num(r: Vec<GVal>) -> f64 {
    match r.as_slice() {
        [GVal::Num(n)] => *n,
        _ => panic!("expected single number, got {r:?}"),
    }
}

/// Resolve an element traverser to its graph id (for tests asserting ids).
fn ids(t: Traversal) -> Vec<String> {
    let mut g = modern();
    t.run(&mut g)
        .iter()
        .map(|v| match v {
            GVal::Node(i) => g.vid.text(*i).to_string(),
            GVal::Edge(e) => g.edge_id(*e).into_owned(),
            other => format!("{other:?}"),
        })
        .collect()
}

/// Sort a result map's entries by string key.
fn map_entries(g: &GVal) -> Vec<(String, GVal)> {
    match g {
        GVal::Map(entries) => entries.iter().map(|(k, val)| (s(k), val.clone())).collect(),
        _ => panic!("expected map, got {g:?}"),
    }
}

/// An `inject(...)` source carrying raw GVals (so we can inject Null / List).
fn inject_src(vs: Vec<GVal>) -> Traversal {
    Traversal {
        steps: vec![Step::Inject(vs)],
    }
}

// ===== values.test.ts =====

// SKIPPED: `V('1').values()` (no keys) — the columnar store enumerates present
// keys in global key-interning order, not per-vertex insertion order, so the
// emitted value order ([29,"marko"]) diverges from TS ([ "marko",29 ]).

#[test]
fn p5_values_filters_missing_key() {
    assert_eq!(
        q(g().V().values(&["age"])),
        vec![
            GVal::Num(29.0),
            GVal::Num(27.0),
            GVal::Num(32.0),
            GVal::Num(35.0)
        ]
    );
}

#[test]
fn p5_values_multiple_keys() {
    assert_eq!(
        q(g().V().values(&["name", "age"])),
        vec![
            GVal::Str("marko".into()),
            GVal::Num(29.0),
            GVal::Str("vadas".into()),
            GVal::Num(27.0),
            GVal::Str("josh".into()),
            GVal::Num(32.0),
            GVal::Str("peter".into()),
            GVal::Num(35.0),
            GVal::Str("lop".into()),
            GVal::Str("ripple".into()),
        ]
    );
}

#[test]
fn p5_values_chained_has_out_values() {
    assert_eq!(
        ordered(q(g()
            .V()
            .has("name", P::eq("marko"))
            .out(&["KNOWS"])
            .values(&["name"]))),
        vec!["vadas", "josh"]
    );
}

#[test]
fn p5_values_out_then_values_order() {
    assert_eq!(
        ordered(q(g().v_ids(&["1"]).out(&[]).values(&["name"]))),
        vec!["vadas", "josh", "lop"]
    );
}

#[test]
fn p5_values_all_names() {
    assert_eq!(
        ordered(q(g().V().values(&["name"]))),
        vec!["marko", "vadas", "josh", "peter", "lop", "ripple"]
    );
}

#[test]
fn p5_values_out_by_label_then_values() {
    assert_eq!(
        ordered(q(g().v_ids(&["1"]).out(&["KNOWS"]).values(&["name"]))),
        vec!["vadas", "josh"]
    );
}

#[test]
fn p5_values_has_out_created_values() {
    assert_eq!(
        ordered(q(g()
            .V()
            .has("name", P::eq("marko"))
            .out(&["CREATED"])
            .values(&["name"]))),
        vec!["lop"]
    );
}

#[test]
fn p5_values_has_values_age() {
    assert_eq!(
        q(g().V().has("name", P::eq("marko")).values(&["age"])),
        vec![GVal::Num(29.0)]
    );
}

#[test]
fn p5_values_out_out_values() {
    assert_eq!(
        ordered(q(g().V().out(&[]).out(&[]).values(&["name"]))),
        vec!["ripple", "lop"]
    );
}

#[test]
fn p5_values_chained_predicate_has_age() {
    assert_eq!(
        ordered(q(g()
            .V()
            .has("name", P::eq("marko"))
            .out(&["KNOWS"])
            .has("age", P::gt(29))
            .values(&["name"]))),
        vec!["josh"]
    );
}

// ===== sum.test.ts =====

#[test]
fn p5_sum_numbers() {
    assert_eq!(q(g().V().values(&["age"]).sum()), vec![GVal::Num(123.0)]);
}

#[test]
fn p5_sum_with_repeat() {
    // V().repeat(both()).times(3).values('age').sum() — 1471 in the modern graph.
    let r = g()
        .V()
        .repeat(__().both(&[]))
        .times(3)
        .values(&["age"])
        .sum();
    assert_eq!(q(r), vec![GVal::Num(1471.0)]);
}

#[test]
fn p5_sum_filters_null() {
    // inject(null, 10, 9, null).sum() — nulls dropped → 19.
    let r = inject_src(vec![
        GVal::Null,
        GVal::Num(10.0),
        GVal::Num(9.0),
        GVal::Null,
    ])
    .sum();
    assert_eq!(q(r), vec![GVal::Num(19.0)]);
}

#[test]
fn p5_sum_local_of_folded_list() {
    assert_eq!(
        q(g().V().values(&["age"]).fold().sum_local()),
        vec![GVal::Num(123.0)]
    );
}

#[test]
fn p5_min_local_of_folded_list() {
    let r = with_step(g().V().values(&["age"]).fold(), Step::Min(Scope::Local));
    assert_eq!(q(r), vec![GVal::Num(27.0)]);
}

#[test]
fn p5_max_local_of_folded_list() {
    let r = with_step(g().V().values(&["age"]).fold(), Step::Max(Scope::Local));
    assert_eq!(q(r), vec![GVal::Num(35.0)]);
}

#[test]
fn p5_mean_local_of_folded_list() {
    let r = with_step(g().V().values(&["age"]).fold(), Step::Mean(Scope::Local));
    assert_eq!(q(r), vec![GVal::Num(30.75)]);
}

#[test]
fn p5_sum_local_empty_fold_yields_null() {
    // inject([]).sum(Scope.local) — empty local fold → null.
    let r = inject_src(vec![GVal::List(vec![])]).sum_local();
    assert_eq!(q(r), vec![GVal::Null]);
}

// ===== dedup.test.ts =====

#[test]
fn p5_dedup_strings() {
    assert_eq!(
        q(g().V().values(&["lang"])),
        vec![GVal::Str("java".into()), GVal::Str("java".into())]
    );
    assert_eq!(
        q(g().V().values(&["lang"]).dedup()),
        vec![GVal::Str("java".into())]
    );
}

#[test]
fn p5_dedup_select_cartesian_shape() {
    // V().as(a).out(CREATED).as(b).in(CREATED).as(c).select(a,b,c) — 10 rows
    // of {a,b,c} vertex maps (the cartesian shape before any dedup).
    let r = q(g()
        .V()
        .as_("a")
        .out(&["CREATED"])
        .as_("b")
        .in_(&["CREATED"])
        .as_("c")
        .select(&["a", "b", "c"]));
    let triples: Vec<(String, String, String)> = r
        .iter()
        .map(|m| {
            let e = map_entries(m);
            let resolve = |g: &GVal| match g {
                GVal::Node(_) => g.clone(),
                other => other.clone(),
            };
            // resolve to ids via a throwaway graph lookup
            (
                vid(&resolve(&e[0].1)),
                vid(&resolve(&e[1].1)),
                vid(&resolve(&e[2].1)),
            )
        })
        .collect();
    assert_eq!(
        triples,
        vec![
            ("1".into(), "3".into(), "1".into()),
            ("1".into(), "3".into(), "4".into()),
            ("1".into(), "3".into(), "6".into()),
            ("4".into(), "5".into(), "4".into()),
            ("4".into(), "3".into(), "1".into()),
            ("4".into(), "3".into(), "4".into()),
            ("4".into(), "3".into(), "6".into()),
            ("6".into(), "3".into(), "1".into()),
            ("6".into(), "3".into(), "4".into()),
            ("6".into(), "3".into(), "6".into()),
        ]
    );
}

#[test]
fn p5_dedup_by_label_keeps_one_per_label() {
    // V().dedup().by(T.label).values('name') — first PERSON, first SOFTWARE.
    let r = g().V().dedup().by_token(Token::Label).values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["marko", "lop"]);
}

#[test]
fn p5_dedup_after_out_created() {
    let r = g().V().has_label(&["PERSON"]).out(&["CREATED"]).dedup();
    assert_eq!(ids(r), vec!["3", "5"]);
}

#[test]
fn p5_dedup_via_oute_inv() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .out_e(&["CREATED"])
        .in_v()
        .dedup();
    assert_eq!(ids(r), vec!["3", "5"]);
}

// ===== properties.test.ts =====

/// A `{key, value}` property object as the engine emits it.
fn prop_obj(key: &str, value: GVal) -> GVal {
    // Owner is ignored by `GVal`'s PartialEq, so a placeholder is fine here.
    GVal::property(GVal::Null, key.into(), value)
}

#[test]
fn p5_properties_one_vertex_named() {
    assert_eq!(
        q(g().V().has_id(&["1"]).properties(&["name"])),
        vec![prop_obj("name", GVal::Str("marko".into()))]
    );
}

#[test]
fn p5_properties_named_across_all() {
    assert_eq!(
        q(g().V().properties(&["name"])),
        vec![
            prop_obj("name", GVal::Str("marko".into())),
            prop_obj("name", GVal::Str("vadas".into())),
            prop_obj("name", GVal::Str("josh".into())),
            prop_obj("name", GVal::Str("peter".into())),
            prop_obj("name", GVal::Str("lop".into())),
            prop_obj("name", GVal::Str("ripple".into())),
        ]
    );
}

#[test]
fn p5_properties_multiple_keys_flatten() {
    assert_eq!(
        q(g().V().has_id(&["1"]).properties(&["name", "age"])),
        vec![
            prop_obj("name", GVal::Str("marko".into())),
            prop_obj("age", GVal::Num(29.0)),
        ]
    );
}

#[test]
fn p5_properties_no_keys_yields_all() {
    assert_eq!(
        q(g().V().has_id(&["3"]).properties(&[])),
        vec![
            prop_obj("name", GVal::Str("lop".into())),
            prop_obj("lang", GVal::Str("java".into())),
        ]
    );
}

#[test]
fn p5_properties_count() {
    assert_eq!(
        one_num(q(g().V().has_id(&["1"]).properties(&["name"]).count())),
        1.0
    );
}

// ===== inject.test.ts =====

#[test]
fn p5_inject_string_appends() {
    // V('4').out().values('name').inject('daniel') — injected value first.
    let r = g()
        .v_ids(&["4"])
        .out(&[])
        .values(&["name"])
        .inject(["daniel"]);
    assert_eq!(ordered(q(r)), vec!["daniel", "ripple", "lop"]);
}

#[test]
fn p5_inject_as_source_in_order() {
    let r = inject_src(vec!["a".into(), "b".into(), "c".into()]);
    assert_eq!(ordered(q(r)), vec!["a", "b", "c"]);
}

#[test]
fn p5_inject_preserves_arrays_no_unfold() {
    // inject([1,2,3],[4,5]) — lists stay as single values.
    let r = inject_src(vec![
        GVal::List(vec![GVal::Num(1.0), GVal::Num(2.0), GVal::Num(3.0)]),
        GVal::List(vec![GVal::Num(4.0), GVal::Num(5.0)]),
    ]);
    assert_eq!(
        q(r),
        vec![
            GVal::List(vec![GVal::Num(1.0), GVal::Num(2.0), GVal::Num(3.0)]),
            GVal::List(vec![GVal::Num(4.0), GVal::Num(5.0)]),
        ]
    );
}

// ===== valueMap.test.ts =====

// SKIPPED: `valueMap()` (no keys) — same key-ordering divergence as values():
// the columnar store yields keys in global interning order, not per-vertex
// insertion order, so the entry order (age before name) diverges from TS.

#[test]
fn p5_valuemap_single_property() {
    let r = q(g().V().value_map(&["age"]));
    let rows: Vec<Vec<(String, GVal)>> = r.iter().map(map_entries).collect();
    assert_eq!(
        rows,
        vec![
            vec![("age".into(), GVal::Num(29.0))],
            vec![("age".into(), GVal::Num(27.0))],
            vec![("age".into(), GVal::Num(32.0))],
            vec![("age".into(), GVal::Num(35.0))],
            vec![],
            vec![],
        ]
    );
}

#[test]
fn p5_valuemap_skips_missing_keys() {
    let r = q(g().V().value_map(&["age", "blah"]));
    let rows: Vec<Vec<(String, GVal)>> = r.iter().map(map_entries).collect();
    assert_eq!(
        rows,
        vec![
            vec![("age".into(), GVal::Num(29.0))],
            vec![("age".into(), GVal::Num(27.0))],
            vec![("age".into(), GVal::Num(32.0))],
            vec![("age".into(), GVal::Num(35.0))],
            vec![],
            vec![],
        ]
    );
}

#[test]
fn p5_valuemap_on_edges() {
    let r = q(g().E().value_map(&[]));
    let rows: Vec<Vec<(String, GVal)>> = r.iter().map(map_entries).collect();
    assert_eq!(
        rows,
        vec![
            vec![("weight".into(), GVal::Num(0.5))],
            vec![("weight".into(), GVal::Num(1.0))],
            vec![("weight".into(), GVal::Num(0.4))],
            vec![("weight".into(), GVal::Num(1.0))],
            vec![("weight".into(), GVal::Num(0.4))],
            vec![("weight".into(), GVal::Num(0.2))],
        ]
    );
}

// ===== propertyMap.test.ts =====

fn one_list(v: GVal) -> GVal {
    GVal::List(vec![v])
}

// SKIPPED: `propertyMap()` (no keys) — same key-ordering divergence as values():
// global key-interning order vs per-vertex insertion order (age before name).

#[test]
fn p5_propertymap_single_key_skips_missing() {
    let r = q(g().V().property_map(&["age"]));
    let rows: Vec<Vec<(String, GVal)>> = r.iter().map(map_entries).collect();
    assert_eq!(
        rows,
        vec![
            vec![("age".into(), one_list(GVal::Num(29.0)))],
            vec![("age".into(), one_list(GVal::Num(27.0)))],
            vec![("age".into(), one_list(GVal::Num(32.0)))],
            vec![("age".into(), one_list(GVal::Num(35.0)))],
            vec![],
            vec![],
        ]
    );
}

#[test]
fn p5_propertymap_skips_unknown_keys() {
    let r = q(g().V().property_map(&["age", "blah"]));
    let rows: Vec<Vec<(String, GVal)>> = r.iter().map(map_entries).collect();
    assert_eq!(
        rows,
        vec![
            vec![("age".into(), one_list(GVal::Num(29.0)))],
            vec![("age".into(), one_list(GVal::Num(27.0)))],
            vec![("age".into(), one_list(GVal::Num(32.0)))],
            vec![("age".into(), one_list(GVal::Num(35.0)))],
            vec![],
            vec![],
        ]
    );
}

#[test]
fn p5_propertymap_on_edges() {
    let r = q(g().E().property_map(&[]));
    let rows: Vec<Vec<(String, GVal)>> = r.iter().map(map_entries).collect();
    assert_eq!(
        rows,
        vec![
            vec![("weight".into(), one_list(GVal::Num(0.5)))],
            vec![("weight".into(), one_list(GVal::Num(1.0)))],
            vec![("weight".into(), one_list(GVal::Num(0.4)))],
            vec![("weight".into(), one_list(GVal::Num(1.0)))],
            vec![("weight".into(), one_list(GVal::Num(0.4)))],
            vec![("weight".into(), one_list(GVal::Num(0.2)))],
        ]
    );
}

// ===== loops.test.ts =====

// NOTE: `repeat(out()).until(loops().is(2))` and
// `repeat(out()).times(3).emit(loops().is(gt(1)))` now match TS — the loop
// counter increments on entry and on body output (loops() counts from 1 in the
// first body pass). See divergence_tests::repeat_until_loops_stops_after_first_pass
// and ::repeat_emit_loops_predicate_offset.

#[test]
fn p5_loops_body_filter_emit_all() {
    // V('1').repeat(out().hasLabel(PERSON)).times(3).emit() — {vadas, josh}.
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]).has_label(&["PERSON"]))
        .times(3)
        .emit_all()
        .values(&["name"]);
    assert_eq!(sorted(q(r)), vec!["josh", "vadas"]);
}

// ===== shortestPath.test.ts =====

/// Resolve each emitted shortest path's vertices to ids.
fn sp_paths(t: Traversal) -> Vec<Vec<String>> {
    let mut g = modern();
    t.run(&mut g)
        .iter()
        .map(|p| match p {
            GVal::List(vs) => vs
                .iter()
                .map(|v| match v {
                    GVal::Node(i) => g.vid.text(*i).to_string(),
                    other => format!("{other:?}"),
                })
                .collect(),
            other => panic!("expected a path list, got {other:?}"),
        })
        .collect()
}

#[test]
fn p5_shortest_path_target_marko_josh() {
    let paths = sp_paths(
        g().V()
            .has("name", P::eq("marko"))
            .shortest_path_to(__().has("name", P::eq("josh"))),
    );
    assert_eq!(paths, vec![vec!["1".to_string(), "4".to_string()]]);
}

#[test]
fn p5_shortest_path_multi_hop_marko_ripple() {
    let paths = sp_paths(
        g().V()
            .has("name", P::eq("marko"))
            .shortest_path_to(__().has("name", P::eq("ripple"))),
    );
    assert_eq!(
        paths,
        vec![vec!["1".to_string(), "4".to_string(), "5".to_string()]]
    );
}

#[test]
fn p5_shortest_path_no_target_reaches_all() {
    let paths = sp_paths(g().V().has("name", P::eq("marko")).shortest_path());
    let reached: std::collections::HashSet<String> =
        paths.iter().map(|p| p.last().unwrap().clone()).collect();
    assert_eq!(
        reached,
        ["1", "2", "3", "4", "5", "6"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    );
}

#[test]
fn p5_shortest_path_direction_is_configurable() {
    // vadas (id 2) has only an incoming edge (marko→vadas). Undirected (default)
    // reaches beyond it; a directed `out` search reaches only vadas itself.
    let both = sp_paths(g().V().has("name", P::eq("vadas")).shortest_path());
    assert!(both.len() > 1, "undirected reaches marko and beyond");

    let out = sp_paths(
        g().V()
            .has("name", P::eq("vadas"))
            .shortest_path()
            .with_shortest_path_edges("Direction.OUT")
            .unwrap(),
    );
    assert_eq!(out, vec![vec!["2".to_string()]]);

    // `Direction.BOTH` explicitly is the same as the default.
    let both2 = sp_paths(
        g().V()
            .has("name", P::eq("vadas"))
            .shortest_path()
            .with_shortest_path_edges("Direction.BOTH")
            .unwrap(),
    );
    assert_eq!(both2.len(), both.len());
}

// ===== id.test.ts =====

#[test]
fn p5_id_all_vertices() {
    assert_eq!(ordered(q(g().V().id())), vec!["1", "2", "4", "6", "3", "5"]);
}

#[test]
fn p5_id_with_is_filters() {
    // V('1').out().id().is(eq('2')) — only the '2' (vadas) id survives.
    let r = g().v_ids(&["1"]).out(&[]).id().is(P::eq("2"));
    assert_eq!(ordered(q(r)), vec!["2"]);
}

#[test]
fn p5_id_of_out_edges() {
    // V('1').outE().id() — edge ids 7, 8, 9.
    assert_eq!(
        ordered(q(g().v_ids(&["1"]).out_e(&[]).id())),
        vec!["7", "8", "9"]
    );
}

// ===== as.test.ts =====

#[test]
fn p5_as_is_noop_on_stream() {
    let r = q(g().v_ids(&["1"]).as_("a").values(&["name"]));
    assert_eq!(r, vec![GVal::Str("marko".into())]);
}

#[test]
fn p5_as_multiple_no_effect_on_return() {
    let r = g()
        .V()
        .as_("a")
        .out(&[])
        .as_("b")
        .out(&[])
        .as_("c")
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["ripple", "lop"]);
}

#[test]
fn p5_as_feeds_select_a_b() {
    let r = g()
        .v_ids(&["1"])
        .as_("a")
        .out(&["KNOWS"])
        .as_("b")
        .select(&["a", "b"]);
    let pairs: Vec<(String, String)> = q(r)
        .iter()
        .map(|m| {
            let e = map_entries(m);
            (vid(&e[0].1), vid(&e[1].1))
        })
        .collect();
    assert_eq!(
        pairs,
        vec![("1".into(), "2".into()), ("1".into(), "4".into()),]
    );
}

// ===== inV.test.ts =====

#[test]
fn p5_inv_oute_inv_names() {
    let r = q(g().v_ids(&["4"]).out_e(&[]).in_v().values(&["name"]));
    assert_eq!(ordered(r), vec!["ripple", "lop"]);
}

#[test]
fn p5_inv_oute_inv_ids() {
    let r = g().v_ids(&["4"]).out_e(&[]).in_v();
    assert_eq!(ids(r), vec!["5", "3"]);
}

// ===== path-tracking.test.ts (correctness only) =====

#[test]
fn p5_path_yields_full_accumulated_path() {
    // Chain a→b→c; path() over out().out() yields [a,b,c].
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["E"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["E"],"properties":{}}"#,
    ];
    let mut gr = ndjson::decode(&lines.join("\n")).unwrap();
    let out = g().v_ids(&["a"]).out(&[]).out(&[]).path().run(&mut gr);
    assert_eq!(out.len(), 1);
    let path_ids: Vec<String> = match &out[0] {
        GVal::List(vs) => vs
            .iter()
            .map(|v| match v {
                GVal::Node(i) => gr.vid.text(*i).to_string(),
                other => format!("{other:?}"),
            })
            .collect(),
        other => panic!("expected path list, got {other:?}"),
    };
    assert_eq!(path_ids, vec!["a", "b", "c"]);
}

#[test]
fn p5_simple_path_filters_revisits() {
    // both()/both() on a→b can walk back a→b→a; simplePath drops the revisit.
    let lines = [
        r#"{"type":"node","id":"a","labels":["N"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["N"],"properties":{}}"#,
        r#"{"type":"node","id":"c","labels":["N"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["E"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["E"],"properties":{}}"#,
    ];
    let mut g1 = ndjson::decode(&lines.join("\n")).unwrap();
    let with_simple = g()
        .v_ids(&["a"])
        .both(&["E"])
        .both(&["E"])
        .simple_path()
        .run(&mut g1);
    let mut g2 = ndjson::decode(&lines.join("\n")).unwrap();
    let without = g().v_ids(&["a"]).both(&["E"]).both(&["E"]).run(&mut g2);
    assert!(with_simple.len() < without.len());
}

// --- helper: resolve a GVal::Node to its id against a fresh modern graph ----
// (the modern fixture is deterministic, so a re-decode maps dense ids back).
fn vid(v: &GVal) -> String {
    let g = modern();
    match v {
        GVal::Node(i) => g.vid.text(*i).to_string(),
        GVal::Edge(e) => g.edge_id(*e).into_owned(),
        other => format!("{other:?}"),
    }
}
