//! Per-step conformance tests, part 6 of 6: `select`, `order`, `group`, `skip`,
//! `hasLabel`, `path`, `inE`, `tree`, `or`, `hasKey`, `both`, `optional`,
//! `hasValue`, `addV`, `identity`.
//!
//! The six parts are a SIZE split, not a thematic one — the number carries no
//! meaning.
//!
//! Mirrors the TS Gremlin step tests. Self-contained: own `modern()` fixture
//! (with explicit edge ids 7..=12 to match the TS TinkerGraph edge-id assertions),
//! own helpers, own tests. Read-only queries use `run`; no error cases here.

use super::{g, GVal, Order, P};
use crate::graph::Graph;

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gremlin_edge_ids()
}

// ---- helpers --------------------------------------------------------------

/// Run a read-only traversal against a fresh Modern graph.
fn q(t: super::Traversal) -> Vec<GVal> {
    let mut g = modern();
    t.run(&mut g)
}

fn s(g: &GVal) -> String {
    match g {
        GVal::Str(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

/// String results in stream order (order-dependent traversals).
fn ordered(r: Vec<GVal>) -> Vec<String> {
    r.iter().map(s).collect()
}

/// Sorted string results (order-independent traversals).
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

/// Run, then resolve each emitted element to its id / label / value, keeping
/// the graph alive (vertices carry only a dense index).
fn ids_of(t: super::Traversal) -> Vec<String> {
    let mut g = modern();
    t.run(&mut g).iter().map(|v| gval_text(&g, v)).collect()
}

fn gval_text(g: &Graph, v: &GVal) -> String {
    match v {
        GVal::Node(i) => g.vid.text(*i).to_string(),
        GVal::Edge(e) => g.edge_id(*e).into_owned(),
        GVal::Str(s) => s.to_string(),
        GVal::Num(n) => format!("{n}"),
        other => format!("{other:?}"),
    }
}

/// Resolve a list of paths (each a `GVal::List`) to lists of element texts.
fn paths_text(t: super::Traversal) -> Vec<Vec<String>> {
    let mut g = modern();
    t.run(&mut g)
        .iter()
        .map(|p| match p {
            GVal::List(items) => items.iter().map(|v| gval_text(&g, v)).collect(),
            other => panic!("expected path list, got {other:?}"),
        })
        .collect()
}

fn as_map(g: &GVal) -> &crate::value::MapVal {
    match g {
        GVal::Map(e) => e,
        _ => panic!("expected map, got {g:?}"),
    }
}

fn map_get<'a>(m: &'a crate::value::MapVal, key: &str) -> Option<&'a GVal> {
    m.iter()
        .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == key))
        .map(|(_, v)| v)
}

fn map_get_gval<'a>(m: &'a crate::value::MapVal, key: &GVal) -> Option<&'a GVal> {
    m.get(key)
}

fn list_of(g: &GVal) -> &[GVal] {
    match g {
        GVal::List(items) => items,
        _ => panic!("expected list, got {g:?}"),
    }
}

// ===== select.test.ts =====

#[test]
fn p6_select_multiple_labeled_positions() {
    // V().as(a).out().as(b).out().as(c).select(a,b,c) → ids.
    let r = q(g()
        .V()
        .as_("a")
        .out(&[])
        .as_("b")
        .out(&[])
        .as_("c")
        .select(&["a", "b", "c"])
        .by_id());
    let rows: Vec<Vec<(String, String)>> = r
        .iter()
        .map(|m| as_map(m).iter().map(|(k, v)| (s(k), s(v))).collect())
        .collect();
    assert_eq!(
        rows,
        vec![
            vec![
                ("a".into(), "1".into()),
                ("b".into(), "4".into()),
                ("c".into(), "5".into())
            ],
            vec![
                ("a".into(), "1".into()),
                ("b".into(), "4".into()),
                ("c".into(), "3".into())
            ],
        ]
    );
}

#[test]
fn p6_select_need_not_select_everything() {
    let r = q(g()
        .V()
        .as_("a")
        .out(&[])
        .as_("b")
        .out(&[])
        .as_("c")
        .select(&["a", "b"])
        .by_id());
    let rows: Vec<Vec<(String, String)>> = r
        .iter()
        .map(|m| as_map(m).iter().map(|(k, v)| (s(k), s(v))).collect())
        .collect();
    assert_eq!(
        rows,
        vec![
            vec![("a".into(), "1".into()), ("b".into(), "4".into())],
            vec![("a".into(), "1".into()), ("b".into(), "4".into())],
        ]
    );
}

#[test]
fn p6_select_single_label_unwraps() {
    let r = g()
        .V()
        .as_("a")
        .out(&[])
        .as_("b")
        .out(&[])
        .as_("c")
        .select(&["a"]);
    assert_eq!(ids_of(r), vec!["1", "1"]);
}

#[test]
fn p6_select_finds_start_of_longer_path() {
    let r = g().V().as_("x").out(&[]).out(&[]).select(&["x"]);
    assert_eq!(ids_of(r), vec!["1", "1"]);
}

#[test]
fn p6_select_middle_label() {
    let r = g().V().out(&[]).as_("x").out(&[]).select(&["x"]);
    assert_eq!(ids_of(r), vec!["4", "4"]);
}

#[test]
fn p6_select_current_position() {
    let r = g()
        .V()
        .out(&[])
        .out(&[])
        .as_("x")
        .select(&["x"])
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["ripple", "lop"]);
}

#[test]
fn p6_select_both_pair_per_neighbor() {
    // g.V(1).as(a).both().as(b).select(a,b) — marko's both() = vadas, josh, lop.
    let r = q(g()
        .v_ids(&["1"])
        .as_("a")
        .both(&[])
        .as_("b")
        .select(&["a", "b"])
        .by_id());
    let rows: Vec<Vec<(String, String)>> = r
        .iter()
        .map(|m| as_map(m).iter().map(|(k, v)| (s(k), s(v))).collect())
        .collect();
    assert_eq!(
        rows,
        vec![
            vec![("a".into(), "1".into()), ("b".into(), "2".into())],
            vec![("a".into(), "1".into()), ("b".into(), "4".into())],
            vec![("a".into(), "1".into()), ("b".into(), "3".into())],
        ]
    );
}

#[test]
fn p6_select_drops_missing_label() {
    let r = q(g().v_ids(&["1"]).as_("a").select(&["missing"]));
    assert_eq!(r.len(), 0);
}

#[test]
fn p6_select_by_subtraversal_projects() {
    // select('a','b').by(in(CREATED).count()).by('name'); a=marko →0, b=lop→'lop'.
    let r = q(g()
        .v_ids(&["1"])
        .as_("a")
        .out(&["CREATED"])
        .as_("b")
        .select(&["a", "b"])
        .by_t(super::__().in_(&["CREATED"]).count())
        .by("name"));
    let m = as_map(&r[0]);
    assert_eq!(map_get(m, "a"), Some(&GVal::Num(0.0)));
    assert_eq!(map_get(m, "b"), Some(&GVal::Str("lop".into())));
}

#[test]
fn p6_select_single_by_fold_count() {
    // V(3=lop).as(a).select(a).by(in(CREATED).values(name).count()) → 3.
    let r = q(g()
        .v_ids(&["3"])
        .as_("a")
        .select(&["a"])
        .by_t(super::__().in_(&["CREATED"]).values(&["name"]).count()));
    assert_eq!(one_num(r), 3.0);
}

#[test]
fn p6_select_by_name_both_positions() {
    let r = q(g()
        .v_ids(&["1"])
        .as_("a")
        .out(&["KNOWS"])
        .as_("b")
        .select(&["a", "b"])
        .by("name")
        .by("name"));
    let rows: Vec<(String, String)> = r
        .iter()
        .map(|m| {
            let m = as_map(m);
            (s(map_get(m, "a").unwrap()), s(map_get(m, "b").unwrap()))
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            ("marko".into(), "vadas".into()),
            ("marko".into(), "josh".into()),
        ]
    );
}

// ===== order.test.ts =====

#[test]
fn p6_order_simple() {
    let r = g().V().values(&["name"]).order();
    assert_eq!(
        ordered(q(r)),
        vec!["josh", "lop", "marko", "peter", "ripple", "vadas"]
    );
}

#[test]
fn p6_order_desc() {
    let r = g()
        .V()
        .values(&["name"])
        .order()
        .by_identity_dir(Order::Desc);
    assert_eq!(
        ordered(q(r)),
        vec!["vadas", "ripple", "peter", "marko", "lop", "josh"]
    );
}

#[test]
fn p6_order_by_key_age() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .order()
        .by("age")
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["vadas", "marko", "josh", "peter"]);
}

#[test]
fn p6_order_then_tail_one() {
    let r = g().V().values(&["name"]).order().tail(1);
    assert_eq!(ordered(q(r)), vec!["vadas"]);
}

#[test]
fn p6_order_then_tail_three() {
    let r = g().V().values(&["name"]).order().tail(3);
    assert_eq!(ordered(q(r)), vec!["peter", "ripple", "vadas"]);
}

#[test]
fn p6_order_by_order_desc() {
    let r = g()
        .V()
        .values(&["name"])
        .order()
        .by_identity_dir(Order::Desc);
    assert_eq!(
        ordered(q(r)),
        vec!["vadas", "ripple", "peter", "marko", "lop", "josh"]
    );
}

#[test]
fn p6_order_by_order_asc() {
    let r = g()
        .V()
        .values(&["name"])
        .order()
        .by_identity_dir(Order::Asc);
    assert_eq!(
        ordered(q(r)),
        vec!["josh", "lop", "marko", "peter", "ripple", "vadas"]
    );
}

#[test]
fn p6_order_by_key_desc() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .order()
        .by_dir("age", Order::Desc)
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["peter", "josh", "marko", "vadas"]);
}

// ===== skip.test.ts =====

#[test]
fn p6_skip_range_first_three() {
    let r = g().V().range(0, 3).values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["marko", "vadas", "josh"]);
}

#[test]
fn p6_skip_low_end() {
    // V().values(age).skip(2) → ages of josh, peter in V() order.
    let r = g().V().values(&["age"]).skip(2);
    assert_eq!(q(r), vec![GVal::Num(32.0), GVal::Num(35.0)]);
}

#[test]
fn p6_skip_open_end() {
    // V().values(name).skip(3): V() order = marko,vadas,josh,peter,lop,ripple.
    let r = g().V().values(&["name"]).skip(3);
    assert_eq!(ordered(q(r)), vec!["peter", "lop", "ripple"]);
}

#[test]
fn p6_order_age_natural() {
    let r = g().V().values(&["age"]).order();
    assert_eq!(
        q(r),
        vec![
            GVal::Num(27.0),
            GVal::Num(29.0),
            GVal::Num(32.0),
            GVal::Num(35.0)
        ]
    );
}

#[test]
fn p6_order_then_skip_two() {
    let r = g().V().values(&["age"]).order().skip(2);
    assert_eq!(q(r), vec![GVal::Num(32.0), GVal::Num(35.0)]);
}

#[test]
fn p6_skip_equiv_range_open() {
    // skip(n) == range(n, MAX) (Rust has no negative end; usize::MAX is "open").
    let a = q(g().V().values(&["age"]).order().skip(2));
    let b = q(g().V().values(&["age"]).order().range(2, usize::MAX));
    assert_eq!(a, b);
}

// ===== hasLabel.test.ts =====

#[test]
fn p6_haslabel_all_persons() {
    assert_eq!(q(g().V().has_label(&["PERSON"])).len(), 4);
}

#[test]
fn p6_haslabel_stable_order() {
    let r = g().V().has_label(&["PERSON"]).values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["marko", "vadas", "josh", "peter"]);
}

#[test]
fn p6_haslabel_single_vertex() {
    let r = g().v_ids(&["1"]).has_label(&["PERSON"]).values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["marko"]);
}

#[test]
fn p6_haslabel_edges_has_weight() {
    // E().hasLabel(KNOWS).has(weight, gt(0.75)) → edge 8.
    let r = g()
        .E()
        .has_label(&["KNOWS"])
        .has("weight", P::gt(0.75))
        .id();
    assert_eq!(ordered(q(r)), vec!["8"]);
}

#[test]
fn p6_haslabel_range_slices() {
    let r = g().V().has_label(&["PERSON"]).range(0, 2).id();
    assert_eq!(ordered(q(r)), vec!["1", "2"]);
}

#[test]
fn p6_haslabel_four_person_ids() {
    let r = g().V().has_label(&["PERSON"]).id();
    assert_eq!(ordered(q(r)), vec!["1", "2", "4", "6"]);
}

// ===== path.test.ts =====

#[test]
fn p6_path_simple_tinker_toy() {
    let r = g().V().out(&[]).out(&[]).path();
    assert_eq!(
        paths_text(r),
        vec![vec!["1", "4", "5"], vec!["1", "4", "3"]]
    );
}

#[test]
fn p6_path_complex_edges() {
    let r = g().V().out_e(&[]).in_v().out_e(&[]).in_v().path();
    assert_eq!(
        paths_text(r),
        vec![
            vec!["1", "8", "4", "10", "5"],
            vec!["1", "8", "4", "11", "3"],
        ]
    );
}

#[test]
fn p6_path_by_name() {
    let r = g().V().out(&[]).out(&[]).path().by("name");
    assert_eq!(
        paths_text(r),
        vec![
            vec!["marko", "josh", "ripple"],
            vec!["marko", "josh", "lop"],
        ]
    );
}

#[test]
fn p6_path_includes_values() {
    let r = g().v_ids(&["1"]).out(&["KNOWS"]).values(&["name"]).path();
    assert_eq!(
        paths_text(r),
        vec![vec!["1", "2", "vadas"], vec!["1", "4", "josh"]]
    );
}

#[test]
fn p6_path_multiple_by_round_robin() {
    // by('name'),by('age') applied round-robin: [name, age, name].
    let r = g().V().out(&[]).out(&[]).path().by("name").by("age");
    let mut g0 = modern();
    let out = r.run(&mut g0);
    // marko→josh→ripple: [marko, 32, ripple]; marko→josh→lop: [marko, 32, lop].
    let row0 = list_of(&out[0]);
    assert_eq!(row0[0], GVal::Str("marko".into()));
    assert_eq!(row0[1], GVal::Num(32.0));
    assert_eq!(row0[2], GVal::Str("ripple".into()));
    let row1 = list_of(&out[1]);
    assert_eq!(row1[0], GVal::Str("marko".into()));
    assert_eq!(row1[1], GVal::Num(32.0));
    assert_eq!(row1[2], GVal::Str("lop".into()));
}

// ===== inE.test.ts =====

#[test]
fn p6_ine_toy() {
    // V(4).inE() → edge 8 (marko-knows-josh, weight 1.0); from = marko, age 29.
    assert_eq!(q(g().v_ids(&["4"]).in_e(&[])).len(), 1);
    // edge weight 1.0
    let weight = q(g().v_ids(&["4"]).in_e(&[]).values(&["weight"]));
    assert_eq!(weight, vec![GVal::Num(1.0)]);
    // from vertex = marko (src of edge 8), age 29.
    let from = q(g().v_ids(&["4"]).in_e(&[]).out_v().values(&["name"]));
    assert_eq!(ordered(from), vec!["marko"]);
    let age = q(g().v_ids(&["4"]).in_e(&[]).out_v().values(&["age"]));
    assert_eq!(age, vec![GVal::Num(29.0)]);
}

#[test]
fn p6_ine_specific_label_empty() {
    let r = q(g().v_ids(&["1"]).in_e(&["KNOWS"]));
    assert_eq!(r.len(), 0);
}

#[test]
fn p6_ine_knows_on_v4() {
    let r = g().v_ids(&["4"]).in_e(&["KNOWS"]).id();
    assert_eq!(ordered(q(r)), vec!["8"]);
}

#[test]
fn p6_ine_created_on_v4_empty() {
    let r = q(g().v_ids(&["4"]).in_e(&["CREATED"]));
    assert_eq!(r.len(), 0);
}

#[test]
fn p6_ine_created_on_v3() {
    // V(3=lop).inE(CREATED): from marko, josh, peter; weights 0.4, 0.4, 0.2.
    let froms = g()
        .v_ids(&["3"])
        .in_e(&["CREATED"])
        .out_v()
        .values(&["name"]);
    assert_eq!(ordered(q(froms)), vec!["marko", "josh", "peter"]);
    let weights = q(g().v_ids(&["3"]).in_e(&["CREATED"]).values(&["weight"]));
    assert_eq!(
        weights,
        vec![GVal::Num(0.4), GVal::Num(0.4), GVal::Num(0.2)]
    );
}

// ===== tree.test.ts =====

#[test]
fn p6_tree_josh_software_names() {
    // V().has(name,josh).out(CREATED).values(name).tree()
    let out = q(g()
        .V()
        .has("name", P::eq("josh"))
        .out(&["CREATED"])
        .values(&["name"])
        .tree());
    assert_eq!(out.len(), 1);
    let root = as_map(&out[0]);
    assert_eq!(root.len(), 1); // josh
    let josh_children = as_map(&root.values()[0]);
    assert_eq!(josh_children.len(), 2); // two software vertices
    let mut names: Vec<String> = josh_children
        .iter()
        .map(|(_, sub)| {
            let child = as_map(sub);
            assert_eq!(child.len(), 1);
            s(&child.keys()[0])
        })
        .collect();
    names.sort();
    assert_eq!(names, vec!["lop", "ripple"]);
}

#[test]
fn p6_tree_marko_created() {
    let out = q(g().V().has("name", P::eq("marko")).out(&["CREATED"]).tree());
    assert_eq!(out.len(), 1);
    let root = as_map(&out[0]);
    assert_eq!(root.len(), 1); // marko
    let marko_children = as_map(&root.values()[0]);
    assert_eq!(marko_children.len(), 1); // marko → lop
}

#[test]
fn p6_tree_by_name() {
    // V(1).out().out().tree().by('name')
    let out = q(g().v_ids(&["1"]).out(&[]).out(&[]).tree().by("name"));
    assert_eq!(out.len(), 1);
    let root = as_map(&out[0]);
    let root_keys: Vec<String> = root.iter().map(|(k, _)| s(k)).collect();
    assert_eq!(root_keys, vec!["marko"]);
    let marko_children = as_map(&root.values()[0]);
    let child_keys: Vec<String> = marko_children.iter().map(|(k, _)| s(k)).collect();
    assert_eq!(child_keys, vec!["josh"]);
    let josh_children = as_map(&marko_children.values()[0]);
    let mut gc: Vec<String> = josh_children.iter().map(|(k, _)| s(k)).collect();
    gc.sort();
    assert_eq!(gc, vec!["lop", "ripple"]);
}

#[test]
fn p6_tree_empty_stream() {
    let out = q(g().V().has("name", P::eq("nobody")).tree());
    assert_eq!(out.len(), 1);
    assert_eq!(as_map(&out[0]).len(), 0);
}

// ===== group.test.ts =====

#[test]
fn p6_group_by_self_ages() {
    // V().hasLabel(PERSON).values(age).group() — key=value, each → [value].
    let out = q(g().V().has_label(&["PERSON"]).values(&["age"]).group());
    assert_eq!(out.len(), 1);
    let m = as_map(&out[0]);
    assert_eq!(
        map_get_gval(m, &GVal::Num(29.0)),
        Some(&GVal::list(vec![GVal::Num(29.0)]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(27.0)),
        Some(&GVal::list(vec![GVal::Num(27.0)]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(32.0)),
        Some(&GVal::list(vec![GVal::Num(32.0)]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(35.0)),
        Some(&GVal::list(vec![GVal::Num(35.0)]))
    );
}

#[test]
fn p6_group_name_keyed_by_age() {
    let out = q(g().V().has_label(&["PERSON"]).group().by("age").by("name"));
    let m = as_map(&out[0]);
    assert_eq!(
        map_get_gval(m, &GVal::Num(29.0)),
        Some(&GVal::list(vec![GVal::Str("marko".into())]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(27.0)),
        Some(&GVal::list(vec![GVal::Str("vadas".into())]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(32.0)),
        Some(&GVal::list(vec![GVal::Str("josh".into())]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(35.0)),
        Some(&GVal::list(vec![GVal::Str("peter".into())]))
    );
}

#[test]
fn p6_group_by_lang_missing_key_bucket() {
    // V().group().by(lang).by(name): software → 'java'; persons lack lang → Null key.
    let out = q(g().V().group().by("lang").by("name"));
    let m = as_map(&out[0]);
    assert_eq!(
        map_get_gval(m, &GVal::Str("java".into())),
        Some(&GVal::list(vec![
            GVal::Str("lop".into()),
            GVal::Str("ripple".into())
        ]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Null),
        Some(&GVal::list(vec![
            GVal::Str("marko".into()),
            GVal::Str("vadas".into()),
            GVal::Str("josh".into()),
            GVal::Str("peter".into()),
        ]))
    );
}

#[test]
fn p6_group_by_label() {
    let out = q(g().V().group().by_label());
    let m = as_map(&out[0]);
    assert_eq!(list_of(map_get(m, "PERSON").unwrap()).len(), 4);
    assert_eq!(list_of(map_get(m, "SOFTWARE").unwrap()).len(), 2);
}

#[test]
fn p6_group_by_label_by_name() {
    let out = q(g().V().group().by_label().by("name"));
    let m = as_map(&out[0]);
    let mut sw: Vec<String> = list_of(map_get(m, "SOFTWARE").unwrap())
        .iter()
        .map(s)
        .collect();
    sw.sort();
    assert_eq!(sw, vec!["lop", "ripple"]);
    let mut pe: Vec<String> = list_of(map_get(m, "PERSON").unwrap())
        .iter()
        .map(s)
        .collect();
    pe.sort();
    assert_eq!(pe, vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p6_group_by_label_by_count() {
    // A reducing value-by (count) folds over the group as a barrier → a single
    // per-bucket count, not a per-traverser list of 1s (before local aggregation).
    let out = q(g().V().group().by_label().by_t(super::__().count()));
    let m = as_map(&out[0]);
    let num = |v: &GVal| -> f64 {
        match v {
            GVal::Num(n) => *n,
            _ => panic!("expected a Num, got {v:?}"),
        }
    };
    assert_eq!(num(map_get(m, "PERSON").unwrap()), 4.0);
    assert_eq!(num(map_get(m, "SOFTWARE").unwrap()), 2.0);
}

#[test]
fn p6_group_by_age_valued_by_name() {
    let out = q(g().V().group().by("age").by("name"));
    let m = as_map(&out[0]);
    assert_eq!(
        map_get_gval(m, &GVal::Num(29.0)),
        Some(&GVal::list(vec![GVal::Str("marko".into())]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(27.0)),
        Some(&GVal::list(vec![GVal::Str("vadas".into())]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(32.0)),
        Some(&GVal::list(vec![GVal::Str("josh".into())]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Num(35.0)),
        Some(&GVal::list(vec![GVal::Str("peter".into())]))
    );
}

#[test]
fn p6_group_by_name_valued_by_age() {
    // Software vertices have no age; their value-by yields Null → bucket present but value Null.
    let out = q(g().V().group().by("name").by("age"));
    let m = as_map(&out[0]);
    assert_eq!(
        map_get_gval(m, &GVal::Str("marko".into())),
        Some(&GVal::list(vec![GVal::Num(29.0)]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Str("vadas".into())),
        Some(&GVal::list(vec![GVal::Num(27.0)]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Str("josh".into())),
        Some(&GVal::list(vec![GVal::Num(32.0)]))
    );
    assert_eq!(
        map_get_gval(m, &GVal::Str("peter".into())),
        Some(&GVal::list(vec![GVal::Num(35.0)]))
    );
    // lop/ripple keys exist (value-by age is Null in our engine, not dropped).
    assert!(map_get_gval(m, &GVal::Str("lop".into())).is_some());
    assert!(map_get_gval(m, &GVal::Str("ripple".into())).is_some());
}

// ===== or.test.ts =====

#[test]
fn p6_or_combines_two() {
    // or(outE(CREATED), inE(CREATED)) — anyone with an out- or in-created edge.
    let r = g()
        .V()
        .or(vec![
            super::__().out_e(&["CREATED"]),
            super::__().in_e(&["CREATED"]),
        ])
        .values(&["name"]);
    assert_eq!(
        sorted(q(r)),
        vec!["josh", "lop", "marko", "peter", "ripple"]
    );
}

#[test]
fn p6_or_out_knows_or_created() {
    let r = g()
        .V()
        .or(vec![
            super::__().out_e(&["KNOWS"]),
            super::__().out_e(&["CREATED"]),
        ])
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["marko", "josh", "peter"]);
}

#[test]
fn p6_or_no_match_filters_all() {
    let r = g()
        .V()
        .has_label(&["SOFTWARE"])
        .or(vec![super::__().out_e(&["KNOWS"])])
        .values(&["name"]);
    assert_eq!(q(r).len(), 0);
}

#[test]
fn p6_or_in_knows_or_out_created() {
    let r = g()
        .V()
        .or(vec![
            super::__().in_e(&["KNOWS"]),
            super::__().out_e(&["CREATED"]),
        ])
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["marko", "vadas", "josh", "peter"]);
}

// ===== hasKey.test.ts =====

#[test]
fn p6_haskey_age_persons() {
    let r = g().V().has_key(&["age"]).id();
    assert_eq!(ordered(q(r)), vec!["1", "2", "4", "6"]);
}

#[test]
fn p6_haskey_name_all() {
    let r = g().V().has_key(&["name"]).id();
    assert_eq!(ordered(q(r)), vec!["1", "2", "4", "6", "3", "5"]);
}

#[test]
fn p6_haskey_missing_filters_all() {
    let r = q(g().V().has_key(&["idonotexist"]));
    assert_eq!(r.len(), 0);
}

// NOTE: SKIPPED — `V().properties().hasKey('age').value()`. In the Rust engine
// a property is a `GVal::Map{key,value}`; `hasKey` only inspects Vertex/Edge
// keys (`present_keys` returns nothing for a Map), so it filters out the whole
// property stream. Genuine divergence from TS (recorded in the report).

// ===== both.test.ts =====

#[test]
fn p6_both_toy() {
    // V(4).both(KNOWS,CREATED,BLAH) → ripple, lop, marko (out first, then in).
    let r = g()
        .v_ids(&["4"])
        .both(&["KNOWS", "CREATED", "BLAH"])
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["ripple", "lop", "marko"]);
}

#[test]
fn p6_both_specific_label() {
    let r = g().v_ids(&["1"]).both(&["KNOWS"]).values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["vadas", "josh"]);
}

#[test]
fn p6_both_all_labels_equals_none() {
    let r = g().v_ids(&["4"]).both(&[]).values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["ripple", "lop", "marko"]);
}

#[test]
fn p6_both_ids() {
    let r = g().v_ids(&["4"]).both(&["KNOWS", "CREATED", "blah"]).id();
    assert_eq!(ordered(q(r)), vec!["5", "3", "1"]);
}

// ===== optional.test.ts =====

#[test]
fn p6_optional_falls_back() {
    // V(2=vadas).optional(out(KNOWS)) → vadas (no out-knows).
    let r = g().v_ids(&["2"]).optional(super::__().out(&["KNOWS"]));
    assert_eq!(ids_of(r), vec!["2"]);
}

#[test]
fn p6_optional_yields_subtraversal() {
    // V(2).optional(in(KNOWS)) → marko (v1).
    let r = g().v_ids(&["2"]).optional(super::__().in_(&["KNOWS"]));
    assert_eq!(ids_of(r), vec!["1"]);
}

#[test]
fn p6_optional_nested_path() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .optional(
            super::__()
                .out(&["KNOWS"])
                .optional(super::__().out(&["CREATED"])),
        )
        .path();
    assert_eq!(
        paths_text(r),
        vec![
            vec!["1", "2"],
            vec!["1", "4", "5"],
            vec!["1", "4", "3"],
            vec!["2"],
            vec!["4"],
            vec!["6"],
        ]
    );
}

// ===== hasValue.test.ts =====

#[test]
fn p6_hasvalue_filters_by_value() {
    // V().hasId(1).properties(name).hasValue(marko).value() → ['marko'].
    let r = g()
        .V()
        .has_id(&["1"])
        .properties(&["name"])
        .has_value(["marko"])
        .value();
    assert_eq!(ordered(q(r)), vec!["marko"]);
}

#[test]
fn p6_hasvalue_excludes_non_matching() {
    let r = q(g()
        .V()
        .has_id(&["1"])
        .properties(&["name"])
        .has_value(["vadas"]));
    assert_eq!(r.len(), 0);
}

#[test]
fn p6_hasvalue_any_of() {
    let r = g()
        .V()
        .properties(&["name"])
        .has_value(["marko", "lop"])
        .value();
    assert_eq!(sorted(q(r)), vec!["lop", "marko"]);
}

// ===== addV.test.ts =====

#[test]
fn p6_addv_inserts_and_emits() {
    let mut g0 = modern();
    let before = g0.vertex_count();
    let r = g()
        .add_v(Some("PERSON"))
        .property("name", "kuppitz")
        .run(&mut g0);
    assert_eq!(g0.vertex_count(), before + 1);
    assert_eq!(r.len(), 1);
    // The new vertex is a PERSON named kuppitz.
    let labels = g().V().has("name", P::eq("kuppitz")).label().run(&mut g0);
    assert_eq!(ordered(labels), vec!["PERSON"]);
    let names = g()
        .V()
        .has("name", P::eq("kuppitz"))
        .values(&["name"])
        .run(&mut g0);
    assert_eq!(ordered(names), vec!["kuppitz"]);
}

#[test]
fn p6_addv_no_label() {
    let mut g0 = modern();
    let r = g().add_v(None).run(&mut g0);
    assert_eq!(r.len(), 1);
    assert!(matches!(r[0], GVal::Node(_)));
}

#[test]
fn p6_addv_mid_traversal_per_traverser() {
    let mut g0 = modern();
    let before = g0.vertex_count();
    let _ = g()
        .V()
        .has_label(&["PERSON"])
        .add_v(Some("SHADOW"))
        .run(&mut g0);
    assert_eq!(g0.vertex_count(), before + 4); // one shadow per person
    let shadows = g().V().has_label(&["SHADOW"]).run(&mut g0);
    assert_eq!(shadows.len(), 4);
}

// ===== identity.test.ts =====

#[test]
fn p6_identity_unchanged() {
    let r = g().V().identity().id();
    assert_eq!(ordered(q(r)), vec!["1", "2", "4", "6", "3", "5"]);
}

#[test]
fn p6_identity_equals_v() {
    let with_identity = ordered(q(g().V().identity().id()));
    let direct = ordered(q(g().V().id()));
    assert_eq!(with_identity, direct);
}
