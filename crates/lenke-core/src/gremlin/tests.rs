//! Gremlin executor tests over the canonical TinkerPop "Modern" graph, porting
//! the TS `@lenke/gremlin` suites (social / filters / aggregations / paths /
//! repeat / select-as / software-creators) plus the modulator / projection /
//! side-effect / mutation features. TS closures are expressed as sub-traversals.

use super::{g, GVal, Order, Pop, Token, __, P};
use crate::graph::{Graph, Value};
use crate::ndjson;

/// The shared TinkerPop "Modern" fixture (see `crate::fixtures`).
fn modern() -> Graph {
    crate::fixtures::modern_gremlin()
}

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

fn one_num(r: Vec<GVal>) -> f64 {
    match r.as_slice() {
        [GVal::Num(n)] => *n,
        _ => panic!("expected single number, got {r:?}"),
    }
}

/// Sort a result map's entries by string key for deterministic assertions.
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

fn list_names(g: &GVal) -> Vec<String> {
    match g {
        GVal::List(items) => names(items.to_vec()),
        _ => panic!("expected list, got {g:?}"),
    }
}

// ===== sources / movement =====

#[test]
fn v_all_and_count() {
    assert_eq!(one_num(q(g().V().count())), 6.0);
}

#[test]
fn v_by_id() {
    assert_eq!(names(q(g().v_ids(&["1"]).values(&["name"]))), vec!["marko"]);
}

#[test]
fn repeat_default_cap_matches_ts_100() {
    // A directed 5-cycle a→b→c→d→e→a. `repeat(out())` with no `times()` runs the
    // default iteration cap; emit() (post-form) fires once per iteration and the
    // frontier stays size 1, so the emitted count is the cap — which must be 100
    // (the TS engine's cap), not the old 64.
    let lines = [
        r#"{"type":"node","id":"a","labels":["V"],"properties":{}}"#,
        r#"{"type":"node","id":"b","labels":["V"],"properties":{}}"#,
        r#"{"type":"node","id":"c","labels":["V"],"properties":{}}"#,
        r#"{"type":"node","id":"d","labels":["V"],"properties":{}}"#,
        r#"{"type":"node","id":"e","labels":["V"],"properties":{}}"#,
        r#"{"type":"edge","from":"a","to":"b","labels":["E"],"properties":{}}"#,
        r#"{"type":"edge","from":"b","to":"c","labels":["E"],"properties":{}}"#,
        r#"{"type":"edge","from":"c","to":"d","labels":["E"],"properties":{}}"#,
        r#"{"type":"edge","from":"d","to":"e","labels":["E"],"properties":{}}"#,
        r#"{"type":"edge","from":"e","to":"a","labels":["E"],"properties":{}}"#,
    ];
    let mut g = crate::ndjson::decode(&lines.join("\n")).unwrap();
    let r = super::parse("g.V('a').repeat(__.out()).emit().count()")
        .unwrap()
        .run(&mut g);
    assert_eq!(r, vec![GVal::Num(100.0)]);
}

#[test]
fn e_by_id_resolves_directly_in_id_order() {
    // E(ids) resolves each id directly (like V(ids)) and yields per requested id in
    // id order — mirroring the TS engine — not a full edge scan in edge-index order.
    let lines = [
        r#"{"type":"node","id":"1","labels":["V"],"properties":{}}"#,
        r#"{"type":"node","id":"2","labels":["V"],"properties":{}}"#,
        r#"{"type":"edge","id":"e-a","from":"1","to":"2","labels":["E"],"properties":{}}"#,
        r#"{"type":"edge","id":"e-b","from":"2","to":"1","labels":["E"],"properties":{}}"#,
    ];
    let mut g = crate::ndjson::decode(&lines.join("\n")).unwrap();
    let r = super::parse("g.E('e-b','e-a').id()").unwrap().run(&mut g);
    let ids: Vec<String> = r
        .iter()
        .map(|v| match v {
            GVal::Str(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(ids, vec!["e-b", "e-a"]);
}

#[test]
fn out_multi_label_order_matters() {
    assert_eq!(
        ordered(q(g()
            .v_ids(&["1"])
            .out(&["CREATED", "KNOWS"])
            .values(&["name"]))),
        vec!["lop", "vadas", "josh"]
    );
}

#[test]
fn out_all_neighbors_of_marko() {
    assert_eq!(
        names(q(g().v_ids(&["1"]).out(&[]).values(&["name"]))),
        vec!["josh", "lop", "vadas"]
    );
}

#[test]
fn oute_inv_equals_out() {
    let a = names(q(g().v_ids(&["1"]).out(&["KNOWS"]).values(&["name"])));
    let b = names(q(g()
        .v_ids(&["1"])
        .out_e(&["KNOWS"])
        .in_v()
        .values(&["name"])));
    assert_eq!(a, b);
    assert_eq!(a, vec!["josh", "vadas"]);
}

#[test]
fn in_created_creators_of_lop() {
    assert_eq!(
        names(q(g()
            .V()
            .has("name", P::eq("lop"))
            .in_(&["CREATED"])
            .values(&["name"]))),
        vec!["josh", "marko", "peter"]
    );
}

#[test]
fn both_neighborhood() {
    assert_eq!(
        names(q(g().v_ids(&["1"]).both(&[]).dedup().values(&["name"]))),
        vec!["josh", "lop", "vadas"]
    );
}

#[test]
fn edge_source_and_count() {
    assert_eq!(one_num(q(g().E().count())), 6.0);
}

#[test]
fn other_v_from_marko_edges() {
    // marko's incident edges, otherV back from marko ⇒ the far endpoints.
    assert_eq!(
        names(q(g().v_ids(&["1"]).both_e(&[]).other_v().values(&["name"]))),
        vec!["josh", "lop", "vadas"]
    );
}

// ===== filters / predicates =====

#[test]
fn has_age_gt_30() {
    assert_eq!(
        names(q(g().V().has("age", P::gt(30)).values(&["name"]))),
        vec!["josh", "peter"]
    );
}

#[test]
fn between_inside_outside() {
    assert_eq!(
        names(q(g().V().has("age", P::between(28, 33)).values(&["name"]))),
        vec!["josh", "marko"]
    );
    assert_eq!(
        names(q(g().V().has("age", P::inside(27, 32)).values(&["name"]))),
        vec!["marko"]
    );
    assert_eq!(
        names(q(g().V().has("age", P::outside(28, 33)).values(&["name"]))),
        vec!["peter", "vadas"]
    );
}

#[test]
fn within_without() {
    assert_eq!(
        names(q(g()
            .V()
            .has("name", P::within(["josh", "marko"]))
            .values(&["name"]))),
        vec!["josh", "marko"]
    );
    assert_eq!(
        names(q(g()
            .V()
            .has_label(&["PERSON"])
            .has("name", P::without(["josh", "marko"]))
            .values(&["name"]))),
        vec!["peter", "vadas"]
    );
}

#[test]
fn text_predicates() {
    assert_eq!(
        names(q(g()
            .V()
            .has("name", P::starts_with("ma"))
            .values(&["name"]))),
        vec!["marko"]
    );
    assert_eq!(
        names(q(g().V().has("name", P::containing("o")).values(&["name"]))),
        vec!["josh", "lop", "marko"]
    );
}

#[test]
fn has_id_and_has_not() {
    assert_eq!(
        names(q(g().V().has_id(&["1"]).values(&["name"]))),
        vec!["marko"]
    );
    // hasNot('age') keeps software (no age property).
    assert_eq!(
        names(q(g().V().has_not(&["age"]).values(&["name"]))),
        vec!["lop", "ripple"]
    );
}

#[test]
fn has_key_keeps_elements_with_property() {
    assert_eq!(
        names(q(g().V().has_key(&["lang"]).values(&["name"]))),
        vec!["lop", "ripple"]
    );
}

#[test]
fn software_has_no_age() {
    assert_eq!(
        q(g().V().has_label(&["SOFTWARE"]).values(&["age"])).len(),
        0
    );
}

// ===== combinators (closures → sub-traversals) =====

#[test]
fn and_knows_out_and_young() {
    let r = g()
        .V()
        .and(vec![
            __().out_e(&["KNOWS"]),
            __().values(&["age"]).is(P::lt(30)),
        ])
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["marko"]);
}

#[test]
fn or_created_out_or_many_creators() {
    let r = g()
        .V()
        .or(vec![
            __().out_e(&["CREATED"]),
            __().in_(&["CREATED"]).count().is(P::gt(1)),
        ])
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["josh", "lop", "marko", "peter"]);
}

#[test]
fn not_created_more_than_one() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .not(__().out(&["CREATED"]).count().is(P::gt(1)))
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["marko", "peter", "vadas"]);
}

#[test]
fn chained_where_no_created_has_knows_in() {
    let r = g()
        .V()
        .where_(__().not(__().out(&["CREATED"])))
        .where_(__().in_(&["KNOWS"]))
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["vadas"]);
}

#[test]
fn where_count_is_gte_2() {
    let r = g()
        .V()
        .where_(__().in_(&["CREATED"]).count().is(P::gte(2)))
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["lop"]);
}

#[test]
fn marko_friends_who_created() {
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .out(&["KNOWS"])
        .where_(__().out(&["CREATED"]))
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["josh"]);
}

#[test]
fn coalesce_first_nonempty() {
    // coalesce(out CREATED names, constant 'none') per person.
    let r = g().V().has_label(&["PERSON"]).coalesce(vec![
        __().out(&["CREATED"]).values(&["name"]),
        __().constant("none"),
    ]);
    // marko→lop, josh→{lop,ripple}, peter→lop, vadas→none
    assert_eq!(names(q(r)), vec!["lop", "lop", "lop", "none", "ripple"]);
}

#[test]
fn optional_falls_back_to_input() {
    let r = g()
        .V()
        .has("name", P::eq("vadas"))
        .optional(__().out(&["CREATED"]));
    // vadas creates nothing ⇒ optional yields vadas itself.
    assert_eq!(names(q(r.values(&["name"]))), vec!["vadas"]);
}

#[test]
fn choose_branches_on_label() {
    let r = g().V().choose_else(
        __().has_label(&["PERSON"]),
        __().values(&["name"]),
        __().constant("sw"),
    );
    assert_eq!(
        names(q(r)),
        vec!["josh", "marko", "peter", "sw", "sw", "vadas"]
    );
}

#[test]
fn union_name_and_age() {
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .union(vec![__().values(&["name"]), __().values(&["age"])]);
    let out = q(r);
    assert_eq!(out, vec![GVal::Str("marko".into()), GVal::Num(29.0)]);
}

#[test]
fn local_out_count_per_person() {
    // local(out().count()) counts each person's out-degree per traverser.
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .local(__().out(&[]).count());
    assert_eq!(one_num(q(r)), 3.0);
}

// ===== aggregates / by modulators =====

#[test]
fn group_count_by_label() {
    let out = q(g().V().group_count().by_label());
    let m = map_sorted(&out[0]);
    assert_eq!(
        m,
        vec![
            ("PERSON".into(), GVal::Num(4.0)),
            ("SOFTWARE".into(), GVal::Num(2.0))
        ]
    );
}

#[test]
fn group_names_by_label() {
    let out = q(g().V().group().by_label().by("name"));
    let m = map_sorted(&out[0]);
    assert_eq!(m[0].0, "PERSON");
    assert_eq!(list_names(&m[0].1), vec!["josh", "marko", "peter", "vadas"]);
    assert_eq!(m[1].0, "SOFTWARE");
    assert_eq!(list_names(&m[1].1), vec!["lop", "ripple"]);
}

#[test]
fn group_count_by_age_value() {
    let out = q(g()
        .V()
        .has_label(&["PERSON"])
        .values(&["age"])
        .group_count());
    let m = map_sorted(&out[0]);
    assert_eq!(m.len(), 4);
    assert!(m.iter().all(|(_, n)| *n == GVal::Num(1.0)));
}

#[test]
fn group_software_by_lang() {
    let out = q(g()
        .V()
        .has_label(&["SOFTWARE"])
        .group()
        .by("lang")
        .by("name"));
    let m = map_sorted(&out[0]);
    assert_eq!(m.len(), 1);
    assert_eq!(m[0].0, "java");
    assert_eq!(list_names(&m[0].1), vec!["lop", "ripple"]);
}

#[test]
fn group_count_edges_by_label() {
    let out = q(g().V().out_e(&[]).group_count().by_label());
    let m = map_sorted(&out[0]);
    assert_eq!(
        m,
        vec![
            ("CREATED".into(), GVal::Num(4.0)),
            ("KNOWS".into(), GVal::Num(2.0))
        ]
    );
}

#[test]
fn order_by_age_desc() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .order_by("age", Order::Desc)
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["peter", "josh", "marko", "vadas"]);
}

#[test]
fn order_by_name_asc() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .order()
        .by("name")
        .values(&["name"]);
    assert_eq!(ordered(q(r)), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn sum_mean_max_min_of_age() {
    assert_eq!(
        one_num(q(g().V().has_label(&["PERSON"]).values(&["age"]).sum())),
        123.0
    );
    assert_eq!(
        one_num(q(g().V().has_label(&["PERSON"]).values(&["age"]).mean())),
        30.75
    );
    assert_eq!(
        one_num(q(g().V().has_label(&["PERSON"]).values(&["age"]).max())),
        35.0
    );
    assert_eq!(
        one_num(q(g().V().has_label(&["PERSON"]).values(&["age"]).min())),
        27.0
    );
}

#[test]
fn fold_then_local_count() {
    // fold to one list, then local count of its length.
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .values(&["name"])
        .fold()
        .count_local();
    assert_eq!(one_num(q(r)), 4.0);
}

// ===== project =====

#[test]
fn project_name_and_created_count() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .project(&["name", "created"])
        .by("name")
        .by_t(__().out_e(&["CREATED"]).count());
    let out = q(r);
    // Per person, a map {name, created}.
    let mut got: Vec<(String, f64)> = out
        .iter()
        .map(|g| {
            let m = match g {
                GVal::Map(e) => e,
                _ => panic!(),
            };
            let name = s(&m[0].1);
            let created = match m[1].1 {
                GVal::Num(n) => n,
                _ => panic!(),
            };
            (name, created)
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        vec![
            ("josh".into(), 2.0),
            ("marko".into(), 1.0),
            ("peter".into(), 1.0),
            ("vadas".into(), 0.0)
        ]
    );
}

#[test]
fn project_marko_degrees() {
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .project(&["id", "out", "in"])
        .by_id()
        .by_t(__().out_e(&[]).count())
        .by_t(__().in_e(&[]).count());
    let out = q(r);
    let m = match &out[0] {
        GVal::Map(e) => e,
        _ => panic!(),
    };
    assert_eq!(s(&m[0].1), "1");
    assert_eq!(m[1].1, GVal::Num(3.0)); // out-degree
    assert_eq!(m[2].1, GVal::Num(0.0)); // in-degree
}

// ===== select / as =====

#[test]
fn select_three_labels() {
    let r = g()
        .V()
        .as_("a")
        .out(&[])
        .as_("b")
        .out(&[])
        .as_("c")
        .select(&["a", "b", "c"])
        .by_id()
        .by_id()
        .by_id();
    let out = q(r);
    // marko→josh→{ripple,lop}; map of ids.
    let maps: Vec<Vec<(String, String)>> = out
        .iter()
        .map(|g| match g {
            GVal::Map(e) => e.iter().map(|(k, v)| (s(k), s(v))).collect(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(maps.len(), 2);
    assert!(maps
        .iter()
        .all(|m| m[0] == ("a".to_string(), "1".to_string())
            && m[1] == ("b".to_string(), "4".to_string())));
}

#[test]
fn select_single_label_unwraps() {
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .as_("a")
        .out(&["KNOWS"])
        .select(&["a"])
        .values(&["name"]);
    // 'a' recalls marko for each of the two friends ⇒ ['marko','marko'].
    assert_eq!(names(q(r)), vec!["marko", "marko"]);
}

#[test]
fn select_by_name() {
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .as_("a")
        .out(&["CREATED"])
        .as_("b")
        .select(&["a", "b"])
        .by("name")
        .by("name");
    let out = q(r);
    let m = match &out[0] {
        GVal::Map(e) => e,
        _ => panic!(),
    };
    assert_eq!(s(&m[0].1), "marko");
    assert_eq!(s(&m[1].1), "lop");
}

#[test]
fn where_key_compares_tags() {
    // Pairs (a, b) where both are persons and b is older than a, via tag compare.
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .as_("a")
        .out(&["KNOWS"])
        .as_("b")
        .where_key("a", P::lt(GVal::Str("b".into())))
        .by("age")
        .by("age")
        .select(&["b"])
        .values(&["name"]);
    // a=marko(29); b in {vadas(27), josh(32)}; keep where a.age < b.age ⇒ josh.
    assert_eq!(names(q(r)), vec!["josh"]);
}

// ===== paths =====

#[test]
fn path_by_name_two_hops() {
    let r = g().V().out(&[]).out(&[]).path().by("name");
    let out = q(r);
    let mut paths: Vec<Vec<String>> = out.iter().map(list_names_ordered).collect();
    paths.sort();
    assert_eq!(
        paths,
        vec![
            vec!["marko", "josh", "lop"],
            vec!["marko", "josh", "ripple"]
        ]
    );
}

#[test]
fn simple_path_excludes_cycle() {
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .out(&["CREATED"])
        .in_(&["CREATED"])
        .simple_path()
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["josh", "peter"]);
}

#[test]
fn cyclic_path_retains_cycle() {
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .out(&["CREATED"])
        .in_(&["CREATED"])
        .cyclic_path()
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["marko"]);
}

#[test]
fn tree_from_marko() {
    let out = q(g().v_ids(&["1"]).out(&["KNOWS"]).tree());
    // root → marko → {vadas, josh} → {}
    let m = map_sorted(&out[0]);
    assert_eq!(m.len(), 1); // single root: marko (id 1)
    let marko_children = map_sorted(&m[0].1);
    assert_eq!(marko_children.len(), 2);
}

// ===== repeat =====

#[test]
fn repeat_times_two() {
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .times(2)
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["lop", "ripple"]);
}

#[test]
fn repeat_until_software() {
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .until(__().has_label(&["SOFTWARE"]))
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["lop", "lop", "ripple"]);
}

#[test]
fn repeat_times_emit() {
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .times(2)
        .emit_all()
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["josh", "lop", "lop", "ripple", "vadas"]);
}

#[test]
fn repeat_emit_filtered() {
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .times(2)
        .emit(__().has("lang", P::eq("java")))
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["lop", "lop", "ripple"]);
}

#[test]
fn repeat_times_one_equals_out() {
    let r = g()
        .v_ids(&["1"])
        .repeat(__().out(&[]))
        .times(1)
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["josh", "lop", "vadas"]);
}

// ===== cardinality / scope =====

#[test]
fn limit_and_range() {
    assert_eq!(q(g().V().limit(2)).len(), 2);
    assert_eq!(q(g().V().range(1, 3)).len(), 2);
    assert_eq!(q(g().V().tail(2)).len(), 2);
}

#[test]
fn local_range_on_fold() {
    // fold all names then take first 2 locally.
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .order()
        .by("name")
        .values(&["name"])
        .fold()
        .range_local(0, 2);
    let out = q(r);
    assert_eq!(list_names_ordered(&out[0]), vec!["josh", "marko"]);
}

// ===== misc / projection =====

#[test]
fn value_map_of_marko() {
    let out = q(g()
        .V()
        .has("name", P::eq("marko"))
        .value_map(&["name", "age"]));
    let m = match &out[0] {
        GVal::Map(e) => e,
        _ => panic!(),
    };
    assert!(m.contains(&(GVal::Str("name".into()), GVal::Str("marko".into()))));
    assert!(m.contains(&(GVal::Str("age".into()), GVal::Num(29.0))));
}

#[test]
fn element_map_of_marko_includes_id_label() {
    let out = q(g().V().has("name", P::eq("marko")).element_map(&["name"]));
    let m = map_sorted(&out[0]);
    assert!(m.iter().any(|(k, v)| k == "id" && s(v) == "1"));
    assert!(m.iter().any(|(k, v)| k == "label" && s(v) == "PERSON"));
    assert!(m.iter().any(|(k, v)| k == "name" && s(v) == "marko"));
}

#[test]
fn id_and_label_steps() {
    assert_eq!(names(q(g().v_ids(&["1"]).id())), vec!["1"]);
    assert_eq!(names(q(g().v_ids(&["1"]).label())), vec!["PERSON"]);
}

#[test]
fn unfold_a_folded_list() {
    let r = g()
        .V()
        .has_label(&["SOFTWARE"])
        .values(&["name"])
        .fold()
        .unfold();
    assert_eq!(names(q(r)), vec!["lop", "ripple"]);
}

#[test]
fn constant_and_inject() {
    assert_eq!(
        names(q(g().V().has("name", P::eq("marko")).constant("hi"))),
        vec!["hi"]
    );
    let r = g().inject([1, 2, 3]);
    assert_eq!(q(r), vec![GVal::Num(1.0), GVal::Num(2.0), GVal::Num(3.0)]);
}

// ===== side effects =====

#[test]
fn aggregate_then_cap() {
    let r = g()
        .V()
        .has_label(&["PERSON"])
        .values(&["name"])
        .aggregate("names")
        .cap("names");
    let out = q(r);
    assert_eq!(list_names(&out[0]), vec!["josh", "marko", "peter", "vadas"]);
}

// ===== edge properties =====

#[test]
fn strong_knows_edges() {
    let r = g()
        .V()
        .out_e(&["KNOWS"])
        .has("weight", P::gt(0.75))
        .in_v()
        .values(&["name"]);
    assert_eq!(names(q(r)), vec!["josh"]);
}

#[test]
fn marko_created_edge_weight() {
    assert_eq!(
        q(g().v_ids(&["1"]).out_e(&["CREATED"]).values(&["weight"])),
        vec![GVal::Num(0.4)]
    );
}

// ===== mutation =====

#[test]
fn add_vertex_and_property() {
    let mut g0 = modern();
    let r = g()
        .add_v(Some("PERSON"))
        .property("name", "newbie")
        .property("age", 40)
        .values(&["name"])
        .run(&mut g0);
    assert_eq!(names(r), vec!["newbie"]);
    // The new vertex is queryable.
    assert_eq!(
        one_num(g().V().has("name", P::eq("newbie")).count().run(&mut g0)),
        1.0
    );
}

// ===== null is a first-class property value (deliberate TinkerPop divergence) =====
// TinkerPop disallows null property values; here `property(k, null)` STORES a
// present null — visible in values/valueMap, and has(k) is true. It's a present
// property distinct from an absent one. Deleting a property is a SEPARATE op:
// `.properties(k).drop()` (see `properties_drop_removes_the_property`).

#[test]
fn property_set_to_null_is_stored_and_visible_not_removed() {
    let mut g0 = modern();
    // marko gets a present-null `nick`.
    let _ = g()
        .V()
        .has("name", P::eq("marko"))
        .property("nick", GVal::Null)
        .run(&mut g0);

    // values('nick') yields a present Null — not nothing.
    let vals = g()
        .V()
        .has("name", P::eq("marko"))
        .values(&["nick"])
        .run(&mut g0);
    assert_eq!(
        vals,
        vec![GVal::Null],
        "a present null is projected, not dropped"
    );

    // has(key) (existence) is true for a present null.
    assert_eq!(
        one_num(
            g().V()
                .has("name", P::eq("marko"))
                .has_key(&["nick"])
                .count()
                .run(&mut g0)
        ),
        1.0,
        "has(key) is true for a present null"
    );

    // valueMap() carries the nick=null entry.
    let vm = g()
        .V()
        .has("name", P::eq("marko"))
        .value_map(&["nick"])
        .run(&mut g0);
    assert_eq!(vm, vec![GVal::Map(vec![(GVal::from("nick"), GVal::Null)])]);
}

#[test]
fn properties_drop_removes_the_property() {
    // The Gremlin-native way to DELETE a property (since `property(k, null)` now
    // stores a null): traverse to the property element and `.drop()` it.
    let mut g0 = modern();
    let _ = g()
        .V()
        .has("name", P::eq("marko"))
        .property("nick", GVal::Null)
        .run(&mut g0);
    assert_eq!(
        one_num(
            g().V()
                .has("name", P::eq("marko"))
                .has_key(&["nick"])
                .count()
                .run(&mut g0)
        ),
        1.0
    );

    // .properties('nick').drop() removes the (present-null) property outright.
    let _ = g()
        .V()
        .has("name", P::eq("marko"))
        .properties(&["nick"])
        .drop()
        .run(&mut g0);
    assert_eq!(
        g().V()
            .has("name", P::eq("marko"))
            .values(&["nick"])
            .run(&mut g0),
        Vec::<GVal>::new(),
        "the property is gone after drop"
    );
    assert_eq!(
        one_num(
            g().V()
                .has("name", P::eq("marko"))
                .has_key(&["nick"])
                .count()
                .run(&mut g0)
        ),
        0.0,
        "has(key) is false after drop"
    );

    // A real-valued property drops the same way.
    let _ = g()
        .V()
        .has("name", P::eq("marko"))
        .properties(&["age"])
        .drop()
        .run(&mut g0);
    assert_eq!(
        g().V()
            .has("name", P::eq("marko"))
            .values(&["age"])
            .run(&mut g0),
        Vec::<GVal>::new()
    );
}

#[test]
fn dedup_by_label_dedupes_on_the_tagged_value() {
    // marko's two KNOWS-neighbors (vadas, josh) both carry a='marko', so
    // dedup('a') collapses them to ONE. The old code dropped the label arg and
    // deduped by the current value (vadas != josh) → kept both.
    let mut g = modern();
    let by_label =
        super::parse("g.V().as('a').out('KNOWS').dedup('a').select('a').values('name')").unwrap();
    assert_eq!(by_label.run(&mut g), vec![GVal::Str("marko".into())]);

    let mut g2 = modern();
    let by_value =
        super::parse("g.V().as('a').out('KNOWS').dedup().select('a').values('name')").unwrap();
    assert_eq!(by_value.run(&mut g2).len(), 2); // dedup-by-value keeps both
}

#[test]
fn drop_cannot_be_spoofed_by_a_project_map() {
    // Regression: a `project('key')` result is a Map with a `key` entry; it must
    // NOT be mistaken for a property element by drop() (that would delete an
    // arbitrary property). The owner now rides the `Property` element itself, so
    // a Map can never spoof one. Before the fix this deleted `age` everywhere.
    let mut g = modern();
    let t = super::parse("g.V().project('key').by(constant('age')).drop()").unwrap();
    let _ = t.run(&mut g);
    // All four PERSON vertices keep their age — nothing was deleted.
    let ages = super::parse("g.V().values('age').count()").unwrap();
    assert_eq!(one_num(ages.run(&mut g)), 4.0);
}

#[test]
fn add_edge_between_tagged() {
    let mut g0 = modern();
    // marko --LIKES--> ripple
    let _ = g()
        .V()
        .has("name", P::eq("marko"))
        .as_("a")
        .V()
        .has("name", P::eq("ripple"))
        .add_e("LIKES")
        .from_tag("a")
        .run(&mut g0);
    let r = g()
        .V()
        .has("name", P::eq("marko"))
        .out(&["LIKES"])
        .values(&["name"])
        .run(&mut g0);
    assert_eq!(names(r), vec!["ripple"]);
}

#[test]
fn drop_removes_vertex() {
    let mut g0 = modern();
    let _ = g().V().has("name", P::eq("vadas")).drop().run(&mut g0);
    assert_eq!(one_num(g().V().count().run(&mut g0)), 5.0);
}

#[test]
fn group_count_by_token_label() {
    // by_token(Token::Label) is equivalent to by_label().
    let out = q(g().V().group_count().by_token(Token::Label));
    let m = map_sorted(&out[0]);
    assert_eq!(
        m,
        vec![
            ("PERSON".into(), GVal::Num(4.0)),
            ("SOFTWARE".into(), GVal::Num(2.0))
        ]
    );
}

#[test]
fn select_pop_first_vs_last() {
    // Tag 'a' twice (marko, then the friend); first/last pick different ends.
    let first = g()
        .v_ids(&["1"])
        .as_("a")
        .out(&["KNOWS"])
        .as_("a")
        .select_pop(Pop::First, &["a"])
        .values(&["name"]);
    assert_eq!(names(q(first)), vec!["marko", "marko"]);
    let last = g()
        .v_ids(&["1"])
        .as_("a")
        .out(&["KNOWS"])
        .as_("a")
        .select_pop(Pop::Last, &["a"])
        .values(&["name"]);
    assert_eq!(names(q(last)), vec!["josh", "vadas"]);
}

// ===== textual Gremlin parser =====

/// Parse a Gremlin string, run it, return result values.
fn qs(query: &str) -> Vec<GVal> {
    let mut g = modern();
    let t = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

#[test]
fn sack_folds_and_reads_the_default() {
    // withSack(init) + sack(op).by(proj) merges into the sack; sack() reads it.
    // marko's age is 29.
    assert_eq!(
        qs("g.withSack(100).V().has('name','marko').sack(sum).by('age').sack()"),
        vec![GVal::Num(129.0)]
    );
    assert_eq!(
        qs("g.withSack(0).V().has('name','marko').sack(assign).by('age').sack()"),
        vec![GVal::Num(29.0)]
    );
    // A read before any write returns the withSack default (no sack stored).
    assert_eq!(
        qs("g.withSack(7).V().has('name','marko').sack()"),
        vec![GVal::Num(7.0)]
    );
}

#[test]
fn sack_without_with_sack_faults() {
    // sack() with no preceding withSack() is a usage error, not a silent empty.
    let mut g = modern();
    let t = super::parse("g.V().sack()").unwrap();
    assert_eq!(
        super::try_run(&mut g, &t).unwrap_err().code,
        crate::error_codes::ErrorCode::InvalidGraphOp
    );
}

/// The text dialect can express and compare temporal literals — `date(...)`,
/// `datetime(...)`, `duration(...)` — so an as-of predicate is writable.
///
/// Regression, two independent gaps that had to be closed together: the grammar
/// had no temporal constructor (every spelling was `E_SYNTAX`, so the literal was
/// inexpressible), and `gcmp` had no `Temporal` arm (so once it parsed, ordering
/// still raised `E_INVALID_VALUE`). Between them, a bitemporal query — the
/// `vf <= t` slice every as-of read needs — could not be written on this dialect
/// at all, on the engine the perf guidance recommends as the default.
#[test]
fn text_dialect_temporal_literals_and_ordering() {
    let mut g = ndjson::decode(
        &[
            r#"{"type":"node","id":"a","labels":["V"],"properties":{"vf":{"@date":"2020-01-01"},"n":1}}"#,
            r#"{"type":"node","id":"b","labels":["V"],"properties":{"vf":{"@date":"2022-06-15"},"n":2}}"#,
        ]
        .join("\n"),
    )
    .unwrap();
    let run = |g: &mut Graph, q: &str| -> Vec<GVal> {
        super::parse(q)
            .unwrap_or_else(|e| panic!("parse `{q}`: {e}"))
            .run(g)
    };
    let nums = |vs: Vec<GVal>| -> Vec<f64> {
        vs.into_iter()
            .filter_map(|v| match v {
                GVal::Num(n) => Some(n),
                _ => None,
            })
            .collect()
    };

    // Equality against a DATE literal.
    assert_eq!(
        nums(run(
            &mut g,
            "g.V().has('vf', date('2020-01-01')).values('n')"
        )),
        vec![1.0]
    );
    // Ordering — the as-of slice shape.
    assert_eq!(
        nums(run(
            &mut g,
            "g.V().has('vf', lte(date('2021-01-01'))).values('n')"
        )),
        vec![1.0]
    );
    assert_eq!(
        nums(run(
            &mut g,
            "g.V().has('vf', gt(date('2021-01-01'))).values('n')"
        )),
        vec![2.0]
    );
    assert_eq!(
        nums(run(
            &mut g,
            "g.V().has('vf', between(date('2019-01-01'), date('2021-01-01'))).values('n')"
        )),
        vec![1.0]
    );
    // A duration literal parses (durations are deliberately NOT relationally
    // ordered, so this only asserts the grammar accepts the constructor).
    assert!(super::parse("g.V().has('d', duration('P1D')).values('n')").is_ok());
}

#[test]
fn cf_where_pred_and_order_local_column() {
    // `where(neq('me'))`: among lop's co-creators (marko, josh, peter), exclude the
    // seed marko → josh, peter.
    assert_eq!(
        names(qs(
            "g.V('1').as('me').out('CREATED').in('CREATED').where(neq('me')).values('name')"
        )),
        vec!["josh", "peter"]
    );

    // `order(local).by(values, desc).select(Column.keys)` ranks a groupCount Map by
    // descending count: lop (created by 3) outranks ripple (created by 1).
    let ranked = qs(
        "g.V().out('CREATED').groupCount().by('name').order(Scope.local).by(values, desc).select(Column.keys)",
    );
    match &ranked[..] {
        [GVal::List(items)] => assert_eq!(
            items.iter().map(s).collect::<Vec<_>>(),
            vec!["lop".to_string(), "ripple".to_string()]
        ),
        other => panic!("expected one ranked list, got {other:?}"),
    }

    // `by(keys, desc)` sorts on the entry KEY instead: ripple > lop lexically.
    let by_keys = qs(
        "g.V().out('CREATED').groupCount().by('name').order(Scope.local).by(keys, desc).select(Column.values)",
    );
    match &by_keys[..] {
        // ripple's count (1) first, then lop's count (3).
        [GVal::List(items)] => assert_eq!(items.as_ref(), [GVal::Num(1.0), GVal::Num(3.0)]),
        other => panic!("expected one value list, got {other:?}"),
    }
}

#[test]
fn parse_basic_chain() {
    assert_eq!(
        names(qs("g.V().has('name', 'marko').out('KNOWS').values('name')")),
        vec!["josh", "vadas"]
    );
}

#[test]
fn parse_predicate_call() {
    assert_eq!(
        names(qs("g.V().has('age', gt(30)).values('name')")),
        vec!["josh", "peter"]
    );
    assert_eq!(
        names(qs(
            "g.V().has('name', within('josh','marko')).values('name')"
        )),
        vec!["josh", "marko"]
    );
    assert_eq!(
        names(qs("g.V().has('age', between(28, 33)).values('name')")),
        vec!["josh", "marko"]
    );
}

#[test]
fn parse_count_and_group() {
    assert_eq!(one_num(qs("g.V().hasLabel('PERSON').count()")), 4.0);
    let out = qs("g.V().groupCount().by(T.label)");
    assert_eq!(
        map_sorted(&out[0]),
        vec![
            ("PERSON".into(), GVal::Num(4.0)),
            ("SOFTWARE".into(), GVal::Num(2.0))
        ]
    );
}

#[test]
fn parse_order_by_desc() {
    let r = qs("g.V().hasLabel('PERSON').order().by('age', desc).values('name')");
    assert_eq!(ordered(r), vec!["peter", "josh", "marko", "vadas"]);
}

#[test]
fn parse_nested_traversals() {
    // where with anonymous sub-traversal
    assert_eq!(
        names(qs(
            "g.V().where(__.in('CREATED').count().is(gte(2))).values('name')"
        )),
        vec!["lop"]
    );
    // repeat with anonymous body
    assert_eq!(
        names(qs("g.V('1').repeat(__.out()).times(2).values('name')")),
        vec!["lop", "ripple"]
    );
    // project with by sub-traversal
    let r = qs("g.V().has('name','marko').project('out').by(__.outE().count())");
    let m = match &r[0] {
        GVal::Map(e) => e,
        _ => panic!(),
    };
    assert_eq!(m[0].1, GVal::Num(3.0));
}

#[test]
fn parse_select_and_as() {
    let r = qs("g.V().has('name','marko').as('a').out('CREATED').as('b').select('a','b').by('name').by('name')");
    let m = match &r[0] {
        GVal::Map(e) => e,
        _ => panic!(),
    };
    assert_eq!(s(&m[0].1), "marko");
    assert_eq!(s(&m[1].1), "lop");
}

#[test]
fn parse_union_and_coalesce() {
    let r = qs("g.V().has('name','marko').union(__.values('name'), __.values('age'))");
    assert_eq!(r, vec![GVal::Str("marko".into()), GVal::Num(29.0)]);
}

#[test]
fn parse_to_json_round_trip() {
    let mut g = modern();
    let t = super::parse("g.V().hasLabel('PERSON').order().by('name').values('name')").unwrap();
    let vals = t.run(&mut g);
    let json = super::exec::results_to_json(&g, &vals);
    assert_eq!(json, r#"["josh","marko","peter","vadas"]"#);
}

#[test]
fn parse_vertex_json_has_id_label() {
    let mut g = modern();
    let t = super::parse("g.V('1')").unwrap();
    let vals = t.run(&mut g);
    let json = super::exec::results_to_json(&g, &vals);
    // Full `{id, labels, properties}` form — byte-identical to GQL `RETURN n`.
    assert_eq!(
        json,
        r#"[{"id":"1","labels":["PERSON"],"properties":{"age":29,"name":"marko"}}]"#
    );
}

// ===== property-index seeding (results must equal the scan path) =====

/// Run a query against a fresh Modern graph with the given vertex indexes built.
fn q_idx(indexes: &[&str], t: super::Traversal) -> Vec<GVal> {
    let mut g = modern();
    for k in indexes {
        g.create_vertex_index(k);
    }
    t.run(&mut g)
}

#[test]
fn index_eq_matches_scan() {
    let scan = names(q(g()
        .V()
        .has("name", P::eq("marko"))
        .out(&["KNOWS"])
        .values(&["name"])));
    let idx = names(q_idx(
        &["name"],
        g().V()
            .has("name", P::eq("marko"))
            .out(&["KNOWS"])
            .values(&["name"]),
    ));
    assert_eq!(scan, idx);
    assert_eq!(idx, vec!["josh", "vadas"]);
}

#[test]
fn index_range_matches_scan() {
    let want = vec!["josh", "peter"];
    assert_eq!(
        names(q(g().V().has("age", P::gt(30)).values(&["name"]))),
        want
    );
    assert_eq!(
        names(q_idx(
            &["age"],
            g().V().has("age", P::gt(30)).values(&["name"])
        )),
        want
    );
    // between / inside
    assert_eq!(
        names(q_idx(
            &["age"],
            g().V().has("age", P::between(28, 33)).values(&["name"])
        )),
        vec!["josh", "marko"]
    );
    assert_eq!(
        names(q_idx(
            &["age"],
            g().V().has("age", P::inside(27, 32)).values(&["name"])
        )),
        vec!["marko"]
    );
}

#[test]
fn index_within_and_startswith() {
    assert_eq!(
        names(q_idx(
            &["name"],
            g().V()
                .has("name", P::within(["josh", "marko"]))
                .values(&["name"])
        )),
        vec!["josh", "marko"]
    );
    assert_eq!(
        names(q_idx(
            &["name"],
            g().V().has("name", P::starts_with("ma")).values(&["name"])
        )),
        vec!["marko"]
    );
    // prefix that matches two: 'lop' / 'ripple' → 'r' only ripple
    assert_eq!(
        names(q_idx(
            &["name"],
            g().V().has("name", P::starts_with("r")).values(&["name"])
        )),
        vec!["ripple"]
    );
}

#[test]
fn index_range_does_not_bleed_types() {
    // age index, gt(0) must not return software (no age) — type-block bounded.
    assert_eq!(
        names(q_idx(
            &["age"],
            g().V().has("age", P::gt(0)).values(&["name"])
        )),
        vec!["josh", "marko", "peter", "vadas"]
    );
}

#[test]
fn edge_index_eq_seeds() {
    let mut gr = modern();
    gr.create_edge_index("weight");
    // weight == 1.0 → marko-knows-josh and josh-created-ripple.
    assert_eq!(
        one_num(g().E().has("weight", P::eq(1.0)).count().run(&mut gr)),
        2.0
    );
    // range: weight >= 0.5 → those two plus marko-knows-vadas (0.5) = 3.
    assert_eq!(
        one_num(g().E().has("weight", P::gte(0.5)).count().run(&mut gr)),
        3.0
    );
}

#[test]
fn index_live_add() {
    let mut gr = modern();
    gr.create_vertex_index("name");
    gr.add_vertex(
        &["PERSON".to_string()],
        vec![
            ("name".to_string(), Value::Str("zoe".into())),
            ("age".to_string(), Value::Num(50.0)),
        ],
    );
    assert_eq!(
        names(
            g().V()
                .has("name", P::eq("zoe"))
                .values(&["name"])
                .run(&mut gr)
        ),
        vec!["zoe"]
    );
}

#[test]
fn index_live_update() {
    let mut gr = modern();
    gr.create_vertex_index("name");
    let marko = gr.vid.get("1").unwrap();
    gr.set_vertex_prop(marko, "name", Value::Str("mark".into()));
    assert_eq!(
        g().V().has("name", P::eq("marko")).count().run(&mut gr),
        vec![GVal::Num(0.0)]
    ); // old gone
    assert_eq!(
        names(
            g().V()
                .has("name", P::eq("mark"))
                .values(&["name"])
                .run(&mut gr)
        ),
        vec!["mark"]
    ); // new present
}

#[test]
fn index_live_remove() {
    let mut gr = modern();
    gr.create_vertex_index("name");
    let vadas = gr.vid.get("2").unwrap();
    let _ = gr.remove_vertex(vadas, true);
    assert_eq!(
        g().V().has("name", P::eq("vadas")).count().run(&mut gr),
        vec![GVal::Num(0.0)]
    );
}

#[test]
fn edge_index_live_remove() {
    let mut gr = modern();
    gr.create_edge_index("weight");
    // remove one of the two weight-1.0 edges via Gremlin drop.
    let _ = g()
        .v_ids(&["1"])
        .out_e(&["KNOWS"])
        .has("weight", P::eq(1.0))
        .drop()
        .run(&mut gr);
    assert_eq!(
        one_num(g().E().has("weight", P::eq(1.0)).count().run(&mut gr)),
        1.0
    );
}

// helper used above
fn list_names_ordered(g: &GVal) -> Vec<String> {
    match g {
        GVal::List(items) => items.iter().map(s).collect(),
        _ => panic!("expected list, got {g:?}"),
    }
}

// --- match() — declarative pattern matching (ports steps/match.test.ts) ------

/// Normalize select(...)-of-match results into a sorted set of sorted
/// `(label, name)` rows so assertions are order-independent.
fn match_rows(r: Vec<GVal>) -> Vec<Vec<(String, String)>> {
    let mut rows: Vec<Vec<(String, String)>> = r
        .iter()
        .map(|m| {
            let mut entries: Vec<(String, String)> =
                map_sorted(m).into_iter().map(|(k, v)| (k, s(&v))).collect();
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
fn match_declarative_and_of_fragments() {
    let r = q(g()
        .V()
        .match_(vec![
            __().as_("a").out(&["CREATED"]).as_("b"),
            __().as_("b").has("name", P::eq("lop")),
            __().as_("b").in_(&["CREATED"]).as_("c"),
            __().as_("c").has("age", P::eq(29)),
        ])
        .select(&["a", "c"])
        .by("name"));
    let mut want = vec![
        pairs(&[("a", "marko"), ("c", "marko")]),
        pairs(&[("a", "josh"), ("c", "marko")]),
        pairs(&[("a", "peter"), ("c", "marko")]),
    ];
    want.sort();
    assert_eq!(match_rows(r), want);
}

#[test]
fn match_chained_pattern_with_embedded_has() {
    let r = q(g()
        .V()
        .match_(vec![
            __().as_("a")
                .out(&["CREATED"])
                .has("name", P::eq("lop"))
                .as_("b"),
            __().as_("b")
                .in_(&["CREATED"])
                .has("age", P::eq(29))
                .as_("c"),
        ])
        .select(&["a", "c"])
        .by("name"));
    let mut want = vec![
        pairs(&[("a", "marko"), ("c", "marko")]),
        pairs(&[("a", "josh"), ("c", "marko")]),
        pairs(&[("a", "peter"), ("c", "marko")]),
    ];
    want.sort();
    assert_eq!(match_rows(r), want);
}

#[test]
fn match_combined_with_where_neq() {
    let r = q(g()
        .V()
        .match_(vec![
            __().as_("a").out(&["CREATED"]).as_("b"),
            __().as_("b").in_(&["CREATED"]).as_("c"),
        ])
        .where_key("a", P::neq(GVal::Str("c".into())))
        .select(&["a", "c"])
        .by("name"));
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
fn match_nested_not() {
    let r = q(g()
        .V()
        .as_("a")
        .out(&["KNOWS"])
        .as_("b")
        .match_(vec![
            __().as_("b").out(&["CREATED"]).as_("c"),
            __().not(__().as_("c").in_(&["CREATED"]).as_("a")),
        ])
        .select(&["a", "b", "c"])
        .by("name"));
    assert_eq!(
        match_rows(r),
        vec![pairs(&[("a", "marko"), ("b", "josh"), ("c", "ripple")])]
    );
}

// --- subgraph() — accumulate matching edges (ports steps/subgraph.test.ts) ---
//
// The Rust GVal has no graph type, so cap() of a subgraph key yields a
// {vertices, edges} id-list map rather than the TS engine's Graph object; the
// collected membership (and thus counts) match.

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
fn subgraph_collects_knows_edges() {
    // 2 KNOWS edges (marko→vadas, marko→josh) over 3 vertices.
    let r = q(g().E().has_label(&["KNOWS"]).subgraph("sg").cap("sg"));
    assert_eq!(subgraph_counts(r), (3, 2));
}

#[test]
fn subgraph_chained_accumulation() {
    // marko knows {vadas, josh}; josh created {lop, ripple} → 2 edges, 3 vertices.
    let r = q(g()
        .V()
        .out_e(&["KNOWS"])
        .subgraph("knowsG")
        .in_v()
        .out_e(&["CREATED"])
        .subgraph("createdG")
        .in_v()
        .cap("createdG"));
    assert_eq!(subgraph_counts(r), (3, 2));
}

// --- shortestPath() (ports steps/shortestPath.test.ts) -----------------------

/// Run a shortestPath traversal and resolve each emitted path's vertices to ids.
fn sp_paths(t: super::Traversal) -> Vec<Vec<String>> {
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
fn shortest_path_target_via_with() {
    // marko —knows→ josh, one hop.
    let paths = sp_paths(
        g().V()
            .has("name", P::eq("marko"))
            .shortest_path_to(__().has("name", P::eq("josh"))),
    );
    assert_eq!(paths, vec![vec!["1".to_string(), "4".to_string()]]);
}

#[test]
fn shortest_path_multi_hop() {
    // marko —knows→ josh —created→ ripple, two hops (the shortest route).
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
fn shortest_path_no_target_reaches_all() {
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

// --- hardening: parser robustness + repeat budget (G2/G5/G6) ----------------

#[test]
fn parser_deep_nesting_is_an_error_not_a_crash() {
    // Without a depth guard this overflows the native stack and aborts the
    // process (uncatchable); it must instead be a clean parse error.
    let deep = format!("g.V(){}{}", ".repeat(".repeat(2000), "out()");
    let q = format!("{deep}{}", ")".repeat(2000));
    assert!(super::parse(&q).is_err());
}

#[test]
fn parser_missing_step_args_error_not_panic() {
    for q in [
        "g.V().limit()",
        "g.V().skip()",
        "g.V().range(1)",
        "g.V().sample()",
        "g.V().constant()",
        "g.V().as()",
        "g.V().aggregate()",
        "g.V().property('k')",
    ] {
        assert!(super::parse(q).is_err(), "expected a parse error for `{q}`");
    }
}

#[test]
fn parser_rejects_non_integer_counts() {
    for q in ["g.V().limit(-5)", "g.V().limit(2.5)", "g.V().range(0, -1)"] {
        assert!(super::parse(q).is_err(), "expected a parse error for `{q}`");
    }
}

#[test]
fn parser_valid_counts_still_parse() {
    for q in [
        "g.V().limit(3)",
        "g.V().range(1, 4)",
        "g.V().repeat(out()).times(2)",
    ] {
        assert!(super::parse(q).is_ok(), "expected `{q}` to parse");
    }
}

#[test]
fn repeat_budget_guards_runaway_on_dense_graph() {
    // A complete directed graph on 8 vertices: repeat(both()) with no
    // termination grows the frontier explosively → must hit the budget.
    let mut lines: Vec<String> = Vec::new();
    for i in 0..8 {
        lines.push(format!(
            r#"{{"type":"node","id":"{i}","labels":["N"],"properties":{{}}}}"#
        ));
    }
    for i in 0..8 {
        for j in 0..8 {
            if i != j {
                lines.push(format!(
                    r#"{{"type":"edge","from":"{i}","to":"{j}","labels":["R"],"properties":{{}}}}"#
                ));
            }
        }
    }
    let mut g = crate::ndjson::decode(&lines.join("\n")).unwrap();
    let t = super::parse("g.V().repeat(both())").unwrap();
    let err = super::try_run(&mut g, &t).unwrap_err();
    assert_eq!(err.code, crate::error_codes::ErrorCode::ResourceExhausted);
}

#[test]
fn order_by_mixed_type_property_faults_not_panics() {
    // A schemaless property that is a number on some vertices and a string on
    // others makes `order().by('p')` compare incomparable pairs. The comparator
    // must be a TOTAL order or Rust's sort_by panics ("does not implement a total
    // order") — which under panic=abort would abort the host. ~60 vertices to
    // reliably trip sort_by's total-order check. try_run surfaces the recorded
    // type fault as E_INVALID_VALUE; run must not panic.
    let lines: Vec<String> = (0..100u64)
        .map(|i| {
            // Pseudo-random number/string split (a strict alternation doesn't
            // reliably trip sort_by's total-order check; a shuffled one does).
            let p = if (i.wrapping_mul(2_654_435_761) >> 16) & 1 == 0 {
                format!("{i}")
            } else {
                format!("\"s{i}\"")
            };
            format!(r#"{{"type":"node","id":"{i}","labels":["T"],"properties":{{"p":{p}}}}}"#)
        })
        .collect();
    let mut g = crate::ndjson::decode(&lines.join("\n")).unwrap();
    let t = super::parse("g.V().order().by('p')").unwrap();
    assert_eq!(
        super::try_run(&mut g, &t).unwrap_err().code,
        crate::error_codes::ErrorCode::InvalidValue
    );
    // Infallible path: best-effort, but must not panic.
    let _ = t.run(&mut g);
}

#[test]
fn lexer_preserves_utf8_string_literals() {
    let lines = [r#"{"type":"node","id":"1","labels":["P"],"properties":{"name":"café"}}"#];
    let mut g = crate::ndjson::decode(&lines.join("\n")).unwrap();
    let t = super::parse("g.V().has('name','café').values('name')").unwrap();
    assert_eq!(t.run(&mut g), vec![GVal::Str("café".into())]);
}

#[test]
fn lexer_decodes_string_escapes() {
    let mut g = modern();
    let t = super::parse(r"g.inject('a\nb')").unwrap();
    assert_eq!(t.run(&mut g), vec![GVal::Str("a\nb".into())]);
}

// --- G7-G9 (Rust): TinkerPop Comparable semantics — throw on incomparable ----

#[test]
fn comparison_of_incomparable_types_faults() {
    let mut g = modern();
    // names are strings; gt(5) compares them to a number → incomparable.
    let t = super::parse("g.V().values('name').is(gt(5))").unwrap();
    assert_eq!(
        super::try_run(&mut g, &t).unwrap_err().code,
        crate::error_codes::ErrorCode::InvalidValue
    );
}

#[test]
fn addv_and_property_reject_malformed_names() {
    use crate::error_codes::ErrorCode::InvalidValue;
    let mut g = modern();
    // Gremlin takes arbitrary label/key strings, so a `::` label / empty key is
    // guarded at the step (codec ingestion has its own gate). try_run surfaces it.
    let bad = [
        "g.addV('a::b')",        // GraphSON multi-label separator in a label
        "g.addV('')",            // empty label
        "g.V().property('', 1)", // empty property key
    ];
    for src in bad {
        let t = super::parse(src).unwrap();
        assert_eq!(
            super::try_run(&mut g, &t).unwrap_err().code,
            InvalidValue,
            "{src}"
        );
    }
    // A well-formed addV/property is fine.
    assert!(super::try_run(&mut g, &super::parse("g.addV('Robot')").unwrap()).is_ok());
}

#[test]
fn order_over_mixed_types_faults() {
    let mut g = modern();
    let t = super::parse("g.inject(3, 'a', 1).order()").unwrap();
    assert_eq!(
        super::try_run(&mut g, &t).unwrap_err().code,
        crate::error_codes::ErrorCode::InvalidValue
    );
}

#[test]
fn sum_of_non_numeric_faults() {
    let mut g = modern();
    let t = super::parse("g.V().values('name').sum()").unwrap();
    assert_eq!(
        super::try_run(&mut g, &t).unwrap_err().code,
        crate::error_codes::ErrorCode::InvalidValue
    );
}

#[test]
fn comparable_predicate_and_aggregation_still_work() {
    let mut g = modern();
    // age > 30 → josh(32), peter(35) → count 2; no coercion, no fault.
    let t = super::parse("g.V().values('age').is(gt(30)).count()").unwrap();
    assert_eq!(super::try_run(&mut g, &t).unwrap(), vec![GVal::Num(2.0)]);
}

// --- math(): infix arithmetic — cross-engine parity with @lenke/gremlin --------

#[test]
fn math_arithmetic_over_values() {
    // ages *2, insertion order marko/vadas/josh/peter: 29,27,32,35 → 58,54,64,70.
    let r = q(g()
        .V()
        .has_label(&["PERSON"])
        .values(&["age"])
        .math("_ * 2"));
    assert_eq!(
        r,
        vec![
            GVal::Num(58.0),
            GVal::Num(54.0),
            GVal::Num(64.0),
            GVal::Num(70.0)
        ]
    );
}

#[test]
fn math_parens_and_precedence() {
    // (10 - 2) / 2 + 1 = 5 — parens override, then * / before + -.
    let r = q(g().inject([GVal::Num(10.0)]).math("(_ - 2) / 2 + 1"));
    assert_eq!(one_num(r), 5.0);
}

#[test]
fn math_by_projects_the_operand() {
    // math('_ + 1').by('age') projects each vertex through `age` before adding.
    let r = q(g().V().has_label(&["PERSON"]).math("_ + 1").by("age"));
    assert_eq!(
        r,
        vec![
            GVal::Num(30.0),
            GVal::Num(28.0),
            GVal::Num(33.0),
            GVal::Num(36.0)
        ]
    );
}

#[test]
fn math_over_nonnumeric_is_a_type_fault() {
    // A non-numeric operand faults (TinkerPop requires numbers), matching the TS
    // engine's `math`. Surfaced by try_run as InvalidValue.
    let mut g = modern();
    let t = super::parse("g.V().values('name').math('_ + 1')").unwrap();
    assert!(super::try_run(&mut g, &t).is_err());
}

#[test]
fn math_malformed_expression_faults() {
    let mut g = modern();
    let t = super::parse("g.inject(1).math('_ +')").unwrap();
    assert!(super::try_run(&mut g, &t).is_err());
}

// --- math(): functions, `^`/`%`, unary, constants (parity with @lenke/gremlin) -

#[test]
fn math_functions_use_the_shared_f64_kernel() {
    // Each function must call the SAME primitive as the GQL `call_scalar` kernel,
    // so `math()` stays bit-identical to GQL and to the TS engine.
    let f = |expr: &str| one_num(q(g().inject([GVal::Num(0.7)]).math(expr)));
    assert_eq!(f("sin(_)"), 0.7_f64.sin());
    assert_eq!(f("cos(_)"), 0.7_f64.cos());
    assert_eq!(f("tan(_)"), 0.7_f64.tan());
    assert_eq!(f("asin(_)"), 0.7_f64.asin());
    assert_eq!(f("acos(_)"), 0.7_f64.acos());
    assert_eq!(f("atan(_)"), 0.7_f64.atan());
    assert_eq!(f("sinh(_)"), 0.7_f64.sinh());
    assert_eq!(f("cosh(_)"), 0.7_f64.cosh());
    assert_eq!(f("tanh(_)"), 0.7_f64.tanh());
    assert_eq!(f("sqrt(_)"), 0.7_f64.sqrt());
    assert_eq!(f("abs(-_)"), 0.7_f64);
    assert_eq!(f("ceil(_)"), 1.0);
    assert_eq!(f("floor(_)"), 0.0);
    assert_eq!(f("exp(_)"), 0.7_f64.exp());
    assert_eq!(f("ln(_)"), 0.7_f64.ln());
    assert_eq!(f("log10(_)"), 0.7_f64.log10());
    assert_eq!(f("signum(_)"), 1.0);
    assert_eq!(f("signum(-_)"), -1.0);
}

#[test]
fn math_two_arg_functions() {
    let f = |expr: &str| one_num(q(g().inject([GVal::Num(2.0)]).math(expr)));
    assert_eq!(f("pow(_, 10)"), 1024.0);
    assert_eq!(f("atan2(_, 1)"), 2.0_f64.atan2(1.0));
    // log(base, value): log base 2 of 8 == 3 (via value.ln()/base.ln()).
    assert_eq!(f("log(_, 8)"), 8.0_f64.ln() / 2.0_f64.ln());
}

#[test]
fn math_power_operator_right_assoc() {
    let n = |expr: &str| one_num(q(g().inject([GVal::Num(0.0)]).math(expr)));
    assert_eq!(n("2 ^ 3 ^ 2"), 512.0); // right-associative
    assert_eq!(n("2 * 3 ^ 2"), 18.0); // `^` above `*`
    assert_eq!(n("-2 ^ 2"), 4.0); // unary binds tighter than `^`
    assert_eq!(n("2 ^ -1"), 0.5);
}

#[test]
fn math_modulo_and_unary() {
    let n = |expr: &str| one_num(q(g().inject([GVal::Num(10.0)]).math(expr)));
    assert_eq!(n("_ % 3"), 1.0);
    assert_eq!(n("-_ + 3"), -7.0);
    assert_eq!(n("- -_"), 10.0);
    assert_eq!(n("2 * 3 % 4"), 2.0); // `%` same tier as `*`, left-to-right
}

#[test]
fn math_constants_pi_and_e() {
    let n = |expr: &str| one_num(q(g().inject([GVal::Num(0.0)]).math(expr)));
    assert_eq!(n("pi"), std::f64::consts::PI);
    assert_eq!(n("e"), std::f64::consts::E);
    assert_eq!(n("2 * pi"), 2.0 * std::f64::consts::PI);
}

#[test]
fn math_variable_shadows_function_name() {
    // A bound tag named `sin` resolves as the variable, not the sine function.
    let r = q(g().inject([GVal::Num(42.0)]).as_("sin").math("sin + 1"));
    assert_eq!(one_num(r), 43.0);
}

#[test]
fn math_unknown_function_faults() {
    let mut g = modern();
    let t = super::parse("g.inject(1).math('nope(_)')").unwrap();
    assert!(super::try_run(&mut g, &t).is_err());
}

#[test]
fn math_bare_juxtaposition_function_form() {
    // `sin _` == `sin(_)`; binds tighter than binary ops; right-assoc chains;
    // unary arg (so `abs -3` works). Parity with TS `evalMath`.
    let n = |expr: &str, x: f64| one_num(q(g().inject([GVal::Num(x)]).math(expr)));
    assert_eq!(n("sin _", 0.7), 0.7_f64.sin());
    assert_eq!(n("sin(_)", 0.7), 0.7_f64.sin()); // agrees with paren form
    assert_eq!(n("sin _ + 1", 0.7), 0.7_f64.sin() + 1.0);
    assert_eq!(n("sin _ * 2", 0.7), 0.7_f64.sin() * 2.0);
    assert_eq!(n("-sin _", 0.7), -(0.7_f64.sin()));
    assert_eq!(n("abs -3", 0.0), 3.0);
    assert_eq!(n("sin cos _", 0.7), 0.7_f64.cos().sin()); // right-assoc
    assert_eq!(n("sqrt _", 2.0), 2.0_f64.sqrt());
    assert_eq!(n("sin (_ + 1)", 0.7), (0.7_f64 + 1.0).sin());
}

#[test]
fn math_bare_form_multiarg_requires_parens() {
    // `atan2` is 2-arg; the bare form is unary-only, so bare `atan2 _` faults.
    let mut g = modern();
    let t = super::parse("g.inject(1).math('atan2 _')").unwrap();
    assert!(super::try_run(&mut g, &t).is_err());
}

#[test]
fn math_bare_form_variable_shadows_function() {
    // A bound tag `sin` wins over the sine function even in the bare position:
    // `sin` resolves to the variable, leaving `_` as trailing input → fault
    // (byte-identical to TS). With just `sin`, it returns the variable.
    let r = q(g().inject([GVal::Num(42.0)]).as_("sin").math("sin"));
    assert_eq!(one_num(r), 42.0);
    let mut g2 = modern();
    let t = super::parse("g.inject(42).as('sin').math('sin _')").unwrap();
    assert!(super::try_run(&mut g2, &t).is_err());
}

// --- branch(): switch on a sub-plan's result — parity with @lenke/gremlin ------

#[test]
fn branch_routes_by_label() {
    // PERSON → name; SOFTWARE → 'a software'. Per-traverser, insertion order.
    let r = q(g()
        .V()
        .branch(__().label())
        .option("PERSON", __().values(&["name"]))
        .option("SOFTWARE", __().constant("a software")));
    assert_eq!(
        ordered(r),
        vec![
            "marko",
            "vadas",
            "josh",
            "peter",
            "a software",
            "a software"
        ]
    );
}

#[test]
fn branch_default_via_option_none() {
    // age 29 → 'young', everyone else falls to the default 'older'.
    let r = q(g()
        .V()
        .has_label(&["PERSON"])
        .branch(__().values(&["age"]))
        .option(29, __().constant("young"))
        .option_none(__().constant("older")));
    assert_eq!(ordered(r), vec!["young", "older", "older", "older"]);
}

#[test]
fn branch_parses_none_default_from_text() {
    // `option(none, …)` is TinkerPop's Pick.none default; parse it from text.
    let mut g = modern();
    let t = super::parse(
        "g.V().hasLabel('PERSON').branch(values('age'))\
         .option(29, constant('young')).option(none, constant('older'))",
    )
    .unwrap();
    assert_eq!(
        ordered(t.run(&mut g)),
        vec!["young", "older", "older", "older"]
    );
}

#[test]
fn branch_no_default_drops_unmatched() {
    // Without a default, traversers whose test result matches no option vanish.
    let r = q(g()
        .V()
        .has_label(&["PERSON"])
        .branch(__().values(&["age"]))
        .option(29, __().constant("young")));
    assert_eq!(ordered(r), vec!["young"]);
}

// --- regex() predicate — parity with @lenke/gremlin ---------------------------

#[test]
fn regex_anchored_and_unanchored() {
    // `^ma` anchors to the start → marko only.
    assert_eq!(
        names(q(g().V().has("name", P::regex("^ma")).values(&["name"]))),
        vec!["marko"]
    );
    // Unanchored `o` searches anywhere → marko, josh, lop (like JS RegExp.test).
    assert_eq!(
        names(q(g().V().has("name", P::regex("o")).values(&["name"]))),
        vec!["josh", "lop", "marko"]
    );
}

#[test]
fn regex_parses_textp_namespace() {
    let mut g = modern();
    let t = super::parse("g.V().has('name', TextP.regex('^r')).values('name')").unwrap();
    assert_eq!(ordered(t.run(&mut g)), vec!["ripple"]);
}

#[test]
fn regex_invalid_pattern_is_a_parse_error() {
    // Validated at parse time (like the TS `regex()` constructor), not per value.
    assert!(super::parse("g.V().has('name', regex('['))").is_err());
}

// ===== JSON output characterization =====
//
// These pin `results_to_json`'s exact bytes so the upcoming hand-rolled writer
// (which drops `serde_json`) can be proven equivalent. `serde_json::Map` is a
// `BTreeMap`, so object keys come out lexicographically sorted — the sync
// live-query layer diffs cells by `JSON.stringify` byte-equality, so that
// canonical order is load-bearing and the writer must preserve it.
//
// Split deliberately: `..._escaping_and_structure` is INVARIANT (any diff there
// is a regression), while `..._numbers` is the one part expected to change when
// serde goes — its ryu output (`29.0`, `-0.0`) becomes the shared `js_number`
// (`29`, `0`), matching the TS engine and the ndjson/codec paths. All consumers
// parse the carrier back to numbers, so that change is invisible downstream.

fn results_json(vals: Vec<GVal>) -> String {
    super::exec::results_to_json(&modern(), &vals)
}

#[test]
fn results_json_escaping_and_structure() {
    // String escaping: `"` and `\` escaped, `/` NOT escaped, control chars via
    // `\b \t \n \f \r` shortcuts else `\u00XX`, non-ASCII left as raw UTF-8.
    assert_eq!(results_json(vec![GVal::from("a\"b")]), r#"["a\"b"]"#);
    assert_eq!(results_json(vec![GVal::from("a\\b")]), r#"["a\\b"]"#);
    assert_eq!(results_json(vec![GVal::from("a/b")]), r#"["a/b"]"#);
    assert_eq!(
        results_json(vec![GVal::from("x\t\n\ry")]),
        r#"["x\t\n\ry"]"#
    );
    assert_eq!(
        results_json(vec![GVal::from("x\u{08}\u{0c}y")]),
        r#"["x\b\fy"]"#
    );
    assert_eq!(
        results_json(vec![GVal::from("x\u{01}y")]),
        r#"["x\u0001y"]"#
    );
    assert_eq!(
        results_json(vec![GVal::from("café\u{1F980}")]),
        "[\"café\u{1F980}\"]"
    );
    assert_eq!(results_json(vec![GVal::from("")]), r#"[""]"#);

    // Structure: empty containers, nesting, bool, null. (String-valued to keep
    // this test free of the number formatting that the refactor will change.)
    assert_eq!(results_json(vec![GVal::list(vec![])]), "[[]]");
    assert_eq!(results_json(vec![GVal::Map(vec![])]), "[{}]");
    assert_eq!(
        results_json(vec![GVal::list(vec![
            GVal::from("a"),
            GVal::list(vec![GVal::from("z")]),
        ])]),
        r#"[["a",["z"]]]"#
    );
    assert_eq!(
        results_json(vec![GVal::Bool(true), GVal::Bool(false)]),
        "[true,false]"
    );
    assert_eq!(results_json(vec![GVal::Null]), "[null]");

    // Map keys sorted lexicographically (serde BTreeMap); string values so the
    // ordering is the only thing under test here.
    assert_eq!(
        results_json(vec![GVal::Map(vec![
            (GVal::from("zzz"), GVal::from("z")),
            (GVal::from("age"), GVal::from("a")),
            (GVal::from("name"), GVal::from("m")),
        ])]),
        r#"[{"age":"a","name":"m","zzz":"z"}]"#
    );

    // Graph elements serialize to the full `{id, labels, properties}` form (edge:
    // `{id, from, to, labels, properties}`) — byte-identical to GQL and the TS engine.
    assert_eq!(
        results_json(vec![GVal::Node(0)]),
        r#"[{"id":"1","labels":["PERSON"],"properties":{"age":29,"name":"marko"}}]"#
    );
    assert_eq!(
        results_json(vec![GVal::Edge(0)]),
        r#"[{"id":"e0","from":"1","to":"2","labels":["KNOWS"],"properties":{"weight":0.5}}]"#
    );
}

#[test]
fn results_json_numbers() {
    // Numbers now go through the shared `js_number` (was serde/ryu): 29.0→29,
    // -0.0→0, the numeric map key 5.0→5. Exponential forms are unchanged. This
    // matches the TS engine + ndjson/codec; all consumers parse the carrier back
    // to numbers, so the change is invisible downstream.
    assert_eq!(results_json(vec![GVal::Num(29.0)]), "[29]");
    assert_eq!(results_json(vec![GVal::Num(1.5)]), "[1.5]");
    assert_eq!(results_json(vec![GVal::Num(-0.0)]), "[0]");
    assert_eq!(results_json(vec![GVal::Num(1e21)]), "[1e+21]");
    assert_eq!(results_json(vec![GVal::Num(1e-7)]), "[1e-7]");
    // Non-finite → null (not representable in JSON).
    assert_eq!(results_json(vec![GVal::Num(f64::NAN)]), "[null]");
    assert_eq!(results_json(vec![GVal::Num(f64::INFINITY)]), "[null]");
    // Non-string map key is stringified via the same number formatting.
    assert_eq!(
        results_json(vec![GVal::Map(vec![(GVal::Num(5.0), GVal::from("v"))])]),
        r#"[{"5":"v"}]"#
    );
}

// --- OLAP algorithm steps (local compute; withComputer is a spec-currency no-op) ---

#[test]
fn algo_page_rank_writes_and_passes_through() {
    // pageRank() computes over the whole graph, writes the default property, and
    // passes the incoming traversers through so `.values()` reads the scores back.
    let scores = q(g()
        .V()
        .page_rank(None)
        .values(&["gremlin.pageRankVertexProgram.pageRank"]));
    // One score per vertex, all finite numbers.
    assert_eq!(scores.len(), 6);
    assert!(scores
        .iter()
        .all(|v| matches!(v, GVal::Num(n) if n.is_finite() && *n > 0.0)));
}

#[test]
fn algo_page_rank_custom_property_and_alpha() {
    // pageRank(alpha) + .with(PageRank.propertyName, 'pr') writes where asked.
    let scores = q(g()
        .V()
        .page_rank(Some(0.85))
        .with_algo_property("pr".to_string())
        .values(&["pr"]));
    assert_eq!(scores.len(), 6);
    // The default property was not written.
    assert!(q(g()
        .V()
        .page_rank(Some(0.85))
        .with_algo_property("pr".to_string())
        .values(&["gremlin.pageRankVertexProgram.pageRank"]))
    .is_empty());
}

#[test]
fn algo_connected_component_single_component() {
    // The Modern graph is one weakly-connected component: every vertex shares the
    // same component id (the min-insertion-index root's external id).
    let comps = q(g()
        .V()
        .connected_component()
        .values(&["gremlin.connectedComponentVertexProgram.component"])
        .dedup());
    assert_eq!(comps.len(), 1);
}

#[test]
fn algo_peer_pressure_writes_cluster() {
    let clusters = q(g()
        .V()
        .peer_pressure()
        .values(&["gremlin.peerPressureVertexProgram.cluster"]));
    // One cluster label per vertex; labels are external-id strings.
    assert_eq!(clusters.len(), 6);
    assert!(clusters.iter().all(|v| matches!(v, GVal::Str(_))));
}

#[test]
fn algo_peer_pressure_times() {
    // .with(PeerPressure.times, 1) caps iterations without error.
    let clusters = q(g()
        .V()
        .peer_pressure()
        .with_algo_times(1)
        .values(&["gremlin.peerPressureVertexProgram.cluster"]));
    assert_eq!(clusters.len(), 6);
}

#[test]
fn algo_parse_string_form() {
    // The string parser accepts the TinkerPop OLAP surface: withComputer() as a
    // no-op marker, pageRank()/connectedComponent()/peerPressure(), and the
    // .with(<Algo>.propertyName / .times) modulators.
    let scores = super::parse(
        "g.withComputer().V().pageRank().values('gremlin.pageRankVertexProgram.pageRank')",
    )
    .unwrap()
    .run(&mut modern());
    assert_eq!(scores.len(), 6);

    let comps = super::parse(
        "g.V().connectedComponent().values('gremlin.connectedComponentVertexProgram.component').dedup()",
    )
    .unwrap()
    .run(&mut modern());
    assert_eq!(comps.len(), 1);

    let pr = super::parse("g.V().pageRank(0.85).with(PageRank.propertyName, 'pr').values('pr')")
        .unwrap()
        .run(&mut modern());
    assert_eq!(pr.len(), 6);

    let clusters = super::parse(
        "g.V().peerPressure().with(PeerPressure.times, 2).values('gremlin.peerPressureVertexProgram.cluster')",
    )
    .unwrap()
    .run(&mut modern());
    assert_eq!(clusters.len(), 6);
}

#[test]
fn algo_parse_edges_modulator_rejected() {
    // The .edges() modulator is not yet supported — it errors rather than silently
    // ignoring the requested edge set.
    let err = super::parse("g.V().pageRank().with(PageRank.edges, __.outE())");
    assert!(
        err.is_err(),
        "expected .with(PageRank.edges,...) to be rejected"
    );
}

#[test]
fn gremlin_reads_a_stored_map_property() {
    // A stored record/map property reads back as a `GVal::Map` (string keys),
    // flowing through `values()`/`valueMap()` like any Gremlin map.
    let mut gr = ndjson::decode(
        r#"{"type":"node","id":"a","labels":["P"],"properties":{"meta":{"city":"NYC","zip":"10001"}}}"#,
    )
    .unwrap();
    let r = g().V().values(&["meta"]).run(&mut gr);
    match r.as_slice() {
        [GVal::Map(pairs)] => {
            // Keys are the stored (canonical, sorted) fields, as GVal::Str.
            let keys: Vec<String> = pairs
                .iter()
                .map(|(k, _)| match k {
                    GVal::Str(s) => s.to_string(),
                    other => format!("{other:?}"),
                })
                .collect();
            assert_eq!(keys, vec!["city".to_string(), "zip".to_string()]);
            assert!(matches!(&pairs[0].1, GVal::Str(s) if s.as_ref() == "NYC"));
        }
        other => panic!("expected one GVal::Map, got {other:?}"),
    }
}
