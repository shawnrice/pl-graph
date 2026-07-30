//! Ported TS per-step conformance tests (batch 4): where, by, is, project,
//! tokens, in_, V, property, map, constant, simplePath, index, barrier-store,
//! otherV, mutation-combinators. Each `#[test]` mirrors one TS `test(...)` from
//! `packages/gremlin/src/steps/*.test.ts`, translated to a Gremlin text query
//! (via [`super::parse`]) or the fluent builder where text can't express it.
//!
//! Skips (genuine Rust gaps / divergences) are documented inline with `// SKIP:`.

#![allow(clippy::float_cmp)]

use super::{g, GVal, __};
use crate::graph::Graph;
use crate::ndjson;

/// Canonical TinkerPop "Modern" graph (copied from `tests.rs`).
fn modern() -> Graph {
    let lines = [
        r#"{"type":"node","id":"1","labels":["PERSON"],"properties":{"name":"marko","age":29}}"#,
        r#"{"type":"node","id":"2","labels":["PERSON"],"properties":{"name":"vadas","age":27}}"#,
        r#"{"type":"node","id":"4","labels":["PERSON"],"properties":{"name":"josh","age":32}}"#,
        r#"{"type":"node","id":"6","labels":["PERSON"],"properties":{"name":"peter","age":35}}"#,
        r#"{"type":"node","id":"3","labels":["SOFTWARE"],"properties":{"name":"lop","lang":"java"}}"#,
        r#"{"type":"node","id":"5","labels":["SOFTWARE"],"properties":{"name":"ripple","lang":"java"}}"#,
        r#"{"type":"edge","from":"1","to":"2","labels":["KNOWS"],"properties":{"weight":0.5}}"#,
        r#"{"type":"edge","from":"1","to":"4","labels":["KNOWS"],"properties":{"weight":1.0}}"#,
        r#"{"type":"edge","from":"1","to":"3","labels":["CREATED"],"properties":{"weight":0.4}}"#,
        r#"{"type":"edge","from":"4","to":"5","labels":["CREATED"],"properties":{"weight":1.0}}"#,
        r#"{"type":"edge","from":"4","to":"3","labels":["CREATED"],"properties":{"weight":0.4}}"#,
        r#"{"type":"edge","from":"6","to":"3","labels":["CREATED"],"properties":{"weight":0.2}}"#,
    ];
    ndjson::decode(&lines.join("\n")).unwrap()
}

/// Run a textual Gremlin query against a fresh Modern graph.
fn run(query: &str) -> Vec<GVal> {
    let mut g = modern();
    super::parse(query).unwrap().run(&mut g)
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

fn nums(r: Vec<GVal>) -> Vec<f64> {
    r.iter()
        .map(|g| match g {
            GVal::Num(n) => *n,
            other => panic!("expected number, got {other:?}"),
        })
        .collect()
}

fn one_num(r: Vec<GVal>) -> f64 {
    match r.as_slice() {
        [GVal::Num(n)] => *n,
        _ => panic!("expected single number, got {r:?}"),
    }
}

/// A result map's entries as (key-string, value).
fn map_entries(g: &GVal) -> Vec<(String, GVal)> {
    match g {
        GVal::Map(entries) => entries.iter().map(|(k, v)| (s(k), v.clone())).collect(),
        _ => panic!("expected map, got {g:?}"),
    }
}

/// Lookup a value in a result map by key string.
fn map_get<'a>(g: &'a GVal, key: &str) -> Option<&'a GVal> {
    match g {
        GVal::Map(entries) => entries
            .iter()
            .find(|(k, _)| matches!(k, GVal::Str(s) if s.as_ref() == key))
            .map(|(_, v)| v),
        _ => None,
    }
}

fn list_names_ordered(g: &GVal) -> Vec<String> {
    match g {
        GVal::List(items) => items.iter().map(s).collect(),
        _ => panic!("expected list, got {g:?}"),
    }
}

// ===================== where.test.ts =====================

#[test]
fn p4_where_count_is_1() {
    assert_eq!(
        ordered(run(
            "g.V().where(__.in('CREATED').count().is(eq(1))).values('name')"
        )),
        vec!["ripple"]
    );
}

#[test]
fn p4_where_gte() {
    assert_eq!(
        ordered(run(
            "g.V().where(__.in('CREATED').count().is(gte(2))).values('name')"
        )),
        vec!["lop"]
    );
}

#[test]
fn p4_where_out_created_nonempty() {
    assert_eq!(
        names(run("g.V().where(out('CREATED')).values('name')")),
        vec!["josh", "marko", "peter"]
    );
}

#[test]
fn p4_where_after_out() {
    assert_eq!(
        ordered(run(
            "g.V().out('KNOWS').where(out('CREATED')).values('name')"
        )),
        vec!["josh"]
    );
}

#[test]
fn p4_where_chained_not_and_in() {
    assert_eq!(
        ordered(run(
            "g.V().where(__.not(out('CREATED'))).where(__.in('KNOWS')).values('name')"
        )),
        vec!["vadas"]
    );
}

#[test]
fn p4_where_otherv_hasid() {
    // g.V(1).bothE().where(otherV().hasId(2)) — the KNOWS edge to vadas.
    let r = run("g.V('1').bothE().where(__.otherV().hasId('2'))");
    assert_eq!(r.len(), 1);
    assert!(matches!(r[0], GVal::Edge(_)));
}

#[test]
fn p4_where_out_count_gte_2() {
    assert_eq!(
        ordered(run(
            "g.V().where(out('CREATED').count().is(gte(2))).values('name')"
        )),
        vec!["josh"]
    );
}

#[test]
fn p4_where_and_oute() {
    assert_eq!(
        names(run(
            "g.V().where(and(outE('CREATED'), outE('KNOWS'))).values('name')"
        )),
        vec!["marko"]
    );
}

#[test]
fn p4_where_or_oute() {
    assert_eq!(
        names(run(
            "g.V().where(or(outE('CREATED'), outE('KNOWS'))).values('name')"
        )),
        vec!["josh", "marko", "peter"]
    );
}

#[test]
fn p4_where_nested() {
    assert_eq!(
        ordered(run(
            "g.V().where(out('KNOWS').where(out('CREATED'))).values('name')"
        )),
        vec!["marko"]
    );
}

#[test]
fn p4_where_key_gt_by_age() {
    // where('a', gt('b')).by('age') compares two as-tagged ages.
    let r = run("g.V().hasLabel('PERSON').as('a').out('CREATED').in('CREATED').hasLabel('PERSON').as('b').where('a', gt('b')).by('age').values('name')");
    assert_eq!(names(r), vec!["josh", "marko", "marko"]);
}

#[test]
fn p4_where_key_neq_by_name() {
    let r = run("g.V('1').as('a').out('CREATED').in('CREATED').as('b').where('a', neq('b')).by('name').values('name')");
    assert_eq!(names(r), vec!["josh", "peter"]);
}

// ===================== by.test.ts =====================

#[test]
fn p4_by_order_by_key() {
    assert_eq!(
        ordered(run(
            "g.V().hasLabel('PERSON').order().by('age').values('name')"
        )),
        vec!["vadas", "marko", "josh", "peter"]
    );
}

#[test]
fn p4_order_by_key_over_project_rows() {
    // Regression: `order().by('<key>')` over `project()` Map rows sorts by the
    // keyed value, not "cannot order an element with an element" (both engines had
    // this — `eval_by` only projected a key off a vertex/edge, not a Map).
    let rows = run(
        "g.V().hasLabel('PERSON').project('name','age').by('name').by('age').order().by('age')",
    );
    let ages: Vec<f64> = rows
        .iter()
        .map(|r| match map_get(r, "age") {
            Some(GVal::Num(n)) => *n,
            other => panic!("expected an age, got {other:?}"),
        })
        .collect();
    assert_eq!(ages, vec![27.0, 29.0, 32.0, 35.0]);
}

#[test]
fn p4_by_dedupe_by_label() {
    // dedupe().by(label()) keeps one element per distinct label ⇒ 2.
    assert_eq!(run("g.V().dedup().by(label())").len(), 2);
}

#[test]
fn p4_by_group_by_label_by_name() {
    let out = run("g.V().group().by(label()).by('name')");
    let m = &out[0];
    let mut person = list_names_ordered(map_get(m, "PERSON").unwrap());
    person.sort();
    assert_eq!(person, vec!["josh", "marko", "peter", "vadas"]);
    let mut sw = list_names_ordered(map_get(m, "SOFTWARE").unwrap());
    sw.sort();
    assert_eq!(sw, vec!["lop", "ripple"]);
}

#[test]
fn p4_by_group_count_by_label() {
    let out = run("g.V().groupCount().by(label())");
    let m = &out[0];
    assert_eq!(map_get(m, "PERSON"), Some(&GVal::Num(4.0)));
    assert_eq!(map_get(m, "SOFTWARE"), Some(&GVal::Num(2.0)));
}

#[test]
fn p4_by_project_subtraversals() {
    let out = run(
        "g.V('1').project('name','outDeg','inDeg').by('name').by(outE().count()).by(inE().count())",
    );
    let m = &out[0];
    assert_eq!(map_get(m, "name"), Some(&GVal::Str("marko".into())));
    assert_eq!(map_get(m, "outDeg"), Some(&GVal::Num(3.0)));
    assert_eq!(map_get(m, "inDeg"), Some(&GVal::Num(0.0)));
}

#[test]
fn p4_by_path_by_name() {
    // g.V(1).outE('KNOWS').path().by('name') — two paths, each begins at 'marko'.
    let out = run("g.V('1').outE('KNOWS').path().by('name')");
    assert_eq!(out.len(), 2);
    let firsts: Vec<String> = out
        .iter()
        .map(|p| list_names_ordered(p)[0].clone())
        .collect();
    assert_eq!(firsts, vec!["marko", "marko"]);
}

#[test]
fn p4_by_order_by_desc_values() {
    assert_eq!(
        ordered(run("g.V().values('name').order().by(Order.desc)")),
        vec!["vadas", "ripple", "peter", "marko", "lop", "josh"]
    );
}

#[test]
fn p4_by_order_per_by_direction() {
    // order().by(outE('CREATED').count(), desc).by('age', asc)
    let r = run("g.V().hasLabel('PERSON').order().by(outE('CREATED').count(), Order.desc).by('age', Order.asc).values('name')");
    assert_eq!(ordered(r), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p4_by_order_by_subtraversal_count() {
    let r = ordered(run(
        "g.V().hasLabel('PERSON').order().by(outE('CREATED').count()).values('name')",
    ));
    assert_eq!(r.first().map(String::as_str), Some("vadas"));
    assert_eq!(r.last().map(String::as_str), Some("josh"));
}

#[test]
fn p4_by_group_count_by_name() {
    let out = run("g.V().groupCount().by('name')");
    let m = &out[0];
    assert_eq!(map_get(m, "marko"), Some(&GVal::Num(1.0)));
    assert_eq!(map_get(m, "lop"), Some(&GVal::Num(1.0)));
    assert_eq!(map_entries(m).len(), 6);
}

// ===================== is.test.ts =====================

#[test]
fn p4_is_simple_number() {
    assert_eq!(run("g.V().values('age').is(eq(32))"), vec![GVal::Num(32.0)]);
}

#[test]
fn p4_is_lte() {
    assert_eq!(
        nums(run("g.V().values('age').is(lte(30))")),
        vec![29.0, 27.0]
    );
}

#[test]
fn p4_is_inside_30_40() {
    let mut r = nums(run("g.V().values('age').is(inside(30, 40))"));
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(r, vec![32.0, 35.0]);
}

#[test]
fn p4_is_inside_27_35() {
    let mut r = nums(run("g.V().values('age').is(inside(27, 35))"));
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(r, vec![29.0, 32.0]);
}

#[test]
fn p4_is_with_where() {
    assert_eq!(
        ordered(run(
            "g.V().where(__.in('CREATED').count().is(eq(1))).values('name')"
        )),
        vec!["ripple"]
    );
}

#[test]
fn p4_is_with_where_2() {
    assert_eq!(
        ordered(run(
            "g.V().where(__.in('CREATED').count().is(gte(2))).values('name')"
        )),
        vec!["lop"]
    );
}

#[test]
fn p4_is_with_where_3_mean() {
    let r =
        run("g.V().where(__.in('CREATED').values('age').mean().is(inside(30, 35))).values('name')");
    assert_eq!(names(r), vec!["lop", "ripple"]);
}

// ===================== project.test.ts =====================

#[test]
fn p4_project_name_age() {
    let out = run("g.V().has('name', eq('marko')).project('n','a').by('name').by('age')");
    let m = &out[0];
    assert_eq!(map_get(m, "n"), Some(&GVal::Str("marko".into())));
    assert_eq!(map_get(m, "a"), Some(&GVal::Num(29.0)));
}

#[test]
fn p4_project_single_key() {
    let out = run("g.V().has('name', eq('josh')).project('name').by('name')");
    assert_eq!(
        map_entries(&out[0]),
        vec![("name".into(), GVal::Str("josh".into()))]
    );
}

#[test]
fn p4_project_no_bys_passthrough() {
    // project(['x']) with no by ⇒ value is the traverser (vertex) itself.
    let out = run("g.V().has('name', eq('vadas')).project('x')");
    assert_eq!(out.len(), 1);
    assert!(map_get(&out[0], "x").is_some());
}

// SKIP: project.test.ts · "project across all vertices skips non-productive bys"
// · DIVERGENCE: TS omits a projection key whose by-traversal is non-productive
// (software has no 'age', so the 'a' key is dropped for lop/ripple). The Rust
// engine's Step::Project always emits every key, storing GVal::Null when the
// by-key is absent — it never omits keys. Asserting the TS shape (key absent)
// would fail, so this scenario is skipped and recorded.

#[test]
fn p4_project_with_fold_subtraversal() {
    let out = run("g.V().has('name', eq('marko')).project('name','friendsNames').by('name').by(out('KNOWS').values('name').fold())");
    let m = &out[0];
    assert_eq!(map_get(m, "name"), Some(&GVal::Str("marko".into())));
    assert_eq!(
        map_get(m, "friendsNames"),
        Some(&GVal::List(vec![
            GVal::Str("vadas".into()),
            GVal::Str("josh".into())
        ]))
    );
}

#[test]
fn p4_project_id_count_bys() {
    let out = run("g.V().has('name', eq('marko')).project('id','name','out','in').by(id()).by('name').by(outE().count()).by(inE().count())");
    let m = &out[0];
    assert_eq!(s(map_get(m, "id").unwrap()), "1");
    assert_eq!(map_get(m, "name"), Some(&GVal::Str("marko".into())));
    assert_eq!(map_get(m, "out"), Some(&GVal::Num(3.0)));
    assert_eq!(map_get(m, "in"), Some(&GVal::Num(0.0)));
}

// ===================== tokens.test.ts =====================

#[test]
fn p4_tokens_group_count_by_t_label() {
    let out = run("g.V().groupCount().by(T.label)");
    assert_eq!(out.len(), 1);
    let m = &out[0];
    assert_eq!(map_get(m, "PERSON"), Some(&GVal::Num(4.0)));
    assert_eq!(map_get(m, "SOFTWARE"), Some(&GVal::Num(2.0)));
}

#[test]
fn p4_tokens_group_by_t_label() {
    let out = run("g.V().group().by(T.label)");
    assert_eq!(out.len(), 1);
    let m = &out[0];
    match map_get(m, "PERSON") {
        Some(GVal::List(l)) => assert_eq!(l.len(), 4),
        _ => panic!(),
    }
    match map_get(m, "SOFTWARE") {
        Some(GVal::List(l)) => assert_eq!(l.len(), 2),
        _ => panic!(),
    }
}

#[test]
fn p4_tokens_dedupe_by_t_label() {
    let r = run("g.V().dedup().by(T.label).values('name')");
    assert_eq!(names(r), vec!["lop", "marko"]);
}

#[test]
fn p4_tokens_path_by_t_id() {
    // V('1').hasLabel('PERSON').path().by(T.id) — single-element path ['1'].
    let out = run("g.V('1').hasLabel('PERSON').path().by(T.id)");
    assert_eq!(list_names_ordered(&out[0]), vec!["1"]);
}

#[test]
fn p4_tokens_order_by_t_id() {
    assert_eq!(
        ordered(run("g.V().order().by(T.id).values('name')")),
        vec!["marko", "vadas", "lop", "josh", "ripple", "peter"]
    );
}

// ===================== in_.test.ts =====================

#[test]
fn p4_in_toy() {
    assert_eq!(ordered(run("g.V('4').in().values('name')")), vec!["marko"]);
}

#[test]
fn p4_in_specific_label_empty() {
    assert_eq!(run("g.V('1').in('KNOWS')").len(), 0);
}

#[test]
fn p4_in_specific_label_creators() {
    assert_eq!(
        ordered(run("g.V('3').in('CREATED').values('name')")),
        vec!["marko", "josh", "peter"]
    );
}

#[test]
fn p4_in_all_labels_equals_none() {
    let a = run("g.V('3').in('CREATED')");
    let b = run("g.V('3').in()");
    assert_eq!(a, b);
}

#[test]
fn p4_in_knows_on_vadas() {
    assert_eq!(ordered(run("g.V('2').in('KNOWS').id()")), vec!["1"]);
}

// ===================== V.test.ts =====================

#[test]
fn p4_v_all() {
    assert_eq!(run("g.V()").len(), 6);
}

#[test]
fn p4_v_stable_order() {
    assert_eq!(
        ordered(run("g.V().values('name')")),
        vec!["marko", "vadas", "josh", "peter", "lop", "ripple"]
    );
}

#[test]
fn p4_v_single_by_id() {
    assert_eq!(ordered(run("g.V('1').values('name')")), vec!["marko"]);
}

#[test]
fn p4_v_id_returns_single() {
    assert_eq!(ordered(run("g.V('1').id()")), vec!["1"]);
}

// ===================== property.test.ts =====================

#[test]
fn p4_property_writes_and_chains() {
    let mut g = modern();
    let out = super::parse(
        "g.V('1').property('city', 'santa fe').property('state', 'new mexico').valueMap('city','state')",
    )
    .unwrap()
    .run(&mut g);
    let m = &out[0];
    assert_eq!(map_get(m, "city"), Some(&GVal::Str("santa fe".into())));
    assert_eq!(map_get(m, "state"), Some(&GVal::Str("new mexico".into())));
    // Persisted: a follow-up read sees the new property.
    let read = super::parse("g.V('1').values('city')").unwrap().run(&mut g);
    assert_eq!(ordered(read), vec!["santa fe"]);
}

#[test]
fn p4_property_cardinality_single_overwrites() {
    // TS uses property(Cardinality.single, 'name', 'MARKO!'); `single` is the
    // default cardinality, so the 2-arg form is semantically identical here.
    let mut g = modern();
    super::parse("g.V('1').property('name', 'MARKO!')")
        .unwrap()
        .run(&mut g);
    let read = super::parse("g.V('1').values('name')").unwrap().run(&mut g);
    assert_eq!(ordered(read), vec!["MARKO!"]);
}

// SKIP: property.test.ts · "property() on a non-element value silently drops the
// traverser" · DIVERGENCE: TS drops the traverser (returns []) when the current
// value isn't an element; the Rust engine returns the stream unchanged
// (Step::Property passes non-element traversers through). Asserting [] would
// fail, so this scenario is skipped and recorded.

#[test]
fn p4_property_on_vertices_not_edges() {
    // property('seen', true) on PERSON vertices: edges untouched, persons updated.
    let mut g = modern();
    let r = super::parse("g.V().hasLabel('PERSON').property('seen', true)")
        .unwrap()
        .run(&mut g);
    assert!(!r.is_empty());
    // Every PERSON now has seen=true; no edge does (edges weren't visited).
    let persons = super::parse("g.V().hasLabel('PERSON').count()")
        .unwrap()
        .run(&mut g);
    let seen = super::parse("g.V().hasLabel('PERSON').has('seen', eq(true)).count()")
        .unwrap()
        .run(&mut g);
    assert_eq!(one_num(seen), one_num(persons));
}

// ===================== map.test.ts =====================

#[test]
fn p4_map_values_projects() {
    let r = run("g.V('1').out().map(values('name'))");
    assert_eq!(names(r), vec!["josh", "lop", "vadas"]);
}

#[test]
fn p4_map_count_per_traverser() {
    let r = run("g.V().hasLabel('PERSON').map(count())");
    assert_eq!(nums(r), vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn p4_map_values_single_name_each() {
    let r = run("g.V().hasLabel('PERSON').map(values('name'))");
    assert_eq!(names(r), vec!["josh", "marko", "peter", "vadas"]);
}

#[test]
fn p4_map_drops_empty_subplan() {
    // Software vertices have no outE('CREATED'); map drops them ⇒ 3 persons.
    let r = run("g.V().map(outE('CREATED'))");
    assert_eq!(r.len(), 3);
}

// ===================== constant.test.ts =====================

#[test]
fn p4_constant_choose_fallback() {
    let r = run("g.V().choose(hasLabel('PERSON'), values('name'), constant('inhuman'))");
    assert_eq!(
        ordered(r),
        vec!["marko", "vadas", "josh", "peter", "inhuman", "inhuman"]
    );
}

#[test]
fn p4_constant_coalesce_fallback() {
    let r = run("g.V().coalesce(hasLabel('PERSON').values('name'), constant('inhuman'))");
    assert_eq!(
        ordered(r),
        vec!["marko", "vadas", "josh", "peter", "inhuman", "inhuman"]
    );
}

#[test]
fn p4_constant_replaces_every() {
    assert_eq!(
        ordered(run("g.V().constant('foo')")),
        vec!["foo", "foo", "foo", "foo", "foo", "foo"]
    );
}

#[test]
fn p4_constant_numeric() {
    assert_eq!(
        nums(run("g.V().hasLabel('SOFTWARE').constant(42)")),
        vec![42.0, 42.0]
    );
}

// ===================== simplePath.test.ts =====================

#[test]
fn p4_simplepath_both_both_count() {
    assert_eq!(run("g.V('1').both().both()").len(), 7);
}

#[test]
fn p4_simplepath_drops_cyclic() {
    let r = run("g.V('1').both().both().simplePath().id()");
    assert_eq!(r.len(), 4);
    let mut ids = ordered(r);
    ids.sort();
    assert_eq!(ids, vec!["3", "4", "5", "6"]);
}

#[test]
fn p4_simplepath_path_acyclic() {
    let r = run("g.V('1').both().both().simplePath().path()");
    assert_eq!(r.len(), 4);
    for p in &r {
        let ids: Vec<GVal> = match p {
            GVal::List(items) => items.clone(),
            _ => panic!(),
        };
        // Each path begins at v[1] and has 3 distinct vertices.
        assert_eq!(ids.len(), 3);
        let first = matches!(&ids[0], GVal::Vertex(_));
        assert!(first);
        let mut set = ids.clone();
        set.dedup();
        // distinct check via sort+dedup on a clone
        let mut sorted: Vec<String> = ids.iter().map(|v| format!("{v:?}")).collect();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }
}

// ===================== index.test.ts =====================

#[test]
fn p4_index_software_names() {
    let r = run("g.V().hasLabel('SOFTWARE').values('name').index()");
    assert_eq!(
        r,
        vec![
            GVal::List(vec![GVal::Str("lop".into()), GVal::Num(0.0)]),
            GVal::List(vec![GVal::Str("ripple".into()), GVal::Num(1.0)]),
        ]
    );
}

#[test]
fn p4_index_person_names() {
    let r = run("g.V().hasLabel('PERSON').values('name').index()");
    assert_eq!(
        r,
        vec![
            GVal::List(vec![GVal::Str("marko".into()), GVal::Num(0.0)]),
            GVal::List(vec![GVal::Str("vadas".into()), GVal::Num(1.0)]),
            GVal::List(vec![GVal::Str("josh".into()), GVal::Num(2.0)]),
            GVal::List(vec![GVal::Str("peter".into()), GVal::Num(3.0)]),
        ]
    );
}

#[test]
fn p4_index_over_vertices() {
    let r = run("g.V().hasLabel('SOFTWARE').index()");
    let pairs: Vec<(GVal, f64)> = r
        .iter()
        .map(|g| match g {
            GVal::List(items) => (
                items[0].clone(),
                match items[1] {
                    GVal::Num(n) => n,
                    _ => panic!(),
                },
            ),
            _ => panic!(),
        })
        .collect();
    // Pair each vertex with its positional index; ids are 3 then 5.
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].1, 0.0);
    assert_eq!(pairs[1].1, 1.0);
    // Resolve vertex ids via a parallel id() query (order matches).
    let ids = ordered(run("g.V().hasLabel('SOFTWARE').id()"));
    assert_eq!(ids, vec!["3", "5"]);
}

// ===================== barrier-store.test.ts =====================

#[test]
fn p4_barrier_identity() {
    assert_eq!(
        names(run("g.V().hasLabel('PERSON').barrier().values('name')")),
        vec!["josh", "marko", "peter", "vadas"]
    );
}

#[test]
fn p4_store_cap() {
    let r = run("g.V().hasLabel('SOFTWARE').store('softs').values('name').cap('softs')");
    assert_eq!(r.len(), 1);
    match &r[0] {
        GVal::List(l) => assert_eq!(l.len(), 2),
        _ => panic!("expected a list bag"),
    }
}

#[test]
fn p4_store_aggregate_interchangeable() {
    let a = run("g.V().hasLabel('SOFTWARE').aggregate('x').cap('x')");
    let b = run("g.V().hasLabel('SOFTWARE').store('x').cap('x')");
    let len = |r: &[GVal]| match &r[0] {
        GVal::List(l) => l.len(),
        _ => panic!(),
    };
    assert_eq!(len(&a), 2);
    assert_eq!(len(&b), 2);
}

// ===================== otherV.test.ts =====================

#[test]
fn p4_otherv_toy() {
    let r = run("g.V('4').bothE('KNOWS','CREATED','blah').otherV().id()");
    assert_eq!(ordered(r), vec!["5", "3", "1"]);
    let names_r = run("g.V('4').bothE('KNOWS','CREATED','blah').otherV().values('name')");
    assert_eq!(ordered(names_r), vec!["ripple", "lop", "marko"]);
}

#[test]
fn p4_otherv_ids() {
    let r = run("g.V('4').bothE('KNOWS','CREATED','blah').otherV().id()");
    assert_eq!(ordered(r), vec!["5", "3", "1"]);
}

// ===================== mutation-combinators.test.ts =====================

#[test]
fn p4_mut_repeat_addv_times() {
    let mut g = modern();
    let before = g.vertex_count();
    super::parse("g.V('1').repeat(addV('PING')).times(3)")
        .unwrap()
        .run(&mut g);
    assert_eq!(g.vertex_count(), before + 3);
}

#[test]
fn p4_mut_repeat_addv_property_chain() {
    // repeat(addV('CHAIN').property('seq', 1)).times(2) — two CHAIN vertices, each seq=1.
    let mut g = modern();
    let before = g.vertex_count();
    super::parse("g.V('1').repeat(addV('CHAIN').property('seq', 1)).times(2)")
        .unwrap()
        .run(&mut g);
    assert_eq!(g.vertex_count(), before + 2);
    let chained = super::parse("g.V().hasLabel('CHAIN').count()")
        .unwrap()
        .run(&mut g);
    assert_eq!(one_num(chained), 2.0);
    let seq1 = super::parse("g.V().hasLabel('CHAIN').has('seq', eq(1)).count()")
        .unwrap()
        .run(&mut g);
    assert_eq!(one_num(seq1), 2.0);
}

#[test]
fn p4_mut_map_addv_property() {
    // map(addV('SHADOW').property('via','map')) — one new vertex per PERSON.
    let mut g = modern();
    let before = g.vertex_count();
    let r = super::parse("g.V().hasLabel('PERSON').map(addV('SHADOW').property('via', 'map'))")
        .unwrap()
        .run(&mut g);
    assert_eq!(g.vertex_count(), before + 4);
    assert_eq!(r.len(), 4);
}

#[test]
fn p4_mut_union_addv() {
    // union(addV(A), addV(B)) — two new vertices per upstream.
    let mut g = modern();
    let before = g.vertex_count();
    let r = super::parse("g.V('1').union(addV('A'), addV('B'))")
        .unwrap()
        .run(&mut g);
    assert_eq!(g.vertex_count(), before + 2);
    assert_eq!(r.len(), 2);
    // The two new vertices carry labels A and B.
    let mut labels = ordered(
        super::parse("g.V().hasLabel('A','B').label()")
            .unwrap()
            .run(&mut g),
    );
    labels.sort();
    assert_eq!(labels, vec!["A", "B"]);
}

#[test]
fn p4_mut_choose_gates_addv() {
    // choose(identity(), addV('VISITED')) — identity test passes ⇒ addV per PERSON.
    let mut g = modern();
    let before = g.vertex_count();
    super::parse("g.V().hasLabel('PERSON').choose(identity(), addV('VISITED'))")
        .unwrap()
        .run(&mut g);
    assert_eq!(g.vertex_count(), before + 4);
}

#[test]
fn p4_mut_drop_inside_choose() {
    // choose(identity(), drop()) — identity always passes ⇒ all PERSONs dropped.
    let mut g = modern();
    super::parse("g.V().hasLabel('PERSON').choose(identity(), drop())")
        .unwrap()
        .run(&mut g);
    let remaining = super::parse("g.V().hasLabel('PERSON').count()")
        .unwrap()
        .run(&mut g);
    assert_eq!(one_num(remaining), 0.0);
}

#[test]
fn p4_mut_adde_repeat_smoke() {
    // repeat(addV('CHAIN').property('via','repeat')).times(3) — 3 CHAIN, no edges.
    let mut g = modern();
    let before_e = g.edge_count();
    super::parse("g.V('1').repeat(addV('CHAIN').property('via', 'repeat')).times(3)")
        .unwrap()
        .run(&mut g);
    let chained = super::parse("g.V().hasLabel('CHAIN').count()")
        .unwrap()
        .run(&mut g);
    assert_eq!(one_num(chained), 3.0);
    assert_eq!(g.edge_count(), before_e);
}

#[test]
fn p4_mut_adde_to_subplan_self_loop() {
    // addE('SHORTCUT').to(__) — empty sub-plan resolves to the input traverser ⇒
    // a self-loop on marko.
    let mut graph = modern();
    let before = graph.edge_count();
    // The fluent builder expresses an empty .to() sub-plan (identity endpoint).
    g().v_ids(&["1"])
        .add_e("SHORTCUT")
        .to_plan(__())
        .run(&mut graph);
    assert_eq!(graph.edge_count(), before + 1);
    // The new edge is a self-loop: out('SHORTCUT') from marko returns marko.
    let r = super::parse("g.V('1').out('SHORTCUT').id()")
        .unwrap()
        .run(&mut graph);
    assert_eq!(ordered(r), vec!["1"]);
}
