//! Core's OWN Gremlin tests, run against BOTH engines. The traversal-building and
//! result helpers are swapped to the `dual` shim (signature-identical to core's
//! `gremlin::Traversal` + `P` + helpers), so each `#[test]` body — copied VERBATIM
//! from `lenke-core/src/gremlin/tests.rs` et al. — builds one traversal that runs on
//! core AND on the engine, and `q(...)` asserts the two agree before the body's own
//! `assert_eq!` checks the expected value. Core is the oracle; the engine must match.

#[path = "support/dual.rs"]
mod dual;

use dual::{g, GVal, Order, Pop, Token, __, P};
use lenke_core::graph::{Graph, Value};
use lenke_core::ndjson;
use lenke_core::value::Value as CoreVal;
use lenke_engine::value::Value as EngVal;

// ── fixtures: the canonical Modern graph in both dialects ────────────────────

const MODERN_CORE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../lenke-core/src/fixtures/modern_gremlin.ndjson"
));

fn core_graph() -> lenke_core::graph::Graph {
    lenke_core::ndjson::decode(MODERN_CORE).expect("core modern fixture")
}

fn engine_store() -> lenke_engine::store::Store {
    engine_store_from(MODERN_CORE)
}

/// Build an engine store from CORE-dialect ndjson (as `lenke_core::ndjson::encode`
/// emits) — so the engine runs on the exact same graph the core test built.
fn engine_store_from(core_ndjson: &str) -> lenke_engine::store::Store {
    let mut out = String::new();
    for line in core_ndjson.lines().filter(|l| !l.trim().is_empty()) {
        out.push_str(&core_line_to_engine(line));
        out.push('\n');
    }
    lenke_engine::ndjson::from_ndjson(&out).expect("engine fixture")
}

/// A traversal that MUTATES the graph — dual-checking it would re-run the write on an
/// already-written graph, so those run on core only (writes are covered by core's own
/// contract tests).
fn is_write(query: &str) -> bool {
    ["addV", "addE", ".drop(", ".property("]
        .iter()
        .any(|w| query.contains(w))
}

fn core_line_to_engine(line: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(line).expect("fixture json");
    let o = v.as_object().expect("obj");
    let props = o
        .get("properties")
        .cloned()
        .unwrap_or(serde_json::json!({}));
    if o.get("type").and_then(|t| t.as_str()) == Some("edge") {
        let label = o["labels"][0].as_str().unwrap_or("");
        let mut m = serde_json::Map::new();
        if let Some(id) = o.get("id").filter(|v| !v.is_null()) {
            m.insert("id".into(), id.clone());
        }
        m.insert("from".into(), o["from"].clone());
        m.insert("to".into(), o["to"].clone());
        m.insert("type".into(), serde_json::json!(label));
        m.insert("props".into(), props);
        serde_json::Value::Object(m).to_string()
    } else {
        serde_json::json!({
            "id": o["id"], "labels": o["labels"], "props": props
        })
        .to_string()
    }
}

// ── a comparable value form ──────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Cmp {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    List(Vec<Cmp>),
    Map(Vec<(Cmp, Cmp)>),
    Other(String),
}

fn num_key(n: f64) -> String {
    if n.is_nan() {
        "nan".into()
    } else if n.is_finite() && n == n.trunc() {
        format!("i{}", n as i64)
    } else {
        format!("f{n:.9}")
    }
}

fn norm_core(v: &CoreVal, g: &lenke_core::graph::Graph) -> Cmp {
    match v {
        CoreVal::Null => Cmp::Null,
        CoreVal::Bool(b) => Cmp::Bool(*b),
        CoreVal::Num(n) => Cmp::Num(num_key(*n)),
        CoreVal::Str(s) => Cmp::Str(s.to_string()),
        CoreVal::Node(id) => Cmp::Str(g.vid.text(*id).to_string()),
        CoreVal::List(xs) => Cmp::List(xs.iter().map(|x| norm_core(x, g)).collect()),
        CoreVal::Map(m) => {
            let mut es: Vec<(Cmp, Cmp)> = m
                .iter()
                .map(|(k, v)| (norm_core(k, g), norm_core(v, g)))
                .collect();
            es.sort();
            Cmp::Map(es)
        }
        other => Cmp::Other(format!("{other:?}")),
    }
}

fn norm_eng(v: &EngVal) -> Cmp {
    match v {
        EngVal::Null => Cmp::Null,
        EngVal::Bool(b) => Cmp::Bool(*b),
        EngVal::Num(n) => Cmp::Num(num_key(*n)),
        EngVal::Str(s) => Cmp::Str(s.to_string()),
        EngVal::List(xs) => Cmp::List(xs.iter().map(norm_eng).collect()),
        // A bare vertex renders as a {id,labels,properties} element map; core renders
        // it as Node→ext-id. Canonicalize both to the ext-id string so bare-element
        // results compare (the engine has no Value::Node).
        EngVal::Map(m) if is_bare_vertex(m) => norm_eng(&vertex_id(m)),
        EngVal::Map(m) => {
            let mut es: Vec<(Cmp, Cmp)> =
                m.iter().map(|(k, v)| (norm_eng(k), norm_eng(v))).collect();
            es.sort();
            Cmp::Map(es)
        }
        other => Cmp::Other(format!("{other:?}")),
    }
}

fn is_bare_vertex(m: &[(EngVal, EngVal)]) -> bool {
    let keys: std::collections::BTreeSet<&str> = m
        .iter()
        .filter_map(|(k, _)| {
            if let EngVal::Str(s) = k {
                Some(s.as_ref())
            } else {
                None
            }
        })
        .collect();
    keys.len() == m.len() && keys == ["id", "labels", "properties"].into_iter().collect()
}

fn vertex_id(m: &[(EngVal, EngVal)]) -> EngVal {
    m.iter()
        .find(|(k, _)| matches!(k, EngVal::Str(s) if s.as_ref() == "id"))
        .map(|(_, v)| v.clone())
        .unwrap_or(EngVal::Null)
}

/// Collapse a bare-vertex `{id,labels,properties}` map to its ext-id anywhere in a `Cmp`
/// tree. Core renders a vertex bag (e.g. `subgraph()`'s `vertices`) as element maps while
/// the engine renders ext-id strings; both denote the same vertex set, and the harness
/// contract compares vertices by ext-id (there is no `Value::Node`). Idempotent on the
/// engine side, whose bare vertices `norm_eng` already reduced.
fn collapse_bare(c: Cmp) -> Cmp {
    match c {
        Cmp::List(xs) => Cmp::List(xs.into_iter().map(collapse_bare).collect()),
        Cmp::Map(es) => {
            let keys: std::collections::BTreeSet<&str> = es
                .iter()
                .filter_map(|(k, _)| if let Cmp::Str(s) = k { Some(s.as_str()) } else { None })
                .collect();
            let is_vertex = keys.len() == es.len()
                && keys == ["id", "labels", "properties"].into_iter().collect();
            if is_vertex {
                let id = es
                    .into_iter()
                    .find(|(k, _)| matches!(k, Cmp::Str(s) if s == "id"))
                    .map(|(_, v)| v)
                    .unwrap_or(Cmp::Null);
                collapse_bare(id)
            } else {
                Cmp::Map(
                    es.into_iter()
                        .map(|(k, v)| (collapse_bare(k), collapse_bare(v)))
                        .collect(),
                )
            }
        }
        other => other,
    }
}

/// True when a value contains a raw element the normalizer can't canonicalize — such a
/// case can't be differentially compared (the engine has no Value::Node), so it's
/// checked on CORE only (the body's own assert still runs).
fn has_other(c: &Cmp) -> bool {
    match c {
        Cmp::Other(_) => true,
        Cmp::List(xs) => xs.iter().any(has_other),
        Cmp::Map(es) => es.iter().any(|(k, v)| has_other(k) || has_other(v)),
        _ => false,
    }
}

// ── the dual runner: core's `q`, but asserting engine == core ────────────────

fn modern() -> lenke_core::graph::Graph {
    core_graph()
}

/// Assert the engine agrees with core's result for `query`. Skips the compare only
/// when core produced a raw element (no `Value::Node` in the engine) — the body's own
/// assertions on `core_res` still run. Compared order-independently (Gremlin order is
/// unspecified without an explicit `order()`; ordered tests use `ordered()` on core).
fn assert_engine_matches(query: &str, core_res: &[GVal], store: &lenke_engine::store::Store, cg: &lenke_core::graph::Graph) {
    if is_write(query) {
        return; // writes run on core only (see is_write)
    }
    let core_cmp: Vec<Cmp> = core_res.iter().map(|v| norm_core(v, cg)).collect();
    if core_cmp.iter().any(has_other) {
        return;
    }
    match lenke_engine::gremlin::parse(query) {
        Ok(plan) => {
            let rows = lenke_engine::exec::run(&plan, store);
            let mut a: Vec<Cmp> = core_cmp.into_iter().map(collapse_bare).collect();
            let mut b: Vec<Cmp> = rows
                .rows
                .iter()
                .flatten()
                .map(norm_eng)
                .map(collapse_bare)
                .collect();
            a.sort();
            b.sort();
            assert_eq!(a, b, "engine != core for `{query}`");
        }
        Err(e) => panic!("engine cannot parse `{query}`: {e}"),
    }
}

/// Build once, run on BOTH engines, assert they agree, return core's result for the
/// test body's own `assert_eq!`.
fn q(t: dual::Traversal) -> Vec<GVal> {
    let query = t.query();
    let mut g = core_graph();
    let store = engine_store();
    let core_res = t.run(&mut g);
    assert_engine_matches(&query, &core_res, &store, &g);
    core_res
}

// A dual-running `super::parse(...)` replacement: parses on core (so `.unwrap()`/
// `.is_err()` behave exactly as core's), and `.run()` also runs the string on the
// engine and asserts agreement. `parse().is_err()` cases additionally check the engine
// rejects the same string (parse-parity).
struct ParsedT {
    query: String,
    core: lenke_core::gremlin::Traversal,
}

fn parse(query: &str) -> Result<ParsedT, String> {
    match lenke_core::gremlin::parse(query) {
        Ok(core) => Ok(ParsedT {
            query: query.to_string(),
            core,
        }),
        Err(e) => {
            assert!(
                lenke_engine::gremlin::parse(query).is_err(),
                "core rejects `{query}` but engine accepts it"
            );
            Err(e)
        }
    }
}

impl ParsedT {
    fn run(&self, graph: &mut lenke_core::graph::Graph) -> Vec<GVal> {
        // Run the engine on the SAME graph core uses (encode → engine store).
        let store = engine_store_from(&lenke_core::ndjson::encode(graph));
        let core_res = self.core.run(graph);
        assert_engine_matches(&self.query, &core_res, &store, graph);
        core_res
    }
}

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

fn s(g: &GVal) -> String {
    match g {
        GVal::Str(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

fn names(r: Vec<GVal>) -> Vec<String> {
    let mut v: Vec<String> = r.iter().map(s).collect();
    v.sort();
    v
}

fn ordered(r: Vec<GVal>) -> Vec<String> {
    r.iter().map(s).collect()
}

fn one_num(r: Vec<GVal>) -> f64 {
    match r.as_slice() {
        [GVal::Num(n)] => *n,
        _ => panic!("expected single number, got {r:?}"),
    }
}

// ── core-contract shims (fault codes / JSON are core-internal, not dual-checked) ──

trait CoreRef {
    fn cref(&self) -> &lenke_core::gremlin::Traversal;
}
impl CoreRef for dual::Traversal {
    fn cref(&self) -> &lenke_core::gremlin::Traversal {
        self.core_ref()
    }
}
impl CoreRef for ParsedT {
    fn cref(&self) -> &lenke_core::gremlin::Traversal {
        &self.core
    }
}

/// Core's fallible run (returns its fault-code Result). These verify CORE's error
/// contract, which the engine does not reproduce byte-for-byte by design — so they run
/// on core only; every value-producing read still dual-checks via `q`/`assert_engine_matches`.
fn try_run(
    g: &mut lenke_core::graph::Graph,
    t: &impl CoreRef,
) -> lenke_core::error::CodeResult<Vec<GVal>> {
    lenke_core::gremlin::try_run(g, t.cref())
}

fn results_to_json(g: &lenke_core::graph::Graph, vals: &[GVal]) -> String {
    lenke_core::gremlin::exec::results_to_json(g, vals)
}

// ── core's Gremlin tests, ported verbatim (bodies unchanged; builder/helpers shimmed) ──

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
    let mut g = lenke_core::ndjson::decode(&lines.join("\n")).unwrap();
    let r = parse("g.V('a').repeat(__.out()).emit().count()")
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
    let mut g = lenke_core::ndjson::decode(&lines.join("\n")).unwrap();
    let r = parse("g.E('e-b','e-a').id()").unwrap().run(&mut g);
    let ids: Vec<String> = r
        .iter()
        .map(|v| match v {
            GVal::Str(s) => s.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(ids, vec!["e-b", "e-a"]);
}

/// The builder form of [`p1_out_label_order_does_not_group_the_result`] — see
/// that test for why label-argument grouping is not a contract.
#[test]
fn out_multi_label_returns_the_same_set_either_order() {
    let one = names(q(g()
        .v_ids(&["1"])
        .out(&["CREATED", "KNOWS"])
        .values(&["name"])));
    let other = names(q(g()
        .v_ids(&["1"])
        .out(&["KNOWS", "CREATED"])
        .values(&["name"])));

    assert_eq!(one, other);
    assert_eq!(one, vec!["josh", "lop", "vadas"]);
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
            let name = s(&m.values()[0]);
            let created = match m.values()[1] {
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
    assert_eq!(s(&m.values()[0]), "1");
    assert_eq!(m.values()[1], GVal::Num(3.0)); // out-degree
    assert_eq!(m.values()[2], GVal::Num(0.0)); // in-degree
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
    assert_eq!(s(&m.values()[0]), "marko");
    assert_eq!(s(&m.values()[1]), "lop");
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
    assert_eq!(
        m.get(&GVal::Str("name".into())),
        Some(&GVal::Str("marko".into()))
    );
    assert_eq!(m.get(&GVal::Str("age".into())), Some(&GVal::Num(29.0)));
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
    assert_eq!(vm, vec![GVal::map(vec![(GVal::from("nick"), GVal::Null)])]);
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
        parse("g.V().as('a').out('KNOWS').dedup('a').select('a').values('name')").unwrap();
    assert_eq!(by_label.run(&mut g), vec![GVal::Str("marko".into())]);

    let mut g2 = modern();
    let by_value = parse("g.V().as('a').out('KNOWS').dedup().select('a').values('name')").unwrap();
    assert_eq!(by_value.run(&mut g2).len(), 2); // dedup-by-value keeps both
}

#[test]
fn drop_cannot_be_spoofed_by_a_project_map() {
    // Regression: a `project('key')` result is a Map with a `key` entry; it must
    // NOT be mistaken for a property element by drop() (that would delete an
    // arbitrary property). The owner now rides the `Property` element itself, so
    // a Map can never spoof one. Before the fix this deleted `age` everywhere.
    let mut g = modern();
    let t = parse("g.V().project('key').by(constant('age')).drop()").unwrap();
    let _ = t.run(&mut g);
    // All four PERSON vertices keep their age — nothing was deleted.
    let ages = parse("g.V().values('age').count()").unwrap();
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
    let t = parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

/// Alias of [`qs`] — some core step-test files name the string runner `run`.
#[allow(dead_code)] // used by later ported step-test files
fn run(query: &str) -> Vec<GVal> {
    qs(query)
}

/// Parse + run with vertex indexes declared on the CORE graph. Index seeding is a
/// planner OPTIMIZATION — it must not change the result set — so the engine (whose
/// planner seeks automatically) still matches core row-for-row via the dual check.
fn q_vidx(indexes: &[&str], query: &str) -> Vec<GVal> {
    let mut g = modern();
    for k in indexes {
        g.create_vertex_index(k);
    }
    let t = parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

const MODERN_EIDS_CORE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../lenke-core/src/fixtures/modern_gremlin_edge_ids.ndjson"
));

/// The Modern graph whose edges carry EXTERNAL ids (for `g.E('id')`, `id()` on edges).
fn modern_eids() -> lenke_core::graph::Graph {
    lenke_core::ndjson::decode(MODERN_EIDS_CORE).expect("core modern edge-ids fixture")
}

/// Parse + run against the edge-id Modern graph (dual-checked, like [`qs`]).
fn qs_e(query: &str) -> Vec<GVal> {
    let mut g = modern_eids();
    let t = parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
    t.run(&mut g)
}

/// Alias of [`qs_e`] — some core step-test files name the edge-id runner `qs_eids`.
#[allow(dead_code)]
fn qs_eids(query: &str) -> Vec<GVal> {
    qs_e(query)
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

/// Resolve element-ids in a result list of vertices/edges (core-side).
#[allow(dead_code)]
fn ids(g: &Graph, r: &[GVal]) -> Vec<String> {
    r.iter()
        .map(|v| match v {
            GVal::Node(i) => g.vid.text(*i).to_string(),
            GVal::Edge(e) => format!("e{e}"),
            other => format!("{other:?}"),
        })
        .collect()
}

/// Assert an `elementMap`/map result equals the wanted entries (order-independent).
#[allow(dead_code)]
fn assert_emap(got: &GVal, want: &[(&str, GVal)]) {
    let m = map_sorted(got);
    let mut w: Vec<(String, GVal)> = want
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    w.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(m, w);
}

/// Look up a numeric count keyed by `key` in a single-result group map.
#[allow(dead_code)]
fn map_get_num(r: &[GVal], key: &GVal) -> Option<f64> {
    match r.first() {
        Some(GVal::Map(entries)) => entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| match v {
                GVal::Num(n) => *n,
                other => panic!("expected num value, got {other:?}"),
            }),
        other => panic!("expected map, got {other:?}"),
    }
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
    let t = parse("g.V().sack()").unwrap();
    assert_eq!(
        try_run(&mut g, &t).unwrap_err().code,
        lenke_core::error_codes::ErrorCode::InvalidGraphOp
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
        parse(q)
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
    assert!(parse("g.V().has('d', duration('P1D')).values('n')").is_ok());
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
    assert_eq!(m.values()[0], GVal::Num(3.0));
}

#[test]
fn parse_select_and_as() {
    let r = qs("g.V().has('name','marko').as('a').out('CREATED').as('b').select('a','b').by('name').by('name')");
    let m = match &r[0] {
        GVal::Map(e) => e,
        _ => panic!(),
    };
    assert_eq!(s(&m.values()[0]), "marko");
    assert_eq!(s(&m.values()[1]), "lop");
}

#[test]
fn parse_union_and_coalesce() {
    let r = qs("g.V().has('name','marko').union(__.values('name'), __.values('age'))");
    assert_eq!(r, vec![GVal::Str("marko".into()), GVal::Num(29.0)]);
}

#[test]
fn parse_to_json_round_trip() {
    let mut g = modern();
    let t = parse("g.V().hasLabel('PERSON').order().by('name').values('name')").unwrap();
    let vals = t.run(&mut g);
    let json = results_to_json(&g, &vals);
    assert_eq!(json, r#"["josh","marko","peter","vadas"]"#);
}

#[test]
fn parse_vertex_json_has_id_label() {
    let mut g = modern();
    let t = parse("g.V('1')").unwrap();
    let vals = t.run(&mut g);
    let json = results_to_json(&g, &vals);
    // Full `{id, labels, properties}` form — byte-identical to GQL `RETURN n`.
    assert_eq!(
        json,
        r#"[{"id":"1","labels":["PERSON"],"properties":{"age":29,"name":"marko"}}]"#
    );
}

// ===== property-index seeding (results must equal the scan path) =====

/// Run a query against a fresh Modern graph with the given vertex indexes built.
fn q_idx(indexes: &[&str], t: dual::Traversal) -> Vec<GVal> {
    let query = t.query();
    let mut g = modern();
    for k in indexes {
        g.create_vertex_index(k);
    }
    let core_res = t.run(&mut g);
    assert_engine_matches(&query, &core_res, &engine_store(), &g);
    core_res
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
fn sp_paths(t: dual::Traversal) -> Vec<Vec<String>> {
    let query = t.query();
    let mut g = modern();
    let core_res = t.run(&mut g);
    assert_engine_matches(&query, &core_res, &engine_store(), &g);
    core_res
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
    assert!(parse(&q).is_err());
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
        assert!(parse(q).is_err(), "expected a parse error for `{q}`");
    }
}

#[test]
fn parser_rejects_non_integer_counts() {
    for q in ["g.V().limit(-5)", "g.V().limit(2.5)", "g.V().range(0, -1)"] {
        assert!(parse(q).is_err(), "expected a parse error for `{q}`");
    }
}

#[test]
fn parser_valid_counts_still_parse() {
    for q in [
        "g.V().limit(3)",
        "g.V().range(1, 4)",
        "g.V().repeat(out()).times(2)",
    ] {
        assert!(parse(q).is_ok(), "expected `{q}` to parse");
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
    let mut g = lenke_core::ndjson::decode(&lines.join("\n")).unwrap();
    let t = parse("g.V().repeat(both())").unwrap();
    let err = try_run(&mut g, &t).unwrap_err();
    assert_eq!(
        err.code,
        lenke_core::error_codes::ErrorCode::ResourceExhausted
    );
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
    let mut g = lenke_core::ndjson::decode(&lines.join("\n")).unwrap();
    let t = parse("g.V().order().by('p')").unwrap();
    assert_eq!(
        try_run(&mut g, &t).unwrap_err().code,
        lenke_core::error_codes::ErrorCode::InvalidValue
    );
    // Infallible path: best-effort, but must not panic.
    let _ = t.run(&mut g);
}

#[test]
fn lexer_preserves_utf8_string_literals() {
    let lines = [r#"{"type":"node","id":"1","labels":["P"],"properties":{"name":"café"}}"#];
    let mut g = lenke_core::ndjson::decode(&lines.join("\n")).unwrap();
    let t = parse("g.V().has('name','café').values('name')").unwrap();
    assert_eq!(t.run(&mut g), vec![GVal::Str("café".into())]);
}

#[test]
fn lexer_decodes_string_escapes() {
    let mut g = modern();
    let t = parse(r"g.inject('a\nb')").unwrap();
    assert_eq!(t.run(&mut g), vec![GVal::Str("a\nb".into())]);
}

// --- G7-G9 (Rust): TinkerPop Comparable semantics — throw on incomparable ----

#[test]
fn comparison_of_incomparable_types_faults() {
    let mut g = modern();
    // names are strings; gt(5) compares them to a number → incomparable.
    let t = parse("g.V().values('name').is(gt(5))").unwrap();
    assert_eq!(
        try_run(&mut g, &t).unwrap_err().code,
        lenke_core::error_codes::ErrorCode::InvalidValue
    );
}

#[test]
fn addv_and_property_reject_malformed_names() {
    use lenke_core::error_codes::ErrorCode::InvalidValue;
    let mut g = modern();
    // Gremlin takes arbitrary label/key strings, so a `::` label / empty key is
    // guarded at the step (codec ingestion has its own gate). try_run surfaces it.
    let bad = [
        "g.addV('a::b')",        // GraphSON multi-label separator in a label
        "g.addV('')",            // empty label
        "g.V().property('', 1)", // empty property key
    ];
    for src in bad {
        let t = parse(src).unwrap();
        assert_eq!(try_run(&mut g, &t).unwrap_err().code, InvalidValue, "{src}");
    }
    // A well-formed addV/property is fine.
    assert!(try_run(&mut g, &parse("g.addV('Robot')").unwrap()).is_ok());
}

#[test]
fn order_over_mixed_types_faults() {
    let mut g = modern();
    let t = parse("g.inject(3, 'a', 1).order()").unwrap();
    assert_eq!(
        try_run(&mut g, &t).unwrap_err().code,
        lenke_core::error_codes::ErrorCode::InvalidValue
    );
}

#[test]
fn sum_of_non_numeric_faults() {
    let mut g = modern();
    let t = parse("g.V().values('name').sum()").unwrap();
    assert_eq!(
        try_run(&mut g, &t).unwrap_err().code,
        lenke_core::error_codes::ErrorCode::InvalidValue
    );
}

#[test]
fn comparable_predicate_and_aggregation_still_work() {
    let mut g = modern();
    // age > 30 → josh(32), peter(35) → count 2; no coercion, no fault.
    let t = parse("g.V().values('age').is(gt(30)).count()").unwrap();
    assert_eq!(try_run(&mut g, &t).unwrap(), vec![GVal::Num(2.0)]);
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
    let t = parse("g.V().values('name').math('_ + 1')").unwrap();
    assert!(try_run(&mut g, &t).is_err());
}

#[test]
fn math_malformed_expression_faults() {
    let mut g = modern();
    let t = parse("g.inject(1).math('_ +')").unwrap();
    assert!(try_run(&mut g, &t).is_err());
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
    let t = parse("g.inject(1).math('nope(_)')").unwrap();
    assert!(try_run(&mut g, &t).is_err());
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
    let t = parse("g.inject(1).math('atan2 _')").unwrap();
    assert!(try_run(&mut g, &t).is_err());
}

#[test]
fn math_bare_form_variable_shadows_function() {
    // A bound tag `sin` wins over the sine function even in the bare position:
    // `sin` resolves to the variable, leaving `_` as trailing input → fault
    // (byte-identical to TS). With just `sin`, it returns the variable.
    let r = q(g().inject([GVal::Num(42.0)]).as_("sin").math("sin"));
    assert_eq!(one_num(r), 42.0);
    let mut g2 = modern();
    let t = parse("g.inject(42).as('sin').math('sin _')").unwrap();
    assert!(try_run(&mut g2, &t).is_err());
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
    let t = parse(
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
    let t = parse("g.V().has('name', TextP.regex('^r')).values('name')").unwrap();
    assert_eq!(ordered(t.run(&mut g)), vec!["ripple"]);
}

#[test]
fn regex_invalid_pattern_is_a_parse_error() {
    // Validated at parse time (like the TS `regex()` constructor), not per value.
    assert!(parse("g.V().has('name', regex('['))").is_err());
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
    results_to_json(&modern(), &vals)
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
    assert_eq!(results_json(vec![GVal::map(vec![])]), "[{}]");
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

    // Map keys in INSERTION order — the order the map was built in.
    //
    // This used to sort lexicographically, to match `serde_json::Map`. The sync
    // layer that needed it wants a DETERMINISTIC order, not a sorted one, and
    // every map is built from a `Vec`. Sorting made this renderer disagree with
    // GQL's, with this module's own `push_result_value`, and with the TS engine,
    // and it silently undid `order(local)` on a map.
    assert_eq!(
        results_json(vec![GVal::map(vec![
            (GVal::from("zzz"), GVal::from("z")),
            (GVal::from("age"), GVal::from("a")),
            (GVal::from("name"), GVal::from("m")),
        ])]),
        r#"[{"zzz":"z","age":"a","name":"m"}]"#
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
        results_json(vec![GVal::map(vec![(GVal::Num(5.0), GVal::from("v"))])]),
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
    let scores =
        parse("g.withComputer().V().pageRank().values('gremlin.pageRankVertexProgram.pageRank')")
            .unwrap()
            .run(&mut modern());
    assert_eq!(scores.len(), 6);

    let comps = parse(
        "g.V().connectedComponent().values('gremlin.connectedComponentVertexProgram.component').dedup()",
    )
    .unwrap()
    .run(&mut modern());
    assert_eq!(comps.len(), 1);

    let pr = parse("g.V().pageRank(0.85).with(PageRank.propertyName, 'pr').values('pr')")
        .unwrap()
        .run(&mut modern());
    assert_eq!(pr.len(), 6);

    let clusters = parse(
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
    let err = parse("g.V().pageRank().with(PageRank.edges, __.outE())");
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
            assert!(matches!(&pairs.values()[0], GVal::Str(s) if s.as_ref() == "NYC"));
        }
        other => panic!("expected one GVal::Map, got {other:?}"),
    }
}
// ==== 80 tests from step_tests_1.rs ====
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

/// Multi-label `out()` returns the same SET whichever order the labels are
/// given in. It does not group by label argument.
///
/// It used to: `adj_in_label_order` materialized a `Vec` per direction per source
/// vertex and re-scanned it once per label so that `out('CREATED','KNOWS')`
/// emitted the CREATED edges first. Nothing requires that. TinkerPop specifies no
/// order for `out()`; TinkerGraph appears to group by argument only because it
/// stores `Map<label, Set<Edge>>`, which the TS engine mirrors and the native CSR
/// store does not — and this repo's policy already records adjacency order as
/// unspecified and native-vs-TS divergence there as expected. Paying a per-vertex
/// allocation to imitate another engine's storage layout bought nothing, and cost
/// a second adjacency walk outside `lenke_core::seek`.
#[test]
fn p1_out_label_order_does_not_group_the_result() {
    let mut a = ordered(qs("g.V('1').out('CREATED','KNOWS').values('name')"));
    let mut b = ordered(qs("g.V('1').out('KNOWS','CREATED').values('name')"));

    a.sort();
    b.sort();

    assert_eq!(a, b);
    assert_eq!(a, vec!["josh", "lop", "vadas"]);
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
    let r = dual::g()
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
    let t = dual::g()
        .V()
        .has("name", P::not(P::within(["vadas", "marko"])))
        .values(&["name"]);
    assert_eq!(
        ordered(t.run(&mut g)),
        vec!["josh", "peter", "lop", "ripple"]
    );
}

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

#[test]
fn p1_value_unwraps_properties() {
    assert_eq!(
        ordered(qs("g.V().hasId('1').properties('name').value()")),
        vec!["marko"]
    );
}

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
    let t = parse("g.E().has('weight', 1.0).id()").unwrap();
    let mut got = ordered(t.run(&mut indexed));
    let mut want = ordered(
        parse("g.E().has('weight', 1.0).id()")
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
            parse("g.E().has('weight', eq(0.4)).count()")
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
    let mut got = nums(parse(q).unwrap().run(&mut indexed));
    let mut want = nums(parse(q).unwrap().run(&mut plain));
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got, want);
}

// ==== 76 tests from step_tests_2.rs ====
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
    assert_eq!(r, vec![GVal::list(vec![])]);
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

#[test]
fn p2_adde_to_subplan() {
    // marko -[NEMESIS]-> peter; input is FROM, sub-plan is TO.
    let mut g = modern();
    let before = q(dual::g().E().count());
    let r = parse("g.V('1').addE('NEMESIS').to(__.V('6'))")
        .unwrap()
        .run(&mut g);
    assert_eq!(r.len(), 1);
    // edge count went up by one
    let after = dual::g().E().count().run(&mut g);
    assert_eq!(after, vec![GVal::Num(7.0)]);
    assert_eq!(before, vec![GVal::Num(6.0)]);
    // the new edge connects marko -> peter with label NEMESIS
    let names_out = parse("g.V('1').out('NEMESIS').values('name')")
        .unwrap()
        .run(&mut g);
    assert_eq!(ordered(names_out), vec!["peter"]);
}

#[test]
fn p2_adde_from_tag() {
    // tag marko, hop to out-neighbors, addE('META').from('start').to(V('6')).
    let mut g = modern();
    let r =
        parse("g.V('1').as('start').out('KNOWS').addE('META').from('start').to(__.V('6'))")
            .unwrap()
            .run(&mut g);
    assert_eq!(r.len(), 2); // marko knows vadas + josh → 2 edges
    let count = dual::g().E().count().run(&mut g);
    assert_eq!(count, vec![GVal::Num(8.0)]);
    // both new META edges go marko -> peter
    let metas = parse("g.V('1').out('META').values('name')")
        .unwrap()
        .run(&mut g);
    assert_eq!(names(metas), vec!["peter", "peter"]);
}

#[test]
fn p2_adde_with_property() {
    let mut g = modern();
    parse("g.V('1').addE('KNOWS').to(__.V('6')).property('weight', 0.42)")
        .unwrap()
        .run(&mut g);
    let w = parse("g.V('1').outE('KNOWS').has('weight', eq(0.42)).values('weight')")
        .unwrap()
        .run(&mut g);
    assert_eq!(w, vec![GVal::Num(0.42)]);
}

#[test]
fn p2_add_e_unresolvable_endpoint_faults() {
    let mut g = modern();
    let t = parse("g.V('1').addE('NEMESIS').to(__.V('999'))").unwrap();
    let err = try_run(&mut g, &t).unwrap_err();
    assert_eq!(err.code, lenke_core::error_codes::ErrorCode::MissingVertex);
}

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

#[test]
fn p2_fail_throws_with_message() {
    // fail() on a non-empty stream is a DataException surfaced by try_run —
    // carrying the user's message — NOT a process-aborting panic. TS throws a
    // catchable error here too. (`run` ignores it, matching the addV/addE faults.)
    let mut g = modern();
    let t =
        parse("g.V().hasLabel('PERSON').has('name', eq('peter')).fold().fail('Test Fail')")
            .unwrap();
    let err = try_run(&mut g, &t).unwrap_err();
    assert_eq!(err.code, lenke_core::error_codes::ErrorCode::DataException);
    assert!(err.message.contains("Test Fail"), "got: {}", err.message);
}

#[test]
fn p2_fail_no_throw_on_empty_stream() {
    // Empty stream: fail() is a pass-through — no fault, even via try_run.
    let mut g = modern();
    let t = parse("g.V().has('name', eq('nobody')).fail('should not fire')").unwrap();
    assert!(try_run(&mut g, &t).unwrap().is_empty());
}

#[test]
fn p2_fail_default_message() {
    let mut g = modern();
    let t = parse("g.V().fail()").unwrap();
    let err = try_run(&mut g, &t).unwrap_err();
    assert_eq!(err.code, lenke_core::error_codes::ErrorCode::DataException);
    assert!(
        err.message.contains("fail() reached"),
        "got: {}",
        err.message
    );
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

// ==== 71 tests from step_tests_3.rs ====
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
    assert_eq!(r[0], GVal::list(vec![]));
}

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
        parse("g.V('1').repeat(__.addV('REP').property('via', 'rep')).times(2)").unwrap();
    let _ = t.run(&mut g);
    assert_eq!(g.vertex_count(), before + 2);
}

#[test]
fn p3_subplan_map_body_adds_vertices() {
    // map(addV('SHADOW')...) over the four people adds 4 vertices.
    let mut g = modern();
    let before = g.vertex_count();
    let t = parse("g.V().hasLabel('PERSON').map(__.addV('SHADOW').property('via', 'map'))")
        .unwrap();
    let _ = t.run(&mut g);
    assert_eq!(g.vertex_count(), before + 4);
}

#[test]
fn p3_subplan_repeat_until_times_zero_smoke() {
    // repeat(identity).until(count().is(eq(0))).times(0) — smoke: doesn't panic.
    let mut g = modern();
    let t = parse("g.V('1').repeat(__.identity()).until(__.count().is(eq(0))).times(0)")
        .unwrap();
    let r = t.run(&mut g);
    // smoke: a Vec is produced.
    let _ = r.len();
}

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

#[test]
fn p3_e_by_external_id() {
    assert_eq!(ordered(qs_eids("g.E('7').id()")), vec!["7"]);
    assert_eq!(ordered(qs_eids("g.E('11').id()")), vec!["11"]);
}

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

#[test]
fn p3_drop_vertex_removes_and_emits_nothing() {
    let mut g = modern();
    let before = g.vertex_count();
    let r = parse("g.V('2').drop()").unwrap().run(&mut g);
    assert_eq!(r, Vec::<GVal>::new());
    assert_eq!(g.vertex_count(), before - 1);
    // vadas (id 2) is gone.
    let mut g2 = g;
    let cnt = parse("g.V().has('name', 'vadas').count()")
        .unwrap()
        .run(&mut g2);
    assert_eq!(cnt, vec![GVal::Num(0.0)]);
}

#[test]
fn p3_drop_vertex_cascades_incident_edges() {
    let mut g = modern();
    let edges_before = g.edge_count();
    // marko (id 1) has 3 incident edges.
    let _ = parse("g.V('1').drop()").unwrap().run(&mut g);
    assert_eq!(g.edge_count(), edges_before - 3);
}

#[test]
fn p3_drop_edges_leaves_vertices() {
    let mut g = modern();
    let v_before = g.vertex_count();
    let _ = parse("g.E().hasLabel('CREATED').drop()")
        .unwrap()
        .run(&mut g);
    assert_eq!(g.vertex_count(), v_before);
    // No CREATED edges remain.
    let mut g2 = g;
    let cnt = parse("g.E().hasLabel('CREATED').count()")
        .unwrap()
        .run(&mut g2);
    assert_eq!(cnt, vec![GVal::Num(0.0)]);
}

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

