//! Gremlin differential parity: run a corpus of traversals against BOTH the
//! columnar `lenke-engine` and the reference `lenke-core`, over the canonical
//! TinkerPop "Modern" graph, and assert identical results. Each case is authored
//! once as a neutral step AST (`support/gremlin_ast.rs`) and lowered to the engine's
//! Gremlin string and to a core `Traversal` — so neither side's expected values are
//! hand-written. A case the engine cannot parse/run, or that diverges, is a real
//! parity gap; the baseline in `KNOWN_GREMLIN_GAPS` records the ones not yet closed.
//!
//! This is the Gremlin counterpart of `gql_corpus.rs`. Scalar-returning traversals
//! only, for now (element/map/list canonicalization is the next expansion).

#[path = "support/gremlin_ast.rs"]
mod ast;

use ast::{Pred, Step, Steps, Val};
use lenke_core::gremlin::Order as COrder;
use lenke_core::value::Value as CoreVal;
use lenke_engine::value::Value as EngVal;

// ── the Modern fixture, embedded once, emitted in both dialects ──────────────

/// (external id, label, name, age?, lang?)
type NodeRow = (
    &'static str,
    &'static str,
    &'static str,
    Option<f64>,
    Option<&'static str>,
);
const NODES: &[NodeRow] = &[
    ("1", "PERSON", "marko", Some(29.0), None),
    ("2", "PERSON", "vadas", Some(27.0), None),
    ("4", "PERSON", "josh", Some(32.0), None),
    ("6", "PERSON", "peter", Some(35.0), None),
    ("3", "SOFTWARE", "lop", None, Some("java")),
    ("5", "SOFTWARE", "ripple", None, Some("java")),
];

/// (edge id, from, to, label, weight)
const EDGES: &[(&str, &str, &str, &str, f64)] = &[
    ("7", "1", "2", "KNOWS", 0.5),
    ("8", "1", "4", "KNOWS", 1.0),
    ("9", "1", "3", "CREATED", 0.4),
    ("10", "4", "5", "CREATED", 1.0),
    ("11", "4", "3", "CREATED", 0.4),
    ("12", "6", "3", "CREATED", 0.2),
];

fn node_props(name: &str, age: Option<f64>, lang: Option<&str>, prop_key: &str) -> String {
    let mut fields = vec![format!(r#""name":"{name}""#)];
    if let Some(a) = age {
        fields.push(format!(r#""age":{a}"#));
    }
    if let Some(l) = lang {
        fields.push(format!(r#""lang":"{l}""#));
    }
    let _ = prop_key;
    fields.join(",")
}

fn engine_ndjson() -> String {
    let mut s = String::new();
    for (id, label, name, age, lang) in NODES {
        s.push_str(&format!(
            r#"{{"id":"{id}","labels":["{label}"],"props":{{{}}}}}"#,
            node_props(name, *age, *lang, "props")
        ));
        s.push('\n');
    }
    for (id, from, to, label, w) in EDGES {
        s.push_str(&format!(
            r#"{{"id":"{id}","from":"{from}","to":"{to}","type":"{label}","props":{{"weight":{w}}}}}"#
        ));
        s.push('\n');
    }
    s
}

fn core_ndjson() -> String {
    let mut s = String::new();
    for (id, label, name, age, lang) in NODES {
        s.push_str(&format!(
            r#"{{"type":"node","id":"{id}","labels":["{label}"],"properties":{{{}}}}}"#,
            node_props(name, *age, *lang, "properties")
        ));
        s.push('\n');
    }
    for (id, from, to, label, w) in EDGES {
        s.push_str(&format!(
            r#"{{"type":"edge","id":"{id}","labels":["{label}"],"from":"{from}","to":"{to}","properties":{{"weight":{w}}}}}"#
        ));
        s.push('\n');
    }
    s
}

// ── a comparable value form (scalars; elements/maps deferred) ────────────────

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Cmp {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    /// A list — element order PRESERVED (fold/path are order-bearing).
    List(Vec<Cmp>),
    /// A map — entries SORTED by key (content parity; insertion order is a
    /// separate concern the ordered cases don't exercise through a bare map).
    Map(Vec<(Cmp, Cmp)>),
    /// A raw element (`Node`/`Edge`) or any value the normalizer can't reduce to a
    /// canonical scalar. Its presence means the case must project to `id()`/`values`
    /// to be comparable — the harness flags it rather than guessing an identity.
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

fn sorted_map(mut entries: Vec<(Cmp, Cmp)>) -> Cmp {
    entries.sort();
    Cmp::Map(entries)
}

fn norm_eng(v: &EngVal) -> Cmp {
    match v {
        EngVal::Null => Cmp::Null,
        EngVal::Bool(b) => Cmp::Bool(*b),
        EngVal::Num(n) => Cmp::Num(num_key(*n)),
        EngVal::Str(s) => Cmp::Str(s.to_string()),
        EngVal::List(xs) => Cmp::List(xs.iter().map(norm_eng).collect()),
        EngVal::Map(m) => sorted_map(m.iter().map(|(k, v)| (norm_eng(k), norm_eng(v))).collect()),
        other => Cmp::Other(format!("{other:?}")),
    }
}

fn norm_core(v: &CoreVal) -> Cmp {
    match v {
        CoreVal::Null => Cmp::Null,
        CoreVal::Bool(b) => Cmp::Bool(*b),
        CoreVal::Num(n) => Cmp::Num(num_key(*n)),
        CoreVal::Str(s) => Cmp::Str(s.to_string()),
        CoreVal::List(xs) => Cmp::List(xs.iter().map(norm_core).collect()),
        CoreVal::Map(m) => sorted_map(
            m.iter()
                .map(|(k, v)| (norm_core(k), norm_core(v)))
                .collect(),
        ),
        other => Cmp::Other(format!("{other:?}")),
    }
}

/// True when any value (at any depth) is an un-canonicalizable `Other` — the case
/// must project elements to `id()`/`values` to be comparable, so the harness skips it.
fn has_other(c: &Cmp) -> bool {
    match c {
        Cmp::Other(_) => true,
        Cmp::List(xs) => xs.iter().any(has_other),
        Cmp::Map(es) => es.iter().any(|(k, v)| has_other(k) || has_other(v)),
        _ => false,
    }
}

fn has_nonscalar(v: &[Cmp]) -> bool {
    v.iter().any(has_other)
}

// ── run both engines ─────────────────────────────────────────────────────────

fn run_engine(store: &lenke_engine::store::Store, query: &str) -> Result<Vec<Cmp>, String> {
    let plan = lenke_engine::gremlin::parse(query)?;
    let rows = lenke_engine::exec::run(&plan, store);
    Ok(rows.rows.iter().flatten().map(norm_eng).collect())
}

fn run_core(graph: &mut lenke_core::graph::Graph, steps: &Steps) -> Vec<Cmp> {
    steps.to_core().run(graph).iter().map(norm_core).collect()
}

struct Case {
    name: &'static str,
    steps: Steps,
    ordered: bool,
}

fn c(name: &'static str, ordered: bool, steps: Vec<Step>) -> Case {
    Case {
        name,
        steps: Steps(steps),
        ordered,
    }
}

/// Gremlin parity cases the engine does NOT yet match — UNFINISHED, MEASURED gaps
/// (not "won't-fix"), each naming concrete engine work the harness verifies the
/// moment it's closed. Empty = every case in the corpus is at parity.
const KNOWN_GREMLIN_GAPS: &[&str] = &[];

#[test]
fn gremlin_parity() {
    use Step::*;
    let store = lenke_engine::ndjson::from_ndjson(&engine_ndjson()).expect("engine fixture");
    let mut graph = lenke_core::ndjson::decode(&core_ndjson()).expect("core fixture");

    let cases: Vec<Case> = vec![
        // — sources & simple hops —
        c("v_count", false, vec![V, Count]),
        c("v_values_name", false, vec![V, Values(vec!["name"])]),
        c(
            "person_names",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["name"])],
        ),
        c(
            "marko_out_names",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["KNOWS"]),
                Values(vec!["name"]),
            ],
        ),
        c(
            "marko_out_created",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["CREATED"]),
                Values(vec!["name"]),
            ],
        ),
        c(
            "both_names",
            false,
            vec![
                V,
                HasVal("name", Val::S("lop")),
                Both(vec!["CREATED"]),
                Values(vec!["name"]),
            ],
        ),
        c(
            "in_names",
            false,
            vec![
                V,
                HasVal("name", Val::S("lop")),
                In(vec!["CREATED"]),
                Values(vec!["name"]),
            ],
        ),
        // — predicates —
        c(
            "age_gt_30",
            false,
            vec![V, Has("age", Pred::Gt(Val::N(30.0))), Values(vec!["name"])],
        ),
        c(
            "age_gte_32",
            false,
            vec![V, Has("age", Pred::Gte(Val::N(32.0))), Values(vec!["name"])],
        ),
        c(
            "age_lt_30",
            false,
            vec![V, Has("age", Pred::Lt(Val::N(30.0))), Values(vec!["name"])],
        ),
        c(
            "age_within",
            false,
            vec![
                V,
                Has("age", Pred::Within(vec![Val::N(29.0), Val::N(35.0)])),
                Values(vec!["name"]),
            ],
        ),
        c(
            "age_between",
            false,
            vec![
                V,
                Has("age", Pred::Between(Val::N(30.0), Val::N(40.0))),
                Values(vec!["name"]),
            ],
        ),
        c(
            "is_gt_28",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["age"]),
                Is(Pred::Gt(Val::N(28.0))),
            ],
        ),
        c(
            "name_starts_m",
            false,
            vec![
                V,
                Has("name", Pred::StartingWith("m")),
                Values(vec!["name"]),
            ],
        ),
        // — labels / ids —
        c("labels", false, vec![V, Label]),
        c("ids", false, vec![V, Id]),
        c(
            "out_ids",
            false,
            vec![V, HasVal("name", Val::S("marko")), Out(vec!["KNOWS"]), Id],
        ),
        // — dedup / order / limit / range —
        c(
            "created_names_dedup",
            false,
            vec![V, Out(vec!["CREATED"]), Values(vec!["name"]), Dedup],
        ),
        c(
            "order_names_asc",
            true,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["name"]), Order],
        ),
        c(
            "order_age_desc_limit2",
            true,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Order,
                ByKey("age", COrder::Desc),
                Values(vec!["name"]),
                Limit(2),
            ],
        ),
        c(
            "names_limit3",
            false,
            vec![V, Values(vec!["name"]), Limit(3)],
        ),
        // — aggregates —
        c(
            "count_persons",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Count],
        ),
        c(
            "sum_ages",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["age"]), Sum],
        ),
        c(
            "max_age",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["age"]), Max],
        ),
        c(
            "min_age",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["age"]), Min],
        ),
        c(
            "mean_age",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["age"]), Mean],
        ),
        // — multi-hop —
        c(
            "marko_out_out_names",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["KNOWS"]),
                Out(vec!["CREATED"]),
                Values(vec!["name"]),
            ],
        ),
        // — multi-label edge step (known engine gap) —
        c(
            "both_multi_label",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["KNOWS", "CREATED"]),
                Values(vec!["name"]),
            ],
        ),
        // — element identity via id() (ext-id strings, canonical on both sides) —
        c("all_ids", false, vec![V, Id]),
        c("person_ids", false, vec![V, HasLabel(vec!["PERSON"]), Id]),
        c(
            "marko_out_ids",
            false,
            vec![V, HasVal("name", Val::S("marko")), Out(vec!["KNOWS"]), Id],
        ),
        c(
            "created_in_ids",
            false,
            vec![V, HasLabel(vec!["SOFTWARE"]), In(vec!["CREATED"]), Id],
        ),
        // — maps of scalars (valueMap / elementMap) —
        c(
            "valuemap_name",
            false,
            vec![V, HasLabel(vec!["PERSON"]), ValueMap(vec!["name"])],
        ),
        c(
            "valuemap_name_age",
            false,
            vec![V, HasLabel(vec!["PERSON"]), ValueMap(vec!["name", "age"])],
        ),
        c(
            "elementmap_marko",
            false,
            vec![V, HasVal("name", Val::S("marko")), ElementMap(vec!["name"])],
        ),
        // — lists (fold) —
        c(
            "names_fold",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["name"]), Fold],
        ),
        c(
            "ages_ordered_fold",
            true,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["age"]),
                Order,
                Fold,
            ],
        ),
        // — groupCount —
        c(
            "groupcount_lang",
            false,
            vec![V, HasLabel(vec!["SOFTWARE"]), GroupCount, By("lang")],
        ),
        c("groupcount_label", false, vec![V, Label, GroupCount]),
        // — path projected to names —
        c(
            "path_names",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["KNOWS"]),
                Path,
                By("name"),
            ],
        ),
        // — dedup count / where —
        c(
            "created_names_count",
            false,
            vec![V, Out(vec!["CREATED"]), Values(vec!["name"]), Dedup, Count],
        ),
        c(
            "where_knows_out",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Where(vec![Out(vec!["KNOWS"])]),
                Values(vec!["name"]),
            ],
        ),
        // — untyped hops —
        c(
            "out_untyped_names",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec![]),
                Values(vec!["name"]),
            ],
        ),
        c(
            "both_untyped_count",
            false,
            vec![V, HasVal("name", Val::S("marko")), Both(vec![]), Count],
        ),
        // — has variants —
        c(
            "hasnot_lang",
            false,
            vec![V, HasNot("lang"), Values(vec!["name"])],
        ),
        c(
            "has_neq",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Has("name", Pred::Neq(Val::S("marko"))),
                Values(vec!["name"]),
            ],
        ),
        c(
            "has_within_names",
            false,
            vec![
                V,
                Has("name", Pred::Within(vec![Val::S("marko"), Val::S("lop")])),
                Values(vec!["name"]),
            ],
        ),
        c(
            "has_without_lang",
            false,
            vec![
                V,
                HasLabel(vec!["SOFTWARE"]),
                Has("lang", Pred::Without(vec![Val::S("python")])),
                Values(vec!["name"]),
            ],
        ),
        c(
            "is_within",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["age"]),
                Is(Pred::Within(vec![Val::N(29.0), Val::N(35.0)])),
            ],
        ),
        // — as/select —
        c(
            "as_select_names",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                As("p"),
                Values(vec!["name"]),
                As("n"),
                Select(vec!["n"]),
            ],
        ),
        // — project —
        c(
            "project_name_age",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Project(vec!["n", "a"]),
                By("name"),
                By("age"),
            ],
        ),
        // — union —
        c(
            "union_names_ages",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Union(vec![
                    vec![Values(vec!["name"])],
                    vec![Out(vec!["KNOWS"]), Values(vec!["name"])],
                ]),
            ],
        ),
        // — coalesce —
        c(
            "coalesce_lang_name",
            false,
            vec![
                V,
                Coalesce(vec![vec![Values(vec!["lang"])], vec![Values(vec!["name"])]]),
            ],
        ),
        // — optional —
        c(
            "optional_out",
            false,
            vec![
                V,
                HasVal("name", Val::S("vadas")),
                Optional(vec![Out(vec!["KNOWS"])]),
                Values(vec!["name"]),
            ],
        ),
        // — local —
        c(
            "local_out_count",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Local(vec![Out(vec!["CREATED"]), Count]),
            ],
        ),
        // — repeat/times —
        c(
            "repeat_out_2_ids",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Repeat(vec![Out(vec![])]),
                Times(1),
                Id,
            ],
        ),
        // — dedup / order variants —
        c(
            "order_age_asc_names",
            true,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Order,
                ByKey("age", COrder::Asc),
                Values(vec!["name"]),
            ],
        ),
        c(
            "order_value_desc",
            true,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["age"]),
                Order,
                ByValue(COrder::Desc),
            ],
        ),
        c(
            "dedup_lang",
            false,
            vec![V, HasLabel(vec!["SOFTWARE"]), Values(vec!["lang"]), Dedup],
        ),
        // — count/sum/min/max on hop —
        c(
            "out_created_count",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Out(vec!["CREATED"]), Count],
        ),
        c(
            "sum_weights",
            false,
            vec![V, OutE(vec!["CREATED"]), Values(vec!["weight"]), Sum],
        ),
        // — valueMap / elementMap variants —
        c(
            "valuemap_all_software",
            false,
            vec![V, HasLabel(vec!["SOFTWARE"]), ValueMap(vec![])],
        ),
        c(
            "elementmap_lang",
            false,
            vec![V, HasLabel(vec!["SOFTWARE"]), ElementMap(vec!["lang"])],
        ),
        // — unfold / constant / inject —
        c(
            "fold_unfold",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["name"]),
                Fold,
                Unfold,
            ],
        ),
        c(
            "constant_x",
            false,
            vec![V, HasVal("name", Val::S("marko")), Constant(Val::S("x"))],
        ),
        // — simplePath / cyclicPath —
        c(
            "simplepath_count",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["KNOWS"]),
                SimplePath,
                Count,
            ],
        ),
        // — identity / barrier —
        c(
            "identity_names",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Identity, Values(vec!["name"])],
        ),
        c(
            "barrier_names",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Values(vec!["name"]), Barrier],
        ),
        // — subgraph('sg').cap('sg') — {vertices, edges} element-record Map —
        c(
            "subgraph_knows",
            false,
            vec![V, OutE(vec!["KNOWS"]), Subgraph("sg"), Cap("sg")],
        ),
        c(
            "subgraph_created",
            false,
            vec![V, OutE(vec!["CREATED"]), Subgraph("sg"), Cap("sg")],
        ),
        // — branch(values('k')).option(m, hop).option(none, hop) — value-routed —
        c(
            "branch_by_lang",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Branch(
                    "age",
                    vec![
                        (Some(Val::N(29.0)), vec![Out(vec!["KNOWS"])]),
                        (None, vec![Out(vec!["CREATED"])]),
                    ],
                ),
                Values(vec!["name"]),
            ],
        ),
        // — tagged where('a', op('b')): compare two step-label values (by identity) —
        c(
            "where_key_neq",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                As("a"),
                Out(vec!["KNOWS"]),
                As("b"),
                WhereKeyTag("a", "neq", "b"),
                Values(vec!["name"]),
            ],
        ),
        c(
            "where_key_eq_self",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                As("a"),
                Out(vec!["KNOWS"]),
                As("b"),
                WhereKeyTag("a", "eq", "a"),
                Values(vec!["name"]),
            ],
        ),
        // — sack: per-traverser accumulator —
        c(
            "sack_sum_age",
            false,
            vec![
                WithSack(Val::N(100.0)),
                V,
                HasLabel(vec!["PERSON"]),
                SackBy("sum", "age"),
                SackRead,
            ],
        ),
        c(
            "sack_assign_age",
            false,
            vec![
                WithSack(Val::N(0.0)),
                V,
                HasLabel(vec!["PERSON"]),
                SackBy("assign", "age"),
                SackRead,
            ],
        ),
        c(
            "sack_read_init",
            false,
            vec![
                WithSack(Val::N(7.0)),
                V,
                HasVal("name", Val::S("marko")),
                SackRead,
            ],
        ),
        // — dedup(labels): keyed distinct on a tagged value —
        c(
            "dedup_by_lang_tag",
            false,
            vec![
                V,
                HasLabel(vec!["SOFTWARE"]),
                Values(vec!["lang"]),
                As("l"),
                DedupLabels(vec!["l"]),
            ],
        ),
        c(
            "dedup_created_targets",
            false,
            vec![
                V,
                Out(vec!["CREATED"]),
                As("c"),
                DedupLabels(vec!["c"]),
                Values(vec!["name"]),
            ],
        ),
        // — tree(): nested-map fold of vertex-hop paths —
        c(
            "tree_by_name",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["KNOWS"]),
                Out(vec!["CREATED"]),
                Tree,
                By("name"),
            ],
        ),
        c(
            "tree_persons_by_name",
            false,
            vec![V, HasLabel(vec!["PERSON"]), Tree, By("name")],
        ),
        // NOTE: bare tree() keys by the vertex ELEMENT — the engine renders an element
        // map, core keys by the vertex itself (the no-Value::Node gap), so it is not
        // canonically comparable here; the tree().by('k') forms above are.
        // — inject: add literal values to the whole stream —
        c(
            "inject_one",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["name"]),
                Inject(vec![Val::S("x")]),
            ],
        ),
        c(
            "inject_many",
            false,
            vec![
                V,
                HasLabel(vec!["SOFTWARE"]),
                Values(vec!["name"]),
                Inject(vec![Val::S("a"), Val::S("b")]),
            ],
        ),
        c(
            "inject_count",
            false,
            vec![V, Values(vec!["name"]), Inject(vec![Val::S("x")]), Count],
        ),
        // — tail(local, k): last k of each list cell —
        c(
            "tail_local_2",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["age"]),
                Order,
                Fold,
                TailLocal(2),
            ],
        ),
        // — select with by-modulators (projects each tagged element to a property) —
        c(
            "select_by_name",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                As("a"),
                Out(vec!["KNOWS"]),
                As("b"),
                Select(vec!["a", "b"]),
                By("name"),
            ],
        ),
        c(
            "select_one_by_name",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                As("a"),
                Out(vec!["KNOWS"]),
                Select(vec!["a"]),
                By("name"),
            ],
        ),
        // — OLAP annotate (pageRank / connectedComponent / peerPressure) —
        c(
            "pagerank_scores",
            false,
            vec![
                V,
                PageRank(None),
                Values(vec!["gremlin.pageRankVertexProgram.pageRank"]),
            ],
        ),
        c(
            "pagerank_alpha",
            false,
            vec![
                V,
                PageRank(Some(0.85)),
                Values(vec!["gremlin.pageRankVertexProgram.pageRank"]),
            ],
        ),
        c(
            "connected_component_ids",
            false,
            vec![
                V,
                ConnectedComponent,
                Values(vec!["gremlin.connectedComponentVertexProgram.component"]),
            ],
        ),
        c(
            "connected_component_dedup",
            false,
            vec![
                V,
                ConnectedComponent,
                Values(vec!["gremlin.connectedComponentVertexProgram.component"]),
                Dedup,
            ],
        ),
        c(
            "peer_pressure_clusters",
            false,
            vec![
                V,
                PeerPressure,
                Values(vec!["gremlin.peerPressureVertexProgram.cluster"]),
                Dedup,
            ],
        ),
        // — aggregate/store + cap —
        c(
            "aggregate_cap_names",
            false,
            vec![
                V,
                HasLabel(vec!["PERSON"]),
                Values(vec!["name"]),
                Aggregate("x"),
                Cap("x"),
            ],
        ),
        c(
            "store_cap_names",
            false,
            vec![
                V,
                HasLabel(vec!["SOFTWARE"]),
                Values(vec!["name"]),
                Store("s"),
                Cap("s"),
            ],
        ),
        c(
            "aggregate_passthrough",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Out(vec!["CREATED"]),
                Aggregate("x"),
                Values(vec!["name"]),
            ],
        ),
        // NOTE: store('s').values(...).cap('s') where the bag holds VERTICES then
        // projects id() is NOT covered — the engine represents a bagged vertex as its
        // dense-id Num (no Value::Node), so a later id() can't resolve it. Cases that
        // aggregate/cap SCALARS (above) are at parity; element-identity-through-a-bag
        // is the same representation gap the corpus sidesteps by projecting first.
        // — multi-step union with value bodies —
        c(
            "union_multi_step",
            false,
            vec![
                V,
                HasVal("name", Val::S("marko")),
                Union(vec![
                    vec![Out(vec!["KNOWS"]), Values(vec!["name"])],
                    vec![Out(vec!["CREATED"]), Values(vec!["name"])],
                ]),
            ],
        ),
    ];

    let mut diverged: Vec<String> = Vec::new();
    let mut skipped_nonscalar = 0usize;
    let mut compared = 0usize;

    for case in &cases {
        let query = case.steps.to_query();
        let core_res = run_core(&mut graph, &case.steps);
        if has_nonscalar(&core_res) {
            skipped_nonscalar += 1;
            continue;
        }
        let eng_res = match run_engine(&store, &query) {
            Ok(r) => r,
            Err(e) => {
                if !KNOWN_GREMLIN_GAPS.contains(&case.name) {
                    diverged.push(format!("{} — ENGINE ERROR: {e}  [{query}]", case.name));
                }
                continue;
            }
        };
        compared += 1;
        let (mut a, mut b) = (eng_res.clone(), core_res.clone());
        if !case.ordered {
            a.sort();
            b.sort();
        }
        if a != b && !KNOWN_GREMLIN_GAPS.contains(&case.name) {
            diverged.push(format!(
                "{} — DIVERGE\n    engine: {a:?}\n    core:   {b:?}\n    [{query}]",
                case.name
            ));
        }
    }

    eprintln!(
        "gremlin parity: {} cases, {compared} compared, {skipped_nonscalar} skipped (non-scalar), {} diverge",
        cases.len(),
        diverged.len()
    );
    assert!(
        diverged.is_empty(),
        "gremlin parity divergences:\n{}",
        diverged.join("\n")
    );
}
