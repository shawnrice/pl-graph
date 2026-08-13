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
    /// Non-scalar (element/map/list/path) — kept comparable and labelled so a
    /// mismatch surfaces, but flagged so the harness can skip it from the scalar set.
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

fn norm_eng(v: &EngVal) -> Cmp {
    match v {
        EngVal::Null => Cmp::Null,
        EngVal::Bool(b) => Cmp::Bool(*b),
        EngVal::Num(n) => Cmp::Num(num_key(*n)),
        EngVal::Str(s) => Cmp::Str(s.to_string()),
        other => Cmp::Other(format!("{other:?}")),
    }
}

fn norm_core(v: &CoreVal) -> Cmp {
    match v {
        CoreVal::Null => Cmp::Null,
        CoreVal::Bool(b) => Cmp::Bool(*b),
        CoreVal::Num(n) => Cmp::Num(num_key(*n)),
        CoreVal::Str(s) => Cmp::Str(s.to_string()),
        other => Cmp::Other(format!("{other:?}")),
    }
}

/// True when a result set contains a non-scalar (so the scalar harness skips it).
fn has_nonscalar(v: &[Cmp]) -> bool {
    v.iter().any(|c| matches!(c, Cmp::Other(_)))
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

/// Gremlin parity cases the engine does NOT yet match — kept green as known gaps.
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
