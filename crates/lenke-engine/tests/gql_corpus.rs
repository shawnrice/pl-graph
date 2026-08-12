//! Shared GQL conformance corpus — one set of cases, both engines.
//!
//! Cases are extracted (mechanically) from lenke-core's behavioral GQL tests into
//! `tests/gql_corpus/*.jsonl`. Each case carries a core-dialect NDJSON `fixture` and
//! a read `query`; the runner loads the fixture into BOTH lenke-core (the reference)
//! and lenke-engine (converting the fixture to the engine's dialect so ONE fixture
//! drives both), runs the query on each, and asserts the engine's result matches
//! core's. Core's own inline tests still pin core to the spec; this extends that same
//! query surface to the engine.
//!
//! A case is JSON: `{ "name", "fixture" (ndjson string), "query", "ordered"? }`.
//! When core rejects the query (parse/exec error), the engine must reject it too.
//! Comparison is by VALUE (multiset unless `ordered`), through `num_key` (exact
//! integers, 1e-9 otherwise) — cross-engine float bit-identity is not claimed.

use lenke_core::gql::eval::Params as CoreParams;
use lenke_core::graph::Value as CoreVal;
use lenke_engine::value::Value as EngVal;
use serde_json::Value as J;
use std::path::PathBuf;

// ── a value form both engines' cells map into ────────────────────────────────
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Cell {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Other(String),
}
fn num_key(n: f64) -> String {
    if n == 0.0 {
        "0".into() // normalize -0.0
    } else if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("i{}", n as i64)
    } else if n.is_nan() {
        "nan".into()
    } else {
        format!("f{n:.9}")
    }
}
fn norm_eng(v: &EngVal) -> Cell {
    match v {
        EngVal::Null => Cell::Null,
        EngVal::Bool(b) => Cell::Bool(*b),
        EngVal::Num(n) => Cell::Num(num_key(*n)),
        EngVal::Str(s) => Cell::Str(s.to_string()),
        o => Cell::Other(format!("{o:?}")),
    }
}
fn norm_core(v: &CoreVal) -> Cell {
    match v {
        CoreVal::Null => Cell::Null,
        CoreVal::Bool(b) => Cell::Bool(*b),
        CoreVal::Num(n) => Cell::Num(num_key(*n)),
        CoreVal::Str(s) => Cell::Str(s.to_string()),
        o => Cell::Other(format!("{o:?}")),
    }
}

#[derive(PartialEq)]
enum Outcome {
    Rows(Vec<Vec<Cell>>),
    Err,
}

// ── fixture conversion: core-dialect NDJSON → engine dialect ─────────────────
/// One core NDJSON line → the engine's dialect (`{"id","labels","props"}` for a
/// node, `{"id"?,"from","to","type","props"}` for an edge). Non-data lines (schema
/// ops the engine loader does not need) pass through as-is if unrecognized. Returns
/// `None` to drop a line that is not node/edge data.
fn core_line_to_engine(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    let j: J = serde_json::from_str(line).ok()?;
    let obj = j.as_object()?;
    let ty = obj.get("type").and_then(J::as_str).unwrap_or("");
    let props = obj
        .get("properties")
        .cloned()
        .unwrap_or_else(|| J::Object(Default::default()));
    match ty {
        "node" => {
            let mut out = serde_json::Map::new();
            out.insert("id".into(), obj.get("id").cloned().unwrap_or(J::Null));
            out.insert(
                "labels".into(),
                obj.get("labels").cloned().unwrap_or(J::Array(vec![])),
            );
            out.insert("props".into(), props);
            Some(J::Object(out).to_string())
        }
        "edge" => {
            let mut out = serde_json::Map::new();
            if let Some(id) = obj.get("id") {
                out.insert("id".into(), id.clone());
            }
            out.insert("from".into(), obj.get("from").cloned().unwrap_or(J::Null));
            out.insert("to".into(), obj.get("to").cloned().unwrap_or(J::Null));
            // core edge type is the (single) entry of `labels`; engine uses `type`.
            let etype = obj
                .get("labels")
                .and_then(J::as_array)
                .and_then(|a| a.first())
                .cloned()
                .unwrap_or(J::String("".into()));
            out.insert("type".into(), etype);
            out.insert("props".into(), props);
            Some(J::Object(out).to_string())
        }
        _ => None, // schema / unknown line: the engine fixture does not need it
    }
}

/// The TinkerPop "Modern" graph (core-dialect NDJSON) — many core tests use it, so
/// a case may set `"fixture": "@modern"` instead of inlining these 12 lines.
const MODERN: &str = include_str!("../../lenke-core/src/fixtures/modern_gql.ndjson");

/// Resolve a `fixture` field: `@modern` → the Modern graph, else the string is the
/// fixture's core-dialect NDJSON verbatim.
fn resolve_fixture(fixture: &str) -> &str {
    match fixture.trim() {
        "@modern" => MODERN,
        other => other,
    }
}

fn run_core(fixture: &str, query: &str) -> Outcome {
    let fixture = resolve_fixture(fixture);
    let mut g = match lenke_core::ndjson::decode(fixture) {
        Ok(g) => g,
        Err(e) => panic!("core fixture failed to decode: {e}"),
    };
    let Ok(prep) = lenke_core::gql::prepare(query) else {
        return Outcome::Err;
    };
    match prep.execute(&mut g, &CoreParams::new()) {
        Ok(rs) => Outcome::Rows(
            rs.rows()
                .map(|r| r.iter().map(norm_core).collect())
                .collect(),
        ),
        Err(_) => Outcome::Err,
    }
}

fn run_engine(fixture: &str, query: &str) -> Outcome {
    let fixture = resolve_fixture(fixture);
    let eng_nd: String = fixture
        .lines()
        .filter_map(core_line_to_engine)
        .collect::<Vec<_>>()
        .join("\n");
    let store = match lenke_engine::ndjson::from_ndjson(&eng_nd) {
        Ok(s) => s,
        Err(e) => panic!("engine fixture failed to load: {e}"),
    };
    let Ok(plan) = lenke_engine::gql::parse(query) else {
        return Outcome::Err;
    };
    let plan = lenke_engine::opt::optimize_indexed(plan, &store);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lenke_engine::exec::try_run(&plan, &store)
    })) {
        Ok(Ok(rows)) => Outcome::Rows(
            rows.rows
                .iter()
                .map(|r| r.iter().map(norm_eng).collect())
                .collect(),
        ),
        _ => Outcome::Err,
    }
}

fn eq_outcome(a: &Outcome, b: &Outcome, ordered: bool) -> bool {
    match (a, b) {
        (Outcome::Err, Outcome::Err) => true,
        (Outcome::Rows(x), Outcome::Rows(y)) => {
            if ordered {
                x == y
            } else {
                let mut xs = x.clone();
                let mut ys = y.clone();
                xs.sort();
                ys.sort();
                xs == ys
            }
        }
        _ => false,
    }
}

#[test]
fn gql_corpus_engine_matches_core() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/gql_corpus");
    if !dir.exists() {
        eprintln!("no corpus dir yet: {}", dir.display());
        return;
    }
    let mut total = 0usize;
    let mut skipped_core_err = 0usize;
    let mut fails: Vec<String> = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    for path in files {
        let text = std::fs::read_to_string(&path).expect("read corpus file");
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let fname = path.file_name().unwrap().to_string_lossy().to_string();
            let case: J = match serde_json::from_str(line) {
                Ok(c) => c,
                Err(e) => {
                    fails.push(format!("{fname}:{}: bad JSON: {e}", lineno + 1));
                    continue;
                }
            };
            let name = case.get("name").and_then(J::as_str).unwrap_or("?").to_string();
            let fixture = case.get("fixture").and_then(J::as_str).unwrap_or("").to_string();
            let query = case.get("query").and_then(J::as_str).unwrap_or("").to_string();
            let ordered = case.get("ordered").and_then(J::as_bool).unwrap_or(false);
            total += 1;
            // A fixture that fails to load or a run that panics is recorded, not fatal —
            // one bad case must not abort the whole corpus run.
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let core = run_core(&fixture, &query);
                let eng = run_engine(&fixture, &query);
                (core, eng)
            }));
            let (core, eng) = match ran {
                Ok(p) => p,
                Err(_) => {
                    fails.push(format!("{fname}::{name}: panicked (fixture/run) for: {query}"));
                    continue;
                }
            };
            // A query core itself rejects is not an engine-conformance case; still
            // require the engine to reject it too (error parity).
            if matches!(core, Outcome::Err) {
                skipped_core_err += 1;
                if !matches!(eng, Outcome::Err) {
                    fails.push(format!("{fname}::{name}: core rejects but engine accepts: {query}"));
                }
                continue;
            }
            if !eq_outcome(&core, &eng, ordered) {
                fails.push(format!("{fname}::{name}: engine != core for: {query}"));
            }
        }
    }
    eprintln!(
        "gql corpus: {total} cases, {skipped_core_err} core-rejected (error-parity checked), {} mismatches",
        fails.len()
    );
    assert!(
        fails.is_empty(),
        "{} corpus mismatches:\n{}",
        fails.len(),
        fails
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
