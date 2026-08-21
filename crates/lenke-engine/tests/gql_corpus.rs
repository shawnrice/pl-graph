//! GQL conformance corpus — an engine regression snapshot.
//!
//! ~1200 curated GQL read/write cases (`tests/gql_corpus/*.jsonl`), each a
//! core-dialect NDJSON `fixture` and a `query`. This was a differential against
//! `lenke-core`; core has since been deleted, its byte-identity contract now upheld
//! by the TS engine fuzzers. The frozen `snapshots.jsonl` — captured while core still
//! existed and the differential was green, so each recorded outcome equals core's
//! spec-anchored answer — is now the oracle: the engine must reproduce every case's
//! recorded outcome (rows, compared as a multiset unless `ordered`, or a rejection).
//!
//! A case is JSON `{ "name", "fixture" (ndjson string or `@modern`), "query",
//! "ordered"? }`. Comparison is by VALUE through `num_key` (exact integers, 1e-9
//! otherwise) — cross-run float bit-identity is not claimed.
//!
//! Regenerate the snapshot with `CORPUS_SNAPSHOT=1 cargo test -p lenke-engine --test
//! gql_corpus` after an INTENDED behavior change (review the diff — an unexplained
//! change there is a regression). A new case with no snapshot fails until regenerated.

use lenke_engine::value::Value as EngVal;
use serde_json::Value as J;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── a value form engine cells map into ───────────────────────────────────────
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

#[derive(Clone, PartialEq, Debug)]
enum Outcome {
    Rows(Vec<Vec<Cell>>),
    Err,
}

// ── snapshot (de)serialization: Outcome ⇄ JSON ───────────────────────────────
fn cell_to_json(c: &Cell) -> J {
    match c {
        Cell::Null => J::Null,
        Cell::Bool(b) => J::Bool(*b),
        Cell::Num(s) => serde_json::json!({ "n": s }),
        Cell::Str(s) => serde_json::json!({ "s": s }),
        Cell::Other(s) => serde_json::json!({ "o": s }),
    }
}
fn cell_from_json(j: &J) -> Cell {
    match j {
        J::Null => Cell::Null,
        J::Bool(b) => Cell::Bool(*b),
        J::Object(m) if m.contains_key("n") => Cell::Num(m["n"].as_str().unwrap_or("").to_string()),
        J::Object(m) if m.contains_key("s") => Cell::Str(m["s"].as_str().unwrap_or("").to_string()),
        J::Object(m) if m.contains_key("o") => {
            Cell::Other(m["o"].as_str().unwrap_or("").to_string())
        }
        other => Cell::Other(other.to_string()),
    }
}
fn outcome_to_json(o: &Outcome) -> J {
    match o {
        Outcome::Err => serde_json::json!({ "err": true }),
        Outcome::Rows(rows) => serde_json::json!({
            "rows": rows.iter().map(|r| J::Array(r.iter().map(cell_to_json).collect())).collect::<Vec<_>>()
        }),
    }
}
fn outcome_from_json(j: &J) -> Outcome {
    if j.get("err").and_then(J::as_bool) == Some(true) {
        return Outcome::Err;
    }
    let rows = j
        .get("rows")
        .and_then(J::as_array)
        .map(|rows| {
            rows.iter()
                .map(|r| {
                    r.as_array()
                        .map(|cells| cells.iter().map(cell_from_json).collect())
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default();
    Outcome::Rows(rows)
}

// ── fixture conversion: core-dialect NDJSON → engine dialect ─────────────────
/// One core NDJSON line → the engine's dialect (`{"id","labels","props"}` for a
/// node, `{"id"?,"from","to","labels","props"}` for an edge). Returns `None` to drop
/// a line that is not node/edge data (a schema op the engine loader does not need).
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
            // An edge's type is its FIRST label; any further labels are secondary
            // (multi-label edges). Pass the whole `labels` array through — the engine
            // ndjson loader reads the first as the type and the rest as extras.
            let labels = obj
                .get("labels")
                .and_then(J::as_array)
                .cloned()
                .unwrap_or_default();
            out.insert("labels".into(), J::Array(labels));
            out.insert("props".into(), props);
            Some(J::Object(out).to_string())
        }
        _ => None, // schema / unknown line: the engine fixture does not need it
    }
}

/// The TinkerPop "Modern" graph (core-dialect NDJSON) — many cases set
/// `"fixture": "@modern"` instead of inlining these lines. Vendored into the engine
/// test tree (was `../lenke-core/src/fixtures/modern_gql.ndjson`).
const MODERN: &str = include_str!("fixtures/modern_gql.ndjson");

/// Resolve a `fixture` field: `@modern` → the Modern graph, else the string is the
/// fixture's core-dialect NDJSON verbatim.
fn resolve_fixture(fixture: &str) -> &str {
    match fixture.trim() {
        "@modern" => MODERN,
        other => other,
    }
}

fn run_engine(fixture: &str, query: &str) -> Outcome {
    let fixture = resolve_fixture(fixture);
    let eng_nd: String = fixture
        .lines()
        .filter_map(core_line_to_engine)
        .collect::<Vec<_>>()
        .join("\n");
    let mut store = match lenke_engine::ndjson::from_ndjson(&eng_nd) {
        Ok(s) => s,
        Err(e) => panic!("engine fixture failed to load: {e}"),
    };
    let Ok(plan) = lenke_engine::gql::parse(query) else {
        return Outcome::Err;
    };
    let plan = lenke_engine::opt::optimize_indexed(plan, &store);
    // `execute` handles both reads and writes: a write mutates the (fresh, single-use)
    // store and may RETURN rows (`INSERT … RETURN`); a read falls through to `try_run`
    // over a shared borrow. Routing everything through it keeps write-then-return cases
    // comparable to how the snapshot was captured.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lenke_engine::exec::execute(&plan, &mut store)
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

/// One corpus case as loaded from a `*.jsonl` line.
struct Case {
    key: String, // `file::name`
    fixture: String,
    query: String,
    ordered: bool,
}

/// Load every `*.jsonl` case in the corpus dir, in a stable (sorted) order.
fn load_cases(dir: &Path) -> Vec<Case> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        // `snapshots.jsonl` is the recorded oracle, not a corpus file — never a case.
        .filter(|p| p.file_name().is_some_and(|n| n != "snapshots.jsonl"))
        .collect();
    files.sort();
    let mut cases = Vec::new();
    for path in files {
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        let text = std::fs::read_to_string(&path).expect("read corpus file");
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let case: J = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("{fname}: bad case JSON: {e}\n{line}"));
            let name = case.get("name").and_then(J::as_str).unwrap_or("?");
            cases.push(Case {
                key: format!("{fname}::{name}"),
                fixture: case
                    .get("fixture")
                    .and_then(J::as_str)
                    .unwrap_or("")
                    .to_string(),
                query: case
                    .get("query")
                    .and_then(J::as_str)
                    .unwrap_or("")
                    .to_string(),
                ordered: case.get("ordered").and_then(J::as_bool).unwrap_or(false),
            });
        }
    }
    cases
}

fn snapshot_path(dir: &Path) -> PathBuf {
    dir.join("snapshots.jsonl")
}

/// Run every case on the engine and write `snapshots.jsonl` (`{"key","out"}` lines,
/// key-sorted). Guarded behind `CORPUS_SNAPSHOT=1` — a deliberate, reviewable act.
fn regenerate_snapshot(dir: &Path, cases: &[Case]) {
    let mut lines: Vec<(String, String)> = Vec::new();
    for case in cases {
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_engine(&case.fixture, &case.query)
        }))
        .unwrap_or(Outcome::Err);
        let rec = serde_json::json!({ "key": case.key, "out": outcome_to_json(&out) });
        lines.push((case.key.clone(), rec.to_string()));
    }
    lines.sort();
    lines.dedup_by(|a, b| a.0 == b.0);
    let body = lines
        .into_iter()
        .map(|(_, l)| l)
        .collect::<Vec<_>>()
        .join("\n");
    let header = "// Engine regression snapshot for the GQL corpus — regenerate with\n\
                  // CORPUS_SNAPSHOT=1 (see gql_corpus.rs). One {\"key\",\"out\"} per case.\n";
    std::fs::write(snapshot_path(dir), format!("{header}{body}\n")).expect("write snapshot");
    eprintln!("wrote {} snapshot cases to snapshots.jsonl", cases.len());
}

/// Load `snapshots.jsonl` into `key → recorded Outcome`.
fn load_snapshot(dir: &Path) -> HashMap<String, Outcome> {
    let text = std::fs::read_to_string(snapshot_path(dir)).unwrap_or_default();
    let mut map = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let rec: J = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if let Some(key) = rec.get("key").and_then(J::as_str) {
            if let Some(out) = rec.get("out") {
                map.insert(key.to_string(), outcome_from_json(out));
            }
        }
    }
    map
}

#[test]
fn gql_corpus_engine_matches_snapshot() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/gql_corpus");
    if !dir.exists() {
        eprintln!("no corpus dir yet: {}", dir.display());
        return;
    }
    let cases = load_cases(&dir);

    if std::env::var("CORPUS_SNAPSHOT").is_ok() {
        regenerate_snapshot(&dir, &cases);
        return;
    }

    let snapshot = load_snapshot(&dir);
    let mut fails: Vec<String> = Vec::new();
    let mut missing = 0usize;

    for case in &cases {
        if std::env::var("CORPUS_TRACE").is_ok() {
            eprintln!("[case] {}", case.key);
        }
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_engine(&case.fixture, &case.query)
        }))
        .unwrap_or_else(|_| {
            fails.push(format!("{}: panicked for: {}", case.key, case.query));
            Outcome::Err
        });
        match snapshot.get(&case.key) {
            None => {
                missing += 1;
                fails.push(format!(
                    "{}: no snapshot (regenerate with CORPUS_SNAPSHOT=1): {}",
                    case.key, case.query
                ));
            }
            Some(want) => {
                if !eq_outcome(want, &got, case.ordered) {
                    fails.push(format!("{}: engine != snapshot for: {}", case.key, case.query));
                }
            }
        }
    }

    if let Ok(dump) = std::env::var("CORPUS_DUMP") {
        std::fs::write(&dump, fails.join("\n")).ok();
    }
    eprintln!(
        "gql corpus: {} cases, {} snapshot mismatches ({missing} missing snapshots)",
        cases.len(),
        fails.len(),
    );
    assert!(
        fails.is_empty(),
        "{} corpus mismatches vs snapshot (a regression, or a new/changed case needing \
         CORPUS_SNAPSHOT=1):\n{}",
        fails.len(),
        fails
            .iter()
            .take(40)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    );
}
