//! Differential fuzzer: generate random graphs and random GQL, run each through
//! BOTH `lenke-engine` (with its optimizer) and `lenke-core`, and assert they
//! AGREE. The Rust-native analogue of `packages/native/src/differential-fuzz.test.ts`
//! (which fuzzes the TS engine vs lenke-core) — the new engine has no TS/FFI
//! binding, so the oracle is lenke-core called directly as a dev-dependency.
//!
//! The hand-picked conformance suite (25 shapes) covers what someone thought to
//! write down; this covers the combinations nobody would — nested predicates,
//! arithmetic over absent/present-null props, grouped aggregates over odd key
//! distributions, deterministic paging.
//!
//! Staying inside the SHARED, divergence-free surface (so a mismatch is a real bug,
//! not a known difference):
//!   - Each property key holds ONE type across all nodes → no cross-type compare
//!     (engine total-orders; core throws — the documented J1 divergence).
//!   - ORDER BY only ever names projected aliases (engine scopes ORDER BY to output
//!     columns), and always ends with the unique `id` alias → a total, deterministic
//!     order, so a paged/ordered result compares position-for-position with no ties.
//!   - Aggregate results compare as multisets (row order unspecified without ORDER
//!     BY); paging is only ever added on top of a fully-ordered projection.
//!   - Numbers compare via `num_key` (exact for integers, 1e-9 otherwise) — an
//!     `avg`/`sum` can differ in the last ulp between two summation orders.
//!
//! Reproduce a failure: `FUZZ_SEED=<n> cargo test --test differential_fuzz`.
//! `FUZZ_ITERS=<n>` sets the count (default 400).

use lenke_core::gql::eval::Params as CoreParams;
use lenke_core::graph::Value as CoreVal;
use lenke_engine::value::Value as EngVal;

// The shared hard-shape generator (quantified/group/nested/shortest patterns),
// reused by the perf fuzzer too. Its `Rng` is the same xorshift64* this file used.
#[path = "support/gql_shapes.rs"]
mod gql_shapes;
use gql_shapes::{Caps, Rng, Schema};

/// This fixture's GQL vocabulary for the shared generator: label `N`, edge type `R`,
/// numeric prop `a`, unique id `id`, numeric edge prop `w`.
const SCHEMA: Schema = Schema {
    label: "N",
    etype: "R",
    num: "a",
    id: "id",
    ew: "w",
};

// ── the random graph ─────────────────────────────────────────────────────────

/// Adversarial-but-finite numeric pool (ingest maps NaN/Inf→null in both engines,
/// so stored numbers are finite; -0.0 and extremes still stress rendering/grouping).
const NUMS: &[f64] = &[
    0.0, -0.0, 1.0, -1.0, 2.0, 3.0, 42.0, -7.0, 1e15, -1e15, 0.5, 100.0,
];
/// Small string pool incl. values that stress rendering; NO query-literal use, so
/// no GQL escaping needed — strings are only projected/compared prop-vs-prop.
const STRS: &[&str] = &[
    "",
    "a",
    "b",
    "carol",
    "x\ty",
    "quote\"d",
    "🙂",
    "line\nbreak",
];

/// A node's props: `id` (unique number, the ORDER BY tiebreak), `a` (number|absent),
/// `b` (string|absent). `None` means the property is ABSENT (distinct from present
/// null); `Some(None)` means present-null; `Some(Some(v))` a present value.
struct Node {
    id: u32,
    a: Option<Option<f64>>,
    b: Option<Option<&'static str>>,
}

struct Graph {
    nodes: Vec<Node>,
    edges: Vec<(u32, u32)>,
}

fn gen_graph(rng: &mut Rng) -> Graph {
    let n = 3 + rng.below(10); // 3..=12 nodes
    let mut nodes = Vec::with_capacity(n);
    for id in 0..n as u32 {
        // Each prop: absent (1/4), present-null (1/4), or a present value (1/2).
        let a = match rng.below(4) {
            0 => None,
            1 => Some(None),
            _ => Some(Some(*rng.pick(NUMS))),
        };
        let b = match rng.below(4) {
            0 => None,
            1 => Some(None),
            _ => Some(Some(*rng.pick(STRS))),
        };
        nodes.push(Node { id, a, b });
    }
    let mut edges = Vec::new();
    let ecount = rng.below(2 * n + 1);
    for _ in 0..ecount {
        edges.push((rng.below(n) as u32, rng.below(n) as u32));
    }
    Graph { nodes, edges }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Render one node's props object (shared by both dialects; `id` always present).
fn props_json(nd: &Node) -> String {
    let mut fields = vec![format!(r#""id":{}"#, nd.id)];
    if let Some(a) = nd.a {
        fields.push(match a {
            None => r#""a":null"#.to_string(),
            Some(v) => format!(r#""a":{}"#, json_number(v)),
        });
    }
    if let Some(b) = nd.b {
        fields.push(match b {
            None => r#""b":null"#.to_string(),
            Some(s) => format!(r#""b":"{}""#, json_escape(s)),
        });
    }
    format!("{{{}}}", fields.join(","))
}

/// A finite f64 as JSON (both engines ingest -0.0 fine; we keep it distinct).
fn json_number(v: f64) -> String {
    if v == 0.0 && v.is_sign_negative() {
        "-0.0".to_string()
    } else {
        format!("{v}")
    }
}

fn engine_ndjson(g: &Graph) -> String {
    let mut s = String::new();
    for nd in &g.nodes {
        s.push_str(&format!(
            r#"{{"id":{},"labels":["N"],"props":{}}}"#,
            nd.id,
            props_json(nd)
        ));
        s.push('\n');
    }
    for (i, (f, t)) in g.edges.iter().enumerate() {
        s.push_str(&format!(
            r#"{{"from":{f},"to":{t},"type":"R","props":{{"w":{}}}}}"#,
            edge_w(i)
        ));
        s.push('\n');
    }
    s
}

/// A deterministic numeric edge property, identical in both dialects (for per-hop /
/// per-rep edge predicates on `e.w`).
fn edge_w(i: usize) -> u32 {
    (i.wrapping_mul(37) % 100) as u32
}

fn core_ndjson(g: &Graph) -> String {
    let mut s = String::new();
    for nd in &g.nodes {
        s.push_str(&format!(
            r#"{{"type":"node","id":"{}","labels":["N"],"properties":{}}}"#,
            nd.id,
            props_json(nd)
        ));
        s.push('\n');
    }
    for (i, (f, t)) in g.edges.iter().enumerate() {
        s.push_str(&format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"{f}","to":"{t}","properties":{{"w":{}}}}}"#,
            edge_w(i)
        ));
        s.push('\n');
    }
    s
}

// ── the random query ─────────────────────────────────────────────────────────

/// How to compare the two result sets for a generated query.
#[derive(Clone, Copy, PartialEq)]
enum Cmp2 {
    Ordered,  // a deterministic ORDER BY … , id → position-for-position
    Multiset, // no ORDER BY → compare as a bag
}

/// A generated query plus how to compare its results.
struct Query {
    text: String,
    cmp: Cmp2,
}

/// A numeric-typed scalar expression over the endpoint bound as `var`.
fn num_expr(rng: &mut Rng, var: &str) -> String {
    match rng.below(4) {
        0 => format!("{var}.a"),
        1 => format!("{var}.a + {}", json_number(*rng.pick(NUMS))),
        2 => format!("{var}.a * {}", json_number(*rng.pick(NUMS))),
        _ => format!("{var}.a - {var}.a"), // always 0 or null — exercises null-prop
    }
}

/// A boolean predicate over `var`, type-safe (numeric ops on `a`, equality/IS NULL
/// on `b`), so neither engine hits a cross-type comparison.
fn predicate(rng: &mut Rng, var: &str, depth: u32) -> String {
    if depth == 0 || rng.chance(1, 2) {
        return match rng.below(6) {
            0 => format!(
                "{var}.a {} {}",
                rng.pick(&["<", "<=", ">", ">=", "=", "<>"]),
                json_number(*rng.pick(NUMS))
            ),
            1 => format!("{var}.a IS NULL"),
            2 => format!("{var}.a IS NOT NULL"),
            3 => format!("{var}.b IS NULL"),
            4 => format!("{var}.b IS NOT NULL"),
            _ => format!("{var}.a = {var}.a"), // null-safe? a=a is null when a null
        };
    }
    let l = predicate(rng, var, depth - 1);
    let r = predicate(rng, var, depth - 1);
    match rng.below(3) {
        0 => format!("({l}) AND ({r})"),
        1 => format!("({l}) OR ({r})"),
        _ => format!("NOT ({l})"),
    }
}

fn gen_query(rng: &mut Rng, n_nodes: usize, hard: Option<Caps>) -> Query {
    // A HARD shape (quantified/group/nested/shortest pattern) about half the time
    // when enabled: anchored at a random real node id, reducing every path/group
    // binding to scalars, compared as a multiset. This is the byte-identity net for
    // the constructs the flat grammar below never reaches.
    if let Some(caps) = hard {
        if rng.chance(1, 2) {
            let src = rng.below(n_nodes);
            if let Some(h) = gql_shapes::gen_hard(rng, &SCHEMA, &caps, src) {
                return Query {
                    text: h.text,
                    cmp: Cmp2::Multiset,
                };
            }
        }
    }
    // Pattern: node-only, or a 1-hop that binds the endpoint `m`. The hop is spelled
    // typed OR untyped (bracketed `-[]->` / `-[e]->`) — all equivalent since there is
    // one edge type, so it exercises the untyped-relationship parse against core.
    let two_hop = rng.chance(2, 5);
    let (pattern, var) = if two_hop {
        let rel = *rng.pick(&["-[:R]->", "-[]->", "-[e]->"]);
        (
            match rel {
                "-[:R]->" => "MATCH (n:N)-[:R]->(m:N)",
                "-[]->" => "MATCH (n:N)-[]->(m:N)",
                _ => "MATCH (n:N)-[e]->(m:N)",
            },
            "m",
        )
    } else {
        ("MATCH (n:N)", "n")
    };
    let where_clause = if rng.chance(3, 5) {
        format!(" WHERE {}", predicate(rng, var, 2))
    } else {
        String::new()
    };

    // A DISTINCT projection (compared as a set) — meaningful only without the
    // unique id, so it lives in its own branch. The multi-column plain-prop forms
    // (`DISTINCT n.a, n.b` etc.) exercise the fused multi-column distinct-scan fast
    // path, including its dict-code composite key over the string column `b`.
    if rng.chance(1, 5) {
        let items = match rng.below(4) {
            0 => {
                let e = if rng.chance(1, 2) {
                    num_expr(rng, var)
                } else {
                    format!("{var}.b")
                };
                format!("{e} AS d")
            }
            1 => format!("{var}.b AS d0, {var}.a AS d1"),
            2 => format!("{var}.a AS d0, {var}.b AS d1"),
            _ => format!("{var}.a AS d0, {var}.b AS d1, {var}.id AS d2"),
        };
        return Query {
            text: format!("{pattern}{where_clause} RETURN DISTINCT {items}"),
            cmp: Cmp2::Multiset,
        };
    }

    if rng.chance(1, 2) {
        // Aggregate: whole-set, or grouped by one key. Multiset comparison.
        let agg = match rng.below(7) {
            0 => "count(*)".to_string(),
            1 => format!("count({var}.a)"),
            2 => format!("count(DISTINCT {var}.a)"),
            3 => format!("sum({var}.a)"),
            4 => format!("avg({var}.a)"),
            5 => format!("min({var}.a)"),
            _ => format!("max({var}.a)"),
        };
        if rng.chance(1, 2) {
            // grouped by a projected key (implicit grouping on the non-agg item)
            let key = if rng.chance(1, 2) {
                format!("{var}.a")
            } else {
                format!("{var}.b")
            };
            Query {
                text: format!("{pattern}{where_clause} RETURN {key} AS k, {agg} AS v"),
                cmp: Cmp2::Multiset,
            }
        } else {
            Query {
                text: format!("{pattern}{where_clause} RETURN {agg} AS v"),
                cmp: Cmp2::Multiset,
            }
        }
    } else {
        // Projection: 1-2 exprs + the unique id, ORDER BY every alias then id →
        // fully deterministic; optionally paged (safe because totally ordered).
        let mut items = vec![format!("{var}.id AS a0")];
        let mut order = vec!["a0".to_string()];
        let extra = 1 + rng.below(2);
        for k in 0..extra {
            let alias = format!("a{}", k + 1);
            let e = if rng.chance(1, 2) {
                num_expr(rng, var)
            } else {
                format!("{var}.b")
            };
            items.push(format!("{e} AS {alias}"));
            order.push(alias);
        }
        // ORDER BY all non-id aliases first, id last as the unique tiebreak.
        order.rotate_left(1);
        let mut text = format!(
            "{pattern}{where_clause} RETURN {} ORDER BY {}",
            items.join(", "),
            order.join(", ")
        );
        if rng.chance(1, 3) {
            let skip = rng.below(4);
            let lim = 1 + rng.below(5);
            text.push_str(&format!(" SKIP {skip} LIMIT {lim}"));
        }
        Query {
            text,
            cmp: Cmp2::Ordered,
        }
    }
}

// ── run + compare ────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Cell {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Other(String),
}
fn num_key(n: f64) -> String {
    if n.is_finite() && n == n.trunc() {
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

enum Outcome {
    Rows(Vec<Vec<Cell>>),
    Err,
    ParseErr,
}

fn run_engine(store: &lenke_engine::store::Store, q: &str) -> Outcome {
    let Ok(plan) = lenke_engine::gql::parse(q) else {
        return Outcome::ParseErr;
    };
    let plan = lenke_engine::opt::optimize(plan);
    match lenke_engine::exec::try_run(&plan, store) {
        Ok(rows) => Outcome::Rows(
            rows.rows
                .iter()
                .map(|r| r.iter().map(norm_eng).collect())
                .collect(),
        ),
        Err(_) => Outcome::Err,
    }
}

fn run_core(graph: &mut lenke_core::graph::Graph, q: &str) -> Outcome {
    let Ok(prepared) = lenke_core::gql::prepare(q) else {
        return Outcome::ParseErr;
    };
    match prepared.execute(graph, &CoreParams::new()) {
        Ok(rs) => Outcome::Rows(
            rs.rows()
                .map(|r| r.iter().map(norm_core).collect())
                .collect(),
        ),
        Err(_) => Outcome::Err,
    }
}

#[test]
fn engine_agrees_with_core_on_random_queries() {
    let seed: u64 = std::env::var("FUZZ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let iters: usize = std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

    // Hard-shape coverage: FUZZ_HARD=off | supported (default) | all (adds nested).
    // `supported` keeps CI green; `all` drives/verifies the nested implementation.
    let hard = match std::env::var("FUZZ_HARD").as_deref() {
        Ok("off") => None,
        Ok("all") => Some(Caps::all()),
        _ => Some(Caps::supported()),
    };

    let mut compared = 0usize; // cases where both engines produced rows
    let mut skipped = 0usize; // parse mismatches (generator over-reach)

    for it in 0..iters {
        let g = gen_graph(&mut rng);
        let q = gen_query(&mut rng, g.nodes.len(), hard);
        let store = lenke_engine::ndjson::from_ndjson(&engine_ndjson(&g)).expect("engine load");
        let mut graph = lenke_core::ndjson::decode(&core_ndjson(&g)).expect("core load");

        let e = run_engine(&store, &q.text);
        let c = run_core(&mut graph, &q.text);

        let repro = || {
            format!(
                "\nSEED={seed} iter={it}\nquery: {}\nengine ndjson:\n{}",
                q.text,
                engine_ndjson(&g)
            )
        };

        match (e, c) {
            (Outcome::Rows(mut er), Outcome::Rows(mut cr)) => {
                if q.cmp == Cmp2::Multiset {
                    er.sort();
                    cr.sort();
                }
                assert_eq!(er, cr, "result mismatch{}", repro());
                compared += 1;
            }
            (Outcome::Err, Outcome::Err) => { /* agree on rejection */ }
            // One engine errored and the other didn't: a real divergence to surface.
            (Outcome::Rows(_), Outcome::Err) => {
                panic!("engine returned rows, core ERRORED{}", repro())
            }
            (Outcome::Err, Outcome::Rows(_)) => {
                panic!("core returned rows, engine ERRORED{}", repro())
            }
            // A parse mismatch means the generator produced syntax one side lacks —
            // not a conformance bug. Count it; a flood would mean the generator drifted.
            _ => skipped += 1,
        }
    }

    // The suite is only meaningful if most cases actually compared rows.
    assert!(
        compared * 3 >= iters,
        "too few real comparisons: {compared}/{iters} compared, {skipped} parse-skipped (seed {seed})"
    );
    eprintln!(
        "differential_fuzz: seed {seed}, {iters} iters, {compared} compared, {skipped} skipped"
    );
}
