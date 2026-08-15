//! Generative perf-fuzzer: `lenke-engine` vs `lenke-core` over THOUSANDS of
//! randomly-composed GQL shapes, to find slowdowns no hand-written bench would.
//!
//! It composes building blocks — a labelled source, 0-3 (optionally var-length /
//! reversed / typed) hops, an optional comma-pattern, an optional WHERE built from
//! numeric/string/`IN`/`NOT`/`AND`/`OR` predicates, and a tail that is either an
//! aggregate (grouped or scalar, `count`/`sum`/`avg`/`min`/`max`, `DISTINCT`), a
//! projection (arithmetic, string/numeric functions, `CASE`, `coalesce`, `DISTINCT`,
//! `ORDER BY`/`SKIP`/`LIMIT`), or a `CALL` algorithm — plus `WITH` pipelines. Every
//! block contributes a feature TAG.
//!
//! Each generated query is CANONICALIZED to a template (literals → `#`/`$`) and
//! deduped, so N random queries collapse to the unique STRUCTURES worth timing.
//! Each template's representative is timed min-of-`REPS` on both engines; the ratio
//! is core_ms/engine_ms (>1 = engine faster). Invalid-on-either-engine queries are
//! skipped with the reason; a row-count MISMATCH is flagged loudly (a free
//! correctness signal). The report ranks the slowest templates and averages the
//! ratio per feature tag, so a slow building block is named.
//!
//! Native only. Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example perf_fuzz
//!   FUZZ_N=20000 BENCH_N=100000 PERF_SEED=7 cargo run --release ... --example perf_fuzz

use lenke_core::gql::eval::Params as CoreParams;
use lenke_engine::store::{Builder, Store};
use lenke_engine::value::Value;
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

// The shared hard-shape generator (quantified/group/nested/shortest patterns),
// reused from the differential fuzzer so the perf sweep exercises the same
// constructs. Its `Rng` is the same xorshift64* this example used.
#[path = "../tests/support/gql_shapes.rs"]
mod gql_shapes;
use gql_shapes::Rng;

/// This fixture's GQL vocabulary for the shared hard-shape generator. `id` is the
/// (non-unique) `age` prop — fine for perf, where the anchor just bounds the source
/// set; `ew` is the edge weight `w`.
const HARD_SCHEMA: gql_shapes::Schema = gql_shapes::Schema {
    label: "Person",
    etype: "R",
    num: "age",
    id: "age",
    ew: "w",
};

// --- fixture (identical graph to both engines) -------------------------------

const CITIES: &[&str] = &[
    "oslo",
    "bergen",
    "troms",
    "stavanger",
    "bodo",
    "tromso",
    "aalborg",
    "moss",
];

fn engine_fixture(n: u32, deg: u32) -> Store {
    let mut b = Builder::default();
    for i in 0..n {
        let props = [
            ("name", Value::Str(format!("n{i}").into())),
            ("age", Value::Num(f64::from(i % 100))),
            ("score", Value::Num(f64::from(i.wrapping_mul(7) % 1000))),
            ("city", Value::Str((*pick_city(i)).into())),
        ];
        if i % 3 == 0 {
            b.node(&["Person", "VIP"], &props);
        } else {
            b.node(&["Person"], &props);
        }
    }
    for i in 0..n {
        for d in 0..deg {
            let to = (i
                .wrapping_mul(7)
                .wrapping_add(d.wrapping_mul(13))
                .wrapping_add(1))
                % n;
            b.edge(i, to, "R");
        }
        if i % 5 == 0 {
            b.edge(i, (i.wrapping_mul(3).wrapping_add(2)) % n, "F");
        }
    }
    let mut st = b.build();
    for eid in st.all_edges() {
        st.set_edge_prop(eid, "w", Value::Num(f64::from(eid % 1000)));
    }
    // Var-length closures now enumerate fully (no silent hop cap), so an unbounded
    // shape runs to the trail limit before it errors. Lower the limit to make a
    // completing run feasible; the default matches the shipped 1M-row bound.
    let trail = env_u32("FUZZ_TRAIL_LIMIT", 1_000_000);
    st.set_limit(lenke_engine::store::ConfigId::LimitsTrail, u64::from(trail));
    // FUZZ_INDEX=1 backs the node properties with hash indexes so the planner can seed an
    // equality on them (and reverse-walk a selective endpoint) — the same indexes the core
    // fixture gets, so the comparison stays fair. Default: no indexes (the original bench).
    if env_u32("FUZZ_INDEX", 0) != 0 {
        for k in ["name", "city", "age", "score"] {
            st.create_index(k); // hash: equality / IN seeds
            st.create_range_index(k); // range: <,<=,>,>= seeds (matches core's range-capable index)
        }
    }
    st
}

fn pick_city(i: u32) -> &'static &'static str {
    &CITIES[(i % CITIES.len() as u32) as usize]
}

fn core_fixture(n: u32, deg: u32) -> lenke_core::graph::Graph {
    let mut s = String::new();
    for i in 0..n {
        let labels = if i % 3 == 0 {
            r#"["Person","VIP"]"#
        } else {
            r#"["Person"]"#
        };
        s.push_str(&format!(
            r#"{{"type":"node","id":"{i}","labels":{labels},"properties":{{"name":"n{i}","age":{},"score":{},"city":"{}"}}}}"#,
            i % 100,
            i.wrapping_mul(7) % 1000,
            pick_city(i)
        ));
        s.push('\n');
    }
    let mut e = 0u64;
    let push_edge = |s: &mut String, from: u32, to: u32, ty: &str, e: &mut u64| {
        s.push_str(&format!(
            r#"{{"type":"edge","id":"e{e}","labels":["{ty}"],"from":"{from}","to":"{to}","properties":{{"w":{}}}}}"#,
            *e % 1000
        ));
        s.push('\n');
        *e += 1;
    };
    for i in 0..n {
        for d in 0..deg {
            let to = (i
                .wrapping_mul(7)
                .wrapping_add(d.wrapping_mul(13))
                .wrapping_add(1))
                % n;
            push_edge(&mut s, i, to, "R", &mut e);
        }
        if i % 5 == 0 {
            push_edge(
                &mut s,
                i,
                (i.wrapping_mul(3).wrapping_add(2)) % n,
                "F",
                &mut e,
            );
        }
    }
    let mut g = lenke_core::ndjson::decode(&s).expect("core load");
    if env_u32("FUZZ_INDEX", 0) != 0 {
        for k in ["name", "city", "age", "score"] {
            g.create_vertex_index(k);
        }
    }
    g
}

// --- query generator ---------------------------------------------------------

const NUM_PROPS: &[&str] = &["age", "score"];
const STR_PROPS: &[&str] = &["name", "city"];
const LABELS: &[&str] = &["Person", "VIP"];
const ETYPES: &[&str] = &["R", "F"];

struct Gen<'a> {
    rng: &'a mut Rng,
    tags: Vec<&'static str>,
}

impl Gen<'_> {
    fn tag(&mut self, t: &'static str) {
        if !self.tags.contains(&t) {
            self.tags.push(t);
        }
    }

    /// A numeric literal in a realistic range for the props.
    fn num_lit(&mut self) -> String {
        format!("{}", self.rng.below(120))
    }
    fn str_lit(&mut self) -> String {
        if self.rng.chance(1, 2) {
            format!("'{}'", self.rng.pick(CITIES))
        } else {
            format!("'n{}'", self.rng.below(1000))
        }
    }

    /// A predicate over variable `v`'s properties (three-valued-safe, never
    /// cross-type: numeric props get numeric ops, string props get string ops).
    fn predicate(&mut self, v: &str, depth: u32) -> String {
        if depth > 0 && self.rng.chance(2, 5) {
            let op = if self.rng.chance(1, 2) { "AND" } else { "OR" };
            self.tag(if op == "AND" { "pred-and" } else { "pred-or" });
            let a = self.predicate(v, depth - 1);
            let b = self.predicate(v, depth - 1);
            return format!("({a} {op} {b})");
        }
        if self.rng.chance(1, 6) {
            self.tag("pred-not");
            let inner = self.predicate(v, 0);
            return format!("NOT {inner}");
        }
        match self.rng.below(10) {
            0..=3 => {
                // numeric compare
                self.tag("pred-num");
                let p = self.rng.pick(NUM_PROPS);
                let op = *self.rng.pick(&["<", "<=", ">", ">=", "=", "<>"]);
                format!("{v}.{p} {op} {}", self.num_lit())
            }
            4 => {
                // numeric IN
                self.tag("pred-in");
                let p = self.rng.pick(NUM_PROPS);
                format!(
                    "{v}.{p} IN [{}, {}, {}]",
                    self.num_lit(),
                    self.num_lit(),
                    self.num_lit()
                )
            }
            5..=6 => {
                // string compare
                self.tag("pred-str");
                let p = self.rng.pick(STR_PROPS);
                let op = *self.rng.pick(&["=", "<>", "<", ">"]);
                format!("{v}.{p} {op} {}", self.str_lit())
            }
            _ => {
                // string search predicate
                self.tag("pred-strsearch");
                let p = self.rng.pick(STR_PROPS);
                let op = *self.rng.pick(&["STARTS WITH", "ENDS WITH", "CONTAINS"]);
                format!("{v}.{p} {op} {}", self.str_lit())
            }
        }
    }

    /// A numeric scalar expression over `v`.
    fn num_expr(&mut self, v: &str) -> String {
        match self.rng.below(6) {
            0 => {
                let p = self.rng.pick(NUM_PROPS);
                format!("{v}.{p}")
            }
            1 => {
                self.tag("expr-arith");
                let p = self.rng.pick(NUM_PROPS);
                let op = *self.rng.pick(&["+", "-", "*"]);
                format!("({v}.{p} {op} {})", self.num_lit())
            }
            2 => {
                self.tag("expr-numfn");
                let f = *self.rng.pick(&["abs", "floor", "ceil", "round", "sign"]);
                let p = self.rng.pick(NUM_PROPS);
                format!("{f}({v}.{p} - {})", self.num_lit())
            }
            3 => {
                self.tag("expr-coalesce");
                let p = self.rng.pick(NUM_PROPS);
                format!("coalesce({v}.{p}, 0)")
            }
            4 => {
                self.tag("expr-case");
                let p = self.rng.pick(NUM_PROPS);
                format!("CASE WHEN {v}.{p} > {} THEN 1 ELSE 0 END", self.num_lit())
            }
            _ => {
                let p = self.rng.pick(NUM_PROPS);
                format!("{v}.{p}")
            }
        }
    }

    /// A string scalar expression over `v`.
    fn str_expr(&mut self, v: &str) -> String {
        match self.rng.below(4) {
            0 => {
                let p = self.rng.pick(STR_PROPS);
                format!("{v}.{p}")
            }
            1 => {
                self.tag("expr-strfn");
                let f = *self.rng.pick(&["upper", "lower", "trim"]);
                let p = self.rng.pick(STR_PROPS);
                format!("{f}({v}.{p})")
            }
            2 => {
                self.tag("expr-substr");
                let p = self.rng.pick(STR_PROPS);
                format!("substring({v}.{p}, 0, {})", 1 + self.rng.below(3))
            }
            _ => {
                let p = self.rng.pick(STR_PROPS);
                format!("{v}.{p}")
            }
        }
    }

    /// The MATCH pattern: a labelled source, then 0-4 hops (weighted shallow), any
    /// hop optionally variable-length ({1,2}..{2,4}), reversed, or EDGE-BOUND (which
    /// enables per-hop edge predicates — the AML "structuring" shape). Returns the
    /// pattern text, the last bound node variable, the hop count, and the bound edge
    /// variables. Deep/var-length chains explode; the timeout guard is the backstop,
    /// and the query builder biases deep chains toward a LIMIT / reducer.
    fn pattern(&mut self) -> (String, String, usize, Vec<String>) {
        let label = self.rng.pick(LABELS);
        let mut pat = format!("(a:{label})");
        let mut last = "a".to_string();
        let mut edges: Vec<String> = Vec::new();
        // Weighted toward shallow so the fuzz isn't all explosions, but reaches 4.
        let hops = match self.rng.below(12) {
            0..=4 => 1,
            5..=7 => 2,
            8..=10 => 3,
            _ => 4,
        };
        let vars = ["b", "c", "d", "e"];
        for (h, var) in vars.iter().enumerate().take(hops) {
            let ety = self.rng.pick(ETYPES);
            if self.rng.chance(1, 6) {
                // variable-length hop, up to {2,4}
                self.tag("varlen");
                let lo = 1 + self.rng.below(2);
                let hi = lo + 1 + self.rng.below(2);
                pat.push_str(&format!("-[:{ety}]->{{{lo},{hi}}}({var})"));
            } else if self.rng.chance(1, 4) {
                self.tag("hop-in");
                pat.push_str(&format!("<-[:{ety}]-({var})"));
            } else if self.rng.chance(1, 3) {
                // edge-bound hop — enables an edge predicate on `e{h}`
                self.tag("hop-edge");
                let ev = format!("e{h}");
                pat.push_str(&format!("-[{ev}:{ety}]->({var})"));
                edges.push(ev);
            } else {
                self.tag("hop-out");
                pat.push_str(&format!("-[:{ety}]->({var})"));
            }
            last = (*var).to_string();
        }
        (pat, last, hops, edges)
    }

    /// A full query. `algo_ok` allows a CALL tail.
    fn query(&mut self) -> String {
        // A HARD shape (quantified / subpath group / nested / shortest) from the
        // shared generator — the same constructs the differential fuzzer verifies,
        // timed here to price them. Anchored on a random `age` bucket.
        if self.rng.chance(1, 6) {
            let src = self.rng.below(120);
            if let Some(h) =
                gql_shapes::gen_hard(self.rng, &HARD_SCHEMA, &gql_shapes::Caps::all(), src)
            {
                for t in h.tags {
                    self.tag(t);
                }
                return h.text;
            }
        }
        // A CALL-algorithm query (its own shape).
        if self.rng.chance(1, 12) {
            self.tag("call-algo");
            let (proc, yield_col) = *self.rng.pick(&[
                ("degree", "degree"),
                ("connected_components", "componentId"),
                ("pagerank", "score"),
                ("label_propagation", "label"),
                ("on_cycle", "onCycle"),
                ("strongly_connected_components", "componentId"),
            ]);
            let agg = if self.rng.chance(1, 2) {
                format!("count(DISTINCT {yield_col})")
            } else {
                "count(*)".to_string()
            };
            return format!("CALL {proc}() YIELD {yield_col} RETURN {agg} AS c");
        }

        let (pat, tgt, hops, edges) = self.pattern();
        let mut q = format!("MATCH {pat}");

        // Optional comma-pattern joined on the target (a linear extension). Only at
        // low hop depth, so the joined expansion stays bounded.
        if hops <= 1 && self.rng.chance(1, 8) {
            self.tag("comma-join");
            let ety = self.rng.pick(ETYPES);
            q.push_str(&format!(", ({tgt})-[:{ety}]->(z)"));
        }

        // WHERE: node predicate on the target and/or EDGE predicates on bound edges
        // (a single `e.w <op> k`, or the AML per-hop `e1.w > e2.w` structuring chain).
        let mut wheres: Vec<String> = Vec::new();
        if self.rng.chance(3, 5) {
            self.tag("where");
            wheres.push(self.predicate(&tgt, 2));
        }
        if edges.len() >= 2 && self.rng.chance(1, 2) {
            self.tag("edge-pred-chain");
            for pair in edges.windows(2) {
                wheres.push(format!("{}.w > {}.w", pair[0], pair[1]));
            }
        } else if let Some(ev) = edges.first() {
            if self.rng.chance(1, 2) {
                self.tag("edge-pred");
                let op = *self.rng.pick(&["<", ">", "=", "<>"]);
                wheres.push(format!("{ev}.w {op} {}", self.rng.below(1000)));
            }
        }
        if !wheres.is_empty() {
            q.push_str(&format!(" WHERE {}", wheres.join(" AND ")));
        }

        // A deep chain (>=3 hops) is biased toward a bounding LIMIT — the realistic
        // AML "layering" shape (`… LIMIT 5000`), and where a streaming top-K / cap
        // would pay off.
        let deep = hops >= 3;

        // Optional WITH pipeline stage (project a numeric alias then filter).
        if self.rng.chance(1, 10) {
            self.tag("with");
            let e = self.num_expr(&tgt);
            q.push_str(&format!(" WITH {e} AS wv WHERE wv >= {}", self.num_lit()));
            return format!("{q} RETURN count(*) AS c");
        }

        // Tail: aggregate or projection (deep chains lean to a LIMIT'd projection).
        if !deep && self.rng.chance(1, 2) {
            self.aggregate_tail(&tgt).map_or_else(
                || format!("{q} RETURN count(*) AS c"),
                |t| format!("{q} {t}"),
            )
        } else if deep && self.rng.chance(3, 5) {
            // Bounded layering: project the endpoints with a LIMIT.
            self.tag("proj");
            self.tag("limit");
            format!(
                "{q} RETURN {tgt}.name AS s, {tgt}.city AS t LIMIT {}",
                100 + self.rng.below(5000)
            )
        } else {
            format!("{q} {}", self.projection_tail(&tgt))
        }
    }

    fn aggregate_tail(&mut self, v: &str) -> Option<String> {
        self.tag("agg");
        let agg = match self.rng.below(8) {
            0 => "count(*)".to_string(),
            1 => {
                self.tag("agg-distinct");
                let p = self.rng.pick(NUM_PROPS);
                format!("count(DISTINCT {v}.{p})")
            }
            2 => format!("sum({})", self.num_expr(v)),
            3 => format!("avg({v}.{})", self.rng.pick(NUM_PROPS)),
            4 => format!("min({v}.{})", self.rng.pick(NUM_PROPS)),
            5 => format!("max({v}.{})", self.rng.pick(NUM_PROPS)),
            6 => format!("min({v}.{})", self.rng.pick(STR_PROPS)),
            _ => format!("max({v}.{})", self.rng.pick(STR_PROPS)),
        };
        if self.rng.chance(1, 2) {
            // grouped by a key
            self.tag("agg-grouped");
            let key = if self.rng.chance(1, 2) {
                format!("{v}.{}", self.rng.pick(NUM_PROPS))
            } else {
                self.str_expr(v)
            };
            Some(format!("RETURN {key} AS k, {agg} AS val"))
        } else {
            Some(format!("RETURN {agg} AS val"))
        }
    }

    fn projection_tail(&mut self, v: &str) -> String {
        self.tag("proj");
        let distinct = self.rng.chance(1, 4);
        if distinct {
            self.tag("distinct");
        }
        let n_items = 1 + self.rng.below(2);
        let mut items = Vec::new();
        let mut aliases = Vec::new();
        for k in 0..n_items {
            let alias = format!("x{k}");
            let e = match self.rng.below(3) {
                0 => self.num_expr(v),
                1 => self.str_expr(v),
                _ => format!("{v}.{}", self.rng.pick(STR_PROPS)),
            };
            items.push(format!("{e} AS {alias}"));
            aliases.push(alias);
        }
        let mut tail = format!(
            "RETURN {}{}",
            if distinct { "DISTINCT " } else { "" },
            items.join(", ")
        );
        // ORDER BY output aliases (never an expression alongside DISTINCT), optional
        // paging. Ordering by aliases is valid on both engines.
        if !distinct && self.rng.chance(1, 2) {
            self.tag("order");
            let key = self.rng.pick(&aliases).clone();
            let dir = if self.rng.chance(1, 2) { " DESC" } else { "" };
            tail.push_str(&format!(" ORDER BY {key}{dir}"));
            if self.rng.chance(1, 2) {
                self.tag("limit");
                tail.push_str(&format!(" LIMIT {}", 1 + self.rng.below(100)));
            }
        } else if self.rng.chance(1, 4) {
            self.tag("limit");
            tail.push_str(&format!(" LIMIT {}", 1 + self.rng.below(100)));
        }
        tail
    }
}

/// Canonicalize a query to a template: numeric literals → `#`, quoted strings →
/// `$`, so queries differing only in constants collapse to one structure.
fn template(q: &str) -> String {
    let mut out = String::with_capacity(q.len());
    let bytes = q.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '\'' {
            // a quoted string literal → $
            out.push('$');
            i += 1;
            while i < bytes.len() && bytes[i] != b'\'' {
                i += 1;
            }
            i += 1; // closing quote
        } else if c.is_ascii_digit() {
            // a run of digits (and a decimal point) → #
            out.push('#');
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

// --- harness -----------------------------------------------------------------

/// Time the engine on a background thread with a wall-clock BUDGET. A query that
/// blows the budget (a pathological materialization) is reported as a finding
/// (`TIMEOUT`) rather than hanging the whole fuzz — the worker thread is abandoned
/// (it finishes on its own; `Store` is shared read-only via `Arc`).
fn time_engine_guarded(
    store: &Arc<Store>,
    q: &str,
    reps: usize,
    budget: Duration,
) -> Result<(f64, usize), String> {
    let parsed = lenke_engine::gql::parse(q).map_err(|e| format!("parse: {e}"))?;
    let plan = Arc::new(lenke_engine::opt::optimize_indexed(parsed, store.as_ref()));
    let (tx, rx) = mpsc::channel();
    let (s, p) = (Arc::clone(store), Arc::clone(&plan));
    std::thread::spawn(move || {
        let mut best = f64::MAX;
        let mut rows = 0;
        for _ in 0..reps {
            let t = Instant::now();
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                lenke_engine::exec::try_run(&p, &s)
            }));
            match out {
                // A runtime fault (E_RESOURCE on an unbounded closure, a failed CAST) is a
                // categorized outcome, not a panic — surface its message so the report can
                // separate "engine correctly refused" from "engine crashed".
                Ok(Ok(o)) => {
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    rows = o.rows.len();
                }
                Ok(Err(e)) => {
                    let _ = tx.send(Err(e));
                    return;
                }
                Err(_) => {
                    let _ = tx.send(Err("engine panic".to_string()));
                    return;
                }
            }
        }
        let _ = tx.send(Ok((best, rows)));
    });
    match rx.recv_timeout(budget) {
        Ok(r) => r,
        Err(_) => Err("TIMEOUT".to_string()),
    }
}

fn time_core(
    graph: &mut lenke_core::graph::Graph,
    q: &str,
    reps: usize,
) -> Result<(f64, usize), String> {
    let prepared = lenke_core::gql::prepare(q).map_err(|e| format!("parse: {e}"))?;
    let params = CoreParams::new();
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let t = Instant::now();
        let rs = prepared
            .execute(graph, &params)
            .map_err(|e| format!("exec: {e}"))?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        rows = rs.nrows;
    }
    Ok((best, rows))
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Timed {
    ratio: f64,
    e_ms: f64,
    c_ms: f64,
    rows: usize,
    tags: Vec<&'static str>,
    query: String,
}

fn main() {
    const REPS: usize = 3;
    let n = env_u32("BENCH_N", 100_000);
    let deg = env_u32("BENCH_DEG", 4);
    let gen_n = env_u32("FUZZ_N", 20_000) as usize;
    let seed = std::env::var("PERF_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xC0FFEE);
    let max_time = env_u32("FUZZ_MAX_TEMPLATES", 4000) as usize;

    let budget = Duration::from_millis(u64::from(env_u32("FUZZ_BUDGET_MS", 3000)));
    eprintln!("fixture: {n} nodes, deg {deg}; generating {gen_n} queries (seed {seed})");
    let estore = Arc::new(engine_fixture(n, deg));
    let mut cgraph = core_fixture(n, deg);

    // Targeted mode: time a fixed list of queries from a file (one per line; a leading
    // `<...>  ` prefix up to the last "  " is stripped so the loss-dump/cluster lines work
    // as-is) on the SAME fixture and harness. For attributing a specific optimization.
    if let Ok(path) = std::env::var("FUZZ_QUERIES") {
        let text = std::fs::read_to_string(&path).expect("read FUZZ_QUERIES file");
        println!("targeted: timing queries from {path}\n  {:>7} {:>9} {:>9}  query", "ratio", "eng_ms", "core_ms");
        for line in text.lines() {
            let q = line.rsplit("  ").next().unwrap_or(line).trim();
            if q.is_empty() || !q.to_ascii_uppercase().contains("MATCH") {
                continue;
            }
            let e = time_engine_guarded(&estore, q, REPS, budget);
            let c = time_core(&mut cgraph, q, REPS);
            match (e, c) {
                (Ok((e_ms, er)), Ok((c_ms, cr))) => {
                    let flag = if er != cr { "  ROWS DIFFER!" } else { "" };
                    println!("  {:>6.2}x {e_ms:>9.3} {c_ms:>9.3}  {q}{flag}", c_ms / e_ms);
                }
                (e, c) => println!("  (skipped: eng={e:?} core={c:?})  {q}"),
            }
        }
        return;
    }

    // 1) Generate and dedup to unique templates (keep the first concrete instance).
    let mut rng = Rng(seed);
    let mut templates: HashMap<String, (String, Vec<&'static str>)> = HashMap::new();
    for _ in 0..gen_n {
        let mut g = Gen {
            rng: &mut rng,
            tags: Vec::new(),
        };
        let q = g.query();
        let tags = g.tags;
        templates.entry(template(&q)).or_insert((q, tags));
    }
    eprintln!("{} unique templates", templates.len());

    // 2) Time each template. The ENGINE runs first under a wall-clock budget; only
    //    if it finishes do we run core (inline). Invalid queries are skipped with
    //    the reason; a TIMEOUT or row-count MISMATCH is flagged.
    let mut results: Vec<Timed> = Vec::new();
    let mut skips: HashMap<String, usize> = HashMap::new();
    let mut mismatches: Vec<(String, usize, usize)> = Vec::new();
    let mut timeouts: Vec<String> = Vec::new();
    // Recursive path shapes (var-length, subpath-group quantifiers, shortest) run on a
    // 1 GiB scoped stack and can enumerate deeply; with FUZZ_SKIP_VARLEN set they are
    // skipped so a run stays bounded in memory. Their behavior is characterized
    // separately (the engine refuses oversized closures with E_RESOURCE); this leaves
    // the analytical-shape regression/mismatch signal, which is what the flag is for.
    let skip_recursive = env_u32("FUZZ_SKIP_VARLEN", 0) != 0;
    let is_recursive = |q: &str| {
        const MARKERS: [&str; 8] = ["->{", "<-{", "SHORTEST", "){", ")*", ")+", "->*", "->+"];
        MARKERS.iter().any(|m| q.contains(m))
    };
    // Deterministic order so the timed subset is identical across runs and across commits
    // (HashMap iteration is randomized) — required to diff a baseline vs a change on one seed.
    let mut ordered: Vec<&(String, Vec<&'static str>)> = templates.values().collect();
    ordered.sort_by(|a, b| a.0.cmp(&b.0));
    let mut timed = 0usize;
    let mut processed = 0usize;
    let mut recursive_skipped = 0usize;
    let run_start = Instant::now();
    for (q, tags) in ordered {
        if timed >= max_time {
            break;
        }
        if skip_recursive && is_recursive(q) {
            recursive_skipped += 1;
            continue;
        }
        processed += 1;
        if processed.is_multiple_of(250) {
            eprintln!(
                "  progress: {processed} processed, {timed} timed, {} errored, {} timeout ({:.0}s)",
                skips.values().sum::<usize>(),
                timeouts.len(),
                run_start.elapsed().as_secs_f64()
            );
        }
        let e = match time_engine_guarded(&estore, q, REPS, budget) {
            Ok(v) => v,
            Err(why) if why == "TIMEOUT" => {
                timeouts.push(q.clone());
                continue;
            }
            Err(why) => {
                dump_skip("engine", &first_line(&why), q);
                *skips
                    .entry(format!("engine {}", first_line(&why)))
                    .or_insert(0) += 1;
                continue;
            }
        };
        let c = match time_core(&mut cgraph, q, REPS) {
            Ok(v) => v,
            Err(why) => {
                dump_skip("core", &first_line(&why), q);
                *skips
                    .entry(format!("core {}", first_line(&why)))
                    .or_insert(0) += 1;
                continue;
            }
        };
        timed += 1;
        let (e_ms, e_rows) = e;
        let (c_ms, c_rows) = c;
        if e_rows != c_rows {
            mismatches.push((q.clone(), e_rows, c_rows));
        }
        results.push(Timed {
            ratio: c_ms / e_ms,
            e_ms,
            c_ms,
            rows: e_rows,
            tags: tags.clone(),
            query: q.clone(),
        });
    }

    // 3) Report.
    let total = results.len();
    let wins = results.iter().filter(|r| r.ratio >= 1.0).count();
    println!(
        "\n=== {total} templates timed | {wins} win / {} lose | {} skipped kinds | {recursive_skipped} recursive skipped ===",
        total - wins,
        skips.values().sum::<usize>()
    );

    if !mismatches.is_empty() {
        println!("\n!!! ROW-COUNT MISMATCHES (correctness signal) !!!");
        for (q, e, c) in mismatches.iter().take(20) {
            println!("  engine={e} core={c}  {q}");
        }
    }
    if !timeouts.is_empty() {
        println!(
            "\n!!! {} ENGINE TIMEOUTS (> {} ms — pathological shapes) !!!",
            timeouts.len(),
            budget.as_millis()
        );
        for q in timeouts.iter().take(15) {
            println!("  {q}");
        }
    }
    if !skips.is_empty() {
        println!("\nskip kinds:");
        let mut sk: Vec<(&String, &usize)> = skips.iter().collect();
        sk.sort_by(|a, b| b.1.cmp(a.1));
        for (why, count) in sk.iter().take(12) {
            println!("  {count:>5}  {why}");
        }
    }

    // Per-feature average ratio (which building block is slow), min counts.
    let mut per_tag: HashMap<&'static str, (f64, usize, usize)> = HashMap::new();
    for r in &results {
        for t in &r.tags {
            let e = per_tag.entry(t).or_insert((0.0, 0, 0));
            e.0 += r.ratio;
            e.1 += 1;
            if r.ratio < 1.0 {
                e.2 += 1;
            }
        }
    }
    let mut tag_rows: Vec<(&&str, f64, usize, usize)> = per_tag
        .iter()
        .map(|(t, (sum, n, lose))| (t, sum / *n as f64, *n, *lose))
        .collect();
    tag_rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("\nper-feature mean ratio (worst first):");
    println!("  {:>7}  {:>5}  {:>5}  feature", "mean", "n", "lose");
    for (t, mean, cnt, lose) in &tag_rows {
        println!("  {mean:>7.2}  {cnt:>5}  {lose:>5}  {t}");
    }

    // Slowest templates (only where the absolute time is meaningful — skip noise
    // below 0.2ms on both sides).
    results.sort_by(|a, b| a.ratio.partial_cmp(&b.ratio).unwrap());
    println!("\nslowest templates (ratio, engine_ms, core_ms, rows, tags, query):");
    let mut shown = 0;
    for r in &results {
        if r.e_ms < 0.2 && r.c_ms < 0.2 {
            continue; // noise floor
        }
        if r.ratio >= 0.9 {
            break;
        }
        println!(
            "  {:>5.2}x  {:>8.3} {:>8.3} {:>8}  [{}]  {}",
            r.ratio,
            r.e_ms,
            r.c_ms,
            r.rows,
            r.tags.join(","),
            r.query
        );
        shown += 1;
        if shown >= 45 {
            break;
        }
    }
    if shown == 0 {
        println!("  (none below 0.9x above the noise floor)");
    }

    // Dump EVERY losing template (ratio < 1.0) as TSV for offline clustering — the
    // printed list is truncated to the slowest few. `results` is already ratio-sorted.
    if let Ok(path) = std::env::var("FUZZ_LOSS_DUMP") {
        let mut out = String::from("ratio\te_ms\tc_ms\trows\ttags\tquery\n");
        for r in results.iter().filter(|r| r.ratio < 1.0) {
            out.push_str(&format!(
                "{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\n",
                r.ratio,
                r.e_ms,
                r.c_ms,
                r.rows,
                r.tags.join(","),
                r.query
            ));
        }
        match std::fs::write(&path, out) {
            Ok(()) => eprintln!("wrote losing templates to {path}"),
            Err(e) => eprintln!("could not write {path}: {e}"),
        }
    }

    // Every timed template, keyed by query, for a deterministic baseline-vs-change diff.
    if let Ok(path) = std::env::var("FUZZ_DUMP_ALL") {
        let mut rows: Vec<&Timed> = results.iter().collect();
        rows.sort_by(|a, b| a.query.cmp(&b.query));
        let mut out = String::new();
        for r in rows {
            out.push_str(&format!("{:.3}\t{}\t{}\t{}\n", r.ratio, r.e_ms, r.rows, r.query));
        }
        match std::fs::write(&path, out) {
            Ok(()) => eprintln!("wrote {} timed templates to {path}", results.len()),
            Err(e) => eprintln!("could not write {path}: {e}"),
        }
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(60).collect()
}

/// Append a skipped query (which engine, why, the query) to `FUZZ_SKIP_DUMP` if set —
/// so the actual heavy/rejected shapes can be sampled, not just tallied by reason.
fn dump_skip(side: &str, why: &str, q: &str) {
    if let Ok(path) = std::env::var("FUZZ_SKIP_DUMP") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{side}\t{why}\t{q}");
        }
    }
}
