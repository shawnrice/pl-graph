//! Correctness census: probe many GQL feature families against lenke-core and
//! COLLECT every divergence (bucketed, with a minimal repro) instead of stopping
//! at the first — the input to a correctness to-do list.
//!
//! Each probe generates random queries stressing ONE feature; every case is run
//! through both engines and classified:
//!   OK         — both agree (rows equal, or both error)
//!   SEMANTIC   — both returned rows but they DIFFER (a value/semantics bug)
//!   MISSING    — core returned rows, the engine ERRORED (unsupported/broken)
//!   CORE_ERR   — the engine returned rows, core ERRORED (often a known divergence,
//!                e.g. cross-type ordering; still surfaced)
//! One repro (query + which side) is kept per (probe, class). Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example fuzz_report
//!   FUZZ_ITERS=4000 cargo run --release ... --example fuzz_report

use lenke_core::gql::eval::Params as CoreParams;
use lenke_core::graph::Value as CoreVal;
use lenke_engine::value::Value as EngVal;
use std::collections::BTreeMap;

// ── PRNG ─────────────────────────────────────────────────────────────────────
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
    fn chance(&mut self, num: u32, den: u32) -> bool {
        (self.next() % u64::from(den)) < u64::from(num)
    }
}

const NUMS: &[&str] = &["0", "-0.0", "1", "-1", "2", "3", "42", "-7", "0.5", "100"];
// Safe GQL literals (no escaping). Includes an astral emoji and a mixed
// ASCII+emoji string so the UTF-16-vs-chars string model is actually exercised —
// `size`/`substring`/`left`/`split` diverge there iff the engine miscounts units.
const STRS: &[&str] = &["a", "b", "carol", "", "😀", "x😀y", "aXbXc"];

// ── random graph (props: id unique, a number|null|absent, b string|null|absent,
//    edges carry w number|null|absent) ────────────────────────────────────────
struct Graph {
    node_lines: Vec<(u32, String)>,
    edges: Vec<(u32, u32, String)>, // (from, to, props-json)
}
fn gen_graph(rng: &mut Rng) -> Graph {
    let n = 3 + rng.below(8);
    let mut node_lines = Vec::new();
    for id in 0..n as u32 {
        let mut f = vec![format!(r#""id":{id}"#)];
        match rng.below(4) {
            0 => {}
            1 => f.push(r#""a":null"#.into()),
            _ => f.push(format!(r#""a":{}"#, rng.pick(NUMS))),
        }
        match rng.below(4) {
            0 => {}
            1 => f.push(r#""b":null"#.into()),
            _ => f.push(format!(r#""b":"{}""#, rng.pick(STRS))),
        }
        node_lines.push((id, format!("{{{}}}", f.join(","))));
    }
    let mut edges = Vec::new();
    for _ in 0..rng.below(2 * n + 1) {
        let props = match rng.below(3) {
            0 => "{}".to_string(),
            1 => r#"{"w":null}"#.to_string(),
            _ => format!(r#"{{"w":{}}}"#, rng.pick(NUMS)),
        };
        edges.push((rng.below(n) as u32, rng.below(n) as u32, props));
    }
    Graph { node_lines, edges }
}
fn engine_ndjson(g: &Graph) -> String {
    let mut s = String::new();
    for (id, p) in &g.node_lines {
        s.push_str(&format!(r#"{{"id":{id},"labels":["N"],"props":{p}}}"#));
        s.push('\n');
    }
    for (f, t, p) in &g.edges {
        s.push_str(&format!(
            r#"{{"from":{f},"to":{t},"type":"R","props":{p}}}"#
        ));
        s.push('\n');
    }
    s
}
fn core_ndjson(g: &Graph) -> String {
    let mut s = String::new();
    for (id, p) in &g.node_lines {
        s.push_str(&format!(
            r#"{{"type":"node","id":"{id}","labels":["N"],"properties":{p}}}"#
        ));
        s.push('\n');
    }
    for (i, (f, t, p)) in g.edges.iter().enumerate() {
        s.push_str(&format!(
            r#"{{"type":"edge","id":"e{i}","labels":["R"],"from":"{f}","to":"{t}","properties":{p}}}"#
        ));
        s.push('\n');
    }
    s
}

// ── run + compare ────────────────────────────────────────────────────────────
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Cell {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Other(String),
}
fn numk(n: f64) -> String {
    if n.is_finite() && n == n.trunc() {
        format!("i{}", n as i64)
    } else if n.is_nan() {
        "nan".into()
    } else {
        format!("f{n:.6}")
    }
}
// Type-AGNOSTIC canonical serialization of compound values (List/Record/Map), so an
// engine Map and a core Map with the same contents compare EQUAL — the two Value
// enums have different Debug type tags, which previously made every element map
// (RETURN n) look divergent. Primitive leaves reuse `numk`. Node maps render the
// same on both sides now, so this is what proves it.
fn canon_e(v: &EngVal) -> String {
    match v {
        EngVal::Null => "z".into(),
        EngVal::Bool(b) => format!("B{b}"),
        EngVal::Num(n) => format!("N{}", numk(*n)),
        EngVal::Str(s) => format!("S{s}"),
        EngVal::List(x) => format!("[{}]", x.iter().map(canon_e).collect::<Vec<_>>().join(",")),
        EngVal::Record(f) => format!(
            "R{{{}}}",
            f.iter()
                .map(|(k, v)| format!("{k}={}", canon_e(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        EngVal::Map(m) => format!(
            "M{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}={}", canon_e(k), canon_e(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        o => format!("?{o:?}"),
    }
}
fn canon_c(v: &CoreVal) -> String {
    match v {
        CoreVal::Null => "z".into(),
        CoreVal::Bool(b) => format!("B{b}"),
        CoreVal::Num(n) => format!("N{}", numk(*n)),
        CoreVal::Str(s) => format!("S{s}"),
        CoreVal::List(x) => format!("[{}]", x.iter().map(canon_c).collect::<Vec<_>>().join(",")),
        CoreVal::Map(m) => format!(
            "M{{{}}}",
            m.iter()
                .map(|(k, v)| format!("S{k}={}", canon_c(v)))
                .collect::<Vec<_>>()
                .join(",")
        ),
        o => format!("?{o:?}"),
    }
}
fn ne(v: &EngVal) -> Cell {
    match v {
        EngVal::Null => Cell::Null,
        EngVal::Bool(b) => Cell::Bool(*b),
        EngVal::Num(n) => Cell::Num(numk(*n)),
        EngVal::Str(s) => Cell::Str(s.to_string()),
        EngVal::List(_) | EngVal::Record(_) | EngVal::Map(_) => Cell::Other(canon_e(v)),
        o => Cell::Other(format!("{o:?}")),
    }
}
fn nc(v: &CoreVal) -> Cell {
    match v {
        CoreVal::Null => Cell::Null,
        CoreVal::Bool(b) => Cell::Bool(*b),
        CoreVal::Num(n) => Cell::Num(numk(*n)),
        CoreVal::Str(s) => Cell::Str(s.to_string()),
        CoreVal::List(_) | CoreVal::Map(_) => Cell::Other(canon_c(v)),
        o => Cell::Other(format!("{o:?}")),
    }
}
enum Out {
    Rows(Vec<Vec<Cell>>),
    Err,
    Parse,
    Panic,
}
fn eng(store: &lenke_engine::store::Store, q: &str) -> Out {
    let Ok(p) = lenke_engine::gql::parse(q) else {
        return Out::Parse;
    };
    let p = lenke_engine::opt::optimize(p);
    // A PANIC is the worst correctness bug (the engine must only ever return
    // Err) — catch it so the census reports the offending query instead of
    // aborting the whole run.
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        lenke_engine::exec::try_run(&p, store)
    }));
    match caught {
        Ok(Ok(r)) => Out::Rows(
            r.rows
                .iter()
                .map(|row| row.iter().map(ne).collect())
                .collect(),
        ),
        Ok(Err(_)) => Out::Err,
        Err(_) => Out::Panic,
    }
}
fn core(g: &mut lenke_core::graph::Graph, q: &str) -> Out {
    let Ok(p) = lenke_core::gql::prepare(q) else {
        return Out::Parse;
    };
    match p.execute(g, &CoreParams::new()) {
        Ok(rs) => Out::Rows(rs.rows().map(|r| r.iter().map(nc).collect()).collect()),
        Err(_) => Out::Err,
    }
}

#[derive(Default)]
struct Tally {
    ok: u32,
    semantic: u32,
    missing: u32,
    core_err: u32,
    panic: u32,
    parse_skip: u32,
    repro_semantic: Option<String>,
    repro_missing: Option<String>,
    repro_core_err: Option<String>,
    repro_panic: Option<String>,
}

fn main() {
    let iters: usize = std::env::var("FUZZ_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000);
    // Silence the default panic hook — the census CATCHES panics and reports them
    // as its worst bucket, so the per-panic backtrace spam is just noise.
    std::panic::set_hook(Box::new(|_| {}));

    // Each probe: (name, generator). The generator returns (query, ordered?) where
    // `ordered` picks position-for-position vs multiset comparison.
    type Gen = fn(&mut Rng) -> (String, bool);
    let probes: &[(&str, Gen)] = &[
        ("baseline_scan_filter", p_baseline),
        ("element_map", p_element),
        ("return_star", p_return_star),
        ("edge_element", p_edge_element),
        ("order_expr", p_order_expr),
        ("collect", p_collect),
        ("union", p_union),
        ("optional_match", p_optional),
        ("edge_props", p_edge_props),
        ("string_literal_eq", p_string_lit),
        ("str_infix", p_str_infix),
        ("arithmetic", p_arith),
        ("case_when", p_case),
        ("coalesce_nullif", p_coalesce),
        ("in_list", p_in_list),
        ("order_desc_nulls", p_order_desc),
        ("string_fns", p_string_fns),
        ("numeric_fns", p_numeric_fns),
        ("temporal_fns", p_temporal_fns),
        ("list_element_fns", p_list_fns),
        ("list_algebra_fns", p_list_algebra),
        ("type_fn", p_type_fn),
        ("element_id_fn", p_element_id),
        ("cast_fns", p_cast_fns),
        ("cross_type_cmp", p_cross_type),
        ("aggregates", p_agg),
        ("distinct", p_distinct),
        ("is_null_semantics", p_isnull),
    ];

    let mut tallies: BTreeMap<&str, Tally> = BTreeMap::new();
    for (name, gen) in probes {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ name.len() as u64 | 1);
        let t = tallies.entry(name).or_default();
        for _ in 0..iters {
            let g = gen_graph(&mut rng);
            let (q, ordered) = gen(&mut rng);
            let store = lenke_engine::ndjson::from_ndjson(&engine_ndjson(&g)).unwrap();
            let mut cg = lenke_core::ndjson::decode(&core_ndjson(&g)).unwrap();
            match (eng(&store, &q), core(&mut cg, &q)) {
                // A panic is the worst outcome — surface it regardless of core.
                (Out::Panic, _) => {
                    t.panic += 1;
                    t.repro_panic.get_or_insert_with(|| q.clone());
                }
                (Out::Rows(mut a), Out::Rows(mut b)) => {
                    if !ordered {
                        a.sort();
                        b.sort();
                    }
                    if a == b {
                        t.ok += 1;
                    } else {
                        t.semantic += 1;
                        t.repro_semantic.get_or_insert_with(|| q.clone());
                    }
                }
                (Out::Err | Out::Parse, Out::Rows(_)) => {
                    t.missing += 1;
                    t.repro_missing.get_or_insert_with(|| q.clone());
                }
                (Out::Rows(_), Out::Err | Out::Parse) => {
                    t.core_err += 1;
                    t.repro_core_err.get_or_insert_with(|| q.clone());
                }
                _ => t.parse_skip += 1,
            }
        }
    }

    println!("  probe                       ok  SEMANTIC  MISSING  CORE_ERR   PANIC   skip");
    for (name, t) in &tallies {
        println!(
            "{name:<22} {:>9} {:>9} {:>8} {:>9} {:>7} {:>6}",
            t.ok, t.semantic, t.missing, t.core_err, t.panic, t.parse_skip
        );
    }
    println!("\n── repros (one per probe×class) ──");
    for (name, t) in &tallies {
        if let Some(r) = &t.repro_panic {
            println!("PANIC     [{name}]  {r}");
        }
        if let Some(r) = &t.repro_semantic {
            println!("SEMANTIC  [{name}]  {r}");
        }
        if let Some(r) = &t.repro_missing {
            println!("MISSING   [{name}]  {r}");
        }
        if let Some(r) = &t.repro_core_err {
            println!("CORE_ERR  [{name}]  {r}");
        }
    }
}

// ── probe generators (each stresses one feature) ─────────────────────────────
fn var(rng: &mut Rng) -> &'static str {
    if rng.chance(2, 5) {
        "m"
    } else {
        "n"
    }
}
fn pattern(v: &str) -> &'static str {
    if v == "m" {
        "MATCH (n:N)-[r:R]->(m:N)"
    } else {
        "MATCH (n:N)"
    }
}
fn cmp_op(rng: &mut Rng) -> &'static str {
    rng.pick(&["<", "<=", ">", ">=", "=", "<>"])
}

fn p_element(rng: &mut Rng) -> (String, bool) {
    // Return a BARE node binding — exercises node-element map rendering
    // ({id, labels, properties}) against core, the RETURN n / RETURN * shape.
    let v = var(rng);
    (
        format!("{} RETURN {v}.id AS a0, {v} AS a1 ORDER BY a0", pattern(v)),
        true,
    )
}
fn p_return_star(rng: &mut Rng) -> (String, bool) {
    // RETURN * expands to every bound binding (node OR edge element maps), in slot
    // order, compared as a multiset. The two-hop pattern binds an edge var `r`, so
    // this now exercises edge-element rendering too.
    (format!("{} RETURN *", pattern(var(rng))), false)
}
fn p_optional(rng: &mut Rng) -> (String, bool) {
    // OPTIONAL MATCH left-outer hop: a node with no R-neighbour survives with `m`
    // NULL. Tests the null sentinel via property read / element render / count(m).
    // Compared as a multiset.
    let ret = *rng.pick(&["m.a AS a1", "m.b AS a1", "m AS a1", "count(m) AS a1"]);
    (
        format!("MATCH (n:N) OPTIONAL MATCH (n)-[:R]->(m) RETURN n.id AS a0, {ret}"),
        false,
    )
}
fn p_union(rng: &mut Rng) -> (String, bool) {
    // Two RETURN arms over the same label. UNION dedups the combined rows, UNION ALL
    // keeps them; names come from the left arm. Compared as a multiset.
    let all = if rng.chance(1, 2) { " ALL" } else { "" };
    let (e1, e2) = *rng.pick(&[
        ("n.a", "m.a"),
        ("n.b", "m.b"),
        ("n.a", "m.b"),
        ("n.id", "m.id"),
    ]);
    (
        format!("MATCH (n:N) RETURN {e1} AS a0 UNION{all} MATCH (m:N) RETURN {e2} AS a0"),
        false,
    )
}
fn p_collect(rng: &mut Rng) -> (String, bool) {
    // collect_list — grouped or scalar; nulls dropped, list in row order. The list
    // order must match, so grouped rows are compared as a multiset but each list is
    // structural. Scan order is id-order on both engines.
    let v = var(rng);
    if rng.chance(1, 2) {
        (
            format!(
                "{} RETURN {v}.b AS a0, collect_list({v}.a) AS a1 ORDER BY a0",
                pattern(v)
            ),
            true,
        )
    } else {
        (
            format!("{} RETURN collect_list({v}.a) AS a0", pattern(v)),
            false,
        )
    }
}
fn p_order_expr(rng: &mut Rng) -> (String, bool) {
    // ORDER BY an UNPROJECTED expression (v.a, not a returned column), with v.id
    // appended to fully break ties → a deterministic order both engines must match.
    let v = var(rng);
    let d = if rng.chance(1, 2) { " DESC" } else { "" };
    let lim = if rng.chance(1, 3) {
        format!(" LIMIT {}", 1 + rng.below(4))
    } else {
        String::new()
    };
    (
        format!(
            "{} RETURN {v}.b AS a0 ORDER BY {v}.a{d}, {v}.id{lim}",
            pattern(v)
        ),
        true,
    )
}
fn p_edge_element(_rng: &mut Rng) -> (String, bool) {
    // A bare EDGE binding — edge-element map rendering {id,from,to,labels,properties}
    // against core, compared as a multiset.
    ("MATCH (n:N)-[r:R]->(m:N) RETURN r".to_string(), false)
}
fn p_baseline(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    (
        format!(
            "{} WHERE {v}.a {} {} RETURN {v}.id AS a0, {v}.a AS a1 ORDER BY a1, a0",
            pattern(v),
            cmp_op(rng),
            rng.pick(NUMS)
        ),
        true,
    )
}
fn p_edge_props(rng: &mut Rng) -> (String, bool) {
    // r is only bound on the 1-hop pattern.
    let op = cmp_op(rng);
    let num = *rng.pick(NUMS);
    if rng.chance(1, 2) {
        (
            format!("MATCH (n:N)-[r:R]->(m:N) WHERE r.w {op} {num} RETURN n.id AS a0, r.w AS a1 ORDER BY a1, a0"),
            true,
        )
    } else {
        (
            "MATCH (n:N)-[r:R]->(m:N) RETURN r.w AS w, count(*) AS c".to_string(),
            false,
        )
    }
}
fn p_string_lit(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    (
        format!(
            "{} WHERE {v}.b {} '{}' RETURN {v}.id AS a0, {v}.b AS a1 ORDER BY a0",
            pattern(v),
            rng.pick(&["=", "<>"]),
            rng.pick(STRS)
        ),
        true,
    )
}
fn p_arith(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let e = match rng.below(5) {
        0 => format!("{v}.a / {}", rng.pick(NUMS)),
        1 => format!("{v}.a % {}", rng.pick(NUMS)),
        2 => format!("-{v}.a"),
        3 => format!("{v}.a + {v}.a * 2"),
        _ => format!("({v}.a - {}) / {}", rng.pick(NUMS), rng.pick(NUMS)),
    };
    (
        format!("{} RETURN {v}.id AS a0, {e} AS a1 ORDER BY a0", pattern(v)),
        true,
    )
}
fn p_case(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    (
        format!(
            "{} RETURN {v}.id AS a0, CASE WHEN {v}.a {} {} THEN 'hi' WHEN {v}.a IS NULL THEN 'nul' ELSE 'lo' END AS a1 ORDER BY a0",
            pattern(v), cmp_op(rng), rng.pick(NUMS)
        ),
        true,
    )
}
fn p_coalesce(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let e = match rng.below(3) {
        0 => format!("coalesce({v}.a, {})", rng.pick(NUMS)),
        1 => format!("coalesce({v}.a, {v}.id)"),
        _ => format!("nullif({v}.a, {})", rng.pick(NUMS)),
    };
    (
        format!("{} RETURN {v}.id AS a0, {e} AS a1 ORDER BY a0", pattern(v)),
        true,
    )
}
fn p_in_list(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    (
        format!(
            "{} WHERE {v}.a IN [{}, {}, {}] RETURN {v}.id AS a0 ORDER BY a0",
            pattern(v),
            rng.pick(NUMS),
            rng.pick(NUMS),
            rng.pick(NUMS)
        ),
        true,
    )
}
fn p_order_desc(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let dir = if rng.chance(1, 2) { "DESC" } else { "ASC" };
    (
        format!(
            "{} RETURN {v}.id AS a0, {v}.a AS a1 ORDER BY a1 {dir}, a0",
            pattern(v)
        ),
        true,
    )
}
fn p_string_fns(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let f = rng.pick(&[
        "upper",
        "lower",
        "trim",
        "ltrim",
        "rtrim",
        "btrim",
        "reverse",
        "length",
        "char_length",
        "character_length",
        "size",
        "left",
        "right",
        "split",
        "substring",
        "replace",
    ]);
    let call = match *f {
        "left" | "right" => format!("{f}({v}.b, {})", 1 + rng.below(4)),
        // Both a non-empty and an EMPTY delimiter (the empty case diverged: Rust's
        // split("") vs the JS per-code-unit model).
        "split" => format!("{f}({v}.b, '{}')", rng.pick(&["X", "", "😀"])),
        // Vary start (incl. <= 0) and presence of the length arg — exercises the
        // ISO 1-based indexing and the from-the-front shrink on a non-positive start.
        "substring" => {
            let start = ["0", "1", "2", "-1"][rng.below(4)];
            if rng.chance(1, 2) {
                format!("{f}({v}.b, {start}, {})", rng.below(4))
            } else {
                format!("{f}({v}.b, {start})")
            }
        }
        "replace" => format!("{f}({v}.b, '{}', '{}')", rng.pick(STRS), rng.pick(STRS)),
        _ => format!("{f}({v}.b)"),
    };
    (
        format!(
            "{} RETURN {v}.id AS a0, {call} AS a1 ORDER BY a0",
            pattern(v)
        ),
        true,
    )
}
fn p_temporal_fns(rng: &mut Rng) -> (String, bool) {
    // Component accessors over literal temporals — core spells them with the
    // underscore sigil (`_year`, `_month`, …), which the engine must accept.
    let v = var(rng);
    let (ctor, comp) = match rng.below(4) {
        0 => (
            "date('2020-05-17')",
            ["_year", "_month", "_day"][rng.below(3)],
        ),
        1 => (
            "datetime('2020-05-17T13:45:06')",
            ["_year", "_hour", "_minute", "_second"][rng.below(4)],
        ),
        2 => (
            "local_time('13:45:06')",
            ["_hour", "_minute", "_second", "_year"][rng.below(4)], // _year of a time → NULL, both
        ),
        _ => (
            "date('1999-12-31')",
            ["_year", "_month", "_day"][rng.below(3)],
        ),
    };
    (
        format!(
            "{} RETURN {v}.id AS a0, {comp}({ctor}) AS a1 ORDER BY a0",
            pattern(v)
        ),
        true,
    )
}
fn p_numeric_fns(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let f = rng.pick(&[
        "abs", "sign", "floor", "ceil", "ceiling", "round", "sqrt", "exp", "ln", "log", "sin",
        "cos", "tan", "power", "mod", "e", "pi", "degrees", "radians",
    ]);
    let call = match *f {
        "e" | "pi" => format!("{f}()"),
        "power" | "mod" | "log" => format!("{f}({v}.a, 2)"),
        _ => format!("{f}({v}.a)"),
    };
    (
        format!(
            "{} RETURN {v}.id AS a0, {call} AS a1 ORDER BY a0",
            pattern(v)
        ),
        true,
    )
}
fn p_list_fns(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let f = rng.pick(&["head", "last", "tail", "reverse", "size", "range", "keys"]);
    let call = match *f {
        "range" => "range(1, 4)".to_string(),
        "keys" => format!("keys({v})"),
        _ => format!("{f}([1, 2, 3])"),
    };
    (
        format!(
            "{} RETURN {v}.id AS a0, {call} AS a1 ORDER BY a0",
            pattern(v)
        ),
        true,
    )
}
fn p_list_algebra(rng: &mut Rng) -> (String, bool) {
    // Random small-int list literals (a few with a repeat or a null) through the
    // set/list-algebra functions.
    let lst = |rng: &mut Rng| -> String {
        let n = 1 + rng.below(4);
        let items: Vec<String> = (0..n)
            .map(|_| match rng.below(5) {
                0 => "null".to_string(),
                k => (k % 3).to_string(),
            })
            .collect();
        format!("[{}]", items.join(", "))
    };
    let a = lst(rng);
    let b = lst(rng);
    let e = match rng.below(6) {
        0 => format!("append({a}, 9)"),
        1 => format!("list_contains({a}, 1)"),
        2 => format!("list_sort({a})"),
        3 => format!("list_union({a}, {b})"),
        4 => format!("difference({a}, {b})"),
        _ => format!("intersection({a}, {b})"),
    };
    (
        format!("MATCH (n:N) RETURN n.id AS a0, {e} AS a1 ORDER BY a0"),
        true,
    )
}
fn p_type_fn(_rng: &mut Rng) -> (String, bool) {
    // type(edge) over the 1-hop pattern.
    (
        "MATCH (n:N)-[r:R]->(m:N) RETURN n.id AS a0, type(r) AS a1 ORDER BY a0, a1".to_string(),
        true,
    )
}
fn p_element_id(rng: &mut Rng) -> (String, bool) {
    // Preserved external ids for nodes and edges.
    if rng.chance(1, 2) {
        (
            "MATCH (n:N) RETURN element_id(n) AS a0 ORDER BY a0".to_string(),
            true,
        )
    } else {
        (
            "MATCH (n:N)-[r:R]->(m:N) RETURN element_id(r) AS a0 ORDER BY a0".to_string(),
            true,
        )
    }
}
fn p_str_infix(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let (op, arg) = match rng.below(3) {
        0 => ("CONTAINS", rng.pick(STRS)),
        1 => ("STARTS WITH", rng.pick(STRS)),
        _ => ("ENDS WITH", rng.pick(STRS)),
    };
    (
        format!(
            "{} WHERE {v}.b {op} '{arg}' RETURN {v}.id AS a0, {v}.b AS a1 ORDER BY a0",
            pattern(v)
        ),
        true,
    )
}
fn p_cast_fns(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let f = rng.pick(&[
        "to_string",
        "tostring",
        "to_integer",
        "tointeger",
        "to_float",
        "tofloat",
        "to_boolean",
        "toboolean",
    ]);
    (
        format!(
            "{} RETURN {v}.id AS a0, {f}({v}.a) AS a1 ORDER BY a0",
            pattern(v)
        ),
        true,
    )
}
fn p_cross_type(rng: &mut Rng) -> (String, bool) {
    // n.a (num) vs n.b (str): a KNOWN divergence (core throws on ordering).
    let v = var(rng);
    (
        format!(
            "{} WHERE {v}.a {} {v}.b RETURN {v}.id AS a0 ORDER BY a0",
            pattern(v),
            cmp_op(rng)
        ),
        true,
    )
}
fn p_agg(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let a = rng.pick(&[
        "count(*)",
        "count(DISTINCT n.a)",
        "sum(n.a)",
        "avg(n.a)",
        "min(n.a)",
        "max(n.a)",
        "min(n.b)",
        "max(n.b)",
    ]);
    let a = a.replace("n.", &format!("{v}."));
    if rng.chance(1, 2) {
        (format!("{} RETURN {v}.b AS k, {a} AS x", pattern(v)), false)
    } else {
        (format!("{} RETURN {a} AS x", pattern(v)), false)
    }
}
fn p_distinct(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let e = if rng.chance(1, 2) {
        format!("{v}.a")
    } else {
        format!("{v}.b")
    };
    (format!("{} RETURN DISTINCT {e} AS d", pattern(v)), false)
}
fn p_isnull(rng: &mut Rng) -> (String, bool) {
    let v = var(rng);
    let k = if rng.chance(1, 2) { "a" } else { "b" };
    let t = rng.pick(&["IS NULL", "IS NOT NULL"]);
    (
        format!(
            "{} WHERE {v}.{k} {t} RETURN {v}.id AS a0 ORDER BY a0",
            pattern(v)
        ),
        true,
    )
}
