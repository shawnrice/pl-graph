//! Generative Gremlin perf-fuzzer: `lenke-engine` vs `lenke-core` over THOUSANDS of
//! randomly-composed traversals, to find slowdowns (and any lingering divergence) no
//! hand-written bench would. The Gremlin analog of `perf_fuzz.rs` (which fuzzes GQL).
//!
//! It composes building blocks — a source (`g.V()`, `hasLabel`-narrowed, or id-seeded),
//! either a `repeat(...)` walk (`times`/`emit`/`until`/body-filter) or 0-3 vertex/edge
//! hops, an optional element filter (`has`/`hasLabel`/`where`/`dedup`), and a tail that
//! is a barrier (`count`/`fold`), a value stream (`values`/`id`/`label`), a reducing
//! aggregate (`mean`/`sum`/`min`/`max`), a grouping (`groupCount`/`group`), an
//! order+limit, a `path()`/`tree()` (with/without `by`), or `valueMap`/`elementMap`.
//! Every block contributes a feature TAG.
//!
//! Each generated traversal is CANONICALIZED to a template (literals → `#`/`$`) and
//! deduped, so N random queries collapse to the unique STRUCTURES worth timing. Each
//! template's representative is timed min-of-`REPS` on both engines; the ratio is
//! core_ms/engine_ms (>1 = engine faster). The ENGINE runs first under a wall-clock
//! BUDGET (a pathological shape is reported as a TIMEOUT, not a hang); a row-count
//! MISMATCH between the two is flagged loudly — a free correctness signal over shapes
//! the ported corpus never exercised. The report ranks the slowest templates and
//! averages the ratio per feature tag, so a slow building block is named.
//!
//! Both engines run their NATURAL Gremlin path: engine = `gremlin::parse` + `exec::run`
//! (the path the ported suite validates); set `GREMLIN_OPT=1` to additionally run the
//! engine plan through `opt::optimize_indexed` (unindexed store, so it is a near-no-op —
//! there for measuring the optimizer's own overhead). Neither side is given indexes, so
//! this isolates the two EXECUTION models, not index seeding.
//!
//! Native only. Run:
//!   cargo run --release --manifest-path crates/lenke-engine/Cargo.toml \
//!     --example gremlin_perf_fuzz
//!   FUZZ_N=20000 BENCH_N=50000 PERF_SEED=7 cargo run --release ... --example gremlin_perf_fuzz

use lenke_engine::store::Store;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

// --- rng (self-contained xorshift64*) ---------------------------------------

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn chance(&mut self, num: u32, den: u32) -> bool {
        self.below(den as usize) < num as usize
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        let i = self.below(xs.len());
        &xs[i]
    }
}

// --- fixture (one ndjson → both engines, so ids/edges are identical) ---------

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

/// Build BOTH engines' ndjson from the SAME logical graph, so ids/edges/props are
/// identical and only the serialization differs (engine: `props`, edge `type`; core:
/// `properties`, `type:node|edge`, `labels`). Returns `(engine_ndjson, core_ndjson)`.
fn fixture_ndjson(n: u32, deg: u32) -> (String, String) {
    let mut es = String::new();
    let mut cs = String::new();
    for i in 0..n {
        let labels = if i % 3 == 0 {
            r#"["Person","VIP"]"#
        } else {
            r#"["Person"]"#
        };
        let props = format!(
            r#""name":"n{i}","age":{},"score":{},"city":"{}""#,
            i % 100,
            i.wrapping_mul(7) % 1000,
            CITIES[(i % CITIES.len() as u32) as usize],
        );
        es.push_str(&format!(
            r#"{{"id":"{i}","labels":{labels},"props":{{{props}}}}}"#
        ));
        es.push('\n');
        cs.push_str(&format!(
            r#"{{"type":"node","id":"{i}","labels":{labels},"properties":{{{props}}}}}"#
        ));
        cs.push('\n');
    }
    let mut e = 0u64;
    let mut push_edge = |es: &mut String, cs: &mut String, from: u32, to: u32, ty: &str| {
        let w = e % 1000;
        es.push_str(&format!(
            r#"{{"id":"e{e}","from":"{from}","to":"{to}","type":"{ty}","props":{{"w":{w}}}}}"#
        ));
        es.push('\n');
        cs.push_str(&format!(
            r#"{{"type":"edge","id":"e{e}","labels":["{ty}"],"from":"{from}","to":"{to}","properties":{{"w":{w}}}}}"#
        ));
        cs.push('\n');
        e += 1;
    };
    for i in 0..n {
        for d in 0..deg {
            let to = (i
                .wrapping_mul(7)
                .wrapping_add(d.wrapping_mul(13))
                .wrapping_add(1))
                % n;
            push_edge(&mut es, &mut cs, i, to, "R");
        }
        if i % 5 == 0 {
            push_edge(
                &mut es,
                &mut cs,
                i,
                (i.wrapping_mul(3).wrapping_add(2)) % n,
                "F",
            );
        }
    }
    (es, cs)
}

// --- traversal generator -----------------------------------------------------

const NUM_PROPS: &[&str] = &["age", "score"];
const STR_PROPS: &[&str] = &["name", "city"];
const LABELS: &[&str] = &["Person", "VIP"];
const ETYPES: &[&str] = &["R", "F"];

struct Gen<'a> {
    rng: &'a mut Rng,
    n: usize,
    tags: Vec<&'static str>,
}

impl Gen<'_> {
    fn tag(&mut self, t: &'static str) {
        if !self.tags.contains(&t) {
            self.tags.push(t);
        }
    }

    /// A string literal matching a real property value (so filters actually select).
    fn str_lit(&mut self) -> String {
        if self.rng.chance(1, 2) {
            format!("'{}'", self.rng.pick(CITIES))
        } else {
            format!("'n{}'", self.rng.below(self.n))
        }
    }

    fn source(&mut self) -> String {
        match self.rng.below(10) {
            0..=4 => {
                self.tag("src-scan");
                "g.V()".to_string()
            }
            5..=7 => {
                self.tag("src-label");
                format!("g.V().hasLabel('{}')", self.rng.pick(LABELS))
            }
            _ => {
                self.tag("src-id");
                let k = 1 + self.rng.below(3);
                let ids: Vec<String> = (0..k)
                    .map(|_| format!("'{}'", self.rng.below(self.n)))
                    .collect();
                format!("g.V({})", ids.join(", "))
            }
        }
    }

    /// One vertex or edge hop. Edge hops are always paired back to a vertex frontier
    /// (`outE().inV()`, …) so every following tail sees a vertex stream.
    fn hop(&mut self) -> String {
        match self.rng.below(10) {
            0..=3 => {
                self.tag("hop-out");
                self.dir_hop("out")
            }
            4..=5 => {
                self.tag("hop-in");
                self.dir_hop("in")
            }
            6..=7 => {
                self.tag("hop-both");
                self.dir_hop("both")
            }
            _ => {
                self.tag("hop-edge");
                let et = if self.rng.chance(1, 2) {
                    format!("'{}'", self.rng.pick(ETYPES))
                } else {
                    String::new()
                };
                let (e, v) =
                    *self
                        .rng
                        .pick(&[("outE", "inV"), ("inE", "outV"), ("bothE", "otherV")]);
                format!(".{e}({et}).{v}()")
            }
        }
    }
    fn dir_hop(&mut self, d: &str) -> String {
        if self.rng.chance(1, 2) {
            format!(".{d}('{}')", self.rng.pick(ETYPES))
        } else {
            format!(".{d}()")
        }
    }

    fn repeat_block(&mut self) -> String {
        self.tag("repeat");
        // Prefer single-direction hops (a `both()` walk fans out ~2x/iteration) and cap
        // the depth at 2 so a fixed walk stays bounded (deg^2 endpoints/source) — both
        // engines run it inline, so an unbounded blow-up would hang the fuzz.
        let d = *self.rng.pick(&["out", "out", "in", "in", "both"]);
        let et = if self.rng.chance(1, 3) {
            format!("'{}'", self.rng.pick(ETYPES))
        } else {
            String::new()
        };
        let body = format!("__.{d}({et})");
        let times = 1 + self.rng.below(2);
        match self.rng.below(6) {
            0..=2 => format!(".repeat({body}).times({times})"),
            3 => {
                self.tag("repeat-emit");
                format!(".repeat({body}).times({times}).emit()")
            }
            4 => {
                // `until` bounded by a `times` cap — an unbounded until on a cyclic graph
                // would run to the 100-iteration ceiling (a catastrophic fan-out).
                self.tag("repeat-until");
                format!(
                    ".repeat({body}).times(2).until(__.hasLabel('{}'))",
                    self.rng.pick(LABELS)
                )
            }
            _ => {
                self.tag("repeat-bodyfilter");
                format!(
                    ".repeat({body}.hasLabel('{}')).times({times})",
                    self.rng.pick(LABELS)
                )
            }
        }
    }

    fn filter_block(&mut self) -> String {
        match self.rng.below(8) {
            0..=2 => {
                self.tag("has-num");
                let p = self.rng.pick(NUM_PROPS);
                let op = *self.rng.pick(&["gt", "gte", "lt", "lte", "eq", "neq"]);
                format!(".has('{p}', {op}({}))", self.rng.below(100))
            }
            3..=4 => {
                self.tag("has-str");
                let p = *self.rng.pick(STR_PROPS);
                format!(".has('{p}', {})", self.str_lit())
            }
            5 => {
                self.tag("has-label");
                format!(".hasLabel('{}')", self.rng.pick(LABELS))
            }
            6 => {
                self.tag("where-hop");
                let d = *self.rng.pick(&["out", "in", "both"]);
                format!(".where(__.{d}())")
            }
            _ => {
                self.tag("dedup");
                ".dedup()".to_string()
            }
        }
    }

    fn tail(&mut self) -> String {
        match self.rng.below(16) {
            0..=2 => {
                self.tag("count");
                ".count()".to_string()
            }
            3..=4 => {
                self.tag("values");
                format!(
                    ".values('{}')",
                    self.rng.pick(&["name", "age", "city", "score"])
                )
            }
            5 => {
                self.tag("agg");
                let a = *self.rng.pick(&["mean", "sum", "max", "min"]);
                format!(".values('{}').{a}()", self.rng.pick(NUM_PROPS))
            }
            6 => {
                self.tag("groupcount");
                if self.rng.chance(1, 2) {
                    format!(".groupCount().by('{}')", self.rng.pick(STR_PROPS))
                } else {
                    ".groupCount().by(T.label)".to_string()
                }
            }
            7 => {
                self.tag("group");
                format!(
                    ".group().by('{}').by('{}')",
                    self.rng.pick(STR_PROPS),
                    self.rng.pick(NUM_PROPS)
                )
            }
            8 => {
                self.tag("order-limit");
                format!(
                    ".order().by('{}').limit({})",
                    self.rng.pick(NUM_PROPS),
                    1 + self.rng.below(50)
                )
            }
            9..=10 => {
                self.tag("path");
                if self.rng.chance(1, 2) {
                    format!(".path().by('{}')", self.rng.pick(STR_PROPS))
                } else {
                    ".path()".to_string()
                }
            }
            11 => {
                self.tag("tree");
                if self.rng.chance(1, 2) {
                    format!(".tree().by('{}')", self.rng.pick(STR_PROPS))
                } else {
                    ".tree()".to_string()
                }
            }
            12 => {
                self.tag("valuemap");
                if self.rng.chance(1, 2) {
                    ".valueMap()".to_string()
                } else {
                    ".elementMap()".to_string()
                }
            }
            13 => {
                self.tag("id-label");
                if self.rng.chance(1, 2) {
                    ".id()".to_string()
                } else {
                    ".label()".to_string()
                }
            }
            _ => {
                self.tag("fold");
                ".fold()".to_string()
            }
        }
    }

    fn query(&mut self) -> String {
        let mut q = self.source();
        if self.rng.chance(1, 4) {
            q += &self.repeat_block();
        } else {
            let nhops = self.rng.below(4);
            for _ in 0..nhops {
                q += &self.hop();
            }
        }
        if self.rng.chance(2, 5) {
            q += &self.filter_block();
        }
        q += &self.tail();
        q
    }
}

/// Canonicalize a query to a template: numeric literals → `#`, quoted strings → `$`,
/// so queries differing only in constants collapse to one structure.
fn template(q: &str) -> String {
    let mut out = String::with_capacity(q.len());
    let bytes = q.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '\'' {
            out.push('$');
            i += 1;
            while i < bytes.len() && bytes[i] != b'\'' {
                i += 1;
            }
            i += 1;
        } else if c.is_ascii_digit() {
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

fn time_engine_guarded(
    store: &Arc<Store>,
    q: &str,
    reps: usize,
    budget: Duration,
    opt: bool,
    materialize: bool,
) -> Result<(f64, usize), String> {
    let plan = lenke_engine::gremlin::parse(q).map_err(|e| format!("parse: {e}"))?;
    let plan = if opt {
        lenke_engine::opt::optimize_indexed(plan, store.as_ref())
    } else {
        plan
    };
    let plan = Arc::new(plan);
    let (tx, rx) = mpsc::channel();
    let (s, p) = (Arc::clone(store), Arc::clone(&plan));
    std::thread::spawn(move || {
        let mut best = f64::MAX;
        let mut rows = 0;
        for _ in 0..reps {
            let t = Instant::now();
            // Serialize to JSON inside the timed region when `materialize` — the SAME
            // work core is charged (results_to_json), so string/element shapes are a fair
            // fight (the shipped lnk_e_gremlin_json path serializes too).
            let out = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let r = lenke_engine::exec::run(&p, &s);
                let n = r.rows.len();
                if materialize {
                    std::hint::black_box(lenke_engine::json::gremlin_results_json(&r));
                }
                n
            }));
            match out {
                Ok(n) => {
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    rows = n;
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
    materialize: bool,
) -> Result<(f64, usize), String> {
    let t = lenke_core::gremlin::parse(q).map_err(|e| format!("parse: {e}"))?;
    let mut best = f64::MAX;
    let mut rows = 0;
    for _ in 0..reps {
        let ti = Instant::now();
        let rs = lenke_core::gremlin::try_run(graph, &t).map_err(|e| format!("exec: {e:?}"))?;
        // Core returns LAZY GVal::Node/Edge handles; the engine's run() fully MATERIALIZES
        // its result (render_cell → element-map Values). To compare "time to produce the
        // usable result" (not "time to a lazy handle"), force core to render its output —
        // else element-returning shapes (fold/valueMap/path/id) measure work core simply
        // deferred. Mirrors the GQL harness, which times core's materialized RowSet.
        if materialize {
            let s = lenke_core::gremlin::exec::results_to_json(graph, &rs);
            std::hint::black_box(&s);
        }
        best = best.min(ti.elapsed().as_secs_f64() * 1e3);
        rows = rs.len();
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

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(60).collect()
}

fn main() {
    const REPS: usize = 3;
    let n = env_u32("BENCH_N", 10_000);
    let deg = env_u32("BENCH_DEG", 4);
    let gen_n = env_u32("FUZZ_N", 20_000) as usize;
    let seed = std::env::var("PERF_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xC0FFEE);
    let max_time = env_u32("FUZZ_MAX_TEMPLATES", 2000) as usize;
    let budget = Duration::from_millis(u64::from(env_u32("FUZZ_BUDGET_MS", 1500)));
    // Optimize by default — the shipped lnk_e_gremlin_json path runs optimize_indexed,
    // so a fair comparison does too (on an unindexed fixture it is near-free anyway).
    let opt = env_u32("GREMLIN_OPT", 1) != 0;
    // Materialize core's result (render to output) so element-returning shapes compare
    // the SAME work the engine's run() already does. Set MATERIALIZE=0 to measure core's
    // lazy try_run instead (engine looks worse on fold/valueMap/path — an artifact).
    let materialize = env_u32("MATERIALIZE", 1) != 0;

    eprintln!(
        "fixture: {n} nodes, deg {deg}; generating {gen_n} traversals (seed {seed}); engine-opt={opt}"
    );
    let (engine_nd, core_nd) = fixture_ndjson(n, deg);
    let estore = Arc::new(lenke_engine::ndjson::from_ndjson(&engine_nd).expect("engine fixture"));
    let mut cgraph = lenke_core::ndjson::decode(&core_nd).expect("core fixture");

    // 1) Generate and dedup to unique templates (keep the first concrete instance).
    let mut rng = Rng(seed);
    let mut templates: HashMap<String, (String, Vec<&'static str>)> = HashMap::new();
    for _ in 0..gen_n {
        let mut g = Gen {
            rng: &mut rng,
            n: n as usize,
            tags: Vec::new(),
        };
        let q = g.query();
        let tags = g.tags;
        templates.entry(template(&q)).or_insert((q, tags));
    }
    eprintln!("{} unique templates", templates.len());

    // 2) Time each template. ENGINE first (budget-guarded); then core inline.
    let mut results: Vec<Timed> = Vec::new();
    let mut skips: HashMap<String, usize> = HashMap::new();
    let mut mismatches: Vec<(String, usize, usize)> = Vec::new();
    let mut timeouts: Vec<String> = Vec::new();
    let mut engine_panics: Vec<String> = Vec::new();
    let mut timed = 0usize;
    for (q, tags) in templates.values() {
        if timed >= max_time {
            break;
        }
        let e = match time_engine_guarded(&estore, q, REPS, budget, opt, materialize) {
            Ok(v) => v,
            Err(why) if why == "TIMEOUT" => {
                timeouts.push(q.clone());
                continue;
            }
            Err(why) if why == "engine panic" => {
                engine_panics.push(q.clone());
                continue;
            }
            Err(why) => {
                *skips
                    .entry(format!("engine {}", first_line(&why)))
                    .or_insert(0) += 1;
                continue;
            }
        };
        let c = match time_core(&mut cgraph, q, REPS, materialize) {
            Ok(v) => v,
            Err(why) => {
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
        "\n=== {total} templates timed | {wins} win / {} lose | {} skipped kinds ===",
        total - wins,
        skips.values().sum::<usize>()
    );

    if !mismatches.is_empty() {
        println!("\n!!! ROW-COUNT MISMATCHES (correctness signal) !!!");
        for (q, e, c) in mismatches.iter().take(25) {
            println!("  engine={e} core={c}  {q}");
        }
    }
    if !engine_panics.is_empty() {
        println!(
            "\n!!! {} ENGINE PANICS (parseable, core-runnable) !!!",
            engine_panics.len()
        );
        for q in engine_panics.iter().take(15) {
            println!("  {q}");
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

    // Per-feature average ratio (which building block is slow).
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
    println!("\nper-feature mean ratio (worst first, >1 = engine faster):");
    println!("  {:>7}  {:>5}  {:>5}  feature", "mean", "n", "lose");
    for (t, mean, cnt, lose) in &tag_rows {
        println!("  {mean:>7.2}  {cnt:>5}  {lose:>5}  {t}");
    }

    // Slowest templates (only where the absolute time is meaningful).
    results.sort_by(|a, b| a.ratio.partial_cmp(&b.ratio).unwrap());
    println!("\nslowest templates (ratio, engine_ms, core_ms, rows, tags, query):");
    let mut shown = 0;
    for r in &results {
        if r.e_ms < 0.2 && r.c_ms < 0.2 {
            continue;
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
}
