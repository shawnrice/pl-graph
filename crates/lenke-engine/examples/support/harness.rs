//! Shared benchmark harness for the consolidated `lenke-engine` bench corpus.
//!
//! This file is NOT an example itself — it is `#[path]`-included by each themed
//! bench binary (`ingest_bench`, `query_bench`, …) and by `bench_all`, so every
//! benchmark shares ONE timing loop, ONE deterministic RNG and ONE set of fixture
//! builders. Adding a case means adding a row to a group module, never a new
//! binary. See `examples/README.md` for the by-question index.
//!
//! House rules (from CLAUDE.md, enforced here): min-of-`reps` wall time (never a
//! mean); `std::hint::black_box` on every result so a run cannot be elided; a
//! dep-free LCG for reproducibility (`rand`/`Math.random` are off-limits — a bench
//! compared across builds must be deterministic); and sweep the size across the
//! 200k–1M cache transition rather than trusting one point.

use std::time::Instant;

/// Run-wide knobs, from the environment and the first CLI argument.
///
/// - `BENCH_REPS=<n>`  — samples per case (min is reported). Default 7.
/// - `BENCH_N=<n>`     — override the primary sweep size (a group may ignore it
///   for a case whose question is about the size sweep itself).
/// - first CLI arg     — a case filter: a substring matched against each case's
///   `group/case` label, so `-- ingest` runs one group and `-- phases` one case.
pub struct Cfg {
    pub reps: usize,
    pub scale: Option<usize>,
    pub filter: Option<String>,
}

impl Cfg {
    #[must_use]
    pub fn from_env() -> Self {
        let reps = std::env::var("BENCH_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(7);
        let scale = std::env::var("BENCH_N").ok().and_then(|s| s.parse().ok());
        // Skip a leading `--` some shells pass through before the filter word.
        let filter = std::env::args().skip(1).find(|a| a != "--");
        Self {
            reps,
            scale,
            filter,
        }
    }

    /// Whether `label` (a `group/case` string) is selected by the filter. No
    /// filter selects everything; a filter selects any label that contains it.
    #[must_use]
    pub fn want(&self, label: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|f| label.contains(f.as_str()))
    }
}

/// A section header for a group's output, so a `bench_all` run reads as a
/// sequence of clearly separated tables.
pub fn section(title: &str) {
    println!("\n=== {title} ===");
}

/// Min-of-`reps` wall time in MILLISECONDS for `f`, with the result black-boxed.
/// `f` returns a value that is fed to `black_box` so the optimizer cannot drop
/// the work; a warm-up run precedes the measured ones.
pub fn best_ms<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    std::hint::black_box(f());
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let out = f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(out);
    }
    best
}

/// Min-of-`reps` wall time in MICROSECONDS — for per-query / per-op costs that
/// would round to zero in milliseconds.
pub fn best_us<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    best_ms(reps, &mut f) * 1e3
}

/// A tiny deterministic LCG (Numerical Recipes constants). Reproducible target
/// selection without a dependency — the same fixture every build.
pub struct Lcg(pub u64);

impl Lcg {
    #[must_use]
    pub fn seeded() -> Self {
        Self(0x9E37_79B9_7F4A_7C15)
    }

    pub fn next(&mut self, bound: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as u32) % bound.max(1)
    }
}

/// Build an NDJSON document of `nodes` `Person` nodes (each with `props` string
/// properties `k0..`), optionally followed by `deg` `KNOWS` out-edges per node to
/// deterministic pseudo-random targets. The single knob every ingest/codec/scale
/// question shares, so the fixtures are comparable across groups.
#[must_use]
pub fn social_ndjson(nodes: usize, props: usize, deg: usize) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(nodes * (1 + deg));
    for i in 0..nodes {
        let p: Vec<String> = (0..props)
            .map(|k| format!(r#""k{k}":"v{i}_{k}""#))
            .collect();
        lines.push(format!(
            r#"{{"type":"node","id":"v{i}","labels":["Person"],"properties":{{{}}}}}"#,
            p.join(",")
        ));
    }
    if deg > 0 && nodes > 0 {
        let mut rng = Lcg::seeded();
        let mut e = 0usize;
        for from in 0..nodes {
            for _ in 0..deg {
                let to = rng.next(nodes as u32);
                lines.push(format!(
                    r#"{{"type":"edge","id":"e{e}","labels":["KNOWS"],"from":"v{from}","to":"v{to}","properties":{{"w":1}}}}"#
                ));
                e += 1;
            }
        }
    }
    lines.join("\n")
}

const CITIES: [&str; 8] = [
    "Springfield",
    "Shelbyville",
    "Ogdenville",
    "Capital City",
    "Cypress Creek",
    "North Haverbrook",
    "Brockway",
    "Waverly Hills",
];
const DEPTS: [&str; 5] = ["eng", "sales", "ops", "legal", "hr"];

/// A richer in-memory `Store` for query/storage/index/scale questions: `nodes`
/// `Person` nodes each with `name` (unique-ish), `age` (0..100, numeric),
/// `city` (8-way) and `dept` (5-way) properties, plus `deg` `KNOWS` out-edges to
/// deterministic targets. The 8-/5-way low-cardinality props feed grouping and
/// index-seek questions; `age` feeds numeric filters. Built via `Builder`, so no
/// NDJSON text is held resident during a measurement.
#[must_use]
pub fn social_store(nodes: u32, deg: u32) -> lenke_engine::store::Store {
    use lenke_engine::store::Builder;
    use lenke_engine::value::Value;
    let mut b = Builder::default();
    for i in 0..nodes {
        b.node(
            &["Person"],
            &[
                ("name", Value::Str(format!("name{i}").into())),
                ("age", Value::Num(f64::from(i % 100))),
                ("city", Value::Str(CITIES[(i % 8) as usize].into())),
                ("dept", Value::Str(DEPTS[(i % 5) as usize].into())),
            ],
        );
    }
    let mut rng = Lcg::seeded();
    for i in 0..nodes {
        for _ in 0..deg {
            b.edge(i, rng.next(nodes), "KNOWS");
        }
    }
    b.build()
}

/// Resident set size in bytes, read from `/proc/self/status` (`VmRSS`). `None`
/// off Linux. Used by the memory question; call after building a graph, and
/// subtract a pre-build baseline to isolate the graph's footprint.
#[must_use]
pub fn rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}
