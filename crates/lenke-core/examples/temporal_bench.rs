//! Temporal-column micro-benchmark. Builds a graph whose Person nodes each carry
//! one property of every temporal type (Date, Time, DateTime, ZonedTime,
//! ZonedDateTime, Duration) and times the operations that de-boxing should speed
//! up: scan+filter, ORDER BY, projection, and min/max aggregate over each column.
//!
//! Today every temporal property lives in a `Column::Mixed` (`Vec<Option<Value>>`,
//! ~40 B/slot, scalar eval only). This is the BASELINE; re-run after de-boxing
//! temporals into typed packed columns to measure the gain per type.
//!
//! Comparison values are passed as `$p` params (`Val::Temporal`) so the same query
//! shape works for the types with no literal form (Time / Zoned*). Run:
//!   cargo run --release --example temporal_bench

use std::time::Instant;

use lenke_core::gql::eval::{Params, Val};
use lenke_core::gql::prepare;
use lenke_core::graph::{Builder, Graph, NodeRec, Value};
use lenke_core::temporal::Temporal;

const N: usize = 200_000; // persons

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// One temporal type under test: how to build the i-th stored value, and the
/// mid-range comparison value used by the filter/aggregate queries.
struct TempKind {
    key: &'static str,
    /// Build a spread-out value for element `i` (so sorts/filters are non-trivial).
    make: fn(usize) -> Temporal,
    /// The `$p` comparison value (≈ the middle of the range).
    probe: Temporal,
}

fn t(tag: &str, s: &str) -> Temporal {
    Temporal::parse(tag, s).unwrap()
}

/// Read the process resident-set size (KiB) from /proc — a rough memory proxy.
fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).map(|p| p.to_string()))
        .and_then(|pages| pages.parse::<u64>().ok())
        .map(|pages| pages * 4) // pages are 4 KiB
        .unwrap_or(0)
}

fn kinds() -> Vec<TempKind> {
    vec![
        TempKind {
            key: "d_date",
            make: |i| {
                // 1900-01-01 + (i mod 73000) days ≈ a 200-year spread.
                let days = -25_567 + (i % 73_000) as i32;
                Temporal::Date(lenke_core::temporal::Date { days })
            },
            probe: t("date", "2000-01-01"),
        },
        TempKind {
            key: "d_time",
            make: |i| {
                let secs = (i % 86_400) as u32;
                Temporal::Time(lenke_core::temporal::Time { secs, nanos: 0 })
            },
            probe: t("localtime", "12:00:00"),
        },
        TempKind {
            key: "d_datetime",
            make: |i| {
                let secs = 1_577_836_800 + (i % 63_072_000) as i64; // 2020 + up to 2y
                Temporal::DateTime(lenke_core::temporal::DateTime { secs, nanos: 0 })
            },
            probe: t("datetime", "2021-01-01T00:00:00"),
        },
        TempKind {
            key: "d_ztime",
            make: |i| {
                let secs = (i % 86_400) as u32;
                Temporal::ZonedTime(lenke_core::temporal::ZonedTime {
                    secs,
                    nanos: 0,
                    offset: 60,
                })
            },
            probe: t("zoned_time", "12:00:00+01:00"),
        },
        TempKind {
            key: "d_zdatetime",
            make: |i| {
                let secs = 1_577_836_800 + (i % 63_072_000) as i64;
                Temporal::ZonedDateTime(lenke_core::temporal::ZonedDateTime {
                    secs,
                    nanos: 0,
                    offset: 60,
                })
            },
            probe: t("zoned_datetime", "2021-01-01T00:00:00+01:00"),
        },
        TempKind {
            key: "d_duration",
            make: |i| {
                let days = (i % 100_000) as i64;
                Temporal::Duration(lenke_core::temporal::Duration {
                    months: 0,
                    days,
                    secs: 0,
                    nanos: 0,
                })
            },
            probe: t("duration", "P50000D"),
        },
    ]
}

fn build(kinds: &[TempKind]) -> Graph {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut b = Builder::default();
    for i in 0..N {
        // Shuffle the index so stored order isn't already sorted.
        let j = (rng.next() % N as u64) as usize;
        let mut props: Vec<(String, Value)> =
            vec![("age".to_string(), Value::Num((j % 80) as f64))];
        for k in kinds {
            props.push((k.key.to_string(), Value::Temporal((k.make)(j))));
        }
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["Person".to_string()],
            props,
        ));
    }
    b.finalize()
}

/// Run `q` (with a single `$p` temporal param) `iters` times; return avg µs.
fn bench(g: &mut Graph, q: &str, probe: &Temporal, iters: u32) -> f64 {
    let plan = prepare(q).unwrap();
    let mut p = Params::new();
    p.insert("p".to_string(), Val::Temporal(*probe));
    let _ = plan.execute(g, &p).unwrap(); // warm up
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = plan.execute(g, &p).unwrap();
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    let ks = kinds();

    let rss0 = rss_kib();
    let t0 = Instant::now();
    let mut g = build(&ks);
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let rss1 = rss_kib();
    eprintln!(
        "built {} vertices ({} temporal cols) in {:.1} ms; RSS +{} MiB ({:.1} B/vertex across {} cols)\n",
        g.vertex_count(),
        ks.len(),
        build_ms,
        (rss1 - rss0) / 1024,
        (rss1 - rss0) as f64 * 1024.0 / (N * ks.len()) as f64,
        ks.len(),
    );

    // Per-type memory: packed column bytes vs the Mixed-equivalent (the real
    // 10× / 1.25× axis — distinct from the speed table below).
    println!(
        "{:<14} {:>10} {:>10} {:>8}",
        "type", "packed", "mixed", "ratio"
    );
    println!("{}", "-".repeat(44));
    for k in &ks {
        if let Some((packed, mixed)) = g.vertex_prop_bytes(k.key) {
            println!(
                "{:<14} {:>7} B/v {:>7} B/v {:>7.2}×",
                k.key,
                packed / N,
                mixed / N,
                mixed as f64 / packed as f64,
            );
        }
    }
    println!();

    // Per (type, op): filter-count, ORDER BY (no limit), project, min/max.
    let ops: &[(&str, &str, u32)] = &[
        (
            "filter>p count",
            "MATCH (n:Person) WHERE n.{K} > $p RETURN count(*) AS c",
            50,
        ),
        (
            "order by K",
            "MATCH (n:Person) RETURN n.{K} ORDER BY n.{K}",
            30,
        ),
        (
            "top-k 20",
            "MATCH (n:Person) RETURN n.{K} ORDER BY n.{K} LIMIT 20",
            100,
        ),
        ("project K", "MATCH (n:Person) RETURN n.{K}", 50),
        (
            "min/max K",
            "MATCH (n:Person) RETURN min(n.{K}) AS lo, max(n.{K}) AS hi",
            50,
        ),
    ];

    println!("{:<14} {:<16} {:>12}", "type", "op", "avg");
    println!("{}", "-".repeat(44));
    for k in &ks {
        for (label, tmpl, iters) in ops {
            let q = tmpl.replace("{K}", k.key);
            let us = bench(&mut g, &q, &k.probe, *iters);
            let pretty = if us >= 1000.0 {
                format!("{:.2} ms", us / 1000.0)
            } else {
                format!("{us:.1} us")
            };
            println!("{:<14} {:<16} {pretty:>12}", k.key, label);
        }
        println!();
    }
}
