//! Differential + edge-case tests for the de-boxed temporal columns and their
//! vectorized fast paths (`gather_temporal`, `temporal_cmp_vec`, `temporal_minmax`).
//!
//! The core safety net is a **scalar-vs-vectorized differential**: the same query
//! is run with the vectorized scan forced ON and forced OFF (via
//! `with_vec_override`), and the two results must agree. The scalar path is the
//! trusted oracle, so any divergence introduced by a typed temporal fast path is
//! caught — for every temporal kind, every comparison operator, ORDER BY spec
//! (asc/desc, nulls first/last, multi-key, skip/limit), and min/max. Row order is
//! only asserted where the query pins it (ORDER BY / aggregate / single row);
//! otherwise results are compared as multisets (unordered is unspecified).

use super::eval::{with_vec_override, Params, Val};
use super::parse;
use crate::graph::{Builder, EdgeRec, Graph, NodeRec, Value};
use crate::temporal::{Date, DateTime, Duration, Temporal, Time, ZonedDateTime, ZonedTime};

// --- tiny deterministic RNG (xorshift) --------------------------------------

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn chance(&mut self, num: u32, den: u32) -> bool {
        (self.next_u64() % den as u64) < num as u64
    }
}

// --- temporal value construction (spread by index) --------------------------

/// The six temporal kinds under test: a stored-key name, and a builder that maps
/// an index to a spread-out value of that kind.
struct Kind {
    key: &'static str,
    make: fn(i64) -> Temporal,
}

const KINDS: &[Kind] = &[
    Kind {
        key: "k_date",
        make: |i| {
            Temporal::Date(Date {
                days: (i % 4000) as i32 - 2000,
            })
        },
    },
    Kind {
        key: "k_time",
        make: |i| {
            Temporal::Time(Time {
                secs: (i as u32 * 37) % 86_400,
                nanos: (i as u32 % 3) * 1000,
            })
        },
    },
    Kind {
        key: "k_datetime",
        make: |i| {
            Temporal::DateTime(DateTime {
                secs: i * 3600 - 100_000,
                nanos: (i as u32 % 2) * 500,
            })
        },
    },
    Kind {
        key: "k_ztime",
        make: |i| {
            Temporal::ZonedTime(ZonedTime {
                secs: (i as u32 * 53) % 86_400,
                nanos: 0,
                offset: ((i % 5) as i16 - 2) * 60,
            })
        },
    },
    Kind {
        key: "k_zdatetime",
        make: |i| {
            Temporal::ZonedDateTime(ZonedDateTime {
                secs: i * 1800 - 50_000,
                nanos: 0,
                offset: ((i % 5) as i16 - 2) * 60,
            })
        },
    },
    Kind {
        key: "k_duration",
        make: |i| {
            Temporal::Duration(Duration {
                months: i % 7,
                days: (i * 3) % 100,
                secs: i * 61,
                nanos: 0,
            })
        },
    },
];

fn kind(key: &str) -> &'static Kind {
    KINDS.iter().find(|k| k.key == key).unwrap()
}

/// A graph of `n` `:T` nodes; each carries every temporal kind, but a fraction of
/// slots are absent (null) and values repeat (index mod a modulus) so ties, nulls,
/// and duplicates are all exercised. One `:T`-`:T` edge per node carries a `date`
/// temporal edge property (to cover edge columns too).
fn build(n: usize, seed: u64) -> Graph {
    let mut rng = Rng::new(seed);
    let mut b = Builder::default();
    for i in 0..n {
        let mut props: Vec<(String, Value)> = vec![("g".to_string(), Value::Num((i % 4) as f64))];
        for k in KINDS {
            // ~20% absent; values repeat within a small window for ties.
            if rng.chance(4, 5) {
                let idx = (i % 53) as i64;
                props.push((k.key.to_string(), Value::Temporal((k.make)(idx))));
            }
        }
        b.nodes.push(NodeRec::owned(
            format!("p{i}"),
            vec!["T".to_string()],
            props,
        ));
    }
    for i in 0..n {
        b.edges.push(EdgeRec::owned(
            format!("p{i}"),
            format!("p{}", (i + 1) % n),
            "R".to_string(),
            vec![(
                "ed".to_string(),
                Value::Temporal(Temporal::Date(Date {
                    days: (i % 300) as i32,
                })),
            )],
            None,
        ));
    }
    b.finalize()
}

// --- differential harness ----------------------------------------------------

fn run(g: &mut Graph, q: &str, p: &Params, vec: bool) -> Vec<Vec<Value>> {
    with_vec_override(vec, || {
        parse(q)
            .unwrap_or_else(|e| panic!("parse: {e}\n{q}"))
            .execute(g, p)
            .unwrap_or_else(|e| panic!("exec: {e}\n{q}"))
            .rows()
            .map(|r| r.to_vec())
            .collect()
    })
}

/// Canonical string for a value (so unordered results compare as multisets).
fn vkey(v: &Value) -> String {
    match v {
        Value::Null => "∅".to_string(),
        Value::Bool(b) => format!("b{b}"),
        Value::Num(n) => format!("n{n}"),
        Value::Str(s) => format!("s{s}"),
        Value::Temporal(t) => format!("t{}", t.format()),
        Value::List(xs) => format!("[{}]", xs.iter().map(vkey).collect::<Vec<_>>().join(",")),
        Value::Map(_) => "m".to_string(),
    }
}

fn rowkey(r: &[Value]) -> String {
    r.iter().map(vkey).collect::<Vec<_>>().join("|")
}

/// Vectorized == scalar, with row order asserted (ORDER BY / aggregate / count).
fn diff_ordered(g: &mut Graph, q: &str, p: &Params) -> Vec<Vec<Value>> {
    let on = run(g, q, p, true);
    let off = run(g, q, p, false);
    assert_eq!(
        on.iter().map(|r| rowkey(r)).collect::<Vec<_>>(),
        off.iter().map(|r| rowkey(r)).collect::<Vec<_>>(),
        "vectorized ≠ scalar (ordered):\n{q}"
    );
    on
}

/// Vectorized == scalar as a multiset (no ORDER BY — row order is unspecified).
fn diff_unordered(g: &mut Graph, q: &str, p: &Params) {
    let mut on: Vec<String> = run(g, q, p, true).iter().map(|r| rowkey(r)).collect();
    let mut off: Vec<String> = run(g, q, p, false).iter().map(|r| rowkey(r)).collect();
    on.sort();
    off.sort();
    assert_eq!(on, off, "vectorized ≠ scalar (multiset):\n{q}");
}

fn param(t: Temporal) -> Params {
    let mut p = Params::new();
    p.insert("p".to_string(), Val::Temporal(t));
    p
}

// --- fuzz: filter (temporal col <op> scalar), all kinds/ops/orders ----------

const OPS: &[&str] = &["<", "<=", ">", ">=", "=", "<>"];

#[test]
fn fuzz_temporal_filter_vec_eq_scalar() {
    for seed in 0..250u64 {
        let mut rng = Rng::new(0xF117_0000 + seed);
        let mut g = build(180, seed);
        let k = &KINDS[rng.below(KINDS.len())];
        let op = OPS[rng.below(OPS.len())];
        let probe = (k.make)((rng.below(53)) as i64);
        let p = param(probe);
        // Both operand orders; count is order-independent so `diff_ordered` is fine.
        let q1 = format!("MATCH (n:T) WHERE n.{} {op} $p RETURN count(*) AS c", k.key);
        let q2 = format!("MATCH (n:T) WHERE $p {op} n.{} RETURN count(*) AS c", k.key);
        diff_ordered(&mut g, &q1, &p);
        diff_ordered(&mut g, &q2, &p);
        // Also project the surviving rows (multiset) to exercise the mask + gather.
        let q3 = format!(
            "MATCH (n:T) WHERE n.{k} {op} $p RETURN element_id(n) AS id",
            k = k.key
        );
        diff_unordered(&mut g, &q3, &p);
    }
}

// --- fuzz: ORDER BY (the target of the deferred typed sort) ------------------

#[test]
fn fuzz_temporal_order_by_vec_eq_scalar() {
    for seed in 0..250u64 {
        let mut rng = Rng::new(0x0DE1_0000 + seed);
        let mut g = build(180, seed);
        let k = &KINDS[rng.below(KINDS.len())];
        let dir = if rng.chance(1, 2) { "ASC" } else { "DESC" };
        // element_id tiebreak → a deterministic total order across both engines.
        let base = format!(
            "MATCH (n:T) RETURN element_id(n) AS id, n.{k} AS v ORDER BY n.{k} {dir}, element_id(n)",
            k = k.key
        );
        let full = diff_ordered(&mut g, &base, &Params::new());
        // SKIP/LIMIT window must equal the slice of the full order.
        let flen = full.len();
        let s = rng.below(flen + 2);
        let l = rng.below(flen + 2);
        let paged = format!("{base} SKIP {s} LIMIT {l}");
        let got = diff_ordered(&mut g, &paged, &Params::new());
        let end = (s + l).min(flen);
        let want = if s >= flen { &[][..] } else { &full[s..end] };
        assert_eq!(
            got.iter().map(|r| rowkey(r)).collect::<Vec<_>>(),
            want.iter().map(|r| rowkey(r)).collect::<Vec<_>>(),
            "paged slice ≠ full[{s}..{end}]:\n{paged}"
        );
    }
}

#[test]
fn fuzz_temporal_order_by_nulls_and_multikey() {
    for seed in 0..150u64 {
        let mut rng = Rng::new(0x0DE2_0000 + seed);
        let mut g = build(160, seed);
        let k = &KINDS[rng.below(KINDS.len())];
        let dir = if rng.chance(1, 2) { "ASC" } else { "DESC" };
        let nulls = match rng.below(3) {
            0 => "",
            1 => " NULLS FIRST",
            _ => " NULLS LAST",
        };
        // Multi-key: temporal primary (has nulls), numeric secondary, id tiebreak.
        let q = format!(
            "MATCH (n:T) RETURN element_id(n) AS id ORDER BY n.{k} {dir}{nulls}, n.g ASC, element_id(n)",
            k = k.key
        );
        diff_ordered(&mut g, &q, &Params::new());
    }
}

// --- fuzz: min / max --------------------------------------------------------

#[test]
fn fuzz_temporal_minmax_vec_eq_scalar() {
    for seed in 0..200u64 {
        let mut rng = Rng::new(0x00A0_0000 + seed);
        let mut g = build(200, seed);
        let k = &KINDS[rng.below(KINDS.len())];
        for f in ["min", "max"] {
            let q = format!("MATCH (n:T) RETURN {f}(n.{k}) AS m", k = k.key);
            diff_ordered(&mut g, &q, &Params::new());
        }
        // grouped min/max (currently scalar) must still match itself across engines.
        let q = format!(
            "MATCH (n:T) RETURN n.g AS g, min(n.{k}) AS lo, max(n.{k}) AS hi ORDER BY n.g",
            k = k.key
        );
        diff_ordered(&mut g, &q, &Params::new());
    }
}

// --- fuzz: projection (gather) ----------------------------------------------

#[test]
fn fuzz_temporal_projection_vec_eq_scalar() {
    for seed in 0..150u64 {
        let mut rng = Rng::new(0x9401_0000 + seed);
        let mut g = build(160, seed);
        let k = &KINDS[rng.below(KINDS.len())];
        let q = format!(
            "MATCH (n:T) RETURN element_id(n) AS id, n.{k} AS v",
            k = k.key
        );
        diff_unordered(&mut g, &q, &Params::new());
    }
}

// --- targeted: every kind, every op, both engines agree ---------------------

#[test]
fn every_kind_every_op_vec_eq_scalar() {
    let mut g = build(120, 7);
    for k in KINDS {
        let probe = (k.make)(20);
        let p = param(probe);
        for op in OPS {
            let q = format!(
                "MATCH (n:T) WHERE n.{k} {op} $p RETURN count(*) AS c",
                k = k.key
            );
            diff_ordered(&mut g, &q, &p);
        }
        for f in ["min", "max"] {
            let q = format!("MATCH (n:T) RETURN {f}(n.{k}) AS m", k = k.key);
            diff_ordered(&mut g, &q, &Params::new());
        }
        let q = format!(
            "MATCH (n:T) RETURN n.{k} AS v ORDER BY n.{k} ASC, element_id(n)",
            k = k.key
        );
        diff_ordered(&mut g, &q, &Params::new());
    }
}

// --- targeted: duration is relationally unordered but has a total order ------

#[test]
fn duration_unordered_in_predicate_total_in_sort_and_minmax() {
    let mut g = build(120, 11);
    let p = param((kind("k_duration").make)(5));
    // Every relational compare is UNKNOWN → 0 rows, both engines.
    for op in ["<", "<=", ">", ">="] {
        let q = format!("MATCH (n:T) WHERE n.k_duration {op} $p RETURN count(*) AS c");
        let rows = diff_ordered(&mut g, &q, &p);
        assert_eq!(rows[0][0], Value::Num(0.0), "duration {op} should be empty");
    }
    // But ORDER BY and min/max use the deterministic total order — must agree.
    diff_ordered(
        &mut g,
        "MATCH (n:T) RETURN n.k_duration AS v ORDER BY n.k_duration ASC, element_id(n)",
        &Params::new(),
    );
    diff_ordered(
        &mut g,
        "MATCH (n:T) RETURN min(n.k_duration) AS m",
        &Params::new(),
    );
    diff_ordered(
        &mut g,
        "MATCH (n:T) RETURN max(n.k_duration) AS m",
        &Params::new(),
    );
}

// --- targeted: cross-kind compare is UNKNOWN (empty) both engines -----------

#[test]
fn cross_kind_compare_is_unknown_both_engines() {
    let mut g = build(80, 13);
    // date column vs a datetime param → cross-kind → UNKNOWN → 0 rows.
    let p = param((kind("k_datetime").make)(5));
    for op in ["<", ">", "<=", ">="] {
        let q = format!("MATCH (n:T) WHERE n.k_date {op} $p RETURN count(*) AS c");
        let rows = diff_ordered(&mut g, &q, &p);
        assert_eq!(rows[0][0], Value::Num(0.0));
    }
    // Eq across kinds is defined (always false) → also 0, and `<>` → all present.
    let q = "MATCH (n:T) WHERE n.k_date = $p RETURN count(*) AS c";
    assert_eq!(diff_ordered(&mut g, q, &p)[0][0], Value::Num(0.0));
}

// --- targeted: literal and $param scalars agree (date/datetime have literals) -

#[test]
fn temporal_filter_literal_matches_param() {
    let mut g = build(140, 17);
    // A literal DATE and the equivalent $param must give identical results, and
    // both must match across engines.
    let d = Temporal::Date(Date::parse("2020-06-15").unwrap());
    let p = param(d);
    let lit = "MATCH (n:T) WHERE n.k_date > DATE '2020-06-15' RETURN count(*) AS c";
    let par = "MATCH (n:T) WHERE n.k_date > $p RETURN count(*) AS c";
    let a = diff_ordered(&mut g, lit, &Params::new());
    let b = diff_ordered(&mut g, par, &p);
    assert_eq!(a, b, "literal vs param disagree");
}

// --- targeted: edge temporal columns ----------------------------------------

#[test]
fn edge_temporal_column_filter_order_minmax() {
    let mut g = build(150, 19);
    let p = param(Temporal::Date(Date { days: 150 }));
    diff_ordered(
        &mut g,
        "MATCH (a:T)-[r:R]->(b) WHERE r.ed > $p RETURN count(*) AS c",
        &p,
    );
    diff_ordered(
        &mut g,
        "MATCH (a:T)-[r:R]->(b) RETURN min(r.ed) AS lo, max(r.ed) AS hi",
        &Params::new(),
    );
    diff_unordered(
        &mut g,
        "MATCH (a:T)-[r:R]->(b) RETURN r.ed AS v",
        &Params::new(),
    );
}

// --- targeted: all-null temporal column (min/max → null, filter → empty) -----

#[test]
fn all_null_temporal_column() {
    // No node carries `k_missing`; the column is absent everywhere.
    let mut g = build(60, 23);
    let p = param(Temporal::Date(Date { days: 0 }));
    let q = "MATCH (n:T) WHERE n.k_missing > $p RETURN count(*) AS c";
    assert_eq!(diff_ordered(&mut g, q, &p)[0][0], Value::Num(0.0));
    let m = diff_ordered(
        &mut g,
        "MATCH (n:T) RETURN min(n.k_missing) AS m",
        &Params::new(),
    );
    assert_eq!(m[0][0], Value::Null);
}

// --- storage: de-box round-trips every kind (via NDJSON encode/decode) -------

#[test]
fn deboxed_columns_round_trip_all_kinds() {
    let g = build(90, 29);
    // Encode → decode → re-encode: the wire form must be stable through the packed
    // columns (the codec reads them via `value_id`).
    let enc1 = crate::ndjson::encode(&g);
    let g2 = crate::ndjson::decode(&enc1).unwrap();
    let enc2 = crate::ndjson::encode(&g2);
    assert_eq!(enc1, enc2, "temporal columns changed the NDJSON round-trip");
}

// --- storage: type promotion to Mixed still reads correctly ------------------

#[test]
fn temporal_key_promotes_to_mixed_on_conflict() {
    // A key that first sees a Date then a String promotes its column to Mixed; both
    // values must still read back (and a temporal-subkind conflict promotes too).
    let lines = [
        r#"{"type":"node","id":"a","labels":["T"],"properties":{"x":{"@date":"2020-01-01"}}}"#,
        r#"{"type":"node","id":"b","labels":["T"],"properties":{"x":"hello"}}"#,
        r#"{"type":"node","id":"c","labels":["T"],"properties":{"y":{"@date":"2020-01-01"}}}"#,
        r#"{"type":"node","id":"d","labels":["T"],"properties":{"y":{"@datetime":"2020-01-01T00:00:00"}}}"#,
    ];
    let mut g = crate::ndjson::decode(&lines.join("\n")).unwrap();
    // The Date row and the String row are both retrievable (Mixed column).
    let got = run(
        &mut g,
        "MATCH (n:T) WHERE n.x IS NOT NULL RETURN element_id(n) AS id, n.x AS v",
        &Params::new(),
        false,
    );
    assert_eq!(got.len(), 2);
    // Date + DateTime under one key → Mixed; both read back with their own kind.
    let ys = run(
        &mut g,
        "MATCH (n:T) WHERE n.y IS NOT NULL RETURN n.y AS v",
        &Params::new(),
        false,
    );
    assert_eq!(ys.len(), 2);
}
