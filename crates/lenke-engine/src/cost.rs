//! Cost / cardinality estimation — the shared "brain" that fast-path routing
//! decisions consult, instead of each hand-coding a structural gate.
//!
//! In memory, the store IS the statistics: we COUNT exact base cardinalities
//! (label buckets, hash-index buckets, dict distincts, degree) in O(1) rather than
//! sampling histograms the way a disk engine must — so the estimation error that
//! dominates a disk planner simply does not exist for those. Only DERIVED
//! cardinalities (after a traversal / filter / join, where an exact count would
//! cost as much as the operation) are ESTIMATED, via cheap fan-out and selectivity
//! propagation.
//!
//! Crucially, every routing choice this informs is between BYTE-IDENTICAL physical
//! operators (materialize vs. fold/stream), so a wrong estimate only trades time —
//! it can never change the answer. That makes a lightweight, forgiving model the
//! right amount of machinery here.

use crate::ir::{CompareOp, Expr, Plan};
use crate::store::Store;

// Default selectivities for the ESTIMATED (non-exact) cases. Deliberately coarse —
// the fuzzer calibrates, and byte-identity means a miss only costs time.
const EQ_SEL: f64 = 0.10; // `prop = lit` with no index
const RANGE_SEL: f64 = 0.33; // `prop < / <= / > / >=  lit`
const STR_SEARCH_SEL: f64 = 0.25; // STARTS WITH / ENDS WITH / CONTAINS
const DEFAULT_SEL: f64 = 0.33; // anything not specifically modelled

/// A cardinality estimate for a plan's output row count.
#[derive(Clone, Copy, Debug)]
pub struct Card {
    /// Estimated output rows.
    pub rows: f64,
    /// True when the estimate came PURELY from exact base counts — no fan-out or
    /// selectivity guess entered it. A caller may treat an `exact` estimate as a
    /// hard number (e.g. an index-seek's bucket size) rather than a hint.
    pub exact: bool,
}

impl Card {
    fn exact(rows: usize) -> Self {
        Self {
            rows: rows as f64,
            exact: true,
        }
    }
    fn approx(rows: f64) -> Self {
        Self {
            rows: rows.max(0.0),
            exact: false,
        }
    }
    fn scale(self, factor: f64, still_exact: bool) -> Self {
        Self {
            rows: (self.rows * factor).max(0.0),
            exact: self.exact && still_exact,
        }
    }
}

/// The graph's average out-degree — an exact base fact (`edges / live nodes`),
/// used as the per-hop fan-out. `Both`-direction hops see roughly twice this.
fn avg_degree(store: &Store) -> f64 {
    let n = store.live_node_count();
    if n == 0 {
        0.0
    } else {
        store.edge_count() as f64 / n as f64
    }
}

/// Estimate `plan`'s output cardinality, bottom-up.
#[must_use]
pub fn estimate(plan: &Plan, store: &Store) -> Card {
    match plan {
        // --- exact base cardinalities -------------------------------------------
        Plan::Scan { label: Some(l) } => Card::exact(store.nodes_with_label(l).len()),
        Plan::Scan { label: None } => Card::exact(store.live_node_count()),
        Plan::EdgeScan => Card::exact(store.edge_count()),
        Plan::Row => Card::exact(1),
        // A transaction-control command yields no rows.
        Plan::TxControl { .. } => Card::exact(0),
        // A list unwind multiplies the input by the (unknown) list length — treat it
        // as a small fan-out, like a low-degree expansion.
        Plan::Unwind { input, .. } => estimate(input, store).scale(4.0, false),
        Plan::NodeSeed { ext_ids } => Card::exact(ext_ids.len()),
        Plan::EdgeSeed { ext_ids } => Card::exact(ext_ids.len()),
        Plan::IndexSeek { key, value, .. } => store.index_bucket_len(key, value).map_or_else(
            || Card::approx(store.live_node_count() as f64 * EQ_SEL),
            Card::exact,
        ),
        Plan::RangeSeek { label, .. } => {
            Card::approx(store.nodes_with_label(label).len() as f64 * RANGE_SEL)
        }

        // --- selectivity / fan-out propagation ----------------------------------
        Plan::Filter { input, pred } => {
            let (sel, sel_exact) = selectivity(pred, store);
            estimate(input, store).scale(sel, sel_exact)
        }
        Plan::Expand {
            input, edge_label, ..
        }
        | Plan::IntervalExpand {
            input, edge_label, ..
        } => {
            // One hop multiplies by the fan-out of the matching edge type. Never
            // exact (per-node degree varies), and an unknown edge type yields none.
            let fanout = if unknown_edge(edge_label, store) {
                0.0
            } else {
                avg_degree(store)
            };
            estimate(input, store).scale(fanout, false)
        }
        Plan::OptionalExpand {
            input, edge_label, ..
        } => {
            // A left-outer hop keeps at least the source row (a miss → NULL), so the
            // multiplier is at least 1.
            let fanout = if unknown_edge(edge_label, store) {
                1.0
            } else {
                avg_degree(store).max(1.0)
            };
            estimate(input, store).scale(fanout, false)
        }
        Plan::VarLength {
            input,
            edge_label,
            min,
            max,
            ..
        }
        | Plan::RepeatGroup {
            input,
            edge_label,
            min,
            max,
            ..
        } => {
            let d = if unknown_edge(edge_label, store) {
                0.0
            } else {
                avg_degree(store)
            };
            // Σ_{h=min..=max} d^h — the number of length-h walks per source, summed
            // over the quantifier range (h=0 contributes the source itself).
            let mut mult = 0.0;
            let mut dh = d.powi(i32::try_from(*min).unwrap_or(0));
            for _ in *min..=*max {
                mult += dh;
                dh *= d;
            }
            estimate(input, store).scale(mult, false)
        }
        // A nested group enumerates outer×inner repetition-decompositions — a large
        // fan-out; approximate as a degree-driven blow-up over the outer×inner range.
        Plan::NestedGroup { input, max, .. } => {
            let d = avg_degree(store).max(1.0);
            estimate(input, store).scale(d.powi(i32::try_from(*max).unwrap_or(1).max(1)), false)
        }
        Plan::ShortestPath { input, .. } => {
            // ANY-shortest emits ~one row per reachable target; without running a
            // BFS we cap at the whole graph — a deliberate over-estimate that routes
            // such queries toward the bounded-memory path.
            estimate(input, store).scale(store.live_node_count() as f64, false)
        }

        // --- reducing / bounding operators --------------------------------------
        Plan::Aggregate { input, keys, .. } => {
            if keys.is_empty() {
                Card::exact(1) // scalar aggregate: one row
            } else {
                // Grouped: one row per distinct key group. An exact dict distinct
                // count when the (single) key is a bare dict-encoded property; else
                // a fraction of the input.
                group_estimate(keys, input, store)
            }
        }
        Plan::GroupToMap { .. } => Card::exact(1), // folds all groups into one Map row
        Plan::AlgoAnnotate { input, .. } => estimate(input, store), // pass-through, +1 column
        Plan::Tree { .. } => Card::exact(1),       // folds all paths into one nested Map row
        Plan::MapSlot { input, .. } => estimate(input, store), // pass-through (append/overwrite a slot)
        Plan::Enumerate { input, .. } => estimate(input, store), // one list per row
        Plan::Sample { input, n } => {
            let e = estimate(input, store);
            Card::approx(e.rows.min(*n as f64))
        }
        // Out/In keep one row per edge; Both fans out to two.
        Plan::EdgeVertex { input, which, .. } => {
            let f = if matches!(which, crate::ir::Dir::Both) {
                2.0
            } else {
                1.0
            };
            estimate(input, store).scale(f, false)
        }
        Plan::Subgraph { .. } => Card::exact(1),
        Plan::ShortestPathEnum { input, .. } => estimate(input, store).scale(4.0, false),
        Plan::Distinct { input } => {
            // At most the input; without a cheap distinct count assume heavy overlap.
            estimate(input, store).scale(0.5, false)
        }
        Plan::DistinctBy { input, .. } => estimate(input, store).scale(0.5, false),
        Plan::OrderPage {
            input, skip, limit, ..
        } => {
            let inp = estimate(input, store);
            match limit {
                Some(l) => {
                    let cap = (skip.unwrap_or(0) + l) as f64;
                    Card {
                        rows: inp.rows.min(cap),
                        exact: inp.exact && inp.rows <= cap,
                    }
                }
                None => inp,
            }
        }
        Plan::Tail { input, n } => {
            let inp = estimate(input, store);
            Card {
                rows: inp.rows.min(*n as f64),
                exact: inp.exact,
            }
        }

        // A left-outer correlated scan yields >= 1 row per input row (the NULL pad).
        Plan::OptionalScan { input, .. } => estimate(input, store).scale(2.0, false),
        // A leading OPTIONAL MATCH yields at least one row (the null pad when empty).
        Plan::NullPadIfEmpty { input, .. } => {
            let inp = estimate(input, store);
            Card {
                rows: inp.rows.max(1.0),
                exact: inp.exact,
            }
        }

        // --- row-preserving / structural ----------------------------------------
        Plan::Project { input, .. }
        | Plan::SortLocal { input, .. }
        | Plan::Update { input, .. } => estimate(input, store),
        Plan::Union { left, right, .. } => {
            let (a, b) = (estimate(left, store), estimate(right, store));
            Card::approx(a.rows + b.rows)
        }
        Plan::Branch { input, bodies } => {
            estimate(input, store).scale(bodies.len().max(1) as f64, false)
        }
        Plan::Join { left, right, .. } => {
            // A hash join on shared node identity: rows ≈ left × right / max(dim), a
            // crude but adequate cap for routing.
            let (a, b) = (estimate(left, store), estimate(right, store));
            let denom = a.rows.max(b.rows).max(1.0);
            Card::approx(a.rows * b.rows / denom)
        }
        Plan::CallProcedure { .. } => Card::approx(store.live_node_count() as f64),
        Plan::CallInline { input, .. } => {
            // A correlated lateral subquery: one row expands to ~fan-out rows.
            estimate(input, store).scale(avg_degree(store).max(1.0), false)
        }

        // Writes / seeds with no row output worth costing. `InsertReturn` projects
        // a single created-node row, so its output is ~1 too.
        Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::AddEdge { .. }
        | Plan::Merge { .. } => Card::approx(1.0),
    }
}

/// Estimate the number of GROUP rows for a grouped aggregate. When the sole key is
/// a bare property that is dict-encoded, `dict.len()` is the EXACT distinct count —
/// the very number a disk planner would sample-and-miss. Otherwise fall back to a
/// fraction of the input (capped by it).
fn group_estimate(keys: &[(String, Expr)], input: &Plan, store: &Store) -> Card {
    let inp = estimate(input, store);
    if let [(_, Expr::Prop { key, .. })] = keys {
        if let Some(distinct) = store.distinct_count(key) {
            return Card {
                rows: (distinct as f64).min(inp.rows.max(1.0)),
                exact: false, // exact distinct of the column, but the input may filter
            };
        }
    }
    // Unknown grouping cardinality: assume moderate collapse, never above the input.
    Card::approx((inp.rows * 0.5).min(inp.rows).max(1.0))
}

/// True when the edge-type constraint is TYPED (non-empty) but NONE of its names
/// resolve to a known edge type — the traversal then matches nothing. An empty list
/// is "any type" (not unknown).
fn unknown_edge(edge_label: &[String], store: &Store) -> bool {
    !edge_label.is_empty() && edge_label.iter().all(|name| store.etype_id(name).is_none())
}

/// Estimate a predicate's selectivity in `[0, 1]` and whether it is exact. Uses the
/// EXACT index-bucket ratio for an indexed `=`; coarse constants otherwise. AND/OR/
/// NOT compose under independence (good enough for routing).
fn selectivity(pred: &Expr, store: &Store) -> (f64, bool) {
    match pred {
        Expr::And(a, b) => {
            let (sa, ea) = selectivity(a, store);
            let (sb, eb) = selectivity(b, store);
            (sa * sb, ea && eb)
        }
        Expr::Or(a, b) => {
            let (sa, _) = selectivity(a, store);
            let (sb, _) = selectivity(b, store);
            ((sa + sb - sa * sb).clamp(0.0, 1.0), false)
        }
        Expr::Not(x) => {
            let (s, _) = selectivity(x, store);
            (1.0 - s, false)
        }
        Expr::Compare { op, left, right } => compare_selectivity(*op, left, right, store),
        Expr::In { haystack, .. } => {
            let n = if let Expr::List { items } = haystack.as_ref() {
                items.len()
            } else {
                3
            };
            ((n as f64 * EQ_SEL).min(1.0), false)
        }
        Expr::Call { name, .. }
            if matches!(name.as_str(), "starts_with" | "ends_with" | "contains") =>
        {
            (STR_SEARCH_SEL, false)
        }
        Expr::PropertyExists { .. } | Expr::IsNull { .. } => (0.9, false),
        _ => (DEFAULT_SEL, false),
    }
}

fn compare_selectivity(op: CompareOp, left: &Expr, right: &Expr, store: &Store) -> (f64, bool) {
    // Normalize to `prop <op> literal` (or its mirror).
    let (key, lit, op) = match (left, right) {
        (Expr::Prop { key, .. }, Expr::Lit(v)) => (key, v, op),
        (Expr::Lit(v), Expr::Prop { key, .. }) => (key, v, mirror(op)),
        // prop-vs-prop / expr compares: not modelled precisely.
        _ => return (DEFAULT_SEL, false),
    };
    match op {
        CompareOp::Eq => {
            // An indexed `=` gives the EXACT selectivity: bucket / live nodes.
            if let Some(bucket) = store.index_bucket_len(key, lit) {
                let n = store.live_node_count().max(1) as f64;
                (bucket as f64 / n, true)
            } else if let (Some(d), false) = (store.distinct_count(key), lit.is_null()) {
                // A dict column's `=` hits ~one of its distinct values (uniform).
                (1.0 / d.max(1) as f64, false)
            } else {
                (EQ_SEL, false)
            }
        }
        CompareOp::Ne => {
            let (s, _) = compare_selectivity(CompareOp::Eq, left, right, store);
            (1.0 - s, false)
        }
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => (RANGE_SEL, false),
    }
}

fn mirror(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::Ne => CompareOp::Ne,
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Ge => CompareOp::Le,
    }
}

/// The routing budget: the estimated intermediate row count above which a
/// bounded-memory operator (fold / stream) is preferred over materializing the
/// whole batch. Resource-aware — the memory ceiling scales with available RAM, so
/// a smaller box streams sooner — but also bounded by a fixed TIME crossover (the
/// point where materialize's allocate-plus-second-pass exceeds an incremental
/// fold, roughly size- not RAM-dependent). Env-overridable for calibration.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub materialize_rows: f64,
}

impl Budget {
    /// Roughly the bytes a materialized intermediate row costs (a few boxed columns).
    const BYTES_PER_ROW: f64 = 64.0;
    /// The fixed time-crossover: above this many rows, folding beats materializing
    /// regardless of RAM (calibrated on the perf-fuzzer; see `try_varlen_agg`).
    const TIME_CROSSOVER_ROWS: f64 = 300_000.0;

    /// The default budget: `min(time-crossover, memory-ceiling)`, so a large box
    /// uses the time crossover and a small box streams sooner to stay in memory.
    #[must_use]
    pub fn default_budget() -> Self {
        if let Ok(v) = std::env::var("PERF_MATERIALIZE_ROWS") {
            if let Ok(n) = v.parse::<f64>() {
                return Self {
                    materialize_rows: n,
                };
            }
        }
        let mem_ceiling = (available_ram_bytes() as f64 * 0.10) / Self::BYTES_PER_ROW;
        Self {
            materialize_rows: Self::TIME_CROSSOVER_ROWS.min(mem_ceiling),
        }
    }
}

/// Whether `input`'s intermediate is estimated large enough that a reducing
/// operator should produce it with BOUNDED MEMORY (fold / stream) rather than
/// materialize it. Byte-identity holds either way — this only trades time.
#[must_use]
pub fn prefer_bounded_memory(input: &Plan, store: &Store, budget: &Budget) -> bool {
    estimate(input, store).rows > budget.materialize_rows
}

/// Available RAM in bytes (Linux `MemAvailable`); a conservative 4 GiB default
/// where it cannot be read, so the memory ceiling never becomes accidentally tiny.
fn available_ram_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map_or(4 << 30, |kb| kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Dir, PathMode};
    use crate::store::{Builder, Store};
    use crate::value::Value;

    const CITIES: &[&str] = &["oslo", "bergen", "troms", "moss"];

    fn fixture(n: u32, deg: u32) -> Store {
        let mut b = Builder::default();
        for i in 0..n {
            let city = Value::Str(CITIES[(i % 4) as usize].into());
            if i % 3 == 0 {
                b.node(&["Person", "VIP"], &[("city", city)]);
            } else {
                b.node(&["Person"], &[("city", city)]);
            }
        }
        for i in 0..n {
            for d in 0..deg {
                b.edge(i, (i * 7 + d * 13 + 1) % n, "R");
            }
        }
        b.build()
    }

    fn scan(label: &str) -> Plan {
        Plan::Scan {
            label: Some(label.into()),
        }
    }

    #[test]
    fn base_scans_are_exact() {
        let st = fixture(3000, 4);
        let all = estimate(&scan("Person"), &st);
        assert!(all.exact);
        assert_eq!(all.rows as u32, 3000);
        // Every third node is a VIP.
        let vip = estimate(&scan("VIP"), &st);
        assert!(vip.exact);
        assert_eq!(vip.rows as u32, st.nodes_with_label("VIP").len() as u32);
    }

    #[test]
    fn expand_and_varlength_scale_by_fanout() {
        let (n, deg) = (3000u32, 4u32);
        let st = fixture(n, deg);
        // avg_degree = edges/nodes = deg. A 1-hop ≈ nodes × deg.
        let one = estimate(&scan("Person").expand(0, Dir::Out, &["R".to_string()]), &st);
        assert!(!one.exact);
        assert!((one.rows - f64::from(n * deg)).abs() < 1.0, "1-hop ≈ n*deg");
        // VarLength {1,2}: n × (deg + deg²).
        let two = estimate(
            &scan("Person").var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail),
            &st,
        );
        let expected = f64::from(n) * (f64::from(deg) + f64::from(deg * deg));
        assert!((two.rows - expected).abs() < 2.0, "varlen ≈ n*(d+d²)");
        // The deep intermediate (n×(d+d²) = 60k) routes to bounded memory under a
        // budget below it.
        let budget = Budget {
            materialize_rows: 50_000.0,
        };
        assert!(prefer_bounded_memory(
            &scan("Person").var_length(0, Dir::Out, &["R".to_string()], 1, 2, PathMode::Trail),
            &st,
            &budget
        ));
    }

    #[test]
    fn grouped_aggregate_uses_exact_dict_distinct() {
        let st = fixture(3000, 4);
        // `city` has 4 distinct values → dict-encoded → exact group count.
        assert_eq!(st.distinct_count("city"), Some(4));
        let plan = scan("Person").aggregate(vec![("k".into(), prop(0, "city"))], vec![agg_count()]);
        let g = estimate(&plan, &st);
        assert_eq!(g.rows as u32, 4, "group count = dict distinct");
    }

    #[test]
    fn scalar_aggregate_is_one_row() {
        let st = fixture(1000, 4);
        let plan = scan("Person").aggregate(Vec::new(), vec![agg_count()]);
        let c = estimate(&plan, &st);
        assert!(c.exact);
        assert_eq!(c.rows as u32, 1);
    }

    #[test]
    fn limit_caps_the_estimate() {
        let st = fixture(3000, 4);
        let plan = scan("Person")
            .expand(0, Dir::Out, &["R".to_string()])
            .order_page(Vec::new(), None, Some(50));
        assert_eq!(estimate(&plan, &st).rows as u32, 50);
    }

    // --- tiny plan-builder helpers ---
    fn prop(slot: usize, key: &str) -> Expr {
        Expr::Prop {
            slot,
            key: key.into(),
        }
    }
    fn agg_count() -> crate::ir::Agg {
        crate::ir::Agg {
            func: crate::ir::AggFn::Count,
            arg: None,
            distinct: false,
            name: "c".into(),
            frac: None,
            null_on_empty: false,
        }
    }
}
