//! Execution: pull a batch up through the plan, then materialize the projection.
//!
//! Expression evaluation is columnar — `eval` produces a `Col` over the whole
//! batch, reading typed storage columns in bulk where it can. It calls the value
//! contract for every comparison and equality; it never restates those rules.
//! This is the lineage-FREE strategy; the lineage-preserving strategy for the
//! same operators lands with the operators (path/tags) that need it.

use std::collections::HashMap;

use crate::batch::{Batch, Col};
use crate::ir::{Agg, AggFn, CompareOp, Dir, Expr, Plan};
use crate::store::{Column, Store};
use crate::value::{self, Value};

/// A materialized result: column names and rows of values. `Value` intentionally
/// has no `PartialEq` (f64/NaN policy lives in the value contract, not a derive),
/// so compare results through `value::equals`/`cmp_total`, not `==`.
#[derive(Debug)]
pub struct Rows {
    pub names: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Run `plan` over `store`, returning materialized rows. A plan must end in a
/// `Project` (the only operator that names output columns); a plan without one
/// surfaces slot 0 under a single implicit column so partial plans stay runnable
/// in tests.
#[must_use]
pub fn run(plan: &Plan, store: &Store) -> Rows {
    match plan {
        Plan::Project { input, items } => {
            let batch = pull(input, store);
            let names = items.iter().map(|(n, _)| n.clone()).collect();
            let cols: Vec<Col> = items.iter().map(|(_, e)| eval(e, store, &batch)).collect();
            let n = batch.rows();
            let rows = (0..n)
                .map(|i| cols.iter().map(|c| c.value_at(i)).collect())
                .collect();
            Rows { names, rows }
        }
        // Aggregate names its own output columns (keys then aggregates), so it is
        // a valid terminal without a Project on top.
        Plan::Aggregate { keys, aggs, .. } => {
            let batch = pull(plan, store);
            let mut names: Vec<String> = keys.iter().map(|(n, _)| n.clone()).collect();
            names.extend(aggs.iter().map(|a| a.name.clone()));
            let n = batch.rows();
            let rows = (0..n)
                .map(|i| batch.slots.iter().map(|c| c.value_at(i)).collect())
                .collect();
            Rows { names, rows }
        }
        other => {
            let batch = pull(other, store);
            let n = batch.rows();
            let slot0 = batch.slot(0);
            let rows = (0..n).map(|i| vec![slot0.value_at(i)]).collect();
            Rows {
                names: vec!["_".to_string()],
                rows,
            }
        }
    }
}

/// Pull a batch up through a (non-terminal) plan node.
fn pull(plan: &Plan, store: &Store) -> Batch {
    match plan {
        Plan::Scan { label } => {
            let ids = match label {
                Some(l) => store.nodes_with_label(l).to_vec(),
                None => store.all_nodes(),
            };
            Batch::single(Col::Nodes(ids))
        }
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
        } => expand(
            &pull(input, store),
            store,
            *from,
            *dir,
            edge_label.as_deref(),
        ),
        Plan::Filter { input, pred } => {
            let batch = pull(input, store);
            let mask = eval(pred, store, &batch);
            let keep: Vec<usize> = match &mask {
                Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                other => (0..other.len())
                    .filter(|&i| other.value_at(i).is_true())
                    .collect(),
            };
            batch.gather(&keep)
        }
        Plan::Aggregate { input, keys, aggs } => aggregate(&pull(input, store), store, keys, aggs),
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } => order_page(&pull(input, store), store, keys, *skip, *limit),
        Plan::Project { input, .. } => pull(input, store),
    }
}

/// Sort the batch by `keys` (stable; ascending via `cmp_total`, descending its
/// reverse), then keep the window `[skip, skip+limit)`. Reorders every slot
/// together, so bound variables stay row-aligned.
fn order_page(
    batch: &Batch,
    store: &Store,
    keys: &[crate::ir::SortKey],
    skip: Option<usize>,
    limit: Option<usize>,
) -> Batch {
    let n = batch.rows();
    let mut idx: Vec<usize> = (0..n).collect();
    if !keys.is_empty() {
        let key_cols: Vec<Col> = keys.iter().map(|k| eval(&k.expr, store, batch)).collect();
        // Stable sort: equal keys keep input order, so the last key's ties fall
        // back to arrival order deterministically.
        idx.sort_by(|&a, &b| {
            for (kc, k) in key_cols.iter().zip(keys) {
                let ord = value::cmp_total(&kc.value_at(a), &kc.value_at(b));
                let ord = if k.descending { ord.reverse() } else { ord };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
    }
    let start = skip.unwrap_or(0).min(idx.len());
    let end = limit.map_or(idx.len(), |l| start.saturating_add(l).min(idx.len()));
    batch.gather(&idx[start..end])
}

/// Group `batch` by `keys` and compute `aggs` per group. Output slots are the key
/// columns (one value per group, taken from each group's first row) followed by
/// the aggregate columns. Group order is first-seen: a group's index is the
/// order its first row arrived, which is the order it is emitted.
fn aggregate(batch: &Batch, store: &Store, keys: &[(String, Expr)], aggs: &[Agg]) -> Batch {
    let n = batch.rows();
    let key_cols: Vec<Col> = keys.iter().map(|(_, e)| eval(e, store, batch)).collect();

    // Bucket rows into groups. With no keys the whole input is one group; with a
    // scalar aggregate over EMPTY input that one group is still emitted (SQL:
    // `count(*)` over nothing is 0, one row), which is why the empty case is
    // handled explicitly rather than falling out of the loop.
    let (rows_of, first_row): (Vec<Vec<usize>>, Vec<usize>) = if keys.is_empty() {
        (vec![(0..n).collect()], vec![0])
    } else {
        let mut of: HashMap<String, usize> = HashMap::new();
        let mut rows_of: Vec<Vec<usize>> = Vec::new();
        let mut first_row: Vec<usize> = Vec::new();
        for i in 0..n {
            let mut k = String::new();
            for kc in &key_cols {
                k.push_str(&value::group_key(&kc.value_at(i)));
                k.push('\u{1}'); // separator: distinct keys never run together
            }
            let g = *of.entry(k).or_insert_with(|| {
                rows_of.push(Vec::new());
                first_row.push(i);
                rows_of.len() - 1
            });
            rows_of[g].push(i);
        }
        (rows_of, first_row)
    };

    // Key output columns: each key's value at its group's first row.
    let mut slots: Vec<Col> = if keys.is_empty() {
        Vec::new()
    } else {
        key_cols.iter().map(|c| c.gather(&first_row)).collect()
    };

    // Aggregate output columns: fold each agg over each group's rows.
    for agg in aggs {
        let arg_col = agg.arg.as_ref().map(|e| eval(e, store, batch));
        let out: Vec<Value> = rows_of
            .iter()
            .map(|rows| fold_agg(agg, arg_col.as_ref(), rows))
            .collect();
        slots.push(Col::Gen(out));
    }

    Batch::of(slots)
}

/// Fold one aggregate over the rows of a single group. Null policy and ordering
/// come from the value contract; nothing here restates them.
fn fold_agg(agg: &Agg, arg_col: Option<&Col>, rows: &[usize]) -> Value {
    // `count(*)` — no argument — is the group's row count.
    if agg.func == AggFn::Count && agg.arg.is_none() {
        return Value::Num(rows.len() as f64);
    }
    let Some(col) = arg_col else {
        return Value::Null; // sum/min/max/avg with no argument is undefined
    };
    // The non-null argument values of this group.
    let vals: Vec<Value> = rows
        .iter()
        .map(|&i| col.value_at(i))
        .filter(|v| !v.is_null())
        .collect();

    match agg.func {
        AggFn::Count => {
            if agg.distinct {
                let mut seen = std::collections::HashSet::new();
                let n = vals
                    .iter()
                    .filter(|v| seen.insert(value::group_key(v)))
                    .count();
                Value::Num(n as f64)
            } else {
                Value::Num(vals.len() as f64)
            }
        }
        AggFn::Sum => sum(&vals),
        AggFn::Avg => match sum(&vals) {
            Value::Num(s) if !vals.is_empty() => Value::Num(s / vals.len() as f64),
            _ => Value::Null,
        },
        AggFn::Min => vals
            .into_iter()
            .min_by(value::cmp_total)
            .unwrap_or(Value::Null),
        AggFn::Max => vals
            .into_iter()
            .max_by(value::cmp_total)
            .unwrap_or(Value::Null),
    }
}

/// Sum the numeric values. An empty set sums to NULL (SQL), not 0 — only
/// `count` is 0 over nothing. A non-numeric non-null value makes the sum NULL
/// rather than coercing to NaN.
fn sum(vals: &[Value]) -> Value {
    if vals.is_empty() {
        return Value::Null;
    }
    let mut total = 0.0;
    for v in vals {
        match v {
            Value::Num(x) => total += x,
            _ => return Value::Null,
        }
    }
    Value::Num(total)
}

/// A hop: for each input row, expand the node in slot `from` along `dir`,
/// filtered by `edge_label`; emit one output row per matching neighbour with the
/// existing slots replicated and the neighbour appended as a new slot. This is
/// the bulk (lineage-free) strategy: `keep` records which input row each output
/// row came from, `nbrs` the landed node — the existing slots are gathered by
/// `keep`, so no per-row struct is built.
fn expand(batch: &Batch, store: &Store, from: usize, dir: Dir, edge_label: Option<&str>) -> Batch {
    // An empty expand still appends the landed slot (all rows dropped), so the
    // output has K+1 slots exactly as a successful expand would — a projection
    // referencing the new slot must not go out of bounds.
    let empty = || {
        let mut slots: Vec<Col> = batch.slots.iter().map(|_| Col::Nodes(vec![])).collect();
        slots.push(Col::Nodes(vec![]));
        Batch::of(slots)
    };
    // Resolve the edge label to an interned id up front; an unknown label matches
    // nothing (not everything).
    let want: Option<u32> = match edge_label {
        None => None,
        Some(name) => match store.etype_id(name) {
            Some(id) => Some(id),
            None => return empty(),
        },
    };
    let Col::Nodes(src) = batch.slot(from) else {
        // Only a node frontier can be expanded; anything else yields nothing.
        return empty();
    };

    let type_ok = |et: u32| want.is_none_or(|w| w == et);
    let mut keep = Vec::new();
    let mut nbrs = Vec::new();
    for (row, &v) in src.iter().enumerate() {
        let out = matches!(dir, Dir::Out | Dir::Both);
        let inc = matches!(dir, Dir::In | Dir::Both);
        if out {
            for a in store.out(v) {
                if type_ok(a.etype) {
                    keep.push(row);
                    nbrs.push(a.nbr);
                }
            }
        }
        if inc {
            for a in store.inc(v) {
                if type_ok(a.etype) {
                    keep.push(row);
                    nbrs.push(a.nbr);
                }
            }
        }
    }

    let mut slots: Vec<Col> = batch.slots.iter().map(|c| c.gather(&keep)).collect();
    slots.push(Col::Nodes(nbrs));
    Batch::of(slots)
}

/// Evaluate `expr` over every row of `batch`, producing a column.
fn eval(expr: &Expr, store: &Store, batch: &Batch) -> Col {
    match expr {
        Expr::Slot(n) => batch.slot(*n).clone(),
        Expr::Lit(v) => broadcast(v.clone(), batch.rows()),
        Expr::Prop { slot, key } => read_property(store, batch.slot(*slot), key),
        Expr::Not(inner) => {
            let c = eval(inner, store, batch);
            map_bool(&c, |b| b.map(|x| !x))
        }
        Expr::And(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        }),
        Expr::Or(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        }),
        Expr::Compare { op, left, right } => {
            let l = eval(left, store, batch);
            let r = eval(right, store, batch);
            compare(*op, &l, &r)
        }
    }
}

/// Read `key` off an element frontier as a column, bulk-gathering the typed
/// storage column and staying unboxed when it and every read entry are
/// present-and-typed; fall to `Gen` (with nulls) otherwise.
fn read_property(store: &Store, col: &Col, key: &str) -> Col {
    let Col::Nodes(ids) = col else {
        return Col::Gen(vec![Value::Null; col.len()]);
    };
    let Some(column) = store.column(key) else {
        return Col::Gen(vec![Value::Null; ids.len()]);
    };
    match column {
        Column::Num { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Num(ids.iter().map(|&i| data[i as usize]).collect())
        }
        Column::Str { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Str(ids.iter().map(|&i| data[i as usize].clone()).collect())
        }
        Column::Bool { data, present } if ids.iter().all(|&i| present[i as usize]) => {
            Col::Bool(ids.iter().map(|&i| data[i as usize]).collect())
        }
        _ => Col::Gen(ids.iter().map(|&i| store.prop(i, key)).collect()),
    }
}

fn broadcast(v: Value, n: usize) -> Col {
    match v {
        Value::Num(x) => Col::Num(vec![x; n]),
        Value::Bool(b) => Col::Bool(vec![b; n]),
        Value::Str(s) => Col::Str(vec![s; n]),
        Value::Null => Col::Gen(vec![Value::Null; n]),
    }
}

/// Compare two columns elementwise into a `Bool` column. `=`/`<>` use the value
/// contract's `equals`; ordering uses `cmp_total`. A NULL operand yields UNKNOWN,
/// carried as a `Gen` cell of `Null` so the three-valued logic upstream sees it.
fn compare(op: CompareOp, l: &Col, r: &Col) -> Col {
    let n = l.len().min(r.len());
    let mut out = Vec::with_capacity(n);
    let mut any_unknown = false;
    for i in 0..n {
        let a = l.value_at(i);
        let b = r.value_at(i);
        if a.is_null() || b.is_null() {
            any_unknown = true;
            out.push(None);
            continue;
        }
        let res = match op {
            CompareOp::Eq => value::equals(&a, &b),
            CompareOp::Ne => !value::equals(&a, &b),
            CompareOp::Lt => value::cmp_total(&a, &b).is_lt(),
            CompareOp::Le => value::cmp_total(&a, &b).is_le(),
            CompareOp::Gt => value::cmp_total(&a, &b).is_gt(),
            CompareOp::Ge => value::cmp_total(&a, &b).is_ge(),
        };
        out.push(Some(res));
    }
    if any_unknown {
        Col::Gen(
            out.into_iter()
                .map(|o| o.map_or(Value::Null, Value::Bool))
                .collect(),
        )
    } else {
        Col::Bool(out.into_iter().map(|o| o.expect("no unknowns")).collect())
    }
}

/// Read a column as three-valued booleans (None = UNKNOWN).
fn as_truth(col: &Col) -> Vec<Option<bool>> {
    match col {
        Col::Bool(bs) => bs.iter().map(|&b| Some(b)).collect(),
        other => (0..other.len())
            .map(|i| match other.value_at(i) {
                Value::Bool(b) => Some(b),
                _ => None,
            })
            .collect(),
    }
}

fn map_bool(col: &Col, f: impl Fn(Option<bool>) -> Option<bool>) -> Col {
    truth_to_col(as_truth(col).into_iter().map(f).collect())
}

fn zip_bool(
    store: &Store,
    batch: &Batch,
    l: &Expr,
    r: &Expr,
    f: impl Fn(Option<bool>, Option<bool>) -> Option<bool>,
) -> Col {
    let lc = as_truth(&eval(l, store, batch));
    let rc = as_truth(&eval(r, store, batch));
    let n = lc.len().min(rc.len());
    truth_to_col((0..n).map(|i| f(lc[i], rc[i])).collect())
}

fn truth_to_col(out: Vec<Option<bool>>) -> Col {
    if out.iter().all(Option::is_some) {
        Col::Bool(out.into_iter().map(|o| o.expect("all some")).collect())
    } else {
        Col::Gen(
            out.into_iter()
                .map(|o| o.map_or(Value::Null, Value::Bool))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Plan;
    use crate::store::Builder;
    use std::sync::Arc;

    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }
    fn prop(slot: usize, key: &str) -> Expr {
        Expr::Prop {
            slot,
            key: key.to_string(),
        }
    }
    fn lit(v: Value) -> Expr {
        Expr::Lit(v)
    }
    fn cmp(op: CompareOp, l: Expr, r: Expr) -> Expr {
        Expr::Compare {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }
    fn scan(label: &str) -> Plan {
        Plan::Scan {
            label: Some(label.to_string()),
        }
    }
    fn names_of(out: &Rows, col: usize) -> Vec<String> {
        out.rows
            .iter()
            .map(|r| match &r[col] {
                Value::Str(x) => x.to_string(),
                other => format!("{other:?}"),
            })
            .collect()
    }

    fn social() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
        let bob = b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
        let c = b.node(&["Person"], &[("name", s("carol")), ("age", n(40.0))]);
        let proj = b.node(&["Project"], &[("name", s("graphdb"))]);
        b.edge(a, bob, "KNOWS");
        b.edge(a, c, "KNOWS");
        b.edge(bob, c, "KNOWS");
        b.edge(a, proj, "WORKS_ON");
        b.build()
    }

    // --- relational core (unchanged behavior, now slot-addressed) ---

    #[test]
    fn scan_label_and_project() {
        let store = social();
        let out = run(
            &scan("Person").project(vec![("name".into(), prop(0, "name"))]),
            &store,
        );
        assert_eq!(out.rows.len(), 3);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob", "carol"]);
    }

    #[test]
    fn filter_numeric_then_project() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Gt, prop(0, "age"), lit(n(28.0))))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    #[test]
    fn absent_property_is_null_and_filters_as_unknown() {
        let store = social();
        // Project has no age → `age >= 0` is UNKNOWN for it → dropped.
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Ge, prop(0, "age"), lit(n(0.0))))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 3);
    }

    #[test]
    fn equality_is_cross_type_false() {
        let store = social();
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Eq, prop(0, "age"), lit(s("30"))))
            .project(vec![("name".into(), prop(0, "name"))]);
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    // --- Expand ---

    /// `MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name` — two slots bound,
    /// row per matching edge.
    #[test]
    fn expand_binds_both_ends() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .project(vec![
                ("a".into(), prop(0, "name")),
                ("b".into(), prop(1, "name")),
            ]);
        let out = run(&plan, &store);
        let mut pairs: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), as_str(&r[1])))
            .collect();
        pairs.sort();
        // a→b, a→c, b→c (KNOWS only; the WORKS_ON edge is excluded)
        assert_eq!(
            pairs,
            vec![
                ("alice".into(), "bob".into()),
                ("alice".into(), "carol".into()),
                ("bob".into(), "carol".into()),
            ]
        );
    }

    /// An edge-label filter selects: WORKS_ON reaches only the Project.
    #[test]
    fn expand_filters_by_edge_label() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("WORKS_ON"))
            .project(vec![("t".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["graphdb"]);
    }

    /// Filtering on the FAR end after an expand — the far slot's property.
    #[test]
    fn filter_on_the_expanded_end() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("KNOWS"))
            .filter(cmp(CompareOp::Ge, prop(1, "age"), lit(n(40.0))))
            .project(vec![("a".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // Only edges landing on carol(40): alice→carol, bob→carol.
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob"]);
    }

    /// Incoming direction: who KNOWS carol.
    #[test]
    fn expand_incoming() {
        let store = social();
        let plan = scan("Person")
            .filter(cmp(CompareOp::Eq, prop(0, "name"), lit(s("carol"))))
            .expand(0, Dir::In, Some("KNOWS"))
            .project(vec![("who".into(), prop(1, "name"))]);
        let out = run(&plan, &store);
        let mut got = names_of(&out, 0);
        got.sort();
        assert_eq!(got, vec!["alice", "bob"]);
    }

    /// An unknown edge label matches nothing.
    #[test]
    fn expand_unknown_label_is_empty() {
        let store = social();
        let plan = scan("Person")
            .expand(0, Dir::Out, Some("NOPE"))
            .project(vec![("x".into(), prop(1, "name"))]);
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    fn as_str(v: &Value) -> String {
        match v {
            Value::Str(x) => x.to_string(),
            other => format!("{other:?}"),
        }
    }

    fn num(v: &Value) -> f64 {
        match v {
            Value::Num(x) => *x,
            other => panic!("expected a number, got {other:?}"),
        }
    }

    // --- Aggregate / group-by ---

    fn agg(func: AggFn, arg: Option<Expr>, distinct: bool, name: &str) -> Agg {
        Agg {
            func,
            arg,
            distinct,
            name: name.to_string(),
        }
    }

    /// Scalar `count(*)` over a label — one row, the count.
    #[test]
    fn scalar_count_star() {
        let store = social();
        let plan = scan("Person").aggregate(vec![], vec![agg(AggFn::Count, None, false, "c")]);
        let out = run(&plan, &store);
        assert_eq!(out.names, vec!["c"]);
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&out.rows[0][0]), 3.0); // alice, bob, carol
    }

    /// sum / min / max / avg over the age column, hand-computed: 30,25,40.
    #[test]
    fn scalar_sum_min_max_avg() {
        let store = social();
        let plan = scan("Person").aggregate(
            vec![],
            vec![
                agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
                agg(AggFn::Min, Some(prop(0, "age")), false, "lo"),
                agg(AggFn::Max, Some(prop(0, "age")), false, "hi"),
                agg(AggFn::Avg, Some(prop(0, "age")), false, "av"),
            ],
        );
        let out = run(&plan, &store);
        let r = &out.rows[0];
        assert_eq!(num(&r[0]), 95.0); // 30+25+40
        assert_eq!(num(&r[1]), 25.0);
        assert_eq!(num(&r[2]), 40.0);
        assert_eq!(num(&r[3]), 95.0 / 3.0);
    }

    /// `count(*)` grouped by a property — a row per distinct value, first-seen
    /// order. Group on `city`: alice/carol="nyc", bob="sf".
    #[test]
    fn group_count_by_property() {
        let mut b = Builder::default();
        b.node(&["P"], &[("city", s("nyc"))]);
        b.node(&["P"], &[("city", s("sf"))]);
        b.node(&["P"], &[("city", s("nyc"))]);
        let store = b.build();
        let plan = scan("P").aggregate(
            vec![("city".into(), prop(0, "city"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        let out = run(&plan, &store);
        assert_eq!(out.names, vec!["city", "c"]);
        // first-seen order: nyc (row 0), then sf (row 1).
        assert_eq!(as_str(&out.rows[0][0]), "nyc");
        assert_eq!(num(&out.rows[0][1]), 2.0);
        assert_eq!(as_str(&out.rows[1][0]), "sf");
        assert_eq!(num(&out.rows[1][1]), 1.0);
    }

    /// `count(arg)` ignores nulls; `count(DISTINCT arg)` ignores nulls AND
    /// duplicates. Ages: 10, 10, null, 20 → count=3, distinct=2.
    #[test]
    fn count_arg_and_count_distinct_skip_nulls() {
        let mut b = Builder::default();
        b.node(&["P"], &[("v", n(10.0))]);
        b.node(&["P"], &[("v", n(10.0))]);
        b.node(&["P"], &[]); // no v → null
        b.node(&["P"], &[("v", n(20.0))]);
        let store = b.build();
        let plan = scan("P").aggregate(
            vec![],
            vec![
                agg(AggFn::Count, Some(prop(0, "v")), false, "c"),
                agg(AggFn::Count, Some(prop(0, "v")), true, "cd"),
            ],
        );
        let out = run(&plan, &store);
        assert_eq!(num(&out.rows[0][0]), 3.0); // non-null count
        assert_eq!(num(&out.rows[0][1]), 2.0); // distinct non-null: {10, 20}
    }

    /// sum over an empty group is NULL, not 0 — only count is 0 over nothing.
    #[test]
    fn sum_over_empty_is_null_count_is_zero() {
        let store = social();
        // No node has this label → empty input to the scalar aggregate.
        let plan = Plan::Scan {
            label: Some("Nonexistent".into()),
        }
        .aggregate(
            vec![],
            vec![
                agg(AggFn::Count, None, false, "c"),
                agg(AggFn::Sum, Some(prop(0, "age")), false, "s"),
            ],
        );
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 1); // scalar aggregate still emits one row
        assert_eq!(num(&out.rows[0][0]), 0.0); // count(*) = 0
        assert!(out.rows[0][1].is_null()); // sum = NULL
    }

    /// A grouped aggregate over empty input emits ZERO rows (unlike the scalar
    /// case) — there are no groups.
    #[test]
    fn grouped_over_empty_is_zero_rows() {
        let store = social();
        let plan = Plan::Scan {
            label: Some("Nonexistent".into()),
        }
        .aggregate(
            vec![("k".into(), prop(0, "age"))],
            vec![agg(AggFn::Count, None, false, "c")],
        );
        assert_eq!(run(&plan, &store).rows.len(), 0);
    }

    /// Aggregate after an Expand: out-degree per person (count of KNOWS edges),
    /// grouped by the source. alice→2, bob→1, carol→0(absent).
    #[test]
    fn count_out_degree_grouped_by_source() {
        let store = social();
        let plan = scan("Person").expand(0, Dir::Out, Some("KNOWS")).aggregate(
            vec![("who".into(), prop(0, "name"))],
            vec![agg(AggFn::Count, None, false, "deg")],
        );
        let out = run(&plan, &store);
        let mut got: Vec<(String, f64)> = out
            .rows
            .iter()
            .map(|r| (as_str(&r[0]), num(&r[1])))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        // carol has no outgoing KNOWS, so she is absent from the expanded rows.
        assert_eq!(got, vec![("alice".into(), 2.0), ("bob".into(), 1.0)]);
    }

    // --- Order + Page ---

    fn asc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: false,
        }
    }
    fn desc(slot: usize, key: &str) -> crate::ir::SortKey {
        crate::ir::SortKey {
            expr: prop(slot, key),
            descending: true,
        }
    }

    /// ORDER BY age ascending, then project name.
    #[test]
    fn order_by_ascending() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![asc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // ages 30,25,40 -> bob(25), alice(30), carol(40)
        assert_eq!(names_of(&out, 0), vec!["bob", "alice", "carol"]);
    }

    /// Descending reverses it.
    #[test]
    fn order_by_descending() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![desc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["carol", "alice", "bob"]);
    }

    /// ORDER BY ... LIMIT is a top-k prefix of the sorted order.
    #[test]
    fn order_then_limit_is_top_k() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![desc(0, "age")], None, Some(2))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        assert_eq!(names_of(&out, 0), vec!["carol", "alice"]); // two oldest
    }

    /// SKIP then LIMIT is a paging window over the sorted order.
    #[test]
    fn order_skip_limit_paging_window() {
        let store = social();
        let plan = scan("Person")
            .order_page(vec![asc(0, "age")], Some(1), Some(1))
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // sorted bob,alice,carol; skip 1, take 1 -> alice
        assert_eq!(names_of(&out, 0), vec!["alice"]);
    }

    /// Nulls sort LAST in ascending order (the value contract's policy).
    #[test]
    fn nulls_sort_last_ascending() {
        let mut b = Builder::default();
        b.node(&["P"], &[("name", s("has30")), ("age", n(30.0))]);
        b.node(&["P"], &[("name", s("noage"))]); // null age
        b.node(&["P"], &[("name", s("has10")), ("age", n(10.0))]);
        let store = b.build();
        let plan = scan("P")
            .order_page(vec![asc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // 10, 30, then null last
        assert_eq!(names_of(&out, 0), vec!["has10", "has30", "noage"]);
    }

    /// Multi-key: city ascending, then age descending within a city.
    #[test]
    fn multi_key_order() {
        let mut b = Builder::default();
        b.node(
            &["P"],
            &[("name", s("a")), ("city", s("nyc")), ("age", n(30.0))],
        );
        b.node(
            &["P"],
            &[("name", s("b")), ("city", s("sf")), ("age", n(40.0))],
        );
        b.node(
            &["P"],
            &[("name", s("c")), ("city", s("nyc")), ("age", n(50.0))],
        );
        let store = b.build();
        let plan = scan("P")
            .order_page(vec![asc(0, "city"), desc(0, "age")], None, None)
            .project(vec![("name".into(), prop(0, "name"))]);
        let out = run(&plan, &store);
        // nyc: c(50) before a(30); then sf: b(40)
        assert_eq!(names_of(&out, 0), vec!["c", "a", "b"]);
    }
}
