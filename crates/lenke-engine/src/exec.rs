//! Execution: pull a batch up through the plan, then materialize the projection.
//!
//! Expression evaluation is columnar — `eval` produces a `Col` over the whole
//! batch, reading typed storage columns in bulk where it can. It calls the value
//! contract for every comparison and equality; it never restates those rules.
//! This is the lineage-FREE strategy; the lineage-preserving strategy for the
//! same operators lands with the graph operators that need it.

use crate::batch::{Batch, Col};
use crate::ir::{CompareOp, Expr, Plan};
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
/// `Project` (the only operator that names output columns); scanning or filtering
/// without a projection is not a complete query.
pub fn run(plan: &Plan, store: &Store) -> Rows {
    match plan {
        Plan::Project { input, items } => {
            let batch = pull(input, store);
            let names = items.iter().map(|(n, _)| n.clone()).collect();
            // One output column per item, evaluated over the surviving batch.
            let cols: Vec<Col> = items.iter().map(|(_, e)| eval(e, store, &batch)).collect();
            let n = batch.len();
            let rows = (0..n)
                .map(|i| cols.iter().map(|c| c.value_at(i)).collect())
                .collect();
            Rows { names, rows }
        }
        // A bare scan/filter with no projection surfaces the element ids under a
        // single implicit column, so partial plans are still runnable in tests.
        other => {
            let batch = pull(other, store);
            let n = batch.len();
            let rows = (0..n).map(|i| vec![batch.col.value_at(i)]).collect();
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
            Batch::plain(Col::Nodes(ids))
        }
        Plan::Filter { input, pred } => {
            let batch = pull(input, store);
            let mask = eval(pred, store, &batch);
            let keep: Vec<usize> = match &mask {
                Col::Bool(bs) => (0..bs.len()).filter(|&i| bs[i]).collect(),
                // A non-boolean predicate is UNKNOWN for every row → keep none.
                // (This is where a boxed predicate would be checked per row; the
                // typed Bool column is the common, bulk case.)
                other => (0..other.len())
                    .filter(|&i| other.value_at(i).is_true())
                    .collect(),
            };
            Batch::plain(batch.col.gather(&keep))
        }
        Plan::Project { .. } => {
            // A projection mid-plan collapses to its element frontier for the
            // slice; nested projections arrive with subqueries.
            pull_project_as_frontier(plan, store)
        }
    }
}

fn pull_project_as_frontier(plan: &Plan, store: &Store) -> Batch {
    if let Plan::Project { input, .. } = plan {
        pull(input, store)
    } else {
        unreachable!("called with a non-Project plan")
    }
}

/// Evaluate `expr` over every row of `batch`, producing a column. The element
/// frontier is `batch.col` (a `Nodes` column in the slice); `Var`/`Prop` read
/// against it.
fn eval(expr: &Expr, store: &Store, batch: &Batch) -> Col {
    match expr {
        Expr::Var => batch.col.clone(),
        Expr::Lit(v) => broadcast(v.clone(), batch.len()),
        Expr::Prop { key } => read_property(store, &batch.col, key),
        Expr::Not(inner) => {
            let c = eval(inner, store, batch);
            map_bool(&c, |b| b.map(|x| !x))
        }
        Expr::And(l, r) => zip_bool(store, batch, l, r, |a, b| match (a, b) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None, // UNKNOWN
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

/// Read `key` off an element frontier as a column, gathering the typed storage
/// column in bulk where the frontier is nodes. Non-node frontiers have no
/// properties → all null.
fn read_property(store: &Store, col: &Col, key: &str) -> Col {
    let Col::Nodes(ids) = col else {
        return Col::Gen(vec![Value::Null; col.len()]);
    };
    let Some(column) = store.column(key) else {
        return Col::Gen(vec![Value::Null; ids.len()]);
    };
    // Bulk gather from the typed column, staying unboxed when it and every read
    // entry is present-and-typed; fall to Gen (with nulls) otherwise.
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
        // Carry UNKNOWNs precisely so a later NOT/AND/OR is three-valued.
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
                Value::Null => None,
                _ => None, // non-boolean is UNKNOWN in a logical context
            })
            .collect(),
    }
}

fn map_bool(col: &Col, f: impl Fn(Option<bool>) -> Option<bool>) -> Col {
    let out: Vec<Option<bool>> = as_truth(col).into_iter().map(f).collect();
    truth_to_col(out)
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
    fn prop(key: &str) -> Expr {
        Expr::Prop {
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

    fn fixture() -> Store {
        let mut b = Builder::default();
        b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
        b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
        b.node(
            &["Person", "Admin"],
            &[("name", s("carol")), ("age", n(40.0))],
        );
        b.node(&["Robot"], &[("name", s("r2"))]); // no age
        b.build()
    }

    /// Scan a label → project a property. The whole pipeline end to end.
    #[test]
    fn scan_label_and_project() {
        let store = fixture();
        let plan = Plan::Scan {
            label: Some("Person".to_string()),
        }
        .project(vec![("name".to_string(), prop("name"))]);
        let out = run(&plan, &store);
        assert_eq!(out.names, vec!["name"]);
        let names: Vec<Value> = out.rows.into_iter().map(|mut r| r.remove(0)).collect();
        assert_eq!(names.len(), 3); // alice, bob, carol — not the Robot
        assert!(names
            .iter()
            .any(|v| matches!(v, Value::Str(x) if &**x == "alice")));
        assert!(!names
            .iter()
            .any(|v| matches!(v, Value::Str(x) if &**x == "r2")));
    }

    /// Filter on a numeric property, then project — the bulk numeric path.
    #[test]
    fn filter_numeric_then_project() {
        let store = fixture();
        let plan = Plan::Scan {
            label: Some("Person".to_string()),
        }
        .filter(cmp(CompareOp::Gt, prop("age"), lit(n(28.0))))
        .project(vec![
            ("name".to_string(), prop("name")),
            ("age".to_string(), prop("age")),
        ]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2); // alice(30), carol(40)
        for row in &out.rows {
            assert!(matches!(row[1], Value::Num(a) if a > 28.0));
        }
    }

    /// A property absent on some scanned nodes reads as NULL, and a comparison
    /// against NULL is UNKNOWN → the row drops (three-valued filter).
    #[test]
    fn absent_property_is_null_and_filters_as_unknown() {
        let store = fixture();
        // Scan ALL nodes; the Robot has no age, so `age > 0` is UNKNOWN for it.
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Ge, prop("age"), lit(n(0.0))))
            .project(vec![("name".to_string(), prop("name"))]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 3); // the three People; Robot dropped (NULL age)
    }

    /// `=` is the value contract's equality: cross-type is false, not an error.
    #[test]
    fn equality_is_cross_type_false() {
        let store = fixture();
        let plan = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Eq, prop("age"), lit(s("30"))))
            .project(vec![("name".to_string(), prop("name"))]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 0); // number 30 never equals string "30"
    }

    /// AND is three-valued: an UNKNOWN conjunct doesn't keep a row.
    #[test]
    fn three_valued_and() {
        let store = fixture();
        let plan = Plan::Scan { label: None }
            .filter(Expr::And(
                Box::new(cmp(CompareOp::Ge, prop("age"), lit(n(0.0)))),
                Box::new(cmp(CompareOp::Lt, prop("age"), lit(n(35.0)))),
            ))
            .project(vec![("name".to_string(), prop("name"))]);
        let out = run(&plan, &store);
        assert_eq!(out.rows.len(), 2); // alice(30), bob(25); carol(40) excluded, robot UNKNOWN
    }
}
