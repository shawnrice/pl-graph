//! The rewrite-rule optimizer: meaning-preserving `Plan -> Plan` transforms on
//! the neutral IR. This is the point of having one IR — a rule is written ONCE
//! and fires on plans from either front-end (GQL or Gremlin), because by the time
//! it runs the plan no longer knows which language produced it.
//!
//! Each rule is a pure function that either rewrites a node or leaves it. The
//! [`optimize`] driver rewrites children first (bottom-up), applies the local
//! rules, and repeats to a fixpoint. Every rule is tested two ways: the result
//! ROWS are unchanged (run original vs optimized, compare as bags) and the plan
//! SHAPE changed as intended.

use crate::ir::{Expr, Plan};

/// Apply the rule set to a fixpoint (bounded, so a misbehaving rule cannot spin).
#[must_use]
pub fn optimize(plan: Plan) -> Plan {
    let mut plan = plan;
    for _ in 0..64 {
        let (next, changed) = rewrite(plan);
        plan = next;
        if !changed {
            break;
        }
    }
    plan
}

/// Rewrite one node: optimize its children, then apply the local rules to it.
/// Returns the new plan and whether anything changed.
fn rewrite(plan: Plan) -> (Plan, bool) {
    let (plan, child_changed) = map_children(plan);
    let (plan, local_changed) = apply_local(plan);
    (plan, child_changed || local_changed)
}

/// Rebuild a node with its children individually rewritten.
fn map_children(plan: Plan) -> (Plan, bool) {
    match plan {
        // Leaves: no children to rewrite.
        p @ (Plan::Scan { .. } | Plan::Insert { .. }) => (p, false),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
        } => {
            let (i, c) = rewrite(*input);
            (
                Plan::Expand {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                },
                c,
            )
        }
        Plan::VarLength {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            trail,
        } => {
            let (i, c) = rewrite(*input);
            (
                Plan::VarLength {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                    min,
                    max,
                    trail,
                },
                c,
            )
        }
        Plan::ShortestPath {
            input,
            from,
            dir,
            edge_label,
            max,
        } => {
            let (i, c) = rewrite(*input);
            (
                Plan::ShortestPath {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                    max,
                },
                c,
            )
        }
        Plan::Filter { input, pred } => {
            let (i, c) = rewrite(*input);
            (
                Plan::Filter {
                    input: Box::new(i),
                    pred,
                },
                c,
            )
        }
        Plan::Aggregate { input, keys, aggs } => {
            let (i, c) = rewrite(*input);
            (
                Plan::Aggregate {
                    input: Box::new(i),
                    keys,
                    aggs,
                },
                c,
            )
        }
        Plan::OrderPage {
            input,
            keys,
            skip,
            limit,
        } => {
            let (i, c) = rewrite(*input);
            (
                Plan::OrderPage {
                    input: Box::new(i),
                    keys,
                    skip,
                    limit,
                },
                c,
            )
        }
        Plan::Project { input, items } => {
            let (i, c) = rewrite(*input);
            (
                Plan::Project {
                    input: Box::new(i),
                    items,
                },
                c,
            )
        }
        Plan::Distinct { input } => {
            let (i, c) = rewrite(*input);
            (Plan::Distinct { input: Box::new(i) }, c)
        }
        Plan::Join { left, right, on } => {
            let (l, cl) = rewrite(*left);
            let (r, cr) = rewrite(*right);
            (
                Plan::Join {
                    left: Box::new(l),
                    right: Box::new(r),
                    on,
                },
                cl || cr,
            )
        }
    }
}

/// The local rules, tried in order at a single node.
fn apply_local(plan: Plan) -> (Plan, bool) {
    match plan {
        Plan::Filter { input, pred } => match *input {
            // filter-merge: `Filter(Filter(x, p2), p1)` -> `Filter(x, p1 AND p2)`.
            Plan::Filter {
                input: inner,
                pred: p_inner,
            } => (
                Plan::Filter {
                    input: inner,
                    pred: Expr::And(Box::new(pred), Box::new(p_inner)),
                },
                true,
            ),
            // predicate pushdown below an Expand: legal when the predicate reads
            // only slots that exist BELOW the expand (i.e. not the slot the expand
            // appends). The appended slot index equals the input's width.
            Plan::Expand {
                input: ein,
                from,
                dir,
                edge_label,
            } if refs_below(&pred, width(&ein)) => (
                Plan::Expand {
                    input: Box::new(Plan::Filter { input: ein, pred }),
                    from,
                    dir,
                    edge_label,
                },
                true,
            ),
            // predicate pushdown into a Join's LEFT side: legal when the predicate
            // reads only left slots (indices < left width; the join keeps the left
            // slots' indices, so no remap is needed). Right-side pushdown would
            // need a slot remap and is deferred.
            Plan::Join { left, right, on } if refs_below(&pred, width(&left)) => (
                Plan::Join {
                    left: Box::new(Plan::Filter { input: left, pred }),
                    right,
                    on,
                },
                true,
            ),
            other => (
                Plan::Filter {
                    input: Box::new(other),
                    pred,
                },
                false,
            ),
        },
        other => (other, false),
    }
}

/// Does `expr` reference only slots `< bound` (so it can move below an operator
/// whose output starts at slot `bound`)? An expression that references no slots
/// (a constant, or a Path) trivially qualifies.
fn refs_below(expr: &Expr, bound: usize) -> bool {
    max_slot(expr).is_none_or(|m| m < bound)
}

/// The highest slot index an expression reads, or `None` if it reads no slots.
fn max_slot(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Slot(n) => Some(*n),
        Expr::Prop { slot, .. } => Some(*slot),
        Expr::Lit(_) | Expr::Path => None,
        Expr::Not(x) => max_slot(x),
        Expr::And(a, b) | Expr::Or(a, b) => merge_max(max_slot(a), max_slot(b)),
        Expr::Compare { left, right, .. } => merge_max(max_slot(left), max_slot(right)),
    }
}

fn merge_max(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) | (None, x) => x,
    }
}

/// The number of slots a plan's output rows carry — used to know which slots
/// exist below an operator for pushdown legality.
fn width(plan: &Plan) -> usize {
    match plan {
        Plan::Scan { .. } => 1,
        Plan::Insert { .. } => 0, // a write carries no output row

        Plan::Expand { input, .. }
        | Plan::VarLength { input, .. }
        | Plan::ShortestPath { input, .. } => width(input) + 1,
        Plan::Filter { input, .. } | Plan::OrderPage { input, .. } | Plan::Distinct { input } => {
            width(input)
        }
        Plan::Project { items, .. } => items.len(),
        Plan::Aggregate { keys, aggs, .. } => keys.len() + aggs.len(),
        Plan::Join { left, right, .. } => width(left) + width(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{run, Rows};
    use crate::ir::{CompareOp, Dir, Expr, Plan};
    use crate::store::{Builder, Store};
    use crate::value::Value;
    use std::sync::Arc;

    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }
    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn prop(slot: usize, key: &str) -> Expr {
        Expr::Prop {
            slot,
            key: key.to_string(),
        }
    }
    fn cmp(op: CompareOp, l: Expr, r: Expr) -> Expr {
        Expr::Compare {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    fn social() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
        let bob = b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
        let c = b.node(&["Person"], &[("name", s("carol")), ("age", n(40.0))]);
        b.edge(a, bob, "KNOWS");
        b.edge(a, c, "KNOWS");
        b.edge(bob, c, "KNOWS");
        b.build()
    }

    fn bag(rows: &Rows) -> Vec<String> {
        let mut out: Vec<String> = rows
            .rows
            .iter()
            .map(|r| r.iter().map(|v| format!("{v:?};")).collect::<String>())
            .collect();
        out.sort();
        out
    }

    /// Optimizing must never change the answer — the core invariant.
    fn assert_rows_preserved(plan: &Plan, store: &Store) -> Plan {
        let before = bag(&run(plan, store));
        let opt = optimize(plan.clone());
        assert_eq!(before, bag(&run(&opt, store)), "optimize changed the rows");
        opt
    }

    #[test]
    fn pushdown_below_expand() {
        let store = social();
        // Filter on slot 0 (the source) sits above the Expand; it should move
        // below it.
        let plan = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, Some("KNOWS"))
        .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))));
        let opt = assert_rows_preserved(&plan, &store);
        // Shape: Expand{ input: Filter{ Scan } }.
        match opt {
            Plan::Expand { input, .. } => {
                assert!(
                    matches!(*input, Plan::Filter { .. }),
                    "filter now below expand"
                );
            }
            other => panic!("expected Expand at top, got {other:?}"),
        }
    }

    #[test]
    fn no_pushdown_when_predicate_reads_the_expanded_slot() {
        let store = social();
        // Filter on slot 1 (the expanded neighbour) cannot move below the Expand.
        let plan = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, Some("KNOWS"))
        .filter(cmp(CompareOp::Ge, prop(1, "age"), Expr::Lit(n(40.0))));
        let opt = assert_rows_preserved(&plan, &store);
        // Shape unchanged: Filter still on top of Expand.
        match opt {
            Plan::Filter { input, .. } => {
                assert!(
                    matches!(*input, Plan::Expand { .. }),
                    "filter stays above expand"
                );
            }
            other => panic!("expected Filter at top, got {other:?}"),
        }
    }

    #[test]
    fn adjacent_filters_merge() {
        let store = social();
        let plan = Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(cmp(CompareOp::Ge, prop(0, "age"), Expr::Lit(n(28.0))))
        .filter(cmp(CompareOp::Le, prop(0, "age"), Expr::Lit(n(35.0))));
        let opt = assert_rows_preserved(&plan, &store);
        // And the answer: only alice(30) is in [28,35].
        assert_eq!(run(&opt, &store).rows.len(), 1);
        // Shape: one Filter (an And) over the Scan.
        match &opt {
            Plan::Filter { input, pred } => {
                assert!(matches!(pred, Expr::And(..)), "merged into an AND");
                assert!(
                    matches!(**input, Plan::Scan { .. }),
                    "single filter over scan"
                );
            }
            other => panic!("expected a single Filter, got {other:?}"),
        }
    }

    #[test]
    fn driver_reaches_fixpoint_merge_then_pushdown() {
        let store = social();
        // Two filters (slot 0) above an Expand: the driver must MERGE them and
        // then PUSH the merged filter below the Expand — two rules, to a fixpoint.
        let plan = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, Some("KNOWS"))
        .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))))
        .filter(cmp(CompareOp::Ge, prop(0, "age"), Expr::Lit(n(0.0))));
        let opt = assert_rows_preserved(&plan, &store);
        match opt {
            Plan::Expand { input, .. } => match *input {
                Plan::Filter { input, pred } => {
                    assert!(matches!(pred, Expr::And(..)), "the two filters merged");
                    assert!(matches!(*input, Plan::Scan { .. }));
                }
                other => panic!("expected merged Filter below Expand, got {other:?}"),
            },
            other => panic!("expected Expand at top, got {other:?}"),
        }
    }

    #[test]
    fn pushdown_into_join_left() {
        let store = social();
        // A filter on a left-side slot pushes into the Join's left input.
        let left = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, Some("KNOWS"));
        let right = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, Some("KNOWS"));
        // left width is 2 (a, b); a filter on slot 0 (left's a) pushes left.
        let plan = Plan::join(left, right, vec![(0, 0)]).filter(cmp(
            CompareOp::Eq,
            prop(0, "name"),
            Expr::Lit(s("alice")),
        ));
        let opt = assert_rows_preserved(&plan, &store);
        match opt {
            Plan::Join { left, .. } => {
                // The left input now begins with a Filter somewhere it was pushed
                // to (top of the left subtree, above or below its own expand).
                let has_filter = plan_contains_filter(&left);
                assert!(has_filter, "filter pushed into the left side");
            }
            other => panic!("expected Join at top, got {other:?}"),
        }
    }

    fn plan_contains_filter(p: &Plan) -> bool {
        match p {
            Plan::Filter { .. } => true,
            Plan::Expand { input, .. }
            | Plan::VarLength { input, .. }
            | Plan::ShortestPath { input, .. }
            | Plan::Aggregate { input, .. }
            | Plan::OrderPage { input, .. }
            | Plan::Project { input, .. }
            | Plan::Distinct { input } => plan_contains_filter(input),
            Plan::Join { left, right, .. } => {
                plan_contains_filter(left) || plan_contains_filter(right)
            }
            Plan::Scan { .. } | Plan::Insert { .. } => false,
        }
    }

    /// Optimizing a plan from the GQL front-end preserves rows — the rules fire
    /// on either language's output because the IR is neutral.
    #[test]
    fn optimizes_a_parsed_gql_plan() {
        let store = social();
        let plan = crate::gql::parse(
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'alice' RETURN b.name AS b",
        )
        .unwrap();
        // The WHERE (slot 0) should end up pushed below the Expand.
        let _ = assert_rows_preserved(&plan, &store);
    }
}
