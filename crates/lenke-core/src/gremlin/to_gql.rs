//! A Gremlin TAIL, compiled to a GQL projection.
//!
//! `pattern::compile` already turns a linear traversal PREFIX into GQL's `CPath`,
//! which is how Gremlin gets GQL's planner (4-45x, `route_audit`). This is the
//! other half: the steps AFTER the prefix, compiled into the `CProjection` that
//! GQL's own projection/aggregation machinery runs. Together they make a Gremlin
//! traversal a GQL statement.
//!
//! # Why this exists
//!
//! Gremlin's terminals re-implement projection, aggregation, DISTINCT, ORDER BY
//! and paging that GQL already has — 868 executable lines in `apply` plus ~550 in
//! the column terminals, against one set of clauses that does the same work. Two
//! implementations of one algebra is where a fix or an optimization lands on one
//! side and not the other, and it is ~959KB of the wasm bundle.
//!
//! # What this does NOT do yet
//!
//! Everything it cannot express returns `None` and the existing paths run
//! unchanged, so declining is always safe. The tail vocabulary below is the first
//! slice; the migration finishes when it covers the whole mainstream surface and
//! the arms it replaces can be deleted in one pass. They cannot be deleted
//! piecemeal: an arm only goes when its step can never reach the stream, which
//! means covering it in EVERY context including the sub-traversals of the steps
//! that stay (per-traverser state — `sack`, the side-effect collections, path
//! history, `sample` — which do not compile to this IR at all).
//!
//! # The boundary
//!
//! Per-language differences are handled by the CALLER shaping columns, never by a
//! dialect branch inside GQL's evaluator. Three are already known: Gremlin drops
//! a row whose property key is ABSENT where GQL nulls it (read the column's own
//! validity mask — asking for `PROPERTY_EXISTS` as a second item reads the column
//! twice and costs 3.6x), a stored map renders as a `Map` not a `Record`, and a
//! self-loop is two adjacencies not one (`Ctx::loops`, already a parameter).

use crate::gql::plan::{compile_program, CAgg, CExpr, CProjection, CReturnItem};
use crate::gremlin::{Scope, Step};

use super::pattern::Compiled;

/// How the projected columns become Gremlin values.
///
/// The columns are the same either way; what differs is the container, which is
/// exactly the axis the two languages disagree on and the reason this is the
/// caller's job rather than the evaluator's.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Shape {
    /// One value per row, from column 0.
    Rows,
    /// A single row holding one value — a global aggregate.
    Scalar,
}

pub(super) struct Tail {
    pub proj: CProjection,
    pub shape: Shape,
}

/// An empty projection to fill in — every field spelled once, here, so the arms
/// below stay readable.
fn blank(items: Vec<CReturnItem>) -> CProjection {
    let out_names = items.iter().map(|i| i.name.clone()).collect::<Vec<_>>();

    CProjection {
        star: false,
        distinct: false,
        aggregating: false,
        group_by: Vec::new(),
        aggs: Vec::new(),
        out_len: items.len(),
        star_cols: Vec::new(),
        order_by: Vec::new(),
        order_overlay: Vec::new(),
        order_needs_output: false,
        having: None,
        skip: None,
        limit: None,
        out_names,
        items,
    }
}

fn item(expr: CExpr, name: &str, is_agg: bool) -> CReturnItem {
    CReturnItem {
        prog: compile_program(&expr),
        expr,
        name: name.to_string(),
        is_agg,
    }
}

/// The key's index in `key_names`, appending it if the prefix did not already
/// need it. `CExpr::Prop` refers to a property by this index, not by name.
fn key_ref(keys: &mut Vec<String>, k: &str) -> usize {
    keys.iter().position(|n| n == k).unwrap_or_else(|| {
        keys.push(k.to_string());
        keys.len() - 1
    })
}

/// Which aggregate a Gremlin reducing terminal is, when it is one.
fn reducer(step: &Step) -> Option<crate::gql::plan::AggFn> {
    use crate::gql::plan::AggFn;

    Some(match step {
        Step::Sum(Scope::Global) => AggFn::Sum,
        Step::Min(Scope::Global) => AggFn::Min,
        Step::Max(Scope::Global) => AggFn::Max,
        Step::Mean(Scope::Global) => AggFn::Avg,
        _ => return None,
    })
}

/// Compile the steps after a prefix into a projection over its end slot.
///
/// `None` = not expressible; the caller keeps its existing route.
pub(super) fn tail(c: &Compiled, keys: &mut Vec<String>, rest: &[Step]) -> Option<Tail> {
    let slot = c.end_slot;

    match rest {
        // `values(k)` — one value per row off the end of the prefix. The rows
        // whose key is absent are dropped by the CALLER, from the column's own
        // validity mask.
        [Step::Values(ks)] if ks.len() == 1 => {
            let kr = key_ref(keys, &ks[0]);

            Some(Tail {
                proj: blank(vec![item(
                    CExpr::Prop {
                        var_slot: slot,
                        key_ref: kr,
                    },
                    "v",
                    false,
                )]),
                shape: Shape::Rows,
            })
        }
        // `values(k).<sum|min|max|mean>()` — a global aggregate over that column.
        // `count()` is deliberately absent: Gremlin counts ROWS and its own arm
        // answers that as the frontier length without touching a column at all.
        [Step::Values(ks), red] if ks.len() == 1 && reducer(red).is_some() => {
            let kr = key_ref(keys, &ks[0]);
            let func = reducer(red)?;
            let arg = CExpr::Prop {
                var_slot: slot,
                key_ref: kr,
            };
            let mut proj = blank(vec![item(CExpr::AggRef(0), "v", true)]);

            proj.aggregating = true;
            proj.aggs = vec![CAgg {
                func,
                arg: Some(arg),
                distinct: false,
                star: false,
                frac: None,
            }];

            Some(Tail {
                proj,
                shape: Shape::Scalar,
            })
        }
        // `values(k).dedup()` — DISTINCT over the projected column.
        [Step::Values(ks), Step::Dedupe { labels, bys }]
            if ks.len() == 1 && labels.is_empty() && bys.is_empty() =>
        {
            let kr = key_ref(keys, &ks[0]);
            let mut proj = blank(vec![item(
                CExpr::Prop {
                    var_slot: slot,
                    key_ref: kr,
                },
                "v",
                false,
            )]);

            proj.distinct = true;

            Some(Tail {
                proj,
                shape: Shape::Rows,
            })
        }
        // NOT YET: `order().by(k)`. It compiles, and it sorts WRONG — a sort key
        // referencing an input slot has to be published through `order_overlay`
        // (the scope ORDER BY resolves against), and without that GQL ordered by
        // the output element instead. The symptom was subtle: the first rows
        // matched and then a vertex with `n = 0` sorted after one with `n = 10`.
        // Caught by comparing against the route this replaces rather than against
        // the stream — the stream enumerates in a different order by design, which
        // hides a real ordering bug among permitted ones.
        _ => None,
    }
}
