//! Bind query parameters into a PREPARED plan: replace every unbound
//! [`Expr::Param`] with the supplied [`Expr::Lit`] value, in place, before the
//! plan is optimized and executed.
//!
//! A prepared statement is parsed ONCE ([`crate::gql::parse_prepared`], which
//! leaves `$name` as `Expr::Param`) and then, per run, its plan is cloned and
//! bound here with fresh values — so the parse cost is paid once across a loop of
//! executions. Binding before `opt` keeps the value typed (never spliced into
//! query text) and lets the planner seed indexes on the now-literal predicate,
//! exactly as a direct parameterized query does.
//!
//! Both walks are EXHAUSTIVE (no `_` arm), so a new `Plan`/`Expr` variant fails to
//! compile until it is handled here. A `Param` that still slips through to
//! evaluation is a loud unbound-parameter error (the `eval` safety net), never a
//! wrong result.

use crate::ir::{Expr, Plan};
use crate::value::Value;
use std::collections::HashMap;

/// Replace every `Expr::Param($name)` in `plan` with its bound value. Errors on
/// the first parameter with no supplied value.
pub fn bind_params(plan: &mut Plan, params: &[(String, Value)]) -> Result<(), String> {
    let map: HashMap<&str, &Value> = params.iter().map(|(k, v)| (k.as_str(), v)).collect();
    bind_plan(plan, &map)
}

fn bind_plan(plan: &mut Plan, params: &HashMap<&str, &Value>) -> Result<(), String> {
    match plan {
        // Leaves — no child plan, no bindable expression. (Insert / AddEdge /
        // CallProcedure carry only literal values; a prepared statement cannot put a
        // parameter in a literal-only position, so none hold an `Expr::Param`.)
        Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::EdgeSeed { .. }
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. }
        | Plan::TxControl { .. } => {}

        // Single-input wrappers with no expression of their own.
        Plan::Sample { input, .. }
        | Plan::Enumerate { input, .. }
        | Plan::EdgeVertex { input, .. }
        | Plan::GroupToMap { input }
        | Plan::Subgraph { input, .. }
        | Plan::Tree { input, .. }
        | Plan::AlgoAnnotate { input, .. }
        | Plan::Tail { input, .. }
        | Plan::Distinct { input }
        | Plan::DistinctBy { input, .. }
        | Plan::NullPadIfEmpty { input, .. }
        | Plan::ShortestPathEnum { input, .. }
        | Plan::Expand { input, .. }
        | Plan::OptionalExpand { input, .. }
        | Plan::SortLocal { input, .. } => bind_plan(input, params)?,

        // Single-input + one or more expressions.
        Plan::Unwind { input, list, .. } => {
            bind_plan(input, params)?;
            bind_expr(list, params)?;
        }
        Plan::ShortestPath {
            input, edge_pred, ..
        } => {
            bind_plan(input, params)?;
            if let Some(p) = edge_pred {
                bind_expr(p, params)?;
            }
        }
        Plan::IntervalExpand {
            input, qlo, qhi, ..
        } => {
            bind_plan(input, params)?;
            bind_expr(qlo, params)?;
            bind_expr(qhi, params)?;
        }
        Plan::VarLength {
            input,
            until,
            body_filter,
            ..
        } => {
            bind_plan(input, params)?;
            if let Some(p) = until {
                bind_expr(p, params)?;
            }
            if let Some(p) = body_filter {
                bind_expr(p, params)?;
            }
        }
        Plan::RepeatGroup {
            input,
            per_rep_pred,
            ..
        }
        | Plan::NestedGroup {
            input,
            per_rep_pred,
            ..
        } => {
            bind_plan(input, params)?;
            if let Some(p) = per_rep_pred {
                bind_expr(p, params)?;
            }
        }
        Plan::Filter { input, pred } => {
            bind_plan(input, params)?;
            bind_expr(pred, params)?;
        }
        Plan::PathRecord { input, value, .. } => {
            bind_plan(input, params)?;
            bind_expr(value, params)?;
        }
        Plan::Aggregate { input, keys, aggs } => {
            bind_plan(input, params)?;
            for (_, e) in keys {
                bind_expr(e, params)?;
            }
            for agg in aggs {
                if let Some(arg) = &mut agg.arg {
                    bind_expr(arg, params)?;
                }
            }
        }
        Plan::MapSlot { input, value, .. } => {
            bind_plan(input, params)?;
            bind_expr(value, params)?;
        }
        Plan::OrderPage { input, keys, .. } => {
            bind_plan(input, params)?;
            for k in keys {
                bind_expr(&mut k.expr, params)?;
            }
        }
        Plan::Project { input, items } => {
            bind_plan(input, params)?;
            for (_, e) in items {
                bind_expr(e, params)?;
            }
        }
        Plan::OptionalScan { input, filters, .. } => {
            bind_plan(input, params)?;
            for (_, e) in filters {
                bind_expr(e, params)?;
            }
        }
        Plan::CallInline {
            input,
            body,
            yields,
            ..
        } => {
            bind_plan(input, params)?;
            bind_plan(body, params)?;
            for (_, e) in yields {
                bind_expr(e, params)?;
            }
        }
        Plan::InsertReturn { tail, .. } => bind_plan(tail, params)?,
        // Write ops: SET `value` expressions and the _MERGE on-create/on-update
        // assignments can carry a parameter (an upsert loop with $-values).
        Plan::Update { input, ops } => {
            bind_plan(input, params)?;
            for op in ops {
                if let crate::ir::SetOp::Set { value, .. } = op {
                    bind_expr(value, params)?;
                }
            }
        }
        Plan::UpdateReturn { input, ops, tail } => {
            bind_plan(input, params)?;
            for op in ops {
                if let crate::ir::SetOp::Set { value, .. } = op {
                    bind_expr(value, params)?;
                }
            }
            bind_plan(tail, params)?;
        }
        Plan::InsertFrom {
            input,
            nodes,
            edges,
        } => {
            bind_plan(input, params)?;
            for n in nodes {
                for (_, e) in &mut n.props {
                    bind_expr(e, params)?;
                }
            }
            for ed in edges {
                for (_, e) in &mut ed.props {
                    bind_expr(e, params)?;
                }
            }
        }
        Plan::Merge {
            on_create,
            on_update,
            tail,
            ..
        } => {
            for (_, e) in on_create {
                bind_expr(e, params)?;
            }
            if let crate::ir::MergeUpdate::Set { assigns, filter } = on_update {
                for (_, e) in assigns {
                    bind_expr(e, params)?;
                }
                if let Some(f) = filter {
                    bind_expr(f, params)?;
                }
            }
            if let Some(tail) = tail {
                bind_plan(tail, params)?;
            }
        }
        Plan::MergeEdge {
            on_create,
            on_update,
            ..
        } => {
            for (_, _, e) in on_create {
                bind_expr(e, params)?;
            }
            if let crate::ir::MergeEdgeUpdate::Set { assigns, filter } = on_update {
                for (_, _, e) in assigns {
                    bind_expr(e, params)?;
                }
                if let Some(f) = filter {
                    bind_expr(f, params)?;
                }
            }
        }

        // Multi-input.
        Plan::Union { left, right, .. } | Plan::Join { left, right, .. } => {
            bind_plan(left, params)?;
            bind_plan(right, params)?;
        }
        Plan::Branch { input, bodies } => {
            bind_plan(input, params)?;
            for b in bodies {
                bind_plan(b, params)?;
            }
        }
        Plan::PerElementBranch {
            input, cond, arms, ..
        } => {
            bind_plan(input, params)?;
            if let Some(c) = cond {
                bind_plan(c, params)?;
            }
            for a in arms {
                bind_plan(a, params)?;
            }
        }
        Plan::Reconverge { input, .. } => bind_plan(input, params)?,
    }
    Ok(())
}

fn bind_expr(e: &mut Expr, params: &HashMap<&str, &Value>) -> Result<(), String> {
    match e {
        Expr::Param(name) => {
            let v = params
                .get(name.as_str())
                .ok_or_else(|| format!("unbound parameter `${name}`"))?;
            *e = Expr::Lit((*v).clone());
        }
        // Leaves — nothing to bind. `GremlinPath` is Gremlin-only and its `bys` are
        // render modulators, not expressions, so no GQL `$param` can appear there.
        Expr::Slot(_)
        | Expr::IsLabeled { .. }
        | Expr::Prop { .. }
        | Expr::Lit(_)
        | Expr::Path
        | Expr::PropertyExists { .. }
        | Expr::PathAccess { .. }
        | Expr::GremlinPath { .. }
        | Expr::GremlinFullPath { .. } => {}

        Expr::Not(a) => bind_expr(a, params)?,
        Expr::And(a, b) | Expr::Or(a, b) | Expr::Xor(a, b) => {
            bind_expr(a, params)?;
            bind_expr(b, params)?;
        }
        Expr::Compare { left, right, .. } | Expr::Arith { left, right, .. } => {
            bind_expr(left, params)?;
            bind_expr(right, params)?;
        }
        Expr::In { needle, haystack } => {
            bind_expr(needle, params)?;
            bind_expr(haystack, params)?;
        }
        Expr::Call { args, .. } | Expr::GraphPred { args, .. } | Expr::List { items: args } => {
            for a in args {
                bind_expr(a, params)?;
            }
        }
        Expr::Record { fields } | Expr::MapLit { entries: fields } => {
            for (_, v) in fields {
                bind_expr(v, params)?;
            }
        }
        Expr::Case {
            branches,
            otherwise,
        } => {
            for (c, v) in branches {
                bind_expr(c, params)?;
                bind_expr(v, params)?;
            }
            if let Some(o) = otherwise {
                bind_expr(o, params)?;
            }
        }
        Expr::Field { base, .. } => bind_expr(base, params)?,
        Expr::Index { base, index, .. } => {
            bind_expr(base, params)?;
            bind_expr(index, params)?;
        }
        Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => bind_expr(expr, params)?,

        // Subqueries — recurse the nested plan (and any scalar it yields).
        Expr::Exists { body, .. }
        | Expr::CountSubquery { body, .. }
        | Expr::UncorrelatedExists { body, .. }
        | Expr::UncorrelatedCount { body, .. } => bind_plan(body, params)?,
        Expr::ScalarSubquery { body, scalar, .. }
        | Expr::CollectSubquery { body, scalar, .. }
        | Expr::AggSubquery { body, scalar, .. } => {
            bind_plan(body, params)?;
            bind_expr(scalar, params)?;
        }
        Expr::UncorrelatedScalar { body, .. } => bind_plan(body, params)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests;
