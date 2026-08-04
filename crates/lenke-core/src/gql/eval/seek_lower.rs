//! Lowering GQL predicates into the shared [`ElementSeek`].
//!
//! This is the GQL-specific half of `docs/design/query-ir.md`: it knows about
//! `CExpr` and nothing about indexes. Everything below it — which conjunct is
//! most selective, how an `OR` unions, how two bounds on one key fold into a
//! range — lives in [`crate::seek`] and is shared with Gremlin.
//!
//! # Collapsing spellings
//!
//! The point is that semantically identical predicates produce the SAME
//! `ElementSeek`, not merely two structures that happen to cost the same:
//!
//! ```text
//!   WHERE u.k = $x          ┐
//!   WHERE $x = u.k          ├─  all four lower to
//!   MATCH (u:P {k: $x})     │   conj: [k Eq Param(0)]
//!   WHERE u.k IN [$x]       ┘
//! ```
//!
//! Each of those was a separate arm of the old recogniser, and three of the four
//! were once missing — costing 60x, 107x and 220x while returning the right
//! answer. A missing arm here cannot be silent in the same way, because
//! `seek_lower_tests` asserts the structures are equal.

use std::sync::Arc;

use super::super::ast::{CompareOp, Lit};
use super::super::plan::{CExpr, CNode};
use super::{prop_path as eval_prop_path, val_to_idxkey, Ctx, Val};
use crate::graph::{Graph, IdxKey};
use crate::seek::{Bindings, Branch, ElementSeek, KeyPredicate, Operand, SeekOp};

/// GQL's parameter slots, as the shared layer sees them.
pub(super) struct GqlBindings<'a>(pub &'a [Val]);

impl Bindings for GqlBindings<'_> {
    fn scalar(&self, slot: usize) -> Option<IdxKey> {
        val_to_idxkey(self.0.get(slot)?)
    }

    fn list(&self, slot: usize) -> Option<Vec<IdxKey>> {
        match self.0.get(slot)? {
            Val::List(items) => items.iter().map(val_to_idxkey).collect(),
            // `IN $p` where `$p` is a scalar is a one-element list.
            single => Some(vec![val_to_idxkey(single)?]),
        }
    }
}

/// Only these four map onto an index seek. `Ne` is the complement of a point —
/// most of the index — and every other operator is not an ordering at all.
fn seek_op(op: CompareOp) -> Option<SeekOp> {
    match op {
        CompareOp::Eq => Some(SeekOp::Eq),
        CompareOp::Lt => Some(SeekOp::Lt),
        CompareOp::Le => Some(SeekOp::Le),
        CompareOp::Gt => Some(SeekOp::Gt),
        CompareOp::Ge => Some(SeekOp::Ge),
        CompareOp::Ne => None,
    }
}

/// The value side of a comparison: an inline literal, or a parameter slot left
/// unresolved so the plan stays reusable across bindings.
fn operand(e: &CExpr) -> Option<Operand> {
    match e {
        CExpr::Lit(lit) => lit_key(lit).map(Operand::Lit),
        CExpr::Param(slot) => Some(Operand::Param(*slot)),
        _ => None,
    }
}

fn lit_key(lit: &Lit) -> Option<IdxKey> {
    match lit {
        Lit::Str(s) => Some(IdxKey::Str(s.as_str().into())),
        Lit::Num(n) => Some(IdxKey::Num(*n)),
        Lit::Bool(b) => Some(IdxKey::Bool(*b)),
        Lit::Temporal(t) => t.index_key().map(|(k, key)| IdxKey::Temporal(k, key)),
        Lit::Null => None,
    }
}

/// The (var slot, dotted index path) an expression addresses.
///
/// Resolves through `Ctx`, which means execution time. Moving it to plan time
/// means reading `CQuery::key_names` instead — the names are already there —
/// and is the next increment.
fn prop_path(e: &CExpr, graph: &Graph, ctx: &Ctx, edge: bool) -> Option<(usize, String)> {
    eval_prop_path(e, graph, ctx, edge)
}

/// One `left OP right` as a predicate on `want_slot`, in EITHER operand order.
///
/// `$x = u.k` is `u.k = $x`, and `5 <= u.n` is `u.n >= 5` — flipping the
/// operator along with the operands. Reading only the left side is what made
/// constant-first spellings cost 107x.
///
/// A `want_slot` of `None` is an ANONYMOUS element, and takes nothing. It reads
/// as "no slot to match against", which is not the same as "any slot" — every
/// caller passes an element's own `var_slot`, so `None` means the element has no
/// name, and a predicate naming a variable therefore cannot be about it.
/// Accepting one anyway seeded a node from a constraint on a DIFFERENT variable:
/// `MATCH ()-[:R]->(b) WHERE b.n = 7` seeded the anonymous start from `b.n = 7`,
/// walked out of the 50 vertices that happened to have `n = 7` themselves, and
/// returned 0 rows where the answer is 150. Silently wrong and 26x faster for it,
/// which is why no benchmark caught it.
fn compare(
    op: CompareOp,
    left: &CExpr,
    right: &CExpr,
    graph: &Graph,
    ctx: &Ctx,
    want_slot: Option<usize>,
    edge: bool,
) -> Option<KeyPredicate> {
    let (slot, key, op, value) = match prop_path(left, graph, ctx, edge) {
        Some((slot, key)) => (slot, key, seek_op(op)?, operand(right)?),
        None => {
            let (slot, key) = prop_path(right, graph, ctx, edge)?;

            (slot, key, seek_op(op)?.flipped(), operand(left)?)
        }
    };

    if want_slot != Some(slot) {
        return None;
    }

    Some(KeyPredicate {
        key: Arc::from(key.as_str()),
        op,
        operand: value,
    })
}

/// Collect everything `e` says about `want_slot` into `out`.
///
/// Conjunction is additive — an `AND` contributes each of its conjuncts, and one
/// it cannot read is simply not contributed, because any single conjunct still
/// bounds a valid superset. Disjunction is all-or-nothing: an `OR` with one
/// unreadable branch contributes NOTHING, since a union missing a branch is no
/// longer a superset and would drop rows.
fn collect(
    e: &CExpr,
    graph: &Graph,
    ctx: &Ctx,
    want_slot: Option<usize>,
    edge: bool,
    out: &mut ElementSeek,
) {
    match e {
        CExpr::Compare { op, left, right } => {
            if let Some(p) = compare(*op, left, right, graph, ctx, want_slot, edge) {
                out.conj_push(p);
            }
        }
        CExpr::And(items) => {
            for it in items {
                collect(it, graph, ctx, want_slot, edge, out);
            }
        }
        CExpr::Or(items) => {
            if let Some(branches) = branches_of(items, graph, ctx, want_slot, edge) {
                out.push_branches(branches);
            }
        }
        CExpr::In {
            expr,
            list,
            negated: false,
        } => {
            let Some((slot, key)) = prop_path(expr, graph, ctx, edge) else {
                return;
            };

            if want_slot != Some(slot) {
                return;
            }

            let key: Arc<str> = Arc::from(key.as_str());

            match list.as_ref() {
                CExpr::List(items) => {
                    let Some(values) = items.iter().map(operand).collect::<Option<Vec<_>>>() else {
                        return; // a computed element ⇒ not a constant list
                    };

                    out.push_any_of(key, values);
                }
                // The list lives in a parameter; its length is unknown until
                // execution, so the disjunction stays symbolic.
                CExpr::Param(slot) => out.push_any_of_param(key, *slot),
                _ => {}
            }
        }
        // `NOT IN` / `<>` / `IS NULL` and everything else: no seek. Deliberately
        // silent — these are not gaps, they are predicates whose matches no point
        // or range seek enumerates.
        _ => {}
    }
}

/// Every branch of an `OR` as its own conjunction, or `None` if any branch
/// contributes nothing seekable.
fn branches_of(
    items: &[CExpr],
    graph: &Graph,
    ctx: &Ctx,
    want_slot: Option<usize>,
    edge: bool,
) -> Option<Vec<Branch>> {
    let mut branches = Vec::with_capacity(items.len());

    for it in items {
        let mut sub = ElementSeek::same_kind(edge);

        collect(it, graph, ctx, want_slot, edge, &mut sub);

        // A nested `OR` flattens: `(a OR b) OR c` is `a OR b OR c`, one union
        // rather than a union containing a union.
        branches.extend(sub.into_branches()?);
    }

    Some(branches)
}

/// A label expression as a flat id list, when it is one.
///
/// Only `:A` and `:A|B` lower — a negation, conjunction or wildcard is not a set
/// of ids and keeps GQL's own evaluator. Lowering what fits is the point; the
/// rest is not a gap.
///
/// GQL's rule is ANY: `(n:Person)` matches a vertex carrying that label anywhere
/// in its list, unlike Gremlin's first-label-only `hasLabel`. The IR carries the
/// rule, so both share the seeding underneath.
pub(super) fn lower_labels(
    expr: &super::super::plan::CLabelExpr,
    ctx: &Ctx,
    edge: bool,
) -> Option<Vec<u32>> {
    use super::super::plan::CLabelExpr as L;

    match expr {
        L::Label(r) => {
            let (v, e) = ctx.labels[*r];
            // An unresolved name matches nothing — an EMPTY id list, not "no
            // constraint".
            Some(if edge { e } else { v }.map_or_else(Vec::new, |id| vec![id]))
        }
        L::Or(l, r) => {
            let mut ids = lower_labels(l, ctx, edge)?;

            ids.extend(lower_labels(r, ctx, edge)?);
            Some(ids)
        }
        _ => None,
    }
}

/// What a WHERE clause plus a pattern's inline `{k: v}` constraints say about
/// one element variable.
pub(super) fn element_seek(
    where_: Option<&CExpr>,
    inline: &[(&str, &CExpr)],
    graph: &Graph,
    ctx: &Ctx,
    want_slot: Option<usize>,
    edge: bool,
) -> ElementSeek {
    let mut out = ElementSeek::same_kind(edge);

    // `MATCH (u:P {k: $x})` is `MATCH (u:P) WHERE u.k = $x`. Same structure, so
    // the same seek — this pair was a 60x gap.
    for (name, value) in inline {
        // `CPropConstraint` already carries the key NAME, so an inline
        // constraint needs no `Ctx` at all — it is plan-time-ready today.
        if let Some(op) = operand(value) {
            out.conj_push(KeyPredicate {
                key: Arc::from(*name),
                op: SeekOp::Eq,
                operand: op,
            });
        }
    }

    if let Some(w) = where_ {
        collect(w, graph, ctx, want_slot, edge, &mut out);
    }

    out
}

/// The inline constraints of a relationship.
pub(super) fn inline_of_rel(rel: &super::super::plan::CRel) -> Vec<(&str, &CExpr)> {
    rel.props
        .iter()
        .map(|pc| (pc.key.as_str(), &pc.value))
        .collect()
}

/// The inline constraints of a node, in the shape [`element_seek`] wants.
pub(super) fn inline_of(node: &CNode) -> Vec<(&str, &CExpr)> {
    node.props
        .iter()
        .map(|pc| (pc.key.as_str(), &pc.value))
        .collect()
}

/// Every vertex matching one pattern node, through the shared scan loop.
///
/// The single place GQL lowers "a node plus what constrains it" into
/// [`crate::seek`]: the label (when it is a flat id list), the inline `{k: v}`
/// constraints and the seekable part of `anchor` become IR, and whatever is left
/// — an arbitrary predicate, a negated or conjoined label — becomes the residual
/// closure. Both the isolated-node scan and the traversal start seed call this,
/// which is what stops the lowering from being written twice.
///
/// `anchor` is the WHERE to seed from: the clause's for an isolated node, and for
/// a traversal start the clause's OR the node's own — a conjunct of either must
/// hold for every matching row, so seeding from it only narrows.
pub(super) fn scan_node(
    graph: &Graph,
    ctx: &Ctx,
    node: &CNode,
    anchor: Option<&CExpr>,
    scope_len: usize,
    cap: Option<usize>,
) -> Vec<u32> {
    let label_ids = node
        .label
        .as_ref()
        .and_then(|l| lower_labels(l, ctx, false));
    let mut seek = element_seek(anchor, &inline_of(node), graph, ctx, node.var_slot, false);

    if let Some(ids) = label_ids.clone() {
        seek.set_labels(ids);
    }

    let binds = GqlBindings(ctx.params);
    let residual_label = label_ids.is_none();
    let needs_check = !node.props.is_empty() || node.where_.is_some();

    // Nothing left over: hand the shared loop a no-op residual it can
    // monomorphize away rather than a closure to call per candidate.
    if !residual_label && !needs_check {
        return seek.scan_capped(graph, &binds, cap, || graph.vertex_indices().collect());
    }

    let mut b = super::Binding::with_len(scope_len.max(1));

    seek.scan_with(
        graph,
        &binds,
        cap,
        || graph.vertex_indices().collect(),
        |vi| {
            if residual_label && !super::matches_label(graph, ctx, vi, node.label.as_ref()) {
                return false;
            }

            if !needs_check {
                return true;
            }

            if let Some(slot) = node.var_slot {
                b.set(slot, super::Val::Node(vi));
            }

            super::satisfies(
                graph,
                ctx,
                &super::Val::Node(vi),
                &node.props,
                node.where_.as_ref(),
                &b,
            )
        },
    )
}
