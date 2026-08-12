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

use crate::ir::{CompareOp, Expr, Plan};
use crate::store::Store;
use crate::value::Value;

/// Frontier width of a PURE Scan/Expand chain — `Some(width)` when `plan` is only
/// seeds and expands (so every one of its slots is a bound node/edge element), else
/// `None`. Mirrors `exec::chain_width`; kept here so the optimizer can prove "every
/// slot is an element" without reaching into exec. A `bind_edge` Expand appends two
/// slots (edge then node); anything else (Filter/Project/Aggregate/…) is not a pure
/// chain, so a slot there might be a projected scalar — hence `None`.
fn pure_chain_width(plan: &Plan) -> Option<usize> {
    match plan {
        Plan::Scan { .. } | Plan::IndexSeek { .. } | Plan::RangeSeek { .. } => Some(1),
        Plan::Expand {
            input, bind_edge, ..
        } => Some(pure_chain_width(input)? + if *bind_edge { 2 } else { 1 }),
        _ => None,
    }
}

/// What the planner may know about the store's PHYSICAL indexes, so a seed rule can
/// prefer a conjunct backed by a real index over one that would only scan. Kept
/// abstract (not `&Store`) so the optimizer stays a pure `Plan -> Plan` transform and
/// so callers with no store (plan-shape tests) can pass [`NoIndexes`].
pub trait IndexOracle {
    /// A hash index exists on the exact (possibly dotted) property path `key`.
    fn has_hash_index(&self, key: &str) -> bool;
    /// A range index exists on property `key`.
    fn has_range_index(&self, key: &str) -> bool;
}

/// The "no physical indexes" oracle: every seed becomes a scan-fallback seek, which
/// is exactly the behavior before the optimizer could see indexes. Used by the
/// store-less [`optimize`] and by plan-shape tests.
pub struct NoIndexes;
impl IndexOracle for NoIndexes {
    fn has_hash_index(&self, _key: &str) -> bool {
        false
    }
    fn has_range_index(&self, _key: &str) -> bool {
        false
    }
}

impl IndexOracle for crate::store::Store {
    fn has_hash_index(&self, key: &str) -> bool {
        Store::has_hash_index(self, key)
    }
    fn has_range_index(&self, key: &str) -> bool {
        Store::has_range_index(self, key)
    }
}

/// Apply the rule set to a fixpoint, blind to any physical indexes — a seedable
/// predicate becomes a scan-fallback `IndexSeek`/`RangeSeek`. For index-aware
/// planning (which conjunct of a multi-predicate filter to seed), use
/// [`optimize_indexed`] with the store.
#[must_use]
pub fn optimize(plan: Plan) -> Plan {
    optimize_indexed(plan, &NoIndexes)
}

/// Apply the rule set to a fixpoint, letting `idx` steer index-sensitive rules (the
/// multi-predicate seed picks a conjunct backed by a real index when one exists —
/// otherwise it still seeds one conjunct onto the typed-scan fast path, since a
/// blind seek scans anyway). Bounded so a misbehaving rule cannot spin.
#[must_use]
pub fn optimize_indexed(plan: Plan, idx: &dyn IndexOracle) -> Plan {
    let mut plan = plan;
    for _ in 0..64 {
        let (next, changed) = rewrite(plan, idx);
        plan = next;
        if !changed {
            break;
        }
    }
    plan
}

/// Rewrite one node: optimize its children, then apply the local rules to it.
/// Returns the new plan and whether anything changed.
fn rewrite(plan: Plan, idx: &dyn IndexOracle) -> (Plan, bool) {
    let (plan, child_changed) = map_children(plan, idx);
    let (plan, local_changed) = apply_local(plan, idx);
    (plan, child_changed || local_changed)
}

/// Rebuild a node with its children individually rewritten.
fn map_children(plan: Plan, idx: &dyn IndexOracle) -> (Plan, bool) {
    match plan {
        // Leaves: no children to rewrite. (`Row` only lives inside an EXISTS body,
        // which the optimizer never descends into, but it is still a leaf.)
        // `InsertReturn`'s `tail` is a Row-seeded projection with nothing for the
        // read-side rewrites (index seeds, pushdown) to act on, so it is a leaf too.
        p @ (Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::Row
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. }
        | Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. }
        | Plan::CallProcedure { .. }) => (p, false),
        Plan::Expand {
            input,
            from,
            dir,
            edge_label,
            bind_edge,
        } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::Expand {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                    bind_edge,
                },
                c,
            )
        }
        Plan::OptionalExpand {
            input,
            from,
            dir,
            edge_label,
            keep_source,
        } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::OptionalExpand {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                    keep_source,
                },
                c,
            )
        }
        Plan::IntervalExpand {
            input,
            from,
            dir,
            edge_label,
            lo_key,
            hi_key,
            qlo,
            qhi,
            bind_edge,
        } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::IntervalExpand {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                    lo_key,
                    hi_key,
                    qlo,
                    qhi,
                    bind_edge,
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
            mode,
        } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::VarLength {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                    min,
                    max,
                    mode,
                },
                c,
            )
        }
        Plan::ShortestPath {
            input,
            from,
            dir,
            edge_label,
            min,
            max,
            selector,
        } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::ShortestPath {
                    input: Box::new(i),
                    from,
                    dir,
                    edge_label,
                    min,
                    max,
                    selector,
                },
                c,
            )
        }
        Plan::Filter { input, pred } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::Filter {
                    input: Box::new(i),
                    pred,
                },
                c,
            )
        }
        Plan::Aggregate {
            input,
            keys,
            mut aggs,
        } => {
            let (i, c) = rewrite(*input, idx);
            // `count(x)` over a bound ELEMENT is `count(*)`: a pattern variable bound
            // to a node/edge is never null, so counting it counts every row. When the
            // input is a pure Scan/Expand chain EVERY slot is such an element, so
            // rewrite a non-DISTINCT `count(Slot(k))` to the argument-free form — that
            // canonicalizes `count(n)`/`count(b)` onto the O(1) `count(*)` fast path,
            // closing a spelling perf cliff (`count(n)` was a full scan, `count(*)`
            // O(1)). DISTINCT is left alone (`count(DISTINCT n)` is distinct elements,
            // not the row count). A `count(n.prop)` argument is a `Prop`, not a `Slot`,
            // so it is untouched — it genuinely counts non-null property values.
            if pure_chain_width(&i).is_some() {
                for agg in &mut aggs {
                    if agg.func == crate::ir::AggFn::Count
                        && !agg.distinct
                        && matches!(agg.arg.as_ref(), Some(Expr::Slot(_)))
                    {
                        agg.arg = None;
                    }
                }
            }
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
            let (i, c) = rewrite(*input, idx);
            // Fuse a pure PAGE over a pure SORT into one OrderPage. Gremlin lowers
            // `order().by(k)` then `range(lo, hi)` to two stacked OrderPages — an inner
            // full sort (no page) and an outer page (no keys) — so the sort ran over
            // ALL rows and the top-K / late-materialization fast path (which needs the
            // sort and the limit on the SAME node) never fired: `order().by().range()`
            // was ~7x its GQL `ORDER BY … LIMIT` twin. "Sort by k, then take
            // [skip, skip+limit)" is exactly "sort by k with that page", so merging is
            // meaning-preserving — but ONLY when the outer adds no keys and the inner
            // has no page of its own (an inner limit would pre-truncate the rows).
            if keys.is_empty() {
                if let Plan::OrderPage {
                    input: inner,
                    keys: inner_keys,
                    skip: None,
                    limit: None,
                } = i
                {
                    if !inner_keys.is_empty() {
                        return (
                            Plan::OrderPage {
                                input: inner,
                                keys: inner_keys,
                                skip,
                                limit,
                            },
                            true, // merged two nodes into one — a real change
                        );
                    }
                    // Not mergeable after all — rebuild the inner OrderPage we moved out.
                    return (
                        Plan::OrderPage {
                            input: Box::new(Plan::OrderPage {
                                input: inner,
                                keys: inner_keys,
                                skip: None,
                                limit: None,
                            }),
                            keys,
                            skip,
                            limit,
                        },
                        c,
                    );
                }
            }
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
            let (i, c) = rewrite(*input, idx);
            (
                Plan::Project {
                    input: Box::new(i),
                    items,
                },
                c,
            )
        }
        Plan::Distinct { input } => {
            let (i, c) = rewrite(*input, idx);
            (Plan::Distinct { input: Box::new(i) }, c)
        }
        Plan::Tail { input, n } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::Tail {
                    input: Box::new(i),
                    n,
                },
                c,
            )
        }
        Plan::SortLocal { input, descending } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::SortLocal {
                    input: Box::new(i),
                    descending,
                },
                c,
            )
        }
        Plan::Update { input, ops } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::Update {
                    input: Box::new(i),
                    ops,
                },
                c,
            )
        }
        Plan::Join { left, right, on } => {
            let (l, cl) = rewrite(*left, idx);
            let (r, cr) = rewrite(*right, idx);
            (
                Plan::Join {
                    left: Box::new(l),
                    right: Box::new(r),
                    on,
                },
                cl || cr,
            )
        }
        Plan::Union {
            left,
            right,
            all,
            op,
        } => {
            let (l, cl) = rewrite(*left, idx);
            let (r, cr) = rewrite(*right, idx);
            (
                Plan::Union {
                    left: Box::new(l),
                    right: Box::new(r),
                    all,
                    op,
                },
                cl || cr,
            )
        }
        // Optimize the outer `input`, but leave the correlated `body` alone: it is
        // rooted at `Plan::Row` and evaluated by `pull_body`, which expects the raw
        // Expand/Filter chain — a seed rule would rewrite it into an uneval-able
        // shape.
        Plan::CallInline {
            input,
            body,
            yields,
            outer_width,
        } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::CallInline {
                    input: Box::new(i),
                    body,
                    yields,
                    outer_width,
                },
                c,
            )
        }
        // Rewrite the input; the correlated branch bodies are left as-is (like an
        // EXISTS/CALL body, the optimizer does not descend into them).
        Plan::Branch { input, bodies } => {
            let (i, c) = rewrite(*input, idx);
            (
                Plan::Branch {
                    input: Box::new(i),
                    bodies,
                },
                c,
            )
        }
    }
}

/// If `pred` is an equality between slot-0's property and a literal — in EITHER
/// spelling (`prop = v` or `v = prop`) — return `(key, value)` for an index seek.
/// Only `=` (not ranges), only slot 0 (the scanned node). Handling both spellings
/// is the load-bearing part: a missed spelling silently keeps scanning.
/// One seedable conjunct: an equality (→ `IndexSeek`) or a range (→ `RangeSeek`).
enum Seed {
    Index(String, Value),
    Range(String, CompareOp, Value),
}

/// Given a conjunction `a AND b AND …`, pick ONE conjunct to seed and return it
/// with the residual predicate (the remaining conjuncts, re-`AND`ed). The pick is
/// INDEX-AWARE: an equality/range conjunct backed by a real physical index wins,
/// because that seek reads the index instead of scanning; only when nothing is
/// indexed does it fall back to seeding an equality (then a range) onto the
/// typed-scan fast path — a blind seek would scan the whole label anyway, so which
/// unindexed conjunct is chosen changes no cost, just avoids the far slower
/// `Filter(And(…))(Scan)`. `None` if the predicate is not an `AND`, or no conjunct
/// is seekable.
fn seed_from_conjuncts(pred: &Expr, idx: &dyn IndexOracle) -> Option<(Seed, Option<Expr>)> {
    fn flatten<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
        match e {
            Expr::And(a, b) => {
                flatten(a, out);
                flatten(b, out);
            }
            other => out.push(other),
        }
    }
    let mut conjuncts = Vec::new();
    flatten(pred, &mut conjuncts);
    if conjuncts.len() < 2 {
        return None; // not a conjunction — the single-comparison arms handle it
    }
    // Selection priority (best first): an INDEXED equality, then an INDEXED range,
    // then any equality (typed-scan fast path), then any range.
    let eq_key = |c: &Expr| seek_target(c).map(|(k, _)| k);
    let range_key = |c: &Expr| range_seek_target(c).map(|(k, _, _)| k);
    let pick = conjuncts
        .iter()
        .position(|c| eq_key(c).is_some_and(|k| idx.has_hash_index(&k)))
        .or_else(|| {
            conjuncts
                .iter()
                .position(|c| range_key(c).is_some_and(|k| idx.has_range_index(&k)))
        })
        .or_else(|| conjuncts.iter().position(|c| seek_target(c).is_some()))
        .or_else(|| {
            conjuncts
                .iter()
                .position(|c| range_seek_target(c).is_some())
        })?;
    let seed = if let Some((k, v)) = seek_target(conjuncts[pick]) {
        Seed::Index(k, v)
    } else {
        let (k, op, v) = range_seek_target(conjuncts[pick])?;
        Seed::Range(k, op, v)
    };
    // Residual: the other conjuncts, re-`AND`ed (or `None` if the seed was the only
    // one besides itself — i.e. exactly two conjuncts leaves a lone residual).
    let residual = conjuncts
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != pick)
        .map(|(_, c)| (*c).clone())
        .reduce(|a, b| Expr::And(Box::new(a), Box::new(b)));
    Some((seed, residual))
}

fn seek_target(pred: &Expr) -> Option<(String, Value)> {
    let Expr::Compare {
        op: CompareOp::Eq,
        left,
        right,
    } = pred
    else {
        return None;
    };
    // One side must be a literal, the other a (possibly dotted) property PATH on
    // slot 0. Both spellings; a dotted `n.rec.sub` seeds a dotted IndexSeek.
    let (path_expr, v) = match (left.as_ref(), right.as_ref()) {
        (e, Expr::Lit(v)) => (e, v),
        (Expr::Lit(v), e) => (e, v),
        _ => return None,
    };
    prop_path(path_expr).map(|k| (k, v.clone()))
}

/// The dotted property path an expression reads on slot 0, or `None` if it is not
/// a slot-0 property/field chain. `n.age` → `"age"`, `n.meta.city` → `"meta.city"`.
fn prop_path(e: &Expr) -> Option<String> {
    match e {
        Expr::Prop { slot: 0, key } => Some(key.clone()),
        Expr::Field { base, key } => Some(format!("{}.{key}", prop_path(base)?)),
        _ => None,
    }
}

/// A range comparison with its operands swapped — used to normalize
/// `lit <op> prop` to `prop <op'> lit`.
fn flip_range(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Ge => CompareOp::Le,
        // Not range ops; range_seek_target never reaches here.
        CompareOp::Eq | CompareOp::Ne => op,
    }
}

/// If `pred` is a RANGE comparison (`<`,`<=`,`>`,`>=`) between slot-0's property
/// and a literal — in either spelling — return `(key, op, value)` oriented with
/// the property on the left (`prop <op> value`). Flipping the op for the
/// `lit <op> prop` spelling is load-bearing: else `5 < n` never seeks.
fn range_seek_target(pred: &Expr) -> Option<(String, CompareOp, Value)> {
    let Expr::Compare { op, left, right } = pred else {
        return None;
    };
    if !matches!(
        op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return None;
    }
    match (left.as_ref(), right.as_ref()) {
        (Expr::Prop { slot: 0, key }, Expr::Lit(v)) => Some((key.clone(), *op, v.clone())),
        (Expr::Lit(v), Expr::Prop { slot: 0, key }) => {
            Some((key.clone(), flip_range(*op), v.clone()))
        }
        _ => None,
    }
}

/// The local rules, tried in order at a single node.
fn apply_local(plan: Plan, idx: &dyn IndexOracle) -> (Plan, bool) {
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
                bind_edge,
            } if refs_below(&pred, width(&ein)) => (
                Plan::Expand {
                    input: Box::new(Plan::Filter { input: ein, pred }),
                    from,
                    dir,
                    edge_label,
                    bind_edge,
                },
                true,
            ),
            // predicate pushdown below a VarLength / ShortestPath: both append the
            // reached endpoint at slot `width(input)`, keeping every input slot in
            // place, so any conjunct that reads only those input slots (the classic
            // case: a filter on the traversal SOURCE, `WHERE a.age = 1`) filters the
            // input BEFORE the expansion. The predicate is SPLIT — the source part is
            // pushed, a residual on the target stays above — because otherwise a
            // mixed `a.age = 1 AND b.age = 2` refuses to push at all and the walk
            // runs from every node: catastrophic for an unbounded `->*` reach
            // (measured: a source-filtered ANY SHORTEST went from "does not finish"
            // to instant).
            Plan::VarLength {
                input: vin,
                from,
                dir,
                edge_label,
                min,
                max,
                mode,
            } => {
                let (below, above) = split_pushable(pred, width(&vin));
                match below {
                    // No conjunct reads only the input — rebuild unchanged.
                    None => {
                        let vl = Plan::VarLength {
                            input: vin,
                            from,
                            dir,
                            edge_label,
                            min,
                            max,
                            mode,
                        };
                        (
                            Plan::Filter {
                                input: Box::new(vl),
                                pred: above.expect("a filter predicate is non-empty"),
                            },
                            false,
                        )
                    }
                    Some(below) => {
                        let inner = Plan::VarLength {
                            input: Box::new(Plan::Filter {
                                input: vin,
                                pred: below,
                            }),
                            from,
                            dir,
                            edge_label,
                            min,
                            max,
                            mode,
                        };
                        match above {
                            Some(a) => (
                                Plan::Filter {
                                    input: Box::new(inner),
                                    pred: a,
                                },
                                true,
                            ),
                            None => (inner, true),
                        }
                    }
                }
            }
            Plan::ShortestPath {
                input: sin,
                from,
                dir,
                edge_label,
                min,
                max,
                selector,
            } => {
                let (below, above) = split_pushable(pred, width(&sin));
                match below {
                    None => {
                        let sp = Plan::ShortestPath {
                            input: sin,
                            from,
                            dir,
                            edge_label,
                            min,
                            max,
                            selector,
                        };
                        (
                            Plan::Filter {
                                input: Box::new(sp),
                                pred: above.expect("a filter predicate is non-empty"),
                            },
                            false,
                        )
                    }
                    Some(below) => {
                        let inner = Plan::ShortestPath {
                            input: Box::new(Plan::Filter {
                                input: sin,
                                pred: below,
                            }),
                            from,
                            dir,
                            edge_label,
                            min,
                            max,
                            selector,
                        };
                        match above {
                            Some(a) => (
                                Plan::Filter {
                                    input: Box::new(inner),
                                    pred: a,
                                },
                                true,
                            ),
                            None => (inner, true),
                        }
                    }
                }
            }
            // interval-overlap fusion: `Filter(r.lo <= X AND r.hi >= Y)` over a
            // bind_edge Expand → an `IntervalExpand` (seek-or-scan). The predicate
            // reads the bound EDGE slot (= the expand's input width), so the
            // pushdown arm above never fires for it; here it fuses into the hop so
            // an interval-indexed store can seek. Non-interval predicates on the
            // edge fall through unchanged.
            Plan::Expand {
                input: ein,
                from,
                dir,
                edge_label,
                bind_edge: true,
            } => {
                let iw = width(&ein);
                if let Some((lo_key, hi_key, qlo, qhi)) = interval_pattern(&pred, iw) {
                    (
                        Plan::IntervalExpand {
                            input: ein,
                            from,
                            dir,
                            edge_label,
                            lo_key,
                            hi_key,
                            qlo: Box::new(qlo),
                            qhi: Box::new(qhi),
                            bind_edge: true,
                        },
                        true,
                    )
                } else {
                    (
                        Plan::Filter {
                            input: Box::new(Plan::Expand {
                                input: ein,
                                from,
                                dir,
                                edge_label,
                                bind_edge: true,
                            }),
                            pred,
                        },
                        false,
                    )
                }
            }
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
            // index seed: `Filter(prop <op> literal) over Scan(label)` -> a seek.
            // `=` seeds an IndexSeek, a range op a RangeSeek; both are semantic
            // no-ops (the seek yields exactly Scan+Filter rows) and both spellings
            // are handled (see `seek_target`/`range_seek_target`) so neither
            // silently keeps scanning.
            Plan::Scan { label: Some(l) } => {
                if let Some((key, value)) = seek_target(&pred) {
                    (
                        Plan::IndexSeek {
                            label: l,
                            key,
                            value,
                        },
                        true,
                    )
                } else if let Some((key, op, value)) =
                    range_seek_target(&pred).filter(|(k, _, _)| idx.has_range_index(k))
                {
                    // Only seed a RangeSeek when a range index can actually serve it.
                    // Without one, RangeSeek's fallback SCANS and BOXES each cell, which
                    // is SLOWER than leaving a `Filter(Scan)` — that hits the vectorized
                    // raw-`&str`/raw-f64 compare in `try_filter_keep`. (A standalone
                    // range predicate here must match the conjunct path, which already
                    // gates its Range seed on `has_range_index`.)
                    (
                        Plan::RangeSeek {
                            label: l,
                            key,
                            op,
                            value,
                        },
                        true,
                    )
                } else if let Some((seed, residual)) = seed_from_conjuncts(&pred, idx) {
                    // `Filter(a = x AND …)(Scan)` — seed ONE conjunct and keep the rest
                    // as a residual filter over the seek, so `WHERE k = x AND …` costs
                    // the same as the inline `(n:L {k: x, …})` (which seeds because it
                    // lowers to stacked single filters). Without this a multi-predicate
                    // WHERE ran the whole conjunction over a full Scan — measured 34x
                    // its inline twin. The seek is semantically Scan+Filter(conjunct),
                    // so peeling one conjunct out is a no-op on the rows.
                    let seek = match seed {
                        Seed::Index(key, value) => Plan::IndexSeek {
                            label: l,
                            key,
                            value,
                        },
                        Seed::Range(key, op, value) => Plan::RangeSeek {
                            label: l,
                            key,
                            op,
                            value,
                        },
                    };
                    let out = match residual {
                        Some(pred) => Plan::Filter {
                            input: Box::new(seek),
                            pred,
                        },
                        None => seek,
                    };
                    (out, true)
                } else {
                    (
                        Plan::Filter {
                            input: Box::new(Plan::Scan { label: Some(l) }),
                            pred,
                        },
                        false,
                    )
                }
            }
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

/// Flatten a top-level AND tree into its conjuncts (order preserved).
fn flatten_and(e: Expr, out: &mut Vec<Expr>) {
    match e {
        Expr::And(a, b) => {
            flatten_and(*a, out);
            flatten_and(*b, out);
        }
        other => out.push(other),
    }
}

/// Rebuild a left-leaning AND-chain from conjuncts; `None` if empty.
fn and_all(conjs: Vec<Expr>) -> Option<Expr> {
    let mut it = conjs.into_iter();
    let first = it.next()?;
    Some(it.fold(first, |acc, e| Expr::And(Box::new(acc), Box::new(e))))
}

/// Split `pred`'s conjuncts into those referencing only slots `< bound` (pushable
/// below an operator that appends slots ≥ `bound`) and the rest. AND is symmetric
/// for the keep-TRUE filter, so re-grouping is exact. Returns `(below, above)`.
fn split_pushable(pred: Expr, bound: usize) -> (Option<Expr>, Option<Expr>) {
    let mut conj = Vec::new();
    flatten_and(pred, &mut conj);
    let (below, above): (Vec<Expr>, Vec<Expr>) =
        conj.into_iter().partition(|c| refs_below(c, bound));
    (and_all(below), and_all(above))
}

/// Classify one comparison against the edge in slot `edge_slot` as an interval
/// endpoint constraint: `Prop{edge_slot,k} <= bound` (the LO axis, `false`) or
/// `Prop{edge_slot,k} >= bound` (the HI axis, `true`), including the mirrored
/// spellings (`bound >= prop`, `bound <= prop`). Returns `(is_hi, key, bound)`.
fn interval_side(c: &Expr, edge_slot: usize) -> Option<(bool, String, Expr)> {
    let Expr::Compare { op, left, right } = c else {
        return None;
    };
    // Put the edge Prop on the left, flipping the operator if it was on the right.
    let (key, bound, op) = match (&**left, &**right) {
        (Expr::Prop { slot, key }, _) if *slot == edge_slot => {
            (key.clone(), (**right).clone(), *op)
        }
        (_, Expr::Prop { slot, key }) if *slot == edge_slot => {
            (key.clone(), (**left).clone(), flip_cmp(*op))
        }
        _ => return None,
    };
    match op {
        CompareOp::Le => Some((false, key, bound)), // prop <= bound → lo axis
        CompareOp::Ge => Some((true, key, bound)),  // prop >= bound → hi axis
        _ => None,                                  // strict/eq don't map to closed overlap
    }
}

/// Swap the operands' order of a comparison (so `a OP b` ⇔ `b flip(OP) a`).
fn flip_cmp(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Ge => CompareOp::Le,
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::Ne => CompareOp::Ne,
    }
}

/// Recognize `r.lo <= X AND r.hi >= Y` (in any spelling/order) on the edge bound
/// at slot `edge_slot` (which equals the expand's input width). Returns
/// `(lo_key, hi_key, qlo, qhi)` for an `IntervalExpand` (`qlo = Y`, `qhi = X`).
/// Both bounds must be evaluable over the input row (reference only slots below
/// the hop), so they never depend on the edge/node the hop appends.
fn interval_pattern(pred: &Expr, edge_slot: usize) -> Option<(String, String, Expr, Expr)> {
    let Expr::And(a, b) = pred else { return None };
    let sa = interval_side(a, edge_slot)?;
    let sb = interval_side(b, edge_slot)?;
    // Need exactly one lo-axis and one hi-axis constraint.
    let ((_, lo_key, qhi), (_, hi_key, qlo)) = match (sa.0, sb.0) {
        (false, true) => (sa, sb),
        (true, false) => (sb, sa),
        _ => return None,
    };
    if !refs_below(&qhi, edge_slot) || !refs_below(&qlo, edge_slot) {
        return None;
    }
    Some((lo_key, hi_key, qlo, qhi))
}

/// The highest slot index an expression reads, or `None` if it reads no slots.
fn max_slot(expr: &Expr) -> Option<usize> {
    match expr {
        Expr::Slot(n) => Some(*n),
        Expr::Prop { slot, .. } => Some(*slot),
        Expr::Lit(_) | Expr::Path | Expr::PathAccess { .. } => None,
        Expr::Not(x) => max_slot(x),
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Xor(a, b)
        | Expr::Arith {
            left: a, right: b, ..
        }
        | Expr::In {
            needle: a,
            haystack: b,
        } => merge_max(max_slot(a), max_slot(b)),
        Expr::Call { args, .. } | Expr::List { items: args } => {
            args.iter().fold(None, |acc, a| merge_max(acc, max_slot(a)))
        }
        Expr::Record { fields } | Expr::MapLit { entries: fields } => fields
            .iter()
            .fold(None, |acc, (_, e)| merge_max(acc, max_slot(e))),
        Expr::Field { base, .. } => max_slot(base),
        Expr::Index { base, index } => merge_max(max_slot(base), max_slot(index)),
        Expr::Case {
            branches,
            otherwise,
        } => {
            let mut m = otherwise.as_deref().and_then(max_slot);
            for (c, v) in branches {
                m = merge_max(m, merge_max(max_slot(c), max_slot(v)));
            }
            m
        }
        Expr::Compare { left, right, .. } => merge_max(max_slot(left), max_slot(right)),
        Expr::Cast { expr, .. } | Expr::IsNull { expr, .. } => max_slot(expr),
        Expr::PropertyExists { slot, .. } => Some(*slot),
        // EXISTS correlates on outer slots below `outer_width`; claim it reads up
        // to the topmost, so the predicate is never pushed below an operator that
        // binds a variable it might reference.
        Expr::Exists { outer_width, .. } | Expr::CountSubquery { outer_width, .. } => {
            outer_width.checked_sub(1)
        }
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
        Plan::Scan { .. }
        | Plan::NodeSeed { .. }
        | Plan::EdgeScan
        | Plan::IndexSeek { .. }
        | Plan::RangeSeek { .. } => 1,
        // A named procedure yields exactly two columns: node id + its result.
        Plan::CallProcedure { .. } => 2,
        // `Row` never appears in an outer plan (it lives only in an EXISTS body,
        // which pushdown does not traverse); width is meaningless here.
        Plan::Row => 0,
        // Writes carry no output row. `InsertReturn` is a write too — its RETURN
        // rows are produced by the executor, not by this read-side width pass.
        Plan::Insert { .. }
        | Plan::InsertReturn { .. }
        | Plan::Update { .. }
        | Plan::Merge { .. }
        | Plan::AddEdge { .. } => 0,

        // A bind_edge Expand appends TWO slots (edge then node).
        Plan::Expand {
            input, bind_edge, ..
        }
        | Plan::IntervalExpand {
            input, bind_edge, ..
        } => width(input) + if *bind_edge { 2 } else { 1 },
        Plan::VarLength { input, .. }
        | Plan::ShortestPath { input, .. }
        | Plan::Branch { input, .. }
        | Plan::OptionalExpand { input, .. } => width(input) + 1,
        Plan::Filter { input, .. }
        | Plan::OrderPage { input, .. }
        | Plan::Distinct { input }
        | Plan::Tail { input, .. }
        | Plan::SortLocal { input, .. } => width(input),
        Plan::Project { items, .. } => items.len(),
        Plan::Aggregate { keys, aggs, .. } => keys.len() + aggs.len(),
        Plan::Join { left, right, .. } => width(left) + width(right),
        // UNION's result columns are the LEFT arm's.
        Plan::Union { left, .. } => width(left),
        // Outer slots kept, plus one column per yielded subquery expression.
        Plan::CallInline {
            outer_width,
            yields,
            ..
        } => outer_width + yields.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::{run, Rows};
    use crate::ir::{CompareOp, Dir, Expr, PathMode, Plan};
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

    /// `Scan(label) + Filter(prop = lit)` seeds to `IndexSeek` — for BOTH
    /// spellings, which must land on the SAME seek target (the
    /// equivalent-spellings-cost-the-same invariant) and preserve the rows.
    #[test]
    fn scan_filter_eq_seeds_index_both_spellings() {
        let store = social();
        let plan_of = |pred| {
            Plan::Scan {
                label: Some("Person".into()),
            }
            .filter(pred)
            .project(vec![("name".into(), prop(0, "name"))])
        };
        let a = plan_of(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))));
        let b = plan_of(cmp(CompareOp::Eq, Expr::Lit(s("alice")), prop(0, "name")));

        let oa = assert_rows_preserved(&a, &store);
        let ob = assert_rows_preserved(&b, &store);

        // Both spellings become Project{ IndexSeek{Person, name, "alice"} }.
        let target = |p: &Plan| -> (String, String, Value) {
            let Plan::Project { input, .. } = p else {
                panic!("expected Project, got {p:?}")
            };
            let Plan::IndexSeek { label, key, value } = input.as_ref() else {
                panic!("expected IndexSeek under Project, got {input:?}")
            };
            (label.clone(), key.clone(), value.clone())
        };
        let (la, ka, va) = target(&oa);
        let (lb, kb, vb) = target(&ob);
        assert_eq!((la.as_str(), ka.as_str()), ("Person", "name"));
        assert_eq!((la, ka), (lb, kb)); // identical target for both spellings
        assert!(matches!(&va, Value::Str(x) if &**x == "alice"));
        assert!(matches!(&vb, Value::Str(x) if &**x == "alice"));
        assert_eq!(bag(&run(&oa, &store)), vec!["Str(\"alice\");"]);
    }

    /// A dotted record-field equality `n.meta.city = 'NYC'` seeds a dotted
    /// `IndexSeek` — both spellings land the SAME target — and gives the right
    /// rows both WITH an index (index_lookup) and WITHOUT (the path-resolving scan
    /// fallback).
    #[test]
    fn dotted_field_eq_seeds_index_both_spellings() {
        use crate::value::make_record;
        use std::sync::Arc;
        let build = || {
            let mut b = Builder::default();
            b.node(
                &["Person"],
                &[
                    ("name", s("alice")),
                    ("meta", make_record(vec![(Arc::from("city"), s("NYC"))])),
                ],
            );
            b.node(
                &["Person"],
                &[
                    ("name", s("bob")),
                    ("meta", make_record(vec![(Arc::from("city"), s("LA"))])),
                ],
            );
            b.build()
        };
        let field = Expr::Field {
            base: Box::new(prop(0, "meta")),
            key: "city".into(),
        };
        let plan_of = |pred| {
            Plan::Scan {
                label: Some("Person".into()),
            }
            .filter(pred)
            .project(vec![("name".into(), prop(0, "name"))])
        };
        let a = plan_of(cmp(CompareOp::Eq, field.clone(), Expr::Lit(s("NYC"))));
        let b = plan_of(cmp(CompareOp::Eq, Expr::Lit(s("NYC")), field.clone()));

        let mut store = build();
        store.create_index("meta.city");
        let oa = assert_rows_preserved(&a, &store);
        let ob = assert_rows_preserved(&b, &store);

        let target = |p: &Plan| -> (String, Value) {
            let Plan::Project { input, .. } = p else {
                panic!("expected Project, got {p:?}")
            };
            let Plan::IndexSeek { key, value, .. } = input.as_ref() else {
                panic!("expected a (dotted) IndexSeek, got {input:?}")
            };
            (key.clone(), value.clone())
        };
        let (ka, va) = target(&oa);
        let (kb, _) = target(&ob);
        assert_eq!(ka, "meta.city");
        assert_eq!(ka, kb); // both spellings, same dotted target
        assert!(matches!(&va, Value::Str(x) if &**x == "NYC"));
        assert_eq!(bag(&run(&oa, &store)), vec!["Str(\"alice\");"]);

        // Same seed, NO index: the scan fallback resolves the path and matches.
        let no_index = build();
        let oc = optimize(a.clone());
        assert!(
            matches!(&oc, Plan::Project { input, .. } if matches!(input.as_ref(), Plan::IndexSeek { .. }))
        );
        assert_eq!(bag(&run(&oc, &no_index)), vec!["Str(\"alice\");"]);
    }

    /// A range filter is NOT seeded (that is D2); an unlabelled scan cannot seek.
    /// An unlabelled scan cannot seek (IndexSeek/RangeSeek need a label), so its
    /// filter is preserved. (A labelled range filter now seeds — see the range
    /// seed test.)
    #[test]
    fn unlabelled_scan_not_seeded() {
        let store = social();
        let unlabelled = Plan::Scan { label: None }
            .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))))
            .project(vec![("name".into(), prop(0, "name"))]);
        assert!(plan_contains_filter(&assert_rows_preserved(
            &unlabelled,
            &store
        )));
    }

    /// A range filter over a labelled scan seeds to a `RangeSeek`, for BOTH
    /// spellings (`age > 28` and `28 < age`), which land the SAME target and
    /// preserve rows (equivalent-spellings for ranges).
    #[test]
    fn range_filter_seeds_both_spellings() {
        // A RangeSeek is only planned when a range index can serve it (else the
        // vectorized Filter path is faster than the seek's scan-and-box fallback),
        // so this fixture builds one on `age`.
        let mut store = social();
        store.create_range_index("age");
        let plan_of = |pred| {
            Plan::Scan {
                label: Some("Person".into()),
            }
            .filter(pred)
            .project(vec![("name".into(), prop(0, "name"))])
        };
        // age > 28  and  28 < age  are the same predicate, different spelling.
        let a = plan_of(cmp(CompareOp::Gt, prop(0, "age"), Expr::Lit(n(28.0))));
        let b = plan_of(cmp(CompareOp::Lt, Expr::Lit(n(28.0)), prop(0, "age")));
        // Optimize through the indexed path (both spellings must normalize to the
        // same RangeSeek) and confirm the rows are unchanged from the raw plan.
        let oa = optimize_indexed(a.clone(), &store);
        let ob = optimize_indexed(b.clone(), &store);
        assert_eq!(bag(&run(&a, &store)), bag(&run(&oa, &store)));
        assert_eq!(bag(&run(&b, &store)), bag(&run(&ob, &store)));

        let target = |p: &Plan| -> (String, CompareOp, Value) {
            let Plan::Project { input, .. } = p else {
                panic!("expected Project, got {p:?}")
            };
            let Plan::RangeSeek { key, op, value, .. } = input.as_ref() else {
                panic!("expected RangeSeek under Project, got {input:?}")
            };
            (key.clone(), *op, value.clone())
        };
        let (ka, opa, va) = target(&oa);
        let (kb, opb, vb) = target(&ob);
        assert_eq!(ka, "age");
        assert_eq!(opa, CompareOp::Gt); // both normalize to prop > 28
        assert_eq!((ka, opa), (kb, opb));
        assert!(matches!(va, Value::Num(x) if x == 28.0));
        assert!(matches!(vb, Value::Num(x) if x == 28.0));
        // answer: alice(30), carol(40)
        let mut got = bag(&run(&oa, &store));
        got.sort();
        assert_eq!(got, vec!["Str(\"alice\");", "Str(\"carol\");"]);
    }

    #[test]
    fn pushdown_below_expand() {
        let store = social();
        // Filter on slot 0 (the source) sits above the Expand; it should move
        // below it.
        let plan = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))));
        let opt = assert_rows_preserved(&plan, &store);
        // Shape: Expand{ input: <pushed-down predicate> }. The pushed filter over
        // Scan(label) then seeds to an IndexSeek — either form proves the
        // predicate moved below the Expand.
        match opt {
            Plan::Expand { input, .. } => {
                assert!(
                    matches!(*input, Plan::Filter { .. } | Plan::IndexSeek { .. }),
                    "predicate now below expand (as Filter or IndexSeek)"
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
        .expand(0, Dir::Out, &["KNOWS".to_string()])
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
    fn varlen_split_pushdown_source_below_target_above() {
        let store = social();
        // `a.name = 'alice' AND b.age >= 40` over a var-length hop: the source
        // conjunct (slot 0) pushes below the VarLength; the target conjunct (slot 1,
        // the appended endpoint) stays above. Rows must be unchanged.
        let plan = Plan::Scan {
            label: Some("Person".into()),
        }
        .var_length(0, Dir::Out, &["KNOWS".to_string()], 1, 2, PathMode::Trail)
        .filter(Expr::And(
            Box::new(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice")))),
            Box::new(cmp(CompareOp::Ge, prop(1, "age"), Expr::Lit(n(40.0)))),
        ));
        let opt = assert_rows_preserved(&plan, &store);
        // Shape: Filter{target} over VarLength{ input: <pushed source> }.
        match opt {
            Plan::Filter { input, pred } => {
                assert!(!refs_below(&pred, 1), "the residual reads the target slot");
                let Plan::VarLength { input: vin, .. } = *input else {
                    panic!("expected VarLength under the residual filter");
                };
                assert!(
                    matches!(*vin, Plan::Filter { .. } | Plan::IndexSeek { .. }),
                    "the source predicate moved below the VarLength"
                );
            }
            other => panic!("expected a residual Filter on top, got {other:?}"),
        }
    }

    #[test]
    fn shortest_path_source_filter_pushes_down() {
        let store = social();
        // A pure source filter over ShortestPath pushes fully below it (no residual).
        let plan = Plan::Scan {
            label: Some("Person".into()),
        }
        .shortest_path(
            0,
            Dir::Out,
            &["KNOWS".to_string()],
            1,
            None,
            crate::ir::ShortestSelector::Any,
        )
        .filter(cmp(CompareOp::Eq, prop(0, "name"), Expr::Lit(s("alice"))));
        let opt = assert_rows_preserved(&plan, &store);
        match opt {
            Plan::ShortestPath { input, .. } => assert!(
                matches!(*input, Plan::Filter { .. } | Plan::IndexSeek { .. }),
                "source predicate now below the ShortestPath"
            ),
            other => panic!("expected ShortestPath at top, got {other:?}"),
        }
    }

    #[test]
    fn adjacent_filters_merge() {
        let store = social();
        // Unlabelled scan so seeding (which needs a label) does not fire — this
        // isolates the merge rule. social() is all-Person, so the rows match.
        let plan = Plan::Scan { label: None }
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
        // Unlabelled scan so seeding does not fire, isolating merge + pushdown.
        let plan = Plan::Scan { label: None }
            .expand(0, Dir::Out, &["KNOWS".to_string()])
            .filter(cmp(CompareOp::Le, prop(0, "age"), Expr::Lit(n(100.0))))
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
        // Unlabelled left scan so the pushed filter stays a Filter (not seeded),
        // which is what this test checks.
        let left = Plan::Scan { label: None }.expand(0, Dir::Out, &["KNOWS".to_string()]);
        let right = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["KNOWS".to_string()]);
        // left width is 2 (a, b); a filter on slot 0 (left's a) pushes left.
        let plan = Plan::join(left, right, vec![(0, 0)]).filter(cmp(
            CompareOp::Ge,
            prop(0, "age"),
            Expr::Lit(n(30.0)),
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
            | Plan::OptionalExpand { input, .. }
            | Plan::IntervalExpand { input, .. }
            | Plan::VarLength { input, .. }
            | Plan::ShortestPath { input, .. }
            | Plan::Aggregate { input, .. }
            | Plan::OrderPage { input, .. }
            | Plan::Project { input, .. }
            | Plan::Update { input, .. }
            | Plan::CallInline { input, .. }
            | Plan::Distinct { input }
            | Plan::Tail { input, .. }
            | Plan::Branch { input, .. }
            | Plan::SortLocal { input, .. } => plan_contains_filter(input),
            Plan::Join { left, right, .. } | Plan::Union { left, right, .. } => {
                plan_contains_filter(left) || plan_contains_filter(right)
            }
            Plan::InsertReturn { tail, .. } => plan_contains_filter(tail),
            Plan::Scan { .. }
            | Plan::NodeSeed { .. }
            | Plan::EdgeScan
            | Plan::Row
            | Plan::IndexSeek { .. }
            | Plan::RangeSeek { .. }
            | Plan::Insert { .. }
            | Plan::Merge { .. }
            | Plan::AddEdge { .. }
            | Plan::CallProcedure { .. } => false,
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

    /// `count(<bound element>)` over a pure chain canonicalizes to argument-free
    /// `count(*)` (same plan → same O(1) fast path), while `count(<property>)` and
    /// `count(DISTINCT …)` are left alone. The spelling-perf-cliff guard.
    #[test]
    fn count_of_bound_element_canonicalizes_to_count_star() {
        let store = social();
        let plan_str = |q: &str| format!("{:?}", optimize(crate::gql::parse(q).unwrap()));

        // count(n) == count(*) and count(b) == count(*) (1-hop), same optimized plan.
        assert_eq!(
            plan_str("MATCH (n:Person) RETURN count(n) AS c"),
            plan_str("MATCH (n:Person) RETURN count(*) AS c"),
        );
        assert_eq!(
            plan_str("MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b) AS c"),
            plan_str("MATCH (a:Person)-[:KNOWS]->(b) RETURN count(*) AS c"),
        );
        // A property count and a DISTINCT element count are NOT count(*).
        assert_ne!(
            plan_str("MATCH (n:Person) RETURN count(n.age) AS c"),
            plan_str("MATCH (n:Person) RETURN count(*) AS c"),
        );
        assert_ne!(
            plan_str("MATCH (n:Person) RETURN count(DISTINCT n) AS c"),
            plan_str("MATCH (n:Person) RETURN count(*) AS c"),
        );

        // And every one of these still returns the SAME rows before/after optimizing.
        for q in [
            "MATCH (n:Person) RETURN count(n) AS c",
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b) AS c",
            "MATCH (n:Person) RETURN count(n.age) AS c",
        ] {
            assert_rows_preserved(&crate::gql::parse(q).unwrap(), &store);
        }
    }

    /// A multi-predicate `WHERE a = x AND b = y` seeds ONE conjunct into a seek and
    /// keeps the rest as a residual filter — so it costs the same as the inline
    /// `(n:L {a: x, b: y})` twin (which already seeded), not a full-scan conjunction.
    /// The `count(*)`-shaped fixture is the one that showed the 34x gap.
    #[test]
    fn multi_predicate_where_seeds_a_conjunct() {
        let store = social();
        let seeded = |q: &str| {
            let p = optimize(crate::gql::parse(q).unwrap());
            format!("{p:?}").contains("IndexSeek")
        };
        // Both AND and the inline map seed; a lone equality already did.
        assert!(seeded(
            "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN count(*) AS c"
        ));
        assert!(seeded(
            "MATCH (n:Person {name: 'alice', age: 30}) RETURN count(*) AS c"
        ));
        // The two spellings optimize to the SAME plan (same seek + residual).
        assert_eq!(
            format!(
                "{:?}",
                optimize(
                    crate::gql::parse(
                        "MATCH (n:Person {name: 'alice', age: 30}) RETURN n.age AS a"
                    )
                    .unwrap()
                )
            ),
            format!(
                "{:?}",
                optimize(
                    crate::gql::parse(
                        "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN n.age AS a"
                    )
                    .unwrap()
                )
            ),
        );
        // And the rewrite never changes the rows (a 3-conjunct case too).
        for q in [
            "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN n.name AS x",
            "MATCH (n:Person) WHERE n.age >= 20 AND n.age <= 40 AND n.name = 'carol' RETURN n.name AS x",
        ] {
            assert_rows_preserved(&crate::gql::parse(q).unwrap(), &store);
        }
    }

    /// The multi-predicate seed is INDEX-AWARE: it seeds the conjunct backed by a
    /// real index (so the seek reads the index, not a scan), falling back to the
    /// first seekable conjunct only when nothing is indexed. The store-less
    /// `optimize` stays blind (unchanged behavior).
    #[test]
    fn multi_predicate_seed_prefers_the_indexed_conjunct() {
        let fresh = || {
            let mut b = Builder::default();
            for i in 0..40u32 {
                b.node(
                    &["Person"],
                    &[
                        ("dept", s(if i % 2 == 0 { "eng" } else { "sales" })),
                        ("age", n(f64::from(i % 10))),
                    ],
                );
            }
            b.build()
        };
        let q = "MATCH (n:Person) WHERE n.dept = 'eng' AND n.age = 4 RETURN n.age AS a";
        let plan = crate::gql::parse(q).unwrap();
        let seeded_key = |store: &Store| -> Option<String> {
            fn find(p: &Plan) -> Option<String> {
                match p {
                    Plan::IndexSeek { key, .. } => Some(key.clone()),
                    Plan::Filter { input, .. }
                    | Plan::Project { input, .. }
                    | Plan::Aggregate { input, .. } => find(input),
                    _ => None,
                }
            }
            find(&optimize_indexed(plan.clone(), store))
        };

        // No index → first seekable conjunct (dept).
        let mut plain = fresh();
        assert_eq!(seeded_key(&plain).as_deref(), Some("dept"));
        // Index on age → the seed switches to age (the indexed conjunct).
        plain.create_index("age");
        assert_eq!(seeded_key(&plain).as_deref(), Some("age"));
        // Index on dept instead → seeds dept.
        let mut dept_idx = fresh();
        dept_idx.create_index("dept");
        assert_eq!(seeded_key(&dept_idx).as_deref(), Some("dept"));
        // The store-less path is unchanged (blind): seeds the first conjunct.
        assert!(format!("{:?}", optimize(plan)).contains("key: \"dept\""));
    }

    /// Gremlin `order().by(k)` + `range(lo, hi)` lowers to two stacked OrderPages;
    /// merging the page into the sort must NOT change which rows come back — the
    /// delicate case is a TIE at the page boundary, where the surviving rows depend
    /// on tie-breaking. A fixture with many equal keys stresses exactly that.
    #[test]
    fn stacked_orderpage_merge_preserves_rows_under_ties() {
        let mut b = Builder::default();
        // 12 nodes, keys in {0,1,2} — heavy ties, so a top-k boundary lands inside a
        // tie group and the merged vs stacked forms must agree on which rows win.
        for i in 0..12u32 {
            b.node(
                &["P"],
                &[("k", n(f64::from(i % 3))), ("id", n(f64::from(i)))],
            );
        }
        let store = b.build();
        // Gremlin forms that produce stacked OrderPages (sort, then page).
        for q in [
            "g.V().hasLabel('P').order().by('k', desc).range(0, 2).values('id')",
            "g.V().hasLabel('P').order().by('k').range(1, 4).values('id')",
            "g.V().hasLabel('P').order().by('k', desc).range(2, 5).values('id')",
        ] {
            let plan = crate::gremlin::parse(q).unwrap();
            // Sanity: the UNoptimized plan really is two stacked OrderPages.
            assert!(
                matches!(&plan, Plan::Project { input, .. }
                    if matches!(input.as_ref(), Plan::OrderPage { input: inner, keys, .. }
                        if keys.is_empty() && matches!(inner.as_ref(), Plan::OrderPage { .. }))),
                "expected stacked OrderPages for `{q}`"
            );
            assert_rows_preserved(&plan, &store);
        }
    }
}
