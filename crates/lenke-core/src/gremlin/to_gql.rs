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

use crate::gql::plan::{
    compile_program, CAgg, CCount, CExpr, CProjection, CReturnItem, CSortItem, ScalarFn,
};
use crate::gremlin::{By, Order, Scope, Step};

/// How the projected columns become Gremlin values.
///
/// The columns are the same either way; what differs is the container, which is
/// exactly the axis the two languages disagree on and the reason this is the
/// caller's job rather than the evaluator's.
#[derive(Clone, PartialEq, Eq)]
pub(super) enum Shape {
    /// One value per row, from column 0.
    Rows,
    /// A single row holding one value — a global aggregate.
    Scalar,
    /// One `Map` from two columns: keys then counts. Gremlin's `groupCount`.
    Map,
    /// One `Map` per row from N columns, all sharing one key vector. Gremlin's
    /// `project(k1, k2, …)`.
    Maps { keys: Vec<String> },
    /// A single row holding a `List` of every value. Gremlin's `fold()`.
    List,
    /// The projected column is a per-row BOOL selecting rows of the INPUT element
    /// column — `where(…)` / `not(…)`. The shaper returns the surviving elements,
    /// not the bools, so navigation can continue off them.
    Retain { negated: bool },
}

pub(super) struct Tail {
    pub proj: CProjection,
    pub shape: Shape,
    /// The property `values(k)` read, when the output is that column.
    ///
    /// Gremlin DROPS a row whose key is absent and KEEPS one whose stored value is
    /// null. A projection erases which column a null came from, so the shaper has
    /// to re-ask the store: for a typed column the validity mask says exactly, a
    /// `Str`/`Temporal` column boxes an absence as null and cannot hold a stored
    /// one, and a `Mixed`/`Record` column can hold both — so that last case
    /// DECLINES rather than guessing.
    pub absent_key: Option<String>,
    /// A `[lo, hi)` window (`limit`/`skip`/`range`), applied by the CALLER AFTER
    /// `absent_key` has dropped its rows.
    ///
    /// It cannot live on `proj.skip`/`proj.limit`: `CProjection`'s own paging
    /// windows over the SCANNED rows (`project_frame_cols` pages `sc.n`, the raw
    /// frame, before any absent-row drop — that drop is Gremlin-only and happens
    /// in the caller's `shape_projection`), so a window installed there would
    /// include rows `values(k)` is about to drop and select the wrong slice. The
    /// route this replaces pages AFTER the drop too — `column_terminal` shapes
    /// the bare `values(k)` column first, then `col_terminal_tagged`'s
    /// `Limit`/`Skip`/`Range` arms call `Col::page` on what is left — so this
    /// mirrors that ordering exactly instead of the projection's.
    pub page: Option<(usize, usize)>,
    /// An `is(P)` narrowing the projected VALUE column, applied by the caller
    /// after the absent-key drop and before any window.
    ///
    /// Not put into the plan as a WHERE on the element, though `values(k).is(P)`
    /// and `WHERE u.k P` select the same rows: an `is(P)` over a non-number column
    /// is a type FAULT in Gremlin rather than a skipped row, and the shaper is
    /// where that decision already lives (`num_test`, and its NaN rule). Keeping
    /// it there means one implementation of the rule, not two that must agree.
    ///
    /// Aggregates therefore DECLINE this arm: GQL would fold the column before the
    /// caller ever sees it, so a filter applied afterwards would be too late.
    pub filter: Option<crate::gremlin::P>,
    /// The property `order().by(k)` sorts on, when the tail sorts.
    ///
    /// Gremlin's ORDERABILITY is narrower than GQL's: comparing two genuinely
    /// incomparable non-null values (a number against a string on the same
    /// schemaless key) is a recorded TYPE FAULT that `try_run` surfaces
    /// (`order_by_mixed_type_property_faults_not_panics`), while GQL's `ORDER
    /// BY` never faults — it total-orders by type rank instead
    /// (`gql::eval::cmp_total`). A projection can't tell the caller which pairs
    /// it compared, so the shaper re-asks the store: a `Num`/`Bool`/`Str`/
    /// `Temporal` column (or no column at all) is homogeneous by construction,
    /// so GQL's silent total order and Gremlin's fault-free comparator agree on
    /// every pair — but `Mixed`/`Record`/`Vec` CAN hold an incomparable pair
    /// GQL would happily rank and Gremlin must fault on, so that case DECLINES
    /// rather than silently dropping the fault.
    pub order_key: Option<String>,
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

/// The `[lo, hi)` window a global paging step selects, in the same shape
/// `Col::page(lo, hi)` takes — `limit(n)` is `[0, n)`, `skip(n)` is `[n, ∞)`
/// (`usize::MAX` stands in for "to the end", exactly as `col_terminal_tagged`'s
/// own `Skip` arm spells it), and `range(lo, hi)` is that pair verbatim. `Local`
/// scope is a per-row slice of a LIST value, not a row window, and stays `None`.
fn page_of(step: &Step) -> Option<(usize, usize)> {
    match step {
        Step::Limit(n, Scope::Global) => Some((0, *n)),
        Step::Skip(n, Scope::Global) => Some((*n, usize::MAX)),
        Step::Range(lo, hi, Scope::Global) => Some((*lo, *hi)),
        _ => None,
    }
}

/// The label's index in `label_names`, appending it if unseen — the same
/// plan-time interning `key_ref` does for properties. `CLabelExpr::Label(r)`
/// refers to a label by this index, and `project_ids` resolves the whole list
/// once against the graph's dictionaries.
fn label_ref(labels: &mut Vec<String>, name: &str) -> usize {
    labels.iter().position(|n| n == name).unwrap_or_else(|| {
        labels.push(name.to_string());
        labels.len() - 1
    })
}

/// A `where(…)`/`not(…)` body that is ONE adjacency hop, as a GQL `EXISTS`.
///
/// Mirrors the shapes `gql::eval::exists_semi_join_vec` can answer in bulk — a
/// single untyped-or-typed hop off the bound slot, optionally landing on a label.
/// Anything richer (a chain, a property test, an inner predicate) returns `None`
/// and Gremlin's own `semi_join_hop` keeps it.
fn exists_of(slot: usize, labels: &mut Vec<String>, body: &[Step]) -> Option<CExpr> {
    use crate::gql::ast::Direction;
    use crate::gql::plan::{CLabelExpr, CNode, CPath, CRel, CSegment};

    let (dir, names, tail) = match body {
        [Step::Out(l), t @ ..] => (Direction::Out, l, t),
        [Step::In(l), t @ ..] => (Direction::In, l, t),
        [Step::Both(l), t @ ..] => (Direction::Both, l, t),
        _ => return None,
    };
    // One type only: a multi-name hop is a disjunction over edge types, which
    // `CLabelExpr::Label` does not spell and `exists_semi_join_vec` would have to
    // widen to accept.
    let rel_label = match names.as_slice() {
        [] => None,
        [one] => Some(CLabelExpr::Label(label_ref(labels, one))),
        _ => return None,
    };
    // An optional `hasLabel` on the LANDED node, same restriction.
    let node_label = match tail {
        [] => None,
        [Step::HasLabel(ls)] => match ls.as_slice() {
            [one] => Some(CLabelExpr::Label(label_ref(labels, one))),
            _ => return None,
        },
        _ => return None,
    };
    let node = |label| CNode {
        var_slot: None,
        label,
        props: Vec::new(),
        where_: None,
    };

    Some(CExpr::Exists {
        patterns: vec![CPath {
            start: CNode {
                var_slot: Some(slot),
                ..node(None)
            },
            segments: vec![CSegment {
                rel: CRel {
                    var_slot: None,
                    label: rel_label,
                    direction: dir,
                    props: Vec::new(),
                    where_: None,
                    quantifier: None,
                },
                node: node(node_label),
                unit: None,
            }],
            path_var_slot: None,
            selector: crate::gql::ast::PathSelector::Walk,
            mode: crate::gql::ast::PathMode::Trail,
        }],
        where_: None,
        sub_len: 0,
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
/// `is_edge` is whatever the CALLER already knows about the frontier's element
/// kind (`c.end_is_edge` off the pattern, or the `is_edge` a `Col::Elems` already
/// carries) — every entry point has it in scope, so threading it in costs
/// nothing new. `label()` is the one arm that needs it: it is not answerable at
/// all for a vertex (see that arm's comment).
///
/// `None` = not expressible; the caller keeps its existing route.
pub(super) fn tail(
    slot: usize,
    is_edge: bool,
    keys: &mut Vec<String>,
    labels: &mut Vec<String>,
    rest: &[Step],
) -> Option<Tail> {
    match rest {
        // `where(body)` / `not(body)` — a per-row EXISTS over the bound slot,
        // projected as a bool column the shaper uses to RETAIN input elements.
        //
        // Worth routing here rather than leaving to `semi_join_hop`: GQL's
        // `exists_semi_join_vec` answers this as a bulk adjacency test and prices
        // the same question 2.8-3.6x FASTER than Gremlin's own arm
        // (`where_not_typed_hop_timing_probe`). A vertex frontier only — the arm
        // this defers to walks a VERTEX's adjacency, and an edge id read as one
        // walks whatever vertex shares its number.
        [head @ (Step::Where(sub) | Step::Not(sub))] if !is_edge => {
            let expr = exists_of(slot, labels, &sub.steps)?;

            Some(Tail {
                proj: blank(vec![item(expr, "keep", false)]),
                shape: Shape::Retain {
                    negated: matches!(head, Step::Not(_)),
                },
                absent_key: None,
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `id()` — the element's external id. `element_id(n)` is GQL's own
        // scalar function, and `crate::value::Value::element_id` is the ONE
        // implementation both engines call — `exec::elem_id` here, `ElementId`
        // in `gql/eval/scalar_fns.rs` (that arm's comment: "Shared with
        // Gremlin's `id()`") — so the two renderings cannot drift, for either
        // element kind.
        [Step::Id] => Some(Tail {
            proj: blank(vec![item(
                CExpr::Scalar {
                    func: ScalarFn::ElementId,
                    args: vec![CExpr::Var(slot)],
                },
                "v",
                false,
            )]),
            shape: Shape::Rows,
            absent_key: None,
            page: None,
            filter: None,
            order_key: None,
        }),
        // `label()` — the element's label, but ONLY for a known EDGE frontier.
        //
        // GQL's `type(e)` is openCypher's edge-type accessor, and its own doc
        // comment in `scalar_fns.rs` says it directly: "type stays SINGULAR...
        // exactly as Gremlin's label() reports a multi-label vertex's first
        // label — both have to return one." For an edge that is exactly
        // `exec::elem_label`'s rendering (`graph.etype.arc(e_type[e])`), so the
        // two agree.
        //
        // For a VERTEX there is no equivalent. GQL's only other candidate,
        // `labels(n)`, returns EVERY label — SORTED, as a `List` — while
        // `exec::elem_label` reads `graph.vertex_labels(v).first()`, which is
        // `Graph::vlabels`, a vector in INSERTION order, not sorted. For a
        // vertex carrying more than one label those disagree in both VALUE
        // (first-inserted vs. alphabetically-first) and TYPE (`Str`/`Null` vs.
        // `List`) — not a close match, a different question. So this arm
        // declines whenever `is_edge` is false, and the caller's existing
        // `elem_terminal` arm (`exec.rs`) answers a vertex `label()` instead.
        [Step::Label] if is_edge => Some(Tail {
            proj: blank(vec![item(
                CExpr::Scalar {
                    func: ScalarFn::Type,
                    args: vec![CExpr::Var(slot)],
                },
                "v",
                false,
            )]),
            shape: Shape::Rows,
            absent_key: None,
            page: None,
            filter: None,
            order_key: None,
        }),
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
                absent_key: Some(ks[0].clone()),
                page: None,
                filter: None,
                order_key: None,
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
                // An aggregate SKIPS nulls itself; there is no row to drop.
                absent_key: None,
                page: None,
                filter: None,
                order_key: None,
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
                absent_key: Some(ks[0].clone()),
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `outV()` / `inV()` off an EDGE frontier — a gather to the endpoint,
        // yielding an ELEMENT column so navigation continues off it. `bothV()` is
        // NOT here: it emits BOTH ends per edge, which is a row-count change no
        // projection expresses (one input row, two output rows).
        [step @ (Step::OutV | Step::InV)] if is_edge => {
            let expr = CExpr::Scalar {
                func: if matches!(step, Step::OutV) {
                    crate::gql::plan::ScalarFn::EdgeSource
                } else {
                    crate::gql::plan::ScalarFn::EdgeTarget
                },
                args: vec![CExpr::Var(slot)],
            };

            Some(Tail {
                proj: blank(vec![item(expr, "v", false)]),
                shape: Shape::Rows,
                absent_key: None,
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `groupCount().by(k)` — GROUP BY the property, count each group. Both
        // engines emit groups in FIRST-SEEN order, so the sequence matches without
        // an ORDER BY. A `by()` carrying a direction sorts the result and is a
        // different step; `single_key_by` refuses one and so does this.
        [Step::GroupCount(bys)] => {
            let [crate::gremlin::By::Key(k, None)] = bys.as_slice() else {
                return None;
            };
            let kr = key_ref(keys, k);
            let key_expr = CExpr::Prop {
                var_slot: slot,
                key_ref: kr,
            };
            let mut proj = blank(vec![
                item(key_expr.clone(), "k", false),
                item(CExpr::AggRef(0), "c", true),
            ]);

            proj.aggregating = true;
            proj.group_by = vec![item(key_expr, "k", false)];
            proj.aggs = vec![CAgg {
                func: crate::gql::plan::AggFn::Count,
                arg: None,
                distinct: false,
                star: true,
                frac: None,
            }];

            Some(Tail {
                proj,
                shape: Shape::Map,
                // A tally keys on the value, and an absent key tallies under a
                // NULL key rather than dropping the row — TinkerPop 3.5, and what
                // the arm this replaces does.
                absent_key: None,
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `group().by(k).by(values(v).<sum|min|max|mean>())` — GROUP BY property
        // `k`, with the reducer as an aggregate over property `v`, one row per
        // group. The same shape as `groupCount().by(k)` below (`Shape::Map`),
        // except the second column is a REDUCED VALUE instead of a per-group
        // count. `exec::grouped_reduce` is the SAME guard the stream arm this
        // replaces uses (`exec.rs::elem_terminal`'s `[Step::Group(bys)]` arm) —
        // reused, not re-derived, so the two admissibility rules (which `by()`
        // pairs, which reducers) cannot drift apart.
        //
        // `count()` is deliberately absent from `grouped_reduce`'s reducers, and
        // this arm inherits that: `CExpr::Prop` reads an absent key and a stored
        // null as the same `Val::Null`, which is exactly right for `sum`/`min`/
        // `max`/`mean` (TinkerPop 3.5 "ignore null values when other numbers are
        // present") and WRONG for a count, where an absent key contributes
        // nothing and a stored null contributes one.
        [Step::Group(bys)] if let Some((kkey, vkey, red)) = super::exec::grouped_reduce(bys) => {
            let kr = key_ref(keys, kkey);
            let vr = key_ref(keys, vkey);
            let key_expr = CExpr::Prop {
                var_slot: slot,
                key_ref: kr,
            };
            let val_expr = CExpr::Prop {
                var_slot: slot,
                key_ref: vr,
            };
            let mut proj = blank(vec![
                item(key_expr.clone(), "k", false),
                item(CExpr::AggRef(0), "c", true),
            ]);

            proj.aggregating = true;
            proj.group_by = vec![item(key_expr, "k", false)];
            proj.aggs = vec![CAgg {
                func: red.as_agg_fn(),
                arg: Some(val_expr),
                distinct: false,
                star: false,
                frac: None,
            }];

            Some(Tail {
                proj,
                shape: Shape::Map,
                // The GROUP key tallies like `groupCount().by(k)` — an absent
                // key groups under a NULL key rather than dropping the row. The
                // VALUE argument's own absent/null rows are skipped by the
                // aggregate itself (an `Agg` never drops a ROW, only an
                // argument), so there is no row-drop for `absent_key` to do
                // here either.
                absent_key: None,
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `values(k).count()` — the count of rows whose key is NOT ABSENT.
        // Gremlin's `values(k)` drops an absent row before `count()` ever sees
        // it, so this is `count` over the SURVIVING rows, not `count(*)` over
        // the frontier.
        //
        // GQL's `count(expr)` (non-star) already skips a `Null` argument the
        // same way its `sum`/`min`/`max`/`avg` cousins do — every aggregate path
        // (`AggValue::step` in `matcher.rs`, `eval_aggregate` in `eval.rs`,
        // `fused_global_agg`'s "count(prop): number of present values" in
        // `scan.rs`) filters on `is_nullish`/a presence mask before counting.
        // For a column that cannot hold a STORED null next to an ABSENT row —
        // `Num`/`Bool`/`Str`/`Temporal`, or no column at all — `Val::Null` can
        // only mean "absent", so `count(prop)`'s skip and Gremlin's drop agree
        // on every row. A `Mixed`/`Record` column CAN hold a genuine stored null
        // beside an absent row and boxes both the same way, so `count(prop)`
        // cannot tell "skip because absent" from "skip because a stored value
        // happens to be null" — declines through `absent_key`'s homogeneity
        // check in `shape_projection` (`homogeneous_or_absent`, the same check
        // `order_key` already uses) rather than assume.
        [Step::Values(ks), Step::Count(Scope::Global)] if ks.len() == 1 => {
            let kr = key_ref(keys, &ks[0]);
            let mut proj = blank(vec![item(CExpr::AggRef(0), "v", true)]);

            proj.aggregating = true;
            proj.aggs = vec![CAgg {
                func: crate::gql::plan::AggFn::Count,
                arg: Some(CExpr::Prop {
                    var_slot: slot,
                    key_ref: kr,
                }),
                distinct: false,
                star: false,
                frac: None,
            }];

            Some(Tail {
                proj,
                shape: Shape::Scalar,
                // Not a row-drop here (an aggregate collapses to one row before
                // any drop could apply) — `shape_projection` reads this as "the
                // column `count(...)` read from", to decline when that column's
                // type cannot be trusted to agree with Gremlin's absent/null
                // split (see the arm's own comment above).
                absent_key: Some(ks[0].clone()),
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `values(k).is(P)` — the projected column, NARROWED. Optionally paged,
        // in that order: `column_terminal` applies its `is` before handing the tail
        // on, so a window must see the filtered column, not the raw one.
        //
        // This is the shape that keeps `elem_terminal`'s `[Step::Values(keys),
        // tail @ ..]` arm alive — `values_arm_reachability_probe` walks eight
        // traversals and this is the only one that reaches it.
        [Step::Values(ks), Step::Is(p), after @ ..]
            if ks.len() == 1
                && matches!(after, [] | [_] if after.first().is_none_or(|s| page_of(s).is_some())) =>
        {
            let kr = key_ref(keys, &ks[0]);
            let page = after.first().and_then(page_of);

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
                absent_key: Some(ks[0].clone()),
                page,
                filter: Some(p.clone()),
                order_key: None,
            })
        }
        // `values(k).<limit|skip|range>()` — a `[lo, hi)` window over the column,
        // taken AFTER the absent-key drop (see `Tail::page`). `page_of` maps each
        // step to the same `(lo, hi)` pair `Col::page` itself takes, so the three
        // spellings and their edge cases (`limit(0)`, `range` with `hi <= lo`, `hi`
        // past the row count) all clamp exactly the way `col.page` already does —
        // there is no separate clamp to keep in sync.
        [Step::Values(ks), page_step] if ks.len() == 1 && page_of(page_step).is_some() => {
            let kr = key_ref(keys, &ks[0]);
            let (lo, hi) = page_of(page_step)?;

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
                absent_key: Some(ks[0].clone()),
                page: Some((lo, hi)),
                filter: None,
                order_key: None,
            })
        }
        // `values(k).fold()` — one row holding a `List` of every value, after the
        // same absent-key drop the bare form performs (a fold does not resurrect a
        // dropped row; it just never sees one).
        [Step::Values(ks), Step::Fold] if ks.len() == 1 => {
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
                shape: Shape::List,
                absent_key: Some(ks[0].clone()),
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `project('a','b').by('k1').by('k2')` — one resolved property column
        // per key, zipped into one `Map` per row that all share one key
        // vector (`Shape::Maps`, shaped in `exec::shape_projection`). Only the
        // modulators that are a plain property or the element itself: a
        // sub-traversal `by()` can read the path and a token `by()` projects
        // something that is not a column, so both keep the stream form —
        // `projectable_bys` is the SAME guard the arm this replaces uses, so
        // the two cannot drift on which `by()`s are admissible.
        //
        // Unlike `values(k)`, an absent key is NOT dropped here — `project()`
        // holds a `null` for it, so `absent_key` stays `None`.
        [Step::Project(pkeys, bys)]
            if !pkeys.is_empty() && super::exec::projectable_bys(pkeys, bys) =>
        {
            let items = (0..pkeys.len())
                .map(|i| {
                    let expr = match bys.get(i) {
                        Some(By::Key(k, _)) => CExpr::Prop {
                            var_slot: slot,
                            key_ref: key_ref(keys, k),
                        },
                        // Missing `by()`, or an explicit `by()` (identity) —
                        // both project the element itself.
                        _ => CExpr::Var(slot),
                    };

                    item(expr, &format!("v{i}"), false)
                })
                .collect::<Vec<_>>();

            Some(Tail {
                proj: blank(items),
                shape: Shape::Maps {
                    keys: pkeys.clone(),
                },
                // `project()` holds a NULL for an absent key rather than dropping
                // the row, and it does not page.
                absent_key: None,
                page: None,
                filter: None,
                order_key: None,
            })
        }
        // `order().by(k)` [`.limit(n)`] — ORDER BY the property, keep the
        // ELEMENTS (unlike `values(k)`, the output item is the element itself).
        //
        // The sort key reads a property off the INPUT slot, which the output row
        // does not carry (it carries the element, not the property) — so it has
        // to be published through `order_overlay`, the extra scope ORDER BY
        // resolves against (`CProjection::order_overlay`, `gql/plan.rs`). The
        // FIRST attempt pointed the sort key straight at the raw input slot
        // (`var_slot: slot`) and sorted WRONG: a vertex with `n = 0` sorted after
        // one with `n = 10`. The reason is that `order_by` expressions are
        // resolved against the SORT scope, not the input scope — slots
        // `0..out_len` name an output column, `out_len..` name the overlay's
        // input slots in order (mirrors what `plan.rs::projection` builds for a
        // real `MATCH … RETURN … ORDER BY`: `sort_scope` = output columns, then
        // `order_overlay`). With one output item (`out_len == 1`) and the
        // property's only overlay entry at position 0, the sort key's slot is
        // `out_len + 0`, not `slot`.
        //
        // Gremlin's null placement is the MINIMUM of the order, not a pinned end:
        // it sorts first ascending and last descending, because `gcmp_total`
        // ranks `Null` below every other value and the whole comparator then
        // reverses for `desc` (`ordering_places_nulls_first_without_faulting`).
        // GQL's `nulls_first` is the opposite shape — an ABSOLUTE placement,
        // independent of `descending` (`compare_sort` in `gql/eval/statement.rs`)
        // — so matching Gremlin's "first ascending, last descending" means
        // flipping the flag WITH the direction: `Some(!descending)`, not the
        // unconditional `Some(true)` a nulls-are-always-first reading suggests.
        //
        // The trailing `limit(n)` goes on `proj.limit`/`CCount::Lit`, NOT
        // `Tail::page`, and that is a DELIBERATE choice, not the default for a
        // paging step: `Tail::page` exists because `CProjection`'s own paging
        // windows over the raw SCANNED rows, before Gremlin's absent-key drop —
        // wrong when a drop is still to come. This arm's `absent_key` is always
        // `None` (`order().by(k)` sorts an absent key to a null, it never drops
        // the row), so there is no drop for a window to race, and with
        // `order_by` non-empty `proj.window` slices the SORTED index, not the
        // scan (`gql::eval::CProjection::window`'s doc: "a sorted one [windows]
        // over the sorted index") — exactly `ORDER BY … LIMIT n`'s ISO meaning,
        // and it keeps GQL's top-k partial-sort shortcut (`keep_smallest`)
        // instead of a full sort + slice `Tail::page` would fall back to.
        [Step::Order(bys, desc, Scope::Global), after @ ..]
            if matches!(after, [] | [Step::Limit(_, Scope::Global)]) =>
        {
            let [By::Key(k, dir)] = bys.as_slice() else {
                return None;
            };
            let kr = key_ref(keys, k);
            let descending = dir.map_or(*desc, |d| d == Order::Desc);
            let items = vec![item(CExpr::Var(slot), "v", false)];
            let out_len = items.len();
            let mut proj = blank(items);

            proj.order_overlay = vec![slot];
            proj.order_by = vec![CSortItem {
                expr: CExpr::Prop {
                    var_slot: out_len,
                    key_ref: kr,
                },
                descending,
                nulls_first: Some(!descending),
            }];
            // The key is entirely from the overlay (slot `out_len`, never < it),
            // so `order_needs_output` stays false — matches what the planner
            // computes via `refs_slot_below(&sort_expr, out_len)` for this shape.
            proj.order_needs_output = false;

            if let [Step::Limit(n, Scope::Global)] = after {
                proj.limit = Some(CCount::Lit(*n));
            }

            Some(Tail {
                proj,
                shape: Shape::Rows,
                // The row is KEPT with a null key, not dropped — `order().by(k)`
                // reads the property like `eval_by`/`prop` do (absent → `Null`,
                // sorted, not filtered), unlike `values(k)`'s drop-on-absent.
                absent_key: None,
                page: None,
                filter: None,
                order_key: Some(k.clone()),
            })
        }
        // `elementMap(keys)` was investigated and has NO arm here — it cannot be
        // expressed as a `CProjection` at all, for two independent reasons, not
        // one repaired by the other:
        //
        // 1. Its key set is PER-ROW. `exec::element_map_of` (via
        //    `projected_keys_into`) emits an entry only for a key that is
        //    PRESENT on that element — `keys` empty means "every present key",
        //    and even an explicit list SKIPS an absent one rather than nulling
        //    it. A heterogeneous frontier (some elements missing `m`, say) then
        //    yields maps with DIFFERENT key sets per row. `Shape::Maps` (the
        //    container `project()` uses) requires one shared key vector for
        //    every row — `CProjection::items` is a fixed list of N columns,
        //    compiled once — so there is no way to compile "however many keys
        //    this particular element happens to have" into it. `CExpr::Record`
        //    has the same problem the other way: its field list is fixed at
        //    COMPILE time, so it cannot omit a field only for the rows where the
        //    key is absent.
        // 2. An edge's map carries two NESTED maps (`IN`/`OUT`, each `{id,
        //    label}` of an endpoint) that neither `Shape` nor any `CExpr` can
        //    produce — nesting a `Maps`-shaped row inside a single cell of
        //    another isn't a shape this IR has.
        //
        // Either alone would be enough to decline; both apply. `exec.rs`'s
        // `[Step::ElementMap(keys)]` arm keeps this one.
        _ => None,
    }
}
