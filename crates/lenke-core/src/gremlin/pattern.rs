//! A Gremlin linear prefix, compiled to the shared PATTERN IR.
//!
//! `g.V().out('R').hasLabel('W')` and `MATCH ()-[:R]->(b:W)` are the same query.
//! Read as a step list the first executes left to right — so a filter landing
//! after the hop cannot inform the seed, and every vertex is expanded before
//! anything is discarded. Read as a PATTERN it is planned as a whole:
//! `try_orient_node_seed` sees the selective end is `b`, seeds there, and walks
//! the adjacency backwards. On a 50k/150k graph that is 138x.
//!
//! So this does not add a shortcut. It hands Gremlin the planner it never had,
//! by translating into the one GQL already uses:
//!
//! ```text
//!   g.V().hasLabel('P').has('k', 1).out('R').hasLabel('Q')
//!         └── start CNode ─────────┘  └seg rel┘ └seg node┘
//! ```
//!
//! Only the shapes where the translation is EXACT are accepted; everything else
//! returns `None` and the caller streams as before. In particular a predicate
//! that is not a plain equality stays out, because `has(k, gt(5))` on a string
//! column is a type FAULT in Gremlin and three-valued unknown in GQL — the same
//! reason `ElementSeek::columnar` declines it.

use crate::gql::ast::Lit;
use crate::gql::ast::{Direction, PathMode, PathSelector};
use crate::gql::plan::{CExpr, CLabelExpr, CNode, CPath, CPropConstraint, CRel, CSegment};
use crate::gremlin::{GVal, Step, P};
#[cfg(feature = "bailprobe")]
use crate::pipeline::{OpClass, Route};

/// A compiled prefix: the pattern, the interning tables its refs index into, and
/// how many steps were consumed.
pub(super) struct Compiled {
    pub path: CPath,
    pub label_names: Vec<String>,
    pub key_names: Vec<String>,
    /// Slot the last node binds to — the traverser's value after the prefix.
    pub end_slot: usize,
    pub scope_len: usize,
    pub consumed: usize,
}

#[derive(Default)]
struct Interner {
    labels: Vec<String>,
    keys: Vec<String>,
}

impl Interner {
    fn label(&mut self, name: &str) -> usize {
        self.intern(name, true)
    }
    fn key(&mut self, name: &str) -> usize {
        self.intern(name, false)
    }
    fn intern(&mut self, name: &str, is_label: bool) -> usize {
        let v = if is_label {
            &mut self.labels
        } else {
            &mut self.keys
        };

        v.iter().position(|n| n == name).unwrap_or_else(|| {
            v.push(name.to_string());
            v.len() - 1
        })
    }
}

/// A `hasLabel` as a label expression: one name is a `Label`, several are the
/// `Or` chain GQL builds for `:A|B`. An EMPTY list matches nothing, which no
/// label expression spells, so it declines.
fn label_expr(names: &[String], it: &mut Interner) -> Option<CLabelExpr> {
    let mut out: Option<CLabelExpr> = None;

    for n in names {
        let one = CLabelExpr::Label(it.label(n));

        out = Some(match out {
            None => one,
            Some(prev) => CLabelExpr::Or(Box::new(prev), Box::new(one)),
        });
    }

    out
}

/// A literal `GVal` as a GQL literal. Only the scalars a property can hold and
/// an index can seek; anything else declines.
fn lit_of(v: &GVal) -> Option<Lit> {
    match v {
        GVal::Null => Some(Lit::Null),
        GVal::Bool(b) => Some(Lit::Bool(*b)),
        GVal::Num(n) => Some(Lit::Num(*n)),
        GVal::Str(s) => Some(Lit::Str(s.to_string())),
        GVal::Temporal(t) => Some(Lit::Temporal(*t)),
        _ => None,
    }
}

/// Fold the `has`/`hasLabel` steps at the front of `steps` into `node`, and
/// return how many were consumed.
///
/// Only EQUALITY is folded. A range or set predicate is expressible as a
/// `where_` on the node, but `has('k', gt(5))` over a string column faults in
/// Gremlin and is unknown in GQL, so translating it would change which queries
/// throw — the divergence `ElementSeek::columnar` already declines to avoid.
fn absorb_filters(steps: &[Step], node: &mut CNode, it: &mut Interner) -> usize {
    let mut n = 0;

    for step in steps {
        match step {
            Step::HasLabel(names) if node.label.is_none() => match label_expr(names, it) {
                Some(e) => node.label = Some(e),
                None => break,
            },
            Step::Has(key, P::Eq(v)) => match lit_of(v) {
                Some(l) => node.props.push(CPropConstraint {
                    key: key.clone(),
                    key_ref: it.key(key),
                    value: CExpr::Lit(l),
                }),
                None => break,
            },
            _ => break,
        }

        n += 1;
    }

    n
}

/// The direction and type names of a hop step, or `None` if it is not one.
fn hop_of(step: &Step) -> Option<(Direction, &[String])> {
    match step {
        Step::Out(l) => Some((Direction::Out, l)),
        Step::In(l) => Some((Direction::In, l)),
        Step::Both(l) => Some((Direction::Both, l)),
        _ => None,
    }
}

/// Compile the longest prefix of `steps` that is a plain pattern.
///
/// Requires a bare `V()` start (an id-seeded one is already a point lookup) and
/// at least one hop — with none the caller's own seeding path is what runs, and
/// this would only add a translation.
pub(super) fn compile(steps: &[Step]) -> Option<Compiled> {
    let [Step::V(ids), rest @ ..] = steps else {
        return None;
    };

    if !ids.is_empty() {
        return None;
    }

    let mut it = Interner::default();
    let mut consumed = 1;
    let mut start = CNode {
        var_slot: Some(0),
        label: None,
        props: Vec::new(),
        where_: None,
    };
    let taken = absorb_filters(rest, &mut start, &mut it);
    consumed += taken;

    let mut segments: Vec<CSegment> = Vec::new();
    let mut slot = 0;

    while let Some((dir, types)) = steps.get(consumed).and_then(hop_of) {
        consumed += 1;
        slot += 1;

        let mut node = CNode {
            var_slot: Some(slot),
            label: None,
            props: Vec::new(),
            where_: None,
        };

        consumed += absorb_filters(&steps[consumed..], &mut node, &mut it);
        segments.push(CSegment {
            rel: CRel {
                var_slot: None,
                label: if types.is_empty() {
                    None
                } else {
                    label_expr(types, &mut it)?.into()
                },
                direction: dir,
                props: Vec::new(),
                where_: None,
                quantifier: None,
            },
            node,
            unit: None,
        });
    }

    if segments.is_empty() {
        return None;
    }

    // Decline unless some node PAST the start is constrained.
    //
    // That is exactly the condition under which planning can contribute: the
    // planner's whole job here is to seed the selective end, and when the only
    // constraints sit on the start, the selective end IS the start and written
    // order already seeds it. Compiling anyway costs a full pattern scan and a
    // multi-slot frame to arrive at the plan the step list already had —
    // measured 1.28x on `V().hasLabel(P).out(KNOWS).values(name)`, where the
    // caller's own expand does the same walk with no frame at all.
    //
    // Note this asks about NODES, not the edge type: both paths already push a
    // type down into the adjacency, so a typed hop is not something to orient
    // toward.
    let can_orient = segments
        .iter()
        .any(|s| s.node.label.is_some() || !s.node.props.is_empty());

    if !can_orient {
        return None;
    }

    Some(Compiled {
        path: CPath {
            start,
            segments,
            path_var_slot: None,
            selector: PathSelector::Walk,
            mode: PathMode::Trail,
        },
        label_names: it.labels,
        key_names: it.keys,
        end_slot: slot,
        scope_len: slot + 1,
        consumed,
    })
}

/// Classify a Gremlin step for [`crate::pipeline`].
///
/// NOT load-bearing at runtime, and behind the probe feature for that reason.
/// The boundary analysis was written to be the first routing decision, and then
/// offering the planned ids to the column terminals turned out to subsume it:
/// "try the column path, else stream" answers the same question without asking
/// it, because a terminal that wants a column is exactly a boundary. It stays
/// because the classification is what measures the corpus — 55% Stream, 18%
/// Columnar, 26% Decline — and because the general case it was written for is
/// still coming: a boundary whose tail `column_paths` declines currently streams,
/// where a frame would serve it better.
///
/// The per-language half of the routing decision: only this engine knows what
/// its steps do, and only the shared rule knows what to do about it. The
/// question each answer is about is MEMORY — does this step need every row at
/// once — not what the step is called.
#[cfg(feature = "bailprobe")]
pub(super) fn class_of(step: &Step) -> OpClass {
    match step {
        // A row in, zero or more out. Filters, hops, and per-row projections.
        Step::V(_)
        | Step::E(_)
        | Step::Has(..)
        | Step::HasLabel(..)
        | Step::HasNot(..)
        | Step::HasKey(..)
        | Step::HasId(..)
        | Step::HasValue(..)
        | Step::Is(_)
        | Step::Out(..)
        | Step::In(..)
        | Step::Both(..)
        | Step::OutE(..)
        | Step::InE(..)
        | Step::BothE(..)
        | Step::InV
        | Step::OutV
        | Step::OtherV
        | Step::BothV
        | Step::Values(..)
        | Step::Id
        | Step::Label
        | Step::Value
        | Step::Properties(..)
        | Step::Constant(_)
        | Step::Identity
        | Step::None(..)
        | Step::Limit(..)
        | Step::Skip(..)
        | Step::Range(..)
        | Step::Unfold => OpClass::Streaming,

        // Consumes every row, emits one, holds no buffer to do it. A streaming
        // fold beats materializing a frame first.
        Step::Count(_) | Step::Sum(_) | Step::Mean(_) | Step::Min(_) | Step::Max(_) => {
            OpClass::Reducing
        }

        // Cannot emit until every row exists. THE boundary — the rows are being
        // materialized whatever runs them, so a column is free at this point.
        Step::Order(..)
        | Step::Dedupe { .. }
        | Step::Group(..)
        | Step::GroupCount(..)
        | Step::Fold
        | Step::Tail(..)
        | Step::Sample(_)
        | Step::Barrier => OpClass::Buffering,

        // Carries per-row state the shared layer cannot model: a path, a sack, a
        // tag, a branch, or a sub-traversal that could contain any of them.
        _ => OpClass::Opaque,
    }
}

/// The route this traversal should take, and where its first boundary is.
#[cfg(feature = "bailprobe")]
pub(super) fn shape(steps: &[Step]) -> (Route, Option<usize>) {
    let classes: Vec<OpClass> = steps.iter().map(class_of).collect();

    (
        crate::pipeline::route(&classes),
        crate::pipeline::first_boundary(&classes),
    )
}
