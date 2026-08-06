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
use crate::gremlin::{GVal, Scope, Step, P};

/// A compiled prefix: the pattern, the interning tables its refs index into, and
/// how many steps were consumed.
pub(super) struct Compiled {
    pub path: CPath,
    pub label_names: Vec<String>,
    pub key_names: Vec<String>,
    /// Slot the last element binds to — the traverser's value after the prefix.
    pub end_slot: usize,
    /// Whether that slot holds an EDGE (`outE('R')` with no landing step).
    pub end_is_edge: bool,
    /// `as(label)` bindings the prefix absorbed: label → (slot, is_edge).
    ///
    /// A tag IS a slot. Gremlin accumulates a LIST per label because a label
    /// under `repeat()` is visited once per iteration, and `select(Pop.all)`
    /// reads all of them — but a linear prefix visits each label exactly once, so
    /// the list has one entry and a `var_slot` carries it exactly.
    pub tags: Vec<(String, usize, bool)>,
    pub scope_len: usize,
    pub consumed: usize,
    /// Whether the plan enumerates in a DIFFERENT order than the streamed
    /// traversal would.
    ///
    /// Set only by the `g.E()` desugar. `g.E()` visits edges in id order and the
    /// plan walks each vertex's adjacency instead — the same rows, a different
    /// sequence. A `count()` cannot tell; `g.E().inV().hasLabel('P')` can, and the
    /// TS engine keeps the streamed order, so taking this branch there would
    /// break byte-identity. The caller must check
    /// [`order_insensitive`] before using a reordering plan.
    ///
    /// A `V()` prefix never sets it: the streamed traversal walks vertex by
    /// vertex too, which is why those lower with the order intact.
    pub reorders: bool,
}

/// Whether `rest` can observe the ORDER of the rows handed to it.
///
/// Deliberately a short allowlist rather than a "does it look ordered" test: a
/// wrong answer here is a silently reordered result, and the shapes worth
/// reordering for are the reducing terminals. `fold`, `limit`, `range`, `tail`
/// and a bare projection all read the sequence, so none of them are here.
pub(super) fn order_insensitive(rest: &[Step]) -> bool {
    let reducing = |s: &Step| {
        matches!(
            s,
            Step::Count(Scope::Global)
                | Step::Sum(Scope::Global)
                | Step::Min(Scope::Global)
                | Step::Max(Scope::Global)
                | Step::Mean(Scope::Global)
        )
    };

    match rest {
        [t] => reducing(t),
        // `dedup()` keeps FIRST-seen, so which duplicate survives depends on the
        // order — but a count of the survivors does not.
        [Step::Dedupe { labels, bys }, t] if labels.is_empty() && bys.is_empty() => reducing(t),
        _ => false,
    }
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
fn absorb_filters(
    steps: &[Step],
    label: &mut Option<CLabelExpr>,
    props: &mut Vec<CPropConstraint>,
    it: &mut Interner,
    tags: &mut Vec<String>,
) -> usize {
    let mut n = 0;

    for step in steps {
        match step {
            // `as(x)` names the element the prefix is standing on. It filters
            // nothing, so it can sit anywhere among the filters without changing
            // what they mean — and stopping the scan at one is what made
            // `V().as('a').out('R').hasLabel('W')` decline with zero segments.
            Step::As(name) => tags.push(name.clone()),
            Step::HasLabel(names) if label.is_none() => match label_expr(names, it) {
                Some(e) => *label = Some(e),
                None => break,
            },
            Step::Has(key, P::Eq(v)) => match lit_of(v) {
                Some(l) => props.push(CPropConstraint {
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

/// The same, for a node.
fn absorb_node(
    steps: &[Step],
    node: &mut CNode,
    it: &mut Interner,
    tags: &mut Vec<String>,
) -> usize {
    absorb_filters(steps, &mut node.label, &mut node.props, it, tags)
}

/// The direction and type names of an EDGE hop, or `None` if it is not one.
///
/// `outE('R').inV()` is `out('R')` and `MATCH ()-[:R]->(b)` — the same pattern
/// written with the edge named. Spelled apart it can also STOP on the edge, and
/// that is the shape a vertex hop cannot express at all.
fn edge_hop_of(step: &Step) -> Option<(Direction, &[String])> {
    match step {
        Step::OutE(l) => Some((Direction::Out, l)),
        Step::InE(l) => Some((Direction::In, l)),
        Step::BothE(l) => Some((Direction::Both, l)),
        _ => None,
    }
}

/// Whether `step` moves from an edge to the FAR endpoint of a hop in `dir`.
///
/// Only the far end continues the pattern. `outE().outV()` walks back to the
/// vertex it came from, and `bothE().bothV()` emits both ends — neither is
/// another segment, so both decline rather than compile into a pattern that
/// means something else.
///
/// `bothE().otherV()` is here for the same reason `both()` is — see `hop_of`.
fn lands_far(dir: Direction, step: &Step) -> bool {
    matches!(
        (dir, step),
        (Direction::Out, Step::InV) | (Direction::In, Step::OutV) | (Direction::Both, Step::OtherV)
    )
}

/// The direction and type names of a hop step, or `None` if it is not one.
///
/// `both()` is here, and briefly was not. Gremlin traverses a self-loop TWICE —
/// it is an out-edge and an in-edge of the same vertex — where GQL's
/// `MATCH (a)-[:R]-(b)` yields it once: 4 rows against 3 on a two-vertex graph
/// with one loop. That looked like a reason the shared segment could not carry
/// an undirected hop at all.
///
/// It is not. `crate::seek::SelfLoops` had already parameterized exactly this,
/// with the two variants named for the two languages; nothing had plumbed it, so
/// `build_scan` said `Once` unconditionally. The policy now rides on `Ctx::loops`
/// and the planner service asks for `Twice`, which is what a per-language
/// contract should look like — carried, not a reason to decline.
fn hop_of(step: &Step) -> Option<(Direction, &[String])> {
    match step {
        Step::Out(l) => Some((Direction::Out, l)),
        Step::In(l) => Some((Direction::In, l)),
        Step::Both(l) => Some((Direction::Both, l)),
        _ => None,
    }
}

/// A body that is NOTHING but hops, as `(direction, edge-type expression)` pairs.
///
/// Nothing but hops, because a filter or a projection inside the body changes
/// what each repetition yields — the repetitions are only interchangeable while
/// each one is a bare traversal step.
fn plain_hops(steps: &[Step], it: &mut Interner) -> Option<Vec<(Direction, Option<CLabelExpr>)>> {
    let mut out = Vec::new();
    let mut i = 0;

    while i < steps.len() {
        let (dir, types) = match (hop_of(&steps[i]), edge_hop_of(&steps[i])) {
            (Some((d, t)), _) => {
                i += 1;
                (d, t)
            }
            // Spelled apart, the landing step is part of the hop.
            (_, Some((d, t))) if steps.get(i + 1).is_some_and(|v| lands_far(d, v)) => {
                i += 2;
                (d, t)
            }
            _ => return None,
        };

        out.push((
            dir,
            if types.is_empty() {
                None
            } else {
                Some(label_expr(types, it)?)
            },
        ));
    }

    (!out.is_empty()).then_some(out)
}

/// Compile the longest prefix of `steps` that is a plain pattern.
///
/// Requires a bare `V()` start (an id-seeded one is already a point lookup) and
/// at least one hop — with none the caller's own seeding path is what runs, and
/// this would only add a translation.
#[cfg(feature = "bailprobe")]
macro_rules! decline {
    ($why:expr) => {{
        crate::gql::eval::scan::bailprobe::hit_step(format!("PATTERN {}", $why));
        return None;
    }};
}
#[cfg(not(feature = "bailprobe"))]
macro_rules! decline {
    ($why:expr) => {
        return None
    };
}

/// REJECTED (measured neutral): declining before `absorb_node` when the step
/// list contains no hop at all. Most traversals that reach here have none —
/// `g.V().has('k', 1).values('n')` is a bare-node scan and those are ~700 of the
/// declines — and absorbing first CLONES each property key into a constraint for
/// a pattern about to be thrown away. On `gremlin_index_bench`, whose whole cost
/// is an index seek run thousands of times, it changed nothing: 1.0us against
/// 1.0us, 5.2 against 5.2. The clone is real and is not what those queries spend
/// their time on.
pub(super) fn compile(steps: &[Step]) -> Option<Compiled> {
    compile_inner(steps, false)
}

/// [`compile`] without the orientation decline, for a rewritten `match(…)`.
///
/// That decline reasons about what the ALTERNATIVE costs: when only the start is
/// constrained, a left-to-right walk already seeds the selective end, so a frame
/// buys nothing and costs 1.28x. A `match` has no such alternative — it falls to
/// a backtracking solver with no planner behind it, which measured 3.6ms against
/// 0.000ms for the same question in GQL. Anything the planner does is better than
/// that, including nothing.
pub(super) fn compile_chain(steps: &[Step]) -> Option<Compiled> {
    compile_inner(steps, true)
}

fn compile_inner(steps: &[Step], forced: bool) -> Option<Compiled> {
    // `g.E()` IS `g.V().outE()`: every edge appears exactly once as an out-edge
    // of its source, self-loops included, so the two enumerate the same multiset
    // with the traverser on the edge either way. Only the ORDER differs — edge id
    // against source-vertex adjacency — and an unordered result does not promise
    // one (the same rule that lets the V() prefix reorder today).
    //
    // Desugaring rather than teaching the loop a second start shape is what makes
    // every capability the prefix already has — edge filters, `as()` tags,
    // landing steps, `repeat()` — apply to an edge source for free. It is also
    // why this is a rewrite and not a copy: there is one prefix compiler.
    if let [Step::E(ids), rest @ ..] = steps {
        if !ids.is_empty() {
            decline!("E(ids) seeded start");
        }

        let mut desugared = Vec::with_capacity(rest.len() + 2);

        desugared.push(Step::V(Vec::new()));
        desugared.push(Step::OutE(Vec::new()));
        desugared.extend_from_slice(rest);

        let mut compiled = compile_inner(&desugared, forced)?;

        // Only worth it if the traversal LEAVES the edge. A prefix that stops on
        // the edge is already answered from the edge property column directly
        // (`column_terminal`), and the pattern branch runs BEFORE that — so
        // compiling this shape would take a fast path away rather than add one.
        // Measured: `g.E().has('w', 1).count()` 0.18ms from the column, 1.30ms
        // planned as `()-[r {w: 1}]->()`.
        if compiled.end_is_edge {
            decline!("E() start that stays on the edge");
        }

        // Two synthetic steps stood in for one real one, and `consumed` indexes
        // the CALLER's list.
        compiled.consumed -= 1;
        compiled.reorders = true;

        return Some(compiled);
    }

    let [Step::V(ids), rest @ ..] = steps else {
        decline!("start is not V()");
    };

    if !ids.is_empty() {
        decline!("V(ids) seeded start");
    }

    let mut it = Interner::default();
    let mut consumed = 1;
    let mut start = CNode {
        // UNBOUND, deliberately. The start is filtered — its label and inline
        // constraints seed the scan — but nothing downstream reads it: the
        // traverser's value after the prefix is the LAST element, and the planner
        // service asks for that slot alone. Binding it anyway made `build_scan`
        // carry a second id column through every hop and replicate it on every
        // fan-out, for a column discarded on return.
        var_slot: None,
        label: None,
        props: Vec::new(),
        where_: None,
    };
    // Slots are handed out only to elements something READS: a tagged one, or
    // the last, which is the traverser's value. An intermediate node the pattern
    // merely walks through costs a column carried and replicated on every
    // fan-out, for nothing.
    let mut next_slot = 0usize;
    let mut tags: Vec<(String, usize, bool)> = Vec::new();
    let mut start_tags: Vec<String> = Vec::new();
    let taken = absorb_node(rest, &mut start, &mut it, &mut start_tags);

    consumed += taken;

    if !start_tags.is_empty() {
        let s = next_slot;

        next_slot += 1;
        start.var_slot = Some(s);
        tags.extend(start_tags.into_iter().map(|n| (n, s, false)));
    }

    let mut segments: Vec<CSegment> = Vec::new();
    // Set when the traversal STOPS on an edge (`outE('R').has('w', 1)`), which is
    // the one shape a vertex hop cannot spell. The loop then has to end: there is
    // no next segment to start from an edge.
    let mut ends_on_edge = false;

    while let Some(step) = steps.get(consumed) {
        // `repeat(<hops>).times(n)` IS those hops n times over, so it is n
        // segments. Exact for Gremlin because a repeat is a WALK — an edge may be
        // traversed twice — which is what n plain segments mean. (GQL's `{n,n}`
        // is a TRAIL and therefore is NOT this; see `try_count_varlen_upto_2`.)
        //
        // `until`/`emit` decide PER TRAVERSER whether to stop or to yield, which
        // is a predicate over the stream rather than a fixed shape, and an
        // unbounded repeat has no length to unroll. `lower_hops` unrolls under
        // exactly these conditions; this is the same rule, one layer up.
        if let Step::Repeat {
            body,
            times: Some(n),
            until: None,
            emit: None,
            ..
        } = step
        {
            // Not `?`: the decline is TALLIED, which is the whole point of
            // knowing which shapes the compiler turns away.
            let hops = match plain_hops(&body.steps, &mut it) {
                Some(h) => h,
                None => decline!("repeat body is not plain hops"),
            };

            if *n == 0 || hops.is_empty() {
                decline!("repeat of nothing");
            }

            consumed += 1;

            for _ in 0..*n {
                for (dir, label) in &hops {
                    segments.push(CSegment {
                        rel: CRel {
                            var_slot: None,
                            label: label.clone(),
                            direction: *dir,
                            props: Vec::new(),
                            where_: None,
                            quantifier: None,
                        },
                        node: CNode {
                            var_slot: None,
                            label: None,
                            props: Vec::new(),
                            where_: None,
                        },
                        unit: None,
                    });
                }
            }

            // Filters after the repeat constrain where it LANDED, exactly as they
            // would after the equivalent written-out hop.
            let mut node_tags: Vec<String> = Vec::new();
            let last = segments.last_mut()?;

            consumed += absorb_node(&steps[consumed..], &mut last.node, &mut it, &mut node_tags);

            if !node_tags.is_empty() {
                let slot = next_slot;

                next_slot += 1;
                last.node.var_slot = Some(slot);
                tags.extend(node_tags.into_iter().map(|n| (n, slot, false)));
            }

            continue;
        }

        // A vertex hop is an edge hop with its landing already applied.
        let (dir, types, spelled_apart) = match (hop_of(step), edge_hop_of(step)) {
            (Some((d, t)), _) => (d, t, false),
            (_, Some((d, t))) => (d, t, true),
            _ => break,
        };

        consumed += 1;

        let mut rel_tags: Vec<String> = Vec::new();
        let mut node_tags: Vec<String> = Vec::new();
        let mut rel = CRel {
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
        };
        let mut node = CNode {
            var_slot: None,
            label: None,
            props: Vec::new(),
            where_: None,
        };

        if spelled_apart {
            // Filters here constrain the EDGE — `outE('R').has('w', 1)`.
            consumed += absorb_filters(
                &steps[consumed..],
                &mut rel.label,
                &mut rel.props,
                &mut it,
                &mut rel_tags,
            );

            match steps.get(consumed) {
                Some(v) if lands_far(dir, v) => consumed += 1,
                // Stopping on the edge, or on a landing that is not the far end.
                // The latter declines: `outE().outV()` returns to where it came
                // from and `bothE().bothV()` emits both ends, so neither is the
                // segment this would compile it into.
                Some(Step::InV | Step::OutV | Step::OtherV | Step::BothV) => {
                    decline!("landing step is not the far end")
                }
                _ => ends_on_edge = true,
            }
        }

        if ends_on_edge {
            // An UNDIRECTED hop that stops on the edge declines. Everywhere else
            // the self-loop difference is a property of the EXPANSION, and
            // `Ctx::loops` carries it; here the planner does not expand at all —
            // it enumerates the edges a seed selects, and an enumeration visits
            // each edge once whatever its direction. `bothE('R').has('w', 1)`
            // yields a loop twice in the stream and once from the plan, and there
            // is no policy to set: "how many times does an enumeration list one
            // row" is not a question with two answers.
            if dir == Direction::Both {
                decline!("undirected hop ending on the edge");
            }
        } else {
            consumed += absorb_node(&steps[consumed..], &mut node, &mut it, &mut node_tags);
        }

        // The edge binds when it is tagged or when the traversal stops on it; the
        // node binds when it is tagged, and unconditionally at the end (fixed up
        // after the loop, once "the end" is known).
        if ends_on_edge || !rel_tags.is_empty() {
            let s = next_slot;

            next_slot += 1;
            rel.var_slot = Some(s);
            tags.extend(rel_tags.into_iter().map(|n| (n, s, true)));
        }

        if !ends_on_edge && !node_tags.is_empty() {
            let s = next_slot;

            next_slot += 1;
            node.var_slot = Some(s);
            tags.extend(node_tags.into_iter().map(|n| (n, s, false)));
        }

        segments.push(CSegment {
            rel,
            node,
            unit: None,
        });

        if ends_on_edge {
            break;
        }
    }

    // REJECTED too, and for a different reason than the one below: letting a
    // HOPLESS prefix compile costs almost nothing, and buys nothing either.
    //
    // Perf is roughly neutral — `plan_pattern_ids` on a segment-less pattern is
    // 5us, against the 144us `V.hasLabel(V).values(n)` spends reading the
    // properties. Measured over two alternating rounds, most shapes were neutral
    // or slightly faster (`dedup().count()` 0.519ms -> 0.492ms, `id()` 0.415ms ->
    // 0.402ms) and two were not: an indexed `has('n', 7).count()` went 0.032ms ->
    // 0.040ms and `.values('n')` 0.034ms -> 0.042ms, the fixed `compile` + `Ctx`
    // setup showing up against a query too small to hide it.
    //
    // The reason not to is that it removes no code. `try_values` would stop
    // seeing hopless shapes, but it still needs `lower_prefix` to seed the ones
    // WITH hops and `lower_hops` to walk them, so nothing can be deleted and the
    // setup is paid for nothing.
    if segments.is_empty() {
        decline!(format!(
            "no hop; stopped at {}",
            steps
                .get(consumed)
                .map_or_else(|| "end".to_string(), step_kind)
        ));
    }

    // REJECTED, measured twice, so the next attempt starts past it: routing the
    // UNCONSTRAINED shapes through here as well, to make `try_count`/`try_values`
    // deletable. They answer 436 traversals across the suite against this path's
    // 156, so deleting them is the only large consolidation left — and it costs
    // more than it saves.
    //
    // Directly, on 50k vertices with one out-edge each, asking `plan_pattern_ids`
    // for the far end of `()-[:R]->(b)`: 0.578ms through the frame, against
    // 0.188ms for the old path's whole `V.out(R).count()`. Adding a walk route to
    // `plan_pattern_ids` — `seek::walk_ids`, the walk both engines already share,
    // skipping the frame entirely — brought that to 0.276ms, still short.
    //
    // End to end with this decline relaxed and that walk route in place:
    //
    //   V.out(R)              0.304ms -> 0.360ms
    //   V.out(R).count()      0.188ms -> 0.284ms
    //   V.out(R).values(n)    0.353ms -> 0.415ms
    //
    // What is left is not the frame: it is `compile` plus a `Ctx` plus a
    // materialized id column, where the old path walks and counts in place
    // (`seek::walk_count`). A pattern that HAS something to plan earns that
    // setup back; one that does not cannot.
    //
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
    // An edge PROPERTY counts too: the planner can seed from an edge property
    // index, which a left-to-right walk has no way to reach. An edge TYPE does
    // not — both paths already push that into the adjacency.
    let can_orient = segments
        .iter()
        .any(|s| s.node.label.is_some() || !s.node.props.is_empty() || !s.rel.props.is_empty());

    if !can_orient && !forced {
        decline!("nothing past the start is constrained");
    }

    // The traverser's value after the prefix is the last element. If the walk
    // ended on a node it may not have been bound yet — nothing tagged it — so
    // bind it now.
    // `ends_on_edge` is the loop's own record of where the walk stopped. It
    // cannot be inferred from which slots are bound: `outE('R').as('e').inV()`
    // binds the EDGE (it is tagged) and still lands on the node.
    let last = segments.last_mut()?;
    let end_is_edge = ends_on_edge;

    if !end_is_edge && last.node.var_slot.is_none() {
        last.node.var_slot = Some(next_slot);
        next_slot += 1;
    }

    let end_slot = if end_is_edge {
        last.rel.var_slot?
    } else {
        last.node.var_slot?
    };

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
        end_slot,
        end_is_edge,
        tags,
        scope_len: next_slot.max(1),
        consumed,
        reorders: false,
    })
}

/// The variant name of a step, for the decline tally.
#[cfg(feature = "bailprobe")]
fn step_kind(step: &Step) -> String {
    format!("{step:?}")
        .split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or("?")
        .to_string()
}
