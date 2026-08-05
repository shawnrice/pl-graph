//! What a traversal needs, decided once by reading it.
//!
//! A step list is a compilable structure, and three separate questions used to
//! be asked of it by three separate hand-maintained lists:
//!
//! - does anything read `Trav::path`, so the accumulation must be kept
//! - does anything read the `as(label)` tag map, so a lowered prefix must carry
//!   its bindings back out
//! - where the pipeline BOUNDARIES are, for [`crate::pipeline`]
//!
//! Three lists meant three chances to answer a new step in one and forget it in
//! the others, and that is exactly what happened: `where(eq('a'))` and
//! `math('a + b')` both read tags, neither was listed, and a lowered prefix
//! dropped the bindings out from under them — `as('a').out('R').where(eq('a'))`
//! returned nothing where the answer was one vertex. The path list had the same
//! step and got it right, which is what made it invisible.
//!
//! So each step declares its facts ONCE, in [`facts`], and the unlisted case is
//! maximally conservative: reads the path, reads the tags, opaque to the
//! pipeline. A step added later costs a missed optimization until someone lists
//! it, never a wrong answer.
//!
//! # Which sub-traversals count
//!
//! A step's bodies fall in two kinds and only one of them can see the traverser:
//!
//! - a FILTER or BRANCH body — `where(…)`, `not(…)`, `repeat(…)`, `choose(…)` —
//!   is seeded with the traverser itself (`sub_nonempty(graph, ctx, sub, t)`), so
//!   it sees the outer path and the outer tags. [`carried`] returns these, and
//!   the analysis recurses into them.
//! - a `by()` MODULATOR runs on `Trav::root(value)` in `eval_by` — a fresh
//!   traverser with an empty path and an empty tag map. It can see neither, so it
//!   is not recursed into.
//!
//! That distinction was previously guessed at separately by each list, and both
//! guessed wrong in the safe direction: `dedup().by(k)` was called path-bound
//! because "a modulator may read the path", which cost every such traversal a
//! per-traverser path clone.

use crate::gremlin::{Step, Traversal};
use crate::pipeline::{OpClass, Route};

/// What one step needs from the traverser, and what it does to the stream.
#[derive(Clone, Copy)]
pub(super) struct Facts {
    /// Reads `Trav::path`.
    pub path: bool,
    /// Reads the `as(label)` tag map.
    pub tags: bool,
    /// How it treats the rows flowing through it.
    pub class: OpClass,
}

impl Facts {
    /// Reads nothing; a row in, zero or more out.
    const STREAM: Self = Self {
        path: false,
        tags: false,
        class: OpClass::Streaming,
    };
    /// Reads nothing; consumes every row and emits one, holding no buffer.
    const REDUCE: Self = Self {
        path: false,
        tags: false,
        class: OpClass::Reducing,
    };
    /// Reads nothing; cannot emit until every row exists.
    const BUFFER: Self = Self {
        path: false,
        tags: false,
        class: OpClass::Buffering,
    };
    /// Everything, which is what an unrecognized step gets.
    const OPAQUE: Self = Self {
        path: true,
        tags: true,
        class: OpClass::Opaque,
    };

    const fn reads_tags(self) -> Self {
        Self { tags: true, ..self }
    }
    const fn reads_path(self) -> Self {
        Self { path: true, ..self }
    }
}

/// Everything one step declares about itself.
///
/// The `_` arm is the conservative one on purpose — see the module note. Adding
/// a variant here is the whole cost of teaching every analysis about a new step.
#[allow(clippy::match_same_arms)] // grouped by MEANING, not by answer
pub(super) fn facts(step: &Step) -> Facts {
    match step {
        // Sources, filters, hops and per-row projections: a row in, zero or more
        // out, no history and no buffer.
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
        | Step::BothV
        | Step::Values(..)
        | Step::ValueMap(..)
        | Step::PropertyMap(..)
        | Step::Properties(..)
        | Step::ElementMap(..)
        | Step::Value
        | Step::Id
        | Step::Label
        | Step::Constant(_)
        | Step::Identity
        | Step::None(..)
        | Step::Limit(..)
        | Step::Skip(..)
        | Step::Range(..)
        | Step::Unfold
        | Step::Project(..) => Facts::STREAM,

        // `as(x)` WRITES a tag from the current value and reads no history.
        Step::As(_) => Facts::STREAM,

        // Consumes every row, emits one, holds no buffer to do it.
        Step::Count(_) | Step::Sum(_) | Step::Mean(_) | Step::Min(_) | Step::Max(_) => {
            Facts::REDUCE
        }

        // Cannot emit until every row exists — THE boundary. Their `by()`
        // modulators cannot reach the traverser, so they read nothing.
        Step::Order(..)
        | Step::Group(..)
        | Step::GroupCount(..)
        | Step::Fold
        | Step::Tail(..)
        | Step::Sample(_)
        | Step::Barrier
        | Step::Aggregate(_)
        | Step::Store(_)
        | Step::Cap(_) => Facts::BUFFER,

        // `dedup(labels)` keys on TAGS; `dedup()` and `dedup().by(k)` do not.
        Step::Dedupe { labels, .. } => {
            if labels.is_empty() {
                Facts::BUFFER
            } else {
                Facts::BUFFER.reads_tags()
            }
        }

        // Genuine path readers. `OtherV` asks which end it came from,
        // `simplePath`/`cyclicPath` ask whether it has been here before, and
        // `path`/`tree` emit the history itself.
        Step::OtherV | Step::SimplePath | Step::CyclicPath => Facts::STREAM.reads_path(),
        Step::Path(_) | Step::Tree(_) => Facts::STREAM.reads_path(),

        // Genuine tag readers. Every one of these resolves a LABEL against the
        // tag map — `where(eq('a'))` and `math('a + b')` included, which is the
        // pair a second list missed.
        Step::Select { .. }
        | Step::WhereKey(..)
        | Step::WherePred(_)
        | Step::Math { .. }
        | Step::Match(_) => Facts::STREAM.reads_tags(),
        // …and this one reads both: its endpoints may be tags, and it walks.
        Step::ShortestPath { .. } => Facts::OPAQUE,

        // Carries a sub-traversal seeded with the traverser. The step itself
        // reads nothing; `carried` hands its bodies back and the analysis
        // recurses, so `where(out('R'))` stays cheap and
        // `where(out('R').simplePath())` does not.
        Step::Where(_)
        | Step::Not(_)
        | Step::Optional(_)
        | Step::Local(_)
        | Step::Map(_)
        | Step::FlatMap(_)
        | Step::SideEffect(_)
        | Step::And(_)
        | Step::Or(_)
        | Step::Union(_)
        | Step::Coalesce(_)
        | Step::Choose { .. }
        | Step::Repeat { .. } => Facts {
            path: false,
            tags: false,
            // Opaque to the PIPELINE regardless of the body: the shared layer
            // cannot model a per-row branch.
            class: OpClass::Opaque,
        },

        // Everything else — sacks, mutations, the graph algorithms, anything
        // added since. Conservative by construction.
        _ => Facts::OPAQUE,
    }
}

/// The sub-traversals seeded with the TRAVERSER, which therefore see its path
/// and its tags.
///
/// Deliberately not `by()` modulators: `eval_by` runs each on a fresh
/// `Trav::root(value)`, so a modulator sees neither. Returning them here would
/// re-introduce the over-conservatism this module exists to remove.
pub(super) fn carried(step: &Step) -> Vec<&Traversal> {
    match step {
        Step::Where(t)
        | Step::Not(t)
        | Step::Optional(t)
        | Step::Local(t)
        | Step::Map(t)
        | Step::FlatMap(t)
        | Step::SideEffect(t) => vec![t],
        Step::And(ts) | Step::Or(ts) | Step::Union(ts) | Step::Coalesce(ts) | Step::Match(ts) => {
            ts.iter().collect()
        }
        Step::Choose { test, then_, else_ } => [Some(&**test), Some(&**then_), else_.as_deref()]
            .into_iter()
            .flatten()
            .collect(),
        // `until_before` / `emit_before` are placement flags, not bodies — the
        // bodies they place are `until` and `emit`, which are already here.
        Step::Repeat {
            body, until, emit, ..
        } => [Some(&**body), until.as_deref(), emit.as_deref()]
            .into_iter()
            .flatten()
            .collect(),
        _ => Vec::new(),
    }
}

/// What a whole traversal needs, from one walk over it.
#[derive(Clone, Copy)]
pub(super) struct Shape {
    // `route` and `first_boundary` are read by the probe and by this module's
    // tests, not by the executor — the same standing as `crate::pipeline::route`
    // itself. Offering the planned ids to the column terminals answers the
    // boundary question without asking it, because a terminal that wants a
    // column IS a boundary. They are computed here anyway because the classes
    // are already in hand, and because the general case they were written for is
    // still open: a boundary whose tail `column_paths` declines still streams.
    //
    /// Any step, at any depth, reads `Trav::path`.
    pub needs_path: bool,
    /// Any step, at any depth, reads the tag map.
    pub reads_tags: bool,
    /// Which execution model the plan should take.
    #[cfg_attr(not(feature = "bailprobe"), allow(dead_code))]
    pub route: Route,
    /// Index of the first step that has to see every row at once.
    #[cfg_attr(not(feature = "bailprobe"), allow(dead_code))]
    pub first_boundary: Option<usize>,
}

/// Read a step list once and answer every question about it.
///
/// REJECTED (measured neutral): a second, allocation-free entry point returning
/// just `(needs_path, reads_tags)`, so the executor would not build the
/// `Vec<OpClass>` it never reads. The reasoning was that `run_collect` asks
/// those two questions on every execution and `gremlin_index_bench` runs a point
/// lookup thousands of times. It moved nothing — `within (3 values)` 1.0us
/// against 1.0us, `startsWith` 5.2 against 5.2, `eq point lookup` 0.4 against
/// 0.3 — so the allocation was not the cost, and a second entry point is exactly
/// the shape that let `path_free` and `reads_tags` drift apart to begin with.
pub(super) fn analyze(steps: &[Step]) -> Shape {
    let mut classes = Vec::with_capacity(steps.len());
    let mut needs_path = false;
    let mut reads_tags = false;

    for step in steps {
        let f = facts(step);

        needs_path |= f.path;
        reads_tags |= f.tags;
        classes.push(f.class);

        // A carried body's needs are the outer traverser's needs.
        for sub in carried(step) {
            let inner = analyze(&sub.steps);

            needs_path |= inner.needs_path;
            reads_tags |= inner.reads_tags;
        }
    }

    Shape {
        needs_path,
        reads_tags,
        route: crate::pipeline::route(&classes),
        first_boundary: crate::pipeline::first_boundary(&classes),
    }
}

#[cfg(test)]
mod tests {
    use super::{analyze, carried, facts};
    use crate::gremlin::{Step, Traversal};
    use crate::pipeline::{OpClass, Route};

    fn t(src: &str) -> Traversal {
        crate::gremlin::parse(src).unwrap_or_else(|e| panic!("`{src}` parses: {e}"))
    }

    /// The fail-safe default, asserted directly rather than hoped for.
    ///
    /// This is the property that makes one declaration site safe to add a step
    /// to: forget it entirely and it costs a missed optimization, never a wrong
    /// answer. No query-level test can pin it — a step nothing has written a
    /// query for is exactly the step that has no query.
    #[test]
    fn an_unlisted_step_declares_that_it_reads_everything() {
        for step in [
            Step::Drop,
            Step::Sack {
                op: None,
                bys: Vec::new(),
            },
            Step::Inject(Vec::new()),
            Step::Loops,
            Step::SelectColumn(crate::gremlin::Column::Keys),
        ] {
            let f = facts(&step);

            assert!(f.path, "{step:?} must be assumed to read the path");
            assert!(f.tags, "{step:?} must be assumed to read the tags");
            assert_eq!(
                f.class,
                OpClass::Opaque,
                "{step:?} must be assumed opaque to the pipeline"
            );
        }
    }

    /// A CARRIED body's needs are the outer traverser's needs.
    #[test]
    fn a_carried_body_propagates_what_it_reads() {
        assert!(!analyze(&t("g.V().where(__.out('R'))").steps).needs_path);
        assert!(analyze(&t("g.V().where(__.out('R').simplePath())").steps).needs_path);
        assert!(analyze(&t("g.V().repeat(__.out('R').path()).times(2)").steps).needs_path);
        assert!(!analyze(&t("g.V().repeat(__.out('R')).times(2)").steps).needs_path);
        assert!(analyze(&t("g.V().not(__.where(__.out('R').cyclicPath()))").steps).needs_path);
        // Tags propagate the same way, through the same recursion.
        assert!(!analyze(&t("g.V().where(__.out('R'))").steps).reads_tags);
        assert!(analyze(&t("g.V().where(__.select('a'))").steps).reads_tags);
    }

    /// A `by()` MODULATOR's needs do not — it runs on a fresh root.
    ///
    /// The distinction this module is built on, and the one both old lists got
    /// wrong in the safe direction. If `eval_by` ever stops seeding a fresh
    /// `Trav::root`, this is what says so.
    #[test]
    fn a_by_modulator_reads_neither_the_path_nor_the_tags() {
        for src in [
            "g.V().out('R').dedup().by(__.path())",
            "g.V().out('R').order().by(__.path())",
            "g.V().out('R').group().by(__.path()).by(__.count())",
            "g.V().out('R').project('x').by(__.path())",
        ] {
            let shape = analyze(&t(src).steps);

            assert!(!shape.needs_path, "`{src}` does not read the outer path");
            assert!(!shape.reads_tags, "`{src}` does not read the outer tags");
        }
    }

    /// Every step that genuinely reads one says so.
    #[test]
    fn the_readers_declare_what_they_read() {
        for src in [
            "g.V().out('R').path()",
            "g.V().out('R').simplePath()",
            "g.V().out('R').cyclicPath()",
            "g.V().bothE('R').otherV()",
            "g.V().out('R').tree()",
        ] {
            assert!(analyze(&t(src).steps).needs_path, "`{src}` reads the path");
        }

        for src in [
            "g.V().as('a').out('R').select('a')",
            "g.V().as('a').out('R').where(eq('a'))",
            "g.V().as('a').out('R').as('b').where('a', neq('b'))",
            "g.V().as('a').out('R').dedup('a')",
            "g.V().as('a').out('R').math('_ + 1')",
        ] {
            assert!(analyze(&t(src).steps).reads_tags, "`{src}` reads the tags");
        }
    }

    /// `as` and the plain terminals read neither — the case worth being right
    /// about, since it is what lets a prefix lower at all.
    #[test]
    fn a_plain_prefix_reads_neither() {
        for src in [
            "g.V().as('a').out('R').hasLabel('W').count()",
            "g.V().out('R').values('k')",
            "g.V().out('R').dedup().count()",
            "g.V().out('R').order().by('k').limit(3)",
        ] {
            let shape = analyze(&t(src).steps);

            assert!(!shape.needs_path, "`{src}` reads no path");
            assert!(!shape.reads_tags, "`{src}` reads no tag");
        }
    }

    /// The pipeline half of the same walk.
    #[test]
    fn the_route_follows_the_boundaries() {
        assert_eq!(
            analyze(&t("g.V().out('R').count()").steps).route,
            Route::Stream
        );
        assert_eq!(
            analyze(&t("g.V().out('R').values('k')").steps).route,
            Route::Stream
        );

        let ordered = analyze(&t("g.V().out('R').order().by('k')").steps);

        assert_eq!(ordered.route, Route::Columnar);
        assert_eq!(ordered.first_boundary, Some(2));

        // An opaque step outranks a boundary: there is nothing to decide.
        assert_eq!(
            analyze(&t("g.V().out('R').order().by('k').sack(assign)").steps).route,
            Route::Decline
        );
    }

    /// `carried` returns the bodies seeded with the traverser, and nothing else.
    #[test]
    fn carried_returns_every_body_that_sees_the_traverser() {
        assert_eq!(carried(&t("g.V().where(__.out('R'))").steps[1]).len(), 1);
        assert_eq!(
            carried(&t("g.V().union(__.out('R'), __.out('S'), __.identity())").steps[1]).len(),
            3
        );
        assert_eq!(
            carried(&t("g.V().choose(__.hasLabel('W'), __.out('R'), __.out('S'))").steps[1]).len(),
            3
        );
        // A repeat's body, its `until` and its `emit` — the placement flags are
        // not bodies.
        assert_eq!(
            carried(&t("g.V().repeat(__.out('R')).until(__.hasLabel('W')).emit()").steps[1]).len(),
            3
        );
        // A `by()` is not carried.
        assert!(carried(&t("g.V().order().by(__.path())").steps[1]).is_empty());
    }
}
