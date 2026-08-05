//! What SHAPE is this plan, and therefore how should it run?
//!
//! Both engines had this decision hard-coded per query shape, in their own way,
//! and they disagreed. It is one question with one answer: **is there a point in
//! this plan where every row has to exist at once?**
//!
//! Everything else follows from that. A `distinct` or an `order` cannot emit its
//! first row until it has seen its last, so the rows are getting materialized no
//! matter which engine runs them — and once they are materialized, the operation
//! is the same operation over the same buffer, in either language. Whereas a
//! plan with no such point can stream, and if it also has a `LIMIT` it *should*,
//! because streaming stops early and materializing cannot.
//!
//! So the classification is about MEMORY, not about which language wrote it:
//!
//! - [`OpClass::Streaming`] — a row in, zero or more rows out, no buffer. Filters
//!   and expansions and projections.
//! - [`OpClass::Reducing`] — consumes every row, emits one, and needs no buffer
//!   to do it: `count`, `sum`, `min`. Blocking, but free. A streaming fold beats
//!   materializing here, which is what GQL's `agg-no-where` routing already knew
//!   and Gremlin had no way to say.
//! - [`OpClass::Buffering`] — cannot emit until every row exists: `order`,
//!   `dedup`, `group`, `fold`. THE boundary. The rows are being materialized
//!   anyway, so the columnar form is free at this point and the operation should
//!   run over a column rather than over a stream of boxed rows.
//! - [`OpClass::Opaque`] — carries per-row state the shared layer cannot model
//!   (a path, a sack, a branch). Not a boundary; a decline.
//!
//! Each engine classifies its OWN operations — that part is irreducibly
//! per-language — and the routing rule below is shared.
//!
//! # What actually uses this, as of 2026-08-05
//!
//! Only Gremlin classifies, in `gremlin::analysis::facts`, and only the probe
//! and that module's tests read the `Route` it produces. The executor does not
//! branch on it. That is worth stating plainly rather than leaving the module to
//! imply otherwise:
//!
//! - Offering the planned ids to Gremlin's COLUMN TERMINALS answers the boundary
//!   question without asking it — a terminal that wants a column IS a boundary —
//!   so "try the column path, else stream" subsumed the routing at the one site
//!   that would have branched on it.
//! - GQL has no classifier. Routing it by boundary alone was written and
//!   measured: `MATCH (a:V)-[:R]->(b) WHERE a.n > 900 RETURN b.n` went from
//!   0.317ms to 8.176ms, because declining the frame does NOT reach a streaming
//!   executor — GQL's fallback is a per-row binding-table interpreter. The
//!   distinction that mattered turned out to be how many COLUMNS the join has to
//!   carry, not stream-versus-columnar; see `scan::streamed_frame`.
//!
//! What IS shared is the Reducing route itself: [`crate::seek::walk_count`] walks
//! every hop but the last and counts the last in place, and both engines call it.
//! That is the piece this classification was pointing at, arrived at directly.

/// How an operation treats the rows flowing through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// A row in, zero or more out, no buffer.
    Streaming,
    /// Consumes every row and emits one, without holding them.
    Reducing,
    /// Cannot emit until every row exists. The boundary.
    Buffering,
    /// Per-row state the shared layer cannot model.
    Opaque,
}

/// Which execution model a plan should take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// Materialize into columns: something downstream buffers anyway, so pay for
    /// the frame once and let the boundary operate on a column.
    Columnar,
    /// Stream row-at-a-time: nothing buffers, or a `LIMIT` can stop early, and
    /// materializing would do strictly more work.
    Stream,
    /// The shared layer cannot model this plan; the engine runs it its own way.
    Decline,
}

/// The shape of a plan, as a routing decision.
///
/// Deliberately three lines. A `LIMIT` is NOT part of it: a limit ahead of a
/// boundary shrinks the buffer but does not remove it, and a limit with no
/// boundary already streams. Where a limit genuinely changes the answer is one
/// layer down, in whether the cap is pushed into the WALK — GQL learned that the
/// expensive way, capping by truncating a materialized result made
/// `RETURN … LIMIT 100` 92x slower than stopping the scan early. That belongs to
/// `scan_capped`, not here.
#[must_use]
pub fn route(classes: &[OpClass]) -> Route {
    if classes.contains(&OpClass::Opaque) {
        return Route::Decline;
    }

    if classes.contains(&OpClass::Buffering) {
        return Route::Columnar;
    }

    // Nothing has to hold the rows: a filter/expand chain, possibly ending in a
    // fold that keeps one accumulator. Materializing would be strictly more work.
    Route::Stream
}

/// Index of the first operation that has to see every row at once, if any.
///
/// Everything before it is a candidate for one materialized frame; the boundary
/// itself then runs over columns rather than over boxed rows.
#[must_use]
pub fn first_boundary(classes: &[OpClass]) -> Option<usize> {
    classes.iter().position(|c| *c == OpClass::Buffering)
}

#[cfg(test)]
mod tests {
    use super::{first_boundary, route, OpClass::*, Route};

    #[test]
    fn a_buffering_step_routes_columnar_and_is_found() {
        // filter, expand, dedup, count
        let plan = [Streaming, Streaming, Buffering, Reducing];

        assert_eq!(route(&plan), Route::Columnar);
        assert_eq!(first_boundary(&plan), Some(2));
    }

    #[test]
    fn a_pure_fold_streams() {
        // A reducing terminal holds one accumulator; a frame would be pure cost.
        let plan = [Streaming, Streaming, Reducing];

        assert_eq!(route(&plan), Route::Stream);
        assert_eq!(first_boundary(&plan), None);
    }

    #[test]
    fn a_plain_filter_chain_streams() {
        assert_eq!(route(&[Streaming, Streaming]), Route::Stream);
    }

    #[test]
    fn per_row_state_declines_whatever_else_is_there() {
        // An opaque op outranks a boundary: the shared layer cannot model it at
        // all, so there is nothing to decide.
        assert_eq!(route(&[Streaming, Buffering, Opaque]), Route::Decline);
    }
}
