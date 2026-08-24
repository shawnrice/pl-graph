//! The from-scratch query engine.
//!
//! Built to the design in `docs/design/engine-from-scratch.md`: two languages
//! compile into one neutral algebra ([`ir`]); one columnar batch model
//! ([`batch`]) executes it ([`exec`]) over a typed columnar store ([`store`]);
//! and one value contract ([`value`]) owns representation and semantics for every
//! layer. Independent of `lenke-core` — this is the new engine, grown against the
//! existing one and its conformance suite as an oracle, not built on it.
//!
//! Status: the design's build order is complete for its subset.
//! - [`value`] — the value contract: representation + semantics (order, equality,
//!   coercion, null/NaN, grouping) in one place.
//! - [`store`] — typed columnar store (unboxed property columns, label buckets,
//!   adjacency with edge ids).
//! - [`batch`] — the one batch type (slot columns) + an optional lineage sidecar.
//! - [`ir`] — the neutral, language-agnostic algebra.
//! - [`exec`] — Scan, Filter, Project, Expand, Aggregate/group-by, OrderPage,
//!   Distinct, Join, VarLength (trail vs walk), ShortestPath, and the
//!   lineage-preserving strategy (path carried only when the plan reads it).
//! - [`gql`] — GQL front-end: MATCH (single/multi-hop, directed/undirected/
//!   var-length, comma-join), WHERE, RETURN with aggregation, DISTINCT,
//!   ORDER/SKIP/LIMIT.
//! - [`gremlin`] — Gremlin front-end over the SAME IR; the payoff test proves a
//!   GQL query and its Gremlin equivalent produce identical rows.
//! - [`opt`] — rewrite-rule optimizer (predicate pushdown, filter-merge) that
//!   fires on plans from either language.
//!
//! Deferred within the subset (documented at their sites): tags/sack lineage
//! (only path is carried); lineage through Join/VarLength/ShortestPath; a
//! cost-based optimizer; right-side join pushdown.

pub mod algo;
pub mod arrow;
pub mod binary;
pub mod bind;
// The textual codecs (pg-json/pg-text/graphson/csv) over the shared crate — the
// `codecs` feature; a minimal build drops them (NDJSON/binary stay, native).
#[cfg(feature = "codecs")]
pub mod codec;
pub mod cost;
pub mod exec;
// The C ABI (`lnk_*` exports). Gated behind `capi` so the engine's `#[no_mangle]`
// symbols never collide with core's when core links this crate for engine-compare.
#[cfg(feature = "capi")]
pub mod ffi;
#[cfg(feature = "capi")]
pub mod ffi_error;
pub mod gql;
pub mod gremlin;
pub mod ir;
pub mod json;
pub mod ndjson;
pub mod opt;
pub mod prepared;
pub mod schema_op;
pub mod store;

// Foundational types live in the `lenke-engine-core` leaf crate so they compile and
// cache independently of the query layers. Re-exported here so `crate::{value, batch,
// temporal, error_codes}` and `lenke_engine::{…}` paths are unchanged. `error_codes`
// keeps its @generated sync-with-@lenke/errors contract (drift test targets core now).
pub use lenke_engine_core::{batch, error_codes, temporal, value};
