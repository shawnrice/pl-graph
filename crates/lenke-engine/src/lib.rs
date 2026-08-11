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
pub mod batch;
pub mod cost;
pub mod exec;
pub mod gql;
pub mod gremlin;
pub mod ir;
pub mod ndjson;
pub mod opt;
pub mod store;
pub mod temporal;
pub mod value;
