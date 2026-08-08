//! The from-scratch query engine.
//!
//! Built to the design in `docs/design/engine-from-scratch.md`: two languages
//! compile into one neutral algebra ([`ir`]); one columnar batch model
//! ([`batch`]) executes it ([`exec`]) over a typed columnar store ([`store`]);
//! and one value contract ([`value`]) owns representation and semantics for every
//! layer. Independent of `lenke-core` — this is the new engine, grown against the
//! existing one and its conformance suite as an oracle, not built on it.
//!
//! Status: first vertical slice — Scan → Filter → Project executing end to end.
//! Graph operators (Expand, VarLength, ShortestPath), aggregation, ordering,
//! effects, the optimizer rules, the lineage-preserving operator strategy, and
//! the GQL/Gremlin front-ends land in subsequent slices, in the build order the
//! design lays out.

pub mod batch;
pub mod exec;
pub mod gql;
pub mod gremlin;
pub mod ir;
pub mod opt;
pub mod store;
pub mod value;
