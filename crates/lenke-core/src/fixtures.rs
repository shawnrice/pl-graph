//! Shared test graphs, stored as real `.ndjson` documents and decoded through the
//! ordinary NDJSON pipeline.
//!
//! These used to be inline `&str` arrays copied into every test module — fourteen
//! copies of three graphs, which meant a fixture change had to be applied by hand
//! in a dozen places and drift was invisible. Keeping them as documents also means
//! the fixtures are exercised by the decoder the same way real input is, rather
//! than being assembled by a builder that only tests use.
//!
//! `include_str!` embeds them at compile time, so there is no runtime file IO and
//! nothing to resolve relative to the working directory (the tests also run under
//! wasm, where there is no filesystem).
//!
//! All three are the TinkerPop "Modern" graph — same six vertices, same six edges,
//! same weights and ids as the canonical TinkerPop definition (`v[1]`..`v[6]`,
//! `e[7][1-knows->2]`..`e[12]`). The one deliberate departure is LABEL CASING,
//! which follows each language's conventions rather than TinkerPop's lowercase,
//! matching the TS fixtures these mirror.
//!
//! Three files, because the remaining differences are load-bearing — do not merge
//! them:
//!   - [`modern_gql`] labels nodes `Person` / `Software` and edges `KNOWS` /
//!     `CREATED` (the GQL convention: PascalCase labels, SCREAMING_SNAKE types).
//!     Mirrors `packages/gql/src/fixtures/createTestSocialGraph.ts` exactly,
//!     element ids included, so a ported test asserts the same ids as its TS twin.
//!   - [`modern_gremlin`] labels everything SCREAMING_SNAKE (`PERSON`, `KNOWS`),
//!     mirroring `packages/core/src/fixtures/createTestTinkerGraph.ts`, and leaves
//!     edges id-less so they take the canonical `e{index}` form.
//!   - [`modern_gremlin_edge_ids`] is that graph with explicit edge ids `7..=12`,
//!     which the step tests that assert TinkerGraph edge ids need.

use crate::graph::Graph;
use crate::ndjson;

/// The TinkerPop "Modern" graph as the GQL suites use it: canonical ids (`1`..`6`
/// for nodes, `7`..`12` for edges), labelled `Person` / `Software` + `KNOWS` /
/// `CREATED`. A byte-for-byte mirror of the TS `createTestSocialGraph`.
pub(crate) fn modern_gql() -> Graph {
    decode(include_str!("fixtures/modern_gql.ndjson"))
}

/// The TinkerPop "Modern" graph as the Gremlin suites use it: nodes keyed by
/// number (`1`..`6`), labelled `PERSON` / `SOFTWARE`, edges id-less (so they take
/// the canonical `e{index}` id).
pub(crate) fn modern_gremlin() -> Graph {
    decode(include_str!("fixtures/modern_gremlin.ndjson"))
}

/// [`modern_gremlin`] with explicit edge ids `7..=12` — for the step tests that
/// assert the TinkerGraph edge ids rather than the canonical ones.
pub(crate) fn modern_gremlin_edge_ids() -> Graph {
    decode(include_str!("fixtures/modern_gremlin_edge_ids.ndjson"))
}

/// A fixture that fails to decode is a broken fixture, not a failing test — so
/// this panics with the decoder's message rather than surfacing as an assertion
/// failure somewhere far away.
fn decode(doc: &str) -> Graph {
    ndjson::decode(doc).unwrap_or_else(|e| panic!("fixture failed to decode: {e}"))
}
