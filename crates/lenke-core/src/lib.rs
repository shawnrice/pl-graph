//! Columnar labeled-property-graph (LPG) core: a mutable in-memory graph with
//! dense `u32` vertex ids and typed, contiguous property columns, plus the query
//! and serialization surfaces built on it.
//!
//! Versus the TS core (edges as objects indexed in nested hash maps), adjacency
//! and properties are stored columnar — cache-friendly, and de-boxed on the hot
//! paths. On top sit two query engines (ISO-GQL and Gremlin), the (de)serialization
//! codecs, and an Apache Arrow result surface. Everything above the graph is
//! feature-gated (see the `[features]` table in Cargo.toml) so a minimal build —
//! e.g. a frontend wasm bundle — ships only what it uses.
//!
//! Binding-agnostic: `ffi` exposes a C ABI for bun:ffi (and later wasm-bindgen)
//! over a stateful graph handle.

// The engine is safe Rust. `unsafe` is denied crate-wide and re-permitted only in the
// C-ABI boundary modules (`ffi`, `ffi_error`), which additionally deny
// `unsafe_op_in_unsafe_fn` so every raw-pointer op there is an explicit, minimal block.
#![deny(unsafe_code)]

// Core (always compiled): the columnar graph, the fingerprint query subset, and
// the C-ABI surface.
pub mod error;
pub mod error_codes;
pub mod ffi;
pub mod ffi_error;
#[cfg(test)]
mod fixtures;
mod interval_index;
// In-engine graph algorithms. Config is a JSON object, so this rides on the
// shared JSON parser (the `ndjson` feature) — present in every build that can
// load a graph; only the `gql`-only minimal wasm bundle omits it.
#[cfg(feature = "ndjson")]
pub mod algo;
/// One fast hash for both engines — see the module docs.
pub(crate) mod fxhash;
pub mod graph;
/// Is there a point where every row must exist at once? — see the module docs.
pub mod pipeline;
pub mod query;
/// The shared index access path both query engines lower into.
pub mod seek;
pub mod temporal;
/// The runtime value both query engines carry.
pub mod value;

// Composable capabilities — gated so a minimal (e.g. frontend wasm) build ships
// only what it uses. See the `[features]` table in Cargo.toml.
#[cfg(feature = "arrow")]
pub mod arrow;
#[cfg(feature = "codecs")]
pub mod codec;
#[cfg(feature = "gql")]
pub mod gql;
#[cfg(feature = "gremlin")]
pub mod gremlin;
// Shared JSON writer primitives (js_number + string escaper), used by every
// serde-free JSON surface. gql hand-rolls its own tabular output and omits it.
#[cfg(any(feature = "gremlin", feature = "ndjson"))]
mod jsonfmt;
// The shared hand-rolled JSON *parser* — the read side of ndjson + the codecs
// (which imply ndjson). gql/gremlin never parse JSON, so they omit it.
#[cfg(feature = "ndjson")]
mod json;
#[cfg(feature = "ndjson")]
pub mod ndjson;

#[cfg(test)]
mod fuzz_tests;

/// The decline tallies, dumped after everything else has run.
///
/// At the CRATE ROOT and named `zzz_` because the tallies are process-wide and
/// libtest runs in NAME order under `--test-threads=1`. Inside `gql::tests` this
/// sorted before every `gremlin::` test and reported the Gremlin map as empty.
#[cfg(all(test, feature = "bailprobe"))]
mod zzz_probe {
    #[test]
    #[ignore]
    fn dump() {
        println!(
            "\n=== GQL: shapes that declined the columnar frame ===\n{}\n\n\
             === GREMLIN: steps that ran through the stream ===\n{}\n",
            crate::gql::eval::scan::bailprobe::dump(),
            crate::gql::eval::scan::bailprobe::dump_steps()
        );
    }
}
