//! Foundational types shared across the lenke engine, split into their own crate so
//! they compile and cache independently of the query layers (ir / store / exec /
//! gql / gremlin). Nothing here depends on those layers.
//!
//! - [`value`] — the `Value` model (one numeric type f64, first-class null, lists,
//!   maps, temporals) and its ordering/equality/rendering contracts.
//! - [`batch`] — the columnar execution unit: `Batch` of typed `Col` slots plus the
//!   `Lineage` sidecar (node/edge paths + Gremlin step-history).
//! - [`temporal`] — Date/Time/DateTime/Duration value types and their arithmetic.
//! - [`error_codes`] — the stable `E_*` error-code strings (mirrored to TS).
//!
//! Re-exported from `lenke-engine` as `lenke_engine::{value, batch, temporal,
//! error_codes}`, so `crate::value::…` paths inside the engine are unchanged.

pub mod batch;
pub mod error_codes;
pub mod gstr;
pub mod json_fmt;
pub mod temporal;
pub mod value;
