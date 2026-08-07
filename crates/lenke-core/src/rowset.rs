//! [`RowSet`] — the materialized result both query entry points return.
//!
//! Its own module since 2026-08-07. It used to live in `query.rs` beside a
//! second, hand-rolled GQL subset that existed to produce a `(count, sum,
//! checksum)` fingerprint for benchmark comparison. That engine is gone — see
//! the commit that removed it — but `RowSet` is the real result type, reached by
//! `gql::eval`, `algo`, `arrow` and the FFI, so it outlives its old neighbour.

use crate::graph::Value;

/// A materialized result: column names plus a **columnar** cell buffer — a
/// single flat row-major `Vec<Value>` (cell `(i, j)` at `i*ncols + j`) instead
/// of a `Vec` per row, so building an N-row result is one amortized allocation,
/// not N small ones. Cells are the core graph `Value` model so the rowset
/// round-trips losslessly to JSON for the FFI / wasm boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct RowSet {
    pub cols: Vec<String>,
    /// Flat row-major cells; `nrows * cols.len()` long.
    pub data: Vec<Value>,
    pub nrows: usize,
}

impl RowSet {
    pub fn new(cols: Vec<String>) -> Self {
        Self {
            cols,
            data: Vec::new(),
            nrows: 0,
        }
    }
    /// Same, with room for `nrows` rows already reserved.
    ///
    /// The flat buffer is the point of this type — one amortized allocation
    /// rather than one per row — but "amortized" still means a doubling growth
    /// that copies the whole buffer log(n) times, and a `Value` is 40 bytes. A
    /// 20k-row projection was moving 800KB through several of those reallocs on
    /// the way to a size it knew before it started.
    pub fn with_capacity(cols: Vec<String>, nrows: usize) -> Self {
        let n = cols.len();

        Self {
            cols,
            data: Vec::with_capacity(nrows.saturating_mul(n)),
            nrows: 0,
        }
    }
    pub fn ncols(&self) -> usize {
        self.cols.len()
    }
    /// Row `i` as a slice of its cells.
    pub fn row(&self, i: usize) -> &[Value] {
        let c = self.cols.len();
        &self.data[i * c..i * c + c]
    }
    /// Iterate rows as cell slices.
    pub fn rows(&self) -> impl Iterator<Item = &[Value]> {
        let c = self.cols.len().max(1); // chunks(0) panics; empty-col → no rows
        self.data.chunks(c).take(self.nrows)
    }
    /// Append a row (its cells; must be exactly `ncols`).
    pub fn push_row(&mut self, cells: impl IntoIterator<Item = Value>) {
        self.data.extend(cells);
        self.nrows += 1;
        debug_assert_eq!(self.data.len(), self.nrows * self.cols.len());
    }
    /// Drop the most recently pushed row (used to undo a DISTINCT duplicate).
    pub fn pop_row(&mut self) {
        self.data.truncate(self.data.len() - self.cols.len());
        self.nrows -= 1;
    }
    /// Apply SKIP/LIMIT in place over the flat buffer.
    pub fn apply_skip_limit(&mut self, skip: usize, limit: Option<usize>) {
        let c = self.cols.len();
        let skip = skip.min(self.nrows);
        if skip > 0 {
            self.data.drain(0..skip * c);
            self.nrows -= skip;
        }
        if let Some(n) = limit {
            if self.nrows > n {
                self.data.truncate(n * c);
                self.nrows = n;
            }
        }
    }

    /// Serialize to a compact `{"columns":[...],"rows":[[...]]}` JSON document —
    /// the carrier for both bun:ffi and the wasm binding, where a single buffer
    /// crossing beats marshalling cell-by-cell. Hand-rolled (no `serde_json`) so
    /// the core query path carries no JSON dependency — that's what lets a
    /// minimal frontend wasm build (GQL only) drop `serde_json` entirely.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(self.cols.len() * 16 + self.nrows * 32);
        out.push_str("{\"columns\":[");
        for (i, c) in self.cols.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            crate::jsonfmt::push_json_str(&mut out, c);
        }
        out.push_str("],\"rows\":[");
        for (ri, r) in self.rows().enumerate() {
            if ri > 0 {
                out.push(',');
            }
            out.push('[');
            for (ci, cell) in r.iter().enumerate() {
                if ci > 0 {
                    out.push(',');
                }
                crate::jsonfmt::push_value(&mut out, cell);
            }
            out.push(']');
        }
        out.push_str("]}");
        out
    }
}
