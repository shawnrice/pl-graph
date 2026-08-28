//! Columnar result egress in the `ARW1` blob format — the dependency-free carrier
//! lenke-core lays a query result out in: a validity bitmap plus a typed
//! values/offsets buffer per column, inside one self-describing blob whose buffers
//! ARE Apache Arrow's physical column layout (little-endian, 8-byte aligned,
//! LSB-first validity bitmap, `i32` Utf8 offsets). Ported to match lenke-core's
//! byte layout for the scalar column types.
//!
//! ## Blob layout (all integers little-endian), from lenke-core's `arrow.rs`:
//! ```text
//! header (24 bytes):  magic "ARW1" | version:u32=1 | nrows:u64 | ncols:u64
//! column descriptors (ncols × 40 bytes), each 10×u32:
//!   type null_count name_off name_len validity_off validity_len
//!   buf1_off buf1_len buf2_off buf2_len
//! body: every referenced buffer, each 8-byte aligned; offsets are blob-relative.
//! ```
//! Type tags: 1 Float64, 2 Bool, 3 Utf8, 4 FixedSizeList<Float64>, 5 Struct.
//! Column type is inferred per column: all present cells record/map → `Struct`
//! (typed child columns, pre-order flattened); all present cells all-numeric
//! same-length lists → `FixedSizeList<Float64>`; all `Num` → Float64; all `Bool`
//! → Bool; anything else (or a mix) → Utf8 (each cell stringified).
//!
//! [`to_arrow_ipc`] layers the standard Apache Arrow IPC framing (flatbuffer
//! `Schema`/`RecordBatch`/`Footer` messages) over these exact buffers — stream or
//! file/Feather-v2 — so DuckDB/Polars/pandas consume the result directly. It is a
//! verbatim port of lenke-core's framing over the identical ARW1 blob, so the
//! bytes match lenke-core (and thus the TS/apache-arrow encoder) exactly.

use crate::exec::Rows;
use crate::value::Value;
use std::fmt::Write as _;

pub const T_FLOAT64: u32 = 1;
pub const T_BOOL: u32 = 2;
pub const T_UTF8: u32 = 3;
/// A fixed-dimension numeric list column → Arrow `FixedSizeList<Float64>[dim]`.
/// `buf1` is the flat child `f64` values (`nrows × dim`), validity is the
/// LIST-level bitmap, and `dim` rides the otherwise-empty `buf2_len` descriptor
/// slot — so the fixed 40-byte column descriptor is unchanged.
pub const T_FIXED_LIST: u32 = 4;
/// A record/map column → Arrow `Struct<field: type, …>`. The struct has NO values
/// buffer of its own — only a validity bitmap (a null row = a null/absent
/// record) — and one typed CHILD column per field. Its child COUNT rides the
/// otherwise-empty `buf2_len` slot, and its `n` child descriptors follow it in
/// the descriptor array in pre-order (a child may itself be a struct → nesting).
/// The header's `ncols` counts only TOP-LEVEL columns. A scalar-only blob has no
/// struct descriptors, so its bytes are unchanged (byte-identical to before this
/// type existed — and to lenke-core's scalar blob).
pub const T_STRUCT: u32 = 5;

const HEADER_LEN: usize = 24;
const COLDESC_LEN: usize = 40;

/// Round `v` up to a multiple of 8 (Arrow buffer alignment).
fn align8(v: usize) -> usize {
    (v + 7) & !7
}

/// Render a cell as Utf8 text for the mixed-column fallback (validity carries the
/// null, so a null is an empty span). Scalars match lenke-core (`Num` via `{n}`,
/// `Temporal` via its ISO form). A record/map/list only reaches here in a
/// type-MIXED column; a uniform one becomes a real `Struct`/`FixedSizeList`.
fn cell_str(c: &Value, out: &mut String) {
    match c {
        Value::Null => {}
        Value::Node(_) | Value::Edge(_) => {
            unreachable!("element ref is never a stored property value")
        }
        Value::Str(s) => out.push_str(s),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(n) => {
            let _ = write!(out, "{n}");
        }
        Value::Temporal(t) => out.push_str(&t.format()),
        Value::List(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                cell_str(it, out);
            }
            out.push(']');
        }
        Value::Record(fields) => {
            out.push('{');
            for (i, (k, v)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "{k}=");
                cell_str(v, out);
            }
            out.push('}');
        }
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                cell_str(k, out);
                out.push('=');
                cell_str(v, out);
            }
            out.push('}');
        }
    }
}

/// The sorted union of a struct column's field names + a per-cell field lookup.
/// A cell is "struct-like" if it is a `Record` (ISO record; keys already sorted,
/// string names) or a `Map` whose keys are all strings (the Gremlin map that can
/// name struct fields). This mirrors lenke-core, whose result-side `Map` (the
/// serialized record form) is what it turns into a `Struct`.
fn struct_fields(cell: &Value) -> Option<Vec<(&str, &Value)>> {
    match cell {
        Value::Record(fields) => Some(fields.iter().map(|(k, v)| (k.as_ref(), v)).collect()),
        Value::Map(pairs) => pairs
            .iter()
            .map(|(k, v)| match k {
                Value::Str(s) => Some((s.as_ref(), v)),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

/// Value for `key` in a struct-like cell, or `Null` if the cell omits it (or is
/// not struct-like).
fn field_value(cell: &Value, key: &str) -> Value {
    struct_fields(cell)
        .and_then(|fs| {
            fs.into_iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.clone())
        })
        .unwrap_or(Value::Null)
}

/// The shared dimension iff every present cell is an all-numeric list of the SAME
/// length (≥ 1) → a `FixedSizeList<Float64>` column; else `None` (ragged, empty,
/// non-numeric, or a non-list present cell → the caller falls back to `Utf8`).
fn fixed_numeric_dim(cells: &[Value]) -> Option<usize> {
    let mut dim: Option<usize> = None;
    let mut saw = false;
    for c in cells {
        match c {
            Value::Null => {}
            Value::List(items) if items.iter().all(|v| matches!(v, Value::Num(_))) => {
                saw = true;
                match dim {
                    None => dim = Some(items.len()),
                    Some(d) if d != items.len() => return None,
                    _ => {}
                }
            }
            _ => return None,
        }
    }
    dim.filter(|&d| saw && d >= 1)
}

/// A single result column in typed form (the Arrow physical types we emit),
/// mirroring lenke-core's `ArrowColumn` so the assembled bytes match. `valid =
/// None` means no nulls.
enum ArrowColumn {
    Num {
        data: Vec<f64>,
        valid: Option<Vec<bool>>,
    },
    Bool {
        data: Vec<bool>,
        valid: Option<Vec<bool>>,
    },
    Utf8 {
        offsets: Vec<i32>,
        bytes: Vec<u8>,
        valid: Option<Vec<bool>>,
    },
    /// `FixedSizeList<Float64>[dim]`. `data` is the flat child values (`nrows ×
    /// dim`, a null list contributing `dim` zeros); `valid` is the list-level mask.
    FixedList {
        dim: usize,
        data: Vec<f64>,
        valid: Option<Vec<bool>>,
    },
    /// `Struct`. `valid` is the struct-level null mask; each child is a full typed
    /// column of the same length. Children are ordered by field name (canonical).
    Struct {
        valid: Option<Vec<bool>>,
        children: Vec<(String, ArrowColumn)>,
    },
}

impl ArrowColumn {
    /// Infer a column's physical type from its cells, matching lenke-core:
    /// all-record/map → `Struct`; all-numeric-same-length-list → `FixedSizeList`;
    /// present Nums → Float64; present Bools → Bool; else Utf8.
    fn from_values(cells: &[Value]) -> Self {
        let n = cells.len();
        // A column whose present cells are all records/maps → a real Struct (typed
        // child columns), not a stringified blob. Fields = sorted union of names; a
        // row that omits a field contributes null to that child. Recurses.
        if !cells.is_empty()
            && cells
                .iter()
                .all(|c| c.is_null() || struct_fields(c).is_some())
            && cells.iter().any(|c| struct_fields(c).is_some())
        {
            let mut keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for c in cells {
                if let Some(fs) = struct_fields(c) {
                    for (k, _) in fs {
                        keys.insert(k.to_string());
                    }
                }
            }
            let valid: Vec<bool> = cells.iter().map(|c| struct_fields(c).is_some()).collect();
            let any_null = valid.iter().any(|&v| !v);
            let children: Vec<(String, Self)> = keys
                .iter()
                .map(|k| {
                    let child: Vec<Value> = cells.iter().map(|c| field_value(c, k)).collect();
                    (k.clone(), Self::from_values(&child))
                })
                .collect();
            return Self::Struct {
                valid: any_null.then_some(valid),
                children,
            };
        }
        // A fixed-dim numeric-list column → a real FixedSizeList<Float64>.
        if let Some(dim) = fixed_numeric_dim(cells) {
            let mut data = Vec::with_capacity(n * dim);
            let mut valid = vec![true; n];
            let mut any_null = false;
            for (i, c) in cells.iter().enumerate() {
                match c {
                    Value::List(items) => {
                        data.extend(items.iter().map(|v| match v {
                            Value::Num(x) => *x,
                            _ => 0.0,
                        }));
                    }
                    _ => {
                        valid[i] = false;
                        any_null = true;
                        data.extend(std::iter::repeat_n(0.0, dim));
                    }
                }
            }
            return Self::FixedList {
                dim,
                data,
                valid: any_null.then_some(valid),
            };
        }
        let (mut seen_num, mut seen_bool, mut seen_other) = (false, false, false);
        let mut any_null = false;
        let mut valid = vec![true; n];
        for (i, c) in cells.iter().enumerate() {
            match c {
                Value::Null => {
                    valid[i] = false;
                    any_null = true;
                }
                Value::Num(_) => seen_num = true,
                Value::Bool(_) => seen_bool = true,
                _ => seen_other = true,
            }
        }
        let valid = any_null.then_some(valid);
        if seen_other || (seen_num && seen_bool) {
            let mut offsets = Vec::with_capacity(n + 1);
            let mut bytes = Vec::new();
            let mut s = String::new();
            offsets.push(0i32);
            for c in cells {
                s.clear();
                cell_str(c, &mut s);
                bytes.extend_from_slice(s.as_bytes());
                offsets.push(bytes.len() as i32);
            }
            Self::Utf8 {
                offsets,
                bytes,
                valid,
            }
        } else if seen_bool {
            Self::Bool {
                data: cells
                    .iter()
                    .map(|c| matches!(c, Value::Bool(true)))
                    .collect(),
                valid,
            }
        } else {
            Self::Num {
                data: cells
                    .iter()
                    .map(|c| if let Value::Num(x) = c { *x } else { 0.0 })
                    .collect(),
                valid,
            }
        }
    }

    fn valid_mask(&self) -> &Option<Vec<bool>> {
        match self {
            Self::Num { valid, .. }
            | Self::Bool { valid, .. }
            | Self::Utf8 { valid, .. }
            | Self::FixedList { valid, .. }
            | Self::Struct { valid, .. } => valid,
        }
    }

    /// (tag, null_count, validity, buf1, buf2, extra) for blob assembly. `extra`
    /// is the list size for `FixedList` (0 otherwise); it rides `buf2_len`. A
    /// `Struct` never reaches here — it is flattened by [`flatten_descs`].
    fn encode(&self, nrows: usize) -> (u32, u32, Vec<u8>, Vec<u8>, Vec<u8>, u32) {
        let (null_count, validity) = encode_validity(self.valid_mask(), nrows);
        let (tag, buf1, buf2, extra) = match self {
            Self::Num { data, .. } => {
                let mut b = Vec::with_capacity(data.len() * 8);
                for v in data {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                (T_FLOAT64, b, Vec::new(), 0)
            }
            Self::Bool { data, .. } => {
                let mut b = vec![0u8; data.len().div_ceil(8)];
                for (i, &v) in data.iter().enumerate() {
                    if v {
                        b[i / 8] |= 1 << (i % 8);
                    }
                }
                (T_BOOL, b, Vec::new(), 0)
            }
            Self::Utf8 { offsets, bytes, .. } => {
                let mut b = Vec::with_capacity(offsets.len() * 4);
                for o in offsets {
                    b.extend_from_slice(&o.to_le_bytes());
                }
                (T_UTF8, b, bytes.clone(), 0)
            }
            Self::FixedList { dim, data, .. } => {
                let mut b = Vec::with_capacity(data.len() * 8);
                for v in data {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                (T_FIXED_LIST, b, Vec::new(), *dim as u32)
            }
            Self::Struct { .. } => {
                unreachable!("structs are flattened by flatten_descs, not encoded")
            }
        };
        (tag, null_count, validity, buf1, buf2, extra)
    }
}

/// Validity bitmap (LSB-first) + null count from a presence mask; `None` ⇒ all
/// valid ⇒ no bitmap.
fn encode_validity(mask: &Option<Vec<bool>>, nrows: usize) -> (u32, Vec<u8>) {
    match mask {
        None => (0, Vec::new()),
        Some(mask) => {
            let mut bitmap = vec![0u8; nrows.div_ceil(8)];
            let mut nulls = 0u32;
            for (i, &v) in mask.iter().enumerate() {
                if v {
                    bitmap[i / 8] |= 1 << (i % 8);
                } else {
                    nulls += 1;
                }
            }
            (nulls, bitmap)
        }
    }
}

/// One ARW1 descriptor entry (pre-order): tag, null count, name, buffers. `extra`
/// rides `buf2_len` — the list size for `FixedList`, the child count for
/// `Struct`, else 0. A `Struct` contributes only its validity buffer.
struct FlatDesc<'a> {
    tag: u32,
    null_count: u32,
    name: &'a str,
    validity: Vec<u8>,
    buf1: Vec<u8>,
    buf2: Vec<u8>,
    extra: u32,
}

/// Flatten a (possibly nested) column into pre-order descriptors: a struct
/// descriptor followed by its children's descriptors (recursively).
fn flatten_descs<'a>(
    name: &'a str,
    col: &'a ArrowColumn,
    nrows: usize,
    out: &mut Vec<FlatDesc<'a>>,
) {
    match col {
        ArrowColumn::Struct { valid, children } => {
            let (null_count, validity) = encode_validity(valid, nrows);
            out.push(FlatDesc {
                tag: T_STRUCT,
                null_count,
                name,
                validity,
                buf1: Vec::new(),
                buf2: Vec::new(),
                extra: children.len() as u32,
            });
            for (child_name, child) in children {
                flatten_descs(child_name, child, nrows, out);
            }
        }
        other => {
            let (tag, null_count, validity, buf1, buf2, extra) = other.encode(nrows);
            out.push(FlatDesc {
                tag,
                null_count,
                name,
                validity,
                buf1,
                buf2,
                extra,
            });
        }
    }
}

/// Encode a query result as an `ARW1` columnar blob. Byte-for-byte identical to
/// lenke-core's `to_arrow` for the same logical table (asserted by the
/// cross-engine `arrow_parity` test).
#[must_use]
pub fn to_arrow(rows: &Rows) -> Vec<u8> {
    let ncols = rows.names.len();
    let nrows = rows.rows.len();
    // Column-major typed columns.
    let cols: Vec<ArrowColumn> = (0..ncols)
        .map(|j| {
            let cells: Vec<Value> = rows.rows.iter().map(|r| r[j].clone()).collect();
            ArrowColumn::from_values(&cells)
        })
        .collect();

    // The header counts TOP-LEVEL columns; a struct's children ride along as extra
    // pre-order descriptors, so `flat` can be longer than `ncols`.
    let mut flat: Vec<FlatDesc> = Vec::new();
    for (name, col) in rows.names.iter().zip(&cols) {
        flatten_descs(name, col, nrows, &mut flat);
    }

    let body_base = align8(HEADER_LEN + flat.len() * COLDESC_LEN);
    let mut body: Vec<u8> = Vec::new();
    let mut descs: Vec<[u32; 10]> = Vec::with_capacity(flat.len());
    for fd in &flat {
        let mut place = |bytes: &[u8]| -> (u32, u32) {
            while !body.len().is_multiple_of(8) {
                body.push(0);
            }
            let off = (body_base + body.len()) as u32;
            body.extend_from_slice(bytes);
            (off, bytes.len() as u32)
        };
        let (name_off, name_len) = place(fd.name.as_bytes());
        let (val_off, val_len) = place(&fd.validity);
        let (b1_off, b1_len) = place(&fd.buf1);
        let (b2_off, mut b2_len) = place(&fd.buf2);
        // FixedSizeList (dim) and Struct (child count) have no buf2; their
        // otherwise-zero buf2_len carries that count so a reader can rebuild them.
        if fd.tag == T_FIXED_LIST || fd.tag == T_STRUCT {
            b2_len = fd.extra;
        }
        descs.push([
            fd.tag,
            fd.null_count,
            name_off,
            name_len,
            val_off,
            val_len,
            b1_off,
            b1_len,
            b2_off,
            b2_len,
        ]);
    }

    let mut blob = Vec::with_capacity(body_base + body.len());
    blob.extend_from_slice(b"ARW1");
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&(nrows as u64).to_le_bytes());
    blob.extend_from_slice(&(ncols as u64).to_le_bytes());
    for d in &descs {
        for w in d {
            blob.extend_from_slice(&w.to_le_bytes());
        }
    }
    while blob.len() < body_base {
        blob.push(0);
    }
    blob.extend_from_slice(&body);
    blob
}

// ── Apache Arrow IPC framing ────────────────────────────────────────────────
//
// Ported verbatim from lenke-core's `arrow.rs`: the ARW1 buffers above already ARE
// Arrow's physical column layout, so real Arrow IPC is those buffers concatenated
// (the RecordBatch body) plus the standard flatbuffer Schema / RecordBatch / Footer
// messages. Because lenke-engine's ARW1 blob is byte-identical to lenke-core's,
// this framing yields byte-identical IPC (asserted by `tests/arrow_parity.rs`).

const METADATA_V5: i16 = 4; // MetadataVersion.V5
const MSG_SCHEMA: u8 = 1; // MessageHeader.Schema
const MSG_RECORD_BATCH: u8 = 3; // MessageHeader.RecordBatch
const TYPE_FLOATINGPOINT: u8 = 3; // Type.FloatingPoint
const TYPE_UTF8: u8 = 5; // Type.Utf8
const TYPE_BOOL: u8 = 6; // Type.Bool
const TYPE_STRUCT: u8 = 13; // Type.Struct_
const TYPE_FIXEDSIZELIST: u8 = 16; // Type.FixedSizeList
const PRECISION_DOUBLE: i16 = 2; // Precision.DOUBLE

/// A minimal back-to-front FlatBuffers builder — mirrors the TS one in
/// `@lenke/native/arrow` exactly (tables + vtables, offset/struct vectors, strings,
/// inline scalars), so both engines emit byte-identical IPC. Values are written
/// toward the front of `buf`; offsets are measured from the end.
struct Fbb {
    buf: Vec<u8>,
    space: usize,
    minalign: usize,
    vtable: Vec<usize>,
    object_start: usize,
}

impl Fbb {
    fn new() -> Self {
        Self {
            buf: vec![0u8; 1024],
            space: 1024,
            minalign: 1,
            vtable: Vec::new(),
            object_start: 0,
        }
    }

    fn offset(&self) -> usize {
        self.buf.len() - self.space
    }

    fn grow(&mut self) {
        let old = self.buf.len();
        let mut nb = vec![0u8; old * 2];
        nb[old..].copy_from_slice(&self.buf);
        self.buf = nb;
        self.space += old;
    }

    fn prep(&mut self, size: usize, additional: usize) {
        if size > self.minalign {
            self.minalign = size;
        }
        let align_size = self.offset().wrapping_add(additional).wrapping_neg() & (size - 1);
        while self.space < align_size + size + additional {
            self.grow();
        }
        for _ in 0..align_size {
            self.space -= 1;
            self.buf[self.space] = 0;
        }
    }

    fn ensure(&mut self, n: usize) {
        while self.space < n {
            self.grow();
        }
    }

    fn pad(&mut self, n: usize) {
        self.ensure(n);
        for _ in 0..n {
            self.space -= 1;
            self.buf[self.space] = 0;
        }
    }

    fn add_u8(&mut self, v: u8) {
        self.prep(1, 0);
        self.space -= 1;
        self.buf[self.space] = v;
    }

    fn add_i16(&mut self, v: i16) {
        self.prep(2, 0);
        self.space -= 2;
        self.buf[self.space..self.space + 2].copy_from_slice(&v.to_le_bytes());
    }

    fn add_i32(&mut self, v: i32) {
        self.prep(4, 0);
        self.space -= 4;
        self.buf[self.space..self.space + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn add_i64(&mut self, v: i64) {
        self.prep(8, 0);
        self.space -= 8;
        self.buf[self.space..self.space + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// A forward uoffset to a previously-built object at rev-offset `off`.
    fn add_offset(&mut self, off: usize) {
        self.prep(4, 0);
        let val = (self.offset() - off + 4) as i32;
        self.space -= 4;
        self.buf[self.space..self.space + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn create_string(&mut self, s: &str) -> usize {
        let bytes = s.as_bytes();
        self.add_u8(0); // trailing null
        self.prep(4, bytes.len());
        self.ensure(bytes.len());
        self.space -= bytes.len();
        self.buf[self.space..self.space + bytes.len()].copy_from_slice(bytes);
        self.add_i32(bytes.len() as i32);
        self.offset()
    }

    fn start_vector(&mut self, elem_size: usize, num_elems: usize, alignment: usize) {
        self.prep(4, elem_size * num_elems);
        self.prep(alignment, elem_size * num_elems);
    }

    fn end_vector(&mut self, num_elems: usize) -> usize {
        self.add_i32(num_elems as i32);
        self.offset()
    }

    fn offset_vector(&mut self, offsets: &[usize]) -> usize {
        self.start_vector(4, offsets.len(), 4);
        for &off in offsets.iter().rev() {
            self.add_offset(off);
        }
        self.end_vector(offsets.len())
    }

    /// A vector of 16-byte `{a, b}` i64 structs (FieldNode / Buffer).
    fn struct_vector16(&mut self, structs: &[(i64, i64)]) -> usize {
        self.start_vector(16, structs.len(), 8);
        for &(a, b) in structs.iter().rev() {
            // Back-to-front → forward layout is [a, b].
            self.add_i64(b);
            self.add_i64(a);
        }
        self.end_vector(structs.len())
    }

    /// A vector of one 24-byte `Block` struct: offset:i64 @0, metaDataLength:i32 @8
    /// (+4 pad), bodyLength:i64 @16.
    fn block_vector(&mut self, offset: i64, metadata_len: i32, body_len: i64) -> usize {
        self.start_vector(24, 1, 8);
        // Back-to-front → forward [offset, metaDataLength, pad(4), bodyLength].
        self.add_i64(body_len);
        self.pad(4);
        self.add_i32(metadata_len);
        self.add_i64(offset);
        self.end_vector(1)
    }

    fn start_object(&mut self, numfields: usize) {
        self.vtable = vec![0usize; numfields];
        self.object_start = self.offset();
    }

    fn slot(&mut self, voffset: usize) {
        self.vtable[voffset] = self.offset();
    }

    fn add_field_i8(&mut self, voffset: usize, value: u8, def: u8) {
        if value != def {
            self.add_u8(value);
            self.slot(voffset);
        }
    }

    fn add_field_i16(&mut self, voffset: usize, value: i16, def: i16) {
        if value != def {
            self.add_i16(value);
            self.slot(voffset);
        }
    }

    fn add_field_i32(&mut self, voffset: usize, value: i32, def: i32) {
        if value != def {
            self.add_i32(value);
            self.slot(voffset);
        }
    }

    fn add_field_i64(&mut self, voffset: usize, value: i64, def: i64) {
        if value != def {
            self.add_i64(value);
            self.slot(voffset);
        }
    }

    fn add_field_offset(&mut self, voffset: usize, value: usize) {
        if value != 0 {
            self.add_offset(value);
            self.slot(voffset);
        }
    }

    fn end_object(&mut self) -> usize {
        self.add_i32(0); // soffset placeholder
        let vtableloc = self.offset();

        let mut i = self.vtable.len() as isize - 1;
        while i >= 0 && self.vtable[i as usize] == 0 {
            i -= 1;
        }
        let trimmed = (i + 1) as usize;

        while i >= 0 {
            let v = self.vtable[i as usize];
            self.add_i16(if v != 0 { (vtableloc - v) as i16 } else { 0 });
            i -= 1;
        }

        self.add_i16((vtableloc - self.object_start) as i16); // object size
        self.add_i16(((trimmed + 2) * 2) as i16); // vtable byte size

        // Point the object's soffset at the vtable we just wrote.
        let cur = self.offset();
        let pos = self.buf.len() - vtableloc;
        self.buf[pos..pos + 4].copy_from_slice(&((cur - vtableloc) as i32).to_le_bytes());
        vtableloc
    }

    fn finish(mut self, root: usize) -> Vec<u8> {
        self.prep(self.minalign, 4);
        self.add_offset(root);
        self.buf[self.space..].to_vec()
    }
}

/// One column's ARW1 view for IPC framing: name, Arrow type tag, null count, and
/// the Arrow buffers in order (validity, then values / offsets+data). A `Struct`
/// carries only its validity buffer and its typed `children` (each a full `IpcCol`).
struct IpcCol<'a> {
    name: &'a str,
    tag: u32,
    null_count: i64,
    /// List dimension for a `FixedList` column (the `FixedSizeList` listSize); 0
    /// otherwise. A FixedList contributes a second (child) field-node of length
    /// `nrows × dim`.
    dim: usize,
    buffers: Vec<&'a [u8]>,
    /// Child columns for a `Struct` (empty for every other type).
    children: Vec<Self>,
}

/// Read a little-endian `u32` at byte offset `o` in `b`.
fn u32le(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}

/// Build the Arrow `Field` sub-table for one column; returns its offset.
fn build_field(b: &mut Fbb, col: &IpcCol, empty_children: usize) -> usize {
    let name_off = b.create_string(col.name);
    let (type_type, type_off, children) = match col.tag {
        T_FLOAT64 => {
            b.start_object(1);
            b.add_field_i16(0, PRECISION_DOUBLE, 0);
            (TYPE_FLOATINGPOINT, b.end_object(), empty_children)
        }
        T_BOOL => {
            b.start_object(0);
            (TYPE_BOOL, b.end_object(), empty_children)
        }
        T_STRUCT => {
            // Nested types build inner-first: each child Field, then the children
            // vector, then this struct's (empty) type table.
            let child_fields: Vec<usize> = col
                .children
                .iter()
                .map(|c| build_field(b, c, empty_children))
                .collect();
            let children_vec = b.offset_vector(&child_fields);
            b.start_object(0); // Struct_ has no type-table fields
            (TYPE_STRUCT, b.end_object(), children_vec)
        }
        T_FIXED_LIST => {
            // The child `item: Float64` field (nested types build inner-first).
            let child_name = b.create_string("item");
            b.start_object(1);
            b.add_field_i16(0, PRECISION_DOUBLE, 0);
            let child_type = b.end_object();
            b.start_object(7);
            b.add_field_offset(0, child_name);
            b.add_field_i8(1, 1, 0); // nullable
            b.add_field_i8(2, TYPE_FLOATINGPOINT, 0);
            b.add_field_offset(3, child_type);
            b.add_field_offset(5, empty_children);
            let child_field = b.end_object();
            let children_vec = b.offset_vector(&[child_field]);
            // FixedSizeList type table: `listSize: int` (field 0).
            b.start_object(1);
            b.add_field_i32(0, col.dim as i32, 0);
            (TYPE_FIXEDSIZELIST, b.end_object(), children_vec)
        }
        _ => {
            b.start_object(0);
            (TYPE_UTF8, b.end_object(), empty_children)
        }
    };
    b.start_object(7);
    b.add_field_offset(0, name_off); // name
    b.add_field_i8(1, 1, 0); // nullable = true
    b.add_field_i8(2, type_type, 0); // type_type (union discriminant)
    b.add_field_offset(3, type_off); // type (union value)
    b.add_field_offset(5, children); // children ([item] for FixedSizeList, else empty)
    b.end_object()
}

/// Build the Arrow `Schema` sub-table for these columns; returns its offset.
fn build_schema(b: &mut Fbb, cols: &[IpcCol]) -> usize {
    let empty_children = b.offset_vector(&[]);
    let fields: Vec<usize> = cols
        .iter()
        .map(|c| build_field(b, c, empty_children))
        .collect();
    let fields_vec = b.offset_vector(&fields);
    b.start_object(4);
    b.add_field_offset(1, fields_vec); // fields (endianness defaults to Little)
    b.end_object()
}

/// A finished, framed `Schema` IPC message (metadata only, no body).
fn schema_message(cols: &[IpcCol]) -> Vec<u8> {
    let mut b = Fbb::new();
    let schema_off = build_schema(&mut b, cols);
    b.start_object(5);
    b.add_field_i16(0, METADATA_V5, 0);
    b.add_field_i8(1, MSG_SCHEMA, 0);
    b.add_field_offset(2, schema_off);
    let msg = b.end_object();
    encapsulate(&b.finish(msg), None)
}

/// A finished, framed `RecordBatch` IPC message with its data body.
fn record_batch_message(
    cols: &[IpcCol],
    nrows: usize,
    buffers: &[(i64, i64)],
    body: &[u8],
) -> Vec<u8> {
    let mut b = Fbb::new();
    // One field-node per field in depth-first pre-order: a FixedSizeList adds a
    // second node for its `item` child; a Struct adds a node per child (recursively).
    fn push_nodes(col: &IpcCol, nrows: usize, nodes: &mut Vec<(i64, i64)>) {
        nodes.push((nrows as i64, col.null_count));
        match col.tag {
            T_FIXED_LIST => nodes.push(((nrows * col.dim) as i64, 0)),
            T_STRUCT => {
                for c in &col.children {
                    push_nodes(c, nrows, nodes);
                }
            }
            _ => {}
        }
    }
    let mut nodes: Vec<(i64, i64)> = Vec::with_capacity(cols.len());
    for c in cols {
        push_nodes(c, nrows, &mut nodes);
    }
    let buffers_vec = b.struct_vector16(buffers);
    let nodes_vec = b.struct_vector16(&nodes);
    b.start_object(5);
    b.add_field_i64(0, nrows as i64, 0); // length
    b.add_field_offset(1, nodes_vec);
    b.add_field_offset(2, buffers_vec);
    let rb_off = b.end_object();
    b.start_object(5);
    b.add_field_i16(0, METADATA_V5, 0);
    b.add_field_i8(1, MSG_RECORD_BATCH, 0);
    b.add_field_offset(2, rb_off);
    b.add_field_i64(3, body.len() as i64, 0); // bodyLength
    let msg = b.end_object();
    encapsulate(&b.finish(msg), Some(body))
}

/// The file-layout `Footer`: the schema again + one Block per record batch.
fn footer_bytes(cols: &[IpcCol], rb_offset: i64, metadata_len: i32, body_len: i64) -> Vec<u8> {
    let mut b = Fbb::new();
    let schema_off = build_schema(&mut b, cols);
    let record_batches = b.block_vector(rb_offset, metadata_len, body_len);
    b.start_object(5);
    b.add_field_i16(0, METADATA_V5, 0);
    b.add_field_offset(1, schema_off);
    b.add_field_offset(3, record_batches);
    let footer = b.end_object();
    b.finish(footer)
}

/// Wrap a flatbuffer message in the IPC encapsulation (continuation + size +
/// padding, then the body). The body offset lands on an 8-byte boundary.
fn encapsulate(meta: &[u8], body: Option<&[u8]>) -> Vec<u8> {
    let meta_padded = (meta.len() + 7) & !7;
    let body_len = body.map_or(0, <[u8]>::len);
    let mut out = Vec::with_capacity(8 + meta_padded + body_len);
    out.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // continuation marker
    out.extend_from_slice(&(meta_padded as i32).to_le_bytes()); // metadata size (incl padding)
    out.extend_from_slice(meta);
    out.resize(8 + meta_padded, 0);
    if let Some(body) = body {
        out.extend_from_slice(body);
    }
    out
}

/// Transcode an `ARW1` columnar blob into standard Apache Arrow IPC bytes —
/// `file` selects the file / Feather-v2 layout, else the IPC stream layout.
/// Float64 / Bool / Utf8 / FixedSizeList / Struct columns (the tags `ARW1` emits)
/// are supported.
pub fn arrow_ipc_from_blob(blob: &[u8], file: bool) -> Vec<u8> {
    let nrows = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;
    let ncols = u64::from_le_bytes(blob[16..24].try_into().unwrap()) as usize;

    // Read one descriptor at pre-order index `*idx` into an `IpcCol`, advancing
    // `*idx` past it and (recursively) its struct children.
    fn read_col<'a>(blob: &'a [u8], idx: &mut usize) -> IpcCol<'a> {
        let d = 24 + *idx * 40;
        *idx += 1;
        let tag = u32le(blob, d) as u32;
        let null_count = u32le(blob, d + 4) as i64;
        let name = std::str::from_utf8(
            &blob[u32le(blob, d + 8)..u32le(blob, d + 8) + u32le(blob, d + 12)],
        )
        .unwrap_or("");
        let validity = &blob[u32le(blob, d + 16)..u32le(blob, d + 16) + u32le(blob, d + 20)];
        let buf1 = &blob[u32le(blob, d + 24)..u32le(blob, d + 24) + u32le(blob, d + 28)];
        // For a FixedSizeList / Struct, `buf2_len` carries a COUNT (dim / children),
        // not a buffer length — so read buf2 as a buffer only for Utf8.
        let (buffers, dim, children) = if tag == T_UTF8 {
            let buf2 = &blob[u32le(blob, d + 32)..u32le(blob, d + 32) + u32le(blob, d + 36)];
            (vec![validity, buf1, buf2], 0, Vec::new())
        } else if tag == T_FIXED_LIST {
            // Arrow buffer order: list validity, child validity (empty, no element
            // nulls), child values (buf1).
            (
                vec![validity, &[][..], buf1],
                u32le(blob, d + 36),
                Vec::new(),
            )
        } else if tag == T_STRUCT {
            // A struct has only a validity buffer; its children follow in pre-order.
            let n_children = u32le(blob, d + 36);
            let children: Vec<IpcCol> = (0..n_children).map(|_| read_col(blob, idx)).collect();
            (vec![validity], 0, children)
        } else {
            (vec![validity, buf1], 0, Vec::new())
        };
        IpcCol {
            name,
            tag,
            null_count,
            dim,
            buffers,
            children,
        }
    }

    let mut cols: Vec<IpcCol> = Vec::with_capacity(ncols);
    let mut idx = 0usize;
    for _ in 0..ncols {
        cols.push(read_col(blob, &mut idx));
    }

    // Body: every Arrow buffer concatenated on an 8-byte boundary in depth-first
    // order (a struct's own validity, then each child's buffers), the whole body
    // padded to 8; record each buffer's (offset, length).
    let mut body: Vec<u8> = Vec::new();
    let mut buffers: Vec<(i64, i64)> = Vec::new();
    fn push_buffers(col: &IpcCol, body: &mut Vec<u8>, buffers: &mut Vec<(i64, i64)>) {
        for b in &col.buffers {
            while !body.len().is_multiple_of(8) {
                body.push(0);
            }
            buffers.push((body.len() as i64, b.len() as i64));
            body.extend_from_slice(b);
        }
        for c in &col.children {
            push_buffers(c, body, buffers);
        }
    }
    for col in &cols {
        push_buffers(col, &mut body, &mut buffers);
    }
    while !body.len().is_multiple_of(8) {
        body.push(0);
    }

    let schema_msg = schema_message(&cols);
    let rb_msg = record_batch_message(&cols, nrows, &buffers, &body);

    let mut out = Vec::new();
    if !file {
        out.extend_from_slice(&schema_msg);
        out.extend_from_slice(&rb_msg);
        out.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]); // end-of-stream
        return out;
    }

    let magic = b"ARROW1\0\0";
    let rb_offset = (magic.len() + schema_msg.len()) as i64;
    let metadata_len = (rb_msg.len() - body.len()) as i32;
    let footer = footer_bytes(&cols, rb_offset, metadata_len, body.len() as i64);
    out.extend_from_slice(magic);
    out.extend_from_slice(&schema_msg);
    out.extend_from_slice(&rb_msg);
    out.extend_from_slice(&footer);
    out.extend_from_slice(&(footer.len() as i32).to_le_bytes());
    out.extend_from_slice(b"ARROW1");
    out
}

/// Encode a query result directly as Apache Arrow IPC bytes (the pure-Rust egress
/// path). `file` selects the file / Feather-v2 layout, else the IPC stream. The
/// bytes are byte-for-byte identical to lenke-core's `to_arrow_ipc` for the same
/// logical table (the ARW1 blob is identical, and this framing is a verbatim port).
#[must_use]
pub fn to_arrow_ipc(rows: &Rows, file: bool) -> Vec<u8> {
    arrow_ipc_from_blob(&to_arrow(rows), file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::Flat;
    use crate::value::make_record;
    use std::sync::Arc;

    fn u32_at(blob: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(blob[off..off + 4].try_into().unwrap())
    }
    fn u64_at(blob: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(blob[off..off + 8].try_into().unwrap())
    }

    /// A minimal hand-rolled ARW1 reader for round-trip assertions (the real
    /// apache-arrow verifier is TS-side). Returns per column: (name, tag, cells).
    fn decode(blob: &[u8]) -> Vec<(String, u32, Vec<Value>)> {
        assert_eq!(&blob[0..4], b"ARW1");
        assert_eq!(u32_at(blob, 4), 1);
        let nrows = u64_at(blob, 8) as usize;
        let ncols = u64_at(blob, 16) as usize;
        let mut out = Vec::new();
        for c in 0..ncols {
            let d = HEADER_LEN + c * COLDESC_LEN;
            let tag = u32_at(blob, d);
            let name = std::str::from_utf8(
                &blob[u32_at(blob, d + 8) as usize..][..u32_at(blob, d + 12) as usize],
            )
            .unwrap()
            .to_string();
            let val_off = u32_at(blob, d + 16) as usize;
            let val_len = u32_at(blob, d + 20) as usize;
            let valid = |i: usize| val_len == 0 || (blob[val_off + i / 8] >> (i % 8)) & 1 == 1;
            let b1 = u32_at(blob, d + 24) as usize;
            let b2 = u32_at(blob, d + 32) as usize;
            let cells: Vec<Value> = (0..nrows)
                .map(|i| {
                    if !valid(i) {
                        return Value::Null;
                    }
                    match tag {
                        T_FLOAT64 => Value::Num(f64::from_le_bytes(
                            blob[b1 + i * 8..b1 + i * 8 + 8].try_into().unwrap(),
                        )),
                        T_BOOL => Value::Bool((blob[b1 + i / 8] >> (i % 8)) & 1 == 1),
                        T_UTF8 => {
                            let lo = u32_at(blob, b1 + i * 4) as usize;
                            let hi = u32_at(blob, b1 + (i + 1) * 4) as usize;
                            Value::Str(std::str::from_utf8(&blob[b2 + lo..b2 + hi]).unwrap().into())
                        }
                        other => panic!("unexpected tag {other}"),
                    }
                })
                .collect();
            out.push((name, tag, cells));
        }
        out
    }

    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }

    #[test]
    fn arrow_blob_header_and_types() {
        let rows = Rows {
            names: vec!["age".into(), "name".into(), "ok".into()],
            rows: Flat::from_rows(vec![
                vec![n(30.0), s("alice"), Value::Bool(true)],
                vec![n(25.0), s("bob"), Value::Bool(false)],
            ]),
        };
        let blob = to_arrow(&rows);
        assert_eq!(&blob[0..4], b"ARW1");
        assert_eq!(u64_at(&blob, 8), 2); // nrows
        assert_eq!(u64_at(&blob, 16), 3); // ncols
                                          // Every buffer offset is 8-aligned (Float64/Int32 views must be valid).
        for c in 0..3 {
            let d = HEADER_LEN + c * COLDESC_LEN;
            assert_eq!(u32_at(&blob, d + 24) % 8, 0, "buf1 aligned");
        }
        let decoded = decode(&blob);
        assert_eq!(decoded[0].0, "age");
        assert_eq!(decoded[0].1, T_FLOAT64);
        assert_eq!(decoded[1].1, T_UTF8);
        assert_eq!(decoded[2].1, T_BOOL);
    }

    #[test]
    fn arrow_round_trips_cells_including_nulls() {
        let rows = Rows {
            names: vec!["v".into(), "s".into()],
            rows: Flat::from_rows(vec![
                vec![n(1.5), s("x")],
                vec![Value::Null, Value::Null], // a null in each column
                vec![n(-2.0), s("z")],
            ]),
        };
        let blob = to_arrow(&rows);
        let decoded = decode(&blob);
        // Float64 column with a null in the middle.
        let vcol = &decoded[0].2;
        assert!(matches!(vcol[0], Value::Num(x) if x == 1.5));
        assert!(vcol[1].is_null());
        assert!(matches!(vcol[2], Value::Num(x) if x == -2.0));
        // Utf8 column round-trips, null preserved via the validity bitmap.
        let scol = &decoded[1].2;
        assert!(matches!(&scol[0], Value::Str(x) if &**x == "x"));
        assert!(scol[1].is_null());
        assert!(matches!(&scol[2], Value::Str(x) if &**x == "z"));
        // The null column reports its null count in the descriptor.
        assert_eq!(u32_at(&blob, HEADER_LEN + 4), 1); // col 0 null_count
    }

    #[test]
    fn mixed_num_and_bool_becomes_utf8() {
        let rows = Rows {
            names: vec!["mixed".into()],
            rows: Flat::from_rows(vec![vec![n(1.0)], vec![Value::Bool(true)]]),
        };
        let decoded = decode(&to_arrow(&rows));
        assert_eq!(decoded[0].1, T_UTF8);
        assert!(matches!(&decoded[0].2[0], Value::Str(x) if &**x == "1"));
        assert!(matches!(&decoded[0].2[1], Value::Str(x) if &**x == "true"));
    }

    // --- nested columns (I2b): FixedSizeList<Float64> and Struct ---

    /// The `k`-th descriptor's fields, by name. (Descriptors are pre-order, so a
    /// struct's children follow it; there can be more descriptors than `ncols`.)
    fn desc(blob: &[u8], k: usize) -> (u32, u32, u32, u32) {
        let d = HEADER_LEN + k * COLDESC_LEN;
        // (tag, b1_off, b1_len, b2_len) — b2_len carries dim/child-count for
        // FixedList/Struct.
        (
            u32_at(blob, d),
            u32_at(blob, d + 24),
            u32_at(blob, d + 28),
            u32_at(blob, d + 36),
        )
    }
    fn f64_at(blob: &[u8], off: usize) -> f64 {
        f64::from_le_bytes(blob[off..off + 8].try_into().unwrap())
    }

    /// A column of same-length all-numeric lists → FixedSizeList<Float64>[dim],
    /// dim riding buf2_len, buf1 the flat nrows×dim child values.
    #[test]
    fn fixed_size_list_layout() {
        let rows = Rows {
            names: vec!["pair".into()],
            rows: Flat::from_rows(vec![
                vec![Value::List(vec![n(1.0), n(2.0)])],
                vec![Value::List(vec![n(3.0), n(4.0)])],
            ]),
        };
        let blob = to_arrow(&rows);
        assert_eq!(u64_at(&blob, 16), 1); // one top-level column
        let (tag, b1_off, b1_len, dim) = desc(&blob, 0);
        assert_eq!(tag, T_FIXED_LIST);
        assert_eq!(dim, 2, "dim rides buf2_len");
        assert_eq!(b1_len, 2 * 2 * 8, "nrows*dim f64s"); // 2 rows × 2 × 8 bytes
        let vals: Vec<f64> = (0..4)
            .map(|i| f64_at(&blob, b1_off as usize + i * 8))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]); // row-major flat child values
    }

    /// A record column → Struct: one top-level column, but three flattened
    /// descriptors in pre-order (struct, then children sorted by name: a, z).
    #[test]
    fn struct_layout_is_preorder() {
        let rows = Rows {
            names: vec!["rec".into()],
            rows: Flat::from_rows(vec![
                vec![make_record(vec![
                    (Arc::from("z"), s("x")),
                    (Arc::from("a"), n(1.0)),
                ])],
                vec![make_record(vec![
                    (Arc::from("z"), s("y")),
                    (Arc::from("a"), n(2.0)),
                ])],
            ]),
        };
        let blob = to_arrow(&rows);
        assert_eq!(u64_at(&blob, 16), 1); // header counts TOP-LEVEL columns only
        let (stag, _, _, nchild) = desc(&blob, 0);
        assert_eq!(stag, T_STRUCT);
        assert_eq!(nchild, 2, "child count rides buf2_len");
        // children sorted by name: a (Float64) then z (Utf8)
        let d1 = HEADER_LEN + COLDESC_LEN;
        let a_name = std::str::from_utf8(
            &blob[u32_at(&blob, d1 + 8) as usize..][..u32_at(&blob, d1 + 12) as usize],
        )
        .unwrap();
        assert_eq!(a_name, "a");
        assert_eq!(desc(&blob, 1).0, T_FLOAT64);
        let d2 = HEADER_LEN + 2 * COLDESC_LEN;
        let z_name = std::str::from_utf8(
            &blob[u32_at(&blob, d2 + 8) as usize..][..u32_at(&blob, d2 + 12) as usize],
        )
        .unwrap();
        assert_eq!(z_name, "z");
        assert_eq!(desc(&blob, 2).0, T_UTF8);
    }

    /// A struct-level null (a null row) is marked in the struct's validity, and the
    /// missing row contributes null to every child.
    #[test]
    fn struct_null_row_marked() {
        let rows = Rows {
            names: vec!["rec".into()],
            rows: Flat::from_rows(vec![
                vec![make_record(vec![(Arc::from("a"), n(1.0))])],
                vec![Value::Null],
            ]),
        };
        let blob = to_arrow(&rows);
        // struct descriptor null_count (field 1) == 1
        assert_eq!(u32_at(&blob, HEADER_LEN + 4), 1);
        // child `a` descriptor also has null_count 1 (the missing row).
        assert_eq!(u32_at(&blob, HEADER_LEN + COLDESC_LEN + 4), 1);
    }

    /// A Gremlin map with all-string keys becomes a Struct too (mirrors how
    /// lenke-core turns a result-side map into a struct).
    #[test]
    fn string_keyed_map_becomes_struct() {
        let map = Value::Map(Arc::new(vec![(s("a"), n(1.0)), (s("b"), n(2.0))]));
        let rows = Rows {
            names: vec!["m".into()],
            rows: Flat::from_rows(vec![vec![map]]),
        };
        let blob = to_arrow(&rows);
        assert_eq!(desc(&blob, 0).0, T_STRUCT);
        assert_eq!(desc(&blob, 0).3, 2); // two children
    }
}
