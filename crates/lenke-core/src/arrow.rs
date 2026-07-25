//! Zero-copy columnar result encoding in the **Apache Arrow** in-memory format.
//!
//! A query result is single-owner and consume-once, so instead of serializing it
//! to JSON (serialize on this side, parse on the caller's) we lay the columns out
//! as Arrow buffers — a validity bitmap plus a typed values/offsets buffer per
//! column — inside one self-describing blob. The caller (bun:ffi on the server,
//! wasm + `apache-arrow` in the browser) views those buffers in place with no
//! copy and no parse: numeric columns become a `Float64Array` over the same
//! bytes, strings an offsets+data pair, etc.
//!
//! The buffers themselves are exactly Arrow's columnar spec (little-endian,
//! 8-byte aligned, LSB-first validity bitmap, `i32` Utf8 offsets), so the JS side
//! reconstructs real `arrow.Vector`s via `makeData` with zero copy. The envelope
//! around them is a compact custom header (below) rather than Arrow's flatbuffer
//! IPC framing — that keeps this compact carrier dependency-free. The standard
//! Arrow **IPC** wrapper (for `tableFromIPC` / DuckDB / Polars / pandas) layers on
//! top of these exact buffers without changing them: natively via [`to_arrow_ipc`] /
//! [`arrow_ipc_from_blob`] (also `lnk_query_arrow_ipc` / `RustGraph.queryArrowIpc`,
//! no JS re-encode), or JS-side via `toArrowIPC` in `@lenke/native/arrow`. The two
//! encoders are byte-identical.
//!
//! ## Blob layout (all integers little-endian)
//! ```text
//! header (24 bytes):  magic "ARW1" | version:u32 | nrows:u64 | ncols:u64
//! column descriptors (ncols × 40 bytes), each:
//!   type:u32  null_count:u32
//!   name_off:u32 name_len:u32          (utf8 column name)
//!   validity_off:u32 validity_len:u32  (bitmap; len 0 ⇒ no nulls)
//!   buf1_off:u32 buf1_len:u32          (Float64: values; Bool: bitmap; Utf8: i32 offsets[n+1])
//!   buf2_off:u32 buf2_len:u32          (Utf8: data bytes; else len 0)
//! body: every referenced buffer, each 8-byte aligned; offsets are blob-relative.
//! ```
//! The blob's base pointer is 8-byte aligned by the FFI allocator, and every
//! buffer offset is a multiple of 8, so `Float64Array`/`Int32Array` views are
//! valid directly over `(base + off)`.

use std::fmt::Write as _;

use crate::graph::Value;
use crate::query::RowSet;

/// Arrow type tag (a minimal subset: every result cell maps to one of these).
pub const T_FLOAT64: u32 = 1;
pub const T_BOOL: u32 = 2;
pub const T_UTF8: u32 = 3;
/// A fixed-dimension numeric list column → Arrow `FixedSizeList<Float64>[dim]`.
/// In the ARW1 blob its `buf1` is the flat child `f64` values (`nrows × dim`), its
/// validity is the LIST-level bitmap, and its `dim` rides the (otherwise-empty)
/// `buf2_len` descriptor slot — so the fixed 40-byte column descriptor is unchanged.
pub const T_FIXED_LIST: u32 = 4;

const HEADER_LEN: usize = 24;
const COLDESC_LEN: usize = 40;

/// Round `v` up to a multiple of 8 (Arrow buffer alignment).
fn align8(v: usize) -> usize {
    (v + 7) & !7
}

/// Render a cell as text for a Utf8 column (validity carries the null, so a null
/// contributes an empty span).
fn cell_str(c: &Value, out: &mut String) {
    match c {
        Value::Null => {}
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
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(k);
                out.push('=');
                cell_str(v, out);
            }
            out.push('}');
        }
    }
}

/// One column's built Arrow buffers (pre-assembly).
/// A single result column in typed form (the Arrow physical types we emit).
/// `valid = None` means no nulls. The vectorized engine builds `Num`/`Bool`
/// straight from its `f64`/`bool` columns — no `Val`/`Value` boxing — while
/// `from_values` covers the scalar/RowSet path and string/element columns.
pub enum ArrowColumn {
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
    /// A fixed-dimension numeric list → `FixedSizeList<Float64>[dim]`. `data` is the
    /// flat child values (`nrows × dim`, a null list contributing `dim` zeros);
    /// `valid` is the LIST-level null mask.
    FixedList {
        dim: usize,
        data: Vec<f64>,
        valid: Option<Vec<bool>>,
    },
}

/// The shared dimension iff every present cell is an all-numeric list of the SAME
/// length (≥ 1) → a `FixedSizeList<Float64>` column; else `None` (ragged, empty,
/// non-numeric, or a non-list present cell → the caller falls back to `Utf8`).
fn fixed_numeric_dim(cells: &[&Value]) -> Option<usize> {
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

impl ArrowColumn {
    /// Build a column from generic `Value` cells (infers the physical type).
    pub fn from_values<'a>(cells: impl Iterator<Item = &'a Value>) -> Self {
        let cells: Vec<&Value> = cells.collect();
        let n = cells.len();
        // A fixed-dim numeric-list column → a real `FixedSizeList<Float64>` (not
        // Utf8 JSON): a null list contributes `dim` zeros to the flat child values.
        if let Some(dim) = fixed_numeric_dim(&cells) {
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
        let mut seen_num = false;
        let mut seen_bool = false;
        let mut seen_other = false;
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
        let valid = if any_null { Some(valid) } else { None };
        if seen_other || (seen_num && seen_bool) {
            let mut offsets = Vec::with_capacity(n + 1);
            let mut bytes = Vec::new();
            let mut s = String::new();
            offsets.push(0i32);
            for c in &cells {
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

    /// Build a Utf8 column from already-rendered strings (one per row, `None` =
    /// null). Lets the engine flatten element/string columns once, in place.
    pub fn utf8_from(strings: impl Iterator<Item = Option<String>>) -> Self {
        let mut offsets = vec![0i32];
        let mut bytes = Vec::new();
        let mut valid = Vec::new();
        let mut any_null = false;
        for s in strings {
            match s {
                Some(s) => {
                    bytes.extend_from_slice(s.as_bytes());
                    valid.push(true);
                }
                None => {
                    any_null = true;
                    valid.push(false);
                }
            }
            offsets.push(bytes.len() as i32);
        }
        Self::Utf8 {
            offsets,
            bytes,
            valid: if any_null { Some(valid) } else { None },
        }
    }

    fn valid_mask(&self) -> &Option<Vec<bool>> {
        match self {
            Self::Num { valid, .. }
            | Self::Bool { valid, .. }
            | Self::Utf8 { valid, .. }
            | Self::FixedList { valid, .. } => valid,
        }
    }

    /// (type tag, null_count, validity bitmap, buf1, buf2, dim) for blob assembly.
    /// `dim` is non-zero only for `FixedList` (the list size); it rides `buf2_len`.
    fn encode(&self, nrows: usize) -> (u32, u32, Vec<u8>, Vec<u8>, Vec<u8>, u32) {
        let (null_count, validity) = match self.valid_mask() {
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
        };
        let (tag, buf1, buf2, dim) = match self {
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
            // buf1 = flat child f64 values (nrows × dim); dim rides buf2_len.
            Self::FixedList { dim, data, .. } => {
                let mut b = Vec::with_capacity(data.len() * 8);
                for v in data {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                (T_FIXED_LIST, b, Vec::new(), *dim as u32)
            }
        };
        (tag, null_count, validity, buf1, buf2, dim)
    }
}

/// Encode a [`RowSet`] as an Arrow columnar blob (see module docs for layout).
/// Assemble an Arrow columnar blob from typed columns (see module docs for the
/// layout). `nrows` is the row count (columns must all be that long).
pub fn to_arrow_cols(names: &[String], cols: &[ArrowColumn], nrows: usize) -> Vec<u8> {
    let ncols = cols.len();
    #[allow(
        clippy::type_complexity,
        reason = "ad-hoc per-column (tag, null_count, validity, buf1, buf2) tuple local to encoding"
    )]
    let encoded: Vec<(u32, u32, Vec<u8>, Vec<u8>, Vec<u8>, u32)> =
        cols.iter().map(|c| c.encode(nrows)).collect();

    // Body base: after header + descriptors, aligned to 8.
    let body_base = align8(HEADER_LEN + ncols * COLDESC_LEN);
    let mut body: Vec<u8> = Vec::new();
    let mut descs: Vec<[u32; 10]> = Vec::with_capacity(ncols);
    for (j, (tag, null_count, validity, buf1, buf2, dim)) in encoded.iter().enumerate() {
        let mut place = |bytes: &[u8]| -> (u32, u32) {
            while !body.len().is_multiple_of(8) {
                body.push(0);
            }
            let off = (body_base + body.len()) as u32;
            body.extend_from_slice(bytes);
            (off, bytes.len() as u32)
        };
        let (name_off, name_len) = place(names[j].as_bytes());
        let (val_off, val_len) = place(validity);
        let (b1_off, b1_len) = place(buf1);
        let (b2_off, mut b2_len) = place(buf2);
        // A FixedSizeList has no buf2 buffer; its (otherwise-zero) buf2_len carries
        // the list dimension so a reader can rebuild the child field / node.
        if *tag == T_FIXED_LIST {
            b2_len = *dim;
        }
        descs.push([
            *tag,
            *null_count,
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

    // Assemble: header, descriptors, pad to body_base, body.
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

/// Encode a [`RowSet`] as an Arrow columnar blob (the scalar / fallback path,
/// inferring each column's type from its `Value` cells).
pub fn to_arrow(rs: &RowSet) -> Vec<u8> {
    let ncols = rs.cols.len();
    let cols: Vec<ArrowColumn> = (0..ncols)
        .map(|j| ArrowColumn::from_values((0..rs.nrows).map(move |i| &rs.data[i * ncols + j])))
        .collect();
    to_arrow_cols(&rs.cols, &cols, rs.nrows)
}

// ── Apache Arrow IPC framing ────────────────────────────────────────────────
//
// The `ARW1` buffers above already ARE Arrow's physical column layout, so real
// Arrow IPC is just those buffers concatenated (the RecordBatch body) plus the
// standard flatbuffer `Schema` / `RecordBatch` / `Footer` messages. This produces
// bytes byte-for-byte identical to the TS encoder in `@lenke/native/arrow`, so a
// Rust consumer (or the one-shot `lnk_query_arrow_ipc`) gets IPC without a JS hop.

// Arrow flatbuffer enum values we emit.
const METADATA_V5: i16 = 4; // MetadataVersion.V5
const MSG_SCHEMA: u8 = 1; // MessageHeader.Schema
const MSG_RECORD_BATCH: u8 = 3; // MessageHeader.RecordBatch
const TYPE_FLOATINGPOINT: u8 = 3; // Type.FloatingPoint
const TYPE_UTF8: u8 = 5; // Type.Utf8
const TYPE_BOOL: u8 = 6; // Type.Bool
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
/// the Arrow buffers in order (validity, then values / offsets+data).
struct IpcCol<'a> {
    name: &'a str,
    tag: u32,
    null_count: i64,
    /// List dimension for a `FixedList` column (the `FixedSizeList` listSize); 0
    /// otherwise. A FixedList contributes a second (child) field-node of length
    /// `nrows × dim`.
    dim: usize,
    buffers: Vec<&'a [u8]>,
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
    // One field-node per field in depth-first order: a FixedSizeList adds a second
    // node for its `item` child (length nrows × dim, no nulls).
    let mut nodes: Vec<(i64, i64)> = Vec::with_capacity(cols.len());
    for c in cols {
        nodes.push((nrows as i64, c.null_count));
        if c.tag == T_FIXED_LIST {
            nodes.push(((nrows * c.dim) as i64, 0));
        }
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
/// Float64 / Bool / Utf8 columns (the tags `ARW1` emits) are supported.
pub fn arrow_ipc_from_blob(blob: &[u8], file: bool) -> Vec<u8> {
    let nrows = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;
    let ncols = u64::from_le_bytes(blob[16..24].try_into().unwrap()) as usize;

    let mut cols: Vec<IpcCol> = Vec::with_capacity(ncols);
    for c in 0..ncols {
        let d = 24 + c * 40;
        let tag = u32le(blob, d) as u32;
        let null_count = u32le(blob, d + 4) as i64;
        let name = std::str::from_utf8(
            &blob[u32le(blob, d + 8)..u32le(blob, d + 8) + u32le(blob, d + 12)],
        )
        .unwrap_or("");
        let validity = &blob[u32le(blob, d + 16)..u32le(blob, d + 16) + u32le(blob, d + 20)];
        let buf1 = &blob[u32le(blob, d + 24)..u32le(blob, d + 24) + u32le(blob, d + 28)];
        // For a FixedSizeList, `buf2_len` carries the list dimension, not a buffer
        // length — so read buf2 only for Utf8.
        let (buffers, dim) = if tag == T_UTF8 {
            let buf2 = &blob[u32le(blob, d + 32)..u32le(blob, d + 32) + u32le(blob, d + 36)];
            (vec![validity, buf1, buf2], 0)
        } else if tag == T_FIXED_LIST {
            // Arrow buffer order: list validity, child validity (empty, no element
            // nulls), child values (buf1).
            (vec![validity, &[][..], buf1], u32le(blob, d + 36))
        } else {
            (vec![validity, buf1], 0)
        };
        cols.push(IpcCol {
            name,
            tag,
            null_count,
            dim,
            buffers,
        });
    }

    // Body: every Arrow buffer concatenated on an 8-byte boundary, whole body
    // padded to 8; record each buffer's (offset, length).
    let mut body: Vec<u8> = Vec::new();
    let mut buffers: Vec<(i64, i64)> = Vec::new();
    for col in &cols {
        for b in &col.buffers {
            while !body.len().is_multiple_of(8) {
                body.push(0);
            }
            buffers.push((body.len() as i64, b.len() as i64));
            body.extend_from_slice(b);
        }
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

/// Encode a [`RowSet`] directly as Apache Arrow IPC bytes (the pure-Rust egress
/// path — no JS round-trip). `file` selects the file / Feather layout.
pub fn to_arrow_ipc(rs: &RowSet, file: bool) -> Vec<u8> {
    arrow_ipc_from_blob(&to_arrow(rs), file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn u32_at(b: &[u8], off: usize) -> u32 {
        u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
    }
    fn u64_at(b: &[u8], off: usize) -> u64 {
        u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
    }

    /// Decode the blob back into (type, nulls, values) per column to verify the
    /// layout round-trips — the same reading a JS consumer does.
    #[allow(
        clippy::type_complexity,
        reason = "ad-hoc decoded-column tuple in a round-trip test helper"
    )]
    fn decode(blob: &[u8]) -> (usize, Vec<(u32, Vec<Option<String>>)>) {
        assert_eq!(&blob[0..4], b"ARW1");
        let nrows = u64_at(blob, 8) as usize;
        let ncols = u64_at(blob, 16) as usize;
        let mut out = Vec::new();
        for j in 0..ncols {
            let d = HEADER_LEN + j * COLDESC_LEN;
            let tag = u32_at(blob, d);
            let null_count = u32_at(blob, d + 4);
            let val_off = u32_at(blob, d + 16) as usize;
            let val_len = u32_at(blob, d + 20) as usize;
            let b1_off = u32_at(blob, d + 24) as usize;
            let b2_off = u32_at(blob, d + 32) as usize;
            let b2_len = u32_at(blob, d + 36) as usize;
            // alignment invariant: every buffer offset is 8-aligned.
            assert_eq!(b1_off % 8, 0, "buf1 must be 8-aligned");
            let valid = |i: usize| -> bool {
                if val_len == 0 {
                    return true; // no bitmap ⇒ all valid
                }
                blob[val_off + i / 8] & (1 << (i % 8)) != 0
            };
            let mut vals = Vec::with_capacity(nrows);
            let mut seen_nulls = 0;
            for i in 0..nrows {
                if !valid(i) {
                    seen_nulls += 1;
                    vals.push(None);
                    continue;
                }
                let s = match tag {
                    T_FLOAT64 => {
                        let o = b1_off + i * 8;
                        format!("{}", f64::from_le_bytes(blob[o..o + 8].try_into().unwrap()))
                    }
                    T_BOOL => (blob[b1_off + i / 8] & (1 << (i % 8)) != 0).to_string(),
                    _ => {
                        let start = u32_at(blob, b1_off + i * 4) as usize;
                        let end = u32_at(blob, b1_off + (i + 1) * 4) as usize;
                        assert!(end <= b2_len);
                        String::from_utf8(blob[b2_off + start..b2_off + end].to_vec()).unwrap()
                    }
                };
                vals.push(Some(s));
            }
            assert_eq!(seen_nulls as u32, null_count);
            out.push((tag, vals));
        }
        (nrows, out)
    }

    fn rowset(cols: &[&str], rows: Vec<Vec<Value>>) -> RowSet {
        let mut rs = RowSet::new(cols.iter().map(|s| s.to_string()).collect());
        for r in rows {
            rs.push_row(r);
        }
        rs
    }

    fn list(xs: &[f64]) -> Value {
        Value::List(xs.iter().map(|x| Value::Num(*x)).collect())
    }

    #[test]
    fn fixed_dim_numeric_list_encodes_as_fixed_size_list() {
        let rs = rowset(
            &["h"],
            vec![
                vec![list(&[1.5, 2.5, 3.5])],
                vec![Value::Null], // a null list → dim zeros in the child, list-invalid
                vec![list(&[4.0, 5.0, 6.0])],
            ],
        );
        let blob = to_arrow(&rs);
        let nrows = u64_at(&blob, 8) as usize;
        assert_eq!(nrows, 3);
        let d = HEADER_LEN;
        assert_eq!(u32_at(&blob, d), T_FIXED_LIST, "tag");
        assert_eq!(u32_at(&blob, d + 4), 1, "one null list");
        // dim rides buf2_len; buf1 is the flat child values (nrows × dim f64).
        let dim = u32_at(&blob, d + 36) as usize;
        assert_eq!(dim, 3);
        let b1_off = u32_at(&blob, d + 24) as usize;
        let b1_len = u32_at(&blob, d + 28) as usize;
        assert_eq!(b1_len, nrows * dim * 8);
        let read = |o: usize| f64::from_le_bytes(blob[o..o + 8].try_into().unwrap());
        assert_eq!(read(b1_off), 1.5);
        assert_eq!(read(b1_off + 2 * 8), 3.5);
        assert_eq!(read(b1_off + 3 * 8), 0.0, "null row's child span is zeros");
        assert_eq!(read(b1_off + 6 * 8), 4.0);
        // The full IPC framing must not panic (byte-validated vs apache-arrow in
        // packages/native/src/arrow.test.ts).
        let _ = to_arrow_ipc(&rs, false);
        let _ = to_arrow_ipc(&rs, true);
    }

    #[test]
    fn ragged_or_non_numeric_list_columns_fall_back_to_utf8() {
        // Differing lengths under one column → not a FixedSizeList.
        let ragged = rowset(&["h"], vec![vec![list(&[1.0, 2.0])], vec![list(&[3.0])]]);
        assert_eq!(u32_at(&to_arrow(&ragged), HEADER_LEN), T_UTF8);
        // A non-numeric element → not a FixedSizeList.
        let mixed = rowset(
            &["h"],
            vec![vec![Value::List(vec![
                Value::Num(1.0),
                Value::Str(Arc::from("x")),
            ])]],
        );
        assert_eq!(u32_at(&to_arrow(&mixed), HEADER_LEN), T_UTF8);
        // An all-null column is not a FixedSizeList either (no dim to infer).
        let empty = rowset(&["h"], vec![vec![Value::Null]]);
        assert_ne!(u32_at(&to_arrow(&empty), HEADER_LEN), T_FIXED_LIST);
    }

    #[test]
    fn float_bool_utf8_roundtrip() {
        let s = |x: &str| Value::Str(Arc::from(x));
        let rs = rowset(
            &["age", "flag", "name"],
            vec![
                vec![Value::Num(29.0), Value::Bool(true), s("marko")],
                vec![Value::Num(35.0), Value::Bool(false), s("peter")],
            ],
        );
        let blob = to_arrow(&rs);
        let (nrows, cols) = decode(&blob);
        assert_eq!(nrows, 2);
        assert_eq!(cols[0].0, T_FLOAT64);
        assert_eq!(cols[1].0, T_BOOL);
        assert_eq!(cols[2].0, T_UTF8);
        assert_eq!(cols[0].1, vec![Some("29".into()), Some("35".into())]);
        assert_eq!(cols[1].1, vec![Some("true".into()), Some("false".into())]);
        assert_eq!(cols[2].1, vec![Some("marko".into()), Some("peter".into())]);
    }

    #[test]
    fn nulls_set_validity_bitmap() {
        let s = |x: &str| Value::Str(Arc::from(x));
        let rs = rowset(
            &["n", "name"],
            vec![
                vec![Value::Num(1.0), s("a")],
                vec![Value::Null, Value::Null],
            ],
        );
        let blob = to_arrow(&rs);
        let (nrows, cols) = decode(&blob);
        assert_eq!(nrows, 2);
        assert_eq!(cols[0].1, vec![Some("1".into()), None]);
        assert_eq!(cols[1].1, vec![Some("a".into()), None]);
        // null_count recorded per column
        assert_eq!(u32_at(&blob, HEADER_LEN + 4), 1);
    }

    #[test]
    fn mixed_column_falls_back_to_utf8() {
        let s = |x: &str| Value::Str(Arc::from(x));
        let rs = rowset(&["x"], vec![vec![Value::Num(1.0)], vec![s("hi")]]);
        let blob = to_arrow(&rs);
        let (_, cols) = decode(&blob);
        assert_eq!(cols[0].0, T_UTF8);
        assert_eq!(cols[0].1, vec![Some("1".into()), Some("hi".into())]);
    }

    #[test]
    fn empty_result_is_valid_blob() {
        let rs = rowset(&["a", "b"], vec![]);
        let blob = to_arrow(&rs);
        let (nrows, cols) = decode(&blob);
        assert_eq!(nrows, 0);
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn end_to_end_query_to_arrow() {
        // Real path: decode a graph, run a GQL query, encode the RowSet as Arrow.
        let lines = [
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"name":"marko","age":29}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"name":"vadas","age":27}}"#,
        ];
        let mut g = crate::ndjson::decode(&lines.join("\n")).unwrap();
        let rs = crate::gql::parse("MATCH (n:P) RETURN n.name, n.age ORDER BY n.age")
            .unwrap()
            .execute(&mut g, &crate::gql::eval::Params::new())
            .unwrap();
        let (nrows, cols) = decode(&to_arrow(&rs));
        assert_eq!(nrows, 2);
        assert_eq!(cols[0].0, T_UTF8); // name
        assert_eq!(cols[1].0, T_FLOAT64); // age
        assert_eq!(cols[0].1, vec![Some("vadas".into()), Some("marko".into())]); // age-sorted
        assert_eq!(cols[1].1, vec![Some("27".into()), Some("29".into())]);
    }

    #[test]
    fn typed_path_matches_rowset_path() {
        // The boxing-free `execute_arrow` must produce byte-identical Arrow to the
        // RowSet path (`to_arrow(execute())`) for every shape — typed fast path
        // (plain projection) and fallback (aggregate / mixed / nulls) alike.
        let lines = [
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"name":"marko","age":29,"active":true}}"#,
            r#"{"type":"node","id":"b","labels":["P"],"properties":{"name":"vadas","age":27}}"#,
            r#"{"type":"node","id":"c","labels":["P"],"properties":{"name":"josh","age":32,"active":false}}"#,
        ];
        let queries = [
            "MATCH (n:P) RETURN n.name, n.age", // typed: Utf8 + Float64
            "MATCH (n:P) RETURN n.active",      // typed: Bool with a null
            "MATCH (n:P) WHERE n.age > 28 RETURN n.age", // typed + WHERE
            "MATCH (n:P) RETURN n.age * 2 + 1 AS x", // typed: computed numeric
            "MATCH (n:P) RETURN count(*) AS c", // fallback: aggregate
            "MATCH (n:P) RETURN n.dept",        // all-null column
        ];
        for q in queries {
            let mut g1 = crate::ndjson::decode(&lines.join("\n")).unwrap();
            let mut g2 = crate::ndjson::decode(&lines.join("\n")).unwrap();
            let params = crate::gql::eval::Params::new();
            let typed = crate::gql::parse(q)
                .unwrap()
                .execute_arrow(&mut g1, &params)
                .unwrap();
            let rs = crate::gql::parse(q)
                .unwrap()
                .execute(&mut g2, &params)
                .unwrap();
            assert_eq!(typed, to_arrow(&rs), "blob mismatch for `{q}`");
        }
    }

    #[test]
    fn ipc_framing_invariants_and_determinism() {
        let s = |x: &str| Value::Str(Arc::from(x));
        let rs = rowset(
            &["name", "age", "flag"],
            vec![
                vec![s("marko"), Value::Num(29.0), Value::Bool(true)],
                vec![s("vadas"), Value::Num(27.0), Value::Null],
            ],
        );

        let stream = to_arrow_ipc(&rs, false);
        // A stream starts with an encapsulated message (continuation marker) and ends
        // with the 8-byte end-of-stream marker (continuation + zero length).
        assert_eq!(&stream[0..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(
            &stream[stream.len() - 8..],
            &[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]
        );

        let file = to_arrow_ipc(&rs, true);
        // The file / Feather layout is bracketed by the ARROW1 magic.
        assert_eq!(&file[0..6], b"ARROW1");
        assert_eq!(&file[file.len() - 6..], b"ARROW1");

        // Deterministic: same rows → identical bytes (byte-identity to the TS encoder
        // is proven in packages/native/src/arrow.test.ts against apache-arrow).
        assert_eq!(to_arrow_ipc(&rs, false), stream);
        assert_eq!(arrow_ipc_from_blob(&to_arrow(&rs), true), file);

        // An empty result still frames validly (schema + zero-row batch).
        let empty = rowset(&["a", "b"], vec![]);
        let es = to_arrow_ipc(&empty, false);
        assert_eq!(&es[0..4], &[0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(es.len() > 8);
    }
}
