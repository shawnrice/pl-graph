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
//! Type tags: 1 Float64, 2 Bool, 3 Utf8. Column type is inferred per column: all
//! present cells `Num` → Float64; all present `Bool` → Bool; anything else (or a
//! mix) → Utf8 (each cell stringified). The `FixedSizeList`/`Struct` column types
//! and the flatbuffer Arrow-IPC wrapper that layers on these exact buffers are a
//! later slice (I2b).

use crate::exec::Rows;
use crate::value::Value;
use std::fmt::Write as _;

pub const T_FLOAT64: u32 = 1;
pub const T_BOOL: u32 = 2;
pub const T_UTF8: u32 = 3;

const HEADER_LEN: usize = 24;
const COLDESC_LEN: usize = 40;

/// Round `v` up to a multiple of 8 (Arrow buffer alignment).
fn align8(v: usize) -> usize {
    (v + 7) & !7
}

/// Render a cell as Utf8 text (validity carries the null, so a null is an empty
/// span). Scalars match lenke-core (`Num` via `{n}`, `Temporal` via its ISO form);
/// list/record/map get a compact form (their exact byte parity is deferred to I2b
/// with the Struct/list column types).
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

/// One encoded column: its tag, validity bitmap (empty ⇒ no nulls), null count,
/// and the two Arrow buffers (buf2 empty except for Utf8 data).
struct EncCol {
    tag: u32,
    null_count: u32,
    validity: Vec<u8>,
    buf1: Vec<u8>,
    buf2: Vec<u8>,
}

/// Validity bitmap (LSB-first) + null count from a presence mask; `None` ⇒ all
/// valid ⇒ no bitmap.
fn encode_validity(mask: Option<&[bool]>, nrows: usize) -> (u32, Vec<u8>) {
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

/// Infer a column's Arrow type from its cells and encode its buffers, matching
/// lenke-core's scalar inference (present Nums → Float64; present Bools → Bool;
/// else Utf8).
fn encode_column(cells: &[Value]) -> EncCol {
    let n = cells.len();
    let mut any_null = false;
    let mut valid = vec![true; n];
    let (mut seen_num, mut seen_bool, mut seen_other) = (false, false, false);
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
    let (null_count, validity) = encode_validity(any_null.then_some(valid.as_slice()), n);

    if seen_other || (seen_num && seen_bool) {
        // Utf8: i32 offsets[n+1] in buf1, data bytes in buf2.
        let mut offsets = Vec::with_capacity((n + 1) * 4);
        let mut bytes = Vec::new();
        offsets.extend_from_slice(&0i32.to_le_bytes());
        let mut s = String::new();
        for c in cells {
            s.clear();
            cell_str(c, &mut s);
            bytes.extend_from_slice(s.as_bytes());
            offsets.extend_from_slice(&(bytes.len() as i32).to_le_bytes());
        }
        EncCol {
            tag: T_UTF8,
            null_count,
            validity,
            buf1: offsets,
            buf2: bytes,
        }
    } else if seen_bool {
        let mut b = vec![0u8; n.div_ceil(8)];
        for (i, c) in cells.iter().enumerate() {
            if matches!(c, Value::Bool(true)) {
                b[i / 8] |= 1 << (i % 8);
            }
        }
        EncCol {
            tag: T_BOOL,
            null_count,
            validity,
            buf1: b,
            buf2: Vec::new(),
        }
    } else {
        // Float64 (also the all-null column's default): each cell's f64, 0.0 where
        // absent (validity marks the null).
        let mut b = Vec::with_capacity(n * 8);
        for c in cells {
            let x = if let Value::Num(x) = c { *x } else { 0.0 };
            b.extend_from_slice(&x.to_le_bytes());
        }
        EncCol {
            tag: T_FLOAT64,
            null_count,
            validity,
            buf1: b,
            buf2: Vec::new(),
        }
    }
}

/// Encode a query result as an `ARW1` columnar blob.
#[must_use]
pub fn to_arrow(rows: &Rows) -> Vec<u8> {
    let ncols = rows.names.len();
    let nrows = rows.rows.len();
    // Column-major cells.
    let cols: Vec<EncCol> = (0..ncols)
        .map(|j| {
            let cells: Vec<Value> = rows.rows.iter().map(|r| r[j].clone()).collect();
            encode_column(&cells)
        })
        .collect();

    let body_base = align8(HEADER_LEN + ncols * COLDESC_LEN);
    let mut body: Vec<u8> = Vec::new();
    let mut descs: Vec<[u32; 10]> = Vec::with_capacity(ncols);
    for (name, col) in rows.names.iter().zip(&cols) {
        let mut place = |bytes: &[u8]| -> (u32, u32) {
            while !body.len().is_multiple_of(8) {
                body.push(0);
            }
            let off = (body_base + body.len()) as u32;
            body.extend_from_slice(bytes);
            (off, bytes.len() as u32)
        };
        let (name_off, name_len) = place(name.as_bytes());
        let (val_off, val_len) = place(&col.validity);
        let (b1_off, b1_len) = place(&col.buf1);
        let (b2_off, b2_len) = place(&col.buf2);
        descs.push([
            col.tag,
            col.null_count,
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

#[cfg(test)]
mod tests {
    use super::*;
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
                            Value::Str(Arc::from(
                                std::str::from_utf8(&blob[b2 + lo..b2 + hi]).unwrap(),
                            ))
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
        Value::Str(Arc::from(x))
    }

    #[test]
    fn arrow_blob_header_and_types() {
        let rows = Rows {
            names: vec!["age".into(), "name".into(), "ok".into()],
            rows: vec![
                vec![n(30.0), s("alice"), Value::Bool(true)],
                vec![n(25.0), s("bob"), Value::Bool(false)],
            ],
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
            rows: vec![
                vec![n(1.5), s("x")],
                vec![Value::Null, Value::Null], // a null in each column
                vec![n(-2.0), s("z")],
            ],
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
            rows: vec![vec![n(1.0)], vec![Value::Bool(true)]],
        };
        let decoded = decode(&to_arrow(&rows));
        assert_eq!(decoded[0].1, T_UTF8);
        assert!(matches!(&decoded[0].2[0], Value::Str(x) if &**x == "1"));
        assert!(matches!(&decoded[0].2[1], Value::Str(x) if &**x == "true"));
    }
}
