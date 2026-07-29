//! Mutable columnar LPG: dense u32 vertex indices, dictionary-encoded
//! labels/keys/edge-types, typed contiguous property columns, and per-vertex
//! adjacency lists.
//!
//! This is a **working** in-memory graph, not a build-once artifact: vertices
//! and edges can be added, relabelled, re-propertied, and deleted at runtime
//! (deletes leave tombstones; live counts are tracked). Bulk decode builds it in
//! one pass. The property columns are contiguous so the GQL engine's vectorized
//! filter path (`gql::eval`) reads them without per-row `Val` boxing.
//!
//! Property model: a key's column is typed by its first non-null value
//! (Num=f64, Str=interned, Bool); a value that doesn't fit promotes the column
//! to a `Mixed` fallback so nothing is ever lost. Absent slots use a presence
//! bitset. `null` is a **first-class stored value** — present and distinct from
//! an absent slot (a stored null lives in a `Mixed` column as `Some(Null)`);
//! use [`Properties::is_present`] for presence, not "value == Null". Vertices
//! and edges share the same [`Properties`] store type.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::temporal::{Temporal, TemporalKind};

use crate::error::{CodeError, CodeResult};
use crate::error_codes::ErrorCode;

/// String interner backed by `Arc<str>`: `intern` is amortized O(1), `text`
/// reverses, and `arc` hands out a cheap shared clone (refcount bump, no alloc).
/// The interned `Arc` flows column → `Val` → output `Value` as refcount bumps,
/// so a string property is never re-allocated end to end. `Arc` (not `Rc`) keeps
/// the graph `Send` — needed for the parallel ndjson decode and a shared
/// read-only graph on the server.
#[derive(Default, Debug, Clone)]
pub struct Dict {
    map: HashMap<Arc<str>, u32>,
    pub strings: Vec<Arc<str>>,
}

impl Dict {
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.strings.len() as u32;
        let arc: Arc<str> = Arc::from(s);
        self.strings.push(arc.clone());
        self.map.insert(arc, id);
        id
    }
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
    pub fn text(&self, id: u32) -> &str {
        &self.strings[id as usize]
    }
    /// A shared clone of the interned string (refcount bump, no allocation).
    pub fn arc(&self, id: u32) -> Arc<str> {
        self.strings[id as usize].clone()
    }
    pub fn len(&self) -> usize {
        self.strings.len()
    }
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

/// Compact presence bitset (1 bit/element). Auto-grows on `set`; `get` is
/// bounds-safe (a slot never written reads as absent), which is what lets the
/// property columns grow one element at a time under mutation.
#[derive(Debug, Clone, Default)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    pub fn zeros(n: usize) -> Self {
        Self {
            words: vec![0u64; n.div_ceil(64)],
        }
    }
    #[inline]
    pub fn set(&mut self, i: usize) {
        let w = i >> 6;
        if w >= self.words.len() {
            self.words.resize(w + 1, 0);
        }
        self.words[w] |= 1u64 << (i & 63);
    }
    #[inline]
    pub fn clear(&mut self, i: usize) {
        if let Some(word) = self.words.get_mut(i >> 6) {
            *word &= !(1u64 << (i & 63));
        }
    }
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        self.words
            .get(i >> 6)
            .is_some_and(|w| (w >> (i & 63)) & 1 == 1)
    }
    /// Are all of the first `len` bits set? Answers "is this column fully present?"
    /// in O(len/64) whole-word checks — the fast test that lets a fused aggregate
    /// scan drop the per-element presence branch (and so autovectorize). A word
    /// that was never written reads as absent, so a short `words` vec is `false`.
    pub fn all_set(&self, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        let full_words = len >> 6;
        if full_words > self.words.len() {
            return false;
        }
        if self.words[..full_words].iter().any(|&w| w != u64::MAX) {
            return false;
        }
        let rem = len & 63;
        if rem == 0 {
            return true;
        }
        // The final partial word must have its low `rem` bits set.
        let mask = (1u64 << rem) - 1;
        self.words
            .get(full_words)
            .is_some_and(|&w| w & mask == mask)
    }
}

/// A JSON-ish scalar/list value, matching the vendor-neutral LPG value model.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(Arc<str>),
    /// An ISO temporal scalar (`DATE`/`LOCAL DATETIME`/`DURATION`). A stored
    /// property value, like Num/Str/Bool.
    Temporal(crate::temporal::Temporal),
    List(Vec<Self>),
    /// An ordered key→value object. NOT a stored property value (a property is
    /// only ever a scalar or list) — it appears only in query results, as the
    /// serialized form of a returned node/edge reference (`{id, labels,
    /// properties}`), so `RETURN n` yields something useful rather than a bare
    /// id. Keys are emitted sorted, for a deterministic, engine-agnostic shape.
    Map(Vec<(Arc<str>, Self)>),
}

/// A typed property column. Length == its store's element count.
#[derive(Clone, Debug)]
pub enum Column {
    /// Numbers as f64 (absent = NaN, also flagged in `present`).
    Num {
        data: Vec<f64>,
        present: BitSet,
    },
    /// Interned string ids (absent = u32::MAX).
    Str {
        data: Vec<u32>,
        present: BitSet,
    },
    Bool {
        data: Vec<bool>,
        present: BitSet,
    },
    /// A homogeneous temporal column: values de-boxed into per-type packed arrays
    /// (see [`TemporalCol`]) instead of a 40-byte `Option<Value>` slot. Absent
    /// slots hold a zero payload, flagged in `present`.
    Temporal {
        data: TemporalCol,
        present: BitSet,
    },
    /// A homogeneous fixed-dimension numeric-vector column: `dim` contiguous f64
    /// per element, row-major (`len × dim`), so an element's vector is a zero-copy
    /// `&[f64]` slice. Absent slots hold a zero payload, flagged in `present`. This
    /// is what recovers memory + read speed over boxing a `Value::List` in `Mixed`
    /// (8·dim B/elem contiguous vs a ~40 B slot + a heap `Vec` of boxed `Value::Num`).
    /// A list whose length differs or whose elements aren't all numbers promotes the
    /// column to `Mixed`, like any other type disagreement.
    Vec {
        data: Vec<f64>,
        dim: usize,
        present: BitSet,
    },
    /// Heterogeneous / list / mixed-type keys: keep the raw values.
    Mixed {
        data: Vec<Option<Value>>,
    },
    /// A **de-boxed record** column: the store for a property key that carries a
    /// declared `RECORD { … }` type constraint (R-CONSTRAINTS). Because the
    /// constraint is a contract on the field set + types, a conforming map is
    /// scattered across one typed sub-column per field (`meta.city` → a `Str`
    /// column, `meta.tier` → a `Num` column) instead of a ~40 B boxed
    /// `Value::Map` slot + a heap `Vec` of pairs. This recovers memory, closes the
    /// codec-decode gap (fields decode straight into their columns), lets a field
    /// vectorize, and makes `n.meta.city` a direct sub-column read.
    ///
    /// A property key is GLOBAL but a record constraint is per-`(label, key)`, so
    /// an element of another label may store `meta` as a scalar or a
    /// differently-shaped map — those can't be scattered, so they stay boxed in
    /// `escaped` (a sparse overlay, usually empty). This keeps de-boxing SOUND for
    /// a shared key while still de-boxing the conforming majority.
    Record {
        /// Element holds a CONFORMING map (fields scattered below) or a stored
        /// null (see `nulls`). NOT set for an absent element or an escaped one.
        present: BitSet,
        /// Among `present`, the value is a stored `Null` (not a map).
        nulls: BitSet,
        /// Declared field names, sorted (canonical map order) — parallel to `fields`.
        field_names: Vec<Arc<str>>,
        /// One typed sub-column per declared field; each sized to the element count.
        /// A nested `RECORD`-typed field is itself a de-boxed `Column::Record`
        /// (recursive); a `list` / `any record` field stays boxed (`Mixed`).
        fields: Vec<Self>,
        /// Per-field stored-null bitsets, parallel to `fields`: bit `idx` set means
        /// that field holds a *present null* for element `idx` (its `fields` slot is
        /// absent). This keeps a nullable field's sub-column TYPED — a stored null no
        /// longer forces it to `Mixed`. A field is absent when neither its `fields`
        /// slot is present nor its `field_nulls` bit is set.
        field_nulls: Vec<BitSet>,
        /// Non-conforming values (a scalar, a list, a differently-shaped map from
        /// another label) kept boxed by element index. An entry here overrides
        /// `present`/`fields` for that element.
        escaped: std::collections::HashMap<u32, Value>,
    },
}

/// The dimension of `items` iff it is a non-empty all-numeric list (→ a typed
/// `Column::Vec`), else `None` (→ `Mixed`). An empty list or any non-number
/// element stays boxed.
fn numeric_vec_dim(items: &[Value]) -> Option<usize> {
    (!items.is_empty() && items.iter().all(|v| matches!(v, Value::Num(_)))).then_some(items.len())
}

/// Element `idx`'s `dim`-wide slice of a row-major vector column's flat `data`.
fn vec_slice(data: &[f64], dim: usize, idx: usize) -> &[f64] {
    &data[idx * dim..idx * dim + dim]
}

/// Packed, struct-of-arrays storage for a homogeneous temporal column — one
/// variant per [`TemporalKind`], each holding the type's native integer
/// components in parallel `Vec`s. This is what recovers memory over `Mixed`
/// (`DATE` = one `Vec<i32>` = 4 B/slot vs 40 B) and lets scans read tight integer
/// loops. `DURATION` keeps its components separate (SoA) so an `ORDER BY` that
/// resolves on `months` streams only the 8-byte primary array.
#[derive(Clone, Debug)]
pub enum TemporalCol {
    Date(Vec<i32>),
    Time {
        secs: Vec<u32>,
        nanos: Vec<u32>,
    },
    DateTime {
        secs: Vec<i64>,
        nanos: Vec<u32>,
    },
    ZonedTime {
        secs: Vec<u32>,
        nanos: Vec<u32>,
        offset: Vec<i16>,
    },
    ZonedDateTime {
        secs: Vec<i64>,
        nanos: Vec<u32>,
        offset: Vec<i16>,
    },
    Duration {
        months: Vec<i64>,
        days: Vec<i64>,
        secs: Vec<i64>,
        nanos: Vec<u32>,
    },
}

impl TemporalCol {
    /// A fresh column of `len` zero-filled (absent) slots for `kind`.
    fn with_len(kind: TemporalKind, len: usize) -> Self {
        match kind {
            TemporalKind::Date => Self::Date(vec![0; len]),
            TemporalKind::Time => Self::Time {
                secs: vec![0; len],
                nanos: vec![0; len],
            },
            TemporalKind::DateTime => Self::DateTime {
                secs: vec![0; len],
                nanos: vec![0; len],
            },
            TemporalKind::ZonedTime => Self::ZonedTime {
                secs: vec![0; len],
                nanos: vec![0; len],
                offset: vec![0; len],
            },
            TemporalKind::ZonedDateTime => Self::ZonedDateTime {
                secs: vec![0; len],
                nanos: vec![0; len],
                offset: vec![0; len],
            },
            TemporalKind::Duration => Self::Duration {
                months: vec![0; len],
                days: vec![0; len],
                secs: vec![0; len],
                nanos: vec![0; len],
            },
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Date(d) => d.len(),
            Self::Time { secs, .. } | Self::ZonedTime { secs, .. } => secs.len(),
            Self::DateTime { secs, .. } | Self::ZonedDateTime { secs, .. } => secs.len(),
            Self::Duration { months, .. } => months.len(),
        }
    }

    /// Which [`TemporalKind`] this (homogeneous) column stores.
    pub(crate) fn kind(&self) -> TemporalKind {
        match self {
            Self::Date(_) => TemporalKind::Date,
            Self::Time { .. } => TemporalKind::Time,
            Self::DateTime { .. } => TemporalKind::DateTime,
            Self::ZonedTime { .. } => TemporalKind::ZonedTime,
            Self::ZonedDateTime { .. } => TemporalKind::ZonedDateTime,
            Self::Duration { .. } => TemporalKind::Duration,
        }
    }

    /// A single `i128` sort key at row `i` that is **monotonic with `cmp_total`
    /// within this (homogeneous) column** — a dense ORDER BY key that sorts like a
    /// numeric column, no `Val`/`cmp_total` dispatch. Each instant kind packs its
    /// components (highest-significance field first; the signed zoned `offset` is
    /// biased to unsigned so the packed order matches the derived `Ord`). Field
    /// ranges (`secs < 2^17` for a wall-clock, `nanos < 2^30`, `offset` an `i16`)
    /// keep the packs non-overlapping and inside `i128`. `Duration`'s 4-component
    /// lexicographic order does NOT reduce to one integer → `None` (that column
    /// falls back to the `cmp_total` comparator).
    ///
    /// SAFETY-OF-CORRECTNESS: this packing must stay monotonic with each struct's
    /// derived `Ord`; a change to a field's range or `Ord` order would silently
    /// break it. The `fuzz_temporal_order_by_vec_eq_scalar` differential (every
    /// kind, both engines) is what guards that invariant — the compiler can't.
    pub(crate) fn monotonic_key(&self, i: usize) -> Option<i128> {
        // `offset` (whole minutes, i16) biased to a non-negative 0..=65535.
        const OFF_BIAS: i128 = 32_768;
        Some(match self {
            Self::Date(d) => d[i] as i128,
            Self::Time { secs, nanos } => ((secs[i] as i128) << 32) | (nanos[i] as i128),
            Self::DateTime { secs, nanos } => ((secs[i] as i128) << 32) | (nanos[i] as i128),
            Self::ZonedTime {
                secs,
                nanos,
                offset,
            } => {
                ((secs[i] as i128) << 48)
                    | ((nanos[i] as i128) << 16)
                    | ((offset[i] as i128) + OFF_BIAS)
            }
            Self::ZonedDateTime {
                secs,
                nanos,
                offset,
            } => {
                ((secs[i] as i128) << 48)
                    | ((nanos[i] as i128) << 16)
                    | ((offset[i] as i128) + OFF_BIAS)
            }
            Self::Duration { .. } => return None,
        })
    }

    /// Bytes per stored slot across all component arrays (the packed width — 4 for
    /// `Date`, 28 for `Duration`, etc.). Diagnostic, for memory measurement.
    fn slot_bytes(&self) -> usize {
        use std::mem::size_of;
        match self {
            Self::Date(_) => size_of::<i32>(),
            Self::Time { .. } => 2 * size_of::<u32>(),
            Self::DateTime { .. } => size_of::<i64>() + size_of::<u32>(),
            Self::ZonedTime { .. } => 2 * size_of::<u32>() + size_of::<i16>(),
            Self::ZonedDateTime { .. } => size_of::<i64>() + size_of::<u32>() + size_of::<i16>(),
            Self::Duration { .. } => 3 * size_of::<i64>() + size_of::<u32>(),
        }
    }

    /// Append one zero (absent) slot to every component array.
    fn push_absent(&mut self) {
        match self {
            Self::Date(d) => d.push(0),
            Self::Time { secs, nanos } => {
                secs.push(0);
                nanos.push(0);
            }
            Self::DateTime { secs, nanos } => {
                secs.push(0);
                nanos.push(0);
            }
            Self::ZonedTime {
                secs,
                nanos,
                offset,
            } => {
                secs.push(0);
                nanos.push(0);
                offset.push(0);
            }
            Self::ZonedDateTime {
                secs,
                nanos,
                offset,
            } => {
                secs.push(0);
                nanos.push(0);
                offset.push(0);
            }
            Self::Duration {
                months,
                days,
                secs,
                nanos,
            } => {
                months.push(0);
                days.push(0);
                secs.push(0);
                nanos.push(0);
            }
        }
    }

    /// Reconstruct the `Temporal` at `i` (caller guarantees `present`).
    pub(crate) fn get(&self, i: usize) -> Temporal {
        use crate::temporal as t;
        match self {
            Self::Date(d) => Temporal::Date(t::Date { days: d[i] }),
            Self::Time { secs, nanos } => Temporal::Time(t::Time {
                secs: secs[i],
                nanos: nanos[i],
            }),
            Self::DateTime { secs, nanos } => Temporal::DateTime(t::DateTime {
                secs: secs[i],
                nanos: nanos[i],
            }),
            Self::ZonedTime {
                secs,
                nanos,
                offset,
            } => Temporal::ZonedTime(t::ZonedTime {
                secs: secs[i],
                nanos: nanos[i],
                offset: offset[i],
            }),
            Self::ZonedDateTime {
                secs,
                nanos,
                offset,
            } => Temporal::ZonedDateTime(t::ZonedDateTime {
                secs: secs[i],
                nanos: nanos[i],
                offset: offset[i],
            }),
            Self::Duration {
                months,
                days,
                secs,
                nanos,
            } => Temporal::Duration(t::Duration {
                months: months[i],
                days: days[i],
                secs: secs[i],
                nanos: nanos[i],
            }),
        }
    }

    /// Write `val`'s components at `i`, returning `false` if `val`'s kind doesn't
    /// match this column (the caller then promotes to `Mixed`).
    fn set(&mut self, i: usize, val: &Temporal) -> bool {
        match (self, val) {
            (Self::Date(d), Temporal::Date(v)) => d[i] = v.days,
            (Self::Time { secs, nanos }, Temporal::Time(v)) => {
                secs[i] = v.secs;
                nanos[i] = v.nanos;
            }
            (Self::DateTime { secs, nanos }, Temporal::DateTime(v)) => {
                secs[i] = v.secs;
                nanos[i] = v.nanos;
            }
            (
                Self::ZonedTime {
                    secs,
                    nanos,
                    offset,
                },
                Temporal::ZonedTime(v),
            ) => {
                secs[i] = v.secs;
                nanos[i] = v.nanos;
                offset[i] = v.offset;
            }
            (
                Self::ZonedDateTime {
                    secs,
                    nanos,
                    offset,
                },
                Temporal::ZonedDateTime(v),
            ) => {
                secs[i] = v.secs;
                nanos[i] = v.nanos;
                offset[i] = v.offset;
            }
            (
                Self::Duration {
                    months,
                    days,
                    secs,
                    nanos,
                },
                Temporal::Duration(v),
            ) => {
                months[i] = v.months;
                days[i] = v.days;
                secs[i] = v.secs;
                nanos[i] = v.nanos;
            }
            _ => return false,
        }
        true
    }
}

impl Column {
    /// Append one absent slot (grows the column by one element).
    fn push_absent(&mut self) {
        match self {
            Self::Num { data, .. } => data.push(f64::NAN),
            Self::Str { data, .. } => data.push(u32::MAX),
            Self::Bool { data, .. } => data.push(false),
            Self::Temporal { data, present: _ } => data.push_absent(),
            // An absent bit reads `false`, so `present` needs no growth here.
            Self::Vec { data, dim, .. } => data.extend(std::iter::repeat_n(0.0, *dim)),
            Self::Mixed { data } => data.push(None),
            // `present`/`nulls` bitsets read absent past their end, so only the
            // field sub-columns grow; `escaped` is sparse and untouched.
            Self::Record { fields, .. } => {
                for f in fields {
                    f.push_absent();
                }
            }
        }
    }
    fn element_len(&self) -> usize {
        match self {
            Self::Num { data, .. } => data.len(),
            Self::Str { data, .. } => data.len(),
            Self::Bool { data, .. } => data.len(),
            Self::Temporal { data, .. } => data.len(),
            Self::Vec { data, dim, .. } => data.len().checked_div(*dim).unwrap_or(0),
            Self::Mixed { data } => data.len(),
            // Every de-boxed record has ≥1 field (0-field records aren't de-boxed).
            Self::Record { fields, .. } => fields.first().map_or(0, Self::element_len),
        }
    }

    /// Heap bytes this column occupies: the packed data array(s) plus the presence
    /// bitset. Diagnostic — for memory profiling (see [`Graph::vertex_prop_bytes`]).
    fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        let bits = |p: &BitSet| p.words.len() * size_of::<u64>();
        match self {
            Self::Num { data, present } => data.len() * size_of::<f64>() + bits(present),
            Self::Str { data, present } => data.len() * size_of::<u32>() + bits(present),
            Self::Bool { data, present } => data.len() * size_of::<bool>() + bits(present),
            Self::Temporal { data, present } => data.len() * data.slot_bytes() + bits(present),
            Self::Vec { data, present, .. } => data.len() * size_of::<f64>() + bits(present),
            Self::Mixed { data } => data.len() * size_of::<Option<Value>>(),
            Self::Record {
                present,
                nulls,
                fields,
                field_nulls,
                escaped,
                ..
            } => {
                bits(present)
                    + bits(nulls)
                    + field_nulls.iter().map(bits).sum::<usize>()
                    + fields.iter().map(Self::heap_bytes).sum::<usize>()
                    + escaped.len() * (size_of::<u32>() + size_of::<Option<Value>>())
            }
        }
    }
}

/// A columnar property store: typed columns keyed by property-key id, each of
/// length `len` elements. Vertices and edges use this **identically** — a
/// property is a property regardless of whether its element is a node or a
/// relationship. The graph holds two: one indexed by vertex, one by edge.
#[derive(Debug, Default, Clone)]
pub struct Properties {
    pub keys: Dict,
    /// Columns indexed by **dense key id** (`keys.intern` order), so a resolved
    /// id is an array index — no per-access hash. Every interned key has a column
    /// (an all-null key gets an empty `Mixed`).
    pub cols: Vec<Column>,
    /// Element count the columns are sized to (vertex count, or edge count).
    pub len: usize,
}

impl Properties {
    /// The column for `key`, if any.
    pub fn col(&self, key: &str) -> Option<&Column> {
        self.keys.get(key).map(|kid| &self.cols[kid as usize])
    }

    /// Value at element `idx` for `key` as a core [`Value`] (absent → `Null`).
    /// `strs` is the graph-wide interner backing `Column::Str`.
    pub fn value(&self, idx: usize, key: &str, strs: &Dict) -> Value {
        match self.keys.get(key) {
            Some(kid) => self.value_id(idx, kid, strs),
            None => Value::Null,
        }
    }

    /// Value at element `idx` for the already-resolved key id `kid` — the hot
    /// path: an array index, no hashing.
    pub fn value_id(&self, idx: usize, kid: u32, strs: &Dict) -> Value {
        match self.cols.get(kid as usize) {
            Some(Column::Num { data, present }) if present.get(idx) => Value::Num(data[idx]),
            Some(Column::Bool { data, present }) if present.get(idx) => Value::Bool(data[idx]),
            Some(Column::Str { data, present }) if present.get(idx) => {
                Value::Str(strs.arc(data[idx]))
            }
            Some(Column::Temporal { data, present }) if present.get(idx) => {
                Value::Temporal(data.get(idx))
            }
            Some(Column::Vec { data, dim, present }) if present.get(idx) => Value::List(
                vec_slice(data, *dim, idx)
                    .iter()
                    .map(|x| Value::Num(*x))
                    .collect(),
            ),
            Some(Column::Mixed { data }) => data[idx].clone().unwrap_or(Value::Null),
            Some(
                col @ Column::Record {
                    field_names: _,
                    fields: _,
                    ..
                },
            ) => col_get(col, idx, strs).unwrap_or(Value::Null),
            _ => Value::Null,
        }
    }

    /// Read element `idx`'s value at `descent` under key id `kid`, without
    /// materializing the whole record. For a de-boxed [`Column::Record`], the first
    /// descent segment resolves to a sub-column and reads it DIRECTLY (the
    /// `n.meta.city` win — one typed read, no map allocation); a deeper descent then
    /// walks the boxed sub-value. For a boxed map in a `Mixed` column, walks the
    /// stored map. Absent / missing / not-a-map at a hop → `Null`.
    pub(crate) fn field_at(&self, idx: usize, kid: u32, descent: &[&str], strs: &Dict) -> Value {
        match self.cols.get(kid as usize) {
            Some(Column::Record {
                present,
                nulls,
                field_names,
                fields,
                field_nulls,
                escaped,
            }) => {
                if let Some(v) = escaped.get(&(idx as u32)) {
                    return value_at_descent(v, descent).cloned().unwrap_or(Value::Null);
                }
                if descent.is_empty() {
                    return if present.get(idx) && !nulls.get(idx) {
                        record_map(field_names, fields, field_nulls, idx, strs)
                    } else {
                        Value::Null
                    };
                }
                if !present.get(idx) || nulls.get(idx) {
                    return Value::Null;
                }
                match field_names.binary_search_by(|n| n.as_ref().cmp(descent[0])) {
                    // A stored-null field reads back as `Null` directly (its typed
                    // sub-column slot is absent).
                    Ok(fi) if field_nulls[fi].get(idx) => Value::Null,
                    Ok(fi) => match col_get(&fields[fi], idx, strs) {
                        Some(fv) if descent.len() == 1 => fv,
                        Some(fv) => value_at_descent(&fv, &descent[1..])
                            .cloned()
                            .unwrap_or(Value::Null),
                        None => Value::Null,
                    },
                    Err(_) => Value::Null,
                }
            }
            Some(Column::Mixed { data }) => match data.get(idx).and_then(Option::as_ref) {
                Some(root) => value_at_descent(root, descent)
                    .cloned()
                    .unwrap_or(Value::Null),
                None => Value::Null,
            },
            _ => Value::Null,
        }
    }

    /// Zero-copy view of element `idx`'s `key` as a numeric-vector slice — the fast
    /// read path (`neighborAggregate`) when the key is a typed [`Column::Vec`] and
    /// present. `None` when absent or when the list is still boxed in a `Mixed`
    /// column (the caller then falls back to [`value`](Self::value)).
    pub fn vector_id(&self, idx: usize, kid: u32) -> Option<&[f64]> {
        match self.cols.get(kid as usize) {
            Some(Column::Vec { data, dim, present }) if present.get(idx) => {
                Some(vec_slice(data, *dim, idx))
            }
            _ => None,
        }
    }

    /// [`vector_id`](Self::vector_id) by key name.
    pub fn vector(&self, idx: usize, key: &str) -> Option<&[f64]> {
        self.keys.get(key).and_then(|kid| self.vector_id(idx, kid))
    }

    /// Does element `idx` HAVE property `key` — regardless of whether its value
    /// is a stored `Null`? This is the true presence test. `value(...) == Null`
    /// is NOT presence: it's also true for an absent key, because `Null` is a
    /// first-class stored value here (see [`set_value`](Self::set_value)) that a
    /// read cannot distinguish from absence. Enumeration/serialization must gate
    /// on this, not on the value.
    pub fn is_present(&self, idx: usize, key: &str) -> bool {
        self.keys
            .get(key)
            .is_some_and(|kid| self.is_present_id(idx, kid))
    }

    /// True if any stored value in this store contains a map/record (at any
    /// depth). The flat codecs (pg-text / csv) use this to reject an export they
    /// can't faithfully carry — a map only ever lives boxed in a `Mixed` column.
    pub fn has_map_value(&self) -> bool {
        self.cols.iter().any(|c| match c {
            Column::Mixed { data } => data
                .iter()
                .any(|v| v.as_ref().is_some_and(value_contains_map)),
            // A de-boxed record IS a map; a scalar escapee might not be, but the
            // conforming fields make this key un-flattenable regardless.
            Column::Record { .. } => true,
            _ => false,
        })
    }

    /// [`is_present`](Self::is_present) for an already-resolved key id.
    pub fn is_present_id(&self, idx: usize, kid: u32) -> bool {
        match self.cols.get(kid as usize) {
            Some(
                Column::Num { present, .. }
                | Column::Str { present, .. }
                | Column::Bool { present, .. }
                | Column::Temporal { present, .. }
                | Column::Vec { present, .. },
            ) => present.get(idx),
            Some(Column::Mixed { data }) => data[idx].is_some(),
            // Present iff it holds a value (map/null) OR an escaped one.
            Some(Column::Record {
                present, escaped, ..
            }) => escaped.contains_key(&(idx as u32)) || present.get(idx),
            None => false,
        }
    }

    /// Append one element slot (absent in every existing column).
    fn push_element(&mut self) {
        for col in &mut self.cols {
            col.push_absent();
        }
        self.len += 1;
    }

    /// Set element `idx`'s `key` to `v`, creating the column if needed and
    /// promoting it to `Mixed` if `v`'s type disagrees with the existing one.
    ///
    /// `Null` is a FIRST-CLASS stored value: `set_value(idx, key, Null)` stores a
    /// *present* null (promoting the column to `Mixed`) — it does NOT remove the
    /// property. A stored null and an absent key are distinct (`is_present`
    /// tells them apart), though both read back as `Null` and are `IS NULL`
    /// (SQL/GQL three-valued logic). Removal is explicit: [`remove_value`], GQL
    /// `REMOVE`, or Gremlin `.properties(k).drop()`. This mirrors the TS engine
    /// and GQL's null-typed value model — and is a deliberate divergence from
    /// Cypher/TinkerPop, where `SET x = null` (and null property values) mean
    /// removal.
    pub fn set_value(&mut self, idx: usize, key: &str, v: Value, strs: &mut Dict) {
        let kid = self.keys.intern(key) as usize;
        if kid >= self.cols.len() {
            // brand-new key: a column of `len` absent slots, then set below.
            self.cols.push(empty_col_for(&v, self.len));
        }
        let col = &mut self.cols[kid];
        if !col_set(col, idx, &v, strs) {
            // type mismatch — promote the column to Mixed, then set.
            *col = to_mixed(col, strs);
            col_set(col, idx, &v, strs);
        }
    }

    /// Remove element `idx`'s `key` (no-op if absent).
    pub fn remove_value(&mut self, idx: usize, key: &str) {
        if let Some(kid) = self.keys.get(key) {
            if let Some(col) = self.cols.get_mut(kid as usize) {
                col_clear(col, idx);
            }
        }
    }

    /// De-box the column for `key` into a [`Column::Record`] typed by `spec` — the
    /// backfill a `RECORD`-typed constraint declaration triggers. Existing values
    /// are scattered (a conforming map → its field sub-columns; a null → the null
    /// marker; anything else → the escape overlay). Idempotent: a key already a
    /// `Record` (an earlier constraint on the same key) is left as-is. A 0-field
    /// record isn't de-boxed (degenerate).
    pub(crate) fn debox_record(&mut self, key: &str, spec: &TypeSpec, strs: &mut Dict) {
        let TypeSpec::Record(defs) = spec else { return };
        if defs.is_empty() {
            return;
        }
        let kid = self.keys.intern(key) as usize;
        if kid >= self.cols.len() {
            self.cols.push(Column::Mixed {
                data: vec![None; self.len],
            });
        }
        if matches!(self.cols[kid], Column::Record { .. }) {
            return;
        }
        let mut rec = empty_record_column(defs, self.len);
        let old = std::mem::replace(&mut self.cols[kid], Column::Mixed { data: Vec::new() });
        for i in 0..self.len {
            if let Some(v) = col_get(&old, i, strs) {
                col_set(&mut rec, i, &v, strs);
            }
        }
        self.cols[kid] = rec;
    }

    /// Re-box a de-boxed [`Column::Record`] back to a plain `Mixed` column — the
    /// inverse of [`debox_record`], run when the last record constraint on `key` is
    /// dropped so the shape is no longer a contract. No-op if `key` isn't de-boxed.
    pub(crate) fn rebox_record(&mut self, key: &str, strs: &Dict) {
        if let Some(kid) = self.keys.get(key) {
            let kid = kid as usize;
            if matches!(self.cols.get(kid), Some(Column::Record { .. })) {
                self.cols[kid] = to_mixed(&self.cols[kid], strs);
            }
        }
    }
}

/// One adjacency slot yielded while expanding a vertex: the edge's index, the
/// vertex on the other end, and the edge type id.
#[derive(Clone, Copy, Debug)]
pub struct Adj {
    pub eidx: u32,
    pub nbr: u32,
    pub etype: u32,
}

/// The scalar type a TYPE constraint (R-CONSTRAINTS) can require of a property
/// value. Mirrors the TS `ScalarTypeName`; `number` maps to `Num` (the f64 model
/// has no integer/float split), `list` is "an array" (elements unconstrained).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropType {
    Str,
    Num,
    Bool,
    Date,
    Time,
    DateTime,
    ZonedTime,
    ZonedDateTime,
    Duration,
    List,
}

impl PropType {
    fn from_name(s: &str) -> Option<Self> {
        match s {
            "string" => Some(Self::Str),
            "number" => Some(Self::Num),
            "boolean" => Some(Self::Bool),
            "date" => Some(Self::Date),
            "localtime" => Some(Self::Time),
            "datetime" => Some(Self::DateTime),
            "zoned_time" => Some(Self::ZonedTime),
            "zoned_datetime" => Some(Self::ZonedDateTime),
            "duration" => Some(Self::Duration),
            "list" => Some(Self::List),
            _ => None,
        }
    }

    /// The exact inverse of [`from_name`](Self::from_name) — the scalar-type name a
    /// `createTypeConstraint` op carries. Used by [`Graph::dump_schema`] to round-trip
    /// a declared type constraint back into a replayable op string.
    fn to_name(self) -> &'static str {
        match self {
            Self::Str => "string",
            Self::Num => "number",
            Self::Bool => "boolean",
            Self::Date => "date",
            Self::Time => "localtime",
            Self::DateTime => "datetime",
            Self::ZonedTime => "zoned_time",
            Self::ZonedDateTime => "zoned_datetime",
            Self::Duration => "duration",
            Self::List => "list",
        }
    }
}

/// A declared property TYPE for an R-CONSTRAINT: either a scalar [`PropType`] or
/// an ISO `RECORD { field :: type, … }` (a *closed* record — an exact field set,
/// each field itself a `TypeSpec`, so records nest). This is the contract a
/// record-typed constraint enforces; a fixed-shape record is what later lets the
/// store de-box it into columns. `PropType` stays `Copy`; only the (rare) record
/// type carries the nested `Vec`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TypeSpec {
    Scalar(PropType),
    /// A closed record: sorted fields. Each is `(name, type, not_null)`. A field
    /// is nullable/optional by default (absent OR a null value both satisfy it) —
    /// ISO makes `NOT NULL` the explicit marker, so it is the *required* flag: a
    /// `not_null` field must be present with a non-null value. Closed on extras: a
    /// value may not carry a field outside the declared set. Sorted names →
    /// structurally-equal record types compare equal.
    Record(Vec<(Arc<str>, Self, bool)>),
    /// The ISO OPEN record type — `ANY RECORD` (or a bare `RECORD` with no field
    /// spec). Matches any map value regardless of shape; carries no field
    /// contract, so it is NOT de-boxed (the column stays boxed `Mixed`).
    AnyRecord,
}

impl TypeSpec {
    /// Parse a constraint type name: a scalar (`string`, `number`, …) or a record
    /// `record { field :: type, … }` (`:` or `::` accepted, whitespace ignored,
    /// nesting allowed). `None` on any malformed input — the caller faults.
    pub fn parse(s: &str) -> Option<Self> {
        let mut p = TypeParser {
            s: s.as_bytes(),
            i: 0,
        };
        let t = p.parse_type()?;
        p.skip_ws();
        (p.i == p.s.len()).then_some(t)
    }

    /// Like [`parse`](Self::parse) but also consumes an optional TOP-LEVEL `NOT
    /// NULL` modifier (`string NOT NULL`), returning it as the second tuple element.
    /// A top-level `NOT NULL` makes the property REQUIRED (present + non-null),
    /// exactly like a `createRequiredConstraint` — the type-surface spelling of the
    /// same guarantee, mirroring the per-field `NOT NULL` inside a record.
    pub fn parse_with_not_null(s: &str) -> Option<(Self, bool)> {
        let mut p = TypeParser {
            s: s.as_bytes(),
            i: 0,
        };
        let t = p.parse_type()?;
        let not_null = if p.eat_kw("not") {
            if !p.eat_kw("null") {
                return None;
            }
            true
        } else {
            false
        };
        p.skip_ws();
        (p.i == p.s.len()).then_some((t, not_null))
    }

    /// Re-emit the canonical type name (round-trips through [`parse`](Self::parse));
    /// used by `dump_schema` to replay a declared constraint.
    fn to_name(&self) -> String {
        match self {
            Self::Scalar(t) => t.to_name().to_string(),
            Self::Record(fields) => {
                let inner: Vec<String> = fields
                    .iter()
                    .map(|(k, t, not_null)| {
                        let nn = if *not_null { " NOT NULL" } else { "" };
                        format!("{k}::{}{nn}", t.to_name())
                    })
                    .collect();
                format!("record{{{}}}", inner.join(","))
            }
            Self::AnyRecord => "any record".to_string(),
        }
    }
}

/// A tiny recursive-descent parser for a constraint type expression.
struct TypeParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl TypeParser<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    /// Read a bare identifier (a field name or a scalar-type keyword).
    fn ident(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.i;
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
        {
            self.i += 1;
        }
        (self.i > start).then(|| String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }
    fn eat(&mut self, c: u8) -> bool {
        self.skip_ws();
        if self.i < self.s.len() && self.s[self.i] == c {
            self.i += 1;
            true
        } else {
            false
        }
    }
    /// Consume the identifier `word` (case-insensitive) if it's next; otherwise
    /// leave the cursor untouched.
    fn eat_kw(&mut self, word: &str) -> bool {
        let save = self.i;
        if self.ident().is_some_and(|w| w.eq_ignore_ascii_case(word)) {
            true
        } else {
            self.i = save;
            false
        }
    }
    fn parse_type(&mut self) -> Option<TypeSpec> {
        let word = self.ident()?;
        // ISO `<record type> ::= [ANY] RECORD [<field types spec>]`. `ANY RECORD`
        // (and a bare `RECORD` with no `{…}`) is the OPEN record — any map.
        if word.eq_ignore_ascii_case("any") {
            return self.eat_kw("record").then_some(TypeSpec::AnyRecord);
        }
        if word.eq_ignore_ascii_case("record") {
            self.skip_ws();
            return if self.i < self.s.len() && self.s[self.i] == b'{' {
                self.parse_record() // closed: `record { … }`
            } else {
                Some(TypeSpec::AnyRecord) // bare `record` → open
            };
        }
        PropType::from_name(&word).map(TypeSpec::Scalar)
    }
    /// `{ field [::|:] type [, …] }` — the `record` keyword already consumed.
    fn parse_record(&mut self) -> Option<TypeSpec> {
        if !self.eat(b'{') {
            return None;
        }
        let mut fields: Vec<(Arc<str>, TypeSpec, bool)> = Vec::new();
        self.skip_ws();
        if !self.eat(b'}') {
            loop {
                let name = self.ident()?;
                // `::` or `:` between a field name and its type.
                if !self.eat(b':') {
                    return None;
                }
                self.eat(b':'); // optional second colon
                let ty = self.parse_type()?;
                // ISO `NOT NULL` after the field type → a required (present,
                // non-null) field; absent it, the field is nullable/optional.
                let not_null = if self.eat_kw("not") {
                    if !self.eat_kw("null") {
                        return None;
                    }
                    true
                } else {
                    false
                };
                let key: Arc<str> = name.into();
                match fields.binary_search_by(|(k, _, _)| k.as_ref().cmp(key.as_ref())) {
                    Ok(i) => fields[i] = (key, ty, not_null), // duplicate → last wins
                    Err(i) => fields.insert(i, (key, ty, not_null)),
                }
                if self.eat(b',') {
                    continue;
                }
                if self.eat(b'}') {
                    break;
                }
                return None;
            }
        }
        Some(TypeSpec::Record(fields))
    }
}

/// Does a stored value satisfy a declared [`TypeSpec`]? A top-level `Null` is
/// exempt (nullability of the property itself is the separate REQUIRED
/// constraint). A record is CLOSED on extras (no field outside the declared set)
/// and each field is optional by default — absent OR a null value both satisfy a
/// nullable field; a `NOT NULL` field must be present with a non-null value that
/// matches its type. Keys are canonical (sorted) on both sides.
fn value_matches(v: &Value, spec: &TypeSpec) -> bool {
    if matches!(v, Value::Null) {
        return true;
    }
    match spec {
        // A scalar constraint governs scalar values; a value with no scalar type
        // (a map) is exempt — the record path governs maps. Matches the scalar
        // write-check (`type_conflict_on_set`), which skips a non-scalar value.
        TypeSpec::Scalar(ty) => value_type(v).is_none_or(|got| got == *ty),
        TypeSpec::Record(fields) => {
            let Value::Map(pairs) = v else { return false };
            // No extra fields: every present key must be a declared field.
            if pairs.iter().any(|(vk, _)| {
                fields
                    .binary_search_by(|(fk, _, _)| fk.as_ref().cmp(vk))
                    .is_err()
            }) {
                return false;
            }
            // Each declared field: present → match its type (null only if the
            // field is nullable); absent → OK unless the field is `NOT NULL`.
            fields.iter().all(|(fk, ft, not_null)| {
                match pairs.binary_search_by(|(vk, _)| vk.as_ref().cmp(fk)) {
                    Ok(i) => {
                        let fv = &pairs[i].1;
                        if matches!(fv, Value::Null) {
                            !not_null // a null is OK only for a nullable field
                        } else {
                            value_matches(fv, ft)
                        }
                    }
                    Err(_) => !not_null, // absent is OK only for a nullable field
                }
            })
        }
        // The open record type — any map value, any shape (a non-map present value
        // is a violation, like a wrong-typed scalar).
        TypeSpec::AnyRecord => matches!(v, Value::Map(_)),
    }
}

/// A scalar field value for [`schema_op`] — the small set of JSON leaf shapes a
/// [`SchemaOp`](Graph::dump_schema) object carries.
enum Jv<'a> {
    /// A JSON string (escaped on write).
    S(&'a str),
    /// A JSON number.
    N(u32),
    /// A JSON number, or `null` (an unbounded cardinality `max`).
    NOpt(Option<u32>),
}

/// Build one `SchemaOp` JSON object: `{"op":<op>, <k>:<v>, …}` with `op` first,
/// then each field in the given order. String values are JSON-escaped via the
/// shared `jsonfmt` so the output is byte-identical to `JSON.stringify`.
fn schema_op(op: &str, fields: &[(&str, Jv)]) -> String {
    use std::fmt::Write as _;
    let mut s = String::from("{");
    crate::jsonfmt::push_json_str(&mut s, "op");
    s.push(':');
    crate::jsonfmt::push_json_str(&mut s, op);
    for (k, v) in fields {
        s.push(',');
        crate::jsonfmt::push_json_str(&mut s, k);
        s.push(':');
        match v {
            Jv::S(x) => crate::jsonfmt::push_json_str(&mut s, x),
            Jv::N(n) => {
                let _ = write!(s, "{n}");
            }
            Jv::NOpt(Some(n)) => {
                let _ = write!(s, "{n}");
            }
            Jv::NOpt(None) => s.push_str("null"),
        }
    }
    s.push('}');
    s
}

/// An immutable **CSR** (compressed-sparse-row) snapshot of the adjacency: one
/// flat `Adj` array per direction with an `n+1` offset array, so expanding vertex
/// `v` is a contiguous slice `adj[off[v]..off[v+1]]`. This is the read-optimized
/// *base*; the mutable `Vec<Vec<Adj>>` lists stay the write-path *delta*. Chasing
/// a per-vertex `Vec` header into its own heap allocation (a cache miss + no
/// prefetch across vertices) becomes a sequential walk of one array — the win that
/// matters for multi-hop traversal. Built lazily and cached (see [`Graph::csr`]),
/// dropped on any topology mutation so a rebuild reflects the change.
struct Csr {
    out_off: Vec<u32>,
    out_adj: Vec<Adj>,
    in_off: Vec<u32>,
    in_adj: Vec<Adj>,
}

/// Adjacency reads since the last invalidation before the CSR snapshot is rebuilt.
/// Low enough that a bulk scan gets locality almost immediately; high enough that a
/// write→read→write→read workload never pays the O(V+E) repack.
const CSR_WARM_READS: u32 = 64;

/// Flatten per-vertex adjacency `Vec`s into `(offsets, concatenated slots)`.
fn csr_pack(adjs: &[Vec<Adj>]) -> (Vec<u32>, Vec<Adj>) {
    let total: usize = adjs.iter().map(Vec::len).sum();
    let mut off = Vec::with_capacity(adjs.len() + 1);
    let mut flat = Vec::with_capacity(total);
    off.push(0);
    for a in adjs {
        flat.extend_from_slice(a);
        off.push(flat.len() as u32);
    }
    (off, flat)
}

impl Csr {
    fn build(out: &[Vec<Adj>], in_: &[Vec<Adj>]) -> Self {
        let (out_off, out_adj) = csr_pack(out);
        let (in_off, in_adj) = csr_pack(in_);
        Self {
            out_off,
            out_adj,
            in_off,
            in_adj,
        }
    }
    #[inline]
    fn out(&self, v: u32) -> &[Adj] {
        Self::slice(&self.out_off, &self.out_adj, v)
    }
    #[inline]
    fn in_(&self, v: u32) -> &[Adj] {
        Self::slice(&self.in_off, &self.in_adj, v)
    }
    #[inline]
    fn slice<'a>(off: &[u32], adj: &'a [Adj], v: u32) -> &'a [Adj] {
        let v = v as usize;
        match (off.get(v), off.get(v + 1)) {
            (Some(&lo), Some(&hi)) => &adj[lo as usize..hi as usize],
            _ => &[],
        }
    }
}

/// The scalar type of a stored value, or `None` for null / a `Map` (both
/// type-exempt — a null has no type, and a record has no scalar `PropType`).
fn value_type(v: &Value) -> Option<PropType> {
    match v {
        Value::Null | Value::Map(_) => None,
        Value::Bool(_) => Some(PropType::Bool),
        Value::Num(_) => Some(PropType::Num),
        Value::Str(_) => Some(PropType::Str),
        Value::Temporal(Temporal::Date(_)) => Some(PropType::Date),
        Value::Temporal(Temporal::Time(_)) => Some(PropType::Time),
        Value::Temporal(Temporal::DateTime(_)) => Some(PropType::DateTime),
        Value::Temporal(Temporal::ZonedTime(_)) => Some(PropType::ZonedTime),
        Value::Temporal(Temporal::ZonedDateTime(_)) => Some(PropType::ZonedDateTime),
        Value::Temporal(Temporal::Duration(_)) => Some(PropType::Duration),
        Value::List(_) => Some(PropType::List),
    }
}

/// The mutable columnar graph.
pub struct Graph {
    /// Vertex slots (including tombstoned). Index space for queries is `0..n`.
    pub n: usize,
    live_n: usize,
    v_live: Vec<bool>,
    /// external string id <-> dense index
    pub vid: Dict,
    pub labels: Dict,
    pub etype: Dict,
    /// graph-wide string interner backing every `Column::Str` (vertex and edge)
    pub strs: Dict,
    /// per-vertex label ids
    vlabels: Vec<Vec<u32>>,
    /// inverted index: label id -> live vertices (query seeds)
    by_label: HashMap<u32, Vec<u32>>,
    /// vertex property columns (indexed by vertex)
    pub props: Properties,
    /// edge property columns (indexed by edge) — same store type as `props`
    pub edge_props: Properties,
    /// edges (parallel arrays); `e_live` tombstones deletions
    pub e_src: Vec<u32>,
    pub e_dst: Vec<u32>,
    pub e_type: Vec<u32>,
    /// inverted index: edge type id -> live edges. The edge analogue of
    /// `by_label`; seeds `()-[:T]->()` patterns from the type directly instead
    /// of scanning every edge. Always on (same as `by_label`), maintained by the
    /// edge mutations.
    by_etype: HashMap<u32, Vec<u32>>,
    /// Optional external edge ids — a **lazy** overlay so edges can round-trip a
    /// user-assigned string id. The dense edge index is the canonical identity;
    /// these maps are empty unless ids are supplied (codecs / `set_edge_id`), so
    /// the common in-memory path pays nothing. `eid_fwd`: edge index -> id (for
    /// encode); `eid_rev`: id -> edge index (for lookup / addressability).
    eid_fwd: HashMap<u32, Arc<str>>,
    eid_rev: HashMap<Arc<str>, u32>,
    /// Reactive change tracking (for `useSyncExternalStore`-style snapshots):
    /// `version` is a monotonic counter bumped on every mutation — an O(1)
    /// "did anything change?" signal. `epochs` is per-token (label / edge-type /
    /// property-key name) for *finer* invalidation: topology changes bump the
    /// element's labels/types and keys; a property write bumps only that key. So
    /// `epoch("Person")` moves iff Person membership changed, `epoch("age")` iff
    /// some age value changed. Keyed by name, so it's bounded by schema size.
    version: u64,
    epochs: HashMap<String, u64>,
    e_live: Vec<bool>,
    live_e: usize,
    /// per-vertex out / in adjacency — the mutable write-path *delta*. Reads go
    /// through the cached [`Csr`] snapshot (`csr`) built from these.
    out: Vec<Vec<Adj>>,
    in_: Vec<Vec<Adj>>,
    /// Lazily-built, cached CSR snapshot of `out`/`in_` for cache-friendly reads.
    /// `get_or_init` fills it on first expansion; every topology mutation calls
    /// [`Graph::invalidate_csr`] to drop it so the next read rebuilds. Kept in a
    /// `OnceLock` (not a plain `Option`) so a shared read-only `&Graph` can build
    /// it once without `&mut`, and `Graph` stays `Send`/`Sync`.
    csr: std::sync::OnceLock<Csr>,
    /// Adjacency reads since the snapshot was last dropped, for the rebuild
    /// heuristic in [`Graph::adj`]. `AtomicU32` (not `Cell`) so `&Graph` stays
    /// `Send`/`Sync` alongside the `OnceLock`.
    csr_reads: std::sync::atomic::AtomicU32,
    /// counter for synthesized ids of vertices created at runtime
    synth: u64,
    /// Opt-in secondary indexes over vertex / edge property values: key name →
    /// ordered map (value → live element ids). A `BTreeMap` answers both equality
    /// (`get`) and range (`range`) from one structure. Keyed by name (not key-id)
    /// so an index can be declared and maintained even before any element carries
    /// the key. Built via [`Graph::create_vertex_index`]; kept current by the
    /// mutation methods. Absent key ⇒ no index (full scan).
    vidx: PropIndex,
    eidx: PropIndex,
    /// UNIQUE constraints over vertex properties: label name → the sorted
    /// property keys that must be unique among live vertices carrying that label.
    /// Each constrained key is index-backed (declaring the constraint creates the
    /// vertex index), so enforcement and `_MERGE` key lookups seek rather than
    /// scan. Null/list values are exempt (SQL semantics — NULLs are distinct),
    /// which also matches what the value index can hold. See
    /// `docs/design/gql-extensions.md` §3.
    v_unique: HashMap<String, Vec<String>>,
    /// REQUIRED constraints: `label` → the property keys that must be present and
    /// non-null on every live vertex carrying that label (R-CONSTRAINTS). Unlike
    /// `v_unique` these need no backing index — enforcement is a presence check.
    v_required: HashMap<String, Vec<String>>,
    /// TYPE constraints: `label` → (`key` → the scalar type its present, non-null
    /// values must be). Null/absent are exempt (R-CONSTRAINTS).
    v_type: HashMap<String, HashMap<String, PropType>>,
    /// RECORD-typed constraints: `label` → (`key` → a closed `RECORD {…}` shape
    /// its present, non-null values must match). The record analogue of `v_type`,
    /// kept separate so the scalar path stays a `Copy` `PropType` map. A declared
    /// record shape is the contract that later lets the store de-box the column.
    v_record: HashMap<String, HashMap<String, TypeSpec>>,
    /// `label` → the set of scalar-typed keys declared `NOT NULL` (`string NOT
    /// NULL`). A parallel, additive set beside `v_type` (the scalar map stays a
    /// `Copy` `PropType`): a `NOT NULL` key is REQUIRED (present + non-null), so
    /// enforcement folds this into the required checks. Kept distinct from
    /// `v_required` so dropping the type constraint removes only ITS not-null,
    /// leaving any independently-declared required constraint intact.
    v_type_not_null: HashMap<String, std::collections::HashSet<String>>,
    /// UNIQUE constraints over **edge** properties: edge-type name → the sorted
    /// property keys that must be unique among live edges of that type. The edge
    /// analogue of `v_unique`, backed by the edge property index (`eidx`).
    e_unique: HashMap<String, Vec<String>>,
    /// REQUIRED constraints over edges: edge-type → the keys that must be present
    /// and non-null on every live edge of that type. The edge analogue of `v_required`.
    e_required: HashMap<String, Vec<String>>,
    /// TYPE constraints over edges: edge-type → (`key` → scalar type). The edge
    /// analogue of `v_type` (named `e_type_constraints` because `e_type: Vec<u32>`
    /// already holds the per-edge type ids).
    e_type_constraints: HashMap<String, HashMap<String, PropType>>,
    /// RECORD-typed constraints over edges — the edge analogue of `v_record`.
    e_record: HashMap<String, HashMap<String, TypeSpec>>,
    /// Edge-type → scalar keys declared `NOT NULL`. The edge analogue of
    /// `v_type_not_null`.
    e_type_not_null: HashMap<String, std::collections::HashSet<String>>,
    /// CARDINALITY constraints: bound the DEGREE of every vertex carrying `label`
    /// over `etype` in `direction` (0 = out / the vertex is the edge source, 1 =
    /// in / the target) to `min..=max` (`max: None` unbounded). A small flat list
    /// (schema-sized), searched linearly; keyed by `(label, etype, direction)` for
    /// declare-replace and drop. Max is checked at commit against touched
    /// endpoints; min is commit-time only (unsatisfiable by a single write). A
    /// self-loop counts once for out and once for in. See `docs/design/r-tx.md`.
    v_cardinality: Vec<CardinalityRule>,
    /// VALIDATOR constraints: a custom GQL boolean predicate per label (a vertex
    /// label OR an edge type — one string namespace). Every element carrying the
    /// label must satisfy the predicate at the mutation boundary; SQL-`CHECK`
    /// semantics — rejected only on a *definite* `false`, a null/unknown result
    /// passes. Keyed by label; a label may carry several. The predicate is parsed
    /// and lowered once at declare time (into a `CPredicate`) and evaluated in the
    /// GQL evaluator against each touched element at the commit boundary and in the
    /// declare-time scan. Byte-identical with the TS `createValidator`.
    v_validators: HashMap<String, Vec<ValidatorRule>>,
    /// Graph-level INVARIANTS (cross-write assertions): a whole-graph GQL query
    /// that must hold after every transaction that wrote something. Unlike a
    /// per-element validator, an invariant is evaluated ONCE per commit against
    /// the fully-staged graph — it is VIOLATED iff any cell in its result set is
    /// boolean `false` (everything else — `true`/`null`/non-boolean/empty — holds).
    /// Each entry stores the query source (for messaging/introspection) and the
    /// query parsed+lowered once at declare time. Byte-identical with the TS
    /// `createInvariant`. Keyed insertion order is irrelevant; `invariants()`
    /// sorts by name.
    v_invariants: Vec<InvariantRule>,
    /// Transaction state (R-TX). `tx_depth > 0` means an open transaction: writes
    /// still apply eagerly to the live store (read-your-writes with no overlay),
    /// but each mutation records an inverse op in `tx_undo`, the built-in
    /// constraint checks defer to commit (the touched vertex ids collect in
    /// `tx_touched`), and a rollback replays the undo log newest-first. Nesting is
    /// flat (a depth counter): the outermost frame owns commit/rollback, matching
    /// the TS core. `applying_undo` is true only while a rollback replays inverse
    /// ops, which must neither re-record undo nor re-note touched vertices. The
    /// undo `Vec` allocates lazily (empty until the first in-tx mutation), so an
    /// auto-commit frame around a read-only statement costs nothing.
    tx_depth: usize,
    tx_undo: Vec<Undo>,
    tx_touched: Vec<u32>,
    /// Edge analogue of `tx_touched`: edge indices whose built-in edge constraints
    /// must be re-checked at commit (R-TX deferral for edge writes).
    tx_touched_edges: Vec<u32>,
    /// The vertices touched by the most recent committed write — a snapshot of
    /// `tx_touched` taken at commit (before it's cleared), so a caller can derive
    /// that write's value-scope for CDC routing (`last_write_scope`). Content-derived
    /// scope extraction rides the touched set the commit already collects.
    last_touched: Vec<u32>,
    applying_undo: bool,
    /// Anti-resource-abuse ceiling on GQL operator-chain length, passed to the
    /// parser on each query (see `gql::parser::DEFAULT_MAX_CHAIN`). Defaults to
    /// 10_000; set at graph creation via the native `maxOperatorChain` option.
    max_operator_chain: usize,
    /// Access mode of the active explicit transaction opened by ISO GQL
    /// `START TRANSACTION READ ONLY` (see the gql eval layer). Set true by that
    /// statement, cleared on commit/rollback. Only the GQL statement executor reads
    /// it — the core mutators are access-mode agnostic; a read-only write is
    /// rejected at the statement boundary before any mutation applies.
    tx_read_only: bool,
}

/// A deep, independent copy of the graph — the fast substrate for a fork/branch
/// (`graph.copy()` over the FFI), an O(V+E) clone of the columnar store rather than
/// a serialize→parse NDJSON round-trip that also re-validates and re-indexes.
///
/// Every data field is cloned faithfully, so the copy is indistinguishable from
/// the original — same element ids (a base-vs-copy diff by id is exact), same
/// indexes, constraints, and any open transaction frame (the undo log clones, so
/// the copy can roll back independently). Only the lazy CSR read-cache is reset:
/// it is pure derived state rebuilt on first traversal, and cloning a warm cache
/// would copy work the write path would discard. A struct literal (not `..`) so
/// the compiler rejects the clone if a field is ever added and not handled here.
impl Clone for Graph {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            live_n: self.live_n,
            v_live: self.v_live.clone(),
            vid: self.vid.clone(),
            labels: self.labels.clone(),
            etype: self.etype.clone(),
            strs: self.strs.clone(),
            vlabels: self.vlabels.clone(),
            by_label: self.by_label.clone(),
            props: self.props.clone(),
            edge_props: self.edge_props.clone(),
            e_src: self.e_src.clone(),
            e_dst: self.e_dst.clone(),
            e_type: self.e_type.clone(),
            by_etype: self.by_etype.clone(),
            eid_fwd: self.eid_fwd.clone(),
            eid_rev: self.eid_rev.clone(),
            version: self.version,
            epochs: self.epochs.clone(),
            e_live: self.e_live.clone(),
            live_e: self.live_e,
            out: self.out.clone(),
            in_: self.in_.clone(),
            // Derived read-cache — reset, rebuilt lazily on first traversal.
            csr: std::sync::OnceLock::new(),
            csr_reads: std::sync::atomic::AtomicU32::new(0),
            synth: self.synth,
            vidx: self.vidx.clone(),
            eidx: self.eidx.clone(),
            v_unique: self.v_unique.clone(),
            v_required: self.v_required.clone(),
            v_type: self.v_type.clone(),
            v_record: self.v_record.clone(),
            v_type_not_null: self.v_type_not_null.clone(),
            e_unique: self.e_unique.clone(),
            e_required: self.e_required.clone(),
            e_type_constraints: self.e_type_constraints.clone(),
            e_record: self.e_record.clone(),
            e_type_not_null: self.e_type_not_null.clone(),
            v_cardinality: self.v_cardinality.clone(),
            v_validators: self.v_validators.clone(),
            v_invariants: self.v_invariants.clone(),
            tx_depth: self.tx_depth,
            tx_undo: self.tx_undo.clone(),
            tx_touched: self.tx_touched.clone(),
            tx_touched_edges: self.tx_touched_edges.clone(),
            last_touched: self.last_touched.clone(),
            applying_undo: self.applying_undo,
            max_operator_chain: self.max_operator_chain,
            tx_read_only: self.tx_read_only,
        }
    }
}

/// One inverse op recorded by a mutation while a transaction frame is open, to be
/// replayed (newest-first) on rollback. The tombstone-based delete model makes
/// these cheap: undo of an insert = tombstone the slot; undo of a delete =
/// un-tombstone it (the columns are never cleared on delete, so property values
/// survive in place); undo of a property write = restore the prior columnar value.
#[derive(Clone)]
enum Undo {
    /// An inserted vertex — undo by tombstoning it (`remove_vertex`, detach).
    InsertVertex(u32),
    /// An inserted edge — undo by tombstoning it (`remove_edge`).
    InsertEdge(u32),
    /// A vertex property write — restore the prior value (`Some`) or absence (`None`).
    VProp(u32, String, Option<Value>),
    /// An edge property write — restore the prior value (`Some`) or absence (`None`).
    EProp(u32, String, Option<Value>),
    /// A label newly added to a vertex — undo by removing it.
    VLabelAdd(u32, String),
    /// A label removed from a vertex — undo by re-adding it.
    VLabelRemove(u32, String),
    /// An edge type replaced (edges carry a single type) — restore the prior type name.
    EType(u32, String),
    /// A deleted vertex — undo by un-tombstoning the slot and restoring its labels
    /// (its incident edges are restored by their own `DeleteEdge` inverses, which
    /// were recorded during the delete cascade and so replay after this one).
    DeleteVertex { vi: u32, labels: Vec<u32> },
    /// A deleted edge — undo by un-tombstoning it and restoring any external-id overlay.
    DeleteEdge { ei: u32, eid: Option<Arc<str>> },
}

/// A declared CARDINALITY constraint: every live vertex carrying `label` must
/// have `min <= degree <= max` over `etype` in `direction` (0 = out, 1 = in).
/// `max: None` is unbounded. The Rust analogue of the TS `CardinalityConstraint`.
#[derive(Clone, Debug)]
struct CardinalityRule {
    label: String,
    etype: String,
    direction: u8,
    min: u32,
    max: Option<u32>,
}

/// A registered VALIDATOR: its bind variable name, its GQL predicate source (for
/// messaging / introspection), and the predicate parsed+lowered once at declare
/// time. The Rust analogue of the TS `{ varName, src, fn }` validator entry.
#[derive(Clone)]
struct ValidatorRule {
    var: String,
    src: String,
    pred: crate::gql::plan::CPredicate,
}

/// A registered graph-level INVARIANT: its name, its GQL query source (for
/// messaging / introspection), and the query parsed+lowered once at declare time
/// into a reusable [`crate::gql::Prepared`] plan. Evaluated against the fully-
/// staged graph at commit; VIOLATED iff any result cell is boolean `false`. The
/// Rust analogue of the TS `{ src, fn }` invariant entry.
#[derive(Clone)]
struct InvariantRule {
    name: String,
    src: String,
    plan: std::sync::Arc<crate::gql::Prepared>,
}

/// Which deferred constraint check failed at commit. All surface to the caller as
/// `ConstraintViolation`, but are kept distinct for messaging / FFI codes.
pub enum TxCommitError {
    /// `commit_tx` was called with no open transaction.
    NoTx,
    /// A required-property constraint is unsatisfied on a touched vertex.
    Required,
    /// A type constraint is violated on a touched vertex.
    Type,
    /// A unique constraint is violated on a touched vertex.
    Unique,
    /// A cardinality (degree-bound) constraint is violated on a touched vertex.
    Cardinality,
    /// A custom VALIDATOR predicate failed on a touched vertex/edge, or the
    /// predicate itself faulted while evaluating (e.g. an unknown function). The
    /// carried [`CodeError`] is surfaced verbatim — a `ConstraintViolation` for a
    /// definite-`false` predicate, or the evaluation fault's own code.
    Validator(CodeError),
    /// A graph-level INVARIANT query returned a `false` cell (a cross-write
    /// assertion failed), or the query itself faulted while evaluating. The
    /// carried [`CodeError`] is surfaced verbatim — a `ConstraintViolation` for a
    /// definite-`false` result cell, or the evaluation fault's own code.
    Invariant(CodeError),
}

/// A set of property indexes (key name → ordered value buckets).
type PropIndex = HashMap<String, std::collections::BTreeMap<IdxKey, Vec<u32>>>;

/// An index key/path split into its root property and the descent into a stored
/// map: `"meta.city"` → `("meta", ["city"])`; a plain `"name"` → `("name", [])`.
/// Property/field names with a literal `.` aren't reachable as a nested path (a
/// rare edge case), consistent with the `.`-as-access convention.
fn split_index_path(path: &str) -> (&str, Vec<&str>) {
    let mut segs = path.split('.');
    let root = segs.next().unwrap_or("");
    (root, segs.collect())
}

/// Follow a descent of field names into a value, returning the leaf — or `None`
/// if a segment isn't present or the value isn't a map there. Maps are canonical
/// (sorted keys), so each hop is a binary search. An empty descent is the value
/// itself (a plain top-level index).
pub(crate) fn value_at_descent<'a>(v: &'a Value, descent: &[&str]) -> Option<&'a Value> {
    let mut cur = v;
    for seg in descent {
        let Value::Map(pairs) = cur else { return None };
        match pairs.binary_search_by(|(k, _)| k.as_ref().cmp(*seg)) {
            Ok(i) => cur = &pairs[i].1,
            Err(_) => return None,
        }
    }
    Some(cur)
}

/// Element `id`'s property `key` was set/removed with `value`; update every index
/// whose path is rooted at `key`. A top-level index indexes `value` itself; a
/// dotted-path index (`meta.city`) indexes the scalar leaf at that descent. The
/// leaf isn't indexable (null / list / map / absent) → that index is skipped.
fn idx_apply(map: &mut PropIndex, key: &str, id: u32, value: &Value, add: bool) {
    for (path, bt) in map.iter_mut() {
        let (root, descent) = split_index_path(path);
        if root != key {
            continue;
        }
        let Some(leaf) = value_at_descent(value, &descent) else {
            continue;
        };
        let Some(k) = IdxKey::from_value(leaf) else {
            continue;
        };
        if add {
            bt.entry(k).or_default().push(id);
        } else if let Some(bucket) = bt.get_mut(&k) {
            bucket.retain(|&x| x != id);
            if bucket.is_empty() {
                bt.remove(&k);
            }
        }
    }
}

/// Backfill an index for `path` (a property name or a dotted `root.field…` path
/// into a stored map) over a property store (vertex or edge).
fn build_prop_index(
    store: &Properties,
    live: &[bool],
    strs: &Dict,
    path: &str,
    n: usize,
) -> std::collections::BTreeMap<IdxKey, Vec<u32>> {
    let mut map: std::collections::BTreeMap<IdxKey, Vec<u32>> = std::collections::BTreeMap::new();
    let (root, descent) = split_index_path(path);
    let Some(kid) = store.keys.get(root) else {
        return map;
    };
    for id in 0..n as u32 {
        if !live.get(id as usize).copied().unwrap_or(false) {
            continue;
        }
        let v = store.value_id(id as usize, kid, strs);
        if let Some(k) = value_at_descent(&v, &descent).and_then(IdxKey::from_value) {
            map.entry(k).or_default().push(id);
        }
    }
    map
}

/// Union the buckets of one key's ordered index that fall within `bound`. Bounds
/// carry a type (e.g. `Num(30)`), so the scan stays within that type block —
/// `{gt: 30}` never bleeds into string values.
fn range_seek(
    map: &std::collections::BTreeMap<IdxKey, Vec<u32>>,
    bound: &RangeBound,
) -> Option<Vec<u32>> {
    use std::ops::Bound;
    let lo = match (&bound.gte, &bound.gt) {
        (Some(k), _) => Bound::Included(k.clone()),
        (None, Some(k)) => Bound::Excluded(k.clone()),
        (None, None) => Bound::Unbounded,
    };
    let rank = [&bound.gt, &bound.gte, &bound.lt, &bound.lte]
        .into_iter()
        .flatten()
        .next()
        .map(IdxKey::rank);
    let mut out = Vec::new();
    for (k, ids) in map.range((lo, Bound::Unbounded)) {
        if let Some(r) = rank {
            if k.rank() < r {
                continue;
            }
            if k.rank() > r {
                break;
            }
        }
        if bound.lt.as_ref().is_some_and(|b| k >= b) || bound.lte.as_ref().is_some_and(|b| k > b) {
            break;
        }
        out.extend_from_slice(ids);
    }
    Some(out)
}

/// A totally-ordered key for the property index: type rank (Bool < Num < Str)
/// then value, so a numeric range seek never bleeds into string values.
#[derive(Clone, Debug)]
pub enum IdxKey {
    Bool(bool),
    Num(f64),
    Str(Arc<str>),
    /// Temporal as `(kind_rank, monotonic_key)` — see [`Temporal::index_key`]. The
    /// kind rank keeps Date/Time/DateTime/Zoned* in disjoint ranges (their `i128`
    /// keys aren't cross-kind comparable); Duration is excluded (no monotonic key).
    Temporal(u8, i128),
}

impl IdxKey {
    fn rank(&self) -> u8 {
        match self {
            Self::Bool(_) => 0,
            Self::Num(_) => 1,
            Self::Str(_) => 2,
            Self::Temporal(..) => 3,
        }
    }
    /// Build from a core [`Value`] (absent / list / duration → not indexable).
    fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::Bool(b) => Some(Self::Bool(*b)),
            Value::Num(n) => Some(Self::Num(*n)),
            Value::Str(s) => Some(Self::Str(s.clone())),
            Value::Temporal(t) => t.index_key().map(|(k, key)| Self::Temporal(k, key)),
            _ => None,
        }
    }
}

impl PartialEq for IdxKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for IdxKey {}
impl PartialOrd for IdxKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for IdxKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Num(a), Self::Num(b)) => a.total_cmp(b),
            (Self::Str(a), Self::Str(b)) => a.as_ref().cmp(b.as_ref()),
            // Within temporals: kind rank first, then the monotonic key.
            (Self::Temporal(ka, va), Self::Temporal(kb, vb)) => ka.cmp(kb).then(va.cmp(vb)),
            _ => self.rank().cmp(&other.rank()),
        }
    }
}

/// Inclusive/exclusive range bounds for a property-index range seek.
#[derive(Clone, Debug, Default)]
pub struct RangeBound {
    pub gt: Option<IdxKey>,
    pub gte: Option<IdxKey>,
    pub lt: Option<IdxKey>,
    pub lte: Option<IdxKey>,
}

impl Graph {
    // --- reads -------------------------------------------------------------

    pub fn vertex_count(&self) -> usize {
        self.live_n
    }
    pub fn edge_count(&self) -> usize {
        self.live_e
    }
    /// True if any vertex or edge property holds a map/record value (at any
    /// depth). The flat codecs (pg-text / csv) reject an export containing one,
    /// since a nested record has no faithful line/column representation — use a
    /// structured format (ndjson / graphson / pg-json) instead.
    pub fn has_map_property(&self) -> bool {
        self.props.has_map_value() || self.edge_props.has_map_value()
    }
    /// Diagnostic: `(packed_heap_bytes, mixed_equiv_bytes)` for vertex property
    /// `key`'s column — the actual heap it uses vs what the same column would cost
    /// boxed in a `Mixed` (`len × size_of::<Option<Value>>()`). `None` if the key
    /// is unknown. Used to measure the de-boxing memory win per type.
    pub fn vertex_prop_bytes(&self, key: &str) -> Option<(usize, usize)> {
        let kid = self.props.keys.get(key)?;
        let col = self.props.cols.get(kid as usize)?;
        let mixed = col.element_len() * std::mem::size_of::<Option<Value>>();
        Some((col.heap_bytes(), mixed))
    }
    /// Total edge slots (including tombstoned) — for encoders that scan them.
    pub fn edge_slots(&self) -> usize {
        self.e_src.len()
    }
    pub fn is_vertex_live(&self, v: u32) -> bool {
        self.v_live.get(v as usize).copied().unwrap_or(false)
    }

    /// The distinct values of property `key` across the vertices touched by the most
    /// recent committed write — the content-derived **value-scope** of that write,
    /// for CDC interest routing (e.g. `last_write_scope("room")` → `["42"]` after a
    /// write into room 42). Rides the touched set the commit already collects, so
    /// it's a handful of columnar reads (see `examples/cdc_extract_bench.rs`). Tombstoned
    /// vertices and elements without the key are skipped. Values render like a scope
    /// token: numbers without a trailing `.0`, strings verbatim, booleans as
    /// `true`/`false`.
    pub fn last_write_scope(&self, key: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for &vi in &self.last_touched {
            if !self.is_vertex_live(vi) {
                continue;
            }
            let rendered = match self.props.value(vi as usize, key, &self.strs) {
                Value::Null => continue,
                Value::Str(s) => s.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Num(n) => format!("{n}"),
                // A structured value can't be a scope token.
                Value::List(_) | Value::Map(_) | Value::Temporal(_) => continue,
            };
            if !out.contains(&rendered) {
                out.push(rendered);
            }
        }
        out
    }
    pub fn is_edge_live(&self, e: u32) -> bool {
        self.e_live.get(e as usize).copied().unwrap_or(false)
    }
    /// Live vertex indices (skips tombstones) — the full candidate seed set.
    pub fn vertex_indices(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.n as u32).filter(move |&v| self.v_live[v as usize])
    }

    /// The first live edge of type `etype` from `from` to `to`, if any — the
    /// structural key `_MERGE`'s edge form upserts on (ensures at most one such
    /// edge). First-by-adjacency-order, matching the TS engine.
    pub fn find_edge(&self, from: u32, to: u32, etype: &str) -> Option<u32> {
        let tid = self.etype.get(etype)?;
        self.out.get(from as usize)?.iter().find_map(|a| {
            (a.nbr == to && a.etype == tid && self.e_live[a.eidx as usize]).then_some(a.eidx)
        })
    }

    // --- property indexes (opt-in secondary indexes over property values) --

    /// Declare (and backfill) a secondary index over a **vertex** property. An
    /// eq/range filter on this key can then seed from the index instead of a full
    /// scan. Kept current by the mutation methods. Idempotent.
    pub fn create_vertex_index(&mut self, key: &str) {
        let map = build_prop_index(&self.props, &self.v_live, &self.strs, key, self.n);
        self.vidx.insert(key.to_string(), map);
    }
    /// Declare (and backfill) a secondary index over an **edge** property.
    pub fn create_edge_index(&mut self, key: &str) {
        let map = build_prop_index(
            &self.edge_props,
            &self.e_live,
            &self.strs,
            key,
            self.e_src.len(),
        );
        self.eidx.insert(key.to_string(), map);
    }
    /// Drop a vertex index. Rejected (`InvalidGraphOp`) if the key backs a unique
    /// constraint — dropping it would downgrade enforcement to a scan (or, on the
    /// TS twin, silently lose it); drop the constraint first. Idempotent otherwise.
    pub fn drop_vertex_index(&mut self, key: &str) -> CodeResult<()> {
        if self
            .v_unique
            .values()
            .any(|keys| keys.iter().any(|k| k == key))
        {
            return Err(CodeError::new(
                ErrorCode::InvalidGraphOp,
                "cannot drop the vertex index; it backs a unique constraint — drop the constraint first",
            ));
        }
        self.vidx.remove(key);
        Ok(())
    }
    /// Drop an edge index. Edge analogue of [`drop_vertex_index`](Self::drop_vertex_index):
    /// rejected if the key backs an edge unique constraint.
    pub fn drop_edge_index(&mut self, key: &str) -> CodeResult<()> {
        if self
            .e_unique
            .values()
            .any(|keys| keys.iter().any(|k| k == key))
        {
            return Err(CodeError::new(
                ErrorCode::InvalidGraphOp,
                "cannot drop the edge index; it backs a unique constraint — drop the constraint first",
            ));
        }
        self.eidx.remove(key);
        Ok(())
    }

    pub fn vertex_indexed(&self, key: &str) -> bool {
        self.vidx.contains_key(key)
    }
    pub fn edge_indexed(&self, key: &str) -> bool {
        self.eidx.contains_key(key)
    }

    /// Does any vertex index depend on property `key` — either indexing it
    /// directly or via a dotted path rooted at it (`key.field…`)? A write to
    /// `key` must refresh all such indexes, so the maintenance gates use this
    /// rather than an exact-name `contains_key`.
    fn any_vidx_rooted_at(&self, key: &str) -> bool {
        self.vidx.keys().any(|p| split_index_path(p).0 == key)
    }
    fn any_eidx_rooted_at(&self, key: &str) -> bool {
        self.eidx.keys().any(|p| split_index_path(p).0 == key)
    }

    /// The vertex property keys that currently carry a secondary index, sorted
    /// for a deterministic listing.
    pub fn vertex_indexes(&self) -> Vec<String> {
        let mut ks: Vec<String> = self.vidx.keys().cloned().collect();
        ks.sort();
        ks
    }
    /// The edge property keys that currently carry a secondary index, sorted.
    pub fn edge_indexes(&self) -> Vec<String> {
        let mut ks: Vec<String> = self.eidx.keys().cloned().collect();
        ks.sort();
        ks
    }

    // --- unique constraints (declared over `(label, property key)`) ---------
    // At most one live vertex carrying `label` may hold a given non-null value
    // for `key`. Backed by the vertex property index (so lookups seek). This is
    // the Pattern-B primitive `_MERGE` keys on; see `docs/design/gql-extensions.md`.

    /// Declare a UNIQUE constraint on `(label, key)`. Creates the backing vertex
    /// index if absent, then registers the constraint. Idempotent. Fails with
    /// [`ErrorCode::ConstraintViolation`] if the *current* data already violates
    /// it — an already-broken constraint is meaningless (SQL rejects the unique
    /// index build the same way).
    pub fn create_unique_constraint(&mut self, label: &str, key: &str) -> CodeResult<()> {
        if !self.vertex_indexed(key) {
            self.create_vertex_index(key);
        }
        if self.first_label_prop_duplicate(label, key).is_some() {
            return Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "existing data already violates the unique constraint being declared",
            ));
        }
        let keys = self.v_unique.entry(label.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Drop a unique constraint. The backing index is left in place (drop it via
    /// [`Graph::drop_vertex_index`] if unwanted). Idempotent.
    pub fn drop_unique_constraint(&mut self, label: &str, key: &str) {
        if let Some(keys) = self.v_unique.get_mut(label) {
            keys.retain(|k| k != key);
            if keys.is_empty() {
                self.v_unique.remove(label);
            }
        }
    }

    /// Property keys under a unique constraint for `label` (sorted; empty if
    /// none). `_MERGE` intersects this with the pattern to infer the conflict key.
    pub fn unique_keys(&self, label: &str) -> &[String] {
        self.v_unique.get(label).map_or(&[], Vec::as_slice)
    }

    /// True iff `(label, key)` carries a unique constraint.
    pub fn has_unique_constraint(&self, label: &str, key: &str) -> bool {
        self.v_unique
            .get(label)
            .is_some_and(|ks| ks.iter().any(|k| k == key))
    }

    /// Every declared unique constraint as sorted `(label, key)` pairs — a
    /// deterministic listing for host introspection.
    pub fn unique_constraints(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .v_unique
            .iter()
            .flat_map(|(l, ks)| ks.iter().map(move |k| (l.clone(), k.clone())))
            .collect();
        out.sort();
        out
    }

    /// The single live vertex carrying `label` whose `key == value`, if any (≤1
    /// under the constraint). The `_MERGE` create-vs-update decision. A non-null
    /// scalar `value` seeks the index; null/list yield `None` (exempt).
    pub fn unique_lookup(&self, label: &str, key: &str, value: &Value) -> Option<u32> {
        self.vertices_with_label_value(label, key, value)
            .into_iter()
            .next()
    }

    /// If adding a vertex with `labels` + `props` would break a unique constraint,
    /// the offending `(label, key, existing vertex)`. Drives INSERT enforcement;
    /// `exclude` skips one vertex (itself, for a re-check). Only constrained keys
    /// present in `props` are checked; null/list values are exempt.
    pub fn unique_conflict(
        &self,
        labels: &[String],
        props: &[(String, Value)],
        exclude: Option<u32>,
    ) -> Option<(String, String, u32)> {
        if self.v_unique.is_empty() {
            return None;
        }
        for label in labels {
            for key in self.unique_keys(label) {
                let Some((_, value)) = props.iter().find(|(k, _)| k == key) else {
                    continue;
                };
                let hit = self
                    .vertices_with_label_value(label, key, value)
                    .into_iter()
                    .find(|&v| Some(v) != exclude);
                if let Some(existing) = hit {
                    return Some((label.clone(), key.clone(), existing));
                }
            }
        }
        None
    }

    /// If setting `vi.key = value` would break a unique constraint on one of
    /// `vi`'s labels, the offending `(label, existing vertex)`.
    pub fn unique_conflict_on_set(
        &self,
        vi: u32,
        key: &str,
        value: &Value,
    ) -> Option<(String, u32)> {
        for (label, keys) in &self.v_unique {
            if !keys.iter().any(|k| k == key) {
                continue;
            }
            let Some(lid) = self.labels.get(label) else {
                continue;
            };
            if !self.vlabels[vi as usize].contains(&lid) {
                continue;
            }
            if let Some(existing) = self
                .vertices_with_label_value(label, key, value)
                .into_iter()
                .find(|&v| v != vi)
            {
                return Some((label.clone(), existing));
            }
        }
        None
    }

    // --- required constraints (R-CONSTRAINTS) --------------------------------
    // Every live vertex carrying `label` must hold a present, non-null value for
    // each required `key`. Enforced in the write path (INSERT/SET/REMOVE) like
    // `unique`; declarative (no closures), so it is byte-identical to the TS core.
    // No backing index is needed — enforcement is a presence check.

    /// Declare a REQUIRED constraint on `(label, key)`. Idempotent. Fails with
    /// [`ErrorCode::ConstraintViolation`] if any live vertex with `label` lacks a
    /// present, non-null `key` — an already-violated constraint is meaningless.
    pub fn create_required_constraint(&mut self, label: &str, key: &str) -> CodeResult<()> {
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if self.vlabels[vi as usize].contains(&lid)
                    && matches!(self.props.value(vi as usize, key, &self.strs), Value::Null)
                {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the required constraint being declared",
                    ));
                }
            }
        }
        let keys = self.v_required.entry(label.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Drop a required constraint. Idempotent.
    pub fn drop_required_constraint(&mut self, label: &str, key: &str) {
        if let Some(keys) = self.v_required.get_mut(label) {
            keys.retain(|k| k != key);
            if keys.is_empty() {
                self.v_required.remove(label);
            }
        }
    }

    /// Property keys required for `label` (sorted; empty if none).
    pub fn required_keys(&self, label: &str) -> &[String] {
        self.v_required.get(label).map_or(&[], Vec::as_slice)
    }

    /// True iff `(label, key)` carries a required constraint.
    pub fn has_required_constraint(&self, label: &str, key: &str) -> bool {
        self.v_required
            .get(label)
            .is_some_and(|ks| ks.iter().any(|k| k == key))
    }

    /// Every declared required constraint as sorted `(label, key)` pairs.
    pub fn required_constraints(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .v_required
            .iter()
            .flat_map(|(l, ks)| ks.iter().map(move |k| (l.clone(), k.clone())))
            .collect();
        out.sort();
        out
    }

    /// The first `(label, key)` a new vertex with these `labels`/`props` would
    /// violate by omitting a required key (absent or null value), or `None`.
    pub fn missing_required(
        &self,
        labels: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.v_required.is_empty() && self.v_type_not_null.is_empty() {
            return None;
        }
        for label in labels {
            for key in self.effective_required_keys(label) {
                let present = props
                    .iter()
                    .any(|(k, v)| k == key && !matches!(v, Value::Null));
                if !present {
                    return Some((label.clone(), key.to_string()));
                }
            }
        }
        None
    }

    /// The keys REQUIRED (present + non-null) for `label`: the declared required
    /// constraints UNION the scalar keys declared `NOT NULL` on a type constraint.
    fn effective_required_keys(&self, label: &str) -> Vec<&str> {
        let mut ks: Vec<&str> = self
            .required_keys(label)
            .iter()
            .map(String::as_str)
            .collect();
        if let Some(nn) = self.v_type_not_null.get(label) {
            for k in nn {
                if !ks.contains(&k.as_str()) {
                    ks.push(k);
                }
            }
        }
        ks
    }

    /// True iff `key` is required by a label currently on vertex `vi` (so it can't
    /// be removed or set to null).
    pub fn is_required_key(&self, vi: u32, key: &str) -> bool {
        let carries = |label: &str| {
            self.labels
                .get(label)
                .is_some_and(|lid| self.vlabels[vi as usize].contains(&lid))
        };
        for (label, keys) in &self.v_required {
            if keys.iter().any(|k| k == key) && carries(label) {
                return true;
            }
        }
        // A scalar `NOT NULL` type constraint makes the key required too.
        for (label, keys) in &self.v_type_not_null {
            if keys.contains(key) && carries(label) {
                return true;
            }
        }
        false
    }

    /// If adding `label` to vertex `vi` would violate a required key the vertex
    /// lacks (absent or null), that key; else `None`.
    pub fn required_missing_for_label(&self, vi: u32, label: &str) -> Option<String> {
        for key in self.effective_required_keys(label) {
            if matches!(self.props.value(vi as usize, key, &self.strs), Value::Null) {
                return Some(key.to_string());
            }
        }
        None
    }

    // --- type constraints (R-CONSTRAINTS) ------------------------------------
    // Every present, non-null value under a constrained `key` on a vertex with
    // `label` must be of the declared scalar type. Null/absent are exempt.
    // Enforced in the write path; byte-identical to the TS core.

    /// Declare a TYPE constraint on `(label, key)` requiring `type_name` (one of
    /// string/number/boolean/date/datetime/duration/list). Fails with
    /// `InvalidValue` for an unknown type name, or `ConstraintViolation` if any
    /// existing vertex holds a present, non-null `key` of a different type.
    pub fn create_type_constraint(
        &mut self,
        label: &str,
        key: &str,
        type_name: &str,
    ) -> CodeResult<()> {
        let Some((spec, not_null)) = TypeSpec::parse_with_not_null(type_name) else {
            return Err(CodeError::new(
                ErrorCode::InvalidValue,
                "unknown or malformed type name for a type constraint",
            ));
        };
        // Validate existing data against the declared type (a null is exempt) — and,
        // when `NOT NULL`, that no label vertex already holds an absent/null value.
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if !self.vlabels[vi as usize].contains(&lid) {
                    continue;
                }
                let v = self.props.value(vi as usize, key, &self.strs);
                if !value_matches(&v, &spec) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the type constraint being declared",
                    ));
                }
                if not_null && matches!(v, Value::Null) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the NOT NULL constraint being declared",
                    ));
                }
            }
        }
        // Scalar → the `Copy` `v_type` map (unchanged fast path); a record shape →
        // the parallel `v_record` map.
        match spec {
            TypeSpec::Scalar(ty) => {
                self.v_type
                    .entry(label.to_string())
                    .or_default()
                    .insert(key.to_string(), ty);
                if not_null {
                    self.v_type_not_null
                        .entry(label.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
            record => {
                // De-box the store for this key into typed sub-columns — the shape
                // is now a contract (see [`Column::Record`]). A no-op for `AnyRecord`
                // (no field contract → stays boxed). Future writes scatter in place.
                self.props.debox_record(key, &record, &mut self.strs);
                self.v_record
                    .entry(label.to_string())
                    .or_default()
                    .insert(key.to_string(), record);
                // A record-level `NOT NULL` makes the whole property required.
                if not_null {
                    self.v_type_not_null
                        .entry(label.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
        }
        Ok(())
    }

    /// Drop a type constraint (scalar or record). Idempotent.
    pub fn drop_type_constraint(&mut self, label: &str, key: &str) {
        if let Some(keys) = self.v_type.get_mut(label) {
            keys.remove(key);
            if keys.is_empty() {
                self.v_type.remove(label);
            }
        }
        if let Some(keys) = self.v_record.get_mut(label) {
            keys.remove(key);
            if keys.is_empty() {
                self.v_record.remove(label);
            }
        }
        // Drop this constraint's `NOT NULL` (leaving any independently-declared
        // required constraint on the same key intact).
        if let Some(keys) = self.v_type_not_null.get_mut(label) {
            keys.remove(key);
            if keys.is_empty() {
                self.v_type_not_null.remove(label);
            }
        }
        // Re-box the column once NO label still constrains this key as a record.
        if !self.v_record.values().any(|ks| ks.contains_key(key)) {
            self.props.rebox_record(key, &self.strs);
        }
    }

    /// The first `(label, key)` a new vertex with these `labels`/`props` would
    /// violate by holding a wrong-typed value, or `None`.
    pub fn type_violation(
        &self,
        labels: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.v_type.is_empty() && self.v_record.is_empty() {
            return None;
        }
        for label in labels {
            if let Some(cs) = self.v_type.get(label) {
                for (key, ty) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if let Some(got) = value_type(v) {
                            if got != *ty {
                                return Some((label.clone(), key.clone()));
                            }
                        }
                    }
                }
            }
            if let Some(cs) = self.v_record.get(label) {
                for (key, spec) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if !value_matches(v, spec) {
                            return Some((label.clone(), key.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// True iff setting `vi.key = value` would break a type constraint on one of
    /// `vi`'s labels. A null value is exempt.
    pub fn type_conflict_on_set(&self, vi: u32, key: &str, value: &Value) -> bool {
        // Scalar constraints: a non-scalar (null/map) has no scalar type, so it
        // can't conflict with a scalar declaration here.
        if let Some(got) = value_type(value) {
            for (label, cs) in &self.v_type {
                if let Some(ty) = cs.get(key) {
                    if let Some(lid) = self.labels.get(label) {
                        if self.vlabels[vi as usize].contains(&lid) && got != *ty {
                            return true;
                        }
                    }
                }
            }
        }
        // Record constraints: a map value must match the declared shape (a null is
        // exempt via `value_matches`).
        for (label, cs) in &self.v_record {
            if let Some(spec) = cs.get(key) {
                if let Some(lid) = self.labels.get(label) {
                    if self.vlabels[vi as usize].contains(&lid) && !value_matches(value, spec) {
                        return true;
                    }
                }
            }
        }
        false
    }

    // --- edge constraints (R-CONSTRAINTS, edge types) -----------------------
    // Direct mirror of the vertex unique/required/type constraints, keyed by edge
    // TYPE instead of node label, enforced against the edge property store
    // (`edge_props`) and the edge property index (`eidx`). Byte-identical to the
    // TS edge constraints. Enforcement is deferred to commit (see
    // `run_deferred_checks`), exactly like the vertex ones.

    /// Declare a UNIQUE constraint on `(edge_type, key)`. Creates the backing edge
    /// index if absent. Fails with `ConstraintViolation` if the current data
    /// already violates it. Idempotent.
    pub fn create_edge_unique_constraint(&mut self, etype: &str, key: &str) -> CodeResult<()> {
        if !self.edge_indexed(key) {
            self.create_edge_index(key);
        }
        if self.first_etype_prop_duplicate(etype, key).is_some() {
            return Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                "existing data already violates the edge unique constraint being declared",
            ));
        }
        let keys = self.e_unique.entry(etype.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Drop an edge unique constraint. The backing index is left in place. Idempotent.
    pub fn drop_edge_unique_constraint(&mut self, etype: &str, key: &str) {
        if let Some(keys) = self.e_unique.get_mut(etype) {
            keys.retain(|k| k != key);
            if keys.is_empty() {
                self.e_unique.remove(etype);
            }
        }
    }

    /// Property keys under a unique constraint for `etype` (sorted; empty if none).
    pub fn edge_unique_keys(&self, etype: &str) -> &[String] {
        self.e_unique.get(etype).map_or(&[], Vec::as_slice)
    }

    /// True iff `(edge_type, key)` carries a unique constraint.
    pub fn has_edge_unique_constraint(&self, etype: &str, key: &str) -> bool {
        self.e_unique
            .get(etype)
            .is_some_and(|ks| ks.iter().any(|k| k == key))
    }

    /// Every declared edge unique constraint as sorted `(edge_type, key)` pairs.
    pub fn edge_unique_constraints(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .e_unique
            .iter()
            .flat_map(|(t, ks)| ks.iter().map(move |k| (t.clone(), k.clone())))
            .collect();
        out.sort();
        out
    }

    /// If adding an edge of `etypes` with `props` would break a unique constraint,
    /// the offending `(edge_type, key, existing edge)`. `exclude` skips one edge.
    pub fn edge_unique_conflict(
        &self,
        etypes: &[String],
        props: &[(String, Value)],
        exclude: Option<u32>,
    ) -> Option<(String, String, u32)> {
        if self.e_unique.is_empty() {
            return None;
        }
        for etype in etypes {
            for key in self.edge_unique_keys(etype) {
                let Some((_, value)) = props.iter().find(|(k, _)| k == key) else {
                    continue;
                };
                let hit = self
                    .edges_with_etype_value(etype, key, value)
                    .into_iter()
                    .find(|&e| Some(e) != exclude);
                if let Some(existing) = hit {
                    return Some((etype.clone(), key.clone(), existing));
                }
            }
        }
        None
    }

    /// Declare a REQUIRED constraint on `(edge_type, key)`. Fails with
    /// `ConstraintViolation` if any live edge of `etype` lacks a present, non-null
    /// `key`. Idempotent.
    pub fn create_edge_required_constraint(&mut self, etype: &str, key: &str) -> CodeResult<()> {
        if let Some(edges) = self.edges_with_etype_name(etype) {
            for &ei in edges {
                if matches!(
                    self.edge_props.value(ei as usize, key, &self.strs),
                    Value::Null
                ) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the edge required constraint being declared",
                    ));
                }
            }
        }
        let keys = self.e_required.entry(etype.to_string()).or_default();
        if !keys.iter().any(|k| k == key) {
            keys.push(key.to_string());
            keys.sort();
        }
        Ok(())
    }

    /// Drop an edge required constraint. Idempotent.
    pub fn drop_edge_required_constraint(&mut self, etype: &str, key: &str) {
        if let Some(keys) = self.e_required.get_mut(etype) {
            keys.retain(|k| k != key);
            if keys.is_empty() {
                self.e_required.remove(etype);
            }
        }
    }

    /// Property keys required for edge type `etype` (sorted; empty if none).
    pub fn edge_required_keys(&self, etype: &str) -> &[String] {
        self.e_required.get(etype).map_or(&[], Vec::as_slice)
    }

    /// True iff `(edge_type, key)` carries a required constraint.
    pub fn has_edge_required_constraint(&self, etype: &str, key: &str) -> bool {
        self.e_required
            .get(etype)
            .is_some_and(|ks| ks.iter().any(|k| k == key))
    }

    /// Every declared edge required constraint as sorted `(edge_type, key)` pairs.
    pub fn edge_required_constraints(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .e_required
            .iter()
            .flat_map(|(t, ks)| ks.iter().map(move |k| (t.clone(), k.clone())))
            .collect();
        out.sort();
        out
    }

    /// The first `(edge_type, key)` a new edge with these `etypes`/`props` would
    /// violate by omitting a required key (absent or null value), or `None`.
    pub fn edge_missing_required(
        &self,
        etypes: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.e_required.is_empty() && self.e_type_not_null.is_empty() {
            return None;
        }
        for etype in etypes {
            let mut keys: Vec<&str> = self
                .edge_required_keys(etype)
                .iter()
                .map(String::as_str)
                .collect();
            if let Some(nn) = self.e_type_not_null.get(etype) {
                for k in nn {
                    if !keys.contains(&k.as_str()) {
                        keys.push(k);
                    }
                }
            }
            for key in keys {
                let present = props
                    .iter()
                    .any(|(k, v)| k == key && !matches!(v, Value::Null));
                if !present {
                    return Some((etype.clone(), key.to_string()));
                }
            }
        }
        None
    }

    /// Declare a TYPE constraint on `(edge_type, key)` requiring `type_name`. Fails
    /// with `InvalidValue` for an unknown type name, or `ConstraintViolation` if
    /// any existing edge holds a present, non-null `key` of a different type.
    pub fn create_edge_type_constraint(
        &mut self,
        etype: &str,
        key: &str,
        type_name: &str,
    ) -> CodeResult<()> {
        let Some((spec, not_null)) = TypeSpec::parse_with_not_null(type_name) else {
            return Err(CodeError::new(
                ErrorCode::InvalidValue,
                "unknown or malformed type name for an edge type constraint",
            ));
        };
        if let Some(edges) = self.edges_with_etype_name(etype) {
            for &ei in edges {
                let v = self.edge_props.value(ei as usize, key, &self.strs);
                if !value_matches(&v, &spec) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the edge type constraint being declared",
                    ));
                }
                if not_null && matches!(v, Value::Null) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the NOT NULL constraint being declared",
                    ));
                }
            }
        }
        match spec {
            TypeSpec::Scalar(ty) => {
                self.e_type_constraints
                    .entry(etype.to_string())
                    .or_default()
                    .insert(key.to_string(), ty);
                if not_null {
                    self.e_type_not_null
                        .entry(etype.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
            record => {
                self.edge_props.debox_record(key, &record, &mut self.strs);
                self.e_record
                    .entry(etype.to_string())
                    .or_default()
                    .insert(key.to_string(), record);
                if not_null {
                    self.e_type_not_null
                        .entry(etype.to_string())
                        .or_default()
                        .insert(key.to_string());
                }
            }
        }
        Ok(())
    }

    /// Drop an edge type constraint (scalar or record). Idempotent.
    pub fn drop_edge_type_constraint(&mut self, etype: &str, key: &str) {
        if let Some(keys) = self.e_type_constraints.get_mut(etype) {
            keys.remove(key);
            if keys.is_empty() {
                self.e_type_constraints.remove(etype);
            }
        }
        if let Some(keys) = self.e_record.get_mut(etype) {
            keys.remove(key);
            if keys.is_empty() {
                self.e_record.remove(etype);
            }
        }
        if let Some(keys) = self.e_type_not_null.get_mut(etype) {
            keys.remove(key);
            if keys.is_empty() {
                self.e_type_not_null.remove(etype);
            }
        }
        if !self.e_record.values().any(|ks| ks.contains_key(key)) {
            self.edge_props.rebox_record(key, &self.strs);
        }
    }

    /// The declared type for edge `(edge_type, key)`, or `None`.
    pub fn edge_type_constraint(&self, etype: &str, key: &str) -> Option<PropType> {
        self.e_type_constraints.get(etype)?.get(key).copied()
    }

    /// Every declared edge type constraint as sorted `(edge_type, key)` pairs.
    pub fn edge_type_constraints(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .e_type_constraints
            .iter()
            .flat_map(|(t, ks)| ks.keys().map(move |k| (t.clone(), k.clone())))
            .collect();
        out.sort();
        out
    }

    /// The first `(edge_type, key)` a new edge with these `etypes`/`props` would
    /// violate by holding a wrong-typed value, or `None`.
    pub fn edge_type_violation(
        &self,
        etypes: &[String],
        props: &[(String, Value)],
    ) -> Option<(String, String)> {
        if self.e_type_constraints.is_empty() && self.e_record.is_empty() {
            return None;
        }
        for etype in etypes {
            if let Some(cs) = self.e_type_constraints.get(etype) {
                for (key, ty) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if let Some(got) = value_type(v) {
                            if got != *ty {
                                return Some((etype.clone(), key.clone()));
                            }
                        }
                    }
                }
            }
            if let Some(cs) = self.e_record.get(etype) {
                for (key, spec) in cs {
                    if let Some((_, v)) = props.iter().find(|(k, _)| k == key) {
                        if !value_matches(v, spec) {
                            return Some((etype.clone(), key.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Live edges of type `etype` whose property `key == value`. Seeks the backing
    /// edge index (a constraint always creates one), falling back to a scan.
    /// Non-indexable values (null/list) yield an empty set — exempt from uniqueness.
    fn edges_with_etype_value(&self, etype: &str, key: &str, value: &Value) -> Vec<u32> {
        let Some(idxk) = IdxKey::from_value(value) else {
            return Vec::new();
        };
        let Some(tid) = self.etype.get(etype) else {
            return Vec::new();
        };
        match self.edges_by_prop(key, &idxk) {
            Some(ids) => ids
                .iter()
                .copied()
                .filter(|&e| self.is_edge_live(e) && self.e_type[e as usize] == tid)
                .collect(),
            None => (0..self.e_src.len() as u32)
                .filter(|&e| {
                    self.is_edge_live(e)
                        && self.e_type[e as usize] == tid
                        && self.edge_props.value(e as usize, key, &self.strs) == *value
                })
                .collect(),
        }
    }

    /// The first pair of live `etype`-edges that share a value for `key` — for
    /// validating an edge unique constraint against existing data at declare time.
    fn first_etype_prop_duplicate(&self, etype: &str, key: &str) -> Option<(u32, u32)> {
        let tid = self.etype.get(etype)?;
        let bt = self.eidx.get(key)?;
        for ids in bt.values() {
            let mut with_type = ids
                .iter()
                .copied()
                .filter(|&e| self.is_edge_live(e) && self.e_type[e as usize] == tid);
            if let (Some(a), Some(b)) = (with_type.next(), with_type.next()) {
                return Some((a, b));
            }
        }
        None
    }

    /// The single type name an edge carries (empty vec for a type-less edge) — the
    /// edge analogue of a vertex's label list (an edge has exactly one type).
    fn edge_type_names(&self, ei: u32) -> Vec<String> {
        let name = self.etype.text(self.e_type[ei as usize]).to_string();
        if name.is_empty() {
            Vec::new()
        } else {
            vec![name]
        }
    }

    /// A live edge's present properties as `(key, value)` pairs — the shape the edge
    /// constraint predicates consume. Edge analogue of `vertex_props`.
    fn edge_props_of(&self, ei: u32) -> Vec<(String, Value)> {
        let i = ei as usize;
        let mut out = Vec::new();
        for kid in 0..self.edge_props.cols.len() as u32 {
            if self.edge_props.is_present_id(i, kid) {
                let key = self.edge_props.keys.text(kid).to_string();
                let val = self.edge_props.value_id(i, kid, &self.strs);
                out.push((key, val));
            }
        }
        out
    }

    // --- cardinality constraints (R-CONSTRAINTS, degree bounds) --------------
    // Bound the degree of every vertex carrying `label` over `etype` in
    // `direction` (0 = out / the vertex is the edge source, 1 = in / the target).
    // Max is deferred to commit against touched endpoints (the GQL layer runs
    // every statement in an auto-commit frame, so a single over-max edge INSERT is
    // caught there); min is commit-time only (unsatisfiable by a single write).
    // The edge write paths note both endpoints as touched; `run_deferred_checks`
    // re-checks them. Byte-identical to the TS core.

    /// Number of live `etype` edges for which `vi` is the SOURCE (out-degree). The
    /// adjacency lists hold only live edges, so this is a filtered count. A
    /// self-loop appears in `out` once, so it counts once here (and once for `in`).
    pub fn out_degree(&self, vi: u32, etype: &str) -> u32 {
        let Some(tid) = self.etype.get(etype) else {
            return 0;
        };
        self.out[vi as usize]
            .iter()
            .filter(|a| a.etype == tid)
            .count() as u32
    }

    /// Number of live `etype` edges for which `vi` is the TARGET (in-degree).
    pub fn in_degree(&self, vi: u32, etype: &str) -> u32 {
        let Some(tid) = self.etype.get(etype) else {
            return 0;
        };
        self.in_[vi as usize]
            .iter()
            .filter(|a| a.etype == tid)
            .count() as u32
    }

    /// Degree of `vi` over `etype` in `direction` (0 = out, 1 = in).
    fn degree_dir(&self, vi: u32, etype: &str, direction: u8) -> u32 {
        if direction == 0 {
            self.out_degree(vi, etype)
        } else {
            self.in_degree(vi, etype)
        }
    }

    /// Declare a CARDINALITY constraint bounding the degree of every vertex
    /// carrying `label` over `etype` in `direction` (0 = out, 1 = in) to
    /// `min..=max` (`max: None` unbounded). Re-declaring `(label, etype,
    /// direction)` replaces the bounds. Fails with `ConstraintViolation` if any
    /// existing vertex already violates it (mirrors unique/required declare-time).
    pub fn create_cardinality_constraint(
        &mut self,
        label: &str,
        etype: &str,
        direction: u8,
        min: u32,
        max: Option<u32>,
    ) -> CodeResult<()> {
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if !self.vlabels[vi as usize].contains(&lid) {
                    continue;
                }
                let d = self.degree_dir(vi, etype, direction);
                if d < min || max.is_some_and(|m| d > m) {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the cardinality constraint being declared",
                    ));
                }
            }
        }
        let rule = CardinalityRule {
            label: label.to_string(),
            etype: etype.to_string(),
            direction,
            min,
            max,
        };
        if let Some(existing) = self.v_cardinality.iter_mut().find(|c| {
            c.label == rule.label && c.etype == rule.etype && c.direction == rule.direction
        }) {
            *existing = rule;
        } else {
            self.v_cardinality.push(rule);
        }
        Ok(())
    }

    /// Drop a cardinality constraint on `(label, etype, direction)`. Idempotent.
    pub fn drop_cardinality_constraint(&mut self, label: &str, etype: &str, direction: u8) {
        self.v_cardinality
            .retain(|c| !(c.label == label && c.etype == etype && c.direction == direction));
    }

    /// Every declared cardinality constraint as sorted `(label, etype, direction,
    /// min, max)` tuples — introspection, sorted for a deterministic listing.
    pub fn cardinality_constraints(&self) -> Vec<(String, String, u8, u32, Option<u32>)> {
        let mut out: Vec<(String, String, u8, u32, Option<u32>)> = self
            .v_cardinality
            .iter()
            .map(|c| (c.label.clone(), c.etype.clone(), c.direction, c.min, c.max))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        out
    }

    // --- VALIDATORS (custom GQL-predicate constraints) -----------------------

    /// Declare a VALIDATOR on `label` (a vertex label OR an edge type): every
    /// element carrying `label` must satisfy the GQL boolean `predicate`, with the
    /// element bound to `var`. Appends (a label may carry several). The predicate
    /// is parsed+lowered once here. Two failure modes, distinguished by error code
    /// so the FFI can map them: an unparseable predicate returns
    /// `ErrorCode::Syntax`; existing data that already evaluates to a definite
    /// `false` returns `ErrorCode::ConstraintViolation` (the declare-time scan).
    /// SQL-`CHECK` semantics — a null/unknown result passes.
    pub fn create_validator(&mut self, label: &str, var: &str, predicate: &str) -> CodeResult<()> {
        let expr = crate::gql::parser::parse_predicate(predicate)
            .map_err(|e| CodeError::new(ErrorCode::Syntax, e.message))?;

        // Reject a predicate that references any variable *other* than the declared
        // `var` at DECLARE time. Such a name (`x.age` when the binding is `u`, or a
        // bare `age`) is unbound → the predicate reads UNKNOWN → the SQL-`CHECK`
        // never fires and the validator silently does nothing. A predicate with no
        // variable at all (a constant like `1 = 1`) is legitimately allowed. Uses
        // `ErrorCode::Syntax` (the FFI already maps a bad predicate to `-2`/`E_SYNTAX`)
        // so both engines reject identically.
        if let Some(name) = crate::gql::plan::free_predicate_vars(&expr)
            .into_iter()
            .find(|n| n != var)
        {
            return Err(CodeError::new(
                ErrorCode::Syntax,
                format!(
                    "validator predicate references unbound variable `{name}` \
                     (only the declared variable `{var}` is in scope)"
                ),
            ));
        }

        let pred = crate::gql::plan::lower_predicate(var, &expr);

        // Declare-time scan: reject if any existing element carrying `label` (a
        // vertex OR an edge — one namespace) currently evaluates to a definite
        // false. An already-violated validator is meaningless (mirrors the other
        // constraints). A predicate evaluation fault (e.g. an unknown function)
        // surfaces verbatim via `?`.
        if let Some(lid) = self.labels.get(label) {
            for vi in self.vertex_indices() {
                if self.vlabels[vi as usize].contains(&lid)
                    && crate::gql::eval::eval_predicate(
                        self,
                        &pred,
                        crate::gql::eval::Val::Node(vi),
                    )? == Some(false)
                {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the validator being declared",
                    ));
                }
            }
        }

        if let Some(tid) = self.etype.get(label) {
            // `edges_with_etype` borrows `self.by_etype`; copy the indices out so the
            // per-edge `eval_predicate(self, …)` isn't a second overlapping borrow.
            let eids: Vec<u32> = self.edges_with_etype(tid).to_vec();
            for ei in eids {
                if self.is_edge_live(ei)
                    && crate::gql::eval::eval_predicate(
                        self,
                        &pred,
                        crate::gql::eval::Val::Edge(ei),
                    )? == Some(false)
                {
                    return Err(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        "existing data already violates the validator being declared",
                    ));
                }
            }
        }

        self.v_validators
            .entry(label.to_string())
            .or_default()
            .push(ValidatorRule {
                var: var.to_string(),
                src: predicate.to_string(),
                pred,
            });
        Ok(())
    }

    /// Drop every validator declared on `label`. Idempotent.
    pub fn drop_validator(&mut self, label: &str) {
        self.v_validators.remove(label);
    }

    /// Every declared validator as `(label, var, src)`, sorted by `(label, src)`.
    /// The compiled predicate is internal. Introspection for tests/tooling.
    pub fn validators(&self) -> Vec<(String, String, String)> {
        let mut out: Vec<(String, String, String)> = self
            .v_validators
            .iter()
            .flat_map(|(label, rules)| {
                rules
                    .iter()
                    .map(move |r| (label.clone(), r.var.clone(), r.src.clone()))
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
        out
    }

    /// Check every validator declared on a touched vertex `vi`. `Ok(())` if all
    /// pass (a null/unknown result passes); `Err` on a definite `false` or an
    /// evaluation fault. The commit-time check (the eager per-write gate is the
    /// statement's auto-commit, which runs this via `run_deferred_checks`).
    fn check_validators_vertex(&self, vi: u32) -> CodeResult<()> {
        if self.v_validators.is_empty() {
            return Ok(());
        }
        for &lid in &self.vlabels[vi as usize] {
            let name = self.labels.text(lid);
            if let Some(rules) = self.v_validators.get(name) {
                for rule in rules {
                    if crate::gql::eval::eval_predicate(
                        self,
                        &rule.pred,
                        crate::gql::eval::Val::Node(vi),
                    )? == Some(false)
                    {
                        return Err(CodeError::new(
                            ErrorCode::ConstraintViolation,
                            format!("validator '{}' on '{}' violated", rule.src, name),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Edge analogue of [`Graph::check_validators_vertex`].
    fn check_validators_edge(&self, ei: u32) -> CodeResult<()> {
        if self.v_validators.is_empty() {
            return Ok(());
        }
        for name in self.edge_type_names(ei) {
            if let Some(rules) = self.v_validators.get(&name) {
                for rule in rules {
                    if crate::gql::eval::eval_predicate(
                        self,
                        &rule.pred,
                        crate::gql::eval::Val::Edge(ei),
                    )? == Some(false)
                    {
                        return Err(CodeError::new(
                            ErrorCode::ConstraintViolation,
                            format!("validator '{}' on '{}' violated", rule.src, name),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Declare a graph-level INVARIANT `name` = a whole-graph GQL `query` that must
    /// hold after every write transaction. The query is parsed+lowered once here;
    /// an unparseable query returns [`ErrorCode::Syntax`] (mapped to `-2` at the
    /// FFI). VIOLATED iff any cell in its result set is boolean `false` (everything
    /// else — `true`/`null`/non-boolean/empty — holds). A declare-time run rejects
    /// with [`ErrorCode::ConstraintViolation`] if the current graph already
    /// violates it (an already-broken invariant is meaningless, mirroring the
    /// validators/constraints). Re-declaring the same `name` replaces the prior
    /// query. Byte-identical with the TS `createInvariant`.
    pub fn create_invariant(&mut self, name: &str, query: &str) -> CodeResult<()> {
        let plan =
            crate::gql::prepare(query).map_err(|e| CodeError::new(ErrorCode::Syntax, e.message))?;

        // Declare-time run against the current graph: reject on a definite-`false`
        // cell (or surface an evaluation fault verbatim via `?`).
        let rows = crate::gql::run_invariant(&plan, self)?;
        if Self::invariant_violated(&rows) {
            return Err(CodeError::new(
                ErrorCode::ConstraintViolation,
                format!("existing data already violates the invariant '{name}'"),
            ));
        }

        // Replace any prior invariant of the same name (declare is idempotent-ish:
        // last query wins), then append.
        self.v_invariants.retain(|r| r.name != name);
        self.v_invariants.push(InvariantRule {
            name: name.to_string(),
            src: query.to_string(),
            plan: std::sync::Arc::new(plan),
        });
        Ok(())
    }

    /// Drop the graph-level invariant named `name`. Idempotent.
    pub fn drop_invariant(&mut self, name: &str) {
        self.v_invariants.retain(|r| r.name != name);
    }

    /// Every declared invariant as `(name, src)`, sorted by name. The compiled
    /// query plan is internal. Introspection for tests/tooling.
    pub fn invariants(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .v_invariants
            .iter()
            .map(|r| (r.name.clone(), r.src.clone()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The full active schema as a JSON array of replayable **op objects** — the
    /// read side of [`create_*`] (the inverse of applying them). Each element
    /// mirrors the TS `SchemaOp` union (`{"op":"createUniqueConstraint","label":…,
    /// "key":…}`, …), emitted in a fixed section order (indexes → node constraints →
    /// edge constraints → cardinality → validators → invariants), each section
    /// sorted, so the output is **deterministic** for a given schema. The snapshot
    /// codec calls this to persist schema alongside the graph NDJSON; a cold boot
    /// replays each op via `applySchemaOp`, so a restored replica keeps the
    /// constraints/validators/indexes it can't reconstruct from data alone.
    pub fn dump_schema(&self) -> String {
        let mut ops: Vec<String> = Vec::new();

        for k in self.vertex_indexes() {
            ops.push(schema_op("createVertexIndex", &[("key", Jv::S(&k))]));
        }
        for k in self.edge_indexes() {
            ops.push(schema_op("createEdgeIndex", &[("key", Jv::S(&k))]));
        }
        for (label, key) in self.unique_constraints() {
            ops.push(schema_op(
                "createUniqueConstraint",
                &[("label", Jv::S(&label)), ("key", Jv::S(&key))],
            ));
        }
        for (label, key) in self.required_constraints() {
            ops.push(schema_op(
                "createRequiredConstraint",
                &[("label", Jv::S(&label)), ("key", Jv::S(&key))],
            ));
        }
        // Node type constraints carry the type name (scalar OR a `record{…}`),
        // which the `(label, key)` readers drop — iterate both maps directly
        // (sorted for determinism), so a record constraint round-trips too.
        let mut vtypes: Vec<(String, String, String)> = self
            .v_type
            .iter()
            .flat_map(|(label, ks)| {
                ks.iter().map(move |(k, t)| {
                    // Round-trip a scalar `NOT NULL` type constraint.
                    let nn = self
                        .v_type_not_null
                        .get(label)
                        .is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", t.to_name())
                    } else {
                        t.to_name().to_string()
                    };
                    (label.clone(), k.clone(), ty)
                })
            })
            .chain(self.v_record.iter().flat_map(|(label, ks)| {
                ks.iter().map(move |(k, spec)| {
                    let nn = self
                        .v_type_not_null
                        .get(label)
                        .is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", spec.to_name())
                    } else {
                        spec.to_name()
                    };
                    (label.clone(), k.clone(), ty)
                })
            }))
            .collect();
        vtypes.sort();
        for (label, key, ty) in vtypes {
            ops.push(schema_op(
                "createTypeConstraint",
                &[
                    ("label", Jv::S(&label)),
                    ("key", Jv::S(&key)),
                    ("type", Jv::S(&ty)),
                ],
            ));
        }
        for (etype, key) in self.edge_unique_constraints() {
            ops.push(schema_op(
                "createEdgeUniqueConstraint",
                &[("edgeType", Jv::S(&etype)), ("key", Jv::S(&key))],
            ));
        }
        for (etype, key) in self.edge_required_constraints() {
            ops.push(schema_op(
                "createEdgeRequiredConstraint",
                &[("edgeType", Jv::S(&etype)), ("key", Jv::S(&key))],
            ));
        }
        let mut etypes: Vec<(String, String, String)> = self
            .e_type_constraints
            .iter()
            .flat_map(|(et, ks)| {
                ks.iter().map(move |(k, t)| {
                    let nn = self.e_type_not_null.get(et).is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", t.to_name())
                    } else {
                        t.to_name().to_string()
                    };
                    (et.clone(), k.clone(), ty)
                })
            })
            .chain(self.e_record.iter().flat_map(|(et, ks)| {
                ks.iter().map(move |(k, spec)| {
                    let nn = self.e_type_not_null.get(et).is_some_and(|s| s.contains(k));
                    let ty = if nn {
                        format!("{} NOT NULL", spec.to_name())
                    } else {
                        spec.to_name()
                    };
                    (et.clone(), k.clone(), ty)
                })
            }))
            .collect();
        etypes.sort();
        for (etype, key, ty) in etypes {
            ops.push(schema_op(
                "createEdgeTypeConstraint",
                &[
                    ("edgeType", Jv::S(&etype)),
                    ("key", Jv::S(&key)),
                    ("type", Jv::S(&ty)),
                ],
            ));
        }
        for (label, etype, dir, min, max) in self.cardinality_constraints() {
            let direction = if dir == 0 { "out" } else { "in" };
            ops.push(schema_op(
                "createCardinalityConstraint",
                &[
                    ("label", Jv::S(&label)),
                    ("edgeType", Jv::S(&etype)),
                    ("direction", Jv::S(direction)),
                    ("min", Jv::N(min)),
                    ("max", Jv::NOpt(max)),
                ],
            ));
        }
        for (label, var, predicate) in self.validators() {
            ops.push(schema_op(
                "createValidator",
                &[
                    ("label", Jv::S(&label)),
                    ("varName", Jv::S(&var)),
                    ("predicate", Jv::S(&predicate)),
                ],
            ));
        }
        for (name, query) in self.invariants() {
            ops.push(schema_op(
                "createInvariant",
                &[("name", Jv::S(&name)), ("query", Jv::S(&query))],
            ));
        }

        let mut json = String::from("[");
        json.push_str(&ops.join(","));
        json.push(']');
        json
    }

    /// `false`-only-fails: a result set VIOLATES an invariant iff any cell is a
    /// boolean `false`. A `true`, a `null`, a non-boolean value (number/string/
    /// list/map/temporal), or an empty result set all HOLD. Byte-identical to the
    /// TS `invariantViolated` (`cell === false`).
    fn invariant_violated(rows: &crate::query::RowSet) -> bool {
        rows.data.iter().any(|v| matches!(v, Value::Bool(false)))
    }

    /// Run every declared invariant against the fully-staged graph. Called from
    /// [`Graph::commit_tx`] only when the transaction actually wrote something.
    /// `Ok(())` if all hold; `Err` carrying the failing invariant's error (a
    /// `ConstraintViolation` for a `false` cell, or an evaluation fault's own code).
    fn check_invariants(&mut self) -> CodeResult<()> {
        if self.v_invariants.is_empty() {
            return Ok(());
        }
        // Move the rules out so the read-only `run_invariant(&plan, self)` can take
        // `&mut self` without overlapping the borrow; the run never mutates the
        // registry, so restoring the same Vec afterwards is exact.
        let rules = std::mem::take(&mut self.v_invariants);
        let mut failure: Option<CodeError> = None;
        for rule in &rules {
            match crate::gql::run_invariant(&rule.plan, self) {
                Ok(rows) if Self::invariant_violated(&rows) => {
                    failure = Some(CodeError::new(
                        ErrorCode::ConstraintViolation,
                        format!("invariant '{}' violated", rule.name),
                    ));
                    break;
                }
                Ok(_) => {}
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        self.v_invariants = rules;
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// True iff a touched vertex `vi` violates any cardinality constraint on one of
    /// its labels (degree below `min` or above `max`). The commit-time check.
    fn cardinality_violation(&self, vi: u32) -> bool {
        if self.v_cardinality.is_empty() {
            return false;
        }
        let lids = &self.vlabels[vi as usize];
        for c in &self.v_cardinality {
            let Some(lid) = self.labels.get(&c.label) else {
                continue;
            };
            if !lids.contains(&lid) {
                continue;
            }
            let d = self.degree_dir(vi, &c.etype, c.direction);
            if d < c.min || c.max.is_some_and(|m| d > m) {
                return true;
            }
        }
        false
    }

    /// Note both endpoints of edge `ei` as touched for the commit-time cardinality
    /// recheck (their degree changed). No-op outside a transaction / during a
    /// rollback replay, or when no cardinality constraint is declared. Called by
    /// the edge write paths (`add_edge` / `remove_edge`), so a vertex-delete
    /// cascade re-checks the surviving neighbor too — mirrors the TS core, whose
    /// `insertEdge` / `removeEdge` note endpoints at the same core boundary.
    fn cardinality_note_endpoints(&mut self, ei: u32) {
        if self.v_cardinality.is_empty() || !self.tx_active() {
            return;
        }
        let i = ei as usize;
        let (from, to) = (self.e_src[i], self.e_dst[i]);
        self.tx_touched.push(from);
        self.tx_touched.push(to);
    }

    /// Live vertices carrying `label` whose property `key == value`. Seeks the
    /// backing index (a constraint always creates one), falling back to a scan if
    /// somehow unindexed. Non-indexable values (null/list) yield an empty set —
    /// exempt from uniqueness (SQL: NULLs distinct), matching the value index.
    fn vertices_with_label_value(&self, label: &str, key: &str, value: &Value) -> Vec<u32> {
        let Some(idxk) = IdxKey::from_value(value) else {
            return Vec::new();
        };
        let Some(lid) = self.labels.get(label) else {
            return Vec::new();
        };
        match self.vertices_by_prop(key, &idxk) {
            Some(ids) => ids
                .iter()
                .copied()
                .filter(|&v| self.vlabels[v as usize].contains(&lid))
                .collect(),
            None => self
                .vertex_indices()
                .filter(|&v| {
                    self.vlabels[v as usize].contains(&lid)
                        && self.props.value(v as usize, key, &self.strs) == *value
                })
                .collect(),
        }
    }

    /// The first pair of live `label`-vertices that share a value for `key` — for
    /// validating a unique constraint against existing data at declare time.
    /// Reuses the (freshly built) backing index; null/list values are exempt.
    fn first_label_prop_duplicate(&self, label: &str, key: &str) -> Option<(u32, u32)> {
        let lid = self.labels.get(label)?;
        let bt = self.vidx.get(key)?;
        for ids in bt.values() {
            let mut with_label = ids
                .iter()
                .copied()
                .filter(|&v| self.vlabels[v as usize].contains(&lid));
            if let (Some(a), Some(b)) = (with_label.next(), with_label.next()) {
                return Some((a, b));
            }
        }
        None
    }

    /// Equality seek over vertices: live vertices whose `key` == `value` (None = no index).
    pub fn vertices_by_prop(&self, key: &str, value: &IdxKey) -> Option<&[u32]> {
        Some(
            self.vidx
                .get(key)?
                .get(value)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        )
    }
    /// Equality seek over edges.
    pub fn edges_by_prop(&self, key: &str, value: &IdxKey) -> Option<&[u32]> {
        Some(
            self.eidx
                .get(key)?
                .get(value)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        )
    }
    /// Cardinality of a vertex equality seek (for cardinality-based seed selection).
    pub fn count_by_prop(&self, key: &str, value: &IdxKey) -> Option<usize> {
        Some(self.vidx.get(key)?.get(value).map_or(0, Vec::len))
    }
    /// Range seek over vertices (union of buckets in `bound`, type-block bounded).
    pub fn vertices_by_prop_range(&self, key: &str, bound: &RangeBound) -> Option<Vec<u32>> {
        range_seek(self.vidx.get(key)?, bound)
    }
    /// Range seek over edges.
    pub fn edges_by_prop_range(&self, key: &str, bound: &RangeBound) -> Option<Vec<u32>> {
        range_seek(self.eidx.get(key)?, bound)
    }

    /// The cached CSR snapshot, built (once) from `out`/`in_` on first use and
    /// reused until a topology mutation drops it. Disjoint-field capture lets the
    /// init closure read `out`/`in_` while `get_or_init` holds `csr`.
    /// Drop the CSR snapshot — called by every topology mutation so a later read
    /// rebuilds it. A no-op cost when it was never built.
    fn invalidate_csr(&mut self) {
        self.csr.take();
        self.csr_reads
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// One vertex's adjacency slots, `out` selecting the forward index.
    ///
    /// The CSR snapshot is a **cache-locality optimization for bulk traversal, not
    /// a correctness requirement**: a single vertex's slots are equally available
    /// from the `out`/`in_` delta, in O(degree), and in the identical order
    /// (`csr_pack` concatenates the per-vertex `Vec`s as-is). Building it is
    /// O(V+E), so serving reads exclusively through `get_or_init` meant the first
    /// read after *every* write repacked the entire graph — an interleaved
    /// write→read workload paid O(V+E) per read and went quadratic overall, while
    /// warm read-only scans looked perfectly fine.
    ///
    /// So: use the snapshot when it already exists, otherwise read the delta and
    /// only rebuild once enough reads have accumulated to amortize the repack. A
    /// bulk scan crosses the threshold almost immediately and gets its locality; a
    /// write-heavy workload never pays for a snapshot it would discard.
    #[inline]
    fn adj(&self, v: u32, out: bool) -> &[Adj] {
        if let Some(c) = self.csr.get() {
            return if out { c.out(v) } else { c.in_(v) };
        }

        if self
            .csr_reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            >= CSR_WARM_READS
        {
            let c = self.csr.get_or_init(|| Csr::build(&self.out, &self.in_));

            return if out { c.out(v) } else { c.in_(v) };
        }

        let delta = if out { &self.out } else { &self.in_ };

        delta.get(v as usize).map_or(&[][..], Vec::as_slice)
    }

    /// Out-edges of `v` as adjacency slots.
    pub fn out_adj(&self, v: u32) -> impl Iterator<Item = Adj> + '_ {
        self.adj(v, true).iter().copied()
    }
    /// In-edges of `v` as adjacency slots (the reverse index).
    pub fn in_adj(&self, v: u32) -> impl Iterator<Item = Adj> + '_ {
        self.adj(v, false).iter().copied()
    }
    /// Out-neighbors of `v` whose edge type is `etype` (or all if `None`).
    pub fn out_neighbors(&self, v: u32, etype: Option<u32>) -> impl Iterator<Item = u32> + '_ {
        self.adj(v, true).iter().filter_map(move |a| match etype {
            Some(t) if a.etype != t => None,
            _ => Some(a.nbr),
        })
    }

    /// Labels carried by vertex `v`, as label ids.
    pub fn vertex_labels(&self, v: u32) -> &[u32] {
        &self.vlabels[v as usize]
    }
    /// Does vertex `v` carry label id `l`?
    pub fn has_label(&self, v: u32, l: u32) -> bool {
        self.vlabels[v as usize].contains(&l)
    }
    /// Live vertices carrying label `l`.
    pub fn vertices_with_label(&self, l: u32) -> &[u32] {
        self.by_label.get(&l).map_or(&[], |v| v.as_slice())
    }
    /// Live edges of type id `t` (the seed for `()-[:T]->()` patterns).
    pub fn edges_with_etype(&self, t: u32) -> &[u32] {
        self.by_etype.get(&t).map_or(&[], |e| e.as_slice())
    }
    /// Live edges of type `name`, or `None` if the type was never interned.
    pub fn edges_with_etype_name(&self, name: &str) -> Option<&[u32]> {
        self.etype.get(name).map(|t| self.edges_with_etype(t))
    }

    /// The id of edge `eidx`: its assigned external id, or — since every edge has
    /// an id — the canonical `e{index}` derived from its dense index. The
    /// synthetic id is computed on demand, so the id overlay stays lazy and the
    /// load path pays nothing. Used by codecs (which always emit it) and the
    /// engines' `id()` step.
    pub fn edge_id(&self, eidx: u32) -> std::borrow::Cow<'_, str> {
        match self.eid_fwd.get(&eidx) {
            Some(s) => std::borrow::Cow::Borrowed(s.as_ref()),
            None => std::borrow::Cow::Owned(format!("e{eidx}")),
        }
    }
    /// The edge carrying id `id` — the reverse of [`Graph::edge_id`]. Resolves an
    /// assigned external id first, then the canonical `e{index}` form of a live,
    /// id-less edge (an explicit id shadows a colliding `e{n}`).
    pub fn edge_by_id(&self, id: &str) -> Option<u32> {
        if let Some(&e) = self.eid_rev.get(id) {
            return Some(e);
        }
        let n: u32 = id.strip_prefix('e')?.parse().ok()?;
        self.is_edge_live(n).then_some(n)
    }

    /// The dense index of the vertex with external `id`, or `None`. Non-mutating
    /// (unlike `vid.intern`) — used to detect id clashes on bulk append.
    pub fn vertex_by_id(&self, id: &str) -> Option<u32> {
        self.vid.get(id)
    }
    // --- reactive change tracking ----------------------------------------

    /// Monotonic mutation counter. An unchanged value means nothing has mutated
    /// since it was last read — the O(1) check a `getSnapshot` uses to return a
    /// referentially-stable snapshot.
    pub fn version(&self) -> u64 {
        self.version
    }
    /// Per-token change epoch for a label / edge-type / property-key `name`
    /// (0 if never touched). Lets a live query recompute only when one of its
    /// declared dependencies actually changed.
    pub fn epoch(&self, name: &str) -> u64 {
        self.epochs.get(name).copied().unwrap_or(0)
    }
    /// Bump the global version (called by every mutation).
    fn bump(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
    /// Bump one token's epoch.
    fn touch(&mut self, name: &str) {
        *self.epochs.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Assign (or replace) edge `eidx`'s external id. No-op for a dead edge.
    pub fn set_edge_id(&mut self, eidx: u32, id: &str) {
        if !self.is_edge_live(eidx) {
            return;
        }
        self.bump();
        // Drop any prior id for this edge (and its reverse entry) before re-binding.
        if let Some(old) = self.eid_fwd.remove(&eidx) {
            self.eid_rev.remove(&old);
        }
        let arc: Arc<str> = Arc::from(id);
        self.eid_fwd.insert(eidx, arc.clone());
        self.eid_rev.insert(arc, eidx);
    }

    // --- transactions (R-TX) -----------------------------------------------
    // An atomic mutation boundary with rollback + deferred constraint checks.
    // Mechanism: eager-apply + undo-log + deferred-check-at-commit. Writes apply
    // immediately (read-your-writes), each recording an inverse op; the built-in
    // constraint checks defer to commit, run once against the fully-staged graph;
    // on failure the whole transaction rolls back via the undo log. The engine is
    // single-writer and synchronous — no concurrency, MVCC, or isolation levels.
    // Byte-identical to the TS core (`packages/core/src/core/Graph.ts`).

    /// True while a transaction is open and recording writes (not during a
    /// rollback replay). Mutations consult this to decide whether to record undo /
    /// note a touched vertex.
    #[inline]
    pub fn tx_active(&self) -> bool {
        self.tx_depth > 0 && !self.applying_undo
    }

    /// Is a transaction currently open (at any nesting depth)?
    #[inline]
    pub fn in_transaction(&self) -> bool {
        self.tx_depth > 0
    }

    /// Is the active explicit transaction READ ONLY? Set by ISO GQL
    /// `START TRANSACTION READ ONLY`, cleared on commit/rollback. The GQL statement
    /// executor consults this to reject a write statement in a read-only transaction.
    #[inline]
    pub fn tx_read_only(&self) -> bool {
        self.tx_read_only
    }

    /// Set/clear the active transaction's READ ONLY access mode (see [`Graph::tx_read_only`]).
    #[inline]
    pub fn set_tx_read_only(&mut self, read_only: bool) {
        self.tx_read_only = read_only;
    }

    /// The configured operator-chain ceiling (see [`Graph::max_operator_chain`]).
    #[inline]
    pub fn max_operator_chain(&self) -> usize {
        self.max_operator_chain
    }

    /// Set the operator-chain ceiling (the native `maxOperatorChain` graph option).
    #[inline]
    pub fn set_max_operator_chain(&mut self, n: usize) {
        self.max_operator_chain = n;
    }

    /// Open a transaction frame. Nesting increments depth; the outermost frame
    /// owns commit/rollback (flat, savepoint-less), matching the TS core.
    pub fn begin_tx(&mut self) {
        self.tx_depth += 1;
    }

    /// Close the current frame. An inner commit just decrements depth. The
    /// outermost commit runs the deferred constraint checks against the fully
    /// staged graph — on failure it rolls the whole transaction back via the undo
    /// log and returns the failure — then discards the undo/touched state.
    pub fn commit_tx(&mut self) -> Result<(), TxCommitError> {
        if self.tx_depth == 0 {
            return Err(TxCommitError::NoTx);
        }
        self.tx_depth -= 1;
        if self.tx_depth > 0 {
            return Ok(()); // an inner commit — the outermost frame finalizes
        }
        if let Err(e) = self.run_deferred_checks() {
            self.apply_undo_and_reset();
            return Err(e);
        }
        // Graph-level invariants (cross-write assertions): run ONCE against the
        // fully-staged graph, AFTER the per-element deferred checks, but only if
        // this transaction actually wrote something — a pure-read commit skips
        // them (no spurious cost/throw). The undo log is non-empty iff a write was
        // recorded during the frame. On failure, roll the whole transaction back.
        if !self.tx_undo.is_empty() {
            if let Err(e) = self.check_invariants() {
                self.apply_undo_and_reset();
                return Err(TxCommitError::Invariant(e));
            }
        }
        // Snapshot the touched vertices so a caller can derive this write's
        // value-scope (`last_write_scope`) after the transaction closes. Only a
        // write leaves a non-empty undo log — a pure-read commit clears the snapshot.
        if self.tx_undo.is_empty() {
            self.last_touched.clear();
        } else {
            self.last_touched.clone_from(&self.tx_touched);
        }
        self.tx_undo.clear();
        self.tx_touched.clear();
        self.tx_touched_edges.clear();
        Ok(())
    }

    /// Roll the current transaction back: replay the undo log in reverse, discard
    /// the touched set. A no-op if no transaction is open. Idempotent.
    pub fn rollback_tx(&mut self) {
        if self.tx_depth == 0 {
            return;
        }
        self.apply_undo_and_reset();
    }

    /// Record an inverse op to replay on rollback (no-op outside a transaction or
    /// during an undo replay).
    #[inline]
    fn record_undo(&mut self, inverse: Undo) {
        if self.tx_active() {
            self.tx_undo.push(inverse);
        }
    }

    /// Note a vertex whose built-in constraints must be re-checked at commit. The
    /// per-write gates (in the GQL eval layer) call this instead of throwing
    /// immediately while a transaction is open, so an intermediate state — a node
    /// added before its mandatory property, two rows that momentarily collide —
    /// doesn't trip a constraint the final state satisfies.
    #[inline]
    pub fn tx_note_touched(&mut self, vi: u32) {
        if self.tx_active() {
            self.tx_touched.push(vi);
        }
    }

    /// Note an edge whose built-in edge constraints must be re-checked at commit —
    /// the edge analogue of [`Graph::tx_note_touched`] (R-TX deferral for edges).
    #[inline]
    pub fn tx_note_touched_edge(&mut self, ei: u32) {
        if self.tx_active() {
            self.tx_touched_edges.push(ei);
        }
    }

    /// The current undo-log depth, for [`Graph::rollback_statement`]. Taken before
    /// a statement's frame opens so its writes can be undone without touching the
    /// writes an enclosing transaction already staged.
    #[inline]
    pub fn tx_undo_mark(&self) -> usize {
        self.tx_undo.len()
    }

    /// Roll back only the writes recorded since `mark`, closing ONE frame and
    /// leaving any enclosing transaction **open and usable**.
    ///
    /// This is what a failing *statement* inside an explicit transaction needs:
    /// per-statement atomicity (a faulting statement leaves no trace) without
    /// destroying the surrounding transaction. [`Graph::rollback_tx`] cannot serve
    /// here — it resets `tx_depth` to 0 unconditionally, so an application that
    /// caught a statement error (probing an optional feature, say) would silently
    /// fall out of its transaction and auto-commit every subsequent write, and the
    /// later `throw` that should roll everything back would find no frame open.
    ///
    /// The touched sets are deliberately not trimmed: a stale entry is harmless
    /// (`run_deferred_checks` skips anything no longer live) and trimming would
    /// need a parallel mark per set.
    pub fn rollback_statement(&mut self, mark: usize) {
        if self.tx_depth == 0 {
            return;
        }
        self.applying_undo = true;
        while self.tx_undo.len() > mark {
            if let Some(u) = self.tx_undo.pop() {
                self.apply_one_undo(u);
            }
        }
        self.applying_undo = false;
        self.tx_depth -= 1;
        if self.tx_depth == 0 {
            self.tx_undo.clear();
            self.tx_touched.clear();
            self.tx_touched_edges.clear();
        }
    }

    /// Replay the undo log newest-first and reset all transaction state to closed.
    fn apply_undo_and_reset(&mut self) {
        self.applying_undo = true;
        let undo = std::mem::take(&mut self.tx_undo);
        for u in undo.into_iter().rev() {
            self.apply_one_undo(u);
        }
        self.applying_undo = false;
        self.tx_depth = 0;
        self.tx_undo.clear();
        self.tx_touched.clear();
        self.tx_touched_edges.clear();
    }

    /// Apply a single inverse op. Runs with `applying_undo == true`, so the
    /// mutation methods it calls neither re-record undo nor re-note touched
    /// vertices — they only restore known-good state and keep the indexes current.
    fn apply_one_undo(&mut self, u: Undo) {
        match u {
            Undo::InsertVertex(vi) => {
                let _ = self.remove_vertex(vi, true);
            }
            Undo::InsertEdge(ei) => self.remove_edge(ei),
            Undo::VProp(vi, key, Some(v)) => self.set_vertex_prop(vi, &key, v),
            Undo::VProp(vi, key, None) => self.remove_vertex_prop(vi, &key),
            Undo::EProp(ei, key, Some(v)) => self.set_edge_prop(ei, &key, v),
            Undo::EProp(ei, key, None) => self.remove_edge_prop(ei, &key),
            Undo::VLabelAdd(vi, name) => self.remove_vertex_label(vi, &name),
            Undo::VLabelRemove(vi, name) => self.add_vertex_label(vi, &name),
            Undo::EType(ei, name) => self.add_edge_label(ei, &name),
            Undo::DeleteVertex { vi, labels } => self.untombstone_vertex(vi, &labels),
            Undo::DeleteEdge { ei, eid } => self.untombstone_edge(ei, eid),
        }
    }

    /// Re-run the built-in vertex constraints (required / type / unique) against
    /// every vertex touched during the transaction, now that all writes are
    /// staged. A vertex added then removed within the transaction is skipped.
    fn run_deferred_checks(&self) -> Result<(), TxCommitError> {
        // `tx_touched` collects one entry PER write (a K-clause `SET` on a vertex
        // pushes it K times), and each per-element check below hydrates the whole
        // property row — O(row width). So a naive pass is O(touches × width), i.e.
        // quadratic in the number of properties a wide `SET` writes. Two guards keep
        // it linear without changing observable behaviour:
        //   1. Skip a side entirely when no constraint of a kind it checks is
        //      declared — the loop body could never return `Err`, so skipping it is
        //      behaviour-identical (and the common no-constraint write pays nothing).
        //   2. De-duplicate touched ids in FIRST-SEEN order — a vertex only needs
        //      rechecking once, and keeping first-seen order means the *first*
        //      violation encountered (hence the returned error) is unchanged.
        // Validators live in `v_validators` keyed by label OR edge type, so both the
        // vertex and edge passes gate on it.
        let check_vertices = !self.v_required.is_empty()
            || !self.v_type.is_empty()
            || !self.v_unique.is_empty()
            || !self.v_cardinality.is_empty()
            || !self.v_validators.is_empty();
        if check_vertices {
            let mut seen = HashSet::with_capacity(self.tx_touched.len());
            for &vi in &self.tx_touched {
                if !seen.insert(vi) {
                    continue; // already checked this vertex in this commit
                }
                if !self.is_vertex_live(vi) {
                    continue; // added then removed within the transaction — nothing to check
                }
                let labels: Vec<String> = self.vlabels[vi as usize]
                    .iter()
                    .map(|&l| self.labels.text(l).to_string())
                    .collect();
                let props = self.vertex_props(vi);
                if self.missing_required(&labels, &props).is_some() {
                    return Err(TxCommitError::Required);
                }
                if self.type_violation(&labels, &props).is_some() {
                    return Err(TxCommitError::Type);
                }
                if self.unique_conflict(&labels, &props, Some(vi)).is_some() {
                    return Err(TxCommitError::Unique);
                }
                // Cardinality: a vertex is touched when added OR when an incident edge
                // is added/removed (either endpoint's degree changed). This commit is
                // where BOTH bounds land — max (also caught eagerly for a direct
                // addEdge on the TS side) and min (commit-time only, since a single
                // write can't satisfy a positive lower bound).
                if self.cardinality_violation(vi) {
                    return Err(TxCommitError::Cardinality);
                }
                // Custom validators (a definite-false predicate, or an evaluation fault
                // like an unknown function) — surfaced with their own carried error.
                if let Err(e) = self.check_validators_vertex(vi) {
                    return Err(TxCommitError::Validator(e));
                }
            }
        }
        // Edge constraints: re-check every edge touched during the transaction
        // against the fully-staged graph (edge analogue of the vertex loop above).
        let check_edges = !self.e_required.is_empty()
            || !self.e_type_constraints.is_empty()
            || !self.e_unique.is_empty()
            || !self.v_validators.is_empty();
        if check_edges {
            let mut seen = HashSet::with_capacity(self.tx_touched_edges.len());
            for &ei in &self.tx_touched_edges {
                if !seen.insert(ei) {
                    continue; // already checked this edge in this commit
                }
                if !self.is_edge_live(ei) {
                    continue; // added then removed within the transaction — nothing to check
                }
                let etypes = self.edge_type_names(ei);
                let props = self.edge_props_of(ei);
                if self.edge_missing_required(&etypes, &props).is_some() {
                    return Err(TxCommitError::Required);
                }
                if self.edge_type_violation(&etypes, &props).is_some() {
                    return Err(TxCommitError::Type);
                }
                if self
                    .edge_unique_conflict(&etypes, &props, Some(ei))
                    .is_some()
                {
                    return Err(TxCommitError::Unique);
                }
                if let Err(e) = self.check_validators_edge(ei) {
                    return Err(TxCommitError::Validator(e));
                }
            }
        }
        Ok(())
    }

    /// A live vertex's present properties as `(key, value)` pairs — the shape the
    /// constraint predicates consume. A stored null is present (and included).
    fn vertex_props(&self, vi: u32) -> Vec<(String, Value)> {
        let i = vi as usize;
        let mut out = Vec::new();
        for kid in 0..self.props.cols.len() as u32 {
            if self.props.is_present_id(i, kid) {
                let key = self.props.keys.text(kid).to_string();
                let val = self.props.value_id(i, kid, &self.strs);
                out.push((key, val));
            }
        }
        out
    }

    /// Reverse a vertex delete: un-tombstone the slot in place (its columns were
    /// never cleared on delete, so property values survive) and rebuild its label
    /// membership + property indexes. Adjacency is repopulated by the incident
    /// edges' own `DeleteEdge` inverses (replayed after this one).
    fn untombstone_vertex(&mut self, vi: u32, labels: &[u32]) {
        let i = vi as usize;
        if self.is_vertex_live(vi) {
            return;
        }
        self.v_live[i] = true;
        self.live_n += 1;
        self.vlabels[i] = labels.to_vec();
        for &lid in labels {
            self.by_label.entry(lid).or_default().push(vi);
        }
        if !self.vidx.is_empty() {
            for key in self.vidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.props.value(i, &key, &self.strs);
                idx_apply(&mut self.vidx, &key, vi, &val, true);
            }
        }
        self.bump();
        let mut names: Vec<String> = labels
            .iter()
            .map(|&l| self.labels.text(l).to_string())
            .collect();
        for kid in 0..self.props.cols.len() as u32 {
            if self.props.is_present_id(i, kid) {
                names.push(self.props.keys.text(kid).to_string());
            }
        }
        for name in names {
            self.touch(&name);
        }
    }

    /// Reverse an edge delete: un-tombstone it in place and restore its type
    /// bucket, both endpoints' adjacency, property indexes, and external-id overlay.
    fn untombstone_edge(&mut self, ei: u32, eid: Option<Arc<str>>) {
        let i = ei as usize;
        if self.is_edge_live(ei) {
            return;
        }
        self.e_live[i] = true;
        self.live_e += 1;
        let tid = self.e_type[i];
        let (src, dst) = (self.e_src[i], self.e_dst[i]);
        self.by_etype.entry(tid).or_default().push(ei);
        self.out[src as usize].push(Adj {
            eidx: ei,
            nbr: dst,
            etype: tid,
        });
        self.in_[dst as usize].push(Adj {
            eidx: ei,
            nbr: src,
            etype: tid,
        });
        if !self.eidx.is_empty() {
            for key in self.eidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.edge_props.value(i, &key, &self.strs);
                idx_apply(&mut self.eidx, &key, ei, &val, true);
            }
        }
        if let Some(arc) = eid {
            self.eid_fwd.insert(ei, arc.clone());
            self.eid_rev.insert(arc, ei);
        }
        self.bump();
        let mut names: Vec<String> = vec![self.etype.text(tid).to_string()];
        for kid in 0..self.edge_props.cols.len() as u32 {
            if self.edge_props.is_present_id(i, kid) {
                names.push(self.edge_props.keys.text(kid).to_string());
            }
        }
        for name in names {
            self.touch(&name);
        }
    }

    // --- mutation ----------------------------------------------------------

    fn fresh_id(&mut self) -> String {
        loop {
            let id = format!("_n{}", self.synth);
            self.synth += 1;
            if self.vid.get(&id).is_none() {
                return id;
            }
        }
    }

    /// Add a vertex with the given labels and properties; returns its index.
    /// Reject a graph holding a malformed label / edge type / property key (see
    /// [`validate_label`] / [`validate_prop_key`]). One cheap pass over the
    /// interned name dictionaries (distinct names, not per-element). Called at
    /// the codec ingestion boundary so loaded data can't smuggle in a name that
    /// won't round-trip through every codec.
    pub fn validate_wellformed(&self) -> CodeResult<()> {
        for name in self.labels.strings.iter().chain(self.etype.strings.iter()) {
            validate_label(name)?;
        }
        for name in self
            .props
            .keys
            .strings
            .iter()
            .chain(self.edge_props.keys.strings.iter())
        {
            validate_prop_key(name)?;
        }
        Ok(())
    }

    pub fn add_vertex(&mut self, labels: &[String], props: Vec<(String, Value)>) -> u32 {
        let id = self.fresh_id();
        self.add_vertex_with_id(&id, labels, props)
    }

    /// Append a vertex carrying an **explicit** external id (vs `add_vertex`,
    /// which mints one). The id must be fresh — a caller that might collide
    /// checks `vid.get(id)` first (bulk append / merge does). The building block
    /// for id-preserving bulk ingest into a live graph.
    pub fn add_vertex_with_id(
        &mut self,
        id: &str,
        labels: &[String],
        props: Vec<(String, Value)>,
    ) -> u32 {
        let vi = self.vid.intern(id);
        debug_assert_eq!(vi as usize, self.n, "add_vertex_with_id expects a fresh id");
        self.v_live.push(true);
        self.live_n += 1;
        let lids: Vec<u32> = labels.iter().map(|l| self.labels.intern(l)).collect();
        for &lid in &lids {
            self.by_label.entry(lid).or_default().push(vi);
        }
        self.vlabels.push(lids);
        self.out.push(Vec::new());
        self.in_.push(Vec::new());
        self.props.push_element();
        for (k, v) in props {
            if self.any_vidx_rooted_at(&k) {
                idx_apply(&mut self.vidx, &k, vi, &v, true);
            }
            self.touch(&k);
            self.props.set_value(vi as usize, &k, v, &mut self.strs);
        }
        self.n += 1;
        // Topology change: drop the CSR snapshot, bump the global version and the
        // new vertex's labels.
        self.invalidate_csr();
        self.bump();
        for l in labels {
            self.touch(l);
        }
        // Undo of an insert = tombstone the slot (detach removes any edges added
        // to it later — but on reverse replay those are already undone).
        self.record_undo(Undo::InsertVertex(vi));
        vi
    }

    /// Add an edge `from -> to` of `etype` with properties; returns its index.
    pub fn add_edge(
        &mut self,
        from: u32,
        to: u32,
        etype: &str,
        props: Vec<(String, Value)>,
    ) -> u32 {
        let ei = self.e_src.len() as u32;
        let tid = self.etype.intern(etype);
        self.e_src.push(from);
        self.e_dst.push(to);
        self.e_type.push(tid);
        self.by_etype.entry(tid).or_default().push(ei);
        self.e_live.push(true);
        self.live_e += 1;
        self.out[from as usize].push(Adj {
            eidx: ei,
            nbr: to,
            etype: tid,
        });
        self.in_[to as usize].push(Adj {
            eidx: ei,
            nbr: from,
            etype: tid,
        });
        self.edge_props.push_element();
        for (k, v) in props {
            if self.any_eidx_rooted_at(&k) {
                idx_apply(&mut self.eidx, &k, ei, &v, true);
            }
            self.touch(&k);
            self.edge_props
                .set_value(ei as usize, &k, v, &mut self.strs);
        }
        // Topology change: drop the CSR snapshot, bump the global version and type.
        self.invalidate_csr();
        self.bump();
        self.touch(etype);
        self.record_undo(Undo::InsertEdge(ei));
        // Both endpoints' degree changed — note them for the commit-time
        // cardinality recheck (no-op unless inside a transaction with a
        // cardinality constraint declared).
        self.cardinality_note_endpoints(ei);
        ei
    }

    /// Whether vertex `vi`'s `id` property IS its identity — i.e. a string `id`
    /// equal to the external id (as set by `INSERT (:P {id: 'x'})`). A numeric or
    /// absent `id`, or an external id that diverges from it, is an ordinary
    /// property and remains SET-able; a matching string `id` is fixed.
    pub fn vertex_id_is_identity(&self, vi: u32) -> bool {
        matches!(
            self.props.value(vi as usize, "id", &self.strs),
            Value::Str(s) if s.as_ref() == self.vid.text(vi)
        )
    }

    /// The edge analogue of [`Graph::vertex_id_is_identity`]: whether edge `ei`'s
    /// `id` property IS its external id (a string `id` set at INSERT), and so fixed.
    pub fn edge_id_is_identity(&self, ei: u32) -> bool {
        matches!(
            self.edge_props.value(ei as usize, "id", &self.strs),
            Value::Str(s) if s.as_ref() == self.edge_id(ei).as_ref()
        )
    }

    pub fn set_vertex_prop(&mut self, vi: u32, key: &str, v: Value) {
        if self.tx_active() {
            let prior = if self.props.is_present(vi as usize, key) {
                Some(self.props.value(vi as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::VProp(vi, key.to_string(), prior));
        }
        if self.any_vidx_rooted_at(key) {
            let old = self.props.value(vi as usize, key, &self.strs);
            idx_apply(&mut self.vidx, key, vi, &old, false);
        }
        self.props.set_value(vi as usize, key, v, &mut self.strs);
        if self.any_vidx_rooted_at(key) {
            let new = self.props.value(vi as usize, key, &self.strs);
            idx_apply(&mut self.vidx, key, vi, &new, true);
        }
        // Value change: bump only this key (not the element's labels), so a
        // label-only/topology query isn't invalidated by an unrelated edit.
        self.bump();
        self.touch(key);
    }
    pub fn remove_vertex_prop(&mut self, vi: u32, key: &str) {
        if self.tx_active() {
            let prior = if self.props.is_present(vi as usize, key) {
                Some(self.props.value(vi as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::VProp(vi, key.to_string(), prior));
        }
        if self.any_vidx_rooted_at(key) {
            let old = self.props.value(vi as usize, key, &self.strs);
            idx_apply(&mut self.vidx, key, vi, &old, false);
        }
        self.props.remove_value(vi as usize, key);
        self.bump();
        self.touch(key);
    }
    pub fn set_edge_prop(&mut self, ei: u32, key: &str, v: Value) {
        if self.tx_active() {
            let prior = if self.edge_props.is_present(ei as usize, key) {
                Some(self.edge_props.value(ei as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::EProp(ei, key.to_string(), prior));
        }
        if self.any_eidx_rooted_at(key) {
            let old = self.edge_props.value(ei as usize, key, &self.strs);
            idx_apply(&mut self.eidx, key, ei, &old, false);
        }
        self.edge_props
            .set_value(ei as usize, key, v, &mut self.strs);
        if self.any_eidx_rooted_at(key) {
            let new = self.edge_props.value(ei as usize, key, &self.strs);
            idx_apply(&mut self.eidx, key, ei, &new, true);
        }
        self.bump();
        self.touch(key);
    }
    pub fn remove_edge_prop(&mut self, ei: u32, key: &str) {
        if self.tx_active() {
            let prior = if self.edge_props.is_present(ei as usize, key) {
                Some(self.edge_props.value(ei as usize, key, &self.strs))
            } else {
                None
            };
            self.record_undo(Undo::EProp(ei, key.to_string(), prior));
        }
        if self.any_eidx_rooted_at(key) {
            let old = self.edge_props.value(ei as usize, key, &self.strs);
            idx_apply(&mut self.eidx, key, ei, &old, false);
        }
        self.edge_props.remove_value(ei as usize, key);
        self.bump();
        self.touch(key);
    }

    pub fn add_vertex_label(&mut self, vi: u32, name: &str) {
        let lid = self.labels.intern(name);
        if !self.vlabels[vi as usize].contains(&lid) {
            self.vlabels[vi as usize].push(lid);
            self.by_label.entry(lid).or_default().push(vi);
            self.bump();
            self.touch(name);
            self.record_undo(Undo::VLabelAdd(vi, name.to_string()));
        }
    }
    pub fn remove_vertex_label(&mut self, vi: u32, name: &str) {
        if let Some(lid) = self.labels.get(name) {
            let had = self.vlabels[vi as usize].contains(&lid);
            self.vlabels[vi as usize].retain(|&x| x != lid);
            if let Some(bucket) = self.by_label.get_mut(&lid) {
                bucket.retain(|&x| x != vi);
            }
            self.bump();
            self.touch(name);
            if had {
                self.record_undo(Undo::VLabelRemove(vi, name.to_string()));
            }
        }
    }

    /// An edge carries a single type; relabelling replaces it (last wins).
    pub fn add_edge_label(&mut self, ei: u32, name: &str) {
        let tid = self.etype.intern(name);
        let i = ei as usize;
        // Move the edge between type buckets when its type actually changes.
        let old = self.e_type[i];
        // Capture the prior type name (for the rollback inverse) before it changes.
        if old != tid && self.tx_active() {
            let old_name = self.etype.text(old).to_string();
            self.record_undo(Undo::EType(ei, old_name));
        }
        if old != tid {
            if let Some(bucket) = self.by_etype.get_mut(&old) {
                bucket.retain(|&x| x != ei);
            }
            if self.is_edge_live(ei) {
                self.by_etype.entry(tid).or_default().push(ei);
            }
        }
        self.e_type[i] = tid;
        let (src, dst) = (self.e_src[i] as usize, self.e_dst[i] as usize);
        for a in self.out[src].iter_mut().filter(|a| a.eidx == ei) {
            a.etype = tid;
        }
        for a in self.in_[dst].iter_mut().filter(|a| a.eidx == ei) {
            a.etype = tid;
        }
        if old != tid {
            // Both the old and new type's membership changed; the etype stored in
            // the adjacency slots changed too, so the CSR snapshot is stale.
            let old_name = self.etype.text(old).to_string();
            self.invalidate_csr();
            self.bump();
            self.touch(&old_name);
            self.touch(name);
        }
    }
    pub fn remove_edge_label(&mut self, ei: u32, _name: &str) {
        // Single-type edges: removing the label clears the type to empty.
        self.add_edge_label(ei, "");
    }

    /// Delete an edge (tombstone + unlink from both endpoints' adjacency).
    pub fn remove_edge(&mut self, ei: u32) {
        let i = ei as usize;
        if !self.is_edge_live(ei) {
            return;
        }
        // Both endpoints' degree will drop — note them for the commit-time
        // cardinality recheck (min may now be unmet). Endpoints read from the
        // still-intact e_src/e_dst; no-op outside a transaction / rollback replay.
        self.cardinality_note_endpoints(ei);
        // Record the inverse (un-tombstone) before tombstoning: capture any
        // external-id overlay, which the removal below drops.
        if self.tx_active() {
            let eid = self.eid_fwd.get(&ei).cloned();
            self.record_undo(Undo::DeleteEdge { ei, eid });
        }
        // Drop the edge from every edge property index before tombstoning.
        if !self.eidx.is_empty() {
            for key in self.eidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.edge_props.value(i, &key, &self.strs);
                idx_apply(&mut self.eidx, &key, ei, &val, false);
            }
        }
        // Invalidate the edge's type and every property key it carried.
        let mut touched: Vec<String> = vec![self.etype.text(self.e_type[i]).to_string()];
        for kid in 0..self.edge_props.cols.len() as u32 {
            // Presence, not value: a stored-null key is present and its epoch
            // must still be bumped on delete.
            if self.edge_props.is_present_id(i, kid) {
                touched.push(self.edge_props.keys.text(kid).to_string());
            }
        }
        self.e_live[i] = false;
        self.live_e -= 1;
        if let Some(bucket) = self.by_etype.get_mut(&self.e_type[i]) {
            bucket.retain(|&x| x != ei);
        }
        // Drop any external id overlay for this edge.
        if let Some(old) = self.eid_fwd.remove(&ei) {
            self.eid_rev.remove(&old);
        }
        let (src, dst) = (self.e_src[i] as usize, self.e_dst[i] as usize);
        self.out[src].retain(|a| a.eidx != ei);
        self.in_[dst].retain(|a| a.eidx != ei);
        self.invalidate_csr();
        self.bump();
        for name in touched {
            self.touch(&name);
        }
    }

    /// Delete a vertex. Without `detach`, a vertex that still has edges is an
    /// error (ISO/Cypher semantics); with `detach`, incident edges go first.
    pub fn remove_vertex(&mut self, vi: u32, detach: bool) -> CodeResult<()> {
        let i = vi as usize;
        if !self.is_vertex_live(vi) {
            return Ok(());
        }
        let incident: Vec<u32> = self.out[i]
            .iter()
            .chain(self.in_[i].iter())
            .map(|a| a.eidx)
            .collect();
        if !detach && !incident.is_empty() {
            return Err(CodeError::new(
                ErrorCode::InvalidGraphOp,
                "cannot delete a vertex that still has relationships; use DETACH DELETE",
            ));
        }
        for ei in incident {
            self.remove_edge(ei);
        }
        // Invalidate the vertex's labels and every property key it carried
        // (gathered before the columns/labels are cleared below).
        let mut touched: Vec<String> = self.vlabels[i]
            .iter()
            .map(|&l| self.labels.text(l).to_string())
            .collect();
        for kid in 0..self.props.cols.len() as u32 {
            // Presence, not value (stored null is present) — see remove_edge.
            if self.props.is_present_id(i, kid) {
                touched.push(self.props.keys.text(kid).to_string());
            }
        }
        for lid in self.vlabels[i].clone() {
            if let Some(bucket) = self.by_label.get_mut(&lid) {
                bucket.retain(|&x| x != vi);
            }
        }
        // Drop the vertex from every vertex property index.
        if !self.vidx.is_empty() {
            for key in self.vidx.keys().cloned().collect::<Vec<_>>() {
                let val = self.props.value(i, &key, &self.strs);
                idx_apply(&mut self.vidx, &key, vi, &val, false);
            }
        }
        // Capture the labels for the rollback inverse before clearing them (the
        // columns are left intact, so property values survive the tombstone).
        let undo_labels: Vec<u32> = if self.tx_active() {
            self.vlabels[i].clone()
        } else {
            Vec::new()
        };
        self.vlabels[i].clear();
        self.out[i].clear();
        self.in_[i].clear();
        self.v_live[i] = false;
        self.live_n -= 1;
        self.invalidate_csr();
        self.bump();
        for name in touched {
            self.touch(&name);
        }
        // Recorded last (after the cascade's per-edge `DeleteEdge` inverses), so a
        // reverse replay un-tombstones the vertex first, then re-adds its edges.
        self.record_undo(Undo::DeleteVertex {
            vi,
            labels: undo_labels,
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Column construction / promotion helpers (shared by build + mutation).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Num,
    Str,
    Bool,
    /// A homogeneous temporal column, keyed by which temporal variant. Two
    /// different temporal sub-kinds in one key promote to `Mixed`, like any other
    /// type disagreement.
    Temporal(TemporalKind),
    /// A homogeneous fixed-dimension numeric-vector column ([`Column::Vec`]), keyed
    /// by the vector length. Two different lengths in one key promote to `Mixed`.
    Vec(usize),
    Mixed,
}

fn value_kind(v: &Value) -> Option<Kind> {
    match v {
        Value::Num(_) => Some(Kind::Num),
        Value::Str(_) => Some(Kind::Str),
        Value::Bool(_) => Some(Kind::Bool),
        Value::Null => None, // nulls don't determine a column's type
        // A temporal value gets a packed, de-boxed column keyed by its kind (see
        // [`TemporalCol`]); a key mixing temporal sub-kinds falls back to `Mixed`.
        Value::Temporal(t) => Some(Kind::Temporal(t.kind())),
        // An all-numeric fixed-length list packs into a de-boxed `Vec` column; a
        // mixed / variable-length / non-numeric list stays `Mixed`.
        Value::List(items) => Some(numeric_vec_dim(items).map_or(Kind::Mixed, Kind::Vec)),
        // A map/record is inherently variable-shape, so it lives boxed in a
        // `Mixed` column (like a non-numeric list) — never a de-boxed SoA column.
        Value::Map(_) => Some(Kind::Mixed),
    }
}

/// Does a value contain a map/record anywhere (itself, or nested in a list)?
fn value_contains_map(v: &Value) -> bool {
    match v {
        Value::Map(_) => true,
        Value::List(items) => items.iter().any(value_contains_map),
        _ => false,
    }
}

/// Normalize a value to its canonical stored form: every map (at any depth) has
/// its fields sorted by key, with duplicate keys collapsed last-wins. Sorted keys
/// are the invariant the whole engine relies on — equality is a slice compare,
/// serialization is a straight emit, and the sync-diff `JSON.stringify` byte-
/// equality holds. Only maps/lists are rebuilt; a scalar is a plain clone, so the
/// typed-column hot paths (which never reach this) pay nothing.
///
/// Map field NAMES are **interned** through `strs`, so 50k `{city, tier, …}`
/// records share ONE `Arc<str>` per distinct field instead of allocating a fresh
/// key per record — the codec-decode win (each decoder allocates keys per map).
fn canonical_value(v: &Value, strs: &mut Dict) -> Value {
    match v {
        Value::Map(pairs) => {
            let mut out: Vec<(Arc<str>, Value)> = Vec::with_capacity(pairs.len());
            for (k, val) in pairs {
                let id = strs.intern(k); // shared, deduped field name
                let key = strs.arc(id);
                let cv = canonical_value(val, strs);
                match out.binary_search_by(|(ek, _)| ek.as_ref().cmp(k.as_ref())) {
                    // Duplicate field name → last write wins (JS-object / SQL-ish).
                    Ok(i) => out[i].1 = cv,
                    Err(i) => out.insert(i, (key, cv)),
                }
            }
            Value::Map(out)
        }
        Value::List(items) => Value::List(items.iter().map(|x| canonical_value(x, strs)).collect()),
        other => other.clone(),
    }
}

/// A fresh, all-absent column of `len` slots for a resolved [`Kind`].
fn empty_col_for_kind(kind: Option<Kind>, len: usize) -> Column {
    match kind {
        Some(Kind::Num) => Column::Num {
            data: vec![f64::NAN; len],
            present: BitSet::zeros(len),
        },
        Some(Kind::Str) => Column::Str {
            data: vec![u32::MAX; len],
            present: BitSet::zeros(len),
        },
        Some(Kind::Bool) => Column::Bool {
            data: vec![false; len],
            present: BitSet::zeros(len),
        },
        Some(Kind::Temporal(tk)) => Column::Temporal {
            data: TemporalCol::with_len(tk, len),
            present: BitSet::zeros(len),
        },
        Some(Kind::Vec(dim)) => Column::Vec {
            data: vec![0.0; len * dim],
            dim,
            present: BitSet::zeros(len),
        },
        _ => Column::Mixed {
            data: vec![None; len],
        },
    }
}

/// A fresh, all-absent column sized to `len`, typed for a (non-null) value.
fn empty_col_for(v: &Value, len: usize) -> Column {
    empty_col_for_kind(value_kind(v), len)
}

/// Set element `idx` in a column; returns `false` if the value's type doesn't
/// fit the column (the caller then promotes to `Mixed`).
fn col_set(col: &mut Column, idx: usize, v: &Value, strs: &mut Dict) -> bool {
    match (col, v) {
        (Column::Num { data, present }, Value::Num(x)) => {
            data[idx] = *x;
            present.set(idx);
            true
        }
        (Column::Str { data, present }, Value::Str(s)) => {
            data[idx] = strs.intern(s);
            present.set(idx);
            true
        }
        (Column::Bool { data, present }, Value::Bool(b)) => {
            data[idx] = *b;
            present.set(idx);
            true
        }
        (Column::Temporal { data, present }, Value::Temporal(t)) => {
            // A different temporal sub-kind doesn't fit → `false` promotes to Mixed.
            if data.set(idx, t) {
                present.set(idx);
                true
            } else {
                false
            }
        }
        (Column::Vec { data, dim, present }, Value::List(items)) => {
            // Fits only an all-numeric list of the column's dimension; anything else
            // (different length, a non-number) returns `false` → promote to Mixed.
            if numeric_vec_dim(items) != Some(*dim) {
                return false;
            }
            for (j, it) in items.iter().enumerate() {
                if let Value::Num(x) = it {
                    data[idx * *dim + j] = *x;
                }
            }
            present.set(idx);
            true
        }
        (Column::Mixed { data }, val) => {
            // Canonicalize on the way in so a stored map's keys are always sorted
            // (the boxed path only — scalars never reach here).
            data[idx] = Some(canonical_value(val, strs));
            true
        }
        (
            Column::Record {
                present,
                nulls,
                field_names,
                fields,
                field_nulls,
                escaped,
            },
            val,
        ) => {
            // Clear every field slot (typed sub-column AND its stored-null bit) at idx.
            let clear_fields = |fields: &mut [Column], field_nulls: &mut [BitSet]| {
                for f in fields.iter_mut() {
                    col_clear(f, idx);
                }
                for fnull in field_nulls.iter_mut() {
                    fnull.clear(idx);
                }
            };
            // The record constraint guarantees a label-matching write conforms, but
            // this column is shared across labels — a value that isn't a conforming
            // map (all keys declared) stays boxed in `escaped`. Canonicalize first so
            // field keys are sorted (binary search) and escaped values are canonical.
            match canonical_value(val, strs) {
                Value::Null => {
                    escaped.remove(&(idx as u32));
                    present.set(idx);
                    nulls.set(idx);
                    clear_fields(fields, field_nulls);
                }
                Value::Map(pairs)
                    if pairs.iter().all(|(k, _)| {
                        field_names.binary_search_by(|n| n.as_ref().cmp(k)).is_ok()
                    }) =>
                {
                    escaped.remove(&(idx as u32));
                    present.set(idx);
                    nulls.clear(idx);
                    for (fi, name) in field_names.iter().enumerate() {
                        match pairs.binary_search_by(|(k, _)| k.as_ref().cmp(name)) {
                            // A present, NON-null field value scatters into its typed
                            // sub-column (promoting only on a genuine type mismatch).
                            Ok(pi) if !matches!(pairs[pi].1, Value::Null) => {
                                field_nulls[fi].clear(idx);
                                let fv = &pairs[pi].1;
                                if !col_set(&mut fields[fi], idx, fv, strs) {
                                    fields[fi] = to_mixed(&fields[fi], strs);
                                    col_set(&mut fields[fi], idx, fv, strs);
                                }
                            }
                            // A present STORED-NULL field: mark its null bit, keep the
                            // sub-column typed (no Mixed promotion — the 1b win).
                            Ok(_) => {
                                col_clear(&mut fields[fi], idx);
                                field_nulls[fi].set(idx);
                            }
                            // A nullable field the value omits → absent in its column.
                            Err(_) => {
                                col_clear(&mut fields[fi], idx);
                                field_nulls[fi].clear(idx);
                            }
                        }
                    }
                }
                other => {
                    present.clear(idx);
                    nulls.clear(idx);
                    clear_fields(fields, field_nulls);
                    escaped.insert(idx as u32, other);
                }
            }
            true
        }
        _ => false,
    }
}

/// Read a single column at element `i` as an owned [`Value`], or `None` if the
/// slot is absent. The read-side inverse of [`col_set`], shared by [`to_mixed`],
/// record synthesis, and [`Properties::field_at`].
fn col_get(col: &Column, i: usize, strs: &Dict) -> Option<Value> {
    match col {
        Column::Num { data, present } if present.get(i) => Some(Value::Num(data[i])),
        Column::Bool { data, present } if present.get(i) => Some(Value::Bool(data[i])),
        Column::Str { data, present } if present.get(i) => Some(Value::Str(strs.arc(data[i]))),
        Column::Temporal { data, present } if present.get(i) => Some(Value::Temporal(data.get(i))),
        Column::Vec { data, dim, present } if present.get(i) => Some(Value::List(
            vec_slice(data, *dim, i)
                .iter()
                .map(|x| Value::Num(*x))
                .collect(),
        )),
        Column::Mixed { data } => data.get(i).cloned().flatten(),
        Column::Record {
            present,
            nulls,
            field_names,
            fields,
            field_nulls,
            escaped,
        } => {
            if let Some(v) = escaped.get(&(i as u32)) {
                Some(v.clone())
            } else if present.get(i) {
                Some(if nulls.get(i) {
                    Value::Null
                } else {
                    record_map(field_names, fields, field_nulls, i, strs)
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Clear (mark absent) element `idx` of a column — the read-side inverse of a set.
fn col_clear(col: &mut Column, idx: usize) {
    match col {
        Column::Num { present, .. }
        | Column::Str { present, .. }
        | Column::Bool { present, .. }
        | Column::Temporal { present, .. }
        | Column::Vec { present, .. } => present.clear(idx),
        Column::Mixed { data } => {
            if idx < data.len() {
                data[idx] = None;
            }
        }
        Column::Record {
            present,
            nulls,
            fields,
            field_nulls,
            escaped,
            ..
        } => {
            escaped.remove(&(idx as u32));
            present.clear(idx);
            nulls.clear(idx);
            for f in fields.iter_mut() {
                col_clear(f, idx);
            }
            for fnull in field_nulls.iter_mut() {
                fnull.clear(idx);
            }
        }
    }
}

/// Synthesize the canonical map for element `i` of a de-boxed record from its
/// present field sub-columns. An absent field is omitted (nullable/optional); a
/// field holding a stored null (its `field_nulls` bit set) is included as `Null`.
/// Field names are sorted, so the result is canonical.
fn record_map(
    names: &[Arc<str>],
    fields: &[Column],
    field_nulls: &[BitSet],
    i: usize,
    strs: &Dict,
) -> Value {
    let pairs: Vec<(Arc<str>, Value)> = names
        .iter()
        .enumerate()
        .filter_map(|(fi, n)| {
            if field_nulls[fi].get(i) {
                Some((n.clone(), Value::Null)) // a present stored-null field
            } else {
                col_get(&fields[fi], i, strs).map(|v| (n.clone(), v))
            }
        })
        .collect();
    Value::Map(pairs)
}

/// Build an empty (all-absent) de-boxed [`Column::Record`] from sorted field
/// defs, sized to `len`. A `RECORD`-typed field is RECURSIVELY de-boxed into a
/// nested `Column::Record` (so `n.meta.geo.lat` lives in its own typed
/// sub-column); a scalar → its typed column; a `list` / `any record` / degenerate
/// 0-field record → boxed `Mixed`.
fn empty_record_column(defs: &[(Arc<str>, TypeSpec, bool)], len: usize) -> Column {
    let field_names: Vec<Arc<str>> = defs.iter().map(|(n, ..)| n.clone()).collect();
    let fields: Vec<Column> = defs
        .iter()
        .map(|(_, t, _)| match t {
            TypeSpec::Record(inner) if !inner.is_empty() => empty_record_column(inner, len),
            _ => empty_col_for_kind(field_kind(t), len),
        })
        .collect();
    let field_nulls: Vec<BitSet> = (0..field_names.len()).map(|_| BitSet::zeros(len)).collect();
    Column::Record {
        present: BitSet::zeros(len),
        nulls: BitSet::zeros(len),
        field_names,
        fields,
        field_nulls,
        escaped: std::collections::HashMap::new(),
    }
}

/// The de-boxed column kind for a declared record field type: a scalar maps to
/// its typed column; a `list` / `any record` / nested record stays boxed here
/// (`Mixed`) — a nested closed record is handled by [`empty_record_column`].
fn field_kind(t: &TypeSpec) -> Option<Kind> {
    match t {
        TypeSpec::Scalar(pt) => Some(match pt {
            PropType::Str => Kind::Str,
            PropType::Num => Kind::Num,
            PropType::Bool => Kind::Bool,
            PropType::Date => Kind::Temporal(TemporalKind::Date),
            PropType::Time => Kind::Temporal(TemporalKind::Time),
            PropType::DateTime => Kind::Temporal(TemporalKind::DateTime),
            PropType::ZonedTime => Kind::Temporal(TemporalKind::ZonedTime),
            PropType::ZonedDateTime => Kind::Temporal(TemporalKind::ZonedDateTime),
            PropType::Duration => Kind::Temporal(TemporalKind::Duration),
            PropType::List => Kind::Mixed,
        }),
        // A nested record OR an open `any record` field stays boxed (`Mixed`).
        TypeSpec::Record(_) | TypeSpec::AnyRecord => Some(Kind::Mixed),
    }
}

/// Materialize any column into a `Mixed` column (loses no values).
fn to_mixed(col: &Column, strs: &Dict) -> Column {
    let len = col.element_len();
    let mut data: Vec<Option<Value>> = Vec::with_capacity(len);
    for i in 0..len {
        let v = match col {
            Column::Num { data, present } if present.get(i) => Some(Value::Num(data[i])),
            Column::Bool { data, present } if present.get(i) => Some(Value::Bool(data[i])),
            Column::Str { data, present } if present.get(i) => Some(Value::Str(strs.arc(data[i]))),
            Column::Temporal { data, present } if present.get(i) => {
                Some(Value::Temporal(data.get(i)))
            }
            Column::Vec { data, dim, present } if present.get(i) => Some(Value::List(
                vec_slice(data, *dim, i)
                    .iter()
                    .map(|x| Value::Num(*x))
                    .collect(),
            )),
            Column::Mixed { data } => data[i].clone(),
            rec @ Column::Record { .. } => col_get(rec, i, strs),
            _ => None,
        };
        data.push(v);
    }
    Column::Mixed { data }
}

// ---------------------------------------------------------------------------
// Builder: accumulate node/edge records, then finalize into the columnar form.
// ---------------------------------------------------------------------------

pub struct NodeRec {
    pub id: String,
    pub labels: Vec<String>,
    pub props: Vec<(String, Value)>,
}

pub struct EdgeRec {
    pub src: String,
    pub dst: String,
    pub etype: String,
    pub props: Vec<(String, Value)>,
    /// Optional external string id. The dense edge index is the edge's canonical
    /// identity; this is an opt-in overlay (set by codecs that carry edge ids) so
    /// a user-assigned id survives a serialization round-trip. `None` ⇒ id-less.
    pub id: Option<String>,
}

#[derive(Default)]
pub struct Builder {
    pub nodes: Vec<NodeRec>,
    pub edges: Vec<EdgeRec>,
}

/// Build a typed property store for `len` elements from `(index, props)` items.
/// A key's column type is inferred from its first non-null value; values that
/// disagree land in `Mixed` (lossless). Shared by the vertex and edge builds.
fn build_props(len: usize, items: &[(usize, &[(String, Value)])], strs: &mut Dict) -> Properties {
    let mut props = Properties {
        keys: Dict::default(),
        cols: Vec::new(),
        len,
    };
    // Infer a kind per key (by dense key id) from its first non-null value.
    let mut kinds: HashMap<u32, Kind> = HashMap::new();
    for (_, item) in items {
        for (k, v) in *item {
            let kid = props.keys.intern(k);
            if let Some(vk) = value_kind(v) {
                kinds
                    .entry(kid)
                    .and_modify(|cur| {
                        if *cur != vk {
                            *cur = Kind::Mixed;
                        }
                    })
                    .or_insert(vk);
            }
        }
    }
    // One column per interned key (dense by id); an all-null key gets an empty Mixed.
    props.cols = (0..props.keys.len() as u32)
        .map(|kid| empty_col_for_kind(kinds.get(&kid).copied(), len))
        .collect();
    for (idx, item) in items {
        for (k, v) in *item {
            // Store every value, `Null` included — a present null promotes the
            // column to `Mixed` (mirrors `set_value`; null is a first-class value).
            let kid = props.keys.get(k).unwrap() as usize;
            let col = &mut props.cols[kid];
            if !col_set(col, *idx, v, strs) {
                *col = to_mixed(col, strs);
                col_set(col, *idx, v, strs);
            }
        }
    }
    props
}

impl Builder {
    /// Like [`finalize`](Self::finalize), but enforces a **declared-nodes**
    /// contract: every edge endpoint must be a declared node. Returns
    /// `MissingVertex` instead of silently fabricating a phantom vertex (the
    /// lenient `finalize` behavior, kept for streaming NDJSON where endpoints are
    /// legitimately created on demand). The JSON document codecs (pg-json,
    /// graphson) use this so a dangling edge is an error, mirroring the TS codecs.
    pub fn finalize_strict(self) -> CodeResult<Graph> {
        let declared: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_str()).collect();
        for e in &self.edges {
            let missing = if !declared.contains(e.src.as_str()) {
                Some(&e.src)
            } else if !declared.contains(e.dst.as_str()) {
                Some(&e.dst)
            } else {
                None
            };
            if let Some(id) = missing {
                return Err(CodeError::new(
                    ErrorCode::MissingVertex,
                    format!(
                        "edge references a non-existent vertex '{id}' (from='{}', to='{}')",
                        e.src, e.dst
                    ),
                ));
            }
        }
        Ok(self.finalize())
    }

    pub fn finalize(self) -> Graph {
        let Self { nodes, edges } = self;
        let mut vid = Dict::default();

        // (1) Dense indices: declared nodes first (in order), then edge endpoints.
        for node in &nodes {
            vid.intern(&node.id);
        }
        for e in &edges {
            vid.intern(&e.src);
            vid.intern(&e.dst);
        }
        let n = vid.len();

        // Duplicate-id semantics, matching the TS core's idempotent add: a node
        // id is **first-wins** (later records with the same id are ignored), and
        // an edge with an already-seen *assigned* id is **dropped** (its endpoints
        // are still interned above, as TS ensures them before the dedup check).
        // Borrowed-`&str` sets keep this allocation-free on the common path.
        let keep_node: Vec<bool> = {
            let mut seen: HashSet<&str> = HashSet::with_capacity(nodes.len());
            nodes.iter().map(|nd| seen.insert(nd.id.as_str())).collect()
        };
        let kept_edges: Vec<&EdgeRec> = {
            let mut seen: HashSet<&str> = HashSet::with_capacity(edges.len());
            edges
                .iter()
                .filter(|e| match &e.id {
                    Some(id) => seen.insert(id.as_str()),
                    None => true, // id-less edges get a unique e{index}; never dup
                })
                .collect()
        };

        // (2) Labels: per-vertex list + inverted (label -> live vertices).
        let mut vlabels: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut labels = Dict::default();
        let mut by_label: HashMap<u32, Vec<u32>> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            if !keep_node[idx] {
                continue; // first-wins: ignore a duplicate node id's labels
            }
            let vi = vid.get(&node.id).unwrap();
            for l in &node.labels {
                let lid = labels.intern(l);
                vlabels[vi as usize].push(lid);
                by_label.entry(lid).or_default().push(vi);
            }
        }

        // (3) Vertex property columns. `strs` is graph-wide, shared with edges.
        let mut strs = Dict::default();
        let node_items: Vec<(usize, &[(String, Value)])> = nodes
            .iter()
            .enumerate()
            .filter(|(idx, _)| keep_node[*idx])
            .map(|(_, nd)| (vid.get(&nd.id).unwrap() as usize, nd.props.as_slice()))
            .collect();
        let props = build_props(n, &node_items, &mut strs);

        // (4) Edges: parallel arrays + per-vertex out/in adjacency.
        let mut etype = Dict::default();
        let e = kept_edges.len();
        let mut e_src = vec![0u32; e];
        let mut e_dst = vec![0u32; e];
        let mut e_type = vec![0u32; e];
        let mut out: Vec<Vec<Adj>> = vec![Vec::new(); n];
        let mut in_: Vec<Vec<Adj>> = vec![Vec::new(); n];
        let mut by_etype: HashMap<u32, Vec<u32>> = HashMap::new();
        // Lazy external-id overlay: only edges that carry an id land here.
        let mut eid_fwd: HashMap<u32, Arc<str>> = HashMap::new();
        let mut eid_rev: HashMap<Arc<str>, u32> = HashMap::new();
        for (i, ed) in kept_edges.iter().enumerate() {
            let s = vid.get(&ed.src).unwrap();
            let d = vid.get(&ed.dst).unwrap();
            let t = etype.intern(&ed.etype);
            e_src[i] = s;
            e_dst[i] = d;
            e_type[i] = t;
            by_etype.entry(t).or_default().push(i as u32);
            out[s as usize].push(Adj {
                eidx: i as u32,
                nbr: d,
                etype: t,
            });
            in_[d as usize].push(Adj {
                eidx: i as u32,
                nbr: s,
                etype: t,
            });
            if let Some(id) = &ed.id {
                let arc: Arc<str> = Arc::from(id.as_str());
                eid_fwd.insert(i as u32, arc.clone());
                eid_rev.insert(arc, i as u32);
            }
        }

        // (5) Edge property columns — same machinery, indexed by edge index.
        let edge_items: Vec<(usize, &[(String, Value)])> = kept_edges
            .iter()
            .enumerate()
            .map(|(i, ed)| (i, ed.props.as_slice()))
            .collect();
        let edge_props = build_props(e, &edge_items, &mut strs);

        Graph {
            n,
            live_n: n,
            v_live: vec![true; n],
            vid,
            labels,
            etype,
            strs,
            vlabels,
            by_label,
            props,
            edge_props,
            e_src,
            e_dst,
            e_type,
            by_etype,
            eid_fwd,
            eid_rev,
            version: 0,
            epochs: HashMap::new(),
            e_live: vec![true; e],
            live_e: e,
            out,
            in_,
            csr: std::sync::OnceLock::new(),
            csr_reads: std::sync::atomic::AtomicU32::new(0),
            synth: 0,
            vidx: HashMap::new(),
            eidx: HashMap::new(),
            v_unique: HashMap::new(),
            v_required: HashMap::new(),
            v_type: HashMap::new(),
            v_record: HashMap::new(),
            v_type_not_null: HashMap::new(),
            e_unique: HashMap::new(),
            e_required: HashMap::new(),
            e_type_constraints: HashMap::new(),
            e_record: HashMap::new(),
            e_type_not_null: HashMap::new(),
            v_cardinality: Vec::new(),
            v_validators: HashMap::new(),
            v_invariants: Vec::new(),
            tx_depth: 0,
            tx_undo: Vec::new(),
            tx_touched: Vec::new(),
            tx_touched_edges: Vec::new(),
            last_touched: Vec::new(),
            applying_undo: false,
            tx_read_only: false,
            max_operator_chain: 10_000,
        }
    }
}

/// A **well-formed label** (node label or edge type): non-empty and free of the
/// `::` sequence. GraphSON joins a node's labels with `::`, so a `::` inside one
/// label is ambiguous/unrepresentable there (and bare GQL can't name it either).
/// An empty label collapses to "no labels" in GraphSON/CSV. Constraining the
/// model to well-formed labels keeps every codec's round-trip unambiguous.
pub fn validate_label(name: &str) -> CodeResult<()> {
    if name.is_empty() {
        return Err(CodeError::new(
            ErrorCode::InvalidValue,
            "a label / edge type must be non-empty",
        ));
    }
    if name.contains("::") {
        return Err(CodeError::new(
            ErrorCode::InvalidValue,
            format!("a label / edge type cannot contain '::' (the GraphSON multi-label separator): {name:?}"),
        ));
    }
    Ok(())
}

/// A **well-formed property key**: non-empty (an empty key has no CSV column
/// header / no `key:value` pg-text form, and is meaningless).
pub fn validate_prop_key(name: &str) -> CodeResult<()> {
    if name.is_empty() {
        return Err(CodeError::new(
            ErrorCode::InvalidValue,
            "a property key must be non-empty",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod wellformed_names {
    //! Labels/edge-types must be non-empty and `::`-free (GraphSON's multi-label
    //! separator); property keys must be non-empty. Enforced at ingestion.
    use super::*;

    #[test]
    fn label_rules() {
        assert!(validate_label("Person").is_ok());
        assert!(validate_label("a:b").is_ok()); // a single colon is fine
        assert!(validate_label("").is_err()); // empty collapses to "no labels"
        assert!(validate_label("a::b").is_err()); // GraphSON multi-label separator
        assert!(validate_label("::").is_err());
    }

    #[test]
    fn key_rules() {
        assert!(validate_prop_key("name").is_ok());
        assert!(validate_prop_key("a::b").is_ok()); // keys are never `::`-joined
        assert!(validate_prop_key("").is_err());
    }
}

#[cfg(test)]
mod vector_column {
    //! The typed fixed-dim numeric-vector column (`Column::Vec`): an all-numeric
    //! fixed-length list packs into a de-boxed f64 column, invisibly to callers
    //! (`value` reconstructs the identical `Value::List`), and a non-conforming
    //! write promotes it to `Mixed` losslessly.
    use super::*;

    fn decode(s: &str) -> Graph {
        crate::ndjson::decode(s).unwrap()
    }

    /// Which column variant backs `key` in the vertex store.
    fn col_name(g: &Graph, key: &str) -> &'static str {
        match g.props.col(key) {
            Some(Column::Vec { .. }) => "vec",
            Some(Column::Mixed { .. }) => "mixed",
            Some(Column::Num { .. }) => "num",
            Some(Column::Record { .. }) => "record",
            _ => "other",
        }
    }

    #[test]
    fn numeric_list_packs_into_a_vec_column_and_reads_back_identically() {
        let g = decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1.5,2.5,3.5]}}
{"type":"node","id":"b","labels":["N"],"properties":{"h":[9,8,7]}}"#,
        );
        assert_eq!(
            col_name(&g, "h"),
            "vec",
            "an all-numeric fixed-len list is a Vec column"
        );
        // `value` reconstructs the exact `Value::List` a caller would have seen.
        let a = g.vid.get("a").unwrap() as usize;
        assert_eq!(
            g.props.value(a, "h", &g.strs),
            Value::List(vec![Value::Num(1.5), Value::Num(2.5), Value::Num(3.5)])
        );
        // Zero-copy slice accessor.
        assert_eq!(g.props.vector(a, "h"), Some(&[1.5, 2.5, 3.5][..]));
        // NDJSON round-trips byte-for-byte (the Vec column encodes like a boxed list).
        let round = crate::ndjson::encode(&g);
        let g2 = crate::ndjson::decode(&round).unwrap();
        assert_eq!(crate::ndjson::encode(&g2), round);
        assert_eq!(col_name(&g2, "h"), "vec");
    }

    #[test]
    fn a_ragged_or_non_numeric_list_stays_mixed() {
        // Different lengths under one key → Mixed.
        let g = decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,2,3]}}
{"type":"node","id":"b","labels":["N"],"properties":{"h":[1,2]}}"#,
        );
        assert_eq!(col_name(&g, "h"), "mixed");
        // A non-numeric element → Mixed.
        let g2 = decode(r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,"x",3]}}"#);
        assert_eq!(col_name(&g2, "h"), "mixed");
        assert!(g2
            .props
            .vector(g2.vid.get("a").unwrap() as usize, "h")
            .is_none());
    }

    #[test]
    fn a_mismatched_set_promotes_the_vec_column_to_mixed_losslessly() {
        let mut g = decode(
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1.0,2.0]}}
{"type":"node","id":"b","labels":["N"],"properties":{"h":[3.0,4.0]}}"#,
        );
        assert_eq!(col_name(&g, "h"), "vec");
        // Overwrite b's vector with a length-3 list → promotes the whole column.
        let b = g.vid.get("b").unwrap();
        g.set_vertex_prop(b, "h", Value::List(vec![Value::Num(5.0); 3]));
        assert_eq!(
            col_name(&g, "h"),
            "mixed",
            "a dim mismatch promotes to Mixed"
        );
        // a's original vector survives the promotion.
        let a = g.vid.get("a").unwrap() as usize;
        assert_eq!(
            g.props.value(a, "h", &g.strs),
            Value::List(vec![Value::Num(1.0), Value::Num(2.0)])
        );
        assert_eq!(
            g.props.value(b as usize, "h", &g.strs),
            Value::List(vec![Value::Num(5.0), Value::Num(5.0), Value::Num(5.0)])
        );
    }

    #[test]
    fn vec_column_uses_less_heap_than_a_boxed_list() {
        // 8 B/f64 contiguous vs a ~40 B Option<Value> slot per element (plus the
        // uncounted heap Vec of boxed Nums that a Mixed list also carries).
        let g = decode(r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1,2,3,4]}}"#);
        match g.props.col("h").unwrap() {
            Column::Vec { data, dim, .. } => {
                assert_eq!(*dim, 4);
                assert_eq!(data.len(), 4); // one element × dim
            }
            _ => panic!("expected a Vec column"),
        }
    }

    #[test]
    fn removing_a_vector_clears_presence_but_a_reset_repopulates() {
        let mut g =
            decode(r#"{"type":"node","id":"a","labels":["N"],"properties":{"h":[1.0,2.0]}}"#);
        let a = g.vid.get("a").unwrap();
        g.remove_vertex_prop(a, "h");
        assert!(!g.props.is_present(a as usize, "h"));
        assert_eq!(g.props.value(a as usize, "h", &g.strs), Value::Null);
        // Re-set with a conforming vector — the column is still a Vec.
        g.set_vertex_prop(a, "h", Value::List(vec![Value::Num(7.0), Value::Num(8.0)]));
        assert_eq!(g.props.vector(a as usize, "h"), Some(&[7.0, 8.0][..]));
    }
}

#[cfg(test)]
mod last_write_scope {
    //! The content-derived CDC value-scope of the most recent committed write.
    use super::*;

    fn run(g: &mut Graph, q: &str) {
        crate::gql::parse(q)
            .unwrap()
            .execute(g, &crate::gql::eval::Params::new())
            .unwrap();
    }

    #[test]
    fn scope_reflects_the_last_write_touched_values() {
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Msg"],"properties":{"room":1}}"#,
        )
        .unwrap();

        // An INSERT into room 42 → scope ["42"] (a number renders without `.0`).
        run(&mut g, "INSERT (:Msg {room: 42, body: 'hi'})");
        assert_eq!(g.last_write_scope("room"), vec!["42".to_string()]);

        // A SET touching the seed vertex (room 1) → scope ["1"].
        run(&mut g, "MATCH (m:Msg {room: 1}) SET m.body = 'edited'");
        assert_eq!(g.last_write_scope("room"), vec!["1".to_string()]);

        // A write touching two rooms → both, distinct, in touch order.
        run(
            &mut g,
            "INSERT (:Msg {room: 7}), (:Msg {room: 7}), (:Msg {room: 9})",
        );
        let scope = g.last_write_scope("room");
        assert_eq!(scope.len(), 2);
        assert!(scope.contains(&"7".to_string()) && scope.contains(&"9".to_string()));

        // A string scope key renders verbatim; a missing key contributes nothing.
        run(&mut g, "INSERT (:Msg {tenant: 'acme', body: 'x'})");
        assert_eq!(g.last_write_scope("tenant"), vec!["acme".to_string()]);
        assert!(g.last_write_scope("room").is_empty()); // that write set no room
    }
}

#[cfg(test)]
mod null_is_first_class {
    //! `null` is a stored, present property value — NOT sugar for removal. These
    //! lock in the semantics `set_value`/`is_present`/`remove_value` agree on,
    //! and guard against a regression back to the old "SET null removes" model
    //! (a deliberate divergence from Cypher/TinkerPop).
    use super::*;

    fn props(len: usize) -> Properties {
        let mut p = Properties::default();
        for _ in 0..len {
            p.push_element();
        }
        p
    }

    #[test]
    fn a_stored_null_is_present_and_distinct_from_absent() {
        let mut strs = Dict::default();
        let mut p = props(2);
        p.set_value(0, "k", Value::Null, &mut strs); // row 0: present null; row 1: untouched

        assert!(p.is_present(0, "k"), "a stored null is present");
        assert!(
            matches!(p.value(0, "k", &strs), Value::Null),
            "and reads back as Null"
        );
        assert!(!p.is_present(1, "k"), "an unset key is absent");
        assert!(
            matches!(p.value(1, "k", &strs), Value::Null),
            "absent also reads as Null"
        );
    }

    #[test]
    fn setting_null_stores_it_without_disturbing_a_typed_column() {
        // A Num key set to null on another row keeps both — the column promotes
        // to Mixed rather than the null vanishing.
        let mut strs = Dict::default();
        let mut p = props(2);
        p.set_value(0, "k", Value::Num(5.0), &mut strs);
        p.set_value(1, "k", Value::Null, &mut strs);

        assert!(matches!(p.value(0, "k", &strs), Value::Num(n) if n == 5.0));
        assert!(p.is_present(1, "k"));
        assert!(matches!(p.value(1, "k", &strs), Value::Null));
    }

    #[test]
    fn remove_value_deletes_even_a_stored_null() {
        let mut strs = Dict::default();
        let mut p = props(1);
        p.set_value(0, "k", Value::Null, &mut strs);
        assert!(p.is_present(0, "k"));

        p.remove_value(0, "k"); // explicit removal is the ONLY way to unset it
        assert!(!p.is_present(0, "k"));
    }
}

#[cfg(test)]
mod storable_maps {
    //! A `Value::Map` is a first-class STORED property (boxed in a `Mixed`
    //! column, like a non-numeric list), canonicalized to sorted keys on the way
    //! in — the substrate foundation for GQL records / Gremlin maps.
    use super::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn map(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn stored_map_roundtrips_with_keys_sorted() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        // Author keys OUT of order; storage must canonicalize to sorted.
        p.set_value(
            0,
            "meta",
            map(&[("name", s("marko")), ("age", Value::Num(29.0))]),
            &mut strs,
        );
        assert!(p.is_present(0, "meta"));
        assert_eq!(
            p.value(0, "meta", &strs),
            map(&[("age", Value::Num(29.0)), ("name", s("marko"))]),
        );
    }

    #[test]
    fn nested_maps_and_lists_are_canonicalized_recursively() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(
            0,
            "m",
            map(&[
                ("z", Value::Num(1.0)),
                (
                    "a",
                    Value::List(vec![map(&[("y", Value::Num(2.0)), ("x", Value::Num(3.0))])]),
                ),
            ]),
            &mut strs,
        );
        assert_eq!(
            p.value(0, "m", &strs),
            map(&[
                (
                    "a",
                    Value::List(vec![map(&[("x", Value::Num(3.0)), ("y", Value::Num(2.0))])]),
                ),
                ("z", Value::Num(1.0)),
            ]),
        );
    }

    #[test]
    fn duplicate_field_names_collapse_last_wins() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(
            0,
            "m",
            Value::Map(vec![
                ("k".into(), Value::Num(1.0)),
                ("k".into(), Value::Num(2.0)),
            ]),
            &mut strs,
        );
        assert_eq!(p.value(0, "m", &strs), map(&[("k", Value::Num(2.0))]));
    }

    #[test]
    fn map_null_field_is_preserved_and_distinct_from_absence() {
        // A present field with a null value survives the round-trip (null is a
        // first-class value inside a record, mirroring the top-level policy).
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(0, "m", map(&[("k", Value::Null)]), &mut strs);
        assert_eq!(p.value(0, "m", &strs), map(&[("k", Value::Null)]));
    }

    #[test]
    fn a_map_key_coexists_with_scalar_keys_via_mixed() {
        let mut strs = Dict::default();
        let mut p = Properties::default();
        p.push_element();
        p.set_value(0, "n", Value::Num(1.0), &mut strs);
        p.set_value(0, "m", map(&[("a", Value::Num(1.0))]), &mut strs);
        assert!(matches!(p.value(0, "n", &strs), Value::Num(n) if n == 1.0));
        assert_eq!(p.value(0, "m", &strs), map(&[("a", Value::Num(1.0))]));
    }

    // A three-vertex graph with a nested `meta.city` field, for the dotted-path
    // index. Never index the map — index the scalar leaf at the path.
    fn city_graph() -> Graph {
        crate::ndjson::decode(
            "{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"P\"],\"properties\":{\"meta\":{\"city\":\"NYC\"}}}\n\
             {\"type\":\"node\",\"id\":\"b\",\"labels\":[\"P\"],\"properties\":{\"meta\":{\"city\":\"LA\"}}}\n\
             {\"type\":\"node\",\"id\":\"c\",\"labels\":[\"P\"],\"properties\":{\"meta\":{\"city\":\"NYC\"}}}",
        )
        .unwrap()
    }

    #[test]
    fn dotted_path_index_seeks_a_nested_field() {
        let mut g = city_graph();
        g.create_vertex_index("meta.city");
        // Both NYC vertices (a=0, c=2), the one LA vertex (b=1).
        let nyc = g
            .vertices_by_prop("meta.city", &IdxKey::Str("NYC".into()))
            .unwrap();
        assert_eq!(nyc, &[0, 2]);
        let la = g
            .vertices_by_prop("meta.city", &IdxKey::Str("LA".into()))
            .unwrap();
        assert_eq!(la, &[1]);
        // A city with no vertex → an empty (but present) bucket.
        assert_eq!(
            g.vertices_by_prop("meta.city", &IdxKey::Str("SF".into())),
            Some(&[][..]),
        );
    }

    #[test]
    fn dotted_path_index_maintained_on_write() {
        let mut g = city_graph();
        g.create_vertex_index("meta.city");
        // Move vertex b (LA → NYC): the index must follow. Bucket order is
        // unspecified, so compare as a set.
        g.set_vertex_prop(1, "meta", map(&[("city", s("NYC"))]));
        let mut nyc = g
            .vertices_by_prop("meta.city", &IdxKey::Str("NYC".into()))
            .unwrap()
            .to_vec();
        nyc.sort_unstable();
        assert_eq!(nyc, vec![0, 1, 2]);
        assert_eq!(
            g.vertices_by_prop("meta.city", &IdxKey::Str("LA".into())),
            Some(&[][..]),
        );
    }

    #[test]
    fn dotted_path_index_skips_absent_or_nonscalar_leaves() {
        let mut g = city_graph();
        // An index into a field that doesn't exist on any vertex → empty index.
        g.create_vertex_index("meta.zip");
        assert_eq!(
            g.vertices_by_prop("meta.zip", &IdxKey::Str("10001".into())),
            Some(&[][..]),
        );
        // Point one vertex's `meta.zip` at a nested map (non-scalar) → not indexed.
        g.set_vertex_prop(
            0,
            "meta",
            map(&[("city", s("NYC")), ("zip", map(&[("k", s("v"))]))]),
        );
        assert_eq!(
            g.vertices_by_prop("meta.zip", &IdxKey::Str("10001".into())),
            Some(&[][..]),
        );
    }
}

#[cfg(test)]
mod transactions {
    //! R-TX: an explicit transaction over the GQL eval mutation path must roll
    //! back to byte-identical prior state, and commit must persist. The eval layer
    //! wraps each statement in its own auto-commit frame, so these tests exercise
    //! the *nested* case (explicit begin → statements → rollback/commit), where
    //! the inner per-statement frames join the outer one.
    use super::*;
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) {
        parse(q)
            .unwrap()
            .execute(g, &Params::new())
            .unwrap_or_else(|e| panic!("query failed: {q}: {e:?}"));
    }

    #[test]
    fn rollback_restores_exact_prior_state() {
        let mut g = ndjson::decode("").unwrap();
        // Seed committed data (outside any explicit transaction).
        run(&mut g, "INSERT (:User {name: 'Seed', age: 1})");
        let before = ndjson::encode(&g);
        let vc_before = g.vertex_count();

        g.begin_tx();
        // A brand-new vertex (insert) and a mutation of the seed (property write).
        run(&mut g, "INSERT (:User {name: 'A'})");
        run(
            &mut g,
            "MATCH (u:User {name: 'Seed'}) SET u.name = 'Changed', u.age = 99",
        );
        // Read-your-writes: the staged inserts are visible inside the transaction.
        assert_eq!(g.vertex_count(), vc_before + 1);

        g.rollback_tx();

        assert_eq!(g.vertex_count(), vc_before, "vertex_count restored");
        assert_eq!(ndjson::encode(&g), before, "serialization byte-identical");
        // The seed's property values are exactly as before.
        let rows = parse("MATCH (u:User {name: 'Seed'}) RETURN u.age")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(
            rows.rows().count(),
            1,
            "the changed-then-rolled-back seed is back"
        );
    }

    #[test]
    fn commit_persists() {
        let mut g = ndjson::decode("").unwrap();
        g.begin_tx();
        run(&mut g, "INSERT (:User {name: 'A'})");
        assert!(matches!(g.commit_tx(), Ok(())));
        assert_eq!(g.vertex_count(), 1, "the committed insert persists");
        assert!(!g.in_transaction());
    }

    #[test]
    fn rollback_restores_deleted_vertex_and_its_edge() {
        // DETACH DELETE cascades an edge removal; rollback must un-tombstone both
        // the vertex and the edge in place (byte-identical serialization).
        let mut g = ndjson::decode("").unwrap();
        run(
            &mut g,
            "INSERT (:User {name: 'A'})-[:KNOWS {since: 2020}]->(:User {name: 'B'})",
        );
        let before = ndjson::encode(&g);
        let (vc, ec) = (g.vertex_count(), g.edge_count());

        g.begin_tx();
        run(&mut g, "MATCH (u:User {name: 'A'}) DETACH DELETE u");
        assert_eq!(g.vertex_count(), vc - 1);
        assert_eq!(g.edge_count(), ec - 1);

        g.rollback_tx();

        assert_eq!(g.vertex_count(), vc, "vertex restored");
        assert_eq!(g.edge_count(), ec, "cascaded edge restored");
        assert_eq!(ndjson::encode(&g), before, "serialization byte-identical");
    }

    #[test]
    fn per_statement_atomicity_leaves_no_partial_write() {
        // A single INSERT of two rows whose second collides under a unique
        // constraint must leave ZERO rows — the whole statement rolls back.
        let mut g = ndjson::decode("").unwrap();
        g.create_unique_constraint("Acct", "email").unwrap();
        let err = parse("INSERT (:Acct {email: 'a@x.io'}), (:Acct {email: 'a@x.io'})")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.vertex_count(), 0, "the faulting statement left no trace");
    }
}

#[cfg(test)]
mod cardinality {
    //! R-CONSTRAINTS cardinality (degree bounds), exercised over the GQL eval
    //! path (each statement is an auto-commit frame, so max AND min land at the
    //! per-statement commit). Byte-identical to the TS core.
    use super::*;
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<()> {
        parse(q).unwrap().execute(g, &Params::new()).map(|_| ())
    }

    #[test]
    fn exactly_one_via_gql_commit() {
        let mut g = ndjson::decode("").unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap();

        // Node + mandatory edge in one INSERT (one auto-commit frame) satisfies it.
        run(
            &mut g,
            "INSERT (:Purchase {id: 'o1'})-[:PLACED_BY]->(:Customer {id: 'c1'})",
        )
        .unwrap();
        assert_eq!(g.vertex_count(), 2);

        // A bare Purchase with no PLACED_BY out-edge is degree 0 < min → rejected, and
        // the statement rolls back (no trace).
        let err = run(&mut g, "INSERT (:Purchase {id: 'o2'})").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.vertex_count(), 2, "the rejected INSERT left no trace");
    }

    #[test]
    fn over_max_is_rejected_at_commit() {
        let mut g = ndjson::decode("").unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 0, Some(1))
            .unwrap();
        run(
            &mut g,
            "INSERT (:Purchase {id: 'o1'})-[:PLACED_BY]->(:Customer {id: 'c1'})",
        )
        .unwrap();
        // A second PLACED_BY out-edge from o1 pushes its out-degree to 2 > max 1.
        let err = run(
            &mut g,
            "MATCH (o:Purchase {id: 'o1'}), (c:Customer {id: 'c1'}) INSERT (o)-[:PLACED_BY]->(c)",
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.edge_count(), 1, "the over-max edge rolled back");
    }

    #[test]
    fn remove_edge_below_min_rolls_back() {
        let mut g = ndjson::decode("").unwrap();
        run(
            &mut g,
            "INSERT (:Purchase {id: 'o1'})-[:PLACED_BY]->(:Customer {id: 'c1'})",
        )
        .unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap();
        // Deleting the only PLACED_BY edge drops o1 to degree 0 < min → rejected.
        let err = run(&mut g, "MATCH (:Purchase)-[r:PLACED_BY]->() DELETE r").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.edge_count(), 1, "the delete rolled back");
    }

    #[test]
    fn declare_time_scan_and_self_loop_degree() {
        let mut g = ndjson::decode("").unwrap();
        run(&mut g, "INSERT (:Purchase {id: 'o1'})").unwrap(); // degree 0
                                                               // min:1 over existing degree-0 data → rejected at declare time.
        let err = g
            .create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);

        // A self-loop counts once for out and once for in.
        run(
            &mut g,
            "MATCH (o:Purchase {id: 'o1'}) INSERT (o)-[:SELF]->(o)",
        )
        .unwrap();
        // The sole Purchase vertex is index 0 (first inserted); `id` is a property,
        // not the external vertex identity, so degree is read by index here.
        assert_eq!(g.out_degree(0, "SELF"), 1);
        assert_eq!(g.in_degree(0, "SELF"), 1);
    }

    #[test]
    fn drop_and_introspection() {
        let mut g = ndjson::decode("").unwrap();
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 1, Some(1))
            .unwrap();
        g.create_cardinality_constraint("Customer", "PRIMARY", 1, 0, Some(1))
            .unwrap();
        assert_eq!(
            g.cardinality_constraints(),
            vec![
                ("Customer".into(), "PRIMARY".into(), 1, 0, Some(1)),
                ("Purchase".into(), "PLACED_BY".into(), 0, 1, Some(1)),
            ]
        );
        // Re-declaring replaces the bounds (not a second entry).
        g.create_cardinality_constraint("Purchase", "PLACED_BY", 0, 0, None)
            .unwrap();
        assert_eq!(g.cardinality_constraints().len(), 2);
        g.drop_cardinality_constraint("Purchase", "PLACED_BY", 0);
        assert_eq!(
            g.cardinality_constraints(),
            vec![("Customer".into(), "PRIMARY".into(), 1, 0, Some(1))]
        );
        g.drop_cardinality_constraint("Purchase", "PLACED_BY", 0); // idempotent
    }
}

#[cfg(test)]
mod validator {
    //! R-CONSTRAINTS custom validators (a GQL boolean predicate per label),
    //! exercised over the GQL eval path (each statement is an auto-commit frame,
    //! so the predicate is re-checked against every touched element at the
    //! per-statement commit). SQL-`CHECK` semantics — a definite `false` fails, a
    //! null/unknown passes. Byte-identical to the TS `createValidator`.
    use super::*;
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<()> {
        parse(q).unwrap().execute(g, &Params::new()).map(|_| ())
    }

    #[test]
    fn per_write_reject_accept_and_null_passes() {
        let mut g = ndjson::decode("").unwrap();
        g.create_validator("User", "u", "u.age >= 0 AND u.age < 150")
            .unwrap();

        let err = run(&mut g, "INSERT (:User {age: -5})").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.vertex_count(), 0, "the rejected INSERT left no trace");

        run(&mut g, "INSERT (:User {age: 20})").unwrap();
        // No `age` → `u.age` is null → predicate UNKNOWN → passes (SQL-CHECK).
        run(&mut g, "INSERT (:User {name: 'Ada'})").unwrap();
        run(&mut g, "INSERT (:User {age: null, name: 'Bo'})").unwrap();
        assert_eq!(g.vertex_count(), 3);
    }

    #[test]
    fn declare_time_scan_rejects_violating_data() {
        let mut g = ndjson::decode("").unwrap();
        run(&mut g, "INSERT (:User {age: -5})").unwrap();

        let err = g.create_validator("User", "u", "u.age >= 0").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        // The rejected declaration registered nothing.
        assert!(g.validators().is_empty());
    }

    #[test]
    fn deferred_within_a_transaction() {
        // Briefly-invalid-then-fixed across an explicit multi-statement frame → the
        // final state satisfies the validator, so the transaction commits.
        let mut g2 = ndjson::decode("").unwrap();
        g2.create_validator("User", "u", "u.age >= 0").unwrap();
        g2.begin_tx();
        parse("INSERT (:User {id: 'a', age: -5})")
            .unwrap()
            .execute(&mut g2, &Params::new())
            .unwrap();
        parse("MATCH (u:User {id: 'a'}) SET u.age = 5")
            .unwrap()
            .execute(&mut g2, &Params::new())
            .unwrap();
        assert!(g2.commit_tx().is_ok(), "final state valid → commits");
        assert_eq!(g2.vertex_count(), 1);

        // Left invalid across the frame → the whole transaction rolls back.
        let mut g3 = ndjson::decode("").unwrap();
        g3.create_validator("User", "u", "u.age >= 0").unwrap();
        g3.begin_tx();
        parse("INSERT (:User {id: 'b', age: -1})")
            .unwrap()
            .execute(&mut g3, &Params::new())
            .unwrap();
        let err = g3.commit_tx().unwrap_err();
        assert!(matches!(err, TxCommitError::Validator(_)));
        g3.rollback_tx();
        assert_eq!(g3.vertex_count(), 0, "rolled back");
    }

    #[test]
    fn edge_validator() {
        let mut g = ndjson::decode("").unwrap();
        g.create_validator("KNOWS", "r", "r.weight >= 0").unwrap();

        let err = run(
            &mut g,
            "INSERT (:P {name: 'a'})-[:KNOWS {weight: -1}]->(:P {name: 'b'})",
        )
        .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert_eq!(g.edge_count(), 0, "rejected edge left no trace");

        run(
            &mut g,
            "INSERT (:P {name: 'a'})-[:KNOWS {weight: 5}]->(:P {name: 'b'})",
        )
        .unwrap();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn drop_and_introspection() {
        let mut g = ndjson::decode("").unwrap();
        g.create_validator("User", "u", "u.age >= 0").unwrap();
        g.create_validator("User", "u", "u.age < 150").unwrap();

        assert_eq!(
            g.validators(),
            vec![
                ("User".into(), "u".into(), "u.age < 150".into()),
                ("User".into(), "u".into(), "u.age >= 0".into()),
            ]
        );

        g.drop_validator("User");
        assert!(g.validators().is_empty());
        // No validator left → a previously-rejected write now succeeds.
        run(&mut g, "INSERT (:User {age: -5})").unwrap();
        assert_eq!(g.vertex_count(), 1);
    }

    #[test]
    fn unparseable_predicate_is_a_syntax_error() {
        let mut g = ndjson::decode("").unwrap();
        assert_eq!(
            g.create_validator("User", "u", "u.age >>>")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        assert_eq!(
            g.create_validator("User", "u", "").unwrap_err().code,
            ErrorCode::Syntax
        );
        // A predicate smuggling in an extra clause is rejected too.
        assert_eq!(
            g.create_validator("User", "u", "true RETURN 1")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
    }

    #[test]
    fn predicate_referencing_the_wrong_variable_is_rejected_at_declare_time() {
        let mut g = ndjson::decode("").unwrap();
        // The predicate references `x`, but the element binds to `u` — `x.age` is
        // unbound → the predicate reads UNKNOWN → the SQL-CHECK never fires and the
        // validator would silently do nothing. Reject it at DECLARE time (Syntax).
        assert_eq!(
            g.create_validator("User", "u", "x.age >= 0")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        // A bare unbound name (no dotted property) is rejected too.
        assert_eq!(
            g.create_validator("User", "u", "age >= 0")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        // The rejected declarations registered nothing.
        assert!(g.validators().is_empty());

        // The declared variable is fine, and a constant predicate (references NO
        // variable at all) is legitimately allowed.
        g.create_validator("User", "u", "u.age >= 0").unwrap();
        g.create_validator("User", "u", "1 = 1").unwrap();
        assert_eq!(g.validators().len(), 2);

        // A sub-query pattern variable is bound *within* the sub-query, so a
        // predicate that references only `u` and its own sub-pattern vars is fine.
        g.create_validator("User", "u", "EXISTS { (v) WHERE v.age = u.age }")
            .unwrap();
    }
}

#[cfg(test)]
mod invariant {
    //! Graph-level INVARIANTS (cross-write assertions): a whole-graph GQL query
    //! run ONCE per write transaction against the fully-staged graph. `false`-only
    //! -fails — VIOLATED iff a result cell is boolean `false`; everything else
    //! (`true`/`null`/non-boolean/empty) holds. Enforced in `commit_tx` after the
    //! per-element deferred checks, and only when the transaction wrote something.
    //! Byte-identical to the TS `createInvariant`.
    use super::*;
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<()> {
        parse(q).unwrap().execute(g, &Params::new()).map(|_| ())
    }

    // Two accounts summing to zero; the classic double-entry ledger. The `name`
    // property (not the ndjson node id) is what MATCH patterns key on.
    const LEDGER: &str = "\
{\"type\":\"node\",\"id\":\"a\",\"labels\":[\"Acct\"],\"properties\":{\"name\":\"a\",\"balance\":100}}
{\"type\":\"node\",\"id\":\"b\",\"labels\":[\"Acct\"],\"properties\":{\"name\":\"b\",\"balance\":-100}}";

    #[test]
    fn balanced_transfer_commits_unbalanced_rolls_back() {
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();

        // A transfer that keeps the sum at zero commits.
        g.begin_tx();
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 70").unwrap();
        run(&mut g, "MATCH (b:Acct {name: 'b'}) SET b.balance = -70").unwrap();
        assert!(g.commit_tx().is_ok(), "sum still 0 → commits");

        // An unbalanced half-transfer rolls the whole transaction back.
        g.begin_tx();
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 999").unwrap();
        let err = g.commit_tx().unwrap_err();
        assert!(matches!(err, TxCommitError::Invariant(_)));
        g.rollback_tx();

        // The balances are unchanged from the last good commit (70 / -70).
        let rows = parse("MATCH (a:Acct) RETURN sum(a.balance) AS s")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(rows.row(0)[0], Value::Num(0.0));
    }

    #[test]
    fn single_statement_unbalanced_write_rejected() {
        // Every GQL statement auto-commits, so a single unbalanced SET trips the
        // invariant at its own commit boundary (no explicit transaction needed).
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();

        let err = run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 5").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        // Rolled back — the balance is still 100.
        let rows = parse("MATCH (a:Acct {name: 'a'}) RETURN a.balance AS b")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(rows.row(0)[0], Value::Num(100.0));
    }

    #[test]
    fn declare_time_rejects_already_violating_graph() {
        let mut g = ndjson::decode(LEDGER).unwrap();
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 5").ok(); // no invariant yet → fine
                                                                          // Now the sum is -95, so declaring the invariant must reject.
        let err = g
            .create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        assert!(
            g.invariants().is_empty(),
            "rejected declaration stored nothing"
        );
    }

    #[test]
    fn count_invariant_at_least_one_admin() {
        let seed = "\
{\"type\":\"node\",\"id\":\"u1\",\"labels\":[\"User\"],\"properties\":{\"name\":\"u1\",\"role\":\"Admin\"}}
{\"type\":\"node\",\"id\":\"u2\",\"labels\":[\"User\"],\"properties\":{\"name\":\"u2\",\"role\":\"Member\"}}";
        let mut g = ndjson::decode(seed).unwrap();
        g.create_invariant(
            "has_admin",
            "MATCH (u:User) WHERE u.role = 'Admin' RETURN count(u) > 0",
        )
        .unwrap();

        // Demote the member → still one admin → holds.
        run(&mut g, "MATCH (u:User {name: 'u2'}) SET u.role = 'Guest'").unwrap();
        // Demote the last admin → count drops to 0 → violated, rolled back.
        let err = run(&mut g, "MATCH (u:User {name: 'u1'}) SET u.role = 'Guest'").unwrap_err();
        assert_eq!(err.code, ErrorCode::ConstraintViolation);
        let rows = parse("MATCH (u:User {role: 'Admin'}) RETURN count(u) AS n")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert_eq!(rows.row(0)[0], Value::Num(1.0));
    }

    #[test]
    fn pure_read_transaction_does_not_run_the_invariant() {
        // The gate proof: with the graph in a state that VIOLATES the invariant, a
        // pure-read transaction must still commit (the invariant is not run), while
        // a transaction that writes anything trips it. We break the sum via the
        // direct store API (which bypasses the GQL auto-commit that would catch it)
        // to set up a violating-but-committed state.
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();

        // Directly skew one balance so the sum is now -50 (invariant would fail).
        let vi = g.vertex_indices().next().unwrap();
        g.set_vertex_prop(vi, "balance", Value::Num(50.0));

        // A pure-read transaction commits — the invariant is skipped (nothing written).
        g.begin_tx();
        parse("MATCH (a:Acct) RETURN a.balance")
            .unwrap()
            .execute(&mut g, &Params::new())
            .unwrap();
        assert!(g.commit_tx().is_ok(), "pure-read commit skips invariants");

        // But a transaction that writes runs the invariant against the (violating)
        // staged graph and rolls back.
        g.begin_tx();
        run(&mut g, "MATCH (a:Acct {name: 'b'}) SET a.balance = -100").unwrap();
        assert!(
            matches!(g.commit_tx().unwrap_err(), TxCommitError::Invariant(_)),
            "a writing commit runs the invariant"
        );
        g.rollback_tx();
    }

    #[test]
    fn drop_and_introspection() {
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("balanced", "MATCH (a:Acct) RETURN sum(a.balance) = 0")
            .unwrap();
        g.create_invariant("has_acct", "MATCH (a:Acct) RETURN count(a) >= 0")
            .unwrap();
        assert_eq!(
            g.invariants(),
            vec![
                (
                    "balanced".into(),
                    "MATCH (a:Acct) RETURN sum(a.balance) = 0".into()
                ),
                (
                    "has_acct".into(),
                    "MATCH (a:Acct) RETURN count(a) >= 0".into()
                ),
            ]
        );

        g.drop_invariant("balanced");
        assert_eq!(
            g.invariants(),
            vec![(
                "has_acct".into(),
                "MATCH (a:Acct) RETURN count(a) >= 0".into()
            )]
        );
        // Dropped → a previously-rejected unbalanced write now succeeds.
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 5").unwrap();
    }

    #[test]
    fn unparseable_query_is_a_syntax_error() {
        let mut g = ndjson::decode("").unwrap();
        assert_eq!(
            g.create_invariant("bad", "MATCH (a:Acct) RETURN >>>")
                .unwrap_err()
                .code,
            ErrorCode::Syntax
        );
        assert_eq!(
            g.create_invariant("empty", "").unwrap_err().code,
            ErrorCode::Syntax
        );
    }

    #[test]
    fn non_boolean_and_null_and_empty_all_hold() {
        // `false`-only-fails: a non-boolean cell, a null cell, and an empty result
        // set each HOLD (only a literal `false` cell fails).
        let mut g = ndjson::decode(LEDGER).unwrap();
        g.create_invariant("nonbool", "MATCH (a:Acct) RETURN sum(a.balance)")
            .unwrap(); // yields 0 (a number, not false) → holds
        g.create_invariant("nullcell", "MATCH (a:Acct) RETURN a.missing")
            .unwrap(); // null cells → hold
        g.create_invariant("empty", "MATCH (z:NoSuchLabel) RETURN z.x = z.x")
            .unwrap(); // empty result → holds
                       // A write still commits (all three hold regardless of the balance sum).
        run(&mut g, "MATCH (a:Acct {name: 'a'}) SET a.balance = 12345").unwrap();
    }
}

#[cfg(test)]
mod clone_graph {
    //! `Graph: Clone` — the fast fork/branch substrate. A deep, independent copy
    //! of the columnar store (element ids preserved exactly), the native half of
    //! `graph.copy()` over the FFI. Mirrors the TS `Graph.copy()` for parity.
    use super::*;
    use crate::gql::eval::Params;
    use crate::gql::parse;
    use crate::ndjson;

    fn run(g: &mut Graph, q: &str) -> CodeResult<Vec<Vec<crate::graph::Value>>> {
        parse(q)
            .unwrap()
            .execute(g, &Params::new())
            .map(|rs| rs.rows().map(<[Value]>::to_vec).collect())
    }

    #[test]
    fn clone_is_independent_and_preserves_ids_and_constraints() {
        let mut base = ndjson::decode(
            &[
                r#"{"type":"node","id":"a","labels":["P"],"properties":{"id":"a","v":1}}"#,
                r#"{"type":"node","id":"b","labels":["P"],"properties":{"id":"b","v":2}}"#,
                r#"{"type":"edge","id":"e1","from":"a","to":"b","labels":["R"],"properties":{}}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        base.create_unique_constraint("P", "id").unwrap();

        let mut copy = base.clone();

        // Independent: a write to one is invisible to the other.
        run(&mut copy, "INSERT (:P {id: 'c', v: 3})").unwrap();
        run(&mut base, "MATCH (n:P {id: 'a'}) SET n.v = 99").unwrap();
        assert_eq!(
            run(&mut base, "MATCH (n:P) RETURN count(*) AS c").unwrap(),
            vec![vec![Value::Num(2.0)]]
        );
        assert_eq!(
            run(&mut copy, "MATCH (n:P) RETURN count(*) AS c").unwrap(),
            vec![vec![Value::Num(3.0)]]
        );
        assert_eq!(
            run(&mut copy, "MATCH (n:P {id: 'a'}) RETURN n.v AS v").unwrap(),
            vec![vec![Value::Num(1.0)]] // unaffected by base's SET
        );

        // Ids preserved exactly: the edge still connects the same endpoints.
        assert_eq!(
            run(
                &mut copy,
                "MATCH (:P {id: 'a'})-[:R]->(x:P) RETURN x.id AS id"
            )
            .unwrap(),
            vec![vec![Value::Str("b".into())]]
        );

        // Indexes come along and are functional (declared + populated): the seek
        // path is used, and the index is listed on the copy.
        base.create_vertex_index("v");
        base.create_edge_index("w");
        let mut copy2 = base.clone();
        assert!(copy2.vertex_indexes().contains(&"v".to_string()));
        assert!(copy2.edge_indexes().contains(&"w".to_string()));
        // `b.v` is 2 (untouched); `a.v` was set to 99 earlier in this test.
        assert_eq!(
            run(&mut copy2, "MATCH (n:P {v: 2}) RETURN n.id AS id").unwrap(),
            vec![vec![Value::Str("b".into())]]
        );

        // Every constraint kind is enforced on the copy, not just unique. Required
        // + type on a fresh graph so the checks are unambiguous.
        let mut g2 = ndjson::decode(
            &[r#"{"type":"node","id":"a","labels":["P"],"properties":{"id":"a","age":30}}"#]
                .join("\n"),
        )
        .unwrap();
        g2.create_required_constraint("P", "id").unwrap();
        g2.create_type_constraint("P", "age", "number").unwrap();
        let mut copy3 = g2.clone();
        assert!(run(&mut copy3, "INSERT (:P {age: 1})").is_err()); // missing required id
        assert!(run(&mut copy3, "INSERT (:P {id: 'z', age: 'x'})").is_err()); // wrong type

        // The unique constraint came along: a duplicate id is rejected in the copy.
        assert!(run(&mut copy, "INSERT (:P {id: 'a'})").is_err());
    }
}

#[cfg(test)]
mod record_type_spec {
    use super::*;

    fn sc(k: &str, t: PropType) -> (Arc<str>, TypeSpec, bool) {
        (k.into(), TypeSpec::Scalar(t), false)
    }
    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }

    #[test]
    fn parse_scalar_and_record_types() {
        assert_eq!(
            TypeSpec::parse("string"),
            Some(TypeSpec::Scalar(PropType::Str))
        );
        // Fields canonicalized to sorted order; `::` and `:` both accepted.
        assert_eq!(
            TypeSpec::parse("record { tier :: number, city :: string }"),
            Some(TypeSpec::Record(vec![
                sc("city", PropType::Str),
                sc("tier", PropType::Num)
            ])),
        );
        assert_eq!(
            TypeSpec::parse("record{a:number}"),
            Some(TypeSpec::Record(vec![sc("a", PropType::Num)])),
        );
        // Nested record.
        assert_eq!(
            TypeSpec::parse("record{addr::record{city::string}}"),
            Some(TypeSpec::Record(vec![(
                "addr".into(),
                TypeSpec::Record(vec![sc("city", PropType::Str)]),
                false,
            )])),
        );
        // A `NOT NULL` field parses to the required flag and round-trips.
        let nn = TypeSpec::parse("record{id::string NOT NULL,tier::number}").unwrap();
        assert_eq!(
            nn,
            TypeSpec::Record(vec![
                ("id".into(), TypeSpec::Scalar(PropType::Str), true),
                sc("tier", PropType::Num),
            ]),
        );
        assert_eq!(TypeSpec::parse(&nn.to_name()), Some(nn));
        assert_eq!(TypeSpec::parse("record{id::string NOT}"), None); // NOT without NULL
                                                                     // Round-trips through to_name.
        let t = TypeSpec::parse("record{city::string,tier::number}").unwrap();
        assert_eq!(TypeSpec::parse(&t.to_name()), Some(t));
        // Malformed.
        assert_eq!(TypeSpec::parse("record{a}"), None);
        assert_eq!(TypeSpec::parse("nope"), None);
        assert_eq!(TypeSpec::parse("record{a:string"), None);
    }

    #[test]
    fn value_matches_record_type() {
        let ty = TypeSpec::parse("record{city::string,tier::number}").unwrap();
        // Exact shape matches.
        assert!(value_matches(
            &vmap(&[("city", s("NYC")), ("tier", Value::Num(2.0))]),
            &ty
        ));
        // A null value satisfies any type (REQUIRED is separate).
        assert!(value_matches(&Value::Null, &ty));
        // A null FIELD is allowed (field-level required is separate).
        assert!(value_matches(
            &vmap(&[("city", Value::Null), ("tier", Value::Num(2.0))]),
            &ty
        ));
        // Wrong field type → no match.
        assert!(!value_matches(
            &vmap(&[("city", Value::Num(1.0)), ("tier", Value::Num(2.0))]),
            &ty
        ));
        // A missing NULLABLE field is OK (optional by default); the empty record
        // is OK too. An EXTRA field is rejected (closed on extras).
        assert!(value_matches(&vmap(&[("city", s("NYC"))]), &ty));
        assert!(value_matches(&vmap(&[]), &ty));
        assert!(!value_matches(
            &vmap(&[
                ("city", s("NYC")),
                ("tier", Value::Num(2.0)),
                ("x", Value::Num(1.0))
            ]),
            &ty,
        ));
        // A non-map value → no match.
        assert!(!value_matches(&Value::Num(1.0), &ty));

        // `NOT NULL` makes a field required (present + non-null).
        let req = TypeSpec::parse("record{id::string NOT NULL,tier::number}").unwrap();
        assert!(value_matches(&vmap(&[("id", s("x"))]), &req)); // tier optional
        assert!(!value_matches(&vmap(&[("tier", Value::Num(1.0))]), &req)); // id absent
        assert!(!value_matches(&vmap(&[("id", Value::Null)]), &req)); // id null
        assert!(value_matches(
            &vmap(&[("id", s("x")), ("tier", Value::Null)]),
            &req
        )); // tier nullable null OK
    }
}

#[cfg(test)]
mod record_constraint {
    use super::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }
    fn base() -> Graph {
        crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn declare_record_constraint_validates_existing_data() {
        let mut g = base();
        // Matching shape → declares OK.
        assert!(g
            .create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .is_ok());
        // A conflicting shape against existing data → rejected at declaration.
        let mut g2 = base();
        assert!(g2
            .create_type_constraint("Person", "meta", "record{city::number,tier::number}")
            .is_err());
    }

    #[test]
    fn record_constraint_enforced_on_set_and_insert() {
        let mut g = base();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        // A well-shaped write passes.
        assert!(!g.type_conflict_on_set(
            0,
            "meta",
            &vmap(&[("city", s("LA")), ("tier", Value::Num(3.0))])
        ));
        // A wrong field type conflicts.
        assert!(g.type_conflict_on_set(
            0,
            "meta",
            &vmap(&[("city", Value::Num(1.0)), ("tier", Value::Num(3.0))])
        ));
        // A missing NULLABLE field does NOT conflict (optional by default); an
        // EXTRA field does (closed on extras).
        assert!(!g.type_conflict_on_set(0, "meta", &vmap(&[("city", s("LA"))])));
        assert!(g.type_conflict_on_set(
            0,
            "meta",
            &vmap(&[
                ("city", s("LA")),
                ("tier", Value::Num(1.0)),
                ("x", Value::Num(9.0))
            ])
        ));
        // A null is exempt.
        assert!(!g.type_conflict_on_set(0, "meta", &Value::Null));
        // A new-vertex insert with a bad meta is a type_violation.
        assert!(g
            .type_violation(
                &["Person".to_string()],
                &[(
                    "meta".to_string(),
                    vmap(&[("city", Value::Num(9.0)), ("tier", Value::Num(1.0))])
                )],
            )
            .is_some());
    }

    #[test]
    fn dropping_the_record_constraint_lifts_enforcement() {
        let mut g = base();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        g.drop_type_constraint("Person", "meta");
        assert!(!g.type_conflict_on_set(0, "meta", &vmap(&[("city", Value::Num(1.0))])));
    }
}

/// Scalar `NOT NULL` on a type constraint (`string NOT NULL`) — the type-surface
/// spelling of a required (present + non-null) property, mirroring per-field
/// `NOT NULL` in a record.
#[cfg(test)]
mod scalar_not_null {
    use super::*;

    fn node(props: &str) -> Graph {
        crate::ndjson::decode(&format!(
            r#"{{"type":"node","id":"a","labels":["P"],"properties":{props}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn parse_roundtrips_a_top_level_not_null() {
        assert_eq!(
            TypeSpec::parse_with_not_null("string"),
            Some((TypeSpec::Scalar(PropType::Str), false))
        );
        assert_eq!(
            TypeSpec::parse_with_not_null("string NOT NULL"),
            Some((TypeSpec::Scalar(PropType::Str), true))
        );
        // case-insensitive, whitespace-tolerant
        assert_eq!(
            TypeSpec::parse_with_not_null("number  not null"),
            Some((TypeSpec::Scalar(PropType::Num), true))
        );
        assert_eq!(TypeSpec::parse_with_not_null("string NOT"), None); // NOT without NULL
        assert_eq!(TypeSpec::parse_with_not_null("bogus"), None);
    }

    #[test]
    fn declare_requires_existing_data_present_and_non_null() {
        // present + non-null → OK
        assert!(node(r#"{"name":"marko"}"#)
            .create_type_constraint("P", "name", "string NOT NULL")
            .is_ok());
        // absent → declare fails
        assert!(node("{}")
            .create_type_constraint("P", "name", "string NOT NULL")
            .is_err());
        // stored null → declare fails
        assert!(node(r#"{"name":null}"#)
            .create_type_constraint("P", "name", "string NOT NULL")
            .is_err());
        // a plain (nullable) type constraint stays exempt from absent/null
        assert!(node("{}")
            .create_type_constraint("P", "name", "string")
            .is_ok());
    }

    #[test]
    fn missing_required_folds_in_the_not_null_type_constraint() {
        let mut g = node(r#"{"name":"marko"}"#);
        g.create_type_constraint("P", "name", "string NOT NULL")
            .unwrap();
        let labels = ["P".to_string()];
        // absent / null → a required violation; present non-null → OK.
        assert!(g.missing_required(&labels, &[]).is_some());
        assert!(g
            .missing_required(&labels, &[("name".to_string(), Value::Null)])
            .is_some());
        assert!(g
            .missing_required(&labels, &[("name".to_string(), Value::Str("x".into()))])
            .is_none());
        // A wrong TYPE is still a separate type violation.
        assert!(g
            .type_violation(&labels, &[("name".to_string(), Value::Num(1.0))])
            .is_some());
    }

    #[test]
    fn dropping_leaves_an_independent_required_intact() {
        let mut g = node(r#"{"name":"marko"}"#);
        g.create_required_constraint("P", "name").unwrap(); // declared independently
        g.create_type_constraint("P", "name", "string NOT NULL")
            .unwrap();
        g.drop_type_constraint("P", "name");
        // The type not-null is gone, but the independent required still enforces.
        assert!(!g.v_type_not_null.contains_key("P"));
        assert!(g.missing_required(&["P".to_string()], &[]).is_some());
    }

    #[test]
    fn dump_schema_roundtrips_scalar_not_null() {
        let mut g = node(r#"{"name":"marko"}"#);
        g.create_type_constraint("P", "name", "string NOT NULL")
            .unwrap();
        assert!(g.dump_schema().contains(r#""type":"string NOT NULL""#));
    }

    #[test]
    fn not_null_on_a_record_type_is_now_supported() {
        // Previously rejected; a whole-record NOT NULL is now a valid constraint
        // (see the `any_record_and_record_not_null` module). Over PRESENT data it
        // declares cleanly.
        assert!(node(r#"{"meta":{"a":1}}"#)
            .create_type_constraint("P", "meta", "record{a::number} NOT NULL")
            .is_ok());
    }

    #[test]
    fn edge_scalar_not_null_enforces_and_roundtrips() {
        let mut g = crate::ndjson::decode(
            concat!(
                r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"edge","id":"e","from":"a","to":"b","labels":["LINK"],"properties":{"w":1.5}}"#,
            ),
        )
        .unwrap();
        g.create_edge_type_constraint("LINK", "w", "number NOT NULL")
            .unwrap();
        let et = ["LINK".to_string()];
        assert!(g.edge_missing_required(&et, &[]).is_some());
        assert!(g
            .edge_missing_required(&et, &[("w".to_string(), Value::Num(2.0))])
            .is_none());
        assert!(g.dump_schema().contains(r#""type":"number NOT NULL""#));
    }
}

/// ISO `<record type> ::= [ANY] RECORD [<field spec>] [NOT NULL]` — the OPEN
/// record (`ANY RECORD` / bare `RECORD`) and a whole-record `NOT NULL`.
#[cfg(test)]
mod any_record_and_record_not_null {
    use super::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }
    fn node(props: &str) -> Graph {
        crate::ndjson::decode(&format!(
            r#"{{"type":"node","id":"a","labels":["P"],"properties":{props}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn parse_open_record_forms() {
        // `any record`, bare `record` → the open type; canonical name is `any record`.
        assert_eq!(TypeSpec::parse("any record"), Some(TypeSpec::AnyRecord));
        assert_eq!(TypeSpec::parse("record"), Some(TypeSpec::AnyRecord));
        assert_eq!(TypeSpec::parse("ANY  RECORD"), Some(TypeSpec::AnyRecord));
        assert_eq!(TypeSpec::AnyRecord.to_name(), "any record");
        assert_eq!(TypeSpec::parse("any"), None); // ANY without RECORD
                                                  // The closed form still parses to a Record.
        assert!(matches!(
            TypeSpec::parse("record{a::number}"),
            Some(TypeSpec::Record(_))
        ));
    }

    #[test]
    fn any_record_matches_any_map_but_not_a_scalar() {
        assert!(value_matches(
            &vmap(&[("x", Value::Num(1.0))]),
            &TypeSpec::AnyRecord
        ));
        assert!(value_matches(&vmap(&[]), &TypeSpec::AnyRecord)); // empty map OK
        assert!(value_matches(&Value::Null, &TypeSpec::AnyRecord)); // top-level null exempt
        assert!(!value_matches(&Value::Num(1.0), &TypeSpec::AnyRecord)); // a scalar is not a record
    }

    #[test]
    fn any_record_constraint_enforces_and_does_not_debox() {
        let mut g = node(r#"{"meta":{"city":"NYC"}}"#);
        g.create_type_constraint("P", "meta", "any record").unwrap();
        // Any-shaped map passes; a scalar is a type violation.
        assert!(!g.type_conflict_on_set(0, "meta", &vmap(&[("anything", Value::Num(9.0))])));
        assert!(g.type_conflict_on_set(0, "meta", &Value::Num(1.0)));
        // Open record has no field contract → the column stays boxed (NOT de-boxed).
        assert!(matches!(g.props.col("meta"), Some(Column::Mixed { .. })));
    }

    #[test]
    fn record_level_not_null_parses_and_is_required() {
        assert_eq!(
            TypeSpec::parse_with_not_null("record{a::number} NOT NULL"),
            TypeSpec::parse("record{a::number}").map(|s| (s, true))
        );
        assert_eq!(
            TypeSpec::parse_with_not_null("any record NOT NULL"),
            Some((TypeSpec::AnyRecord, true))
        );

        // A closed record + NOT NULL: present map OK; absent / null → required violation.
        let mut g = node(r#"{"meta":{"city":"NYC"}}"#);
        g.create_type_constraint("P", "meta", "record{city::string} NOT NULL")
            .unwrap();
        let labels = ["P".to_string()];
        assert!(g.missing_required(&labels, &[]).is_some());
        assert!(g
            .missing_required(&labels, &[("meta".to_string(), Value::Null)])
            .is_some());
        assert!(g
            .missing_required(&labels, &[("meta".to_string(), vmap(&[("city", s("LA"))]))])
            .is_none());
        // A NOT NULL closed record STILL de-boxes (presence is orthogonal to shape).
        assert!(matches!(g.props.col("meta"), Some(Column::Record { .. })));
        assert!(g
            .dump_schema()
            .contains(r#""type":"record{city::string} NOT NULL""#));
    }

    #[test]
    fn declaring_record_not_null_over_absent_data_fails() {
        // A label vertex missing the key → the NOT NULL declare is rejected.
        assert!(node("{}")
            .create_type_constraint("P", "meta", "any record NOT NULL")
            .is_err());
        // A present map satisfies it.
        assert!(node(r#"{"meta":{"a":1}}"#)
            .create_type_constraint("P", "meta", "any record NOT NULL")
            .is_ok());
    }

    #[test]
    fn dropping_a_record_not_null_removes_its_requiredness() {
        let mut g = node(r#"{"meta":{"a":1}}"#);
        g.create_type_constraint("P", "meta", "any record NOT NULL")
            .unwrap();
        g.drop_type_constraint("P", "meta");
        assert!(g.missing_required(&["P".to_string()], &[]).is_none());
    }
}

/// R-CONSTRAINTS Step 2: a declared RECORD constraint de-boxes the property key's
/// column into typed per-field sub-columns ([`Column::Record`]). These tests pin
/// the substrate: de-boxing happens on declare, every read round-trips
/// byte-identically to the boxed map, non-conforming values (a shared key across
/// labels) stay correct via the escape overlay, backfill + drop-rebox work, and a
/// field reads straight from its sub-column.
#[cfg(test)]
mod record_debox {
    use super::*;

    fn s(x: &str) -> Value {
        Value::Str(x.into())
    }
    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn vmap(pairs: &[(&str, Value)]) -> Value {
        Value::Map(
            pairs
                .iter()
                .map(|(k, v)| ((*k).into(), v.clone()))
                .collect(),
        )
    }
    /// A graph with `Person` a whose `meta = {city, tier}` and a record constraint
    /// already declared (so `meta` is de-boxed).
    fn declared() -> Graph {
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        g
    }
    fn read(g: &Graph, idx: usize) -> Value {
        g.props.value(idx, "meta", &g.strs)
    }
    fn write(g: &mut Graph, idx: usize, v: Value) {
        g.props.set_value(idx, "meta", v, &mut g.strs);
    }
    fn col_name(g: &Graph, key: &str) -> &'static str {
        match g.props.col(key) {
            Some(Column::Record { .. }) => "record",
            Some(Column::Mixed { .. }) => "mixed",
            _ => "other",
        }
    }

    #[test]
    fn declaring_deboxes_the_column_and_types_the_fields() {
        let g = declared();
        assert_eq!(col_name(&g, "meta"), "record");
        // The field sub-columns are TYPED (string→Str, number→Num), not boxed.
        let Some(Column::Record {
            field_names,
            fields,
            ..
        }) = g.props.col("meta")
        else {
            panic!("meta should be a de-boxed record column");
        };
        assert_eq!(
            field_names.iter().map(|n| n.as_ref()).collect::<Vec<_>>(),
            ["city", "tier"] // sorted canonical order
        );
        assert!(matches!(fields[0], Column::Str { .. }));
        assert!(matches!(fields[1], Column::Num { .. }));
        // The backfilled value reads back identically to the boxed map.
        assert_eq!(read(&g, 0), vmap(&[("city", s("NYC")), ("tier", n(2.0))]));
    }

    #[test]
    fn a_stored_null_field_keeps_its_sub_column_typed() {
        // 1b: a nullable field set to an explicit null records the null in the
        // per-field `field_nulls` bitset and keeps the sub-column TYPED — it no
        // longer promotes to `Mixed`. The value still round-trips.
        let mut g = declared();
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", Value::Null)]));
        assert_eq!(
            read(&g, 0),
            vmap(&[("city", s("LA")), ("tier", Value::Null)])
        );
        let Some(Column::Record {
            fields,
            field_nulls,
            ..
        }) = g.props.col("meta")
        else {
            panic!();
        };
        // `tier` (field 1) stayed a Num column; its null bit is set for element 0.
        assert!(matches!(fields[1], Column::Num { .. }), "tier stayed typed");
        assert!(field_nulls[1].get(0), "tier null recorded in field_nulls");
        assert!(!field_nulls[0].get(0), "city is not null");
        // Overwriting the null with a value clears the null bit.
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", n(9.0))]));
        let Some(Column::Record { field_nulls, .. }) = g.props.col("meta") else {
            panic!();
        };
        assert!(
            !field_nulls[1].get(0),
            "null bit cleared on a non-null write"
        );
    }

    #[test]
    fn a_nested_record_field_deboxes_recursively() {
        // 1a: a RECORD-typed field is itself a de-boxed `Column::Record`.
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["P"],"properties":{"addr":{"geo":{"lat":1.0,"lng":2.0}}}}"#,
        )
        .unwrap();
        g.create_type_constraint("P", "addr", "record{geo::record{lat::number,lng::number}}")
            .unwrap();
        let Some(Column::Record { fields, .. }) = g.props.col("addr") else {
            panic!("addr should be de-boxed");
        };
        // The `geo` field is a NESTED record column whose own fields are typed.
        let Column::Record {
            field_names: geo_names,
            fields: geo_fields,
            ..
        } = &fields[0]
        else {
            panic!("geo should be a nested record column, not boxed");
        };
        assert_eq!(
            geo_names.iter().map(|n| n.as_ref()).collect::<Vec<_>>(),
            ["lat", "lng"]
        );
        assert!(matches!(geo_fields[0], Column::Num { .. }));
        // Reads round-trip (whole record + a deep field).
        assert_eq!(
            g.props.value(0, "addr", &g.strs),
            vmap(&[("geo", vmap(&[("lat", n(1.0)), ("lng", n(2.0))]))])
        );
        let kid = g.props.keys.get("addr").unwrap();
        assert_eq!(g.props.field_at(0, kid, &["geo", "lat"], &g.strs), n(1.0));
    }

    #[test]
    fn every_record_shape_roundtrips_byte_identically() {
        let mut g = declared();
        for v in [
            vmap(&[("city", s("LA")), ("tier", n(3.0))]),     // full
            vmap(&[("city", s("SF"))]),                       // nullable field omitted
            vmap(&[("city", s("X")), ("tier", Value::Null)]), // field stored null
            vmap(&[]),                                        // empty map (present, not absent)
        ] {
            write(&mut g, 0, v.clone());
            assert_eq!(read(&g, 0), v, "round-trip mismatch");
            assert!(g.props.is_present(0, "meta"));
        }
        // A stored null at the top level reads back as null but stays PRESENT.
        write(&mut g, 0, Value::Null);
        assert_eq!(read(&g, 0), Value::Null);
        assert!(g.props.is_present(0, "meta"));
        // Removal is distinct from a stored null: absent, not present.
        g.props.remove_value(0, "meta");
        assert_eq!(read(&g, 0), Value::Null);
        assert!(!g.props.is_present(0, "meta"));
    }

    #[test]
    fn nonconforming_values_stay_correct_via_the_escape_overlay() {
        // The column is de-boxed for `Person.meta`, but the key is global — a
        // scalar or a differently-shaped map must still round-trip.
        let mut g = declared();
        let a = g.add_vertex(&["Other".into()], vec![]);
        let b = g.add_vertex(&["Other".into()], vec![]);
        write(&mut g, a as usize, n(42.0)); // a scalar escapes
        write(
            &mut g,
            b as usize,
            vmap(&[("lat", n(1.0)), ("lng", n(2.0))]), // an extra-keyed map escapes
        );
        assert_eq!(read(&g, a as usize), n(42.0));
        assert_eq!(
            read(&g, b as usize),
            vmap(&[("lat", n(1.0)), ("lng", n(2.0))])
        );
        assert!(g.props.is_present(a as usize, "meta"));
        // Overwriting an escapee with a conforming map clears the escape.
        write(&mut g, a as usize, vmap(&[("city", s("NYC"))]));
        assert_eq!(read(&g, a as usize), vmap(&[("city", s("NYC"))]));
        let Some(Column::Record { escaped, .. }) = g.props.col("meta") else {
            panic!();
        };
        assert!(!escaped.contains_key(&a), "escape not cleared on reconform");
        assert!(escaped.contains_key(&b), "b still escaped");
    }

    #[test]
    fn typing_a_mixed_population_succeeds_and_deboxes_each_faithfully() {
        // The scenario: pre-existing vertices, then you type the label. A LABELED
        // vertex that already conforms de-boxes into fields; a labeled vertex that
        // merely LACKS the property is exempt (nullable); a DIFFERENT-label vertex
        // holding a non-conforming value isn't checked and escapes. Declaration
        // succeeds and every value survives byte-identically.
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        let b = g.add_vertex(&["Person".into()], vec![]) as usize; // labeled, meta ABSENT
        let c = g.add_vertex(&["Other".into()], vec![("meta".into(), n(42.0))]) as usize; // other label, scalar
        assert!(g
            .create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .is_ok());
        assert_eq!(col_name(&g, "meta"), "record");
        assert_eq!(read(&g, 0), vmap(&[("city", s("NYC")), ("tier", n(2.0))])); // scattered
        assert!(!g.props.is_present(b, "meta")); // still absent (nullable, exempt)
        assert_eq!(read(&g, c as usize), n(42.0)); // escaped, unchanged
    }

    #[test]
    fn a_labeled_violator_makes_typing_throw_and_deboxes_nothing() {
        // If ANY live vertex carrying the label already violates the shape, the
        // declaration throws — atomically. No constraint is recorded and the column
        // is NOT de-boxed (no half-applied state, no grandfathered landmine).
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        // A second Person whose meta is a scalar — a violation of the record shape.
        g.add_vertex(&["Person".into()], vec![("meta".into(), n(1.0))]);
        assert!(g
            .create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .is_err());
        assert_eq!(col_name(&g, "meta"), "mixed", "column untouched on failure");
        // No constraint recorded → a would-be-violating write no longer conflicts.
        assert!(!g.type_conflict_on_set(0, "meta", &n(7.0)));
    }

    #[test]
    fn declaring_before_bulk_append_scatters_directly_never_boxing() {
        // The order the user expects to pay off: declare the constraint on an empty
        // graph, THEN bulk-ingest. `ndjson::append` routes through add_vertex_with_id
        // → set_value → the Record arm, so conforming maps scatter straight into the
        // typed sub-columns — no `Value::Map` is ever boxed, nothing escapes.
        let mut g = crate::ndjson::decode("").unwrap();
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        assert_eq!(col_name(&g, "meta"), "record"); // empty Record column exists up front
        let batch = [
            r#"{"type":"node","id":"p0","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":1}}}"#,
            // nullable `tier` omitted — a conforming partial record
            r#"{"type":"node","id":"p1","labels":["Person"],"properties":{"meta":{"city":"LA"}}}"#,
        ]
        .join("\n");
        crate::ndjson::append(&mut g, &batch).unwrap();

        let Some(Column::Record {
            fields, escaped, ..
        }) = g.props.col("meta")
        else {
            panic!("meta stayed boxed after a declare-then-append");
        };
        assert!(matches!(fields[0], Column::Str { .. }));
        assert!(matches!(fields[1], Column::Num { .. }));
        assert!(
            escaped.is_empty(),
            "a conforming bulk append must never box/escape"
        );
        let p0 = g.vertex_by_id("p0").unwrap() as usize;
        let p1 = g.vertex_by_id("p1").unwrap() as usize;
        assert_eq!(read(&g, p0), vmap(&[("city", s("NYC")), ("tier", n(1.0))]));
        assert_eq!(read(&g, p1), vmap(&[("city", s("LA"))]));
    }

    #[test]
    fn field_at_reads_a_deboxed_field_directly() {
        let mut g = declared();
        let kid = g.props.keys.get("meta").unwrap();
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", n(3.0))]));
        assert_eq!(g.props.field_at(0, kid, &["city"], &g.strs), s("LA"));
        assert_eq!(g.props.field_at(0, kid, &["tier"], &g.strs), n(3.0));
        // A field the (nullable) value omits → Null.
        write(&mut g, 0, vmap(&[("city", s("LA"))]));
        assert_eq!(g.props.field_at(0, kid, &["tier"], &g.strs), Value::Null);
        // An undeclared segment → Null (closed record).
        assert_eq!(g.props.field_at(0, kid, &["nope"], &g.strs), Value::Null);
        // On a stored-null / absent record, a field is Null.
        write(&mut g, 0, Value::Null);
        assert_eq!(g.props.field_at(0, kid, &["city"], &g.strs), Value::Null);
        // On an escapee, `field_at` walks the boxed value.
        let e = g.add_vertex(&["Other".into()], vec![]) as usize;
        write(&mut g, e, vmap(&[("lat", n(9.0))]));
        assert_eq!(g.props.field_at(e, kid, &["lat"], &g.strs), n(9.0));
    }

    #[test]
    fn backfill_on_declare_matches_the_boxed_reads() {
        // Store several boxed maps FIRST, snapshot their reads, then declare.
        let mut g = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        let b = g.add_vertex(
            &["Person".into()],
            vec![("meta".into(), vmap(&[("city", s("LA"))]))],
        ) as usize;
        let c = g.add_vertex(
            &["Person".into()],
            vec![("meta".into(), vmap(&[("city", s("SF")), ("tier", n(7.0))]))],
        ) as usize;
        let before: Vec<Value> = (0..=c).map(|i| read(&g, i)).collect();
        assert_eq!(col_name(&g, "meta"), "mixed");
        g.create_type_constraint("Person", "meta", "record{city::string,tier::number}")
            .unwrap();
        assert_eq!(col_name(&g, "meta"), "record");
        let after: Vec<Value> = (0..=c).map(|i| read(&g, i)).collect();
        assert_eq!(before, after, "backfill changed a value");
        assert!(g.props.is_present(b, "meta") && g.props.is_present(c, "meta"));
    }

    #[test]
    fn dropping_the_constraint_reboxes_to_mixed_without_data_loss() {
        let mut g = declared();
        write(&mut g, 0, vmap(&[("city", s("LA")), ("tier", n(3.0))]));
        let before = read(&g, 0);
        g.drop_type_constraint("Person", "meta");
        assert_eq!(col_name(&g, "meta"), "mixed");
        assert_eq!(read(&g, 0), before, "rebox changed the value");
    }

    #[test]
    fn a_second_label_on_the_same_key_keeps_it_deboxed_until_both_drop() {
        let mut g = declared();
        g.create_type_constraint("Company", "meta", "record{city::string,tier::number}")
            .unwrap();
        assert_eq!(col_name(&g, "meta"), "record");
        // Dropping one of two constraints on the key must NOT re-box.
        g.drop_type_constraint("Person", "meta");
        assert_eq!(col_name(&g, "meta"), "record");
        // Dropping the last one re-boxes.
        g.drop_type_constraint("Company", "meta");
        assert_eq!(col_name(&g, "meta"), "mixed");
    }

    #[test]
    fn ndjson_encodes_a_deboxed_record_identically_to_the_boxed_map() {
        // The same graph, boxed vs de-boxed, must serialize to the same NDJSON.
        let boxed = crate::ndjson::decode(
            r#"{"type":"node","id":"a","labels":["Person"],"properties":{"meta":{"city":"NYC","tier":2}}}"#,
        )
        .unwrap();
        let deboxed = declared();
        assert_eq!(col_name(&deboxed, "meta"), "record");
        assert_eq!(
            crate::ndjson::encode(&boxed),
            crate::ndjson::encode(&deboxed)
        );
    }

    #[test]
    fn edge_record_constraint_deboxes_and_roundtrips() {
        let mut g = crate::ndjson::decode(
            concat!(
                r#"{"type":"node","id":"a","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"node","id":"b","labels":["P"],"properties":{}}"#,
                "\n",
                r#"{"type":"edge","id":"e","from":"a","to":"b","labels":["LINK"],"properties":{"meta":{"w":0.5}}}"#,
            ),
        )
        .unwrap();
        g.create_edge_type_constraint("LINK", "meta", "record{w::number}")
            .unwrap();
        assert!(matches!(
            g.edge_props.col("meta"),
            Some(Column::Record { .. })
        ));
        assert_eq!(
            g.edge_props.value(0, "meta", &g.strs),
            vmap(&[("w", n(0.5))])
        );
        g.drop_edge_type_constraint("LINK", "meta");
        assert!(matches!(
            g.edge_props.col("meta"),
            Some(Column::Mixed { .. })
        ));
        assert_eq!(
            g.edge_props.value(0, "meta", &g.strs),
            vmap(&[("w", n(0.5))])
        );
    }
}

#[cfg(test)]
mod temporal_index_key_tests {
    use super::*;
    use crate::temporal as t;

    /// The scalar `Temporal::index_key` i128 MUST equal the column's
    /// `monotonic_key` bit-for-bit — otherwise a key built from a query literal
    /// won't match a key built from a stored column and the index silently returns
    /// wrong rows. Guards against the two encodings drifting apart.
    #[test]
    fn temporal_index_key_matches_column() {
        let cases: Vec<(TemporalKind, Vec<Temporal>)> = vec![
            (
                TemporalKind::Date,
                vec![
                    Temporal::Date(t::Date { days: -1000 }),
                    Temporal::Date(t::Date { days: 0 }),
                    Temporal::Date(t::Date { days: 19_723 }),
                ],
            ),
            (
                TemporalKind::DateTime,
                vec![
                    Temporal::DateTime(t::DateTime { secs: -5, nanos: 0 }),
                    Temporal::DateTime(t::DateTime {
                        secs: 1_700_000_000,
                        nanos: 123,
                    }),
                ],
            ),
            (
                TemporalKind::ZonedDateTime,
                vec![
                    Temporal::ZonedDateTime(t::ZonedDateTime {
                        secs: 1_700_000_000,
                        nanos: 5,
                        offset: -120,
                    }),
                    Temporal::ZonedDateTime(t::ZonedDateTime {
                        secs: 1_700_000_000,
                        nanos: 5,
                        offset: 300,
                    }),
                ],
            ),
            (
                TemporalKind::Time,
                vec![Temporal::Time(t::Time {
                    secs: 3600,
                    nanos: 42,
                })],
            ),
            (
                TemporalKind::ZonedTime,
                vec![Temporal::ZonedTime(t::ZonedTime {
                    secs: 3600,
                    nanos: 42,
                    offset: 60,
                })],
            ),
        ];
        for (kind, vals) in cases {
            let mut col = TemporalCol::with_len(kind, vals.len());
            for (i, v) in vals.iter().enumerate() {
                assert!(col.set(i, v), "{kind:?} slot {i}: set kind mismatch");
            }
            for (i, v) in vals.iter().enumerate() {
                let col_key = col.monotonic_key(i).expect("indexable kind has a key");
                assert_eq!(
                    col.get(i).index_key().unwrap().1,
                    col_key,
                    "{kind:?} slot {i}: get→scalar drift"
                );
                assert_eq!(
                    v.index_key().unwrap().1,
                    col_key,
                    "{kind:?} slot {i}: scalar drift"
                );
            }
        }
        // Duration has no monotonic key on either side.
        assert!(Temporal::Duration(t::Duration {
            months: 1,
            days: 2,
            secs: 3,
            nanos: 4
        })
        .index_key()
        .is_none());
    }

    /// Within a kind the key is monotonic with the value's own order; across kinds
    /// the kind rank keeps them disjoint so a range seek never interleaves them.
    #[test]
    fn temporal_index_key_is_monotonic_and_kind_disjoint() {
        let dates: Vec<Temporal> = (0..40)
            .map(|k| Temporal::Date(t::Date { days: k * 9 - 137 }))
            .collect();
        for w in dates.windows(2) {
            assert!(
                w[0].index_key().unwrap() < w[1].index_key().unwrap(),
                "date order broke"
            );
        }
        // A max Date still ranks below a min DateTime (disjoint kind ranges).
        let big_date = Temporal::Date(t::Date { days: i32::MAX })
            .index_key()
            .unwrap();
        let small_dt = Temporal::DateTime(t::DateTime {
            secs: i64::MIN,
            nanos: 0,
        })
        .index_key()
        .unwrap();
        assert!(
            big_date < small_dt,
            "kind ranks must keep Date below DateTime"
        );
    }
}
