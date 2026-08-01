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

use std::borrow::Cow;
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
#[derive(Debug, Clone)]
pub struct Dict {
    /// Open-addressed index into `strings`: each slot is `(hash, id + 1)`, with
    /// `0` marking empty. Power-of-two length, linear probing.
    ///
    /// This replaced a `HashMap<Arc<str>, u32>`, and the reason is memory access
    /// rather than hashing. That map stored a 16-byte fat pointer per entry, and
    /// any probe surviving the control-byte filter had to DEREFERENCE the `Arc`
    /// to compare the key — a second cache miss, into a separate allocation, for
    /// every lookup. Past roughly a million entries the dictionary no longer fits
    /// in cache and those misses become the cost of ingest.
    ///
    /// Here a slot is 8 bytes, so a cache line holds eight of them, and the
    /// stored hash rejects a non-match without touching the string at all. The
    /// string is dereferenced only when the full 32-bit hash already matched.
    ///
    /// The hash itself is unchanged — still `RandomState`, still SipHash, still
    /// seeded per process. This is deliberate: the keys are ids, labels and
    /// property names taken straight from a document, so a weaker hash would let
    /// a crafted upload collide them and make ingest quadratic.
    table: Vec<(u32, u32)>,
    hasher: std::collections::hash_map::RandomState,
    pub strings: Vec<Arc<str>>,
}

impl Default for Dict {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Dict {
    /// A dictionary sized for `n` distinct entries up front, so the table never
    /// has to grow and rehash mid-build.
    pub fn with_capacity(n: usize) -> Self {
        // Keep the load factor under 7/8.
        let want = (n * 8 / 7).next_power_of_two().max(16);

        Self {
            table: vec![(0, 0); want],
            hasher: std::collections::hash_map::RandomState::new(),
            strings: Vec::with_capacity(n),
        }
    }

    fn hash_of(&self, s: &str) -> u32 {
        use std::hash::{BuildHasher, Hasher};

        let mut h = self.hasher.build_hasher();

        h.write(s.as_bytes());
        // Fold to 32 bits; the low bits pick the slot, so keep the high entropy.
        (h.finish() >> 32) as u32
    }

    /// Slot holding `s`, or the first empty slot where it would go.
    fn probe(&self, s: &str, hash: u32) -> usize {
        let mask = self.table.len() - 1;
        let mut i = hash as usize & mask;

        loop {
            let (h, id) = self.table[i];

            if id == 0 || (h == hash && &*self.strings[(id - 1) as usize] == s) {
                return i;
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let mut bigger = vec![(0u32, 0u32); self.table.len() * 2];
        let mask = bigger.len() - 1;

        for &(h, id) in &self.table {
            if id != 0 {
                let mut i = h as usize & mask;

                while bigger[i].1 != 0 {
                    i = (i + 1) & mask;
                }
                bigger[i] = (h, id);
            }
        }
        self.table = bigger;
    }

    pub fn intern(&mut self, s: &str) -> u32 {
        let hash = self.hash_of(s);
        let mut slot = self.probe(s, hash);

        if self.table[slot].1 != 0 {
            return self.table[slot].1 - 1;
        }

        // Grow at 7/8, which also guarantees the table always holds an empty
        // slot — `probe` relies on that to terminate.
        if (self.strings.len() + 1) * 8 >= self.table.len() * 7 {
            self.grow();
            slot = self.probe(s, hash);
        }

        let id = self.strings.len() as u32;

        self.strings.push(Arc::from(s));
        self.table[slot] = (hash, id + 1);

        id
    }

    pub fn get(&self, s: &str) -> Option<u32> {
        let hash = self.hash_of(s);
        let (_, id) = self.table[self.probe(s, hash)];

        (id != 0).then(|| id - 1)
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
    /// declared `RECORD { … }` type constraint. Because the
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

/// The scalar type a TYPE constraint can require of a property
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

/// A declared property TYPE for a type constraint: either a scalar [`PropType`] or
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
/// Why the per-vertex adjacency is a `Vec<Vec<Adj>>` and stays one.
///
/// It looks wasteful, and partly is: a vertex with any edge owns a separate heap
/// allocation, so building a million-vertex graph makes ~2M small ones. Measured
/// on 1M edges, spreading them over 1M vertices rather than 10k costs ~380 ms of
/// a 1537 ms decode.
///
/// Storing the first two entries inline (a hand-rolled small-vector, no new
/// dependency) was built and measured. It recovered only ~5% of the build, since
/// most of that 380 ms turned out to be the header array's cache footprint
/// rather than the allocations, and the inline enum is 32 bytes against a
/// `Vec`'s 24, so the array grew by a third and gave some of it back. Traversal
/// did not benefit at all, because a warm read goes through the packed CSR
/// snapshot below and never touches this structure. And `add_edge` got 11%
/// SLOWER (4.28 -> 3.82 M ops/sec) — the enum dispatch and spill check on every
/// push.
///
/// Which is the same lesson the CSR experiment left behind: this structure is on
/// the WRITE path, and reads already have a flat one. Optimizing it for read
/// locality targets a cost that has already been paid elsewhere, while any
/// per-push overhead lands on every mutation.
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
/// Host-configurable settings for one graph — the single place engine knobs live,
/// so adding one does not mean adding another `set_*` export and another ABI bump.
///
/// Sections keep the space from becoming a flat pile as it grows; today there is
/// one (`limits`), and the FFI setter is keyed by a stable [`ConfigId`] rather
/// than by section+field, so a new setting is purely additive across the ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphConfig {
    pub limits: GraphLimits,
}

/// Resource ceilings. These are ANTI-RUNAWAY bounds, not semantics: a query under
/// the ceiling behaves identically whatever the ceiling is, and tripping one is
/// always a loud `E_RESOURCE_EXHAUSTED` (or `E_SYNTAX`, for the parse-time
/// `operator_chain`), never a truncated result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphLimits {
    /// Ceiling on the element count `range(start, end [, step])` may materialize.
    /// A GQL list is a MATERIALIZED value in both engines (`Val::List(Vec<Val>)` /
    /// a JS array) — indexing, `size`, sorting, equality and serialization all
    /// assume that — so a range cannot be produced lazily without a lazy-list
    /// variant threaded through both value models. Unbounded, `range(0, 1e21)` is
    /// not merely slow: the f64 counter stops advancing at 2^53 (`i += 1.0` is a
    /// no-op there), so the loop never terminates while pushing, and the host dies
    /// on an OOM kill instead of the query erroring.
    pub range: u64,
    /// Per-expansion cap on trail-traversal steps; a guard against exponential
    /// blowup on a dense graph. Mirrored by the TS engine's `TRAIL_BUDGET`.
    pub trail: u64,
    /// Ceiling on the intermediate frontier a fixed-length multi-segment scan may
    /// materialize. A chain `(a)-[]->(b)-[]->(c)-[]->…` fans out the cross-product
    /// of partial matches segment by segment, and the trailing LIMIT only prunes
    /// the *last* segment — every earlier layer is built in full first. On a dense
    /// graph that reaches billions of rows and takes the host down with an OOM kill
    /// rather than the query erroring. Generous enough that a real analytical join
    /// clears it; only a runaway cross-product trips it. NATIVE-ONLY — the TS
    /// engine has no vectorized frontier, so it ignores this one.
    pub intermediate: u64,
    /// Ceiling on GQL operator-chain length (`a AND b AND …`, `x + y + …`), applied
    /// by the PARSER rather than at evaluation time, so an over-long chain is
    /// `E_SYNTAX`. Anti-resource-abuse only — the n-ary AST never overflows the
    /// stack. A `prepare()` call may override it for that statement.
    pub operator_chain: u64,
}

impl Default for GraphLimits {
    fn default() -> Self {
        Self {
            range: 1_000_000,
            trail: 1_000_000,
            intermediate: 50_000_000,
            operator_chain: 10_000,
        }
    }
}

/// Stable wire ids for [`GraphConfig`] settings, so the FFI setter is ONE export
/// keyed by id rather than one export per knob (which is how the surface grew a
/// `lnk_graph_set_max_operator_chain` and was about to grow more). Ids are
/// append-only: an artifact predating a setting reports it as unknown rather than
/// silently ignoring it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ConfigId {
    LimitsRange = 0,
    LimitsTrail = 1,
    LimitsIntermediate = 2,
    LimitsOperatorChain = 3,
}

impl ConfigId {
    #[must_use]
    pub const fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::LimitsRange),
            1 => Some(Self::LimitsTrail),
            2 => Some(Self::LimitsIntermediate),
            3 => Some(Self::LimitsOperatorChain),
            _ => None,
        }
    }
}

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
    /// RI-tree interval indexes over edge `[lo_key, hi_key)` temporal pairs: an as-of
    /// `lo <= v AND hi > v` seeds from `tree.stab(v)`, and TWO indexes (valid `[vf,vt)`
    /// + transaction `[tf,tt)`) intersect for a bitemporal as-of. Maintained on writes.
    edge_interval_idxs: Vec<EdgeIntervalIdx>,
    /// UNIQUE constraints over vertex properties: label name → the sorted
    /// property keys that must be unique among live vertices carrying that label.
    /// Each constrained key is index-backed (declaring the constraint creates the
    /// vertex index), so enforcement and `_MERGE` key lookups seek rather than
    /// scan. Null/list values are exempt (SQL semantics — NULLs are distinct),
    /// which also matches what the value index can hold. See
    /// `docs/design/gql-extensions.md` §3.
    v_unique: HashMap<String, Vec<String>>,
    /// REQUIRED constraints: `label` → the property keys that must be present and
    /// non-null on every live vertex carrying that label. Unlike
    /// `v_unique` these need no backing index — enforcement is a presence check.
    v_required: HashMap<String, Vec<String>>,
    /// TYPE constraints: `label` → (`key` → the scalar type its present, non-null
    /// values must be). Null/absent are exempt.
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
    /// self-loop counts once for out and once for in. See `docs/design/transactions.md`.
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
    /// Transaction state. `tx_depth > 0` means an open transaction: writes
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
    /// must be re-checked at commit (deferred to commit for edge writes).
    tx_touched_edges: Vec<u32>,
    /// The vertices touched by the most recent committed write — a snapshot of
    /// `tx_touched` taken at commit (before it's cleared), so a caller can derive
    /// that write's value-scope for CDC routing (`last_write_scope`). Content-derived
    /// scope extraction rides the touched set the commit already collects.
    last_touched: Vec<u32>,
    /// How many interned names have already been checked for well-formedness
    /// (`labels`, `etype`, vertex prop keys, edge prop keys). Dictionaries are
    /// append-only and a valid name stays valid, so a commit only has to check
    /// the tail added since the last one — usually nothing at all. See
    /// [`Graph::validate_new_names`].
    names_checked: [usize; 4],
    applying_undo: bool,
    /// Anti-resource-abuse ceiling on GQL operator-chain length, passed to the
    /// parser on each query (see `gql::parser::DEFAULT_MAX_CHAIN`). Defaults to
    /// 10_000; set at graph creation via the native `maxOperatorChain` option.
    /// Host-configurable settings (see [`GraphConfig`]) — every engine knob,
    /// including the operator-chain ceiling that used to be its own field and its
    /// own FFI export.
    config: GraphConfig,
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
            edge_interval_idxs: self.edge_interval_idxs.clone(),
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
            names_checked: self.names_checked,
            applying_undo: self.applying_undo,
            config: self.config,
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
    /// A write introduced a label / edge type / property key that is not
    /// well-formed — an empty key or label, or a label carrying the GraphSON
    /// `::` separator. Caught at commit so EVERY write path is covered by one
    /// check, and rolled back like any other commit failure.
    MalformedName(CodeError),
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

/// An RI-tree interval index over an edge `[lo_key, hi_key)` temporal pair.
#[derive(Clone)]
struct EdgeIntervalIdx {
    lo_key: String,
    hi_key: String,
    tree: crate::interval_index::RiTree,
}

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

    /// This edge's temporal property `key` as a monotonic `i128` (None if absent /
    /// non-temporal / duration) — the RI-tree interval endpoint.
    pub(crate) fn edge_interval_key(&self, ei: u32, key: &str) -> Option<i128> {
        match self.edge_props.value(ei as usize, key, &self.strs) {
            Value::Temporal(t) => t.index_key().map(|(_, k)| k),
            _ => None,
        }
    }

    /// True if `key` is an endpoint of ANY edge interval index — a write to it moves
    /// that interval, so its RI-tree must be updated.
    fn key_is_interval_endpoint(&self, key: &str) -> bool {
        self.edge_interval_idxs
            .iter()
            .any(|i| i.lo_key == key || i.hi_key == key)
    }

    /// Endpoint key pairs of every interval index, owned (to sidestep the borrow between
    /// reading an edge's endpoints and mutating a tree).
    fn interval_specs(&self) -> Vec<(String, String)> {
        self.edge_interval_idxs
            .iter()
            .map(|i| (i.lo_key.clone(), i.hi_key.clone()))
            .collect()
    }

    /// Remove edge `ei` from every interval index whose endpoints it carries (no-op if
    /// none). Endpoints are read *before* the caller mutates.
    fn interval_idx_remove(&mut self, ei: u32) {
        for (n, (lo_key, hi_key)) in self.interval_specs().into_iter().enumerate() {
            if let (Some(lo), Some(hi)) = (
                self.edge_interval_key(ei, &lo_key),
                self.edge_interval_key(ei, &hi_key),
            ) {
                self.edge_interval_idxs[n].tree.remove(lo, hi, ei);
            }
        }
    }

    /// Insert edge `ei` into every interval index whose endpoints it carries.
    fn interval_idx_insert(&mut self, ei: u32) {
        for (n, (lo_key, hi_key)) in self.interval_specs().into_iter().enumerate() {
            if let (Some(lo), Some(hi)) = (
                self.edge_interval_key(ei, &lo_key),
                self.edge_interval_key(ei, &hi_key),
            ) {
                self.edge_interval_idxs[n].tree.insert(lo, hi, ei);
            }
        }
    }

    /// Declare (and backfill) an RI-tree interval index over an edge `[lo_key, hi_key)`
    /// temporal pair. Maintained on every edge mutation. Declaring a SECOND one (e.g.
    /// transaction-time `[tf, tt)` alongside valid-time `[vf, vt)`) lets a bitemporal
    /// as-of intersect both dimensions. Re-declaring the same pair rebuilds it. An edge
    /// missing an endpoint is simply not registered — the `WHERE` verifies regardless,
    /// so it's never a wrong answer, only a missed acceleration.
    pub fn create_edge_interval_index(&mut self, lo_key: &str, hi_key: &str) {
        self.edge_interval_idxs
            .retain(|i| !(i.lo_key == lo_key && i.hi_key == hi_key));
        let mut tree = crate::interval_index::RiTree::new();
        for ei in 0..self.e_src.len() as u32 {
            if !self.e_live[ei as usize] {
                continue;
            }
            if let (Some(lo), Some(hi)) = (
                self.edge_interval_key(ei, lo_key),
                self.edge_interval_key(ei, hi_key),
            ) {
                tree.insert(lo, hi, ei);
            }
        }
        self.edge_interval_idxs.push(EdgeIntervalIdx {
            lo_key: lo_key.to_string(),
            hi_key: hi_key.to_string(),
            tree,
        });
    }

    /// Candidate edge ids whose `[lo, hi]` contains point `q`, via the `n`-th interval
    /// index. A superset the caller's `WHERE` then verifies.
    pub fn edge_interval_stab_nth(&self, n: usize, q: i128) -> Vec<u32> {
        self.edge_interval_idxs[n].tree.stab(q)
    }

    /// Candidate edge ids whose `[lo, hi]` overlaps `[d1, d2]`, via the `n`-th index.
    pub fn edge_interval_overlap_nth(&self, n: usize, d1: i128, d2: i128) -> Vec<u32> {
        self.edge_interval_idxs[n].tree.overlap(d1, d2)
    }

    /// The result SIZE of a stab / overlap on the `n`-th interval index without
    /// materializing — for cheap cross-axis selectivity comparison (seed from the
    /// smallest; never materialize a non-selective stab like "everything believed now").
    pub fn edge_interval_stab_len_nth(&self, n: usize, q: i128) -> usize {
        self.edge_interval_idxs[n].tree.stab_len(q)
    }
    pub fn edge_interval_overlap_len_nth(&self, n: usize, d1: i128, d2: i128) -> usize {
        self.edge_interval_idxs[n].tree.overlap_len(d1, d2)
    }

    /// The `(lo_key, hi_key)` each edge interval index covers, in order — for the seed
    /// selector to recognize matching predicates and intersect across dimensions.
    pub(crate) fn edge_interval_index_specs(&self) -> Vec<(&str, &str)> {
        self.edge_interval_idxs
            .iter()
            .map(|i| (i.lo_key.as_str(), i.hi_key.as_str()))
            .collect()
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

/// One element's property list as the builder sees it: its dense index plus a
/// borrowed view of its key/value pairs.
type PropItem<'a, 'b> = (usize, &'b [(Cow<'a, str>, Value)]);

/// Lift an owned label list into the record shape. The records borrow from their
/// source document where they can (NDJSON hands out slices of the input), but a
/// codec that has to unescape, split or synthesize a name owns its strings and
/// says so here.
pub fn owned_labels(v: Vec<String>) -> Vec<Cow<'static, str>> {
    v.into_iter().map(Cow::Owned).collect()
}

/// The property-list counterpart of [`owned_labels`].
pub fn owned_props(v: Vec<(String, Value)>) -> Vec<(Cow<'static, str>, Value)> {
    v.into_iter().map(|(k, val)| (Cow::Owned(k), val)).collect()
}

pub struct NodeRec<'a> {
    pub id: Cow<'a, str>,
    pub labels: Vec<Cow<'a, str>>,
    pub props: Vec<(Cow<'a, str>, Value)>,
}

pub struct EdgeRec<'a> {
    pub src: Cow<'a, str>,
    pub dst: Cow<'a, str>,
    pub etype: Cow<'a, str>,
    pub props: Vec<(Cow<'a, str>, Value)>,
    /// Optional external string id. The dense edge index is the edge's canonical
    /// identity; this is an opt-in overlay (set by codecs that carry edge ids) so
    /// a user-assigned id survives a serialization round-trip. `None` ⇒ id-less.
    pub id: Option<Cow<'a, str>>,
}

impl NodeRec<'static> {
    /// A record that owns its strings — for a caller that synthesizes names
    /// rather than slicing them out of a document (fixtures, benches, and the
    /// codecs that unescape).
    pub fn owned(id: String, labels: Vec<String>, props: Vec<(String, Value)>) -> Self {
        Self {
            id: Cow::Owned(id),
            labels: owned_labels(labels),
            props: owned_props(props),
        }
    }
}

impl EdgeRec<'static> {
    /// The edge counterpart of [`NodeRec::owned`].
    pub fn owned(
        src: String,
        dst: String,
        etype: String,
        props: Vec<(String, Value)>,
        id: Option<String>,
    ) -> Self {
        Self {
            src: Cow::Owned(src),
            dst: Cow::Owned(dst),
            etype: Cow::Owned(etype),
            props: owned_props(props),
            id: id.map(Cow::Owned),
        }
    }
}

#[derive(Default)]
pub struct Builder<'a> {
    pub nodes: Vec<NodeRec<'a>>,
    pub edges: Vec<EdgeRec<'a>>,
}

/// Build a typed property store for `len` elements from `(index, props)` items.
/// A key's column type is inferred from its first non-null value; values that
/// disagree land in `Mixed` (lossless). Shared by the vertex and edge builds.
fn build_props(len: usize, items: &[PropItem], strs: &mut Dict) -> Properties {
    let mut props = Properties {
        keys: Dict::default(),
        cols: Vec::new(),
        len,
    };
    // Infer a kind per key (by dense key id) from its first non-null value.
    //
    // `kinds` is indexed by the dense key id rather than hashed on it, and each
    // property's id is remembered so the store pass below does not have to look
    // the key up a second time. Together those remove two of the three hashes a
    // property used to cost (intern, kind lookup, re-lookup) — on a 16-property
    // element that was 48 hashes where 16 will do.
    let mut kinds: Vec<Option<Kind>> = Vec::new();
    let mut kids: Vec<u32> = Vec::with_capacity(items.iter().map(|(_, it)| it.len()).sum());
    for (_, item) in items {
        for (k, v) in *item {
            let kid = props.keys.intern(k);
            kids.push(kid);
            if kinds.len() <= kid as usize {
                kinds.resize(kid as usize + 1, None);
            }
            if let Some(vk) = value_kind(v) {
                match &mut kinds[kid as usize] {
                    Some(cur) if *cur != vk => *cur = Kind::Mixed,
                    Some(_) => {}
                    slot @ None => *slot = Some(vk),
                }
            }
        }
    }
    // One column per interned key (dense by id); an all-null key gets an empty Mixed.
    props.cols = (0..props.keys.len() as u32)
        .map(|kid| empty_col_for_kind(kinds.get(kid as usize).copied().flatten(), len))
        .collect();
    let mut kid_at = kids.into_iter();
    for (idx, item) in items {
        for (_k, v) in *item {
            // Store every value, `Null` included — a present null promotes the
            // column to `Mixed` (mirrors `set_value`; null is a first-class value).
            let kid = kid_at.next().expect("one key id recorded per property") as usize;
            let col = &mut props.cols[kid];
            if !col_set(col, *idx, v, strs) {
                *col = to_mixed(col, strs);
                col_set(col, *idx, v, strs);
            }
        }
    }
    props
}

impl Builder<'_> {
    /// Like [`finalize`](Self::finalize), but enforces a **declared-nodes**
    /// contract: every edge endpoint must be a declared node. Returns
    /// `MissingVertex` instead of silently fabricating a phantom vertex (the
    /// lenient `finalize` behavior, kept for streaming NDJSON where endpoints are
    /// legitimately created on demand). The JSON document codecs (pg-json,
    /// graphson) use this so a dangling edge is an error, mirroring the TS codecs.
    pub fn finalize_strict(self) -> CodeResult<Graph> {
        let declared: HashSet<&str> = self.nodes.iter().map(|n| n.id.as_ref()).collect();
        for e in &self.edges {
            let missing = if !declared.contains(e.src.as_ref()) {
                Some(&e.src)
            } else if !declared.contains(e.dst.as_ref()) {
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
        // Sized for the worst case (every endpoint a fresh id) so the id
        // dictionary never rehashes mid-build. Over-reserving costs one
        // allocation; growing costs a full rehash of everything interned so far,
        // about twenty times over on a 400k-element load.
        let mut vid = Dict::with_capacity(nodes.len() + edges.len() * 2);

        // (1) Dense indices: declared nodes first (in order), then edge endpoints.
        // The dense id each node landed on is kept, so the label pass below does
        // not have to hash the id string all over again to rediscover it.
        let node_vi: Vec<u32> = nodes.iter().map(|node| vid.intern(&node.id)).collect();
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
            nodes.iter().map(|nd| seen.insert(nd.id.as_ref())).collect()
        };
        // An edge id is looked up here to reject a duplicate, and again later to
        // build the external-id overlay — twice over the same strings, in two
        // separate tables. One pass does both: the overlay IS the duplicate
        // check, since an id already in it has been seen.
        //
        // This drops a `HashSet<&str>` sized to the edge count. Real graphs carry
        // several edges per node, so that table — and the million-plus hashes and
        // probes it takes — scales with the EDGE count, which is the dimension
        // that grows.
        let mut eid_fwd: HashMap<u32, Arc<str>> = HashMap::with_capacity(edges.len());
        let mut eid_rev: HashMap<Arc<str>, u32> = HashMap::with_capacity(edges.len());
        let kept_edges: Vec<&EdgeRec> = {
            let mut kept: Vec<&EdgeRec> = Vec::with_capacity(edges.len());

            for ed in &edges {
                if let Some(id) = &ed.id {
                    let arc: Arc<str> = Arc::from(id.as_ref());

                    match eid_rev.entry(arc) {
                        std::collections::hash_map::Entry::Occupied(_) => continue, // dup → drop
                        std::collections::hash_map::Entry::Vacant(slot) => {
                            let idx = kept.len() as u32;

                            eid_fwd.insert(idx, slot.key().clone());
                            slot.insert(idx);
                        }
                    }
                }
                // An id-less edge takes the canonical `e{index}` form and can
                // never collide, so it is kept without consulting the overlay.
                kept.push(ed);
            }

            kept
        };

        // (2) Labels: per-vertex list + inverted (label -> live vertices).
        //
        // Counted first, then filled. Growing these by pushing means every
        // vertex's label list allocates and the per-label bucket doubles its way
        // up — on a load where most nodes share one label that bucket reallocates
        // and copies its way to the full vertex count. One counting pass makes
        // every allocation below exact and final. Push ORDER is unchanged, which
        // is what the byte-identical output depends on.
        let mut labels = Dict::default();
        let mut label_ids: Vec<u32> =
            Vec::with_capacity(nodes.iter().map(|nd| nd.labels.len()).sum());
        let mut vlabel_count = vec![0u32; n];
        let mut label_count: Vec<u32> = Vec::new();
        for (idx, node) in nodes.iter().enumerate() {
            if !keep_node[idx] {
                continue; // first-wins: ignore a duplicate node id's labels
            }
            let vi = node_vi[idx];
            for l in &node.labels {
                let lid = labels.intern(l);
                label_ids.push(lid);
                vlabel_count[vi as usize] += 1;
                if label_count.len() <= lid as usize {
                    label_count.resize(lid as usize + 1, 0);
                }
                label_count[lid as usize] += 1;
            }
        }

        let mut vlabels: Vec<Vec<u32>> = vlabel_count
            .iter()
            .map(|&c| Vec::with_capacity(c as usize))
            .collect();
        let mut by_label: HashMap<u32, Vec<u32>> = HashMap::with_capacity(label_count.len());
        for (lid, &c) in label_count.iter().enumerate() {
            if c > 0 {
                by_label.insert(lid as u32, Vec::with_capacity(c as usize));
            }
        }

        let mut label_at = label_ids.into_iter();
        for (idx, node) in nodes.iter().enumerate() {
            if !keep_node[idx] {
                continue;
            }
            let vi = node_vi[idx];
            for _ in &node.labels {
                let lid = label_at.next().expect("one label id recorded per label");
                vlabels[vi as usize].push(lid);
                by_label.entry(lid).or_default().push(vi);
            }
        }

        // (3) Vertex property columns. `strs` is graph-wide, shared with edges.
        let mut strs = Dict::default();
        let node_items: Vec<PropItem> = nodes
            .iter()
            .enumerate()
            .filter(|(idx, _)| keep_node[*idx])
            .map(|(idx, nd)| (node_vi[idx] as usize, nd.props.as_slice()))
            .collect();
        let props = build_props(n, &node_items, &mut strs);

        // (4) Edges: parallel arrays + per-vertex out/in adjacency.
        let mut etype = Dict::default();
        let e = kept_edges.len();
        let mut e_src = vec![0u32; e];
        let mut e_dst = vec![0u32; e];
        let mut e_type = vec![0u32; e];
        // Endpoints and type resolved ONCE, then counted, then filled. The old
        // single pass pushed into `out[s]`/`in_[d]`, so every vertex touched by an
        // edge allocated an adjacency list and doubled it as its degree grew — a
        // hub reallocating and copying its way up. Resolving first also means the
        // id dictionary is consulted once per endpoint instead of once per pass.
        let resolved: Vec<(u32, u32, u32)> = kept_edges
            .iter()
            .map(|ed| {
                (
                    vid.get(&ed.src).unwrap(),
                    vid.get(&ed.dst).unwrap(),
                    etype.intern(&ed.etype),
                )
            })
            .collect();
        let mut out_deg = vec![0u32; n];
        let mut in_deg = vec![0u32; n];
        let mut etype_count: Vec<u32> = Vec::new();
        for &(sv, dv, t) in &resolved {
            out_deg[sv as usize] += 1;
            in_deg[dv as usize] += 1;
            if etype_count.len() <= t as usize {
                etype_count.resize(t as usize + 1, 0);
            }
            etype_count[t as usize] += 1;
        }

        let mut out: Vec<Vec<Adj>> = out_deg
            .iter()
            .map(|&d| Vec::with_capacity(d as usize))
            .collect();
        let mut in_: Vec<Vec<Adj>> = in_deg
            .iter()
            .map(|&d| Vec::with_capacity(d as usize))
            .collect();
        let mut by_etype: HashMap<u32, Vec<u32>> = HashMap::with_capacity(etype_count.len());
        for (t, &c) in etype_count.iter().enumerate() {
            if c > 0 {
                by_etype.insert(t as u32, Vec::with_capacity(c as usize));
            }
        }
        for i in 0..kept_edges.len() {
            let (s, d, t) = resolved[i];
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
        }

        // (5) Edge property columns — same machinery, indexed by edge index.
        let edge_items: Vec<PropItem> = kept_edges
            .iter()
            .enumerate()
            .map(|(i, ed)| (i, ed.props.as_slice()))
            .collect();
        let edge_props = build_props(e, &edge_items, &mut strs);

        Graph {
            n,
            live_n: n,
            v_live: vec![true; n],
            names_checked: [0; 4],
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
            edge_interval_idxs: Vec::new(),
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
            config: GraphConfig::default(),
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

// Constraint, validator, invariant, and schema-dump methods (a second `impl Graph`).
mod constraints;
mod transactions;

#[cfg(test)]
mod tests;
