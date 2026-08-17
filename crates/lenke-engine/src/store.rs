//! A minimal typed columnar graph store — enough for the first execution slice
//! (scan a label, read a property, filter, project). Nodes only for now; edges,
//! adjacency, indexes, and temporal columns join in later slices.
//!
//! Properties are stored in TYPED columns (`Column`), not boxed values, so a
//! numeric property arrives at the batch layer as an unboxed `f64` run. That is
//! the whole point of the columnar model, present from the first slice rather
//! than retrofitted.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::value::Value;

/// Identity hasher for dense `u32` keys (edge ids). Edge ids are assigned
/// sequentially, so hashing an id to itself spreads the SwissTable buckets nearly
/// perfectly while skipping SipHash's per-probe mixing — the edge-property map is
/// probed once per edge on every edge-property read, and that hashing dominated
/// (a `HashMap<u32,_>` probe measured ~65ns; identity hashing removes the mix).
/// `write` keeps a correct (if unused) byte fallback so a non-`u32` key can never
/// silently mis-hash.
#[derive(Default)]
pub struct U32Hasher(u64);
impl std::hash::Hasher for U32Hasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }
    fn write_u32(&mut self, n: u32) {
        self.0 = u64::from(n);
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}
type U32BuildHasher = std::hash::BuildHasherDefault<U32Hasher>;
/// One edge-property key's per-eid values, identity-hashed (see [`U32Hasher`]).
pub type EdgeMap = HashMap<u32, Value, U32BuildHasher>;

/// A `Value` ordered by the value contract's total order (`cmp_total`) — the key
/// type for a range index's `BTreeMap`. It DELEGATES to `cmp_total`; it does not
/// restate ordering. `Eq` is "compares equal under `cmp_total`", so values the
/// total order ties (two NaNs, `-0.0`/`0.0`) share a bucket, exactly as grouping
/// does.
#[derive(Clone)]
struct OrdVal(Value);

impl PartialEq for OrdVal {
    fn eq(&self, other: &Self) -> bool {
        crate::value::cmp_total(&self.0, &other.0) == Ordering::Equal
    }
}
impl Eq for OrdVal {}
impl PartialOrd for OrdVal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdVal {
    fn cmp(&self, other: &Self) -> Ordering {
        crate::value::cmp_total(&self.0, &other.0)
    }
}

/// A range index on one node property `key`: values ordered by `cmp_total` ->
/// node ids. Holds only NON-NULL present values (a null never passes a range
/// predicate — the operand makes it UNKNOWN — so excluding it keeps the seek in
/// step with a scan+filter).
#[derive(Clone)]
struct RangeIndex {
    key: String,
    map: BTreeMap<OrdVal, Vec<u32>>,
}

/// One indexed interval: an out-edge's `[lo, hi]` (read from two numeric edge
/// props at build time), plus the edge id and its neighbour. Copied inline so an
/// overlap seek never touches the boxed edge-property map — the whole point of the
/// index (the boxed post-filter is what `examples/interval_bench` measures as the
/// cost).
#[derive(Clone, Copy)]
struct Iv {
    lo: f64,
    hi: f64,
    eid: u32,
    nbr: u32,
}

/// The opt-in edge INTERVAL index for one `(lo_key, hi_key)` pair over OUT-edges.
/// Per source node its intervals are held BOTH sorted by `lo` ascending and by
/// `hi` ascending, so an overlap query `[qlo, qhi]` (an edge overlaps iff
/// `lo <= qhi AND hi >= qlo`) can SEED from whichever axis is more selective and
/// post-filter the other — the RI-tree-lite rule from the bitemporal index (seed
/// from the selective axis; never materialize and intersect both stabs).
#[derive(Clone)]
struct IntervalIndex {
    lo_key: String,
    hi_key: String,
    by_lo: Vec<Vec<Iv>>,
    by_hi: Vec<Vec<Iv>>,
}

/// A typed property column, indexed by node id. `present[i]` is false where node
/// `i` does not carry this property (reads as `Value::Null`). One variant per
/// value type; a heterogeneous property falls to `Gen`.
#[derive(Clone, Debug)]
pub enum Column {
    Num {
        data: Vec<f64>,
        present: Vec<bool>,
    },
    Str {
        data: Vec<Arc<str>>,
        present: Vec<bool>,
    },
    /// A DICTIONARY-encoded string column: `codes[i]` indexes `dict` (the distinct
    /// values, in first-appearance order), so a low-cardinality column (a `dept` of
    /// five values across 100k rows) stores one small `dict` plus a `u32` per row.
    /// DISTINCT / GROUP BY / equality over it dedup and match on the `u32` code
    /// instead of hashing string content — that is the whole point. Produced ONLY by
    /// [`materialize`] when the cardinality stays under a cap; reads yield the SAME
    /// `Value::Str` as an equivalent `Str` column, and any typed write decodes it
    /// back to `Str` first (writes are the cold path), so it is a pure read encoding.
    Dict {
        dict: Vec<Arc<str>>,
        codes: Vec<u32>,
        present: Vec<bool>,
    },
    Bool {
        data: Vec<bool>,
        present: Vec<bool>,
    },
    /// A homogeneous temporal column: every present value is the SAME kind
    /// (`kind`). A temporal of a DIFFERENT kind — or a non-temporal — written to it
    /// promotes it to `Gen`, matching lenke-core's one-kind-per-column model.
    Temporal {
        kind: crate::temporal::TemporalKind,
        data: Vec<crate::temporal::Temporal>,
        present: Vec<bool>,
    },
    Gen {
        data: Vec<Value>,
        present: Vec<bool>,
    },
}

impl Column {
    fn with_capacity_num(n: usize) -> Self {
        Self::Num {
            data: vec![0.0; n],
            present: vec![false; n],
        }
    }

    /// Read node `i`'s value from this column (NULL if absent). The per-node
    /// accessor operators use when they hold a `&Column` directly.
    #[must_use]
    pub fn read(&self, i: usize) -> Value {
        let idx = i;
        match self {
            Self::Num { data, present } if present[idx] => Value::Num(data[idx]),
            Self::Str { data, present } if present[idx] => Value::Str(data[idx].clone()),
            Self::Dict {
                dict,
                codes,
                present,
            } if present[idx] => Value::Str(dict[codes[idx] as usize].clone()),
            Self::Bool { data, present } if present[idx] => Value::Bool(data[idx]),
            Self::Temporal { data, present, .. } if present[idx] => Value::Temporal(data[idx]),
            Self::Gen { data, present } if present[idx] => data[idx].clone(),
            _ => Value::Null,
        }
    }

    /// The number of node slots this column holds (== `node_count`).
    fn len(&self) -> usize {
        match self {
            Self::Num { present, .. }
            | Self::Str { present, .. }
            | Self::Dict { present, .. }
            | Self::Bool { present, .. }
            | Self::Temporal { present, .. }
            | Self::Gen { present, .. } => present.len(),
        }
    }

    /// A fresh column sized for `n` nodes, all absent, whose type matches `v`'s —
    /// or `Gen` for a value with no unboxed column form (`Null`, `List`).
    fn new_absent(v: &Value, n: usize) -> Self {
        match v {
            Value::Num(_) => Self::Num {
                data: vec![0.0; n],
                present: vec![false; n],
            },
            Value::Str(_) => Self::Str {
                data: vec![Arc::from(""); n],
                present: vec![false; n],
            },
            Value::Bool(_) => Self::Bool {
                data: vec![false; n],
                present: vec![false; n],
            },
            // A temporal seeds a homogeneous typed column of its kind; absent slots
            // hold the value itself as a harmless placeholder (present-gated).
            Value::Temporal(t) => Self::Temporal {
                kind: t.kind(),
                data: vec![*t; n],
                present: vec![false; n],
            },
            // Records/maps (and null/list) have no typed column form yet — Gen.
            Value::Null | Value::List(_) | Value::Record(_) | Value::Map(_) => Self::Gen {
                data: vec![Value::Null; n],
                present: vec![false; n],
            },
        }
    }

    /// Append one absent slot — what every existing column does when a node is
    /// added, so all columns stay length `node_count`.
    fn push_absent(&mut self) {
        match self {
            Self::Num { data, present } => {
                data.push(0.0);
                present.push(false);
            }
            Self::Str { data, present } => {
                data.push(Arc::from(""));
                present.push(false);
            }
            Self::Dict { codes, present, .. } => {
                codes.push(0);
                present.push(false);
            }
            Self::Bool { data, present } => {
                data.push(false);
                present.push(false);
            }
            Self::Temporal {
                kind,
                data,
                present,
            } => {
                data.push(kind.zero());
                present.push(false);
            }
            Self::Gen { data, present } => {
                data.push(Value::Null);
                present.push(false);
            }
        }
    }

    /// Whether this column can store `v` without a type change. A temporal column
    /// accepts only its OWN kind (a different kind promotes to `Gen`).
    fn accepts(&self, v: &Value) -> bool {
        match (self, v) {
            (Self::Num { .. }, Value::Num(_))
            | (Self::Str { .. }, Value::Str(_))
            | (Self::Dict { .. }, Value::Str(_))
            | (Self::Bool { .. }, Value::Bool(_))
            | (Self::Gen { .. }, _) => true,
            (Self::Temporal { kind, .. }, Value::Temporal(t)) => t.kind() == *kind,
            _ => false,
        }
    }

    /// Rebuild as a `Gen` column preserving present values — the promotion a typed
    /// column undergoes when a value of another type is written to it.
    fn to_gen(&self) -> Self {
        let n = self.len();
        let mut data = vec![Value::Null; n];
        let mut present = vec![false; n];
        for i in 0..n {
            if self.present_at(i) {
                data[i] = self.read(i);
                present[i] = true;
            }
        }
        Self::Gen { data, present }
    }

    /// Whether node `i` has a present (non-NULL) value in this column.
    #[must_use]
    pub fn present_at(&self, i: usize) -> bool {
        match self {
            Self::Num { present, .. }
            | Self::Str { present, .. }
            | Self::Dict { present, .. }
            | Self::Bool { present, .. }
            | Self::Temporal { present, .. }
            | Self::Gen { present, .. } => present[i],
        }
    }

    /// Replace a `Dict` encoding with the equivalent `Str` column in place — the
    /// one-time cost a dictionary-encoded column pays on its first value write, so
    /// every mutator below can assume the plain `Str` representation. A no-op on any
    /// other variant.
    fn decode_dict(&mut self) {
        let Self::Dict {
            dict,
            codes,
            present,
        } = self
        else {
            return;
        };
        let dict = std::mem::take(dict);
        let codes = std::mem::take(codes);
        let present = std::mem::take(present);
        let data: Vec<Arc<str>> = codes
            .iter()
            .zip(&present)
            .map(|(&c, &p)| {
                if p {
                    dict[c as usize].clone()
                } else {
                    Arc::from("")
                }
            })
            .collect();
        *self = Self::Str { data, present };
    }

    /// Dictionary-encode this column in place if it is a low-cardinality `Str`
    /// column (a categorical `city`/`dept`/`status`) — the forward of [`decode_dict`].
    /// A no-op on any other variant, or a `Str` column too high-cardinality to pay
    /// (see [`dict_encode`]). Incremental `add_node_with_id` builds plain `Str`
    /// columns; a bulk loader runs this once so categorical columns get the same
    /// code-based encoding the `materialize` path already gives, which turns GROUP BY
    /// / DISTINCT / equality over them into `u32`-code work instead of string hashing.
    fn try_dict_encode(&mut self) {
        if let Self::Str { data, present } = self {
            let data = std::mem::take(data);
            let present = std::mem::take(present);
            *self = dict_encode(data, present)
                .unwrap_or_else(|(data, present)| Self::Str { data, present });
        }
    }

    /// Set node `i` to `v`, marking it present. The caller guarantees the column
    /// `accepts` `v` (promoting to `Gen` first if not).
    fn set(&mut self, i: usize, v: Value) {
        // A dictionary column is a read encoding — decode to `Str` before mutating,
        // so it never has to grow its dict on the write path.
        self.decode_dict();
        match (self, v) {
            (Self::Num { data, present }, Value::Num(x)) => {
                data[i] = x;
                present[i] = true;
            }
            (Self::Str { data, present }, Value::Str(s)) => {
                data[i] = s;
                present[i] = true;
            }
            (Self::Bool { data, present }, Value::Bool(b)) => {
                data[i] = b;
                present[i] = true;
            }
            (Self::Temporal { data, present, .. }, Value::Temporal(t)) => {
                data[i] = t;
                present[i] = true;
            }
            (Self::Gen { data, present }, v) => {
                data[i] = v;
                present[i] = true;
            }
            _ => unreachable!("column must accept the value (promote to Gen first)"),
        }
    }

    /// Mark node `i` absent — a removed property reads as NULL again.
    fn set_absent(&mut self, i: usize) {
        match self {
            Self::Num { present, .. }
            | Self::Str { present, .. }
            | Self::Dict { present, .. }
            | Self::Bool { present, .. }
            | Self::Temporal { present, .. }
            | Self::Gen { present, .. } => present[i] = false,
        }
    }

    /// Drop the last node slot — the inverse of `push_absent`, used when a
    /// transaction rolls back the node that a logged `add_node` appended.
    fn pop_last(&mut self) {
        match self {
            Self::Num { data, present } => {
                data.pop();
                present.pop();
            }
            Self::Str { data, present } => {
                data.pop();
                present.pop();
            }
            Self::Dict { codes, present, .. } => {
                codes.pop();
                present.pop();
            }
            Self::Bool { data, present } => {
                data.pop();
                present.pop();
            }
            Self::Temporal { data, present, .. } => {
                data.pop();
                present.pop();
            }
            Self::Gen { data, present } => {
                data.pop();
                present.pop();
            }
        }
    }
}

/// One entry in a transaction's undo log — the inverse of a single mutation,
/// captured with just enough prior state to reverse it exactly. Applied in
/// reverse order on rollback, with logging disabled so undos do not re-log.
/// One change a committed transaction made — the unit of the observation-only CDC
/// stream. Recorded alongside the undo log (1:1 with each mutation) and handed to
/// observers AFTER commit, so it can never veto a write. Ids reference the store's
/// dense node ids / monotonic edge ids.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    NodeAdded(u32),
    NodeDeleted(u32),
    NodeProp { node: u32, key: String },
    EdgeAdded(u32),
    EdgeDeleted(u32),
    EdgeProp { eid: u32, key: String },
}

#[derive(Clone)]
enum Undo {
    /// Undo `add_node`: pop the last (highest-id) node. Adds grow `node_count`
    /// monotonically, so reverse-order undo always pops the current top.
    AddNode,
    /// Undo `add_edge`: delete the edge by its id from both endpoints.
    AddEdge { u: u32, v: u32, eid: u32 },
    /// Undo `set_prop`/`remove_prop`: restore the cell to its prior state.
    RestoreCell {
        node: u32,
        key: String,
        prev_present: bool,
        prev_value: Value,
    },
    /// Undo `set_edge_prop`/`remove_edge_prop`: restore the edge cell (`None` =
    /// was absent).
    RestoreEdgeCell {
        eid: u32,
        key: String,
        prev: Option<Value>,
    },
    /// Undo `delete_edge`: re-insert the exact adjacency entries removed, each
    /// tagged `(node, is_out, adj)` so it goes back to the right list.
    RestoreEdge { entries: Vec<(u32, bool, Adj)> },
    /// Undo `delete_node`: un-tombstone and restore its adjacency (both its own
    /// lists and the neighbours' mirrors), label memberships, and properties.
    RestoreNode {
        id: u32,
        out: Vec<Adj>,
        inc: Vec<Adj>,
        labels: Vec<String>,
        props: Vec<(String, bool, Value)>,
    },
}

/// One adjacency entry: the neighbour node, the edge's interned type id, and the
/// edge's identity (`eid`). A directed edge appears once in its source's `out`
/// and once in its target's `in`, both with the SAME `eid` — so trail semantics
/// (no edge reused within one path) can dedup on `eid` regardless of direction.
#[derive(Clone, Copy, Debug)]
pub struct Adj {
    pub nbr: u32,
    pub etype: u32,
    pub eid: u32,
}

/// A scalar / list / open-record property type for a TYPE constraint. The names
/// mirror the scalar set lenke-core's `PropType` accepts (its closed-record
/// `record { … }` specs are a later addition here). `AnyRecord` is `any record`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PropType {
    String,
    Number,
    Boolean,
    Date,
    LocalTime,
    DateTime,
    ZonedTime,
    ZonedDateTime,
    Duration,
    List,
}

impl PropType {
    /// Parse a scalar type keyword. `None` for anything else (a record keyword is
    /// handled by [`TypeSpec::parse`]).
    fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "string" => Self::String,
            "number" => Self::Number,
            "boolean" => Self::Boolean,
            "date" => Self::Date,
            "localtime" => Self::LocalTime,
            "datetime" => Self::DateTime,
            "zoned_time" => Self::ZonedTime,
            "zoned_datetime" => Self::ZonedDateTime,
            "duration" => Self::Duration,
            "list" => Self::List,
            _ => return None,
        })
    }
    fn to_name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::Date => "date",
            Self::LocalTime => "localtime",
            Self::DateTime => "datetime",
            Self::ZonedTime => "zoned_time",
            Self::ZonedDateTime => "zoned_datetime",
            Self::Duration => "duration",
            Self::List => "list",
        }
    }
}

/// A declared property TYPE: a scalar, a closed `record { field :: type, … }` (an
/// exact field set, each field itself a `TypeSpec`, so records nest), or the open
/// `any record`. Mirrors lenke-core's `TypeSpec`.
#[derive(Clone, PartialEq, Eq)]
pub enum TypeSpec {
    Scalar(PropType),
    /// A closed record: sorted `(name, type, not_null)` fields. A field is optional
    /// (absent or null both satisfy it) unless `not_null`. Closed on extras.
    Record(Vec<(String, TypeSpec, bool)>),
    /// The open record type — matches any record value regardless of shape.
    AnyRecord,
}

impl TypeSpec {
    /// Parse a constraint type name (scalar / `record { … }` / `any record`),
    /// optionally suffixed with a top-level ` NOT NULL`. `None` on malformed input.
    fn parse_with_not_null(s: &str) -> Option<(Self, bool)> {
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

    /// The canonical name for the schema dump (round-trips through `parse`).
    fn to_name(&self) -> String {
        match self {
            Self::Scalar(t) => t.to_name().to_string(),
            Self::Record(fields) => {
                let inner: Vec<String> = fields
                    .iter()
                    .map(|(k, t, nn)| {
                        format!("{k}::{}{}", t.to_name(), if *nn { " NOT NULL" } else { "" })
                    })
                    .collect();
                format!("record{{{}}}", inner.join(","))
            }
            Self::AnyRecord => "any record".to_string(),
        }
    }
}

/// A tiny recursive-descent parser for a constraint type expression (ported from
/// lenke-core's `TypeParser`).
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
        if word.eq_ignore_ascii_case("any") {
            return self.eat_kw("record").then_some(TypeSpec::AnyRecord);
        }
        if word.eq_ignore_ascii_case("record") {
            self.skip_ws();
            return if self.i < self.s.len() && self.s[self.i] == b'{' {
                self.parse_record()
            } else {
                Some(TypeSpec::AnyRecord)
            };
        }
        PropType::from_name(&word).map(TypeSpec::Scalar)
    }
    fn parse_record(&mut self) -> Option<TypeSpec> {
        if !self.eat(b'{') {
            return None;
        }
        let mut fields: Vec<(String, TypeSpec, bool)> = Vec::new();
        self.skip_ws();
        if !self.eat(b'}') {
            loop {
                let name = self.ident()?;
                if !self.eat(b':') {
                    return None;
                }
                self.eat(b':'); // optional second colon (`::`)
                let ty = self.parse_type()?;
                let not_null = if self.eat_kw("not") {
                    if !self.eat_kw("null") {
                        return None;
                    }
                    true
                } else {
                    false
                };
                match fields.binary_search_by(|(k, _, _)| k.as_str().cmp(&name)) {
                    Ok(i) => fields[i] = (name, ty, not_null), // duplicate → last wins
                    Err(i) => fields.insert(i, (name, ty, not_null)),
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

/// The scalar [`PropType`] a value satisfies, or `None` for a value EXEMPT from a
/// scalar constraint (`Null` and a record/map). Mirrors core's `value_type`.
fn value_prop_type(v: &Value) -> Option<PropType> {
    Some(match v {
        Value::Null | Value::Record(_) | Value::Map(_) => return None,
        Value::Str(_) => PropType::String,
        Value::Num(_) => PropType::Number,
        Value::Bool(_) => PropType::Boolean,
        Value::List(_) => PropType::List,
        Value::Temporal(t) => match t.tag() {
            "localtime" => PropType::LocalTime,
            "datetime" => PropType::DateTime,
            "zoned_time" => PropType::ZonedTime,
            "zoned_datetime" => PropType::ZonedDateTime,
            "duration" => PropType::Duration,
            _ => PropType::Date,
        },
    })
}

/// Whether `v` satisfies `spec`. A top-level `Null` always passes here (the
/// property's own nullability is the separate `not_null` check, applied by the
/// caller); a record/map is exempt from a scalar type. A closed record is checked
/// closed-on-extras with each field optional unless `NOT NULL`. Mirrors core's
/// `value_matches`.
fn value_matches(v: &Value, spec: &TypeSpec) -> bool {
    if matches!(v, Value::Null) {
        return true;
    }
    match spec {
        TypeSpec::Scalar(ty) => value_prop_type(v).is_none_or(|got| got == *ty),
        TypeSpec::AnyRecord => matches!(v, Value::Record(_) | Value::Map(_)),
        TypeSpec::Record(fields) => {
            let Value::Record(pairs) = v else {
                return false;
            };
            // No extra fields: every present key must be a declared field.
            if pairs.iter().any(|(vk, _)| {
                fields
                    .binary_search_by(|(fk, _, _)| fk.as_str().cmp(vk))
                    .is_err()
            }) {
                return false;
            }
            fields.iter().all(|(fk, ft, not_null)| {
                match pairs.binary_search_by(|(vk, _)| vk.as_ref().cmp(fk.as_str())) {
                    Ok(i) => {
                        let fv = &pairs[i].1;
                        if matches!(fv, Value::Null) {
                            !not_null
                        } else {
                            value_matches(fv, ft)
                        }
                    }
                    Err(_) => !not_null,
                }
            })
        }
    }
}

/// Parse a type-constraint spec (scalar / `record{…}` / `any record`), optionally
/// suffixed ` NOT NULL`. `Err` (`E_INVALID_VALUE`) on malformed input, matching core.
fn parse_type_spec(spec: &str) -> Result<(TypeSpec, bool), String> {
    TypeSpec::parse_with_not_null(spec).ok_or_else(|| {
        "E_INVALID_VALUE: unknown or malformed type name for a type constraint".to_string()
    })
}

/// Whether a value participates in a unique/index set (a scalar). A null, list, or
/// record is exempt (mirrors core's index-backed edge uniqueness).
fn is_indexable(v: &Value) -> bool {
    matches!(
        v,
        Value::Str(_) | Value::Num(_) | Value::Bool(_) | Value::Temporal(_)
    )
}

/// A declared TYPE constraint: a property `key` on a vertex label OR edge type
/// (`target`) must be of `ty` (and non-null when `not_null`).
#[derive(Clone)]
struct TypeRule {
    target: String,
    key: String,
    ty: TypeSpec,
    not_null: bool,
}

/// A declared CARDINALITY constraint: a `label` vertex's degree of `etype` edges
/// in `direction` (0 = out, 1 = in) must lie in `min..=max` (`max: None` = ∞).
#[derive(Clone)]
struct CardRule {
    label: String,
    etype: String,
    direction: u8,
    min: u32,
    max: Option<u32>,
}

/// A declared VALIDATOR: a predicate `pred` (parsed from `src`, with the element
/// bound to `var`) that must not be definitely-false for any element carrying
/// `target` (a vertex label or edge type). SQL-`CHECK` semantics: only `false`
/// fails; `null`/unknown passes.
#[derive(Clone)]
struct ValidatorRule {
    target: String,
    var: String,
    src: String,
    /// The composed check queries (vertex + edge form of `MATCH (var:target) WHERE
    /// NOT (src) RETURN var LIMIT 1`) — a non-empty result is a violation. Built by
    /// the exec layer at declaration, run by it on every write.
    checks: Vec<crate::ir::Plan>,
}

/// A declared INVARIANT: a whole-graph GQL query (`plan`, from `src`) that must
/// not yield a boolean-`false` cell.
#[derive(Clone)]
struct InvariantRule {
    name: String,
    src: String,
    /// The parsed whole-graph query; run by the exec layer on every write — any
    /// boolean-`false` cell in the result is a violation.
    plan: crate::ir::Plan,
}

/// The graph. Nodes are dense ids `0..node_count`. Labels and properties are
/// looked up by name; a label bucket (`by_label`) is the seed for a scan.
/// Resource ceilings — ANTI-RUNAWAY bounds, not semantics. A query under the ceiling
/// behaves identically whatever the ceiling is; tripping one is a loud
/// `E_RESOURCE_EXHAUSTED`, never a truncated result. Mirrors `lenke-core`'s `GraphLimits`
/// (same fields, same defaults) so the columnar engine enforces the SAME guards at the
/// SAME thresholds as the shipped row engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphLimits {
    /// Ceiling on `range(start, end [, step])` element count.
    pub range: u64,
    /// Cap on total variable-length / `repeat` traversal rows a single expansion may
    /// emit — the guard against exponential blowup on a dense graph (core's `trail`).
    pub trail: u64,
    /// Ceiling on the intermediate frontier a fixed-length multi-segment scan may
    /// materialize before the trailing hop/LIMIT prunes it.
    pub intermediate: u64,
    /// Ceiling on operator-chain length (parser-applied; `E_SYNTAX` when tripped).
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

/// Stable wire ids for [`GraphLimits`] knobs — the FFI limit setter is ONE export keyed
/// by id (append-only), matching core's `ConfigId` so the same host code drives both
/// engines.
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

/// Adjacency is per-node out/in lists (a simple layout for now; a CSR pack is a
/// later optimization that changes nothing above this module).
#[derive(Default)]
pub struct Store {
    node_count: usize,
    /// Resource ceilings (default [`GraphLimits::default`]); set via [`Store::set_limit`].
    limits: GraphLimits,
    /// label name -> the sorted node ids carrying it (the scan seed).
    by_label: HashMap<String, Vec<u32>>,
    /// property name -> its typed column (length == node_count).
    props: HashMap<String, Column>,
    /// Cached SORTED property-key list for element-map materialization. Property keys
    /// are only ever ADDED to `props` (a removed value keeps its column), so `props.len()`
    /// changing is exactly the key SET changing — the cache self-invalidates on a length
    /// mismatch, needing no write-path hook. Avoids the per-node `prop_keys()` clone+sort.
    prop_keys_cache: std::sync::RwLock<(usize, std::sync::Arc<[std::sync::Arc<str>]>)>,
    /// Cached forward node→min-label map (see `min_label_map`), keyed on `node_count`.
    /// Labels are immutable once a node is created and `by_label` only grows as nodes
    /// are added, so a `node_count` mismatch is exactly "a node was added" — the one
    /// event that can change the map — and the cache self-invalidates with no write hook.
    #[allow(clippy::type_complexity)]
    min_label_cache:
        std::sync::RwLock<Option<(usize, std::sync::Arc<(Vec<std::sync::Arc<str>>, Vec<u32>)>)>>,
    /// edge-type name -> interned id, and the reverse.
    etype_ids: HashMap<String, u32>,
    /// per-node outgoing / incoming adjacency, indexed by node id.
    out_adj: Vec<Vec<Adj>>,
    in_adj: Vec<Vec<Adj>>,
    /// next edge id to hand out — monotonic, so an out/in pair shares one id and
    /// ids stay unique across incremental writes.
    next_eid: u32,
    /// edge type id per eid (indexed by eid; grows 1:1 with `next_eid`, never
    /// shrinks — a deleted eid's entry lingers, safe since eids are never reused).
    /// The reverse of an `Adj`'s `etype`, needed by `type(edge)` which has only an
    /// eid in hand, not an adjacency entry.
    edge_etype: Vec<u32>,
    /// Per-edge (by eid) flag: does this edge carry SECONDARY labels? Lets a type
    /// filter skip the `edge_extra` HashMap probe for the overwhelming majority of
    /// edges (single-label) whose primary type simply did not match — a scattered
    /// `bool` read instead of a SipHash-of-`u32` + bucket chase. Parallel to
    /// `edge_etype` (grows with it, never shrinks). Empty ⇒ treated as all-false.
    edge_has_extra: Vec<bool>,
    /// SECONDARY edge labels (eid -> the labels past the first), mirroring core's
    /// `e_extra`. An edge's *type* is its first label (`edge_etype`); a multi-label
    /// edge — `-[:X:Y]->` / an ndjson `"labels":["X","Y"]` — carries the rest here.
    /// SPARSE: empty unless some edge has >1 label, so a single-label graph pays
    /// nothing and `edge_has_label` stays the single `u32` compare it replaced.
    edge_extra: HashMap<u32, Vec<u32>>,
    /// `(src, dst)` node ids per eid (indexed by eid; grows 1:1 with `next_eid` and
    /// never shrinks, like `edge_etype`). Lets an edge be rendered as core's
    /// `{id, from, to, labels, properties}` map from its eid alone, without scanning
    /// adjacency for its endpoints.
    edge_ends: Vec<(u32, u32)>,
    /// PRESERVED external ids — the stable, user-facing identity of each element,
    /// carried verbatim through ingest → store → egress (and returned by
    /// `element_id`). Ingest uses the id from the file; a created element (INSERT /
    /// addV / addE / Builder) is minted one. Indexed by dense node id / eid.
    node_ext: Vec<Arc<str>>,
    edge_ext: Vec<Arc<str>>,
    /// external node id → dense id, for resolving edge endpoints on ingest and
    /// looking a node up by its stable id. A tombstoned node keeps its entry.
    ext_to_node: HashMap<Arc<str>, u32>,
    /// tombstones, indexed by node id. A deleted node keeps its id slot (ids are
    /// dense and never reused) but is skipped by every scan and carries no edges
    /// or properties. `deleted.len() == node_count`.
    deleted: Vec<bool>,
    /// the active transaction's undo log, or `None` outside a transaction
    /// (autocommit — mutations apply directly and record nothing).
    undo: Option<Vec<Undo>>,
    /// the active transaction's change list (observation-only CDC), `Some` exactly
    /// when a transaction is open; moved to `last_commit` on commit, dropped on
    /// rollback. Grows 1:1 with the undo log.
    changes: Option<Vec<Change>>,
    /// the change list of the MOST RECENT committed transaction — what an observer
    /// reads after a write. Empty until the first commit.
    last_commit: Vec<Change>,
    /// declared unique constraints as `(label, keys)` — at most one live node per
    /// label may carry a given key tuple. Enforced by the write statements, not
    /// the store primitives (which stay infallible for rollback).
    unique: Vec<(String, Vec<String>)>,
    /// declared required-property constraints as `(label, key)` — every live node
    /// with `label` must carry a PRESENT value for `key` (present-null counts, per
    /// the null-first-class policy; only absence violates). Enforced by the write
    /// statements, like `unique`.
    required: Vec<(String, String)>,
    /// declared edge unique constraints as `(edge type, keys)` — at most one live
    /// edge of the type may carry a given key tuple (null/list values exempt).
    e_unique: Vec<(String, Vec<String>)>,
    /// declared edge required constraints as `(edge type, key)`.
    e_required: Vec<(String, String)>,
    /// declared vertex TYPE constraints (scalar/list/open-record + NOT NULL).
    v_type: Vec<TypeRule>,
    /// declared edge TYPE constraints (`target` is the edge type).
    e_type: Vec<TypeRule>,
    /// declared cardinality constraints (edge-degree bounds per vertex).
    cardinality: Vec<CardRule>,
    /// declared validators (a per-element predicate; `target` is a label or type).
    validators: Vec<ValidatorRule>,
    /// declared invariants (a whole-graph query that must hold after a write).
    invariants: Vec<InvariantRule>,
    /// edge properties: key -> (eid -> value). Boxed (not columnar) — edges are a
    /// less hot path than node scans, and eids are sparse after deletes. A deleted
    /// edge's props are left behind (eids are never reused, so a dead eid is never
    /// read); reclaiming them is a later tidy.
    edge_props: HashMap<String, EdgeMap>,
    /// hash indexes on a node property `key`: value's group-key bytes -> node ids
    /// (any label; the seek intersects with the label). Maintained on writes
    /// through the primitives, so a transaction rollback (which replays the
    /// primitives) keeps them consistent.
    indexes: Vec<Index>,
    /// range indexes on a node property `key` (ordered by `cmp_total`).
    ranges: Vec<RangeIndex>,
    /// OPT-IN edge-type index: `edge_type_index` is false and the vectors empty
    /// unless [`create_edge_type_index`](Store::create_edge_type_index) was called
    /// (so a graph that does not need it pays nothing — see `examples/expand_bench`
    /// for why the win is confined to high-degree, many-type nodes). When on,
    /// `out_type_idx[node]`/`in_type_idx[node]` map an edge-type id to that node's
    /// adjacency of that type, so a type-filtered hop seeks the bucket instead of
    /// scanning the whole adjacency. Kept correct across writes/deletes/rollback by
    /// a per-node rebuild whenever a node's flat adjacency changes (the flat lists
    /// stay the source of truth), with an O(1) push on the `add_edge` hot path.
    edge_type_index: bool,
    out_type_idx: Vec<HashMap<u32, Vec<Adj>>>,
    in_type_idx: Vec<HashMap<u32, Vec<Adj>>>,
    /// OPT-IN edge interval index (`None` unless `create_interval_index` was
    /// called). Built from edge props at creation and maintained through the
    /// mutation primitives; an interval-key edge-prop change triggers a full
    /// rebuild (rare — intervals are typically bulk-loaded before the index).
    interval: Option<IntervalIndex>,
    /// CSR READ OVERLAY of the adjacency. `out_adj`/`in_adj` (per-node `Vec`s) stay
    /// the source of truth for writes/rollback; this flattens them into one
    /// contiguous array per direction (offset `off[v]..off[v+1]` is node `v`'s
    /// slice) so a traversal streams cache-friendly memory instead of chasing a
    /// scattered `Vec` pointer per node. Built at load and rebuilt on demand; any
    /// adjacency write clears `csr_fresh`, and `out`/`inc` then fall back to the
    /// per-node `Vec`s (correct, just no CSR speedup until the next rebuild). The
    /// flat arrays are built in `out_adj` order, so a CSR slice is byte-identical to
    /// the `Vec` slice — order-sensitive summations/traversals are unaffected.
    csr_out_off: Vec<u32>,
    csr_out: Vec<Adj>,
    csr_in_off: Vec<u32>,
    csr_in: Vec<Adj>,
    csr_fresh: bool,
    /// PER-TYPE CSR partition of the adjacency, indexed by etype id: `csr_out_typed[t]`
    /// is `(off, adj)` where node `v`'s type-`t` OUT edges are `adj[off[v]..off[v+1]]`,
    /// in the SAME order as they appear in `out_adj[v]` (so a single-type hop iterates a
    /// contiguous slice byte-identically to the flat scan filtering `etype == t`, but
    /// touches only the matching edges — a sparse type no longer pays for the dense
    /// types' degree). Built alongside the flat CSR in `rebuild_csr` and gated by the
    /// same `csr_fresh`; a stale overlay makes the accessor return `None` and the caller
    /// falls back to the flat scan. Untyped/disjunction hops keep the flat CSR (its
    /// out_adj order, which a type-grouped concat would not preserve).
    csr_out_typed: Vec<(Vec<u32>, Vec<Adj>)>,
    csr_in_typed: Vec<(Vec<u32>, Vec<Adj>)>,
    /// TYPED READ OVERLAY of the numeric edge properties. `edge_props` (the boxed
    /// eid→Value maps) stays the source of truth for writes / egress / codecs /
    /// rollback; this densifies each HOMOGENEOUSLY-NUMERIC edge key into a raw
    /// `Vec<f64>` + present bitset indexed by eid, so an edge-property filter /
    /// projection reads a contiguous `f64` (no per-edge hash probe + `Value` unbox —
    /// the cost that dominated `edge/wfilter`/`wproj`). Built at load and rebuilt on
    /// demand; any edge-property write clears `edge_num_fresh`, and readers then fall
    /// back to the boxed `edge_prop` (correct, just no speedup until the next
    /// rebuild). A key with any non-numeric present value is omitted (readers use the
    /// boxed path for it). Values are the SAME as `edge_prop`, so nothing observable
    /// changes — it is a pure read encoding, like the CSR overlay.
    edge_num: HashMap<String, (Vec<f64>, Vec<bool>)>,
    edge_num_fresh: bool,
    /// A monotonic mutation counter, bumped by every data mutation primitive (see
    /// [`Store::touch`]). NOT part of any query result, codec, or ordering — pure
    /// out-of-band metadata for host-side change detection (`useSyncExternalStore`),
    /// so it never affects cross-engine byte-identity. Read via [`Store::version`].
    version: u64,
    /// Per-TOKEN change epochs: a label / edge-type / property-key name → the
    /// [`version`](Store::version) at which a change last touched it. The FINE
    /// invalidation signal behind global `version` — a React subscription watches
    /// only the tokens its query reads, so an unrelated mutation does not
    /// re-render it. Also metadata (never in a result), read via [`Store::epoch`].
    epochs: HashMap<String, u64>,
}

// Store holds two derived caches behind `RwLock` (which is not `Clone`), so `Clone`
// is hand-written: every data field is cloned; the caches clone their current value
// (identical `props` / `node_count` keep it valid). A missing field is a compile
// error, so this stays complete as the struct grows.
impl Clone for Store {
    fn clone(&self) -> Self {
        Self {
            node_count: self.node_count,
            limits: self.limits,
            by_label: self.by_label.clone(),
            props: self.props.clone(),
            prop_keys_cache: std::sync::RwLock::new(
                self.prop_keys_cache
                    .read()
                    .expect("prop_keys_cache poisoned")
                    .clone(),
            ),
            min_label_cache: std::sync::RwLock::new(
                self.min_label_cache
                    .read()
                    .expect("min_label_cache poisoned")
                    .clone(),
            ),
            etype_ids: self.etype_ids.clone(),
            out_adj: self.out_adj.clone(),
            in_adj: self.in_adj.clone(),
            next_eid: self.next_eid,
            edge_etype: self.edge_etype.clone(),
            edge_has_extra: self.edge_has_extra.clone(),
            edge_extra: self.edge_extra.clone(),
            edge_ends: self.edge_ends.clone(),
            node_ext: self.node_ext.clone(),
            edge_ext: self.edge_ext.clone(),
            ext_to_node: self.ext_to_node.clone(),
            deleted: self.deleted.clone(),
            undo: self.undo.clone(),
            changes: self.changes.clone(),
            last_commit: self.last_commit.clone(),
            unique: self.unique.clone(),
            required: self.required.clone(),
            e_unique: self.e_unique.clone(),
            e_required: self.e_required.clone(),
            v_type: self.v_type.clone(),
            e_type: self.e_type.clone(),
            cardinality: self.cardinality.clone(),
            validators: self.validators.clone(),
            invariants: self.invariants.clone(),
            edge_props: self.edge_props.clone(),
            indexes: self.indexes.clone(),
            ranges: self.ranges.clone(),
            edge_type_index: self.edge_type_index,
            out_type_idx: self.out_type_idx.clone(),
            in_type_idx: self.in_type_idx.clone(),
            interval: self.interval.clone(),
            csr_out_off: self.csr_out_off.clone(),
            csr_out: self.csr_out.clone(),
            csr_in_off: self.csr_in_off.clone(),
            csr_in: self.csr_in.clone(),
            csr_fresh: self.csr_fresh,
            csr_out_typed: self.csr_out_typed.clone(),
            csr_in_typed: self.csr_in_typed.clone(),
            edge_num: self.edge_num.clone(),
            edge_num_fresh: self.edge_num_fresh,
            version: self.version,
            epochs: self.epochs.clone(),
        }
    }
}

impl Store {
    /// Bump the mutation counter. Called by every data-mutation primitive.
    fn touch(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    /// Stamp `token`'s change epoch with the current version. Call AFTER `touch`
    /// (so `version` is already bumped) with each token a mutation names.
    fn bump_epoch(&mut self, token: &str) {
        let v = self.version;
        // Avoid a String alloc when the token is already tracked (the common case).
        if let Some(e) = self.epochs.get_mut(token) {
            *e = v;
        } else {
            self.epochs.insert(token.to_string(), v);
        }
    }

    /// The monotonic mutation version — changes on every data mutation, for
    /// host-side change detection. Not observable in query results (see the field).
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// The change epoch of one token (a label / edge-type / property-key name): the
    /// `version` at which a change last touched it, or 0 if never. A subscriber to
    /// `token` re-reads when this rises. See the `epochs` field.
    #[must_use]
    pub fn epoch(&self, token: &str) -> u64 {
        self.epochs.get(token).copied().unwrap_or(0)
    }

    /// Keys carrying a hash (equality) node index, each rendered as a dotted path.
    #[must_use]
    pub fn hash_index_keys(&self) -> Vec<String> {
        self.indexes.iter().map(|i| i.path.join(".")).collect()
    }

    /// Keys carrying a range (ordered) node index.
    #[must_use]
    pub fn range_index_keys(&self) -> Vec<String> {
        self.ranges.iter().map(|r| r.key.clone()).collect()
    }

    /// The edge interval index's `(lo_key, hi_key)`, if one exists.
    #[must_use]
    pub fn interval_index_keys(&self) -> Option<(String, String)> {
        self.interval
            .as_ref()
            .map(|iv| (iv.lo_key.clone(), iv.hi_key.clone()))
    }
}

/// A hash index on a node property PATH. `path` is `["age"]` for a plain property
/// or `["meta", "city"]` for a dotted record-field path; the index keys on the
/// value found by descending record fields (`resolve_path`). A plain index
/// (length-1 path) behaves exactly as before.
#[derive(Clone)]
struct Index {
    path: Vec<String>,
    map: HashMap<Vec<u8>, Vec<u32>>,
}

impl Index {
    /// The base (top-level) property whose column drives this index's upkeep.
    fn base(&self) -> &str {
        &self.path[0]
    }
}

/// Descend record fields `sub` from `v`; an empty `sub` returns `v` (a plain
/// index). A non-record along the way, or a missing field, resolves to `Null`.
fn resolve_path(v: &Value, sub: &[String]) -> Value {
    let mut cur = v.clone();
    for k in sub {
        cur = match &cur {
            Value::Record(fields) => crate::value::record_field(fields, k),
            _ => Value::Null,
        };
    }
    cur
}

impl Store {
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// The active resource ceilings (see [`GraphLimits`]).
    #[must_use]
    pub fn limits(&self) -> GraphLimits {
        self.limits
    }

    /// Override one resource ceiling, keyed by its stable [`ConfigId`] — the single entry
    /// point a host (or the FFI) uses to configure limits, matching core's keyed setter.
    pub fn set_limit(&mut self, id: ConfigId, value: u64) {
        match id {
            ConfigId::LimitsRange => self.limits.range = value,
            ConfigId::LimitsTrail => self.limits.trail = value,
            ConfigId::LimitsIntermediate => self.limits.intermediate = value,
            ConfigId::LimitsOperatorChain => self.limits.operator_chain = value,
        }
    }

    /// The interned id for an edge-type name, or `None` if no edge ever used it
    /// (so a hop on it matches nothing).
    #[must_use]
    pub fn etype_id(&self, name: &str) -> Option<u32> {
        self.etype_ids.get(name).copied()
    }

    /// Every interned edge-type id — the universe an edge-label NEGATION (`-[:!T]->`)
    /// complements against. Order is unspecified (a membership set for the caller).
    #[must_use]
    pub fn all_etype_ids(&self) -> Vec<u32> {
        self.etype_ids.values().copied().collect()
    }

    /// How many distinct edge types are interned. A hop whose wanted-type set has
    /// this many DISTINCT ids matches every edge, so a count can read raw degrees
    /// instead of type-checking each edge (see exec's `matching_degree`).
    #[must_use]
    pub fn num_etypes(&self) -> usize {
        self.etype_ids.len()
    }

    /// A node's outgoing adjacency.
    #[must_use]
    pub fn out(&self, node: u32) -> &[Adj] {
        if self.csr_fresh {
            let v = node as usize;
            if v + 1 < self.csr_out_off.len() {
                let (a, b) = (
                    self.csr_out_off[v] as usize,
                    self.csr_out_off[v + 1] as usize,
                );
                return &self.csr_out[a..b];
            }
            return &[];
        }
        self.out_adj.get(node as usize).map_or(&[], Vec::as_slice)
    }

    /// A node's incoming adjacency.
    #[must_use]
    pub fn inc(&self, node: u32) -> &[Adj] {
        if self.csr_fresh {
            let v = node as usize;
            if v + 1 < self.csr_in_off.len() {
                let (a, b) = (self.csr_in_off[v] as usize, self.csr_in_off[v + 1] as usize);
                return &self.csr_in[a..b];
            }
            return &[];
        }
        self.in_adj.get(node as usize).map_or(&[], Vec::as_slice)
    }

    /// (Re)build the CSR read overlay from the per-node adjacency, preserving each
    /// node's neighbour ORDER exactly (so a CSR slice equals the `Vec` slice). O(V+E)
    /// — called once after a load, and on demand after a batch of writes. Marks the
    /// overlay fresh, so subsequent `out`/`inc` use the contiguous arrays.
    pub fn rebuild_csr(&mut self) {
        let n = self.out_adj.len();
        self.csr_out_off.clear();
        self.csr_out.clear();
        self.csr_in_off.clear();
        self.csr_in.clear();
        self.csr_out_off.reserve(n + 1);
        self.csr_in_off.reserve(n + 1);
        self.csr_out_off.push(0);
        for adj in &self.out_adj {
            self.csr_out.extend_from_slice(adj);
            self.csr_out_off
                .push(u32::try_from(self.csr_out.len()).expect("edge count exceeds u32"));
        }
        self.csr_in_off.push(0);
        for adj in &self.in_adj {
            self.csr_in.extend_from_slice(adj);
            self.csr_in_off
                .push(u32::try_from(self.csr_in.len()).expect("edge count exceeds u32"));
        }
        // Per-type partitions: one (off, adj) per etype id, filled in out_adj/in_adj order.
        let ntypes = self.etype_ids.len();
        let build_typed = |adjs: &[Vec<Adj>]| -> Vec<(Vec<u32>, Vec<Adj>)> {
            let mut typed: Vec<(Vec<u32>, Vec<Adj>)> = (0..ntypes)
                .map(|_| {
                    let mut off = Vec::with_capacity(n + 1);
                    off.push(0u32);
                    (off, Vec::new())
                })
                .collect();
            for adj in adjs {
                for a in adj {
                    if let Some(t) = typed.get_mut(a.etype as usize) {
                        t.1.push(*a);
                    }
                }
                for t in &mut typed {
                    t.0.push(u32::try_from(t.1.len()).expect("edge count exceeds u32"));
                }
            }
            typed
        };
        self.csr_out_typed = build_typed(&self.out_adj);
        self.csr_in_typed = build_typed(&self.in_adj);
        self.csr_fresh = true;
    }

    /// Node `v`'s OUT edges of type `etype`, a contiguous slice in out_adj order — the
    /// per-type CSR fast path for a single-type hop. `None` when the overlay is stale (the
    /// caller falls back to the flat scan) or the etype id is unknown.
    #[inline]
    #[must_use]
    pub fn out_typed_csr(&self, v: u32, etype: u32) -> Option<&[Adj]> {
        if !self.csr_fresh {
            return None;
        }
        let (off, adj) = self.csr_out_typed.get(etype as usize)?;
        let i = v as usize;
        if i + 1 < off.len() {
            Some(&adj[off[i] as usize..off[i + 1] as usize])
        } else {
            Some(&[])
        }
    }

    /// ALL out-edges of type `etype` across every source, contiguous in (source, out_adj)
    /// order — the whole per-type partition. Lets a hop whose source set is the entire graph
    /// count/scan the type's edges directly (5x fewer than the node count for a sparse type)
    /// without walking every source's offset. `None` if the overlay is stale or the etype is
    /// unknown.
    #[inline]
    #[must_use]
    pub fn out_typed_flat(&self, etype: u32) -> Option<&[Adj]> {
        if !self.csr_fresh {
            return None;
        }
        self.csr_out_typed
            .get(etype as usize)
            .map(|(_, adj)| adj.as_slice())
    }

    /// Node `v`'s IN edges of type `etype`, a contiguous slice in in_adj order. The
    /// in-side twin of [`out_typed_csr`].
    #[inline]
    #[must_use]
    pub fn in_typed_csr(&self, v: u32, etype: u32) -> Option<&[Adj]> {
        if !self.csr_fresh {
            return None;
        }
        let (off, adj) = self.csr_in_typed.get(etype as usize)?;
        let i = v as usize;
        if i + 1 < off.len() {
            Some(&adj[off[i] as usize..off[i + 1] as usize])
        } else {
            Some(&[])
        }
    }

    /// Drop the CSR overlay (an adjacency write happened); `out`/`inc` fall back to
    /// the per-node `Vec`s until the next [`Self::rebuild_csr`].
    fn invalidate_csr(&mut self) {
        self.csr_fresh = false;
    }

    /// Every LIVE node id — the scan universe when no label narrows it. Deleted
    /// ids are skipped (tombstoned), so a whole-graph scan never yields them.
    #[must_use]
    pub fn all_nodes(&self) -> Vec<u32> {
        (0..self.node_count as u32)
            .filter(|&i| !self.deleted[i as usize])
            .collect()
    }

    /// Every LIVE edge id, in id order — the source for Gremlin `g.E()`. Each edge
    /// appears exactly once as an out-edge (mirroring core's `g.E()` == `g.V().outE()`
    /// desugar), and `delete_edge` retains-out the adjacency entry, so iterating
    /// `out_adj` already reflects liveness with no separate tombstone check.
    #[must_use]
    pub fn all_edges(&self) -> Vec<u32> {
        let mut eids: Vec<u32> = self
            .out_adj
            .iter()
            .flat_map(|adj| adj.iter().map(|a| a.eid))
            .collect();
        eids.sort_unstable();
        eids
    }

    /// Whether node `id` is live (not tombstoned). Scans that iterate the id space
    /// directly consult this to skip deleted nodes.
    #[must_use]
    pub fn is_alive(&self, id: u32) -> bool {
        !self.deleted[id as usize]
    }

    /// The number of LIVE nodes — `count(*)` over an unlabelled scan without
    /// materializing the id vector. O(n) over the tombstone bitmap (no allocation);
    /// deletions are rare, so the common all-live case is a fast bitmap sweep.
    #[must_use]
    pub fn live_node_count(&self) -> usize {
        self.node_count - self.deleted.iter().filter(|&&d| d).count()
    }

    /// The node ids carrying `label`, or an empty slice for an unknown label
    /// (which matches nothing — never "everything").
    #[must_use]
    pub fn nodes_with_label(&self, label: &str) -> &[u32] {
        self.by_label.get(label).map_or(&[], Vec::as_slice)
    }

    /// Whether node `id` carries `label`. A binary search of the label bucket, which
    /// is kept SORTED ascending (ids are pushed in increasing insertion order and
    /// `retain` on delete preserves order) — O(log |label|), so a per-node label
    /// test costs a search, not the whole-bucket scan `labels_of` does.
    #[must_use]
    pub fn is_labeled(&self, id: u32, label: &str) -> bool {
        self.by_label
            .get(label)
            .is_some_and(|bucket| bucket.binary_search(&id).is_ok())
    }

    /// A node's value for `key`, or `Null` if the key or the node's entry is
    /// absent. The typed read; the batch layer's bulk gather uses the column
    /// directly (see `column`).
    #[must_use]
    pub fn prop(&self, node: u32, key: &str) -> Value {
        self.props
            .get(key)
            .map_or(Value::Null, |c| c.read(node as usize))
    }

    /// Read a node's value at a (possibly dotted) property PATH: the base property,
    /// then descend record sub-fields. A plain key reads exactly like [`prop`]. The
    /// scan-fallback twin of a dotted [`index_lookup`].
    #[must_use]
    pub fn prop_path(&self, node: u32, dotted_key: &str) -> Value {
        let path: Vec<String> = dotted_key.split('.').map(String::from).collect();
        resolve_path(&self.prop(node, &path[0]), &path[1..])
    }

    /// The typed column for `key`, for a bulk gather. `None` = no such property.
    #[must_use]
    pub fn column(&self, key: &str) -> Option<&Column> {
        self.props.get(key)
    }

    // --- Enumeration (for egress / snapshot) -----------------------------

    /// All node-property keys, sorted — a deterministic field order for dumps.
    #[must_use]
    pub fn prop_keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.props.keys().cloned().collect();
        k.sort();
        k
    }

    /// The sorted property keys as shared `Arc<str>`, cached (see `prop_keys_cache`).
    /// The hot path for materializing node element maps — one refcount bump per node
    /// instead of cloning+sorting the whole key list each time.
    #[must_use]
    pub fn prop_keys_arc(&self) -> std::sync::Arc<[std::sync::Arc<str>]> {
        let len = self.props.len();
        {
            let g = self
                .prop_keys_cache
                .read()
                .expect("prop_keys_cache poisoned");
            if g.0 == len {
                return std::sync::Arc::clone(&g.1);
            }
        }
        let mut keys: Vec<std::sync::Arc<str>> = self
            .props
            .keys()
            .map(|k| std::sync::Arc::from(k.as_str()))
            .collect();
        keys.sort();
        let arc: std::sync::Arc<[std::sync::Arc<str>]> = keys.into();
        let mut g = self
            .prop_keys_cache
            .write()
            .expect("prop_keys_cache poisoned");
        *g = (len, std::sync::Arc::clone(&arc));
        arc
    }

    /// All edge-property keys, sorted.
    #[must_use]
    pub fn edge_prop_keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.edge_props.keys().cloned().collect();
        k.sort();
        k
    }

    /// The lexicographically SMALLEST label node `id` carries (Gremlin `label()` — a
    /// vertex's single label) as a BORROWED name, without allocating/sorting the whole
    /// label list the way `labels_of(id).first()` did. The caller interns the `Arc`
    /// (few distinct labels), so `V().label()` over a big frontier does one allocation
    /// per label, not per row. `min` via a bucket binary-search per label.
    #[must_use]
    pub fn min_label_name(&self, id: u32) -> Option<&str> {
        self.by_label
            .iter()
            .filter(|(_, ids)| ids.binary_search(&id).is_ok())
            .map(|(l, _)| l.as_str())
            .min()
    }

    /// A forward node → min-label mapping for the WHOLE store, built in one pass:
    /// `(names, code_of)` where `names` are the distinct labels interned once in
    /// ascending order and `code_of[id]` indexes them (`u32::MAX` = unlabelled).
    ///
    /// [`min_label_name`] answers one node by probing every label bucket with a
    /// binary search — O(labels · log n) per node with cache-hostile random access,
    /// which is the cost of a whole-frontier `V().label()`. Inverting the buckets
    /// once (a sequential scan per bucket, ascending label order so the FIRST bucket
    /// to claim a node is its min) turns that whole column into O(total membership)
    /// build + O(1) per row. Callers processing a large node frontier use this
    /// instead of the per-node probe.
    /// The map is a derived read-only index, so it is CACHED (keyed on `node_count`,
    /// see `min_label_cache`) — a repeated `V().label()` against a stable store rebuilds
    /// it once, not per call. Returned as an `Arc` so a cache hit is a refcount bump.
    #[must_use]
    pub fn min_label_map(&self) -> Arc<(Vec<Arc<str>>, Vec<u32>)> {
        let n = self.node_count();
        {
            let g = self
                .min_label_cache
                .read()
                .expect("min_label_cache poisoned");
            if let Some((cached_n, map)) = g.as_ref() {
                if *cached_n == n {
                    return Arc::clone(map);
                }
            }
        }
        let mut buckets: Vec<(&String, &Vec<u32>)> = self.by_label.iter().collect();
        buckets.sort_by(|a, b| a.0.cmp(b.0));
        let names: Vec<Arc<str>> = buckets.iter().map(|(l, _)| Arc::from(l.as_str())).collect();
        let mut code_of = vec![u32::MAX; n];
        for (code, (_, ids)) in buckets.iter().enumerate() {
            // Iterate the bucket in its native (ascending id) order — a sequential
            // read — and claim only nodes no earlier (smaller) label already took.
            for &id in ids.iter() {
                let slot = &mut code_of[id as usize];
                if *slot == u32::MAX {
                    *slot = code as u32;
                }
            }
        }
        let map = Arc::new((names, code_of));
        *self
            .min_label_cache
            .write()
            .expect("min_label_cache poisoned") = Some((n, Arc::clone(&map)));
        map
    }

    /// The labels carried by node `id`, sorted. Each label bucket is kept SORTED
    /// ascending (see `is_labeled`), so membership is a BINARY SEARCH — a linear
    /// `contains` here made per-node materialization (element maps in fold/valueMap/
    /// path/id) O(total label membership), which dominated those shapes.
    #[must_use]
    pub fn labels_of(&self, id: u32) -> Vec<String> {
        let mut ls: Vec<String> = self
            .by_label
            .iter()
            .filter(|(_, ids)| ids.binary_search(&id).is_ok())
            .map(|(l, _)| l.clone())
            .collect();
        ls.sort();
        ls
    }

    /// The declared unique constraints as `(label, keys)` — for snapshot/schema
    /// egress.
    #[must_use]
    pub fn unique_constraints(&self) -> Vec<(String, Vec<String>)> {
        self.unique.clone()
    }

    /// The interned edge-type id's name (the reverse of `etype_id`).
    #[must_use]
    pub fn etype_name(&self, etype: u32) -> Option<String> {
        self.etype_ids
            .iter()
            .find(|(_, &id)| id == etype)
            .map(|(name, _)| name.clone())
    }

    /// The type name of edge `eid` (for `type(edge)`), or `None` if the eid is out
    /// of range. Looks up the eid's interned etype then its name.
    #[must_use]
    pub fn edge_type_name(&self, eid: u32) -> Option<String> {
        let etype = *self.edge_etype.get(eid as usize)?;
        self.etype_name(etype)
    }

    /// All of edge `eid`'s labels — its type (first) then any secondary labels
    /// (multi-label edges), in that order.
    #[must_use]
    pub fn edge_labels_of(&self, eid: u32) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(t) = self.edge_type_name(eid) {
            out.push(t);
        }
        if let Some(extra) = self.edge_extra.get(&eid) {
            for &tid in extra {
                if let Some(name) = self.etype_name(tid) {
                    out.push(name);
                }
            }
        }
        out
    }

    /// A node's preserved external id (for `element_id`), or `None` out of range.
    #[must_use]
    pub fn node_ext_id(&self, id: u32) -> Option<Arc<str>> {
        self.node_ext.get(id as usize).cloned()
    }

    /// The `(src, dst)` node ids of edge `eid`, or `None` if the eid is unknown.
    #[must_use]
    pub fn edge_endpoints(&self, eid: u32) -> Option<(u32, u32)> {
        self.edge_ends.get(eid as usize).copied()
    }

    /// An edge's preserved external id (for `element_id`), or `None` out of range.
    #[must_use]
    pub fn edge_ext_id(&self, eid: u32) -> Option<Arc<str>> {
        self.edge_ext.get(eid as usize).cloned()
    }

    /// The dense id of the live node with external id `ext`, or `None`.
    #[must_use]
    pub fn node_by_ext(&self, ext: &str) -> Option<u32> {
        self.ext_to_node
            .get(ext)
            .copied()
            .filter(|&id| !self.deleted[id as usize])
    }

    // --- Unique constraints ----------------------------------------------
    //
    // A unique constraint declares that at most one live node with `label` may
    // carry a given tuple of `keys` values. Enforced by the write STATEMENTS
    // (execute) after a mutation, not by the store primitives — those stay
    // infallible so rollback can always run. Key equality uses the value
    // contract's grouping (`group_key_into`), so two absent/NULL keys collide
    // (consistent with lenke's first-class-null policy, not SQL's distinct NULLs).

    /// Declare a unique constraint on `(label, keys)`. Errors if the CURRENT data
    /// already violates it (you cannot declare a constraint the graph breaks).
    pub fn create_unique_constraint(&mut self, label: &str, keys: &[&str]) -> Result<(), String> {
        let keys: Vec<String> = keys.iter().map(|s| (*s).to_string()).collect();
        self.check_label_unique(label, &keys)?;
        self.unique.push((label.to_string(), keys));
        Ok(())
    }

    /// Check every unique constraint on `label` against the live nodes; `Err`
    /// names the first violated one. Write statements call this after mutating a
    /// constrained label.
    pub fn check_unique_for_label(&self, label: &str) -> Result<(), String> {
        for (l, keys) in &self.unique {
            if l == label {
                self.check_label_unique(l, keys)?;
            }
        }
        Ok(())
    }

    /// The keys of a unique constraint on `label` all of which appear in `have` —
    /// the `_MERGE` conflict-target inference. `None` if no such constraint.
    #[must_use]
    pub fn unique_keys_for(&self, label: &str, have: &[String]) -> Option<Vec<String>> {
        self.unique
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, keys)| keys)
            .find(|keys| keys.iter().all(|k| have.contains(k)))
            .cloned()
    }

    /// Infer the `_MERGE` conflict target: the one unique constraint on `label`
    /// whose keys are all present in `have`. Errors if there is none (the merge
    /// has no key) or more than one (ambiguous — the pattern must disambiguate).
    pub fn infer_merge_key(&self, label: &str, have: &[String]) -> Result<Vec<String>, String> {
        let covered: Vec<&Vec<String>> = self
            .unique
            .iter()
            .filter(|(l, keys)| l == label && keys.iter().all(|k| have.contains(k)))
            .map(|(_, keys)| keys)
            .collect();
        match covered.as_slice() {
            [] => Err(format!(
                "E_MERGE: _MERGE on `{label}` has no applicable unique constraint"
            )),
            [one] => Ok((*one).clone()),
            _ => Err(format!(
                "E_MERGE: _MERGE on `{label}` is ambiguous — the pattern touches several unique constraints"
            )),
        }
    }

    fn check_label_unique(&self, label: &str, keys: &[String]) -> Result<(), String> {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        for &id in self.nodes_with_label(label) {
            let mut buf = Vec::new();
            for k in keys {
                crate::value::group_key_into(&self.prop(id, k), &mut buf);
            }
            if !seen.insert(buf) {
                return Err(format!(
                    "E_UNIQUE: unique constraint on {label}({}) violated",
                    keys.join(", ")
                ));
            }
        }
        Ok(())
    }

    // --- Required-property constraints ------------------------------------
    //
    // A required constraint declares that every live node with `label` carries a
    // PRESENT value for `key` (present-null counts — only absence violates).
    // Enforced by the write statements after a mutation, like `unique`.

    /// The declared required constraints as `(label, key)` — for snapshot/schema.
    #[must_use]
    pub fn required_constraints(&self) -> Vec<(String, String)> {
        self.required.clone()
    }

    /// Declare a required-property constraint on `(label, key)`. Errors if the
    /// CURRENT data already violates it (a labelled node missing the property).
    pub fn create_required_constraint(&mut self, label: &str, key: &str) -> Result<(), String> {
        self.check_label_required(label, key)?;
        self.required.push((label.to_string(), key.to_string()));
        Ok(())
    }

    /// Check every required constraint on `label` against the live nodes; `Err`
    /// names the first violated one. Write statements call this after mutating a
    /// constrained label.
    pub fn check_required_for_label(&self, label: &str) -> Result<(), String> {
        for (l, key) in &self.required {
            if l == label {
                self.check_label_required(l, key)?;
            }
        }
        Ok(())
    }

    fn check_label_required(&self, label: &str, key: &str) -> Result<(), String> {
        for &id in self.nodes_with_label(label) {
            if !self.has_prop(id, key) {
                return Err(format!(
                    "E_REQUIRED: required constraint on {label}({key}) violated"
                ));
            }
        }
        Ok(())
    }

    // --- Deferred constraint checks (the write-path enforcement hook) ------
    //
    // A write STATEMENT runs inside a transaction (begin → mutate → commit); the
    // store primitives stay infallible so rollback can always replay them. After
    // the mutations, the statement calls `run_deferred_checks` ONCE — it derives
    // the touched set from the open transaction's CDC change list and re-checks
    // every constraint that could be affected, mirroring lenke-core's commit-time
    // `run_deferred_checks`. On the first violation it returns the coded message
    // and the caller rolls the whole statement back.

    /// Re-check every constraint the open transaction's changes could have
    /// violated. `Ok(())` outside a transaction or when nothing changed.
    pub fn run_deferred_checks(&self) -> Result<(), String> {
        let Some(changes) = &self.changes else {
            return Ok(());
        };
        if changes.is_empty() {
            return Ok(());
        }

        // Touched, still-live node labels + edge types (an add or a property write
        // can violate; a delete removes the element, so it cannot). First-seen
        // order, deduped. `edge_changed` gates the cardinality re-check.
        let mut labels: Vec<String> = Vec::new();
        let mut etypes: Vec<String> = Vec::new();
        let mut edge_changed = false;
        for c in changes {
            match c {
                Change::NodeAdded(n) | Change::NodeProp { node: n, .. } => {
                    if self.is_alive(*n) {
                        for l in self.labels_of(*n) {
                            if !labels.contains(&l) {
                                labels.push(l);
                            }
                        }
                    }
                }
                Change::EdgeAdded(eid) | Change::EdgeProp { eid, .. } => {
                    edge_changed = true;
                    if let Some(t) = self.edge_type_name(*eid) {
                        if !etypes.contains(&t) {
                            etypes.push(t);
                        }
                    }
                }
                Change::EdgeDeleted(_) => edge_changed = true,
                Change::NodeDeleted(_) => {}
            }
        }

        // Vertex constraints on every touched label.
        for l in &labels {
            self.check_unique_for_label(l)?;
            self.check_required_for_label(l)?;
            self.check_type_for_target(l, false)?;
        }
        // Edge constraints on every touched edge type.
        for t in &etypes {
            self.check_edge_unique_for_type(t)?;
            self.check_edge_required_for_type(t)?;
            self.check_type_for_target(t, true)?;
        }
        // Cardinality: an edge add/delete shifts degrees, and a new node of a
        // constrained label can break a MIN. Re-check every rule when either applies.
        if !self.cardinality.is_empty()
            && (edge_changed
                || labels
                    .iter()
                    .any(|l| self.cardinality.iter().any(|r| &r.label == l)))
        {
            self.check_all_cardinality()?;
        }
        // Validators and invariants (predicate / whole-graph query) need the query
        // evaluator, so exec runs them after this via `enforce_expr_constraints`.
        Ok(())
    }

    // --- Edge unique / required constraints -------------------------------

    /// The declared edge unique constraints as `(edge type, keys)`.
    #[must_use]
    pub fn edge_unique_constraints(&self) -> Vec<(String, Vec<String>)> {
        self.e_unique.clone()
    }

    /// Declare an edge unique constraint on `(etype, keys)` — at most one live edge
    /// of `etype` may carry a given tuple of non-null scalar values. Null/list
    /// values are exempt (matching core, whose edge unique is index-backed). Errors
    /// if the CURRENT data already violates it.
    pub fn create_edge_unique_constraint(
        &mut self,
        etype: &str,
        keys: &[&str],
    ) -> Result<(), String> {
        let keys: Vec<String> = keys.iter().map(|s| (*s).to_string()).collect();
        self.check_etype_unique(etype, &keys)?;
        self.e_unique.push((etype.to_string(), keys));
        Ok(())
    }

    fn check_edge_unique_for_type(&self, etype: &str) -> Result<(), String> {
        for (t, keys) in &self.e_unique {
            if t == etype {
                self.check_etype_unique(t, keys)?;
            }
        }
        Ok(())
    }

    fn check_etype_unique(&self, etype: &str, keys: &[String]) -> Result<(), String> {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        for eid in self.edges_of_type(etype) {
            let vals: Vec<Value> = keys.iter().map(|k| self.edge_prop(eid, k)).collect();
            // A null/list/record value is exempt (not part of the uniqueness set).
            if vals.iter().any(|v| !is_indexable(v)) {
                continue;
            }
            let mut buf = Vec::new();
            for v in &vals {
                crate::value::group_key_into(v, &mut buf);
            }
            if !seen.insert(buf) {
                return Err(format!(
                    "E_UNIQUE: edge unique constraint on {etype}({}) violated",
                    keys.join(", ")
                ));
            }
        }
        Ok(())
    }

    /// The declared edge required constraints as `(edge type, key)`.
    #[must_use]
    pub fn edge_required_constraints(&self) -> Vec<(String, String)> {
        self.e_required.clone()
    }

    /// Declare an edge required constraint on `(etype, key)` — every live edge of
    /// `etype` must carry a PRESENT value for `key`. Errors if data already breaks it.
    pub fn create_edge_required_constraint(
        &mut self,
        etype: &str,
        key: &str,
    ) -> Result<(), String> {
        self.check_etype_required(etype, key)?;
        self.e_required.push((etype.to_string(), key.to_string()));
        Ok(())
    }

    fn check_edge_required_for_type(&self, etype: &str) -> Result<(), String> {
        for (t, key) in &self.e_required {
            if t == etype {
                self.check_etype_required(t, key)?;
            }
        }
        Ok(())
    }

    fn check_etype_required(&self, etype: &str, key: &str) -> Result<(), String> {
        for eid in self.edges_of_type(etype) {
            if !self.has_edge_prop(eid, key) {
                return Err(format!(
                    "E_REQUIRED: edge required constraint on {etype}({key}) violated"
                ));
            }
        }
        Ok(())
    }

    /// The live edge ids whose PRIMARY type is `etype` (an edge's first label).
    fn edges_of_type(&self, etype: &str) -> Vec<u32> {
        self.all_edges()
            .into_iter()
            .filter(|&e| self.edge_type_name(e).as_deref() == Some(etype))
            .collect()
    }

    // --- Type constraints (vertex + edge) ---------------------------------

    /// The declared vertex type constraints as `(label, key, type_name, not_null)`.
    #[must_use]
    pub fn type_constraints(&self) -> Vec<(String, String, String, bool)> {
        self.v_type
            .iter()
            .map(|r| (r.target.clone(), r.key.clone(), r.ty.to_name(), r.not_null))
            .collect()
    }

    /// The declared edge type constraints as `(edge type, key, type_name, not_null)`.
    #[must_use]
    pub fn edge_type_constraints(&self) -> Vec<(String, String, String, bool)> {
        self.e_type
            .iter()
            .map(|r| (r.target.clone(), r.key.clone(), r.ty.to_name(), r.not_null))
            .collect()
    }

    /// Declare a type constraint on `(target, key)`: a vertex label's or edge type's
    /// property must be of `type_name` (a scalar name, `list`, or `record`/`any
    /// record`, optionally suffixed ` NOT NULL`). `edge` selects the namespace.
    /// Errors `E_INVALID_VALUE` on an unknown name, `E_TYPE` if data already breaks it.
    pub fn create_type_constraint(
        &mut self,
        target: &str,
        key: &str,
        type_name: &str,
        edge: bool,
    ) -> Result<(), String> {
        let (ty, not_null) = parse_type_spec(type_name)?;
        self.check_target_type(target, key, &ty, not_null, edge, true)?;
        let rule = TypeRule {
            target: target.to_string(),
            key: key.to_string(),
            ty,
            not_null,
        };
        if edge {
            self.e_type.push(rule);
        } else {
            self.v_type.push(rule);
        }
        Ok(())
    }

    fn check_type_for_target(&self, target: &str, edge: bool) -> Result<(), String> {
        let rules = if edge { &self.e_type } else { &self.v_type };
        for r in rules {
            if r.target == target {
                self.check_target_type(target, &r.key, &r.ty, r.not_null, edge, false)?;
            }
        }
        Ok(())
    }

    /// Check `key` on every element carrying `target` against type `ty`/`not_null`.
    /// `declaring` picks the message (existing-data vs write violation).
    fn check_target_type(
        &self,
        target: &str,
        key: &str,
        ty: &TypeSpec,
        not_null: bool,
        edge: bool,
        declaring: bool,
    ) -> Result<(), String> {
        let value_of = |id: u32| -> Value {
            if edge {
                self.edge_prop(id, key)
            } else {
                self.prop(id, key)
            }
        };
        let ids: Vec<u32> = if edge {
            self.edges_of_type(target)
        } else {
            self.nodes_with_label(target).to_vec()
        };
        for id in ids {
            let v = value_of(id);
            // The property's own nullability (`not_null`) is checked separately from
            // the type match (a top-level null is type-exempt).
            let ok = !(not_null && matches!(v, Value::Null)) && value_matches(&v, ty);
            if !ok {
                return Err(if declaring {
                    "E_TYPE: existing data already violates the type constraint being declared"
                        .to_string()
                } else {
                    "E_TYPE: write violates a type constraint (a value is not of the declared type)"
                        .to_string()
                });
            }
        }
        Ok(())
    }

    // --- Cardinality constraints ------------------------------------------

    /// The declared cardinality constraints as `(label, etype, direction, min, max)`.
    #[must_use]
    pub fn cardinality_constraints(&self) -> Vec<(String, String, u8, u32, Option<u32>)> {
        self.cardinality
            .iter()
            .map(|r| (r.label.clone(), r.etype.clone(), r.direction, r.min, r.max))
            .collect()
    }

    /// The declared validators as `(target, var, predicate source)` — for the dump.
    #[must_use]
    pub fn validators(&self) -> Vec<(String, String, String)> {
        self.validators
            .iter()
            .map(|r| (r.target.clone(), r.var.clone(), r.src.clone()))
            .collect()
    }

    /// The declared invariants as `(name, query source)` — for the dump.
    #[must_use]
    pub fn invariants(&self) -> Vec<(String, String)> {
        self.invariants
            .iter()
            .map(|r| (r.name.clone(), r.src.clone()))
            .collect()
    }

    /// Store a validator (`target` = a vertex label or edge type, `var` = the bound
    /// name, `src` = the predicate). `checks` are the composed check queries the
    /// exec layer built (it does the declaration-time evaluation before calling).
    pub(crate) fn declare_validator(
        &mut self,
        target: &str,
        var: &str,
        src: &str,
        checks: Vec<crate::ir::Plan>,
    ) {
        self.validators.push(ValidatorRule {
            target: target.to_string(),
            var: var.to_string(),
            src: src.to_string(),
            checks,
        });
    }

    /// Store an invariant (replacing any prior one of the same `name`). The exec
    /// layer parses `plan` and runs the declaration-time check before calling.
    pub(crate) fn declare_invariant(&mut self, name: &str, src: &str, plan: crate::ir::Plan) {
        self.invariants.retain(|r| r.name != name);
        self.invariants.push(InvariantRule {
            name: name.to_string(),
            src: src.to_string(),
            plan,
        });
    }

    /// Every validator check query — a non-empty result from any is a violation.
    #[must_use]
    pub(crate) fn validator_check_plans(&self) -> Vec<&crate::ir::Plan> {
        self.validators
            .iter()
            .flat_map(|r| r.checks.iter())
            .collect()
    }

    /// Every invariant as `(name, query plan)` — a boolean-`false` cell is a violation.
    #[must_use]
    pub(crate) fn invariant_plans(&self) -> Vec<(&str, &crate::ir::Plan)> {
        self.invariants
            .iter()
            .map(|r| (r.name.as_str(), &r.plan))
            .collect()
    }

    /// Declare a cardinality constraint: a `label` vertex's `etype`-edge degree in
    /// `direction` (0 out / 1 in) must lie in `min..=max` (`max: None` = ∞).
    /// Re-declaring the same `(label, etype, direction)` replaces the bounds.
    pub fn create_cardinality_constraint(
        &mut self,
        label: &str,
        etype: &str,
        direction: u8,
        min: u32,
        max: Option<u32>,
    ) -> Result<(), String> {
        self.check_cardinality_rule(label, etype, direction, min, max, true)?;
        if let Some(r) = self
            .cardinality
            .iter_mut()
            .find(|r| r.label == label && r.etype == etype && r.direction == direction)
        {
            r.min = min;
            r.max = max;
        } else {
            self.cardinality.push(CardRule {
                label: label.to_string(),
                etype: etype.to_string(),
                direction,
                min,
                max,
            });
        }
        Ok(())
    }

    fn check_all_cardinality(&self) -> Result<(), String> {
        for r in &self.cardinality {
            self.check_cardinality_rule(&r.label, &r.etype, r.direction, r.min, r.max, false)?;
        }
        Ok(())
    }

    fn check_cardinality_rule(
        &self,
        label: &str,
        etype: &str,
        direction: u8,
        min: u32,
        max: Option<u32>,
        declaring: bool,
    ) -> Result<(), String> {
        let tid = self.etype_id(etype);
        for &id in self.nodes_with_label(label) {
            let adj = if direction == 1 {
                self.inc(id)
            } else {
                self.out(id)
            };
            let degree = match tid {
                Some(t) => adj.iter().filter(|a| a.etype == t).count() as u32,
                None => 0,
            };
            if degree < min || max.is_some_and(|m| degree > m) {
                return Err(if declaring {
                    "E_CARDINALITY: existing data already violates the cardinality constraint being declared".to_string()
                } else {
                    "E_CARDINALITY: write violates a cardinality constraint (a vertex's edge degree is outside its declared min..max bound)".to_string()
                });
            }
        }
        Ok(())
    }

    // --- Mutation ---------------------------------------------------------
    //
    // The write path. Nodes get the next dense id; every existing property column
    // grows one absent slot so all columns stay length `node_count`. Edges take a
    // monotonic `eid` shared by their out/in entries. Properties are set into a
    // typed column, promoting it to `Gen` on a type change. These are the store
    // primitives the language write statements (Phase B) and transactions (A3)
    // build on; they do not enforce constraints — that is a later, higher layer.

    /// After a bulk load, dictionary-encode every eligible categorical `Str` property
    /// column. Incremental adds build plain `Str` columns; a bulk loader
    /// ([`crate::ndjson::from_ndjson`]) calls this ONCE at the end so a low-cardinality
    /// `city`/`dept`/`status` gets the code-based encoding that makes GROUP BY /
    /// DISTINCT / equality match on a `u32` code instead of hashing string content. A
    /// high-cardinality column (`name`, an id) is left as `Str` by `dict_encode`'s cap.
    pub fn dict_encode_columns(&mut self) {
        for col in self.props.values_mut() {
            col.try_dict_encode();
        }
    }

    /// Add a node with `labels` and `(key, value)` properties; returns its id.
    pub fn add_node(&mut self, labels: &[&str], props: &[(&str, Value)]) -> u32 {
        // touch()/epoch bumps happen in add_node_with_id (the leaf ingest also calls).
        // A created node mints its external id from its dense id (stable for the
        // life of the store — dense ids are never reused). Ingest supplies the
        // file's id via `add_node_with_id`.
        let ext: Arc<str> = Arc::from(self.node_count.to_string().as_str());
        self.add_node_with_id(&ext, labels, props)
    }

    /// Add a node carrying an explicit external id (used by ingest, which preserves
    /// the id from the file). Returns the dense id.
    pub fn add_node_with_id(
        &mut self,
        ext: &Arc<str>,
        labels: &[&str],
        props: &[(&str, Value)],
    ) -> u32 {
        self.touch();
        for l in labels {
            self.bump_epoch(l);
        }
        for (k, _) in props {
            self.bump_epoch(k);
        }
        self.invalidate_csr(); // a new node changes the adjacency shape
        let id = self.node_count as u32;
        self.node_count += 1;
        self.node_ext.push(Arc::clone(ext));
        self.ext_to_node.insert(Arc::clone(ext), id);
        // Keep every existing column the same length as the node set.
        for col in self.props.values_mut() {
            col.push_absent();
        }
        self.out_adj.push(Vec::new());
        self.in_adj.push(Vec::new());
        if self.edge_type_index {
            self.out_type_idx.push(HashMap::new());
            self.in_type_idx.push(HashMap::new());
        }
        if let Some(ix) = &mut self.interval {
            ix.by_lo.push(Vec::new());
            ix.by_hi.push(Vec::new());
        }
        self.deleted.push(false);
        for l in labels {
            // ids are handed out increasing, so appending keeps the bucket sorted.
            self.by_label.entry((*l).to_string()).or_default().push(id);
        }
        for (k, v) in props {
            // Apply the initial props directly; the single AddNode undo (which
            // pops the whole node) reverses them, so they are not logged twice.
            self.apply_set_prop(id, k, v.clone());
        }
        if let Some(log) = &mut self.undo {
            log.push(Undo::AddNode);
        }
        self.record_change(Change::NodeAdded(id));
        id
    }

    /// Add a directed edge `from -[label]-> to`; returns its `eid` (shared by the
    /// out and in adjacency entries). Interns the edge type if new. The returned
    /// eid lets the caller attach edge properties to the new edge.
    pub fn add_edge(&mut self, from: u32, to: u32, label: &str) -> u32 {
        // A created edge mints its external id as `e<eid>`; ingest supplies the
        // file's id via `add_edge_with_id`.
        let ext: Arc<str> = Arc::from(format!("e{}", self.next_eid).as_str());
        self.add_edge_with_id(&ext, from, to, label)
    }

    /// Add an edge carrying an explicit external id (used by ingest). Returns the
    /// eid.
    pub fn add_edge_with_id(&mut self, ext: &Arc<str>, from: u32, to: u32, label: &str) -> u32 {
        self.touch();
        self.bump_epoch(label);
        self.invalidate_csr();
        self.invalidate_edge_num(); // next_eid grows; the eid-indexed overlay is stale
        assert!(
            (from as usize) < self.node_count && (to as usize) < self.node_count,
            "edge endpoint out of range"
        );
        let next = self.etype_ids.len() as u32;
        let etype = *self.etype_ids.entry(label.to_string()).or_insert(next);
        let eid = self.next_eid;
        self.next_eid += 1;
        debug_assert_eq!(
            self.edge_etype.len() as u32,
            eid,
            "edge_etype indexed by eid"
        );
        self.edge_etype.push(etype);
        self.edge_has_extra.push(false); // no secondary labels until set_edge_extra_labels
        self.edge_ends.push((from, to));
        debug_assert_eq!(self.edge_ext.len() as u32, eid, "edge_ext indexed by eid");
        self.edge_ext.push(Arc::clone(ext));
        self.out_adj[from as usize].push(Adj {
            nbr: to,
            etype,
            eid,
        });
        self.in_adj[to as usize].push(Adj {
            nbr: from,
            etype,
            eid,
        });
        if self.edge_type_index {
            // Hot path: a single append to each endpoint's type bucket — O(1), not
            // a per-node rebuild.
            self.out_type_idx[from as usize]
                .entry(etype)
                .or_default()
                .push(Adj {
                    nbr: to,
                    etype,
                    eid,
                });
            self.in_type_idx[to as usize]
                .entry(etype)
                .or_default()
                .push(Adj {
                    nbr: from,
                    etype,
                    eid,
                });
        }
        if self.interval.is_some() {
            // The new edge carries no interval props yet (they arrive via
            // set_edge_prop, which rebuilds); reindexing now keeps `from`'s buckets
            // consistent and picks the edge up once its props are set.
            self.reindex_node_interval(from);
        }
        if let Some(log) = &mut self.undo {
            log.push(Undo::AddEdge {
                u: from,
                v: to,
                eid,
            });
        }
        self.record_change(Change::EdgeAdded(eid));
        eid
    }

    /// Attach SECONDARY labels to an edge (the labels past its first/type). Ingest
    /// of a multi-label `"labels":[…]` edge calls this after `add_edge_with_id`;
    /// `extra_names` is the label list with the first (already the edge type)
    /// dropped. Names are interned like an edge type. A no-op for an empty list, so
    /// a single-label graph never touches `edge_extra`.
    pub fn set_edge_extra_labels(&mut self, eid: u32, extra_names: &[&str]) {
        if extra_names.is_empty() {
            return;
        }
        let ids: Vec<u32> = extra_names
            .iter()
            .map(|name| {
                let next = self.etype_ids.len() as u32;
                *self.etype_ids.entry((*name).to_string()).or_insert(next)
            })
            .collect();
        self.edge_extra.insert(eid, ids);
        if let Some(flag) = self.edge_has_extra.get_mut(eid as usize) {
            *flag = true;
        }
    }

    /// Does edge `eid` carry label `tid`? Checks the primary type then, only when
    /// some edge in the graph is multi-label, the secondary set. Mirrors core's
    /// `edge_has_label`: the `is_empty` guard keeps a single-label graph at one
    /// `u32` compare.
    #[must_use]
    pub fn edge_has_label(&self, eid: u32, tid: u32) -> bool {
        self.edge_etype.get(eid as usize).is_some_and(|&t| t == tid)
            || (self
                .edge_has_extra
                .get(eid as usize)
                .copied()
                .unwrap_or(false)
                && self
                    .edge_extra
                    .get(&eid)
                    .is_some_and(|extra| extra.contains(&tid)))
    }

    /// As [`edge_has_label`](Self::edge_has_label), for a caller that ALREADY holds
    /// the edge's primary type (`first`, mirrored on every `Adj`). Mirrors core's
    /// `edge_type_matches`: starting from `first` (already in a register) skips the
    /// random `edge_etype[eid]` re-read `edge_has_label` does — a cache miss per edge
    /// to learn something the caller had. Core measured that re-read at 4.3x on a
    /// per-row correlated edge count; an adjacency type-filter is the same hot loop.
    #[must_use]
    pub fn edge_type_matches(&self, first: u32, eid: u32, tid: u32) -> bool {
        first == tid
            || (self
                .edge_has_extra
                .get(eid as usize)
                .copied()
                .unwrap_or(false)
                && self
                    .edge_extra
                    .get(&eid)
                    .is_some_and(|extra| extra.contains(&tid)))
    }

    /// Whether ANY edge carries more than one label — lets an adjacency walk keep
    /// its single-`u32`-compare fast path when no edge is multi-label.
    #[must_use]
    pub fn has_multi_label_edges(&self) -> bool {
        !self.edge_extra.is_empty()
    }

    /// Create the opt-in edge-type index and build it from the current adjacency.
    /// Idempotent; after this a type-filtered hop seeks a per-node type bucket
    /// rather than scanning the whole adjacency (see the field docs and
    /// `examples/expand_bench`). Subsequent writes maintain it.
    pub fn create_edge_type_index(&mut self) {
        self.edge_type_index = true;
        self.out_type_idx = vec![HashMap::new(); self.node_count];
        self.in_type_idx = vec![HashMap::new(); self.node_count];
        for node in 0..self.node_count as u32 {
            self.reindex_node_etypes(node);
        }
    }

    /// Whether the opt-in edge-type index is active.
    #[must_use]
    pub fn has_edge_type_index(&self) -> bool {
        self.edge_type_index
    }

    /// Node `node`'s outgoing adjacency of edge-type `etype` (empty if none, or if
    /// the index is off — callers gate on [`has_edge_type_index`](Store::has_edge_type_index)).
    #[must_use]
    pub fn out_typed(&self, node: u32, etype: u32) -> &[Adj] {
        self.out_type_idx
            .get(node as usize)
            .and_then(|m| m.get(&etype))
            .map_or(&[], Vec::as_slice)
    }

    /// Node `node`'s incoming adjacency of edge-type `etype`.
    #[must_use]
    pub fn in_typed(&self, node: u32, etype: u32) -> &[Adj] {
        self.in_type_idx
            .get(node as usize)
            .and_then(|m| m.get(&etype))
            .map_or(&[], Vec::as_slice)
    }

    /// Rebuild one node's type buckets from its (authoritative) flat adjacency. A
    /// no-op when the index is off. Called after any adjacency change other than
    /// the `add_edge` hot path, so the index needs no per-edge delta bookkeeping.
    fn reindex_node_etypes(&mut self, node: u32) {
        if !self.edge_type_index {
            return;
        }
        let i = node as usize;
        let mut om: HashMap<u32, Vec<Adj>> = HashMap::new();
        for a in &self.out_adj[i] {
            om.entry(a.etype).or_default().push(*a);
        }
        let mut im: HashMap<u32, Vec<Adj>> = HashMap::new();
        for a in &self.in_adj[i] {
            im.entry(a.etype).or_default().push(*a);
        }
        self.out_type_idx[i] = om;
        self.in_type_idx[i] = im;
    }

    // --- opt-in edge interval index (G4) ---

    /// Create the opt-in interval index over OUT-edges for the numeric edge props
    /// `(lo_key, hi_key)` and build it from the current edges. Replaces any prior
    /// interval index. After this, [`for_each_overlap`](Store::for_each_overlap)
    /// seeks a node's edges whose `[lo, hi]` overlaps a query interval instead of
    /// scanning the adjacency and reading the boxed props.
    pub fn create_interval_index(&mut self, lo_key: &str, hi_key: &str) {
        self.interval = Some(IntervalIndex {
            lo_key: lo_key.to_string(),
            hi_key: hi_key.to_string(),
            by_lo: vec![Vec::new(); self.node_count],
            by_hi: vec![Vec::new(); self.node_count],
        });
        for node in 0..self.node_count as u32 {
            self.reindex_node_interval(node);
        }
    }

    /// Whether an interval index on exactly `(lo_key, hi_key)` is active.
    #[must_use]
    pub fn has_interval_index(&self, lo_key: &str, hi_key: &str) -> bool {
        self.interval
            .as_ref()
            .is_some_and(|ix| ix.lo_key == lo_key && ix.hi_key == hi_key)
    }

    /// Whether `key` is one of the active interval index's axes (so a change to it
    /// invalidates the index).
    fn interval_uses_key(&self, key: &str) -> bool {
        self.interval
            .as_ref()
            .is_some_and(|ix| ix.lo_key == key || ix.hi_key == key)
    }

    /// Call `f(eid, nbr)` for each OUT-edge of `node` whose interval `[lo, hi]`
    /// overlaps `[qlo, qhi]` (i.e. `lo <= qhi && hi >= qlo`), seeking via the
    /// interval index. Seeds from whichever axis is the more selective (fewer
    /// candidates) and post-filters the other — never intersecting both. A no-op if
    /// no interval index is active or `node` is out of range.
    pub fn for_each_overlap(&self, node: u32, qlo: f64, qhi: f64, mut f: impl FnMut(u32, u32)) {
        let Some(ix) = &self.interval else { return };
        let Some(by_lo) = ix.by_lo.get(node as usize) else {
            return;
        };
        let by_hi = &ix.by_hi[node as usize];
        // # with lo <= qhi (a prefix of by_lo); # with hi >= qlo (a suffix of by_hi).
        let n_lo = by_lo.partition_point(|iv| iv.lo <= qhi);
        let n_hi = by_hi.len() - by_hi.partition_point(|iv| iv.hi < qlo);
        if n_lo <= n_hi {
            for iv in &by_lo[..n_lo] {
                if iv.hi >= qlo {
                    f(iv.eid, iv.nbr);
                }
            }
        } else {
            for iv in &by_hi[by_hi.len() - n_hi..] {
                if iv.lo <= qhi {
                    f(iv.eid, iv.nbr);
                }
            }
        }
    }

    /// Rebuild one source node's interval buckets from its current out-edges and
    /// their (boxed) props. An edge missing either numeric interval prop is skipped
    /// (it cannot be range-sought). No-op when the index is off.
    fn reindex_node_interval(&mut self, node: u32) {
        if self.interval.is_none() {
            return;
        }
        let (lo_key, hi_key) = {
            let ix = self.interval.as_ref().unwrap();
            (ix.lo_key.clone(), ix.hi_key.clone())
        };
        let i = node as usize;
        let mut ivs: Vec<Iv> = Vec::new();
        for a in &self.out_adj[i] {
            if let (Value::Num(lo), Value::Num(hi)) = (
                self.edge_prop(a.eid, &lo_key),
                self.edge_prop(a.eid, &hi_key),
            ) {
                ivs.push(Iv {
                    lo,
                    hi,
                    eid: a.eid,
                    nbr: a.nbr,
                });
            }
        }
        let mut by_lo = ivs.clone();
        by_lo.sort_by(|a, b| a.lo.total_cmp(&b.lo));
        let mut by_hi = ivs;
        by_hi.sort_by(|a, b| a.hi.total_cmp(&b.hi));
        let ix = self.interval.as_mut().unwrap();
        ix.by_lo[i] = by_lo;
        ix.by_hi[i] = by_hi;
    }

    /// Rebuild the whole interval index (used after an interval-key edge-prop change,
    /// where the affected source node is not cheaply known from the eid).
    fn rebuild_interval(&mut self) {
        if self.interval.is_none() {
            return;
        }
        // Resize per-node vectors in case node_count changed, then reindex all.
        {
            let n = self.node_count;
            let ix = self.interval.as_mut().unwrap();
            ix.by_lo = vec![Vec::new(); n];
            ix.by_hi = vec![Vec::new(); n];
        }
        for node in 0..self.node_count as u32 {
            self.reindex_node_interval(node);
        }
    }

    /// Whether node `node` carries a present value for `key` (distinct from a
    /// present `Null`, which `prop` cannot tell from absence).
    #[must_use]
    pub fn has_prop(&self, node: u32, key: &str) -> bool {
        self.props
            .get(key)
            .is_some_and(|c| c.present_at(node as usize))
    }

    /// Set node `node`'s `key` to `value`, creating the column if new and
    /// promoting it to `Gen` if `value`'s type differs from the column's.
    pub fn set_prop(&mut self, node: u32, key: &str, value: Value) {
        self.touch();
        self.bump_epoch(key);
        let rec = self.undo.is_some().then(|| Undo::RestoreCell {
            node,
            key: key.to_string(),
            prev_present: self.has_prop(node, key),
            prev_value: self.prop(node, key),
        });
        self.apply_set_prop(node, key, value);
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
        self.record_change(Change::NodeProp {
            node,
            key: key.to_string(),
        });
    }

    fn apply_set_prop(&mut self, node: u32, key: &str, value: Value) {
        // Index upkeep: capture the OLD base value before writing, and a copy of
        // the new one (reads first, then the column write, then the index writes —
        // distinct fields, no borrow clash). A hash index may be dotted, so it is
        // keyed by the BASE property `key`; the range index is single-key.
        let care = self.index_on_base(key) || self.is_range_indexed(key);
        let old = (care && self.has_prop(node, key)).then(|| self.prop(node, key));
        let new_for_index = care.then(|| value.clone());

        let n = self.node_count;
        let col = self
            .props
            .entry(key.to_string())
            .or_insert_with(|| Column::new_absent(&value, n));
        if !col.accepts(&value) {
            *col = col.to_gen();
        }
        col.set(node as usize, value);

        if let Some(nv) = new_for_index {
            self.reindex_node(key, node, old.as_ref(), Some(&nv));
            if self.is_range_indexed(key) {
                if let Some(old) = &old {
                    self.range_remove(key, old, node);
                }
                self.range_add(key, &nv, node);
            }
        }
    }

    /// Remove node `node`'s `key` — it reads as NULL again. (Distinct from setting
    /// it to a stored `Null`; that distinction is a Phase-E concern.)
    pub fn remove_prop(&mut self, node: u32, key: &str) {
        self.touch();
        self.bump_epoch(key);
        let rec = self.undo.is_some().then(|| Undo::RestoreCell {
            node,
            key: key.to_string(),
            prev_present: self.has_prop(node, key),
            prev_value: self.prop(node, key),
        });
        self.apply_remove_prop(node, key);
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
        self.record_change(Change::NodeProp {
            node,
            key: key.to_string(),
        });
    }

    fn apply_remove_prop(&mut self, node: u32, key: &str) {
        // Drop the node from the index(es) for its OLD value, if indexed.
        let old = ((self.index_on_base(key) || self.is_range_indexed(key))
            && self.has_prop(node, key))
        .then(|| self.prop(node, key));
        if let Some(col) = self.props.get_mut(key) {
            col.set_absent(node as usize);
        }
        if let Some(old) = &old {
            self.reindex_node(key, node, Some(old), None);
            if self.is_range_indexed(key) {
                self.range_remove(key, old, node);
            }
        }
    }

    // --- Edge properties -------------------------------------------------
    //
    // Keyed by the edge's `eid` (shared by its out/in adjacency entries), so a
    // property is one value per edge regardless of direction. Undo-logged like
    // node properties.

    /// Read edge `eid`'s `key` (NULL if absent).
    #[must_use]
    pub fn edge_prop(&self, eid: u32, key: &str) -> Value {
        self.edge_props
            .get(key)
            .and_then(|m| m.get(&eid))
            .cloned()
            .unwrap_or(Value::Null)
    }

    /// The raw per-eid map for one edge-property `key`, if any edge carries it.
    /// Lets a reader resolve the key ONCE and probe many eids against that single
    /// inner map (the exec edge-column fast path), instead of re-hashing `key` per
    /// edge through [`Self::edge_prop`].
    #[must_use]
    pub fn edge_prop_map(&self, key: &str) -> Option<&EdgeMap> {
        self.edge_props.get(key)
    }

    /// The dense numeric READ overlay for edge-property `key` — `(data, present)`
    /// indexed by eid — or `None` when the overlay is stale (an edge write happened),
    /// the key is absent, or the key is not homogeneously numeric. When present,
    /// `data[eid]`/`present[eid]` read the SAME value as [`Self::edge_prop`] with no
    /// hash probe or `Value` unbox. Callers fall back to `edge_prop` on `None`.
    #[must_use]
    pub fn edge_num_column(&self, key: &str) -> Option<(&[f64], &[bool])> {
        if !self.edge_num_fresh {
            return None;
        }
        self.edge_num
            .get(key)
            .map(|(d, p)| (d.as_slice(), p.as_slice()))
    }

    /// Rebuild the numeric edge-property overlay from the boxed `edge_props` source of
    /// truth: one dense `Vec<f64>` + present bitset per HOMOGENEOUSLY-NUMERIC key,
    /// indexed by eid. A key with any non-numeric present value is omitted. Called at
    /// load; a write invalidates it (readers fall back until the next rebuild).
    pub fn rebuild_edge_num(&mut self) {
        let n = self.next_eid as usize;
        self.edge_num.clear();
        for (key, map) in &self.edge_props {
            if map.values().all(|v| matches!(v, Value::Num(_))) {
                let mut data = vec![0.0f64; n];
                let mut present = vec![false; n];
                for (&eid, v) in map {
                    if let Value::Num(x) = v {
                        let i = eid as usize;
                        if i < n {
                            data[i] = *x;
                            present[i] = true;
                        }
                    }
                }
                self.edge_num.insert(key.clone(), (data, present));
            }
        }
        self.edge_num_fresh = true;
    }

    fn invalidate_edge_num(&mut self) {
        self.edge_num_fresh = false;
    }

    /// Keep the numeric edge overlay in step with a `set_edge_prop` (called AFTER the
    /// boxed map is updated). A no-op when the overlay is stale. A Num write updates
    /// or (on a key's first appearance / re-promotion) full-builds that key's dense
    /// arrays from the boxed map; a non-Num write demotes the key (readers fall back
    /// to the boxed path for it). Keeps `edge_num_column` byte-identical to
    /// `edge_prop`.
    fn edge_num_on_set(&mut self, eid: u32, key: &str) {
        if !self.edge_num_fresh {
            return;
        }
        let is_num = matches!(
            self.edge_props.get(key).and_then(|m| m.get(&eid)),
            Some(Value::Num(_))
        );
        if !is_num {
            self.edge_num.remove(key); // a non-numeric present value → not an overlay key
            return;
        }
        if self.edge_num.contains_key(key) {
            let Some(Value::Num(x)) = self.edge_props.get(key).and_then(|m| m.get(&eid)) else {
                return;
            };
            let x = *x;
            if let Some((data, present)) = self.edge_num.get_mut(key) {
                data[eid as usize] = x;
                present[eid as usize] = true;
            }
            return;
        }
        // First value for this key (or a re-promotion): build its arrays from the boxed
        // map, which now includes this write — only when EVERY present value is Num.
        let n = self.next_eid as usize;
        if let Some(map) = self.edge_props.get(key) {
            if map.values().all(|v| matches!(v, Value::Num(_))) {
                let mut data = vec![0.0f64; n];
                let mut present = vec![false; n];
                for (&e, v) in map {
                    if let Value::Num(x) = v {
                        data[e as usize] = *x;
                        present[e as usize] = true;
                    }
                }
                self.edge_num.insert(key.to_string(), (data, present));
            }
        }
    }

    /// Keep the overlay in step with a `remove_edge_prop` (the key reads NULL again).
    fn edge_num_on_remove(&mut self, eid: u32, key: &str) {
        if !self.edge_num_fresh {
            return;
        }
        if let Some((_, present)) = self.edge_num.get_mut(key) {
            if let Some(p) = present.get_mut(eid as usize) {
                *p = false;
            }
        }
    }

    /// Whether edge `eid` carries a present value for `key`.
    #[must_use]
    pub fn has_edge_prop(&self, eid: u32, key: &str) -> bool {
        self.edge_props
            .get(key)
            .is_some_and(|m| m.contains_key(&eid))
    }

    /// Set edge `eid`'s `key` to `value`.
    pub fn set_edge_prop(&mut self, eid: u32, key: &str, value: Value) {
        let rec = self.undo.is_some().then(|| Undo::RestoreEdgeCell {
            eid,
            key: key.to_string(),
            prev: self.edge_props.get(key).and_then(|m| m.get(&eid)).cloned(),
        });
        self.edge_props
            .entry(key.to_string())
            .or_default()
            .insert(eid, value);
        self.edge_num_on_set(eid, key); // keep the typed read overlay in step
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
        // An interval-axis change moves an edge's interval; the source node isn't
        // cheaply known from the eid, so rebuild the (opt-in, rarely-mutated) index.
        if self.interval_uses_key(key) {
            self.rebuild_interval();
        }
        self.record_change(Change::EdgeProp {
            eid,
            key: key.to_string(),
        });
    }

    // --- Property indexes ------------------------------------------------
    //
    // A hash index on a node property `key`, mapping a value's grouping bytes to
    // the node ids carrying it. Built from current data on create and maintained
    // by the mutation primitives (so rollback, which replays them, stays
    // consistent). Index equality is grouping (group_key) — the seek layer maps
    // that to predicate `=` (NaN/null match nothing) so results match a scan.

    /// Create a hash index on a node property PATH (idempotent). `key` is a plain
    /// property (`age`) or a dotted record-field path (`meta.city`). Builds from
    /// the current live nodes that carry the base property, keying on the resolved
    /// (descended) value.
    pub fn create_index(&mut self, key: &str) {
        let path: Vec<String> = key.split('.').map(String::from).collect();
        if self.indexes.iter().any(|i| i.path == path) {
            return;
        }
        let sub = &path[1..];
        let mut map: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
        if let Some(col) = self.props.get(&path[0]) {
            for id in 0..self.node_count {
                if !self.deleted[id] && col.present_at(id) {
                    let v = resolve_path(&col.read(id), sub);
                    map.entry(crate::value::group_key(&v))
                        .or_default()
                        .push(id as u32);
                }
            }
        }
        self.indexes.push(Index { path, map });
    }

    /// Drop the vertex index(es) on `key` — the hash index on that exact path
    /// AND/OR the range index on that key. Idempotent: a no-op (still `Ok`) if no
    /// such index exists. REJECTED if `key` backs a unique constraint — drop the
    /// constraint first (mirrors lenke-core, which keeps a unique constraint's
    /// backing index an invariant).
    pub fn drop_vertex_index(&mut self, key: &str) -> Result<(), String> {
        if self
            .unique
            .iter()
            .any(|(_, keys)| keys.iter().any(|k| k == key))
        {
            return Err(
                "E_INVALID_GRAPH_OP: cannot drop the vertex index; it backs a unique constraint — \
                 drop the constraint first"
                    .to_string(),
            );
        }
        let path: Vec<String> = key.split('.').map(String::from).collect();
        self.indexes.retain(|i| i.path != path);
        self.ranges.retain(|r| r.key != key);
        Ok(())
    }

    /// Drop the edge interval index if `key` is one of its `[lo, hi]` keys.
    /// Idempotent (a no-op otherwise). REJECTED if `key` backs an edge unique
    /// constraint (mirrors lenke-core).
    pub fn drop_edge_index(&mut self, key: &str) -> Result<(), String> {
        if self
            .e_unique
            .iter()
            .any(|(_, keys)| keys.iter().any(|k| k == key))
        {
            return Err(
                "E_INVALID_GRAPH_OP: cannot drop the edge index; it backs a unique constraint — \
                 drop the constraint first"
                    .to_string(),
            );
        }
        if let Some(iv) = &self.interval {
            if iv.lo_key == key || iv.hi_key == key {
                self.interval = None;
            }
        }
        Ok(())
    }

    /// Whether any hash index is driven by the base property `base` (so a write to
    /// it must maintain that index).
    #[must_use]
    fn index_on_base(&self, base: &str) -> bool {
        self.indexes.iter().any(|i| i.base() == base)
    }

    /// Whether a HASH index exists on the exact (possibly dotted) property path
    /// `key` — what an `IndexSeek` on that key would actually use. The planner reads
    /// this to seed a real index rather than a scan-fallback `IndexSeek`.
    #[must_use]
    pub fn has_hash_index(&self, key: &str) -> bool {
        self.indexes
            .iter()
            .any(|i| i.path.iter().map(String::as_str).eq(key.split('.')))
    }

    /// Whether a RANGE index exists on property `key` — what a `RangeSeek` on that
    /// key would actually use.
    #[must_use]
    pub fn has_range_index(&self, key: &str) -> bool {
        self.is_range_indexed(key)
    }

    /// Candidate node ids (ANY label) whose property PATH `key` groups equal to
    /// `value`, or `None` if no index exists on that path. Deleted ids filtered.
    #[must_use]
    pub fn index_lookup(&self, key: &str, value: &Value) -> Option<Vec<u32>> {
        let path: Vec<String> = key.split('.').map(String::from).collect();
        let idx = self.indexes.iter().find(|i| i.path == path)?;
        let gk = crate::value::group_key(value);
        Some(
            idx.map
                .get(&gk)
                .map(|ids| {
                    ids.iter()
                        .copied()
                        .filter(|&id| !self.deleted[id as usize])
                        .collect()
                })
                .unwrap_or_default(),
        )
    }

    /// The size of the hash-index bucket for `key = value` — an EXACT selectivity
    /// for an `=` seek, O(1) (the bucket `Vec`'s length; it may count a few
    /// tombstoned ids, which is fine for an estimate). `None` if no index on `key`.
    #[must_use]
    pub fn index_bucket_len(&self, key: &str, value: &Value) -> Option<usize> {
        let path: Vec<String> = key.split('.').map(String::from).collect();
        let idx = self.indexes.iter().find(|i| i.path == path)?;
        let gk = crate::value::group_key(value);
        Some(idx.map.get(&gk).map_or(0, Vec::len))
    }

    /// Total edges ever created (O(1), monotonic). Used as
    /// `avg_degree = edge_count / node_count` for a fan-out estimate; a few deleted
    /// edges lingering only nudge the estimate, which never affects correctness.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.next_eid as usize
    }

    /// The EXACT number of distinct present values in a low-cardinality
    /// (dict-encoded) column — the group count for a grouping key, `dict.len()`,
    /// O(1). `None` for an absent or non-dict column (a high-cardinality string /
    /// numeric column has no cheap exact distinct count — that is where estimation,
    /// not counting, is unavoidable).
    #[must_use]
    pub fn distinct_count(&self, key: &str) -> Option<usize> {
        match self.column(key)? {
            Column::Dict { dict, .. } => Some(dict.len()),
            _ => None,
        }
    }

    /// Update every hash index whose BASE is `base` for a node whose base value
    /// changed from `old` to `new` (`None` = absent on that side).
    fn reindex_node(&mut self, base: &str, node: u32, old: Option<&Value>, new: Option<&Value>) {
        let matching: Vec<usize> = self
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, i)| i.base() == base)
            .map(|(j, _)| j)
            .collect();
        for j in matching {
            let sub = self.indexes[j].path[1..].to_vec();
            if let Some(old) = old {
                let og = crate::value::group_key(&resolve_path(old, &sub));
                if let Some(bucket) = self.indexes[j].map.get_mut(&og) {
                    bucket.retain(|&x| x != node);
                }
            }
            if let Some(new) = new {
                let ng = crate::value::group_key(&resolve_path(new, &sub));
                self.indexes[j].map.entry(ng).or_default().push(node);
            }
        }
    }

    /// Remove node `id`'s entries from every index (used by delete/pop).
    fn index_drop_node(&mut self, id: u32) {
        // For each hash index whose base the node carries, the group key of the
        // node's resolved (descended) value — then drop it from that index.
        let removals: Vec<(usize, Vec<u8>)> = self
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, ix)| self.has_prop(id, ix.base()))
            .map(|(j, ix)| {
                let v = resolve_path(&self.prop(id, ix.base()), &ix.path[1..]);
                (j, crate::value::group_key(&v))
            })
            .collect();
        for (j, gk) in removals {
            if let Some(bucket) = self.indexes[j].map.get_mut(&gk) {
                bucket.retain(|&x| x != id);
            }
        }
        // Range indexes hold non-null values only.
        let range_removals: Vec<(String, Value)> = self
            .ranges
            .iter()
            .filter(|ix| self.has_prop(id, &ix.key))
            .filter_map(|ix| {
                let v = self.prop(id, &ix.key);
                (!v.is_null()).then(|| (ix.key.clone(), v))
            })
            .collect();
        for (k, v) in range_removals {
            self.range_remove(&k, &v, id);
        }
    }

    /// Create a range index on node property `key` (idempotent). Built from the
    /// current live nodes carrying a NON-NULL value for it.
    pub fn create_range_index(&mut self, key: &str) {
        if self.ranges.iter().any(|i| i.key == key) {
            return;
        }
        let mut map: BTreeMap<OrdVal, Vec<u32>> = BTreeMap::new();
        if let Some(col) = self.props.get(key) {
            for id in 0..self.node_count {
                if !self.deleted[id] && col.present_at(id) {
                    let v = col.read(id);
                    if !v.is_null() {
                        map.entry(OrdVal(v)).or_default().push(id as u32);
                    }
                }
            }
        }
        self.ranges.push(RangeIndex {
            key: key.to_string(),
            map,
        });
    }

    #[must_use]
    fn is_range_indexed(&self, key: &str) -> bool {
        self.ranges.iter().any(|i| i.key == key)
    }

    fn range_add(&mut self, key: &str, value: &Value, node: u32) {
        if value.is_null() {
            return;
        }
        if let Some(ix) = self.ranges.iter_mut().find(|i| i.key == key) {
            ix.map.entry(OrdVal(value.clone())).or_default().push(node);
        }
    }

    fn range_remove(&mut self, key: &str, value: &Value, node: u32) {
        if value.is_null() {
            return;
        }
        if let Some(ix) = self.ranges.iter_mut().find(|i| i.key == key) {
            if let Some(bucket) = ix.map.get_mut(&OrdVal(value.clone())) {
                bucket.retain(|&x| x != node);
            }
        }
    }

    /// Candidate node ids (ANY label) whose `key` satisfies `prop <op> value`
    /// under `cmp_total`, or `None` if no range index exists on `key`. `op` is one
    /// of `Lt`/`Le`/`Gt`/`Ge`. A null `value` matches nothing. Deleted filtered.
    #[must_use]
    pub fn range_lookup(
        &self,
        key: &str,
        op: crate::ir::CompareOp,
        value: &Value,
    ) -> Option<Vec<u32>> {
        use crate::ir::CompareOp::{Ge, Gt, Le, Lt};
        use std::ops::Bound::{Excluded, Included, Unbounded};
        let ix = self.ranges.iter().find(|i| i.key == key)?;
        if value.is_null() {
            return Some(Vec::new());
        }
        let k = OrdVal(value.clone());
        let bounds: (std::ops::Bound<OrdVal>, std::ops::Bound<OrdVal>) = match op {
            Gt => (Excluded(k), Unbounded),
            Ge => (Included(k), Unbounded),
            Lt => (Unbounded, Excluded(k)),
            Le => (Unbounded, Included(k)),
            _ => return Some(Vec::new()), // not a range op
        };
        Some(
            ix.map
                .range(bounds)
                .flat_map(|(_, ids)| ids.iter().copied())
                .filter(|&id| !self.deleted[id as usize])
                .collect(),
        )
    }

    /// Two-sided range seek: the ids whose `key` falls in the interval bounded by
    /// `lo` below and `hi` above. Seeds the exact intersection of two bounds on the
    /// same key (`k >= a AND k < b`) in one BTree walk. A contradictory or empty
    /// interval (`a > b`, or a null endpoint — nulls are not range-comparable)
    /// returns an empty set rather than panicking in `BTreeMap::range`.
    pub fn range_between(
        &self,
        key: &str,
        lo: std::ops::Bound<&Value>,
        hi: std::ops::Bound<&Value>,
    ) -> Option<Vec<u32>> {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        let ix = self.ranges.iter().find(|i| i.key == key)?;
        let null_end =
            |b: std::ops::Bound<&Value>| matches!(b, Included(v) | Excluded(v) if v.is_null());
        if null_end(lo) || null_end(hi) {
            return Some(Vec::new());
        }
        let conv = |b: std::ops::Bound<&Value>| match b {
            Included(v) => Included(OrdVal(v.clone())),
            Excluded(v) => Excluded(OrdVal(v.clone())),
            Unbounded => Unbounded,
        };
        let lb = conv(lo);
        let ub = conv(hi);
        // Empty-interval guard: BTreeMap::range panics when start > end, or when
        // start == end with either side excluded.
        let empty = match (&lb, &ub) {
            (Included(a), Included(b)) => a > b,
            (Included(a), Excluded(b))
            | (Excluded(a), Included(b))
            | (Excluded(a), Excluded(b)) => a >= b,
            _ => false,
        };
        if empty {
            return Some(Vec::new());
        }
        Some(
            ix.map
                .range((lb, ub))
                .flat_map(|(_, ids)| ids.iter().copied())
                .filter(|&id| !self.deleted[id as usize])
                .collect(),
        )
    }

    /// Remove edge `eid`'s `key` (reads NULL again).
    pub fn remove_edge_prop(&mut self, eid: u32, key: &str) {
        self.touch();
        self.bump_epoch(key);
        let rec = self.undo.is_some().then(|| Undo::RestoreEdgeCell {
            eid,
            key: key.to_string(),
            prev: self.edge_props.get(key).and_then(|m| m.get(&eid)).cloned(),
        });
        if let Some(m) = self.edge_props.get_mut(key) {
            m.remove(&eid);
        }
        self.edge_num_on_remove(eid, key); // keep the typed read overlay in step
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
        if self.interval_uses_key(key) {
            self.rebuild_interval();
        }
        self.record_change(Change::EdgeProp {
            eid,
            key: key.to_string(),
        });
    }

    /// Delete the edge identified by `eid` between endpoints `u` and `v`. The eid
    /// is unique and shared by the edge's out/in entries; removing it from both
    /// endpoints' out AND in lists deletes the edge regardless of which endpoint
    /// was its source (so it is safe to call with the endpoints in either order,
    /// e.g. from a hop matched via incoming adjacency). A no-op if already gone.
    pub fn delete_edge(&mut self, u: u32, v: u32, eid: u32) {
        self.touch();
        if let Some(t) = self.edge_type_name(eid) {
            self.bump_epoch(&t);
        }
        self.invalidate_csr();
        self.invalidate_edge_num();
        let logging = self.undo.is_some();
        let mut removed: Vec<(u32, bool, Adj)> = Vec::new();
        for node in [u, v] {
            if let Some(adj) = self.out_adj.get_mut(node as usize) {
                if logging {
                    removed.extend(
                        adj.iter()
                            .filter(|a| a.eid == eid)
                            .map(|a| (node, true, *a)),
                    );
                }
                adj.retain(|a| a.eid != eid);
            }
            if let Some(adj) = self.in_adj.get_mut(node as usize) {
                if logging {
                    removed.extend(
                        adj.iter()
                            .filter(|a| a.eid == eid)
                            .map(|a| (node, false, *a)),
                    );
                }
                adj.retain(|a| a.eid != eid);
            }
        }
        if self.edge_type_index {
            self.reindex_node_etypes(u);
            self.reindex_node_etypes(v);
        }
        if self.interval.is_some() {
            // The interval index is on OUT-edges; the deleted eid was an out-edge
            // of whichever endpoint is its source, so reindex both to be safe.
            self.reindex_node_interval(u);
            self.reindex_node_interval(v);
        }
        if let Some(log) = &mut self.undo {
            log.push(Undo::RestoreEdge { entries: removed });
        }
        self.record_change(Change::EdgeDeleted(eid));
    }

    /// Delete node `id`: tombstone it (its dense id is never reused), detach every
    /// incident edge from the neighbour's mirror list, drop its adjacency, remove
    /// it from every label bucket, and clear its properties. After this it is
    /// absent from all scans and traversals. A no-op if already deleted.
    pub fn delete_node(&mut self, id: u32) {
        self.touch();
        for l in self.labels_of(id) {
            self.bump_epoch(&l);
        }
        self.invalidate_csr();
        self.invalidate_edge_num(); // deletes incident edges
        let i = id as usize;
        if self.deleted[i] {
            return;
        }
        // Drop the node from any property indexes (reads its props, still present).
        self.index_drop_node(id);
        // Capture the full prior state BEFORE mutating, if a transaction is open.
        let (labels, props) = if self.undo.is_some() {
            let labels = self
                .by_label
                .iter()
                .filter(|(_, b)| b.contains(&id))
                .map(|(k, _)| k.clone())
                .collect();
            let props = self
                .props
                .iter()
                .map(|(k, c)| (k.clone(), c.present_at(i), c.read(i)))
                .collect();
            (labels, props)
        } else {
            (Vec::new(), Vec::new())
        };

        // Detach each incident edge's mirror entry on the neighbour, by eid.
        let out = std::mem::take(&mut self.out_adj[i]);
        for a in &out {
            self.in_adj[a.nbr as usize].retain(|m| m.eid != a.eid);
        }
        let inc = std::mem::take(&mut self.in_adj[i]);
        for a in &inc {
            self.out_adj[a.nbr as usize].retain(|m| m.eid != a.eid);
        }
        // A self-loop appears in both `out` and `inc`; its mirror was in this
        // node's own lists, already emptied by the takes above — nothing dangling.

        // Remove from every label bucket (no per-node label list yet, so sweep).
        for bucket in self.by_label.values_mut() {
            bucket.retain(|&x| x != id);
        }
        // Clear its properties.
        for col in self.props.values_mut() {
            col.set_absent(i);
        }
        self.deleted[i] = true;

        if self.edge_type_index {
            // Node i's buckets go empty; each neighbour lost this node's mirror.
            self.reindex_node_etypes(id);
            for a in out.iter().chain(inc.iter()) {
                if a.nbr != id {
                    self.reindex_node_etypes(a.nbr);
                }
            }
        }
        if self.interval.is_some() {
            // Interval index is on OUT-edges: id's own out-edges are gone, and each
            // IN-neighbour (an edge `nbr -> id`) lost one of ITS out-edges.
            self.reindex_node_interval(id);
            for a in &inc {
                if a.nbr != id {
                    self.reindex_node_interval(a.nbr);
                }
            }
        }

        if let Some(log) = &mut self.undo {
            log.push(Undo::RestoreNode {
                id,
                out,
                inc,
                labels,
                props,
            });
        }
        // A node deletion cascades its incident edges; the CDC stream reports it as
        // one NodeDeleted (the edges are implied), keeping 1:1 with the undo.
        self.record_change(Change::NodeDeleted(id));
    }

    // --- Transactions ----------------------------------------------------
    //
    // An undo log makes a group of mutations atomic. `begin` opens it; every
    // mutation then records its inverse; `commit` discards the log (changes
    // stand); `rollback` applies the inverses in reverse. `savepoint`/`rollback_to`
    // give per-statement atomicity within a transaction (a statement rolls back
    // to its savepoint on failure without abandoning the whole transaction).
    // Constraint checks and event buffering are deferred to Phase H.

    /// Open a transaction. Panics if one is already open (no nesting yet).
    pub fn begin(&mut self) {
        assert!(self.undo.is_none(), "nested transactions are not supported");
        self.undo = Some(Vec::new());
        self.changes = Some(Vec::new());
    }

    /// Commit: the changes stand, the undo log is discarded, and the transaction's
    /// change list becomes the observable `last_commit` (CDC).
    pub fn commit(&mut self) {
        self.undo = None;
        self.last_commit = self.changes.take().unwrap_or_default();
    }

    /// Roll back every change since `begin`, in reverse, and close the
    /// transaction. A no-op outside a transaction. The change list is dropped (a
    /// rolled-back transaction is observed to have changed nothing).
    pub fn rollback(&mut self) {
        self.changes = None;
        if let Some(log) = self.undo.take() {
            // `undo` is now None, so the inverse mutations below do not re-log.
            for rec in log.into_iter().rev() {
                self.apply_undo(rec);
            }
        }
        // The interval index depends on edge PROPS as well as adjacency, both of
        // which are restored by the undos above in an order that per-record index
        // maintenance can't safely track — so rebuild it once against the fully
        // restored graph. (The edge-type index depends only on adjacency, which
        // each undo record reindexes as it restores it.)
        if self.interval.is_some() {
            self.rebuild_interval();
        }
    }

    /// The change list of the most recent committed transaction — the
    /// observation-only CDC stream. Read after a write; cannot veto it.
    #[must_use]
    pub fn last_commit_changes(&self) -> &[Change] {
        &self.last_commit
    }

    /// The DISTINCT content-derived scopes the last commit touched, plus a
    /// fail-open flag. The scope of a NODE change is its `scope_key` property (the
    /// host assigns what that means — e.g. `"room"`/`"tenant"`); a change with no
    /// derivable scope (an edge change, or a deleted/absent node) sets the flag,
    /// meaning "relevant to ALL clients." A subscriber to scope `S` treats the
    /// commit as relevant iff `open || scopes contains S`. This is an OPTIMIZATION,
    /// not a security boundary (fail-open): the host owns the scope-key authority
    /// (the engine derives, it does not mint one). Scopes are `cmp_total`-sorted.
    #[must_use]
    pub fn touched_scopes(&self, scope_key: &str) -> (Vec<Value>, bool) {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut scopes: Vec<Value> = Vec::new();
        let mut open = false;
        for ch in &self.last_commit {
            let node = match ch {
                Change::NodeAdded(n)
                | Change::NodeProp { node: n, .. }
                | Change::NodeDeleted(n) => Some(*n),
                Change::EdgeAdded(_) | Change::EdgeDeleted(_) | Change::EdgeProp { .. } => None,
            };
            match node {
                Some(n) => {
                    let v = self.prop(n, scope_key);
                    if v.is_null() {
                        open = true; // absent/deleted scope → visible to all
                    } else if seen.insert(crate::value::group_key(&v)) {
                        scopes.push(v);
                    }
                }
                None => open = true, // an edge change has no node scope → fail-open
            }
        }
        scopes.sort_by(crate::value::cmp_total);
        (scopes, open)
    }

    /// [`touched_scopes`](Store::touched_scopes) rendered as the CDC command's JSON
    /// result: `{"scopes":[…],"open":<bool>}`. A subscriber to scope `S` treats the
    /// last commit as relevant iff `open || scopes` contains `S`.
    pub fn last_write_scope_json(&self, scope_key: &str) -> String {
        let (scopes, open) = self.touched_scopes(scope_key);
        let mut out = String::from("{\"scopes\":[");
        for (i, v) in scopes.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            crate::ndjson::encode_value(&mut out, v);
        }
        out.push_str("],\"open\":");
        out.push_str(if open { "true" } else { "false" });
        out.push('}');
        out
    }

    /// Record a change into the active transaction's list (no-op outside a txn).
    /// Grows 1:1 with the undo log, so `rollback_to` can truncate both by length.
    fn record_change(&mut self, c: Change) {
        if let Some(ch) = &mut self.changes {
            ch.push(c);
        }
    }

    /// A mark for per-statement atomicity: the current undo-log length. Zero
    /// outside a transaction.
    #[must_use]
    pub fn savepoint(&self) -> usize {
        self.undo.as_ref().map_or(0, Vec::len)
    }

    /// Undo every change recorded after `mark`, keeping the transaction open and
    /// the changes up to `mark`. Used to roll back a single failed statement.
    pub fn rollback_to(&mut self, mark: usize) {
        // The change list grows 1:1 with the undo log, so it truncates to the same
        // mark — the undone statement's changes vanish from the CDC stream too.
        if let Some(ch) = &mut self.changes {
            ch.truncate(mark);
        }
        if let Some(mut log) = self.undo.take() {
            let mut undone = Vec::new();
            while log.len() > mark {
                undone.push(log.pop().expect("len > mark"));
            }
            // `undo` is None while applying, so these inverses do not re-log.
            for rec in undone {
                self.apply_undo(rec);
            }
            self.undo = Some(log);
        }
        // Rebuild the interval index against the restored graph (see `rollback`).
        if self.interval.is_some() {
            self.rebuild_interval();
        }
    }

    /// Run `f` in a transaction: commit if it returns `Ok`, roll back if `Err`.
    /// (Does not catch panics — an unwinding closure leaves the log un-applied;
    /// panic-safety is a later concern.)
    pub fn transaction<T, E>(
        &mut self,
        f: impl FnOnce(&mut Store) -> Result<T, E>,
    ) -> Result<T, E> {
        self.begin();
        match f(self) {
            Ok(v) => {
                self.commit();
                Ok(v)
            }
            Err(e) => {
                self.rollback();
                Err(e)
            }
        }
    }

    /// Apply one undo record. The caller has taken the log out (`self.undo` is
    /// `None`), so the primitive mutations invoked here do not re-log.
    fn apply_undo(&mut self, rec: Undo) {
        // Rollback replays adjacency changes; conservatively drop the overlay (a
        // prop-only undo also clears it — harmless, just forces one rebuild).
        self.invalidate_csr();
        self.invalidate_edge_num();
        match rec {
            Undo::AddNode => self.pop_last_node(),
            Undo::AddEdge { u, v, eid } => self.delete_edge(u, v, eid),
            Undo::RestoreCell {
                node,
                key,
                prev_present,
                prev_value,
            } => {
                if prev_present {
                    self.apply_set_prop(node, &key, prev_value);
                } else {
                    self.apply_remove_prop(node, &key);
                }
            }
            Undo::RestoreEdgeCell { eid, key, prev } => match prev {
                Some(v) => {
                    self.edge_props.entry(key).or_default().insert(eid, v);
                }
                None => {
                    if let Some(m) = self.edge_props.get_mut(&key) {
                        m.remove(&eid);
                    }
                }
            },
            Undo::RestoreEdge { entries } => {
                let touched: Vec<u32> = if self.edge_type_index {
                    entries.iter().map(|(node, _, _)| *node).collect()
                } else {
                    Vec::new()
                };
                for (node, is_out, adj) in entries {
                    if is_out {
                        self.out_adj[node as usize].push(adj);
                    } else {
                        self.in_adj[node as usize].push(adj);
                    }
                }
                for node in touched {
                    self.reindex_node_etypes(node);
                }
            }
            Undo::RestoreNode {
                id,
                out,
                inc,
                labels,
                props,
            } => {
                let i = id as usize;
                self.deleted[i] = false;
                // Restore mirrors on OTHER nodes; self-loops live in id's own
                // lists and are restored by the assignments below.
                for a in &out {
                    if a.nbr != id {
                        self.in_adj[a.nbr as usize].push(Adj {
                            nbr: id,
                            etype: a.etype,
                            eid: a.eid,
                        });
                    }
                }
                for a in &inc {
                    if a.nbr != id {
                        self.out_adj[a.nbr as usize].push(Adj {
                            nbr: id,
                            etype: a.etype,
                            eid: a.eid,
                        });
                    }
                }
                self.out_adj[i] = out;
                self.in_adj[i] = inc;
                if self.edge_type_index {
                    // Rebuild id's buckets, and each neighbour that regained a
                    // mirror (read the restored adjacency for the neighbour set).
                    self.reindex_node_etypes(id);
                    let nbrs: Vec<u32> = self.out_adj[i]
                        .iter()
                        .chain(self.in_adj[i].iter())
                        .map(|a| a.nbr)
                        .filter(|&nb| nb != id)
                        .collect();
                    for nb in nbrs {
                        self.reindex_node_etypes(nb);
                    }
                }
                for l in labels {
                    // Re-insert in SORTED position (not push): a delete removed this
                    // id from the middle, so appending would leave the bucket
                    // unsorted, breaking both the id-order scan seed and the
                    // binary-search label intersection in `index_seek_ids`. Adds are
                    // monotonic and deletes retain order, so this keeps buckets sorted.
                    let bucket = self.by_label.entry(l).or_default();
                    let pos = bucket.partition_point(|&x| x < id);
                    bucket.insert(pos, id);
                }
                for (k, present, value) in props {
                    if present {
                        self.apply_set_prop(id, &k, value);
                    } else {
                        self.apply_remove_prop(id, &k);
                    }
                }
            }
        }
    }

    /// Pop the last (highest-id) node — the inverse of a logged `add_node`.
    fn pop_last_node(&mut self) {
        self.invalidate_csr();
        debug_assert!(self.node_count > 0);
        let id = (self.node_count - 1) as u32;
        // Drop it from any indexes while its props still exist.
        self.index_drop_node(id);
        for b in self.by_label.values_mut() {
            b.retain(|&x| x != id);
        }
        for col in self.props.values_mut() {
            col.pop_last();
        }
        self.out_adj.pop();
        self.in_adj.pop();
        if self.edge_type_index {
            self.out_type_idx.pop();
            self.in_type_idx.pop();
        }
        if let Some(ix) = &mut self.interval {
            ix.by_lo.pop();
            ix.by_hi.pop();
        }
        if let Some(ext) = self.node_ext.pop() {
            self.ext_to_node.remove(&ext);
        }
        self.deleted.pop();
        self.node_count -= 1;
    }
}

/// Builds a `Store`. Node ids are assigned in insertion order.
#[derive(Default)]
pub struct Builder {
    node_count: usize,
    by_label: HashMap<String, Vec<u32>>,
    // Collected as (node, value) pairs per key, materialized into typed columns
    // at `build()` — so the builder stays simple and the store stays typed.
    props: HashMap<String, Vec<(u32, Value)>>,
    etype_ids: HashMap<String, u32>,
    edges: Vec<(u32, u32, u32)>, // (from, to, etype)
}

impl Builder {
    /// Add a node with `labels` and `(key, value)` properties; returns its id.
    pub fn node(&mut self, labels: &[&str], props: &[(&str, Value)]) -> u32 {
        let id = self.node_count as u32;
        self.node_count += 1;
        for l in labels {
            self.by_label.entry((*l).to_string()).or_default().push(id);
        }
        for (k, v) in props {
            self.props
                .entry((*k).to_string())
                .or_default()
                .push((id, v.clone()));
        }
        id
    }

    /// Add a directed edge `from -[label]-> to`.
    pub fn edge(&mut self, from: u32, to: u32, label: &str) {
        let next = self.etype_ids.len() as u32;
        let etype = *self.etype_ids.entry(label.to_string()).or_insert(next);
        self.edges.push((from, to, etype));
    }

    #[must_use]
    pub fn build(self) -> Store {
        let n = self.node_count;
        let props = self
            .props
            .into_iter()
            .map(|(k, pairs)| (k, materialize(pairs, n)))
            .collect();
        let mut out_adj = vec![Vec::new(); n];
        let mut in_adj = vec![Vec::new(); n];
        let edge_count = self.edges.len() as u32;
        // Builder-created elements mint external ids (dense id string for nodes,
        // `e<eid>` for edges) — the same scheme `add_node`/`add_edge` use.
        let node_ext: Vec<Arc<str>> = (0..n).map(|i| Arc::from(i.to_string().as_str())).collect();
        let ext_to_node: HashMap<Arc<str>, u32> = node_ext
            .iter()
            .enumerate()
            .map(|(i, e)| (Arc::clone(e), i as u32))
            .collect();
        let edge_ext: Vec<Arc<str>> = (0..edge_count)
            .map(|e| Arc::from(format!("e{e}").as_str()))
            .collect();
        let mut edge_etypes: Vec<u32> = Vec::with_capacity(self.edges.len());
        let mut edge_ends: Vec<(u32, u32)> = Vec::with_capacity(self.edges.len());
        for (eid, (from, to, etype)) in self.edges.into_iter().enumerate() {
            let eid = eid as u32;
            edge_etypes.push(etype); // index == eid (edges laid down in order)
            edge_ends.push((from, to));
            out_adj[from as usize].push(Adj {
                nbr: to,
                etype,
                eid,
            });
            in_adj[to as usize].push(Adj {
                nbr: from,
                etype,
                eid,
            });
        }
        let mut st = Store {
            node_count: n,
            limits: GraphLimits::default(),
            by_label: self.by_label,
            props,
            prop_keys_cache: std::sync::RwLock::default(),
            min_label_cache: std::sync::RwLock::default(),
            etype_ids: self.etype_ids,
            out_adj,
            in_adj,
            // Incremental edges continue the id sequence the build laid down.
            next_eid: edge_count,
            edge_etype: edge_etypes,
            edge_extra: HashMap::new(),
            // Builder edges are single-label (`edge_extra` empty), so no edge carries
            // secondary labels; a later set_edge_extra_labels flips the bit.
            edge_has_extra: vec![false; edge_count as usize],
            edge_ends,
            node_ext,
            edge_ext,
            ext_to_node,
            deleted: vec![false; n],
            undo: None,
            changes: None,
            last_commit: Vec::new(),
            unique: Vec::new(),
            required: Vec::new(),
            e_unique: Vec::new(),
            e_required: Vec::new(),
            v_type: Vec::new(),
            e_type: Vec::new(),
            cardinality: Vec::new(),
            validators: Vec::new(),
            invariants: Vec::new(),
            edge_props: HashMap::new(),
            indexes: Vec::new(),
            ranges: Vec::new(),
            // The edge-type index is opt-in; bulk build never turns it on (the
            // caller runs `create_edge_type_index` after load if it wants it).
            edge_type_index: false,
            out_type_idx: Vec::new(),
            in_type_idx: Vec::new(),
            interval: None,
            csr_out_off: Vec::new(),
            csr_out: Vec::new(),
            csr_in_off: Vec::new(),
            csr_in: Vec::new(),
            csr_fresh: false,
            csr_out_typed: Vec::new(),
            csr_in_typed: Vec::new(),
            edge_num: HashMap::new(),
            edge_num_fresh: false,
            version: 0,
            epochs: HashMap::new(),
        };
        // Flatten the freshly-built adjacency into the CSR read overlay, and densify
        // the numeric edge properties into the typed read overlay.
        st.rebuild_csr();
        st.rebuild_edge_num();
        st
    }
}

/// Attempt to dictionary-encode a string column. `Ok(Column::Dict)` when the number
/// of DISTINCT present values stays under a cap (a categorical column — `dept`,
/// `city`, `status`); `Err((data, present))` hands the buffers back for a plain
/// `Str` column when the cardinality is too high for the encoding to pay (`name`, an
/// id, free text). The probe aborts the moment the dict crosses the cap, so a
/// high-cardinality column costs only the capped prefix, not a full scan.
fn dict_encode(
    data: Vec<Arc<str>>,
    present: Vec<bool>,
) -> Result<Column, (Vec<Arc<str>>, Vec<bool>)> {
    const CAP: usize = 4096;
    let mut dict: Vec<Arc<str>> = Vec::new();
    let mut lookup: HashMap<Arc<str>, u32> = HashMap::new();
    let mut codes = vec![0u32; data.len()];
    for (i, s) in data.iter().enumerate() {
        if !present[i] {
            continue;
        }
        let code = if let Some(&c) = lookup.get(s) {
            c
        } else {
            if dict.len() >= CAP {
                return Err((data, present));
            }
            let c = dict.len() as u32;
            dict.push(s.clone());
            lookup.insert(s.clone(), c);
            c
        };
        codes[i] = code;
    }
    // Encode only when the values actually REPEAT (distinct ≤ half the rows). An
    // effectively-unique column (`name`, an email, free text) would spend a `u32`
    // per row AND a full dict for no dedup benefit and a slower indirected read, so
    // it stays `Str`. `dept`/`city`/`status`-shaped columns clear this easily.
    if dict.is_empty() || dict.len() * 2 > data.len() {
        return Err((data, present));
    }
    Ok(Column::Dict {
        dict,
        codes,
        present,
    })
}

/// Turn `(node, value)` pairs into the tightest typed column. Homogeneous
/// numeric/string/bool columns unbox; anything mixed falls to `Gen`.
fn materialize(pairs: Vec<(u32, Value)>, n: usize) -> Column {
    let all = |f: &dyn Fn(&Value) -> bool| pairs.iter().all(|(_, v)| f(v));
    if all(&|v| matches!(v, Value::Num(_))) {
        let mut col = Column::with_capacity_num(n);
        if let Column::Num { data, present } = &mut col {
            for (i, v) in pairs {
                if let Value::Num(x) = v {
                    data[i as usize] = x;
                    present[i as usize] = true;
                }
            }
        }
        col
    } else if all(&|v| matches!(v, Value::Str(_))) {
        let mut data: Vec<Arc<str>> = vec![Arc::from(""); n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            if let Value::Str(s) = v {
                data[i as usize] = s;
                present[i as usize] = true;
            }
        }
        dict_encode(data, present).unwrap_or_else(|(data, present)| Column::Str { data, present })
    } else if all(&|v| matches!(v, Value::Bool(_))) {
        let mut data = vec![false; n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            if let Value::Bool(b) = v {
                data[i as usize] = b;
                present[i as usize] = true;
            }
        }
        Column::Bool { data, present }
    } else if let Some(kind) = homogeneous_temporal_kind(&pairs) {
        let mut data = vec![kind.zero(); n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            if let Value::Temporal(t) = v {
                data[i as usize] = t;
                present[i as usize] = true;
            }
        }
        Column::Temporal {
            kind,
            data,
            present,
        }
    } else {
        let mut data = vec![Value::Null; n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            data[i as usize] = v;
            present[i as usize] = true;
        }
        Column::Gen { data, present }
    }
}

/// The single temporal kind shared by every pair, or `None` if any pair is
/// non-temporal or the kinds are mixed (→ a `Gen` column).
fn homogeneous_temporal_kind(pairs: &[(u32, Value)]) -> Option<crate::temporal::TemporalKind> {
    let mut kind = None;
    for (_, v) in pairs {
        let Value::Temporal(t) = v else {
            return None;
        };
        match kind {
            None => kind = Some(t.kind()),
            Some(k) if k == t.kind() => {}
            Some(_) => return None, // mixed kinds fall back to Gen
        }
    }
    kind
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_bumps_on_every_mutation() {
        let mut s = Store::default();
        assert_eq!(s.version(), 0);
        let a = s.add_node(&["N"], &[]);
        let v1 = s.version();
        assert!(v1 > 0, "add_node bumps version");
        let b = s.add_node(&["N"], &[]);
        assert!(s.version() > v1, "second add_node bumps again");
        let v2 = s.version();
        let e = s.add_edge(a, b, "R");
        assert!(s.version() > v2, "add_edge bumps");
        let v3 = s.version();
        s.set_prop(a, "k", Value::Num(1.0));
        assert!(s.version() > v3, "set_prop bumps");
        let v4 = s.version();
        s.delete_edge(a, b, e);
        assert!(s.version() > v4, "delete_edge bumps");
    }

    #[test]
    fn epoch_bumps_only_the_tokens_a_change_touches() {
        let mut st = Store::default();
        let a = st.add_node(&["Person"], &[("name", s("alice"))]);
        // Both the label and the property key were touched by the add.
        let (p0, n0) = (st.epoch("Person"), st.epoch("name"));
        assert!(p0 > 0 && n0 > 0);
        assert_eq!(st.epoch("Project"), 0, "an untouched token stays 0");

        // A property change bumps only that property's epoch, not the label's.
        st.set_prop(a, "age", Value::Num(30.0));
        assert!(st.epoch("age") > n0);
        assert_eq!(st.epoch("Person"), p0, "unrelated label epoch is unchanged");
        assert_eq!(
            st.epoch("name"),
            n0,
            "unrelated property epoch is unchanged"
        );

        // Epoch never exceeds the global version.
        assert!(st.epoch("age") <= st.version());
    }

    #[test]
    fn clone_is_deep_and_independent() {
        let mut s = Store::default();
        let a = s.add_node(&["N"], &[("k", Value::Num(1.0))]);
        let b = s.add_node(&["N"], &[]);
        s.add_edge(a, b, "R");
        let ver = s.version();
        let snap = s.clone();
        assert_eq!(snap.version(), ver, "clone copies the version");
        // Mutating the original must not touch the clone.
        s.add_node(&["N"], &[]);
        assert_eq!(snap.node_count(), 2, "clone unaffected by later mutation");
        assert_eq!(s.node_count(), 3);
        assert_eq!(snap.edge_count(), 1);
        assert_eq!(snap.version(), ver, "clone version is frozen at copy time");
    }

    /// The CSR read overlay must (a) match the per-node adjacency exactly after a
    /// build, (b) reflect a write IMMEDIATELY (invalidation → Vec fallback), and
    /// (c) match again after an explicit rebuild.
    #[test]
    fn csr_overlay_matches_and_invalidates_on_write() {
        let mut b = Builder::default();
        b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        b.edge(0, 1, "R");
        b.edge(0, 2, "R");
        let mut st = b.build();
        let out = |st: &Store, v: u32| -> Vec<u32> { st.out(v).iter().map(|a| a.nbr).collect() };
        let inc = |st: &Store, v: u32| -> Vec<u32> { st.inc(v).iter().map(|a| a.nbr).collect() };
        // (a) fresh CSR after build: neighbour ORDER preserved.
        assert!(st.csr_fresh);
        assert_eq!(out(&st, 0), vec![1, 2]);
        // (b) a write clears the overlay and is visible at once via the Vec fallback.
        st.add_edge(0, 1, "R");
        assert!(!st.csr_fresh);
        assert_eq!(out(&st, 0), vec![1, 2, 1]);
        assert_eq!(inc(&st, 1), vec![0, 0]);
        // (c) rebuild re-enables the CSR and it still matches.
        st.rebuild_csr();
        assert!(st.csr_fresh);
        assert_eq!(out(&st, 0), vec![1, 2, 1]);
        assert_eq!(inc(&st, 1), vec![0, 0]);
    }

    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }
    fn n(x: f64) -> Value {
        Value::Num(x)
    }

    #[test]
    fn temporal_props_use_a_typed_column_and_promote_on_mixed_kind() {
        use crate::temporal::{Date, Temporal, TemporalKind, Time};
        let d = |iso: &str| Value::Temporal(Temporal::Date(Date::parse(iso).unwrap()));
        let mut b = Builder::default();
        b.node(&["P"], &[("born", d("1990-01-01"))]);
        b.node(&["P"], &[("born", d("2000-01-01"))]);
        let mut st = b.build();
        // Homogeneous Date props de-box into a typed Temporal column of kind Date.
        assert!(matches!(
            st.column("born"),
            Some(Column::Temporal {
                kind: TemporalKind::Date,
                ..
            })
        ));
        match st.prop(0, "born") {
            Value::Temporal(Temporal::Date(x)) => assert_eq!(x.format(), "1990-01-01"),
            o => panic!("expected a Date, got {o:?}"),
        }
        // Writing a DIFFERENT temporal kind promotes the column to Gen; both
        // values still read back correctly.
        st.set_prop(
            1,
            "born",
            Value::Temporal(Temporal::Time(Time::parse("12:00:00").unwrap())),
        );
        assert!(matches!(st.column("born"), Some(Column::Gen { .. })));
        assert!(matches!(
            st.prop(0, "born"),
            Value::Temporal(Temporal::Date(_))
        ));
        assert!(matches!(
            st.prop(1, "born"),
            Value::Temporal(Temporal::Time(_))
        ));
    }

    #[test]
    fn commit_records_the_change_list_rollback_records_nothing() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]); // outside a txn → nothing observed
        assert!(st.last_commit_changes().is_empty());

        // A committed transaction publishes exactly its changes, in order.
        st.begin();
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        st.set_prop(a, "age", n(1.0));
        let eid = st.add_edge(a, b, "R");
        st.commit();
        assert_eq!(
            st.last_commit_changes(),
            &[
                Change::NodeAdded(b),
                Change::NodeProp {
                    node: a,
                    key: "age".into(),
                },
                Change::EdgeAdded(eid),
            ]
        );

        // A rolled-back transaction publishes nothing: `last_commit` still shows
        // the previous COMMIT, unchanged (rollback is not an event).
        let previous: Vec<Change> = st.last_commit_changes().to_vec();
        st.begin();
        st.set_prop(a, "age", n(2.0));
        st.rollback();
        assert_eq!(st.last_commit_changes(), previous.as_slice());
    }

    #[test]
    fn touched_scopes_are_the_distinct_rooms_a_commit_writes() {
        let str_scopes = |scopes: &[Value]| -> Vec<String> {
            scopes
                .iter()
                .map(|v| match v {
                    Value::Str(x) => x.to_string(),
                    o => format!("{o:?}"),
                })
                .collect()
        };
        let mut st = Builder::default().build();
        st.begin();
        st.add_node(&["Msg"], &[("room", s("A"))]);
        st.add_node(&["Msg"], &[("room", s("B"))]);
        st.add_node(&["Msg"], &[("room", s("A"))]); // duplicate room A
        st.commit();
        let (scopes, open) = st.touched_scopes("room");
        assert_eq!(str_scopes(&scopes), vec!["A", "B"]); // distinct, sorted
        assert!(!open); // every change was scopable

        // A node with no `room` property → fail-open (visible to all).
        st.begin();
        st.add_node(&["Sys"], &[]);
        st.commit();
        let (scopes2, open2) = st.touched_scopes("room");
        assert!(scopes2.is_empty());
        assert!(open2);
    }

    #[test]
    fn last_write_scope_json_renders_scopes_and_open_flag() {
        let mut st = Builder::default().build();
        st.begin();
        st.add_node(&["Msg"], &[("room", s("A"))]);
        st.add_node(&["Msg"], &[("room", s("B"))]);
        st.commit();
        assert_eq!(
            st.last_write_scope_json("room"),
            r#"{"scopes":["A","B"],"open":false}"#
        );

        // An unscopable change (no `room`) flips open to true.
        st.begin();
        st.add_node(&["Sys"], &[]);
        st.commit();
        assert_eq!(
            st.last_write_scope_json("room"),
            r#"{"scopes":[],"open":true}"#
        );
    }

    #[test]
    fn cdc_reports_delete_as_one_node_deleted() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]);
        let b = st.add_node(&["P"], &[]);
        st.add_edge(a, b, "R");
        st.begin();
        st.delete_node(a); // cascades the edge — reported as one NodeDeleted
        st.commit();
        assert_eq!(st.last_commit_changes(), &[Change::NodeDeleted(a)]);
    }

    #[test]
    fn required_constraint_declared_and_checked() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("a@x"))]);
        // Every User has email → the constraint declares, and the check passes.
        assert!(st.create_required_constraint("User", "email").is_ok());
        assert!(st.check_required_for_label("User").is_ok());
        // A User missing email → the check fails (present-null would pass; absence
        // is the violation).
        st.add_node(&["User"], &[("name", s("b"))]);
        assert!(st.check_required_for_label("User").is_err());
        // Declaring on already-violating data errors.
        let mut st2 = Builder::default().build();
        st2.add_node(&["User"], &[("name", s("x"))]);
        assert!(st2.create_required_constraint("User", "email").is_err());
    }

    #[test]
    fn dotted_path_index_maintained_through_mutations() {
        use crate::value::make_record;
        let city = |c: &str| make_record(vec![(Arc::from("city"), s(c))]);
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("meta", city("NYC"))]);
        let b = st.add_node(&["P"], &[("meta", city("LA"))]);
        let c = st.add_node(&["P"], &[("meta", city("NYC"))]);
        // Built from existing data: index on the record sub-field `meta.city`.
        st.create_index("meta.city");
        let nyc = |st: &Store| {
            let mut v = st.index_lookup("meta.city", &s("NYC")).unwrap();
            v.sort_unstable();
            v
        };
        assert_eq!(nyc(&st), vec![a, c]);
        assert_eq!(st.index_lookup("meta.city", &s("LA")).unwrap(), vec![b]);

        // Maintained on a write: change b's city to NYC.
        st.set_prop(b, "meta", city("NYC"));
        assert_eq!(nyc(&st), vec![a, b, c]);
        // …and on delete.
        st.delete_node(a);
        assert_eq!(nyc(&st), vec![b, c]);
        // No index on this path → None (distinct from an empty match).
        assert!(st.index_lookup("meta.zip", &n(1.0)).is_none());
    }

    /// Build an empty store, add two nodes and an edge, and read it all back —
    /// hand-verified: ids 0 and 1, one out-edge 0→1 mirrored as an in-edge.
    #[test]
    fn add_nodes_and_edge_then_read_back() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        assert_eq!((a, b), (0, 1));
        assert_eq!(st.node_count(), 2);
        st.add_edge(a, b, "R");
        assert_eq!(st.nodes_with_label("P"), &[0, 1]);
        assert_eq!(st.out(a).len(), 1);
        assert_eq!(st.out(a)[0].nbr, b);
        assert_eq!(st.inc(b)[0].nbr, a);
        assert_eq!(st.out(a)[0].eid, st.inc(b)[0].eid); // shared edge id
        assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
    }

    /// Adding a node AFTER a property column exists extends that column with an
    /// absent slot: the old node keeps its value, the new node reads NULL.
    #[test]
    fn add_node_extends_existing_columns() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("age", n(30.0))]);
        let b = st.add_node(&["P"], &[]); // no age
        assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 30.0));
        assert!(st.prop(b, "age").is_null());
    }

    /// Writing a value of a different type promotes the column to `Gen`; both the
    /// old-typed and new value read back correctly.
    #[test]
    fn set_prop_promotes_on_type_change() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("v", n(1.0))]);
        let b = st.add_node(&["P"], &[("v", n(2.0))]);
        st.set_prop(a, "v", s("two")); // Num column, Str value -> promote to Gen
        assert!(matches!(st.prop(a, "v"), Value::Str(x) if &*x == "two"));
        assert!(matches!(st.prop(b, "v"), Value::Num(x) if x == 2.0)); // preserved
    }

    /// `remove_prop` makes the property read NULL again; overwriting sets it.
    #[test]
    fn set_and_remove_prop() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("age", n(30.0))]);
        st.set_prop(a, "age", n(31.0));
        assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 31.0));
        st.remove_prop(a, "age");
        assert!(st.prop(a, "age").is_null());
    }

    /// A repeated edge type interns once (same `etype`) but each edge gets a
    /// distinct `eid`. Ids continue after a `build()`-created edge.
    #[test]
    fn edge_type_interns_once_ids_unique() {
        let mut b = Builder::default();
        let x = b.node(&["P"], &[]);
        let y = b.node(&["P"], &[]);
        b.edge(x, y, "R"); // eid 0 at build
        let mut st = b.build();
        st.add_edge(x, y, "R"); // eid 1, same type
        st.add_edge(x, y, "S"); // eid 2, new type
        assert_eq!(st.out(x).len(), 3);
        let eids: Vec<u32> = st.out(x).iter().map(|a| a.eid).collect();
        assert_eq!(eids, vec![0, 1, 2]); // continued, unique
        assert_eq!(st.out(x)[0].etype, st.out(x)[1].etype); // R == R
        assert_ne!(st.out(x)[1].etype, st.out(x)[2].etype); // R != S
    }

    /// `delete_edge` removes the edge from both endpoints and is idempotent.
    #[test]
    fn delete_edge_detaches_both_sides() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        st.delete_edge(a, b, eid);
        assert!(st.out(a).is_empty());
        assert!(st.inc(b).is_empty());
        st.delete_edge(a, b, eid); // no-op the second time
        assert!(st.out(a).is_empty());
    }

    /// `delete_node` tombstones the node, detaches its edges from the neighbours'
    /// mirror lists, clears its props, and drops it from scans. Hand-traced on
    /// a→b, a→c, b→c: deleting b leaves a→c only, c with one incoming (from a).
    #[test]
    fn label_bucket_stays_sorted_through_delete_rollback() {
        let mut st = Builder::default().build();
        let ids: Vec<u32> = (0..6)
            .map(|i| st.add_node(&["P"], &[("age", n(f64::from(i)))]))
            .collect();
        st.create_index("age");
        // Delete a MIDDLE node in a transaction, then roll back (un-tombstone).
        st.begin();
        st.delete_node(ids[2]);
        st.rollback();
        // The label bucket must be sorted again — the restore re-inserts in place,
        // not appended — so the id-order scan seed and the binary-search label
        // intersection in `index_seek_ids` stay correct.
        let bucket = st.nodes_with_label("P");
        assert_eq!(bucket.len(), 6);
        assert!(
            bucket.windows(2).all(|w| w[0] < w[1]),
            "bucket not sorted after rollback: {bucket:?}"
        );
        // And the hash index still resolves the restored middle node.
        assert_eq!(st.index_lookup("age", &n(2.0)).unwrap(), vec![ids[2]]);
    }

    #[test]
    fn delete_node_tombstones_and_cleans_up() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let c = st.add_node(&["P"], &[("name", s("c"))]);
        st.add_edge(a, b, "R");
        st.add_edge(a, c, "R");
        st.add_edge(b, c, "R");
        st.delete_node(b);

        assert!(!st.is_alive(b));
        assert_eq!(st.all_nodes(), vec![a, c]);
        assert_eq!(st.nodes_with_label("P"), &[a, c]); // b removed from bucket
        assert_eq!(st.out(a).len(), 1); // a→b gone, a→c stays
        assert_eq!(st.out(a)[0].nbr, c);
        assert_eq!(st.inc(c).len(), 1); // b→c gone, a→c stays
        assert_eq!(st.inc(c)[0].nbr, a);
        assert!(st.out(b).is_empty());
        assert!(st.prop(b, "name").is_null()); // props cleared
        assert!(!st.prop(a, "name").is_null()); // neighbour intact
        st.delete_node(b); // idempotent
        assert_eq!(st.all_nodes(), vec![a, c]);
    }

    /// A self-loop is detached without panicking when its node is deleted.
    #[test]
    fn delete_node_with_self_loop() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        st.add_edge(a, a, "R");
        st.delete_node(a);
        assert!(!st.is_alive(a));
        assert!(st.out(a).is_empty());
        assert!(st.inc(a).is_empty());
    }

    // --- Transactions ---

    /// Commit keeps the changes; the log is discarded.
    #[test]
    fn commit_keeps_changes() {
        let mut st = Builder::default().build();
        st.begin();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        st.commit();
        assert_eq!(st.node_count(), 1);
        assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
    }

    /// Rolling back an `add_node` truly removes it: node_count returns to 0 and
    /// the columns shrink back (not merely tombstoned).
    #[test]
    fn rollback_add_node_shrinks_back() {
        let mut st = Builder::default().build();
        st.begin();
        st.add_node(&["P"], &[("name", s("a"))]);
        st.add_node(&["P"], &[("name", s("b"))]);
        assert_eq!(st.node_count(), 2);
        st.rollback();
        assert_eq!(st.node_count(), 0);
        assert!(st.all_nodes().is_empty());
        assert!(st.nodes_with_label("P").is_empty());
    }

    /// Rolling back `set_prop` restores the exact prior cell (present value).
    #[test]
    fn rollback_set_prop_restores_value() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("age", n(30.0))]); // committed (autocommit)
        st.begin();
        st.set_prop(a, "age", n(99.0));
        st.set_prop(a, "age", s("oops")); // also promotes column to Gen
        assert!(matches!(st.prop(a, "age"), Value::Str(x) if &*x == "oops"));
        st.rollback();
        assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 30.0));
    }

    /// Rolling back a newly-set property (absent before) makes it absent again.
    #[test]
    fn rollback_new_prop_becomes_absent() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]);
        st.begin();
        st.set_prop(a, "age", n(30.0));
        st.rollback();
        assert!(st.prop(a, "age").is_null());
        assert!(!st.has_prop(a, "age"));
    }

    /// Rolling back `add_edge` removes it from both endpoints.
    #[test]
    fn rollback_add_edge() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.begin();
        st.add_edge(a, b, "R");
        st.rollback();
        assert!(st.out(a).is_empty());
        assert!(st.inc(b).is_empty());
    }

    /// Rolling back `delete_node` restores it fully: tombstone, adjacency (its own
    /// lists AND the neighbours' mirrors), label membership, and properties.
    /// Hand-traced on a→b, b→c: delete b, then roll back → identical to before.
    #[test]
    fn rollback_delete_node_restores_everything() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let c = st.add_node(&["P"], &[("name", s("c"))]);
        st.add_edge(a, b, "R");
        st.add_edge(b, c, "R");
        st.begin();
        st.delete_node(b);
        assert!(!st.is_alive(b));
        st.rollback();

        assert!(st.is_alive(b));
        assert_eq!(st.nodes_with_label("P").len(), 3);
        assert!(matches!(st.prop(b, "name"), Value::Str(x) if &*x == "b"));
        // adjacency restored on all three nodes
        assert_eq!(st.out(a).len(), 1); // a→b
        assert_eq!(st.out(a)[0].nbr, b);
        assert_eq!(st.out(b).len(), 1); // b→c
        assert_eq!(st.out(b)[0].nbr, c);
        assert_eq!(st.inc(b).len(), 1); // a→b mirror
        assert_eq!(st.inc(c).len(), 1); // b→c mirror
    }

    /// `savepoint` + `rollback_to` give per-statement atomicity: the first
    /// statement's writes survive, the second's are undone, the transaction stays
    /// open, and the final commit keeps only the first.
    #[test]
    fn savepoint_rolls_back_one_statement() {
        let mut st = Builder::default().build();
        st.begin();
        let a = st.add_node(&["P"], &[("name", s("a"))]); // statement 1
        let mark = st.savepoint();
        let b = st.add_node(&["P"], &[("name", s("b"))]); // statement 2
        st.add_edge(a, b, "R");
        st.rollback_to(mark); // undo statement 2 only
        assert_eq!(st.node_count(), 1); // b popped
        assert!(st.out(a).is_empty()); // edge gone
        st.commit();
        assert_eq!(st.node_count(), 1);
        assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
    }

    // --- Edge properties ---

    /// Set / read / remove an edge property, keyed by the edge's eid.
    #[test]
    fn edge_property_set_read_remove() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        assert!(st.edge_prop(eid, "weight").is_null()); // absent
        st.set_edge_prop(eid, "weight", n(0.5));
        assert!(st.has_edge_prop(eid, "weight"));
        assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 0.5));
        st.remove_edge_prop(eid, "weight");
        assert!(!st.has_edge_prop(eid, "weight"));
        assert!(st.edge_prop(eid, "weight").is_null());
    }

    /// The numeric edge overlay tracks the boxed source of truth through every
    /// mutation, and demotes a key that gains a non-numeric value.
    #[test]
    fn edge_num_overlay_tracks_boxed() {
        // Edges via the Builder (as the bench / from_ndjson do), so build() leaves the
        // overlay fresh and the incremental set-maintenance keeps it so.
        let mut b = Builder::default();
        let ids: Vec<u32> = (0..6).map(|_| b.node(&[], &[])).collect();
        for i in 0..5 {
            b.edge(ids[i], ids[i + 1], "R");
        }
        let mut st = b.build();
        let eids: Vec<u32> = (0..5).map(|i| st.out(i)[0].eid).collect();
        // Overlay agrees with edge_prop for every eid, for a numeric key.
        let agree = |st: &Store, key: &str| {
            let Some((data, present)) = st.edge_num_column(key) else {
                return None; // key not overlaid
            };
            for (idx, &eid) in eids.iter().enumerate() {
                let boxed = st.edge_prop(eid, key);
                let ov = if present[eid as usize] {
                    Value::Num(data[eid as usize])
                } else {
                    Value::Null
                };
                assert!(
                    crate::value::equals(&boxed, &ov) || (boxed.is_null() && ov.is_null()),
                    "overlay/boxed mismatch at eid {eid} (row {idx})"
                );
            }
            Some(())
        };
        for &eid in &eids {
            st.set_edge_prop(eid, "w", n(f64::from(eid) * 2.0));
        }
        assert!(
            agree(&st, "w").is_some(),
            "w should be overlaid after bulk set"
        );
        // Remove one, mutate another — overlay still agrees.
        st.remove_edge_prop(eids[1], "w");
        st.set_edge_prop(eids[2], "w", n(99.0));
        agree(&st, "w");
        // A non-numeric write demotes the key (readers fall back to boxed).
        st.set_edge_prop(eids[0], "w", s("hello"));
        assert!(
            st.edge_num_column("w").is_none(),
            "a Str value demotes the overlay"
        );
        assert!(matches!(st.edge_prop(eids[0], "w"), Value::Str(ref x) if &**x == "hello"));
        // A separate numeric key stays overlaid and correct.
        st.set_edge_prop(eids[3], "k", n(7.0));
        st.set_edge_prop(eids[4], "k", n(8.0));
        assert!(agree(&st, "k").is_some());
        // A fresh add_edge invalidates the overlay (eid space grew) → boxed fallback.
        st.add_edge(0, 2, "R");
        assert!(
            st.edge_num_column("k").is_none(),
            "add_edge invalidates the overlay"
        );
        assert!(matches!(st.edge_prop(eids[3], "k"), Value::Num(x) if x == 7.0));
        // boxed still right
    }

    /// An edge property write rolls back with the transaction.
    #[test]
    fn edge_property_rolls_back() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        st.set_edge_prop(eid, "weight", n(1.0)); // committed (autocommit)
        st.begin();
        st.set_edge_prop(eid, "weight", n(2.0));
        st.set_edge_prop(eid, "fresh", s("x"));
        st.rollback();
        assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 1.0)); // restored
        assert!(!st.has_edge_prop(eid, "fresh")); // new key gone
    }

    // --- Unique constraints ---

    /// A unique constraint on already-conforming data is accepted; check passes.
    #[test]
    fn unique_constraint_accepts_conforming_data() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("a@x"))]);
        st.add_node(&["User"], &[("email", s("b@x"))]);
        assert!(st.create_unique_constraint("User", &["email"]).is_ok());
        assert!(st.check_unique_for_label("User").is_ok());
    }

    /// Declaring a constraint the data already violates errors.
    #[test]
    fn unique_constraint_rejects_existing_duplicate() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("dup"))]);
        st.add_node(&["User"], &[("email", s("dup"))]);
        assert!(st.create_unique_constraint("User", &["email"]).is_err());
    }

    /// After a constraint, a duplicate added at the store level is detected by the
    /// check (the store primitive itself stays infallible; enforcement is the
    /// caller's, as the write statements do).
    #[test]
    fn unique_check_detects_new_duplicate() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("x"))]);
        st.create_unique_constraint("User", &["email"]).unwrap();
        st.add_node(&["User"], &[("email", s("x"))]); // primitive allows it
        assert!(st.check_unique_for_label("User").is_err()); // check catches it
    }

    /// Conflict-target inference: the constraint keys are returned when the
    /// pattern's key set covers them.
    #[test]
    fn unique_keys_for_infers_target() {
        let mut st = Builder::default().build();
        st.create_unique_constraint("User", &["email"]).unwrap();
        assert_eq!(
            st.unique_keys_for("User", &["email".into(), "name".into()]),
            Some(vec!["email".into()])
        );
        assert_eq!(st.unique_keys_for("User", &["name".into()]), None);
        assert_eq!(st.unique_keys_for("Other", &["email".into()]), None);
    }

    /// `transaction` commits on `Ok` and rolls back on `Err`.
    #[test]
    fn transaction_commits_ok_rolls_back_err() {
        let mut st = Builder::default().build();
        let r: Result<u32, ()> = st.transaction(|s| Ok(s.add_node(&["P"], &[])));
        assert!(r.is_ok());
        assert_eq!(st.node_count(), 1);

        let r: Result<(), &str> = st.transaction(|s| {
            s.add_node(&["P"], &[]);
            Err("boom")
        });
        assert_eq!(r, Err("boom"));
        assert_eq!(st.node_count(), 1); // the aborted add was rolled back
    }

    // --- opt-in edge-type index (G5) ---

    /// A small multi-type graph: node 0 knows 1 and likes 2; node 1 knows 2.
    fn typed_graph() -> Store {
        let mut b = Builder::default();
        for _ in 0..3 {
            b.node(&["V"], &[]);
        }
        b.edge(0, 1, "KNOWS");
        b.edge(0, 2, "LIKES");
        b.edge(1, 2, "KNOWS");
        b.build()
    }

    /// The typed neighbours of `node` along `etype` (out), as a sorted id list.
    fn out_ids(st: &Store, node: u32, ty: &str) -> Vec<u32> {
        let et = st.etype_id(ty).unwrap();
        let mut v: Vec<u32> = st.out_typed(node, et).iter().map(|a| a.nbr).collect();
        v.sort_unstable();
        v
    }

    /// The index, once built, agrees with a manual type-filter of the flat
    /// adjacency for every node and type.
    #[test]
    fn edge_type_index_matches_flat_scan() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        assert!(st.has_edge_type_index());
        for node in 0..st.node_count() as u32 {
            for ty in ["KNOWS", "LIKES"] {
                let et = st.etype_id(ty).unwrap();
                let mut scan: Vec<u32> = st
                    .out(node)
                    .iter()
                    .filter(|a| a.etype == et)
                    .map(|a| a.nbr)
                    .collect();
                scan.sort_unstable();
                assert_eq!(out_ids(&st, node, ty), scan, "node {node} type {ty}");
            }
        }
        // node 0: KNOWS -> {1}, LIKES -> {2}
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1]);
        assert_eq!(out_ids(&st, 0, "LIKES"), vec![2]);
    }

    /// add_edge keeps the index current (the O(1) hot path).
    #[test]
    fn edge_type_index_tracks_add() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        st.add_edge(0, 2, "KNOWS");
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1, 2]); // 0 now KNOWS 1 and 2
    }

    /// delete_edge and delete_node keep the index current, including neighbours'
    /// incoming buckets.
    #[test]
    fn edge_type_index_tracks_delete() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        let et = st.etype_id("KNOWS").unwrap();
        // delete 0-KNOWS->1: gone from 0's out bucket AND 1's in bucket.
        st.delete_edge(0, 1, 0);
        assert_eq!(out_ids(&st, 0, "KNOWS"), Vec::<u32>::new());
        let in1: Vec<u32> = st.in_typed(1, et).iter().map(|a| a.nbr).collect();
        assert_eq!(in1, Vec::<u32>::new());
        // delete node 2: removes 0-LIKES->2 and 1-KNOWS->2 mirrors.
        st.delete_node(2);
        assert_eq!(out_ids(&st, 0, "LIKES"), Vec::<u32>::new());
        assert_eq!(out_ids(&st, 1, "KNOWS"), Vec::<u32>::new());
    }

    /// Transaction rollback restores the index exactly (a per-node rebuild off the
    /// restored flat adjacency, so no delta bookkeeping can drift).
    #[test]
    fn edge_type_index_survives_rollback() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        st.begin();
        st.add_edge(0, 2, "KNOWS"); // 0 KNOWS {1,2} inside the txn
        st.delete_edge(1, 2, 2); // 1 KNOWS {} inside the txn
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1, 2]);
        st.rollback();
        // Back to the committed shape: 0 KNOWS {1}, 1 KNOWS {2}.
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1]);
        assert_eq!(out_ids(&st, 1, "KNOWS"), vec![2]);
    }

    /// A node added AFTER the index exists grows the index and indexes its edges.
    #[test]
    fn edge_type_index_grows_with_new_node() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        let three = st.add_node(&["V"], &[]);
        st.add_edge(three, 0, "LIKES");
        assert_eq!(out_ids(&st, three, "LIKES"), vec![0]);
    }

    // --- opt-in edge interval index (G4) ---

    /// One Emp node (0) with `degree` HELD edges to role node 1, edge d carrying
    /// interval `[d, d+width]`.
    fn interval_graph(degree: u32, width: i64) -> Store {
        let mut b = Builder::default();
        b.node(&["Emp"], &[]);
        b.node(&["Role"], &[]);
        let mut st = b.build();
        for d in 0..degree {
            let eid = st.add_edge(0, 1, "HELD");
            st.set_edge_prop(eid, "vf", n(f64::from(d)));
            st.set_edge_prop(eid, "vt", n((i64::from(d) + width) as f64));
        }
        st
    }

    /// Overlap eids from the index (sorted), vs a brute-force scan of the flat
    /// adjacency reading the boxed props.
    fn overlap_eids(st: &Store, node: u32, qlo: f64, qhi: f64) -> Vec<u32> {
        let mut v = Vec::new();
        st.for_each_overlap(node, qlo, qhi, |eid, _| v.push(eid));
        v.sort_unstable();
        v
    }
    fn overlap_bruteforce(st: &Store, node: u32, qlo: f64, qhi: f64) -> Vec<u32> {
        let mut v: Vec<u32> = st
            .out(node)
            .iter()
            .filter(|a| {
                matches!((st.edge_prop(a.eid, "vf"), st.edge_prop(a.eid, "vt")),
                    (Value::Num(lo), Value::Num(hi)) if lo <= qhi && hi >= qlo)
            })
            .map(|a| a.eid)
            .collect();
        v.sort_unstable();
        v
    }

    /// The seek agrees with a brute-force overlap scan for point queries across the
    /// timeline AND for wider interval queries (both seed axes exercised).
    #[test]
    fn interval_seek_matches_bruteforce() {
        let st = {
            let mut s = interval_graph(64, 4);
            s.create_interval_index("vf", "vt");
            s
        };
        assert!(st.has_interval_index("vf", "vt"));
        // as-of points across the whole timeline (0..=67), incl. the ends where one
        // axis is far more selective than the other.
        for t in 0..=67 {
            let q = f64::from(t);
            assert_eq!(
                overlap_eids(&st, 0, q, q),
                overlap_bruteforce(&st, 0, q, q),
                "point t={t}"
            );
        }
        // wider ranges
        for &(lo, hi) in &[(10.0, 20.0), (0.0, 100.0), (63.0, 63.0), (-5.0, 2.0)] {
            assert_eq!(
                overlap_eids(&st, 0, lo, hi),
                overlap_bruteforce(&st, 0, lo, hi),
                "range [{lo},{hi}]"
            );
        }
    }

    /// Writes keep the interval index current: a new edge+interval appears, a
    /// changed interval moves, a deleted edge vanishes.
    #[test]
    fn interval_index_tracks_writes() {
        let mut st = interval_graph(4, 2); // edges: [0,2],[1,3],[2,4],[3,5]
        st.create_interval_index("vf", "vt");
        // as-of t=10 → none.
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
        // add an edge covering t=10.
        let e = st.add_edge(0, 1, "HELD");
        st.set_edge_prop(e, "vf", n(8.0));
        st.set_edge_prop(e, "vt", n(12.0));
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), vec![e]);
        // move it off t=10.
        st.set_edge_prop(e, "vt", n(9.0));
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
        // delete it.
        st.set_edge_prop(e, "vt", n(12.0));
        st.delete_edge(0, 1, e);
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
    }

    /// Rollback restores the interval index exactly (a full rebuild against the
    /// restored graph, so prop AND adjacency undo ordering can't drift it).
    #[test]
    fn interval_index_survives_rollback() {
        let mut st = interval_graph(4, 2);
        st.create_interval_index("vf", "vt");
        let before: Vec<Vec<u32>> = (0..8)
            .map(|t| overlap_eids(&st, 0, f64::from(t), f64::from(t)))
            .collect();
        st.begin();
        let e = st.add_edge(0, 1, "HELD");
        st.set_edge_prop(e, "vf", n(0.0));
        st.set_edge_prop(e, "vt", n(100.0)); // covers everything, inside the txn
        st.delete_edge(0, 1, 0); // and drop the first committed edge
        assert!(overlap_eids(&st, 0, 3.0, 3.0).contains(&e));
        st.rollback();
        let after: Vec<Vec<u32>> = (0..8)
            .map(|t| overlap_eids(&st, 0, f64::from(t), f64::from(t)))
            .collect();
        assert_eq!(before, after);
    }

    // --- type / cardinality / edge / drop-index constraints ---------------

    #[test]
    fn type_constraint_enforced_on_write() {
        let mut st = Store::default();
        assert!(st
            .create_type_constraint("P", "age", "number", false)
            .is_ok());
        st.begin();
        st.add_node(&["P"], &[("age", Value::Str("old".into()))]);
        assert!(
            st.run_deferred_checks().is_err(),
            "string age violates number"
        );
        st.rollback();
        st.begin();
        st.add_node(&["P"], &[("age", n(42.0))]);
        assert!(st.run_deferred_checks().is_ok());
        st.commit();
    }

    #[test]
    fn type_constraint_unknown_name_is_invalid_value() {
        let mut st = Store::default();
        let e = st
            .create_type_constraint("P", "age", "bogus", false)
            .unwrap_err();
        assert!(e.starts_with("E_INVALID_VALUE"), "{e}");
    }

    #[test]
    fn not_null_type_constraint_rejects_absent_and_null() {
        let mut st = Store::default();
        st.create_type_constraint("P", "name", "string NOT NULL", false)
            .unwrap();
        for missing in [vec![], vec![("name", Value::Null)]] {
            st.begin();
            st.add_node(&["P"], &missing);
            assert!(st.run_deferred_checks().is_err(), "NOT NULL must reject");
            st.rollback();
        }
        st.begin();
        st.add_node(&["P"], &[("name", Value::Str("ann".into()))]);
        assert!(st.run_deferred_checks().is_ok());
        st.commit();
    }

    #[test]
    fn cardinality_min_and_max_enforced() {
        let mut st = Store::default();
        // out-degree of KNOWS must be exactly 1.
        st.create_cardinality_constraint("P", "KNOWS", 0, 1, Some(1))
            .unwrap();
        st.begin();
        st.add_node(&["P"], &[]); // 0 edges → below min
        assert!(st.run_deferred_checks().is_err());
        st.rollback();
        st.begin();
        let a = st.add_node(&["P"], &[]);
        let b = st.add_node(&["T"], &[]);
        let c = st.add_node(&["T"], &[]);
        st.add_edge(a, b, "KNOWS");
        st.add_edge(a, c, "KNOWS"); // out-degree 2 → above max
        assert!(st.run_deferred_checks().is_err());
        st.rollback();
        st.begin();
        let a = st.add_node(&["P"], &[]);
        let b = st.add_node(&["T"], &[]);
        st.add_edge(a, b, "KNOWS"); // exactly 1
        assert!(st.run_deferred_checks().is_ok());
        st.commit();
    }

    #[test]
    fn edge_unique_enforced_and_null_exempt() {
        let mut st = Store::default();
        st.create_edge_unique_constraint("PAID", &["ref"]).unwrap();
        st.begin();
        let a = st.add_node(&["A"], &[]);
        let b = st.add_node(&["A"], &[]);
        let e1 = st.add_edge(a, b, "PAID");
        st.set_edge_prop(e1, "ref", Value::Str("x".into()));
        let e2 = st.add_edge(a, b, "PAID");
        st.set_edge_prop(e2, "ref", Value::Str("x".into())); // duplicate
        assert!(st.run_deferred_checks().is_err());
        st.rollback();
        // Two edges with an absent `ref` are exempt (nulls don't collide).
        st.begin();
        let a = st.add_node(&["A"], &[]);
        let b = st.add_node(&["A"], &[]);
        st.add_edge(a, b, "PAID");
        st.add_edge(a, b, "PAID");
        assert!(st.run_deferred_checks().is_ok());
        st.commit();
    }

    #[test]
    fn edge_required_enforced_on_write() {
        let mut st = Store::default();
        st.create_edge_required_constraint("PAID", "amt").unwrap();
        st.begin();
        let a = st.add_node(&["A"], &[]);
        let b = st.add_node(&["A"], &[]);
        st.add_edge(a, b, "PAID"); // missing amt
        assert!(st.run_deferred_checks().is_err());
        st.rollback();
    }

    #[test]
    fn drop_index_removes_and_guards_a_backing_unique() {
        let mut st = Store::default();
        st.add_node(&["P"], &[("age", n(1.0))]);
        st.create_index("age");
        assert!(st.has_hash_index("age"));
        assert!(st.drop_vertex_index("age").is_ok());
        assert!(!st.has_hash_index("age"));
        assert!(st.drop_vertex_index("age").is_ok(), "drop is idempotent");
        // Dropping the index behind a unique constraint is rejected.
        st.create_index("email");
        st.create_unique_constraint("P", &["email"]).unwrap();
        let e = st.drop_vertex_index("email").unwrap_err();
        assert!(e.starts_with("E_INVALID_GRAPH_OP"), "{e}");
    }

    #[test]
    fn declaring_a_constraint_the_data_breaks_is_rejected() {
        let mut st = Store::default();
        st.add_node(&["P"], &[("age", Value::Str("nope".into()))]);
        let e = st
            .create_type_constraint("P", "age", "number", false)
            .unwrap_err();
        assert!(e.starts_with("E_TYPE"), "{e}");
    }

    /// A CLOSED record type: field types + NOT NULL fields + closed-on-extras, with
    /// a nested record. A node whose `m` is a matching record accepts the constraint;
    /// each way of breaking the shape rejects it.
    #[test]
    fn closed_record_type_constraint() {
        let spec = "record{a::number,b::string NOT NULL,c::record{d::boolean}}";
        // `m` built from a nested JSON object → a canonical Value::Record.
        let node = |m: &str| {
            crate::ndjson::from_ndjson(&format!(
                "{{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{{\"m\":{m}}}}}\n"
            ))
            .unwrap()
        };
        let declare = |mut st: Store| st.create_type_constraint("P", "m", spec, false);

        // Conforming (optional `a`/`c` omitted; required `b` present; nested ok).
        assert!(declare(node(r#"{"b":"x"}"#)).is_ok());
        assert!(declare(node(r#"{"a":1,"b":"x","c":{"d":true}}"#)).is_ok());
        // Wrong scalar field type.
        assert!(declare(node(r#"{"a":"nope","b":"x"}"#)).is_err());
        // Missing a NOT NULL field.
        assert!(declare(node(r#"{"a":1}"#)).is_err());
        // Extra field (closed on extras).
        assert!(declare(node(r#"{"b":"x","z":2}"#)).is_err());
        // Nested field wrong type.
        assert!(declare(node(r#"{"b":"x","c":{"d":5}}"#)).is_err());
        // A null property is exempt (the property is nullable without NOT NULL).
        assert!(declare(node("null")).is_ok());
        // A non-record value violates a record type.
        assert!(declare(node("42")).is_err());
    }

    #[test]
    fn any_record_type_constraint() {
        let node = |m: &str| {
            crate::ndjson::from_ndjson(&format!(
                "{{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{{\"m\":{m}}}}}\n"
            ))
            .unwrap()
        };
        // `any record` accepts any record shape but rejects a scalar.
        assert!(node(r#"{"anything":1,"here":true}"#)
            .create_type_constraint("P", "m", "any record", false)
            .is_ok());
        assert!(node("42")
            .create_type_constraint("P", "m", "any record", false)
            .is_err());
    }

    #[test]
    fn record_type_name_round_trips() {
        // A declared record type dumps a spec that parses back to the same rule.
        let mut st = crate::ndjson::from_ndjson(
            "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"m\":{\"a\":1,\"b\":\"x\"}}}\n",
        )
        .unwrap();
        let spec = "record{a::number,b::string NOT NULL}";
        st.create_type_constraint("P", "m", spec, false).unwrap();
        let (_, _, dumped, _) = st.type_constraints().into_iter().next().unwrap();
        // Re-declaring from the dumped name succeeds on the same data (round-trips).
        let mut st2 = crate::ndjson::from_ndjson(
            "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"m\":{\"a\":1,\"b\":\"x\"}}}\n",
        )
        .unwrap();
        assert!(
            st2.create_type_constraint("P", "m", &dumped, false).is_ok(),
            "dumped: {dumped}"
        );
    }
}
