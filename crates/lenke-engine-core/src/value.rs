//! The value contract: the representation AND the semantics, in one place.
//!
//! This is the module the design says to pin first. Every other layer — storage,
//! the batch operators, the eventual front-ends — consults it and never restates
//! its rules. There is exactly one answer here to each of: how two values
//! compare (`cmp_total`), whether they are equal (`equals`), whether a value is
//! null (`is_null`), and whether a predicate value keeps a row (`is_true`).
//!
//! Policy, stated once so it cannot drift:
//! - **One numeric type**: `f64`. There is no integer/float split.
//! - **Null is a stored value**, not the absence of one. `is_null` is true only
//!   for `Null`.
//! - **NaN in the total order is the greatest number** (so sorts/min/max are
//!   deterministic), but **NaN is never equal to anything, including itself**,
//!   under `equals` (the predicate `=`), matching IEEE and JS.
//! - **`-0.0` and `0.0` are equal** and compare equal.
//! - **Cross-type `equals` is simply false** — a number never equals a string —
//!   rather than an error. Cross-type *ordering* is a language-level decision the
//!   front-end makes; `cmp_total` itself is total and never fails, because sorts
//!   and grouping need a deterministic order over any mix.

use crate::gstr::GStr;
use crate::temporal::Temporal;
use std::cmp::Ordering;
use std::sync::Arc;

/// A runtime value. Map/record values join this enum as later slices land, and
/// every addition gets its arm in the functions below — which is the point of one
/// contract.
#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(GStr),
    /// An ISO temporal value (`DATE`/`LOCAL TIME`/`LOCAL DATETIME`). Ordering and
    /// equality delegate to [`Temporal`], keeping the rules there; this contract
    /// only decides where temporals sit in the CROSS-type order (see `rank`).
    Temporal(Temporal),
    /// An ordered list of values — e.g. a path (its node ids). Added with the
    /// lineage slice; every contract function below gains its arm.
    List(Vec<Value>),
    /// An ISO `<record>`: string field names, **kept sorted** so equality is a
    /// pairwise slice compare and the wire form is canonical. `Arc`-boxed so a
    /// per-row binding clone is a refcount bump. Build via [`make_record`], which
    /// sorts and de-duplicates keys (last write wins).
    Record(Arc<[(GStr, Value)]>),
    /// A TinkerPop map (Gremlin): **any** value as a key, **insertion-ordered** —
    /// `valueMap`/`project`/`select` preserve the order the traversal produced. So
    /// equality and ordering are POSITIONAL (order-sensitive), unlike a `Record`.
    /// `Arc`-boxed for a cheap per-row clone.
    Map(Arc<Vec<(Value, Value)>>),
    /// An UNBOXED graph element reference carrying its dense id — a node or an edge that
    /// has flowed into a VALUE position (a heterogeneous `Col::Gen`, e.g. Gremlin
    /// `inject(…)` or a mixed branch) WITHOUT being rendered to an element map. Lets a
    /// downstream `out()`/`hasLabel()`/`values()` recognize and traverse it (identity
    /// preserved), and renders to its element map only at egress. `Col::Nodes`/`Edges`
    /// frontiers stay the unboxed representation; this is the value-column carrier.
    Node(u32),
    Edge(u32),
}

/// Build a [`Value::Record`] from `pairs`: duplicate keys collapse (last write
/// wins) and the fields are sorted by key, so two records with the same contents
/// are byte-identical and equality is a slice compare. The ONE place a record is
/// canonicalized.
#[must_use]
pub fn make_record(pairs: Vec<(GStr, Value)>) -> Value {
    let mut out: Vec<(GStr, Value)> = Vec::with_capacity(pairs.len());
    for (k, v) in pairs {
        if let Some(slot) = out.iter_mut().find(|(ek, _)| *ek == k) {
            slot.1 = v; // last write wins
        } else {
            out.push((k, v));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Value::Record(out.into())
}

/// Look up `key` in a record's sorted fields; `Null` when absent.
#[must_use]
pub fn record_field(fields: &[(GStr, Value)], key: &str) -> Value {
    fields
        .binary_search_by(|(k, _)| k.as_ref().cmp(key))
        .map_or(Value::Null, |i| fields[i].1.clone())
}

impl Value {
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Does this value keep a row when used as a filter predicate? Three-valued:
    /// only a literal TRUE keeps; FALSE and UNKNOWN (null, or a non-boolean) drop.
    #[must_use]
    pub fn is_true(&self) -> bool {
        matches!(self, Self::Bool(true))
    }

    /// The type rank for the cross-type total order. Kept private: only
    /// `cmp_total` should depend on the numbering.
    const fn rank(&self) -> u8 {
        // Cross-type sort rank, matching core's `type_rank` (Num < Str < Bool <
        // Temporal < compound) so ORDER BY / min / max / list_sort over a mixed
        // column agrees byte-for-byte. Compound kinds keep distinct ranks (their
        // relative order beyond core's shared "else" only matters for mixed-compound
        // sorts, which stay as they were).
        match self {
            Self::Num(_) => 0,
            Self::Str(_) => 1,
            Self::Bool(_) => 2,
            Self::Temporal(_) => 3,
            Self::List(_) => 4,
            Self::Record(_) => 5,
            Self::Map(_) => 6,
            // SPIKE: unboxed element refs. Placed above the compound kinds; a mixed
            // element-vs-scalar ORDER BY is exotic — refine against TS if the fuzzer flags it.
            Self::Node(_) => 8,
            Self::Edge(_) => 9,
            // Null sorts LAST — it is the greatest in the total order.
            Self::Null => 7,
        }
    }
}

/// Predicate equality (`=`). Three-valued at the language level, but this returns
/// a concrete bool: cross-type is false, NaN is equal to nothing, `-0.0 == 0.0`.
/// A caller that needs the NULL-propagating three-valued form checks `is_null`
/// first (null `=` anything is UNKNOWN, which is not this function's job).
#[must_use]
pub fn equals(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        // `==` on f64 is already NaN-unequal and treats -0.0 == 0.0.
        (Value::Num(x), Value::Num(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        // Same-kind temporals compare by their fields; cross-kind is false (a Date
        // never equals a DateTime), which `Temporal`'s derived `==` already gives.
        (Value::Temporal(x), Value::Temporal(y)) => x == y,
        // Lists are equal elementwise (same length, each pair equal). A NaN
        // element still makes the lists unequal, as `equals` does per element.
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| equals(p, q))
        }
        // Records are equal iff the same sorted (key, value) fields — keys by
        // string equality, values recursively. A NaN value makes them unequal.
        (Value::Record(x), Value::Record(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|((k1, v1), (k2, v2))| k1 == k2 && equals(v1, v2))
        }
        // A Map compares POSITIONALLY (insertion order matters): same length, and
        // each key/value pair equal in order.
        (Value::Map(x), Value::Map(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y.iter())
                    .all(|((k1, v1), (k2, v2))| equals(k1, k2) && equals(v1, v2))
        }
        // Element refs are equal iff the same kind AND the same dense id.
        (Value::Node(x), Value::Node(y)) | (Value::Edge(x), Value::Edge(y)) => x == y,
        _ => false,
    }
}

/// A canonical, hashable grouping key. This is DISTINCT from `equals`: grouping
/// (GROUP BY, DISTINCT, `count(DISTINCT …)`) must treat two NaNs as the same
/// group and `-0.0`/`0.0` as the same group — the opposite of predicate
/// equality, where NaN equals nothing. Defining it here, once, is why the two
/// never drift.
///
/// The key is raw bytes, not text: a leading type tag keeps types apart, and
/// every value is self-delimiting (fixed width, or length-prefixed) so keys
/// concatenate unambiguously — a multi-column group key is just the columns'
/// bytes in order, with no separator. Hot callers append into a reused buffer
/// via [`group_key_into`] to avoid a per-row allocation.
#[must_use]
pub fn group_key(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    group_key_into(v, &mut out);
    out
}

/// A value's number for ARITHMETIC: a FINITE `Num` only. Non-finite (`NaN`/`Inf`)
/// and non-numeric values return `None`, so arithmetic on them yields NULL — the
/// engine's "NaN/Inf → null" numeric policy, defined here once. Callers combine
/// two `as_num`s and re-check finiteness of the RESULT (e.g. `1/0`).
#[must_use]
pub fn as_num(v: &Value) -> Option<f64> {
    match v {
        Value::Num(x) if x.is_finite() => Some(*x),
        _ => None,
    }
}

/// Any `Num`, INCLUDING NaN / ±Inf — the operand gate for IEEE arithmetic and the
/// numeric scalar functions, which propagate non-finite results (a computed NaN is
/// a real signal — e.g. `sqrt(-1)` — kept internally like lenke-core, and coerced
/// to null only at the JSON egress boundary). Contrast [`as_num`], which is
/// finite-only for callers that need a usable index/count (`substring`, `left`).
#[must_use]
pub fn num_of(v: &Value) -> Option<f64> {
    match v {
        Value::Num(x) => Some(*x),
        _ => None,
    }
}

/// The target type of a `CAST`, mirrored from `ir::CastTarget` so the conversion
/// table lives beside the rest of the value contract. Kept in sync by the
/// `From<ir::CastTarget>` below — a new target forces an arm here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastTarget {
    Integer,
    Float,
    String,
    Boolean,
    List,
}

/// Coerce `v` to `target`, the ONE home for the conversion table. Policy (fixed
/// by the design decision): a failed conversion is `Err(E_INVALID_VALUE)` — the
/// caller turns that into a thrown error — while a NULL input is NULL for every
/// target. `INTEGER` truncates toward zero (still an `f64`, since the engine has
/// one numeric type); `Float` is the identity on numbers. The set of accepted
/// source→target pairs is deliberately broad: string↔number, anything→string,
/// number↔bool (`0`/nonzero), and string→bool (`"true"`/`"false"`).
///
/// # Errors
/// Returns `Err` with an `E_INVALID_VALUE` message when the source value has no
/// conversion to `target` (a non-numeric string to a number, a non-`true`/`false`
/// string to a boolean, a non-finite number to an integer, a list to a scalar).
pub fn cast(v: &Value, target: CastTarget) -> Result<Value, String> {
    // NULL casts to NULL for every target — the one rule that precedes the table.
    if v.is_null() {
        return Ok(Value::Null);
    }
    let bad = |from: &str, to: &str| Err(format!("E_INVALID_VALUE: cannot cast {from} to {to}"));
    match target {
        CastTarget::Integer | CastTarget::Float => {
            let n = match v {
                Value::Num(x) => *x,
                Value::Bool(b) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                Value::Str(s) => s
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| format!("E_INVALID_VALUE: cannot cast string {s:?} to number"))?,
                Value::List(_) => return bad("list", "number"),
                Value::Temporal(_) => return bad("temporal", "number"),
                Value::Record(_) => return bad("record", "number"),
                Value::Map(_) => return bad("map", "number"),
                Value::Node(_) => return bad("node", "number"),
                Value::Edge(_) => return bad("edge", "number"),
                Value::Null => unreachable!("null handled above"),
            };
            if target == CastTarget::Integer {
                // Truncate toward zero. A non-finite value has no integer form.
                if !n.is_finite() {
                    return bad("non-finite number", "integer");
                }
                Ok(Value::Num(n.trunc()))
            } else {
                Ok(Value::Num(n))
            }
        }
        CastTarget::String => Ok(Value::Str(GStr::from(
            match v {
                Value::Str(s) => return Ok(Value::Str(s.clone())),
                // Numbers render exactly as they do on JSON egress — JS `Number.toString`
                // (`crate::json_fmt::js_number`), NOT Rust's decimal-at-all-magnitudes `{}`; a
                // non-finite number has no textual form here.
                Value::Num(x) => {
                    if x.is_finite() {
                        crate::json_fmt::js_number(*x)
                    } else {
                        return bad("non-finite number", "string");
                    }
                }
                Value::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
                // A temporal casts to its ISO-8601 string.
                Value::Temporal(t) => t.format(),
                // A composite serializes rather than faulting: a list joins its elements
                // with "," and a record/map renders as JSON — same as `to_string` and
                // core's `js_str`.
                Value::List(_) | Value::Record(_) | Value::Map(_) => crate::json_fmt::js_str(v),
                Value::Node(_) => return bad("node", "string"),
                Value::Edge(_) => return bad("edge", "string"),
                Value::Null => unreachable!("null handled above"),
            }
            .as_str(),
        ))),
        // `CAST(x AS LIST)`: a list passes through; a string splits into its UTF-16
        // code-unit characters (the `split('')` unit model); any other non-null scalar
        // becomes a singleton list. Matches core's `to_list`.
        CastTarget::List => match v {
            Value::List(_) => Ok(v.clone()),
            Value::Str(s) => Ok(Value::List(
                s.encode_utf16()
                    .map(|u| Value::Str(GStr::from(String::from_utf16_lossy(&[u]).as_str())))
                    .collect(),
            )),
            other => Ok(Value::List(vec![other.clone()])),
        },
        CastTarget::Boolean => match v {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            // Numeric truthiness: zero is false, any other FINITE number is true. A NaN is
            // a data exception — it is a live value in GQL (only nulled at JSON egress), so
            // casting it to a boolean is an invalid cast that faults HERE (at the cast),
            // not a null that trips a type error at some later consumer. (`Inf` is nonzero
            // → true, matching the fn form.)
            Value::Num(x) if x.is_nan() => bad("NaN", "boolean"),
            Value::Num(x) => Ok(Value::Bool(*x != 0.0)),
            // The SQL boolean spellings (case-insensitive, trimmed); an unrecognized
            // string is a data exception (strict CAST — `CAST('1' AS INT)` throws too).
            Value::Str(s) => match s.trim().to_ascii_lowercase().as_str() {
                "t" | "true" | "y" | "yes" | "on" | "1" => Ok(Value::Bool(true)),
                "f" | "false" | "n" | "no" | "off" | "0" => Ok(Value::Bool(false)),
                _ => bad(&format!("string {s:?}"), "boolean"),
            },
            Value::List(_) => bad("list", "boolean"),
            Value::Temporal(_) => bad("temporal", "boolean"),
            Value::Record(_) => bad("record", "boolean"),
            Value::Map(_) => bad("map", "boolean"),
            Value::Node(_) => bad("node", "boolean"),
            Value::Edge(_) => bad("edge", "boolean"),
            Value::Null => unreachable!("null handled above"),
        },
    }
}

/// The canonical bit pattern a number groups by: all NaNs collapse to one
/// pattern, and `-0.0`/`0.0` collapse to `+0.0` — so each groups with its kind,
/// the inverse of predicate equality. This is the ONE place that rule lives; a
/// typed grouping fast path over an `f64` column keys on this instead of boxing
/// each value through [`group_key_into`].
#[must_use]
pub fn num_group_bits(x: f64) -> u64 {
    if x.is_nan() {
        f64::NAN.to_bits()
    } else if x == 0.0 {
        0.0f64.to_bits()
    } else {
        x.to_bits()
    }
}

/// Append `v`'s grouping key bytes to `out` — the allocation-free primitive
/// [`group_key`] wraps. Grouping semantics (NaN canonicalization, `-0.0`/`0.0`
/// collapse, type separation) live here and nowhere else.
pub fn group_key_into(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Null => out.push(0),
        Value::Bool(b) => {
            out.push(1);
            out.push(u8::from(*b));
        }
        Value::Num(x) => {
            out.push(2);
            out.extend_from_slice(&num_group_bits(*x).to_le_bytes());
        }
        Value::Str(t) => {
            out.push(3);
            // Length-prefixed so a string's bytes cannot bleed into the next key.
            out.extend_from_slice(&(t.len() as u64).to_le_bytes());
            out.extend_from_slice(t.as_bytes());
        }
        Value::Temporal(t) => {
            out.push(4);
            // Two temporals group together iff they render identically AND are the
            // same kind. The ISO string is canonical (equal temporals format the
            // same); the kind tag keeps a Date and a Time that happen to render
            // alike apart.
            let iso = t.format();
            out.extend_from_slice(t.tag().as_bytes());
            out.push(0); // tag/value separator (tags are ascii, never contain NUL)
            out.extend_from_slice(&(iso.len() as u64).to_le_bytes());
            out.extend_from_slice(iso.as_bytes());
        }
        Value::List(items) => {
            out.push(5);
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                group_key_into(it, out);
            }
        }
        Value::Record(fields) => {
            // Keys are sorted (canonical), so the byte key is stable: field count,
            // then each (length-prefixed key, value key).
            out.push(6);
            out.extend_from_slice(&(fields.len() as u64).to_le_bytes());
            for (k, v) in fields.iter() {
                out.extend_from_slice(&(k.len() as u64).to_le_bytes());
                out.extend_from_slice(k.as_bytes());
                group_key_into(v, out);
            }
        }
        Value::Map(pairs) => {
            // Insertion order is significant, so the key is the pairs in order —
            // each key's own group_key then its value's.
            out.push(7);
            out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
            for (k, v) in pairs.iter() {
                group_key_into(k, out);
                group_key_into(v, out);
            }
        }
        // Element refs group by kind tag + dense id, so DISTINCT over a node stream dedups
        // by identity (`dedup()` on a heterogeneous stream).
        Value::Node(id) => {
            out.push(8);
            out.extend_from_slice(&id.to_le_bytes());
        }
        Value::Edge(id) => {
            out.push(9);
            out.extend_from_slice(&id.to_le_bytes());
        }
    }
}

/// The THREE-VALUED comparison the ordering OPERATORS (`<` `<=` `>` `>=`) use:
/// `None` (UNKNOWN) whenever the two values are not comparable — different types,
/// or a NaN operand (IEEE-unordered). This is distinct from [`cmp_total`], which
/// imposes a deterministic TOTAL order for sort/min/max/grouping; the operators
/// must instead yield UNKNOWN (→ NULL, → a dropped WHERE row) on incomparable
/// operands, matching lenke-core and SQL/Cypher 3VL. Only same-type scalars are
/// comparable here; temporals compare only within the same kind.
#[must_use]
pub fn cmp_partial(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x.partial_cmp(y), // None iff a NaN operand
        (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
        (Value::Str(x), Value::Str(y)) => Some(x.cmp(y)),
        // Same-kind temporals order chronologically — EXCEPT two durations, which are only
        // PARTIALLY ordered (W3C XML Schema): an incomparable pair (month vs spanning days)
        // is UNKNOWN. `partial_cmp_pred` handles the split (sort/min/max still use cmp_total).
        (Value::Temporal(x), Value::Temporal(y)) if x.kind() == y.kind() => x.partial_cmp_pred(y),
        // Different types (incl. cross-kind temporals), collections, or null:
        // incomparable via an ordering operator → UNKNOWN.
        _ => None,
    }
}

/// A deterministic total order over ANY pair of values — never panics. Used by
/// sort, min/max, and grouping tiebreaks. Nulls sort last; NaN is the greatest
/// number; `-0.0` and `0.0` tie. Distinct types order by rank so a mixed column
/// still has one stable order.
#[must_use]
pub fn cmp_total(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Num(x), Value::Num(y)) => cmp_num_total(*x, *y),
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
        // Temporals order by kind then chronologically — the rule lives in
        // `Temporal::cmp_total`; this only dispatches to it.
        (Value::Temporal(x), Value::Temporal(y)) => x.cmp_total(y),
        // Lexicographic over elements; a shorter prefix sorts first.
        (Value::List(x), Value::List(y)) => x
            .iter()
            .zip(y)
            .map(|(p, q)| cmp_total(p, q))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or_else(|| x.len().cmp(&y.len())),
        // Records: lexicographic over sorted (key, then value) pairs; a shorter
        // record sorts first. Deterministic total order for ORDER BY / grouping.
        (Value::Record(x), Value::Record(y)) => x
            .iter()
            .zip(y.iter())
            .map(|((k1, v1), (k2, v2))| k1.cmp(k2).then_with(|| cmp_total(v1, v2)))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or_else(|| x.len().cmp(&y.len())),
        // Maps: lexicographic over insertion-ordered (key, then value) pairs.
        (Value::Map(x), Value::Map(y)) => x
            .iter()
            .zip(y.iter())
            .map(|((k1, v1), (k2, v2))| cmp_total(k1, k2).then_with(|| cmp_total(v1, v2)))
            .find(|o| *o != Ordering::Equal)
            .unwrap_or_else(|| x.len().cmp(&y.len())),
        // Same-kind element refs order by dense id (a deterministic tiebreak for sort/dedup).
        (Value::Node(x), Value::Node(y)) | (Value::Edge(x), Value::Edge(y)) => x.cmp(y),
        _ => a.rank().cmp(&b.rank()),
    }
}

/// Total order over `f64`: normal order, but NaN is greatest and all NaNs tie,
/// and `-0.0` ties with `0.0`. This is the one place NaN ordering is decided.
#[must_use]
pub fn cmp_num_total(x: f64, y: f64) -> Ordering {
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        // `total_cmp` would split -0.0 < 0.0; we want them equal, so use partial
        // and unwrap — neither is NaN here.
        (false, false) => x.partial_cmp(&y).expect("neither operand is NaN"),
    }
}

#[cfg(test)]
mod tests;
