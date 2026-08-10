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
    Str(Arc<str>),
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
    Record(Arc<[(Arc<str>, Value)]>),
    /// A TinkerPop map (Gremlin): **any** value as a key, **insertion-ordered** —
    /// `valueMap`/`project`/`select` preserve the order the traversal produced. So
    /// equality and ordering are POSITIONAL (order-sensitive), unlike a `Record`.
    /// `Arc`-boxed for a cheap per-row clone.
    Map(Arc<Vec<(Value, Value)>>),
}

/// Build a [`Value::Record`] from `pairs`: duplicate keys collapse (last write
/// wins) and the fields are sorted by key, so two records with the same contents
/// are byte-identical and equality is a slice compare. The ONE place a record is
/// canonicalized.
#[must_use]
pub fn make_record(pairs: Vec<(Arc<str>, Value)>) -> Value {
    let mut out: Vec<(Arc<str>, Value)> = Vec::with_capacity(pairs.len());
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
pub fn record_field(fields: &[(Arc<str>, Value)], key: &str) -> Value {
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
        match self {
            Self::Bool(_) => 0,
            Self::Num(_) => 1,
            Self::Str(_) => 2,
            Self::Temporal(_) => 3,
            Self::List(_) => 4,
            Self::Record(_) => 5,
            Self::Map(_) => 6,
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

/// The target type of a `CAST`, mirrored from `ir::CastTarget` so the conversion
/// table lives beside the rest of the value contract. Kept in sync by the
/// `From<ir::CastTarget>` below — a new target forces an arm here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastTarget {
    Integer,
    Float,
    String,
    Boolean,
}

impl From<crate::ir::CastTarget> for CastTarget {
    fn from(t: crate::ir::CastTarget) -> Self {
        match t {
            crate::ir::CastTarget::Integer => Self::Integer,
            crate::ir::CastTarget::Float => Self::Float,
            crate::ir::CastTarget::String => Self::String,
            crate::ir::CastTarget::Boolean => Self::Boolean,
        }
    }
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
        CastTarget::String => Ok(Value::Str(Arc::from(
            match v {
                Value::Str(s) => return Ok(Value::Str(s.clone())),
                // Numbers render as they do on egress (`f64::to_string`); a non-finite
                // number has no textual form here.
                Value::Num(x) => {
                    if x.is_finite() {
                        x.to_string()
                    } else {
                        return bad("non-finite number", "string");
                    }
                }
                Value::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
                // A temporal casts to its ISO-8601 string.
                Value::Temporal(t) => t.format(),
                Value::List(_) => return bad("list", "string"),
                Value::Record(_) => return bad("record", "string"),
                Value::Map(_) => return bad("map", "string"),
                Value::Null => unreachable!("null handled above"),
            }
            .as_str(),
        ))),
        CastTarget::Boolean => match v {
            Value::Bool(b) => Ok(Value::Bool(*b)),
            // Numeric truthiness: zero is false, every other (incl. NaN) is true.
            Value::Num(x) => Ok(Value::Bool(*x != 0.0)),
            Value::Str(s) => match s.trim() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => bad(&format!("string {s:?}"), "boolean"),
            },
            Value::List(_) => bad("list", "boolean"),
            Value::Temporal(_) => bad("temporal", "boolean"),
            Value::Record(_) => bad("record", "boolean"),
            Value::Map(_) => bad("map", "boolean"),
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
        (Value::Temporal(x), Value::Temporal(y)) if x.kind() == y.kind() => Some(x.cmp_total(y)),
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
mod tests {
    use super::*;

    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }

    #[test]
    fn null_is_the_only_null() {
        assert!(Value::Null.is_null());
        assert!(!n(0.0).is_null());
        assert!(!Value::Bool(false).is_null());
    }

    #[test]
    fn only_true_keeps_a_row() {
        assert!(Value::Bool(true).is_true());
        assert!(!Value::Bool(false).is_true());
        assert!(!Value::Null.is_true());
        assert!(!n(1.0).is_true()); // a non-boolean is UNKNOWN, not true
    }

    #[test]
    fn equality_policy() {
        assert!(equals(&n(1.0), &n(1.0)));
        assert!(equals(&n(0.0), &n(-0.0))); // -0 == 0
        assert!(!equals(&n(f64::NAN), &n(f64::NAN))); // NaN != NaN
        assert!(!equals(&n(1.0), &s("1"))); // cross-type is false, not an error
        assert!(equals(&Value::Null, &Value::Null));
        assert!(equals(&s("a"), &s("a")));
    }

    #[test]
    fn as_num_is_a_finite_num_only() {
        // The arithmetic gate: only a FINITE `Num` yields a number; NaN/Inf and any
        // non-numeric value are `None`, so arithmetic on them produces NULL.
        assert_eq!(as_num(&n(1.5)), Some(1.5));
        assert_eq!(as_num(&n(0.0)), Some(0.0));
        assert_eq!(as_num(&n(-0.0)), Some(-0.0));
        assert_eq!(as_num(&n(f64::NAN)), None);
        assert_eq!(as_num(&n(f64::INFINITY)), None);
        assert_eq!(as_num(&n(f64::NEG_INFINITY)), None);
        assert_eq!(as_num(&s("1")), None);
        assert_eq!(as_num(&Value::Bool(true)), None);
        assert_eq!(as_num(&Value::Null), None);
    }

    #[test]
    fn cmp_partial_is_three_valued() {
        use std::cmp::Ordering;
        // Same-type scalars are comparable.
        assert_eq!(cmp_partial(&n(1.0), &n(2.0)), Some(Ordering::Less));
        assert_eq!(cmp_partial(&s("a"), &s("b")), Some(Ordering::Less));
        // Cross-type is UNKNOWN (None) — the operator yields NULL, NOT a rank Bool.
        assert_eq!(cmp_partial(&n(1.0), &s("a")), None);
        assert_eq!(cmp_partial(&Value::Bool(true), &n(1.0)), None);
        // A NaN operand is IEEE-unordered → UNKNOWN (even though `cmp_total` ranks
        // NaN greatest for sorting).
        assert_eq!(cmp_partial(&n(f64::NAN), &n(1.0)), None);
        assert_eq!(cmp_total(&n(f64::NAN), &n(1.0)), Ordering::Greater);
    }

    #[test]
    fn num_group_bits_canonicalize_nan_and_signed_zero() {
        // Every NaN — whatever the bit payload (quiet, signalling, sign) — groups
        // the same; -0.0 folds to +0.0; distinct finite numbers stay distinct.
        let quiet = f64::NAN;
        let signalling = f64::from_bits(0x7ff0_0000_0000_0001);
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
        assert!(quiet.is_nan() && signalling.is_nan() && neg_nan.is_nan());
        assert_eq!(num_group_bits(quiet), num_group_bits(signalling));
        assert_eq!(num_group_bits(quiet), num_group_bits(neg_nan));
        assert_eq!(num_group_bits(0.0), num_group_bits(-0.0));
        assert_ne!(num_group_bits(1.0), num_group_bits(2.0));
    }

    #[test]
    fn cross_type_ordering_is_total_never_a_throw() {
        // DIVERGENCE (recorded for J1): lenke-engine's `cmp_total` is a single
        // deterministic total order over ANY pair — cross-type orders by rank and
        // never faults — whereas lenke-core's GQL ordering raises E_INVALID_VALUE
        // on incompatible types. Chosen so sort/group/min-max are total here.
        assert_eq!(cmp_total(&n(1.0), &s("a")), Ordering::Less); // Num(1) < Str(2)
        assert_eq!(cmp_total(&Value::Bool(true), &n(0.0)), Ordering::Less); // Bool(0) < Num(1)
        assert_eq!(cmp_total(&s("z"), &Value::Null), Ordering::Less); // Str(2) < Null(last)
                                                                      // …and it is a strict total order: antisymmetric on the same pair.
        assert_eq!(cmp_total(&s("a"), &n(1.0)), Ordering::Greater);
    }

    #[test]
    fn group_key_is_the_inverse_of_equals_on_nan_and_zero() {
        // Grouping is the OPPOSITE of predicate equality on exactly these cases:
        // two NaNs share a group though NaN != NaN; -0 and 0 share a group.
        assert_eq!(group_key(&n(f64::NAN)), group_key(&n(f64::NAN)));
        assert_eq!(group_key(&n(0.0)), group_key(&n(-0.0)));
        // …and it still separates the ordinary cases and keeps types apart.
        assert_ne!(group_key(&n(1.0)), group_key(&n(2.0)));
        assert_ne!(group_key(&n(1.0)), group_key(&s("1"))); // no cross-type collision
        assert_ne!(group_key(&Value::Bool(true)), group_key(&n(1.0)));
        assert_eq!(group_key(&Value::Null), group_key(&Value::Null));
    }

    fn b(x: bool) -> Value {
        Value::Bool(x)
    }

    #[test]
    fn cast_null_is_null_for_every_target() {
        for t in [
            CastTarget::Integer,
            CastTarget::Float,
            CastTarget::String,
            CastTarget::Boolean,
        ] {
            assert!(cast(&Value::Null, t).unwrap().is_null());
        }
    }

    #[test]
    fn cast_to_integer_truncates_toward_zero() {
        // Positive and negative both truncate toward zero (not floor).
        assert!(equals(
            &cast(&n(3.9), CastTarget::Integer).unwrap(),
            &n(3.0)
        ));
        assert!(equals(
            &cast(&n(-3.9), CastTarget::Integer).unwrap(),
            &n(-3.0)
        ));
        // Via a string, and via a bool.
        assert!(equals(
            &cast(&s("  7.5 "), CastTarget::Integer).unwrap(),
            &n(7.0)
        ));
        assert!(equals(
            &cast(&b(true), CastTarget::Integer).unwrap(),
            &n(1.0)
        ));
        assert!(equals(
            &cast(&b(false), CastTarget::Integer).unwrap(),
            &n(0.0)
        ));
        // A non-finite number has no integer form → throw.
        assert!(cast(&n(f64::NAN), CastTarget::Integer).is_err());
        assert!(cast(&n(f64::INFINITY), CastTarget::Integer).is_err());
        // A non-numeric string → throw.
        assert!(cast(&s("nope"), CastTarget::Integer).is_err());
    }

    #[test]
    fn cast_to_float_parses_and_identities() {
        assert!(equals(&cast(&n(2.5), CastTarget::Float).unwrap(), &n(2.5)));
        assert!(equals(
            &cast(&s("2.5"), CastTarget::Float).unwrap(),
            &n(2.5)
        ));
        assert!(equals(&cast(&b(true), CastTarget::Float).unwrap(), &n(1.0)));
        assert!(cast(&s("x"), CastTarget::Float).is_err());
    }

    #[test]
    fn cast_to_string_renders_scalars() {
        assert!(equals(&cast(&n(3.0), CastTarget::String).unwrap(), &s("3")));
        assert!(equals(
            &cast(&b(true), CastTarget::String).unwrap(),
            &s("true")
        ));
        assert!(equals(
            &cast(&s("hi"), CastTarget::String).unwrap(),
            &s("hi")
        ));
        // Non-finite numbers and lists have no textual form here → throw.
        assert!(cast(&n(f64::NAN), CastTarget::String).is_err());
        assert!(cast(&Value::List(vec![n(1.0)]), CastTarget::String).is_err());
    }

    #[test]
    fn cast_to_boolean_uses_zero_and_true_false() {
        assert!(equals(
            &cast(&b(true), CastTarget::Boolean).unwrap(),
            &b(true)
        ));
        // Numeric truthiness: 0 is false, everything else true.
        assert!(equals(
            &cast(&n(0.0), CastTarget::Boolean).unwrap(),
            &b(false)
        ));
        assert!(equals(
            &cast(&n(-2.0), CastTarget::Boolean).unwrap(),
            &b(true)
        ));
        // Only the words true/false convert (trimmed); anything else throws.
        assert!(equals(
            &cast(&s(" true "), CastTarget::Boolean).unwrap(),
            &b(true)
        ));
        assert!(equals(
            &cast(&s("false"), CastTarget::Boolean).unwrap(),
            &b(false)
        ));
        assert!(cast(&s("yes"), CastTarget::Boolean).is_err());
    }

    #[test]
    fn temporal_equality_and_order_in_the_contract() {
        use crate::temporal::{Date, Temporal};
        let d1 = Value::Temporal(Temporal::Date(Date::parse("2024-01-01").unwrap()));
        let d1b = Value::Temporal(Temporal::Date(Date::parse("2024-01-01").unwrap()));
        let d2 = Value::Temporal(Temporal::Date(Date::parse("2024-06-01").unwrap()));
        // Same date is equal and groups together; different dates order chronologically.
        assert!(equals(&d1, &d1b));
        assert!(!equals(&d1, &d2));
        assert_eq!(group_key(&d1), group_key(&d1b));
        assert_ne!(group_key(&d1), group_key(&d2));
        assert_eq!(cmp_total(&d1, &d2), Ordering::Less);
        // A temporal never equals a string or a number (cross-type is false), and
        // sits between Str and List/Null in the total order.
        assert!(!equals(&d1, &s("2024-01-01")));
        assert!(!equals(&d1, &n(0.0)));
        assert_eq!(cmp_total(&s("z"), &d1), Ordering::Less); // Str(2) < Temporal(3)
        assert_eq!(cmp_total(&d1, &Value::Null), Ordering::Less); // Temporal < Null(last)
    }

    #[test]
    fn record_canonicalizes_and_compares_by_contents() {
        // make_record sorts keys and collapses duplicates (last write wins).
        let r1 = make_record(vec![(Arc::from("b"), n(2.0)), (Arc::from("a"), n(1.0))]);
        let Value::Record(f) = &r1 else {
            panic!("not a record")
        };
        assert_eq!(f[0].0.as_ref(), "a");
        assert_eq!(f[1].0.as_ref(), "b");
        let dedup = make_record(vec![(Arc::from("k"), n(1.0)), (Arc::from("k"), n(2.0))]);
        let Value::Record(g) = &dedup else {
            panic!("not a record")
        };
        assert_eq!(g.len(), 1);
        assert!(equals(&g[0].1, &n(2.0))); // last wins

        // Equality / grouping are independent of insertion order (both canonical).
        let r2 = make_record(vec![(Arc::from("a"), n(1.0)), (Arc::from("b"), n(2.0))]);
        assert!(equals(&r1, &r2));
        assert_eq!(group_key(&r1), group_key(&r2));
        // A different value → not equal, different group.
        let r3 = make_record(vec![(Arc::from("a"), n(9.0)), (Arc::from("b"), n(2.0))]);
        assert!(!equals(&r1, &r3));
        assert_ne!(group_key(&r1), group_key(&r3));

        // Field lookup, and cross-type rank (List < Record < Null).
        assert!(equals(&record_field(f, "a"), &n(1.0)));
        assert!(record_field(f, "missing").is_null());
        assert_eq!(cmp_total(&Value::List(vec![]), &r1), Ordering::Less);
        assert_eq!(cmp_total(&r1, &Value::Null), Ordering::Less);
    }

    #[test]
    fn map_is_positional_unlike_record() {
        // A Map's equality and grouping are ORDER-SENSITIVE (insertion order),
        // the opposite of a Record's sorted keys.
        let m1 = Value::Map(Arc::new(vec![(s("a"), n(1.0)), (s("b"), n(2.0))]));
        let m2 = Value::Map(Arc::new(vec![(s("a"), n(1.0)), (s("b"), n(2.0))]));
        let reordered = Value::Map(Arc::new(vec![(s("b"), n(2.0)), (s("a"), n(1.0))]));
        assert!(equals(&m1, &m2));
        assert_eq!(group_key(&m1), group_key(&m2));
        assert!(!equals(&m1, &reordered)); // order matters
        assert_ne!(group_key(&m1), group_key(&reordered));
        // Any-typed keys are permitted (unlike Record's string keys).
        let numkey = Value::Map(Arc::new(vec![(n(1.0), s("x"))]));
        assert!(equals(
            &numkey,
            &Value::Map(Arc::new(vec![(n(1.0), s("x"))]))
        ));
        // Cross-type rank: Record (5) < Map (6) < Null (7).
        let rec = make_record(vec![(Arc::from("a"), n(1.0))]);
        assert_eq!(cmp_total(&rec, &m1), Ordering::Less);
        assert_eq!(cmp_total(&m1, &Value::Null), Ordering::Less);
    }

    #[test]
    fn total_order_is_total_and_deterministic() {
        // NaN is the greatest number.
        assert_eq!(cmp_num_total(f64::NAN, 1e308), Ordering::Greater);
        assert_eq!(cmp_num_total(f64::NAN, f64::NAN), Ordering::Equal);
        assert_eq!(cmp_num_total(-0.0, 0.0), Ordering::Equal);
        assert_eq!(cmp_num_total(1.0, 2.0), Ordering::Less);

        // Nulls sort last, across types, without panicking.
        let mut xs = [Value::Null, n(2.0), s("z"), Value::Bool(true), n(1.0)];
        xs.sort_by(cmp_total);
        // rank order: Bool, Num, Num, Str, Null
        assert!(matches!(xs[0], Value::Bool(true)));
        assert!(matches!(xs[1], Value::Num(x) if x == 1.0));
        assert!(matches!(xs[2], Value::Num(x) if x == 2.0));
        assert!(matches!(xs[3], Value::Str(_)));
        assert!(matches!(xs[4], Value::Null));
    }
}
