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

use std::cmp::Ordering;
use std::sync::Arc;

/// A runtime value. The minimal set the first execution slice needs; temporal,
/// list, and map/record join this enum as later slices land, and every addition
/// gets its arm in the functions below — which is the point of one contract.
#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(Arc<str>),
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
            // Null sorts LAST — it is the greatest in the total order.
            Self::Null => 3,
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
        _ => false,
    }
}

/// A deterministic total order over ANY pair of values — never panics. Used by
/// sort, min/max, and (once it lands) grouping tiebreaks. Nulls sort last; NaN
/// is the greatest number; `-0.0` and `0.0` tie. Distinct types order by rank so
/// a mixed column still has one stable order.
#[must_use]
pub fn cmp_total(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Num(x), Value::Num(y)) => cmp_num_total(*x, *y),
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
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
