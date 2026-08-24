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
    // lenke-engine's `cmp_total` is a single deterministic total order over ANY
    // pair — cross-type orders by rank and never faults — where the `<=`/`<`
    // OPERATOR raises E_INVALID_VALUE on incompatible types. The rank matches
    // core's `type_rank`: Num < Str < Bool < Temporal < compound < Null, so
    // ORDER BY / min / max over a mixed column agree byte-for-byte with core.
    assert_eq!(cmp_total(&n(1.0), &s("a")), Ordering::Less); // Num(0) < Str(1)
    assert_eq!(cmp_total(&Value::Bool(true), &n(0.0)), Ordering::Greater); // Bool(2) > Num(0)
    assert_eq!(cmp_total(&s("z"), &Value::Bool(true)), Ordering::Less); // Str(1) < Bool(2)
    assert_eq!(cmp_total(&s("z"), &Value::Null), Ordering::Less); // Str(1) < Null(last)
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
    // A non-finite NUMBER still has no textual form here → throw.
    assert!(cast(&n(f64::NAN), CastTarget::String).is_err());
    // A list / record now SERIALIZES (matching `to_string` and core's `js_str`),
    // not throws: a list joins its elements with "," and a record renders as JSON.
    assert!(equals(
        &cast(&Value::List(vec![n(1.0), n(2.0)]), CastTarget::String).unwrap(),
        &s("1,2")
    ));
}

#[test]
fn cast_nan_to_boolean_is_an_invalid_cast() {
    // A NaN is a live value in GQL (only nulled at JSON egress), so casting it to a
    // boolean faults at the cast rather than yielding a null; an `Inf` is nonzero → true.
    assert!(cast(&n(f64::NAN), CastTarget::Boolean).is_err());
    assert!(equals(
        &cast(&n(f64::INFINITY), CastTarget::Boolean).unwrap(),
        &b(true)
    ));
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
    // The SQL boolean spellings convert (case-insensitive, trimmed); an
    // unrecognized string throws.
    for t in [" true ", "yes", "Y", "ON", "1", "t"] {
        assert!(
            equals(&cast(&s(t), CastTarget::Boolean).unwrap(), &b(true)),
            "{t}"
        );
    }
    for f in ["false", "no", "N", "off", "0", "F"] {
        assert!(
            equals(&cast(&s(f), CastTarget::Boolean).unwrap(), &b(false)),
            "{f}"
        );
    }
    assert!(cast(&s("maybe"), CastTarget::Boolean).is_err());
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
    // rank order (matching core): Num, Num, Str, Bool, Null
    assert!(matches!(xs[0], Value::Num(x) if x == 1.0));
    assert!(matches!(xs[1], Value::Num(x) if x == 2.0));
    assert!(matches!(xs[2], Value::Str(_)));
    assert!(matches!(xs[3], Value::Bool(true)));
    assert!(matches!(xs[4], Value::Null));
}
