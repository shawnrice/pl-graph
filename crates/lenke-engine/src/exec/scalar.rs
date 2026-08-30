use super::*;
use crate::gstr::GStr;
use crate::value::{self, Value};

/// Dispatch a scalar function over its already-evaluated argument row. Arity is
/// enforced by the parser, so indexing `args` here is safe. NULL / wrong-type
/// arguments yield NULL (no coercion, no throw).
/// The fallible wrapper around [`call_scalar`]: nearly every scalar function is
/// total, but a temporal component accessor of a kind that lacks that component
/// (`_year` of a time, `_hour` of a date) FAULTS with `E_INVALID_VALUE`. A non-null
/// NON-temporal argument (a number or string — never coerced) is ALSO a data
/// exception (matching the TS engine); only a NULL arg yields NULL (nullish propagation).
pub(super) fn call_scalar_checked(name: &str, args: &[Value]) -> Result<Value, String> {
    if matches!(
        name,
        "_year" | "_month" | "_day" | "_hour" | "_minute" | "_second"
    ) {
        return match &args[0] {
            Value::Temporal(t) => date_part(name.trim_start_matches('_'), *t)
                .map(|n| Value::Num(n as f64))
                .ok_or_else(|| {
                    format!(
                        "E_INVALID_VALUE: {} is undefined for this temporal kind",
                        name.trim_start_matches('_')
                    )
                }),
            Value::Null => Ok(Value::Null),
            _ => Err(format!(
                "E_INVALID_VALUE: {}() requires a temporal value (a string is not coerced)",
                name.trim_start_matches('_')
            )),
        };
    }
    // Numeric scalar functions take numbers only — a non-null, non-numeric argument
    // (a string, bool, temporal, list) is a data exception, never coerced (the same
    // SQL rule as arithmetic and the temporal accessors above). `sqrt('1e300')`
    // throws rather than silently returning NULL or coercing the string. A NULL arg
    // still propagates to NULL inside `call_scalar`. Unlike an arithmetic OPERATOR
    // (which propagates null before type-checking), a named function VALIDATES its
    // non-null arguments even beside a null: `atan2(null, duration)` is a type error.
    if matches!(
        name,
        "abs"
            | "sign"
            | "floor"
            | "ceil"
            | "ceiling"
            | "sqrt"
            | "exp"
            | "ln"
            | "log10"
            | "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "cot"
            | "degrees"
            | "radians"
            | "round"
            | "log"
            | "power"
            | "mod"
            | "atan2"
    ) && args
        .iter()
        .any(|a| !a.is_null() && !matches!(a, Value::Num(_)))
    {
        return Err(format!(
            "E_INVALID_VALUE: {name}() requires a number (a string is not coerced)"
        ));
    }
    // String / byte scalar functions take strings — a non-null, non-string argument is a
    // data exception, never coerced (the same rule as the numeric functions above; only a
    // NULL arg propagates to NULL). Mixed-arity fns type each position: `left`/`right`/
    // `substring` take a string then number(s); the rest are all-string. `reverse`/`size`/
    // `cardinality` are polymorphic over a string OR a list, so they fault only on neither.
    {
        let (str_pos, num_pos): (&[usize], &[usize]) = match name {
            "upper" | "lower" | "char_length" | "character_length" | "byte_length"
            | "octet_length" => (&[0], &[]),
            "trim" | "btrim" | "ltrim" | "rtrim" => (&[0, 1], &[]),
            "split" | "starts_with" | "ends_with" | "contains" | "regex_match" => (&[0, 1], &[]),
            "replace" => (&[0, 1, 2], &[]),
            "left" | "right" => (&[0], &[1]),
            "substring" => (&[0], &[1, 2]),
            _ => (&[], &[]),
        };
        for &i in str_pos {
            if let Some(a) = args.get(i) {
                if !a.is_null() && !matches!(a, Value::Str(_)) {
                    return Err(format!(
                        "E_INVALID_VALUE: {name}() requires a string (a number is not coerced)"
                    ));
                }
            }
        }
        for &i in num_pos {
            if let Some(a) = args.get(i) {
                if !a.is_null() && !matches!(a, Value::Num(_)) {
                    return Err(format!(
                        "E_INVALID_VALUE: {name}() requires a number (a string is not coerced)"
                    ));
                }
            }
        }
        if matches!(name, "reverse" | "size" | "cardinality")
            && args
                .first()
                .is_some_and(|a| !a.is_null() && !matches!(a, Value::Str(_) | Value::List(_)))
        {
            return Err(format!(
                "E_INVALID_VALUE: {name}() requires a string or list"
            ));
        }
        // `||` concatenation (lowered to `concat`): operands must be homogeneous and
        // concatenable — all strings OR all lists. A NULL operand makes the whole result
        // NULL (handled in the fold); a non-null operand that is neither a string nor a
        // list, or a string/list mix, is a data exception — never JS-string-coerced.
        if name == "concat" {
            let non_null: Vec<&Value> = args.iter().filter(|a| !a.is_null()).collect();
            let all_str = non_null.iter().all(|a| matches!(a, Value::Str(_)));
            let all_list = non_null.iter().all(|a| matches!(a, Value::List(_)));
            if !non_null.is_empty() && !all_str && !all_list {
                return Err(
                    "E_INVALID_VALUE: || requires all operands to be strings or all lists \
                     (values are not coerced)"
                        .into(),
                );
            }
        }
    }
    // `to_boolean(NaN)` / `CAST(NaN AS BOOLEAN)` is an invalid conversion — a NaN is a live
    // value in GQL (only nulled at JSON egress), so it faults at the conversion rather than
    // becoming a null that trips a type error at a later consumer. (`Inf` → true, nonzero.)
    if matches!(name, "to_boolean" | "toboolean")
        && matches!(args.first(), Some(Value::Num(x)) if x.is_nan())
    {
        return Err("E_INVALID_VALUE: cannot convert NaN to a boolean".into());
    }
    Ok(call_scalar(name, args))
}

/// Does the (non-null) value `v` match the IS TYPED scalar category `category`?
/// `integer` requires an integral finite number; `float` any number. Replicates
/// the now-removed lenke-core's `category_matches`/`value_is_typed_ty`.
fn scalar_is_typed(category: &str, v: &Value) -> bool {
    match category {
        "any" => true,
        "null" => false, // v is non-null here
        "bool" => matches!(v, Value::Bool(_)),
        "string" => matches!(v, Value::Str(_)),
        "integer" => matches!(v, Value::Num(n) if n.is_finite() && n.fract() == 0.0),
        "float" => matches!(v, Value::Num(_)),
        "list" => matches!(v, Value::List(_)),
        "record" => matches!(v, Value::Record(_)),
        "date" | "local_time" | "local_datetime" | "zoned_time" | "zoned_datetime" | "duration" => {
            use crate::temporal::TemporalKind as K;
            if let Value::Temporal(t) = v {
                let want = match category {
                    "date" => K::Date,
                    "local_time" => K::Time,
                    "local_datetime" => K::DateTime,
                    "zoned_time" => K::ZonedTime,
                    "zoned_datetime" => K::ZonedDateTime,
                    _ => K::Duration,
                };
                t.kind() == want
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Does the record value `v` conform to a CLOSED record type `schema` (a `Record`
/// of field → descriptor `List[category, not_null, nested_schema_or_null]`)? Closed:
/// every field of `v` must be declared; every declared field must be present-and-
/// typed or absent-and-nullable; a null field value is allowed only when nullable;
/// a `record`-category field recurses into its nested schema (or, for an open
/// `RECORD` with no schema, only checks the value is a record).
fn record_matches_schema(v: &Value, schema: &[(GStr, Value)]) -> bool {
    let Value::Record(fields) = v else {
        return false;
    };
    // No undeclared field.
    for (k, _) in fields.iter() {
        if !schema.iter().any(|(sk, _)| sk == k) {
            return false;
        }
    }
    for (sk, desc) in schema.iter() {
        let Value::List(d) = desc else {
            return false;
        };
        let category = match &d[0] {
            Value::Str(s) => s.as_ref(),
            _ => return false,
        };
        let field_not_null = matches!(d[1], Value::Bool(true));
        let nested = &d[2];
        match fields.iter().find(|(fk, _)| fk == sk) {
            None => {
                if field_not_null {
                    return false; // required field absent
                }
            }
            Some((_, fv)) => {
                if fv.is_null() {
                    if field_not_null {
                        return false;
                    }
                } else if category == "record" {
                    match nested {
                        Value::Record(sub) => {
                            if !record_matches_schema(fv, sub) {
                                return false;
                            }
                        }
                        // Open `RECORD` (no `{…}`): only require a record value.
                        _ => {
                            if !matches!(fv, Value::Record(_)) {
                                return false;
                            }
                        }
                    }
                } else if !scalar_is_typed(category, fv) {
                    return false;
                }
            }
        }
    }
    true
}

fn call_scalar(name: &str, args: &[Value]) -> Value {
    match name {
        // variadic
        "coalesce" => args
            .iter()
            .find(|v| !v.is_null())
            .cloned()
            .unwrap_or(Value::Null),
        // `x IS [NOT] TYPED <type> [NOT NULL]` desugars here: args are (value,
        // category, not_null). A NULL value conforms to any nullable type (so it is
        // `!not_null`); else the value's runtime type must match the category —
        // replicated from the now-removed lenke-core's `category_matches`/`value_is_typed_ty`.
        "__is_typed" => {
            let v = &args[0];
            let category = match &args[1] {
                Value::Str(s) => s.as_ref(),
                _ => return Value::Null,
            };
            let not_null = matches!(args[2], Value::Bool(true));
            if v.is_null() {
                return Value::Bool(!not_null);
            }
            Value::Bool(scalar_is_typed(category, v))
        }
        // `x IS [NOT] TYPED RECORD { f :: TYPE [NOT NULL], … }` — a CLOSED record type.
        // arg[1] encodes the schema as a `Record` mapping each field to a descriptor
        // `List[category, not_null, nested_schema_or_null]`. A closed record conforms
        // iff it carries NO undeclared field and every declared field is present-and-
        // typed or absent-and-nullable (recursively for nested records).
        "__is_typed_record" => {
            let v = &args[0];
            let not_null = matches!(args[2], Value::Bool(true));
            if v.is_null() {
                return Value::Bool(!not_null);
            }
            let Value::Record(schema) = &args[1] else {
                return Value::Null;
            };
            Value::Bool(record_matches_schema(v, schema))
        }
        // `a || b || …` — left-associative concat (the parser folds a `||` run into
        // one call). Matches the TS engine's `concat_step` fold: ANY null operand → NULL; two
        // lists concatenate element-wise; otherwise both sides JS-string-coerce (via
        // `to_string_fn`) and join.
        "concat" => {
            let mut acc = args.first().cloned().unwrap_or(Value::Null);
            for r in &args[1..] {
                acc = concat_step(&acc, r);
            }
            acc
        }
        // Non-finite CLASSIFIERS (leading-underscore extensions): TOTAL boolean
        // predicates — true iff the argument IS that kind of IEEE-754 value, false for
        // everything else (finite / null / string / any non-matching number). Never null,
        // never throw. GQL has no NaN/Infinity literal or `IS NAN` predicate, so these are
        // the way to test for the special values that are otherwise only visible via
        // comparisons/ordering. (Non-finite renders as null only at JSON egress.)
        "_is_nan" => Value::Bool(matches!(&args[0], Value::Num(x) if x.is_nan())),
        "_is_infinite" => Value::Bool(matches!(&args[0], Value::Num(x) if x.is_infinite())),
        "_is_finite" => Value::Bool(matches!(&args[0], Value::Num(x) if x.is_finite())),
        // numeric constants (0 args)
        "e" => Value::Num(std::f64::consts::E),
        "pi" => Value::Num(std::f64::consts::PI),
        // numeric (1 arg)
        "abs" | "sign" | "floor" | "ceil" | "ceiling" | "sqrt" | "exp" | "ln" | "log10" | "sin"
        | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "cot"
        | "degrees" | "radians" => scalar_num_fn(name, &args[0]),
        // `round(x)` rounds to an integer; `round(x, digits)` to `digits` decimal
        // places (negative rounds left of the point). Half away from zero, matching
        // the TS engine; the `(x*f).round()/f` form is bit-identical (do not reformulate).
        "round" => match value::num_of(&args[0]) {
            Some(x) => {
                let digits = args
                    .get(1)
                    .and_then(value::num_of)
                    .map_or(0, |d| d.trunc() as i32);
                let f = 10f64.powi(digits);
                Value::Num((x * f).round() / f)
            }
            None => Value::Null,
        },
        // numeric (2 args). `log(a, b)` is log-base-a of b = ln(b)/ln(a) (matches
        // the TS engine's argument order); `mod` is the fn form of `%` (NaN on a zero
        // divisor — it does NOT throw like the `%` OPERATOR, which the TS engine reserves for
        // the operator); `atan2(y, x)` is the two-argument arctangent. NaN/Inf
        // results are KEPT (K4), coerced only at JSON egress.
        "log" | "power" | "mod" | "atan2" => {
            match (value::num_of(&args[0]), value::num_of(&args[1])) {
                (Some(x), Some(y)) => Value::Num(match name {
                    "log" => y.ln() / x.ln(),
                    "power" => x.powf(y),
                    // atan2 is the one math fn whose result distinguishes the sign of a zero
                    // operand (`atan2(-0, -1) = -PI` vs `atan2(+0, -1) = +PI`). We treat -0
                    // and +0 as one value everywhere, so fold -0 to +0 on both inputs (the
                    // pure-TS engine does the same). Every other fn collapses -0 to 0 or null
                    // at egress already.
                    "atan2" => (x + 0.0).atan2(y + 0.0),
                    _ => x % y,
                }),
                _ => Value::Null,
            }
        }
        // nullif(a, b): NULL when a == b (value-contract equality), else a.
        "nullif" => {
            if !args[0].is_null() && !args[1].is_null() && value::equals(&args[0], &args[1]) {
                Value::Null
            } else {
                args[0].clone()
            }
        }
        // Cast FUNCTIONS: NULL on a failed/inapplicable conversion (unlike `CAST`,
        // which throws — and unlike `CAST`, these do NOT coerce a Bool to a number).
        "to_integer" | "tointeger" => to_number(&args[0], true),
        "to_float" | "tofloat" => to_number(&args[0], false),
        "to_string" | "tostring" => to_string_fn(&args[0]),
        "to_boolean" | "toboolean" => to_boolean_fn(&args[0]),
        // `to_list`: a list → itself; a string → its UTF-16 code-unit chars (the JS
        // `split('')` model, kept for byte-identity); a non-nullish scalar → a
        // singleton list; null / non-finite number → null. Matches the TS engine's ToList.
        "to_list" | "tolist" => match &args[0] {
            Value::List(_) => args[0].clone(),
            Value::Str(s) => Value::List(
                s.encode_utf16()
                    .map(|u| Value::Str(GStr::from(String::from_utf16_lossy(&[u]).as_str())))
                    .collect(),
            ),
            Value::Num(n) if !n.is_finite() => Value::Null,
            Value::Null => Value::Null,
            other => Value::List(vec![other.clone()]),
        },
        // string (1 arg → string/number)
        "upper" => str_map(&args[0], str::to_uppercase),
        "lower" => str_map(&args[0], str::to_lowercase),
        // `trim` is both-sides; a 2nd (char-set) arg from the SQL-spec form is
        // honored by routing through btrim (identical to the TS engine's Trim).
        "trim" => trim_fn("btrim", args),
        // ltrim/rtrim/btrim: 1 arg trims WHITESPACE from that side; a 2nd string
        // arg is the set of characters to strip instead.
        "ltrim" | "rtrim" | "btrim" => trim_fn(name, args),
        // reverse is polymorphic: a string reverses by char, a list by element;
        // anything else is NULL (matches the TS engine, e.g. reverse(number) → NULL).
        // reverse: a string reverses by UTF-16 unit (JS model — a surrogate pair
        // reversed decodes lossily to U+FFFD, byte-identical to the TS engine), a list by
        // element; anything else is NULL.
        "reverse" => match &args[0] {
            Value::Str(s) => {
                let mut units: Vec<u16> = s.encode_utf16().collect();
                units.reverse();
                Value::Str(String::from_utf16_lossy(&units).into())
            }
            Value::List(v) => Value::List(v.iter().rev().cloned().collect()),
            _ => Value::Null,
        },
        // left/right(s, n): the first / last n UTF-16 units (n ≥ len → the whole
        // string; n ≤ 0 → empty).
        "left" | "right" => match (&args[0], value::num_of(&args[1])) {
            (Value::Str(s), Some(k)) => {
                let units = utf16_len(s);
                let take = (k.max(0.0) as usize).min(units);
                let out = if name == "left" {
                    utf16_slice(s, 0, take)
                } else {
                    utf16_slice(s, units - take, take)
                };
                Value::Str(out.into())
            }
            _ => Value::Null,
        },
        // split(s, delim) → a list of substrings. An EMPTY delimiter splits into one
        // element per UTF-16 unit (JS model), matching the TS engine — NOT Rust's `split("")`.
        "split" => match (&args[0], &args[1]) {
            (Value::Str(s), Value::Str(d)) => {
                let parts: Vec<Value> = if d.is_empty() {
                    s.encode_utf16()
                        .map(|u| Value::Str(String::from_utf16_lossy(&[u]).into()))
                        .collect()
                } else {
                    s.split(d.as_ref()).map(|p| Value::Str(p.into())).collect()
                };
                Value::List(parts)
            }
            _ => Value::Null,
        },
        // Length of a string in UTF-16 code units (JS `.length` model), matching
        // the TS engine; `byte_length`/`octet_length` count UTF-8 bytes. (`length` is NOT an ISO
        // GQL function — the standard has CHAR_LENGTH/OCTET_LENGTH/PATH_LENGTH/CARDINALITY
        // — so it is rejected as unknown, not aliased here.)
        "char_length" | "character_length" => match &args[0] {
            Value::Str(s) => Value::Num(utf16_len(s) as f64),
            _ => Value::Null,
        },
        "byte_length" | "octet_length" => match &args[0] {
            Value::Str(s) => Value::Num(s.len() as f64),
            _ => Value::Null,
        },
        // string predicates (2 args → bool)
        "starts_with" => str_bool(&args[0], &args[1], |s, sub| s.starts_with(sub)),
        "ends_with" => str_bool(&args[0], &args[1], |s, sub| s.ends_with(sub)),
        "contains" => str_bool(&args[0], &args[1], |s, sub| s.contains(sub)),
        "regex_match" => regex_match(&args[0], &args[1]),
        // replace(s, from[, to]) — `to` defaults to "" (the TS engine); an EMPTY search
        // returns the string unchanged (the TS engine), NOT Rust's insert-everywhere.
        "replace" => match (&args[0], &args[1]) {
            (Value::Str(s), Value::Str(f)) => {
                let t = match args.get(2) {
                    Some(Value::Str(t)) => t.to_string(),
                    Some(v) if !v.is_null() => return Value::Null,
                    _ => String::new(),
                };
                if f.is_empty() {
                    Value::Str(s.clone())
                } else {
                    Value::Str(s.replace(f.as_ref(), &t).into())
                }
            }
            _ => Value::Null,
        },
        // substring(s, start[, len]) — ISO 1-based, UTF-16-unit indexed
        "substring" => substring(args),
        // `size` (and its ISO/SQL alias `cardinality`) is polymorphic over a collection OR
        // a string (UTF-16 units), like the TS engine; a non-collection non-string is NULL.
        "size" | "cardinality" => match &args[0] {
            Value::List(v) => Value::Num(v.len() as f64),
            Value::Str(s) => Value::Num(utf16_len(s) as f64),
            _ => Value::Null,
        },
        "head" => match &args[0] {
            Value::List(v) => v.first().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // tail: all but the first element (empty list → empty).
        "tail" => match &args[0] {
            Value::List(v) => Value::List(v.iter().skip(1).cloned().collect()),
            _ => Value::Null,
        },
        // append(list, x) → the list with x appended.
        "append" => match &args[0] {
            Value::List(v) => {
                let mut out = v.clone();
                out.push(args[1].clone());
                Value::List(out)
            }
            _ => Value::Null,
        },
        // list_contains(list, x) → 1.0 if any element equals x, else 0.0 (a NUMBER,
        // not a bool — matching the TS engine; `null` matches `null` via `equals`).
        "list_contains" => match &args[0] {
            Value::List(v) => Value::Num(f64::from(v.iter().any(|e| value::equals(e, &args[1])))),
            _ => Value::Null,
        },
        // list_sort(list, [order], [nullOrder]) — the value contract's total order,
        // reversed for `'desc'`, with absolute null placement (`'first'`/`'last'`,
        // default last). Mirrors ORDER BY / the TS engine's compare_sort byte-for-byte. A
        // stored list never holds NaN (it becomes null at ingest), so `is_null`
        // covers every nullish element.
        "list_sort" => {
            match &args[0] {
                Value::List(v) => {
                    let descending = matches!(args.get(1), Some(Value::Str(s)) if s.eq_ignore_ascii_case("desc"));
                    let nulls_first = matches!(args.get(2), Some(Value::Str(s)) if s.eq_ignore_ascii_case("first"));
                    let mut out = v.clone();
                    out.sort_by(|x, y| {
                        use std::cmp::Ordering;
                        match (x.is_null(), y.is_null()) {
                            (true, true) => Ordering::Equal,
                            (true, false) => {
                                if nulls_first {
                                    Ordering::Less
                                } else {
                                    Ordering::Greater
                                }
                            }
                            (false, true) => {
                                if nulls_first {
                                    Ordering::Greater
                                } else {
                                    Ordering::Less
                                }
                            }
                            (false, false) => {
                                let o = value::cmp_total(x, y);
                                if descending {
                                    o.reverse()
                                } else {
                                    o
                                }
                            }
                        }
                    });
                    Value::List(out)
                }
                _ => Value::Null,
            }
        }
        // Set algebra over lists — all DEDUPED (by value equality), matching the TS engine.
        // union: a's elements then b's, deduped. intersection: elements of a also
        // in b, deduped. difference: elements of a not in b, deduped.
        "list_union" | "difference" | "intersection" => match (&args[0], &args[1]) {
            (Value::List(a), Value::List(b)) => Value::List(list_set_op(name, a, b)),
            _ => Value::Null,
        },
        // range(start, end[, step]) — INCLUSIVE of both ends; default step 1; a
        // zero step is NULL; a start past end with the wrong sign yields an empty
        // list (matches the TS engine).
        "range" => {
            let step = if args.len() == 3 {
                value::as_num(&args[2]).map(f64::trunc)
            } else {
                Some(1.0)
            };
            match (
                value::as_num(&args[0]).map(f64::trunc),
                value::as_num(&args[1]).map(f64::trunc),
                step,
            ) {
                (Some(a), Some(b), Some(st)) if st != 0.0 => {
                    // COUNT-driven, not comparison-driven: `cur += st` stops advancing
                    // once `cur` reaches 2^53 (a no-op in f64), so a `while cur <= b`
                    // loop never terminates even when the count is tiny — e.g.
                    // range(9007199254740992, 9007199254740994) has just 3 elements.
                    // Compute the count up front (matching the TS engine), and cap the
                    // allocation. The emitted values still come from repeated addition.
                    let count = ((b - a) / st).floor() + 1.0;
                    if count.is_nan() || count <= 0.0 {
                        Value::List(Vec::new())
                    } else {
                        let n = if count > 10_000_001.0 {
                            10_000_001
                        } else {
                            count as usize
                        };
                        let mut out = Vec::with_capacity(n);
                        let mut cur = a;
                        for _ in 0..n {
                            out.push(Value::Num(cur));
                            cur += st;
                        }
                        Value::List(out)
                    }
                }
                _ => Value::Null,
            }
        }
        "last" => match &args[0] {
            Value::List(v) => v.last().cloned().unwrap_or(Value::Null),
            _ => Value::Null,
        },
        // Temporal component accessors (1 arg → number, or NULL when the component
        // is undefined for that kind). Core spells these with the leading-underscore
        // extension sigil (`_year`); the bare ISO name is not in the grammar.
        "_year" | "_month" | "_day" | "_hour" | "_minute" | "_second" => match &args[0] {
            Value::Temporal(t) => date_part(name.trim_start_matches('_'), *t)
                .map_or(Value::Null, |n| Value::Num(n as f64)),
            _ => Value::Null,
        },
        // Temporal constructors: parse a string, or coerce between kinds.
        "date" => temporal_ctor(&args[0], "date"),
        "local_time" => temporal_ctor(&args[0], "localtime"),
        "datetime" | "local_datetime" => temporal_ctor(&args[0], "datetime"),
        "zoned_time" => temporal_ctor(&args[0], "zoned_time"),
        "zoned_datetime" => temporal_ctor(&args[0], "zoned_datetime"),
        "duration" => temporal_ctor(&args[0], "duration"),
        // The exact span from a to b (b - a), in fixed units; cross-kind → NULL.
        "duration_between" => match (&args[0], &args[1]) {
            (Value::Temporal(x), Value::Temporal(y)) => duration_between(*x, *y),
            _ => Value::Null,
        },
        // Path accessors (nodes/relationships/path_length/elements) are not scalar
        // Call functions — they read the lineage sidecar via `Expr::PathAccess`.
        _ => Value::Null, // parser rejects unknown names; defensive
    }
}

/// Extract a calendar/clock component from a temporal value. `None` when the
/// component is undefined for that kind (`year`/`month`/`day` of a time-only
/// value, or `hour`/`minute`/`second` of a date). Zoned values decompose in their
/// stored offset (the local wall clock), as they render; euclidean division so
/// pre-epoch instants floor correctly. Ported from the now-removed lenke-core for agreement.
fn date_part(func: &str, t: crate::temporal::Temporal) -> Option<i64> {
    use crate::temporal::{civil_from_days, Temporal};
    const SPD: i64 = 86_400;
    match func {
        "year" | "month" | "day" => {
            let days = match t {
                Temporal::Date(x) => i64::from(x.days),
                Temporal::DateTime(x) => x.secs.div_euclid(SPD),
                Temporal::ZonedDateTime(x) => (x.secs + i64::from(x.offset) * 60).div_euclid(SPD),
                _ => return None,
            };
            let (y, m, d) = civil_from_days(days);
            Some(match func {
                "year" => y,
                "month" => i64::from(m),
                _ => i64::from(d),
            })
        }
        "hour" | "minute" | "second" => {
            let tod = match t {
                Temporal::Time(x) => i64::from(x.secs),
                Temporal::DateTime(x) => x.secs.rem_euclid(SPD),
                Temporal::ZonedTime(x) => {
                    (i64::from(x.secs) + i64::from(x.offset) * 60).rem_euclid(SPD)
                }
                Temporal::ZonedDateTime(x) => (x.secs + i64::from(x.offset) * 60).rem_euclid(SPD),
                _ => return None,
            };
            Some(match func {
                "hour" => tod / 3600,
                "minute" => (tod / 60) % 60,
                _ => tod % 60,
            })
        }
        _ => None,
    }
}

/// Temporal constructor: build a temporal of `kind` from a string (parsed) or
/// coerce another temporal into it (`date(datetime)` → the date part,
/// `datetime(date)` → midnight, `local_time(datetime)` → the time-of-day). A
/// bare `YYYY-MM-DD` string to a datetime target coerces to midnight. Anything
/// with no sensible conversion → NULL. Ported from the now-removed lenke-core for agreement.
pub(crate) fn temporal_ctor(v: &Value, kind: &str) -> Value {
    use crate::temporal::{Date, DateTime, Temporal, Time};
    const SPD: i64 = 86_400;
    match v {
        // A date-only string to a datetime target → midnight.
        Value::Str(s) if kind == "datetime" && !s.contains(['T', ' ']) => Date::parse(s)
            .map(|d| {
                Value::Temporal(Temporal::DateTime(DateTime {
                    secs: i64::from(d.days) * SPD,
                    nanos: 0,
                }))
            })
            .unwrap_or(Value::Null),
        Value::Str(s) => Temporal::parse(kind, s)
            .map(Value::Temporal)
            .unwrap_or(Value::Null),
        Value::Temporal(t) => match (kind, t) {
            ("date", Temporal::Date(_))
            | ("localtime", Temporal::Time(_))
            | ("datetime", Temporal::DateTime(_))
            | ("duration", Temporal::Duration(_)) => Value::Temporal(*t),
            ("date", Temporal::DateTime(dt)) => Value::Temporal(Temporal::Date(Date {
                days: dt.secs.div_euclid(SPD) as i32,
            })),
            ("localtime", Temporal::DateTime(dt)) => Value::Temporal(Temporal::Time(Time {
                secs: u32::try_from(dt.secs.rem_euclid(SPD)).expect("0..86_400"),
                nanos: dt.nanos,
            })),
            ("datetime", Temporal::Date(d)) => Value::Temporal(Temporal::DateTime(DateTime {
                secs: i64::from(d.days) * SPD,
                nanos: 0,
            })),
            _ => Value::Null, // e.g. duration(date) — no sensible conversion
        },
        _ => Value::Null,
    }
}

/// The EXACT span from `a` to `b` (b − a), in fixed units only: whole days for
/// two dates, seconds+nanos for two datetimes. Any cross-kind pair (or a
/// duration operand) → NULL. Ported from the now-removed lenke-core.
fn duration_between(a: crate::temporal::Temporal, b: crate::temporal::Temporal) -> Value {
    use crate::temporal::{Duration, Temporal};
    match (a, b) {
        (Temporal::Date(x), Temporal::Date(y)) => Value::Temporal(Temporal::Duration(Duration {
            months: 0,
            days: i64::from(y.days) - i64::from(x.days),
            secs: 0,
            nanos: 0,
        })),
        (Temporal::DateTime(x), Temporal::DateTime(y)) => {
            let mut secs = y.secs - x.secs;
            let mut nanos = i64::from(y.nanos) - i64::from(x.nanos);
            if nanos < 0 {
                nanos += 1_000_000_000;
                secs -= 1;
            }
            Value::Temporal(Temporal::Duration(Duration {
                months: 0,
                days: 0,
                secs,
                nanos: u32::try_from(nanos).expect("0..1e9 after carry"),
            }))
        }
        _ => Value::Null,
    }
}

/// Temporal `+`/`-`/`*` when either operand is temporal: instant ± duration
/// (anchored — months clamped, then days, then time), instant − instant (the
/// exact span), duration ± duration (component-wise), duration × integer. An
/// undefined combination is `Ok(Null)`; a result outside the representable
/// range is a THROWN fault (`Err`) — not a silent null. Ported from the now-removed lenke-core.
pub(super) fn temporal_arith(
    op: crate::ir::ArithOp,
    lv: &Value,
    rv: &Value,
) -> Result<Value, String> {
    use crate::ir::ArithOp::{Add, Mul, Sub};
    use crate::temporal::Temporal as T;
    use Value::Temporal as VT;
    let dur = |r: Option<crate::temporal::Duration>| {
        r.map(|d| VT(T::Duration(d)))
            .ok_or_else(|| "E_INVALID_VALUE: duration component out of range".to_string())
    };
    let inst = |r: Option<T>| {
        r.map(VT)
            .ok_or_else(|| "E_INVALID_VALUE: temporal result out of range".to_string())
    };
    match (op, lv, rv) {
        // duration ± duration (component-wise).
        (Add, VT(T::Duration(a)), VT(T::Duration(b))) => dur(a.add(b)),
        (Sub, VT(T::Duration(a)), VT(T::Duration(b))) => dur(a.add(&b.negate())),
        // instant ± duration (either order for +; dur±dur already handled above).
        (Add, VT(t), VT(T::Duration(d))) | (Add, VT(T::Duration(d)), VT(t)) => {
            inst(t.add_duration(d))
        }
        (Sub, VT(t), VT(T::Duration(d))) => inst(t.add_duration(&d.negate())),
        // instant − instant → the exact span from b to a (a − b).
        (Sub, VT(a), VT(b)) => Ok(duration_between(*b, *a)),
        // duration × INTEGER (either order); a non-integer factor is NULL.
        (Mul, VT(T::Duration(d)), Value::Num(n)) | (Mul, Value::Num(n), VT(T::Duration(d))) => {
            if n.is_finite() && n.fract() == 0.0 {
                dur(d.scale(*n as i64))
            } else {
                Ok(Value::Null)
            }
        }
        _ => Ok(Value::Null),
    }
}

/// Map a string value through `f`; NULL/non-string yields NULL.
fn str_map(v: &Value, f: impl Fn(&str) -> String) -> Value {
    match v {
        Value::Str(s) => Value::Str(f(s).into()),
        _ => Value::Null,
    }
}

/// A two-string predicate; NULL/non-string operand yields NULL.
/// Apply a comparison as a plain bool (UNKNOWN → false). For the shortestPath
/// target filter, where a non-matching/incomparable destination is simply excluded.
pub(super) fn cmp_apply(op: CompareOp, a: &Value, b: &Value) -> bool {
    match op {
        CompareOp::Eq => value::equals(a, b),
        CompareOp::Ne => !value::equals(a, b),
        CompareOp::Lt => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_lt),
        CompareOp::Le => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_le),
        CompareOp::Gt => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_gt),
        CompareOp::Ge => value::cmp_partial(a, b).is_some_and(std::cmp::Ordering::is_ge),
    }
}

fn str_bool(a: &Value, b: &Value, f: impl Fn(&str, &str) -> bool) -> Value {
    match (a, b) {
        (Value::Str(s), Value::Str(sub)) => Value::Bool(f(s, sub)),
        _ => Value::Null,
    }
}

thread_local! {
    /// Compiled-regex cache for the Gremlin `regex()` predicate, bounded like the TS engine's.
    static REGEX_CACHE: std::cell::RefCell<std::collections::HashMap<String, regex::Regex>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// `regex_match(value, pattern)` → Gremlin `regex(pattern)`: true when `value` is a
/// string the pattern finds a match in. A non-string is false; an invalid pattern
/// (already rejected at parse time) is false. Byte-identical to the TS engine's `regex_is_match`
/// — same `regex` crate, same bounded thread-local cache.
fn regex_match(v: &Value, pat: &Value) -> Value {
    let (Value::Str(s), Value::Str(p)) = (v, pat) else {
        return Value::Bool(false);
    };
    REGEX_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if !cache.contains_key(p.as_ref()) {
            if cache.len() >= 1000 {
                cache.clear();
            }
            match regex::Regex::new(p) {
                Ok(re) => {
                    cache.insert(p.to_string(), re);
                }
                Err(_) => return Value::Bool(false),
            }
        }
        Value::Bool(cache.get(p.as_ref()).is_some_and(|re| re.is_match(s)))
    })
}

/// Slice `s` by UTF-16 code UNITS `[start, start+len)` (JS `String.slice` /
/// `.length` model), decoding back to UTF-8. A slice that splits a surrogate pair
/// yields U+FFFD there (lossy) — byte-identical to the TS engine
/// (`utf16_slice`). The whole string model here counts UTF-16 units, NOT `chars()`,
/// so `size('😀')` is 2 (a surrogate pair), matching the TS engine.
fn utf16_slice(s: &str, start: usize, len: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    let end = start.saturating_add(len).min(units.len());
    let start = start.min(end);
    String::from_utf16_lossy(&units[start..end])
}

/// Length of `s` in UTF-16 code units — the JS `.length` model the TS engine uses.
fn utf16_len(s: &str) -> usize {
    s.encode_utf16().count()
}

/// `substring(s, start[, len])` — ISO/SQL **1-based** start, indexed by UTF-16 code
/// UNIT (matching the TS engine exactly). A `start <= 0` shrinks the window from the
/// front (SQL semantics); an omitted `len` runs to the end. NULL for a null string
/// or start.
fn substring(args: &[Value]) -> Value {
    if args[0].is_null() || args[1].is_null() {
        return Value::Null;
    }
    let Value::Str(s) = &args[0] else {
        return Value::Null;
    };
    // 1-based → 0-based offset; a start <= 0 shrinks the window from the front.
    let zero_start = value::num_of(&args[1]).unwrap_or(0.0) - 1.0;
    let from = zero_start.max(0.0) as usize;
    let count = match args.get(2) {
        Some(z) if !z.is_null() => {
            let end = (zero_start + value::num_of(z).unwrap_or(0.0)).max(0.0) as usize;
            end.saturating_sub(from)
        }
        _ => usize::MAX,
    };
    Value::Str(utf16_slice(s, from, count).into())
}

/// Apply a unary numeric scalar function. A NULL / non-numeric argument yields
/// NULL; a computed NaN/Inf result (e.g. `sqrt(-1)`, `ln(0)`) is KEPT (IEEE, like
/// the TS engine — coerced to null only at JSON egress). `sign(0)` is 0 and `sign(NaN)`
/// is NaN (unlike `f64::signum`, which is ±1 for both); rounding is f64's
/// round-half-away-from-zero.
/// The finite→finite unary numeric functions, as raw `f64 -> f64` closures that
/// match [`scalar_num_fn`] EXACTLY. Restricted to functions that cannot introduce
/// NaN/Inf from a finite input (`sqrt`/`ln`/`exp`/… can, so they are excluded):
/// the result column then keeps the all-finite invariant of a stored `Num` column,
/// and the vectorized path is byte-identical to the boxed one. `None` = not eligible.
pub(super) fn unary_finite_num_fn(name: &str) -> Option<fn(f64) -> f64> {
    Some(match name {
        "abs" => f64::abs,
        "floor" => f64::floor,
        "ceil" | "ceiling" => f64::ceil,
        "round" => f64::round,
        "sign" => |x: f64| {
            if x.is_nan() {
                f64::NAN
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
        _ => return None,
    })
}

fn scalar_num_fn(name: &str, v: &Value) -> Value {
    let Some(x) = value::num_of(v) else {
        return Value::Null;
    };
    let r = match name {
        "abs" => x.abs(),
        // NaN is not a number, so it has no sign — `sign(NaN)` stays NaN (→ null at
        // egress), matching JS `Math.sign(NaN)`. Without this guard NaN falls through both
        // `> 0` and `< 0` (both false for NaN) to the `0.0` else-arm, a wrong answer.
        "sign" => {
            if x.is_nan() {
                f64::NAN
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "floor" => x.floor(),
        "ceil" | "ceiling" => x.ceil(),
        "round" => x.round(),
        "sqrt" => x.sqrt(),
        // Transcendentals — native libm, matching the TS engine's native build. A
        // domain-invalid result (e.g. `ln(-1)`, `cot(0)`) is NaN/Inf and, for now,
        // falls to NULL through the finite gate below (K4 will KEEP it, like the TS engine).
        "exp" => x.exp(),
        "ln" => x.ln(),
        "log10" => x.log10(),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "cot" => 1.0 / x.tan(),
        // Multiply-then-divide, NOT `to_degrees`/`to_radians`: the latter pre-round
        // the 180/PI (resp. PI/180) constant and land one ULP off the TS engine's byte-exact
        // `(n*180)/PI` / `(n*PI)/180`.
        "degrees" => (x * 180.0) / std::f64::consts::PI,
        "radians" => (x * std::f64::consts::PI) / 180.0,
        _ => return Value::Null, // parser rejects unknown names; defensive
    };
    // NaN/Inf are KEPT (K4) — a computed NaN (`sqrt(-1)`, `ln(-1)`) is a real
    // signal, coerced to null only at the JSON egress boundary, matching the TS engine.
    Value::Num(r)
}

/// `to_integer`/`to_float` FUNCTION and the `CAST(x AS INTEGER|FLOAT)` it backs: a Num
/// (truncated for integer), a BOOLEAN (`true`→1, `false`→0 — the ISO-GQL/Ultipa explicit
/// conversion), or a parseable finite string. A list/record/temporal is NULL. (These are
/// EXPLICIT conversions, so a bool converts; the implicit paths — arithmetic, `sum`, … —
/// still never coerce a bool.)
fn to_number(v: &Value, integer: bool) -> Value {
    let n = match v {
        Value::Num(x) => *x,
        Value::Bool(b) => f64::from(u8::from(*b)),
        // A string that parses to a NON-finite value (`'1e1000'` → inf, `'nan'`) is
        // NULL — the fn form never yields inf/NaN, matching the TS engine's `.filter(is_finite)`.
        Value::Str(s) => match s.trim().parse::<f64>() {
            Ok(x) if x.is_finite() => x,
            _ => return Value::Null,
        },
        _ => return Value::Null,
    };
    if integer {
        if n.is_finite() {
            Value::Num(n.trunc())
        } else {
            Value::Null
        }
    } else {
        Value::Num(n)
    }
}

/// `to_string` FUNCTION: NULL→NULL, finite Num→its egress text, Bool→"true"/
/// "false", Str→itself, Temporal→its ISO form; a non-finite number is NULL.
/// One step of the `||` fold, matching the TS engine's `concat_step`: null propagates, two
/// lists concatenate, otherwise both operands JS-string-coerce and join.
fn concat_step(l: &Value, r: &Value) -> Value {
    if l.is_null() || r.is_null() {
        return Value::Null;
    }
    if let (Value::List(a), Value::List(b)) = (l, r) {
        return Value::List(a.iter().chain(b.iter()).cloned().collect());
    }
    match (to_string_fn(l), to_string_fn(r)) {
        (Value::Str(a), Value::Str(b)) => Value::Str(format!("{a}{b}").into()),
        // A non-stringable operand (e.g. a map) → NULL, as the TS engine's js_str-of-unknown does.
        _ => Value::Null,
    }
}

fn to_string_fn(v: &Value) -> Value {
    match v {
        Value::Null => Value::Null,
        Value::Str(s) => Value::Str(s.clone()),
        Value::Bool(b) => Value::Str((if *b { "true" } else { "false" }).into()),
        // Finite number as text, formatted like JS `Number.toString` (`-0` → "0",
        // exponential past the 1e21 / 1e-6 thresholds) — NOT Rust's `{}` (which is decimal
        // at all magnitudes and would give a different STRING for e.g. 1e-7).
        Value::Num(x) if x.is_finite() => Value::Str(crate::json::js_number(*x).into()),
        Value::Temporal(t) => Value::Str(t.format().into()),
        // A LIST joins its elements' string form (a null element → "", like JS
        // `Array.join`); a RECORD/MAP renders as its canonical JSON — matching the TS engine's
        // `js_str`, which serializes composites rather than returning NULL.
        Value::List(_) | Value::Record(_) | Value::Map(_) => {
            Value::Str(crate::json::js_str(v).into())
        }
        _ => Value::Null,
    }
}

/// `to_boolean` FUNCTION: a Bool passes through; the strings "true"/"false"
/// (trimmed, case-insensitive) convert; anything else is NULL.
fn to_boolean_fn(v: &Value) -> Value {
    match v {
        Value::Bool(b) => Value::Bool(*b),
        // A number coerces like C truthiness: nonzero → true, zero → false.
        Value::Num(x) => Value::Bool(*x != 0.0),
        Value::Str(s) => {
            let t = s.trim();
            if t.eq_ignore_ascii_case("true") {
                Value::Bool(true)
            } else if t.eq_ignore_ascii_case("false") {
                Value::Bool(false)
            } else {
                Value::Null
            }
        }
        _ => Value::Null,
    }
}

/// Set algebra over two lists, all producing a DEDUPED result (by the value
/// contract's `equals`, so `null` collapses with `null`): `list_union` = a then
/// the b-elements not already present; `intersection` = a-elements also in b;
/// `difference` = a-elements not in b. Order follows first appearance in `a`
/// (then `b` for union). O(n·m) — lists are small.
fn list_set_op(name: &str, a: &[Value], b: &[Value]) -> Vec<Value> {
    let contains = |xs: &[Value], v: &Value| xs.iter().any(|x| value::equals(x, v));
    let mut out: Vec<Value> = Vec::new();
    let push_unique = |out: &mut Vec<Value>, v: &Value| {
        if !contains(out, v) {
            out.push(v.clone());
        }
    };
    match name {
        "intersection" => {
            for v in a {
                if contains(b, v) {
                    push_unique(&mut out, v);
                }
            }
        }
        "difference" => {
            for v in a {
                if !contains(b, v) {
                    push_unique(&mut out, v);
                }
            }
        }
        _ => {
            // union: everything in a, then b's new elements, deduped throughout.
            for v in a.iter().chain(b.iter()) {
                push_unique(&mut out, v);
            }
        }
    }
    out
}

/// `ltrim`/`rtrim`/`btrim`: strip whitespace (1 arg) or a given char set (2 args)
/// from the left / right / both ends of a string. Non-string → NULL.
fn trim_fn(name: &str, args: &[Value]) -> Value {
    let Value::Str(s) = &args[0] else {
        return Value::Null;
    };
    // A 2nd string arg is the set of chars to strip; otherwise strip whitespace.
    let set: Option<Vec<char>> = match args.get(1) {
        None => None,
        Some(Value::Str(cs)) => Some(cs.chars().collect()),
        Some(_) => return Value::Null, // a non-string char set
    };
    let strip = |c: char| {
        set.as_ref()
            .map_or_else(|| c.is_whitespace(), |v| v.contains(&c))
    };
    let trimmed = match name {
        "ltrim" => s.trim_start_matches(strip),
        "rtrim" => s.trim_end_matches(strip),
        _ => s.trim_matches(strip), // btrim
    };
    Value::Str(trimmed.into())
}
