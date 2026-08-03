//! ISO scalar functions dispatched on the resolved function enum — numeric
//! (abs/round/trig/…), string (substring/split/replace/…, UTF-16 semantics),
//! conversion (to_string/to_integer/…), list, and graph (labels/type/keys/…).
//! Extracted from the evaluator (`super`); shares its helpers via `use super::*`.
use super::*;

// --- scalar functions (dispatched on the resolved enum) ----------------------

/// Slice `len` UTF-16 code units starting at unit index `start` (JS
/// `String.slice` semantics), decoding back to a `String`. A slice that splits a
/// surrogate pair yields U+FFFD there (lossy) — an extreme edge JS keeps as a
/// lone surrogate; not worth carrying invalid UTF-16 through the engine for.
pub(super) fn utf16_slice(s: &str, start: usize, len: usize) -> String {
    let units: Vec<u16> = s.encode_utf16().collect();
    let end = start.saturating_add(len).min(units.len());
    let start = start.min(end);
    String::from_utf16_lossy(&units[start..end])
}

/// Extract a calendar/clock component from a temporal value. `None` means the
/// component is undefined for that temporal kind (`year`/`month`/`day` of a
/// time-only value, or `hour`/`minute`/`second` of a date) — the caller faults.
/// Zoned values are decomposed in their own stored offset (the local wall
/// clock), matching how they render. Division is euclidean so pre-epoch instants
/// (negative seconds) floor correctly, byte-identical to the TS `Math.floor`.
pub(super) fn date_part(func: ScalarFn, t: crate::temporal::Temporal) -> Option<i64> {
    use crate::temporal::{civil_from_days, Temporal};
    const SPD: i64 = 86_400;
    match func {
        ScalarFn::Year | ScalarFn::Month | ScalarFn::Day => {
            let days = match t {
                Temporal::Date(x) => i64::from(x.days),
                Temporal::DateTime(x) => x.secs.div_euclid(SPD),
                Temporal::ZonedDateTime(x) => (x.secs + i64::from(x.offset) * 60).div_euclid(SPD),
                _ => return None,
            };
            let (y, m, d) = civil_from_days(days);
            Some(match func {
                ScalarFn::Year => y,
                ScalarFn::Month => i64::from(m),
                _ => i64::from(d),
            })
        }
        ScalarFn::Hour | ScalarFn::Minute | ScalarFn::Second => {
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
                ScalarFn::Hour => tod / 3600,
                ScalarFn::Minute => (tod / 60) % 60,
                _ => tod % 60,
            })
        }
        _ => None,
    }
}

pub(super) fn call_scalar(graph: &Graph, ctx: &Ctx, func: ScalarFn, args: &[Val]) -> Val {
    use ScalarFn::*;
    let a = args.first();
    let b = args.get(1);
    let un = |f: fn(f64) -> f64| match a {
        Some(v) if !is_nullish(v) => Val::Num(f(num_of(v).unwrap_or(f64::NAN))),
        _ => Val::Null,
    };
    let us = |f: fn(&str) -> Val| match a {
        Some(v) if !is_nullish(v) => f(&js_str(graph, v)),
        _ => Val::Null,
    };
    let bn = |f: fn(f64, f64) -> f64| match (a, b) {
        (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => Val::Num(f(
            num_of(x).unwrap_or(f64::NAN),
            num_of(y).unwrap_or(f64::NAN),
        )),
        _ => Val::Null,
    };
    match func {
        Abs => un(f64::abs),
        Ceil => un(f64::ceil),
        Floor => un(f64::floor),
        Sqrt => un(f64::sqrt),
        Exp => un(f64::exp),
        Ln => un(f64::ln),
        Log10 => un(f64::log10),
        Sin => un(f64::sin),
        Cos => un(f64::cos),
        Tan => un(f64::tan),
        Cot => un(|n| 1.0 / n.tan()),
        Asin => un(f64::asin),
        Acos => un(f64::acos),
        Atan => un(f64::atan),
        Sinh => un(f64::sinh),
        Cosh => un(f64::cosh),
        Tanh => un(f64::tanh),
        // Spelled as the multiply-then-divide the TS twin uses, NOT `f64::to_degrees`
        // / `to_radians` — those are `n * (180/PI)` / `n * (PI/180)`, which pre-round
        // the constant and land one ulp away from `(n * 180) / PI`. Plain multiply and
        // divide are exactly specified by IEEE 754, so this form is byte-identical.
        Degrees => un(|n| (n * 180.0) / std::f64::consts::PI),
        Radians => un(|n| (n * std::f64::consts::PI) / 180.0),
        // pi()/e() are 0-arg constants; sign()/round() null-in → null-out.
        Pi => Val::Num(std::f64::consts::PI),
        E => Val::Num(std::f64::consts::E),
        Sign => match a {
            Some(v) if !is_nullish(v) => {
                let x = num_of(v).unwrap_or(f64::NAN);
                // -1 | 0 | 1 (NaN passes through) — matches the TS `mathSign`,
                // NOT `f64::signum` (which yields +1 for 0.0).
                Val::Num(if x.is_nan() {
                    f64::NAN
                } else if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                })
            }
            _ => Val::Null,
        },
        Round => match a {
            Some(v) if !is_nullish(v) => {
                let x = num_of(v).unwrap_or(f64::NAN);
                let digits = match b {
                    Some(d) if !is_nullish(d) => num_of(d).unwrap_or(0.0).trunc() as i32,
                    _ => 0,
                };
                // `f64::round` is already half-away-from-zero (the TS engine
                // reproduces this via `roundHalfAway`); same op order → same bits.
                let f = 10f64.powi(digits);
                Val::Num((x * f).round() / f)
            }
            _ => Val::Null,
        },
        Upper => us(|s| vstr(s.to_uppercase())),
        Lower => us(|s| vstr(s.to_lowercase())),
        // `trim`/`btrim` (both ends), `ltrim` (leading), `rtrim` (trailing). The
        // optional 2nd arg is a SET of characters to strip; absent → whitespace
        // (byte-identical to `str::trim*`, which is `char::is_whitespace`).
        Trim => trim_arm(a, b, graph, true, true),
        Ltrim => trim_arm(a, b, graph, true, false),
        Rtrim => trim_arm(a, b, graph, false, true),
        // String length/slicing count UTF-16 code units, matching JS `.length`
        // (the TS engine) — NOT Unicode code points. So `size('😀')` == 2, and
        // `left`/`right` slice on the same unit as JS `String.slice`.
        CharLength => us(|s| Val::Num(s.encode_utf16().count() as f64)),
        // KNOWN LIMITATION (won't-fix): `powf` is glibc's `pow`, which differs
        // from V8's `Math.pow`/`**` (the TS engine) by ≤1 ULP on some inputs —
        // e.g. power(0.7,10) → …4af here vs …4ae in JS; power(2,-0.5) → …bcd vs
        // …bcc. So `power`/`pow`/`^` are NOT byte-identical cross-engine on those
        // inputs; a true fix needs a shared deterministic pow kernel. See
        // packages/gql/README.md.
        Power => bn(|x, y| x.powf(y)),
        Mod => bn(|x, y| x % y),
        Log => bn(|base, value| value.ln() / base.ln()),
        // `atan2(y, x)` — the ISO GQL binary arctangent (quadrant-correct angle).
        Atan2 => bn(|y, x| y.atan2(x)),
        Size => match a {
            Some(Val::List(items)) => Val::Num(items.len() as f64),
            Some(Val::Str(s)) => Val::Num(s.encode_utf16().count() as f64),
            // `length`/`path_length` over a path: the hop (edge) count.
            Some(Val::Path(p)) => Val::Num(p.edges.len() as f64),
            _ => Val::Null,
        },
        Left => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                let s = js_str(graph, x);
                let n = num_of(y).unwrap_or(0.0).max(0.0) as usize;
                vstr(utf16_slice(&s, 0, n))
            }
            _ => Val::Null,
        },
        Right => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                let s = js_str(graph, x);
                let units = s.encode_utf16().count();
                let n = num_of(y).unwrap_or(0.0);
                if n <= 0.0 {
                    vstr("")
                } else {
                    let n = (n as usize).min(units);
                    vstr(utf16_slice(&s, units - n, n))
                }
            }
            _ => Val::Null,
        },
        Coalesce => args
            .iter()
            .find(|x| !is_nullish(x))
            .cloned()
            .unwrap_or(Val::Null),
        Nullif => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) && val_eq(x, y) => Val::Null,
            (Some(x), _) => x.clone(),
            _ => Val::Null,
        },
        // Shared with Gremlin's `id()` — see `Val::element_id` for why one copy.
        ElementId => a.map_or(Val::Null, |v| Val::element_id(graph, v)),
        // --- graph functions --- (label/key order is unspecified → sorted for
        // deterministic, cross-engine-identical output)
        Labels => match a {
            Some(Val::Node(i)) => {
                let mut ls: Vec<String> = graph
                    .vertex_labels(*i)
                    .iter()
                    .map(|&l| graph.labels.text(l).to_string())
                    .collect();
                ls.sort_unstable();
                Val::List(ls.into_iter().map(vstr).collect())
            }
            _ => Val::Null,
        },
        Type => match a {
            Some(Val::Edge(e)) => vstr(graph.etype.text(graph.e_type[*e as usize]).to_string()),
            _ => Val::Null,
        },
        Keys => {
            let store_idx = match a {
                Some(Val::Node(i)) => Some((&graph.props, *i as usize)),
                Some(Val::Edge(e)) => Some((&graph.edge_props, *e as usize)),
                _ => None,
            };
            match store_idx {
                Some((store, idx)) => {
                    let mut ks: Vec<String> = (0..store.keys.len() as u32)
                        .filter(|&kid| store.is_present_id(idx, kid))
                        .map(|kid| store.keys.text(kid).to_string())
                        .collect();
                    ks.sort_unstable();
                    Val::List(ks.into_iter().map(vstr).collect())
                }
                None => Val::Null,
            }
        }
        // --- path functions (ISO GQL) — vertices/edges kept as live element
        // handles, so each still serializes richly and supports property reads.
        PathNodes => match a {
            Some(Val::Path(p)) => {
                let vertices = &p.vertices;
                Val::List(vertices.iter().map(|&v| Val::Node(v)).collect())
            }
            _ => Val::Null,
        },
        PathEdges => match a {
            Some(Val::Path(p)) => {
                let edges = &p.edges;
                Val::List(edges.iter().map(|&e| Val::Edge(e)).collect())
            }
            _ => Val::Null,
        },
        PathElements => match a {
            Some(Val::Path(p)) => {
                let (vertices, edges) = (&p.vertices, &p.edges);
                let mut out = Vec::with_capacity(vertices.len() + edges.len());
                for (i, &v) in vertices.iter().enumerate() {
                    if i > 0 {
                        out.push(Val::Edge(edges[i - 1]));
                    }

                    out.push(Val::Node(v));
                }

                Val::list(out)
            }
            _ => Val::Null,
        },
        // --- conversion (null in → null out) ---
        ToString => match a {
            Some(v) if !is_nullish(v) => vstr(js_str(graph, v)),
            _ => Val::Null,
        },
        ToInteger => match a {
            Some(Val::Num(n)) => Val::Num(n.trunc()),
            // `.filter(is_finite)`: a non-finite spelling ("inf"/"nan") parses in Rust
            // but must yield NULL, matching the TS strict grammar (numericStringToFloat).
            Some(Val::Str(s)) => s
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite())
                .map_or(Val::Null, |n| Val::Num(n.trunc())),
            _ => Val::Null,
        },
        ToFloat => match a {
            Some(Val::Num(n)) => Val::Num(*n),
            Some(Val::Str(s)) => s
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite())
                .map_or(Val::Null, Val::Num),
            _ => Val::Null,
        },
        ToBoolean => match a {
            Some(Val::Bool(b)) => Val::Bool(*b),
            Some(Val::Num(n)) if !n.is_nan() => Val::Bool(*n != 0.0),
            Some(Val::Str(s)) => match s.trim().to_lowercase().as_str() {
                "true" | "yes" | "1" => Val::Bool(true),
                "false" | "no" | "0" => Val::Bool(false),
                _ => Val::Null,
            },
            _ => Val::Null,
        },
        ToList => match a {
            Some(v @ Val::List(_)) => v.clone(),
            // A string → its UTF-16 code-unit characters (same unit model as
            // split('')); any other non-null scalar → a singleton list.
            Some(Val::Str(s)) => Val::List(
                s.encode_utf16()
                    .map(|u| vstr(String::from_utf16_lossy(&[u])))
                    .collect(),
            ),
            Some(v) if !is_nullish(v) => Val::list(vec![v.clone()]),
            _ => Val::Null,
        },
        // --- string predicates / measurement ---
        Contains => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                Val::Bool(js_str(graph, x).contains(js_str(graph, y).as_str()))
            }
            _ => Val::Null,
        },
        StartsWith => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                Val::Bool(js_str(graph, x).starts_with(js_str(graph, y).as_str()))
            }
            _ => Val::Null,
        },
        EndsWith => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                Val::Bool(js_str(graph, x).ends_with(js_str(graph, y).as_str()))
            }
            _ => Val::Null,
        },
        ByteLength => match a {
            Some(v) if !is_nullish(v) => Val::Num(js_str(graph, v).len() as f64),
            _ => Val::Null,
        },
        // --- string / list ---
        Substring => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                let s = js_str(graph, x);
                // ISO GQL: 1-based start (SQL `SUBSTRING`). Convert to a 0-based
                // offset; a start <= 0 shrinks the window from the front (SQL
                // semantics), byte-identical to the TS engine.
                let zero_start = num_of(y).unwrap_or(0.0) - 1.0;
                let from = zero_start.max(0.0) as usize;
                let count = match args.get(2) {
                    Some(z) if !is_nullish(z) => {
                        let end = (zero_start + num_of(z).unwrap_or(0.0)).max(0.0) as usize;
                        end.saturating_sub(from)
                    }
                    _ => usize::MAX,
                };
                vstr(utf16_slice(&s, from, count))
            }
            _ => Val::Null,
        },
        Split => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                let s = js_str(graph, x);
                let delim = js_str(graph, y);
                let parts: Vec<Val> = if delim.is_empty() {
                    // Empty delimiter → one element per UTF-16 code unit (JS
                    // `.length` model), matching the TS engine. A lone surrogate
                    // decodes to U+FFFD (`from_utf16_lossy`) — see the module note
                    // on the UTF-16 non-conformance; this keeps both engines
                    // byte-identical (UTF-8 can't carry a lone surrogate).
                    s.encode_utf16()
                        .map(|u| vstr(String::from_utf16_lossy(&[u])))
                        .collect()
                } else {
                    s.split(delim.as_str()).map(vstr).collect()
                };
                Val::list(parts)
            }
            _ => Val::Null,
        },
        Replace => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                let s = js_str(graph, x);
                let search = js_str(graph, y);
                let repl = match args.get(2) {
                    Some(z) if !is_nullish(z) => js_str(graph, z),
                    _ => String::new(),
                };
                if search.is_empty() {
                    vstr(s)
                } else {
                    vstr(s.replace(search.as_str(), &repl))
                }
            }
            _ => Val::Null,
        },
        Head => match a {
            Some(Val::List(items)) => items.first().cloned().unwrap_or(Val::Null),
            _ => Val::Null,
        },
        Last => match a {
            Some(Val::List(items)) => items.last().cloned().unwrap_or(Val::Null),
            _ => Val::Null,
        },
        Tail => match a {
            Some(Val::List(items)) => Val::List(items.iter().skip(1).cloned().collect()),
            _ => Val::Null,
        },
        Append => match a {
            // The element may be null (a first-class value); only a null LIST is
            // null-in → null-out.
            Some(Val::List(items)) => {
                let mut v = items.to_vec();
                v.push(b.cloned().unwrap_or(Val::Null));
                Val::list(v)
            }
            _ => Val::Null,
        },
        // --- set-style list functions (all dedup; first occurrence wins) ---
        ListUnion => match (a, b) {
            (Some(Val::List(x)), Some(Val::List(y))) => {
                let mut out = Vec::new();
                for v in x.iter().chain(y.iter()) {
                    push_unique(&mut out, v);
                }
                Val::list(out)
            }
            _ => Val::Null,
        },
        Intersection => match (a, b) {
            (Some(Val::List(x)), Some(Val::List(y))) => {
                let mut out = Vec::new();
                for v in x.iter() {
                    if y.iter().any(|w| val_eq(w, v)) {
                        push_unique(&mut out, v);
                    }
                }
                Val::list(out)
            }
            _ => Val::Null,
        },
        Difference => match (a, b) {
            (Some(Val::List(x)), Some(Val::List(y))) => {
                let mut out = Vec::new();
                for v in x.iter() {
                    if !y.iter().any(|w| val_eq(w, v)) {
                        push_unique(&mut out, v);
                    }
                }
                Val::list(out)
            }
            _ => Val::Null,
        },
        // ISO GQL `list_contains` returns the numeric 1 / 0 (per its Return Type),
        // not a boolean. The value may be null (a first-class value).
        ListContains => match a {
            Some(Val::List(items)) => {
                let found = b.is_some_and(|v| items.iter().any(|w| val_eq(w, v)));
                Val::Num(if found { 1.0 } else { 0.0 })
            }
            _ => Val::Null,
        },
        // list_sort(list, [order], [nullOrder]) — reuses the ORDER BY total order
        // (`compare_sort`) so a sorted list matches ORDER BY byte-for-byte. Stable.
        ListSort => match a {
            Some(Val::List(items)) => {
                let descending = matches!(b, Some(Val::Str(s)) if s.eq_ignore_ascii_case("desc"));
                let nulls_first = match args.get(2) {
                    Some(Val::Str(s)) if s.eq_ignore_ascii_case("first") => Some(true),
                    Some(Val::Str(s)) if s.eq_ignore_ascii_case("last") => Some(false),
                    _ => None,
                };
                let mut sorted = items.to_vec();
                sorted.sort_by(|x, y| compare_sort(x, y, descending, nulls_first));
                Val::list(sorted)
            }
            _ => Val::Null,
        },
        Range => match (a, b) {
            (Some(x), Some(y)) if !is_nullish(x) && !is_nullish(y) => {
                let s = num_of(x).unwrap_or(0.0).trunc();
                let e = num_of(y).unwrap_or(0.0).trunc();
                let st = match args.get(2) {
                    Some(z) if !is_nullish(z) => num_of(z).unwrap_or(1.0).trunc(),
                    _ => 1.0,
                };
                if st == 0.0 {
                    Val::Null // a zero step has no defined progression
                } else {
                    // Inclusive of both bounds (Cypher/ISO convention). The element
                    // count is computed UP FRONT, for two reasons. It bounds the
                    // allocation against `RANGE_BUDGET` before a single push — the
                    // list is materialized, so an unbounded range is an OOM kill
                    // rather than a query error. And it makes the loop COUNT-driven
                    // instead of comparison-driven: `i += st` stops advancing once
                    // `i` reaches 2^53 (a no-op in f64), so `while i <= e` never
                    // terminates for a large enough end — even when the count is
                    // tiny, as in `range(9007199254740992, 9007199254740994)`.
                    // The values themselves still come from repeated addition, so
                    // the emitted sequence is unchanged.
                    let count = ((e - s) / st).floor() + 1.0;
                    if count.is_nan() || count <= 0.0 {
                        // A backwards span (or a NaN bound) yields no elements.
                        Val::list(Vec::new())
                    } else if count > graph.limits().range as f64 {
                        ctx.set_fault(FAULT_RANGE_BUDGET);
                        Val::Null
                    } else {
                        let n = count as usize;
                        let mut out = Vec::with_capacity(n);
                        let mut i = s;
                        for _ in 0..n {
                            out.push(Val::Num(i));
                            i += st;
                        }
                        Val::list(out)
                    }
                }
            }
            _ => Val::Null,
        },
        Reverse => match a {
            Some(Val::List(items)) => Val::List(items.iter().rev().cloned().collect()),
            // Reverse by UTF-16 code unit (JS `.length` model), lossy-decoding
            // the reversed units the same way the TS engine does. Reversing
            // across a surrogate pair is inherently lossy → U+FFFD on both.
            Some(Val::Str(s)) => {
                let mut units: Vec<u16> = s.encode_utf16().collect();
                units.reverse();
                vstr(String::from_utf16_lossy(&units))
            }
            _ => Val::Null,
        },
        DateOf => temporal_ctor(a, "date"),
        LocalTimeOf => temporal_ctor(a, "localtime"),
        DateTimeOf => temporal_ctor(a, "datetime"),
        ZonedTimeOf => temporal_ctor(a, "zoned_time"),
        ZonedDateTimeOf => temporal_ctor(a, "zoned_datetime"),
        DurationOf => temporal_ctor(a, "duration"),
        DurationBetween => match (a, b) {
            (Some(Val::Temporal(x)), Some(Val::Temporal(y))) => duration_between(x, y),
            _ => Val::Null, // null operand or a non-temporal → UNKNOWN
        },
        // Temporal component extraction. Null in → null out; a temporal that
        // carries the component → its integer value; anything else (a string, a
        // number, or a temporal lacking the component — `year` of a time, `hour`
        // of a date) faults loudly rather than coercing or returning null.
        Year | Month | Day | Hour | Minute | Second => match a {
            None => Val::Null,
            Some(v) if is_nullish(v) => Val::Null,
            Some(Val::Temporal(t)) => match date_part(func, *t) {
                Some(n) => Val::Num(n as f64),
                None => {
                    ctx.set_fault(FAULT_DATE_PART);
                    Val::Null
                }
            },
            Some(_) => {
                ctx.set_fault(FAULT_DATE_PART);
                Val::Null
            }
        },
        Unknown => Val::Null,
    }
}

/// The `date(x)` / `local_datetime(x)` / `duration(x)` constructors: parse a
/// string, or convert a temporal by kind (`date(datetime)` → the date part,
/// `local_datetime(date)` → midnight). Null / bad string / unconvertible → null
/// (lenient, like the `to_*` conversions).
pub(super) fn temporal_ctor(v: Option<&Val>, kind: &str) -> Val {
    use crate::temporal::{Date, DateTime, Temporal, Time};
    const SECS_PER_DAY: i64 = 86_400;
    let Some(v) = v else { return Val::Null };
    match v {
        // A bare date-only `YYYY-MM-DD` (no time part) coerces to midnight for a
        // datetime target — consistent with date() and the DATE `$__now` → midnight
        // precedent. Mirrors the TS `temporalCtor`.
        Val::Str(s) if kind == "datetime" && !s.contains(['T', ' ']) => Date::parse(s)
            .map(|d| {
                Val::Temporal(Temporal::DateTime(DateTime {
                    secs: d.days as i64 * SECS_PER_DAY,
                    nanos: 0,
                }))
            })
            .unwrap_or(Val::Null),
        Val::Str(s) => Temporal::parse(kind, s)
            .map(Val::Temporal)
            .unwrap_or(Val::Null),
        Val::Temporal(t) => match (kind, t) {
            ("date", Temporal::Date(_))
            | ("localtime", Temporal::Time(_))
            | ("datetime", Temporal::DateTime(_))
            | ("duration", Temporal::Duration(_)) => Val::Temporal(*t),
            ("date", Temporal::DateTime(dt)) => Val::Temporal(Temporal::Date(Date {
                days: dt.secs.div_euclid(SECS_PER_DAY) as i32,
            })),
            // local_time(datetime) → the time-of-day part.
            ("localtime", Temporal::DateTime(dt)) => Val::Temporal(Temporal::Time(Time {
                secs: dt.secs.rem_euclid(SECS_PER_DAY) as u32,
                nanos: dt.nanos,
            })),
            ("datetime", Temporal::Date(d)) => Val::Temporal(Temporal::DateTime(DateTime {
                secs: d.days as i64 * SECS_PER_DAY,
                nanos: 0,
            })),
            _ => Val::Null, // e.g. duration(date) — no sensible conversion
        },
        _ => Val::Null,
    }
}

/// `duration_between(a, b)` = the EXACT span from `a` to `b` (b − a). Both ends
/// are pinned, so the result is a measurement, expressed only in fixed units:
/// whole days for two dates, seconds+nanos for two datetimes. Cross-kind pairs
/// (or duration operands) → null.
pub(super) fn duration_between(
    a: &crate::temporal::Temporal,
    b: &crate::temporal::Temporal,
) -> Val {
    use crate::temporal::{Duration, Temporal};
    match (a, b) {
        (Temporal::Date(x), Temporal::Date(y)) => Val::Temporal(Temporal::Duration(Duration {
            months: 0,
            days: (y.days - x.days) as i64,
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
            Val::Temporal(Temporal::Duration(Duration {
                months: 0,
                days: 0,
                secs,
                nanos: nanos as u32,
            }))
        }
        _ => Val::Null,
    }
}

/// Temporal arithmetic for `+`/`-`/`*` when either operand is temporal: an
/// instant ± a (nominal) duration anchors the duration to the concrete date
/// (calendar months clamped, then days, then time); instant − instant is the
/// exact span; duration ± duration is component-wise; duration × integer scales.
/// Any undefined combination → null.
pub(super) fn temporal_arith(ctx: &Ctx, op: crate::gql::ast::ArithOp, lv: &Val, rv: &Val) -> Val {
    use crate::gql::ast::ArithOp;
    use crate::temporal::{Duration, Temporal as T};
    if is_nullish(lv) || is_nullish(rv) {
        return Val::Null;
    }
    // A duration whose sum/scale overflows the representable (f64-safe-integer)
    // range is a **data exception**, not a silent null — the result is a real
    // duration we can't store, so fail loud (byte-identical to TS), like division
    // by zero.
    let dur = |r: Option<Duration>| match r {
        Some(d) => Val::Temporal(T::Duration(d)),
        None => {
            ctx.set_fault(FAULT_DURATION_OVERFLOW);
            Val::Null
        }
    };
    // Instant ± duration whose result leaves the representable date range (Date is
    // i32 days, ≈±5.88M years) is likewise a **data exception**, not a silent null:
    // the target date is a real calendar date we can't store, so fail loud — same
    // as duration overflow and division by zero (supersedes the old D4 → null).
    let inst = |r: Option<T>| match r {
        Some(t) => Val::Temporal(t),
        None => {
            ctx.set_fault(FAULT_DATE_OVERFLOW);
            Val::Null
        }
    };
    match (op, lv, rv) {
        (ArithOp::Add, Val::Temporal(T::Duration(a)), Val::Temporal(T::Duration(b))) => {
            dur(a.add(b))
        }
        (ArithOp::Sub, Val::Temporal(T::Duration(a)), Val::Temporal(T::Duration(b))) => {
            dur(a.add(&b.negate()))
        }
        // instant ± duration (either order for +).
        (ArithOp::Add, Val::Temporal(t), Val::Temporal(T::Duration(d)))
        | (ArithOp::Add, Val::Temporal(T::Duration(d)), Val::Temporal(t)) => {
            inst(t.add_duration(d))
        }
        (ArithOp::Sub, Val::Temporal(t), Val::Temporal(T::Duration(d))) => {
            inst(t.add_duration(&d.negate()))
        }
        // instant − instant → the exact span from `b` to `a` (a − b).
        (ArithOp::Sub, Val::Temporal(a), Val::Temporal(b)) => duration_between(b, a),
        // duration × INTEGER (either order). A calendar duration (with a
        // `months` component) has no meaningful fractional multiple, so a
        // non-integer factor is invalid → null, never a silently-truncated value.
        (ArithOp::Mul, Val::Temporal(T::Duration(d)), Val::Num(n))
        | (ArithOp::Mul, Val::Num(n), Val::Temporal(T::Duration(d))) => {
            if n.fract() == 0.0 && n.is_finite() {
                dur(d.scale(*n as i64))
            } else {
                Val::Null
            }
        }
        _ => Val::Null,
    }
}
