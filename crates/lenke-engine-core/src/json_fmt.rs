//! Pure Value -> JSON / JS-string formatters (no query layer). Split from the
//! engine's `json` module so the value layer can format itself without depending
//! on `exec` (the `Rows` formatters stay in `lenke-engine`).

use crate::value::Value;
use std::fmt::Write;

/// Append `v` to `out` as JSON.
pub fn write_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => write_number(out, *x),
        Value::Str(s) => write_string(out, s),
        // A temporal renders TAGGED — `{"@date":"2020-01-01"}`, `{"@duration":"P1D"}`,
        // … — matching core's query-result form (and the NDJSON/elementMap wire form),
        // not a bare ISO string.
        Value::Temporal(t) => {
            out.push_str("{\"@");
            out.push_str(t.tag());
            out.push_str("\":");
            write_string(out, &t.format());
            out.push('}');
        }
        Value::List(items) => {
            out.push('[');
            for (i, it) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, it);
            }
            out.push(']');
        }
        Value::Record(fields) => {
            // A record renders as an object keyed by field name, in field order.
            out.push('{');
            for (i, (k, val)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, val);
            }
            out.push('}');
        }
        Value::Map(pairs) => {
            // A map (element map, group map, …) renders as an object in INSERTION order
            // — key order is significant for the value-level compare. Keys are stringified.
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                match k {
                    Value::Str(s) => write_string(out, s),
                    // A non-string key (a numeric group key, T.label token) → its JSON
                    // scalar text as the object key, matching core's map rendering.
                    other => {
                        let mut kb = String::new();
                        write_value(&mut kb, other);
                        write_string(out, kb.trim_matches('"'));
                    }
                }
                out.push(':');
                write_value(out, val);
            }
            out.push('}');
        }
        // SPIKE: an element ref should be RESOLVED to its element map by render_cell before it
        // reaches JSON egress; a bare one here means an egress path did not resolve it. Emit a
        // distinctive marker (not a panic) so the differential fuzzer surfaces the leak.
        Value::Node(id) => {
            let _ = write!(out, "\"__UNRESOLVED_NODE_{id}__\"");
        }
        Value::Edge(id) => {
            let _ = write!(out, "\"__UNRESOLVED_EDGE_{id}__\"");
        }
    }
}

/// A finite number as its shortest round-tripping text; a non-finite one as `null`
/// (JSON has no NaN/Infinity — core emits null too, per the NaN policy).
pub fn write_number(out: &mut String, x: f64) {
    if !x.is_finite() {
        out.push_str("null");
    } else if x.fract() == 0.0 && x.abs() < 1e15 {
        // Integer-valued: emit without a decimal point (matches `js_number` for this
        // range but skips the `{:e}` round-trip — a hot-path shortcut).
        let _ = write!(out, "{}", x as i64);
    } else {
        out.push_str(&js_number(x));
    }
}

/// Format `x` exactly as JavaScript's `Number.prototype.toString` does — the fixed /
/// exponential threshold (decimal-point position `n > 21` or `n <= -6` → exponential),
/// and `-0` → `"0"`. Rust's `{}` formats an f64 in decimal at ALL magnitudes
/// (`1e-7` → `0.0000001`, `1e21` → `1000000000000000000000`), which is a different STRING
/// from JS even though it parses to the same f64 — so any number that becomes a STRING
/// (`to_string` / `CAST AS STRING` / `||`) must route through here for byte-identity with
/// core / the TS engine. Ported from lenke-core's `jsonfmt::js_number`. `{:e}` gives the
/// shortest round-tripping mantissa; this just places the decimal point per the spec.
pub fn js_number(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string(); // also normalizes -0.0 → "0" (JS drops the sign)
    }
    let neg = x < 0.0;
    let sci = format!("{:e}", x.abs()); // e.g. "1.5e21", "1e-7"
    let (mant, exp_str) = sci.split_once('e').expect("{:e} always has an 'e'");
    let exp: i32 = exp_str.parse().expect("valid base-10 exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32; // significant digits
    let n = exp + 1; // ECMA `n`: position of the decimal point

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if k <= n && n <= 21 {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= 21 {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-n) as usize));
        out.push_str(&digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let e = n - 1;
        out.push(if e >= 0 { '+' } else { '-' });
        out.push_str(&e.abs().to_string());
    }
    out
}

/// Stringify a value like JS `String(v)` / core's `js_str` — the coercion behind
/// `to_string` and `CAST(x AS STRING)` for the COMPOSITE types (a scalar's own arm in
/// `to_string_fn` is the fast path). A LIST joins its elements with "," (a null element →
/// "", matching JS `Array.prototype.join`, NOT the top-level "null"); a RECORD/MAP renders
/// as its canonical JSON (byte-identical to how it serializes in a result). Numbers use
/// `js_number`, with non-finite ones spelled "NaN"/"Infinity"/"-Infinity" (JS `String`).
pub fn js_str(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Num(x) if x.is_nan() => "NaN".to_string(),
        Value::Num(x) if x.is_infinite() => {
            if *x > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
        }
        Value::Num(x) => js_number(*x),
        Value::Str(s) => s.to_string(),
        Value::Temporal(t) => t.format(),
        Value::List(items) => items
            .iter()
            .map(|x| match x {
                Value::Null => String::new(),
                other => js_str(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Record(_) | Value::Map(_) => {
            let mut s = String::new();
            write_value(&mut s, v);
            s
        }
        // SPIKE: element refs should be resolved before egress (see write_value).
        Value::Node(id) => format!("__UNRESOLVED_NODE_{id}__"),
        Value::Edge(id) => format!("__UNRESOLVED_EDGE_{id}__"),
    }
}

/// A JSON string literal (RFC 8259 escaping).
pub fn write_string(out: &mut String, s: &str) {
    out.push('"');
    // Fast path: if NO byte needs escaping — none is `"`, `\`, or a control byte
    // (< 0x20) — the literal is `s` verbatim, so copy it in one `push_str` (a memcpy)
    // instead of walking `char`s. UTF-8 continuation/lead bytes are all >= 0x80, so a
    // byte scan never false-flags a multi-byte char. Byte-identical to the loop below.
    if !s.bytes().any(|b| b < 0x20 || b == b'"' || b == b'\\') {
        out.push_str(s);
        out.push('"');
        return;
    }
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
