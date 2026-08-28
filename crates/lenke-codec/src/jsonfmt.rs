//! Serde-free JSON serialization primitives — one number formatter, one string
//! escaper, one value writer — byte-identical to `lenke-core`'s `jsonfmt` (so the
//! shared codecs emit exactly what core did before they moved here).

use crate::model::Value;
use std::fmt::Write as _;

/// Append `s` as a JSON string literal (quotes included), standard escape set.
/// Matches `serde_json` / JS `JSON.stringify`.
pub fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    // Fast path: nothing to escape (ids/names/most values) → one bulk copy, no
    // per-char match. Only `"`, `\`, and control bytes (< 0x20) escape, and each is a
    // single byte in UTF-8, so scanning bytes is exact.
    if s.bytes().all(|b| b >= 0x20 && b != b'"' && b != b'\\') {
        out.push_str(s);
        out.push('"');
        return;
    }
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Format a finite `f64` exactly as JS `Number.prototype.toString` (ECMA-262):
/// fixed for `-6 < n <= 21`, exponential outside, `-0` → `0`.
#[must_use]
pub fn js_number(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let neg = x < 0.0;
    let sci = format!("{:e}", x.abs());
    let (mant, exp_str) = sci.split_once('e').expect("{:e} always has an 'e'");
    let exp: i32 = exp_str.parse().expect("valid base-10 exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp + 1;

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

/// A finite number, or `null` for NaN/±Infinity.
pub fn push_num(out: &mut String, x: f64) {
    if x.is_finite() {
        out.push_str(&js_number(x));
    } else {
        out.push_str("null");
    }
}

/// The JSON tagged form of a temporal: `{"@<tag>":"<iso>"}`.
pub fn push_temporal(out: &mut String, tag: &str, iso: &str) {
    out.push_str("{\"@");
    out.push_str(tag);
    out.push_str("\":");
    push_json_str(out, iso);
    out.push('}');
}

/// Emit a neutral [`Value`] as a JSON value (the one writer for every codec).
pub fn push_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => push_num(out, *x),
        Value::Str(s) => push_json_str(out, s),
        Value::Temporal { tag, iso } => push_temporal(out, tag, iso),
        Value::List(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_value(out, e);
            }
            out.push(']');
        }
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, e)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_str(out, k);
                out.push(':');
                push_value(out, e);
            }
            out.push('}');
        }
    }
}
