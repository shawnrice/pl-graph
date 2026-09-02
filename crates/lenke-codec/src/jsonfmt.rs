//! Serde-free JSON serialization primitives — one number formatter, one string
//! escaper, one value writer — byte-identical to the TS engine's `jsonfmt` (so the
//! shared codecs emit exactly what the now-removed lenke-core did before they moved here).

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
/// fixed for `-6 < n <= 21`, exponential outside, `-0` → `0`. Allocating wrapper
/// over [`push_js_number`] for callers that need an owned `String`.
#[must_use]
pub fn js_number(x: f64) -> String {
    let mut out = String::new();
    push_js_number(&mut out, x);
    out
}

/// A fixed 32-byte scratch that `write!` can target with no heap allocation — big
/// enough for any `{:e}`-formatted `f64` (≤17 mantissa digits + `.` + `e±NNN`).
struct Scratch {
    buf: [u8; 32],
    len: usize,
}
impl std::fmt::Write for Scratch {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let b = s.as_bytes();
        let end = self.len + b.len();
        if end > self.buf.len() {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..end].copy_from_slice(b);
        self.len = end;
        Ok(())
    }
}

/// Write `x` as its JS `Number.toString` form straight into `out` — the hot-path
/// twin of [`js_number`] with NO per-number allocation. Byte-identical (proven in
/// tests): the shortest decimal (`{:e}`) and the exponent digits both land in stack
/// buffers, and the ECMA-262 fixed/exponential formatting is the same.
pub fn push_js_number(out: &mut String, x: f64) {
    if x == 0.0 {
        out.push('0');
        return;
    }
    let neg = x < 0.0;
    let mut sci = Scratch {
        buf: [0; 32],
        len: 0,
    };
    let _ = write!(sci, "{:e}", x.abs());
    // ASCII by construction (digits, '.', 'e', '-', '+').
    let sci_s = unsafe { std::str::from_utf8_unchecked(&sci.buf[..sci.len]) };
    let (mant, exp_str) = sci_s.split_once('e').expect("{:e} always has an 'e'");
    let exp: i32 = exp_str.parse().expect("valid base-10 exponent");
    // Significant digits with the '.' removed, into a stack buffer (≤17 for f64).
    let mut dig = [0u8; 24];
    let mut dn = 0;
    for &c in mant.as_bytes() {
        if c != b'.' {
            dig[dn] = c;
            dn += 1;
        }
    }
    let digits = unsafe { std::str::from_utf8_unchecked(&dig[..dn]) };
    let k = dn as i32;
    let n = exp + 1;

    if neg {
        out.push('-');
    }
    if k <= n && n <= 21 {
        out.push_str(digits);
        for _ in 0..(n - k) {
            out.push('0');
        }
    } else if 0 < n && n <= 21 {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        for _ in 0..(-n) {
            out.push('0');
        }
        out.push_str(digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        let e = n - 1;
        out.push(if e >= 0 { '+' } else { '-' });
        push_u32(out, e.unsigned_abs());
    }
}

/// Append `v`'s decimal digits with no allocation (a tiny itoa for the exponent).
fn push_u32(out: &mut String, v: u32) {
    if v == 0 {
        out.push('0');
        return;
    }
    let mut b = [0u8; 10];
    let mut i = 0;
    let mut v = v;
    while v > 0 {
        b[i] = b'0' + (v % 10) as u8;
        i += 1;
        v /= 10;
    }
    while i > 0 {
        i -= 1;
        out.push(b[i] as char);
    }
}

/// A finite number, or `null` for NaN/±Infinity.
pub fn push_num(out: &mut String, x: f64) {
    if x.is_finite() {
        push_js_number(out, x);
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

#[cfg(test)]
mod jsnum_tests {
    use super::*;

    // The ORIGINAL allocating implementation, as an oracle.
    fn reference(x: f64) -> String {
        if x == 0.0 {
            return "0".to_string();
        }
        let neg = x < 0.0;
        let sci = format!("{:e}", x.abs());
        let (mant, exp_str) = sci.split_once('e').unwrap();
        let exp: i32 = exp_str.parse().unwrap();
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

    #[test]
    fn push_js_number_matches_the_original_across_the_range() {
        let mut xs: Vec<f64> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            100.0,
            1e21,
            1e-6,
            1e-7,
            9.999999e20,
            1.5e21,
            123456789012345.0,
            0.1,
            0.2,
            0.3,
            1.0 / 3.0,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            2.5,
            42.0,
            -0.0001,
            12345.6789,
            1e308,
            1e-308,
        ];
        // A spread of magnitudes and mantissas.
        let mut seed: u64 = 0x1234_5678;
        for _ in 0..20000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = seed ^ (seed >> 29);
            let x = f64::from_bits(bits);
            if x.is_finite() {
                xs.push(x);
            }
        }
        for &x in &xs {
            assert_eq!(
                js_number(x),
                reference(x),
                "diverged at {x:?} (bits {:#x})",
                x.to_bits()
            );
        }
    }
}
