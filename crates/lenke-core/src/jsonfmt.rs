//! Shared JSON serialization primitives — one number formatter and one string
//! escaper — so the (serde-free) JSON writers all emit byte-identical output:
//! the gremlin result carrier, the ndjson encoder, and the extra codecs.
//!
//! Used by every JSON-serializing surface: the gremlin result carrier, the ndjson
//! encoder, the extra codecs, and the gql engine's row output + value
//! stringification (`js_number` — so a numeric cell / `toString(n)` renders in
//! exponential form for extreme magnitudes exactly as JS `Number::toString` does,
//! matching the TS engine byte-for-byte).

use std::fmt::Write as _;

/// Append `s` as a JSON string literal (surrounding quotes included), using the
/// standard escape set — `\" \\ \b \t \n \f \r` shortcuts, `\u00XX` for other
/// control chars, raw UTF-8 for everything else. Matches `serde_json` and JS
/// `JSON.stringify`, so the same string serializes identically on both engines.
pub(crate) fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
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

/// Format a finite `f64` exactly as JavaScript's `Number.prototype.toString`
/// (ECMA-262 Number::toString) would — fixed notation for `-6 < n ≤ 21`,
/// exponential (`1e+21`, `1e-7`) outside that, and `-0` normalized to `0`. This
/// keeps number output byte-identical to the TS side. Rust's `{:e}` gives the
/// shortest round-tripping mantissa; we just place the decimal point / pick
/// fixed-vs-exponential per the spec. Non-finite input is the caller's concern.
pub(crate) fn js_number(x: f64) -> String {
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

/// A finite number, or `null` for NaN/±Infinity (not representable in JSON).
pub(crate) fn push_num(out: &mut String, x: f64) {
    if x.is_finite() {
        out.push_str(&js_number(x));
    } else {
        out.push_str("null");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_number_matches_js_tostring() {
        // Must match JavaScript Number.prototype.toString byte-for-byte, incl.
        // the fixed/exponential threshold (n>21 or n≤-6) and -0 → "0".
        let cases: &[(f64, &str)] = &[
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (-1.5, "-1.5"),
            (100.0, "100"),
            (0.5, "0.5"),
            (1234.5, "1234.5"),
            (12300.0, "12300"),
            (0.1, "0.1"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1e20, "100000000000000000000"),
            (1e21, "1e+21"),
            (1.5e21, "1.5e+21"),
            (1e-10, "1e-10"),
            (1e100, "1e+100"),
            (-1e-7, "-1e-7"),
            (1.25, "1.25"),
        ];
        for &(x, want) in cases {
            assert_eq!(js_number(x), want, "js_number({x})");
        }
    }

    #[test]
    fn json_string_escaping() {
        let esc = |s: &str| {
            let mut out = String::new();
            push_json_str(&mut out, s);
            out
        };
        assert_eq!(esc("a\"b"), r#""a\"b""#);
        assert_eq!(esc("a\\b"), r#""a\\b""#);
        assert_eq!(esc("a/b"), r#""a/b""#); // '/' is not escaped
        assert_eq!(esc("\u{08}\t\n\u{0c}\r"), r#""\b\t\n\f\r""#);
        assert_eq!(esc("\u{01}"), r#""\u0001""#);
        assert_eq!(esc("café\u{1F980}"), "\"café\u{1F980}\""); // non-ASCII raw
    }
}
