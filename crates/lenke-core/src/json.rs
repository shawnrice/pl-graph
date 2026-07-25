//! A small, strict, dependency-free JSON parser — the read side of the crate's
//! JSON surfaces (ndjson decode, pg-json / graphson decode), replacing
//! `serde_json`. Acceptance/rejection matches `serde_json`: RFC 8259 with a
//! 128-deep nesting cap, no trailing content, no trailing commas, no leading
//! zeros, and strings reject lone surrogates and unescaped control chars.
//! Object keys are de-duplicated last-value-wins (like `serde_json::Map`) and
//! kept in first-seen order.

/// A parsed JSON value — the slice of `serde_json::Value` the decoders consume.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Self>),
    Obj(Vec<(String, Self)>),
}

impl Json {
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub(crate) fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub(crate) fn as_object(&self) -> Option<&[(String, Self)]> {
        match self {
            Self::Obj(o) => Some(o),
            _ => None,
        }
    }
    /// The value bound to `key` in an object (last-wins on duplicate keys —
    /// but the parser already de-duplicated, so at most one match); `None`
    /// for non-objects or a missing key.
    pub(crate) fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Recognize a single-key tagged-temporal object `{"@<kind>":"<iso>"}` (the
/// wire form emitted by the JSON-family codecs). `None` if `pairs` isn't that
/// shape — the caller then treats the object as outside the LPG value model;
/// `Some(Err)` if the tag matched but the ISO string is malformed.
pub(crate) fn temporal_from_pairs(
    pairs: &[(String, Json)],
) -> Option<Result<crate::temporal::Temporal, String>> {
    let [(k, v)] = pairs else { return None };
    crate::temporal::Temporal::from_json_tag(k, v.as_str()?)
}

const MAX_DEPTH: usize = 128;

/// Parse a complete JSON document. `Err(())` on any malformed input (the callers
/// map it to their own `InvalidJson`-style error).
pub(crate) fn parse(s: &str) -> Result<Json, ()> {
    let mut p = Parser {
        b: s.as_bytes(),
        i: 0,
        depth: 0,
    };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(()); // trailing content
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }
    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.i += 1;
        }
        c
    }
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, ()> {
        self.ws();
        match self.peek().ok_or(())? {
            b'n' => self.lit("null", Json::Null),
            b't' => self.lit("true", Json::Bool(true)),
            b'f' => self.lit("false", Json::Bool(false)),
            b'"' => Ok(Json::Str(self.string()?)),
            b'[' => self.array(),
            b'{' => self.object(),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(()),
        }
    }

    fn lit(&mut self, kw: &str, val: Json) -> Result<Json, ()> {
        if self.b[self.i..].starts_with(kw.as_bytes()) {
            self.i += kw.len();
            Ok(val)
        } else {
            Err(())
        }
    }

    fn array(&mut self) -> Result<Json, ()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(());
        }
        self.i += 1; // '['
        let mut out = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Json::Arr(out));
        }
        loop {
            out.push(self.value()?);
            self.ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err(()),
            }
        }
        self.depth -= 1;
        Ok(Json::Arr(out))
    }

    fn object(&mut self) -> Result<Json, ()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(());
        }
        self.i += 1; // '{'
        let mut out: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Json::Obj(out));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(()); // keys must be strings
            }
            let key = self.string()?;
            self.ws();
            if self.bump() != Some(b':') {
                return Err(());
            }
            let val = self.value()?;
            match out.iter_mut().find(|(k, _)| *k == key) {
                Some(e) => e.1 = val, // duplicate key: last wins (serde_json::Map)
                None => out.push((key, val)),
            }
            self.ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(()),
            }
        }
        self.depth -= 1;
        Ok(Json::Obj(out))
    }

    fn number(&mut self) -> Result<Json, ()> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        // Integer part: a lone `0`, or `1-9` followed by any digits (no leading 0s).
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            _ => return Err(()),
        }
        // Fraction.
        if self.peek() == Some(b'.') {
            self.i += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        // Exponent.
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(());
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        // The slice is ASCII by construction, so from_utf8 never fails.
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|_| ())?;
        text.parse::<f64>().map(Json::Num).map_err(|_| ())
    }

    fn string(&mut self) -> Result<String, ()> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            // Copy a run of ordinary bytes (a valid UTF-8 substring — breaks only
            // at ASCII `"`, `\`, or a control byte, all of which are char boundaries).
            let start = self.i;
            while let Some(c) = self.peek() {
                if c == b'"' || c == b'\\' || c < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|_| ())?);
            match self.bump().ok_or(())? {
                b'"' => return Ok(out),
                b'\\' => self.escape(&mut out)?,
                _ => return Err(()), // an unescaped control char (< 0x20)
            }
        }
    }

    fn escape(&mut self, out: &mut String) -> Result<(), ()> {
        match self.bump().ok_or(())? {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{08}'),
            b'f' => out.push('\u{0c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let hi = self.hex4()?;
                let ch = if (0xD800..=0xDBFF).contains(&hi) {
                    // High surrogate: must be followed by `\uXXXX` low surrogate.
                    if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                        return Err(());
                    }
                    let lo = self.hex4()?;
                    if !(0xDC00..=0xDFFF).contains(&lo) {
                        return Err(());
                    }
                    let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                    char::from_u32(cp).ok_or(())?
                } else if (0xDC00..=0xDFFF).contains(&hi) {
                    return Err(()); // lone low surrogate
                } else {
                    char::from_u32(hi).ok_or(())?
                };
                out.push(ch);
            }
            _ => return Err(()),
        }
        Ok(())
    }

    fn hex4(&mut self) -> Result<u32, ()> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = match self.bump().ok_or(())? {
                c @ b'0'..=b'9' => (c - b'0') as u32,
                c @ b'a'..=b'f' => (c - b'a' + 10) as u32,
                c @ b'A'..=b'F' => (c - b'A' + 10) as u32,
                _ => return Err(()),
            };
            v = v * 16 + d;
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> Json {
        parse(s).unwrap()
    }
    fn bad(s: &str) -> bool {
        parse(s).is_err()
    }

    #[test]
    fn scalars() {
        assert_eq!(ok("null"), Json::Null);
        assert_eq!(ok("true"), Json::Bool(true));
        assert_eq!(ok("false"), Json::Bool(false));
        assert_eq!(ok(" 42 "), Json::Num(42.0));
        assert_eq!(ok("-7"), Json::Num(-7.0));
        assert_eq!(ok("1.5"), Json::Num(1.5));
        assert_eq!(ok("1.5e3"), Json::Num(1500.0));
        assert_eq!(ok("2.5e-3"), Json::Num(0.0025));
        assert_eq!(ok("\"hi\""), Json::Str("hi".into()));
    }

    #[test]
    fn strings_escapes_and_unicode() {
        assert_eq!(ok(r#""a\"b\\c\/d""#), Json::Str("a\"b\\c/d".into()));
        assert_eq!(
            ok(r#""\b\f\n\r\t""#),
            Json::Str("\u{08}\u{0c}\n\r\t".into())
        );
        assert_eq!(ok(r#""Aé""#), Json::Str("A\u{e9}".into()));
        assert_eq!(ok(r#""🦀""#), Json::Str("\u{1F980}".into())); // surrogate pair
        assert_eq!(ok("\"café\u{1F980}\""), Json::Str("café\u{1F980}".into())); // raw UTF-8
    }

    #[test]
    fn arrays_and_objects() {
        assert_eq!(ok("[]"), Json::Arr(vec![]));
        assert_eq!(ok("{}"), Json::Obj(vec![]));
        assert_eq!(
            ok("[1,2,3]"),
            Json::Arr(vec![Json::Num(1.0), Json::Num(2.0), Json::Num(3.0)])
        );
        assert_eq!(
            ok(r#"{ "a" : 1 , "b" : [true, null] }"#),
            Json::Obj(vec![
                ("a".into(), Json::Num(1.0)),
                ("b".into(), Json::Arr(vec![Json::Bool(true), Json::Null])),
            ])
        );
        // Duplicate key: last wins.
        assert_eq!(
            ok(r#"{"a":1,"a":2}"#),
            Json::Obj(vec![("a".into(), Json::Num(2.0))])
        );
    }

    #[test]
    fn rejects_malformed() {
        assert!(bad(""));
        assert!(bad("{not json"));
        assert!(bad("[1,2,]")); // trailing comma
        assert!(bad("{\"a\":1,}")); // trailing comma
        assert!(bad("01")); // leading zero
        assert!(bad("1.")); // bare fraction
        assert!(bad("1e")); // bare exponent
        assert!(bad("-")); // lone sign
        assert!(bad("42 43")); // trailing content
        assert!(bad("nul")); // truncated keyword
        assert!(bad("'single'")); // single quotes
        assert!(bad("{a:1}")); // unquoted key
        assert!(bad("\"\\x\"")); // bad escape
        assert!(bad("\"\\ud83e\"")); // lone high surrogate
        assert!(bad("\"\\udd80\"")); // lone low surrogate
        assert!(bad("\"raw\tcontrol\"".replace("\\t", "\t").as_str())); // literal control in string
    }

    #[test]
    fn deep_nesting_is_bounded() {
        let deep = format!("{}1{}", "[".repeat(2000), "]".repeat(2000));
        assert!(bad(&deep));
    }

    #[test]
    fn accessors() {
        let v = ok(r#"{"s":"x","n":5,"xs":[1]}"#);
        assert_eq!(v.get("s").and_then(Json::as_str), Some("x"));
        assert_eq!(v.get("n").and_then(Json::as_f64), Some(5.0));
        assert_eq!(
            v.get("xs").and_then(Json::as_array).map(<[_]>::len),
            Some(1)
        );
        assert!(v.get("missing").is_none());
        assert!(v.as_object().is_some());
    }
}
