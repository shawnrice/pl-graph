//! Query-parameter decoding for the FFI / wasm / napi boundary.
//!
//! Bindings cross as one flat JSON object (`{"name": value, …}`) and are
//! decoded here into [`Params`] ([`Val`] bindings). This is the whole safety
//! story for parameterized queries: **values never touch the GQL parser** —
//! they bind to already-parsed `$name` slots at execute time — so this module
//! only has to be a faithful *data* decoder, never an escaper.
//!
//! Hand-rolled (like the GQL lexer and `query::to_json`) so the `gql`-only
//! build keeps its no-`serde_json` promise. Deliberately strict and small:
//!
//! - the top level must be a single JSON object;
//! - values may be `string | number | true | false | null`, or a flat array
//!   of those (→ [`Val::List`]);
//! - nested objects and nested arrays are rejected — a binding is a value,
//!   not a document;
//! - trailing input after the closing `}` is rejected;
//! - duplicate keys follow JSON convention (last wins).
//!
//! All JSON numbers decode to [`Val::Num`] (`f64`), matching the engine's
//! single numeric type.

use std::collections::HashMap;

use crate::error::{CodeError, CodeResult};
use crate::error_codes::ErrorCode;
use crate::gql::eval::{Params, Val};

/// Decode a flat JSON object of bindings into [`Params`].
pub fn params_from_json(text: &str) -> CodeResult<Params> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };

    p.skip_ws();
    let params = p.object()?;
    p.skip_ws();

    if p.pos != p.bytes.len() {
        return Err(p.err("trailing characters after params object"));
    }

    Ok(params)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    /// A **syntax** failure: these bytes are not valid JSON. Bindings cross the
    /// FFI boundary as a JSON document, so a malformed one is exactly that.
    fn err(&self, msg: &str) -> CodeError {
        CodeError::new(
            ErrorCode::InvalidJson,
            format!("params: {msg} (at byte {})", self.pos),
        )
    }

    /// A **value** failure: the JSON parsed fine and then broke a rule about what
    /// a binding may hold. `[[1]]`, `{"a":1}` and `{"@date":"nope"}` are all
    /// well-formed JSON — calling them invalid JSON would blame the transport for
    /// a value the caller got wrong, and would report a code the TS engine (which
    /// takes JS objects and parses no JSON at all) has no way to mirror honestly.
    fn err_value(&self, msg: &str) -> CodeError {
        CodeError::new(
            ErrorCode::InvalidValue,
            format!("params: {msg} (at byte {})", self.pos),
        )
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> CodeResult<()> {
        if self.peek() == Some(byte) {
            self.pos += 1;

            Ok(())
        } else {
            Err(self.err(&format!("expected `{}`", byte as char)))
        }
    }

    fn object(&mut self) -> CodeResult<Params> {
        self.expect(b'{')?;
        let mut params = HashMap::new();
        self.skip_ws();

        if self.peek() == Some(b'}') {
            self.pos += 1;

            return Ok(params);
        }

        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            let value = self.value(true)?;
            params.insert(key, value); // duplicate keys: last wins (JSON convention)
            self.skip_ws();

            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;

                    return Ok(params);
                }
                _ => return Err(self.err("expected `,` or `}` in params object")),
            }
        }
    }

    /// One binding value. `allow_list` is true only at the top of a binding —
    /// a list may not contain lists (bindings are values, not documents).
    fn value(&mut self, allow_list: bool) -> CodeResult<Val> {
        match self.peek() {
            Some(b'"') => Ok(Val::Str(self.string()?.into())),
            Some(b't') => self.literal(b"true", Val::Bool(true)),
            Some(b'f') => self.literal(b"false", Val::Bool(false)),
            Some(b'n') => self.literal(b"null", Val::Null),
            Some(b'[') if allow_list => self.list(),
            Some(b'[') => Err(self.err_value("nested arrays are not valid param values")),
            // The only valid object param value is a tagged temporal
            // (`{"@date":"…"}` / `@datetime` / `@duration`).
            Some(b'{') => self.temporal_object(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.err("expected a param value")),
        }
    }

    /// A single-key tagged temporal object `{"@date":"…"}` as a `Val::Temporal`.
    /// The one object shape allowed as a param value — so a `$d` can bind a
    /// DATE/DATETIME/DURATION (and the reserved `$__now` carries `current_*`).
    fn temporal_object(&mut self) -> CodeResult<Val> {
        self.expect(b'{')?;
        self.skip_ws();
        let key = self.string()?;
        self.skip_ws();
        self.expect(b':')?;
        self.skip_ws();
        // The KEY decides whether this object can be a binding at all, so settle
        // that before demanding a string value. `{"a":1}` is well-formed JSON that
        // simply isn't a param; faulting on the `1` would report a syntax error
        // about the document when the caller's mistake was the value.
        if !crate::temporal::Temporal::is_json_tag(&key) {
            return Err(self.err_value("the only valid object param value is a tagged temporal"));
        }
        if self.peek() != Some(b'"') {
            return Err(self.err_value("a tagged temporal's value must be an ISO-8601 string"));
        }

        let iso = self.string()?;
        self.skip_ws();
        self.expect(b'}')?;
        match crate::temporal::Temporal::from_json_tag(&key, &iso) {
            Some(Ok(t)) => Ok(Val::Temporal(t)),
            Some(Err(e)) => Err(self.err_value(&format!("invalid temporal param: {e}"))),
            None => Err(self.err_value("the only valid object param value is a tagged temporal")),
        }
    }

    fn literal(&mut self, word: &[u8], val: Val) -> CodeResult<Val> {
        if self.bytes[self.pos..].starts_with(word) {
            self.pos += word.len();

            Ok(val)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn list(&mut self) -> CodeResult<Val> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();

        if self.peek() == Some(b']') {
            self.pos += 1;

            return Ok(Val::List(items));
        }

        loop {
            self.skip_ws();
            items.push(self.value(false)?);
            self.skip_ws();

            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;

                    return Ok(Val::List(items));
                }
                _ => return Err(self.err("expected `,` or `]` in param list")),
            }
        }
    }

    /// JSON number grammar: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
    fn number(&mut self) -> CodeResult<Val> {
        let start = self.pos;

        if self.peek() == Some(b'-') {
            self.pos += 1;
        }

        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(self.err("invalid number")),
        }

        if self.peek() == Some(b'.') {
            self.pos += 1;

            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("invalid number: expected digits after `.`"));
            }

            self.digits();
        }

        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;

            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }

            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("invalid number: expected exponent digits"));
            }

            self.digits();
        }

        // SAFETY-free: the span is ASCII by construction of the grammar above.
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("number span is ASCII by construction");
        let n: f64 = text.parse().map_err(|_| self.err("number out of range"))?;

        if !n.is_finite() {
            return Err(self.err("number out of range"));
        }

        Ok(Val::Num(n))
    }

    fn digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
    }

    /// A JSON string with full escape handling, including `\uXXXX` and UTF-16
    /// surrogate pairs. Control characters must be escaped, per JSON.
    fn string(&mut self) -> CodeResult<String> {
        self.expect(b'"')?;
        let mut out = String::new();

        loop {
            let Some(b) = self.peek() else {
                return Err(self.err("unterminated string"));
            };

            match b {
                b'"' => {
                    self.pos += 1;

                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    out.push(self.escape()?);
                }
                0x00..=0x1f => return Err(self.err("unescaped control character in string")),
                _ => {
                    // Copy one whole UTF-8 scalar (the input is a valid &str).
                    let len = utf8_len(b);
                    let span = self
                        .bytes
                        .get(self.pos..self.pos + len)
                        .ok_or_else(|| self.err("truncated UTF-8 sequence"))?;
                    out.push_str(std::str::from_utf8(span).map_err(|_| self.err("invalid UTF-8"))?);
                    self.pos += len;
                }
            }
        }
    }

    fn escape(&mut self) -> CodeResult<char> {
        let Some(b) = self.peek() else {
            return Err(self.err("unterminated escape"));
        };

        self.pos += 1;

        Ok(match b {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{0008}',
            b'f' => '\u{000c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => return self.unicode_escape(),
            _ => return Err(self.err("invalid escape sequence")),
        })
    }

    fn unicode_escape(&mut self) -> CodeResult<char> {
        let hi = self.hex4()?;

        // A high surrogate must be followed by `\uXXXX` with a low surrogate;
        // together they name one supplementary-plane scalar.
        if (0xd800..=0xdbff).contains(&hi) {
            if self.peek() == Some(b'\\') && self.bytes.get(self.pos + 1) == Some(&b'u') {
                self.pos += 2;
                let lo = self.hex4()?;

                if !(0xdc00..=0xdfff).contains(&lo) {
                    return Err(self.err("invalid low surrogate"));
                }

                let scalar = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);

                return char::from_u32(scalar).ok_or_else(|| self.err("invalid surrogate pair"));
            }

            return Err(self.err("lone high surrogate"));
        }

        if (0xdc00..=0xdfff).contains(&hi) {
            return Err(self.err("lone low surrogate"));
        }

        char::from_u32(hi).ok_or_else(|| self.err("invalid unicode escape"))
    }

    fn hex4(&mut self) -> CodeResult<u32> {
        let span = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.err("truncated \\u escape"))?;
        let mut v = 0u32;

        for &b in span {
            v = v * 16
                + match b {
                    b'0'..=b'9' => u32::from(b - b'0'),
                    b'a'..=b'f' => u32::from(b - b'a' + 10),
                    b'A'..=b'F' => u32::from(b - b'A' + 10),
                    _ => return Err(self.err("invalid hex digit in \\u escape")),
                };
        }

        self.pos += 4;

        Ok(v)
    }
}

/// Byte length of the UTF-8 scalar starting with `b` (input is a valid &str,
/// so the lead byte fully determines it).
const fn utf8_len(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_of(v: &Val) -> &str {
        match v {
            Val::Str(s) => s,
            other => panic!("expected Str, got {other:?}"),
        }
    }

    #[test]
    fn scalars_decode() {
        let p = params_from_json(r#"{"s":"hi","n":2.5,"i":-7,"b":true,"z":null}"#).unwrap();
        assert_eq!(str_of(&p["s"]), "hi");
        assert!(matches!(p["n"], Val::Num(x) if (x - 2.5).abs() < 1e-12));
        assert!(matches!(p["i"], Val::Num(x) if (x + 7.0).abs() < 1e-12));
        assert!(matches!(p["b"], Val::Bool(true)));
        assert!(matches!(p["z"], Val::Null));
    }

    #[test]
    fn empty_object_is_empty_params() {
        assert!(params_from_json("{}").unwrap().is_empty());
        assert!(params_from_json("  { }  ").unwrap().is_empty());
    }

    #[test]
    fn flat_lists_decode_and_nested_reject() {
        let p = params_from_json(r#"{"xs":[1,"two",false,null]}"#).unwrap();
        let Val::List(xs) = &p["xs"] else {
            panic!("expected list")
        };
        assert_eq!(xs.len(), 4);
        assert!(params_from_json(r#"{"xs":[[1]]}"#).is_err());
        assert!(params_from_json(r#"{"xs":{"a":1}}"#).is_err());
        assert!(params_from_json(r#"{"xs":[{"a":1}]}"#).is_err());
    }

    #[test]
    fn escapes_decode_faithfully() {
        let p = params_from_json(r#"{"s":"a\"b\\c\/d\n\t\r\b\f"}"#).unwrap();
        assert_eq!(str_of(&p["s"]), "a\"b\\c/d\n\t\r\u{8}\u{c}");
    }

    #[test]
    fn unicode_escapes_and_surrogate_pairs() {
        let p = params_from_json(r#"{"s":"é☃ 😀"}"#).unwrap();
        assert_eq!(str_of(&p["s"]), "é☃ 😀");
        assert!(params_from_json(r#"{"s":"\ud83d"}"#).is_err()); // lone high
        assert!(params_from_json(r#"{"s":"\ude00"}"#).is_err()); // lone low
        assert!(params_from_json(r#"{"s":"\ud83dA"}"#).is_err()); // bad low
    }

    #[test]
    fn raw_utf8_passes_through() {
        let p = params_from_json(r#"{"s":"héllo — 世界 🚀"}"#).unwrap();
        assert_eq!(str_of(&p["s"]), "héllo — 世界 🚀");
    }

    #[test]
    fn injection_shaped_strings_are_just_data() {
        // The whole point: these decode as inert data, never touching the
        // GQL parser. Quotes, operators, and clause keywords stay literal.
        let hostile = r#"{"name":"' OR 1=1 --","q":"\"}) DELETE p //","tick":"`$x`"}"#;
        let p = params_from_json(hostile).unwrap();
        assert_eq!(str_of(&p["name"]), "' OR 1=1 --");
        assert_eq!(str_of(&p["q"]), "\"}) DELETE p //");
        assert_eq!(str_of(&p["tick"]), "`$x`");
    }

    #[test]
    fn malformed_inputs_reject_with_invalid_json() {
        for bad in [
            "",
            "null",
            "[]",
            "42",
            r#""str""#,
            "{",
            r#"{"a"}"#,
            r#"{"a":}"#,
            r#"{"a":1,}"#,
            r#"{"a":1} trailing"#,
            r#"{"a":01}"#,
            r#"{"a":1.}"#,
            r#"{"a":1e}"#,
            r#"{"a":truthy}"#,
            "{\"a\":\"\u{0009}raw-tab-ok?\"}", // raw control char in string
            r#"{"a":"\x41"}"#,
        ] {
            let e = params_from_json(bad).unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidJson, "should reject: {bad}");
        }
    }

    #[test]
    fn duplicate_keys_last_wins() {
        let p = params_from_json(r#"{"a":1,"a":2}"#).unwrap();
        assert!(matches!(p["a"], Val::Num(x) if (x - 2.0).abs() < 1e-12));
    }

    #[test]
    fn number_grammar_edges() {
        let p = params_from_json(r#"{"a":0,"b":-0.5,"c":1e3,"d":2.5E-2}"#).unwrap();
        assert!(matches!(p["a"], Val::Num(x) if x == 0.0));
        assert!(matches!(p["b"], Val::Num(x) if (x + 0.5).abs() < 1e-12));
        assert!(matches!(p["c"], Val::Num(x) if (x - 1000.0).abs() < 1e-9));
        assert!(matches!(p["d"], Val::Num(x) if (x - 0.025).abs() < 1e-12));
        assert!(params_from_json(r#"{"a":1e999}"#).is_err()); // overflows to inf
    }

    // -----------------------------------------------------------------------
    // Error CLASS: a malformed document is invalid JSON; a well-formed document
    // holding an unacceptable value is an invalid VALUE. Bindings arrive here as
    // a JSON document, so both failures are reachable and they are not the same
    // mistake — one blames the transport, the other the caller's value. The
    // parser used to stamp `InvalidJson` on every failure, which also gave the TS
    // engine (which takes JS objects and parses no JSON at all) a code it could
    // only mirror dishonestly.
    // -----------------------------------------------------------------------

    fn code_of(json: &str) -> ErrorCode {
        params_from_json(json).expect_err("should have failed").code
    }

    #[test]
    fn malformed_json_is_invalid_json() {
        for bad in [
            r#"{"p": }"#,
            r#"{"p": tru}"#,
            r#"{"p" "x"}"#,
            r#"{"p": "unterminated"#,
            r#"{"p": 1,}"#,
            "{",
            r#"{"p": "\ud800"}"#,
        ] {
            assert_eq!(code_of(bad), ErrorCode::InvalidJson, "for {bad}");
        }
    }

    #[test]
    fn a_nested_array_is_an_invalid_value_not_invalid_json() {
        // `[[1]]` parses as JSON perfectly well; it just isn't a binding.
        assert_eq!(code_of(r#"{"p": [[1]]}"#), ErrorCode::InvalidValue);
    }

    #[test]
    fn a_non_temporal_object_is_an_invalid_value() {
        // Both of these are well-formed JSON — the value shape is what is wrong,
        // and the string-valued case must not be reported differently from the
        // number-valued one just because the parser reaches it sooner.
        assert_eq!(code_of(r#"{"p": {"a":1}}"#), ErrorCode::InvalidValue);
        assert_eq!(code_of(r#"{"p": {"a":"x"}}"#), ErrorCode::InvalidValue);
    }

    #[test]
    fn a_bad_temporal_string_is_an_invalid_value() {
        assert_eq!(
            code_of(r#"{"p": {"@date":"nope"}}"#),
            ErrorCode::InvalidValue
        );
        assert_eq!(
            code_of(r#"{"p": {"@duration":"nope"}}"#),
            ErrorCode::InvalidValue
        );
    }

    #[test]
    fn a_tagged_temporal_with_a_non_string_value_is_an_invalid_value() {
        assert_eq!(code_of(r#"{"p": {"@date":1}}"#), ErrorCode::InvalidValue);
        assert_eq!(code_of(r#"{"p": {"@date":null}}"#), ErrorCode::InvalidValue);
    }

    #[test]
    fn a_well_formed_tagged_temporal_still_binds() {
        let p = params_from_json(r#"{"p": {"@date":"2020-01-01"}}"#).expect("should bind");

        assert!(matches!(p["p"], Val::Temporal(_)));
    }
}
