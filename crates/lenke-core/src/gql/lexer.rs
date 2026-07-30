//! Hand-written lexer for the GQL subset — a port of TS `lexer.ts`. Turns query
//! text into a flat token stream. Multi-char operators are matched greedily
//! (`<->` beats `<-` beats `<`); keywords are case-insensitive; identifiers and
//! string literals keep their original case.

use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tt {
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Dot,
    Comma,
    Pipe,    // |
    Amp,     // &
    Bang,    // !
    Percent, // %
    Plus,
    Star,
    Slash,
    Concat, // ||
    Dash,
    RArrow,  // ->
    LArrow,  // <-
    Tilde,   // ~
    LTilde,  // <~
    TildeR,  // ~>
    LRArrow, // <->
    Eq,
    Neq, // <>
    Lt,
    Gt,
    Lte,
    Gte,
    Number,
    Str,
    Param, // $name
    Ident,
    Keyword,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tt: Tt,
    /// Source text (identifiers/strings) or the lowercased keyword.
    pub value: String,
    /// The verbatim source text with original casing. Only set for `Keyword`
    /// tokens (whose `value` is lowercased for structural matching) — the parser
    /// needs it to echo the user's exact spelling when suggesting the
    /// backtick-delimited form of a reserved word (`` `Order` ``, not `` `order` ``).
    pub raw: Option<String>,
    /// Set only for `Number` tokens.
    pub num: Option<f64>,
    /// True for a backtick-delimited identifier — may be any word, even reserved.
    pub delimited: bool,
    /// Zero-based source offset, for error messages.
    pub pos: usize,
}

/// A lex/parse error carrying the source offset.
#[derive(Debug, Clone)]
pub struct SyntaxError {
    pub message: String,
    pub pos: usize,
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at position {})", self.message, self.pos)
    }
}

pub fn err<T>(message: impl Into<String>, pos: usize) -> Result<T, SyntaxError> {
    Err(SyntaxError {
        message: message.into(),
        pos,
    })
}

const KEYWORDS: &[&str] = &[
    "match",
    "optional",
    "with",
    "where",
    "let",
    "filter",
    "next",
    "call",
    "yield",
    "insert",
    "set",
    "remove",
    "delete",
    "detach",
    "nodetach",
    "finish",
    "return",
    "for",
    "as",
    "is",
    "in",
    "and",
    "or",
    "xor",
    "not",
    "distinct",
    "all",
    "any",
    "shortest",
    "case",
    "when",
    "then",
    "else",
    "end",
    "exists",
    "count",
    "nulls",
    "unknown",
    "limit",
    "union",
    "except",
    "intersect",
    "order",
    "by",
    "asc",
    "ascending",
    "desc",
    "descending",
    "skip",
    "offset",
    "true",
    "false",
    "null",
];

fn keywords() -> &'static HashSet<&'static str> {
    static K: OnceLock<HashSet<&'static str>> = OnceLock::new();
    K.get_or_init(|| KEYWORDS.iter().copied().collect())
}

// The complete ISO/IEC 39075 reserved-word list (verbatim from the TS port).
const RESERVED_WORDS: &str =
    "abs acos all all_different and any array as asc ascending asin at atan atan2 avg big bigint \
binary bool boolean both btrim by byte_length bytes call cardinality case cast ceil ceiling \
char char_length character_length characteristics close coalesce collect_list commit copy cos \
cosh cot count create current_date current_graph current_property_graph current_schema \
current_time current_timestamp date datetime day dec decimal degrees delete desc descending \
detach distinct double drop duration duration_between element_id else end except exists exp \
false filter finish float float16 float32 float64 float128 float256 floor for from group having \
home_graph home_property_graph home_schema hour if implies in insert int integer int8 integer8 \
int16 integer16 int32 integer32 int64 integer64 int128 integer128 int256 integer256 intersect \
interval is leading left let like limit list ln local local_datetime local_time \
local_timestamp log log10 lower ltrim match max min minute mod month next nodetach normalize \
not nothing null nulls nullif octet_length of offset optional or order otherwise parameter \
parameters path path_length paths percentile_cont percentile_disc power precision \
property_exists radians real record remove replace reset return right rollback rtrim same \
schema second select session session_user set signed sin sinh size skip small smallint sqrt \
start stddev_pop stddev_samp string sum tan tanh then time timestamp trailing trim true typed \
ubigint uint uint8 uint16 uint32 uint64 uint128 uint256 union unknown unsigned upper use \
usmallint value varbinary varchar variable when where with xor year yield zoned zoned_datetime \
zoned_time \
abstract aggregate aggregates alter catalog clear clone constraint current_role current_user \
data directory dryrun exact existing function gqlstatus grant instant infinity number numeric \
on open partition procedure product project query records reference rename revoke substring \
system_user temporal unique unit values whitespace";

fn reserved() -> &'static HashSet<&'static str> {
    static R: OnceLock<HashSet<&'static str>> = OnceLock::new();
    R.get_or_init(|| RESERVED_WORDS.split(' ').collect())
}

/// Is `word` (case-insensitive) an ISO reserved word, hence not a bare identifier?
pub fn is_reserved(word: &str) -> bool {
    reserved().contains(word.to_ascii_lowercase().as_str())
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}
fn is_ident_part(c: char) -> bool {
    is_ident_start(c) || c.is_ascii_digit()
}

/// Decode the backslash escape beginning at byte `i` (the backslash). Returns the
/// decoded text and the byte index just past the escape.
fn read_escape(b: &[u8], src: &str, i: usize) -> Result<(String, usize), SyntaxError> {
    let esc = b[i + 1] as char;
    let simple = match esc {
        '\\' => Some('\\'),
        '\'' => Some('\''),
        '"' => Some('"'),
        't' => Some('\t'),
        'n' => Some('\n'),
        'r' => Some('\r'),
        'b' => Some('\u{0008}'),
        'f' => Some('\u{000C}'),
        _ => None,
    };
    if let Some(ch) = simple {
        return Ok((ch.to_string(), i + 2));
    }
    if esc == 'u' || esc == 'U' {
        let width = if esc == 'u' { 4 } else { 6 };
        let end = i + 2 + width;
        // `.get` (not `&src[..]`) so an escape whose window runs past the end OR
        // crosses a UTF-8 char boundary (e.g. `\u123é`) yields the escape error
        // rather than panicking — which under panic="abort" would abort the host.
        let Some(hex) = src.get(i + 2..end) else {
            return err(
                format!("Invalid \\{esc} escape (expected {width} hex digits)"),
                i,
            );
        };
        match u32::from_str_radix(hex, 16).ok().and_then(char::from_u32) {
            Some(ch) => Ok((ch.to_string(), end)),
            None => err(
                format!("Invalid \\{esc} escape (expected {width} hex digits)"),
                i,
            ),
        }
    } else {
        Ok((esc.to_string(), i + 2))
    }
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, SyntaxError> {
    let b = src.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    let push = |tokens: &mut Vec<Token>, tt: Tt, value: &str, pos: usize| {
        tokens.push(Token {
            tt,
            value: value.to_string(),
            raw: None,
            num: None,
            delimited: false,
            pos,
        });
    };

    while i < b.len() {
        let c = b[i] as char;

        if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
            i += 1;
            continue;
        }

        // `.get` (not `&src[..]`) so a slice that would cross a UTF-8 char
        // boundary — e.g. `i` sits just before a multi-byte char — yields None
        // rather than panicking (a `'😀'` literal used to crash the lexer here).
        let two = src.get(i..i + 2).unwrap_or("");

        // Comments: `//` and `--` line, `/* */` block. Note `--` is a comment,
        // NOT an undirected edge — the main divergence from Cypher.
        if two == "//" || two == "--" {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if two == "/*" {
            i += 2;
            while i < b.len() && src.get(i..(i + 2).min(b.len())) != Some("*/") {
                i += 1;
            }
            if i >= b.len() {
                return err("Unterminated block comment", i);
            }
            i += 2;
            continue;
        }

        // Three-char operator (greedy): `<->`.
        if src.get(i..i + 3) == Some("<->") {
            push(&mut tokens, Tt::LRArrow, "<->", i);
            i += 3;
            continue;
        }

        // Two-char operators.
        let two_tt = match two {
            "->" => Some(Tt::RArrow),
            "<-" => Some(Tt::LArrow),
            "<~" => Some(Tt::LTilde),
            "~>" => Some(Tt::TildeR),
            "<>" => Some(Tt::Neq),
            "<=" => Some(Tt::Lte),
            ">=" => Some(Tt::Gte),
            "||" => Some(Tt::Concat),
            _ => None,
        };
        if let Some(tt) = two_tt {
            push(&mut tokens, tt, two, i);
            i += 2;
            continue;
        }

        // A `.` immediately followed by a digit is a leading-dot float (`.5`).
        let dot_number = c == '.' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit();
        if !dot_number {
            let single = match c {
                '(' => Some(Tt::LParen),
                ')' => Some(Tt::RParen),
                '[' => Some(Tt::LBracket),
                ']' => Some(Tt::RBracket),
                '{' => Some(Tt::LBrace),
                '}' => Some(Tt::RBrace),
                ':' => Some(Tt::Colon),
                '.' => Some(Tt::Dot),
                ',' => Some(Tt::Comma),
                '|' => Some(Tt::Pipe),
                '&' => Some(Tt::Amp),
                '!' => Some(Tt::Bang),
                '%' => Some(Tt::Percent),
                '+' => Some(Tt::Plus),
                '*' => Some(Tt::Star),
                '/' => Some(Tt::Slash),
                '-' => Some(Tt::Dash),
                '~' => Some(Tt::Tilde),
                '=' => Some(Tt::Eq),
                '<' => Some(Tt::Lt),
                '>' => Some(Tt::Gt),
                _ => None,
            };
            if let Some(tt) = single {
                push(&mut tokens, tt, &c.to_string(), i);
                i += 1;
                continue;
            }
        }

        // String literals: single or double quoted.
        if c == '\'' || c == '"' {
            let start = i;
            let quote = b[i];
            i += 1;
            let mut s = String::new();
            while i < b.len() && b[i] != quote {
                if b[i] == b'\\' && i + 1 < b.len() {
                    let (text, next) = read_escape(b, src, i)?;
                    s.push_str(&text);
                    i = next;
                    continue;
                }
                // copy one UTF-8 char
                let ch = src[i..].chars().next().unwrap();
                s.push(ch);
                i += ch.len_utf8();
            }
            if i >= b.len() {
                return err("Unterminated string literal", start);
            }
            i += 1; // closing quote
            tokens.push(Token {
                tt: Tt::Str,
                value: s,
                raw: None,
                num: None,
                delimited: false,
                pos: start,
            });
            continue;
        }

        // Delimited identifier: backtick — keeps exact spelling, never a keyword.
        // A backtick inside is written doubled (`` `a``b` `` → a`b), the ISO/SQL
        // delimiter-escape convention, so any string can round-trip.
        if c == '`' {
            let start = i;
            i += 1;
            let mut name = String::new();
            loop {
                if i >= b.len() {
                    return err("Unterminated delimited identifier", start);
                }
                if b[i] == b'`' {
                    if b.get(i + 1) == Some(&b'`') {
                        name.push('`'); // escaped backtick (doubled)
                        i += 2;
                        continue;
                    }
                    break; // a lone backtick closes the identifier
                }
                let ch = src[i..].chars().next().unwrap();
                name.push(ch);
                i += ch.len_utf8();
            }
            i += 1;
            tokens.push(Token {
                tt: Tt::Ident,
                value: name,
                raw: None,
                num: None,
                delimited: true,
                pos: start,
            });
            continue;
        }

        // Parameter: `$name`.
        if c == '$' {
            let start = i;
            i += 1;
            let name_start = i;
            while i < b.len() && is_ident_part(b[i] as char) {
                i += 1;
            }
            if i == name_start {
                return err("Expected a parameter name after `$`", start);
            }
            tokens.push(Token {
                tt: Tt::Param,
                value: src[name_start..i].to_string(),
                raw: None,
                num: None,
                delimited: false,
                pos: start,
            });
            continue;
        }

        // Numbers: decimal (fraction/exponent/underscores) and 0x/0o/0b bases.
        if c.is_ascii_digit() || dot_number {
            let start = i;
            let is_base = c == '0'
                && i + 1 < b.len()
                && matches!(b[i + 1], b'x' | b'X' | b'o' | b'O' | b'b' | b'B');
            // Only integer forms (no fraction/exponent) are held to the
            // safe-integer range; floats may exceed it (they're approximate).
            let mut is_integer = true;
            if is_base {
                // Each base admits only its own digits. Sharing one hex class let
                // `0b1019AF` / `0o789` lex as a token that then collapsed to NaN.
                let base = (b[i + 1] as char).to_ascii_lowercase();
                i += 2;
                let digits_start = i;
                while i < b.len() && is_base_digit(b[i], base) {
                    i += 1;
                }
                if i == digits_start {
                    return err(
                        format!("Malformed numeric literal '{}'", &src[start..i]),
                        start,
                    );
                }
            } else {
                let digits = |i: &mut usize| {
                    while *i < b.len() && ((b[*i] as char).is_ascii_digit() || b[*i] == b'_') {
                        *i += 1;
                    }
                };
                digits(&mut i);
                if i < b.len() && b[i] == b'.' {
                    is_integer = false;
                    i += 1;
                    digits(&mut i);
                }
                if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                    is_integer = false;
                    i += 1;
                    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                        i += 1;
                    }
                    digits(&mut i);
                }
            }
            let text = &src[start..i];
            let cleaned: String = text.chars().filter(|&ch| ch != '_').collect();
            // Reject a malformed mantissa/base (→ NaN) and overflow (→ Infinity);
            // otherwise a garbage literal would flow into the AST silently.
            let num = match parse_number(&cleaned) {
                Some(n) if n.is_finite() => n,
                _ => return err(format!("Malformed numeric literal '{text}'"), start),
            };
            // An integer literal past 2^53 loses precision as an f64 — reject it
            // rather than carry a value that differs from what was written.
            if is_integer && num.abs() > MAX_SAFE_INTEGER {
                return err(
                    format!("Integer literal '{text}' exceeds the safe integer range"),
                    start,
                );
            }
            tokens.push(Token {
                tt: Tt::Number,
                value: text.to_string(),
                raw: None,
                num: Some(num),
                delimited: false,
                pos: start,
            });
            continue;
        }

        // Identifiers and keywords.
        if is_ident_start(c) {
            let start = i;
            while i < b.len() && is_ident_part(b[i] as char) {
                i += 1;
            }
            let text = &src[start..i];
            let lower = text.to_ascii_lowercase();
            if keywords().contains(lower.as_str()) {
                // Keep `value` lowercased for structural dispatch, but carry the
                // original casing in `raw` so a reserved-word rejection can echo
                // the user's exact spelling as a delimited identifier.
                tokens.push(Token {
                    tt: Tt::Keyword,
                    value: lower,
                    raw: Some(text.to_string()),
                    num: None,
                    delimited: false,
                    pos: start,
                });
            } else {
                push(&mut tokens, Tt::Ident, text, start);
            }
            continue;
        }

        return err(format!("Unexpected character '{c}'"), i);
    }

    push(&mut tokens, Tt::Eof, "", src.len());
    Ok(tokens)
}

/// 2^53 — the largest integer an `f64` represents exactly.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

/// Is `byte` a valid digit (or `_` separator) for integer `base` (`x`/`o`/`b`)?
fn is_base_digit(byte: u8, base: char) -> bool {
    if byte == b'_' {
        return true;
    }
    match base {
        'x' => byte.is_ascii_hexdigit(),
        'o' => (b'0'..=b'7').contains(&byte),
        _ => byte == b'0' || byte == b'1',
    }
}

/// Parse a numeric literal, honoring the `0x`/`0o`/`0b` bases JS `Number()`
/// accepts. Returns `None` on a malformed/overflowing literal so the caller can
/// raise a lex error instead of silently producing `NaN`.
fn parse_number(s: &str) -> Option<f64> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return u64::from_str_radix(hex, 16).ok().map(|n| n as f64);
    }
    if let Some(oct) = s.strip_prefix("0o").or_else(|| s.strip_prefix("0O")) {
        return u64::from_str_radix(oct, 8).ok().map(|n| n as f64);
    }
    if let Some(bin) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        return u64::from_str_radix(bin, 2).ok().map(|n| n as f64);
    }
    s.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::tokenize;

    #[test]
    fn malformed_unicode_escape_errors_not_panics() {
        // A `\u` escape whose hex window crosses a multi-byte char (`é` is two
        // bytes) must error, not panic on a non-char-boundary slice — which under
        // panic="abort" would abort the host.
        assert!(tokenize("RETURN \"\\u123é\"").is_err());
        assert!(tokenize("RETURN \"\\U12345é\"").is_err());
        // A well-formed escape still tokenizes.
        assert!(tokenize("RETURN \"\\u0041\"").is_ok());
    }
}
