//! A parser for the textual **Gremlin** language (the Groovy-style wire form,
//! e.g. `g.V().has('name','marko').out('knows').values('name')`) into a
//! [`Traversal`] plan. This is the string carrier the FFI needs — the JS side
//! ships a query string and Rust parses + executes it, exactly as the GQL FFI
//! does for GQL text.
//!
//! Grammar (recursive descent): a traversal is a source (`g` / `__` / a bare
//! step) followed by a `.method(args)` chain. An argument is a literal, a
//! predicate (`gt(30)`, `within('a','b')`), a token (`T.label`, `Order.desc`,
//! `Pop.first`, `Scope.local`), or a nested traversal (`__.out().count()`).

use super::{Column, GVal, Order, Pop, Step, Token, Traversal, __, P};
use crate::temporal::{Date, DateTime, Duration, Temporal, Time, ZonedDateTime, ZonedTime};

// --- lexer ------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Num(f64),
    Dot,
    LParen,
    RParen,
    Comma,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            _ if c.is_ascii_whitespace() => i += 1,
            b'.' => {
                // A dot starting a number? Only if followed by a digit and not part of a chain.
                out.push(Tok::Dot);
                i += 1;
            }
            b'(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            b',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            b'\'' | b'"' => {
                let quote = c as char;
                let start = i;
                i += 1;
                let mut s = String::new();
                let mut terminated = false;
                // Iterate by `char`, not byte, so multi-byte UTF-8 literals
                // survive intact; decode the common escapes (`\n` etc.) rather
                // than dropping the backslash and keeping the bare letter.
                while i < b.len() {
                    let ch = src[i..].chars().next().unwrap();
                    if ch == quote {
                        i += ch.len_utf8();
                        terminated = true;
                        break;
                    }
                    if ch == '\\' {
                        i += 1; // past the backslash (ASCII, 1 byte)
                        let Some(esc) = src[i..].chars().next() else {
                            break;
                        };
                        s.push(match esc {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            other => other,
                        });
                        i += esc.len_utf8();
                        continue;
                    }
                    s.push(ch);
                    i += ch.len_utf8();
                }
                if !terminated {
                    return Err(format!("unterminated string at {start}"));
                }
                out.push(Tok::Str(s));
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                if c == b'-' {
                    i += 1;
                }
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                    i += 1;
                }
                let text = &src[start..i];
                let n: f64 = text.parse().map_err(|_| format!("bad number `{text}`"))?;
                out.push(Tok::Num(n));
            }
            _ if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(Tok::Ident(src[start..i].to_string()));
            }
            _ => return Err(format!("unexpected char `{}`", c as char)),
        }
    }
    Ok(out)
}

// --- parser -----------------------------------------------------------------

const PREDS: &[&str] = &[
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "inside",
    "outside",
    "within",
    "without",
    "startingWith",
    "startsWith",
    "endingWith",
    "containing",
    "notContaining",
    "regex",
];

/// A parsed argument before it's bound to a specific step.
enum Arg {
    Str(String),
    Num(f64),
    Bool(bool),
    /// A temporal literal: `date('2020-01-01')`, `datetime('…')`, `duration('P1D')`,
    /// `time('…')`, `zoned_time('…')`, `zoned_datetime('…')`. The executor has
    /// carried `GVal::Temporal` all along and reads stored temporals fine — only
    /// the text grammar could not *write* one, so an as-of predicate
    /// (`has('vf', lte(date('2021-06-01')))`) was inexpressible in this dialect
    /// and a bitemporal query could not be written at all.
    Temporal(Temporal),
    Pred(P),
    Trav(Traversal),
    Order(Order),
    Pop(Pop),
    Token(Token),
    Column(Column),
    SackOp(super::SackOp),
}

impl Arg {
    fn as_gval(&self) -> Result<GVal, String> {
        match self {
            Self::Str(s) => Ok(GVal::Str(s.as_str().into())),
            Self::Num(n) => Ok(GVal::Num(*n)),
            Self::Bool(b) => Ok(GVal::Bool(*b)),
            Self::Temporal(t) => Ok(GVal::Temporal(*t)),
            _ => Err("expected a literal value".into()),
        }
    }
    fn as_str(&self) -> Result<&str, String> {
        match self {
            Self::Str(s) => Ok(s),
            _ => Err("expected a string".into()),
        }
    }
}

/// Names accepted as temporal literal constructors in the text dialect.
const TEMPORAL_CTORS: &[&str] = &[
    "date",
    "datetime",
    "time",
    "duration",
    "zoned_time",
    "zoned_datetime",
];

/// Build a [`Temporal`] from a constructor name and its string argument, reusing
/// the per-kind parsers so the accepted syntax matches GQL exactly.
fn parse_temporal_literal(ctor: &str, lit: &str) -> Result<Temporal, String> {
    match ctor {
        "date" => Date::parse(lit).map(Temporal::Date),
        "datetime" => DateTime::parse(lit).map(Temporal::DateTime),
        "time" => Time::parse(lit).map(Temporal::Time),
        "duration" => Duration::parse(lit).map(Temporal::Duration),
        "zoned_time" => ZonedTime::parse(lit).map(Temporal::ZonedTime),
        "zoned_datetime" => ZonedDateTime::parse(lit).map(Temporal::ZonedDateTime),
        other => Err(format!("unknown temporal constructor '{other}'")),
    }
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    /// Recursive-descent nesting depth (every nested sub-traversal / predicate
    /// passes through `arg`, so guarding it there bounds the whole recursion).
    depth: usize,
}

/// Nesting ceiling. Unbounded recursion on deeply nested input
/// (`repeat(repeat(repeat(…)))`, `union(union(…))`, `not(not(…))`) would
/// otherwise overflow the native stack and *abort the process* (uncatchable).
const MAX_DEPTH: usize = 256;

/// Validate a numeric step argument as a non-negative integer count. Rust's
/// `f64 as usize` cast saturates silently (`-5 → 0`, `1e30 → usize::MAX`), so a
/// raw cast would accept negatives/floats/huge values; this rejects them.
fn as_count(n: f64, step: &str) -> Result<usize, String> {
    if !n.is_finite() || n < 0.0 || n.fract() != 0.0 {
        return Err(format!("{step}: expected a non-negative integer, got {n}"));
    }
    Ok(n as usize)
}

/// The `i`-th numeric argument as a validated count, or a clean error if absent
/// (a missing positional arg used to panic via `nums[i]` indexing).
fn count_at(nums: &[f64], i: usize, step: &str) -> Result<usize, String> {
    match nums.get(i) {
        Some(n) => as_count(*n, step),
        None => Err(format!("{step}: missing numeric argument")),
    }
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn peek_at(&self, n: usize) -> Option<&Tok> {
        self.toks.get(self.pos + n)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }
    fn expect(&mut self, t: &Tok) -> Result<(), String> {
        if self.peek() == Some(t) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {t:?}, found {:?}", self.peek()))
        }
    }

    /// Parse a full traversal: optional `g`/`__` source then a `.method(...)` chain.
    fn traversal(&mut self) -> Result<Traversal, String> {
        let mut t = __();
        // Pre-form emit()/until() modulators (before their repeat) stash here,
        // local to THIS traversal so a nested repeat body has its own.
        let mut pending = PendingMods::default();
        match self.peek() {
            Some(Tok::Ident(id)) if id == "g" || id == "__" => {
                self.pos += 1;
            }
            Some(Tok::Ident(_)) => {
                // Bare anonymous step (e.g. `out('knows')`): parse the first method
                // directly, no leading dot.
                let name = self.ident()?;
                t = self.method(t, &name, &mut pending)?;
            }
            other => return Err(format!("expected a traversal, found {other:?}")),
        }
        while self.peek() == Some(&Tok::Dot) {
            self.pos += 1;
            let name = self.ident()?;
            t = self.method(t, &name, &mut pending)?;
        }
        Ok(t)
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(format!("expected an identifier, found {other:?}")),
        }
    }

    /// The string argument of a literal constructor, e.g. the `'2020-01-01'` in
    /// `date('2020-01-01')`.
    fn string(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Tok::Str(s)) => Ok(s),
            other => Err(format!("expected a string literal, found {other:?}")),
        }
    }

    /// Parse `( arg, arg, ... )`.
    fn args(&mut self) -> Result<Vec<Arg>, String> {
        self.expect(&Tok::LParen)?;
        let mut args = Vec::new();
        if self.peek() == Some(&Tok::RParen) {
            self.pos += 1;
            return Ok(args);
        }
        loop {
            args.push(self.arg()?);
            match self.next() {
                Some(Tok::Comma) => continue,
                Some(Tok::RParen) => break,
                other => return Err(format!("expected `,` or `)`, found {other:?}")),
            }
        }
        Ok(args)
    }

    fn arg(&mut self) -> Result<Arg, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            self.depth -= 1;
            return Err("query nested too deeply".to_string());
        }
        let r = self.arg_inner();
        self.depth -= 1;
        r
    }

    fn arg_inner(&mut self) -> Result<Arg, String> {
        match self.peek() {
            Some(Tok::Str(_)) => match self.next() {
                Some(Tok::Str(s)) => Ok(Arg::Str(s)),
                _ => unreachable!(),
            },
            Some(Tok::Num(_)) => match self.next() {
                Some(Tok::Num(n)) => Ok(Arg::Num(n)),
                _ => unreachable!(),
            },
            Some(Tok::Ident(id)) => {
                let id = id.clone();
                if id == "true" {
                    self.pos += 1;
                    return Ok(Arg::Bool(true));
                }
                if id == "false" {
                    self.pos += 1;
                    return Ok(Arg::Bool(false));
                }
                // Temporal literal constructors — the text analogue of GQL's
                // `DATE '2020-01-01'`. Parsed with the same `Temporal` parsers the
                // GQL lexer uses, so both frontends accept identical spellings and
                // reject identical ones.
                if TEMPORAL_CTORS.contains(&id.as_str()) && self.peek_at(1) == Some(&Tok::LParen) {
                    self.pos += 2; // ident + '('
                    let lit = self.string()?;
                    self.expect(&Tok::RParen)?;

                    return Ok(Arg::Temporal(parse_temporal_literal(&id, &lit)?));
                }
                // Bare tokens (static-import style: `desc`, `local`, `first`, …),
                // i.e. an enum member not used as a call.
                if self.peek_at(1) != Some(&Tok::LParen) {
                    if let Some(a) = bare_token(&id) {
                        self.pos += 1;
                        return Ok(a);
                    }
                }
                // Token namespaces: T.id, Order.desc, Pop.first, Scope.local.
                if matches!(
                    id.as_str(),
                    "T" | "Order"
                        | "Pop"
                        | "Scope"
                        | "Column"
                        | "Direction"
                        | "TextP"
                        | "P"
                        | "ShortestPath"
                        | "PageRank"
                        | "ConnectedComponent"
                        | "PeerPressure"
                        | "__"
                ) && self.peek_at(1) == Some(&Tok::Dot)
                {
                    if id == "TextP" || id == "P" {
                        // TextP.containing(...) / P.gt(...) — skip the namespace, parse the call.
                        self.pos += 2; // ident + dot
                        let name = self.ident()?;
                        return Ok(Arg::Pred(self.predicate(&name)?));
                    }
                    if id == "__" {
                        // __.out()... anonymous traversal
                        return Ok(Arg::Trav(self.traversal()?));
                    }
                    self.pos += 2; // namespace ident + dot
                    let member = self.ident()?;
                    return self.token(&id, &member);
                }
                // A predicate call?
                if PREDS.contains(&id.as_str()) && self.peek_at(1) == Some(&Tok::LParen) {
                    self.pos += 1;
                    return Ok(Arg::Pred(self.predicate(&id)?));
                }
                if id == "not" && self.peek_at(1) == Some(&Tok::LParen) {
                    // `not(...)` — predicate if the inner is a predicate/literal,
                    // else a `not` traversal step.
                    if self.inner_is_predicate() {
                        self.pos += 2; // not (
                        let inner = self.arg()?;
                        self.expect(&Tok::RParen)?;
                        return match inner {
                            Arg::Pred(p) => Ok(Arg::Pred(P::not(p))),
                            other => Ok(Arg::Pred(P::not(P::eq(other.as_gval()?)))),
                        };
                    }
                    return Ok(Arg::Trav(self.traversal()?));
                }
                // Otherwise: a nested traversal starting at this step (bare or via chain).
                Ok(Arg::Trav(self.traversal()?))
            }
            other => Err(format!("unexpected argument {other:?}")),
        }
    }

    /// Lookahead: is the token after `not(` a predicate or literal (vs a step)?
    fn inner_is_predicate(&self) -> bool {
        match self.peek_at(2) {
            Some(Tok::Str(_)) | Some(Tok::Num(_)) => true,
            Some(Tok::Ident(id)) => PREDS.contains(&id.as_str()) || id == "true" || id == "false",
            _ => false,
        }
    }

    fn token(&mut self, ns: &str, member: &str) -> Result<Arg, String> {
        Ok(match (ns, member) {
            ("T", "id") => Arg::Token(Token::Id),
            ("T", "label") => Arg::Token(Token::Label),
            ("T", "key") => Arg::Token(Token::Key),
            ("T", "value") => Arg::Token(Token::Value),
            ("Column", "keys") => Arg::Column(Column::Keys),
            ("Column", "values") => Arg::Column(Column::Values),
            ("Direction", "OUT" | "IN" | "BOTH") => Arg::Str(format!("Direction.{member}")),
            ("Order", "asc" | "incr") => Arg::Order(Order::Asc),
            ("Order", "desc" | "decr") => Arg::Order(Order::Desc),
            ("Pop", "first") => Arg::Pop(Pop::First),
            ("Pop", "last") => Arg::Pop(Pop::Last),
            ("Pop", "all") => Arg::Pop(Pop::All),
            ("Operator", "assign") => Arg::SackOp(super::SackOp::Assign),
            ("Operator", "sum") => Arg::SackOp(super::SackOp::Sum),
            ("Operator", "mult") => Arg::SackOp(super::SackOp::Mult),
            ("Operator", "min") => Arg::SackOp(super::SackOp::Min),
            ("Operator", "max") => Arg::SackOp(super::SackOp::Max),
            ("Scope", _) => Arg::Str(format!("Scope.{member}")), // consumed by scope-aware steps
            ("ShortestPath", _) => Arg::Str(format!("ShortestPath.{member}")), // consumed by with()
            ("PageRank", _) => Arg::Str(format!("PageRank.{member}")), // consumed by with()
            ("ConnectedComponent", _) => Arg::Str(format!("ConnectedComponent.{member}")),
            ("PeerPressure", _) => Arg::Str(format!("PeerPressure.{member}")),
            _ => return Err(format!("unknown token {ns}.{member}")),
        })
    }

    /// Parse a predicate call `name(args)` (the leading ident already consumed).
    fn predicate(&mut self, name: &str) -> Result<P, String> {
        let args = self.args()?;
        let g = |i: usize| {
            args.get(i)
                .ok_or_else(|| format!("{name}: missing arg"))
                .and_then(Arg::as_gval)
        };
        Ok(match name {
            "eq" => P::Eq(g(0)?),
            "neq" => P::Neq(g(0)?),
            "gt" => P::Gt(g(0)?),
            "gte" => P::Gte(g(0)?),
            "lt" => P::Lt(g(0)?),
            "lte" => P::Lte(g(0)?),
            "between" => P::Between(g(0)?, g(1)?),
            "inside" => P::Inside(g(0)?, g(1)?),
            "outside" => P::Outside(g(0)?, g(1)?),
            "within" => P::Within(args.iter().map(Arg::as_gval).collect::<Result<_, _>>()?),
            "without" => P::Without(args.iter().map(Arg::as_gval).collect::<Result<_, _>>()?),
            "startingWith" | "startsWith" => P::StartsWith(
                args.first()
                    .ok_or("startsWith: missing")?
                    .as_str()?
                    .to_string(),
            ),
            "endingWith" => P::EndingWith(
                args.first()
                    .ok_or("endingWith: missing")?
                    .as_str()?
                    .to_string(),
            ),
            "containing" => P::Containing(
                args.first()
                    .ok_or("containing: missing")?
                    .as_str()?
                    .to_string(),
            ),
            "notContaining" => P::NotContaining(
                args.first()
                    .ok_or("notContaining: missing")?
                    .as_str()?
                    .to_string(),
            ),
            "regex" => {
                let pat = args.first().ok_or("regex: missing")?.as_str()?;
                // Validate up front (like the TS `regex()` constructor) so an
                // invalid pattern is a clean parse error, not a per-value fault.
                regex::Regex::new(pat).map_err(|e| format!("regex: invalid pattern: {e}"))?;
                P::Regex(pat.to_string())
            }
            _ => return Err(format!("unknown predicate `{name}`")),
        })
    }

    /// Dispatch a `.method(args)` onto the traversal `t`. `pending` carries
    /// pre-form repeat modulators (see [`PendingMods`]).
    fn method(
        &mut self,
        t: Traversal,
        name: &str,
        pending: &mut PendingMods,
    ) -> Result<Traversal, String> {
        let args = self.args()?;
        // scope helper: a leading Scope.local arg means the *_local variant.
        let scope_local = |a: &[Arg]| matches!(a.first(), Some(Arg::Str(s)) if s == "Scope.local");
        let nums_after_scope = |a: &[Arg]| -> Vec<f64> {
            a.iter()
                .filter_map(|x| match x {
                    Arg::Num(n) => Some(*n),
                    _ => None,
                })
                .collect()
        };

        let str_labels: Vec<&str> = args
            .iter()
            .filter_map(|a| {
                if let Arg::Str(s) = a {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();

        // Repeat modulators may appear BEFORE their repeat in TinkerPop
        // (`emit().repeat(body)` / `until().repeat(body)`). Our builder attaches a
        // modulator to the LAST step, so a pre-repeat one would be silently
        // dropped; stash it and let the next `repeat` consume it (emit → pre-form
        // via `emit_before`; until is while-semantics either side). This is also
        // where `repeat` itself is built, so it can pull in any stashed modulator.
        // Post-form (`repeat(body).emit()`) still flows through the match below.
        // See Priyanka r4: `emit().repeat(out('MEMBER_OF'))` needs the zero-hop start.
        let ends_in_repeat = matches!(t.steps.last(), Some(Step::Repeat { .. }));
        match name {
            "emit" if !ends_in_repeat => {
                pending.emit = Some(if args.is_empty() {
                    __()
                } else {
                    self.one_trav(args)?
                });
                return Ok(t);
            }
            "until" if !ends_in_repeat => {
                pending.until = Some(self.one_trav(args)?);
                return Ok(t);
            }
            "repeat" => {
                let mut r = t.repeat(self.one_trav(args)?);
                if let Some(e) = pending.emit.take() {
                    r = r.emit_before(e);
                }
                if let Some(u) = pending.until.take() {
                    // A pre-form `until(cond).repeat(body)` is while-do (checked
                    // BEFORE the body); the post-form `.until()` below is do-while.
                    r = r.until_before(u);
                }
                return Ok(r);
            }
            _ => {}
        }

        Ok(match name {
            // sources
            "V" if args.is_empty() => t.V(),
            "V" => t.v_ids(&str_labels),
            "E" if args.is_empty() => t.E(),
            "E" => t.e_ids(&str_labels),
            // movement
            "out" => t.out(&str_labels),
            "in" => t.in_(&str_labels),
            "both" => t.both(&str_labels),
            "outE" => t.out_e(&str_labels),
            "inE" => t.in_e(&str_labels),
            "bothE" => t.both_e(&str_labels),
            "outV" => t.out_v(),
            "inV" => t.in_v(),
            "otherV" => t.other_v(),
            "bothV" => t.both_v(),
            // filters
            "has" => match args.as_slice() {
                [k] => t.has_key(&[k.as_str()?]),
                [k, v] => match v {
                    Arg::Pred(p) => t.has(k.as_str()?, p.clone()),
                    other => t.has_val(k.as_str()?, other.as_gval()?),
                },
                [lbl, k, v] => match v {
                    Arg::Pred(p) => t.has_label_key(lbl.as_str()?, k.as_str()?, p.clone()),
                    other => t.has_label_key(lbl.as_str()?, k.as_str()?, P::eq(other.as_gval()?)),
                },
                _ => return Err("has: expected 1-3 args".into()),
            },
            "hasLabel" => t.has_label(&str_labels),
            "hasId" => t.has_id(&str_labels),
            "hasKey" => t.has_key(&str_labels),
            "hasNot" => t.has_not(&str_labels),
            "hasValue" => t.has_value(
                args.iter()
                    .map(Arg::as_gval)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            "is" => match &args[..] {
                [Arg::Pred(p)] => t.is(p.clone()),
                [other] => t.is(P::eq(other.as_gval()?)),
                _ => return Err("is: expected 1 arg".into()),
            },
            "dedup" => {
                // `dedup('a','b')` dedupes by the values tagged at those labels;
                // `dedup()`/`dedup().by(...)` by value/projection.
                let labels: Vec<String> = args
                    .iter()
                    .filter_map(|a| match a {
                        Arg::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                t.dedup_labels(labels)
            }
            "simplePath" => t.simple_path(),
            "cyclicPath" => t.cyclic_path(),
            // projection
            "values" => t.values(&str_labels),
            "valueMap" => t.value_map(&str_labels),
            "propertyMap" => t.property_map(&str_labels),
            "elementMap" => t.element_map(&str_labels),
            "properties" => t.properties(&str_labels),
            "value" => t.value(),
            "id" => t.id(),
            "label" => t.label(),
            "path" => t.path(),
            "project" => t.project(&str_labels),
            "tree" => t.tree(),
            // by / modulators
            "by" => self.bind_by(t, args)?,
            // cardinality
            "limit" => {
                let n = count_at(&nums_after_scope(&args), 0, "limit")?;
                if scope_local(&args) {
                    t.limit_local(n)
                } else {
                    t.limit(n)
                }
            }
            "skip" => {
                let n = count_at(&nums_after_scope(&args), 0, "skip")?;
                if scope_local(&args) {
                    t.skip_local(n)
                } else {
                    t.skip(n)
                }
            }
            "range" => {
                let ns = nums_after_scope(&args);
                let (lo, hi) = (count_at(&ns, 0, "range")?, count_at(&ns, 1, "range")?);
                if scope_local(&args) {
                    t.range_local(lo, hi)
                } else {
                    t.range(lo, hi)
                }
            }
            "tail" => {
                let n = match nums_after_scope(&args).first() {
                    Some(n) => as_count(*n, "tail")?,
                    None => 1,
                };
                if scope_local(&args) {
                    t.tail_local(n)
                } else {
                    t.tail(n)
                }
            }
            "sample" => t.sample(count_at(&nums_after_scope(&args), 0, "sample")?),
            // aggregates
            "count" => {
                if scope_local(&args) {
                    t.count_local()
                } else {
                    t.count()
                }
            }
            "sum" => {
                if scope_local(&args) {
                    t.sum_local()
                } else {
                    t.sum()
                }
            }
            "min" => {
                if scope_local(&args) {
                    t.min_local()
                } else {
                    t.min()
                }
            }
            "max" => {
                if scope_local(&args) {
                    t.max_local()
                } else {
                    t.max()
                }
            }
            "mean" => {
                if scope_local(&args) {
                    t.mean_local()
                } else {
                    t.mean()
                }
            }
            "fold" => t.fold(),
            "order" if scope_local(&args) => t.order_local(),
            "order" => t.order(),
            "group" => t.group(),
            "groupCount" => t.group_count(),
            // combinators
            "where" => match args.as_slice() {
                [Arg::Trav(sub)] => t.where_(sub.clone()),
                [Arg::Pred(p)] => t.where_pred(p.clone()),
                [Arg::Str(start), Arg::Pred(p)] => t.where_key(start, p.clone()),
                _ => {
                    return Err(
                        "where: expected (traversal), (predicate), or (key, predicate)".into(),
                    )
                }
            },
            "and" => t.and(self.travs(args)?),
            "or" => t.or(self.travs(args)?),
            "not" => t.not(self.one_trav(args)?),
            "union" => t.union(self.travs(args)?),
            "match" => t.match_(self.travs(args)?),
            "coalesce" => t.coalesce(self.travs(args)?),
            "optional" => t.optional(self.one_trav(args)?),
            "local" => t.local(self.one_trav(args)?),
            "map" => t.map(self.one_trav(args)?),
            "flatMap" => t.flat_map(self.one_trav(args)?),
            "filter" => t.filter(self.one_trav(args)?),
            "sideEffect" => t.side_effect(self.one_trav(args)?),
            "choose" => {
                let mut ts = self.travs(args)?;
                match ts.len() {
                    2 => t.choose(ts.remove(0), ts.remove(0)),
                    3 => t.choose_else(ts.remove(0), ts.remove(0), ts.remove(0)),
                    _ => return Err("choose: expected 2 or 3 traversals".into()),
                }
            }
            "branch" => t.branch(self.one_trav(args)?),
            "option" => self.bind_option(t, args)?,
            "aggregate" => t.aggregate(args.first().ok_or("aggregate: expected a key")?.as_str()?),
            "store" => t.store(args.first().ok_or("store: expected a key")?.as_str()?),
            "cap" => t.cap(args.first().ok_or("cap: expected a key")?.as_str()?),
            "subgraph" => t.subgraph(args.first().ok_or("subgraph: expected a name")?.as_str()?),
            "shortestPath" => t.shortest_path(),
            // Whole-graph OLAP algorithm steps (run locally). `pageRank(alpha?)`.
            "pageRank" => t.page_rank(match args.as_slice() {
                [] => None,
                [a] => match a.as_gval()? {
                    GVal::Num(n) => Some(n),
                    _ => return Err("pageRank(alpha): expected a number".into()),
                },
                _ => return Err("pageRank: expected an optional alpha".into()),
            }),
            "connectedComponent" => t.connected_component(),
            "peerPressure" => t.peer_pressure(),
            // `withComputer()` marks OLAP intent; lenke always computes in-process, so
            // it is accepted as a no-op (any args ignored).
            "withComputer" => t,
            "with" => match args.as_slice() {
                [Arg::Str(opt), Arg::Trav(target)] if opt == "ShortestPath.target" => {
                    t.with_shortest_path_target(target.clone())
                }
                // TinkerPop: `.with(ShortestPath.edges, Direction.OUT|IN|BOTH)` chooses
                // which incident edges the search follows (default BOTH = undirected).
                [Arg::Str(opt), rest] if opt == "ShortestPath.edges" => {
                    t.with_shortest_path_edges(rest.as_str()?)?
                }
                [Arg::Str(opt), rest] if opt.ends_with(".propertyName") => {
                    t.with_algo_property(rest.as_str()?.to_string())
                }
                [Arg::Str(opt), rest] if opt.ends_with(".times") => match rest.as_gval()? {
                    GVal::Num(n) => t.with_algo_times(n as u32),
                    _ => return Err("with(times): expected a number".into()),
                },
                _ => return Err("with(): unsupported option".into()),
            },
            "barrier" => t.barrier(),
            // iteration (`repeat` is handled above so it can pull in pre-form
            // emit()/until() modulators)
            "times" => t.times(count_at(&nums_after_scope(&args), 0, "times")?),
            "until" => t.until(self.one_trav(args)?),
            "emit" => {
                if args.is_empty() {
                    t.emit_all()
                } else {
                    t.emit(self.one_trav(args)?)
                }
            }
            // tagging / select
            "as" => t.as_(args.first().ok_or("as: expected a tag name")?.as_str()?),
            "select" => match args.first() {
                Some(Arg::Column(c)) => t.select_column(*c),
                Some(Arg::Pop(p)) => t.select_pop(*p, &str_labels),
                _ => t.select(&str_labels),
            },
            // misc
            "unfold" => t.unfold(),
            "index" => t.index(),
            "loops" => t.loops(),
            "constant" => t.constant(
                args.first()
                    .ok_or("constant: expected a value")?
                    .as_gval()?,
            ),
            "math" => t.math(
                args.first()
                    .ok_or("math: expected an expression")?
                    .as_str()?,
            ),
            "identity" => t.identity(),
            "inject" => t.inject(
                args.iter()
                    .map(Arg::as_gval)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            "none" => match &args[..] {
                [Arg::Pred(p)] => t.none_pred(p.clone()),
                _ => t.none(),
            },
            "fail" => t.fail(
                args.first()
                    .and_then(|a| a.as_str().ok())
                    .unwrap_or("fail() reached"),
            ),
            // mutation
            "addV" => t.add_v(args.first().and_then(|a| a.as_str().ok())),
            "addE" => t.add_e(
                args.first()
                    .ok_or("addE: expected an edge label")?
                    .as_str()?,
            ),
            "from" => match &args[..] {
                [Arg::Str(tag)] => t.from_tag(tag),
                [Arg::Trav(p)] => t.from_plan(p.clone()),
                _ => return Err("from: expected a tag or traversal".into()),
            },
            "to" => match &args[..] {
                [Arg::Str(tag)] => t.to_tag(tag),
                [Arg::Trav(p)] => t.to_plan(p.clone()),
                _ => return Err("to: expected a tag or traversal".into()),
            },
            "property" => {
                let k = args.first().ok_or("property: expected a key")?.as_str()?;
                let arg = args.get(1).ok_or("property: expected a value")?;
                // A traversal value is standard TinkerPop (`property('deg',
                // __.outE().count())`) — evaluated per element, not a literal.
                match arg {
                    Arg::Trav(sub) => t.property_trav(k, sub.clone()),
                    _ => t.property(k, arg.as_gval()?),
                }
            }
            "drop" => t.drop(),
            "withSack" => {
                let v = args
                    .first()
                    .ok_or("withSack: expected an initial value")?
                    .as_gval()?;
                t.with_sack(v)
            }
            "sack" => match args.first() {
                None => t.sack(),
                Some(Arg::SackOp(op)) => t.sack_op(*op),
                Some(_) => {
                    return Err("sack: expected an operator (sum/mult/min/max/assign)".into())
                }
            },
            _ => return Err(format!("unknown step `{name}`")),
        })
    }

    /// `by(...)` — attach an identity/key/token/traversal modulator (+ direction).
    fn bind_by(&self, t: Traversal, args: Vec<Arg>) -> Result<Traversal, String> {
        let dir = args.iter().find_map(|a| {
            if let Arg::Order(o) = a {
                Some(*o)
            } else {
                None
            }
        });
        let primary = args.iter().find(|a| !matches!(a, Arg::Order(_)));
        Ok(match (primary, dir) {
            (None, None) => t.by_identity(),
            (None, Some(d)) => t.by_identity_dir(d),
            (Some(Arg::Str(k)), None) => t.by(k),
            (Some(Arg::Str(k)), Some(d)) => t.by_dir(k, d),
            (Some(Arg::Token(tok)), _) => t.by_token(*tok),
            (Some(Arg::Column(c)), None) => t.by_column(*c),
            (Some(Arg::Column(c)), Some(d)) => t.by_column_dir(*c, d),
            (Some(Arg::Trav(p)), None) => t.by_t(p.clone()),
            (Some(Arg::Trav(p)), Some(d)) => t.by_t_dir(p.clone(), d),
            _ => return Err("by: unsupported modulator".into()),
        })
    }

    /// Attach an `option(match, traversal)` modulator to the most recent
    /// `branch()`. `option(none, traversal)` (TinkerPop's `Pick.none`) sets the
    /// default branch — `none` parses as a bare `none()` traversal, detected here.
    fn bind_option(&self, mut t: Traversal, args: Vec<Arg>) -> Result<Traversal, String> {
        if args.len() != 2 {
            return Err("option: expected (match, traversal)".into());
        }
        let sub = match &args[1] {
            Arg::Trav(tr) => tr.clone(),
            _ => return Err("option: 2nd arg must be a traversal".into()),
        };
        let is_default = matches!(
            &args[0],
            Arg::Trav(tr) if tr.steps.len() == 1 && matches!(tr.steps[0], Step::None(None))
        );
        match t.steps.last_mut() {
            Some(Step::Branch {
                options, default, ..
            }) => {
                if is_default {
                    *default = Some(Box::new(sub));
                } else {
                    let m = args[0].as_gval()?;
                    options.push((m, sub));
                }
            }
            _ => return Err("option: no preceding branch()".into()),
        }
        Ok(t)
    }

    fn travs(&self, args: Vec<Arg>) -> Result<Vec<Traversal>, String> {
        args.into_iter()
            .map(|a| match a {
                Arg::Trav(t) => Ok(t),
                _ => Err("expected a traversal argument".to_string()),
            })
            .collect()
    }
    fn one_trav(&self, args: Vec<Arg>) -> Result<Traversal, String> {
        self.travs(args)?
            .into_iter()
            .next()
            .ok_or_else(|| "expected a traversal argument".into())
    }
}

/// A bare (static-import) enum-member token: `asc`/`desc`, `local`/`global`,
/// `first`/`last`/`all`. `None` if `id` isn't one of these.
fn bare_token(id: &str) -> Option<Arg> {
    Some(match id {
        "asc" | "incr" => Arg::Order(Order::Asc),
        "desc" | "decr" => Arg::Order(Order::Desc),
        // Static-import Column selectors for `order(local).by(values|keys, …)`.
        "keys" => Arg::Column(Column::Keys),
        "values" => Arg::Column(Column::Values),
        // Static-import `sack(sum)` / `sack(assign)` merge operators.
        "assign" => Arg::SackOp(super::SackOp::Assign),
        "sum" => Arg::SackOp(super::SackOp::Sum),
        "mult" => Arg::SackOp(super::SackOp::Mult),
        "min" => Arg::SackOp(super::SackOp::Min),
        "max" => Arg::SackOp(super::SackOp::Max),
        "local" => Arg::Str("Scope.local".to_string()),
        "global" => Arg::Str("Scope.global".to_string()),
        "first" => Arg::Pop(Pop::First),
        "last" => Arg::Pop(Pop::Last),
        "all" => Arg::Pop(Pop::All),
        // TinkerPop's `Pick.none` — the default branch in `branch().option(none, …)`.
        // Represented as a bare `none()` traversal, which `bind_option` detects.
        "none" => Arg::Trav(Traversal {
            steps: vec![Step::None(None)],
        }),
        _ => return None,
    })
}

/// Parse a textual Gremlin query into a [`Traversal`].
/// Pre-form repeat modulators — `emit()` / `until()` appearing BEFORE their
/// `repeat(body)` in the text — stashed during a traversal parse and applied when
/// the next `repeat` is built. (TinkerPop allows the modulator to precede its
/// repeat; our builder attaches modulators to the last step, so without this a
/// pre-repeat one would be silently dropped.)
#[derive(Default)]
struct PendingMods {
    emit: Option<Traversal>,
    until: Option<Traversal>,
}

pub fn parse(src: &str) -> Result<Traversal, String> {
    let toks = lex(src)?;
    let mut p = Parser {
        toks,
        pos: 0,
        depth: 0,
    };
    let t = p.traversal()?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens from position {}", p.pos));
    }
    Ok(t)
}
