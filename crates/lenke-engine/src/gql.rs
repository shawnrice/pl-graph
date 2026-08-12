//! GQL front-end: parse a subset of ISO-GQL into the neutral IR ([`crate::ir`]).
//!
//! This is a thin compiler — it produces a `Plan` and knows nothing about
//! execution. Correctness is defined by the IR it emits: the tests assert that a
//! parsed query runs to the same rows as the hand-built plan for that query, so
//! the parser is right exactly when it reproduces the already-tested IR.
//!
//! Subset so far (grows per iteration):
//! `MATCH (a[:L]) [ -[:R]-> (b) ]* [WHERE <pred>] RETURN <item> [, <item>]*`
//! — a node, a chain of directed/undirected relationship hops, a three-valued
//! WHERE (comparisons, AND/OR/NOT, parens), and a projection of `var` / `var.key`
//! / literal items with optional `AS alias`. Aggregation, ORDER/SKIP/LIMIT,
//! DISTINCT, comma-joins, and variable-length join in later iterations.

use std::collections::{HashMap, HashSet};

use crate::ir::{AggFn, CastTarget, CompareOp, Dir, Expr, PathMode, PathPart, Plan};
use crate::value::Value;

/// A parsed node pattern head: its optional variable, the SEED label (a single
/// positive label used for `Scan`, or `None`), inline property map (empty when
/// absent), the token span of an optional inline `WHERE` predicate, and a residual
/// LABEL predicate — `Some` only for a compound label expression (`:A&B`, `:A|B`,
/// `:!A`) that the seed label does not fully cover, applied as a filter on the slot.
type ParsedNode = (
    Option<String>,
    Option<String>,
    Vec<(String, Value)>,
    Option<(usize, usize)>,
    Option<LabelExpr>,
);

/// A boolean label expression (`%` = any, a label, `!`/`&`/`|`), parsed from a node's
/// `:…` (or `IS …`) label position. Lowered per-slot to a predicate over the node's
/// label set. `%` and a bare positive label need no residual filter (a bare label is
/// the seed `Scan`, `%` scans all); the compound forms do.
enum LabelExpr {
    Any,
    Label(String),
    Not(Box<LabelExpr>),
    And(Box<LabelExpr>, Box<LabelExpr>),
    Or(Box<LabelExpr>, Box<LabelExpr>),
}

impl LabelExpr {
    /// A single positive label to seed the `Scan` on — `Label(L)`, or the first one
    /// found down a conjunction chain (`A & …` seeds on `A`); `None` otherwise.
    fn seed_label(&self) -> Option<String> {
        match self {
            LabelExpr::Label(l) => Some(l.clone()),
            LabelExpr::And(a, b) => a.seed_label().or_else(|| b.seed_label()),
            _ => None,
        }
    }
    /// `Some(self)` when a residual per-slot filter is needed — every compound form.
    /// A bare label is covered by the seed `Scan`; `%` scans everything.
    fn needs_filter(self) -> Option<LabelExpr> {
        match self {
            LabelExpr::Any | LabelExpr::Label(_) => None,
            other => Some(other),
        }
    }
}

/// The label predicate for a NON-seed node (a landing node — it has no `Scan` to
/// carry a seed label, so the WHOLE label constraint is a filter). `None` when the
/// node is unlabelled or `:%` (any). Combines the seed label and the compound
/// residual that `node()` split apart.
fn landing_label_filter(
    label: Option<String>,
    label_expr: Option<LabelExpr>,
    slot: usize,
) -> Option<Expr> {
    if let Some(le) = label_expr {
        Some(lower_label_expr(&le, slot)) // compound already includes the seed label
    } else {
        label.map(|l| lower_label_expr(&LabelExpr::Label(l), slot))
    }
}

/// Lower a label expression to a boolean predicate over the node in `slot`: a label
/// `L` → `'L' IN labels(slot)`, `%` → TRUE, with the boolean operators.
fn lower_label_expr(le: &LabelExpr, slot: usize) -> Expr {
    match le {
        LabelExpr::Any => Expr::Lit(Value::Bool(true)),
        LabelExpr::Label(l) => Expr::In {
            needle: Box::new(Expr::Lit(Value::Str(l.clone().into()))),
            haystack: Box::new(Expr::Call {
                name: "labels".into(),
                args: vec![Expr::Slot(slot)],
            }),
        },
        LabelExpr::Not(x) => Expr::Not(Box::new(lower_label_expr(x, slot))),
        LabelExpr::And(a, b) => Expr::And(
            Box::new(lower_label_expr(a, slot)),
            Box::new(lower_label_expr(b, slot)),
        ),
        LabelExpr::Or(a, b) => Expr::Or(
            Box::new(lower_label_expr(a, slot)),
            Box::new(lower_label_expr(b, slot)),
        ),
    }
}

/// Turn a node's inline properties `{k: v, …}` into a chain of `Eq` filters on its
/// slot — the exact lowering of `WHERE slot.k = v AND …`. Sharing this single form
/// is what makes `(n:L {k: v})` and `MATCH (n:L) WHERE n.k = v` optimize to the same
/// plan (and seed the same index), so the two spellings cannot cost differently.
fn node_prop_filters(mut plan: Plan, slot: usize, props: Vec<(String, Value)>) -> Plan {
    for (k, val) in props {
        plan = plan.filter(Expr::Compare {
            op: CompareOp::Eq,
            left: Box::new(Expr::Prop { slot, key: k }),
            right: Box::new(Expr::Lit(val)),
        });
    }
    plan
}

/// A parsed relationship pattern `-[var:Type {props}]->`: direction, edge type,
/// an optional bound variable, and inline properties (a match filter in a
/// pattern, edge properties to write in an INSERT).
struct Rel {
    dir: Dir,
    /// The edge types: EMPTY for an UNTYPED relationship (`-->`, `-[r]->`, `-[]->`)
    /// which traverses edges of ANY type, one entry for `-[:T]->`, or several for a
    /// disjunction `-[:A|B]->` (matches an edge whose type is any of them).
    etypes: Vec<String>,
    var: Option<String>,
    props: Vec<(String, Value)>,
    /// Token span of an inline `WHERE pred` on the edge (`-[e:T WHERE pred]->`),
    /// re-parsed once the edge is bound to a slot; `None` when absent.
    where_range: Option<(usize, usize)>,
}

/// Aggregate outputs referenced inside a projection expression use a slot at this
/// base + local index (`count(*) + 1` → `Slot(AGG_SLOT_BASE) + 1`), distinguishing
/// them from ordinary binding slots so `apply_items` can rewrite them to the real
/// post-aggregation column once the group schema is assembled.
const AGG_SLOT_BASE: usize = 1 << 40;

/// A parsed RETURN item: a keyed expression (a grouping key / plain projection), a
/// bare aggregate, or an expression that CONTAINS aggregates (`count(*) + 1`) — the
/// last carries the hoisted aggregates and an expression that references them by
/// `AGG_SLOT_BASE`-offset slots.
enum RetItem {
    Key(String, Expr),
    Agg(crate::ir::Agg),
    AggExpr {
        name: String,
        expr: Expr,
        aggs: Vec<crate::ir::Agg>,
    },
}

impl RetItem {
    fn name(&self) -> String {
        match self {
            Self::Key(n, _) => n.clone(),
            Self::Agg(a) => a.name.clone(),
            Self::AggExpr { name, .. } => name.clone(),
        }
    }
    fn has_agg(&self) -> bool {
        matches!(self, Self::Agg(_) | Self::AggExpr { .. })
    }
}

/// Map an aggregate function name (case-insensitive) to its `AggFn`.
/// `left IN [items]` desugared to `left = i0 OR left = i1 OR …`. An empty list is
/// a constant FALSE (`1 = 0`), so `x IN []` never matches; a non-empty list keeps
/// the `=` operator's three-valued semantics (a NULL element/operand → UNKNOWN).
fn in_chain(left: &Expr, items: Vec<Expr>) -> Expr {
    let eq = |item: Expr| Expr::Compare {
        op: CompareOp::Eq,
        left: Box::new(left.clone()),
        right: Box::new(item),
    };
    let mut it = items.into_iter();
    match it.next() {
        None => Expr::Compare {
            op: CompareOp::Eq,
            left: Box::new(Expr::Lit(crate::value::Value::Num(1.0))),
            right: Box::new(Expr::Lit(crate::value::Value::Num(0.0))),
        },
        Some(first) => it.fold(eq(first), |acc, item| {
            Expr::Or(Box::new(acc), Box::new(eq(item)))
        }),
    }
}

fn agg_fn(name: &str) -> Option<AggFn> {
    Some(match name.to_ascii_uppercase().as_str() {
        "COUNT" => AggFn::Count,
        "SUM" => AggFn::Sum,
        "MIN" => AggFn::Min,
        "MAX" => AggFn::Max,
        "AVG" => AggFn::Avg,
        // Core's list aggregate is `collect_list` (SKIPS nulls); `collect` is a
        // superset alias. Distinct from Gremlin fold's null-keeping `Collect`.
        "COLLECT_LIST" | "COLLECT" => AggFn::CollectList,
        "STDDEV_POP" => AggFn::StddevPop,
        "STDDEV_SAMP" => AggFn::StddevSamp,
        "PERCENTILE_CONT" => AggFn::PercentileCont,
        "PERCENTILE_DISC" => AggFn::PercentileDisc,
        _ => return None,
    })
}

/// Parse a GQL query into a plan, or an error message.
pub fn parse(query: &str) -> Result<Plan, String> {
    let toks = lex(query)?;
    let mut p = Parser {
        toks,
        pos: 0,
        scope: HashMap::new(),
        slots: 0,
        path_vars: HashSet::new(),
        lets: Vec::new(),
        suppress_in: false,
        path_mode: PathMode::Trail,
        having_aggs: None,
        having_base: 0,
    };
    let mut plan = p.query()?;
    // `<query> UNION [ALL] <query> …`: each arm is an independent query with a fresh
    // binding scope. Left-associative.
    loop {
        let op = if p.eat_kw("UNION") {
            crate::ir::CombineOp::Union
        } else if p.eat_kw("EXCEPT") {
            crate::ir::CombineOp::Except
        } else if p.eat_kw("INTERSECT") {
            crate::ir::CombineOp::Intersect
        } else {
            break;
        };
        let all = p.eat_kw("ALL");
        p.scope = HashMap::new();
        p.slots = 0;
        p.path_vars = HashSet::new();
        let right = p.query()?;
        plan = Plan::Union {
            left: Box::new(plan),
            right: Box::new(right),
            all,
            op,
        };
    }
    if p.pos != p.toks.len() {
        return Err(format!("unexpected trailing input at token {}", p.pos));
    }
    Ok(plan)
}

// --- lexer -------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Dot,
    Comma,
    Star,
    Minus,
    Plus,
    Slash,
    Percent,
    Concat, // ||
    RArrow, // ->
    LArrow, // <-
    Tilde,  // ~ (undirected relationship delimiter)
    Pipe,   // | (edge-type disjunction in `[:A|B]`)
    Amp,    // & (multi-label conjunction in `INSERT (n:A&B)`)
    Bang,   // ! (label negation in `(n:!A)`)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Ident(String),
    Str(String),
    Num(f64),
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // Comments, checked before any operator: `--` and `//` run to end of line,
        // `/* … */` is a block. `--` is UNCONDITIONALLY a comment (GQL has no `--`
        // edge — undirected uses `~`), so this must precede the `-` and `/` arms.
        if (c == '-' && b.get(i + 1) == Some(&'-')) || (c == '/' && b.get(i + 1) == Some(&'/')) {
            i += 2;
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && b.get(i + 1) == Some(&'*') {
            i += 2;
            while i < b.len() && !(b[i] == '*' && b.get(i + 1) == Some(&'/')) {
                i += 1;
            }
            if i >= b.len() {
                return Err("unterminated block comment".into());
            }
            i += 2; // past the closing `*/`
            continue;
        }
        match c {
            '(' => out.push(Tok::LParen),
            ')' => out.push(Tok::RParen),
            '[' => out.push(Tok::LBracket),
            ']' => out.push(Tok::RBracket),
            '{' => out.push(Tok::LBrace),
            '}' => out.push(Tok::RBrace),
            ':' => out.push(Tok::Colon),
            // A `.` immediately followed by a digit is a leading-dot float (`.5`);
            // it falls through to the numeric arm below. A bare `.` is the accessor.
            '.' if !matches!(b.get(i + 1), Some(d) if d.is_ascii_digit()) => out.push(Tok::Dot),
            ',' => out.push(Tok::Comma),
            '*' => out.push(Tok::Star),
            '+' => out.push(Tok::Plus),
            '/' => out.push(Tok::Slash),
            '%' => out.push(Tok::Percent),
            '~' => out.push(Tok::Tilde),
            '|' => {
                if b.get(i + 1) == Some(&'|') {
                    out.push(Tok::Concat);
                    i += 1;
                } else {
                    // A bare `|` is the edge-type-disjunction separator (`[:A|B]`).
                    out.push(Tok::Pipe);
                }
            }
            '&' => out.push(Tok::Amp),
            '!' => out.push(Tok::Bang),
            '=' => out.push(Tok::Eq),
            '-' => {
                if b.get(i + 1) == Some(&'>') {
                    out.push(Tok::RArrow);
                    i += 1;
                } else {
                    out.push(Tok::Minus);
                }
            }
            '<' => match b.get(i + 1) {
                Some('>') => {
                    out.push(Tok::Ne);
                    i += 1;
                }
                Some('=') => {
                    out.push(Tok::Le);
                    i += 1;
                }
                Some('-') => {
                    out.push(Tok::LArrow);
                    i += 1;
                }
                _ => out.push(Tok::Lt),
            },
            '>' => {
                if b.get(i + 1) == Some(&'=') {
                    out.push(Tok::Ge);
                    i += 1;
                } else {
                    out.push(Tok::Gt);
                }
            }
            '\'' => {
                // A single-quoted string with backslash escapes: the simple set
                // (`\\ \' \" \t \n \r \b \f`), `\uXXXX` (4 hex) / `\UXXXXXX` (6 hex)
                // code points, and any other `\x` → `x` (drop the backslash). Mirrors
                // core's lexer; a malformed `\u`/`\U` is a syntax error (kept as an
                // intentional reject-parity divergence).
                let mut t = String::new();
                i += 1;
                loop {
                    if i >= b.len() {
                        return Err("unterminated string literal".into());
                    }
                    let ch = b[i];
                    if ch == '\'' {
                        break; // closing quote
                    }
                    if ch == '\\' {
                        let Some(&esc) = b.get(i + 1) else {
                            return Err("unterminated string escape".into());
                        };
                        match esc {
                            '\\' => t.push('\\'),
                            '\'' => t.push('\''),
                            '"' => t.push('"'),
                            't' => t.push('\t'),
                            'n' => t.push('\n'),
                            'r' => t.push('\r'),
                            'b' => t.push('\u{0008}'),
                            'f' => t.push('\u{000C}'),
                            'u' | 'U' => {
                                let width = if esc == 'u' { 4 } else { 6 };
                                let end = i + 2 + width;
                                let cp = b
                                    .get(i + 2..end)
                                    .map(|w| w.iter().collect::<String>())
                                    .and_then(|h| u32::from_str_radix(&h, 16).ok())
                                    .and_then(char::from_u32)
                                    .ok_or_else(|| {
                                        format!(
                                            "invalid \\{esc} escape (expected {width} hex digits)"
                                        )
                                    })?;
                                t.push(cp);
                                i = end;
                                continue;
                            }
                            other => t.push(other), // unknown escape: drop the backslash
                        }
                        i += 2;
                        continue;
                    }
                    t.push(ch);
                    i += 1;
                }
                out.push(Tok::Str(t));
            }
            _ if c.is_ascii_digit() || c == '.' => {
                // Radix prefixes `0x`/`0o`/`0b` — an integer in that base (value as
                // f64, matching core). Else a decimal.
                if c == '0' && i + 1 < b.len() {
                    let radix = match b[i + 1].to_ascii_lowercase() {
                        'x' => Some(16u32),
                        'o' => Some(8),
                        'b' => Some(2),
                        _ => None,
                    };
                    if let Some(radix) = radix {
                        let start = i + 2;
                        let mut j = start;
                        while j < b.len() && b[j].is_digit(radix) {
                            j += 1;
                        }
                        let digits: String = b[start..j].iter().collect();
                        let v = u64::from_str_radix(&digits, radix)
                            .map_err(|_| format!("bad radix literal `{}`", &digits))?;
                        out.push(Tok::Num(v as f64));
                        i = j;
                        continue;
                    }
                }
                // Decimal: integer part, optional `.fraction`, optional `e[+/-]exp`.
                // Underscores are permitted inside digit runs (`1_000`) and stripped
                // before parsing; a leading-dot float (`.5`) has no integer part.
                // Matches core's lexer exactly.
                let start = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '_') {
                    i += 1;
                }
                if i < b.len() && b[i] == '.' {
                    i += 1;
                    while i < b.len() && (b[i].is_ascii_digit() || b[i] == '_') {
                        i += 1;
                    }
                }
                if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                    i += 1;
                    if i < b.len() && (b[i] == '+' || b[i] == '-') {
                        i += 1;
                    }
                    while i < b.len() && (b[i].is_ascii_digit() || b[i] == '_') {
                        i += 1;
                    }
                }
                let text: String = b[start..i].iter().collect();
                let cleaned: String = text.chars().filter(|&ch| ch != '_').collect();
                let n: f64 = cleaned
                    .parse()
                    .map_err(|_| format!("bad number `{text}`"))?;
                out.push(Tok::Num(n));
                continue; // i already advanced past the number
            }
            _ if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                    i += 1;
                }
                out.push(Tok::Ident(b[start..i].iter().collect()));
                continue;
            }
            other => return Err(format!("unexpected character `{other}`")),
        }
        i += 1;
    }
    Ok(out)
}

// --- parser ------------------------------------------------------------------

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    /// variable name -> slot index.
    scope: HashMap<String, usize>,
    /// number of bound slots so far (the next slot to assign).
    slots: usize,
    /// Names bound to the current row's PATH (a `MATCH p = ANY SHORTEST …`). They
    /// resolve to `Expr::Path` rather than a slot, since the path is the lineage
    /// sidecar — one per row — not a batch column.
    path_vars: HashSet<String>,
    /// `LET name = expr IN …` local bindings (a stack for nesting). A reference to
    /// such a name in an expression inlines its bound `Expr` (substitution), so LET
    /// needs no runtime concept — the body is a plain expression once parsed.
    lets: Vec<(String, Expr)>,
    /// While parsing a `LET` binding value, the top-level `IN` is the LET separator,
    /// NOT the membership operator — so `LET x = 2 + 3 IN body` binds `2+3`, not
    /// `2+3 IN body`. This suppresses the `IN`-membership handling in `cmp_expr`.
    suppress_in: bool,
    /// The path mode of the pattern currently being parsed, set per `match_body()`
    /// from a leading mode keyword (`WALK`/`TRAIL`/`SIMPLE`/`ACYCLIC`, default
    /// `Trail`) and read at the `var_length` build. See [`PathMode`].
    path_mode: PathMode,
    /// While parsing a `HAVING` predicate, aggregate calls in expression position
    /// (`count(*) > 1`) are HOISTED into this list and replaced with a `Slot` into
    /// the post-aggregation schema (see `select_with_having`). `None` everywhere
    /// else — an aggregate outside HAVING/return-items is an error.
    having_aggs: Option<Vec<crate::ir::Agg>>,
    /// Post-aggregation schema index of the FIRST hoisted HAVING aggregate
    /// (`keys.len() + select_aggs.len()`); the i-th lands at `having_base + i`.
    having_base: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Tok) -> Result<(), String> {
        if self.eat(t) {
            Ok(())
        } else {
            Err(format!("expected {t:?} at token {}", self.pos))
        }
    }

    /// Whether the next token is the keyword `kw` (case-insensitive), without
    /// consuming it.
    fn peek_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case(kw))
    }

    /// Consume the next token if it is an identifier equal (case-insensitive) to
    /// `kw` — the keyword test (keywords are not reserved, just matched here).
    fn eat_kw(&mut self, kw: &str) -> bool {
        if let Some(Tok::Ident(s)) = self.peek() {
            if s.eq_ignore_ascii_case(kw) {
                self.pos += 1;
                return true;
            }
        }
        false
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(format!("expected an identifier, got {other:?}")),
        }
    }

    // query := MATCH pattern [WHERE expr]
    //          RETURN [DISTINCT] items [ORDER BY keys] [SKIP n] [LIMIT n]
    fn query(&mut self) -> Result<Plan, String> {
        if self.eat_kw("_MERGE") {
            return self.merge();
        }
        if self.eat_kw("INSERT") {
            return self.insert();
        }
        // A top-level named-procedure call: `CALL name(config) [YIELD …]` invoking a
        // built-in graph algorithm. (The inline `CALL (scope) { … }` form only
        // occurs after a MATCH, inside `query_tail`.)
        if self.eat_kw("CALL") {
            return self.call_procedure();
        }
        // A bare `RETURN <items>` with no MATCH — a "return statement" (ISO GQL primary
        // query). It projects its items over ONE unit row (`Plan::Row`), so literals,
        // arithmetic, function calls and `count(*)` (= 1) evaluate with no bindings.
        if self.peek_kw("RETURN") {
            return self.query_tail(Plan::Row);
        }
        // `SELECT items [FROM MATCH …]` — the SQL-style projection, pure sugar for
        // MATCH…RETURN.
        if self.peek_kw("SELECT") {
            return self.select_statement();
        }
        if !self.eat_kw("MATCH") {
            return Err("expected MATCH, INSERT, CALL, or RETURN".into());
        }
        // Named-path form: `MATCH p = <selector> (a)-[:R]->*(b)`. The path variable
        // binds to the row's path (lineage); the rest (WHERE/WITH/RETURN) is shared.
        if matches!(self.peek(), Some(Tok::Ident(_)))
            && self.toks.get(self.pos + 1) == Some(&Tok::Eq)
        {
            let pname = self.ident()?;
            self.expect(&Tok::Eq)?;
            self.path_vars.insert(pname);
            // With a shortest-path selector it is a shortest-path pattern; without
            // one it is a plain named path (the ISO default WALK/TRAIL body), which
            // binds the pattern's lineage exactly like an unnamed MATCH — the path
            // variable just makes it readable via path_length(p)/nodes(p)/edges(p).
            if let Some(selector) = self.parse_shortest_selector()? {
                let (mut plan, scope, slots) = self.shortest_pattern(selector)?;
                self.scope = scope;
                self.slots = slots;
                if self.eat_kw("WHERE") {
                    plan = plan.filter(self.expr()?);
                }
                return self.query_tail(plan);
            }
            let plan = self.match_body()?;
            return self.query_tail(plan);
        }
        // Bare selector form: `MATCH ALL SHORTEST (a)-[:R]->*(x)` — no path variable,
        // just the reached endpoints.
        if let Some(selector) = self.parse_shortest_selector()? {
            let (mut plan, scope, slots) = self.shortest_pattern(selector)?;
            self.scope = scope;
            self.slots = slots;
            if self.eat_kw("WHERE") {
                plan = plan.filter(self.expr()?);
            }
            return self.query_tail(plan);
        }
        let plan = self.match_body()?;
        self.query_tail(plan)
    }

    /// Parse a MATCH pattern body — the `MATCH` keyword (and any named-path head)
    /// already consumed: an optional leading path mode, the comma-joined pattern
    /// list, publishing `scope`/`slots`, and an optional trailing `WHERE`. Returns
    /// the plan. Shared by `query()` and `SELECT … FROM MATCH …`.
    fn match_body(&mut self) -> Result<Plan, String> {
        // Optional leading path mode: `MATCH WALK …` lets a variable-length hop
        // reuse edges; `TRAIL` (the ISO default, and the engine's) forbids reusing
        // an edge; `SIMPLE`/`ACYCLIC` forbid reusing a NODE (Simple permits the
        // closing `start == end`). The keyword can only appear here — a pattern
        // always begins with `(`, and the named-path form was handled above — so a
        // bare Ident is unambiguously a mode word.
        self.path_mode = PathMode::Trail;
        if self.eat_kw("WALK") {
            self.path_mode = PathMode::Walk;
        } else if self.eat_kw("TRAIL") {
            self.path_mode = PathMode::Trail;
        } else if self.eat_kw("SIMPLE") {
            self.path_mode = PathMode::Simple;
        } else if self.eat_kw("ACYCLIC") {
            self.path_mode = PathMode::Acyclic;
        }
        // A comma-separated list of patterns, joined on shared variables. Each
        // pattern parses in its OWN slot space; join maps a shared variable's
        // left slot to its right slot, and the merged scope shifts the right
        // pattern's slots by the left width (the Join operator's convention).
        let (mut plan, mut scope, mut slots) = self.pattern()?;
        while self.eat(&Tok::Comma) {
            // Fold a shared-start LINEAR comma pattern into chained expansion. A
            // pattern that begins at an already-bound variable and only introduces
            // NEW variables (no cycle-closing) is exactly a continuing MATCH from
            // that variable — index-nested-loop over its adjacency — rather than an
            // independent Scan of the whole label + hash join. The join re-scans and
            // materializes both sides (measured ~11x slower on a two-hop join,
            // `join/tri` in vs_core_bench); the chained expand walks adjacency. Any
            // non-foldable shape (first node unbound/relabeled, or a landing var that
            // re-binds an existing one) rewinds and takes the correct hash join.
            let saved = self.pos;
            if matches!(self.peek(), Some(Tok::LParen)) {
                let probe = self.node()?;
                let start = match &probe {
                    (Some(v), None, props, None, None) if props.is_empty() => scope.get(v).copied(),
                    _ => None,
                };
                if let Some(from) = start {
                    let (mut sc, mut sl) = (scope.clone(), slots);
                    let before = sl;
                    let cand = self.extend_chain(plan.clone(), &mut sc, &mut sl, from)?;
                    // A cycle-close silently rebinds a previously-bound variable into
                    // the freshly-appended slot range (extend_chain only appends). If
                    // no old binding moved, the pattern was a pure linear extension.
                    let rebound = scope
                        .iter()
                        .any(|(k, &old)| old < before && sc.get(k) != Some(&old));
                    if !rebound {
                        plan = cand;
                        scope = sc;
                        slots = sl;
                        continue;
                    }
                }
                self.pos = saved;
            }
            let (p2, s2, k2) = self.pattern()?;
            let on: Vec<(usize, usize)> = s2
                .iter()
                .filter_map(|(v, &rslot)| scope.get(v).map(|&lslot| (lslot, rslot)))
                .collect();
            plan = Plan::join(plan, p2, on);
            for (v, &rslot) in &s2 {
                // Shared vars keep their left slot; new vars land at slots+rslot.
                scope.entry(v.clone()).or_insert(slots + rslot);
            }
            slots += k2;
        }
        // The mode applies only to this MATCH's own pattern hops; clear it so it
        // never leaks into a var-length hop inside a later `EXISTS { … }` subquery.
        self.path_mode = PathMode::Trail;
        // Publish the merged scope for WHERE/RETURN/ORDER to resolve variables.
        self.scope = scope;
        self.slots = slots;
        if self.eat_kw("WHERE") {
            let pred = self.expr()?;
            plan = plan.filter(pred);
        }
        Ok(plan)
    }

    /// The clauses after the first `MATCH … [WHERE]`: chained parts
    /// (`WITH`/continuing `MATCH`/`CALL`), then the write tail (`SET`/`REMOVE`) or
    /// the read tail (`RETURN` + `ORDER BY`/`SKIP`/`LIMIT`). Shared by the ordinary
    /// and the named-path (`ANY SHORTEST`) entry points.
    fn query_tail(&mut self, mut plan: Plan) -> Result<Plan, String> {
        // Chained query parts: a `WITH` projection boundary (which rebinds scope
        // to its carried columns) or a continuing `MATCH` (which extends the
        // working table from a carried variable). Loops until the tail clause.
        loop {
            if self.eat_kw("WITH") {
                plan = self.with_clause(plan)?;
            } else if self.eat_kw("OPTIONAL") {
                plan = self.optional_match(plan)?;
            } else if self.eat_kw("MATCH") {
                plan = self.match_continue(plan)?;
            } else if self.eat_kw("CALL") {
                plan = self.call_inline(plan)?;
            } else {
                break;
            }
        }
        // Statement-position `ORDER BY [OFFSET n] [LIMIT n]` BEFORE `RETURN`: sort
        // and page the bound rows, then the following RETURN projects. Core allows
        // this as a standalone order-and-page clause. `SKIP` is not a valid STARTER
        // here (only ORDER/OFFSET/LIMIT), matching core.
        if self.peek_kw("ORDER") || self.peek_kw("OFFSET") || self.peek_kw("LIMIT") {
            let keys = if self.eat_kw("ORDER") {
                if !self.eat_kw("BY") {
                    return Err("expected BY after ORDER".into());
                }
                self.standalone_sort_keys()?
            } else {
                Vec::new()
            };
            let skip = if self.eat_kw("OFFSET") || self.eat_kw("SKIP") {
                Some(self.usize_lit()?)
            } else {
                None
            };
            let limit = if self.eat_kw("LIMIT") {
                Some(self.usize_lit()?)
            } else {
                None
            };
            plan = plan.order_page(keys, skip, limit);
        }
        // Write tail: MATCH … (SET … | REMOVE …)+  — updates the bound nodes and
        // returns no rows. Otherwise the read tail (RETURN …).
        if self.peek_kw("SET")
            || self.peek_kw("REMOVE")
            || self.peek_kw("DELETE")
            || self.peek_kw("DETACH")
        {
            let ops = self.set_ops()?;
            return Ok(Plan::Update {
                input: Box::new(plan),
                ops,
            });
        }
        if !self.eat_kw("RETURN") {
            return Err("expected RETURN, SET, REMOVE, WITH, or MATCH".into());
        }
        let distinct = self.eat_kw("DISTINCT");
        let items = self.return_items()?;
        self.project_and_page(plan, distinct, items)
    }

    /// Given an already-parsed projection `items` (a RETURN or SELECT list), build
    /// the projection/aggregation over `plan`, then parse and apply an optional
    /// trailing `ORDER BY` (over output aliases or binding expressions, via hidden
    /// columns) and `OFFSET`/`SKIP`/`LIMIT` paging, plus `DISTINCT`. Shared by the
    /// `RETURN` tail and `SELECT … FROM MATCH …`.
    fn project_and_page(
        &mut self,
        plan: Plan,
        distinct: bool,
        mut items: Vec<RetItem>,
    ) -> Result<Plan, String> {
        // An explicit `GROUP BY <keys>` after the RETURN list (`RETURN u.n AS a,
        // count(*) AS c GROUP BY u.n ORDER BY a`) names the grouping keys. The
        // non-aggregate RETURN items already ARE the implicit grouping keys, so —
        // matching the SELECT path, which consumes GROUP BY before calling here —
        // parse it (for syntax + scope) and let the ordinary aggregate path group.
        // (On the SELECT path GROUP BY is already gone, so this is a no-op there.)
        if self.eat_kw("GROUP") {
            if !self.eat_kw("BY") {
                return Err("expected BY after GROUP".into());
            }
            loop {
                self.expr()?;
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        let visible: Vec<String> = items.iter().map(RetItem::name).collect();
        let has_agg = items.iter().any(|it| matches!(it, RetItem::Agg(_)));
        // When grouping, the non-aggregate items are the group keys; they occupy the
        // FIRST columns of the aggregate output (keys before aggregates), so an
        // `ORDER BY` over a group-key EXPRESSION (`ORDER BY s.name`) maps to that
        // column even though the bindings are gone.
        let key_slots: Vec<(Expr, usize)> = items
            .iter()
            .filter_map(|it| match it {
                RetItem::Key(_, e) => Some(e.clone()),
                RetItem::Agg(_) | RetItem::AggExpr { .. } => None,
            })
            .enumerate()
            .map(|(i, e)| (e, i))
            .collect();

        // ORDER BY: a key that is a visible output alias sorts by that column; a key
        // that is an EXPRESSION over the bindings (`ORDER BY n.age`, `a.x + a.y`) is
        // projected as a HIDDEN column here, sorted on, then dropped by a final
        // projection — so ORDER BY is not limited to the returned columns.
        let mut hidden: Vec<(String, Expr)> = Vec::new();
        let keys = if self.eat_kw("ORDER") {
            if !self.eat_kw("BY") {
                return Err("expected BY after ORDER".into());
            }
            self.order_keys(&visible, has_agg, &key_slots, &mut hidden)?
        } else {
            Vec::new()
        };
        for (name, e) in &hidden {
            items.push(RetItem::Key(name.clone(), e.clone()));
        }

        let (mut plan, _all_names) = apply_items(plan, &items);
        if distinct {
            if !hidden.is_empty() {
                return Err(
                    "ORDER BY an expression together with DISTINCT is not supported; \
                     project the sort key and ORDER BY its alias"
                        .into(),
                );
            }
            plan = plan.distinct();
        }
        // `OFFSET` is the ISO spelling of `SKIP` — a synonym here (core accepts both).
        let skip = if self.eat_kw("SKIP") || self.eat_kw("OFFSET") {
            Some(self.usize_lit()?)
        } else {
            None
        };
        let limit = if self.eat_kw("LIMIT") {
            Some(self.usize_lit()?)
        } else {
            None
        };
        if !keys.is_empty() || skip.is_some() || limit.is_some() {
            plan = plan.order_page(keys, skip, limit);
        }
        // Drop the hidden ORDER-BY columns, restoring exactly the visible outputs.
        if !hidden.is_empty() {
            let proj = visible
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), Expr::Slot(i)))
                .collect();
            plan = plan.project(proj);
        }
        Ok(plan)
    }

    /// Scan forward from the cursor for a bracket-depth-0 `FROM` — the boundary
    /// between a SELECT's item list and its `FROM MATCH` — returning its token
    /// index, or `None` if the SELECT has no FROM (a constant projection). Nested
    /// parens/brackets/braces (e.g. `count(*)`, `TRIM(x FROM y)`) are skipped; a
    /// depth-0 `UNION` ends the scan (the FROM would belong to a later arm).
    fn scan_for_from(&self) -> Option<usize> {
        let mut depth = 0i32;
        let mut i = self.pos;
        while let Some(t) = self.toks.get(i) {
            match t {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Ident(s) if depth == 0 && s.eq_ignore_ascii_case("FROM") => return Some(i),
                Tok::Ident(s) if depth == 0 && s.eq_ignore_ascii_case("UNION") => return None,
                _ => {}
            }
            i += 1;
        }
        None
    }

    /// `SELECT [DISTINCT] items [FROM [<graph>] MATCH pattern [WHERE]] [GROUP BY g]
    /// [ORDER BY o] [OFFSET/LIMIT]` — pure sugar for MATCH…RETURN. The items are
    /// written BEFORE the FROM that binds their variables, so parse FROM MATCH first
    /// (populating scope), then rewind to parse the items. GROUP BY is handled by
    /// the implicit grouping in `apply_items` (which keys by the non-aggregate
    /// items); explicit HAVING is a later phase.
    fn select_statement(&mut self) -> Result<Plan, String> {
        self.eat_kw("SELECT");
        let distinct = self.eat_kw("DISTINCT");
        let items_start = self.pos;
        let (plan, items) = if let Some(from_pos) = self.scan_for_from() {
            self.pos = from_pos;
            self.eat_kw("FROM");
            // An optional graph reference (CURRENT_GRAPH / HOME_GRAPH / …) may precede
            // MATCH — lenke is single-graph, so consume and ignore a lone ident here.
            if !self.peek_kw("MATCH") && matches!(self.peek(), Some(Tok::Ident(_))) {
                self.pos += 1;
            }
            if !self.eat_kw("MATCH") {
                return Err("expected MATCH after FROM in a SELECT".into());
            }
            let plan = self.match_body()?;
            let after_match = self.pos;
            // Rewind to parse the items against the now-populated binding scope.
            self.pos = items_start;
            let items = self.return_items()?;
            if self.pos != from_pos {
                return Err("unexpected tokens in the SELECT list before FROM".into());
            }
            self.pos = after_match;
            (plan, items)
        } else {
            // No FROM: a constant projection over one unit row (no bindings).
            self.scope = HashMap::new();
            self.slots = 0;
            let items = self.return_items()?;
            (Plan::Row, items)
        };
        // GROUP BY keys (expressions over the input bindings).
        let mut group_keys: Vec<Expr> = Vec::new();
        if self.eat_kw("GROUP") {
            if !self.eat_kw("BY") {
                return Err("expected BY after GROUP".into());
            }
            loop {
                group_keys.push(self.expr()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        // A HAVING clause needs the explicit aggregate pipeline; without it the
        // non-aggregated SELECT items are already the implicit grouping keys, so the
        // ordinary projection path handles GROUP BY (which was consumed above).
        if self.peek_kw("HAVING") {
            return self.select_with_having(plan, distinct, items, group_keys);
        }
        self.project_and_page(plan, distinct, items)
    }

    /// Parse and hoist an aggregate call encountered in a HAVING predicate: `func`
    /// is its `AggFn` (its name is already consumed; `(` is next). Records the `Agg`
    /// in `having_aggs` and returns a `Slot` into the post-aggregation schema.
    fn hoist_having_agg(&mut self, func: AggFn) -> Result<Expr, String> {
        self.expect(&Tok::LParen)?;
        let (arg, distinct, frac) = if self.eat(&Tok::Star) {
            self.expect(&Tok::RParen)?;
            (None, false, None)
        } else {
            let distinct = self.eat_kw("DISTINCT");
            let a = self.expr()?;
            let frac = if self.eat(&Tok::Comma) {
                match self.expr()? {
                    Expr::Lit(Value::Num(f)) => Some(f),
                    _ => return Err("percentile fraction must be a numeric constant".into()),
                }
            } else {
                None
            };
            self.expect(&Tok::RParen)?;
            (Some(a), distinct, frac)
        };
        let aggs = self.having_aggs.as_mut().expect("in HAVING");
        let idx = aggs.len();
        aggs.push(crate::ir::Agg {
            func,
            arg,
            distinct,
            name: format!("__h{idx}"),
            frac,
        });
        Ok(Expr::Slot(self.having_base + idx))
    }

    /// `SELECT items FROM MATCH … GROUP BY g HAVING h [ORDER BY] [paging]` — the
    /// grouped/HAVING form. Aggregates in the SELECT list and (hoisted) in HAVING
    /// share one `Aggregate` over the group keys; HAVING then filters the grouped
    /// rows (a group-key reference in HAVING is rewritten to its key column, an
    /// aggregate to its output column), and the SELECT items project in item order —
    /// dropping any HAVING-only aggregate column.
    fn select_with_having(
        &mut self,
        plan: Plan,
        distinct: bool,
        items: Vec<RetItem>,
        group_keys: Vec<Expr>,
    ) -> Result<Plan, String> {
        // Split the SELECT list into key (non-aggregate) items and aggregate items.
        let select_keys: Vec<(String, Expr)> = items
            .iter()
            .filter_map(|it| match it {
                RetItem::Key(n, e) => Some((n.clone(), e.clone())),
                RetItem::Agg(_) | RetItem::AggExpr { .. } => None,
            })
            .collect();
        let select_aggs: Vec<crate::ir::Agg> = items
            .iter()
            .filter_map(|it| match it {
                RetItem::Agg(a) => Some(a.clone()),
                RetItem::Key(..) | RetItem::AggExpr { .. } => None,
            })
            .collect();
        // Grouping keys: the SELECT non-aggregate items, plus any GROUP BY expression
        // not already among them (matched structurally).
        let mut keys: Vec<(String, Expr)> = select_keys.clone();
        for gk in &group_keys {
            if !keys.iter().any(|(_, e)| expr_eq(e, gk)) {
                keys.push((format!("__gk{}", keys.len()), gk.clone()));
            }
        }
        // Parse HAVING with aggregate hoisting active. Aggregates land after the
        // keys and the SELECT-list aggregates in the post-aggregation schema.
        self.eat_kw("HAVING");
        self.having_base = keys.len() + select_aggs.len();
        self.having_aggs = Some(Vec::new());
        let having_raw = self.expr()?;
        let having_aggs = self.having_aggs.take().expect("in HAVING");
        // A group-key reference in HAVING (`n.age >= 35`) reads the key column, not
        // the input property — rewrite it to the key's post-aggregation slot.
        let having = rewrite_group_keys(having_raw, &keys);

        let mut aggs = select_aggs.clone();
        aggs.extend(having_aggs);
        let mut p = plan.aggregate(keys.clone(), aggs).filter(having);

        // Project the SELECT items in ITEM order onto the post-aggregation schema
        // (`[keys…, select_aggs…, having_aggs…]`), dropping the HAVING-only columns.
        let mut proj: Vec<(String, Expr)> = Vec::with_capacity(items.len());
        for it in &items {
            match it {
                RetItem::Key(name, e) => {
                    let ki = keys
                        .iter()
                        .position(|(_, ke)| expr_eq(ke, e))
                        .ok_or("a SELECT item is neither an aggregate nor a group key")?;
                    proj.push((name.clone(), Expr::Slot(ki)));
                }
                RetItem::Agg(a) => {
                    let ai = select_aggs
                        .iter()
                        .position(|sa| sa.name == a.name)
                        .expect("select agg");
                    proj.push((a.name.clone(), Expr::Slot(keys.len() + ai)));
                }
                RetItem::AggExpr { .. } => {
                    return Err(
                        "an aggregate expression in a SELECT with GROUP BY/HAVING is not \
                         supported"
                            .into(),
                    )
                }
            }
        }
        let out_names: Vec<String> = proj.iter().map(|(n, _)| n.clone()).collect();
        p = p.project(proj);
        if distinct {
            p = p.distinct();
        }
        // ORDER BY (over the output aliases) and paging.
        let sort_keys = if self.eat_kw("ORDER") {
            if !self.eat_kw("BY") {
                return Err("expected BY after ORDER".into());
            }
            self.sort_keys(&out_names)?
        } else {
            Vec::new()
        };
        let skip = if self.eat_kw("OFFSET") || self.eat_kw("SKIP") {
            Some(self.usize_lit()?)
        } else {
            None
        };
        let limit = if self.eat_kw("LIMIT") {
            Some(self.usize_lit()?)
        } else {
            None
        };
        if !sort_keys.is_empty() || skip.is_some() || limit.is_some() {
            p = p.order_page(sort_keys, skip, limit);
        }
        Ok(p)
    }

    /// Parse a shortest-path SELECTOR if one is present: `ANY SHORTEST` → Any,
    /// `ALL SHORTEST` → All, `SHORTEST 1` → Any, `SHORTEST 1 GROUP[S]` → All. Returns
    /// `None` (consuming nothing) when the next tokens are not a selector. `SHORTEST k`
    /// for k ≠ 1, and bare `ANY`/`ALL` (walk) without `SHORTEST`, are errors here.
    fn parse_shortest_selector(&mut self) -> Result<Option<crate::ir::ShortestSelector>, String> {
        use crate::ir::ShortestSelector;
        let next_is = |p: &Self, kw: &str| matches!(p.toks.get(p.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case(kw));
        if self.peek_kw("ANY") && next_is(self, "SHORTEST") {
            self.pos += 2;
            return Ok(Some(ShortestSelector::Any));
        }
        if self.peek_kw("ALL") && next_is(self, "SHORTEST") {
            self.pos += 2;
            return Ok(Some(ShortestSelector::All));
        }
        if self.eat_kw("SHORTEST") {
            let k = self.usize_lit()?;
            let group = self.eat_kw("GROUP") || self.eat_kw("GROUPS");
            return match k {
                1 if group => Ok(Some(ShortestSelector::All)),
                1 => Ok(Some(ShortestSelector::Any)),
                _ => Err("SHORTEST k for k other than 1 is not supported".into()),
            };
        }
        Ok(None)
    }

    /// The pattern after a shortest-path selector: `(a[:L] [{props}])-[:R]->*(b …)`.
    /// Seed label+props seed the `Scan`; the endpoint's props filter above the hop
    /// (its label is ignored, as elsewhere for a landing node); a same-variable
    /// endpoint (`…*(a)`) adds a `seed == endpoint` equality. `*` → min 0 (the seed
    /// is a zero-length path to itself), `+` → min 1. Inline WHERE / bound edge stay
    /// rejected. Binds seed→slot 0, endpoint→slot 1.
    fn shortest_pattern(
        &mut self,
        selector: crate::ir::ShortestSelector,
    ) -> Result<(Plan, HashMap<String, usize>, usize), String> {
        let mut scope: HashMap<String, usize> = HashMap::new();
        let (va, la, va_props, va_where, _va_le) = self.node()?;
        if let Some(v) = &va {
            scope.insert(v.clone(), 0);
        }
        let rel = self.rel()?;
        let min = if self.eat(&Tok::Star) {
            0
        } else if self.eat(&Tok::Plus) {
            1
        } else {
            return Err("a shortest path requires a `*` or `+` quantifier".into());
        };
        let (vb, _lb, vb_props, vb_where, _vb_le) = self.node()?;
        if va_where.is_some() || vb_where.is_some() || rel.where_range.is_some() {
            return Err("inline WHERE on a shortest-path element is not supported".into());
        }
        // Seed node: label + inline props seed the scan (slot 0).
        let mut plan = node_prop_filters(Plan::Scan { label: la }, 0, va_props);
        plan = plan.shortest_path(0, rel.dir, &rel.etypes, min, None, selector);
        // Endpoint node at slot 1: inline props filter it; its label is ignored (as
        // for any landing node in this subset).
        plan = node_prop_filters(plan, 1, vb_props);
        // A same-variable endpoint closes back on the seed: constrain slot 1 == slot 0.
        if let Some(v) = &vb {
            if scope.get(v) == Some(&0) {
                plan = plan.filter(Expr::Compare {
                    op: CompareOp::Eq,
                    left: Box::new(Expr::Slot(0)),
                    right: Box::new(Expr::Slot(1)),
                });
            } else {
                scope.insert(v.clone(), 1);
            }
        }
        Ok((plan, scope, 2))
    }

    // sort keys := name [ASC|DESC] ( ',' name [ASC|DESC] )*
    // Each `name` is an OUTPUT column (by alias); it maps to that output slot.
    /// ORDER BY over columns already in scope (the WITH boundary rebinds scope to the
    /// projected outputs, so a key names an output column or an expression over them,
    /// both evaluable on the post-projection batch — no hidden column needed).
    fn sort_keys(&mut self, out_names: &[String]) -> Result<Vec<crate::ir::SortKey>, String> {
        let mut keys = Vec::new();
        loop {
            let name = self.ident()?;
            let slot = out_names
                .iter()
                .position(|n| n == &name)
                .ok_or_else(|| format!("ORDER BY `{name}` is not an output column"))?;
            let descending = if self.eat_kw("DESC") {
                true
            } else {
                self.eat_kw("ASC");
                false
            };
            let nulls_first = self.parse_nulls_order()?;
            keys.push(crate::ir::SortKey {
                expr: Expr::Slot(slot),
                descending,
                nulls_first,
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(keys)
    }

    /// Parse an optional `NULLS FIRST|LAST` after a sort key's ASC/DESC. The default
    /// (no clause) is `false` (nulls sort LAST, both directions — matching core).
    fn parse_nulls_order(&mut self) -> Result<bool, String> {
        if self.eat_kw("NULLS") {
            if self.eat_kw("FIRST") {
                Ok(true)
            } else if self.eat_kw("LAST") {
                Ok(false)
            } else {
                Err("expected FIRST or LAST after NULLS".into())
            }
        } else {
            Ok(false)
        }
    }

    /// Parse a statement-position ORDER BY key list (a standalone sort BEFORE the
    /// RETURN): each key is a full expression over the current bindings with an
    /// optional ASC/DESC. Simpler than `order_keys` — the rows are the bindings, so
    /// no hidden output column is needed.
    fn standalone_sort_keys(&mut self) -> Result<Vec<crate::ir::SortKey>, String> {
        let mut keys = Vec::new();
        loop {
            let expr = self.expr()?;
            let descending = if self.eat_kw("DESC") {
                true
            } else {
                self.eat_kw("ASC");
                false
            };
            let nulls_first = self.parse_nulls_order()?;
            keys.push(crate::ir::SortKey {
                expr,
                descending,
                nulls_first,
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(keys)
    }

    /// Parse the ORDER BY key list. Each key is either a bare VISIBLE output alias
    /// (→ sort by that column) or an EXPRESSION over the bindings, which is appended
    /// to `hidden` (projected as a temporary column) and sorted on. An expression key
    /// is rejected when the projection aggregates (the bindings are gone by then).
    fn order_keys(
        &mut self,
        visible: &[String],
        has_agg: bool,
        key_slots: &[(Expr, usize)],
        hidden: &mut Vec<(String, Expr)>,
    ) -> Result<Vec<crate::ir::SortKey>, String> {
        // Is the next token a bare visible alias — an ident naming a visible column,
        // followed by a sort TERMINATOR (comma / ASC / DESC / SKIP / LIMIT / end)?
        // `n.age` is NOT (the `.` continues an expression), so it falls to `expr()`.
        let terminator = |t: Option<&Tok>| {
            matches!(t, None | Some(Tok::Comma))
                || matches!(t, Some(Tok::Ident(s))
                    if ["DESC", "ASC", "SKIP", "OFFSET", "LIMIT", "NULLS"].iter().any(|k| s.eq_ignore_ascii_case(k)))
        };
        let mut keys = Vec::new();
        loop {
            let alias = match self.peek() {
                Some(Tok::Ident(name)) => visible
                    .iter()
                    .position(|n| n == name)
                    .filter(|_| terminator(self.toks.get(self.pos + 1))),
                _ => None,
            };
            let expr = if let Some(slot) = alias {
                self.bump();
                Expr::Slot(slot)
            } else {
                let e = self.expr()?;
                // ORDER BY an expression that IS a projected item's expression
                // (`RETURN u.n AS a … ORDER BY u.n`) sorts by that OUTPUT column, not
                // a hidden one — so it composes with DISTINCT (a hidden sort column
                // would change the DISTINCT row) and, under aggregation, references a
                // group key. Falls through to a hidden column only for a genuinely
                // new expression, which DISTINCT then rejects below.
                if let Some((_, slot)) = key_slots.iter().find(|(ke, _)| expr_eq(ke, &e)) {
                    Expr::Slot(*slot)
                } else if has_agg {
                    return Err("ORDER BY with aggregation must reference an output \
                                column alias or a group key"
                        .into());
                } else {
                    let slot = visible.len() + hidden.len();
                    hidden.push((format!("__order{}", hidden.len()), e));
                    Expr::Slot(slot)
                }
            };
            let descending = if self.eat_kw("DESC") {
                true
            } else {
                self.eat_kw("ASC"); // optional, default ascending
                false
            };
            // `NULLS FIRST|LAST` overrides the default (nulls last, both directions).
            let nulls_first = self.parse_nulls_order()?;
            keys.push(crate::ir::SortKey {
                expr,
                descending,
                nulls_first,
            });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(keys)
    }

    fn usize_lit(&mut self) -> Result<usize, String> {
        match self.bump() {
            Some(Tok::Num(n)) if n >= 0.0 && n.fract() == 0.0 => Ok(n as usize),
            other => Err(format!("expected a non-negative integer, got {other:?}")),
        }
    }

    // merge := _MERGE '(' [var] ':' Label [ '{' props '}' ] ')'
    //          [ _ON_CREATE SET assign_list ]
    //          [ _ON_UPDATE SET assign_list [WHERE expr] | _ON_UPDATE_NOTHING ]
    // The sigil'd container owns the upsert semantics; the pattern and SET/WHERE
    // inside stay bare (they are standalone-valid GQL). See gql-extensions.md §1.
    fn merge(&mut self) -> Result<Plan, String> {
        self.expect(&Tok::LParen)?;
        let var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident()?)
        } else {
            None
        };
        // v1: exactly one label (the upsert target's).
        self.expect(&Tok::Colon)?;
        let label = self.ident()?;
        let props = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.props()?
        } else {
            Vec::new()
        };
        self.expect(&Tok::RParen)?;

        // Bind the merged node at slot 0 so _ON_CREATE/_ON_UPDATE SET and WHERE
        // resolve `var.key`.
        self.scope = HashMap::new();
        if let Some(v) = &var {
            self.scope.insert(v.clone(), 0);
        }
        self.slots = 1;

        let on_create = if self.eat_kw("_ON_CREATE") {
            if !self.eat_kw("SET") {
                return Err("expected SET after _ON_CREATE".into());
            }
            self.assign_list()?
        } else {
            Vec::new()
        };

        let on_update = if self.eat_kw("_ON_UPDATE_NOTHING") {
            crate::ir::MergeUpdate::Nothing
        } else if self.eat_kw("_ON_UPDATE") {
            if !self.eat_kw("SET") {
                return Err("expected SET after _ON_UPDATE".into());
            }
            let assigns = self.assign_list()?;
            let filter = if self.eat_kw("WHERE") {
                Some(self.expr()?)
            } else {
                None
            };
            crate::ir::MergeUpdate::Set { assigns, filter }
        } else {
            crate::ir::MergeUpdate::Clobber
        };

        Ok(Plan::Merge {
            label,
            props,
            on_create,
            on_update,
        })
    }

    // assign_list := var '.' key '=' expr ( ',' var '.' key '=' expr )*
    // The `var` is the merged node (slot 0); its slot is inherent, so only the
    // (key, value) pair is kept.
    fn assign_list(&mut self) -> Result<Vec<(String, Expr)>, String> {
        let mut out = Vec::new();
        loop {
            let (_slot, key) = self.slot_dot_key()?;
            self.expect(&Tok::Eq)?;
            let value = self.expr()?;
            out.push((key, value));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(out)
    }

    // set_ops := ( SET assign (',' assign)* | REMOVE ref (',' ref)* )+
    // assign  := var '.' key '=' expr        ref := var '.' key
    // Interleaved SET/REMOVE clauses accumulate in order (later writes win).
    fn set_ops(&mut self) -> Result<Vec<crate::ir::SetOp>, String> {
        let mut ops = Vec::new();
        loop {
            if self.eat_kw("SET") {
                loop {
                    let (slot, key) = self.slot_dot_key()?;
                    self.expect(&Tok::Eq)?;
                    let value = self.expr()?;
                    ops.push(crate::ir::SetOp::Set { slot, key, value });
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            } else if self.eat_kw("REMOVE") {
                loop {
                    let (slot, key) = self.slot_dot_key()?;
                    ops.push(crate::ir::SetOp::Remove { slot, key });
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            } else if self.peek_kw("DETACH") || self.peek_kw("DELETE") {
                // DELETE var[, var] / DETACH DELETE var[, var]. Each var names a bound
                // element (node or edge). DETACH also removes a node's edges.
                let detach = self.eat_kw("DETACH");
                if !self.eat_kw("DELETE") {
                    return Err("expected DELETE after DETACH".into());
                }
                loop {
                    let var = self.ident()?;
                    let slot = *self
                        .scope
                        .get(&var)
                        .ok_or_else(|| format!("unknown variable `{var}`"))?;
                    ops.push(crate::ir::SetOp::Delete { slot, detach });
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            } else {
                break;
            }
        }
        if ops.is_empty() {
            return Err("expected SET, REMOVE, or DELETE".into());
        }
        Ok(ops)
    }

    // A bound `var.key` reference (the target of a SET/REMOVE).
    fn slot_dot_key(&mut self) -> Result<(usize, String), String> {
        let var = self.ident()?;
        let slot = *self
            .scope
            .get(&var)
            .ok_or_else(|| format!("unknown variable `{var}`"))?;
        self.expect(&Tok::Dot)?;
        let key = self.ident()?;
        Ok((slot, key))
    }

    // insert := INSERT insert_path ( ',' insert_path )*
    // Creates new nodes and the edges among them. Variables are scoped to this
    // INSERT: first mention defines the node (labels + props), later mentions
    // reference it (bare `(x)`). Edges must be directed and carry no properties
    // yet (the store has no edge-property model).
    fn insert(&mut self) -> Result<Plan, String> {
        let mut nodes: Vec<crate::ir::InsertNode> = Vec::new();
        let mut edges: Vec<crate::ir::InsertEdge> = Vec::new();
        let mut var_to_idx: HashMap<String, usize> = HashMap::new();
        loop {
            self.insert_path(&mut nodes, &mut edges, &mut var_to_idx)?;
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        // `INSERT (…) RETURN …`: the created nodes are bound into scope so a
        // following projection can read them. Each node keeps the slot equal to
        // its creation index (the same index `var_to_idx` records), so the tail's
        // `Expr::Prop{slot}` lines up with the seeded row the executor builds.
        if self.peek_kw("RETURN")
            || self.peek_kw("ORDER")
            || self.peek_kw("OFFSET")
            || self.peek_kw("LIMIT")
        {
            self.scope = var_to_idx;
            self.slots = nodes.len();
            let tail = self.query_tail(Plan::Row)?;
            return Ok(Plan::InsertReturn {
                nodes,
                edges,
                tail: Box::new(tail),
            });
        }
        Ok(Plan::Insert { nodes, edges })
    }

    // insert_path := insert_node ( rel insert_node )*
    fn insert_path(
        &mut self,
        nodes: &mut Vec<crate::ir::InsertNode>,
        edges: &mut Vec<crate::ir::InsertEdge>,
        var_to_idx: &mut HashMap<String, usize>,
    ) -> Result<(), String> {
        let mut prev = self.insert_node(nodes, var_to_idx)?;
        while matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde)) {
            let rel = self.rel()?;
            if rel.where_range.is_some() {
                return Err("inline WHERE on an INSERT relationship is not supported".into());
            }
            let next = self.insert_node(nodes, var_to_idx)?;
            let (from, to) = match rel.dir {
                Dir::Out => (prev, next),
                Dir::In => (next, prev),
                Dir::Both => {
                    return Err("INSERT requires a directed relationship".into());
                }
            };
            // An inserted edge has exactly one concrete type — a `|`-disjunction is
            // a MATCH construct, not creatable.
            let etype =
                match rel.etypes.as_slice() {
                    [t] => t.clone(),
                    [] => return Err("INSERT of a relationship requires an edge type".into()),
                    _ => return Err(
                        "INSERT of a relationship requires a single edge type, not a disjunction"
                            .into(),
                    ),
                };
            edges.push(crate::ir::InsertEdge {
                from,
                to,
                etype,
                props: rel.props,
            });
            prev = next;
        }
        Ok(())
    }

    // insert_node := '(' [var] (':' Label)* [ '{' props '}' ] ')'
    // Returns the node's index. A first mention with a known var defines it; a
    // later bare mention references it.
    fn insert_node(
        &mut self,
        nodes: &mut Vec<crate::ir::InsertNode>,
        var_to_idx: &mut HashMap<String, usize>,
    ) -> Result<usize, String> {
        self.expect(&Tok::LParen)?;
        let var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident()?)
        } else {
            None
        };
        let mut labels = Vec::new();
        while self.eat(&Tok::Colon) {
            labels.push(self.ident()?);
            // `&` conjoins additional labels (`:A&B`). Only AND is creatable —
            // `|`/`!` are label-expression forms that don't denote a single node.
            while self.eat(&Tok::Amp) {
                labels.push(self.ident()?);
            }
        }
        let props = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.props()?
        } else {
            Vec::new()
        };
        self.expect(&Tok::RParen)?;

        if let Some(v) = &var {
            if let Some(&idx) = var_to_idx.get(v) {
                // A reference to an already-defined node may not re-decorate it.
                if !labels.is_empty() || !props.is_empty() {
                    return Err(format!("variable `{v}` is already defined in this INSERT"));
                }
                return Ok(idx);
            }
        }
        let idx = nodes.len();
        nodes.push(crate::ir::InsertNode { labels, props });
        if let Some(v) = var {
            var_to_idx.insert(v, idx);
        }
        Ok(idx)
    }

    // props := '{' [ key ':' literal ( ',' key ':' literal )* ] '}'
    /// After an inline `WHERE` inside a pattern element (`(v:L WHERE pred)` or
    /// `-[e:T WHERE pred]->`), capture the predicate's token span WITHOUT parsing.
    /// The element's variable isn't bound to a slot until `pattern`/`extend_chain`
    /// assigns one, so the expression can't resolve `v.k` yet; `parse_captured_where`
    /// parses the span once the binding exists. The span runs to the element's own
    /// closing `)`/`]` — the first unmatched closer at bracket-depth 0.
    fn capture_inline_where(&mut self) -> (usize, usize) {
        let start = self.pos;
        let mut depth = 0i32;
        while let Some(t) = self.toks.get(self.pos) {
            match t {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                _ => {}
            }
            self.pos += 1;
        }
        (start, self.pos)
    }

    /// Parse a captured inline-`WHERE` span (see `capture_inline_where`) as a
    /// predicate expression, with `self.scope` already carrying the element's
    /// binding. Restores the cursor afterward so pattern parsing continues where
    /// it left off.
    fn parse_captured_where(&mut self, range: (usize, usize)) -> Result<Expr, String> {
        let saved = self.pos;
        self.pos = range.0;
        let e = self.expr()?;
        if self.pos != range.1 {
            return Err("unexpected tokens in inline WHERE predicate".into());
        }
        self.pos = saved;
        Ok(e)
    }

    fn props(&mut self) -> Result<Vec<(String, Value)>, String> {
        self.expect(&Tok::LBrace)?;
        let mut out = Vec::new();
        if self.eat(&Tok::RBrace) {
            return Ok(out);
        }
        loop {
            let key = self.ident()?;
            self.expect(&Tok::Colon)?;
            out.push((key, self.literal_value()?));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(out)
    }

    // A literal property value: number, string, the keyword true/false/null, or a
    // `[...]` list of literal values (used e.g. by `CALL personalized_pagerank(
    // {sourceNodes: ['a', 'b']})`; a list is a first-class stored property value).
    fn literal_value(&mut self) -> Result<Value, String> {
        if self.peek() == Some(&Tok::LBracket) {
            self.bump();
            let mut items = Vec::new();
            if !self.eat(&Tok::RBracket) {
                loop {
                    items.push(self.literal_value()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBracket)?;
            }
            return Ok(Value::List(items));
        }
        match self.bump() {
            Some(Tok::Num(n)) => Ok(Value::Num(n)),
            Some(Tok::Str(s)) => Ok(Value::Str(s.into())),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("true") => Ok(Value::Bool(true)),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("false") => Ok(Value::Bool(false)),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("null") => Ok(Value::Null),
            other => Err(format!("expected a literal value, got {other:?}")),
        }
    }

    // pattern := node ( rel [quantifier] node )*
    // Parsed in its OWN slot space (0-based), returning (plan, var->slot, width);
    // sharing across comma-patterns is resolved by name at join time.
    fn pattern(&mut self) -> Result<(Plan, HashMap<String, usize>, usize), String> {
        let mut scope: HashMap<String, usize> = HashMap::new();
        let mut slots = 0usize;
        // Unanchored subpath group `((x)-[:R]->(y)){n,m} (t)` — synthesize an
        // anonymous seed node (scan every node), matching core, then chain from it.
        if self.is_subpath_group_start() {
            let from = slots;
            slots += 1;
            let plan =
                self.extend_chain(Plan::Scan { label: None }, &mut scope, &mut slots, from)?;
            return Ok((plan, scope, slots));
        }
        let (var, label, props, where_range, label_expr) = self.node()?;
        if let Some(v) = var {
            scope.insert(v, slots);
        }
        let from = slots;
        slots += 1;
        // Inline props on the seed node become filters over the Scan — the same
        // shape a `WHERE` produces, so the optimizer's index-seeding sees both alike.
        let mut seed = node_prop_filters(Plan::Scan { label }, from, props);
        // A compound label expression (`:A&B`, `:A|B`, `:!A`) applies a residual
        // predicate on top of the seed `Scan`.
        if let Some(le) = label_expr {
            seed = seed.filter(lower_label_expr(&le, from));
        }
        if let Some(r) = where_range {
            self.scope = scope.clone();
            seed = seed.filter(self.parse_captured_where(r)?);
        }
        let plan = self.extend_chain(seed, &mut scope, &mut slots, from)?;
        Ok((plan, scope, slots))
    }

    /// Parse the `( rel [quantifier] node )*` tail of a pattern, extending `plan`
    /// (and `scope`/`slots`) hop by hop from the node in slot `from`. Shared by
    /// the initial `pattern` (which starts from a fresh `Scan`) and a continuing
    /// `MATCH` after `WITH` (which starts from a carried node's slot), so both
    /// spell a hop identically.
    fn extend_chain(
        &mut self,
        mut plan: Plan,
        scope: &mut HashMap<String, usize>,
        slots: &mut usize,
        mut from: usize,
    ) -> Result<Plan, String> {
        while matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde))
            || self.is_subpath_group_start()
        {
            // A parenthesized subpath group `((x)-[e:R]->(y)){n,m} (t)`. A SINGLE-edge
            // group with no per-rep predicate and no downstream group-variable use is
            // exactly a variable-length hop (same edge-distinct/path-mode reachability
            // to the endpoint), so it lowers to `var_length`. The endpoint `(t)` that
            // follows is parsed and bound like any landing node.
            if self.is_subpath_group_start() {
                let (dir, etypes, min, max) = self.parse_subpath_group()?;
                let (v2, v2_label, v2_props, v2_where, v2_le) = self.node()?;
                let node_slot = *slots;
                if let Some(v) = v2 {
                    scope.insert(v, node_slot);
                }
                *slots += 1;
                plan = plan.var_length(from, dir, &etypes, min, max, self.path_mode);
                from = node_slot;
                if let Some(pred) = landing_label_filter(v2_label, v2_le, from) {
                    plan = plan.filter(pred);
                }
                plan = node_prop_filters(plan, from, v2_props);
                if let Some(r) = v2_where {
                    self.scope = scope.clone();
                    plan = plan.filter(self.parse_captured_where(r)?);
                }
                continue;
            }
            let rel = self.rel()?;
            let quant = self.opt_quantifier()?;
            let (v2, v2_label, v2_props, v2_where, v2_le) = self.node()?;
            // A relationship variable, inline edge properties, or an inline edge
            // WHERE require binding the edge as a slot (edge at `slots`, node at
            // `slots+1`) so `e.k` can resolve.
            let bind = rel.var.is_some() || !rel.props.is_empty() || rel.where_range.is_some();
            if let Some((min, max)) = quant {
                if bind {
                    return Err(
                        "a relationship variable / edge properties on a variable-length \
                         pattern are not supported"
                            .into(),
                    );
                }
                let node_slot = *slots;
                if let Some(v) = v2 {
                    scope.insert(v, node_slot);
                }
                *slots += 1;
                // The leading path mode (default TRAIL) selects the reuse semantics.
                plan = plan.var_length(from, rel.dir, &rel.etypes, min, max, self.path_mode);
                from = node_slot;
            } else if bind {
                let edge_slot = *slots;
                if let Some(rv) = &rel.var {
                    scope.insert(rv.clone(), edge_slot);
                }
                let node_slot = *slots + 1;
                if let Some(v) = v2 {
                    scope.insert(v, node_slot);
                }
                *slots += 2;
                plan = plan.expand_edge(from, rel.dir, &rel.etypes);
                // Inline edge props are a match filter on the bound edge.
                for (k, val) in rel.props {
                    plan = plan.filter(Expr::Compare {
                        op: CompareOp::Eq,
                        left: Box::new(Expr::Prop {
                            slot: edge_slot,
                            key: k,
                        }),
                        right: Box::new(Expr::Lit(val)),
                    });
                }
                // Inline edge `WHERE` — an arbitrary predicate on the bound edge
                // (and any variable bound so far), applied with both the edge and
                // the landing node in scope.
                if let Some(r) = rel.where_range {
                    self.scope = scope.clone();
                    plan = plan.filter(self.parse_captured_where(r)?);
                }
                from = node_slot;
            } else {
                let node_slot = *slots;
                if let Some(v) = v2 {
                    scope.insert(v, node_slot);
                }
                *slots += 1;
                plan = plan.expand(from, rel.dir, &rel.etypes);
                from = node_slot;
            }
            // The landing node's LABEL constrains it (as core does) — a filter on the
            // node's label set, since a landing node has no seed `Scan`.
            if let Some(pred) = landing_label_filter(v2_label, v2_le, from) {
                plan = plan.filter(pred);
            }
            // Inline props on the landing node filter it, exactly as a WHERE would.
            // (`from` is now that node's slot in every branch above.)
            plan = node_prop_filters(plan, from, v2_props);
            // Inline `WHERE` on the landing node — an arbitrary predicate, applied
            // with the node (and everything bound so far) in scope.
            if let Some(r) = v2_where {
                self.scope = scope.clone();
                plan = plan.filter(self.parse_captured_where(r)?);
            }
        }
        Ok(plan)
    }

    /// True when the cursor is at a parenthesized subpath group `((…)…`: a `(`
    /// immediately followed by another `(` (a node pattern always opens with a single
    /// `(`, so `((` can only be a group).
    fn is_subpath_group_start(&self) -> bool {
        matches!(self.peek(), Some(Tok::LParen))
            && matches!(self.toks.get(self.pos + 1), Some(Tok::LParen))
    }

    /// Parse a SINGLE-edge subpath group `((x)-[e:R]->(y)){n,m}` and return its
    /// direction, edge types, and quantifier bounds. The inner variables are group
    /// variables (bound as lists across repetitions in ISO GQL) — this endpoint-only
    /// lowering ignores them, so a case that REFERENCES a group variable downstream
    /// stays unsupported. Inner-node labels/properties, a per-rep `WHERE`, a bound
    /// edge's props, and a multi-hop body are all rejected (later increments).
    fn parse_subpath_group(&mut self) -> Result<(Dir, Vec<String>, u32, u32), String> {
        self.expect(&Tok::LParen)?; // the group's own opening paren
        let bad_inner =
            |n: &ParsedNode| n.1.is_some() || n.4.is_some() || !n.2.is_empty() || n.3.is_some();
        let src = self.node()?;
        if bad_inner(&src) {
            return Err(
                "a label/property/WHERE on a subpath-group inner node is not supported yet".into(),
            );
        }
        let rel = self.rel()?;
        if !rel.props.is_empty() || rel.where_range.is_some() {
            return Err(
                "edge properties / a per-hop WHERE on a subpath group are not supported yet".into(),
            );
        }
        let tgt = self.node()?;
        if bad_inner(&tgt) {
            return Err(
                "a label/property/WHERE on a subpath-group inner node is not supported yet".into(),
            );
        }
        if matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde)) {
            return Err("a multi-hop subpath-group body is not supported yet".into());
        }
        if self.peek_kw("WHERE") {
            return Err("a per-repetition WHERE on a subpath group is not supported yet".into());
        }
        self.expect(&Tok::RParen)?; // close the group
        let (min, max) = self
            .opt_quantifier()?
            .ok_or("a subpath group requires a `{n,m}` / `*` / `+` quantifier")?;
        Ok((rel.dir, rel.etypes, min, max))
    }

    /// `OPTIONAL MATCH (a)-[:R]->(x)` — a LEFT-OUTER single hop from a bound `a`. If
    /// `a` has no matching neighbour, the row is kept with `x` NULL. Single-hop,
    /// node-only, no bound edge; an inner `WHERE` is rejected (it filters the optional
    /// match, which is not yet modelled) rather than mis-applied as a top-level filter.
    fn optional_match(&mut self, plan: Plan) -> Result<Plan, String> {
        if !self.eat_kw("MATCH") {
            return Err("expected MATCH after OPTIONAL".into());
        }
        let (var, label, props, start_where, _le) = self.node()?;
        let Some(v) = var else {
            return Err("OPTIONAL MATCH must start from a bound variable".into());
        };
        if start_where.is_some() {
            return Err("inline WHERE inside OPTIONAL MATCH is not supported yet".into());
        }
        if label.is_some() {
            return Err(format!(
                "bound variable `{v}` cannot be re-labeled in OPTIONAL MATCH"
            ));
        }
        if !props.is_empty() {
            return Err(
                "inline properties on the OPTIONAL MATCH start node are not supported; \
                        use WHERE"
                    .into(),
            );
        }
        let Some(&from) = self.scope.get(&v) else {
            return Err(format!(
                "OPTIONAL MATCH must start from a bound variable; `{v}` is not in scope"
            ));
        };
        let rel = self.rel()?;
        if rel.var.is_some() || !rel.props.is_empty() || rel.where_range.is_some() {
            return Err(
                "a relationship variable / edge properties on OPTIONAL MATCH are not supported"
                    .into(),
            );
        }
        let (v2, _lbl2, v2_props, v2_where, _v2_le) = self.node()?;
        if !v2_props.is_empty() || v2_where.is_some() {
            return Err(
                "inline properties on the OPTIONAL MATCH landing node are not supported; use WHERE"
                    .into(),
            );
        }
        let node_slot = self.slots;
        if let Some(nv) = v2 {
            self.scope.insert(nv, node_slot);
        }
        self.slots += 1;
        if self.peek_kw("WHERE") {
            return Err("WHERE inside OPTIONAL MATCH is not supported yet".into());
        }
        Ok(Plan::OptionalExpand {
            input: Box::new(plan),
            from,
            dir: rel.dir,
            edge_label: rel.etypes,
            // GQL OPTIONAL MATCH lands NULL for a node with no match.
            keep_source: false,
        })
    }

    /// A continuing `MATCH` after `WITH`: it must start from a variable already
    /// carried into scope and extends the working table from that node (rather
    /// than scanning afresh). A fresh/disconnected subsequent pattern — one whose
    /// first node is unbound — is not supported in this subset.
    fn match_continue(&mut self, plan: Plan) -> Result<Plan, String> {
        let (var, label, props, start_where, _le) = self.node()?;
        let Some(v) = var else {
            return Err("a MATCH after WITH must start from a bound variable".into());
        };
        if start_where.is_some() {
            return Err(
                "inline WHERE on a continuing MATCH's start variable is not supported; use WHERE"
                    .into(),
            );
        }
        if label.is_some() {
            return Err(format!(
                "bound variable `{v}` cannot be re-labeled in a continuing MATCH"
            ));
        }
        if !props.is_empty() {
            return Err(format!(
                "bound variable `{v}` cannot be re-constrained with inline properties in a \
                 continuing MATCH; use WHERE"
            ));
        }
        let Some(&from) = self.scope.get(&v) else {
            return Err(format!(
                "continuing MATCH must start from a carried variable; `{v}` is not in scope"
            ));
        };
        // Move scope/slots out so `extend_chain` can borrow them while it also
        // borrows `self` (the parser cursor); restore them afterwards.
        let mut scope = std::mem::take(&mut self.scope);
        let mut slots = self.slots;
        let mut plan = self.extend_chain(plan, &mut scope, &mut slots, from)?;
        self.scope = scope;
        self.slots = slots;
        if self.eat_kw("WHERE") {
            plan = plan.filter(self.expr()?);
        }
        Ok(plan)
    }

    /// A `WITH` boundary: project/aggregate the working table (exactly as `RETURN`
    /// would), then rebind scope so the carried output columns are a fresh slot
    /// space (`name -> column index`) for the following part. `ORDER BY/SKIP/LIMIT`
    /// ride the projection; a trailing `WHERE` is a post-projection (HAVING)
    /// filter, matching lenke-core's `WITH … WHERE`.
    fn with_clause(&mut self, plan: Plan) -> Result<Plan, String> {
        let distinct = self.eat_kw("DISTINCT");
        let items = self.return_items()?;
        let (mut plan, out_names) = apply_items(plan, &items);
        if distinct {
            plan = plan.distinct();
        }
        let mut scope = HashMap::new();
        for (i, name) in out_names.iter().enumerate() {
            scope.insert(name.clone(), i);
        }
        self.scope = scope;
        self.slots = out_names.len();
        let keys = if self.eat_kw("ORDER") {
            if !self.eat_kw("BY") {
                return Err("expected BY after ORDER".into());
            }
            self.sort_keys(&out_names)?
        } else {
            Vec::new()
        };
        // `OFFSET` is the ISO spelling of `SKIP` — a synonym here (core accepts both).
        let skip = if self.eat_kw("SKIP") || self.eat_kw("OFFSET") {
            Some(self.usize_lit()?)
        } else {
            None
        };
        let limit = if self.eat_kw("LIMIT") {
            Some(self.usize_lit()?)
        } else {
            None
        };
        if !keys.is_empty() || skip.is_some() || limit.is_some() {
            plan = plan.order_page(keys, skip, limit);
        }
        if self.eat_kw("WHERE") {
            plan = plan.filter(self.expr()?);
        }
        Ok(plan)
    }

    /// An inline correlated subquery `CALL (scope) { MATCH … [WHERE] RETURN … }`.
    /// The subquery imports only the named `scope` variables, continues its pattern
    /// from one of them, and its `RETURN` columns are appended to each outer row
    /// (a lateral join). The named-procedure form (`CALL name(cfg) YIELD …`) is
    /// deferred to the algorithms phase — its catalog is those procedures.
    fn call_inline(&mut self, plan: Plan) -> Result<Plan, String> {
        let outer_width = self.slots;
        // The inline form opens with a `(scope)`; anything else (a bare name) is
        // the deferred named-procedure call.
        if !self.eat(&Tok::LParen) {
            return Err("only the inline `CALL (scope) { … }` form is supported; \
                        named-procedure CALL is deferred to the algorithms phase"
                .into());
        }
        let mut scope_vars: Vec<String> = Vec::new();
        if self.peek() != Some(&Tok::RParen) {
            loop {
                scope_vars.push(self.ident()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen)?;
        if scope_vars.is_empty() {
            return Err("CALL (…) needs at least one scope variable to correlate on".into());
        }
        for v in &scope_vars {
            if !self.scope.contains_key(v) {
                return Err(format!("CALL scope variable `{v}` is not bound"));
            }
        }

        self.expect(&Tok::LBrace)?;
        if !self.eat_kw("MATCH") {
            return Err("a CALL subquery must begin with MATCH".into());
        }
        let (var, label, props, start_where, _le) = self.node()?;
        let Some(v) = var else {
            return Err("a CALL subquery pattern must start from a scope variable".into());
        };
        if start_where.is_some() {
            return Err("inline WHERE on a CALL subquery start variable is not supported".into());
        }
        if label.is_some() {
            return Err(format!(
                "scope variable `{v}` cannot be re-labeled inside a CALL subquery"
            ));
        }
        if !props.is_empty() {
            return Err(format!(
                "scope variable `{v}` cannot be re-constrained with inline properties inside a \
                 CALL subquery; use WHERE"
            ));
        }
        if !scope_vars.contains(&v) {
            return Err(format!(
                "a CALL subquery must start from a scope variable; `{v}` is not in scope"
            ));
        }
        let from = self.scope[&v];
        // The sub-scope imports ONLY the declared scope variables (at their outer
        // slots); body variables append from `outer_width` onward.
        let mut sub_scope: HashMap<String, usize> = scope_vars
            .iter()
            .map(|s| (s.clone(), self.scope[s]))
            .collect();
        let mut sub_slots = outer_width;
        let body = self.extend_chain(Plan::Row, &mut sub_scope, &mut sub_slots, from)?;

        // Parse WHERE/RETURN against the sub-scope. A parse error discards the
        // whole parser, so there is no need to restore scope on the error paths.
        let outer_scope = std::mem::replace(&mut self.scope, sub_scope);
        self.slots = sub_slots;
        let body = if self.eat_kw("WHERE") {
            body.filter(self.expr()?)
        } else {
            body
        };
        if !self.eat_kw("RETURN") {
            return Err("a CALL subquery needs a RETURN".into());
        }
        let items = self.return_items()?;
        if items.iter().any(RetItem::has_agg) {
            return Err("an aggregating RETURN inside CALL { … } is not supported".into());
        }
        let yields: Vec<(String, Expr)> = items
            .into_iter()
            .map(|it| match it {
                RetItem::Key(name, e) => (name, e),
                RetItem::Agg(_) | RetItem::AggExpr { .. } => {
                    unreachable!("aggregates rejected above")
                }
            })
            .collect();
        self.expect(&Tok::RBrace)?;

        // Restore the outer scope and bind the yields as its new trailing columns;
        // the subquery's internal variables do not survive.
        self.scope = outer_scope;
        for (i, (name, _)) in yields.iter().enumerate() {
            self.scope.insert(name.clone(), outer_width + i);
        }
        self.slots = outer_width + yields.len();
        Ok(Plan::CallInline {
            input: Box::new(plan),
            body: Box::new(body),
            yields,
            outer_width,
        })
    }

    /// A top-level named-procedure call `CALL name(config) [YIELD col [AS a], …]`.
    /// The procedure (a built-in graph algorithm) produces `[node, <result>]`; a
    /// YIELD selects/renames those columns (default = both). The result flows on to
    /// any following clause (RETURN/WITH/…), or the CALL is a complete statement.
    fn call_procedure(&mut self) -> Result<Plan, String> {
        let name = self.ident()?;
        let result_col = crate::algo::procedure_result_col(&name)
            .ok_or_else(|| format!("unknown procedure `{name}`"))?;
        self.expect(&Tok::LParen)?;
        let config = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.props()?
        } else {
            Vec::new()
        };
        self.expect(&Tok::RParen)?;
        // The procedure's output columns: node at slot 0, result at slot 1.
        self.scope = HashMap::from([("node".to_string(), 0), (result_col.to_string(), 1)]);
        self.slots = 2;

        let items: Vec<(String, Expr)> = if self.eat_kw("YIELD") {
            let mut ys = Vec::new();
            loop {
                let col = self.ident()?;
                let slot = *self
                    .scope
                    .get(&col)
                    .ok_or_else(|| format!("YIELD `{col}` is not a procedure output"))?;
                let alias = if self.eat_kw("AS") {
                    self.ident()?
                } else {
                    col
                };
                ys.push((alias, Expr::Slot(slot)));
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            ys
        } else {
            vec![
                ("node".into(), Expr::Slot(0)),
                (result_col.to_string(), Expr::Slot(1)),
            ]
        };
        // Rebind scope to the (aliased) yielded columns for any following clause.
        self.scope = items
            .iter()
            .enumerate()
            .map(|(i, (n, _))| (n.clone(), i))
            .collect();
        self.slots = items.len();
        let plan = Plan::CallProcedure { name, config }.project(items);
        if self.pos == self.toks.len() {
            Ok(plan)
        } else {
            self.query_tail(plan)
        }
    }

    /// An optional `{n}` / `{n,m}` / `{n,}` quantifier after a relationship. An
    /// open upper bound is capped at `MAX_VARLEN` hops for this subset.
    fn opt_quantifier(&mut self) -> Result<Option<(u32, u32)>, String> {
        // Abbreviations: `+` == `{1,}` (one or more), `*` == `{0,}` (zero or more) —
        // the ISO shorthands for an unbounded quantifier. In this position (after a
        // relationship, before a node) `+`/`*` are unambiguously quantifiers.
        if self.eat(&Tok::Plus) {
            return Ok(Some((1, MAX_VARLEN)));
        }
        if self.eat(&Tok::Star) {
            return Ok(Some((0, MAX_VARLEN)));
        }
        if !self.eat(&Tok::LBrace) {
            return Ok(None);
        }
        let min = self.u32_lit()?;
        let max = if self.eat(&Tok::Comma) {
            if matches!(self.peek(), Some(Tok::Num(_))) {
                self.u32_lit()?
            } else {
                MAX_VARLEN // `{n,}` — open upper bound, capped
            }
        } else {
            min // `{n}` — exact
        };
        self.expect(&Tok::RBrace)?;
        if min > max {
            return Err(format!("quantifier {{{min},{max}}} has min > max"));
        }
        Ok(Some((min, max)))
    }

    fn u32_lit(&mut self) -> Result<u32, String> {
        match self.bump() {
            Some(Tok::Num(n)) if n >= 0.0 && n.fract() == 0.0 => Ok(n as u32),
            other => Err(format!("expected a non-negative integer, got {other:?}")),
        }
    }

    // node := '(' [var] [':' Label] ')'
    fn node(&mut self) -> Result<ParsedNode, String> {
        self.expect(&Tok::LParen)?;
        // An identifier here is the node variable; the label (if any) follows ':'.
        let var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident()?)
        } else {
            None
        };
        // The label: `:LabelExpr` or the `IS LabelExpr` introducer. A bare positive
        // label is the seed `Scan` label; a compound expression (`:A&B`, `:A|B`,
        // `:!A`, `:%`) yields a residual per-slot filter (`label_expr`).
        let (label, label_expr) = if self.eat(&Tok::Colon) || self.eat_kw("IS") {
            let le = self.parse_label_expr()?;
            (le.seed_label(), le.needs_filter())
        } else {
            (None, None)
        };
        // Inline property map `(n:Label {k: v, …})`: a match filter on the node,
        // the exact equivalent of `WHERE n.k = v AND …`. It lowers to the SAME `Eq`
        // filters (see `node_prop_filters`), so the two spellings optimize to the
        // same plan and seed the same index — equivalent spellings cost the same.
        let props = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.props()?
        } else {
            Vec::new()
        };
        // Inline `WHERE pred` after the label/props: an arbitrary predicate on the
        // node, equivalent to a trailing `WHERE`. Captured now, parsed once the
        // node's slot exists (see `capture_inline_where`).
        let where_range = if self.eat_kw("WHERE") {
            Some(self.capture_inline_where())
        } else {
            None
        };
        self.expect(&Tok::RParen)?;
        Ok((var, label, props, where_range, label_expr))
    }

    /// Parse a boolean label expression: `factor ('&' factor)* ('|' …)*` with `!`
    /// negation and `%` wildcard, `!` binding tighter than `&` tighter than `|`.
    fn parse_label_expr(&mut self) -> Result<LabelExpr, String> {
        let mut left = self.label_and()?;
        while self.eat(&Tok::Pipe) {
            left = LabelExpr::Or(Box::new(left), Box::new(self.label_and()?));
        }
        Ok(left)
    }

    fn label_and(&mut self) -> Result<LabelExpr, String> {
        let mut left = self.label_factor()?;
        while self.eat(&Tok::Amp) {
            left = LabelExpr::And(Box::new(left), Box::new(self.label_factor()?));
        }
        Ok(left)
    }

    fn label_factor(&mut self) -> Result<LabelExpr, String> {
        if self.eat(&Tok::Bang) {
            return Ok(LabelExpr::Not(Box::new(self.label_factor()?)));
        }
        if self.eat(&Tok::Percent) {
            return Ok(LabelExpr::Any);
        }
        if self.eat(&Tok::LParen) {
            let e = self.parse_label_expr()?;
            self.expect(&Tok::RParen)?;
            return Ok(e);
        }
        Ok(LabelExpr::Label(self.ident()?))
    }

    // rel := '-' '[' ':' R ']' '->'   (out)
    //      | '-' '[' ':' R ']' '-'    (both)
    //      | '<-' '[' ':' R ']' '-'   (in)
    // rel := ('-' | '~' | '<-') '[' [var] ':' Type [ '{' props '}' ] ']' ('->' | '-' | '~')
    // Captures an optional relationship VARIABLE and inline edge PROPERTIES. `~` is
    // the undirected delimiter: like `-`, it carries NO direction, so `~[...]~`
    // (and any `-`/`~` mix) is `Dir::Both`, exactly as core resolves it.
    fn rel(&mut self) -> Result<Rel, String> {
        let incoming = self.eat(&Tok::LArrow);
        if !incoming && !self.eat(&Tok::Minus) && !self.eat(&Tok::Tilde) {
            return Err(format!("expected `-`, `~`, or `<-` at token {}", self.pos));
        }
        self.expect(&Tok::LBracket)?;
        let var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident()?)
        } else {
            None
        };
        // `:Type` is OPTIONAL — `-[r]->` / `-[]->` is an UNTYPED hop (any edge type),
        // matching core's bracketed untyped relationship. (Core's BARE `-->` has
        // different semantics — it matches nothing — so it is deliberately NOT
        // accepted here, to avoid a silent result divergence.)
        // `:Type` with an optional `|`-disjunction (`:A|B|C`) — an edge matches if
        // its type is ANY of them. Empty = untyped (any type).
        let mut etypes = Vec::new();
        if self.eat(&Tok::Colon) {
            etypes.push(self.ident()?);
            while self.eat(&Tok::Pipe) {
                etypes.push(self.ident()?);
            }
        }
        let props = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.props()?
        } else {
            Vec::new()
        };
        // Inline `WHERE pred` on the edge, equivalent to a trailing `WHERE`.
        let where_range = if self.eat_kw("WHERE") {
            Some(self.capture_inline_where())
        } else {
            None
        };
        self.expect(&Tok::RBracket)?;
        let dir = if incoming {
            if !self.eat(&Tok::Minus) && !self.eat(&Tok::Tilde) {
                return Err(format!(
                    "expected `-` or `~` to close a relationship at token {}",
                    self.pos
                ));
            }
            Dir::In
        } else if self.eat(&Tok::RArrow) {
            Dir::Out
        } else if self.eat(&Tok::Minus) || self.eat(&Tok::Tilde) {
            Dir::Both
        } else {
            return Err(format!("malformed relationship at token {}", self.pos));
        };
        Ok(Rel {
            dir,
            etypes,
            var,
            props,
            where_range,
        })
    }

    // items := item ( ',' item )*
    // item  := aggregate [AS name] | expr [AS name]
    fn return_items(&mut self) -> Result<Vec<RetItem>, String> {
        let mut items = Vec::new();
        loop {
            // `*` expands to EVERY bound variable, projected in slot (declaration)
            // order — each column named after its variable. Composes with more items
            // (`RETURN *, count(*)`).
            if self.eat(&Tok::Star) {
                let mut vars: Vec<(usize, String)> =
                    self.scope.iter().map(|(k, &s)| (s, k.clone())).collect();
                vars.sort_by_key(|&(s, _)| s);
                if vars.is_empty() {
                    return Err("RETURN * requires at least one bound variable".into());
                }
                for (slot, name) in vars {
                    items.push(RetItem::Key(name, Expr::Slot(slot)));
                }
                if !self.eat(&Tok::Comma) {
                    break;
                }
                continue;
            }
            let idx = items.len();
            let item = if let Some(func) = self.peek_agg() {
                let save = self.pos;
                let (agg_arg, distinct, frac) = self.aggregate_call()?;
                // A bare aggregate is followed by `AS`/`,`/a tail keyword; an operator
                // means the aggregate is embedded in a larger expression (`count(*)+1`),
                // which we re-parse with aggregate hoisting into a `RetItem::AggExpr`.
                if is_operator_continuation(self.peek()) {
                    self.pos = save;
                    self.having_base = AGG_SLOT_BASE;
                    self.having_aggs = Some(Vec::new());
                    let expr = self.expr()?;
                    let aggs = self.having_aggs.take().expect("hoisting");
                    let name = if self.eat_kw("AS") {
                        self.ident()?
                    } else {
                        self.item_name(&expr, idx)
                    };
                    RetItem::AggExpr { name, expr, aggs }
                } else {
                    let name = if self.eat_kw("AS") {
                        self.ident()?
                    } else {
                        format!("col{idx}")
                    };
                    RetItem::Agg(crate::ir::Agg {
                        func,
                        arg: agg_arg,
                        distinct,
                        name,
                        frac,
                    })
                }
            } else {
                let e = self.expr()?;
                let name = if self.eat_kw("AS") {
                    self.ident()?
                } else {
                    self.item_name(&e, idx)
                };
                RetItem::Key(name, e)
            };
            items.push(item);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(items)
    }

    /// If the next tokens are `aggName (` return the aggregate function.
    fn peek_agg(&self) -> Option<AggFn> {
        if let Some(Tok::Ident(s)) = self.peek() {
            if self.toks.get(self.pos + 1) == Some(&Tok::LParen) {
                return agg_fn(s);
            }
        }
        None
    }

    // aggregate_call := aggName '(' ( '*' | [DISTINCT] expr ) ')'
    // returns (arg, distinct); arg is None only for `count(*)`.
    fn aggregate_call(&mut self) -> Result<(Option<Expr>, bool, Option<f64>), String> {
        self.pos += 1; // the aggregate name (already validated by peek_agg)
        self.expect(&Tok::LParen)?;
        if self.eat(&Tok::Star) {
            self.expect(&Tok::RParen)?;
            return Ok((None, false, None));
        }
        let distinct = self.eat_kw("DISTINCT");
        let arg = self.expr()?;
        // A second argument is the ordered-set fraction of
        // `percentile_cont(x, f)`/`percentile_disc(x, f)` — a constant in [0, 1].
        let frac = if self.eat(&Tok::Comma) {
            match self.expr()? {
                Expr::Lit(Value::Num(f)) => Some(f),
                _ => return Err("percentile fraction must be a numeric constant".into()),
            }
        } else {
            None
        };
        self.expect(&Tok::RParen)?;
        Ok((Some(arg), distinct, frac))
    }

    // Expression precedence: OR < AND < NOT < comparison < primary.
    fn expr(&mut self) -> Result<Expr, String> {
        self.or_expr()
    }

    fn or_expr(&mut self) -> Result<Expr, String> {
        // OR and XOR share one left-associative precedence level (ISO), above AND.
        // Binary left-nesting here is equivalent to core's flatten-same/nest-on-
        // switch: `a OR b XOR c` parses as `(a OR b) XOR c`.
        let mut left = self.and_expr()?;
        loop {
            if self.eat_kw("OR") {
                let right = self.and_expr()?;
                left = Expr::Or(Box::new(left), Box::new(right));
            } else if self.eat_kw("XOR") {
                let right = self.and_expr()?;
                left = Expr::Xor(Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.not_expr()?;
        while self.eat_kw("AND") {
            let right = self.not_expr()?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> Result<Expr, String> {
        if self.eat_kw("NOT") {
            Ok(Expr::Not(Box::new(self.not_expr()?)))
        } else {
            self.cmp_expr()
        }
    }

    fn cmp_expr(&mut self) -> Result<Expr, String> {
        // Comparison operands are concat expressions (`||` binds tighter than a
        // comparison, looser than `+`/`-` — the ISO precedence core uses).
        let left = self.concat_expr()?;
        // Postfix `IS [NOT] NULL` — a definite null test, checked before the
        // binary comparison operators (a value is one or the other, not both).
        if self.eat_kw("IS") {
            let negated = self.eat_kw("NOT");
            // `IS [NOT] NULL | UNKNOWN | TRUE | FALSE`. UNKNOWN == NULL for a boolean
            // 3VL value. `x IS TRUE` is TRUE iff x is present and true — exactly
            // `coalesce(x = true, false)` (a NULL/non-bool x → false); `IS NOT TRUE`
            // negates it. All desugar to existing exprs (no new node).
            if self.eat_kw("NULL") || self.peek_kw("UNKNOWN") {
                self.eat_kw("UNKNOWN");
                return Ok(Expr::IsNull {
                    expr: Box::new(left),
                    negated,
                });
            }
            // `x IS [NOT] TYPED <value type> [NOT NULL]` — the ISO value-type
            // predicate. Desugars to `__is_typed(x, '<category>', <not_null>)`; a NULL
            // value conforms to any nullable type. Only the scalar/record/list
            // categories are handled — temporal and closed-record-schema types
            // deliberately parse-error (left to the general path / baseline).
            if self.eat_kw("TYPED") {
                let category = self.value_type_category()?;
                let not_null = if self.peek_kw("NOT") {
                    // `NOT NULL` modifier — but not if this `NOT` starts another clause.
                    let save = self.pos;
                    self.eat_kw("NOT");
                    if self.eat_kw("NULL") {
                        true
                    } else {
                        self.pos = save;
                        false
                    }
                } else {
                    false
                };
                let call = Expr::Call {
                    name: "__is_typed".into(),
                    args: vec![
                        left,
                        Expr::Lit(Value::Str(category.into())),
                        Expr::Lit(Value::Bool(not_null)),
                    ],
                };
                return Ok(if negated {
                    Expr::Not(Box::new(call))
                } else {
                    call
                });
            }
            // `x IS [NOT] LABELED <label>` — a definite label-set membership test,
            // the keyword form of the `x:Label` predicate. Desugars to
            // `'<label>' IN labels(x)` (the label set is never null, so `IS NOT
            // LABELED` is a plain negation).
            if self.eat_kw("LABELED") {
                let label = self.ident()?;
                let pred = Expr::In {
                    needle: Box::new(Expr::Lit(Value::Str(label.into()))),
                    haystack: Box::new(Expr::Call {
                        name: "labels".into(),
                        args: vec![left],
                    }),
                };
                return Ok(if negated {
                    Expr::Not(Box::new(pred))
                } else {
                    pred
                });
            }
            let want = if self.eat_kw("TRUE") {
                true
            } else if self.eat_kw("FALSE") {
                false
            } else {
                return Err("expected NULL, UNKNOWN, TRUE, or FALSE after IS [NOT]".into());
            };
            let is = Expr::Call {
                name: "coalesce".into(),
                args: vec![
                    Expr::Compare {
                        op: CompareOp::Eq,
                        left: Box::new(left),
                        right: Box::new(Expr::Lit(Value::Bool(want))),
                    },
                    Expr::Lit(Value::Bool(false)),
                ],
            };
            return Ok(if negated { Expr::Not(Box::new(is)) } else { is });
        }
        // `left [NOT] IN <list literal>` — desugars to an OR-chain of equality
        // tests, so its three-valued behavior falls out of the `=` operator (a
        // NULL element or operand makes a non-match UNKNOWN, not false).
        let saved = self.pos;
        let negated_in = !self.suppress_in && self.eat_kw("NOT");
        if !self.suppress_in && self.eat_kw("IN") {
            let rhs = self.concat_expr()?;
            // A list LITERAL desugars to an OR-chain (more optimizable); any other
            // list expression (a property, param, function result) uses the runtime
            // `Expr::In`. Both are three-valued identically.
            let member = match rhs {
                Expr::List { items } => in_chain(&left, items),
                haystack => Expr::In {
                    needle: Box::new(left),
                    haystack: Box::new(haystack),
                },
            };
            return Ok(if negated_in {
                Expr::Not(Box::new(member))
            } else {
                member
            });
        }
        self.pos = saved; // the NOT (if any) was not part of a NOT IN
                          // String infix predicates `CONTAINS` / `STARTS WITH` / `ENDS WITH` desugar
                          // to their scalar functions — three-valued (a NULL/non-string operand → NULL).
        let str_fn = if self.eat_kw("CONTAINS") {
            Some("contains")
        } else if self.eat_kw("STARTS") {
            if !self.eat_kw("WITH") {
                return Err("expected WITH after STARTS".into());
            }
            Some("starts_with")
        } else if self.eat_kw("ENDS") {
            if !self.eat_kw("WITH") {
                return Err("expected WITH after ENDS".into());
            }
            Some("ends_with")
        } else {
            None
        };
        if let Some(name) = str_fn {
            let right = self.concat_expr()?;
            return Ok(Expr::Call {
                name: name.to_string(),
                args: vec![left, right],
            });
        }
        let op = match self.peek() {
            Some(Tok::Eq) => CompareOp::Eq,
            Some(Tok::Ne) => CompareOp::Ne,
            Some(Tok::Lt) => CompareOp::Lt,
            Some(Tok::Le) => CompareOp::Le,
            Some(Tok::Gt) => CompareOp::Gt,
            Some(Tok::Ge) => CompareOp::Ge,
            _ => return Ok(left),
        };
        self.pos += 1;
        let right = self.concat_expr()?;
        Ok(Expr::Compare {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    // concat_expr := add_expr ( '||' add_expr )*  — string/list concatenation, a
    // level between comparison and additive (the ISO precedence). A run of `||`
    // folds into ONE n-ary `concat(...)` call (matching core's flat Concat node), so
    // the null-propagation and js-string coercion live in the `concat` scalar fn.
    fn concat_expr(&mut self) -> Result<Expr, String> {
        let first = self.add_expr()?;
        if !matches!(self.peek(), Some(Tok::Concat)) {
            return Ok(first);
        }
        let mut args = vec![first];
        while self.eat(&Tok::Concat) {
            args.push(self.add_expr()?);
        }
        Ok(Expr::Call {
            name: "concat".to_string(),
            args,
        })
    }

    // add_expr := mul_expr ( ('+' | '-') mul_expr )*
    fn add_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.mul_expr()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => crate::ir::ArithOp::Add,
                Some(Tok::Minus) => crate::ir::ArithOp::Sub,
                _ => break,
            };
            self.pos += 1;
            let right = self.mul_expr()?;
            left = Expr::Arith {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    // mul_expr := unary ( ('*' | '/' | '%') unary )*
    fn mul_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => crate::ir::ArithOp::Mul,
                Some(Tok::Slash) => crate::ir::ArithOp::Div,
                Some(Tok::Percent) => crate::ir::ArithOp::Rem,
                _ => break,
            };
            self.pos += 1;
            let right = self.unary()?;
            left = Expr::Arith {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    // unary := '-' unary | primary. Unary minus desugars to `0 - x` so null/
    // non-numeric propagation is the ordinary Arith rule.
    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat(&Tok::Minus) {
            let e = self.unary()?;
            Ok(Expr::Arith {
                op: crate::ir::ArithOp::Sub,
                left: Box::new(Expr::Lit(Value::Num(0.0))),
                right: Box::new(e),
            })
        } else {
            self.primary()
        }
    }

    // primary := '(' expr ')' | literal | var '.' key | var
    fn primary(&mut self) -> Result<Expr, String> {
        // `LET name = expr [, name = expr]* IN body END` — local bindings. Each
        // binding is pushed onto `self.lets` (later bindings may reference earlier
        // ones), the body is parsed with them in scope (a reference inlines the bound
        // expr), then they are popped. The body IS the substituted expression.
        if self.peek_kw("LET") {
            self.pos += 1; // LET
            let base = self.lets.len();
            loop {
                let name = self.ident()?;
                self.expect(&Tok::Eq)?;
                // The binding value's top-level `IN` is the LET separator, not
                // membership (a parenthesized `IN` is restored inside `primary`).
                let saved = self.suppress_in;
                self.suppress_in = true;
                let val = self.expr()?;
                self.suppress_in = saved;
                self.lets.push((name, val));
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            if !self.eat_kw("IN") {
                return Err("expected IN after LET bindings".into());
            }
            let body = self.expr()?;
            if !self.eat_kw("END") {
                return Err("expected END to close LET … IN".into());
            }
            self.lets.truncate(base);
            return Ok(body);
        }
        match self.peek().cloned() {
            Some(Tok::LParen) => {
                self.pos += 1;
                // A parenthesized group restores normal `IN`-membership even inside a
                // LET binding value (`LET x = (a IN [1,2]) IN …`).
                let saved = self.suppress_in;
                self.suppress_in = false;
                let e = self.expr()?;
                self.suppress_in = saved;
                self.expect(&Tok::RParen)?;
                self.field_chain(e)
            }
            Some(Tok::LBracket) => {
                // A list literal `[a, b, …]` (empty `[]` allowed). In expression
                // position `[` always starts a list — rel-pattern brackets only
                // occur inside `-[:R]-`, which the pattern parser handles.
                self.pos += 1;
                let mut items = Vec::new();
                if self.peek() != Some(&Tok::RBracket) {
                    loop {
                        items.push(self.expr()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RBracket)?;
                self.field_chain(Expr::List { items })
            }
            Some(Tok::LBrace) => {
                // A record literal `{k: expr, …}` (empty `{}` allowed). In
                // expression position `{` always starts a record — inline node
                // props `(a {k: v})` are handled by the pattern parser, not here.
                self.pos += 1;
                let mut fields = Vec::new();
                if self.peek() != Some(&Tok::RBrace) {
                    loop {
                        let key = self.ident()?;
                        self.expect(&Tok::Colon)?;
                        fields.push((key, self.expr()?));
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RBrace)?;
                self.field_chain(Expr::Record { fields })
            }
            Some(Tok::Num(n)) => {
                self.pos += 1;
                Ok(Expr::Lit(Value::Num(n)))
            }
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Expr::Lit(Value::Str(s.into())))
            }
            Some(Tok::Ident(s)) => {
                self.pos += 1;
                // keyword literals first
                if s.eq_ignore_ascii_case("true") {
                    return Ok(Expr::Lit(Value::Bool(true)));
                }
                if s.eq_ignore_ascii_case("false") {
                    return Ok(Expr::Lit(Value::Bool(false)));
                }
                if s.eq_ignore_ascii_case("null") {
                    return Ok(Expr::Lit(Value::Null));
                }
                // Searched CASE.
                if s.eq_ignore_ascii_case("case") {
                    return self.case_expr();
                }
                // CAST(<expr> AS <TYPE>) — parsed before the generic call form
                // because of its `AS TYPE` tail.
                if s.eq_ignore_ascii_case("cast") {
                    return self.cast_expr();
                }
                // PROPERTY_EXISTS(<var>, <key>) — the second arg is a bare
                // property NAME, not an expression, so it can't ride `call`.
                if s.eq_ignore_ascii_case("property_exists") {
                    return self.property_exists_expr();
                }
                // EXISTS { <pattern> [WHERE <pred>] } — a correlated subquery, so
                // it takes `{ … }`, not the `(args)` of a scalar call.
                if s.eq_ignore_ascii_case("exists") {
                    return self.exists_expr();
                }
                // COUNT { <pattern> } — the correlated count subquery (braces), as
                // opposed to the `count(*)`/`count(x)` aggregate (parens).
                if s.eq_ignore_ascii_case("count") && matches!(self.peek(), Some(Tok::LBrace)) {
                    return self.count_subquery_expr();
                }
                // Two-word zoned literal `ZONED TIME '…'` / `ZONED DATETIME '…'`.
                if s.eq_ignore_ascii_case("zoned") {
                    if let Some(Tok::Ident(kind)) = self.peek().cloned() {
                        let ztag = match kind.to_ascii_uppercase().as_str() {
                            "TIME" => Some("zoned_time"),
                            "DATETIME" => Some("zoned_datetime"),
                            _ => None,
                        };
                        if let Some(ztag) = ztag {
                            if let Some(Tok::Str(lit)) = self.toks.get(self.pos + 1).cloned() {
                                self.pos += 2; // consume TIME/DATETIME and the string
                                let t = crate::temporal::Temporal::parse(ztag, &lit)
                                    .map_err(|e| format!("invalid ZONED {kind} literal: {e}"))?;
                                return Ok(Expr::Lit(Value::Temporal(t)));
                            }
                        }
                    }
                }
                // Typed temporal literal `DATE '…'` / `TIME '…'` / `DATETIME '…'` /
                // `DURATION '…'`: a temporal keyword directly followed by a string.
                // (A bare `date` not followed by a string stays an ordinary
                // variable/property.)
                if let Some(tag) = temporal_tag(&s) {
                    if let Some(Tok::Str(lit)) = self.peek() {
                        let lit = lit.clone();
                        self.pos += 1;
                        let t = crate::temporal::Temporal::parse(tag, &lit)
                            .map_err(|e| format!("invalid {s} literal: {e}"))?;
                        return Ok(Expr::Lit(Value::Temporal(t)));
                    }
                }
                // Inside a HAVING predicate, an aggregate call in expression position
                // (`count(*) > 1`) is hoisted into `having_aggs` and replaced by a
                // reference to its post-aggregation output column.
                if self.having_aggs.is_some() && self.peek() == Some(&Tok::LParen) {
                    if let Some(func) = agg_fn(&s) {
                        return self.hoist_having_agg(func);
                    }
                }
                // A scalar function call `name(args…)`. (Aggregates are handled in
                // return_items, never reached here.) A call may be subscripted /
                // field-accessed (`edges(p)[0].w`), so route through `field_chain`.
                if self.peek() == Some(&Tok::LParen) {
                    let call = self.call(&s)?;
                    return self.field_chain(call);
                }
                // A path variable resolves to the current row's path (lineage),
                // not a slot — there is exactly one path per row.
                if self.path_vars.contains(&s) {
                    return Ok(Expr::Path);
                }
                // A `LET`-bound local name inlines its bound expression (innermost
                // binding wins), before any graph-variable resolution.
                if let Some((_, e)) = self.lets.iter().rev().find(|(n, _)| n == &s) {
                    return Ok(e.clone());
                }
                let slot = *self
                    .scope
                    .get(&s)
                    .ok_or_else(|| format!("unknown variable `{s}`"))?;
                // `x:LabelExpr` — a boolean label predicate in expression position
                // (`WHERE x:Person`, `WHERE x:Person|Software`, `x:A&B`, `x:!A`).
                // Lowered via the shared label-expression lowering to a predicate
                // over `labels(x)` (a single label is `'Label' IN labels(x)`).
                if self.eat(&Tok::Colon) {
                    let le = self.parse_label_expr()?;
                    return Ok(lower_label_expr(&le, slot));
                }
                if self.eat(&Tok::Dot) {
                    // The FIRST `.key` stays a `Prop` (the shape the optimizer
                    // seeks on); any further `.key` are record-field accesses on
                    // the value it produced (e.g. `n.meta.city`).
                    let key = self.ident()?;
                    self.field_chain(Expr::Prop { slot, key })
                } else {
                    Ok(Expr::Slot(slot))
                }
            }
            other => Err(format!("expected an expression, got {other:?}")),
        }
    }

    /// Consume trailing `.field` accessors on a non-variable base (a record/paren
    /// expression), building nested `Expr::Field`. (A bare variable handles its
    /// own single `.prop` in `primary`, keeping that the optimizer's `Prop` shape.)
    fn field_chain(&mut self, mut base: Expr) -> Result<Expr, String> {
        loop {
            if self.eat(&Tok::Dot) {
                let key = self.ident()?;
                base = Expr::Field {
                    base: Box::new(base),
                    key,
                };
            } else if self.eat(&Tok::LBracket) {
                // Subscript `base[index]` — a list element or a record/map field.
                let index = self.expr()?;
                self.expect(&Tok::RBracket)?;
                base = Expr::Index {
                    base: Box::new(base),
                    index: Box::new(index),
                };
            } else {
                break;
            }
        }
        Ok(base)
    }

    // case := CASE [subject] (WHEN expr THEN expr)+ [ELSE expr] END
    // WHEN/THEN/ELSE/END are contextual keywords. The SEARCHED form has no subject
    // (`WHEN <cond>`); the SIMPLE form has one (`CASE <e> WHEN <v>`), which desugars
    // to searched `WHEN e = v THEN …`. A NULL subject makes every `e = v` UNKNOWN, so
    // no branch matches and it falls to ELSE — 3VL, matching core.
    fn case_expr(&mut self) -> Result<Expr, String> {
        let subject = if self.peek_kw("WHEN") {
            None
        } else {
            Some(self.expr()?)
        };
        if !self.peek_kw("WHEN") {
            return Err("expected WHEN in CASE".into());
        }
        let mut branches = Vec::new();
        while self.eat_kw("WHEN") {
            let when = self.expr()?;
            if !self.eat_kw("THEN") {
                return Err("expected THEN in CASE".into());
            }
            let cond = match &subject {
                None => when,
                Some(subj) => Expr::Compare {
                    op: CompareOp::Eq,
                    left: Box::new(subj.clone()),
                    right: Box::new(when),
                },
            };
            branches.push((cond, self.expr()?));
        }
        let otherwise = if self.eat_kw("ELSE") {
            Some(Box::new(self.expr()?))
        } else {
            None
        };
        if !self.eat_kw("END") {
            return Err("expected END to close CASE".into());
        }
        Ok(Expr::Case {
            branches,
            otherwise,
        })
    }

    // cast := CAST '(' expr AS TYPE ')'. The engine has one numeric type, so
    // INTEGER and its aliases map to `Integer` (truncating) and the float/number
    // spellings to `Float`; the coercion table itself lives in `value::cast`.
    fn cast_expr(&mut self) -> Result<Expr, String> {
        self.expect(&Tok::LParen)?;
        let expr = Box::new(self.expr()?);
        if !self.eat_kw("AS") {
            return Err("expected AS in CAST(expr AS TYPE)".into());
        }
        let ty = self.ident()?;
        let target = match ty.to_ascii_uppercase().as_str() {
            "INTEGER" | "INT" => CastTarget::Integer,
            "FLOAT" | "DOUBLE" | "REAL" | "NUMBER" | "NUMERIC" => CastTarget::Float,
            "STRING" | "VARCHAR" | "TEXT" | "CHAR" => CastTarget::String,
            "BOOL" | "BOOLEAN" => CastTarget::Boolean,
            other => return Err(format!("unknown CAST target type `{other}`")),
        };
        self.expect(&Tok::RParen)?;
        Ok(Expr::Cast { target, expr })
    }

    // property_exists := PROPERTY_EXISTS '(' var ',' key ')'. Both operands are
    // bare names: the variable resolves to a slot, the key stays a literal
    // property name. Presence-not-value — see `Expr::PropertyExists`.
    fn property_exists_expr(&mut self) -> Result<Expr, String> {
        self.expect(&Tok::LParen)?;
        let var = self.ident()?;
        let slot = *self
            .scope
            .get(&var)
            .ok_or_else(|| format!("unknown variable `{var}`"))?;
        self.expect(&Tok::Comma)?;
        let key = self.ident()?;
        self.expect(&Tok::RParen)?;
        Ok(Expr::PropertyExists { slot, key })
    }

    // exists := EXISTS '{' node ( rel [quant] node )* [WHERE pred] '}'
    // A correlated existence check: the pattern's first node must be a variable
    // already bound in the outer scope (the correlation), and the body extends
    // from it. A trailing WHERE is a sub-pattern predicate over the body scope.
    fn exists_expr(&mut self) -> Result<Expr, String> {
        let (body, outer_width) = self.correlated_subquery_body("EXISTS")?;
        Ok(Expr::Exists {
            body: Box::new(body),
            outer_width,
        })
    }

    // count_subquery := COUNT '{' node ( rel [quant] node )* [WHERE pred] '}' — the
    // number of sub-matches per outer row (distinct from the `count(*)` aggregate,
    // which takes `(…)`). Same correlated body as EXISTS.
    /// Parse the `<value type>` of an `IS TYPED` predicate into the category string
    /// `category_matches` understands. Only the scalar/record/list vocabulary; a
    /// temporal type or a closed `RECORD { … }` schema returns an error (those cases
    /// are left unhandled rather than answered wrong).
    fn value_type_category(&mut self) -> Result<&'static str, String> {
        let ty = self.ident()?;
        Ok(match ty.to_ascii_uppercase().as_str() {
            "ANY" => {
                if self.peek_kw("RECORD") {
                    self.eat_kw("RECORD");
                    "record"
                } else {
                    "any"
                }
            }
            "RECORD" => {
                if matches!(self.peek(), Some(Tok::LBrace)) {
                    return Err("IS TYPED closed RECORD schema is not supported".into());
                }
                "record"
            }
            "INTEGER" | "INT" => "integer",
            "FLOAT" => "float",
            "STRING" => "string",
            "BOOLEAN" | "BOOL" => "bool",
            "LIST" => "list",
            "NULL" => "null",
            "DATE" => "date",
            "DURATION" => "duration",
            // Two-word temporal types: `LOCAL TIME`/`LOCAL DATETIME`, `ZONED
            // TIME`/`ZONED DATETIME`.
            "LOCAL" | "ZONED" => {
                let zoned = ty.eq_ignore_ascii_case("ZONED");
                let unit = self.ident()?;
                match (zoned, unit.to_ascii_uppercase().as_str()) {
                    (false, "TIME") => "local_time",
                    (false, "DATETIME") => "local_datetime",
                    (true, "TIME") => "zoned_time",
                    (true, "DATETIME") => "zoned_datetime",
                    _ => return Err(format!("IS TYPED {ty} {unit} is not supported")),
                }
            }
            other => return Err(format!("IS TYPED {other} is not supported")),
        })
    }

    fn count_subquery_expr(&mut self) -> Result<Expr, String> {
        let (body, outer_width) = self.correlated_subquery_body("COUNT")?;
        Ok(Expr::CountSubquery {
            body: Box::new(body),
            outer_width,
        })
    }

    /// Parse the `{ <pattern> [WHERE pred] }` body shared by `EXISTS { … }` and
    /// `COUNT { … }`: a pattern correlated on an outer-bound start variable, rooted at
    /// `Plan::Row`, with slot `outer_width` reserved for the evaluator's provenance
    /// column. Returns `(body, outer_width)`. `kw` names the construct for errors.
    fn correlated_subquery_body(&mut self, kw: &str) -> Result<(Plan, usize), String> {
        self.expect(&Tok::LBrace)?;
        // The pattern may be written with an explicit leading `MATCH` — `EXISTS {
        // MATCH (a)-[:R]->(b) }` — the full-statement form; accept it as sugar.
        self.eat_kw("MATCH");
        let outer_width = self.slots;
        let (var, label, props, start_where, le) = self.node()?;
        let Some(v) = var else {
            return Err(format!("{kw} pattern must start from a bound variable"));
        };
        if start_where.is_some() {
            return Err(format!(
                "inline WHERE on a {kw} start variable is not supported; use a trailing WHERE"
            ));
        }

        let mut sub_scope = self.scope.clone();
        let mut sub_slots = outer_width + 1;
        let body = if let Some(&from) = self.scope.get(&v) {
            // FORWARD: the first node is the bound correlated variable; it may not be
            // re-labeled or re-constrained. Extend the chain from it.
            if label.is_some() || le.is_some() {
                return Err(format!(
                    "bound variable `{v}` cannot be re-labeled inside {kw}"
                ));
            }
            if !props.is_empty() {
                return Err(format!(
                    "bound variable `{v}` cannot be re-constrained with inline properties \
                     inside {kw}; use WHERE"
                ));
            }
            self.extend_chain(Plan::Row, &mut sub_scope, &mut sub_slots, from)?
        } else {
            // REVERSE (single hop): the first node is a LOCAL variable; the correlated
            // (bound) variable is the LANDING — `EXISTS { (m)-[:R]->(n) }` with `n`
            // outer. Traverse from the bound endpoint backward to the local node.
            let rel = self.rel()?;
            if self.opt_quantifier()?.is_some() {
                return Err(format!(
                    "a variable-length {kw} correlated on the landing node is not supported"
                ));
            }
            if rel.var.is_some() || !rel.props.is_empty() || rel.where_range.is_some() {
                return Err(format!(
                    "a bound edge / edge properties on a landing-correlated {kw} is not supported"
                ));
            }
            let (vb, vb_label, vb_props, vb_where, vb_le) = self.node()?;
            let Some(vb) = vb else {
                return Err(format!("{kw} must correlate on a bound variable"));
            };
            let Some(&from) = self.scope.get(&vb) else {
                return Err(format!(
                    "{kw} must start from or land on a bound (correlated) variable; neither \
                     `{v}` nor `{vb}` is in scope"
                ));
            };
            if vb_label.is_some() || vb_le.is_some() || !vb_props.is_empty() || vb_where.is_some() {
                return Err(format!(
                    "the correlated variable `{vb}` cannot be re-constrained inside {kw}"
                ));
            }
            // Reverse the hop direction and expand from the bound endpoint; the local
            // node lands at the same slot the forward path would use.
            let rev_dir = match rel.dir {
                Dir::Out => Dir::In,
                Dir::In => Dir::Out,
                Dir::Both => Dir::Both,
            };
            let local_slot = outer_width + 1;
            sub_scope.insert(v.clone(), local_slot);
            let mut body = Plan::Row.expand(from, rev_dir, &rel.etypes);
            if let Some(pred) = landing_label_filter(label, le, local_slot) {
                body = body.filter(pred);
            }
            body = node_prop_filters(body, local_slot, props);
            if let Some(r) = start_where {
                self.scope = sub_scope.clone();
                body = body.filter(self.parse_captured_where(r)?);
            }
            body
        };
        let body = if self.eat_kw("WHERE") {
            let saved_scope = std::mem::replace(&mut self.scope, sub_scope);
            let saved_slots = std::mem::replace(&mut self.slots, sub_slots);
            let pred = self.expr()?;
            self.scope = saved_scope;
            self.slots = saved_slots;
            body.filter(pred)
        } else {
            body
        };
        self.expect(&Tok::RBrace)?;
        Ok((body, outer_width))
    }

    // call := name '(' [ expr (',' expr)* ] ')'  — a scalar function.
    // Validates the name and arity here so `eval` only sees well-formed calls.
    fn call(&mut self, name: &str) -> Result<Expr, String> {
        self.expect(&Tok::LParen)?;
        // `TRIM` has the SQL spec grammar `TRIM([LEADING|TRAILING|BOTH] [char] FROM
        // src)` (as well as the plain `TRIM(src)`), which the generic comma-arg loop
        // can't parse — special-case it into the ordinary ltrim/rtrim/trim calls.
        if name.eq_ignore_ascii_case("trim") {
            return self.trim_call();
        }
        let mut args = Vec::new();
        if self.peek() != Some(&Tok::RParen) {
            loop {
                args.push(self.expr()?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen)?;
        let lname = name.to_ascii_lowercase();
        // Path accessors read the lineage sidecar, not a value — route them to
        // `Expr::PathAccess`. Their sole argument must be a path variable.
        if let Some(part) = path_part(&lname) {
            if !matches!(args.as_slice(), [Expr::Path]) {
                return Err(format!("{lname}() takes a path variable"));
            }
            return Ok(Expr::PathAccess { part });
        }
        let arity_ok = match lname.as_str() {
            // 0 args (numeric constants)
            "e" | "pi" => args.is_empty(),
            // 1 arg
            "abs" | "sign" | "floor" | "ceil" | "ceiling" | "sqrt" | "exp" | "ln" | "log10"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh"
            | "cot" | "degrees" | "radians" | "upper" | "lower" | "trim" | "length" | "size"
            | "head" | "last" | "year" | "month" | "day" | "hour" | "minute" | "second"
            | "_year" | "_month" | "_day" | "_hour" | "_minute" | "_second" | "date"
            | "local_time" | "datetime" | "local_datetime" | "zoned_time" | "zoned_datetime"
            | "duration" | "to_integer" | "tointeger" | "to_float" | "tofloat" | "to_string"
            | "tostring" | "to_boolean" | "toboolean" | "char_length" | "character_length"
            | "byte_length" | "octet_length" | "reverse" | "tail" | "keys" | "labels" | "type"
            | "property_names" | "element_id" => args.len() == 1,
            // `round(x)` or `round(x, digits)`
            "round" => args.len() == 1 || args.len() == 2,
            // `list_sort(list)` / `(list, order)` / `(list, order, nullOrder)`
            "list_sort" => (1..=3).contains(&args.len()),
            // list algebra (2 args)
            "append" | "list_contains" | "list_union" | "difference" | "intersection" => {
                args.len() == 2
            }
            // 1 or 2 args (bare form trims whitespace; a 2nd arg is the char set)
            "ltrim" | "rtrim" | "btrim" => args.len() == 1 || args.len() == 2,
            // 2 args
            "starts_with" | "ends_with" | "contains" | "duration_between" | "nullif" | "log"
            | "power" | "mod" | "left" | "right" | "split" | "atan2" => args.len() == 2,
            // 2 or 3 args
            "range" => args.len() == 2 || args.len() == 3,
            // 3 args
            "replace" => args.len() == 3,
            // 2 or 3 args
            "substring" => args.len() == 2 || args.len() == 3,
            // variadic (≥1)
            "coalesce" => !args.is_empty(),
            _ => return Err(format!("unknown function `{name}`")),
        };
        if !arity_ok {
            return Err(format!(
                "{lname}() called with the wrong number of arguments"
            ));
        }
        Ok(Expr::Call { name: lname, args })
    }

    /// Parse the SQL-spec `TRIM` body — the `(` is already consumed. A leading
    /// `LEADING`/`TRAILING`/`BOTH` (default BOTH) selects the side and desugars to
    /// `ltrim`/`rtrim`/`trim`; an optional trim character precedes `FROM src`. So
    /// `TRIM(LEADING 'x' FROM s)` → `ltrim(s, 'x')` and `TRIM(s)` → `trim(s)`. The
    /// char, when present, is the SECOND argument, matching core.
    fn trim_call(&mut self) -> Result<Expr, String> {
        let fname = if self.eat_kw("LEADING") {
            "ltrim"
        } else if self.eat_kw("TRAILING") {
            "rtrim"
        } else {
            self.eat_kw("BOTH");
            "trim"
        };
        let args = if self.eat_kw("FROM") {
            // `TRIM([side] FROM src)` — whitespace trim, no char set.
            vec![self.expr()?]
        } else {
            let e1 = self.expr()?;
            if self.eat_kw("FROM") {
                // `TRIM([side] char FROM src)` — char is the 2nd arg.
                vec![self.expr()?, e1]
            } else {
                // Plain `TRIM(src)`.
                vec![e1]
            }
        };
        self.expect(&Tok::RParen)?;
        Ok(Expr::Call {
            name: fname.into(),
            args,
        })
    }
}

/// An open `{n,}` upper bound is capped here (path enumeration is exponential;
/// an unbounded quantifier needs the reachability form, not enumeration).
const MAX_VARLEN: u32 = 32;

/// Build the projection (or aggregation) a `RETURN`/`WITH` item list describes,
/// attached above `plan`, and return it with the ordered output column names. An
/// aggregate anywhere makes it an `Aggregate` whose non-aggregate items are the
/// implicit GROUP BY keys; otherwise a plain `Project`. Shared by `RETURN` and
/// `WITH` so the two build identical shapes.
/// Structural equality of two expressions (the IR `Expr` is not `PartialEq`), via
/// their debug rendering. Used to match a HAVING/SELECT expression against a group
/// key — exact enough for keys, which are simple property/slot expressions.
fn expr_eq(a: &Expr, b: &Expr) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

/// Rewrite a HAVING predicate so any sub-expression equal to a group key becomes a
/// `Slot` into the post-aggregation schema (that key's column index) — a group-key
/// reference reads the grouped column, not the pre-aggregation property. Aggregates
/// were already replaced with slots at parse time (see `hoist_having_agg`).
fn rewrite_group_keys(e: Expr, keys: &[(String, Expr)]) -> Expr {
    if let Some(i) = keys.iter().position(|(_, ke)| expr_eq(ke, &e)) {
        return Expr::Slot(i);
    }
    let go = |b: Box<Expr>| Box::new(rewrite_group_keys(*b, keys));
    match e {
        Expr::Not(x) => Expr::Not(go(x)),
        Expr::And(a, b) => Expr::And(go(a), go(b)),
        Expr::Or(a, b) => Expr::Or(go(a), go(b)),
        Expr::Xor(a, b) => Expr::Xor(go(a), go(b)),
        Expr::Compare { op, left, right } => Expr::Compare {
            op,
            left: go(left),
            right: go(right),
        },
        Expr::Arith { op, left, right } => Expr::Arith {
            op,
            left: go(left),
            right: go(right),
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: go(expr),
            negated,
        },
        Expr::In { needle, haystack } => Expr::In {
            needle: go(needle),
            haystack: go(haystack),
        },
        Expr::Call { name, args } => Expr::Call {
            name,
            args: args
                .into_iter()
                .map(|a| rewrite_group_keys(a, keys))
                .collect(),
        },
        other => other,
    }
}

/// Is the token an infix operator that would continue an expression after a parsed
/// aggregate (`count(*) <op> …`)? Used to tell a bare aggregate item from one
/// embedded in a larger projection expression.
fn is_operator_continuation(t: Option<&Tok>) -> bool {
    match t {
        Some(
            Tok::Plus
            | Tok::Minus
            | Tok::Star
            | Tok::Slash
            | Tok::Percent
            | Tok::Concat
            | Tok::Eq
            | Tok::Ne
            | Tok::Lt
            | Tok::Le
            | Tok::Gt
            | Tok::Ge,
        ) => true,
        // Keyword operators (`OR`/`AND`/`XOR`/`IN`/`IS`).
        Some(Tok::Ident(s)) => {
            matches!(
                s.to_ascii_uppercase().as_str(),
                "OR" | "AND" | "XOR" | "IN" | "IS"
            )
        }
        _ => false,
    }
}

/// Rewrite an aggregate-expression's `AGG_SLOT_BASE`-offset slots to their real
/// post-aggregation columns: the i-th hoisted aggregate of an item lands at
/// `base + i` in the `[keys…, aggs…]` schema.
fn rewrite_agg_slots(e: Expr, base: usize) -> Expr {
    let go = |b: Box<Expr>| Box::new(rewrite_agg_slots(*b, base));
    match e {
        Expr::Slot(s) if s >= AGG_SLOT_BASE => Expr::Slot(base + (s - AGG_SLOT_BASE)),
        Expr::Not(x) => Expr::Not(go(x)),
        Expr::And(a, b) => Expr::And(go(a), go(b)),
        Expr::Or(a, b) => Expr::Or(go(a), go(b)),
        Expr::Xor(a, b) => Expr::Xor(go(a), go(b)),
        Expr::Compare { op, left, right } => Expr::Compare {
            op,
            left: go(left),
            right: go(right),
        },
        Expr::Arith { op, left, right } => Expr::Arith {
            op,
            left: go(left),
            right: go(right),
        },
        Expr::IsNull { expr, negated } => Expr::IsNull {
            expr: go(expr),
            negated,
        },
        Expr::In { needle, haystack } => Expr::In {
            needle: go(needle),
            haystack: go(haystack),
        },
        Expr::Call { name, args } => Expr::Call {
            name,
            args: args
                .into_iter()
                .map(|a| rewrite_agg_slots(a, base))
                .collect(),
        },
        other => other,
    }
}

fn apply_items(plan: Plan, items: &[RetItem]) -> (Plan, Vec<String>) {
    let has_agg = items.iter().any(RetItem::has_agg);
    let has_agg_expr = items.iter().any(|it| matches!(it, RetItem::AggExpr { .. }));
    let out_names: Vec<String> = items.iter().map(RetItem::name).collect();
    let plan = if !has_agg {
        let proj = items
            .iter()
            .map(|it| match it {
                RetItem::Key(name, e) => (name.clone(), e.clone()),
                _ => unreachable!("no aggregates on this branch"),
            })
            .collect();
        plan.project(proj)
    } else if !has_agg_expr {
        // Simple aggregate: keys then aggregates, output = the aggregate columns.
        let keys = items
            .iter()
            .filter_map(|it| match it {
                RetItem::Key(name, e) => Some((name.clone(), e.clone())),
                _ => None,
            })
            .collect();
        let aggs = items
            .iter()
            .filter_map(|it| match it {
                RetItem::Agg(a) => Some(a.clone()),
                _ => None,
            })
            .collect();
        plan.aggregate(keys, aggs)
    } else {
        // An aggregate embedded in a projection expression: aggregate over the group
        // keys and ALL hoisted aggregates, then PROJECT each item over the
        // `[keys…, aggs…]` schema (a bare agg → its column; an agg-expression → its
        // slot-rewritten expression; a key → its key column).
        let keys: Vec<(String, Expr)> = items
            .iter()
            .filter_map(|it| match it {
                RetItem::Key(name, e) => Some((name.clone(), e.clone())),
                _ => None,
            })
            .collect();
        let k = keys.len();
        let mut aggs: Vec<crate::ir::Agg> = Vec::new();
        let mut proj: Vec<(String, Expr)> = Vec::with_capacity(items.len());
        let mut key_i = 0usize;
        for it in items {
            match it {
                RetItem::Key(name, _) => {
                    proj.push((name.clone(), Expr::Slot(key_i)));
                    key_i += 1;
                }
                RetItem::Agg(a) => {
                    let pos = k + aggs.len();
                    aggs.push(a.clone());
                    proj.push((a.name.clone(), Expr::Slot(pos)));
                }
                RetItem::AggExpr {
                    name,
                    expr,
                    aggs: ia,
                } => {
                    let base = k + aggs.len();
                    aggs.extend(ia.iter().cloned());
                    proj.push((name.clone(), rewrite_agg_slots(expr.clone(), base)));
                }
            }
        }
        plan.aggregate(keys, aggs).project(proj)
    };
    (plan, out_names)
}

/// A default output-column name for an un-aliased item, resolving a bare variable
/// reference (`Expr::Slot`) back to the variable's name — so `WITH a` / `RETURN a`
/// carries the column under `a`, not `col0`. Falls back to `default_name` for
/// anything not tied to a single bound name.
impl Parser {
    fn item_name(&self, e: &Expr, idx: usize) -> String {
        if let Expr::Slot(n) = e {
            if let Some((name, _)) = self.scope.iter().find(|(_, &slot)| slot == *n) {
                return name.clone();
            }
        }
        default_name(e, idx)
    }
}

/// Map a temporal-literal keyword to its `Temporal` kind tag, or `None`. `TIME`
/// and `DATETIME` are the LOCAL (zone-less) forms.
fn temporal_tag(kw: &str) -> Option<&'static str> {
    Some(match kw.to_ascii_uppercase().as_str() {
        "DATE" => "date",
        "TIME" => "localtime",
        // `TIMESTAMP` is core's alias for a (local) DATETIME literal.
        "DATETIME" | "TIMESTAMP" => "datetime",
        "DURATION" => "duration",
        _ => return None,
    })
}

/// Map a path-accessor function name to its `PathPart`, or `None` if it is not one.
/// `edges` is core's spelling for the relationships accessor — accepted for parity
/// alongside the engine's `relationships` (a superset alias).
fn path_part(name: &str) -> Option<PathPart> {
    Some(match name {
        "nodes" => PathPart::Nodes,
        "relationships" | "edges" => PathPart::Relationships,
        "path_length" => PathPart::Length,
        "elements" => PathPart::Elements,
        _ => return None,
    })
}

/// A default output-column name for an un-aliased item.
fn default_name(e: &Expr, idx: usize) -> String {
    match e {
        Expr::Prop { key, .. } => key.clone(),
        _ => format!("col{idx}"),
    }
}

#[cfg(test)]
mod tests {
    use crate::exec::{run, Rows};
    use crate::store::{Builder, Store};
    use crate::value::Value;
    use std::sync::Arc;

    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }
    fn n(x: f64) -> Value {
        Value::Num(x)
    }

    fn social() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["Person"], &[("name", s("alice")), ("age", n(30.0))]);
        let bob = b.node(&["Person"], &[("name", s("bob")), ("age", n(25.0))]);
        let c = b.node(&["Person"], &[("name", s("carol")), ("age", n(40.0))]);
        let proj = b.node(&["Project"], &[("name", s("graphdb"))]);
        b.edge(a, bob, "KNOWS");
        b.edge(a, c, "KNOWS");
        b.edge(bob, c, "KNOWS");
        b.edge(a, proj, "WORKS_ON");
        b.build()
    }

    /// Rows as a sorted multiset of `col=value;` strings — order-independent
    /// comparison, since result order is unspecified without ORDER BY.
    fn bag(rows: &Rows) -> Vec<String> {
        let mut out: Vec<String> = rows
            .rows
            .iter()
            .map(|r| {
                rows.names
                    .iter()
                    .zip(r)
                    .map(|(k, v)| format!("{k}={v:?};"))
                    .collect::<String>()
            })
            .collect();
        out.sort();
        out
    }

    /// The parser is correct iff parse->run reproduces the hand-built plan.
    fn assert_same(query: &str, hand: &crate::ir::Plan, store: &Store) {
        let parsed = super::parse(query).unwrap_or_else(|e| panic!("parse `{query}`: {e}"));
        assert_eq!(
            bag(&run(&parsed, store)),
            bag(&run(hand, store)),
            "parsed plan differs for `{query}`"
        );
    }

    /// Inline node-property maps `(n:L {k: v, …})` are a match filter — the same
    /// rows as the `WHERE` spelling, on the seed node AND a hop's landing node.
    #[test]
    fn inline_property_maps_match_where() {
        let store = social();
        let same = |inline: &str, wher: &str| {
            let a = bag(&run(&super::parse(inline).unwrap(), &store));
            let b = bag(&run(&super::parse(wher).unwrap(), &store));
            assert_eq!(a, b, "`{inline}` vs `{wher}`");
        };
        // Seed node, single and multi-property.
        same(
            "MATCH (n:Person {name: 'alice'}) RETURN n.age AS a",
            "MATCH (n:Person) WHERE n.name = 'alice' RETURN n.age AS a",
        );
        same(
            "MATCH (n:Person {name: 'alice', age: 30}) RETURN n.name AS x",
            "MATCH (n:Person) WHERE n.name = 'alice' AND n.age = 30 RETURN n.name AS x",
        );
        // Landing node of a hop.
        same(
            "MATCH (a:Person {name: 'alice'})-[:KNOWS]->(b {name: 'carol'}) RETURN b.age AS a",
            "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name = 'alice' AND b.name = 'carol' RETURN b.age AS a",
        );
        // Empty map is a no-op filter (all rows).
        same(
            "MATCH (n:Person {}) RETURN n.name AS x",
            "MATCH (n:Person) RETURN n.name AS x",
        );
        // A non-matching constraint yields nothing.
        assert!(bag(&run(
            &super::parse("MATCH (n:Person {name: 'nobody'}) RETURN n.name AS x").unwrap(),
            &store
        ))
        .is_empty());
    }

    #[test]
    fn single_node_return_property() {
        use crate::ir::{Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "name".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        )]);
        assert_same("MATCH (p:Person) RETURN p.name", &hand, &store);
    }

    #[test]
    fn cast_parses_target_and_runs() {
        use crate::ir::{CastTarget, Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "a".into(),
            Expr::Cast {
                target: CastTarget::String,
                expr: Box::new(Expr::Prop {
                    slot: 0,
                    key: "age".into(),
                }),
            },
        )]);
        assert_same(
            "MATCH (p:Person) RETURN CAST(p.age AS STRING) AS a",
            &hand,
            &store,
        );
    }

    #[test]
    fn cast_integer_alias_and_bad_type() {
        use crate::ir::{CastTarget, Expr, Plan};
        let store = social();
        // The `INT` alias parses to the same `Integer` target as `INTEGER`.
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "a".into(),
            Expr::Cast {
                target: CastTarget::Integer,
                expr: Box::new(Expr::Prop {
                    slot: 0,
                    key: "age".into(),
                }),
            },
        )]);
        assert_same(
            "MATCH (p:Person) RETURN CAST(p.age AS INT) AS a",
            &hand,
            &store,
        );
        // An unknown target type is a parse error, not a silent fallback.
        assert!(super::parse("MATCH (p:Person) RETURN CAST(p.age AS WIDGET) AS a").is_err());
    }

    /// A store with all three null states on `P.age`: present non-null, absent,
    /// and present-null. These are what separate `IS NULL` (a value test) from
    /// `PROPERTY_EXISTS` (a presence test).
    fn null_states() -> Store {
        let mut b = Builder::default();
        b.node(&["P"], &[("name", s("has")), ("age", n(30.0))]);
        b.node(&["P"], &[("name", s("absent"))]);
        b.node(&["P"], &[("name", s("null"))]);
        let mut st = b.build();
        st.set_prop(2, "age", Value::Null); // node 2: present, but Null
        st
    }

    /// The set of `name` values a query returns, sorted — order is unspecified.
    fn names(store: &Store, query: &str) -> Vec<String> {
        let out = run(&super::parse(query).unwrap(), store);
        let mut got: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                other => format!("{other:?}"),
            })
            .collect();
        got.sort();
        got
    }

    #[test]
    fn is_null_is_a_value_test() {
        use crate::ir::{Expr, Plan};
        let st = null_states();
        // IS NULL is TRUE for both absent and present-null; IS NOT NULL only for
        // the present non-null. (A definite predicate — no row is UNKNOWN.)
        assert_eq!(
            names(&st, "MATCH (p:P) WHERE p.age IS NULL RETURN p.name"),
            vec!["absent", "null"]
        );
        assert_eq!(
            names(&st, "MATCH (p:P) WHERE p.age IS NOT NULL RETURN p.name"),
            vec!["has"]
        );
        // Parse cross-check against the hand-built plan.
        let hand = Plan::Scan {
            label: Some("P".into()),
        }
        .filter(Expr::IsNull {
            expr: Box::new(Expr::Prop {
                slot: 0,
                key: "age".into(),
            }),
            negated: false,
        })
        .project(vec![(
            "name".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        )]);
        assert_same(
            "MATCH (p:P) WHERE p.age IS NULL RETURN p.name AS name",
            &hand,
            &st,
        );
    }

    #[test]
    fn property_exists_is_a_presence_test() {
        use crate::ir::{Expr, Plan};
        let st = null_states();
        // PROPERTY_EXISTS is TRUE wherever the value is PRESENT — including the
        // present-null — and FALSE only for the absent node. This is the case
        // `IS NOT NULL` cannot express: "null" appears here but not above.
        assert_eq!(
            names(
                &st,
                "MATCH (p:P) WHERE PROPERTY_EXISTS(p, age) RETURN p.name"
            ),
            vec!["has", "null"]
        );
        let hand = Plan::Scan {
            label: Some("P".into()),
        }
        .filter(Expr::PropertyExists {
            slot: 0,
            key: "age".into(),
        })
        .project(vec![(
            "name".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        )]);
        assert_same(
            "MATCH (p:P) WHERE PROPERTY_EXISTS(p, age) RETURN p.name AS name",
            &hand,
            &st,
        );
    }

    #[test]
    fn with_aggregate_then_having_filter() {
        use crate::ir::{AggFn, CompareOp, Dir, Expr, Plan};
        let store = social();
        // KNOWS out-degree: alice=2 (bob,carol), bob=1 (carol), carol=0. WITH
        // aggregates the degree, then WHERE filters it (HAVING) — which a single
        // RETURN cannot do. Only alice survives n >= 2.
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .aggregate(
            vec![("a".into(), Expr::Slot(0))],
            vec![crate::ir::Agg {
                func: AggFn::Count,
                arg: Some(Expr::Slot(1)),
                distinct: false,
                name: "n".into(),
                frac: None,
            }],
        )
        .filter(Expr::Compare {
            op: CompareOp::Ge,
            left: Box::new(Expr::Slot(1)),
            right: Box::new(Expr::Lit(Value::Num(2.0))),
        })
        .project(vec![
            (
                "name".into(),
                Expr::Prop {
                    slot: 0,
                    key: "name".into(),
                },
            ),
            ("n".into(), Expr::Slot(1)),
        ]);
        let q = "MATCH (a:Person)-[:KNOWS]->(b) WITH a, count(b) AS n WHERE n >= 2 \
                 RETURN a.name AS name, n";
        assert_same(q, &hand, &store);
        // And the concrete answer: alice with degree 2.
        let out = run(&super::parse(q).unwrap(), &store);
        assert_eq!(out.rows.len(), 1);
        assert!(crate::value::equals(&col(&out, 0, "name"), &s("alice")));
        assert_eq!(num(&col(&out, 0, "n")), 2.0);
    }

    #[test]
    fn with_carries_a_node_into_a_continuing_match() {
        use crate::ir::{CompareOp, Dir, Expr, Plan};
        let store = social();
        // Carry `a`, filter it (HAVING), then continue the pattern FROM `a`. Only
        // alice(30)/carol(40) pass age>=30; alice KNOWS bob,carol and carol knows
        // no one out, so the endpoints are bob and carol.
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![("a".into(), Expr::Slot(0))])
        .filter(Expr::Compare {
            op: CompareOp::Ge,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "age".into(),
            }),
            right: Box::new(Expr::Lit(Value::Num(30.0))),
        })
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .project(vec![(
            "name".into(),
            Expr::Prop {
                slot: 1,
                key: "name".into(),
            },
        )]);
        let q = "MATCH (a:Person) WITH a WHERE a.age >= 30 \
                 MATCH (a)-[:KNOWS]->(b) RETURN b.name AS name";
        assert_same(q, &hand, &store);
        assert_eq!(names(&store, q), vec!["bob", "carol"]);
    }

    #[test]
    fn with_order_by_alias_and_limit_pages() {
        let store = social();
        // WITH projects age+name, pages by age DESC LIMIT 2 (carol 40, alice 30 —
        // bob 25 is dropped), then RETURN name. The surviving set is {alice,carol}.
        let q = "MATCH (p:Person) WITH p.age AS age, p.name AS name \
                 ORDER BY age DESC LIMIT 2 RETURN name";
        assert_eq!(names(&store, q), vec!["alice", "carol"]);
    }

    #[test]
    fn continuing_match_from_unbound_variable_errors() {
        // After `WITH a`, only `a` is in scope; a continuing MATCH from an unbound
        // variable is a clear parse error, not a silent fresh scan.
        let err =
            super::parse("MATCH (a:Person) WITH a MATCH (z)-[:KNOWS]->(y) RETURN y.name AS name")
                .unwrap_err();
        assert!(err.contains("not in scope"), "got: {err}");
    }

    #[test]
    fn exists_correlated_subpattern() {
        use crate::ir::{Dir, Expr, Plan};
        let store = social();
        // Who has an outgoing KNOWS? alice (bob,carol) and bob (carol); carol has
        // none. EXISTS is a definite predicate over the correlated node `p`.
        let body = Plan::Row.expand(0, Dir::Out, &["KNOWS".to_string()]);
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(Expr::Exists {
            body: Box::new(body),
            outer_width: 1,
        })
        .project(vec![(
            "name".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        )]);
        let q = "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(x) } RETURN p.name AS name";
        assert_same(q, &hand, &store);
        assert_eq!(names(&store, q), vec!["alice", "bob"]);
    }

    #[test]
    fn exists_with_inner_where_on_body_var() {
        let store = social();
        // The sub-pattern's WHERE filters the reached node: only a KNOWS target
        // younger than 30 counts. alice knows bob(25) → yes; bob knows only
        // carol(40) → no; carol knows no one → no. So only alice qualifies.
        let q = "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(x) WHERE x.age < 30 } \
                 RETURN p.name AS name";
        assert_eq!(names(&store, q), vec!["alice"]);
    }

    #[test]
    fn exists_where_correlates_on_the_outer_row() {
        let store = social();
        // The sub-WHERE references the OUTER node `p`: does p know someone older?
        // alice(30) knows carol(40) → yes; bob(25) knows carol(40) → yes;
        // carol(40) knows no one → no.
        let q = "MATCH (p:Person) WHERE EXISTS { (p)-[:KNOWS]->(x) WHERE x.age > p.age } \
                 RETURN p.name AS name";
        assert_eq!(names(&store, q), vec!["alice", "bob"]);
    }

    #[test]
    fn not_exists_negates_the_predicate() {
        let store = social();
        // EXISTS is a definite Bool, so NOT composes cleanly: the Persons with NO
        // outgoing KNOWS. Only carol (alice and bob both know someone).
        let q = "MATCH (p:Person) WHERE NOT EXISTS { (p)-[:KNOWS]->(x) } RETURN p.name AS name";
        assert_eq!(names(&store, q), vec!["carol"]);
    }

    #[test]
    fn exists_from_unbound_variable_errors() {
        // The correlated variable must be a bound endpoint — a fully-fresh scan
        // inside EXISTS (NEITHER endpoint bound) is not this construct.
        let err = super::parse(
            "MATCH (p:Person) WHERE EXISTS { (z)-[:KNOWS]->(x) } RETURN p.name AS name",
        )
        .unwrap_err();
        assert!(err.contains("in scope"), "got: {err}");
    }

    #[test]
    fn call_inline_lateral_join() {
        use crate::ir::{Dir, Expr, Plan};
        let store = social();
        // For each Person, expand KNOWS in a subquery and yield the friend's name
        // — a lateral join. carol knows no one, so she drops out (inner join).
        let call = Plan::CallInline {
            input: Box::new(Plan::Scan {
                label: Some("Person".into()),
            }),
            body: Box::new(Plan::Row.expand(0, Dir::Out, &["KNOWS".to_string()])),
            yields: vec![(
                "friend".into(),
                Expr::Prop {
                    slot: 1,
                    key: "name".into(),
                },
            )],
            outer_width: 1,
        };
        let hand = call.project(vec![
            (
                "name".into(),
                Expr::Prop {
                    slot: 0,
                    key: "name".into(),
                },
            ),
            ("friend".into(), Expr::Slot(1)),
        ]);
        let q = "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(x) RETURN x.name AS friend } \
                 RETURN p.name AS name, friend";
        assert_same(q, &hand, &store);
        assert_eq!(
            bag(&run(&super::parse(q).unwrap(), &store)),
            vec![
                "name=Str(\"alice\");friend=Str(\"bob\");",
                "name=Str(\"alice\");friend=Str(\"carol\");",
                "name=Str(\"bob\");friend=Str(\"carol\");",
            ]
        );
    }

    #[test]
    fn call_inline_subquery_where() {
        let store = social();
        // The subquery's WHERE filters the reached node: only friends older than 30
        // (carol) count. alice and bob both know carol; carol knows no one.
        let q = "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(x) WHERE x.age > 30 \
                 RETURN x.name AS friend } RETURN p.name AS name, friend";
        assert_eq!(
            bag(&run(&super::parse(q).unwrap(), &store)),
            vec![
                "name=Str(\"alice\");friend=Str(\"carol\");",
                "name=Str(\"bob\");friend=Str(\"carol\");",
            ]
        );
    }

    #[test]
    fn call_inline_yield_correlates_on_outer() {
        let store = social();
        // The yield expression mixes the subquery node and the OUTER node: the age
        // gap x.age - p.age. alice(30)->bob(25)=-5, alice->carol(40)=10,
        // bob(25)->carol(40)=15. carol knows no one.
        let q = "MATCH (p:Person) CALL (p) { MATCH (p)-[:KNOWS]->(x) \
                 RETURN x.age - p.age AS gap } RETURN gap";
        let out = run(&super::parse(q).unwrap(), &store);
        let mut gaps: Vec<f64> = out
            .rows
            .iter()
            .map(|r| match r[0] {
                Value::Num(x) => x,
                ref o => panic!("expected Num, got {o:?}"),
            })
            .collect();
        gaps.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        assert_eq!(gaps, vec![-5.0, 10.0, 15.0]);
    }

    #[test]
    fn call_named_form_is_deferred() {
        // The named-procedure form has no catalog yet (the algorithms it calls are
        // a later phase); it is a clear error, not a silent no-op.
        let err = super::parse("MATCH (p:Person) CALL foo() RETURN p.name AS name").unwrap_err();
        assert!(
            err.contains("named-procedure CALL is deferred"),
            "got: {err}"
        );
    }

    #[test]
    fn call_inline_unbound_scope_errors() {
        let err = super::parse(
            "MATCH (p:Person) CALL (z) { MATCH (z)-[:KNOWS]->(x) RETURN x.name AS f } \
             RETURN p.name AS name",
        )
        .unwrap_err();
        assert!(err.contains("not bound"), "got: {err}");
    }

    /// A straight chain a→b→c→d over LINK edges (node ids 0,1,2,3). Shortest
    /// paths have distinct, checkable lengths — unlike the dense `social()`.
    fn chain() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[("name", s("a"))]);
        let bb = b.node(&["N"], &[("name", s("b"))]);
        let c = b.node(&["N"], &[("name", s("c"))]);
        let d = b.node(&["N"], &[("name", s("d"))]);
        b.edge(a, bb, "LINK");
        b.edge(bb, c, "LINK");
        b.edge(c, d, "LINK");
        b.build()
    }

    #[test]
    fn any_shortest_path_length() {
        use crate::ir::{Dir, Expr, PathPart, Plan};
        let store = chain();
        // Shortest LINK paths from `a`: the `*` quantifier admits the zero-length
        // path to `a` itself (len 0), then b at 1 hop, c at 2, d at 3.
        let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) WHERE x.name = 'a' \
                 RETURN y.name AS y, path_length(p) AS len";
        assert_eq!(
            bag(&run(&super::parse(q).unwrap(), &store)),
            vec![
                "y=Str(\"a\");len=Num(0.0);",
                "y=Str(\"b\");len=Num(1.0);",
                "y=Str(\"c\");len=Num(2.0);",
                "y=Str(\"d\");len=Num(3.0);",
            ]
        );
        // Parse cross-check against the hand-built ShortestPath plan (all sources).
        // `*` is min 0 (the seed is a zero-length path to itself).
        let hand = Plan::Scan { label: None }
            .shortest_path(
                0,
                Dir::Out,
                &["LINK".to_string()],
                0,
                None,
                crate::ir::ShortestSelector::Any,
            )
            .project(vec![(
                "len".into(),
                Expr::PathAccess {
                    part: PathPart::Length,
                },
            )]);
        assert_same(
            "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) RETURN path_length(p) AS len",
            &hand,
            &store,
        );
    }

    #[test]
    fn any_shortest_nodes_reconstructs_the_chain() {
        let store = chain();
        // The full path a→b→c→d is reconstructed (BFS predecessors), so nodes(p)
        // is the node-id chain [0,1,2,3], not just the endpoint.
        let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) \
                 WHERE x.name = 'a' AND y.name = 'd' RETURN nodes(p) AS ns";
        let out = run(&super::parse(q).unwrap(), &store);
        assert_eq!(out.rows.len(), 1);
        let ids: Vec<f64> = match &out.rows[0][0] {
            Value::List(items) => items
                .iter()
                .map(|v| match v {
                    Value::Num(x) => *x,
                    o => panic!("expected Num in path, got {o:?}"),
                })
                .collect(),
            o => panic!("expected a List from nodes(p), got {o:?}"),
        };
        assert_eq!(ids, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn any_shortest_relationships_are_the_traversed_edges() {
        let store = chain();
        // Edges are created a→b, b→c, c→d (ids 0,1,2). The shortest path a→d
        // traverses all three, in order — relationships(p) recovers them.
        let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) \
                 WHERE x.name = 'a' AND y.name = 'd' RETURN relationships(p) AS es";
        let out = run(&super::parse(q).unwrap(), &store);
        assert_eq!(out.rows.len(), 1);
        let eids: Vec<f64> = match &out.rows[0][0] {
            Value::List(items) => items
                .iter()
                .map(|v| match v {
                    Value::Num(x) => *x,
                    o => panic!("expected Num edge id, got {o:?}"),
                })
                .collect(),
            o => panic!("expected a List from relationships(p), got {o:?}"),
        };
        assert_eq!(eids, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn any_shortest_elements_interleave_nodes_and_edges() {
        let store = chain();
        // elements(p) for a→d is n0,e0,n1,e1,n2,e2,n3 = 0,0,1,1,2,2,3 (node ids
        // 0..3, edge ids 0..2). Nodes and edges are both Num here, so compare the
        // flat sequence.
        let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) \
                 WHERE x.name = 'a' AND y.name = 'd' RETURN elements(p) AS els";
        let out = run(&super::parse(q).unwrap(), &store);
        assert_eq!(out.rows.len(), 1);
        let seq: Vec<f64> = match &out.rows[0][0] {
            Value::List(items) => items
                .iter()
                .map(|v| match v {
                    Value::Num(x) => *x,
                    o => panic!("expected Num, got {o:?}"),
                })
                .collect(),
            o => panic!("expected a List from elements(p), got {o:?}"),
        };
        assert_eq!(seq, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn path_accessor_requires_a_path_variable() {
        // A path accessor on a non-path expression is a clear parse error.
        let err = super::parse("MATCH (a:Person) RETURN relationships(a.name) AS x").unwrap_err();
        assert!(err.contains("path variable"), "got: {err}");
    }

    #[test]
    fn named_path_over_plain_pattern_is_accepted() {
        // A named path does NOT require a shortest-path selector: `MATCH p = <plain
        // pattern>` binds the pattern's (WALK/TRAIL) lineage, readable via
        // path_length(p)/nodes(p)/edges(p). Both a fixed hop and a var-length body
        // parse.
        assert!(super::parse("MATCH p = (a)-[:LINK]->(b) RETURN p").is_ok());
        assert!(super::parse("MATCH p = (a)-[:LINK]->{1,3}(b) RETURN path_length(p) AS n").is_ok());
    }

    #[test]
    fn any_shortest_requires_a_quantifier() {
        let err =
            super::parse("MATCH p = ANY SHORTEST (a)-[:LINK]->(b) RETURN a.name AS a").unwrap_err();
        assert!(err.contains("quantifier"), "got: {err}");
    }

    #[test]
    fn temporal_literals_render_and_compare() {
        let store = social();
        // The three zone-less literals parse and round-trip to their ISO form.
        let out = run(
            &super::parse(
                "MATCH (p:Person) RETURN DATE '2024-01-15' AS d, TIME '13:45:06' AS t, \
                 DATETIME '2024-01-15T09:00:00' AS dt",
            )
            .unwrap(),
            &store,
        );
        let iso = |v: &Value| match v {
            Value::Temporal(t) => t.format(),
            o => panic!("expected Temporal, got {o:?}"),
        };
        assert_eq!(iso(&col(&out, 0, "d")), "2024-01-15");
        assert_eq!(iso(&col(&out, 0, "t")), "13:45:06");
        assert_eq!(iso(&col(&out, 0, "dt")), "2024-01-15T09:00:00");

        // Date ordering as a constant predicate: earlier < later keeps all rows,
        // the reverse keeps none, and equality holds.
        let count = |q: &str| run(&super::parse(q).unwrap(), &store).rows.len();
        assert_eq!(
            count("MATCH (p:Person) WHERE DATE '2024-01-01' < DATE '2024-06-01' RETURN p.name"),
            3
        );
        assert_eq!(
            count("MATCH (p:Person) WHERE DATE '2024-06-01' < DATE '2024-01-01' RETURN p.name"),
            0
        );
        assert_eq!(
            count("MATCH (p:Person) WHERE DATE '2024-01-01' = DATE '2024-01-01' RETURN p.name"),
            3
        );
        // A malformed literal is a parse error.
        assert!(super::parse("MATCH (p:Person) RETURN DATE '2024-13-01' AS d").is_err());
    }

    #[test]
    fn duration_and_zoned_literals() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) RETURN DURATION 'P1Y2M' AS d, \
                 ZONED DATETIME '2024-01-15T12:00:00+01:00' AS z, \
                 ZONED TIME '13:45:00Z' AS zt",
            )
            .unwrap(),
            &store,
        );
        let iso = |v: &Value| match v {
            Value::Temporal(t) => t.format(),
            o => panic!("expected Temporal, got {o:?}"),
        };
        assert_eq!(iso(&col(&out, 0, "d")), "P14M"); // 1Y2M = 14 months, canonical
        assert_eq!(iso(&col(&out, 0, "z")), "2024-01-15T12:00:00+01:00");
        assert_eq!(iso(&col(&out, 0, "zt")), "13:45:00Z");
        // A malformed duration literal is a parse error.
        assert!(super::parse("MATCH (p:Person) RETURN DURATION 'nope' AS d").is_err());
    }

    #[test]
    fn temporal_component_accessors() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) RETURN year(DATE '2024-03-15') AS y, \
                 month(DATE '2024-03-15') AS mo, day(DATE '2024-03-15') AS d, \
                 hour(TIME '13:45:06') AS h, minute(TIME '13:45:06') AS mi, \
                 second(TIME '13:45:06') AS se, year(DATETIME '2020-07-04T09:30:00') AS dty",
            )
            .unwrap(),
            &store,
        );
        for (name, want) in [
            ("y", 2024.0),
            ("mo", 3.0),
            ("d", 15.0),
            ("h", 13.0),
            ("mi", 45.0),
            ("se", 6.0),
            ("dty", 2020.0),
        ] {
            assert_eq!(num(&col(&out, 0, name)), want, "{name}");
        }
        // A component undefined for the kind FAULTS with E_INVALID_VALUE (year of a
        // time, hour of a date) — matching core, which errors rather than NULLs.
        for q in [
            "MATCH (p:Person) RETURN year(TIME '01:02:03') AS y",
            "MATCH (p:Person) RETURN hour(DATE '2024-01-01') AS h",
        ] {
            let err = crate::exec::try_run(&super::parse(q).unwrap(), &store);
            assert!(
                matches!(&err, Err(e) if e.contains("E_INVALID_VALUE")),
                "expected fault for `{q}`, got {err:?}"
            );
        }
    }

    #[test]
    fn temporal_constructors_and_coercion() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) RETURN \
                 date('2024-03-15') AS d1, \
                 datetime('2024-03-15') AS d2, \
                 date(DATETIME '2024-03-15T09:30:00') AS d3, \
                 datetime(DATE '2024-03-15') AS d4, \
                 local_time(DATETIME '2024-03-15T09:30:45') AS d5, \
                 duration('P1Y2M') AS d6",
            )
            .unwrap(),
            &store,
        );
        let iso = |v: &Value| match v {
            Value::Temporal(t) => t.format(),
            o => panic!("expected Temporal, got {o:?}"),
        };
        assert_eq!(iso(&col(&out, 0, "d1")), "2024-03-15"); // parse
        assert_eq!(iso(&col(&out, 0, "d2")), "2024-03-15T00:00:00"); // date-str → midnight
        assert_eq!(iso(&col(&out, 0, "d3")), "2024-03-15"); // datetime → date part
        assert_eq!(iso(&col(&out, 0, "d4")), "2024-03-15T00:00:00"); // date → midnight
        assert_eq!(iso(&col(&out, 0, "d5")), "09:30:45"); // datetime → time part
        assert_eq!(iso(&col(&out, 0, "d6")), "P14M"); // 1Y2M canonical
                                                      // A malformed constructor argument is NULL, not an error.
        let out2 = run(
            &super::parse("MATCH (p:Person) RETURN date('garbage') AS d").unwrap(),
            &store,
        );
        assert!(col(&out2, 0, "d").is_null());
    }

    #[test]
    fn duration_between_is_exact() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) RETURN \
                 duration_between(DATE '2020-01-15', DATE '2020-04-20') AS a, \
                 duration_between(DATETIME '2020-01-01T00:00:00', \
                 DATETIME '2020-01-01T01:01:01') AS b, \
                 duration_between(DATE '2020-01-01', DATETIME '2020-01-01T00:00:00') AS c",
            )
            .unwrap(),
            &store,
        );
        let iso = |v: &Value| match v {
            Value::Temporal(t) => t.format(),
            o => panic!("expected Temporal, got {o:?}"),
        };
        assert_eq!(iso(&col(&out, 0, "a")), "P96D"); // 96 days (2020 is a leap year)
        assert_eq!(iso(&col(&out, 0, "b")), "PT3661S"); // 1h1m1s
        assert!(col(&out, 0, "c").is_null()); // cross-kind → NULL
    }

    #[test]
    fn temporal_arithmetic() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) RETURN \
                 DATE '2024-01-31' + DURATION 'P1M' AS clamp_leap, \
                 DATE '2023-01-31' + DURATION 'P1M' AS clamp, \
                 DATE '2024-01-15' + DURATION 'P10D' AS plus_days, \
                 DATETIME '2024-01-15T10:00:00' + DURATION 'PT3661S' AS dt_plus, \
                 DATE '2024-04-20' - DATE '2024-01-15' AS span, \
                 DURATION 'P1M' + DURATION 'P2D' AS dsum, \
                 DURATION 'P2D' * 3 AS dscale",
            )
            .unwrap(),
            &store,
        );
        let iso = |v: &Value| match v {
            Value::Temporal(t) => t.format(),
            o => panic!("expected Temporal, got {o:?}"),
        };
        assert_eq!(iso(&col(&out, 0, "clamp_leap")), "2024-02-29"); // Jan31+1M → Feb29 (leap)
        assert_eq!(iso(&col(&out, 0, "clamp")), "2023-02-28"); // non-leap → Feb28
        assert_eq!(iso(&col(&out, 0, "plus_days")), "2024-01-25");
        assert_eq!(iso(&col(&out, 0, "dt_plus")), "2024-01-15T11:01:01"); // +1h1m1s
        assert_eq!(iso(&col(&out, 0, "span")), "P96D"); // Jan15→Apr20, leap year
        assert_eq!(iso(&col(&out, 0, "dsum")), "P1M2D");
        assert_eq!(iso(&col(&out, 0, "dscale")), "P6D");

        // A non-integer duration scale is NULL (no meaningful fractional month).
        let out2 = run(
            &super::parse("MATCH (p:Person) RETURN DURATION 'P2D' * 1.5 AS d").unwrap(),
            &store,
        );
        assert!(col(&out2, 0, "d").is_null());
    }

    #[test]
    fn temporal_arithmetic_overflow_throws() {
        let store = social();
        // Adding ~8.3M years leaves the representable i32-day date range: a THROWN
        // fault (E_INVALID_VALUE via the fallible pipeline), not a silent null.
        let plan =
            super::parse("MATCH (p:Person) RETURN DATE '2024-01-01' + DURATION 'P100000000M' AS d")
                .unwrap();
        let err = crate::exec::try_run(&plan, &store).unwrap_err();
        assert!(err.contains("E_INVALID_VALUE"), "got: {err}");
    }

    #[test]
    fn record_literal_and_field_access() {
        use crate::ir::{CompareOp, Expr, Plan};
        let store = social();
        // Build a record from a matched node, carry it through WITH, read fields.
        let out = run(
            &super::parse(
                "MATCH (p:Person) WHERE p.name = 'alice' \
                 WITH {name: p.name, age: p.age} AS r RETURN r.name AS n, r.age AS a, \
                 r.missing AS m",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(out.rows.len(), 1);
        assert!(crate::value::equals(&col(&out, 0, "n"), &s("alice")));
        assert_eq!(num(&col(&out, 0, "a")), 30.0);
        assert!(col(&out, 0, "m").is_null()); // absent field → NULL

        // A returned record has its keys sorted (canonical), whatever the literal
        // order; cross-checked against the hand-built Record plan.
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(Expr::Compare {
            op: CompareOp::Eq,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "name".into(),
            }),
            right: Box::new(Expr::Lit(Value::Str("alice".into()))),
        })
        .project(vec![(
            "r".into(),
            Expr::Record {
                fields: vec![
                    ("b".into(), Expr::Lit(Value::Num(2.0))),
                    ("a".into(), Expr::Lit(Value::Num(1.0))),
                ],
            },
        )]);
        let q = "MATCH (p:Person) WHERE p.name = 'alice' RETURN {b: 2, a: 1} AS r";
        assert_same(q, &hand, &store);
        let out2 = run(&super::parse(q).unwrap(), &store);
        match &col(&out2, 0, "r") {
            Value::Record(f) => {
                assert_eq!(f[0].0.as_ref(), "a"); // sorted
                assert_eq!(f[1].0.as_ref(), "b");
            }
            o => panic!("expected a Record, got {o:?}"),
        }
    }

    #[test]
    fn field_access_on_a_record_literal() {
        use crate::ir::{Expr, Plan};
        let store = social();
        // `{lit}.field` and a chained `.outer.inner` on nested record literals.
        let out = run(
            &super::parse(
                "MATCH (p:Person) WHERE p.name = 'alice' RETURN \
                 {a: 1, b: 2}.b AS x, {outer: {inner: 7}}.outer.inner AS y, \
                 {a: 1}.missing AS m",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(num(&col(&out, 0, "x")), 2.0);
        assert_eq!(num(&col(&out, 0, "y")), 7.0);
        assert!(col(&out, 0, "m").is_null()); // absent field → NULL

        // Cross-check `{a: 1}.a` against the hand-built Field(Record) plan.
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(Expr::Compare {
            op: crate::ir::CompareOp::Eq,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "name".into(),
            }),
            right: Box::new(Expr::Lit(Value::Str("alice".into()))),
        })
        .project(vec![(
            "x".into(),
            Expr::Field {
                base: Box::new(Expr::Record {
                    fields: vec![("a".into(), Expr::Lit(Value::Num(1.0)))],
                }),
                key: "a".into(),
            },
        )]);
        assert_same(
            "MATCH (p:Person) WHERE p.name = 'alice' RETURN {a: 1}.a AS x",
            &hand,
            &store,
        );
    }

    #[test]
    fn nested_field_access_on_a_stored_record() {
        let mut store = social();
        // Store a record property, then read nested fields via `n.rec.field`.
        crate::exec::execute(
            &super::parse(
                "MATCH (p:Person) WHERE p.name = 'alice' \
                 SET p.meta = {city: 'NYC', zip: 10001}",
            )
            .unwrap(),
            &mut store,
        )
        .unwrap();
        let out = run(
            &super::parse(
                "MATCH (p:Person) WHERE p.name = 'alice' \
                 RETURN p.meta.city AS c, p.meta.zip AS z, p.meta.absent AS a",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(out.rows.len(), 1);
        assert!(crate::value::equals(&col(&out, 0, "c"), &s("NYC")));
        assert_eq!(num(&col(&out, 0, "z")), 10001.0);
        assert!(col(&out, 0, "a").is_null()); // absent nested field → NULL
    }

    #[test]
    fn stored_dates_round_trip_and_filter() {
        let mut store = social();
        // Store birthdates on two Persons, then find those born before 2000.
        for (who, born) in [("alice", "1990-05-01"), ("bob", "2005-03-03")] {
            let q = format!("MATCH (p:Person) WHERE p.name = '{who}' SET p.born = DATE '{born}'");
            crate::exec::execute(&super::parse(&q).unwrap(), &mut store).unwrap();
        }
        // alice(1990) qualifies; bob(2005) does not; carol has no `born` (NULL,
        // so the comparison is UNKNOWN and she is filtered out).
        assert_eq!(
            names(
                &store,
                "MATCH (p:Person) WHERE p.born < DATE '2000-01-01' RETURN p.name AS name"
            ),
            vec!["alice"]
        );
        // The stored date reads back as its ISO string.
        let out = run(
            &super::parse("MATCH (p:Person) WHERE p.name = 'alice' RETURN p.born AS born").unwrap(),
            &store,
        );
        match &col(&out, 0, "born") {
            Value::Temporal(t) => assert_eq!(t.format(), "1990-05-01"),
            o => panic!("expected Temporal, got {o:?}"),
        }
    }

    #[test]
    fn cdc_observes_a_committed_insert() {
        use crate::store::Change;
        let mut store = Builder::default().build();
        crate::exec::execute(
            &super::parse("INSERT (:P {name: 'a'}), (:P {name: 'b'})").unwrap(),
            &mut store,
        )
        .unwrap();
        // The INSERT is txn-wrapped, so its two node adds surface as CDC changes.
        assert_eq!(
            store.last_commit_changes(),
            &[Change::NodeAdded(0), Change::NodeAdded(1)]
        );
    }

    #[test]
    fn required_constraint_rejects_insert_without_the_key() {
        let mut store = Builder::default().build();
        store.create_required_constraint("User", "email").unwrap();
        // INSERT carrying the required key succeeds.
        crate::exec::execute(
            &super::parse("INSERT (:User {email: 'a@x'})").unwrap(),
            &mut store,
        )
        .unwrap();
        // INSERT missing it is rejected and rolled back (node count unchanged).
        let before = store.node_count();
        let err = crate::exec::execute(
            &super::parse("INSERT (:User {name: 'b'})").unwrap(),
            &mut store,
        )
        .unwrap_err();
        assert!(err.contains("E_REQUIRED"), "got: {err}");
        assert_eq!(store.node_count(), before);
    }

    /// A directed triangle a→b→c→a (ids 0,1,2) + an isolated node d (3).
    fn triangle_store() -> Store {
        let mut b = Builder::default();
        let a = b.node(&["N"], &[]);
        let bb = b.node(&["N"], &[]);
        let c = b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        b.edge(a, bb, "R");
        b.edge(bb, c, "R");
        b.edge(c, a, "R");
        b.build()
    }

    #[test]
    fn call_degree_procedure_yield_and_default() {
        use crate::ir::{Expr, Plan};
        let store = triangle_store();
        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        // Out-degrees: each triangle node 1, the isolated node 0.
        let want = vec![(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)];
        assert_eq!(rows_of("CALL degree() YIELD node, degree"), want);
        // No YIELD → the default [node, <result>] columns.
        assert_eq!(rows_of("CALL degree()"), want);
        // YIELD renames the output columns.
        let out = run(
            &super::parse("CALL degree() YIELD node AS n, degree AS d").unwrap(),
            &store,
        );
        assert_eq!(out.names, vec!["n".to_string(), "d".to_string()]);

        // Parse→run matches the hand-built plan (CallProcedure under a Project).
        let hand = Plan::CallProcedure {
            name: "degree".into(),
            config: vec![],
        }
        .project(vec![
            ("node".into(), Expr::Slot(0)),
            ("degree".into(), Expr::Slot(1)),
        ]);
        assert_same("CALL degree()", &hand, &store);
    }

    #[test]
    fn call_closeness_procedure_yields_centrality() {
        let store = triangle_store();
        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        // Directed OUT triangle: each member Σdist=3 → 1/3; the isolated node → 0.
        let want = vec![
            (0.0, 1.0 / 3.0),
            (1.0, 1.0 / 3.0),
            (2.0, 1.0 / 3.0),
            (3.0, 0.0),
        ];
        assert_eq!(rows_of("CALL closeness() YIELD node, centrality"), want);
        // Default columns (no YIELD) are [node, centrality].
        assert_eq!(rows_of("CALL closeness()"), want);
        let out = run(&super::parse("CALL closeness()").unwrap(), &store);
        assert_eq!(
            out.names,
            vec!["node".to_string(), "centrality".to_string()]
        );
    }

    #[test]
    fn call_scc_procedure_yields_component_id() {
        let store = triangle_store();
        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        // The directed triangle {0,1,2} is one SCC (rep 0); the isolated node is {3}.
        let want = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 3.0)];
        assert_eq!(
            rows_of("CALL strongly_connected_components() YIELD node, componentId"),
            want
        );
        assert_eq!(rows_of("CALL strongly_connected_components()"), want);
        let out = run(
            &super::parse("CALL strongly_connected_components()").unwrap(),
            &store,
        );
        assert_eq!(
            out.names,
            vec!["node".to_string(), "componentId".to_string()]
        );
    }

    #[test]
    fn call_on_cycle_procedure_yields_on_cycle_flag() {
        let store = triangle_store();
        let rows_of = |q: &str| -> Vec<(f64, bool)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| {
                    let b = match &r[1] {
                        Value::Bool(b) => *b,
                        other => panic!("onCycle should be a Bool, got {other:?}"),
                    };
                    (node_id(&r[0]), b)
                })
                .collect()
        };
        // The triangle members are on a cycle (Bool true, matching core's `onCycle`
        // type); the isolated node is not (Bool false).
        let want = vec![(0.0, true), (1.0, true), (2.0, true), (3.0, false)];
        assert_eq!(rows_of("CALL on_cycle() YIELD node, onCycle"), want);
        assert_eq!(rows_of("CALL on_cycle()"), want);
        let out = run(&super::parse("CALL on_cycle()").unwrap(), &store);
        assert_eq!(out.names, vec!["node".to_string(), "onCycle".to_string()]);
    }

    #[test]
    fn call_betweenness_procedure_yields_centrality() {
        let store = triangle_store();
        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        // Directed triangle: each member is the sole intermediary of one 2-hop path
        // → 1.0; the isolated node → 0.0.
        let want = vec![(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)];
        assert_eq!(rows_of("CALL betweenness() YIELD node, centrality"), want);
        assert_eq!(rows_of("CALL betweenness()"), want);
        let out = run(&super::parse("CALL betweenness()").unwrap(), &store);
        assert_eq!(
            out.names,
            vec!["node".to_string(), "centrality".to_string()]
        );
    }

    #[test]
    fn call_shortest_path_procedure_yields_distance() {
        let store = triangle_store();
        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        // OUT from source "0" on the triangle: hop distances 0,1,2 to the three
        // members; the isolated node is unreachable and absent.
        let want = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        assert_eq!(
            rows_of("CALL shortest_path({source: '0'}) YIELD node, distance"),
            want
        );
        assert_eq!(rows_of("CALL shortest_path({source: '0'})"), want);
        let out = run(
            &super::parse("CALL shortest_path({source: '0'})").unwrap(),
            &store,
        );
        assert_eq!(out.names, vec!["node".to_string(), "distance".to_string()]);
        // A `target` restricts the result to just that vertex's distance.
        assert_eq!(
            rows_of("CALL shortest_path({source: '0', target: '2'}) YIELD node, distance"),
            vec![(2.0, 2.0)]
        );
        // An unreachable target (the isolated node) yields nothing.
        assert!(rows_of("CALL shortest_path({source: '0', target: '3'})").is_empty());
    }

    #[test]
    fn call_shortest_path_astar() {
        // 0→1 (w=10), 0→2 (1), 2→1 (1): A* 0→1 returns the exact shortest distance (2),
        // the same as Dijkstra, guided by the algorithm:'astar' backend.
        let mut bld = Builder::default();
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[]);
        let mut store = bld.build();
        let e0 = store.add_edge(0, 1, "R");
        store.set_edge_prop(e0, "w", Value::Num(10.0));
        let e1 = store.add_edge(0, 2, "R");
        store.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = store.add_edge(2, 1, "R");
        store.set_edge_prop(e2, "w", Value::Num(1.0));

        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        assert_eq!(
            rows_of(
                "CALL shortest_path({source:'0', target:'1', weightProperty:'w', \
                 algorithm:'astar'}) YIELD node, distance"
            ),
            vec![(1.0, 2.0)]
        );
    }

    #[test]
    fn call_shortest_path_weighted() {
        // 0→1 (w=10), 0→2 (w=1), 2→1 (w=1): weighted 0→1 = 2 (light detour), while
        // unweighted 0→1 = 1 (direct edge).
        let mut bld = Builder::default();
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[]);
        let mut store = bld.build();
        let e0 = store.add_edge(0, 1, "R");
        store.set_edge_prop(e0, "w", Value::Num(10.0));
        let e1 = store.add_edge(0, 2, "R");
        store.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = store.add_edge(2, 1, "R");
        store.set_edge_prop(e2, "w", Value::Num(1.0));

        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        assert_eq!(
            rows_of("CALL shortest_path({source: '0', weightProperty: 'w'}) YIELD node, distance"),
            vec![(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]
        );
        // Same query without the weight → the hop distance to 1 is 1.
        assert_eq!(
            rows_of("CALL shortest_path({source: '0'})"),
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)]
        );
    }

    #[test]
    fn call_closeness_weighted() {
        // 0→1 (w=10), 0→2 (1), 2→1 (1): weighted closeness of 0 is 1/3 (Dijkstra
        // sum 3), vs unweighted 1/2 (hop sum 2).
        let mut bld = Builder::default();
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[]);
        let mut store = bld.build();
        let e0 = store.add_edge(0, 1, "R");
        store.set_edge_prop(e0, "w", Value::Num(10.0));
        let e1 = store.add_edge(0, 2, "R");
        store.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = store.add_edge(2, 1, "R");
        store.set_edge_prop(e2, "w", Value::Num(1.0));

        let close0 = |q: &str| -> f64 { num(&run(&super::parse(q).unwrap(), &store).rows[0][1]) };
        assert!(
            (close0("CALL closeness({weightProperty: 'w'}) YIELD node, centrality") - 1.0 / 3.0)
                .abs()
                < 1e-12
        );
        assert!((close0("CALL closeness()") - 1.0 / 2.0).abs() < 1e-12);
    }

    #[test]
    fn call_betweenness_weighted() {
        // Diamond with a heavy 2→3 branch: weighted betweenness routes all 0→3
        // dependency through node 1 (1.0), where unweighted splits it 0.5/0.5.
        let mut bld = Builder::default();
        for _ in 0..4 {
            bld.node(&["N"], &[]);
        }
        let mut store = bld.build();
        let e0 = store.add_edge(0, 1, "R");
        store.set_edge_prop(e0, "w", Value::Num(1.0));
        let e1 = store.add_edge(0, 2, "R");
        store.set_edge_prop(e1, "w", Value::Num(1.0));
        let e2 = store.add_edge(1, 3, "R");
        store.set_edge_prop(e2, "w", Value::Num(1.0));
        let e3 = store.add_edge(2, 3, "R");
        store.set_edge_prop(e3, "w", Value::Num(5.0));

        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        assert_eq!(
            rows_of("CALL betweenness({weightProperty: 'w'}) YIELD node, centrality"),
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 0.0)]
        );
        assert_eq!(
            rows_of("CALL betweenness()"),
            vec![(0.0, 0.0), (1.0, 0.5), (2.0, 0.5), (3.0, 0.0)]
        );
    }

    #[test]
    fn call_neighbor_aggregate_weighted() {
        // 0→1 (w=1, h=[2]), 0→2 (w=3, h=[4]): weighted mean at 0 is 14/(1+3)=3.5.
        let mut bld = Builder::default();
        let f = |x: f64| Value::List(vec![Value::Num(x)]);
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[("h", f(2.0))]);
        bld.node(&["N"], &[("h", f(4.0))]);
        let mut store = bld.build();
        let e0 = store.add_edge(0, 1, "R");
        store.set_edge_prop(e0, "w", Value::Num(1.0));
        let e1 = store.add_edge(0, 2, "R");
        store.set_edge_prop(e1, "w", Value::Num(3.0));

        let out = run(
            &super::parse(
                "CALL neighbor_aggregate({feature: 'h', op: 'mean', direction: 'out', \
                 weightProperty: 'w'}) YIELD node, vector",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(format!("{:?}", out.rows[0][1]), "List([Num(3.5)])");
    }

    #[test]
    fn call_neighbor_aggregate_gcn() {
        // 0→1, 0→2 (unweighted); h(1)=[2], h(2)=[4]. GCN sum at 0 folds each
        // contributor by 1/sqrt(deg_0·deg_nbr) = 1/sqrt(2).
        let mut bld = Builder::default();
        let f = |x: f64| Value::List(vec![Value::Num(x)]);
        bld.node(&["N"], &[]);
        bld.node(&["N"], &[("h", f(2.0))]);
        bld.node(&["N"], &[("h", f(4.0))]);
        let mut store = bld.build();
        store.add_edge(0, 1, "R");
        store.add_edge(0, 2, "R");

        let out = run(
            &super::parse(
                "CALL neighbor_aggregate({feature: 'h', op: 'sum', direction: 'out', \
                 norm: 'gcn'}) YIELD node, vector",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(
            format!("{:?}", out.rows[0][1]),
            "List([Num(4.242640687119285)])"
        );
    }

    #[test]
    fn call_personalized_pagerank_yields_score() {
        let store = triangle_store();
        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        // Seeding node "0" via the sourceNodes list makes 0 the strict max and leaves
        // the unreachable isolated node at 0; the yield column is `score`.
        let seeded = rows_of("CALL personalized_pagerank({sourceNodes: ['0']}) YIELD node, score");
        assert_eq!(seeded.len(), 4);
        assert!(seeded[0].1 > seeded[1].1 && seeded[0].1 > seeded[2].1);
        assert_eq!(seeded[3], (3.0, 0.0));
        // Default columns are [node, score].
        let out = run(
            &super::parse("CALL personalized_pagerank({sourceNodes: ['0']})").unwrap(),
            &store,
        );
        assert_eq!(out.names, vec!["node".to_string(), "score".to_string()]);
    }

    #[test]
    fn call_neighbor_aggregate_yields_vector() {
        // a(0)=[1,2], b(1)=[3,4]; a→b. OUT-sum at a folds b's vector; b has none.
        let mut bld = Builder::default();
        let vec = |xs: &[f64]| Value::List(xs.iter().map(|&x| Value::Num(x)).collect());
        let a = bld.node(&["N"], &[("h", vec(&[1.0, 2.0]))]);
        let b = bld.node(&["N"], &[("h", vec(&[3.0, 4.0]))]);
        bld.edge(a, b, "R");
        let store = bld.build();

        let out = run(
            &super::parse(
                "CALL neighbor_aggregate({feature: 'h', op: 'sum', direction: 'out'}) \
                 YIELD node, vector",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(out.names, vec!["node".to_string(), "vector".to_string()]);
        // Node a's aggregate is b's feature [3,4]; node b's is the zero vector.
        assert_eq!(out.rows.len(), 2);
        assert_eq!(
            format!("{:?}", out.rows[0][1]),
            "List([Num(3.0), Num(4.0)])"
        );
        assert_eq!(
            format!("{:?}", out.rows[1][1]),
            "List([Num(0.0), Num(0.0)])"
        );
    }

    #[test]
    fn call_peer_pressure_yields_cluster() {
        // Sink 1→0, 2→0, 3→0: node 0 joins cluster 1 (tie to smallest ext id); the
        // sources keep their own cluster. The yield column is `cluster`.
        let mut bld = Builder::default();
        let a = bld.node(&["N"], &[]);
        let x = bld.node(&["N"], &[]);
        let y = bld.node(&["N"], &[]);
        let z = bld.node(&["N"], &[]);
        bld.edge(x, a, "R");
        bld.edge(y, a, "R");
        bld.edge(z, a, "R");
        let store = bld.build();

        let rows_of = |q: &str| -> Vec<(f64, f64)> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| (node_id(&r[0]), num(&r[1])))
                .collect()
        };
        let want = vec![(0.0, 1.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        assert_eq!(rows_of("CALL peer_pressure() YIELD node, cluster"), want);
        assert_eq!(rows_of("CALL peer_pressure()"), want);
        let out = run(&super::parse("CALL peer_pressure()").unwrap(), &store);
        assert_eq!(out.names, vec!["node".to_string(), "cluster".to_string()]);
    }

    #[test]
    fn call_procedure_config_and_components() {
        let store = triangle_store();
        // degree with direction=both: each triangle node 2, isolated 0.
        let both: Vec<f64> = run(
            &super::parse("CALL degree({direction: 'both'}) YIELD degree").unwrap(),
            &store,
        )
        .rows
        .iter()
        .map(|r| num(&r[0]))
        .collect();
        assert_eq!(both, vec![2.0, 2.0, 2.0, 0.0]);
        // connected_components: triangle → component 0, isolated → 3.
        let comps: Vec<(f64, f64)> = run(
            &super::parse("CALL connected_components() YIELD node, componentId").unwrap(),
            &store,
        )
        .rows
        .iter()
        .map(|r| (node_id(&r[0]), num(&r[1])))
        .collect();
        assert_eq!(comps, vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 3.0)]);
    }

    #[test]
    fn call_procedure_errors() {
        // Unknown procedure and unknown YIELD column are both parse errors.
        assert!(super::parse("CALL bogus()").is_err());
        assert!(super::parse("CALL degree() YIELD nope").is_err());
    }

    #[test]
    fn where_filter_and_alias() {
        use crate::ir::{CompareOp, Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(Expr::Compare {
            op: CompareOp::Gt,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "age".into(),
            }),
            right: Box::new(Expr::Lit(Value::Num(28.0))),
        })
        .project(vec![(
            "who".into(),
            Expr::Prop {
                slot: 0,
                key: "name".into(),
            },
        )]);
        assert_same(
            "MATCH (p:Person) WHERE p.age > 28 RETURN p.name AS who",
            &hand,
            &store,
        );
    }

    #[test]
    fn one_hop_binds_both() {
        use crate::ir::{Dir, Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .project(vec![
            (
                "a".into(),
                Expr::Prop {
                    slot: 0,
                    key: "name".into(),
                },
            ),
            (
                "b".into(),
                Expr::Prop {
                    slot: 1,
                    key: "name".into(),
                },
            ),
        ]);
        assert_same(
            "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b",
            &hand,
            &store,
        );
    }

    #[test]
    fn two_hops_and_where_conjunction() {
        use crate::ir::{Dir, Expr, Plan};
        let store = social();
        // (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE a.name='alice' AND c.age>=40
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .expand(1, Dir::Out, &["KNOWS".to_string()])
        .filter(Expr::And(
            Box::new(Expr::Compare {
                op: crate::ir::CompareOp::Eq,
                left: Box::new(Expr::Prop {
                    slot: 0,
                    key: "name".into(),
                }),
                right: Box::new(Expr::Lit(Value::Str("alice".into()))),
            }),
            Box::new(Expr::Compare {
                op: crate::ir::CompareOp::Ge,
                left: Box::new(Expr::Prop {
                    slot: 2,
                    key: "age".into(),
                }),
                right: Box::new(Expr::Lit(Value::Num(40.0))),
            }),
        ))
        .project(vec![(
            "c".into(),
            Expr::Prop {
                slot: 2,
                key: "name".into(),
            },
        )]);
        assert_same(
            "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) \
             WHERE a.name = 'alice' AND c.age >= 40 RETURN c.name AS c",
            &hand,
            &store,
        );
        // and the direct answer, hand-checked: alice->b->c with c.age>=40 is
        // alice->bob->carol and alice->carol->? carol KNOWS nobody, so only carol.
        let out = run(&super::parse("MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE a.name = 'alice' AND c.age >= 40 RETURN c.name AS c").unwrap(), &store);
        assert_eq!(out.rows.len(), 1);
    }

    #[test]
    fn incoming_direction() {
        use crate::ir::{Dir, Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(Expr::Compare {
            op: crate::ir::CompareOp::Eq,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "name".into(),
            }),
            right: Box::new(Expr::Lit(Value::Str("carol".into()))),
        })
        .expand(0, Dir::In, &["KNOWS".to_string()])
        .project(vec![(
            "who".into(),
            Expr::Prop {
                slot: 1,
                key: "name".into(),
            },
        )]);
        assert_same(
            "MATCH (c:Person)<-[:KNOWS]-(who) WHERE c.name = 'carol' RETURN who.name AS who",
            &hand,
            &store,
        );
    }

    #[test]
    fn parse_errors_are_reported_not_panicked() {
        assert!(super::parse("MATCH (p:Person").is_err()); // unclosed
        assert!(super::parse("MATCH (p:Person) RETURN q.name").is_err()); // unknown var
        assert!(super::parse("RETURN 1").is_ok()); // bare RETURN is a valid statement
        assert!(super::parse("RETURN").is_err()); // …but RETURN needs at least one item
        assert!(super::parse("MATCH (p:Person) WHERE p.age > RETURN p.name").is_err());
    }

    // --- part 2: aggregation, DISTINCT, ORDER/SKIP/LIMIT ---

    fn num(v: &Value) -> f64 {
        match v {
            Value::Num(x) => *x,
            other => panic!("expected number, got {other:?}"),
        }
    }
    fn col(rows: &Rows, r: usize, name: &str) -> Value {
        let i = rows.names.iter().position(|n| n == name).expect("column");
        rows.rows[r][i].clone()
    }

    /// The numeric id of a NODE-element result map (`{id: "N", labels, properties}`),
    /// which is how a node binding now renders (matching core).
    fn node_id(v: &Value) -> f64 {
        match v {
            Value::Map(m) => m
                .iter()
                .find_map(|(k, val)| match (k, val) {
                    (Value::Str(k), Value::Str(id)) if &**k == "id" => id.parse().ok(),
                    _ => None,
                })
                .expect("node map carries a string id"),
            other => num(other),
        }
    }

    #[test]
    fn scalar_count_star() {
        let store = social();
        let out = run(
            &super::parse("MATCH (p:Person) RETURN count(*) AS c").unwrap(),
            &store,
        );
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&col(&out, 0, "c")), 3.0);
    }

    #[test]
    fn group_count_by_property() {
        let store = social();
        // group people by age bucket... simpler: count Persons by their own name
        // (each unique) is 1 each — instead group by a shared value. Use KNOWS
        // out-degree: (a)-[:KNOWS]->(b) grouped by a.name.
        let out = run(
            &super::parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS who, count(*) AS deg")
                .unwrap(),
            &store,
        );
        let mut got: Vec<(String, f64)> = out
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(w), Value::Num(d)) => (w.to_string(), *d),
                _ => panic!("shape"),
            })
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got, vec![("alice".into(), 2.0), ("bob".into(), 1.0)]);
    }

    #[test]
    fn sum_min_max_avg_match_hand_built() {
        use crate::ir::{Agg, AggFn, Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .aggregate(
            vec![],
            vec![
                Agg {
                    func: AggFn::Sum,
                    arg: Some(Expr::Prop {
                        slot: 0,
                        key: "age".into(),
                    }),
                    distinct: false,
                    name: "s".into(),
                    frac: None,
                },
                Agg {
                    func: AggFn::Avg,
                    arg: Some(Expr::Prop {
                        slot: 0,
                        key: "age".into(),
                    }),
                    distinct: false,
                    name: "a".into(),
                    frac: None,
                },
            ],
        );
        assert_same(
            "MATCH (p:Person) RETURN sum(p.age) AS s, avg(p.age) AS a",
            &hand,
            &store,
        );
    }

    #[test]
    fn return_distinct() {
        let store = social();
        // distinct set of nodes reachable by KNOWS: {bob, carol}
        let out = run(
            &super::parse("MATCH (a:Person)-[:KNOWS]->(b) RETURN DISTINCT b.name AS who").unwrap(),
            &store,
        );
        let mut got: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                _ => panic!(),
            })
            .collect();
        got.sort();
        assert_eq!(got, vec!["bob", "carol"]);
    }

    #[test]
    fn order_by_limit_top_k() {
        let store = social();
        // oldest two people by age desc: carol(40), alice(30)
        let out = run(
            &super::parse(
                "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age DESC LIMIT 2",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(out.rows.len(), 2);
        assert_eq!(num(&col(&out, 0, "age")), 40.0);
        assert_eq!(num(&col(&out, 1, "age")), 30.0);
    }

    #[test]
    fn order_by_skip_limit_window() {
        let store = social();
        // ascending age: bob(25), alice(30), carol(40); skip 1 limit 1 -> alice
        let out = run(
            &super::parse("MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age ASC SKIP 1 LIMIT 1").unwrap(),
            &store,
        );
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&col(&out, 0, "age")), 30.0);
    }

    #[test]
    fn order_by_aggregate_alias() {
        let store = social();
        // out-degree desc: alice(2) then bob(1)
        let out = run(
            &super::parse(
                "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS who, count(*) AS deg ORDER BY deg DESC",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(out.rows.len(), 2);
        assert_eq!(num(&col(&out, 0, "deg")), 2.0);
        assert_eq!(num(&col(&out, 1, "deg")), 1.0);
    }

    #[test]
    fn order_by_unknown_column_errors() {
        assert!(super::parse("MATCH (p:Person) RETURN p.name AS name ORDER BY age").is_err());
    }

    // --- part 3: comma-join and variable-length ---

    #[test]
    fn comma_join_shared_variable() {
        use crate::ir::{Dir, Expr, Plan};
        let store = social();
        // (a)-[:KNOWS]->(b), (a)-[:WORKS_ON]->(c) sharing a. Only alice has both.
        let left = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["KNOWS".to_string()]);
        let right = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["WORKS_ON".to_string()]);
        let hand = Plan::join(left, right, vec![(0, 0)]).project(vec![
            (
                "a".into(),
                Expr::Prop {
                    slot: 0,
                    key: "name".into(),
                },
            ),
            (
                "b".into(),
                Expr::Prop {
                    slot: 1,
                    key: "name".into(),
                },
            ),
            (
                "c".into(),
                Expr::Prop {
                    slot: 3,
                    key: "name".into(),
                },
            ),
        ]);
        assert_same(
            "MATCH (a:Person)-[:KNOWS]->(b), (a:Person)-[:WORKS_ON]->(c) \
             RETURN a.name AS a, b.name AS b, c.name AS c",
            &hand,
            &store,
        );
        // hand-checked: alice KNOWS {bob,carol} x WORKS_ON {graphdb} = 2 rows.
        let out = run(
            &super::parse(
                "MATCH (a:Person)-[:KNOWS]->(b), (a:Person)-[:WORKS_ON]->(c) RETURN c.name AS c",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(out.rows.len(), 2);
    }

    #[test]
    fn var_length_range() {
        use crate::ir::{Dir, Expr, PathMode, Plan};
        let store = social();
        // (a)-[:KNOWS]->{1,2}(b) from alice: b(len1)={bob,carol}, then len2 from
        // those: bob->carol. Trail. Cross-check vs hand-built VarLength.
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .filter(Expr::Compare {
            op: crate::ir::CompareOp::Eq,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "name".into(),
            }),
            right: Box::new(Expr::Lit(Value::Str("alice".into()))),
        })
        .var_length(0, Dir::Out, &["KNOWS".to_string()], 1, 2, PathMode::Trail)
        .project(vec![(
            "b".into(),
            Expr::Prop {
                slot: 1,
                key: "name".into(),
            },
        )]);
        assert_same(
            "MATCH (a:Person)-[:KNOWS]->{1,2}(b) WHERE a.name = 'alice' RETURN b.name AS b",
            &hand,
            &store,
        );
    }

    #[test]
    fn var_length_exact_and_open() {
        let store = social();
        // exact {2}: alice's 2-hop KNOWS endpoints = {carol} (alice->bob->carol).
        let out = run(
            &super::parse(
                "MATCH (a:Person)-[:KNOWS]->{2}(b) WHERE a.name = 'alice' RETURN b.name AS b",
            )
            .unwrap(),
            &store,
        );
        let mut got: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                _ => panic!(),
            })
            .collect();
        got.sort();
        assert_eq!(got, vec!["carol"]);

        // open {1,} is accepted (capped) and reaches everyone reachable.
        assert!(super::parse(
            "MATCH (a:Person)-[:KNOWS]->{1,}(b) WHERE a.name = 'alice' RETURN b.name AS b"
        )
        .is_ok());
    }

    #[test]
    fn bad_quantifier_errors() {
        assert!(super::parse("MATCH (a:Person)-[:KNOWS]->{3,1}(b) RETURN a.name AS a").is_err());
    }

    /// The algebraic degree-product count for a bounded OUT var-length must equal the
    /// DFS enumeration — on a graph large enough that the formula path FIRES, and
    /// with a self-loop that exercises the reused-edge trail correction.
    /// The 3-hop edge-product `count(*)` must equal the DFS enumeration — on a graph
    /// with a cycle and a self-loop, since a FIXED chain is a WALK (edges may repeat,
    /// no trail correction).
    #[test]
    fn three_hop_product_count_matches_enumeration() {
        use crate::store::Builder;
        let mut b = Builder::default();
        b.node(&["N"], &[]);
        b.node(&["N"], &[]);
        b.node(&["N"], &[]); // a=0, b=1, c=2
        b.edge(0, 1, "R");
        b.edge(1, 2, "R");
        b.edge(2, 0, "R"); // 3-cycle
        b.edge(0, 0, "R"); // self-loop
        let st = b.build();
        let q = "MATCH (a:N)-[:R]->()-[:R]->()-[:R]->(d)";
        let count = match &run(
            &super::parse(&format!("{q} RETURN count(*) AS c")).unwrap(),
            &st,
        )
        .rows[0][0]
        {
            Value::Num(x) => *x as usize,
            other => panic!("not a count: {other:?}"),
        };
        let enumerated = run(&super::parse(&format!("{q} RETURN d.z AS d")).unwrap(), &st)
            .rows
            .len();
        assert_eq!(count, enumerated, "product count != enumerated walks");
    }

    #[test]
    fn edge_filtered_count_matches_enumeration() {
        use crate::store::Builder;
        use crate::value::Value;
        // A small graph with a numeric edge property `w`; the streaming edge-filtered
        // count must equal the enumerated matching-row count for each predicate.
        let mut b = Builder::default();
        for _ in 0..200 {
            b.node(&["P"], &[]);
        }
        for i in 0u32..200 {
            for d in 0u32..3 {
                b.edge(i, (i * 7 + d * 13 + 1) % 200, "R");
            }
        }
        let mut st = b.build();
        for eid in st.all_edges() {
            st.set_edge_prop(eid, "w", Value::Num(f64::from(eid % 10)));
        }
        let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x as usize,
            other => panic!("not a count: {other:?}"),
        };
        for wc in ["r.w > 5", "r.w >= 2 AND r.w < 5", "r.w = 3", "3 > r.w"] {
            let c = count(&format!(
                "MATCH (a:P)-[r:R]->(b) WHERE {wc} RETURN count(*) AS c"
            ));
            let rows = run(
                &super::parse(&format!(
                    "MATCH (a:P)-[r:R]->(b) WHERE {wc} RETURN r.w AS w"
                ))
                .unwrap(),
                &st,
            )
            .rows
            .len();
            assert_eq!(c, rows, "edge-filtered count != enumerated for `{wc}`");
        }
    }

    #[test]
    fn streaming_num_filtered_count_matches_enumeration() {
        use crate::store::Builder;
        // Includes a NULL age (every 11th) and a NaN age (every 13th) so the count's
        // NULL-gating and NaN-drops-from-ordering rules are exercised; the streaming
        // count must equal the enumerated survivor count for each predicate spelling.
        let mut b = Builder::default();
        for i in 0..3000u32 {
            let mut props = vec![("name", Value::Str(format!("n{i}").into()))];
            let age = if i % 13 == 0 {
                Some(f64::NAN)
            } else if i % 11 == 0 {
                None
            } else {
                Some(f64::from(i % 100))
            };
            if let Some(a) = age {
                props.push(("age", Value::Num(a)));
            }
            b.node(&["P"], &props);
        }
        let st = b.build();
        let cnt = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x as usize,
            other => panic!("not a count: {other:?}"),
        };
        // Each: count(*) (streaming) == the enumerated row count (RETURN name).
        for wc in [
            "p.age > 50",
            "50 < p.age", // flipped operands, same predicate
            "p.age >= 10 AND p.age < 20",
            "p.age <= 10 AND p.age >= 5",
            "p.age = 42",
            "p.age <> 42",
        ] {
            let c = cnt(&format!("MATCH (p:P) WHERE {wc} RETURN count(*) AS c"));
            let rows = run(
                &super::parse(&format!("MATCH (p:P) WHERE {wc} RETURN p.name AS n")).unwrap(),
                &st,
            )
            .rows
            .len();
            assert_eq!(c, rows, "streaming count != enumerated for `{wc}`");
        }
    }

    #[test]
    fn string_concat_operator() {
        use crate::store::Builder;
        let mut b = Builder::default();
        b.node(
            &["P"],
            &[("name", Value::Str("ab".into())), ("age", Value::Num(7.0))],
        );
        let st = b.build();
        let one = |q: &str| -> Value { run(&super::parse(q).unwrap(), &st).rows[0][0].clone() };
        // string || string, chain, num coercion (7 → "7"), null propagation, list concat.
        assert!(
            matches!(one("MATCH (p:P) RETURN p.name || '!' AS x"), Value::Str(ref s) if &**s == "ab!")
        );
        assert!(
            matches!(one("MATCH (p:P) RETURN 'a' || 'b' || 'c' AS x"), Value::Str(ref s) if &**s == "abc")
        );
        assert!(
            matches!(one("MATCH (p:P) RETURN p.name || '-' || p.age AS x"), Value::Str(ref s) if &**s == "ab-7")
        );
        assert!(matches!(
            one("MATCH (p:P) RETURN p.missing || 'x' AS x"),
            Value::Null
        ));
        assert!(matches!(
            one("MATCH (p:P) RETURN 'x' || p.missing AS x"),
            Value::Null
        ));
        assert!(
            matches!(one("MATCH (p:P) RETURN [1, 2] || [3] AS x"), Value::List(ref v) if v.len() == 3)
        );
        // Precedence: `||` binds looser than `+`, tighter than `=`. `1 + 2 || 3` is
        // `(1+2) || 3` = "33"; used in WHERE it is a concat operand of the comparison.
        assert!(
            matches!(one("MATCH (p:P) RETURN 1 + 2 || 3 AS x"), Value::Str(ref s) if &**s == "33")
        );
        // A lone `|` is not an operator.
        assert!(super::parse("MATCH (p:P) RETURN p.age | 1 AS x").is_err());
    }

    #[test]
    fn low_card_num_distinct_matches_hashing() {
        use crate::store::Builder;
        use std::collections::BTreeSet;
        // Columns exercising: low-card ints (age), a NULL every 5th (age absent),
        // high-card ints past the trivial range (uniq), and non-integers (frac, must
        // fall back to hashing). The bitset path must agree with the hashing path on
        // BOTH count(DISTINCT) and the DISTINCT value SET.
        let mut b = Builder::default();
        for i in 0..2000u32 {
            let mut props = vec![
                ("uniq", Value::Num(f64::from(i))),
                ("frac", Value::Num(f64::from(i % 9) + 0.25)),
            ];
            // age covers every value 0..49; the NULL condition (i%7) is independent of
            // the value so the present ages are still the full {0..49}.
            if i % 7 != 0 {
                props.push(("age", Value::Num(f64::from(i % 50))));
            }
            b.node(&["N"], &props);
        }
        let st = b.build();
        let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x as usize,
            other => panic!("not a count: {other:?}"),
        };
        let set = |q: &str| -> BTreeSet<String> {
            run(&super::parse(q).unwrap(), &st)
                .rows
                .iter()
                .map(|r| format!("{:?}", r[0]))
                .collect()
        };
        // count(DISTINCT k) == the size of the DISTINCT value set, for every column.
        for k in ["age", "uniq", "frac"] {
            let c = count(&format!("MATCH (n:N) RETURN count(DISTINCT n.{k}) AS c"));
            let s = set(&format!("MATCH (n:N) RETURN DISTINCT n.{k} AS v"));
            // The DISTINCT set includes a NULL for `age` (absent every 5th node), which
            // count(DISTINCT) excludes — so the set is one larger exactly there.
            let extra = usize::from(k == "age");
            assert_eq!(c + extra, s.len(), "count vs set mismatch for {k}");
        }
        // Concrete expected values: age = {0..49} plus NULL.
        let age = set("MATCH (n:N) RETURN DISTINCT n.age AS v");
        assert_eq!(age.len(), 51);
        assert!(age.contains("Null"));
        assert_eq!(count("MATCH (n:N) RETURN count(DISTINCT n.age) AS c"), 50);
    }

    #[test]
    fn varlen_degree_formula_matches_enumeration() {
        use crate::store::Builder;
        // 1000 nodes, degree 4 → est_paths (1000·4²=16k) > 2·(V+E) (~10k), so the
        // formula fires for {1,2}/{2,2}; a self-loop on node 0 tests the correction.
        let mut b = Builder::default();
        for _ in 0..1000 {
            b.node(&["N"], &[]);
        }
        for i in 0u32..1000 {
            for d in 0u32..4 {
                b.edge(i, (i * 7 + d * 13 + 1) % 1000, "R");
            }
        }
        b.edge(0, 0, "R"); // self-loop → reused-edge trail exclusion
        let st = b.build();
        let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x as usize,
            other => panic!("not a count: {other:?}"),
        };
        // The count(*) (formula) must equal the enumerated row count (RETURN b).
        for (lo, hi) in [(1u32, 1u32), (1, 2), (2, 2)] {
            let formula = count(&format!(
                "MATCH (a:N)-[:R]->{{{lo},{hi}}}(b) RETURN count(*) AS c"
            ));
            let enumerated = run(
                &super::parse(&format!(
                    "MATCH (a:N)-[:R]->{{{lo},{hi}}}(b) RETURN b.z AS b"
                ))
                .unwrap(),
                &st,
            )
            .rows
            .len();
            assert_eq!(formula, enumerated, "mismatch for {{{lo},{hi}}}");
        }
    }

    #[test]
    fn varlen_distinct_count_bfs_matches_enumeration() {
        use crate::store::Builder;
        use std::collections::HashSet;
        // A graph with cycles and a self-loop, so shortest-distance reachability and
        // the walk-enumerated endpoint SET are non-trivially exercised.
        let mut b = Builder::default();
        for i in 0..300 {
            b.node(&["N"], &[("k", Value::Num(f64::from(i)))]); // unique per-node key
        }
        for i in 0u32..300 {
            for d in 0u32..3 {
                b.edge(i, (i * 7 + d * 11 + 1) % 300, "R");
            }
        }
        b.edge(0, 0, "R"); // self-loop
        let st = b.build();
        let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x as usize,
            other => panic!("not a count: {other:?}"),
        };
        // For every min≤1 bound the BFS fast path fires; compare to the DISTINCT set
        // of endpoints the enumerating path emits. IN and BOTH exercise reverse hops.
        for (lo, hi) in [(0u32, 2u32), (1, 1), (1, 3), (1, 4)] {
            for dir in ["->", "<-", "-"] {
                let (l, r) = match dir {
                    "->" => ("-[:R]", "->"),
                    "<-" => ("<-[:R]", "-"),
                    _ => ("-[:R]", "-"),
                };
                let pat = format!("(a:N){l}{r}{{{lo},{hi}}}(b)");
                let fast = count(&format!("MATCH {pat} RETURN count(DISTINCT b) AS c"));
                let rows = run(
                    &super::parse(&format!("MATCH {pat} RETURN b.k AS b")).unwrap(),
                    &st,
                )
                .rows;
                let enumerated: HashSet<String> =
                    rows.iter().map(|r| format!("{:?}", r[0])).collect();
                assert_eq!(
                    fast,
                    enumerated.len(),
                    "BFS distinct != enumerated distinct for {pat}"
                );
            }
        }
    }

    /// Cardinality-driven anchor flip: a selective indexed `=` on the traversal
    /// TARGET seeds the target and walks reverse edges instead of scanning every
    /// source. The result multiset must equal the forward walk — INCLUDING excluding
    /// a non-source-label node reached in reverse (`bot` is not a `Person`).
    #[test]
    fn anchor_flip_matches_forward_and_respects_source_label() {
        use crate::store::{Builder, Store};
        let mut b = Builder::default();
        // ids 0..3 in insertion order.
        b.node(&["Person"], &[("name", Value::Str("p1".into()))]); // 0
        b.node(&["Person"], &[("name", Value::Str("p2".into()))]); // 1
        b.node(&["Bot"], &[("name", Value::Str("bot".into()))]); // 2 (not Person)
        b.node(&["Person"], &[("name", Value::Str("target".into()))]); // 3
        b.edge(0, 3, "R");
        b.edge(1, 3, "R");
        b.edge(2, 3, "R"); // bot -> target
        let mut st = b.build();
        let q = "MATCH (a:Person)-[:R]->(b) WHERE b.name = 'target' RETURN a.name AS a";
        let names = |st: &Store| {
            let mut v: Vec<String> = run(&super::parse(q).unwrap(), st)
                .rows
                .iter()
                .map(|r| match &r[0] {
                    Value::Str(s) => s.to_string(),
                    o => format!("{o:?}"),
                })
                .collect();
            v.sort();
            v
        };
        // Forward (no index): only the two Person sources reach the target.
        let forward = names(&st);
        assert_eq!(forward, vec!["p1".to_string(), "p2".to_string()]);
        // With an index on `name` the anchor flips (target count 1 < Person count 3);
        // it must give the SAME set — `bot`, reached walking reverse, is excluded.
        st.create_index("name");
        assert_eq!(names(&st), forward);
    }

    /// The raw string-search filter fast path (STARTS WITH / ENDS WITH / CONTAINS)
    /// must match the boxed `str_bool` for a dict-encoded (low-cardinality) column
    /// and for a row missing the property (→ UNKNOWN → dropped).
    #[test]
    fn string_search_fast_path_dict_and_null() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        // `city` is low-cardinality → dict-encoded; one node omits it entirely.
        execute(
            &super::parse(
                "INSERT (:P {city: 'oslo'}), (:P {city: 'bergen'}), (:P {city: 'oslo'}), (:P {n: 1})",
            )
            .unwrap(),
            &mut st,
        )
        .unwrap();
        let count = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x as i64,
            other => panic!("not a count: {other:?}"),
        };
        assert_eq!(
            count("MATCH (p:P) WHERE p.city STARTS WITH 'os' RETURN count(*) AS c"),
            2
        );
        assert_eq!(
            count("MATCH (p:P) WHERE p.city ENDS WITH 'en' RETURN count(*) AS c"),
            1
        );
        assert_eq!(
            count("MATCH (p:P) WHERE p.city CONTAINS 'o' RETURN count(*) AS c"),
            2
        ); // oslo, oslo (bergen has no 'o')
           // The property-less node is UNKNOWN, never matched.
        assert_eq!(
            count("MATCH (p:P) WHERE p.city STARTS WITH '' RETURN count(*) AS c"),
            3
        );
    }

    /// A scalar / grouped aggregate over a traversal streams the source in blocks
    /// into running accumulators (no full endpoint multiset). The result must equal
    /// the materializing path — checked here on a hand-built chain graph.
    #[test]
    fn streaming_aggregate_over_traversal_is_exact() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        // a(1)->b(2)->c(3), a(1)->d(4); scores 2,3,4 reachable at 1..2 hops from a.
        execute(
            &super::parse(
                "INSERT (a:P {g: 0, s: 1})-[:R]->(b:P {g: 1, s: 2})-[:R]->(c:P {g: 1, s: 3}), \
                 (a)-[:R]->(d:P {g: 0, s: 4})",
            )
            .unwrap(),
            &mut st,
        )
        .unwrap();
        let one = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x,
            other => panic!("not a number: {other:?}"),
        };
        // Scalar over a 1..2-hop reach from a: endpoints b(2), d(4) at hop 1; c(3) at
        // hop 2 → {2,4,3}. min=2, max=4, sum=9, count=3.
        let base = "MATCH (a:P {s: 1})-[:R]->{1,2}(x)";
        assert_eq!(one(&format!("{base} RETURN min(x.s) AS v")), 2.0);
        assert_eq!(one(&format!("{base} RETURN max(x.s) AS v")), 4.0);
        assert_eq!(one(&format!("{base} RETURN sum(x.s) AS v")), 9.0);
        assert_eq!(one(&format!("{base} RETURN count(*) AS v")), 3.0);
        // Grouped by a property: g=1 for {b,c}=(2,3), g=0 for {d}=(4). sums 5 and 4.
        let rows = run(
            &super::parse("MATCH (p:P)-[:R]->(q) RETURN q.g AS g, sum(q.s) AS v").unwrap(),
            &st,
        );
        let mut got: Vec<(i64, i64)> = rows
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Num(g), Value::Num(v)) => (*g as i64, *v as i64),
                _ => panic!(),
            })
            .collect();
        got.sort();
        // q.g over edges a->b(1), b->c(1), a->d(0): group 1 sums s of {b,c}=2+3=5;
        // group 0 sums s of {d}=4.
        assert_eq!(got, vec![(0, 4), (1, 5)]);
    }

    /// `DISTINCT … LIMIT k` streams with incremental dedup and stops at `k` distinct
    /// rows — it must equal the first `k` of the full distinct (first-seen order) and
    /// never exceed the total distinct count.
    #[test]
    fn streaming_distinct_limit_equals_full_prefix() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        // Nodes with ages cycling 0..4 so there are exactly 5 distinct target ages.
        let mut q = String::from("INSERT ");
        for i in 0..20 {
            if i > 0 {
                q.push_str(", ");
            }
            q.push_str(&format!("(:P {{age: {}}})", i % 5));
        }
        execute(&super::parse(&q).unwrap(), &mut st).unwrap();
        // Give every P a self-ish edge so a hop exists (chain them a->a+1).
        execute(
            &super::parse("MATCH (a:P), (b:P) WHERE a.age = 0 AND b.age = 1 CREATE (a)-[:R]->(b)")
                .unwrap_or_else(|_| super::parse("MATCH (p:P) RETURN p.age AS a").unwrap()),
            &mut st,
        )
        .ok();
        let rows = |query: &str| {
            let mut v: Vec<String> = run(&super::parse(query).unwrap(), &st)
                .rows
                .iter()
                .map(|r| format!("{r:?}"))
                .collect();
            v.sort();
            v
        };
        // Over a plain scan (streamable): DISTINCT age has 5 values.
        let full = rows("MATCH (p:P) RETURN DISTINCT p.age AS x");
        assert_eq!(full.len(), 5);
        // LIMIT 3 yields exactly 3 distinct rows, all a subset of the full set.
        let lim = rows("MATCH (p:P) RETURN DISTINCT p.age AS x LIMIT 3");
        assert_eq!(lim.len(), 3);
        assert!(lim.iter().all(|r| full.contains(r)));
        // LIMIT beyond the total returns exactly the full distinct set.
        let big = rows("MATCH (p:P) RETURN DISTINCT p.age AS x LIMIT 999");
        assert_eq!(big, full);
    }

    /// A bare `LIMIT`/`SKIP` (no ORDER BY) over a filtered/expanded chain streams the
    /// source in blocks and stops early — and must return EXACTLY the same rows the
    /// full materialize-then-slice would (the block order preserves scan order).
    #[test]
    fn streaming_limit_equals_full_prefix() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        // A chain of nodes 0->1->…; each has age = id, so a filter + expand + limit
        // is non-trivial and the count fast-paths don't apply.
        execute(
            &super::parse(
                "INSERT (a:P {age: 1})-[:R]->(b:P {age: 2})-[:R]->(c:P {age: 3}), \
                 (d:P {age: 4})-[:R]->(e:P {age: 5}), (f:P {age: 6})-[:R]->(g:P {age: 7})",
            )
            .unwrap(),
            &mut st,
        )
        .unwrap();
        let rows = |q: &str| {
            run(&super::parse(q).unwrap(), &st)
                .rows
                .iter()
                .map(|r| format!("{r:?}"))
                .collect::<Vec<_>>()
        };
        let base = "MATCH (a:P)-[:R]->(b) WHERE b.age > 2 RETURN b.age AS x";
        let full = rows(base);
        assert!(full.len() >= 2, "need enough rows to slice");
        // LIMIT streams; it must equal the full result's prefix.
        let lim = rows(&format!("{base} LIMIT 2"));
        assert_eq!(lim, full[..2].to_vec());
        // SKIP + LIMIT streams to skip+limit then slices — equal to the full window.
        let win = rows(&format!("{base} SKIP 1 LIMIT 2"));
        let end = 3.min(full.len());
        assert_eq!(win, full[1..end].to_vec());
        // A LIMIT larger than the result returns everything (no truncation).
        let big = rows(&format!("{base} LIMIT 10000"));
        assert_eq!(big, full);
    }

    /// The vectorized finite→finite unary numeric functions (abs/floor/ceil/round/
    /// sign) must produce exactly what the boxed `scalar_num_fn` does, so an
    /// aggregate over them is correct. Known inputs, hand-computed sums.
    #[test]
    fn vectorized_unary_num_fns_are_exact() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        // `p.v - 5` yields -3 / 2.5 / 0 (a raw Num column via the Arith fast path),
        // which the unary functions then vectorize.
        execute(
            &super::parse("INSERT (:P {v: 2.0}), (:P {v: 7.5}), (:P {v: 5.0})").unwrap(),
            &mut st,
        )
        .unwrap();
        let s = |q: &str| match &run(&super::parse(q).unwrap(), &st).rows[0][0] {
            Value::Num(x) => *x,
            other => panic!("not a number: {other:?}"),
        };
        assert_eq!(s("MATCH (p:P) RETURN sum(abs(p.v - 5)) AS s"), 5.5); // 3 + 2.5 + 0
        assert_eq!(s("MATCH (p:P) RETURN sum(floor(p.v - 5)) AS s"), -1.0); // -3 + 2 + 0
        assert_eq!(s("MATCH (p:P) RETURN sum(ceil(p.v - 5)) AS s"), 0.0); // -3 + 3 + 0
        assert_eq!(s("MATCH (p:P) RETURN sum(sign(p.v - 5)) AS s"), 0.0); // -1 + 1 + 0
        assert_eq!(s("MATCH (p:P) RETURN sum(round(p.v - 5)) AS s"), 0.0); // -3 + 3(2.5→3) + 0
    }

    /// The `count(*)`-over-VarLength fast path must equal the materializing path's
    /// row count for every quantifier / direction — including trail exclusion on the
    /// cycles in `social`. Guards `try_varlen_count`.
    #[test]
    fn varlen_count_matches_materialized() {
        let store = social();
        let count_star = |q: &str| match &run(&super::parse(q).unwrap(), &store).rows[0][0] {
            Value::Num(x) => *x as usize,
            other => panic!("not a count: {other:?}"),
        };
        let materialized = |q: &str| run(&super::parse(q).unwrap(), &store).rows.len();
        for (ct, mt) in [
            (
                "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN count(*) AS c",
                "MATCH (a:Person)-[:KNOWS]->{1,2}(b) RETURN b.name AS b",
            ),
            (
                "MATCH (a:Person)-[:KNOWS]->{1,3}(b) RETURN count(*) AS c",
                "MATCH (a:Person)-[:KNOWS]->{1,3}(b) RETURN b.name AS b",
            ),
            (
                "MATCH (a:Person)-[:KNOWS]->{2}(b) RETURN count(*) AS c",
                "MATCH (a:Person)-[:KNOWS]->{2}(b) RETURN b.name AS b",
            ),
            (
                "MATCH (a:Person)<-[:KNOWS]-{1,2}(b) RETURN count(*) AS c",
                "MATCH (a:Person)<-[:KNOWS]-{1,2}(b) RETURN b.name AS b",
            ),
        ] {
            assert_eq!(count_star(ct), materialized(mt), "mismatch for `{ct}`");
        }
        // Unknown edge type → zero paths, and the fast path must agree.
        assert_eq!(
            count_star("MATCH (a:Person)-[:NOPE]->{1,2}(b) RETURN count(*) AS c"),
            0
        );
    }

    /// `NOT p` over compares / AND / OR is pushed into the raw filter fast paths by
    /// inversion; each form must return the SAME rows as its hand-inverted twin.
    #[test]
    fn not_pushdown_equals_inverted() {
        let store = social();
        let rows = |q: &str| {
            let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| match &r[0] {
                    Value::Str(s) => s.to_string(),
                    other => format!("{other:?}"),
                })
                .collect();
            v.sort();
            v
        };
        // NOT (compare)
        assert_eq!(
            rows("MATCH (p:Person) WHERE NOT p.age > 30 RETURN p.name AS n"),
            rows("MATCH (p:Person) WHERE p.age <= 30 RETURN p.name AS n"),
        );
        // NOT (a AND b) ≡ NOT a OR NOT b
        assert_eq!(
            rows("MATCH (p:Person) WHERE NOT (p.age >= 20 AND p.age < 40) RETURN p.name AS n"),
            rows("MATCH (p:Person) WHERE p.age < 20 OR p.age >= 40 RETURN p.name AS n"),
        );
        // NOT (a OR b) ≡ NOT a AND NOT b
        assert_eq!(
            rows("MATCH (p:Person) WHERE NOT (p.age < 25 OR p.age > 60) RETURN p.name AS n"),
            rows("MATCH (p:Person) WHERE p.age >= 25 AND p.age <= 60 RETURN p.name AS n"),
        );
    }

    /// A shared-start LINEAR comma pattern `…, (b)-[:R]->(c)` (b bound, c new) folds
    /// into a chained expansion — no hash Join — and returns exactly what the join
    /// spelling would. Guards the `join/tri` optimization.
    #[test]
    fn comma_join_linear_folds_to_chain() {
        use crate::ir::{Dir, Expr, Plan};
        let store = social();
        let q = "MATCH (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(c) RETURN c.name AS c";
        let plan = super::parse(q).unwrap();
        // The fold fired: the plan is a chain of Expands, with no Join operator.
        assert!(
            !format!("{plan:?}").contains("Join"),
            "expected the linear comma pattern to fold into a chain, got a Join"
        );
        // …and it equals the same shape written as one chained MATCH.
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, &["KNOWS".to_string()])
        .expand(1, Dir::Out, &["KNOWS".to_string()])
        .project(vec![(
            "c".into(),
            Expr::Prop {
                slot: 2,
                key: "name".into(),
            },
        )]);
        assert_same(q, &hand, &store);
    }

    /// A cycle-CLOSING comma pattern `(a)-[:R]->(b), (b)-[:R]->(a)` must NOT fold
    /// (a chained expand would rebind `a` rather than require the walk return to it);
    /// it falls back to the hash Join, which equates the shared endpoints.
    #[test]
    fn comma_join_cycle_close_keeps_join() {
        let store = social();
        // carol KNOWS alice and alice KNOWS carol? alice->bob->carol, carol->? In
        // `social`, the only mutual KNOWS pair drives the count; we assert the plan
        // shape (Join kept) and that it runs without rebinding to a wrong answer.
        let q = "MATCH (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(a) RETURN a.name AS a";
        let plan = super::parse(q).unwrap();
        assert!(
            format!("{plan:?}").contains("Join"),
            "a cycle-closing comma pattern must keep the hash Join"
        );
        // Every returned `a` must genuinely sit on a 2-cycle a->b->a.
        let out = run(&plan, &store);
        for row in &out.rows {
            let Value::Str(name) = &row[0] else {
                panic!("expected a name")
            };
            let back = run(
                &super::parse(&format!(
                    "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(a2) \
                     WHERE a.name = '{name}' AND a2.name = '{name}' RETURN a.name AS a"
                ))
                .unwrap(),
                &store,
            );
            assert!(!back.rows.is_empty(), "{name} is not on a real 2-cycle");
        }
    }

    // --- part 3.5: arithmetic (E1) ---

    /// Precedence: `2 + 3 * 4` = 14 (multiply binds tighter).
    #[test]
    fn arithmetic_precedence() {
        let store = social();
        let out = run(
            &super::parse("MATCH (p:Person) RETURN 2 + 3 * 4 AS x").unwrap(),
            &store,
        );
        assert_eq!(num(&col(&out, 0, "x")), 14.0);
    }

    /// Parsed `p.age * 2 + 1` matches the hand-built nested Arith plan.
    #[test]
    fn arithmetic_parse_matches_hand() {
        use crate::ir::{ArithOp, Expr, Plan};
        let store = social();
        let mul = Expr::Arith {
            op: ArithOp::Mul,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "age".into(),
            }),
            right: Box::new(Expr::Lit(n(2.0))),
        };
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "x".into(),
            Expr::Arith {
                op: ArithOp::Add,
                left: Box::new(mul),
                right: Box::new(Expr::Lit(n(1.0))),
            },
        )]);
        assert_same("MATCH (p:Person) RETURN p.age * 2 + 1 AS x", &hand, &store);
    }

    /// Arithmetic in WHERE: `p.age % 2 = 0` keeps even ages (alice 30, carol 40).
    #[test]
    fn arithmetic_in_where() {
        let store = social();
        let out = run(
            &super::parse("MATCH (p:Person) WHERE p.age % 2 = 0 RETURN p.name AS name").unwrap(),
            &store,
        );
        let mut got: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                _ => panic!(),
            })
            .collect();
        got.sort();
        assert_eq!(got, vec!["alice", "carol"]);
    }

    /// Unary minus: `-p.age` for alice(30) is -30.
    #[test]
    fn unary_minus() {
        let store = social();
        let out = run(
            &super::parse("MATCH (p:Person) WHERE p.name = 'alice' RETURN -p.age AS x").unwrap(),
            &store,
        );
        assert_eq!(num(&col(&out, 0, "x")), -30.0);
    }

    // --- part 3.6: scalar functions (E2) ---

    /// Numeric functions compute; hand-checked on alice(age 30).
    #[test]
    fn scalar_numeric_functions() {
        let store = social();
        let q = "MATCH (p:Person) WHERE p.name = 'alice' \
                 RETURN abs(-p.age) AS a, floor(p.age / 4) AS f, ceil(p.age / 4) AS c, \
                        round(p.age / 4) AS r, sqrt(p.age - 5) AS s, sign(p.age - 100) AS g";
        let out = run(&super::parse(q).unwrap(), &store);
        assert_eq!(num(&col(&out, 0, "a")), 30.0); // abs(-30)
        assert_eq!(num(&col(&out, 0, "f")), 7.0); // floor(7.5)
        assert_eq!(num(&col(&out, 0, "c")), 8.0); // ceil(7.5)
        assert_eq!(num(&col(&out, 0, "r")), 8.0); // round(7.5) -> 8
        assert_eq!(num(&col(&out, 0, "s")), 5.0); // sqrt(25)
        assert_eq!(num(&col(&out, 0, "g")), -1.0); // sign(30-100)
    }

    /// A numeric fn on a NULL/non-numeric/negative-sqrt argument yields NULL.
    #[test]
    fn scalar_fn_null_and_domain() {
        let store = social();
        // proj node has no age → abs(age) is NULL for it.
        let out = run(
            &super::parse("MATCH (n) RETURN abs(n.age) AS a").unwrap(),
            &store,
        );
        assert_eq!(out.rows.iter().filter(|r| r[0].is_null()).count(), 1);
        // sqrt of a negative KEEPS NaN (a real signal), matching lenke-core — it is
        // coerced to null only at JSON egress, not in the result value (K4).
        let out = run(
            &super::parse("MATCH (p:Person) WHERE p.name='alice' RETURN sqrt(0 - p.age) AS s")
                .unwrap(),
            &store,
        );
        assert!(matches!(col(&out, 0, "s"), crate::value::Value::Num(x) if x.is_nan()));
    }

    /// `CONTAINS` / `STARTS WITH` / `ENDS WITH` infix predicates desugar to the
    /// scalar functions and filter three-valued (a NULL operand drops the row).
    #[test]
    fn string_infix_predicates() {
        let mut b = crate::store::Builder::default();
        b.node(&["N"], &[("s", crate::value::Value::Str("carol".into()))]);
        b.node(&["N"], &[("s", crate::value::Value::Str("bob".into()))]);
        b.node(&["N"], &[]); // s absent → NULL
        let store = b.build();
        let names = |q: &str| -> Vec<String> {
            let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| match &r[0] {
                    crate::value::Value::Str(x) => x.to_string(),
                    o => format!("{o:?}"),
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            names("MATCH (n:N) WHERE n.s CONTAINS 'o' RETURN n.s AS s"),
            vec!["bob", "carol"]
        );
        assert_eq!(
            names("MATCH (n:N) WHERE n.s STARTS WITH 'ca' RETURN n.s AS s"),
            vec!["carol"]
        );
        assert_eq!(
            names("MATCH (n:N) WHERE n.s ENDS WITH 'ob' RETURN n.s AS s"),
            vec!["bob"]
        );
        // NULL operand → UNKNOWN → dropped (the s-absent node never matches).
        assert_eq!(
            names("MATCH (n:N) WHERE n.s CONTAINS 'zzz' RETURN n.s AS s"),
            Vec::<String>::new()
        );
    }

    /// `coalesce` returns the first non-null argument.
    #[test]
    fn coalesce_first_non_null() {
        let store = social();
        // proj has name but no age: coalesce(age, 99) = 99 for proj, real age else.
        let out = run(
            &super::parse("MATCH (n) WHERE n.name = 'graphdb' RETURN coalesce(n.age, 99) AS x")
                .unwrap(),
            &store,
        );
        assert_eq!(num(&col(&out, 0, "x")), 99.0);
        let out = run(
            &super::parse("MATCH (p:Person) WHERE p.name='alice' RETURN coalesce(p.age, 99) AS x")
                .unwrap(),
            &store,
        );
        assert_eq!(num(&col(&out, 0, "x")), 30.0);
    }

    /// Parsed `abs(p.age)` matches the hand-built Call plan.
    #[test]
    fn scalar_fn_parse_matches_hand() {
        use crate::ir::{Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "a".into(),
            Expr::Call {
                name: "abs".into(),
                args: vec![Expr::Prop {
                    slot: 0,
                    key: "age".into(),
                }],
            },
        )]);
        assert_same("MATCH (p:Person) RETURN abs(p.age) AS a", &hand, &store);
    }

    #[test]
    fn scalar_fn_errors() {
        assert!(super::parse("MATCH (p:Person) RETURN nope(p.age) AS x").is_err()); // unknown fn
        assert!(super::parse("MATCH (p:Person) RETURN abs(p.age, 1) AS x").is_err()); // arity
        assert!(super::parse("MATCH (p:Person) RETURN coalesce() AS x").is_err());
        // arity
    }

    // --- part 3.7: CASE (E3) ---

    /// Searched CASE picks the first true branch; ELSE otherwise. Ages 30/25/40:
    /// >=40 → "old", >=30 → "mid", else "young".
    #[test]
    fn case_branch_selection() {
        let store = social();
        let q = "MATCH (p:Person) RETURN p.name AS name, \
                 CASE WHEN p.age >= 40 THEN 'old' WHEN p.age >= 30 THEN 'mid' ELSE 'young' END AS band";
        let out = run(&super::parse(q).unwrap(), &store);
        let mut got: Vec<(String, String)> = out
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(a), Value::Str(b)) => (a.to_string(), b.to_string()),
                _ => panic!("shape"),
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("alice".into(), "mid".into()), // 30
                ("bob".into(), "young".into()), // 25
                ("carol".into(), "old".into()), // 40
            ]
        );
    }

    /// No ELSE and no matching branch → NULL; a NULL/false condition is skipped
    /// (proj has no age, so `p.age >= 30` is UNKNOWN → skipped → NULL, no ELSE).
    #[test]
    fn case_no_else_and_null_condition() {
        let store = social();
        let out = run(
            &super::parse("MATCH (n) RETURN CASE WHEN n.age >= 30 THEN 'y' END AS x").unwrap(),
            &store,
        );
        // alice(30),carol(40) → 'y'; bob(25) → NULL; proj(no age) → NULL.
        let ys = out
            .rows
            .iter()
            .filter(|r| matches!(&r[0], Value::Str(s) if &**s == "y"))
            .count();
        let nulls = out.rows.iter().filter(|r| r[0].is_null()).count();
        assert_eq!(ys, 2);
        assert_eq!(nulls, 2);
    }

    /// Parsed CASE matches the hand-built `Expr::Case`.
    #[test]
    fn case_parse_matches_hand() {
        use crate::ir::{CompareOp, Expr, Plan};
        let store = social();
        let cond = Expr::Compare {
            op: CompareOp::Ge,
            left: Box::new(Expr::Prop {
                slot: 0,
                key: "age".into(),
            }),
            right: Box::new(Expr::Lit(n(30.0))),
        };
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "x".into(),
            Expr::Case {
                branches: vec![(cond, Expr::Lit(s("sr")))],
                otherwise: Some(Box::new(Expr::Lit(s("jr")))),
            },
        )]);
        assert_same(
            "MATCH (p:Person) RETURN CASE WHEN p.age >= 30 THEN 'sr' ELSE 'jr' END AS x",
            &hand,
            &store,
        );
    }

    #[test]
    fn case_errors() {
        // The simple form is now supported (desugars to searched CASE).
        assert!(
            super::parse("MATCH (p:Person) RETURN CASE p.age WHEN 30 THEN 'x' END AS y").is_ok()
        );
        assert!(
            super::parse("MATCH (p:Person) RETURN CASE WHEN p.age >= 30 THEN 'x' AS y").is_err()
        ); // no END
    }

    /// The simple CASE form `CASE <subject> WHEN <v> THEN …` desugars to searched
    /// CASE (`WHEN subject = v`); a NULL subject matches no branch (3VL) → ELSE.
    #[test]
    fn simple_case_form() {
        let store = social();
        let val = |q: &str| -> String {
            format!("{:?}", run(&super::parse(q).unwrap(), &store).rows[0][0])
        };
        assert_eq!(
            val("RETURN CASE 5 WHEN 1 THEN 'a' WHEN 5 THEN 'b' ELSE 'c' END AS r"),
            "Str(\"b\")"
        );
        assert_eq!(
            val("RETURN CASE 42 WHEN 1 THEN 'a' ELSE 'c' END AS r"),
            "Str(\"c\")"
        );
        // A NULL subject never equals a WHEN value → falls to ELSE.
        assert_eq!(
            val("MATCH (p:Person) WHERE p.name = 'alice' RETURN CASE p.nope WHEN 1 THEN 'a' ELSE 'none' END AS r"),
            "Str(\"none\")"
        );
    }

    /// `ORDER BY … NULLS FIRST|LAST` overrides the default null placement (last).
    #[test]
    fn order_by_nulls_first_last() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"age\":30}}\n",
            "{\"id\":\"b\",\"labels\":[\"P\"],\"props\":{\"age\":null}}\n",
            "{\"id\":\"c\",\"labels\":[\"P\"],\"props\":{\"age\":40}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let col0 = |q: &str| -> Vec<String> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| format!("{:?}", r[0]))
                .collect()
        };
        // ASC NULLS FIRST puts the null ahead of 30, 40.
        assert_eq!(
            col0("MATCH (n:P) RETURN n.age AS age ORDER BY n.age ASC NULLS FIRST"),
            vec!["Null", "Num(30.0)", "Num(40.0)"]
        );
        // DESC NULLS LAST keeps the null after 40, 30.
        assert_eq!(
            col0("MATCH (n:P) RETURN n.age AS age ORDER BY n.age DESC NULLS LAST"),
            vec!["Num(40.0)", "Num(30.0)", "Null"]
        );
    }

    /// String-literal backslash escapes decode to their characters; `\\uXXXX` /
    /// `\\UXXXXXX` are code points; a malformed unicode escape is a syntax error.
    #[test]
    fn string_escapes() {
        let store = social();
        let val = |q: &str| -> String {
            match run(&super::parse(q).unwrap(), &store).rows[0][0] {
                Value::Str(ref s) => s.to_string(),
                ref o => panic!("want str, got {o:?}"),
            }
        };
        assert_eq!(val(r"RETURN '\n' AS r"), "\n");
        assert_eq!(val(r"RETURN '\t' AS r"), "\t");
        assert_eq!(val(r"RETURN '\\' AS r"), "\\");
        assert_eq!(val(r"RETURN '\'' AS r"), "'");
        assert_eq!(val(r"RETURN '\u0041' AS r"), "A");
        assert_eq!(val(r"RETURN '\U01F600' AS r"), "\u{1F600}");
        // A malformed \u escape is rejected (agreeing with core).
        assert!(super::parse(r"RETURN '\uH' AS x").is_err());
    }

    /// `x IS [NOT] LABELED L` tests the element's label set (the keyword form of
    /// the `x:L` predicate).
    #[test]
    fn is_labeled_predicate() {
        let store = social();
        let n = |q: &str| -> f64 {
            match run(&super::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("want num, got {o:?}"),
            }
        };
        let total = n("MATCH (x) RETURN count(*) AS c");
        let persons = n("MATCH (x) WHERE x IS LABELED Person RETURN count(*) AS c");
        assert!(persons > 0.0 && persons <= total);
        // IS NOT LABELED is the complement.
        assert_eq!(
            n("MATCH (x) WHERE x IS NOT LABELED Person RETURN count(*) AS c"),
            total - persons
        );
        // Agrees with the `x:Label` predicate form.
        assert_eq!(persons, n("MATCH (x) WHERE x:Person RETURN count(*) AS c"));
    }

    /// Node label algebra: conjunction `:A&B`, disjunction `:A|B`, negation `:!A`,
    /// wildcard `:%`, and the `IS L` introducer, over multi-label nodes.
    #[test]
    fn node_label_algebra() {
        let nd = concat!(
            "{\"id\":\"pa\",\"labels\":[\"Person\",\"Admin\"],\"props\":{}}\n",
            "{\"id\":\"p\",\"labels\":[\"Person\"],\"props\":{}}\n",
            "{\"id\":\"s\",\"labels\":[\"Software\"],\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let n = |q: &str| -> f64 {
            match run(&super::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("want num, got {o:?}"),
            }
        };
        assert_eq!(n("MATCH (x:Person&Admin) RETURN count(*) AS c"), 1.0); // only pa
        assert_eq!(n("MATCH (x:Person|Software) RETURN count(*) AS c"), 3.0); // pa, p, s
        assert_eq!(n("MATCH (x:!Software) RETURN count(*) AS c"), 2.0); // pa, p
        assert_eq!(n("MATCH (x:%) RETURN count(*) AS c"), 3.0); // any label
        assert_eq!(n("MATCH (x IS Person) RETURN count(*) AS c"), 2.0); // pa, p (= :Person)
    }

    /// A landing (non-seed) node's label constrains the hop, as core does — in a
    /// plain MATCH and inside a COUNT{}/EXISTS{} subquery body.
    #[test]
    fn landing_node_label() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"name\":\"a\"}}\n",
            "{\"id\":\"t\",\"labels\":[\"N\",\"Target\"],\"props\":{}}\n",
            "{\"id\":\"x\",\"labels\":[\"N\"],\"props\":{}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"t\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"a\",\"to\":\"x\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let n = |q: &str| -> f64 {
            match run(&super::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("want num, got {o:?}"),
            }
        };
        // Plain MATCH: a -> Target lands only on `t` (1), not `x`.
        assert_eq!(n("MATCH (a)-[:R]->(b:Target) RETURN count(*) AS c"), 1.0);
        // Without the label, both neighbours count.
        assert_eq!(n("MATCH (a)-[:R]->(b) RETURN count(*) AS c"), 2.0);
        // The same constraint inside a COUNT{} subquery body.
        assert_eq!(
            n("MATCH (a {name:'a'}) RETURN COUNT { (a)-[:R]->(:Target) } AS c"),
            1.0
        );
    }

    /// The cross-type total order (Num < Str < Bool < Temporal < compound < Null,
    /// matching core) drives ORDER BY / min / max over a mixed-type column.
    #[test]
    fn mixed_type_total_order() {
        let nd = concat!(
            "{\"id\":\"1\",\"labels\":[\"X\"],\"props\":{\"v\":2}}\n",
            "{\"id\":\"2\",\"labels\":[\"X\"],\"props\":{\"v\":\"a\"}}\n",
            "{\"id\":\"3\",\"labels\":[\"X\"],\"props\":{\"v\":1}}\n",
            "{\"id\":\"4\",\"labels\":[\"X\"],\"props\":{\"v\":true}}\n",
            "{\"id\":\"5\",\"labels\":[\"X\"],\"props\":{\"v\":\"b\"}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let col0 = |q: &str| -> Vec<String> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| format!("{:?}", r[0]))
                .collect()
        };
        // ORDER BY asc: numbers, then strings, then bool.
        assert_eq!(
            col0("MATCH (n:X) RETURN n.v AS v ORDER BY n.v"),
            vec![
                "Num(1.0)",
                "Num(2.0)",
                "Str(\"a\")",
                "Str(\"b\")",
                "Bool(true)"
            ]
        );
        // min = the smallest number; max = the bool (highest rank present).
        assert_eq!(col0("MATCH (n:X) RETURN min(n.v) AS m"), vec!["Num(1.0)"]);
        assert_eq!(col0("MATCH (n:X) RETURN max(n.v) AS m"), vec!["Bool(true)"]);
    }

    /// `ORDER BY` over a group-key EXPRESSION works under implicit grouping — the
    /// key column is ordered even though the bindings are gone post-aggregation.
    #[test]
    fn grouped_order_by_group_key() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"P\"],\"props\":{\"city\":\"z\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"P\"],\"props\":{\"city\":\"a\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"P\"],\"props\":{\"city\":\"a\"}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        // Group by city, count, ORDER BY the group-key expression n.city.
        let rows: Vec<String> = run(
            &super::parse("MATCH (n:P) RETURN n.city, count(*) AS c ORDER BY n.city").unwrap(),
            &store,
        )
        .rows
        .iter()
        .map(|r| format!("{:?},{:?}", r[0], r[1]))
        .collect();
        assert_eq!(rows, vec!["Str(\"a\"),Num(2.0)", "Str(\"z\"),Num(1.0)"]);
    }

    /// `TIMESTAMP '…'` is core's alias for a (local) DATETIME literal.
    #[test]
    fn timestamp_is_datetime_alias() {
        let store = social();
        let val = |q: &str| format!("{:?}", run(&super::parse(q).unwrap(), &store).rows[0][0]);
        // TIMESTAMP parses and compares equal to the same DATETIME literal.
        assert_eq!(
            val("RETURN TIMESTAMP '2021-06-15T08:30:00' = DATETIME '2021-06-15T08:30:00' AS x"),
            "Bool(true)"
        );
        assert_eq!(
            val("RETURN TIMESTAMP '2021-06-15T08:30:00.5' >= DATETIME '2021-06-15T08:30:00' AS x"),
            "Bool(true)"
        );
    }

    /// An aggregate nested in a projection expression (`count(*) + 1`) hoists the
    /// aggregate into the group and projects the surrounding arithmetic over it.
    #[test]
    fn aggregate_in_projection_expression() {
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"T\"],\"props\":{}}\n",
            "{\"id\":\"b\",\"labels\":[\"T\"],\"props\":{}}\n",
            "{\"id\":\"c\",\"labels\":[\"T\"],\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let n = |q: &str| -> f64 {
            match run(&super::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("want num, got {o:?}"),
            }
        };
        assert_eq!(n("MATCH (t:T) RETURN count(*) + 1 AS c"), 4.0);
        assert_eq!(n("MATCH (t:T) RETURN count(*) * 2 - 1 AS c"), 5.0);
        // A bare aggregate is unaffected.
        assert_eq!(n("MATCH (t:T) RETURN count(*) AS c"), 3.0);
    }

    /// A label EXPRESSION in a WHERE predicate (`x:A|B`, `x:A&B`, `x:!A`) lowers via
    /// the shared label-expression lowering, like the pattern-position label algebra.
    #[test]
    fn label_expr_in_predicate() {
        let nd = concat!(
            "{\"id\":\"pa\",\"labels\":[\"Person\",\"Admin\"],\"props\":{}}\n",
            "{\"id\":\"p\",\"labels\":[\"Person\"],\"props\":{}}\n",
            "{\"id\":\"s\",\"labels\":[\"Software\"],\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let n = |q: &str| -> f64 {
            match run(&super::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("want num, got {o:?}"),
            }
        };
        assert_eq!(
            n("MATCH (x) WHERE x:Person|Software RETURN count(*) AS c"),
            3.0
        );
        assert_eq!(
            n("MATCH (x) WHERE x:Person&Admin RETURN count(*) AS c"),
            1.0
        );
        assert_eq!(n("MATCH (x) WHERE x:!Software RETURN count(*) AS c"), 2.0);
        // A single label is unchanged.
        assert_eq!(n("MATCH (x) WHERE x:Person RETURN count(*) AS c"), 2.0);
    }

    /// A reverse-correlated COUNT/EXISTS subquery — the outer variable is the hop's
    /// LANDING (`COUNT { (m)-[:R]->(n) }`), so the body traverses from the bound
    /// endpoint backward. In-degree, incoming direction, and a local-node label all
    /// resolve correctly.
    #[test]
    fn reverse_correlated_subquery() {
        let nd = concat!(
            "{\"id\":\"n0\",\"labels\":[\"Node\"],\"props\":{\"name\":\"n0\"}}\n",
            "{\"id\":\"n1\",\"labels\":[\"Node\"],\"props\":{\"name\":\"n1\"}}\n",
            "{\"id\":\"n2\",\"labels\":[\"Node\"],\"props\":{\"name\":\"n2\"}}\n",
            "{\"id\":\"e1\",\"from\":\"n0\",\"to\":\"n1\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"n0\",\"to\":\"n1\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e3\",\"from\":\"n0\",\"to\":\"n2\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let n = |q: &str| -> f64 {
            match run(&super::parse(q).unwrap(), &store).rows[0][0] {
                Value::Num(x) => x,
                ref o => panic!("want num, got {o:?}"),
            }
        };
        // in-degree of n1 = 2.
        assert_eq!(
            n("MATCH (n:Node) WHERE n.name='n1' RETURN COUNT { (m)-[:R]->(n) } AS c"),
            2.0
        );
        // out-degree of n0 via incoming arrow at the local node = 3.
        assert_eq!(
            n("MATCH (n:Node) WHERE n.name='n0' RETURN COUNT { (m)<-[:R]-(n) } AS c"),
            3.0
        );
        // local-node label filter narrows the reverse hop.
        assert_eq!(
            n("MATCH (n:Node) WHERE n.name='n1' RETURN COUNT { (m:Node)-[:R]->(n) } AS c"),
            2.0
        );
    }

    /// A single-edge parenthesized subpath group `((x)-[e:R]->(y)){n,m}(t)` lowers
    /// to a variable-length hop to the endpoint (endpoint-only; group vars ignored).
    #[test]
    fn subpath_group_single_edge() {
        // chain a -> b -> c -> d
        let nd = concat!(
            "{\"id\":\"a\",\"labels\":[\"N\"],\"props\":{\"id\":\"a\"}}\n",
            "{\"id\":\"b\",\"labels\":[\"N\"],\"props\":{\"id\":\"b\"}}\n",
            "{\"id\":\"c\",\"labels\":[\"N\"],\"props\":{\"id\":\"c\"}}\n",
            "{\"id\":\"d\",\"labels\":[\"N\"],\"props\":{\"id\":\"d\"}}\n",
            "{\"id\":\"e1\",\"from\":\"a\",\"to\":\"b\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e2\",\"from\":\"b\",\"to\":\"c\",\"type\":\"R\",\"props\":{}}\n",
            "{\"id\":\"e3\",\"from\":\"c\",\"to\":\"d\",\"type\":\"R\",\"props\":{}}\n",
        );
        let store = crate::ndjson::from_ndjson(nd).unwrap();
        let ids = |q: &str| -> Vec<String> {
            let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| format!("{:?}", r[0]))
                .collect();
            v.sort();
            v
        };
        // 1..2 reps from a: b (1), c (2).
        assert_eq!(
            ids("MATCH (s:N {id:'a'}) ((x)-[e:R]->(y)){1,2} (t) RETURN t.id AS id"),
            vec!["Str(\"b\")", "Str(\"c\")"]
        );
        // Anonymous inner nodes + exact {2}: only c.
        assert_eq!(
            ids("MATCH (s:N {id:'a'}) (()-[:R]->()){2} (t) RETURN t.id AS id"),
            vec!["Str(\"c\")"]
        );
        // Unanchored group: every 1-rep landing = b, c, d.
        assert_eq!(
            ids("MATCH ((x)-[:R]->(y)){1,1} (t) RETURN t.id AS id"),
            vec!["Str(\"b\")", "Str(\"c\")", "Str(\"d\")"]
        );
    }

    // --- part 3.8: string functions (E4a) ---

    /// upper/lower/trim/length/substring/replace on alice's name — hand-computed.
    #[test]
    fn string_functions() {
        let store = social();
        let q = "MATCH (p:Person) WHERE p.name = 'alice' RETURN \
                 upper(p.name) AS u, length(p.name) AS l, substring(p.name, 1, 3) AS sub, \
                 replace(p.name, 'a', 'A') AS rep";
        let out = run(&super::parse(q).unwrap(), &store);
        assert!(matches!(col(&out, 0, "u"), Value::Str(x) if &*x == "ALICE"));
        assert_eq!(num(&col(&out, 0, "l")), 5.0); // "alice"
        assert!(matches!(col(&out, 0, "sub"), Value::Str(x) if &*x == "ali")); // ISO 1-based [1,4)
        assert!(matches!(col(&out, 0, "rep"), Value::Str(x) if &*x == "Alice"));
    }

    /// String predicates return Bool; a non-string / null argument yields NULL.
    #[test]
    fn string_predicates_and_null() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) WHERE p.name='alice' \
                 RETURN starts_with(p.name,'al') AS s, contains(p.name,'zz') AS c, upper(p.age) AS bad",
            )
            .unwrap(),
            &store,
        );
        assert!(matches!(col(&out, 0, "s"), Value::Bool(true)));
        assert!(matches!(col(&out, 0, "c"), Value::Bool(false)));
        assert!(col(&out, 0, "bad").is_null()); // upper of a number → NULL
    }

    /// substring past the end clamps; a negative index is NULL.
    #[test]
    fn substring_edges() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) WHERE p.name='alice' \
                 RETURN substring(p.name, 3) AS tail, substring(p.name, 10) AS past",
            )
            .unwrap(),
            &store,
        );
        assert!(matches!(col(&out, 0, "tail"), Value::Str(x) if &*x == "ice")); // ISO 1-based, from unit 2
        assert!(matches!(col(&out, 0, "past"), Value::Str(x) if x.is_empty())); // clamped
                                                                                // A start <= 0 shrinks the window from the front (SQL semantics), so it
                                                                                // returns the whole string — matching core, NOT NULL.
        let neg = run(
            &super::parse(
                "MATCH (p:Person) WHERE p.name='alice' RETURN substring(p.name, -1) AS x",
            )
            .unwrap(),
            &store,
        );
        assert!(matches!(col(&neg, 0, "x"), Value::Str(s) if &*s == "alice"));
    }

    /// OPTIONAL MATCH is a left-outer hop: a node with no matching neighbour survives
    /// with the optional variable NULL; `count(x)` skips those nulls.
    #[test]
    fn optional_match_left_outer() {
        let store = social(); // alice-KNOWS->bob, alice-KNOWS->carol, bob-KNOWS->carol
        let q =
            "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name AS an, b.name AS bn";
        let out = run(&super::parse(q).unwrap(), &store);
        // carol has no outgoing KNOWS → one row (carol, null).
        let mut pairs: Vec<(String, Option<String>)> = out
            .rows
            .iter()
            .map(|r| {
                let a = match &r[0] {
                    Value::Str(s) => s.to_string(),
                    o => format!("{o:?}"),
                };
                let b = match &r[1] {
                    Value::Str(s) => Some(s.to_string()),
                    Value::Null => None,
                    o => Some(format!("{o:?}")),
                };
                (a, b)
            })
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("alice".into(), Some("bob".into())),
                ("alice".into(), Some("carol".into())),
                ("bob".into(), Some("carol".into())),
                ("carol".into(), None), // left-outer: kept with NULL
            ]
        );
        // count(x) over the optional skips the null (carol → 0).
        let counts = run(
            &super::parse(
                "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b) RETURN a.name AS an, count(b) AS c",
            )
            .unwrap(),
            &store,
        );
        let carol = counts
            .rows
            .iter()
            .find(|r| matches!(&r[0], Value::Str(s) if &**s == "carol"))
            .unwrap();
        assert!(matches!(carol[1], Value::Num(x) if x == 0.0));
    }

    /// `UNION` concatenates two query arms' rows and dedups; `UNION ALL` keeps dups;
    /// the result's column names come from the LEFT arm.
    #[test]
    fn union_and_union_all() {
        let mut b = Builder::default();
        b.node(&["P"], &[("v", s("a"))]);
        b.node(&["P"], &[("v", s("a"))]); // duplicate value
        b.node(&["Q"], &[("v", s("b"))]);
        let store = b.build();
        let vals = |q: &str| -> Vec<String> {
            let mut v: Vec<String> = run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| match &r[0] {
                    Value::Str(s) => s.to_string(),
                    o => format!("{o:?}"),
                })
                .collect();
            v.sort();
            v
        };
        // UNION dedups: {a, a} ∪ {b} → [a, b].
        assert_eq!(
            vals("MATCH (p:P) RETURN p.v AS x UNION MATCH (q:Q) RETURN q.v AS x"),
            vec!["a", "b"]
        );
        // UNION ALL keeps every row: a, a, b.
        assert_eq!(
            vals("MATCH (p:P) RETURN p.v AS x UNION ALL MATCH (q:Q) RETURN q.v AS x"),
            vec!["a", "a", "b"]
        );
        // Column names come from the LEFT arm even if the right differs.
        assert_eq!(
            run(
                &super::parse("MATCH (p:P) RETURN p.v AS x UNION MATCH (q:Q) RETURN q.v AS y")
                    .unwrap(),
                &store
            )
            .names,
            vec!["x".to_string()]
        );
    }

    /// `collect_list(x)` gathers a group's values into a list in row order, SKIPPING
    /// nulls (core's semantics — distinct from Gremlin fold, which keeps them).
    #[test]
    fn collect_list_aggregate_skips_nulls() {
        let mut b = Builder::default();
        // dept eng: ages 1, (null), 3 ; dept ops: age 5
        b.node(&["P"], &[("d", s("eng")), ("age", n(1.0))]);
        b.node(&["P"], &[("d", s("eng"))]); // no age → null, dropped by collect_list
        b.node(&["P"], &[("d", s("eng")), ("age", n(3.0))]);
        b.node(&["P"], &[("d", s("ops")), ("age", n(5.0))]);
        let store = b.build();
        let out = run(
            &super::parse("MATCH (p:P) RETURN p.d AS d, collect_list(p.age) AS ages ORDER BY d")
                .unwrap(),
            &store,
        );
        // Groups ordered by d: eng then ops. eng's list is [1, 3] (null skipped).
        let list = |r: usize| match &out.rows[r][1] {
            Value::List(v) => v
                .iter()
                .map(|x| match x {
                    Value::Num(n) => *n,
                    _ => f64::NAN,
                })
                .collect::<Vec<_>>(),
            _ => panic!("expected a list"),
        };
        assert_eq!(list(0), vec![1.0, 3.0]);
        assert_eq!(list(1), vec![5.0]);
        // `collect` is a superset alias for the same thing.
        assert!(super::parse("MATCH (p:P) RETURN collect(p.age) AS a").is_ok());
    }

    /// ORDER BY can sort by an UNPROJECTED expression (`ORDER BY n.age` when only
    /// `n.name` is returned) — projected as a hidden column, sorted, then dropped.
    #[test]
    fn order_by_unprojected_expression() {
        let mut b = Builder::default();
        for (nm, age) in [("c", 3.0), ("a", 1.0), ("b", 2.0)] {
            b.node(&["P"], &[("name", s(nm)), ("age", n(age))]);
        }
        let store = b.build();
        let names = |q: &str| -> Vec<String> {
            run(&super::parse(q).unwrap(), &store)
                .rows
                .iter()
                .map(|r| match &r[0] {
                    Value::Str(s) => s.to_string(),
                    o => format!("{o:?}"),
                })
                .collect()
        };
        // Sort by the unprojected age; only name is returned.
        assert_eq!(
            names("MATCH (p:P) RETURN p.name AS nm ORDER BY p.age"),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            names("MATCH (p:P) RETURN p.name AS nm ORDER BY p.age DESC"),
            vec!["c", "b", "a"]
        );
        // The output is exactly the returned column (hidden sort column dropped).
        assert_eq!(
            run(
                &super::parse("MATCH (p:P) RETURN p.name AS nm ORDER BY p.age").unwrap(),
                &store
            )
            .names,
            vec!["nm".to_string()]
        );
    }

    /// `RETURN r` (a bound edge) renders core's edge element map —
    /// `{id, from, to, labels, properties}`.
    #[test]
    fn return_edge_renders_element_map() {
        let store = social();
        let out = run(
            &super::parse("MATCH (a:Person {name:'alice'})-[r:KNOWS]->(b) RETURN r").unwrap(),
            &store,
        );
        let Value::Map(m) = &out.rows[0][0] else {
            panic!("expected an edge map, got {:?}", out.rows[0][0]);
        };
        let keys: Vec<&str> = m
            .iter()
            .map(|(k, _)| match k {
                Value::Str(s) => s.as_ref(),
                _ => "?",
            })
            .collect();
        assert_eq!(keys, vec!["id", "from", "to", "labels", "properties"]);
        // labels is a list carrying the edge type.
        assert!(
            matches!(&m[3].1, Value::List(l) if matches!(&l[0], Value::Str(s) if &**s == "KNOWS"))
        );
    }

    /// `RETURN *` projects every bound variable, in slot (declaration) order, each
    /// column named for its variable.
    #[test]
    fn return_star_expands_bound_vars() {
        let store = social();
        // Two bound node vars → two columns, `a` then `b`, both node maps.
        let out = run(
            &super::parse("MATCH (a:Person {name:'alice'})-[:KNOWS]->(b) RETURN *").unwrap(),
            &store,
        );
        assert_eq!(out.names, vec!["a".to_string(), "b".to_string()]);
        assert!(out
            .rows
            .iter()
            .all(|r| matches!(&r[0], Value::Map(_)) && matches!(&r[1], Value::Map(_))));
        // `*` composes with an explicit item.
        let out2 = run(
            &super::parse("MATCH (n:Person) RETURN *, n.name AS nm").unwrap(),
            &store,
        );
        assert_eq!(out2.names, vec!["n".to_string(), "nm".to_string()]);
    }

    /// `RETURN n` (a bare node binding) renders the element MAP core produces —
    /// `{id, labels(sorted), properties(sorted)}` — not the bare node id.
    #[test]
    fn return_node_renders_element_map() {
        let store = social();
        let out = run(
            &super::parse("MATCH (p:Person {name:'alice'}) RETURN p").unwrap(),
            &store,
        );
        let Value::Map(m) = &out.rows[0][0] else {
            panic!("expected a node map, got {:?}", out.rows[0][0]);
        };
        // Top-level keys, in order.
        let keys: Vec<&str> = m
            .iter()
            .map(|(k, _)| match k {
                Value::Str(s) => s.as_ref(),
                _ => "?",
            })
            .collect();
        assert_eq!(keys, vec!["id", "labels", "properties"]);
        // labels is a List; properties is a Map carrying name='alice'.
        assert!(
            matches!(&m[1].1, Value::List(l) if matches!(&l[0], Value::Str(s) if &**s == "Person"))
        );
        let Value::Map(props) = &m[2].1 else {
            panic!("properties must be a map")
        };
        assert!(props
            .iter()
            .any(|(k, v)| matches!((k, v), (Value::Str(k), Value::Str(v)) if &**k == "name" && &**v == "alice")));
    }

    /// An untyped relationship `-[r]->` / `-[]->` traverses edges of ANY type;
    /// `alice` has one KNOWS and one WORKS_ON out-edge, so untyped sees both while a
    /// `:KNOWS` hop sees only one.
    #[test]
    fn untyped_relationship_traverses_all_types() {
        let store = social();
        let names = |q: &str| {
            let out = run(&super::parse(q).unwrap(), &store);
            let i = out.names.iter().position(|n| n == "n").expect("column n");
            let mut v: Vec<String> = out
                .rows
                .iter()
                .filter_map(|r| match &r[i] {
                    Value::Str(s) => Some(s.to_string()),
                    _ => None,
                })
                .collect();
            v.sort();
            v
        };
        // Bare-variable and empty-bracket untyped forms both traverse everything.
        assert_eq!(
            names("MATCH (a:Person {name:'alice'})-[r]->(b) RETURN b.name AS n"),
            vec!["bob", "carol", "graphdb"],
        );
        assert_eq!(
            names("MATCH (a:Person {name:'alice'})-[]->(b) RETURN b.name AS n"),
            vec!["bob", "carol", "graphdb"],
        );
        // A typed hop is narrower.
        assert_eq!(
            names("MATCH (a:Person {name:'alice'})-[:KNOWS]->(b) RETURN b.name AS n"),
            vec!["bob", "carol"],
        );
    }

    /// Parsed `upper(p.name)` matches the hand-built Call.
    #[test]
    fn string_fn_parse_matches_hand() {
        use crate::ir::{Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "u".into(),
            Expr::Call {
                name: "upper".into(),
                args: vec![Expr::Prop {
                    slot: 0,
                    key: "name".into(),
                }],
            },
        )]);
        assert_same("MATCH (p:Person) RETURN upper(p.name) AS u", &hand, &store);
    }

    #[test]
    fn string_fn_arity_errors() {
        assert!(super::parse("MATCH (p:Person) RETURN upper(p.name, 1) AS x").is_err());
        assert!(super::parse("MATCH (p:Person) RETURN replace(p.name, 'a') AS x").is_err());
    }

    // --- part 3.9: list literal + list functions (E4b) ---

    /// A list literal can hold non-constant elements; size/head/last read it.
    #[test]
    fn list_literal_and_functions() {
        let store = social();
        let q = "MATCH (p:Person) WHERE p.name='alice' RETURN \
                 size([p.age, 1, 2]) AS n, head([p.age, 1, 2]) AS h, last([p.age, 1, 2]) AS t";
        let out = run(&super::parse(q).unwrap(), &store);
        assert_eq!(num(&col(&out, 0, "n")), 3.0);
        assert_eq!(num(&col(&out, 0, "h")), 30.0); // p.age (alice)
        assert_eq!(num(&col(&out, 0, "t")), 2.0);
    }

    /// Empty list: size 0, head/last NULL. A list fn on a non-list is NULL.
    #[test]
    fn empty_list_and_non_list() {
        let store = social();
        let out = run(
            &super::parse(
                "MATCH (p:Person) WHERE p.name='alice' \
                 RETURN size([]) AS z, head([]) AS h, size(p.age) AS bad",
            )
            .unwrap(),
            &store,
        );
        assert_eq!(num(&col(&out, 0, "z")), 0.0);
        assert!(col(&out, 0, "h").is_null());
        assert!(col(&out, 0, "bad").is_null()); // size of a number → NULL
    }

    /// Parsed `[p.age, 1]` matches the hand-built `Expr::List`.
    #[test]
    fn list_literal_parse_matches_hand() {
        use crate::ir::{Expr, Plan};
        let store = social();
        let hand = Plan::Scan {
            label: Some("Person".into()),
        }
        .project(vec![(
            "xs".into(),
            Expr::List {
                items: vec![
                    Expr::Prop {
                        slot: 0,
                        key: "age".into(),
                    },
                    Expr::Lit(n(1.0)),
                ],
            },
        )]);
        assert_same("MATCH (p:Person) RETURN [p.age, 1] AS xs", &hand, &store);
    }

    // --- part 4: INSERT (write statements) ---

    /// Parsed INSERT matches the hand-built `Plan::Insert`: execute both onto
    /// fresh stores and confirm they answer the same query identically (and that
    /// the insert actually happened).
    #[test]
    fn insert_parse_matches_hand_plan() {
        use crate::exec::execute;
        use crate::ir::{InsertEdge, InsertNode, Plan};
        let hand = Plan::Insert {
            nodes: vec![
                InsertNode {
                    labels: vec!["Person".into()],
                    props: vec![("name".into(), s("x")), ("age".into(), n(1.0))],
                },
                InsertNode {
                    labels: vec!["Person".into()],
                    props: vec![("name".into(), s("y"))],
                },
            ],
            edges: vec![InsertEdge {
                from: 0,
                to: 1,
                etype: "KNOWS".into(),
                props: vec![],
            }],
        };
        let query = "INSERT (a:Person {name: 'x', age: 1})-[:KNOWS]->(b:Person {name: 'y'})";
        let mut st_p = Builder::default().build();
        let mut st_h = Builder::default().build();
        execute(&super::parse(query).unwrap(), &mut st_p).unwrap();
        execute(&hand, &mut st_h).unwrap();
        let probe = "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b, a.age AS age";
        let pp = super::parse(probe).unwrap();
        assert_eq!(bag(&run(&pp, &st_p)), bag(&run(&pp, &st_h)));
        assert_eq!(run(&pp, &st_p).rows.len(), 1); // the insert happened
    }

    /// A repeated variable references the same node, not a new one: `(a) … , (a)…`
    /// creates ONE `a`.
    #[test]
    fn insert_reuses_variable() {
        use crate::exec::execute;
        let mut store = Builder::default().build();
        execute(
            &super::parse("INSERT (a:P {name: 'a'}), (a)-[:R]->(b:P {name: 'b'})").unwrap(),
            &mut store,
        )
        .unwrap();
        let edge = run(
            &super::parse("MATCH (x:P)-[:R]->(y) RETURN x.name AS x, y.name AS y").unwrap(),
            &store,
        );
        assert_eq!(edge.rows.len(), 1);
        let cnt = run(
            &super::parse("MATCH (p:P) RETURN count(*) AS c").unwrap(),
            &store,
        );
        assert_eq!(num(&col(&cnt, 0, "c")), 2.0); // a reused, not duplicated
    }

    #[test]
    fn insert_errors() {
        assert!(super::parse("INSERT (a:P)-[:R]-(b:P)").is_err()); // undirected
        assert!(super::parse("INSERT (a:P {n: 1}), (a:P {n: 2})").is_err()); // redefine var
    }

    // --- part 5: SET / REMOVE (update statements) ---

    /// Parsed SET/REMOVE matches the hand-built `Plan::Update`: run both onto
    /// fresh copies and confirm the resulting property reads agree.
    #[test]
    fn update_parse_matches_hand_plan() {
        use crate::exec::execute;
        use crate::ir::{Expr, Plan, SetOp};
        let hand = Plan::Update {
            input: Box::new(
                Plan::Scan {
                    label: Some("Person".into()),
                }
                .filter(Expr::Compare {
                    op: crate::ir::CompareOp::Eq,
                    left: Box::new(Expr::Prop {
                        slot: 0,
                        key: "name".into(),
                    }),
                    right: Box::new(Expr::Lit(s("alice"))),
                }),
            ),
            ops: vec![
                SetOp::Set {
                    slot: 0,
                    key: "age".into(),
                    value: Expr::Lit(n(41.0)),
                },
                SetOp::Remove {
                    slot: 0,
                    key: "name".into(),
                },
            ],
        };
        let query = "MATCH (p:Person) WHERE p.name = 'alice' SET p.age = 41 REMOVE p.name";
        let mut st_p = social();
        let mut st_h = social();
        execute(&super::parse(query).unwrap(), &mut st_p).unwrap();
        execute(&hand, &mut st_h).unwrap();
        // Compare the whole Person table (age + name) between the two stores.
        let probe = "MATCH (p:Person) RETURN p.age AS age, p.name AS name";
        let pp = super::parse(probe).unwrap();
        assert_eq!(bag(&run(&pp, &st_p)), bag(&run(&pp, &st_h)));
    }

    /// SET only touches WHERE-matched rows; others are unchanged (hand-computed:
    /// only alice's age becomes 100).
    #[test]
    fn update_respects_where() {
        use crate::exec::execute;
        let mut store = social();
        execute(
            &super::parse("MATCH (p:Person) WHERE p.name = 'alice' SET p.age = 100").unwrap(),
            &mut store,
        )
        .unwrap();
        let out = run(
            &super::parse("MATCH (p:Person) RETURN p.name AS name, p.age AS age").unwrap(),
            &store,
        );
        let mut got: Vec<(String, f64)> = out
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(nm), Value::Num(a)) => (nm.to_string(), *a),
                _ => panic!("shape"),
            })
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            got,
            vec![
                ("alice".into(), 100.0),
                ("bob".into(), 25.0),
                ("carol".into(), 40.0)
            ]
        );
    }

    /// The null policy: `SET x = null` STORES a present null (has_prop true, reads
    /// null); `REMOVE x` makes it absent (has_prop false). Distinct operations.
    #[test]
    fn set_null_stores_remove_deletes() {
        use crate::exec::execute;
        let mut store = social();
        // node 0 = alice. SET a present null on 'age', remove 'name'.
        execute(
            &super::parse("MATCH (p:Person) WHERE p.name = 'alice' SET p.age = null").unwrap(),
            &mut store,
        )
        .unwrap();
        assert!(store.has_prop(0, "age")); // present…
        assert!(store.prop(0, "age").is_null()); // …but null
        execute(
            &super::parse("MATCH (p:Person) WHERE p.age = 25 REMOVE p.age").unwrap(),
            &mut store,
        )
        .unwrap();
        // bob (age 25) had age removed → absent
        assert!(!store.has_prop(1, "age"));
    }

    #[test]
    fn update_errors_on_unknown_var() {
        assert!(super::parse("MATCH (p:Person) SET q.age = 1").is_err());
    }

    // --- part 6: _MERGE (keyed upsert) ---

    fn user_store() -> Store {
        let mut st = Builder::default().build();
        st.create_unique_constraint("User", &["email"]).unwrap();
        st
    }
    fn merge(store: &mut Store, q: &str) -> Result<(), String> {
        crate::exec::execute(&super::parse(q).unwrap(), store).map(|_| ())
    }

    /// Create path: absent key → node created with all pattern props.
    #[test]
    fn merge_creates_when_absent() {
        let mut st = user_store();
        merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
        assert_eq!(st.nodes_with_label("User").len(), 1);
        assert!(matches!(st.prop(0, "email"), Value::Str(x) if &*x == "a"));
        assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A"));
    }

    /// Idempotence + default clobber: a second _MERGE on the same key updates the
    /// SAME node, clobbering the non-key payload to the new pattern value.
    #[test]
    fn merge_is_idempotent_and_clobbers_payload() {
        let mut st = user_store();
        merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
        merge(&mut st, "_MERGE (u:User {email: 'a', name: 'B'})").unwrap();
        assert_eq!(st.nodes_with_label("User").len(), 1); // no duplicate
        assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "B")); // clobbered
    }

    /// `_ON_CREATE SET` fires only on create; `_ON_UPDATE SET` REPLACES the
    /// default clobber (so the pattern payload is NOT re-clobbered on update).
    #[test]
    fn merge_on_create_and_on_update_dispositions() {
        let mut st = user_store();
        merge(
            &mut st,
            "_MERGE (u:User {email: 'a', name: 'A'}) _ON_CREATE SET u.created = true",
        )
        .unwrap();
        assert!(matches!(st.prop(0, "created"), Value::Bool(true)));
        // update with a new name, but _ON_UPDATE replaces the default: name stays
        // 'A', only seen is written; created stays (on_create didn't re-fire).
        merge(
            &mut st,
            "_MERGE (u:User {email: 'a', name: 'C'}) _ON_UPDATE SET u.seen = 1",
        )
        .unwrap();
        assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A")); // NOT clobbered
        assert!(matches!(st.prop(0, "seen"), Value::Num(x) if x == 1.0));
        assert!(matches!(st.prop(0, "created"), Value::Bool(true))); // survived
    }

    /// A WHERE-gated `_ON_UPDATE` whose predicate is false is a no-op (not an
    /// error): the existing value is left untouched.
    #[test]
    fn merge_on_update_where_gate_false_is_noop() {
        let mut st = user_store();
        merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
        merge(&mut st, "MATCH (u:User) SET u.version = 5").ok(); // seed a version
                                                                 // incoming version 3 is not newer → gate false → name unchanged.
        merge(
            &mut st,
            "_MERGE (u:User {email: 'a'}) _ON_UPDATE SET u.name = 'Z' WHERE u.version < 3",
        )
        .unwrap();
        assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A"));
    }

    /// `_ON_UPDATE_NOTHING` leaves the existing node untouched.
    #[test]
    fn merge_on_update_nothing() {
        let mut st = user_store();
        merge(&mut st, "_MERGE (u:User {email: 'a', name: 'A'})").unwrap();
        merge(
            &mut st,
            "_MERGE (u:User {email: 'a', name: 'X'}) _ON_UPDATE_NOTHING",
        )
        .unwrap();
        assert!(matches!(st.prop(0, "name"), Value::Str(x) if &*x == "A"));
    }

    /// `_MERGE` on a label with no applicable unique constraint errors.
    #[test]
    fn merge_without_constraint_errors() {
        let mut st = Builder::default().build(); // no constraint
        assert!(merge(&mut st, "_MERGE (u:User {email: 'a'})").is_err());
    }

    // --- part 7: relationship variables & edge properties (B5c) ---

    /// INSERT writes inline edge properties; a bound relationship variable reads
    /// them back (`r.weight`) alongside the landed node (`b.name`).
    #[test]
    fn insert_edge_props_then_read_via_rel_var() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        execute(
            &super::parse("INSERT (a:P {name: 'a'})-[:R {weight: 0.5}]->(b:P {name: 'b'})")
                .unwrap(),
            &mut st,
        )
        .unwrap();
        let out = run(
            &super::parse("MATCH (a:P)-[r:R]->(b) RETURN r.weight AS w, b.name AS who").unwrap(),
            &st,
        );
        assert_eq!(out.rows.len(), 1);
        assert_eq!(num(&col(&out, 0, "w")), 0.5);
        assert!(matches!(col(&out, 0, "who"), Value::Str(x) if &*x == "b"));
    }

    /// An edge property present on SOME edges must not use the all-present raw
    /// `Col::Num` fast path — the reader falls back to the null-carrying column, so
    /// the missing cell reads NULL (and is dropped by a numeric filter).
    #[test]
    fn edge_property_partly_present_reads_null() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        execute(
            &super::parse(
                "INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'}), \
                 (a)-[:R]->(c:P {name: 'c'})",
            )
            .unwrap(),
            &mut st,
        )
        .unwrap();
        // Projection: the w-less edge reads NULL, not a panic or a stale 0.
        let out = run(
            &super::parse("MATCH (a:P)-[r:R]->(b) RETURN b.name AS who, r.w AS w").unwrap(),
            &st,
        );
        let mut got: Vec<(String, Value)> = out
            .rows
            .iter()
            .map(|r| match (&r[0], &r[1]) {
                (Value::Str(s), w) => (s.to_string(), w.clone()),
                _ => panic!(),
            })
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(got[0].0, "b");
        assert_eq!(num(&got[0].1), 0.5);
        assert_eq!(got[1].0, "c");
        assert!(got[1].1.is_null());
        // Filter: `r.w > 0.4` keeps only the present-and-matching edge.
        let out = run(
            &super::parse("MATCH (a:P)-[r:R]->(b) WHERE r.w > 0.4 RETURN count(*) AS c").unwrap(),
            &st,
        );
        assert_eq!(num(&col(&out, 0, "c")), 1.0);
    }

    /// WHERE on an edge property filters edges: only the 0.5 edge passes `> 0.4`.
    #[test]
    fn where_on_edge_property() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        execute(
            &super::parse(
                "INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'}), \
                 (a)-[:R {w: 0.2}]->(c:P {name: 'c'})",
            )
            .unwrap(),
            &mut st,
        )
        .unwrap();
        let out = run(
            &super::parse("MATCH (a:P)-[r:R]->(b) WHERE r.w > 0.4 RETURN b.name AS who").unwrap(),
            &st,
        );
        let got: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                _ => panic!(),
            })
            .collect();
        assert_eq!(got, vec!["b"]);
    }

    /// SET on a bound relationship writes an EDGE property.
    #[test]
    fn set_edge_property_via_rel_var() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        execute(
            &super::parse("INSERT (a:P {name: 'a'})-[:R {w: 1}]->(b:P {name: 'b'})").unwrap(),
            &mut st,
        )
        .unwrap();
        execute(
            &super::parse("MATCH (a:P)-[r:R]->(b) SET r.w = 9").unwrap(),
            &mut st,
        )
        .unwrap();
        let out = run(
            &super::parse("MATCH (a:P)-[r:R]->(b) RETURN r.w AS w").unwrap(),
            &st,
        );
        assert_eq!(num(&col(&out, 0, "w")), 9.0);
    }

    /// An inline edge property in a MATCH pattern is a match filter on the edge.
    #[test]
    fn inline_edge_prop_is_a_match_filter() {
        use crate::exec::execute;
        let mut st = Builder::default().build();
        execute(
            &super::parse(
                "INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'}), \
                 (a)-[:R {w: 0.2}]->(c:P {name: 'c'})",
            )
            .unwrap(),
            &mut st,
        )
        .unwrap();
        let out = run(
            &super::parse("MATCH (a:P)-[r:R {w: 0.5}]->(b) RETURN b.name AS who").unwrap(),
            &st,
        );
        let got: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                _ => panic!(),
            })
            .collect();
        assert_eq!(got, vec!["b"]);
    }

    /// A bound relationship read lowers to `expand_edge` — cross-check vs the hand
    /// plan (edge at slot 1, node at slot 2).
    #[test]
    fn rel_var_read_matches_hand_plan() {
        use crate::exec::execute;
        use crate::ir::{Expr, Plan};
        let mut st = Builder::default().build();
        execute(
            &super::parse("INSERT (a:P {name: 'a'})-[:R {w: 0.5}]->(b:P {name: 'b'})").unwrap(),
            &mut st,
        )
        .unwrap();
        let hand = Plan::Scan {
            label: Some("P".into()),
        }
        .expand_edge(0, crate::ir::Dir::Out, &["R".to_string()])
        .project(vec![(
            "w".into(),
            Expr::Prop {
                slot: 1,
                key: "w".into(),
            },
        )]);
        assert_same("MATCH (a:P)-[r:R]->(b) RETURN r.w AS w", &hand, &st);
    }

    /// A relationship variable on a variable-length pattern is rejected (deferred).
    #[test]
    fn rel_var_on_varlength_errors() {
        assert!(super::parse("MATCH (a:P)-[r:R]->{1,2}(b) RETURN r.w AS w").is_err());
    }

    /// Parsed `_MERGE` matches the hand-built `Plan::Merge` (create + on_update):
    /// run both onto fresh constrained stores, confirm identical resulting props.
    #[test]
    fn merge_parse_matches_hand_plan() {
        use crate::exec::execute;
        use crate::ir::{Expr, MergeUpdate, Plan};
        let hand = Plan::Merge {
            label: "User".into(),
            props: vec![("email".into(), s("a")), ("name".into(), s("A"))],
            on_create: vec![("created".into(), Expr::Lit(Value::Bool(true)))],
            on_update: MergeUpdate::Set {
                assigns: vec![("seen".into(), Expr::Lit(n(1.0)))],
                filter: None,
            },
        };
        let query = "_MERGE (u:User {email: 'a', name: 'A'}) _ON_CREATE SET u.created = true \
                     _ON_UPDATE SET u.seen = 1";
        let mut st_p = user_store();
        let mut st_h = user_store();
        execute(&super::parse(query).unwrap(), &mut st_p).unwrap();
        execute(&hand, &mut st_h).unwrap();
        let probe = "MATCH (u:User) RETURN u.email AS e, u.name AS nm, u.created AS c";
        let pp = super::parse(probe).unwrap();
        assert_eq!(bag(&run(&pp, &st_p)), bag(&run(&pp, &st_h)));
    }

    /// `INSERT (n:…) RETURN n.…` binds the created node into scope so the trailing
    /// projection reads it — the engine's first write-then-return path. The
    /// returned row equals reading the same node back from the mutated store.
    #[test]
    fn insert_return_binds_created_node() {
        use crate::exec::execute;
        let mut st = social();
        let out = execute(
            &super::parse("INSERT (n:Person {name: 'z'}) RETURN n.name").unwrap(),
            &mut st,
        )
        .unwrap();
        // Exactly one projected row for the one created node.
        assert_eq!(out.rows.len(), 1);
        // …and it matches reading that node back (proving the bind, not a constant).
        let probe = super::parse("MATCH (n:Person {name: 'z'}) RETURN n.name").unwrap();
        assert_eq!(bag(&out), bag(&run(&probe, &st)));
    }

    /// The projection may read several properties of the created node.
    #[test]
    fn insert_return_projects_multiple_props() {
        use crate::exec::execute;
        let mut st = social();
        let out = execute(
            &super::parse("INSERT (n:Person {name: 'newbie', age: 99}) RETURN n.name, n.age")
                .unwrap(),
            &mut st,
        )
        .unwrap();
        assert_eq!(out.rows.len(), 1);
        let probe = super::parse("MATCH (n:Person {name: 'newbie'}) RETURN n.name, n.age").unwrap();
        assert_eq!(bag(&out), bag(&run(&probe, &st)));
    }

    /// `&`-separated labels create a multi-labelled node (`n:Person&Admin`): the
    /// created node answers a MATCH on EITHER label.
    #[test]
    fn insert_return_multi_label_ampersand() {
        use crate::exec::execute;
        let mut st = social();
        let out = execute(
            &super::parse("INSERT (n:Person&Admin {name: 'root'}) RETURN n.name").unwrap(),
            &mut st,
        )
        .unwrap();
        assert_eq!(out.rows.len(), 1);
        // The node carries BOTH labels — reachable via Admin and via Person.
        let by_admin = super::parse("MATCH (n:Admin) RETURN n.name").unwrap();
        assert_eq!(bag(&out), bag(&run(&by_admin, &st)));
        let by_person = super::parse("MATCH (n:Person {name: 'root'}) RETURN n.name").unwrap();
        assert_eq!(bag(&run(&by_person, &st)), bag(&run(&by_admin, &st)));
    }
}
