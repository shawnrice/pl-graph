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

use crate::gstr::GStr;
use std::collections::{HashMap, HashSet};

use crate::ir::{AggFn, CastTarget, CompareOp, Dir, Expr, PathMode, PathPart, Plan, TxKind};
use crate::value::Value;

/// Whether `e`'s type is STATICALLY known to be non-boolean — a value that can never be
/// a truth value, decided from the expression shape alone (no row data). A non-bool /
/// non-null literal, arithmetic, a list / record / map constructor, a non-boolean CAST,
/// a path or path accessor, and the count/collect/scalar-aggregate subqueries are all
/// definitely non-boolean. Everything dynamic — a property, parameter, function result,
/// field/subscript, `CASE` result, or a comparison/connective (already boolean) — is NOT
/// flagged here; a dynamic value is type-checked per row by `exec::as_truth` instead.
fn definitely_non_bool(e: &Expr) -> bool {
    match e {
        Expr::Lit(v) => !matches!(v, Value::Bool(_) | Value::Null),
        Expr::Arith { .. }
        | Expr::List { .. }
        | Expr::Record { .. }
        | Expr::MapLit { .. }
        | Expr::Path
        | Expr::PathAccess { .. }
        | Expr::GremlinPath { .. }
        | Expr::CountSubquery { .. }
        | Expr::CollectSubquery { .. }
        | Expr::AggSubquery { .. } => true,
        Expr::Cast { target, .. } => !matches!(target, CastTarget::Boolean),
        _ => false,
    }
}

/// Static (plan-time) boolean-context type check, matching Postgres and the pure-TS
/// engine: a value whose type is statically known to be non-boolean is rejected wherever
/// a truth value is required — a `WHERE` / `FILTER` / `HAVING` / per-repetition / edge
/// predicate, an `AND` / `OR` / `XOR` / `NOT` operand, or a `CASE WHEN` condition —
/// regardless of whether any row would reach it. This fires even on an empty match, and
/// is what makes `WHERE 0` / `WHERE (x AND {a: 1})` a plan-time `E_INVALID_VALUE` in both
/// engines. It closes the differential-fuzzer divergence where a selective seek narrowed
/// the rows to zero before `as_truth` could reject a sibling non-boolean conjunct at
/// runtime: the reject now happens before execution, independent of the seek. A
/// dynamically-typed operand (a bare property, `NOT n.s`, a function result) is left to
/// the per-row `as_truth` check — the engine is schemaless, so its type is unknowable
/// here. `bool_ctx` marks whether `e` sits in a boolean position.
fn check_bool_ctx(e: &Expr, bool_ctx: bool) -> Result<(), String> {
    if bool_ctx && definitely_non_bool(e) {
        return Err(crate::exec::TRUTH_TYPE_ERR.to_string());
    }
    match e {
        Expr::And(a, b) | Expr::Or(a, b) | Expr::Xor(a, b) => {
            check_bool_ctx(a, true)?;
            check_bool_ctx(b, true)?;
        }
        Expr::Not(x) => check_bool_ctx(x, true)?,
        Expr::Compare { left, right, .. } | Expr::Arith { left, right, .. } => {
            check_bool_ctx(left, false)?;
            check_bool_ctx(right, false)?;
        }
        Expr::In { needle, haystack } => {
            check_bool_ctx(needle, false)?;
            check_bool_ctx(haystack, false)?;
        }
        Expr::Call { args, .. } => {
            for a in args {
                check_bool_ctx(a, false)?;
            }
        }
        Expr::Case {
            branches,
            otherwise,
        } => {
            for (cond, val) in branches {
                check_bool_ctx(cond, true)?;
                check_bool_ctx(val, false)?;
            }
            if let Some(o) = otherwise {
                check_bool_ctx(o, false)?;
            }
        }
        Expr::List { items } => {
            for it in items {
                check_bool_ctx(it, false)?;
            }
        }
        Expr::Record { fields } => {
            for (_, v) in fields {
                check_bool_ctx(v, false)?;
            }
        }
        Expr::MapLit { entries, .. } => {
            for (_, v) in entries {
                check_bool_ctx(v, false)?;
            }
        }
        Expr::Field { base, .. } | Expr::Cast { expr: base, .. } => check_bool_ctx(base, false)?,
        Expr::Index { base, index, .. } => {
            check_bool_ctx(base, false)?;
            check_bool_ctx(index, false)?;
        }
        Expr::IsNull { expr, .. } => check_bool_ctx(expr, false)?,
        _ => {}
    }
    Ok(())
}

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
    // Correlated inline-property expressions `{k: <expr>}` (see `props`), captured as
    // token spans and lowered to `k = <expr>` filters once the node's slot is bound.
    PropExprs,
);

/// Correlated inline-property values `{k: <expr>}` captured as `(key, token-span)`
/// pairs — parsed to filters later (see `props` / `apply_prop_exprs`).
type PropExprs = Vec<(String, (usize, usize))>;

/// A parsed node in a context that does not carry correlated inline-property
/// expressions (see `node_plain`) — `ParsedNode` without its trailing `prop_exprs`.
type PlainNode = (
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

/// The ISO/IEC 39075 reserved words (verbatim from the now-removed lenke-core's list) — none may be a
/// bare identifier (a variable or label name). See [`Parser::ident_binding`].
const RESERVED_WORDS: &str = "abs acos all all_different and any array as asc ascending asin at atan \
atan2 avg big bigint binary bool boolean both btrim by byte_length bytes call cardinality case cast \
ceil ceiling char char_length character_length characteristics close coalesce collect_list commit \
copy cos cosh cot count create current_date current_graph current_property_graph current_schema \
current_time current_timestamp date datetime day dec decimal degrees delete desc descending detach \
distinct double drop duration duration_between element_id else end except exists exp false filter \
finish float float16 float32 float64 float128 float256 floor for from group having home_graph \
home_property_graph home_schema hour if implies in insert int integer int8 integer8 int16 integer16 \
int32 integer32 int64 integer64 int128 integer128 int256 integer256 intersect interval is leading \
left let like limit list ln local local_datetime local_time local_timestamp log log10 lower ltrim \
match max min minute mod month next nodetach normalize not nothing null nulls nullif octet_length of \
offset optional or order otherwise parameter parameters path path_length paths percentile_cont \
percentile_disc power precision property_exists radians real record remove replace reset return \
right rollback rtrim same schema second select session session_user set signed sin sinh size skip \
small smallint sqrt start stddev_pop stddev_samp string sum tan tanh then time timestamp trailing \
trim true typed ubigint uint uint8 uint16 uint32 uint64 uint128 uint256 union unknown unsigned upper \
use usmallint value varbinary varchar variable when where with xor year yield zoned zoned_datetime \
zoned_time abstract aggregate aggregates alter catalog clear clone constraint current_role \
current_user data directory dryrun exact existing function gqlstatus grant instant infinity number \
numeric on open partition procedure product project query records reference rename revoke substring \
system_user temporal unique unit values whitespace";

/// Is `word` (case-insensitive) an ISO reserved word, hence not a bare identifier?
fn is_reserved_word(word: &str) -> bool {
    use std::collections::HashSet;
    use std::sync::OnceLock;
    static R: OnceLock<HashSet<&'static str>> = OnceLock::new();
    R.get_or_init(|| RESERVED_WORDS.split(' ').collect())
        .contains(word.to_ascii_lowercase().as_str())
}

/// Turn a node's inline properties `{k: v, …}` into a chain of `Eq` filters on its
/// slot — the exact lowering of `WHERE slot.k = v AND …`. Sharing this single form
/// is what makes `(n:L {k: v})` and `MATCH (n:L) WHERE n.k = v` optimize to the same
/// plan (and seed the same index), so the two spellings cannot cost differently.
fn node_prop_filters(mut plan: Plan, slot: usize, props: Vec<(String, Value)>) -> Plan {
    for (k, val) in props {
        // An inline `{k: null}` constraint is an IS NULL test — it matches a node
        // whose `k` is null/absent — NOT the three-valued `k = null` (which is UNKNOWN
        // and matches nothing). Matches the TS engine's structural constraint semantics.
        let f = if val.is_null() {
            Expr::IsNull {
                expr: Box::new(Expr::Prop { slot, key: k }),
                negated: false,
            }
        } else {
            Expr::Compare {
                op: CompareOp::Eq,
                left: Box::new(Expr::Prop { slot, key: k }),
                right: Box::new(Expr::Lit(val)),
            }
        };
        plan = plan.filter(f);
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
    /// Inline edge property EXPRESSIONS (`-[:R {w: date('…'), n: a.n}]->`), populated
    /// only in an INSERT context (`rel(insert_ctx = true)`); a MATCH edge rejects them
    /// (use an inline WHERE). Empty otherwise.
    prop_exprs: Vec<(String, Expr)>,
    /// Token span of an inline `WHERE pred` on the edge (`-[e:T WHERE pred]->`),
    /// re-parsed once the edge is bound to a slot; `None` when absent.
    where_range: Option<(usize, usize)>,
}

/// Aggregate outputs referenced inside a projection expression use a slot at this
/// base + local index (`count(*) + 1` → `Slot(AGG_SLOT_BASE) + 1`), distinguishing
/// them from ordinary binding slots so `apply_items` can rewrite them to the real
/// post-aggregation column once the group schema is assembled.
///
/// It is only a sentinel far above any real per-query binding slot (a query has dozens,
/// never millions) and is never serialized, so its exact value is immaterial — but it
/// MUST fit in `usize` on every target. `1 << 28` clears real slots by a factor of
/// hundreds of millions while still fitting 32-bit `usize` (wasm32); `1 << 40` did not,
/// and const-overflowed the wasm build.
const AGG_SLOT_BASE: usize = 1 << 28;

/// Maximum expression-nesting depth accepted by the parser (see [`Parser::depth`]).
/// Far above any legitimate query (real expressions nest a handful deep) and safely
/// below the stack-overflow threshold on the SMALLEST stack the parser runs on — the
/// wasm CLI (~1 MB) and cargo's 2 MB test threads, not just the 8 MB native main
/// stack. MUST match the TS `@lenke/gql` parser's cap so the two engines accept/reject
/// the same queries.
const MAX_EXPR_DEPTH: usize = 128;

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
        // The TS engine's list aggregate is `collect_list` (SKIPS nulls); `collect` is a
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
    parse_with_params(query, &[])
}

/// Parse a GQL query, substituting each `$name` with the supplied parameter value
/// at parse time. An unbound `$name` is a parse error. The values are typed, so
/// they are never spliced into the query text (no injection), and the planner sees
/// literals (so `WHERE k = $p` / `{k: $p}` still seed an index).
pub fn parse_with_params(query: &str, params: &[(String, Value)]) -> Result<Plan, String> {
    parse_internal(query, params, false)
}

/// Parse a query in PREPARED mode: each `$name` becomes an unbound
/// [`Expr::Param`](crate::ir::Expr::Param) instead of being substituted, so the
/// parsed plan can be cached and bound to different values per run (see
/// [`crate::bind::bind_params`]). Params in `LIMIT`/`SKIP` and literal-only
/// positions (INSERT / procedure config) are not supported in prepared mode.
pub fn parse_prepared(query: &str) -> Result<Plan, String> {
    parse_internal(query, &[], true)
}

fn parse_internal(query: &str, params: &[(String, Value)], prepared: bool) -> Result<Plan, String> {
    let toks = lex(query)?;
    let mut p = Parser {
        toks,
        pos: 0,
        scope: HashMap::new(),
        slots: 0,
        path_vars: HashSet::new(),
        group_node_slots: HashSet::new(),
        group_edge_slots: HashSet::new(),
        group_var_depth: HashMap::new(),
        lets: Vec::new(),
        suppress_in: false,
        path_mode: PathMode::Trail,
        having_aggs: None,
        having_base: 0,
        params: params.iter().cloned().collect(),
        prepared,
        no_next: false,
        depth: 0,
    };
    // ISO transaction-control command (`START TRANSACTION`/`COMMIT`/`ROLLBACK`)? A
    // linear query never begins with a bare START/COMMIT/ROLLBACK, so there is no
    // ambiguity. It is a standalone statement — no UNION tail.
    if p.peek_kw("START") || p.peek_kw("COMMIT") || p.peek_kw("ROLLBACK") {
        let plan = p.parse_tx_control()?;
        if p.pos != p.toks.len() {
            return Err(format!("unexpected trailing input at token {}", p.pos));
        }
        return Ok(plan);
    }
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
        p.group_node_slots = HashSet::new();
        p.group_edge_slots = HashSet::new();
        p.group_var_depth = HashMap::new();
        // A set operator combined with NEXT is unsupported (both engines reject it);
        // a UNION arm must not consume a trailing NEXT as its own pipeline boundary.
        p.no_next = true;
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
    /// A query parameter reference `$name` (the name, without the `$`).
    Param(String),
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
            '$' => {
                // A query parameter `$name` — the name uses the bare-identifier char set.
                let start = i + 1;
                let mut j = start;
                while j < b.len() && (b[j].is_alphanumeric() || b[j] == '_') {
                    j += 1;
                }
                if j == start {
                    return Err("expected a parameter name after `$`".into());
                }
                out.push(Tok::Param(b[start..j].iter().collect()));
                i = j;
                continue;
            }
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
                // the TS engine's lexer; a malformed `\u`/`\U` is a syntax error (kept as an
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
                // f64, matching the TS engine). Else a decimal.
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
                // Matches the TS engine's lexer exactly.
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

/// The parsed shape of a quantified subpath group `((x)-[e1]->(m)…){n,m}`: its
/// uniform hop direction/edge-type, the quantifier bounds (in REPETITIONS), the hop
/// count `k`, the inner variable names per position (`node_vars` len `k+1`,
/// `edge_vars` len `k`), and an optional single-hop per-rep predicate.
struct SubpathGroup {
    dir: Dir,
    etypes: Vec<String>,
    min: u32,
    max: u32,
    k: u32,
    node_vars: Vec<Option<String>>,
    edge_vars: Vec<Option<String>>,
    per_rep_pred: Option<Expr>,
    /// An INNER quantifier on a single-hop endpoint-only body — `( ()-[:R]->{a,b}()
    /// ){c,d}`. The whole thing reaches the same endpoints as a var-length over the
    /// combined bounds `[a*c, b*d]`, so the caller desugars it to one `var_length`.
    inner_quant: Option<(u32, u32)>,
}

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
    /// Slots holding a group-variable NODE list (`x`/`y` of a quantified subpath
    /// group) resp. EDGE list (`e`) — so `x[i]`/`e[i]` can tag the element kind and
    /// `x[i].prop` resolves the node/edge property. See `Plan::RepeatGroup`.
    group_node_slots: HashSet<usize>,
    group_edge_slots: HashSet<usize>,
    /// The list-nesting DEPTH of a group-variable slot (1 = flat `x[i]`, 2 = a nested
    /// group's `x[i][j]`). Absent → 1. Decides which subscript level yields the typed
    /// element in `field_chain` (only the `depth`-th subscript is a node/edge).
    group_var_depth: HashMap<usize, u8>,
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
    /// Query parameters (`$name` → value), supplied by the caller. A `$name` is
    /// substituted to its `Value` at parse time (see `lookup_param`), so the value
    /// is typed — never spliced into query text (no injection) — and the planner
    /// sees a literal (index seeding on `WHERE k = $p` / `{k: $p}` still fires).
    params: HashMap<String, Value>,
    /// PREPARED mode: emit each `$name` as an unbound [`crate::ir::Expr::Param`]
    /// instead of substituting it, so the plan can be cached and bound per run. Set
    /// by [`parse_prepared`]; the `params` map is empty then.
    prepared: bool,
    /// Set while parsing a UNION arm: a `NEXT` pipeline boundary combined with a set
    /// operator is a documented limitation (both engines reject it), so `query_tail`
    /// refuses to consume `NEXT` here rather than silently re-associating the union.
    no_next: bool,
    /// Current expression-nesting depth, bounded by [`MAX_EXPR_DEPTH`] via [`Parser::nest`].
    /// A recursive-descent parser recurses once per nested `(…)`/`[…]`/`NOT`/unary-`-`/`!`,
    /// so an unbounded query string (`RETURN [[[[…]]]]`) would otherwise overflow the
    /// native stack (SIGSEGV) or trap the wasm REPL — before any operator-chain limit,
    /// which is only checked post-parse. The [`crate::ir::Expr`] tree the parser builds is
    /// also walked recursively by `optimize`/`exec`, so the cap protects them too.
    depth: usize,
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

    /// Run `f` one expression-nesting level deeper, rejecting past [`MAX_EXPR_DEPTH`]
    /// before the stack overflows. Wraps each self-recursive descent (`expr`, `NOT`,
    /// unary `-`, `!`); the depth is decremented on every non-fatal return so sibling
    /// breadth (a wide list) never accumulates — only genuine nesting does.
    fn nest<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, String>) -> Result<T, String> {
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            self.depth -= 1;

            return Err(format!(
                "E_RESOURCE_EXHAUSTED: expression nesting exceeds the maximum depth of {MAX_EXPR_DEPTH}"
            ));
        }
        let r = f(self);
        self.depth -= 1;

        r
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

    /// Parse an ISO GQL transaction-control command:
    ///   `START TRANSACTION [READ ONLY | READ WRITE]` | `COMMIT [WORK]` | `ROLLBACK [WORK]`
    /// Mirrors the TS engine's grammar — the access mode is optional (default READ WRITE), and
    /// `WORK` is an optional ISO noise word on COMMIT/ROLLBACK. The single-program
    /// combined form (`START TRANSACTION <stmts> … COMMIT` in one query) is NOT parsed,
    /// matching the TS engine: issue the commands as separate statements.
    fn parse_tx_control(&mut self) -> Result<Plan, String> {
        if self.eat_kw("START") {
            if !self.eat_kw("TRANSACTION") {
                return Err("expected TRANSACTION after START".to_string());
            }
            // Optional `READ ONLY | READ WRITE` access mode.
            let read_only = if self.eat_kw("READ") {
                if self.eat_kw("ONLY") {
                    true
                } else if self.eat_kw("WRITE") {
                    false
                } else {
                    return Err("expected ONLY or WRITE after READ".to_string());
                }
            } else {
                false
            };
            return Ok(Plan::TxControl {
                kind: TxKind::Start,
                read_only,
            });
        }
        let kind = if self.eat_kw("COMMIT") {
            TxKind::Commit
        } else if self.eat_kw("ROLLBACK") {
            TxKind::Rollback
        } else {
            return Err("expected START TRANSACTION, COMMIT, or ROLLBACK".to_string());
        };
        self.eat_kw("WORK"); // optional ISO noise word
        Ok(Plan::TxControl {
            kind,
            read_only: false,
        })
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(format!("expected an identifier, got {other:?}")),
        }
    }

    /// An identifier in a BINDING position (a variable or label name), where an ISO
    /// reserved word is not a bare identifier (`MATCH (select)` / `(n:Match)` are
    /// rejected, matching the TS engine). Reserved words stay usable as keywords, function
    /// names, and property keys — only a fresh binding name is constrained.
    fn ident_binding(&mut self) -> Result<String, String> {
        let s = self.ident()?;
        if is_reserved_word(&s) {
            return Err(format!(
                "`{s}` is a reserved word and cannot be a bare identifier"
            ));
        }
        Ok(s)
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
        // A leading `FOR <var> IN <list>` — a list-unwind that seeds the query (no
        // prior MATCH), rooted at one unit row. `Plan::Row` materializes as a single
        // dummy column, so the first binding lands at slot 1 (past it).
        if self.eat_kw("FOR") {
            self.slots = 1;
            let plan = self.for_clause(Plan::Row)?;
            return self.query_tail(plan);
        }
        // A leading `OPTIONAL MATCH <pattern>` with no prior MATCH: the pattern's
        // matches, OR — if it matches nothing — one all-NULL row (the ISO left-outer
        // against the implicit single unit row). `NullPadIfEmpty` supplies that row.
        if self.peek_kw("OPTIONAL") {
            self.eat_kw("OPTIONAL");
            if !self.eat_kw("MATCH") {
                return Err("expected MATCH after OPTIONAL".into());
            }
            let plan = self.match_body()?;
            let width = self.slots;
            let plan = Plan::NullPadIfEmpty {
                input: Box::new(plan),
                width,
            };
            return self.query_tail(plan);
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
                    plan = plan.filter(self.bool_pred()?);
                }
                return self.query_tail(plan);
            }
            let any = self.parse_bare_selector();
            let mut plan = self.match_body()?;
            if any == Some(true) {
                plan = plan.distinct();
            }
            return self.query_tail(plan);
        }
        // Bare selector form: `MATCH ALL SHORTEST (a)-[:R]->*(x)` — no path variable,
        // just the reached endpoints.
        if let Some(selector) = self.parse_shortest_selector()? {
            let (mut plan, scope, slots) = self.shortest_pattern(selector)?;
            self.scope = scope;
            self.slots = slots;
            if self.eat_kw("WHERE") {
                plan = plan.filter(self.bool_pred()?);
            }
            return self.query_tail(plan);
        }
        // Bare `ALL`/`ANY` selector (no SHORTEST): `ALL` is the default (every path);
        // `ANY` keeps one arbitrary path per endpoint (dedup the pattern's bindings).
        let any = self.parse_bare_selector();
        let mut plan = self.match_body()?;
        if any == Some(true) {
            plan = plan.distinct();
        }
        self.query_tail(plan)
    }

    /// A BARE `ALL` / `ANY` path selector (NOT `… SHORTEST`, which `parse_shortest_
    /// selector` owns). Returns `Some(true)` for `ANY` (dedup one path per endpoint),
    /// `Some(false)` for `ALL` (the default — every path), `None` for neither.
    fn parse_bare_selector(&mut self) -> Option<bool> {
        let next_is_shortest = matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s))
            if s.eq_ignore_ascii_case("SHORTEST"));
        if next_is_shortest {
            return None;
        }
        if self.peek_kw("ALL") {
            self.eat_kw("ALL");
            Some(false)
        } else if self.peek_kw("ANY") {
            self.eat_kw("ANY");
            Some(true)
        } else {
            None
        }
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
        } else if self.eat_kw("REPEATABLE") {
            // `REPEATABLE ELEMENTS` — the ISO match mode that allows reusing any
            // element: a WALK.
            self.eat_kw("ELEMENTS");
            self.path_mode = PathMode::Walk;
        } else if self.eat_kw("DIFFERENT") {
            // `DIFFERENT EDGES` — no edge reused: the default TRAIL.
            self.eat_kw("EDGES");
            self.path_mode = PathMode::Trail;
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
            // the `join/tri` fan-out pattern); the chained expand walks adjacency. Any
            // non-foldable shape (first node unbound/relabeled, or a landing var that
            // re-binds an existing one) rewinds and takes the correct hash join.
            let saved = self.pos;
            if matches!(self.peek(), Some(Tok::LParen)) {
                let probe = self.node_plain()?;
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
            let pred = self.bool_pred()?;
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
            } else if self.peek_kw("RETURN") && !self.no_next && self.scan_for_next() {
                // `RETURN … NEXT …` (ISO statement composition): the RETURN is a
                // PIPELINE boundary, not the terminal projection — its columns become
                // the next statement's driving table. It behaves exactly as a `WITH`
                // (elements stay handles, so `NEXT MATCH (person)…` can re-traverse a
                // carried node), so route it through `with_clause`, then an optional
                // `YIELD` selects/renames the piped columns, and the loop continues
                // into the next part (FILTER / MATCH / LET / RETURN).
                self.eat_kw("RETURN");
                plan = self.with_clause(plan)?;
                if !self.eat_kw("NEXT") {
                    return Err("expected NEXT after a pipelined RETURN".into());
                }
                if self.eat_kw("YIELD") {
                    plan = self.next_yield(plan)?;
                }
            } else if self.eat_kw("LET") {
                plan = self.let_clause(plan)?;
            } else if self.eat_kw("OPTIONAL") {
                // `OPTIONAL CALL (…) { … }` is a LEFT-outer correlated subquery;
                // `OPTIONAL MATCH …` is the single-hop left join.
                if self.eat_kw("CALL") {
                    plan = self.call_inline(plan, true)?;
                } else {
                    plan = self.optional_match(plan)?;
                }
            } else if self.eat_kw("MATCH") {
                plan = self.match_continue(plan)?;
            } else if self.eat_kw("CALL") {
                plan = self.call_inline(plan, false)?;
            } else if self.eat_kw("FOR") {
                plan = self.for_clause(plan)?;
            } else if self.eat_kw("FILTER") {
                // `FILTER [WHERE] <cond>` — the ISO standalone filtering statement: a
                // predicate over the current bindings, no projection.
                self.eat_kw("WHERE");
                plan = plan.filter(self.bool_pred()?);
            } else {
                break;
            }
        }
        // Statement-position `ORDER BY [OFFSET n] [LIMIT n]` BEFORE `RETURN`: sort
        // and page the bound rows, then the following RETURN projects. Core allows
        // this as a standalone order-and-page clause, and allows it to REPEAT (`ORDER
        // BY … LIMIT 2 ORDER BY … DESC LIMIT 1` — page then re-page), so this loops.
        // `SKIP` is not a valid STARTER here (only ORDER/OFFSET/LIMIT), matching the TS engine.
        while self.peek_kw("ORDER") || self.peek_kw("OFFSET") || self.peek_kw("LIMIT") {
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
            // `MATCH … SET … RETURN …`: read-after-write. The RETURN projects over the
            // SAME frontier the writes touched, seeded from `Row` and read against the
            // mutated store (see `Plan::UpdateReturn`). Without a RETURN it is a plain
            // Update yielding no rows.
            if self.eat_kw("RETURN") {
                let distinct = self.eat_kw("DISTINCT");
                let items = self.return_items()?;
                let tail = self.project_and_page(Plan::Row, distinct, items)?;
                return Ok(Plan::UpdateReturn {
                    input: Box::new(plan),
                    ops,
                    tail: Box::new(tail),
                });
            }
            return Ok(Plan::Update {
                input: Box::new(plan),
                ops,
            });
        }
        // A row-driven INSERT tail: `FOR … INSERT (…)` (or `MATCH … INSERT (…)`) —
        // create the templated nodes/edges once per input row, evaluating each
        // property EXPRESSION against that row (so `FOR x IN […] INSERT (:N {v: x})`
        // reads the unwound `x`). The bare top-level `INSERT` (constant literals, no
        // input) stays on the `insert()` path in `query()`.
        if self.eat_kw("INSERT") {
            return self.insert_from(plan);
        }
        if !self.eat_kw("RETURN") {
            return Err("expected RETURN, SET, REMOVE, WITH, INSERT, or MATCH".into());
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
        let mut group_by_present = false;
        let mut group_exprs: Vec<Expr> = Vec::new();
        if self.eat_kw("GROUP") {
            if !self.eat_kw("BY") {
                return Err("expected BY after GROUP".into());
            }
            loop {
                group_exprs.push(self.expr()?);
                group_by_present = true;
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        let visible: Vec<String> = items.iter().map(RetItem::name).collect();
        let has_agg = items.iter().any(|it| matches!(it, RetItem::Agg(_)));
        let has_agg_expr = items.iter().any(|it| matches!(it, RetItem::AggExpr { .. }));
        // A GROUP BY key that is NOT already a visible non-aggregate output item is a
        // HIDDEN grouping key: it widens the grouping (`RETURN count(*) GROUP BY
        // e.dept` = one row per dept) without appearing in the output. Appended as a
        // hidden Key item so the aggregate groups on it, then dropped by a final
        // schema-aware projection. (Only for the simple-aggregate shape; an
        // aggregate-expression projection already re-projects in item order.)
        let visible_key_exprs: Vec<Expr> = items
            .iter()
            .filter_map(|it| match it {
                RetItem::Key(_, e) => Some(e.clone()),
                _ => None,
            })
            .collect();
        let mut extra_group: Vec<(String, Expr)> = Vec::new();
        if has_agg && !has_agg_expr {
            for ge in &group_exprs {
                if !visible_key_exprs.iter().any(|k| expr_eq(k, ge)) {
                    let name = format!("__gk{}", extra_group.len());
                    extra_group.push((name, ge.clone()));
                }
            }
        }
        for (name, e) in &extra_group {
            items.push(RetItem::Key(name.clone(), e.clone()));
        }
        // A simple aggregate's plan schema is `[keys… , aggs…]` (see `apply_items`), which
        // is NOT the RETURN item order when an aggregate precedes a key (`RETURN count(*) AS
        // c, n.k AS g`). It ALWAYS needs the final reorder-to-visible-order projection, not
        // only when there are extra (hidden) group keys to drop.
        let needs_schema_proj = (has_agg && !has_agg_expr) || !extra_group.is_empty();
        // `GROUP BY <keys>` with NO aggregate is DISTINCT over the projection (the
        // returned items ARE the keys), matching the TS engine.
        let group_distinct = group_by_present && !has_agg;
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

        // The column each visible output name occupies in the plan's schema. For a
        // simple aggregate the schema is `[keys… , aggs…]` in item order, NOT the
        // RETURN order, so an ORDER-BY alias must resolve to the true column (an agg
        // sits after every key). For non-aggregate / aggregate-expression plans the
        // columns already match the visible order.
        let visible_cols: Vec<usize> = if has_agg && !has_agg_expr {
            let mut schema: Vec<String> = Vec::new();
            for it in &items {
                if let RetItem::Key(n, _) = it {
                    schema.push(n.clone());
                }
            }
            for it in &items {
                if let RetItem::Agg(a) = it {
                    schema.push(a.name.clone());
                }
            }
            visible
                .iter()
                .map(|n| schema.iter().position(|s| s == n).unwrap_or(0))
                .collect()
        } else {
            (0..visible.len()).collect()
        };

        // ORDER BY: a key that is a visible output alias sorts by that column; a key
        // that is an EXPRESSION over the bindings (`ORDER BY n.age`, `a.x + a.y`) is
        // projected as a HIDDEN column here, sorted on, then dropped by a final
        // projection — so ORDER BY is not limited to the returned columns.
        let mut hidden: Vec<(String, Expr)> = Vec::new();
        let keys = if self.eat_kw("ORDER") {
            if !self.eat_kw("BY") {
                return Err("expected BY after ORDER".into());
            }
            // An ORDER-BY *expression* may reference an output alias by name inside a
            // larger expression (`ORDER BY (LET x = a IN x END)`, where `a` is a
            // RETURN alias). Expose the non-aggregate aliases' defining expressions as
            // LET-style locals for the duration of the sort-key parse, so a bare alias
            // name inlines to its definition (evaluated over the bindings), then remove
            // them. (A bare top-level alias still takes the visible-column fast path.)
            let let_base = self.lets.len();
            if !has_agg {
                for it in &items {
                    if let RetItem::Key(name, e) = it {
                        // Do NOT shadow a bound graph variable that is projected bare
                        // (`RETURN a … ORDER BY a.name`): `a.name` must read the node's
                        // property from the binding scope, and bare `ORDER BY a` already
                        // takes the visible-column fast path in `order_keys`. Inlining `a`
                        // here would resolve it to its output column and orphan the
                        // trailing `.name`. Only inline aliases that are NOT graph vars
                        // (computed expressions like `RETURN u.n AS a … ORDER BY a * 2`).
                        if !self.scope.contains_key(name) {
                            self.lets.push((name.clone(), e.clone()));
                        }
                    }
                }
            }
            let ks = self.order_keys(&visible, &visible_cols, has_agg, &key_slots, &mut hidden)?;
            self.lets.truncate(let_base);
            ks
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
        } else if group_distinct && hidden.is_empty() {
            plan = plan.distinct();
        }
        // `OFFSET` is the ISO spelling of `SKIP` — a synonym here (the TS engine accepts both).
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
        // Drop the hidden columns (ORDER-BY sort keys and/or non-returned GROUP BY
        // keys), restoring exactly the visible outputs. When extra group keys were
        // added the aggregate schema is `[keys… , aggs…]` in item order — not the
        // RETURN order — so map each visible name to its true column; otherwise the
        // projection produced columns in visible order already.
        if needs_schema_proj {
            let mut schema: Vec<String> = Vec::new();
            for it in &items {
                if let RetItem::Key(n, _) = it {
                    schema.push(n.clone());
                }
            }
            for it in &items {
                if let RetItem::Agg(a) = it {
                    schema.push(a.name.clone());
                }
            }
            let proj = visible
                .iter()
                .map(|n| {
                    let col = schema
                        .iter()
                        .position(|s| s == n)
                        .expect("a visible output name must exist in the aggregate schema");
                    (n.clone(), Expr::Slot(col))
                })
                .collect();
            plan = plan.project(proj);
        } else if !hidden.is_empty() {
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

    /// Whether a bracket-depth-0 `NEXT` follows the cursor within this linear query —
    /// the marker that the RETURN about to be parsed is a PIPELINE boundary (its output
    /// feeds the next statement), not the terminal projection. A depth-0 `UNION` stops
    /// the scan: `UNION` binds tighter than `NEXT`, so the RETURN belongs to the union
    /// arm, and the whole union is the NEXT operand.
    fn scan_for_next(&self) -> bool {
        let mut depth = 0i32;
        let mut i = self.pos;
        while let Some(t) = self.toks.get(i) {
            match t {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Ident(s) if depth == 0 && s.eq_ignore_ascii_case("NEXT") => return true,
                Tok::Ident(s) if depth == 0 && s.eq_ignore_ascii_case("UNION") => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// `YIELD col [AS alias], …` after `NEXT`: select (and optionally rename) the piped
    /// columns, rebinding scope to exactly the yielded ones.
    fn next_yield(&mut self, plan: Plan) -> Result<Plan, String> {
        let mut items: Vec<RetItem> = Vec::new();
        loop {
            let col = self.ident()?;
            let slot = *self
                .scope
                .get(&col)
                .ok_or_else(|| format!("YIELD: unknown column `{col}`"))?;
            let name = if self.eat_kw("AS") {
                self.ident()?
            } else {
                col
            };
            items.push(RetItem::Key(name, Expr::Slot(slot)));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        let (plan, out_names) = apply_items(plan, &items);
        self.scope = out_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        self.slots = out_names.len();
        Ok(plan)
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
            null_on_empty: false,
            numeric_only: false,
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
        let having_raw = self.bool_pred()?;
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
                _ => Ok(Some(ShortestSelector::ShortestK { k: k as u32, group })),
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
        let (va, la, va_props, va_where, _va_le) = self.node_plain()?;
        if let Some(v) = &va {
            scope.insert(v.clone(), 0);
        }
        let rel = self.rel(false)?;
        // `*` → min 0 unbounded, `+` → min 1 unbounded, `{n,m}` → a bounded hop range.
        let (min, max) = if self.eat(&Tok::Star) {
            (0, None)
        } else if self.eat(&Tok::Plus) {
            (1, None)
        } else if let Some((lo, hi)) = self.opt_quantifier()? {
            (lo, Some(hi))
        } else {
            return Err("a shortest path requires a `*`, `+`, or `{n,m}` quantifier".into());
        };
        let (vb, _lb, vb_props, vb_where, _vb_le) = self.node_plain()?;
        if va_where.is_some() || vb_where.is_some() {
            return Err("inline WHERE on a shortest-path node is not supported".into());
        }
        // A per-hop edge `WHERE` (`-[e:R WHERE e.w > 5]->*`) filters which edges the
        // path may traverse — parse it against a SCALAR mini-scope (the edge at slot
        // 0), independent of the outer bindings, since it is evaluated per edge.
        let edge_pred = if let Some(r) = rel.where_range {
            let Some(evar) = rel.var.clone() else {
                return Err(
                    "a per-hop edge WHERE on a shortest path needs an edge variable".into(),
                );
            };
            let saved_scope = std::mem::take(&mut self.scope);
            let saved_slots = self.slots;
            self.scope = HashMap::from([(evar, 0usize)]);
            self.slots = 1;
            let pred = self.parse_captured_where(r)?;
            self.scope = saved_scope;
            self.slots = saved_slots;
            Some(Box::new(pred))
        } else {
            None
        };
        // Seed node: label + inline props seed the scan (slot 0).
        let mut plan = node_prop_filters(Plan::Scan { label: la }, 0, va_props);
        plan = plan.shortest_path(0, rel.dir, &rel.etypes, min, max, selector, edge_pred);
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
    /// (no clause) is `false` (nulls sort LAST, both directions — matching the TS engine).
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
        visible_cols: &[usize],
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
                    .filter(|_| terminator(self.toks.get(self.pos + 1)))
                    .map(|i| visible_cols[i]),
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
            Some(Tok::Param(name)) if self.prepared => Err(format!(
                "parameter `${name}` in LIMIT/SKIP is not supported in a prepared statement"
            )),
            Some(Tok::Param(name)) => match self.lookup_param(&name)? {
                Value::Num(n) if n >= 0.0 && n.fract() == 0.0 => Ok(n as usize),
                _ => Err(format!("parameter `${name}` is not a non-negative integer")),
            },
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
            self.literal_props("a _MERGE key")?
        } else {
            Vec::new()
        };
        self.expect(&Tok::RParen)?;

        // An edge form `(a:A {..})-[m:R {..}]->(b:B {..})` — upsert the single edge
        // between the two key-matched endpoints. Detected by an edge delimiter after
        // the first node.
        if matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde)) {
            return self.merge_edge(var, label, props);
        }

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
                Some(self.bool_pred()?)
            } else {
                None
            };
            crate::ir::MergeUpdate::Set { assigns, filter }
        } else {
            crate::ir::MergeUpdate::Clobber
        };

        // `_MERGE (…) RETURN <items> [ORDER BY/OFFSET/LIMIT]` — a read-after-write tail
        // over the merged node (bound at slot 0, above). Mirrors `INSERT … RETURN`;
        // query_tail also supplies ORDER BY/OFFSET/LIMIT on the single-row result.
        let tail = if self.peek_kw("RETURN")
            || self.peek_kw("ORDER")
            || self.peek_kw("OFFSET")
            || self.peek_kw("LIMIT")
        {
            Some(Box::new(self.query_tail(Plan::Row)?))
        } else {
            None
        };

        Ok(Plan::Merge {
            label,
            props,
            on_create,
            on_update,
            tail,
        })
    }

    /// The `_MERGE` edge form, entered once the first node's `-`/`<-`/`~` delimiter
    /// is seen. `start_*` is that first node. Parses the relationship and the second
    /// node, binds start/end/edge at slots 0/1/2, then the dispositions (slot-aware,
    /// since a disposition may target either endpoint or the edge).
    fn merge_edge(
        &mut self,
        start_var: Option<String>,
        start_label: String,
        start_props: Vec<(String, crate::value::Value)>,
    ) -> Result<Plan, String> {
        let rel = self.rel(false)?;
        self.expect(&Tok::LParen)?;
        let end_var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident()?)
        } else {
            None
        };
        self.expect(&Tok::Colon)?;
        let end_label = self.ident()?;
        let end_props = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.literal_props("a _MERGE key")?
        } else {
            Vec::new()
        };
        self.expect(&Tok::RParen)?;

        // Exactly one plain edge type (no `|`-disjunction, no `!`-negation).
        if rel.etypes.len() != 1 || rel.etypes[0] == "!" {
            return Err("E_INVALID_GRAPH_OP: a _MERGE edge must carry exactly one type".into());
        }
        let etype = rel.etypes[0].clone();
        if rel.where_range.is_some() {
            return Err("E_INVALID_GRAPH_OP: a _MERGE edge does not take an inline WHERE".into());
        }

        // Slots: start = 0, end = 1, edge = 2 — for the disposition expressions.
        self.scope = HashMap::new();
        if let Some(v) = &start_var {
            self.scope.insert(v.clone(), 0);
        }
        if let Some(v) = &end_var {
            self.scope.insert(v.clone(), 1);
        }
        if let Some(v) = &rel.var {
            self.scope.insert(v.clone(), 2);
        }
        self.slots = 3;

        let on_create = if self.eat_kw("_ON_CREATE") {
            if !self.eat_kw("SET") {
                return Err("expected SET after _ON_CREATE".into());
            }
            self.assign_list_slotted()?
        } else {
            Vec::new()
        };

        let on_update = if self.eat_kw("_ON_UPDATE_NOTHING") {
            crate::ir::MergeEdgeUpdate::Nothing
        } else if self.eat_kw("_ON_UPDATE") {
            if !self.eat_kw("SET") {
                return Err("expected SET after _ON_UPDATE".into());
            }
            let assigns = self.assign_list_slotted()?;
            let filter = if self.eat_kw("WHERE") {
                Some(self.bool_pred()?)
            } else {
                None
            };
            crate::ir::MergeEdgeUpdate::Set { assigns, filter }
        } else {
            crate::ir::MergeEdgeUpdate::Clobber
        };

        Ok(Plan::MergeEdge {
            start_label,
            start_props,
            end_label,
            end_props,
            dir: rel.dir,
            etype,
            edge_props: rel.props,
            on_create,
            on_update,
        })
    }

    // assign_list_slotted := var '.' key '=' expr ( ',' … )* — like `assign_list`
    // but KEEPS each assignment's target slot (edge `_MERGE` writes span the two
    // endpoints and the edge).
    fn assign_list_slotted(&mut self) -> Result<Vec<(usize, String, Expr)>, String> {
        let mut out = Vec::new();
        loop {
            let (slot, key) = self.slot_dot_key()?;
            self.expect(&Tok::Eq)?;
            let value = self.expr()?;
            out.push((slot, key, value));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(out)
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
                    // `SET n:Label` / `SET n IS Label` mutates the label set;
                    // `SET n.key = expr` mutates a property.
                    let (slot, label) = self.slot_label()?;
                    if let Some(label) = label {
                        ops.push(crate::ir::SetOp::AddLabel { slot, label });
                    } else {
                        self.expect(&Tok::Dot)?;
                        let key = self.ident()?;
                        self.expect(&Tok::Eq)?;
                        let value = self.expr()?;
                        ops.push(crate::ir::SetOp::Set { slot, key, value });
                    }
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
            } else if self.eat_kw("REMOVE") {
                loop {
                    // `REMOVE n:Label` / `REMOVE n IS Label` vs `REMOVE n.key`.
                    let (slot, label) = self.slot_label()?;
                    if let Some(label) = label {
                        ops.push(crate::ir::SetOp::RemoveLabel { slot, label });
                    } else {
                        self.expect(&Tok::Dot)?;
                        let key = self.ident()?;
                        ops.push(crate::ir::SetOp::Remove { slot, key });
                    }
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

    // Read a bound variable and its slot; if a `:Label` / `IS Label` follows,
    // consume the label and return it (`Some`) — the caller emits a label op.
    // Otherwise the next token is a `.key` the caller reads (`None`).
    fn slot_label(&mut self) -> Result<(usize, Option<String>), String> {
        let var = self.ident()?;
        let slot = *self
            .scope
            .get(&var)
            .ok_or_else(|| format!("unknown variable `{var}`"))?;
        if self.eat(&Tok::Colon) || self.eat_kw("IS") {
            Ok((slot, Some(self.ident()?)))
        } else {
            Ok((slot, None))
        }
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
        // Parse each property as an EXPRESSION (a literal lifts to `Expr::Lit`), so a
        // plain INSERT accepts a constant expression — `duration('P1D')`, `1 + 1`,
        // `date('2020-01-01')` — exactly as TS does. A plain INSERT has no enclosing
        // bindings (`self.scope` is empty here), so every property is constant and no
        // node resolves to a bound slot.
        let mut nodes: Vec<crate::ir::InsertNodeExpr> = Vec::new();
        let mut edges: Vec<crate::ir::InsertEdgeExpr> = Vec::new();
        let mut var_to_idx: HashMap<String, usize> = HashMap::new();
        loop {
            self.insert_path_expr(&mut nodes, &mut edges, &mut var_to_idx)?;
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        // `INSERT (…) RETURN …`: the created nodes are bound into scope so a
        // following projection can read them. Each node keeps the slot equal to
        // its creation index (the same index `var_to_idx` records), so the tail's
        // `Expr::Prop{slot}` lines up with the seeded row the executor builds. The
        // RETURN path reads the created nodes by slot from the literal plan, so a
        // (rare) constant expression here is not yet supported.
        if self.peek_kw("RETURN")
            || self.peek_kw("ORDER")
            || self.peek_kw("OFFSET")
            || self.peek_kw("LIMIT")
        {
            let Some((nodes, edges)) = Self::lower_insert_literal(&nodes, &edges) else {
                return Err(
                    "an INSERT … RETURN property value must be a literal (evaluate a constant \
                     expression in a plain INSERT, or via FOR/MATCH … INSERT)"
                        .into(),
                );
            };
            self.scope = var_to_idx;
            self.slots = nodes.len();
            let tail = self.query_tail(Plan::Row)?;
            return Ok(Plan::InsertReturn {
                nodes,
                edges,
                tail: Box::new(tail),
            });
        }
        // Keep the constant-literal fast path (`Plan::Insert`) when every value is a
        // literal; otherwise evaluate the constant expressions ONCE over a single
        // empty row (`InsertFrom` has a store at exec time, so `duration()` and a
        // host-wired clock resolve correctly).
        if let Some((nodes, edges)) = Self::lower_insert_literal(&nodes, &edges) {
            return Ok(Plan::Insert { nodes, edges });
        }
        Ok(Plan::InsertFrom {
            input: Box::new(Plan::Row),
            nodes,
            edges,
        })
    }

    /// Lower expression-form INSERT templates back to the constant-literal plan when
    /// every property is an `Expr::Lit` and no node is a bound reference — the fast
    /// path that keeps a plain `INSERT` on `Plan::Insert`. `None` if any value is a
    /// non-literal expression (the caller then evaluates via `InsertFrom`).
    fn lower_insert_literal(
        enodes: &[crate::ir::InsertNodeExpr],
        eedges: &[crate::ir::InsertEdgeExpr],
    ) -> Option<(Vec<crate::ir::InsertNode>, Vec<crate::ir::InsertEdge>)> {
        let lit_props = |props: &[(String, Expr)]| -> Option<Vec<(String, crate::value::Value)>> {
            props
                .iter()
                .map(|(k, e)| match e {
                    Expr::Lit(v) => Some((k.clone(), v.clone())),
                    _ => None,
                })
                .collect()
        };
        let mut nodes = Vec::with_capacity(enodes.len());
        for n in enodes {
            if n.bound.is_some() {
                return None;
            }
            nodes.push(crate::ir::InsertNode {
                labels: n.labels.clone(),
                props: lit_props(&n.props)?,
            });
        }
        let mut edges = Vec::with_capacity(eedges.len());
        for e in eedges {
            edges.push(crate::ir::InsertEdge {
                from: e.from,
                to: e.to,
                etype: e.etype.clone(),
                props: lit_props(&e.props)?,
            });
        }
        Some((nodes, edges))
    }

    /// A row-driven INSERT (`FOR … INSERT (…)` / `MATCH … INSERT (…)`): parse the
    /// comma-separated node/edge templates — each property is an EXPRESSION over the
    /// input row's scope — and wrap `input` in a `Plan::InsertFrom`.
    fn insert_from(&mut self, input: Plan) -> Result<Plan, String> {
        let mut nodes: Vec<crate::ir::InsertNodeExpr> = Vec::new();
        let mut edges: Vec<crate::ir::InsertEdgeExpr> = Vec::new();
        let mut var_to_idx: HashMap<String, usize> = HashMap::new();
        loop {
            self.insert_path_expr(&mut nodes, &mut edges, &mut var_to_idx)?;
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(Plan::InsertFrom {
            input: Box::new(input),
            nodes,
            edges,
        })
    }

    /// One `(a)-[:R]->(b)…` path of a row-driven INSERT (see `insert_from`). Mirrors
    /// `insert_path` but produces expression-valued templates.
    fn insert_path_expr(
        &mut self,
        nodes: &mut Vec<crate::ir::InsertNodeExpr>,
        edges: &mut Vec<crate::ir::InsertEdgeExpr>,
        var_to_idx: &mut HashMap<String, usize>,
    ) -> Result<(), String> {
        let mut prev = self.insert_node_expr(nodes, var_to_idx)?;
        while matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde)) {
            let rel = self.rel(true)?;
            if rel.where_range.is_some() {
                return Err("inline WHERE on an INSERT relationship is not supported".into());
            }
            let next = self.insert_node_expr(nodes, var_to_idx)?;
            let (from, to) = match rel.dir {
                Dir::Out => (prev, next),
                Dir::In => (next, prev),
                Dir::Both => return Err("INSERT requires a directed relationship".into()),
            };
            let etype =
                match rel.etypes.as_slice() {
                    [t] => t.clone(),
                    [] => return Err("INSERT of a relationship requires an edge type".into()),
                    _ => return Err(
                        "INSERT of a relationship requires a single edge type, not a disjunction"
                            .into(),
                    ),
                };
            // Literal inline props lift to `Expr::Lit`; captured expression props
            // (`{w: date('…'), n: a.n}`) are already `Expr`. Together they form the
            // edge template, evaluated per row (a constant folds once).
            let props = rel
                .props
                .into_iter()
                .map(|(k, v)| (k, Expr::Lit(v)))
                .chain(rel.prop_exprs)
                .collect();
            edges.push(crate::ir::InsertEdgeExpr {
                from,
                to,
                etype,
                props,
            });
            prev = next;
        }
        Ok(())
    }

    /// One `(v :Labels {k: <expr>, …})` node of a row-driven INSERT. A property value
    /// may be any expression over the input row's bindings (a literal is lifted to
    /// `Expr::Lit`); a repeated variable re-references the same template.
    fn insert_node_expr(
        &mut self,
        nodes: &mut Vec<crate::ir::InsertNodeExpr>,
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
            while self.eat(&Tok::Amp) {
                labels.push(self.ident()?);
            }
        }
        let props = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.insert_expr_props()?
        } else {
            Vec::new()
        };
        self.expect(&Tok::RParen)?;
        if let Some(v) = &var {
            if let Some(&idx) = var_to_idx.get(v) {
                if !labels.is_empty() || !props.is_empty() {
                    return Err(format!("variable `{v}` is already defined in this INSERT"));
                }
                return Ok(idx);
            }
            // A bare `(v)` naming a variable BOUND by an enclosing MATCH references that
            // matched node — `MATCH (a),(b) INSERT (a)-[:E]->(b)` connects them, it does not
            // create fresh nodes. Labels/properties on a bound reference are not supported.
            if labels.is_empty() && props.is_empty() {
                if let Some(&slot) = self.scope.get(v) {
                    let idx = nodes.len();
                    nodes.push(crate::ir::InsertNodeExpr {
                        labels,
                        props,
                        bound: Some(slot),
                    });
                    var_to_idx.insert(v.clone(), idx);
                    return Ok(idx);
                }
            }
        }
        let idx = nodes.len();
        nodes.push(crate::ir::InsertNodeExpr {
            labels,
            props,
            bound: None,
        });
        if let Some(v) = var {
            var_to_idx.insert(v, idx);
        }
        Ok(idx)
    }

    /// A `{k: <expr>, …}` property map for a row-driven INSERT: literals and full
    /// expressions alike, each returned as an `Expr` (a literal as `Expr::Lit`). The
    /// expressions resolve against the CURRENT scope (the FOR/MATCH bindings).
    fn insert_expr_props(&mut self) -> Result<Vec<(String, Expr)>, String> {
        let (lits, exprs) = self.props()?;
        let mut out: Vec<(String, Expr)> =
            lits.into_iter().map(|(k, v)| (k, Expr::Lit(v))).collect();
        for (k, range) in exprs {
            out.push((k, self.parse_captured_where(range)?));
        }
        Ok(out)
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

    /// Inline property map `{k: v, …}`. A value is a plain LITERAL when
    /// `literal_value` consumes it whole (the next token is a `,` or `}`); anything
    /// else — `{n: a.n}`, `{n: $p + 1}` — is a correlated EXPRESSION whose token span
    /// is CAPTURED here and lowered later (once the element's slot is bound) as the
    /// filter `k = <expr>`, the exact equivalent of an inline `WHERE k = <expr>`. The
    /// two return vectors keep literals (seedable) and expressions apart; a position
    /// that only permits literals (INSERT / _MERGE / CALL config) rejects a non-empty
    /// expression vector.
    fn props(&mut self) -> Result<(Vec<(String, Value)>, PropExprs), String> {
        self.expect(&Tok::LBrace)?;
        let mut lits = Vec::new();
        let mut exprs = Vec::new();
        if self.eat(&Tok::RBrace) {
            return Ok((lits, exprs));
        }
        loop {
            let key = self.ident()?;
            self.expect(&Tok::Colon)?;
            let save = self.pos;
            match self.literal_value() {
                Ok(v) if matches!(self.peek(), Some(Tok::Comma) | Some(Tok::RBrace)) => {
                    lits.push((key, v));
                }
                _ => {
                    self.pos = save;
                    exprs.push((key, self.capture_prop_value()));
                }
            }
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok((lits, exprs))
    }

    /// A property map in a position that permits ONLY literal values (INSERT / _MERGE
    /// / CALL config) — a correlated `{k: <expr>}` is rejected with a context message.
    fn literal_props(&mut self, ctx: &str) -> Result<Vec<(String, Value)>, String> {
        let (lits, exprs) = self.props()?;
        if !exprs.is_empty() {
            return Err(format!(
                "{ctx} property values must be literals, not expressions"
            ));
        }
        Ok(lits)
    }

    /// Capture the token span of a non-literal inline-property VALUE: from the cursor
    /// to the first top-level `,` (next property) or `}` (map close), tracking bracket
    /// depth so a nested `(…)`/`[…]`/`{…}` inside the value does not end it. Parsed as
    /// an expression later by `parse_captured_where`, like an inline WHERE.
    fn capture_prop_value(&mut self) -> (usize, usize) {
        let start = self.pos;
        let mut depth = 0i32;
        while let Some(t) = self.toks.get(self.pos) {
            match t {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket => depth -= 1,
                Tok::RBrace => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                }
                Tok::Comma if depth == 0 => break,
                _ => {}
            }
            self.pos += 1;
        }
        (start, self.pos)
    }

    /// Lower captured inline-property expressions (see `props`) into `k = <expr>`
    /// filters over the element at `slot`, with `scope` carrying every binding so a
    /// correlated value (`(b {n: a.n})`) resolves the outer variable. A no-op when
    /// `exprs` is empty (the common case — no cursor or scope is touched then).
    fn apply_prop_exprs(
        &mut self,
        mut plan: Plan,
        slot: usize,
        exprs: PropExprs,
        scope: &HashMap<String, usize>,
    ) -> Result<Plan, String> {
        if exprs.is_empty() {
            return Ok(plan);
        }
        self.scope = scope.clone();
        for (k, range) in exprs {
            let val = self.parse_captured_where(range)?;
            plan = plan.filter(Expr::Compare {
                op: CompareOp::Eq,
                left: Box::new(Expr::Prop { slot, key: k }),
                right: Box::new(val),
            });
        }
        Ok(plan)
    }

    // A literal property value: number, string, the keyword true/false/null, or a
    // `[...]` list of literal values (used e.g. by `CALL personalized_pagerank(
    // {sourceNodes: ['a', 'b']})`; a list is a first-class stored property value).
    /// Resolve a `$name` parameter to its value. A missing binding carries the
    /// `E_MISSING_PARAMETER` wire code (the FFI routes the prefix) — a supplied-but-
    /// unbound param is a missing parameter, not a syntax error, matching TS.
    fn lookup_param(&self, name: &str) -> Result<Value, String> {
        self.params
            .get(name)
            .cloned()
            .ok_or_else(|| format!("E_MISSING_PARAMETER: unbound parameter `${name}`"))
    }

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
        // A record/map literal `{k: <literal>, …}` (empty `{}` allowed) — a constant
        // record value, so it is seedable in an INSERT / _MERGE / CALL-config
        // position (a field whose value is a non-literal EXPRESSION makes the whole
        // record non-literal, which `props` then routes to the expression path).
        if self.peek() == Some(&Tok::LBrace) {
            self.bump();
            let mut fields: Vec<(GStr, Value)> = Vec::new();
            if !self.eat(&Tok::RBrace) {
                loop {
                    let key = self.ident()?;
                    self.expect(&Tok::Colon)?;
                    fields.push((key.into(), self.literal_value()?));
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBrace)?;
            }
            return Ok(crate::value::make_record(fields));
        }
        // A leading `-` on a numeric literal (`{n: -0.0}`, `-3`): negate the number.
        if self.eat(&Tok::Minus) {
            return match self.bump() {
                Some(Tok::Num(n)) => Ok(Value::Num(-n)),
                other => Err(format!("expected a number after `-`, got {other:?}")),
            };
        }
        match self.bump() {
            Some(Tok::Num(n)) => Ok(Value::Num(n)),
            Some(Tok::Str(s)) => Ok(Value::Str(s.into())),
            // In prepared mode a `$name` is not a literal — return Err so the caller
            // (props) re-parses it via the expression path into an `Expr::Param`.
            Some(Tok::Param(name)) if !self.prepared => self.lookup_param(&name),
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
        if self.is_subpath_group_start() {
            if self.subpath_group_is_quantified() {
                // Unanchored QUANTIFIED subpath group `((x)-[:R]->(y)){n,m} (t)` —
                // synthesize an anonymous seed node (scan every node), matching the TS engine,
                // then chain from it (the group lowers to a var_length hop).
                let from = slots;
                slots += 1;
                let plan =
                    self.extend_chain(Plan::Scan { label: None }, &mut scope, &mut slots, from)?;
                return Ok((plan, scope, slots));
            }
            // A NAMED path may not bind an unquantified subpath group (ISO: a group
            // is a path factor only when quantified) — the TS engine rejects it, so match that
            // rather than binding a lineage the reference engine would not.
            if !self.path_vars.is_empty() {
                return Err(
                    "a named path over an unquantified subpath group is not supported".into(),
                );
            }
            // An UNQUANTIFIED group `(( <pattern> [WHERE p] ))` is just a scoping
            // paren: unwrap the outer `(`, parse the inner pattern inline, apply the
            // group's trailing WHERE over the inner scope, then close the group.
            self.expect(&Tok::LParen)?;
            let plan = self.grouped_subpattern(&mut scope, &mut slots)?;
            return Ok((plan, scope, slots));
        }
        let (var, label, props, where_range, label_expr, prop_exprs) = self.node()?;
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
        seed = self.apply_prop_exprs(seed, from, prop_exprs, &scope)?;
        let plan = self.extend_chain(seed, &mut scope, &mut slots, from)?;
        Ok((plan, scope, slots))
    }

    /// Parse the body of an UNQUANTIFIED subpath group — the outer `(` already
    /// consumed: an inner leading node (labels/props/inline-WHERE all allowed), its
    /// hop tail, and the group's trailing `WHERE` (a predicate over the whole inner
    /// scope), then the closing `)`. The group is a scoping paren, so its bindings
    /// join the enclosing pattern's `scope`/`slots` directly.
    fn grouped_subpattern(
        &mut self,
        scope: &mut HashMap<String, usize>,
        slots: &mut usize,
    ) -> Result<Plan, String> {
        let (var, label, props, where_range, label_expr, prop_exprs) = self.node()?;
        if let Some(v) = var {
            scope.insert(v, *slots);
        }
        let from = *slots;
        *slots += 1;
        let mut seed = node_prop_filters(Plan::Scan { label }, from, props);
        if let Some(le) = label_expr {
            seed = seed.filter(lower_label_expr(&le, from));
        }
        if let Some(r) = where_range {
            self.scope = scope.clone();
            seed = seed.filter(self.parse_captured_where(r)?);
        }
        seed = self.apply_prop_exprs(seed, from, prop_exprs, scope)?;
        let mut plan = self.extend_chain(seed, scope, slots, from)?;
        if self.eat_kw("WHERE") {
            self.scope = scope.clone();
            self.slots = *slots;
            plan = plan.filter(self.bool_pred()?);
        }
        self.expect(&Tok::RParen)?;
        Ok(plan)
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
                // A NESTED subpath group (an inner quantifier inside the outer group)
                // routes to the general nested parser producing a `Plan::NestedGroup`;
                // a flat single-level group stays on `parse_subpath_group`.
                if self.subpath_group_is_nested() {
                    let (p, nf) = self.parse_nested_group(plan, scope, slots, from)?;
                    plan = p;
                    from = nf;
                    continue;
                }
                let g = self.parse_subpath_group()?;
                let k = g.k;
                // The endpoint `(t)` is OPTIONAL — `((x)-[e]->(y)){2}` (anonymous
                // landing) is valid when only the group variables are used.
                let (v2, v2_label, v2_props, v2_where, v2_le) =
                    if matches!(self.peek(), Some(Tok::LParen)) {
                        self.node_plain()?
                    } else {
                        (None, None, Vec::new(), None, None)
                    };
                let node_slot = *slots;
                if let Some(v) = v2 {
                    scope.insert(v, node_slot);
                }
                *slots += 1;
                // An endpoint-only NESTED group `( ()-[:R]->{a,b}() ){1} (t)` — a
                // SINGLE outer repetition — reaches exactly the same endpoints (once
                // each) as one var-length `{a,b}`, so it desugars. With MORE than one
                // outer rep the same endpoint is reached once per rep-DECOMPOSITION (a
                // multiplicity a flat var-length cannot reproduce), so that stays
                // unsupported.
                if let Some((imin, imax)) = g.inner_quant {
                    if g.min != 1 || g.max != 1 {
                        return Err("a multi-repetition nested subpath group is not \
                                    supported yet"
                            .into());
                    }
                    plan = plan.var_length(from, g.dir, &g.etypes, imin, imax, self.path_mode);
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
                // Each NAMED inner variable becomes a GROUP variable — a list column
                // appended after the endpoint. Node variables at unit positions 0..=k
                // (`NodeAt`), edge variables at 0..k (`EdgeAt`); a node group var and
                // an edge group var are tracked so `x[i].prop` resolves. Interleave in
                // position order (node p, edge p, node p+1, …) — the order does not
                // matter to the executor (each carries its position) but keeps slots tidy.
                let mut group_binds: Vec<(crate::ir::GroupPos, usize)> = Vec::new();
                for p in 0..=k as usize {
                    if let Some(n) = g.node_vars[p].clone() {
                        let s = *slots;
                        *slots += 1;
                        scope.insert(n, s);
                        self.group_node_slots.insert(s);
                        group_binds.push((crate::ir::GroupPos::NodeAt(p as u32), s));
                    }
                    if p < k as usize {
                        if let Some(n) = g.edge_vars[p].clone() {
                            let s = *slots;
                            *slots += 1;
                            scope.insert(n, s);
                            self.group_edge_slots.insert(s);
                            group_binds.push((crate::ir::GroupPos::EdgeAt(p as u32), s));
                        }
                    }
                }
                // `min`/`max` are in REPETITIONS; a rep spans `k` hops → the var-length
                // hop bounds are `min*k..=max*k`. No group vars, no per-rep filter, and
                // a single hop → the endpoint-only var_length lowering; otherwise a
                // RepeatGroup (group lists, multi-hop, and/or a per-repetition WHERE).
                let (hop_min, hop_max) = (g.min * k, g.max * k);
                plan = if group_binds.is_empty() && g.per_rep_pred.is_none() && k == 1 {
                    plan.var_length(from, g.dir, &g.etypes, hop_min, hop_max, self.path_mode)
                } else {
                    Plan::RepeatGroup {
                        input: Box::new(plan),
                        from,
                        dir: g.dir,
                        edge_label: g.etypes,
                        min: hop_min,
                        max: hop_max,
                        mode: self.path_mode,
                        endpoint_slot: node_slot,
                        group_binds,
                        k,
                        per_rep_pred: g.per_rep_pred.map(Box::new),
                    }
                };
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
            let rel = self.rel(false)?;
            let quant = self.opt_quantifier()?;
            let (v2, v2_label, v2_props, v2_where, v2_le, v2_prop_exprs) = self.node()?;
            // A relationship variable, inline edge properties, or an inline edge
            // WHERE require binding the edge as a slot (edge at `slots`, node at
            // `slots+1`) so `e.k` can resolve.
            let bind = rel.var.is_some() || !rel.props.is_empty() || rel.where_range.is_some();
            if let Some((min, max)) = quant {
                let node_slot = *slots;
                // A landing variable ALREADY in scope is a repeated pattern variable
                // (`(a)…(a)`, or a `WHERE EXISTS { (a)-[…]->+(b) }` correlated on both
                // ends) — the two positions must be the SAME node, an equality join
                // added after the hop rather than a rebind.
                let repeat_eq = match &v2 {
                    Some(v) => match scope.get(v) {
                        Some(&existing) => Some(existing),
                        None => {
                            scope.insert(v.clone(), node_slot);
                            None
                        }
                    },
                    None => None,
                };
                *slots += 1;
                if bind {
                    // A per-hop edge filter on a var-length hop → a RepeatGroup (k=1)
                    // whose per_rep_pred tests each edge (the edge at scalar slot 1).
                    // Covers inline edge PROPERTIES (`-[:R {k:v}]->{n,m}`) and an inline
                    // edge WHERE (`-[e:R WHERE …]->{n,m}`). The edge variable is bound
                    // only INSIDE the WHERE (at the scalar slot), so a WHERE referencing
                    // an OUTER variable is not yet supported (it fails to resolve).
                    let mut pred: Option<Expr> = None;
                    let and = |p: Option<Expr>, cmp: Expr| {
                        Some(match p {
                            None => cmp,
                            Some(prev) => Expr::And(Box::new(prev), Box::new(cmp)),
                        })
                    };
                    for (k, val) in rel.props {
                        let cmp = Expr::Compare {
                            op: CompareOp::Eq,
                            left: Box::new(Expr::Prop { slot: 1, key: k }),
                            right: Box::new(Expr::Lit(val)),
                        };
                        pred = and(pred, cmp);
                    }
                    if let Some(r) = rel.where_range {
                        let saved_scope = std::mem::take(&mut self.scope);
                        let saved_slots = self.slots;
                        let mut mini: HashMap<String, usize> = HashMap::new();
                        if let Some(ev) = &rel.var {
                            mini.insert(ev.clone(), 1);
                        }
                        // The hop SOURCE is visible to a per-hop WHERE: any outer
                        // variable bound at `from` (the anchor `a` in `(a)-[e WHERE
                        // a.k = …]->{…}`) maps to a dedicated mini-slot (3) that
                        // `rep_pred_ok` fills with the path source. Other outer
                        // variables are still out of reach (they are not on the path).
                        for (name, &sl) in scope.iter() {
                            if sl == from {
                                mini.insert(name.clone(), 3);
                            }
                        }
                        self.scope = mini;
                        self.slots = 4;
                        let w = self.parse_captured_where(r)?;
                        self.scope = saved_scope;
                        self.slots = saved_slots;
                        pred = and(pred, w);
                    }
                    plan = Plan::RepeatGroup {
                        input: Box::new(plan),
                        from,
                        dir: rel.dir,
                        edge_label: rel.etypes,
                        min,
                        max,
                        mode: self.path_mode,
                        endpoint_slot: node_slot,
                        group_binds: Vec::new(),
                        k: 1,
                        per_rep_pred: pred.map(Box::new),
                    };
                } else {
                    // The leading path mode (default TRAIL) selects the reuse semantics.
                    plan = plan.var_length(from, rel.dir, &rel.etypes, min, max, self.path_mode);
                }
                from = node_slot;
                if let Some(existing) = repeat_eq {
                    plan = plan.filter(Expr::Compare {
                        op: CompareOp::Eq,
                        left: Box::new(Expr::Slot(node_slot)),
                        right: Box::new(Expr::Slot(existing)),
                    });
                }
            } else if bind {
                let edge_slot = *slots;
                if let Some(rv) = &rel.var {
                    scope.insert(rv.clone(), edge_slot);
                }
                let node_slot = *slots + 1;
                // A landing variable already in scope is a repeated pattern variable
                // (a self-loop `(u)-[r]->(u)`) — an equality join, not a rebind.
                let repeat_eq = match &v2 {
                    Some(v) => match scope.get(v) {
                        Some(&existing) => Some(existing),
                        None => {
                            scope.insert(v.clone(), node_slot);
                            None
                        }
                    },
                    None => None,
                };
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
                if let Some(existing) = repeat_eq {
                    plan = plan.filter(Expr::Compare {
                        op: CompareOp::Eq,
                        left: Box::new(Expr::Slot(node_slot)),
                        right: Box::new(Expr::Slot(existing)),
                    });
                }
            } else {
                let node_slot = *slots;
                let repeat_eq = match &v2 {
                    Some(v) => match scope.get(v) {
                        Some(&existing) => Some(existing),
                        None => {
                            scope.insert(v.clone(), node_slot);
                            None
                        }
                    },
                    None => None,
                };
                *slots += 1;
                plan = plan.expand(from, rel.dir, &rel.etypes);
                from = node_slot;
                if let Some(existing) = repeat_eq {
                    plan = plan.filter(Expr::Compare {
                        op: CompareOp::Eq,
                        left: Box::new(Expr::Slot(node_slot)),
                        right: Box::new(Expr::Slot(existing)),
                    });
                }
            }
            // The landing node's LABEL constrains it (as the TS engine does) — a filter on the
            // node's label set, since a landing node has no seed `Scan`.
            if let Some(pred) = landing_label_filter(v2_label, v2_le, from) {
                plan = plan.filter(pred);
            }
            // Inline props on the landing node filter it, exactly as a WHERE would.
            // (`from` is now that node's slot in every branch above.)
            plan = node_prop_filters(plan, from, v2_props);
            // A correlated inline-property expression `(b {n: a.n})` — lowered to the
            // filter `b.n = a.n`, the exact equivalent of `(b WHERE b.n = a.n)`, with
            // every outer binding in scope.
            plan = self.apply_prop_exprs(plan, from, v2_prop_exprs, scope)?;
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

    /// At a subpath-group start (`self.pos` on the outer `(`), is the group NESTED —
    /// i.e. does an INNER quantifier (`{n…}` / `*` / `+`) appear inside it, before the
    /// outer `)`? A `{` counts only when followed by a digit (a quantifier, not a
    /// node/edge property map) and outside `[...]` edge brackets. Nested groups route
    /// to `parse_nested_group`; flat groups stay on `parse_subpath_group`.
    fn subpath_group_is_nested(&self) -> bool {
        let mut pdepth = 0i32;
        let mut bdepth = 0i32;
        let mut i = self.pos;
        let mut started = false;
        while let Some(t) = self.toks.get(i) {
            match t {
                Tok::LParen => {
                    pdepth += 1;
                    started = true;
                }
                Tok::RParen => {
                    pdepth -= 1;
                    if started && pdepth == 0 {
                        return false; // outer group closed with no inner quantifier
                    }
                }
                Tok::LBracket => bdepth += 1,
                Tok::RBracket => bdepth -= 1,
                Tok::Star | Tok::Plus if pdepth >= 1 && bdepth == 0 => return true,
                Tok::LBrace
                    if pdepth >= 1
                        && bdepth == 0
                        && matches!(self.toks.get(i + 1), Some(Tok::Num(_))) =>
                {
                    return true
                }
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// At a subpath-group start (`self.pos` on the group's outer `(`), does a
    /// quantifier (`{n,m}` / `*` / `+`) follow the group's matching `)`? A quantified
    /// group is a repeated hop (lowers to var_length); an unquantified one is just a
    /// scoping paren around a sub-pattern. Scans balanced parens to the group close.
    fn subpath_group_is_quantified(&self) -> bool {
        let mut depth = 0usize;
        let mut i = self.pos;
        while let Some(t) = self.toks.get(i) {
            match t {
                Tok::LParen => depth += 1,
                Tok::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return matches!(
                            self.toks.get(i + 1),
                            Some(Tok::LBrace | Tok::Star | Tok::Plus)
                        );
                    }
                }
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// Parse a subpath group `((x)-[e1:R]->(m)…-[ek:R]->(y)){n,m}` — one or more hops
    /// per repetition unit (all hops must share direction and edge type). Returns the
    /// unit's shape and its inner variable NAMES per position (each, when named,
    /// becomes a GROUP variable — a list across repetitions; see `Plan::RepeatGroup`).
    /// Inner-node labels/properties and a bound edge's props stay rejected; a per-rep
    /// `WHERE` is single-hop only (multi-hop needs both edges bound per rep).
    fn parse_subpath_group(&mut self) -> Result<SubpathGroup, String> {
        self.expect(&Tok::LParen)?; // the group's own opening paren
        let bad_inner =
            |n: &PlainNode| n.1.is_some() || n.4.is_some() || !n.2.is_empty() || n.3.is_some();
        let first = self.node_plain()?;
        if bad_inner(&first) {
            return Err(
                "a label/property/WHERE on a subpath-group inner node is not supported yet".into(),
            );
        }
        let mut node_vars: Vec<Option<String>> = vec![first.0.clone()];
        let mut edge_vars: Vec<Option<String>> = Vec::new();
        let mut dir: Option<Dir> = None;
        let mut etypes: Option<Vec<String>> = None;
        // One or more hops, each `-[e:R]->(n)`; all hops must agree on direction and
        // edge type (a mixed-type/-direction unit is not supported).
        loop {
            let rel = self.rel(false)?;
            if !rel.props.is_empty() || rel.where_range.is_some() {
                return Err(
                    "edge properties / a per-hop WHERE on a subpath group are not supported yet"
                        .into(),
                );
            }
            match dir {
                None => dir = Some(rel.dir),
                Some(d) if d == rel.dir => {}
                _ => {
                    return Err("a subpath group with mixed hop directions is not supported".into())
                }
            }
            match &etypes {
                None => etypes = Some(rel.etypes.clone()),
                Some(e) if *e == rel.etypes => {}
                _ => {
                    return Err("a subpath group with mixed hop edge types is not supported".into())
                }
            }
            edge_vars.push(rel.var.clone());
            // An INNER quantifier — `( ()-[:R]->{a,b}() ){c,d}` (a var-length inside
            // the group). Two supported shapes:
            //  - FIXED `{m,m}` (anonymous edge): each rep is exactly `m` hops, so it is
            //    a `k = m` unit — the source and target may be named group variables;
            //    the `m-1` intermediates are anonymous.
            //  - VARIABLE `{a,b}` but ANONYMOUS endpoints: an endpoint-only nested
            //    group, desugared to a var-length by the caller (single outer rep only).
            if let Some((imin, imax)) = self.opt_quantifier()? {
                let n = self.node_plain()?;
                if bad_inner(&n) {
                    return Err(
                        "a label/property/WHERE on a subpath-group inner node is not \
                                supported yet"
                            .into(),
                    );
                }
                if node_vars.len() != 1 || rel.var.is_some() {
                    return Err("a quantified subpath-group body with a bound inner edge \
                                or a preceding hop is not supported yet"
                        .into());
                }
                if imin == imax && imin >= 1 {
                    // `k = m` unit: source at position 0, target at position m, the
                    // intermediates anonymous. The single parsed (anonymous) edge var
                    // stands for all `m` hops.
                    let k = imin;
                    edge_vars = vec![None; k as usize];
                    node_vars = vec![first.0.clone()];
                    for _ in 1..k {
                        node_vars.push(None);
                    }
                    node_vars.push(n.0.clone());
                    self.expect(&Tok::RParen)?;
                    let (min, max) = self
                        .opt_quantifier()?
                        .ok_or("a subpath group requires a `{n,m}` / `*` / `+` quantifier")?;
                    return Ok(SubpathGroup {
                        dir: dir.expect("at least one hop"),
                        etypes: etypes.expect("at least one hop"),
                        min,
                        max,
                        k,
                        node_vars,
                        edge_vars,
                        per_rep_pred: None,
                        inner_quant: None,
                    });
                }
                if first.0.is_some() || n.0.is_some() {
                    return Err("a variable-length subpath-group body with bound inner \
                                variables is not supported yet"
                        .into());
                }
                node_vars.push(n.0.clone());
                self.expect(&Tok::RParen)?;
                let (min, max) = self
                    .opt_quantifier()?
                    .ok_or("a subpath group requires a `{n,m}` / `*` / `+` quantifier")?;
                return Ok(SubpathGroup {
                    dir: dir.expect("at least one hop"),
                    etypes: etypes.expect("at least one hop"),
                    min,
                    max,
                    k: 1,
                    node_vars,
                    edge_vars,
                    per_rep_pred: None,
                    inner_quant: Some((imin, imax)),
                });
            }
            let n = self.node_plain()?;
            if bad_inner(&n) {
                return Err(
                    "a label/property/WHERE on a subpath-group inner node is not supported yet"
                        .into(),
                );
            }
            node_vars.push(n.0.clone());
            if !matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde)) {
                break;
            }
        }
        let k = edge_vars.len() as u32;
        // A PER-REPETITION `WHERE`, evaluated at each rep boundary over the rep's
        // SCALAR variables — node at unit position `p` (0..=k) at mini-scope slot
        // `2p`, edge at position `p` (0..k) at slot `2p+1`. So a single hop is
        // x=0/e=1/y=2; a two-hop unit is x=0/e1=1/m=2/e2=3/y=4. Independent of the
        // group list bindings.
        let per_rep_pred = if self.eat_kw("WHERE") {
            let saved_scope = std::mem::take(&mut self.scope);
            let saved_slots = self.slots;
            let mut mini: HashMap<String, usize> = HashMap::new();
            for (p, nv) in node_vars.iter().enumerate() {
                if let Some(n) = nv {
                    mini.insert(n.clone(), 2 * p);
                }
            }
            for (p, ev) in edge_vars.iter().enumerate() {
                if let Some(n) = ev {
                    mini.insert(n.clone(), 2 * p + 1);
                }
            }
            self.scope = mini;
            self.slots = 2 * k as usize + 1;
            let pred = self.bool_pred()?;
            self.scope = saved_scope;
            self.slots = saved_slots;
            Some(pred)
        } else {
            None
        };
        self.expect(&Tok::RParen)?; // close the group
        let (min, max) = self
            .opt_quantifier()?
            .ok_or("a subpath group requires a `{n,m}` / `*` / `+` quantifier")?;
        Ok(SubpathGroup {
            dir: dir.expect("at least one hop"),
            etypes: etypes.expect("at least one hop"),
            min,
            max,
            k,
            node_vars,
            edge_vars,
            per_rep_pred,
            inner_quant: None,
        })
    }

    /// Parse a NESTED subpath group — the two-level shapes the flat parser rejects:
    ///   family 3: `( ((x)-[e:R]->(y)){a,b} ){c,d} (t)` — a group inside a group;
    ///             x/e/y bind as LIST-OF-LISTS (depth 2).
    ///   family 4: `( (x)-[e:R]->{lo,hi}(y) ){c,d} (t)` — a quantified inner hop;
    ///             x/y bind once per OUTER rep (depth 1), e is a list-of-lists (depth 2).
    /// Builds a `Plan::NestedGroup` over a `GUnit` and binds each named inner variable
    /// as a group list at its nesting depth. Returns the extended plan and the new
    /// `from` (the endpoint slot). `self.pos` is on the outer `(`.
    /// Build a nested inner hop's PER-HOP edge predicate from its inline props
    /// (`{k:v}` → `e.k = v`) and inline `WHERE` — both over the edge at mini-scope
    /// slot 0 (the shape `edge_pred_ok` evaluates). `None` when the hop is unfiltered.
    fn edge_pred_from_rel(&mut self, rel: &Rel) -> Result<Option<Expr>, String> {
        let and = |p: Option<Expr>, c: Expr| {
            Some(match p {
                None => c,
                Some(prev) => Expr::And(Box::new(prev), Box::new(c)),
            })
        };
        let mut pred: Option<Expr> = None;
        for (k, val) in &rel.props {
            pred = and(
                pred,
                Expr::Compare {
                    op: CompareOp::Eq,
                    left: Box::new(Expr::Prop {
                        slot: 0,
                        key: k.clone(),
                    }),
                    right: Box::new(Expr::Lit(val.clone())),
                },
            );
        }
        if let Some(r) = rel.where_range {
            let saved = std::mem::take(&mut self.scope);
            let saved_slots = self.slots;
            let mut mini: HashMap<String, usize> = HashMap::new();
            if let Some(ev) = &rel.var {
                mini.insert(ev.clone(), 0);
            }
            self.scope = mini;
            self.slots = 1;
            let w = self.parse_captured_where(r)?;
            self.scope = saved;
            self.slots = saved_slots;
            pred = and(pred, w);
        }
        Ok(pred)
    }

    fn parse_nested_group(
        &mut self,
        plan: Plan,
        scope: &mut HashMap<String, usize>,
        slots: &mut usize,
        from: usize,
    ) -> Result<(Plan, usize), String> {
        use crate::ir::{GElem, GUnit};
        // A parsed body element before slot assignment (variable NAMES, not slots).
        enum Seg {
            Hop {
                edge: Option<String>,
                target: Option<String>,
                dir: Dir,
                etypes: Vec<String>,
                epred: Option<Expr>,
            },
            Sub {
                inner: Vec<Seg>,
                start: Option<String>,
                min: u32,
                max: u32,
                target: Option<String>,
            },
        }
        let bad_inner =
            |n: &PlainNode| n.1.is_some() || n.4.is_some() || !n.2.is_empty() || n.3.is_some();
        let noinner = "a label/property/WHERE on a nested subpath-group inner node is not \
                       supported yet";
        self.expect(&Tok::LParen)?; // outer group `(`
                                    // family 3 iff the body is itself a group (`( (( …`).
        let family3 = matches!(self.peek(), Some(Tok::LParen))
            && matches!(self.toks.get(self.pos + 1), Some(Tok::LParen));

        // Parse the outer body into (outer-start var, element sequence). Family 3 is a
        // single inner GROUP `((…)…){a,b}` → one Sub; the general form is a hop
        // sequence `<node> (<rel> [quant] <node>)+` where a quantified hop is a Sub.
        let (outer_start, segs): (Option<String>, Vec<Seg>) = if family3 {
            self.expect(&Tok::LParen)?; // inner group `(`
            let x = self.node_plain()?;
            if bad_inner(&x) {
                return Err(noinner.into());
            }
            let mut inner: Vec<Seg> = Vec::new();
            loop {
                let rel = self.rel(false)?;
                let n = self.node_plain()?;
                if bad_inner(&n) {
                    return Err(noinner.into());
                }
                let epred = self.edge_pred_from_rel(&rel)?;
                inner.push(Seg::Hop {
                    edge: rel.var,
                    target: n.0,
                    dir: rel.dir,
                    etypes: rel.etypes,
                    epred,
                });
                if !matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde)) {
                    break;
                }
            }
            self.expect(&Tok::RParen)?; // inner group `)`
            let (imin, imax) = self
                .opt_quantifier()?
                .ok_or("the inner group of a nested subpath needs a quantifier")?;
            (
                None,
                vec![Seg::Sub {
                    inner,
                    start: x.0,
                    min: imin,
                    max: imax,
                    target: None,
                }],
            )
        } else {
            let start = self.node_plain()?;
            if bad_inner(&start) {
                return Err(noinner.into());
            }
            let mut segs: Vec<Seg> = Vec::new();
            loop {
                let rel = self.rel(false)?;
                let epred = self.edge_pred_from_rel(&rel)?;
                if let Some((imin, imax)) = self.opt_quantifier()? {
                    // A quantified hop `-[e]->{lo,hi}` — a Sub with a bare single-hop inner.
                    let n = self.node_plain()?;
                    if bad_inner(&n) {
                        return Err(noinner.into());
                    }
                    segs.push(Seg::Sub {
                        inner: vec![Seg::Hop {
                            edge: rel.var,
                            target: None,
                            dir: rel.dir,
                            etypes: rel.etypes,
                            epred,
                        }],
                        start: None,
                        min: imin,
                        max: imax,
                        target: n.0,
                    });
                } else {
                    let n = self.node_plain()?;
                    if bad_inner(&n) {
                        return Err(noinner.into());
                    }
                    segs.push(Seg::Hop {
                        edge: rel.var,
                        target: n.0,
                        dir: rel.dir,
                        etypes: rel.etypes,
                        epred,
                    });
                }
                if !matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde)) {
                    break;
                }
            }
            (start.0, segs)
        };

        // An optional PER-REP `WHERE` (after the body, before the outer `)`), then the
        // outer `)` and the outer quantifier.
        let perrep_range = if self.eat_kw("WHERE") {
            Some(self.capture_inline_where())
        } else {
            None
        };
        self.expect(&Tok::RParen)?; // outer group `)`
        let (omin, omax) = self
            .opt_quantifier()?
            .ok_or("a nested subpath group needs an outer quantifier")?;

        // The endpoint `(t)` (optional). The executor's output columns are `[input…,
        // endpoint, binds…]`, so the endpoint slot comes FIRST, then one bind slot per
        // named group variable.
        let (tvar, t_label, t_props, t_where, t_le) = if matches!(self.peek(), Some(Tok::LParen)) {
            self.node_plain()?
        } else {
            (None, None, Vec::new(), None, None)
        };
        let node_slot = *slots;
        *slots += 1;
        if let Some(v) = tvar {
            scope.insert(v, node_slot);
        }

        // Assign a slot to each NAMED group variable (endpoint-first already done). A
        // variable's list-nesting depth = its enclosing quantifiers: 1 for an
        // outer-level variable, 2 for one inside a Sub. `assign` records the slot into
        // `scope`, the node/edge sets, `group_var_depth`, and `bind_slots` (ascending).
        let mut bind_slots: Vec<usize> = Vec::new();
        let mut assign =
            |this: &mut Self, name: &Option<String>, is_edge: bool, depth: u8| -> Option<usize> {
                let n = name.clone()?;
                let s = *slots;
                *slots += 1;
                scope.insert(n, s);
                if is_edge {
                    this.group_edge_slots.insert(s);
                } else {
                    this.group_node_slots.insert(s);
                }
                this.group_var_depth.insert(s, depth);
                bind_slots.push(s);
                Some(s)
            };
        let outer_start_slot = assign(self, &outer_start, false, 1);

        // Build the GUnit from the skeleton, assigning slots as we go.
        let mut elems: Vec<GElem> = Vec::with_capacity(segs.len());
        for seg in &segs {
            match seg {
                Seg::Hop {
                    edge,
                    target,
                    dir,
                    etypes,
                    epred,
                } => {
                    let eslot = assign(self, edge, true, 1);
                    let tslot = assign(self, target, false, 1);
                    elems.push(GElem::Hop {
                        dir: *dir,
                        etypes: etypes.clone(),
                        edge_slot: eslot,
                        target_slot: tslot,
                        edge_pred: epred.clone().map(Box::new),
                    });
                }
                Seg::Sub {
                    inner,
                    start,
                    min,
                    max,
                    target,
                } => {
                    let sub_start = assign(self, start, false, 2);
                    let mut inner_elems: Vec<GElem> = Vec::with_capacity(inner.len());
                    for h in inner {
                        let Seg::Hop {
                            edge,
                            target,
                            dir,
                            etypes,
                            epred,
                        } = h
                        else {
                            unreachable!("a Sub's inner is flat (hops only)")
                        };
                        let eslot = assign(self, edge, true, 2);
                        let tslot = assign(self, target, false, 2);
                        inner_elems.push(GElem::Hop {
                            dir: *dir,
                            etypes: etypes.clone(),
                            edge_slot: eslot,
                            target_slot: tslot,
                            edge_pred: epred.clone().map(Box::new),
                        });
                    }
                    let sub_target = assign(self, target, false, 1);
                    elems.push(GElem::Sub {
                        unit: Box::new(GUnit {
                            start_slot: sub_start,
                            elems: inner_elems,
                        }),
                        min: *min,
                        max: *max,
                        target_slot: sub_target,
                    });
                }
            }
        }
        let unit = GUnit {
            start_slot: outer_start_slot,
            elems,
        };

        // The PER-REP `WHERE` sees each variable ONE nesting level shallower (the
        // per-rep view). Parse it with the group depths temporarily decremented.
        let per_rep_pred = if let Some(r) = perrep_range {
            let saved: Vec<(usize, u8)> = bind_slots
                .iter()
                .map(|&s| (s, self.group_var_depth[&s]))
                .collect();
            for &s in &bind_slots {
                let d = self.group_var_depth[&s];
                self.group_var_depth.insert(s, d.saturating_sub(1));
            }
            self.scope = scope.clone();
            let pred = self.parse_captured_where(r)?;
            for (s, d) in saved {
                self.group_var_depth.insert(s, d);
            }
            Some(pred)
        } else {
            None
        };

        let mut plan = Plan::NestedGroup {
            input: Box::new(plan),
            from,
            unit,
            min: omin,
            max: omax,
            mode: self.path_mode,
            endpoint_slot: node_slot,
            bind_slots,
            per_rep_pred: per_rep_pred.map(Box::new),
        };
        if let Some(pred) = landing_label_filter(t_label, t_le, node_slot) {
            plan = plan.filter(pred);
        }
        plan = node_prop_filters(plan, node_slot, t_props);
        if let Some(r) = t_where {
            self.scope = scope.clone();
            plan = plan.filter(self.parse_captured_where(r)?);
        }
        Ok((plan, node_slot))
    }

    /// `OPTIONAL MATCH (a)-[:R]->(x)` — a LEFT-OUTER single hop from a bound `a`. If
    /// `a` has no matching neighbour, the row is kept with `x` NULL. Single-hop,
    /// node-only, no bound edge; an inner `WHERE` is rejected (it filters the optional
    /// match, which is not yet modelled) rather than mis-applied as a top-level filter.
    fn optional_match(&mut self, plan: Plan) -> Result<Plan, String> {
        if !self.eat_kw("MATCH") {
            return Err("expected MATCH after OPTIONAL".into());
        }
        // Parse the leading node MANUALLY so its inline props may be EXPRESSIONS
        // (`{name: name}` correlates on a bound variable) — `node()` only takes
        // literal props. A FRESH variable with a label and NO following relationship is
        // a left-outer correlated SCAN (`Plan::OptionalScan`, the FOR-driven form); a
        // BOUND variable followed by a hop is the left-outer expand below.
        self.expect(&Tok::LParen)?;
        let var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident()?)
        } else {
            None
        };
        let label = if self.eat(&Tok::Colon) {
            Some(self.ident()?)
        } else {
            None
        };
        let mut filters: Vec<(String, Expr)> = Vec::new();
        if matches!(self.peek(), Some(Tok::LBrace)) {
            self.expect(&Tok::LBrace)?;
            if !self.eat(&Tok::RBrace) {
                loop {
                    let key = self.ident()?;
                    self.expect(&Tok::Colon)?;
                    filters.push((key, self.expr()?));
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RBrace)?;
            }
        }
        let inline_where = self.eat_kw("WHERE");
        self.expect(&Tok::RParen)?;
        let has_rel = matches!(self.peek(), Some(Tok::Minus | Tok::LArrow | Tok::Tilde));
        let bound = var.as_ref().and_then(|v| self.scope.get(v)).copied();

        // Fresh-variable single node (no rel) → a left-outer correlated scan.
        if bound.is_none() && !has_rel {
            if inline_where {
                return Err("inline WHERE inside OPTIONAL MATCH is not supported yet".into());
            }
            let node_slot = self.slots;
            if let Some(v) = var {
                self.scope.insert(v, node_slot);
            }
            self.slots += 1;
            return Ok(Plan::OptionalScan {
                input: Box::new(plan),
                label,
                filters,
                node_slot,
            });
        }

        // Otherwise the bound-variable left-outer HOP form.
        let Some(v) = var else {
            return Err("OPTIONAL MATCH must start from a bound variable".into());
        };
        if inline_where {
            return Err("inline WHERE inside OPTIONAL MATCH is not supported yet".into());
        }
        if label.is_some() {
            return Err(format!(
                "bound variable `{v}` cannot be re-labeled in OPTIONAL MATCH"
            ));
        }
        if !filters.is_empty() {
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
        let rel = self.rel(false)?;
        if !rel.props.is_empty() || rel.where_range.is_some() {
            return Err(
                "edge properties / an inline edge WHERE on OPTIONAL MATCH are not supported".into(),
            );
        }
        let (v2, _lbl2, v2_props, v2_where, _v2_le) = self.node_plain()?;
        if !v2_props.is_empty() || v2_where.is_some() {
            return Err(
                "inline properties on the OPTIONAL MATCH landing node are not supported; use WHERE"
                    .into(),
            );
        }
        // A bound edge variable binds the edge at `slots` and the landing node at
        // `slots+1` (matching a plain bound-edge hop); otherwise just the node.
        let bind_edge = rel.var.is_some();
        if bind_edge {
            let edge_slot = self.slots;
            if let Some(rv) = &rel.var {
                self.scope.insert(rv.clone(), edge_slot);
            }
            self.slots += 1;
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
            bind_edge,
        })
    }

    /// A continuing `MATCH` after `WITH`: it must start from a variable already
    /// carried into scope and extends the working table from that node (rather
    /// than scanning afresh). A fresh/disconnected subsequent pattern — one whose
    /// first node is unbound — is not supported in this subset.
    fn match_continue(&mut self, plan: Plan) -> Result<Plan, String> {
        let (var, label, props, start_where, label_expr) = self.node_plain()?;
        // A continuing MATCH whose first node is a FRESH (unbound) variable is a new
        // INDEPENDENT pattern, cross-joined with the working table (`MATCH (p:P) MATCH
        // (q:Q) …`) — parsed in its own slot space, then joined on any shared variable.
        let bound_start = var.as_ref().and_then(|v| self.scope.get(v)).copied();
        if bound_start.is_none() {
            let mut sub_scope: HashMap<String, usize> = HashMap::new();
            if let Some(v) = &var {
                sub_scope.insert(v.clone(), 0);
            }
            let mut sub_slots = 1usize;
            let mut seed = node_prop_filters(Plan::Scan { label }, 0, props);
            if let Some(le) = label_expr {
                seed = seed.filter(lower_label_expr(&le, 0));
            }
            if let Some(r) = start_where {
                self.scope = sub_scope.clone();
                seed = seed.filter(self.parse_captured_where(r)?);
            }
            let p2 = self.extend_chain(seed, &mut sub_scope, &mut sub_slots, 0)?;
            let width = self.slots;
            let on: Vec<(usize, usize)> = sub_scope
                .iter()
                .filter_map(|(v, &r)| self.scope.get(v).map(|&l| (l, r)))
                .collect();
            let mut plan = Plan::join(plan, p2, on);
            for (v, &r) in &sub_scope {
                self.scope.entry(v.clone()).or_insert(width + r);
            }
            self.slots = width + sub_slots;
            if self.eat_kw("WHERE") {
                plan = plan.filter(self.bool_pred()?);
            }
            return Ok(plan);
        }
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
            plan = plan.filter(self.bool_pred()?);
        }
        Ok(plan)
    }

    /// A `WITH` boundary: project/aggregate the working table (exactly as `RETURN`
    /// would), then rebind scope so the carried output columns are a fresh slot
    /// space (`name -> column index`) for the following part. `ORDER BY/SKIP/LIMIT`
    /// ride the projection; a trailing `WHERE` is a post-projection (HAVING)
    /// filter, matching the TS engine's `WITH … WHERE`.
    /// `LET name = expr [, name = expr]*` — the ISO additive-binding clause: ADD the
    /// new bindings to the working table, carrying every existing binding forward
    /// (unlike WITH, which projects only its listed items). Distinct from the `LET …
    /// IN … END` *expression* (parsed in `expr`), which this never reaches: here the
    /// binding value runs to the next clause, with no `IN`/`END`.
    fn let_clause(&mut self, plan: Plan) -> Result<Plan, String> {
        // Bindings are SEQUENTIAL, left-to-right (ISO): a later binding sees the
        // earlier ones in the SAME LET (`LET x = p.age, y = x + 1`). Each parsed
        // binding is pushed onto the inline-`lets` stack so a following reference
        // substitutes it (`y`'s expr becomes `p.age + 1`); the stack is truncated
        // after the loop, since the LET-clause names become COLUMNS, not inline locals.
        let let_base = self.lets.len();
        let mut new_binds: Vec<(String, Expr)> = Vec::new();
        loop {
            let name = self.ident()?;
            self.expect(&Tok::Eq)?;
            let e = self.expr()?;
            self.lets.push((name.clone(), e.clone()));
            new_binds.push((name, e));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.lets.truncate(let_base);
        // Pass every existing binding through, in slot order (a HashMap iteration is
        // unordered — the slot index is the stable key), then append the new ones.
        let mut existing: Vec<(usize, String)> =
            self.scope.iter().map(|(n, &s)| (s, n.clone())).collect();
        existing.sort();
        let mut items: Vec<RetItem> = existing
            .iter()
            .map(|(s, n)| RetItem::Key(n.clone(), Expr::Slot(*s)))
            .collect();
        for (n, e) in new_binds {
            items.push(RetItem::Key(n, e));
        }
        let (plan, out_names) = apply_items(plan, &items);
        let mut scope = HashMap::new();
        for (i, name) in out_names.iter().enumerate() {
            scope.insert(name.clone(), i);
        }
        self.scope = scope;
        self.slots = out_names.len();
        Ok(plan)
    }

    /// `FOR <var> IN <list> [WITH ORDINALITY|OFFSET <ord>]` — ISO list unwind. The
    /// `FOR` keyword is already consumed. Binds `var` (and the optional ordinal) into
    /// scope and appends a `Plan::Unwind`. `WITH ORDINALITY`/`WITH OFFSET` here is
    /// part of the FOR (not the separate WITH clause), so only that spelling is eaten.
    fn for_clause(&mut self, plan: Plan) -> Result<Plan, String> {
        let var = self.ident()?;
        if !self.eat_kw("IN") {
            return Err("expected IN after the FOR variable".into());
        }
        let list = self.expr()?;
        let var_slot = self.slots;
        self.slots += 1;
        self.scope.insert(var, var_slot);
        let ordinal = if self.peek_kw("WITH")
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s))
                if s.eq_ignore_ascii_case("ORDINALITY") || s.eq_ignore_ascii_case("OFFSET"))
        {
            self.eat_kw("WITH");
            let one_based = self.eat_kw("ORDINALITY");
            if !one_based {
                self.eat_kw("OFFSET");
            }
            let ord_var = self.ident()?;
            let ord_slot = self.slots;
            self.slots += 1;
            self.scope.insert(ord_var, ord_slot);
            Some((ord_slot, one_based))
        } else {
            None
        };
        Ok(Plan::Unwind {
            input: Box::new(plan),
            list: Box::new(list),
            var_slot,
            ordinal,
        })
    }

    fn with_clause(&mut self, plan: Plan) -> Result<Plan, String> {
        let distinct = self.eat_kw("DISTINCT");
        let items = self.return_items()?;
        // Carry group-variable list typing across the WITH boundary: `WITH e AS hops`
        // (where `e` is an edge group list) must keep `hops` an EDGE list, so a later
        // `hops[i].amt` still resolves the edge property. A bare `Slot` item re-projects
        // to output column `i` (item order, no aggregate), so map the old
        // node-/edge-list slots onto their new columns.
        if !items.iter().any(RetItem::has_agg) {
            let (mut new_node, mut new_edge) = (HashSet::new(), HashSet::new());
            for (i, it) in items.iter().enumerate() {
                if let RetItem::Key(_, Expr::Slot(s)) = it {
                    if self.group_node_slots.contains(s) {
                        new_node.insert(i);
                    }
                    if self.group_edge_slots.contains(s) {
                        new_edge.insert(i);
                    }
                }
            }
            self.group_node_slots = new_node;
            self.group_edge_slots = new_edge;
        } else {
            self.group_node_slots.clear();
            self.group_edge_slots.clear();
        }
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
        // `OFFSET` is the ISO spelling of `SKIP` — a synonym here (the TS engine accepts both).
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
            plan = plan.filter(self.bool_pred()?);
        }
        Ok(plan)
    }

    /// An inline correlated subquery `CALL (scope) { MATCH … [WHERE] RETURN … }`.
    /// The subquery imports only the named `scope` variables, continues its pattern
    /// from one of them, and its `RETURN` columns are appended to each outer row
    /// (a lateral join). The named-procedure form (`CALL name(cfg) YIELD …`) is
    /// deferred to the algorithms phase — its catalog is those procedures.
    fn call_inline(&mut self, plan: Plan, optional: bool) -> Result<Plan, String> {
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
            if optional {
                return Err("OPTIONAL CALL () { … } (uncorrelated) is not supported".into());
            }
            return self.call_inline_uncorrelated(plan, outer_width);
        }
        for v in &scope_vars {
            if !self.scope.contains_key(v) {
                return Err(format!("CALL scope variable `{v}` is not bound"));
            }
        }

        self.expect(&Tok::LBrace)?;
        // Parse the first arm, then any `UNION`/`EXCEPT`/`INTERSECT` tail (each a full
        // MATCH…RETURN arm). call_arm leaves `self.scope` restored to the outer scope,
        // so each following arm re-imports the scope variables cleanly.
        let (body, items) = self.call_arm(&scope_vars, outer_width)?;
        let mut parts_raw: Vec<(crate::ir::CombineOp, bool, Plan, Vec<RetItem>)> = Vec::new();
        loop {
            let op = if self.eat_kw("UNION") {
                crate::ir::CombineOp::Union
            } else if self.eat_kw("EXCEPT") {
                crate::ir::CombineOp::Except
            } else if self.eat_kw("INTERSECT") {
                crate::ir::CombineOp::Intersect
            } else {
                break;
            };
            let all = self.eat_kw("ALL");
            let (b, it) = self.call_arm(&scope_vars, outer_width)?;
            parts_raw.push((op, all, b, it));
        }
        self.expect(&Tok::RBrace)?;

        // A single scalar-aggregate RETURN (no set-op tail) is a correlated aggregate
        // subquery: for each outer row, reduce the body's sub-matches. `COUNT` counts them
        // (0 if none — the outer row survives); `SUM`/`AVG`/`MIN`/`MAX` reduce the argument
        // (NULL over an empty match — SQL aggregate-of-nothing). Appended via a projection
        // that passes the outer columns through and adds the aggregate.
        if parts_raw.is_empty() && items.len() == 1 {
            if let RetItem::Agg(agg) = &items[0] {
                use crate::ir::AggFn;
                let subquery = if agg.distinct {
                    None
                } else {
                    match agg.func {
                        AggFn::Count => Some(Expr::CountSubquery {
                            body: Box::new(body.clone()),
                            outer_width,
                        }),
                        AggFn::Sum | AggFn::Avg | AggFn::Min | AggFn::Max => {
                            // The aggregate must have an argument (`sum(*)` is not a thing).
                            agg.arg.clone().map(|arg| Expr::AggSubquery {
                                body: Box::new(body.clone()),
                                scalar: Box::new(arg),
                                func: agg.func,
                                outer_width,
                            })
                        }
                        _ => None,
                    }
                };
                if let Some(sub) = subquery {
                    let name = agg.name.clone();
                    let mut by_slot: Vec<Option<String>> = vec![None; outer_width];
                    for (n, &s) in self.scope.iter() {
                        if s < outer_width {
                            by_slot[s] = Some(n.clone());
                        }
                    }
                    let mut proj: Vec<(String, Expr)> = by_slot
                        .into_iter()
                        .enumerate()
                        .map(|(i, nm)| (nm.unwrap_or_else(|| format!("col{i}")), Expr::Slot(i)))
                        .collect();
                    proj.push((name.clone(), sub));
                    self.scope.insert(name, outer_width);
                    self.slots = outer_width + 1;
                    return Ok(plan.project(proj));
                }
            }
        }
        let yields = call_items_to_yields(items)?;
        let parts: Vec<crate::ir::CallPart> = parts_raw
            .into_iter()
            .map(|(op, all, b, it)| {
                Ok(crate::ir::CallPart {
                    op,
                    all,
                    body: b,
                    yields: call_items_to_yields(it)?,
                })
            })
            .collect::<Result<_, String>>()?;

        // Bind the (first arm's) yields as the outer scope's new trailing columns; the
        // subquery's internal variables do not survive.
        for (i, (name, _)) in yields.iter().enumerate() {
            self.scope.insert(name.clone(), outer_width + i);
        }
        self.slots = outer_width + yields.len();
        Ok(Plan::CallInline {
            input: Box::new(plan),
            body: Box::new(body),
            yields,
            outer_width,
            optional,
            parts,
        })
    }

    /// Parse ONE arm of an inline correlated CALL body: `MATCH <pattern> [WHERE …]
    /// RETURN <items>`. The pattern may start from a declared scope variable (an Expand
    /// rooted at it) or from a FRESH `(x:Label)` node (a Scan cross-joined with the
    /// prov-seed — every seed row × every matching node). The body is provenance-tagged:
    /// slot `outer_width` is RESERVED for the per-outer-row id (the layout the exec
    /// seeds), so body variables append from `outer_width + 1`. Returns the arm's body
    /// plan (before projection) and its raw RETURN items; `self.scope` is left restored
    /// to the outer scope.
    fn call_arm(
        &mut self,
        scope_vars: &[String],
        outer_width: usize,
    ) -> Result<(Plan, Vec<RetItem>), String> {
        if !self.eat_kw("MATCH") {
            return Err("a CALL subquery must begin with MATCH".into());
        }
        // Import the scope variables at their OUTER slots; the prov id sits at
        // `outer_width`, so fresh body variables start at `outer_width + 1`.
        let mut sub_scope: HashMap<String, usize> = scope_vars
            .iter()
            .map(|s| (s.clone(), self.scope[s]))
            .collect();
        let mut sub_slots = outer_width + 1;
        let (var, label, props, start_where, le) = self.node_plain()?;
        if start_where.is_some() {
            return Err("inline WHERE on a CALL subquery start node is not supported".into());
        }
        // A start node naming a declared scope variable roots an Expand from it (and may
        // not be re-labeled or re-constrained); anything else is a fresh correlated Scan.
        let scope_root = match &var {
            Some(v) if scope_vars.contains(v) => {
                if label.is_some() || le.is_some() {
                    return Err(format!(
                        "scope variable `{v}` cannot be re-labeled inside a CALL subquery"
                    ));
                }
                if !props.is_empty() {
                    return Err(format!(
                        "scope variable `{v}` cannot be re-constrained with inline properties \
                         inside a CALL subquery; use WHERE"
                    ));
                }
                true
            }
            _ => false,
        };
        let (mut body, from) = if scope_root {
            (Plan::Row, self.scope[var.as_ref().unwrap()])
        } else {
            // Fresh scan: the new node lands at `outer_width + 1`; when pull_body reaches
            // this leaf the prov-seed carries exactly `outer_width + 1` columns, so the
            // Scan cross-joins each seed row with every matching node.
            if le.is_some() {
                return Err(
                    "a compound label on a CALL fresh-scan start node is not supported".into(),
                );
            }
            let node_slot = sub_slots; // outer_width + 1
            if let Some(v) = &var {
                sub_scope.insert(v.clone(), node_slot);
            }
            sub_slots += 1;
            let b = node_prop_filters(Plan::Scan { label }, node_slot, props);
            (b, node_slot)
        };
        body = self.extend_chain(body, &mut sub_scope, &mut sub_slots, from)?;

        let outer_scope = std::mem::replace(&mut self.scope, sub_scope);
        self.slots = sub_slots;
        if self.eat_kw("WHERE") {
            body = body.filter(self.bool_pred()?);
        }
        if !self.eat_kw("RETURN") {
            self.scope = outer_scope;
            return Err("a CALL subquery needs a RETURN".into());
        }
        let items = self.return_items()?;
        self.scope = outer_scope;
        Ok((body, items))
    }

    /// `CALL () { <subquery> }` — an UNCORRELATED (empty-scope) inline subquery. The
    /// body is an INDEPENDENT query (a fresh MATCH … RETURN, aggregates allowed) run
    /// once; its rows CROSS-JOIN the outer working table, appending the yielded
    /// columns. The `()` scope imports nothing, so a reference to an outer variable is
    /// ISOLATED — resolved to NULL (matching the TS engine's scope isolation), which lets a
    /// body like `WHERE c = a` compile and simply match nothing. The `CALL (` and
    /// `)` are already consumed; the outer scope is intact in `self`.
    fn call_inline_uncorrelated(&mut self, plan: Plan, outer_width: usize) -> Result<Plan, String> {
        self.expect(&Tok::LBrace)?;
        if !self.eat_kw("MATCH") {
            return Err("a CALL subquery must begin with MATCH".into());
        }
        // Parse the body in a FRESH slot space, with outer variables isolated to NULL
        // (a LET-style local shadowing any outer binding). A parse error discards the
        // whole parser, so the saved state need not be restored on the error paths.
        let outer_scope = std::mem::take(&mut self.scope);
        let let_base = self.lets.len();
        for name in outer_scope.keys() {
            self.lets.push((name.clone(), Expr::Lit(Value::Null)));
        }
        self.slots = 0;
        let body = self.match_body()?;
        if !self.eat_kw("RETURN") {
            return Err("a CALL subquery needs a RETURN".into());
        }
        let items = self.return_items()?;
        let (mut body_out, out_names) = apply_items(body, &items);

        // A `UNION`/`EXCEPT`/`INTERSECT` tail: each arm is an INDEPENDENT global query
        // (fresh slot space, same outer-var isolation), combined with the ordinary
        // top-level set operator. The whole combined body is a single GLOBAL result set
        // that then cross-joins the outer table — no per-outer-row grouping needed
        // (nothing correlates).
        loop {
            let op = if self.eat_kw("UNION") {
                crate::ir::CombineOp::Union
            } else if self.eat_kw("EXCEPT") {
                crate::ir::CombineOp::Except
            } else if self.eat_kw("INTERSECT") {
                crate::ir::CombineOp::Intersect
            } else {
                break;
            };
            let all = self.eat_kw("ALL");
            // A fresh arm scope (the outer-var isolation lets stay in place).
            self.scope = HashMap::new();
            self.slots = 0;
            self.path_vars = HashSet::new();
            self.group_node_slots = HashSet::new();
            self.group_edge_slots = HashSet::new();
            self.group_var_depth = HashMap::new();
            if !self.eat_kw("MATCH") {
                return Err("a CALL subquery must begin with MATCH".into());
            }
            let arm_body = self.match_body()?;
            if !self.eat_kw("RETURN") {
                return Err("a CALL subquery needs a RETURN".into());
            }
            let arm_items = self.return_items()?;
            let (arm_out, _) = apply_items(arm_body, &arm_items);
            body_out = Plan::Union {
                left: Box::new(body_out),
                right: Box::new(arm_out),
                all,
                op,
            };
        }
        self.expect(&Tok::RBrace)?;

        // Restore the outer scope; the yields append as its new trailing columns
        // (cross-join layout: right slot `j` lands at `outer_width + j`).
        self.lets.truncate(let_base);
        self.scope = outer_scope;
        for (i, name) in out_names.iter().enumerate() {
            self.scope.insert(name.clone(), outer_width + i);
        }
        self.slots = outer_width + out_names.len();
        Ok(Plan::join(plan, body_out, Vec::new()))
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
            self.literal_props("a CALL config")?
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

    /// `node()` for the contexts that do NOT support a correlated inline-property
    /// expression `{k: <expr>}` — everything except a pattern's anchor and hop-landing
    /// nodes. A captured expression here is an explicit error, never a silent drop.
    fn node_plain(&mut self) -> Result<PlainNode, String> {
        let (v, l, p, w, le, exprs) = self.node()?;
        if !exprs.is_empty() {
            return Err(
                "an inline property expression is only supported on a MATCH pattern node; use an inline WHERE here".into(),
            );
        }
        Ok((v, l, p, w, le))
    }

    // node := '(' [var] [':' Label] ')'
    fn node(&mut self) -> Result<ParsedNode, String> {
        self.expect(&Tok::LParen)?;
        // An identifier here is the node variable; the label (if any) follows ':'.
        let var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident_binding()?)
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
        let (props, prop_exprs) = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.props()?
        } else {
            (Vec::new(), Vec::new())
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
        Ok((var, label, props, where_range, label_expr, prop_exprs))
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
        Ok(LabelExpr::Label(self.ident_binding()?))
    }

    // rel := '-' '[' ':' R ']' '->'   (out)
    //      | '-' '[' ':' R ']' '-'    (both)
    //      | '<-' '[' ':' R ']' '-'   (in)
    // rel := ('-' | '~' | '<-') '[' [var] ':' Type [ '{' props '}' ] ']' ('->' | '-' | '~')
    // Captures an optional relationship VARIABLE and inline edge PROPERTIES. `~` is
    // the undirected delimiter: like `-`, it carries NO direction, so `~[...]~`
    // (and any `-`/`~` mix) is `Dir::Both`, exactly as the TS engine resolves it.
    fn rel(&mut self, insert_ctx: bool) -> Result<Rel, String> {
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
        // matching the TS engine's bracketed untyped relationship. (The TS engine's BARE `-->` has
        // different semantics — it matches nothing — so it is deliberately NOT
        // accepted here, to avoid a silent result divergence.)
        // `:Type` with an optional `|`-disjunction (`:A|B|C`) — an edge matches if
        // its type is ANY of them. Empty = untyped (any type).
        // A leading `!` NEGATES the type set (`:!T` / `:!(A|B|C)`) — the hop matches
        // any edge whose type is NOT one of the named ones. Encoded as a "!" sentinel
        // first element, which `want_etypes` resolves to the complement id set.
        let mut etypes = Vec::new();
        if self.eat(&Tok::Colon) {
            if self.eat(&Tok::Bang) {
                etypes.push("!".to_string());
                if self.eat(&Tok::LParen) {
                    etypes.push(self.ident()?);
                    while self.eat(&Tok::Pipe) {
                        etypes.push(self.ident()?);
                    }
                    self.expect(&Tok::RParen)?;
                } else {
                    etypes.push(self.ident()?);
                }
            } else {
                etypes.push(self.ident()?);
                while self.eat(&Tok::Pipe) {
                    etypes.push(self.ident()?);
                }
            }
        }
        let mut prop_exprs = Vec::new();
        let props = if matches!(self.peek(), Some(Tok::LBrace)) {
            let (lits, exprs) = self.props()?;
            if !exprs.is_empty() {
                if insert_ctx {
                    // An INSERT edge CREATES the relationship, so a property expression
                    // (`-[:R {w: date('…'), n: a.n}]->`) is evaluated and stored — parse
                    // each captured span against the current (MATCH/FOR) scope.
                    for (k, range) in exprs {
                        prop_exprs.push((k, self.parse_captured_where(range)?));
                    }
                } else {
                    // A MATCH edge property is a FILTER, not a creation — an expression
                    // there uses an inline edge `WHERE` (the correlated-edge lowering).
                    return Err(
                        "an inline edge property expression is not supported; use an inline WHERE on the edge".into(),
                    );
                }
            }
            lits
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
            prop_exprs,
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
                        null_on_empty: false,
                        numeric_only: false,
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
        // One nesting level per `expr` (re)entry — every `(…)`, `[…]`, function arg,
        // `LET` body, `CASE`, and `IN`-list funnels back through here.
        self.nest(|p| p.or_expr())
    }

    /// Parse an expression that sits in a BOOLEAN context (`WHERE` / `FILTER` /
    /// `HAVING` / per-repetition / edge / `EXISTS`-body predicate) and reject a
    /// statically-non-boolean value at parse time (see [`check_bool_ctx`]).
    fn bool_pred(&mut self) -> Result<Expr, String> {
        let e = self.expr()?;
        check_bool_ctx(&e, true)?;
        Ok(e)
    }

    fn or_expr(&mut self) -> Result<Expr, String> {
        // OR and XOR share one left-associative precedence level (ISO), above AND.
        // Binary left-nesting here is equivalent to the TS engine's flatten-same/nest-on-
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
            // `NOT NOT …` recurses here (not through `expr`), so guard the re-entry.
            Ok(Expr::Not(Box::new(self.nest(|p| p.not_expr())?)))
        } else {
            self.cmp_expr()
        }
    }

    fn cmp_expr(&mut self) -> Result<Expr, String> {
        // Comparison operands are concat expressions (`||` binds tighter than a
        // comparison, looser than `+`/`-` — the ISO precedence the TS engine uses).
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
                // A CLOSED record type `RECORD { f :: TYPE [NOT NULL], … }` is parsed
                // into a schema Value and checked by `__is_typed_record`. (A bare/ANY
                // `RECORD` — no `{…}` — stays the plain `record` category below.)
                if self.peek_kw("RECORD")
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::LBrace))
                {
                    self.eat_kw("RECORD");
                    let schema = self.parse_record_schema()?;
                    let not_null = self.parse_typed_not_null();
                    let call = Expr::Call {
                        name: "__is_typed_record".into(),
                        args: vec![left, Expr::Lit(schema), Expr::Lit(Value::Bool(not_null))],
                    };
                    return Ok(if negated {
                        Expr::Not(Box::new(call))
                    } else {
                        call
                    });
                }
                let category = self.value_type_category()?;
                let not_null = self.parse_typed_not_null();
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
            // `e IS [NOT] DIRECTED` — a graph-element predicate (every edge here is
            // directed; a null element yields NULL).
            if self.eat_kw("DIRECTED") {
                return Ok(Expr::GraphPred {
                    op: crate::ir::GraphPredOp::IsDirected,
                    args: vec![left],
                    negated,
                });
            }
            // `a IS [NOT] SOURCE OF e` / `a IS [NOT] DESTINATION OF e`.
            if self.eat_kw("SOURCE") || self.peek_kw("DESTINATION") {
                let is_source = !self.eat_kw("DESTINATION");
                if !self.eat_kw("OF") {
                    return Err("expected OF after IS SOURCE / DESTINATION".into());
                }
                let edge = self.primary()?;
                return Ok(Expr::GraphPred {
                    op: if is_source {
                        crate::ir::GraphPredOp::IsSourceOf
                    } else {
                        crate::ir::GraphPredOp::IsDestinationOf
                    },
                    args: vec![left, edge],
                    negated,
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
    // folds into ONE n-ary `concat(...)` call (matching the TS engine's flat Concat node), so
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
            // Unary `- - - …` recurses here (not through `expr`), so guard the re-entry.
            let e = self.nest(|p| p.unary())?;
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
        // `!expr` — the tight-binding unary NOT (like C's `!`, tighter than the
        // comparison operators, unlike the keyword `NOT` which sits above them). So
        // `!(1=2) = true` parses as `(!(1=2)) = true`.
        if self.eat(&Tok::Bang) {
            // `!!!…` recurses here (not through `expr`), so guard the re-entry.
            return Ok(Expr::Not(Box::new(self.nest(|p| p.primary())?)));
        }
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
            Some(Tok::Param(name)) => {
                // In prepared mode `$name` is an unbound Param (bound per run); a
                // direct query substitutes it to its typed value now. Either way a
                // field chain may follow (`$rec.k`, `$list[0]`) → route through it.
                self.pos += 1;
                let e = if self.prepared {
                    Expr::Param(name)
                } else {
                    Expr::Lit(self.lookup_param(&name)?)
                };
                self.field_chain(e)
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
                // VALUE { MATCH <pattern> RETURN count(*) } — a correlated scalar
                // subquery (only the count(*) shape, which is a degree).
                if s.eq_ignore_ascii_case("value") && matches!(self.peek(), Some(Tok::LBrace)) {
                    return self.value_count_subquery_expr();
                }
                // `ALL_DIFFERENT(a, b, …)` / `SAME(a, b, …)` — graph-element identity
                // predicates over their element arguments.
                if (s.eq_ignore_ascii_case("all_different") || s.eq_ignore_ascii_case("same"))
                    && matches!(self.peek(), Some(Tok::LParen))
                {
                    let op = if s.eq_ignore_ascii_case("same") {
                        crate::ir::GraphPredOp::Same
                    } else {
                        crate::ir::GraphPredOp::AllDifferent
                    };
                    self.expect(&Tok::LParen)?;
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
                    return Ok(Expr::GraphPred {
                        op,
                        args,
                        negated: false,
                    });
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
                // ISO niladic current-datetime functions read the reserved `$__now`
                // clock param the host injects; absent → null. The engine stays pure —
                // it never reads a wall clock itself. A DATE `$__now` coerces to a
                // midnight datetime for the timestamp forms (via `temporal_ctor`). A
                // bare name or an empty `()` is the now-function; `local_time(arg)` is
                // the constructor, which falls through to the call path below.
                if let Some(kind) = current_temporal_kind(&s) {
                    let niladic = self.peek() != Some(&Tok::LParen)
                        || self.toks.get(self.pos + 1) == Some(&Tok::RParen);
                    if niladic {
                        if self.eat(&Tok::LParen) {
                            self.expect(&Tok::RParen)?;
                        }
                        let val = self
                            .params
                            .get("__now")
                            .map_or(Value::Null, |v| crate::exec::temporal_ctor(v, kind));
                        return Ok(Expr::Lit(val));
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
                    // A bare variable may still be subscripted (`x[0]` — a group
                    // variable list), so route through `field_chain` (a no-op when
                    // nothing follows).
                    self.field_chain(Expr::Slot(slot))
                }
            }
            other => Err(format!("expected an expression, got {other:?}")),
        }
    }

    /// Consume trailing `.field` accessors on a non-variable base (a record/paren
    /// expression), building nested `Expr::Field`. (A bare variable handles its
    /// own single `.prop` in `primary`, keeping that the optimizer's `Prop` shape.)
    fn field_chain(&mut self, mut base: Expr) -> Result<Expr, String> {
        // A group-variable list yields its typed element (node/edge) only at its FULL
        // nesting depth: a depth-1 `x[i]` is a node, but a depth-2 `x[i][j]` indexes an
        // inner list first (Plain) and the SECOND subscript is the node. Track the
        // group root's kind+depth and how many subscripts have been applied.
        let mut group: Option<(u8, crate::ir::ElemKind, u8)> = match &base {
            Expr::Slot(s) if self.group_node_slots.contains(s) => Some((
                self.group_var_depth.get(s).copied().unwrap_or(1),
                crate::ir::ElemKind::Node,
                0,
            )),
            Expr::Slot(s) if self.group_edge_slots.contains(s) => Some((
                self.group_var_depth.get(s).copied().unwrap_or(1),
                crate::ir::ElemKind::Edge,
                0,
            )),
            _ => None,
        };
        loop {
            if self.eat(&Tok::Dot) {
                let key = self.ident()?;
                base = Expr::Field {
                    base: Box::new(base),
                    key,
                };
                group = None; // a `.field` ends the group-index chain
            } else if self.eat(&Tok::LBracket) {
                // Subscript `base[index]`. On a group-variable list the subscript at the
                // variable's full depth yields the typed element (so `.prop` resolves the
                // node/edge property); shallower subscripts index inner lists (Plain).
                let index = self.expr()?;
                self.expect(&Tok::RBracket)?;
                let elem = match &mut group {
                    Some((depth, kind, applied)) => {
                        *applied += 1;
                        if *applied == *depth {
                            *kind
                        } else {
                            crate::ir::ElemKind::Plain
                        }
                    }
                    None => crate::ir::ElemKind::Plain,
                };
                base = Expr::Index {
                    base: Box::new(base),
                    index: Box::new(index),
                    elem,
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
    // no branch matches and it falls to ELSE — 3VL, matching the TS engine.
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
        // Read the target type name, joining the two-word `LOCAL`/`ZONED` temporal
        // forms (`LOCAL DATETIME` → `local_datetime`, `ZONED TIME` → `zoned_time`),
        // matching the TS engine's `read_type_name`.
        let mut ty = self.ident()?;
        let lead = ty.to_ascii_lowercase();
        if (lead == "local" || lead == "zoned")
            && matches!(self.peek(), Some(Tok::Ident(w))
                if matches!(w.to_ascii_lowercase().as_str(), "datetime" | "time"))
        {
            let w = self.ident()?;
            ty = format!("{lead}_{}", w.to_ascii_lowercase());
        }
        // A TEMPORAL cast DESUGARS to the matching temporal constructor function
        // (`CAST(x AS DATE)` → `date(x)`), exactly as the TS engine does — `TIMESTAMP` is a
        // DATETIME alias. A scalar cast keeps the throwing `CastTarget` path below.
        let temporal_fn = match ty.to_ascii_lowercase().as_str() {
            "date" => Some("date"),
            "datetime" | "timestamp" => Some("datetime"),
            "local_datetime" => Some("local_datetime"),
            "local_time" => Some("local_time"),
            "zoned_time" => Some("zoned_time"),
            "zoned_datetime" => Some("zoned_datetime"),
            "duration" => Some("duration"),
            _ => None,
        };
        if let Some(fname) = temporal_fn {
            self.expect(&Tok::RParen)?;
            return Ok(Expr::Call {
                name: fname.to_string(),
                args: vec![*expr],
            });
        }
        let target = match ty.to_ascii_uppercase().as_str() {
            "INTEGER" | "INT" => CastTarget::Integer,
            "FLOAT" | "DOUBLE" | "REAL" | "NUMBER" | "NUMERIC" => CastTarget::Float,
            "STRING" | "VARCHAR" | "TEXT" | "CHAR" => CastTarget::String,
            "BOOL" | "BOOLEAN" => CastTarget::Boolean,
            "LIST" => CastTarget::List,
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
        // An UNCORRELATED body (references no outer variable) is a self-contained
        // existence check — `EXISTS { MATCH (x:N) MATCH (y:M) }` — run once.
        if self.subquery_is_uncorrelated() {
            let (body, _, _) = self.parse_uncorrelated_subquery_body()?;
            self.expect(&Tok::RBrace)?;
            return Ok(Expr::UncorrelatedExists {
                body: Box::new(body),
            });
        }
        let (body, outer_width, _, _) = self.correlated_subquery_body("EXISTS")?;
        self.expect(&Tok::RBrace)?;
        Ok(Expr::Exists {
            body: Box::new(body),
            outer_width,
        })
    }

    /// True when the subquery body at `self.pos` (a `{`) references NO variable bound
    /// in the OUTER scope — i.e. it is self-contained (uncorrelated). Scans the
    /// balanced-brace body for any identifier that names an outer binding.
    fn subquery_is_uncorrelated(&self) -> bool {
        let mut depth = 0usize;
        let mut i = self.pos;
        while let Some(t) = self.toks.get(i) {
            match t {
                Tok::LBrace => depth += 1,
                Tok::RBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return true;
                    }
                }
                Tok::Ident(s) if self.scope.contains_key(s) => return false,
                _ => {}
            }
            i += 1;
        }
        true
    }

    /// Parse the body of an UNCORRELATED subquery — `{ [MATCH <pattern> [WHERE]]* }`,
    /// one or more MATCH clauses cross-joined (a shared variable becomes a join key),
    /// or an empty body (`Plan::Row`, for a `RETURN`-only VALUE). The opening `{` is
    /// consumed; the trailing `RETURN`/`}` are left to the caller. Parsed in a FRESH
    /// scope; the outer scope is RESTORED before returning, and the body's own scope
    /// and width are returned so a VALUE caller can parse a scalar RETURN against them.
    fn parse_uncorrelated_subquery_body(
        &mut self,
    ) -> Result<(Plan, HashMap<String, usize>, usize), String> {
        self.expect(&Tok::LBrace)?;
        // A parse error aborts the whole parse, so the outer scope need only be
        // restored on the success path.
        let outer_scope = self.scope.clone();
        let outer_slots = self.slots;
        let mut scope: HashMap<String, usize> = HashMap::new();
        let mut slots = 0usize;
        let mut plan = Plan::Row; // a RETURN-only body projects over one unit row
        if self.peek_kw("MATCH") || matches!(self.peek(), Some(Tok::LParen)) {
            self.eat_kw("MATCH");
            self.scope = HashMap::new();
            self.slots = 0;
            plan = self.match_body()?;
            scope = std::mem::take(&mut self.scope);
            slots = self.slots;
            while self.eat_kw("MATCH") {
                self.scope = HashMap::new();
                self.slots = 0;
                let p2 = self.match_body()?;
                let s2 = std::mem::take(&mut self.scope);
                let k2 = self.slots;
                let on: Vec<(usize, usize)> = s2
                    .iter()
                    .filter_map(|(v, &r)| scope.get(v).map(|&l| (l, r)))
                    .collect();
                plan = Plan::join(plan, p2, on);
                for (v, &r) in &s2 {
                    scope.entry(v.clone()).or_insert(slots + r);
                }
                slots += k2;
            }
        }
        self.scope = outer_scope;
        self.slots = outer_slots;
        Ok((plan, scope, slots))
    }

    // count_subquery := COUNT '{' node ( rel [quant] node )* [WHERE pred] '}' — the
    // number of sub-matches per outer row (distinct from the `count(*)` aggregate,
    // which takes `(…)`). Same correlated body as EXISTS.
    /// Trailing `NOT NULL` modifier on an IS TYPED type — but not if the `NOT` begins
    /// a separate clause. Returns whether a `NOT NULL` was consumed.
    fn parse_typed_not_null(&mut self) -> bool {
        if self.peek_kw("NOT") {
            let save = self.pos;
            self.eat_kw("NOT");
            if self.eat_kw("NULL") {
                return true;
            }
            self.pos = save;
        }
        false
    }

    /// Parse a closed record schema `{ f :: TYPE [NOT NULL], … }` (the opening
    /// `RECORD` keyword already consumed) into a `Value::Record` mapping each field to
    /// a descriptor `List[category, not_null, nested_schema_or_null]`, the shape
    /// `record_matches_schema` checks. A field type may itself be a nested `RECORD
    /// { … }`. An empty `{}` is a closed record with no fields.
    fn parse_record_schema(&mut self) -> Result<Value, String> {
        self.expect(&Tok::LBrace)?;
        let mut fields: Vec<(GStr, Value)> = Vec::new();
        if !matches!(self.peek(), Some(Tok::RBrace)) {
            loop {
                let name = self.ident()?;
                self.expect(&Tok::Colon)?;
                self.expect(&Tok::Colon)?;
                let (category, nested): (&'static str, Value) = if self.peek_kw("RECORD")
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::LBrace))
                {
                    self.eat_kw("RECORD");
                    ("record", self.parse_record_schema()?)
                } else {
                    (self.value_type_category()?, Value::Null)
                };
                let not_null = self.parse_typed_not_null();
                let desc = Value::List(vec![
                    Value::Str(category.into()),
                    Value::Bool(not_null),
                    nested,
                ]);
                fields.push((name.into(), desc));
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
        }
        self.expect(&Tok::RBrace)?;
        Ok(crate::value::make_record(fields))
    }

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
        if self.subquery_is_uncorrelated() {
            let (body, _, _) = self.parse_uncorrelated_subquery_body()?;
            self.expect(&Tok::RBrace)?;
            return Ok(Expr::UncorrelatedCount {
                body: Box::new(body),
            });
        }
        let (body, outer_width, _, _) = self.correlated_subquery_body("COUNT")?;
        self.expect(&Tok::RBrace)?;
        Ok(Expr::CountSubquery {
            body: Box::new(body),
            outer_width,
        })
    }

    // value := VALUE '{' MATCH <pattern> RETURN <expr> '}' — a correlated scalar
    // subquery. `RETURN count(*)` lowers to a CountSubquery (a degree); any other
    // scalar expression (`RETURN b.name`) becomes a ScalarSubquery: the body's single
    // value per outer row (NULL when the body matches nothing).
    fn value_count_subquery_expr(&mut self) -> Result<Expr, String> {
        // An UNCORRELATED body is a self-contained query — build the whole thing
        // (pattern + RETURN projection/aggregate) and run it once.
        if self.subquery_is_uncorrelated() {
            let (body, body_scope, body_slots) = self.parse_uncorrelated_subquery_body()?;
            if !self.eat_kw("RETURN") {
                return Err("a VALUE subquery must end with RETURN <expr>".into());
            }
            let saved_scope = std::mem::replace(&mut self.scope, body_scope);
            let saved_slots = std::mem::replace(&mut self.slots, body_slots);
            let items = self.return_items()?;
            let (full, _) = apply_items(body, &items);
            self.scope = saved_scope;
            self.slots = saved_slots;
            self.expect(&Tok::RBrace)?;
            return Ok(Expr::UncorrelatedScalar {
                body: Box::new(full),
            });
        }
        let (body, outer_width, sub_scope, sub_slots) = self.correlated_subquery_body("VALUE")?;
        if !self.eat_kw("RETURN") {
            return Err("a VALUE subquery must end with RETURN <expr>".into());
        }
        // `RETURN count(*)` → CountSubquery (no scope needed).
        if self.peek_kw("COUNT")
            && matches!(self.toks.get(self.pos + 1), Some(Tok::LParen))
            && matches!(self.toks.get(self.pos + 2), Some(Tok::Star))
        {
            self.eat_kw("COUNT");
            self.expect(&Tok::LParen)?;
            self.expect(&Tok::Star)?;
            self.expect(&Tok::RParen)?;
            self.expect(&Tok::RBrace)?;
            return Ok(Expr::CountSubquery {
                body: Box::new(body),
                outer_width,
            });
        }
        // A scalar RETURN expression, parsed against the body's sub-scope.
        let saved_scope = std::mem::replace(&mut self.scope, sub_scope);
        let saved_slots = std::mem::replace(&mut self.slots, sub_slots);
        let scalar = self.expr()?;
        self.scope = saved_scope;
        self.slots = saved_slots;
        self.expect(&Tok::RBrace)?;
        Ok(Expr::ScalarSubquery {
            body: Box::new(body),
            scalar: Box::new(scalar),
            outer_width,
        })
    }

    /// Parse the `{ <pattern> [WHERE pred] }` body shared by `EXISTS { … }` and
    /// `COUNT { … }`: a pattern correlated on an outer-bound start variable, rooted at
    /// `Plan::Row`, with slot `outer_width` reserved for the evaluator's provenance
    /// column. Returns `(body, outer_width)`. `kw` names the construct for errors.
    #[allow(clippy::type_complexity)]
    fn correlated_subquery_body(
        &mut self,
        kw: &str,
    ) -> Result<(Plan, usize, HashMap<String, usize>, usize), String> {
        self.expect(&Tok::LBrace)?;
        // The pattern may be written with an explicit leading `MATCH` — `EXISTS {
        // MATCH (a)-[:R]->(b) }` — the full-statement form; accept it as sugar.
        self.eat_kw("MATCH");
        let outer_width = self.slots;
        let (var, label, props, start_where, le) = self.node_plain()?;
        if start_where.is_some() {
            return Err(format!(
                "inline WHERE on a {kw} start node is not supported; use a trailing WHERE"
            ));
        }
        // FORWARD only when the start node NAMES a bound variable; an anonymous start
        // (`(:Person)-[:R]->(s)`) or a local-named one falls to the REVERSE branch, where
        // the correlated (bound) variable is the landing endpoint.
        let forward_from = var
            .as_ref()
            .and_then(|v| self.scope.get(v).copied())
            .map(|from| (var.clone().unwrap(), from));

        let mut sub_scope = self.scope.clone();
        let mut sub_slots = outer_width + 1;
        let body = if let Some((v, from)) = forward_from {
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
            // REVERSE (single hop): the start node is a LOCAL node — named (`(m)-[:R]->(n)`)
            // or ANONYMOUS (`(:Person)-[:CREATED]->(s)`) — and the correlated (bound)
            // variable is the LANDING. Traverse from the bound endpoint backward to the
            // local node; the local node's label/props become a landing filter.
            let rel = self.rel(false)?;
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
            let (vb, vb_label, vb_props, vb_where, vb_le) = self.node_plain()?;
            let Some(vb) = vb else {
                return Err(format!("{kw} must correlate on a bound variable"));
            };
            let Some(&from) = self.scope.get(&vb) else {
                return Err(format!(
                    "{kw} must start from or land on a bound (correlated) variable; neither \
                     the start node nor `{vb}` is in scope"
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
            if let Some(v) = &var {
                sub_scope.insert(v.clone(), local_slot);
            }
            let mut body = Plan::Row.expand(from, rev_dir, &rel.etypes);
            if let Some(pred) = landing_label_filter(label, le, local_slot) {
                body = body.filter(pred);
            }
            body = node_prop_filters(body, local_slot, props);
            body
        };
        let body = if self.eat_kw("WHERE") {
            let saved_scope = std::mem::replace(&mut self.scope, sub_scope.clone());
            let saved_slots = std::mem::replace(&mut self.slots, sub_slots);
            let pred = self.bool_pred()?;
            self.scope = saved_scope;
            self.slots = saved_slots;
            body.filter(pred)
        } else {
            body
        };
        // The caller consumes any trailing `RETURN …` (for VALUE) and the closing
        // `}`; it also owns the sub-scope so it can parse a scalar RETURN expression.
        Ok((body, outer_width, sub_scope, sub_slots))
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
        // `cardinality` is POLYMORPHIC (list/string via the normal path below), but ISO
        // also defines it over a PATH — the count of nodes + edges. Route only the
        // path-variable form to the accessor; a list/string arg falls through.
        if lname == "cardinality" && matches!(args.as_slice(), [Expr::Path]) {
            return Ok(Expr::PathAccess {
                part: PathPart::Cardinality,
            });
        }
        let arity_ok = match lname.as_str() {
            // 0 args (numeric constants)
            "e" | "pi" => args.is_empty(),
            // 1 arg
            "abs" | "sign" | "floor" | "ceil" | "ceiling" | "sqrt" | "exp" | "ln" | "log10"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh"
            | "cot" | "degrees" | "radians" | "upper" | "lower" | "trim" | "size"
            | "cardinality" | "head" | "last"
            // Temporal component accessors carry the leading-underscore extension sigil
            // (`_year`), matching the TS engine; the bare ISO spellings are NOT in the grammar.
            | "_year" | "_month" | "_day" | "_hour" | "_minute" | "_second"
            // Non-finite CLASSIFIERS (leading-underscore extensions — NOT ISO). Total
            // boolean predicates over the IEEE-754 special values that GQL has no
            // literal or predicate for: `_is_nan`/`_is_infinite`/`_is_finite`. Never
            // null, never throw — a non-number argument is simply not that kind.
            | "_is_nan" | "_is_infinite" | "_is_finite" | "date"
            | "local_time" | "datetime" | "local_datetime" | "zoned_time" | "zoned_datetime"
            | "duration" | "to_integer" | "tointeger" | "to_float" | "tofloat" | "to_string"
            | "tostring" | "to_boolean" | "toboolean" | "char_length" | "character_length"
            | "byte_length" | "octet_length" | "reverse" | "tail" | "keys" | "labels" | "type"
            | "property_names" | "element_id" | "to_list" | "tolist" => args.len() == 1,
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
            // Prefixed so the FFI codes it `E_UNKNOWN_FUNCTION` (more specific than the
            // generic parse `E_SYNTAX`), matching the pure-TS engine — a caught error tells
            // the host it was an unknown name, not a malformed query.
            _ => return Err(format!("E_UNKNOWN_FUNCTION: unknown function `{name}`")),
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
    /// char, when present, is the SECOND argument, matching the TS engine.
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

/// An open upper bound (`+`, `*`, `{n,}`) is genuinely unbounded — matching the TS engine's
/// transitive-closure semantics. Enumeration terminates anyway in Trail mode (a trail
/// can't repeat an edge, so its length is bounded by the edge count), and the trail
/// LIMIT is the anti-runaway guard: a closure too large to enumerate fails LOUDLY with
/// `E_RESOURCE_EXHAUSTED` rather than being silently truncated to a wrong answer (which a
/// hard 32-hop cap here did — it returned a subset of the real result set).
const MAX_VARLEN: u32 = u32::MAX;

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

/// Convert an inline-CALL arm's RETURN items into `(name, Expr)` yields, rejecting any
/// aggregate (a set-op / plain CALL body may not aggregate — only the single-arm
/// `COUNT(*)` special case, handled by the caller before this).
fn call_items_to_yields(items: Vec<RetItem>) -> Result<Vec<(String, Expr)>, String> {
    items
        .into_iter()
        .map(|it| match it {
            RetItem::Key(name, e) => Ok((name, e)),
            RetItem::Agg(_) | RetItem::AggExpr { .. } => {
                Err("an aggregating RETURN inside CALL { … } is not supported".to_string())
            }
        })
        .collect()
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
        // An unaliased property access is named `binding.key` (the expression text,
        // e.g. `n.id`), matching the TS engine — not the bare property key `id`. Resolve the
        // binding's name from its slot; fall back to `default_name` for an unnamed slot.
        if let Expr::Prop { slot, key } = e {
            if let Some((var, _)) = self.scope.iter().find(|(_, &s)| s == *slot) {
                return format!("{var}.{key}");
            }
        }
        // A bare bound path (`RETURN p`) takes the path variable's name. There is
        // exactly one path per row (the lineage), so the single declared path var is it.
        if matches!(e, Expr::Path) {
            if let Some(name) = self.path_vars.iter().next() {
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
        // `TIMESTAMP` is the TS engine's alias for a (local) DATETIME literal.
        "DATETIME" | "TIMESTAMP" => "datetime",
        "DURATION" => "duration",
        _ => return None,
    })
}

/// Map an ISO niladic current-datetime function to the temporal kind it reads the
/// `$__now` clock as: `current_timestamp`/`local_timestamp` → a local `datetime`,
/// `current_date` → a `date`, `current_time`/`local_time` → a local time-of-day
/// (`localtime`; null for a DATE `$__now`). Kept in step with the pure-TS desugaring
/// so the two engines stay byte-identical.
///
/// `local_time` is only niladic when argumentless — `local_time('13:47:09')` is the
/// constructor, which reaches this path with a `(`; the caller resolves the niladic
/// form from a bare identifier, so a constructor call never lands here.
fn current_temporal_kind(name: &str) -> Option<&'static str> {
    Some(match name {
        "current_timestamp" | "local_timestamp" => "datetime",
        "current_date" => "date",
        "current_time" | "local_time" => "localtime",
        _ => return None,
    })
}

/// Map a path-accessor function name to its `PathPart`, or `None` if it is not one.
/// The ISO GQL name for the edge list is `edges`; the Cypher-ism `relationships` is NOT
/// accepted (it is a non-ISO alias — use `edges`), so it falls through to an unknown
/// function, matching pure-TS.
fn path_part(name: &str) -> Option<PathPart> {
    Some(match name {
        "nodes" => PathPart::Nodes,
        "edges" => PathPart::Relationships,
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
mod tests;
