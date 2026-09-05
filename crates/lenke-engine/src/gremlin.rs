//! Gremlin front-end: parse a traversal into the SAME neutral IR ([`crate::ir`])
//! that the GQL front-end targets. This is the proof of the design — both
//! languages are thin compilers over one algebra, and the payoff test asserts
//! that a GQL query and its Gremlin equivalent lower to plans producing identical
//! rows.
//!
//! Subset: `g.V().hasLabel('L').has('k', <val|P.op(val)>).out|in|both('R')
//! .values('k') | .count() | .dedup() | .order().by('k'[,asc|desc])
//! .limit(n) | .range(lo,hi) | .groupCount().by('k')`. The traversal's implicit
//! current element is a slot, hops append slots, exactly as in the IR.

use crate::ir::{Agg, AggFn, ArithOp, CompareOp, Dir, Expr, PathMode, Plan, SortKey};
use crate::value::Value;
use std::collections::HashMap;

/// A Gremlin step carries at most one edge label; the plan builders take the
/// `&[String]` edge-type list (empty = any) that GQL's `|`-disjunction produces.
fn etypes_of(label: Option<&str>) -> Vec<String> {
    label.into_iter().map(str::to_string).collect()
}

/// Reject a malformed NAME written by `addV`/`addE`/`property`: an empty name, or one
/// containing the GraphSON multi-label separator `::` (which would break round-tripping
/// through the codecs). Gremlin is otherwise permissive about arbitrary label/key
/// strings — this only guards the write steps, matching the gate the TS engine applied there.
fn check_write_name(kind: &str, name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err(format!("a {kind} must not be empty"));
    }
    if name.contains("::") {
        return Err(format!(
            "a {kind} must not contain the `::` label separator (got `{name}`)"
        ));
    }
    Ok(())
}

/// Whether `plan` can be used as a correlated-EXISTS body — i.e. it PRESERVES the
/// provenance column the EXISTS eval appends (only column-appending / row-filtering
/// ops). A `Project`/`Aggregate`/`Distinct` etc. would drop or reshape that column, so
/// EXISTS over such a body cannot back-map its survivors (it would panic). Used to
/// decide whether a coalesce arm gets an EXACT full-body existence guard or the
/// leading-hop approximation.
/// A frontier-PRESERVING step keeps an edge in scope for a later endpoint move — the same set
/// the pure-TS engine's `PRESERVE_STEPS` uses so `edge_scope` tracks `hasEdge` byte-for-byte.
fn is_edge_scope_preserving(lname: &str) -> bool {
    matches!(
        lname,
        "has"
            | "haslabel"
            | "hasid"
            | "haskey"
            | "hasnot"
            | "hasnotlabel"
            | "hasvalue"
            | "where"
            | "and"
            | "or"
            | "not"
            | "is"
            | "dedup"
            | "limit"
            | "skip"
            | "range"
            | "tail"
            | "order"
            | "by"
            | "as"
            | "identity"
            | "barrier"
            | "sample"
            | "coin"
            | "none"
            | "simplepath"
            | "cyclicpath"
            | "aggregate"
            | "store"
            | "sideeffect"
            | "filter"
            | "profile"
            | "drop"
            | "property"
            | "repeat"
    )
}

/// Combine a branch's arms' edge-in-scope (port of the pure-TS `combineHasEdge`): any arm that
/// DEFINITELY lacks an edge (`Some(false)`) makes a following endpoint faultable; all-edge arms
/// keep it in scope; anything else is unknown (never faults).
fn combine_edge_scope(arms: &[Option<bool>]) -> Option<bool> {
    if arms.contains(&Some(false)) {
        Some(false)
    } else if !arms.is_empty() && arms.iter().all(|a| *a == Some(true)) {
        Some(true)
    } else {
        None
    }
}

/// Maximum sub-traversal nesting depth accepted by the parser (see [`Parser::depth`]).
/// Far above any real traversal and safely below the stack-overflow threshold on the
/// smallest stack the parser runs on (the wasm CLI, cargo's 2 MB test threads) — the
/// `step` frame is large, so this is deliberately conservative. MUST match the TS
/// `@lenke/gremlin` parser's cap so the two engines accept/reject the same queries.
const MAX_TRAVERSAL_DEPTH: usize = 128;

pub fn parse(query: &str) -> Result<Plan, String> {
    let toks = lex(query)?;
    // A traversal that reads a full `path()`/`tree()` records each value-producing step's
    // frontier into `Lineage::steps` (via `Plan::PathRecord`) so the path can carry projected
    // scalars, an edge source, etc. Pre-scanned so the lowering knows to emit those records
    // from the first step; harmless if a `path`/`tree` ident is actually a property name (the
    // records are no-ops when no path is read).
    // Only a TOP-LEVEL `path()`/`tree()` (paren-depth 0) drives the full-history recording;
    // a `path()` inside a branch arm (`optional(path())`, `union(path(), …)`) keeps the
    // established per-arm lineage, since `PathRecord` is emitted only on the main chain and
    // mixing the two layouts is unsound.
    let building_full_path = {
        // A `path()`/`tree()` ANYWHERE (top level OR inside a branch arm) drives the full
        // per-step history: the leading steps before a branch must record their frontier so a
        // path() inside an arm sees the whole history (`E().inV().coalesce(path(), …)`).
        toks.iter().any(|t| {
            matches!(t, Tok::Ident(s) if s.eq_ignore_ascii_case("path") || s.eq_ignore_ascii_case("tree"))
        })
    };
    let mut p = Parser {
        toks,
        building_full_path,
        pos: 0,
        current: 0,
        slots: 1,
        labels: HashMap::new(),
        edge_hop: None,
        pending_repeat: None,
        pending_until: None,
        path_ok: true,
        edge_path_ok: true,
        path_has_edges: false,
        path_leaf: None,
        path_ok_pre_step: true,
        caps: std::collections::HashMap::new(),
        algo_props: std::collections::HashMap::new(),
        last_algo: None,
        on_edge: false,
        prop_keys: None,
        first_labels: std::collections::HashMap::new(),
        all_labels: std::collections::HashMap::new(),
        sack_slot: None,
        subgraph_caps: std::collections::HashMap::new(),
        pending_write: None,
        current_is_map: false,
        current_is_element: false,
        current_is_path: false,
        current_is_scalar: false,
        edge_scope: None,
        last_arm_edge_scope: None,
        last_arm_frontier: (false, false, false),
        last_arm_last_step: String::new(),
        frontier_from_reducer: false,
        depth: 0,
    };
    p.traversal()
}

// --- lexer -------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Dot,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Ident(String),
    Str(String),
    Num(f64),
}

/// Build `left = v0 OR left = v1 OR …` — the membership test `within(v0, v1, …)`
/// desugars to. An empty list is a constant FALSE (`1 = 0`), so `within()` and
/// (negated) `without()` behave sensibly.
// ── math() expression parsing ────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
enum MathTok {
    Num(f64),
    Ident(String),
    Op(char), // + - * / % ^
    LParen,
    RParen,
    Comma,
    Underscore,
}

fn math_lex(s: &str) -> Result<Vec<MathTok>, String> {
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '+' | '-' | '*' | '/' | '%' | '^' => {
                out.push(MathTok::Op(c));
                i += 1;
            }
            '(' => {
                out.push(MathTok::LParen);
                i += 1;
            }
            ')' => {
                out.push(MathTok::RParen);
                i += 1;
            }
            ',' => {
                out.push(MathTok::Comma);
                i += 1;
            }
            '_' if !(i + 1 < b.len() && (b[i + 1].is_alphanumeric() || b[i + 1] == '_')) => {
                out.push(MathTok::Underscore);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.' || b[i] == 'e') {
                    i += 1;
                }
                let n: f64 = b[start..i]
                    .iter()
                    .collect::<String>()
                    .parse()
                    .map_err(|_| "math(): bad number".to_string())?;
                out.push(MathTok::Num(n));
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < b.len() && (b[i].is_alphanumeric() || b[i] == '_') {
                    i += 1;
                }
                out.push(MathTok::Ident(b[start..i].iter().collect()));
            }
            other => return Err(format!("math(): unexpected char `{other}`")),
        }
    }
    Ok(out)
}

fn math_const(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        "e" => Some(std::f64::consts::E),
        _ => None,
    }
}

fn is_unary_math_fn(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "sqrt"
            | "abs"
            | "ceil"
            | "floor"
            | "exp"
            | "ln"
            | "log10"
            | "signum"
    )
}

/// Whether `name` is a known `math()` function — the numeric scalar functions (0/1/2
/// arg). `math()` is numeric-only, so this is a safe subset of the GQL scalar surface;
/// a name outside it is an unknown function, rejected at parse (the engine assumes
/// function names are validated before evaluation, like GQL's `call`).
fn is_known_math_fn(name: &str) -> bool {
    is_unary_math_fn(name)
        || matches!(
            name,
            "e" | "pi"
                | "ceiling"
                | "sign"
                | "cot"
                | "degrees"
                | "radians"
                | "round"
                | "log"
                | "pow"
                | "power"
                | "mod"
                | "atan2"
                | "min"
                | "max"
        )
}

/// Map a math() function name to the engine scalar-fn name (identical kernels).
fn math_fn_name(name: &str) -> &str {
    match name {
        "signum" => "sign",
        "pow" => "power",
        other => other,
    }
}

struct MathParser<'a> {
    toks: Vec<MathTok>,
    pos: usize,
    operand: &'a Expr,
    labels: &'a std::collections::HashMap<String, usize>,
}

impl MathParser<'_> {
    fn peek(&self) -> Option<&MathTok> {
        self.toks.get(self.pos)
    }
    fn expr(&mut self) -> Result<Expr, String> {
        let mut left = self.term()?;
        while let Some(MathTok::Op(op @ ('+' | '-'))) = self.peek().cloned() {
            self.pos += 1;
            let right = self.term()?;
            left = Expr::Arith {
                op: if op == '+' {
                    ArithOp::Add
                } else {
                    ArithOp::Sub
                },
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }
    fn term(&mut self) -> Result<Expr, String> {
        let mut left = self.power()?;
        while let Some(MathTok::Op(op @ ('*' | '/' | '%'))) = self.peek().cloned() {
            self.pos += 1;
            let right = self.power()?;
            left = Expr::Arith {
                op: match op {
                    '*' => ArithOp::Mul,
                    '/' => ArithOp::Div,
                    _ => ArithOp::Rem,
                },
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }
    fn power(&mut self) -> Result<Expr, String> {
        let base = self.unary()?;
        if let Some(MathTok::Op('^')) = self.peek() {
            self.pos += 1;
            let exp = self.power()?; // right-associative
            return Ok(Expr::Call {
                name: "power".into(),
                args: vec![base, exp],
            });
        }
        Ok(base)
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if let Some(MathTok::Op(op @ ('-' | '+'))) = self.peek().cloned() {
            self.pos += 1;
            let e = self.unary()?;
            return Ok(if op == '-' {
                Expr::Arith {
                    op: ArithOp::Sub,
                    left: Box::new(Expr::Lit(Value::Num(0.0))),
                    right: Box::new(e),
                }
            } else {
                e
            });
        }
        self.primary()
    }
    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek().cloned() {
            Some(MathTok::Num(n)) => {
                self.pos += 1;
                Ok(Expr::Lit(Value::Num(n)))
            }
            Some(MathTok::Underscore) => {
                self.pos += 1;
                Ok(self.operand.clone())
            }
            Some(MathTok::LParen) => {
                self.pos += 1;
                let e = self.expr()?;
                if self.peek() != Some(&MathTok::RParen) {
                    return Err("math(): expected `)`".into());
                }
                self.pos += 1;
                Ok(e)
            }
            Some(MathTok::Ident(name)) => {
                self.pos += 1;
                // `name(args)` function call.
                if self.peek() == Some(&MathTok::LParen) {
                    self.pos += 1;
                    let mut args = vec![self.expr()?];
                    while self.peek() == Some(&MathTok::Comma) {
                        self.pos += 1;
                        args.push(self.expr()?);
                    }
                    if self.peek() != Some(&MathTok::RParen) {
                        return Err("math(): expected `)` after args".into());
                    }
                    self.pos += 1;
                    // Reject an unknown function at PARSE (the evaluator assumes names
                    // were validated and silently NULLs an unknown one otherwise). The
                    // code is E_INVALID_VALUE (applied by `parse_math`) to match the
                    // pure-TS engine, which treats every math() failure as a value error.
                    if !is_known_math_fn(&name) {
                        return Err(format!("math(): unknown function `{name}`"));
                    }
                    return Ok(Expr::Call {
                        name: math_fn_name(&name).to_string(),
                        args,
                    });
                }
                // Bare unary application `sin _` / `sin 2` / `abs -3` (only unary
                // functions). A signed operand (`abs -3`) is taken as the function's
                // argument (mXparser juxtaposition binds tighter than the outer `-`).
                // A name BOUND as a step-label shadows the function (`math('sin + 1')`
                // with an `as('sin')` reads the variable), so skip the application then.
                if is_unary_math_fn(&name)
                    && !self.labels.contains_key(&name)
                    && matches!(
                        self.peek(),
                        Some(
                            MathTok::Num(_)
                                | MathTok::Underscore
                                | MathTok::LParen
                                | MathTok::Ident(_)
                                | MathTok::Op('-' | '+')
                        )
                    )
                {
                    let arg = self.unary()?;
                    return Ok(Expr::Call {
                        name: math_fn_name(&name).to_string(),
                        args: vec![arg],
                    });
                }
                // A constant, else a named step-label variable.
                if let Some(c) = math_const(&name) {
                    return Ok(Expr::Lit(Value::Num(c)));
                }
                match self.labels.get(&name) {
                    Some(&slot) => Ok(Expr::Slot(slot)),
                    None => Err(format!("math(): unknown variable `{name}`")),
                }
            }
            other => Err(format!("math(): unexpected token {other:?}")),
        }
    }
}

fn or_of_equals(left: &Expr, vals: &[Value]) -> Expr {
    let eq = |v: &Value| Expr::Compare {
        op: CompareOp::Eq,
        left: Box::new(left.clone()),
        right: Box::new(Expr::Lit(v.clone())),
    };
    let mut it = vals.iter();
    match it.next() {
        None => Expr::Compare {
            op: CompareOp::Eq,
            left: Box::new(Expr::Lit(Value::Num(1.0))),
            right: Box::new(Expr::Lit(Value::Num(0.0))),
        },
        Some(first) => it.fold(eq(first), |acc, v| Expr::Or(Box::new(acc), Box::new(eq(v)))),
    }
}

/// A parsed `group().by(...)` modulator: a property key, the current element, or a
/// reducing `count()` traversal. Used to build the key-by and value-by of a group.
enum GroupBy {
    Key(String),
    /// An `id`/`label` token by-modulator, carrying its element expression.
    KeyExpr(String, Expr),
    Element,
    Count,
    /// A reducing sub-traversal value-by — `by(__.values('v').sum())` etc.: the
    /// aggregate function and its (optional) per-element argument expression.
    Reduce(AggFn, Option<Expr>),
}

/// TinkerPop: a keying `by()` (group/groupCount/order/dedup/select) that yields NO
/// VALUE — an absent property — FILTERS the traverser rather than keying it under a
/// null bucket. For each by-key that lowered to a bare property projection
/// (`Expr::Prop`), build a `PROPERTY_EXISTS` gate (the one predicate separating absent
/// from a stored null); AND several together. A non-`Prop` key (the element itself, an
/// id/label token, a computed expr) always yields a value → no gate. `None` = no gate
/// needed. The caller `plan.filter(...)`s the result before the keying operator.
fn key_present_gate(key_exprs: &[&Expr]) -> Option<Expr> {
    let mut gate: Option<Expr> = None;
    for e in key_exprs {
        if let Expr::Prop { slot, key } = e {
            let g = Expr::PropertyExists {
                slot: *slot,
                key: key.clone(),
            };
            gate = Some(match gate {
                Some(acc) => Expr::And(Box::new(acc), Box::new(g)),
                None => g,
            });
        }
    }

    gate
}

/// Shift every slot reference `>= threshold` up by one — used when a correlated
/// subquery body gets a provenance column inserted at `threshold`, pushing the body's
/// own appended slots up. Recurses through the common scalar Expr shapes.
fn shift_body_slots(e: &mut Expr, threshold: usize) {
    match e {
        Expr::Slot(s) => {
            if *s >= threshold {
                *s += 1;
            }
        }
        Expr::Prop { slot, .. } | Expr::PropertyExists { slot, .. } => {
            if *slot >= threshold {
                *slot += 1;
            }
        }
        Expr::Call { args, .. } | Expr::List { items: args } => {
            for a in args {
                shift_body_slots(a, threshold);
            }
        }
        Expr::Compare { left, right, .. }
        | Expr::Arith { left, right, .. }
        | Expr::And(left, right)
        | Expr::Or(left, right) => {
            shift_body_slots(left, threshold);
            shift_body_slots(right, threshold);
        }
        Expr::Not(x) => shift_body_slots(x, threshold),
        _ => {}
    }
}

/// A stable string tag for a comparison operator, for the `list_none` scan fn.
fn compare_op_tag(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "eq",
        CompareOp::Ne => "neq",
        CompareOp::Gt => "gt",
        CompareOp::Ge => "gte",
        CompareOp::Lt => "lt",
        CompareOp::Le => "lte",
    }
}

/// A runtime label-membership predicate over `slot`: `label ∈ labels(slot)`, OR-ed
/// across `labels` (matching Gremlin `hasLabel('A','B')` = has ANY of them). Uses the
/// list-valued `labels()` element function and `Expr::In`, so it works anywhere in a
/// traversal (not just folded into a scan).
fn label_membership(slot: usize, labels: &[String]) -> Expr {
    // A single vectorized membership test over the label buckets — no per-row list
    // materialization (the old `In(Lit, labels(slot))` OR-chain allocated a label list
    // and string-compared per row, which dominated `hasLabel` filters on big frontiers).
    Expr::IsLabeled {
        slot,
        labels: labels.to_vec(),
    }
}

/// If `plan` is the `values('k')` lowering — `Project([(_, Prop{s,k})])` over
/// `Filter(PropertyExists{s,k})` — unwrap it to `(chain, Prop{s,k})` so a FOLLOWING
/// reducing aggregate folds the property DIRECTLY over the chain. Equivalent (a reduce
/// skips nulls, so filtering absent first changes nothing) and it lets the frontier /
/// var-length aggregate fast-paths recognize the chain, which a values wrapper hides —
/// turning `<hops>.values(k).sum()` from a full frontier materialization into the fused
/// path. Otherwise the plan is returned unchanged with `Slot(current)` as the arg.
fn unwrap_values_fold(plan: Plan, current: usize) -> (Plan, Expr) {
    let is_shape = matches!(&plan, Plan::Project { items, input }
        if items.len() == 1
            && matches!(&items[0].1, Expr::Prop { slot, key }
                if matches!(input.as_ref(), Plan::Filter { pred, .. }
                    if matches!(pred, Expr::PropertyExists { slot: ps, key: pk } if ps == slot && pk == key))));
    if !is_shape {
        return (plan, Expr::Slot(current));
    }
    let Plan::Project { items, input } = plan else {
        unreachable!("shape checked")
    };
    let arg = items.into_iter().next().expect("one item").1; // Prop{slot,key}
    let Plan::Filter { input: chain, .. } = *input else {
        unreachable!("shape checked")
    };
    (*chain, arg)
}

/// Whether a plan is a write (so read steps cannot chain after it).
fn is_write(plan: &Plan) -> bool {
    matches!(
        plan,
        Plan::Insert { .. } | Plan::Update { .. } | Plan::Merge { .. } | Plan::AddEdge { .. }
    )
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
        match c {
            '.' => out.push(Tok::Dot),
            '[' => out.push(Tok::LBracket),
            ']' => out.push(Tok::RBracket),
            '(' => out.push(Tok::LParen),
            ')' => out.push(Tok::RParen),
            ',' => out.push(Tok::Comma),
            '\'' | '"' => {
                let quote = c;
                let mut t = String::new();
                i += 1;
                // Decode the common escapes (`\n \t \r`, `\\`, `\'`) rather than
                // dropping the backslash — byte-identical to the TS engine's lexer.
                let mut terminated = false;
                while i < b.len() {
                    let ch = b[i];
                    if ch == quote {
                        i += 1;
                        terminated = true;
                        break;
                    }
                    if ch == '\\' && i + 1 < b.len() {
                        i += 1;
                        t.push(match b[i] {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            other => other,
                        });
                        i += 1;
                        continue;
                    }
                    t.push(ch);
                    i += 1;
                }
                if !terminated {
                    return Err("unterminated string literal".into());
                }
                out.push(Tok::Str(t));
                continue;
            }
            _ if c.is_ascii_digit()
                || (c == '-' && b.get(i + 1).is_some_and(char::is_ascii_digit)) =>
            {
                let start = i;
                i += 1;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                    i += 1;
                }
                let text: String = b[start..i].iter().collect();
                let n: f64 = text.parse().map_err(|_| format!("bad number `{text}`"))?;
                out.push(Tok::Num(n));
                continue;
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

/// An open `repeat(<hop>)` awaiting its modulators. `times(n)` bounds the walk;
/// `emit()`/`emit(pred)` and `until(pred)` emit at every depth (min becomes 1) and
/// optionally filter the emitted endpoint. Flushed into a `VarLength` walk (endpoint
/// at `out_slot`) by `flush_repeat` when a non-modulator step or the end follows.
struct RepeatCtx {
    dir: Dir,
    label: Option<String>,
    /// the slot the loop hops FROM (the element repeat() was reached on).
    from: usize,
    /// the slot the walk's endpoint lands in (== the width at flush time).
    out_slot: usize,
    /// `times(n)` bound, if given; absent means the default iteration cap.
    times: Option<u32>,
    /// emit/until makes the walk emit at every depth ≥ 1 (min = 1).
    min_one: bool,
    /// an `emit(pred)`/`until(pred)` filter on the emitted endpoint (at `out_slot`).
    filter: Option<Expr>,
    /// an inner `.as('tag')` in the body — bound to the endpoint at flush time.
    bind_tag: Option<String>,
    /// a `loops()`-based min-depth override (e.g. `emit(loops().is(gt(1)))` emits from
    /// depth 2). Overrides the default `min` in `flush_repeat`.
    min_override: Option<u32>,
    /// `path_ok` at the moment `repeat()` opened. The walk lowers to a pure vertex-hop
    /// VarLength (which records full path lineage), so `flush_repeat` restores this —
    /// letting `path()`/`tree()` follow a `repeat(<vertex-hop>)` when the prefix was
    /// itself path-answerable.
    path_ok_at_open: bool,
    /// A Gremlin `until(pred)` stop condition on the walk. The pre-form
    /// `until(pred).repeat(body)` (while-do, checked BEFORE the body — `until_pre`) and
    /// the post-form `repeat(body).until(pred)` (do-while) both land here; `flush_repeat`
    /// sets `min` to 0 (pre) or 1 (post) and lowers to `var_length_until`.
    until: Option<Expr>,
    /// True when `until` came from the PRE-form (`until(pred).repeat(body)`): a source
    /// already satisfying `pred` emits at depth 0.
    until_pre: bool,
    /// A `repeat(<hop>.<filter>)` body filter on the hop TARGET (e.g.
    /// `repeat(out().hasLabel('PERSON'))`) — a target failing it is pruned. `None` = a
    /// bare hop body.
    body_filter: Option<Expr>,
    /// A `repeat(<hop>.where(loops().is(<op>(n))))` in-body depth guard, lowered to a
    /// max-depth cap on the walk (`loops()` == depth + 1). `None` = no cap from a body
    /// `where(loops())`.
    max_cap: Option<u32>,
    /// A degenerate `repeat(identity())` body: the walk doesn't move, so `flush_repeat`
    /// passes the frontier through unchanged (exact for `times(0)`).
    identity_body: bool,
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    /// the slot the current element lives in (hops advance it).
    current: usize,
    /// number of bound element slots.
    slots: usize,
    /// `as('x')` step labels -> the slot bound when the label was set. `select('x')`
    /// resolves through this.
    labels: HashMap<String, usize>,
    /// Set by an edge hop (`outE`/`inE`/`bothE`) and consumed by the very next
    /// vertex-move step (`inV`/`outV`/`otherV`): `(landed node slot, hop direction)`.
    /// The edge hop leaves the current element on the EDGE; the vertex move steps
    /// back onto the endpoint the hop already landed. Cleared at the top of every
    /// `step`, so a vertex move only resolves when it IMMEDIATELY follows the edge
    /// hop — exactly Gremlin's requirement.
    edge_hop: Option<(usize, Dir)>,
    /// Set by `repeat(<hop>)` and held open across its modulators (`times`/`emit`/
    /// `until`); the next non-modulator step (or the end) flushes it into a
    /// `VarLength` walk. See [`RepeatCtx`] and `flush_repeat`.
    pending_repeat: Option<RepeatCtx>,
    /// A PRE-form `until(pred)` seen BEFORE its `repeat(body)` (`until(pred).repeat(…)`,
    /// while-do). Stashed here until the following `repeat()` opens and consumes it.
    pending_until: Option<Expr>,
    /// Whether `path()` can be answered as an INTERLEAVED node/edge chain: true while the
    /// traversal is `V`-source + explicit edge hops (`outE`/`inE`/`bothE`) and their
    /// `inV`/`outV`/`otherV` vertex moves (plus element filters) — a `[v0,e0,v1,e1,…]`
    /// path. A plain `out`/`in`/`both` (which hides its edge) or any other step breaks
    /// it. `path_has_edges` records that at least one edge hop occurred (so a bare
    /// vertex-only path still renders through the nodes-only path, not this one).
    edge_path_ok: bool,
    /// At least one explicit edge hop (`outE`/`inE`/`bothE`) occurred — the signal to
    /// render `path()` as the interleaved node/edge chain rather than nodes-only.
    path_has_edges: bool,
    /// A trailing `values('k')` on a vertex-hop chain that a following `tree()` folds
    /// in as its leaf level (`out(...).values('name').tree()`). The `values` step, when
    /// it sees `tree()` next, records the key here INSTEAD of projecting (which would
    /// drop the lineage), keeping the vertex frontier so tree() reads the node path.
    path_leaf: Option<String>,
    /// `path_ok` as it was at the START of the current step, before that step's taint —
    /// the values-into-tree check needs the pre-step value (the taint runs first).
    path_ok_pre_step: bool,
    /// Whether `path()` can still be answered: true while the traversal is a pure
    /// vertex-hop chain (`V`-source + `out`/`in`/`both` + element filters), whose
    /// Gremlin path is exactly the node sequence the engine's lineage records. Any
    /// other step (edge hops, var-length, value projections, barriers, the `E`
    /// source) makes the Gremlin path step-dependent in a way the nodes-only
    /// rendering would not match, so `path()` is deferred once this is false.
    path_ok: bool,
    /// Named side-effect bags for `aggregate`/`store` → revealed by `cap`. Each entry
    /// snapshots the plan PREFIX and the current-slot expression at the point the bag
    /// was filled, so `cap(key)` folds exactly that stream (matching the TS engine, where the
    /// bag holds the elements as they were at aggregate/store time, not after later
    /// value projections).
    caps: std::collections::HashMap<String, (Plan, Expr)>,
    /// OLAP annotate property name → the slot its computed value lands in (see
    /// `AlgoAnnotate`). A following `values(<property>)` reads that slot instead of a
    /// store property.
    algo_props: std::collections::HashMap<String, usize>,
    last_algo: Option<usize>,
    /// True when the current frontier holds EDGES (the `E` source, `EdgeSeed`, an
    /// `outE`/`inE`/`bothE` hop, or a coalesce/union of edge bodies) rather than nodes.
    /// A bare `inV`/`outV`/`bothV` off an edge frontier reads its endpoint; off a node
    /// frontier it is an error (a vertex move must follow an edge step).
    on_edge: bool,
    /// Set by `properties('k'…)`: the property keys the current element is a property
    /// STREAM over (the element stays current, present-filtered). A following
    /// `value()`/`key()`/`label()`/`hasValue()`/`count()` reads through this. `None`
    /// when not in a property stream; `Some(keys)` with the keys (empty = all present).
    prop_keys: Option<Vec<String>>,
    /// The FIRST slot each `as('x')` label was bound to (labels holds the LAST). A
    /// tag rebound across a hop has two bindings; `select(Pop.first, 'x')` reads this
    /// one, `select(Pop.last, 'x')` (the default) reads `labels`.
    first_labels: std::collections::HashMap<String, usize>,
    /// EVERY slot each `as('x')` label was bound to, in order — `select(Pop.all, 'x')`
    /// returns them all as a list.
    all_labels: std::collections::HashMap<String, Vec<usize>>,
    /// The slot carrying the per-traverser `sack` accumulator (a column appended by
    /// `withSack(init)`), or None when no sack is in play.
    sack_slot: Option<usize>,
    /// Named subgraph bags (`subgraph('sg')` → snapshot plan + the edge slot), revealed
    /// by `cap('sg')` as a `{vertices, edges}` Map.
    subgraph_caps: std::collections::HashMap<String, (Plan, usize)>,
    /// A `property()`-Update whose read tail we are now building — TinkerPop's
    /// `property(k, v).values(k)` (read-after-write). Set when a read step first follows
    /// a `property()` write: the write `(input, ops)` is stashed here and the working
    /// plan resets to `Row` (the seeded frontier) so the tail builds over it; `traversal`
    /// wraps the finished tail back into a [`Plan::UpdateReturn`].
    pending_write: Option<(Box<Plan>, Vec<crate::ir::SetOp>)>,
    /// True when the current traverser is a Map (a `project`/`group`/`valueMap`/…
    /// row), false on an element or scalar frontier. Gremlin's `select('k')` reads a
    /// Map ENTRY only on a Map; on a vertex/edge an unbound tag matches nothing and the
    /// traverser drops. Frontier-preserving steps (`filter`/`order`/`dedup`/…) leave it
    /// unchanged; every other step resets it, and the Map producers set it true.
    current_is_map: bool,
    /// True when the current frontier holds graph ELEMENTS (vertices or edges) rather
    /// than scalars/collections — known statically from the step chain. Set true by the
    /// frontier-move steps (V/E/out/in/both/*V/*E/addV/addE), false by every scalar/
    /// collection/map producer (values/id/label/count/aggregates/valueMap/…), and left
    /// unchanged by frontier-preserving filters/barriers. Read by `sum`/`min`/`max`/
    /// `mean` and bare `order()` to reject a bare-element aggregate/sort (TinkerPop:
    /// you cannot sum or order graph elements — `g.V().sum()` / `g.V().order()` throw).
    current_is_element: bool,
    /// True when the current frontier holds PATHS (`path()`) — a non-numeric,
    /// non-comparable sequence. Tracked like `current_is_element` and read by the
    /// same `sum`/`min`/`max`/`mean` guard: TinkerPop's `path().sum()` throws (a
    /// path is not a number). Preserving filters keep it; any producer that turns
    /// paths into scalars/elements (`unfold`/`values`/`count`/…) clears it.
    current_is_path: bool,
    /// True when the current frontier holds a DEFINITE scalar (a number/string/id) — set by
    /// the scalar producers `values`/`value`/`id`/`label`/`count`/`sum`/`min`/`max`/`mean`/
    /// `math`/`loops`/`inject`, cleared by every element / map / path / ambiguous producer.
    /// Distinct from `!current_is_element`, which also covers UNKNOWN frontiers (a
    /// `union`/`unfold`/`select` output that MIGHT be an element) where nothing must fault.
    /// Read by the element-type-algebra guard in [`step`] (adjacency/edge-hop/endpoint/
    /// projection on a scalar faults), mirroring the pure-TS engine's `Frontier` classifier.
    current_is_scalar: bool,
    /// Tri-state "is an EDGE in scope for an endpoint move (`inV`/`outV`/`bothV`/`otherV`)",
    /// a faithful port of the pure-TS engine's `hasEdge`: `Some(true)` an edge is reachable,
    /// `Some(false)` definitely none, `None` unknown (never faults). An edge source sets true,
    /// a vertex move / scalar producer sets false, a frontier-preserving filter/barrier keeps
    /// it, and a branch COMBINES its arms (any false → false, all true → true, else unknown).
    /// The endpoint guard in [`step`] faults on `Some(false)` — this is what catches an endpoint
    /// after a branch (`coalesce(count(), inE()).outV()`), where `current_is_element` is false.
    edge_scope: Option<bool>,
    /// The `edge_scope` at the END of the most recent [`parse_sub_body_seeded`] (captured before
    /// the flags restore), so a branch lowering can combine its arms' edge-in-scope.
    last_arm_edge_scope: Option<bool>,
    /// The output-frontier KIND flags `(is_element, is_scalar, is_path)` at the END of the most
    /// recent [`parse_sub_body_seeded`] (captured before the restore), so a branch whose body
    /// ALWAYS PRODUCES (`optional`, port of pure-TS `bodyAlwaysProduces`) can take the body's
    /// ending frontier as the post-branch frontier instead of clearing it to unknown.
    last_arm_frontier: (bool, bool, bool),
    /// The lowercase name of the LAST step in the most recent [`parse_sub_body_seeded`] body,
    /// so `optional` can test `bodyAlwaysProduces` on the literal terminal step (an agg reducer
    /// `count`/`fold`/`sum`/`min`/`max`/`mean`) exactly as pure-TS does — NOT the propagated
    /// `frontier_from_reducer`, which survives a following `limit`/`dedup` and would over-fault.
    last_arm_last_step: String,
    /// A branch's output is a clean element/scalar/path frontier only when EVERY arm agrees;
    /// mixed arms (`coalesce(outE, hasLabel)`) leave an UNKNOWN frontier so a following static
    /// type check (sum-over-element, values-on-scalar) does not fault (matches pure-TS).
    /// The previous step was a REDUCING barrier (`count`/`sum`/`min`/`max`/`mean`/`fold`).
    /// TinkerPop resets the path at a reducing barrier, so a following `path()` yields just
    /// `[reduced value]` (`count().path()` → `[7]`), not the pre-barrier traverser history.
    frontier_from_reducer: bool,
    /// True when the traversal reads a full `path()`/`tree()` — the lowering then emits a
    /// `Plan::PathRecord` after each value-producing step (see [`parse`]).
    building_full_path: bool,
    /// Current sub-traversal nesting depth, bounded by [`MAX_TRAVERSAL_DEPTH`] via
    /// [`Parser::nest`]. Each nested anonymous sub-traversal (`where(__.where(__.…))`)
    /// re-enters [`Parser::step`], so an unbounded query would otherwise overflow the
    /// native stack (SIGSEGV) or trap the wasm REPL. The [`crate::ir::Plan`] tree is also
    /// walked recursively downstream, so the cap protects those passes too.
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

    /// Run `f` one sub-traversal-nesting level deeper, rejecting past
    /// [`MAX_TRAVERSAL_DEPTH`] before the stack overflows. Wraps [`Parser::step`], so a
    /// flat step chain (each `step` returns before the next) never accumulates — only a
    /// nested sub-traversal (`where(__.where(…))`, which re-enters `step`) does.
    fn nest<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, String>) -> Result<T, String> {
        self.depth += 1;
        if self.depth > MAX_TRAVERSAL_DEPTH {
            self.depth -= 1;

            return Err(format!(
                "E_RESOURCE_EXHAUSTED: traversal nesting exceeds the maximum depth of {MAX_TRAVERSAL_DEPTH}"
            ));
        }
        let r = f(self);
        self.depth -= 1;

        r
    }

    fn expect(&mut self, t: &Tok) -> Result<(), String> {
        if self.peek() == Some(t) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {t:?} at token {}", self.pos))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            other => Err(format!("expected identifier, got {other:?}")),
        }
    }

    fn str_arg(&mut self) -> Result<String, String> {
        match self.bump() {
            Some(Tok::Str(s)) => Ok(s),
            other => Err(format!("expected a string argument, got {other:?}")),
        }
    }

    /// Parse the content of a `by(...)` group/select/project key: a `'prop'` string,
    /// or the `id`/`label` (also `T.id`/`T.label`) element token. Returns a display
    /// name and the value expression over `slot`.
    /// Parse ONE `order().by(<body>)` modulator's content (cursor just after `(`,
    /// leaves it AT the closing `)`): a property, a direction on the current value, an
    /// id/label/T token, or a degree sub-traversal `[__.]<hop>('L').count()`, each with
    /// an optional trailing `, asc|desc`. Returns `(sort_expr, descending)`.
    /// Whether a keying `by('k')` (group/groupCount/order/dedup/select) should install the
    /// absent-property filter (a no-value `by()` drops the traverser, per TinkerPop). TRUE
    /// on an ELEMENT or UNKNOWN frontier (e.g. after `unfold()`), where `by('k')` reads an
    /// element property and PROPERTY_EXISTS is the right absence test. FALSE on a MAP frontier
    /// (a `project()`/`group()` row — `by('k')` is a field read) or a definite SCALAR frontier
    /// (a Prop read there is not an element-property presence test), leaving those unchanged.
    fn by_absent_gates(&self) -> bool {
        !self.current_is_map && !self.current_is_scalar
    }

    fn order_by_body(&mut self, current: usize) -> Result<(Expr, bool), String> {
        // Empty by() → the current value, ascending.
        if self.peek() == Some(&Tok::RParen) {
            return Ok((Expr::Slot(current), false));
        }
        // A property key.
        if matches!(self.peek(), Some(Tok::Str(_))) {
            let key = self.str_arg()?;
            let descending = if self.peek() == Some(&Tok::Comma) {
                self.bump();
                self.order_dir()?
            } else {
                false
            };
            return Ok((Expr::Prop { slot: current, key }, descending));
        }
        // A bare direction (asc/desc/Order.*) on the current value.
        if matches!(self.peek(), Some(Tok::Ident(s)) if {
            let l = s.to_ascii_lowercase();
            l == "asc" || l == "desc" || l == "order"
        }) {
            let descending = self.order_dir()?;
            return Ok((Expr::Slot(current), descending));
        }
        // An id/label/T token.
        if matches!(self.peek(), Some(Tok::Ident(s)) if {
            let l = s.to_ascii_lowercase();
            l == "id" || l == "label" || l == "t"
        }) {
            let (_, e) = self.by_key_expr(current)?;
            let descending = if self.peek() == Some(&Tok::Comma) {
                self.bump();
                self.order_dir()?
            } else {
                false
            };
            return Ok((e, descending));
        }
        // A `select('k')` sub-traversal: the sort key is the entry `k` of the current
        // Map traverser (a `project()`/`group()` row). `select` on a Map projects the
        // entry — so `project(...).order().by(select('k'))` sorts by that field rather
        // than erroring. Byte-identical to the TS engine's `evalBy(select('k'))` on the Map.
        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("select")) {
            self.bump();
            self.expect(&Tok::LParen)?;
            let key = self.str_arg()?;
            self.expect(&Tok::RParen)?;
            let expr = Expr::Field {
                base: Box::new(Expr::Slot(current)),
                key,
            };
            let descending = if self.peek() == Some(&Tok::Comma) {
                self.bump();
                self.order_dir()?
            } else {
                false
            };
            return Ok((expr, descending));
        }
        // A degree sub-traversal: `[__.]<hop>('L'…).count()`.
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        let hop = self.ident()?.to_ascii_lowercase();
        let (dir, is_edge) = match hop.as_str() {
            "out" => (Dir::Out, false),
            "in" => (Dir::In, false),
            "both" => (Dir::Both, false),
            "oute" => (Dir::Out, true),
            "ine" => (Dir::In, true),
            "bothe" => (Dir::Both, true),
            other => {
                return Err(format!(
                    "order().by(<traversal>): unsupported body `{other}`"
                ))
            }
        };
        self.expect(&Tok::LParen)?;
        let mut labels: Vec<String> = Vec::new();
        if matches!(self.peek(), Some(Tok::Str(_))) {
            labels.push(self.str_arg()?);
            while self.peek() == Some(&Tok::Comma) {
                self.bump();
                labels.push(self.str_arg()?);
            }
        }
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Dot)?;
        let c = self.ident()?;
        if !c.eq_ignore_ascii_case("count") {
            return Err("order().by(<traversal>) body must end with .count()".into());
        }
        self.expect(&Tok::LParen)?;
        self.expect(&Tok::RParen)?;
        let body = if is_edge {
            Plan::Row.expand_edge_gremlin(current, dir, &labels)
        } else {
            Plan::Row.expand(current, dir, &labels)
        };
        let expr = Expr::CountSubquery {
            body: Box::new(body),
            outer_width: self.slots,
        };
        let descending = if self.peek() == Some(&Tok::Comma) {
            self.bump();
            self.order_dir()?
        } else {
            false
        };
        Ok((expr, descending))
    }

    /// Parse a simple comparison predicate `[P.]op(literal)` → (op, value). Used where
    /// only eq/neq/gt/gte/lt/lte against one literal is meaningful (e.g. `none(pred)`).
    fn simple_predicate(&mut self) -> Result<(CompareOp, Value), String> {
        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("P")) {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        let op_name = self.ident()?.to_ascii_lowercase();
        let op = match op_name.as_str() {
            "eq" => CompareOp::Eq,
            "neq" => CompareOp::Ne,
            "gt" => CompareOp::Gt,
            "gte" => CompareOp::Ge,
            "lt" => CompareOp::Lt,
            "lte" => CompareOp::Le,
            other => return Err(format!("expected a comparison predicate, got `{other}`")),
        };
        self.expect(&Tok::LParen)?;
        let val = self.literal()?;
        self.expect(&Tok::RParen)?;
        Ok((op, val))
    }

    /// Consume an optional empty `()` — `by(label())` vs the bare token `by(label)`.
    fn eat_empty_parens(&mut self) {
        if self.peek() == Some(&Tok::LParen) && self.toks.get(self.pos + 1) == Some(&Tok::RParen) {
            self.bump();
            self.bump();
        }
    }

    fn by_key_expr(&mut self, slot: usize) -> Result<(String, Expr), String> {
        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("T")) {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        match self.peek().cloned() {
            Some(Tok::Str(k)) => {
                self.bump();
                Ok((k.clone(), Expr::Prop { slot, key: k }))
            }
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("id") => {
                self.bump();
                self.eat_empty_parens();
                Ok((
                    "id".into(),
                    Expr::Call {
                        name: "element_id".into(),
                        args: vec![Expr::Slot(slot)],
                    },
                ))
            }
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("label") => {
                self.bump();
                self.eat_empty_parens();
                Ok((
                    "label".into(),
                    Expr::Call {
                        name: "element_label".into(),
                        args: vec![Expr::Slot(slot)],
                    },
                ))
            }
            other => Err(format!(
                "by(...): expected a key or id/label token, got {other:?}"
            )),
        }
    }

    /// Parse the body of a `project(...).by(<body>)` modulator (the cursor sits just
    /// after the opening `(`; leaves it AT the closing `)`). Supports the same key/id/
    /// label forms as [`by_key_expr`], the bare `by()` (the element itself), and a
    /// degree sub-traversal `[__.](out|in|both|outE|inE|bothE)('L'…).count()` — a
    /// correlated `CountSubquery`, the only nested body the TS engine's projections use here.
    fn project_by_body(&mut self, elem_slot: usize) -> Result<Expr, String> {
        // bare by() → the element itself.
        if self.peek() == Some(&Tok::RParen) {
            return Ok(Expr::Slot(elem_slot));
        }
        // by('key') / by(id) / by(label) / by(T.id) / by(T.label).
        let token_form = matches!(self.peek(), Some(Tok::Str(_)))
            || matches!(self.peek(), Some(Tok::Ident(s)) if {
                let l = s.to_ascii_lowercase();
                l == "id" || l == "label" || l == "t"
            });
        if token_form {
            return Ok(self.by_key_expr(elem_slot)?.1);
        }
        // A correlated sub-traversal body — one navigating hop, then a reduction:
        //   `hop('L'…).count()`                → scalar CountSubquery
        //   `hop('L'…).fold()`                 → list CollectSubquery of the elements
        //   `hop('L'…).values('k').fold()`     → list CollectSubquery of a property
        // Built manually (rather than via parse_sub_body) so the CollectSubquery's
        // provenance column survives: an inner `values(...)` PROJECTS the body down to
        // a single column, which would drop the provenance the correlated collect needs.
        let width = self.slots;
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        // Element accessors as a sub-traversal: `by(__.id())` / `by(__.label())`.
        if matches!(self.peek(), Some(Tok::Ident(s)) if {
            let l = s.to_ascii_lowercase();
            (l == "id" || l == "label") && self.toks.get(self.pos + 1) == Some(&Tok::LParen)
        }) {
            let acc = self.ident()?.to_ascii_lowercase();
            self.eat_empty_parens();
            let func = if acc == "id" {
                "element_id"
            } else {
                "element_label"
            };
            return Ok(Expr::Call {
                name: func.into(),
                args: vec![Expr::Slot(elem_slot)],
            });
        }
        let hop = self.ident()?.to_ascii_lowercase();
        let (dir, is_edge) = match hop.as_str() {
            "out" => (Dir::Out, false),
            "in" => (Dir::In, false),
            "both" => (Dir::Both, false),
            "oute" => (Dir::Out, true),
            "ine" => (Dir::In, true),
            "bothe" => (Dir::Both, true),
            other => {
                return Err(format!(
                    "project().by(<traversal>): only a single-hop reducing body is supported, got `{other}`"
                ))
            }
        };
        self.expect(&Tok::LParen)?;
        let mut labels: Vec<String> = Vec::new();
        if matches!(self.peek(), Some(Tok::Str(_))) {
            labels.push(self.str_arg()?);
            while self.peek() == Some(&Tok::Comma) {
                self.bump();
                labels.push(self.str_arg()?);
            }
        }
        self.expect(&Tok::RParen)?;
        // The neighbour lands one past the provenance column (inserted at `width`).
        let landed = width + 1;
        let mut body = if is_edge {
            Plan::Row.expand_edge_gremlin(elem_slot, dir, &labels)
        } else {
            Plan::Row.expand(elem_slot, dir, &labels)
        };
        self.expect(&Tok::Dot)?;
        let reducer = self.ident()?.to_ascii_lowercase();
        // Optional `.values('k')` between the hop and `.fold()`.
        let mut val_key: Option<String> = None;
        let reducer = if reducer == "values" {
            self.expect(&Tok::LParen)?;
            val_key = Some(self.str_arg()?);
            self.expect(&Tok::RParen)?;
            self.expect(&Tok::Dot)?;
            self.ident()?.to_ascii_lowercase()
        } else {
            reducer
        };
        self.expect(&Tok::LParen)?;
        self.expect(&Tok::RParen)?;
        match reducer.as_str() {
            "count" if val_key.is_none() => Ok(Expr::CountSubquery {
                body: Box::new(body),
                outer_width: width,
            }),
            "fold" => {
                let scalar = match val_key {
                    Some(key) => {
                        // Keep only neighbours that HAVE the property (values() skips
                        // absent), then collect its value.
                        body = body.filter(Expr::PropertyExists {
                            slot: landed,
                            key: key.clone(),
                        });
                        Expr::Prop { slot: landed, key }
                    }
                    None => Expr::Slot(landed),
                };
                Ok(Expr::CollectSubquery {
                    body: Box::new(body),
                    scalar: Box::new(scalar),
                    outer_width: width,
                })
            }
            other => Err(format!(
                "project().by(<traversal>): reducing body must end in count()/fold(), got `{other}`"
            )),
        }
    }

    // traversal := 'g' '.' ( 'V' '(' ')' | 'addV' '(' Label ')' ) ( '.' step )*
    fn traversal(&mut self) -> Result<Plan, String> {
        let g = self.ident()?;
        if !g.eq_ignore_ascii_case("g") {
            return Err(format!("expected `g`, got `{g}`"));
        }
        self.expect(&Tok::Dot)?;
        // Source-config prefixes before the real head: `g.withSack(init).V()…`
        // (seed the sack after the source is built) and `g.withComputer().V()…`
        // (a no-op marker). Multiple may chain.
        let mut sack_init: Option<Value> = None;
        loop {
            match self.peek() {
                Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("withSack") => {
                    self.bump();
                    self.expect(&Tok::LParen)?;
                    sack_init = Some(self.literal()?);
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                }
                Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("withComputer") => {
                    self.bump();
                    self.expect(&Tok::LParen)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                }
                _ => break,
            }
        }
        let head = self.ident()?;
        let mut plan = match head.to_ascii_lowercase().as_str() {
            "v" => {
                self.expect(&Tok::LParen)?;
                // A vertex source has NO edge in scope, so a following endpoint (inV/outV/…)
                // faults — set it here (the source isn't run through the step classifier).
                self.edge_scope = Some(false);
                // Three head shapes:
                //   g.V()               — all vertices (Scan).
                //   g.V('a', 'b', …)    — the vertices with those EXTERNAL ids, a
                //                         read source resolved at exec time (NodeSeed).
                //   g.V(<num>).addE(…)  — the numeric-id anchor of an addE write.
                if matches!(self.peek(), Some(Tok::Num(_))) {
                    let from = self.u_id()?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                    let step = self.ident()?;
                    if !step.eq_ignore_ascii_case("addE") {
                        return Err("g.V(<numeric id>) is only supported before addE()".into());
                    }
                    self.expect(&Tok::LParen)?;
                    let etype = self.str_arg()?;
                    self.expect(&Tok::RParen)?;
                    self.finish_add_edge(Some(from), None, etype)?
                } else if matches!(self.peek(), Some(Tok::Str(_))) {
                    let mut ext_ids = vec![self.str_arg()?];
                    while self.peek() == Some(&Tok::Comma) {
                        self.bump();
                        ext_ids.push(self.str_arg()?);
                    }
                    self.expect(&Tok::RParen)?;
                    Plan::NodeSeed { ext_ids }
                } else {
                    self.expect(&Tok::RParen)?;
                    Plan::Scan { label: None }
                }
            }
            "e" => {
                self.expect(&Tok::LParen)?;
                // `g.E()` seeds every live edge; `g.E('id', …)` seeds the edges with
                // those external ids (in request order).
                self.path_ok = false;
                if matches!(self.peek(), Some(Tok::Str(_))) {
                    let mut ext_ids = vec![self.str_arg()?];
                    while self.peek() == Some(&Tok::Comma) {
                        self.bump();
                        ext_ids.push(self.str_arg()?);
                    }
                    self.expect(&Tok::RParen)?;
                    self.on_edge = true;
                    self.edge_scope = Some(true);
                    Plan::EdgeSeed { ext_ids }
                } else {
                    self.expect(&Tok::RParen)?;
                    // An edge source makes the path start on an edge, not a node.
                    self.on_edge = true;
                    self.edge_scope = Some(true);
                    Plan::EdgeScan
                }
            }
            "adde" => {
                // g.addE('T').from(V(a)).to(V(b)).property(...)
                self.expect(&Tok::LParen)?;
                let etype = self.str_arg()?;
                check_write_name("edge label", &etype)?;
                self.expect(&Tok::RParen)?;
                self.finish_add_edge(None, None, etype)?
            }
            "addv" => {
                // addV('Label') creates one vertex; following property() steps
                // fold into it (see `apply_property`).
                self.expect(&Tok::LParen)?;
                let label = self.str_arg()?;
                check_write_name("vertex label", &label)?;
                self.expect(&Tok::RParen)?;
                Plan::Insert {
                    nodes: vec![crate::ir::InsertNode {
                        labels: vec![label],
                        props: vec![],
                    }],
                    edges: vec![],
                }
            }
            "inject" => {
                // g.inject(v1, v2, …): a SOURCE that seeds the stream with the literal
                // values — an unwind of the value list over a single Row.
                self.expect(&Tok::LParen)?;
                let vals = self.literal_list()?;
                self.expect(&Tok::RParen)?;
                self.current = 0;
                self.slots = 1;
                Plan::Unwind {
                    input: Box::new(Plan::Row),
                    list: Box::new(Expr::Lit(Value::List(vals))),
                    var_slot: 1,
                    ordinal: None,
                }
                .project(vec![("inject".to_string(), Expr::Slot(1))])
            }
            other => return Err(format!("expected V() or addV(...), got `{other}`")),
        };
        // Seed the sack accumulator as an appended column carried alongside the
        // element frontier (a MapSlot append, so the node frontier is preserved).
        if let Some(init) = sack_init {
            let slot = self.slots;
            plan = plan.map_slot(slot, Expr::Lit(init), true);
            self.slots += 1;
            self.sack_slot = Some(slot);
        }
        // The traversal SOURCE sets the initial frontier kind: `V`/`E`/`addV` land on
        // graph ELEMENTS, `inject` (and anything else) on scalars. Each step updates it.
        self.current_is_element = matches!(head.to_ascii_lowercase().as_str(), "v" | "e" | "addv");
        while self.peek() == Some(&Tok::Dot) {
            self.pos += 1;
            let step_name = match self.peek() {
                Some(Tok::Ident(s)) => s.to_ascii_lowercase(),
                _ => String::new(),
            };
            plan = self.step(plan)?;
            plan = self.maybe_path_record(plan, &step_name);
        }
        plan = self.flush_repeat(plan)?;
        if self.pos != self.toks.len() {
            return Err(format!("unexpected trailing input at token {}", self.pos));
        }
        // A bare traversal that ends on a hop (no terminal projection) leaves the
        // current element in a slot > 0, but the result renderer emits slot 0 (the
        // source). Project the current element into slot 0 so the output is the
        // traverser's element — `g.V('3').in('CREATED')` yields the neighbours, not
        // three copies of the source.
        // The element slot BEFORE the render projection below resets it — the element a
        // terminal `property()` must emit lives here in the write frontier.
        let elem_slot = self.current;
        // A TERMINAL write EMITS its created/mutated element (TinkerPop). Convert it to
        // the *Return variant with an element-projecting tail BEFORE the render
        // projection — otherwise that projection wraps the write in a Project and shallow
        // `is_write` runs the whole thing as a READ (see the drop() finalization). `drop`
        // (a Delete Update) stays terminal and emits nothing.
        plan = match plan {
            // Only when the frontier is an ELEMENT — `property()` on a non-element
            // (`inject(5).property(...)`) drops it (emits nothing), so leave the bare
            // Update, which no-ops and returns empty.
            Plan::Update { input, ops }
                if self.current_is_element
                    && !ops
                        .iter()
                        .any(|o| matches!(o, crate::ir::SetOp::Delete { .. })) =>
            {
                Plan::UpdateReturn {
                    input,
                    ops,
                    tail: Box::new(
                        Plan::Row.project(vec![("_".to_string(), Expr::Slot(elem_slot))]),
                    ),
                }
            }
            // addV (props folded in) — the InsertReturn seed binds the created node at
            // slot 0, so the tail projects it.
            Plan::Insert { nodes, edges } => Plan::InsertReturn {
                nodes,
                edges,
                tail: Box::new(Plan::Row.project(vec![("_".to_string(), Expr::Slot(0))])),
            },
            other => other,
        };
        if self.current != 0
            && !matches!(plan, Plan::UpdateReturn { .. } | Plan::InsertReturn { .. })
        {
            plan = plan.project(vec![("_".to_string(), Expr::Slot(self.current))]);
            self.current = 0;
            self.slots = 1;
        }
        // A folded `property()`-Update read tail (`property(k,v).values(k)`): wrap it
        // back into an UpdateReturn so the writes run, then `plan` reads them over the
        // frontier (read-after-write).
        if let Some((input, ops)) = self.pending_write.take() {
            plan = Plan::UpdateReturn {
                input,
                ops,
                tail: Box::new(plan),
            };
        }
        Ok(plan)
    }

    /// Parse a run of `.by(...)` modulators after `path()`/`tree()` into cycled path
    /// projectors: `by('k')` → the element's property, `by(id)` / `by(label)` (bare, `T.`- or
    /// `__.`-qualified) → its id / label. Empty when there is no `by`.
    fn parse_path_bys(&mut self) -> Result<Vec<crate::ir::GPathBy>, String> {
        let mut bys: Vec<crate::ir::GPathBy> = Vec::new();
        while self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
        {
            self.expect(&Tok::Dot)?;
            self.ident()?; // by
            self.expect(&Tok::LParen)?;
            if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                self.bump();
                self.expect(&Tok::Dot)?;
            }
            if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("T")) {
                self.bump();
                self.expect(&Tok::Dot)?;
            }
            let by = match self.peek() {
                Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("id") => {
                    self.bump();
                    self.eat_empty_parens();
                    crate::ir::GPathBy::Id
                }
                Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("label") => {
                    self.bump();
                    self.eat_empty_parens();
                    crate::ir::GPathBy::Label
                }
                _ => crate::ir::GPathBy::Prop(self.str_arg()?),
            };
            self.expect(&Tok::RParen)?;
            bys.push(by);
        }
        Ok(bys)
    }

    fn step(&mut self, plan: Plan) -> Result<Plan, String> {
        // Every nested sub-traversal re-enters here; the depth guard bounds nesting
        // before the parser (and the downstream Plan walkers) overflow the stack.
        self.nest(move |p| p.step_inner(plan))
    }

    fn step_inner(&mut self, plan: Plan) -> Result<Plan, String> {
        let name = self.ident()?;
        self.expect(&Tok::LParen)?;
        let lname = name.to_ascii_lowercase();
        // An edge hop's landed endpoint is only reachable by the vertex move that
        // IMMEDIATELY follows it; consume the record here so any other step clears it.
        let prev_edge_hop = self.edge_hop.take();
        // TinkerPop: an ELEMENT step applied straight to a PATH throws (an ImmutablePath
        // is not an Element/Vertex/Edge). `unfold()` turns the path into its elements and
        // `count`/`fold`/… consume it; `range`/`order`/… preserve it (the current_is_path
        // classifier tracks that), so the element step is only rejected while the frontier
        // is still a path.
        if self.current_is_path
            && matches!(
                lname.as_str(),
                "out"
                    | "in"
                    | "both"
                    | "oute"
                    | "ine"
                    | "bothe"
                    | "inv"
                    | "outv"
                    | "bothv"
                    | "otherv"
                    | "values"
                    | "value"
                    | "valuemap"
                    | "propertymap"
                    | "properties"
                    | "property"
                    | "key"
                    | "id"
                    | "label"
                    | "has"
                    | "hasnot"
                    | "haslabel"
                    | "hasid"
                    | "haskey"
                    | "hasvalue"
            )
        {
            return Err(format!(
                "{lname}() is not defined on a path — a path is not an element; \
                 unfold() it into its elements first"
            ));
        }
        // Element-type algebra — the SAME static rejection the pure-TS engine applies (and
        // TinkerPop's runtime ClassCastException, verified against gremlin-console): the
        // INCOMING frontier (before this step) must match what the step consumes. A scalar
        // frontier is `current_is_scalar`; an UNKNOWN frontier (a union/unfold/select output,
        // `!element && !scalar && !map && !path`) never faults — a missed fault is safe, a
        // false one breaks a valid query.
        {
            // A DEFINITE edge frontier for ADJACENCY (out/outE on an edge faults). After a
            // branch `current_is_element` is cleared to "unknown", so this is false there even
            // for an all-edge branch — matching pure-TS, which runs `bothE` after
            // `optional(...)` over edges and yields [] rather than faulting. `on_edge` alone
            // still stays true for the ENDPOINT read off a reconverged edge.
            let on_edge_frontier = self.current_is_element && self.on_edge;
            let on_scalar_frontier = self.current_is_scalar;
            match lname.as_str() {
                // Adjacency + edge hops navigate FROM a vertex.
                "out" | "in" | "both" | "oute" | "ine" | "bothe"
                    if on_edge_frontier || on_scalar_frontier =>
                {
                    return Err(format!(
                        "{lname}() moves from a vertex, but the frontier is {} — {} before {lname}()",
                        if on_edge_frontier { "an edge" } else { "a scalar" },
                        if on_edge_frontier {
                            "use an endpoint step (inV()/outV()/otherV())"
                        } else {
                            "project to a vertex"
                        },
                    ));
                }
                // Endpoints move to an edge's endpoint, so they need an edge DEFINITELY in
                // scope (`edge_scope == Some(false)` — a vertex/scalar frontier, or a branch
                // whose arms all lack one, e.g. `coalesce(count(), inE()).outV()`). An unknown
                // scope (`None`) never faults, matching the pure-TS `hasEdge === false` guard.
                "inv" | "outv" | "bothv" | "otherv" if self.edge_scope == Some(false) => {
                    return Err(format!(
                        "{lname}() moves to an edge endpoint, but no edge is in scope — reach an \
                         edge (outE()/inE()/bothE()) before {lname}()"
                    ));
                }
                // Projections read a property/id/label off an ELEMENT.
                "values" | "value" | "id" | "label" | "properties" | "propertymap" | "valuemap"
                | "elementmap" | "key"
                    if on_scalar_frontier =>
                {
                    return Err(format!(
                        "{lname}() reads from a graph element, but the frontier is a projected \
                         scalar (values()/id()/label()/count()/inject()); it has no such value"
                    ));
                }
                _ => {}
            }
        }
        // A pending `repeat` stays open across its modulators (times/emit/until); any
        // other step flushes it into a VarLength walk first.
        let plan = if self.pending_repeat.is_some()
            && !matches!(lname.as_str(), "times" | "emit" | "until")
        {
            self.flush_repeat(plan)?
        } else {
            plan
        };
        // The pre-taint value — a `values('k')` that folds into a following `tree()`
        // needs to know it was still on a pure vertex chain BEFORE this step's taint.
        self.path_ok_pre_step = self.path_ok;
        // Whether the INCOMING frontier was a Map — read by `select` (which consumes the
        // Map to project an entry) before the reset below clobbers it.
        let incoming_is_map = self.current_is_map;
        // The current-is-Map flag survives only the frontier-preserving steps (they
        // filter/reorder the same traversers); every other step recomputes it — a hop or
        // scalar read clears it here, a Map producer sets it true in its own arm below.
        if !matches!(
            lname.as_str(),
            "has"
                | "hasnot"
                | "hasid"
                | "haslabel"
                | "hasnotlabel"
                | "haskey"
                | "hasvalue"
                | "where"
                | "filter"
                | "is"
                | "and"
                | "or"
                | "not"
                | "dedup"
                | "order"
                | "by"
                | "limit"
                | "tail"
                | "skip"
                | "range"
                | "as"
                | "identity"
                | "barrier"
                | "aggregate"
                | "store"
                | "sideeffect"
                | "sample"
                | "coin"
                | "none"
                | "cyclicpath"
                | "simplepath"
                | "profile"
        ) {
            self.current_is_map = false;
        }
        // Only a pure vertex-hop chain keeps `path()` answerable; every other step
        // taints it (`path()` and the element filters are path-preserving). The repeat
        // modulators are exempt: `repeat(<vertex-hop>)` lowers to a VarLength walk that
        // records full path lineage (flush_repeat restores `path_ok` to its pre-repeat
        // value), so a following `path()`/`tree()` stays answerable.
        if !matches!(
            lname.as_str(),
            "out"
                | "in"
                | "both"
                | "has"
                | "haslabel"
                | "hasnot"
                | "hasid"
                | "and"
                | "or"
                | "not"
                | "path"
                | "tree"
                | "simplepath"
                | "cyclicpath"
                | "repeat"
                | "times"
                | "emit"
                | "until"
        ) {
            self.path_ok = false;
        }
        // The INTERLEAVED edge-path (`outE().inV()…path()`) stays answerable through the
        // explicit edge hops and their vertex moves plus element filters; anything else
        // (a plain vertex hop, a value projection, a barrier) breaks the alternation.
        if !matches!(
            lname.as_str(),
            "oute" | "ine" | "bothe"
                | "inv" | "outv" | "otherv" | "bothv"
                | "has" | "haslabel" | "hasnot" | "hasid"
                | "and" | "or" | "not"
                | "path" | "tree" | "simplepath" | "cyclicpath"
                // coalesce decides its own edge_path_ok in-arm (an edge-yielding body
                // keeps it; a non-edge one clears it), so exempt it from this taint.
                | "coalesce"
        ) {
            self.edge_path_ok = false;
        }
        if matches!(lname.as_str(), "oute" | "ine" | "bothe") {
            self.path_has_edges = true;
        }

        // --- write steps ---
        if lname == "property" {
            let key = self.str_arg()?;
            check_write_name("property key", &key)?;
            self.expect(&Tok::Comma)?;
            let val = self.property_value_expr()?;
            self.expect(&Tok::RParen)?;
            return Ok(self.apply_property(plan, key, val));
        }
        if lname == "drop" {
            self.expect(&Tok::RParen)?;
            if is_write(&plan) {
                return Err("drop() cannot follow a write step".into());
            }
            // Delete the current elements of the traversal. `drop()` is TERMINAL and
            // emits nothing, so reset `current` to 0: otherwise the finalizer wraps this
            // Update in a render Project (for an element left in slot != 0, e.g. the edge
            // from `outE()`), and shallow `is_write(Project(Update))` would miss the
            // write — running `outE().drop()` as a read that deletes nothing.
            let del_slot = self.current;
            self.current = 0;
            return Ok(Plan::Update {
                input: Box::new(plan),
                // Gremlin drop() removes the element AND its incident edges.
                ops: vec![crate::ir::SetOp::Delete {
                    slot: del_slot,
                    detach: true,
                }],
            });
        }
        // TinkerPop: `property()` is NOT terminal — it returns the mutated element, so
        // read steps may follow and observe the just-written values (`property(k,v)
        // .values(k)`). Fold the pending property()-Update aside: stash the write and
        // continue the read tail over the (unchanged) frontier, seeded from `Row`;
        // `traversal` wraps it back into an UpdateReturn. `addV`/`addE`/`drop` stay
        // terminal for reads (a created/dropped element has no read frontier here).
        let is_property_update = matches!(&plan, Plan::Update { ops, .. }
            if ops.iter().all(|op| !matches!(op, crate::ir::SetOp::Delete { .. })));
        let plan = if is_property_update && self.pending_write.is_none() {
            let Plan::Update { input, ops } = plan else {
                unreachable!()
            };
            self.pending_write = Some((input, ops));
            Plan::Row
        } else if is_write(&plan) {
            // drop() (an Update of Deletes) and addV/addE/Merge stay terminal for reads.
            return Err(format!("step `{lname}` cannot follow a write step"));
        } else {
            plan
        };

        let plan = match lname.as_str() {
            "haslabel" => {
                let mut labels = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    labels.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                let cur = self.current;
                // A single label on a still-bare scan folds in (the efficient form
                // right after V()); otherwise it is a runtime label-membership filter,
                // so `hasLabel` works after a hop and with multiple labels too.
                if labels.len() == 1 && matches!(plan, Plan::Scan { label: None }) {
                    Plan::Scan {
                        label: Some(labels.remove(0)),
                    }
                } else {
                    plan.filter(label_membership(cur, &labels))
                }
            }
            "has" => {
                let first = self.str_arg()?;
                if self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    // has(label, key, pred) — the 3-arg form — is a label check AND a
                    // property predicate. Detect it: a string key followed by another
                    // comma (has(k, pred) has no comma after its predicate).
                    if matches!(self.peek(), Some(Tok::Str(_)))
                        && self.toks.get(self.pos + 1) == Some(&Tok::Comma)
                    {
                        let key = self.str_arg()?;
                        self.expect(&Tok::Comma)?;
                        let prop = self.has_predicate(key)?;
                        self.expect(&Tok::RParen)?;
                        let label = label_membership(self.current, &[first]);
                        plan.filter(Expr::And(Box::new(label), Box::new(prop)))
                    } else {
                        // has(k, pred) — a value predicate on property `first`.
                        let pred = self.has_predicate(first)?;
                        self.expect(&Tok::RParen)?;
                        plan.filter(pred)
                    }
                } else {
                    // has(k) — key EXISTENCE.
                    self.expect(&Tok::RParen)?;
                    plan.filter(Expr::PropertyExists {
                        slot: self.current,
                        key: first,
                    })
                }
            }
            // hasId('a', …): keep the element iff its EXTERNAL id is one of the given
            // ids — an `element_id`-in-list predicate (an OR of equals).
            "hasid" => {
                let mut ids = vec![Value::Str(self.str_arg()?.into())];
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    ids.push(Value::Str(self.str_arg()?.into()));
                }
                self.expect(&Tok::RParen)?;
                let left = Expr::Call {
                    name: "element_id".to_string(),
                    args: vec![Expr::Slot(self.current)],
                };
                plan.filter(or_of_equals(&left, &ids))
            }
            // hasNot(k): the element must NOT carry property `k`.
            "hasnot" => {
                // hasNot('k') keeps elements WITHOUT key k; variadic hasNot('a','b')
                // keeps those with NONE of the keys (AND of the negations).
                let mut keys = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    keys.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                let mut pred: Option<Expr> = None;
                for key in keys {
                    let neg = Expr::Not(Box::new(Expr::PropertyExists {
                        slot: self.current,
                        key,
                    }));
                    pred = Some(match pred {
                        None => neg,
                        Some(p) => Expr::And(Box::new(p), Box::new(neg)),
                    });
                }
                plan.filter(pred.expect("hasNot has at least one key"))
            }
            "haskey" => {
                // Keep elements that HAVE any of the listed property keys.
                let mut keys = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    keys.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                let mut pred: Option<Expr> = None;
                for k in keys {
                    let pe = Expr::PropertyExists {
                        slot: self.current,
                        key: k,
                    };
                    pred = Some(match pred {
                        None => pe,
                        Some(p) => Expr::Or(Box::new(p), Box::new(pe)),
                    });
                }
                plan.filter(pred.expect("hasKey needs a key"))
            }
            "union" => {
                // union(<hop>, <hop>, …): for each element, concatenate every branch's
                // frontier. v1 scopes each branch to a single out/in/both hop off the
                // current element; all branches land their neighbour at the same slot,
                // so the union is a continuable node frontier.
                let from = self.current;
                let width = self.slots;
                // Seed the arms with the outer edge hop so a leading `otherV()`/`inV()`
                // off `V().outE()` resolves against its origin (see parse_sub_body_seeded).
                self.edge_hop = prev_edge_hop;
                let mut bodies = Vec::new();
                let mut arm_edge_scopes: Vec<Option<bool>> = Vec::new();
                loop {
                    // union() runs each arm over the WHOLE incoming stream (the branch body
                    // is seeded with the full batch, not per element), so a reducing barrier
                    // in an arm — `count()`, `limit(1)`, `fold()` — reduces the whole stream:
                    // `union(out().count(), in().count())` yields the TOTAL out- and in-degree,
                    // not per-vertex, matching TinkerPop.
                    let (body, oc, _os) = self.parse_sub_body(from, width)?;
                    // Reconverge every arm to a UNIFORM width-1 frontier with its element at
                    // slot 0 — a 2-hop arm beside a 1-hop arm otherwise lands its element at a
                    // different slot, and a downstream `values()` would read a mid-hop node.
                    // Lineage-preserving, so path()-through-union still works.
                    bodies.push(body.reconverge(oc));
                    arm_edge_scopes.push(self.last_arm_edge_scope);
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                self.current = 0;
                self.slots = 1;
                self.edge_hop = None; // the reconverged frontier is not a just-hopped edge
                self.edge_scope = combine_edge_scope(&arm_edge_scopes);
                // Post-branch VALUE + ADJACENCY frontier is unknown (see coalesce); only
                // edge_scope (endpoints) and an all-path frontier propagate.
                self.current_is_element = false;
                self.current_is_scalar = false;
                self.current_is_path = false;
                self.on_edge = self.edge_scope == Some(true);
                plan.branch(bodies)
            }
            "optional" => {
                // optional(<body>) = coalesce(<body>, identity): the body's frontier for
                // elements it produces output for, else the SOURCE element unchanged. The
                // body arm runs unconditionally; the fallback keeps the source where the
                // body is empty (NOT EXISTS body). A full sub-traversal, reconverging like
                // union — so `optional(out('a').out('b'))` (multi-hop) works, not just a
                // single hop.
                let from = self.current;
                let slots = self.slots;
                let incoming_edge_scope = self.edge_scope; // the fallback keeps the source
                                                           // Seed the arm with the outer edge hop (see parse_sub_body_seeded) so a
                                                           // leading `otherV()`/`inV()` off `V().outE()` resolves against its origin.
                self.edge_hop = prev_edge_hop;
                // An aggregate-terminal body ALWAYS produces one value per element, so the
                // identity fallback never fires — lower straight to the per-element
                // aggregate projection (`optional(count())` → `[1,1,…]`, not the whole-
                // stream `[6, …source…]`). NOT over a PATH or SCALAR frontier: the fast path
                // builds the aggregate without running the body's leading step through `step()`,
                // so it would skip the frontier fault — a hop on a path (`path().optional(
                // both('KNOWS').count())`) or a `values()`/hop on a scalar (`id().optional(
                // values('age').count())`) must be rejected. Route those to the general body
                // below, which parses each step through `step()` and faults (a bare `count()`
                // over a scalar does NOT fault there, so `id().optional(count())` still works).
                if !self.current_is_path && !self.current_is_scalar {
                    if let Some(agg) = self.try_per_element_agg(from, slots)? {
                        self.expect(&Tok::RParen)?;
                        let p = plan.project(vec![("optional".to_string(), agg)]);
                        self.current = 0;
                        self.slots = 1;
                        self.edge_hop = None;
                        self.edge_scope = Some(false); // an aggregate body yields a scalar
                        self.current_is_scalar = true; // matches TS bodyAlwaysProduces → a scalar
                        self.current_is_element = false;
                        // A reducer RESETS the traverser path, so a following `path()` is the
                        // reduced value (`optional(count()).order().range(0,2).path()` → `[[1],…]`,
                        // not the pre-reduce vertex history).
                        self.frontier_from_reducer = true;
                        return Ok(p);
                    }
                }
                // A body that ALWAYS produces exactly one output per element — a single
                // `id()`/`label()`/`path()` projection (no hop or filter to drop a row) —
                // makes the identity fallback dead: `optional(id()) ≡ id()`, `optional(path())
                // ≡ path()`. Lower to just the body, else the Exists guard over a projecting
                // body wrongly reports "empty" and the fallback double-emits (`limit(3).
                // optional(id())` returned the ids AND the vertices; `optional(path()).count()`
                // was 14, not 7).
                {
                    let mut p = self.pos;
                    if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
                        p += 1;
                        if self.toks.get(p) == Some(&Tok::Dot) {
                            p += 1;
                        }
                    }
                    let inner_name = match self.toks.get(p) {
                        Some(Tok::Ident(s)) => s.to_ascii_lowercase(),
                        _ => String::new(),
                    };
                    let is_always_scalar = matches!(inner_name.as_str(), "id" | "label" | "path")
                        && self.toks.get(p + 1) == Some(&Tok::LParen)
                        && self.toks.get(p + 2) == Some(&Tok::RParen)
                        && self.toks.get(p + 3) == Some(&Tok::RParen); // arm-terminal `)`
                    if is_always_scalar {
                        // Parse the single projection step straight onto the frontier.
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        let body = self.step(plan)?;
                        self.expect(&Tok::RParen)?;
                        // The body IS the frontier (fallback dead), so its projected value must be
                        // recorded into the step-history exactly as the bare step would be — else a
                        // following `path()` misses it (`optional(path()).path()` failed to nest the
                        // inner path; `optional(id()).path()` dropped the id element).
                        return Ok(self.maybe_path_record(body, &inner_name));
                    }
                }
                // General case: run the body PER ELEMENT (a single incoming traverser at a
                // time), emitting its output where it produces and the source element where
                // it does not — TinkerPop's per-traverser `optional`. Running per element is
                // what makes a barrier inside the body (`limit(2)`, `skip(1)`, `dedup()`)
                // apply to THAT element's sub-stream rather than the whole batch, which the
                // old whole-stream EXISTS-guarded branch got wrong.
                let (body, oc, _os) = self.parse_sub_body_seeded(Plan::Row, from, slots)?;
                let body_edge_scope = self.last_arm_edge_scope;
                let body_frontier = self.last_arm_frontier;
                // `bodyAlwaysProduces` (port of pure-TS): a body whose LITERAL last step is an
                // aggregate reducer (count/fold/sum/min/max/mean) emits exactly one value per
                // element, so the identity fallback is dead and `optional(<body>) ≡ <body>` — the
                // frontier is the body's ending frontier (a scalar → a following out()/values()
                // statically faults, matching TS). Checked on the literal terminal, NOT the
                // propagated `frontier_from_reducer` (which survives a following limit/dedup).
                let body_always_produces = matches!(
                    self.last_arm_last_step.as_str(),
                    "count" | "fold" | "sum" | "min" | "max" | "mean"
                );
                self.expect(&Tok::RParen)?;
                self.current = 0;
                self.slots = 1;
                self.edge_hop = None; // the reconverged frontier is not a just-hopped edge
                if body_always_produces {
                    // Take the body's ending frontier — the fallback never fires.
                    self.edge_scope = body_edge_scope;
                    self.current_is_element = body_frontier.0;
                    self.current_is_scalar = body_frontier.1;
                    self.current_is_path = body_frontier.2;
                    self.on_edge = self.edge_scope == Some(true);
                    // The reducer terminal RESETS the traverser path, so a following path() is
                    // the reduced value (consistent with the aggregate fast-path above and
                    // `count().path()`); parse_sub_body_seeded restored the flag to the branch
                    // input's, so set it back here for the reducer output.
                    self.frontier_from_reducer = true;
                    return Ok(plan.per_element_branch(
                        crate::ir::PerElemKind::Optional,
                        None,
                        vec![body.reconverge(oc)],
                        from,
                    ));
                }
                // The output is the body's frontier OR the source (fallback) — an edge is in
                // scope only where BOTH have one.
                self.edge_scope = combine_edge_scope(&[incoming_edge_scope, body_edge_scope]);
                // Post-branch VALUE + ADJACENCY frontier is UNKNOWN (element/scalar/path cleared)
                // so a following sum/values/adjacency reaches the runtime check; only edge_scope
                // (endpoints) propagates, and on_edge (an all-edge branch's endpoint read).
                self.current_is_element = false;
                self.current_is_scalar = false;
                self.current_is_path = false;
                self.on_edge = self.edge_scope == Some(true);
                plan.per_element_branch(
                    crate::ir::PerElemKind::Optional,
                    None,
                    vec![body.reconverge(oc)],
                    from,
                )
            }
            "coalesce" => {
                // coalesce(<hop>, <hop>, …): per element, the FIRST branch that
                // produces a result. Each branch k fires when no earlier branch's hop
                // exists AND its own does — an Exists guard chain — then expands. All
                // branches land at the same slot, so it reconverges like union.
                // coalesce of all-`values('k')` bodies — `coalesce(values('lang'),
                // values('name'))` = the FIRST PRESENT property (drop the element if
                // none present). Lowers to a scalar Case over PropertyExists + a filter
                // that keeps only rows with at least one present, sidestepping the
                // Exists-over-a-projection provenance limitation.
                let from = self.current;
                // The all-`values` fast-path (a scalar Case over PropertyExists) skips the
                // per-arm element-type guard, so take it ONLY over a DEFINITE element frontier.
                // Over a scalar/path input (`count().coalesce(values,values)`,
                // `path().coalesce(values,values)`) it wrongly succeeded where pure-TS faults on
                // the arm's `values()`; routing those to the general per-element path below runs
                // each arm through `step()`, which faults values-on-scalar / values-on-path. An
                // UNKNOWN frontier (post-branch) also falls through → the arm reaches the runtime.
                let fast_keys = if self.current_is_element {
                    self.try_all_values_bodies()
                } else {
                    None
                };
                if let Some(keys) = fast_keys {
                    let present = |k: &str| Expr::PropertyExists {
                        slot: from,
                        key: k.to_string(),
                    };
                    let mut any: Option<Expr> = None;
                    for k in &keys {
                        any = Some(match any {
                            None => present(k),
                            Some(p) => Expr::Or(Box::new(p), Box::new(present(k))),
                        });
                    }
                    let branches = keys
                        .iter()
                        .map(|k| {
                            (
                                present(k),
                                Expr::Prop {
                                    slot: from,
                                    key: k.clone(),
                                },
                            )
                        })
                        .collect();
                    let p = plan
                        .filter(any.expect("coalesce has at least one body"))
                        .project(vec![(
                            "coalesce".to_string(),
                            Expr::Case {
                                branches,
                                otherwise: None,
                            },
                        )]);
                    self.current = 0;
                    self.slots = 1;
                    self.edge_scope = Some(false); // an all-values coalesce yields a scalar
                                                   // Post-branch frontier is UNKNOWN (like the general coalesce below), NOT the
                                                   // stale incoming element — else `outE().coalesce(values, values).both()` read
                                                   // the pre-coalesce edge frontier and statically faulted, where pure-TS clears
                                                   // to unknown and lets both() reach the runtime (→ [] / count 0).
                    self.current_is_element = false;
                    self.current_is_scalar = false;
                    self.current_is_path = false;
                    self.on_edge = false;
                    return Ok(p);
                }
                // General: run each arm PER ELEMENT (TinkerPop per-traverser coalesce) — the
                // FIRST arm producing ≥1 output for that element wins. Each arm is a plain
                // sub-traversal reconverged to one output slot; the "first non-empty" routing
                // is decided at exec (Plan::PerElementBranch), so no exclusion-guard chain and
                // no whole-stream EXISTS approximation is needed — and a barrier inside an arm
                // (`limit(3)`, `skip(1)`, `count()`, `dedup()`) applies to THAT element's arm.
                let slots = self.slots;
                // Seed the arms with the outer edge hop so a leading `otherV()`/`inV()`
                // off `V().outE()` resolves against its origin (see parse_sub_body_seeded).
                self.edge_hop = prev_edge_hop;
                let mut bodies = Vec::new();
                let mut any_edge = false; // an edge-hop body → coalesce yields an edge frontier
                let mut arm_edge_scopes: Vec<Option<bool>> = Vec::new();
                loop {
                    if self.peek_leading_is_edge() {
                        any_edge = true;
                    }
                    let (body, oc, _os) = self.parse_sub_body_seeded(Plan::Row, from, slots)?;
                    bodies.push(body.reconverge(oc));
                    arm_edge_scopes.push(self.last_arm_edge_scope);
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                self.current = 0;
                self.slots = 1;
                self.edge_hop = None; // the reconverged frontier is not a just-hopped edge
                                      // An edge is in scope only where EVERY arm has one (port of TS combineHasEdge).
                self.edge_scope = combine_edge_scope(&arm_edge_scopes);
                // The post-branch frontier is UNKNOWN for VALUE and ADJACENCY checks (matches
                // pure-TS: a following sum/values/adjacency reaches the runtime check rather
                // than a static fault) — clear element/scalar/on_edge. Only `edge_scope` is
                // propagated, for the ENDPOINT fault (`coalesce(count(), inE()).outV()`), and a
                // path frontier when EVERY arm is a path (for a following path()).
                self.current_is_element = false;
                self.current_is_scalar = false;
                self.current_is_path = false;
                // A following endpoint reads off a reconverged edge only when EVERY arm has an
                // edge in scope (Some(true)); a mixed edge+scalar branch is Some(false) and
                // faults, an edge+path branch stays Some(true) and works. Adjacency stays
                // lenient via `current_is_element` (cleared).
                self.on_edge = self.edge_scope == Some(true);
                // A path() after the branch uses the per-step history (GremlinFullPath), NOT the
                // interleaved node/edge GremlinPath: the branch output can be mixed (some rows
                // end on an edge, some on a vertex), and GremlinPath's single static
                // `ends_on_edge` would wrongly append the landed endpoint for the edge rows.
                self.edge_path_ok = false;
                if any_edge {
                    self.path_has_edges = true;
                }
                return Ok(plan.per_element_branch(
                    crate::ir::PerElemKind::Coalesce,
                    None,
                    bodies,
                    from,
                ));
            }
            "match" => {
                // match(<fragment>, …). A fragment is one of:
                //   as('s').<hop>('L'?)[.has(…)]*.as('e')  — a hop binding (with optional
                //                                             embedded property filters)
                //   as('s')[.has(…)]+                       — a filter on a bound tag
                //   not(as('s').<hop>('L'?).as('e'))        — an anti-join between tags
                // Greedy solve: bind the entry to the first fragment's start tag, then
                // repeatedly apply any fragment whose needed tag(s) are already bound.
                let entry = self.current;
                let mut plan = plan;
                enum Frag {
                    Hop {
                        start: String,
                        dir: Dir,
                        label: Option<String>,
                        filters: Vec<(String, Option<(CompareOp, Value)>)>,
                        end: String,
                    },
                    Filter {
                        start: String,
                        filters: Vec<(String, Option<(CompareOp, Value)>)>,
                    },
                    Not {
                        start: String,
                        dir: Dir,
                        label: Option<String>,
                        end: String,
                    },
                }
                let mut frags: Vec<Frag> = Vec::new();
                let mut first_start: Option<String> = None;
                loop {
                    if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                        self.bump();
                        self.expect(&Tok::Dot)?;
                    }
                    let head = self.ident()?;
                    if head.eq_ignore_ascii_case("not") {
                        // not( [__.] as('s').<hop>('L'?).as('e') )
                        self.expect(&Tok::LParen)?;
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        let (start, dir, label, end) = self.match_hop_head()?;
                        self.expect(&Tok::RParen)?; // close not(...)
                        if first_start.is_none() {
                            first_start = Some(start.clone());
                        }
                        frags.push(Frag::Not {
                            start,
                            dir,
                            label,
                            end: end.ok_or("match() not(...) fragment must end with as('tag')")?,
                        });
                    } else {
                        if !head.eq_ignore_ascii_case("as") {
                            return Err("match() fragment must start with as('tag')".into());
                        }
                        self.expect(&Tok::LParen)?;
                        let start = self.str_arg()?;
                        self.expect(&Tok::RParen)?;
                        if first_start.is_none() {
                            first_start = Some(start.clone());
                        }
                        self.expect(&Tok::Dot)?;
                        // Next step decides the fragment kind: a hop, or a bare filter.
                        let is_hop = matches!(self.peek(), Some(Tok::Ident(s)) if {
                            let l = s.to_ascii_lowercase();
                            l == "out" || l == "in" || l == "both"
                        });
                        if is_hop {
                            let dir = match self.ident()?.to_ascii_lowercase().as_str() {
                                "out" => Dir::Out,
                                "in" => Dir::In,
                                _ => Dir::Both,
                            };
                            self.expect(&Tok::LParen)?;
                            let label = if matches!(self.peek(), Some(Tok::Str(_))) {
                                Some(self.str_arg()?)
                            } else {
                                None
                            };
                            self.expect(&Tok::RParen)?;
                            // Zero or more embedded `.has(...)` on the landed element.
                            let mut filters = Vec::new();
                            while self.peek() == Some(&Tok::Dot)
                                && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("has"))
                            {
                                self.expect(&Tok::Dot)?;
                                filters.push(self.parse_match_has()?);
                            }
                            self.expect(&Tok::Dot)?;
                            let as2 = self.ident()?;
                            if !as2.eq_ignore_ascii_case("as") {
                                return Err("match() hop fragment must end with as('tag')".into());
                            }
                            self.expect(&Tok::LParen)?;
                            let end = self.str_arg()?;
                            self.expect(&Tok::RParen)?;
                            frags.push(Frag::Hop {
                                start,
                                dir,
                                label,
                                filters,
                                end,
                            });
                        } else {
                            // A bare filter fragment: one or more `has(...)` on the tag.
                            let mut filters = vec![self.parse_match_has()?];
                            while self.peek() == Some(&Tok::Dot)
                                && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("has"))
                            {
                                self.expect(&Tok::Dot)?;
                                filters.push(self.parse_match_has()?);
                            }
                            frags.push(Frag::Filter { start, filters });
                        }
                    }
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                if let Some(s0) = first_start {
                    self.labels.entry(s0).or_insert(entry);
                }
                let has_expr = |slot: usize, f: &(String, Option<(CompareOp, Value)>)| {
                    let exists = Expr::PropertyExists {
                        slot,
                        key: f.0.clone(),
                    };
                    match &f.1 {
                        None => exists,
                        Some((op, v)) => Expr::And(
                            Box::new(exists),
                            Box::new(Expr::Compare {
                                op: *op,
                                left: Box::new(Expr::Prop {
                                    slot,
                                    key: f.0.clone(),
                                }),
                                right: Box::new(Expr::Lit(v.clone())),
                            }),
                        ),
                    }
                };
                let mut applied = vec![false; frags.len()];
                loop {
                    let mut progress = false;
                    for i in 0..frags.len() {
                        if applied[i] {
                            continue;
                        }
                        match &frags[i] {
                            Frag::Hop {
                                start,
                                dir,
                                label,
                                filters,
                                end,
                            } => {
                                let Some(&start_slot) = self.labels.get(start) else {
                                    continue;
                                };
                                let landed = self.slots;
                                plan = plan.expand(start_slot, *dir, &etypes_of(label.as_deref()));
                                self.slots += 1;
                                for f in filters {
                                    plan = plan.filter(has_expr(landed, f));
                                }
                                match self.labels.get(end).copied() {
                                    Some(existing) => {
                                        plan = plan.filter(Expr::Compare {
                                            op: CompareOp::Eq,
                                            left: Box::new(Expr::Slot(landed)),
                                            right: Box::new(Expr::Slot(existing)),
                                        });
                                    }
                                    None => {
                                        self.labels.insert(end.clone(), landed);
                                    }
                                }
                            }
                            Frag::Filter { start, filters } => {
                                let Some(&slot) = self.labels.get(start) else {
                                    continue;
                                };
                                for f in filters {
                                    plan = plan.filter(has_expr(slot, f));
                                }
                            }
                            Frag::Not {
                                start,
                                dir,
                                label,
                                end,
                            } => {
                                let (Some(&s_slot), Some(&e_slot)) =
                                    (self.labels.get(start), self.labels.get(end))
                                else {
                                    continue;
                                };
                                // NOT ∃ an edge start→end: the anti-join over a correlated
                                // one-hop existence constrained to the bound end slot. The
                                // Exists exec inserts a provenance column at `self.slots`,
                                // so the hop's landed node lands one past it.
                                let landed = self.slots + 1;
                                let body = Plan::Row
                                    .expand(s_slot, *dir, &etypes_of(label.as_deref()))
                                    .filter(Expr::Compare {
                                        op: CompareOp::Eq,
                                        left: Box::new(Expr::Slot(landed)),
                                        right: Box::new(Expr::Slot(e_slot)),
                                    });
                                plan = plan.filter(Expr::Not(Box::new(Expr::Exists {
                                    body: Box::new(body),
                                    outer_width: self.slots,
                                })));
                            }
                        }
                        applied[i] = true;
                        progress = true;
                    }
                    if !progress {
                        break;
                    }
                }
                if applied.iter().any(|&a| !a) {
                    return Err(
                        "match(): patterns with no bound start tag are not solvable in this subset"
                            .into(),
                    );
                }
                self.current = entry;
                plan
            }
            "branch" => {
                // branch(<test>).option(m, <hop>)….option(none, <hop>): route each
                // element by the TEST value — the option whose match value equals it,
                // else the `none` default. v1: test is values('k') or label(), each
                // option body a single hop. Guards are choose-style (test == m; default
                // = test matches none), all landing at the same slot like union.
                let from = self.current;
                if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                    self.bump();
                    self.expect(&Tok::Dot)?;
                }
                let test_name = self.ident()?.to_ascii_lowercase();
                self.expect(&Tok::LParen)?;
                let test_expr = match test_name.as_str() {
                    "values" => {
                        let k = self.str_arg()?;
                        self.expect(&Tok::RParen)?;
                        Expr::Prop { slot: from, key: k }
                    }
                    "label" => {
                        self.expect(&Tok::RParen)?;
                        Expr::Call {
                            name: "element_label".to_string(),
                            args: vec![Expr::Slot(from)],
                        }
                    }
                    other => {
                        return Err(format!(
                            "branch(<test>): only values('k') or label() supported, got `{other}`"
                        ))
                    }
                };
                self.expect(&Tok::RParen)?; // close branch(...)
                                            // Parse `.option(m, <body>)` modulators (m = literal, or bare `none`
                                            // for the default). Each body is an arbitrary sub-traversal, gated on
                                            // the INPUT element by the routing guard (so no Exists-provenance issue).
                let width = self.slots;
                let mut bodies = Vec::new();
                let mut matched: Vec<Value> = Vec::new();
                let mut land = (0usize, 1usize);
                while self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("option"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `option`
                    self.expect(&Tok::LParen)?;
                    let is_none = matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("none"));
                    let guard = if is_none {
                        self.bump();
                        if self.peek() == Some(&Tok::LParen) {
                            self.bump();
                            self.expect(&Tok::RParen)?;
                        }
                        Expr::Not(Box::new(or_of_equals(&test_expr, &matched)))
                    } else {
                        let v = self.literal()?;
                        matched.push(v.clone());
                        Expr::Compare {
                            op: CompareOp::Eq,
                            left: Box::new(test_expr.clone()),
                            right: Box::new(Expr::Lit(v)),
                        }
                    };
                    self.expect(&Tok::Comma)?;
                    let seed = Plan::Row.filter(guard);
                    let (body, oc, os) = self.parse_sub_body_seeded(seed, from, width)?;
                    land = (oc, os);
                    self.expect(&Tok::RParen)?;
                    bodies.push(body);
                }
                self.current = land.0;
                self.slots = land.1;
                plan.branch(bodies)
            }
            "choose" => {
                // choose(<cond>, <then>[, <else>]): route each element by whether <cond>
                // produces output — PER ELEMENT (TinkerPop). cond/then/else are plain
                // sub-traversals; the routing runs at exec (Plan::PerElementBranch), so a
                // barrier/aggregate inside any of them applies per traverser and the cond's
                // truth is UNIFORMLY "produces >=1 output": a filter cond yields the element
                // iff it passes, a navigating cond iff the hop exists, an aggregate cond
                // (`count()`) always. An absent else passes the source element through.
                let from = self.current;
                let slots = self.slots;
                let incoming_edge_scope = self.edge_scope; // an absent else is the source
                                                           // The cond: a sub-traversal whose truth is "produces output". Seed the outer
                                                           // edge hop so a leading otherV()/inV() resolves against its origin.
                self.edge_hop = prev_edge_hop;
                let (cond_body, _cc, _cs) = self.parse_sub_body_seeded(Plan::Row, from, slots)?;
                self.expect(&Tok::Comma)?;
                // A `drop()` then-arm (no else): a WRITE, which the per-element read path cannot
                // run. Lower it to a TOP-LEVEL Update over the guarded frontier (the elements
                // whose cond produces output) — `is_write` then sees it and run_update deletes.
                if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("drop"))
                    && self.toks.get(self.pos + 1) == Some(&Tok::LParen)
                    && self.toks.get(self.pos + 2) == Some(&Tok::RParen)
                    && self.toks.get(self.pos + 3) == Some(&Tok::RParen)
                {
                    self.bump(); // drop
                    self.expect(&Tok::LParen)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::RParen)?; // close choose(...)
                    let guard = Expr::Exists {
                        body: Box::new(cond_body),
                        outer_width: slots,
                    };
                    return Ok(Plan::Update {
                        input: Box::new(plan.filter(guard)),
                        ops: vec![crate::ir::SetOp::Delete {
                            slot: from,
                            detach: true,
                        }],
                    });
                }
                // The then arm.
                self.edge_hop = prev_edge_hop;
                let (then_b, then_oc, _os) = self.parse_sub_body_seeded(Plan::Row, from, slots)?;
                let then_es = self.last_arm_edge_scope;
                let mut arms = vec![then_b.reconverge(then_oc)];
                let mut arm_edge_scopes = vec![then_es];
                let has_else = self.peek() == Some(&Tok::Comma);
                if has_else {
                    self.bump();
                    self.edge_hop = prev_edge_hop;
                    let (else_b, else_oc, _os) =
                        self.parse_sub_body_seeded(Plan::Row, from, slots)?;
                    arm_edge_scopes.push(self.last_arm_edge_scope);
                    arms.push(else_b.reconverge(else_oc));
                } else {
                    // The implicit else is the SOURCE element — combine with the incoming
                    // frontier for the output-type inference below.
                    arm_edge_scopes.push(incoming_edge_scope);
                }
                self.expect(&Tok::RParen)?;
                self.current = 0;
                self.slots = 1;
                self.edge_hop = None; // the reconverged frontier is not a just-hopped edge
                self.edge_scope = combine_edge_scope(&arm_edge_scopes);
                // Post-branch VALUE + ADJACENCY frontier is unknown (see coalesce); only
                // edge_scope (endpoints), an all-path frontier, and on_edge for an all-edge
                // branch (endpoint read) propagate.
                self.current_is_element = false;
                self.current_is_scalar = false;
                self.current_is_path = false;
                self.on_edge = self.edge_scope == Some(true);
                return Ok(plan.per_element_branch(
                    crate::ir::PerElemKind::Choose { has_else },
                    Some(cond_body),
                    arms,
                    from,
                ));
            }
            "and" | "or" => {
                // and(f1, f2, …) / or(f1, f2, …): each child is an element filter
                // (has/hasNot/nested and/or/not); combine their predicates and apply
                // one Filter. The `(` was consumed at the top of `step`.
                let parts = self.child_filter_list()?;
                self.expect(&Tok::RParen)?;
                let mut it = parts.into_iter();
                let combined = match it.next() {
                    Some(first) => it.fold(first, |acc, e| {
                        if lname == "and" {
                            Expr::And(Box::new(acc), Box::new(e))
                        } else {
                            Expr::Or(Box::new(acc), Box::new(e))
                        }
                    }),
                    // Empty: `or()` matches nothing (false), `and()` everything (true).
                    None => Expr::Lit(Value::Bool(lname == "and")),
                };
                plan.filter(combined)
            }
            "not" => {
                // not(f): negate a single child element filter. The `(` was consumed
                // at the top of `step`.
                let inner = self.child_filter_expr()?;
                self.expect(&Tok::RParen)?;
                plan.filter(Expr::Not(Box::new(inner)))
            }
            "out" | "in" | "both" => {
                // 0 args → ANY edge type (argless out()); 1+ → a disjunction over the
                // listed types, exactly as GQL's `-[:A|B]->` lowers (the plan builder
                // takes the whole `&[String]` etype list, empty = any).
                let mut labels: Vec<String> = Vec::new();
                if !matches!(self.peek(), Some(Tok::RParen)) {
                    loop {
                        labels.push(self.str_arg()?);
                        if self.peek() == Some(&Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RParen)?;
                let dir = match name.to_ascii_lowercase().as_str() {
                    "out" => Dir::Out,
                    "in" => Dir::In,
                    _ => Dir::Both,
                };
                let from = self.current;
                self.current = self.slots;
                self.slots += 1;
                self.on_edge = false;
                // Gremlin `both()` traverses a self-loop TWICE (out- AND in-edge).
                if matches!(dir, Dir::Both) {
                    plan.expand_both_gremlin(from, &labels)
                } else {
                    plan.expand(from, dir, &labels)
                }
            }
            "repeat" => {
                // `repeat(<hop>)` v1: the body is a SINGLE anonymous hop
                // (`out`/`in`/`both`, optionally `__`-prefixed). Held OPEN (not built)
                // so the modulators times/emit/until can shape the walk; `out_slot`
                // (the endpoint) is pre-allocated as the width so an emit/until
                // predicate parsed before the flush references it. The LParen was
                // already consumed at the top of `step`.
                // A degenerate `repeat(identity())` body doesn't move the frontier; hold
                // it open with an identity marker so the modulators still attach, then
                // flush to a passthrough (exact for `times(0)`; a reasonable smoke result
                // otherwise). The endpoint stays the current element (no new slot).
                let is_identity = {
                    let mut p = self.pos;
                    if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
                        p += 1;
                        if self.toks.get(p) == Some(&Tok::Dot) {
                            p += 1;
                        }
                    }
                    matches!(self.toks.get(p), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("identity"))
                };
                if is_identity {
                    if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                        self.bump();
                        self.expect(&Tok::Dot)?;
                    }
                    self.ident()?; // identity
                    self.expect(&Tok::LParen)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::RParen)?; // close repeat(...)
                    self.pending_repeat = Some(RepeatCtx {
                        dir: Dir::Out,
                        label: None,
                        from: self.current,
                        out_slot: self.current,
                        times: None,
                        min_one: false,
                        filter: None,
                        bind_tag: None,
                        min_override: None,
                        path_ok_at_open: self.path_ok,
                        until: None,
                        until_pre: false,
                        body_filter: None,
                        max_cap: None,
                        identity_body: true,
                    });
                    return Ok(plan);
                }
                let (dir, label, bind_tag, body_filter, max_cap) = self.repeat_body()?;
                self.expect(&Tok::RParen)?;
                let from = self.current;
                let out_slot = self.slots; // endpoint == width at flush time
                self.current = out_slot; // emit/until predicates read the endpoint
                self.pending_repeat = Some(RepeatCtx {
                    dir,
                    label,
                    from,
                    out_slot,
                    times: None,
                    min_one: false,
                    filter: None,
                    bind_tag,
                    min_override: None,
                    path_ok_at_open: self.path_ok,
                    // A PRE-form `until(pred).repeat(body)` stashed its predicate; attach
                    // it now as a while-do stop (checked before the body → `until_pre`).
                    until: self.pending_until.take(),
                    until_pre: true,
                    body_filter,
                    max_cap,
                    identity_body: false,
                });
                plan
            }
            "times" => {
                let n = self.usize_arg()?;
                self.expect(&Tok::RParen)?;
                let n = u32::try_from(n).map_err(|_| "times(n): n too large")?;
                let ctx = self
                    .pending_repeat
                    .as_mut()
                    .ok_or("times(n) must follow repeat(<hop>)")?;
                ctx.times = Some(n);
                plan
            }
            "emit" => {
                // `emit()` / `emit(pred)`: emit at EVERY depth (min → 1), optionally
                // filtering the emitted endpoint. A `loops()` predicate instead bounds
                // the emitted DEPTH (`emit(loops().is(gt(1)))` → from depth 2).
                if self.pending_repeat.is_none() {
                    return Err("emit() must follow repeat(<hop>)".into());
                }
                if let Some((op, n)) = self.try_loops_predicate()? {
                    self.expect(&Tok::RParen)?;
                    // TinkerPop `loops()` == depth + 1 at the emit check, so a predicate
                    // on loops maps to a depth bound one lower: gt(k) → depth ≥ k,
                    // ge(k)/eq(k) → depth ≥ k-1.
                    let min = match op {
                        CompareOp::Gt => n,
                        CompareOp::Ge | CompareOp::Eq => n.saturating_sub(1),
                        _ => 1,
                    };
                    let ctx = self.pending_repeat.as_mut().unwrap();
                    ctx.min_one = true;
                    ctx.min_override = Some(min);
                    return Ok(plan);
                }
                let filter = if self.peek() == Some(&Tok::RParen) {
                    None
                } else {
                    Some(self.child_filter_expr()?)
                };
                self.expect(&Tok::RParen)?;
                let ctx = self.pending_repeat.as_mut().unwrap();
                ctx.min_one = true;
                if let Some(f) = filter {
                    ctx.filter = Some(f);
                }
                plan
            }
            "until" => {
                // `until(pred)`: loop until the endpoint satisfies `pred`. A
                // `loops().is(eq(n))` predicate is exactly `times(n)` (stop after n
                // hops). Otherwise a prune-on-match walk (see the `until` field on
                // Plan::VarLength). Two positions: POST-form `repeat(body).until(pred)`
                // (do-while) sets the stop on the pending ctx; PRE-form
                // `until(pred).repeat(body)` (while-do) is stashed until repeat() opens.
                if self.pending_repeat.is_none() {
                    // PRE-form: no loops() shorthand here (a while-do count is unusual);
                    // stash the endpoint predicate for the following repeat().
                    let f = self.child_filter_expr()?;
                    self.expect(&Tok::RParen)?;
                    self.pending_until = Some(f);
                    return Ok(plan);
                }
                if let Some((op, n)) = self.try_loops_predicate()? {
                    self.expect(&Tok::RParen)?;
                    // `loops()` == depth + 1, so stopping when loops == n means the walk
                    // ran to depth n-1 — a fixed-length `times(n-1)` walk.
                    let times = match op {
                        CompareOp::Eq | CompareOp::Ge => n.saturating_sub(1),
                        CompareOp::Gt => n,
                        _ => return Err("until(loops().is(<op>)) unsupported op".into()),
                    };
                    let ctx = self.pending_repeat.as_mut().unwrap();
                    ctx.times = Some(times);
                    ctx.min_one = false;
                    return Ok(plan);
                }
                let f = self.child_filter_expr()?;
                self.expect(&Tok::RParen)?;
                let ctx = self.pending_repeat.as_mut().unwrap();
                // POST-form do-while: at least one body iteration, then stop on match.
                ctx.until = Some(f);
                ctx.until_pre = false;
                plan
            }
            "oute" | "ine" | "bothe" => {
                // Edge-yielding hop: bind the traversed edge as a slot and leave the
                // current element ON that edge (so `.values`/`.count`/`.dedup` see the
                // edge). `expand_edge` appends TWO slots — edge at W, the landed
                // endpoint node at W+1 — so a following inV/outV/otherV can step onto
                // the endpoint the hop already resolved.
                let mut labels: Vec<String> = Vec::new();
                if !matches!(self.peek(), Some(Tok::RParen)) {
                    loop {
                        labels.push(self.str_arg()?);
                        if self.peek() == Some(&Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RParen)?;
                let dir = match lname.as_str() {
                    "oute" => Dir::Out,
                    "ine" => Dir::In,
                    _ => Dir::Both,
                };
                let from = self.current;
                let edge_slot = self.slots; // W
                let node_slot = self.slots + 1; // W+1: the landed endpoint
                self.current = edge_slot;
                self.on_edge = true;
                self.slots += 2;
                self.edge_hop = Some((node_slot, dir));
                plan.expand_edge_gremlin(from, dir, &labels)
            }
            "inv" | "outv" | "otherv" | "bothv" => {
                self.expect(&Tok::RParen)?;
                match prev_edge_hop {
                    // Immediately after outE/inE/bothE, the hop already landed the OTHER
                    // endpoint in `node_slot`: `otherV` for any direction; `inV` (dst)
                    // only when we went OUT; `outV` (src) only when we went IN. The
                    // origin-returning combinations fall through to EdgeVertex below.
                    Some((node_slot, dir))
                        if matches!(
                            (lname.as_str(), dir),
                            ("otherv", _) | ("inv", Dir::Out) | ("outv", Dir::In)
                        ) =>
                    {
                        self.current = node_slot;
                        self.on_edge = false;
                        plan
                    }
                    // Otherwise (a bare edge frontier — `g.E().outV()`,
                    // `coalesce(outE(...)).inV()` — or an origin-returning combination):
                    // read the endpoint straight off the edge at the current slot. Valid when
                    // the frontier holds edges (`on_edge`) OR MIGHT (a reconverged branch whose
                    // edge scope is unknown, `edge_scope != Some(false)` — the type check above
                    // already faulted a DEFINITE non-edge); EdgeVertex skips the non-edge rows.
                    _ if self.on_edge || self.edge_scope != Some(false) => {
                        // otherV needs a reference vertex. Off a DEFINITE bare edge source
                        // (`g.E()`, `g.E().hasLabel(...)` — still `current_is_element`) there is
                        // none, so it faults statically (matching pure-TS). Off a reconverged
                        // BRANCH of edges (`current_is_element` cleared) it resolves the far
                        // endpoint from the lineage reference vertex at runtime.
                        if lname == "otherv" && self.current_is_element {
                            return Err("otherV() off a bare edge frontier is not supported — \
                                        it has no reference vertex"
                                .into());
                        }
                        self.on_edge = false;
                        // This endpoint reads OFF the edge (not the interleaved landed vertex an
                        // outE/inE pre-recorded), so the node-lineage GremlinPath renders no
                        // longer matches — e.g. `outE().outV()` returns to the SOURCE, not the
                        // recorded dst. Drop the interleaved-edge-path fast path so a following
                        // path() uses the per-step history (GremlinFullPath), which is correct.
                        self.edge_path_ok = false;
                        let other = lname == "otherv";
                        let which = match lname.as_str() {
                            "outv" => Dir::Out,
                            "inv" => Dir::In,
                            _ => Dir::Both, // bothv, or otherv (ignored when `other`)
                        };
                        let edge_slot = self.current;
                        let landed = self.slots;
                        self.current = landed;
                        self.slots += 1;
                        Plan::EdgeVertex {
                            input: Box::new(plan),
                            edge_slot,
                            which,
                            other,
                        }
                    }
                    // A vertex move with no edge frontier and no preceding edge hop.
                    _ => {
                        return Err(format!(
                            "{name}() must immediately follow outE()/inE()/bothE()"
                        ))
                    }
                }
            }
            "values" => {
                // values('k', …): emit the value of each listed property that is
                // PRESENT on the element — an ABSENT property yields nothing (the TS engine
                // skips it; a present-but-null value is kept, per the null-first-class
                // policy). Single key: filter-present then project. Multiple keys: a
                // per-element branch over each present key.
                let mut keys = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    keys.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                let from = self.current;
                // `<vertex-hop chain>.values('k').tree()`: fold the projected value in as
                // the tree's LEAF level. Projecting here would collapse the batch and drop
                // the lineage tree() reads, so instead keep the vertex frontier, filter to
                // rows that HAVE the key (values() skips absent), and record the leaf key.
                if keys.len() == 1
                    && self.path_ok_pre_step
                    && matches!(self.peek(), Some(Tok::Dot))
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("tree"))
                {
                    let key = keys.pop().expect("one key");
                    let p = plan.filter(Expr::PropertyExists {
                        slot: from,
                        key: key.clone(),
                    });
                    // Keep the vertex chain answerable so tree() reads the node lineage.
                    self.path_ok = true;
                    self.path_leaf = Some(key);
                    return Ok(p);
                }
                let p = if keys.len() == 1 && self.algo_props.contains_key(&keys[0]) {
                    // An OLAP annotate property reads the computed slot (always present).
                    let slot = self.algo_props[&keys[0]];
                    plan.project(vec![(keys[0].clone(), Expr::Slot(slot))])
                } else if keys.len() == 1 {
                    let key = keys.pop().expect("one key");
                    plan.filter(Expr::PropertyExists {
                        slot: from,
                        key: key.clone(),
                    })
                    .project(vec![(key.clone(), Expr::Prop { slot: from, key })])
                } else {
                    let bodies = keys
                        .iter()
                        .map(|k| {
                            Plan::Row
                                .filter(Expr::PropertyExists {
                                    slot: from,
                                    key: k.clone(),
                                })
                                .project(vec![(
                                    k.clone(),
                                    Expr::Prop {
                                        slot: from,
                                        key: k.clone(),
                                    },
                                )])
                        })
                        .collect();
                    plan.branch(bodies)
                };
                self.current = 0;
                self.slots = 1;
                p
            }
            // (When in a `properties(...)` stream, `label()`/`key()` mean the property
            // key — handled by the guarded prop arm below, so exclude label here.)
            "id" | "label" if !(lname == "label" && self.prop_keys.is_some()) => {
                // Element accessors: `id()` → the preserved external id (polymorphic
                // node/edge, via `element_id`); `label()` → a single label string (a
                // vertex's label or an edge's type, via `element_label`). Both project
                // the current element slot to a scalar and reset to a value stream.
                self.expect(&Tok::RParen)?;
                let func = if lname == "id" {
                    "element_id"
                } else {
                    "element_label"
                };
                let p = plan.project(vec![(
                    lname.clone(),
                    Expr::Call {
                        name: func.to_string(),
                        args: vec![Expr::Slot(self.current)],
                    },
                )]);
                self.current = 0;
                self.slots = 1;
                p
            }
            "path" => {
                self.expect(&Tok::RParen)?;
                // The frontier becomes a PATH (a list), not an edge — so no edge is definitely in
                // scope. The early returns below skip the post-step classifier, so clear it here;
                // a following endpoint in a reconverged branch then treats the arm as
                // could-have-an-edge (None), not definitely-none (Some(false)).
                self.edge_scope = None;
                // After a REDUCING barrier (`count()`/`sum()`/`fold()`/…) TinkerPop resets the
                // traverser path, so `count().path()` is `[7]` — just the reduced value, not
                // the pre-barrier history. Lower to a single-element list of the current value.
                if self.frontier_from_reducer {
                    let _ = self.parse_path_bys()?; // a .by() on a scalar path is a no-op
                    let p = plan.project(vec![(
                        "path".to_string(),
                        Expr::List {
                            items: vec![Expr::Slot(self.current)],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    self.current_is_element = false;
                    self.current_is_path = true;
                    // The frontier is now a PATH, not the reducer's scalar — clear the reducer
                    // flag (this early return skips the post-step classifier) so a following
                    // `path()` takes the path-of-path branch, not another barrier reset.
                    self.frontier_from_reducer = false;
                    return Ok(p);
                }
                // `path().path()`: TinkerPop records the FIRST path's output as a new history
                // element, so the second yields `path[…prior history…, path[…]]`. The prior
                // `path()` already appended its own output to the step-history (the "path"
                // arm of the `PathRecord` loop, `STEP_SCALAR`), so the full per-step render
                // picks it up — but ONLY the full-history render sees it (the node-lineage /
                // GremlinPath renders below read a different sidecar). Route path-of-path
                // there directly rather than falling into the vertex-hop branch.
                if self.current_is_path {
                    let bys = self.parse_path_bys()?;
                    let p = plan.project(vec![("path".to_string(), Expr::GremlinFullPath { bys })]);
                    self.current = 0;
                    self.slots = 1;
                    self.current_is_element = false;
                    self.current_is_path = true;
                    return Ok(p);
                }
                // An interleaved node/edge path (`outE().inV()…`) renders through GremlinPath
                // (lineage `values` zipped with `edges`); a pure vertex-hop chain (incl. a
                // `repeat(<hop>)` walk) stays on the nodes-only `Expr::Path`. Both keep their
                // established behavior. Everything else — a value projection, an `E()` source,
                // a barrier, a branch — uses the full per-step history from `PathRecord`.
                if self.path_has_edges && self.edge_path_ok {
                    let ends_on_edge = self.on_edge;
                    let bys = self.parse_path_bys()?;
                    let p = plan.project(vec![(
                        "path".to_string(),
                        Expr::GremlinPath { ends_on_edge, bys },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    self.current_is_element = false;
                    self.current_is_path = true;
                    return Ok(p);
                }
                if self.path_ok {
                    // Vertex-hop path — the node sequence; `.by('k')` projects each element.
                    let call = if self.peek() == Some(&Tok::Dot)
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                    {
                        self.expect(&Tok::Dot)?;
                        self.ident()?; // `by`
                        self.expect(&Tok::LParen)?;
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("T"))
                        {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        let key = match self.peek() {
                            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("id") => {
                                self.bump();
                                self.eat_empty_parens();
                                "\u{0}id".to_string()
                            }
                            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("label") => {
                                self.bump();
                                self.eat_empty_parens();
                                "\u{0}label".to_string()
                            }
                            _ => self.str_arg()?,
                        };
                        self.expect(&Tok::RParen)?;
                        Expr::Call {
                            name: "path_values".to_string(),
                            args: vec![Expr::Path, Expr::Lit(Value::Str(key.into()))],
                        }
                    } else {
                        Expr::Call {
                            name: "path_nodes".to_string(),
                            args: vec![Expr::Path],
                        }
                    };
                    let p = plan.project(vec![("path".to_string(), call)]);
                    self.current = 0;
                    self.slots = 1;
                    // The frontier is now a PATH (a later element step must fault) — set here
                    // because this early return skips the end-of-step frontier classifier.
                    self.current_is_element = false;
                    self.current_is_path = true;
                    return Ok(p);
                }
                // Previously DEFERRED — the full per-step traverser history (vertices, edges
                // AND projected scalars, in step order) that `PathRecord` recorded, matching
                // TinkerPop. Optional `.by(...)` projects each element, cycled positionally.
                let bys = self.parse_path_bys()?;
                let p = plan.project(vec![("path".to_string(), Expr::GremlinFullPath { bys })]);
                self.current = 0;
                self.slots = 1;
                self.current_is_element = false;
                self.current_is_path = true;
                return Ok(p);
            }
            "simplepath" | "cyclicpath" => {
                self.expect(&Tok::RParen)?;
                if !self.path_ok {
                    return Err("simplePath()/cyclicPath() are only supported over a pure \
                                vertex-hop chain (V-source + out/in/both)"
                        .into());
                }
                // simplePath keeps traversers whose node path has NO repeat;
                // cyclicPath keeps those that DO. `path_has_dup` reads the lineage
                // node path (`Expr::Path`), which also switches lineage tracking on.
                let has_dup = Expr::Call {
                    name: "path_has_dup".to_string(),
                    args: vec![Expr::Path],
                };
                let pred = if lname == "simplepath" {
                    Expr::Not(Box::new(has_dup))
                } else {
                    has_dup
                };
                // A filter does not change the current element, so path_ok is
                // preserved (simplepath/cyclicpath are in the path-preserving set) —
                // a following path() still works.
                plan.filter(pred)
            }
            "elementmap" => {
                // elementMap() → the TS engine's FLAT element map {id, label, <props…>} (plus
                // IN/OUT for edges); elementMap('k',…) filters the properties. Lowers
                // to the gremlin-only `element_map` exec fn (element slot + key list).
                let mut fn_args = vec![Expr::Slot(self.current)];
                if !matches!(self.peek(), Some(Tok::RParen)) {
                    loop {
                        fn_args.push(Expr::Lit(Value::Str(self.str_arg()?.into())));
                        if self.peek() == Some(&Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RParen)?;
                let p = plan.project(vec![(
                    "elementMap".to_string(),
                    Expr::Call {
                        name: "element_map".to_string(),
                        args: fn_args,
                    },
                )]);
                self.current = 0;
                self.slots = 1;
                self.current_is_map = true;
                p
            }
            "valuemap" | "propertymap" => {
                // valueMap() → a PROPERTIES-only map (no id/label tokens) with scalar
                // values; valueMap('k1',…) filters keys. A leading boolean —
                // valueMap(true[, 'k1'…]), TinkerPop's includeTokens — also emits id +
                // label (an elementMap without an edge's IN/OUT). propertyMap() is the
                // same but each value is wrapped in a single-element list. Both lower to
                // the gremlin-only `value_map`/`property_map` exec fn; the second arg is
                // always the include-tokens Bool (downstream reads it, skips it as a key).
                let fn_name = if lname == "propertymap" {
                    "property_map"
                } else {
                    "value_map"
                };
                let mut fn_args = vec![Expr::Slot(self.current)];
                // The include-tokens boolean is a valueMap-only overload (TinkerPop has no
                // propertyMap(boolean)); propertyMap keeps its keys-only signature.
                let mut include_tokens = false;
                if lname == "valuemap"
                    && matches!(self.peek(), Some(Tok::Ident(s))
                        if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("false"))
                {
                    include_tokens = matches!(self.literal()?, Value::Bool(true));
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                    }
                }
                fn_args.push(Expr::Lit(Value::Bool(include_tokens)));
                if !matches!(self.peek(), Some(Tok::RParen)) {
                    loop {
                        fn_args.push(Expr::Lit(Value::Str(self.str_arg()?.into())));
                        if self.peek() == Some(&Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RParen)?;
                let p = plan.project(vec![(
                    fn_name.to_string(),
                    Expr::Call {
                        name: fn_name.to_string(),
                        args: fn_args,
                    },
                )]);
                self.current = 0;
                self.slots = 1;
                self.current_is_map = true;
                p
            }
            "properties" => {
                // `properties('k'…)` is a stream of the element's Property objects. The
                // engine has no Property value; instead the element stays current,
                // present-filtered on the key(s), and a following value()/key()/label()/
                // hasValue()/count() reads through `prop_keys`. Single key: filter present
                // on it. Multiple/all keys: keep the element (a following terminal fans
                // out or the bare Property result is skipped by the harness).
                let mut keys: Vec<String> = Vec::new();
                if !matches!(self.peek(), Some(Tok::RParen)) {
                    loop {
                        keys.push(self.str_arg()?);
                        if self.peek() == Some(&Tok::Comma) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(&Tok::RParen)?;
                let p = if keys.len() == 1 {
                    plan.filter(Expr::PropertyExists {
                        slot: self.current,
                        key: keys[0].clone(),
                    })
                } else {
                    plan
                };
                self.prop_keys = Some(keys);
                p
            }
            "value" => {
                self.expect(&Tok::RParen)?;
                match self.prop_keys.take() {
                    Some(keys) => {
                        // `properties('k').value()` → the property VALUE (element's `k`).
                        let key = keys
                            .first()
                            .cloned()
                            .ok_or("value() after a multi-key properties() is not yet supported")?;
                        let p = plan.project(vec![(
                            "value".to_string(),
                            Expr::Prop {
                                slot: self.current,
                                key,
                            },
                        )]);
                        self.current = 0;
                        self.slots = 1;
                        p
                    }
                    // value() on a non-property is the value itself (identity).
                    None => plan,
                }
            }
            "key" | "label" if self.prop_keys.is_some() => {
                // `properties('k').key()`/`.label()` → the property KEY. A single-key
                // stream is a constant; an all-keys `properties()` fans out the element's
                // present property keys (one row per key) via keys(element) + unwind.
                self.expect(&Tok::RParen)?;
                let keys = self.prop_keys.take().unwrap();
                if keys.len() == 1 {
                    let p = plan.project(vec![(
                        "key".to_string(),
                        Expr::Lit(Value::Str(keys[0].clone().into())),
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    // all/multi keys: append the element's key LIST as a new slot, then
                    // unwind it into a stream of one key per row (the unfold pattern).
                    let keys_slot = self.slots;
                    let listed = plan.map_slot(
                        keys_slot,
                        Expr::Call {
                            name: "keys".to_string(),
                            args: vec![Expr::Slot(self.current)],
                        },
                        true,
                    );
                    self.slots += 1;
                    let var_slot = self.slots;
                    self.slots += 1;
                    let unwound = Plan::Unwind {
                        input: Box::new(listed),
                        list: Box::new(Expr::Slot(keys_slot)),
                        var_slot,
                        ordinal: None,
                    };
                    let p = unwound.project(vec![("key".to_string(), Expr::Slot(var_slot))]);
                    self.current = 0;
                    self.slots = 1;
                    p
                }
            }
            "hasvalue" if self.prop_keys.is_some() => {
                // `properties('k').hasValue(v…)` → keep the property whose value is one of
                // v… (an OR-of-equals on the element's `k`), staying in the stream.
                let key = self
                    .prop_keys
                    .as_ref()
                    .and_then(|ks| ks.first())
                    .cloned()
                    .ok_or("hasValue() after a multi-key properties() is not yet supported")?;
                let mut vals = vec![self.literal()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    vals.push(self.literal()?);
                }
                self.expect(&Tok::RParen)?;
                let left = Expr::Prop {
                    slot: self.current,
                    key,
                };
                plan.filter(or_of_equals(&left, &vals))
            }
            "where" => {
                // Tagged key form `where('a', op('b'))`: keep traversers where the
                // value at step-label `a` relates (op) to the value at label `b` — a
                // slot-vs-slot comparison (the TS engine's WhereKey; the predicate's rhs is a
                // step-label, not a literal). Detected by a leading string + comma.
                if matches!(self.peek(), Some(Tok::Str(_)))
                    && self.toks.get(self.pos + 1) == Some(&Tok::Comma)
                {
                    let start = self.str_arg()?;
                    self.expect(&Tok::Comma)?;
                    let op_name = self.ident()?.to_ascii_lowercase();
                    self.expect(&Tok::LParen)?;
                    let end = self.str_arg()?;
                    self.expect(&Tok::RParen)?; // close op(...)
                    self.expect(&Tok::RParen)?; // close where(...)
                                                // Optional `.by('k')` modulators pick a PROPERTY to compare on each
                                                // side (`where('a',lt('b')).by('age')` → a.age < b.age). One `by`
                                                // applies to both sides; two apply to start then end in order. With
                                                // no `by` the tagged elements are compared directly (slot vs slot).
                    let mut bys: Vec<String> = Vec::new();
                    while self.peek() == Some(&Tok::Dot)
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                    {
                        self.expect(&Tok::Dot)?;
                        self.ident()?; // by
                        self.expect(&Tok::LParen)?;
                        bys.push(self.str_arg()?);
                        self.expect(&Tok::RParen)?;
                    }
                    let slot_of = |l: &str| {
                        self.labels
                            .get(l)
                            .copied()
                            .ok_or_else(|| format!("where('{l}', …): no step is labelled `{l}`"))
                    };
                    let op = match op_name.as_str() {
                        "eq" => CompareOp::Eq,
                        "neq" => CompareOp::Ne,
                        "gt" => CompareOp::Gt,
                        "gte" => CompareOp::Ge,
                        "lt" => CompareOp::Lt,
                        "lte" => CompareOp::Le,
                        other => {
                            return Err(format!("where('{start}', {other}(…)) needs a comparison"))
                        }
                    };
                    let side = |slot: usize, idx: usize| match bys.get(idx).or_else(|| bys.first())
                    {
                        Some(k) => Expr::Prop {
                            slot,
                            key: k.clone(),
                        },
                        None => Expr::Slot(slot),
                    };
                    let pred = Expr::Compare {
                        op,
                        left: Box::new(side(slot_of(&start)?, 0)),
                        right: Box::new(side(slot_of(&end)?, 1)),
                    };
                    return Ok(plan.filter(pred));
                }
                // Two forms: where(<hop>) is a SEMI-JOIN — keep the element if it HAS
                // such an adjacency — and where(P) filters the current VALUE by a
                // predicate. A leading `out`/`in`/`both` (or `__`) marks the hop form.
                // A traversal filter (`__.`, or a step head) routes through the general
                // filter-child machinery (hops → Exists, `<hop>.count().is()` →
                // CountSubquery, not/and/or, hasLabel, …).
                let is_traversal = matches!(self.peek(), Some(Tok::Ident(s)) if {
                    let l = s.to_ascii_lowercase();
                    s == "__" || matches!(l.as_str(),
                        "out" | "in" | "both" | "oute" | "ine" | "bothe"
                        | "not" | "and" | "or" | "has" | "hasnot" | "haslabel" | "haskey")
                });
                if is_traversal {
                    let e = self.child_filter_expr()?;
                    self.expect(&Tok::RParen)?; // close where(...)
                    plan.filter(e)
                } else {
                    // Predicate on the current value: `where(op(v))`. If the rhs is a
                    // single bound step-label — `where(neq('me'))` after tagging `me` —
                    // compare the current value to that TAG's value (the TS engine's tagged
                    // where); otherwise fall through to the literal-predicate path.
                    let tag_form = match (self.peek(), self.toks.get(self.pos + 1)) {
                        (Some(Tok::Ident(op)), Some(Tok::LParen)) => {
                            let cop = match op.to_ascii_lowercase().as_str() {
                                "eq" => Some(CompareOp::Eq),
                                "neq" => Some(CompareOp::Ne),
                                "gt" => Some(CompareOp::Gt),
                                "gte" => Some(CompareOp::Ge),
                                "lt" => Some(CompareOp::Lt),
                                "lte" => Some(CompareOp::Le),
                                _ => None,
                            };
                            match (cop, self.toks.get(self.pos + 2)) {
                                (Some(op), Some(Tok::Str(s))) => {
                                    self.labels.get(s).copied().map(|slot| (op, slot))
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    if let Some((op, slot)) = tag_form {
                        self.bump(); // op
                        self.bump(); // (
                        self.bump(); // tag string
                        self.expect(&Tok::RParen)?; // close op(...)
                        self.expect(&Tok::RParen)?; // close where(...)
                        plan.filter(Expr::Compare {
                            op,
                            left: Box::new(Expr::Slot(self.current)),
                            right: Box::new(Expr::Slot(slot)),
                        })
                    } else {
                        let pred = self.predicate_expr(Expr::Slot(self.current))?;
                        self.expect(&Tok::RParen)?;
                        plan.filter(pred)
                    }
                }
            }
            // is(P) / is(op(v)) / is(literal): filter the current VALUE by a
            // predicate — same as `where` on the value stream. A bare literal is an
            // equality test (predicate_expr handles it).
            "is" => {
                let pred = self.predicate_expr(Expr::Slot(self.current))?;
                self.expect(&Tok::RParen)?;
                plan.filter(pred)
            }
            "count" => {
                // count() folds the whole stream to one number; count(local) is
                // per-row — the size of the current list cell (a fold()'d list).
                let is_local = self.parse_scope_is_local()?;
                self.expect(&Tok::RParen)?;
                if is_local {
                    let p = plan.project(vec![(
                        "count".to_string(),
                        Expr::Call {
                            name: "list_count".to_string(),
                            args: vec![Expr::Slot(self.current)],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    let p = plan.aggregate(
                        vec![],
                        vec![Agg {
                            func: AggFn::Count,
                            arg: None,
                            distinct: false,
                            name: "count".into(),
                            frac: None,
                            null_on_empty: false,
                            numeric_only: false,
                        }],
                    );
                    self.current = 0;
                    self.slots = 1;
                    p
                }
            }
            "min" | "max" | "sum" | "mean" => {
                // Global: fold the whole value stream to one scalar. Local: reduce the
                // current list cell per row (the numeric elements) via a list fn.
                let is_local = self.parse_scope_is_local()?;
                self.expect(&Tok::RParen)?;
                if is_local {
                    let fname = match lname.as_str() {
                        "min" => "list_min",
                        "max" => "list_max",
                        "sum" => "list_sum",
                        _ => "list_mean",
                    };
                    let p = plan.project(vec![(
                        lname.clone(),
                        Expr::Call {
                            name: fname.to_string(),
                            args: vec![Expr::Slot(self.current)],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    // TinkerPop: min/max/sum/mean over raw graph ELEMENTS throws — a
                    // Vertex/Edge is neither numeric (sum/mean) nor comparable (min/max).
                    // Project first with `values('<key>')`. Static over a DEFINITE element
                    // frontier (matches the pure-TS build-time Frontier check). After a
                    // branch the frontier is UNKNOWN (`current_is_element` is cleared), so a
                    // `coalesce(...).sum()` whose arms produce nothing reaches the RUNTIME
                    // check and yields [null] instead of faulting here.
                    if self.current_is_element {
                        return Err(format!(
                            "{lname}() over graph elements is not supported — a vertex/edge \
                             is not a number; project with values('<key>') first"
                        ));
                    }
                    if self.current_is_path {
                        return Err(format!(
                            "{lname}() over paths is not supported — a path is not a number; \
                             project its elements first (e.g. unfold().values('<key>'))"
                        ));
                    }
                    let func = match lname.as_str() {
                        "min" => AggFn::Min,
                        "max" => AggFn::Max,
                        "sum" => AggFn::Sum,
                        _ => AggFn::Avg,
                    };
                    // Fold the property DIRECTLY over the chain (unwrap a preceding
                    // values(k)) so the aggregate fast-paths see the hop/var-length chain.
                    let (agg_in, arg) = unwrap_values_fold(plan, self.current);
                    let p = agg_in.aggregate(
                        vec![],
                        vec![Agg {
                            func,
                            arg: Some(arg),
                            distinct: false,
                            name: lname.clone(),
                            frac: None,
                            // Gremlin numeric-agg semantics: sum() of nothing is NULL and
                            // a non-numeric sum()/mean() propagates NaN (never faults).
                            null_on_empty: matches!(func, AggFn::Sum | AggFn::Avg),
                            // Gremlin min()/max() only compare numbers (GQL total-orders).
                            numeric_only: matches!(func, AggFn::Min | AggFn::Max),
                        }],
                    );
                    self.current = 0;
                    self.slots = 1;
                    p
                }
            }
            "fold" => {
                self.expect(&Tok::RParen)?;
                // Collect the whole value stream into one row holding a list (see
                // AggFn::Collect). Bare fold() only; fold(seed, biFn) is deferred.
                let p = plan.aggregate(
                    vec![],
                    vec![Agg {
                        func: AggFn::Collect,
                        arg: Some(Expr::Slot(self.current)),
                        distinct: false,
                        name: "fold".into(),
                        frac: None,
                        null_on_empty: false,
                        numeric_only: false,
                    }],
                );
                self.current = 0;
                self.slots = 1;
                p
            }
            "dedup" => {
                // dedup() dedups the whole row; dedup('a','b') keeps the first row per
                // distinct tuple of the values TAGGED at those labels (keyed distinct).
                let mut labels: Vec<String> = Vec::new();
                if matches!(self.peek(), Some(Tok::Str(_))) {
                    labels.push(self.str_arg()?);
                    while self.peek() == Some(&Tok::Comma) {
                        self.bump();
                        labels.push(self.str_arg()?);
                    }
                }
                self.expect(&Tok::RParen)?;
                // Optional `.by('k'|id|label)` modulators: dedup by the TUPLE of those
                // by-values of the element (keep the first per distinct tuple).
                let mut by_slots: Vec<usize> = Vec::new();
                let mut by_exprs: Vec<Expr> = Vec::new();
                let mut p = plan;
                while self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // by
                    self.expect(&Tok::LParen)?;
                    let (_, e) = self.by_key_expr(self.current)?;
                    self.expect(&Tok::RParen)?;
                    by_exprs.push(e.clone());
                    let w = self.slots;
                    p = p.map_slot(w, e, true);
                    self.slots += 1;
                    by_slots.push(w);
                }
                if !by_slots.is_empty() {
                    // TinkerPop: a `by('k')` that yields no value (absent property) FILTERS
                    // the traverser — dedup().by('age') drops elements without `age` rather
                    // than deduping them under one null key. Gate BEFORE the distinct, and
                    // only on an element/unknown frontier (a Map/scalar `by('k')` is a field read).
                    if self.by_absent_gates() {
                        let refs: Vec<&Expr> = by_exprs.iter().collect();
                        if let Some(g) = key_present_gate(&refs) {
                            p = p.filter(g);
                        }
                    }

                    return Ok(p.distinct_by(by_slots));
                }
                let plan = p;
                if labels.is_empty() {
                    // Gremlin dedup() dedups on the CURRENT traverser value, not the
                    // whole row — after a hop the row also carries the source, so a
                    // whole-row distinct would keep duplicate neighbours reached from
                    // different sources.
                    plan.distinct_by(vec![self.current])
                } else {
                    let key_slots = labels
                        .iter()
                        .map(|l| {
                            self.labels
                                .get(l)
                                .copied()
                                .ok_or_else(|| format!("dedup('{l}'): no step is labelled `{l}`"))
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    plan.distinct_by(key_slots)
                }
            }
            "limit" => {
                // limit(n) — the first n rows. limit(local, n) — the first n of each
                // list CELL (a slice `[0, n)`), like tail(local, k) but from the front.
                let is_local = self.parse_scope_is_local()?;
                if is_local {
                    self.expect(&Tok::Comma)?;
                    let n = self.usize_arg()?;
                    self.expect(&Tok::RParen)?;
                    let p = plan.project(vec![(
                        "limit".to_string(),
                        Expr::Call {
                            name: "list_range".to_string(),
                            args: vec![
                                Expr::Slot(self.current),
                                Expr::Lit(Value::Num(0.0)),
                                Expr::Lit(Value::Num(n as f64)),
                            ],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    let n = self.usize_arg()?;
                    self.expect(&Tok::RParen)?;
                    // Fuse `order().by(…).limit(n)` into ONE top-N OrderPage: with a
                    // limit the sort is a partial select_nth (O(rows·log n)), not a full
                    // O(rows·log rows) sort followed by a slice. Only when the input sort
                    // has no page of its own (skip/limit None) — else the tighter bound
                    // would be lost.
                    match plan {
                        Plan::OrderPage {
                            input,
                            keys,
                            skip: None,
                            limit: None,
                            fault_on_element,
                        } => Plan::OrderPage {
                            input,
                            keys,
                            skip: None,
                            limit: Some(n),
                            fault_on_element,
                        },
                        other => other.order_page(vec![], None, Some(n)),
                    }
                }
            }
            "skip" => {
                // skip(n) — drop the first n rows. skip(local, n) — drop the first n of
                // each list CELL.
                let is_local = self.parse_scope_is_local()?;
                if is_local {
                    self.expect(&Tok::Comma)?;
                    let n = self.usize_arg()?;
                    self.expect(&Tok::RParen)?;
                    let p = plan.project(vec![(
                        "skip".to_string(),
                        Expr::Call {
                            name: "list_skip".to_string(),
                            args: vec![Expr::Slot(self.current), Expr::Lit(Value::Num(n as f64))],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    let n = self.usize_arg()?;
                    self.expect(&Tok::RParen)?;
                    plan.order_page(vec![], Some(n), None)
                }
            }
            "tail" => {
                // tail(n) — the LAST n rows of the stream (default 1), the mirror of
                // limit. tail(local[, k]) — the last k of each list CELL instead.
                let is_local = self.parse_scope_is_local()?;
                let n = if is_local {
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                        self.usize_arg()?
                    } else {
                        1
                    }
                } else if matches!(self.peek(), Some(Tok::Num(_))) {
                    self.usize_arg()?
                } else {
                    1
                };
                self.expect(&Tok::RParen)?;
                if is_local {
                    let p = plan.project(vec![(
                        "tail".to_string(),
                        Expr::Call {
                            name: "list_tail".to_string(),
                            args: vec![Expr::Slot(self.current), Expr::Lit(Value::Num(n as f64))],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    plan.tail(n)
                }
            }
            "range" => {
                // range(lo, hi) — the half-open row window [lo, hi). range(local, lo,
                // hi) — the same slice WITHIN each list cell instead of over the stream.
                let is_local = self.parse_scope_is_local()?;
                if is_local {
                    self.expect(&Tok::Comma)?;
                }
                let lo = self.usize_arg()?;
                self.expect(&Tok::Comma)?;
                let hi = self.usize_arg()?;
                self.expect(&Tok::RParen)?;
                // range(lo, hi) with hi < lo is an invalid window — TinkerPop throws
                // IllegalArgumentException at build time (data-independent), so fault at parse
                // to match (the pure-TS engine does the same). limit(0) = range(0, 0) stays
                // valid (hi == lo → the empty window). `hi == -1` (unlimited) is unsupported.
                if hi < lo {
                    return Err(format!(
                        "E_INVALID_VALUE: range({lo}, {hi}) is not a valid window — the high \
                         bound must be greater than or equal to the low bound"
                    ));
                }
                if is_local {
                    let p = plan.project(vec![(
                        "range".to_string(),
                        Expr::Call {
                            name: "list_range".to_string(),
                            args: vec![
                                Expr::Slot(self.current),
                                Expr::Lit(Value::Num(lo as f64)),
                                Expr::Lit(Value::Num(hi as f64)),
                            ],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    plan.order_page(vec![], Some(lo), Some(hi.saturating_sub(lo)))
                }
            }
            "order" => {
                // Optional scope: order()/order(global) sort the stream;
                // order(local) sorts within the current list/map cell.
                let is_local = self.parse_scope_is_local()?;
                self.expect(&Tok::RParen)?;
                if is_local {
                    // order(local)[.by([keys|values,] [asc|desc])]. For a LIST the `by`
                    // is a direction (elements sort by natural order); for a MAP an
                    // optional `keys`/`values` Column token picks the sort axis
                    // (default `values`), then an optional direction.
                    let (descending, by_key) = if self.peek() == Some(&Tok::Dot)
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                    {
                        self.expect(&Tok::Dot)?;
                        self.ident()?; // `by`
                        self.expect(&Tok::LParen)?;
                        // Optional `Column.keys`/`Column.values` (or bare `keys`/`values`).
                        let mut by_key = false;
                        if matches!(self.peek(), Some(Tok::Ident(s)) if {
                            let l = s.to_ascii_lowercase();
                            l == "keys" || l == "values" || l == "column"
                        }) {
                            let mut col = self.ident()?;
                            if self.peek() == Some(&Tok::Dot) {
                                self.bump();
                                col = self.ident()?; // strip `Column.`
                            }
                            by_key = col.eq_ignore_ascii_case("keys");
                            if self.peek() == Some(&Tok::Comma) {
                                self.bump();
                            }
                        }
                        let d = if self.peek() == Some(&Tok::RParen) {
                            false
                        } else {
                            self.order_dir()?
                        };
                        self.expect(&Tok::RParen)?;
                        (d, by_key)
                    } else {
                        (false, false)
                    };
                    plan.sort_local(descending, by_key)
                } else {
                    // Global stream sort with zero or more `.by(...)` modulators, each a
                    // sort key (applied in order). A key body is a property, a direction
                    // on the current value, an id/label/T token, or a degree sub-traversal
                    // `[__.]<hop>('L').count()` — each with an optional trailing direction.
                    let mut keys: Vec<SortKey> = Vec::new();
                    while self.peek() == Some(&Tok::Dot)
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                    {
                        self.expect(&Tok::Dot)?;
                        self.ident()?; // `by`
                        self.expect(&Tok::LParen)?;
                        let (expr, descending) = self.order_by_body(self.current)?;
                        self.expect(&Tok::RParen)?;
                        keys.push(SortKey {
                            expr,
                            descending,
                            // TinkerPop treats null as part of the total order that DESC
                            // reverses — nulls FIRST in asc, LAST in desc (unlike GQL's
                            // direction-INDEPENDENT `NULLS FIRST`). `row_cmp` places nulls
                            // by this flag alone, so encode the direction here.
                            nulls_first: !descending,
                        });
                    }
                    if keys.is_empty() {
                        keys.push(SortKey {
                            expr: Expr::Slot(self.current),
                            descending: false,
                            nulls_first: true,
                        });
                    }
                    // TinkerPop: order() over raw graph ELEMENTS throws — a vertex/edge has
                    // no natural order. A `by('<key>')`/`by(<traversal>)` projection makes a
                    // comparable key (fine); a bare `order()` or a direction-only `by(desc)`
                    // sorts the element itself, which faults. Known statically from the chain.
                    if self.current_is_element
                        && keys
                            .iter()
                            .any(|k| matches!(k.expr, Expr::Slot(s) if s == self.current))
                    {
                        return Err(
                            "order() over graph elements is not supported — elements have no \
                             natural order; use order().by('<key>')"
                                .into(),
                        );
                    }
                    // TinkerPop: a `by('k')` sort key that yields no value (absent property)
                    // FILTERS the traverser out of the ordering (order().by('age') drops
                    // elements without `age`) — gate on PROPERTY_EXISTS for each property key.
                    // Element frontier only: a `by('k')` over a Map/scalar row (e.g.
                    // `project(...).order().by('k')`) is a field read, not an element property.
                    let key_refs: Vec<&Expr> = keys.iter().map(|k| &k.expr).collect();
                    let gate = if self.by_absent_gates() {
                        key_present_gate(&key_refs)
                    } else {
                        None
                    };
                    let gated = match gate {
                        Some(g) => plan.filter(g),
                        None => plan,
                    };
                    // Gremlin: `order()` over raw elements faults (the runtime backstop for a
                    // mixed/branch frontier the build-time check above cannot classify).
                    gated.order_page_strict(keys, None, None)
                }
            }
            "as" => {
                // Label the current slot; the plan is unchanged (select resolves it).
                let label = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                self.first_labels
                    .entry(label.clone())
                    .or_insert(self.current);
                self.all_labels
                    .entry(label.clone())
                    .or_default()
                    .push(self.current);
                self.labels.insert(label, self.current);
                plan
            }
            "select" => {
                // `select(Column.keys|values)` (or bare `keys`/`values`): project a
                // Map cell to the LIST of its keys or values, in the map's current
                // order. Distinct from tag selection — no string label follows.
                if matches!(self.peek(), Some(Tok::Ident(s)) if {
                    let l = s.to_ascii_lowercase();
                    l == "keys" || l == "values" || l == "column"
                }) {
                    let mut col = self.ident()?;
                    if self.peek() == Some(&Tok::Dot) {
                        self.bump();
                        col = self.ident()?; // strip `Column.`
                    }
                    self.expect(&Tok::RParen)?;
                    let fname = if col.eq_ignore_ascii_case("keys") {
                        "map_keys"
                    } else {
                        "map_values"
                    };
                    let p = plan.project(vec![(
                        "select".into(),
                        Expr::Call {
                            name: fname.into(),
                            args: vec![Expr::Slot(self.current)],
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    return Ok(p);
                }
                // An optional leading `Pop` token (`First`/`Last`/`All`, or the
                // `Pop.`-prefixed forms) picks WHICH binding of a rebound tag to read.
                // `First` reads the first binding (`first_labels`); `All` returns every
                // binding as a list; the default is the last.
                let mut pop_first = false;
                let mut pop_all = false;
                if matches!(self.peek(), Some(Tok::Ident(s)) if {
                    let l = s.to_ascii_lowercase();
                    l == "first" || l == "last" || l == "all" || l == "pop"
                }) && self.toks.get(self.pos + 1) != Some(&Tok::LParen)
                {
                    let mut tok = self.ident()?;
                    if self.peek() == Some(&Tok::Dot) {
                        self.bump();
                        tok = self.ident()?; // strip the `Pop.` prefix
                    }
                    pop_first = tok.eq_ignore_ascii_case("first");
                    pop_all = tok.eq_ignore_ascii_case("all");
                    self.expect(&Tok::Comma)?;
                }
                // One or more labels. A single label projects that element; two or
                // more build an insertion-ordered Map keyed by the labels.
                let mut labels = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.pos += 1;
                    labels.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                // Trailing `.by('key')` modulators project each selected element to a
                // property; they CYCLE across the labels (the TS engine's `bys[i % bys.len()]`).
                // `by('k')` only for now (a nested by-traversal is deferred).
                // A by-modulator is a property key or an `id`/`label` element token.
                enum SelBy {
                    Key(String),
                    Id,
                    Label,
                    /// A degree sub-traversal `[__.]<hop>('L'…).count()` — the count of
                    /// the tagged element's neighbours; built per tag from its slot.
                    Degree(Dir, Vec<String>, bool),
                }
                let mut bys: Vec<SelBy> = Vec::new();
                while self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `by`
                    self.expect(&Tok::LParen)?;
                    // A degree sub-traversal `[__.]<hop>('L'…)[.values('k')].count()`.
                    let deg_head = {
                        let mut p = self.pos;
                        if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
                            p += 1;
                            if self.toks.get(p) == Some(&Tok::Dot) {
                                p += 1;
                            }
                        }
                        matches!(self.toks.get(p), Some(Tok::Ident(s)) if matches!(
                            s.to_ascii_lowercase().as_str(),
                            "out" | "in" | "both" | "oute" | "ine" | "bothe"))
                    };
                    if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("T")) {
                        self.bump();
                        self.expect(&Tok::Dot)?;
                    }
                    let by = if deg_head {
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        let hop = self.ident()?.to_ascii_lowercase();
                        let (dir, is_edge) = match hop.as_str() {
                            "out" => (Dir::Out, false),
                            "in" => (Dir::In, false),
                            "both" => (Dir::Both, false),
                            "oute" => (Dir::Out, true),
                            "ine" => (Dir::In, true),
                            _ => (Dir::Both, true),
                        };
                        self.expect(&Tok::LParen)?;
                        let mut ls: Vec<String> = Vec::new();
                        if matches!(self.peek(), Some(Tok::Str(_))) {
                            ls.push(self.str_arg()?);
                            while self.peek() == Some(&Tok::Comma) {
                                self.bump();
                                ls.push(self.str_arg()?);
                            }
                        }
                        self.expect(&Tok::RParen)?;
                        // Optional intermediate `.values('k')` (counts present names —
                        // same cardinality as the hop, so ignored for a count).
                        if self.peek() == Some(&Tok::Dot)
                            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("values"))
                        {
                            self.expect(&Tok::Dot)?;
                            self.ident()?; // values
                            self.expect(&Tok::LParen)?;
                            self.str_arg()?;
                            self.expect(&Tok::RParen)?;
                        }
                        self.expect(&Tok::Dot)?;
                        let c = self.ident()?; // count
                        if !c.eq_ignore_ascii_case("count") {
                            return Err("select().by(<traversal>) must end with .count()".into());
                        }
                        self.expect(&Tok::LParen)?;
                        self.expect(&Tok::RParen)?;
                        SelBy::Degree(dir, ls, is_edge)
                    } else {
                        match self.peek().cloned() {
                            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("id") => {
                                self.bump();
                                SelBy::Id
                            }
                            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("label") => {
                                self.bump();
                                SelBy::Label
                            }
                            _ => SelBy::Key(self.str_arg()?),
                        }
                    };
                    self.expect(&Tok::RParen)?;
                    bys.push(by);
                }
                let table = if pop_first {
                    &self.first_labels
                } else {
                    &self.labels
                };
                let slot_of = |l: &str| {
                    table
                        .get(l)
                        .copied()
                        .ok_or_else(|| format!("select('{l}'): no step is labelled `{l}`"))
                };
                let val_of = |i: usize, l: &str| -> Result<Expr, String> {
                    let slot = slot_of(l)?;
                    Ok(if bys.is_empty() {
                        Expr::Slot(slot)
                    } else {
                        match &bys[i % bys.len()] {
                            SelBy::Key(k) => Expr::Prop {
                                slot,
                                key: k.clone(),
                            },
                            SelBy::Id => Expr::Call {
                                name: "element_id".into(),
                                args: vec![Expr::Slot(slot)],
                            },
                            SelBy::Label => Expr::Call {
                                name: "element_label".into(),
                                args: vec![Expr::Slot(slot)],
                            },
                            SelBy::Degree(dir, ls, is_edge) => {
                                let body = if *is_edge {
                                    Plan::Row.expand_edge_gremlin(slot, *dir, ls)
                                } else {
                                    Plan::Row.expand(slot, *dir, ls)
                                };
                                Expr::CountSubquery {
                                    body: Box::new(body),
                                    outer_width: self.slots,
                                }
                            }
                        }
                    })
                };
                // A label that is not a bound step tag falls back to the TS engine's Scoping: if
                // the current traverser is a Map (a `project()`/`group()` row), `select(k)`
                // projects the entry `k`. We can't statically prove the frontier is a Map,
                // so we emit the map/record field read (`Expr::Field`) — byte-identical when
                // the value IS such a Map (the in-spec use, e.g.
                // `project(...).select('k')`), which is the only case this reaches. Only the
                // plain form (no Pop, no by-modulator) falls back; an unbound tag under a
                // `Pop.all`/by-modulated select still drops every traverser as before.
                let input_current = self.current;
                let any_unbound = labels.iter().any(|l| {
                    if pop_all {
                        !self.all_labels.contains_key(l)
                    } else {
                        !self.labels.contains_key(l)
                    }
                });
                if any_unbound && (pop_all || !bys.is_empty()) {
                    self.current = 0;
                    self.slots = 1;
                    return Ok(plan.filter(Expr::Lit(Value::Bool(false))));
                }
                if any_unbound && !incoming_is_map {
                    // An unbound tag on an element/scalar frontier (not a Map) matches
                    // nothing — the TS engine drops the traverser, so filter it all away.
                    self.current = 0;
                    self.slots = 1;
                    return Ok(plan.filter(Expr::Lit(Value::Bool(false))));
                }
                if any_unbound {
                    // Bound labels read their tagged slot; unbound ones read the Map entry.
                    let field_of = |l: &str| -> Expr {
                        match self.labels.get(l) {
                            Some(&slot) => Expr::Slot(slot),
                            None => Expr::Field {
                                base: Box::new(Expr::Slot(input_current)),
                                key: l.to_string(),
                            },
                        }
                    };
                    let p = if labels.len() == 1 {
                        plan.project(vec![(labels[0].clone(), field_of(&labels[0]))])
                    } else {
                        self.current_is_map = true; // multi-label select yields a Map
                        let entries = labels.iter().map(|l| (l.clone(), field_of(l))).collect();
                        plan.project(vec![(
                            "select".into(),
                            Expr::MapLit {
                                entries,
                                omit_absent: false,
                            },
                        )])
                    };
                    self.current = 0;
                    self.slots = 1;
                    return Ok(p);
                }
                let p = if pop_all {
                    // Pop.all: every binding of the (single) tag, as a list.
                    let slots = self.all_labels.get(&labels[0]).cloned().ok_or_else(|| {
                        format!("select('{}'): no step is labelled it", labels[0])
                    })?;
                    let items = slots.into_iter().map(Expr::Slot).collect();
                    plan.project(vec![(labels[0].clone(), Expr::List { items })])
                } else if labels.len() == 1 {
                    // TinkerPop: a `by('k')` that yields no value (absent property) FILTERS
                    // the traverser — `select('a').by('age')` drops a selected element with
                    // no `age`. Gate on PROPERTY_EXISTS before projecting.
                    let v = val_of(0, &labels[0])?;
                    let gate = if self.by_absent_gates() {
                        key_present_gate(&[&v])
                    } else {
                        None
                    };
                    let gated = match gate {
                        Some(g) => plan.filter(g),
                        None => plan,
                    };
                    gated.project(vec![(labels[0].clone(), v)])
                } else {
                    self.current_is_map = true; // multi-label select yields a Map
                    let entries = labels
                        .iter()
                        .enumerate()
                        .map(|(i, l)| Ok((l.clone(), val_of(i, l)?)))
                        .collect::<Result<Vec<_>, String>>()?;
                    // A no-value by() on ANY selected label filters the whole traverser.
                    let refs: Vec<&Expr> = entries.iter().map(|(_, e)| e).collect();
                    let gate = if self.by_absent_gates() {
                        key_present_gate(&refs)
                    } else {
                        None
                    };
                    let gated = match gate {
                        Some(g) => plan.filter(g),
                        None => plan,
                    };
                    gated.project(vec![(
                        "select".into(),
                        Expr::MapLit {
                            entries,
                            omit_absent: false,
                        },
                    )])
                };
                self.current = 0;
                self.slots = 1;
                p
            }
            "project" => {
                // project('a','b',…).by(x).by(y) → one Map per traverser, keyed by the
                // labels. Value for key i is the i-th `by` modulator, or the current
                // element when there is no i-th `by` (the TS engine's `bys.get(i)` — NOT cycled).
                let mut keys = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.pos += 1;
                    keys.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                let elem_slot = self.current;
                // Consume the trailing `.by(...)` modulators (like groupCount().by()).
                let mut bys: Vec<Expr> = Vec::new();
                while self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `by`
                    self.expect(&Tok::LParen)?;
                    let by = self.project_by_body(elem_slot)?;
                    self.expect(&Tok::RParen)?;
                    bys.push(by);
                }
                let entries: Vec<(String, Expr)> = keys
                    .iter()
                    .enumerate()
                    .map(|(i, k)| {
                        let v = bys.get(i).cloned().unwrap_or(Expr::Slot(elem_slot));
                        (k.clone(), v)
                    })
                    .collect();
                let p = plan.project(vec![(
                    "project".into(),
                    Expr::MapLit {
                        entries,
                        omit_absent: true,
                    },
                )]);
                self.current = 0;
                self.slots = 1;
                self.current_is_map = true;
                p
            }
            "groupcount" => {
                self.expect(&Tok::RParen)?;
                // `groupCount()` groups by the current element; `groupCount().by('k')`
                // groups by a property of it. The `.by(...)` modulator is optional.
                let key_expr = if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `by`
                    self.expect(&Tok::LParen)?;
                    let ke = self.by_key_expr(self.current)?;
                    self.expect(&Tok::RParen)?;
                    ke
                } else {
                    ("key".to_string(), Expr::Slot(self.current))
                };
                // An absent-property key-by filters the traverser out (no null bucket) —
                // element frontier only (a Map/scalar `by('k')` is a field read).
                let gate = if self.by_absent_gates() {
                    key_present_gate(&[&key_expr.1])
                } else {
                    None
                };
                let gated = match gate {
                    Some(g) => plan.filter(g),
                    None => plan,
                };
                let p = gated
                    .aggregate(
                        vec![key_expr],
                        vec![Agg {
                            func: AggFn::Count,
                            arg: None,
                            distinct: false,
                            name: "count".into(),
                            frac: None,
                            null_on_empty: false,
                            numeric_only: false,
                        }],
                    )
                    // Gremlin groupCount() is a single {key: count} Map, not (k,c) rows.
                    .group_to_map();
                self.current = 0;
                self.slots = 1; // one Map column
                self.current_is_map = true;
                p
            }
            "group" => {
                self.expect(&Tok::RParen)?;
                // group().by(key).by(value) → grouped aggregation. Core shapes this as
                // one {key: value} map; the engine represents a grouped result as ROWS
                // of (key, value), consistent with groupCount() — the row model has no
                // whole-stream single-map fold. The first `.by` is the group key, the
                // second the reducing/mapping value; both optional.
                let elem_slot = self.current;
                // Collect up to two trailing `.by(...)` modulators.
                let mut bys: Vec<GroupBy> = Vec::new();
                while bys.len() < 2
                    && self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `by`
                    self.expect(&Tok::LParen)?;
                    let by = if matches!(self.peek(), Some(Tok::Str(_))) {
                        GroupBy::Key(self.str_arg()?)
                    } else if self.peek() == Some(&Tok::RParen) {
                        GroupBy::Element
                    } else {
                        // A reducing sub-traversal, optionally `__.`-prefixed.
                        let dunder = matches!(self.peek(), Some(Tok::Ident(s)) if s == "__");
                        if dunder {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("count"))
                        {
                            self.ident()?; // count
                            self.expect(&Tok::LParen)?;
                            self.expect(&Tok::RParen)?;
                            GroupBy::Count
                        } else if matches!(self.peek(), Some(Tok::Ident(s)) if {
                            let l = s.to_ascii_lowercase();
                            matches!(l.as_str(), "out" | "in" | "both" | "oute" | "ine" | "bothe")
                        }) {
                            // `<hop>('L').count()` value-by — the group's total degree =
                            // Σ over the group of each element's neighbour count.
                            let hop = self.ident()?.to_ascii_lowercase();
                            let (hdir, hedge) = match hop.as_str() {
                                "out" => (Dir::Out, false),
                                "in" => (Dir::In, false),
                                "both" => (Dir::Both, false),
                                "oute" => (Dir::Out, true),
                                "ine" => (Dir::In, true),
                                _ => (Dir::Both, true),
                            };
                            self.expect(&Tok::LParen)?;
                            let mut ls: Vec<String> = Vec::new();
                            if matches!(self.peek(), Some(Tok::Str(_))) {
                                ls.push(self.str_arg()?);
                                while self.peek() == Some(&Tok::Comma) {
                                    self.bump();
                                    ls.push(self.str_arg()?);
                                }
                            }
                            self.expect(&Tok::RParen)?;
                            self.expect(&Tok::Dot)?;
                            let c = self.ident()?;
                            if !c.eq_ignore_ascii_case("count") {
                                return Err("group().by(<hop>…) must end with .count()".into());
                            }
                            self.expect(&Tok::LParen)?;
                            self.expect(&Tok::RParen)?;
                            let body = if hedge {
                                Plan::Row.expand_edge_gremlin(elem_slot, hdir, &ls)
                            } else {
                                Plan::Row.expand(elem_slot, hdir, &ls)
                            };
                            let deg = Expr::CountSubquery {
                                body: Box::new(body),
                                outer_width: self.slots,
                            };
                            GroupBy::Reduce(AggFn::Sum, Some(deg))
                        } else if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("values"))
                        {
                            // values('k'…)[.<agg>()] — a per-group aggregate over the
                            // (possibly several) keys' present values.
                            self.ident()?; // values
                            self.expect(&Tok::LParen)?;
                            let mut ks = vec![self.str_arg()?];
                            while self.peek() == Some(&Tok::Comma) {
                                self.bump();
                                ks.push(self.str_arg()?);
                            }
                            self.expect(&Tok::RParen)?;
                            // Optional trailing `.<agg>()`; a bare values() folds (Collect).
                            let func = if self.peek() == Some(&Tok::Dot) {
                                self.bump();
                                let a = self.ident()?.to_ascii_lowercase();
                                self.expect(&Tok::LParen)?;
                                self.expect(&Tok::RParen)?;
                                match a.as_str() {
                                    "sum" => AggFn::Sum,
                                    "min" => AggFn::Min,
                                    "max" => AggFn::Max,
                                    "mean" => AggFn::Avg,
                                    "count" => AggFn::Count,
                                    other => {
                                        return Err(format!(
                                            "group().by(values(...).{other}()) is not supported"
                                        ))
                                    }
                                }
                            } else {
                                AggFn::Collect
                            };
                            // Per key, the per-row contribution: `count()` counts PRESENT
                            // values (a stored null is present) via a presence marker;
                            // the others use the raw property. Multiple keys flatten into
                            // a LIST arg that fold_grouped reduces element-wise.
                            let per_key = |k: String| -> Expr {
                                if func == AggFn::Count {
                                    Expr::Case {
                                        branches: vec![(
                                            Expr::PropertyExists {
                                                slot: elem_slot,
                                                key: k,
                                            },
                                            Expr::Lit(Value::Num(1.0)),
                                        )],
                                        otherwise: None,
                                    }
                                } else {
                                    Expr::Prop {
                                        slot: elem_slot,
                                        key: k,
                                    }
                                }
                            };
                            let arg = if ks.len() == 1 {
                                per_key(ks.remove(0))
                            } else {
                                Expr::List {
                                    items: ks.into_iter().map(per_key).collect(),
                                }
                            };
                            GroupBy::Reduce(func, Some(arg))
                        } else if !dunder
                            && matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("id") || s.eq_ignore_ascii_case("label") || s.eq_ignore_ascii_case("T"))
                        {
                            let (name, e) = self.by_key_expr(elem_slot)?;
                            GroupBy::KeyExpr(name, e)
                        } else {
                            return Err(
                                "group().by(<nested traversal>): only count()/values(...).<agg>() supported"
                                    .into(),
                            );
                        }
                    };
                    self.expect(&Tok::RParen)?;
                    bys.push(by);
                }
                // Key-by (first modulator): a property, or the element itself.
                let key_expr = match bys.first() {
                    Some(GroupBy::Key(k)) => (
                        k.clone(),
                        Expr::Prop {
                            slot: elem_slot,
                            key: k.clone(),
                        },
                    ),
                    Some(GroupBy::KeyExpr(name, e)) => (name.clone(), e.clone()),
                    // by(count()) as a key makes no sense; treat absent/element/count
                    // key as grouping by the element.
                    _ => ("key".to_string(), Expr::Slot(elem_slot)),
                };
                // Value-by (second modulator): count() reduces; a property or the
                // element folds (collect, keeping nulls = Gremlin fold).
                let value_agg = match bys.get(1) {
                    Some(GroupBy::Count) => Agg {
                        func: AggFn::Count,
                        arg: None,
                        distinct: false,
                        name: "value".into(),
                        frac: None,
                        null_on_empty: false,
                        numeric_only: false,
                    },
                    Some(GroupBy::Key(k)) => Agg {
                        func: AggFn::Collect,
                        arg: Some(Expr::Prop {
                            slot: elem_slot,
                            key: k.clone(),
                        }),
                        distinct: false,
                        name: "value".into(),
                        frac: None,
                        null_on_empty: false,
                        numeric_only: false,
                    },
                    Some(GroupBy::KeyExpr(_, e)) => Agg {
                        func: AggFn::Collect,
                        arg: Some(e.clone()),
                        distinct: false,
                        name: "value".into(),
                        frac: None,
                        null_on_empty: false,
                        numeric_only: false,
                    },
                    Some(GroupBy::Reduce(func, arg)) => Agg {
                        func: *func,
                        arg: arg.clone(),
                        distinct: false,
                        name: "value".into(),
                        frac: None,
                        // Gremlin numeric-agg semantics: sum() of nothing is NULL, and a
                        // non-numeric sum()/mean() propagates NaN (never faults).
                        null_on_empty: matches!(func, AggFn::Sum | AggFn::Avg),
                        // Gremlin min()/max() only compare numbers (GQL total-orders).
                        numeric_only: matches!(func, AggFn::Min | AggFn::Max),
                    },
                    // Default (no second by) or bare by(): fold the group's elements.
                    _ => Agg {
                        func: AggFn::Collect,
                        arg: Some(Expr::Slot(elem_slot)),
                        distinct: false,
                        name: "value".into(),
                        frac: None,
                        null_on_empty: false,
                        numeric_only: false,
                    },
                };
                // A `by('k')` key that yields no value (absent property) filters the
                // traverser out of the grouping (no null bucket), matching TinkerPop. Only
                // on an ELEMENT frontier — a `by('k')` over a Map/scalar row is a field
                // read (PROPERTY_EXISTS does not apply there), so leave it ungated.
                let gate = if self.by_absent_gates() {
                    key_present_gate(&[&key_expr.1])
                } else {
                    None
                };
                let gated = match gate {
                    Some(g) => plan.filter(g),
                    None => plan,
                };
                let p = gated
                    .aggregate(vec![key_expr], vec![value_agg])
                    .group_to_map();
                self.current = 0;
                self.slots = 1; // one Map column
                p
            }
            "constant" => {
                // Replace every traverser with a constant value (Gremlin constant(x)).
                let v = self.literal()?;
                self.expect(&Tok::RParen)?;
                let p = plan.project(vec![("constant".to_string(), Expr::Lit(v))]);
                self.current = 0;
                self.slots = 1;
                p
            }
            "math" => {
                // math('<expr>')[.by('k')]: evaluate a math expression. `_` is the
                // current value, or the `.by('k')` projection of it; named identifiers
                // resolve to step-label variables. Lowers to the engine arithmetic /
                // scalar-fn kernels (bit-identical to GQL).
                let expr_str = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                let operand = if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `by`
                    self.expect(&Tok::LParen)?;
                    let k = self.str_arg()?;
                    self.expect(&Tok::RParen)?;
                    Expr::Prop {
                        slot: self.current,
                        key: k,
                    }
                } else {
                    Expr::Slot(self.current)
                };
                let e = self.parse_math(&expr_str, &operand)?;
                let p = plan.project(vec![("math".to_string(), e)]);
                self.current = 0;
                self.slots = 1;
                p
            }
            "identity" => {
                // Pass-through — the current element is unchanged (Gremlin identity()).
                self.expect(&Tok::RParen)?;
                plan
            }
            "sample" => {
                // sample(n): a fixed-seed shuffle of the stream, truncated to n.
                let n = self.usize_arg()?;
                self.expect(&Tok::RParen)?;
                Plan::Sample {
                    input: Box::new(plan),
                    n,
                }
            }
            "index" => {
                // index(): pair each element with its 0-based position in the stream —
                // one [element, position] list per row.
                self.expect(&Tok::RParen)?;
                let p = Plan::Enumerate {
                    input: Box::new(plan),
                    slot: self.current,
                };
                self.current = 0;
                self.slots = 1;
                p
            }
            "none" => {
                // none() drops EVERY traverser — an always-false filter. none(pred) keeps
                // the traverser iff NO element of the current value (a list cell, or a
                // scalar treated as a 1-element list) satisfies `pred` — a `list_none`
                // scan lowering the comparison to an (op, value) pair.
                if self.peek() == Some(&Tok::RParen) {
                    self.expect(&Tok::RParen)?;
                    plan.filter(Expr::Lit(Value::Bool(false)))
                } else {
                    let (op, val) = self.simple_predicate()?;
                    self.expect(&Tok::RParen)?;
                    let e = Expr::Call {
                        name: "list_none".to_string(),
                        args: vec![
                            Expr::Slot(self.current),
                            Expr::Lit(Value::Str(compare_op_tag(op).into())),
                            Expr::Lit(val),
                        ],
                    };
                    plan.filter(e)
                }
            }
            "filter" => {
                // filter(<traversal>): keep the element iff the sub-traversal produces
                // output — the same semi-join machinery `where(<traversal>)` uses.
                let e = self.child_filter_expr()?;
                self.expect(&Tok::RParen)?;
                plan.filter(e)
            }
            "map" => {
                // map(<body>): per traverser, the body's FIRST result (empties dropped).
                // A single-VALUE body is a per-row projection: `count()` maps each
                // traverser to 1; `values('k')`/`constant`/`id`/`label` project that
                // value. (A navigating body — map(out()) taking the first neighbour —
                // is deferred.) `map(count())` is the reducing-barrier case that must
                // NOT fold the whole stream.
                let from = self.current;
                let mut probe = self.pos;
                if matches!(self.toks.get(probe), Some(Tok::Ident(s)) if s == "__") {
                    probe += 1;
                    if self.toks.get(probe) == Some(&Tok::Dot) {
                        probe += 1;
                    }
                }
                let is_count = matches!(self.toks.get(probe), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("count"))
                    && self.toks.get(probe + 1) == Some(&Tok::LParen)
                    && self.toks.get(probe + 2) == Some(&Tok::RParen);
                if is_count {
                    // Consume `[__.]count()` then close map(...).
                    if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                        self.bump();
                        self.expect(&Tok::Dot)?;
                    }
                    self.ident()?; // count
                    self.expect(&Tok::LParen)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::RParen)?; // close map(...)
                    let p = plan.project(vec![("map".to_string(), Expr::Lit(Value::Num(1.0)))]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else if let Some(val) = self.parse_single_value_body(from)? {
                    self.expect(&Tok::RParen)?;
                    let p = plan.project(vec![("map".to_string(), val)]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    return Err("map(<navigating traversal>) is not yet supported".into());
                }
            }
            "flatmap" => {
                // flatMap(<traversal>): apply the body's step chain to the current
                // frontier and continue from its output. The body chains directly onto
                // `plan` (NOT Row-rooted). (map() differs — a reducing-barrier body like
                // count() folds the whole stream — so it is NOT handled here.)
                if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                    self.bump();
                    self.expect(&Tok::Dot)?;
                }
                let mut p = plan;
                if matches!(self.peek(), Some(Tok::Ident(_))) {
                    p = self.step(p)?;
                    while self.peek() == Some(&Tok::Dot) {
                        self.bump();
                        p = self.step(p)?;
                    }
                }
                self.expect(&Tok::RParen)?;
                p
            }
            "sideeffect" => {
                // sideEffect(<traversal>): run the body for its SIDE EFFECTS; the main
                // stream passes through unchanged. The one observable effect is a
                // named-bag aggregate/store, which must snapshot the OUTER frontier (a
                // Row-rooted sub-parse would snapshot the wrong plan), so it is handled
                // here; every other body is a pure identity (its output is discarded).
                let mut probe = self.pos;
                if matches!(self.toks.get(probe), Some(Tok::Ident(s)) if s == "__") {
                    probe += 1;
                    if self.toks.get(probe) == Some(&Tok::Dot) {
                        probe += 1;
                    }
                }
                let is_bag = matches!(self.toks.get(probe), Some(Tok::Ident(s)) if {
                    let l = s.to_ascii_lowercase();
                    l == "aggregate" || l == "store"
                });
                if is_bag {
                    if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                        self.bump();
                        self.expect(&Tok::Dot)?;
                    }
                    self.ident()?; // aggregate/store
                    self.expect(&Tok::LParen)?;
                    let key = self.str_arg()?;
                    self.expect(&Tok::RParen)?; // close aggregate(...)
                    self.expect(&Tok::RParen)?; // close sideEffect(...)
                    self.caps
                        .insert(key, (plan.clone(), Expr::Slot(self.current)));
                    plan
                } else {
                    // Parse-and-discard the body; the main stream is identity.
                    let _ = self.parse_sub_body(self.current, self.slots)?;
                    self.expect(&Tok::RParen)?;
                    plan
                }
            }
            "sack" => {
                // sack() reads the per-traverser accumulator; sack(op).by('k') folds a
                // property into it in place (the frontier passes through). Requires a
                // preceding withSack(init).
                let sack = self
                    .sack_slot
                    .ok_or("sack() requires a preceding g.withSack(init)")?;
                if self.peek() == Some(&Tok::RParen) {
                    self.expect(&Tok::RParen)?;
                    let p = plan.project(vec![("sack".to_string(), Expr::Slot(sack))]);
                    self.current = 0;
                    self.slots = 1;
                    p
                } else {
                    let op = self.ident()?.to_ascii_lowercase();
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                    let by = self.ident()?;
                    if !by.eq_ignore_ascii_case("by") {
                        return Err("sack(op) must be followed by by('k')".into());
                    }
                    self.expect(&Tok::LParen)?;
                    let k = self.str_arg()?;
                    self.expect(&Tok::RParen)?;
                    let s = || Expr::Slot(sack);
                    let val = || Expr::Prop {
                        slot: self.current,
                        key: k.clone(),
                    };
                    let arith = |op| Expr::Arith {
                        op,
                        left: Box::new(s()),
                        right: Box::new(val()),
                    };
                    let cmp_keep = |op| Expr::Case {
                        branches: vec![(
                            Expr::Compare {
                                op,
                                left: Box::new(s()),
                                right: Box::new(val()),
                            },
                            s(),
                        )],
                        otherwise: Some(Box::new(val())),
                    };
                    let new = match op.as_str() {
                        "sum" => arith(ArithOp::Add),
                        "mult" => arith(ArithOp::Mul),
                        "assign" => val(),
                        "min" => cmp_keep(CompareOp::Le),
                        "max" => cmp_keep(CompareOp::Ge),
                        other => return Err(format!("unsupported sack operator `{other}`")),
                    };
                    plan.map_slot(sack, new, false)
                }
            }
            "local" => {
                // local(<traversal>) runs the body PER input element. v1 supports the
                // common reducing form local(<hop chain>.count()) — a per-element
                // (correlated) count that keeps every input row (0 for no matches),
                // lowered to Expr::CountSubquery over the hop body. A body without a
                // trailing count() is deferred (a general per-element sub-traversal
                // needs grouped-by-input scoping the row model does not yet have).
                let from = self.current;
                let width = self.slots;
                let save = self.pos;
                let (body, _oc, _os) = self.parse_sub_body(from, width)?;
                match body {
                    Plan::Aggregate { input, keys, aggs }
                        if keys.is_empty()
                            && aggs.len() == 1
                            && matches!(aggs[0].func, AggFn::Count) =>
                    {
                        self.expect(&Tok::RParen)?;
                        let expr = Expr::CountSubquery {
                            body: input,
                            outer_width: width,
                        };
                        let p = plan.project(vec![("local".to_string(), expr)]);
                        self.current = 0;
                        self.slots = 1;
                        p
                    }
                    // `local(<hop>.fold())` — a per-element COLLECT into a list.
                    Plan::Aggregate { input, keys, aggs }
                        if keys.is_empty()
                            && aggs.len() == 1
                            && matches!(aggs[0].func, AggFn::Collect) =>
                    {
                        self.expect(&Tok::RParen)?;
                        // The CollectSubquery exec inserts a provenance column at
                        // `outer_width`, so a body slot at/after it shifts up by one.
                        let mut scalar = aggs[0].arg.clone().unwrap_or(Expr::Slot(self.current));
                        shift_body_slots(&mut scalar, width);
                        let expr = Expr::CollectSubquery {
                            body: input,
                            scalar: Box::new(scalar),
                            outer_width: width,
                        };
                        let p = plan.project(vec![("local".to_string(), expr)]);
                        self.current = 0;
                        self.slots = 1;
                        p
                    }
                    // A non-reducing hop/value chain — `local(outE().inV())` — is per
                    // element already the same as applying the chain (each input keeps
                    // its own outputs), so re-parse the body onto the current plan.
                    _ if !matches!(body, Plan::Aggregate { .. }) => {
                        self.pos = save;
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        let mut p = plan;
                        if matches!(self.peek(), Some(Tok::Ident(_))) {
                            p = self.step(p)?;
                            while self.peek() == Some(&Tok::Dot) {
                                self.bump();
                                p = self.step(p)?;
                            }
                        }
                        self.expect(&Tok::RParen)?; // close local(...)
                        p
                    }
                    _ => {
                        return Err(
                            "local(<traversal>) beyond a hop chain or count()/fold() is deferred"
                                .into(),
                        )
                    }
                }
            }
            "withcomputer" => {
                // A no-op marker (lenke always computes in-process), matching the TS engine.
                self.expect(&Tok::RParen)?;
                plan
            }
            "pagerank" | "connectedcomponent" | "peerpressure" => {
                // OLAP annotate: run the algorithm over the store and attach the
                // per-node result at a new slot; a following values(<default property>)
                // reads it. Pass-through: the vertex frontier is unchanged. Optional
                // `pageRank(alpha)` sets the damping factor; `.with(...)` modulators
                // are deferred (default property names / iterations only for now).
                let alpha = if lname == "pagerank" {
                    if let Some(Tok::Num(n)) = self.peek().cloned() {
                        self.bump();
                        Some(n)
                    } else {
                        None
                    }
                } else {
                    None
                };
                self.expect(&Tok::RParen)?;
                let (algo, prop) = match lname.as_str() {
                    "pagerank" => (
                        crate::ir::GremlinAlgo::PageRank {
                            damping: alpha.unwrap_or(0.85),
                            iterations: 20,
                        },
                        "gremlin.pageRankVertexProgram.pageRank",
                    ),
                    "connectedcomponent" => (
                        crate::ir::GremlinAlgo::ConnectedComponent,
                        "gremlin.connectedComponentVertexProgram.component",
                    ),
                    _ => (
                        crate::ir::GremlinAlgo::PeerPressure { iterations: 30 },
                        "gremlin.peerPressureVertexProgram.cluster",
                    ),
                };
                let node_slot = self.current;
                let out_slot = self.slots;
                let p = plan.algo_annotate(algo, None, node_slot);
                self.slots += 1;
                self.algo_props.insert(prop.to_string(), out_slot);
                self.last_algo = Some(out_slot);
                self.path_ok = false;
                p
            }
            "with" => {
                // OLAP config modulator on the preceding algo step: `with([X.]propertyName,
                // 'name')` aliases the result property; `with([X.]times, n)` sets the
                // iteration count. `with([X.]target, …)` (shortestPath) is deferred.
                let mut key = self.ident()?;
                if self.peek() == Some(&Tok::Dot) {
                    self.bump();
                    key = self.ident()?; // strip the `PageRank.` / `PeerPressure.` prefix
                }
                self.expect(&Tok::Comma)?;
                match key.to_ascii_lowercase().as_str() {
                    "propertyname" => {
                        let name = self.str_arg()?;
                        self.expect(&Tok::RParen)?;
                        if let Some(slot) = self.last_algo {
                            // Rename, don't alias: the default property is no longer
                            // readable once `propertyName` redirects the result (the TS engine
                            // writes ONLY where asked), so drop the default mapping.
                            self.algo_props.retain(|_, &mut s| s != slot);
                            self.algo_props.insert(name, slot);
                        }
                        plan
                    }
                    "times" => {
                        let n = self.usize_arg()? as u32;
                        self.expect(&Tok::RParen)?;
                        // Re-issue the last AlgoAnnotate with the requested iterations.
                        match plan {
                            Plan::AlgoAnnotate {
                                input,
                                algo,
                                edge_label,
                                node_slot,
                            } => {
                                let algo = match algo {
                                    crate::ir::GremlinAlgo::PageRank { damping, .. } => {
                                        crate::ir::GremlinAlgo::PageRank {
                                            damping,
                                            iterations: n,
                                        }
                                    }
                                    crate::ir::GremlinAlgo::PeerPressure { .. } => {
                                        crate::ir::GremlinAlgo::PeerPressure { iterations: n }
                                    }
                                    other => other,
                                };
                                Plan::AlgoAnnotate {
                                    input,
                                    algo,
                                    edge_label,
                                    node_slot,
                                }
                            }
                            other => other,
                        }
                    }
                    "target" => {
                        // shortestPath().with(target, __.has('k'[, op(v)])): restrict the
                        // enumeration to paths whose destination matches the predicate.
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        let target = self.parse_match_has()?;
                        self.expect(&Tok::RParen)?;
                        match plan {
                            Plan::ShortestPathEnum {
                                input, node_slot, ..
                            } => Plan::ShortestPathEnum {
                                input,
                                node_slot,
                                target: Some(target),
                            },
                            _ => {
                                return Err("with(target, …) applies only to shortestPath()".into())
                            }
                        }
                    }
                    other => return Err(format!("with({other}, …) is not yet supported")),
                }
            }
            "barrier" => {
                // A lazy-barrier is a no-op in this eager executor (matching the TS engine, where
                // barrier() is the identity step). Bulk-collect semantics are invisible.
                self.expect(&Tok::RParen)?;
                plan
            }
            "shortestpath" => {
                // shortestPath(): for each source vertex, emit one path (list of vertex
                // external ids) per shortest undirected path to every reachable vertex.
                self.expect(&Tok::RParen)?;
                let p = plan.shortest_path_enum(self.current);
                self.current = 0;
                self.slots = 1;
                p
            }
            "tree" => {
                // tree()[.by('k')]: fold every traverser's vertex-hop path into one
                // nested Map. Only over a pure vertex-hop chain (path lineage = node
                // ids), like path().
                self.expect(&Tok::RParen)?;
                if !self.path_ok {
                    return Err(
                        "tree() is only supported over a pure vertex-hop chain (V-source \
                         + out/in/both); value projections and edge steps are deferred"
                            .into(),
                    );
                }
                let by = if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `by`
                    self.expect(&Tok::LParen)?;
                    let k = self.str_arg()?;
                    self.expect(&Tok::RParen)?;
                    Some(k)
                } else {
                    None
                };
                // A trailing `values('k')` (recorded by the values arm) folds in as the
                // leaf level of the tree.
                let leaf = self.path_leaf.take();
                let p = plan.tree(by, leaf);
                self.current = 0;
                self.slots = 1;
                p
            }
            "aggregate" | "store" => {
                // A named side-effect bag: record the CURRENT stream (plan prefix +
                // the current-slot value) under `key`, then pass through unchanged. In
                // this eager executor aggregate and store are identical (both eagerly
                // collect), matching the TS engine. Revealed later by cap(key).
                let key = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                self.caps
                    .insert(key, (plan.clone(), Expr::Slot(self.current)));
                plan
            }
            "subgraph" => {
                // Collect the current EDGE frontier into a named bag; cap('sg') reveals
                // it as a {vertices, edges} Map. Pass-through side effect (the snapshot
                // captures the plan prefix + the edge slot at this point).
                let key = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                self.subgraph_caps.insert(key, (plan.clone(), self.current));
                plan
            }
            "cap" => {
                // Reveal a bag. A subgraph bag → a {vertices, edges} Map (via
                // Plan::Subgraph over the snapshot); otherwise a single list, folding
                // the aggregate/store SNAPSHOT (AggFn::Collect keeps nulls) for
                // byte-identity with the TS engine.
                let key = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                if let Some((snap, edge_slot)) = self.subgraph_caps.get(&key).cloned() {
                    self.current = 0;
                    self.slots = 1;
                    return Ok(snap.subgraph(edge_slot));
                }
                let Some((snap, expr)) = self.caps.get(&key).cloned() else {
                    // An unfilled key caps to a single EMPTY list (the TS engine), not an error.
                    self.current = 0;
                    self.slots = 1;
                    return Ok(Plan::Row.project(vec![(key, Expr::Lit(Value::List(vec![])))]));
                };
                let p = snap.aggregate(
                    vec![],
                    vec![Agg {
                        func: AggFn::Collect,
                        arg: Some(expr),
                        distinct: false,
                        name: key,
                        frac: None,
                        null_on_empty: false,
                        numeric_only: false,
                    }],
                );
                self.current = 0;
                self.slots = 1;
                p
            }
            "inject" => {
                // inject(v1, v2, …): ADD the literal values to the WHOLE stream (a
                // whole-stream union with a literal-rows plan), not once per element.
                // Normalize the current stream to its single element column first so
                // both union arms are one column wide.
                let vals = self.literal_list()?;
                self.expect(&Tok::RParen)?;
                let cur = plan.project(vec![("inject".to_string(), Expr::Slot(self.current))]);
                // Literal rows: unwind the value list over a single Row (which pulls as
                // one dummy column), then project the appended element — which Unwind
                // places at the NEXT slot, i.e. slot 1 after Row's dummy slot 0.
                let lit_plan = Plan::Unwind {
                    input: Box::new(Plan::Row),
                    list: Box::new(Expr::Lit(Value::List(vals))),
                    var_slot: 1,
                    ordinal: None,
                }
                .project(vec![("inject".to_string(), Expr::Slot(1))]);
                // TinkerPop's `inject` PREPENDS the injected values to the incoming
                // stream (`g.V().inject(0)` → `[0, v1, …]`), so the literal rows are the
                // LEFT arm. Column name/width still agree (both one "inject" column).
                let p = Plan::Union {
                    left: Box::new(lit_plan),
                    right: Box::new(cur),
                    all: true,
                    op: crate::ir::CombineOp::Union,
                };
                self.current = 0;
                self.slots = 1;
                p
            }
            "unfold" => {
                // Flatten the current list-valued stream (Gremlin unfold): each list
                // element becomes its own traverser. Lowers to Unwind over the current
                // slot; a fold()/unfold() round-trips back to the element stream.
                self.expect(&Tok::RParen)?;
                let list = Expr::Slot(self.current);
                let var_slot = self.slots;
                self.slots += 1;
                let unwound = Plan::Unwind {
                    input: Box::new(plan),
                    list: Box::new(list),
                    var_slot,
                    ordinal: None,
                };
                // Project the unwound element as the single output column (as values()
                // does) so the terminal render reads it, not the original list slot.
                let p = unwound.project(vec![("unfold".to_string(), Expr::Slot(var_slot))]);
                self.current = 0;
                self.slots = 1;
                p
            }
            other => return Err(format!("unsupported Gremlin step `{other}`")),
        };
        // Edge-transparent steps operate ON the edge frontier and stay on it, so a
        // following inV()/outV()/otherV() still resolves the hop's landed endpoint —
        // re-arm the edge-hop pointer the top-of-step take() cleared. Covers every frontier-
        // PRESERVING filter / barrier / slice / order (TinkerPop tracks the path across them,
        // so `outE().range(0,2).hasLabel('X').otherV()` resolves — it faulted here before).
        if matches!(
            lname.as_str(),
            "has"
                | "hasnot"
                | "haslabel"
                | "hasid"
                | "haskey"
                | "hasvalue"
                | "subgraph"
                | "aggregate"
                | "store"
                | "sideeffect"
                | "dedup"
                | "where"
                | "and"
                | "or"
                | "not"
                | "is"
                | "filter"
                | "as"
                | "identity"
                | "barrier"
                | "order"
                | "limit"
                | "skip"
                | "range"
                | "tail"
                | "sample"
                | "coin"
                | "none"
                | "simplepath"
                | "cyclicpath"
                | "profile"
        ) {
            self.edge_hop = prev_edge_hop;
        }
        // Track whether the frontier now holds graph ELEMENTS (vertices/edges), known
        // statically from the step's output kind. Frontier-MOVE steps land on an element;
        // scalar/collection/map producers do not; frontier-PRESERVING filters/barriers
        // (has/where/dedup/limit/range/order/as/simplePath/…) leave it unchanged (the
        // catch-all). Ambiguous producers (union/coalesce/select/map/flatMap/unfold/cap/
        // inject/OLAP) reset to false — a MISSED element-fault is safe; a false one breaks
        // a valid query. (`sum`/`order` read this BEFORE it is updated for the step.)
        match lname.as_str() {
            "v" | "e" | "out" | "in" | "both" | "inv" | "outv" | "otherv" | "bothv" | "oute"
            | "ine" | "bothe" | "addv" | "adde" => {
                self.current_is_element = true;
                self.current_is_path = false;
                self.current_is_scalar = false;
            }
            // A path frontier: not an element, IS a path — sum/min/max/mean throws over it.
            "path" => {
                self.current_is_element = false;
                self.current_is_path = true;
                self.current_is_scalar = false;
            }
            // DEFINITE scalar producers (a number/string/id) — mirrors the pure-TS engine's
            // SCALAR_STEPS. These set `current_is_scalar` so the element-type-algebra guard in
            // `step` faults a following navigation/projection (`id().out()`, `count().inV()`).
            "values" | "value" | "id" | "label" | "count" | "sum" | "min" | "max" | "mean"
            | "math" | "loops" | "inject" => {
                self.current_is_element = false;
                self.current_is_path = false;
                self.current_is_scalar = true;
            }
            // Ambiguous / map / collection producers: NOT an element, but NOT a definite
            // scalar either (a `union`/`unfold`/`select`/`valueMap` output might be an element
            // or a map) — leave `current_is_scalar` false so nothing faults (a missed fault is
            // safe; a false one breaks a valid query, exactly as the TS 'unknown' frontier).
            "key" | "constant" | "signum" | "mult" | "pow" | "pi" | "propertyname" | "valuemap"
            | "elementmap" | "propertymap" | "properties" | "property" | "project" | "group"
            | "groupcount" | "fold" | "unfold" | "tree" | "cap" | "union" | "coalesce"
            | "choose" | "optional" | "branch" | "flatmap" | "map" | "select" | "sack"
            | "index" | "pagerank" | "peerpressure" | "connectedcomponent" | "shortestpath"
            | "subgraph" => {
                self.current_is_element = false;
                self.current_is_path = false;
                self.current_is_scalar = false;
            }
            _ => {} // filters / barriers / side-effects / modulators preserve the frontier
        }
        // Edge-in-scope tracking (port of the pure-TS `nextHasEdge`): an edge source makes an
        // edge reachable, a vertex move / scalar producer consumes/clears it, a preserving
        // filter/barrier keeps it, a branch already set its COMBINED value in its own arm, and
        // everything else is unknown.
        match lname.as_str() {
            "e" | "oute" | "ine" | "bothe" | "adde" => self.edge_scope = Some(true),
            "v" | "out" | "in" | "both" | "addv" | "inv" | "outv" | "bothv" | "otherv"
            | "values" | "value" | "id" | "label" | "count" | "sum" | "min" | "max" | "mean"
            | "math" | "loops" | "inject" => self.edge_scope = Some(false),
            // A branch lowering set `edge_scope` to the combination of its arms — keep it.
            "union" | "coalesce" | "choose" | "optional" | "branch" => {}
            // Frontier-PRESERVING filters / barriers / slices / side-effects keep the edge in
            // scope (matches TS PRESERVE_STEPS); any other producer clears it to unknown.
            other if is_edge_scope_preserving(other) => {}
            _ => self.edge_scope = None,
        }
        // A reducing barrier RESETS the traverser path (TinkerPop): a following `path()` yields
        // just its reduced value. A frontier-PRESERVING barrier after it (limit/skip/dedup/order)
        // keeps the reset, so `count().limit(2).path()` is still `[count]`.
        self.frontier_from_reducer = matches!(
            lname.as_str(),
            "count" | "sum" | "min" | "max" | "mean" | "fold"
        ) || (self.frontier_from_reducer
            && is_edge_scope_preserving(lname.as_str()));
        Ok(plan)
    }

    /// Parse the trailing `.from(V(id))` / `.to(V(id))` / `.property('k', v)`
    /// modulators of an `addE` and build the edge write. `from`/`to` may already be
    /// set (the `g.V(a).addE(...)` anchor sets `from`).
    fn finish_add_edge(
        &mut self,
        mut from: Option<u32>,
        mut to: Option<u32>,
        etype: String,
    ) -> Result<Plan, String> {
        let mut props = Vec::new();
        while self.peek() == Some(&Tok::Dot) {
            self.pos += 1;
            let step = self.ident()?;
            self.expect(&Tok::LParen)?;
            match step.to_ascii_lowercase().as_str() {
                "from" => {
                    from = Some(self.v_id_arg()?);
                    self.expect(&Tok::RParen)?;
                }
                "to" => {
                    to = Some(self.v_id_arg()?);
                    self.expect(&Tok::RParen)?;
                }
                "property" => {
                    let key = self.str_arg()?;
                    check_write_name("property key", &key)?;
                    self.expect(&Tok::Comma)?;
                    let val = self.literal()?;
                    self.expect(&Tok::RParen)?;
                    props.push((key, val));
                }
                other => return Err(format!("unsupported addE modulator `{other}`")),
            }
        }
        let from = from.ok_or("addE needs a from(V(id)) or a g.V(id) anchor")?;
        let to = to.ok_or("addE needs a to(V(id))")?;
        Ok(Plan::AddEdge {
            from,
            to,
            etype,
            props,
        })
    }

    /// A `V(<id>)` argument (the numeric node id).
    fn v_id_arg(&mut self) -> Result<u32, String> {
        let v = self.ident()?;
        if !v.eq_ignore_ascii_case("V") {
            return Err(format!("expected V(id), got `{v}`"));
        }
        self.expect(&Tok::LParen)?;
        let id = self.u_id()?;
        self.expect(&Tok::RParen)?;
        Ok(id)
    }

    /// A non-negative integer node id.
    fn u_id(&mut self) -> Result<u32, String> {
        match self.bump() {
            Some(Tok::Num(n)) if n >= 0.0 && n.fract() == 0.0 => Ok(n as u32),
            other => Err(format!("expected a node id, got {other:?}")),
        }
    }

    /// Apply a `property('k', v)` step. On an `addV` (a one-node `Insert`) it
    /// folds into that node's properties; on a read traversal it wraps (or extends)
    /// an `Update` that SETs the property on the current elements.
    fn apply_property(&self, plan: Plan, key: String, val: Expr) -> Plan {
        match plan {
            // `addV('L').property(k, <literal>)` folds the value straight into the new
            // node. A traversal-induced value (`constant`, a degree count) has no meaning
            // on a not-yet-created vertex, so it falls through to the post-insert Update.
            Plan::Insert { mut nodes, edges }
                if edges.is_empty() && nodes.len() == 1 && matches!(val, Expr::Lit(_)) =>
            {
                let Expr::Lit(v) = val else { unreachable!() };
                nodes[0].props.push((key, v));
                Plan::Insert { nodes, edges }
            }
            Plan::Update { input, mut ops } => {
                ops.push(crate::ir::SetOp::Set {
                    slot: self.current,
                    key,
                    value: val,
                });
                Plan::Update { input, ops }
            }
            read => Plan::Update {
                input: Box::new(read),
                ops: vec![crate::ir::SetOp::Set {
                    slot: self.current,
                    key,
                    value: val,
                }],
            },
        }
    }

    /// The value argument of `property(key, <value>)`. Besides a plain literal, TinkerPop
    /// accepts a child traversal evaluated per element (a "traversal-induced value"):
    /// `constant(v)` (a per-element constant) and a degree sub-traversal
    /// `[__.]<hop>('L'…).count()` (the out/in/both-degree of the current element). The
    /// child is rooted at the current traverser, so the count is over *its* neighbours —
    /// the same `CountSubquery` the `order().by(<degree>)` body builds.
    fn property_value_expr(&mut self) -> Result<Expr, String> {
        // `constant(v)` — a per-element constant.
        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("constant")) {
            self.bump();
            self.expect(&Tok::LParen)?;
            let v = self.literal()?;
            self.expect(&Tok::RParen)?;
            return Ok(Expr::Lit(v));
        }
        // A degree sub-traversal `[__.]<hop>('L'…).count()`.
        let is_hop = {
            let mut p = self.pos;
            if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
                p += 1;
                if self.toks.get(p) == Some(&Tok::Dot) {
                    p += 1;
                }
            }
            matches!(self.toks.get(p), Some(Tok::Ident(s)) if matches!(
                s.to_ascii_lowercase().as_str(),
                "out" | "in" | "both" | "oute" | "ine" | "bothe"))
        };
        if is_hop {
            return self.degree_count_subquery("property(key, <traversal>)");
        }
        // Otherwise a plain literal value.
        Ok(Expr::Lit(self.literal()?))
    }

    /// Parse a degree sub-traversal `[__.]<hop>('L'…)[.values('k')].count()` rooted at the
    /// current slot into a [`Expr::CountSubquery`] — the count of the current element's
    /// neighbours along `<hop>`. Shared by `order().by(<degree>)`, `select().by(<degree>)`
    /// and `property(key, <degree>)`; `ctx` names the caller for the error message.
    fn degree_count_subquery(&mut self, ctx: &str) -> Result<Expr, String> {
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        let hop = self.ident()?.to_ascii_lowercase();
        let (dir, is_edge) = match hop.as_str() {
            "out" => (Dir::Out, false),
            "in" => (Dir::In, false),
            "both" => (Dir::Both, false),
            "oute" => (Dir::Out, true),
            "ine" => (Dir::In, true),
            "bothe" => (Dir::Both, true),
            other => return Err(format!("{ctx}: unsupported body `{other}`")),
        };
        self.expect(&Tok::LParen)?;
        let mut labels: Vec<String> = Vec::new();
        if matches!(self.peek(), Some(Tok::Str(_))) {
            labels.push(self.str_arg()?);
            while self.peek() == Some(&Tok::Comma) {
                self.bump();
                labels.push(self.str_arg()?);
            }
        }
        self.expect(&Tok::RParen)?;
        // Optional intermediate `.values('k')` (counts present names — same cardinality as
        // the hop, so it does not change a count).
        if self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("values"))
        {
            self.expect(&Tok::Dot)?;
            self.ident()?; // values
            self.expect(&Tok::LParen)?;
            self.str_arg()?;
            self.expect(&Tok::RParen)?;
        }
        self.expect(&Tok::Dot)?;
        let c = self.ident()?;
        if !c.eq_ignore_ascii_case("count") {
            return Err(format!("{ctx} must end with .count()"));
        }
        self.expect(&Tok::LParen)?;
        self.expect(&Tok::RParen)?;
        let body = if is_edge {
            Plan::Row.expand_edge_gremlin(self.current, dir, &labels)
        } else {
            Plan::Row.expand(self.current, dir, &labels)
        };
        Ok(Expr::CountSubquery {
            body: Box::new(body),
            outer_width: self.slots,
        })
    }

    /// The second argument of `has('k', …)`: a predicate against property `key`.
    fn has_predicate(&mut self, key: String) -> Result<Expr, String> {
        let left = Expr::Prop {
            slot: self.current,
            key: key.clone(),
        };
        // TinkerPop: `has(k, neq(v))` / `has(k, without(…))` KEEP an element that lacks
        // `k` (the negation is vacuously satisfied), unlike a positive predicate whose
        // 3VL drops a missing key. (The TEXT negations — notContaining etc. — instead
        // require the key present, like the positive text predicates, so they are NOT in
        // this set.) Lower neq/without as `NOT(PropertyExists(k) AND <positive base>)`.
        let mut probe = self.pos;
        if matches!(self.toks.get(probe), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("P") || s.eq_ignore_ascii_case("TextP"))
            && self.toks.get(probe + 1) == Some(&Tok::Dot)
        {
            probe += 2;
        }
        let is_negated = matches!(self.toks.get(probe), Some(Tok::Ident(s)) if matches!(
            s.to_ascii_lowercase().as_str(),
            "neq" | "without"
        ));
        if is_negated {
            let pred = self.predicate_expr(left)?;
            // predicate_expr already produced the NEGATED form (`Compare(Ne, …)` or
            // `Not(base)`); recover the POSITIVE base and guard it with presence.
            let base = match pred {
                Expr::Not(inner) => *inner,
                Expr::Compare {
                    op: CompareOp::Ne,
                    left,
                    right,
                } => Expr::Compare {
                    op: CompareOp::Eq,
                    left,
                    right,
                },
                other => return Ok(other), // not a recognized negation; leave as-is
            };
            let exists = Expr::PropertyExists {
                slot: self.current,
                key,
            };
            return Ok(Expr::Not(Box::new(Expr::And(
                Box::new(exists),
                Box::new(base),
            ))));
        }
        self.predicate_expr(left)
    }

    /// Parse ONE child traversal of `and`/`or`/`not` as a boolean predicate `Expr`
    /// over the current element — without applying a filter. v1 accepts only element
    /// FILTERS: `has('k')`, `has('k', pred)`, `hasNot('k')`, and nested
    /// `and`/`or`/`not` of those (an optional `__.` anonymous-traversal prefix is
    /// allowed). A navigating child (`out('X')`, …) is a semi-join needing a
    /// subquery and is deferred with an explicit error. Leaves the cursor after the
    /// child's own closing `)`.
    fn child_filter_expr(&mut self) -> Result<Expr, String> {
        let start = self.pos;
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        let name = self.ident()?.to_ascii_lowercase();
        self.expect(&Tok::LParen)?;
        let expr = match name.as_str() {
            "has" => {
                let key = self.str_arg()?;
                let e = if self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    // In a FILTER-CHILD position (not/and/or), the predicate must be
                    // 2-VALUED so `not(has('n',1))` sees a definite false for a stored
                    // null (3VL null would make Not→null→dropped, but the TS engine keeps it).
                    // Coerce the (possibly-null) predicate to false-unless-true.
                    let pred = self.has_predicate(key)?;
                    Expr::Case {
                        branches: vec![(pred, Expr::Lit(Value::Bool(true)))],
                        otherwise: Some(Box::new(Expr::Lit(Value::Bool(false)))),
                    }
                } else {
                    Expr::PropertyExists {
                        slot: self.current,
                        key,
                    }
                };
                self.expect(&Tok::RParen)?;
                e
            }
            "hasnot" => {
                let key = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                Expr::Not(Box::new(Expr::PropertyExists {
                    slot: self.current,
                    key,
                }))
            }
            "and" | "or" => {
                let parts = self.child_filter_list()?;
                self.expect(&Tok::RParen)?;
                let mut it = parts.into_iter();
                let first = it
                    .next()
                    .ok_or_else(|| format!("{name}() needs at least one child traversal"))?;
                it.fold(first, |acc, e| {
                    if name == "and" {
                        Expr::And(Box::new(acc), Box::new(e))
                    } else {
                        Expr::Or(Box::new(acc), Box::new(e))
                    }
                })
            }
            "not" => {
                let inner = self.child_filter_expr()?;
                self.expect(&Tok::RParen)?;
                Expr::Not(Box::new(inner))
            }
            // A navigating hop child is a semi-join predicate: does the current
            // element HAVE such an adjacency? It builds the same `Expr::Exists` the
            // `where(<hop>)` step does, so `not(out('L'))` is the anti-join and
            // `and(out('L'), has(…))` mixes an edge test with a property test.
            "haslabel" => {
                let mut ls = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    ls.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                label_membership(self.current, &ls)
            }
            // `values('k')` in filter position is a presence test; a trailing
            // `.is(<pred>)` narrows it to a value predicate. Expressed inline
            // (PropertyExists [AND Compare]) rather than an Exists over a
            // projection body — a projection collapses the batch and drops the
            // provenance column the Exists machinery reads.
            "values" => {
                let key = self.str_arg()?;
                if self.peek() == Some(&Tok::Comma) {
                    return Err(
                        "values() with multiple keys is not supported in a filter child".into(),
                    );
                }
                self.expect(&Tok::RParen)?;
                let exists = Expr::PropertyExists {
                    slot: self.current,
                    key: key.clone(),
                };
                if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("is"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // is
                    self.expect(&Tok::LParen)?;
                    let pred = self.predicate_expr(Expr::Prop {
                        slot: self.current,
                        key: key.clone(),
                    })?;
                    self.expect(&Tok::RParen)?;
                    // 2-valued (like the has child): `not(values('n').is(gt(5)))` must see
                    // a definite false for a stored/absent null, not 3VL null → dropped.
                    Expr::Case {
                        branches: vec![(
                            Expr::And(Box::new(exists), Box::new(pred)),
                            Expr::Lit(Value::Bool(true)),
                        )],
                        otherwise: Some(Box::new(Expr::Lit(Value::Bool(false)))),
                    }
                } else {
                    exists
                }
            }
            // `label()`/`id()` in filter position, optionally narrowed by `.is(pred)` —
            // `filter(label().is(eq('PERSON')))`. Inline (a Call [compared]) rather than
            // an Exists over a projection body.
            "label" | "id" => {
                self.expect(&Tok::RParen)?;
                let val = Expr::Call {
                    name: if name == "label" {
                        "element_label".into()
                    } else {
                        "element_id".into()
                    },
                    args: vec![Expr::Slot(self.current)],
                };
                if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("is"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // is
                    self.expect(&Tok::LParen)?;
                    let pred = self.predicate_expr(val)?;
                    self.expect(&Tok::RParen)?;
                    pred
                } else {
                    // A bare `label()`/`id()` always yields a value → always true.
                    Expr::Lit(Value::Bool(true))
                }
            }
            "out" | "in" | "both" | "oute" | "ine" | "bothe" => {
                let dir = match name.as_str() {
                    "out" | "oute" => Dir::Out,
                    "in" | "ine" => Dir::In,
                    _ => Dir::Both,
                };
                let mut labels: Vec<String> = Vec::new();
                if matches!(self.peek(), Some(Tok::Str(_))) {
                    labels.push(self.str_arg()?);
                    while self.peek() == Some(&Tok::Comma) {
                        self.bump();
                        labels.push(self.str_arg()?);
                    }
                }
                self.expect(&Tok::RParen)?;
                let is_edge = matches!(name.as_str(), "oute" | "ine" | "bothe");
                let hop = if is_edge {
                    Plan::Row.expand_edge_gremlin(self.current, dir, &labels)
                } else {
                    Plan::Row.expand(self.current, dir, &labels)
                };
                // A trailing `.values('k').<agg>().is(<pred>)` is a correlated reducing
                // scalar test — `where(in('CREATED').values('age').mean().is(inside(…)))`.
                // Collect the neighbours' `k` per outer row (CollectSubquery), reduce it
                // with the list aggregate, then compare (an aggregate INSIDE an Exists
                // body would collapse its provenance, hence the collect-then-reduce).
                if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("values"))
                    && !is_edge
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // values
                    self.expect(&Tok::LParen)?;
                    let k = self.str_arg()?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                    let agg = self.ident()?.to_ascii_lowercase();
                    let list_fn = match agg.as_str() {
                        "mean" => "list_mean",
                        "sum" => "list_sum",
                        "min" => "list_min",
                        "max" => "list_max",
                        "count" => "list_count",
                        other => {
                            return Err(format!("filter child values(...).{other}() unsupported"))
                        }
                    };
                    self.expect(&Tok::LParen)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                    let isn = self.ident()?;
                    if !isn.eq_ignore_ascii_case("is") {
                        return Err("values(...).<agg>() in a filter child needs is(...)".into());
                    }
                    self.expect(&Tok::LParen)?;
                    // scalar = the neighbour's `k`, shifted past the provenance column.
                    let mut scalar = Expr::Prop {
                        slot: self.slots,
                        key: k,
                    };
                    shift_body_slots(&mut scalar, self.slots);
                    let collected = Expr::CollectSubquery {
                        body: Box::new(hop),
                        scalar: Box::new(scalar),
                        outer_width: self.slots,
                    };
                    let reduced = Expr::Call {
                        name: list_fn.to_string(),
                        args: vec![collected],
                    };
                    let pred = self.predicate_expr(reduced)?;
                    self.expect(&Tok::RParen)?;
                    return Ok(pred);
                }
                // A trailing `.count().is(<pred>)` on the child is a DEGREE test — a
                // correlated CountSubquery compared per the predicate (an aggregate
                // inside an Exists body would collapse its provenance). Otherwise the
                // child is an existence semi-join.
                if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("count"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // count
                    self.expect(&Tok::LParen)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                    let isn = self.ident()?; // is
                    if !isn.eq_ignore_ascii_case("is") {
                        return Err("count() in a filter child must be followed by is(...)".into());
                    }
                    self.expect(&Tok::LParen)?;
                    let cnt = Expr::CountSubquery {
                        body: Box::new(hop),
                        outer_width: self.slots,
                    };
                    let pred = self.predicate_expr(cnt)?;
                    self.expect(&Tok::RParen)?;
                    pred
                } else if self.peek() == Some(&Tok::Dot) {
                    // A trailing step (`out('KNOWS').where(out('CREATED'))`) makes the
                    // child a multi-step sub-traversal: Exists over the whole chain.
                    // Reserve the provenance slot (parse at width+1) so an inner filter
                    // or nested Exists numbers new slots PAST it — the correlated exec
                    // inserts a provenance column at `outer_width`.
                    self.pos = start;
                    let width = self.slots;
                    let (body, _oc, _os) =
                        self.parse_sub_body_seeded(Plan::Row, self.current, width + 1)?;
                    Expr::Exists {
                        body: Box::new(body),
                        outer_width: width,
                    }
                } else {
                    Expr::Exists {
                        body: Box::new(hop),
                        outer_width: self.slots,
                    }
                }
            }
            // Any other child: a general sub-traversal filter — keep the element if the
            // body produces ≥1 output (Exists over the Row-rooted sub-plan). Reserve the
            // provenance slot (parse at width+1), as the trailing-step case above.
            _ => {
                self.pos = start;
                let width = self.slots;
                let (body, _oc, _os) =
                    self.parse_sub_body_seeded(Plan::Row, self.current, width + 1)?;
                Expr::Exists {
                    body: Box::new(body),
                    outer_width: width,
                }
            }
        };
        // Chained ELEMENT filters conjoin: `has('n',1).has('n',1)`, `hasLabel(..).has(..)`.
        // (A hop-led chain like `out().where(...)` is a nested sub-traversal, handled by
        // the catch-all above, not conjoined here.)
        let mut expr = expr;
        while self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if {
                matches!(s.to_ascii_lowercase().as_str(),
                    "has" | "hasnot" | "haslabel" | "haskey" | "hasvalue")
            })
        {
            self.bump(); // the `.`
            let next = self.child_filter_expr()?;
            expr = Expr::And(Box::new(expr), Box::new(next));
        }
        Ok(expr)
    }

    /// Parse a Gremlin `math('…')` expression string into an engine `Expr`. `operand`
    /// resolves the `_` variable (and any named step-label variable via `var_slot`).
    /// Grammar precedence (mXparser/TinkerPop): `+ -` < `* / %` < `^` (right-assoc) <
    /// unary `- +` < primary (number, `(expr)`, `name(args)`, bare unary `sin _`,
    /// constant `pi`/`e`, or a variable). Maps to the same f64 kernels GQL uses.
    fn parse_math(&self, src: &str, operand: &Expr) -> Result<Expr, String> {
        // math() is an evaluation sublanguage: every failure — unknown function,
        // unknown variable, malformed expression — is a value error, matching the
        // pure-TS engine (which throws E_INVALID_VALUE for all of them). The prefix
        // routes it past the Gremlin parser's default E_SYNTAX classification at the
        // FFI boundary. See `crate::ffi` (Gremlin parse-error branch).
        self.parse_math_inner(src, operand).map_err(|e| {
            if e.starts_with("E_INVALID_VALUE: ") {
                e
            } else {
                format!("E_INVALID_VALUE: {e}")
            }
        })
    }

    fn parse_math_inner(&self, src: &str, operand: &Expr) -> Result<Expr, String> {
        let toks = math_lex(src)?;
        let mut mp = MathParser {
            toks,
            pos: 0,
            operand,
            labels: &self.labels,
        };
        let e = mp.expr()?;
        if mp.pos != mp.toks.len() {
            return Err(format!("math('{src}'): trailing tokens"));
        }
        Ok(e)
    }

    /// Parse a match hop head `as('s').<hop>('L'?)[.as('e')]`, returning
    /// `(start_tag, direction, edge_label, end_tag?)`. Shared by the `not(...)`
    /// fragment (cursor already past any leading `__.`).
    #[allow(clippy::type_complexity)]
    fn match_hop_head(&mut self) -> Result<(String, Dir, Option<String>, Option<String>), String> {
        let as1 = self.ident()?;
        if !as1.eq_ignore_ascii_case("as") {
            return Err("match() fragment must start with as('tag')".into());
        }
        self.expect(&Tok::LParen)?;
        let start = self.str_arg()?;
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Dot)?;
        let dir = match self.ident()?.to_ascii_lowercase().as_str() {
            "out" => Dir::Out,
            "in" => Dir::In,
            "both" => Dir::Both,
            other => return Err(format!("match() hop must be out/in/both, got `{other}`")),
        };
        self.expect(&Tok::LParen)?;
        let label = if matches!(self.peek(), Some(Tok::Str(_))) {
            Some(self.str_arg()?)
        } else {
            None
        };
        self.expect(&Tok::RParen)?;
        let end = if self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("as"))
        {
            self.expect(&Tok::Dot)?;
            self.ident()?; // as
            self.expect(&Tok::LParen)?;
            let e = self.str_arg()?;
            self.expect(&Tok::RParen)?;
            Some(e)
        } else {
            None
        };
        Ok((start, dir, label, end))
    }

    /// Parse a match `has('k'[, [P.]op(v)])` filter (cursor AT the `has` ident),
    /// returning the key and an optional `(comparison, literal)` bound.
    fn parse_match_has(&mut self) -> Result<(String, Option<(CompareOp, Value)>), String> {
        let h = self.ident()?;
        if !h.eq_ignore_ascii_case("has") {
            return Err("match() filter fragment expects has(...)".into());
        }
        self.expect(&Tok::LParen)?;
        let key = self.str_arg()?;
        let pred = if self.peek() == Some(&Tok::Comma) {
            self.bump();
            // `has('k', <literal>)` is equality; `has('k', [P.]op(v))` a comparison.
            let is_op = matches!(self.peek(), Some(Tok::Ident(s)) if {
                let l = s.to_ascii_lowercase();
                s.as_str() == "P" || matches!(l.as_str(), "eq"|"neq"|"gt"|"gte"|"lt"|"lte")
            }) && self.toks.get(self.pos + 1) == Some(&Tok::Dot)
                || matches!(self.peek(), Some(Tok::Ident(s)) if {
                    matches!(s.to_ascii_lowercase().as_str(), "eq"|"neq"|"gt"|"gte"|"lt"|"lte")
                }) && self.toks.get(self.pos + 1) == Some(&Tok::LParen);
            if !is_op {
                // A bare literal value → equality.
                let v = self.literal()?;
                Some((CompareOp::Eq, v))
            } else {
                if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("P")) {
                    self.bump();
                    self.expect(&Tok::Dot)?;
                }
                let op = self.ident()?.to_ascii_lowercase();
                self.expect(&Tok::LParen)?;
                let v = self.literal()?;
                self.expect(&Tok::RParen)?;
                let cop = match op.as_str() {
                    "eq" => CompareOp::Eq,
                    "neq" => CompareOp::Ne,
                    "gt" => CompareOp::Gt,
                    "gte" => CompareOp::Ge,
                    "lt" => CompareOp::Lt,
                    "lte" => CompareOp::Le,
                    other => return Err(format!("match() has(): unsupported predicate `{other}`")),
                };
                Some((cop, v))
            }
        } else {
            None
        };
        self.expect(&Tok::RParen)?;
        Ok((key, pred))
    }

    /// Try to parse ONE single-VALUE sub-traversal body — `[__.]constant(v)`,
    /// `[__.]values('k')` (single key), `[__.]id()`, `[__.]label()` — into the `Expr`
    /// it yields per row (reading the element at `from`). Returns `None` with the
    /// cursor restored when the body is not one of these (e.g. a hop). Used by
    /// `choose` to decide between the Case and the union-of-hops lowering.
    fn parse_single_value_body(&mut self, from: usize) -> Result<Option<Expr>, String> {
        let save = self.pos;
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            if self.peek() == Some(&Tok::Dot) {
                self.bump();
            } else {
                self.pos = save;
                return Ok(None);
            }
        }
        let name = match self.peek() {
            Some(Tok::Ident(s)) => s.to_ascii_lowercase(),
            _ => {
                self.pos = save;
                return Ok(None);
            }
        };
        let expr = match name.as_str() {
            "constant" => {
                self.bump();
                self.expect(&Tok::LParen)?;
                let v = self.literal()?;
                self.expect(&Tok::RParen)?;
                Expr::Lit(v)
            }
            "values" => {
                self.bump();
                self.expect(&Tok::LParen)?;
                let k = self.str_arg()?;
                // A multi-key values() is not a single value — leave it for another path.
                if self.peek() == Some(&Tok::Comma) {
                    self.pos = save;
                    return Ok(None);
                }
                self.expect(&Tok::RParen)?;
                Expr::Prop { slot: from, key: k }
            }
            "id" | "label" => {
                self.bump();
                self.expect(&Tok::LParen)?;
                self.expect(&Tok::RParen)?;
                Expr::Call {
                    name: if name == "id" {
                        "element_id".into()
                    } else {
                        "element_label".into()
                    },
                    args: vec![Expr::Slot(from)],
                }
            }
            _ => {
                self.pos = save;
                return Ok(None);
            }
        };
        Ok(Some(expr))
    }

    /// Try to parse the whole coalesce argument list as bodies that are EACH a single
    /// `[__.]values('k')` projection, consuming through the closing `)`. Returns the
    /// keys in order, or `None` (cursor restored) if any body is something else.
    fn try_all_values_bodies(&mut self) -> Option<Vec<String>> {
        let save = self.pos;
        let mut keys = Vec::new();
        loop {
            if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                self.bump();
                if self.peek() == Some(&Tok::Dot) {
                    self.bump();
                } else {
                    self.pos = save;
                    return None;
                }
            }
            match self.peek() {
                Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("values") => self.bump(),
                _ => {
                    self.pos = save;
                    return None;
                }
            };
            if self.peek() != Some(&Tok::LParen) {
                self.pos = save;
                return None;
            }
            self.bump();
            let k = match self.peek().cloned() {
                Some(Tok::Str(s)) => {
                    self.bump();
                    s
                }
                _ => {
                    self.pos = save;
                    return None;
                }
            };
            if self.peek() != Some(&Tok::RParen) {
                self.pos = save;
                return None;
            }
            self.bump();
            keys.push(k);
            match self.peek() {
                Some(Tok::Comma) => {
                    self.bump();
                }
                Some(Tok::RParen) => {
                    self.bump();
                    break;
                }
                _ => {
                    self.pos = save;
                    return None;
                }
            }
        }
        Some(keys)
    }

    /// Parse an anonymous sub-traversal `__.step().step()…` (a branch/union/coalesce/
    /// local body) into a `Plan::Row`-rooted sub-plan that correlates on the incoming
    /// element at slot `from` (row width `width`). Returns the sub-plan and the slot /
    /// width the body LANDS at, so the caller can set the post-branch frontier. The
    /// parser's current/slots/edge-hop/repeat state is saved and restored around it.
    /// Record a value-producing step's frontier into the Gremlin step-history when a full
    /// `path()`/`tree()` is read (`building_full_path`) — a vertex/edge move or a value
    /// projection. Filters/barriers add nothing. Used by both the top-level chain AND a branch
    /// arm's chain, so a `path()` after/inside a branch sees the arm's hops.
    fn maybe_path_record(&self, plan: Plan, step_name: &str) -> Plan {
        if !self.building_full_path {
            return plan;
        }
        let tag = match step_name {
            "oute" | "ine" | "bothe" => Some(crate::batch::STEP_EDGE),
            "out" | "in" | "both" | "inv" | "outv" | "otherv" | "bothv" => {
                Some(crate::batch::STEP_NODE)
            }
            // A `path()` output is itself a path element (so `path().path()` nests it);
            // it records as a raw scalar — the produced list, rendered verbatim.
            "values" | "value" | "id" | "label" | "key" | "path" => Some(crate::batch::STEP_SCALAR),
            _ => None,
        };
        match tag {
            Some(tag) => Plan::PathRecord {
                input: Box::new(plan),
                value: Expr::Slot(self.current),
                tag,
            },
            None => plan,
        }
    }

    fn parse_sub_body(
        &mut self,
        from: usize,
        width: usize,
    ) -> Result<(Plan, usize, usize), String> {
        self.parse_sub_body_seeded(Plan::Row, from, width)
    }

    /// Like [`parse_sub_body`] but the body chains onto `seed` (a `Plan::Row`, or a
    /// `Row` pre-filtered by a coalesce exclusion guard) instead of a bare `Row`.
    fn parse_sub_body_seeded(
        &mut self,
        seed: Plan,
        from: usize,
        width: usize,
    ) -> Result<(Plan, usize, usize), String> {
        let saved_current = self.current;
        let saved_slots = self.slots;
        // The body INHERITS the current edge-hop record (usually None), so a branch arm
        // whose outer frontier is an edge reached THROUGH a vertex (`V().outE()`) lets a
        // leading `otherV()`/`inV()` resolve against that origin — the branch handlers
        // seed `edge_hop` with the outer hop before calling. A bare-edge outer frontier
        // (`E()`) has no such record, so the arm's `otherV()` still faults.
        let saved_edge = self.edge_hop;
        let saved_repeat = self.pending_repeat.take();
        let saved_path_ok = self.path_ok;
        let saved_on_edge = self.on_edge;
        // An arm starts fresh from the BRANCH frontier, so its `path()` answerability is the
        // branch input's — NOT the branch step's own post-taint value. The classifier taints
        // `path_ok=false` for the branch step (union/coalesce/… are not path-preserving) before
        // this runs, which would push a pure-vertex-hop arm's `path()` onto the (unrecorded)
        // step-history instead of the node-lineage the arm's hops actually extend. Restore the
        // pre-taint value so `union(out().path())` reads `[src, nbr]`, not `[src]`.
        self.path_ok = self.path_ok_pre_step;
        // Each branch arm starts from the frontier AT the branch (the input), not from the
        // previous arm's output — save/restore the frontier-kind flags so the element-type
        // guard classifies every arm's first step against the branch input (mirrors the
        // pure-TS checkSteps, which recurses per-arm from the branch frontier).
        let saved_is_element = self.current_is_element;
        let saved_is_scalar = self.current_is_scalar;
        let saved_is_path = self.current_is_path;
        let saved_is_map = self.current_is_map;
        let saved_edge_scope = self.edge_scope;
        // Each arm starts from the branch-input frontier, so a reducer in one arm must NOT leak
        // its path-reset into the NEXT arm's `path()` lowering: `union(count(), path())` left
        // `frontier_from_reducer` set after the count() arm, so the path() arm rendered just its
        // reduced value `[dst]` instead of the full `[src, dst]` — order-dependently (path() first
        // was fine). Restore to the branch-input value at exit (below) so the next arm is clean.
        let saved_from_reducer = self.frontier_from_reducer;
        self.current = from;
        self.slots = width;
        // Optional leading `__.` (an anonymous traversal). The body is then a `.`-
        // separated step chain; the FIRST step may appear bare (`out(...)`) or after
        // the `__.` — both spellings occur in branch bodies.
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        let mut body = seed;
        // Parse the arm's step chain, recording each value-producing step into the step-history
        // when a full path()/tree() is read — so a path() after/inside a branch sees the arm's
        // hops (`union(dedup().bothE(...), …).path()`), not just the seed.
        let step_name_of = |p: &Self| match p.peek() {
            Some(Tok::Ident(s)) => s.to_ascii_lowercase(),
            _ => String::new(),
        };
        let mut last_step = String::new();
        if matches!(self.peek(), Some(Tok::Ident(_))) {
            let name = step_name_of(self);
            body = self.step(body)?;
            body = self.maybe_path_record(body, &name);
            last_step = name;
            while self.peek() == Some(&Tok::Dot) {
                self.bump();
                let name = step_name_of(self);
                body = self.step(body)?;
                body = self.maybe_path_record(body, &name);
                last_step = name;
            }
        }
        let out_current = self.current;
        let out_slots = self.slots;
        // Capture the arm's OUTPUT edge-in-scope + frontier KIND + terminal step before restoring,
        // so a branch lowering can combine its arms (and `optional` can take an always-producing
        // body's ending frontier).
        self.last_arm_edge_scope = self.edge_scope;
        self.last_arm_frontier = (
            self.current_is_element,
            self.current_is_scalar,
            self.current_is_path,
        );
        self.last_arm_last_step = last_step;
        self.current = saved_current;
        self.slots = saved_slots;
        self.edge_hop = saved_edge;
        self.pending_repeat = saved_repeat;
        self.path_ok = saved_path_ok;
        self.on_edge = saved_on_edge;
        self.current_is_element = saved_is_element;
        self.current_is_scalar = saved_is_scalar;
        self.current_is_path = saved_is_path;
        self.current_is_map = saved_is_map;
        self.edge_scope = saved_edge_scope;
        self.frontier_from_reducer = saved_from_reducer;
        Ok((body, out_current, out_slots))
    }

    /// Peek-parse a per-element aggregate-terminal branch arm for coalesce/choose/
    /// optional (NOT union, which is whole-stream): an optional single navigating hop,
    /// an optional `values('k')`, then `count`/`fold`/`sum`/`min`/`max`/`mean`, arm-
    /// terminal (`,`/`)`). Returns the correlated per-outer-row aggregate expression —
    /// one value per incoming element (bare `count()` counts the self-row → 1, bare
    /// `fold()` collects the self-row → `[self]`, `out().count()` is the per-element
    /// out-degree). Returns `None` with the cursor restored when the arm is not that
    /// shape, so it falls through to the whole-stream branch body.
    fn try_per_element_agg(&mut self, from: usize, width: usize) -> Result<Option<Expr>, String> {
        let save = self.pos;
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            if self.peek() == Some(&Tok::Dot) {
                self.bump();
            }
        }
        // Optional single navigating hop; else reduce over the element itself. The
        // neighbour lands one past the provenance column (inserted at `width`), mirroring
        // `project_by_body`.
        let hop_dir = match self.peek() {
            Some(Tok::Ident(s)) => match s.to_ascii_lowercase().as_str() {
                "out" => Some((Dir::Out, false)),
                "in" => Some((Dir::In, false)),
                "both" => Some((Dir::Both, false)),
                "oute" => Some((Dir::Out, true)),
                "ine" => Some((Dir::In, true)),
                "bothe" => Some((Dir::Both, true)),
                _ => None,
            },
            _ => None,
        };
        // A leading hop navigates FROM a vertex. If the incoming frontier is DEFINITELY not a
        // vertex (an edge or scalar branch input, `inE(…).choose(outE(…).count(), …)`), bail so
        // the general arm path faults via the adjacency/edge-hop guard rather than silently
        // building a subquery. (An UNKNOWN frontier might be a vertex — keep building.)
        if hop_dir.is_some() && (self.on_edge || self.current_is_scalar) {
            self.pos = save;
            return Ok(None);
        }
        let (mut body, landed) = if let Some((dir, is_edge)) = hop_dir {
            self.bump();
            if self.expect(&Tok::LParen).is_err() {
                self.pos = save;
                return Ok(None);
            }
            let mut labels: Vec<String> = Vec::new();
            if matches!(self.peek(), Some(Tok::Str(_))) {
                labels.push(self.str_arg()?);
                while self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    labels.push(self.str_arg()?);
                }
            }
            if self.expect(&Tok::RParen).is_err() {
                self.pos = save;
                return Ok(None);
            }
            if self.peek() != Some(&Tok::Dot) {
                self.pos = save;
                return Ok(None);
            }
            self.bump();
            let b = if is_edge {
                Plan::Row.expand_edge_gremlin(from, dir, &labels)
            } else {
                Plan::Row.expand(from, dir, &labels)
            };
            (b, width + 1)
        } else {
            (Plan::Row, from)
        };
        // A BARE leading barrier (skip/limit/range, no hop) empties or keeps the single self-row
        // per element: `skip(1).count()` → 0, `limit(0).fold()` → [], `range(0,1).count()` → 1.
        // (Only the bare case — after a hop a barrier slices the NEIGHBOUR set, handled elsewhere.)
        let mut bare_empty = false;
        if hop_dir.is_none() {
            while matches!(self.peek(), Some(Tok::Ident(s)) if {
                let l = s.to_ascii_lowercase();
                (l == "skip" || l == "limit" || l == "range") && self.toks.get(self.pos + 1) == Some(&Tok::LParen)
            }) {
                let b = self.ident()?.to_ascii_lowercase();
                self.expect(&Tok::LParen)?;
                match b.as_str() {
                    "skip" => {
                        if self.usize_arg()? > 0 {
                            bare_empty = true; // skip(n>0) drops the single self-row
                        }
                    }
                    "limit" => {
                        if self.usize_arg()? == 0 {
                            bare_empty = true; // limit(0) drops it
                        }
                    }
                    _ => {
                        // range(lo, hi): the size-1 self-row survives iff lo==0 and hi>lo.
                        let lo = self.usize_arg()?;
                        self.expect(&Tok::Comma)?;
                        let hi = self.usize_arg()?;
                        if !(lo == 0 && hi > 0) {
                            bare_empty = true;
                        }
                    }
                }
                if self.expect(&Tok::RParen).is_err() || self.peek() != Some(&Tok::Dot) {
                    self.pos = save;
                    return Ok(None);
                }
                self.bump();
            }
        }
        // A BARE leading FILTER before the reducer (`has('k', …).count()`,
        // `hasLabel('L').fold()`) keeps or drops the single self-row per element, so it folds
        // into the correlated body: count()=1/0, fold()=[self]/[]. (Bare case only; a filter
        // after a hop is on the neighbour and stays in the general path.)
        if hop_dir.is_none() && !bare_empty {
            let saved_cur = self.current;
            self.current = from; // the filter reads the self element
            while matches!(self.peek(), Some(Tok::Ident(s)) if {
                let l = s.to_ascii_lowercase();
                matches!(
                    l.as_str(),
                    "has" | "hasnot" | "haslabel" | "hasid" | "haskey" | "hasvalue" | "where"
                        | "and" | "or" | "not" | "is"
                )
            }) {
                match self.child_filter_expr() {
                    Ok(pred) => body = body.filter(pred),
                    Err(_) => {
                        self.current = saved_cur;
                        self.pos = save;
                        return Ok(None);
                    }
                }
                if self.peek() == Some(&Tok::Dot) {
                    self.bump();
                } else {
                    break;
                }
            }
            self.current = saved_cur;
        }
        // Optional single-key `values('k')` before the reducer.
        let mut val_key: Option<String> = None;
        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("values")) {
            self.bump();
            if self.expect(&Tok::LParen).is_err() {
                self.pos = save;
                return Ok(None);
            }
            if !matches!(self.peek(), Some(Tok::Str(_))) {
                self.pos = save;
                return Ok(None);
            }
            let k = self.str_arg()?;
            // A multi-key values() is not a single scalar — bail.
            if self.peek() == Some(&Tok::Comma) || self.expect(&Tok::RParen).is_err() {
                self.pos = save;
                return Ok(None);
            }
            if self.peek() != Some(&Tok::Dot) {
                self.pos = save;
                return Ok(None);
            }
            self.bump();
            val_key = Some(k);
        }
        let reducer = match self.peek() {
            Some(Tok::Ident(s)) => s.to_ascii_lowercase(),
            _ => {
                self.pos = save;
                return Ok(None);
            }
        };
        if !matches!(
            reducer.as_str(),
            "count" | "fold" | "sum" | "min" | "max" | "mean"
        ) {
            self.pos = save;
            return Ok(None);
        }
        self.bump();
        // Only the nullary reducer form is a per-element scalar (`fold(seed, fn)` etc.
        // are not) — require exactly `()` then an arm terminator.
        if self.expect(&Tok::LParen).is_err() || self.peek() != Some(&Tok::RParen) {
            self.pos = save;
            return Ok(None);
        }
        self.bump();
        if !matches!(self.peek(), Some(&Tok::Comma) | Some(&Tok::RParen)) {
            self.pos = save;
            return Ok(None);
        }
        // A leading barrier emptied the self-row, so the reducer runs over NOTHING per element:
        // count → 0, fold → [], sum/min/max/mean → NULL (Gremlin's empty-aggregate).
        if bare_empty {
            return Ok(Some(match reducer.as_str() {
                "count" => Expr::Lit(Value::Num(0.0)),
                "fold" => Expr::Lit(Value::List(Vec::new())),
                _ => Expr::Lit(Value::Null),
            }));
        }
        let expr = match reducer.as_str() {
            "count" => {
                if let Some(k) = &val_key {
                    body = body.filter(Expr::PropertyExists {
                        slot: landed,
                        key: k.clone(),
                    });
                }
                Expr::CountSubquery {
                    body: Box::new(body),
                    outer_width: width,
                }
            }
            "fold" => {
                let scalar = match val_key {
                    Some(k) => {
                        body = body.filter(Expr::PropertyExists {
                            slot: landed,
                            key: k.clone(),
                        });
                        Expr::Prop {
                            slot: landed,
                            key: k,
                        }
                    }
                    None => Expr::Slot(landed),
                };
                Expr::CollectSubquery {
                    body: Box::new(body),
                    scalar: Box::new(scalar),
                    outer_width: width,
                }
            }
            other => {
                // sum/min/max/mean need a numeric scalar — a `values('k')` to reduce.
                // Without one (reducing over a node/edge) there is no per-element scalar;
                // let it fall through to the whole-stream body.
                let Some(k) = val_key else {
                    self.pos = save;
                    return Ok(None);
                };
                let func = match other {
                    "sum" => AggFn::Sum,
                    "min" => AggFn::Min,
                    "max" => AggFn::Max,
                    _ => AggFn::Avg,
                };
                body = body.filter(Expr::PropertyExists {
                    slot: landed,
                    key: k.clone(),
                });
                Expr::AggSubquery {
                    body: Box::new(body),
                    scalar: Box::new(Expr::Prop {
                        slot: landed,
                        key: k,
                    }),
                    func,
                    outer_width: width,
                }
            }
        };
        Ok(Some(expr))
    }

    /// True when the coalesce/union body ahead starts with an EDGE hop
    /// (`[__.](outE|inE|bothE)`), so the reconverged frontier holds edges.
    fn peek_leading_is_edge(&self) -> bool {
        let mut p = self.pos;
        if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
            p += 1;
            if self.toks.get(p) == Some(&Tok::Dot) {
                p += 1;
            }
        }
        matches!(self.toks.get(p), Some(Tok::Ident(s)) if {
            let l = s.to_ascii_lowercase();
            l == "oute" || l == "ine" || l == "bothe"
        })
    }

    /// Parse a comma-separated list of child filter traversals up to (but not
    /// consuming) the enclosing `)`.
    fn child_filter_list(&mut self) -> Result<Vec<Expr>, String> {
        // An empty list — `and()` / `or()` — is allowed (the caller supplies the
        // identity: `or()` matches nothing, `and()` matches everything).
        if self.peek() == Some(&Tok::RParen) {
            return Ok(Vec::new());
        }
        let mut parts = vec![self.child_filter_expr()?];
        while self.peek() == Some(&Tok::Comma) {
            self.bump();
            parts.push(self.child_filter_expr()?);
        }
        Ok(parts)
    }

    /// Parse a Gremlin predicate argument and build the full comparison `Expr`
    /// against `left`. Accepts an optional `P.` prefix, then one of:
    /// `op(literal)` (`gt`, `neq`, …), `within(a, b, …)` / `without(a, b, …)`
    /// (membership, an OR-of-equals and its negation), or a bare literal
    /// (equality). Shared by `has(...)` and `where(...)`.
    fn predicate_expr(&mut self, left: Expr) -> Result<Expr, String> {
        // Strip a `P.` / `TextP.` namespace prefix (`has('n', TextP.regex('^r'))`).
        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("P") || s.eq_ignore_ascii_case("TextP"))
        {
            self.pos += 1;
            self.expect(&Tok::Dot)?;
        }
        // An identifier immediately applied to `(` is an operator; anything else is
        // a bare literal compared for equality. A temporal constructor (`date(…)`,
        // `datetime(…)`, …) reads as a LITERAL, not an operator — `has('vf',
        // date('…'))` is an equality, and `lte(date('…'))` nests it as the bound.
        let is_temporal_ctor = matches!(self.peek(), Some(Tok::Ident(s)) if matches!(
            s.to_ascii_lowercase().as_str(),
            "date" | "datetime" | "time" | "duration" | "zoned_time" | "zoned_datetime"
        ));
        if !is_temporal_ctor
            && matches!(self.peek(), Some(Tok::Ident(_)))
            && self.toks.get(self.pos + 1) == Some(&Tok::LParen)
        {
            let op_name = self.ident()?.to_ascii_lowercase();
            self.expect(&Tok::LParen)?;
            // within/without take a value LIST and desugar to an OR-of-equals.
            if op_name == "within" || op_name == "without" {
                let vals = self.literal_list()?;
                self.expect(&Tok::RParen)?;
                let member = or_of_equals(&left, &vals);
                return Ok(if op_name == "without" {
                    Expr::Not(Box::new(member))
                } else {
                    member
                });
            }
            // Range predicates take TWO bounds. `between(lo,hi)` is lo-inclusive,
            // hi-EXCLUSIVE (TinkerPop); `inside(lo,hi)` is exclusive both ends;
            // `outside(lo,hi)` is the complement (`< lo OR > hi`).
            if matches!(op_name.as_str(), "between" | "inside" | "outside") {
                let lo = self.literal()?;
                self.expect(&Tok::Comma)?;
                let hi = self.literal()?;
                self.expect(&Tok::RParen)?;
                let cmp = |op: CompareOp, v: &Value| Expr::Compare {
                    op,
                    left: Box::new(left.clone()),
                    right: Box::new(Expr::Lit(v.clone())),
                };
                return Ok(match op_name.as_str() {
                    "between" => Expr::And(
                        Box::new(cmp(CompareOp::Ge, &lo)),
                        Box::new(cmp(CompareOp::Lt, &hi)),
                    ),
                    "inside" => Expr::And(
                        Box::new(cmp(CompareOp::Gt, &lo)),
                        Box::new(cmp(CompareOp::Lt, &hi)),
                    ),
                    _ => Expr::Or(
                        Box::new(cmp(CompareOp::Lt, &lo)),
                        Box::new(cmp(CompareOp::Gt, &hi)),
                    ),
                });
            }
            // Text predicates (`TextP`): a single string bound, desugaring to the
            // same `starts_with`/`ends_with`/`contains` scalar the GQL infix forms use.
            if let Some(fname) = match op_name.as_str() {
                "startingwith" | "startswith" => Some("starts_with"),
                "endingwith" | "endswith" => Some("ends_with"),
                "containing" => Some("contains"),
                "notstartingwith" | "notendingwith" | "notcontaining" => None,
                _ => Some(""),
            }
            .filter(|f| !f.is_empty())
            {
                let val = self.literal()?;
                self.expect(&Tok::RParen)?;
                return Ok(Expr::Call {
                    name: fname.to_string(),
                    args: vec![left, Expr::Lit(val)],
                });
            }
            // `regex(pattern)` / `TextP.regex(pattern)`: a `regex_match` call. The
            // pattern is validated at PARSE time (like the TS engine), so a bad pattern is a
            // parse error rather than a silent no-match at runtime.
            if op_name == "regex" {
                let val = self.literal()?;
                self.expect(&Tok::RParen)?;
                if let Value::Str(p) = &val {
                    regex::Regex::new(p).map_err(|e| format!("regex: invalid pattern: {e}"))?;
                } else {
                    return Err("regex(pattern): pattern must be a string".into());
                }
                return Ok(Expr::Call {
                    name: "regex_match".to_string(),
                    args: vec![left, Expr::Lit(val)],
                });
            }
            // Negated text predicates: NOT of the positive form.
            if matches!(
                op_name.as_str(),
                "notstartingwith" | "notendingwith" | "notcontaining"
            ) {
                let fname = match op_name.as_str() {
                    "notstartingwith" => "starts_with",
                    "notendingwith" => "ends_with",
                    _ => "contains",
                };
                let val = self.literal()?;
                self.expect(&Tok::RParen)?;
                return Ok(Expr::Not(Box::new(Expr::Call {
                    name: fname.to_string(),
                    args: vec![left, Expr::Lit(val)],
                })));
            }
            let val = self.literal()?;
            self.expect(&Tok::RParen)?;
            let op = match op_name.as_str() {
                "eq" => CompareOp::Eq,
                "neq" => CompareOp::Ne,
                "gt" => CompareOp::Gt,
                "gte" => CompareOp::Ge,
                "lt" => CompareOp::Lt,
                "lte" => CompareOp::Le,
                other => return Err(format!("unsupported predicate `{other}`")),
            };
            Ok(Expr::Compare {
                op,
                left: Box::new(left),
                right: Box::new(Expr::Lit(val)),
            })
        } else {
            Ok(Expr::Compare {
                op: CompareOp::Eq,
                left: Box::new(left),
                right: Box::new(Expr::Lit(self.literal()?)),
            })
        }
    }

    /// A comma-separated list of literals (the arguments of `within`/`without`).
    fn literal_list(&mut self) -> Result<Vec<Value>, String> {
        let mut vals = Vec::new();
        if self.peek() != Some(&Tok::RParen) {
            loop {
                vals.push(self.literal()?);
                if self.peek() != Some(&Tok::Comma) {
                    break;
                }
                self.pos += 1;
            }
        }
        Ok(vals)
    }

    fn literal(&mut self) -> Result<Value, String> {
        // Temporal constructors — `date('…')`, `datetime('…')`, `time('…')`,
        // `duration('…')`, `zoned_time`/`zoned_datetime`. The text dialect spells
        // local time `time(…)` where the shared tag is `localtime`; every kind goes
        // through the one `Temporal::parse` the codecs and GQL parser use.
        if let Some(Tok::Ident(name)) = self.peek() {
            let ctor = name.to_ascii_lowercase();
            if matches!(
                ctor.as_str(),
                "date" | "datetime" | "time" | "duration" | "zoned_time" | "zoned_datetime"
            ) && self.toks.get(self.pos + 1) == Some(&Tok::LParen)
            {
                self.bump(); // ctor name
                self.expect(&Tok::LParen)?;
                let lit = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                let tag = if ctor == "time" { "localtime" } else { &ctor };
                let t = crate::temporal::Temporal::parse(tag, &lit)?;
                return Ok(Value::Temporal(t));
            }
        }
        // A list literal `[a, b, …]` (nestable).
        if self.peek() == Some(&Tok::LBracket) {
            self.bump();
            let mut items = Vec::new();
            if self.peek() != Some(&Tok::RBracket) {
                loop {
                    items.push(self.literal()?);
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
            }
            self.expect(&Tok::RBracket)?;
            return Ok(Value::List(items));
        }
        match self.bump() {
            Some(Tok::Str(s)) => Ok(Value::Str(s.into())),
            Some(Tok::Num(n)) => Ok(Value::Num(n)),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("true") => Ok(Value::Bool(true)),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("false") => Ok(Value::Bool(false)),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("null") => Ok(Value::Null),
            other => Err(format!("expected a literal, got {other:?}")),
        }
    }

    fn usize_arg(&mut self) -> Result<usize, String> {
        match self.bump() {
            Some(Tok::Num(n)) if n >= 0.0 && n.fract() == 0.0 => Ok(n as usize),
            other => Err(format!("expected a non-negative integer, got {other:?}")),
        }
    }

    /// Parse a `repeat(...)` body — v1 accepts only a SINGLE anonymous hop:
    /// `out('L')` / `in('L')` / `both('L')` (argless = any type), with an optional
    /// leading `__.` (the TinkerPop anonymous-traversal spawn). Returns the hop's
    /// `(direction, edge label)`. Multi-step bodies, nested repeats, filters, etc.
    /// are deferred with an explicit error. The cursor starts just after `repeat(`
    /// and stops at the body's closing `)` (left for the caller to consume).
    /// Parse a `[__.]loops().is([P.]op(n))` predicate, returning `(op, n)`; `None` (cursor
    /// restored) if the next tokens are not that shape. Used by `emit`/`until` to bound
    /// the repeat DEPTH by the loop counter.
    fn try_loops_predicate(&mut self) -> Result<Option<(CompareOp, u32)>, String> {
        let save = self.pos;
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            if self.peek() == Some(&Tok::Dot) {
                self.bump();
            } else {
                self.pos = save;
                return Ok(None);
            }
        }
        if !matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("loops")) {
            self.pos = save;
            return Ok(None);
        }
        // Consume `loops().is([P.]op(n))`.
        self.ident()?; // loops
        self.expect(&Tok::LParen)?;
        self.expect(&Tok::RParen)?;
        self.expect(&Tok::Dot)?;
        let isn = self.ident()?;
        if !isn.eq_ignore_ascii_case("is") {
            self.pos = save;
            return Ok(None);
        }
        self.expect(&Tok::LParen)?;
        let (op, val) = self.simple_predicate()?;
        self.expect(&Tok::RParen)?;
        let n = match val {
            Value::Num(x) if x >= 0.0 => x as u32,
            _ => return Err("loops().is(op(n)): n must be a non-negative integer".into()),
        };
        Ok(Some((op, n)))
    }

    #[allow(clippy::type_complexity)]
    fn repeat_body(
        &mut self,
    ) -> Result<
        (
            Dir,
            Option<String>,
            Option<String>,
            Option<Expr>,
            Option<u32>,
        ),
        String,
    > {
        // Optional `__.` anonymous-traversal prefix.
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        let hop = self.ident()?;
        let dir = match hop.to_ascii_lowercase().as_str() {
            "out" => Dir::Out,
            "in" => Dir::In,
            "both" => Dir::Both,
            other => {
                return Err(format!(
                    "repeat() body must be a single out/in/both hop, got `{other}`"
                ))
            }
        };
        self.expect(&Tok::LParen)?;
        let mut labels: Vec<String> = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            loop {
                labels.push(self.str_arg()?);
                if self.peek() == Some(&Tok::Comma) {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        self.expect(&Tok::RParen)?;
        let label = match labels.len() {
            0 => None,
            1 => Some(labels.remove(0)),
            _ => {
                return Err(
                    "repeat() body hop with multiple edge labels is not yet supported".into(),
                )
            }
        };
        // An optional inner `.as('tag')` binds the tag to the walk's endpoint (the last
        // iteration's landing) — `repeat(out('CREATED').as('a')).times(1).select('a')`.
        let mut tag = None;
        if self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("as"))
        {
            self.expect(&Tok::Dot)?;
            self.ident()?; // as
            self.expect(&Tok::LParen)?;
            tag = Some(self.str_arg()?);
            self.expect(&Tok::RParen)?;
        }
        // An optional trailing element filter on the hop TARGET —
        // `repeat(out().hasLabel('PERSON'))` / `repeat(out().has('k', v))`. It becomes a
        // per-hop body filter: a target failing it is pruned from the walk. (`self.current`
        // is still the source here, but the body filter is evaluated over a one-row
        // mini-batch whose every slot carries the landed node, so the slot is immaterial.)
        let mut body_filter = None;
        let mut max_cap = None;
        if self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if {
                matches!(s.to_ascii_lowercase().as_str(), "haslabel" | "has" | "hasnot" | "haskey")
            })
        {
            self.expect(&Tok::Dot)?;
            // child_filter_expr consumes the filter's own parens; the repeat arm closes
            // the outer `repeat(...)` paren after this returns.
            body_filter = Some(self.child_filter_expr()?);
        } else if self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("where"))
        {
            // `where(loops().is(<op>(n)))` — an in-body depth guard. `loops()` == depth+1,
            // so the walk continues while the guard holds: `lt(n)` caps depth at n-1,
            // `le(n)`/`lte(n)` at n. (A non-loops body `where(...)` is not yet supported.)
            self.expect(&Tok::Dot)?;
            self.ident()?; // where
            self.expect(&Tok::LParen)?;
            let Some((op, n)) = self.try_loops_predicate()? else {
                return Err("repeat(<hop>.where(...)) supports only where(loops().is(...))".into());
            };
            self.expect(&Tok::RParen)?; // close where(...)
            max_cap = Some(match op {
                CompareOp::Lt => n.saturating_sub(1),
                CompareOp::Le => n,
                other => {
                    return Err(format!(
                        "repeat(<hop>.where(loops().is({other:?}(n)))) unsupported op"
                    ))
                }
            });
        }
        Ok((dir, label, tag, body_filter, max_cap))
    }

    /// Close an open `repeat(...)` into a `VarLength` walk. `times(n)` alone is a
    /// fixed-length walk (min = max = n); an `emit`/`until` modulator emits at every
    /// depth (min = 1) up to `n` or the default iteration cap, optionally filtering
    /// the emitted endpoint. Walk mode (Gremlin allows revisiting edges).
    fn flush_repeat(&mut self, plan: Plan) -> Result<Plan, String> {
        let Some(ctx) = self.pending_repeat.take() else {
            return Ok(plan);
        };
        const CAP: u32 = 100; // the TS engine's default iteration cap
                              // A `repeat(identity())` walk never moves the frontier, so the ONLY thing the
                              // modulators decide is whether any depth is emittable. A post-form `until` is a
                              // do-while (min 1); with `times(0)` (max 0) min > max, so nothing survives —
                              // matching the TS engine. Otherwise the frontier passes through.
        if ctx.identity_body {
            let (min, max) = if ctx.until.is_some() {
                (1, ctx.times.unwrap_or(CAP))
            } else {
                match (ctx.min_one, ctx.times) {
                    (false, Some(n)) => (n, n),
                    (true, Some(n)) => (1, n),
                    (true, None) => (1, CAP),
                    (false, None) => (0, 0),
                }
            };
            return Ok(if min > max {
                plan.filter(Expr::Lit(Value::Bool(false)))
            } else {
                plan
            });
        }
        // A fixed-`times` walk carrying an inner `.as('tag')` UNROLLS into N explicit
        // hops, so the tag binds once PER ITERATION (a distinct slot each), which is what
        // select(Pop.first/all, 'tag') reads (first binding / every binding as a list).
        // A single VarLength endpoint would bind the tag ONCE. Only the plain fixed walk
        // unrolls — a variable emit/until/body-filter/loops-cap walk keeps the VarLength
        // and binds just the endpoint.
        if let (Some(tag), false, Some(n)) = (ctx.bind_tag.as_ref(), ctx.min_one, ctx.times) {
            if n >= 1 && ctx.until.is_none() && ctx.body_filter.is_none() && ctx.max_cap.is_none() {
                let etypes = etypes_of(ctx.label.as_deref());
                let tag = tag.clone();
                self.slots = ctx.out_slot; // reclaim the pre-allocated endpoint slot
                let mut p = plan;
                let mut cur = ctx.from;
                for _ in 0..n {
                    let landed = self.slots;
                    p = if matches!(ctx.dir, Dir::Both) {
                        p.expand_both_gremlin(cur, &etypes)
                    } else {
                        p.expand(cur, ctx.dir, &etypes)
                    };
                    self.slots += 1;
                    self.first_labels.entry(tag.clone()).or_insert(landed);
                    self.all_labels.entry(tag.clone()).or_default().push(landed);
                    self.labels.insert(tag.clone(), landed);
                    cur = landed;
                }
                self.current = cur;
                self.slots = cur + 1;
                self.path_ok = ctx.path_ok_at_open;
                return Ok(p);
            }
        }
        // An `until(pred)` walk runs to the cap (or `times`), emitting only on a match:
        // pre-form is while-do (min 0 — a source may satisfy `pred`), post-form do-while
        // (min 1). Otherwise `times`/`emit` decide the bounds as before.
        let (min, max) = if ctx.until.is_some() {
            let lo = u32::from(!ctx.until_pre);
            (lo, ctx.times.unwrap_or(CAP))
        } else {
            match (ctx.min_one, ctx.times) {
                (false, Some(n)) => (n, n),
                (true, Some(n)) => (1, n),
                (true, None) => (1, CAP),
                (false, None) => {
                    return Err(
                        "repeat(<hop>) must be closed by times(n), emit() or until(pred)".into(),
                    )
                }
            }
        };
        // A `loops()` emit predicate raises the minimum emitted depth.
        let min = ctx.min_override.map_or(min, |m| m.max(min));
        // An in-body `where(loops())` depth guard caps the max (never above the bound).
        let max = ctx.max_cap.map_or(max, |c| c.min(max));
        let p = plan.var_length_until(
            ctx.from,
            ctx.dir,
            &etypes_of(ctx.label.as_deref()),
            min,
            max,
            PathMode::Walk,
            ctx.until.map(Box::new),
            ctx.body_filter.map(Box::new),
            matches!(ctx.dir, Dir::Both),
        );
        // The walk appended its endpoint at `out_slot` (the width before this call);
        // account for it and land the current element there.
        self.current = ctx.out_slot;
        self.slots = ctx.out_slot + 1;
        // A pure vertex-hop walk records full path lineage — restore path-answerability
        // to what it was before the repeat modulators tainted it.
        self.path_ok = ctx.path_ok_at_open;
        // An inner `.as('tag')` binds the tag to the endpoint (the last landing).
        if let Some(tag) = ctx.bind_tag {
            self.first_labels.entry(tag.clone()).or_insert(ctx.out_slot);
            self.all_labels
                .entry(tag.clone())
                .or_default()
                .push(ctx.out_slot);
            self.labels.insert(tag, ctx.out_slot);
        }
        Ok(match ctx.filter {
            Some(f) => p.filter(f),
            None => p,
        })
    }

    /// Consume an optional `Scope` inside `order(...)`: bare `local`/`global`, or
    /// `Scope.local`/`Scope.global`. Returns true for local; no arg defaults to
    /// global. Leaves the cursor at the closing `)`.
    fn parse_scope_is_local(&mut self) -> Result<bool, String> {
        match self.peek() {
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("local") => {
                self.pos += 1;
                Ok(true)
            }
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("global") => {
                self.pos += 1;
                Ok(false)
            }
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("Scope") => {
                self.pos += 1;
                self.expect(&Tok::Dot)?;
                Ok(self.ident()?.eq_ignore_ascii_case("local"))
            }
            _ => Ok(false),
        }
    }

    /// `asc`/`desc` — a bare ident, a quoted string, or `Order.desc`.
    fn order_dir(&mut self) -> Result<bool, String> {
        let word = match self.bump() {
            Some(Tok::Ident(s)) => {
                // allow `Order.desc` / `Order.asc`
                if s.eq_ignore_ascii_case("Order") {
                    self.expect(&Tok::Dot)?;
                    self.ident()?
                } else {
                    s
                }
            }
            Some(Tok::Str(s)) => s,
            other => return Err(format!("expected asc/desc, got {other:?}")),
        };
        match word.to_ascii_lowercase().as_str() {
            "desc" => Ok(true),
            "asc" => Ok(false),
            other => Err(format!("expected asc/desc, got `{other}`")),
        }
    }
}

#[cfg(test)]
mod tests;
