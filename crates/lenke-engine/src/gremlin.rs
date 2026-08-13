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

pub fn parse(query: &str) -> Result<Plan, String> {
    let toks = lex(query)?;
    let mut p = Parser {
        toks,
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
        "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "sinh" | "cosh" | "tanh" | "sqrt"
            | "abs" | "ceil" | "floor" | "exp" | "ln" | "log10" | "signum"
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
                op: if op == '+' { ArithOp::Add } else { ArithOp::Sub },
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
                // dropping the backslash — byte-identical to core's lexer.
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
    /// was filled, so `cap(key)` folds exactly that stream (matching core, where the
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
            other => return Err(format!("order().by(<traversal>): unsupported body `{other}`")),
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
    /// correlated `CountSubquery`, the only nested body core's projections use here.
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
            let func = if acc == "id" { "element_id" } else { "element_label" };
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
                    Plan::EdgeSeed { ext_ids }
                } else {
                    self.expect(&Tok::RParen)?;
                    // An edge source makes the path start on an edge, not a node.
                    self.on_edge = true;
                    Plan::EdgeScan
                }
            }
            "adde" => {
                // g.addE('T').from(V(a)).to(V(b)).property(...)
                self.expect(&Tok::LParen)?;
                let etype = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                self.finish_add_edge(None, None, etype)?
            }
            "addv" => {
                // addV('Label') creates one vertex; following property() steps
                // fold into it (see `apply_property`).
                self.expect(&Tok::LParen)?;
                let label = self.str_arg()?;
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
        while self.peek() == Some(&Tok::Dot) {
            self.pos += 1;
            plan = self.step(plan)?;
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
        if self.current != 0 {
            plan = plan.project(vec![("_".to_string(), Expr::Slot(self.current))]);
            self.current = 0;
            self.slots = 1;
        }
        Ok(plan)
    }

    fn step(&mut self, plan: Plan) -> Result<Plan, String> {
        let name = self.ident()?;
        self.expect(&Tok::LParen)?;
        let lname = name.to_ascii_lowercase();
        // An edge hop's landed endpoint is only reachable by the vertex move that
        // IMMEDIATELY follows it; consume the record here so any other step clears it.
        let prev_edge_hop = self.edge_hop.take();
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
            self.expect(&Tok::Comma)?;
            let val = self.literal()?;
            self.expect(&Tok::RParen)?;
            return Ok(self.apply_property(plan, key, val));
        }
        if lname == "drop" {
            self.expect(&Tok::RParen)?;
            if is_write(&plan) {
                return Err("drop() cannot follow a write step".into());
            }
            // Delete the current elements of the traversal.
            return Ok(Plan::Update {
                input: Box::new(plan),
                // Gremlin drop() removes the element AND its incident edges.
                ops: vec![crate::ir::SetOp::Delete {
                    slot: self.current,
                    detach: true,
                }],
            });
        }
        // A read step cannot follow a write step (addV/property/drop are terminal
        // for reads).
        if is_write(&plan) {
            return Err(format!("step `{lname}` cannot follow a write step"));
        }

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
                let mut bodies = Vec::new();
                let mut land;
                loop {
                    // A `<hop>.count()` branch is a PER-ELEMENT count (each input keeps
                    // its own degree), not a global fold — lower it to a Row-projected
                    // CountSubquery. Any other body parses as a Row-rooted sub-plan.
                    if let Some(body) = self.try_count_body(from, width)? {
                        bodies.push(body);
                        land = (0, 1);
                    } else {
                        let (body, oc, os) = self.parse_sub_body(from, width)?;
                        bodies.push(body);
                        land = (oc, os);
                    }
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                // Each body lands its result at the same slot (value bodies project to
                // slot 0; single-hop bodies land the neighbour at the input width) —
                // the concatenated frontier continues from there.
                self.current = land.0;
                self.slots = land.1;
                plan.branch(bodies)
            }
            "optional" => {
                // optional(<hop>): advance to the hop's neighbour(s) if any, else keep
                // the element unchanged. This is OptionalExpand with keep_source — a
                // missed row lands the SOURCE element (not null), so the frontier
                // continues either way. v1 is a single hop.
                let from = self.current;
                let (dir, label) = self.hop_body()?;
                self.expect(&Tok::RParen)?;
                self.current = self.slots;
                self.slots += 1;
                plan.optional_expand(from, dir, &etypes_of(label.as_deref()), true, false)
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
                if let Some(keys) = self.try_all_values_bodies() {
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
                    return Ok(p);
                }
                // General: each body is a sub-traversal reconverging to one output slot.
                // A body CONTRIBUTES when its leading hop exists (`out('CREATED')…`), or
                // ALWAYS when it has no leading hop (`constant(v)`, a bare value). Body k
                // fires only where every earlier body did NOT — an exclusion guard on the
                // seed `Row`, then the body's own steps (a hop chain, a projection, …).
                let slots = self.slots;
                let mut bodies = Vec::new();
                let mut prior: Option<Expr> = None; // OR of the earlier branches' existence
                let mut land;
                let mut any_edge = false; // an edge-hop body → coalesce yields an edge frontier
                loop {
                    if self.peek_leading_is_edge() {
                        any_edge = true;
                    }
                    let this = match self.peek_leading_hop() {
                        Some((dir, labels)) => Expr::Exists {
                            body: Box::new(Plan::Row.expand(from, dir, &labels)),
                            outer_width: slots,
                        },
                        // A body led by a FILTER (`hasLabel('PERSON').values('name')`)
                        // produces output only where the filter holds; that predicate is
                        // the guard. Peek it via child_filter_expr, then restore the
                        // cursor so parse_sub_body re-parses the full body.
                        None if self.peek_leading_is_filter() => {
                            let save = self.pos;
                            let pred = self.child_filter_expr()?;
                            self.pos = save;
                            pred
                        }
                        None => Expr::Lit(Value::Bool(true)), // a value body always produces
                    };
                    let guard = match &prior {
                        None => this.clone(),
                        Some(p) => Expr::And(
                            Box::new(Expr::Not(Box::new(p.clone()))),
                            Box::new(this.clone()),
                        ),
                    };
                    let (body, oc, os) =
                        self.parse_sub_body_seeded(Plan::Row.filter(guard), from, slots)?;
                    bodies.push(body);
                    land = (oc, os);
                    prior = Some(match prior {
                        None => this,
                        Some(p) => Expr::Or(Box::new(p), Box::new(this)),
                    });
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                self.current = land.0;
                self.slots = land.1;
                self.on_edge = any_edge;
                // An edge-yielding coalesce keeps the interleaved edge-path answerable
                // (its edge lands in the lineage); a non-edge coalesce breaks it.
                if any_edge {
                    self.path_has_edges = true;
                } else {
                    self.edge_path_ok = false;
                }
                plan.branch(bodies)
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
                // choose(<pred>, <then>, <else>): route each element by a filter
                // predicate. When both arms are single-VALUE bodies (`values('k')`,
                // `constant(v)`, `id()`, `label()`) it is a per-row Case projection;
                // otherwise both arms are hops reconverging like union.
                let from = self.current;
                let pred = self.child_filter_expr()?;
                self.expect(&Tok::Comma)?;
                // A `drop()` then-arm (no else): delete the pred-matching elements —
                // a terminal write, so nothing reconverges. `choose(identity(), drop())`
                // deletes everything the predicate keeps.
                if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("drop"))
                    && self.toks.get(self.pos + 1) == Some(&Tok::LParen)
                {
                    self.bump(); // drop
                    self.expect(&Tok::LParen)?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::RParen)?; // close choose(...)
                    return Ok(Plan::Update {
                        input: Box::new(plan.filter(pred)),
                        ops: vec![crate::ir::SetOp::Delete {
                            slot: from,
                            detach: true,
                        }],
                    });
                }
                if let Some(then_val) = self.parse_single_value_body(from)? {
                    self.expect(&Tok::Comma)?;
                    let else_val = self.parse_single_value_body(from)?.ok_or(
                        "choose(): mixing a value arm with a non-value arm is not supported",
                    )?;
                    self.expect(&Tok::RParen)?;
                    let p = plan.project(vec![(
                        "choose".to_string(),
                        Expr::Case {
                            branches: vec![(pred, then_val)],
                            otherwise: Some(Box::new(else_val)),
                        },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    return Ok(p);
                }
                // The THEN arm is a hop; its neighbour lands at the reconverge slot W.
                let (t_dir, t_label) = self.hop_body()?;
                let land = self.slots;
                let then_body =
                    Plan::Row
                        .filter(pred.clone())
                        .expand(from, t_dir, &etypes_of(t_label.as_deref()));
                // The ELSE arm is a hop, `identity()`, or absent (implicit identity). An
                // identity/absent arm passes the element through — copy it into slot W so
                // both arms reconverge there.
                let else_is_hop = self.peek() == Some(&Tok::Comma) && {
                    // The else arm starts after the comma and an optional `__.`.
                    let mut p = self.pos + 1;
                    if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
                        p += 1;
                        if self.toks.get(p) == Some(&Tok::Dot) {
                            p += 1;
                        }
                    }
                    matches!(self.toks.get(p), Some(Tok::Ident(s)) if {
                        let l = s.to_ascii_lowercase();
                        l == "out" || l == "in" || l == "both"
                    })
                };
                let else_body = if else_is_hop {
                    self.expect(&Tok::Comma)?;
                    let (e_dir, e_label) = self.hop_body()?;
                    Plan::Row.filter(Expr::Not(Box::new(pred))).expand(
                        from,
                        e_dir,
                        &etypes_of(e_label.as_deref()),
                    )
                } else {
                    // `, identity()` or nothing → pass the element through at slot W.
                    if self.peek() == Some(&Tok::Comma) {
                        self.bump();
                        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
                            self.bump();
                            self.expect(&Tok::Dot)?;
                        }
                        let id = self.ident()?; // identity
                        if !id.eq_ignore_ascii_case("identity") {
                            return Err(format!("choose(): unsupported else arm `{id}`"));
                        }
                        self.expect(&Tok::LParen)?;
                        self.expect(&Tok::RParen)?;
                    }
                    Plan::Row.filter(Expr::Not(Box::new(pred))).map_slot(
                        land,
                        Expr::Slot(from),
                        true,
                    )
                };
                self.expect(&Tok::RParen)?;
                self.current = land;
                self.slots = land + 1;
                plan.branch(vec![then_body, else_body])
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
                    // read the endpoint straight off the edge at the current slot. Only
                    // valid when the frontier actually holds edges.
                    _ if self.on_edge => {
                        self.on_edge = false;
                        let which = match lname.as_str() {
                            "outv" => Dir::Out,
                            "inv" => Dir::In,
                            "bothv" => Dir::Both,
                            _ => {
                                return Err(
                                    "otherV() off a bare edge frontier is not supported".into(),
                                )
                            }
                        };
                        let edge_slot = self.current;
                        let landed = self.slots;
                        self.current = landed;
                        self.slots += 1;
                        Plan::EdgeVertex {
                            input: Box::new(plan),
                            edge_slot,
                            which,
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
                // PRESENT on the element — an ABSENT property yields nothing (core
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
                // An interleaved node/edge path (`outE().inV()…`) renders through
                // GremlinPath (lineage `values` zipped with `edges`); a bare vertex-hop
                // chain stays on the nodes-only path below.
                if self.path_has_edges && self.edge_path_ok {
                    let ends_on_edge = self.on_edge;
                    let mut bys: Vec<crate::ir::GPathBy> = Vec::new();
                    while self.peek() == Some(&Tok::Dot)
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                    {
                        self.expect(&Tok::Dot)?;
                        self.ident()?; // by
                        self.expect(&Tok::LParen)?;
                        // by('k') | by(id) | by(label) | by(T.id|T.label) | by(__.id()|__.label()).
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
                    let p = plan.project(vec![(
                        "path".to_string(),
                        Expr::GremlinPath { ends_on_edge, bys },
                    )]);
                    self.current = 0;
                    self.slots = 1;
                    return Ok(p);
                }
                if !self.path_ok {
                    return Err("path() is only supported over a pure vertex-hop chain \
                                (V-source + out/in/both); edge steps, var-length, value \
                                projections and the E source are deferred"
                        .into());
                }
                // Gremlin path() over a vertex-hop chain is the sequence of vertices
                // visited. The engine's lineage records exactly that node sequence
                // (`Expr::Path` → the ids); `path_nodes` renders each id as its
                // element map so the path elements are vertices, matching core. The
                // `Expr::Path` argument both feeds the ids and makes `needs_lineage`
                // switch tracking on.
                // An optional `.by('k')` projects each path ELEMENT to a property
                // (Gremlin `path().by('name')` → a list of names, not vertex maps).
                let call = if self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // `by`
                    self.expect(&Tok::LParen)?;
                    // by('k') → each element's property; by(id|label|T.id|T.label) → the
                    // element's ext-id / label (sentinel keys the path_values fn maps).
                    if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("T")) {
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
                p
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
                // elementMap() → core's FLAT element map {id, label, <props…>} (plus
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
                p
            }
            "valuemap" | "propertymap" => {
                // valueMap() → a PROPERTIES-only map (no id/label tokens) with scalar
                // values; valueMap('k1',…) filters keys. propertyMap() is the same but
                // each value is wrapped in a single-element list. Both lower to the
                // gremlin-only `value_map`/`property_map` exec fn (element slot + keys).
                let fn_name = if lname == "propertymap" {
                    "property_map"
                } else {
                    "value_map"
                };
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
                    fn_name.to_string(),
                    Expr::Call {
                        name: fn_name.to_string(),
                        args: fn_args,
                    },
                )]);
                self.current = 0;
                self.slots = 1;
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
                        let key = keys.first().cloned().ok_or(
                            "value() after a multi-key properties() is not yet supported",
                        )?;
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
                // slot-vs-slot comparison (core's WhereKey; the predicate's rhs is a
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
                    // compare the current value to that TAG's value (core's tagged
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
                            frac: None, null_on_empty: false,
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
                        frac: None, null_on_empty: false,
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
                let mut p = plan;
                while self.peek() == Some(&Tok::Dot)
                    && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                {
                    self.expect(&Tok::Dot)?;
                    self.ident()?; // by
                    self.expect(&Tok::LParen)?;
                    let (_, e) = self.by_key_expr(self.current)?;
                    self.expect(&Tok::RParen)?;
                    let w = self.slots;
                    p = p.map_slot(w, e, true);
                    self.slots += 1;
                    by_slots.push(w);
                }
                if !by_slots.is_empty() {
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
                        } => Plan::OrderPage {
                            input,
                            keys,
                            skip: None,
                            limit: Some(n),
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
                            args: vec![
                                Expr::Slot(self.current),
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
                            nulls_first: true, // Gremlin: NULLs first
                        });
                    }
                    if keys.is_empty() {
                        keys.push(SortKey {
                            expr: Expr::Slot(self.current),
                            descending: false,
                            nulls_first: true,
                        });
                    }
                    plan.order_page(keys, None, None)
                }
            }
            "as" => {
                // Label the current slot; the plan is unchanged (select resolves it).
                let label = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                self.first_labels.entry(label.clone()).or_insert(self.current);
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
                // property; they CYCLE across the labels (core's `bys[i % bys.len()]`).
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
                // A select over an UNBOUND tag drops every traverser (core yields
                // nothing), rather than erroring — the tag is a compile-time slot, so an
                // unknown one produces no output.
                let any_unbound = labels.iter().any(|l| {
                    if pop_all {
                        !self.all_labels.contains_key(l)
                    } else {
                        !self.labels.contains_key(l)
                    }
                });
                if any_unbound {
                    self.current = 0;
                    self.slots = 1;
                    return Ok(plan.filter(Expr::Lit(Value::Bool(false))));
                }
                let p = if pop_all {
                    // Pop.all: every binding of the (single) tag, as a list.
                    let slots = self
                        .all_labels
                        .get(&labels[0])
                        .cloned()
                        .ok_or_else(|| format!("select('{}'): no step is labelled it", labels[0]))?;
                    let items = slots.into_iter().map(Expr::Slot).collect();
                    plan.project(vec![(labels[0].clone(), Expr::List { items })])
                } else if labels.len() == 1 {
                    plan.project(vec![(labels[0].clone(), val_of(0, &labels[0])?)])
                } else {
                    let entries = labels
                        .iter()
                        .enumerate()
                        .map(|(i, l)| Ok((l.clone(), val_of(i, l)?)))
                        .collect::<Result<Vec<_>, String>>()?;
                    plan.project(vec![("select".into(), Expr::MapLit { entries })])
                };
                self.current = 0;
                self.slots = 1;
                p
            }
            "project" => {
                // project('a','b',…).by(x).by(y) → one Map per traverser, keyed by the
                // labels. Value for key i is the i-th `by` modulator, or the current
                // element when there is no i-th `by` (core's `bys.get(i)` — NOT cycled).
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
                let p = plan.project(vec![("project".into(), Expr::MapLit { entries })]);
                self.current = 0;
                self.slots = 1;
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
                let p = plan
                    .aggregate(
                        vec![key_expr],
                        vec![Agg {
                            func: AggFn::Count,
                            arg: None,
                            distinct: false,
                            name: "count".into(),
                            frac: None, null_on_empty: false,
                        }],
                    )
                    // Gremlin groupCount() is a single {key: count} Map, not (k,c) rows.
                    .group_to_map();
                self.current = 0;
                self.slots = 1; // one Map column
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
                        frac: None, null_on_empty: false,
                    },
                    Some(GroupBy::Key(k)) => Agg {
                        func: AggFn::Collect,
                        arg: Some(Expr::Prop {
                            slot: elem_slot,
                            key: k.clone(),
                        }),
                        distinct: false,
                        name: "value".into(),
                        frac: None, null_on_empty: false,
                    },
                    Some(GroupBy::KeyExpr(_, e)) => Agg {
                        func: AggFn::Collect,
                        arg: Some(e.clone()),
                        distinct: false,
                        name: "value".into(),
                        frac: None, null_on_empty: false,
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
                    },
                    // Default (no second by) or bare by(): fold the group's elements.
                    _ => Agg {
                        func: AggFn::Collect,
                        arg: Some(Expr::Slot(elem_slot)),
                        distinct: false,
                        name: "value".into(),
                        frac: None, null_on_empty: false,
                    },
                };
                let p = plan
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
                    Expr::Prop { slot: self.current, key: k }
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
                        let mut scalar =
                            aggs[0].arg.clone().unwrap_or(Expr::Slot(self.current));
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
                    _ => return Err(
                        "local(<traversal>) beyond a hop chain or count()/fold() is deferred"
                            .into(),
                    ),
                }
            }
            "withcomputer" => {
                // A no-op marker (lenke always computes in-process), matching core.
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
                            // readable once `propertyName` redirects the result (core
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
                                        crate::ir::GremlinAlgo::PageRank { damping, iterations: n }
                                    }
                                    crate::ir::GremlinAlgo::PeerPressure { .. } => {
                                        crate::ir::GremlinAlgo::PeerPressure { iterations: n }
                                    }
                                    other => other,
                                };
                                Plan::AlgoAnnotate { input, algo, edge_label, node_slot }
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
                                return Err(
                                    "with(target, …) applies only to shortestPath()".into()
                                )
                            }
                        }
                    }
                    other => {
                        return Err(format!("with({other}, …) is not yet supported"))
                    }
                }
            }
            "barrier" => {
                // A lazy-barrier is a no-op in this eager executor (matching core, where
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
                // collect), matching core. Revealed later by cap(key).
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
                // byte-identity with core.
                let key = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                if let Some((snap, edge_slot)) = self.subgraph_caps.get(&key).cloned() {
                    self.current = 0;
                    self.slots = 1;
                    return Ok(snap.subgraph(edge_slot));
                }
                let Some((snap, expr)) = self.caps.get(&key).cloned() else {
                    // An unfilled key caps to a single EMPTY list (core), not an error.
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
                        frac: None, null_on_empty: false,
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
                let p = Plan::Union {
                    left: Box::new(cur),
                    right: Box::new(lit_plan),
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
        // re-arm the edge-hop pointer the top-of-step take() cleared.
        if matches!(
            lname.as_str(),
            "has" | "hasnot" | "haskey" | "hasvalue" | "subgraph" | "aggregate" | "store"
                | "dedup" | "where" | "as"
        ) {
            self.edge_hop = prev_edge_hop;
        }
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
    fn apply_property(&self, plan: Plan, key: String, val: Value) -> Plan {
        match plan {
            Plan::Insert { mut nodes, edges } if edges.is_empty() && nodes.len() == 1 => {
                nodes[0].props.push((key, val));
                Plan::Insert { nodes, edges }
            }
            Plan::Update { input, mut ops } => {
                ops.push(crate::ir::SetOp::Set {
                    slot: self.current,
                    key,
                    value: Expr::Lit(val),
                });
                Plan::Update { input, ops }
            }
            read => Plan::Update {
                input: Box::new(read),
                ops: vec![crate::ir::SetOp::Set {
                    slot: self.current,
                    key,
                    value: Expr::Lit(val),
                }],
            },
        }
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
            return Ok(Expr::Not(Box::new(Expr::And(Box::new(exists), Box::new(base)))));
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
                    // null (3VL null would make Not→null→dropped, but core keeps it).
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

    /// Parse a single anonymous hop body — `[__.] (out|in|both) ( [label] )` — and
    /// return its `(direction, edge label)`. Shared by the branch steps (union) that
    /// take hop sub-traversals. Multi-label / multi-step bodies are deferred.
    /// Parse a Gremlin `math('…')` expression string into an engine `Expr`. `operand`
    /// resolves the `_` variable (and any named step-label variable via `var_slot`).
    /// Grammar precedence (mXparser/TinkerPop): `+ -` < `* / %` < `^` (right-assoc) <
    /// unary `- +` < primary (number, `(expr)`, `name(args)`, bare unary `sin _`,
    /// constant `pi`/`e`, or a variable). Maps to the same f64 kernels GQL uses.
    fn parse_math(&self, src: &str, operand: &Expr) -> Result<Expr, String> {
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

    fn hop_body(&mut self) -> Result<(Dir, Option<String>), String> {
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        let name = self.ident()?;
        let dir = match name.to_ascii_lowercase().as_str() {
            "out" => Dir::Out,
            "in" => Dir::In,
            "both" => Dir::Both,
            other => {
                return Err(format!(
                    "a branch traversal must be a single out/in/both hop, got `{other}`"
                ))
            }
        };
        self.expect(&Tok::LParen)?;
        let label = if matches!(self.peek(), Some(Tok::Str(_))) {
            let l = self.str_arg()?;
            if self.peek() == Some(&Tok::Comma) {
                return Err("a branch hop with multiple edge labels is not yet supported".into());
            }
            Some(l)
        } else {
            None
        };
        self.expect(&Tok::RParen)?;
        Ok((dir, label))
    }

    /// Parse a match hop head `as('s').<hop>('L'?)[.as('e')]`, returning
    /// `(start_tag, direction, edge_label, end_tag?)`. Shared by the `not(...)`
    /// fragment (cursor already past any leading `__.`).
    #[allow(clippy::type_complexity)]
    fn match_hop_head(
        &mut self,
    ) -> Result<(String, Dir, Option<String>, Option<String>), String> {
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
                    other => {
                        return Err(format!("match() has(): unsupported predicate `{other}`"))
                    }
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
        let saved_edge = self.edge_hop.take();
        let saved_repeat = self.pending_repeat.take();
        let saved_path_ok = self.path_ok;
        let saved_on_edge = self.on_edge;
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
        if matches!(self.peek(), Some(Tok::Ident(_))) {
            body = self.step(body)?;
            while self.peek() == Some(&Tok::Dot) {
                self.bump();
                body = self.step(body)?;
            }
        }
        let out_current = self.current;
        let out_slots = self.slots;
        self.current = saved_current;
        self.slots = saved_slots;
        self.edge_hop = saved_edge;
        self.pending_repeat = saved_repeat;
        self.path_ok = saved_path_ok;
        self.on_edge = saved_on_edge;
        Ok((body, out_current, out_slots))
    }

    /// Look ahead — WITHOUT consuming — for a coalesce/branch body's leading navigating
    /// hop `[__.](out|in|both)('L', …)`, returning its `(direction, edge labels)`. Used
    /// to form a body's existence guard before parsing it. `None` when the body does not
    /// start with a hop (a value body such as `constant(v)` always produces output).
    /// Parse a union branch that is a PER-ELEMENT `[__.](out|in|both)('L'…).count()`,
    /// returning a `Row`-projected `CountSubquery` (one count per input row). Returns
    /// `None` with the cursor unchanged when the body is not that shape.
    fn try_count_body(&mut self, from: usize, width: usize) -> Result<Option<Plan>, String> {
        let save = self.pos;
        let mut p = self.pos;
        if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
            p += 1;
            if self.toks.get(p) == Some(&Tok::Dot) {
                p += 1;
            }
        }
        let dir = match self.toks.get(p) {
            Some(Tok::Ident(s)) => match s.to_ascii_lowercase().as_str() {
                "out" => Dir::Out,
                "in" => Dir::In,
                "both" => Dir::Both,
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        // Consume `[__.]<hop>(labels).count()`, bailing (cursor restored) on any mismatch.
        if matches!(self.peek(), Some(Tok::Ident(s)) if s == "__") {
            self.bump();
            self.expect(&Tok::Dot)?;
        }
        self.ident()?; // hop (dir already resolved)
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
        let is_count = self.peek() == Some(&Tok::Dot)
            && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("count"));
        if !is_count {
            self.pos = save;
            return Ok(None);
        }
        self.expect(&Tok::Dot)?;
        self.ident()?; // count
        self.expect(&Tok::LParen)?;
        self.expect(&Tok::RParen)?;
        let count = Expr::CountSubquery {
            body: Box::new(Plan::Row.expand(from, dir, &labels)),
            outer_width: width,
        };
        Ok(Some(Plan::Row.project(vec![("count".to_string(), count)])))
    }

    /// True when the coalesce body ahead starts with an element FILTER
    /// (`hasLabel`/`has`/`hasNot`/`hasKey`/`hasValue`/`where`/`not`/`and`/`or`/`filter`),
    /// so the body only produces output where that filter holds.
    fn peek_leading_is_filter(&self) -> bool {
        let mut p = self.pos;
        if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
            p += 1;
            if self.toks.get(p) == Some(&Tok::Dot) {
                p += 1;
            }
        }
        matches!(self.toks.get(p), Some(Tok::Ident(s)) if matches!(
            s.to_ascii_lowercase().as_str(),
            "haslabel" | "has" | "hasnot" | "haskey" | "hasvalue" | "where" | "not"
                | "and" | "or" | "filter"
        ))
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

    fn peek_leading_hop(&self) -> Option<(Dir, Vec<String>)> {
        let mut p = self.pos;
        if matches!(self.toks.get(p), Some(Tok::Ident(s)) if s == "__") {
            p += 1;
            if self.toks.get(p) != Some(&Tok::Dot) {
                return None;
            }
            p += 1;
        }
        let dir = match self.toks.get(p) {
            Some(Tok::Ident(s)) => match s.to_ascii_lowercase().as_str() {
                "out" => Dir::Out,
                "in" => Dir::In,
                "both" => Dir::Both,
                _ => return None,
            },
            _ => return None,
        };
        p += 1;
        if self.toks.get(p) != Some(&Tok::LParen) {
            return None;
        }
        p += 1;
        let mut labels = Vec::new();
        while let Some(Tok::Str(s)) = self.toks.get(p) {
            labels.push(s.clone());
            p += 1;
            if self.toks.get(p) == Some(&Tok::Comma) {
                p += 1;
            } else {
                break;
            }
        }
        Some((dir, labels))
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
            // pattern is validated at PARSE time (like core), so a bad pattern is a
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
    ) -> Result<(Dir, Option<String>, Option<String>, Option<Expr>, Option<u32>), String> {
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
        // matching core. Otherwise the frontier passes through.
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
            if n >= 1
                && ctx.until.is_none()
                && ctx.body_filter.is_none()
                && ctx.max_cap.is_none()
            {
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
            self.all_labels.entry(tag.clone()).or_default().push(ctx.out_slot);
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

    /// The VALUE cells of each row, order-independent and NAME-independent — so a
    /// GQL result and a Gremlin result can be compared even though their column
    /// names differ.
    fn value_bag(rows: &Rows) -> Vec<String> {
        let mut out: Vec<String> = rows
            .rows
            .iter()
            .map(|r| r.iter().map(|v| format!("{v:?};")).collect::<String>())
            .collect();
        out.sort();
        out
    }

    fn gremlin_rows(q: &str, store: &Store) -> Rows {
        let plan = super::parse(q).unwrap_or_else(|e| panic!("parse gremlin `{q}`: {e}"));
        run(&plan, store)
    }
    fn gql_rows(q: &str, store: &Store) -> Rows {
        let plan = crate::gql::parse(q).unwrap_or_else(|e| panic!("parse gql `{q}`: {e}"));
        run(&plan, store)
    }

    /// `is(P)` filters the VALUE stream by a predicate (like `where`); `is(literal)`
    /// is an equality test. Ages are alice 30, bob 25, carol 40.
    #[test]
    fn gremlin_is_value_predicate() {
        let store = social();
        let gt = value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').values('age').is(gt(28))",
            &store,
        ));
        assert_eq!(gt, vec!["Num(30.0);", "Num(40.0);"]);
        // Bare literal → equality.
        let eq = value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').values('age').is(25)",
            &store,
        ));
        assert_eq!(eq, vec!["Num(25.0);"]);
    }

    /// `g.V('id', …)` is a READ source: seed the frontier with exactly the vertices
    /// carrying those external ids (dense-id strings here), then traverse as usual.
    /// A missing id contributes nothing — like core's `g.V(<absent>)`.
    #[test]
    fn gremlin_v_by_external_id_read_source() {
        let store = social();
        // Single id → that vertex.
        let one = value_bag(&gremlin_rows("g.V('0').values('name')", &store));
        assert_eq!(one, vec!["Str(\"alice\");"]);
        // Several ids → their union, order-independent.
        let many = value_bag(&gremlin_rows("g.V('1', '2').values('name')", &store));
        assert_eq!(many, vec!["Str(\"bob\");", "Str(\"carol\");"]);
        // Seeds a real frontier: hops compose off it.
        let alice_out = value_bag(&gremlin_rows(
            "g.V('0').out('KNOWS').values('name')",
            &store,
        ));
        assert_eq!(alice_out, vec!["Str(\"bob\");", "Str(\"carol\");"]);
        // A non-existent id yields nothing (no error).
        let gone = value_bag(&gremlin_rows("g.V('999').values('name')", &store));
        assert!(
            gone.is_empty(),
            "missing id must contribute nothing: {gone:?}"
        );
    }

    /// The 3-arg `has('Label','k',pred)` is a label check AND a property predicate,
    /// and `hasLabel` now works anywhere (after a hop, and with multiple labels =
    /// ANY of them) via a runtime `label ∈ labels(n)` membership — not just folded
    /// into the scan right after `V()`.
    #[test]
    fn gremlin_has_label_forms() {
        let store = social();
        // has(label, key, pred) == 'Label' IN labels(n) AND n.key = pred.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().has('Person','name','alice').values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n) WHERE 'Person' IN labels(n) AND n.name='alice' RETURN n.name",
                &store,
            )),
        );
        // hasLabel after a hop (not right after V()): alice's non-Person neighbour.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V('0').out().hasLabel('Project').values('name')",
                &store,
            )),
            vec!["Str(\"graphdb\");"],
        );
        // Multi-label hasLabel matches ANY of the labels (all 4 nodes here).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person','Project').count()",
                &store
            )),
            vec!["Num(4.0);"],
        );
        // The single-label-after-V() fast path and 2-arg has still work.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Project').values('name')",
                &store
            )),
            vec!["Str(\"graphdb\");"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().has('name','bob').values('name')",
                &store
            )),
            vec!["Str(\"bob\");"],
        );
    }

    /// `elementMap()` is core's FLAT element map — `{id, label, <props…>}` for a
    /// node, plus `IN`/`OUT` endpoint stubs for an edge — with a SINGULAR label and
    /// the properties flattened alongside the tokens. `elementMap('k',…)` filters the
    /// properties. This is the Gremlin/TinkerPop shape (distinct from the nested
    /// `{id, labels, properties}` render used for a bare returned element).
    #[test]
    fn gremlin_element_map_flat_shape() {
        let store = social();
        // Node: id + singular label + flattened (sorted) present properties.
        assert_eq!(
            value_bag(&gremlin_rows("g.V('0').elementMap()", &store)),
            vec![
                "Map([(Str(\"id\"), Str(\"0\")), (Str(\"label\"), Str(\"Person\")), \
                 (Str(\"age\"), Num(30.0)), (Str(\"name\"), Str(\"alice\"))]);",
            ],
        );
        // A key filter restricts the flattened properties.
        assert_eq!(
            value_bag(&gremlin_rows("g.V('0').elementMap('name')", &store)),
            vec![
                "Map([(Str(\"id\"), Str(\"0\")), (Str(\"label\"), Str(\"Person\")), \
                 (Str(\"name\"), Str(\"alice\"))]);",
            ],
        );
        // Edge: id + type label + IN (destination) / OUT (source) stubs, matching
        // core's element_map_val (IN = e_dst, OUT = e_src). alice(0)→bob(1) KNOWS = e0.
        let edge = value_bag(&gremlin_rows("g.V('0').outE('KNOWS').elementMap()", &store));
        assert!(edge.iter().any(|s| s.contains(
            "(Str(\"id\"), Str(\"e0\")), (Str(\"label\"), Str(\"KNOWS\")), \
             (Str(\"IN\"), Map([(Str(\"id\"), Str(\"1\")), (Str(\"label\"), Str(\"Person\"))])), \
             (Str(\"OUT\"), Map([(Str(\"id\"), Str(\"0\")), (Str(\"label\"), Str(\"Person\"))]))"
        )));
    }

    /// `coalesce(<hop>, …)` takes the FIRST branch that yields per element (an Exists
    /// guard chain over the same Branch reconverge); `choose(<pred>, <thenHop>,
    /// <elseHop>)` routes by a predicate; `optional(<hop>)` advances if the hop
    /// yields, else keeps the element (OptionalExpand keep_source). All keep a
    /// continuable frontier.
    #[test]
    fn gremlin_coalesce_choose_optional() {
        let store = social();
        // coalesce: WORKS_ON if present (alice→graphdb), else out KNOWS (bob→carol);
        // carol has neither → nothing.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').coalesce(out('WORKS_ON'), out('KNOWS')).values('name')",
                &store,
            )),
            vec!["Str(\"carol\");", "Str(\"graphdb\");"],
        );
        // choose: alice routes to out KNOWS (bob, carol); the others to out WORKS_ON
        // (none, since only alice has it).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').choose(has('name','alice'), out('KNOWS'), out('WORKS_ON')).values('name')",
                &store,
            )),
            vec!["Str(\"bob\");", "Str(\"carol\");"],
        );
        // optional: alice→bob,carol; bob→carol; carol has no out KNOWS → stays carol.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').optional(out('KNOWS')).values('name')",
                &store,
            )),
            vec![
                "Str(\"bob\");",
                "Str(\"carol\");",
                "Str(\"carol\");",
                "Str(\"carol\");",
            ],
        );
        // A missed optional keeps the element (frontier continues).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V('2').optional(out('KNOWS')).values('name')",
                &store
            )),
            vec!["Str(\"carol\");"],
        );
    }

    /// `union(<hop>, …)` concatenates each branch's frontier per element and — unlike
    /// GQL's materializing UNION — keeps it a node frontier, so the traversal
    /// CONTINUES (`.values()`, `.count()`, another hop). This is core's per-traverser
    /// branch-and-reconverge, expressed columnar via Plan::Branch over pull_body.
    #[test]
    fn gremlin_union_of_hops() {
        let store = social();
        // alice's KNOWS targets (bob, carol) unioned with her WORKS_ON target (graphdb).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V('0').union(out('KNOWS'), out('WORKS_ON')).values('name')",
                &store,
            )),
            vec!["Str(\"bob\");", "Str(\"carol\");", "Str(\"graphdb\");"],
        );
        // The union frontier continues: count() sees all three, values() reads them.
        assert_eq!(
            value_bag(&gremlin_rows("g.V('0').union(out(), in()).count()", &store)),
            vec!["Num(3.0);"],
        );
        // bob: out KNOWS (carol) unioned with in KNOWS (alice).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V('1').union(out('KNOWS'), in('KNOWS')).values('name')",
                &store,
            )),
            vec!["Str(\"alice\");", "Str(\"carol\");"],
        );
        // Multi-step branch bodies are now supported (arbitrary sub-traversals per
        // branch, not just a single hop): alice's 2-hop KNOWS reach unioned with a
        // value body still parses and runs.
        assert!(super::parse("g.V().union(out().out(), in())").is_ok());
        assert!(super::parse("g.V().union(values('name'), out('KNOWS').values('name'))").is_ok());
    }

    /// The scope-LOCAL aggregates `count`/`sum`/`mean`/`min`/`max`(local) reduce the
    /// current list cell PER ROW (after `fold()`), where the bare/global forms fold
    /// the whole stream to one scalar. Over the folded Person ages [30,25,40]: local
    /// count 3, sum 95, mean 95/3, min 25, max 40 — and the local sum equals the
    /// global sum of the same values.
    #[test]
    fn gremlin_local_scope_aggregates() {
        let store = social();
        let folded = "g.V().hasLabel('Person').values('age').fold()";
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{folded}.count(local)"), &store)),
            vec!["Num(3.0);"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{folded}.sum(local)"), &store)),
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').values('age').sum()",
                &store
            )),
        );
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{folded}.min(local)"), &store)),
            vec!["Num(25.0);"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{folded}.max(local)"), &store)),
            vec!["Num(40.0);"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{folded}.mean(local)"), &store)),
            vec!["Num(31.666666666666668);"],
        );
    }

    /// `tail(n)` keeps the LAST n rows of the committed order — the mirror of
    /// `limit(n)` (the first n). After `order().by('age')` it is the top-n by age.
    #[test]
    fn gremlin_tail_is_the_last_n() {
        let store = social();
        let ages = "g.V().hasLabel('Person').order().by('age').values('age')";
        // tail(2) = the two largest ages (the tail of the ascending order); limit(2)
        // = the two smallest — different windows of the same order.
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{ages}.tail(2)"), &store)),
            vec!["Num(30.0);", "Num(40.0);"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{ages}.limit(2)"), &store)),
            vec!["Num(25.0);", "Num(30.0);"],
        );
        // Default n = 1 (the single largest); an oversized n keeps everything.
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{ages}.tail()"), &store)),
            vec!["Num(40.0);"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(&format!("{ages}.tail(99)"), &store)),
            vec!["Num(25.0);", "Num(30.0);", "Num(40.0);"],
        );
    }

    /// `hasId('a', …)` keeps the element iff its external id is one of the given ids
    /// — an `element_id`-in-list filter, verified equal to the GQL `element_id(n) = …`
    /// predicate. Works on nodes and edges.
    #[test]
    fn gremlin_has_id_filters_by_external_id() {
        let store = social();
        assert_eq!(
            value_bag(&gremlin_rows("g.V().hasId('0','1').values('name')", &store)),
            value_bag(&gql_rows(
                "MATCH (n) WHERE element_id(n)='0' OR element_id(n)='1' RETURN n.name",
                &store,
            )),
        );
        // A single id, and an edge id.
        assert_eq!(
            value_bag(&gremlin_rows("g.V().hasId('2').values('name')", &store)),
            vec!["Str(\"carol\");"],
        );
        assert_eq!(
            value_bag(&gremlin_rows("g.E().hasId('e0').count()", &store)),
            vec!["Num(1.0);"],
        );
    }

    /// `simplePath()` keeps traversers whose vertex path has NO repeat; `cyclicPath()`
    /// keeps those that DO — a partition of the stream. They read the lineage node
    /// path (like `path()`), so they are scoped to pure vertex-hop chains. A 2-hop
    /// `both` walk from a node returns to it on half the paths (the cyclic ones).
    #[test]
    fn gremlin_simple_and_cyclic_path() {
        let store = social();
        // 2-hop BOTH from alice: [0,1,0] and [0,2,0] return to alice (cyclic); [0,1,2]
        // and [0,2,1] reach a new node (simple).
        let base = "g.V('0').both('KNOWS').both('KNOWS')";
        assert_eq!(
            value_bag(&gremlin_rows(
                &format!("{base}.simplePath().values('name')"),
                &store
            )),
            vec!["Str(\"bob\");", "Str(\"carol\");"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(
                &format!("{base}.cyclicPath().values('name')"),
                &store
            )),
            vec!["Str(\"alice\");", "Str(\"alice\");"],
        );
        // The two are complementary: together they are the whole stream (4 paths).
        let all = value_bag(&gremlin_rows(&format!("{base}.count()"), &store));
        assert_eq!(all, vec!["Num(4.0);"]);
        assert_eq!(
            value_bag(&gremlin_rows(
                &format!("{base}.simplePath().count()"),
                &store
            )),
            vec!["Num(2.0);"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(
                &format!("{base}.cyclicPath().count()"),
                &store
            )),
            vec!["Num(2.0);"],
        );
        // Only over a pure vertex-hop chain — a value stream is deferred.
        assert!(super::parse("g.V().values('name').simplePath()").is_err());
    }

    /// `and`/`or`/`not` accept navigating hop children too, each a semi-join
    /// `Expr::Exists` (the same construction as `where(<hop>)`). So `not(out('L'))`
    /// is the ANTI-join (elements without such an edge) and `and(out('L'), has(…))`
    /// mixes an edge test with a property test — verified equal to the GQL
    /// EXISTS/NOT EXISTS forms.
    #[test]
    fn gremlin_and_or_not_hop_children() {
        let store = social();
        // not(out(KNOWS)) = the anti-join: vertices WITHOUT an out-KNOWS edge.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().not(out('KNOWS')).values('name')",
                &store
            )),
            value_bag(&gql_rows(
                "MATCH (n) WHERE NOT EXISTS { (n)-[:KNOWS]->() } RETURN n.name",
                &store,
            )),
        );
        // and(<hop>, <property>) mixes a semi-join with a predicate.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().and(out('KNOWS'), has('name','alice')).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n) WHERE EXISTS { (n)-[:KNOWS]->() } AND n.name='alice' RETURN n.name",
                &store,
            )),
        );
        // or of two different hops.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().or(out('WORKS_ON'), in('KNOWS')).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n) WHERE EXISTS { (n)-[:WORKS_ON]->() } OR EXISTS { (n)<-[:KNOWS]-() } \
                 RETURN n.name",
                &store,
            )),
        );
        // A multi-label hop child is now supported (Exists over a disjunction hop).
        assert!(super::parse("g.V().and(out('A','B'), has('k'))").is_ok());
    }

    /// `where(<hop>)` is a semi-join: keep the current element iff it HAS such an
    /// adjacency. It lowers to an `Expr::Exists` whose body seeds `Plan::Row` and
    /// expands from the current slot — the same shape GQL's `EXISTS { … }` builds —
    /// so it equals the GQL `WHERE EXISTS { (n)-[:L]->() }` form. `where(P)` (the
    /// value-predicate form) is unchanged.
    #[test]
    fn gremlin_where_hop_semijoin() {
        let store = social();
        // Vertices with an out-KNOWS edge (alice, bob) == the GQL EXISTS form.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().where(out('KNOWS')).values('name')",
                &store
            )),
            value_bag(&gql_rows(
                "MATCH (n) WHERE EXISTS { (n)-[:KNOWS]->() } RETURN n.name",
                &store,
            )),
        );
        // Incoming KNOWS (bob, carol) == the reverse EXISTS form.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().where(in('KNOWS')).values('name')",
                &store
            )),
            value_bag(&gql_rows(
                "MATCH (n) WHERE EXISTS { (n)<-[:KNOWS]-() } RETURN n.name",
                &store,
            )),
        );
        // Argless where(out()) is any out-edge; both() is either direction.
        assert_eq!(
            value_bag(&gremlin_rows("g.V().where(out()).values('name')", &store)),
            vec!["Str(\"alice\");", "Str(\"bob\");"],
        );
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().where(both('KNOWS')).values('name')",
                &store
            )),
            vec!["Str(\"alice\");", "Str(\"bob\");", "Str(\"carol\");"],
        );
        // The value-predicate form still works (age > 28 → carol, alice).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').values('age').where(gt(28))",
                &store,
            )),
            vec!["Num(30.0);", "Num(40.0);"],
        );
        // A multi-label where-hop is a semi-join over either edge type.
        assert!(super::parse("g.V().where(out('A','B'))").is_ok());
    }

    /// `and(f1,f2,…)`, `or(f1,f2,…)`, `not(f)` combine element filters (has/hasNot,
    /// nested and/or/not) into one predicate over the current element — the direct
    /// Gremlin spelling of a boolean `WHERE`. Verified equal to the equivalent GQL
    /// `WHERE … AND/OR/NOT …`. Navigating child traversals (semi-joins) are deferred.
    #[test]
    fn gremlin_and_or_not_filter_combinators() {
        let store = social();
        // and: two conjoined predicates == GQL AND.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').and(has('age', gt(28)), has('name', neq('carol'))).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n:Person) WHERE n.age > 28 AND n.name <> 'carol' RETURN n.name",
                &store,
            )),
        );
        // or: disjunction == GQL OR.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').or(has('age', lt(28)), has('name', 'carol')).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n:Person) WHERE n.age < 28 OR n.name = 'carol' RETURN n.name",
                &store,
            )),
        );
        // not: negation == GQL NOT.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').not(has('age', gt(28))).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n:Person) WHERE NOT (n.age > 28) RETURN n.name",
                &store,
            )),
        );
        // Nested and/or compose (an or inside an and) == the GQL parenthesized form.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').and(has('age', gt(20)), or(has('name','bob'), has('name','carol'))).values('name')",
                &store,
            )),
            value_bag(&gql_rows(
                "MATCH (n:Person) WHERE n.age > 20 AND (n.name = 'bob' OR n.name = 'carol') RETURN n.name",
                &store,
            )),
        );
        // Navigating child traversals are now semi-joins (see
        // `gremlin_and_or_not_hop_children`); they parse rather than error.
        assert!(super::parse("g.V().and(out('KNOWS'), has('age', gt(1)))").is_ok());
        assert!(super::parse("g.V().not(out('KNOWS'))").is_ok());
    }

    /// `group().by(key).by(value)` is a grouped aggregation. Core shapes it as one
    /// {key: value} map; the engine represents a grouped result as ROWS of (key,
    /// value), consistent with `groupCount()`. `by(count())` reduces to a count (so
    /// `group().by('k').by(count())` == `groupCount().by('k')`); a property/element
    /// value folds the group (collect, Gremlin fold), elements folded as their ids.
    #[test]
    fn gremlin_group_by_key_and_value() {
        let store = social();
        // by(count()) reduces — identical to groupCount().by('k').
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').out('KNOWS').group().by('name').by(count())",
                &store,
            )),
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').out('KNOWS').groupCount().by('name')",
                &store,
            )),
        );
        // Default value-by folds the group's ELEMENTS, each rendered as its element
        // map (like a top-level vertex, so a folded vertex canonicalizes the same way).
        // Names are unique here, so each group holds one element: alice(0), bob(1),
        // carol(2).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').group().by('name')",
                &store
            )),
            // group() folds to ONE Gremlin Map {name: [elements]} (first-seen key
            // order), matching core.
            vec![
                "Map([(Str(\"alice\"), List([Map([(Str(\"id\"), Str(\"0\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(30.0)), (Str(\"name\"), Str(\"alice\"))]))])])), (Str(\"bob\"), List([Map([(Str(\"id\"), Str(\"1\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(25.0)), (Str(\"name\"), Str(\"bob\"))]))])])), (Str(\"carol\"), List([Map([(Str(\"id\"), Str(\"2\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(40.0)), (Str(\"name\"), Str(\"carol\"))]))])]))]);",
            ],
        );
        // A property value-by folds that property per group.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').group().by('name').by('age')",
                &store,
            )),
            vec![
                "Map([(Str(\"alice\"), List([Num(30.0)])), (Str(\"bob\"), List([Num(25.0)])), (Str(\"carol\"), List([Num(40.0)]))]);",
            ],
        );
        // A <hop>.count() value-by is the group's total degree (a Sum of per-element
        // degrees).
        assert!(super::parse("g.V().group().by('name').by(out().count())").is_ok());
    }

    /// `project('a','b').by(x).by(y)` builds one insertion-ordered Map per traverser:
    /// key i takes the i-th `by` modulator, or the current element when there is no
    /// i-th `by` (core's `bys.get(i)`, not cycled). `by('key')` reads a property; a
    /// key with no `by` yields the element as its id, consistent with `select()`.
    #[test]
    fn gremlin_project_by_modulators() {
        let store = social();
        // Two by-modulators → {n: name, a: age}, keys in project order.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').project('n','a').by('name').by('age')",
                &store,
            )),
            vec![
                "Map([(Str(\"n\"), Str(\"alice\")), (Str(\"a\"), Num(30.0))]);",
                "Map([(Str(\"n\"), Str(\"bob\")), (Str(\"a\"), Num(25.0))]);",
                "Map([(Str(\"n\"), Str(\"carol\")), (Str(\"a\"), Num(40.0))]);",
            ],
        );
        // A key with fewer bys than keys defaults to the current element, rendered as
        // its element map (like a top-level vertex): project('n','self').by('name') →
        // {n: name, self: {id,labels,properties}}.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().has('name','bob').project('n','self').by('name')",
                &store,
            )),
            vec![
                "Map([(Str(\"n\"), Str(\"bob\")), (Str(\"self\"), Map([(Str(\"id\"), Str(\"1\")), (Str(\"labels\"), List([Str(\"Person\")])), (Str(\"properties\"), Map([(Str(\"age\"), Num(25.0)), (Str(\"name\"), Str(\"bob\"))]))]))]);"
            ],
        );
        // A degree `<hop>.count()` by-modulator is a correlated count subquery.
        assert!(super::parse("g.V().project('c').by(out().count())").is_ok());
    }

    /// `path()` over a pure vertex-hop chain yields the sequence of vertices visited,
    /// each rendered as its element map (not a bare id). Verified structurally: the
    /// path elements ARE node maps whose id sequence is the hop sequence; and every
    /// non-vertex-hop shape (edge steps, the E source, value projections, var-length)
    /// is deferred rather than mis-answered.
    #[test]
    fn gremlin_path_vertex_hop_chain() {
        let store = social();
        // Pull the "id" of each node-map element, per path, as a sorted set of seqs.
        fn id_seqs(q: &str, store: &Store) -> Vec<Vec<String>> {
            let rows = gremlin_rows(q, store);
            let mut out: Vec<Vec<String>> = rows
                .rows
                .iter()
                .map(|r| match &r[0] {
                    Value::List(elems) => elems
                        .iter()
                        .map(|e| match e {
                            Value::Map(pairs) => pairs
                                .iter()
                                .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "id"))
                                .and_then(|(_, v)| match v {
                                    Value::Str(s) => Some(s.to_string()),
                                    _ => None,
                                })
                                .expect("path element is a node map with an id"),
                            o => panic!("path element not a node map: {o:?}"),
                        })
                        .collect(),
                    o => panic!("path is not a list: {o:?}"),
                })
                .collect();
            out.sort();
            out
        }
        // Single vertex → a one-element path.
        assert_eq!(
            id_seqs("g.V('0').path()", &store),
            vec![vec!["0".to_string()]]
        );
        // One hop from alice(0) → [0,1] and [0,2].
        assert_eq!(
            id_seqs("g.V('0').out('KNOWS').path()", &store),
            vec![
                vec!["0".to_string(), "1".to_string()],
                vec!["0".to_string(), "2".to_string()],
            ],
        );
        // Two hops: alice->bob->carol is the only length-2 KNOWS walk from alice.
        assert_eq!(
            id_seqs("g.V('0').out('KNOWS').out('KNOWS').path()", &store),
            vec![vec!["0".to_string(), "1".to_string(), "2".to_string()]],
        );
        // An interleaved node/edge path (`outE().inV()`) renders through GremlinPath.
        assert!(super::parse("g.V().outE().inV().path()").is_ok());
        // Still deferred: the bare `E` source, and a value projection before path().
        assert!(super::parse("g.E().path()").is_err());
        assert!(super::parse("g.V().values('name').path()").is_err());
        // A `repeat(<vertex-hop>)` walk records full path lineage, so path() over it
        // parses (the VarLength endpoints carry their node chains).
        assert!(super::parse("g.V().repeat(out('KNOWS')).times(2).path()").is_ok());
    }

    /// `valueMap()` projects a PROPERTIES-only map (no id/label tokens) with scalar
    /// values; `valueMap('k',…)` filters keys. Present-properties only — the Project
    /// node has no `age`, so its map omits it. The maps equal the `properties`
    /// sub-map of the engine's GQL element render, which is byte-identical to core.
    #[test]
    fn gremlin_valuemap_properties_only() {
        let store = social();
        // All properties (keys sorted): the three Persons carry name+age, the
        // Project only name.
        assert_eq!(
            value_bag(&gremlin_rows("g.V().valueMap()", &store)),
            vec![
                "Map([(Str(\"age\"), Num(25.0)), (Str(\"name\"), Str(\"bob\"))]);",
                "Map([(Str(\"age\"), Num(30.0)), (Str(\"name\"), Str(\"alice\"))]);",
                "Map([(Str(\"age\"), Num(40.0)), (Str(\"name\"), Str(\"carol\"))]);",
                "Map([(Str(\"name\"), Str(\"graphdb\"))]);",
            ],
        );
        // Key filter: only the named property, when present.
        assert_eq!(
            value_bag(&gremlin_rows("g.V().valueMap('name')", &store)),
            vec![
                "Map([(Str(\"name\"), Str(\"alice\"))]);",
                "Map([(Str(\"name\"), Str(\"bob\"))]);",
                "Map([(Str(\"name\"), Str(\"carol\"))]);",
                "Map([(Str(\"name\"), Str(\"graphdb\"))]);",
            ],
        );
        // Filtering an absent key drops it: the Project keeps only name under
        // valueMap('name','age').
        assert_eq!(
            value_bag(&gremlin_rows("g.V().valueMap('name','age')", &store)),
            value_bag(&gremlin_rows("g.V().valueMap()", &store)),
        );
    }

    /// `id()` projects the element's preserved external id (via `element_id`), and
    /// `label()` a single label string — a vertex's label or an edge's type (via
    /// `element_label`), both polymorphic over the current node/edge slot. Verified
    /// vs the engine's own GQL `element_id`/`type` and vs the fixture's known labels.
    #[test]
    fn gremlin_id_and_label_accessors() {
        let store = social();
        // id() == element_id over the same elements.
        assert_eq!(
            value_bag(&gremlin_rows("g.V().id()", &store)),
            value_bag(&gql_rows("MATCH (n) RETURN element_id(n)", &store)),
        );
        assert_eq!(
            value_bag(&gremlin_rows("g.V().outE().id()", &store)),
            value_bag(&gql_rows("MATCH ()-[r]->() RETURN element_id(r)", &store)),
        );
        // Vertex label() == its single label (Person x3, Project x1).
        assert_eq!(
            value_bag(&gremlin_rows("g.V().label().dedup()", &store)),
            vec!["Str(\"Person\");", "Str(\"Project\");"],
        );
        // Edge label() == edge type, matching GQL type().
        assert_eq!(
            value_bag(&gremlin_rows("g.V().outE().label()", &store)),
            value_bag(&gql_rows("MATCH ()-[r]->() RETURN type(r)", &store)),
        );
    }

    /// `repeat(<hop>).times(n)` applies a single anonymous hop exactly n times — a
    /// walk of length n. Verified against the equivalent chain of n plain hops (both
    /// are walks, so this exercises `VarLength{min:n,max:n,trail:false}` end to end);
    /// GQL var-length is a TRAIL, so the chained-hop equivalent is the right oracle.
    #[test]
    fn gremlin_repeat_times_fixed_length_walk() {
        let store = social();
        // times(n) == n chained hops, for out and both, by rows and by count.
        for (repeat_form, chain_form) in [
            (
                "g.V().repeat(out('KNOWS')).times(1).values('name')",
                "g.V().out('KNOWS').values('name')",
            ),
            (
                "g.V().repeat(out('KNOWS')).times(2).values('name')",
                "g.V().out('KNOWS').out('KNOWS').values('name')",
            ),
            (
                "g.V().repeat(__.out('KNOWS')).times(2).values('name')",
                "g.V().out('KNOWS').out('KNOWS').values('name')",
            ),
            (
                "g.V().repeat(both('KNOWS')).times(2).count()",
                "g.V().both('KNOWS').both('KNOWS').count()",
            ),
        ] {
            assert_eq!(
                value_bag(&gremlin_rows(repeat_form, &store)),
                value_bag(&gremlin_rows(chain_form, &store)),
                "{repeat_form} must equal {chain_form}",
            );
        }
        // Deferred / malformed forms error rather than silently mis-answer.
        assert!(super::parse("g.V().repeat(out('KNOWS'))").is_err()); // bare repeat = unbounded
        assert!(super::parse("g.V().repeat(out('KNOWS')).values('name')").is_err()); // not closed by times
        assert!(super::parse("g.V().times(2)").is_err()); // times without repeat
        assert!(super::parse("g.V().repeat(out('A').out('B')).times(2)").is_err());
        // multi-step body
    }

    /// `outE`/`inE`/`bothE` bind the traversed edge (current element becomes the
    /// edge), and the canonical vertex move steps back onto the endpoint the hop
    /// landed: `outE().inV()` == `out()`, `inE().outV()` == `in()`, `*E().otherV()`
    /// == the corresponding both/out/in. Verified against the plain hops (same IR),
    /// and `outE().count()` == `g.E().count()` (every edge is one node's out-edge).
    #[test]
    fn gremlin_edge_hops_and_endpoint_moves() {
        let store = social();
        // The edge frontier: outE over all vertices touches every edge once.
        assert_eq!(
            value_bag(&gremlin_rows("g.V().outE().count()", &store)),
            value_bag(&gremlin_rows("g.E().count()", &store)),
        );
        // Canonical edge-step + vertex-move pairs equal the plain hops.
        for (edge_form, hop_form) in [
            (
                "g.V().outE('KNOWS').inV().values('name')",
                "g.V().out('KNOWS').values('name')",
            ),
            (
                "g.V().inE('KNOWS').outV().values('name')",
                "g.V().in('KNOWS').values('name')",
            ),
            (
                "g.V().bothE('KNOWS').otherV().values('name')",
                "g.V().both('KNOWS').values('name')",
            ),
        ] {
            assert_eq!(
                value_bag(&gremlin_rows(edge_form, &store)),
                value_bag(&gremlin_rows(hop_form, &store)),
                "{edge_form} must equal {hop_form}",
            );
        }
        // Origin-returning combinations read the endpoint straight off the edge.
        assert!(super::parse("g.V().outE().outV().values('name')").is_ok());
        assert!(super::parse("g.V().inE().inV().values('name')").is_ok());
        assert!(super::parse("g.V().bothE().inV().values('name')").is_ok());
        // A vertex move with no edge frontier is still an error.
        assert!(super::parse("g.V().inV()").is_err());
    }

    /// `g.E()` is an all-edges READ source: it seeds the frontier with every live
    /// edge (`social()` has 4: three KNOWS + one WORKS_ON). Cross-checked against the
    /// engine's own GQL front-end — the anonymous directed pattern `()-[r]->()` — so
    /// both lowerings of "every edge" agree; the GQL side is itself proven vs core by
    /// the differential fuzzer. Counting through g.E() exercises the Col::Edges
    /// frontier end to end.
    #[test]
    fn gremlin_e_all_edges_read_source() {
        let store = social();
        let ge = value_bag(&gremlin_rows("g.E().count()", &store));
        assert_eq!(ge, vec!["Num(4.0);"]);
        // Same "count every edge" via GQL's directed anonymous pattern.
        let gql = value_bag(&gql_rows("MATCH ()-[r]->() RETURN count(r)", &store));
        assert_eq!(ge, gql);
        // g.E('id') (edges by external id) seeds the named edge.
        assert!(super::parse("g.E('e0')").is_ok());
    }

    /// `has(k)` filters elements that CARRY property `k`; `hasNot(k)` those that
    /// don't — matching core. Only the `Project` node (graphdb) lacks `age`.
    #[test]
    fn gremlin_has_key_existence_and_hasnot() {
        let store = social();
        let has_age = value_bag(&gremlin_rows("g.V().has('age').values('name')", &store));
        assert_eq!(
            has_age,
            vec!["Str(\"alice\");", "Str(\"bob\");", "Str(\"carol\");"]
        );
        let no_age = value_bag(&gremlin_rows("g.V().hasNot('age').values('name')", &store));
        assert_eq!(no_age, vec!["Str(\"graphdb\");"]);
        // has(k, pred) — the value-predicate form — still works.
        assert!(super::parse("g.V().has('age', gt(28)).values('name')").is_ok());
    }

    /// Argless `out()`/`in()`/`both()` traverse edges of ANY type (matching core),
    /// where a labelled hop is narrower — alice's WORKS_ON target only shows up
    /// through the untyped hop.
    #[test]
    fn gremlin_argless_out_traverses_all_edge_types() {
        let store = social();
        let all = value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').out().values('name')",
            &store,
        ));
        assert!(
            all.iter().any(|r| r.contains("graphdb")),
            "argless out() must follow WORKS_ON too: {all:?}"
        );
        let knows = value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').out('KNOWS').values('name')",
            &store,
        ));
        assert!(
            !knows.iter().any(|r| r.contains("graphdb")),
            "typed out('KNOWS') must not: {knows:?}"
        );
        // in()/both() accept the argless form too.
        assert!(super::parse("g.V().hasLabel('Person').in().values('name')").is_ok());
        assert!(super::parse("g.V().hasLabel('Person').both().values('name')").is_ok());
    }

    /// THE PAYOFF: equivalent GQL and Gremlin queries lower to plans producing
    /// the same rows. Both are thin front-ends over one neutral IR.
    #[test]
    fn gql_and_gremlin_agree() {
        let store = social();
        let pairs = [
            (
                "MATCH (p:Person) RETURN p.name",
                "g.V().hasLabel('Person').values('name')",
            ),
            (
                "MATCH (p:Person) WHERE p.age > 28 RETURN p.name",
                "g.V().hasLabel('Person').has('age', P.gt(28)).values('name')",
            ),
            (
                "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name",
                "g.V().hasLabel('Person').out('KNOWS').values('name')",
            ),
            (
                "MATCH (p:Person) RETURN count(*) AS c",
                "g.V().hasLabel('Person').count()",
            ),
            (
                "MATCH (a:Person)-[:KNOWS]->(b) RETURN DISTINCT b.name",
                "g.V().hasLabel('Person').out('KNOWS').values('name').dedup()",
            ),
            // NOTE: GQL GROUP BY and Gremlin groupCount() do NOT agree by design —
            // GQL yields relational (key, count) ROWS, Gremlin yields a single
            // {key: count} Map (see `bare_group_count_groups_by_the_current_element`).
            // So no groupCount pair belongs in this row-equality list.
        ];
        for (gql, gremlin) in pairs {
            assert_eq!(
                value_bag(&gql_rows(gql, &store)),
                value_bag(&gremlin_rows(gremlin, &store)),
                "GQL `{gql}` and Gremlin `{gremlin}` disagree",
            );
        }
    }

    /// order().by(...).values(...) sorts elements, then projects — hand-checked.
    #[test]
    fn order_by_then_values() {
        let store = social();
        let out = gremlin_rows(
            "g.V().hasLabel('Person').order().by('age', desc).values('name')",
            &store,
        );
        let names: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                _ => panic!(),
            })
            .collect();
        assert_eq!(names, vec!["carol", "alice", "bob"]); // 40, 30, 25
    }

    /// range(lo, hi) is a paging window.
    #[test]
    fn range_is_a_window() {
        let store = social();
        let out = gremlin_rows(
            "g.V().hasLabel('Person').order().by('age').range(1, 2).values('name')",
            &store,
        );
        // ages asc: bob(25), alice(30), carol(40); range(1,2) -> alice
        assert_eq!(out.rows.len(), 1);
        match &out.rows[0][0] {
            Value::Str(x) => assert_eq!(&**x, "alice"),
            _ => panic!(),
        }
    }

    /// The single list cell of a folded result: exactly one row, one column,
    /// which is a `Value::List`. Returned as debug strings, since `Value` has no
    /// `PartialEq` (equality is the value contract's, not derived) — the exact,
    /// ordered element sequence is what an `order(local)` test needs to pin.
    fn fold_list(out: &Rows) -> Vec<String> {
        assert_eq!(out.rows.len(), 1, "fold emits exactly one row");
        assert_eq!(out.rows[0].len(), 1, "fold emits one column");
        match &out.rows[0][0] {
            Value::List(items) => items.iter().map(|v| format!("{v:?}")).collect(),
            other => panic!("expected a list cell, got {other:?}"),
        }
    }

    /// The same debug-string projection for an expected element list.
    fn dbg(items: &[Value]) -> Vec<String> {
        items.iter().map(|v| format!("{v:?}")).collect()
    }

    /// fold() collects the whole value stream into one list. Order is unspecified
    /// without a preceding sort, so the SET of names is what's pinned here.
    #[test]
    fn fold_collects_the_stream() {
        let store = social();
        let out = gremlin_rows("g.V().hasLabel('Person').values('name').fold()", &store);
        let mut got = fold_list(&out);
        got.sort();
        let mut want = dbg(&[s("alice"), s("bob"), s("carol")]);
        want.sort();
        // Person names are alice/bob/carol; graphdb is a Project, excluded.
        assert_eq!(got, want);
    }

    /// fold() over an empty stream still emits exactly one row: the empty list.
    #[test]
    fn fold_of_empty_is_one_empty_list() {
        let store = social();
        let out = gremlin_rows("g.V().hasLabel('Nope').values('name').fold()", &store);
        assert_eq!(fold_list(&out), Vec::<String>::new());
    }

    /// order(local) sorts WITHIN the folded list — ascending by the value contract.
    #[test]
    fn order_local_sorts_the_list_ascending() {
        let store = social();
        let out = gremlin_rows(
            "g.V().hasLabel('Person').values('name').fold().order(local)",
            &store,
        );
        // names sorted ascending, exact order (not a set)
        assert_eq!(fold_list(&out), dbg(&[s("alice"), s("bob"), s("carol")]));
    }

    /// order(local).by(desc) reverses the within-list order; numeric elements sort
    /// numerically (the value contract), not lexically.
    #[test]
    fn order_local_by_desc_on_numbers() {
        let store = social();
        let out = gremlin_rows(
            "g.V().hasLabel('Person').values('age').fold().order(local).by(desc)",
            &store,
        );
        // ages 25/30/40 descending
        assert_eq!(fold_list(&out), dbg(&[n(40.0), n(30.0), n(25.0)]));
    }

    /// `Scope.local` is an accepted spelling of the local scope.
    #[test]
    fn order_scope_local_spelling() {
        let store = social();
        let out = gremlin_rows(
            "g.V().hasLabel('Person').values('name').fold().order(Scope.local)",
            &store,
        );
        assert_eq!(fold_list(&out), dbg(&[s("alice"), s("bob"), s("carol")]));
    }

    /// order(local) faults nothing on a scalar cell — it passes through unchanged
    /// (there is no list to sort), so a global order() is still the stream sort.
    #[test]
    fn order_local_passthrough_on_scalar() {
        let store = social();
        // No fold: each row's slot-0 is a scalar name; order(local) leaves it be.
        let out = gremlin_rows(
            "g.V().hasLabel('Person').values('name').order(local)",
            &store,
        );
        let mut got: Vec<String> = out
            .rows
            .iter()
            .map(|r| match &r[0] {
                Value::Str(x) => x.to_string(),
                other => panic!("{other:?}"),
            })
            .collect();
        got.sort();
        assert_eq!(got, vec!["alice", "bob", "carol"]);
    }

    #[test]
    fn errors_not_panics() {
        assert!(super::parse("g.V(").is_err());
        assert!(super::parse("g.E().values('x')").is_ok()); // g.E() is an all-edges source now
        assert!(super::parse("g.E('e0')").is_ok()); // edges by external id
        assert!(super::parse("g.V().frobnicate()").is_err()); // unknown step
        assert!(super::parse("g.V().has('k')").is_ok()); // has(k) is key-existence now
        assert!(super::parse("g.V().has()").is_err()); // has still needs a key
    }

    // --- writes: addV / property / drop ---

    fn exec(q: &str, store: &mut Store) {
        crate::exec::execute(&super::parse(q).unwrap(), store).unwrap();
    }

    /// `g.addV('L').property(...)` creates one vertex with those properties.
    #[test]
    fn add_vertex_with_properties() {
        let mut st = Builder::default().build();
        exec(
            "g.addV('Person').property('name', 'x').property('age', 1)",
            &mut st,
        );
        assert_eq!(st.node_count(), 1);
        assert_eq!(st.nodes_with_label("Person"), &[0]);
        assert!(matches!(st.prop(0, "name"), Value::Str(v) if &*v == "x"));
        assert!(matches!(st.prop(0, "age"), Value::Num(v) if v == 1.0));
    }

    /// `g.V()...property(k, v)` sets the property on the matched vertices only.
    #[test]
    fn property_step_sets_matched() {
        let mut st = social();
        exec(
            "g.V().hasLabel('Person').has('name', 'alice').property('age', 99)",
            &mut st,
        );
        assert!(matches!(st.prop(0, "age"), Value::Num(v) if v == 99.0)); // alice
        assert!(matches!(st.prop(1, "age"), Value::Num(v) if v == 25.0)); // bob unchanged
    }

    /// `g.V()...drop()` deletes the matched vertices.
    #[test]
    fn drop_step_deletes_matched() {
        let mut st = social();
        exec(
            "g.V().hasLabel('Person').has('name', 'bob').drop()",
            &mut st,
        );
        assert!(!st.is_alive(1)); // bob
        assert_eq!(st.nodes_with_label("Person"), &[0, 2]); // alice, carol
    }

    /// Cross-language agreement: a Gremlin `addV` and the equivalent GQL `INSERT`
    /// produce the same graph.
    #[test]
    fn gremlin_and_gql_writes_agree() {
        let mut g1 = Builder::default().build();
        let mut g2 = Builder::default().build();
        exec("g.addV('P').property('name', 'z')", &mut g1);
        crate::exec::execute(
            &crate::gql::parse("INSERT (:P {name: 'z'})").unwrap(),
            &mut g2,
        )
        .unwrap();
        let probe = crate::gql::parse("MATCH (p:P) RETURN p.name AS n").unwrap();
        assert_eq!(value_bag(&run(&probe, &g1)), value_bag(&run(&probe, &g2)));
    }

    #[test]
    fn write_step_errors() {
        assert!(super::parse("g.addE('R')").is_err()); // no from/to
        assert!(super::parse("g.V().drop().count()").is_err()); // read after write
        assert!(super::parse("g.addV('P').out('R')").is_err()); // read after write
    }

    // --- addE (B6) ---

    /// `g.V(a).addE('T').to(V(b)).property(...)` creates one edge with props.
    #[test]
    fn add_edge_anchored() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]);
        let b = st.add_node(&["P"], &[]);
        exec("g.V(0).addE('R').to(V(1)).property('weight', 0.5)", &mut st);
        assert_eq!(st.out(a).len(), 1);
        assert_eq!(st.out(a)[0].nbr, b);
        let eid = st.out(a)[0].eid;
        assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 0.5));
    }

    /// `g.addE('T').from(V(a)).to(V(b))` is the unanchored form.
    #[test]
    fn add_edge_from_to() {
        let mut st = Builder::default().build();
        st.add_node(&["P"], &[]);
        st.add_node(&["P"], &[]);
        exec("g.addE('R').from(V(0)).to(V(1))", &mut st);
        assert_eq!(st.out(0).len(), 1);
        assert_eq!(st.out(0)[0].nbr, 1);
    }

    #[test]
    fn add_edge_errors() {
        // Missing `to` is a parse error (finish_add_edge requires both endpoints).
        assert!(super::parse("g.addE('R').from(V(0))").is_err());
        // Out-of-range endpoint is a runtime error.
        let mut st = Builder::default().build();
        st.add_node(&["P"], &[]);
        assert!(
            crate::exec::execute(&super::parse("g.V(0).addE('R').to(V(9))").unwrap(), &mut st)
                .is_err()
        );
    }

    #[test]
    fn local_count_is_per_element() {
        let store = social();
        // local(out('KNOWS').count()) is the per-vertex out-degree, keeping vertices
        // with zero (unlike a global count). alice→2, bob→1, carol→0, +Project has 0.
        let counts = value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').local(out('KNOWS').count())",
            &store,
        ));
        // social() has 3 persons; every one contributes a count (0 kept). KNOWS is
        // alice→bob, alice→carol, bob→carol → out-degrees carol=0, bob=1, alice=2.
        assert_eq!(
            counts,
            vec!["Num(0.0);", "Num(1.0);", "Num(2.0);"],
            "one count per person, zeros kept",
        );
        // A non-reducing hop chain is applied per element (local is transparent to it).
        assert!(super::parse("g.V().local(out('KNOWS').values('name'))").is_ok());
    }

    #[test]
    fn olap_annotate_steps_attach_a_readable_property() {
        let store = social();
        // pageRank(): every vertex gets a numeric score under the default property.
        let pr = gremlin_rows(
            "g.V().pageRank().values('gremlin.pageRankVertexProgram.pageRank')",
            &store,
        );
        assert_eq!(pr.rows.len(), 4, "one score per person");
        assert!(
            pr.rows.iter().all(|r| matches!(r[0], Value::Num(_))),
            "pageRank values are numbers: {:?}",
            pr.rows
        );
        // connectedComponent(): the whole social graph is one component → one id.
        let cc = value_bag(&gremlin_rows(
            "g.V().connectedComponent().values('gremlin.connectedComponentVertexProgram.component').dedup()",
            &store,
        ));
        assert_eq!(cc.len(), 1, "one component id (all connected): {cc:?}");
        // The component id is an external-id STRING (the root vertex), like core.
        assert!(
            cc[0].starts_with("Str("),
            "component id is a string: {cc:?}"
        );
        // A non-algo property still reads the store after an annotate (pass-through).
        assert_eq!(
            value_bag(&gremlin_rows("g.V().pageRank().values('name')", &store)),
            value_bag(&gremlin_rows("g.V().values('name')", &store)),
        );
    }

    #[test]
    fn aggregate_store_cap_side_effect_bag() {
        let store = social();
        // aggregate('x') fills a bag with the value stream; cap('x') reveals it as one
        // list. store is an alias in this eager executor.
        let agg = value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').values('name').aggregate('x').cap('x')",
            &store,
        ));
        let sto = value_bag(&gremlin_rows(
            "g.V().hasLabel('Person').values('name').store('x').cap('x')",
            &store,
        ));
        assert_eq!(agg, sto, "aggregate and store are interchangeable");
        assert_eq!(agg.len(), 1, "cap yields exactly one row (a list)");
        // aggregate() alone is a pass-through side effect (no effect on results).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V('0').out('KNOWS').aggregate('x').values('name')",
                &store,
            )),
            value_bag(&gremlin_rows(
                "g.V('0').out('KNOWS').values('name')",
                &store
            )),
        );
        // cap of an unfilled key yields a single EMPTY list (core), not an error.
        assert_eq!(
            value_bag(&gremlin_rows("g.V('1').cap('nope')", &store)),
            vec!["List([]);"],
        );
        // barrier() and identity() are pass-throughs.
        for step in ["barrier", "identity"] {
            assert_eq!(
                value_bag(&gremlin_rows(
                    &format!("g.V().hasLabel('Person').{step}().values('name')"),
                    &store,
                )),
                value_bag(&gremlin_rows(
                    "g.V().hasLabel('Person').values('name')",
                    &store
                )),
            );
        }
    }

    #[test]
    fn value_aggregates_fold_the_stream() {
        let store = social();
        // Person ages are 30, 25, 40.
        for (step, want) in [("max", 40.0), ("min", 25.0), ("sum", 95.0)] {
            let q = format!("g.V().hasLabel('Person').values('age').{step}()");
            let out = gremlin_rows(&q, &store);
            assert_eq!(out.rows.len(), 1, "{step}");
            match out.rows[0][0] {
                Value::Num(x) => assert_eq!(x, want, "{step}"),
                ref o => panic!("{step}: expected Num, got {o:?}"),
            }
        }
        // mean = 95 / 3.
        let out = gremlin_rows("g.V().hasLabel('Person').values('age').mean()", &store);
        match out.rows[0][0] {
            Value::Num(x) => assert!((x - 95.0 / 3.0).abs() < 1e-9, "mean was {x}"),
            ref o => panic!("mean: expected Num, got {o:?}"),
        }
    }

    #[test]
    fn where_filters_the_value_stream() {
        let store = social();
        // Ages > 28: 30 and 40. Both the bare and P.-prefixed spellings work.
        for q in [
            "g.V().hasLabel('Person').values('age').where(gt(28))",
            "g.V().hasLabel('Person').values('age').where(P.gt(28))",
        ] {
            assert_eq!(
                value_bag(&gremlin_rows(q, &store)),
                vec!["Num(30.0);", "Num(40.0);"],
                "{q}"
            );
        }
    }

    #[test]
    fn as_labels_and_select_projects_it() {
        let store = social();
        // Label the source as `p`, hop, then select `p` back and read its name.
        // KNOWS edges: alice->bob, alice->carol, bob->carol, so the sources are
        // alice, alice, bob.
        let q = "g.V().hasLabel('Person').as('p').out('KNOWS').select('p').values('name')";
        assert_eq!(
            value_bag(&gremlin_rows(q, &store)),
            vec!["Str(\"alice\");", "Str(\"alice\");", "Str(\"bob\");"]
        );
    }

    #[test]
    fn within_and_without_membership() {
        let store = social();
        // within is an OR-of-equals; without is its negation.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').has('name', within('alice','carol')).values('age')",
                &store
            )),
            vec!["Num(30.0);", "Num(40.0);"]
        );
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').has('name', without('alice','carol')).values('age')",
                &store
            )),
            vec!["Num(25.0);"]
        );
        // within also works in where(...) on the value stream.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').values('age').where(within(25, 40))",
                &store
            )),
            vec!["Num(25.0);", "Num(40.0);"]
        );
    }

    #[test]
    fn bare_group_count_groups_by_the_current_element() {
        let store = social();
        // KNOWS targets are bob, carol, carol → one Map {bob:1, carol:2} (Gremlin
        // groupCount is a single Map, not (key,count) rows). Bare groupCount() over
        // the name stream and the .by('name') form agree.
        let want = vec!["Map([(Str(\"bob\"), Num(1.0)), (Str(\"carol\"), Num(2.0))]);"];
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').out('KNOWS').values('name').groupCount()",
                &store
            )),
            want
        );
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').out('KNOWS').groupCount().by('name')",
                &store
            )),
            want
        );
    }

    #[test]
    fn select_errors() {
        // A select over an UNKNOWN label drops every traverser (core yields nothing),
        // rather than erroring — whether alone or inside a multi-select.
        assert!(super::parse("g.V().as('p').select('q')").is_ok());
        assert!(super::parse("g.V().as('a').out('R').as('b').select('a','z')").is_ok());
    }

    #[test]
    fn multi_select_builds_an_ordered_map() {
        let store = social();
        // bob KNOWS carol only: select('p','f') is a Map {p: bob, f: carol},
        // insertion-ordered (p then f), the values being the vertices' element maps.
        let out = gremlin_rows(
            "g.V().hasLabel('Person').has('name', 'bob').as('p').out('KNOWS').as('f') \
             .select('p', 'f')",
            &store,
        );
        assert_eq!(out.rows.len(), 1);
        // The `id` field of a vertex element-map value.
        let map_id = |v: &Value| match v {
            Value::Map(m) => m
                .iter()
                .find(|(k, _)| matches!(k, Value::Str(s) if &**s == "id"))
                .map(|(_, id)| id.clone()),
            _ => None,
        };
        match &out.rows[0][0] {
            Value::Map(pairs) => {
                assert_eq!(pairs.len(), 2);
                assert!(matches!(&pairs[0].0, Value::Str(s) if &**s == "p"));
                assert!(matches!(&pairs[1].0, Value::Str(s) if &**s == "f"));
                assert!(matches!(map_id(&pairs[0].1), Some(Value::Str(s)) if &*s == "1")); // bob
                assert!(matches!(map_id(&pairs[1].1), Some(Value::Str(s)) if &*s == "2")); // carol
            }
            o => panic!("expected a Map, got {o:?}"),
        }
    }

    #[test]
    fn has_accepts_a_bare_predicate() {
        let store = social();
        // has('age', gt(28)) (no P. prefix) now parses and agrees with P.gt(28).
        let bare = gremlin_rows(
            "g.V().hasLabel('Person').has('age', gt(28)).values('name')",
            &store,
        );
        let with_p = gremlin_rows(
            "g.V().hasLabel('Person').has('age', P.gt(28)).values('name')",
            &store,
        );
        assert_eq!(value_bag(&bare), value_bag(&with_p));
        assert_eq!(value_bag(&bare), vec!["Str(\"alice\");", "Str(\"carol\");"]);
    }
}
