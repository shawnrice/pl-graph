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

use crate::ir::{Agg, AggFn, CompareOp, Dir, Expr, Plan, SortKey};
use crate::value::Value;
use std::collections::HashMap;

pub fn parse(query: &str) -> Result<Plan, String> {
    let toks = lex(query)?;
    let mut p = Parser {
        toks,
        pos: 0,
        current: 0,
        slots: 1,
        labels: HashMap::new(),
    };
    p.traversal()
}

// --- lexer -------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Dot,
    LParen,
    RParen,
    Comma,
    Ident(String),
    Str(String),
    Num(f64),
}

/// Build `left = v0 OR left = v1 OR …` — the membership test `within(v0, v1, …)`
/// desugars to. An empty list is a constant FALSE (`1 = 0`), so `within()` and
/// (negated) `without()` behave sensibly.
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
            '(' => out.push(Tok::LParen),
            ')' => out.push(Tok::RParen),
            ',' => out.push(Tok::Comma),
            '\'' | '"' => {
                let quote = c;
                let mut t = String::new();
                i += 1;
                while i < b.len() && b[i] != quote {
                    t.push(b[i]);
                    i += 1;
                }
                if i >= b.len() {
                    return Err("unterminated string literal".into());
                }
                out.push(Tok::Str(t));
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

    // traversal := 'g' '.' ( 'V' '(' ')' | 'addV' '(' Label ')' ) ( '.' step )*
    fn traversal(&mut self) -> Result<Plan, String> {
        let g = self.ident()?;
        if !g.eq_ignore_ascii_case("g") {
            return Err(format!("expected `g`, got `{g}`"));
        }
        self.expect(&Tok::Dot)?;
        let head = self.ident()?;
        let mut plan = match head.to_ascii_lowercase().as_str() {
            "v" => {
                self.expect(&Tok::LParen)?;
                // g.V(<id>) is supported ONLY as the anchor of an addE (other
                // V(id) read traversals are deferred — the Scan has no by-id form).
                if matches!(self.peek(), Some(Tok::Num(_))) {
                    let from = self.u_id()?;
                    self.expect(&Tok::RParen)?;
                    self.expect(&Tok::Dot)?;
                    let step = self.ident()?;
                    if !step.eq_ignore_ascii_case("addE") {
                        return Err("g.V(id) is only supported before addE()".into());
                    }
                    self.expect(&Tok::LParen)?;
                    let etype = self.str_arg()?;
                    self.expect(&Tok::RParen)?;
                    self.finish_add_edge(Some(from), None, etype)?
                } else {
                    self.expect(&Tok::RParen)?;
                    Plan::Scan { label: None }
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
            other => return Err(format!("expected V() or addV(...), got `{other}`")),
        };
        while self.peek() == Some(&Tok::Dot) {
            self.pos += 1;
            plan = self.step(plan)?;
        }
        if self.pos != self.toks.len() {
            return Err(format!("unexpected trailing input at token {}", self.pos));
        }
        Ok(plan)
    }

    fn step(&mut self, plan: Plan) -> Result<Plan, String> {
        let name = self.ident()?;
        self.expect(&Tok::LParen)?;
        let lname = name.to_ascii_lowercase();

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
                let label = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                // Fold into the scan when it is still bare (the common form right
                // after V()); otherwise it would need a label-filter operator.
                match plan {
                    Plan::Scan { label: None } => Plan::Scan { label: Some(label) },
                    _ => return Err("hasLabel is only supported right after V()".into()),
                }
            }
            "has" => {
                let key = self.str_arg()?;
                self.expect(&Tok::Comma)?;
                let pred = self.has_predicate(key)?;
                self.expect(&Tok::RParen)?;
                plan.filter(pred)
            }
            "out" | "in" | "both" => {
                // 0 args → ANY edge type (argless out()); 1 → that type. Multi-label
                // out('A','B') is a follow-up (needs a union of hops).
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
                let edge_label: Option<&str> = match labels.len() {
                    0 => None,
                    1 => Some(labels[0].as_str()),
                    _ => {
                        return Err(
                            "out()/in()/both() with multiple edge labels is not yet supported"
                                .into(),
                        )
                    }
                };
                let dir = match name.to_ascii_lowercase().as_str() {
                    "out" => Dir::Out,
                    "in" => Dir::In,
                    _ => Dir::Both,
                };
                let from = self.current;
                self.current = self.slots;
                self.slots += 1;
                plan.expand(from, dir, edge_label)
            }
            "values" => {
                let key = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                let p = plan.project(vec![(
                    key.clone(),
                    Expr::Prop {
                        slot: self.current,
                        key,
                    },
                )]);
                // The value stream is now the single output column — subsequent
                // steps (where/min/max/…) address it at slot 0.
                self.current = 0;
                self.slots = 1;
                p
            }
            "where" => {
                // where(P.op(v)) / where(op(v)) / where(within(...)) — filter the
                // current traverser's VALUE by a predicate (typically after values).
                let pred = self.predicate_expr(Expr::Slot(self.current))?;
                self.expect(&Tok::RParen)?;
                plan.filter(pred)
            }
            "count" => {
                self.expect(&Tok::RParen)?;
                let p = plan.aggregate(
                    vec![],
                    vec![Agg {
                        func: AggFn::Count,
                        arg: None,
                        distinct: false,
                        name: "count".into(),
                    }],
                );
                self.current = 0;
                self.slots = 1;
                p
            }
            "min" | "max" | "sum" | "mean" => {
                self.expect(&Tok::RParen)?;
                // Fold the current value stream to a single scalar. `mean` is the
                // value contract's average (Avg); the rest are their namesakes.
                let func = match lname.as_str() {
                    "min" => AggFn::Min,
                    "max" => AggFn::Max,
                    "sum" => AggFn::Sum,
                    _ => AggFn::Avg,
                };
                let p = plan.aggregate(
                    vec![],
                    vec![Agg {
                        func,
                        arg: Some(Expr::Slot(self.current)),
                        distinct: false,
                        name: lname.clone(),
                    }],
                );
                self.current = 0;
                self.slots = 1;
                p
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
                    }],
                );
                self.current = 0;
                self.slots = 1;
                p
            }
            "dedup" => {
                self.expect(&Tok::RParen)?;
                plan.distinct()
            }
            "limit" => {
                let n = self.usize_arg()?;
                self.expect(&Tok::RParen)?;
                plan.order_page(vec![], None, Some(n))
            }
            "range" => {
                let lo = self.usize_arg()?;
                self.expect(&Tok::Comma)?;
                let hi = self.usize_arg()?;
                self.expect(&Tok::RParen)?;
                plan.order_page(vec![], Some(lo), Some(hi.saturating_sub(lo)))
            }
            "order" => {
                // Optional scope: order()/order(global) sort the stream;
                // order(local) sorts within the current list/map cell.
                let is_local = self.parse_scope_is_local()?;
                self.expect(&Tok::RParen)?;
                if is_local {
                    // order(local)[.by(asc|desc)] — the `by` here is a DIRECTION,
                    // not a property key (list elements sort by natural order).
                    let descending = if self.peek() == Some(&Tok::Dot)
                        && matches!(self.toks.get(self.pos + 1), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("by"))
                    {
                        self.expect(&Tok::Dot)?;
                        self.ident()?; // `by`
                        self.expect(&Tok::LParen)?;
                        let d = self.order_dir()?;
                        self.expect(&Tok::RParen)?;
                        d
                    } else {
                        false
                    };
                    plan.sort_local(descending)
                } else {
                    // order().by('k'[, asc|desc]) — stream sort by a property.
                    self.expect(&Tok::Dot)?;
                    let by = self.ident()?;
                    if !by.eq_ignore_ascii_case("by") {
                        return Err("order() must be followed by by(...)".into());
                    }
                    self.expect(&Tok::LParen)?;
                    let key = self.str_arg()?;
                    let descending = if self.peek() == Some(&Tok::Comma) {
                        self.pos += 1;
                        self.order_dir()?
                    } else {
                        false
                    };
                    self.expect(&Tok::RParen)?;
                    plan.order_page(
                        vec![SortKey {
                            expr: Expr::Prop {
                                slot: self.current,
                                key,
                            },
                            descending,
                            nulls_first: true, // Gremlin: NULLs first
                        }],
                        None,
                        None,
                    )
                }
            }
            "as" => {
                // Label the current slot; the plan is unchanged (select resolves it).
                let label = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                self.labels.insert(label, self.current);
                plan
            }
            "select" => {
                // One or more labels. A single label projects that element; two or
                // more build an insertion-ordered Map keyed by the labels.
                let mut labels = vec![self.str_arg()?];
                while self.peek() == Some(&Tok::Comma) {
                    self.pos += 1;
                    labels.push(self.str_arg()?);
                }
                self.expect(&Tok::RParen)?;
                let slot_of = |l: &str| {
                    self.labels
                        .get(l)
                        .copied()
                        .ok_or_else(|| format!("select('{l}'): no step is labelled `{l}`"))
                };
                let p = if labels.len() == 1 {
                    plan.project(vec![(labels[0].clone(), Expr::Slot(slot_of(&labels[0])?))])
                } else {
                    let entries = labels
                        .iter()
                        .map(|l| Ok((l.clone(), Expr::Slot(slot_of(l)?))))
                        .collect::<Result<Vec<_>, String>>()?;
                    plan.project(vec![("select".into(), Expr::MapLit { entries })])
                };
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
                    let key = self.str_arg()?;
                    self.expect(&Tok::RParen)?;
                    (
                        key.clone(),
                        Expr::Prop {
                            slot: self.current,
                            key,
                        },
                    )
                } else {
                    ("key".to_string(), Expr::Slot(self.current))
                };
                let p = plan.aggregate(
                    vec![key_expr],
                    vec![Agg {
                        func: AggFn::Count,
                        arg: None,
                        distinct: false,
                        name: "count".into(),
                    }],
                );
                self.current = 0;
                self.slots = 2; // group key + count
                p
            }
            other => return Err(format!("unsupported Gremlin step `{other}`")),
        };
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
            key,
        };
        self.predicate_expr(left)
    }

    /// Parse a Gremlin predicate argument and build the full comparison `Expr`
    /// against `left`. Accepts an optional `P.` prefix, then one of:
    /// `op(literal)` (`gt`, `neq`, …), `within(a, b, …)` / `without(a, b, …)`
    /// (membership, an OR-of-equals and its negation), or a bare literal
    /// (equality). Shared by `has(...)` and `where(...)`.
    fn predicate_expr(&mut self, left: Expr) -> Result<Expr, String> {
        if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("P")) {
            self.pos += 1;
            self.expect(&Tok::Dot)?;
        }
        // An identifier immediately applied to `(` is an operator; anything else is
        // a bare literal compared for equality.
        if matches!(self.peek(), Some(Tok::Ident(_)))
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
        match self.bump() {
            Some(Tok::Str(s)) => Ok(Value::Str(s.into())),
            Some(Tok::Num(n)) => Ok(Value::Num(n)),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("true") => Ok(Value::Bool(true)),
            Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("false") => Ok(Value::Bool(false)),
            other => Err(format!("expected a literal, got {other:?}")),
        }
    }

    fn usize_arg(&mut self) -> Result<usize, String> {
        match self.bump() {
            Some(Tok::Num(n)) if n >= 0.0 && n.fract() == 0.0 => Ok(n as usize),
            other => Err(format!("expected a non-negative integer, got {other:?}")),
        }
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
            (
                // groupCount after out() groups by the NEIGHBOUR's name (the
                // current element post-hop), so the GQL equivalent groups by b.
                "MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name AS who, count(*) AS c",
                "g.V().hasLabel('Person').out('KNOWS').groupCount().by('name')",
            ),
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
        assert!(super::parse("g.E().values('x')").is_err()); // only V()/addV() supported
        assert!(super::parse("g.V().frobnicate()").is_err()); // unknown step
        assert!(super::parse("g.V().has('k')").is_err()); // has needs a value
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
        // KNOWS targets are bob, carol, carol → {bob:1, carol:2}. Bare groupCount()
        // over the name stream and the .by('name') form agree.
        let want = vec!["Str(\"bob\");Num(1.0);", "Str(\"carol\");Num(2.0);"];
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
        // An unknown label errors — whether alone or inside a multi-select.
        assert!(super::parse("g.V().as('p').select('q')").is_err());
        assert!(super::parse("g.V().as('a').out('R').as('b').select('a','z')").is_err());
    }

    #[test]
    fn multi_select_builds_an_ordered_map() {
        let store = social();
        // bob KNOWS carol only: select('p','f') is a Map {p: bob(1), f: carol(2)},
        // insertion-ordered (p then f), values are the node ids.
        let out = gremlin_rows(
            "g.V().hasLabel('Person').has('name', 'bob').as('p').out('KNOWS').as('f') \
             .select('p', 'f')",
            &store,
        );
        assert_eq!(out.rows.len(), 1);
        match &out.rows[0][0] {
            Value::Map(pairs) => {
                assert_eq!(pairs.len(), 2);
                assert!(matches!(&pairs[0].0, Value::Str(s) if &**s == "p"));
                assert!(matches!(&pairs[1].0, Value::Str(s) if &**s == "f"));
                assert!(matches!(pairs[0].1, Value::Num(x) if x == 1.0)); // bob
                assert!(matches!(pairs[1].1, Value::Num(x) if x == 2.0)); // carol
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
