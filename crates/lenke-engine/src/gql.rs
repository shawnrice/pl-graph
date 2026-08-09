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

use std::collections::HashMap;

use crate::ir::{AggFn, CompareOp, Dir, Expr, Plan};
use crate::value::Value;

/// A parsed relationship pattern `-[var:Type {props}]->`: direction, edge type,
/// an optional bound variable, and inline properties (a match filter in a
/// pattern, edge properties to write in an INSERT).
struct Rel {
    dir: Dir,
    etype: String,
    var: Option<String>,
    props: Vec<(String, Value)>,
}

/// A parsed RETURN item: a keyed expression (a grouping key / plain projection)
/// or an aggregate.
enum RetItem {
    Key(String, Expr),
    Agg(crate::ir::Agg),
}

impl RetItem {
    fn name(&self) -> String {
        match self {
            Self::Key(n, _) => n.clone(),
            Self::Agg(a) => a.name.clone(),
        }
    }
}

/// Map an aggregate function name (case-insensitive) to its `AggFn`.
fn agg_fn(name: &str) -> Option<AggFn> {
    Some(match name.to_ascii_uppercase().as_str() {
        "COUNT" => AggFn::Count,
        "SUM" => AggFn::Sum,
        "MIN" => AggFn::Min,
        "MAX" => AggFn::Max,
        "AVG" => AggFn::Avg,
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
    };
    let plan = p.query()?;
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
    RArrow, // ->
    LArrow, // <-
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
        match c {
            '(' => out.push(Tok::LParen),
            ')' => out.push(Tok::RParen),
            '[' => out.push(Tok::LBracket),
            ']' => out.push(Tok::RBracket),
            '{' => out.push(Tok::LBrace),
            '}' => out.push(Tok::RBrace),
            ':' => out.push(Tok::Colon),
            '.' => out.push(Tok::Dot),
            ',' => out.push(Tok::Comma),
            '*' => out.push(Tok::Star),
            '+' => out.push(Tok::Plus),
            '/' => out.push(Tok::Slash),
            '%' => out.push(Tok::Percent),
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
                let mut t = String::new();
                i += 1;
                while i < b.len() && b[i] != '\'' {
                    t.push(b[i]);
                    i += 1;
                }
                if i >= b.len() {
                    return Err("unterminated string literal".into());
                }
                out.push(Tok::Str(t));
            }
            _ if c.is_ascii_digit() => {
                let start = i;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                    i += 1;
                }
                let text: String = b[start..i].iter().collect();
                let n: f64 = text.parse().map_err(|_| format!("bad number `{text}`"))?;
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
        if !self.eat_kw("MATCH") {
            return Err("expected MATCH or INSERT".into());
        }
        // A comma-separated list of patterns, joined on shared variables. Each
        // pattern parses in its OWN slot space; join maps a shared variable's
        // left slot to its right slot, and the merged scope shifts the right
        // pattern's slots by the left width (the Join operator's convention).
        let (mut plan, mut scope, mut slots) = self.pattern()?;
        while self.eat(&Tok::Comma) {
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
        // Publish the merged scope for WHERE/RETURN/ORDER to resolve variables.
        self.scope = scope;
        self.slots = slots;
        if self.eat_kw("WHERE") {
            let pred = self.expr()?;
            plan = plan.filter(pred);
        }
        // Write tail: MATCH … (SET … | REMOVE …)+  — updates the bound nodes and
        // returns no rows. Otherwise the read tail (RETURN …).
        if self.peek_kw("SET") || self.peek_kw("REMOVE") {
            let ops = self.set_ops()?;
            return Ok(Plan::Update {
                input: Box::new(plan),
                ops,
            });
        }
        if !self.eat_kw("RETURN") {
            return Err("expected RETURN, SET, or REMOVE".into());
        }
        let distinct = self.eat_kw("DISTINCT");
        let items = self.return_items()?;

        // Aggregate iff any item is an aggregate; the non-aggregate items are the
        // implicit GROUP BY keys. Otherwise a plain projection.
        let has_agg = items.iter().any(|it| matches!(it, RetItem::Agg(_)));
        let out_names: Vec<String> = items.iter().map(RetItem::name).collect();
        plan = if has_agg {
            let keys = items
                .iter()
                .filter_map(|it| match it {
                    RetItem::Key(name, e) => Some((name.clone(), e.clone())),
                    RetItem::Agg(_) => None,
                })
                .collect();
            let aggs = items
                .iter()
                .filter_map(|it| match it {
                    RetItem::Agg(a) => Some(a.clone()),
                    RetItem::Key(..) => None,
                })
                .collect();
            plan.aggregate(keys, aggs)
        } else {
            let proj = items
                .into_iter()
                .map(|it| match it {
                    RetItem::Key(name, e) => (name, e),
                    RetItem::Agg(_) => unreachable!("no aggregates on this branch"),
                })
                .collect();
            plan.project(proj)
        };

        if distinct {
            plan = plan.distinct();
        }

        // ORDER BY / SKIP / LIMIT, over the OUTPUT columns (referenced by name),
        // so it composes above aggregation and DISTINCT alike.
        let keys = if self.eat_kw("ORDER") {
            if !self.eat_kw("BY") {
                return Err("expected BY after ORDER".into());
            }
            self.sort_keys(&out_names)?
        } else {
            Vec::new()
        };
        let skip = if self.eat_kw("SKIP") {
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
        Ok(plan)
    }

    // sort keys := name [ASC|DESC] ( ',' name [ASC|DESC] )*
    // Each `name` is an OUTPUT column (by alias); it maps to that output slot.
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
                self.eat_kw("ASC"); // optional, default ascending
                false
            };
            keys.push(crate::ir::SortKey {
                expr: Expr::Slot(slot),
                descending,
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
            } else {
                break;
            }
        }
        if ops.is_empty() {
            return Err("expected SET or REMOVE".into());
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
        while matches!(self.peek(), Some(Tok::Minus | Tok::LArrow)) {
            let rel = self.rel()?;
            let next = self.insert_node(nodes, var_to_idx)?;
            let (from, to) = match rel.dir {
                Dir::Out => (prev, next),
                Dir::In => (next, prev),
                Dir::Both => {
                    return Err("INSERT requires a directed relationship".into());
                }
            };
            edges.push(crate::ir::InsertEdge {
                from,
                to,
                etype: rel.etype,
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

    // A literal property value: number, string, or the keyword true/false/null.
    fn literal_value(&mut self) -> Result<Value, String> {
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
        let (var, label) = self.node()?;
        if let Some(v) = var {
            scope.insert(v, slots);
        }
        slots += 1;
        let mut plan = Plan::Scan { label };
        while matches!(self.peek(), Some(Tok::Minus | Tok::LArrow)) {
            let rel = self.rel()?;
            let quant = self.opt_quantifier()?;
            let (v2, _lbl2) = self.node()?; // a hop's landing-node label is ignored for now
            let from = slots - 1;
            // A relationship variable or inline edge properties require binding the
            // edge as a slot (edge at `slots`, node at `slots+1`).
            let bind = rel.var.is_some() || !rel.props.is_empty();
            if let Some((min, max)) = quant {
                if bind {
                    return Err(
                        "a relationship variable / edge properties on a variable-length \
                         pattern are not supported"
                            .into(),
                    );
                }
                if let Some(v) = v2 {
                    scope.insert(v, slots);
                }
                slots += 1;
                plan = plan.var_length(from, rel.dir, Some(&rel.etype), min, max, true);
            } else if bind {
                let edge_slot = slots;
                if let Some(rv) = &rel.var {
                    scope.insert(rv.clone(), edge_slot);
                }
                if let Some(v) = v2 {
                    scope.insert(v, slots + 1);
                }
                slots += 2;
                plan = plan.expand_edge(from, rel.dir, Some(&rel.etype));
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
            } else {
                if let Some(v) = v2 {
                    scope.insert(v, slots);
                }
                slots += 1;
                plan = plan.expand(from, rel.dir, Some(&rel.etype));
            }
        }
        Ok((plan, scope, slots))
    }

    /// An optional `{n}` / `{n,m}` / `{n,}` quantifier after a relationship. An
    /// open upper bound is capped at `MAX_VARLEN` hops for this subset.
    fn opt_quantifier(&mut self) -> Result<Option<(u32, u32)>, String> {
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
    fn node(&mut self) -> Result<(Option<String>, Option<String>), String> {
        self.expect(&Tok::LParen)?;
        // An identifier here is the node variable; the label (if any) follows ':'.
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
        self.expect(&Tok::RParen)?;
        Ok((var, label))
    }

    // rel := '-' '[' ':' R ']' '->'   (out)
    //      | '-' '[' ':' R ']' '-'    (both)
    //      | '<-' '[' ':' R ']' '-'   (in)
    // rel := ('-' | '<-') '[' [var] ':' Type [ '{' props '}' ] ']' ('->' | '-')
    // Captures an optional relationship VARIABLE and inline edge PROPERTIES.
    fn rel(&mut self) -> Result<Rel, String> {
        let incoming = self.eat(&Tok::LArrow);
        if !incoming {
            self.expect(&Tok::Minus)?;
        }
        self.expect(&Tok::LBracket)?;
        let var = if matches!(self.peek(), Some(Tok::Ident(_))) {
            Some(self.ident()?)
        } else {
            None
        };
        self.expect(&Tok::Colon)?;
        let etype = self.ident()?;
        let props = if matches!(self.peek(), Some(Tok::LBrace)) {
            self.props()?
        } else {
            Vec::new()
        };
        self.expect(&Tok::RBracket)?;
        let dir = if incoming {
            self.expect(&Tok::Minus)?;
            Dir::In
        } else if self.eat(&Tok::RArrow) {
            Dir::Out
        } else if self.eat(&Tok::Minus) {
            Dir::Both
        } else {
            return Err(format!("malformed relationship at token {}", self.pos));
        };
        Ok(Rel {
            dir,
            etype,
            var,
            props,
        })
    }

    // items := item ( ',' item )*
    // item  := aggregate [AS name] | expr [AS name]
    fn return_items(&mut self) -> Result<Vec<RetItem>, String> {
        let mut items = Vec::new();
        loop {
            let idx = items.len();
            let item = if let Some(func) = self.peek_agg() {
                let (agg_arg, distinct) = self.aggregate_call()?;
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
                })
            } else {
                let e = self.expr()?;
                let name = if self.eat_kw("AS") {
                    self.ident()?
                } else {
                    default_name(&e, idx)
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
    fn aggregate_call(&mut self) -> Result<(Option<Expr>, bool), String> {
        self.pos += 1; // the aggregate name (already validated by peek_agg)
        self.expect(&Tok::LParen)?;
        if self.eat(&Tok::Star) {
            self.expect(&Tok::RParen)?;
            return Ok((None, false));
        }
        let distinct = self.eat_kw("DISTINCT");
        let arg = self.expr()?;
        self.expect(&Tok::RParen)?;
        Ok((Some(arg), distinct))
    }

    // Expression precedence: OR < AND < NOT < comparison < primary.
    fn expr(&mut self) -> Result<Expr, String> {
        self.or_expr()
    }

    fn or_expr(&mut self) -> Result<Expr, String> {
        let mut left = self.and_expr()?;
        while self.eat_kw("OR") {
            let right = self.and_expr()?;
            left = Expr::Or(Box::new(left), Box::new(right));
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
        // Comparison operands are arithmetic expressions (arith binds tighter).
        let left = self.add_expr()?;
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
        let right = self.add_expr()?;
        Ok(Expr::Compare {
            op,
            left: Box::new(left),
            right: Box::new(right),
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
        match self.peek().cloned() {
            Some(Tok::LParen) => {
                self.pos += 1;
                let e = self.expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
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
                // A scalar function call `name(args…)`. (Aggregates are handled in
                // return_items, never reached here.)
                if self.peek() == Some(&Tok::LParen) {
                    return self.call(&s);
                }
                let slot = *self
                    .scope
                    .get(&s)
                    .ok_or_else(|| format!("unknown variable `{s}`"))?;
                if self.eat(&Tok::Dot) {
                    let key = self.ident()?;
                    Ok(Expr::Prop { slot, key })
                } else {
                    Ok(Expr::Slot(slot))
                }
            }
            other => Err(format!("expected an expression, got {other:?}")),
        }
    }

    // call := name '(' [ expr (',' expr)* ] ')'  — a scalar function.
    // Validates the name and arity here so `eval` only sees well-formed calls.
    fn call(&mut self, name: &str) -> Result<Expr, String> {
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
        let lname = name.to_ascii_lowercase();
        match lname.as_str() {
            "abs" | "sign" | "floor" | "ceil" | "round" | "sqrt" => {
                if args.len() != 1 {
                    return Err(format!("{lname}() takes exactly one argument"));
                }
            }
            "coalesce" => {
                if args.is_empty() {
                    return Err("coalesce() takes at least one argument".into());
                }
            }
            _ => return Err(format!("unknown function `{name}`")),
        }
        Ok(Expr::Call { name: lname, args })
    }
}

/// An open `{n,}` upper bound is capped here (path enumeration is exponential;
/// an unbounded quantifier needs the reachability form, not enumeration).
const MAX_VARLEN: u32 = 32;

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
        .expand(0, Dir::Out, Some("KNOWS"))
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
        .expand(0, Dir::Out, Some("KNOWS"))
        .expand(1, Dir::Out, Some("KNOWS"))
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
        .expand(0, Dir::In, Some("KNOWS"))
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
        assert!(super::parse("RETURN 1").is_err()); // no MATCH
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
                },
                Agg {
                    func: AggFn::Avg,
                    arg: Some(Expr::Prop {
                        slot: 0,
                        key: "age".into(),
                    }),
                    distinct: false,
                    name: "a".into(),
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
        .expand(0, Dir::Out, Some("KNOWS"));
        let right = Plan::Scan {
            label: Some("Person".into()),
        }
        .expand(0, Dir::Out, Some("WORKS_ON"));
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
        use crate::ir::{Dir, Expr, Plan};
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
        .var_length(0, Dir::Out, Some("KNOWS"), 1, 2, true)
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
        // sqrt of a negative is NULL (non-finite result).
        let out = run(
            &super::parse("MATCH (p:Person) WHERE p.name='alice' RETURN sqrt(0 - p.age) AS s")
                .unwrap(),
            &store,
        );
        assert!(col(&out, 0, "s").is_null());
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
        .expand_edge(0, crate::ir::Dir::Out, Some("R"))
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
}
