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

use crate::ir::{AggFn, CastTarget, CompareOp, Dir, Expr, PathPart, Plan};
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
        path_vars: HashSet::new(),
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
    /// Names bound to the current row's PATH (a `MATCH p = ANY SHORTEST …`). They
    /// resolve to `Expr::Path` rather than a slot, since the path is the lineage
    /// sidecar — one per row — not a batch column.
    path_vars: HashSet<String>,
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
        // Named-path form: `MATCH p = ANY SHORTEST (a)-[:R]->*(b)`. The path
        // variable binds to the row's path (lineage); the rest of the query
        // (WHERE/WITH/RETURN) is shared with the ordinary pattern below.
        if matches!(self.peek(), Some(Tok::Ident(_)))
            && self.toks.get(self.pos + 1) == Some(&Tok::Eq)
        {
            let (mut plan, scope, slots) = self.shortest_path_binding()?;
            self.scope = scope;
            self.slots = slots;
            if self.eat_kw("WHERE") {
                plan = plan.filter(self.expr()?);
            }
            return self.query_tail(plan);
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
        self.query_tail(plan)
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
            } else if self.eat_kw("MATCH") {
                plan = self.match_continue(plan)?;
            } else if self.eat_kw("CALL") {
                plan = self.call_inline(plan)?;
            } else {
                break;
            }
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
            return Err("expected RETURN, SET, REMOVE, WITH, or MATCH".into());
        }
        let distinct = self.eat_kw("DISTINCT");
        let items = self.return_items()?;
        let (mut plan, out_names) = apply_items(plan, &items);

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

    /// `p = ANY SHORTEST (a)-[:R]->*(b)` — the ANY-shortest named-path pattern.
    /// Binds `p` to the row's path, `a` to slot 0 and `b` to slot 1, and plans a
    /// `ShortestPath` hop. The `*`/`+` quantifier is unbounded (`{…}` shortest is
    /// deferred); an edge type is required, as elsewhere in this subset.
    fn shortest_path_binding(&mut self) -> Result<(Plan, HashMap<String, usize>, usize), String> {
        let pname = self.ident()?;
        self.expect(&Tok::Eq)?;
        if !(self.eat_kw("ANY") && self.eat_kw("SHORTEST")) {
            return Err(
                "a named path requires `ANY SHORTEST` (other path selectors \
                        are not supported)"
                    .into(),
            );
        }
        let mut scope: HashMap<String, usize> = HashMap::new();
        let (va, la) = self.node()?;
        if let Some(v) = va {
            scope.insert(v, 0);
        }
        let rel = self.rel()?;
        // The reachability quantifier: `*` or `+` (both unbounded here).
        if !(self.eat(&Tok::Star) || self.eat(&Tok::Plus)) {
            return Err(
                "`ANY SHORTEST` requires a `*` or `+` quantifier on the relationship".into(),
            );
        }
        let (vb, _lb) = self.node()?;
        if let Some(v) = vb {
            scope.insert(v, 1);
        }
        self.path_vars.insert(pname);
        let plan =
            Plan::Scan { label: la }.shortest_path(0, rel.dir, Some(rel.etype.as_str()), None);
        Ok((plan, scope, 2))
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
        let from = slots;
        slots += 1;
        let plan = self.extend_chain(Plan::Scan { label }, &mut scope, &mut slots, from)?;
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
        while matches!(self.peek(), Some(Tok::Minus | Tok::LArrow)) {
            let rel = self.rel()?;
            let quant = self.opt_quantifier()?;
            let (v2, _lbl2) = self.node()?; // a hop's landing-node label is ignored for now
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
                let node_slot = *slots;
                if let Some(v) = v2 {
                    scope.insert(v, node_slot);
                }
                *slots += 1;
                plan = plan.var_length(from, rel.dir, Some(&rel.etype), min, max, true);
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
                from = node_slot;
            } else {
                let node_slot = *slots;
                if let Some(v) = v2 {
                    scope.insert(v, node_slot);
                }
                *slots += 1;
                plan = plan.expand(from, rel.dir, Some(&rel.etype));
                from = node_slot;
            }
        }
        Ok(plan)
    }

    /// A continuing `MATCH` after `WITH`: it must start from a variable already
    /// carried into scope and extends the working table from that node (rather
    /// than scanning afresh). A fresh/disconnected subsequent pattern — one whose
    /// first node is unbound — is not supported in this subset.
    fn match_continue(&mut self, plan: Plan) -> Result<Plan, String> {
        let (var, label) = self.node()?;
        let Some(v) = var else {
            return Err("a MATCH after WITH must start from a bound variable".into());
        };
        if label.is_some() {
            return Err(format!(
                "bound variable `{v}` cannot be re-labeled in a continuing MATCH"
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
        let (var, label) = self.node()?;
        let Some(v) = var else {
            return Err("a CALL subquery pattern must start from a scope variable".into());
        };
        if label.is_some() {
            return Err(format!(
                "scope variable `{v}` cannot be re-labeled inside a CALL subquery"
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
        if items.iter().any(|it| matches!(it, RetItem::Agg(_))) {
            return Err("an aggregating RETURN inside CALL { … } is not supported".into());
        }
        let yields: Vec<(String, Expr)> = items
            .into_iter()
            .map(|it| match it {
                RetItem::Key(name, e) => (name, e),
                RetItem::Agg(_) => unreachable!("aggregates rejected above"),
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
        // Postfix `IS [NOT] NULL` — a definite null test, checked before the
        // binary comparison operators (a value is one or the other, not both).
        if self.eat_kw("IS") {
            let negated = self.eat_kw("NOT");
            if !self.eat_kw("NULL") {
                return Err("expected NULL after IS [NOT]".into());
            }
            return Ok(Expr::IsNull {
                expr: Box::new(left),
                negated,
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
                Ok(Expr::List { items })
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
                // A scalar function call `name(args…)`. (Aggregates are handled in
                // return_items, never reached here.)
                if self.peek() == Some(&Tok::LParen) {
                    return self.call(&s);
                }
                // A path variable resolves to the current row's path (lineage),
                // not a slot — there is exactly one path per row.
                if self.path_vars.contains(&s) {
                    return Ok(Expr::Path);
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

    // case := CASE (WHEN expr THEN expr)+ [ELSE expr] END   (searched form)
    // WHEN/THEN/ELSE/END are contextual keywords. The simple form
    // `CASE <e> WHEN <v> …` is deferred (would desugar to `WHEN e = v THEN …`).
    fn case_expr(&mut self) -> Result<Expr, String> {
        if !self.peek_kw("WHEN") {
            return Err("only searched CASE (CASE WHEN … END) is supported".into());
        }
        let mut branches = Vec::new();
        while self.eat_kw("WHEN") {
            let cond = self.expr()?;
            if !self.eat_kw("THEN") {
                return Err("expected THEN in CASE".into());
            }
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
        self.expect(&Tok::LBrace)?;
        let outer_width = self.slots;
        let (var, label) = self.node()?;
        let Some(v) = var else {
            return Err("EXISTS pattern must start from a bound variable".into());
        };
        if label.is_some() {
            return Err(format!(
                "bound variable `{v}` cannot be re-labeled inside EXISTS"
            ));
        }
        let Some(&from) = self.scope.get(&v) else {
            return Err(format!(
                "EXISTS must start from a bound (correlated) variable; `{v}` is not in scope"
            ));
        };
        // The body's sub-scope: the outer variables stay at their slots, slot
        // `outer_width` is reserved for the provenance column the evaluator adds,
        // and new body variables land at `outer_width + 1` onward.
        let mut sub_scope = self.scope.clone();
        let mut sub_slots = outer_width + 1;
        let body = self.extend_chain(Plan::Row, &mut sub_scope, &mut sub_slots, from)?;
        // An optional WHERE inside the braces, resolved against the body scope.
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
        Ok(Expr::Exists {
            body: Box::new(body),
            outer_width,
        })
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
        // Path accessors read the lineage sidecar, not a value — route them to
        // `Expr::PathAccess`. Their sole argument must be a path variable.
        if let Some(part) = path_part(&lname) {
            if !matches!(args.as_slice(), [Expr::Path]) {
                return Err(format!("{lname}() takes a path variable"));
            }
            return Ok(Expr::PathAccess { part });
        }
        let arity_ok = match lname.as_str() {
            // 1 arg
            "abs" | "sign" | "floor" | "ceil" | "round" | "sqrt" | "upper" | "lower" | "trim"
            | "length" | "size" | "head" | "last" | "year" | "month" | "day" | "hour"
            | "minute" | "second" | "date" | "local_time" | "datetime" | "local_datetime"
            | "zoned_time" | "zoned_datetime" | "duration" => args.len() == 1,
            // 2 args
            "starts_with" | "ends_with" | "contains" | "duration_between" => args.len() == 2,
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
}

/// An open `{n,}` upper bound is capped here (path enumeration is exponential;
/// an unbounded quantifier needs the reachability form, not enumeration).
const MAX_VARLEN: u32 = 32;

/// Build the projection (or aggregation) a `RETURN`/`WITH` item list describes,
/// attached above `plan`, and return it with the ordered output column names. An
/// aggregate anywhere makes it an `Aggregate` whose non-aggregate items are the
/// implicit GROUP BY keys; otherwise a plain `Project`. Shared by `RETURN` and
/// `WITH` so the two build identical shapes.
fn apply_items(plan: Plan, items: &[RetItem]) -> (Plan, Vec<String>) {
    let has_agg = items.iter().any(|it| matches!(it, RetItem::Agg(_)));
    let out_names: Vec<String> = items.iter().map(RetItem::name).collect();
    let plan = if has_agg {
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
            .iter()
            .map(|it| match it {
                RetItem::Key(name, e) => (name.clone(), e.clone()),
                RetItem::Agg(_) => unreachable!("no aggregates on this branch"),
            })
            .collect();
        plan.project(proj)
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
        "DATETIME" => "datetime",
        "DURATION" => "duration",
        _ => return None,
    })
}

/// Map a path-accessor function name to its `PathPart`, or `None` if it is not
/// one — the four ISO path functions (NOT `vertices`/`edges`).
fn path_part(name: &str) -> Option<PathPart> {
    Some(match name {
        "nodes" => PathPart::Nodes,
        "relationships" => PathPart::Relationships,
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
        .expand(0, Dir::Out, Some("KNOWS"))
        .aggregate(
            vec![("a".into(), Expr::Slot(0))],
            vec![crate::ir::Agg {
                func: AggFn::Count,
                arg: Some(Expr::Slot(1)),
                distinct: false,
                name: "n".into(),
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
        .expand(0, Dir::Out, Some("KNOWS"))
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
        let body = Plan::Row.expand(0, Dir::Out, Some("KNOWS"));
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
        // The correlated start must be a bound variable — a fresh scan inside
        // EXISTS is not this construct.
        let err = super::parse(
            "MATCH (p:Person) WHERE EXISTS { (z)-[:KNOWS]->(x) } RETURN p.name AS name",
        )
        .unwrap_err();
        assert!(err.contains("not in scope"), "got: {err}");
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
            body: Box::new(Plan::Row.expand(0, Dir::Out, Some("KNOWS"))),
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
        // Shortest LINK paths from `a`: b at 1 hop, c at 2, d at 3.
        let q = "MATCH p = ANY SHORTEST (x)-[:LINK]->*(y) WHERE x.name = 'a' \
                 RETURN y.name AS y, path_length(p) AS len";
        assert_eq!(
            bag(&run(&super::parse(q).unwrap(), &store)),
            vec![
                "y=Str(\"b\");len=Num(1.0);",
                "y=Str(\"c\");len=Num(2.0);",
                "y=Str(\"d\");len=Num(3.0);",
            ]
        );
        // Parse cross-check against the hand-built ShortestPath plan (all sources).
        let hand = Plan::Scan { label: None }
            .shortest_path(0, Dir::Out, Some("LINK"), None)
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
    fn named_path_requires_any_shortest() {
        let err = super::parse("MATCH p = (a)-[:LINK]->(b) RETURN p").unwrap_err();
        assert!(err.contains("ANY SHORTEST"), "got: {err}");
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
        // A component undefined for the kind is NULL (year of a time, hour of a date).
        let out2 = run(
            &super::parse(
                "MATCH (p:Person) RETURN year(TIME '01:02:03') AS y, \
                 hour(DATE '2024-01-01') AS h",
            )
            .unwrap(),
            &store,
        );
        assert!(col(&out2, 0, "y").is_null());
        assert!(col(&out2, 0, "h").is_null());
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
        assert!(
            super::parse("MATCH (p:Person) RETURN CASE p.age WHEN 30 THEN 'x' END AS y").is_err()
        ); // simple form deferred
        assert!(
            super::parse("MATCH (p:Person) RETURN CASE WHEN p.age >= 30 THEN 'x' AS y").is_err()
        ); // no END
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
        assert!(matches!(col(&out, 0, "sub"), Value::Str(x) if &*x == "lic")); // 0-based [1,4)
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
        assert!(matches!(col(&out, 0, "tail"), Value::Str(x) if &*x == "ce")); // from idx 3
        assert!(matches!(col(&out, 0, "past"), Value::Str(x) if x.is_empty())); // clamped
        assert!(
            super::parse("MATCH (p:Person) RETURN substring(p.name, -1) AS x")
                .map(|pl| run(&pl, &store).rows[0][0].is_null())
                .unwrap_or(false)
        ); // negative start → NULL (parses; evals null)
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
