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

use crate::ir::{CompareOp, Dir, Expr, Plan};
use crate::value::Value;

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
    Colon,
    Dot,
    Comma,
    Minus,
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
            ':' => out.push(Tok::Colon),
            '.' => out.push(Tok::Dot),
            ',' => out.push(Tok::Comma),
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

    fn bind(&mut self, var: Option<String>) -> usize {
        let slot = self.slots;
        self.slots += 1;
        if let Some(v) = var {
            self.scope.insert(v, slot);
        }
        slot
    }

    // query := MATCH pattern [WHERE expr] RETURN items
    fn query(&mut self) -> Result<Plan, String> {
        if !self.eat_kw("MATCH") {
            return Err("expected MATCH".into());
        }
        let mut plan = self.pattern()?;
        if self.eat_kw("WHERE") {
            let pred = self.expr()?;
            plan = plan.filter(pred);
        }
        if !self.eat_kw("RETURN") {
            return Err("expected RETURN".into());
        }
        let items = self.return_items()?;
        Ok(plan.project(items))
    }

    // pattern := node ( rel node )*
    fn pattern(&mut self) -> Result<Plan, String> {
        let (var, label) = self.node()?;
        let slot = self.bind(var);
        debug_assert_eq!(slot, 0, "the first node is slot 0");
        let mut plan = Plan::Scan { label };
        while matches!(self.peek(), Some(Tok::Minus | Tok::LArrow)) {
            let (dir, edge) = self.rel()?;
            let (v2, _lbl2) = self.node()?; // a hop's landing-node label is ignored for now
            let from = slot_before_hop(&self.scope, self.slots);
            self.bind(v2);
            plan = plan.expand(from, dir, Some(&edge));
        }
        Ok(plan)
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
    fn rel(&mut self) -> Result<(Dir, String), String> {
        let incoming = self.eat(&Tok::LArrow);
        if !incoming {
            self.expect(&Tok::Minus)?;
        }
        self.expect(&Tok::LBracket)?;
        self.expect(&Tok::Colon)?;
        let edge = self.ident()?;
        self.expect(&Tok::RBracket)?;
        if incoming {
            self.expect(&Tok::Minus)?;
            Ok((Dir::In, edge))
        } else if self.eat(&Tok::RArrow) {
            Ok((Dir::Out, edge))
        } else if self.eat(&Tok::Minus) {
            Ok((Dir::Both, edge))
        } else {
            Err(format!("malformed relationship at token {}", self.pos))
        }
    }

    // items := item ( ',' item )*   ;   item := expr [AS name]
    fn return_items(&mut self) -> Result<Vec<(String, Expr)>, String> {
        let mut items = Vec::new();
        loop {
            let e = self.expr()?;
            let name = if self.eat_kw("AS") {
                self.ident()?
            } else {
                default_name(&e, items.len())
            };
            items.push((name, e));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(items)
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
        let left = self.primary()?;
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
        let right = self.primary()?;
        Ok(Expr::Compare {
            op,
            left: Box::new(left),
            right: Box::new(right),
        })
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
}

/// The slot the current hop expands FROM: the most recently bound node, i.e. the
/// slot just before the one this hop will bind. With `slots` already counting the
/// nodes bound so far and the new node not yet bound, that is `slots - 1`.
fn slot_before_hop(_scope: &HashMap<String, usize>, slots: usize) -> usize {
    slots - 1
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
}
