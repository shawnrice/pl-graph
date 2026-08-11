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
        edge_hop: None,
        pending_repeat: None,
        path_ok: true,
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

/// A parsed `group().by(...)` modulator: a property key, the current element, or a
/// reducing `count()` traversal. Used to build the key-by and value-by of a group.
enum GroupBy {
    Key(String),
    Element,
    Count,
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
    /// Set by an edge hop (`outE`/`inE`/`bothE`) and consumed by the very next
    /// vertex-move step (`inV`/`outV`/`otherV`): `(landed node slot, hop direction)`.
    /// The edge hop leaves the current element on the EDGE; the vertex move steps
    /// back onto the endpoint the hop already landed. Cleared at the top of every
    /// `step`, so a vertex move only resolves when it IMMEDIATELY follows the edge
    /// hop — exactly Gremlin's requirement.
    edge_hop: Option<(usize, Dir)>,
    /// Set by `repeat(<hop>)` and consumed by the very next `times(n)`: the single
    /// hop `(direction, edge label)` the loop body applies. `repeat` alone is an
    /// unbounded loop (unsupported), so the body is held here until `times` closes
    /// it into a fixed-length `VarLength{min:n,max:n}`. A `repeat` not immediately
    /// followed by `times` is an error (see the guard at the top of `step`).
    pending_repeat: Option<(Dir, Option<String>)>,
    /// Whether `path()` can still be answered: true while the traversal is a pure
    /// vertex-hop chain (`V`-source + `out`/`in`/`both` + element filters), whose
    /// Gremlin path is exactly the node sequence the engine's lineage records. Any
    /// other step (edge hops, var-length, value projections, barriers, the `E`
    /// source) makes the Gremlin path step-dependent in a way the nodes-only
    /// rendering would not match, so `path()` is deferred once this is false.
    path_ok: bool,
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
                // `g.E()` seeds every live edge. `g.E('id', …)` (edges by external
                // id) is deferred — it needs an edge-liveness-checked reverse ext
                // map the store does not carry yet.
                if matches!(self.peek(), Some(Tok::Str(_))) {
                    return Err("g.E(id) (edges by external id) is not supported yet".into());
                }
                self.expect(&Tok::RParen)?;
                // An edge source makes the path start on an edge, not a node.
                self.path_ok = false;
                Plan::EdgeScan
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
        if self.pending_repeat.is_some() {
            return Err("repeat(<hop>) must be followed by times(n)".into());
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
        // An edge hop's landed endpoint is only reachable by the vertex move that
        // IMMEDIATELY follows it; consume the record here so any other step clears it.
        let prev_edge_hop = self.edge_hop.take();
        // A `repeat(body)` is only valid when closed by the very next `times(n)`.
        let prev_repeat = self.pending_repeat.take();
        if prev_repeat.is_some() && lname != "times" {
            return Err("repeat(<hop>) must be immediately followed by times(n)".into());
        }
        // Only a pure vertex-hop chain keeps `path()` answerable; every other step
        // taints it (`path()` and the element filters are path-preserving).
        if !matches!(
            lname.as_str(),
            "out"
                | "in"
                | "both"
                | "has"
                | "hasnot"
                | "hasid"
                | "and"
                | "or"
                | "not"
                | "path"
                | "simplepath"
                | "cyclicpath"
        ) {
            self.path_ok = false;
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
                // has(k, pred) — value predicate; has(k) — key EXISTENCE.
                let pred = if self.peek() == Some(&Tok::Comma) {
                    self.bump();
                    self.has_predicate(key)?
                } else {
                    Expr::PropertyExists {
                        slot: self.current,
                        key,
                    }
                };
                self.expect(&Tok::RParen)?;
                plan.filter(pred)
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
                let key = self.str_arg()?;
                self.expect(&Tok::RParen)?;
                plan.filter(Expr::Not(Box::new(Expr::PropertyExists {
                    slot: self.current,
                    key,
                })))
            }
            "and" | "or" => {
                // and(f1, f2, …) / or(f1, f2, …): each child is an element filter
                // (has/hasNot/nested and/or/not); combine their predicates and apply
                // one Filter. The `(` was consumed at the top of `step`.
                let parts = self.child_filter_list()?;
                self.expect(&Tok::RParen)?;
                let mut it = parts.into_iter();
                let first = it
                    .next()
                    .ok_or_else(|| format!("{lname}() needs at least one child traversal"))?;
                let combined = it.fold(first, |acc, e| {
                    if lname == "and" {
                        Expr::And(Box::new(acc), Box::new(e))
                    } else {
                        Expr::Or(Box::new(acc), Box::new(e))
                    }
                });
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
            "repeat" => {
                // `repeat(<hop>)` v1: the body is a SINGLE anonymous hop
                // (`out`/`in`/`both`, optionally `__`-prefixed). Held pending until
                // the following `times(n)` closes it into a fixed-length walk. The
                // LParen was already consumed at the top of `step`.
                let (dir, label) = self.repeat_body()?;
                self.expect(&Tok::RParen)?;
                self.pending_repeat = Some((dir, label));
                plan
            }
            "times" => {
                let n = self.usize_arg()?;
                self.expect(&Tok::RParen)?;
                let (dir, label) =
                    prev_repeat.ok_or("times(n) must immediately follow repeat(<hop>)")?;
                // `repeat(out('L')).times(n)` applies the hop exactly n times — a
                // WALK of length n (Gremlin allows revisiting edges, so trail=false,
                // unlike GQL var-length which is a trail). min == max == n.
                let n = u32::try_from(n).map_err(|_| "times(n): n too large")?;
                let from = self.current;
                self.current = self.slots;
                self.slots += 1;
                plan.var_length(from, dir, label.as_deref(), n, n, false)
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
                let edge_label: Option<&str> =
                    match labels.len() {
                        0 => None,
                        1 => Some(labels[0].as_str()),
                        _ => return Err(
                            "outE()/inE()/bothE() with multiple edge labels is not yet supported"
                                .into(),
                        ),
                    };
                let dir = match lname.as_str() {
                    "oute" => Dir::Out,
                    "ine" => Dir::In,
                    _ => Dir::Both,
                };
                let from = self.current;
                let edge_slot = self.slots; // W
                let node_slot = self.slots + 1; // W+1: the landed endpoint
                self.current = edge_slot;
                self.slots += 2;
                self.edge_hop = Some((node_slot, dir));
                plan.expand_edge(from, dir, edge_label)
            }
            "inv" | "outv" | "otherv" => {
                self.expect(&Tok::RParen)?;
                let (node_slot, dir) = prev_edge_hop.ok_or_else(|| {
                    format!("{name}() must immediately follow outE()/inE()/bothE()")
                })?;
                // The hop already landed the OTHER endpoint (the neighbour) in
                // `node_slot`. That endpoint is: `otherV` for any direction; `inV`
                // (the edge head/dst) only when we went OUT; `outV` (the edge
                // tail/src) only when we went IN. The origin-returning combinations
                // (`outE().outV()`, `inE().inV()`, and inV/outV after `bothE`) need the
                // pre-hop vertex, which this pointer-move does not carry — deferred.
                let ok = matches!(
                    (lname.as_str(), dir),
                    ("otherv", _) | ("inv", Dir::Out) | ("outv", Dir::In)
                );
                if !ok {
                    return Err(format!(
                        "{name}() after this edge step is not yet supported (returns the origin vertex)"
                    ));
                }
                self.current = node_slot;
                plan
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
            "id" | "label" => {
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
                let p = plan.project(vec![(
                    "path".to_string(),
                    Expr::Call {
                        name: "path_nodes".to_string(),
                        args: vec![Expr::Path],
                    },
                )]);
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
            "valuemap" => {
                // valueMap() → a PROPERTIES-only map (no id/label tokens) with scalar
                // values; valueMap('k1',…) filters keys. Lowers to the gremlin-only
                // `value_map` exec fn: element slot, then the filter keys as literals.
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
                    "valueMap".to_string(),
                    Expr::Call {
                        name: "value_map".to_string(),
                        args: fn_args,
                    },
                )]);
                self.current = 0;
                self.slots = 1;
                p
            }
            "where" => {
                // Two forms: where(<hop>) is a SEMI-JOIN — keep the element if it HAS
                // such an adjacency — and where(P) filters the current VALUE by a
                // predicate. A leading `out`/`in`/`both` (or `__`) marks the hop form.
                let is_hop = matches!(self.peek(), Some(Tok::Ident(s)) if {
                    let l = s.to_ascii_lowercase();
                    s == "__" || l == "out" || l == "in" || l == "both"
                });
                if is_hop {
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
                                "where() traversal must be a single out/in/both hop, got `{other}`"
                            ))
                        }
                    };
                    self.expect(&Tok::LParen)?;
                    // Optional single edge label (argless = any type). A multi-label
                    // hop or a multi-step chain in where() is deferred.
                    let label = if matches!(self.peek(), Some(Tok::Str(_))) {
                        let l = self.str_arg()?;
                        if self.peek() == Some(&Tok::Comma) {
                            return Err(
                                "where() hop with multiple edge labels is not yet supported".into(),
                            );
                        }
                        Some(l)
                    } else {
                        None
                    };
                    self.expect(&Tok::RParen)?; // close the hop
                    self.expect(&Tok::RParen)?; // close where(...)
                                                // Correlated existence check: does the current element have such
                                                // an edge? The body seeds `Plan::Row` (the outer row) and expands
                                                // from the current slot — the same shape GQL's EXISTS { … } builds.
                    let body = Plan::Row.expand(self.current, dir, label.as_deref());
                    plan.filter(Expr::Exists {
                        body: Box::new(body),
                        outer_width: self.slots,
                    })
                } else {
                    let pred = self.predicate_expr(Expr::Slot(self.current))?;
                    self.expect(&Tok::RParen)?;
                    plan.filter(pred)
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
                    let by = if matches!(self.peek(), Some(Tok::Str(_))) {
                        // by('key') → property access on the current element.
                        let key = self.str_arg()?;
                        Expr::Prop {
                            slot: elem_slot,
                            key,
                        }
                    } else if self.peek() == Some(&Tok::RParen) {
                        // bare by() → the current element itself.
                        Expr::Slot(elem_slot)
                    } else {
                        return Err(
                            "project().by(<nested traversal>) is not yet supported (use by('key') or by())"
                                .into(),
                        );
                    };
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
                    } else if matches!(self.peek(), Some(Tok::Ident(s)) if s.eq_ignore_ascii_case("count"))
                    {
                        // by(count()) — a reducing traversal.
                        self.ident()?;
                        self.expect(&Tok::LParen)?;
                        self.expect(&Tok::RParen)?;
                        GroupBy::Count
                    } else {
                        return Err(
                            "group().by(<nested traversal>) is only supported as by(count()) so far"
                                .into(),
                        );
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
                    },
                    Some(GroupBy::Key(k)) => Agg {
                        func: AggFn::Collect,
                        arg: Some(Expr::Prop {
                            slot: elem_slot,
                            key: k.clone(),
                        }),
                        distinct: false,
                        name: "value".into(),
                    },
                    // Default (no second by) or bare by(): fold the group's elements.
                    _ => Agg {
                        func: AggFn::Collect,
                        arg: Some(Expr::Slot(elem_slot)),
                        distinct: false,
                        name: "value".into(),
                    },
                };
                let p = plan.aggregate(vec![key_expr], vec![value_agg]);
                self.current = 0;
                self.slots = 2; // group key + value
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

    /// Parse ONE child traversal of `and`/`or`/`not` as a boolean predicate `Expr`
    /// over the current element — without applying a filter. v1 accepts only element
    /// FILTERS: `has('k')`, `has('k', pred)`, `hasNot('k')`, and nested
    /// `and`/`or`/`not` of those (an optional `__.` anonymous-traversal prefix is
    /// allowed). A navigating child (`out('X')`, …) is a semi-join needing a
    /// subquery and is deferred with an explicit error. Leaves the cursor after the
    /// child's own closing `)`.
    fn child_filter_expr(&mut self) -> Result<Expr, String> {
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
                    self.has_predicate(key)?
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
            "out" | "in" | "both" => {
                let dir = match name.as_str() {
                    "out" => Dir::Out,
                    "in" => Dir::In,
                    _ => Dir::Both,
                };
                let label = if matches!(self.peek(), Some(Tok::Str(_))) {
                    let l = self.str_arg()?;
                    if self.peek() == Some(&Tok::Comma) {
                        return Err(
                            "and/or/not hop child with multiple edge labels is not yet supported"
                                .into(),
                        );
                    }
                    Some(l)
                } else {
                    None
                };
                self.expect(&Tok::RParen)?;
                let body = Plan::Row.expand(self.current, dir, label.as_deref());
                Expr::Exists {
                    body: Box::new(body),
                    outer_width: self.slots,
                }
            }
            other => {
                return Err(format!(
                    "and/or/not child `{other}()` is not a supported filter (use \
                     has/hasNot/out/in/both/and/or/not)"
                ))
            }
        };
        Ok(expr)
    }

    /// Parse a comma-separated list of child filter traversals up to (but not
    /// consuming) the enclosing `)`.
    fn child_filter_list(&mut self) -> Result<Vec<Expr>, String> {
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

    /// Parse a `repeat(...)` body — v1 accepts only a SINGLE anonymous hop:
    /// `out('L')` / `in('L')` / `both('L')` (argless = any type), with an optional
    /// leading `__.` (the TinkerPop anonymous-traversal spawn). Returns the hop's
    /// `(direction, edge label)`. Multi-step bodies, nested repeats, filters, etc.
    /// are deferred with an explicit error. The cursor starts just after `repeat(`
    /// and stops at the body's closing `)` (left for the caller to consume).
    fn repeat_body(&mut self) -> Result<(Dir, Option<String>), String> {
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
        Ok((dir, label))
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
        // A multi-label hop child is deferred, not mis-parsed.
        assert!(super::parse("g.V().and(out('A','B'), has('k'))").is_err());
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
        // A multi-label where-hop is deferred, not mis-parsed.
        assert!(super::parse("g.V().where(out('A','B'))").is_err());
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
        // Default value-by folds the group's ELEMENTS (as ids). Names are unique here,
        // so each group holds one element: alice(0), bob(1), carol(2).
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').group().by('name')",
                &store
            )),
            vec![
                "Str(\"alice\");List([Num(0.0)]);",
                "Str(\"bob\");List([Num(1.0)]);",
                "Str(\"carol\");List([Num(2.0)]);",
            ],
        );
        // A property value-by folds that property per group.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().hasLabel('Person').group().by('name').by('age')",
                &store,
            )),
            vec![
                "Str(\"alice\");List([Num(30.0)]);",
                "Str(\"bob\");List([Num(25.0)]);",
                "Str(\"carol\");List([Num(40.0)]);",
            ],
        );
        // Non-count reducing traversals are deferred, not mis-parsed.
        assert!(super::parse("g.V().group().by('name').by(out().count())").is_err());
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
        // A key with fewer bys than keys defaults to the current element (its id here,
        // like select()): project('n','self').by('name') → {n: name, self: <id>}.
        assert_eq!(
            value_bag(&gremlin_rows(
                "g.V().has('name','bob').project('n','self').by('name')",
                &store,
            )),
            vec!["Map([(Str(\"n\"), Str(\"bob\")), (Str(\"self\"), Num(1.0))]);"],
        );
        // Nested-traversal by-modulators are deferred, not mis-parsed.
        assert!(super::parse("g.V().project('c').by(out().count())").is_err());
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
        // Deferred shapes error explicitly.
        assert!(super::parse("g.V().outE().inV().path()").is_err());
        assert!(super::parse("g.E().path()").is_err());
        assert!(super::parse("g.V().values('name').path()").is_err());
        assert!(super::parse("g.V().repeat(out('KNOWS')).times(2).path()").is_err());
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
        // Origin-returning / ambiguous combinations are deferred, not mis-answered.
        assert!(super::parse("g.V().outE().outV().values('name')").is_err());
        assert!(super::parse("g.V().inE().inV().values('name')").is_err());
        assert!(super::parse("g.V().bothE().inV().values('name')").is_err());
        // A vertex move must immediately follow an edge step.
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
        // g.E('id') (edges by external id) is deferred, not silently mis-parsed.
        assert!(super::parse("g.E('e0')").is_err());
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
        assert!(super::parse("g.E().values('x')").is_ok()); // g.E() is an all-edges source now
        assert!(super::parse("g.E('e0')").is_err()); // edges-by-id form still deferred
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
