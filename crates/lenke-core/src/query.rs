//! A GQL subset large enough to compare past `count`: MATCH a linear pattern,
//! an AND-chain of WHERE comparisons, and RETURN with aggregates
//! (count/sum/avg/min/max), implicit GROUP BY, DISTINCT, ORDER BY, and LIMIT.
//!
//! Every result reduces to a `(count, sum, checksum)` fingerprint both engines
//! compute identically: row count, the sum of numeric cells (compared with an
//! epsilon, since float summation order differs), and an FNV fold over the
//! *exact* cells — strings and integer-valued numbers only (floats are left to
//! `sum`). It's order-sensitive iff the query has ORDER BY.

use crate::graph::{Column, Dict, Graph, Value};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone)]
enum Lit {
    Num(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone)]
struct NodeP {
    var: Option<String>,
    label: Option<String>,
}

#[derive(Debug, Clone)]
struct RelP {
    etype: Option<String>,
}

#[derive(Debug, Clone)]
struct Pred {
    var: String,
    key: String,
    op: Op,
    lit: Lit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Debug, Clone)]
enum RetItem {
    /// `a.key` (or bare `a`, key=None → the vertex id string)
    Plain { var: String, key: Option<String> },
    /// `count(*)` / `sum(a.key)` / …
    Agg {
        func: AggFunc,
        var: Option<String>,
        key: Option<String>,
    },
}

#[derive(Debug, Clone)]
struct OrderBy {
    /// index into the RETURN items
    col: usize,
    desc: bool,
}

#[derive(Debug, Clone)]
pub struct Query {
    nodes: Vec<NodeP>,
    rels: Vec<RelP>,
    preds: Vec<Pred>,
    distinct: bool,
    items: Vec<RetItem>,
    order: Option<OrderBy>,
    limit: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryResult {
    pub count: u64,
    pub sum: f64,
    pub checksum: u64,
}

/// A materialized result: column names plus a **columnar** cell buffer — a
/// single flat row-major `Vec<Value>` (cell `(i, j)` at `i*ncols + j`) instead
/// of a `Vec` per row, so building an N-row result is one amortized allocation,
/// not N small ones. Cells are the core graph `Value` model so the rowset
/// round-trips losslessly to JSON for the FFI / wasm boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct RowSet {
    pub cols: Vec<String>,
    /// Flat row-major cells; `nrows * cols.len()` long.
    pub data: Vec<Value>,
    pub nrows: usize,
}

impl RowSet {
    pub fn new(cols: Vec<String>) -> Self {
        Self {
            cols,
            data: Vec::new(),
            nrows: 0,
        }
    }
    /// Same, with room for `nrows` rows already reserved.
    ///
    /// The flat buffer is the point of this type — one amortized allocation
    /// rather than one per row — but "amortized" still means a doubling growth
    /// that copies the whole buffer log(n) times, and a `Value` is 40 bytes. A
    /// 20k-row projection was moving 800KB through several of those reallocs on
    /// the way to a size it knew before it started.
    pub fn with_capacity(cols: Vec<String>, nrows: usize) -> Self {
        let n = cols.len();

        Self {
            cols,
            data: Vec::with_capacity(nrows.saturating_mul(n)),
            nrows: 0,
        }
    }
    pub fn ncols(&self) -> usize {
        self.cols.len()
    }
    /// Row `i` as a slice of its cells.
    pub fn row(&self, i: usize) -> &[Value] {
        let c = self.cols.len();
        &self.data[i * c..i * c + c]
    }
    /// Iterate rows as cell slices.
    pub fn rows(&self) -> impl Iterator<Item = &[Value]> {
        let c = self.cols.len().max(1); // chunks(0) panics; empty-col → no rows
        self.data.chunks(c).take(self.nrows)
    }
    /// Append a row (its cells; must be exactly `ncols`).
    pub fn push_row(&mut self, cells: impl IntoIterator<Item = Value>) {
        self.data.extend(cells);
        self.nrows += 1;
        debug_assert_eq!(self.data.len(), self.nrows * self.cols.len());
    }
    /// Drop the most recently pushed row (used to undo a DISTINCT duplicate).
    pub fn pop_row(&mut self) {
        self.data.truncate(self.data.len() - self.cols.len());
        self.nrows -= 1;
    }
    /// Apply SKIP/LIMIT in place over the flat buffer.
    pub fn apply_skip_limit(&mut self, skip: usize, limit: Option<usize>) {
        let c = self.cols.len();
        let skip = skip.min(self.nrows);
        if skip > 0 {
            self.data.drain(0..skip * c);
            self.nrows -= skip;
        }
        if let Some(n) = limit {
            if self.nrows > n {
                self.data.truncate(n * c);
                self.nrows = n;
            }
        }
    }

    /// Serialize to a compact `{"columns":[...],"rows":[[...]]}` JSON document —
    /// the carrier for both bun:ffi and the wasm binding, where a single buffer
    /// crossing beats marshalling cell-by-cell. Hand-rolled (no `serde_json`) so
    /// the core query path carries no JSON dependency — that's what lets a
    /// minimal frontend wasm build (GQL only) drop `serde_json` entirely.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(self.cols.len() * 16 + self.nrows * 32);
        out.push_str("{\"columns\":[");
        for (i, c) in self.cols.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            crate::jsonfmt::push_json_str(&mut out, c);
        }
        out.push_str("],\"rows\":[");
        for (ri, r) in self.rows().enumerate() {
            if ri > 0 {
                out.push(',');
            }
            out.push('[');
            for (ci, cell) in r.iter().enumerate() {
                if ci > 0 {
                    out.push(',');
                }
                push_json_value(&mut out, cell);
            }
            out.push(']');
        }
        out.push_str("]}");
        out
    }
}

/// Emit a core [`Value`] as JSON. Non-finite numbers (NaN/±Inf) have no JSON
/// form, so they serialize as null — matching how a JS engine would surface an
/// absent/undefined numeric cell.
fn push_json_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Num(x) => {
            if x.is_finite() {
                // ECMA-262 `Number::toString` (exponential for very large/small
                // magnitudes, -0 → "0") so every numeric result cell matches JS
                // `JSON.stringify` / the TS engine — Rust's Display never does that.
                out.push_str(&crate::jsonfmt::js_number(*x));
            } else {
                out.push_str("null");
            }
        }
        Value::Str(s) => crate::jsonfmt::push_json_str(out, s),
        Value::Temporal(t) => out.push_str(&t.json_tagged()),
        Value::List(items) => {
            out.push('[');
            for (i, e) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_json_value(out, e);
            }
            out.push(']');
        }
        Value::Map(pairs) => {
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                crate::jsonfmt::push_json_str(out, k);
                out.push(':');
                push_json_value(out, val);
            }
            out.push('}');
        }
    }
}

// ---------- a projected cell ----------

#[derive(Clone, Copy)]
enum Cell {
    Num(f64),
    Str(u32), // interned in the projection-local dict
    Null,
}

// ---------- tokenizer ----------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    Sym(String),
}

fn tokenize(s: &str) -> Result<Vec<Tok>, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '\'' || c == '"' {
            let q = c;
            i += 1;
            let start = i;
            while i < b.len() && b[i] as char != q {
                i += 1;
            }
            out.push(Tok::Str(s[start..i].to_string()));
            i += 1;
        } else if c.is_ascii_digit()
            || (c == '-' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit())
        {
            let start = i;
            i += 1;
            while i < b.len() && {
                let d = b[i] as char;
                d.is_ascii_digit() || d == '.'
            } {
                i += 1;
            }
            out.push(Tok::Num(s[start..i].parse().map_err(|_| "bad number")?));
        } else if c.is_alphabetic() || c == '_' || c == '*' {
            if c == '*' {
                out.push(Tok::Sym("*".into()));
                i += 1;
                continue;
            }
            let start = i;
            while i < b.len() && {
                let d = b[i] as char;
                d.is_alphanumeric() || d == '_'
            } {
                i += 1;
            }
            out.push(Tok::Ident(s[start..i].to_string()));
        } else {
            let two = if i + 1 < b.len() { &s[i..i + 2] } else { "" };
            let three = if i + 2 < b.len() { &s[i..i + 3] } else { "" };
            if three == "]->" {
                out.push(Tok::Sym("]->".into()));
                i += 3;
            } else if two == "-[" {
                out.push(Tok::Sym("-[".into()));
                i += 2;
            } else if two == "<=" || two == ">=" || two == "<>" {
                out.push(Tok::Sym(two.into()));
                i += 2;
            } else {
                out.push(Tok::Sym(c.to_string()));
                i += 1;
            }
        }
    }
    Ok(out)
}

// ---------- parser ----------

struct P {
    toks: Vec<Tok>,
    i: usize,
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i)
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.i).cloned();
        self.i += 1;
        t
    }
    fn eat_sym(&mut self, s: &str) -> Result<(), String> {
        match self.next() {
            Some(Tok::Sym(x)) if x == s => Ok(()),
            o => Err(format!("expected '{s}', got {o:?}")),
        }
    }
    fn eat_kw(&mut self, kw: &str) -> Result<(), String> {
        match self.next() {
            Some(Tok::Ident(x)) if x.eq_ignore_ascii_case(kw) => Ok(()),
            o => Err(format!("expected '{kw}', got {o:?}")),
        }
    }
    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Ident(x)) if x.eq_ignore_ascii_case(kw))
    }
    fn is_sym(&self, s: &str) -> bool {
        matches!(self.peek(), Some(Tok::Sym(x)) if x == s)
    }
    fn ident(&mut self) -> Result<String, String> {
        match self.next() {
            Some(Tok::Ident(x)) => Ok(x),
            o => Err(format!("expected identifier, got {o:?}")),
        }
    }

    fn node(&mut self) -> Result<NodeP, String> {
        self.eat_sym("(")?;
        let mut var = None;
        let mut label = None;
        if let Some(Tok::Ident(v)) = self.peek() {
            var = Some(v.clone());
            self.next();
        }
        if self.is_sym(":") {
            self.next();
            label = Some(self.ident()?);
        }
        self.eat_sym(")")?;
        Ok(NodeP { var, label })
    }

    fn rel(&mut self) -> Result<RelP, String> {
        self.eat_sym("-[")?;
        let mut etype = None;
        if self.is_sym(":") {
            self.next();
            etype = Some(self.ident()?);
        }
        self.eat_sym("]->")?;
        Ok(RelP { etype })
    }

    fn op(&mut self) -> Result<Op, String> {
        match self.next() {
            Some(Tok::Sym(s)) => match s.as_str() {
                "=" => Ok(Op::Eq),
                "<>" => Ok(Op::Ne),
                "<" => Ok(Op::Lt),
                ">" => Ok(Op::Gt),
                "<=" => Ok(Op::Le),
                ">=" => Ok(Op::Ge),
                _ => Err(format!("bad operator {s}")),
            },
            o => Err(format!("expected operator, got {o:?}")),
        }
    }

    fn lit(&mut self) -> Result<Lit, String> {
        match self.next() {
            Some(Tok::Num(n)) => Ok(Lit::Num(n)),
            Some(Tok::Str(s)) => Ok(Lit::Str(s)),
            Some(Tok::Ident(x)) if x.eq_ignore_ascii_case("true") => Ok(Lit::Bool(true)),
            Some(Tok::Ident(x)) if x.eq_ignore_ascii_case("false") => Ok(Lit::Bool(false)),
            o => Err(format!("expected literal, got {o:?}")),
        }
    }

    fn pred(&mut self) -> Result<Pred, String> {
        let var = self.ident()?;
        self.eat_sym(".")?;
        let key = self.ident()?;
        let op = self.op()?;
        let lit = self.lit()?;
        Ok(Pred { var, key, op, lit })
    }

    fn agg_func(name: &str) -> Option<AggFunc> {
        match name.to_ascii_lowercase().as_str() {
            "count" => Some(AggFunc::Count),
            "sum" => Some(AggFunc::Sum),
            "avg" => Some(AggFunc::Avg),
            "min" => Some(AggFunc::Min),
            "max" => Some(AggFunc::Max),
            _ => None,
        }
    }

    fn ret_item(&mut self) -> Result<RetItem, String> {
        // aggregate?  name '(' ('*' | var '.' key) ')'
        if let Some(Tok::Ident(name)) = self.peek().cloned() {
            if let Some(func) = Self::agg_func(&name) {
                // lookahead for '('
                if matches!(self.toks.get(self.i + 1), Some(Tok::Sym(s)) if s == "(") {
                    self.next(); // name
                    self.eat_sym("(")?;
                    if self.is_sym("*") {
                        self.next();
                        self.eat_sym(")")?;
                        return Ok(RetItem::Agg {
                            func,
                            var: None,
                            key: None,
                        });
                    }
                    let var = self.ident()?;
                    self.eat_sym(".")?;
                    let key = self.ident()?;
                    self.eat_sym(")")?;
                    return Ok(RetItem::Agg {
                        func,
                        var: Some(var),
                        key: Some(key),
                    });
                }
            }
        }
        // plain: var ('.' key)?
        let var = self.ident()?;
        let mut key = None;
        if self.is_sym(".") {
            self.next();
            key = Some(self.ident()?);
        }
        Ok(RetItem::Plain { var, key })
    }

    fn parse(&mut self) -> Result<Query, String> {
        self.eat_kw("MATCH")?;
        let mut nodes = vec![self.node()?];
        let mut rels = Vec::new();
        while self.is_sym("-[") {
            rels.push(self.rel()?);
            nodes.push(self.node()?);
        }
        let mut preds = Vec::new();
        if self.is_kw("WHERE") {
            self.next();
            preds.push(self.pred()?);
            while self.is_kw("AND") {
                self.next();
                preds.push(self.pred()?);
            }
        }
        self.eat_kw("RETURN")?;
        let distinct = if self.is_kw("DISTINCT") {
            self.next();
            true
        } else {
            false
        };
        let mut items = vec![self.ret_item()?];
        while self.is_sym(",") {
            self.next();
            items.push(self.ret_item()?);
        }
        let mut order = None;
        if self.is_kw("ORDER") {
            self.next();
            self.eat_kw("BY")?;
            // ORDER BY var.key — match it to a RETURN item position.
            let var = self.ident()?;
            let mut key = None;
            if self.is_sym(".") {
                self.next();
                key = Some(self.ident()?);
            }
            let desc = if self.is_kw("DESC") {
                self.next();
                true
            } else {
                if self.is_kw("ASC") {
                    self.next();
                }
                false
            };
            let col = items
                .iter()
                .position(|it| match it {
                    RetItem::Plain { var: v, key: k } => *v == var && *k == key,
                    _ => false,
                })
                .unwrap_or(0);
            order = Some(OrderBy { col, desc });
        }
        let mut limit = None;
        if self.is_kw("LIMIT") {
            self.next();
            match self.next() {
                Some(Tok::Num(n)) => limit = Some(n as usize),
                o => return Err(format!("expected LIMIT number, got {o:?}")),
            }
        }
        Ok(Query {
            nodes,
            rels,
            preds,
            distinct,
            items,
            order,
            limit,
        })
    }
}

pub fn parse(s: &str) -> Result<Query, String> {
    let mut p = P {
        toks: tokenize(s)?,
        i: 0,
    };
    p.parse()
}

// ---------- executor ----------

fn col_of<'a>(g: &'a Graph, key: &str) -> Option<&'a Column> {
    g.props.col(key)
}

fn vertex_num(g: &Graph, vi: u32, key: &str) -> Option<f64> {
    match col_of(g, key)? {
        Column::Num { data, present } if present.get(vi as usize) => Some(data[vi as usize]),
        Column::Bool { data, present } if present.get(vi as usize) => {
            Some(if data[vi as usize] { 1.0 } else { 0.0 })
        }
        _ => None,
    }
}

fn pred_holds(g: &Graph, vi: u32, p: &Pred) -> bool {
    match &p.lit {
        Lit::Num(t) => vertex_num(g, vi, &p.key).is_some_and(|x| match p.op {
            Op::Eq => x == *t,
            Op::Ne => x != *t,
            Op::Lt => x < *t,
            Op::Gt => x > *t,
            Op::Le => x <= *t,
            Op::Ge => x >= *t,
        }),
        Lit::Bool(b) => {
            matches!(col_of(g, &p.key), Some(Column::Bool { data, present }) if present.get(vi as usize)
                && (match p.op { Op::Eq => data[vi as usize] == *b, Op::Ne => data[vi as usize] != *b, _ => false }))
        }
        Lit::Str(s) => {
            let want = g.strs.get(s);
            match col_of(g, &p.key) {
                Some(Column::Str { data, present }) if present.get(vi as usize) => {
                    let got = Some(data[vi as usize]);
                    match p.op {
                        Op::Eq => want == got,
                        Op::Ne => want != got,
                        _ => false,
                    }
                }
                _ => false,
            }
        }
    }
}

fn seeds<'a>(g: &'a Graph, node: &NodeP) -> Box<dyn Iterator<Item = u32> + 'a> {
    match node.label.as_ref().and_then(|l| g.labels.get(l)) {
        Some(lid) => Box::new(g.vertices_with_label(lid).iter().copied()),
        None if node.label.is_some() => Box::new(std::iter::empty()),
        None => Box::new(0..g.n as u32),
    }
}

/// Project one binding cell for a plain RETURN item, interning strings locally
/// so the checksum hashes the *text* (matching the TS side).
fn project_cell(
    g: &Graph,
    binding: &[u32],
    pos: usize,
    key: &Option<String>,
    pdict: &mut Dict,
) -> Cell {
    let vi = binding[pos];
    match key {
        None => Cell::Str(pdict.intern(g.vid.text(vi))),
        Some(k) => match col_of(g, k) {
            Some(Column::Num { data, present }) if present.get(vi as usize) => {
                Cell::Num(data[vi as usize])
            }
            Some(Column::Bool { data, present }) if present.get(vi as usize) => {
                Cell::Num(if data[vi as usize] { 1.0 } else { 0.0 })
            }
            Some(Column::Str { data, present }) if present.get(vi as usize) => {
                Cell::Str(pdict.intern(g.strs.text(data[vi as usize])))
            }
            _ => Cell::Null,
        },
    }
}

/// Project one binding cell as a real `Value` (vs. `project_cell`'s lossy
/// benchmark `Cell`). A bare var yields the vertex's external id string; a
/// `var.key` yields the typed property, including `Mixed`/`List` columns the
/// fingerprint path can't represent. Absent → `Value::Null`.
fn project_value(g: &Graph, binding: &[u32], pos: usize, key: &Option<String>) -> Value {
    let vi = binding[pos];
    match key {
        None => Value::Str(g.vid.arc(vi)),
        Some(k) => match col_of(g, k) {
            Some(Column::Num { data, present }) if present.get(vi as usize) => {
                Value::Num(data[vi as usize])
            }
            Some(Column::Bool { data, present }) if present.get(vi as usize) => {
                Value::Bool(data[vi as usize])
            }
            Some(Column::Str { data, present }) if present.get(vi as usize) => {
                Value::Str(g.strs.arc(data[vi as usize]))
            }
            Some(Column::Mixed { data }) => data[vi as usize].clone().unwrap_or(Value::Null),
            _ => Value::Null,
        },
    }
}

fn func_name(f: AggFunc) -> &'static str {
    match f {
        AggFunc::Count => "count",
        AggFunc::Sum => "sum",
        AggFunc::Avg => "avg",
        AggFunc::Min => "min",
        AggFunc::Max => "max",
    }
}

/// The default column name for a RETURN item — its source text, since there's
/// no AS-alias in this subset yet (`a`, `a.age`, `count(*)`, `sum(a.age)`).
fn item_name(it: &RetItem) -> String {
    match it {
        RetItem::Plain { var, key: None } => var.clone(),
        RetItem::Plain { var, key: Some(k) } => format!("{var}.{k}"),
        RetItem::Agg { func, var, key } => {
            let inner = match (var, key) {
                (Some(v), Some(k)) => format!("{v}.{k}"),
                _ => "*".to_string(),
            };
            format!("{}({})", func_name(*func), inner)
        }
    }
}

/// A canonical, hashable key for a `Value` — used to group/dedup rows. Floats
/// key by their bit pattern so distinct NaNs and ±0 stay distinguishable, and a
/// type tag prevents cross-type collisions (e.g. `1` the number vs `"1"`).
fn value_key(v: &Value, out: &mut String) {
    use std::fmt::Write;
    match v {
        Value::Null => out.push('0'),
        Value::Bool(b) => {
            let _ = write!(out, "b{}", *b as u8);
        }
        Value::Num(x) => {
            // `-0.0` and `0.0` are ONE key, matching the main engine — which
            // normalizes here for the reason recorded on `group_num_bits`:
            // grouping agrees with equality everywhere else, and two groups whose
            // rendered values are both `0` is a distinction no result can show.
            //
            // This copy kept the raw bits, so `RETURN DISTINCT u.z` over a `0.0`
            // and a `-0.0` gave two rows here and one there. That matters
            // precisely because this engine exists to be COMPARED against the
            // other one.
            let _ = write!(out, "n{:016x}", crate::value::group_key_bits(*x));
        }
        Value::Str(s) => {
            let _ = write!(out, "s{s}");
        }
        Value::Temporal(t) => {
            let _ = write!(out, "t{}{}", t.tag(), t.format());
        }
        Value::List(items) => {
            out.push('[');
            for it in items {
                value_key(it, out);
                out.push(',');
            }
            out.push(']');
        }
        Value::Map(pairs) => {
            out.push('{');
            for (k, val) in pairs {
                let _ = write!(out, "{k}=");
                value_key(val, out);
                out.push(',');
            }
            out.push('}');
        }
    }
}

fn row_key(cells: &[Value]) -> String {
    let mut s = String::new();
    for c in cells {
        value_key(c, &mut s);
        s.push('\u{1}');
    }
    s
}

/// Total order over `Value` for ORDER BY: Null sorts first, then by type, then
/// by value (numbers numerically, strings lexicographically). NaN sorts after
/// finite numbers so it can't poison the comparator.
fn cmp_value(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    fn rank(v: &Value) -> u8 {
        match v {
            Value::Null => 0,
            Value::Bool(_) => 1,
            Value::Num(_) => 2,
            Value::Str(_) => 3,
            Value::Temporal(_) => 4,
            Value::List(_) => 5,
            Value::Map(_) => 6,
        }
    }
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Temporal(x), Value::Temporal(y)) => x.cmp_total(y),
        (Value::Num(x), Value::Num(y)) => match x.partial_cmp(y) {
            Some(o) => o,
            None => x.is_nan().cmp(&y.is_nan()), // NaN last, NaN==NaN
        },
        (Value::Str(x), Value::Str(y)) => x.cmp(y),
        (Value::List(x), Value::List(y)) => {
            for (xi, yi) in x.iter().zip(y) {
                let o = cmp_value(xi, yi);
                if o != Ordering::Equal {
                    return o;
                }
            }
            x.len().cmp(&y.len())
        }
        (Value::Map(x), Value::Map(y)) => {
            for ((xk, xv), (yk, yv)) in x.iter().zip(y) {
                let o = xk.cmp(yk).then_with(|| cmp_value(xv, yv));
                if o != Ordering::Equal {
                    return o;
                }
            }
            x.len().cmp(&y.len())
        }
        _ => rank(a).cmp(&rank(b)),
    }
}

struct Acc {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}
impl Acc {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    fn step(&mut self, x: Option<f64>) {
        self.count += 1;
        if let Some(x) = x {
            self.sum += x;
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
    }
    fn finalize(&self, func: AggFunc) -> f64 {
        match func {
            AggFunc::Count => self.count as f64,
            AggFunc::Sum => self.sum,
            AggFunc::Avg => {
                if self.count == 0 {
                    0.0
                } else {
                    self.sum / self.count as f64
                }
            }
            AggFunc::Min => {
                if self.min.is_finite() {
                    self.min
                } else {
                    0.0
                }
            }
            AggFunc::Max => {
                if self.max.is_finite() {
                    self.max
                } else {
                    0.0
                }
            }
        }
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_bytes(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// Hash a row's *exact* cells (strings + integer-valued numbers). Non-integer
/// floats are skipped — their magnitude is verified via `sum` with an epsilon.
fn hash_row(row: &[Cell], pdict: &Dict) -> u64 {
    let mut h = FNV_OFFSET;
    for cell in row {
        match cell {
            Cell::Str(id) => fnv_bytes(&mut h, pdict.text(*id).as_bytes()),
            Cell::Num(x) if x.fract() == 0.0 && x.is_finite() => {
                fnv_bytes(&mut h, &(*x as i64).to_le_bytes())
            }
            _ => {}
        }
    }
    h
}

/// Resolved dictionary ids for a query against a specific graph — computed once,
/// then driven by `for_each_binding`. Sharing this between the fingerprint
/// (`run`) and the row engine (`run_rows`) keeps the thing the benchmark
/// validates identical to the thing callers get rows from.
struct Plan {
    /// label id per node position (`None` = unconstrained)
    label_ids: Vec<Option<u32>>,
    /// edge-type id per relationship position (`None` = unconstrained)
    etype_ids: Vec<Option<u32>>,
    /// node position each WHERE predicate binds against
    pred_pos: Vec<usize>,
}

impl Query {
    fn var_pos(&self, name: &str) -> usize {
        self.nodes
            .iter()
            .position(|n| n.var.as_deref() == Some(name))
            .unwrap_or(0)
    }

    /// Resolve labels/edge-types/predicates to ids. Returns `None` when a
    /// referenced *node label* doesn't exist in the graph — that makes the
    /// pattern unsatisfiable, so callers short-circuit to an empty result.
    /// (A missing *edge type* is handled lazily in the walk, not here.)
    fn plan(&self, g: &Graph) -> Option<Plan> {
        let label_ids: Vec<Option<u32>> = self
            .nodes
            .iter()
            .map(|nd| nd.label.as_ref().and_then(|l| g.labels.get(l)))
            .collect();
        for (nd, lid) in self.nodes.iter().zip(&label_ids) {
            if nd.label.is_some() && lid.is_none() {
                return None;
            }
        }
        let etype_ids: Vec<Option<u32>> = self
            .rels
            .iter()
            .map(|r| r.etype.as_ref().and_then(|t| g.etype.get(t)))
            .collect();
        let pred_pos: Vec<usize> = self.preds.iter().map(|p| self.var_pos(&p.var)).collect();
        Some(Plan {
            label_ids,
            etype_ids,
            pred_pos,
        })
    }

    /// The shared traversal: DFS over the out-CSR, binding each node position in
    /// turn, applying per-node label constraints and the final WHERE, and
    /// invoking `f` once per complete binding (a `&[u32]` of vertex indices, one
    /// per pattern node). Both projectors build on top of this.
    fn for_each_binding(&self, g: &Graph, plan: &Plan, f: &mut impl FnMut(&[u32])) {
        let mut binding = vec![0u32; self.nodes.len()];
        let mut stack: Vec<(usize, u32)> = Vec::new();
        for s in seeds(g, &self.nodes[0]) {
            stack.push((0, s));
            while let Some((depth, v)) = stack.pop() {
                if let Some(lid) = plan.label_ids[depth] {
                    if !g.has_label(v, lid) {
                        continue;
                    }
                }
                binding[depth] = v;
                if depth + 1 == self.nodes.len() {
                    let ok = self
                        .preds
                        .iter()
                        .zip(&plan.pred_pos)
                        .all(|(p, &pp)| pred_holds(g, binding[pp], p));
                    if ok {
                        f(&binding);
                    }
                } else {
                    let et = plan.etype_ids[depth];
                    if self.rels[depth].etype.is_some() && et.is_none() {
                        continue;
                    }
                    for nbr in g.out_neighbors(v, et) {
                        stack.push((depth + 1, nbr));
                    }
                }
            }
        }
    }

    /// Materialize the query as real rows of `Value` — the usable engine output.
    pub fn run_rows(&self, g: &Graph) -> RowSet {
        let cols: Vec<String> = self.items.iter().map(item_name).collect();
        let plan = match self.plan(g) {
            Some(p) => p,
            None => return RowSet::new(cols),
        };
        let aggregating = self.items.iter().any(|i| matches!(i, RetItem::Agg { .. }));

        let group_items: Vec<(usize, Option<String>)> = self
            .items
            .iter()
            .filter_map(|it| match it {
                RetItem::Plain { var, key } => Some((self.var_pos(var), key.clone())),
                _ => None,
            })
            .collect();
        let agg_items: Vec<(AggFunc, Option<usize>, Option<String>)> = self
            .items
            .iter()
            .filter_map(|it| match it {
                RetItem::Agg { func, var, key } => {
                    Some((*func, var.as_deref().map(|v| self.var_pos(v)), key.clone()))
                }
                _ => None,
            })
            .collect();

        let mut rows: Vec<Vec<Value>> = Vec::new();

        if aggregating {
            // group key (canonical string) -> (group-key cells, per-agg accumulators)
            let mut groups: std::collections::HashMap<String, (Vec<Value>, Vec<Acc>)> =
                std::collections::HashMap::new();
            // preserve first-seen group order for stable output
            let mut order: Vec<String> = Vec::new();
            self.for_each_binding(g, &plan, &mut |binding| {
                let key_cells: Vec<Value> = group_items
                    .iter()
                    .map(|(pos, key)| project_value(g, binding, *pos, key))
                    .collect();
                let gkey = row_key(&key_cells);
                let entry = groups.entry(gkey.clone()).or_insert_with(|| {
                    order.push(gkey);
                    (
                        key_cells,
                        (0..agg_items.len()).map(|_| Acc::new()).collect(),
                    )
                });
                for (ai, (_, vpos, key)) in agg_items.iter().enumerate() {
                    let x = match (vpos, key) {
                        (Some(pos), Some(k)) => vertex_num(g, binding[*pos], k),
                        _ => None, // count(*)
                    };
                    entry.1[ai].step(x);
                }
            });
            for gkey in &order {
                let (key_cells, accs) = &groups[gkey];
                let mut gi = 0;
                let mut ai = 0;
                let row: Vec<Value> = self
                    .items
                    .iter()
                    .map(|it| match it {
                        RetItem::Plain { .. } => {
                            let c = key_cells[gi].clone();
                            gi += 1;
                            c
                        }
                        RetItem::Agg { func, .. } => {
                            let v = accs[ai].finalize(*func);
                            ai += 1;
                            Value::Num(v)
                        }
                    })
                    .collect();
                rows.push(row);
            }
        } else {
            self.for_each_binding(g, &plan, &mut |binding| {
                let row: Vec<Value> = self
                    .items
                    .iter()
                    .map(|it| match it {
                        RetItem::Plain { var, key } => {
                            project_value(g, binding, self.var_pos(var), key)
                        }
                        RetItem::Agg { .. } => Value::Null,
                    })
                    .collect();
                rows.push(row);
            });
        }

        if self.distinct {
            let mut seen = std::collections::HashSet::new();
            rows.retain(|r| seen.insert(row_key(r)));
        }

        if let Some(ob) = &self.order {
            rows.sort_by(|a, b| {
                let o = cmp_value(&a[ob.col], &b[ob.col]);
                if ob.desc {
                    o.reverse()
                } else {
                    o
                }
            });
        }

        if let Some(n) = self.limit {
            rows.truncate(n);
        }

        let mut rs = RowSet::new(cols);
        for row in rows {
            rs.push_row(row);
        }
        rs
    }

    pub fn run(&self, g: &Graph) -> QueryResult {
        let plan = match self.plan(g) {
            Some(p) => p,
            None => {
                return QueryResult {
                    count: 0,
                    sum: 0.0,
                    checksum: 0,
                }
            }
        };
        let aggregating = self.items.iter().any(|i| matches!(i, RetItem::Agg { .. }));

        // A small interner so projected string *text* drives the checksum.
        let mut pdict = Dict::default();
        let mut rows: Vec<Vec<Cell>> = Vec::new();

        // group state keyed by the plain (group-key) cells' string form
        let group_items: Vec<(usize, Option<String>)> = self
            .items
            .iter()
            .filter_map(|it| match it {
                RetItem::Plain { var, key } => Some((self.var_pos(var), key.clone())),
                _ => None,
            })
            .collect();
        let agg_items: Vec<(AggFunc, Option<usize>, Option<String>)> = self
            .items
            .iter()
            .filter_map(|it| match it {
                RetItem::Agg { func, var, key } => {
                    Some((*func, var.as_deref().map(|v| self.var_pos(v)), key.clone()))
                }
                _ => None,
            })
            .collect();
        let mut groups: std::collections::HashMap<u64, (Vec<Cell>, Vec<Acc>)> =
            std::collections::HashMap::new();

        self.for_each_binding(g, &plan, &mut |binding| {
            if aggregating {
                // group key cells + hash
                let key_cells: Vec<Cell> = group_items
                    .iter()
                    .map(|(pos, key)| project_cell(g, binding, *pos, key, &mut pdict))
                    .collect();
                let gkey = hash_row(&key_cells, &pdict);
                let entry = groups.entry(gkey).or_insert_with(|| {
                    (
                        key_cells,
                        (0..agg_items.len()).map(|_| Acc::new()).collect(),
                    )
                });
                for (ai, (_, vpos, key)) in agg_items.iter().enumerate() {
                    let x = match (vpos, key) {
                        (Some(pos), Some(k)) => vertex_num(g, binding[*pos], k),
                        _ => None, // count(*)
                    };
                    entry.1[ai].step(x);
                }
            } else {
                let cells: Vec<Cell> = self
                    .items
                    .iter()
                    .map(|it| match it {
                        RetItem::Plain { var, key } => {
                            project_cell(g, binding, self.var_pos(var), key, &mut pdict)
                        }
                        RetItem::Agg { .. } => Cell::Null,
                    })
                    .collect();
                rows.push(cells);
            }
        });

        // Materialize aggregating rows in RETURN-item order.
        if aggregating {
            for (key_cells, accs) in groups.into_values() {
                let mut gi = 0;
                let mut ai = 0;
                let row: Vec<Cell> = self
                    .items
                    .iter()
                    .map(|it| match it {
                        RetItem::Plain { .. } => {
                            let c = key_cells[gi];
                            gi += 1;
                            c
                        }
                        RetItem::Agg { func, .. } => {
                            let v = accs[ai].finalize(*func);
                            ai += 1;
                            Cell::Num(v)
                        }
                    })
                    .collect();
                rows.push(row);
            }
        }

        // DISTINCT (post-projection)
        if self.distinct {
            let mut seen = std::collections::HashSet::new();
            rows.retain(|r| seen.insert(hash_row(r, &pdict)));
        }

        // ORDER BY (+ DESC)
        if let Some(ob) = &self.order {
            rows.sort_by(|a, b| {
                let ca = cell_ord_key(&a[ob.col], &pdict);
                let cb = cell_ord_key(&b[ob.col], &pdict);
                let o = ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal);
                if ob.desc {
                    o.reverse()
                } else {
                    o
                }
            });
        }

        // LIMIT
        if let Some(n) = self.limit {
            rows.truncate(n);
        }

        // Fingerprint. The checksum is order-insensitive (sum of per-row
        // hashes): it verifies the result *set*, not tie-order — which can
        // legitimately differ between engines under ORDER BY with ties.
        let mut sum = 0.0;
        let mut checksum: u64 = 0;
        for row in &rows {
            for cell in row {
                if let Cell::Num(x) = cell {
                    if x.is_finite() {
                        sum += *x;
                    }
                }
            }
            checksum = checksum.wrapping_add(hash_row(row, &pdict));
        }
        QueryResult {
            count: rows.len() as u64,
            sum,
            checksum,
        }
    }
}

/// A sortable f64 key for a cell (numbers by value, strings by interned text
/// hashed to keep it numeric — fine for a benchmark fingerprint).
fn cell_ord_key(c: &Cell, pdict: &Dict) -> f64 {
    match c {
        Cell::Num(x) => *x,
        Cell::Str(id) => {
            // order strings by their bytes via a stable numeric projection
            let mut k = 0f64;
            for (i, &b) in pdict.text(*id).as_bytes().iter().take(8).enumerate() {
                k += (b as f64) * 256f64.powi(7 - i as i32);
            }
            k
        }
        Cell::Null => f64::NEG_INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndjson;

    fn fixture() -> Graph {
        let mut lines = Vec::new();
        for i in 0..10 {
            lines.push(format!(
                "{{\"type\":\"node\",\"id\":\"n{i}\",\"labels\":[\"Person\"],\"properties\":{{\"age\":{},\"dept\":\"d{}\",\"active\":{}}}}}",
                i * 10,
                i % 2,
                i % 2 == 0
            ));
        }
        ndjson::decode(&lines.join("\n")).unwrap()
    }

    fn run(g: &Graph, q: &str) -> QueryResult {
        parse(q).unwrap().run(g)
    }

    #[test]
    fn aggregates() {
        let g = fixture();
        // Every aggregate yields ONE result row; its value lands in `sum`.
        let c = run(&g, "MATCH (a:Person) RETURN count(*)");
        assert_eq!((c.count, c.sum), (1, 10.0));
        assert_eq!(run(&g, "MATCH (a:Person) RETURN sum(a.age)").sum, 450.0); // 0+10+..+90
        assert_eq!(run(&g, "MATCH (a:Person) RETURN max(a.age)").sum, 90.0);
        assert_eq!(run(&g, "MATCH (a:Person) RETURN min(a.age)").sum, 0.0);
        assert_eq!(run(&g, "MATCH (a:Person) RETURN avg(a.age)").sum, 45.0);
    }

    #[test]
    fn multi_where() {
        let g = fixture();
        // active (even i) AND age > 30 → i in {4,6,8} → ages 40,60,80 → count 3
        let r = run(
            &g,
            "MATCH (a:Person) WHERE a.age > 30 AND a.active = true RETURN count(*)",
        );
        assert_eq!((r.count, r.sum), (1, 3.0));
    }

    #[test]
    fn group_by_and_distinct() {
        let g = fixture();
        let gb = run(&g, "MATCH (a:Person) RETURN a.dept, count(*)");
        assert_eq!(gb.count, 2); // d0, d1
        let d = run(&g, "MATCH (a:Person) RETURN DISTINCT a.dept");
        assert_eq!(d.count, 2);
    }

    #[test]
    fn order_limit() {
        let g = fixture();
        let r = run(
            &g,
            "MATCH (a:Person) RETURN a.age ORDER BY a.age DESC LIMIT 3",
        );
        assert_eq!(r.count, 3);
        assert_eq!(r.sum, 90.0 + 80.0 + 70.0);
    }

    fn rows(g: &Graph, q: &str) -> RowSet {
        parse(q).unwrap().run_rows(g)
    }

    /// Materialize a flat RowSet into per-row Vecs for assertions (test-only).
    fn rowvecs(r: &RowSet) -> Vec<Vec<Value>> {
        r.rows().map(|x| x.to_vec()).collect()
    }

    #[test]
    fn rows_project_typed_values() {
        let g = fixture();
        // bare var → external id string; .key → typed property value.
        let r = rows(
            &g,
            "MATCH (a:Person) WHERE a.age = 30 RETURN a, a.age, a.dept, a.active",
        );
        assert_eq!(r.cols, vec!["a", "a.age", "a.dept", "a.active"]);
        assert_eq!(r.nrows, 1);
        assert_eq!(
            rowvecs(&r)[0],
            vec![
                Value::Str("n3".into()),
                Value::Num(30.0),
                Value::Str("d1".into()), // i=3 → dept d(3%2)=d1
                Value::Bool(false),      // i=3 → active (3%2==0) == false
            ]
        );
    }

    #[test]
    fn rows_order_desc_limit_keeps_real_value_order() {
        let g = fixture();
        let r = rows(
            &g,
            "MATCH (a:Person) RETURN a.age ORDER BY a.age DESC LIMIT 3",
        );
        assert_eq!(
            rowvecs(&r),
            vec![
                vec![Value::Num(90.0)],
                vec![Value::Num(80.0)],
                vec![Value::Num(70.0)]
            ]
        );
    }

    #[test]
    fn rows_group_by_aggregate() {
        let g = fixture();
        let r = rows(
            &g,
            "MATCH (a:Person) RETURN a.dept, count(*), sum(a.age) ORDER BY a.dept",
        );
        assert_eq!(r.cols, vec!["a.dept", "count(*)", "sum(a.age)"]);
        // d0: i in {0,2,4,6,8} ages 0+20+40+60+80=200; d1: {1,3,5,7,9} 10+30+50+70+90=250
        assert_eq!(
            rowvecs(&r),
            vec![
                vec![Value::Str("d0".into()), Value::Num(5.0), Value::Num(200.0)],
                vec![Value::Str("d1".into()), Value::Num(5.0), Value::Num(250.0)],
            ]
        );
    }

    #[test]
    fn rows_distinct() {
        let g = fixture();
        let r = rows(
            &g,
            "MATCH (a:Person) RETURN DISTINCT a.dept ORDER BY a.dept",
        );
        assert_eq!(
            rowvecs(&r),
            vec![vec![Value::Str("d0".into())], vec![Value::Str("d1".into())]]
        );
    }

    #[test]
    fn rows_missing_label_is_empty_with_columns() {
        let g = fixture();
        let r = rows(&g, "MATCH (a:Ghost) RETURN a.age");
        assert_eq!(r.cols, vec!["a.age"]);
        assert!(r.nrows == 0);
    }

    #[test]
    fn rowset_json_shape() {
        let g = fixture();
        let r = rows(
            &g,
            "MATCH (a:Person) WHERE a.age = 0 RETURN a.dept, a.age, a.active",
        );
        // Integer-valued floats render without a trailing `.0` (matching the
        // ndjson codec's `push_num`); `0` and `0.0` parse to the same JS number.
        assert_eq!(
            r.to_json(),
            r#"{"columns":["a.dept","a.age","a.active"],"rows":[["d0",0,true]]}"#
        );
    }

    /// The hand-rolled emitter is encode-only over a fixed shape, but it still has
    /// to be exactly right. Use the *trusted* `serde_json` parser as an oracle:
    /// emit a spread of values with our code, parse the result back with serde,
    /// and assert it reconstructs each value (NaN/±Inf documented to become null).
    /// This proves our output is valid JSON and semantically faithful.
    // Uses the crate's own hand-rolled JSON parser (crate::json) as the oracle —
    // gql writer ↔ json parser dogfood each other. Gated on `ndjson` (where that
    // parser lives); gql-only test runs skip it, full runs cover it.
    #[cfg(feature = "ndjson")]
    #[test]
    fn to_json_round_trips_through_the_json_parser() {
        use crate::graph::Value::{Bool, List, Null, Num, Str};
        let cases: Vec<Value> = vec![
            Str("plain".into()),
            Str("".into()),
            Str("quote\" back\\slash".into()),
            Str("ws\n\t\r and ctrl \u{1}\u{1f}".into()),
            Str("del\u{7f} unicode é ✓ 🎉".into()),
            // The sneaky ones: a bare null byte and an embedded one (lowest
            // control char →  ); a zero-width space and BOM (invisible but
            // ordinary codepoints, emitted raw); U+2028/U+2029 (valid in JSON,
            // historically broke JS string literals — JSON.parse handles them).
            Str("\u{0}".into()),
            Str("embedded\u{0}null".into()),
            Str("zero\u{200b}width\u{feff}bom".into()),
            Str("line\u{2028}para\u{2029}sep".into()),
            Bool(true),
            Bool(false),
            Null,
            Num(0.0),
            Num(-0.0),
            Num(42.0),
            Num(-3.5),
            Num(1e20),
            Num(1e-12),
            Num(123456789.125),
            Num(f64::NAN),
            Num(f64::INFINITY),
            Num(f64::NEG_INFINITY),
            List(vec![]),
            List(vec![Num(1.0), Str("a,b\"c".into()), Bool(false), Null]),
            List(vec![List(vec![Num(2.0)]), Str("nested".into())]),
        ];
        let mut rs = RowSet::new(vec!["v".to_string()]);
        for c in &cases {
            rs.push_row([c.clone()]);
        }
        // The parsed tree borrows its source, so the JSON text needs a binding
        // rather than living as a temporary.
        let json = rs.to_json();
        let doc = crate::json::parse(&json).expect("hand-rolled JSON must parse");
        let rows = doc
            .get("rows")
            .and_then(crate::json::Json::as_array)
            .expect("rows array");
        assert_eq!(rows.len(), cases.len());
        for (i, want) in cases.iter().enumerate() {
            let cell = &rows[i].as_array().expect("each row is an array")[0];
            assert!(json_matches(cell, want), "case {i}: {want:?} → {cell:?}");
        }
    }

    /// Compare a parsed [`crate::json::Json`] against our core `Value` (numbers by
    /// f64, so `42` and `42.0` agree; non-finite numbers must have become null).
    #[cfg(feature = "ndjson")]
    fn json_matches(got: &crate::json::Json, want: &Value) -> bool {
        use crate::json::Json;
        match want {
            Value::Null => matches!(got, Json::Null),
            Value::Bool(b) => matches!(got, Json::Bool(g) if g == b),
            Value::Num(x) if x.is_finite() => got.as_f64() == Some(*x),
            Value::Num(_) => matches!(got, Json::Null),
            Value::Str(s) => got.as_str() == Some(s.as_ref()),
            Value::Temporal(t) => {
                got.as_object().and_then(crate::json::temporal_from_pairs) == Some(Ok(*t))
            }
            Value::List(items) => got.as_array().is_some_and(|a| {
                a.len() == items.len() && a.iter().zip(items).all(|(g, w)| json_matches(g, w))
            }),
            Value::Map(pairs) => got.as_object().is_some_and(|o| {
                o.len() == pairs.len()
                    && pairs
                        .iter()
                        .all(|(k, w)| got.get(k.as_ref()).is_some_and(|g| json_matches(g, w)))
            }),
        }
    }
}

#[cfg(test)]
mod signed_zero_agreement {
    /// The fingerprint engine and the real one must agree about grouping, since
    /// agreeing is the whole reason this engine exists.
    ///
    /// `-0.0` and `0.0` are ONE key. This engine keyed on raw bits and gave two
    /// rows where `gql` gave one — a divergence between two implementations in
    /// one crate, on a decision that had been settled and applied to only one of
    /// them.
    #[test]
    fn signed_zeros_are_one_group_in_both_engines() {
        let lines = [
            r#"{"type":"node","id":"a","labels":["N"],"properties":{"z":0.0}}"#,
            r#"{"type":"node","id":"b","labels":["N"],"properties":{"z":-0.0}}"#,
        ]
        .join("\n");
        let mut g = crate::ndjson::decode(&lines).expect("fixture decodes");

        // Both signs really are stored; without that this test proves nothing.
        let kid = g.props.keys.get("z").expect("the key exists");
        let bits: Vec<u64> = (0..2)
            .map(
                |i| match crate::value::Value::from_column(&g.props, kid, i, &g.strs, false) {
                    crate::value::Value::Num(x) => x.to_bits(),
                    other => panic!("expected a number, got {other:?}"),
                },
            )
            .collect();

        assert_ne!(bits[0], bits[1], "both zero signs must reach the column");

        let q = "MATCH (u:N) RETURN DISTINCT u.z";

        assert_eq!(super::parse(q).expect("parses").run_rows(&g).nrows, 1);
        assert_eq!(
            crate::gql::parse(q)
                .expect("parses")
                .execute(&mut g, &crate::gql::eval::Params::new())
                .expect("runs")
                .nrows,
            1
        );
    }
}
