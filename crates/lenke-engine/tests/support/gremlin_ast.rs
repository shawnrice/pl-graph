//! A neutral Gremlin step AST shared by the parity harness. One case is authored
//! once and lowered two ways: [`Steps::to_query`] serializes it to the string the
//! ENGINE's own Gremlin parser consumes, and [`Steps::to_core`] builds the
//! equivalent `lenke_core::gremlin::Traversal` through core's public fluent API.
//! Running both against the same fixture and comparing results is a true
//! differential test — neither side's expected values are hand-transcribed.
//!
//! Adding a step: add a `Step` variant, then handle it in BOTH `emit_step` (engine
//! string) and `apply_core` (core builder). A variant with no core builder method
//! or no engine syntax is exactly the parity gap the harness exists to surface.

#![allow(dead_code)]

use lenke_core::gremlin::{
    self as cg, Order as COrder, SackOp as CSackOp, Traversal as CTrav, P as CP,
};

/// A literal value in a step argument (`has('age', 29)`, `constant('x')`).
#[derive(Clone, Debug)]
pub enum Val {
    S(&'static str),
    N(f64),
    B(bool),
}

/// A `has`/`is`/`where` predicate, mirroring core's `P` constructors.
#[derive(Clone, Debug)]
pub enum Pred {
    Eq(Val),
    Neq(Val),
    Gt(Val),
    Gte(Val),
    Lt(Val),
    Lte(Val),
    Between(Val, Val),
    Inside(Val, Val),
    Within(Vec<Val>),
    Without(Vec<Val>),
    StartingWith(&'static str),
    Containing(&'static str),
}

/// A single Gremlin step. Sub-traversal steps carry a nested `Vec<Step>` (an
/// anonymous `__` traversal).
#[derive(Clone, Debug)]
pub enum Step {
    V,
    E,
    Out(Vec<&'static str>),
    In(Vec<&'static str>),
    Both(Vec<&'static str>),
    OutE(Vec<&'static str>),
    InE(Vec<&'static str>),
    BothE(Vec<&'static str>),
    OutV,
    InV,
    BothV,
    OtherV,
    Has(&'static str, Pred),
    HasVal(&'static str, Val),
    HasLabel(Vec<&'static str>),
    HasNot(&'static str),
    Is(Pred),
    Values(Vec<&'static str>),
    ValueMap(Vec<&'static str>),
    ElementMap(Vec<&'static str>),
    Label,
    Id,
    Count,
    Sum,
    Min,
    Max,
    Mean,
    Fold,
    Dedup,
    SimplePath,
    CyclicPath,
    Order,
    /// `.by('key', dir)` modulator on the preceding step.
    ByKey(&'static str, COrder),
    /// `.by(dir)` modulator — natural order of the current value, explicit direction.
    ByValue(COrder),
    /// `.by('key')` modulator (identity direction / projection).
    By(&'static str),
    Limit(usize),
    Range(usize, usize),
    Tail(usize),
    Skip(usize),
    GroupCount,
    Group,
    As(&'static str),
    Select(Vec<&'static str>),
    Where(Vec<Step>),
    WhereKey(&'static str, Pred),
    And(Vec<Vec<Step>>),
    Or(Vec<Vec<Step>>),
    Not(Vec<Step>),
    Union(Vec<Vec<Step>>),
    Coalesce(Vec<Vec<Step>>),
    Optional(Vec<Step>),
    Local(Vec<Step>),
    Repeat(Vec<Step>),
    Times(usize),
    Until(Vec<Step>),
    Emit,
    Path,
    Project(Vec<&'static str>),
    Unfold,
    Constant(Val),
    Identity,
    Barrier,
    Aggregate(&'static str),
    Store(&'static str),
    Cap(&'static str),
    Subgraph(&'static str),
    ShortestPath,
    PageRank(Option<f64>),
    ConnectedComponent,
    PeerPressure,
    TailLocal(usize),
    Inject(Vec<Val>),
    Tree,
    DedupLabels(Vec<&'static str>),
    WithSack(Val),
    SackRead,
    SackBy(&'static str, &'static str),
    WhereKeyTag(&'static str, &'static str, &'static str),
    Branch(&'static str, Vec<(Option<Val>, Vec<Step>)>),
}

/// A whole traversal: an ordered list of steps.
#[derive(Clone, Debug)]
pub struct Steps(pub Vec<Step>);

// ── engine-string serialization ─────────────────────────────────────────────

fn emit_val(v: &Val) -> String {
    match v {
        Val::S(s) => format!("'{s}'"),
        Val::N(n) if n.fract() == 0.0 => format!("{}", *n as i64),
        Val::N(n) => format!("{n}"),
        Val::B(b) => format!("{b}"),
    }
}

fn emit_pred(p: &Pred) -> String {
    match p {
        Pred::Eq(v) => format!("eq({})", emit_val(v)),
        Pred::Neq(v) => format!("neq({})", emit_val(v)),
        Pred::Gt(v) => format!("gt({})", emit_val(v)),
        Pred::Gte(v) => format!("gte({})", emit_val(v)),
        Pred::Lt(v) => format!("lt({})", emit_val(v)),
        Pred::Lte(v) => format!("lte({})", emit_val(v)),
        Pred::Between(a, b) => format!("between({},{})", emit_val(a), emit_val(b)),
        Pred::Inside(a, b) => format!("inside({},{})", emit_val(a), emit_val(b)),
        Pred::Within(vs) => format!("within({})", join_vals(vs)),
        Pred::Without(vs) => format!("without({})", join_vals(vs)),
        Pred::StartingWith(s) => format!("startingWith('{s}')"),
        Pred::Containing(s) => format!("containing('{s}')"),
    }
}

fn join_vals(vs: &[Val]) -> String {
    vs.iter().map(emit_val).collect::<Vec<_>>().join(",")
}

fn join_labels(ls: &[&str]) -> String {
    ls.iter()
        .map(|l| format!("'{l}'"))
        .collect::<Vec<_>>()
        .join(",")
}

fn emit_sub(steps: &[Step]) -> String {
    let mut s = String::from("__");
    for st in steps {
        s.push('.');
        s.push_str(&emit_step(st));
    }
    s
}

fn emit_step(st: &Step) -> String {
    use Step::*;
    match st {
        V => "V()".into(),
        E => "E()".into(),
        Out(l) => format!("out({})", join_labels(l)),
        In(l) => format!("in({})", join_labels(l)),
        Both(l) => format!("both({})", join_labels(l)),
        OutE(l) => format!("outE({})", join_labels(l)),
        InE(l) => format!("inE({})", join_labels(l)),
        BothE(l) => format!("bothE({})", join_labels(l)),
        OutV => "outV()".into(),
        InV => "inV()".into(),
        BothV => "bothV()".into(),
        OtherV => "otherV()".into(),
        Has(k, p) => format!("has('{k}',{})", emit_pred(p)),
        HasVal(k, v) => format!("has('{k}',{})", emit_val(v)),
        HasLabel(l) => format!("hasLabel({})", join_labels(l)),
        HasNot(k) => format!("hasNot('{k}')"),
        Is(p) => format!("is({})", emit_pred(p)),
        Values(l) => format!("values({})", join_labels(l)),
        ValueMap(l) => format!("valueMap({})", join_labels(l)),
        ElementMap(l) => format!("elementMap({})", join_labels(l)),
        Label => "label()".into(),
        Id => "id()".into(),
        Count => "count()".into(),
        Sum => "sum()".into(),
        Min => "min()".into(),
        Max => "max()".into(),
        Mean => "mean()".into(),
        Fold => "fold()".into(),
        Dedup => "dedup()".into(),
        SimplePath => "simplePath()".into(),
        CyclicPath => "cyclicPath()".into(),
        Order => "order()".into(),
        ByKey(k, d) => format!(
            "by('{k}',{})",
            if matches!(d, COrder::Desc) {
                "desc"
            } else {
                "asc"
            }
        ),
        ByValue(d) => format!(
            "by({})",
            if matches!(d, COrder::Desc) {
                "desc"
            } else {
                "asc"
            }
        ),
        By(k) => format!("by('{k}')"),
        Limit(n) => format!("limit({n})"),
        Range(a, b) => format!("range({a},{b})"),
        Tail(n) => format!("tail({n})"),
        Skip(n) => format!("skip({n})"),
        GroupCount => "groupCount()".into(),
        Group => "group()".into(),
        As(l) => format!("as('{l}')"),
        Select(l) => format!("select({})", join_labels(l)),
        Where(s) => format!("where({})", emit_sub(s)),
        WhereKey(k, p) => format!("where('{k}',{})", emit_pred(p)),
        And(bs) => format!(
            "and({})",
            bs.iter().map(|b| emit_sub(b)).collect::<Vec<_>>().join(",")
        ),
        Or(bs) => format!(
            "or({})",
            bs.iter().map(|b| emit_sub(b)).collect::<Vec<_>>().join(",")
        ),
        Not(s) => format!("not({})", emit_sub(s)),
        Union(bs) => format!(
            "union({})",
            bs.iter().map(|b| emit_sub(b)).collect::<Vec<_>>().join(",")
        ),
        Coalesce(bs) => format!(
            "coalesce({})",
            bs.iter().map(|b| emit_sub(b)).collect::<Vec<_>>().join(",")
        ),
        Optional(s) => format!("optional({})", emit_sub(s)),
        Local(s) => format!("local({})", emit_sub(s)),
        Repeat(s) => format!("repeat({})", emit_sub(s)),
        Times(n) => format!("times({n})"),
        Until(s) => format!("until({})", emit_sub(s)),
        Emit => "emit()".into(),
        Path => "path()".into(),
        Project(l) => format!("project({})", join_labels(l)),
        Unfold => "unfold()".into(),
        Constant(v) => format!("constant({})", emit_val(v)),
        Identity => "identity()".into(),
        Barrier => "barrier()".into(),
        Aggregate(k) => format!("aggregate('{k}')"),
        Store(k) => format!("store('{k}')"),
        Cap(k) => format!("cap('{k}')"),
        Subgraph(k) => format!("subgraph('{k}')"),
        ShortestPath => "shortestPath()".into(),
        PageRank(Some(a)) => format!("pageRank({a})"),
        PageRank(None) => "pageRank()".into(),
        ConnectedComponent => "connectedComponent()".into(),
        PeerPressure => "peerPressure()".into(),
        TailLocal(n) => format!("tail(local,{n})"),
        Inject(vs) => format!("inject({})", join_vals(vs)),
        Tree => "tree()".into(),
        DedupLabels(ls) => format!("dedup({})", join_labels(ls)),
        WithSack(v) => format!("withSack({})", emit_val(v)),
        SackRead => "sack()".into(),
        SackBy(op, k) => format!("sack({op}).by('{k}')"),
        WhereKeyTag(a, op, b) => format!("where('{a}',{op}('{b}'))"),
        Branch(test, opts) => {
            let mut out = format!("branch(values('{test}'))");
            for (m, body) in opts {
                match m {
                    Some(v) => {
                        out.push_str(&format!(".option({},{})", emit_val(v), emit_sub(body)))
                    }
                    None => out.push_str(&format!(".option(none,{})", emit_sub(body))),
                }
            }
            out
        }
    }
}

impl Steps {
    /// The engine-dialect Gremlin string (`g.V().out('KNOWS')…`).
    pub fn to_query(&self) -> String {
        let mut s = String::from("g");
        for st in &self.0 {
            s.push('.');
            s.push_str(&emit_step(st));
        }
        s
    }
}

// ── core-`Traversal` construction ───────────────────────────────────────────

fn cval(v: &Val) -> cg::GVal {
    match v {
        Val::S(s) => cg::GVal::Str((*s).into()),
        Val::N(n) => cg::GVal::Num(*n),
        Val::B(b) => cg::GVal::Bool(*b),
    }
}

fn cpred(p: &Pred) -> CP {
    match p {
        Pred::Eq(v) => CP::eq(cval(v)),
        Pred::Neq(v) => CP::neq(cval(v)),
        Pred::Gt(v) => CP::gt(cval(v)),
        Pred::Gte(v) => CP::gte(cval(v)),
        Pred::Lt(v) => CP::lt(cval(v)),
        Pred::Lte(v) => CP::lte(cval(v)),
        Pred::Between(a, b) => CP::between(cval(a), cval(b)),
        Pred::Inside(a, b) => CP::inside(cval(a), cval(b)),
        Pred::Within(vs) => CP::within(vs.iter().map(cval).collect::<Vec<_>>()),
        Pred::Without(vs) => CP::without(vs.iter().map(cval).collect::<Vec<_>>()),
        Pred::StartingWith(s) => CP::starts_with(s),
        Pred::Containing(s) => CP::containing(s),
    }
}

fn csub(steps: &[Step]) -> CTrav {
    let mut t = cg::__();
    for st in steps {
        t = apply_core(t, st);
    }
    t
}

fn apply_core(t: CTrav, st: &Step) -> CTrav {
    use Step::*;
    match st {
        V => t.V(),
        E => t.E(),
        Out(l) => t.out(l),
        In(l) => t.in_(l),
        Both(l) => t.both(l),
        OutE(l) => t.out_e(l),
        InE(l) => t.in_e(l),
        BothE(l) => t.both_e(l),
        OutV => t.out_v(),
        InV => t.in_v(),
        BothV => t.both_v(),
        OtherV => t.other_v(),
        Has(k, p) => t.has(k, cpred(p)),
        HasVal(k, v) => t.has_val(k, cval(v)),
        HasLabel(l) => t.has_label(l),
        HasNot(k) => t.has_not(&[k]),
        Is(p) => t.is(cpred(p)),
        Values(l) => t.values(l),
        ValueMap(l) => t.value_map(l),
        ElementMap(l) => t.element_map(l),
        Label => t.label(),
        Id => t.id(),
        Count => t.count(),
        Sum => t.sum(),
        Min => t.min(),
        Max => t.max(),
        Mean => t.mean(),
        Fold => t.fold(),
        Dedup => t.dedup(),
        SimplePath => t.simple_path(),
        CyclicPath => t.cyclic_path(),
        Order => t.order(),
        ByKey(k, d) => t.by_dir(k, *d),
        ByValue(d) => t.by_identity_dir(*d),
        By(k) => t.by(k),
        Limit(n) => t.limit(*n),
        Range(a, b) => t.range(*a, *b),
        Tail(n) => t.tail(*n),
        Skip(n) => t.skip(*n),
        GroupCount => t.group_count(),
        Group => t.group(),
        As(l) => t.as_(l),
        Select(l) => t.select(l),
        Where(s) => t.where_(csub(s)),
        WhereKey(k, p) => t.where_key(k, cpred(p)),
        And(bs) => t.and(bs.iter().map(|b| csub(b)).collect()),
        Or(bs) => t.or(bs.iter().map(|b| csub(b)).collect()),
        Not(s) => t.not(csub(s)),
        Union(bs) => t.union(bs.iter().map(|b| csub(b)).collect()),
        Coalesce(bs) => t.coalesce(bs.iter().map(|b| csub(b)).collect()),
        Optional(s) => t.optional(csub(s)),
        Local(s) => t.local(csub(s)),
        Repeat(s) => t.repeat(csub(s)),
        Times(n) => t.times(*n),
        Until(s) => t.until(csub(s)),
        Emit => t.emit_all(),
        Path => t.path(),
        Project(l) => t.project(l),
        Unfold => t.unfold(),
        Constant(v) => t.constant(cval(v)),
        Identity => t.identity(),
        Barrier => t.barrier(),
        Aggregate(k) => t.aggregate(k),
        Store(k) => t.store(k),
        Cap(k) => t.cap(k),
        Subgraph(k) => t.subgraph(k),
        ShortestPath => t.shortest_path(),
        PageRank(a) => t.page_rank(*a),
        ConnectedComponent => t.connected_component(),
        PeerPressure => t.peer_pressure(),
        TailLocal(n) => t.tail_local(*n),
        Inject(vs) => t.inject(vs.iter().map(cval).collect::<Vec<_>>()),
        Tree => t.tree(),
        DedupLabels(ls) => t.dedup_labels(ls.iter().map(|s| s.to_string()).collect()),
        WithSack(v) => t.with_sack(cval(v)),
        SackRead => t.sack(),
        SackBy(op, k) => {
            let o = match *op {
                "sum" => CSackOp::Sum,
                "mult" => CSackOp::Mult,
                "assign" => CSackOp::Assign,
                "min" => CSackOp::Min,
                _ => CSackOp::Max,
            };
            t.sack_op(o).by(k)
        }
        WhereKeyTag(a, op, b) => {
            let rhs = cg::GVal::Str((*b).into());
            let p = match *op {
                "eq" => CP::eq(rhs),
                "neq" => CP::neq(rhs),
                "gt" => CP::gt(rhs),
                "gte" => CP::gte(rhs),
                "lt" => CP::lt(rhs),
                _ => CP::lte(rhs),
            };
            t.where_key(a, p)
        }
        Branch(test, opts) => {
            let mut t = t.branch(cg::__().values(&[test]));
            for (m, body) in opts {
                match m {
                    Some(v) => t = t.option(cval(v), csub(body)),
                    None => t = t.option_none(csub(body)),
                }
            }
            t
        }
    }
}

impl Steps {
    /// Build the equivalent core `Traversal` via the public fluent API.
    pub fn to_core(&self) -> CTrav {
        let mut t = cg::g();
        for st in &self.0 {
            t = apply_core(t, st);
        }
        t
    }
}
