//! An engine Gremlin builder: every method has the SAME signature core's
//! `gremlin::Traversal` had, but each call only appends the equivalent engine query
//! fragment — `query()` hands back the string, which the harness parses and runs on
//! the engine. Core's Gremlin test bodies were written against a builder of this
//! shape, so they still compile and run verbatim; the engine is now the sole engine
//! (the `lenke-core` oracle it was dual-checked against has been deleted, its
//! byte-identity contract now upheld by the TS differential fuzzers).
//!
//! `GVal`, `Order`, `Column`, `Pop`, `Token`, `Scope`, `SackOp` are small local
//! mirrors of core's types (the bodies name them); the engine speaks query strings,
//! not a builder, so it has no equivalents of its own.
//!
//! A method whose engine spelling is not yet handled still builds the string; the
//! engine's parser then errors on it — surfacing the real gap when the test runs.

#![allow(
    dead_code,
    non_snake_case,
    clippy::wrong_self_convention,
    clippy::should_implement_trait
)]

// ── local value + enum mirrors (the bodies name these) ───────────────────────

/// A Gremlin value, mirroring the variants core's `Value`/`GVal` exposed to the test
/// bodies. `Node` carries a vertex's EXTERNAL id string (the engine has no interior
/// `Value::Node`; a bare vertex arrives as a `{id,labels,properties}` map, which the
/// result converter collapses to `Node(ext_id)` so `V()` reads like core's).
#[derive(Clone, Debug, PartialEq)]
pub enum GVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<GVal>),
    Map(MapVal),
    /// A vertex, by EXTERNAL id (the engine has no interior `Value::Node`).
    Node(String),
    /// An edge, by external id.
    Edge(String),
    /// A Gremlin `Property` result: `(owner, key, value)` — owner ignored by equality,
    /// mirroring core's `PropertyVal`.
    Property(Box<GVal>, String, Box<GVal>),
}

/// A TinkerPop map (any-value keys, insertion-ordered) — the shape the ported bodies
/// name via `GVal::Map`, with the `iter`/`values`/`get`/`into_pairs` surface they use.
#[derive(Clone, Debug)]
pub struct MapVal(pub Vec<(GVal, GVal)>);

impl MapVal {
    pub fn from_pairs(pairs: Vec<(GVal, GVal)>) -> Self {
        MapVal(pairs)
    }
    pub fn iter(&self) -> std::slice::Iter<'_, (GVal, GVal)> {
        self.0.iter()
    }
    pub fn values(&self) -> Vec<GVal> {
        self.0.iter().map(|(_, v)| v.clone()).collect()
    }
    pub fn keys(&self) -> Vec<GVal> {
        self.0.iter().map(|(k, _)| k.clone()).collect()
    }
    pub fn get(&self, key: &GVal) -> Option<&GVal> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn into_pairs(self) -> Vec<(GVal, GVal)> {
        self.0
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// Insertion-ordered equality (positional), like core's Gremlin `MapVal`.
impl PartialEq for MapVal {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl GVal {
    /// A list value (`GVal::list(vec![…])`).
    pub fn list(items: Vec<GVal>) -> GVal {
        GVal::List(items)
    }
    /// A map value from key/value pairs.
    pub fn map(pairs: Vec<(GVal, GVal)>) -> GVal {
        GVal::Map(MapVal::from_pairs(pairs))
    }
    /// A Gremlin `Property` value (owner is ignored by equality).
    pub fn property(owner: GVal, key: impl Into<String>, value: GVal) -> GVal {
        GVal::Property(Box::new(owner), key.into(), Box::new(value))
    }
}

impl From<&str> for GVal {
    fn from(s: &str) -> Self {
        GVal::Str(s.to_string())
    }
}
impl From<String> for GVal {
    fn from(s: String) -> Self {
        GVal::Str(s)
    }
}
impl From<&String> for GVal {
    fn from(s: &String) -> Self {
        GVal::Str(s.clone())
    }
}
impl From<bool> for GVal {
    fn from(b: bool) -> Self {
        GVal::Bool(b)
    }
}
impl From<i32> for GVal {
    fn from(n: i32) -> Self {
        GVal::Num(n as f64)
    }
}
impl From<i64> for GVal {
    fn from(n: i64) -> Self {
        GVal::Num(n as f64)
    }
}
impl From<usize> for GVal {
    fn from(n: usize) -> Self {
        GVal::Num(n as f64)
    }
}
impl From<f64> for GVal {
    fn from(n: f64) -> Self {
        GVal::Num(n)
    }
}
impl<T: Into<GVal>> From<Vec<T>> for GVal {
    fn from(xs: Vec<T>) -> Self {
        GVal::List(xs.into_iter().map(Into::into).collect())
    }
}

/// Sort direction (`order().by(..., asc|desc)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Asc,
    Desc,
    Shuffle,
}

/// `select(keys|values)` on a map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Column {
    Keys,
    Values,
}

/// `select(pop, ...)` — which tagged value when a label was bound many times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pop {
    First,
    Last,
    All,
    Mixed,
}

/// A `by(token)` element accessor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token {
    Id,
    Label,
    Key,
    Value,
}

/// `count(local)` / `order(local)` scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Global,
    Local,
}

/// A `sack(op)` accumulator operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SackOp {
    Sum,
    Mult,
    Min,
    Max,
    Assign,
}

fn dir_word(d: Order) -> &'static str {
    match d {
        Order::Desc => "desc",
        _ => "asc",
    }
}

/// Render a `GVal` as an engine literal fragment.
pub fn val(v: &GVal) -> String {
    match v {
        GVal::Str(s) => format!("'{s}'"),
        GVal::Num(n) if n.fract() == 0.0 && n.is_finite() => format!("{}", *n as i64),
        GVal::Num(n) => format!("{n}"),
        GVal::Bool(b) => format!("{b}"),
        GVal::Null => "null".into(),
        // A list renders as a `[…]` literal (each element via `val`), so
        // inject([1,2,3]) drives a real list-literal query, not Debug output.
        GVal::List(items) => {
            let inner = items.iter().map(val).collect::<Vec<_>>().join(",");
            format!("[{inner}]")
        }
        other => format!("{other:?}"),
    }
}

fn labels(ls: &[&str]) -> String {
    ls.iter()
        .map(|l| format!("'{l}'"))
        .collect::<Vec<_>>()
        .join(",")
}

// ── predicate shim ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct P {
    eng: String,
}

impl P {
    /// The engine fragment this predicate renders to (e.g. `gt(30)`).
    pub fn frag(&self) -> &str {
        &self.eng
    }
    pub fn eq(v: impl Into<GVal>) -> Self {
        Self {
            eng: format!("eq({})", val(&v.into())),
        }
    }
    pub fn neq(v: impl Into<GVal>) -> Self {
        Self {
            eng: format!("neq({})", val(&v.into())),
        }
    }
    pub fn gt(v: impl Into<GVal>) -> Self {
        Self {
            eng: format!("gt({})", val(&v.into())),
        }
    }
    pub fn gte(v: impl Into<GVal>) -> Self {
        Self {
            eng: format!("gte({})", val(&v.into())),
        }
    }
    pub fn lt(v: impl Into<GVal>) -> Self {
        Self {
            eng: format!("lt({})", val(&v.into())),
        }
    }
    pub fn lte(v: impl Into<GVal>) -> Self {
        Self {
            eng: format!("lte({})", val(&v.into())),
        }
    }
    pub fn between(a: impl Into<GVal>, b: impl Into<GVal>) -> Self {
        Self {
            eng: format!("between({},{})", val(&a.into()), val(&b.into())),
        }
    }
    pub fn inside(a: impl Into<GVal>, b: impl Into<GVal>) -> Self {
        Self {
            eng: format!("inside({},{})", val(&a.into()), val(&b.into())),
        }
    }
    pub fn outside(a: impl Into<GVal>, b: impl Into<GVal>) -> Self {
        Self {
            eng: format!("outside({},{})", val(&a.into()), val(&b.into())),
        }
    }
    pub fn within<V: Into<GVal>>(vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        Self {
            eng: format!("within({frag})"),
        }
    }
    pub fn without<V: Into<GVal>>(vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        Self {
            eng: format!("without({frag})"),
        }
    }
    pub fn starts_with(s: &str) -> Self {
        Self {
            eng: format!("startingWith('{s}')"),
        }
    }
    pub fn containing(s: &str) -> Self {
        Self {
            eng: format!("containing('{s}')"),
        }
    }
    pub fn regex(s: &str) -> Self {
        Self {
            eng: format!("regex('{s}')"),
        }
    }
    pub fn not(p: Self) -> Self {
        Self {
            eng: format!("not({})", p.eng),
        }
    }
}

// ── traversal builder ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Traversal {
    eng: String,
}

pub fn g() -> Traversal {
    Traversal { eng: "g".into() }
}

pub fn __() -> Traversal {
    Traversal { eng: "__".into() }
}

impl Traversal {
    pub fn query(&self) -> String {
        self.eng.clone()
    }

    fn step(mut self, frag: &str) -> Self {
        self.eng.push('.');
        self.eng.push_str(frag);
        self
    }

    // sources
    pub fn V(self) -> Self {
        self.step("V()")
    }
    pub fn E(self) -> Self {
        self.step("E()")
    }
    pub fn v_ids(self, ids: &[&str]) -> Self {
        self.step(&format!("V({})", labels(ids)))
    }
    pub fn e_ids(self, ids: &[&str]) -> Self {
        self.step(&format!("E({})", labels(ids)))
    }

    // hops
    pub fn out(self, l: &[&str]) -> Self {
        self.step(&format!("out({})", labels(l)))
    }
    pub fn in_(self, l: &[&str]) -> Self {
        self.step(&format!("in({})", labels(l)))
    }
    pub fn both(self, l: &[&str]) -> Self {
        self.step(&format!("both({})", labels(l)))
    }
    pub fn out_e(self, l: &[&str]) -> Self {
        self.step(&format!("outE({})", labels(l)))
    }
    pub fn in_e(self, l: &[&str]) -> Self {
        self.step(&format!("inE({})", labels(l)))
    }
    pub fn both_e(self, l: &[&str]) -> Self {
        self.step(&format!("bothE({})", labels(l)))
    }
    pub fn out_v(self) -> Self {
        self.step("outV()")
    }
    pub fn in_v(self) -> Self {
        self.step("inV()")
    }
    pub fn other_v(self) -> Self {
        self.step("otherV()")
    }
    pub fn both_v(self) -> Self {
        self.step("bothV()")
    }

    // filters
    pub fn has(self, key: &str, pred: P) -> Self {
        self.step(&format!("has('{key}',{})", pred.eng))
    }
    pub fn has_val(self, key: &str, v: impl Into<GVal>) -> Self {
        self.step(&format!("has('{key}',{})", val(&v.into())))
    }
    pub fn has_label_key(self, label: &str, key: &str, pred: P) -> Self {
        self.step(&format!("has('{label}','{key}',{})", pred.eng))
    }
    pub fn has_label(self, l: &[&str]) -> Self {
        self.step(&format!("hasLabel({})", labels(l)))
    }
    pub fn has_id(self, ids: &[&str]) -> Self {
        self.step(&format!("hasId({})", labels(ids)))
    }
    pub fn has_key(self, keys: &[&str]) -> Self {
        self.step(&format!("hasKey({})", labels(keys)))
    }
    pub fn has_not(self, keys: &[&str]) -> Self {
        self.step(&format!("hasNot({})", labels(keys)))
    }
    pub fn has_value<V: Into<GVal>>(self, vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        self.step(&format!("hasValue({frag})"))
    }
    pub fn is(self, pred: P) -> Self {
        self.step(&format!("is({})", pred.eng))
    }
    pub fn where_(self, sub: Self) -> Self {
        self.step(&format!("where({})", sub.eng))
    }
    pub fn where_key(self, start: &str, pred: P) -> Self {
        self.step(&format!("where('{start}',{})", pred.eng))
    }
    pub fn where_pred(self, pred: P) -> Self {
        self.step(&format!("where({})", pred.eng))
    }
    pub fn and(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        self.step(&format!("and({frag})"))
    }
    pub fn or(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        self.step(&format!("or({frag})"))
    }
    pub fn not(self, sub: Self) -> Self {
        self.step(&format!("not({})", sub.eng))
    }
    pub fn dedup(self) -> Self {
        self.step("dedup()")
    }
    pub fn dedup_labels(self, ls: Vec<String>) -> Self {
        let frag = ls
            .iter()
            .map(|l| format!("'{l}'"))
            .collect::<Vec<_>>()
            .join(",");
        self.step(&format!("dedup({frag})"))
    }
    pub fn simple_path(self) -> Self {
        self.step("simplePath()")
    }
    pub fn cyclic_path(self) -> Self {
        self.step("cyclicPath()")
    }

    // projections
    pub fn values(self, keys: &[&str]) -> Self {
        self.step(&format!("values({})", labels(keys)))
    }
    pub fn value_map(self, keys: &[&str]) -> Self {
        self.step(&format!("valueMap({})", labels(keys)))
    }
    pub fn property_map(self, keys: &[&str]) -> Self {
        self.step(&format!("propertyMap({})", labels(keys)))
    }
    pub fn element_map(self, keys: &[&str]) -> Self {
        self.step(&format!("elementMap({})", labels(keys)))
    }
    pub fn properties(self, keys: &[&str]) -> Self {
        self.step(&format!("properties({})", labels(keys)))
    }
    pub fn value(self) -> Self {
        self.step("value()")
    }
    pub fn id(self) -> Self {
        self.step("id()")
    }
    pub fn label(self) -> Self {
        self.step("label()")
    }
    pub fn path(self) -> Self {
        self.step("path()")
    }
    pub fn project(self, keys: &[&str]) -> Self {
        self.step(&format!("project({})", labels(keys)))
    }
    pub fn tree(self) -> Self {
        self.step("tree()")
    }

    // modulators
    pub fn by(self, key: &str) -> Self {
        self.step(&format!("by('{key}')"))
    }
    pub fn by_identity(self) -> Self {
        self.step("by()")
    }
    pub fn by_dir(self, key: &str, dir: Order) -> Self {
        self.step(&format!("by('{key}',{})", dir_word(dir)))
    }
    pub fn by_identity_dir(self, dir: Order) -> Self {
        self.step(&format!("by({})", dir_word(dir)))
    }
    pub fn by_t(self, t: Self) -> Self {
        self.step(&format!("by({})", t.eng))
    }
    pub fn by_t_dir(self, t: Self, dir: Order) -> Self {
        self.step(&format!("by({},{})", t.eng, dir_word(dir)))
    }

    // paging / bounds
    pub fn limit(self, n: usize) -> Self {
        self.step(&format!("limit({n})"))
    }
    pub fn limit_local(self, n: usize) -> Self {
        self.step(&format!("limit(local,{n})"))
    }
    pub fn skip(self, n: usize) -> Self {
        self.step(&format!("skip({n})"))
    }
    pub fn range(self, a: usize, b: usize) -> Self {
        self.step(&format!("range({a},{b})"))
    }
    pub fn range_local(self, a: usize, b: usize) -> Self {
        self.step(&format!("range(local,{a},{b})"))
    }
    pub fn tail(self, n: usize) -> Self {
        self.step(&format!("tail({n})"))
    }
    pub fn tail_local(self, n: usize) -> Self {
        self.step(&format!("tail(local,{n})"))
    }
    pub fn sample(self, n: usize) -> Self {
        self.step(&format!("sample({n})"))
    }

    // reducers
    pub fn count(self) -> Self {
        self.step("count()")
    }
    pub fn count_local(self) -> Self {
        self.step("count(local)")
    }
    pub fn fold(self) -> Self {
        self.step("fold()")
    }
    pub fn sum(self) -> Self {
        self.step("sum()")
    }
    pub fn sum_local(self) -> Self {
        self.step("sum(local)")
    }
    pub fn min(self) -> Self {
        self.step("min()")
    }
    pub fn min_local(self) -> Self {
        self.step("min(local)")
    }
    pub fn max(self) -> Self {
        self.step("max()")
    }
    pub fn max_local(self) -> Self {
        self.step("max(local)")
    }
    pub fn mean(self) -> Self {
        self.step("mean()")
    }
    pub fn mean_local(self) -> Self {
        self.step("mean(local)")
    }
    pub fn group(self) -> Self {
        self.step("group()")
    }
    pub fn group_count(self) -> Self {
        self.step("groupCount()")
    }

    // order
    pub fn order(self) -> Self {
        self.step("order()")
    }
    pub fn order_local(self) -> Self {
        self.step("order(local)")
    }
    pub fn order_dir(self, dir: Order, _scope: Scope) -> Self {
        self.step(&format!("order().by({})", dir_word(dir)))
    }
    pub fn order_by(self, key: &str, dir: Order) -> Self {
        self.step(&format!("order().by('{key}',{})", dir_word(dir)))
    }

    // branch / control
    pub fn union(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        self.step(&format!("union({frag})"))
    }
    pub fn coalesce(self, plans: Vec<Self>) -> Self {
        let frag = plans
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        self.step(&format!("coalesce({frag})"))
    }
    pub fn optional(self, sub: Self) -> Self {
        self.step(&format!("optional({})", sub.eng))
    }
    pub fn local(self, sub: Self) -> Self {
        self.step(&format!("local({})", sub.eng))
    }
    pub fn choose(self, test: Self, then_: Self) -> Self {
        self.step(&format!("choose({},{})", test.eng, then_.eng))
    }
    pub fn choose_else(self, test: Self, then_: Self, else_: Self) -> Self {
        self.step(&format!("choose({},{},{})", test.eng, then_.eng, else_.eng))
    }
    pub fn branch(self, test: Self) -> Self {
        self.step(&format!("branch({})", test.eng))
    }
    pub fn option(self, m: impl Into<GVal>, plan: Self) -> Self {
        self.step(&format!("option({},{})", val(&m.into()), plan.eng))
    }
    pub fn option_none(self, plan: Self) -> Self {
        self.step(&format!("option(none,{})", plan.eng))
    }
    pub fn map(self, sub: Self) -> Self {
        self.step(&format!("map({})", sub.eng))
    }
    pub fn flat_map(self, sub: Self) -> Self {
        self.step(&format!("flatMap({})", sub.eng))
    }
    pub fn filter(self, sub: Self) -> Self {
        self.step(&format!("filter({})", sub.eng))
    }
    pub fn side_effect(self, sub: Self) -> Self {
        self.step(&format!("sideEffect({})", sub.eng))
    }
    pub fn match_(self, patterns: Vec<Self>) -> Self {
        let frag = patterns
            .iter()
            .map(|p| p.eng.clone())
            .collect::<Vec<_>>()
            .join(",");
        self.step(&format!("match({frag})"))
    }

    // side effects / bags
    pub fn aggregate(self, key: &str) -> Self {
        self.step(&format!("aggregate('{key}')"))
    }
    pub fn store(self, key: &str) -> Self {
        self.step(&format!("store('{key}')"))
    }
    pub fn cap(self, key: &str) -> Self {
        self.step(&format!("cap('{key}')"))
    }
    pub fn subgraph(self, key: &str) -> Self {
        self.step(&format!("subgraph('{key}')"))
    }
    pub fn barrier(self) -> Self {
        self.step("barrier()")
    }

    // OLAP
    pub fn shortest_path(self) -> Self {
        self.step("shortestPath()")
    }
    pub fn shortest_path_to(self, target: Self) -> Self {
        // Engine has no target-filtered form; render the target selector so the engine
        // parser surfaces it as a gap rather than silently computing all paths.
        self.step(&format!("shortestPath().with(target,{})", target.eng))
    }
    pub fn from_tag(self, label: &str) -> Self {
        self.step(&format!("from('{label}')"))
    }
    pub fn page_rank(self, alpha: Option<f64>) -> Self {
        match alpha {
            Some(a) => self.step(&format!("pageRank({a})")),
            None => self.step("pageRank()"),
        }
    }
    pub fn connected_component(self) -> Self {
        self.step("connectedComponent()")
    }
    pub fn peer_pressure(self) -> Self {
        self.step("peerPressure()")
    }

    // repeat
    pub fn repeat(self, body: Self) -> Self {
        self.step(&format!("repeat({})", body.eng))
    }
    pub fn times(self, n: usize) -> Self {
        self.step(&format!("times({n})"))
    }
    pub fn until(self, cond: Self) -> Self {
        self.step(&format!("until({})", cond.eng))
    }
    pub fn emit(self, cond: Self) -> Self {
        self.step(&format!("emit({})", cond.eng))
    }
    pub fn emit_all(self) -> Self {
        self.step("emit()")
    }

    // tags / select
    pub fn as_(self, label: &str) -> Self {
        self.step(&format!("as('{label}')"))
    }
    pub fn select(self, labels_: &[&str]) -> Self {
        self.step(&format!("select({})", labels(labels_)))
    }
    pub fn select_pop(self, pop: Pop, labels_: &[&str]) -> Self {
        let pw = format!("{pop:?}");
        self.step(&format!("select({},{})", pw, labels(labels_)))
    }
    pub fn select_column(self, col: Column) -> Self {
        let cw = if matches!(col, Column::Keys) {
            "keys"
        } else {
            "values"
        };
        self.step(&format!("select({cw})"))
    }

    // misc
    pub fn unfold(self) -> Self {
        self.step("unfold()")
    }
    pub fn constant(self, v: impl Into<GVal>) -> Self {
        self.step(&format!("constant({})", val(&v.into())))
    }
    pub fn math(self, expr: &str) -> Self {
        self.step(&format!("math('{expr}')"))
    }
    pub fn identity(self) -> Self {
        self.step("identity()")
    }
    pub fn inject<V: Into<GVal>>(self, vs: impl IntoIterator<Item = V>) -> Self {
        let gs: Vec<GVal> = vs.into_iter().map(Into::into).collect();
        let frag = gs.iter().map(val).collect::<Vec<_>>().join(",");
        self.step(&format!("inject({frag})"))
    }
    pub fn none(self) -> Self {
        self.step("none()")
    }
    pub fn none_pred(self, pred: P) -> Self {
        self.step(&format!("none({})", pred.eng))
    }
    pub fn fail(self, msg: &str) -> Self {
        self.step(&format!("fail('{msg}')"))
    }
    pub fn index(self) -> Self {
        self.step("index()")
    }
    pub fn loops(self) -> Self {
        self.step("loops()")
    }

    // sack
    pub fn with_sack(self, init: impl Into<GVal>) -> Self {
        // withSack is a source prefix — it must precede the head. Model as prefix.
        let mut t = self;
        t.eng = t
            .eng
            .replacen("g", &format!("g.withSack({})", val(&init.into())), 1);
        t
    }
    pub fn sack(self) -> Self {
        self.step("sack()")
    }
    pub fn sack_op(self, op: SackOp) -> Self {
        let ow = match op {
            SackOp::Sum => "sum",
            SackOp::Mult => "mult",
            SackOp::Min => "min",
            SackOp::Max => "max",
            SackOp::Assign => "assign",
        };
        self.step(&format!("sack({ow})"))
    }

    pub fn by_label(self) -> Self {
        self.step("by(label)")
    }
    pub fn by_id(self) -> Self {
        self.step("by(id)")
    }
    pub fn by_token(self, tok: Token) -> Self {
        let w = match tok {
            Token::Id => "id",
            Token::Label => "label",
            Token::Key => "key",
            Token::Value => "value",
        };
        self.step(&format!("by({w})"))
    }

    // writes
    pub fn add_v(self, label: Option<&str>) -> Self {
        match label {
            Some(l) => self.step(&format!("addV('{l}')")),
            None => self.step("addV()"),
        }
    }
    pub fn add_e(self, label: &str) -> Self {
        self.step(&format!("addE('{label}')"))
    }
    pub fn property(self, key: &str, v: impl Into<GVal>) -> Self {
        self.step(&format!("property('{key}',{})", val(&v.into())))
    }
    pub fn drop(self) -> Self {
        self.step("drop()")
    }

    // OLAP config modulators (attach to the preceding algo step)
    pub fn with_algo_property(self, name: String) -> Self {
        self.step(&format!("with(propertyName,'{name}')"))
    }
    pub fn with_algo_times(self, n: u32) -> Self {
        self.step(&format!("with(times,{n})"))
    }

    /// Parse + run this traversal's engine query on `store`, returning the flattened
    /// results as `GVal`. The test body's own `assert_eq!` then checks the expected.
    pub fn run(self, store: &mut super::EngineGraph) -> Vec<GVal> {
        super::run_query(&self.eng, store)
    }
}
