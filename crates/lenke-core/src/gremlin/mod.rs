//! A Gremlin traversal engine over the same columnar [`Graph`](crate::graph).
//!
//! Peer to the [`gql`](crate::gql) engine: a different query language binding to
//! the one core. Mirrors the TS `@lenke/gremlin` package's model — a plan is
//! a flat list of [`Step`]s (plain data, no closures), predicates are data, and
//! the [`exec`] module runs the step pipeline over a stream of traversers.
//!
//! The DSL is a fluent builder: `g().V().has("name", P::eq("marko")).out(&["KNOWS"])
//! .values(&["name"])`. Anonymous sub-traversals (for `where`/`and`/`repeat`/
//! `project().by(...)`/…) start from [`__`].
//!
//! Closure-bearing steps (`map(fn)`/`filter(fn)`/…) are intentionally omitted:
//! the data-plan model uses sub-traversals instead, which express the same logic.

use std::sync::Arc;

pub mod exec;
pub mod parse;
#[cfg(test)]
mod ported_1;
#[cfg(test)]
mod ported_2;
#[cfg(test)]
mod ported_3;
#[cfg(test)]
mod ported_4;
#[cfg(test)]
mod ported_5;
#[cfg(test)]
mod ported_6;
#[cfg(test)]
mod ported_divergences;
#[cfg(test)]
mod tests;

pub use exec::{run, try_run};
pub use parse::parse;

/// A runtime traversal value. Graph elements are dense ids (like `gql::Val`);
/// `List`/`Map` carry `fold`/`valueMap`/`group`/`select`/`path` results.
///
/// `PartialEq` is hand-written (below) only so a `Property` compares by
/// key+value, ignoring its owner back-reference — the owner is internal
/// drop-routing metadata that is never observable (never serialized), matching
/// the TS engine. Every other variant compares structurally as `derive` would.
#[derive(Clone, Debug)]
pub enum GVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(Arc<str>),
    /// An ISO temporal scalar (`DATE`/`LOCAL DATETIME`/`DURATION`).
    Temporal(crate::temporal::Temporal),
    Vertex(u32),
    Edge(u32),
    List(Vec<Self>),
    /// Insertion-ordered key→value pairs (valueMap / group / select / project).
    Map(Vec<(Self, Self)>),
    /// A property element (from `.properties(k)`): its key/value plus a
    /// back-reference to the owning `Vertex`/`Edge`. The owner is carried
    /// EXPLICITLY (not recovered from the traverser path) so `.drop()` deletes
    /// exactly this property and can never mistake a `project('key')` Map for a
    /// property element. Serializes as `{key, value}` (TinkerPop's shape).
    Property {
        owner: Box<Self>,
        key: Arc<str>,
        value: Box<Self>,
    },
}

impl PartialEq for GVal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Num(a), Self::Num(b)) => a == b, // f64: NaN != NaN, as derive would
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::Temporal(a), Self::Temporal(b)) => a == b,
            (Self::Vertex(a), Self::Vertex(b)) => a == b,
            (Self::Edge(a), Self::Edge(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            (Self::Map(a), Self::Map(b)) => a == b,
            // Owner ignored: a property element's observable identity is its
            // key+value (the owner is internal drop-routing metadata).
            (
                Self::Property {
                    key: k1, value: v1, ..
                },
                Self::Property {
                    key: k2, value: v2, ..
                },
            ) => k1 == k2 && v1 == v2,
            _ => false,
        }
    }
}

impl From<f64> for GVal {
    fn from(n: f64) -> Self {
        Self::Num(n)
    }
}
impl From<i32> for GVal {
    fn from(n: i32) -> Self {
        Self::Num(n as f64)
    }
}
impl From<bool> for GVal {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}
impl From<&str> for GVal {
    fn from(s: &str) -> Self {
        Self::Str(Arc::from(s))
    }
}
impl From<String> for GVal {
    fn from(s: String) -> Self {
        Self::Str(Arc::from(s.as_str()))
    }
}

/// A predicate (data, not a closure) used by `has`/`is`/`where`. Mirrors
/// Gremlin's `P`/`TextP`.
#[derive(Clone, Debug)]
pub enum P {
    Eq(GVal),
    Neq(GVal),
    Gt(GVal),
    Gte(GVal),
    Lt(GVal),
    Lte(GVal),
    /// Inclusive min, exclusive max (Gremlin `between`).
    Between(GVal, GVal),
    /// Exclusive both ends (Gremlin `inside`).
    Inside(GVal, GVal),
    /// `value < min || value > max` (Gremlin `outside`).
    Outside(GVal, GVal),
    Within(Vec<GVal>),
    Without(Vec<GVal>),
    StartsWith(String),
    EndingWith(String),
    Containing(String),
    NotContaining(String),
    /// Match a value against a regular expression (Gremlin `TextP.regex`).
    Regex(String),
    Not(Box<Self>),
}

impl P {
    pub fn eq(v: impl Into<GVal>) -> Self {
        Self::Eq(v.into())
    }
    pub fn neq(v: impl Into<GVal>) -> Self {
        Self::Neq(v.into())
    }
    pub fn gt(v: impl Into<GVal>) -> Self {
        Self::Gt(v.into())
    }
    pub fn gte(v: impl Into<GVal>) -> Self {
        Self::Gte(v.into())
    }
    pub fn lt(v: impl Into<GVal>) -> Self {
        Self::Lt(v.into())
    }
    pub fn lte(v: impl Into<GVal>) -> Self {
        Self::Lte(v.into())
    }
    pub fn between(min: impl Into<GVal>, max: impl Into<GVal>) -> Self {
        Self::Between(min.into(), max.into())
    }
    pub fn inside(min: impl Into<GVal>, max: impl Into<GVal>) -> Self {
        Self::Inside(min.into(), max.into())
    }
    pub fn outside(min: impl Into<GVal>, max: impl Into<GVal>) -> Self {
        Self::Outside(min.into(), max.into())
    }
    pub fn within<V: Into<GVal>>(vs: impl IntoIterator<Item = V>) -> Self {
        Self::Within(vs.into_iter().map(Into::into).collect())
    }
    pub fn without<V: Into<GVal>>(vs: impl IntoIterator<Item = V>) -> Self {
        Self::Without(vs.into_iter().map(Into::into).collect())
    }
    pub fn starts_with(s: &str) -> Self {
        Self::StartsWith(s.to_string())
    }
    pub fn containing(s: &str) -> Self {
        Self::Containing(s.to_string())
    }
    pub fn ending_with(s: &str) -> Self {
        Self::EndingWith(s.to_string())
    }
    pub fn not_containing(s: &str) -> Self {
        Self::NotContaining(s.to_string())
    }
    pub fn regex(s: &str) -> Self {
        Self::Regex(s.to_string())
    }
    #[allow(
        clippy::should_implement_trait,
        reason = "P::not is the Gremlin predicate-negation constructor, not std::ops::Not"
    )]
    pub fn not(p: Self) -> Self {
        Self::Not(Box::new(p))
    }
    /// The single RHS value of a comparison predicate — for `where(start, pred)`,
    /// this names the end-tag label; `None` for range/set/text predicates.
    pub(crate) fn rhs(&self) -> Option<&GVal> {
        match self {
            Self::Eq(v)
            | Self::Neq(v)
            | Self::Gt(v)
            | Self::Gte(v)
            | Self::Lt(v)
            | Self::Lte(v) => Some(v),
            _ => None,
        }
    }
}

/// Sort direction for `order().by(...)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Order {
    Asc,
    Desc,
}

/// `select(pop, ...)` — which tagged value to recall when a label was set
/// multiple times (e.g. inside `repeat(out().as("a"))`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pop {
    First,
    Last,
    All,
}

/// Global (across the stream) vs local (over each traverser's iterable value).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scope {
    Global,
    Local,
}

/// A `T` token projected by a `by()` modulator (`by(T.id)` / `by(T.label)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Token {
    Id,
    Label,
    Key,
    Value,
}

/// A `Column` selector for a Map: its keys or its values. Used by
/// `order(local).by(values, desc)` / `.by(keys)` to sort a Map's entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Column {
    Keys,
    Values,
}

/// A `by()` modulator: how to project a value before a parent step uses it.
#[derive(Clone, Debug)]
pub enum By {
    Identity(Option<Order>),
    Key(String, Option<Order>),
    Token(Token, Option<Order>),
    /// A `Column.keys` / `Column.values` selector — in `order(local)` it sorts a
    /// Map's entries by their key or value.
    Column(Column, Option<Order>),
    Traversal(Box<Traversal>, Option<Order>),
}

impl By {
    fn direction(&self) -> Option<Order> {
        match self {
            Self::Identity(d)
            | Self::Key(_, d)
            | Self::Token(_, d)
            | Self::Column(_, d)
            | Self::Traversal(_, d) => *d,
        }
    }
}

/// An `addE` endpoint: the current traverser, a tagged vertex, or a sub-plan.
#[derive(Clone, Debug)]
pub enum Endpoint {
    Current,
    Tag(String),
    Plan(Box<Traversal>),
}

/// The value written by `property(key, …)`: either a literal, or a
/// traversal-induced value (`property('deg', __.outE().count())`) evaluated
/// against the current element per traverser (standard TinkerPop). The subplan
/// runs rooted at the current traverser; its first output is the value written,
/// and no output leaves the property unset.
#[derive(Clone, Debug)]
pub enum PropVal {
    Lit(GVal),
    Trav(Box<Traversal>),
}

/// One traversal step (plain data — serializable, reorderable).
#[derive(Clone, Debug)]
pub enum Step {
    V(Vec<String>),
    E(Vec<String>),
    Out(Vec<String>),
    In(Vec<String>),
    Both(Vec<String>),
    OutE(Vec<String>),
    InE(Vec<String>),
    BothE(Vec<String>),
    OutV,
    InV,
    OtherV,
    BothV,
    Has(String, P),
    HasLabel(Vec<String>),
    HasId(Vec<String>),
    HasKey(Vec<String>),
    HasNot(Vec<String>),
    HasValue(Vec<GVal>),
    Is(P),
    SimplePath,
    CyclicPath,
    /// `labels` (from `dedup('a','b')`) dedupes by the tuple of values tagged at
    /// those labels; otherwise `bys` (`dedup().by(...)`) or the current value.
    Dedupe {
        labels: Vec<String>,
        bys: Vec<By>,
    },
    Values(Vec<String>),
    ValueMap(Vec<String>),
    PropertyMap(Vec<String>),
    ElementMap(Vec<String>),
    Properties(Vec<String>),
    Value,
    Id,
    Label,
    Path(Vec<By>),
    Project(Vec<String>, Vec<By>),
    Tree(Vec<By>),
    Limit(usize, Scope),
    Skip(usize, Scope),
    Range(usize, usize, Scope),
    Tail(usize, Scope),
    Sample(usize),
    Count(Scope),
    Fold,
    Sum(Scope),
    Min(Scope),
    Max(Scope),
    Mean(Scope),
    Order(Vec<By>, bool, Scope),
    Group(Vec<By>),
    GroupCount(Vec<By>),
    Where(Box<Traversal>),
    WhereKey(String, P, Vec<By>),
    /// `where(neq('me'))` — a predicate-only where: compares the CURRENT traverser
    /// against the value tagged at the predicate's step-label operand.
    WherePred(P),
    And(Vec<Traversal>),
    Or(Vec<Traversal>),
    Not(Box<Traversal>),
    Union(Vec<Traversal>),
    Coalesce(Vec<Traversal>),
    Optional(Box<Traversal>),
    Local(Box<Traversal>),
    Choose {
        test: Box<Traversal>,
        then_: Box<Traversal>,
        else_: Option<Box<Traversal>>,
    },
    /// Switch over a sub-plan's first result: route each traverser to the first
    /// option whose `match` value equals that result, else `default`.
    Branch {
        test: Box<Traversal>,
        options: Vec<(GVal, Traversal)>,
        default: Option<Box<Traversal>>,
    },
    Map(Box<Traversal>),
    FlatMap(Box<Traversal>),
    SideEffect(Box<Traversal>),
    Aggregate(String),
    Store(String),
    Cap(String),
    /// Side-effect: accumulate matching edges (+ endpoints) into a named subgraph.
    Subgraph(String),
    /// Emit the shortest vertex path(s) from each source vertex; `target`
    /// (set via `.with(ShortestPath.target, …)`) filters destinations. `out`/`inn`
    /// pick which incident edges the BFS follows (both true = undirected, the
    /// TinkerPop default; set via `.with(ShortestPath.edges, Direction.OUT|IN|BOTH)`).
    ShortestPath {
        target: Option<Box<Traversal>>,
        out: bool,
        inn: bool,
    },
    /// Whole-graph OLAP algorithm steps (TinkerPop `pageRank()` / `connectedComponent()`
    /// / `peerPressure()`): compute over the whole graph, write each vertex's result to
    /// `property` (a TinkerPop-style default when unset), then pass the incoming stream
    /// through so downstream steps read the property. Run locally — a preceding
    /// `withComputer()` is accepted but the computation is always in-process.
    PageRank {
        property: Option<String>,
        times: Option<u32>,
        alpha: Option<f64>,
    },
    ConnectedComponent {
        property: Option<String>,
    },
    PeerPressure {
        property: Option<String>,
        times: Option<u32>,
    },
    Barrier,
    Repeat {
        body: Box<Traversal>,
        times: Option<usize>,
        until: Option<Box<Traversal>>,
        /// TinkerPop `until` placement: `false` = post-form `repeat(body).until()`
        /// (do-while — body runs at least once, condition checked AFTER it);
        /// `true` = pre-form `until().repeat(body)` (while-do — checked BEFORE).
        until_before: bool,
        emit: Option<Box<Traversal>>,
        emit_before: bool,
    },
    As(String),
    Select {
        labels: Vec<String>,
        pop: Pop,
        bys: Vec<By>,
    },
    /// `select(Column.keys)` / `select(Column.values)` — extract a Map's keys or
    /// values as a list, preserving entry order (the observable reader for
    /// `order(local)`, whose reordering a key-sorted Map serialization would hide).
    SelectColumn(Column),
    /// Declarative pattern matching: each sub-traversal is an `as(start) … [as(end)]`
    /// constraint; emits one traverser per consistent label assignment.
    Match(Vec<Traversal>),
    Unfold,
    Index,
    Loops,
    Constant(GVal),
    /// Evaluate a tiny infix arithmetic expression (`+ - * /`, parens, literals,
    /// `_` = current value, other idents = `as_`-bound labels). Operands are
    /// projected by the cycling `by()` modulators in first-appearance order.
    Math {
        expr: String,
        bys: Vec<By>,
    },
    Identity,
    Inject(Vec<GVal>),
    None(Option<P>),
    Fail(Option<String>),
    AddV(Option<String>),
    AddE {
        label: String,
        from: Endpoint,
        to: Endpoint,
    },
    Property(String, PropVal),
    Drop,
}

fn strs(labels: &[&str]) -> Vec<String> {
    labels.iter().map(|s| s.to_string()).collect()
}

/// A traversal plan: a fluent builder over a flat [`Step`] list.
#[derive(Clone, Debug, Default)]
pub struct Traversal {
    pub steps: Vec<Step>,
}

/// Start an anonymous (child) traversal for `where`/`and`/`repeat`/`by`/… sub-plans.
pub fn __() -> Traversal {
    Traversal::default()
}

/// Start a root traversal source `g`.
pub fn g() -> Traversal {
    Traversal::default()
}

impl Traversal {
    fn push(mut self, s: Step) -> Self {
        self.steps.push(s);
        self
    }

    /// Attach a `by()` modulator to the most recent modulator-bearing step.
    fn attach_by(mut self, by: By) -> Self {
        if let Some(
            Step::Order(bys, _, _)
            | Step::Group(bys)
            | Step::GroupCount(bys)
            | Step::Path(bys)
            | Step::Dedupe { bys, .. }
            | Step::Tree(bys)
            | Step::Project(_, bys)
            | Step::WhereKey(_, _, bys)
            | Step::Math { bys, .. }
            | Step::Select { bys, .. },
        ) = self.steps.last_mut()
        {
            bys.push(by);
        }
        self
    }

    // --- by() modulators ---
    pub fn by(self, key: &str) -> Self {
        self.attach_by(By::Key(key.to_string(), None))
    }
    pub fn by_identity(self) -> Self {
        self.attach_by(By::Identity(None))
    }
    pub fn by_token(self, tok: Token) -> Self {
        self.attach_by(By::Token(tok, None))
    }
    pub fn by_column(self, col: Column) -> Self {
        self.attach_by(By::Column(col, None))
    }
    pub fn by_column_dir(self, col: Column, dir: Order) -> Self {
        self.attach_by(By::Column(col, Some(dir)))
    }
    pub fn by_id(self) -> Self {
        self.attach_by(By::Token(Token::Id, None))
    }
    pub fn by_label(self) -> Self {
        self.attach_by(By::Token(Token::Label, None))
    }
    pub fn by_t(self, t: Self) -> Self {
        self.attach_by(By::Traversal(Box::new(t), None))
    }
    pub fn by_dir(self, key: &str, dir: Order) -> Self {
        self.attach_by(By::Key(key.to_string(), Some(dir)))
    }
    pub fn by_identity_dir(self, dir: Order) -> Self {
        self.attach_by(By::Identity(Some(dir)))
    }
    pub fn by_t_dir(self, t: Self, dir: Order) -> Self {
        self.attach_by(By::Traversal(Box::new(t), Some(dir)))
    }

    // --- sources ---
    #[allow(non_snake_case)]
    pub fn V(self) -> Self {
        self.push(Step::V(vec![]))
    }
    pub fn v_ids(self, ids: &[&str]) -> Self {
        self.push(Step::V(strs(ids)))
    }
    #[allow(non_snake_case)]
    pub fn E(self) -> Self {
        self.push(Step::E(vec![]))
    }
    pub fn e_ids(self, ids: &[&str]) -> Self {
        self.push(Step::E(strs(ids)))
    }

    // --- movement ---
    pub fn out(self, labels: &[&str]) -> Self {
        self.push(Step::Out(strs(labels)))
    }
    pub fn in_(self, labels: &[&str]) -> Self {
        self.push(Step::In(strs(labels)))
    }
    pub fn both(self, labels: &[&str]) -> Self {
        self.push(Step::Both(strs(labels)))
    }
    pub fn out_e(self, labels: &[&str]) -> Self {
        self.push(Step::OutE(strs(labels)))
    }
    pub fn in_e(self, labels: &[&str]) -> Self {
        self.push(Step::InE(strs(labels)))
    }
    pub fn both_e(self, labels: &[&str]) -> Self {
        self.push(Step::BothE(strs(labels)))
    }
    pub fn out_v(self) -> Self {
        self.push(Step::OutV)
    }
    pub fn in_v(self) -> Self {
        self.push(Step::InV)
    }
    pub fn other_v(self) -> Self {
        self.push(Step::OtherV)
    }
    pub fn both_v(self) -> Self {
        self.push(Step::BothV)
    }

    // --- filters ---
    pub fn has(self, key: &str, pred: P) -> Self {
        self.push(Step::Has(key.to_string(), pred))
    }
    pub fn has_val(self, key: &str, v: impl Into<GVal>) -> Self {
        self.push(Step::Has(key.to_string(), P::Eq(v.into())))
    }
    /// `has(label, key, pred)` — `hasLabel(label).has(key, pred)`.
    pub fn has_label_key(self, label: &str, key: &str, pred: P) -> Self {
        self.push(Step::HasLabel(vec![label.to_string()]))
            .push(Step::Has(key.to_string(), pred))
    }
    pub fn has_label(self, labels: &[&str]) -> Self {
        self.push(Step::HasLabel(strs(labels)))
    }
    pub fn has_id(self, ids: &[&str]) -> Self {
        self.push(Step::HasId(strs(ids)))
    }
    pub fn has_key(self, keys: &[&str]) -> Self {
        self.push(Step::HasKey(strs(keys)))
    }
    pub fn has_not(self, keys: &[&str]) -> Self {
        self.push(Step::HasNot(strs(keys)))
    }
    pub fn has_value<V: Into<GVal>>(self, vs: impl IntoIterator<Item = V>) -> Self {
        self.push(Step::HasValue(vs.into_iter().map(Into::into).collect()))
    }
    pub fn is(self, pred: P) -> Self {
        self.push(Step::Is(pred))
    }
    pub fn dedup(self) -> Self {
        self.push(Step::Dedupe {
            labels: vec![],
            bys: vec![],
        })
    }
    /// `dedup('a','b')` — dedupe by the tuple of values tagged at those labels.
    pub fn dedup_labels(self, labels: Vec<String>) -> Self {
        self.push(Step::Dedupe {
            labels,
            bys: vec![],
        })
    }
    pub fn simple_path(self) -> Self {
        self.push(Step::SimplePath)
    }
    pub fn cyclic_path(self) -> Self {
        self.push(Step::CyclicPath)
    }

    // --- projection ---
    pub fn values(self, keys: &[&str]) -> Self {
        self.push(Step::Values(strs(keys)))
    }
    pub fn value_map(self, keys: &[&str]) -> Self {
        self.push(Step::ValueMap(strs(keys)))
    }
    pub fn property_map(self, keys: &[&str]) -> Self {
        self.push(Step::PropertyMap(strs(keys)))
    }
    pub fn element_map(self, keys: &[&str]) -> Self {
        self.push(Step::ElementMap(strs(keys)))
    }
    pub fn properties(self, keys: &[&str]) -> Self {
        self.push(Step::Properties(strs(keys)))
    }
    pub fn value(self) -> Self {
        self.push(Step::Value)
    }
    pub fn id(self) -> Self {
        self.push(Step::Id)
    }
    pub fn label(self) -> Self {
        self.push(Step::Label)
    }
    pub fn path(self) -> Self {
        self.push(Step::Path(vec![]))
    }
    pub fn project(self, keys: &[&str]) -> Self {
        self.push(Step::Project(strs(keys), vec![]))
    }
    pub fn tree(self) -> Self {
        self.push(Step::Tree(vec![]))
    }

    // --- cardinality ---
    pub fn limit(self, n: usize) -> Self {
        self.push(Step::Limit(n, Scope::Global))
    }
    pub fn limit_local(self, n: usize) -> Self {
        self.push(Step::Limit(n, Scope::Local))
    }
    pub fn skip(self, n: usize) -> Self {
        self.push(Step::Skip(n, Scope::Global))
    }
    pub fn skip_local(self, n: usize) -> Self {
        self.push(Step::Skip(n, Scope::Local))
    }
    pub fn range(self, start: usize, end: usize) -> Self {
        self.push(Step::Range(start, end, Scope::Global))
    }
    pub fn range_local(self, start: usize, end: usize) -> Self {
        self.push(Step::Range(start, end, Scope::Local))
    }
    pub fn tail(self, n: usize) -> Self {
        self.push(Step::Tail(n, Scope::Global))
    }
    pub fn tail_local(self, n: usize) -> Self {
        self.push(Step::Tail(n, Scope::Local))
    }
    pub fn sample(self, n: usize) -> Self {
        self.push(Step::Sample(n))
    }

    // --- terminals / aggregates ---
    pub fn count(self) -> Self {
        self.push(Step::Count(Scope::Global))
    }
    pub fn count_local(self) -> Self {
        self.push(Step::Count(Scope::Local))
    }
    pub fn fold(self) -> Self {
        self.push(Step::Fold)
    }
    pub fn sum(self) -> Self {
        self.push(Step::Sum(Scope::Global))
    }
    pub fn sum_local(self) -> Self {
        self.push(Step::Sum(Scope::Local))
    }
    pub fn min(self) -> Self {
        self.push(Step::Min(Scope::Global))
    }
    pub fn min_local(self) -> Self {
        self.push(Step::Min(Scope::Local))
    }
    pub fn max(self) -> Self {
        self.push(Step::Max(Scope::Global))
    }
    pub fn max_local(self) -> Self {
        self.push(Step::Max(Scope::Local))
    }
    pub fn mean(self) -> Self {
        self.push(Step::Mean(Scope::Global))
    }
    pub fn mean_local(self) -> Self {
        self.push(Step::Mean(Scope::Local))
    }
    pub fn order(self) -> Self {
        self.push(Step::Order(vec![], false, Scope::Global))
    }
    pub fn order_local(self) -> Self {
        self.push(Step::Order(vec![], false, Scope::Local))
    }
    pub fn order_by(self, key: &str, dir: Order) -> Self {
        self.push(Step::Order(
            vec![By::Key(key.to_string(), Some(dir))],
            false,
            Scope::Global,
        ))
    }
    pub fn group(self) -> Self {
        self.push(Step::Group(vec![]))
    }
    pub fn group_count(self) -> Self {
        self.push(Step::GroupCount(vec![]))
    }

    // --- branch / combinators ---
    pub fn where_(self, sub: Self) -> Self {
        self.push(Step::Where(Box::new(sub)))
    }
    pub fn where_key(self, start: &str, pred: P) -> Self {
        self.push(Step::WhereKey(start.to_string(), pred, vec![]))
    }
    pub fn where_pred(self, pred: P) -> Self {
        self.push(Step::WherePred(pred))
    }
    pub fn and(self, plans: Vec<Self>) -> Self {
        self.push(Step::And(plans))
    }
    pub fn or(self, plans: Vec<Self>) -> Self {
        self.push(Step::Or(plans))
    }
    pub fn not(self, sub: Self) -> Self {
        self.push(Step::Not(Box::new(sub)))
    }
    pub fn union(self, plans: Vec<Self>) -> Self {
        self.push(Step::Union(plans))
    }
    pub fn coalesce(self, plans: Vec<Self>) -> Self {
        self.push(Step::Coalesce(plans))
    }
    pub fn optional(self, sub: Self) -> Self {
        self.push(Step::Optional(Box::new(sub)))
    }
    pub fn local(self, sub: Self) -> Self {
        self.push(Step::Local(Box::new(sub)))
    }
    pub fn choose(self, test: Self, then_: Self) -> Self {
        self.push(Step::Choose {
            test: Box::new(test),
            then_: Box::new(then_),
            else_: None,
        })
    }
    pub fn choose_else(self, test: Self, then_: Self, else_: Self) -> Self {
        self.push(Step::Choose {
            test: Box::new(test),
            then_: Box::new(then_),
            else_: Some(Box::new(else_)),
        })
    }
    pub fn branch(self, test: Self) -> Self {
        self.push(Step::Branch {
            test: Box::new(test),
            options: vec![],
            default: None,
        })
    }
    /// Attach an `option(match, traversal)` to the most recent `branch()`.
    pub fn option(mut self, m: impl Into<GVal>, plan: Self) -> Self {
        if let Some(Step::Branch { options, .. }) = self.steps.last_mut() {
            options.push((m.into(), plan));
        }
        self
    }
    /// Attach the default branch (`option(none, traversal)`) to `branch()`.
    pub fn option_none(mut self, plan: Self) -> Self {
        if let Some(Step::Branch { default, .. }) = self.steps.last_mut() {
            *default = Some(Box::new(plan));
        }
        self
    }
    pub fn map(self, sub: Self) -> Self {
        self.push(Step::Map(Box::new(sub)))
    }
    pub fn flat_map(self, sub: Self) -> Self {
        self.push(Step::FlatMap(Box::new(sub)))
    }
    pub fn filter(self, sub: Self) -> Self {
        self.push(Step::Where(Box::new(sub)))
    }
    pub fn side_effect(self, sub: Self) -> Self {
        self.push(Step::SideEffect(Box::new(sub)))
    }
    pub fn aggregate(self, key: &str) -> Self {
        self.push(Step::Aggregate(key.to_string()))
    }
    pub fn store(self, key: &str) -> Self {
        self.push(Step::Store(key.to_string()))
    }
    pub fn cap(self, key: &str) -> Self {
        self.push(Step::Cap(key.to_string()))
    }
    pub fn subgraph(self, key: &str) -> Self {
        self.push(Step::Subgraph(key.to_string()))
    }
    pub fn shortest_path(self) -> Self {
        self.push(Step::ShortestPath {
            target: None,
            out: true,
            inn: true,
        })
    }
    /// `shortestPath().with(ShortestPath.target, target)` — restrict destinations.
    pub fn shortest_path_to(self, target: Self) -> Self {
        self.push(Step::ShortestPath {
            target: Some(Box::new(target)),
            out: true,
            inn: true,
        })
    }
    /// Set the target sub-traversal on the most recent `shortestPath` step (the
    /// textual `.with(ShortestPath.target, …)` modulator).
    pub fn with_shortest_path_target(mut self, target: Self) -> Self {
        if let Some(Step::ShortestPath { target: tgt, .. }) = self.steps.last_mut() {
            *tgt = Some(Box::new(target));
        }
        self
    }
    /// Set which incident edges the most recent `shortestPath` step follows, from the
    /// conformant `.with(ShortestPath.edges, Direction.OUT|IN|BOTH)` modulator (`BOTH`
    /// = undirected, the TinkerPop default). Rejects any non-`Direction` value.
    pub fn with_shortest_path_edges(mut self, edges: &str) -> Result<Self, String> {
        let (out, inn) = match edges {
            "Direction.OUT" => (true, false),
            "Direction.IN" => (false, true),
            "Direction.BOTH" => (true, true),
            _ => {
                return Err(
                    "with(ShortestPath.edges, …): expected Direction.OUT, Direction.IN, or Direction.BOTH".into(),
                );
            }
        };
        if let Some(Step::ShortestPath { out: o, inn: i, .. }) = self.steps.last_mut() {
            (*o, *i) = (out, inn);
        }
        Ok(self)
    }
    pub fn page_rank(self, alpha: Option<f64>) -> Self {
        self.push(Step::PageRank {
            property: None,
            times: None,
            alpha,
        })
    }
    pub fn connected_component(self) -> Self {
        self.push(Step::ConnectedComponent { property: None })
    }
    pub fn peer_pressure(self) -> Self {
        self.push(Step::PeerPressure {
            property: None,
            times: None,
        })
    }
    /// `.with(<Algo>.propertyName, name)` — set where the last algorithm step writes.
    pub fn with_algo_property(mut self, name: String) -> Self {
        if let Some(
            Step::PageRank { property, .. }
            | Step::ConnectedComponent { property }
            | Step::PeerPressure { property, .. },
        ) = self.steps.last_mut()
        {
            *property = Some(name);
        }
        self
    }
    /// `.with(<Algo>.times, n)` — set the iteration count for the last algorithm step.
    pub fn with_algo_times(mut self, n: u32) -> Self {
        if let Some(Step::PageRank { times, .. } | Step::PeerPressure { times, .. }) =
            self.steps.last_mut()
        {
            *times = Some(n);
        }
        self
    }
    pub fn barrier(self) -> Self {
        self.push(Step::Barrier)
    }
    pub fn repeat(self, body: Self) -> Self {
        self.push(Step::Repeat {
            body: Box::new(body),
            times: None,
            until: None,
            until_before: false,
            emit: None,
            emit_before: false,
        })
    }
    pub fn times(mut self, n: usize) -> Self {
        if let Some(Step::Repeat { times, .. }) = self.steps.last_mut() {
            *times = Some(n);
        }
        self
    }
    pub fn until(mut self, cond: Self) -> Self {
        // Post-form `repeat(body).until(cond)` — do-while (until_before stays false).
        if let Some(Step::Repeat { until, .. }) = self.steps.last_mut() {
            *until = Some(Box::new(cond));
        }
        self
    }
    pub fn until_before(mut self, cond: Self) -> Self {
        // Pre-form `until(cond).repeat(body)` — while-do.
        if let Some(Step::Repeat {
            until,
            until_before,
            ..
        }) = self.steps.last_mut()
        {
            *until = Some(Box::new(cond));
            *until_before = true;
        }
        self
    }
    pub fn emit(mut self, cond: Self) -> Self {
        if let Some(Step::Repeat { emit, .. }) = self.steps.last_mut() {
            *emit = Some(Box::new(cond));
        }
        self
    }
    /// Empty-condition emit (`emit()` with no filter — emit every body output).
    pub fn emit_all(mut self) -> Self {
        if let Some(Step::Repeat { emit, .. }) = self.steps.last_mut() {
            *emit = Some(Box::new(Self::default()));
        }
        self
    }
    pub fn emit_before(mut self, cond: Self) -> Self {
        if let Some(Step::Repeat {
            emit, emit_before, ..
        }) = self.steps.last_mut()
        {
            *emit = Some(Box::new(cond));
            *emit_before = true;
        }
        self
    }

    // --- tagging / select ---
    pub fn as_(self, label: &str) -> Self {
        self.push(Step::As(label.to_string()))
    }
    pub fn select(self, labels: &[&str]) -> Self {
        self.push(Step::Select {
            labels: strs(labels),
            pop: Pop::Last,
            bys: vec![],
        })
    }
    pub fn select_pop(self, pop: Pop, labels: &[&str]) -> Self {
        self.push(Step::Select {
            labels: strs(labels),
            pop,
            bys: vec![],
        })
    }
    pub fn select_column(self, col: Column) -> Self {
        self.push(Step::SelectColumn(col))
    }
    pub fn match_(self, patterns: Vec<Self>) -> Self {
        self.push(Step::Match(patterns))
    }

    // --- misc ---
    pub fn unfold(self) -> Self {
        self.push(Step::Unfold)
    }
    pub fn index(self) -> Self {
        self.push(Step::Index)
    }
    pub fn loops(self) -> Self {
        self.push(Step::Loops)
    }
    pub fn constant(self, v: impl Into<GVal>) -> Self {
        self.push(Step::Constant(v.into()))
    }
    pub fn math(self, expr: &str) -> Self {
        self.push(Step::Math {
            expr: expr.to_string(),
            bys: vec![],
        })
    }
    pub fn identity(self) -> Self {
        self.push(Step::Identity)
    }
    pub fn inject<V: Into<GVal>>(self, vs: impl IntoIterator<Item = V>) -> Self {
        self.push(Step::Inject(vs.into_iter().map(Into::into).collect()))
    }
    pub fn none(self) -> Self {
        self.push(Step::None(None))
    }
    pub fn none_pred(self, pred: P) -> Self {
        self.push(Step::None(Some(pred)))
    }
    pub fn fail(self, msg: &str) -> Self {
        self.push(Step::Fail(Some(msg.to_string())))
    }

    // --- mutation ---
    pub fn add_v(self, label: Option<&str>) -> Self {
        self.push(Step::AddV(label.map(str::to_string)))
    }
    pub fn add_e(self, label: &str) -> Self {
        self.push(Step::AddE {
            label: label.to_string(),
            from: Endpoint::Current,
            to: Endpoint::Current,
        })
    }
    pub fn from_tag(mut self, label: &str) -> Self {
        if let Some(Step::AddE { from, .. }) = self.steps.last_mut() {
            *from = Endpoint::Tag(label.to_string());
        }
        self
    }
    pub fn to_tag(mut self, label: &str) -> Self {
        if let Some(Step::AddE { to, .. }) = self.steps.last_mut() {
            *to = Endpoint::Tag(label.to_string());
        }
        self
    }
    pub fn from_plan(mut self, plan: Self) -> Self {
        if let Some(Step::AddE { from, .. }) = self.steps.last_mut() {
            *from = Endpoint::Plan(Box::new(plan));
        }
        self
    }
    pub fn to_plan(mut self, plan: Self) -> Self {
        if let Some(Step::AddE { to, .. }) = self.steps.last_mut() {
            *to = Endpoint::Plan(Box::new(plan));
        }
        self
    }
    pub fn property(self, key: &str, v: impl Into<GVal>) -> Self {
        self.push(Step::Property(key.to_string(), PropVal::Lit(v.into())))
    }
    pub fn property_trav(self, key: &str, t: Self) -> Self {
        self.push(Step::Property(key.to_string(), PropVal::Trav(Box::new(t))))
    }
    pub fn drop(self) -> Self {
        self.push(Step::Drop)
    }

    /// Run against `graph` (mutable, since `addV`/`addE`/`property`/`drop` mutate;
    /// read-only traversals just don't touch it — same convention as `gql::execute`).
    pub fn run(&self, graph: &mut crate::graph::Graph) -> Vec<GVal> {
        run(graph, self)
    }
}
