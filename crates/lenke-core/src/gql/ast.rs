//! GQL query AST — a faithful Rust port of the TS `ast.ts`. Plain data only: a
//! parsed query describes *what* to match, never *how*. The surface is the ISO
//! GQL core (`MATCH`/`WHERE`/`RETURN`, ISO ASCII-art patterns, boolean-algebra
//! label expressions, set operators, `WITH`). Comments and semantics track the
//! TS source so the two stay in lockstep.

/// A whole query: one or more linear queries combined by set operators
/// (`p0 UNION p1 EXCEPT p2`, left-associative). `ops[i]` joins `parts[i]` to
/// `parts[i + 1]`, so `ops.len() == parts.len() - 1`.
#[derive(Debug, Clone)]
pub struct Query {
    pub parts: Vec<LinearQuery>,
    pub ops: Vec<SetOp>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SetOpKind {
    Union,
    Except,
    Intersect,
}

/// `UNION` / `EXCEPT` / `INTERSECT`, optionally `ALL` (keep duplicates).
#[derive(Debug, Clone, Copy)]
pub struct SetOp {
    pub op: SetOpKind,
    pub all: bool,
}

/// A linear query: a sequence of clauses ending in `RETURN`/`FINISH`.
#[derive(Debug, Clone)]
pub struct LinearQuery {
    pub clauses: Vec<Clause>,
}

/// A parsed top-level statement: either a linear (pattern) [`Query`] or an ISO
/// GQL transaction-control command. [`super::parse`] returns this; the FFI query
/// path dispatches on the variant (a query lowers + runs; a [`TxControl`] drives
/// the session's transaction frame). Mirrors the TS `Statement` union.
#[derive(Debug, Clone)]
pub enum Statement {
    Query(Query),
    Tx(TxControl),
}

/// ISO/IEC 39075 transaction-control command: `START TRANSACTION [READ ONLY |
/// READ WRITE]`, `COMMIT [WORK]`, `ROLLBACK [WORK]`. Carries no clauses — it drives
/// `Graph::begin_tx`/`commit_tx`/`rollback_tx`. `access_mode` is only meaningful
/// for `Start` (defaults to READ WRITE when omitted). Mirrors the TS `TxControl`.
#[derive(Debug, Clone)]
pub struct TxControl {
    pub kind: TxKind,
    pub access_mode: Option<AccessMode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxKind {
    Start,
    Commit,
    Rollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub enum Clause {
    Match(MatchClause),
    With(WithClause),
    Return(Projection),
    /// `INSERT pattern, …`
    Insert(Vec<PathPattern>),
    /// `_MERGE pattern [_ON_CREATE SET …] [_ON_UPDATE …]` — the lenke keyed-upsert
    /// extension (NOT ISO GQL; sigil-marked, recognized only under the `Lenke`
    /// dialect). See docs/design/gql-extensions.md §2.
    Merge(MergeClause),
    /// `FOR x IN <list> [WITH ORDINALITY|OFFSET n]` — ISO GQL list unwind (the
    /// standard's equivalent of Cypher `UNWIND`). Multiplies the row table by the
    /// list. Bare ISO syntax (no sigil), accepted under every dialect.
    For(ForClause),
    /// `FILTER [WHERE] <condition>` — ISO GQL §14.6 `<filter statement>`. Drops
    /// rows from the working table where the condition is not TRUE (three-valued,
    /// exactly like a `WHERE`). The `WHERE` keyword is optional.
    Filter(Expr),
    /// `LET x = e, y = e, …` — ISO GQL §14.7 `<let statement>`. Binds new value
    /// variables into the current scope (additive: existing bindings are kept).
    /// Bindings are evaluated left-to-right, so a later item may reference an
    /// earlier one (`LET x = 1, y = x + 1`).
    Let(Vec<LetItem>),
    /// `SET n.key = v` / `SET n:Label`
    Set(Vec<SetItem>),
    /// `REMOVE n.key` / `REMOVE n:Label`
    Remove(Vec<RemoveItem>),
    /// `[DETACH] DELETE n, …`
    Delete {
        detach: bool,
        targets: Vec<Expr>,
    },
    /// `FINISH` — run for side effects, return nothing.
    Finish,
    /// `[OPTIONAL] CALL name(args) [YIELD col [AS alias], …]` — an ISO GQL named
    /// procedure call (§`callProcedureStatement` → `namedProcedureCall`). Invokes a
    /// catalog procedure (here: the built-in graph algorithms); `yields` picks and
    /// renames its output columns (`None` = every column, under its own name).
    CallNamed(CallNamed),
    /// `[OPTIONAL] CALL (scope) { … }` — an ISO GQL inline procedure call
    /// (§`inlineProcedureCall`). Runs the nested query once per incoming row
    /// (correlated / lateral), importing only the `scope` variables, and merges
    /// its `RETURN` columns back. OPTIONAL keeps the outer row (nested columns
    /// null-filled) when the subquery is empty.
    CallInline(CallInline),
}

/// One binding of a `LET` clause: `variable = expression`.
#[derive(Debug, Clone)]
pub struct LetItem {
    pub var: String,
    pub expr: Expr,
}

/// An inline procedure call (`CALL (scope) { <nested query> }`).
#[derive(Debug, Clone)]
pub struct CallInline {
    pub optional: bool,
    /// Outer variables the subquery imports (`(a, b)`); empty = none / `()`.
    pub scope: Vec<String>,
    /// The nested query: one or more linear parts joined by set operators
    /// (`UNION`/`EXCEPT`/`INTERSECT`). A plain body is a single part, no ops.
    pub body: Query,
}

/// A named procedure call (`CALL name(args) YIELD …`).
#[derive(Debug, Clone)]
pub struct CallNamed {
    /// `OPTIONAL CALL` — keep the outer row (null-filled) if the call is empty.
    pub optional: bool,
    /// Procedure name (a dotted `parent.name` is joined with `.`).
    pub name: String,
    /// The procedure's configuration, written as a `{key: value}` map argument
    /// (empty if the call is `name()`). For the graph-algorithm procedures these
    /// are the algorithm's config fields (`iterations`, `writeProperty`, …).
    pub config: Vec<PropertyConstraint>,
    pub yields: Option<Vec<YieldItem>>,
}

/// One `YIELD` output item: an output column, optionally renamed with `AS`.
#[derive(Debug, Clone)]
pub struct YieldItem {
    pub name: String,
    pub alias: Option<String>,
}

/// Parse dialect: `Lenke` permits sigil extensions (`_MERGE`); `IsoStrict` rejects
/// them (they stay ordinary identifiers). See docs/design/gql-extensions.md §1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dialect {
    #[default]
    Lenke,
    IsoStrict,
}

/// `_MERGE pattern [_ON_CREATE SET …] [_ON_UPDATE SET … [WHERE p] |
/// _ON_UPDATE_NOTHING]` — v1 upserts a single element keyed by a unique
/// constraint; absent `on_update` = clobber the pattern's payload.
#[derive(Debug, Clone)]
pub struct MergeClause {
    pub pattern: PathPattern,
    pub on_create: Option<Vec<SetItem>>,
    pub on_update: Option<MergeUpdate>,
}

#[derive(Debug, Clone)]
pub enum MergeUpdate {
    /// `_ON_UPDATE SET … [WHERE p]` — replaces the default clobber; runs only if `where_` holds.
    Set {
        items: Vec<SetItem>,
        where_: Option<Expr>,
    },
    /// `_ON_UPDATE_NOTHING` — leave the existing element untouched.
    Nothing,
}

/// `[OPTIONAL] MATCH p1, p2, … [WHERE pred]`.
#[derive(Debug, Clone)]
pub struct MatchClause {
    pub optional: bool,
    pub patterns: Vec<PathPattern>,
    pub where_: Option<Expr>,
}

/// `WITH … [WHERE pred]` — a projection that flows into the next clause.
#[derive(Debug, Clone)]
pub struct WithClause {
    pub projection: Projection,
    pub where_: Option<Expr>,
}

/// `FOR <alias> IN <list> [WITH ORDINALITY|OFFSET <var>]` — unwind a list into
/// one row per element (ISO GQL's UNWIND). The list is evaluated in the scope
/// *before* `alias` is bound, so it cannot reference the alias.
#[derive(Debug, Clone)]
pub struct ForClause {
    pub alias: String,
    pub list: Expr,
    pub ordinal: Option<ForOrdinal>,
}

/// The optional `WITH ORDINALITY <var>` (1-based index) or `WITH OFFSET <var>`
/// (0-based index) counter bound alongside each unwound element.
#[derive(Debug, Clone)]
pub struct ForOrdinal {
    pub kind: OrdKind,
    pub var: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrdKind {
    /// `WITH ORDINALITY` — counts from 1.
    Ordinality,
    /// `WITH OFFSET` — counts from 0.
    Offset,
}

#[derive(Debug, Clone)]
pub enum SetItem {
    Prop {
        variable: String,
        key: String,
        value: Expr,
    },
    Label {
        variable: String,
        label: String,
    },
}

#[derive(Debug, Clone)]
pub enum RemoveItem {
    Prop { variable: String, key: String },
    Label { variable: String, label: String },
}

/// How many of the paths matching a pattern to keep. `Walk` (the ISO default,
/// no selector, and the target of bare `ALL`) keeps every match; `Any` (bare
/// `ANY`) keeps one arbitrary path per endpoint; `AnyShortest` (`ANY SHORTEST`)
/// keeps one fewest-hop path per endpoint pair; `AllShortest` (`ALL SHORTEST`)
/// keeps every path tied for that fewest-hop length; `ShortestK` (`SHORTEST k
/// [GROUP]`) keeps the k shortest paths per endpoint — or, with `group`, every
/// path in the k smallest length-groups.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PathSelector {
    #[default]
    Walk,
    Any,
    AnyShortest,
    AllShortest,
    ShortestK {
        k: u32,
        group: bool,
    },
}

/// A path **mode** (restrictor): which element repeats a matched path may
/// contain. `Trail` (no repeated EDGES) is lenke's default — the spec's nominal
/// `Walk` default is unusable unbounded, so, like Neo4j and Microsoft Fabric, we
/// default to `Trail`. `Walk` allows repeats (bounded only by the quantifier);
/// `Acyclic` forbids any repeated NODE; `Simple` forbids repeated nodes except a
/// closing `start == end` (a cycle back to the seed).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PathMode {
    Walk,
    #[default]
    Trail,
    Simple,
    Acyclic,
}

/// A linear path pattern: a start node followed by `(rel)(node)` segments,
/// optionally bound to a `path_var` (`p = …`) and prefixed by a `selector`
/// (`ANY SHORTEST …`) and/or a `mode` (`TRAIL …`).
#[derive(Debug, Clone)]
pub struct PathPattern {
    pub start: NodePattern,
    pub segments: Vec<Segment>,
    pub path_var: Option<String>,
    pub selector: PathSelector,
    pub mode: PathMode,
}

/// One hop: traverse `rel`, land on `node`.
#[derive(Debug, Clone)]
pub struct Segment {
    pub rel: RelPattern,
    pub node: NodePattern,
    /// For an ISO quantified PARENTHESIZED subpath `((x)-[e]->(y) WHERE …){n,m}`:
    /// the inner FROM-node `(x)` of each repetition, bound PER-HOP so the
    /// per-repetition predicate (`rel.where_`) can reference the hop's source node
    /// (`node` is the per-hop TARGET `y`). `None` for a plain hop or the abbreviated
    /// `-[e]->{n,m}` form (where the source is just the previous node, unnamed).
    /// Group-variable exposure of the inner vars is a later phase.
    pub hop_from: Option<NodePattern>,
}

/// `(variable:LabelExpr {props} WHERE pred)` — all parts optional.
#[derive(Debug, Clone, Default)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub label: Option<LabelExpr>,
    pub props: Vec<PropertyConstraint>,
    pub where_: Option<Expr>,
}

/// One `key: valueExpression` entry of a pattern property map.
#[derive(Debug, Clone)]
pub struct PropertyConstraint {
    pub key: String,
    pub value: Expr,
}

/// An ISO label expression (boolean algebra: `A&B`, `A|B`, `!A`, `%`).
#[derive(Debug, Clone)]
pub enum LabelExpr {
    Label(String),
    Wildcard,
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Out,
    In,
    Both,
}

/// Variable-length quantifier: `*`={0,∞}, `+`={1,∞}, `{n}`, `{n,m}`.
#[derive(Debug, Clone, Copy)]
pub struct Quantifier {
    pub min: u32,
    pub max: Option<u32>,
}

/// A relationship pattern with a direction (see TS `RelPattern`).
#[derive(Debug, Clone)]
pub struct RelPattern {
    pub variable: Option<String>,
    pub label: Option<LabelExpr>,
    pub direction: Direction,
    pub props: Vec<PropertyConstraint>,
    pub where_: Option<Expr>,
    pub quantifier: Option<Quantifier>,
}

/// A `LIMIT` / `OFFSET` bound. ISO GQL's `nonNegativeIntegerSpecification`
/// (opengql:2268) is `unsignedInteger | dynamicParameterSpecification`, so the
/// bound is either an integer literal or a `$param` resolved — and validated to
/// be a non-negative integer — at execution. `SKIP` is the Cypher spelling of
/// `OFFSET` and accepts only a literal (a `$param` after `SKIP` is rejected).
#[derive(Debug, Clone)]
pub enum CountBound {
    Lit(usize),
    Param(String),
}

/// A projection body shared by `RETURN` and `WITH`.
#[derive(Debug, Clone)]
pub struct Projection {
    pub star: bool,
    pub items: Vec<ReturnItem>,
    pub distinct: bool,
    /// ISO `GROUP BY` grouping keys. When non-empty they DRIVE the grouping (and
    /// force it on, even with no aggregate); empty → implicit grouping by the
    /// non-aggregate items.
    pub group_by: Vec<Expr>,
    /// ISO `HAVING <search condition>` — a post-aggregation filter on groups (the
    /// `SELECT` statement only; `RETURN`/`WITH` never set it). References group
    /// keys + aggregates; a group is kept only when it is TRUE.
    pub having: Option<Expr>,
    pub order_by: Vec<SortItem>,
    pub skip: Option<CountBound>,
    pub limit: Option<CountBound>,
}

/// A single RETURN expression with an optional `AS` alias.
#[derive(Debug, Clone)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
}

/// One `ORDER BY` key, with optional ISO `NULLS FIRST` / `NULLS LAST`.
#[derive(Debug, Clone)]
pub struct SortItem {
    pub expr: Expr,
    pub descending: bool,
    /// `Some(true)` = NULLS FIRST, `Some(false)` = NULLS LAST, `None` = default.
    pub nulls_first: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// A scalar literal value.
#[derive(Debug, Clone)]
pub enum Lit {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    /// An ISO temporal literal (`DATE '…'` / `DATETIME '…'` / `DURATION '…'`).
    Temporal(crate::temporal::Temporal),
}

/// Which graph-element predicate an [`Expr::GraphPred`] is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphPredKind {
    /// `<edge> IS [NOT] DIRECTED`.
    Directed,
    /// `<node> IS [NOT] SOURCE OF <edge>`.
    SourceOf,
    /// `<node> IS [NOT] DESTINATION OF <edge>`.
    DestOf,
    /// `ALL_DIFFERENT(a, b, …)` — all operands are pairwise-distinct elements.
    AllDifferent,
    /// `SAME(a, b, …)` — all operands are the same element.
    Same,
}

/// Expression tree (see TS `Expr`). Sub-expressions are boxed.
/// A `<value type>` in the `IS TYPED` predicate (ISO `<value type predicate>`).
/// Scalar leaves carry a normalized category (`integer`/`float`/`string`/`bool`/
/// `list`/`date`/…/`null`/`any`) — the predicate keeps its richer vocabulary (e.g.
/// the integer/float split) rather than the constraint `TypeSpec`'s single numeric
/// type. Records mirror the constraint shape: closed on extras, each field
/// nullable/optional unless `NOT NULL`.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeTest {
    /// A predefined/scalar type family.
    Scalar(String),
    /// The OPEN record type (`ANY RECORD` / bare `RECORD`): any map.
    AnyRecord,
    /// A CLOSED record: an exact, sorted field set — `(name, type, not_null)`.
    Record(Vec<(String, Self, bool)>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Var(String),
    Param(String),
    Prop {
        variable: String,
        key: String,
    },
    Lit(Lit),
    List(Vec<Self>),
    /// ISO `<record constructor>` — `{ field: expr, … }`. Field names are
    /// identifiers; keys are canonicalized (sorted, duplicate last-wins) into a
    /// `Val::Map` at eval time.
    Record(Vec<(String, Self)>),
    /// ISO GQL list element access `base[index]` — 0-based; out of range → null.
    /// Also a record field access when `base` is a map and `index` is a string.
    Index {
        base: Box<Self>,
        index: Box<Self>,
    },
    /// Property access on an arbitrary expression — `edges(p)[0].amount`,
    /// `head(rels).x`. The bare `variable.key` form stays `Prop` (the hot path);
    /// this is the postfix `.key` that chains off a subscript/function result.
    Field {
        base: Box<Self>,
        key: String,
    },
    Compare {
        op: CompareOp,
        left: Box<Self>,
        right: Box<Self>,
    },
    /// A left-associative arithmetic run at one precedence level (additive or
    /// multiplicative): `head` then each `(op, operand)` folded left to right, so
    /// `a - b - c` is `head=a, tail=[(Sub,b),(Sub,c)]`. n-ary (not nested `Box`s)
    /// so a long chain is a flat `Vec`, not a chain-deep tree that would overflow
    /// the stack when walked — see round-12 C1.
    Arith {
        head: Box<Self>,
        tail: Vec<(ArithOp, Self)>,
    },
    /// A left-associative string-concat run `a || b || …`. n-ary; any null
    /// operand yields null.
    Concat(Vec<Self>),
    Neg(Box<Self>),
    /// n-ary boolean runs (fully associative three-valued folds). A run of the
    /// *same* operator flattens into one `Vec`; a mixed `OR`/`XOR` run nests on
    /// the operator switch (both are left-associative at one precedence level).
    And(Vec<Self>),
    Or(Vec<Self>),
    Xor(Vec<Self>),
    Not(Box<Self>),
    IsNull {
        expr: Box<Self>,
        negated: bool,
    },
    /// `x IS [NOT] TRUE|FALSE|UNKNOWN` — `truth` is the target (`None` = UNKNOWN).
    IsTruth {
        expr: Box<Self>,
        truth: Option<bool>,
        negated: bool,
    },
    /// `x IS [NOT] LABELED <label expression>`.
    IsLabeled {
        expr: Box<Self>,
        label: LabelExpr,
        negated: bool,
    },
    /// `x IS [NOT] TYPED <value type> [NOT NULL]` — the ISO value-type predicate.
    /// `ty` is the declared `<value type>` (a scalar family, an open `ANY RECORD`,
    /// or a closed `RECORD {…}`). Null conforms to any nullable type, so
    /// `null IS TYPED T` is true unless `not_null`.
    IsTyped {
        expr: Box<Self>,
        ty: TypeTest,
        not_null: bool,
        negated: bool,
    },
    /// A graph-element predicate: `IS [NOT] DIRECTED`, `IS [NOT] SOURCE/DESTINATION
    /// OF <edge>`, `ALL_DIFFERENT(a, b, …)`, `SAME(a, b, …)`. `args` holds the
    /// element operands (Directed: `[edge]`; Source/Dest: `[node, edge]`;
    /// AllDifferent/Same: `[e1, e2, …]`). `negated` applies only to the `IS NOT`
    /// forms (the function forms are never negated).
    GraphPred {
        kind: GraphPredKind,
        args: Vec<Self>,
        negated: bool,
    },
    /// `PROPERTY_EXISTS(n, key)` — true iff property `key` is *present* on element
    /// `n`, regardless of value. Distinguishes an absent key from a present null
    /// (null is a first-class stored value here), which `n.key IS NOT NULL`
    /// cannot. The second argument is a bare property name, not an expression.
    PropertyExists {
        variable: String,
        key: String,
    },
    In {
        expr: Box<Self>,
        list: Box<Self>,
        negated: bool,
    },
    /// `EXISTS { p1, … [WHERE pred] }` — correlated sub-pattern existence.
    Exists {
        patterns: Vec<PathPattern>,
        where_: Option<Box<Self>>,
    },
    /// `COUNT { p1, … [WHERE pred] }` — correlated sub-pattern match count.
    CountSubquery {
        patterns: Vec<PathPattern>,
        where_: Option<Box<Self>>,
    },
    /// ISO `VALUE { [MATCH p1, … [WHERE pred]] RETURN <expr> }` — a scalar
    /// (single-value) correlated subquery. 0 rows → NULL; exactly one row → its
    /// value; >1 rows with a non-aggregate RETURN → cardinality fault; an
    /// aggregate RETURN folds the whole group to one value.
    ValueSubquery {
        patterns: Vec<PathPattern>,
        where_: Option<Box<Self>>,
        ret: Box<Self>,
    },
    /// ISO `<let value expression>`: `LET x = e1, y = e2 IN <body> END` — binds
    /// scoped locals (each visible to later bindings and the body), then yields
    /// the body's value. A scalar expression, not a clause.
    LetIn {
        bindings: Vec<LetItem>,
        body: Box<Self>,
    },
    /// ISO CASE: `subject` present → simple CASE, else searched.
    Case {
        subject: Option<Box<Self>>,
        whens: Vec<(Self, Self)>,
        else_: Option<Box<Self>>,
    },
    Func {
        name: String,
        args: Vec<Self>,
        distinct: bool,
        star: bool,
    },
}
