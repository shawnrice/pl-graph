//! The neutral, language-agnostic algebra. GQL and Gremlin both compile INTO
//! this; nothing below the (future) front-ends knows which language produced a
//! plan. Slices so far: the relational core (Scan/Filter/Project) plus Expand
//! (a hop). Graph operators beyond the hop (VarLength, ShortestPath), effects,
//! and the lineage-requirement annotation join as later slices land.

use crate::value::Value;

/// Hop direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Out,
    In,
    Both,
}

/// Which shortest paths a `ShortestPath` hop keeps per reachable target. `Any`
/// emits ONE representative (the first BFS reach); `All` emits every distinct
/// minimum-length path (so a target reachable by two shortest paths yields two
/// rows). `SHORTEST 1` reduces to `Any`, `SHORTEST 1 GROUP` to `All`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortestSelector {
    Any,
    All,
    /// `SHORTEST k [GROUP]` (k >= 2): enumerate every trail to each endpoint, order
    /// by (length, discovery), then keep the first `k` (plain) or every path whose
    /// length is among the `k` smallest distinct lengths (`group`). `k == 1` reduces
    /// to `Any`, `k == 1 GROUP` to `All` (handled at parse time).
    ShortestK {
        k: u32,
        group: bool,
    },
}

/// The path-restriction mode of a variable-length hop (ISO GQL). It decides which
/// elements a single path may repeat:
/// - `Walk`: no restriction — edges AND nodes may repeat (`MATCH WALK`).
/// - `Trail`: no edge repeats within a path — the engine's (and ISO's) default.
/// - `Simple`: no node repeats, EXCEPT a path may close on its own start
///   (`start == end`, a cycle).
/// - `Acyclic`: no node repeats at all — not even the start.
///
/// `Trail` is the default so a bare `-[:R]->{1,3}` keeps the historic behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PathMode {
    Walk,
    #[default]
    Trail,
    Simple,
    Acyclic,
}

/// The position of a quantified subpath-group variable within its `k`-hop unit
/// `((x)-[e1]->(m)-[e2]->(y)){…}` — which flat-path slice it collects across
/// repetitions. `NodeAt(p)` is the node at unit position `p` (0..=k) → `verts[rep*k
/// + p]`; `EdgeAt(p)` the edge at position `p` (0..k) → `edges[rep*k + p]`. A single
/// hop is `NodeAt(0)` (source `x`), `EdgeAt(0)` (edge `e`), `NodeAt(1)` (target `y`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupPos {
    NodeAt(u32),
    EdgeAt(u32),
}

/// A subpath-group UNIT for the general (possibly NESTED) matcher — an ordered list
/// of elements plus the unit's own source-variable slot. Mirrors core's `CUnit`. A
/// bound variable's list-nesting depth equals the number of enclosing quantified
/// sub-units, so `bind_nested` assembles each as a `Value::List` (of lists) keyed by
/// the repetition counters of the units it sits inside. Used by `Plan::NestedGroup`;
/// the FLAT single-level case still lowers to `Plan::RepeatGroup`.
#[derive(Clone, Debug)]
pub struct GUnit {
    /// The unit's source node variable (`x` in `(x)-[e]->(y)`), bound once per rep.
    pub start_slot: Option<usize>,
    pub elems: Vec<GElem>,
}

/// One element of a [`GUnit`]: a single hop, or a nested quantified sub-unit.
#[derive(Clone, Debug)]
pub enum GElem {
    /// A single graph hop. `edge_slot`/`target_slot` bind the edge / landing node as
    /// group variables (at THIS unit's nesting depth). `edge_pred` is a PER-HOP edge
    /// filter (`-[e WHERE …]->` / inline `{k:v}`) over the edge at mini-scope slot 0.
    Hop {
        dir: Dir,
        etypes: Vec<String>,
        edge_slot: Option<usize>,
        target_slot: Option<usize>,
        edge_pred: Option<Box<Expr>>,
    },
    /// A nested quantified sub-group `( <unit> ){min,max}`. `target_slot` binds the
    /// sub-group's landing (its last inner hop's target, per outer rep).
    Sub {
        unit: Box<GUnit>,
        min: u32,
        max: u32,
        target_slot: Option<usize>,
    },
}

/// Element typing for a subscript whose base is a group-variable LIST: a group node
/// list (`x`/`y`) makes `x[i]` a node (so `x[i].prop` resolves the node property),
/// an edge list (`e`) makes it an edge. `Plain` is an ordinary list/record subscript
/// (the value at the index, untyped).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ElemKind {
    #[default]
    Plain,
    Node,
    Edge,
}

/// An ISO graph-element predicate over element IDENTITY (a node/edge's id + kind):
/// `IsDirected` (`e IS DIRECTED`), `IsSourceOf`/`IsDestinationOf` (`a IS SOURCE OF
/// e`), `AllDifferent`/`Same` (`ALL_DIFFERENT(…)`, `SAME(…)`). All are three-valued —
/// a NULL / non-element operand yields NULL. See `Expr::GraphPred`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphPredOp {
    IsDirected,
    IsSourceOf,
    IsDestinationOf,
    AllDifferent,
    Same,
}

/// An expression over the current row. A row is a tuple of bound slots; `Slot(n)`
/// is the value at slot `n`, and `Prop { slot, key }` reads a property off the
/// element in that slot.
#[derive(Clone, Debug)]
pub enum Expr {
    /// The value bound at slot `n` (e.g. a scanned or expanded node).
    Slot(usize),
    /// Gremlin `hasLabel(L, …)` / GQL label test: TRUE iff the element in `slot` carries
    /// ANY of `labels`. Evaluated by resolving each label's (sorted) id bucket ONCE and
    /// binary-searching per row — no per-row list materialization or string hashing,
    /// unlike the `In(Lit, labels(slot))` form it replaces.
    IsLabeled {
        slot: usize,
        labels: Vec<String>,
    },
    /// A property of the element in slot `slot`.
    Prop {
        slot: usize,
        key: String,
    },
    /// A constant.
    Lit(Value),
    /// An unbound query parameter `$name`, produced by the parser ONLY in prepared
    /// mode ([`crate::gql::parse_prepared`]). A direct `parse_with_params` call
    /// substitutes `$name` to its [`Lit`](Expr::Param) at parse time and never
    /// emits this. [`crate::bind::bind_params`] replaces it with the supplied value
    /// before `opt`/`exec` run, so a `Param` reaching evaluation is an
    /// unbound-parameter error (the safety net).
    Param(String),
    /// The current row's PATH as a `Value::List` of node ids. Reading it is what
    /// makes a plan require lineage (see `needs_lineage`); over a plan that does
    /// not track lineage it is NULL.
    Path,
    /// A comparison; `=`/`<>` use the value contract's `equals`, ordering uses
    /// `cmp_total`. NULL operands make the result NULL (three-valued).
    Compare {
        op: CompareOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    /// Boolean `left XOR right` — same precedence level as `Or`, left-associative.
    /// Three-valued: a NULL operand yields NULL; otherwise `a != b`.
    Xor(Box<Expr>, Box<Expr>),
    /// Arithmetic `left <op> right`. `f64` math; a NULL, non-numeric, or
    /// non-finite operand — or a non-finite result — yields NULL (via
    /// `value::as_num` + a finiteness check). Unary minus desugars to `0 - x`.
    Arith {
        op: ArithOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// A scalar function call `name(args…)` (name lowercased). Numeric functions
    /// (abs/sign/floor/ceil/round/sqrt) are finite-or-null per arg; `coalesce`
    /// returns the first non-null arg. NOT the aggregates (those are `Agg`).
    Call {
        name: String,
        args: Vec<Expr>,
    },
    /// Searched `CASE`: the value of the FIRST branch whose condition is TRUE
    /// (three-valued — only a literal TRUE selects; FALSE/NULL/UNKNOWN skip), else
    /// `otherwise`, else NULL.
    Case {
        branches: Vec<(Expr, Expr)>,
        otherwise: Option<Box<Expr>>,
    },
    /// A list literal `[a, b, …]`. Evaluated per row into a `Value::List` — the
    /// elements are expressions, so `[p.age, 1]` is not a constant.
    List {
        items: Vec<Expr>,
    },
    /// A record literal `{k: expr, …}` (ISO `<record>`). Per row it evaluates each
    /// field to a `Value::Record` (keys sorted, duplicates last-wins — see
    /// `value::make_record`). Field values are expressions, so `{a: p.age}` is not
    /// a constant.
    Record {
        fields: Vec<(String, Expr)>,
    },
    /// A map literal producing a `Value::Map` (insertion-ordered, string keys).
    /// Built by the Gremlin front-end for multi-label `select('a','b')`; GQL uses
    /// `Record` instead. Values are expressions, evaluated per row.
    MapLit {
        entries: Vec<(String, Expr)>,
    },
    /// Field access on an arbitrary base expression: `<base>.key`. `base` may be a
    /// record/map value (→ the field) or an element frontier (→ its property) —
    /// the general form of `Prop` (which is the slot-shortcut the optimizer sees).
    /// Used for `{…}.k`, `(expr).k`, and chains.
    Field {
        base: Box<Expr>,
        key: String,
    },
    /// ISO GQL list/record subscript `base[index]` — 0-based, out of range → NULL,
    /// null-safe; a numeric index on a list, a string index on a record/map, else
    /// NULL. A non-integer / negative index is NULL.
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
        /// Element typing when `base` is a group-variable list — makes `x[i].prop`
        /// resolve the node/edge property. `Plain` for an ordinary subscript.
        elem: ElemKind,
    },
    /// `CAST(<expr> AS <TYPE>)`. The coercion itself lives in `value::cast` (the
    /// single home for the conversion table); a failed conversion throws
    /// `E_INVALID_VALUE` (the read pipeline is fallible — see `exec::try_run`),
    /// while a NULL input casts to NULL for every target.
    Cast {
        target: CastTarget,
        expr: Box<Expr>,
    },
    /// `<expr> IS [NOT] NULL`. A definite predicate — always TRUE or FALSE, never
    /// UNKNOWN — which is the point of a 3VL null test: `NULL IS NULL` is TRUE,
    /// not NULL. `negated` flips it to `IS NOT NULL`.
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    /// `PROPERTY_EXISTS(<var>, <key>)`: true iff the element in `slot` carries a
    /// *present* value for `key`, regardless of that value. This is the one
    /// predicate that separates an absent property from a present `Null` (null is
    /// a first-class stored value), which `<var>.<key> IS NOT NULL` cannot.
    PropertyExists {
        slot: usize,
        key: String,
    },
    /// A path accessor over the current row's path (the lineage sidecar):
    /// `nodes(p)`, `relationships(p)`, `path_length(p)`, `elements(p)`. There is
    /// one path per row, so it reads the sidecar directly rather than taking a
    /// value — the parser validates that its argument is a path variable. NULL
    /// (empty list / 0) when the plan tracks no lineage.
    PathAccess {
        part: PathPart,
    },
    /// Gremlin `path()` over an INTERLEAVED node/edge chain (`outE().inV()…`): the
    /// lineage's `values` (vertices) zipped with `edges` in traversal order —
    /// `[v0, e0, v1, e1, …]`. `ends_on_edge` drops the premature target the final `outE`
    /// recorded when no `inV` followed (the path stops on the edge). Each element is
    /// rendered by the positionally-CYCLED `bys` (an empty `bys` renders element maps);
    /// a vertex vs edge element is known from its parity in the interleaving.
    GremlinPath {
        ends_on_edge: bool,
        bys: Vec<GPathBy>,
    },
    /// `EXISTS { <pattern> [WHERE <pred>] }` — a correlated existence predicate. A
    /// definite Bool per outer row: TRUE iff the sub-pattern, extended from an
    /// outer-bound variable, matches at least once for that row. `body` is a
    /// `Plan` rooted at `Plan::Row` (the outer rows). `outer_width` is the number
    /// of outer slots — the body correlates on slots below it, so the predicate is
    /// treated as referencing up to `outer_width - 1` (never pushed below the
    /// operators that bind those variables).
    Exists {
        body: Box<Plan>,
        outer_width: usize,
    },
    /// `COUNT { <pattern> [WHERE …] }` — a correlated count subquery. Same body shape
    /// and correlation as [`Expr::Exists`] (a `Plan::Row`-rooted body over the outer
    /// rows, provenance-tagged), but yields the NUMBER of sub-matches per outer row as
    /// a `Num`, not a Bool.
    CountSubquery {
        body: Box<Plan>,
        outer_width: usize,
    },
    /// Gremlin `local(<hop>.fold())`: a correlated COLLECT — the same provenance-tagged
    /// body as [`Expr::CountSubquery`], but gathers `scalar` over the survivors per outer
    /// row into a `Value::List` (the empty list when the body matches nothing).
    CollectSubquery {
        body: Box<Plan>,
        scalar: Box<Expr>,
        outer_width: usize,
    },
    /// A single scalar aggregate over a correlated CALL body — `CALL (p) { MATCH …
    /// RETURN sum(x) AS s }` for `func` in sum/avg/min/max. The same provenance-tagged
    /// body as [`Expr::CollectSubquery`]; `scalar` is the aggregate's argument evaluated
    /// over the body's sub-scope, reduced per outer row under `func`. Empty / all-null
    /// group: `SUM` → 0 (GQL, not SQL's NULL), `AVG`/`MIN`/`MAX` → NULL. `count` stays
    /// [`Expr::CountSubquery`] (empty → 0).
    AggSubquery {
        body: Box<Plan>,
        scalar: Box<Expr>,
        func: AggFn,
        outer_width: usize,
    },
    /// `VALUE { <pattern> RETURN <scalar> }` — a correlated SCALAR subquery. Same
    /// provenance-tagged body run as [`Expr::Exists`]; yields the `scalar` expression
    /// evaluated on the body's single matching row per outer row (NULL when the body
    /// matches nothing). `scalar` is an expression over the body's sub-scope.
    ScalarSubquery {
        body: Box<Plan>,
        scalar: Box<Expr>,
        outer_width: usize,
    },
    /// `EXISTS { … }` whose body references NO outer variable — an UNCORRELATED
    /// existence check (`EXISTS { MATCH (x:N) MATCH (y:M) }`). The body is a
    /// self-contained plan (scans, cross-joins, filters); it is run once and the
    /// same boolean (non-empty?) is broadcast to every outer row.
    UncorrelatedExists {
        body: Box<Plan>,
    },
    /// `COUNT { … }` whose body references NO outer variable — an UNCORRELATED count.
    /// The body is run once and its ROW COUNT broadcast to every outer row.
    UncorrelatedCount {
        body: Box<Plan>,
    },
    /// `VALUE { … RETURN <scalar> }` whose body references NO outer variable — an
    /// UNCORRELATED scalar subquery (`VALUE { RETURN 1+2 }`, `VALUE { MATCH (n)
    /// RETURN count(*) }`). `body` is the WHOLE self-contained query (including the
    /// RETURN projection/aggregate); it is run once and its single value broadcast.
    /// Empty result → NULL; more than one row → an error, like a correlated VALUE.
    UncorrelatedScalar {
        body: Box<Plan>,
    },
    /// An ISO graph-element predicate (`IS DIRECTED`, `IS SOURCE OF`, `ALL_DIFFERENT`,
    /// `SAME`, …) over element identity. `negated` applies the `IS NOT …` spelling
    /// (three-valued: a NULL result stays NULL). `args` are the operand expressions.
    GraphPred {
        op: GraphPredOp,
        args: Vec<Expr>,
        negated: bool,
    },
    /// `needle IN haystack` where `haystack` is a dynamic (non-literal) list
    /// expression (a list property, a param, a function result). A literal
    /// `x IN [a, b]` desugars to an OR-chain at parse time instead; this is the
    /// runtime form. Three-valued, exactly like that OR-chain: TRUE if any element
    /// equals the needle, else UNKNOWN (NULL) if the needle or any element is null
    /// (the membership can't be decided), else FALSE. A non-list haystack is NULL.
    In {
        needle: Box<Expr>,
        haystack: Box<Expr>,
    },
}

/// The target type of a `CAST`. The engine has one numeric type (`f64`), so
/// `Integer` and `Float` both land in `Value::Num`; `Integer` differs only in
/// truncating toward zero. The conversion table for each target is in
/// `value::cast`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastTarget {
    Integer,
    Float,
    String,
    Boolean,
    List,
}

/// Which part of a path an accessor returns. `Length` is the hop count (= number
/// of relationships); `Elements` interleaves nodes and relationships
/// (`n0, e0, n1, …, nk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathPart {
    Nodes,
    Relationships,
    Length,
    Elements,
    /// `cardinality(p)` — the ISO count of ALL elements in a path: nodes + edges
    /// (`path_length` + 1 nodes and `path_length` edges). Distinct from `Length`
    /// (edges only) and `size` (a list-only function that rejects a path).
    Cardinality,
}

/// A `path().by(...)` element projection for the interleaved Gremlin path
/// ([`Expr::GremlinPath`]). `Element` renders the whole element map; `Prop` a property;
/// `Id`/`Label` the element's ext-id / label. Applied CYCLICALLY across path positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GPathBy {
    Element,
    Prop(String),
    Id,
    Label,
}

/// A binary arithmetic operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// A GQL set operation joining two query arms. `Union` concatenates (deduped unless
/// `all`); `Except` keeps left rows absent from the right; `Intersect` keeps left
/// rows present in the right. `Except`/`Intersect` always dedup (the `all` variants
/// are not distinguished here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineOp {
    Union,
    Except,
    Intersect,
}

/// One set-op-tail arm of an inline `CALL` body (see `Plan::CallInline::parts`). `op`
/// is how this arm combines with the accumulated result to its left; `all` is the
/// `UNION ALL` flag (ignored for EXCEPT/INTERSECT, which always dedup). `body` is the
/// arm's provenance-tagged plan and `yields` its RETURN projection.
#[derive(Clone, Debug)]
pub struct CallPart {
    pub op: CombineOp,
    pub all: bool,
    pub body: Plan,
    pub yields: Vec<(String, Expr)>,
}

/// An aggregate function. `Count` with no argument (`arg: None`, `distinct:
/// false`) is `count(*)`; with an argument it counts non-null values; with
/// `distinct` it counts non-null distinct values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggFn {
    Count,
    Sum,
    Min,
    Max,
    Avg,
    /// Collect every value in the group into a `Value::List`, in the group's
    /// row order (so a preceding sort carries through). Nulls are kept — this is
    /// a faithful fold of the stream, not a null-skipping numeric aggregate. This
    /// is Gremlin `fold()`; a group with no rows folds to the empty list.
    Collect,
    /// Like [`Collect`] but SKIPS nulls — GQL `collect_list`. Row order preserved,
    /// an all-null (or empty) group folds to the empty list. Distinct from `Collect`
    /// so Gremlin `fold()` (which keeps nulls) stays unchanged.
    CollectList,
    /// Population / sample standard deviation from the one-pass moments
    /// `sqrt((Σx² − (Σx)²/n) / denom)`, denom = `n` (pop) or `n−1` (samp). `pop` is
    /// NULL over 0 rows, `samp` over fewer than 2. Matches core's `stddev_of`.
    StddevPop,
    StddevSamp,
    /// Ordered-set aggregates `percentile_cont(x, f)` / `percentile_disc(x, f)` — the
    /// interpolated / discrete `f`-th percentile of the group's finite numeric values
    /// (the fraction `f` rides in `Agg::frac`). NULL over an empty group.
    PercentileCont,
    PercentileDisc,
}

/// One aggregate in an `Aggregate` operator: `func(arg)` (or `func(DISTINCT
/// arg)`), output-named `name`. `arg` is `None` only for `count(*)`.
#[derive(Clone, Debug)]
pub struct Agg {
    pub func: AggFn,
    pub arg: Option<Expr>,
    pub distinct: bool,
    pub name: String,
    /// The fraction argument of `percentile_cont`/`percentile_disc` (a constant);
    /// `None` for every other aggregate.
    pub frac: Option<f64>,
    /// When true, an empty / all-null `Sum` yields NULL instead of 0 — the Gremlin
    /// `sum()` contract (no traversers → no value), vs GQL/SQL's `SUM = 0`. Only
    /// affects `Sum` (Min/Max/Avg already null on an empty set).
    pub null_on_empty: bool,
    /// When true, `Min`/`Max` are NUMERIC reductions — a non-null non-numeric value
    /// faults (the Gremlin contract: min()/max() only compare numbers). GQL/SQL keep
    /// the total order (min/max over any orderable type). Only affects Min/Max.
    pub numeric_only: bool,
}

/// One ORDER BY key: an expression, a direction, and where NULLs go. Null
/// placement is a LANGUAGE contract set by the front-end, INDEPENDENT of
/// direction: GQL puts NULLs last in both ASC and DESC (`nulls_first: false`),
/// Gremlin puts them first (`nulls_first: true`). The non-null values order by
/// `cmp_total`, reversed under `descending`; NULLs are then placed at the chosen
/// end regardless of direction. (A NaN is a non-null number — it orders with the
/// numbers via `cmp_total`, not with the NULLs.)
#[derive(Clone, Debug)]
pub struct SortKey {
    pub expr: Expr,
    pub descending: bool,
    pub nulls_first: bool,
}

/// A Gremlin graph-algorithm annotate kind (see `Plan::AlgoAnnotate`). Parameters
/// carry the resolved config; the algorithm math lives in `crate::algo`.
#[derive(Clone, Debug)]
pub enum GremlinAlgo {
    PageRank { damping: f64, iterations: u32 },
    ConnectedComponent,
    PeerPressure { iterations: u32 },
}

/// A logical plan node. A plan is a tree; execution pulls a batch up through it.
#[derive(Clone, Debug)]
pub enum Plan {
    /// Seed the frontier into slot 0: a label bucket, or the universe when
    /// `label` is None.
    Scan { label: Option<String> },
    /// Seed slot 0 with specific nodes by their EXTERNAL ids — Gremlin `g.V(id, …)`.
    /// Resolved at exec time (the parser has no store); an id that resolves to no
    /// live node contributes nothing (matching Gremlin's `g.V(<missing>)`).
    NodeSeed { ext_ids: Vec<String> },
    /// Seed slot 0 with EVERY live edge — Gremlin `g.E()`. The slot holds edge ids
    /// (a `Col::Edges` frontier), not nodes; downstream steps that read the current
    /// element (`values`/`id`/`label`/`count`) resolve it as an edge.
    EdgeScan,
    /// Seed slot 0 with specific EDGES by external id — Gremlin `g.E(id, …)`. Like
    /// [`NodeSeed`] but for the edge frontier; an id resolving to no live edge
    /// contributes nothing. Preserves the requested id order.
    EdgeSeed { ext_ids: Vec<String> },
    /// Gremlin `sample(n)`: a seeded (fixed Mulberry32) partial Fisher-Yates shuffle of
    /// the whole row stream, truncated to `n` — deterministic and byte-identical to core.
    Sample { input: Box<Plan>, n: usize },
    /// Gremlin `index()`: replace each row with the 2-element list `[element, position]`
    /// where `position` is the element's 0-based index IN THE STREAM. Collapses to one
    /// output column (a list per row).
    Enumerate { input: Box<Plan>, slot: usize },
    /// Move from an EDGE frontier to an endpoint VERTEX — Gremlin `outV`/`inV`/`bothV`
    /// off an edge (e.g. `g.E().outV()`, `coalesce(outE(...)).inV()`), where no
    /// preceding `outE`/`inE` pre-landed the endpoint. Reads the edge at `edge_slot`,
    /// appends the endpoint node at a new slot; `Both` fans out to two rows per edge.
    EdgeVertex {
        input: Box<Plan>,
        edge_slot: usize,
        which: Dir,
    },
    /// The correlated current row — the leaf of an `EXISTS { … }` body. It is NOT
    /// a stand-alone source: it yields whatever batch the enclosing `Expr::Exists`
    /// feeds it (the outer rows plus a provenance column), so it only ever appears
    /// inside a body evaluated by `exec::pull_body`, never in the main pipeline.
    Row,
    /// `FOR <var> IN <list> [WITH ORDINALITY|OFFSET <ord>]` — ISO list unwind. For
    /// each input row, evaluate `list` and emit one output row per element, binding
    /// the element at `var_slot` (appended after the input columns). A NULL list and
    /// an empty list yield NO rows; a NON-list scalar is a one-element singleton.
    /// `ordinal` appends a 1-based (`ORDINALITY`) or 0-based (`OFFSET`) counter.
    Unwind {
        input: Box<Plan>,
        list: Box<Expr>,
        var_slot: usize,
        ordinal: Option<(usize, bool)>,
    },
    /// Seed slot 0 with the nodes carrying `label` whose property `key` equals
    /// `value` under predicate `=`. Produces exactly the rows of
    /// `Scan(label) + Filter(key = value)`; uses a property index when one exists,
    /// otherwise scans the label. A NaN/NULL `value` matches nothing (as `=`).
    IndexSeek {
        label: String,
        key: String,
        value: Value,
    },
    /// Seed slot 0 with the nodes carrying `label` whose property `key` satisfies
    /// `key <op> value` for a range `op` (`Lt`/`Le`/`Gt`/`Ge`), under the value
    /// contract's total order. Same rows as `Scan(label)+Filter(key <op> value)`;
    /// uses a range index when one exists, else scans. A NULL `value` (or NULL
    /// property) matches nothing (predicate UNKNOWN).
    RangeSeek {
        label: String,
        key: String,
        op: CompareOp,
        value: Value,
    },
    /// Hop from the element in `from` slot along `dir`/`edge_label`, appending the
    /// landed node as a new slot. Rows without a matching neighbour drop; rows
    /// with several fan out (one output row per neighbour), replicating the
    /// existing slots.
    Expand {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Vec<String>,
        /// When true, also bind the traversed EDGE: the output appends the edge
        /// slot (a `Col::Edges`) THEN the landed node slot — so for input width W,
        /// the edge is slot W and the node slot W+1. When false (the default), only
        /// the node slot is appended, exactly as before.
        bind_edge: bool,
        /// Self-loop policy for an undirected (`Both`) hop: when true, a self-loop is
        /// traversed TWICE (out-edge AND in-edge) — Gremlin's `both()` contract; when
        /// false (the default), once — GQL's `MATCH (a)-[:R]-(b)`. No effect for a
        /// directed hop (one index).
        double_loops: bool,
    },
    /// A LEFT-OUTER single hop — GQL `OPTIONAL MATCH (a)-[:R]->(x)`. Like `Expand`
    /// but a source row with NO matching neighbour is KEPT, its appended node slot
    /// holding the null sentinel (`u32::MAX`, never a real dense id, read back as
    /// NULL by property access / rendering / aggregates). So every input row yields
    /// at least one output row. Node-only (no bound edge).
    OptionalExpand {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Vec<String>,
        /// What a row with NO matching neighbour lands in the appended slot: the
        /// `u32::MAX` null sentinel (GQL `OPTIONAL MATCH` — `false`) or the source
        /// element itself (Gremlin `optional(<hop>)`, which passes the traverser
        /// through unchanged on a miss — `true`). Either way the slot stays a node
        /// frontier, so the result continues.
        keep_source: bool,
        /// Bind the traversed edge too: when `true`, appends an EDGE column (the
        /// `u32::MAX` null sentinel on a miss) BEFORE the node column — so
        /// `OPTIONAL MATCH (a)-[f:R]->(b)` binds `f` at `width` and `b` at `width+1`.
        bind_edge: bool,
    },
    /// An interval-overlap hop: like `Expand`, but keeps only edges whose interval
    /// `[edge.lo_key, edge.hi_key]` overlaps `[qlo, qhi]` (`lo <= qhi AND hi >= qlo`).
    /// Produced by the optimizer from `Expand{bind_edge} + Filter(r.lo <= X AND
    /// r.hi >= Y)`; a seek-or-scan operator (like `IndexSeek`): it uses the store's
    /// interval index when one on `(lo_key, hi_key)` exists (huge win — see
    /// `examples/interval_bench`), else scans the adjacency and applies the overlap
    /// itself, so its rows are IDENTICAL either way. `qlo`/`qhi` are evaluated over
    /// the input row (constants for an "as of" query). The index is over OUT-edges,
    /// so only a `Dir::Out` hop can seek; other directions take the scan path.
    IntervalExpand {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Vec<String>,
        lo_key: String,
        hi_key: String,
        qlo: Box<Expr>,
        qhi: Box<Expr>,
        bind_edge: bool,
    },
    /// A quantified hop: from the element in `from`, reach nodes over `min..=max`
    /// hops of `dir`/`edge_label`, appending EACH reached endpoint as one new
    /// slot — one output row per matching path. `min == 0` includes the source
    /// itself (a zero-length path). `mode` is EXPLICIT and load-bearing: it selects
    /// the path-restriction semantics (see [`PathMode`]). `Trail` (the default)
    /// forbids reusing an edge; `Walk` allows anything; `Simple`/`Acyclic` forbid
    /// reusing a node (Simple permits the closing `start == end`). The modes differ
    /// on a cycle/self-loop and must never be conflated — a quantified repetition
    /// carries its mode, a chain of separate fixed Expands is a walk.
    VarLength {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Vec<String>,
        min: u32,
        max: u32,
        mode: PathMode,
        /// Gremlin `until(pred)` stop condition (both spellings): a walk emits an
        /// endpoint ONLY when `pred` holds (checked at each landing from depth `min`)
        /// and then PRUNES — that branch stops rather than hopping past the match. The
        /// pre-form `until(pred).repeat(body)` sets `min = 0` (while-do: a source that
        /// already satisfies `pred` emits at depth 0); the post-form
        /// `repeat(body).until(pred)` sets `min = 1` (do-while). `None` = a plain
        /// bounded walk (emit every landing in `[min, max]`).
        until: Option<Box<Expr>>,
        /// A Gremlin `repeat(<hop>.<filter>)` body filter (e.g. `repeat(out().hasLabel
        /// ('PERSON'))`): a predicate on the HOP TARGET, applied at each landing past
        /// the source (`len > 0`). A target that fails it is PRUNED — neither emitted
        /// nor descended into — so only surviving nodes continue the walk. `None` = no
        /// body filter. Evaluated over the same one-row mini-batch as `until`.
        body_filter: Option<Box<Expr>>,
        /// Gremlin `repeat(both())` self-loop semantics: an undirected hop crosses a
        /// self-loop TWICE. GQL undirected var-length (`-[:R]-{1,3}`) keeps it once
        /// (`false`); a Gremlin `both()` walk doubles it (`true`, only bites on
        /// `Dir::Both`) — matching core.
        double_loops: bool,
    },
    /// A quantified subpath group `((x)-[e]->(y)){min,max}` that BINDS its inner
    /// variables as GROUP variables — each becomes a LIST over the repetitions. Like
    /// [`VarLength`] (same reachability to `endpoint_slot`), but also appends one
    /// list column per entry of `group_binds` (a `(GroupPos, slot)`): the source
    /// (`x`), edge (`e`), or target (`y`) value at each repetition. SINGLE-HOP unit
    /// only (`k == 1`); multi-hop / nested groups are lowered elsewhere. The endpoint
    /// column is appended FIRST (at `endpoint_slot`), then the group columns in
    /// `group_binds` order.
    RepeatGroup {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Vec<String>,
        min: u32,
        max: u32,
        mode: PathMode,
        endpoint_slot: usize,
        group_binds: Vec<(GroupPos, usize)>,
        /// Hops per repetition unit (`k`): 1 for `((x)-[e]->(y)){…}`, 2 for a
        /// two-hop body, etc. `min`/`max` above are HOP counts (`reps * k`); an
        /// endpoint is emitted only at a rep boundary (`hops % k == 0`).
        k: u32,
        /// A PER-REPETITION `WHERE` (`((x)-[e]->(y) WHERE pred){…}`) — a predicate
        /// over the rep's SCALAR variables at fixed mini-scope slots (source=0,
        /// edge=1, target=2), evaluated at each hop; a hop that fails it is pruned.
        /// `None` = no per-rep filter.
        per_rep_pred: Option<Box<Expr>>,
    },
    /// `( <unit> ){min,max}` where the body itself contains a quantified sub-group
    /// (`( (…){a,b} ){c,d}`) or a quantified inner hop (`( (x)-[e]->{a,b}(y) ){c,d}`)
    /// — the general NESTED subpath group. Each valid repetition-decomposition is one
    /// trail; every bound inner variable is materialized as a (possibly nested) list,
    /// one list level per enclosing quantifier. Output layout: the input columns, then
    /// the endpoint at `endpoint_slot`, then one list column per `bind_slots` entry (in
    /// slot order). The flat single-level case stays on [`RepeatGroup`].
    NestedGroup {
        input: Box<Plan>,
        from: usize,
        unit: GUnit,
        min: u32,
        max: u32,
        mode: PathMode,
        endpoint_slot: usize,
        /// Every bound inner-variable slot, ascending — the trailing list columns.
        bind_slots: Vec<usize>,
        /// A PER-REPETITION `WHERE` (`( <unit> WHERE pred ){c,d}`), evaluated at each
        /// OUTER-rep completion over that rep's variables in the PER-REP view — a
        /// variable one nesting level shallower than globally (an outer-depth var is a
        /// scalar, an inner-Sub var a list over the inner reps). A rep failing it is
        /// pruned. `None` = no per-rep filter.
        per_rep_pred: Option<Box<Expr>>,
    },
    /// Shortest-path reach: BFS from the element in `from` along `dir`/
    /// `edge_label`, emitting EACH reachable target once at its shortest distance
    /// (ANY-shortest — one representative per target, not every shortest path),
    /// with the target appended as a new slot. `max` caps the hop distance
    /// (`None` = unbounded); the source itself is not emitted.
    ShortestPath {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Vec<String>,
        /// Minimum hop count for a target to count as reached — `0` for a `*`
        /// quantifier (the seed itself is a zero-length path to itself), `1` for `+`.
        min: u32,
        max: Option<u32>,
        selector: ShortestSelector,
        /// A per-hop edge predicate (`-[e:R WHERE e.w > 5]->*`) over the edge at
        /// scalar slot 0; an edge failing it is not traversed. `None` = no filter.
        edge_pred: Option<Box<Expr>>,
    },
    /// Keep rows where `pred` is TRUE (three-valued: FALSE and NULL drop).
    Filter { input: Box<Plan>, pred: Expr },
    /// Group rows by the `(name, expr)` keys and compute `aggs` per group. With
    /// no keys, the whole input is one group (a scalar aggregate). Output columns
    /// are the key names followed by the aggregate names. Group order is
    /// first-seen — the order each group's first row arrived.
    Aggregate {
        input: Box<Plan>,
        keys: Vec<(String, Expr)>,
        aggs: Vec<Agg>,
    },
    /// Fold a two-column grouped result (`[key, value]` rows, as `Aggregate`
    /// produces) into ONE row holding a single `Value::Map` of `{key: value}`, in
    /// first-seen key order — Gremlin `group()`/`groupCount()` semantics, which
    /// yield one Map object rather than the relational (key,value) rows that GQL
    /// `GROUP BY` keeps.
    GroupToMap { input: Box<Plan> },
    /// Gremlin `shortestPath()`: for each source vertex at `node_slot`, emit ONE row
    /// per shortest path (undirected BFS) to every reachable vertex, each rendered as
    /// a `Value::List` of the path vertices' EXTERNAL ids. Path order is unspecified
    /// (compare as a multiset).
    ShortestPathEnum {
        input: Box<Plan>,
        node_slot: usize,
        /// `with(target, __.has('k'[, op(v)]))`: keep only paths whose DESTINATION
        /// vertex satisfies this property filter. `None` enumerates to every vertex.
        target: Option<(String, Option<(CompareOp, Value)>)>,
    },
    /// Gremlin `subgraph('sg')` revealed by `cap('sg')`: collect the EDGE frontier at
    /// `edge_slot` (deduped) plus their endpoint vertices (deduped), emitting ONE row
    /// holding a `{vertices: [records], edges: [records]}` Map of self-describing
    /// element records — matching core's `subgraph_vertex`/`subgraph_edge` shapes.
    Subgraph { input: Box<Plan>, edge_slot: usize },
    /// Append or overwrite ONE slot's column with `value` (evaluated per row),
    /// leaving every other slot intact — used to carry a Gremlin `sack` accumulator
    /// alongside the element frontier (a Project would collapse the frontier to the
    /// projected values). `append` adds a new column at the input width; otherwise it
    /// replaces `slot`.
    MapSlot {
        input: Box<Plan>,
        slot: usize,
        value: Box<Expr>,
        append: bool,
    },
    /// Gremlin `tree()` [`.by('k')`]: fold EVERY traverser's vertex-hop path (the
    /// node-id lineage) into ONE nested `Value::Map`, keyed level-by-level by each
    /// path element — rendered as its full element map, or its `by` property. Emits a
    /// single row. Only valid over a pure vertex-hop chain (path lineage = node ids).
    Tree {
        input: Box<Plan>,
        by: Option<String>,
        /// A trailing value projection folded INTO the tree as its leaf level —
        /// `out(...).values('name').tree()` keys the last level by the last vertex's
        /// `leaf_value` property (a scalar), one level deeper than the vertices. `None`
        /// = a plain vertex-path tree.
        leaf_value: Option<String>,
    },
    /// A Gremlin OLAP annotate step (`pageRank`/`connectedComponent`/`peerPressure`):
    /// run the algorithm over the whole store, then APPEND the per-node result as a
    /// new column aligned to the vertex frontier at `node_slot`, so a following
    /// `values(<property>)` (routed to this appended slot) reads it. Pass-through: the
    /// vertex frontier is unchanged, matching core's "stream + attached property".
    AlgoAnnotate {
        input: Box<Plan>,
        algo: GremlinAlgo,
        edge_label: Option<String>,
        node_slot: usize,
    },
    /// Sort by `keys` (empty = no sort, pure paging), then keep the window
    /// `[skip, skip+limit)`. Sorting is STABLE — equal keys keep input order — so
    /// `keys` empty with a `limit` is a plain prefix. Runs before any Project, so
    /// its keys may reference bound slots that the output does not carry.
    OrderPage {
        input: Box<Plan>,
        keys: Vec<SortKey>,
        skip: Option<usize>,
        limit: Option<usize>,
    },
    /// Gremlin `union(t1, t2, …)`: for each input row, run every `body` (a
    /// `Plan::Row`-rooted sub-plan continuing from the current slot) and CONCATENATE
    /// all their sub-rows into one frontier — the columnar form of core's
    /// per-traverser branch-and-reconverge. The bodies land a compatible frontier
    /// (the parser scopes each to a single hop, so every branch appends its element
    /// at the same slot and the concatenated column keeps its node/edge type).
    Branch { input: Box<Plan>, bodies: Vec<Plan> },
    /// Collapse the input to ONE column — the element/value at `slot`, cloned so a
    /// `Nodes`/`Edges` frontier keeps its type — while PRESERVING the lineage sidecar
    /// (unlike `Project`, which drops it). Every arm of a `union`/`coalesce`/`choose`
    /// is wrapped in this so bodies of DIFFERENT depths reconverge to a uniform width
    /// with their element at the SAME slot 0; without it, a 2-hop arm beside a 1-hop
    /// arm landed its element at a different slot and a downstream `values()` read a
    /// mid-hop node. `path()`-through-branch still works because the sidecar survives.
    Reconverge { input: Box<Plan>, slot: usize },
    /// Keep the LAST `n` rows of the input, in input order — Gremlin `tail(n)`. The
    /// symmetric partner of a keyless `OrderPage` limit (the FIRST n): both take a
    /// window of the committed row order, but `tail`'s start offset (`rows - n`) is
    /// only known at exec, so it is its own node rather than an `OrderPage` skip.
    Tail { input: Box<Plan>, n: usize },
    /// Produce output columns: `(name, expr)` per column.
    Project {
        input: Box<Plan>,
        items: Vec<(String, Expr)>,
    },
    /// Deduplicate whole rows across every slot, keeping the first occurrence
    /// (first-seen order). Rows are keyed by the value contract's `group_key`, so
    /// two NaNs / two -0.0s collapse — the grouping notion, not predicate
    /// equality. Placed above a Project, it is `RETURN DISTINCT …`.
    Distinct { input: Box<Plan> },
    /// Gremlin `dedup('a','b')`: keep the FIRST row per distinct tuple of values at
    /// `key_slots` (the tagged slots), preserving every other column — a keyed
    /// first-seen distinct, unlike `Distinct` which keys the whole row.
    DistinctBy {
        input: Box<Plan>,
        key_slots: Vec<usize>,
    },
    /// `<query> UNION [ALL] <query>`: run both arms and concatenate their rows. The
    /// result's column names come from the LEFT arm (core's rule — a name mismatch
    /// is not an error). `UNION` (all=false) deduplicates the combined rows by the
    /// grouping key; `UNION ALL` keeps every row. A shorter arm's rows are padded
    /// with NULLs to the left arm's width.
    Union {
        left: Box<Plan>,
        right: Box<Plan>,
        all: bool,
        op: CombineOp,
    },
    /// Sort WITHIN each row's slot-0 value, in place — Gremlin `order(local)`.
    /// A `List` cell becomes its elements sorted by the value contract's
    /// `cmp_total`; a `Map` cell becomes its pairs sorted by VALUE (TinkerPop's
    /// default local ordering of a map); any other cell passes through unchanged.
    /// `descending` reverses the order. Transparent to output naming (like
    /// `Distinct`/`OrderPage`), since it reorders inside a cell, not across rows.
    SortLocal {
        input: Box<Plan>,
        descending: bool,
        /// Sort a Map cell by its KEYS instead of its values (Gremlin
        /// `order(local).by(keys)` vs the default `by(values)`). Ignored for lists.
        by_key: bool,
    },
    /// Hash-join two sub-plans on shared bound variables. `on` lists
    /// `(left_slot, right_slot)` equalities — for `MATCH (a)-[:R]->(b),
    /// (a)-[:S]->(c)` sharing `a`, that is `[(a_left, a_right)]`.
    ///
    /// Slot convention: output slots are ALL of `left`'s slots followed by ALL of
    /// `right`'s. The shared variable is addressed by its LEFT slot; its right
    /// copy is present but inert (the front-end simply points the variable at the
    /// left slot). This keeps the layout trivially predictable — right slot `j`
    /// becomes output slot `left.len() + j` — at the cost of a duplicated column.
    /// The join key is bound-variable identity (a node/edge), so it is
    /// unambiguous and `group_key`-hashed like everything else.
    Join {
        left: Box<Plan>,
        right: Box<Plan>,
        on: Vec<(usize, usize)>,
    },
    /// A leading `OPTIONAL MATCH <pattern>` with no prior binding: run `input` (the
    /// pattern) and, if it produces NO rows, emit exactly one row of `width` NULL
    /// columns — the ISO left-outer of the pattern against the implicit single unit
    /// row. Non-empty results pass through unchanged.
    NullPadIfEmpty { input: Box<Plan>, width: usize },
    /// `OPTIONAL MATCH (p:Label {k: expr, …})` on a FRESH variable — a LEFT-OUTER
    /// correlated node lookup (the FOR-driven `OPTIONAL MATCH (p:Person {name: name})`
    /// shape). For each input row, find the `label` nodes whose property `k` equals
    /// `expr` evaluated over that row; emit one output row (input columns + the node at
    /// `node_slot`) per match, or one row with the node NULL when none match.
    OptionalScan {
        input: Box<Plan>,
        label: Option<String>,
        filters: Vec<(String, Expr)>,
        node_slot: usize,
    },
    /// `CALL (scope) { <subquery> }` — an inline correlated (lateral) subquery. For
    /// each row of `input`, run `body` (a `Plan::Row`-rooted pattern that continues
    /// from a scope variable) and emit one output row per sub-row: the `input`
    /// row's first `outer_width` slots followed by the `yields` expressions
    /// evaluated over the sub-row. An outer row with no sub-row is dropped (inner
    /// lateral join) UNLESS `optional` — `OPTIONAL CALL` is a LEFT-outer join, so
    /// an outer row with no sub-row survives once with every yield column NULL.
    /// (An aggregating subquery is still deferred.) Only the `yields` columns (not
    /// the subquery's internal variables) survive into the outer scope. The body is
    /// built with a provenance id reserved at slot `outer_width` (see the exec).
    CallInline {
        input: Box<Plan>,
        body: Box<Plan>,
        yields: Vec<(String, Expr)>,
        outer_width: usize,
        optional: bool,
        /// Set-operator tail: additional body arms combined with the first one PER
        /// OUTER ROW (`UNION`/`EXCEPT`/`INTERSECT`, left-associative). Empty for the
        /// single-arm form (Phase 0/1). Each part carries its own body + yields; the
        /// output column names come from the first arm's `yields`. Every part's body
        /// is provenance-tagged at slot `outer_width` exactly like `body`.
        parts: Vec<CallPart>,
    },
    /// `CALL name(config)` — a named built-in procedure (a graph algorithm). A leaf
    /// that runs the algorithm over the whole store and produces a two-slot batch:
    /// slot 0 the node ids (`Col::Nodes`), slot 1 the per-node result (`Col::Num`).
    /// `config` is the `{key: value}` argument map. The parser wraps this in a
    /// Project naming the columns (`node` + the procedure's result column) and
    /// applying any `YIELD`.
    CallProcedure {
        name: String,
        config: Vec<(String, Value)>,
    },
    /// A write: create `nodes` (each with labels and inline properties) and the
    /// `edges` among them (`from`/`to` index into `nodes`). A leaf plan — it reads
    /// no input and produces no rows; it is run through the mutable executor
    /// (`exec::execute`), not pulled. Edge properties are a later slice (the store
    /// has no edge-property model yet).
    Insert {
        nodes: Vec<InsertNode>,
        edges: Vec<InsertEdge>,
    },
    /// An `Insert` whose created nodes are bound into scope for a following
    /// projection (`INSERT (n:Person {…}) RETURN n.name`). Each created node
    /// occupies the slot equal to its creation index in `nodes`, so the `tail`
    /// (a `Plan::Row`-rooted projection) resolves `Expr::Prop{slot}` against the
    /// seeded row. A write — run through `exec::execute`, never pulled as a read;
    /// the `tail` is restricted to pure projections (Row/Project).
    InsertReturn {
        nodes: Vec<InsertNode>,
        edges: Vec<InsertEdge>,
        tail: Box<Plan>,
    },
    /// A write over the rows of a read sub-plan: for each matched row, apply the
    /// `ops` (SET/REMOVE) to the bound nodes. Run through `exec::execute`, not
    /// pulled; produces no rows.
    Update { input: Box<Plan>, ops: Vec<SetOp> },
    /// An `Update` whose written frontier is bound into a following projection —
    /// `MATCH (n) SET n.x = 1 RETURN n.x` and Gremlin `property(k, v).values(k)`.
    /// Runs `input`, applies `ops`, then evaluates `tail` (a `Plan::Row`-rooted read)
    /// over the SAME frontier against the mutated store, so the projection reads the
    /// just-written values. The engine's read-after-write (write-then-return) path for
    /// updates — the twin of [`Plan::InsertReturn`]. A write: run through
    /// `exec::execute`, never pulled. The `tail` is restricted to the operators
    /// `pull_body` seeds (Row/Project/Filter/Expand/Aggregate/…), not paging/distinct.
    UpdateReturn {
        input: Box<Plan>,
        ops: Vec<SetOp>,
        tail: Box<Plan>,
    },
    /// Create ONE edge between two EXISTING nodes (Gremlin `addE`), with inline
    /// properties. A leaf write; `from`/`to` are node ids. Distinct from
    /// `Insert`'s edges, which reference nodes created in the same statement.
    AddEdge {
        from: u32,
        to: u32,
        etype: String,
        props: Vec<(String, Value)>,
    },
    /// Keyed upsert of ONE node (the `_MERGE` extension, spec
    /// docs/design/gql-extensions.md §2). The key is the subset of `props` named
    /// by a unique constraint on `label` (inferred at execution — the store holds
    /// the constraints); no applicable constraint is an error. Absent → create the
    /// node with all `props`, then apply `on_create`. Present → apply `on_update`
    /// (default: clobber the non-key payload to the pattern's values). The merged
    /// node is bound at slot 0 for the `on_create`/`on_update` expressions.
    Merge {
        label: String,
        props: Vec<(String, Value)>,
        on_create: Vec<(String, Expr)>,
        on_update: MergeUpdate,
    },
    /// Keyed upsert of ONE edge between two vertices matched by their unique keys
    /// (the `_MERGE` edge form, spec docs/design/gql-extensions.md §2 v1). Both
    /// endpoints are resolved by the unique key inferred from `*_label`/`*_props`
    /// (a missing endpoint is an error); the edge is keyed structurally by
    /// (from, to, `etype`). Absent → create it with `edge_props`, then `on_create`.
    /// Present → apply `on_update` (default: clobber the edge's props to
    /// `edge_props`). The dispositions bind the start node at slot 0, the end node
    /// at slot 1, and the edge at slot 2.
    MergeEdge {
        start_label: String,
        start_props: Vec<(String, Value)>,
        end_label: String,
        end_props: Vec<(String, Value)>,
        dir: Dir,
        etype: String,
        edge_props: Vec<(String, Value)>,
        on_create: Vec<(usize, String, Expr)>,
        on_update: MergeEdgeUpdate,
    },
    /// An ISO GQL transaction-control command (`START TRANSACTION [READ ONLY|READ
    /// WRITE]`, `COMMIT [WORK]`, `ROLLBACK [WORK]`). Not a query — it drives the
    /// store's transaction frame and yields no rows. `read_only` is only meaningful
    /// for `Start`.
    TxControl { kind: TxKind, read_only: bool },
    /// A row-driven INSERT: `FOR x IN <list> INSERT (…)` (and `MATCH … INSERT`) — the
    /// `input` produces rows, and for EACH row the node/edge templates are created
    /// with their property expressions evaluated against that row. The whole
    /// statement is one atomic unit (all rows commit, or a constraint violation rolls
    /// them all back). Distinct from `Insert`, whose properties are constant literals.
    InsertFrom {
        input: Box<Plan>,
        nodes: Vec<InsertNodeExpr>,
        edges: Vec<InsertEdgeExpr>,
    },
}

/// Which transaction-control command a [`Plan::TxControl`] carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxKind {
    Start,
    Commit,
    Rollback,
}

/// The `_MERGE` update-path disposition when the node already exists.
#[derive(Clone, Debug)]
pub enum MergeUpdate {
    /// Default (bare `_MERGE`): set every non-key payload property to the
    /// pattern's value.
    Clobber,
    /// `_ON_UPDATE SET … [WHERE p]`: replaces the default — apply exactly these
    /// assignments, gated by `filter` (false → no-op, not an error).
    Set {
        assigns: Vec<(String, Expr)>,
        filter: Option<Expr>,
    },
    /// `_ON_UPDATE_NOTHING`: leave the existing node untouched.
    Nothing,
}

/// The `_MERGE` edge-form update disposition when the edge already exists. Unlike
/// [`MergeUpdate`], each assignment carries its target `slot` (0 = start node,
/// 1 = end node, 2 = edge), since a disposition may write to either endpoint or the
/// edge itself.
#[derive(Clone, Debug)]
pub enum MergeEdgeUpdate {
    /// Default (bare `_MERGE`): set every edge property to the pattern's value.
    Clobber,
    /// `_ON_UPDATE SET … [WHERE p]`: apply exactly these assignments, gated by
    /// `filter` (false → no-op, not an error).
    Set {
        assigns: Vec<(usize, String, Expr)>,
        filter: Option<Expr>,
    },
    /// `_ON_UPDATE_NOTHING`: leave the existing edge untouched.
    Nothing,
}

/// One property mutation in an `Update`: set a bound node's property to an
/// expression evaluated per row, or remove it. `slot` is the bound node.
#[derive(Clone, Debug)]
pub enum SetOp {
    Set {
        slot: usize,
        key: String,
        value: Expr,
    },
    Remove {
        slot: usize,
        key: String,
    },
    /// `SET n:Label` — add `label` to the bound node in `slot`. Idempotent.
    AddLabel {
        slot: usize,
        label: String,
    },
    /// `REMOVE n:Label` — strip `label` from the bound node in `slot`. Idempotent.
    RemoveLabel {
        slot: usize,
        label: String,
    },
    /// Delete the bound element in `slot` — a node (GQL `DELETE`/`DETACH DELETE`,
    /// Gremlin `drop()`) or an edge. `detach` deletes a node's incident edges too;
    /// a non-`detach` DELETE of a node that still has edges is an error (Cypher/core
    /// semantics). Applied in op order alongside SET/REMOVE. (Gremlin `drop()`
    /// currently only reaches node slots and sets `detach: true`.)
    Delete {
        slot: usize,
        detach: bool,
    },
}

/// A node to create in an `Insert`: its labels and inline `(key, value)`
/// properties.
#[derive(Clone, Debug)]
pub struct InsertNode {
    pub labels: Vec<String>,
    pub props: Vec<(String, Value)>,
}

/// An edge to create in an `Insert`: a typed relationship from `nodes[from]` to
/// `nodes[to]`, with inline `(key, value)` properties.
#[derive(Clone, Debug)]
pub struct InsertEdge {
    pub from: usize,
    pub to: usize,
    pub etype: String,
    pub props: Vec<(String, Value)>,
}

/// A node to create in an `InsertFrom` (a `FOR … INSERT` / `MATCH … INSERT`): its
/// labels and property EXPRESSIONS, evaluated against each input row (so a property
/// may reference a bound variable, e.g. `FOR x IN […] INSERT (:N {v: x})`).
#[derive(Clone, Debug)]
pub struct InsertNodeExpr {
    pub labels: Vec<String>,
    pub props: Vec<(String, Expr)>,
    /// `Some(slot)` when this pattern node REFERENCES a variable already bound by an
    /// enclosing MATCH (e.g. `MATCH (a),(b) INSERT (a)-[:E]->(b)` connects the matched
    /// nodes) — the exec reads the node id from that slot per row instead of creating a
    /// new node. `None` = create a fresh node from `labels`/`props`.
    pub bound: Option<usize>,
}

/// An edge to create in an `InsertFrom`, per input row: a typed relationship from
/// the row's `nodes[from]` to `nodes[to]`, with property expressions.
#[derive(Clone, Debug)]
pub struct InsertEdgeExpr {
    pub from: usize,
    pub to: usize,
    pub etype: String,
    pub props: Vec<(String, Expr)>,
}

impl Plan {
    #[must_use]
    pub fn expand(self, from: usize, dir: Dir, edge_label: &[String]) -> Self {
        Self::Expand {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.to_vec(),
            bind_edge: false,
            double_loops: false,
        }
    }

    /// Like [`Self::expand`] but traverses a self-loop TWICE for an undirected hop —
    /// Gremlin `both()`'s contract (an out-edge AND an in-edge of the same vertex).
    #[must_use]
    pub fn expand_both_gremlin(self, from: usize, edge_label: &[String]) -> Self {
        Self::Expand {
            input: Box::new(self),
            from,
            dir: Dir::Both,
            edge_label: edge_label.to_vec(),
            bind_edge: false,
            double_loops: true,
        }
    }

    /// Like [`Self::expand`] but also binds the traversed edge as a slot (edge
    /// slot then node slot). Used for `(a)-[r:T]->(b)` where `r` is read.
    #[must_use]
    pub fn expand_edge(self, from: usize, dir: Dir, edge_label: &[String]) -> Self {
        Self::Expand {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.to_vec(),
            bind_edge: true,
            double_loops: false,
        }
    }

    /// Like [`Self::expand_edge`] but with Gremlin `bothE()` self-loop semantics: an
    /// undirected edge hop crosses a self-loop TWICE (it is both an out- and an in-edge),
    /// so `bothE().otherV()` on a self-loop yields the vertex twice — matching core. A
    /// directed edge hop is unaffected (`double_loops` only bites on `Dir::Both`).
    #[must_use]
    pub fn expand_edge_gremlin(self, from: usize, dir: Dir, edge_label: &[String]) -> Self {
        Self::Expand {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.to_vec(),
            bind_edge: true,
            double_loops: matches!(dir, Dir::Both),
        }
    }

    #[must_use]
    pub fn optional_expand(
        self,
        from: usize,
        dir: Dir,
        edge_label: &[String],
        keep_source: bool,
        bind_edge: bool,
    ) -> Self {
        Self::OptionalExpand {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.to_vec(),
            keep_source,
            bind_edge,
        }
    }

    #[must_use]
    pub fn var_length(
        self,
        from: usize,
        dir: Dir,
        edge_label: &[String],
        min: u32,
        max: u32,
        mode: PathMode,
    ) -> Self {
        Self::VarLength {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.to_vec(),
            min,
            max,
            mode,
            until: None,
            body_filter: None,
            double_loops: false,
        }
    }

    /// [`Self::var_length`] with a Gremlin `until(pred)` stop condition and/or a
    /// `repeat(<hop>.<filter>)` body filter (see the `until`/`body_filter` fields on
    /// [`Plan::VarLength`]). `double_loops` gives it Gremlin `both()` self-loop
    /// semantics (a self-loop crossed twice).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn var_length_until(
        self,
        from: usize,
        dir: Dir,
        edge_label: &[String],
        min: u32,
        max: u32,
        mode: PathMode,
        until: Option<Box<Expr>>,
        body_filter: Option<Box<Expr>>,
        double_loops: bool,
    ) -> Self {
        Self::VarLength {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.to_vec(),
            min,
            max,
            mode,
            until,
            body_filter,
            double_loops,
        }
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn shortest_path(
        self,
        from: usize,
        dir: Dir,
        edge_label: &[String],
        min: u32,
        max: Option<u32>,
        selector: ShortestSelector,
        edge_pred: Option<Box<Expr>>,
    ) -> Self {
        Self::ShortestPath {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.to_vec(),
            min,
            max,
            selector,
            edge_pred,
        }
    }

    #[must_use]
    pub fn filter(self, pred: Expr) -> Self {
        Self::Filter {
            input: Box::new(self),
            pred,
        }
    }

    #[must_use]
    pub fn aggregate(self, keys: Vec<(String, Expr)>, aggs: Vec<Agg>) -> Self {
        Self::Aggregate {
            input: Box::new(self),
            keys,
            aggs,
        }
    }

    #[must_use]
    pub fn group_to_map(self) -> Self {
        Self::GroupToMap {
            input: Box::new(self),
        }
    }

    #[must_use]
    pub fn shortest_path_enum(self, node_slot: usize) -> Self {
        Self::ShortestPathEnum {
            input: Box::new(self),
            node_slot,
            target: None,
        }
    }

    #[must_use]
    pub fn subgraph(self, edge_slot: usize) -> Self {
        Self::Subgraph {
            input: Box::new(self),
            edge_slot,
        }
    }

    #[must_use]
    pub fn map_slot(self, slot: usize, value: Expr, append: bool) -> Self {
        Self::MapSlot {
            input: Box::new(self),
            slot,
            value: Box::new(value),
            append,
        }
    }

    #[must_use]
    pub fn tree(self, by: Option<String>, leaf_value: Option<String>) -> Self {
        Self::Tree {
            input: Box::new(self),
            by,
            leaf_value,
        }
    }

    #[must_use]
    pub fn algo_annotate(
        self,
        algo: GremlinAlgo,
        edge_label: Option<String>,
        node_slot: usize,
    ) -> Self {
        Self::AlgoAnnotate {
            input: Box::new(self),
            algo,
            edge_label,
            node_slot,
        }
    }

    #[must_use]
    pub fn order_page(self, keys: Vec<SortKey>, skip: Option<usize>, limit: Option<usize>) -> Self {
        Self::OrderPage {
            input: Box::new(self),
            keys,
            skip,
            limit,
        }
    }

    #[must_use]
    pub fn tail(self, n: usize) -> Self {
        Self::Tail {
            input: Box::new(self),
            n,
        }
    }

    #[must_use]
    pub fn branch(self, bodies: Vec<Plan>) -> Self {
        Self::Branch {
            input: Box::new(self),
            bodies,
        }
    }

    #[must_use]
    pub fn reconverge(self, slot: usize) -> Self {
        Self::Reconverge {
            input: Box::new(self),
            slot,
        }
    }

    #[must_use]
    pub fn project(self, items: Vec<(String, Expr)>) -> Self {
        Self::Project {
            input: Box::new(self),
            items,
        }
    }

    #[must_use]
    pub fn distinct_by(self, key_slots: Vec<usize>) -> Self {
        Self::DistinctBy {
            input: Box::new(self),
            key_slots,
        }
    }

    pub fn distinct(self) -> Self {
        Self::Distinct {
            input: Box::new(self),
        }
    }

    #[must_use]
    pub fn sort_local(self, descending: bool, by_key: bool) -> Self {
        Self::SortLocal {
            input: Box::new(self),
            descending,
            by_key,
        }
    }

    /// A hash join (associated, not chained, since it is binary).
    #[must_use]
    pub fn join(left: Self, right: Self, on: Vec<(usize, usize)>) -> Self {
        Self::Join {
            left: Box::new(left),
            right: Box::new(right),
            on,
        }
    }
}
