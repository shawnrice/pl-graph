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

/// An expression over the current row. A row is a tuple of bound slots; `Slot(n)`
/// is the value at slot `n`, and `Prop { slot, key }` reads a property off the
/// element in that slot.
#[derive(Clone, Debug)]
pub enum Expr {
    /// The value bound at slot `n` (e.g. a scanned or expanded node).
    Slot(usize),
    /// A property of the element in slot `slot`.
    Prop {
        slot: usize,
        key: String,
    },
    /// A constant.
    Lit(Value),
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
}

/// One aggregate in an `Aggregate` operator: `func(arg)` (or `func(DISTINCT
/// arg)`), output-named `name`. `arg` is `None` only for `count(*)`.
#[derive(Clone, Debug)]
pub struct Agg {
    pub func: AggFn,
    pub arg: Option<Expr>,
    pub distinct: bool,
    pub name: String,
}

/// One ORDER BY key: an expression and a direction. Ascending uses the value
/// contract's `cmp_total` (nulls last); descending reverses it (so nulls come
/// first under DESC — the total order reversed, which is the honest default and
/// the front-end can override once NULLS FIRST/LAST syntax lands).
#[derive(Clone, Debug)]
pub struct SortKey {
    pub expr: Expr,
    pub descending: bool,
}

/// A logical plan node. A plan is a tree; execution pulls a batch up through it.
#[derive(Clone, Debug)]
pub enum Plan {
    /// Seed the frontier into slot 0: a label bucket, or the universe when
    /// `label` is None.
    Scan { label: Option<String> },
    /// The correlated current row — the leaf of an `EXISTS { … }` body. It is NOT
    /// a stand-alone source: it yields whatever batch the enclosing `Expr::Exists`
    /// feeds it (the outer rows plus a provenance column), so it only ever appears
    /// inside a body evaluated by `exec::pull_body`, never in the main pipeline.
    Row,
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
        edge_label: Option<String>,
        /// When true, also bind the traversed EDGE: the output appends the edge
        /// slot (a `Col::Edges`) THEN the landed node slot — so for input width W,
        /// the edge is slot W and the node slot W+1. When false (the default), only
        /// the node slot is appended, exactly as before.
        bind_edge: bool,
    },
    /// A quantified hop: from the element in `from`, reach nodes over `min..=max`
    /// hops of `dir`/`edge_label`, appending EACH reached endpoint as one new
    /// slot — one output row per matching path. `min == 0` includes the source
    /// itself (a zero-length path). `trail` is EXPLICIT and load-bearing: when
    /// true no edge may repeat within a single path (a trail); when false edges
    /// may repeat (a walk). The two differ on a cycle/self-loop and must never be
    /// conflated — a quantified repetition is a trail, a chain of separate fixed
    /// Expands is a walk.
    VarLength {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Option<String>,
        min: u32,
        max: u32,
        trail: bool,
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
        edge_label: Option<String>,
        max: Option<u32>,
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
    /// `CALL (scope) { <subquery> }` — an inline correlated (lateral) subquery. For
    /// each row of `input`, run `body` (a `Plan::Row`-rooted pattern that continues
    /// from a scope variable) and emit one output row per sub-row: the `input`
    /// row's first `outer_width` slots followed by the `yields` expressions
    /// evaluated over the sub-row. An outer row with no sub-row is dropped (inner
    /// lateral join); the `OPTIONAL` variant and an aggregating subquery are
    /// deferred. Only the `yields` columns (not the subquery's internal variables)
    /// survive into the outer scope.
    CallInline {
        input: Box<Plan>,
        body: Box<Plan>,
        yields: Vec<(String, Expr)>,
        outer_width: usize,
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
    /// A write over the rows of a read sub-plan: for each matched row, apply the
    /// `ops` (SET/REMOVE) to the bound nodes. Run through `exec::execute`, not
    /// pulled; produces no rows (a RETURN after an update is a later slice).
    Update { input: Box<Plan>, ops: Vec<SetOp> },
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
    /// Delete the bound node in `slot` (Gremlin `drop()` on vertices; the future
    /// GQL `DELETE`). Applied in op order alongside SET/REMOVE.
    Delete {
        slot: usize,
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

impl Plan {
    #[must_use]
    pub fn expand(self, from: usize, dir: Dir, edge_label: Option<&str>) -> Self {
        Self::Expand {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.map(str::to_string),
            bind_edge: false,
        }
    }

    /// Like [`Self::expand`] but also binds the traversed edge as a slot (edge
    /// slot then node slot). Used for `(a)-[r:T]->(b)` where `r` is read.
    #[must_use]
    pub fn expand_edge(self, from: usize, dir: Dir, edge_label: Option<&str>) -> Self {
        Self::Expand {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.map(str::to_string),
            bind_edge: true,
        }
    }

    #[must_use]
    pub fn var_length(
        self,
        from: usize,
        dir: Dir,
        edge_label: Option<&str>,
        min: u32,
        max: u32,
        trail: bool,
    ) -> Self {
        Self::VarLength {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.map(str::to_string),
            min,
            max,
            trail,
        }
    }

    #[must_use]
    pub fn shortest_path(
        self,
        from: usize,
        dir: Dir,
        edge_label: Option<&str>,
        max: Option<u32>,
    ) -> Self {
        Self::ShortestPath {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.map(str::to_string),
            max,
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
    pub fn order_page(self, keys: Vec<SortKey>, skip: Option<usize>, limit: Option<usize>) -> Self {
        Self::OrderPage {
            input: Box::new(self),
            keys,
            skip,
            limit,
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
    pub fn distinct(self) -> Self {
        Self::Distinct {
            input: Box::new(self),
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
