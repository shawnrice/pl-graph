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
    /// Hop from the element in `from` slot along `dir`/`edge_label`, appending the
    /// landed node as a new slot. Rows without a matching neighbour drop; rows
    /// with several fan out (one output row per neighbour), replicating the
    /// existing slots.
    Expand {
        input: Box<Plan>,
        from: usize,
        dir: Dir,
        edge_label: Option<String>,
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
}

impl Plan {
    #[must_use]
    pub fn expand(self, from: usize, dir: Dir, edge_label: Option<&str>) -> Self {
        Self::Expand {
            input: Box::new(self),
            from,
            dir,
            edge_label: edge_label.map(str::to_string),
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
