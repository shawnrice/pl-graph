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
    /// Produce output columns: `(name, expr)` per column.
    Project {
        input: Box<Plan>,
        items: Vec<(String, Expr)>,
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
    pub fn project(self, items: Vec<(String, Expr)>) -> Self {
        Self::Project {
            input: Box::new(self),
            items,
        }
    }
}
