//! The neutral, language-agnostic algebra. GQL and Gremlin both compile INTO
//! this; nothing below the (future) front-ends knows which language produced a
//! plan. The first slice defines the relational core needed to scan, filter, and
//! project; graph operators (Expand, VarLength, …), effects, and the lineage
//! requirement annotation join as later slices land.

use crate::value::Value;

/// An expression over the current row. `Var` is the current element/value of the
/// row; `Prop` reads a property off it. Binding slots (for multi-variable
/// patterns) join when Expand lands — the first slice has a single current row.
#[derive(Clone, Debug)]
pub enum Expr {
    /// The current row's value (e.g. the scanned node).
    Var,
    /// A property of the current element.
    Prop {
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

/// A logical plan node. A plan is a tree; execution pulls a batch up through it.
#[derive(Clone, Debug)]
pub enum Plan {
    /// Seed the frontier: a label bucket, or the universe when `label` is None.
    Scan { label: Option<String> },
    /// Keep rows where `pred` evaluates to TRUE (three-valued: FALSE and NULL
    /// drop).
    Filter { input: Box<Plan>, pred: Expr },
    /// Produce output columns: `(name, expr)` per column.
    Project {
        input: Box<Plan>,
        items: Vec<(String, Expr)>,
    },
}

impl Plan {
    #[must_use]
    pub fn filter(self, pred: Expr) -> Self {
        Self::Filter {
            input: Box::new(self),
            pred,
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
