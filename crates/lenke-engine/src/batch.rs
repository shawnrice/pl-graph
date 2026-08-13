//! The one batch type every operator produces and consumes.
//!
//! A batch is a columnar block of rows. Its `col` is the current value of each
//! row — unboxed where the type allows (`Num`/`Bool`/`Str`), node ids for a
//! frontier (`Nodes`), or a boxed fallback (`Gen`). Its `lineage` sidecar is
//! present only when the plan needs path/tags/sack; the first slice is
//! lineage-free, so the sidecar is scaffolded but always `None` here — the shape
//! is in place for the operators that will carry it.

use std::sync::Arc;

use crate::value::Value;

/// The current-value column of a batch. Unboxed variants are the fast path; the
/// bulk operator strategies read and write these without ever boxing a `Value`.
#[derive(Clone, Debug)]
pub enum Col {
    /// A frontier of node ids.
    Nodes(Vec<u32>),
    /// A frontier of edge ids (the traversed relationships), bound by an Expand
    /// that also binds its edge. A `Prop` on an edge slot reads an edge property.
    Edges(Vec<u32>),
    Num(Vec<f64>),
    Bool(Vec<bool>),
    Str(Vec<Arc<str>>),
    /// Mixed or otherwise unboxable values.
    Gen(Vec<Value>),
}

impl Col {
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Nodes(v) | Self::Edges(v) => v.len(),
            Self::Num(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::Str(v) => v.len(),
            Self::Gen(v) => v.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `i` as a boxed `Value`. The boxing edge — used by the boxed/`Gen`
    /// evaluation paths and by result materialization, not by the bulk kernels.
    #[must_use]
    pub fn value_at(&self, i: usize) -> Value {
        match self {
            // A node id surfaced as a value is still a node; there is no distinct
            // Value::Node yet in this slice, so it reads through as its id in Num
            // form only where a caller asked for a value column, never here — a
            // Nodes column is an element frontier, not a value. Callers that want
            // a value read a property off it instead.
            // `u32::MAX` is the OPTIONAL-MATCH null sentinel (never a real dense id)
            // — an unmatched optional element reads back as NULL.
            Self::Nodes(v) | Self::Edges(v) if v[i] == u32::MAX => Value::Null,
            Self::Nodes(v) | Self::Edges(v) => Value::Num(f64::from(v[i])),
            Self::Num(v) => Value::Num(v[i]),
            Self::Bool(v) => Value::Bool(v[i]),
            Self::Str(v) => Value::Str(v[i].clone()),
            Self::Gen(v) => v[i].clone(),
        }
    }

    /// Gather the rows at `idx`, in that order — the primitive filter/reorder
    /// operators use to keep a subset without boxing.
    #[must_use]
    pub fn gather(&self, idx: &[usize]) -> Self {
        match self {
            Self::Nodes(v) => Self::Nodes(idx.iter().map(|&i| v[i]).collect()),
            Self::Edges(v) => Self::Edges(idx.iter().map(|&i| v[i]).collect()),
            Self::Num(v) => Self::Num(idx.iter().map(|&i| v[i]).collect()),
            Self::Bool(v) => Self::Bool(idx.iter().map(|&i| v[i]).collect()),
            Self::Str(v) => Self::Str(idx.iter().map(|&i| v[i].clone()).collect()),
            Self::Gen(v) => Self::Gen(idx.iter().map(|&i| v[i].clone()).collect()),
        }
    }
}

/// The lineage sidecar: the per-row path, present ONLY when the plan reads it.
/// This is the design's heart — the same batch carries lineage or not, decided
/// per plan, instead of a separate traverser type. Row `i`'s path is
/// `values[offsets[i]..offsets[i+1]]` (Arrow-style list layout: `offsets` has
/// `rows + 1` entries, `offsets[0] == 0`). Path elements are node ids as
/// `Value::Num` for now (there is no dedicated node value yet); tags and sack
/// join this struct when their operators do.
///
/// A path of *k* nodes has *k − 1* edges, carried in a PARALLEL list
/// (`edges`/`edge_offsets`) so `relationships(p)` / `elements(p)` can name the
/// traversed relationships, not just the nodes. Edge ids are `Value::Num` too
/// (no dedicated edge value yet). A single-node seed path has zero edges.
#[derive(Clone, Debug, Default)]
pub struct Lineage {
    pub values: Vec<Value>,
    pub offsets: Vec<usize>,
    pub edges: Vec<Value>,
    pub edge_offsets: Vec<usize>,
}

impl Lineage {
    /// Seed one single-node path per node — what a lineage-tracking Scan produces.
    #[must_use]
    pub fn seed(nodes: &[u32]) -> Self {
        Self {
            values: nodes.iter().map(|&n| Value::Num(f64::from(n))).collect(),
            offsets: (0..=nodes.len()).collect(),
            edges: Vec::new(),
            edge_offsets: vec![0; nodes.len() + 1], // each seed path has 0 edges
        }
    }

    /// An empty sidecar (zero rows) — for a lineage-tracking operator that
    /// produced nothing.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            values: Vec::new(),
            offsets: vec![0],
            edges: Vec::new(),
            edge_offsets: vec![0],
        }
    }

    /// Row `i`'s node path.
    #[must_use]
    pub fn path_at(&self, i: usize) -> &[Value] {
        &self.values[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Row `i`'s edge path (the relationships traversed).
    #[must_use]
    pub fn edges_at(&self, i: usize) -> &[Value] {
        &self.edges[self.edge_offsets[i]..self.edge_offsets[i + 1]]
    }

    /// Reorder/subset the paths by `idx` (parallel to a slot gather).
    #[must_use]
    pub fn gather(&self, idx: &[usize]) -> Self {
        let mut values = Vec::new();
        let mut offsets = vec![0usize];
        let mut edges = Vec::new();
        let mut edge_offsets = vec![0usize];
        for &i in idx {
            values.extend_from_slice(self.path_at(i));
            offsets.push(values.len());
            edges.extend_from_slice(self.edges_at(i));
            edge_offsets.push(edges.len());
        }
        Self {
            values,
            offsets,
            edges,
            edge_offsets,
        }
    }

    /// One output path per `(keep[k], new_nodes[k], new_edges[k])`: the input row
    /// `keep[k]`'s path extended by the node reached over the edge traversed —
    /// what a lineage-tracking Expand produces.
    #[must_use]
    pub fn extend(&self, keep: &[usize], new_nodes: &[u32], new_edges: &[u32]) -> Self {
        let mut values = Vec::new();
        let mut offsets = vec![0usize];
        let mut edges = Vec::new();
        let mut edge_offsets = vec![0usize];
        for (i, &k) in keep.iter().enumerate() {
            values.extend_from_slice(self.path_at(k));
            values.push(Value::Num(f64::from(new_nodes[i])));
            offsets.push(values.len());
            edges.extend_from_slice(self.edges_at(k));
            edges.push(Value::Num(f64::from(new_edges[i])));
            edge_offsets.push(edges.len());
        }
        Self {
            values,
            offsets,
            edges,
            edge_offsets,
        }
    }

    /// Concatenate several sidecars row-wise (union/coalesce branch output): each
    /// input's paths in order, so the result is row-aligned with concatenated slot
    /// columns.
    #[must_use]
    pub fn concat(parts: &[&Lineage]) -> Self {
        let mut values = Vec::new();
        let mut offsets = vec![0usize];
        let mut edges = Vec::new();
        let mut edge_offsets = vec![0usize];
        for lin in parts {
            let rows = lin.offsets.len().saturating_sub(1);
            for i in 0..rows {
                values.extend_from_slice(lin.path_at(i));
                offsets.push(values.len());
                edges.extend_from_slice(lin.edges_at(i));
                edge_offsets.push(edges.len());
            }
        }
        Self {
            values,
            offsets,
            edges,
            edge_offsets,
        }
    }
}

/// One batch flowing between operators. A batch is a set of row-aligned SLOT
/// columns — row `i` is the tuple `(slots[0][i], slots[1][i], …)`. Scan binds
/// slot 0; each Expand appends a slot for the node it lands on; a multi-variable
/// pattern like `(a)-[:R]->(b)` is two slots. This is the uniform binding the IR
/// relies on — a GQL variable and a Gremlin `as()` label are both "slot N".
#[derive(Clone, Debug)]
pub struct Batch {
    pub slots: Vec<Col>,
    pub lineage: Option<Lineage>,
}

impl Batch {
    /// A one-slot, lineage-free batch (the output of a Scan).
    #[must_use]
    pub fn single(col: Col) -> Self {
        Self {
            slots: vec![col],
            lineage: None,
        }
    }

    /// A lineage-free batch of several slots.
    #[must_use]
    pub fn of(slots: Vec<Col>) -> Self {
        Self {
            slots,
            lineage: None,
        }
    }

    /// Row count (every slot has the same length).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.slots.first().map_or(0, Col::len)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows() == 0
    }

    #[must_use]
    pub fn slot(&self, i: usize) -> &Col {
        &self.slots[i]
    }

    /// Gather every slot AND the lineage by the same row indices — the primitive
    /// Filter, OrderPage, and Distinct use to keep/reorder rows while staying
    /// row-aligned. The lineage must be gathered too, or a reorder would
    /// desynchronize paths from their rows.
    #[must_use]
    pub fn gather(&self, idx: &[usize]) -> Self {
        Self {
            slots: self.slots.iter().map(|c| c.gather(idx)).collect(),
            lineage: self.lineage.as_ref().map(|l| l.gather(idx)),
        }
    }
}
