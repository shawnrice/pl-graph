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
            Self::Str(v) => Value::Str(v[i].clone().into()),
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
    /// The full per-step TRAVERSER HISTORY for Gremlin `path()`/`tree()` — the ordered
    /// sequence of values the traverser has been, one entry per value-producing step, exactly
    /// like the pure-TS engine's `Traverser.path`. Unlike `values`/`edges` (which are the
    /// GQL node/edge path — a pattern's bound elements), this interleaves vertices, edges AND
    /// projected scalars in the order the steps ran, so `V().values('name').path()` yields
    /// `[v, 'name']` and `E().path()` yields `[e]`. Row `i`'s history is
    /// `steps[step_off[i]..step_off[i+1]]`, each element tagged by `step_tag`
    /// (0 = node id, 1 = edge id, 2 = scalar value). Present only when a Gremlin full-path is
    /// read (see `needs_gremlin_path`); empty otherwise.
    pub steps: Vec<Value>,
    pub step_tag: Vec<u8>,
    pub step_off: Vec<usize>,
}

/// Tag for a `Lineage::steps` history element.
pub const STEP_NODE: u8 = 0;
pub const STEP_EDGE: u8 = 1;
pub const STEP_SCALAR: u8 = 2;

impl Lineage {
    /// Seed one single-node path per node — what a lineage-tracking Scan produces. The
    /// Gremlin step-history is seeded with the same node (the source vertex is the first
    /// path element); `PathRecord` appends every subsequent value-producing step.
    #[must_use]
    pub fn seed(nodes: &[u32]) -> Self {
        Self {
            values: nodes.iter().map(|&n| Value::Num(f64::from(n))).collect(),
            offsets: (0..=nodes.len()).collect(),
            edges: Vec::new(),
            edge_offsets: vec![0; nodes.len() + 1], // each seed path has 0 edges
            steps: nodes.iter().map(|&n| Value::Num(f64::from(n))).collect(),
            step_tag: vec![STEP_NODE; nodes.len()],
            step_off: (0..=nodes.len()).collect(),
        }
    }

    /// Seed a single-element step-history per row from arbitrary VALUES with the given tag
    /// (`inject(v)` → each `v`'s path is `[v]`). The node/edge path stays empty.
    #[must_use]
    pub fn seed_steps(vals: &[Value], tag: u8) -> Self {
        Self {
            values: Vec::new(),
            offsets: vec![0; vals.len() + 1],
            edges: Vec::new(),
            edge_offsets: vec![0; vals.len() + 1],
            steps: vals.to_vec(),
            step_tag: vec![tag; vals.len()],
            step_off: (0..=vals.len()).collect(),
        }
    }

    /// Seed one single-EDGE step-history per edge — the `E()` source (`E().path()` yields
    /// `[e]`). The node/edge path stays empty (GQL does not read an edge-sourced lineage).
    #[must_use]
    pub fn seed_edges(edges: &[u32]) -> Self {
        Self {
            values: Vec::new(),
            offsets: vec![0; edges.len() + 1],
            edges: Vec::new(),
            edge_offsets: vec![0; edges.len() + 1],
            steps: edges.iter().map(|&e| Value::Num(f64::from(e))).collect(),
            step_tag: vec![STEP_EDGE; edges.len()],
            step_off: (0..=edges.len()).collect(),
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
            steps: Vec::new(),
            step_tag: Vec::new(),
            step_off: vec![0],
        }
    }

    /// Row `i`'s Gremlin step-history slice, paired with the per-element tags. Empty when the
    /// row has no history recorded (a lineage that carried no `steps`, or a row index past the
    /// recorded offsets — e.g. a path read inside a branch whose arm did not record steps).
    #[must_use]
    pub fn steps_at(&self, i: usize) -> (&[Value], &[u8]) {
        if i + 1 >= self.step_off.len() {
            return (&[], &[]);
        }
        let (a, b) = (self.step_off[i], self.step_off[i + 1]);
        (&self.steps[a..b], &self.step_tag[a..b])
    }

    /// Append one value per row to the step-history (what `PathRecord` produces): row `i`'s
    /// history grows by `(vals[i], tag)`. Preserves the node/edge path unchanged.
    #[must_use]
    pub fn push_step(&self, vals: &[Value], tag: u8) -> Self {
        let rows = self.offsets.len().saturating_sub(1);
        let mut steps = Vec::with_capacity(self.steps.len() + rows);
        let mut step_tag = Vec::with_capacity(self.step_tag.len() + rows);
        let mut step_off = vec![0usize];
        for (i, v) in vals.iter().enumerate().take(rows) {
            let (sv, st) = self.steps_at(i);
            steps.extend_from_slice(sv);
            step_tag.extend_from_slice(st);
            steps.push(v.clone());
            step_tag.push(tag);
            step_off.push(steps.len());
        }
        Self {
            values: self.values.clone(),
            offsets: self.offsets.clone(),
            edges: self.edges.clone(),
            edge_offsets: self.edge_offsets.clone(),
            steps,
            step_tag,
            step_off,
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
        let mut steps = Vec::new();
        let mut step_tag = Vec::new();
        let mut step_off = vec![0usize];
        let has_steps = self.step_off.len() > 1;
        for &i in idx {
            values.extend_from_slice(self.path_at(i));
            offsets.push(values.len());
            edges.extend_from_slice(self.edges_at(i));
            edge_offsets.push(edges.len());
            if has_steps {
                let (sv, st) = self.steps_at(i);
                steps.extend_from_slice(sv);
                step_tag.extend_from_slice(st);
            }
            step_off.push(steps.len());
        }
        Self {
            values,
            offsets,
            edges,
            edge_offsets,
            steps,
            step_tag,
            step_off,
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
        // The Gremlin step-history is carried through UNCHANGED (gathered by `keep`); the new
        // node/edge is recorded into it by a following `PathRecord`, not here — an Expand
        // serves both a GQL pattern (edge in the path) and a Gremlin `out()` (edge NOT in the
        // path), so the step append is decided by the lowering, per step.
        let mut steps = Vec::new();
        let mut step_tag = Vec::new();
        let mut step_off = vec![0usize];
        let has_steps = self.step_off.len() > 1;
        for (i, &k) in keep.iter().enumerate() {
            values.extend_from_slice(self.path_at(k));
            values.push(Value::Num(f64::from(new_nodes[i])));
            offsets.push(values.len());
            edges.extend_from_slice(self.edges_at(k));
            edges.push(Value::Num(f64::from(new_edges[i])));
            edge_offsets.push(edges.len());
            if has_steps {
                let (sv, st) = self.steps_at(k);
                steps.extend_from_slice(sv);
                step_tag.extend_from_slice(st);
            }
            step_off.push(steps.len());
        }
        Self {
            values,
            offsets,
            edges,
            edge_offsets,
            steps,
            step_tag,
            step_off,
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
        let mut steps = Vec::new();
        let mut step_tag = Vec::new();
        let mut step_off = vec![0usize];
        for lin in parts {
            let rows = lin.offsets.len().saturating_sub(1);
            let has_steps = lin.step_off.len() > 1;
            for i in 0..rows {
                values.extend_from_slice(lin.path_at(i));
                offsets.push(values.len());
                edges.extend_from_slice(lin.edges_at(i));
                edge_offsets.push(edges.len());
                if has_steps {
                    let (sv, st) = lin.steps_at(i);
                    steps.extend_from_slice(sv);
                    step_tag.extend_from_slice(st);
                }
                step_off.push(steps.len());
            }
        }
        Self {
            values,
            offsets,
            edges,
            edge_offsets,
            steps,
            step_tag,
            step_off,
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
