//! A minimal typed columnar graph store — enough for the first execution slice
//! (scan a label, read a property, filter, project). Nodes only for now; edges,
//! adjacency, indexes, and temporal columns join in later slices.
//!
//! Properties are stored in TYPED columns (`Column`), not boxed values, so a
//! numeric property arrives at the batch layer as an unboxed `f64` run. That is
//! the whole point of the columnar model, present from the first slice rather
//! than retrofitted.

use std::collections::HashMap;
use std::sync::Arc;

use crate::value::Value;

/// A typed property column, indexed by node id. `present[i]` is false where node
/// `i` does not carry this property (reads as `Value::Null`). One variant per
/// value type; a heterogeneous property falls to `Gen`.
#[derive(Debug)]
pub enum Column {
    Num {
        data: Vec<f64>,
        present: Vec<bool>,
    },
    Str {
        data: Vec<Arc<str>>,
        present: Vec<bool>,
    },
    Bool {
        data: Vec<bool>,
        present: Vec<bool>,
    },
    Gen {
        data: Vec<Value>,
        present: Vec<bool>,
    },
}

impl Column {
    fn with_capacity_num(n: usize) -> Self {
        Self::Num {
            data: vec![0.0; n],
            present: vec![false; n],
        }
    }

    /// Read node `i`'s value from this column (NULL if absent). The per-node
    /// accessor operators use when they hold a `&Column` directly.
    #[must_use]
    pub fn read(&self, i: usize) -> Value {
        let idx = i;
        match self {
            Self::Num { data, present } if present[idx] => Value::Num(data[idx]),
            Self::Str { data, present } if present[idx] => Value::Str(data[idx].clone()),
            Self::Bool { data, present } if present[idx] => Value::Bool(data[idx]),
            Self::Gen { data, present } if present[idx] => data[idx].clone(),
            _ => Value::Null,
        }
    }

    /// The number of node slots this column holds (== `node_count`).
    fn len(&self) -> usize {
        match self {
            Self::Num { present, .. }
            | Self::Str { present, .. }
            | Self::Bool { present, .. }
            | Self::Gen { present, .. } => present.len(),
        }
    }

    /// A fresh column sized for `n` nodes, all absent, whose type matches `v`'s —
    /// or `Gen` for a value with no unboxed column form (`Null`, `List`).
    fn new_absent(v: &Value, n: usize) -> Self {
        match v {
            Value::Num(_) => Self::Num {
                data: vec![0.0; n],
                present: vec![false; n],
            },
            Value::Str(_) => Self::Str {
                data: vec![Arc::from(""); n],
                present: vec![false; n],
            },
            Value::Bool(_) => Self::Bool {
                data: vec![false; n],
                present: vec![false; n],
            },
            Value::Null | Value::List(_) => Self::Gen {
                data: vec![Value::Null; n],
                present: vec![false; n],
            },
        }
    }

    /// Append one absent slot — what every existing column does when a node is
    /// added, so all columns stay length `node_count`.
    fn push_absent(&mut self) {
        match self {
            Self::Num { data, present } => {
                data.push(0.0);
                present.push(false);
            }
            Self::Str { data, present } => {
                data.push(Arc::from(""));
                present.push(false);
            }
            Self::Bool { data, present } => {
                data.push(false);
                present.push(false);
            }
            Self::Gen { data, present } => {
                data.push(Value::Null);
                present.push(false);
            }
        }
    }

    /// Whether this column can store `v` without a type change.
    fn accepts(&self, v: &Value) -> bool {
        matches!(
            (self, v),
            (Self::Num { .. }, Value::Num(_))
                | (Self::Str { .. }, Value::Str(_))
                | (Self::Bool { .. }, Value::Bool(_))
                | (Self::Gen { .. }, _)
        )
    }

    /// Rebuild as a `Gen` column preserving present values — the promotion a typed
    /// column undergoes when a value of another type is written to it.
    fn to_gen(&self) -> Self {
        let n = self.len();
        let mut data = vec![Value::Null; n];
        let mut present = vec![false; n];
        for i in 0..n {
            if self.present_at(i) {
                data[i] = self.read(i);
                present[i] = true;
            }
        }
        Self::Gen { data, present }
    }

    fn present_at(&self, i: usize) -> bool {
        match self {
            Self::Num { present, .. }
            | Self::Str { present, .. }
            | Self::Bool { present, .. }
            | Self::Gen { present, .. } => present[i],
        }
    }

    /// Set node `i` to `v`, marking it present. The caller guarantees the column
    /// `accepts` `v` (promoting to `Gen` first if not).
    fn set(&mut self, i: usize, v: Value) {
        match (self, v) {
            (Self::Num { data, present }, Value::Num(x)) => {
                data[i] = x;
                present[i] = true;
            }
            (Self::Str { data, present }, Value::Str(s)) => {
                data[i] = s;
                present[i] = true;
            }
            (Self::Bool { data, present }, Value::Bool(b)) => {
                data[i] = b;
                present[i] = true;
            }
            (Self::Gen { data, present }, v) => {
                data[i] = v;
                present[i] = true;
            }
            _ => unreachable!("column must accept the value (promote to Gen first)"),
        }
    }

    /// Mark node `i` absent — a removed property reads as NULL again.
    fn set_absent(&mut self, i: usize) {
        match self {
            Self::Num { present, .. }
            | Self::Str { present, .. }
            | Self::Bool { present, .. }
            | Self::Gen { present, .. } => present[i] = false,
        }
    }
}

/// One adjacency entry: the neighbour node, the edge's interned type id, and the
/// edge's identity (`eid`). A directed edge appears once in its source's `out`
/// and once in its target's `in`, both with the SAME `eid` — so trail semantics
/// (no edge reused within one path) can dedup on `eid` regardless of direction.
#[derive(Clone, Copy, Debug)]
pub struct Adj {
    pub nbr: u32,
    pub etype: u32,
    pub eid: u32,
}

/// The graph. Nodes are dense ids `0..node_count`. Labels and properties are
/// looked up by name; a label bucket (`by_label`) is the seed for a scan.
/// Adjacency is per-node out/in lists (a simple layout for now; a CSR pack is a
/// later optimization that changes nothing above this module).
#[derive(Default)]
pub struct Store {
    node_count: usize,
    /// label name -> the sorted node ids carrying it (the scan seed).
    by_label: HashMap<String, Vec<u32>>,
    /// property name -> its typed column (length == node_count).
    props: HashMap<String, Column>,
    /// edge-type name -> interned id, and the reverse.
    etype_ids: HashMap<String, u32>,
    /// per-node outgoing / incoming adjacency, indexed by node id.
    out_adj: Vec<Vec<Adj>>,
    in_adj: Vec<Vec<Adj>>,
    /// next edge id to hand out — monotonic, so an out/in pair shares one id and
    /// ids stay unique across incremental writes.
    next_eid: u32,
    /// tombstones, indexed by node id. A deleted node keeps its id slot (ids are
    /// dense and never reused) but is skipped by every scan and carries no edges
    /// or properties. `deleted.len() == node_count`.
    deleted: Vec<bool>,
}

impl Store {
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// The interned id for an edge-type name, or `None` if no edge ever used it
    /// (so a hop on it matches nothing).
    #[must_use]
    pub fn etype_id(&self, name: &str) -> Option<u32> {
        self.etype_ids.get(name).copied()
    }

    /// A node's outgoing adjacency.
    #[must_use]
    pub fn out(&self, node: u32) -> &[Adj] {
        self.out_adj.get(node as usize).map_or(&[], Vec::as_slice)
    }

    /// A node's incoming adjacency.
    #[must_use]
    pub fn inc(&self, node: u32) -> &[Adj] {
        self.in_adj.get(node as usize).map_or(&[], Vec::as_slice)
    }

    /// Every LIVE node id — the scan universe when no label narrows it. Deleted
    /// ids are skipped (tombstoned), so a whole-graph scan never yields them.
    #[must_use]
    pub fn all_nodes(&self) -> Vec<u32> {
        (0..self.node_count as u32)
            .filter(|&i| !self.deleted[i as usize])
            .collect()
    }

    /// Whether node `id` is live (not tombstoned). Scans that iterate the id space
    /// directly consult this to skip deleted nodes.
    #[must_use]
    pub fn is_alive(&self, id: u32) -> bool {
        !self.deleted[id as usize]
    }

    /// The node ids carrying `label`, or an empty slice for an unknown label
    /// (which matches nothing — never "everything").
    #[must_use]
    pub fn nodes_with_label(&self, label: &str) -> &[u32] {
        self.by_label.get(label).map_or(&[], Vec::as_slice)
    }

    /// A node's value for `key`, or `Null` if the key or the node's entry is
    /// absent. The typed read; the batch layer's bulk gather uses the column
    /// directly (see `column`).
    #[must_use]
    pub fn prop(&self, node: u32, key: &str) -> Value {
        self.props
            .get(key)
            .map_or(Value::Null, |c| c.read(node as usize))
    }

    /// The typed column for `key`, for a bulk gather. `None` = no such property.
    #[must_use]
    pub fn column(&self, key: &str) -> Option<&Column> {
        self.props.get(key)
    }

    // --- Mutation ---------------------------------------------------------
    //
    // The write path. Nodes get the next dense id; every existing property column
    // grows one absent slot so all columns stay length `node_count`. Edges take a
    // monotonic `eid` shared by their out/in entries. Properties are set into a
    // typed column, promoting it to `Gen` on a type change. These are the store
    // primitives the language write statements (Phase B) and transactions (A3)
    // build on; they do not enforce constraints — that is a later, higher layer.

    /// Add a node with `labels` and `(key, value)` properties; returns its id.
    pub fn add_node(&mut self, labels: &[&str], props: &[(&str, Value)]) -> u32 {
        let id = self.node_count as u32;
        self.node_count += 1;
        // Keep every existing column the same length as the node set.
        for col in self.props.values_mut() {
            col.push_absent();
        }
        self.out_adj.push(Vec::new());
        self.in_adj.push(Vec::new());
        self.deleted.push(false);
        for l in labels {
            // ids are handed out increasing, so appending keeps the bucket sorted.
            self.by_label.entry((*l).to_string()).or_default().push(id);
        }
        for (k, v) in props {
            self.set_prop(id, k, v.clone());
        }
        id
    }

    /// Add a directed edge `from -[label]-> to`. Interns the edge type if new and
    /// assigns a fresh `eid` shared by the out and in adjacency entries.
    pub fn add_edge(&mut self, from: u32, to: u32, label: &str) {
        assert!(
            (from as usize) < self.node_count && (to as usize) < self.node_count,
            "edge endpoint out of range"
        );
        let next = self.etype_ids.len() as u32;
        let etype = *self.etype_ids.entry(label.to_string()).or_insert(next);
        let eid = self.next_eid;
        self.next_eid += 1;
        self.out_adj[from as usize].push(Adj {
            nbr: to,
            etype,
            eid,
        });
        self.in_adj[to as usize].push(Adj {
            nbr: from,
            etype,
            eid,
        });
    }

    /// Set node `node`'s `key` to `value`, creating the column if new and
    /// promoting it to `Gen` if `value`'s type differs from the column's.
    pub fn set_prop(&mut self, node: u32, key: &str, value: Value) {
        let n = self.node_count;
        let col = self
            .props
            .entry(key.to_string())
            .or_insert_with(|| Column::new_absent(&value, n));
        if !col.accepts(&value) {
            *col = col.to_gen();
        }
        col.set(node as usize, value);
    }

    /// Remove node `node`'s `key` — it reads as NULL again. (Distinct from setting
    /// it to a stored `Null`; that distinction is a Phase-E concern.)
    pub fn remove_prop(&mut self, node: u32, key: &str) {
        if let Some(col) = self.props.get_mut(key) {
            col.set_absent(node as usize);
        }
    }

    /// Delete the edge identified by `eid` between endpoints `u` and `v`. The eid
    /// is unique and shared by the edge's out/in entries; removing it from both
    /// endpoints' out AND in lists deletes the edge regardless of which endpoint
    /// was its source (so it is safe to call with the endpoints in either order,
    /// e.g. from a hop matched via incoming adjacency). A no-op if already gone.
    pub fn delete_edge(&mut self, u: u32, v: u32, eid: u32) {
        for node in [u, v] {
            if let Some(adj) = self.out_adj.get_mut(node as usize) {
                adj.retain(|a| a.eid != eid);
            }
            if let Some(adj) = self.in_adj.get_mut(node as usize) {
                adj.retain(|a| a.eid != eid);
            }
        }
    }

    /// Delete node `id`: tombstone it (its dense id is never reused), detach every
    /// incident edge from the neighbour's mirror list, drop its adjacency, remove
    /// it from every label bucket, and clear its properties. After this it is
    /// absent from all scans and traversals. A no-op if already deleted.
    pub fn delete_node(&mut self, id: u32) {
        let i = id as usize;
        if self.deleted[i] {
            return;
        }
        // Detach each incident edge's mirror entry on the neighbour, by eid.
        let out = std::mem::take(&mut self.out_adj[i]);
        for a in &out {
            self.in_adj[a.nbr as usize].retain(|m| m.eid != a.eid);
        }
        let inc = std::mem::take(&mut self.in_adj[i]);
        for a in &inc {
            self.out_adj[a.nbr as usize].retain(|m| m.eid != a.eid);
        }
        // A self-loop appears in both `out` and `inc`; its mirror was in this
        // node's own lists, already emptied by the takes above — nothing dangling.

        // Remove from every label bucket (no per-node label list yet, so sweep).
        for bucket in self.by_label.values_mut() {
            bucket.retain(|&x| x != id);
        }
        // Clear its properties.
        for col in self.props.values_mut() {
            col.set_absent(i);
        }
        self.deleted[i] = true;
    }
}

/// Builds a `Store`. Node ids are assigned in insertion order.
#[derive(Default)]
pub struct Builder {
    node_count: usize,
    by_label: HashMap<String, Vec<u32>>,
    // Collected as (node, value) pairs per key, materialized into typed columns
    // at `build()` — so the builder stays simple and the store stays typed.
    props: HashMap<String, Vec<(u32, Value)>>,
    etype_ids: HashMap<String, u32>,
    edges: Vec<(u32, u32, u32)>, // (from, to, etype)
}

impl Builder {
    /// Add a node with `labels` and `(key, value)` properties; returns its id.
    pub fn node(&mut self, labels: &[&str], props: &[(&str, Value)]) -> u32 {
        let id = self.node_count as u32;
        self.node_count += 1;
        for l in labels {
            self.by_label.entry((*l).to_string()).or_default().push(id);
        }
        for (k, v) in props {
            self.props
                .entry((*k).to_string())
                .or_default()
                .push((id, v.clone()));
        }
        id
    }

    /// Add a directed edge `from -[label]-> to`.
    pub fn edge(&mut self, from: u32, to: u32, label: &str) {
        let next = self.etype_ids.len() as u32;
        let etype = *self.etype_ids.entry(label.to_string()).or_insert(next);
        self.edges.push((from, to, etype));
    }

    #[must_use]
    pub fn build(self) -> Store {
        let n = self.node_count;
        let props = self
            .props
            .into_iter()
            .map(|(k, pairs)| (k, materialize(pairs, n)))
            .collect();
        let mut out_adj = vec![Vec::new(); n];
        let mut in_adj = vec![Vec::new(); n];
        let edge_count = self.edges.len() as u32;
        for (eid, (from, to, etype)) in self.edges.into_iter().enumerate() {
            let eid = eid as u32;
            out_adj[from as usize].push(Adj {
                nbr: to,
                etype,
                eid,
            });
            in_adj[to as usize].push(Adj {
                nbr: from,
                etype,
                eid,
            });
        }
        Store {
            node_count: n,
            by_label: self.by_label,
            props,
            etype_ids: self.etype_ids,
            out_adj,
            in_adj,
            // Incremental edges continue the id sequence the build laid down.
            next_eid: edge_count,
            deleted: vec![false; n],
        }
    }
}

/// Turn `(node, value)` pairs into the tightest typed column. Homogeneous
/// numeric/string/bool columns unbox; anything mixed falls to `Gen`.
fn materialize(pairs: Vec<(u32, Value)>, n: usize) -> Column {
    let all = |f: &dyn Fn(&Value) -> bool| pairs.iter().all(|(_, v)| f(v));
    if all(&|v| matches!(v, Value::Num(_))) {
        let mut col = Column::with_capacity_num(n);
        if let Column::Num { data, present } = &mut col {
            for (i, v) in pairs {
                if let Value::Num(x) = v {
                    data[i as usize] = x;
                    present[i as usize] = true;
                }
            }
        }
        col
    } else if all(&|v| matches!(v, Value::Str(_))) {
        let mut data: Vec<Arc<str>> = vec![Arc::from(""); n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            if let Value::Str(s) = v {
                data[i as usize] = s;
                present[i as usize] = true;
            }
        }
        Column::Str { data, present }
    } else if all(&|v| matches!(v, Value::Bool(_))) {
        let mut data = vec![false; n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            if let Value::Bool(b) = v {
                data[i as usize] = b;
                present[i as usize] = true;
            }
        }
        Column::Bool { data, present }
    } else {
        let mut data = vec![Value::Null; n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            data[i as usize] = v;
            present[i as usize] = true;
        }
        Column::Gen { data, present }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: &str) -> Value {
        Value::Str(Arc::from(x))
    }
    fn n(x: f64) -> Value {
        Value::Num(x)
    }

    /// Build an empty store, add two nodes and an edge, and read it all back —
    /// hand-verified: ids 0 and 1, one out-edge 0→1 mirrored as an in-edge.
    #[test]
    fn add_nodes_and_edge_then_read_back() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        assert_eq!((a, b), (0, 1));
        assert_eq!(st.node_count(), 2);
        st.add_edge(a, b, "R");
        assert_eq!(st.nodes_with_label("P"), &[0, 1]);
        assert_eq!(st.out(a).len(), 1);
        assert_eq!(st.out(a)[0].nbr, b);
        assert_eq!(st.inc(b)[0].nbr, a);
        assert_eq!(st.out(a)[0].eid, st.inc(b)[0].eid); // shared edge id
        assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
    }

    /// Adding a node AFTER a property column exists extends that column with an
    /// absent slot: the old node keeps its value, the new node reads NULL.
    #[test]
    fn add_node_extends_existing_columns() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("age", n(30.0))]);
        let b = st.add_node(&["P"], &[]); // no age
        assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 30.0));
        assert!(st.prop(b, "age").is_null());
    }

    /// Writing a value of a different type promotes the column to `Gen`; both the
    /// old-typed and new value read back correctly.
    #[test]
    fn set_prop_promotes_on_type_change() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("v", n(1.0))]);
        let b = st.add_node(&["P"], &[("v", n(2.0))]);
        st.set_prop(a, "v", s("two")); // Num column, Str value -> promote to Gen
        assert!(matches!(st.prop(a, "v"), Value::Str(x) if &*x == "two"));
        assert!(matches!(st.prop(b, "v"), Value::Num(x) if x == 2.0)); // preserved
    }

    /// `remove_prop` makes the property read NULL again; overwriting sets it.
    #[test]
    fn set_and_remove_prop() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("age", n(30.0))]);
        st.set_prop(a, "age", n(31.0));
        assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 31.0));
        st.remove_prop(a, "age");
        assert!(st.prop(a, "age").is_null());
    }

    /// A repeated edge type interns once (same `etype`) but each edge gets a
    /// distinct `eid`. Ids continue after a `build()`-created edge.
    #[test]
    fn edge_type_interns_once_ids_unique() {
        let mut b = Builder::default();
        let x = b.node(&["P"], &[]);
        let y = b.node(&["P"], &[]);
        b.edge(x, y, "R"); // eid 0 at build
        let mut st = b.build();
        st.add_edge(x, y, "R"); // eid 1, same type
        st.add_edge(x, y, "S"); // eid 2, new type
        assert_eq!(st.out(x).len(), 3);
        let eids: Vec<u32> = st.out(x).iter().map(|a| a.eid).collect();
        assert_eq!(eids, vec![0, 1, 2]); // continued, unique
        assert_eq!(st.out(x)[0].etype, st.out(x)[1].etype); // R == R
        assert_ne!(st.out(x)[1].etype, st.out(x)[2].etype); // R != S
    }

    /// `delete_edge` removes the edge from both endpoints and is idempotent.
    #[test]
    fn delete_edge_detaches_both_sides() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        st.delete_edge(a, b, eid);
        assert!(st.out(a).is_empty());
        assert!(st.inc(b).is_empty());
        st.delete_edge(a, b, eid); // no-op the second time
        assert!(st.out(a).is_empty());
    }

    /// `delete_node` tombstones the node, detaches its edges from the neighbours'
    /// mirror lists, clears its props, and drops it from scans. Hand-traced on
    /// a→b, a→c, b→c: deleting b leaves a→c only, c with one incoming (from a).
    #[test]
    fn delete_node_tombstones_and_cleans_up() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let c = st.add_node(&["P"], &[("name", s("c"))]);
        st.add_edge(a, b, "R");
        st.add_edge(a, c, "R");
        st.add_edge(b, c, "R");
        st.delete_node(b);

        assert!(!st.is_alive(b));
        assert_eq!(st.all_nodes(), vec![a, c]);
        assert_eq!(st.nodes_with_label("P"), &[a, c]); // b removed from bucket
        assert_eq!(st.out(a).len(), 1); // a→b gone, a→c stays
        assert_eq!(st.out(a)[0].nbr, c);
        assert_eq!(st.inc(c).len(), 1); // b→c gone, a→c stays
        assert_eq!(st.inc(c)[0].nbr, a);
        assert!(st.out(b).is_empty());
        assert!(st.prop(b, "name").is_null()); // props cleared
        assert!(!st.prop(a, "name").is_null()); // neighbour intact
        st.delete_node(b); // idempotent
        assert_eq!(st.all_nodes(), vec![a, c]);
    }

    /// A self-loop is detached without panicking when its node is deleted.
    #[test]
    fn delete_node_with_self_loop() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        st.add_edge(a, a, "R");
        st.delete_node(a);
        assert!(!st.is_alive(a));
        assert!(st.out(a).is_empty());
        assert!(st.inc(a).is_empty());
    }
}
