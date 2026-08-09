//! A minimal typed columnar graph store — enough for the first execution slice
//! (scan a label, read a property, filter, project). Nodes only for now; edges,
//! adjacency, indexes, and temporal columns join in later slices.
//!
//! Properties are stored in TYPED columns (`Column`), not boxed values, so a
//! numeric property arrives at the batch layer as an unboxed `f64` run. That is
//! the whole point of the columnar model, present from the first slice rather
//! than retrofitted.

use std::collections::{HashMap, HashSet};
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

    /// Drop the last node slot — the inverse of `push_absent`, used when a
    /// transaction rolls back the node that a logged `add_node` appended.
    fn pop_last(&mut self) {
        match self {
            Self::Num { data, present } => {
                data.pop();
                present.pop();
            }
            Self::Str { data, present } => {
                data.pop();
                present.pop();
            }
            Self::Bool { data, present } => {
                data.pop();
                present.pop();
            }
            Self::Gen { data, present } => {
                data.pop();
                present.pop();
            }
        }
    }
}

/// One entry in a transaction's undo log — the inverse of a single mutation,
/// captured with just enough prior state to reverse it exactly. Applied in
/// reverse order on rollback, with logging disabled so undos do not re-log.
enum Undo {
    /// Undo `add_node`: pop the last (highest-id) node. Adds grow `node_count`
    /// monotonically, so reverse-order undo always pops the current top.
    AddNode,
    /// Undo `add_edge`: delete the edge by its id from both endpoints.
    AddEdge { u: u32, v: u32, eid: u32 },
    /// Undo `set_prop`/`remove_prop`: restore the cell to its prior state.
    RestoreCell {
        node: u32,
        key: String,
        prev_present: bool,
        prev_value: Value,
    },
    /// Undo `delete_edge`: re-insert the exact adjacency entries removed, each
    /// tagged `(node, is_out, adj)` so it goes back to the right list.
    RestoreEdge { entries: Vec<(u32, bool, Adj)> },
    /// Undo `delete_node`: un-tombstone and restore its adjacency (both its own
    /// lists and the neighbours' mirrors), label memberships, and properties.
    RestoreNode {
        id: u32,
        out: Vec<Adj>,
        inc: Vec<Adj>,
        labels: Vec<String>,
        props: Vec<(String, bool, Value)>,
    },
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
    /// the active transaction's undo log, or `None` outside a transaction
    /// (autocommit — mutations apply directly and record nothing).
    undo: Option<Vec<Undo>>,
    /// declared unique constraints as `(label, keys)` — at most one live node per
    /// label may carry a given key tuple. Enforced by the write statements, not
    /// the store primitives (which stay infallible for rollback).
    unique: Vec<(String, Vec<String>)>,
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

    // --- Unique constraints ----------------------------------------------
    //
    // A unique constraint declares that at most one live node with `label` may
    // carry a given tuple of `keys` values. Enforced by the write STATEMENTS
    // (execute) after a mutation, not by the store primitives — those stay
    // infallible so rollback can always run. Key equality uses the value
    // contract's grouping (`group_key_into`), so two absent/NULL keys collide
    // (consistent with lenke's first-class-null policy, not SQL's distinct NULLs).

    /// Declare a unique constraint on `(label, keys)`. Errors if the CURRENT data
    /// already violates it (you cannot declare a constraint the graph breaks).
    pub fn create_unique_constraint(&mut self, label: &str, keys: &[&str]) -> Result<(), String> {
        let keys: Vec<String> = keys.iter().map(|s| (*s).to_string()).collect();
        self.check_label_unique(label, &keys)?;
        self.unique.push((label.to_string(), keys));
        Ok(())
    }

    /// Check every unique constraint on `label` against the live nodes; `Err`
    /// names the first violated one. Write statements call this after mutating a
    /// constrained label.
    pub fn check_unique_for_label(&self, label: &str) -> Result<(), String> {
        for (l, keys) in &self.unique {
            if l == label {
                self.check_label_unique(l, keys)?;
            }
        }
        Ok(())
    }

    /// The keys of a unique constraint on `label` all of which appear in `have` —
    /// the `_MERGE` conflict-target inference. `None` if no such constraint.
    #[must_use]
    pub fn unique_keys_for(&self, label: &str, have: &[String]) -> Option<Vec<String>> {
        self.unique
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, keys)| keys)
            .find(|keys| keys.iter().all(|k| have.contains(k)))
            .cloned()
    }

    /// Infer the `_MERGE` conflict target: the one unique constraint on `label`
    /// whose keys are all present in `have`. Errors if there is none (the merge
    /// has no key) or more than one (ambiguous — the pattern must disambiguate).
    pub fn infer_merge_key(&self, label: &str, have: &[String]) -> Result<Vec<String>, String> {
        let covered: Vec<&Vec<String>> = self
            .unique
            .iter()
            .filter(|(l, keys)| l == label && keys.iter().all(|k| have.contains(k)))
            .map(|(_, keys)| keys)
            .collect();
        match covered.as_slice() {
            [] => Err(format!(
                "E_MERGE: _MERGE on `{label}` has no applicable unique constraint"
            )),
            [one] => Ok((*one).clone()),
            _ => Err(format!(
                "E_MERGE: _MERGE on `{label}` is ambiguous — the pattern touches several unique constraints"
            )),
        }
    }

    fn check_label_unique(&self, label: &str, keys: &[String]) -> Result<(), String> {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        for &id in self.nodes_with_label(label) {
            let mut buf = Vec::new();
            for k in keys {
                crate::value::group_key_into(&self.prop(id, k), &mut buf);
            }
            if !seen.insert(buf) {
                return Err(format!(
                    "E_UNIQUE: unique constraint on {label}({}) violated",
                    keys.join(", ")
                ));
            }
        }
        Ok(())
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
            // Apply the initial props directly; the single AddNode undo (which
            // pops the whole node) reverses them, so they are not logged twice.
            self.apply_set_prop(id, k, v.clone());
        }
        if let Some(log) = &mut self.undo {
            log.push(Undo::AddNode);
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
        if let Some(log) = &mut self.undo {
            log.push(Undo::AddEdge {
                u: from,
                v: to,
                eid,
            });
        }
    }

    /// Whether node `node` carries a present value for `key` (distinct from a
    /// present `Null`, which `prop` cannot tell from absence).
    #[must_use]
    pub fn has_prop(&self, node: u32, key: &str) -> bool {
        self.props
            .get(key)
            .is_some_and(|c| c.present_at(node as usize))
    }

    /// Set node `node`'s `key` to `value`, creating the column if new and
    /// promoting it to `Gen` if `value`'s type differs from the column's.
    pub fn set_prop(&mut self, node: u32, key: &str, value: Value) {
        let rec = self.undo.is_some().then(|| Undo::RestoreCell {
            node,
            key: key.to_string(),
            prev_present: self.has_prop(node, key),
            prev_value: self.prop(node, key),
        });
        self.apply_set_prop(node, key, value);
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
    }

    fn apply_set_prop(&mut self, node: u32, key: &str, value: Value) {
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
        let rec = self.undo.is_some().then(|| Undo::RestoreCell {
            node,
            key: key.to_string(),
            prev_present: self.has_prop(node, key),
            prev_value: self.prop(node, key),
        });
        self.apply_remove_prop(node, key);
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
    }

    fn apply_remove_prop(&mut self, node: u32, key: &str) {
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
        let logging = self.undo.is_some();
        let mut removed: Vec<(u32, bool, Adj)> = Vec::new();
        for node in [u, v] {
            if let Some(adj) = self.out_adj.get_mut(node as usize) {
                if logging {
                    removed.extend(
                        adj.iter()
                            .filter(|a| a.eid == eid)
                            .map(|a| (node, true, *a)),
                    );
                }
                adj.retain(|a| a.eid != eid);
            }
            if let Some(adj) = self.in_adj.get_mut(node as usize) {
                if logging {
                    removed.extend(
                        adj.iter()
                            .filter(|a| a.eid == eid)
                            .map(|a| (node, false, *a)),
                    );
                }
                adj.retain(|a| a.eid != eid);
            }
        }
        if let Some(log) = &mut self.undo {
            log.push(Undo::RestoreEdge { entries: removed });
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
        // Capture the full prior state BEFORE mutating, if a transaction is open.
        let (labels, props) = if self.undo.is_some() {
            let labels = self
                .by_label
                .iter()
                .filter(|(_, b)| b.contains(&id))
                .map(|(k, _)| k.clone())
                .collect();
            let props = self
                .props
                .iter()
                .map(|(k, c)| (k.clone(), c.present_at(i), c.read(i)))
                .collect();
            (labels, props)
        } else {
            (Vec::new(), Vec::new())
        };

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

        if let Some(log) = &mut self.undo {
            log.push(Undo::RestoreNode {
                id,
                out,
                inc,
                labels,
                props,
            });
        }
    }

    // --- Transactions ----------------------------------------------------
    //
    // An undo log makes a group of mutations atomic. `begin` opens it; every
    // mutation then records its inverse; `commit` discards the log (changes
    // stand); `rollback` applies the inverses in reverse. `savepoint`/`rollback_to`
    // give per-statement atomicity within a transaction (a statement rolls back
    // to its savepoint on failure without abandoning the whole transaction).
    // Constraint checks and event buffering are deferred to Phase H.

    /// Open a transaction. Panics if one is already open (no nesting yet).
    pub fn begin(&mut self) {
        assert!(self.undo.is_none(), "nested transactions are not supported");
        self.undo = Some(Vec::new());
    }

    /// Commit: the changes stand and the undo log is discarded.
    pub fn commit(&mut self) {
        self.undo = None;
    }

    /// Roll back every change since `begin`, in reverse, and close the
    /// transaction. A no-op outside a transaction.
    pub fn rollback(&mut self) {
        if let Some(log) = self.undo.take() {
            // `undo` is now None, so the inverse mutations below do not re-log.
            for rec in log.into_iter().rev() {
                self.apply_undo(rec);
            }
        }
    }

    /// A mark for per-statement atomicity: the current undo-log length. Zero
    /// outside a transaction.
    #[must_use]
    pub fn savepoint(&self) -> usize {
        self.undo.as_ref().map_or(0, Vec::len)
    }

    /// Undo every change recorded after `mark`, keeping the transaction open and
    /// the changes up to `mark`. Used to roll back a single failed statement.
    pub fn rollback_to(&mut self, mark: usize) {
        if let Some(mut log) = self.undo.take() {
            let mut undone = Vec::new();
            while log.len() > mark {
                undone.push(log.pop().expect("len > mark"));
            }
            // `undo` is None while applying, so these inverses do not re-log.
            for rec in undone {
                self.apply_undo(rec);
            }
            self.undo = Some(log);
        }
    }

    /// Run `f` in a transaction: commit if it returns `Ok`, roll back if `Err`.
    /// (Does not catch panics — an unwinding closure leaves the log un-applied;
    /// panic-safety is a later concern.)
    pub fn transaction<T, E>(
        &mut self,
        f: impl FnOnce(&mut Store) -> Result<T, E>,
    ) -> Result<T, E> {
        self.begin();
        match f(self) {
            Ok(v) => {
                self.commit();
                Ok(v)
            }
            Err(e) => {
                self.rollback();
                Err(e)
            }
        }
    }

    /// Apply one undo record. The caller has taken the log out (`self.undo` is
    /// `None`), so the primitive mutations invoked here do not re-log.
    fn apply_undo(&mut self, rec: Undo) {
        match rec {
            Undo::AddNode => self.pop_last_node(),
            Undo::AddEdge { u, v, eid } => self.delete_edge(u, v, eid),
            Undo::RestoreCell {
                node,
                key,
                prev_present,
                prev_value,
            } => {
                if prev_present {
                    self.apply_set_prop(node, &key, prev_value);
                } else {
                    self.apply_remove_prop(node, &key);
                }
            }
            Undo::RestoreEdge { entries } => {
                for (node, is_out, adj) in entries {
                    if is_out {
                        self.out_adj[node as usize].push(adj);
                    } else {
                        self.in_adj[node as usize].push(adj);
                    }
                }
            }
            Undo::RestoreNode {
                id,
                out,
                inc,
                labels,
                props,
            } => {
                let i = id as usize;
                self.deleted[i] = false;
                // Restore mirrors on OTHER nodes; self-loops live in id's own
                // lists and are restored by the assignments below.
                for a in &out {
                    if a.nbr != id {
                        self.in_adj[a.nbr as usize].push(Adj {
                            nbr: id,
                            etype: a.etype,
                            eid: a.eid,
                        });
                    }
                }
                for a in &inc {
                    if a.nbr != id {
                        self.out_adj[a.nbr as usize].push(Adj {
                            nbr: id,
                            etype: a.etype,
                            eid: a.eid,
                        });
                    }
                }
                self.out_adj[i] = out;
                self.in_adj[i] = inc;
                for l in labels {
                    self.by_label.entry(l).or_default().push(id);
                }
                for (k, present, value) in props {
                    if present {
                        self.apply_set_prop(id, &k, value);
                    } else {
                        self.apply_remove_prop(id, &k);
                    }
                }
            }
        }
    }

    /// Pop the last (highest-id) node — the inverse of a logged `add_node`.
    fn pop_last_node(&mut self) {
        debug_assert!(self.node_count > 0);
        let id = (self.node_count - 1) as u32;
        for b in self.by_label.values_mut() {
            b.retain(|&x| x != id);
        }
        for col in self.props.values_mut() {
            col.pop_last();
        }
        self.out_adj.pop();
        self.in_adj.pop();
        self.deleted.pop();
        self.node_count -= 1;
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
            undo: None,
            unique: Vec::new(),
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

    // --- Transactions ---

    /// Commit keeps the changes; the log is discarded.
    #[test]
    fn commit_keeps_changes() {
        let mut st = Builder::default().build();
        st.begin();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        st.commit();
        assert_eq!(st.node_count(), 1);
        assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
    }

    /// Rolling back an `add_node` truly removes it: node_count returns to 0 and
    /// the columns shrink back (not merely tombstoned).
    #[test]
    fn rollback_add_node_shrinks_back() {
        let mut st = Builder::default().build();
        st.begin();
        st.add_node(&["P"], &[("name", s("a"))]);
        st.add_node(&["P"], &[("name", s("b"))]);
        assert_eq!(st.node_count(), 2);
        st.rollback();
        assert_eq!(st.node_count(), 0);
        assert!(st.all_nodes().is_empty());
        assert!(st.nodes_with_label("P").is_empty());
    }

    /// Rolling back `set_prop` restores the exact prior cell (present value).
    #[test]
    fn rollback_set_prop_restores_value() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("age", n(30.0))]); // committed (autocommit)
        st.begin();
        st.set_prop(a, "age", n(99.0));
        st.set_prop(a, "age", s("oops")); // also promotes column to Gen
        assert!(matches!(st.prop(a, "age"), Value::Str(x) if &*x == "oops"));
        st.rollback();
        assert!(matches!(st.prop(a, "age"), Value::Num(x) if x == 30.0));
    }

    /// Rolling back a newly-set property (absent before) makes it absent again.
    #[test]
    fn rollback_new_prop_becomes_absent() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]);
        st.begin();
        st.set_prop(a, "age", n(30.0));
        st.rollback();
        assert!(st.prop(a, "age").is_null());
        assert!(!st.has_prop(a, "age"));
    }

    /// Rolling back `add_edge` removes it from both endpoints.
    #[test]
    fn rollback_add_edge() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.begin();
        st.add_edge(a, b, "R");
        st.rollback();
        assert!(st.out(a).is_empty());
        assert!(st.inc(b).is_empty());
    }

    /// Rolling back `delete_node` restores it fully: tombstone, adjacency (its own
    /// lists AND the neighbours' mirrors), label membership, and properties.
    /// Hand-traced on a→b, b→c: delete b, then roll back → identical to before.
    #[test]
    fn rollback_delete_node_restores_everything() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("name", s("a"))]);
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        let c = st.add_node(&["P"], &[("name", s("c"))]);
        st.add_edge(a, b, "R");
        st.add_edge(b, c, "R");
        st.begin();
        st.delete_node(b);
        assert!(!st.is_alive(b));
        st.rollback();

        assert!(st.is_alive(b));
        assert_eq!(st.nodes_with_label("P").len(), 3);
        assert!(matches!(st.prop(b, "name"), Value::Str(x) if &*x == "b"));
        // adjacency restored on all three nodes
        assert_eq!(st.out(a).len(), 1); // a→b
        assert_eq!(st.out(a)[0].nbr, b);
        assert_eq!(st.out(b).len(), 1); // b→c
        assert_eq!(st.out(b)[0].nbr, c);
        assert_eq!(st.inc(b).len(), 1); // a→b mirror
        assert_eq!(st.inc(c).len(), 1); // b→c mirror
    }

    /// `savepoint` + `rollback_to` give per-statement atomicity: the first
    /// statement's writes survive, the second's are undone, the transaction stays
    /// open, and the final commit keeps only the first.
    #[test]
    fn savepoint_rolls_back_one_statement() {
        let mut st = Builder::default().build();
        st.begin();
        let a = st.add_node(&["P"], &[("name", s("a"))]); // statement 1
        let mark = st.savepoint();
        let b = st.add_node(&["P"], &[("name", s("b"))]); // statement 2
        st.add_edge(a, b, "R");
        st.rollback_to(mark); // undo statement 2 only
        assert_eq!(st.node_count(), 1); // b popped
        assert!(st.out(a).is_empty()); // edge gone
        st.commit();
        assert_eq!(st.node_count(), 1);
        assert!(matches!(st.prop(a, "name"), Value::Str(x) if &*x == "a"));
    }

    // --- Unique constraints ---

    /// A unique constraint on already-conforming data is accepted; check passes.
    #[test]
    fn unique_constraint_accepts_conforming_data() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("a@x"))]);
        st.add_node(&["User"], &[("email", s("b@x"))]);
        assert!(st.create_unique_constraint("User", &["email"]).is_ok());
        assert!(st.check_unique_for_label("User").is_ok());
    }

    /// Declaring a constraint the data already violates errors.
    #[test]
    fn unique_constraint_rejects_existing_duplicate() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("dup"))]);
        st.add_node(&["User"], &[("email", s("dup"))]);
        assert!(st.create_unique_constraint("User", &["email"]).is_err());
    }

    /// After a constraint, a duplicate added at the store level is detected by the
    /// check (the store primitive itself stays infallible; enforcement is the
    /// caller's, as the write statements do).
    #[test]
    fn unique_check_detects_new_duplicate() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("x"))]);
        st.create_unique_constraint("User", &["email"]).unwrap();
        st.add_node(&["User"], &[("email", s("x"))]); // primitive allows it
        assert!(st.check_unique_for_label("User").is_err()); // check catches it
    }

    /// Conflict-target inference: the constraint keys are returned when the
    /// pattern's key set covers them.
    #[test]
    fn unique_keys_for_infers_target() {
        let mut st = Builder::default().build();
        st.create_unique_constraint("User", &["email"]).unwrap();
        assert_eq!(
            st.unique_keys_for("User", &["email".into(), "name".into()]),
            Some(vec!["email".into()])
        );
        assert_eq!(st.unique_keys_for("User", &["name".into()]), None);
        assert_eq!(st.unique_keys_for("Other", &["email".into()]), None);
    }

    /// `transaction` commits on `Ok` and rolls back on `Err`.
    #[test]
    fn transaction_commits_ok_rolls_back_err() {
        let mut st = Builder::default().build();
        let r: Result<u32, ()> = st.transaction(|s| Ok(s.add_node(&["P"], &[])));
        assert!(r.is_ok());
        assert_eq!(st.node_count(), 1);

        let r: Result<(), &str> = st.transaction(|s| {
            s.add_node(&["P"], &[]);
            Err("boom")
        });
        assert_eq!(r, Err("boom"));
        assert_eq!(st.node_count(), 1); // the aborted add was rolled back
    }
}
