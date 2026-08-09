//! A minimal typed columnar graph store — enough for the first execution slice
//! (scan a label, read a property, filter, project). Nodes only for now; edges,
//! adjacency, indexes, and temporal columns join in later slices.
//!
//! Properties are stored in TYPED columns (`Column`), not boxed values, so a
//! numeric property arrives at the batch layer as an unboxed `f64` run. That is
//! the whole point of the columnar model, present from the first slice rather
//! than retrofitted.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::value::Value;

/// A `Value` ordered by the value contract's total order (`cmp_total`) — the key
/// type for a range index's `BTreeMap`. It DELEGATES to `cmp_total`; it does not
/// restate ordering. `Eq` is "compares equal under `cmp_total`", so values the
/// total order ties (two NaNs, `-0.0`/`0.0`) share a bucket, exactly as grouping
/// does.
#[derive(Clone)]
struct OrdVal(Value);

impl PartialEq for OrdVal {
    fn eq(&self, other: &Self) -> bool {
        crate::value::cmp_total(&self.0, &other.0) == Ordering::Equal
    }
}
impl Eq for OrdVal {}
impl PartialOrd for OrdVal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdVal {
    fn cmp(&self, other: &Self) -> Ordering {
        crate::value::cmp_total(&self.0, &other.0)
    }
}

/// A range index on one node property `key`: values ordered by `cmp_total` ->
/// node ids. Holds only NON-NULL present values (a null never passes a range
/// predicate — the operand makes it UNKNOWN — so excluding it keeps the seek in
/// step with a scan+filter).
struct RangeIndex {
    key: String,
    map: BTreeMap<OrdVal, Vec<u32>>,
}

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
    /// A homogeneous temporal column: every present value is the SAME kind
    /// (`kind`). A temporal of a DIFFERENT kind — or a non-temporal — written to it
    /// promotes it to `Gen`, matching lenke-core's one-kind-per-column model.
    Temporal {
        kind: crate::temporal::TemporalKind,
        data: Vec<crate::temporal::Temporal>,
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
            Self::Temporal { data, present, .. } if present[idx] => Value::Temporal(data[idx]),
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
            | Self::Temporal { present, .. }
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
            // A temporal seeds a homogeneous typed column of its kind; absent slots
            // hold the value itself as a harmless placeholder (present-gated).
            Value::Temporal(t) => Self::Temporal {
                kind: t.kind(),
                data: vec![*t; n],
                present: vec![false; n],
            },
            // Records/maps (and null/list) have no typed column form yet — Gen.
            Value::Null | Value::List(_) | Value::Record(_) | Value::Map(_) => Self::Gen {
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
            Self::Temporal {
                kind,
                data,
                present,
            } => {
                data.push(kind.zero());
                present.push(false);
            }
            Self::Gen { data, present } => {
                data.push(Value::Null);
                present.push(false);
            }
        }
    }

    /// Whether this column can store `v` without a type change. A temporal column
    /// accepts only its OWN kind (a different kind promotes to `Gen`).
    fn accepts(&self, v: &Value) -> bool {
        match (self, v) {
            (Self::Num { .. }, Value::Num(_))
            | (Self::Str { .. }, Value::Str(_))
            | (Self::Bool { .. }, Value::Bool(_))
            | (Self::Gen { .. }, _) => true,
            (Self::Temporal { kind, .. }, Value::Temporal(t)) => t.kind() == *kind,
            _ => false,
        }
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
            | Self::Temporal { present, .. }
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
            (Self::Temporal { data, present, .. }, Value::Temporal(t)) => {
                data[i] = t;
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
            | Self::Temporal { present, .. }
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
            Self::Temporal { data, present, .. } => {
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
/// One change a committed transaction made — the unit of the observation-only CDC
/// stream. Recorded alongside the undo log (1:1 with each mutation) and handed to
/// observers AFTER commit, so it can never veto a write. Ids reference the store's
/// dense node ids / monotonic edge ids.
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    NodeAdded(u32),
    NodeDeleted(u32),
    NodeProp { node: u32, key: String },
    EdgeAdded(u32),
    EdgeDeleted(u32),
    EdgeProp { eid: u32, key: String },
}

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
    /// Undo `set_edge_prop`/`remove_edge_prop`: restore the edge cell (`None` =
    /// was absent).
    RestoreEdgeCell {
        eid: u32,
        key: String,
        prev: Option<Value>,
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
    /// the active transaction's change list (observation-only CDC), `Some` exactly
    /// when a transaction is open; moved to `last_commit` on commit, dropped on
    /// rollback. Grows 1:1 with the undo log.
    changes: Option<Vec<Change>>,
    /// the change list of the MOST RECENT committed transaction — what an observer
    /// reads after a write. Empty until the first commit.
    last_commit: Vec<Change>,
    /// declared unique constraints as `(label, keys)` — at most one live node per
    /// label may carry a given key tuple. Enforced by the write statements, not
    /// the store primitives (which stay infallible for rollback).
    unique: Vec<(String, Vec<String>)>,
    /// declared required-property constraints as `(label, key)` — every live node
    /// with `label` must carry a PRESENT value for `key` (present-null counts, per
    /// the null-first-class policy; only absence violates). Enforced by the write
    /// statements, like `unique`.
    required: Vec<(String, String)>,
    /// edge properties: key -> (eid -> value). Boxed (not columnar) — edges are a
    /// less hot path than node scans, and eids are sparse after deletes. A deleted
    /// edge's props are left behind (eids are never reused, so a dead eid is never
    /// read); reclaiming them is a later tidy.
    edge_props: HashMap<String, HashMap<u32, Value>>,
    /// hash indexes on a node property `key`: value's group-key bytes -> node ids
    /// (any label; the seek intersects with the label). Maintained on writes
    /// through the primitives, so a transaction rollback (which replays the
    /// primitives) keeps them consistent.
    indexes: Vec<Index>,
    /// range indexes on a node property `key` (ordered by `cmp_total`).
    ranges: Vec<RangeIndex>,
}

/// A hash index on a node property PATH. `path` is `["age"]` for a plain property
/// or `["meta", "city"]` for a dotted record-field path; the index keys on the
/// value found by descending record fields (`resolve_path`). A plain index
/// (length-1 path) behaves exactly as before.
struct Index {
    path: Vec<String>,
    map: HashMap<Vec<u8>, Vec<u32>>,
}

impl Index {
    /// The base (top-level) property whose column drives this index's upkeep.
    fn base(&self) -> &str {
        &self.path[0]
    }
}

/// Descend record fields `sub` from `v`; an empty `sub` returns `v` (a plain
/// index). A non-record along the way, or a missing field, resolves to `Null`.
fn resolve_path(v: &Value, sub: &[String]) -> Value {
    let mut cur = v.clone();
    for k in sub {
        cur = match &cur {
            Value::Record(fields) => crate::value::record_field(fields, k),
            _ => Value::Null,
        };
    }
    cur
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

    /// Read a node's value at a (possibly dotted) property PATH: the base property,
    /// then descend record sub-fields. A plain key reads exactly like [`prop`]. The
    /// scan-fallback twin of a dotted [`index_lookup`].
    #[must_use]
    pub fn prop_path(&self, node: u32, dotted_key: &str) -> Value {
        let path: Vec<String> = dotted_key.split('.').map(String::from).collect();
        resolve_path(&self.prop(node, &path[0]), &path[1..])
    }

    /// The typed column for `key`, for a bulk gather. `None` = no such property.
    #[must_use]
    pub fn column(&self, key: &str) -> Option<&Column> {
        self.props.get(key)
    }

    // --- Enumeration (for egress / snapshot) -----------------------------

    /// All node-property keys, sorted — a deterministic field order for dumps.
    #[must_use]
    pub fn prop_keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.props.keys().cloned().collect();
        k.sort();
        k
    }

    /// All edge-property keys, sorted.
    #[must_use]
    pub fn edge_prop_keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.edge_props.keys().cloned().collect();
        k.sort();
        k
    }

    /// The labels carried by node `id`, sorted.
    #[must_use]
    pub fn labels_of(&self, id: u32) -> Vec<String> {
        let mut ls: Vec<String> = self
            .by_label
            .iter()
            .filter(|(_, ids)| ids.contains(&id))
            .map(|(l, _)| l.clone())
            .collect();
        ls.sort();
        ls
    }

    /// The declared unique constraints as `(label, keys)` — for snapshot/schema
    /// egress.
    #[must_use]
    pub fn unique_constraints(&self) -> Vec<(String, Vec<String>)> {
        self.unique.clone()
    }

    /// The interned edge-type id's name (the reverse of `etype_id`).
    #[must_use]
    pub fn etype_name(&self, etype: u32) -> Option<String> {
        self.etype_ids
            .iter()
            .find(|(_, &id)| id == etype)
            .map(|(name, _)| name.clone())
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

    // --- Required-property constraints ------------------------------------
    //
    // A required constraint declares that every live node with `label` carries a
    // PRESENT value for `key` (present-null counts — only absence violates).
    // Enforced by the write statements after a mutation, like `unique`.

    /// The declared required constraints as `(label, key)` — for snapshot/schema.
    #[must_use]
    pub fn required_constraints(&self) -> Vec<(String, String)> {
        self.required.clone()
    }

    /// Declare a required-property constraint on `(label, key)`. Errors if the
    /// CURRENT data already violates it (a labelled node missing the property).
    pub fn create_required_constraint(&mut self, label: &str, key: &str) -> Result<(), String> {
        self.check_label_required(label, key)?;
        self.required.push((label.to_string(), key.to_string()));
        Ok(())
    }

    /// Check every required constraint on `label` against the live nodes; `Err`
    /// names the first violated one. Write statements call this after mutating a
    /// constrained label.
    pub fn check_required_for_label(&self, label: &str) -> Result<(), String> {
        for (l, key) in &self.required {
            if l == label {
                self.check_label_required(l, key)?;
            }
        }
        Ok(())
    }

    fn check_label_required(&self, label: &str, key: &str) -> Result<(), String> {
        for &id in self.nodes_with_label(label) {
            if !self.has_prop(id, key) {
                return Err(format!(
                    "E_REQUIRED: required constraint on {label}({key}) violated"
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
        self.record_change(Change::NodeAdded(id));
        id
    }

    /// Add a directed edge `from -[label]-> to`; returns its `eid` (shared by the
    /// out and in adjacency entries). Interns the edge type if new. The returned
    /// eid lets the caller attach edge properties to the new edge.
    pub fn add_edge(&mut self, from: u32, to: u32, label: &str) -> u32 {
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
        self.record_change(Change::EdgeAdded(eid));
        eid
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
        self.record_change(Change::NodeProp {
            node,
            key: key.to_string(),
        });
    }

    fn apply_set_prop(&mut self, node: u32, key: &str, value: Value) {
        // Index upkeep: capture the OLD base value before writing, and a copy of
        // the new one (reads first, then the column write, then the index writes —
        // distinct fields, no borrow clash). A hash index may be dotted, so it is
        // keyed by the BASE property `key`; the range index is single-key.
        let care = self.index_on_base(key) || self.is_range_indexed(key);
        let old = (care && self.has_prop(node, key)).then(|| self.prop(node, key));
        let new_for_index = care.then(|| value.clone());

        let n = self.node_count;
        let col = self
            .props
            .entry(key.to_string())
            .or_insert_with(|| Column::new_absent(&value, n));
        if !col.accepts(&value) {
            *col = col.to_gen();
        }
        col.set(node as usize, value);

        if let Some(nv) = new_for_index {
            self.reindex_node(key, node, old.as_ref(), Some(&nv));
            if self.is_range_indexed(key) {
                if let Some(old) = &old {
                    self.range_remove(key, old, node);
                }
                self.range_add(key, &nv, node);
            }
        }
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
        self.record_change(Change::NodeProp {
            node,
            key: key.to_string(),
        });
    }

    fn apply_remove_prop(&mut self, node: u32, key: &str) {
        // Drop the node from the index(es) for its OLD value, if indexed.
        let old = ((self.index_on_base(key) || self.is_range_indexed(key))
            && self.has_prop(node, key))
        .then(|| self.prop(node, key));
        if let Some(col) = self.props.get_mut(key) {
            col.set_absent(node as usize);
        }
        if let Some(old) = &old {
            self.reindex_node(key, node, Some(old), None);
            if self.is_range_indexed(key) {
                self.range_remove(key, old, node);
            }
        }
    }

    // --- Edge properties -------------------------------------------------
    //
    // Keyed by the edge's `eid` (shared by its out/in adjacency entries), so a
    // property is one value per edge regardless of direction. Undo-logged like
    // node properties.

    /// Read edge `eid`'s `key` (NULL if absent).
    #[must_use]
    pub fn edge_prop(&self, eid: u32, key: &str) -> Value {
        self.edge_props
            .get(key)
            .and_then(|m| m.get(&eid))
            .cloned()
            .unwrap_or(Value::Null)
    }

    /// Whether edge `eid` carries a present value for `key`.
    #[must_use]
    pub fn has_edge_prop(&self, eid: u32, key: &str) -> bool {
        self.edge_props
            .get(key)
            .is_some_and(|m| m.contains_key(&eid))
    }

    /// Set edge `eid`'s `key` to `value`.
    pub fn set_edge_prop(&mut self, eid: u32, key: &str, value: Value) {
        let rec = self.undo.is_some().then(|| Undo::RestoreEdgeCell {
            eid,
            key: key.to_string(),
            prev: self.edge_props.get(key).and_then(|m| m.get(&eid)).cloned(),
        });
        self.edge_props
            .entry(key.to_string())
            .or_default()
            .insert(eid, value);
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
        self.record_change(Change::EdgeProp {
            eid,
            key: key.to_string(),
        });
    }

    // --- Property indexes ------------------------------------------------
    //
    // A hash index on a node property `key`, mapping a value's grouping bytes to
    // the node ids carrying it. Built from current data on create and maintained
    // by the mutation primitives (so rollback, which replays them, stays
    // consistent). Index equality is grouping (group_key) — the seek layer maps
    // that to predicate `=` (NaN/null match nothing) so results match a scan.

    /// Create a hash index on a node property PATH (idempotent). `key` is a plain
    /// property (`age`) or a dotted record-field path (`meta.city`). Builds from
    /// the current live nodes that carry the base property, keying on the resolved
    /// (descended) value.
    pub fn create_index(&mut self, key: &str) {
        let path: Vec<String> = key.split('.').map(String::from).collect();
        if self.indexes.iter().any(|i| i.path == path) {
            return;
        }
        let sub = &path[1..];
        let mut map: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
        if let Some(col) = self.props.get(&path[0]) {
            for id in 0..self.node_count {
                if !self.deleted[id] && col.present_at(id) {
                    let v = resolve_path(&col.read(id), sub);
                    map.entry(crate::value::group_key(&v))
                        .or_default()
                        .push(id as u32);
                }
            }
        }
        self.indexes.push(Index { path, map });
    }

    /// Whether any hash index is driven by the base property `base` (so a write to
    /// it must maintain that index).
    #[must_use]
    fn index_on_base(&self, base: &str) -> bool {
        self.indexes.iter().any(|i| i.base() == base)
    }

    /// Candidate node ids (ANY label) whose property PATH `key` groups equal to
    /// `value`, or `None` if no index exists on that path. Deleted ids filtered.
    #[must_use]
    pub fn index_lookup(&self, key: &str, value: &Value) -> Option<Vec<u32>> {
        let path: Vec<String> = key.split('.').map(String::from).collect();
        let idx = self.indexes.iter().find(|i| i.path == path)?;
        let gk = crate::value::group_key(value);
        Some(
            idx.map
                .get(&gk)
                .map(|ids| {
                    ids.iter()
                        .copied()
                        .filter(|&id| !self.deleted[id as usize])
                        .collect()
                })
                .unwrap_or_default(),
        )
    }

    /// Update every hash index whose BASE is `base` for a node whose base value
    /// changed from `old` to `new` (`None` = absent on that side).
    fn reindex_node(&mut self, base: &str, node: u32, old: Option<&Value>, new: Option<&Value>) {
        let matching: Vec<usize> = self
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, i)| i.base() == base)
            .map(|(j, _)| j)
            .collect();
        for j in matching {
            let sub = self.indexes[j].path[1..].to_vec();
            if let Some(old) = old {
                let og = crate::value::group_key(&resolve_path(old, &sub));
                if let Some(bucket) = self.indexes[j].map.get_mut(&og) {
                    bucket.retain(|&x| x != node);
                }
            }
            if let Some(new) = new {
                let ng = crate::value::group_key(&resolve_path(new, &sub));
                self.indexes[j].map.entry(ng).or_default().push(node);
            }
        }
    }

    /// Remove node `id`'s entries from every index (used by delete/pop).
    fn index_drop_node(&mut self, id: u32) {
        // For each hash index whose base the node carries, the group key of the
        // node's resolved (descended) value — then drop it from that index.
        let removals: Vec<(usize, Vec<u8>)> = self
            .indexes
            .iter()
            .enumerate()
            .filter(|(_, ix)| self.has_prop(id, ix.base()))
            .map(|(j, ix)| {
                let v = resolve_path(&self.prop(id, ix.base()), &ix.path[1..]);
                (j, crate::value::group_key(&v))
            })
            .collect();
        for (j, gk) in removals {
            if let Some(bucket) = self.indexes[j].map.get_mut(&gk) {
                bucket.retain(|&x| x != id);
            }
        }
        // Range indexes hold non-null values only.
        let range_removals: Vec<(String, Value)> = self
            .ranges
            .iter()
            .filter(|ix| self.has_prop(id, &ix.key))
            .filter_map(|ix| {
                let v = self.prop(id, &ix.key);
                (!v.is_null()).then(|| (ix.key.clone(), v))
            })
            .collect();
        for (k, v) in range_removals {
            self.range_remove(&k, &v, id);
        }
    }

    /// Create a range index on node property `key` (idempotent). Built from the
    /// current live nodes carrying a NON-NULL value for it.
    pub fn create_range_index(&mut self, key: &str) {
        if self.ranges.iter().any(|i| i.key == key) {
            return;
        }
        let mut map: BTreeMap<OrdVal, Vec<u32>> = BTreeMap::new();
        if let Some(col) = self.props.get(key) {
            for id in 0..self.node_count {
                if !self.deleted[id] && col.present_at(id) {
                    let v = col.read(id);
                    if !v.is_null() {
                        map.entry(OrdVal(v)).or_default().push(id as u32);
                    }
                }
            }
        }
        self.ranges.push(RangeIndex {
            key: key.to_string(),
            map,
        });
    }

    #[must_use]
    fn is_range_indexed(&self, key: &str) -> bool {
        self.ranges.iter().any(|i| i.key == key)
    }

    fn range_add(&mut self, key: &str, value: &Value, node: u32) {
        if value.is_null() {
            return;
        }
        if let Some(ix) = self.ranges.iter_mut().find(|i| i.key == key) {
            ix.map.entry(OrdVal(value.clone())).or_default().push(node);
        }
    }

    fn range_remove(&mut self, key: &str, value: &Value, node: u32) {
        if value.is_null() {
            return;
        }
        if let Some(ix) = self.ranges.iter_mut().find(|i| i.key == key) {
            if let Some(bucket) = ix.map.get_mut(&OrdVal(value.clone())) {
                bucket.retain(|&x| x != node);
            }
        }
    }

    /// Candidate node ids (ANY label) whose `key` satisfies `prop <op> value`
    /// under `cmp_total`, or `None` if no range index exists on `key`. `op` is one
    /// of `Lt`/`Le`/`Gt`/`Ge`. A null `value` matches nothing. Deleted filtered.
    #[must_use]
    pub fn range_lookup(
        &self,
        key: &str,
        op: crate::ir::CompareOp,
        value: &Value,
    ) -> Option<Vec<u32>> {
        use crate::ir::CompareOp::{Ge, Gt, Le, Lt};
        use std::ops::Bound::{Excluded, Included, Unbounded};
        let ix = self.ranges.iter().find(|i| i.key == key)?;
        if value.is_null() {
            return Some(Vec::new());
        }
        let k = OrdVal(value.clone());
        let bounds: (std::ops::Bound<OrdVal>, std::ops::Bound<OrdVal>) = match op {
            Gt => (Excluded(k), Unbounded),
            Ge => (Included(k), Unbounded),
            Lt => (Unbounded, Excluded(k)),
            Le => (Unbounded, Included(k)),
            _ => return Some(Vec::new()), // not a range op
        };
        Some(
            ix.map
                .range(bounds)
                .flat_map(|(_, ids)| ids.iter().copied())
                .filter(|&id| !self.deleted[id as usize])
                .collect(),
        )
    }

    /// Remove edge `eid`'s `key` (reads NULL again).
    pub fn remove_edge_prop(&mut self, eid: u32, key: &str) {
        let rec = self.undo.is_some().then(|| Undo::RestoreEdgeCell {
            eid,
            key: key.to_string(),
            prev: self.edge_props.get(key).and_then(|m| m.get(&eid)).cloned(),
        });
        if let Some(m) = self.edge_props.get_mut(key) {
            m.remove(&eid);
        }
        if let (Some(rec), Some(log)) = (rec, self.undo.as_mut()) {
            log.push(rec);
        }
        self.record_change(Change::EdgeProp {
            eid,
            key: key.to_string(),
        });
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
        self.record_change(Change::EdgeDeleted(eid));
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
        // Drop the node from any property indexes (reads its props, still present).
        self.index_drop_node(id);
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
        // A node deletion cascades its incident edges; the CDC stream reports it as
        // one NodeDeleted (the edges are implied), keeping 1:1 with the undo.
        self.record_change(Change::NodeDeleted(id));
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
        self.changes = Some(Vec::new());
    }

    /// Commit: the changes stand, the undo log is discarded, and the transaction's
    /// change list becomes the observable `last_commit` (CDC).
    pub fn commit(&mut self) {
        self.undo = None;
        self.last_commit = self.changes.take().unwrap_or_default();
    }

    /// Roll back every change since `begin`, in reverse, and close the
    /// transaction. A no-op outside a transaction. The change list is dropped (a
    /// rolled-back transaction is observed to have changed nothing).
    pub fn rollback(&mut self) {
        self.changes = None;
        if let Some(log) = self.undo.take() {
            // `undo` is now None, so the inverse mutations below do not re-log.
            for rec in log.into_iter().rev() {
                self.apply_undo(rec);
            }
        }
    }

    /// The change list of the most recent committed transaction — the
    /// observation-only CDC stream. Read after a write; cannot veto it.
    #[must_use]
    pub fn last_commit_changes(&self) -> &[Change] {
        &self.last_commit
    }

    /// Record a change into the active transaction's list (no-op outside a txn).
    /// Grows 1:1 with the undo log, so `rollback_to` can truncate both by length.
    fn record_change(&mut self, c: Change) {
        if let Some(ch) = &mut self.changes {
            ch.push(c);
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
        // The change list grows 1:1 with the undo log, so it truncates to the same
        // mark — the undone statement's changes vanish from the CDC stream too.
        if let Some(ch) = &mut self.changes {
            ch.truncate(mark);
        }
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
            Undo::RestoreEdgeCell { eid, key, prev } => match prev {
                Some(v) => {
                    self.edge_props.entry(key).or_default().insert(eid, v);
                }
                None => {
                    if let Some(m) = self.edge_props.get_mut(&key) {
                        m.remove(&eid);
                    }
                }
            },
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
        // Drop it from any indexes while its props still exist.
        self.index_drop_node(id);
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
            changes: None,
            last_commit: Vec::new(),
            unique: Vec::new(),
            required: Vec::new(),
            edge_props: HashMap::new(),
            indexes: Vec::new(),
            ranges: Vec::new(),
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
    } else if let Some(kind) = homogeneous_temporal_kind(&pairs) {
        let mut data = vec![kind.zero(); n];
        let mut present = vec![false; n];
        for (i, v) in pairs {
            if let Value::Temporal(t) = v {
                data[i as usize] = t;
                present[i as usize] = true;
            }
        }
        Column::Temporal {
            kind,
            data,
            present,
        }
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

/// The single temporal kind shared by every pair, or `None` if any pair is
/// non-temporal or the kinds are mixed (→ a `Gen` column).
fn homogeneous_temporal_kind(pairs: &[(u32, Value)]) -> Option<crate::temporal::TemporalKind> {
    let mut kind = None;
    for (_, v) in pairs {
        let Value::Temporal(t) = v else {
            return None;
        };
        match kind {
            None => kind = Some(t.kind()),
            Some(k) if k == t.kind() => {}
            Some(_) => return None, // mixed kinds fall back to Gen
        }
    }
    kind
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

    #[test]
    fn temporal_props_use_a_typed_column_and_promote_on_mixed_kind() {
        use crate::temporal::{Date, Temporal, TemporalKind, Time};
        let d = |iso: &str| Value::Temporal(Temporal::Date(Date::parse(iso).unwrap()));
        let mut b = Builder::default();
        b.node(&["P"], &[("born", d("1990-01-01"))]);
        b.node(&["P"], &[("born", d("2000-01-01"))]);
        let mut st = b.build();
        // Homogeneous Date props de-box into a typed Temporal column of kind Date.
        assert!(matches!(
            st.column("born"),
            Some(Column::Temporal {
                kind: TemporalKind::Date,
                ..
            })
        ));
        match st.prop(0, "born") {
            Value::Temporal(Temporal::Date(x)) => assert_eq!(x.format(), "1990-01-01"),
            o => panic!("expected a Date, got {o:?}"),
        }
        // Writing a DIFFERENT temporal kind promotes the column to Gen; both
        // values still read back correctly.
        st.set_prop(
            1,
            "born",
            Value::Temporal(Temporal::Time(Time::parse("12:00:00").unwrap())),
        );
        assert!(matches!(st.column("born"), Some(Column::Gen { .. })));
        assert!(matches!(
            st.prop(0, "born"),
            Value::Temporal(Temporal::Date(_))
        ));
        assert!(matches!(
            st.prop(1, "born"),
            Value::Temporal(Temporal::Time(_))
        ));
    }

    #[test]
    fn commit_records_the_change_list_rollback_records_nothing() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]); // outside a txn → nothing observed
        assert!(st.last_commit_changes().is_empty());

        // A committed transaction publishes exactly its changes, in order.
        st.begin();
        let b = st.add_node(&["P"], &[("name", s("b"))]);
        st.set_prop(a, "age", n(1.0));
        let eid = st.add_edge(a, b, "R");
        st.commit();
        assert_eq!(
            st.last_commit_changes(),
            &[
                Change::NodeAdded(b),
                Change::NodeProp {
                    node: a,
                    key: "age".into(),
                },
                Change::EdgeAdded(eid),
            ]
        );

        // A rolled-back transaction publishes nothing: `last_commit` still shows
        // the previous COMMIT, unchanged (rollback is not an event).
        let previous: Vec<Change> = st.last_commit_changes().to_vec();
        st.begin();
        st.set_prop(a, "age", n(2.0));
        st.rollback();
        assert_eq!(st.last_commit_changes(), previous.as_slice());
    }

    #[test]
    fn cdc_reports_delete_as_one_node_deleted() {
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[]);
        let b = st.add_node(&["P"], &[]);
        st.add_edge(a, b, "R");
        st.begin();
        st.delete_node(a); // cascades the edge — reported as one NodeDeleted
        st.commit();
        assert_eq!(st.last_commit_changes(), &[Change::NodeDeleted(a)]);
    }

    #[test]
    fn required_constraint_declared_and_checked() {
        let mut st = Builder::default().build();
        st.add_node(&["User"], &[("email", s("a@x"))]);
        // Every User has email → the constraint declares, and the check passes.
        assert!(st.create_required_constraint("User", "email").is_ok());
        assert!(st.check_required_for_label("User").is_ok());
        // A User missing email → the check fails (present-null would pass; absence
        // is the violation).
        st.add_node(&["User"], &[("name", s("b"))]);
        assert!(st.check_required_for_label("User").is_err());
        // Declaring on already-violating data errors.
        let mut st2 = Builder::default().build();
        st2.add_node(&["User"], &[("name", s("x"))]);
        assert!(st2.create_required_constraint("User", "email").is_err());
    }

    #[test]
    fn dotted_path_index_maintained_through_mutations() {
        use crate::value::make_record;
        let city = |c: &str| make_record(vec![(Arc::from("city"), s(c))]);
        let mut st = Builder::default().build();
        let a = st.add_node(&["P"], &[("meta", city("NYC"))]);
        let b = st.add_node(&["P"], &[("meta", city("LA"))]);
        let c = st.add_node(&["P"], &[("meta", city("NYC"))]);
        // Built from existing data: index on the record sub-field `meta.city`.
        st.create_index("meta.city");
        let nyc = |st: &Store| {
            let mut v = st.index_lookup("meta.city", &s("NYC")).unwrap();
            v.sort_unstable();
            v
        };
        assert_eq!(nyc(&st), vec![a, c]);
        assert_eq!(st.index_lookup("meta.city", &s("LA")).unwrap(), vec![b]);

        // Maintained on a write: change b's city to NYC.
        st.set_prop(b, "meta", city("NYC"));
        assert_eq!(nyc(&st), vec![a, b, c]);
        // …and on delete.
        st.delete_node(a);
        assert_eq!(nyc(&st), vec![b, c]);
        // No index on this path → None (distinct from an empty match).
        assert!(st.index_lookup("meta.zip", &n(1.0)).is_none());
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

    // --- Edge properties ---

    /// Set / read / remove an edge property, keyed by the edge's eid.
    #[test]
    fn edge_property_set_read_remove() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        assert!(st.edge_prop(eid, "weight").is_null()); // absent
        st.set_edge_prop(eid, "weight", n(0.5));
        assert!(st.has_edge_prop(eid, "weight"));
        assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 0.5));
        st.remove_edge_prop(eid, "weight");
        assert!(!st.has_edge_prop(eid, "weight"));
        assert!(st.edge_prop(eid, "weight").is_null());
    }

    /// An edge property write rolls back with the transaction.
    #[test]
    fn edge_property_rolls_back() {
        let mut st = Builder::default().build();
        let a = st.add_node(&[], &[]);
        let b = st.add_node(&[], &[]);
        st.add_edge(a, b, "R");
        let eid = st.out(a)[0].eid;
        st.set_edge_prop(eid, "weight", n(1.0)); // committed (autocommit)
        st.begin();
        st.set_edge_prop(eid, "weight", n(2.0));
        st.set_edge_prop(eid, "fresh", s("x"));
        st.rollback();
        assert!(matches!(st.edge_prop(eid, "weight"), Value::Num(x) if x == 1.0)); // restored
        assert!(!st.has_edge_prop(eid, "fresh")); // new key gone
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
