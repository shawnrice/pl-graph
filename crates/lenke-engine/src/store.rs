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

/// One indexed interval: an out-edge's `[lo, hi]` (read from two numeric edge
/// props at build time), plus the edge id and its neighbour. Copied inline so an
/// overlap seek never touches the boxed edge-property map — the whole point of the
/// index (the boxed post-filter is what `examples/interval_bench` measures as the
/// cost).
#[derive(Clone, Copy)]
struct Iv {
    lo: f64,
    hi: f64,
    eid: u32,
    nbr: u32,
}

/// The opt-in edge INTERVAL index for one `(lo_key, hi_key)` pair over OUT-edges.
/// Per source node its intervals are held BOTH sorted by `lo` ascending and by
/// `hi` ascending, so an overlap query `[qlo, qhi]` (an edge overlaps iff
/// `lo <= qhi AND hi >= qlo`) can SEED from whichever axis is more selective and
/// post-filter the other — the RI-tree-lite rule from the bitemporal index (seed
/// from the selective axis; never materialize and intersect both stabs).
struct IntervalIndex {
    lo_key: String,
    hi_key: String,
    by_lo: Vec<Vec<Iv>>,
    by_hi: Vec<Vec<Iv>>,
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
    /// OPT-IN edge-type index: `edge_type_index` is false and the vectors empty
    /// unless [`create_edge_type_index`](Store::create_edge_type_index) was called
    /// (so a graph that does not need it pays nothing — see `examples/expand_bench`
    /// for why the win is confined to high-degree, many-type nodes). When on,
    /// `out_type_idx[node]`/`in_type_idx[node]` map an edge-type id to that node's
    /// adjacency of that type, so a type-filtered hop seeks the bucket instead of
    /// scanning the whole adjacency. Kept correct across writes/deletes/rollback by
    /// a per-node rebuild whenever a node's flat adjacency changes (the flat lists
    /// stay the source of truth), with an O(1) push on the `add_edge` hot path.
    edge_type_index: bool,
    out_type_idx: Vec<HashMap<u32, Vec<Adj>>>,
    in_type_idx: Vec<HashMap<u32, Vec<Adj>>>,
    /// OPT-IN edge interval index (`None` unless `create_interval_index` was
    /// called). Built from edge props at creation and maintained through the
    /// mutation primitives; an interval-key edge-prop change triggers a full
    /// rebuild (rare — intervals are typically bulk-loaded before the index).
    interval: Option<IntervalIndex>,
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
        if self.edge_type_index {
            self.out_type_idx.push(HashMap::new());
            self.in_type_idx.push(HashMap::new());
        }
        if let Some(ix) = &mut self.interval {
            ix.by_lo.push(Vec::new());
            ix.by_hi.push(Vec::new());
        }
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
        if self.edge_type_index {
            // Hot path: a single append to each endpoint's type bucket — O(1), not
            // a per-node rebuild.
            self.out_type_idx[from as usize]
                .entry(etype)
                .or_default()
                .push(Adj {
                    nbr: to,
                    etype,
                    eid,
                });
            self.in_type_idx[to as usize]
                .entry(etype)
                .or_default()
                .push(Adj {
                    nbr: from,
                    etype,
                    eid,
                });
        }
        if self.interval.is_some() {
            // The new edge carries no interval props yet (they arrive via
            // set_edge_prop, which rebuilds); reindexing now keeps `from`'s buckets
            // consistent and picks the edge up once its props are set.
            self.reindex_node_interval(from);
        }
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

    /// Create the opt-in edge-type index and build it from the current adjacency.
    /// Idempotent; after this a type-filtered hop seeks a per-node type bucket
    /// rather than scanning the whole adjacency (see the field docs and
    /// `examples/expand_bench`). Subsequent writes maintain it.
    pub fn create_edge_type_index(&mut self) {
        self.edge_type_index = true;
        self.out_type_idx = vec![HashMap::new(); self.node_count];
        self.in_type_idx = vec![HashMap::new(); self.node_count];
        for node in 0..self.node_count as u32 {
            self.reindex_node_etypes(node);
        }
    }

    /// Whether the opt-in edge-type index is active.
    #[must_use]
    pub fn has_edge_type_index(&self) -> bool {
        self.edge_type_index
    }

    /// Node `node`'s outgoing adjacency of edge-type `etype` (empty if none, or if
    /// the index is off — callers gate on [`has_edge_type_index`](Store::has_edge_type_index)).
    #[must_use]
    pub fn out_typed(&self, node: u32, etype: u32) -> &[Adj] {
        self.out_type_idx
            .get(node as usize)
            .and_then(|m| m.get(&etype))
            .map_or(&[], Vec::as_slice)
    }

    /// Node `node`'s incoming adjacency of edge-type `etype`.
    #[must_use]
    pub fn in_typed(&self, node: u32, etype: u32) -> &[Adj] {
        self.in_type_idx
            .get(node as usize)
            .and_then(|m| m.get(&etype))
            .map_or(&[], Vec::as_slice)
    }

    /// Rebuild one node's type buckets from its (authoritative) flat adjacency. A
    /// no-op when the index is off. Called after any adjacency change other than
    /// the `add_edge` hot path, so the index needs no per-edge delta bookkeeping.
    fn reindex_node_etypes(&mut self, node: u32) {
        if !self.edge_type_index {
            return;
        }
        let i = node as usize;
        let mut om: HashMap<u32, Vec<Adj>> = HashMap::new();
        for a in &self.out_adj[i] {
            om.entry(a.etype).or_default().push(*a);
        }
        let mut im: HashMap<u32, Vec<Adj>> = HashMap::new();
        for a in &self.in_adj[i] {
            im.entry(a.etype).or_default().push(*a);
        }
        self.out_type_idx[i] = om;
        self.in_type_idx[i] = im;
    }

    // --- opt-in edge interval index (G4) ---

    /// Create the opt-in interval index over OUT-edges for the numeric edge props
    /// `(lo_key, hi_key)` and build it from the current edges. Replaces any prior
    /// interval index. After this, [`for_each_overlap`](Store::for_each_overlap)
    /// seeks a node's edges whose `[lo, hi]` overlaps a query interval instead of
    /// scanning the adjacency and reading the boxed props.
    pub fn create_interval_index(&mut self, lo_key: &str, hi_key: &str) {
        self.interval = Some(IntervalIndex {
            lo_key: lo_key.to_string(),
            hi_key: hi_key.to_string(),
            by_lo: vec![Vec::new(); self.node_count],
            by_hi: vec![Vec::new(); self.node_count],
        });
        for node in 0..self.node_count as u32 {
            self.reindex_node_interval(node);
        }
    }

    /// Whether an interval index on exactly `(lo_key, hi_key)` is active.
    #[must_use]
    pub fn has_interval_index(&self, lo_key: &str, hi_key: &str) -> bool {
        self.interval
            .as_ref()
            .is_some_and(|ix| ix.lo_key == lo_key && ix.hi_key == hi_key)
    }

    /// Whether `key` is one of the active interval index's axes (so a change to it
    /// invalidates the index).
    fn interval_uses_key(&self, key: &str) -> bool {
        self.interval
            .as_ref()
            .is_some_and(|ix| ix.lo_key == key || ix.hi_key == key)
    }

    /// Call `f(eid, nbr)` for each OUT-edge of `node` whose interval `[lo, hi]`
    /// overlaps `[qlo, qhi]` (i.e. `lo <= qhi && hi >= qlo`), seeking via the
    /// interval index. Seeds from whichever axis is the more selective (fewer
    /// candidates) and post-filters the other — never intersecting both. A no-op if
    /// no interval index is active or `node` is out of range.
    pub fn for_each_overlap(&self, node: u32, qlo: f64, qhi: f64, mut f: impl FnMut(u32, u32)) {
        let Some(ix) = &self.interval else { return };
        let Some(by_lo) = ix.by_lo.get(node as usize) else {
            return;
        };
        let by_hi = &ix.by_hi[node as usize];
        // # with lo <= qhi (a prefix of by_lo); # with hi >= qlo (a suffix of by_hi).
        let n_lo = by_lo.partition_point(|iv| iv.lo <= qhi);
        let n_hi = by_hi.len() - by_hi.partition_point(|iv| iv.hi < qlo);
        if n_lo <= n_hi {
            for iv in &by_lo[..n_lo] {
                if iv.hi >= qlo {
                    f(iv.eid, iv.nbr);
                }
            }
        } else {
            for iv in &by_hi[by_hi.len() - n_hi..] {
                if iv.lo <= qhi {
                    f(iv.eid, iv.nbr);
                }
            }
        }
    }

    /// Rebuild one source node's interval buckets from its current out-edges and
    /// their (boxed) props. An edge missing either numeric interval prop is skipped
    /// (it cannot be range-sought). No-op when the index is off.
    fn reindex_node_interval(&mut self, node: u32) {
        if self.interval.is_none() {
            return;
        }
        let (lo_key, hi_key) = {
            let ix = self.interval.as_ref().unwrap();
            (ix.lo_key.clone(), ix.hi_key.clone())
        };
        let i = node as usize;
        let mut ivs: Vec<Iv> = Vec::new();
        for a in &self.out_adj[i] {
            if let (Value::Num(lo), Value::Num(hi)) = (
                self.edge_prop(a.eid, &lo_key),
                self.edge_prop(a.eid, &hi_key),
            ) {
                ivs.push(Iv {
                    lo,
                    hi,
                    eid: a.eid,
                    nbr: a.nbr,
                });
            }
        }
        let mut by_lo = ivs.clone();
        by_lo.sort_by(|a, b| a.lo.total_cmp(&b.lo));
        let mut by_hi = ivs;
        by_hi.sort_by(|a, b| a.hi.total_cmp(&b.hi));
        let ix = self.interval.as_mut().unwrap();
        ix.by_lo[i] = by_lo;
        ix.by_hi[i] = by_hi;
    }

    /// Rebuild the whole interval index (used after an interval-key edge-prop change,
    /// where the affected source node is not cheaply known from the eid).
    fn rebuild_interval(&mut self) {
        if self.interval.is_none() {
            return;
        }
        // Resize per-node vectors in case node_count changed, then reindex all.
        {
            let n = self.node_count;
            let ix = self.interval.as_mut().unwrap();
            ix.by_lo = vec![Vec::new(); n];
            ix.by_hi = vec![Vec::new(); n];
        }
        for node in 0..self.node_count as u32 {
            self.reindex_node_interval(node);
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
        // An interval-axis change moves an edge's interval; the source node isn't
        // cheaply known from the eid, so rebuild the (opt-in, rarely-mutated) index.
        if self.interval_uses_key(key) {
            self.rebuild_interval();
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
        if self.interval_uses_key(key) {
            self.rebuild_interval();
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
        if self.edge_type_index {
            self.reindex_node_etypes(u);
            self.reindex_node_etypes(v);
        }
        if self.interval.is_some() {
            // The interval index is on OUT-edges; the deleted eid was an out-edge
            // of whichever endpoint is its source, so reindex both to be safe.
            self.reindex_node_interval(u);
            self.reindex_node_interval(v);
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

        if self.edge_type_index {
            // Node i's buckets go empty; each neighbour lost this node's mirror.
            self.reindex_node_etypes(id);
            for a in out.iter().chain(inc.iter()) {
                if a.nbr != id {
                    self.reindex_node_etypes(a.nbr);
                }
            }
        }
        if self.interval.is_some() {
            // Interval index is on OUT-edges: id's own out-edges are gone, and each
            // IN-neighbour (an edge `nbr -> id`) lost one of ITS out-edges.
            self.reindex_node_interval(id);
            for a in &inc {
                if a.nbr != id {
                    self.reindex_node_interval(a.nbr);
                }
            }
        }

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
        // The interval index depends on edge PROPS as well as adjacency, both of
        // which are restored by the undos above in an order that per-record index
        // maintenance can't safely track — so rebuild it once against the fully
        // restored graph. (The edge-type index depends only on adjacency, which
        // each undo record reindexes as it restores it.)
        if self.interval.is_some() {
            self.rebuild_interval();
        }
    }

    /// The change list of the most recent committed transaction — the
    /// observation-only CDC stream. Read after a write; cannot veto it.
    #[must_use]
    pub fn last_commit_changes(&self) -> &[Change] {
        &self.last_commit
    }

    /// The DISTINCT content-derived scopes the last commit touched, plus a
    /// fail-open flag. The scope of a NODE change is its `scope_key` property (the
    /// host assigns what that means — e.g. `"room"`/`"tenant"`); a change with no
    /// derivable scope (an edge change, or a deleted/absent node) sets the flag,
    /// meaning "relevant to ALL clients." A subscriber to scope `S` treats the
    /// commit as relevant iff `open || scopes contains S`. This is an OPTIMIZATION,
    /// not a security boundary (fail-open): the host owns the scope-key authority
    /// (the engine derives, it does not mint one). Scopes are `cmp_total`-sorted.
    #[must_use]
    pub fn touched_scopes(&self, scope_key: &str) -> (Vec<Value>, bool) {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut scopes: Vec<Value> = Vec::new();
        let mut open = false;
        for ch in &self.last_commit {
            let node = match ch {
                Change::NodeAdded(n)
                | Change::NodeProp { node: n, .. }
                | Change::NodeDeleted(n) => Some(*n),
                Change::EdgeAdded(_) | Change::EdgeDeleted(_) | Change::EdgeProp { .. } => None,
            };
            match node {
                Some(n) => {
                    let v = self.prop(n, scope_key);
                    if v.is_null() {
                        open = true; // absent/deleted scope → visible to all
                    } else if seen.insert(crate::value::group_key(&v)) {
                        scopes.push(v);
                    }
                }
                None => open = true, // an edge change has no node scope → fail-open
            }
        }
        scopes.sort_by(crate::value::cmp_total);
        (scopes, open)
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
        // Rebuild the interval index against the restored graph (see `rollback`).
        if self.interval.is_some() {
            self.rebuild_interval();
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
                let touched: Vec<u32> = if self.edge_type_index {
                    entries.iter().map(|(node, _, _)| *node).collect()
                } else {
                    Vec::new()
                };
                for (node, is_out, adj) in entries {
                    if is_out {
                        self.out_adj[node as usize].push(adj);
                    } else {
                        self.in_adj[node as usize].push(adj);
                    }
                }
                for node in touched {
                    self.reindex_node_etypes(node);
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
                if self.edge_type_index {
                    // Rebuild id's buckets, and each neighbour that regained a
                    // mirror (read the restored adjacency for the neighbour set).
                    self.reindex_node_etypes(id);
                    let nbrs: Vec<u32> = self.out_adj[i]
                        .iter()
                        .chain(self.in_adj[i].iter())
                        .map(|a| a.nbr)
                        .filter(|&nb| nb != id)
                        .collect();
                    for nb in nbrs {
                        self.reindex_node_etypes(nb);
                    }
                }
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
        if self.edge_type_index {
            self.out_type_idx.pop();
            self.in_type_idx.pop();
        }
        if let Some(ix) = &mut self.interval {
            ix.by_lo.pop();
            ix.by_hi.pop();
        }
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
            // The edge-type index is opt-in; bulk build never turns it on (the
            // caller runs `create_edge_type_index` after load if it wants it).
            edge_type_index: false,
            out_type_idx: Vec::new(),
            in_type_idx: Vec::new(),
            interval: None,
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
    fn touched_scopes_are_the_distinct_rooms_a_commit_writes() {
        let str_scopes = |scopes: &[Value]| -> Vec<String> {
            scopes
                .iter()
                .map(|v| match v {
                    Value::Str(x) => x.to_string(),
                    o => format!("{o:?}"),
                })
                .collect()
        };
        let mut st = Builder::default().build();
        st.begin();
        st.add_node(&["Msg"], &[("room", s("A"))]);
        st.add_node(&["Msg"], &[("room", s("B"))]);
        st.add_node(&["Msg"], &[("room", s("A"))]); // duplicate room A
        st.commit();
        let (scopes, open) = st.touched_scopes("room");
        assert_eq!(str_scopes(&scopes), vec!["A", "B"]); // distinct, sorted
        assert!(!open); // every change was scopable

        // A node with no `room` property → fail-open (visible to all).
        st.begin();
        st.add_node(&["Sys"], &[]);
        st.commit();
        let (scopes2, open2) = st.touched_scopes("room");
        assert!(scopes2.is_empty());
        assert!(open2);
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

    // --- opt-in edge-type index (G5) ---

    /// A small multi-type graph: node 0 knows 1 and likes 2; node 1 knows 2.
    fn typed_graph() -> Store {
        let mut b = Builder::default();
        for _ in 0..3 {
            b.node(&["V"], &[]);
        }
        b.edge(0, 1, "KNOWS");
        b.edge(0, 2, "LIKES");
        b.edge(1, 2, "KNOWS");
        b.build()
    }

    /// The typed neighbours of `node` along `etype` (out), as a sorted id list.
    fn out_ids(st: &Store, node: u32, ty: &str) -> Vec<u32> {
        let et = st.etype_id(ty).unwrap();
        let mut v: Vec<u32> = st.out_typed(node, et).iter().map(|a| a.nbr).collect();
        v.sort_unstable();
        v
    }

    /// The index, once built, agrees with a manual type-filter of the flat
    /// adjacency for every node and type.
    #[test]
    fn edge_type_index_matches_flat_scan() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        assert!(st.has_edge_type_index());
        for node in 0..st.node_count() as u32 {
            for ty in ["KNOWS", "LIKES"] {
                let et = st.etype_id(ty).unwrap();
                let mut scan: Vec<u32> = st
                    .out(node)
                    .iter()
                    .filter(|a| a.etype == et)
                    .map(|a| a.nbr)
                    .collect();
                scan.sort_unstable();
                assert_eq!(out_ids(&st, node, ty), scan, "node {node} type {ty}");
            }
        }
        // node 0: KNOWS -> {1}, LIKES -> {2}
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1]);
        assert_eq!(out_ids(&st, 0, "LIKES"), vec![2]);
    }

    /// add_edge keeps the index current (the O(1) hot path).
    #[test]
    fn edge_type_index_tracks_add() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        st.add_edge(0, 2, "KNOWS");
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1, 2]); // 0 now KNOWS 1 and 2
    }

    /// delete_edge and delete_node keep the index current, including neighbours'
    /// incoming buckets.
    #[test]
    fn edge_type_index_tracks_delete() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        let et = st.etype_id("KNOWS").unwrap();
        // delete 0-KNOWS->1: gone from 0's out bucket AND 1's in bucket.
        st.delete_edge(0, 1, 0);
        assert_eq!(out_ids(&st, 0, "KNOWS"), Vec::<u32>::new());
        let in1: Vec<u32> = st.in_typed(1, et).iter().map(|a| a.nbr).collect();
        assert_eq!(in1, Vec::<u32>::new());
        // delete node 2: removes 0-LIKES->2 and 1-KNOWS->2 mirrors.
        st.delete_node(2);
        assert_eq!(out_ids(&st, 0, "LIKES"), Vec::<u32>::new());
        assert_eq!(out_ids(&st, 1, "KNOWS"), Vec::<u32>::new());
    }

    /// Transaction rollback restores the index exactly (a per-node rebuild off the
    /// restored flat adjacency, so no delta bookkeeping can drift).
    #[test]
    fn edge_type_index_survives_rollback() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        st.begin();
        st.add_edge(0, 2, "KNOWS"); // 0 KNOWS {1,2} inside the txn
        st.delete_edge(1, 2, 2); // 1 KNOWS {} inside the txn
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1, 2]);
        st.rollback();
        // Back to the committed shape: 0 KNOWS {1}, 1 KNOWS {2}.
        assert_eq!(out_ids(&st, 0, "KNOWS"), vec![1]);
        assert_eq!(out_ids(&st, 1, "KNOWS"), vec![2]);
    }

    /// A node added AFTER the index exists grows the index and indexes its edges.
    #[test]
    fn edge_type_index_grows_with_new_node() {
        let mut st = typed_graph();
        st.create_edge_type_index();
        let three = st.add_node(&["V"], &[]);
        st.add_edge(three, 0, "LIKES");
        assert_eq!(out_ids(&st, three, "LIKES"), vec![0]);
    }

    // --- opt-in edge interval index (G4) ---

    /// One Emp node (0) with `degree` HELD edges to role node 1, edge d carrying
    /// interval `[d, d+width]`.
    fn interval_graph(degree: u32, width: i64) -> Store {
        let mut b = Builder::default();
        b.node(&["Emp"], &[]);
        b.node(&["Role"], &[]);
        let mut st = b.build();
        for d in 0..degree {
            let eid = st.add_edge(0, 1, "HELD");
            st.set_edge_prop(eid, "vf", n(f64::from(d)));
            st.set_edge_prop(eid, "vt", n((i64::from(d) + width) as f64));
        }
        st
    }

    /// Overlap eids from the index (sorted), vs a brute-force scan of the flat
    /// adjacency reading the boxed props.
    fn overlap_eids(st: &Store, node: u32, qlo: f64, qhi: f64) -> Vec<u32> {
        let mut v = Vec::new();
        st.for_each_overlap(node, qlo, qhi, |eid, _| v.push(eid));
        v.sort_unstable();
        v
    }
    fn overlap_bruteforce(st: &Store, node: u32, qlo: f64, qhi: f64) -> Vec<u32> {
        let mut v: Vec<u32> = st
            .out(node)
            .iter()
            .filter(|a| {
                matches!((st.edge_prop(a.eid, "vf"), st.edge_prop(a.eid, "vt")),
                    (Value::Num(lo), Value::Num(hi)) if lo <= qhi && hi >= qlo)
            })
            .map(|a| a.eid)
            .collect();
        v.sort_unstable();
        v
    }

    /// The seek agrees with a brute-force overlap scan for point queries across the
    /// timeline AND for wider interval queries (both seed axes exercised).
    #[test]
    fn interval_seek_matches_bruteforce() {
        let st = {
            let mut s = interval_graph(64, 4);
            s.create_interval_index("vf", "vt");
            s
        };
        assert!(st.has_interval_index("vf", "vt"));
        // as-of points across the whole timeline (0..=67), incl. the ends where one
        // axis is far more selective than the other.
        for t in 0..=67 {
            let q = f64::from(t);
            assert_eq!(
                overlap_eids(&st, 0, q, q),
                overlap_bruteforce(&st, 0, q, q),
                "point t={t}"
            );
        }
        // wider ranges
        for &(lo, hi) in &[(10.0, 20.0), (0.0, 100.0), (63.0, 63.0), (-5.0, 2.0)] {
            assert_eq!(
                overlap_eids(&st, 0, lo, hi),
                overlap_bruteforce(&st, 0, lo, hi),
                "range [{lo},{hi}]"
            );
        }
    }

    /// Writes keep the interval index current: a new edge+interval appears, a
    /// changed interval moves, a deleted edge vanishes.
    #[test]
    fn interval_index_tracks_writes() {
        let mut st = interval_graph(4, 2); // edges: [0,2],[1,3],[2,4],[3,5]
        st.create_interval_index("vf", "vt");
        // as-of t=10 → none.
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
        // add an edge covering t=10.
        let e = st.add_edge(0, 1, "HELD");
        st.set_edge_prop(e, "vf", n(8.0));
        st.set_edge_prop(e, "vt", n(12.0));
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), vec![e]);
        // move it off t=10.
        st.set_edge_prop(e, "vt", n(9.0));
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
        // delete it.
        st.set_edge_prop(e, "vt", n(12.0));
        st.delete_edge(0, 1, e);
        assert_eq!(overlap_eids(&st, 0, 10.0, 10.0), Vec::<u32>::new());
    }

    /// Rollback restores the interval index exactly (a full rebuild against the
    /// restored graph, so prop AND adjacency undo ordering can't drift it).
    #[test]
    fn interval_index_survives_rollback() {
        let mut st = interval_graph(4, 2);
        st.create_interval_index("vf", "vt");
        let before: Vec<Vec<u32>> = (0..8)
            .map(|t| overlap_eids(&st, 0, f64::from(t), f64::from(t)))
            .collect();
        st.begin();
        let e = st.add_edge(0, 1, "HELD");
        st.set_edge_prop(e, "vf", n(0.0));
        st.set_edge_prop(e, "vt", n(100.0)); // covers everything, inside the txn
        st.delete_edge(0, 1, 0); // and drop the first committed edge
        assert!(overlap_eids(&st, 0, 3.0, 3.0).contains(&e));
        st.rollback();
        let after: Vec<Vec<u32>> = (0..8)
            .map(|t| overlap_eids(&st, 0, f64::from(t), f64::from(t)))
            .collect();
        assert_eq!(before, after);
    }
}
